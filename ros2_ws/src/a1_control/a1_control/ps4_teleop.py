#!/usr/bin/env python3
import time
import math
import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import MotorsState, MotorState


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def expo(x, e):
    # e=1.0: 線形
    # e>1.0: 中心付近を鈍く（細かく操作しやすい）
    # e<1.0: 中心付近を鋭く
    s = 1.0 if x >= 0.0 else -1.0
    return s * (abs(x) ** e)


class PS4Teleop(Node):
    """
    PS4 Joy -> /controller/cmd_vel (Twist)
    - Twist.linear.x  : 速度指令（前進+: m/s 相当）
    - Twist.angular.z : ステア指令（-1〜+1 の無次元）※ヨーレートではない
    """

    def __init__(self):
        super().__init__("ps4_teleop")

        # ===== パラメータ（操作感）=====
        # 左スティックX
        self.axis_steer = self.declare_parameter("axis_steer", 0).value
        # R2（前進用トリガ）
        self.axis_r2    = self.declare_parameter("axis_r2", 5).value
        # L2（後進用トリガ）
        self.axis_l2    = self.declare_parameter("axis_l2", 2).value

        self.deadzone   = float(self.declare_parameter("deadzone", 0.05).value)

        # 最大速度 [m/s] 相当（下流で rps に変換）
        self.max_linear = float(self.declare_parameter("max_linear", 0.6).value)

        # 入力カーブ（中立付近の操作感）
        self.steer_expo    = float(self.declare_parameter("steer_expo", 1.3).value)
        self.throttle_expo = float(self.declare_parameter("throttle_expo", 1.6).value)

        # レート制限（急な変化を抑える）
        self.linear_accel_limit = float(self.declare_parameter("linear_accel_limit", 0.8).value)  # [m/s^2]
        self.steer_rate_limit   = float(self.declare_parameter("steer_rate_limit", 8.0).value)    # [1/s]

        # 通信途絶時の安全停止
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.5).value)

        # 非常停止（joy_node の PS4 ボタン番号: 0=×, 1=○, 2=△, 3=□, 9=OPTIONS）
        # ×で停止をラッチし、トリガを離した状態で OPTIONS を押すと解除
        self.estop_button = int(self.declare_parameter("estop_button", 0).value)
        self.estop_release_button = int(self.declare_parameter("estop_release_button", 9).value)
        # 非常停止中に速度 0 を直接送るモータ ID（cmdvel_to_motor を経由せず、車体の全モータを止める）
        self.estop_motor_ids = [int(i) for i in self.declare_parameter("estop_motor_ids", [1, 2, 3, 4]).value]

        # ===== 内部状態 =====
        self.last_joy_time = time.monotonic()
        self.joy_axes = []
        self.joy_buttons = []

        self.v_cmd = 0.0          # 出力する linear.x
        self.steer_cmd = 0.0      # 出力する angular.z（-1..1）
        self.estopped = False     # 非常停止中（ラッチ）

        self.last_update = time.monotonic()

        # ===== ROS I/O =====
        self.sub = self.create_subscription(Joy, "/joy", self.joy_cb, 10)
        self.pub = self.create_publisher(Twist, "/controller/cmd_vel", 10)
        self.pub_motor = self.create_publisher(MotorsState, "/ros_robot_controller/set_motor", 10)

        # 50Hz で出力更新
        self.timer = self.create_timer(0.02, self.update)

        self.get_logger().info(
            "PS4Teleop: 左スティックX=ステアリング操作, R2=前進, L2=後退, ×=非常停止, OPTIONS=解除"
        )

    def button(self, index):
        return len(self.joy_buttons) > index and self.joy_buttons[index] == 1

    def joy_cb(self, msg: Joy):
        self.last_joy_time = time.monotonic()
        self.joy_axes = list(msg.axes)
        self.joy_buttons = list(msg.buttons)
        # 非常停止は timer を待たず、ボタンを受け取った時点で即座にかける
        if not self.estopped and self.button(self.estop_button):
            self.engage_estop()

    def engage_estop(self):
        self.estopped = True
        self.v_cmd = 0.0
        self.steer_cmd = 0.0
        self.publish_motor_stop()
        self.pub.publish(Twist())
        self.get_logger().warn("非常停止: 全モータ停止。OPTIONS で解除（トリガを離した状態で）")

    def publish_motor_stop(self):
        # 車体の全モータに速度 0 を直接送る
        msg = MotorsState()
        msg.data = [MotorState(id=i, rps=0.0) for i in self.estop_motor_ids]
        self.pub_motor.publish(msg)

    def update(self):
        now = time.monotonic()
        dt = now - self.last_update
        if dt <= 0.0:
            dt = 1e-3
        self.last_update = now

        # ===== 非常停止 =====
        if not self.estopped and self.button(self.estop_button):
            self.engage_estop()
        elif self.estopped and self.button(self.estop_release_button):
            if self.triggers_released():
                self.estopped = False
                self.get_logger().info("非常停止を解除")

        if self.estopped:
            # 解除されるまで 50Hz で全モータ 0 を送り続ける（パケットの取りこぼしがあっても次の周期で止まる）
            self.v_cmd = 0.0
            self.steer_cmd = 0.0
            self.publish_motor_stop()
            self.pub.publish(Twist())
            return

        # Joy が来ていない場合は安全停止
        if (now - self.last_joy_time) > self.cmd_timeout:
            target_v = 0.0
            target_steer = 0.0
        else:
            # ===== 生値取得 =====
            steer_raw = (
                self.joy_axes[self.axis_steer]
                if len(self.joy_axes) > self.axis_steer
                else 0.0
            )
            r2_raw = (
                self.joy_axes[self.axis_r2]
                if len(self.joy_axes) > self.axis_r2
                else 1.0
            )
            l2_raw = (
                self.joy_axes[self.axis_l2]
                if len(self.joy_axes) > self.axis_l2
                else 1.0
            )

            # デッドゾーン（ステア）
            if abs(steer_raw) < self.deadzone:
                steer_raw = 0.0

            # 軸値は [-1, +1] にクリップ
            r2_raw = clamp(r2_raw, -1.0, 1.0)
            l2_raw = clamp(l2_raw, -1.0, 1.0)

            # ===== トリガ正規化 =====
            # PS4 のトリガは「離していると 1.0」「全押しで -1.0」
            #   ->  (1 - axis) / 2  で  [1 → 0, -1 → 1] になる
            throttle_fwd = (1.0 - r2_raw) / 2.0   # R2: 前進
            throttle_back = (1.0 - l2_raw) / 2.0  # L2: 後退

            # わずかなノイズをゼロに丸める
            if throttle_fwd < 0.02:
                throttle_fwd = 0.0
            if throttle_back < 0.02:
                throttle_back = 0.0

            # 前後の合成
            #   正: 前進, 負: 後退
            throttle = throttle_fwd - throttle_back
            throttle = clamp(throttle, -1.0, 1.0)

            # ===== ステア（左右反転）=====
            # ここでマイナスをかけて左右を反転
            steer = expo(clamp(-steer_raw, -1.0, 1.0), self.steer_expo)

            # Expo をスロットルにも適用
            throttle = expo(throttle, self.throttle_expo)

            # 最終的な目標速度
            target_v = self.max_linear * throttle
            target_steer = steer  # -1..+1

        # ===== レート制限 =====
        # v: 加速度制限
        dv_max = self.linear_accel_limit * dt
        self.v_cmd += clamp(target_v - self.v_cmd, -dv_max, dv_max)

        # steer: 変化速度制限
        ds_max = self.steer_rate_limit * dt
        self.steer_cmd += clamp(target_steer - self.steer_cmd, -ds_max, ds_max)

        # ===== publish =====
        t = Twist()
        t.linear.x = float(self.v_cmd)
        t.angular.z = float(self.steer_cmd)
        self.pub.publish(t)


    def triggers_released(self):
        # 解除直後に急発進しないよう、R2/L2 が両方離されている（軸値がほぼ 1.0）ことを確認
        for axis in (self.axis_r2, self.axis_l2):
            if len(self.joy_axes) > axis and self.joy_axes[axis] < 0.9:
                return False
        return True


def main():
    rclpy.init()
    node = PS4Teleop()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl+C では SIGINT が 2 回届くため、終了処理中は以降の SIGINT を無視
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

