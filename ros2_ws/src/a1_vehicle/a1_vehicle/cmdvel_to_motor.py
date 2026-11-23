#!/usr/bin/env python3
import math
import signal
import time
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from geometry_msgs.msg import Twist

from ros_robot_controller_msgs.msg import MotorsState, MotorState
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class CmdvelToMotor(Node):
    """
    /controller/cmd_vel -> 後輪DC + 前輪ステア（バスサーボ HTS-25L）

    解釈：
      linear.x  = 速度指令 v [m/s] 相当（teleopで0〜max_linear）
      angular.z = ステア指令 s [-1..+1]（ヨーレートではない）

    モデル（運動学）：
      s -> δ (最大舵角で線形マップ)
      ω = (v/L) * tan(δ)   （v=0ならω=0）
      vl = v - ω*W/2
      vr = v + ω*W/2
      rps = v_wheel / (pi*D) * rps_gain
    """

    def __init__(self):
        super().__init__("cmdvel_to_motor")

        # ===== 車両パラメータ（後で実機に合わせて調整）=====
        self.wheelbase = float(self.declare_parameter("wheelbase", 0.145).value)      # L [m]
        self.track     = float(self.declare_parameter("track", 0.133).value)          # W [m]
        self.wheel_diam = float(self.declare_parameter("wheel_diam", 0.067).value)    # D [m]

        # ステア最大角（Hiwonder 公式の上限 29deg に合わせる）
        self.max_steer_deg = float(self.declare_parameter("max_steer_deg", 29.0).value)

        # ステアリング用バスサーボ（HTS-25L: 位置 0〜1000 = 0〜240deg、1 単位 0.24deg）
        self.steer_servo_id = int(self.declare_parameter("steer_servo_id", 1).value)
        self.steer_center = int(self.declare_parameter("steer_center", 500).value)    # 直進位置（初期値）
        self.steer_delta  = int(self.declare_parameter("steer_delta", 121).value)     # center±delta（29deg / 0.24deg ≒ 121）
        self.steer_min    = int(self.declare_parameter("steer_min", 379).value)       # サーボ本体の角度制限と同じ
        self.steer_max    = int(self.declare_parameter("steer_max", 621).value)
        self.steer_dir    = float(self.declare_parameter("steer_dir", 1.0).value)     # 左右が逆なら -1.0
        self.steer_duration = float(self.declare_parameter("steer_duration", 0.05).value)  # サーボの補間時間 [s]
        # ステアの移動速度の上限 [位置単位/s]（1 単位 = 0.24deg）。
        # HTS-25L には電流・トルク上限の設定がないため、急な動きを抑えて電流ピークを下げる。
        # 停止中（据え切り）はタイヤの摩擦で負荷が大きいので、さらに遅くする
        self.steer_max_rate = float(self.declare_parameter("steer_max_rate", 800.0).value)
        self.steer_max_rate_stationary = float(self.declare_parameter("steer_max_rate_stationary", 400.0).value)
        self.steer_cmd_pos = float(self.steer_center)   # 速度制限をかけた現在の指令位置
        self.steer_cmd_time = None
        self.last_steer_pos = None
        self.last_steer_time = 0.0

        # モータIDと向き（あなたの環境で動いている設定を踏襲）
        self.motor_left_id  = int(self.declare_parameter("motor_left_id", 2).value)
        self.motor_right_id = int(self.declare_parameter("motor_right_id", 4).value)
        self.left_dir  = float(self.declare_parameter("left_dir", 1.0).value)
        self.right_dir = float(self.declare_parameter("right_dir", -1.0).value)

        # rps換算ゲイン（実際の速度に合わない場合ここで合わせる）
        self.rps_gain = float(self.declare_parameter("rps_gain", 1.0).value)

        # デッドゾーン
        self.deadzone_v = float(self.declare_parameter("deadzone_v", 0.01).value)

        # cmd_vel 途絶時のタイムアウト停止 [s]（teleop は 50Hz で送信）
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.3).value)
        self.last_cmd_time = None   # まだ一度も受信していない
        self.timed_out = False

        # ===== ROS I/O =====
        self.sub = self.create_subscription(Twist, "/controller/cmd_vel", self.cb, 10)
        self.pub_motor = self.create_publisher(MotorsState, "/ros_robot_controller/set_motor", 10)
        self.pub_servo = self.create_publisher(ServosPosition, "/ros_robot_controller/bus_servo/set_position", 10)

        # 20Hz でタイムアウト監視（時刻の比較のみ）
        self.create_timer(0.05, self.watchdog)

        # 起動直後にステアを初期位置（steer_center）へ移動
        self.init_timer = self.create_timer(1.0, self.init_steer)

        self.get_logger().info("cmdvel_to_motor: v=linear.x, steer=angular.z(-1..1)")

    def init_steer(self):
        self.init_timer.cancel()
        self.send_steer(self.steer_center, duration=0.5, force=True)
        self.get_logger().info("ステアを初期位置 %d へ移動" % self.steer_center)

    def send_steer(self, pos, duration=None, force=False):
        # 位置が変わったときだけ送る（バスの通信量を抑え、ros_robot_controller の位置読み取りを妨げない）。
        # 取りこぼし対策で 0.5 s ごとに再送
        now = time.monotonic()
        if not force and pos == self.last_steer_pos and now - self.last_steer_time < 0.5:
            return
        msg = ServosPosition()
        msg.duration = float(self.steer_duration if duration is None else duration)
        p = ServoPosition()
        p.id = self.steer_servo_id
        p.position = int(pos)
        msg.position = [p]
        self.pub_servo.publish(msg)
        self.last_steer_pos = pos
        self.last_steer_time = now

    def stop_motors(self):
        # ステアは現在角のまま、後輪のみ停止
        motors = MotorsState()
        for motor_id in (self.motor_left_id, self.motor_right_id):
            m = MotorState()
            m.id = motor_id
            m.rps = 0.0
            motors.data.append(m)
        self.pub_motor.publish(motors)

    def watchdog(self):
        if self.last_cmd_time is None or self.timed_out:
            return
        if time.monotonic() - self.last_cmd_time > self.cmd_timeout:
            self.stop_motors()
            self.timed_out = True
            self.get_logger().warn("cmd_vel が %.2f s 途絶: モータ停止" % self.cmd_timeout)

    def cb(self, msg: Twist):
        self.last_cmd_time = time.monotonic()
        if self.timed_out:
            self.timed_out = False
            self.get_logger().info("cmd_vel 受信再開")

        v = float(msg.linear.x)
        s = float(msg.angular.z)  # -1..+1 (ステア指令)

        if abs(v) < self.deadzone_v:
            v = 0.0

        s = clamp(s, -1.0, 1.0)

        # ===== ステアリングサーボ位置 =====
        target = self.steer_center + self.steer_dir * self.steer_delta * s
        target = clamp(target, self.steer_min, self.steer_max)

        # 移動速度の上限をかける
        now = time.monotonic()
        dt = 0.0 if self.steer_cmd_time is None else min(now - self.steer_cmd_time, 0.1)
        self.steer_cmd_time = now
        rate = self.steer_max_rate_stationary if v == 0.0 else self.steer_max_rate
        step = rate * dt
        self.steer_cmd_pos += clamp(target - self.steer_cmd_pos, -step, step)

        self.send_steer(int(round(self.steer_cmd_pos)))

        # ===== ステア角 δ [rad]（速度制限後の実際のステア指令から計算） =====
        s_eff = (self.steer_cmd_pos - self.steer_center) / (self.steer_dir * self.steer_delta)
        delta = math.radians(self.max_steer_deg) * s_eff

        # ===== ヨーレート ω [rad/s]（運動学） =====
        if abs(v) < 1e-6:
            omega = 0.0
        else:
            omega = (v / self.wheelbase) * math.tan(delta)

        # ===== 左右後輪の車輪速度 [m/s] =====
        vl = v - omega * (self.track / 2.0)
        vr = v + omega * (self.track / 2.0)

        # ===== 車輪速度 -> rps（回転/秒） =====
        # wheel_rps = v / (pi*D)
        if self.wheel_diam > 1e-6:
            rps_l = (vl / (math.pi * self.wheel_diam)) * self.rps_gain
            rps_r = (vr / (math.pi * self.wheel_diam)) * self.rps_gain
        else:
            rps_l = 0.0
            rps_r = 0.0

        motors = MotorsState()
        motors.data = []

        ml = MotorState()
        ml.id = self.motor_left_id
        ml.rps = float(self.left_dir * rps_l)
        motors.data.append(ml)

        mr = MotorState()
        mr.id = self.motor_right_id
        mr.rps = float(self.right_dir * rps_r)
        motors.data.append(mr)

        self.pub_motor.publish(motors)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    # rclpy のシグナルハンドラを無効化し、Ctrl+C 後もコンテキストを有効なままにして停止指令を送れるようにする
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # SIGINT（Ctrl+C / launch 終了）と SIGTERM の両方で finally の停止処理を通す
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = CmdvelToMotor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C では端末と launch から SIGINT が 2 回届くため、停止処理中は以降のシグナルを無視
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # 終了時に後輪を停止
        node.stop_motors()
        time.sleep(0.1)  # 送信完了待ち
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

