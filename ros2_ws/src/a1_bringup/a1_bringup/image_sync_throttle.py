#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class ImageSyncThrottle(Node):
    """
    RGB と深度画像を撮影時刻でペアにし、一定周期（既定 10Hz）に間引いて再 publish する。

    - 入力: ~/rgb_in, ~/depth_in（launch で remap）
    - 出力: ~/rgb_out, ~/depth_out（同じペアは同じ header を持つ）
    - カメラは間引き後より速い周期（30fps 設定 → 実測 約25Hz）で動かす前提。
      1 フレーム分保留し、撮影時刻が「10Hz の時刻グリッド」に最も近いペアを選ぶ
      （出力は入力より 1 フレーム分 = 約40ms 遅れる）。平均周期は 10Hz。
    """

    def __init__(self):
        super().__init__("image_sync_throttle")

        self.rate = float(self.declare_parameter("rate", 10.0).value)       # 出力周期 [Hz]
        self.slop = float(self.declare_parameter("sync_slop", 0.02).value)  # RGB-深度の許容時刻差 [s]

        self.period = 1.0 / self.rate
        self.next_due = None   # 次に出力したいグリッド時刻 [s]
        self.prev = None       # 直前に受信したペア (t, rgb, depth)

        self.pub_rgb = self.create_publisher(Image, "~/rgb_out", 10)
        self.pub_depth = self.create_publisher(Image, "~/depth_out", 10)

        # 画像は 1 枚 0.6〜0.9MB で分割送信されるため、best effort だと負荷時に欠落する。
        # ascamera は reliable で publish しているので reliable で受ける
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub_rgb = Subscriber(self, Image, "~/rgb_in", qos_profile=qos)
        self.sub_depth = Subscriber(self, Image, "~/depth_in", qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=10, slop=self.slop)
        self.sync.registerCallback(self.cb)

        # 入力・同期・出力の周期を定期的にログ出力（取りこぼしの確認用）
        self.counts = {"rgb": 0, "depth": 0, "pair": 0, "out": 0}
        self.sub_rgb.registerCallback(lambda m: self.count("rgb"))
        self.sub_depth.registerCallback(lambda m: self.count("depth"))
        self.stats_period = float(self.declare_parameter("stats_period", 10.0).value)
        self.create_timer(self.stats_period, self.log_stats)

        self.get_logger().info(
            "image_sync_throttle: %.1f Hz, slop %.3f s" % (self.rate, self.slop))

    def count(self, key):
        self.counts[key] += 1

    def log_stats(self):
        p = self.stats_period
        c = self.counts
        self.get_logger().info(
            "in rgb %.1f Hz / depth %.1f Hz, pairs %.1f Hz, out %.1f Hz"
            % (c["rgb"] / p, c["depth"] / p, c["pair"] / p, c["out"] / p))
        self.counts = dict.fromkeys(c, 0)

    def cb(self, rgb: Image, depth: Image):
        self.count("pair")
        t = stamp_sec(rgb)
        cur = (t, rgb, depth)
        if self.next_due is None:
            self.next_due = t

        if t >= self.next_due:
            # グリッド時刻をまたいだ: 直前のペアと今のペアのうち、グリッドに近い方を出力
            # （取りこぼし直後に古いペアを選ばないよう、直前のペアはグリッドから半周期以内のときだけ候補にする）
            chosen = cur
            if (self.prev is not None
                    and abs(self.prev[0] - self.next_due) <= self.period / 2
                    and abs(self.prev[0] - self.next_due) < abs(t - self.next_due)):
                chosen = self.prev
            self.pub_rgb.publish(chosen[1])
            self.pub_depth.publish(chosen[2])
            self.count("out")

            self.next_due += self.period
            # カメラ停止などで大きく遅れた場合はグリッドを現在時刻に合わせ直す
            if self.next_due <= t:
                self.next_due = t + self.period
            # 出力済みのペアは次の候補にしない
            self.prev = None if chosen is cur else cur
            return

        self.prev = cur


def main():
    rclpy.init()
    node = ImageSyncThrottle()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C で rclpy が先に終了処理を済ませるため、destroy_node は呼ばない
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
