from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class CubeObservation:
    label: str
    xyz_map: np.ndarray


class CubeDetector(Node):
    def __init__(self) -> None:
        super().__init__("cube_detector")
        self.declare_parameter("model_path", "yolov8m.pt")
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("cluster_threshold_m", 0.3)
        self.declare_parameter("target_frame", "map")
        self.declare_parameter(
            "allowed_labels",
            ["chair", "bottle", "cup", "laptop", "backpack"],
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        default_qos = QoSProfile(depth=10)

        self.bridge = CvBridge()
        self.model = YOLO(str(self.get_parameter("model_path").value))
        self.confidence = float(self.get_parameter("confidence").value)
        self.cluster_threshold_m = float(
            self.get_parameter("cluster_threshold_m").value
        )
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.allowed_labels = {
            str(x) for x in self.get_parameter("allowed_labels").value
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_info: Optional[CameraInfo] = None
        self.latest_depth_msg: Optional[Image] = None
        self.unique_cubes: Dict[str, List[np.ndarray]] = {}

        self.marker_pub = self.create_publisher(
            MarkerArray, "/detected_cubes", default_qos
        )
        self.create_subscription(
            CameraInfo, "/camera/rgb/camera_info", self._camera_info_cb, image_qos
        )
        self.create_subscription(
            Image, "/camera/depth/image_raw", self._depth_cb, image_qos
        )
        self.create_subscription(
            Image, "/camera/rgb/image_raw", self._rgb_cb, image_qos
        )

        self.get_logger().info("CubeDetector initialized with YOLOv8m.")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def _rgb_cb(self, msg: Image) -> None:
        if self.camera_info is None or self.latest_depth_msg is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        depth = self._depth_to_meters(self.latest_depth_msg)
        if depth is None:
            return

        results = self.model.predict(source=rgb, conf=self.confidence, verbose=False)
        if not results:
            return

        observations: List[CubeObservation] = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            label = self.model.names.get(cls_id, str(cls_id))
            if label not in self.allowed_labels:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            if cy < 0 or cx < 0 or cy >= depth.shape[0] or cx >= depth.shape[1]:
                continue
            z = float(depth[cy, cx])
            if not np.isfinite(z) or z <= 0.1 or z > 8.0:
                continue

            p_cam = self._unproject_pixel(cx, cy, z, self.camera_info)
            p_map = self._transform_point_to_map(
                p_cam=p_cam,
                source_frame=msg.header.frame_id or "camera_link",
                stamp=msg.header.stamp,
            )
            if p_map is None:
                continue
            observations.append(CubeObservation(label=label, xyz_map=p_map))

        if observations:
            self._update_clusters_and_publish(observations)

    def _depth_to_meters(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            return depth.astype(np.float32)
        if msg.encoding == "16UC1":
            depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            return depth_u16.astype(np.float32) / 1000.0
        self.get_logger().debug("Unsupported depth encoding: %s" % msg.encoding)
        return None

    @staticmethod
    def _unproject_pixel(u: int, v: int, z: float, info: CameraInfo) -> np.ndarray:
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    def _transform_point_to_map(
        self, p_cam: np.ndarray, source_frame: str, stamp
    ) -> Optional[np.ndarray]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        rot = self._quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return rot.dot(p_cam) + trans

    @staticmethod
    def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _update_clusters_and_publish(self, observations: List[CubeObservation]) -> None:
        for obs in observations:
            candidates = self.unique_cubes.setdefault(obs.label, [])
            if not candidates:
                candidates.append(obs.xyz_map.copy())
                continue
            min_idx = None
            min_dist = float("inf")
            for idx, p in enumerate(candidates):
                d = float(np.linalg.norm(obs.xyz_map - p))
                if d < min_dist:
                    min_dist = d
                    min_idx = idx
            if min_dist <= self.cluster_threshold_m and min_idx is not None:
                candidates[min_idx] = 0.8 * candidates[min_idx] + 0.2 * obs.xyz_map
            else:
                candidates.append(obs.xyz_map.copy())

        self.marker_pub.publish(self._build_markers())

    def _build_markers(self) -> MarkerArray:
        out = MarkerArray()
        marker_id = 0
        now = self.get_clock().now().to_msg()
        for label, points in self.unique_cubes.items():
            for idx, p in enumerate(points):
                mk = Marker()
                mk.header.frame_id = self.target_frame
                mk.header.stamp = now
                mk.ns = label
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.CUBE
                mk.action = Marker.ADD
                mk.pose = Pose(
                    position=Point(
                        x=float(p[0]), y=float(p[1]), z=float(max(p[2], 0.05))
                    ),
                    quaternion=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
                mk.scale.x = 0.2
                mk.scale.y = 0.2
                mk.scale.z = 0.2
                mk.color.a = 0.85
                mk.color.r = 0.2
                mk.color.g = 0.8
                mk.color.b = 0.2
                mk.text = "%s_%d" % (label, idx)
                out.markers.append(mk)
        return out


def main(args=None):
    try:
        with rclpy.init(args=args):
            cube_detector = CubeDetector()
            rclpy.spin(cube_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
