from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


class CubeDatabase(Node):
    def __init__(self) -> None:
        super().__init__("cube_database")
        self.declare_parameter("cluster_threshold_m", 0.3)
        self.declare_parameter("query_label", "")
        self.cluster_threshold_m = float(
            self.get_parameter("cluster_threshold_m").value
        )

        self.cubes: Dict[str, List[np.ndarray]] = {}
        self.visited: Dict[Tuple[str, int], bool] = {}

        qos = QoSProfile(depth=10)
        self.create_subscription(MarkerArray, "/detected_cubes", self._detected_cb, qos)
        self.known_pub = self.create_publisher(MarkerArray, "/known_cubes", qos)
        self.get_pose_srv = self.create_service(
            Trigger, "get_cube_pose", self._get_cube_pose_cb
        )
        self.pub_timer = self.create_timer(0.5, self._publish_known)

        self.get_logger().info("CubeDatabase ready.")

    def _detected_cb(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            label = marker.ns
            p = np.array(
                [
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z,
                ],
                dtype=np.float32,
            )
            points = self.cubes.setdefault(label, [])
            if not points:
                points.append(p)
                self.visited[(label, 0)] = False
                continue

            min_idx = None
            min_dist = float("inf")
            for idx, q in enumerate(points):
                dist = float(np.linalg.norm(p - q))
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_idx is not None and min_dist <= self.cluster_threshold_m:
                points[min_idx] = 0.7 * points[min_idx] + 0.3 * p
            else:
                points.append(p)
                self.visited[(label, len(points) - 1)] = False

    def _publish_known(self) -> None:
        markers = MarkerArray()
        marker_id = 0
        now = self.get_clock().now().to_msg()
        for label, points in self.cubes.items():
            for idx, p in enumerate(points):
                visited = self.visited.get((label, idx), False)
                mk = Marker()
                mk.header.frame_id = "map"
                mk.header.stamp = now
                mk.ns = "known_" + label
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = float(p[0])
                mk.pose.position.y = float(p[1])
                mk.pose.position.z = float(max(p[2], 0.1))
                mk.pose.orientation.w = 1.0
                mk.scale.x = 0.25
                mk.scale.y = 0.25
                mk.scale.z = 0.25
                mk.color.a = 0.95
                if visited:
                    mk.color.r, mk.color.g, mk.color.b = 0.1, 0.2, 0.9
                else:
                    mk.color.r, mk.color.g, mk.color.b = 0.9, 0.6, 0.2
                markers.markers.append(mk)
        self.known_pub.publish(markers)

    def _get_cube_pose_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        _ = request
        label = str(self.get_parameter("query_label").value)
        pose = self.get_cube_pose(label)
        if pose is None:
            response.success = False
            response.message = "not_found"
            return response
        data = {
            "label": label,
            "frame_id": pose.header.frame_id,
            "x": pose.pose.position.x,
            "y": pose.pose.position.y,
            "z": pose.pose.position.z,
            "visited": False,
        }
        response.success = True
        response.message = json.dumps(data)
        return response

    def get_cube_pose(self, label: str) -> Optional[PoseStamped]:
        points = self.cubes.get(label, [])
        if not points:
            return None
        for idx, p in enumerate(points):
            if not self.visited.get((label, idx), False):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = float(p[0])
                pose.pose.position.y = float(p[1])
                pose.pose.position.z = float(max(p[2], 0.05))
                pose.pose.orientation.w = 1.0
                return pose
        return None

    def mark_visited(self, label: str, pose: PoseStamped) -> None:
        points = self.cubes.get(label, [])
        if not points:
            return
        target = np.array(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=np.float32,
        )
        min_idx = None
        min_dist = float("inf")
        for idx, p in enumerate(points):
            d = float(np.linalg.norm(target - p))
            if d < min_dist:
                min_dist = d
                min_idx = idx
        if min_idx is not None and min_dist <= 0.6:
            self.visited[(label, min_idx)] = True
