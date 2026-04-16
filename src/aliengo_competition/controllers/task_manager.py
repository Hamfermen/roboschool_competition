from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String

from .cube_database import CubeDatabase


@dataclass
class GoalTrack:
    label: str
    cube_pose: PoseStamped
    stand_pose: PoseStamped


class TaskManager(Node):
    STATE_EXPLORE = "EXPLORE"
    STATE_GOTO_CUBE = "GOTO_CUBE"
    STATE_STAND_AND_CONFIRM = "STAND_AND_CONFIRM"
    STATE_NEXT = "NEXT"
    STATE_DONE = "DONE"

    def __init__(self, cube_db: CubeDatabase) -> None:
        super().__init__("task_manager")
        self.cube_db = cube_db
        self.declare_parameter("target_order", [])
        self.declare_parameter("stand_distance_m", 0.8)
        self.declare_parameter("explore_timeout_s", 15.0)

        self.target_order: List[str] = [
            str(x) for x in self.get_parameter("target_order").value
        ]
        self.current_index = 0
        self.state = self.STATE_EXPLORE
        self.current_track: Optional[GoalTrack] = None
        self.state_deadline = self.get_clock().now()
        self.finished = False
        self.pending_goal = False
        self.last_goal_reached = False
        self.latest_map: Optional[OccupancyGrid] = None
        self.desired_twist = Twist()
        self.stand_distance = float(self.get_parameter("stand_distance_m").value)
        self.explore_timeout = float(self.get_parameter("explore_timeout_s").value)

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.status_pub = self.create_publisher(String, "/task_status", 10)
        self.create_subscription(String, "/target_order", self._target_order_cb, 10)
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, 10)

        self.get_logger().info("TaskManager initialized.")

    def _target_order_cb(self, msg: String) -> None:
        raw = msg.data.strip()
        if not raw:
            return
        parsed = [token.strip() for token in raw.split(",") if token.strip()]
        if parsed:
            self.target_order = parsed
            self.current_index = 0
            self.state = self.STATE_EXPLORE
            self.finished = False
            self.get_logger().info("Received target order: %s" % str(self.target_order))

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def update(self) -> None:
        if self.finished:
            self._set_desired_twist(0.0, 0.0, 0.0)
            return
        if not self.target_order:
            self._publish_status("WAIT_TARGET_ORDER")
            return

        if self.current_index >= len(self.target_order):
            self.finished = True
            self.state = self.STATE_DONE
            self._set_desired_twist(0.0, 0.0, 0.0)
            self._publish_status("DONE")
            self.get_logger().info("All target cubes visited.")
            return

        if self.state == self.STATE_EXPLORE:
            self._step_explore()
        elif self.state == self.STATE_GOTO_CUBE:
            self._step_goto_cube()
        elif self.state == self.STATE_STAND_AND_CONFIRM:
            self._step_stand_and_confirm()
        elif self.state == self.STATE_NEXT:
            self.current_index += 1
            self.current_track = None
            self.state = self.STATE_EXPLORE
            self._publish_status("NEXT_TARGET")

    def _step_explore(self) -> None:
        label = self.target_order[self.current_index]
        pose = self.cube_db.get_cube_pose(label)
        if pose is not None:
            stand_pose = self._compute_stand_pose(pose)
            self.current_track = GoalTrack(
                label=label, cube_pose=pose, stand_pose=stand_pose
            )
            self.state = self.STATE_GOTO_CUBE
            self.pending_goal = False
            self.get_logger().info(
                "Cube found for label=%s, going to stand pose." % label
            )
            return

        if not self.pending_goal:
            goal = self._random_explore_goal()
            if goal is not None:
                self._send_nav_goal(goal)
                self.pending_goal = True
                self.state_deadline = self.get_clock().now() + Duration(
                    seconds=self.explore_timeout
                )
                self._publish_status("EXPLORE_SENT_GOAL")
            else:
                self._publish_status("EXPLORE_NO_MAP")

        if self.get_clock().now() > self.state_deadline and self.pending_goal:
            self.pending_goal = False
            self._publish_status("EXPLORE_RETRY")

    def _step_goto_cube(self) -> None:
        if self.current_track is None:
            self.state = self.STATE_EXPLORE
            return
        if not self.pending_goal:
            self._send_nav_goal(self.current_track.stand_pose)
            self.pending_goal = True
            self.last_goal_reached = False
            self._publish_status("GOTO_CUBE_SENT")
            return
        if self.last_goal_reached:
            self.pending_goal = False
            self.state = self.STATE_STAND_AND_CONFIRM
            self.state_deadline = self.get_clock().now() + Duration(seconds=3.0)
            self._publish_status("STAND_AND_CONFIRM")

    def _step_stand_and_confirm(self) -> None:
        self._set_desired_twist(0.0, 0.0, 0.0)
        if self.get_clock().now() < self.state_deadline:
            return
        if self.current_track is None:
            self.state = self.STATE_EXPLORE
            return
        found_again = self.cube_db.get_cube_pose(self.current_track.label)
        if found_again is None:
            self.get_logger().info("Cube not re-detected, returning to exploration.")
            self.state = self.STATE_EXPLORE
            return
        self.cube_db.mark_visited(
            self.current_track.label, self.current_track.cube_pose
        )
        self.state = self.STATE_NEXT
        self._publish_status("VISITED_%s" % self.current_track.label)

    def _random_explore_goal(self) -> Optional[PoseStamped]:
        if self.latest_map is None:
            return None
        info = self.latest_map.info
        data = self.latest_map.data
        width = info.width
        height = info.height
        if width <= 0 or height <= 0:
            return None

        free_indices = [idx for idx, occ in enumerate(data) if occ == 0]
        if not free_indices:
            return None
        idx = random.choice(free_indices)
        x_cell = idx % width
        y_cell = idx // width
        x = info.origin.position.x + (x_cell + 0.5) * info.resolution
        y = info.origin.position.y + (y_cell + 0.5) * info.resolution
        yaw = random.uniform(-math.pi, math.pi)
        return self._pose_xy_yaw(x, y, yaw, "map")

    def _compute_stand_pose(self, cube_pose: PoseStamped) -> PoseStamped:
        # Stand in front of the cube at fixed distance and face the cube.
        x = cube_pose.pose.position.x
        y = cube_pose.pose.position.y
        yaw_to_cube = math.atan2(y, x)
        stand_x = x - self.stand_distance * math.cos(yaw_to_cube)
        stand_y = y - self.stand_distance * math.sin(yaw_to_cube)
        return self._pose_xy_yaw(
            stand_x, stand_y, yaw_to_cube, cube_pose.header.frame_id or "map"
        )

    def _pose_xy_yaw(
        self, x: float, y: float, yaw: float, frame_id: str
    ) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = frame_id
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.position.z = 0.0
        p.pose.orientation.z = math.sin(yaw * 0.5)
        p.pose.orientation.w = math.cos(yaw * 0.5)
        return p

    def _send_nav_goal(self, pose: PoseStamped) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warning("navigate_to_pose action server not available.")
            self.pending_goal = False
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("Navigation goal rejected.")
            self.pending_goal = False
            self.last_goal_reached = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future) -> None:
        _ = future
        self.last_goal_reached = True
        self.pending_goal = False
        self.get_logger().info("Navigation goal completed.")

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = "%s:%s:%d/%d" % (
            self.state,
            text,
            self.current_index,
            len(self.target_order),
        )
        self.status_pub.publish(msg)

    def _set_desired_twist(self, vx: float, vy: float, wz: float) -> None:
        self.desired_twist.linear.x = float(vx)
        self.desired_twist.linear.y = float(vy)
        self.desired_twist.angular.z = float(wz)

    def get_desired_twist(self) -> Twist:
        if self.state in (self.STATE_STAND_AND_CONFIRM, self.STATE_DONE):
            self._set_desired_twist(0.0, 0.0, 0.0)
        return self.desired_twist

    def is_finished(self) -> bool:
        return self.finished
