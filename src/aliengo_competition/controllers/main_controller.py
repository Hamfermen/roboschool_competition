from __future__ import annotations

import math

import numpy as np

from aliengo_competition.common.run_logger import CompetitionRunLogger
from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState


def _unwrap_env_from_robot(robot: AliengoRobotInterface):
    env = getattr(robot, "env", None)
    while env is not None and hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def _infer_control_dt(robot: AliengoRobotInterface, fallback_dt: float = 0.02) -> float:
    env = _unwrap_env_from_robot(robot)
    dt = getattr(env, "dt", None) if env is not None else None
    try:
        dt_value = float(dt)
        if dt_value > 0.0:
            return dt_value
    except (TypeError, ValueError):
        pass
    return float(fallback_dt)


class _CameraRenderer:
    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (Intel RealSense D435-like)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Отрисовка камеры отключена: не удалось импортировать cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active or not isinstance(camera, CameraState):
            return
        image = camera.rgb
        depth = camera.depth
        if image is None or depth is None:
            return

        rgb = np.asarray(image)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (depth_m * (255.0 / self.depth_max_m)).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(
            depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)

        cv2.putText(
            view,
            "RGB",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            f"Depth 0..{self.depth_max_m:.1f}m",
            (rgb.shape[1] + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False


class _ImageSaver:
    def __init__(self, enabled: bool, output_dir: str = "camera_frames"):
        self.enabled = bool(enabled)
        self.output_dir = str(output_dir)
        self._cv2 = None
        self._frame_count = 0
        if not self.enabled:
            return
        try:
            import cv2
            import os
        except Exception as exc:
            print(f"Сохранение изображений отключено: не удалось импортировать модули ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Изображения будут сохраняться в: {os.path.abspath(self.output_dir)}")
        except Exception as exc:
            print(f"Ошибка при создании директории {self.output_dir}: {exc}")
            self.enabled = False

    def save(self, camera: CameraState, step_index: int = None) -> None:
        if not self.enabled or self._cv2 is None or not isinstance(camera, CameraState):
            return
        
        image = camera.rgb
        if image is None:
            return

        try:
            import os
            rgb = np.asarray(image)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                return
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = rgb[..., :3]
            # Конвертируем RGB в BGR для cv2.imwrite
            rgb_bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
            
            # Генерируем имя файла
            if step_index is not None:
                filename = f"frame_{step_index:06d}.png"
            else:
                filename = f"frame_{self._frame_count:06d}.png"
                self._frame_count += 1
            
            filepath = os.path.join(self.output_dir, filename)
            success = self._cv2.imwrite(filepath, rgb_bgr)
            if not success:
                print(f"[ImageSaver] Не удалось сохранить изображение: {filepath}")
        except Exception as exc:
            print(f"[ImageSaver] Ошибка при сохранении изображения: {exc}")

    def save_depth(self, camera: CameraState, step_index: int = None) -> None:
        """Сохранить карту глубины в виде изображения."""
        if not self.enabled or self._cv2 is None or not isinstance(camera, CameraState):
            return
        
        depth = camera.depth
        if depth is None:
            return

        try:
            import os
            depth_m = np.asarray(depth, dtype=np.float32)
            if depth_m.ndim != 2:
                return
            
            # Нормализуем и конвертируем в 8-bit для сохранения
            depth_max_m = 4.0
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=depth_max_m, neginf=0.0)
            depth_m = np.clip(depth_m, 0.0, depth_max_m)
            depth_u8 = (depth_m * (255.0 / depth_max_m)).astype(np.uint8)
            
            # Применяем цветовую карту для лучшей визуализации
            depth_color = self._cv2.applyColorMap(depth_u8, self._cv2.COLORMAP_TURBO)
            
            # Генерируем имя файла
            if step_index is not None:
                filename = f"depth_{step_index:06d}.png"
            else:
                filename = f"depth_{self._frame_count:06d}.png"
            
            filepath = os.path.join(self.output_dir, filename)
            success = self._cv2.imwrite(filepath, depth_color)
            if not success:
                print(f"[ImageSaver] Не удалось сохранить карту глубины: {filepath}")
        except Exception as exc:
            print(f"[ImageSaver] Ошибка при сохранении карты глубины: {exc}")

    def close(self) -> None:
        if self.enabled:
            print(f"[ImageSaver] Сохранено {self._frame_count} изображений")


def run(
    robot: AliengoRobotInterface,
    steps: int = 15000,
    render_camera: bool = False,
    camera_depth_max_m: float = 4.0,
    seed: int = 0,
    save_camera_frames: bool = False,
    camera_frames_dir: str = "camera_frames",
) -> None:
    robot.reset()
    env = getattr(robot, "env", None)
    if env is None:
        raise ValueError(
            "Интерфейс робота должен предоставлять 'env' для обязательного логирования."
        )

    logger = CompetitionRunLogger(env=env, seed=int(seed))
    camera_renderer = _CameraRenderer(
        enabled=render_camera, depth_max_m=camera_depth_max_m
    )
    image_saver = _ImageSaver(
        enabled=save_camera_frames, output_dir=camera_frames_dir
    )
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Контроллер] dt={control_dt:.4f}с, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )
    object_queue = list(getattr(env, "SEQUENCE_OF_OBJECTS", []))
    print(
        f"[Контроллер] отрисовка_камеры={'включена' if camera_renderer.enabled else 'выключена'}"
    )
    print(f"[Контроллер] object_queue={object_queue}")

    # Редактируемые пользователем блоки в этом файле:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    import threading

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from tf2_ros import Buffer, TransformListener

    from .cube_database import CubeDatabase
    from .cube_detector import CubeDetector
    from .task_manager import TaskManager

    warmup_s = 0.0
    ramp_s = 0.0
    trajectory_period_s = 8.0
    forward_speed_mean = 0.0
    forward_speed_amp = 0.0
    lateral_speed_amp = 0.0
    yaw_rate_amp = 0.0
    yaw_rate_damping = 0.0
    ang_vel_scale = 1.0

    if not rclpy.ok():
        rclpy.init(args=None)

    cube_detector_node = CubeDetector()
    cube_database_node = CubeDatabase()
    task_manager_node = TaskManager(cube_database_node)

    main_tf_buffer = Buffer()
    main_tf_listener = TransformListener(main_tf_buffer, task_manager_node)
    _ = main_tf_listener

    ros_executor = MultiThreadedExecutor(num_threads=4)
    ros_executor.add_node(cube_detector_node)
    ros_executor.add_node(cube_database_node)
    ros_executor.add_node(task_manager_node)
    task_update_timer = task_manager_node.create_timer(0.1, task_manager_node.update)
    _ = task_update_timer

    ros_spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    ros_spin_thread.start()
    # ================== USER PARAMETERS END ==================

    segment_start_t = 0.0

    try:
        initial_observation = robot.get_observation()
        initial_camera_payload = robot.get_camera()
        print(
            "[Контроллер] Предпросмотр API:"
            f" observation_type={type(initial_observation).__name__},"
            f" camera_payload={'да' if initial_camera_payload is not None else 'нет'}"
        )
        if initial_camera_payload is None:
            print(
                "[Контроллер] Предупреждение: данные фронтальной камеры недоступны. "
                "Проверьте, что симулятор не запущен в headless-режиме и что включён front_camera_enabled."
            )

        for step_index in range(total_steps):
            state = robot.get_state()

            # Камеру можно брать и из state, и напрямую через robot.get_camera().
            camera_payload = robot.get_camera()
            camera_state = state.camera
            if (camera_state.rgb is None or camera_state.depth is None) and isinstance(
                camera_payload, dict
            ):
                camera_state = CameraState(
                    rgb=camera_payload.get("image"),
                    depth=camera_payload.get("depth"),
                )
            elif (
                camera_state.rgb is None or camera_state.depth is None
            ) and isinstance(camera_payload, CameraState):
                camera_state = camera_payload
            camera_renderer.show(camera_state)
            image_saver.save(camera_state, step_index=step_index)
            omega_z = state.imu.wz / ang_vel_scale

            # ================= USER CONTROL LOGIC START =================
            # Это основной блок для логики участника.
            # Здесь нужно читать измерения, принимать решение и формировать
            # команды движения. Логирование найденного объекта тоже делается
            # отсюда.
            #
            # Формат данных эквивалентен данных:
            # - вход команды: vx, vy, wz
            # - выход состояния: measured_vx, measured_vy, measured_wz
            # - joint_states: joint_names, relative_dof_pos, dof_vel
            # - imu: base_ang_vel, base_lin_acc
            # - camera: camera_data["image"], camera_data["depth"]
            # - порядок объектов: object_queue
            #
            # Ниже приведён обязательный шаблон. Участник должен:
            # 1. реализовать get_found_object_id(...)
            # 2. при обнаружении объекта вернуть его id
            # 3. обязательно вызвать log_found_object(...)
            #
            # Если объект не найден, верните None.
            sim_t = state.sim_time_s

            joint_names = state.joints.name
            relative_dof_pos = state.q
            dof_vel = state.q_dot
            measured_vx = state.vx
            measured_vy = state.vy
            measured_wz = state.wz
            base_ang_vel = state.imu.angular_velocity_xyz
            base_lin_acc = np.zeros(3, dtype=np.float32)
            camera_data = (
                camera_payload
                if isinstance(camera_payload, dict)
                else {
                    "image": camera_state.rgb,
                    "depth": camera_state.depth,
                }
            )

            # Обязательная обвязка для логирования найденного объекта.
            # Использование:
            # - get_found_object_id(...) должен вернуть id найденного объекта
            #   или None, если объект не найден
            # - log_found_object(...) записывает событие в судейский лог
            # - этот шаблон нельзя удалять: участник обязан реализовать его
            #   в своём решении
            #
            # Пример:
            #     detected_object_id = get_found_object_id(...)
            #     if detected_object_id is not None:
            #         log_found_object(detected_object_id)
            def log_found_object(object_id: int) -> None:
                """ОБЯЗАТЕЛЬНО: вызывайте при обнаружении целевого объекта."""
                logger.log_detected_object_at_time(int(object_id), float(sim_t))

            def get_found_object_id(
                current_state,
                current_camera_data,
                current_object_queue,
            ):
                """ОБЯЗАТЕЛЬНО: замените заглушку своей логикой и возвращайте id объекта или None."""
                return None

            detected_object_id = get_found_object_id(
                state,
                camera_data,
                object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)

            policy_state = {
                "joint_names": list(joint_names),
                "relative_dof_pos": np.asarray(relative_dof_pos, dtype=np.float32),
                "dof_vel": np.asarray(dof_vel, dtype=np.float32),
                "measured_vx": float(measured_vx),
                "measured_vy": float(measured_vy),
                "measured_wz": float(measured_wz),
                "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float32),
                "base_lin_acc": np.asarray(base_lin_acc, dtype=np.float32),
                "camera_rgb_available": camera_data.get("image") is not None,
                "camera_depth_available": camera_data.get("depth") is not None,
            }
            _ = policy_state

            desired_twist = task_manager_node.get_desired_twist()
            if task_manager_node.is_finished():
                vx = 0.0
                vy = 0.0
                vw = 0.0
            else:
                vx = float(desired_twist.linear.x)
                vy = float(desired_twist.linear.y)
                vw = float(desired_twist.angular.z)
            # ================== USER CONTROL LOGIC END ==================

            robot.set_speed(vx, vy, vw)
            robot.step()
            logger.log_step(step_index * control_dt)
            robot.get_observation()  # Пример доступа к наблюдению после step().

            if robot.is_fallen():
                robot.stop()
                robot.reset()
                segment_start_t = (step_index + 1) * control_dt
                print("[Контроллер] робот упал -> сброс")
                continue
    finally:
        logger.close()
        camera_renderer.close()
        image_saver.close()
        robot.stop()
