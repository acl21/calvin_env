import logging
import sys
import time

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from calvin_env.utils.utils import pos_orn_to_matrix, restrict_workspace, z_angle_between

log = logging.getLogger(__name__)

GRIPPER_CLOSING_ACTION = -1
GRIPPER_OPENING_ACTION = 1

DEFAULT_RECORD_INFO = {
    "hold": False,
    "hold_event": False,
    "down": False,
    "dead_man_switch_triggered": False,
    "triggered": False,
    "trigger_release": False,
}


class VrInput:
    """
    HTC Vive / SteamVR teleoperation via Bullet VR.

    Dead-man switch moves the robot. The third-person camera on the monitor is
    aimed with the keyboard (click the OpenCV table window first).
    """

    def __init__(
        self,
        robot,
        workspace_limits,
        button_hold_threshold=60,
        tracking_error_threshold=0.05,
        vr_controller=None,
        visualize_vr_pos=False,
        bullet_client=None,
        calibration_forward=(0.0, 1.0),
        invert_relative_orientation=False,
        monitor=None,
        **kwargs,
    ):
        self.robot = robot
        self.monitor = monitor
        if OmegaConf.is_config(vr_controller):
            vr_controller = OmegaConf.to_container(vr_controller, resolve=True)
        vr_controller = vr_controller or {}
        self.vr_controller_id = vr_controller.get("vr_controller_id", 3)
        self.POSITION = int(vr_controller.get("POSITION", 1))
        self.ORIENTATION = int(vr_controller.get("ORIENTATION", 2))
        self.ANALOG = int(vr_controller.get("ANALOG", 3))
        self.BUTTONS = int(vr_controller.get("BUTTONS", 6))
        self.BUTTON_A = int(vr_controller.get("BUTTON_A", 2))
        self.BUTTON_B = int(vr_controller.get("BUTTON_B", 1))
        self.workspace_limits = None if workspace_limits is None else np.array(workspace_limits)
        self.tracking_error_threshold = tracking_error_threshold
        self.button_hold_threshold = button_hold_threshold
        self.visualize_vr_pos = visualize_vr_pos
        self.calibration_forward = np.array(calibration_forward, dtype=float)[:2]
        self.invert_relative_orientation = bool(invert_relative_orientation)

        self.has_tracking_error = False
        self.button_1_press_counter = 0
        self.prev_action = None
        self.robot_start_pos_offset = None
        self.robot_start_orn_offset = None
        self.out_of_workspace_offset = np.zeros(3)
        self.prev_button_info = dict(DEFAULT_RECORD_INFO)
        self.vr_coord_rotation = np.eye(4)
        self._prev_vr_events = ()
        self.vr_pos_uid = None
        self._logged_vr_layout = False
        self._last_button_log = ()
        self._last_empty_vr_log = 0.0

        self.p = bullet_client if bullet_client is not None else self._connect_shared_memory()
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_VR_PICKING, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)
        if visualize_vr_pos:
            self.vr_pos_uid = self._create_vr_pos_visualization_shape()

        self.calibrate_vr_coord_system()

    def _connect_shared_memory(self):
        bullet = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
        cid = bullet._client
        if cid < 0:
            log.error("Failed to connect to SHARED_MEMORY bullet server.\n Is it running?")
            sys.exit(1)
        log.info(f"Connected to server with id: {cid}")
        return bullet

    def _vr_events(self):
        # Match robot_io: unfiltered events. Filtering by VR_DEVICE_CONTROLLER drops
        # Vive wands that SteamVR reports as GENERIC_TRACKER; allAnalogAxes also
        # shifts the tuple so event[7] is no longer the device type.
        return self.p.getVREvents()

    def _buttons(self, event):
        try:
            return event[self.BUTTONS]
        except (IndexError, TypeError):
            return None

    def _controller_event(self, events):
        if not events:
            now = time.time()
            if now - self._last_empty_vr_log > 50.0:
                log.warning("No VR events yet. Is SteamVR tracking a controller?")
                self._last_empty_vr_log = now
            return None

        if not self._logged_vr_layout:
            sample = events[0]
            device = sample[7] if len(sample) > 7 else "?"
            log.info("VR events: %d device(s), tuple len=%d, field[7]=%s (controller=%s tracker=%s)",
                     len(events), len(sample), device, p.VR_DEVICE_CONTROLLER, p.VR_DEVICE_GENERIC_TRACKER)
            self._logged_vr_layout = True

        gripped = []
        any_button = []
        typed = []
        for event in events:
            buttons = self._buttons(event)
            if buttons is None:
                continue
            down = tuple(i for i, b in enumerate(buttons) if b)
            if down and down != self._last_button_log:
                log.info("VR buttons down: %s (dead-man is index %s, button 1 is index %s)",
                         down, self.BUTTON_A, self.BUTTON_B)
                self._last_button_log = down
                if not (buttons[self.BUTTON_A] & p.VR_BUTTON_IS_DOWN):
                    for cand in (2, 7):
                        if cand in down:
                            log.warning("Dead-man index %s is not pressed; using button %s instead.",
                                        self.BUTTON_A, cand)
                            self.BUTTON_A = cand
                            break
            if buttons[self.BUTTON_A] & p.VR_BUTTON_IS_DOWN:
                gripped.append(event)
            elif down:
                any_button.append(event)
            elif len(event) > 7 and event[7] in (p.VR_DEVICE_CONTROLLER, p.VR_DEVICE_GENERIC_TRACKER):
                typed.append(event)
        if gripped:
            return gripped[0]
        if any_button:
            return any_button[0]
        if typed:
            return typed[0]
        return events[0]

    def _create_vr_pos_visualization_shape(self):
        visual_shape_id = self.p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=0.005)
        return self.p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=[0, 0, 0])

    def _update_monitor(self):
        if self.monitor is not None:
            self.monitor.poll()

    def get_action(self):
        record_info = dict(DEFAULT_RECORD_INFO)
        event = self._controller_event(self._vr_events())
        if event is not None:
            self._prev_vr_events = (event,)
            vr_action = self._vr_event_to_action(event)
            record_info = self._get_record_info(event)

            if not self._dead_mans_switch_down(event):
                self._update_monitor()
                return self.prev_action, record_info
            if self._dead_mans_switch_triggered(event):
                self.has_tracking_error = False
                self.prev_action = None
                self._reset_vr_coord_system(vr_action)
            elif self.has_tracking_error:
                self._update_monitor()
                return self.prev_action, record_info

            if self.robot_start_pos_offset is None:
                self._reset_vr_coord_system(vr_action)

            robot_pos, robot_orn, grip = self._transform_action_vr_to_robot_base(vr_action)
            robot_action = self._to_env_action(robot_pos, robot_orn, grip)
            if self._check_tracking_error(robot_action):
                self.has_tracking_error = True
                log.error(f"Tracking error: {self.prev_action['action'][0]}, {robot_action['action'][0]}")
                self._update_monitor()
                return self.prev_action, record_info

            self.prev_action = robot_action
            self._update_vr_marker(robot_pos, robot_orn)
        self._update_monitor()
        return self.prev_action, record_info

    def get_vr_action(self):
        action, _ = self.get_action()
        return action

    @property
    def prev_vr_events(self):
        return self._prev_vr_events

    def _to_env_action(self, robot_pos, robot_orn, grip):
        return {"type": "cartesian_abs", "action": (robot_pos, robot_orn, int(grip))}

    def _check_tracking_error(self, current_action):
        if self.prev_action is None:
            return False
        prev_pos = self.prev_action["action"][0]
        curr_pos = current_action["action"][0]
        return np.any(np.abs(curr_pos - prev_pos) > self.tracking_error_threshold)

    def _get_record_info(self, event):
        button_1_hold = False
        if self._button_1_down(event):
            self.button_1_press_counter += 1
        else:
            self.button_1_press_counter = 0
        if self.button_1_press_counter >= self.button_hold_threshold:
            button_1_hold = True

        self.prev_button_info = {
            "hold_event": button_1_hold and not self.prev_button_info["hold"],
            "hold": button_1_hold,
            "down": self._button_1_down(event),
            "dead_man_switch_triggered": self._dead_mans_switch_triggered(event),
            "triggered": self._button_1_triggered(event) and self._button_1_down(event),
            "trigger_release": self._button_1_released(event)
            and not self.prev_button_info["hold"]
            and self.prev_button_info["down"],
        }
        return self.prev_button_info

    def _dead_mans_switch_down(self, event):
        return self._button_held(event, self.BUTTON_A, p.VR_BUTTON_IS_DOWN)

    def _dead_mans_switch_triggered(self, event):
        return self._button_held(event, self.BUTTON_A, p.VR_BUTTON_WAS_TRIGGERED)

    def _button_1_down(self, event):
        return self._button_held(event, self.BUTTON_B, p.VR_BUTTON_IS_DOWN)

    def _button_1_triggered(self, event):
        return self._button_held(event, self.BUTTON_B, p.VR_BUTTON_WAS_TRIGGERED)

    def _button_1_released(self, event):
        return self._button_held(event, self.BUTTON_B, p.VR_BUTTON_WAS_RELEASED)

    def _button_held(self, event, button_id, flag):
        buttons = self._buttons(event)
        if buttons is None:
            return False
        try:
            return bool(buttons[button_id] & flag)
        except (IndexError, TypeError):
            return False

    def _vr_pos_rot(self, vr_action):
        """
        Calibrated translation, raw VR orientation.

        Yaw calibration maps controller *motion* onto the table axes. Applying that
        same yaw to orientation conjugates pitch into roll whenever the offset is ~90°.
        """
        pos, orn = vr_action[0], vr_action[1]
        pos_w = self.vr_coord_rotation[:3, :3] @ np.asarray(pos, dtype=float)
        pos_w = pos_w + self.vr_coord_rotation[:3, 3]
        return pos_w, R.from_quat(orn)

    def _reset_vr_coord_system(self, vr_action):
        pos, orn = self.robot.get_tcp_pos_orn()
        vr_pos, R_vr = self._vr_pos_rot(vr_action)
        self.robot_start_pos_offset = pos - vr_pos
        R_tcp = R.from_quat(orn)
        if self.invert_relative_orientation:
            self.robot_start_orn_offset = R_tcp * R_vr
        else:
            self.robot_start_orn_offset = R_vr.inv() * R_tcp
        self.out_of_workspace_offset = np.zeros(3)

    def _transform_action_vr_to_robot_base(self, vr_action):
        vr_pos, R_vr = self._vr_pos_rot(vr_action)
        robot_pos = vr_pos + self.robot_start_pos_offset
        if self.invert_relative_orientation:
            robot_orn = (self.robot_start_orn_offset * R_vr.inv()).as_quat()
        else:
            robot_orn = (R_vr * self.robot_start_orn_offset).as_quat()
        if self.workspace_limits is not None:
            robot_pos = self._enforce_workspace_limits(robot_pos)
        return robot_pos, robot_orn, vr_action[2]

    def _enforce_workspace_limits(self, controller_pos):
        controller_pos = controller_pos - self.out_of_workspace_offset
        clipped_pos = restrict_workspace(self.workspace_limits, controller_pos)
        self.out_of_workspace_offset += controller_pos - clipped_pos
        return clipped_pos

    def _vr_event_to_action(self, event):
        vr_controller_pos = np.array(event[self.POSITION])
        vr_controller_orn = np.array(event[self.ORIENTATION])
        controller_analogue_axis = event[self.ANALOG]
        gripper_action = GRIPPER_CLOSING_ACTION if controller_analogue_axis > 0.1 else GRIPPER_OPENING_ACTION
        return vr_controller_pos, vr_controller_orn, gripper_action

    def _update_vr_marker(self, pos, orn):
        if self.vr_pos_uid is None:
            return
        self.p.resetBasePositionAndOrientation(self.vr_pos_uid, pos, orn)

    def calibrate_vr_coord_system(self):
        start_pose = None
        end_pose = None
        log.info("------------------------------")
        log.info("WASD/Q/E/Z/C/R/F aim the view. V toggles the OpenCV monitor (off = keys move the headset).")
        log.info("Then: dead-man once, move controller FORWARD, press button 1.")
        log.info("------------------------------")
        while True:
            event = self._controller_event(self._vr_events())
            if event is not None:
                vr_action = self._vr_event_to_action(event)
                if self._dead_mans_switch_down(event) and start_pose is None:
                    log.info("Now move the controller forward.")
                    start_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])
                elif self._button_1_down(event) and start_pose is not None and end_pose is None:
                    log.info("Finished setting up VR coordinate system.")
                    end_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])
                if start_pose is not None and end_pose is not None:
                    self._set_vr_coord_transformation(start_pose, end_pose)
                    return
            self._update_monitor()

    def _set_vr_coord_transformation(self, start_pose, end_pose):
        new_x = end_pose[:2, 3] - start_pose[:2, 3]
        norm = np.linalg.norm(new_x)
        if norm < 1e-6:
            log.warning("VR calibration motion too small, leaving identity VR rotation")
            self.vr_coord_rotation = np.eye(4)
            return
        new_x = new_x / norm
        target = self.calibration_forward / np.linalg.norm(self.calibration_forward)
        z_angle = z_angle_between(new_x, target)
        self.vr_coord_rotation = np.eye(4)
        self.vr_coord_rotation[:3, :3] = R.from_euler("z", float(z_angle)).as_matrix()
        log.info(
            "Calibrated VR yaw %.1f deg so controller-forward maps to world (%.2f, %.2f).",
            np.degrees(z_angle),
            target[0],
            target[1],
        )

    def on_env_reset(self):
        self.prev_action = None
        self.robot_start_pos_offset = None
        self.robot_start_orn_offset = None
        self.has_tracking_error = False
        self.out_of_workspace_offset = np.zeros(3)
