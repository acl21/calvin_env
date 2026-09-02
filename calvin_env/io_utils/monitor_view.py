import logging
import time

import cv2
import numpy as np
import pybullet as p

from calvin_env.camera.gripper_camera import GripperCamera

log = logging.getLogger(__name__)

WIN_TABLE = "calvin_env — table"
WIN_GRIPPER = "calvin_env — gripper"
HELP = "WASD pan | Q/E height | Z/C turn | R/F zoom | Shift=fine | V toggle monitor"

_ARROW = {
    81: "left",
    82: "up",
    83: "right",
    84: "down",
    65361: "left",
    65362: "up",
    65363: "right",
    65364: "down",
    2424832: "left",
    2490368: "up",
    2555904: "right",
    2621440: "down",
}


def _key_down(keys, *codes):
    return any(keys.get(int(c), 0) & p.KEY_IS_DOWN for c in codes)


def _key_triggered(keys, *codes):
    return any(keys.get(int(c), 0) & p.KEY_WAS_TRIGGERED for c in codes)


class HeadsetPlayspace:
    """SteamVR tracking origin. WASD/Q/E/Z/C/R/F move the view in the headset."""

    def __init__(self, bullet_client, position=(0.0, -1.2, 1.2), yaw=0.5 * np.pi):
        self.p = bullet_client
        self.pos = np.array(position, dtype=float)
        self.yaw = float(yaw)
        self._last_t = time.time()
        self._apply()

    def _apply(self):
        orn = p.getQuaternionFromEuler([0.0, 0.0, self.yaw])
        try:
            self.p.setVRCameraState(rootPosition=self.pos.tolist(), rootOrientation=orn)
        except Exception as exc:
            log.warning("setVRCameraState failed: %s", exc)

    def apply_held_keys(self, keys):
        now = time.time()
        dt = min(max(now - self._last_t, 0.0), 0.05)
        self._last_t = now
        if dt <= 0 or not keys:
            return

        fine = _key_down(keys, p.B3G_SHIFT)
        step = 0.25 if fine else 1.0
        forward = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        right = np.array([np.sin(self.yaw), -np.cos(self.yaw)])
        moved = False

        if _key_down(keys, ord("w"), ord("W"), p.B3G_UP_ARROW, ord("r"), ord("R")):
            self.pos[:2] += forward * 0.9 * dt * step
            moved = True
        if _key_down(keys, ord("s"), ord("S"), p.B3G_DOWN_ARROW, ord("f"), ord("F")):
            self.pos[:2] -= forward * 0.9 * dt * step
            moved = True
        if _key_down(keys, ord("d"), ord("D"), p.B3G_RIGHT_ARROW):
            self.pos[:2] += right * 0.9 * dt * step
            moved = True
        if _key_down(keys, ord("a"), ord("A"), p.B3G_LEFT_ARROW):
            self.pos[:2] -= right * 0.9 * dt * step
            moved = True
        if _key_down(keys, ord("q"), ord("Q")):
            self.pos[2] = float(np.clip(self.pos[2] + 0.7 * dt * step, 0.0, 2.8))
            moved = True
        if _key_down(keys, ord("e"), ord("E")):
            self.pos[2] = float(np.clip(self.pos[2] - 0.7 * dt * step, 0.0, 2.8))
            moved = True
        if _key_down(keys, ord("z"), ord("Z"), ord("y"), ord("Y")):
            self.yaw += 1.4 * dt * step
            moved = True
        if _key_down(keys, ord("c"), ord("C")):
            self.yaw -= 1.4 * dt * step
            moved = True

        if moved:
            self._apply()


class MonitorView:
    """
    Optional OpenCV third-person view. Press V to toggle it off; WASD then moves
    the headset play-space instead (click the Bullet VR window so it has focus).
    """

    def __init__(
        self,
        env,
        enabled=True,
        width=400,
        height=300,
        fov=50,
        view_fps=12,
        gripper_fps=5,
    ):
        self.env = env
        self.enabled = bool(enabled)
        self.width = int(width)
        self.height = int(height)
        self.fov = fov
        self._view_dt = 1.0 / max(view_fps, 1)
        self._gripper_dt = 1.0 / max(gripper_fps, 1)
        self.look_at = np.array([0.0, 0.0, 0.46], dtype=float)
        self.yaw = 0.5 * np.pi
        self.distance = 1.35
        self.eye_z = 1.15
        self._renderer = None
        self._cam_flags = getattr(p, "ER_NO_SEGMENTATION_MASK", 0)
        self._last_view = 0.0
        self._last_gripper = 0.0
        self._windows_open = False
        self._gripper_img = np.zeros((200, 200, 3), dtype=np.uint8)
        self._gripper_cam = self._try_gripper_cam()
        self.headset = HeadsetPlayspace(env.p)
        if self.enabled:
            self._open_windows()
            self._draw_view(force_gripper=True)
        else:
            log.info("Monitor view off. Click the Bullet VR window, then WASD/Q/E/Z/C/R/F in the headset. V toggles the monitor.")

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self._open_windows()
            self._draw_view(force_gripper=True)
            log.info("Monitor view ON. WASD aims the OpenCV camera. Press V to turn it off.")
        else:
            self._close_windows()
            log.info("Monitor view OFF. Click the Bullet VR window, then WASD moves the headset. Press V to turn the monitor back on.")

    def _try_gripper_cam(self):
        try:
            return GripperCamera(
                fov=75,
                aspect=1,
                nearval=0.01,
                farval=2,
                width=84,
                height=84,
                robot_id=self.env.robot.robot_uid,
                cid=self.env.cid,
                name="gripper",
            )
        except Exception as exc:
            log.warning("Gripper camera unavailable: %s", exc)
            return None

    def _open_windows(self):
        cv2.namedWindow(WIN_TABLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TABLE, (640, 480))
        cv2.namedWindow(WIN_GRIPPER, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_GRIPPER, (200, 200))
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Click this window, then WASD  |  V toggles",
            (12, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.imshow(WIN_TABLE, placeholder)
        cv2.waitKey(1)
        self._windows_open = True
        log.info(
            "Opened monitor windows at %dx%d, ~%.0f Hz. %s",
            self.width,
            self.height,
            1.0 / self._view_dt,
            HELP,
        )

    def _close_windows(self):
        if not self._windows_open:
            return
        cv2.destroyWindow(WIN_TABLE)
        cv2.destroyWindow(WIN_GRIPPER)
        cv2.waitKey(1)
        self._windows_open = False

    def _eye_target(self):
        eye = self.look_at.copy()
        eye[0] -= self.distance * np.cos(self.yaw)
        eye[1] -= self.distance * np.sin(self.yaw)
        eye[2] = self.eye_z
        return eye, self.look_at

    def apply_key(self, key):
        if key is None or key < 0 or key == 255:
            return False

        arrow = _ARROW.get(key)
        ch = key & 0xFF
        fine = 65 <= ch <= 90
        if 65 <= ch <= 90:
            ch += 32
        step = 0.25 if fine else 1.0

        c, s = np.cos(self.yaw), np.sin(self.yaw)
        forward = np.array([c, s])
        right = np.array([s, -c])
        moved = True

        if ch == ord("w") or arrow == "up":
            self.look_at[:2] += forward * 0.05 * step
        elif ch == ord("s") or arrow == "down":
            self.look_at[:2] -= forward * 0.05 * step
        elif ch == ord("d") or arrow == "right":
            self.look_at[:2] += right * 0.05 * step
        elif ch == ord("a") or arrow == "left":
            self.look_at[:2] -= right * 0.05 * step
        elif ch == ord("q"):
            self.eye_z = float(np.clip(self.eye_z + 0.04 * step, 0.35, 2.8))
        elif ch == ord("e"):
            self.eye_z = float(np.clip(self.eye_z - 0.04 * step, 0.35, 2.8))
        elif ch == ord("z") or ch == ord("y"):
            self.yaw += 0.08 * step
        elif ch == ord("c"):
            self.yaw -= 0.08 * step
        elif ch == ord("r"):
            self.distance = float(np.clip(self.distance - 0.06 * step, 0.5, 3.5))
        elif ch == ord("f"):
            self.distance = float(np.clip(self.distance + 0.06 * step, 0.5, 3.5))
        else:
            moved = False

        if moved:
            self._sync_debug_camera()
        return moved

    def _sync_debug_camera(self):
        eye, target = self._eye_target()
        delta = eye - target
        horiz = float(np.linalg.norm(delta[:2]))
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return
        yaw = np.degrees(np.arctan2(delta[0], delta[1]))
        pitch = -np.degrees(np.arctan2(delta[2], horiz))
        try:
            self.env.p.resetDebugVisualizerCamera(dist, yaw, pitch, target.tolist())
        except Exception:
            pass

    def poll(self):
        pkeys = self.env.p.getKeyboardEvents()
        cv_key = cv2.waitKeyEx(1) if self.enabled else -1
        cv_v = self.enabled and cv_key >= 0 and (cv_key & 0xFF) in (ord("v"), ord("V"))
        if cv_v or _key_triggered(pkeys, ord("v"), ord("V")):
            self.toggle()
            return

        if self.enabled:
            moved = self.apply_key(cv_key)
            now = time.time()
            if moved or now - self._last_view >= self._view_dt:
                self._last_view = now
                self._draw_view(force_gripper=False)
        else:
            self.headset.apply_held_keys(pkeys)

    def render(self):
        self.poll()

    def _camera_image(self, width, height, view, proj):
        kwargs = dict(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            shadow=0,
        )
        if self._cam_flags:
            kwargs["flags"] = self._cam_flags
        if self._renderer is None:
            try:
                image = self.env.p.getCameraImage(renderer=p.ER_BULLET_HARDWARE_OPENGL, **kwargs)
                rgb = np.reshape(np.asarray(image[2]), (height, width, 4))
                if rgb.size == height * width * 4:
                    self._renderer = p.ER_BULLET_HARDWARE_OPENGL
                    log.info("Monitor view using hardware OpenGL.")
                    return image
            except Exception:
                pass
            self._renderer = p.ER_TINY_RENDERER
            log.info("Monitor view using TinyRenderer at %d Hz.", round(1.0 / self._view_dt))
        kwargs["renderer"] = self._renderer
        return self.env.p.getCameraImage(**kwargs)

    def _draw_view(self, force_gripper):
        self._sync_debug_camera()
        eye, target = self._eye_target()
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(self.fov, self.width / self.height, 0.01, 10)
        image = self._camera_image(self.width, self.height, view, proj)
        rgb = np.reshape(np.asarray(image[2]), (self.height, self.width, 4))[:, :, :3]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        cv2.putText(bgr, HELP, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow(WIN_TABLE, bgr)

        now = time.time()
        if force_gripper or now - self._last_gripper >= self._gripper_dt:
            self._last_gripper = now
            self._gripper_img = self._gripper_bgr()
        cv2.imshow(WIN_GRIPPER, self._gripper_img)

    def _gripper_bgr(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        if self._gripper_cam is None:
            cv2.putText(blank, "no gripper cam", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return blank
        try:
            rgb, _ = self._gripper_cam.render()
        except Exception as exc:
            log.debug("Gripper camera render failed: %s", exc)
            cv2.putText(blank, "gripper cam failed", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return blank
        return cv2.resize(np.ascontiguousarray(rgb[:, :, ::-1]), (200, 200))
