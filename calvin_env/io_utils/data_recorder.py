import logging
import multiprocessing as mp
import os
import pickle
import threading
import time

log = logging.getLogger(__name__)


class TextToSpeech:
    """Non-blocking TTS; prints if pyttsx3 is unavailable."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.queue = None
        self.process = None
        if not enabled:
            return
        try:
            import pyttsx3  # noqa: F401
        except ModuleNotFoundError:
            log.warning("pyttsx3 not installed, TTS will print to stdout")
            return
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._tts_worker, name="TTS_worker", daemon=True)
        self.process.start()

    def say(self, text):
        log.info(text)
        if self.queue is not None:
            self.queue.put(text)

    def _tts_worker(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            text = self.queue.get()
            if text == "QUIT":
                break
            engine.say(text)
            engine.runAndWait()

    def close(self):
        if self.queue is not None:
            self.queue.put("QUIT")


class DataRecorder:
    """
    Save serialized simulation frames while teleoperating with VR.

    Button 1 release starts/stops an episode. Holding button 1 deletes the last
    episode. Files are written as zero-padded pickles in the hydra run directory,
    which is the format expected by calvin_env.datarenderer.
    """

    def __init__(self, env, record_fps, enable_tts):
        self.env = env
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.process_queue, name="MultiprocessingStorageWorker")
        self.process.start()
        self.running = True
        self.save_frame_cnt = 0
        log.info("Starting serialization worker process")
        self.prev_time = time.time()
        self.loop_time = 1.0 / record_fps
        self.tts = TextToSpeech(enabled=enable_tts)
        self.current_episode_filenames = []
        self.recording = False
        self.delete_thread = None
        self.dead_man_switch_was_down = False

    def step(self, action, state_obs, record_info, info, vr_events=()):
        """
        Returns:
            True if an episode just finished and the env should be reset.
        """
        if record_info is None or "dead_man_switch_triggered" not in record_info:
            return False

        if record_info["dead_man_switch_triggered"]:
            self.dead_man_switch_was_down = True

        episode_done = False
        if record_info["trigger_release"] and not self.recording and not self.is_deleting:
            if not self.dead_man_switch_was_down:
                self.tts.say("please press the dead man switch once before starting to record")
            else:
                self.recording = True
                self.tts.say("start recording")
                self.current_episode_filenames = []
        elif record_info["trigger_release"] and self.recording:
            self.recording = False
            self.save(action, state_obs, True, info, vr_events)
            self.tts.say("finish recording")
            episode_done = True

        if record_info["hold_event"]:
            if self.recording:
                self.recording = False
            self.delete_last_episode()

        if self.recording:
            assert action is not None, "Cannot record a frame without a VR action"
            current_time = time.time()
            delta_t = current_time - self.prev_time
            if delta_t >= self.loop_time:
                log.debug(f"Record FPS: {1 / delta_t:.0f}")
                self.prev_time = current_time
                self.save(action, state_obs, False, info, vr_events)

        return episode_done

    @property
    def is_deleting(self):
        return self.delete_thread is not None and self.delete_thread.is_alive()

    def delete_last_episode(self):
        self.delete_thread = threading.Thread(target=self._delete_last_episode, daemon=True)
        self.delete_thread.start()

    def delete_episode(self):
        """Backward-compatible alias."""
        self.delete_last_episode()

    def _delete_last_episode(self):
        log.info("Delete episode")
        while not self.queue.empty():
            log.info("Wait until files are saved")
            time.sleep(0.01)
        num_frames = len(self.current_episode_filenames)
        self.tts.say(f"Deleting last episode with {num_frames} frames")
        for filename in self.current_episode_filenames:
            if os.path.exists(filename):
                os.remove(filename)
        self.tts.say("Finished deleting")
        self.save_frame_cnt -= num_frames
        self.current_episode_filenames = []

    def save(self, action, state_obs, done, info, vr_events=()):
        filename = f"{str(self.save_frame_cnt).zfill(12)}.pickle"
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        data = self.env.serialize()
        data["vr_events"] = list(vr_events) if vr_events is not None else []
        data["state_obs"] = state_obs
        data["done"] = done
        data["info"] = info
        data["action"] = action
        self.queue.put((filename, data))

    def process_queue(self):
        while True:
            msg = self.queue.get()
            if msg == "QUIT":
                self.running = False
                break
            filename, data = msg
            with open(filename, "wb") as file:
                pickle.dump(data, file)

    def close(self):
        if self.running:
            self.queue.put("QUIT")
            self.process.join()
        self.tts.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
