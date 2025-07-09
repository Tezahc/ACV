import cv2


class VideoRecorder:
    def __init__(self, output_path, fps, frame_size, enabled=True):
        self.enabled = enabled

        if self.enabled:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        else:
            self.writer = None

    def write(self, frame):
        if self.enabled and self.writer is not None:
            self.writer.write(frame)

    def release(self):
        if self.enabled and self.writer is not None:
            self.writer.release()