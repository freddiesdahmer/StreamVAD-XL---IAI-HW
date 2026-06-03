import cv2
import numpy as np
from PIL import Image


class RealtimeVideoStream:
    def __init__(self, source=0, clip_length=16, frame_interval=1):
        self.source = source
        self.clip_length = clip_length
        self.frame_interval = frame_interval
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_buffer = []        # 存原始 PIL Image
        self.display_buffer = []      # 存原始 BGR 帧（用于显示）
        self.running = False

    def start(self):
        self.running = True
        return self

    def read_clip(self):
        """返回 (pil_images, bgr_frames) 供特征提取和显示使用"""
        self.frame_buffer.clear()
        self.display_buffer.clear()

        while len(self.frame_buffer) < self.clip_length and self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            # 保存原始 BGR 帧用于显示
            self.display_buffer.append(frame.copy())

            # 转为 PIL Image 供 CLIP 官方 preprocess 使用
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            self.frame_buffer.append(pil_image)

        if len(self.frame_buffer) == self.clip_length:
            return self.frame_buffer, self.display_buffer
        return None

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
