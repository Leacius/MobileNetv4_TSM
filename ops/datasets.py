import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
from decord import cpu
from decord import VideoReader

class BaseballDataset(Dataset):
    def __init__(self, data_path, labels_file, new_length=8, num_segments=8, frame_interval=1, transform=None):
        self.data_path = Path(data_path)
        self.labels = pd.read_csv(labels_file, header=None, sep=" ", names=["video", "length", "label"]).set_index("video")
        self.video_names = self.labels.index.to_list()
        self.new_length = new_length
        self.num_segments = num_segments
        self.frame_interval = frame_interval
        self.transform = transform

    def _sample_indices(self, total_frames):
        total_segment_span = self.frame_interval * (self.new_length - 1) + 1
        average_duration = (total_frames - total_segment_span + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                      np.random.randint(average_duration, size=self.num_segments)
        elif total_frames > self.num_segments * total_segment_span:
            max_offset = total_frames - total_segment_span
            offsets = np.sort(np.random.randint(max_offset, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,), dtype=int)
        return offsets

    def _extract_frames(self, video_path, total_frames):
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            print(f"[ERROR] Failed to open video: {video_path} | {e}")
            return [Image.new("RGB", (224, 224)) for _ in range(self.new_length * self.num_segments)]

        indices = []
        for offset in self._sample_indices(total_frames):
            indices += [offset + i * self.frame_interval for i in range(self.new_length)]
        indices = [min(i, total_frames - 1) for i in indices]

        frames = []
        for idx in indices:
            idx = int(min(idx, total_frames - 1))  # 強制轉 int
            try:
                img = vr[idx].asnumpy()  # shape: (H, W, C)
                img = Image.fromarray(img)
                frames.append(img)
            except Exception as e:
                print(f"[WARN] Failed to load frame {idx}: {e}")
                frames.append(Image.new("RGB", (224, 224)))
        return frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        item = self.labels.iloc[idx]
        total_frames = item.length
        video_path = self.data_path / f"{video_name}.mp4"
        frames = self._extract_frames(video_path, total_frames)

        if self.transform:
            frames = self.transform(frames)
        # print(f"[DEBUG] label raw value = {item.label}, type = {type(item.label)}")
        return frames, item.label #torch.tensor(item.label, dtype=torch.long)
