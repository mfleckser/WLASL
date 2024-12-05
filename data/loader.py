import os
import json
import random
import numpy as np

from data.start_kit.video_downloader import download_video
from data.start_kit.preprocess import video_to_frames


class Loader:
    total_samples = 21083

    def __init__(self, batch_size=64, mode="all", size=(400, 400), base_path="."):
        self.batch_size = batch_size
        self.num_batches = Loader.total_samples // batch_size
        self.mode = mode
        self.size = size
        self.base_path = base_path

    def load(self):
        for _ in range(self.num_batches):
            yield Loader.fetch_batch(self.batch_size, self.mode, self.size, self.base_path)

    @staticmethod
    def format_label(label: str) -> str:
        """Replace spaces with underscores and remove single quotes"""
        return label.replace(" ", "_").replace("'", "")

    @staticmethod
    def fetch_batch(batch_size, mode, size, base_path):
        """
        Returns list of (data, label) tuples. Data is (n x w x h x c)
        - batch_size: number of data points to yield. default: 100
        - mode: one of: ("train", "test", "val", "all"). Which subset of data to fetch from.
                default: "all"
        - width/height: size to rescale images to. default: (400x400)"""
        data_info = json.load(open(os.path.join(base_path, "start_kit", "WLASL_v0.3.json"), "r"))
        classes = [Loader.format_label(c["gloss"]) for c in data_info]

        x = []
        y = []
        while len(x) < batch_size:
            gloss = random.choice(data_info)
            label = Loader.format_label(gloss["gloss"])

            video_info = random.choice(gloss["instances"])
            if video_info["split"] != mode and mode != "all":
                continue

            video_data = Loader.get_video_data(video_info, size)
                
            if video_data:
                x.append(video_data)
                y.append(classes.index(label))

        return (np.array(x), np.array(y))

    @staticmethod
    def get_video_data(video_info, size=None):
        video_path = download_video(
            video_info["url"], "data/temp_vids", video_info["video_id"]
        )

        if video_path:
            video_data = np.stack(video_to_frames(video_path, size, video_info["bbox"]))
            os.remove(video_path)

            start = video_info["frame_start"] - 1
            end = video_info["frame_end"]

            if end == -1:
                video_data = video_data[start:,:,:,:]
            else:
                video_data = video_data[start:end,:,:,:]
            
            return video_data
        
        return None
