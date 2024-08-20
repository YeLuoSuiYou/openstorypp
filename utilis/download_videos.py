import os
import subprocess
import yt_dlp  # noqa: F401
import cv2
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from datasets import load_dataset
from tqdm.auto import tqdm


class YouTubeKeyframeDownloader:
    def __init__(self, download_dir, max_threads, frames_output_path):
        self.download_dir = download_dir
        self.max_threads = max_threads
        self.frames_output_path = frames_output_path

    def download_clip(self, youtube_id, time_stamps):
        video_link = f"https://www.youtube.com/watch?v={youtube_id}"
        output_filename = f"{youtube_id}.mp4"
        output_path = os.path.join(self.download_dir, output_filename)
        frame_path = os.path.join(self.frames_output_path, youtube_id)

        # Check if the video and frames already exist; if yes, skip downloading
        if os.path.exists(frame_path):
            print(f"Skipping {youtube_id}. Frames already exist.")
            return
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        print(f"Downloading video: {youtube_id}")
        try:
            # Download the video using yt-dlp
            command = [
                "yt-dlp",
                "--verbose",
                "--no-progress",
                "--format",
                "bv*[height<=720]/mp4",
                "--output",
                output_path,
                video_link,
            ]
            subprocess.check_call(command)

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while downloading {video_link}: {str(e)}")
            return
        except Exception as ex:
            print(f"Unknown error occurred while downloading {video_link}: {str(ex)}")
            return

        if not os.path.exists(frame_path):
            os.makedirs(frame_path)

        self.extract_frames(output_path, frame_path, time_stamps, youtube_id)
        os.remove(output_path)

    def parse_timestamp(self, timestamp_str):
        # Split the timestamp string into hours, minutes, seconds, and milliseconds
        parts = timestamp_str.split("-")
        hours, minutes, seconds, milliseconds = map(int, parts)
        # Convert the timestamp to seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds

    def extract_frames(self, video_path, frame_output_path, time_stamps, video_id):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for time_stamp in tqdm(time_stamps):
            time_stamp_second = self.parse_timestamp(time_stamp)
            frame_number = int(time_stamp_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                output_filename = f"{video_id}_keyframe_{time_stamp}.jpg"
                output_filename = os.path.join(frame_output_path, output_filename)
                cv2.imwrite(output_filename, frame)
            else:
                print(f"Failed to extract frame at {time_stamp}s from {video_path}")

        cap.release()

    def download_videos(self, video_stamp_pair):
        with ThreadPoolExecutor(self.max_threads) as executor:
            futures = [
                executor.submit(self.download_clip, video_id, time_stamps)
                for video_id, time_stamps in video_stamp_pair.items()
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download YouTube videos and extract frames"
    )

    parser.add_argument(
        "--download_path",
        type=str,
        help="Path to the download videos",
        default="videos",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        help="Maximum number of threads to use",
        default=2,
    )
    parser.add_argument(
        "--frames_output_path",
        type=str,
        help="Path to the output frames",
        default="frames",
    )
    parser.add_argument(
        "--story_parquet_path",
        type=str,
        help="Path to the story parquet file",
        default="OpenstoryPlusPlus\story",
    )

    args = parser.parse_args()
    download_path = args.download_path
    max_threads = args.max_threads
    frames_output_path = args.frames_output_path
    story_parquet_path = args.story_parquet_path

    downloader = YouTubeKeyframeDownloader(
        download_path,
        max_threads,
        frames_output_path,
    )
    dataset = load_dataset(story_parquet_path, split="train")
    video_stamp_pair = dict()
    for example in tqdm(dataset):
        if example["video_id"] not in video_stamp_pair:
            video_stamp_pair[example["video_id"]] = []
            video_stamp_pair[example["video_id"]].append(example["time_stamp"])
        else:
            video_stamp_pair[example["video_id"]].append(example["time_stamp"])
    downloader.download_videos(video_stamp_pair)
