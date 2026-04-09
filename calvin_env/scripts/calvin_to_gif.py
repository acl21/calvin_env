import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def process_dataset() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="/home/lagandua/data/calvin_debug_dataset_arm", help="CALVIN dataset folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/lagandua/data/temp",
        help="Folder to save processed images",
    )
    parser.add_argument("--desired_key", type=str, default="rgb_static")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for exported GIFs")
    parser.add_argument("--max_frames", type=int, default=300, help="Maximum number of frames to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    key = args.desired_key
    frame_duration_ms = max(1, int(args.max_frames / args.fps))
    output_dir.mkdir(parents=True, exist_ok=True)

    """Process the Calvin dataset and create a smaller version of it."""
    for split in ["training", "validation"]:
        split_path = Path(input_dir) / split
        output_split_path = Path(output_dir) / split
        output_split_path.mkdir(parents=True, exist_ok=True)

        frames = []
        count = 0
        ep_start_end_ids = np.load(split_path / "ep_start_end_ids.npy")
        for start, end in ep_start_end_ids:
            for ind in range(start, end + 1):
                if count >= args.max_frames:
                    break
                npz_file = split_path / f"episode_{ind:07d}.npz"
                data = np.load(npz_file)
                img = data[key]
                frames.append(Image.fromarray(img))
                count += 1
            frames[0].save(
                output_split_path / f"{split}_{start}_{end}_{key}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
            )

if __name__ == "__main__":
    process_dataset()