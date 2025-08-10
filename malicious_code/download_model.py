import modal
import os
import argparse


def download_model(local_path: str, src_path: str) -> None:
    volume = modal.Volume.lookup("qwen3-data", create_if_missing=False)
    os.makedirs(local_path, exist_ok=True)
    volume.download_dir(src_path, local_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("local_path", help="Local path to save model")
    parser.add_argument(
        "--src", default="/data/output/merged_model", help="Modal Volume source path"
    )
    args = parser.parse_args()
    download_model(args.local_path, args.src)
