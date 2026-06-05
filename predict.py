"""Single-image inference CLI for SAM3 LoRA text-conditioned segmentation."""

import argparse

from main import run_predict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one image/object prediction."""

    parser = argparse.ArgumentParser(description="Predict a text-conditioned SAM3 mask.")
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", default="prediction_mask.npy")
    parser.add_argument("--checkpoint")
    parser.add_argument("--lora-dir")
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    run_predict(parse_args())
