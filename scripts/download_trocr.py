"""Download the public quantized TrOCR ONNX checkpoint into the project."""
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data" / "models" / "trocr-onnx"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Xenova/trocr-small-handwritten",
        local_dir=str(destination),
        allow_patterns=["*.json", "*.model", "onnx/*quantized.onnx"],
    )
    print(f"TrOCR checkpoint ready: {destination}")


if __name__ == "__main__":
    main()
