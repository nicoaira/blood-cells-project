"""Load the WBC subtype checkpoint and classify cropped, single-cell RGB images.

    python scripts/wbc_classifier.py path/to/wbc_crop.png [--device cuda] [--weights models/...pt]

The input must contain one WBC, such as a padded crop of a WBC detected by the YOLO model. This classifier
does not locate cells in a whole microscope picture and does not have an unknown/non-WBC output class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

WEIGHTS_NAME = "resnet18-bloodmnist-wbc.pt"
DEFAULT_WEIGHTS = Path(__file__).with_name(WEIGHTS_NAME)
if not DEFAULT_WEIGHTS.is_file():
    DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / WEIGHTS_NAME
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_model(classes: int, pretrained: bool = False):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, classes)
    return model


def image_transform(size: int):
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class WBCClassifier:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a CUDA-enabled PyTorch build or select --device cpu.")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != "resnet18":
            raise ValueError("Unsupported WBC classifier architecture")
        self.names = checkpoint["classes"]
        self.device = torch.device(device)
        self.transform = image_transform(checkpoint["input_size"])
        self.model = make_model(len(self.names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device, memory_format=torch.channels_last).eval()
        self.metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}

    @torch.inference_mode()
    def predict(self, images: list[Image.Image | np.ndarray]) -> list[dict]:
        if not images:
            return []
        tensors = [self.transform((im if isinstance(im, Image.Image) else Image.fromarray(im)).convert("RGB"))
                   for im in images]
        batch = torch.stack(tensors).to(self.device, memory_format=torch.channels_last)
        probabilities = self.model(batch).softmax(dim=1).cpu().numpy()
        return [{"subtype": self.names[int(row.argmax())], "confidence": float(row.max()),
                 "probabilities": {name: float(p) for name, p in zip(self.names, row)}} for row in probabilities]

    def classify_detections(self, image_rgb: np.ndarray, boxes: np.ndarray, wbc_class: int = 0,
                            padding: float = 0.1, batch_size: int = 32) -> list[list[float] | None]:
        """Aligned subtype probabilities for WBC boxes; other cell classes receive None.

        Use square crops with 10% context on each side. Crop coordinates are clamped to the image; portions
        outside the photo are padded with a pale background instead of wrapping around NumPy's negative indices.
        """
        h, w = image_rgb.shape[:2]
        output = [None] * len(boxes)
        crops, indices = [], []
        for i, box in enumerate(boxes):
            if int(box[5]) != wbc_class:
                continue
            x1, y1, x2, y2 = map(float, box[:4])
            if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                continue
            side = max(1, int(np.ceil(max(x2 - x1, y2 - y1) * (1 + 2 * padding))))
            left = int(np.floor((x1 + x2 - side) / 2))
            top = int(np.floor((y1 + y2 - side) / 2))
            l, t, r, b = max(0, left), max(0, top), min(w, left + side), min(h, top + side)
            if l >= r or t >= b:
                continue
            crop = Image.new("RGB", (side, side), (245, 245, 245))
            crop.paste(Image.fromarray(np.ascontiguousarray(image_rgb[t:b, l:r])), (l - left, t - top))
            crops.append(crop); indices.append(i)
        for start in range(0, len(crops), batch_size):
            predictions = self.predict(crops[start:start + batch_size])
            for i, pred in zip(indices[start:start + batch_size], predictions):
                output[i] = [pred["probabilities"][name] for name in self.names]
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="+", type=Path, help="cropped single-WBC images")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = WBCClassifier(args.weights, args.device)
    images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in args.images]
    for path, result in zip(args.images, model.predict(images)):
        print(json.dumps({"image": str(path), **result}))


if __name__ == "__main__":
    main()
