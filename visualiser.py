from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.v2.functional import to_pil_image

from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask


class Visualiser:
    """
    Small helper class for generating CAM visualisations with a pretrained ResNet18 model.

    Designed for:
    - default-layer CAMs for G
    - custom layer CAMs for VG
    - notebook-friendly usage without storing lots of mutable state
    """

    def __init__(self, labels_path: str | Path | None = None) -> None:
        self.weights = get_model_weights(
            "resnet18"
        ).DEFAULT  # get pretrained weights for ResNet 18
        self.model = get_model("resnet18", weights=self.weights).eval()
        self.preprocess = self.weights.transforms()

        if labels_path is not None:
            with open(labels_path, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
            self.labels = {int(k): v for k, v in raw_labels.items()}
        else:
            # if no json file, use built-in labels
            categories = self.weights.meta["categories"]
            self.labels = {i: [str(i), label] for i, label in enumerate(categories)}

        self.layers = {
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
        }

    def load_image(self, image_path: str | Path) -> torch.Tensor:
        """Load an image from disk as a tensor."""
        return decode_image(str(image_path))

    def preprocess_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Apply model-specific transforms and add a batch dimension.
        return self.preprocess(image_tensor).unsqueeze(0)

    def get_label(self, class_index: int) -> str:
        """Return human-readable label for a class index."""
        return self.labels[class_index][1]

    def find_class_indices(self, search_term: str) -> list[tuple[int, str]]:
        """Find ImageNet class indices whose labels contain the given search term."""
        search_term = search_term.lower()
        matches: list[tuple[int, str]] = []

        for idx, (_, label) in self.labels.items():
            if search_term in label.lower():
                matches.append((idx, label))

        return matches

    def get_top_predictions(
        self, logits: torch.Tensor, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Convert logits into a list of top-k predictions with labels and probabilities."""
        probs = softmax(logits, dim=1)  # convert logits to probabilities
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=1)

        results: list[dict[str, Any]] = []
        for prob, idx in zip(top_probs[0], top_ids[0]):
            idx_int = idx.item()
            synset, label = self.labels[idx_int]
            results.append(
                {
                    "index": idx_int,
                    "label": label,
                    "synset": synset,
                    "probability": prob.item(),
                }
            )

        return results

    def generate_cam_overlay(
        self,
        image_path: str | Path,
        target_class: int | None = None,
        layer_name: str | None = None,
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """
        Generate a CAM overlay for a given image.

        Parameters:
        - image_path: path to image file
        - target_class: class index to visualise; if None, uses top predicted class
        - layer_name: one of None, 'layer1', 'layer2', 'layer3', 'layer4'
                      If None, LayerCAM uses its default layer.
        - alpha: overlay transparency

        Returns:
        A dictionary containing the image, logits, predictions, chosen class,
        activation map, and overlay image.
        """
        image_tensor = self.load_image(image_path)
        input_tensor = self.preprocess_image(image_tensor)

        target_layer = None if layer_name is None else self.layers[layer_name]

        with LayerCAM(self.model, target_layer=target_layer) as cam_extractor:
            with torch.enable_grad():
                logits = self.model(input_tensor)

                if target_class is None:
                    target_class = logits.argmax(dim=1).item()

                activation_map = cam_extractor(target_class, logits)[0].squeeze(0)

        original_pil = to_pil_image(image_tensor)
        heatmap_pil = to_pil_image(activation_map, mode="F")
        overlay = overlay_mask(original_pil, heatmap_pil, alpha=alpha)

        return {
            "image_path": str(image_path),
            "layer_name": "default" if layer_name is None else layer_name,
            "target_class": target_class,
            "target_label": self.get_label(target_class),
            "image_tensor": image_tensor,
            "logits": logits,
            "top_predictions": self.get_top_predictions(logits),
            "activation_map": activation_map,
            "overlay": overlay,
        }

    def plot_result(
        self,
        result: dict[str, Any],
        title: str | None = None,
        figsize: tuple[int, int] = (6, 6),
    ) -> None:
        """Plot a single CAM overlay result."""
        plt.figure(figsize=figsize)
        plt.imshow(result["overlay"])
        plt.axis("off")

        if title is not None:
            plt.title(title)
        else:
            plt.title(f'{result["target_label"]} ({result["layer_name"]})')

        plt.show()

    def compare_layers(
        self,
        image_path: str | Path,
        target_class: int,
        layer_names: list[str] | None = None,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Generate CAM overlays for multiple layers for the same image/class.
        """
        if layer_names is None:
            layer_names = ["layer1", "layer3", "layer4"]

        results = []
        for layer_name in layer_names:
            result = self.generate_cam_overlay(
                image_path=image_path,
                target_class=target_class,
                layer_name=layer_name,
                alpha=alpha,
            )
            results.append(result)

        return results

    def plot_layer_comparison(
        self,
        results: list[dict[str, Any]],
        suptitle: str | None = None,
    ) -> None:
        """Plot multiple layer results side by side."""
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))

        if len(results) == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            ax.imshow(result["overlay"])
            ax.axis("off")
            ax.set_title(result["layer_name"])

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)

        plt.tight_layout()
        plt.show()
