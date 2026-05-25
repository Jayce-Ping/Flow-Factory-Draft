# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/geneval.py
"""
GenEval Reward Model

Evaluates text-to-image generation by detecting whether specified objects are
present in generated images with correct counts, colors, and spatial positions.

Uses Mask2Former (via mmdetection) for object detection and CLIP (via open_clip)
for color classification.

Dependencies (heavy, optional):
    - mmcv 1.x (compiled with CUDA ops)
    - mmdetection 2.x
    - open_clip_torch
    - clip_benchmark

See `scripts/install_geneval_deps.sh` for installation instructions.

Dataset Format:
    Each sample's `extra_kwargs` must contain a "geneval_metadata" key with the
    evaluation specification. Example:

        {
            "tag": "color_attr",
            "include": [
                {"class": "giraffe", "count": 1, "color": "brown"},
                {"class": "stop sign", "count": 1, "color": "white"}
            ],
            "prompt": "a photo of a brown giraffe and a white stop sign"
        }

    Supported tags: single_object, two_object, counting, colors, position, color_attr

Config Example (YAML):
    rewards:
      - name: "geneval"
        reward_model: "geneval"
        batch_size: 32
        device: "cuda"
        dtype: bfloat16
        # extra kwargs:
        ckpt_path: "/path/to/reward_ckpts"  # Directory containing mask2former checkpoint
        object_names_path: null  # Path to object_names.txt (auto-detected if null)
        threshold: 0.3  # Detection confidence threshold
        counting_threshold: 0.9  # Higher threshold for counting tasks
        max_objects: 16  # Max detections per class
        nms_threshold: 1.0  # NMS IoU threshold (1.0 = disabled)
        position_threshold: 0.1  # Spatial relation tolerance
        reward_type: "score"  # Options: "score" (continuous), "strict" (binary)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.imports import _is_package_available
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# ======================== Dependency Checks ========================

_GENEVAL_DEPS_AVAILABLE = None
_GENEVAL_IMPORT_ERROR = None


def _check_geneval_deps() -> bool:
    """Check if GenEval dependencies are available."""
    global _GENEVAL_DEPS_AVAILABLE, _GENEVAL_IMPORT_ERROR
    if _GENEVAL_DEPS_AVAILABLE is not None:
        return _GENEVAL_DEPS_AVAILABLE

    required = [
        ("mmdet", "mmdet"),
        ("mmcv", "mmcv"),
        ("open_clip", "open_clip_torch"),
        ("clip_benchmark", "clip_benchmark"),
    ]
    missing = [display for pkg, display in required if not _is_package_available(pkg)]

    if missing:
        _GENEVAL_DEPS_AVAILABLE = False
        _GENEVAL_IMPORT_ERROR = (
            f"GenEval reward model requires the following packages: {', '.join(missing)}.\n"
            "Install them with:\n"
            "    bash scripts/install_geneval_deps.sh\n"
            "Or see the GenEval section in docker/README.md for Docker-based setup."
        )
        return False

    _GENEVAL_DEPS_AVAILABLE = True
    return True


def _require_geneval_deps():
    """Raise ImportError if GenEval dependencies are missing."""
    if not _check_geneval_deps():
        raise ImportError(_GENEVAL_IMPORT_ERROR)


# ======================== Default Object Names ========================

# COCO 80-class names used by Mask2Former
COCO_OBJECT_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]


# ======================== GenEval Reward Model ========================

class GenEvalRewardModel(PointwiseRewardModel):
    """
    GenEval reward model for evaluating text-to-image compositional generation.

    Detects objects in generated images using Mask2Former and evaluates whether
    the generation matches the specified compositional requirements (object count,
    color, spatial position).

    Requires:
        - mmcv 1.x (with CUDA ops)
        - mmdetection 2.x
        - open_clip_torch
        - clip_benchmark
        - Mask2Former COCO checkpoint

    See scripts/install_geneval_deps.sh for setup.
    """

    required_fields = ("prompt", "image")
    use_tensor_inputs = False  # Expects PIL Images

    # Mask2Former model name
    _DETECTOR_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99"
    _CONFIG_RELPATH = "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        _require_geneval_deps()

        import mmdet
        from mmdet.apis import init_detector
        import open_clip
        from clip_benchmark.metrics import zeroshot_classification as zsc

        # Suppress tqdm in clip_benchmark
        zsc.tqdm = lambda it, *args, **kwargs: it

        # Configuration from extra_kwargs
        self.ckpt_path = config.extra_kwargs.get("ckpt_path", None)
        self.object_names_path = config.extra_kwargs.get("object_names_path", None)
        self.threshold = config.extra_kwargs.get("threshold", 0.3)
        self.counting_threshold = config.extra_kwargs.get("counting_threshold", 0.9)
        self.max_objects = config.extra_kwargs.get("max_objects", 16)
        self.nms_threshold = config.extra_kwargs.get("nms_threshold", 1.0)
        self.position_threshold = config.extra_kwargs.get("position_threshold", 0.1)
        self.reward_type = config.extra_kwargs.get("reward_type", "score")

        assert self.reward_type in ("score", "strict"), (
            f"reward_type must be 'score' or 'strict', got '{self.reward_type}'"
        )

        # Resolve checkpoint path
        if self.ckpt_path is None:
            raise ValueError(
                "GenEval reward requires 'ckpt_path' in extra_kwargs pointing to "
                "the directory containing the Mask2Former checkpoint. Example:\n"
                "  rewards:\n"
                "    - name: geneval\n"
                "      reward_model: geneval\n"
                "      ckpt_path: /path/to/reward_ckpts"
            )

        # Load object detector (Mask2Former)
        mmdet_file = mmdet.__file__
        assert mmdet_file is not None, "mmdet.__file__ is None"
        mmdet_package_dir = os.path.dirname(os.path.dirname(mmdet_file))
        config_path = os.path.join(mmdet_package_dir, self._CONFIG_RELPATH)
        if not os.path.exists(config_path):
            # Fallback: try mmdetection source directory
            config_path = os.path.join(
                os.path.dirname(mmdet_package_dir), "mmdetection", self._CONFIG_RELPATH
            )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Mask2Former config not found at expected paths. "
                f"Ensure mmdetection is installed with configs available. "
                f"Tried: {config_path}"
            )

        ckpt_file = os.path.join(self.ckpt_path, f"{self._DETECTOR_NAME}.pth")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                f"Mask2Former checkpoint not found: {ckpt_file}\n"
                f"Download it with:\n"
                f"  wget -P {self.ckpt_path} "
                f"https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
                f"mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
                f"{self._DETECTOR_NAME}.pth"
            )

        logger.info(f"Loading Mask2Former detector from {ckpt_file}")
        self.object_detector = init_detector(config_path, ckpt_file, device=str(self.device))

        # Load CLIP model for color classification
        clip_arch = "ViT-L-14"
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", device=str(self.device)
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_arch)

        # Load object class names
        if self.object_names_path and os.path.exists(self.object_names_path):
            with open(self.object_names_path) as f:
                self.classnames = [line.strip() for line in f]
        else:
            self.classnames = COCO_OBJECT_NAMES

        # Color classifier cache
        self._color_classifiers: Dict[str, Any] = {}

        # Store references for use in inner functions
        self._zsc = zsc

        logger.info(
            f"GenEval reward model initialized: "
            f"threshold={self.threshold}, counting_threshold={self.counting_threshold}, "
            f"reward_type={self.reward_type}"
        )

    # ======================== Core Detection ========================

    def _detect_objects(
        self, images: List[Image.Image], metadatas: List[Dict]
    ) -> List[Dict[str, List]]:
        """Run object detection on a batch of images."""
        from mmdet.apis import inference_detector

        np_images = [np.array(img.convert("RGB")) for img in images]
        results = inference_detector(self.object_detector, np_images)

        all_detected = []
        for result, metadata in zip(results, metadatas):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None

            # Determine confidence threshold based on task
            confidence_threshold = (
                self.counting_threshold
                if metadata.get("tag") == "counting"
                else self.threshold
            )

            detected = {}
            for index, classname in enumerate(self.classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
                ordering = ordering[: self.max_objects].tolist()

                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append(
                        (
                            bbox[index][max_obj],
                            None if segm is None else segm[index][max_obj],
                        )
                    )
                    # NMS filtering
                    ordering = [
                        obj
                        for obj in ordering
                        if self.nms_threshold == 1.0
                        or self._compute_iou(bbox[index][max_obj], bbox[index][obj])
                        < self.nms_threshold
                    ]

                if not detected[classname]:
                    del detected[classname]

            all_detected.append(detected)

        return all_detected

    # ======================== Color Classification ========================

    def _color_classification(
        self, image: Image.Image, bboxes: List, classname: str
    ) -> List[str]:
        """Classify colors of detected objects using CLIP zero-shot."""
        from clip_benchmark.metrics import zeroshot_classification as zsc

        if classname not in self._color_classifiers:
            self._color_classifiers[classname] = zsc.zero_shot_classifier(
                self.clip_model,
                self.clip_tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                str(self.device),
            )

        clf = self._color_classifiers[classname]
        dataset = _ImageCrops(image, bboxes, self.clip_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, num_workers=0
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(
                self.clip_model, clf, dataloader, str(self.device)
            )
            return [COLORS[index.item()] for index in pred.argmax(1)]

    # ======================== Evaluation Logic ========================

    def _evaluate_reward(
        self, image: Image.Image, objects: Dict, metadata: Dict
    ) -> Tuple[bool, float]:
        """
        Evaluate a single image against metadata specifications.
        Returns (is_strict_correct, continuous_score).
        """
        correct = True
        rewards = []
        matched_groups = []

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])

            # Count accuracy reward
            rewards.append(
                max(0.0, 1 - abs(req["count"] - len(found_objects)) / req["count"])
            )

            if len(found_objects) != req["count"]:
                correct = matched = False
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = self._color_classification(image, found_objects, classname)
                    rewards.append(
                        max(
                            0.0,
                            1
                            - abs(req["count"] - colors.count(req["color"]))
                            / req["count"],
                        )
                    )
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False

                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if (
                        target_group >= len(matched_groups)
                        or matched_groups[target_group] is None
                    ):
                        correct = matched = False
                        rewards.append(0.0)
                    else:
                        position_correct = True
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = position_correct = False
                                    break
                            if not position_correct:
                                break
                        rewards.append(1.0 if position_correct else 0.0)

            matched_groups.append(found_objects if matched else None)

        score = sum(rewards) / len(rewards) if rewards else 0.0
        return correct, score

    # ======================== Geometry Helpers ========================

    @staticmethod
    def _compute_iou(box_a, box_b) -> float:
        """Compute IoU between two bounding boxes."""
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn(
            [
                max(box_a[0], box_b[0]),
                max(box_a[1], box_b[1]),
                min(box_a[2], box_b[2]),
                min(box_a[3], box_b[3]),
            ]
        )
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def _relative_position(self, obj_a, obj_b) -> set:
        """Compute relative position of obj_a with respect to obj_b."""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]

        offset = center_a - center_b
        revised_offset = (
            np.maximum(
                np.abs(offset) - self.position_threshold * (dim_a + dim_b), 0
            )
            * np.sign(offset)
        )

        if np.all(np.abs(revised_offset) < 1e-3):
            return set()

        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    # ======================== Main Interface ========================

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        geneval_metadata: Optional[List[Dict]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute GenEval rewards for generated images.

        Args:
            prompt: List of text prompts.
            image: List of generated images (PIL).
            video: List of videos (uses first frame).
            geneval_metadata: List of metadata dicts specifying evaluation criteria.
                Each dict should contain:
                    - "tag": str (single_object, two_object, counting, colors, position, color_attr)
                    - "include": List[Dict] with keys "class", "count", optional "color", "position"
                    - "prompt": str (the prompt used)
                Optional: if not provided, will look in kwargs["extra_kwargs"]

        Returns:
            RewardModelOutput with continuous scores (0-1) or binary strict rewards.
        """
        # Handle video input (use first frame)
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided for GenEval")

        batch_size = len(prompt)
        assert len(image) == batch_size, (
            f"Mismatch: {batch_size} prompts vs {len(image)} images"
        )

        # Resolve metadata
        if geneval_metadata is None:
            # Try to extract from extra_kwargs passed through kwargs
            geneval_metadata = kwargs.get("geneval_metadata", None)

        if geneval_metadata is None:
            raise ValueError(
                "GenEval reward requires 'geneval_metadata' per sample. "
                "Ensure your dataset JSONL includes a 'geneval_metadata' field with "
                "object specifications (tag, include, exclude, prompt)."
            )

        # Parse JSON strings if metadata is stored as string (Arrow serialization workaround)
        parsed_metadata = []
        for meta in geneval_metadata:
            if isinstance(meta, str):
                parsed_metadata.append(json.loads(meta))
            elif isinstance(meta, dict):
                parsed_metadata.append(meta)
            else:
                raise ValueError(
                    f"geneval_metadata entries must be dict or JSON string, got {type(meta)}"
                )
        geneval_metadata = parsed_metadata

        assert len(geneval_metadata) == batch_size, (
            f"Mismatch: {batch_size} prompts vs {len(geneval_metadata)} metadata entries"
        )

        # Ensure images are PIL
        pil_images = []
        for img in image:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                # Should not happen since use_tensor_inputs=False, but handle gracefully
                pil_images.append(Image.fromarray(np.array(img)))

        # Run object detection
        all_detected = self._detect_objects(pil_images, geneval_metadata)

        # Evaluate each image
        scores = []
        strict_rewards = []
        for img, detected, metadata in zip(pil_images, all_detected, geneval_metadata):
            img = ImageOps.exif_transpose(img)
            is_correct, score = self._evaluate_reward(img, detected, metadata)
            scores.append(score)
            strict_rewards.append(1.0 if is_correct else 0.0)

        # Return based on reward_type
        if self.reward_type == "strict":
            rewards_tensor = torch.tensor(strict_rewards, dtype=torch.float32)
        else:
            rewards_tensor = torch.tensor(scores, dtype=torch.float32)

        return RewardModelOutput(
            rewards=rewards_tensor,
            extra_info={
                "geneval_scores": scores,
                "geneval_strict": strict_rewards,
            },
        )


# ======================== Helper Dataset for Color Classification ========================


class _ImageCrops(torch.utils.data.Dataset):
    """Dataset of cropped object regions for CLIP color classification."""

    def __init__(self, image: Image.Image, objects: List, transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999")
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                index,
                self._image.size[::-1],
                mask.shape,
            )
            image = Image.composite(
                self._image, self._blank, Image.fromarray(mask)
            )
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self._transform(image), 0)
