import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
from typing import Optional, Dict, Any, Callable, List, Protocol, Union
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

class TextEncodeCallable(Protocol):
    def __call__(self, texts: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
        ...

class ImageEncodeCallable(Protocol):
    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs: Any) -> Dict[str, Any]:
        ...

class GeneralDataset(Dataset):
    @staticmethod
    def check_exists(dataset_dir: str, split: str) -> bool:
        jsonl_path = os.path.join(os.path.expanduser(dataset_dir), f"{split}.jsonl")
        return os.path.exists(jsonl_path)

    def __init__(
        self,
        dataset_dir : str,
        split:str="train",
        cache_dir="~/.cache/flow_factory/datasets",
        enable_preprocess=True,
        preprocessing_batch_size=16,
        text_encode_func: Optional[TextEncodeCallable] = None,
        image_encode_func: Optional[ImageEncodeCallable] = None,
        **kwargs
    ):
        super().__init__()
        self.data_root = os.path.expanduser(dataset_dir)
        cache_dir = os.path.expanduser(cache_dir)
        
        self.image_dir = os.path.join(self.data_root, "images")
        jsonl_path = os.path.join(self.data_root, f"{split}.jsonl")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Could not find {jsonl_path}")
    
        raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")
        
        if enable_preprocess:
            self._text_encode_func = text_encode_func
            self._image_encode_func = image_encode_func
            
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            fingerprint = f"dataset_{os.path.basename(self.data_root)}_{split}_cache"
            
            self.processed_dataset = raw_dataset.map(
                self._preprocess_batch,
                batched=True,
                batch_size=preprocessing_batch_size,
                fn_kwargs={"image_dir": self.image_dir},
                remove_columns=raw_dataset.column_names,
                new_fingerprint=fingerprint,
                desc="Pre-processing dataset",
                load_from_cache_file=True,
            )
            
            # Only set format to torch if we are reasonably sure the data is NOT ragged.
            # If your images are variable count, this might print a warning and skip.
            try:
                self.processed_dataset.set_format(type="torch", columns=self.processed_dataset.column_names)
            except Exception as e:
                # This is expected behavior for ragged datasets (variable images per sample)
                pass 
                # logger.warning(f"Kept dataset as Python objects (likely due to ragged tensors): {e}")

        else:
            self._text_encode_func = None
            self._image_encode_func = None
            self.processed_dataset = raw_dataset

    def _preprocess_batch(self, batch: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
        
        prompts = batch["prompt"]
        negative_prompts = batch.get("negative_prompt", None)
        img_paths_list = batch.get("images", [[] for _ in range(len(prompts))])
        
        # 1. Process Text
        assert self._text_encode_func is not None, "Text encode function must be provided to process prompts."
        prompt_args = {'prompt': prompts} if negative_prompts is None else {'prompt': prompts, 'negative_prompt': negative_prompts}
        prompt_res = self._text_encode_func(**prompt_args)
        
        # 2. Process Images
        image_args = {}
        collated_image_res = defaultdict(list)

        for img_paths in img_paths_list:
            images = []
            for img_path in img_paths:
                with Image.open(os.path.join(image_dir, img_path)) as img:
                    images.append(img.convert("RGB"))
            if len(images) > 0:
                assert self._image_encode_func is not None, "Image encode function must be provided to process images."
                encoded_single_sample = self._image_encode_func(images, **image_args)
                for k, v in encoded_single_sample.items():
                    collated_image_res[k].append(v)

        # 3. Merge results
        # Handle 'torch.unbind' to convert Batch-Tensors to List-of-Tensors for Arrow storage
        prompt_res = {
            k: (list(torch.unbind(v)) if isinstance(v, torch.Tensor) else v) 
            for k, v in prompt_res.items()
        }

        # Combine all dictionaries. 
        # Prioritize encoded results over raw metadata if keys collide.
        return {**batch, **prompt_res, **collated_image_res}

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle batching of pre-processed samples.
        """
        if not batch:
            return {}

        collated_batch = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]
            # Value is a Tensor -> Stack or List
            if isinstance(values[0], torch.Tensor):
                try:
                    collated_batch[key] = torch.stack(values)
                except:
                    # Dimensions mismatch (ragged) -> Keep as list
                    collated_batch[key] = values
            else:
                collated_batch[key] = values

        return collated_batch