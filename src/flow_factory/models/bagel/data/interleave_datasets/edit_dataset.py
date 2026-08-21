# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random

from PIL import Image, ImageFile, PngImagePlugin

from ..data_utils import pil_img2rgb
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )

        if end_idx - start_idx > 1 and random.random() < 0.5:  # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx - 1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx - 1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=True,
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )
        return data
