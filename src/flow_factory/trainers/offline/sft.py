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

"""Offline supervised flow-matching trainer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from ...contracts import OFFLINE_EXECUTION_CONTRACT
from ...data_utils.offline_dataset import DemonstrationOutputBatch, OfflineBatch
from ...data_utils.offline_train_data import build_offline_train_dataloader
from ..abc import BaseTrainer
from ..common.flow_matching import (
    build_noised_output_state,
    flow_matching_per_sample_loss,
    sample_offline_timesteps,
)
from ..common.offline_batch import bind_output_forward_context, move_condition_to_device
from ..forward_process import forward_velocity_state


class SFTTrainer(BaseTrainer):
    """Train a flow-matching policy from finite demonstration datasets."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"
    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def _build_train_dataloader(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Build the finite demonstration loader without Accelerate reshaping it."""
        dataloader = build_offline_train_dataloader(
            self.config,
            self.accelerator,
            self.adapter.preprocess_func,
            supervision_type="demonstration",
            pipeline_io_contract=self.adapter.pipeline_io_contract,
        )
        return dataloader, {}

    def optimize_batch(self, batch: Any) -> None:
        """Apply one gradient-accumulation microstep to a demonstration batch.

        Every target is encoded on demand. Multiple independently sampled time
        terms are averaged before the single backward call, so they do not alter
        dataloader epoch or gradient-accumulation cadence.

        Args:
            batch: One collated :class:`OfflineBatch` with demonstration output.

        Raises:
            TypeError: If the batch or decoded output has the wrong concrete type.
            ValueError: If the batch declares a non-demonstration supervision type.
        """
        output = self._require_demonstration_batch(batch)
        condition = move_condition_to_device(batch.condition, self.accelerator.device)

        # Evaluation leaves the trainable components in eval mode. Each finite
        # optimization microstep reasserts the policy mode before its model pass.
        self.adapter.train()

        encoded = self.adapter.encode_output_state(output.target_media, condition)
        model_batch = bind_output_forward_context(condition, encoded.forward_context)
        batch_size = len(output.target_media)
        all_timesteps = sample_offline_timesteps(
            self.training_args,
            batch_size=batch_size,
            device=self.accelerator.device,
        )

        with self.accumulate_gradients():
            time_losses = []
            for primary_timesteps in all_timesteps:
                times, noised = build_noised_output_state(
                    self.adapter,
                    encoded.clean_state,
                    primary_timesteps,
                    batch=model_batch,
                )
                with self.autocast():
                    predicted_velocity = forward_velocity_state(
                        self,
                        model_batch,
                        noised.state,
                        times,
                        source="SFT policy",
                    )
                time_losses.append(
                    flow_matching_per_sample_loss(
                        self.adapter,
                        predicted_velocity,
                        noised,
                    )
                )

            per_sample_loss = torch.stack(time_losses, dim=0).mean(dim=0)
            loss = per_sample_loss.mean()
            self.accelerator.backward(loss)

            # Mutate the accumulation window only after a successful backward.
            # A failed encode, forward, objective, or backward therefore cannot
            # contaminate the next optimizer update or advance trainer progress.
            loss_info = self._get_loss_info()
            loss_info["loss"].append(loss.detach())
            loss_info["flow_matching_loss"].append(per_sample_loss.mean().detach())
            if self.accelerator.sync_gradients:
                self._loss_info = self._apply_optimizer_step(loss_info)

    def _get_loss_info(self) -> Dict[str, List[torch.Tensor]]:
        """Return the current loss window, including for lightweight test instances."""
        loss_info = getattr(self, "_loss_info", None)
        if loss_info is None:
            loss_info = defaultdict(list)
            self._loss_info = loss_info
        return loss_info

    @staticmethod
    def _require_demonstration_batch(batch: Any) -> DemonstrationOutputBatch:
        """Return the typed demonstration output after strict branch validation."""
        if not isinstance(batch, OfflineBatch):
            raise TypeError(
                "SFTTrainer.optimize_batch expected OfflineBatch, "
                f"received {type(batch).__name__}: {batch!r}"
            )
        if batch.supervision_type != "demonstration":
            raise ValueError(
                "SFTTrainer requires supervision_type='demonstration', "
                f"received {batch.supervision_type!r}"
            )
        if not isinstance(batch.output, DemonstrationOutputBatch):
            raise TypeError(
                "SFTTrainer expected DemonstrationOutputBatch, "
                f"received {type(batch.output).__name__}"
            )
        return batch.output


__all__ = ["SFTTrainer"]
