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

"""Training arguments for all algorithms.

This package is the public API. All imports that previously worked
against the monolithic ``training_args.py`` continue to work unchanged:

    from flow_factory.hparams.training_args import GRPOTrainingArguments
    from flow_factory.hparams.training_args import get_training_args_class
"""
from ._base import EvaluationArguments, TrainingArguments
from ._registry import get_training_args_class, list_registered_training_args
from .grpo import GRPOTrainingArguments
from .nft import NFTTrainingArguments
from .awm import AWMTrainingArguments
from .dpo import DPOTrainingArguments
from .dgpo import DGPOTrainingArguments
from .crd import CRDTrainingArguments

__all__ = [
    "EvaluationArguments",
    "TrainingArguments",
    "GRPOTrainingArguments",
    "NFTTrainingArguments",
    "AWMTrainingArguments",
    "DPOTrainingArguments",
    "DGPOTrainingArguments",
    "CRDTrainingArguments",
    "get_training_args_class",
    "list_registered_training_args",
]
