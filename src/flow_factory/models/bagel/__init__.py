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

# src/flow_factory/models/bagel/__init__.py
"""
Bagel Model Adapter

Integrates ByteDance's Bagel multimodal model into Flow-Factory.
Supports Text-to-Image and Image(s)-to-Image generation tasks.
"""

from .bagel import BagelAdapter, BagelI2ISample, BagelSample
from .pipeline import BagelPseudoPipeline

__all__ = [
    "BagelAdapter",
    "BagelSample",
    "BagelI2ISample",
    "BagelPseudoPipeline",
]
