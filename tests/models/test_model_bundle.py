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

"""Unit tests for the single-root ModelBundle and its RoutedComponentProxy.

Covers name-dispatch + error paths on `ModelBundle`, parameter registration and
frozen-member behavior (the Wan2.2 "shard both, train one" case), and the
`RoutedComponentProxy`'s call-routing + transparent attribute delegation +
peel-to-inner contract (mirroring `BaseAdapter._unwrap`).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy

# ============================== ModelBundle ==============================


def test_bundle_dispatch_routes_to_named_member():
    lin = nn.Linear(4, 2)
    conv = nn.Conv2d(3, 5, kernel_size=1)
    bundle = ModelBundle({"transformer": lin, "vae": conv})

    x = torch.randn(2, 4)
    assert torch.allclose(bundle("transformer", x), lin(x))

    img = torch.randn(2, 3, 8, 8)
    assert bundle("vae", img).shape == conv(img).shape


def test_bundle_unknown_member_raises():
    bundle = ModelBundle({"transformer": nn.Linear(2, 2)})
    with pytest.raises(KeyError, match="no member"):
        bundle("does_not_exist", torch.randn(1, 2))


def test_bundle_empty_raises():
    with pytest.raises(ValueError, match="at least one member"):
        ModelBundle({})


def test_bundle_non_module_member_raises():
    with pytest.raises(TypeError, match="nn.Module"):
        ModelBundle({"bad": object()})


def test_bundle_registers_all_member_params():
    policy = nn.Linear(4, 2)
    critic = nn.Linear(2, 1)
    bundle = ModelBundle({"policy": policy, "critic": critic})

    bundle_params = set(bundle.parameters())
    assert set(policy.parameters()) <= bundle_params
    assert set(critic.parameters()) <= bundle_params


def test_frozen_member_contributes_no_trainable_params():
    # Wan2.2 case: one trainable member + one frozen-but-bundled member.
    trainable = nn.Linear(4, 2)
    frozen = nn.Linear(4, 2)
    for p in frozen.parameters():
        p.requires_grad_(False)

    bundle = ModelBundle({"transformer": trainable, "transformer_2": frozen})

    grad_params = [p for p in bundle.parameters() if p.requires_grad]
    assert len(grad_params) == len(list(trainable.parameters()))
    # Frozen member is still a registered submodule (so it gets sharded).
    assert "transformer_2" in bundle.members


# ============================== RoutedComponentProxy ==============================


def test_routed_component_proxy_call_routes_through_bundle():
    lin = nn.Linear(4, 2)
    bundle = ModelBundle({"transformer": lin})
    proxy = RoutedComponentProxy(bundle, "transformer", lin)

    x = torch.randn(3, 4)
    assert torch.allclose(proxy(x), lin(x))


def test_routed_component_proxy_forwards_positional_and_kwargs():
    class TwoArg(nn.Module):
        def forward(self, a, b, scale=1.0):
            return (a + b) * scale

    m = TwoArg()
    bundle = ModelBundle({"m": m})
    proxy = RoutedComponentProxy(bundle, "m", m)

    out = proxy(torch.ones(2), torch.ones(2), scale=3.0)
    assert torch.allclose(out, torch.full((2,), 6.0))


def test_routed_component_proxy_delegates_attributes():
    lin = nn.Linear(4, 2)
    lin.custom_marker = 123
    bundle = ModelBundle({"transformer": lin})
    proxy = RoutedComponentProxy(bundle, "transformer", lin)

    assert proxy.custom_marker == 123  # arbitrary attribute
    assert proxy.in_features == 4  # module attribute
    assert list(proxy.parameters()) == list(lin.parameters())  # method delegation
    assert proxy.inner is lin  # explicit inner handle


def test_routed_component_proxy_exposes_inner():
    # `inner` is the handle BaseAdapter._unwrap peels the proxy down to.
    lin = nn.Linear(2, 2)
    bundle = ModelBundle({"transformer": lin})
    proxy = RoutedComponentProxy(bundle, "transformer", lin)

    assert proxy.inner is lin
