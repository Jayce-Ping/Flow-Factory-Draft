import pytest
import torch

from flow_factory.rewards.clap import CLAPRewardModel


def test_clap_preprocessing_rejects_nonfinite_audio_before_model_forward() -> None:
    waveform = torch.zeros(2, 160)
    waveform[1, 12] = torch.nan

    with pytest.raises(
        ValueError,
        match=r"finite audio at index=0.*shape=\(2, 160\).*invalid_values=1",
    ):
        CLAPRewardModel._preprocess_audio(
            object.__new__(CLAPRewardModel),
            [waveform],
            src_sample_rate=48_000,
        )
