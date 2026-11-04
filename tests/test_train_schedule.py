"""LR schedule seam (spec §5.1): linear warmup, then per-step cosine to 0 over the whole run."""

import math

import pytest

from mo433.train import lr_factor


def test_warmup_rises_linearly_to_one_at_end_of_warmup():
    # 4 warmup steps of 20: factors 1/4, 2/4, 3/4, 4/4 for steps 0..3
    assert [lr_factor(s, warmup_steps=4, total_steps=20) for s in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cosine_reaches_zero_at_the_last_step():
    assert lr_factor(19, warmup_steps=4, total_steps=20) == pytest.approx(0.0, abs=1e-12)


def test_cosine_midpoint_is_one_half():
    # after warmup, 16 decay steps: the 8th (step 4+8-1=11) is halfway down the cosine
    assert lr_factor(11, warmup_steps=4, total_steps=20) == pytest.approx(0.5)


def test_decay_is_monotone_and_continuous_after_warmup():
    fs = [lr_factor(s, warmup_steps=10, total_steps=1000) for s in range(1000)]
    assert all(b <= a for a, b in zip(fs[9:], fs[10:]))
    assert max(abs(b - a) for a, b in zip(fs, fs[1:])) <= 0.1 + 1e-9  # warmup increments are 0.1


def test_no_warmup_starts_just_below_one():
    assert lr_factor(0, warmup_steps=0, total_steps=100) == pytest.approx(0.5 * (1 + math.cos(math.pi / 100)))
