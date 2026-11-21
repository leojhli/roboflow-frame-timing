"""Tests for the frame-timing schedules ("when does the fake camera send each frame?")."""
import numpy as np
import pytest

from frametiming.schedules import SCHEDULES, make_schedule


@pytest.mark.parametrize("kind", SCHEDULES)
def test_shape(kind):
    s = make_schedule(kind, n_frames=300)
    assert len(s) == 300 and s[0] == 0.0
    assert all(b >= a for a, b in zip(s, s[1:]))


@pytest.mark.parametrize("kind", SCHEDULES)
def test_average_rate_near_fps(kind):
    s = make_schedule(kind, n_frames=300, fps=30.0)
    assert 299 / s[-1] == pytest.approx(30.0, rel=0.10)


def test_steady_exact():
    s = make_schedule("steady", n_frames=4, fps=30.0)
    assert s == pytest.approx([0, 1 / 30, 2 / 30, 3 / 30])


def test_burst_spacing():
    d = np.diff(make_schedule("burst", n_frames=30))
    assert d[:9] == pytest.approx([0.001] * 9)
    assert d[9] > 0.2


def test_stall_has_one_second_gaps():
    d = np.diff(make_schedule("stall", n_frames=300))
    assert (d >= 0.99).sum() == 3


def test_jitter_seeded():
    assert make_schedule("jitter", seed=1) == make_schedule("jitter", seed=1)
    assert make_schedule("jitter", seed=1) != make_schedule("jitter", seed=2)
