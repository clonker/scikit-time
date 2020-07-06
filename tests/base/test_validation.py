import pytest

import numpy as np
from numpy.testing import *
import sktime as st


def test_time_series_container_list_behavior():
    lengths = np.arange(3, 13)
    original_data = [np.random.normal(size=(n, 50)).astype(np.float32) for n in lengths]
    ts = st.validation.TimeSeriesContainer(original_data)
    assert_(isinstance(ts, list))
    assert_equal(len(ts), len(original_data))
    for ix, t in enumerate(ts):
        assert_equal(t, original_data[ix])
        assert_equal(ts[ix], original_data[ix])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_time_series_container_slicing(dtype):
    lengths = np.arange(3, 13)
    original_data = [np.random.normal(size=(n, 50)).astype(dtype) for n in lengths]
    ts = st.validation.TimeSeriesContainer(original_data)
    for l, t in zip(lengths, ts):
        assert_equal(l, len(t))

    trajs = np.array([2, 2, 9])
    frames = np.array([1, 0, 4])
    data = ts[(trajs, frames)]
    assert_equal(data[0], original_data[2][1])
    assert_equal(data[1], original_data[2][0])
    assert_equal(data[2], original_data[9][4])

    data = ts[1, 1]
    assert_equal(data[0], original_data[1][1])


def test_time_series_container_invalid():
    with assert_raises(ValueError):
        # invalid dtype
        st.validation.TimeSeriesContainer([np.zeros((10, 4), dtype=bool)])

    with assert_raises(ValueError):
        # non-matching dim
        st.validation.TimeSeriesContainer([np.zeros((10, 4), dtype=np.float32), np.zeros((10, 5), dtype=np.float32)])

    with assert_raises(ValueError):
        # non-matching dtype
        st.validation.TimeSeriesContainer([np.zeros((10, 4), dtype=np.float32), np.zeros((10, 4), dtype=np.float64)])

    data = st.validation.TimeSeriesContainer([np.zeros((10, 4), dtype=np.float32), np.zeros((10, 4), dtype=np.float32)])

    with assert_raises(ValueError):
        # out of bounds trajs
        out = data[5, 0]

    with assert_raises(ValueError):
        # out of bounds frame
        out = data[0, 10]
