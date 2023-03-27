import os

import click
from click.testing import CliRunner
import pytest
import rasterio
import numpy as np
import sys

from mysite.utils.rio_hist.scripts.cli import hist, validate_proportion


def test_hist_cli(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, [source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    assert os.path.exists(output)


def test_hist_cli2(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, [source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    with rasterio.open(output) as out:
        assert out.count == 4  # RGBA


def test_hist_cli_lab_colorspace(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, ['-c', 'Lab', '-b', '1,2,3',
               source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    assert os.path.exists(output)


def test_hist_cli_lch_colorspace(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, ['-c', 'LCH', '-b', '1,2,3',
               source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    assert os.path.exists(output)


def test_hist_cli_luv_colorspace(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, ['-c', 'LUV', '-b', '1,2,3',
               source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    assert os.path.exists(output)


def test_hist_cli_xyz_colorspace(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, ['-c', 'XYZ', '-b', '1,2,3',
               source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    assert os.path.exists(output)


def test_hist_cli_plot(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, ['-v', '--plot',
               source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    with rasterio.open(output) as out:
        assert out.count == 4  # RGBA


def test_partial(source_path, reference_path, tmpdir):
    output = tmpdir
    runner = CliRunner()
    result = runner.invoke(
        hist, ['-m', '0.5',
               source_path,
               reference_path,
               output])
    assert result.exit_code == 0
    with rasterio.open(output) as match, \
            rasterio.open(source_path) as src, \
            rasterio.open(reference_path) as ref:
        m = match.read(2)
        s = src.read(2)
        r = ((ref.read(2) / 65365) * 256).astype('uint8')
        assert np.median(s) > np.median(m)  # darker than the source
        assert np.median(r) < np.median(m)  # but not quite as dark as the reference


def test_validate_proportion():
    assert validate_proportion(None, None, 0) == 0.0
    assert validate_proportion(None, None, 0.5) == 0.5
    assert validate_proportion(None, None, 1) == 1.0
    with pytest.raises(click.BadParameter):
        assert validate_proportion(None, None, 9000)
