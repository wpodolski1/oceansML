import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

import pytest
import torch
import xarray as xr

import css
from css import css_formating
from helper_functions import (
    load_globals,
    open_emissions_file,
    open_nc_dataset,
    open_ncs_dataset_enriched,
)


@pytest.mark.parametrize(
    ("opener", "variable"),
    [
        (open_nc_dataset, "DATA_PATH"),
        (open_ncs_dataset_enriched, "DATA_PATH_ENRICHED"),
        (open_emissions_file, "DATA_PATH_EMISSIONS"),
    ],
)
def test_dataset_helpers_use_configured_directory(
    opener: Callable[[str], xr.Dataset],
    variable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = xr.Dataset({"sst": ("time", [27.0, 28.0])})
    _ = cast(Callable[[Path], object], expected.to_netcdf)(tmp_path / "sample.nc")
    monkeypatch.setenv(variable, str(tmp_path))

    with opener("sample.nc") as actual:
        cast(Callable[[xr.Dataset, xr.Dataset], None], xr.testing.assert_identical)(
            actual, expected
        )


@pytest.mark.parametrize(
    ("opener", "variable"),
    [
        (open_nc_dataset, "DATA_PATH"),
        (open_ncs_dataset_enriched, "DATA_PATH_ENRICHED"),
        (open_emissions_file, "DATA_PATH_EMISSIONS"),
    ],
)
def test_dataset_helpers_report_missing_configuration(
    opener: Callable[[str], xr.Dataset], variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(KeyError, match=variable):
        _ = opener("sample.nc")


def test_load_globals_loads_environment_and_returns_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    _ = env_file.write_text("OCEANSML_HELPER_TEST=loaded\n", encoding="utf-8")
    # Restore the environment after python-dotenv updates it directly.
    monkeypatch.setenv("OCEANSML_HELPER_TEST", "")
    monkeypatch.delenv("OCEANSML_HELPER_TEST")

    device = load_globals(str(env_file))

    assert os.environ["OCEANSML_HELPER_TEST"] == "loaded"
    assert type(device) is torch.device
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_css_is_optional_without_ipython(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    assert css_formating() is None


def test_css_does_not_hide_missing_ipython_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_with_missing_dependency(_name: str) -> NoReturn:
        raise ModuleNotFoundError("Missing notebook dependency", name="traitlets")

    monkeypatch.setattr(css, "import_module", import_with_missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="Missing notebook dependency"):
        css_formating()
