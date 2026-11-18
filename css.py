from collections.abc import Callable
from importlib import import_module
from typing import cast


def css_formating() -> None:
    """Apply notebook styling when IPython is installed; otherwise do nothing."""
    try:
        notebook_display = import_module("IPython.display")
    except ModuleNotFoundError as exc:
        if exc.name not in {"IPython", "IPython.display"}:
            raise
        return

    display = cast(Callable[[object], None], notebook_display.display)
    html = cast(Callable[[str], object], notebook_display.HTML)
    display(
        html("""
    <style>
    /* Make all xarray HTML output dark for VS Code Interactive Window */
    .xr-wrap, .xr-header, .xr-data, .xr-attrs, .xr-section, .xr-array-wrap, .xr-var-list, .xr-var-item, .xr-var-name, .xr-var-dims, .xr-var-attrs, .xr-var-data, .xr-attrs dt, .xr-attrs dd, .xr-section-title, .xr-section-summary, .xr-array-data, .xr-indexes, .xr-index, .xr-index-name, .xr-index-data, .xr-coord-list, .xr-coord-name, .xr-coord-dims, .xr-coord-data, .xr-coord-attrs, .xr-coord-item {
        background: #16191d !important;
        color: #e0e0e0 !important;
        border-color: #3c3f41 !important;
    }
    /* Variable names */
    .xr-var-name, .xr-dim-label, .xr-section-title, .xr-section-summary, .xr-attrs dt {
        color: #82aaff !important;
        font-weight: bold !important;
    }
    /* Highlighted/selected */
    .xr-var-item, .xr-coord-item, .xr-array-data tr:hover, .xr-var-data tr:hover {
        background: #20232a !important;
    }
    /* Attribute names */
    .xr-attrs dt {
        color: #bb86fc !important;
    }
    /* Attribute values */
    .xr-attrs dd {
        color: #b0bec5 !important;
    }
    /* Table cells and dtype cells */
    .xr-array-data td, .xr-var-data td, .xr-coord-data td, .xr-index-data td {
        background: #23272e !important;
        color: #e0e0e0 !important;
    }
    .xr-array-data .dtype, .xr-var-data .dtype, .xr-coord-data .dtype, .xr-index-data .dtype {
        background: #23272e !important;
        color: #fffbff !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        padding: 1px 6px !important;
    }
    /* Remove box shadow if any */
    .xr-wrap {
        box-shadow: none !important;
    }
    </style>
    """)
    )
