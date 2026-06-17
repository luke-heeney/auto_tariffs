from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _iter_chrome_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("PLOTLY_CHROME_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))

    candidates.extend([
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
    ])

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            out.append(path)
    return out


def _export_via_headless_chrome(fig: Any, png_path: Path) -> None:
    chrome_candidates = _iter_chrome_candidates()
    if not chrome_candidates:
        raise FileNotFoundError(
            "No Chrome/Chromium executable found. Set PLOTLY_CHROME_PATH to override."
        )

    width = int(getattr(fig.layout, "width", None) or 1100)
    height = int(getattr(fig.layout, "height", None) or 700)

    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs="inline",
        config={"responsive": False},
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>html,body{{margin:0;padding:0;overflow:hidden;background:#fff;width:{width}px;height:{height}px;}}"
        f"body>div{{width:{width}px;height:{height}px;}}</style>"
        "</head><body>"
        f"{fig_html}"
        "</body></html>"
    )

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="plotly_export_", dir="/tmp") as tmp_dir:
        html_path = Path(tmp_dir) / "figure.html"
        html_path.write_text(html, encoding="utf-8")
        profile_dir = Path(tmp_dir) / "chrome_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        for chrome_path in chrome_candidates:
            cmd = [
                str(chrome_path),
                "--headless",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--no-first-run",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                "--no-sandbox",
                f"--user-data-dir={profile_dir}",
                f"--window-size={width},{height}",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=5000",
                f"--screenshot={png_path}",
                html_path.as_uri(),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if proc.returncode == 0 and png_path.exists() and png_path.stat().st_size > 0:
                return

            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr or stdout or f"exit code {proc.returncode}"
            errors.append(f"{chrome_path}: {details}")

    raise RuntimeError(" ; ".join(errors) if errors else "headless Chrome export failed")


def _plot_coords(row: dict[str, Any]) -> tuple[float, float]:
    state = str(row.get("state_abbr", ""))
    lon = float(row["lon"])
    lat = float(row["lat"])

    # Keep non-contiguous states as visible insets in the same plotting window.
    if state == "AK":
        return -124.0, 26.6
    if state == "HI":
        return -115.0, 25.5
    return lon, lat


def _state_polygon_coords(state: str, lon: float, lat: float) -> tuple[float, float]:
    """Return lon/lat-like plotting coordinates with AK/HI drawn as insets."""
    if state == "AK":
        return (lon + 152.0) * 0.25 - 123.5, (lat - 63.0) * 0.25 + 25.5
    if state == "HI":
        return (lon + 157.0) * 1.0 - 116.0, (lat - 20.0) * 1.0 + 24.5
    return lon, lat


def _iter_state_polygon_segments(state: str, lons: list[float], lats: list[float]):
    segment: list[tuple[float, float]] = []
    for lon, lat in zip(lons, lats):
        if not math.isfinite(float(lon)) or not math.isfinite(float(lat)):
            if len(segment) >= 3:
                yield segment
            segment = []
            continue
        segment.append(_state_polygon_coords(state, float(lon), float(lat)))
    if len(segment) >= 3:
        yield segment


def _load_state_shapes() -> dict[str, dict[str, Any]]:
    try:
        from bokeh.sampledata.us_states import data as states
    except Exception as err:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("Bokeh us_states data are required for static map fallback.") from err
    return {abbr.upper(): shape for abbr, shape in states.items()}


def _draw_state_polygons(ax: Any, face_lookup: dict[str, Any]) -> None:
    from matplotlib.patches import Polygon

    states = _load_state_shapes()
    for state, shape in states.items():
        lons = [float(x) for x in shape["lons"]]
        lats = [float(x) for x in shape["lats"]]
        facecolor = face_lookup.get(state, "#e0e0e0")
        for segment in _iter_state_polygon_segments(state, lons, lats):
            ax.add_patch(
                Polygon(
                    segment,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=1,
                )
            )


def _finish_state_axis(ax: Any) -> None:
    ax.set_xlim(-126.8, -66.2)
    ax.set_ylim(23.0, 50.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _export_static_state_map(fig: Any, png_path: Path) -> None:
    meta = getattr(getattr(fig, "layout", None), "meta", None) or {}
    if not isinstance(meta, dict):
        raise ValueError("No static map metadata found.")
    payload = meta.get("replication_static_export") or {}
    if not isinstance(payload, dict) or payload.get("type") != "state_map":
        raise ValueError("No state-map static export payload found.")

    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    if not rows:
        raise ValueError("State-map static export payload has no rows.")

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    values = [float(row["value"]) for row in rows]
    finite_abs = [abs(value) for value in values if math.isfinite(value)]
    zmax = float(payload.get("zmax") or (max(finite_abs) if finite_abs else 1.0))
    if not math.isfinite(zmax) or zmax <= 0:
        zmax = 1.0

    face_lookup: dict[str, Any] = {}
    labels: list[str] = []
    label_rows: list[dict[str, Any]] = []
    norm = mcolors.TwoSlopeNorm(vmin=-zmax, vcenter=0.0, vmax=zmax)
    cmap = cm.get_cmap("RdBu")
    for row in rows:
        value = float(row["value"])
        if math.isfinite(value):
            face_lookup[str(row["state_abbr"]).upper()] = cmap(norm(value))
        label = f"{row['state_abbr']}\n{value:+.1f}%"
        units_millions = row.get("units_millions")
        if units_millions is not None and math.isfinite(float(units_millions)):
            label = f"{label}\n{float(units_millions):.2f}M"
        labels.append(label)
        label_rows.append(row)

    width = int(getattr(fig.layout, "width", None) or 1100)
    height = int(getattr(fig.layout, "height", None) or 700)
    dpi = 160
    fig_mpl, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig_mpl.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    _draw_state_polygons(ax, face_lookup)

    for row, label, value in zip(label_rows, labels, values):
        x, y = _plot_coords(row)
        color = "white" if abs(value) >= 0.6 * zmax else "black"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=5.8,
            color=color,
            linespacing=0.9,
            zorder=3,
        )

    _finish_state_axis(ax)

    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig_mpl.colorbar(scalar, ax=ax, fraction=0.035, pad=0.015)
    colorbar.set_label(str(payload.get("value_label", "% change")), fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    fig_mpl.subplots_adjust(left=0.015, right=0.93, top=0.985, bottom=0.015)
    fig_mpl.savefig(png_path, facecolor="white")
    plt.close(fig_mpl)


def _export_static_state_category_map(fig: Any, png_path: Path) -> None:
    meta = getattr(getattr(fig, "layout", None), "meta", None) or {}
    if not isinstance(meta, dict):
        raise ValueError("No static map metadata found.")
    payload = meta.get("replication_static_export") or {}
    if not isinstance(payload, dict) or payload.get("type") != "state_category_map":
        raise ValueError("No state-category static export payload found.")

    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    if not rows:
        raise ValueError("State-category static export payload has no rows.")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    color_lookup = {
        str(item["key"]): str(item["color"])
        for item in payload.get("categories", [])
        if isinstance(item, dict) and "key" in item and "color" in item
    }
    label_lookup = {
        str(item["key"]): str(item.get("label", item["key"]))
        for item in payload.get("categories", [])
        if isinstance(item, dict) and "key" in item
    }
    state_category = {
        str(row["state_abbr"]).upper(): str(row["category"])
        for row in rows
        if row.get("state_abbr")
    }
    face_lookup = {
        state: color_lookup.get(category, "#bdbdbd")
        for state, category in state_category.items()
    }

    width = int(getattr(fig.layout, "width", None) or 1100)
    height = int(getattr(fig.layout, "height", None) or 700)
    dpi = 160
    fig_mpl, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig_mpl.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    _draw_state_polygons(ax, face_lookup)

    for row in rows:
        state = str(row["state_abbr"]).upper()
        lat = row.get("lat")
        lon = row.get("lon")
        if lat is None or lon is None:
            continue
        x, y = _plot_coords({"state_abbr": state, "lat": lat, "lon": lon})
        ax.text(
            x,
            y,
            state,
            ha="center",
            va="center",
            fontsize=5.8,
            color="black",
            zorder=3,
        )

    title = str(payload.get("title", "")).strip()
    if title:
        ax.set_title(title, fontsize=9, loc="left", pad=2)
    _finish_state_axis(ax)

    handles = [
        Patch(facecolor=color_lookup[key], edgecolor="white", label=label_lookup.get(key, key))
        for key in color_lookup
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            framealpha=0.88,
            fontsize=6.5,
            borderpad=0.4,
            handlelength=1.0,
        )

    fig_mpl.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.015)
    fig_mpl.savefig(png_path, facecolor="white")
    plt.close(fig_mpl)


def save_plotly_figure(fig: Any, path_base: Path) -> None:
    if fig is None:
        return

    path_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_base.with_suffix(".png")
    html_path = path_base.with_suffix(".html")

    try:
        fig.write_image(str(png_path))
        return
    except Exception as err:
        print(f"[warn] Failed to save plotly PNG at {png_path}: {err}")

    try:
        _export_via_headless_chrome(fig, png_path)
        print(f"[info] Wrote Plotly PNG via headless Chrome: {png_path}")
        return
    except Exception as err:
        print(f"[warn] Failed browser PNG fallback at {png_path}: {err}")

    try:
        _export_static_state_map(fig, png_path)
        print(f"[info] Wrote static state-map PNG fallback: {png_path}")
        return
    except Exception as err:
        print(f"[warn] Failed static state-map fallback at {png_path}: {err}")

    try:
        _export_static_state_category_map(fig, png_path)
        print(f"[info] Wrote static state-category-map PNG fallback: {png_path}")
        return
    except Exception as err:
        print(f"[warn] Failed static state-category-map fallback at {png_path}: {err}")

    try:
        fig.write_html(str(html_path))
        print(f"[info] Wrote Plotly HTML fallback: {html_path}")
    except Exception as err:
        print(f"[warn] Failed to save plotly HTML fallback at {html_path}: {err}")
