from __future__ import annotations

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

        for chrome_path in chrome_candidates:
            cmd = [
                str(chrome_path),
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                "--no-sandbox",
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
        fig.write_html(str(html_path))
        print(f"[info] Wrote Plotly HTML fallback: {html_path}")
    except Exception as err:
        print(f"[warn] Failed to save plotly HTML fallback at {html_path}: {err}")
