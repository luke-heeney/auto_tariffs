# Canonical Paper Pipeline

This directory contains the reproducible build for the canonical paper:

- `paper/a_main_draft_0_current_file.tex`
- `paper/Auto_Tariffs.pdf`

The paper pipeline treats the existing BLP demand-estimation results as fixed upstream inputs and regenerates the remaining paper-facing outputs from repo code:

- cost-side regression outputs
- counterfactual outputs and figures
- BLP paper tables derived from the saved results
- appendix tables and figures materialized into `paper/generated/`
- a rebuilt PDF in `paper/build/`

## Entry Point

Run the full pipeline from the repo root:

```bash
make paper
```

Useful variants:

```bash
python paper/build_paper.py --skip-render
python paper/build_paper.py --skip-compile
python paper/build_paper.py --skip-validation
python paper/build_paper.py --strict-canonical-pdf
make paper-fast
make paper-strict
```

`--skip-render` reuses existing upstream outputs but still rebuilds `paper/generated/`.
`--strict-canonical-pdf` makes canonical-PDF drift a hard failure instead of an informational comparison.

## Generated Directories

- `paper/generated/tables/`: paper-facing LaTeX tables
- `paper/generated/graphs/`: paper-facing figures
- `paper/generated/references.bbl`: bibliography used by the canonical TeX
- `paper/build/Auto_Tariffs.reproduced.pdf`: rebuilt paper PDF
- `paper/build/manifest.json`: build manifest and validation results
- `paper/build/canonical_pdf.normalized.txt`: normalized extracted text from `paper/Auto_Tariffs.pdf`
- `paper/build/reproduced_pdf.normalized.txt`: normalized extracted text from the rebuilt PDF
- `paper/build/canonical_pdf.diff`: compact diff excerpt when the rebuilt PDF does not match the canonical PDF

Both `paper/generated/` and `paper/build/` are git-ignored.

## Upstream Assumptions

- BLP demand estimation is not rerun.
- The pipeline uses `post_est/results_config.json` and the existing BLP results pickle referenced there.
- Canonical counterfactual paper assets are built from the latest metadata-valid rebased saved-output bundle for `post_est/results_config.json` and the `no tariff (no subsidy)` reporting baseline.
- In the root downstream replication workflow, `paper/make.sh` is called with `--skip-render` because cost-side and post-estimation outputs have already been regenerated.

## Validation

Validation always checks:

- every `\input{...}` and `\includegraphics{...}` in the canonical TeX resolves to a generated repo-local asset
- the rebuilt PDF exists

Canonical-PDF comparison additionally records:

- page-count parity between the rebuilt PDF and `paper/Auto_Tariffs.pdf`
- normalized extracted-text equality between the rebuilt PDF and `paper/Auto_Tariffs.pdf`
- normalized text snapshots and a compact diff artifact under `paper/build/`

Validation results are written to `paper/build/manifest.json`.

By default, `python paper/build_paper.py` succeeds when the repo-backed paper assets regenerate cleanly and the manuscript compiles. Use `--strict-canonical-pdf` if exact agreement with `paper/Auto_Tariffs.pdf` should be enforced.

At the current state of the pipeline, the rebuilt PDF still differs from the canonical PDF in layout/extraction details, so the strict flag remains informative rather than clean. The diff artifacts make that drift inspectable without blocking the standard rebuild.

## Main Implementation Files

- `paper/build_paper.py`: orchestrates build, compile, and validation
- `paper/paper_assets.py`: generates paper-facing tables, figures, bibliography, and asset checks
