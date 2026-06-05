# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact Python wrapper around SAM 3 segmentation. The main source file is `segmentation.py`, which defines `SegmentationModel` and prompt helpers. Model assets live in `checkpoints/`, including `sam3.pt` and the BPE vocabulary. Sample inputs live in `dataset/`: `images/` contains PNG images, `masks/` contains NumPy mask files, and `annotation.json` stores labels or metadata. VS Code resolves imports from this repo and a sibling SAM 3 checkout at `/home/t000/Desktop/sam3`.

## Build, Test, and Development Commands

No package manager, build script, or test runner is committed. Use the project from the repository root so relative asset paths resolve correctly.

- `python -m py_compile segmentation.py`: quick syntax check.
- `python - <<'PY' ... PY`: run small import or helper checks from the repo root.
- `python segmentation.py`: not an entry point; add a guarded `if __name__ == "__main__":` block before using it this way.

For model inference, ensure `torch`, `numpy`, `Pillow`, and the external `sam3` package are installed and importable.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, snake_case functions and variables, PascalCase classes, and explicit imports at the top. Keep public methods focused on one inference mode, as in `pred`, `pred_mask_by_points`, and `pred_mask_per_obj`. Prefer type hints for public APIs and keep docstrings concise, with arguments and return shapes for images, masks, or tensors.

## Testing Guidelines

There is no committed test suite yet. When adding tests, place them under `tests/` and use `pytest` naming conventions such as `tests/test_segmentation_helpers.py`. Start with deterministic unit tests for helpers like `add_indefinite_article` and `add_articles_to_list`; avoid requiring the large checkpoint for ordinary unit tests. Use integration tests separately for model loading and inference, and document any required GPU or checkpoint assumptions.

## Commit & Pull Request Guidelines

Local Git history is unavailable in this checkout, so no repository-specific commit convention can be inferred. Use short, imperative commit subjects such as `Add point prompt mask tests` or `Fix article handling for prompts`. Pull requests should describe the behavior change, list validation commands run, mention checkpoint or environment requirements, and include before/after outputs or screenshots when changing generated masks or visual behavior.

## Security & Configuration Tips

Do not commit credentials, private datasets, or additional heavyweight checkpoints without confirming storage policy. Keep local paths such as `/home/t000/Desktop/sam3` in editor or environment configuration rather than hard-coding them in library code.
