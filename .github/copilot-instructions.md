# Copilot instructions for graph-measures

Purpose: give an AI coding agent the minimal, high-value orientation to be immediately productive in this repository.

- **Big picture:** This repo provides an extensible system to compute node/edge topological features for graphs. Input graphs are provided as CSV edge lists or NetworkX objects; features are implemented as modular calculators (pure-Python and optional accelerated C++/CUDA variants). The main orchestrators are the `FeatureCalculator`/feature runner code and the `feature_manager` which map requested feature names (strings) to implementations.

- **Important paths & where to look first:**
  - graphMeasures/feature_calculators/: implementations, decorators, adapters (start here to add a new feature).
  - graphMeasures/feature_runners/: code that executes calculators (`feature_calculator_runner.py`, `additional_features_runner.py`).
  - graphMeasures/feature_manager/feature_manager.py: central coordination, feature metadata and orchestration.
  - graphMeasures/graph_features_metadata/: metadata mapping feature names to capabilities and expected output sizes.
  - graphMeasures/feature_calculators/accelerated_feature_calculators/: C++ wrappers and build hooks for accelerated features.
  - build_scripts/run_make_file_accelerated.py: helper to build accelerated components (Linux + Conda expected).
  - local_tests/ and examples/: small runnable graphs and quick-play scripts for debugging and regression checks.

- **Data flow (concise):**
  1. Caller creates a `FeatureCalculator` or invokes a runner and passes a graph (edge-list or NetworkX graph).
  2. Requested feature names (strings like `motif3`, `page_rank`) are looked up in metadata.
  3. The runner invokes the appropriate calculator (python or accelerated) and collects per-node outputs into a pandas DataFrame.
  4. By default results may be pickled/dumped; `should_dump`/`force_build` flags control re-use.

- **Project-specific conventions & patterns:**
  - Features are referenced by short string keys (see `graph_features_metadata/features_metadata.py`). Use those exact strings when requesting features.
  - Feature implementations often return pandas DataFrames or numpy arrays sized per-node; consistent post-processing is handled by runners/manager.
  - Decorators in `feature_calculators/decorators.py` are used to unify logging/argument handling—inspect before modifying calculators.
  - Accelerated code is opt-in via an `acc` flag and requires building C/C++ artifacts in the accelerated directory; the Python code has guards to raise if `acc=True` on unsupported platforms.

- **Build / test / debug commands (discoverable from README & scripts):**
  - Install runtime deps: `python -m pip install -r requirements.txt` (or create Conda env for accelerated builds).
  - Install in editable mode for development: `python -m pip install -e .` or `python setup.py develop`.
  - Run tests / quick checks: `pytest -q` (repo has tests and many `local_tests/*.py` scripts that can be run directly for examples).
  - Build accelerated native code: use `python build_scripts/run_make_file_accelerated.py` or run `make` in the accelerated subdirectories on Linux with the recommended Conda env.

- **Editing / adding a feature (quick checklist):**
  1. Add the implementation under `graphMeasures/feature_calculators/` (follow existing function/class patterns).
  2. If decorator behavior is needed, reuse `decorators.py` rather than copying logic.
  3. Add metadata entry in `graphMeasures/graph_features_metadata/` so the manager can discover the feature name and output shape.
  4. Add tests under `tests/` or `local_tests/` using a small example graph from `examples/`.
  5. If adding accelerated code, add wrapper/build files under `accelerated_feature_calculators/` and document build steps in `build_scripts/`.

- **Common pitfalls to avoid:**
  - Don’t assume accelerated code is available on CI or non-Linux machines—check `acc`/`gpu` guards.
  - Feature names are canonical; misspelling them leads to NaNs or KeyErrors (see README examples).
  - Node ordering: the code normalizes node labels via NetworkX utilities; output ordering may use reindexed nodes (0..n-1).

- **Quick example (from README):**
```py
from graphMeasures import FeatureCalculator
feats = ["motif3","louvain"]
f = FeatureCalculator(path_to_edgelist, feats, acc=False, directed=False, verbose=True)
f.calculate_features(force_build=True)
df = f.get_features()
```

If anything here is unclear or you want me to include more concrete file links or examples (e.g. a minimal test harness), tell me which area to expand.
