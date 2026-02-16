# 🍬 SuiteEval

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![PyTerrier](https://img.shields.io/badge/PyTerrier-Compatible-orange)](https://github.com/terrier-org/pyterrier)

Tools for running **IR evaluation suites** with [PyTerrier](https://github.com/terrier-org/pyterrier).  
SuiteEval helps you define, run, and aggregate evaluations across datasets while managing temporary indices and memory footprint.

## 📘 Overview

**SuiteEval** provides:
- Declaration of **pipelines** (BM25, dense, re-ranking chains).
- Execution of **evaluation suites** (e.g., BEIR-style benchmarks).
- **DatasetContext** utilities for temporary paths and text loading.
- **DataFrame** outputs for downstream analysis.

Workflow:
1) Implement `pipelines(context)` that yields one or more PyTerrier pipelines (optionally named).  
2) Pass it to a suite (e.g., `BEIR`).  
3) Analyse the returned DataFrame.

## 🚀 Getting Started

### Install from PyPI
```bash
pip install suiteeval
```
### Install from source
```bash
git clone https://github.com/Parry-Parry/suiteeval.git
cd suiteeval
pip install -e .
```

## ⚙️ Defining Pipelines

Write a callable that accepts a `DatasetContext` and **returns or yields** pipelines.

- **Return** a list/tuple of pipelines or `(pipeline, name)` pairs; **or**  
- **Yield** pipelines to keep only one large model resident in memory.

`DatasetContext` provides:
- `context.path` — temporary working directory for indices/artifacts.  
- `context.get_corpus_iter()` — iterator suitable for indexing.  
- `context.text_loader()` — attaches document text for re-ranking.

### Example
```python
from suiteeval import BEIR
from pyterrier_pisa import PisaIndex
from pyterrier_dr import ElectraScorer
from pyterrier_t5 import MonoT5ReRanker

def pipelines(context):
    index = PisaIndex(context.path + "/index.pisa")
    index.index(context.get_corpus_iter())

    bm25 = index.bm25()
    yield bm25 >> context.text_loader() >> MonoT5ReRanker(), "BM25 >> monoT5"
    yield bm25 >> context.text_loader() >> ElectraScorer(), "BM25 >> monoELECTRA"

results = BEIR(pipelines)
```

## 🧪 Running Suites

Entry points (e.g., `BEIR`) accept your pipeline factory and return a DataFrame:

```python
results = BEIR(pipelines)  # per-dataset metrics and system names (if provided)
```

## 📦 Reproducibility & Resource Management

- **Temporary indices** live under `context.path` and are cleaned up.
- Prefer **yielding** pipelines when using large models.
- Name systems via `(pipeline, "<name>")` for clear result tables and logs.

### Persistent Index Storage

By default, indices are stored in temporary directories. To persist indices across runs, use the `index_dir` parameter:

```python
# Indices will be stored in ./indices/<corpus-name>/
# Run files will be stored in ./results/<dataset-name>/
results = BEIR(
    pipelines,
    save_dir="./results",   # Where to save run files (per-dataset)
    index_dir="./indices"   # Where to store indices (per-corpus)
)
```

Key differences:
- `save_dir` creates **per-dataset** subdirectories (e.g., `./results/beir-arguana/`)
- `index_dir` creates **per-corpus** subdirectories (e.g., `./indices/beir-arguana/`)
- Multiple datasets sharing a corpus will reuse the same index directory

### Automatic Result Caching

When using `save_dir`, SuiteEval automatically skips inference for pipelines that already have saved run files. If a `{pipeline_name}.res.gz` file exists for all datasets in a corpus, the suite loads results from disk instead of re-running the pipeline.

```python
# First run: executes inference and saves results
results = BEIR(pipelines, save_dir="./results")

# Second run: automatically loads from ./results/{dataset}/{name}.res.gz
results = BEIR(pipelines, save_dir="./results")
```

To force re-running inference, use `save_mode="overwrite"`:

```python
# Always re-run inference, even if files exist
results = BEIR(pipelines, save_dir="./results", save_mode="overwrite")
```

## 🛠️ Compatibility

Works with modern PyTerrier and common extensions  
(e.g., `pyterrier_pisa`, `pyterrier_dr`, `pyterrier_t5`).  
For older environments, ensure standard PyTerrier transformer interfaces.

## 👥 Authors

- [Andrew Parry](mailto:a.parry.1@research.gla.ac.uk)
- [Sean MacAvaney](mailto:Sean.MacAvaney@glasgow.ac.uk)

## 🧾 Version History

| Version | Date       | Changes                                              |
|--------:|------------|------------------------------------------------------|
|   0.1.7 | 2026-02-16 | Tempoary removal of DL23 until qrels are adeed |
|   0.1.6 | 2026-02-03 | Fix duplicate Overall rows, auto-detect all metrics  |
|   0.1.5 | 2026-01-07 | Custom index folder support for persistent indices   |
|   0.1.4 | 2025-12-01 | Fix save directory handling                          |
|   0.1.3 | 2025-12-01 | PyTerrier 1.0 compatibility, mixed datasets support  |
|   0.1.2 | 2025-10-29 | Documentation improvements and bug fixes             |
|     0.1 | 2025-10-03 | Initial release                                      |

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.
