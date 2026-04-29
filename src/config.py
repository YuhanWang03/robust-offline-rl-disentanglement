from pathlib import Path

# Project root: robust-offline-rl-disentanglement/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
SRC_DIR      = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
DOCS_DIR     = PROJECT_ROOT / "docs"
CONFIGS_DIR  = PROJECT_ROOT / "configs"

# Results root — raw data is never moved
RESULTS_DIR     = PROJECT_ROOT / "results"
RAW_METRICS_DIR = RESULTS_DIR / "raw_metrics"

# ── New 4-notebook output structure ──────────────────────────
# results/
# ├── raw_metrics/          ← training outputs, never touched here
# ├── main/                 ← 01_main_results.ipynb
# │   ├── figures/
# │   └── tables/
# ├── ablation/             ← 02_ablation_results.ipynb
# │   ├── figures/
# │   └── tables/
# ├── external_methods/     ← 03_external_methods.ipynb
# │   ├── figures/
# │   └── tables/
# └── comprehensive/        ← 04_comprehensive.ipynb
#     ├── figures/
#     └── tables/

MAIN_DIR          = RESULTS_DIR / "main"
FIGURES_MAIN_DIR  = MAIN_DIR / "figures"
TABLES_MAIN_DIR   = MAIN_DIR / "tables"

ABLATION_DIR          = RESULTS_DIR / "ablation"
FIGURES_ABLATION_DIR  = ABLATION_DIR / "figures"
TABLES_ABLATION_DIR   = ABLATION_DIR / "tables"

EXTERNAL_DIR          = RESULTS_DIR / "external_methods"
FIGURES_EXTERNAL_DIR  = EXTERNAL_DIR / "figures"
TABLES_EXTERNAL_DIR   = EXTERNAL_DIR / "tables"

COMPREHENSIVE_DIR          = RESULTS_DIR / "comprehensive"
FIGURES_COMPREHENSIVE_DIR  = COMPREHENSIVE_DIR / "figures"
TABLES_COMPREHENSIVE_DIR   = COMPREHENSIVE_DIR / "tables"

METHOD_SELECTION_DIR          = RESULTS_DIR / "method_selection"
FIGURES_METHOD_SELECTION_DIR  = METHOD_SELECTION_DIR / "figures"
TABLES_METHOD_SELECTION_DIR   = METHOD_SELECTION_DIR / "tables"

# Artifacts directories
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
OBS_STATS_DIR   = ARTIFACTS_DIR / "obs_stats"

# Optional runtime directories
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
