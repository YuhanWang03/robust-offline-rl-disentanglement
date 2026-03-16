from pathlib import Path

# Project root: robust-offline-rl-disentanglement/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RAW_METRICS_DIR = RESULTS_DIR / "raw_metrics"

# Artifacts directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
OBS_STATS_DIR = ARTIFACTS_DIR / "obs_stats"

# Optional runtime directories
LOGS_DIR = PROJECT_ROOT / "logs"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path