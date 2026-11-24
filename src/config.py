from pathlib import Path

# ---------- Paths ----------
# Resolved relative to this file, so the scripts work from any working directory.
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"          # duplicates dropped + train / test split
TRAIN_PATH = CLEANED_DIR / "train.csv"
TEST_PATH = CLEANED_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"   # selected model + its threshold
REPORTS_DIR = BASE_DIR / "reports"

# ---------- Data ----------
TARGET_COL = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- Cross-validation / metrics ----------
CV_SPLITS = 3
MAIN_METRIC = "PR-AUC"     # picks the best hyperparameters (refit) and the best model
N_JOBS = -1

# ---------- Training ----------
MODEL_SELECT = "decision_tree"   # logistic_regression, decision_tree, random_forest, xgboost


def model_path(model_name, step):
    """Where train.py saves the best model of one search."""
    return MODELS_DIR / f"{model_name}_{step}.joblib"
