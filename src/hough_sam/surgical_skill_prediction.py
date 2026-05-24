"""surgical_skill_prediction.py"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Hashable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .constants import LABELS_FILE, MODELS_DIR, SEG_RESULTS_DIR
from .modeling.data import load_pickled_dataframes_from_directory
from .modeling.features import (
    calculate_case_level_area_feature,
    calculate_case_level_detection_stability,
    calculate_case_level_geometric_feature,
    calculate_case_level_features,
    smooth,
)
from .modeling.train import train_select_dt_and_test, train_select_mlp_and_test

logger = logging.getLogger(__name__)


def label_distribution(case_level_features: Dict[Hashable, Dict[str, Any]], ids: Iterable[Hashable], label_col: str) -> Dict[Any, int]:
    vals = [case_level_features[cid][label_col] for cid in ids]
    unique, counts = np.unique(vals, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def stratified_split_case_indices(
    case_level_features: Dict[Hashable, Dict[str, Any]],
    label_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Hashable], List[Hashable], List[Hashable]]:
    ids = [cid for cid, v in case_level_features.items() if not pd.isna(v.get(label_col))]
    labels = [case_level_features[cid][label_col] for cid in ids]
    from sklearn.model_selection import train_test_split

    if not ids:
        return [], [], []

    try:
        train_ids, temp_ids, y_train, y_temp = train_test_split(ids, labels, train_size=train_ratio, random_state=seed, stratify=labels)
        if val_ratio + test_ratio <= 0:
            return train_ids, [], []
        rel_val = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids, _, _ = train_test_split(temp_ids, y_temp, train_size=rel_val, random_state=seed, stratify=y_temp)
        return list(train_ids), list(val_ids), list(test_ids)
    except Exception:
        # Fallback to random split without stratification
        rng = np.random.default_rng(seed)
        ids_arr = np.array(ids)
        rng.shuffle(ids_arr)
        n = len(ids_arr)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        train_ids = ids_arr[:n_train].tolist()
        val_ids = ids_arr[n_train : n_train + n_val].tolist()
        test_ids = ids_arr[n_train + n_val :].tolist()
        return train_ids, val_ids, test_ids


def augment(cases: Dict[Any, Dict[str, Any]], label_col: str = "incision_architecture_rating", aug_ratios: dict = None, max_ratio: int = 20) -> Dict[int, Dict[str, Any]]:
    if aug_ratios is None:
        aug_ratios = {-1.0: 20, 0.0: 10, 1.0: 2}

    def mask_10pct_random_frames(df: pd.DataFrame, frac: float = 0.10, seed: int = 42) -> Tuple[pd.DataFrame, int]:
        rand = np.random.default_rng(seed)
        eligible = df.index.to_numpy(dtype=int)
        if eligible.size == 0:
            return df.copy(deep=True), 0
        k = max(1, int(round(eligible.size * frac)))
        chosen = rand.choice(eligible, size=min(k, eligible.size), replace=False)
        out = df.drop(labels=chosen, errors="ignore")
        return out, int(eligible.size - k)

    results: Dict[int, Dict[str, Any]] = {}
    i = 0
    for train_case_id in cases:
        case_df = cases[train_case_id].get("__raw_df") if isinstance(cases[train_case_id].get("__raw_df"), pd.DataFrame) else None
        # The notebook used processed_cases global; here we expect caller to provide `cases` where each value
        # already includes the features. The safest approach is to attempt to find the raw frames in the same
        # structure under `cases[case_id]['__raw_df']` else skip augmentation for that case.
        if case_df is None:
            # cannot perform frame-level masking without the raw per-frame DataFrame
            continue

        start, stop, smoothed = smooth(case_df, max_tolerable_jump=5)
        if start < stop:
            input_df = smoothed.loc[start:stop]
            label_val = cases[train_case_id].get(label_col)
            times = aug_ratios.get(label_val, 1)
            for j in range(times):
                augmented_df, size = mask_10pct_random_frames(input_df, seed=j + 1)
                try:
                    geometric_features = calculate_case_level_geometric_feature(augmented_df, start, stop)
                    area_features = calculate_case_level_area_feature(augmented_df, start, stop)
                    detection_stability = calculate_case_level_detection_stability(start, stop)
                    detection_stability["longest_detection_run"] = size
                    labels = ["is_correct", "incision_architecture_rating", "incision_location_rating", "incision_size"]
                    ratings = {label: cases[train_case_id].get(label) for label in labels}
                    results[i * max_ratio + j] = {**geometric_features, **area_features, **detection_stability, **ratings}
                except Exception as e:
                    logger.warning("Error calculating case level features for case %s: %s", train_case_id, e)
            i = i + 1
    return results


def _iter_grid(g: Dict[str, Any]):
    keys = list(g.keys())
    vals = [g[k] for k in keys]
    for combo in np.array(np.meshgrid(*vals, indexing="ij"), dtype=object).reshape(len(keys), -1).T:
        yield {k: v for k, v in zip(keys, combo)}


def main():
    # Load segmentation results (pickled DataFrames)
    processed_cases = load_pickled_dataframes_from_directory(str(SEG_RESULTS_DIR))

    # Compute case level features and carry forward raw df under a key for augmentation convenience
    case_level_features_and_labels: Dict[int, Dict[str, Any]] = {}
    for case_index, case_df in processed_cases.items():
        try:
            feats = calculate_case_level_features(case_df)
            if feats:
                feats["__raw_df"] = case_df  # attach for augmentation if needed
                case_level_features_and_labels[case_index] = feats
        except Exception as e:
            logger.warning("Error calculating case level features for case %s: %s", case_index, e)

    # Load labels and merge
    labels_df = pd.read_excel(LABELS_FILE)
    for _, row in labels_df.iterrows():
        case_name = row["case_name"]
        try:
            case_index = int(case_name[5:9])
        except Exception:
            continue
        if case_index in case_level_features_and_labels:
            is_correct = row.get("Was the incision done correctly? (Y/N)")
            case_level_features_and_labels[case_index].update(
                {
                    "is_correct": str(is_correct).strip().lower() == "y",
                    "incision_architecture_rating": row.get("Incision Architecture Rating (-1 / 0 / 1)"),
                    "incision_location_rating": row.get("Incision Location Rating (-1 / 0 / 1)"),
                    "incision_size": row.get("Incision Size (-1 / 0 / 1)"),
                }
            )

    # Persist case-level CSV like notebook
    df_cases = pd.DataFrame.from_dict(case_level_features_and_labels, orient="index")
    out_csv = Path.cwd() / "case_level_features_with_labels.csv"
    df_cases.to_csv(out_csv, index=False)
    logger.info("Wrote case-level features CSV to %s", out_csv)

    train_case_ids_arc, val_case_ids_arc, test_case_ids_arc = stratified_split_case_indices(
        case_level_features_and_labels, label_col="incision_architecture_rating", train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=10
    )

    train_case_ids_loc, val_case_ids_loc, test_case_ids_loc = stratified_split_case_indices(
        case_level_features_and_labels, label_col="incision_location_rating", train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, seed=10
    )

    train_case_ids_size, val_case_ids_size, test_case_ids_size = stratified_split_case_indices(
        case_level_features_and_labels, label_col="incision_size", train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=10
    )

    # Build datasets
    train_arc = {id: case_level_features_and_labels[id] for id in train_case_ids_arc if id in case_level_features_and_labels}
    valid_arc = {id: case_level_features_and_labels[id] for id in val_case_ids_arc if id in case_level_features_and_labels}
    test_arc = {id: case_level_features_and_labels[id] for id in test_case_ids_arc if id in case_level_features_and_labels}

    # Augmentation: because our stored structure includes '__raw_df' for augmentation we use augment as-is
    augmented_arc = augment(train_arc, label_col="incision_architecture_rating")

    train_augmented_arc = {**train_arc}
    # Merge augmented into train_augmented set
    # augmented returns features dicts compatible with DataFrame creation
    for k, v in augmented_arc.items():
        train_augmented_arc[k] = v

    train_augmented_arc_df = pd.DataFrame.from_dict(train_augmented_arc, orient="index").dropna()
    valid_arc_df = pd.DataFrame.from_dict(valid_arc, orient="index")
    test_arc_df = pd.DataFrame.from_dict(test_arc, orient="index")

    feature_cols = [
        "min_ratio",
        "max_ratio",
        "mean_deviation",
        "max_deviation",
        "min_area",
        "max_area",
        "area_growth_slope_early",
        "area_growth_slope_late",
        "longest_detection_run",
    ]

    # Train Decision Tree on architecture rating
    if not train_augmented_arc_df.empty and not valid_arc_df.empty and not test_arc_df.empty:
        best_model_dt, report_dt, lb_dt = train_select_dt_and_test(
            train_augmented_df=train_augmented_arc_df,
            valid_df=valid_arc_df,
            test_df=test_arc_df,
            label_col="incision_architecture_rating",
            feature_cols=feature_cols,
            selection_metric="accuracy",
        )
        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(best_model_dt, open(str(MODELS_DIR / "best_tree_arc.pkl"), "wb"))
        logger.info("Saved DecisionTree model to %s", str(MODELS_DIR / "best_tree_arc.pkl"))

    # Train MLP for architecture rating
    try:
        best_mlp, rep_mlp, lb_mlp = train_select_mlp_and_test(
            train_augmented_arc_df, valid_arc_df, test_arc_df, feature_cols=feature_cols, label_col="incision_architecture_rating", selection_metric="accuracy", labels=[-1, 0, 1]
        )
        pickle.dump(best_mlp, open(str(MODELS_DIR / "best_mlp_arc.pkl"), "wb"))
        logger.info("Saved MLP model to %s", str(MODELS_DIR / "best_mlp_arc.pkl"))
    except Exception as e:
        logger.warning("MLP training failed: %s", e)


if __name__ == "__main__":
    main()
