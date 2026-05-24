from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .data import build_df_from_feature_label_dict
from .train import train_select_dt_and_test

def iter_grid(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in np.array(np.meshgrid(*vals, indexing="ij"), dtype=object).reshape(len(keys), -1).T:
        yield {k: v for k, v in zip(keys, combo)}

def train_select_dt_and_test(train_augmented_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, label_col: str, *, selection_metric: str = "accuracy", grid: Optional[Dict[str, Any]] = None, random_state: int = 42, return_leaderboard: bool = True) -> Tuple[Pipeline, Dict[str, Any], Optional[pd.DataFrame]]:
    def _Xy(df: pd.DataFrame):
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = df[label_col].to_numpy()
        return X, y
    X_train, y_train = _Xy(train_augmented_df)
    X_val, y_val = _Xy(valid_df)
    X_test, y_test = _Xy(test_df)
    metric_fn = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}.get(selection_metric)
    if metric_fn is None:
        raise ValueError("selection_metric must be 'accuracy' or 'balanced_accuracy'.")
    if grid is None:
        grid = {"criterion": ["gini", "entropy"], "max_depth": [2,3,4], "min_samples_split": [2,5], "min_samples_leaf": [1,2], "ccp_alpha": [0.0, 1e-4]}
    def make_model(params: Dict[str, Any]) -> Pipeline:
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", DecisionTreeClassifier(random_state=random_state, **params))])
    best_model = None
    best_val_score = -np.inf
    rows = []
    for params in iter_grid(grid):
        model = make_model(params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_score = float(metric_fn(y_val, val_pred))
        rows.append({**params, f"val_{selection_metric}": val_score})
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
            best_params = params
    assert best_model is not None
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    report = {"label_col": label_col, "feature_cols": feature_cols, "selection_metric": selection_metric, "best_params": best_params, f"valid_{selection_metric}": float(metric_fn(y_val, val_pred)), "valid_accuracy": float(accuracy_score(y_val, val_pred)), "valid_balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)), "test_accuracy": float(accuracy_score(y_test, test_pred)), "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)), "test_confusion_matrix": confusion_matrix(y_test, test_pred), "test_classification_report": classification_report(y_test, test_pred, digits=4),}
    leaderboard = None
    if return_leaderboard:
        leaderboard = pd.DataFrame(rows).sort_values(by=f"val_{selection_metric}", ascending=False).reset_index(drop=True)
    print(f"[{label_col}] best valid {selection_metric}: {best_val_score:.4f}")
    return best_model, report, leaderboard

def train_select_mlp_and_test(train_augmented_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], label_col: str, *, selection_metric: str = "balanced_accuracy", grid: Optional[Dict[str, Any]] = None, random_state: int = 42, return_leaderboard: bool = True, labels: Optional[List[int]] = None):
    def _Xy(df: pd.DataFrame):
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = df[label_col].to_numpy()
        return X, y
    X_train, y_train = _Xy(train_augmented_df)
    X_val, y_val = _Xy(valid_df)
    X_test, y_test = _Xy(test_df)
    metric_fn = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}.get(selection_metric)
    if metric_fn is None:
        raise ValueError("selection_metric must be 'accuracy' or 'balanced_accuracy'.")
    if grid is None:
        grid = {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-5, 1e-4], "learning_rate_init": [1e-3], "max_iter": [500]}
    from itertools import product
    def _iter_grid(g):
        keys = list(g.keys())
        values = [g[k] if isinstance(g[k], (list, tuple)) else [g[k]] for k in keys]
        for combo in product(*values):
            yield dict(zip(keys, combo))
    def make_model(params: Dict[str, Any]):
        return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", MLPClassifier(random_state=random_state, early_stopping=True, n_iter_no_change=20, **params))])
    best_model = None
    best_val_score = -np.inf
    rows = []
    for params in _iter_grid(grid):
        model = make_model(params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_score = float(metric_fn(y_val, val_pred))
        rows.append({**params, f"val_{selection_metric}": val_score})
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
            best_params = params
    assert best_model is not None
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred, labels=labels) if labels is not None else confusion_matrix(y_test, test_pred)
    cr = classification_report(y_test, test_pred, labels=labels, digits=4, zero_division=0) if labels is not None else classification_report(y_test, test_pred, digits=4, zero_division=0)
    report = {"model_type": "MLPClassifier", "label_col": label_col, "selection_metric": selection_metric, "best_params": best_params, f"valid_{selection_metric}": float(metric_fn(y_val, val_pred)), "test_confusion_matrix": cm, "test_classification_report": cr}
    leaderboard = None
    if return_leaderboard:
        leaderboard = pd.DataFrame(rows).sort_values(by=f"val_{selection_metric}", ascending=False).reset_index(drop=True)
    print(f"[MLP:{label_col}] best valid {selection_metric}: {best_val_score:.4f}")
    return best_model, report, leaderboard

def train_all_labels_baselines(case_to_feats_and_labels: Dict, *, feature_key: str = "features", label_cols=None, random_state: int = 42, selection_metric: str = "balanced_accuracy") -> Dict[str, Any]:
    if label_cols is None:
        label_cols = ["incision_architecture_rating","incision_location_rating","incision_size"]
    df, feature_cols = build_df_from_feature_label_dict(case_to_feats_and_labels, feature_key=feature_key, label_cols=label_cols)
    results = {"feature_cols": feature_cols, "reports": {}, "leaderboards": {}, "models": {}}
    
    for label_col in label_cols:
        splits = None
        # create simple stratified splits
        try:
            train_df, tmp_df = train_test_split(df, train_size=0.6, random_state=random_state, stratify=df[label_col])
        except:
            train_df, tmp_df = train_test_split(df, train_size=0.6, random_state=random_state)
        valid_frac_of_tmp = 0.5
        try:
            valid_df, test_df = train_test_split(tmp_df, train_size=valid_frac_of_tmp, random_state=random_state, stratify=tmp_df[label_col])
        except:
            valid_df, test_df = train_test_split(tmp_df, train_size=valid_frac_of_tmp, random_state=random_state)
        dt_model, dt_report, dt_leader = train_select_dt_and_test(train_df, valid_df, test_df, feature_cols, label_col, selection_metric=selection_metric, random_state=random_state, return_leaderboard=True)
        results["models"][label_col] = {"dt": dt_model}
        results["reports"][label_col] = {"dt": dt_report}
        results["leaderboards"][label_col] = {"dt": dt_leader}
    return results
