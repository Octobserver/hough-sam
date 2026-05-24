import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

def load_pickled_dataframes_from_directory(base_directory) -> Dict[int, pd.DataFrame]:
    results = {}
    p = Path(base_directory)
    for item in p.iterdir():
        if item.is_dir():
            file = f"{item.name}/{item.name}.pkl"
            file_path = os.path.join(base_directory, file)
            try:
                df = pd.read_pickle(file_path)
                if isinstance(df, pd.DataFrame):
                    case_index = int(item.name[5:9])
                    results[case_index] = df
            except Exception:
                continue
    return dict(results)

def load_npy_from_directory(base_directory) -> Dict[int, dict]:
    results = {}
    p = Path(base_directory)
    for item in p.iterdir():
       if item.name.endswith('.npy'):
            results[int(item.stem[5:9])] = {"features": np.load(item)}
    return dict(results)

def build_df_from_feature_label_dict(case_to_feats_and_labels: Dict, *, feature_key: str = "features", label_cols=None):
    import pandas as _pd
    if label_cols is None:
        label_cols = ["incision_architecture_rating","incision_location_rating","incision_size"]
    rows = []
    case_ids = []
    for cid, d in case_to_feats_and_labels.items():
        feats = np.asarray(d[feature_key]).reshape(-1)
        row = {f"f_{i}": feats[i] for i in range(len(feats))}
        for lc in label_cols:
            row[lc] = d[lc]
        rows.append(row)
        case_ids.append(cid)
    df = _pd.DataFrame(rows, index=_pd.Index(case_ids, name="case_id"))
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    df[feature_cols] = df[feature_cols].apply(_pd.to_numeric, errors="coerce").astype(np.float32)
    for lc in label_cols:
        df[lc] = _pd.to_numeric(df[lc], errors="coerce")
    return df, feature_cols
