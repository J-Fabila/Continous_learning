import json
import numpy as np
import pandas as pd
import argparse
import requests
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
from datetime import datetime, timezone

# ACUERDATE DE PASAR LOS METADATOS

def KS_test(old_data,new_data,feature,alpha=0.05):
    # Kolmogorov-Smirnov (KS test)
    # For Continous variables
    empty_1 = old_data[feature].notna().any()
    empty_2 = new_data[feature].notna().any()
    
    p_value = None
    ks_stat = None
    
    if empty_1 == True and empty_2 == True:
        old_data["temp"] = pd.to_numeric(old_data[feature], errors="coerce").fillna(0)
        #print("NANS DETECTADO", dat_1["temp"].isna().sum())
        new_data["temp"] = pd.to_numeric(new_data[feature], errors="coerce").fillna(0)

        ks_stat, p_value = ks_2samp(old_data["temp"], new_data["temp"])
        if p_value < alpha:
            return 2, p_value, ks_stat # Data drift detected
        elif p_value < 1.0 and p_value > alpha:
            drift = 1 # Changes but not a drift
        else:
            drift = 0 # No changes
    elif empty_1 == False and empty_2 == False:
        drift = 3 # both empty, thus, no changes, thus, no data drift
    else:
        drift = 4  # one is empty and the other no, thus, there were changes, thus, there are important changes in distribution
    return drift, p_value, ks_stat

def chi2(old_data, new_data, feature, alpha=0.05):
    empty_1 = old_data[feature].notna().any()
    empty_2 = new_data[feature].notna().any()
    
    p_value = None
    stat = None
    
    if empty_1 == True and empty_2 == True:
        old = old_data[[feature]].copy()
        old["source"] = "old"
        new = new_data[[feature]].copy()
        new["source"] = "new"
        combined = pd.concat([old, new], axis=0)
        combined = combined.dropna(subset=[feature])
        
        contingency = pd.crosstab(combined[feature], combined["source"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            # Not enough data for chi2
            return 3, None, None
            
        stat, p_value, _, _ = chi2_contingency(contingency)
        if p_value < alpha:
            drift = 2 # Data drift
        elif p_value < 1.0 and p_value > alpha:
            drift = 1 # Changes but no data drift
        else:
            drift = 0 # No changes
    elif empty_1 == False and empty_2 == False:
        drift = 3 # both empty, thus, no changes, thus, no data drift
    else:
        drift = 4  # one is empty and the other no, thus, there were changes, thus, there are important changes in distribution
    return drift, p_value, stat

def drift_detection(config):

    data_file = config["data_file_1"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat_1 = pd.read_parquet(data_file)
    elif ext == "csv":
        dat_1 = pd.read_csv(data_file)

    data_file = config["data_file_2"]
    ext = data_file.split(".")[-1]
    if ext == "pqt" or ext == "parquet":
        dat_2 = pd.read_parquet(data_file)
    elif ext == "csv":
        dat_2 = pd.read_csv(data_file)

    with open(config['metadata_file_1'], 'r') as file:
        metadata_1 = json.load(file)

    with open(config['metadata_file_2'], 'r') as file:
        metadata_2 = json.load(file)

    DRIFT_LABELS = {
        0: {"label": "no_changes", "desc": "No changes"},
        1: {"label": "changes_no_drift", "desc": "There are changes but no data drift"},
        2: {"label": "data_drift", "desc": "Data drift"},
        3: {"label": "both_empty_no_drift", "desc": "Both empty, thus, no changes, thus, no data drift"},
        4: {"label": "one_empty_major_change", "desc": "One is empty and the other no, thus, there were changes, thus, there are important changes in distribution"}
    }

    drift_features = []

    def extract_features(metadata):
        feats = []
        if "entries" in metadata and len(metadata["entries"]) > 0:
            if "featureSet" in metadata["entries"][0]:
                feats.extend(metadata["entries"][0]["featureSet"].get("features", []))
        if "entity" in metadata:
            feats.extend(metadata["entity"].get("features", []))
            feats.extend(metadata["entity"].get("outcomes", []))
        
        # Deduplicate
        res = []
        seen = set()
        for f in feats:
            if f["name"] not in seen:
                res.append(f)
                seen.add(f["name"])
        return res

    metadata_1_features = extract_features(metadata_1)
    metadata_2_features = extract_features(metadata_2)
    metadata_2_index = {f["name"]: f for f in metadata_2_features}

    for feat in metadata_1_features:    
        feature = feat["name"]
        feat_dataType = feat.get("dataType", "")

        # IMPORTANT: DATE TIMES ARE NOT ANALYZED
        if feat_dataType == "NUMERIC":
            drift_level, p_value, effect_size = KS_test(dat_1, dat_2, feature, config["alpha"])
            effect_metric = "ks"
        elif feat_dataType in ["NOMINAL", "BOOLEAN"]:
            drift_level, p_value, effect_size = chi2(dat_1, dat_2, feature, config["alpha"])
            effect_metric = "chi2"
        else:
            continue
            
        label_info = DRIFT_LABELS.get(drift_level, {"label": "unknown", "desc": "unknown"})

        # Sanitize p_value and effect_size
        if pd.isna(p_value): p_value = None
        if pd.isna(effect_size): effect_size = None

        feature_data = {
            "feature_name": feature,
            "feature_type": feat_dataType,
            "drift": int(drift_level) if drift_level is not None else None,
            "drift_label": label_info["label"],
            "drift_description": label_info["desc"],
            "p_value": float(p_value) if p_value is not None else None,
            "effect_size": float(effect_size) if effect_size is not None else None,
            "effect_size_metric": effect_metric,
            "baseline_stats": {},
            "current_stats": {}
        }

        feat_2 = metadata_2_index.get(feature, {})
        
        stat_old = feat.get("statistics", {})
        stat_new = feat_2.get("statistics", {})

        if feat_dataType == "NUMERIC":
            feature_data["baseline_stats"] = {
                "Q1": stat_old.get("Q1"), "avg": stat_old.get("avg"), "min": stat_old.get("min"),
                "Q2": stat_old.get("Q2"), "max": stat_old.get("max"), "Q3": stat_old.get("Q3"),
                "numOfNotNull": stat_old.get("numOfNotNull")
            }
            feature_data["current_stats"] = {
                "Q1": stat_new.get("Q1"), "avg": stat_new.get("avg"), "min": stat_new.get("min"),
                "Q2": stat_new.get("Q2"), "max": stat_new.get("max"), "Q3": stat_new.get("Q3"),
                "numOfNotNull": stat_new.get("numOfNotNull")
            }
        elif feat_dataType == "NOMINAL":
            feature_data["baseline_stats"] = {
                "cardinalityPerItem": stat_old.get("cardinalityPerItem", {}),
                "numOfNotNull": stat_old.get("numOfNotNull")
            }
            feature_data["current_stats"] = {
                "cardinalityPerItem": stat_new.get("cardinalityPerItem", {}),
                "numOfNotNull": stat_new.get("numOfNotNull")
            }
        elif feat_dataType == "BOOLEAN":
            feature_data["baseline_stats"] = {
                "numOfTrue": stat_old.get("numOfTrue"),
                "numOfNotNull": stat_old.get("numOfNotNull")
            }
            feature_data["current_stats"] = {
                "numOfTrue": stat_new.get("numOfTrue"),
                "numOfNotNull": stat_new.get("numOfNotNull")
            }

        feature_data["baseline_stats"] = {k: v for k, v in feature_data["baseline_stats"].items() if v is not None}
        feature_data["current_stats"] = {k: v for k, v in feature_data["current_stats"].items() if v is not None}

        drift_features.append(feature_data)

    output = {
        "event_type": "data_drift_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_name": config.get("dataset_name", "unknown"),
        "model_name": config.get("model_name", "unknown"),
        "site": config.get("site", "unknown"),
        "features": drift_features
    }

    with open("data_drift.json", "w") as json_file_out:
        json.dump(output, json_file_out, indent=4)

    logstash_url = "https://matrix.srdc.com.tr/ai4hf/logstash"
    try:
        response = requests.post(
            logstash_url,
            auth=("logstash_internal", "2sgQdH0KrHa5c2lS0LGg"),
            json=output,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        print(f"Successfully sent data drift analysis to Logstash. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data drift analysis to Logstash: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    parser.add_argument("--data_file_1", type=str, default="" , help="Data file 1")
    parser.add_argument("--data_file_2",type=str, default="", help="Data file 2")
    parser.add_argument("--metadata_file_1", type=str, default="", help="metadata file 1")
    parser.add_argument("--metadata_file_2", type=str, default="", help="metadata file 2")
    parser.add_argument("--alpha", type=float, default=0.05 , help="alpha threshold")
    
    # New arguments for structured JSON
    parser.add_argument("--dataset_name", type=str, default="unknown", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="unknown", help="Model name")
    parser.add_argument("--site", type=str, default="unknown", help="Site name")
    parser.add_argument("--event_type", type=str, default="data_drift_analysis", help="Event type")

    args = parser.parse_args()
    config = vars(args)

    drift_detection(config)
    # Aquí se tendría que integrar con la plataforma
