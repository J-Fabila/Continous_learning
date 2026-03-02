import json
import numpy as np
import pandas as pd
import argparse
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

# ACUERDATE DE PASAR LOS METADATOS

def KS_test(old_data,new_data,feature,alpha=0.05):
    # Kolmogorov-Smirnov (KS test)
    # For Continous variables
    empty_1 = old_data[feature].notna().any()
    empty_2 = new_data[feature].notna().any()
    if empty_1 == True and empty_2 == True:
        old_data["temp"] = pd.to_numeric(old_data[feature], errors="coerce").fillna(0)
        #print("NANS DETECTADO", dat_1["temp"].isna().sum())
        new_data["temp"] = pd.to_numeric(new_data[feature], errors="coerce").fillna(0)

        ks_stat, p_value = ks_2samp(old_data["temp"], new_data["temp"])
        if p_value < alpha:
            return 2 # Data drift detected
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
    if empty_1 == True and empty_2 == True:
        old = old_data[[feature]].copy()
        old["source"] = "old"
        new = new_data[[feature]].copy()
        new["source"] = "new"
        combined = pd.concat([old, new], axis=0)
        combined = combined.dropna(subset=[feature])
        contingency = pd.crosstab(combined[feature], combined["source"])
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

    drift_dict = {}
    metadata_2_index = {
        f["name"]: f for f in metadata_2["entity"]["features"]
    }

    #for feat in metadata_1["entity"]["features"]:
    for feat in metadata_1["entries"][0]["featureSet"]["features"]:    
        feature = feat["name"]
        drift_dict[feature] = {
                "name": feature,
                "drift": None,
                "p_value": None,
                "effect_size": None,
                "type": feat["dataType"],
                "metadata_1": {},
                "metadata_2": {}
        }

        # IMPORTANT: DATE TIMES ARE NOT ANALYZED
        if feat["dataType"] == "NUMERIC":
            drift_level, p_value, effect_size = KS_test(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "NOMINAL":
            drift_level, p_value, effect_size = chi2(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "BOOLEAN":
            drift_level, p_value, effect_size = chi2(dat_1,dat_2,feature,config["alpha"])
        #print("FEATURE",feature, feat["dataType"], drift)        
        drift_dict[feature]["drift"] = drift_level
        drift_dict[feature]["p_value"] = p_value
        drift_dict[feature]["effect_size"] = effect_size

        feat_2 = metadata_2_index.get(feature)

        if feat["dataType"] == "NUMERIC":
            drift_dict[feature]["metadata_1"] = {
                "Q1": feat["statistics"]["Q1"],
                "avg": feat["statistics"]["avg"],
                "min": feat["statistics"]["min"],
                "Q2": feat["statistics"]["Q2"],
                "max": feat["statistics"]["max"],
                "Q3": feat["statistics"]["Q3"],
                "numOfNotNull": feat["statistics"]["numOfNotNull"],
            }
            drift_dict[feature]["metadata_2"] = {
                "Q1": feat_2["statistics"]["Q1"],
                "avg": feat_2["statistics"]["avg"],
                "min": feat_2["statistics"]["min"],
                "Q2": feat_2["statistics"]["Q2"],
                "max": feat_2["statistics"]["max"],
                "Q3": feat_2["statistics"]["Q3"],
                "numOfNotNull": feat_2["statistics"]["numOfNotNull"],
            }
        elif feat["dataType"] == "NOMINAL":
            drift_dict[feature]["metadata_1"] = {
                "cardinalityPerItem": feat["statistics"]["cardinalityPerItem"],
                "numOfNotNull": feat["statistics"]["numOfNotNull"]
            }
            drift_dict[feature]["metadata_2"] = {
                "cardinalityPerItem": feat_2["statistics"]["cardinalityPerItem"],
                "numOfNotNull": feat_2["statistics"]["numOfNotNull"]
            }
        elif feat["dataType"] ==  "BOOLEAN":
            drift_dict[feature]["metadata_1"] = {
                "numOfTrue": feat["statistics"]["numOfTrue"],
                "numOfNotNull": feat["statistics"]["numOfNotNull"]
            }
            drift_dict[feature]["metadata_2"] = {
                "numOfTrue": feat_2["statistics"]["numOfTrue"],
                "numOfNotNull": feat_2["statistics"]["numOfNotNull"]
            }

    for feat in metadata_1["entity"]["outcomes"]:
    #for feat in metadata["entries"][0]["featureSet"]["features"]:

        feature = feat["name"]
        if feat["dataType"] == "NUMERIC":
            drift_level, p_value, effect_size = KS_test(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "NOMINAL":
            drift_level, p_value, effect_size = chi2(dat_1,dat_2,feature,config["alpha"])
        elif feat["dataType"] == "BOOLEAN":
            drift_level, p_value, effect_size = chi2(dat_1,dat_2,feature,config["alpha"])
        #print("OUTCOME",feature, feat["dataType"], drift)

    with open("data_drift.json", "w") as json_file_out:
        json.dump(drift_dict, json_file_out, indent=4)
    # Estructura nueva:
    # {"feature":{"drift_level":int,"p_value":float,"effect_size":float,"metadata_old":{METADATA},"metadata_new":{METADATA}}}
    # se podría hacer que devuelva varios valores:
    # 0 : sin cambios
    # 1 : cambio en los valores pero sin data drift
    # 2 : data drift
    # 3 : both empty, thus, no changes, thus, no data drift
    # 4 : one is empty and the other no, thus, there were changes, thus, there are important changes in distribution

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    parser.add_argument("--data_file_1", type=str, default="" , help="Data file 1")
    parser.add_argument("--data_file_2",type=str, default="", help="Data file 2")
    parser.add_argument("--metadata_file_1", type=str, default="", help="metadata file 1")
    parser.add_argument("--metadata_file_2", type=str, default="", help="metadata file 1")
    parser.add_argument("--alpha", type=float, default=0.05 , help="alpha threshold")

    args = parser.parse_args()

    config = vars(args)

    drift_detection(config)
    # Aquí se tendría que integrar con la plataforma
