import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_drift(drift_json_path="data_drift.json", save_path=None):
    """
    Visualiza los resultados de drift detection.
    
    Drift levels:
        0 : sin cambios
        1 : cambios sin drift significativo
        2 : data drift
        3 : ambos vacíos
        4 : uno vacío y otro no
    """

    with open(drift_json_path, "r") as f:
        drift_dict = json.load(f)

    # Convertimos a DataFrame
    rows = []
    for feature, values in drift_dict.items():
        rows.append({
            "feature": feature,
            "drift": values["drift"],
            "p_value": values["p_value"],
            "effect_size": values["effect_size"],
            "type": values["type"]
        })

    df = pd.DataFrame(rows)

    # ---------- RESUMEN ----------
    print("\nResumen Drift Detection\n")
    print(df.sort_values("drift", ascending=False).to_string(index=False))

    print("\nConteo por nivel de drift:")
    print(df["drift"].value_counts().sort_index())

    # ---------- VISUALIZACIÓN ----------
    color_map = {
        0: "green",
        1: "green",
        2: "red",
        3: "blue",
        4: "yellow"
    }

    colors = df["drift"].map(color_map)

    plt.figure(figsize=(12, 6))
    plt.bar(df["feature"], df["drift"], color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Drift Level")
    plt.title("Data Drift per Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"\nGráfico guardado en: {save_path}")
    else:
        plt.show()


def visualize_drift_heatmaps(drift_json_path="data_drift.json", save_prefix=None):
    """
    Visualización avanzada tipo dashboard con heatmaps.
    """

    with open(drift_json_path, "r") as f:
        drift_dict = json.load(f)

    rows = []
    for feature, values in drift_dict.items():
        rows.append({
            "feature": feature,
            "type": values["type"],
            "drift": values["drift"],
            "p_value": values["p_value"],
            "effect_size": values["effect_size"]
        })

    df = pd.DataFrame(rows)

    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["effect_size"] = pd.to_numeric(df["effect_size"], errors="coerce")

    plt.figure(figsize=(12, 8))
    pivot_drift = df.pivot_table(index="feature", columns="type", values="drift")
    plt.imshow(pivot_drift.fillna(-1), aspect="auto")
    plt.colorbar(label="Drift Level")
    plt.yticks(range(len(pivot_drift.index)), pivot_drift.index)
    plt.xticks(range(len(pivot_drift.columns)), pivot_drift.columns)
    plt.title("Drift Level Heatmap")
    plt.tight_layout()

    if save_prefix:
        plt.savefig(f"{save_prefix}_drift_heatmap.png")
    else:
        plt.show()

    plt.figure(figsize=(12, 8))
    pivot_p = df.pivot_table(index="feature", columns="type", values="p_value")
    plt.imshow(pivot_p.fillna(1), aspect="auto")
    plt.colorbar(label="p-value")
    plt.yticks(range(len(pivot_p.index)), pivot_p.index)
    plt.xticks(range(len(pivot_p.columns)), pivot_p.columns)
    plt.title("P-Value Heatmap")
    plt.tight_layout()

    if save_prefix:
        plt.savefig(f"{save_prefix}_pvalue_heatmap.png")
    else:
        plt.show()

    plt.figure(figsize=(12, 8))
    pivot_effect = df.pivot_table(index="feature", columns="type", values="effect_size")
    plt.imshow(pivot_effect.fillna(0), aspect="auto")
    plt.colorbar(label="Effect Size")
    plt.yticks(range(len(pivot_effect.index)), pivot_effect.index)
    plt.xticks(range(len(pivot_effect.columns)), pivot_effect.columns)
    plt.title("Effect Size Heatmap")
    plt.tight_layout()

    if save_prefix:
        plt.savefig(f"{save_prefix}_effectsize_heatmap.png")
    else:
        plt.show()

    print("\nResumen Drift por tipo:")
    print(df.groupby("type")["drift"].value_counts().unstack(fill_value=0))


def visualize_feature_distribution(drift_json_path="data_drift.json", feature_name=""):
    """
    Grafica la distribución aproximada OLD vs NEW
    usando únicamente estadísticas del metadata.
    """

    with open(drift_json_path, "r") as f:
        drift_dict = json.load(f)

    feat = drift_dict[feature_name]
    ftype = feat["type"]

    metadata_1 = feat["metadata_1"]
    metadata_2 = feat["metadata_2"]

    if ftype == "NUMERIC" and "Q1" in metadata_1:

        # Estimación sigma usando IQR
        def estimate_std(q1, q3):
            return (q3 - q1) / 1.35 if q3 != q1 else 1e-6

        mean1 = metadata_1["avg"]
        std1 = estimate_std(metadata_1["Q1"], metadata_1["Q3"])

        mean2 = metadata_2["avg"]
        std2 = estimate_std(metadata_2["Q1"], metadata_2["Q3"])

        xmin = min(metadata_1["min"], metadata_2["min"])
        xmax = max(metadata_1["max"], metadata_2["max"])

        x = np.linspace(xmin, xmax, 200)

        y1 = norm.pdf(x, mean1, std1)
        y2 = norm.pdf(x, mean2, std2)

        plt.figure()
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.title(f"Approx Distribution - {feature_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend(["Old", "New"])
        plt.show()

        # Boxplot comparativo
        plt.figure()
        plt.boxplot([
            [metadata_1["min"], metadata_1["Q1"], metadata_1["Q2"],
             metadata_1["Q3"], metadata_1["max"]],
            [metadata_2["min"], metadata_2["Q1"], metadata_2["Q2"],
             metadata_2["Q3"], metadata_2["max"]]
        ])
        plt.xticks([1, 2], ["Old", "New"])
        plt.title(f"Boxplot Approx - {feature_name}")
        plt.show()

    elif ftype == "NOMINAL":

        old_counts = metadata_1.get("cardinalityPerItem", {})
        new_counts = metadata_2.get("cardinalityPerItem", {})

        categories = list(set(old_counts.keys()) | set(new_counts.keys()))
        old_vals = [old_counts.get(cat, 0) for cat in categories]
        new_vals = [new_counts.get(cat, 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        plt.figure()
        plt.bar(x - width/2, old_vals, width)
        plt.bar(x + width/2, new_vals, width)
        plt.xticks(x, categories, rotation=45)
        plt.title(f"Category Distribution - {feature_name}")
        plt.legend(["Old", "New"])
        plt.show()

    elif ftype == "BOOLEAN":

        old_true = metadata_1.get("numOfTrue", 0)
        new_true = metadata_2.get("numOfTrue", 0)

        old_total = metadata_1.get("numOfNotNull", 1)
        new_total = metadata_2.get("numOfNotNull", 1)

        old_false = old_total - old_true
        new_false = new_total - new_true

        labels = ["True", "False"]
        old_vals = [old_true, old_false]
        new_vals = [new_true, new_false]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure()
        plt.bar(x - width/2, old_vals, width)
        plt.bar(x + width/2, new_vals, width)
        plt.xticks(x, labels)
        plt.title(f"Boolean Distribution - {feature_name}")
        plt.legend(["Old", "New"])
        plt.show()

    else:
        print("No sufficient statistics to plot this feature.")


def visualize_drift_semaphore(drift_json_path="data_drift.json"):
    """
    Visualización tipo SEMÁFORO con colores reales.
    """

    with open(drift_json_path, "r") as f:
        drift_dict = json.load(f)

    rows = []
    for feature, values in drift_dict.items():
        rows.append({
            "feature": feature,
            "drift": values["drift"]
        })

    df = pd.DataFrame(rows).sort_values("drift", ascending=False)

    drift_values = df["drift"].values.reshape(-1, 1)

    # Colores discretos tipo semáforo
    colors = [
        "green",   # 0
        "yellow",  # 1
        "red",     # 2
        "blue",    # 3
        "purple"   # 4
    ]

    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, len(df) * 0.4 + 2))
    plt.imshow(drift_values, aspect="auto", cmap=cmap, norm=norm)

    plt.yticks(range(len(df)), df["feature"])
    plt.xticks([0], ["Drift"])

    cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4])
    cbar.set_label("Drift Level")

    plt.title("Data Drift Semaphore")
    plt.tight_layout()
    plt.show()

    print("\nResumen:")
    print(df["drift"].value_counts().sort_index())


visualize_drift()
visualize_feature_distribution("data_drift.json", "lab_results_hba1c_value_min")
visualize_drift_semaphore()
"""
hyperkalemia_severity_categorizedValue
lab_results_hba1c_value_min
lab_results_hba1c_value_first
echocardiographs_lvef

"""