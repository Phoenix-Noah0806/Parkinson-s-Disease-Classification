
import os
import glob
from pathlib import Path

# ==================================================
# 1. CONFIGURE SPARK ENVIRONMENT
# ==================================================

os.makedirs(r"D:\Temp", exist_ok=True)
os.makedirs(r"D:\spark-temp", exist_ok=True)

os.environ["TEMP"] = r"D:\Temp"
os.environ["TMP"] = r"D:\Temp"

os.environ["SPARK_HOME"] = r"D:\PySpark35"

os.environ["PYSPARK_PYTHON"] = (
    r"D:\ParkinsonsVenv\Scripts\python.exe"
)

os.environ["PYSPARK_DRIVER_PYTHON"] = (
    r"D:\ParkinsonsVenv\Scripts\python.exe"
)

os.environ.pop("HADOOP_USER_CLASSPATH_FIRST", None)

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--driver-memory 2g pyspark-shell"
)


# ==================================================
# 2. IMPORT LIBRARIES
# ==================================================

import pyspark

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ==================================================
# 3. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_DIR / "results"

CSV_DIR = RESULT_DIR / "csv"

CLEANED_PATH = (
    PROJECT_DIR / "data" / "processed" / "cleaned_dataset"
)

IMPORTANCE_PATH = (
    RESULT_DIR / "random_forest_feature_importance"
)

PLOT_DIR = RESULT_DIR / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 4. VERIFY REQUIRED FILES
# ==================================================

required_csv_folders = [
    "model_metrics",
    "confusion_logistic_regression",
    "confusion_random_forest",
    "error_summary"
]

for folder in required_csv_folders:

    path = CSV_DIR / folder

    if not path.exists():
        raise FileNotFoundError(
            f"Required evaluation results not found: {path}"
        )


if not CLEANED_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run 02_prepare_data.py first."
    )

if not IMPORTANCE_PATH.exists():
    raise FileNotFoundError(
        "Feature importance not found. Run 03_train_models.py first."
    )


# ==================================================
# 5. READ SPARK CSV OUTPUT
# ==================================================

def read_spark_csv(folder):

    files = glob.glob(
        str(Path(folder) / "part-*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV part files found in {folder}"
        )

    # Read CSV files written by Spark.
    # Pandas is used only for visualization.

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    return pd.concat(
        frames,
        ignore_index=True
    )


# ==================================================
# 6. INITIALIZE SPARK
# ==================================================

spark = (
    SparkSession.builder
    .appName("Parkinsons_Visualization_Data")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.sql.constraintPropagation.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:

    print("\n========== SPARK CONFIGURATION ==========")

    print("PySpark version:", pyspark.__version__)
    print("Java Spark version:", spark.version)

    if spark.version != pyspark.__version__:
        raise RuntimeError(
            "PySpark and Java Spark versions do not match!"
        )


    # ==================================================
    # 7. LOAD CLASS DISTRIBUTION USING SPARK
    # ==================================================

    print("\n========== LOADING CLASS DISTRIBUTION ==========")

    cleaned_df = spark.read.parquet(
        str(CLEANED_PATH)
    )

    total_records = cleaned_df.count()

    class_distribution_spark = (
        cleaned_df
        .groupBy("label")
        .count()
        .withColumn(
            "percentage",
            F.round(
                F.col("count") / F.lit(total_records) * 100,
                2
            )
        )
        .orderBy("label")
    )

    # Collect only the two aggregated class records
    class_distribution = (
        class_distribution_spark.toPandas()
    )

    print("Class distribution loaded!")


    # ==================================================
    # 8. LOAD RANDOM FOREST FEATURE IMPORTANCE
    # ==================================================

    print("\n========== LOADING FEATURE IMPORTANCE ==========")

    importance_spark = spark.read.parquet(
        str(IMPORTANCE_PATH)
    )

    importance_df = (
        importance_spark
        .orderBy(F.desc("importance"))
        .toPandas()
    )

    print("Feature importance loaded!")


finally:

    spark.stop()


# ==================================================
# 9. LOAD SPARK EVALUATION CSV FILES
# ==================================================

print("\n========== READING EVALUATION RESULTS ==========")

metrics = read_spark_csv(
    CSV_DIR / "model_metrics"
)

lr_confusion = read_spark_csv(
    CSV_DIR / "confusion_logistic_regression"
)

rf_confusion = read_spark_csv(
    CSV_DIR / "confusion_random_forest"
)

error_summary = read_spark_csv(
    CSV_DIR / "error_summary"
)

print("Evaluation files loaded successfully!")


# ==================================================
# 10. SET PLOT STYLE
# ==================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)


# ==================================================
# VISUALIZATION 1: CLASS DISTRIBUTION
# ==================================================

print("\n========== PLOT 1: CLASS DISTRIBUTION ==========")

class_distribution["class_name"] = (
    class_distribution["label"]
    .astype(int)
    .map({
        0: "Healthy",
        1: "Parkinson's"
    })
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=class_distribution,
    x="class_name",
    y="count",
    hue="class_name",
    legend=False
)

plt.title(
    "Parkinson's Dataset Class Distribution"
)

plt.xlabel("Health Status")

plt.ylabel("Number of Voice Recordings")

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")

for index, row in class_distribution.iterrows():

    ax.text(
        index,
        row["count"] + 3,
        f'{row["percentage"]}%',
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("Class distribution chart saved!")


# ==================================================
# REUSABLE CONFUSION MATRIX FUNCTION
# ==================================================

def plot_confusion_matrix(
    data,
    model_name,
    filename
):

    # Convert Spark confusion matrix output
    # into a 2 x 2 matrix for visualization.

    cm = (
        data.pivot(
            index="actual",
            columns="predicted",
            values="count"
        )
        .reindex(
            index=[0, 1],
            columns=[0, 1]
        )
        .fillna(0)
    )

    cm.index = [
        "Healthy (0)",
        "Parkinson's (1)"
    ]

    cm.columns = [
        "Healthy (0)",
        "Parkinson's (1)"
    ]

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=1,
        cbar=True
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.xlabel("Predicted Class")

    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(f"{model_name} confusion matrix saved!")


# ==================================================
# VISUALIZATION 2: LOGISTIC REGRESSION CONFUSION MATRIX
# ==================================================

print("\n========== PLOT 2: LOGISTIC REGRESSION ==========")

plot_confusion_matrix(
    lr_confusion,
    "Logistic Regression",
    "confusion_matrix_logistic_regression.png"
)


# ==================================================
# VISUALIZATION 3: RANDOM FOREST CONFUSION MATRIX
# ==================================================

print("\n========== PLOT 3: RANDOM FOREST ==========")

plot_confusion_matrix(
    rf_confusion,
    "Random Forest",
    "confusion_matrix_random_forest.png"
)


# ==================================================
# VISUALIZATION 4: MODEL PERFORMANCE COMPARISON
# ==================================================

print("\n========== PLOT 4: MODEL COMPARISON ==========")

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "balanced_accuracy"
]

# Ensure numerical metric columns

for column in metric_columns:
    metrics[column] = pd.to_numeric(
        metrics[column]
    )

metrics_long = metrics.melt(
    id_vars=["model"],
    value_vars=metric_columns,
    var_name="metric",
    value_name="score"
)

metrics_long["metric"] = (
    metrics_long["metric"]
    .str.replace("_", " ")
    .str.title()
)

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=metrics_long,
    x="metric",
    y="score",
    hue="model"
)

plt.title(
    "Logistic Regression vs Random Forest"
)

plt.xlabel("Evaluation Metrics")

plt.ylabel("Score")

plt.ylim(0, 1.08)

plt.xticks(rotation=15)

plt.legend(
    title="Classification Models"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("Model performance comparison saved!")


# ==================================================
# VISUALIZATION 5: FALSE POSITIVE / FALSE NEGATIVE
# ==================================================

print("\n========== PLOT 5: ERROR COMPARISON ==========")

if not error_summary.empty:

    # Create a comparison table
    # Rows: error types
    # Columns: classification models

    error_pivot = (
        error_summary.pivot_table(
            index="error_type",
            columns="model",
            values="misclassified_records",
            aggfunc="sum"
        )
    )

    model_order = [
        "Logistic Regression",
        "Random Forest"
    ]

    error_order = [
        "False Positive",
        "False Negative"
    ]

    error_pivot = (
        error_pivot
        .reindex(
            index=error_order,
            columns=model_order
        )
        .fillna(0)
    )

    error_plot_data = (
        error_pivot
        .reset_index()
        .melt(
            id_vars="error_type",
            var_name="model",
            value_name="count"
        )
    )

    plt.figure(figsize=(9, 5))

    ax = sns.barplot(
        data=error_plot_data,
        x="error_type",
        y="count",
        hue="model"
    )

    plt.title(
        "False Positive and False Negative Comparison"
    )

    plt.xlabel("Classification Error")

    plt.ylabel("Number of Misclassified Records")

    plt.legend(
        title="Classification Models"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "error_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print("Error comparison chart saved!")

else:

    print("No classification errors were recorded.")

    plt.figure(figsize=(8, 4))

    plt.text(
        0.5,
        0.5,
        "Both models classified all records correctly",
        ha="center",
        va="center"
    )

    plt.axis("off")

    plt.savefig(
        PLOT_DIR / "error_comparison.png",
        dpi=300
    )

    plt.close()


# ==================================================
# VISUALIZATION 6: RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

print("\n========== PLOT 6: FEATURE IMPORTANCE ==========")

top_features = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
    .head(10)
    .sort_values(
        by="importance",
        ascending=True
    )
)

plt.figure(figsize=(10, 7))

ax = sns.barplot(
    data=top_features,
    x="importance",
    y="feature"
)

plt.title(
    "Top 10 Important Voice Features - Random Forest"
)

plt.xlabel("Feature Importance")

plt.ylabel("Voice Measurement Features")

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("Feature importance chart saved!")


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n======================================")
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated visualizations:")

print("1. Class distribution")
print("2. Logistic Regression confusion matrix")
print("3. Random Forest confusion matrix")
print("4. Model performance comparison")
print("5. False positive / false negative comparison")
print("6. Random Forest feature importance")

print("\nAll charts saved inside:")

print(PLOT_DIR)

print("\nSTAGE 5 COMPLETED SUCCESSFULLY!")
