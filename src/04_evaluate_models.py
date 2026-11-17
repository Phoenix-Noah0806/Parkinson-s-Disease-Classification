
import os
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

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    LongType
)

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator
)


# ==================================================
# 3. VERIFY SPARK ENVIRONMENT
# ==================================================

if pyspark.__version__ != "3.5.8":
    raise RuntimeError(
        f"Expected PySpark 3.5.8, found {pyspark.__version__}"
    )


# ==================================================
# 4. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_DIR / "results"

LR_PREDICTION_PATH = (
    RESULT_DIR / "predictions_logistic_regression"
)

RF_PREDICTION_PATH = (
    RESULT_DIR / "predictions_random_forest"
)

CSV_DIR = RESULT_DIR / "csv"

CSV_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 5. CREATE SPARK SESSION
# ==================================================

spark = (
    SparkSession.builder
    .appName("Parkinsons_Model_Evaluation")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.sql.constraintPropagation.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ==================================================
# 6. REUSABLE CSV EXPORT FUNCTION
# ==================================================

def export_csv(dataframe, output_path):

    (
        dataframe
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(str(output_path))
    )

    print("CSV exported:", output_path)


try:

    print("\n========== SPARK CONFIGURATION ==========")

    print("PySpark version:", pyspark.__version__)
    print("Java Spark version:", spark.version)

    if spark.version != pyspark.__version__:
        raise RuntimeError(
            "Python and Java Spark versions do not match!"
        )


    # ==================================================
    # 7. LOAD PREDICTIONS
    # ==================================================

    print("\n========== LOADING MODEL PREDICTIONS ==========")

    if not LR_PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "Logistic Regression predictions not found!"
        )

    if not RF_PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "Random Forest predictions not found!"
        )

    lr_predictions = (
        spark.read.parquet(str(LR_PREDICTION_PATH))
    )

    rf_predictions = (
        spark.read.parquet(str(RF_PREDICTION_PATH))
    )

    print(
        "Logistic Regression records:",
        lr_predictions.count()
    )

    print(
        "Random Forest records:",
        rf_predictions.count()
    )


    # Validate prediction columns

    required_columns = {
        "name",
        "subject_id",
        "label",
        "prediction",
        "parkinsons_probability"
    }

    for model_name, predictions in [
        ("Logistic Regression", lr_predictions),
        ("Random Forest", rf_predictions)
    ]:

        missing_columns = (
            required_columns - set(predictions.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{model_name} predictions are missing "
                f"columns: {missing_columns}"
            )


    # ==================================================
    # 8. PREPARE PREDICTIONS
    # ==================================================

    def prepare_predictions(dataframe):

        return dataframe.select(

            "name",
            "subject_id",

            F.col("label")
            .cast("double")
            .alias("label"),

            F.col("prediction")
            .cast("double")
            .alias("prediction"),

            F.col("parkinsons_probability")
            .cast("double")
            .alias("parkinsons_probability")
        )


    lr_predictions = prepare_predictions(
        lr_predictions
    )

    rf_predictions = prepare_predictions(
        rf_predictions
    )


    # ==================================================
    # 9. CREATE FULL CONFUSION MATRIX
    # ==================================================

    def create_confusion_matrix(predictions):

        # Count actual and predicted combinations

        counts = (
            predictions
            .groupBy(
                F.col("label").cast("int").alias("actual"),
                F.col("prediction").cast("int").alias("predicted")
            )
            .agg(
                F.count("*").alias("count")
            )
        )

        # Create both possible actual classes

        actual_schema = StructType([
            StructField("actual", IntegerType(), False)
        ])

        actual_values = spark.createDataFrame(
            [(0,), (1,)],
            schema=actual_schema
        )

        # Create both possible predicted classes

        predicted_schema = StructType([
            StructField("predicted", IntegerType(), False)
        ])

        predicted_values = spark.createDataFrame(
            [(0,), (1,)],
            schema=predicted_schema
        )

        # Generate all four possible combinations

        complete_grid = actual_values.crossJoin(
            predicted_values
        )

        # Join actual counts and fill missing cells with zero

        confusion_matrix = (
            complete_grid
            .join(
                counts,
                on=["actual", "predicted"],
                how="left"
            )
            .fillna(
                0,
                subset=["count"]
            )
            .orderBy(
                "actual",
                "predicted"
            )
        )

        return confusion_matrix


    # ==================================================
    # 10. EVALUATE EACH MODEL USING SPARK MLLIB
    # ==================================================

    def evaluate_model(predictions, model_name):

        print("\n======================================")
        print("EVALUATING:", model_name)
        print("======================================")

        # Spark MLlib evaluator

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction"
        )

        # Accuracy

        accuracy = evaluator.setMetricName(
            "accuracy"
        ).evaluate(predictions)

        # Precision for Parkinson's class (1)

        precision = (
            evaluator
            .setMetricName("precisionByLabel")
            .setMetricLabel(1.0)
            .evaluate(predictions)
        )

        # Recall for Parkinson's class (1)

        recall = (
            evaluator
            .setMetricName("recallByLabel")
            .setMetricLabel(1.0)
            .evaluate(predictions)
        )

        # F1-score for Parkinson's class (1)

        f1_score = (
            evaluator
            .setMetricName("fMeasureByLabel")
            .setMetricLabel(1.0)
            .setBeta(1.0)
            .evaluate(predictions)
        )


        # ==================================================
        # 11. CONFUSION MATRIX COUNTS
        # ==================================================

        confusion_matrix = create_confusion_matrix(
            predictions
        )

        print("\nConfusion Matrix:")

        confusion_matrix.show()


        # Only four cells are collected to the Python driver

        cell_values = {
            (row["actual"], row["predicted"]): int(row["count"])
            for row in confusion_matrix.collect()
        }

        tn = cell_values.get((0, 0), 0)

        fp = cell_values.get((0, 1), 0)

        fn = cell_values.get((1, 0), 0)

        tp = cell_values.get((1, 1), 0)


        # Specificity

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        # Balanced accuracy (supplementary metric)

        balanced_accuracy = (
            recall + specificity
        ) / 2


        # ==================================================
        # 12. DISPLAY MODEL RESULTS
        # ==================================================

        print("\nEvaluation Results:")

        print("Accuracy:", round(accuracy, 4))

        print("Precision:", round(precision, 4))

        print("Recall:", round(recall, 4))

        print("F1-score:", round(f1_score, 4))

        print("Specificity:", round(specificity, 4))

        print(
            "Balanced Accuracy:",
            round(balanced_accuracy, 4)
        )

        print("\nConfusion Matrix Values:")

        print("True Negatives:", tn)

        print("False Positives:", fp)

        print("False Negatives:", fn)

        print("True Positives:", tp)


        # Return the results and confusion matrix

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "test_records": predictions.count(),
            "test_subjects": (
                predictions
                .select("subject_id")
                .distinct()
                .count()
            )
        }

        return metrics, confusion_matrix


    # ==================================================
    # 13. EVALUATE LOGISTIC REGRESSION
    # ==================================================

    lr_metrics, lr_confusion = evaluate_model(
        lr_predictions,
        "Logistic Regression"
    )


    # ==================================================
    # 14. EVALUATE RANDOM FOREST
    # ==================================================

    rf_metrics, rf_confusion = evaluate_model(
        rf_predictions,
        "Random Forest"
    )


    # ==================================================
    # 15. CREATE MODEL COMPARISON DATAFRAME
    # ==================================================

    print("\n========== MODEL COMPARISON ==========")

    metrics_schema = StructType([

        StructField("model", StringType(), False),

        StructField("accuracy", DoubleType(), False),

        StructField("precision", DoubleType(), False),

        StructField("recall", DoubleType(), False),

        StructField("f1_score", DoubleType(), False),

        StructField("specificity", DoubleType(), False),

        StructField("balanced_accuracy", DoubleType(), False),

        StructField("tn", LongType(), False),

        StructField("fp", LongType(), False),

        StructField("fn", LongType(), False),

        StructField("tp", LongType(), False),

        StructField("test_records", LongType(), False),

        StructField("test_subjects", LongType(), False)

    ])


    metrics_rows = [

        tuple(lr_metrics.values()),

        tuple(rf_metrics.values())

    ]

    metrics_df = spark.createDataFrame(
        metrics_rows,
        schema=metrics_schema
    )

    metrics_df.show(
        truncate=False
    )


    # ==================================================
    # 16. EXTRACT FALSE POSITIVES AND FALSE NEGATIVES
    # ==================================================

    print("\n========== ERROR ANALYSIS ==========")

    def extract_errors(predictions, model_name):

        errors = (
            predictions
            .filter(
                F.col("label") != F.col("prediction")
            )
            .withColumn(
                "model",
                F.lit(model_name)
            )
            .withColumn(
                "error_type",
                F.when(
                    (F.col("label") == 1) &
                    (F.col("prediction") == 0),
                    F.lit("False Negative")
                ).otherwise(
                    F.lit("False Positive")
                )
            )
            .select(
                "model",
                "name",
                "subject_id",
                "label",
                "prediction",
                "error_type",
                "parkinsons_probability"
            )
        )

        print("\nMisclassified records:", model_name)

        errors.show(
            truncate=False
        )

        return errors


    lr_errors = extract_errors(
        lr_predictions,
        "Logistic Regression"
    )

    rf_errors = extract_errors(
        rf_predictions,
        "Random Forest"
    )

    all_errors = lr_errors.unionByName(
        rf_errors
    )


    # ==================================================
    # 17. SUMMARIZE CLASSIFICATION ERRORS
    # ==================================================

    error_summary = (
        all_errors
        .groupBy(
            "model",
            "error_type"
        )
        .agg(
            F.count("*").alias("misclassified_records"),

            F.countDistinct("subject_id").alias(
                "affected_subjects"
            )
        )
        .orderBy(
            "model",
            "error_type"
        )
    )

    print("\n========== ERROR SUMMARY ==========")

    error_summary.show(
        truncate=False
    )


    # ==================================================
    # 18. EXPORT EVALUATION RESULTS
    # ==================================================

    print("\n========== EXPORTING RESULTS TO CSV ==========")


    # Export model comparison

    export_csv(
        metrics_df,
        CSV_DIR / "model_metrics"
    )


    # Export confusion matrices

    export_csv(
        lr_confusion,
        CSV_DIR / "confusion_logistic_regression"
    )

    export_csv(
        rf_confusion,
        CSV_DIR / "confusion_random_forest"
    )


    # Export predictions

    export_csv(
        lr_predictions,
        CSV_DIR / "predictions_logistic_regression"
    )

    export_csv(
        rf_predictions,
        CSV_DIR / "predictions_random_forest"
    )


    # Export classification errors

    export_csv(
        all_errors,
        CSV_DIR / "misclassified_records"
    )


    # Export error summary

    export_csv(
        error_summary,
        CSV_DIR / "error_summary"
    )


    # ==================================================
    # 19. FINAL SUMMARY
    # ==================================================

    print("\n======================================")
    print("MODEL EVALUATION COMPLETED")
    print("======================================")

    print("\nEvaluated models:")

    print("1. Logistic Regression")

    print("2. Random Forest")

    print("\nModel comparison exported to:")

    print(CSV_DIR / "model_metrics")

    print("\nConfusion matrices exported.")

    print("Prediction CSV files exported.")

    print("False-positive and false-negative records exported.")

    print("\nSTAGE 4 COMPLETED SUCCESSFULLY!")


finally:

    spark.stop()
