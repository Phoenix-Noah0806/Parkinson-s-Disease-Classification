
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
    DoubleType
)

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler
)

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

from pyspark.ml.functions import vector_to_array


# ==================================================
# 3. VERIFY ENVIRONMENT
# ==================================================

if pyspark.__version__ != "3.5.8":
    raise RuntimeError(
        f"Expected PySpark 3.5.8, found {pyspark.__version__}"
    )

if not Path(r"D:\PySpark35\jars").exists():
    raise RuntimeError("Spark JAR directory not found!")

if not Path(
    r"D:\ParkinsonsVenv\Scripts\python.exe"
).exists():
    raise RuntimeError("Virtual environment Python not found!")


# ==================================================
# 4. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_DIR / "data" / "processed" / "train"
)

TEST_PATH = (
    PROJECT_DIR / "data" / "processed" / "test"
)

MODEL_DIR = PROJECT_DIR / "models"

RESULT_DIR = PROJECT_DIR / "results"

LR_MODEL_PATH = MODEL_DIR / "logistic_regression"

RF_MODEL_PATH = MODEL_DIR / "random_forest"

LR_PREDICTION_PATH = (
    RESULT_DIR / "predictions_logistic_regression"
)

RF_PREDICTION_PATH = (
    RESULT_DIR / "predictions_random_forest"
)

IMPORTANCE_PATH = (
    RESULT_DIR / "random_forest_feature_importance"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 5. CREATE SPARK SESSION
# ==================================================

spark = (
    SparkSession.builder
    .appName("Parkinsons_Model_Training")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.sql.constraintPropagation.enabled", "false")
    .config("spark.sql.debug.maxToStringFields", "100")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:

    print("\n========== SPARK CONFIGURATION ==========")

    print("Python PySpark version:", pyspark.__version__)
    print("Java Spark runtime version:", spark.version)

    if spark.version != pyspark.__version__:
        raise RuntimeError(
            "Spark versions do not match!"
        )

    max_heap = (
        spark._jvm.java.lang.Runtime
        .getRuntime()
        .maxMemory()
    )

    print(
        "JVM maximum heap:",
        round(max_heap / (1024 ** 3), 2),
        "GB"
    )


    # ==================================================
    # 6. LOAD PROCESSED DATA
    # ==================================================

    print("\n========== LOADING PROCESSED DATA ==========")

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed train/test datasets not found. "
            "Run 02_prepare_data.py first."
        )

    train_df = spark.read.parquet(
        str(TRAIN_PATH)
    )

    test_df = spark.read.parquet(
        str(TEST_PATH)
    )

    print("Training records:", train_df.count())
    print("Testing records:", test_df.count())


    # ==================================================
    # 7. IDENTIFY FEATURE COLUMNS
    # ==================================================

    print("\n========== FEATURE PREPARATION ==========")

    excluded_cols = [
        "name",
        "subject_id",
        "label"
    ]

    feature_cols = [
        c for c in train_df.columns
        if c not in excluded_cols
    ]

    print("Number of features:", len(feature_cols))

    print("Feature columns:")

    print(feature_cols)

    if len(feature_cols) != 22:
        raise ValueError(
            f"Expected 22 features, found {len(feature_cols)}"
        )

    if set(feature_cols) != {
        c for c in test_df.columns
        if c not in excluded_cols
    }:
        raise ValueError(
            "Training and testing feature columns do not match!"
        )

    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Duplicate feature columns found!")

    print("Target variable: label")


    # ==================================================
    # 8. MODEL 1 - LOGISTIC REGRESSION
    # ==================================================

    print("\n======================================")
    print("MODEL 1: LOGISTIC REGRESSION")
    print("======================================")


    # Step 1: Assemble numerical features

    lr_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="error"
    )


    # Step 2: Standardize numerical features

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )


    # Step 3: Create Logistic Regression classifier

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0
    )


    # Step 4: Build ML pipeline

    lr_pipeline = Pipeline(
        stages=[
            lr_assembler,
            scaler,
            lr
        ]
    )


    # Step 5: Train using training data only

    print("\nTraining Logistic Regression...")

    lr_model = lr_pipeline.fit(train_df)

    print("Logistic Regression trained successfully!")


    # ==================================================
    # 9. LOGISTIC REGRESSION PREDICTIONS
    # ==================================================

    print("\n========== LOGISTIC REGRESSION PREDICTIONS ==========")

    lr_raw_predictions = lr_model.transform(
        test_df
    )

    lr_predictions = (
        lr_raw_predictions
        .select(
            "name",
            "subject_id",

            F.col("label")
            .cast("int")
            .alias("label"),

            F.col("prediction")
            .cast("int")
            .alias("prediction"),

            vector_to_array("probability")[1]
            .alias("parkinsons_probability")
        )
    )

    print("\nSample Logistic Regression predictions:")

    lr_predictions.show(
        10,
        truncate=False
    )


    # ==================================================
    # 10. SAVE LOGISTIC REGRESSION MODEL
    # ==================================================

    print("\n========== SAVING LOGISTIC REGRESSION ==========")

    lr_model.write() \
        .overwrite() \
        .save(str(LR_MODEL_PATH))

    print("Logistic Regression model saved!")

    print("Location:", LR_MODEL_PATH)


    # ==================================================
    # 11. MODEL 2 - RANDOM FOREST
    # ==================================================

    print("\n======================================")
    print("MODEL 2: RANDOM FOREST")
    print("======================================")


    # Step 1: Assemble features

    rf_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="error"
    )


    # Step 2: Initialize Random Forest

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=5,
        seed=42
    )


    # Step 3: Build pipeline

    rf_pipeline = Pipeline(
        stages=[
            rf_assembler,
            rf
        ]
    )


    # Step 4: Train model

    print("\nTraining Random Forest...")

    rf_model = rf_pipeline.fit(train_df)

    print("Random Forest trained successfully!")


    # ==================================================
    # 12. RANDOM FOREST PREDICTIONS
    # ==================================================

    print("\n========== RANDOM FOREST PREDICTIONS ==========")

    rf_raw_predictions = rf_model.transform(
        test_df
    )

    rf_predictions = (
        rf_raw_predictions
        .select(
            "name",
            "subject_id",

            F.col("label")
            .cast("int")
            .alias("label"),

            F.col("prediction")
            .cast("int")
            .alias("prediction"),

            vector_to_array("probability")[1]
            .alias("parkinsons_probability")
        )
    )

    print("\nSample Random Forest predictions:")

    rf_predictions.show(
        10,
        truncate=False
    )


    # ==================================================
    # 13. SAVE RANDOM FOREST MODEL
    # ==================================================

    print("\n========== SAVING RANDOM FOREST ==========")

    rf_model.write() \
        .overwrite() \
        .save(str(RF_MODEL_PATH))

    print("Random Forest model saved!")

    print("Location:", RF_MODEL_PATH)


    # ==================================================
    # 14. SAVE PREDICTIONS AS PARQUET
    # ==================================================

    print("\n========== SAVING MODEL PREDICTIONS ==========")

    lr_predictions.write \
        .mode("overwrite") \
        .parquet(str(LR_PREDICTION_PATH))

    rf_predictions.write \
        .mode("overwrite") \
        .parquet(str(RF_PREDICTION_PATH))

    print("Logistic Regression predictions saved!")

    print("Random Forest predictions saved!")


    # ==================================================
    # 15. RANDOM FOREST FEATURE IMPORTANCE
    # ==================================================

    print("\n========== FEATURE IMPORTANCE ==========")

    # Extract the trained Random Forest classifier
    # from the PipelineModel.

    rf_classifier_model = rf_model.stages[-1]

    importance_values = (
        rf_classifier_model.featureImportances
    )

    importance_rows = [
        (
            feature_cols[i],
            float(importance_values[i])
        )
        for i in range(len(feature_cols))
    ]


    # IMPORTANT:
    # Define the schema explicitly.
    # This avoids Spark's automatic schema-inference issue.

    importance_schema = StructType([

        StructField(
            "feature",
            StringType(),
            False
        ),

        StructField(
            "importance",
            DoubleType(),
            False
        )

    ])


    importance_df = spark.createDataFrame(
        importance_rows,
        schema=importance_schema
    )

    importance_df = importance_df.orderBy(
        F.desc("importance")
    )

    print("\nTop 10 important voice features:")

    importance_df.show(
        10,
        truncate=False
    )


    # Save feature importance

    importance_df.write \
        .mode("overwrite") \
        .parquet(str(IMPORTANCE_PATH))

    print("Feature importance saved!")

    print("Location:", IMPORTANCE_PATH)


    # ==================================================
    # 16. FINAL SUMMARY
    # ==================================================

    print("\n======================================")
    print("MODEL TRAINING COMPLETED")
    print("======================================")

    print("\nModels trained:")

    print("1. Logistic Regression")
    print("2. Random Forest")

    print("\nSaved models:")

    print(LR_MODEL_PATH)
    print(RF_MODEL_PATH)

    print("\nSaved predictions:")

    print(LR_PREDICTION_PATH)
    print(RF_PREDICTION_PATH)

    print("\nSaved feature importance:")

    print(IMPORTANCE_PATH)

    print("\nTesting records:", test_df.count())

    print("\nSTAGE 3 COMPLETED SUCCESSFULLY!")


finally:

    spark.stop()
