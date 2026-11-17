
import os
import re
from pathlib import Path

# ==================================================
# 1. CONFIGURE SPARK ENVIRONMENT
# ==================================================

os.makedirs(r"D:\Temp", exist_ok=True)
os.makedirs(r"D:\spark-temp", exist_ok=True)

os.environ["TEMP"] = r"D:\Temp"
os.environ["TMP"] = r"D:\Temp"

# Use the Spark 3.5.8 installation through the short path
os.environ["SPARK_HOME"] = r"D:\PySpark35"

# Use the short Python path to avoid Windows path issues
os.environ["PYSPARK_PYTHON"] = (
    r"D:\ParkinsonsVenv\Scripts\python.exe"
)

os.environ["PYSPARK_DRIVER_PYTHON"] = (
    r"D:\ParkinsonsVenv\Scripts\python.exe"
)

# Remove conflicting Hadoop classpath preference
os.environ.pop("HADOOP_USER_CLASSPATH_FIRST", None)

# Allocate 2 GB memory to the Spark driver
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--driver-memory 2g pyspark-shell"
)


# ==================================================
# 2. IMPORT LIBRARIES
# ==================================================

import pyspark

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


# ==================================================
# 3. VERIFY SPARK CONFIGURATION
# ==================================================

if pyspark.__version__ != "3.5.8":
    raise RuntimeError(
        f"Expected PySpark 3.5.8, found {pyspark.__version__}"
    )

if not Path(r"D:\PySpark35\jars").exists():
    raise RuntimeError("Spark 3.5.8 JAR directory not found!")

if not Path(
    r"D:\ParkinsonsVenv\Scripts\python.exe"
).exists():
    raise RuntimeError("Python virtual environment path not found!")


# ==================================================
# 4. PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "parkinsons.data"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

CLEANED_PATH = PROCESSED_DIR / "cleaned_dataset"

TRAIN_PATH = PROCESSED_DIR / "train"

TEST_PATH = PROCESSED_DIR / "test"

SEED = 42

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 5. CREATE SPARK SESSION
# ==================================================

spark = (
    SparkSession.builder
    .appName("Parkinsons_Data_Preprocessing")
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

    print("\nSpark Python version:", pyspark.__version__)
    print("Spark Java runtime version:", spark.version)

    if spark.version != pyspark.__version__:
        raise RuntimeError(
            "PySpark and Java Spark versions do not match!"
        )

    max_heap = (
        spark._jvm.java.lang.Runtime
        .getRuntime()
        .maxMemory()
    )

    print(
        "Spark JVM maximum heap:",
        round(max_heap / (1024 ** 3), 2),
        "GB"
    )


    # ==================================================
    # 6. LOAD DATASET
    # ==================================================

    print("\n========== LOADING DATASET ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "FAILFAST")
        .csv(str(DATA_PATH))
    )

    original_count = raw_df.count()

    print("Dataset loaded successfully!")
    print("Original records:", original_count)


    # ==================================================
    # 7. CLEAN COLUMN NAMES
    # ==================================================

    print("\n========== CLEANING COLUMN NAMES ==========")

    def clean_column_name(name):

        name = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            name
        )

        return name.strip("_").lower()


    cleaned_columns = [
        clean_column_name(c)
        for c in raw_df.columns
    ]

    if len(set(cleaned_columns)) != len(cleaned_columns):
        raise ValueError(
            "Duplicate column names found after cleaning!"
        )

    df = raw_df.toDF(*cleaned_columns)

    print("Cleaned columns:")
    print(df.columns)


    # ==================================================
    # 8. PREPARE FEATURES AND TARGET
    # ==================================================

    print("\n========== PREPARING FEATURES ==========")

    if "name" not in df.columns or "status" not in df.columns:
        raise ValueError(
            "Required columns 'name' and 'status' are missing!"
        )

    df = df.withColumnRenamed(
        "status",
        "label"
    )

    feature_cols = [
        c for c in df.columns
        if c not in ["name", "label"]
    ]

    print("Number of predictive features:", len(feature_cols))
    print("Target column: label")

    if len(feature_cols) != 22:
        raise ValueError(
            f"Expected 22 features, found {len(feature_cols)}"
        )

    # Cast all numerical features and target to double

    df = df.select(
        F.col("name").cast("string").alias("name"),

        *[
            F.col(c).cast("double").alias(c)
            for c in feature_cols
        ],

        F.col("label").cast("double").alias("label")
    )


    # ==================================================
    # 9. REMOVE EXACT DUPLICATES
    # ==================================================

    print("\n========== DUPLICATE REMOVAL ==========")

    before_duplicates = df.count()

    df = df.dropDuplicates()

    duplicates_removed = (
        before_duplicates - df.count()
    )

    print("Duplicate records removed:", duplicates_removed)


    # ==================================================
    # 10. HANDLE MISSING VALUES
    # ==================================================

    print("\n========== MISSING VALUE HANDLING ==========")

    before_cleaning = df.count()

    required_cols = (
        ["name"] + feature_cols + ["label"]
    )

    # Remove records with null or NaN values

    df = df.na.drop(
        subset=required_cols
    )

    # Remove blank identifiers

    df = df.filter(
        F.length(F.trim(F.col("name"))) > 0
    )

    records_removed = (
        before_cleaning - df.count()
    )

    print("Records removed:", records_removed)
    print("Records remaining:", df.count())


    # Validate binary target

    invalid_labels = (
        df.filter(
            ~F.col("label").isin(0.0, 1.0)
        )
        .limit(1)
        .count()
    )

    if invalid_labels > 0:
        raise ValueError(
            "Invalid target labels found!"
        )

    print("Target labels validated: 0 and 1")


    # ==================================================
    # 11. EXTRACT SUBJECT IDENTIFIER
    # ==================================================

    print("\n========== SUBJECT IDENTIFICATION ==========")

    # Example:
    # phon_R01_S01_1 -> phon_R01_S01

    df = df.withColumn(
        "subject_id",
        F.regexp_replace(
            F.col("name"),
            r"_[0-9]+$",
            ""
        )
    )

    invalid_identifiers = (
        df.filter(
            ~F.col("name").rlike(r"_[0-9]+$")
        )
        .limit(1)
        .count()
    )

    if invalid_identifiers > 0:
        raise ValueError(
            "Unexpected recording identifier format!"
        )

    total_subjects = (
        df.select("subject_id")
        .distinct()
        .count()
    )

    print("Unique subjects:", total_subjects)


    # ==================================================
    # 12. SAVE CLEANED DATASET
    # ==================================================

    print("\n========== SAVING CLEANED DATA ==========")

    # Materialize the cleaned dataset in Parquet format.
    # Reloading it breaks the previous Spark query lineage.

    df.write \
        .mode("overwrite") \
        .parquet(str(CLEANED_PATH))

    df = spark.read.parquet(
        str(CLEANED_PATH)
    )

    print("Cleaned dataset saved successfully!")

    print("Cleaned records:", df.count())


    # ==================================================
    # 13. VALIDATE SUBJECT LABEL CONSISTENCY
    # ==================================================

    print("\n========== SUBJECT LABEL VALIDATION ==========")

    # Compare minimum and maximum target labels per subject.

    inconsistent_subjects = (
        df.groupBy("subject_id")
        .agg(
            F.min("label").alias("min_label"),
            F.max("label").alias("max_label")
        )
        .filter(
            F.col("min_label") != F.col("max_label")
        )
    )

    if inconsistent_subjects.limit(1).count() > 0:

        inconsistent_subjects.show()

        raise ValueError(
            "Some subjects have inconsistent labels!"
        )

    print("All subjects have consistent labels.")


    # ==================================================
    # 14. CREATE SUBJECT-LEVEL DATASET
    # ==================================================

    print("\n========== SUBJECT CLASS DISTRIBUTION ==========")

    subjects = (
        df.select(
            "subject_id",
            "label"
        )
        .dropDuplicates()
    )

    subjects_by_class = (
        subjects.groupBy("label")
        .count()
        .orderBy("label")
    )

    subjects_by_class.show()

    print("Total subjects:", subjects.count())


    # ==================================================
    # 15. STRATIFIED SUBJECT-LEVEL SPLITTING
    # ==================================================

    print("\n========== CREATING TRAIN/TEST SPLIT ==========")

    ranking_window = (
        Window
        .partitionBy("label")
        .orderBy(
            F.rand(SEED),
            F.col("subject_id")
        )
    )

    class_window = Window.partitionBy("label")

    ranked_subjects = (
        subjects
        .withColumn(
            "row_num",
            F.row_number().over(ranking_window)
        )
        .withColumn(
            "total_subjects",
            F.count("*").over(class_window)
        )
        .cache()
    )

    # Materialize the random ranking to keep split assignments stable

    ranked_subjects.count()

    # Allocate approximately 80% of subjects from each class to training

    train_subjects = (
        ranked_subjects
        .filter(
            F.col("row_num") <=
            F.floor(
                F.col("total_subjects") * 0.8
            )
        )
        .select("subject_id")
    )

    # Remaining subjects are assigned to testing

    test_subjects = (
        ranked_subjects
        .join(
            train_subjects,
            on="subject_id",
            how="left_anti"
        )
        .select("subject_id")
    )


    # ==================================================
    # 16. CREATE TRAINING AND TESTING DATA
    # ==================================================

    print("\n========== GENERATING DATASETS ==========")

    train_df = df.join(
        train_subjects,
        on="subject_id",
        how="inner"
    )

    test_df = df.join(
        test_subjects,
        on="subject_id",
        how="inner"
    )


    # ==================================================
    # 17. VERIFY NO SUBJECT LEAKAGE
    # ==================================================

    print("\n========== DATA LEAKAGE CHECK ==========")

    overlap = (
        train_subjects
        .join(
            test_subjects,
            on="subject_id",
            how="inner"
        )
        .limit(1)
        .count()
    )

    if overlap > 0:
        raise ValueError(
            "Subject leakage detected!"
        )

    print("No subject leakage detected!")


    # Verify both target classes exist in both partitions

    train_classes = (
        train_df.select("label")
        .distinct()
        .count()
    )

    test_classes = (
        test_df.select("label")
        .distinct()
        .count()
    )

    if train_classes != 2:
        raise ValueError(
            "Training set must contain both classes!"
        )

    if test_classes != 2:
        raise ValueError(
            "Testing set must contain both classes!"
        )


    # ==================================================
    # 18. DISPLAY TRAINING INFORMATION
    # ==================================================

    print("\n========== TRAINING INFORMATION ==========")

    train_records = train_df.count()

    train_subject_count = train_subjects.count()

    print("Training records:", train_records)
    print("Training subjects:", train_subject_count)

    print("Training class distribution:")

    train_df.groupBy("label") \
        .count() \
        .orderBy("label") \
        .show()


    # ==================================================
    # 19. DISPLAY TESTING INFORMATION
    # ==================================================

    print("\n========== TESTING INFORMATION ==========")

    test_records = test_df.count()

    test_subject_count = test_subjects.count()

    print("Testing records:", test_records)
    print("Testing subjects:", test_subject_count)

    print("Testing class distribution:")

    test_df.groupBy("label") \
        .count() \
        .orderBy("label") \
        .show()


    # ==================================================
    # 20. VERIFY ALL RECORDS ARE ACCOUNTED FOR
    # ==================================================

    print("\n========== FINAL DATA VERIFICATION ==========")

    final_count = df.count()

    print("Cleaned records:", final_count)
    print("Training records:", train_records)
    print("Testing records:", test_records)

    if train_records + test_records != final_count:
        raise ValueError(
            "Training and testing records do not match!"
        )

    print("All records accounted for!")


    # ==================================================
    # 21. EXPORT PROCESSED DATA
    # ==================================================

    print("\n========== EXPORTING PARQUET DATA ==========")

    train_df.write \
        .mode("overwrite") \
        .parquet(str(TRAIN_PATH))

    test_df.write \
        .mode("overwrite") \
        .parquet(str(TEST_PATH))

    print("Cleaned dataset:", CLEANED_PATH)
    print("Training dataset:", TRAIN_PATH)
    print("Testing dataset:", TEST_PATH)

    print("All processed datasets saved successfully!")


    # ==================================================
    # 22. FINAL SUMMARY
    # ==================================================

    print("\n========== PREPROCESSING SUMMARY ==========")

    print("Original records:", original_count)
    print("Final records:", final_count)
    print("Predictive features:", len(feature_cols))
    print("Unique subjects:", total_subjects)

    print("Training records:", train_records)
    print("Testing records:", test_records)

    print("Subject leakage: None")

    print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY!")

    ranked_subjects.unpersist()


finally:

    spark.stop()
