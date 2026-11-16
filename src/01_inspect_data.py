from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# --------------------------------------------------
# 1. INITIALIZE SPARK
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "parkinsons.data"

spark = (
    SparkSession.builder
    .appName("Parkinsons_Dataset_Inspection")
    .master("local[*]")
    .config("spark.local.dir", "D:/spark-temp")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("mode", "FAILFAST")
    .csv(str(DATA_PATH))
)

print("\n========== DATASET LOADED ==========")

print("Dataset path:", DATA_PATH)


# --------------------------------------------------
# 3. DATASET DIMENSIONS
# --------------------------------------------------

total_records = df.count()

total_columns = len(df.columns)

print("\n========== DATASET INFORMATION ==========")

print("Total records:", total_records)

print("Total columns:", total_columns)

print("Column names:")

print(df.columns)


# --------------------------------------------------
# 4. DISPLAY SAMPLE RECORDS
# --------------------------------------------------

print("\n========== FIRST 10 RECORDS ==========")

df.show(10, truncate=False)


# --------------------------------------------------
# 5. INSPECT DATA TYPES
# --------------------------------------------------

print("\n========== DATA SCHEMA ==========")

df.printSchema()


# --------------------------------------------------
# 6. IDENTIFY FEATURES AND TARGET
# --------------------------------------------------

feature_cols = [
    c for c in df.columns
    if c not in ["name", "status"]
]

print("\n========== FEATURE INFORMATION ==========")

print("Number of predictive features:", len(feature_cols))

print("Features:")

for feature in feature_cols:
    print(feature)

print("\nTarget variable: status")

print("Identifier column: name")


# --------------------------------------------------
# 7. CHECK MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")

missing_expressions = []

for field in df.schema.fields:

    col_name = field.name

    if col_name == "name":

        invalid = (
            F.col(col_name).isNull() |
            (F.trim(F.col(col_name)) == "")
        )

    else:

        numeric_value = F.col(col_name).cast("double")

        invalid = (
            numeric_value.isNull() |
            F.isnan(numeric_value)
        )

    missing_expressions.append(
        F.sum(
            F.when(invalid, 1).otherwise(0)
        ).alias(col_name)
    )


missing_summary = df.select(
    *missing_expressions
)

missing_summary.show(truncate=False)


# --------------------------------------------------
# 8. CHECK EXACT DUPLICATES
# --------------------------------------------------

print("\n========== DUPLICATE ANALYSIS ==========")

distinct_records = df.dropDuplicates().count()

duplicate_count = total_records - distinct_records

print("Exact duplicate records:", duplicate_count)


# Check duplicate recording identifiers

duplicate_names = (
    df.groupBy("name")
    .count()
    .filter(F.col("count") > 1)
)

print("Repeated recording identifiers:")

duplicate_names.show(truncate=False)


# --------------------------------------------------
# 9. TARGET CLASS DISTRIBUTION
# --------------------------------------------------

print("\n========== CLASS DISTRIBUTION ==========")

df.groupBy("status").count().orderBy("status").show()


class_distribution = (
    df.groupBy("status")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / total_records * 100,
            2
        )
    )
    .orderBy("status")
)

class_distribution.show()


# --------------------------------------------------
# 10. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n========== FEATURE STATISTICS ==========")

df.select(feature_cols).describe().show(
    truncate=False
)


# --------------------------------------------------
# 11. IDENTIFY UNIQUE SUBJECTS
# --------------------------------------------------

print("\n========== SUBJECT ANALYSIS ==========")

df = df.withColumn(
    "subject_id",
    F.regexp_replace(
        F.col("name"),
        r"_[0-9]+$",
        ""
    )
)

unique_subjects = (
    df.select("subject_id")
    .distinct()
    .count()
)

print("Unique subjects:", unique_subjects)

print("Sample subject identifiers:")

df.select(
    "name",
    "subject_id",
    "status"
).show(10, truncate=False)


# Check whether each subject has a consistent status

inconsistent_subjects = (
    df.groupBy("subject_id")
    .agg(
        F.countDistinct("status").alias("distinct_labels")
    )
    .filter(F.col("distinct_labels") > 1)
)

print("Subjects with inconsistent labels:")

inconsistent_subjects.show()


# --------------------------------------------------
# 12. CLOSE SPARK
# --------------------------------------------------

spark.stop()

print("\nDataset inspection completed successfully!")