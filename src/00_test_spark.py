
from pyspark.sql import SparkSession

print("Starting Apache Spark...")

spark = (
    SparkSession.builder
    .appName("Parkinsons_Project_Test")
    .master("local[2]")
    .getOrCreate()
)

print("Spark Version:", spark.version)

df = spark.range(5)

df.show()

spark.stop()

print("Spark is working successfully!")
