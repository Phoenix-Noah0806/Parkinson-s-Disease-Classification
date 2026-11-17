import pyspark

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Spark_Version_Check")
    .master("local[2]")
    .getOrCreate()
)

print("Python PySpark version:", pyspark.__version__)
print("Java Spark runtime version:", spark.version)
print("PySpark location:", pyspark.__file__)

spark.stop()

