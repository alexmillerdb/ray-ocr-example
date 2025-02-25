# Databricks notebook source
# MAGIC %run ./00-setup-env

# COMMAND ----------
from pathlib import Path

volume_path = Path(f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/data")
checkpoints_path = Path(f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/checkpoints")
delta_table_name = f"{UC_CATALOG}.{UC_SCHEMA}.pixparse_pdfs"
df = spark.read.parquet(f"{volume_path}/*.parquet")
print(df.count())
display(df)

# COMMAND ----------

# Infer schema from existing files
static_df = spark.read.format("parquet").load(f"{volume_path}/*.parquet")
schema = static_df.schema

# Read as a stream from parquet files with the inferred schema
streaming_df = spark.readStream \
    .format("parquet") \
    .schema(schema) \
    .option("maxFilesPerTrigger", 100) \
    .load(f"{volume_path}/*.parquet")

# Write as a streaming Delta table
streaming_query = streaming_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", checkpoints_path) \
    .option("trigger", "availableNow") \
    .table(delta_table_name)

streaming_query.awaitTermination()