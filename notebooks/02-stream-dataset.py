# Databricks notebook source
volume_path = "/Volumes/alex_m/gen_ai/pdfs/data/"
df = spark.read.parquet(f"{volume_path}/*.parquet")
print(df.count())
display(df)

# COMMAND ----------

# Infer schema from existing files
volume_path = "/Volumes/alex_m/gen_ai/pdfs/data/"
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
    .option("checkpointLocation", "/Volumes/alex_m/gen_ai/pdfs/checkpoint") \
    .table("alex_m.gen_ai.pixparse_pdfs")
    # .start()

# COMMAND ----------

streaming_query.stop()

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE alex_m.gen_ai.pixparse_pdfs
