# Databricks notebook source exported at Tue, 28 Jun 2016 11:09:20 UTC
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# COMMAND ----------

df = sqlContext.read.format('com.databricks.spark.csv')\
    .option('header', 'true')\
    .option('inferschema', 'true')\
    .option('mode', 'DROPMALFORMED')\
    .load('/mnt/calvindudek/apollo/fake_apollo.csv')

df.printSchema()
df.registerTempTable("apollo")

training_data = sqlContext.sql("""
SELECT
  source_cli_hashed,
  masked_nino_hashed,
  year_of_birth,
  median_chargedur,
  median_time_to_ans,
  median_actual_call_dur,
  current_benefit_number,
  current_benefit_bundle,
  past_benefit_number,
  past_benefit_bundle,
  call_frequency,
  time_since_last_call_days,
  number_services_called,
  services_called,
  time_on_benefits_days,
  time_on_benefits_days / (2016 - year_of_birth) * 365 AS percent_of_working_life_on_benefits,
  current_benefit_number - past_benefit_number AS delta_current_vs_past_benefit,
  2016 - year_of_birth AS age
  
FROM
  apollo
LIMIT 1000""")

# COMMAND ----------

training_data.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering + Pipeline

# COMMAND ----------

from pyspark.ml.feature import *
from pyspark.ml import Pipeline

# Splits for the bucketizer
splits = [-float("inf"), 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, float("inf")]

# string_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
# gender = OneHotEncoder(inputCol="gender_index", outputCol="adj_gender")
# age = OneHotEncoder(inputCol="age", outputCol="adj_age")

age = Bucketizer(splits=splits, inputCol="age", outputCol="feat_age")

# benefit transformation
current_benefits = Tokenizer(inputCol="current_benefit_bundle", outputCol="tok_current_benefit_bundle")
previous_benefits = Tokenizer(inputCol="previous_benefit_bundle", outputCol="tok_previous_benefit_bundle")
previous_services = Tokenizer(inputCol="services_called", outputCol="tok_services_called")
htf_current_benefits = HashingTF(inputCol="tok_current_benefit_bundle", outputCol="htf_current_benefit_bundle", numFeatures=100)
htf_previous_benefits = HashingTF(inputCol="tok_previous_benefit_bundle", outputCol="htf_previous_benefit_bundle", numFeatures=100)
htf_previous_services = HashingTF(inputCol="tok_services_called", outputCol="htf_services_called", numFeatures=100)
idf_current_benefits = IDF(inputCol="htf_current_benefit_bundle", outputCol="idf_current_benefit_bundle")
idf_previous_benefits = IDF(inputCol="htf_previous_benefit_bundle", outputCol="idf_previous_benefit_bundle")
idf_previous_services = IDF(inputCol="htf_services_called", outputCol="idf_services_called")

# putting the vector together
vector = VectorAssembler(inputCols=["feat_age", "idf_current_benefit_bundle", "idf_previous_benefit_bundle", 
                                         "idf_services_called", "median_chargedur", "median_time_to_ans",
                                         "median_actual_call_dur", "current_benefit_number", "past_benefit_number", 
                                         "delta_current_vs_past_benefit", "call_frequency", "time_since_last_call_days", 
                                         "number_services_called", "percent_of_working_life_on_benefits"
                          ], outputCol="features")

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

kmeans = KMeans(k=5, seed=10, featuresCol="scaled_features", predictionCol="prediction")

# Building pipelines for Logistic Regression
pipeline = Pipeline(stages=[string_indexer, gender, current_benefits, previous_benefits, 
                            previous_services, htf_current_benefits, htf_previous_benefits,
                            htf_previous_services, idf_current_benefits, idf_previous_benefits,
                            idf_previous_services, vector, scaler, kmeans])

# COMMAND ----------

model = pipeline.fit(training_data)

# COMMAND ----------

model.transform(training_data).take(5)
