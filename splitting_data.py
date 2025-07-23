from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

''' Partition the ratings csv data into training, validation, and test sets '''
base_path = "hdfs:/user/eyw2010_nyu_edu/target/ml-latest-small" # change path
train_path = f"{base_path}/train"
val_path = f"{base_path}/validation"
test_path = f"{base_path}/test"

spark = SparkSession.builder.appName("split_csvs").getOrCreate()

ratings_df = spark.read.csv(f"{base_path}/ratings.csv", header=True, inferSchema=True)

# remove rows with null values from ratings
ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"])

# filter out movies with less than 5 ratings
rating_counts = ratings_df.groupBy("movieId").agg(F.count("rating").alias("rating_count"))
popular_movies = rating_counts.filter(F.col("rating_count") >= 5).select("movieId")
filtered_ratings_df = ratings_df.join(popular_movies, on="movieId", how="inner")

# split into 80% train, 10% val, 10% test
train_df, val_df, test_df = filtered_ratings_df.randomSplit([0.8, 0.1, 0.1], seed=42)

# split validation and test into 40% and 60% subsets based on movieId
val_movie_ids = val_df.select("movieId").distinct()
test_movie_ids = test_df.select("movieId").distinct()
val_40, val_60 = val_movie_ids.randomSplit([0.4, 0.6], seed=42)
test_40, test_60 = test_movie_ids.randomSplit([0.4, 0.6], seed=42)

# join 40% subsets back into the training dataset to mitigate cold start problem
val_40_train = val_df.join(val_40, on="movieId", how="inner")
test_40_train = test_df.join(test_40, on="movieId", how="inner")
train_df = train_df.union(val_40_train).union(test_40_train)

# keep 60% of the remaining data as new validation and test datasets
val_df = val_df.join(val_60, on="movieId", how="inner")
test_df = test_df.join(test_60, on="movieId", how="inner")

print(f"\nSPLIT SIZES: ")
print(f"Size of train_df: {train_df.count()} rows")
print(f"Size of val_df: {val_df.count()} rows")
print(f"Size of test_df: {test_df.count()} rows")
print(f"Total: {train_df.count() + val_df.count() + test_df.count()} rows")

# write spark dataframes to csvs
train_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{train_path}")
val_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{val_path}")
test_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{test_path}")

# rename the part file to the desired name
def rename_part_file(output_dir, new_name):
    files = os.popen(f"hdfs dfs -ls {output_dir}").read().splitlines()
    part_file = [line.split()[-1] for line in files if "part-" in line]
    if part_file:
        os.system(f"hdfs dfs -mv {part_file[0]} {output_dir}/{new_name}")

rename_part_file(train_path, "train.csv")
rename_part_file(val_path, "validation.csv")
rename_part_file(test_path, "test.csv")
