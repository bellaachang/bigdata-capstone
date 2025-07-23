from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, collect_list
from pyspark.mllib.evaluation import RankingMetrics

''' Implement popularity baseline model in Spark with ratings.
This should be simple enough to implement with some basic dataframe computations.'''


base_path = "hdfs:/user/eyw2010_nyu_edu/target/ml-latest-small" # change path

spark = SparkSession.builder.appName("popularity_based_model").getOrCreate()

train_df = spark.read.csv(f"{base_path}/train/train.csv", header=True, inferSchema=True)
test_df = spark.read.csv(f"{base_path}/test/test.csv", header=True, inferSchema=True)
val_df = spark.read.csv(f"{base_path}/validation/validation.csv", header=True, inferSchema=True)

print(f"\nSize of train_df: {train_df.count()} rows")
print(f"\nSize of test_df: {test_df.count()} rows")
print(f"\nSize of val_df: {val_df.count()} rows")

damping_factors = {10, 100, 1000, 10000} 

def train_test(train_df, test_df, damping_factor, k=100):

    # return df with average rating and number of ratings for each movie
    movie_stats = train_df.groupBy("movieId").agg(
        F.avg("rating").alias("avg_rating"),
        F.count("rating").alias("rating_count")
    )

    # apply popularity baseline formula w damping factor
    # groups train_df by movieId and calculates average rating &number of ratings
    movie_popularity_scores = movie_stats.withColumn(
        "popularity_score",
        (F.col("avg_rating") * F.col("rating_count")) / (F.col("rating_count") + F.lit(damping_factor))
    )


    # join movies from test_df with popularity scores
    test_predictions = test_df.join(movie_popularity_scores, on="movieId", how="left") \
        .withColumn("predicted_rating", F.col("popularity_score"))

    test_predictions = test_predictions \
        .withColumn("rank", row_number().over(Window.partitionBy("userId").orderBy(F.desc("predicted_rating")))) \
        .groupBy("userId") \
        .agg(collect_list("movieId").alias("predicted_items"))
    

    # create ground truth by averaging ratings per user and filtering movies above the mean rating
    mean_ratings_per_user = test_df.groupBy("userId").agg(F.mean("rating").alias("mean_rating"))
    movies_with_mean = test_df.join(mean_ratings_per_user, on="userId", how="inner")
    movies_above_mean = movies_with_mean.filter(F.col("rating") > F.col("mean_rating"))
    movies_sorted = movies_above_mean.orderBy("userId", F.desc("rating"))
    test_ground_truth = movies_sorted.groupBy("userId").agg(F.collect_list("movieId").alias("actual_items"))

    test_ranking_data = test_predictions.join(test_ground_truth, on="userId")

    ranking_rdd = test_ranking_data.rdd.map(lambda row: (row["predicted_items"], row["actual_items"]))
    metrics = RankingMetrics(ranking_rdd)

    ndcg = metrics.ndcgAt(k)
    precision = metrics.precisionAt(k)
    mean_ap = metrics.meanAveragePrecisionAt(k)
    recall = metrics.recallAt(k)

    return ndcg, precision, mean_ap, recall


results = []

for damping_factor in damping_factors:
    ndcg, precision, mean_ap, recall = train_test(train_df, val_df, damping_factor, 100)

    results.append({
        "damping_factor": damping_factor,
        "NDCG@100": ndcg,
        "Precision@100": precision,
        "MAP@100": mean_ap,
        "Recall@100": recall
    })

# Hyperparameter Tuning - Find the best damping factor based on NDCG@100
print(f"\nResults:{results}")
best_result = max(results, key=lambda x: x["NDCG@100"])
print("\nBest Damping Factor:")
print(f"Damping Factor: {best_result['damping_factor']}")
print(f"NDCG@100: {best_result['NDCG@100']:.4f}")
print(f"Precision@100: {best_result['Precision@100']:.4f}")
print(f"MAP@100: {best_result['MAP@100']:.4f}")
print(f"Recall@100: {best_result['Recall@100']:.4f}")


# Evaluate test set results (using best damping factor from validation)
best_damping = best_result['damping_factor']
ndcg, precision, mean_ap, recall = train_test(train_df, test_df, best_damping, 100)
print("\nFinal Evaluation on Test Set:")
print(f"Damping Factor: {best_damping}")
print(f"NDCG@100: {ndcg:.4f}")
print(f"Precision@100: {precision:.4f}")
print(f"MAP@100: {mean_ap:.4f}")
print(f"Recall@100: {recall:.4f}")
