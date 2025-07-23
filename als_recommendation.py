from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql.functions import col, collect_list, when, desc, rank, mean
from pyspark.sql.window import Window


spark = SparkSession.builder.appName("ALSRecommendation").getOrCreate()
val_df = spark.read.csv("hdfs:/user/kef7529_nyu_edu/target/ml-latest/validation/validation.csv", header=True, inferSchema=True)
test_df = spark.read.csv("hdfs:/user/kef7529_nyu_edu/target/ml-latest/test/test.csv", header=True, inferSchema=True)
train_df = spark.read.csv("hdfs:/user/kef7529_nyu_edu/target/ml-latest/train/train.csv", header=True, inferSchema=True)

def train_test_als(train_df, test_df, r=10, reg_param=0.1, max_iter=10, k=100):
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=r, maxIter=max_iter, regParam=reg_param, coldStartStrategy="drop")
    model = als.fit(train_df)

    # Top-k recommendations per user
    predictions = model.transform(test_df) 
    
    window_spec = Window.partitionBy('userId').orderBy(desc('prediction'))
    ranked_predictions = predictions.withColumn('rank', rank().over(window_spec))
    top_100_per_user = ranked_predictions.filter(col('rank') <= 100) # change if we want to change k
    top_100_per_user = top_100_per_user.drop(col('rank'))

    top_100_per_user_rdd = top_100_per_user.rdd.map(lambda row: (row['userId'], row['movieId'])).groupByKey().mapValues(list)
    
    mean_ratings_per_user = test_df.groupBy("userId").agg(F.mean("rating").alias("mean_rating"))
    movies_with_mean = test_df.join(mean_ratings_per_user, on="userId", how="inner")
    movies_above_mean = movies_with_mean.filter(F.col("rating") > F.col("mean_rating"))
    movies_above_mean_rdd = movies_above_mean.rdd.map(lambda row: (row['userId'], row['movieId'])).groupByKey().mapValues(list)

    preds_and_labels = top_100_per_user_rdd.join(movies_above_mean_rdd).map(lambda row: (row[1][0], row[1][1])).collect()
    preds_and_labels_par = spark.sparkContext.parallelize(preds_and_labels)

    # ranking evaluation
    metrics = RankingMetrics(preds_and_labels_par)

    ndcg = metrics.ndcgAt(k)
    precision = metrics.precisionAt(k)
    mean_ap = metrics.meanAveragePrecisionAt(k)
    recall = metrics.recallAt(k)

    return ndcg, precision, mean_ap, recall


ranks = [10, 100, 150]
reg_params = [0.01, 0.05, 0.1]
results = []

for r in ranks:
    for reg in reg_params:
        print(f"\nTraining ALS with rank={r}, regParam={reg}, maxIter=10")
        ndcg, precision, mean_ap, recall = train_test_als(train_df, val_df, r, reg, 10, 100)

        results.append({
            "rank": r,
            "regParam": reg,
            "NDCG@100": ndcg,
            "Precision@100": precision,
            "MAP@100": mean_ap,
            "Recall@100": recall
        })
        print(results)

# Pick best based on NDCG
best_result = max(results, key=lambda x: x["NDCG@100"])
print("\nBest Hyperparameters:")
print(best_result)

# Final Test Set Evaluation
ndcg, precision, mean_ap, recall = train_test_als(train_df, test_df, best_result["rank"], best_result["regParam"], 10, 100)
print("\nFinal Evaluation on Test Set:")
print(f"NDCG@100: {ndcg:.4f}")
print(f"Precision@100: {precision:.4f}")
print(f"MAP@100: {mean_ap:.4f}")
print(f"Recall@100: {recall:.4f}")




