from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, MinHashLSH
from pyspark.sql.functions import col, size, collect_list


#spark-submit --deploy-mode client  --executor-memory 8G --driver-memory 4G --num-executors 8 --executor-cores 4 test_q1.py
#hdfs dfs -ls /user/ic2664_nyu_edu/target/ml-latest/q1
#hdfs dfs -cat /user/ic2664_nyu_edu/target/ml-latest/q1/[CSV file]

spark = SparkSession.builder.appName("movie_pairs").getOrCreate()

ratings = spark.read.csv("hdfs:/user/ic2664_nyu_edu/target/ml-latest/ratings.csv", header=True, inferSchema=True)

user_movie_sets = (
    ratings
    .select(col("userId").cast("string"), col("movieId").cast("string"))
    .groupBy("userId")
    .agg(collect_list("movieId").alias("movies"))
    .filter(size("movies") >= 5)
)

cv = CountVectorizer(inputCol="movies", outputCol="features", binary=True)
cv_model = cv.fit(user_movie_sets)
user_features_df = cv_model.transform(user_movie_sets)
user_features_df.cache()

mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = mh.fit(user_features_df)
transformed = model.transform(user_features_df)

similar_users = model.approxSimilarityJoin(transformed, transformed, 0.5, distCol="JaccardDist")

filtered_pairs = (similar_users
    .filter("datasetA.userId < datasetB.userId")
    .selectExpr("datasetA.userId as userA", "datasetB.userId as userB", "1 - JaccardDist as similarity")
    .orderBy("similarity", ascending=False)
    .limit(100))

filtered_pairs.coalesce(1).write.mode("overwrite").option("header", True).csv("hdfs:/user/ic2664_nyu_edu/target/ml-latest/q1")