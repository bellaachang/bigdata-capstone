from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import random

spark = SparkSession.builder.appName("q2_correlations").getOrCreate()

ratings = spark.read.csv("hdfs:/user/ic2664_nyu_edu/target/ml-latest/ratings.csv", header=True, inferSchema=True)
user_counts = ratings.groupBy("userId").count()
filtered_users = user_counts.filter(F.col("count") >= 5).select("userId")
ratings = ratings.join(filtered_users, on="userId")

filtered_pairs = spark.read.csv("hdfs:/user/ic2664_nyu_edu/target/ml-latest/q1/q1_pairs.csv", header=True, inferSchema=True)

def average_pairwise_correlation(filtered_pairs, ratings_df):
    '''
    Compute the average pairwise correlation between user pairs in filtered_pairs
    and their ratings in ratings_df.

    Parameters:
    filtered_pairs (Spark DataFrame): Spark DataFrame containing user pairs and their similarity scores.
    ratings_df (Spark DataFrame): Spark DataFrame containing user ratings.
    '''
    joined = filtered_pairs.join(
        ratings_df.alias("ratingsA"), filtered_pairs.userA == F.col("ratingsA.userId")
    ).join(
        ratings_df.alias("ratingsB"), filtered_pairs.userB == F.col("ratingsB.userId")
    ).select(
        F.col("userA"), F.col("userB"),
        F.col("ratingsA.movieId").alias("movieIdA"),
        F.col("ratingsA.rating").alias("ratingA"),
        F.col("ratingsB.movieId").alias("movieIdB"),
        F.col("ratingsB.rating").alias("ratingB")
    ).filter(F.col("movieIdA") == F.col("movieIdB"))

    # Group by user pairs and compute correlation
    correlations = joined.groupBy("userA", "userB").agg(
        F.corr("ratingA", "ratingB").alias("correlation")
    ).select("correlation").filter(F.col("correlation").isNotNull())

    # Compute the average correlation
    avg_corr = correlations.agg(F.avg("correlation").alias("avg_corr")).collect()[0]["avg_corr"]
    return avg_corr

avg_corr = average_pairwise_correlation(filtered_pairs, ratings)
print(f"Average correlation : {avg_corr:.4f}")

def random_average_pairwise_correlation(ratings_df, num_pairs=100):
    '''
    Compute the average pairwise correlation between random user pairs in ratings_df,
    considering only users whose movie ratings have a variance above 0.

    Parameters:
    ratings_df (Spark DataFrame): Spark DataFrame containing user ratings.
    num_pairs (int): Number of random pairs to generate.
    '''

    # Filter users whose ratings have a variance above 0
    user_variances = ratings_df.groupBy("userId").agg(F.variance("rating").alias("rating_variance"))
    valid_users = user_variances.filter(F.col("rating_variance") > 0).select("userId")

    # Get all unique valid users
    unique_users = valid_users.rdd.flatMap(lambda x: x).collect()

    # Randomly sample pairs of users from unique_users
    sampled_pairs = []
    while len(sampled_pairs) < num_pairs:
        userA, userB = random.sample(unique_users, 2)
        if userA != userB and (userB, userA) not in sampled_pairs:
            sampled_pairs.append((userA, userB))
    
    # Convert sampled pairs to a DataFrame
    sampled_pairs_df = spark.createDataFrame(sampled_pairs, ["userA", "userB"])

    # Join with ratings_df to get the ratings for each user pair
    sampled_ratings = sampled_pairs_df.join(ratings_df.alias("ratingsA"), sampled_pairs_df.userA == F.col("ratingsA.userId")) \
        .join(ratings_df.alias("ratingsB"), sampled_pairs_df.userB == F.col("ratingsB.userId")) \
        .select(
            F.col("userA"), F.col("userB"),
            F.col("ratingsA.movieId").alias("movieIdA"),
            F.col("ratingsA.rating").alias("ratingA"),
            F.col("ratingsB.movieId").alias("movieIdB"),
            F.col("ratingsB.rating").alias("ratingB")
        ).filter(F.col("movieIdA") == F.col("movieIdB"))

    # Group by user pairs and compute correlation
    correlations = sampled_ratings.groupBy("userA", "userB").agg(
        F.corr("ratingA", "ratingB").alias("correlation")
    ).select("correlation").filter(F.col("correlation").isNotNull())

    # Compute the average correlation
    avg_corr = correlations.agg(F.avg("correlation").alias("avg_corr")).collect()[0]["avg_corr"]

    return avg_corr

random_avg_corr = random_average_pairwise_correlation(ratings, 100)
print(f"Random average correlation : {random_avg_corr:.4f}")

# To calculate the average pairwise correlation, we defined a function that takes in a dataframe of user IDs as well as the ratings.csv dataframe. Then, it selects ratings for movies in which both users have in common and joins them. After this, it calculates the correlation between the two users if there are two or more movies. We made the assumption that users need to contain more than 1 movie rating in common as correlation on just 1 movie wouldn’t be meaningful. Then the correlation is appended to a list of correlations, and the average is calculated. For generating 100 random pairs, we defined a function that takes in the ratings.csv dataframe and randomly samples 100 pairs of users. We then calculate the average correlation for these random pairs using the same method as above. Finally, we print out the average correlation for both the filtered pairs and the random pairs.