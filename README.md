# MovieLens Recommender & Customer Segmentation System  
**NYU Big Data Capstone | DSGA1004**

## Overview  
This project involved designing and implementing a scalable movie recommendation and customer segmentation pipeline using the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset (330K users, 86K movies). The solution leveraged distributed computing and collaborative filtering techniques to surface personalized recommendations and identify user similarity patterns at scale.

We used **Apache Spark** on **Google Cloud Dataproc (HDFS)** to handle data processing and model training, focusing on real-world data engineering and machine learning challenges.

---

## Objectives  
- Build a **collaborative filtering recommender system** using Spark's ALS model  
- Segment users by identifying the **top 100 most similar user pairs** ("movie twins")  
- Benchmark against a **popularity-based baseline model**  
- Evaluate model performance using **ranking metrics** (Precision@K, MAP@K)  
- **Prototype and scale** pipeline from small (~600 users) to full dataset (~330K users)

---

## Datasets  
- **Small**: ~9K movies, ~600 users  
- **Full**: ~86K movies, ~330K users  
- Includes: user ratings, tags, and "tag genome" metadata  
- Data stored in CSV format; managed via HDFS in zipped format

---

## Components  

### 1. Customer Segmentation  
- Implemented a **MinHashLSH-based similarity search** to find top 100 "movie twins"  
- Operationalized user similarity based on overlap in rated movie sets  
- **Validated** pairs using correlation of actual numerical ratings vs. 100 random user pairs

### 2. Recommender System  
- **Partitioned** data into training, validation, and test sets via scripted pipeline  
- Built a **popularity-based baseline** recommender for benchmarking  
- Trained **Spark ALS (Alternating Least Squares)** collaborative filtering model  
  - Tuned `rank` and `regularization` hyperparameters on the validation set  
  - Focused on **top-K recommendation quality**  
- **Evaluation**  
  - Used Spark’s `RankingMetrics` to compute Precision@100, MAP@100  
  - Compared ALS model vs. baseline across both validation and test sets

---

## Technologies Used  
- `Apache Spark (PySpark)`  
- `Google Cloud Dataproc & HDFS`  
- `MinHashLSH` for approximate similarity search  
- `pyspark.ml.recommendation.ALS`  
- `Python`, `Jupyter`, `Shell Scripting`

---

## Key Takeaways  
- Designed and deployed a large-scale recommender pipeline in a distributed environment  
- Applied collaborative filtering and customer segmentation techniques using real-world datasets  
- Tuned and validated models using robust ranking metrics  
- Built scalable, reproducible workflows for machine learning in big data ecosystems
