import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOAD DATA

ratings = pd.read_csv("data/rating.csv")
movies = pd.read_csv("data/movie.csv")
tags = pd.read_csv("data/tag.csv")
links = pd.read_csv("data/link.csv")
genome_scores = pd.read_csv("data/genome_scores.csv")
genome_tags = pd.read_csv("data/genome_tags.csv")

for name, df_ in [("ratings", ratings), ("movies", movies), ("tags", tags),
                  ("links", links), ("genome_scores", genome_scores),
                  ("genome_tags", genome_tags)]:
    
    print(f"--- {name} ---")
    print(df_.shape)
    print(df_.head())
    print(df_.info())
    print(df_.describe())
    print(df_.isnull().sum())
    print(df_.duplicated().sum())
    print(df_.memory_usage(deep=True))

# EDA (Exploratory Data Analysis)

ratings_per_user = ratings.groupby("userId")["rating"].size()
plt.figure(figsize=(12, 6))
plt.hist(ratings_per_user, bins=100)
plt.xlabel("Number of Ratings per User")
plt.ylabel("Number of Users")
plt.title("Distribution of Ratings per User")
plt.savefig("user_ratings_distribution.png")
plt.show()

ratings_per_movie = ratings.groupby("movieId")["rating"].size()
plt.figure(figsize=(12, 6))
plt.hist(ratings_per_movie, bins=100)
plt.xlabel("Number of Ratings per Movie")
plt.ylabel("Number of Movies")
plt.title("Distribution of Ratings per Movie")
plt.savefig("movie_ratings_distribution.png")
plt.show()

ratings["rating"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.title("Distribution of Ratings")
plt.savefig("rating_distribution.png")
plt.show()

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings.set_index("timestamp").resample("YE").size().plot()
plt.xlabel("Year")
plt.ylabel("Number of Ratings")
plt.title("Number of Ratings Over Time")
plt.savefig("ratings_over_time.png")
plt.show()
ratings = ratings.reset_index() if "timestamp" not in ratings.columns else ratings

avg_ratings_per_user = ratings.groupby("userId")["rating"].mean()
plt.figure(figsize=(12, 6))
plt.hist(avg_ratings_per_user, bins=100)
plt.xlabel("Average Rating per User")
plt.ylabel("Number of Users")
plt.title("Distribution of Average Ratings per User")
plt.savefig("avg_ratings_per_user_distribution.png")
plt.show()

avg_ratings_per_movie = ratings.groupby("movieId")["rating"].mean()
plt.figure(figsize=(12, 6))
plt.hist(avg_ratings_per_movie, bins=100)
plt.xlabel("Average Rating per Movie")
plt.ylabel("Number of Movies")
plt.title("Distribution of Average Ratings per Movie")
plt.savefig("avg_ratings_per_movie_distribution.png")
plt.show()

genre_counts = movies["genres"].str.split("|").explode().value_counts()
genre_counts.plot(kind="bar", figsize=(12, 6))
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.title("Number of Movies per Genre")
plt.savefig("movies_per_genre_distribution.png")
plt.show()

most_common_tag = tags.groupby("userId")["tag"].count()
plt.figure(figsize=(12, 6))
most_common_tag.sort_values(ascending=False).head(10).plot(kind="bar")
plt.xlabel("User ID")
plt.ylabel("Number of Tags")
plt.title("Top 10 Users with Most Tags")
plt.savefig("top_10_users_with_most_tags.png")
plt.show()

relevance_scores = genome_scores.groupby("tagId")["relevance"].mean()
plt.figure(figsize=(12, 6))
plt.hist(relevance_scores, bins=100)
plt.xlabel("Average Relevance Score")
plt.ylabel("Number of Tags")
plt.title("Distribution of Average Relevance Scores")
plt.savefig("relevance_scores_distribution.png")
plt.show()

yearly_avg_rating = ratings.groupby(ratings["timestamp"].dt.year)["rating"].mean()
plt.figure(figsize=(12, 6))
yearly_avg_rating.plot()
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.title("Average Rating Over Time")
plt.savefig("yearly_avg_rating.png")
plt.show()

monthly_avg_rating = ratings.groupby(ratings["timestamp"].dt.month)["rating"].mean()
plt.figure(figsize=(12, 6))
monthly_avg_rating.plot()
plt.xlabel("Month")
plt.ylabel("Average Rating")
plt.title("Average Rating by Month")
plt.savefig("monthly_avg_rating.png")
plt.show()

daily_activity = ratings.groupby(ratings["timestamp"].dt.date).size()
plt.figure(figsize=(12, 6))
daily_activity.plot()
plt.xlabel("Date")
plt.ylabel("Number of Ratings")
plt.title("Daily Rating Activity")
plt.savefig("daily_rating_activity.png")
plt.show()

user_lifetime = ratings.groupby("userId")["timestamp"].agg(first_rating="min", last_rating="max")

user_lifetime["days_active"] = (user_lifetime["last_rating"] - user_lifetime["first_rating"]).dt.days

# Handling missing values (Cleaning)

tags = tags.dropna(subset=["tag"])
links["tmdbId"] = links["tmdbId"].fillna(0)

# Feature engineering — movie features

movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
movies["release_year"] = movies["release_year"].fillna(movies["release_year"].median()).astype(int)

ratings["rating_year"] = ratings["timestamp"].dt.year
latest_rating_year = ratings["rating_year"].max()  
movies["movie_age"] = latest_rating_year - movies["release_year"]

genre_dummies = movies["genres"].str.get_dummies(sep="|")
movies = pd.concat([movies.drop(columns="genres"), genre_dummies], axis=1)
movies["num_genres"] = genre_dummies.sum(axis=1)

# Feature engineering — user and movie features with biases

global_mean = ratings["rating"].mean()

user_features = ratings.groupby("userId")["rating"].agg(
    user_num_ratings="count", user_avg_rating="mean", 
    user_std_rating="std", user_rating_variance="var",
    ).reset_index()

user_features["user_bias"] = user_features["user_avg_rating"] - global_mean

movie_features = ratings.groupby("movieId")["rating"].agg(
    movie_num_ratings="count", movie_avg_rating="mean",
    movie_std_rating="std", movie_rating_variance="var",
    ).reset_index()

movie_features["movie_bias"] = movie_features["movie_avg_rating"] - global_mean
movie_features["movie_popularity"] = (movie_features["movie_num_ratings"] / movie_features["movie_num_ratings"].max())

# Feature engineering — movie genome matrix

movie_genome_matrix = genome_scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)
movie_genome_matrix.columns = [f"tag_{col}" for col in movie_genome_matrix.columns]
movie_genome_matrix = movie_genome_matrix.reset_index()

df = ratings.merge(movies, on="movieId", how="left")
df = df.merge(user_features, on="userId", how="left")
df = df.merge(movie_features, on="movieId", how="left")
df = df.merge(movie_genome_matrix, on="movieId", how="left")

print("Final df shape:", df.shape)

# The Final df features include a lot of features and merging them all together
# will result in OOM (Out of Memory) issues. So, we will apply PCA to reduce dimensionality of the movie genome matrix features.
# See ya until next time!
