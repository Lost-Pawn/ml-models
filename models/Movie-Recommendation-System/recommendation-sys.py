import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# LOAD DATA
# grouplens/movielens-20m dataset kaggle: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
ratings = pd.read_csv("data/rating.csv")

# my savoir two liner code lmao
if len(ratings) > 3000000:
    ratings = ratings.sample(n=3000000, random_state=42).reset_index(drop=True)

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
links["tmdbId"] = links["tmdbId"].fillna(0) # useless btw

# Feature engineering — movie features

movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
movies["release_year"] = movies["release_year"].fillna(movies["release_year"].median()).astype(int)

genre_dummies = movies["genres"].str.get_dummies(sep="|")
movies = pd.concat([movies.drop(columns="genres"), genre_dummies], axis=1)
movies["num_genres"] = genre_dummies.sum(axis=1)

# Movie genome matrix PCA

genome_matrix_raw = genome_scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)

pca = PCA(n_components=50, random_state=42)
genome_reduced = pca.fit_transform(genome_matrix_raw.values).astype("float32")

movie_genome_matrix = pd.DataFrame(
    genome_reduced,
    columns=[f"genome_pc_{i}" for i in range(genome_reduced.shape[1])],
    index=genome_matrix_raw.index,
).reset_index()

# Split -- before computing rating derived agg to avoid leakage
train_ratings, test_ratings = train_test_split(ratings, test_size=0.3, random_state=42)

# Feature engineering — user and movie features with biases

global_mean = train_ratings["rating"].mean()

user_features = train_ratings.groupby("userId")["rating"].agg(
    user_num_ratings="count", user_avg_rating="mean", user_std_rating="std").reset_index()

user_features["user_std_rating"] = user_features["user_std_rating"].fillna(0)
user_features["user_bias"] = user_features["user_avg_rating"] - global_mean
user_features["user_num_ratings_log"] = np.log1p(user_features["user_num_ratings"].fillna(0))

movie_features = train_ratings.groupby("movieId")["rating"].agg(
    movie_num_ratings="count", movie_avg_rating="mean", movie_std_rating="std").reset_index()

movie_features["movie_std_rating"] = movie_features["movie_std_rating"].fillna(0)
movie_features["movie_bias"] = movie_features["movie_avg_rating"] - global_mean
movie_features["movie_popularity_log"] = np.log1p(movie_features["movie_num_ratings"])

#  Feature lookup dict --> O(1) access during training and inference

movie_id_to_genre = dict(zip(movies["movieId"].astype(str), genre_dummies.values.astype("float32")))
genre_dim = genre_dummies.shape[1]
default_genre = np.zeros(genre_dim, dtype="float32")

genome_ids = movie_genome_matrix["movieId"].astype(str).values
genome_vals = movie_genome_matrix.drop(columns="movieId").values.astype("float32")
movie_id_to_genome = dict(zip(genome_ids, genome_vals))
default_genome = np.zeros(genome_vals.shape[1], dtype="float32")

movie_id_to_bias = dict(zip(movie_features["movieId"].astype(str), movie_features["movie_bias"].astype("float32")))
movie_id_to_pop = dict(zip(movie_features["movieId"].astype(str), movie_features["movie_popularity_log"].astype("float32")))
user_id_to_bias = dict(zip(user_features["userId"].astype(str), user_features["user_bias"].astype("float32")))
user_id_to_activity = dict(zip(user_features["userId"].astype(str), user_features["user_num_ratings_log"].astype("float32")))

def get_genre(mid):     return movie_id_to_genre.get(mid, default_genre)
def get_genome(mid):    return movie_id_to_genome.get(mid, default_genome)
def get_movie_bias(mid): return movie_id_to_bias.get(mid, 0.0)
def get_movie_pop(mid):  return movie_id_to_pop.get(mid, 0.0)
def get_user_bias(uid):  return user_id_to_bias.get(uid, 0.0)
def get_user_activity(uid): return user_id_to_activity.get(uid, 0.0)

# Build TensorFlow dataset

def build_tf_dataset(ratings_split):
    r = ratings_split.copy()
    r["userId"] = r["userId"].astype(str)
    r["movieId"] = r["movieId"].astype(str)

    return tf.data.Dataset.from_tensor_slices({
        "user_id": r["userId"].astype("int32").values,
        "movie_id": r["movieId"].astype("int32").values,
        "rating": r["rating"].values.astype("float32"),
        "genre_vector": np.stack(r["movieId"].map(get_genre).values),
        "genome_vector": np.stack(r["movieId"].map(get_genome).values),
        "movie_bias": r["movieId"].map(get_movie_bias).values.astype("float32"),
        "movie_popularity": r["movieId"].map(get_movie_pop).values.astype("float32"),
        "user_bias": r["userId"].map(get_user_bias).values.astype("float32"),
        "user_activity": r["userId"].map(get_user_activity).values.astype("float32"),
    })

train_ds = build_tf_dataset(train_ratings)
test_ds = build_tf_dataset(test_ratings)

print("Train dataset element spec:", train_ds.element_spec)

# Embedding dimensions 
num_unique_users = int(ratings["userId"].max()) + 1
num_unique_movies = int(ratings["movieId"].max()) + 1
embedding_dim = 50

# Hybrid Recommender Model: Collaborative + Content-based

class HybridRecommender(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_dim, genre_dim, genome_dim):
        super().__init__()

        embedding_reg = tf.keras.regularizers.l2(0.0001)
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_dim, embeddings_initializer="he_normal",
            embeddings_regularizer=embedding_reg, name="user_embedding")
        self.movie_embedding = tf.keras.layers.Embedding(
            num_movies, embedding_dim, embeddings_initializer="he_normal",
            embeddings_regularizer=embedding_reg, name="movie_embedding")
 
        dense_reg = tf.keras.regularizers.l2(0.0001)
        self.dense1 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=dense_reg)
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=dense_reg)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.residual_out = tf.keras.layers.Dense(1, activation=None)
 
    def call(self, inputs, training=False):
        user_emb = self.user_embedding(inputs["user_id"])          
        movie_emb = self.movie_embedding(inputs["movie_id"])       
 
        scalar_feats = tf.stack([
            inputs["movie_popularity"],
            inputs["user_activity"],
        ], axis=1)                                   
 
        x = tf.concat([
            user_emb, movie_emb,
            inputs["genre_vector"],
            inputs["genome_vector"],
            scalar_feats,
        ], axis=1)
 
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        residual = tf.squeeze(self.residual_out(x), axis=1)  #(batch_size, 1) to (batch_size,)
 
        pred = global_mean + inputs["user_bias"] + inputs["movie_bias"] + residual
        return pred
 
 
model = HybridRecommender(
    num_users=num_unique_users,
    num_movies=num_unique_movies,
    embedding_dim=embedding_dim,
    genre_dim=genre_dim,
    genome_dim=genome_vals.shape[1],
)
 
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
loss_fn = tf.keras.losses.MeanSquaredError()
 
BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 5
 
train_ds_batched = train_ds.cache().shuffle(buffer_size=100_000, seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds_batched = test_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
 
# Training loop with early stopping

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        preds = model(batch, training=True)
        loss = loss_fn(batch["rating"], preds) + tf.add_n(model.losses)  
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
 
 
@tf.function
def val_step(batch):
    preds = model(batch, training=False)
    return loss_fn(batch["rating"], preds)
 
best_val_loss = float("inf")
patience_counter = 0
best_weights = None
 
for epoch in range(EPOCHS):
    train_losses = []
    for batch in train_ds_batched:
        train_losses.append(train_step(batch).numpy())
 
    val_losses = []
    for batch in test_ds_batched:
        val_losses.append(val_step(batch).numpy())
 
    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))
    print(f"Epoch {epoch + 1:3d} | train_mse={train_loss:.4f} | val_mse={val_loss:.4f}")
 
    if val_loss < best_val_loss - 0.001: # small threshold to avoid tiny fluctuations
        best_val_loss = val_loss
        patience_counter = 0
        best_weights = [w.numpy() for w in model.trainable_variables]
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1} (best val mse={best_val_loss:.4f})")
            for w, bw in zip(model.trainable_variables, best_weights):
                w.assign(bw)
            break
 
# User-based CF using learned embeddings 

num_users_dim = int(num_unique_users)
num_movies_dim = int(num_unique_movies)

movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))
full_user_embeddings = model.user_embedding(tf.range(num_users_dim)).numpy()

# Sparse matrices: what each user has LIKED (≥4.0) and SEEN (any rating)
positive_ratings = ratings[ratings["rating"] >= 4.0]

ratings_positive = sp.csr_matrix(
    (positive_ratings["rating"].values.astype("float32"),
     (positive_ratings["userId"].values, positive_ratings["movieId"].values)),
    shape=(num_users_dim, num_movies_dim),
)

ratings_seen = sp.csr_matrix(
    (np.ones(len(ratings), dtype="float32"),
     (ratings["userId"].values, ratings["movieId"].values)),
    shape=(num_users_dim, num_movies_dim),
)

def recommend_for_user(user_id, k_neighbors=20, top_n=10):

    # Step 1: find the k most similar users via embedding cosine similarity
    user_emb = full_user_embeddings[[user_id]]                      # (1, emb_dim)
    sims = cosine_similarity(user_emb, full_user_embeddings)[0] # (num_users,)
    sims[user_id] = -1.0                                            # exclude self

    neighbor_ids = np.argsort(sims)[-k_neighbors:]                  # top-k indices

    # Step 2: score every movie by how much those neighbors liked it
    neighbor_sims   = sims[neighbor_ids]                            # (k,)
    neighbor_posmat = ratings_positive[neighbor_ids]                           # (k, num_movies)
    scores          = neighbor_sims @ neighbor_posmat                # (num_movies,)

    # Step 3: zero out movies this user has already seen
    seen_movie_ids     = ratings_seen[user_id].indices
    scores[seen_movie_ids] = -np.inf

    # Step 4: pick top-N and return as (title, score) pairs
    top_ids = np.argsort(scores)[-top_n:][::-1] # pick and reverse
    return [
        (movie_id_to_title.get(mid, f"movieId={mid}"), float(scores[mid]))
        for mid in top_ids if np.isfinite(scores[mid])
    ]

target_users = np.random.default_rng(42).choice(
    ratings["userId"].unique(), size=200, replace=False
)

all_recommendations = {uid: recommend_for_user(uid) for uid in target_users}

for uid in target_users[:5]:
    print(f"\nTop recommendations for user {uid}:")
    for title, score in all_recommendations[uid]:
        print(f"  {title}  (score={score:.2f})")