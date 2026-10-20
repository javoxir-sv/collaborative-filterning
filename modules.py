import pandas as pd
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer


# with polars
class MusicDataLoaderPl:
    def __init__(self, triplets_path, unique_tracks_path, genres_path, sample_fraction=1.0):
        self.triplets_path = triplets_path
        self.unique_tracks_path = unique_tracks_path
        self.genres_path = genres_path
        self.sample_fraction = (sample_fraction)

    def load_data(self):
        print("1. Loading unique tracks dataset...")
        pd_tracks_df = pd.read_csv(
                self.unique_tracks_path,
                sep="<SEP>",
                names=["track_id", "song_id", "artist", "title"],
                low_memory=True,
                engine="python")
        tracks_df = pl.from_pandas(pd_tracks_df, schema_overrides={
            "song_id":pl.datatypes.Categorical,
            "track_id":pl.datatypes.Categorical})
        del pd_tracks_df

        print("2. Loading triplets dataset...")
        triplets_df = pl.read_csv(
            self.triplets_path,
            separator="\t",
            new_columns = ["user_id", "song_id", "play_count"],
            schema = {
                "user_id":pl.datatypes.Categorical,
                "song_id":pl.datatypes.Categorical,
                "play_count":pl.datatypes.Int32},
            has_header = False,
            low_memory=True)
        
        if self.sample_fraction < 1.0:
            print(f"   Sampling {self.sample_fraction * 100}% of triplets dataset...")
            triplets_df = triplets_df.sample(fraction=self.sample_fraction, seed=42)
        
        print("3. Loading genres dataset...")
        genres_df = pl.read_csv(
            self.genres_path,
            separator="\t",
            comment_prefix="#",
            new_columns = ["track_id", "majority-genre", "minorillty-genre"],
            schema_overrides = {"track_id" : pl.datatypes.Categorical},
            has_header = False,
            low_memory=True)

        print("4. Merging datasets...")
        df = triplets_df.join(tracks_df, on="song_id", how="inner")
        df = df.join(genres_df, on="track_id", how="left")
        return df, tracks_df, triplets_df, genres_df


class NonePersonalizedRecommender:
  def __init__(self, df):
    self.df = df

  def get_top_250(self):
    top250 = (
        self.df.group_by(["artist", "title"])
        .agg(pl.col("play_count").sum())
        .sort("play_count", descending=True)
        .head(250)
        .with_row_index("index_number", offset=1))
    return top250

  def get_top_100_by_genre(self, genre):
    if "majority-genre" not in self.df.columns:
      raise ValueError("Genre data was not loaded.")
    
    top_100 = (
        self.df.filter(pl.col("majority-genre").str.to_lowercase() == genre.lower())
        .group_by(["artist", "title"])
        .agg(pl.col("play_count").sum())
        .sort("play_count", descending=True)
        .head(100)
        .with_row_index("index_number", offset=1))
    return top_100


class ContentBasedRecommender:
  def __init__(self, tracks_df, triplets_df):
    self.tracks_df = tracks_df
    self.triplets_df = triplets_df

  def get_collection(self, keyword):
    keyword = keyword.lower()
    matching_tracks = self.tracks_df.filter(pl.col("title").str.to_lowercase().str.contains(keyword))
    song_plays = self.triplets_df.group_by(["song_id"]).agg(pl.col("play_count").sum())
    merged = matching_tracks.join(song_plays, on="song_id", how="inner")
    merged = (
        merged.sort("play_count", descending=True).head(50)
        .select(["artist", "title", "play_count"])
        .with_row_index("index_number", offset=1)
        )
    return merged


class CollaborativeFilteringRecommender:
  def __init__(self, df):
    self.df = df
    self.train_matrix = None
    self.test_matrix = None
    self.user_to_idx = {}
    self.idx_to_user = {}
    self.song_to_idx = {}
    self.idx_to_song = {}

  def prepare_matrices(self, min_user_interactions=20, min_song_interactions=20, use_vectorizer=False):
    user_counts = self.df.group_by(pl.col("user_id"), maintain_order=True).len()
    song_counts = self.df.group_by(pl.col("song_id"), maintain_order=True).len()

    valid_users = user_counts.filter(pl.col("len") > min_user_interactions).select(pl.col("user_id"))
    valid_songs = song_counts.filter(pl.col("len") > min_song_interactions).select(pl.col("song_id"))

    filtered_df = self.df.filter(
        pl.col("user_id").is_in(valid_users['user_id'].implode()) 
        & pl.col("song_id").is_in(valid_songs['song_id'].implode()))
    
    filtered_df = filtered_df.with_columns(pl.lit(1).alias("listened"))

    np.random.seed(42)
    filtered_df = filtered_df.with_columns(
        pl.lit(np.random.rand(len(filtered_df))).alias("rand"))

    filtered_df = filtered_df.with_columns(
        pl.col("rand").rank(method="average", descending=False)
        .over("user_id")
        .truediv(pl.col("rand").count().over("user_id"))
        .alias("rank"))

    train_df = filtered_df.filter(pl.col("rank") > 0.2)
    test_df = filtered_df.filter(pl.col("rank") <= 0.2)

    unique_users = filtered_df['user_id'].unique()
    unique_songs = filtered_df['song_id'].unique()
    n_users = len(unique_users)
    n_songs = len(unique_songs)

    self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
    self.song_to_idx = {s: i for i, s in enumerate(unique_songs)}
    self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
    self.idx_to_song = {v: k for k, v in self.song_to_idx.items()}

    def create_sparse(sub_df):
      u_idx = sub_df['user_id'].replace_strict(self.user_to_idx, return_dtype=pl.Int32).to_numpy()
      s_idx = sub_df['song_id'].replace_strict(self.song_to_idx, return_dtype=pl.Int32).to_numpy()
      
      # filter out any unmapped values if they occur
      valid_mask = (u_idx != None) & (s_idx != None)
      u_idx = u_idx[valid_mask].astype(np.int32)
      s_idx = s_idx[valid_mask].astype(np.int32)
      
      data = np.ones(len(sub_df), dtype=np.float32)
      return csr_matrix((data, (u_idx, s_idx)), shape=(n_users, n_songs))
    
    self.test_matrix = create_sparse(test_df)
    if use_vectorizer:
      raw_train_matrix = create_sparse(train_df)
      transformer = TfidfTransformer(norm='l2', use_idf=True)
      self.train_matrix = transformer.fit_transform(raw_train_matrix)
    else:
      self.train_matrix = create_sparse(train_df)

  # People similar to you listening
  def recommend_for_user(self, user_id, top_n=10):
    u_idx = self.user_to_idx[user_id]

    # getting the user's row / vector
    user_vector = self.train_matrix[u_idx]
    
    # calculating the similarity between all the users and the given user
    similarities = cosine_similarity(self.train_matrix, user_vector).flatten()
    
    # given users similarity setting to negotive to not to effect the top similar users
    similarities[u_idx] = -1
    
    # top n similar users, [::-1] reverses the list
    top_n_sims = similarities.argsort()[::-1][:top_n*5]
    
    # multiplying neighbors vectors by the simlarity wight, it will reshape the probability of the user choosing a song
    neighbor_scores = self.train_matrix[top_n_sims].T @ similarities[top_n_sims]
    
    # this creates a boolean list, setting true for already listened songs
    already_listened = user_vector.toarray().flatten() > 0
    
    # this sets the already listened songs to -1 to avoid choosing them again
    neighbor_scores[already_listened] = -1
    
    # gets the top n similar songs
    top_n_songs = neighbor_scores.argsort()[::-1][:top_n]
    songs_ids = [self.idx_to_song[idx] for idx in top_n_songs]

    song_df = (
        self.df.filter(pl.col("song_id").is_in(songs_ids))
        .select(["song_id", "artist", "title"])
        .unique(subset=["song_id"])
        .head(top_n)
        .select(["artist", "title"])
        .with_row_index("index_number", offset=1)
        )
    return songs_ids, song_df

  # People who listen to this track usually listen
  def recommend_for_song(self, song_id, top_n=10):
    # just like the user, but with training matrix transposed
    s_idx = self.song_to_idx[song_id]
    # rows are songs and columns are users
    item_matrix = self.train_matrix.T
    song_vector = item_matrix[s_idx]

    similarities = cosine_similarity(item_matrix, song_vector).flatten()
    similarities[s_idx] = -1
    
    top_n_songs = similarities.argsort()[::-1][:top_n*5]
    song_ids = [self.idx_to_song[idx] for idx in top_n_songs]
    
    song_df = (
        self.df.filter(pl.col("song_id").is_in(song_ids))
        .select(["song_id", "artist", "title"])
        .unique(subset=["song_id"])
        .head(top_n)
        .select(["artist", "title"])
        .with_row_index("index_number", offset=1)
        )
    return song_ids, song_df


  # evaluating the user similarity based recommendation system
  def evaluate_user_model(self, sample_size=100, k=10):
    precisions = []
    # finding user who actually has items in the test dataset
    test_user_indices = np.where(self.test_matrix.getnnz(axis=1) > 0)[0]
    eval_indices = np.random.choice(test_user_indices, size=min(sample_size, len(test_user_indices)), replace=False)
    
    for u_idx in eval_indices:
      user_id = self.idx_to_user[u_idx]

      # get top k recommendations for the user
      recommended_song_ids, _ = self.recommend_for_user(user_id, top_n=k)
      if not recommended_song_ids:
          continue

      # get the actual songs that user listened in the test dataset
      actual_test_song_indices = self.test_matrix[u_idx].indices
      actual_test_song_ids = {
          self.idx_to_song[s_idx] for s_idx in actual_test_song_indices
      }

      if not actual_test_song_ids:
          continue

      # calculate intersection between recommended and the test user
      hits = len(set(recommended_song_ids).intersection(actual_test_song_ids))

      # calculate p@k for the user
      precision = hits / k
      precisions.append(precision)

    # return the average p@k across all the evaluated recommendations
    avg_precision = np.mean(precisions) if precisions else 0.0
    return avg_precision

  # this is for evaluating the song based recommendation model
  def evaluate_song_model(self, sample_size=100, k=10):
    precisions = []
    test_user_indices = np.where(self.test_matrix.getnnz(axis=1) > 0)[0]
    eval_users = np.random.choice(test_user_indices, size=min(sample_size, len(test_user_indices)), replace=False)
    
    for u_idx in eval_users:
      # get the songs the current user listens to
      train_song_indices = self.train_matrix[u_idx].indices
      if len(train_song_indices) == 0:
          continue

      # pick random training song    
      seed_s_idx = np.random.choice(train_song_indices)
      seed_song_id = self.idx_to_song[seed_s_idx]

      # generate k recommendations for this track
      song_ids, _ = self.recommend_for_song(seed_song_id, top_n=k)
      if not song_ids:
          continue

      # get the actuals songs this user listened to from the test
      actual_test_song_indices = self.test_matrix[u_idx].indices
      actual_test_song_ids = {
          self.idx_to_song[s_idx] for s_idx in actual_test_song_indices
      }

      if not actual_test_song_ids:
          continue

      # calculate inersection between recommended and actually listened
      hits = len(set(song_ids).intersection(actual_test_song_ids))

      # calculate p@k
      precision = hits / k
      precisions.append(precision)

    avg_precision = np.mean(precisions) if precisions else 0.0
    return avg_precision

