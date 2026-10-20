# Simple Music Recommendation

A Python prototype for exploring music recommendations from Million Song Dataset-style listening data. It includes helpers to load and join listening history, track metadata, and genre labels, plus three recommendation approaches:

- **Popularity-based:** most-played tracks overall or within a genre.
- **Title search:** popular tracks whose titles contain a keyword.
- **Collaborative filtering:** recommendations based on similar listeners or similar tracks.

The reusable implementation is in [`modules.py`](./modules.py). [`runtime.ipynb`](./runtime.ipynb) demonstrates loading data, generating recommendations, and evaluating the collaborative-filtering methods. There is no command-line interface or web application.

## Requirements

- Python 3
- The Python packages in [`requirements.txt`](./requirements.txt)
- Dataset files in the formats described below (not included in this repository)

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt, activate the environment with:

```bat
.venv\Scripts\activate
```

To use the notebook locally, start Jupyter:

```bash
jupyter notebook runtime.ipynb
```

The notebook was authored for Google Colab and includes a `google.colab` Drive-mount cell. When running locally, remove or skip that cell and set the dataset paths to your local files. In Colab, mount Drive and update `home_path` in the notebook to the directory containing the data.

## Dataset

The loader expects three files:

| File | Expected content |
| --- | --- |
| `train_triplets.txt` | Tab-separated rows with `user_id`, `song_id`, and `play_count`, with no header |
| `p02_unique_tracks.txt` | `<SEP>`-separated rows containing `track_id`, `song_id`, `artist`, and `title`, with no header |
| `p02_msd_tagtraum_cd2.cls` | Tab-separated genre labels, with `track_id` followed by majority and minority genres; comment lines beginning with `#` are ignored |

These are Million Song Dataset-related files; data is not distributed with this project. Make sure you have the appropriate rights to access and use your chosen dataset. The notebook also refers to zipped triplet and track files and extracts the triplets archive in its configured data directory.

`MusicDataLoaderPl` reads the files, joins listening rows to track metadata by `song_id`, and left-joins genre labels by `track_id`. Its `sample_fraction` option samples the triplets **after loading the triplet file**, so it can reduce the size of the joined dataset but does not avoid reading the full source file.

## Usage

The examples below assume the three dataset files are available locally. Run them from the repository root so that `modules.py` can be imported.

### Load and prepare data

```python
from modules import (
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    MusicDataLoaderPl,
    NonePersonalizedRecommender,
)

data_dir = "/path/to/dataset"

loader = MusicDataLoaderPl(
    triplets_path=f"{data_dir}/train_triplets.txt",
    unique_tracks_path=f"{data_dir}/p02_unique_tracks.txt",
    genres_path=f"{data_dir}/p02_msd_tagtraum_cd2.cls",
    sample_fraction=0.5,  # Use 1.0 for the full triplet dataset.
)
df, tracks_df, triplets_df, genres_df = loader.load_data()
```

The loader returns four Polars DataFrames:

- `df`: listening events enriched with track metadata and genre labels
- `tracks_df`: unique-track metadata
- `triplets_df`: user/song listening counts
- `genres_df`: genre labels

The merged `df` contains `user_id`, `song_id`, `play_count`, `track_id`, `artist`, `title`, `majority-genre`, and `minorillty-genre` (the latter column name is spelled this way in the current implementation).

### Popular tracks

```python
popular = NonePersonalizedRecommender(df)

top_tracks = popular.get_top_250()
top_rock_tracks = popular.get_top_100_by_genre("Rock")
```

Both methods aggregate `play_count` by artist and title, sort by total plays, and return a Polars DataFrame with a one-based `index_number`. Genre names are matched without regard to case. Genre-specific recommendations require the genre data to have been loaded.

### Find tracks by title keyword

```python
search = ContentBasedRecommender(tracks_df, triplets_df)
love_tracks = search.get_collection("love")
```

This returns up to 50 matching tracks, ordered by total plays. The keyword is searched in the title without regard to case.

### Collaborative recommendations

```python
recommender = CollaborativeFilteringRecommender(df)
recommender.prepare_matrices(
    min_user_interactions=20,
    min_song_interactions=20,
    use_vectorizer=False,
)

# Select IDs from the prepared mappings so they are guaranteed to be in the model.
user_id = next(iter(recommender.user_to_idx))
song_id = next(iter(recommender.song_to_idx))

recommended_for_user_ids, recommended_for_user = (
    recommender.recommend_for_user(user_id, top_n=10)
)
recommended_for_song_ids, recommended_for_song = (
    recommender.recommend_for_song(song_id, top_n=10)
)
```

`prepare_matrices` filters out users and songs that do not exceed the minimum interaction thresholds, creates sparse user-song matrices, and reserves approximately 20% of each user's filtered interactions for testing. Recommendations return a pair: a list of song IDs and a Polars DataFrame containing artist/title rows.

- `recommend_for_user(user_id, top_n=10)` finds songs from other listeners with similar listening histories.
- `recommend_for_song(song_id, top_n=10)` finds tracks listened to by users with similar listening patterns for the given track.
- `use_vectorizer=True` applies scikit-learn's TF-IDF transformer to the training matrix before similarity calculations.

The `user_id` and `song_id` passed to these methods must be present in `user_to_idx` and `song_to_idx`, respectively. Filtering may remove IDs that exist in the original dataset.

### Evaluate recommendations

```python
user_precision_at_10 = recommender.evaluate_user_model(sample_size=100, k=10)
song_precision_at_10 = recommender.evaluate_song_model(sample_size=100, k=10)

print(f"User recommendations P@10: {user_precision_at_10:.3f}")
print(f"Song recommendations P@10: {song_precision_at_10:.3f}")
```

These methods return average Precision@K over sampled users, using held-out interactions from the prepared test matrix. Evaluation can take time on a large dataset; reduce `sample_size` while experimenting.

## Performance notes

The triplets dataset can be large. The loader materializes data in memory, and collaborative filtering constructs sparse user-song matrices and computes cosine similarities. Use a smaller `sample_fraction` and/or higher interaction thresholds to explore on a machine with limited memory. A sample can still require substantial memory because the source triplets file is read before sampling.

## Project files

- [`modules.py`](./modules.py) — dataset loader and recommender classes
- [`runtime.ipynb`](./runtime.ipynb) — notebook walkthrough and examples
- [`requirements.txt`](./requirements.txt) — Python dependencies
- [`LICENSE`](./LICENSE) — MIT License

