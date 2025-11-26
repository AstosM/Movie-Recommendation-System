# app.py
# Streamlit Movie Recommendation System with Posters & Homepage Links

import difflib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------ CONFIG & STYLE ------------------------ #
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for nicer UI
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top, #1f2933 0, #020617 60%);
        color: #e5e7eb;
    }
    .rec-card {
        padding: 1rem;
        border-radius: 1rem;
        border: 1px solid #44444433;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.85));
        box-shadow: 0 18px 30px rgba(0,0,0,0.45);
        height: 100%;
    }
    .rec-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
        color: #e5e7eb;
    }
    .rec-meta {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    .rec-tagline {
        font-size: 0.9rem;
        color: #d1d5db;
        margin-top: 0.5rem;
        font-style: italic;
        min-height: 2rem;
    }
    .movie-chip {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background-color: #374151;
        color: #d1d5db;
        font-size: 0.75rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .similarity-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.12);
        color: #bfdbfe;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    .homepage-link a {
        text-decoration: none;
        font-size: 0.8rem;
        color: #93c5fd !important;
    }
    .homepage-link a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------ DATA & MODEL ------------------------ #
@st.cache_data(show_spinner="Loading movie data…")
def load_data(csv_path: str = "movies recommendation dataset.csv") -> pd.DataFrame:
    """
    Load the movies dataset.

    Expects at least these columns:
    - title
    - genres
    - keywords
    - tagline
    - cast
    - director
    Optionally:
    - homepage  (movie website link)
    - poster_url (direct image URL)
    - poster_path (TMDB poster path)
    """
    df = pd.read_csv(csv_path)

    selected_features = ["genres", "keywords", "tagline", "cast", "director"]
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = ""
        df[feature] = df[feature].fillna("")

    df["combined_features"] = (
        df["genres"] + " " +
        df["keywords"] + " " +
        df["tagline"] + " " +
        df["cast"] + " " +
        df["director"]
    )

    return df


@st.cache_resource(show_spinner="Building recommendation model…")
def build_model(df: pd.DataFrame):
    """
    Fit TF-IDF vectorizer and compute cosine similarity matrix.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_vectors = vectorizer.fit_transform(df["combined_features"])
    similarity = cosine_similarity(feature_vectors)
    return vectorizer, similarity


def get_poster_url(row: pd.Series) -> str:
    """
    Get a poster URL for a movie using your dataset columns.

    Priority:
    1. 'homepage' if it looks like a direct image link
    2. 'poster_url' if it is a full URL
    3. 'poster_path' combined with TMDB base URL
    4. Fallback placeholder image
    """
    def is_image_url(url: str) -> bool:
        url_lower = url.lower()
        return url_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))

    # 1. Homepage as direct image (if so)
    homepage = row.get("homepage", "")
    if isinstance(homepage, str):
        homepage = homepage.strip()
        if homepage.startswith(("http://", "https://")) and is_image_url(homepage):
            return homepage

    # 2. Explicit poster_url
    poster_url = row.get("poster_url", "")
    if isinstance(poster_url, str):
        poster_url = poster_url.strip()
        if poster_url.startswith(("http://", "https://")):
            return poster_url

    # 3. TMDB-style poster_path
    poster_path = row.get("poster_path", "")
    if isinstance(poster_path, str) and poster_path.strip():
        path = poster_path.strip()
        if not path.startswith("/"):
            path = "/" + path
        # TMDB base (can be adjusted)
        return f"https://image.tmdb.org/t/p/w500{path}"

    # 4. Fallback
    return "https://static.streamlit.io/examples/cat.jpg"


def recommend_movies(
    df: pd.DataFrame,
    similarity: np.ndarray,
    movie_name: str,
    num_recs: int = 10
):
    """
    Use difflib to match the movie name, then return top N recommendations.
    """
    if not movie_name:
        return None, []

    all_titles = df["title"].astype(str).tolist()
    close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.4)

    if not close_matches:
        return None, []

    close_match = close_matches[0]

    # Use 'index' column if provided, otherwise DataFrame index
    if "index" in df.columns:
        idx = df[df.title == close_match]["index"].values[0]
    else:
        idx = df[df.title == close_match].index[0]

    sim_scores = list(enumerate(similarity[idx]))
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, (movie_idx, score) in enumerate(sorted_sim_scores):
        if i == 0:
            continue  # skip the same movie
        if len(recommendations) >= num_recs:
            break

        row = df.iloc[movie_idx]
        recommendations.append({
            "title": row.get("title", "Unknown"),
            "genres": row.get("genres", ""),
            "tagline": row.get("tagline", ""),
            "score": float(score),
            "poster_url": get_poster_url(row),
            "row": row,
        })

    return close_match, recommendations


# ------------------------ UI LAYOUT ------------------------ #
st.markdown("## 🎬 Movie Recommendation System")
st.markdown(
    """
    Discover movies similar to your favourites using content-based filtering  
    based on **genres, keywords, tagline, cast & director**.
    """
)

with st.sidebar:
    st.header("⚙️ Controls")

    csv_path = st.text_input(
        "CSV file path",
        value="movies recommendation dataset.csv",
        help="Path to your movie dataset CSV"
    )

    num_recs = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=12,
        step=1
    )

    show_raw = st.checkbox("Show raw data preview", value=False)

    st.markdown("---")
    st.caption(
        "Your dataset can include a **homepage** column with movie links, "
        "and optionally **poster_url/poster_path** for better posters."
    )

# Load data & model
try:
    movies_df = load_data(csv_path)
    vectorizer, similarity_matrix = build_model(movies_df)
except Exception as e:
    st.error(f"Error loading data or building model:\n\n{e}")
    st.stop()

if show_raw:
    st.subheader("📄 Data Preview")
    st.dataframe(movies_df.head(50), use_container_width=True)

# Search area
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    user_movie_name = st.text_input(
        "Enter your favourite movie",
        placeholder="e.g. Avatar, The Dark Knight, Inception...",
    )

with col_right:
    all_titles_sorted = sorted(movies_df["title"].astype(str).unique())
    selected_title = st.selectbox(
        "…or pick from the list",
        options=["(None)"] + all_titles_sorted
    )
    if selected_title != "(None)":
        user_movie_name = selected_title

search_btn = st.button(
    "🔍 Get Recommendations",
    type="primary",
    use_container_width=True
)

st.markdown("---")

# ------------------------ RECOMMENDATION RESULTS ------------------------ #
if search_btn:
    original_title, recs = recommend_movies(
        movies_df,
        similarity_matrix,
        user_movie_name,
        num_recs=num_recs
    )

    if original_title is None:
        st.warning("I couldn't find a close match for that movie. Please try another title.")
    else:
        st.subheader(f"Recommendations based on: **{original_title}**")

        # Original movie details
        original_row = movies_df[movies_df.title == original_title].iloc[0]
        with st.expander("Show details of your chosen movie", expanded=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(get_poster_url(original_row), use_container_width=True)
                homepage = original_row.get("homepage", "")
                if isinstance(homepage, str) and homepage.strip().startswith(("http://", "https://")):
                    st.markdown(
                        f'<p class="homepage-link">'
                        f'<a href="{homepage}" target="_blank">🔗 Open movie homepage</a>'
                        f'</p>',
                        unsafe_allow_html=True
                    )
            with c2:
                st.markdown(f"### {original_title}")
                st.markdown(f"**Genres:** {original_row.get('genres', 'N/A')}")
                tagline = original_row.get("tagline", "")
                if isinstance(tagline, str) and tagline.strip():
                    st.markdown(f"**Tagline:** _{tagline}_")
                cast = original_row.get("cast", "")
                if isinstance(cast, str) and cast.strip():
                    st.markdown(
                        f"**Cast:** {cast[:400]}{'…' if len(cast) > 400 else ''}"
                    )
                director = original_row.get("director", "")
                if isinstance(director, str) and director.strip():
                    st.markdown(f"**Director:** {director}")

        st.markdown("### 🎯 Top Recommendations")

        if not recs:
            st.info("No similar movies found. Try a different title or check your dataset.")
        else:
            cols_per_row = 3
            for i in range(0, len(recs), cols_per_row):
                row_recs = recs[i:i + cols_per_row]
                cols = st.columns(len(row_recs))
                for c, rec in zip(cols, row_recs):
                    with c:
                        st.markdown('<div class="rec-card">', unsafe_allow_html=True)

                        # Poster
                        st.image(rec["poster_url"], use_container_width=True)

                        # Title
                        st.markdown(
                            f'<div class="rec-title">{rec["title"]}</div>',
                            unsafe_allow_html=True
                        )

                        # Genres as chips
                        genres = str(rec["genres"]).split("|") if isinstance(rec["genres"], str) else []
                        if genres and genres[0].strip():
                            chips_html = "".join(
                                f'<span class="movie-chip">{g.strip()}</span>'
                                for g in genres if g.strip()
                            )
                            st.markdown(
                                f'<div class="rec-meta">{chips_html}</div>',
                                unsafe_allow_html=True
                            )

                        # Similarity badge
