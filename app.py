# app.py
# STREAMLIT MOVIE RECOMMENDER — NO FALLBACK IMAGES

import difflib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# PAGE CONFIG + CSS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <style>
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
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner="Loading movie dataset…")
def load_data(csv_path="movies recommendation dataset.csv"):
    df = pd.read_csv(csv_path)

    needed = ["genres", "keywords", "tagline", "cast", "director"]
    for f in needed:
        if f not in df.columns:
            df[f] = ""
        df[f] = df[f].fillna("")

    df["combined_features"] = (
        df["genres"] + " " +
        df["keywords"] + " " +
        df["tagline"] + " " +
        df["cast"] + " " +
        df["director"]
    )

    return df


# ------------------------------------------------------------
# BUILD MODEL
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Building recommendation model…")
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(df["combined_features"])
    similarity = cosine_similarity(vectors)
    return vectorizer, similarity


# ------------------------------------------------------------
# POSTER HANDLING — NO FALLBACK IMAGE
# ------------------------------------------------------------
def get_poster_url(row):
    """
    Returns ONLY a dataset poster URL.
    No fallback images.
    Returns None if no valid poster exists.
    """

    def is_image(url: str) -> bool:
        url = url.lower()
        return url.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))

    # 1) homepage if it's a direct image
    homepage = row.get("homepage", "")
    if isinstance(homepage, str):
        homepage = homepage.strip()
        if homepage.startswith(("http://", "https://")) and is_image(homepage):
            return homepage

    # 2) poster_url
    poster_url = row.get("poster_url", "")
    if isinstance(poster_url, str):
        poster_url = poster_url.strip()
        if poster_url.startswith(("http://", "https://")) and is_image(poster_url):
            return poster_url

    # 3) TMDB poster_path
    poster_path = row.get("poster_path", "")
    if isinstance(poster_path, str) and poster_path.strip():
        poster_path = poster_path.strip()
        if is_image(poster_path):
            if not poster_path.startswith("/"):
                poster_path = "/" + poster_path
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    # 4) No image in dataset
    return None


# ------------------------------------------------------------
# RECOMMENDATION LOGIC
# ------------------------------------------------------------
def recommend(df, similarity, movie_name, num_recs=10):
    if not movie_name:
        return None, []

    titles = df["title"].astype(str).tolist()
    match = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.4)

    if not match:
        return None, []

    best = match[0]

    if "index" in df.columns:
        idx = df[df.title == best]["index"].values[0]
    else:
        idx = df[df.title == best].index[0]

    sims = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )

    recs = []
    for i, (movie_idx, score) in enumerate(sims):
        if i == 0:
            continue
        if len(recs) >= num_recs:
            break

        row = df.iloc[movie_idx]
        recs.append({
            "title": row.get("title", "Unknown"),
            "genres": row.get("genres", ""),
            "tagline": row.get("tagline", ""),
            "score": float(score),
            "poster": get_poster_url(row),
            "row": row
        })

    return best, recs


# ------------------------------------------------------------
# UI — HEADER
# ------------------------------------------------------------
st.markdown("## 🎬 Movie Recommendation System")
st.markdown("Find movies similar to your favorites — with posters from your dataset only.")

# SIDEBAR
with st.sidebar:
    st.header("Settings")
    csv_path = st.text_input("CSV file:", "movies recommendation dataset.csv")
    num_recs = st.slider("Recommendations", 5, 30, 12)
    show_raw = st.checkbox("Show dataset preview")

# LOAD DATA + MODEL
try:
    df = load_data(csv_path)
    vectorizer, similarity = build_model(df)
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

if show_raw:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

# SEARCH SECTION
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    movie_name = st.text_input("Enter a movie:")

with col2:
    all_titles = sorted(df["title"].astype(str).unique())
    pick = st.selectbox("…or choose:", ["(None)"] + all_titles)
    if pick != "(None)":
        movie_name = pick

btn = st.button("🔍 Recommend", type="primary", use_container_width=True)
st.markdown("---")


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------
if btn:
    original, recs = recommend(df, similarity, movie_name, num_recs)

    if not original:
        st.warning("No match found. Try another title.")
    else:
        st.subheader(f"Recommendations based on **{original}**")

        # ORIGINAL MOVIE SECTION
        row = df[df.title == original].iloc[0]
        poster = get_poster_url(row)

        with st.expander("Selected movie details", expanded=True):
            c1, c2 = st.columns([1, 2])

            with c1:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown("*(No poster available)*")

                homepage = row.get("homepage", "")
                if isinstance(homepage, str) and homepage.startswith(("http://", "https://")):
                    st.markdown(f"[🔗 Homepage]({homepage})")

            with c2:
                st.markdown(f"### {original}")
                st.markdown(f"**Genres:** {row.get('genres', 'N/A')}")
                tg = row.get("tagline", "")
                if isinstance(tg, str) and tg.strip():
                    st.markdown(f"**Tagline:** _{tg}_")

                cast = row.get("cast", "")
                if isinstance(cast, str) and cast.strip():
                    st.markdown(f"**Cast:** {cast[:400]}{'…' if len(cast) > 400 else ''}")

                director = row.get("director", "")
                if isinstance(director, str) and director.strip():
                    st.markdown(f"**Director:** {director}")

        # RECOMMENDATIONS GRID
        st.markdown("### 🎯 Top Recommendations")

        if not recs:
            st.info("No similar movies found.")
        else:
            cols_per_row = 3
            for i in range(0, len(recs), cols_per_row):
                row_recs = recs[i:i + cols_per_row]
                cols = st.columns(len(row_recs))

                for c, r in zip(cols, row_recs):
                    with c:
                        st.markdown('<div class="rec-card">', unsafe_allow_html=True)

                        # Poster ONLY if exists
                        if r["poster"]:
                            st.image(r["poster"], use_container_width=True)
                        else:
                            st.markdown("*(No poster available)*")

                        st.markdown(f'<div class="rec-title">{r["title"]}</div>', unsafe_allow_html=True)

                        # Genres
                        genres = str(r["genres"]).split("|")
                        chips = "".join(
                            f'<span class="movie-chip">{g.strip()}</span>'
                            for g in genres if g.strip()
                        )
                        st.markdown(f'<div class="rec-meta">{chips}</div>', unsafe_allow_html=True)

                        # Similarity
                        st.markdown(
                            f'<div class="similarity-badge">Score: {r["score"]:.3f}</div>',
                            unsafe_allow_html=True
                        )

                        # Tagline
                        tagline = r["tagline"]
                        if isinstance(tagline, str) and tagline.strip():
                            st.markdown(
                                f'<div class="rec-tagline">"{tagline[:140]}{"…" if len(tagline) > 140 else ""}"</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown('<div class="rec-tagline">No tagline.</div>', unsafe_allow_html=True)

                        # Homepage (if exists)
                        homepage = r["row"].get("homepage", "")
                        if isinstance(homepage, str) and homepage.startswith(("http://", "https://")):
                            st.markdown(f"[🔗 Homepage]({homepage})")

                        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Enter or select a movie and click Recommend.")
