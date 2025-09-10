import streamlit as st
import pickle

# Load saved objects
with open(r"C:\Users\Admin\Desktop\Yashraj\ML\Netflix Recommendation System\netflix_recommender.pkl", "rb") as f:
    df, similarity = pickle.load(f)  # Unpack the loaded object

st.title("🎬 Netflix Recommendation System")

# Recommendation function
def recommend(title):
    if title not in df['Title'].values:   # change 'Title' if your column name differs
        return []
    idx = df[df['Title'] == title].index[0]
    distances = similarity[idx]
    # Get top 5 most similar
    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    return df.iloc[[i[0] for i in movie_list]]['Title'].values.tolist()

# Streamlit UI - dropdown list (selectbox)
movie_name = st.selectbox(
    "Choose a movie/show:",
    df['Title'].values
)

if st.button("Recommend"):
    recs = recommend(movie_name)
    if recs:
        st.subheader("Recommended for you:")
        for r in recs:
            st.write(f"- {r}")
    else:
        st.warning("Movie not found in database.")
