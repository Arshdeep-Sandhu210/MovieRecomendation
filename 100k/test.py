import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix

# Load dataset
ratings = pd.read_csv("ratings.csv")  # Assuming GroupLens dataset
movies = pd.read_csv("movies.csv")

# Ensure IDs are integers
ratings['userId'] = ratings['userId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

# Map userId and movieId to 0-based indexing (to avoid memory issues)
unique_users = {user: i for i, user in enumerate(ratings['userId'].unique())}
unique_movies = {movie: i for i, movie in enumerate(ratings['movieId'].unique())}

ratings['user_index'] = ratings['userId'].map(unique_users)
ratings['movie_index'] = ratings['movieId'].map(unique_movies)

# Create sparse user-item matrix (COO format)
user_item_sparse = coo_matrix(
    (ratings['rating'], (ratings['user_index'], ratings['movie_index']))
).tocsr()  # Convert to CSR format (efficient for row-based operations)

print("Sparse Matrix Shape:", user_item_sparse.shape)  # Should be (num_users, num_movies)

# ✅ Use sparse matrix for KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
knn.fit(user_item_sparse)  # Fit on sparse matrix

# Function to get recommendations
def get_user_recommendations(user_id, k=5):
    if user_id not in unique_users:
        return "User not found"

    user_idx = unique_users[user_id]  # Get 0-based user index

    distances, indices = knn.kneighbors(user_item_sparse[user_idx], n_neighbors=k+1)
    neighbors = indices.flatten()[1:]  # Exclude the user itself

    # Compute average rating across neighbors
    avg_ratings = user_item_sparse[neighbors].mean(axis=0).A1  # Convert sparse row to array

    # Get top 10 recommended movies
    recommended_movie_indices = np.argsort(avg_ratings)[-10:]  # Get top 10 indices
    recommended_movie_ids = [list(unique_movies.keys())[i] for i in recommended_movie_indices]

    # Convert movie IDs to names
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]['title'].values
    return recommended_movies

# Example usage
print(get_user_recommendations(user_id=100000, k=5))
