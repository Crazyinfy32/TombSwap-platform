# Hybrid Recommendation Algorithm (Example: LightFM library)
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Load dataset (replace 'data' with your dataset)
data = fetch_movielens(min_rating=4.0)

# Build interaction matrix
interactions = data['train']

# Build and train the hybrid model
model = LightFM(loss='warp')
model.fit(interactions, epochs=20)

# Get recommendations for a user
user_id = 3
n_users, n_items = interactions.shape
recommendations = model.predict(user_id, range(n_items))

# Print recommended item IDs
print("Recommended Item IDs:", recommendations)
