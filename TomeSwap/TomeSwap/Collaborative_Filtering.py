# Collaborative Filtering Algorithm (Example: Surprise library)
from surprise import Dataset, Reader, SVD

# Load dataset (replace 'ratings.csv' with your dataset)
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file('ratings.csv', reader)

# Build and train the collaborative filtering model
model = SVD()
trainset = data.build_full_trainset()
model.fit(trainset)

# Get recommendations for a user (replace 'user_id' with the actual user ID)
user_id = '123'
recommendations = model.get_neighbors(trainset.to_inner_uid(user_id), k=10)

# Print recommended book IDs
print("Recommended Book IDs:", recommendations)
