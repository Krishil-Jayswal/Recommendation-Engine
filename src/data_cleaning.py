import os
import pandas as pd

base_dir = os.path.join(os.getcwd(), "data", "books")
output_dir = os.path.join(base_dir, "cleaned")
data_dir = os.path.join(base_dir, "raw")
users_file = "Users.csv"
books_file = "Books.csv"
ratings_file = "Ratings.csv"
users_dir = os.path.join(data_dir, users_file)
books_dir = os.path.join(data_dir, books_file)
ratings_dir = os.path.join(data_dir, ratings_file)

users = pd.read_csv(users_dir)
books = pd.read_csv(books_dir, low_memory=False)
ratings = pd.read_csv(ratings_dir)

# Drop the age column
users.drop(columns=["Age"], inplace=True)
# Drop the null values
users.dropna(inplace=True)

# Drop the image url's column
books.drop(columns=books.columns[-3:], inplace=True)
# Drop the null values
books.dropna(inplace=True)

# Valid user and book nodes
num_users = users.shape[0]
valid_books = set(books["ISBN"])

# Validating ratings
ratings = ratings[(ratings["User-ID"] >= 1) & (ratings["User-ID"] <= num_users)]
ratings = ratings[ratings["ISBN"].isin(valid_books)]
ratings.reset_index(inplace=True)

# Saving the cleaned data
os.makedirs(output_dir, exist_ok=True)
users.to_csv(os.path.join(output_dir, users_file), index=False)
books.to_csv(os.path.join(output_dir, books_file), index=False)
ratings.to_csv(os.path.join(output_dir, ratings_file), index=False)

print("Data cleaned successfully.")