import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

# --------------------------
# Config
# --------------------------

# GPU Availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Directories
base_dir = os.path.join(os.getcwd(), "data", "books")
data_dir = os.path.join(base_dir, "cleaned")
output_dir = os.path.join(base_dir, "processed")
users_dir = os.path.join(data_dir, "Users.csv")
books_dir = os.path.join(data_dir, "Books.csv")
ratings_dir = os.path.join(data_dir, "Ratings.csv")

# --------------------------
# Data Loading
# --------------------------

users = pd.read_csv(users_dir)
books = pd.read_csv(books_dir, low_memory=False)
ratings = pd.read_csv(ratings_dir)

# --------------------------
# Vocabulary Config
# --------------------------

# Vocab size
max_users = 1000000
max_books = 1000000
max_authors = 150000
max_publishers = 50000

# --------------------------
# Node Indexing
# --------------------------

# User Id indexing
user2idx = {str(u): i for i, u in enumerate(users["User-ID"].tolist()[:max_users])}
# IBSN indexing
book2idx = {str(b): i for i, b in enumerate(books["ISBN"].tolist()[:max_books])}

# --------------------------
# Feature transformation
# --------------------------

# Embedding the uesr location
location_embeddings = model.encode(users["Location"].tolist(), show_progress_bar=True, convert_to_numpy=True, device=device, batch_size=64)

# Encoding the Book Author
books["Book-Author"] = books["Book-Author"].apply(lambda x: x.lower().strip())
author2idx = {str(a): i+1 for i, a in enumerate(books["Book-Author"].unique()[:max_authors])}
books["author_idx"] = books["Book-Author"].apply(lambda x: author2idx.get(x))

# Encoding the Book Publisher
books["Publisher"] = books["Publisher"].apply(lambda x: x.lower().strip())
publisher2idx = {str(p).lower(): i+1 for i, p in enumerate(books["Publisher"].unique()[:max_publishers])}
books["publisher_idx"] = books["Publisher"].apply(lambda x: publisher2idx.get(x))

# Converting the Year of Publication to numeric
books["year_of_publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce").fillna(0).astype(int)

# Embedding the Book Title
title_embeddings = model.encode(books["Book-Title"].tolist(), show_progress_bar=True, convert_to_numpy=True, device=device, batch_size=64)

# --------------------------
# Graph Building
# --------------------------

# User node Features
user_features = location_embeddings

# Book node Features
book_features = np.concatenate([
    books["author_idx"].to_numpy().reshape(-1, 1),
    books["publisher_idx"].to_numpy().reshape(-1, 1),
    books["year_of_publication"].to_numpy().reshape(-1, 1),
    title_embeddings    
], axis=1)

data = HeteroData()

data["user"].x = torch.tensor(user_features, dtype=torch.float)
data["book"].x = torch.tensor(book_features, dtype=torch.float)

src = [user2idx[str(u)] for u in ratings["User-ID"]]
dst = [book2idx[str(b)] for b in ratings["ISBN"]]
edge_index = torch.tensor([src, dst], dtype=torch.long)
edge_attr = torch.tensor(ratings["Book-Rating"].values, dtype=torch.float).view(-1, 1)

# User to Book edge
data["user", "rates", "book"].edge_index = edge_index
data["user", "rates", "book"].edge_attr = edge_attr

# Book to User edge
data["book", "rev_rates", "user"].edge_index = edge_index.flip(0)
data["book", "rev_rates", "user"].edge_attr = edge_attr

# --------------------------------------
# Saving the Graph and Feature Encoders
# --------------------------------------

os.makedirs(output_dir, exist_ok=True)
torch.save(data, os.path.join(output_dir, "graph.pt"))
with open(os.path.join(output_dir, "author_encoder.json"), "w") as f:
    json.dump(author2idx, f, indent=2)
with open(os.path.join(output_dir, "publisher_encoder.json"), "w") as f:
    json.dump(publisher2idx, f, indent=2)
print("Data processed successfully.")