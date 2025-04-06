import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# loads reviews in the json
def load_reviews(file_path):
    reviews = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                reviews.append(json.loads(line))
    return reviews

# maps each userID to a list opf interactions
# each interaction is a list of asin, the amazon id, and overall, the user rating of each eBook
def create_review_dict(reviews):
    users_reviews = {}
    for r in reviews:
        user = r["reviewerID"]
        asin = r["asin"]
        overall = r["overall"]
        if user not in users_reviews:
            users_reviews[user] = []
        users_reviews[user].append([asin, overall])
    return users_reviews

# makes a binary user-item matrix (1 if user reviewed the book, 0 if not)
# makes a weight matrix using the rating weight mapping
# returns a binary matrix of user-eBook reviews, a weighted matrix for each rating per user
# returns a dictionary mapping user IDs to matrix rows
# returns a dictionary mapping book asins to matrix columns 
def create_user_item_matrices(user_reviews, rating_weight_map):
    user_ids = {}
    item_ids = {}

    # build user and item mappings
    for user, interactions in user_reviews.items():
        if user not in user_ids:
            user_ids[user] = len(user_ids)
        for interaction in interactions:
            asin = interaction[0]
            if asin not in item_ids:
                item_ids[asin] = len(item_ids)
    
    num_users = len(user_ids)
    num_items = len(item_ids)
    
    # initialize matrices
    binary_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    weighted_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    
    # populate matrices
    for user, interactions in user_reviews.items():
        u_id = user_ids[user]
        for interaction in interactions:
            asin, rating = interaction[0], interaction[1]
            if asin in item_ids:
                i_idx = item_ids[asin]
                binary_matrix[u_id, i_idx] = 1.0
                # use the argument rating_weight_map which defines the weight hyperparamters
                # weights default to 1 if not specified
                weight = rating_weight_map.get(rating, 1.0)
                weighted_matrix[u_id, i_idx] = weight

    return binary_matrix, weighted_matrix, user_ids, item_ids
