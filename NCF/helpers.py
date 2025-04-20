import numpy as np
import pandas as pd

def split_data(df, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15, special_test_ratio=0.5):
    """
    Split the data into training, validation, and test sets according to specified ratios.
    The test set is further split into regular test users and special test users (completely new users).
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['user', 'item', 'rating']
        train_ratio (float): Proportion of users for training (default: 0.75)
        val_ratio (float): Proportion of users for validation (default: 0.1)
        test_ratio (float): Proportion of users for test set (default: 0.15)
        special_test_ratio (float): Proportion of test users to use as special test set (default: 0.5)
    
    Returns:
        tuple: (train_df, val_df, test_df, special_test_df)
    """
    # Get unique users and shuffle them to ensure randomness
    unique_users = df['user'].unique()
    np.random.shuffle(unique_users)
    
    # Calculate split indices
    num_users = len(unique_users)
    train_end = int(num_users * train_ratio)
    val_end = train_end + int(num_users * val_ratio)
    test_end = val_end + int(num_users * test_ratio)
    
    # Split users
    train_users = unique_users[:train_end]
    val_users = unique_users[train_end:val_end]
    test_users = unique_users[val_end:test_end]
    
    # Further split test users into regular and special test sets
    num_test_users = len(test_users)
    special_test_end = int(num_test_users * special_test_ratio)
    regular_test_users = test_users[special_test_end:]
    special_test_users = test_users[:special_test_end]
    
    # Create the splits
    train_df = df[df['user'].isin(train_users)]
    val_df = df[df['user'].isin(val_users)]
    test_df = df[df['user'].isin(regular_test_users)]
    special_test_df = df[df['user'].isin(special_test_users)]
    
    return train_df, val_df, test_df, special_test_df

def create_split_datasets(df, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15, special_test_ratio=0.5):
    """
    Create datasets for each split of the data.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['user', 'item', 'rating']
        train_ratio (float): Proportion of users for training
        val_ratio (float): Proportion of users for validation
        test_ratio (float): Proportion of users for test set
        special_test_ratio (float): Proportion of test users to use as special test set
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, special_test_dataset)
    """
    # Split the data
    train_df, val_df, test_df, special_test_df = split_data(
        df, train_ratio, val_ratio, test_ratio, special_test_ratio
    )
    
    # Create datasets for each split
    train_dataset = RatingDataset(train_df)
    val_dataset = RatingDataset(val_df)
    test_dataset = RatingDataset(test_df)
    special_test_dataset = RatingDataset(special_test_df)
    
    return train_dataset, val_dataset, test_dataset, special_test_dataset

def create_split_matrices(user_reviews, rating_weight_map, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15):
    """
    Create binary and weighted matrices for each split of the data.
    
    Args:
        user_reviews (dict): Dictionary mapping user IDs to their reviews
        rating_weight_map (dict): Mapping of ratings to weights
        train_ratio (float): Proportion of users for training
        val_ratio (float): Proportion of users for validation
        test_ratio (float): Proportion of users for regular test set
    
    Returns:
        tuple: (train_matrices, val_matrices, test_matrices, special_test_matrices)
        where each matrices tuple contains (binary_matrix, weighted_matrix, user_ids, item_ids)
    """
    # Split the data
    train_reviews, val_reviews, test_reviews, special_test_reviews = split_data(
        user_reviews, train_ratio, val_ratio, test_ratio
    )
    
    # Create matrices for each split
    train_matrices = create_user_item_matrices(train_reviews, rating_weight_map)
    val_matrices = create_user_item_matrices(val_reviews, rating_weight_map)
    test_matrices = create_user_item_matrices(test_reviews, rating_weight_map)
    special_test_matrices = create_user_item_matrices(special_test_reviews, rating_weight_map)
    
    return train_matrices, val_matrices, test_matrices, special_test_matrices