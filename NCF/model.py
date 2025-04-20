import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dimension=4, mlp_layers=[16,8], dropout=0.3):
        """ 
        num_users: number of users
        num_items: number of items
        embedding_dimension: dimension of the embeddings for the matrix factorization
        mlp_layers: sizes of hidden mulit-layer-perceptron
        dropout: dropout probability for regularization
        """
        super(NCF, self).__init__()

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        #Matrix factorization embeddings
        self.user_embeddings_mf = nn.Embedding(num_users, embedding_dimension)
        self.item_embeddings_mf = nn.Embedding(num_items, embedding_dimension)

        #multi-layer perceptron embeddings
        self.user_embeddings_mlp = nn.Embedding(num_users, embedding_dimension)
        self.item_embeddings_mlp = nn.Embedding(num_items, embedding_dimension)

        #MLP layers with batch normalization
        mlp_modules = []
        input_size = embedding_dimension * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.BatchNorm1d(layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            input_size = layer_size
        
        self.mlp = nn.Sequential(*mlp_modules)

        predict_size = embedding_dimension + mlp_layers[-1]
        
        # Final prediction layer -> output single score
        self.final_layer = nn.Sequential(
            nn.Linear(predict_size, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        """
        user_ids: [batch_size]
        item_ids: [batch_size]
        returns: predicted score [batch_size, 1]
        """
        # GMF part
        user_gmf = self.dropout(self.user_embeddings_mf(user_ids))
        item_gmf = self.dropout(self.item_embeddings_mf(item_ids))
        gmf_output = user_gmf * item_gmf
        
        # MLP part
        user_mlp = self.dropout(self.user_embeddings_mlp(user_ids))
        item_mlp = self.dropout(self.item_embeddings_mlp(item_ids))
        mlp_input = torch.cat((user_mlp, item_mlp), dim=1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate GMF & MLP
        concat = torch.cat((gmf_output, mlp_output), dim=1)
        
        # Final layer
        preds = self.final_layer(concat)
        return preds

def train(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, weight_decay=1e-4):
    """
    Train the NCF model with early stopping and validation monitoring
    
    Args:
        model: NCF model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: maximum number of epochs
        learning_rate: learning rate for optimizer
        weight_decay: L2 regularization strength
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_users, batch_items, batch_ratings in train_loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_ratings = batch_ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_users.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_users, batch_items, batch_ratings in val_loader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_ratings = batch_ratings.to(device)
                
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                val_loss += loss.item() * batch_users.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

def data_split(df):
    """
    Split the data into training and test sets

    df: pandas dataframe in this format:
    user, item, rating
    """
    # Sort by user and (optionally) timestamp or item if available
    df = df.sort_values(by=["user", "item"])  # if no timestamp
    # Define the split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Get unique users
    unique_users = df['user'].unique()
    
    # Split users into two groups: those for train/val and those for test
    # We'll use 80% of users for train/val and include some of their interactions in test
    # The remaining 20% of users will only appear in test (cold-start scenario)
    np.random.seed(42)  # For reproducibility
    train_val_users = np.random.choice(
        unique_users, 
        size=int(0.8 * len(unique_users)), 
        replace=False
    )
    cold_start_users = np.array(list(set(unique_users) - set(train_val_users)))
    
    # Create dataframes for each user group
    train_val_df = df[df['user'].isin(train_val_users)]
    cold_start_df = df[df['user'].isin(cold_start_users)]
    
    # For train/val users, split their interactions
    train_val_indices = train_val_df.index.values
    np.random.shuffle(train_val_indices)
    
    train_size = int(len(train_val_indices) * (train_ratio / (train_ratio + val_ratio)))
    
    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]
    
    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    
    # For test set, combine some interactions from train/val users and all cold-start users
    # Sample some interactions from train/val users for the test set
    train_val_test_size = int(len(train_val_df) * (test_ratio / (train_ratio + val_ratio + test_ratio)))
    train_val_test_indices = np.random.choice(train_val_indices, size=train_val_test_size, replace=False)
    
    # Combine with cold-start users
    test_df = pd.concat([df.loc[train_val_test_indices], cold_start_df])
    
    print(f"Training set: {len(train_df)} interactions, {train_df['user'].nunique()} users")
    print(f"Validation set: {len(val_df)} interactions, {val_df['user'].nunique()} users")
    print(f"Test set: {len(test_df)} interactions, {test_df['user'].nunique()} users")
    print(f"Test set includes {cold_start_df['user'].nunique()} new users not seen during training")
    
    return train_df, val_df, test_df
    
def get_top_k_recommendations(model, user_id, k=10, exclude_interacted=True, item_ids=None):
    """
    Get top-k item recommendations for a given user.
    
    Args:
        model: Trained NCF model
        user_id: ID of the user to get recommendations for
        k: Number of recommendations to return
        exclude_interacted: Whether to exclude items the user has already interacted with
        item_ids: List of all item IDs. If None, will use all items in the model.
    
    Returns:
        tuple: (top_k_items, top_k_scores) where:
            top_k_items: List of k recommended item IDs
            top_k_scores: List of k predicted scores for the recommended items
    """
    device = next(model.parameters()).device
    
    # If item_ids not provided, use all items in the model
    if item_ids is None:
        item_ids = torch.arange(model.item_embeddings_mf.num_embeddings, device=device)
    
    # Create user tensor (repeat user_id for each item)
    user_tensor = torch.full((len(item_ids),), user_id, device=device)
    
    # Get predictions for all items
    with torch.no_grad():
        model.eval()
        predictions = model(user_tensor, item_ids)
        predictions = predictions.squeeze()
    
    # If excluding interacted items, we need to know which items the user has interacted with
    if exclude_interacted:
        # You'll need to pass the user's interaction history
        # This assumes you have a way to get the user's interacted items
        interacted_items = df[df['user'] == user_id]['item'].unique().tolist()
        mask = torch.ones(len(item_ids), dtype=torch.bool, device=device)
        for item in interacted_items:
            mask[item] = False
        predictions = predictions[mask]
        item_ids = item_ids[mask]
    
    # Get top k items and their scores
    top_k_scores, top_k_indices = torch.topk(predictions, k)
    
    return item_ids[top_k_indices].cpu().numpy(), top_k_scores.cpu().numpy()

def evaluate_recommendations(model, test_df, k=10):
    """
    Evaluate the model's recommendations using common metrics.
    
    Args:
        model: Trained NCF model
        test_df: DataFrame containing test interactions
        k: Number of recommendations to evaluate
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    device = next(model.parameters()).device
    all_items = torch.arange(model.item_embeddings_mf.num_embeddings, device=device)
    
    # Initialize metrics
    hits = 0
    total_recommendations = 0
    total_test_items = 0
    
    # For each user in test set
    for user_id in test_df['user'].unique():
        # Get user's test items
        test_items = test_df[test_df['user'] == user_id]['item'].unique()
        total_test_items += len(test_items)
        
        # Get recommendations
        recommended_items, _ = get_top_k_recommendations(
            model, user_id, k=k, exclude_interacted=True, item_ids=all_items
        )
        total_recommendations += len(recommended_items)
        
        # Count hits
        hits += len(set(recommended_items) & set(test_items))
    
    # Calculate metrics
    precision = hits / total_recommendations
    recall = hits / total_test_items
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1
    }

def get_item_asin(item_id, item_encoder):
    """
    Convert an encoded item ID back to its original ASIN.
    
    Args:
        item_id: The encoded item ID (integer)
        item_encoder: The scikit-learn LabelEncoder used to encode the items
        
    Returns:
        str: The original ASIN of the item
    """
    return item_encoder.inverse_transform([item_id])[0]

def get_item_asins(item_ids, item_encoder):
    """
    Convert a list of encoded item IDs back to their original ASINs.
    
    Args:
        item_ids: List of encoded item IDs (integers)
        item_encoder: The scikit-learn LabelEncoder used to encode the items
        
    Returns:
        list: List of original ASINs
    """
    return item_encoder.inverse_transform(item_ids).tolist()
    

