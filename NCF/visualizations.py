import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import torch


def plot_embeddings(model, df, sample_size=1000, save_path=None):
    """Plot PCA visualization of user and item embeddings."""
    model.eval()
    sample_users = torch.tensor(df['user'].unique()[:sample_size], dtype=torch.long)
    sample_items = torch.tensor(df['item'].unique()[:sample_size], dtype=torch.long)

    with torch.no_grad():
        user_embeddings = model.user_embeddings_mf(sample_users).numpy()
        item_embeddings = model.item_embeddings_mf(sample_items).numpy()

    pca = PCA(n_components=2)
    user_2d = pca.fit_transform(user_embeddings)
    item_2d = pca.fit_transform(item_embeddings)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(user_2d[:, 0], user_2d[:, 1], alpha=0.5)
    plt.title('User Embeddings (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(1, 2, 2)
    plt.scatter(item_2d[:, 0], item_2d[:, 1], alpha=0.5)
    plt.title('Item Embeddings (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_rating_distribution(df, save_path=None):
    """Bar chart of the distribution of ratings in the dataset.
    Args:
        df (pd.DataFrame): The dataset.
        save_path (str, optional): The path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(df['rating'].unique(), df['rating'].value_counts())
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(range(1, 6))
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_error_distribution(model, test_loader, save_path=None):
    """
    This function plots the distribution of prediction errors.
    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        save_path (str, optional): The path to save the plot. Defaults to None.
    """
    model.eval()
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for batch_users, batch_items, batch_ratings in test_loader:
            predictions = model(batch_users, batch_items)
            all_predictions.extend(predictions.numpy())
            all_actuals.extend(batch_ratings.numpy())

    errors = np.array(all_actuals) - np.array(all_predictions)
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.title('Error Distribution (Histogram)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Density')
    
    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(errors, vert=False)
    plt.title('Error Distribution (Box Plot)')
    plt.xlabel('Error (Actual - Predicted)')
    
    # Add statistics as text
    stats_text = f"""
    Mean Error: {np.mean(errors):.4f}
    Median Error: {np.median(errors):.4f}
    Std Dev: {np.std(errors):.4f}
    Min Error: {np.min(errors):.4f}
    Max Error: {np.max(errors):.4f}
    """
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(model, test_loader, criterion, save_path=None):
    """Plot ROC curve for binary classification performance using the training criterion.
    Args:
        model (torch.nn.Module): The trained model
        test_loader (torch.utils.data.DataLoader): DataLoader containing test data
        criterion: The loss function used during training (e.g., BCEWithLogitsLoss)
        save_path (str, optional): Path to save the plot. Defaults to None.
    Returns:
        float: The AUC (Area Under Curve) score
    """
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_users, batch_items, batch_ratings in test_loader:
            # Get model outputs
            outputs = model(batch_users, batch_items)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).squeeze()
            
            # Convert ratings to binary labels (1 for high ratings, 0 for low)
            # Assuming ratings are normalized between 0 and 1
            labels = (batch_ratings > 0.5).float()
            
            all_predictions.extend(probs.cpu().numpy())
            all_actuals.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    
    # Ensure we have binary classification data
    if len(np.unique(actuals)) != 2:
        raise ValueError("Data must contain exactly two classes for ROC curve")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(actuals, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Calculate optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    # Plot optimal threshold point
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
             label=f'Optimal threshold: {optimal_threshold:.2f}')
    
    # Plot random guessing line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random guessing')
    

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_training_history(running_loss, save_path=None):
    """This function plots the training loss over epochs.
    Args:
        running_loss (list): A list of training loss values.
        save_path (str, optional): The path to save the plot. Defaults to None.   
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(range(1, len(running_loss) + 1), running_loss, 
             'r-', linewidth=2, marker='o', markersize=8, 
             label='Training Loss')
    
    plt.title('Training Loss Over Epochs', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(range(1, len(running_loss) + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()