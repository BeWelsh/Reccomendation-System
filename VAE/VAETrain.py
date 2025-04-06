import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from VAEModel import VAE
from DataToMatrix import load_reviews, create_review_dict, create_user_item_matrices
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import json

# ==================
# LOAD AND PREP DATA
# ==================

reviews_file = "../Kindle_Store_5.json"

# hyperparameters for rating weights
rating_weight_map = {
    1: 2.5,  # high penalty for very negative rating
    2: 2.0,
    3: 1.0,  # neutral rating
    4: 0.5,
    5: 0.1   # low penalty for perfect rating
}


# load and create user interaction dictionary
reviews = load_reviews(reviews_file)
user_reviews = create_review_dict(reviews)

# create the matrices
binary_matrix, weighted_matrix, user2id, item2id = create_user_item_matrices(user_reviews, rating_weight_map)
num_users = binary_matrix.shape[0]
num_items = binary_matrix.shape[1]
total_reviews = int(np.sum(binary_matrix))

print("Number of users:", num_users)
print("Number of items:", num_items)
print("Total reviews in matrix:", total_reviews)

# ====================================
# TRAIN-TEST SPLIT USING LEAVE-ONE-OUT
# ====================================

# for each user randomly hold out one review for test
train_binary = binary_matrix.copy()
train_weighted = weighted_matrix.copy()
test_binary = np.zeros_like(binary_matrix, dtype=np.float32)

for u in range(num_users):
    pos_items = np.where(binary_matrix[u] == 1)[0]
    if len(pos_items) > 0:
        # randomly select one review
        test_idx = np.random.choice(pos_items, 1, replace=False)[0]
        train_binary[u, test_idx] = 0
        train_weighted[u, test_idx] = 0  # remove weights from the training for exclusion from test
        test_binary[u, test_idx] = 1

# Define a custom Dataset that converts each user's row from the sparse matrix to a dense tensor on the fly.
class UserItemDataset(Dataset):
    def __init__(self, binary_matrix, weighted_matrix):
        # Convert the numpy dense matrices into sparse CSR format.
        self.binary_sparse = csr_matrix(binary_matrix)
        self.weighted_sparse = csr_matrix(weighted_matrix)
        self.num_users = binary_matrix.shape[0]
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        # Get the idx-th row from each sparse matrix as a dense array.
        binary_row = self.binary_sparse.getrow(idx).toarray().squeeze(0)
        weighted_row = self.weighted_sparse.getrow(idx).toarray().squeeze(0)
        # Convert to torch tensors.
        binary_tensor = torch.tensor(binary_row, dtype=torch.float32)
        weighted_tensor = torch.tensor(weighted_row, dtype=torch.float32)
        return binary_tensor, weighted_tensor

# create TensorDataset and Dataloader for training
batch_size = 128
train_dataset = UserItemDataset(train_binary, train_weighted)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Training DataLoader created with", len(train_loader), "batches.")

class UserItemTestDataset(Dataset):
    def __init__(self, test_matrix):
        # Convert the test matrix into a sparse format
        self.test_sparse = csr_matrix(test_matrix)
        self.num_users = test_matrix.shape[0]
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        # Convert the idx-th row to dense format and then to a tensor
        test_row = self.test_sparse.getrow(idx).toarray().squeeze(0)
        test_tensor = torch.tensor(test_row, dtype=torch.float32)
        return test_tensor
    
test_dataset = UserItemTestDataset(test_binary)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print("Testing DataLoader created with", len(test_loader), "batches.")


# ===================================
# CREATE MODEL AND OPTIMIZER FUNCTION
# ===================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# define dimensions of encoder and latent space
hidden_dim = 600
latent_dim = 200

model = VAE(num_items=num_items, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

# create optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Uncomment when you want to use a pretrained model instead of retraining

model.load_state_dict(torch.load("vae_model_checkpoint.pth", map_location=device))
print("Loaded pretrained model checkpoint.")

# test forward pass with dummy input
dummy_batch_size = 16
dummy_input = torch.randint(0, 2, (dummy_batch_size, num_items)).float().to(device)
logits, mu, sigma = model(dummy_input)

# Expected values:
# dummy input shape: (dummy_batch_size, num_items)
# logits shape: (dummy_batch_size, num_items)
# mu shape: (dummy_batch_size, latent_dim)
# sigma shape: (dummy_batch_size, latent_dim)
print("Dummy input shape:", dummy_input.shape)
print("Logits shape:", logits.shape) 
print("Mu shape:", mu.shape)            
print("Sigma shape:", sigma.shape)    

# ====================
# DEFINE LOSS FUNCTION
# ====================

# LOSS FUNCTION: multinomial reconstruction loss and KL divergence
#
# weighted reconstruction loss (L_rec):
# L_rec = -sum_{i=1}^{n} (weight_i * x_i * log(p_i))
# where n is the number of items in vector, and p_i is the probability of item i
#
# KL divergence (KL):
# KL = -0.5 * sum(1 + sigma - mu^2 - exp(sigma))
#
# Total Loss:
# L = L_rec + KL
# 
# # PARAMETERS:
# logits: output of decoder
# x: target matrix
# mu: latent mean
# sigma: latent log variance
# weight: weight matrix
def loss_function(logits, x, mu, logvar, weight):

    # compute log probability with softmax
    log_prob = F.log_softmax(logits, dim=1)
    # weighted reconstruction loss calculated with targets and weights
    rec_loss = -torch.sum(weight * x * log_prob)
    # KL divergence, summing over latent dimensions and batch of sampels from DataLoader
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return total loss
    return rec_loss + kl_loss

# ========
# TRAINING
# ========
# num_epochs = 100
# train_losses = []

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0.0
#     for batch_idx, (batch_binary, batch_weighted) in enumerate(train_loader):
#         # move dataloader batch to GPU/CPU
#         batch_binary = batch_binary.to(device)
#         batch_weighted = batch_weighted.to(device)
        
#         optimizer.zero_grad()
#         logits, mu, sigma = model(batch_binary)
#         loss = loss_function(logits, batch_binary, mu, sigma, batch_weighted)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
    
#     avg_loss = epoch_loss / num_users
#     train_losses.append(avg_loss)
#     print(f"Epoch {epoch+1}/{num_epochs}, Average Loss per User: {avg_loss:.4f}")

# torch.save(model.state_dict(), "vae_model_checkpoint.pth")
# print("Model saved.")

# ================
# TESTING
# ================

# Tests the trained model
#
# Compute Recall@k for each user with Leave One Out
# each user had one review left out from the training data to use as test data
# model generates k recommendations for each user
# excludes known reviews
# Recall@k checks if the held out eBook appears in the top K recommendations
#
# PARAMETERS:
# train_data, test_data: tensors
def test(model, train_data, test_data, k=10):
    model.eval()
    predictions = []
    # create dataloader to process training data in mini batches
    dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch_data = batch[0].to(device)
            logits, _, _ = model(batch_data)
            predictions.append(logits.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    recalls = []
    for u in range(train_data.shape[0]):
        # get training items
        train_items = set(np.where(train_data[u] == 1)[0])
        # get held out test items
        true_items = set(np.where(test_data[u] == 1)[0])
        if len(true_items) == 0:
            continue
        scores = predictions[u]
        # set excluded eBooks to -inf so they won't be recommended
        scores[list(train_items)] = -np.inf
        recommended = np.argpartition(scores, -k)[-k:]
        hit_count = len(true_items.intersection(recommended))
        recalls.append(hit_count / len(true_items))
    
    return np.mean(recalls)

# recall10 = test(model, train_binary, test_binary, k=10)

# print(f"Recall@10: {recall10:.4f}")

try:
    recall10 = test(model, train_binary, test_binary, k=10)
    print(f"Recall@10: {recall10:.4f}")
except Exception as e:
    print("An error occurred during testing:", e)

# ============================
# PLOT TRAINING LOSS
# ============================
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss per User")
plt.grid(True)
plt.tight_layout()
plt.show()

# Suppose you have your evaluation results in a dictionary:
results = {
    "Recall@10": recall10,
    "Training Loss": train_losses,
    "Num Epochs": num_epochs
}

# Save the results to a JSON file:
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)
