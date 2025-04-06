import torch
import torch.nn as nn
import torch.nn.functional as F

# Variational Autoencoder for Kindle eBook recommender system
# ASSUMPTIONS:
# input - binary user-item vector of size num_items
# encoder - one hidden layer with hidden_dim neurons
# latent space - gaussian approx. posterior distribution with latent_dim neurons
# decoder - maps latent space back to logits over all eBooks
# PARAMETERS:
# num_items: number of items in the binary matrix
# hidden_dim: number of neurons in hidden layer
# latent_dim: number of neurons in latent space (dimensionality)
class VAE(nn.Module):
    def __init__(self, num_items, hidden_dim=600, latent_dim=200):
        super(VAE, self).__init__()

        # encoder behavior - maps inputs to hidden layer, outputs paramters for approx. function, then to latent space
        self.fc1 = nn.Linear(num_items, hidden_dim)
        self.hid_mu = nn.Linear(hidden_dim, latent_dim)   # mu - mean
        self.hid_sigma = nn.Linear(hidden_dim, latent_dim)   # sigma - log variance

        #decoder behavior - map latent space back to hidden and output logits
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_items)

    def encode(self, x):
        # encode input x into latent space
        h1 = torch.tanh(self.fc1(x))
        mu = self.hid_mu(h1)
        sigma = self.hid_sigma(h1)
        return mu, sigma

    def reparameterize(self, mu, logvar):
        # reparameterization trick for backpropogation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # decodes latent variable z to logits over all items
        h3 = torch.tanh(self.fc3(z))
        logits = self.fc4(h3)  # no need activation function since loss function takes care of it
        return logits

    def forward(self, x):
        # forward pass of the VAE
        # RETURNS:
        # logits - logits for each item
        # mu - mean of gaussian
        # sigma - log variance of gaussian
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        logits = self.decode(z)
        return logits, mu, sigma
    
