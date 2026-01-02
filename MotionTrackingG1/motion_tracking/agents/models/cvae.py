import torch
from torch import nn
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, action_dim, latent_dim):
        super(CVAE, self).__init__()
        # self.input_dim = input_dim
        # self.condition_dim = condition_dim
        # self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(1024, latent_dim)
        self.logvar_layer = nn.Linear(1024, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        combined = torch.cat([z, c], dim=1)
        return self.decoder(combined)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
