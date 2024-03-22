import pickle

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(680, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 680),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu, log_var = x.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(model, data_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(data_loader):
        data = data.to("cuda")
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    average_loss = train_loss / len(data_loader.dataset)
    print(f"Average loss: {average_loss:.4f}")


def load_data(data_names: list[str]) -> list:
    data = []
    for i in data_names:
        match i:
            case "data":
                with open("tmp/data.pkl", "rb") as f:
                    data.append(pickle.load(f))
            case "data_id_dict":
                with open("tmp/data_id_dict.pkl", "rb") as f:
                    data.append(pickle.load(f))
            case "child_id_dict":
                with open("tmp/child_id_dict.pkl", "rb") as f:
                    data.append(pickle.load(f))
            case "word_dict":
                with open("tmp/word_dict.pkl", "rb") as f:
                    data.append(pickle.load(f))
            case "category_dict":
                with open("tmp/category_dict.pkl", "rb") as f:
                    data.append(pickle.load(f))
    return data
