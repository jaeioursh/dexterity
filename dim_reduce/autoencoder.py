'''
Train an autoencoder to cram many hand dimensions into a little tiny latent space
'''
import torch
from torch import nn
import numpy as np
import pickle
import tqdm


def dataset_stats(x):
    """
    Get stats for normalizing data
    """
    maxes = np.atleast_2d(np.max(x, axis=1)).T
    mins = np.atleast_2d(np.min(x, axis=1)).T
    ranges = maxes - mins
    return mins, ranges


def normalize(x, mins, ranges):
    """
    Smush everything into [0,1] so big dummy model learns good
    @param x: the data to be smushed
    @param mins: original min for each dimension
    @param ranges: original range for each dimension
    @return x_norm: normalized data
    """
    return (x - mins) / ranges


def unnormalize(x_norm, mins, ranges):
    """
    Un-smush back to original data
    @param x_norm: the data to unsmushify
    @param mins: the mins of the original data
    @param ranges: the ranges of the original data
    @return x: unsmushed data
    """
    return (x_norm * ranges) + mins


def model_from_list(l):
    """
    Build a simple relu feedforward net from a list of layer dims.
    """
    layers = []
    for i in range(len(l) - 1):
        n_in = l[i]
        n_out = l[i + 1]

        # Conceptualizing activations "before" layers means we don't have to
        # handle the final hidden->output case as an exception
        if i > 0:
            layers.append(nn.ReLU())

        # Add this layer's linear units
        layers.append(nn.Linear(n_in, n_out))

    # Build into a feedforward
    model = nn.Sequential(*layers)
    return model


class Autoencoder(nn.Module):
    def __init__(self, encoder_dims):
        super().__init__()

        self.encoder = model_from_list(encoder_dims)
        encoder_dims.reverse()
        self.decoder = model_from_list(encoder_dims)

    def forward(self, x):
        x = self.encoder.forward(x)  # Encode
        x = self.decoder.forward(x)  # Decode
        return x


def main():
    # Architecture for our model
    n_dims = 48
    n_hid1 = 30
    n_hid2 = 20
    n_hid3 = 10
    n_latent = 5
    layers = [n_dims, n_hid1, n_hid2, n_hid3, n_latent]

    model = Autoencoder(layers)

    # Import data
    with open("hand_data.pkl", "rb") as f:
        data = pickle.load(f)
    mins, ranges = dataset_stats(data)
    data_norm = normalize(data, mins, ranges)
    np.random.shuffle(data_norm.T)  # In-place shuffle along t dimension

    # Split to train/val
    train_ratio = 0.8
    n_t = round(train_ratio * data_norm.shape[1])
    train = data_norm[:, :n_t]
    val = data_norm[:, n_t:]

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_torch = torch.Tensor(train.T).to(device)
    val_torch = torch.Tensor(val.T).to(device)

    # Set up the good stuff
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=10000,
        gamma=0.5,
    )

    # Train
    epochs = 100000
    losses = []

    pbar = tqdm.trange(epochs)
    try:
        for epoch in pbar:
            # Batch gradient descent for Big Speed
            optim.zero_grad()
            out = model.forward(train_torch)
            loss = loss_fn(train_torch, out)

            loss.backward()
            optim.step()
            scheduler.step()

            # Validation
            out = model.forward(val_torch)
            loss_val = loss_fn(val_torch, out)

            losses.append(loss.detach().cpu())  # Need to detach or memory leaks
            pbar.set_description(f"Train: {loss:.3E}, Val: {loss_val:.3E}, LR: {scheduler.get_last_lr()[0]:.3E}")

            if epoch % 20 == 0:
                torch.cuda.empty_cache()  # Not actually sure if this helped
    except KeyboardInterrupt:
        print("Ending early!")

    with open("ae_trained.pkl", "wb") as f:
        out = {
            'model': model.cpu(),
            'mins': mins.flatten(),
            'ranges': ranges.flatten(),
        }
        pickle.dump(out, f)
        print("Saved model!")


if __name__=="__main__":
    main()
