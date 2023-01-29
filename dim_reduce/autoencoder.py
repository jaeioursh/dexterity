'''
Train an autoencoder to cram many hand dimensions into a little tiny latent space
'''
import torch
from torch import nn
import numpy as np
import pickle
import tqdm
from math import ceil


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
            layers.append(nn.Sigmoid())

        # Add this layer's linear units
        layers.append(nn.Linear(n_in, n_out))

    # Build into a feedforward
    model = nn.Sequential(*layers)
    return model


class Autoencoder(nn.Module):
    def __init__(self, encoder_dims):
        super().__init__()

        self.encoder = model_from_list(encoder_dims)
        self.encoder.add_module("sigmoid_1", nn.Sigmoid())  # Constrains latent dims to [0,1]

        encoder_dims.reverse()
        self.decoder = model_from_list(encoder_dims)
        encoder_dims.reverse()  # put it back how we found it :)

    def forward(self, x):
        x = self.encoder.forward(x)  # Encode
        x = self.decoder.forward(x)  # Decode
        return x


def main():
    # Architecture for our model
    n_dims = [48]
    n_hid = [100,24]
    n_latent = [6]
    layers = n_dims + n_hid + n_latent  # list addition

    model = Autoencoder(layers)

    # Import data
    with open("hand_data.pkl", "rb") as f:
        data = pickle.load(f)
    mins, ranges = dataset_stats(data)
    data_norm = normalize(data, mins, ranges)
    #data_norm=data
    # Split to train/val
    train_ratio = 0.8
    n_t = round(train_ratio * data_norm.shape[1])
    np.random.shuffle(data_norm.T)  # In-place shuffle along instance dimension
    train = data_norm[:, :n_t]
    val = data_norm[:, n_t:]

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    train_torch = torch.Tensor(train.T).to(device)
    val_torch = torch.Tensor(val.T).to(device)

    # Set up the good stuff
    lr = 1e-2
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)


    # Train
    n_cols = train_torch.size()[0]
    print(n_cols)
    batchsize = 256
    epochs = 2000
    losses = []

    pbar = tqdm.trange(epochs, dynamic_ncols=True)
    try:
        for epoch in pbar:
            # Minibatches
            cols = torch.randperm(n_cols)
            
            for i in range(ceil(n_cols / batchsize)):
                # Build minibatch
                start = i * batchsize
                end = min((i+1) * batchsize, n_cols)
                batch = train_torch[start:end,:]

                # Forward pass
                optim.zero_grad()
                out = model.forward(batch)
                l = loss_fn(batch, out)

                # Backward pass
                l.backward()
                optim.step()
                

            
                # Only validate and update progress bar every 100 epochs
                if i % 100 == 0:
                    # Get avg loss from minibatches
                   

                    # Validation
                    out = model.forward(val_torch)
                    loss_val = loss_fn(val_torch, out)

                    #losses.append(loss_val.detach().cpu())  # Need to detach or memory leaks
                    pbar.set_description(f"Val: {loss_val:.3E}")

    except KeyboardInterrupt:
        print("Ending early!")

    layerstr = "_".join([str(l) for l in layers])
    with open(f"ae_trained_{layerstr}.pkl", "wb") as f:
        out = {
            'model': model.cpu(),
            'mins': mins.flatten(),
            'ranges': ranges.flatten(),
            'loss': loss_val,
            'architecture': layers,
            'params': (epochs, batchsize, lr),
        }
        pickle.dump(out, f)
        print("Saved model!")


if __name__=="__main__":
    main()
