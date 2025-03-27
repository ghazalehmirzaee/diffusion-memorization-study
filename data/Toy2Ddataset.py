import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
import os


def moons_dataset(n=8000):
    """Generate a two moons dataset exactly as in tiny-diffusion."""
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return X


def line_dataset(n=8000):
    """Generate a line dataset exactly as in tiny-diffusion."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return X


def circle_dataset(n=8000):
    """Generate a circle dataset exactly as in tiny-diffusion."""
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x ** 2 + y ** 2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return X


def dino_dataset(n=8000):
    """Generate a dinosaur dataset exactly as in tiny-diffusion."""
    try:
        # Load the dataset from the local file
        df = pd.read_csv("DatasaurusDozen.tsv", sep="\t")
        df = df[df["dataset"] == "dino"]

        # Random sampling with specific seed
        rng = np.random.default_rng(42)
        ix = rng.integers(0, len(df), n)

        # Get x and y coordinates with small noise
        x = df["x"].iloc[ix].to_numpy()
        x = x + rng.normal(size=len(x)) * 0.15

        y = df["y"].iloc[ix].to_numpy()
        y = y + rng.normal(size=len(x)) * 0.15

        # Apply exact scaling from tiny-diffusion
        x = (x / 54 - 1) * 4
        y = (y / 48 - 1) * 4

        return np.stack((x, y), axis=1)

    except Exception as e:
        raise RuntimeError(f"Error loading dinosaur dataset: {e}. Make sure 'DatasaurusDozen.tsv' exists.")


def swiss_roll_dataset(n=8000):
    """Generate a swiss roll dataset."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n)
    x = t * np.cos(t) / (4 * np.pi)
    y = t * np.sin(t) / (4 * np.pi)
    X = np.column_stack((x, y)) * 0.75 + np.array([0.25, -0.25])
    return X


def get_dataset(name, n=8000):
    """Get a specific dataset by name as a PyTorch TensorDataset."""
    if name.lower() == "moons":
        data = moons_dataset(n)
    elif name.lower() == "dino":
        data = dino_dataset(n)
    elif name.lower() == "line":
        data = line_dataset(n)
    elif name.lower() == "circle":
        data = circle_dataset(n)
    elif name.lower() == "swiss_roll":
        data = swiss_roll_dataset(n)
    elif name.lower() in get_datasaurus_dozen_datasets():
        data = get_datasaurus_dataset(name.lower(), n)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return TensorDataset(torch.from_numpy(data.astype(np.float32)))


def get_raw_data(name, n=8000):
    """Get raw numpy array for a dataset."""
    if name.lower() == "moons":
        return moons_dataset(n)
    elif name.lower() == "dino":
        return dino_dataset(n)
    elif name.lower() == "line":
        return line_dataset(n)
    elif name.lower() == "circle":
        return circle_dataset(n)
    elif name.lower() == "swiss_roll":
        return swiss_roll_dataset(n)
    elif name.lower() in get_datasaurus_dozen_datasets():
        return get_datasaurus_dataset(name.lower(), n)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(dataset, batch_size=128, shuffle=True):
    """Create a DataLoader from a dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_datasaurus_dozen_datasets():
    """Get list of all dataset names in the Datasaurus Dozen collection."""
    if os.path.exists("DatasaurusDozen.tsv"):
        df = pd.read_csv("DatasaurusDozen.tsv", sep="\t")
        return df["dataset"].unique().tolist()
    return []


def get_datasaurus_dataset(which_set="dino", n=8000):
    """Get a specific dataset from the Datasaurus Dozen collection."""
    try:
        # Read the dataset file
        df = pd.read_csv("DatasaurusDozen.tsv", sep="\t")

        # Filter for the requested dataset
        df_subset = df[df["dataset"] == which_set]

        if len(df_subset) == 0:
            raise ValueError(f"Dataset '{which_set}' not found in the Datasaurus Dozen collection")

        # Random sampling with reproducible seed
        rng = np.random.default_rng(42)
        ix = rng.integers(0, len(df_subset), n)

        # Get x and y coordinates with smaller noise (reduced from 0.15 to 0.05)
        x = df_subset["x"].iloc[ix].to_numpy()
        x = x + rng.normal(size=len(x)) * 0.05

        y = df_subset["y"].iloc[ix].to_numpy()
        y = y + rng.normal(size=len(x)) * 0.05

        # Scale in the same way as tiny-diffusion
        x = (x / 54 - 1) * 4
        y = (y / 48 - 1) * 4

        return np.stack((x, y), axis=1)

    except Exception as e:
        raise RuntimeError(f"Error loading Datasaurus dataset '{which_set}': {e}")


def visualize_dataset(name, n=2000, figsize=(6, 6)):
    """Visualize a single dataset."""
    try:
        # Get the raw data
        data = get_raw_data(name, n)

        # Create the plot
        plt.figure(figsize=figsize)
        plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.7)
        plt.title(name.capitalize())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"Dataset: {name}")
        print(f"Number of points: {len(data)}")
        print(f"X mean: {np.mean(data[:, 0]):.4f}, X std: {np.std(data[:, 0]):.4f}")
        print(f"Y mean: {np.mean(data[:, 1]):.4f}, Y std: {np.std(data[:, 1]):.4f}")
        print(f"Correlation: {np.corrcoef(data[:, 0], data[:, 1])[0, 1]:.4f}")

    except Exception as e:
        print(f"Error visualizing dataset {name}: {e}")


def visualize_all_datasets(n=2000):
    """Visualize all standard datasets in a grid."""
    datasets = ["moons", "dino", "line", "circle", "swiss_roll"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, name in enumerate(datasets):
        try:
            data = get_raw_data(name, n)

            axes[i].scatter(data[:, 0], data[:, 1], s=5, alpha=0.7)
            axes[i].set_title(name.capitalize())
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            axes[i].grid(True)
            axes[i].set_aspect('equal')

        except Exception as e:
            print(f"Error visualizing dataset {name}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading {name}",
                         ha='center', va='center', transform=axes[i].transAxes)

    # Remove unused subplot
    if len(datasets) < len(axes):
        for j in range(len(datasets), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualize_forward_diffusion(name, n_steps=5, T=1000, beta_min=1e-4, beta_max=0.02, n=2000):
    """Visualize the forward diffusion process on a dataset."""
    try:
        # Get the dataset
        data = get_raw_data(name, n)

        # Convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Set up diffusion parameters
        betas = torch.linspace(beta_min, beta_max, T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Select timesteps to visualize
        timesteps = torch.linspace(0, T - 1, n_steps).long()

        # Create figure
        fig, axes = plt.subplots(2, n_steps, figsize=(15, 8))

        # Process each timestep
        for i, t in enumerate(timesteps):
            # For t=0, just use the original data
            if t == 0:
                noisy_data = data_tensor
            else:
                # Add noise according to the forward diffusion process
                noise = torch.randn_like(data_tensor)
                sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
                sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
                noisy_data = sqrt_alphas_cumprod_t * data_tensor + sqrt_one_minus_alphas_cumprod_t * noise

            # Convert to numpy for plotting
            noisy_np = noisy_data.numpy()

            # 2D scatter plot (bottom row)
            axes[1, i].scatter(noisy_np[:, 0], noisy_np[:, 1], s=2, alpha=0.3)
            axes[1, i].set_title(f"t={t.item()}")
            axes[1, i].set_xlabel("x")
            axes[1, i].set_ylabel("y")
            axes[1, i].grid(True)

            # 1D histogram (top row) - x values for simplicity
            axes[0, i].hist(noisy_np[:, 0], bins=50, alpha=0.7, density=True)
            axes[0, i].set_title(f"t={t.item()} (x-axis projection)")
            axes[0, i].set_xlabel("x")
            axes[0, i].set_ylabel("Density")
            axes[0, i].grid(True)

        # Add an overall title
        fig.suptitle(f"Forward Diffusion Process: {name.capitalize()} Dataset", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the title
        plt.show()

    except Exception as e:
        print(f"Error visualizing diffusion process for {name}: {e}")


# Example usage
if __name__ == "__main__":
    # Check if the DatasaurusDozen.tsv file exists
    if os.path.exists("DatasaurusDozen.tsv"):
        print("DatasaurusDozen.tsv file found!")

        # Visualize other datasets
        visualize_all_datasets(n=2000)

        # Visualize forward diffusion on one dataset
        visualize_forward_diffusion("dino", n_steps=5, T=1000)
    else:
        print("DatasaurusDozen.tsv file not found! Please ensure it's in the current directory.")

