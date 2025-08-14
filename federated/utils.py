import torch
import torchvision
from torch.utils.data import Dataset, Subset, ConcatDataset
import random

def load_and_duplicate_cifar(num_copies, transform):
    datasets = []
    for _ in range(num_copies):
        ds = torchvision.datasets.CIFAR10(
            root="./dataset", train=True, download=True, transform=transform
        )
        datasets.append(ds)
    return ConcatDataset(datasets)

def filter_animals(dataset, animal_classes):
    filtered_indices = [
        i for i, (_, label) in enumerate(dataset)
        if label in animal_classes
    ]
    return Subset(dataset, filtered_indices)

def remap_labels(dataset, animal_classes):
    class RemappedDataset(Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return img, animal_classes[label]
        def __len__(self):
            return len(self.subset)
    return RemappedDataset(dataset)

def split_dataset(dataset, num_clients):
    total = len(dataset)
    per_client = total // num_clients
    indices = list(range(total))
    random.shuffle(indices)

    client_datasets = {}
    for i in range(num_clients):
        start = i * per_client
        end = start + per_client
        client_indices = indices[start:end]
        client_datasets[i] = Subset(dataset, client_indices)

    return client_datasets

def assign_client_metadata(num_clients):
    client_info = {}
    for client_id in range(num_clients):
        client_info[client_id] = {
            "battery": round(random.uniform(0.2, 1), 2),
            "network": random.choice(["strong_free", "strong_paid", "weak"]),
            "compute": round(random.uniform(0.2, 1), 2),
        }
    return client_info

