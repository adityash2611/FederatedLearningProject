import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import random
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from model.cnn import AnimalCNN
from federated.utils import (
    load_and_duplicate_cifar, filter_animals, remap_labels, split_dataset, assign_client_metadata
)
from federated.client import Client
from federated.server import Server
from federated.data_manager import DataManager

NUM_CLIENTS = 1000
INITIAL_DATA_PER_CLIENT = 50
NEW_DATA_PER_CLIENT = 25
CLIENTS_PER_ROUND = 200
NUM_ROUNDS = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
animal_classes = {
    2: 0,  # bird
    3: 1,  # cat
    4: 2,  # deer
    5: 3,  # dog
    6: 4,  # frog
    7: 5   # horse
}

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding = 4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading dataset")
full_dataset = load_and_duplicate_cifar(num_copies=3, transform=transform)
filtered = filter_animals(full_dataset, animal_classes)
remapped = remap_labels(filtered, animal_classes)

total_size = len(remapped)
server_size = 5000
test_size = 10000
train_size = total_size - server_size - test_size

train_data, test_data, server_data = random_split(remapped, [train_size, test_size, server_size])
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Creating Clients")
client_datasets = split_dataset(train_data, num_clients = NUM_CLIENTS)
client_metadata = assign_client_metadata(NUM_CLIENTS)

clients = []
for cid in range(NUM_CLIENTS):
    initial_subset = Subset(client_datasets[cid], list(range(INITIAL_DATA_PER_CLIENT)))
    client = Client(
        client_id=cid,
        initial_data=initial_subset,
        metadata=client_metadata[cid],
        batch_size=32,
        device=DEVICE,
    )
    clients.append(client)

print("Setting up Data Manager")
data_manager = DataManager(
    train_data,
    new_data_per_client=NEW_DATA_PER_CLIENT,
)

print("Starting Federated Learning")
server = Server(
    model_fn=AnimalCNN,
    testloader=testloader,
    server_dataset=server_data,
    all_clients=clients,
    data_manager=data_manager,
    device=DEVICE
)

server.train_federated(
    num_rounds=NUM_ROUNDS,
    clients_per_round=CLIENTS_PER_ROUND,
    local_epochs=5,
    lr=0.001,
)