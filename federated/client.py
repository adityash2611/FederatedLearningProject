import torch
from torch.utils.data import DataLoader, ConcatDataset
import copy

class Client:
    def __init__(self, client_id, initial_data, metadata, batch_size=32, device="cpu"):
        self.id = client_id
        self.device = device
        self.local_dataset = initial_data
        self.metadata = metadata
        self.batch_size = batch_size

    def receive_new_data(self, new_data):
        self.local_dataset = ConcatDataset([self.local_dataset, new_data])
        
    def train(self, global_model, epochs=1, lr=0.001):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        dataloader = DataLoader(self.local_dataset, batch_size = self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.state_dict()
    
    def num_samples(self):
        return len(self.local_dataset)
    
    