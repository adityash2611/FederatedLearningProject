import torch
import random
import copy
from torch.utils.data import DataLoader

class Server:
    def __init__(self, model_fn, testloader, server_dataset, all_clients, data_manager, device="cpu"):
        self.global_model = model_fn().to(device)
        self.device = device
        self.testloader = testloader
        self.clients = all_clients
        self.data_manager = data_manager

        print("Pre-training skeleton model at server")
        self._train_on_server_data(server_dataset)

    def _train_on_server_data(self, dataset, epochs = 10, lr=0.001):
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        model = self.global_model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch in loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def select_clients(self, num_clients, battery_thresh=0.3, compute_thresh=0.3):
        eligible = []
        for client in self.clients:
            batt = client.metadata["battery"]
            comp = client.metadata["compute"]
            netw = client.metadata["network"]
            if batt > battery_thresh and comp > compute_thresh and netw != "weak":
                eligible.append(client)

        selected = random.sample(eligible, min(num_clients, len(eligible)))
        return selected
    
    def aggregate_weights(Self, client_weights):
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = avg_weights[key]/len(client_weights)
        return avg_weights
    
    def evaluate(self):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        acc = 100*correct/total
        print(f"Global model accuracy: {acc:.2f}%")
        return acc
    
    def train_federated(self, num_rounds, clients_per_round, local_epochs=3, lr=0.001):
        self.accuracy_log = []
        for round_num in range(1, num_rounds+1):
            print(f"\nRound {round_num}")

            selected_clients = self.select_clients(clients_per_round)
            new_data_dict = self.data_manager.assign_new_data(selected_clients)
            client_weights = []

            for client in selected_clients:
                if client.id in new_data_dict:
                    client.receive_new_data(new_data_dict[client.id])
                updated = client.train(self.global_model, epochs=local_epochs, lr=lr)
                client_weights.append(updated)

            aggregated = self.aggregate_weights(client_weights)
            self.global_model.load_state_dict(aggregated)
            acc = self.evaluate()
            self.accuracy_log.append(acc)
            print(f"Remaining data: {self.data_manager.remaining_data()}")
        torch.save(self.global_model.state_dict(), "federated_animal_cnn.pth")
        print("Global model saved to federated_animal_cnn.pth")
        model_size_bytes = self._get_model_size(self.global_model)
        total_communication_bytes = model_size_bytes * 2 * clients_per_round * num_rounds
        total_communication_MB = total_communication_bytes / (1024 * 1024)
        print(f"Total Communication Cost: {total_communication_MB:.2f} MB")

    def _get_model_size(self, model):
        size = 0
        for param in model.parameters():
            size += param.nelement() * param.element_size()
        return size