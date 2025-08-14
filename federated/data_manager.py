import random
from torch.utils.data import Subset

class DataManager:
    def __init__(self, dataset, new_data_per_client = 20):
        self.dataset = dataset
        self.unassigned_indices = list(range(len(dataset)))
        random.shuffle(self.unassigned_indices)
        self.new_data_per_client = new_data_per_client
        self.current_pointer = 0

    def assign_new_data(self, selected_clients):
        new_data_dict = {}
        for client in selected_clients:
            start = self.current_pointer
            end = start + self.new_data_per_client
            if end>len(self.unassigned_indices):
                print("Not enough data to assign to client")
                break

            indices = self.unassigned_indices[start:end]
            new_data_dict[client.id] = Subset(self.dataset, indices)

            self.current_pointer += self.new_data_per_client

        return new_data_dict
    
    def remaining_data(self):
        return len(self.unassigned_indices) - self.current_pointer