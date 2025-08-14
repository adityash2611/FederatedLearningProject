NAME: Aditya Shrivastava
ID: 2022A8PS1732P

FOLDER STRUCTURE:
federated_learning_project/ 
├── dataset/  # The code downloads the dataset in this folder
├── centralized/ 
│ ├── train.py                  # Centralized model training 
│ └── init.py
├── federated/ 
│ ├── client.py                 # Client logic 
│ ├── server.py                 # Server logic 
│ ├── data_manager.py           # Assigns new data to clients each round 
│ ├── utils.py                  # Dataset loading and preprocessing 
│ └── init.py 
├── model/ 
│ └── cnn.py                    # Model used by both centralized & FL 
├── federated_run.py            # Main script to run federated learning 
├── federated_animal_cnn.pth    # Final federated model
├── centralized_animal_cnn.pth  # Final centralized model

DATASET USED:
We use the CIFAR-10 dataset and filter only 6 animal classes:
=> Bird, Cat, Deer, Dog, Frog, Horse

REQUIREMENTS:
=> Python 3.8+
=> PyTorch
=> torchvision
And everything else is added in the requirements.txt file, so to download them:

pip install -r requirements.txt

RUNNIG THE CODE:
1) RUN CENTRALIZED TRAINING

python centralized/train.py

Saves model to centralized_animal_cnn.pth

2)RUN FEDERATED TRAINING

python federated_run.py

Saves model to federated_animal_cnn.pth


