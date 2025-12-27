from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from model import CIFARClassifier
from data.loader import cifar10_dataset, cifar10_dataset_test

def train():
    dataset_train = DataLoader(cifar10_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    dataset_test = DataLoader(cifar10_dataset_test, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    lr = 0.001
    epochs = 60

    model = CIFARClassifier(lr=lr, epochs=epochs)

    loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train_model(dataset_train, dataset_test, loss, optimizer)

    model.save_state()


if __name__ == "__main__":
    train()