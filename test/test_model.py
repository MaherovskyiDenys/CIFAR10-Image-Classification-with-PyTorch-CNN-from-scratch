import torch
from network.model import CIFARClassifier
from torch.utils.data import DataLoader
from data.loader import cifar10_dataset_test


def test() -> None:
    model = CIFARClassifier()  # Init model

    model.load_state()
    print("Model is loaded.")

    dataset = DataLoader(cifar10_dataset_test, batch_size=128, shuffle=False)
    print("CIFAR-10 dataset test is loaded.")

    loss_func = torch.nn.CrossEntropyLoss()

    result = model._run_epoch(dataset, loss_func)

    print(f"Mean loss: {result['loss']:.4f} | Accuracy: {result['accuracy'] * 100:.2f}% | ({result['correct']} / {result['total']})")

if __name__ == "__main__":
    test()