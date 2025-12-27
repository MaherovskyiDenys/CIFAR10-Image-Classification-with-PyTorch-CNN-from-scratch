import os
import torch
import matplotlib.pyplot as plt

def define_path() -> str:
    """
    Defines path in `network.weights` folder to '.pth' file

    :return: Path
    """

    base = os.path.dirname(__file__)

    filename = os.path.join(base, r"weights/cifar-10.pth")

    return filename

def visualize(img: torch.Tensor) -> None:
    """
    Displays given image dtype: Tensor. Plots it using `matplotlib`

    :param img: Image tensor
    :return: None, Plots image
    """

    img = img.cpu().permute(1, 2, 0)  # [C,H,W] -> [H,W,C]

    plt.imshow(img)
    plt.axis("off")
    plt.show()

def loss_acc_plot(train: list[dict], test: list[dict]) -> None:
    """
    Function that plots Loss/Accuracy graphs for training and test datasets

    Takes a list of dictionaries with structure:
    dict = {
                "loss": loss,
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }

    :param train: List of training outputs
    :param test: List of test outputs
    :return: None, Plots graphs
    """

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5))

    train_loss = [m["loss"] for m in train]
    train_acc = [m["accuracy"] * 100 for m in train]

    test_loss = [m["loss"] for m in test]
    test_acc = [m["accuracy"] * 100 for m in test]

    # Loss
    ax1.plot(train_loss, label="Train")
    ax1.plot(test_loss, label="Test")
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Accuracy
    ax2.plot(train_acc, label="Train")
    ax2.plot(test_acc, label="Test")
    ax2.set_title("Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()