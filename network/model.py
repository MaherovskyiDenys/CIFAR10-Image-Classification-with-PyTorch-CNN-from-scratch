import torch
from torch import nn

from torch.optim import Optimizer  # for a hint
from torch.utils.data import DataLoader  # for a hint

from network.utils import loss_acc_plot, define_path, visualize

class CIFARClassifier(nn.Module):
    def __init__(self, lr: float = 0.001, epochs: int = 60):
        super().__init__()

        # Convolutional layers - 4 blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Hyperparameters
        self.lr = lr
        self.epochs = epochs

        self.path = define_path()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that takes a singe tensor with dims: [n, 3, 32, 32] runs it through the model

        Batch Normalization is applied to convolutional layers before pooling and activation function
        Dropout is used on the fully connected layer during training
        :note (p=0.1 is low because batch normalization is used)

        Returns raw logits from the output layer (no activation function is used)

        Structure:
            [conv -> bn -> relu] ->
            [conv -> bn -> relu -> max_pool] ->
            [conv -> bn -> relu -> max_pool] ->
            [conv -> bn -> relu -> max_pool -> flatten] -> FC

        :param x: Tensor, dims: [n, 3, 32, 32]
        :return: Logits
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # Flatten

        logits = self.fc(x)
        return logits

    def evaluate(self, path: str, show: bool = False):
        """
        Evaluate custom image by given path, displays it when show = True, by default False

        Returns predicted label for this image

        :param path: Path to img file
        :param show: Displays image, by default False
        :return: Predicted class
        """
        img = self._read_img(path, show)

        img = img.unsqueeze(0)  # Adding batch dimension, result: [n, 3, 32, 32]

        self.eval()
        with torch.no_grad():
            logits = self(img)

        classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        return classes[torch.argmax(logits).item()]

    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, loss_func: nn.Module, optimizer: Optimizer) -> None:
        """
        Trains the model using `self._run_epoch` n epochs on given datasets [train_loader, test_loader]

        Every 2 epochs will print epoch, loss and accuracy (%) for both datasets

        Every epoch collect the output per epoch for each dataset,
        Shows graph for both datasets using matplotlib after training is done

        :param train_loader: Train Dataset dtype: DataLoader
        :param test_loader: Test Dataset dtype: DataLoader
        :param loss_func: Loss function
        :param optimizer: Optimizer
        :return: None, shows loss_vs_accuracy graph
        """

        # Plotting params
        train_metrics = []
        test_metrics = []

        for epoch in range(self.epochs):
            # Train dataset
            output_train = self._run_epoch(train_loader, loss_func, optimizer=optimizer)
            train_metrics.append(output_train)

            # Test dataset
            output_test = self._run_epoch(test_loader, loss_func)
            test_metrics.append(output_test)


            # Printing stats every 2 epochs
            if epoch % 2 == 0:
                print(f"Epoch {epoch} | "
                      f"Train Loss: {output_train['loss']:.6f} | "
                      f"Test Loss: {output_test['loss']:.6f} | "
                      f"Train Acc: {output_train['accuracy'] * 100:.2f}% | "
                      f"Test Acc: {output_test['accuracy'] * 100:.2f}%")

        # Plotting
        loss_acc_plot(train_metrics, test_metrics)

    def _read_img(self, path: str, show: bool) -> torch.Tensor:
        """
        Internal function that reads an image by given path, resizes, and normalizes it

        Visualize it (after resizing only) using `matplotlib` when 'show = True'
        Returns image tensor, moves it to `self.device`

        :param path: path of the JPEG, PNG or GIF img
        :return: Image tensor
        """
        from torchvision.io import read_image
        from torchvision.transforms import v2

        img = read_image(path)

        pre_transform = v2.Compose([
            v2.Resize((32, 32)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        img = pre_transform(img)

        if show:
            visualize(img)

        normalize = v2.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768]
        )

        img = normalize(img)

        return img.to(self.device)

    def _run_epoch(self, dataloader: DataLoader, loss_func: nn.Module, optimizer: Optimizer | None = None) -> dict:
        """
        Internal function to run a single epoch through the model automatically sets model to train mode if optimizer is used else eval mode

        Runs through a given dataset using a DataLoader and computes loss using given loss func
        Backprop the model using given optimizer if specified

        :param dataloader: Test Dataset dtype: DataLoader
        :param loss_func: Loss function
        :param optimizer: Optimizer
        :return: dict of calculated loss and accuracy also n of correct/total
        """
        is_train = optimizer is not None

        if is_train:
            self.train()
        else:
            self.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for data, label in dataloader:
                data = data.to(self.device)
                label = label.to(self.device).long()

                logits = self(data)
                loss = loss_func(logits, label)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                predicted = logits.argmax(dim=1)

                correct += (predicted == label).sum().item()
                total += label.size(0)
                total_loss += loss.item()

            output = {
                "loss": total_loss / len(dataloader),
                "accuracy": correct / total,
                "correct": correct,
                "total": total
            }

            return output

    def load_state(self) -> bool:
        """
        Loads model parameters from `self.path` if available
        """
        try:
            params = torch.load(self.path, map_location=self.device)
            self.load_state_dict(params)

            return True

        except FileNotFoundError as err:
            print("Model has not been trained or saved yet")
            raise FileNotFoundError

    def save_state(self) -> None:
        """
        Saves model parameters to `self.path` when called
        """
        params = self.state_dict()
        torch.save(params, self.path)