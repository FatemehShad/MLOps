import torch
import torchvision
from src.model import MnistClassifier
import mlflow
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist_experiment")


class Trainer:

    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.num_epochs = 1

    def get_data(self):
        training_data = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        validation_data = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=4, shuffle=False)
        return training_dataloader, validation_dataloader



    def train_model(self, model, criterion, optimizer, training_dataloader, num_epochs=None):
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        # Add tensorboard logging here
        with mlflow.start_run():
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("momentum", self.momentum)
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("batch_size", 4)
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, data in enumerate(training_dataloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if i % 1000 == 999:
                        # log the running loss to tensorboard
                        writer.add_scalar("training loss", running_loss / 1000, epoch * len(training_dataloader) + i)
                        print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                        mlflow.log_metric("train loss", (running_loss / len(training_dataloader)), step=epoch)
                        running_loss = 0.0

                # mlflow.log_metric("train accuracy", self.evaluate_model(model, training_dataloader), step=epoch)
            print("Finished Training")
            torch.save(model.state_dict(), "mnist_model.pth")
            print("Model saved to mnist_model.pth")
            writer.flush()
            writer.close()

if __name__ == "__main__":
    trainer = Trainer()
    training_dataloader, validation_dataloader = trainer.get_data()
    model = MnistClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=trainer.learning_rate, momentum=trainer.momentum)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("my-second-experiment")
    mlflow.set_system_metrics_sampling_interval(1) 
    trainer.train_model(model, criterion, optimizer, training_dataloader, num_epochs=3)
    
        # model_info = mlflow.pytorch.log_model(model)