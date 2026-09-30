import torch
from torch.utils.data import TensorDataset
from unittest.mock import patch

from src.train import Trainer


def create_fake_mnist_dataset():
    images = torch.randn(20, 1, 28, 28)

    labels = torch.randint(
        low=0,
        high=10,
        size=(20,)
    )

    return TensorDataset(images, labels)


@patch("src.train.torchvision.datasets.MNIST")
def test_get_data_returns_dataloaders(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    training_loader, validation_loader = trainer.get_data()

    assert training_loader is not None
    assert validation_loader is not None


@patch("src.train.torchvision.datasets.MNIST")
def test_training_batch_shape(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    training_loader, _ = trainer.get_data()

    images, labels = next(iter(training_loader))

    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)


@patch("src.train.torchvision.datasets.MNIST")
def test_validation_batch_shape(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    _, validation_loader = trainer.get_data()

    images, labels = next(iter(validation_loader))

    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)


@patch("src.train.torchvision.datasets.MNIST")
def test_labels_are_valid_mnist_classes(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    training_loader, _ = trainer.get_data()

    _, labels = next(iter(training_loader))

    assert torch.all(labels >= 0)
    assert torch.all(labels <= 9)


@patch("src.train.torchvision.datasets.MNIST")
def test_images_have_correct_dimensions(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    training_loader, _ = trainer.get_data()

    images, _ = next(iter(training_loader))

    # batch, channels, height, width
    assert images.ndim == 4

    assert images.shape[1] == 1
    assert images.shape[2] == 28
    assert images.shape[3] == 28


@patch("src.train.torchvision.datasets.MNIST")
def test_data_contains_no_nan(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    training_loader, _ = trainer.get_data()

    images, _ = next(iter(training_loader))

    assert not torch.isnan(images).any()


@patch("src.train.torchvision.datasets.MNIST")
def test_batch_size_is_four(mock_mnist):
    fake_dataset = create_fake_mnist_dataset()

    mock_mnist.return_value = fake_dataset

    trainer = Trainer()

    training_loader, validation_loader = trainer.get_data()

    assert training_loader.batch_size == 4
    assert validation_loader.batch_size == 4