import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock

from src.train import Trainer
from src.model import MnistClassifier


def create_fake_dataloader():
    """Create a small MNIST-like dataset for fast unit testing."""

    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )


def test_trainer_default_parameters():
    trainer = Trainer()

    assert trainer.learning_rate == 0.001
    assert trainer.momentum == 0.9
    assert trainer.num_epochs == 1


@patch("src.train.writer")
@patch("src.train.mlflow")
@patch("src.train.torch.save")
def test_train_model_runs(
    mock_torch_save,
    mock_mlflow,
    mock_writer
):
    trainer = Trainer()

    model = MnistClassifier()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=trainer.learning_rate,
        momentum=trainer.momentum
    )

    dataloader = create_fake_dataloader()

    # Mock MLflow context manager
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    trainer.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        training_dataloader=dataloader,
        num_epochs=1
    )

    mock_mlflow.start_run.assert_called_once()

    mock_mlflow.log_param.assert_any_call(
        "learning_rate",
        trainer.learning_rate
    )

    mock_mlflow.log_param.assert_any_call(
        "momentum",
        trainer.momentum
    )

    mock_torch_save.assert_called_once()

    mock_writer.flush.assert_called_once()
    mock_writer.close.assert_called_once()


@patch("src.train.writer")
@patch("src.train.mlflow")
@patch("src.train.torch.save")
def test_model_parameters_change_after_training(
    mock_torch_save,
    mock_mlflow,
    mock_writer
):
    trainer = Trainer()

    model = MnistClassifier()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=trainer.learning_rate,
        momentum=trainer.momentum
    )

    dataloader = create_fake_dataloader()

    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    trainer.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        training_dataloader=dataloader,
        num_epochs=1
    )

    parameters_changed = False

    for name, parameter in model.named_parameters():

        if not torch.equal(
            parameters_before[name],
            parameter.detach()
        ):
            parameters_changed = True
            break

    assert parameters_changed


@patch("src.train.writer")
@patch("src.train.mlflow")
@patch("src.train.torch.save")
def test_model_is_saved(
    mock_torch_save,
    mock_mlflow,
    mock_writer
):
    trainer = Trainer()

    model = MnistClassifier()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=trainer.learning_rate,
        momentum=trainer.momentum
    )

    dataloader = create_fake_dataloader()

    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    trainer.train_model(
        model,
        criterion,
        optimizer,
        dataloader,
        num_epochs=1
    )

    # mock_torch_save.assert_called_once_with(
    #     model.state_dict(),
    #     "mnist_model.pth"
    # )
        # Check torch.save was called exactly once
    mock_torch_save.assert_called_once()

    # Get arguments passed to torch.save()
    saved_state_dict, saved_path = mock_torch_save.call_args.args

    # Check path
    assert saved_path == "mnist_model.pth"

    # Check that a state_dict was passed
    assert isinstance(saved_state_dict, dict)

    # Check that the saved state_dict has the same keys as the model
    assert saved_state_dict.keys() == model.state_dict().keys()