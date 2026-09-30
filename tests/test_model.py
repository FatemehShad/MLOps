import torch

from src.model import MnistClassifier


def test_model_creation():
    model = MnistClassifier()

    assert model is not None


def test_model_is_torch_module():
    model = MnistClassifier()

    assert isinstance(model, torch.nn.Module)


def test_model_forward_pass():
    model = MnistClassifier()

    # Batch of 4 MNIST images
    inputs = torch.randn(4, 1, 28, 28)

    outputs = model(inputs)

    assert outputs is not None


def test_model_output_shape():
    model = MnistClassifier()

    inputs = torch.randn(4, 1, 28, 28)

    outputs = model(inputs)

    # MNIST has 10 classes
    assert outputs.shape == (4, 10)


def test_single_image_output_shape():
    model = MnistClassifier()

    inputs = torch.randn(1, 1, 28, 28)

    outputs = model(inputs)

    assert outputs.shape == (1, 10)


def test_model_output_is_tensor():
    model = MnistClassifier()

    inputs = torch.randn(4, 1, 28, 28)

    outputs = model(inputs)

    assert isinstance(outputs, torch.Tensor)


def test_model_output_has_no_nan():
    model = MnistClassifier()

    inputs = torch.randn(4, 1, 28, 28)

    outputs = model(inputs)

    assert not torch.isnan(outputs).any()


def test_model_backward_pass():
    model = MnistClassifier()

    inputs = torch.randn(4, 1, 28, 28)

    labels = torch.tensor([0, 1, 2, 3])

    criterion = torch.nn.CrossEntropyLoss()

    outputs = model(inputs)

    loss = criterion(outputs, labels)

    loss.backward()

    # At least one parameter should receive a gradient
    has_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert has_gradient