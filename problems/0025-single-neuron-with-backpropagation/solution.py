import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    X = features.to(dtype=torch.float32)
    y = labels.to(device=X.device, dtype=X.dtype)
    
    # Trainable parameters
    w = nn.Parameter(
        initial_weights.detach().clone().to(
            device=X.device, dtype=X.dtype
        )
    )
    b = nn.Parameter(
        torch.tensor(initial_bias, device=X.device, dtype=X.dtype)
    )

    optimizer = torch.optim.SGD(
        [w, b], lr=learning_rate
    )

    mse_values = []

    for epoch in range(epochs):
        # 1. Reset gradients
        optimizer.zero_grad()

        # 2. Forward propagation
        z = torch.matmul(X, w) + b
        predictions = torch.sigmoid(z)
        
        # 3. Calculate loss
        loss = nn.functional.mse_loss(predictions, y)
        mse_values.append(round(loss.item(), 4))

        # 4. Backpropgation
        loss.backward()

        # 5. Update parameters
        optimizer.step()

    return (
        [round(v, 4) for v in w.detach().tolist()],
        round(b.item(), 4),
        mse_values
    )
    