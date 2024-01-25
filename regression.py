import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(inputs, targets, model, loss_fn, optimizer):
    # Compute prediction and loss
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while working you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previous_loss variable to stop the training when the loss is not changing much.
    """
    learning_rate = 0.01
    num_epochs = 100

    # Determine input size based on the first dimension of X
    input_features = X.shape[1] if len(X.shape) > 1 else 1
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Print loss and predictions for debugging
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
            # Assume test_inputs is defined
            test_inputs = torch.tensor([[20.], [15.], [10.]])
            print('Predictions:', model(test_inputs).detach().numpy())
        
        # Change this condition to stop the training when the loss is not changing much.
        if abs(previous_loss - loss.item()) < 1e-5:
            break

        previous_loss = loss.item()

    return model, loss
