def fit_regression_model(X, y):
    learning_rate = 0.01
    num_epochs = 100
    input_features = X.shape[1]
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
            print('Predictions:', model(X_test).detach().numpy())
        
        # Change this condition to stop the training when the loss is not changing much.
        if abs(previous_loss - loss.item()) < 1e-5:
            break

        previous_loss = loss.item()

    return model, loss


