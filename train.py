import torch
import torch.nn as nn


def train(model, train_loader, epochs=25, device=torch.device('cpu')):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    model.train()

    for epoch in range(epochs):

        print(f'training epoch: {epoch}/{epochs}')
        current_loss = 0
        for X, y_true in train_loader:
            X, y_true = X.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(X)

            loss = criterion(y_pred, y_true)
            current_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'current loss: {current_loss/len(train_loader)}')

    return model


def test(model, test_loader, device=torch.device('cpu')):

    model.to(device)
    total = 0
    correct = 0
    model.eval()

    with torch.no_grad():

        for X, y_true in test_loader:
            X, y_true = X.to(device), y_true.to(device)
            y_pred = model(X)
            _, y_pred = torch.max(y_pred, 1)
            total += len(y_pred)
            correct += (y_pred == y_true).sum().item()

    acc = correct/total
    return acc


def train_continual(model, train_loaders, test_loaders, output_neuron_counts, device, epochs=20, num_tasks=4):

    if len(test_loaders) != num_tasks:
        print('task count and data loader counts do not match!')
        return

    learning_accs = []
    running_test_accs = {i:[] for i in range(num_tasks)}

    for i in range(num_tasks):

        print(f'training on task: {i}')
        current_task_id = i
        current_task_train_loader = train_loaders[current_task_id]
        current_task_output_neurons = output_neuron_counts[i]

        # change model output neuron count
        # model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=current_task_output_neurons, bias=True)

        # train model on current task
        train(model, current_task_train_loader, epochs=epochs, device=device)

        print(f'finished training on task: {i}')

        # test model on previous and current tasks
        for test_task_id in range(num_tasks):
            
            current_task_test_loader = test_loaders[test_task_id]
            if test_task_id > current_task_id:
                test_acc = 0
            else:
                test_acc = test(model, current_task_test_loader, device=device)

            if test_task_id == current_task_id:
                learning_accs.append(test_acc)
            print(f'accuracy on task {test_task_id}: {test_acc}')

            running_test_accs[test_task_id].append(test_acc)

        print(running_test_accs)

    return learning_accs, running_test_accs
