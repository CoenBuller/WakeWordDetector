from tqdm import tqdm
import torch




def one_epoch(
        epoch_index, 
        training_loader, 
        validation_loader, 
        optimizer, 
        model, 
        loss_fn, 
        device):
    
    total_loss = 0.
    n_samples = 0
    last_loss = 0.
    model = model.to(device=device)

    p_bar = tqdm(enumerate(training_loader), total=len(training_loader))

    for i, data in p_bar:

        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device, dtype=torch.float32)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        total_loss += loss
        n_samples += len(data)
        avg_loss = total_loss/n_samples

        p_bar.set_description(f"Avg trainig loss: {avg_loss:.2f}")

    val_loss = 0
    correct = 0
    total = 0 
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):

            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device, dtype=torch.float32)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)
            if outputs.round() == labels:
                correct += 1

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            val_loss += loss
            total += 1

    avg_val_loss = val_loss/total
    accuracy = correct/total
    avg_train_loss = total_loss/n_samples
    return avg_train_loss, avg_val_loss, accuracy