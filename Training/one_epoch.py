import torch

from torch import Tensor
from tqdm import tqdm


class Stats():
    def __init__(self, threshold=0.5):
        self.outputs = []
        self.targets = []
        self.thresh = threshold

    def __call__(self):
        outputs = torch.tensor(self.outputs, dtype=torch.float32, requires_grad=False).detach()
        targets = torch.tensor(self.targets, dtype=torch.float32, requires_grad=False).detach()
        
        rounded_out = torch.round(outputs - (self.thresh - 0.5))
        
        pos = targets == 1
        neg = targets == 0
        
        n_pos = torch.sum(pos).float()
        n_neg = torch.sum(neg).float()
        n_total = n_pos + n_neg
        
        # FIXED: Proper operator precedence with parentheses
        true_pos = torch.sum((rounded_out == 1) & pos).float()
        true_neg = torch.sum((rounded_out == 0) & neg).float()
        false_neg = torch.sum((rounded_out == 0) & pos).float()
        false_pos = torch.sum((rounded_out == 1) & neg).float()
        
        # FIXED: Handle division by zero
        if n_pos == 0 or n_neg == 0:
            print("Warning: Dataset has only one class. Metrics may be undefined.")
            return {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "true_accuracy": 0.0,
                "false_accuracy": 0.0,
                "roc": [],
                "thresholds": []
            }
        
        # FIXED: Correct recall calculation
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0)
        recall = true_pos / n_pos if n_pos > 0 else torch.tensor(0.0)
        accuracy = (true_pos + true_neg) / n_total if n_total > 0 else torch.tensor(0.0)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        
        true_acc = true_pos / n_pos
        false_acc = true_neg / n_neg
        
        roc, thresholds = self._roc_auc(
            outputs=outputs,
            pos=pos,
            neg=neg,
            n_thresholds=200
        )
        
        stats = {
            "f1": f1_score.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "accuracy": accuracy.item(),
            "true_accuracy": true_acc.item(),
            "false_accuracy": false_acc.item(),
            "roc": roc,
            "thresholds": thresholds
        }
        
        return stats

    def _roc_auc(self, outputs: Tensor, pos: Tensor, neg: Tensor, n_thresholds: int = 200):
        thresholds = torch.linspace(0, 1, steps=n_thresholds)
        
        n_pos = torch.sum(pos).float()
        n_neg = torch.sum(neg).float()
        
        roc = []
        
        # Handle case where one class is missing
        if n_pos == 0 or n_neg == 0:
            return [], thresholds
        
        for thresh in thresholds:
            rounded_outputs = torch.round(outputs - (thresh - 0.5))
            
            true_pos = torch.sum((rounded_outputs == 1) & pos).float()
            false_pos = torch.sum((rounded_outputs == 1) & neg).float()
            
            tpr = true_pos / n_pos
            fpr = false_pos / n_neg
            
            roc.append((tpr.item(), fpr.item()))
        
        return roc, thresholds
    
    def add(self, targets: Tensor, outputs: Tensor):
        self.targets.extend(targets.detach().cpu().tolist())
        self.outputs.extend(outputs.detach().cpu().tolist())
        
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Adjust learning weights
        optimizer.step()

        total_loss += loss
        n_samples += len(data)
        avg_loss = total_loss/n_samples

        p_bar.set_description(f"Avg trainig loss: {avg_loss:.2f}")

    val_loss = 0
    total = 0 
    model.eval()
    stats = Stats()
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
            stats.add(targets=labels, outputs=outputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            val_loss += loss
            total += 1


    avg_val_loss = val_loss/total
    avg_train_loss = total_loss/n_samples
    s = stats()

    return avg_train_loss, avg_val_loss, s