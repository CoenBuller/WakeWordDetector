import torch

from torch import Tensor
from tqdm import tqdm


class Stats():
    def __init__(self, threshold=0.5):
        self.outputs = []
        self.targets = []
        self.thresh = threshold

    def __call__(self):
        outputs = torch.tensor(self.outputs, dtype=torch.int8, requires_grad=False).detach()
        rounded_out = torch.round(outputs - (self.thresh - 0.5))
        targets = torch.tensor(self.targets, dtype=torch.int8, requires_grad=False).detach()

        pos = targets == 1
        neg = targets == 0

        n_pos = torch.sum(pos)
        n_neg = torch.sum(neg)

        true_pos = torch.sum(rounded_out == 1 & pos)
        true_neg = torch.sum(rounded_out == 0 & neg)

        false_neg = torch.sum(rounded_out == 0 & pos)
        false_pos = torch.sum(rounded_out == 1 & neg)

        try:
            precision = true_pos / n_pos
            accuracy = (true_pos + true_neg) / (n_pos + n_neg)
            recall = true_pos / torch.sum(true_pos | false_neg)
            f1_score = 2 * precision * recall / (precision + recall)

            true_acc = true_pos / n_pos
            true_neg = true_neg / n_neg

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
                    "true_negative": true_acc.item(),
                    "roc": roc,
                    "thresholds": thresholds
                    }
            
        except ZeroDivisionError:
            stats = {}
            print("There were no positive cases in the validation dataset")

        return stats

    def _roc_auc(self, outputs: Tensor, pos: Tensor, neg: Tensor, n_thresholds: int = 200):
        thresholds = torch.linspace(0, 1, steps=n_thresholds)

        n_pos = torch.sum(pos)
        n_neg = torch.sum(neg)

        roc = []

        for thresh in thresholds:
            rounded_outputs = torch.round(outputs - (thresh - 0.5))

            true_pos = torch.sum(rounded_outputs == 1 & pos)
            false_pos = torch.sum(rounded_outputs == 1 & neg)

            tpr = true_pos / n_pos
            fpr = false_pos / n_neg

            roc.append((tpr.item(), fpr.item()))

        return roc, thresholds


            
    def add(self, targets: Tensor, outputs: Tensor):
        for target_val in targets:
            self.targets.append(target_val)

        for output_val in outputs:
            self.outputs.append(output_val)
        
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