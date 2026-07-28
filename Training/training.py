import numpy as np 
import torch

from sklearn.model_selection import StratifiedShuffleSplit

from torch import nn, Tensor
from torch.optim import AdamW
from torch.nn import BCELoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR


from Augment.augment_config import AugmentConfig
from Training.Dataset import LoadDataset, TrainDataset, ValidateDataset
from Augment.augment_pipeline import AugPipeline

from model import Jarvis
from Training.one_epoch import one_epoch

class WeightedFocalLoss(nn.Module):
    def __init__(self, device, alpha: float | Tensor, gamma=2):
        super(WeightedFocalLoss, self).__init__()

        if type(alpha) == float or (type(alpha) == Tensor and len(alpha) == 1):
            self.alpha = torch.tensor([alpha, 1-alpha]).to(device=device)
        elif type(alpha) == Tensor: 
            self.alpha = alpha.to(device=device)
        else:
            raise ValueError()

        self.gamma = gamma
        self.bce_loss = BCELoss(reduction="none") # type: ignore

    def forward(self, inputs, targets):
        BCE_loss = self.bce_loss(inputs, targets)

        if type(targets) == torch.Tensor:
            targets = targets.type(torch.float32)
        elif type(targets) == np.ndarray:
            targets = torch.tensor(targets, dtype=torch.float32)

        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        fw = (1 - p_t)**self.gamma 

        targets_long = targets.long()  # Convert to long for indexing
        alpha_t = self.alpha[targets_long] # type: ignore

        F_loss = alpha_t * fw * BCE_loss
        return F_loss.mean()

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]

def main():
    config = AugmentConfig()
    
    # Set all seeds consistently
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cpu"
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(config.seed)
        device = "xpu"

    print(f"Device: {device}")

    # Use consistent RNG
    rng = np.random.default_rng(seed=config.seed)
    torch_generator = torch.Generator().manual_seed(config.seed)

    # Applies the data transformations
    train_transformer = AugPipeline(
                            cfg=config, 
                            training=True, 
                            spec_augment=True, 
                            rng=rng
                            )
    
    val_transformer = AugPipeline(
                            cfg=config, 
                            training=False, 
                            spec_augment=False, 
                            rng=rng
                            )
    val_augmentor = AugPipeline(
                        cfg=config, 
                        training=True,
                        spec_augment=False, 
                        rng=rng
                        )



    # Load in entire dataset
    jarvis_dataset = LoadDataset(
                            data_folder="Training/Data", 
                            rng=rng,
                            )   

    # Get labels from your dataset
    all_labels = jarvis_dataset.return_labels().to_numpy()
    print(f"Number of samples: {len(all_labels)} | Number of positive samples: {len(all_labels[all_labels == 1])}")

    # Use StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=config.seed)
    val_indices, train_indices = next(sss.split(np.arange(len(jarvis_dataset)), all_labels))

    # Convert indices to lists
    train_indices = np.array(train_indices.tolist())
    val_indices = np.array(val_indices.tolist())
    print(f"Training length: {len(train_indices)}" )
    print(f"Validation length: {len(val_indices)}" )


    # Create train/test split
    train_df = jarvis_dataset.__getitems__(train_indices)
    val_df = jarvis_dataset.__getitems__(val_indices)
    print(f"Training: {train_df.value_counts("label")}")
    print(f"Validation: {val_df.value_counts("label")}")

    train_dataset = TrainDataset(
                            items=train_df, 
                            target_samplerate=config.sr,
                            rng=rng,
                            transformer=train_transformer
                            )

    validation_dataset = ValidateDataset(
                                items=val_df,
                                transformer=val_transformer,
                                augmentor=val_augmentor,
                                n_augments=50
                                )

    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=config.batch_size, 
                        num_workers=6, 
                        persistent_workers=True,
                        shuffle=True
                        )

    validation_loader = DataLoader(
                            validation_dataset,
                            batch_size=1,
                        )

    #  Models and optimizer
    model = Jarvis(dim_in=(128, 64), dim_out=1).to(device=device)

    # Class weights is based on class counts 
    labels = torch.tensor(train_dataset.return_labels())
    class_counts = torch.bincount(input=labels)
    class_weights = 1 / class_counts
    class_weights /= class_weights.sum()
    print(f"Class weights for Weighted Focal Loss: {class_weights}")
    loss = BCELoss()

    optimizer = AdamW(
                    params=model.parameters(), 
                    lr=config.learning_rate,
                    weight_decay=1e-4
                    )
    
    lr_scheduler = CosineAnnealingLR(
                                optimizer=optimizer,
                                T_max=config.epoch,
                                eta_min=config.min_lr
                                )
    

    epoch_number = 0
    best_f1 = 0
    for epoch in range(config.epoch):
        print(f'EPOCH {epoch_number + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss, val_loss, stats = one_epoch(
                                                epoch_index=epoch_number, 
                                                model=model,
                                                training_loader=train_loader,
                                                validation_loader=validation_loader,
                                                optimizer=optimizer,
                                                loss_fn=loss,
                                                device=device
                                                )


        print(
            f"""
STATS:
    - Loss train: {format_e(train_loss)} 
    - LOSS validation: {format_e(val_loss)} 
    - Accuracy: {stats["accuracy"]:.3f}
    - True Accuracy: {stats["true_accuracy"]:.3f}
    - False Accuracy: {stats["false_accuracy"]:.3f}
    - F1-score: {stats["f1"]:.3f}
    """
        )

        lr_scheduler.step()
        epoch_number += 1
        if stats["f1"] > best_f1:
            best_f1 = stats["f1"]
            torch.save(model, "Jarvis_2_0.pt")

if __name__ == "__main__":
    main()




