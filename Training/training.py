import numpy as np 
import torch

from sklearn.model_selection import StratifiedShuffleSplit

from torch.optim import AdamW
from torch.nn import BCELoss
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Augment.augment_config import AugmentConfig
from Training.Dataset import LoadDataset, TrainDataset, ValidateDataset
from Augment.augment_pipeline import TrainPipeline

from model import Jarvis
from Training.one_epoch import one_epoch


def main():
    config = AugmentConfig()
    rng = np.random.default_rng(seed=config.seed)
    torch_generator = torch.Generator().manual_seed(config.seed)
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Device: {device}")

    # Applies the data transformations
    transformer = TrainPipeline(cfg=config, training=True, rng=rng)

    # Load in entire dataset
    jarvis_dataset = LoadDataset(
                            data_folder="Training/Data", 
                            rng=rng,
                            )   

    # Get labels from your dataset
    all_labels = jarvis_dataset.return_labels().to_numpy()
    print(f"Number of samples: {len(all_labels)} | Number of positive samples: {len(all_labels[all_labels == 1])}")


    # Use StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=config.seed)
    train_indices, val_indices = next(sss.split(np.arange(len(jarvis_dataset)), all_labels))

    # Convert indices to lists
    train_indices = np.array(train_indices.tolist())
    val_indices = np.array(val_indices.tolist())
    print(f"Validation length: {len(val_indices)}" )
    print(f"Validation length: {len(train_indices)}" )


    # Create train/test split
    train_df = jarvis_dataset.__getitems__(train_indices)
    val_df = jarvis_dataset.__getitems__(val_indices)
    print(f"Training: {val_df.value_counts("label")}")
    print(f"Validation: {train_df.value_counts("label")}")

    train_dataset = TrainDataset(
                            items=train_df, 
                            target_samplerate=config.sr,
                            p_silence=config.p_silence,
                            rng=rng,
                            transformer=transformer
                            )

    validation_dataset = ValidateDataset(items=val_df)

    # Calculate sample weights so that the proportions between positive and negative classes is equal
    labels = torch.tensor(train_dataset.return_labels())
    class_counts = torch.bincount(input=labels)
    class_weights = 1/class_counts.float()
    sample_weights = class_weights[labels].numpy()

    sampler = WeightedRandomSampler(
                                    weights=sample_weights, #type: ignore
                                    num_samples=int(len(train_dataset) * (1+config.p_silence)), 
                                    replacement=True,
                                    generator=torch_generator
                                    )

    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=config.batch_size, 
                        sampler=sampler, 
                        num_workers=6, 
                        persistent_workers=True
                        )

    validation_loader = DataLoader(
                            validation_dataset,
                            batch_size=1
                        )

    model = Jarvis(dim_in=(128, 64), dim_out=1).to(device=device)
    loss = BCELoss()
    optimizer = AdamW(params=model.parameters(), lr=config.learning_rate)
    lr_scheduler = ReduceLROnPlateau(
                                optimizer=optimizer,
                                factor=0.5,
                                patience=5,
                                min_lr=config.min_lr
                                )
    

    epoch_number = 0
    min_val_loss = float("inf")
    for epoch in range(config.epoch):
        print(f'EPOCH {epoch_number + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss, val_loss, val_accuracy = one_epoch(
                                                epoch_index=epoch_number, 
                                                model=model,
                                                training_loader=train_loader,
                                                validation_loader=validation_loader,
                                                optimizer=optimizer,
                                                loss_fn=loss,
                                                device=device
                                                )



        print(f'LOSS train: {train_loss:.2f} | LOSS validation: {val_loss:.2f} | Accuracy: {val_accuracy:.2f}')

        lr_scheduler.step(val_loss)
        epoch_number += 1
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(model, "Jarvis.pt")

if __name__ == "__main__":
    main()




