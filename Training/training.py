import numpy as np 
import torch

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss()
from torch.utils.data import WeightedRandomSampler, DataLoader
from Augment.augment_config import AugmentConfig
from Training.Dataset import WakeWordDataset
from Augment.augment_pipeline import MyPipeline

from model import Jarvis


config = AugmentConfig()
rng = np.random.default_rng(seed=config.seed)

# Applies the data transformations
transformer = MyPipeline(cfg=config, training=True, rng=rng)

jarvis_dataset = WakeWordDataset(
                          data_folder="Training/Data", 
                          target_samplerate=config.sr,
                          p_silence=config.p_silence,
                          rng=rng,
                          transformer=transformer
                          )


labels = torch.tensor(jarvis_dataset.return_labels(), dtype=torch.int8)
class_counts = torch.bincount(input=labels)
class_weights = 1/class_counts.float()
sample_weights = class_weights[labels].numpy()

sampler = WeightedRandomSampler(
                                weights=sample_weights, #type: ignore
                                num_samples=2000, 
                                replacement=True,
                                generator=rng
                                )

loader = DataLoader(
                    jarvis_dataset, 
                    batch_size=config.batch_size, 
                    sampler=sampler, 
                    num_workers=4, 
                    generator=rng, 
                    persistent_workers=True
                    )


loss = CrossEntropyLoss()


