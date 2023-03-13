# Simulation Dataset used for testing segmentation models


from torch.utils.data import Dataset
from extras import simulation
import numpy as np

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(256, 256, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]

        ela = np.zeros((256, 256, 3))
        if self.transform:
            image = self.transform(image)
            ela = self.transform(ela).float()


        return {
            "image": image, 
            "mask": mask,
            "ela": ela
        }