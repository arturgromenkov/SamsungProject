from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning import LightningDataModule

class HistologyDataModule(LightningDataModule):
    def __init__(self, batch_size=512):
        super().__init__()
        self.batch_size = batch_size
    def setup(self, stage):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor()])
        # prepare transforms standard to MNIST
        self.train_set = ImageFolder(root="tcga_coad_msi_mss/train", transform=transform)
        self.val_set = ImageFolder(root="tcga_coad_msi_mss/val", transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=3, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=1, pin_memory=True)


