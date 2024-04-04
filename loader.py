from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from utils import preprocess

class HistologyDataModule(LightningDataModule):
    def __init__(self, batch_size=512, transform = transforms.Compose([transforms.ToTensor()])):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
    def setup(self, stage):
        # prepare transforms standard
        self.train_set = ImageFolder(root="tcga_coad_msi_mss/train", transform=preprocess())
        self.val_set = ImageFolder(root="tcga_coad_msi_mss/val", transform=preprocess())

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=3, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=1, pin_memory=True)


