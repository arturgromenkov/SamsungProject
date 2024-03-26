from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

class Loader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.train_set = ImageFolder(root="tcga_coad_msi_mss/train", transform=transforms.ToTensor())
        self.val_set = ImageFolder(root="tcga_coad_msi_mss/val", transform=transforms.ToTensor())

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=3, pin_memory=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=1, pin_memory=True)



