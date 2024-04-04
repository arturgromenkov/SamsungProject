from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from CHistNet import CHistNet
from torch.utils.data import DataLoader
from tqdm import tqdm

def display_cm(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.show()

model = CHistNet.load_from_checkpoint(checkpoint_path="lightning_logs/version_2/checkpoints/model_main_epoch=09.ckpt",
                                      hparams_file="lightning_logs/version_2/hparams.yaml")
model.eval()

test_dataset = ImageFolder(root="tcga_coad_msi_mss/test", transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

y_true = []
y_pred = []

for inputs, targets in tqdm(test_loader):
    outputs = model(inputs.to("cuda"))
    y_true.extend(targets.tolist())
    y_pred.extend(outputs.argmax(dim=1).tolist())
    break

print(y_true)
print(y_pred)

display_cm(y_true,y_pred)
