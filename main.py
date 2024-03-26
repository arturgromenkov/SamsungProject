import PIL.Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from CHistNet import CHistNet
from loader import HistologyDataModule
from callbacks import checkpoint_callback
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

trainer = Trainer(max_epochs=1,
                  accelerator='cuda',
                 gradient_clip_val=0.5,
                  precision='16-mixed',
                  callbacks=[checkpoint_callback(every_n_epochs=1)]
                  )
model = CHistNet()
data_module = HistologyDataModule()
trainer.fit(model,data_module)

model.eval()

y_true = []
y_pred = []

for inputs, targets in tqdm(data_module.val_dataloader()):
    outputs = model(inputs)
    y_true.extend(targets.tolist())
    y_pred.extend(outputs.tolist())
    break

print(y_true)
print(y_pred)

