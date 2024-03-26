import PIL.Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from CHistNet import CHistNet
from loader import Loader
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
loader = Loader(batch_size=512)
trainer.fit(model, loader.train_loader,loader.val_loader)

model.eval()

y_true = []
y_pred = []

for inputs, targets in tqdm(loader.val_loader):
    outputs = model(inputs)
    y_true.extend(targets.tolist())
    y_pred.extend(outputs.argmax(dim=1).tolist())
    break

print(y_true)
print(y_pred)

def display_cm(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.show()


display_cm(y_true,y_pred)
