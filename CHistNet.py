from typing import Any
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import Linear
from torch.nn import Flatten
from torch.nn import LogSoftmax, Dropout, Softmax
from torch.nn.functional import cross_entropy
from torch import optim

class CHistNet(LightningModule):
    def __init__(self,learning_rate = 9.120108393559096e-06, img_shape=(224,224,3),output_nodes=2):
        # call the parent constructor and other hypos
        super().__init__()
        self.learning_rate = learning_rate
        self.flt = Flatten()
        self.fc1 = Linear(in_features=150528, out_features=500)
        self.relu1 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=output_nodes)
        self.output_func = Softmax()
    def forward(self,x):
        x = self.flt(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.output_func(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss = cross_entropy(logits,y)
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss = cross_entropy(logits,y)
        self.log('val_loss',loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),self.learning_rate, weight_decay=2)
