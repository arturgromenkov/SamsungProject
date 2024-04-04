from typing import Any
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn.functional import cross_entropy
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import optim
import torch

class CHistNet(LightningModule):
    def __init__(self,learning_rate = 9.120108393559096e-06, img_shape=(224,224,3),output_nodes=2):
        # call the parent constructor and other hypos
        super().__init__()
        self.learning_rate = learning_rate
        # donwload the pretrained ghostnet model
        self.ghostnet = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True).to('cuda')
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        print(self.ghostnet.parameters())
        for param in self.ghostnet.parameters():
            param.requires_grad = False
        output_layer = torch.nn.Sequential(
            torch.nn.Linear(1280,output_nodes)
        )
        self.ghostnet.classifier = output_layer
        self.output_function = torch.nn.Softmax()
        print(self.ghostnet.__dict__)


    def forward(self,x):
        x = self.ghostnet(x)
        x = self.output_function(x)
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
