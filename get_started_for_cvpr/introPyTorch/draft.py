import pytorch_lightning as pl
import torch
from torch import nn, optim

class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(28 * 28, 10)  # 假设是一个简单的线性层

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
