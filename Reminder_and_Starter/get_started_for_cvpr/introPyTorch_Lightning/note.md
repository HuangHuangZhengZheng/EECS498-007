# Introduction to PyTorch Lightning

```python
import torch
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        ...
    def forward(self, x):
        ...

    #@override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #@override
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    
trainer = pl.Trainer(max_epochs=10， accelerator='auto', devices='auto')
    
trainer.fit(model, train_loader, val_loader)
```
