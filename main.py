import os
from glob import glob
import shutil
import numpy as np
import click
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter

from datasets import MNISTDataset
from models import LinearModel, ConvModel


@click.group()
def cli():
    pass

@cli.command()
@click.option('--model-name', '-M', default='conv')
@click.option('--epoch', '-E', default=10)
def train(model_name, epoch):
    train_ds = MNISTDataset(train=True, limit=60000)
    val_ds = MNISTDataset(train=False, limit=10000)
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False, num_workers=4)

    match model_name:
        case 'linear':
            model = LinearModel()
        case 'conv':
            model = ConvModel()
        case _:
            raise RuntimeError('Invalid model name', model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    d = f'logs/{model_name}'
    if os.path.isdir(d):
        for p in glob(f'{d}/*'):
            print('rm', p)
            os.unlink(p)
    writer = SummaryWriter(log_dir=d)

    for e in range(epoch):
        train_losses = []
        t = tqdm(train_loader)
        for (x, gts) in t:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, gts)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            t.set_description(f'loss: {loss.item():.3f}')
            t.refresh()
        train_loss = np.mean(train_losses)

        val_losses = []
        t = tqdm(val_loader)
        for (x, gts) in t:
            with torch.set_grad_enabled(False):
                preds = model(x)
                loss = criterion(preds, gts)
                val_losses.append(loss.item())
            t.set_description(f'loss: {loss.item():.3f}')
            t.refresh()

        val_loss = np.mean(val_losses)
        writer.add_scalar('loss/train', train_loss, e)
        writer.add_scalar('loss/val', val_loss, e)
        print(f'[{e}/{epoch}] train loss:{train_loss:.3f} val loss: {val_loss:.3f}')




@cli.command()
def predict():
    print('predict')



if __name__ == '__main__':
    cli()
