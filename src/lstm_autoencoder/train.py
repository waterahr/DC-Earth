import torch
from torch import nn, optim
import torch.nn.functional as F
from LSTMAutoEncoder import LSTMAutoEncoder
from prepare_data import EarthDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The epochs need to be trained')
    parser.add_argument('-b', '--batch', type=int, default=4096,
                        help='The batch size in the training progress')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

def train(epochs=100, batch_size=4096, gpu_id=0):
    device = torch.device("cuda:{gid}".format(gid=gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    
    dataset = EarthDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model_prefix = "../../models/LSTMAutoEncoder/"
    os.makedirs(model_prefix, exist_ok=True)
    model = LSTMAutoEncoder()
    model.to(device)
    
    loss = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    
    for epoch in range(epochs):
        model.train()
        lr_scheduler.step()
        dataset.update_sampled_img()
        epoch_loss = 0.0
        for dt, lb in train_loader:
            dt, lb = dt.to(device), lb.to(device)
            out = model(dt)
            loss_value = loss(out, lb)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item() * dt.shape[0]
            
        save_loss = np.sqrt(epoch_loss / len(dataset))
        print("Epoch:{epoch}, Loss:{loss:.6f}".format(epoch=epoch+1, loss=save_loss))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "{model}epoch_{epoch:03}_loss_{loss:.2f}.pth".format(model=model_prefix, epoch=epoch+1, loss=save_loss))
            
            
if __name__ == "__main__":
    args = parse_arg()
    train(args.epochs, args.batch)