#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd

import logging
import os
import sys
import argparse


#TODO: Import dependencies for Debugging andd Profiling

def test(model, testloader, criterion, device):
    
    model.eval()
    #hook.set_mode(smd.modes.EVAL)
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in testloader:
            
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
        
            output = model(data)
            test_loss += criterion(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(testloader.dataset), 100.0 * correct / len(testloader.dataset)
        )
    )

            
def train(model, trainloader, criterion, optimizer, device):
    
    model.train()
    #hook.set_mode(smd.modes.TRAIN)
    
    for batch_idx, (data, target) in enumerate(trainloader):
        
        # Move input and label tensors to the default device
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(trainloader.dataset),
                    100.0 * batch_idx / len(trainloader),
                    loss.item(),
                )
            )
   

       
def net():
    
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model
    

def create_data_loaders(data, batch_size):
        
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    train_path = os.path.join(data, "train")
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transform)    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    test_path = os.path.join(data, "test")
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size )
    
    return trainloader, testloader

def main(args):
    
    #hook = smd.Hook.create_from_json_file()
    #hook.register_hook(model)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model=net()
    model = model.to(device)
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
        
    trainloader, testloader = create_data_loaders(args.data_dir, args.batch_size )
    
    for epoch in range(1, args.epochs +1 ):
        model=train(model, trainloader, loss_criterion, optimizer, device)
        test(model, testloader, loss_criterion, device)
    
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pt'))
    

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
       
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "N", help = "learning rate (default: 0.1)" )
    parser.add_argument( "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type = int, default = 2, metavar="N", help="number of epochs to train (default: 2)"    )
   
                        
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
   
        
    args=parser.parse_args()
    
    main(args)
