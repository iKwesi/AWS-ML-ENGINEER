#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import argparse
import os
import json
import sys
import logging

#TODO: Import dependencies for Debugging andd Profiling
# from smdebug import modes
# from smdebug.profiler.utils import str2bool
# from smdebug.pytorch import get_hook
# import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
#     hook.set_mode(smd.modes.EVAL)
#     hook.register_loss(criterion)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # iterate over data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
#             test_loss += criterion(output, target, reduction = "sum").item() # sum up the batch loss
            test_loss += criterion(output, target).item() * data.size(0)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target.data)
#             preds = output.argmax(dim=1, keepdim=True)
#             correct += preds.eq(target.view_as(preds)).sum().item()
            

    test_loss /= len(test_loader.dataset)
    test_acc = correct /len(test_loader.dataset)

#     print(f"\nTest set: Average loss: {test_loss:.4f} | Accuracy: {correct}/{len(test_loader.dataset)} -> {100.0 * correct/len(test_loader.dataset)}")
    logger.info(f"\nTest set: Average loss: {test_loss:.4f} | Accuracy: {100.0 * test_acc}%")
    return test_loss


def train(model, train_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
#     hook.set_mode(smd.modes.TRAIN)
#     hook.register_loss(criterion)
    running_loss = 0
    correct = 0

    # iterate over the data
#     for batch_idx, (data, target) in enumerate(train_loader):
#     for data, target in enumerate(train_loader, 1):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(data)
#         preds = outputs.argmax(dim=1, keepdim=True)
        _, preds = torch.max(outputs, 1)
#         correct += preds.eq(target.view_as(preds)).sum().item()
        correct += torch.sum(preds == target.data)
        loss = criterion(outputs, target)
        running_loss += loss.item() * data.size(0)
        
        # backward + optimize
        loss.backward()
        optimizer.step()
    train_loss = loss/len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
        # print training info 
        
#     print(
#         "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             running_loss/len(train_loader.dataset), correct, len(train_loader.dataset), 100.0 * correct / len(train_loader.dataset)
#         )
#     )
    logger.info(f"\nTrain set: Average loss: {train_loss:.4f}| Accuracy: {100 * train_acc:.4f}%")
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    # freeze model wieights
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

# create data loaders
def create_data_loaders(data_dir, batch_size, test_batch_size, num_cpus, num_gpus):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # data augmentation and normalization for train data

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # normalization but no data augmentation for test data
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    worker_count = 4
    worker_count =4
    if num_gpus > 0:
        worker_count = min(num_gpus, worker_count)
    elif num_cpus > 0:
        worker_count = min(num_cpus, worker_count)

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=worker_count)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=worker_count)

    return train_loader, val_loader 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir,"model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model
# --------------------------------------------------------------------------------------------------------------------------------------------

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # setup hook
    
#     hook = smd.Hook.create_from_json_file()
#     hook.register_hook(model)
    # send model to the right device
    hook = ''
    model=model.to(device)

    train_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size, args.test_batch_size, args.num_cpus, args.num_gpus)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info(f"Training start...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch: {epoch}\nTraining")
        train(model, train_loader, loss_criterion, optimizer, device, hook)
        '''
    TODO: Test the model to see its accuracy
    '''
        logger.info(f"Testing ...")
#         test(model, valid_loader, loss_criterion, device, hook)
        loss = test(model, valid_loader, loss_criterion, device, hook)
        
    
    
    
    '''
    TODO: Save the trained model
    '''
    logger.info(f"Saving the model...")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="ResNet18 for dogbreed prediction")
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )

    # hyperparameters for optimizer
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="LR", help="learning rate (default: 1e-4)"
    )
    
#     container environment

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])


    args=parser.parse_args()
    
    main(args)
