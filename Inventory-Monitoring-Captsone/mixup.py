from PIL import ImageFile
import smdebug.pytorch as smd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchmetrics
import os
import json
import logging
import sys
import copy

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


ImageFile.LOAD_TRUNCATED_IMAGES = True

accuracy = torchmetrics.Accuracy()


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()  # set model to evaluate model
    hook.set_mode(smd.modes.EVAL)  # set debugging hook
    test_loss = 0
    correct = 0
    alpha = 0.8

    # disable gradient calculation
    with torch.no_grad():
        # iterate over data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, y1, y2, lam = mixup_data(data, target, alpha)
            target = lam * y1 + (1 - lam) * y2
            output = model(data)
#             test_loss += criterion(output, target, reduction = "sum").item() # sum up the batch loss
            test_loss += criterion(output, target).item() * data.size(0)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target.data).item()
            batch_acc = accuracy(preds.cpu(), target.long().cpu())

    test_loss = test_loss / len(test_loader)
    test_acc = correct.double() / len(test_loader.dataset)
    test_accuracy = accuracy.compute()

    accuracy.reset()

    logger.info(
        f"\nTest set: Average loss: {test_loss:.4f} | Accuracy: {100.0 * test_acc}% | test_accuracy: {test_accuracy}")
    return test_loss


def train(model, train_loader, criterion, optimizer):
    model.train()  # set model to training mode
    running_loss = 0
    correct = 0
    alpha = 0.8
    num_classes = 5
    train_accuracy = torchmetrics.Accuracy()

    # iterate over the data
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, y1, y2, lam = mixup_data(inputs, labels, alpha)
        labels = lam * y1 + (1 - lam) * y2
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
#       preds = outputs.argmax(dim=1, keepdim=True)
        _, preds = torch.max(outputs, dim=1)
#       correct += preds.eq(target.view_as(preds)).sum().item()
        correct += torch.sum(preds == labels).item()
        loss = criterion(outputs, labels.long())
        running_loss += loss.item() * inputs.size(0)
#       logger.info(f"loss: {loss}| acc: {torch.sum(preds==target).item()/len(target)}")
        acc = train_accuracy(preds.cpu(), labels.long().cpu())

        # backward + optimize
        loss.backward()
        optimizer.step()
    train_loss = loss/len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    new_acc = train_accuracy.compute()

    train_accuracy.reset()

    logger.info(
        f"\nTrain set: Average loss: {train_loss:.4f}| Accuracy: {100.0 * train_acc:.4f}% | accuracy: {new_acc}")


def train(model, train_loader, criterion, optimizer, device, hook):
    model.train()  # set model to training mode
    hook.set_mode(smd.modes.TRAIN)  # set debugging hook
    running_loss = 0
    correct = 0
    alpha = 0.8
    num_classes = 5
    train_accuracy = torchmetrics.Accuracy()

    # iterate over the data
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        data, y1, y2, lam = mixup_data(data, target, alpha)
        target = lam * y1 + (1 - lam) * y2
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(data)
#         preds = outputs.argmax(dim=1, keepdim=True)
        _, preds = torch.max(outputs, dim=1)
#         correct += preds.eq(target.view_as(preds)).sum().item()
        correct += torch.sum(preds == target).item()
        loss = criterion(outputs, target)
        running_loss += loss.item() * data.size(0)
#         logger.info(f"loss: {loss}| acc: {torch.sum(preds==target).item()/len(target)}")
        acc = accuracy(preds.cpu(), target.long().cpu())

        # backward + optimize
        loss.backward()
        optimizer.step()
    train_loss = loss/len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    new_acc = train_accuracy.compute()

    accuracy.reset()

    logger.info(
        f"\nTrain set: Average loss: {train_loss:.4f}| Accuracy: {100.0 * train_acc:.4f}% | lib_Accuracy: {new_acc}")
# -------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # load the pretrained model
    model = models.resnet101(pretrained=True)

#     freeze model weights
    # for param in model.parameters():
    #     param.requires_grad = False

    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)

    return model


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def create_data_loaders(train_dir, test_dir, batch_size, test_batch_size, num_cpus, num_gpus):

    # data augmentation and normalization for train data

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5),  # applies random brightness
        transforms.RandomRotation(degrees=45),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # normalization but no data augmentation for test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    worker_count = 4
    worker_count = 4
    if num_gpus > 0:
        worker_count = min(num_gpus, worker_count)
    elif num_cpus > 0:
        worker_count = min(num_cpus, worker_count)

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)
#     test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=worker_count, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size,
                             num_workers=worker_count, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size, num_workers=worker_count)

    return train_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # send model to the right device
    model = model.to(device)

    hook = smd.Hook.create_from_json_file()

    train_loader, test_loader = create_data_loaders(
        args.train_dir, args.test_dir, args.batch_size, args.test_batch_size, args.num_cpus, args.num_gpus)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    logger.info(f"Training start...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch: {epoch}\nTraining")
        train(model, train_loader, loss_criterion, optimizer, device, hook)
#         logger.info(f"validating...")
#         test(model, valid_loader, loss_criterion, device)

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info(f"Testing ...")
    test(model, test_loader, loss_criterion, device, hook)

    '''
    TODO: Save the trained model
    '''
    logger.info(f"Saving the model...")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ResNet50 for inventory bin counting")
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         default="s3://sagemaker-us-east-1-429660041905/CV-final-project-dogImages"
#     )
    # hyperparameters for optimizer
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)"
    )
#     parser.add_argument(
#         "--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)"
#     )
#     parser.add_argument(
#         "--weight-decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)"
#     )

#     container environment

    parser.add_argument("--hosts", type=list,
                        default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--curren-host", type=str,
                        default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str,
                        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str,
                        default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test-dir", type=str,
                        default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--output_dir", type=str,
                        default=os.environ["SM_OUTPUT_DATA_DIR"])
#     parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int,
                        default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int,
                        default=os.environ["SM_NUM_CPUS"])

    args = parser.parse_args()

    main(args)
