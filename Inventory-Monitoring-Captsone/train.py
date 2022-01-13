import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import os
import json
import logging
import sys
import copy

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval() # set model to evaluate model
    hook.set_mode(smd.modes.EVAL) # set debugging hook
    test_loss = 0
    correct = 0

    # disable gradient calculation
    with torch.no_grad():
        # iterate over data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
#             test_loss += criterion(output, target, reduction = "sum").item() # sum up the batch loss
            test_loss += criterion(output, target).item() * data.size(0)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target).item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct /len(test_loader.dataset)

    logger.info(f"\nTest set: Average loss: {test_loss:.4f} | Accuracy: {100.0 * test_acc}%")
    return test_loss

def train(model, train_loader, criterion, optimizer, device, hook):
    model.train() # set model to training mode
    hook.set_mode(smd.modes.TRAIN) # set debugging hook 
    running_loss = 0
    correct = 0

    # iterate over the data
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
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
        
        # backward + optimize
        loss.backward()
        optimizer.step()
    train_loss = loss/len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    logger.info(f"\nTrain set: Average loss: {train_loss:.4f}| Accuracy: {100.0 * train_acc:.4f}% ")
# -------------------------------------------------------------------------------------------------------------  

#--------------------------------------------------------------------------------------------------------------
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # load the pretrained model
    model = models.resnet50(pretrained=True)
    
#     freeze model weights
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)

    return model

def create_data_loaders(train_dir, test_dir, batch_size, test_batch_size, num_cpus, num_gpus):

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

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)
#     test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=worker_count)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=worker_count)
#     test_loader = DataLoader(test_dataset, batch_size, num_workers=worker_count)

    return train_loader, test_loader 

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # send model to the right device
    model=model.to(device)
    
    hook = smd.Hook.create_from_json_file()

    train_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.batch_size, args.test_batch_size, args.num_cpus, args.num_gpus)

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

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="ResNet50 for inventory bin counting")
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

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--curren-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
#     parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])


    args=parser.parse_args()
    
    main(args)








# #TODO: Import your dependencies.
# #For instance, below are some dependencies you might need if you are using Pytorch
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data.dataloader import DataLoader

# import copy
# import argparse
# import os
# import logging
# import sys
# from tqdm import tqdm
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# logger=logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# import smdebug.pytorch as smd

# def test(model, test_loader, criterion, hook):
#     '''
#     TODO: Complete this function that can take a model and a 
#           testing data loader and will get the test accuray/loss of the model
#           Remember to include any debugging/profiling hooks that you might need
#     '''
#     model.eval()
#     hook.set_mode(smd.modes.EVAL)
#     running_loss=0
#     running_corrects=0
    
#     for inputs, labels in test_loader:
#         outputs=model(inputs)
#         loss=criterion(outputs, labels)
#         _, preds = torch.max(outputs, 1)
#         running_loss += loss.item() * inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data)

#     total_loss = running_loss // len(test_loader)
#     total_acc = running_corrects.double() // len(test_loader)
    
#     logger.info(f"Testing Loss: {total_loss}")
#     logger.info(f"Testing Accuracy: {total_acc}")

# def train(model, train_loader, validation_loader, criterion, optimizer, epochs, hook):
#     '''
#     TODO: Complete this function that can take a model and
#           data loaders for training and will get train the model
#           Remember to include any debugging/profiling hooks that you might need
#     '''
#     epochs=args.epochs
#     best_loss=1e6
#     image_dataset={'train':train_loader, 'valid':validation_loader}
#     loss_counter=0
    
#     for epoch in range(epochs):
#         logger.info(f"Epoch: {epoch}")
#         for phase in ['train', 'valid']:
#             if phase=='train':
#                 model.train()
#                 hook.set_mode(smd.modes.TRAIN)
#             else:
#                 model.eval()
#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in image_dataset[phase]:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 if phase=='train':
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                 _, preds = torch.max(outputs, 1)
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss // len(image_dataset[phase])
#             epoch_acc = running_corrects // len(image_dataset[phase])
            
#             if phase=='valid':
#                 if epoch_loss<best_loss:
#                     best_loss=epoch_loss
#                 else:
#                     loss_counter+=1


#             logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
#                                                                                  epoch_loss,
#                                                                                  epoch_acc,
#                                                                                  best_loss))
#         if loss_counter==1:
#             break
#         if epoch==0:
#             break
#     return model
    
# def net():
#     '''
#     TODO: Complete this function that initializes your model
#           Remember to use a pretrained model
#     '''
#     model = models.resnet50(pretrained=True)

#     for param in model.parameters():
#         param.requires_grad = False   

#     model.fc = nn.Sequential(
#                    nn.Linear(2048, 128),
#                    nn.ReLU(inplace=True),
#                    nn.Linear(128, 5))
#     return model

# def create_data_loaders(data, batch_size):
#     '''
#     This is an optional function that you may or may not need to implement
#     depending on whether you need to use data loaders or not
#     '''
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         ])

#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         ])
    
#     dataset = ImageFolder(data, transform = test_transform)
# #     train_size = int(len(dataset) * 0.8)
# #     test_size = int(len(dataset)*0.2)
#     #valid_size = int(len(dataset)*0.2)
    
# #     lengths = [int(len(dataset)*0.65), int(len(dataset)*0.2), int(len(dataset)*0.15)]
    
#     train_data, test_data, validation_data = torch.utils.data.random_split(dataset, [5221, 2611, 2609])
    
#     train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers = 4)
#     validation_data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
#     return train_data_loader, test_data_loader, validation_data_loader

# # ------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------
# def main(args):
#     '''
#     TODO: Initialize a model by calling the net function
#     '''
    
#     logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
#     logger.info(f'Data Paths: {args.data}')
    
#     train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    
#     model=net()
#     hook = smd.Hook.create_from_json_file()
#     hook.register_hook(model)
    
#     '''
#     TODO: Create your loss and optimizer
#     '''
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
#     '''
#     TODO: Call the train function to start training your model
#     Remember that you will need to set up a way to get training data from S3
#     '''
#     logger.info("Starting Model Training")
#     model=train(model, train_loader, validation_loader, criterion, optimizer, args.epochs, hook)
    
#     '''
#     TODO: Test the model to see its accuracy
#     '''
#     logger.info("Testing Model")
#     test(model, test_loader, criterion, hook)
    
#     '''
#     TODO: Save the trained model
#     '''
#     logger.info("Saving Model")
#     torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

# if __name__=='__main__':
#     parser=argparse.ArgumentParser()
#     parser.add_argument('--learning_rate', type=float)
#     parser.add_argument('--batch_size', type=int)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
#     args=parser.parse_args()
#     print(args)
    
#     main(args)
