import torch
import torchvision
import torch.nn as nn
import os
import sys
import argparse
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create parser.')
    parser.add_argument('--model_type', default='alexnet', choices=['alexnet', 'resnet', 'vgg', 'sknet'], type=str, help='model type')
    parser.add_argument('--data_type', default='CIFAR10', choices=['CIFAR10', 'FashionMNIST'], type=str, help='data type')
    parser.add_argument('--test_batch_size', default=100, type=int, help='test batch size')
    parser.add_argument('--num_workers', default=40, type=int, help='num workers')
    parser.add_argument('--device', type=str, default='0', help='ID of GPU to use')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--modify', action='store_true', help='whether the model is modified')
    parser.add_argument('--model_name', type=str, default='', help='name of the model stored')
    args = parser.parse_args()
    if args.modify:
        from mod_models import * 
    else:
        from models import * 
    # Initialize device   
    init_device(args.device, args.local_rank)

    # Make data dir if necessary
    data_path = os.path.join('../data', args.data_type)
    model_checkpoint_path = os.path.join('../output', args.data_type)

    # Load train data and test data
    _, test_loader = data_loader(data_path,
                                    args.data_type,
                                    batch_size_test=args.test_batch_size,
                                    num_workers=args.num_workers)
    # Define what device we are using
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model
    if args.model_type == 'alexnet':
        model = AlexNet(num_classes=10).to(device)
    elif args.model_type == 'resnet':
        model = ResNet(num_classes=10).to(device)
    elif args.model_type == 'vgg':
        model = VGG().to(device)
    elif args.model_type == 'sknet':
        model = SKNet(10).to(device)
    # Configure loss function
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(os.path.join(model_checkpoint_path, args.model_name))["state_dict"])
    test_accuracy, _ = test(model, criterion, test_loader, device)
    print("test_acc: {:.3f}".format(test_accuracy))   