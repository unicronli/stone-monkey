import torch
import torchvision
import torch.nn as nn
import os
import sys
import argparse
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create parser.')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--model_type', default='alexnet', choices=['alexnet', 'resnet', 'vgg', 'sknet'], type=str, help='model type')
    parser.add_argument('--data_type', default='CIFAR10', choices=['CIFAR10', 'FashionMNIST'], type=str, help='data type')
    parser.add_argument('--train_batch_size', default=256, type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='test batch size')
    parser.add_argument('--lr0', default=0.1, type=float, help='init learning rate')
    parser.add_argument('--num_workers', default=40, type=int, help='num workers')
    parser.add_argument('--epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--device', type=str, default='0', help='ID of GPU to use')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--modify', action='store_true', help='whether the model is modified')
    parser.add_argument('--dec_lr', action='store_true', help='whether to decreace learning rate')
    args = parser.parse_args()
    if args.modify:
        from mod_models import * 
    else:
        from models import * 
    # Fix seed for reproducibility
    fix_seed(args.seed) 
    # Initialize device   
    init_device(args.device, args.local_rank)

    # Make data dir if necessary
    data_path = os.path.join('../data', args.data_type)
    model_checkpoint_path = os.path.join('../output', args.data_type)
    if os.path.isdir(data_path) == False:
        os.makedirs(data_path)
    if os.path.isdir(model_checkpoint_path) == False:
        os.makedirs(model_checkpoint_path)

    # Load train data and test data
    train_loader, test_loader = data_loader(data_path,
                                            args.data_type,
                                            batch_size_train=args.train_batch_size,
                                            batch_size_test=args.test_batch_size,
                                            num_workers=args.num_workers)
    # Define what device we are using
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model
    if args.model_type == 'alexnet':
        model = AlexNet(num_classes=10, gray=True if args.data_type=='FashionMNIST' else False).to(device)
    elif args.model_type == 'resnet':
        model = ResNet(num_classes=10, gray=True if args.data_type=='FashionMNIST' else False).to(device)
    elif args.model_type == 'vgg':
        model = VGG(gray=True if args.data_type=='FashionMNIST' else False).to(device)
    elif args.model_type == 'sknet':
        model = SKNet(10, gray=True if args.data_type=='FashionMNIST' else False).to(device)
    # Configure optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr0, momentum=0.9)
    ''' optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, betas=(0.9, 0.999)) '''
    # Configure loss function
    criterion = nn.CrossEntropyLoss()

    best_test_accuracy = 0
    model_name = 'model:{}_lr:{}_{}_modify:{}.pth'.format(args.model_type, args.lr0, 'dec' if args.dec_lr else 'con', 'true' if args.modify else 'false')
    lr0 = args.lr0
    for epoch in range(args.epochs):
        if args.dec_lr:
            current_lr = lr0 / 10**int(epoch/20)
            #不加max的时候可以到93.6
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        train_accuracy, train_loss = train(model, criterion, optimizer, train_loader, device)
        test_accuracy, test_loss = test(model, criterion, test_loader, device)
        print ("\nCurent epoch: {}".format(epoch + 1))
        print("train_acc: {:.3f}, train_loss: {:.3f}".format(train_accuracy, train_loss))
        print("test_acc: {:.3f}, test_loss: {:.3f}".format(test_accuracy, test_loss))
        if test_accuracy > best_test_accuracy:
            print("New best result")
            best_test_accuracy = test_accuracy
            save_checkpoint({
                    "train_acc": train_accuracy,
                    "train_loss": train_loss,
                    "test_acc": test_accuracy,
                    "test_loss": test_loss,
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1
                    }, model_checkpoint_path, model_name)
    print("-- Final Result --")
    model.load_state_dict(torch.load(os.path.join(model_checkpoint_path, model_name))["state_dict"])
    test_accuracy, _ = test(model, criterion, test_loader, device)
    print("test_acc: {:.3f}".format(test_accuracy))   