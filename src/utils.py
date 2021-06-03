import os
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST
import numpy as np
from random import Random
import torch.distributed as dist
import torch.nn as nn

def fix_seed(seed):
    # Fix seed for all torch device
    torch.manual_seed(seed)
    # Use deterministic mode on CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Fix seed on numpy
    np.random.seed(seed)


def init_device(device, local_rank=-1, backend='nccl', host=None, port=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if torch.cuda.device_count() > 1:
        host = host or os.environ.get('MASTER_ADDR', 'localhost')
        port = port or os.environ.get('MASTER_PORT', str(Random(0).randint(10000, 20000)))
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend)
        torch.cuda.set_device(local_rank)


def data_loader(data_root, data_type, batch_size_train=0, batch_size_test=0, num_workers=0):
    print("--- Prepare {} dataset ---".format(data_type))
    if data_type == 'CIFAR10':
        DATASET = CIFAR10
    elif data_type == 'FashionMNIST':
        DATASET = FashionMNIST
    train_loader = train_data_loader(data_root, data_type, DATASET, batch_size_train, num_workers) if batch_size_train else None
    test_loader = test_data_loader(data_root, data_type, DATASET, batch_size_test, num_workers) if batch_size_test else None
    return train_loader, test_loader


def train_data_loader(data_root, data_type, DATASET, batch_size_train, num_workers):
    # Training set transforms
    if data_type == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    elif data_type == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
    train_set = DATASET(root=data_root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size_train,
                                               shuffle=True,
                                               num_workers=num_workers)
    return train_loader


def test_data_loader(data_root, data_type, DATASET, batch_size_test, num_workers):
    # Testing set transforms
    if data_type == 'CIFAR10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    elif data_type == 'FashionMNIST':
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
    test_set = DATASET(root=data_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=batch_size_test,
                                               shuffle=False,
                                               num_workers=num_workers)
    return test_loader


def train(model, criterion, optimizer, train_loader, device):
    train_loss = 0.0
    correct = 0
    num_examples = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)
        # Forward
        output = model(data)
        loss = criterion(output, target)
        # Zero the parameters gradients
        optimizer.zero_grad()
        loss.backward()
        # Optimize
        optimizer.step()
        train_loss += loss.item()
        # Calc the corret classfications
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        num_examples += data.shape[0]
    
    train_accuracy = correct / num_examples
    return train_accuracy, train_loss


def test(model, criterion, test_loader, device):
    test_loss = 0.0
    correct = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        # Move to device
        data, target = data.to(device), target.to(device)
        # Forward
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # Calc the corret classfications
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy, test_loss


def save_checkpoint(state, checkpoint_path, filename):
    filepath = os.path.join(checkpoint_path, filename)
    torch.save(state, filepath)