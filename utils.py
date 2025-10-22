import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets 
from torchvision.transforms import v2
from networks import ConvNet
from distillation import DM

def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        img_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
        trainset = datasets.MNIST(data_path, train=True, download=True, transform=transform) 
        testset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        img_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
        trainset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) 
        testset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = trainset.classes

    elif dataset == 'CIFAR10':
        channel = 3
        img_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) 
        testset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = trainset.classes

    elif dataset == 'CIFAR100':
        channel = 3
        img_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) 
        testset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = trainset.classes

    else:
        exit(f'unknown dataset: {dataset}')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=4) #?
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    obj = {
        'channel': channel, 
        'img_size': img_size, 
        'num_classes': num_classes,
        'class_names': class_names,
        'mean': mean, 
        'std': std, 
        'trainset': trainset, 
        'testset': testset, 
        'trainloader': trainloader, 
        'testloader': testloader
    }

    return obj


def get_method(img_synth, label_synth, dataset, args):
    method = args.method 

    if method == 'DM':
        return DM(img_synth, label_synth, dataset, args)
    else: # just filler for now
        return DM(img_synth, label_synth, dataset, args)

class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def epoch(net, dataloader, optimizer, criterion, device, train=True):
    if train: 
        net.train()
    else:
        net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = net(imgs)
        loss = criterion(outputs, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def train_using_synth(net, img_synth, label_synth, args):
    net = net.to(args.device)
    img_synth = img_synth.to(args.device)
    label_synth = label_synth.to(args.device)
    lr = float(args.lr_net)
    num_epochs = int(args.train_iters)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    trainset = TensorDataset(img_synth, label_synth)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=True, num_workers=0)

    for _ in range(num_epochs):
        train_loss, train_acc = epoch(net, trainloader, optimizer, criterion, args.device)
        # print(f'Epoch {epoch} out of {num_epochs}: loss {train_loss : .4f}, accuracy {train_acc : .4f}')

def train_using_real(net, trainloader, args):
    net = net.to(args.device)
    lr = float(args.lr_net)
    num_epochs = int(args.train_iters)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    for _ in range(num_epochs):
        train_loss, train_acc = epoch(net, trainloader, optimizer, criterion, args.device)


def test(net, testloader, args):
    lr = float(args.lr_net)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)
    test_loss, test_acc = epoch(net, testloader, optimizer, criterion, args, train=False)
    return test_loss, test_acc


