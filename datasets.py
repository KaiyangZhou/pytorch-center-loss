import torch
import torchvision
from torch.utils.data import DataLoader

import transforms

class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

__factory = {
    'mnist': MNIST,
}

def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)