import torchvision
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
import torch


class RotationTransform:
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return tf.rotate(x, self.angle, fill=(0,))
     

def get_CIFAR10_task(batch_size=32):

    transform_pipeline = transforms.Compose([
            # transforms.Resize((224, 224)),
			transforms.RandomRotation(degrees=(-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])

    CIFAR10_train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_pipeline, download=True)
    CIFAR10_train_loader = torch.utils.data.DataLoader(dataset=CIFAR10_train_data,batch_size=batch_size,num_workers=2,shuffle=True,pin_memory=True)
    CIFAR10_test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_pipeline, download=True)
    CIFAR10_test_loader = torch.utils.data.DataLoader(dataset=CIFAR10_test_data, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

    return CIFAR10_train_loader, CIFAR10_test_loader, 10


def get_EMNIST_task(batch_size=32):

    # 47 classes

    transform_pipeline = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.RandomRotation(degrees=(-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3081,0.3081)),
        ])

    EMNIST_train_data = torchvision.datasets.EMNIST(root='./data',train=True, split='balanced', transform=transform_pipeline, download=True)
    EMNIST_train_loader = torch.utils.data.DataLoader(dataset=EMNIST_train_data, batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=True)
    EMNIST_test_data = torchvision.datasets.EMNIST(root='./data', train=False, split='balanced', transform=transform_pipeline, download=True)
    EMNIST_test_loader = torch.utils.data.DataLoader(dataset=EMNIST_test_data, batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=False)

    return EMNIST_train_loader, EMNIST_test_loader, 47


def get_FashionMNIST_task(batch_size=32):
      
    transform_pipeline = transforms.Compose([
          transforms.Grayscale(num_output_channels=3),
          transforms.Resize((32, 32)),
          transforms.RandomRotation(degrees=(-30,30)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.2860,0.2860,0.2860), std=(0.3530,0.3530,0.3530))
    ])

    FashionMNIST_train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform_pipeline, download=True)
    FashionMNIST_train_loader = torch.utils.data.DataLoader(dataset=FashionMNIST_train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    FashionMNIST_test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform_pipeline, download=True)
    FashionMNIST_test_loader = torch.utils.data.DataLoader(dataset=FashionMNIST_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return FashionMNIST_train_loader, FashionMNIST_test_loader, 10


def get_GTSRB_task(batch_size=32):
      
    transform_pipeline = transforms.Compose([
          transforms.Resize((32, 32)),
          transforms.RandomRotation(degrees=(-30,30)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.3403, 0.3129, 0.3210), std=(0.2724, 0.2608, 0.2669))
    ])

    GTSRB_train_data = torchvision.datasets.GTSRB(root='./data', split='train', transform=transform_pipeline, download=True)
    GTSRB_train_loader = torch.utils.data.DataLoader(dataset=GTSRB_train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    GTSRB_test_data = torchvision.datasets.GTSRB(root='./data', split='test', transform=transform_pipeline, download=True)
    GTSRB_test_loader = torch.utils.data.DataLoader(dataset=GTSRB_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return GTSRB_train_loader, GTSRB_test_loader, 43


def get_TinyImageNet_task(batch_size=32):
      
    transform_pipeline = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.RandomRotation(degrees=(-30,30)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    TinyImageNet_train_data = torchvision.datasets.ImageFolder(root=f"./data/tiny-imagenet-200/train", transform=transform_pipeline)
    TinyImageNet_train_loader = torch.utils.data.DataLoader(TinyImageNet_train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    TinyImageNet_test_data = torchvision.datasets.ImageFolder(root=f"./data/tiny-imagenet-200/val", transform=transform_pipeline)
    TinyImageNet_test_loader = torch.utils.data.DataLoader(TinyImageNet_test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return TinyImageNet_train_loader, TinyImageNet_test_loader, 200


def get_SVHN_task(batch_size=32):

    transform_pipeline = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])  
    ])

    SVHN_train_data = torchvision.datasets.SVHN(root='./data', split='train', transform=transform_pipeline, download=True)
    SVHN_train_loader = torch.utils.data.DataLoader(dataset=SVHN_train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    SVHN_test_data = torchvision.datasets.SVHN(root='./data', split='test', transform=transform_pipeline, download=True)
    SVHN_test_loader = torch.utils.data.DataLoader(dataset=SVHN_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return SVHN_train_loader, SVHN_test_loader, 10


def get_EuroSAT_task(batch_size=32):
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3443, 0.3803, 0.4078], std=[0.1519, 0.1309, 0.1398])
    ])

    train_split = 0.8
    dataset = torchvision.datasets.EuroSAT(root='./data', transform=transform_pipeline, download=True)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, test_loader, 10


def get_continual_learning_tasks(task_count, batch_size=32):
     
    task_functions = [get_CIFAR10_task, get_FashionMNIST_task, get_SVHN_task, get_EuroSAT_task]

    train_loaders = []
    test_loaders = []
    output_neurons = []

    for i in range(task_count):
        curr_train_loader, curr_test_loader, output_neuron_count = task_functions[i](batch_size=batch_size)
        train_loaders.append(curr_train_loader)
        test_loaders.append(curr_test_loader)
        output_neurons.append(output_neuron_count)

    return train_loaders, test_loaders, output_neurons