# cnn-continual-learning

only works for datasets with 10 output classes as of now
CIFAR-10, SVHN, EuroSAT, FashionMNIST

available models (no loaded weights):
ResNet18, VGG-16

to run: 
'''batch
py run.py --num_tasks 4 --model resnet18 --seed 42 --epochs 1
'''
change arguments as needed

results are logged in a .log file
