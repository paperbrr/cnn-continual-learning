import torch
import numpy as np
import logging
import argparse
from train import train_continual
from models import get_VGG16, get_ResNet18
from tasks import get_continual_learning_tasks


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_tasks", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--model", choices=['resnet18', 'vgg16'], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    logging.basicConfig(filename=f'{args.model}_{args.num_tasks}tasks_{args.epochs}epochs.log',filemode='a', encoding='utf-8', level=logging.INFO)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {'resnet18':get_ResNet18, 
              'vgg16':get_VGG16}
    model = models[args.model](out_features=10)

    train_loaders, test_loaders, output_neuron_counts = get_continual_learning_tasks(task_count=4)
    learning_accs, running_test_accs = train_continual(model=model, train_loaders=train_loaders, test_loaders=test_loaders, output_neuron_counts=output_neuron_counts,device=device, num_tasks=args.num_tasks, epochs=args.epochs)
    
    score = np.mean([running_test_accs[i][-1] for i in running_test_accs.keys()])
    forget = np.mean([max(running_test_accs[i])-running_test_accs[i][-1] for i in range(args.num_tasks)])
    learning_acc = np.mean(learning_accs)

    logging.info(f"Accuracy: {score}, "
                f"Forgetting: {forget}, "
                f"Learning Acc: {learning_acc}, "
                f"Learning Accs: {learning_accs}, "
                f"Running Test Accs: {running_test_accs}")


if __name__ == '__main__':
    main()