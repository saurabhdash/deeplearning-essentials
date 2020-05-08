from __future__ import print_function
from utils import *
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from models.lenet5 import Net   # Import the network from models/lenet5.py
import argparse


def main():
    '''
        This is the main function where the training and testing is called.
        parser is used to pass arguments from the command line for a cleaner code.
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20 , metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--mode', default='Train', type=str, help='To Train or to Test')
    parser.add_argument('--resume', '-r', action='store_true', default =False, help='resume from checkpoint')
    parser.add_argument('--log-interval', type=int, default=10, metavar='L', help='Display the log')

    args = parser.parse_args()

    ''' Training starts from scratch unless resumed'''
    start_epoch = 1

    ''' If there is any checkpoint to be initialized from, pass the path'''
    modelpath = './checkpoint/mnist_cnn.pt'

    '''
    This fixes the random seed to a definite value for better reproducibility. 
    The network is exactly initialized from the same random state whenever it is run so the result is invariant.
    '''
    torch.manual_seed(args.seed)

    '''
    Data Loading in pytorch is very elegant and clean, once you use this, you wouldn't want to go back to anything else.
    This comprises of 2 classes, dataset and dataloader making the entire procedure modular
    
    Dataset: This is a class which handles where the data is, how much it is, and what to return given an index (torchvision has this built-in for mnist, cifar10 etc.)
    Dataloader: This class is responsible for fetching data from the dataset. whenever this is called, it gives the dataset class a list of indices and fetches the data corresponding to those indices
    '''
    kwargs = {'num_workers': 1, 'pin_memory': True} # Bunch of optional arguments for dataloader

    ''' Define training dataloader and training dataset classes'''
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # Always the data must be normalised to ensure the test and train datasets have the same statistics
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    ''' Define test dataloader and test dataset classes'''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    ''' Transfer the model to the GPU'''
    model = Net().to('cuda')

    ''' This is the loss function we will try to minimize'''
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    ''' Define the optimizer to be used. This could be something very simple as vanilla SGD or Adam. We will use the adam optimizer because it works well (alchemy)
        optimizer needs access to model parameters to update them'''
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ''' This might be optional but good to know about this. 
        A scheduler changes the learning rate to beta * lr. 
        This reduction in lr is usually helpful for convergence in the later stages'''
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=args.gamma)

    ''' As models get more complex, we might need to keep track of how loss is behaving during training, so we use tensorboardX.'''
    writer = SummaryWriter()

    ''' If we want to train the model'''
    if args.mode == 'Train':

        ''' We can resume training from where we left off '''
        if args.resume == True:
            model, start_epoch = load_checkpoint(model, modelpath)
            print('here')


        ''' Now that we have defined everything, we can start actual training. We train for args.epoch number of epochs'''
        for epoch in range(start_epoch, args.epochs + 1):
            ''' Train for an epoch'''
            train(args, model, train_loader, optimizer, criterion, writer, epoch)

            ''' Test the model with the updated parameters.
                Doing this every epoch slows the training process and can be done every few epochs'''
            test_accuracy = test(args, model, test_loader, criterion, writer, epoch)

            ''' If we are using an scheduler, use it to update the parameters'''
            # scheduler.step()

            ''' If the save model flag is True, we can save the model as a pickle file'''
        if args.save_model:
            save_checkpoint(model, test_accuracy, epoch)

    elif args.mode == 'Test':
        ''' Instead if we want to test the model'''
        print('Testing the model')
        model, _ = load_checkpoint(model, modelpath)
        test(args, model, test_loader, criterion, writer, epoch=0)

    else:
        ''' We can implement some visualization mode to generate plots'''
        NotImplementedError







if __name__ == '__main__':
    main()
