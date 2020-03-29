'''
    This is the utilities file which has the train, test and various functions defined to be called in main.
'''
from __future__ import print_function
import torch
import os


def train(args, model, train_loader, optimizer, criterion, writer, epoch):
    '''
    This is the training function that runs every epoch.
    inputs:
    args : arguments that are passed from the command line
    model : the network that is being trained
    train_loader: the dataloader class that can fetch data from the train dataset
    optimizer: This is the algorithm that given the gradients tells how to exactly traverse the loss surface
    criterion: This is the loss function (mean square, cross entropy, negative log likelihood etc.)
    writer: This is the tensorboard object that  keeps a track of the loss as it progresses
    epoch: to keep a track of which iteration we are in.
    '''

    '''This essentially tells pytorch that we would be training the model, so it tracks the gradients of each model parameter(weights) 
       and allows batchnorm to keep a running estimate of mean and variance, this also turns on dropouts (if any)'''
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        '''
            As we usually cannot fit the entire dataset into memory, we divide it into chunks or minibatches. 
            at every step, this for loop fetches the input(data) and the ground truth outputs (targets) of the size that is defined in args.batch_size
            train_loader is wrapped in enumerate to also keep track of the index of the batch
        '''

        '''Transfer the data and target tensors to the gpu for computation'''
        data, target = data.to('cuda'), target.to('cuda')

        '''Set all the parameters' gradient values to 0, else they accumulate everytime loss.backwards() is called.'''
        optimizer.zero_grad()

        ''' Now finally apply the model over the data to generate outputs'''
        output = model(data)

        ''' Now that we have the predicted outputs, use whatever loss criterion we defined to calculate the loss of this minibatch'''
        loss = criterion(output, target)

        '''Backpropagate to generate gradients with respect to all the model parameters'''
        loss.backward()

        '''Now take all the gradient information and use it to update the model parameters'''
        optimizer.step()

        '''Actual computation is over, this part is to display and track the training'''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            ''' Add this minibatch loss to tensorboard to be tracked'''
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)



def test(args, model, test_loader, criterion, writer, epoch):
    '''
    This is the testing function
    inputs:
    args : arguments that are passed from the command line
    model : the network that is being trained
    test_loader: the dataloader class that can fetch data from the test dataset
    criterion: This is the loss function (mean square, cross entropy, negative log likelihood etc.)
    writer: This is the tensorboard object that  keeps a track of the loss as it progresses
    epoch: to keep a track of which iteration we are in.
    '''

    ''' This sets the model to the eval phase where batchnorm and dropout behaviour is changed'''
    model.eval()

    ''' The metrics we are concerned with'''
    test_loss = 0
    correct = 0
    accuracy = 0

    with torch.no_grad():
        ''' Unlike the previous time we dont need to keep a track of all the gradient information of the model parameters, so we turn off autograd (which tracks gradients). 
        This saves a shit ton of memory so we can use a larger batch size for testing which makes the testing faster'''

        for data, target in test_loader:
            '''
            As we usually cannot fit the entire dataset into memory, we divide it into chunks or minibatches. 
            at every step, this for loop fetches the input(data) and the ground truth outputs (targets) of the size that is defined in args.test_batch_size
            '''

            '''Transfer the data and target tensors to the gpu for computation'''
            data, target = data.to('cuda'), target.to('cuda')

            ''' Now  apply the model over the data to generate outputs'''
            output = model(data)

            '''Add the loss of the current batch to the total loss'''
            test_loss += criterion(output, target).item()  # sum up batch loss

            '''The label predicted is the output index with the largest probability, so an argmax is taken'''
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            '''check how many predicted labels are equal to the target labels (.item() converts a tensor to a float/int)'''
            correct += pred.eq(target.view_as(pred)).sum().item()

    ''' Divide the total loss accumulated by the number of test samples'''
    test_loss /= len(test_loader.dataset)

    ''' Find the accuracy'''
    accuracy = 100. * correct / len(test_loader.dataset)

    '''Plot/ print the results.'''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    '''Add this loss to tensorboard'''
    writer.add_scalar('Test/Loss', test_loss, epoch)

    return accuracy

def save_checkpoint(model, test_accuracy, epoch):
    '''
        This function creates a dictionary of state variables and pickles them to save a checkpoint
    '''
    print('Saving...')

    ''' Create a dictionary to save some parameters related to the state of the model, this allows us to resume training'''
    state = {
        'net': model.state_dict(),
        'acc': test_accuracy,
        'epoch': epoch,
    }

    ''' Create a directory to save the checkpoints'''

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+'mnist_cnn.pt')


def load_checkpoint(model, modelpath):
    '''
        This function loads a checkpoint
    '''
    print('Loading...')
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    print('Training resuming from epoch {}'.format(start_epoch))
    return model, start_epoch