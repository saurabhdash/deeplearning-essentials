'''
    This file is where the lenet5 network is defined.
    It's easier if the models are in a different file as potentially we might need to try a bunch of models out
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Every network is defined as a pytorch class that inherits the methods in nn.Module (this library contains all the basic building blocks needed to build a network)
'''
class Net(nn.Module):
    def __init__(self):
        '''
            This is where all the different layers are defined, just defined.
            We still havent described how all the components of the network are connected to each other.
        '''
        super(Net, self).__init__()

        '''
            Let's define 2 convolutional layers named conv1 and conv2. 
            conv1 has 1 input channel (as we are dealing with MNIST), 32 output channels, a filter of size 3x3 and a stride of 1 (movement in horizontal and vertical direction).
            similarly, conv2 takes those 32 input channels, and generates 64 outputs channels. Again, the filter is a 3x3 filter and has a stride of 1
        '''
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        '''
            As neural networks have a huge number of parameters, they can memorise the entire training set, which might lead to terrible generalization, 
            A simple way to prevent that is dropouts, where we randomly set a node to 0 with a certain probability 0.25/0.5 everytime we do a forward pass during training.
        '''
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        '''
            After the convolutional layers have extracted the features, we use linear layers (fully connected) to combine them and make a prediction.
            fc1 has 9216 inputs and 128 outputs. fc2 has 10 outputs which give the probability of an input being one of 10 classes.
        '''
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        '''
            Now that we have all our lego blocks, we tell the network how they are connected. This describes how the input transforms as it passes through the model
        '''

        '''Pass the input through the first conv layer and then apply ReLU non-linearlity'''
        x = self.conv1(x)
        x = F.relu(x)

        ''' Repeat the same on the output of the first conv layer+non linearity'''
        x = self.conv2(x)
        x = F.relu(x)

        '''
            As stride 1 convolutions do not decrease the height and width of an image, we use max pool to reduce the size of the image to reduce the number of dimensions.
        '''
        x = F.max_pool2d(x, 2)

        ''' Dropout a few nodes to prevent overfitting'''
        x = self.dropout1(x)

        ''' Flatten the image to a 1d vector to be fed into the fully connected layers'''
        x = torch.flatten(x, 1)

        ''' Pass through a fc layer and then non linearity'''
        x = self.fc1(x)
        x = F.relu(x)

        '''Drop a few nodes'''
        x = self.dropout2(x)

        ''' Pass through another fully connected layer to get the outputs'''
        output = self.fc2(x)


        ''' FYI: No need to calculate softmax to find the actual probabilities of the digits, that is handled by nn.CrossEntropyLoss() '''

        return output
