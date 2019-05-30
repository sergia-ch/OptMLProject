import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dfw.dfw import DFW
from dfw.dfw.losses import MultiClassHingeLoss
from dfw.experiments.utils import accuracy

class SimpleCNN(nn.Module):
    """ A simple CNN """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(optimizer, model, train_loader, loss, epochs = 15):
    """ Train the model using the optimizer on dataset train_loader for |epochs| """
    
    # batch losses
    losses = []
    
    # batch accuracies
    accs = []
    
    # training for epochs
    for epoch in range(epochs):
        # going over the dataset...
        for batch_idx, (x, y) in enumerate(train_loader):
            # DFW can be used with standard pytorch syntax
            x, y = x.cuda(), y.cuda()
            out = model(x)
            acc = accuracy(out, y)
            loss_ = loss(out, y)

            optimizer.zero_grad()
            loss_.backward()

            # NB: DFW needs to have access to the current loss value,
            # (this syntax is compatible with standard pytorch optimizers too)
            optimizer.step(lambda: float(loss_))
            losses.append(loss_.item())
            accs.append(acc)
    return {'train_batch_losses': losses, 'train_batch_accs': accs}

def metrics_post(data_loader, model, loss, name = ''):
    """ Compute metrics for a single model after training """
    
    # all accuracies 
    accs = []
    
    # all losses (will take mean)
    losses = []
    
    # going over the dataset
    for batch_idx, (x, y) in enumerate(data_loader):
        # DFW can be used with standard pytorch syntax
        x, y = x.cuda(), y.cuda()
        
        # model output
        out = model(x)
        
        # accuracy
        acc = accuracy(out, y)
        
        # loss
        loss_ = loss(out, y)
        
        accs.append(acc.item())
        losses.append(loss_.item())
    return {name + 'accuracy': np.mean(accs), name + 'loss': np.mean(losses)}

def metrics_post_all(train_loader, test_loader, model, loss):
    m_train = metrics_post(train_loader, model, loss = loss, name = 'train_')
    m_test = metrics_post(test_loader, model, loss = loss, name = 'test_')
    m_train.update(m_test)
    return m_train
