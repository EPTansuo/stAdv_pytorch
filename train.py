import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms,datasets
from scipy import optimize
from net import Net,fit

def train(epoch=20, lr=0.01, batch_size=128):
    np.random.seed(42)
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = datasets.MNIST(root = './data', train=True, transform = transform, download=True)
    train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=True)

    use_cuda=True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model = Net().to(device)
    summary(model,(1,28,28))

    optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    train_loss,val_loss,train_acc,val_acc=fit(model,device,train_loader,val_loader,optimizer,criterion,epoch)

    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(1,51), train_loss, "*-",label="Training Loss")
    plt.plot(np.arange(1,51), val_loss,"o-",label="Val Loss")
    plt.xlabel("Num of epochs")
    plt.legend()
    plt.show()
    # plt.savefig('loss_event.png')

    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(1,51), train_acc, "*-",label="Training Acc")
    plt.plot(np.arange(1,51), val_acc,"o-",label="Val Acc")
    plt.xlabel("Num of epochs")
    plt.legend()
    plt.show()
    # plt.savefig('accuracy_event.png')

    torch.save(model.state_dict(),'./model.pt')


if __name__=='__main__':
    train()
