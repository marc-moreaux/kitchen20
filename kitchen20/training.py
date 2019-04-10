from envnet import EnvNet
from kitchen20 import Kitchen20
from torch.utils.data import DataLoader
import torch.nn as nn
import utils as U
import torch


# Model
model = EnvNet(20, True)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


# Dataset
batchSize = 32
inputLength = 48000
transforms = []
transforms += [U.random_scale(1.25)]  # Strong augment
transforms += [U.padding(inputLength // 2)]  # Padding
transforms += [U.random_crop(inputLength)]  # Random crop
transforms += [U.normalize(float(2 ** 16 / 2))]  # 16 bit signed
transforms += [U.random_flip()]  # Random +-

trainData = Kitchen20(root='../',
                      transforms=transforms,
                      folds=[1,2,3,4],
                      overwrite=False,
                      audio_rate=44100,
                      use_bc_learning=False)
trainIter = DataLoader(trainData, batch_size=batchSize,
                       shuffle=True, num_workers=2)


inputLength = 64000
transforms = []
transforms += [U.padding(inputLength // 2)]  # Padding
transforms += [U.random_crop(inputLength)]  # Random crop
transforms += [U.normalize(float(2 ** 16 / 2))]  # 16 bit signed
transforms += [U.random_flip()]  # Random +-

valData = Kitchen20(root='../',
                    transforms=transforms,
                    folds=[5,],
                    audio_rate=44100,
                    overwrite=False,
                    use_bc_learning=False)
valIter = DataLoader(valData, batch_size=batchSize,
                     shuffle=True, num_workers=2)


for epoch in range(100):
    tAcc = tLoss = 0
    vAcc = vLoss = 0
    for x, y in trainIter:  # Train epoch
        model.train()
        x = x[:, None, None, :]
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        y_pred = y_pred[:, :, 0, 0]

        # Compute and print loss
        loss = criterion(y_pred, y.long())
        acc = (y_pred.argmax(dim=1).long() == y.long()).sum()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tLoss += loss.item()
        tAcc += acc.item()

    for x, y in valIter:  # Test epoch
        model.eval()
        x = x[:, None, None, :]
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        y_pred = y_pred[:, :, 0, 0]
        loss = criterion(y_pred, y.long())
        acc = (y_pred.argmax(dim=1).long() == y.long()).sum()
        vLoss += loss.item()
        vAcc += acc.item()


    # loss = loss / len(dataset)
    # acc = acc / float(len(dataset))
    print('epoch {} -- train: {}/{} -- val:{}/{}'.format(
        epoch, tAcc, tLoss, vAcc, vLoss))

