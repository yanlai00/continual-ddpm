import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from datasets import MNIST
import torch.optim as optim
import torch.nn as nn
import os

save_path = '/home/yy2694/continual-ddpm/classifiers/'
os.makedirs(save_path, exist_ok=True)
device = 'cuda'

batch_size = 64
trainset = MNIST(299)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
model = torchvision.models.inception_v3(num_classes=10, pretrained=False, aux_logits=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if inputs.shape[1] == 1:
            inputs = inputs.repeat((1, 3, 1, 1))
        assert inputs.shape[1] == 3
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    torch.save(model.state_dict(), os.path.join(save_path, f'canonical_mnist_{epoch}.pth'))
