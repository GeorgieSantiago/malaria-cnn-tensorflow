import torch as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from train_data import TrainData
from cnn import CNN
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from prepare import Prep
def show(img):
    img = img.cpu().numpy()[0]
    # convert image back to Height,Width,Channels
    img = np.transpose(img, (1, 2, 0))
    # show the image
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    try:
        p = Prep()
        p.generate_csv_from_flat()
    except Exception as e:
        print("Skip generating CSV file")
    batch_size = 64
    n_epochs = 4
    train_data: Dataset = TrainData("flat_data.csv")
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    net = CNN(1e-3, input_params=int(2*7*7), output_params=2, fc1_dims=int(142 * 7))
    loss_history = []
    for epoch in range(n_epochs):
        running_loss = 0
        correct = 0
        for idx, data in enumerate(loader):
            inputs, labels = data
            labels = labels.to(net.device)
            if idx % 10 == 0 and idx != 0:
                show(inputs)
            net.zero_grad()
            outputs = net.forward(inputs)
            outputs = T.max(outputs)
            labels = labels.unsqueeze(0)
            print(outputs, labels)
            loss = net.loss(outputs, T.max(labels))
            loss.backward()
            running_loss += loss.item()
#            correct += pred.eq(labels.view_as(pred)).sum().item()

            net.optimizer.step()
            print(correct, "Current Correct")
        loss_history.append(running_loss)
    print(loss_history)
    print(epoch)
    T.save(net.state_dict(), "malaria_cell_model.pt")
