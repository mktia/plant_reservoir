from datetime import datetime
from PIL import Image
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, Type
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()


class Block(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, padding: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)
        #
        # out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, class_size: int = 6):
        super(ResNet18, self).__init__()

        self.in_planes = 64

        # input 640x360x3
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # 320x180 to 64x36
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=5)

        self.layer1 = self._make_layers(block, 64, 2)
        self.layer2 = self._make_layers(block, 128, 2, stride=2)
        # self.layer3 = self._make_layers(block, 256, 2, stride=2)
        # self.layer4 = self._make_layers(block, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, class_size)

    def _make_layers(self, block, planes: int, blocks: int, stride=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet() -> ResNet18:
    return ResNet18(Block)


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def cv2pil(cv_image):
    return Image.fromarray(cv_image[:, :, ::-1])


def pil2cv(pil_image):
    return np.array(pil_image, dtype=np.uint8)[:, :, ::-1]


def load_data():
    videos = ['P2200783_M_L_1_max_pl.mp4',
              'P2200784_M_L_2_max_pl.mp4',
              'P2200785_M_L_3_max_pl.mp4',
              'P2200786_M_R_1_max_pl.mp4',
              'P2200787_M_R_2_max_pl.mp4',
              'P2200788_M_R_3_max_pl.mp4']

    valid_T = 1000
    T = 8000 - valid_T
    test_T = 500

    train_data = []
    valid_data = []
    test_data = []
    for class_id, capture_from in tqdm(enumerate(videos)):
        video_capture = cv2.VideoCapture(f'./video_in/{capture_from}')

        for i in range(T):
            retval, frame = video_capture.read()
            train_data.append((cv2pil(frame), class_id))
        else:
            for j in range(valid_T):
                retval, frame = video_capture.read()
                valid_data.append((frame, class_id))
            else:
                for k in range(test_T):
                    retval, frame = video_capture.read()
                    test_data.append((frame, class_id))

    return train_data, valid_data, test_data


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def validate(net, valid_loader):
    net.eval()
    correct = 0
    total = 0
    preds = torch.tensor([]).float().to(device)
    trues = torch.tensor([]).long().to(device)
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            preds = torch.cat((preds, outputs))
            trues = torch.cat((trues, labels))
        val_loss = criterion(preds, trues)
        err_rate = 1 - correct / total

    return val_loss, err_rate


def main():
    BATCH_SIZE = 128

    train, valid, test = load_data()
    train_transform = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: np.array(img)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.float())
    ])
    train_dataset = ImageDataset(train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: np.array(img)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.float())
    ])
    valid_dataset = ImageDataset(valid, transform=val_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataset = ImageDataset(test, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # test to import images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # print(images.numpy().shape)
    # imshow(torchvision.utils.make_grid(images))

    # training
    net = _resnet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    log = {'train_loss': [], 'valid_loss': [], 'train_err': [], 'valid_err': []}

    EPOCH = 10

    for epoch in tqdm(range(EPOCH)):
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            train_loss, train_err = validate(net, train_loader)
            valid_loss, valid_err = validate(net, valid_loader)
            log['train_loss'].append(train_loss.item())
            log['valid_loss'].append(valid_loss.item())
            log['train_err'].append(train_err)
            log['valid_err'].append(valid_err)
            print(f'train_err:\t{train_err:.1f}')
            print(f'valid_err:\t{valid_err:.1f}')
            scheduler.step(valid_loss)
    else:
        test_loss, test_err = validate(net, test_loader)
        with open(f'./data_out/{datetime.today().isoformat().replace(":", "-").split(".")[0]}.txt', 'w') as f:
            f.write(f'Accuracy: {1 - test_err}, Cross Entropy Loss: {test_loss}')
        print('Finished Training')

    figs = [plt.figure(constrained_layout=True) for i in range(2)]

    ax1 = figs[0].add_subplot()
    ax1.plot(np.arange(EPOCH) + 1, log['train_loss'], label='train_loss')
    ax1.plot(np.arange(EPOCH) + 1, log['valid_loss'], label='valid_loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2 = figs[1].add_subplot()
    ax2.plot(np.arange(EPOCH) + 1, log['train_err'], label='train_err')
    ax2.plot(np.arange(EPOCH) + 1, log['valid_err'], label='valid_err')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('error rate')
    ax2.set_title('Error rate')
    ax2.legend()

    for i in range(2):
        figs[i].savefig(f'./image_out/{datetime.today().isoformat().replace(":", "-").split(".")[0]}_nn_{i}.eps',
                        format='eps')


if __name__ == '__main__':
    main()
