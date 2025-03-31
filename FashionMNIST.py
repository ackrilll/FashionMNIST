import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import multiprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#모델 정의
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = NeuralNet()
params = list(net.parameters())
print('모델: ',net)
print('파라미터 텐서 개수: ',len(params))
print('conv1 텐서 크기: ',params[0].size())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    multiprocessing.freeze_support() # 윈도우 환경에서 필요

    #GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('crrent device: ',device)

    #데이터 전처리
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5),(0.5))])

    #데이터 셋 다운로드
    trainset = datasets.FashionMNIST(root='C:\\Users\\skawl\\PycharmProjects\\FashionMNIST',
                                    train=True, download=True,
                                    transform=transform)

    testset = datasets.FashionMNIST(root='C:\\Users\\skawl\\PycharmProjects\\FashionMNIST',
                                   train=False, download=True,
                                   transform=transform)

    #데이터 로더 정의
    train_loader = DataLoader(trainset,batch_size=128, shuffle = True,num_workers=0)
    test_loader = DataLoader(testset,batch_size=128, shuffle = True,num_workers=0)


    images, labels = next(iter(train_loader))
    print('image shape: ', images.shape)
    print('label shape: ', labels.shape)

    #라벨링( FashionMNIST 데이터 셋 라벨에 있는 숫자를 String 형태로 매핑해주기위한 사전 작업 )
    labels_map = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot'
     }

    figure = plt.figure(figsize=(12, 12))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        image = images[i].squeeze()
        label_idx = labels[i].item()
        label = labels_map[label_idx]
        figure.add_subplot(rows,cols,i)
        plt.title(label)
        plt.axis('off')
        plt.imshow(image,cmap = 'gray')

    plt.show()