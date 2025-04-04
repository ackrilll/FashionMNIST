import numpy as np
import torch
import torch.nn as nn
from Models.models import NeuralNet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import multiprocessing
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    multiprocessing.freeze_support() # 윈도우 환경에서 필요

    #GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = NeuralNet().to(device)
    print('crrent device: ',device)

    #데이터 전처리
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5),(0.5))])

    #데이터 셋 다운로드
    trainset = datasets.FashionMNIST(root='.\\FashionMNIST',
                                    train=True, download=True,
                                    transform=transform)

    testset = datasets.FashionMNIST(root='.\\FashionMNIST',
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

    #훈련 데이터 꺼내서 확인해보기
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

    #손실함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    '''
    # 모델 훈련
    losses = []
    epochs = []
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(' Epoch: {}, Iter: {}, Loss: {}'.format(
                    epoch + 1, i + 1, running_loss))
                losses.append(running_loss)
                epochs.append(epoch)
                running_loss = 0.0
    
    # 로스 그래프
    plt.plot(epochs,losses)
    plt.title('loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    #모델 저장
    PATH = '.\\Parameters\\fashion_mnist2.pth'
    torch.save(net.state_dict(),PATH)
    '''
    #모델 로드
    PATH = '.\\Parameters\\fashion_mnist2.pth'
    net = NeuralNet().to(device)
    net.load_state_dict(torch.load(PATH))

    print('로드된 모델: ', net.parameters)


    # 모델 테스트
    def imshow(image):
        image = image / 2 + 0.5
        npimg = image.numpy()
        fig = plt.figure(figsize=(16,8))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(test_loader)
    images, label = dataiter.__next__()
    images = images.to(device)  # 입력 데이터 GPU 이동
    imshow(torchvision.utils.make_grid((images[:6].cpu())))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # 6개 이미지와 예측 결과 표시
    figure = plt.figure(figsize=(16, 8))
    for i in range(6):
        ax = figure.add_subplot(2, 3, i + 1)
        image = images[i].squeeze().cpu()
        true_label = labels_map[label[i].item()]
        pred_label = labels_map[predicted[i].item()]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {true_label}\nPredicted: {pred_label}')
        ax.axis('off')
    plt.show()

    # 정확도 계산
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('정답률: %d %%' % (100 * correct / total))
