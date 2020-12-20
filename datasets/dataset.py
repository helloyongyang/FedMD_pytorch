import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm


class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform):
        super(Cifar10Dataset, self).__init__()

        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.data)



def gen_balance_client_data(prefix):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    # prefix = "cifar-10-batches-py/"
    all_data = []
    all_label = []


    for data_batch_name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]:
        path = prefix + data_batch_name
        with open(path, 'rb') as infile:
            entry = pickle.load(infile, encoding='latin1')

            all_data.append(entry['data'])
            all_label.extend(entry['labels'])


    all_data = np.vstack(all_data).reshape(-1, 3, 32, 32)
    all_data = all_data.transpose((0, 2, 3, 1))  # convert to HWC

    # print(all_data.shape)
    # print(len(all_label))

    all_label_np = np.array(all_label)
    # print(all_label_np)


    server_num = 1000
    client_num = [2000, 2000]

    server_data = []
    server_label = []

    client_data = {'0': [], '1': []}
    client_label = {'0': [], '1': []}

    cs_data = {'0': [], '1': []}
    cs_label = {'0': [], '1': []}


    data_cls = []
    labels_cls = []

    for cls_label in range(10):
        idx = np.where(all_label_np == cls_label)[0]
        # print(len(idx))

        data_cls.append(all_data[idx])
        labels_cls.append(all_label_np[idx])


    for i in range(10):
        # print(data_cls[i].shape)
        # print(labels_cls[i].shape)

        server_data.append(data_cls[i][0:server_num])
        server_label.append(labels_cls[i][0:server_num])

        point = server_num
        for j in range(len(client_num)):
            client_data[str(j)].append(data_cls[i][point: point + client_num[j]])
            client_label[str(j)].append(labels_cls[i][point: point + client_num[j]])
            point += client_num[j]


    server_data = np.vstack(server_data)
    server_label = np.hstack(server_label)
    print("server_data.shape : ", server_data.shape)
    print("server_label.shape : ", server_label.shape)

    server_dataset = Cifar10Dataset(server_data, server_label, transform)
    client_dataset = []
    cs_dataset = []


    for j in range(len(client_num)):
        client_data[str(j)] = np.vstack(client_data[str(j)])
        client_label[str(j)] = np.hstack(client_label[str(j)])
        print("client_data " + str(j) + " shape : ", client_data[str(j)].shape)
        print("client_data " + str(j) + " shape : ", client_label[str(j)].shape)

        client_dataset.append(Cifar10Dataset(client_data[str(j)], client_label[str(j)], transform))

        cs_data[str(j)] = [server_data, client_data[str(j)]]
        cs_label[str(j)] = [server_label, client_label[str(j)]]
        cs_data[str(j)] = np.vstack(cs_data[str(j)])
        cs_label[str(j)] = np.hstack(cs_label[str(j)])
        print("cs_data " + str(j) + " shape : ", cs_data[str(j)].shape)
        print("cs_label " + str(j) + " shape : ", cs_label[str(j)].shape)

        cs_dataset.append(Cifar10Dataset(cs_data[str(j)], cs_label[str(j)], transform))

        
    


    # test data
    print("***")
    path = prefix + "test_batch"
    with open(path, 'rb') as infile:
        entry = pickle.load(infile, encoding='latin1')
        test_data = entry['data']
        test_label = entry['labels']

    test_data = test_data.reshape(-1, 3, 32, 32)
    test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

    test_label = np.array(test_label)

    print("test_data.shape : ", test_data.shape)
    print("test_label.shape : ", test_label.shape)

    test_dataset = Cifar10Dataset(test_data, test_label, transform)

    return server_dataset, client_dataset, cs_dataset, test_dataset


def val(test_loader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test dataset: %.4f' % (1.0 * correct / total))


