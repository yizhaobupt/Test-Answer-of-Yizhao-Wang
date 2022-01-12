"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, bag_length=10, num_bag=2, seed=1, train=True):  # 
        self.bag_length = bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000       # the number of picture in the MNIST training set
        self.num_in_test = 10000        # the number of picture in the MNIST testing set

        if self.train:
            self.train_bags_list,  self.value_list = self._create_bags()
        else:
            self.test_bags_list,  self.value_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            
            all_imgs = batch_data
            all_labels = batch_labels

        req_imgs = []
        req_labels = []
        idex = []
        for i in range(all_imgs.shape[0]):      # select the image of '0' and '7' out
            if all_labels[i] == 0 or all_labels[i] == 7:
                idex.append(i)

        req_imgs = all_imgs[idex]
        req_labels = all_labels[idex]
        data_len = len(idex)     
        bags_list = []
        value_list = []

        for i in range(self.num_bag):
            bag_length = self.bag_length
            if self.train:  # random choose pictures in the original set to construct training and test set
                indices = torch.LongTensor(self.r.randint(0, data_len, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, data_len, bag_length))

            labels_in_bag = req_labels[indices]
            purity = []
            for i in range(labels_in_bag.shape[0]): # calculate the number of '0' in the bag and give out the purity value
                if labels_in_bag[i] == 0:
                    purity.append(i)

            value_list.append(len(purity)/bag_length)
            bags_list.append(req_imgs[indices])
            

        return bags_list,  value_list

    def __len__(self):
        if self.train:
            return len(self.train_bags_list)
        else:
            return len(self.test_bags_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            
            purity = self.value_list[index]
        else:
            bag = self.test_bags_list[index]
            
            purity = self.value_list[index]

        return bag,  purity


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(bag_length=100,
                                                   num_bag=20,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(bag_length=100,
                                                  num_bag=20,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)


    len_bag_list_train = []
    mnist_bags_train = 0
    for _, (bag, purity) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
  



