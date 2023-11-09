import numpy as np
from load import load_train, load_test
import torch
from torch.utils.data import TensorDataset, DataLoader

class Client(object):
    def __init__(self, trainDataSet, dev, num_example) -> None:
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.num_example = num_example
        self.state = {}

    def localUpdate(self, localBatchSize, localepoch, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localepoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data).squeeze() 
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                # import ipdb; ipdb.set_trace()
                loss = lossFun(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        return Net.state_dict()

class ClientsGroup(object):
    def __init__(self, dev, numOfClients):
        self.dev = dev
        self.clients_set = {}
        self.num_of_clients = numOfClients
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        train_feature, train_label = load_train()
        test_feature,  test_label  = load_test()

        train_data = torch.tensor(train_feature, dtype=torch.float)
        train_label = torch.tensor(train_label, dtype=torch.float)

        test_data = torch.tensor(test_feature, dtype=torch.float)
        test_label = torch.tensor(test_label, dtype=torch.float)

        # 3000 // 10 // 2 = 150
        shared_size = train_label.shape[0] // self.num_of_clients // 2
        # np.random.permutation 将序列进行随机排序
        # 20 个
        shared_id = np.random.permutation(train_label.shape[0] // shared_size)

        for i in range(self.num_of_clients):
            # 获取被分得的两块数据切片
            # 偶数
            shared_id1 = shared_id[i * 2 ]
            # 奇数
            shared_id2 = shared_id[i * 2 + 1]

            data_shared1 = train_data[shared_id1 * shared_size : (shared_id1 + 1) * shared_size]
            data_shared2 = train_data[shared_id2 * shared_size : (shared_id2 + 1) * shared_size]

            label_shared1 = train_label[shared_id1 * shared_size : (shared_id1 + 1) * shared_size]
            label_shared2 = train_label[shared_id2 * shared_size : (shared_id2 + 1) * shared_size]

            local_data, local_label = np.vstack((data_shared1, data_shared2)), \
                                      np.hstack((label_shared1, label_shared2))
            someone = Client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)),
                              self.dev, shared_size * 2)

            self.clients_set['client{}'.format(i)] = someone


if __name__ == '__main__':
    group = ClientsGroup(0, 10)