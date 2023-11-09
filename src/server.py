import os
import argparse
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from clients import ClientsGroup
from model import tianqi_2NN
from load import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_clients", type=int, default=10)
    parser.add_argument("--num_comn", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--cfra", type=float, default=0.5)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--plot", action='store_true', default=False)

    args = parser.parse_args()

    net = tianqi_2NN()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(dev)

    loss_func = torch.nn.MSELoss(reduction='mean')
    opti = optim.SGD(net.parameters(), lr=args.learning_rate)
    explr = optim.lr_scheduler.ExponentialLR(opti, gamma=0.9)

    myClients = ClientsGroup(dev, args.num_of_clients)

    num_in_comm = int(args.num_of_clients)

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()


    # 载入测试集
    test_feature, test_label = load_test()
    testData = torch.tensor(test_feature, dtype=torch.float)
    testLabel = torch.tensor(test_label, dtype=torch.float)
    testDataLoader = DataLoader(TensorDataset(testData, testLabel), batch_size=512, shuffle=False)

    for i in range(1, args.num_comn+1):
        print('----------------------------', i, 'th training-----------------------------')

        order = np.random.permutation(num_in_comm)

        clients_in_comm = ['client{}'.format(i) for i in order[:num_in_comm]]

        sum_parameters = None

        for client in tqdm(clients_in_comm):

            local_parameters = myClients.clients_set[client].localUpdate(args.batchsize, args.epoch, net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        
        # 取平均值
        for var in sum_parameters:
            global_parameters[var] = sum_parameters[var] / num_in_comm

        # 测试
        net.load_state_dict(global_parameters, strict=True)

        sum_diff = 0
        for data, label in testDataLoader:
            data= data.to(dev)
            preds = net(data).squeeze().detach().cpu()
            diff = (preds - label).float()
            diff_mean = diff.mean()
            sum_diff += diff_mean

            if args.plot:
                from matplotlib import pyplot as plt
                index = range(label.size()[0])
                plt.scatter(index, label, label='label', marker='+')
                plt.scatter(index, preds, label='preds', marker="+")
                plt.scatter(index, diff/10, label='diff', marker="+")
                plt.legend()    
                plt.show()


        print('diff:', sum_diff.numpy())


if __name__ == '__main__':
    main()