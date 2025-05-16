from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_set import WEIGHT_DATA_SET
from data import DATA, DATA_defined_prob, DATA_policy_gradient, DATA_fullaction, DATA_action, DATA_partial_logistic, DATA_partial_logistic_deep, DATA_partial_random, DATA_partial_action, DATA_learn_policy, Drfirst_data
from torchvision import transforms
import torchvision
import os
import copy
import torch.utils.data as data
from data_mnist import DATA_mnist
from data_cifar import DATA_CIFAR10 
import regression_utility as ru
import abstain_utility as au
from scipy import stats
import math
from scipy.stats import dirichlet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random
torch.set_default_tensor_type('torch.DoubleTensor')
mean0 = 0.6
var0 = 1
d = 1

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.D_in, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.D_out)),
            # torch.nn.Sigmoid(),
            )

    def forward(self, x):

        x = x.view(-1, self.D_in)
        x = self.model(x.double())
        return x


class PolicyModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(PolicyModel, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.H, self.H),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.H, self.H),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H, self.D_out),
            )

    def forward(self, x):

        x = x.view(-1, self.D_in)
        x = self.model(x.double())
        return x

def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p


def train_regression(args, model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0):
     
    model.train()
    lowerB = -1.0/(2*var0)
   
    grad_yy = np.empty([0])
    # change this if d changes
    grad_yx = np.empty([2, 0])

    lr2 = 1
    lr2 = lr2 * (10 / (10 + np.sqrt(epoch)))

    lr1 = 1
    lr1 = lr1 * (10 / (10 + np.sqrt(epoch)))
    grad_squ_1 = 0.00001
    grad_squ_2 = 0.00001
    for batch_idx, (data, target, weight) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
      
        optimizer.zero_grad()
        output = model(data)


        meanY, varY = ru.predict_regression(weight, Myy, Myx, output, mean0, var0)
        # print(varY)
        grad = ru.M_gradient(output, meanY, varY, target, Myy, Myx)

        grad_squ_1 = grad_squ_1 + grad[0]**2
        grad_squ_2 = grad_squ_2 + grad[1:]**2
        
        
        diff = lr1*(grad[0]) + 0.00000*Myy
        grad_yy = np.concatenate((grad_yy, grad[0]))
        grad_yx = np.concatenate((grad_yx, grad[1:]), 1)
        preM = Myy
        Myy = preM + lr1*(grad[0]/np.sqrt(grad_squ_1)) + 0.00000*Myy

        while Myy[0][0] < lowerB:
            Myy = Myy + np.abs(diff)/2

        Myx = Myx + lr2 *(grad[1:]/np.sqrt(grad_squ_2)) + 0.00000*Myx

        bs = np.shape(output)[0]
      
        output_last = ru.regression_gradient.apply(output, torch.tensor(Myx[0:-1]), torch.reshape(target, (bs, 1)), torch.reshape(meanY, (bs, 1)))
        output_last.backward(torch.ones(output_last.shape),retain_graph=True)
        

        optimizer.step()
    return Myy, Myx


def train_MSE(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for _ in range(epoch):
        total_loss = 0
        for batch_idx, (data, target, weight) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            criterion = nn.MSELoss()
            # loss = args.mse_weight*criterion(output[target==1], target[target==1]) + (1-args.mse_weight)* criterion(output[target==0], target[target==0])
            loss = criterion(output, target)
            loss.backward()
            total_loss +=loss.detach()
            optimizer.step()
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    return model


def test_regression(args, model, Myy, Myx, device, test_loader, mean0, var0):
    model.eval()
    test_loss = 0
    y_prediction = np.empty([1, 0])
    y_var = np.empty([1, 0])
    with torch.no_grad():
        for data, target, weight in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            d = np.shape(data)[0]
            target = torch.reshape(target, (1, d))
            
            meanY, varY = ru.predict_regression(weight, Myy, Myx, output, mean0, var0)
            loss =  -np.log(1/(np.sqrt(varY)*np.sqrt(2*3.14)))+(target-meanY).pow(2)/(2*varY)
            criterion = nn.MSELoss()
            l2loss = criterion(meanY, target)
            test_loss += torch.sum(l2loss)
            y_prediction = np.concatenate((y_prediction, meanY), axis=1)
            y_var = np.concatenate((y_var, varY), axis = 1)
    test_loss /= len(test_loader.dataset)

    return y_prediction, y_var, test_loss



def train_validate_test(args, epoch, loss_type, device, use_cuda, train_model, train_loader, test_loader, validate_loader, n_class, lbd, testflag = True):
    
    if loss_type == 'regression':

        Myy = np.ones((1, 1))
        Myx = np.ones((d+1, 1))
        optimizer = optim.SGD(train_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=lbd)

        for epoch in range(1, epoch + 1):
            Myy, Myx = train_regression(args, train_model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0) 
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, validate_loader, mean0, var0)
        
        if testflag == True:
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, test_loader, mean0, var0)
       
        return train_model, Myy, Myx, meanY, varY, loss
    
def round_value(a):
    if a>1:
        a = 1.0
    elif a<0:
        a = 0
    return a


def sample_action(prob, n_class):
    # sample action
    rand = np.random.uniform(0, 1, 1)
    action = 0
    threshold = 0
    
    for j in range(n_class):
    
        threshold = threshold + prob[j]
        
        if rand[0] < threshold:
            action = j
            break
    return action
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
         
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Covariate Shift')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs-training', type=int, default=40, metavar='N',
                        help='number of epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_pg', type=float, default=0.0003, metavar='LR',
                        help='learning rate for policy gradient (default: 0.001)')
    parser.add_argument('--filename', type=str, default='trainingdata.csv',
                        help='the file that contains training data')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mode', type=int, default=2, metavar='N',
                        help='1, policy is a uniform default policy, 2 is a policy is a known policy trained from biased samples, 3 policy is an unknown policy uniform policy, 4 is an unknown biased policy ')
    parser.add_argument('--dataset', type=int, default=6, metavar='N',
                        help='1, uci, 2 mnist, 3 cifar10')
    parser.add_argument('--policy', type=int, default=0, metavar='N',
                        help='0, policy is a uniform default policy, 1 is a policy small shift policy, 2 policy is large policy, 3 is d uniform policy, 4 is tweak')
    parser.add_argument('--alpha', type=float, default=10, metavar='N',
                        help='direclect shift parameter')
    parser.add_argument('--rou', type=float, default=0.91, metavar='N',
                        help='tweak one shift parameter')
    parser.add_argument('--clip_weight', type=bool, default=False, metavar='N',
                        help='whether clip grad for robust regression')
    parser.add_argument('--epochs_policy_gradient', type=int, default=50, metavar='N',
                        help='epochs for policy gradient')
    parser.add_argument('--save-file-name', type=str, default='', metavar='N',
                        help='save regret filename')
    parser.add_argument('--lr-decay', type=float, default=0.9, metavar='N', help='dedcay rate of lr')
    parser.add_argument('--mse_weight', type=float, default=0.5, metavar='N', help='dedcay rate of lr')

    parser.add_argument('--simulation', type=int, default=1, metavar='N', help='interaction-num')
    parser.add_argument('--evaluate-batch', type=int, default=100, metavar='N', help='interaction-num')
    parser.add_argument('--train-mse', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='if use ips gradient')
    parser.add_argument('--set-seed', action='store_true', default=False)



    args = parser.parse_args()

    for simulation in range(args.simulation):
        if args.set_seed:
            setup_seed(args.seed + simulation)
        evaluate_flag = 0

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        if args.dataset == 3:

            all_data = DATA_CIFAR10(transform=transforms.Compose([
                            # transforms.RandomCrop(32, padding=4),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
            n_class = 10
            n_dim = 32*32*3
        elif args.dataset == 2:
            all_data = DATA_mnist(args.filename)
            n_class = 10
            n_dim = 28*28
        elif args.dataset == 1:
            all_data = DATA(args.filename)
            n_dim = 64
            n_class = 10
        elif args.dataset == 4:
            train_data = DATA('data/covertype/train.csv')
            test_data = DATA('data/covertype/test.csv')
            m_train = len(train_data)
            m_test = len(test_data)


            rand_idx = np.random.permutation(m_train)
            train_data = data.Subset(train_data, rand_idx)

            n_dim = 54
            n_class = 7
        elif args.dataset == 5:
            all_data = DATA('adult_processed.csv')
            n_dim = 92
            n_class = 14
        
        elif args.dataset == 6:


            all_x = np.load('training_x.npy')
            all_a = np.load('training_a.npy')
            all_r = np.load('training_y.npy')

            all_a = (all_a + 1)%24
    

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.fit(all_x)
            all_x = scaler.transform(all_x)

            index = np.random.permutation(len(all_x))

            training_number = 150000
            testing_number = 50000

            training_index = index[:training_number]
            testing_index = index[training_number:training_number + testing_number]
            
            n_dim = all_x.shape[1]
            n_class = 24

            x = all_x[training_index]
            a = all_a[training_index]
            r = all_r[training_index]


            eval_x = all_x[testing_index]
            eval_a = all_a[testing_index]
            eval_r = all_r[testing_index]
            
            n_dim = all_x.shape[1]
            n_class = 24
            
            print('before training sum:',np.sum(r))



        logging_policy = torch.zeros(n_class)
        for aa in a:
            logging_policy[aa] += 1
        logging_policy = logging_policy/training_number
        print(logging_policy)
        

        n_nodes = 256
        policy_default = 1.0/n_class

        # if args.dataset != 4:
        #     data_size = len(all_data)
        #     rand_idx = np.random.permutation(data_size)
        #     training_ratio = 0.6
        #     m_train = int(training_ratio*data_size)
        #     train_data = data.Subset(all_data, rand_idx[0: m_train])
        #     test_data = data.Subset(all_data, rand_idx[m_train: -1])
        #     m_test = data_size - m_train

#         weight_st = np.ones(m_train)

#         weighted_train = WEIGHT_DATA_SET(train_data, weight_st,args)


#         weight_st = np.ones(m_test)
#         weighted_test = WEIGHT_DATA_SET(test_data, weight_st,args)

        # test_loader = data.DataLoader(weighted_test,
        #     batch_size=args.batch_size, shuffle=False, **kwargs)


        train_data_mse = Drfirst_data(x,a,r)

        if args.mode == 1 or args.mode == 3:
            # train robust model
            # save models
            model_robust_list = []
            Myy_robust_list = []
            Myx_robust_list = []
            train_data_robust = train_data_mse
            print(len(train_data_robust))
            for i in range(n_class):
                ## generate training set for action i
                train_action_data = DATA_partial_action(train_data_robust, i)
                # train the regression model for predicting rewards
                train_size = len(train_action_data)
                weight_st = np.ones(train_size)

                weighted_train = WEIGHT_DATA_SET(train_action_data, weight_st,args)

                train_model = Net(n_dim,n_nodes, 1)
                train_model = train_model.to(device)

                validate_size = int(0.1*train_size)
                validate_size = 1 if validate_size else validate_size

                try:
                    train_loader = data.DataLoader(data.Subset(weighted_train, range(validate_size, train_size)),batch_size=args.batch_size, shuffle=True, **kwargs)
                    validate_loader = data.DataLoader(data.Subset(weighted_train, range(0, validate_size)),batch_size=args.batch_size, shuffle=True, **kwargs)
                    #
                    train_model, Myy, Myx, _, _, _ = train_validate_test(args, args.epochs_training, "regression", device, use_cuda, train_model,
                                                                         train_loader, test_loader , validate_loader, n_class, 0.000, testflag = False)
                    if not args.train_mse:
                        train_model, Myy, Myx, _, _, _ = train_validate_test(args, args.epochs_training, "regression",
                                                                             device, use_cuda, train_model,
                                                                             train_loader, test_loader, validate_loader,
                                                                             n_class, 0.000, testflag=False)

                    else:
                        optimizer1 = optim.Adam(train_model.parameters(), lr=args.lr)
                        train_model = train_MSE(args, train_model, device, train_loader, optimizer1,
                                                args.epochs_training)

                        total_loss = 0
                        for batch_idx0, (data0, target0, weight0) in enumerate(validate_loader):

                            output = train_model(data0)
                            criterion = nn.MSELoss()
                            loss = criterion(output, target0)
                            total_loss += loss.detach()

                except:
                    Myy = np.ones((1, 1))
                    Myx = np.ones((d + 1, 1))
                model_robust_list.append(train_model)

                if not args.train_mse:
                    Myy_robust_list.append(Myy)
                    Myx_robust_list.append(Myx)

        test_partial_data = train_data_mse
        # policy using policy model
        policy_learning_model = Net(n_dim,n_nodes, n_class)
        policy_learning_model = policy_learning_model.to(device)

        optimizer = optim.Adam(policy_learning_model.parameters(), lr=args.lr_pg, weight_decay=0.1)

        count = 0
        prob_action_pre = (1.0/n_class) *torch.ones((len(test_partial_data), n_class), dtype = torch.float64)
        action_pre = torch.ones(len(test_partial_data), dtype = torch.int64)
        # regret_list = np.zeros([args.epochs_policy_gradient,int(len(test_partial_data)/args.batch_size)+ 1])
        # test_regret_list = np.zeros([args.epochs_policy_gradient])
        # loss_evaluate_result = []

        best_epoch = 1
        best_loss = 10000

        for epoch in range(1, args.epochs_policy_gradient + 1):
            regret = 0.0
            reward_training = torch.ones(len(test_partial_data), dtype = torch.float64)
            # features_training = test_partial_data.get_features()
            # action_training = test_partial_data.get_action()
            # action_true = test_partial_data.get_action_true()
            reward_true = []
            total_loss = 0
            total_reward = 0
            total_reward_estimated = 0
            with torch.no_grad():
                for i in range(len(test_partial_data)):
                    features, action, reward = test_partial_data[i]

                    if args.mode is 2:
                        action_policy = action
                        action_pre[i] = action_policy
                        policy =  prob_action_pre[i][action_policy]/logging_policy[action_policy]

                        policy = np.clip(policy,0.0001,1000)
                        reward_training[i] = policy

#                     if args.mode is 3:

#                         if action_policy == action:
#                             reward_training[i] = (1.0/policy)*(reward - meanY_robust) + meanY_robust
#                         else:
#                             reward_training[i] = meanY_robust
            reward_training = reward_training * torch.tensor(r)/ torch.mean(reward_training)
            print('training sum:',torch.sum(reward_training))
            policy_gradient_data = DATA_policy_gradient(x, reward_training, action_pre, action_pre)
            
            # construct dataloader
            train_loader = data.DataLoader(policy_gradient_data,batch_size=args.batch_size, shuffle=True, **kwargs)
            optimizer.zero_grad()
  
            policy_learning_model.train()
            prob_action_next = torch.empty([0, n_class])
            # for p in policy_learning_model.parameters():
            #     if p.requires_grad:
            #         print(p.name, p.data)
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):
                evaluate_flag += 1
                optimizer.zero_grad()
                bsize = len(feature_idx)
                output_learning = policy_learning_model(feature_idx)
                
                # print(output_learning)
                prob = my_softmax(output_learning)

                pi_mean = output_learning.max(1, keepdim=True)[1]

                action_onehot = torch.DoubleTensor(bsize,n_class)
                action_onehot.zero_()
                action_onehot.scatter_(1, action_idx.reshape(bsize, 1), 1)

                grad_policy =  ru.policy_gradient.apply(output_learning, prob, reward_idx, action_onehot)
                grad_policy.backward(torch.ones(grad_policy.shape),retain_graph=True)
                optimizer.step()

                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)

                # correct = pi_mean.eq(action_true_idx.view_as(pi_mean)).sum().item()
                # regret = float((bsize - correct)/bsize)
                # regret_list[epoch-1][batch_idx] = regret
#                 if evaluate_flag == args.evaluate_batch:
#                     evaluate_flag = 0
#                     with torch.no_grad():
#                         for feature, target, weight, in test_loader:
#                             feature, target, weight = feature.to(device), target.to(device), weight.to(device)

#                             output = policy_learning_model(feature)
#                             y_prediction = output.max(1, keepdim=True)[1]

#                             correct += y_prediction.eq(target.view_as(y_prediction)).sum().item()
#                     test_loss = 1 - float(correct) / len(test_loader.dataset)
#                     loss_evaluate_result.append(test_loss)
            
    
            
            prob_action_next = torch.empty([0, n_class])
            train_loader = data.DataLoader(policy_gradient_data,batch_size=args.batch_size*8, shuffle=False, **kwargs)
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):
                output_learning = policy_learning_model(feature_idx)
                prob = my_softmax(output_learning)
                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)
            prob_action_pre = torch.tensor(prob_action_next, dtype = torch.float64)
            prob_action_pre = prob_action_pre.reshape((len(test_partial_data), n_class))
            # print(prob_action_pre)
            torch.save(policy_learning_model.state_dict(), 'model/'+str(epoch)+'_.pth')

            
            policy_gradient_data_eval = DATA_policy_gradient(eval_x, reward_training, eval_a, eval_a)
            
            prob_action_next = torch.empty([0, n_class])
            eval_loader = data.DataLoader(policy_gradient_data_eval,batch_size=args.batch_size*8, shuffle=False, **kwargs)
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(eval_loader):
                output_learning = policy_learning_model(feature_idx)
                prob = my_softmax(output_learning)
                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)
            prob_action_pre_eval = torch.tensor(prob_action_next, dtype = torch.float64)
            prob_action_pre_eval = prob_action_pre_eval.reshape((len(policy_gradient_data_eval), n_class))
            
            reward_eval = 0
            eval_policy = []
            for i in range(len(prob_action_pre_eval)):
                    policy = prob_action_pre_eval[i][eval_a[i]]/logging_policy[eval_a[i]]
                    policy = np.clip(policy, 0.0001,10000)
                    # print(policy,reward_eval)
                    eval_policy.append(policy)
                    reward_eval += policy*eval_r[i]

            print(reward_eval, np.mean(np.array(eval_policy)*eval_r)/np.mean(eval_policy), np.sum(eval_r))


            for p in optimizer.param_groups:
                p['lr'] *= args.lr_decay

if __name__ == '__main__':
    main()

        


 
