from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_set import WEIGHT_DATA_SET,WEIGHT_DATA_SET_DRFIRST
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
mean0 = 0.1
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
            )

    def forward(self, x):
    
        x = x.view(-1, self.D_in)
        x = self.model(x.double())
        return x


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

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
        output_last.backward(torch.ones(output_last.shape),retain_graph=False)
        

        optimizer.step()
    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader)))
    # # print(np.shape(grad_yy))
    # # print(np.shape(grad_yx))
    # print('gradient:', np.linalg.norm(grad_yy))
    # print('gradient:', np.linalg.norm(grad_yx))

    return Myy, Myx

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
    # print('Average loss: {:.4f}\n'.format(test_loss))
    # print(target)
    # print(meanY)
    return y_prediction, y_var, test_loss


def train_validate_test(args, epoch, loss_type, device, use_cuda, train_model, train_loader, test_loader, validate_loader, n_class, lbd, testflag = True):
    
    if loss_type == 'regression':

        Myy = np.ones((1, 1))
        Myx = np.ones((d+1, 1))
        optimizer = optim.Adam(train_model.parameters(),lr=args.lr)

        for epoch in range(1, epoch + 1):
            Myy, Myx = train_regression(args, train_model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0) 
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, validate_loader, mean0, var0)
            
        # print('\nTesting on test set')
        
        if testflag == True:
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, test_loader, mean0, var0)
       
        return train_model, Myy, Myx, meanY, varY, loss
    
       
        return train_model, pred_Y, loss

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
def sample_action_batch(p, n=1, items=None):
    s = p.cumsum(axis=1)
    r = np.random.rand(p.shape[0], n, 1)
    q = np.expand_dims(s, 1) >= r
    k = q.argmax(axis=-1)
    if items is not None:
        k = np.asarray(items)[k]
    k = k.reshape(len(k))
    return k
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
         
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Covariate Shift')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs-training', type=int, default=20 , metavar='N',
                        help='number of epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_pg', type=float, default=0.001, metavar='LR',
                        help='learning rate for policy gradient (default: 0.001)')
    parser.add_argument('--filename', type=str, default='trainingdata.csv',
                        help='the file that contains training data')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mode', type=int, default=1, metavar='N',
                        help='1, policy is a uniform default policy, 2 is a policy is a known policy trained from biased samples, 3 policy is an unknown policy uniform policy, 4 is an unknown biased policy ')
    parser.add_argument('--dataset', type=int, default=6, metavar='N',
                        help='1, uci, 2 mnist, 3 cifar10')
    parser.add_argument('--policy', type=int, default=0, metavar='N',
                        help='0, policy is a uniform default policy, 1 is a policy small shift policy, 2 policy is large policy, 3 is d uniform policy, 4 is tweak')
    parser.add_argument('--alpha', type=float, default=10, metavar='N',
                        help='direclect shift parameter')
    parser.add_argument('--rou', type=float, default=0.91, metavar='N',
                        help='tweak one shift parameter')
    parser.add_argument('--clip_weight', type=bool, default=True, metavar='N',
                        help='whether clip grad for robust regression')
    parser.add_argument('--weights_upper_bound', type=float, default=100, metavar='N',
                        help='upper bound for weight clip')
    parser.add_argument('--weights_lower_bound', type=float, default=0.001, metavar='N',
                        help='lower bound for weight clip')
    parser.add_argument('--epochs_policy_gradient', type=int, default=20, metavar='N',
                        help='epochs for policy gradient')
    parser.add_argument('--save-file-name', type=str, default='', metavar='N',
                        help='save regret filename')
    parser.add_argument('--lr-decay', type=float, default=0.9, metavar='N', help='dedcay rate of lr')
    parser.add_argument('--simulation', type=int, default=1, metavar='N', help='interaction-num')
    parser.add_argument('--evaluate-batch', type=int, default=100, metavar='N', help='interaction-num')
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
            n_dim = 54
            n_class = 7
        elif args.dataset == 5:
            all_data = DATA('adult_processed.csv')
            n_dim = 92
        elif args.dataset == 6:


            all_x = np.load('training_x.npy')
            all_a = np.load('training_a_interval.npy')
            all_r = np.load('training_y.npy')

            # all_a = (all_a + 1)%24
    

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
            n_class = max(all_a) + 1

            x = all_x[training_index]
            a = all_a[training_index]
            r = all_r[training_index]


            eval_x = all_x[testing_index]
            eval_a = all_a[testing_index]
            eval_r = all_r[testing_index]
            
            n_dim = all_x.shape[1]
            # n_class = 24
            
            print('before training sum:',np.sum(r))



        n_nodes = 256


        logging_policy = torch.zeros(n_class)
        for aa in a:
            logging_policy[aa] += 1
        logging_policy = logging_policy/training_number
        print(logging_policy)


  
        train_data_mse = Drfirst_data(x,a,r)

        # learning policy using policy model
        policy_learning_model = Net(n_dim,n_nodes, n_class)
        policy_learning_model = policy_learning_model.to(device)

        optimizer = optim.Adam(policy_learning_model.parameters(), lr=args.lr_pg, weight_decay=0.07)

        count = 0
        prob_action_pre = (1.0/n_class) *torch.ones((len(train_data_mse), n_class), dtype = torch.float64)
        action_pre = torch.ones(len(train_data_mse), dtype = torch.int64)



        best_epoch = 1
        best_loss = 10000
        for epoch in range(1, args.epochs_policy_gradient + 1):
            # training using new \pi policy
            model_robust_list = []
            Myy_robust_list = []
            Myx_robust_list = []
            train_data_robust = train_data_mse

            for i in range(n_class):
                index = a == i
                x_i = x[index]
                r_i = r[index]

                weight_st = logging_policy[i]/prob_action_pre[index,i]
                train_size = np.sum(index)

                weighted_train = WEIGHT_DATA_SET_DRFIRST(x_i, r_i, weight_st,args)
                train_model = Net(n_dim,n_nodes, 1)
                train_model = train_model.to(device)

                validate_size = int(0.1*train_size)
                validate_size = 1 if validate_size else validate_size
                try:
                    validate_loader = data.DataLoader(data.Subset(weighted_train, range(0, validate_size)),
                        batch_size=args.batch_size, shuffle=True, **kwargs)
                    # 10% validation set
                    train_loader = data.DataLoader(data.Subset(weighted_train, range(validate_size, train_size)),
                        batch_size=args.batch_size, shuffle=True, **kwargs)


                    train_model, Myy, Myx, _, _, _ = train_validate_test(args, args.epochs_training, "regression", device, use_cuda, train_model,
                        train_loader, validate_loader , validate_loader, n_class, 0.000, testflag = False)
                except:
                    Myy = np.zeros((1, 1))
                    Myx = np.zeros((d + 1, 1))
                model_robust_list.append(train_model)
                Myy_robust_list.append(Myy)
                Myx_robust_list.append(Myx)


            reward_training = torch.ones(len(train_data_mse), dtype = torch.float64)
            # features_training = test_partial_data.get_features()
            # action_training = test_partial_data.get_action()
            # action_true = test_partial_data.get_action_true()
            weight_predict = torch.ones(len(train_data_mse), dtype = torch.float64)

            with torch.no_grad():
                k = 0
                while k < len(reward_training):
                    # features = x[k:k+4096]

                    action_policy = sample_action_batch(prob_action_pre[k:k + 4096])


                    weight_predict[k:k+len(action_policy)] = torch.tensor(logging_policy[action_policy]/prob_action_pre[k:k + len(action_policy)][np.arange(len(action_policy)),action_policy])

                    action_pre[k:k+len(action_policy)] = torch.tensor(action_policy)
                    k+= 4096

                for h in range(n_class):
                    index = a==h
                    features = x[index]
                    weight = weight_predict[index]
                    act = action_pre[index]
        
                    output = model_robust_list[h](torch.tensor(features))
                    meanY_robust, varY = ru.predict_regression(torch.tensor(weight), Myy_robust_list[h], Myx_robust_list[h], output, mean0, var0)
                    reward_training[index] = meanY_robust
            ips_weight = torch.clamp(1/weight_predict,0.1,10)
            print('act',action_pre)
            print('policy',prob_action_pre)
            print('ips',ips_weight)
            print('robustregression',reward_training)
            print(torch.sum(reward_training))
            # reward_training = (1-ips_weight) * reward_training+ ips_weight * r
            # print('dr',reward_training)
            policy_gradient_data = DATA_policy_gradient(x, reward_training, action_pre, a)
            # construct dataloader
            train_loader = data.DataLoader(policy_gradient_data,
                batch_size=args.batch_size, shuffle=True, **kwargs)
            # shuffle should be false because an ordered prob_action_next is needed
            optimizer.zero_grad()
            policy_learning_model.train()
            prob_action_next = torch.empty([0, n_class])
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):
                evaluate_flag+=1
                optimizer.zero_grad()
                bsize = len(feature_idx)
                output_learning = policy_learning_model(feature_idx)
                prob = my_softmax(output_learning)

                pi_mean = output_learning.max(1, keepdim=True)[1]
                action_onehot = torch.DoubleTensor(bsize,n_class)
                action_onehot.zero_()
                action_onehot.scatter_(1, action_idx.reshape(bsize, 1), 1)

                grad_policy = ru.policy_gradient.apply(output_learning, prob, reward_idx, action_onehot)
                grad_policy.backward(torch.ones(grad_policy.shape),retain_graph=False)
                optimizer.step()
                # break
            torch.save(policy_learning_model.state_dict(), 'model/'+str(epoch)+'_.pth')

                



            prob_action_next = torch.empty([0, n_class])
            train_loader = data.DataLoader(policy_gradient_data,
                batch_size=10000, shuffle=False, **kwargs)
            for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):
                # print('prob next')
                output_learning = policy_learning_model(feature_idx)
                prob = my_softmax(output_learning)

                prob = prob.detach().numpy()
                prob_action_next = np.concatenate((prob_action_next, prob), axis=0)


            prob_action_pre = torch.tensor(prob_action_next, dtype=torch.float64)
            prob_action_pre = prob_action_pre.reshape((len(train_data_mse), n_class))

            prob_action_pre = torch.tensor(prob_action_next, dtype = torch.float64)
            prob_action_pre = prob_action_pre.reshape((len(train_data_mse), n_class))

            
            # evaluate

            policy_gradient_data_eval = DATA_policy_gradient(eval_x, reward_training, eval_a, eval_a)
            
            prob_action_next = torch.empty([0, n_class])
            eval_loader = data.DataLoader(policy_gradient_data_eval,batch_size=10000, shuffle=False, **kwargs)
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
                policy = np.clip(policy, 0.1,10)


                eval_policy.append(policy)
                reward_eval += policy*eval_r[i]

            print(reward_eval, np.mean(np.array(eval_policy)*eval_r)/np.mean(eval_policy), np.sum(eval_r))


if __name__ == '__main__':
    main()

        


 
