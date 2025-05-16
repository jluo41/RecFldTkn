from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .DROPO.data_set import WEIGHT_DATA_SET,WEIGHT_DATA_SET_DRFIRST
from .DROPO.data import DATA, DATA_defined_prob, DATA_policy_gradient, DATA_fullaction, DATA_action, DATA_partial_logistic, DATA_partial_logistic_deep, DATA_partial_random, DATA_partial_action, DATA_learn_policy, Drfirst_data
from torchvision import transforms
import torchvision
import os
import copy
import torch.utils.data as data
# from data_mnist import DATA_mnist
# from data_cifar import DATA_CIFAR10 
from .DROPO import regression_utility as ru
from .DROPO import abstain_utility as au
from scipy import stats
import math
from scipy.stats import dirichlet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.preprocessing import StandardScaler


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


# def spectral_norm(module, name='weight'):
#     SpectralNorm.apply(module, name)

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
    return y_prediction, y_var, test_loss


def train_validate_test(args, epoch, loss_type, device, use_cuda, train_model, train_loader, test_loader, validate_loader, n_class, lbd, testflag = True):
    
    if loss_type == 'regression':

        Myy = np.ones((1, 1))
        Myx = np.ones((d+1, 1))
        optimizer = optim.Adam(train_model.parameters(),lr=args.lr)

        for epoch in range(1, epoch + 1):
            Myy, Myx = train_regression(args, train_model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0) 
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, validate_loader, mean0, var0)
        if testflag == True:
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, test_loader, mean0, var0)
        return train_model, Myy, Myx, meanY, varY, loss
        # return train_model, pred_Y, loss
    else:
        raise ValueError(f'loss_type {loss_type} is not supported')
    
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
         

def train_bandit(X, A, R, test_X, test_A, test_R, train_args, model_args):

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    args = train_args

    evaluate_flag = 0

    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    all_x = X
    all_a = A
    all_r = R

    
    scaler = StandardScaler()
    scaler.fit(all_x)
    all_x = scaler.transform(all_x)

    index = np.random.permutation(len(all_x))



    n_dim = all_x.shape[1]
    n_class = max(all_a) + 1

    x = all_x
    a = all_a
    r = all_r


    eval_x = test_X
    eval_a = test_A
    eval_r = test_R

    n_dim = all_x.shape[1]
    # n_class = 24

    n_nodes = 256


    logging_policy = np.ones(n_class)/n_class
    # print(logging_policy)



    train_data_mse = Drfirst_data(x,a,r)

    # learning policy using policy model
    policy_learning_model = Net(n_dim, n_nodes, n_class)
    policy_learning_model = policy_learning_model.to(device)


    optimizer = optim.Adam(policy_learning_model.parameters(), lr=args.lr_pg, weight_decay=0.07)

    count = 0
    prob_action_pre = (1.0/n_class) *np.ones((len(train_data_mse), n_class))
    action_pre = torch.ones(len(train_data_mse), dtype = torch.int64)


    eval_loss = []
    best_epoch = 1
    best_loss = 10000

    model_robust_list = []
    train_data_robust = train_data_mse


    for i in range(n_class):
        index = a == i
        x_i = x[index]
        r_i = r[index]

        X_train, X_test, y_train, y_test = train_test_split(x_i, r_i, test_size=0.2, random_state=42)

        # dtrain = xgb.DMatrix(X_train, label=y_train)
        # dtest =xgb.DMatrix(X_test, label=y_test)




        # evals = [(dtrain, 'train'), (dtest, 'test')]
        train_model = xgb.XGBRegressor(**model_args)
        train_model.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose = 0 )
        y_pred = train_model.predict(X_test)
        print('reward model of action ' + str(i) +':')
        print(' testing MSE', round(np.mean((y_pred - y_test)**2) ,3),' train MSE', round(np.mean((train_model.predict(X_train) - y_train)**2) ,3)  )

        # return train_model.feature_importance(importance_type='gain')
        model_robust_list.append(train_model)

    reward_training = np.ones(len(train_data_mse))
    reward_training_all_action = np.ones((len(train_data_mse),n_class))

    policy_gradient_data_eval = DATA_policy_gradient(eval_x, eval_a, eval_a, eval_a)

    prob_action_next = torch.empty([0, n_class])
    eval_loader = data.DataLoader(policy_gradient_data_eval, batch_size=10000, shuffle=False, **kwargs)
    for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(eval_loader):
        output_learning = policy_learning_model(feature_idx)
        prob = my_softmax(output_learning)
        prob = prob.detach().numpy()
        prob_action_next = np.concatenate((prob_action_next, prob), axis=0)

    eval_reward = np.zeros((len(eval_x), n_class))
    for h in range(n_class):
        eval_reward[:, h] = model_robust_list[h].predict(eval_x)
    action_eval_sampled = sample_action_batch(prob_action_next)
    eval_ope = eval_reward[np.arange(len(action_eval_sampled)), action_eval_sampled]
    eval_ope += prob_action_next[np.arange(len(prob_action_next)), eval_a] * n_class * (eval_r \
                                                                                        - eval_reward[np.arange(
                len(eval_a)), eval_a])
    print('Predicted engagement rate before training, :', np.mean(eval_ope))

    for epoch in range(1, args.epochs_policy_gradient + 1):
        action_policy = sample_action_batch(prob_action_pre)
        for h in range(n_class):
            features = x
            output = model_robust_list[h].predict(features)
            reward_training_all_action[:,h] = output

        reward_training = reward_training_all_action[np.arange(len(reward_training_all_action)), action_policy]

        reward_training = reward_training + (r - reward_training_all_action[np.arange(len(reward_training_all_action)), a]) * (prob_action_pre[np.arange(len(prob_action_pre)),a]*n_class)
        reward_training = reward_training
        print('Predicted engagement rate on train:', np.sum(reward_training)/len(reward_training))
        eval_loss.append((np.sum(reward_training)/len(reward_training)).item())


        policy_gradient_data = DATA_policy_gradient(x, reward_training, action_policy, a)
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
            action_onehot = torch.DoubleTensor(bsize,n_class)
            action_onehot.zero_()
            action_onehot.scatter_(1, action_idx.reshape(bsize, 1), 1)

            grad_policy = ru.policy_gradient.apply(output_learning, prob, reward_idx, action_onehot)
            grad_policy.backward(torch.ones(grad_policy.shape),retain_graph=False)
            optimizer.step()

        prob_action_next = torch.empty([0, n_class])
        train_loader = data.DataLoader(policy_gradient_data,
            batch_size=10000, shuffle=False, **kwargs)
        for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(train_loader):

            output_learning = policy_learning_model(feature_idx)
            prob = my_softmax(output_learning)

            prob = prob.detach().numpy()
            prob_action_next = np.concatenate((prob_action_next, prob), axis=0)

        #
        # prob_action_pre = torch.tensor(prob_action_next, dtype=torch.float64)
        # prob_action_pre = prob_action_pre.reshape((len(train_data_mse), n_class))

        prob_action_pre = torch.tensor(prob_action_next, dtype = torch.float64)
        prob_action_pre = prob_action_pre.reshape((len(train_data_mse), n_class))
        prob_action_pre = prob_action_pre.detach().numpy()

 

        policy_gradient_data_eval = DATA_policy_gradient(eval_x, reward_training, eval_a, eval_a)

        prob_action_next = torch.empty([0, n_class])
        eval_loader = data.DataLoader(policy_gradient_data_eval,batch_size=10000, shuffle=False, **kwargs)
        for batch_idx, (feature_idx, action_idx, reward_idx, action_true_idx) in enumerate(eval_loader):
            output_learning = policy_learning_model(feature_idx)
            prob = my_softmax(output_learning)
            prob = prob.detach().numpy()
            prob_action_next = np.concatenate((prob_action_next, prob), axis=0)

        eval_reward = np.zeros((len(eval_x),n_class))
        for h in range(n_class):
            eval_reward[:,h] = model_robust_list[h].predict(eval_x)
        action_eval_sampled = sample_action_batch(prob_action_next)
        eval_ope = eval_reward[np.arange(len(action_eval_sampled)), action_eval_sampled ]
        eval_ope += prob_action_next[np.arange(len(prob_action_next)), eval_a]*n_class * (eval_r \
                                        - eval_reward[np.arange(len(eval_a)),eval_a])
        print('Predicted engagement rate on test set:', np.mean(eval_ope))

    return policy_learning_model, scaler,  model_robust_list, eval_loss, logging_policy






def train_bandit_model(model_args, 
                       training_args,
                       model_checkpoint_path,
                       X, A, R, test_X, test_A, test_R):
    
    # algorithm = model_args['algorithm']
    # args = {k: v for k, v in model_args.items() if k != 'algorithm'}

    # if not os.path.exists(model_checkpoint_path):
        # train_set = ds_case_final_dict[TrainSetName]
        # eval_sets = ds_case_final_dict['Test']
    # print(training_args)
    model, scaler, model_robust_list,eval_loss,logging_policy = train_bandit(X, A, R, test_X, test_A, test_R, training_args, model_args)


    # if SAVE_MODEL == True:
    # if not os.path.exists(model_checkpoint_path):
        # os.makedirs(model_checkpoint_path)

    # logger.info(f'save model to {model_checkpoint_path}')
    model_dict = {}
    model_dict['model'] = model
    model_dict['scaler'] = scaler
    model_dict['model_robust_list'] = model_robust_list

    return model_dict, eval_loss
