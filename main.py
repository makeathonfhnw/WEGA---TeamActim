import os
from datetime import datetime
import torch as th
import visdom as vis
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from functools import reduce
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import f1_score


import torch.utils.data as Data

class BasicRNN_Regression(th.nn.Module):
    def __init__(self, input_size, state_size, num_class):
        super().__init__()

        #define the elements of the basic RNN equations
        self.W = th.nn.Linear(input_size,state_size)

        self.U = th.nn.Linear(state_size,state_size)

        # define the final layer that maps from the final state of the RNN to the class estimations
        self.classifier = th.nn.Linear(state_size,num_class)
        self.state_size = state_size

    def forward(self, input):
        # define and initialize the internal state
        state = th.zeros(input.shape[0], self.state_size, device=input.device) #8x512
        # input shape 8x1x28x28
#        num_rows = input.shape[1] #28
        num_rows = 1#28


        for i in range(num_rows):
            #input_current = input[:, :, i, :].squeeze(1) #8x28
            input_current = input[:, :] #8x28
            # implement the basic RNN equation here
            state = th.tanh(self.W(input_current)+self.U(state))
        # compute the final classification vector
        logits = self.classifier(state)


        return (logits)


class GRU_Regression(th.nn.Module):
    def __init__(self, input_size, state_size, num_class):
        super().__init__()

        # define the elements of the GRU equations
        self.W_r = th.nn.Linear(input_size,state_size)
        self.U_r = th.nn.Linear(state_size,state_size)

        self.W_z = th.nn.Linear(input_size,state_size)
        self.U_z = th.nn.Linear(state_size,state_size)

        self.W = th.nn.Linear(input_size,state_size)
        self.U = th.nn.Linear(state_size,state_size)

        # define the final layer that maps from the final state of the RNN to the class estimations
        self.classifier = th.nn.Linear(state_size,num_class)

        self.state_size = state_size

    def forward(self, input):
        # define and initialize the internal state
        state = th.zeros(input.shape[0], self.state_size, device=input.device)

        num_rows = 1

        for i in range(num_rows):
            #input_current = input[:, i].squeeze(1)
            input_current = input[:, :] #8x28

            # implement the GRU equation here
            forget_gate = th.sigmoid(self.W_r(input_current) + self.U_r(state))
            update_gate = th.sigmoid(self.W_z(input_current) + self.U_z(state))
            proposal_state = th.tanh(self.W(input_current)+self.U(forget_gate*state))
            state =(1-update_gate)* state + update_gate*proposal_state

        # compute the final classification vector
        logits = self.classifier(state)

        return logits

def train(result_path, num_epoch=12, batch_size=64, seed=123456, visdom_port=8097):
    th.manual_seed(seed)
    np.random.seed(seed)

    current_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    result_path = os.path.join(result_path, current_date)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    device = th.device("cpu")

    # mdoel generatation basic RNN
    model = BasicRNN_Regression(input_size=18, state_size=512, num_class=18)

    # mdoel generatation GRU
    #model = GRU_Classification(input_size=284, state_size=512, num_class=1)

    model.to(device)
    model.train()

    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters model", params)

    # load data
    X_train = pd.read_csv('MAKEathon-2020/preprocessed_data/ankle/X_tr_01.csv')
    X_test = pd.read_csv('MAKEathon-2020/preprocessed_data/ankle/X_ts_01.csv')
    y_train = pd.read_csv('MAKEathon-2020/preprocessed_data/ankle/Y_tr_01.csv')
    y_test = pd.read_csv('MAKEathon-2020/preprocessed_data/ankle/Y_ts_01.csv')
    columns_to_drop = ['Xstd', 'Ystd', 'Zstd', 'Xmad', 'Ymad', 'Zmad']

    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)

    #y_ = (tuple(y_train['label']))

    X_train = th.from_numpy(X_train.values).float()
    y_train = th.from_numpy(y_train.values).float()
    X_test = th.from_numpy(X_test.values).float()
    y_test = th.from_numpy(y_test.values).float()
    y_onehot = th.FloatTensor(batch_size, 10)

    y_train = th.nn.functional.one_hot(y_train.long()).squeeze()
    y_test = th.nn.functional.one_hot(y_test.long()).squeeze()
    torch_dataset = Data.TensorDataset(X_train, y_train)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=0)
    #  optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)
    # loss function
    loss = th.nn.CrossEntropyLoss()

    idx = 0
    loss_view = None
    for epoche in range(num_epoch):
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            #y_onehot.zero_()

            #y_onehot.scatter_(1, batch_y, 1)
            estimated_class_label = model(batch_x)
            loss_value = loss(estimated_class_label, th.max(batch_y, 1)[1])

            optimizer.zero_grad()
            loss_value.backward()

            for p in model.parameters():
                if p.grad is None:
                    continue
                p.grad.data = p.grad.data.clamp(-1, 1)

            optimizer.step()

            if idx % 25 == 0:
                estimate_class = estimated_class_label[0, ...]
                label = batch_y[0, ...]

                state = {"epoche": epoche + 1, "model_state_dict": model.state_dict(),
                         "optimizer_state": optimizer.state_dict()}
                th.save(state, os.path.join(result_path, "model_minst_AE_" + str(idx) + ".pth"))

            if loss_view is None:
                loss_value_ = np.column_stack(np.array([loss_value.item()]))
            else:
                loss_value_ = np.column_stack(np.array([loss_value.item()]))

            idx = idx + 1

            print("Epoch", epoche, " Index", idx, "loss", loss_value.item())

    oupt = model(X_test)
    n_items = len(y_test)
    print(len(oupt))
    print(n_items)
    #pred = oupt.view(n_items)  # all predicted as 1-d
    sum_of_errors = th.sum(th.pow(oupt.detach() - y_test, 2)).item()

    y_sum = th.sum(y_test).item()
    y_sq_sum = th.sum(th.pow(y_test, 2)).item()
    oupt = np.argmax(oupt.detach(), axis=1)
    y_test= np.argmax(y_test, axis=1)
    print('F1: {}'.format(f1_score(y_test, oupt, average="weighted")))


if __name__ == "__main__":
    train(result_path="/tmp")
