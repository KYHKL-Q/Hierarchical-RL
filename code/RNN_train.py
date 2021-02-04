import numpy as np
import json
import torch
import os
from torch import nn, optim

class rnn_train(object):
    def __init__(self,
                learning_rate=0.001,
                iteration=100,
                city='city_sample'):
        self.learning_rate=learning_rate
        self.ineration=iteration
        self.city=city

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self,path_train,path_eval,mode):
        self.batch_count=len(path_train)
        self.batch_size=len(path_train[0])
        self.data_input=list()
        self.data_gt=list()
        for i in range(self.batch_count):
            self.data_input.append(list())
            self.data_gt.append(list())
            for j in range(self.batch_size):
                with open(path_train[i][j],'r') as f:
                    temp = torch.FloatTensor(json.load(f))*0.001
                    if mode=='L':
                        self.data_input[i].append(temp[:,:,3].to(self.device).unsqueeze(0))
                        self.data_gt[i].append(temp[:,:, 1].to(self.device).unsqueeze(0))
                    elif mode=='Iut':
                        self.data_input[i].append(temp[:,:,3].to(self.device).unsqueeze(0))
                        self.data_gt[i].append(temp[:,:, 2].to(self.device).unsqueeze(0))
                    elif mode=='R':
                        self.data_input[i].append(temp[:,:,-1].to(self.device).unsqueeze(0))
                        self.data_gt[i].append(temp[:,:, -2].to(self.device).unsqueeze(0))
                print('Loading train-data {}/{}'.format(i*self.batch_size+j+1,self.batch_size*self.batch_count))
            self.data_input[i] = torch.cat(self.data_input[i], dim=0)
            self.data_gt[i] = torch.cat(self.data_gt[i], dim=0)

        with open(path_eval,'r') as f:
            temp=torch.FloatTensor(json.load(f))*0.001
            if mode=='L':
                self.data_input_eval=temp[:,:,3].to(self.device).unsqueeze(0)
                self.data_gt_eval=temp[:,:, 1].to(self.device).unsqueeze(0)
            elif mode=='Iut':
                self.data_input_eval=temp[:,:,3].to(self.device).unsqueeze(0)
                self.data_gt_eval=temp[:,:, 2].to(self.device).unsqueeze(0)
            elif mode=='R':
                self.data_input_eval=temp[:,:,-1].to(self.device).unsqueeze(0)
                self.data_gt_eval=temp[:,:, -2].to(self.device).unsqueeze(0)

        from Net.RNN import RNN

        self.net=RNN().to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.loss_func=nn.MSELoss()
        self.train_loss_record=list()
        self.eval_loss_record=list()

        loss_min=100000000
        for i in range(self.ineration):
            self.net.train()
            loss_sum=0
            for j in range(self.batch_count):
                y_hat,_=self.net(self.data_input[j])
                loss = self.loss_func(y_hat, self.data_gt[j])
                loss_sum+=loss.cpu().item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('Trianing {}/{} | loss:{}'.format(i+1,self.ineration,loss_sum/self.batch_count))
            self.train_loss_record.append(loss_sum/self.batch_count)

            self.net.eval()
            y_hat, _ = self.net(self.data_input_eval)
            y_hat_round = (y_hat*1000).clamp_min(0).round()
            loss=self.loss_func(y_hat_round,self.data_gt_eval*1000)
            self.eval_loss_record.append(loss.cpu().item())
            print('Evaluating | loss:{}'.format(loss))

            if loss.cpu().item()<loss_min:
                loss_min=loss.cpu().item()
                if os.path.exists(os.path.join('../model',self.city,mode+'_RNN.pth')):
                    os.remove(os.path.join('../model',self.city,mode+'_RNN.pth'))
                torch.save(self.net.state_dict(),os.path.join('../model',self.city,mode+'_RNN.pth'))

        '''
        with open(os.path.join('../model',self.city,mode+'_train_loss.json'),'w') as f:
            json.dump(self.train_loss_record,f)
        with open(os.path.join('../model',self.city,mode+'_eval_loss.json'),'w') as f:
            json.dump(self.eval_loss_record,f)
        '''

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    train_platform = rnn_train()
    train_platform.train(
        path_train='path_train',
        path_eval='path_eval',
        mode='mode')