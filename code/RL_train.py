import numpy as np
import json
import torch
import random
import copy
import os
from torch import nn,optim
from simulator import simulator
from reward_fun import reward_fun

class RL_train(object):
    def __init__(self,
                action_lr=0.001,
                Actor_lr=0.001,
                Critic_lr=0.002,
                decay_rate=0.5,
                batch_size=8,
                pool_volume=128,
                action_explore_factor=0.5,
                Actor_explore_factor=0.05,
                soft_replace_rate=0.1,
                train_steps=150,
                interval=48,
                bed_total=10000,
                mask_total=1000000,
                mask_quality=0.9,
                mask_lasting=48,
                city='city_sample'):
        print('Initializing...')
        self.action_lr=action_lr
        self.Actor_lr=Actor_lr
        self.Critic_lr=Critic_lr
        self.decay_rate=decay_rate
        self.batch_size=batch_size
        self.pool_volume=pool_volume
        self.action_explore_factor=action_explore_factor
        self.Actor_explore_factor=Actor_explore_factor
        self.soft_replace_rate=soft_replace_rate
        self.train_steps=train_steps
        self.interval=interval
        self.bed_total=bed_total
        self.mask_total=mask_total
        self.mask_quality=mask_quality
        self.mask_lasting=mask_lasting
        self.city=city

        #Initialize state and replay buffer
        with open(os.path.join('../data',self.city,'start.json'),'r') as f:
            self.start = np.array(json.load(f))

        from Net.Anet import Anet
        from Net.Qnet import Qnet
        from Net.RNN import RNN
        self.region_num = 673 #Supposed to be modified according to the number of regions in the studied city.

        self.pool_pointer=0
        self.pool_count=0
        self.pool_state=np.zeros([self.pool_volume,self.region_num,8])
        self.pool_bed_action=np.zeros(self.pool_volume)
        self.pool_mask_action=np.zeros(self.pool_volume)
        self.pool_bed_perc=np.zeros(self.pool_volume)
        self.pool_mask_perc=np.zeros(self.pool_volume)
        self.pool_reward=np.zeros(self.pool_volume)
        self.pool_state1=np.zeros([self.pool_volume,self.region_num,8])

        self.current_state=self.start
        self.next_state=self.current_state

        #Initialize the networks
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.Bed_action_eval=Qnet().to(self.device)
        self.Bed_action_eval.train()
        self.Bed_action_target=copy.deepcopy(self.Bed_action_eval)
        self.Bed_action_target.eval()
        self.Bed_action_optimizer=optim.Adam(self.Bed_action_eval.parameters(), lr=self.action_lr)

        self.Bed_Actor_eval=Anet().to(self.device)
        self.Bed_Actor_eval.train()
        self.Bed_Actor_target=copy.deepcopy(self.Bed_Actor_eval)
        self.Bed_Actor_target.eval()
        self.Bed_Actor_optimizer=optim.Adam(self.Bed_Actor_eval.parameters(),lr=self.Actor_lr)

        self.Bed_Critic_eval=Cnet(self.region_num).to(self.device)
        self.Bed_Critic_eval.train()
        self.Bed_Critic_target=copy.deepcopy(self.Bed_Critic_eval)
        self.Bed_Critic_target.eval()
        self.Bed_Critic_optimizer=optim.Adam(self.Bed_Critic_eval.parameters(),lr=self.Critic_lr)

        self.Mask_action_eval=Qnet().to(self.device)
        self.Mask_action_eval.train()
        self.Mask_action_target=copy.deepcopy(self.Mask_action_eval)
        self.Mask_action_target.eval()
        self.Mask_action_optimizer=optim.Adam(self.Mask_action_eval.parameters(),lr=self.action_lr)

        self.Mask_Actor_eval=Anet().to(self.device)
        self.Mask_Actor_eval.train()
        self.Mask_Actor_target=copy.deepcopy(self.Mask_Actor_eval)
        self.Mask_Actor_target.eval()
        self.Mask_Actor_optimizer=optim.Adam(self.Mask_Actor_eval.parameters(),lr=self.Actor_lr)

        self.Mask_Critic_eval=Cnet(self.region_num).to(self.device)
        self.Mask_Critic_eval.train()
        self.Mask_Critic_target=copy.deepcopy(self.Mask_Critic_eval)
        self.Mask_Critic_eval.eval()
        self.Mask_Critic_optimizer=optim.Adam(self.Mask_Critic_eval.parameters(),lr=self.Critic_lr)

        #Initialize the simulator
        self.simulator=simulator(city=self.city)
        self.simulator.reset(self.start)
        print('Initializing down!')

    def train(self):
        print('Start training...')
        loss_record = list()
        train_count = 0
        while train_count < self.train_steps:
            self.current_state=self.start
            self.next_state=self.current_state
            self.simulator.reset(self.start)
            self.is_end=False
            end_count = 0

            #sampling
            step_count=0
            while ((not self.is_end) and step_count <= 60/(self.interval/48)):
                #estimating the long term reward of each action
                bed_action_out = self.Bed_action_eval(torch.FloatTensor(self.current_state[:,[1,2,3,5,6,7]]).to(self.device).unsqueeze(0))
                mask_action_out = self.Mask_action_eval(torch.FloatTensor(self.current_state[:,[1,2,3,5,6,7]]).to(self.device).unsqueeze(0))

                #select an action through epsilon-greedy
                bed_action=torch.argmax(bed_action_out).cpu().item()
                print('bed_action:{}'.format(bed_action))
                rand_temp=random.random()
                if(rand_temp>(1-self.action_explore_factor)):
                    bed_action = int(7 * random.random())

                mask_action=torch.argmax(mask_action_out).cpu().item()
                print('mask_action:{}'.format(mask_action))
                rand_temp=random.random()
                if(rand_temp>(1-self.action_explore_factor)):
                    mask_action = int(7 * random.random())

                bed_perc = self.Bed_Actor_eval(torch.FloatTensor(self.current_state[:,[1,2,3,5,6,7]]).to(self.device).unsqueeze(0))
                mask_perc = self.Mask_Actor_eval(torch.FloatTensor(self.current_state[:,[1,2,3,5,6,7]]).to(self.device).unsqueeze(0))

                bed_perc = bed_perc.squeeze(0).cpu().detach().item()+ self.Actor_explore_factor * np.random.randn()
                bed_perc_clip = np.clip(bed_perc, 0.1, 1)
                print('bed_perc:{}'.format(bed_perc_clip))

                mask_perc = mask_perc.squeeze(0).cpu().detach().item()+ self.Actor_explore_factor * np.random.randn()
                mask_perc_clip = np.clip(mask_perc, 0.1, 1)
                print('mask_perc:{}'.format(mask_perc_clip))

                #get the next state through simulation
                self.next_state, _ = self.simulator.simulate(sim_type='Policy_a', interval=self.interval, bed_total=self.bed_total, bed_action=bed_action, bed_satisfy_perc=bed_perc_clip, mask_on=True, mask_total=self.mask_total, mask_quality=self.mask_quality, mask_lasting=self.mask_lasting, mask_action=mask_action, mask_satisfy_perc=mask_perc_clip)

                if(self.simulator.full and end_count==0):
                    end_count=1
                if((not self.simulator.full) and end_count==1):
                    self.is_end=True
                print('Is_end:{}'.format(self.is_end))

                #get single step reward
                reward=reward_fun(self.current_state,self.next_state)
                print('Reward:{}'.format(reward))

                #put the sample into the replay buffer
                if(self.simulator.full):
                    self.pool_state[self.pool_pointer]=self.current_state
                    self.pool_bed_action[self.pool_pointer]=bed_action
                    self.pool_mask_action[self.pool_pointer]=mask_action
                    self.pool_bed_perc[self.pool_pointer]=bed_perc
                    self.pool_mask_perc[self.pool_pointer]=mask_perc
                    self.pool_reward[self.pool_pointer]=reward
                    self.pool_state1[self.pool_pointer]=self.next_state
                    self.pool_pointer=(self.pool_pointer+1)%self.pool_volume
                    if (self.pool_count<self.pool_volume):
                        self.pool_count=self.pool_count+1
                    print('Sampling:{} sampeles in pool now'.format(self.pool_count))

                #mini batch sampling and updating the parameters
                if (self.pool_count>=self.batch_size):
                    sample_index=random.sample(range(self.pool_count),self.batch_size)
                    sample_state=torch.FloatTensor(self.pool_state[sample_index]).to(self.device)
                    sample_bed_action=self.pool_bed_action[sample_index]
                    sample_mask_action=self.pool_mask_action[sample_index]
                    sample_reward=self.pool_reward[sample_index]
                    sample_bed_perc=torch.FloatTensor(self.pool_bed_perc[sample_index]).to(self.device).unsqueeze(1)
                    sample_mask_perc=torch.FloatTensor(self.pool_mask_perc[sample_index]).to(self.device).unsqueeze(1)
                    sample_state1=torch.FloatTensor(self.pool_state1[sample_index]).to(self.device)

                    bed_action_eval=self.Bed_action_eval(sample_state[:,:,[1,2,3,5,6,7]])
                    bed_action_target=self.Bed_action_target(sample_state1[:,:,[1,2,3,5,6,7]])
                    bed_action_max = torch.argmax(bed_action_target, dim=-1).cpu()

                    mask_action_eval=self.Mask_action_eval(sample_state[:,:,[1,2,3,5,6,7]])
                    mask_action_target=self.Mask_action_target(sample_state1[:,:,[1,2,3,5,6,7]])
                    mask_action_max = torch.argmax(mask_action_target, dim=-1).cpu()

                    bed_perc_eval=self.Bed_Actor_eval(sample_state[:,:,[1,2,3,5,6,7]])
                    bed_perc_target=self.Bed_Actor_target(sample_state1[:,:,[1,2,3,5,6,7]])
                    bed_reward_eval = self.Bed_Critic_eval(sample_state[:,:,[1,2,3,5,6,7]], sample_bed_perc)
                    bed_reward_target=self.Bed_Critic_target(sample_state1[:,:,[1,2,3,5,6,7]],bed_perc_target)

                    mask_perc_eval=self.Mask_Actor_eval(sample_state[:,:,[1,2,3,5,6,7]])
                    mask_perc_target=self.Mask_Actor_target(sample_state1[:,:,[1,2,3,5,6,7]])
                    mask_reward_eval = self.Mask_Critic_eval(sample_state[:,:,[1,2,3,5,6,7]], sample_mask_perc)
                    mask_reward_target=self.Mask_Critic_target(sample_state1[:,:,[1,2,3,5,6,7]],mask_perc_target)

                    loss = 0
                    for i in range(self.batch_size):
                        y = sample_reward[i]
                        y = y + self.decay_rate * (bed_action_target[i][bed_action_max[i]] + mask_action_target[i][mask_action_max[i]] + bed_reward_target[i] + mask_reward_target[i])
                        loss = loss + (bed_reward_eval[i] + mask_reward_eval[i] + bed_action_eval[i][int(sample_bed_action[i])] + mask_action_eval[i][int(sample_mask_action[i])] - y)** 2
                    loss=loss/self.batch_size

                    print('Loss:{}'.format(loss.cpu().item()))
                    self.Bed_action_optimizer.zero_grad()
                    self.Mask_action_optimizer.zero_grad()
                    self.Bed_Critic_optimizer.zero_grad()
                    self.Mask_Critic_optimizer.zero_grad()
                    loss.backward()
                    self.Bed_action_optimizer.step()
                    self.Mask_action_optimizer.step()
                    self.Bed_Critic_optimizer.step()
                    self.Mask_Critic_optimizer.step()

                    bed_reward0 = self.Bed_Critic_eval(sample_state[:,:,[1,2,3,5,6,7]], bed_perc_eval)
                    mask_reward0 = self.Mask_Critic_eval(sample_state[:,:,[1,2,3,5,6,7]], mask_perc_eval)
                    loss_pg = - torch.mean(bed_reward0 + mask_reward0)

                    print('Loss_pg:{}'.format(loss_pg))
                    self.Bed_Actor_optimizer.zero_grad()
                    self.Mask_Actor_optimizer.zero_grad()
                    loss_pg.backward()
                    self.Bed_Actor_optimizer.step()
                    self.Mask_Actor_optimizer.step()

                    loss_record.append([loss.cpu().item(),loss_pg.cpu().item()])

                    #soft replacement
                    for x in self.Bed_action_target.state_dict().keys():
                        eval('self.Bed_action_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                        eval('self.Bed_action_target.'+x+'.data.add_(self.soft_replace_rate*self.Bed_action_eval.'+x+'.data)')
                    for x in self.Mask_action_target.state_dict().keys():
                        eval('self.Mask_action_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                        eval('self.Mask_action_target.'+x+'.data.add_(self.soft_replace_rate*self.Mask_action_eval.'+x+'.data)')
                    for x in self.Bed_Actor_target.state_dict().keys():
                        eval('self.Bed_Actor_target.'+x+'.data.mul_((1-self.soft_replace_rate))')
                        eval('self.Bed_Actor_target.'+x+'.data.add_(self.soft_replace_rate*self.Bed_Actor_eval.'+x+'.data)')
                    for x in self.Bed_Critic_target.state_dict().keys():
                        eval('self.Bed_Critic_target.'+x+'.data.mul_((1-self.soft_replace_rate))')
                        eval('self.Bed_Critic_target.'+x+'.data.add_(self.soft_replace_rate*self.Bed_Critic_eval.'+x+'.data)')
                    for x in self.Mask_Actor_target.state_dict().keys():
                        eval('self.Mask_Actor_target.'+x+'.data.mul_((1-self.soft_replace_rate))')
                        eval('self.Mask_Actor_target.'+x+'.data.add_(self.soft_replace_rate*self.Mask_Actor_eval.'+x+'.data)')
                    for x in self.Mask_Critic_target.state_dict().keys():
                        eval('self.Mask_Critic_target.'+x+'.data.mul_((1-self.soft_replace_rate))')
                        eval('self.Mask_Critic_target.'+x+'.data.add_(self.soft_replace_rate*self.Mask_Critic_eval.'+x+'.data)')

                    train_count += 1
                    print('Training:epoch {}/{}\n'.format(train_count, self.train_steps))

                    if train_count == self.train_steps:
                        break

                #update the state
                self.current_state=self.next_state
                step_count+=1

        #save the models
        torch.save(self.Bed_action_eval.state_dict(), os.path.join('../model',self.city,'bed_action_model.pth'))
        torch.save(self.Mask_action_eval.state_dict(), os.path.join('../model',self.city,'mask_action_model.pth'))
        torch.save(self.Bed_Actor_eval.state_dict(), os.path.join('../model',self.city,'bed_Actor_model.pth'))
        torch.save(self.Mask_Actor_eval.state_dict(), os.path.join('../model',self.city,'mask_Actor_model.pth'))

        '''
        with open(os.path.join('../model',self.city,'loss.json'),'w') as f:
            json.dump(loss_record,f)
        '''
        print('Training complete!')

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    train_platform = RL_train(mask_total=1000000)
    train_platform.train()
