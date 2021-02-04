import numpy as np
import json
import torch
import os
from simulator import simulator
from utils.merge import merge

class RL_test(object):
    def __init__(self,
                bed_total=10000,
                mask_total=1000000,
                mask_quality=0.9,
                mask_lasting=48,
                steps=60,
                interval=48,
                repeat=10,
                city='city_sample'):
        self.bed_total=bed_total
        self.mask_total=mask_total
        self.mask_quality=mask_quality
        self.mask_lasting=mask_lasting
        self.steps=steps
        self.interval=interval
        self.repeat=repeat
        self.city=city

        #data loading
        with open(os.path.join('../data',self.city,'start.json'),'r') as f:
            self.start=np.array(json.load(f))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #simulator initialization
        self.simulator = simulator(city=self.city)

        from Net.Anet import Anet
        from Net.Qnet import Qnet
        from Net.RNN import RNN
        self.region_num = 673 #Supposed to be modified according to the number of regions in the studied city.

        #models loading
        self.Bed_action=Qnet().to(self.device)
        self.Bed_action.load_state_dict(torch.load(os.path.join('../model',self.city,'bed_action_model.pth')))
        self.Bed_action.eval()
        self.Mask_action=Qnet().to(self.device)
        self.Mask_action.load_state_dict(torch.load(os.path.join('../model',self.city,'mask_action_model.pth')))
        self.Mask_action.eval()

        self.Bed_Actor=Anet().to(self.device)
        self.Bed_Actor.load_state_dict(torch.load(os.path.join('../model',self.city,'bed_Actor_model.pth')))
        self.Bed_Actor.eval()
        self.Mask_Actor=Anet().to(self.device)
        self.Mask_Actor.load_state_dict(torch.load(os.path.join('../model',self.city,'mask_Actor_model.pth')))
        self.Mask_Actor.eval()

        self.L_RNN=RNN().to(self.device)
        self.L_RNN.load_state_dict(torch.load(os.path.join('../model',self.city,'L_RNN.pth')))
        self.L_RNN.eval()
        self.Iut_RNN=RNN().to(self.device)
        self.Iut_RNN.load_state_dict(torch.load(os.path.join('../model',self.city,'Iut_RNN.pth')))
        self.Iut_RNN.eval()
        self.R_RNN=RNN().to(self.device)
        self.R_RNN.load_state_dict(torch.load(os.path.join('../model',self.city,'R_RNN.pth')))
        self.R_RNN.eval()

    def test(self):
        for ti in range(self.repeat):
            self.simulator.reset(self.start)
            self.current_state = self.start

            L_h0 = torch.zeros(self.L_RNN.num_layers, 1, self.L_RNN.hidden_size).to(self.device)
            Iut_h0 = torch.zeros(self.Iut_RNN.num_layers, 1, self.Iut_RNN.hidden_size).to(self.device)
            R_h0 = torch.zeros(self.R_RNN.num_layers, 1, self.R_RNN.hidden_size).to(self.device)

            action_record=list()
            perc_record=list()
            for step in range(self.steps):
                info_perfect = torch.FloatTensor(self.current_state).to(self.device).unsqueeze(0)*0.001#1*region_num*8

                L_rebuild, L_h0 = self.L_RNN(info_perfect[:,:,3].unsqueeze(2).permute(0,2,1), L_h0,is_eval=True)
                L_rebuild = L_rebuild.clamp_min(0).round()
                Iut_rebuild, Iut_h0 = self.Iut_RNN(info_perfect[:,:, 3].unsqueeze(2).permute(0, 2, 1), Iut_h0, is_eval=True)
                Iut_rebuild = Iut_rebuild.clamp_min(0).round()
                R_rebuild, R_h0 = self.R_RNN(info_perfect[:,:, -1].unsqueeze(2).permute(0, 2, 1), R_h0, is_eval=True)
                R_rebuild = R_rebuild.clamp_min(0).round()

                info_rebuild = torch.cat((L_rebuild.permute(0,2,1), Iut_rebuild.permute(0,2,1), info_perfect[:,:, 3].unsqueeze(2), info_perfect[:,:, 5].unsqueeze(2), R_rebuild.permute(0,2,1), info_perfect[:,:, -1].unsqueeze(2)), dim=2)*1000

                bed_action_out = self.Bed_action(info_rebuild)
                mask_action_out = self.Mask_action(info_rebuild)
                bed_action=torch.argmax(bed_action_out).cpu().item()
                mask_action=torch.argmax(mask_action_out).cpu().item()
                print('bed_action:{}'.format(bed_action))
                print('mask_action:{}'.format(mask_action))
                action_record.append([bed_action,mask_action])

                bed_perc = self.Bed_Actor(info_rebuild)
                mask_perc = self.Mask_Actor(info_rebuild)
                bed_perc = np.clip(bed_perc.squeeze(0).cpu().detach().item(),0.1,1)
                mask_perc = np.clip(mask_perc.squeeze(0).cpu().detach().item(),0.1,1)
                print('bed_perc:{}'.format(bed_perc))
                print('mask_perc:{}'.format(mask_perc))
                perc_record.append([bed_perc,mask_perc])

                self.next_state, _ = self.simulator.simulate(sim_type='Policy_a', interval=self.interval, bed_total=self.bed_total, bed_action=bed_action, bed_satisfy_perc=bed_perc, mask_on=True, mask_total=self.mask_total, mask_quality=self.mask_quality, mask_lasting=self.mask_lasting, mask_action=mask_action, mask_satisfy_perc=mask_perc, is_save=True, save_time=step)

                print('Testing:{}/{}\n'.format(step+1,self.steps))
                self.current_state=self.next_state

            '''
            with open(os.path.join('../result',self.city,'all_action{}.json'.format(ti)),'w') as f:
                json.dump(action_record,f)
            with open(os.path.join('../result',self.city,'all_perc{}.json'.format(ti)),'w') as f:
                json.dump(perc_record, f)
            '''
            merge(self.steps*self.interval,'overall_'+str(ti)+'.json',self.city)

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    test_platform=RL_test()
    test_platform.test()