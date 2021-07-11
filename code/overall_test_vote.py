import numpy as np
import json
import torch
import os
from simulator import simulator
from utils.merge import merge

class RL_test_vote(object):
    def __init__(self,
                bed_total=10000,
                mask_total=1000000,
                mask_quality=0.9,
                mask_lasting=48,
                steps=60,
                interval=48,
                repeat=20,
                model_number=10,
                city='city_sample'):
        self.bed_total=bed_total
        self.mask_total=mask_total
        self.mask_quality=mask_quality
        self.mask_lasting=mask_lasting
        self.steps=steps
        self.interval=interval
        self.repeat=repeat
        self.model_number=model_number
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
        self.Bed_action_list=list()
        self.Mask_action_list=list()
        self.Bed_Actor_list=list()
        self.Mask_Actor_list=list()
        for i in range(self.model_number):
            temp=Qnet().to(self.device)
            temp.load_state_dict(torch.load(os.path.join('../model',self.city,'bed_action_model_{}.pth'.format(i))))
            temp.eval()
            self.Bed_action_list.append(temp)

            temp=Qnet().to(self.device)
            temp.load_state_dict(torch.load(os.path.join('../model',self.city,'mask_action_model.pth')))
            temp.eval()
            self.Mask_action_list.append(temp)

            temp=Anet().to(self.device)
            temp.load_state_dict(torch.load(os.path.join('../model',self.city,'bed_Actor_model.pth')))
            temp.eval()
            self.Bed_Actor_list.append(temp)

            temp=Anet().to(self.device)
            temp.load_state_dict(torch.load(os.path.join('../model',self.city,'mask_Actor_model.pth')))
            temp.eval()
            self.Mask_Actor_list.append(temp)

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

                bed_action_count=np.zeros(7,dtype=int)
                mask_action_count=np.zeros(7,dtype=int)
                for i in range(self.model_number):
                    bed_action_out = self.Bed_action_list[i](info_rebuild)
                    mask_action_out = self.Mask_action_list[i](info_rebuild)
                    bed_action_count[torch.argmax(bed_action_out).cpu().item()]+=1
                    mask_action_count[torch.argmax(mask_action_out).cpu().item()]+=1


                bed_action=np.argmax(bed_action_count)
                mask_action=np.argmax(mask_action_count)
                action_record.append([bed_action,mask_action])

                bed_perc_list=list()
                mask_perc_list=list()
                for i in range(self.model_number):
                    bed_perc_out = self.Bed_Actor_list[i](info_rebuild)
                    mask_perc_out = self.Mask_Actor_list[i](info_rebuild)
                    bed_perc_list.append(np.clip(bed_perc_out.squeeze(0).cpu().detach().item(),0.1,1))
                    mask_perc_list.append(np.clip(mask_perc_out.squeeze(0).cpu().detach().item(),0.1,1))

                bed_perc=np.mean(np.array(bed_perc_list))
                mask_perc=np.mean(np.array(mask_perc_list))
                perc_record.append([bed_perc,mask_perc])

                self.next_state, _ = self.simulator.simulate(sim_type='Policy_a', interval=self.interval, bed_total=self.bed_total, bed_action=bed_action, bed_satisfy_perc=bed_perc, mask_on=True, mask_total=self.mask_total, mask_quality=self.mask_quality, mask_lasting=self.mask_lasting, mask_action=mask_action, mask_satisfy_perc=mask_perc, is_save=True, save_time=step)

                print('Testing:{}/{}\n'.format(step+1,self.steps))
                self.current_state=self.next_state
            '''
            with open(os.path.join('../result',self.city,'all_action_vote_{}.json'.format(ti)),'w') as f:
                json.dump(action_record,f)
            with open(os.path.join('../result',self.city,'all_perc_vote_{}.json'.format(ti)),'w') as f:
                json.dump(perc_record, f)
            '''
            merge(self.steps*self.interval,'all_vote_'+str(ti)+'.json',self.city)

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    test_platform=RL_test_vote()
    test_platform.test()