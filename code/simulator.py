import numpy as np
import json
import random
import os
from node import node

class simulator(object):
    def __init__(self,city='city_sample'):
        super(simulator, self).__init__()
        self.city = city

        #load data
        print('Loading dynamic pattern...')
        with open(os.path.join('../data',self.city,'prob.json'), 'r') as f:
            self.prob = np.array(json.load(f))
        print('Down!')
        print('Loading population data...')
        with open(os.path.join('../data',self.city,'flow.json'), 'r') as f:
            self.flow = np.array(json.load(f))
        with open(os.path.join('../data',self.city,'dense.json'), 'r') as f:
            self.dense = np.array(json.load(f))
        with open(os.path.join('../data',self.city,'pop_region.json'), 'r') as f:
            self.pop = np.array(json.load(f))
        print('Down!')

        self.START = 1
        self.region_num = len(self.flow)
        self.nodes = list()
        self.full = False

        self.parameters = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    def reset(self, state):  # state:region_num*8
        print('\nReseting...')
        self.nodes = list()
        for i in range(self.region_num):
            self.nodes.append(node(i))
            self.nodes[i].set_susceptible(state[i][0])
            self.nodes[i].set_latent(state[i][1])
            self.nodes[i].set_infected_ut(state[i][2])
            self.nodes[i].set_infected_t(state[i][3])
            self.nodes[i].set_infected_asymptomatic(state[i][4])
            self.nodes[i].set_in_hospital(state[i][5])
            self.nodes[i].set_recovered(state[i][6])
            self.nodes[i].set_death(state[i][7])
        print('Down!')

    def statistic(self):
        S = 0
        L = 0
        Iut = 0
        It = 0
        Ia = 0
        Ih = 0
        R = 0
        D = 0
        for i in range(self.region_num):
            S += self.nodes[i].susceptible
            L += self.nodes[i].latent
            Iut += self.nodes[i].infected_ut
            It += self.nodes[i].infected_t
            Ia += self.nodes[i].infected_asymptomatic
            Ih += self.nodes[i].in_hospital
            R += self.nodes[i].recovered
            D += self.nodes[i].death

        return int(S), int(L), int(Iut), int(It), int(Ia), int(Ih), int(R), int(D)

    def simulate(self,
                 sim_type,
                 interval,
                 bed_total,
                 bed_action=0,
                 mask_on=False,
                 mask_total=0,
                 mask_quality=0,
                 mask_lasting=48,
                 mask_action=0,
                 bed_distribute_perc=[0],
                 mask_distribute_perc=[0],
                 bed_satisfy_perc=1,
                 mask_satisfy_perc=1,
                 it_move=False,
                 is_save=False,
                 save_time=0):
        print('Simulating...')
        #temp variables
        S_temp = np.zeros((self.region_num, self.region_num + 1))
        L_temp = np.zeros((self.region_num, self.region_num + 1))
        Iut_temp = np.zeros((self.region_num, self.region_num + 1))
        if it_move:
            It_temp = np.zeros((self.region_num, self.region_num + 1))
        Ia_temp = np.zeros((self.region_num, self.region_num + 1))
        R_temp = np.zeros((self.region_num, self.region_num + 1))

        Ih_new = 0
        for time in range(interval):
            #mask
            if (time % mask_lasting == 0):
                mask_numbers = np.zeros(self.region_num)
                if mask_on:
                    mask_num = mask_total
                    if sim_type == 'Mean':
                        mask_numbers = np.ones(self.region_num) * int(mask_num / self.region_num)

                    elif sim_type == 'Lottery':
                        order = np.array(range(self.region_num))
                        np.random.shuffle(order)
                        for i in range(self.region_num):
                            if mask_num > 0:
                                mask_numbers[order[i]] = min(mask_num, self.pop[order[i]])
                                mask_num = mask_num - mask_numbers[order[i]]
                            else:
                                break

                    else:
                        patient_num = np.zeros(self.region_num)
                        patient_num_order = np.zeros(self.region_num)
                        for i in range(self.region_num):
                            patient_num[i] = self.nodes[i].infected_t
                            patient_num_order[i] = self.nodes[i].infected_t + self.nodes[i].in_hospital + self.nodes[i].recovered + self.nodes[i].death

                        if sim_type == 'Max_first':
                            order = np.argsort(-patient_num_order)
                            for i in range(self.region_num):
                                if mask_num > 0:
                                    mask_numbers[order[i]] = min(mask_num, self.pop[order[i]])
                                    mask_num = mask_num - mask_numbers[order[i]]
                                else:
                                    break

                        elif sim_type == 'Min_first':
                            order = np.argsort(patient_num_order)
                            for i in range(self.region_num):
                                if mask_num > 0:
                                    mask_numbers[order[i]] = min(mask_num, self.pop[order[i]])
                                    mask_num = mask_num - mask_numbers[order[i]]
                                else:
                                    break

                        elif sim_type == 'Policy':
                            mask_parameter = self.parameters[mask_action]
                            infect_num = patient_num_order/np.mean(patient_num_order)
                            order = np.argsort(-(mask_parameter[0]*infect_num + mask_parameter[1]*self.flow + mask_parameter[2]*self.dense))
                            alloc_mask = np.floor(mask_num*np.array(mask_distribute_perc))
                            for i in range(len(mask_distribute_perc)):
                                mask_numbers[order[i]] = min(self.pop[order[i]], alloc_mask[i])
                                mask_num -= mask_numbers[order[i]]

                            if mask_num > 0:
                                temp_order = np.argsort(-np.array(mask_distribute_perc))
                                for i in range(len(mask_distribute_perc)):
                                    temp=min(mask_num,self.pop[order[temp_order[i]]]-mask_numbers[order[temp_order[i]]])
                                    mask_num = mask_num - temp
                                    mask_numbers[order[temp_order[i]]] += temp
                                    if mask_num == 0:
                                        break

                                while mask_num > 0:
                                    index = np.random.randint(0, self.region_num)
                                    if mask_numbers[index] < self.pop[index]:
                                        mask_numbers[index] += 1
                                        mask_num -= 1
                                    else:
                                        break

                        elif sim_type == 'Policy_a':
                            mask_parameter = self.parameters[mask_action]
                            infect_num = patient_num_order/np.mean(patient_num_order)
                            order = np.argsort(-(mask_parameter[0] * infect_num + mask_parameter[1] * self.flow + mask_parameter[2] * self.dense))

                            for i in range(self.region_num):
                                if mask_num > 0:
                                    mask_numbers[order[i]] = min(mask_num, int(self.pop[order[i]]*mask_satisfy_perc))
                                    mask_num = mask_num - mask_numbers[order[i]]
                                else:
                                    break

            #state transition
            for i in range(self.region_num):
                self.nodes[i].step(mask_numbers[i], mask_quality)

            #hospital beds
            patient_num = np.zeros(self.region_num)
            patient_num_order = np.zeros(self.region_num)
            self.full = False
            bed_num = bed_total
            for i in range(self.region_num):
                patient_num[i] = self.nodes[i].infected_t
                patient_num_order[i] = self.nodes[i].infected_t + self.nodes[i].in_hospital + self.nodes[i].recovered + self.nodes[i].death
                bed_num = bed_num - self.nodes[i].in_hospital

            if bed_num > 0:
                if sim_type == 'Lottery' or sim_type == 'Mean':
                    patient_list = list()
                    for i in range(self.region_num):
                        patient_list.extend([i]*int(patient_num[i]))
                    np.random.shuffle(patient_list)
                    for i in range(len(patient_list)):
                        if bed_num > 0:
                            self.nodes[patient_list[i]].set_infected_t(self.nodes[patient_list[i]].infected_t - 1)
                            self.nodes[patient_list[i]].set_in_hospital(self.nodes[patient_list[i]].in_hospital + 1)
                            bed_num = bed_num - 1
                            Ih_new += 1
                        else:
                            self.full = True
                            break

                elif sim_type == 'Max_first':
                    order = np.argsort(-patient_num_order)
                    for i in range(self.region_num):
                        if bed_num > 0:
                            temp = min(bed_num, patient_num[order[i]])
                            self.nodes[order[i]].set_infected_t(self.nodes[order[i]].infected_t - temp)
                            self.nodes[order[i]].set_in_hospital(self.nodes[order[i]].in_hospital + temp)
                            bed_num = bed_num - temp
                            Ih_new += temp
                            patient_num[order[i]] -= temp
                        else:
                            self.full = True
                            break

                elif sim_type == 'Min_first':
                    order = np.argsort(patient_num_order)
                    for i in range(self.region_num):
                        if bed_num > 0:
                            temp = min(bed_num, patient_num[order[i]])
                            self.nodes[order[i]].set_infected_t(self.nodes[order[i]].infected_t - temp)
                            self.nodes[order[i]].set_in_hospital(self.nodes[order[i]].in_hospital + temp)
                            bed_num = bed_num - temp
                            Ih_new += temp
                            patient_num[order[i]] -= temp
                        else:
                            self.full = True
                            break

                elif sim_type == 'Policy':

                    bed_parameter = self.parameters[bed_action]
                    for i in range(self.region_num):
                        infect_num = patient_num_order/np.mean(patient_num_order)
                        order = np.argsort(-(bed_parameter[0] * infect_num + bed_parameter[1] * self.flow + bed_parameter[2] * self.dense))

                    alloc_bed = np.floor(bed_num*np.array(bed_distribute_perc))

                    if np.sum(alloc_bed) > 0:
                        for i in range(len(bed_distribute_perc)):
                            temp=min(alloc_bed[i],patient_num[order[i]])
                            self.nodes[order[i]].set_infected_t(self.nodes[order[i]].infected_t - temp)
                            self.nodes[order[i]].set_in_hospital(self.nodes[order[i]].in_hospital + temp)
                            bed_num = bed_num - temp
                            Ih_new += temp
                            patient_num[order[i]] -= temp

                    if bed_num > 0:
                        temp_order = np.argsort(-np.array(bed_distribute_perc))
                        for i in range(len(bed_distribute_perc)):
                            temp=min(bed_num,patient_num[order[temp_order[i]]])
                            self.nodes[order[temp_order[i]]].set_infected_t(self.nodes[order[temp_order[i]]].infected_t - temp)
                            self.nodes[order[temp_order[i]]].set_in_hospital(self.nodes[order[temp_order[i]]].in_hospital + temp)
                            bed_num = bed_num - temp
                            Ih_new += temp
                            patient_num[order[i]] -= temp
                            if bed_num == 0:
                                break

                        if bed_num > 0:
                            patient_list = list()
                            for i in range(self.region_num):
                                patient_list.extend([i]*int(patient_num[i]))
                            np.random.shuffle(patient_list)

                            for i in range(len(patient_list)):
                                if bed_num > 0:
                                    self.nodes[patient_list[i]].set_infected_t(self.nodes[patient_list[i]].infected_t - 1)
                                    self.nodes[patient_list[i]].set_in_hospital(self.nodes[patient_list[i]].in_hospital + 1)
                                    bed_num = bed_num-1
                                    Ih_new += 1
                                else:
                                    self.full = True
                                    break
                        else:
                            self.full = True

                elif sim_type == 'Policy_a':
                    bed_parameter = self.parameters[bed_action]
                    infect_num = patient_num_order/np.mean(patient_num_order)
                    order = np.argsort(-(bed_parameter[0] * infect_num + bed_parameter[1] * self.flow + bed_parameter[2] * self.dense))

                    for i in range(self.region_num):
                        temp = min(bed_num, int(patient_num[order[i]] * bed_satisfy_perc))
                        self.nodes[order[i]].set_infected_t(self.nodes[order[i]].infected_t - temp)
                        self.nodes[order[i]].set_in_hospital(self.nodes[order[i]].in_hospital + temp)
                        bed_num = bed_num - temp
                        Ih_new += temp
                        patient_num[order[i]] -= temp
                        if bed_num == 0:
                            self.full = True
                            break

            else:
                self.full = True

            #cross region traveling
            for k in range(self.region_num):
                S_temp[k] = np.random.multinomial(
                    self.nodes[k].susceptible, self.prob[((self.START-1)*48+time) % (7*48)][k])
                L_temp[k] = np.random.multinomial(
                    self.nodes[k].latent, self.prob[((self.START-1)*48+time) % (7*48)][k])
                Iut_temp[k] = np.random.multinomial(
                    self.nodes[k].infected_ut, self.prob[((self.START - 1)* 48 + time) % (7* 48)][k])
                if it_move:
                    It_temp[k] = np.random.multinomial(
                        self.nodes[k].infected_t, self.prob[((self.START-1)*48+time) % (7*48)][k])
                Ia_temp[k] = np.random.multinomial(
                    self.nodes[k].infected_asymptomatic, self.prob[((self.START - 1)* 48 + time) % (7* 48)][k])
                R_temp[k] = np.random.multinomial(
                    self.nodes[k].recovered, self.prob[((self.START - 1)* 48 + time) % (7* 48)][k])

            S_temp_sum0 = np.sum(S_temp, axis=0)
            L_temp_sum0 = np.sum(L_temp, axis=0)
            Iut_temp_sum0 = np.sum(Iut_temp, axis=0)
            if it_move:
                It_temp_sum0 = np.sum(It_temp, axis=0)
            Ia_temp_sum0 = np.sum(Ia_temp, axis=0)
            R_temp_sum0 = np.sum(R_temp, axis=0)
            S_temp_sum1 = np.sum(S_temp, axis=1)
            L_temp_sum1 = np.sum(L_temp, axis=1)
            Iut_temp_sum1 = np.sum(Iut_temp, axis=1)
            if it_move:
                It_temp_sum1 = np.sum(It_temp, axis=1)
            Ia_temp_sum1 = np.sum(Ia_temp, axis=1)
            R_temp_sum1 = np.sum(R_temp, axis=1)

            for k in range(self.region_num):
                self.nodes[k].set_susceptible(
                    self.nodes[k].susceptible+S_temp_sum0[k]-S_temp_sum1[k]+S_temp[k][self.region_num])
                self.nodes[k].set_latent(
                    self.nodes[k].latent+L_temp_sum0[k]-L_temp_sum1[k]+L_temp[k][self.region_num])
                self.nodes[k].set_infected_ut(
                    self.nodes[k].infected_ut + Iut_temp_sum0[k] - Iut_temp_sum1[k] + Iut_temp[k][self.region_num])
                if it_move:
                    self.nodes[k].set_infected_t(
                        self.nodes[k].infected_t+It_temp_sum0[k]-It_temp_sum1[k]+It_temp[k][self.region_num])
                self.nodes[k].set_infected_asymptomatic(
                    self.nodes[k].infected_asymptomatic+Ia_temp_sum0[k]-Ia_temp_sum1[k]+Ia_temp[k][self.region_num])
                self.nodes[k].set_recovered(
                    self.nodes[k].recovered + R_temp_sum0[k] - R_temp_sum1[k] + R_temp[k][self.region_num])

            if(is_save):
                save = list()
                for i in range(self.region_num):
                    temp1 = [self.nodes[i].susceptible, self.nodes[i].latent, self.nodes[i].infected_ut, self.nodes[i].infected_t, self.nodes[i].infected_asymptomatic, self.nodes[i].in_hospital, self.nodes[i].recovered, self.nodes[i].death]
                    save.append(temp1)
                save = np.array(save)
                save = save.astype(np.float)
                with open(os.path.join('../result',self.city,'result_'+str(save_time*interval+time)+'.json'), 'w') as f:
                    json.dump(save.tolist(), f)

        next_state = list()
        for i in range(self.region_num):
            next_state.append([self.nodes[i].susceptible, self.nodes[i].latent, self.nodes[i].infected_ut, self.nodes[i].infected_t, self.nodes[i].infected_asymptomatic, self.nodes[i].in_hospital, self.nodes[i].recovered, self.nodes[i].death])

        a = self.statistic()
        print('S:{},L:{},Iut:{},It:{},Ia:{},Ih:{},R:{},D:{}'.format(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]))
        print('Is Full:{}'.format(self.full))

        return np.array(next_state), Ih_new