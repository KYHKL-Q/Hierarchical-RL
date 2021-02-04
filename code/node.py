import numpy as np
Pa = 0.018  # false negative rate 
eps_1 = 250  # average length of incubation (30min)
eps_1_d = 5.2  # average length of incubation (day)
d = 0.15  # death rate
t = 672  # average recovery time (30min)
t_d = 14  # average recovery time (day)
d_hospital = 0.04  # death rate in hospital
t_hospital = 614  # average recovery time in hospital(30min)
t_hospital_d = 12.8  # average recovery time in hospital(day)
R_0 = 2.68  # basic reproduction number
r_a = 0.6
r_L = 1.0
eps = 1/eps_1
beta = R_0 / (r_L * eps_1_d + (Pa * r_a + (1 - Pa)) * t_d)
beta = np.power(1+beta, 1/48)-1
L_I = eps * (1 - Pa)
L_Ia = eps * Pa
I_D = d / t
I_R = ((1 - d)/(t_d - d))/48

theta = 48 # testing speed (30min)
I_h = 1/theta
Ia_R = 1 / t
Ih_D = d_hospital / t_hospital
Ih_R = ((1 - d_hospital) / (t_hospital_d - d_hospital))/48

class node:
    def __init__(self, id):
        self.id = id
        self.susceptible = 0
        self.infected_ut = 0
        self.infected_t = 0
        self.death = 0
        self.in_hospital = 0
        self.infected_asymptomatic = 0
        self.recovered = 0
        self.latent = 0

    def set_susceptible(self, susceptible):
        self.susceptible = susceptible

    def set_latent(self, latent):
        self.latent = latent

    def set_infected_ut(self, infected_ut):
        self.infected_ut = infected_ut

    def set_infected_t(self, infected_t):
        self.infected_t = infected_t

    def set_infected_asymptomatic(self, infected_asymptomatic):
        self.infected_asymptomatic = infected_asymptomatic

    def set_death(self, death):
        self.death = death

    def set_in_hospital(self, in_hospital):
        self.in_hospital = in_hospital

    def set_recovered(self, recovered):
        self.recovered = recovered

    def step(self,mask_number,mask_quality):
        if (self.susceptible + self.latent + self.infected_ut + self.infected_t + self.infected_asymptomatic + self.in_hospital + self.recovered > 0):
            mask_factor = 1 - np.clip(mask_number / (self.susceptible + self.latent + self.infected_ut + self.infected_t + self.infected_asymptomatic + self.in_hospital + self.recovered), 0, 1) * mask_quality
            lambda_j = ((self.infected_ut +self.infected_t + self.infected_asymptomatic * r_a + self.latent * r_L) / (self.susceptible + self.latent + self.infected_ut + self.infected_t + self.infected_asymptomatic + self.in_hospital + self.recovered)) * beta * mask_factor
            susceptible_to_latent, __ = np.random.multinomial(self.susceptible, [lambda_j, 1])
            self.susceptible -= susceptible_to_latent
            self.latent += susceptible_to_latent

            latent_to_infected, latent_to_Ia, __ = np.random.multinomial(self.latent, [L_I, L_Ia, 1])
            self.infected_ut += latent_to_infected
            self.infected_asymptomatic += latent_to_Ia
            self.latent -= (latent_to_Ia + latent_to_infected)

            prob = I_h

            infected_ut_to_t, __ = np.random.multinomial(self.infected_ut, [prob, 1])
            self.infected_ut -= infected_ut_to_t
            self.infected_t += infected_ut_to_t

            infected_to_death, infected_to_recovered, __ = np.random.multinomial(self.infected_ut, [I_D, I_R, 1])
            self.death += infected_to_death
            self.recovered += infected_to_recovered
            self.infected_ut -= (infected_to_death + infected_to_recovered)
            infected_to_death, infected_to_recovered, __ = np.random.multinomial(self.infected_t, [I_D, I_R, 1])
            self.death += infected_to_death
            self.recovered += infected_to_recovered
            self.infected_t -= (infected_to_death + infected_to_recovered)

            Ia_to_recovered, __ = np.random.multinomial(self.infected_asymptomatic, [Ia_R, 1])
            self.recovered += Ia_to_recovered
            self.infected_asymptomatic -= Ia_to_recovered

            in_hospital_to_death, in_hospital_to_recovered, __ = np.random.multinomial(self.in_hospital, [Ih_D, Ih_R, 1])
            self.death += in_hospital_to_death
            self.recovered += in_hospital_to_recovered
            self.in_hospital -= (in_hospital_to_death + in_hospital_to_recovered)