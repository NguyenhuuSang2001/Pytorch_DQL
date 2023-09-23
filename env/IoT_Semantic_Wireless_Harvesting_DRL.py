import numpy as np
import random
from collections import Counter


np.random.seed(0)
random.seed(0)

class ActionSpace(object):
    def __init__(self, number_action):
        self.n = number_action
    def sample(self):
        return random.randint(0, self.n - 1)

def convert_base(number, base, length = 1):
        if number == 0:
            return [0]*length
        
        digits = []
        negative = False
        
        if number < 0:
            negative = True
            number = abs(number)
        
        while number > 0:
            remainder = number % base
            digits.append(int(remainder))
            number //= base

        if length > len(digits):
            for _ in range(length - len(digits)):
                digits.append(0)
        
        if negative:
            digits.append("-")

        digits = digits[::-1]

        return digits

class IoT_Semantic_Wireless_Harvesting():
    def __init__(self, N_IoT=3, num_channel=4, max_p_trans=7, num_level_p_trans=3, max_step=500):
        
        self.N = N_IoT
        self.num_level_p_trans = num_level_p_trans
        self.max_p_trans = max_p_trans
        self.num_channel = num_channel

        self.confidence_values = [ 0.3, 0.4, 0.5]

        self.E_capacity = np.array([random.randint(10, 15)]*self.N) # Capacity energy UAV
        self.comp_capability = np.array([2.4 * 10**9] * self.N)

        self.E_i0 = np.array([random.randint(1, 2) for _ in range(self.N)])
        self.h_i = np.array([min(0.1, random.random()) for _ in range(self.N)])
        self.d_i = np.array([random.randint(10, 20) for _ in range(self.N)])
        self.t_harv = np.array([3]*self.N)
        self.N_img = 4
        L = 1024*1096 #bits
        self.size_img = L
        self.kappa = 10**-28


        self.start = True
        self.initialization()
        self.action_space = ActionSpace(self._size_action_space())


    def initialization(self):
        # Initialize the environment
        self.e_0_default = np.array([random.randint(3, 10) for _ in range(self.N)])
        self.e_0 = self.e_0_default
        self.e_i = self._calculate_e_i()
        self.f_i = self._calculate_f_i()
        self.G_i = self._calculate_G_i()
        self.denta_t = self._calculate_denta_i()
        self.state = np.concatenate((self.e_i, self.f_i, self.G_i, self.denta_t))

    def _size_action_space(self):
        num_confidences = len(self.confidence_values)**self.N 
        num_channels = (self.num_channel+1)**self.N #add option 0
        num_p_trans = (self.num_level_p_trans+1)**self.N #add option 0

        return num_confidences * num_channels * num_p_trans

    def state_size(self):
        return len(self.state)
    
    def reset(self):
        self.start = True
        self.initialization()
        return self.state, {}


    def step(self, action):
        done = False
        
        # Simulate the communication process and calculate reward
        reward = self._calculate_reward(action)
        
        if reward < 0:
            done = True

        # Simulate state transition
        self.state = self._state_next()

        truncated, terminated = done, done
        
        return self.state, reward, terminated, truncated, {}
    
    def convert_action(self, action_index):
        # gamma1, gamma2, ..., alpha1, alpha2, ..., p_tran1, p_tran2, ..., p_tran_N
        Max_channels = (self.num_channel+1)**self.N
        Max_p_trans = (self.num_level_p_trans+1)**self.N

        ind = action_index

        num_p_trans = ind%Max_p_trans
        ind = ind // Max_p_trans
        num_channels = ind%Max_channels
        ind = ind // Max_channels
        num_gammas = ind
        
        list_p = convert_base(num_p_trans, self.num_level_p_trans+1, self.N )
        list_channel = convert_base(num_channels, self.num_channel+1, self.N)
        list_gamma = convert_base(num_gammas, len(self.confidence_values), self.N)

        action = list_gamma
        action.extend(list_channel)
        action.extend(list_p)

        return np.array(action)

# Ham tinh state
    def _state_next(self):

        self.e_i = self._calculate_e_i()

        self.f_i = self._calculate_f_i()

        self.G_i = self._calculate_G_i()

        self.denta_t = self._calculate_denta_i()
        
        self.state = np.concatenate((self.e_i, self.f_i, self.G_i, self.denta_t))

        return self.state

    def _calculate_e_i(self): 
        if self.start == False:
            self.e_0 = self._calculate_e_0()
        self.e_harv = self._calculate_e_harv()
        self.e_sens = self._calculate_e_sens()
        self.e_comp = self._calculate_e_comp()
        self.start = False

        e = np.column_stack([self.e_0 + self.e_harv, self.E_capacity])
        e = np.min(e, axis=1)

        e_i = e - self.e_sens - self.e_comp

        return e_i


    def _calculate_f_i(self):
        self.f_mean = 10**6
        self.f_variance = 0.05*self.f_mean

        f_i = np.array([int(np.random.normal(self.f_mean, self.f_variance)) for _ in range(self.N)])
        
        return f_i

    def _calculate_denta_i(self):
        T = np.array([11]*self.N)

        self.t_comp =  self._calculate_t_comp()

        denta_t = T - (self.t_harv + self.t_sens + self.t_comp)
        
        return denta_t
    
    def _calculate_G_i(self):
        G_i = np.array([0]*self.N)
        for i in range(self.N):
            c_ij = random.randint(800, 2500)
            for j in range(self.N_img):
                G_i[i] += c_ij

        return G_i

# Ham phu thuoc e
    def _calculate_e_0(self):
        e_0 = self.e_i - self.e_select - self.e_trans

        return e_0
    
    def _calculate_e_sens(self):
        self.t_sens = np.array([2]*self.N)

        e_consumed_sense_bit = 50*10**-9
        sensing_rate = 2

        e_sens = e_consumed_sense_bit * sensing_rate * self.size_img * self.t_sens

        return e_sens

    def _calculate_e_comp(self):
        r_CPU = 6000*self.size_img 

        e_comp = self.kappa * r_CPU * (self.comp_capability**2)

        return e_comp

    def _calculate_e_harv(self):
        P_harv = self._calculate_P_harv()

        e_harv = P_harv * self.t_harv

        return e_harv

    def _calculate_e_select(self):
        C_j1 = np.array([0.3]*self.N)
        C_j2 = np.array([0.05]*self.N)
        C_j3 = 0.2

        r_CPU = C_j1*(self.rho - C_j2)**C_j3

        e_select = self.kappa * r_CPU * (self.comp_capability**2)

        return e_select

    def _calculate_e_trans(self):
        e_trans = self.p_trans * self.t_trans

        return e_trans

    def _calculate_P_harv(self):
        xi = 0.8
        P = 10 #40 dBm -> 10 W
        alpha = 0.1

        P_harv = xi * P * self.h_i * self.d_i**(-alpha)

        return P_harv


#Ham phu thuoc t
    def _calculate_R_i(self):
        W = 10**6
        sigma_2 = 10**-17

        # print("p_trans: ", self.p_trans)
        # print("h_i: ", self.h_i)


        R_i = W*np.log2(1+(self.p_trans*self.h_i)/(W*sigma_2))

        return R_i

# ------
    def _calculate_t_trans(self):
        self.Q_i = np.array([random.randint(8, 15)*10**3 for _ in range(self.N)])
        R_i = self._calculate_R_i()
        # print("R_i:", R_i)

    # ---- fix div 0
        mask = R_i==0
        R_i[mask] = 10**20

        t_trans = self.Q_i / R_i
        

        # print("t_trans: ", t_trans)
        return t_trans

# ------
    def _calculate_t_comp(self):
        t_comp = np.array([random.randint(3, 6) for _ in range(self.N)])

        return t_comp
    
    def _calculate_t_select(self):
        t_select = np.array([random.random() for _ in range(self.N)])

        return t_select


#Ham tinh reward
    def _calculate_reward(self, action_index):
        action = self.convert_action(action_index)
        # gamma1, gamma2, ..., alpha1, alpha2, ..., p_tran1, p_tran2, ..., p_tran_N

        gamma = action[:self.N]
        alpha = action[self.N:2*self.N]
        p = action[2*self.N:]

        self.p_trans = self.max_p_trans/self.num_level_p_trans * p 
        self.rho = np.array([1-self.confidence_values[i] for i in gamma])

        self.t_select = self._calculate_t_select()
        self.t_trans = self._calculate_t_trans()

        self.e_select = self._calculate_e_select()
        self.e_trans = self._calculate_e_trans()



        # print("time out: ", self.denta_t - self.t_select - self.t_trans)
        
        frequency_counter = Counter(alpha)
        select_repeat = 0
        for num, frequency in frequency_counter.items():
            if num > 0 and frequency > 1:
                select_repeat += frequency - 1

        if select_repeat > 0:
            reward = -select_repeat*0.5
            return reward
        
        alloc_energy_err = 0
        for i in range(len(alpha)):
            if alpha[i] > 0 and p[i] == 0:
                alloc_energy_err += 1
        
        if alloc_energy_err > 0:
            reward = -alloc_energy_err * 0.5
            return reward



        flag = [True, True]
        for i in range(self.N):
            if alpha[i] > 0 and p[i] > 0:
                if  self.denta_t[i] - self.t_select[i] - self.t_trans[i] < 0:
                    flag[0] = False
                    break
                if self.e_i[i] - self.e_select[i] - self.e_trans[i] < self.E_i0[i]:
                    flag[1] = False
                    break
        
        sum_rho = 0
        sum_time_use = 0
        sum_t_delta = 0
        for i in range(self.N):
            if alpha[i] > 0:
                sum_rho += self.rho[i]
                sum_time_use += self.t_trans[i] + self.t_select[i]
            sum_t_delta += self.denta_t[i]


        k0, k1, k2 = 5, 1, 10
        if sum_time_use > 0:
            if flag[0] and flag[1]:
                # print("case 1")
                reward = k0*self.N*self.N_img*(1- np.exp(-k1*sum_rho/self.N)) + k2*self.N*min(1.0, sum_time_use/sum_t_delta)
            elif flag[0] and not flag[1]:
                # print("case 2")
                k3, k4, k5 = 1, 1, 1
                reward = -k3*np.sum(np.sqrt((self.e_i - self.e_select - self.e_trans  - self.E_i0)**2)) - k4*np.sqrt((sum_t_delta - sum_time_use)**2) - k5
                reward *= 0.01
            else:
                reward = -0.25/8*k0
        else:
            reward = -0.5/8*k0

        return reward

#------------
    def render(self):
        print("State: \ne_i:", self.e_i)
        print("f_i:", self.f_i)
        print("G_i:", self.G_i)
        print("denta_t:", self.denta_t)



def env_example():
    # Example usage

    env = IoT_Semantic_Wireless_Harvesting()

    return env


if __name__ == "__main__":
    # Reset the environment
    env = env_example()
    initial_state, _ = env.reset()
    print("Initial State:")
    env.render()
    print("action space: ",env.action_space.n)

    action = random.randint(0, env.action_space.n)
    # action = 159998
    print("index action: {} => {}".format(action, env.convert_action(action)))
    new_state, reward, done, _, info = env.step(action)
    print("New State:")
    env.render()
    print("Reward:", reward)