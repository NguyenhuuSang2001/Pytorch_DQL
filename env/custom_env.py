import numpy as np
import random

class ActionSpace(object):
    def __init__(self, number_action):
        self.n = number_action
    def sample(self):
        return random.randint(0, self.n - 1)

class IoTCommunicationEnv():
    def __init__(self, num_user, num_level_power, max_power, num_channel, W_U=10**6, max_step=500):

        self.num_user = num_user
        self.num_level_power = num_level_power
        self.max_power = max_power
        self.num_channel = num_channel

        self.W_U = W_U
        self.GAMMA = [0.0025118864, 0.0039810717, 0.0063095734]
        self.CHANNEL_GAIN = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.d_k = 20
        self.noise_variance = 10**(-17) 

        self.action_space = ActionSpace(self.size_action_space())

        self.state = None 
        self.max_step = max_step
        self.count_step = 0
    
    def size_action_space(self):
        return ((self.num_channel + 1) ** self.num_user)*( (self.num_level_power + 1)**self.num_user)

    def state_size(self):
        return self.num_user + 1

    def reset(self):
        # Initialize the environment
        channel_gains = [np.random.choice(self.CHANNEL_GAIN) for _ in range(self.num_user)]
        gamma_i =  np.random.choice(self.GAMMA)
        channel_gains.append(gamma_i)
        self.state = channel_gains
        return self.state, {}

    def step(self, action):
        done = False
        # self.count_step += 1
        # if self.count_step > self.max_step:
        #     done = True
        #     self.count_step = 0
        
        # Simulate the communication process and calculate reward
        reward = self._calculate_reward(action)
        
        if reward < 0:
            done = True

        # Simulate state transition
        # new_channel_gains = [np.random.normal(0.5, 0.5) for _ in range(self.num_user)]
        new_channel_gains = [np.random.choice(self.CHANNEL_GAIN) for _ in range(self.num_user)]
        gamma_i =  np.random.choice(self.GAMMA)
        new_channel_gains.append(gamma_i)
        self.state = new_channel_gains

        truncated = done
        terminated = done
        
        return self.state, reward, terminated, truncated, {}
    
    def convert_action(self, action_index):
        
        num_level_powers = action_index%((self.num_level_power + 1) ** self.num_user)

        num_channels = action_index //((self.num_level_power + 1) ** self.num_user)

        
        channel_allocation = self.convert_base(num_channels, self.num_channel + 1, self.num_user)
        power_allocation = self.convert_base(num_level_powers, self.num_level_power + 1, self.num_user)
        
        return np.array(channel_allocation + power_allocation)

    def convert_base(self, number, base, size = 1):
        if number == 0:
            return [0]*size
        
        digits = []
        negative = False
        
        if number < 0:
            negative = True
            number = abs(number)
        
        while number > 0:
            remainder = number % base
            digits.append(int(remainder))
            number //= base

        if size > len(digits):
            for _ in range(size - len(digits)):
                digits.append(0)
        
        if negative:
            digits.append("-")

        digits = digits[::-1]

        return digits

    def _calculate_reward(self, action_index):
        action = self.convert_action(action_index)
        channel_allocation = action[:self.num_user ]
        power_levels = action[self.num_user:]

        reward = 0.0

        channel_select_uniq = []
        check_channael = 0

        for channel in channel_allocation:
            if channel != 0:
                if channel in channel_select_uniq:
                    check_channael -= 1
                else: 
                    channel_select_uniq.append(channel)
        if check_channael < 0:
            reward += check_channael * 5
        
        check_power = 0
        for k in range(self.num_user):
            if channel_allocation[k] != 0 and power_levels[k]==0 :
                check_power -= 1
        if check_power < 0:
            reward += check_power * 2

        if check_channael < 0 or check_power < 0:
            # print("reward: ", reward)
            return reward

        for k in range(self.num_user):
            channel_index = int(channel_allocation[k])
            if channel_index == 0:
                continue
            level_power = power_levels[k]
            W_U = self.W_U
            channel_gain = self.state[k]
            d_k = self.d_k # Khoang cach
            interference = self.state[-1]  # Gamma_i parameter
            noise_variance = self.noise_variance # Adjust this based on your scenario
            
            data_rate = 1.0 * level_power / self.num_level_power * self.max_power * channel_gain * d_k**(-2)/ (interference + W_U * noise_variance)
            reward += W_U * np.log2(1 + data_rate)

            reward /= 10**3

        return reward


def env_example():
    # Example usage
    num_user = 4
    num_level_power = 3
    max_power = 0.0316227766
    num_channel = 4

    env = IoTCommunicationEnv(num_user, num_level_power, max_power, num_channel)

    return env


if __name__ == "__main__":
    # Reset the environment
    env = env_example()
    initial_state = env.reset()
    print("Initial State:", initial_state)
    print("action space: ",env.size_action_space())

    # action = random.randint(0, env.action_space.n)
    action = 159998
    print("index action: {} => {}".format(action, env.convert_action(action)))
    new_state, reward, done, info = env.step(action)
    print("New State:", new_state)
    print("Reward:", reward)