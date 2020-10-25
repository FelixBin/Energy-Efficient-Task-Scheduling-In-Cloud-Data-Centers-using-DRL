import numpy as np
np.random.seed(2)

class baselines:
    def __init__(self, n_actions, VMtypes):
        self.n_actions = n_actions
        self.VMtypes = np.array(VMtypes)  # change list to numpy
        # parameters for sensible policy
        '''
        self.sensible_updateT = 5
        self.sensible_counterT = 1
        self.sensible_discount = 0.7  # 0.7 is best, 0.5 and 0.6 OK
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros((2, self.n_actions))  # row 1: jobNum   row 2: sum duration
'''
    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action

    def RR_choose_action(self, job_count):  # round robin policy
        action = (job_count-1) % self.n_actions
        return action

    def early_choose_action(self, idleTs):  # earliest policy
        action = np.argmin(idleTs)
        return action

        '''
    def suit_choose_action(self, attrs):  # suitable policy--random
        jobType = attrs[3]
        while True:
            action = np.random.randint(0, len(self.VMtypes))
            if self.VMtypes[action] == jobType:
                break
        return action
    
    def suit_choose_action(self, attrs):  # suitable policy--best
        jobType = attrs[0]  # e.g. 1
        idleTimes = attrs[1:len(attrs)]
        judge = np.argwhere(self.VMtypes == jobType)  # e.g. [[5],[6],[7],[8],[9]]
        judgeF = judge.reshape((len(judge)))  # e.g. [5,6,7,8,9]
        idleTimes_suit = [idleTimes[w] for w in judgeF]  # e.g. [0.2, 0.1, 0.3, 0.5, 1]
        id = idleTimes_suit.index(min(idleTimes_suit))  # e.g. 1
        action = judgeF[id]  # 6
    
        print('jobType:', jobType)
        print(judgeF)
        print(idleTimes)
        print(idleTimes_suit)
        print(action)
        
        return action

    def sensible_choose_action(self, arrivalT):  # sensible routing policy
        # if need update prob

        if arrivalT >= self.sensible_updateT * self.sensible_counterT:
            # temp_W = self.sensible_sumDurations[1, :] / self.sensible_sumDurations[0, :]
            temp_W = self.sensible_sumDurations[1, :]
            where_are_inf = np.isinf(temp_W)  # if no job on some VMs, set their duraiton = 0
            where_are_nan = np.isnan(temp_W)
            temp_W[where_are_inf] = 0
            temp_W[where_are_nan] = 0
            # update prob
            self.sensible_W = (1-self.sensible_discount)*self.sensible_W + self.sensible_discount*temp_W
            sensible_W_temp = 1/self.sensible_W
            where_are_inf = np.isinf(sensible_W_temp)  # if no job on some VMs, set their duraiton = 0
            where_are_nan = np.isnan(sensible_W_temp)
            sensible_W_temp[where_are_inf] = 0
            sensible_W_temp[where_are_nan] = 0
            self.sensible_probs = sensible_W_temp/sum(sensible_W_temp)
            self.sensible_probsCumsum = self.sensible_probs.cumsum()
            # print('111111111111:', arrivalT, '  ', np.around(self.sensible_probsCumsum, 3))

            # initial paras
            self.sensible_counterT += 1
            self.sensible_sumDurations = np.zeros((2, self.n_actions))

        # choose action
        pro = np.random.uniform()
        action = 0
        for i in range(self.n_actions):
            if pro < self.sensible_probsCumsum[i]:
                break
            else:
                action += 1
        return action

    def sensible_counter(self, duration, VMid):
        self.sensible_sumDurations[0, VMid] += 1
        self.sensible_sumDurations[1, VMid] += duration

    def sensible_reset(self):
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros((2, self.n_actions))
        self.sensible_counterT = 1
        '''