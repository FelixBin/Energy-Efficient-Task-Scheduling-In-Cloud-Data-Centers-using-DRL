import numpy as np
import matplotlib.pyplot as plt
import time
from Myenvironment1 import SchedulingEnvironment
from Agent import DQN
from AgentOthers import baselines


performance_lamda = []
EPISODE = 1
policyNum = 4
policyName = ['random', 'round-robin', 'earliest', 'DQN']
start_learn = 500  # DQN parameter
learn_interval = 1   # DQN parameter
global_step = 0  # DQN parameter
environment = SchedulingEnvironment()
brainRL = DQN(environment.actionNum, environment.s_features)
brainOthers = baselines(environment.actionNum, environment.VMtypes)

t_start = time.time()


for episode in range(EPISODE):
    print('----------------------------Episode', episode, '----------------------------')
    job_c = 1  # job counter
    performance_c = 0
    performance_c_time = 1  # counter for getting performance according to time
    performance_showT = 10
    environment.reset()  
    performance_respTs = []
    performance_successes = []

    while True:
        global_step += 1
        finish, job_attrs = environment.workload(job_c)  # job_attrs = [id, arrival_time, length, type, ddl]
        # DQN policy
        DQN_state = environment.getState(job_attrs, 4)  # job type, VM wait time
        print('step:', global_step, '  DQN_state:', DQN_state, end='')
        if global_step != 1:  # store transition
            brainRL.store_transition(last_state, last_action, last_reward, DQN_state)
        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        Energy, reward_DQN = environment.feedback(job_attrs, action_DQN, 4)
        
        if (global_step > start_learn) and (global_step % learn_interval == 0):  # learn
            brainRL.learn()
        print('Energy:', Energy)
        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN

        # random policy
        action_random = brainOthers.random_choose_action()
        reward_random = environment.feedback(job_attrs, action_random, 1)
        # round robin policy
        action_RR = brainOthers.RR_choose_action(job_c)
        reward_RR = environment.feedback(job_attrs, action_RR, 2)
        # earliest policy
        idleTimes = environment.get_VM_idleT(3)  # get VM state
        action_early = brainOthers.early_choose_action(idleTimes)
        reward_early = environment.feedback(job_attrs, action_early, 3)
        
        # choice 1: get performance according to time
        if job_attrs[1] >= performance_c_time * performance_showT:
            avg_respTs = environment.get_responseTs(policyNum, performance_c, job_c-1)
            performance_respTs.append(avg_respTs)
            successTs = environment.get_successTimes(policyNum, performance_c, job_c-1)
            performance_successes.append(successTs)
            performance_c = job_c - 1
            performance_c_time += 1
        if finish:
            avg_respTs = environment.get_responseTs(policyNum, performance_c, job_c)
            performance_respTs.append(avg_respTs)
            successTs = environment.get_successTimes(policyNum, performance_c, job_c)
            performance_successes.append(successTs)


        job_c += 1
        if finish:
            break

    # episode performance
    startP = 2000
    total_Rewards = environment.get_totalRewards(policyNum, startP)
    avg_allRespTs = environment.get_total_responseTs(policyNum, startP)
    total_success = environment.get_totalSuccess(policyNum, startP)
    avg_util = environment.get_avgUtilitizationRate(policyNum, startP)
    total_Ts = environment.get_totalTimes(policyNum, startP)
    # JobDistribution = environment.get_JobDistribution(policyNum)

    print('total performance (after 2000 jobs):')
    print('[random policy] reward:', total_Rewards[0],
          'success_rate:', total_success[0], ' utilizationRate:', avg_util[0], ' finishT:', total_Ts[0])
    print('[RR policy] reward:', total_Rewards[1],
          'success_rate:', total_success[1], ' utilizationRate:', avg_util[1], ' finishT:', total_Ts[1])
    print('[earliest policy] reward:', total_Rewards[2],
          'success_rate:', total_success[2], ' utilizationRate:', avg_util[2], ' finishT:', total_Ts[2])
    print('[DQN policy] reward:', total_Rewards[3],
          'success_rate:', total_success[3], ' utilizationRate:', avg_util[3], ' finishT:', total_Ts[3])
          
          
    
    startP = 5000
    total_Rewards = environment.get_totalRewards(policyNum, startP)
    avg_allRespTs = environment.get_total_responseTs(policyNum, startP)
    total_success = environment.get_totalSuccess(policyNum, startP)
    avg_util = environment.get_avgUtilitizationRate(policyNum, startP)
    total_Ts = environment.get_totalTimes(policyNum, startP)
    # JobDistribution = environment.get_JobDistribution(policyNum)

    print('total performance (after 5000 jobs):')
    print('[random policy] reward:', total_Rewards[0],
          'success_rate:', total_success[0], ' utilizationRate:', avg_util[0], ' finishT:', total_Ts[0])
    print('[RR policy] reward:', total_Rewards[1],
          'success_rate:', total_success[1], ' utilizationRate:', avg_util[1], ' finishT:', total_Ts[1])
    print('[earliest policy] reward:', total_Rewards[2],
          'success_rate:', total_success[2], ' utilizationRate:', avg_util[2], ' finishT:', total_Ts[2])
    print('[DQN policy] reward:', total_Rewards[3],
          'success_rate:', total_success[3], ' utilizationRate:', avg_util[3], ' finishT:', total_Ts[3])
    
    
    startP = 10000
    total_Rewards = environment.get_totalRewards(policyNum, startP)
    avg_allRespTs = environment.get_total_responseTs(policyNum, startP)
    total_success = environment.get_totalSuccess(policyNum, startP)
    avg_util = environment.get_avgUtilitizationRate(policyNum, startP)
    total_Ts = environment.get_totalTimes(policyNum, startP)
    # JobDistribution = environment.get_JobDistribution(policyNum)

    print('total performance (after 10000 jobs):')
    print('[random policy] reward:', total_Rewards[0],
          'success_rate:', total_success[0], ' utilizationRate:', avg_util[0], ' finishT:', total_Ts[0])
    print('[RR policy] reward:', total_Rewards[1],
          'success_rate:', total_success[1], ' utilizationRate:', avg_util[1], ' finishT:', total_Ts[1])
    print('[earliest policy] reward:', total_Rewards[2],
          'success_rate:', total_success[2], ' utilizationRate:', avg_util[2], ' finishT:', total_Ts[2])
    print('[DQN policy] reward:', total_Rewards[3],
          'success_rate:', total_success[3], ' utilizationRate:', avg_util[3], ' finishT:', total_Ts[3])
    
t_end = time.time()
timer = round(t_end - t_start, 2)
print('\n', 'timer:', timer, 's')


# pic 1: the successRate in one episode (line pic)
draw_success = np.array(performance_successes) * 100
draw_success = np.around(draw_success, 1)
x = range(draw_success.shape[0])
lables =['s-', '^-', 'o-', 'd-', '*-', 'p-']
plt.figure()
for i in range(policyNum):
    y = draw_success[:, i]
    pn = policyName[i]
    la = lables[i]
    plt.plot(x, y, la, label=pn)

plt.xlabel('time')
plt.ylabel('success rate(%)')
plt.legend(loc='best')  # add legend

# x sticks
x_sticks = np.linspace(0, draw_success.shape[0] - 1, draw_success.shape[0])

x_sticks_names = np.linspace(1*performance_showT, draw_success.shape[0]*performance_showT, draw_success.shape[0])
x_sticks_names = x_sticks_names.astype(int)
plt.xticks(x_sticks, x_sticks_names)

plt.grid(True, linestyle="-.", linewidth=1)
plt.title('success rate')
plt.show()

