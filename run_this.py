from RL_brain import DeepQNetwork
import numpy as np
import tensorflow as tf

np.random.seed(6)

actions=np.array([[0,0],[0,0.1],[0,0.2],[0,0.3],[0,0.4],[0,0.5],[0,0.6],[0,0.7],[0,0.8],[0,0.9],[0,1],
                  [1, 0], [1, 0.1], [1, 0.2], [1, 0.3], [1, 0.4], [1, 0.5], [1, 0.6], [1, 0.7], [1, 0.8], [1, 0.9],
                  [1, 1],
                  [2, 0], [2, 0.1], [2, 0.2], [2, 0.3], [2, 0.4], [2, 0.5], [2, 0.6], [2, 0.7], [2, 0.8], [2, 0.9],
                  [2, 1],
                  [3, 0], [3, 0.1], [3, 0.2], [3, 0.3], [3, 0.4], [3, 0.5], [3, 0.6], [3, 0.7], [3, 0.8], [3, 0.9],
                  [3, 1]])


n_actions = len(actions)
n_features=14
lam_local,beta_local,cycle_perbyte,energy_per_l= 0.6,0.4,1,6
lam_re,beta_re,energy_per_r = 0.8,0.2,0.3
local_core_max,local_core_min=200,50
server_core_max,server_core_min=400,150
uplink_max,uplink_min = 350,100
downlink_max,downlink_min = 600,250


def reset():
    np.random.seed(np.random.randint(1,1000))
    workload = np.random.randint(2000,3000)#定义工作量
    local_comp = np.random.randint(90,110)#定义本地可用计算资源
    uplink = np.array([np.random.randint(150,200),np.random.randint(150,200),
                            np.random.randint(150,200),np.random.randint(150,200)])#定义初始上行链路容量
    downlink = np.array([np.random.randint(300,500),np.random.randint(300,500),
                         np.random.randint(300,500),np.random.randint(300,500)])#定义下行链路容量
    servers_cap = np.array([np.random.randint(200,300),np.random.randint(200,300),
                            np.random.randint(200,300),np.random.randint(200,300)])#定义服务器的可用计算资源，服务器数量为4
    observation=np.array([workload,local_comp])
    return np.hstack((observation,servers_cap,uplink,downlink))

def mec_step(observation,action,time1):
    workload,local_comp,servers_cap,uplink,downlink= \
        observation[0],observation[1],observation[2:6],observation[6:10],observation[10:14]
    target_server,percen = int(action[0]),action[1]

    #贪心算法，每次选择可用计算资源最多的服务器
    MAX_c = max(servers_cap)

    # wait_local  = (local_core_max-local_comp)*0.1
    # wait_server = (np.array([server_core_max,server_core_max,server_core_max,server_core_max])-servers_cap)*0.01
    wait_local,wait_server = 2,1
    local_cost = lam_local*workload*cycle_perbyte*(1-percen)/(local_comp)+beta_local*workload*energy_per_l*(1-percen)+lam_local*wait_local

    local_only = lam_local*workload*cycle_perbyte/(local_comp)+beta_local*workload*energy_per_l+lam_local*wait_local

    remote_only = workload * lam_re * (cycle_perbyte  / (servers_cap[target_server]) +
                                       percen / uplink[target_server] + 0.01 / downlink[target_server]) + lam_re * wait_server + \
                  beta_re * energy_per_r * workload

    remote_cost = workload * lam_re * ((cycle_perbyte * percen) / (servers_cap[target_server])+
                percen / uplink[target_server] + (percen * 0.01) / downlink[target_server]) + lam_re * wait_server + \
                 beta_re * energy_per_r * workload * percen

    total_cost = workload * lam_local * ((cycle_perbyte * (1 - percen)) / (local_comp) +
                beta_local * energy_per_l * (1 - percen)) + lam_local * wait_local + \
                 workload * lam_re * ((cycle_perbyte * percen) / (servers_cap[target_server])+
                percen / uplink + (percen * 0.01) / downlink) + lam_re * wait_server + \
                 beta_re * energy_per_r * workload * percen

    total_cost_ = local_cost+remote_cost
    reward = -total_cost_
    # reward = (local_only-total_cost_)/local_only
    np.random.seed(np.random.randint(1,1000))

    #建立下一个过程的模拟生成
    a = np.random.uniform()
    b=0.9
    if (time1>=0) and (time1<=36):
        if (a>b) :
            local_comp = min(local_comp+np.random.randint(0,6),local_core_max)
            for i in range(4):
                servers_cap[i] = min(servers_cap[i] + np.random.randint(0, 15), server_core_max)
                downlink[i] = min(downlink[i]+np.random.randint(0,8),downlink_max)
                uplink[i] = min(uplink[i]+np.random.randint(0,5),uplink_max)

        else:
            local_comp = max(local_comp+np.random.randint(-5,0),local_core_min)
            for i in range(4):
                servers_cap[i] = max(servers_cap[i] + np.random.randint(-14, 0), server_core_min)
                downlink[i] = max(downlink[i] - np.random.randint(0, 8), downlink_min)
                uplink[i] = max(uplink[i] - np.random.randint(0, 5), uplink_min)
        workload += np.random.randint(-100, 200)


    elif (time1>36) and (time1<=72):
        if (a < b):
            local_comp = min(local_comp + np.random.randint(0, 6), local_core_max)
            for i in range(4):
                servers_cap[i] = min(servers_cap[i] + np.random.randint(0, 15), server_core_max)
                downlink[i] = min(downlink[i] + np.random.randint(0, 8), downlink_max)
                uplink[i] = min(uplink[i] + np.random.randint(0, 5), uplink_max)

        else:
            local_comp = max(local_comp + np.random.randint(-5, 0), local_core_min)
            for i in range(4):
                servers_cap[i] = max(servers_cap[i] + np.random.randint(-14, 0), server_core_min)
                downlink[i] = max(downlink[i] - np.random.randint(0, 8), downlink_min)
                uplink[i] = max(uplink[i] - np.random.randint(0, 5), uplink_min)
        workload += np.random.randint(-200, 100)


    elif (time1>72) and (time1<=108):
        if (a > b):
            local_comp = min(local_comp + np.random.randint(0, 6), local_core_max)
            for i in range(4):
                servers_cap[i] = min(servers_cap[i] + np.random.randint(0, 15), server_core_max)
                downlink[i] = min(downlink[i] + np.random.randint(0, 8), downlink_max)
                uplink[i] = min(uplink[i] + np.random.randint(0, 5), uplink_max)

        else:
            local_comp = max(local_comp + np.random.randint(-5, 0), local_core_min)
            for i in range(4):
                servers_cap[i] = max(servers_cap[i] + np.random.randint(-14, 0), server_core_min)
                downlink[i] = max(downlink[i] - np.random.randint(0, 8), downlink_min)
                uplink[i] = max(uplink[i] - np.random.randint(0, 5), uplink_min)
        workload += np.random.randint(-100, 200)
    observation_ = np.array([workload,local_comp])
    observation_1 = np.hstack((observation_,servers_cap,uplink,downlink))
    return  observation_1,reward,local_only,remote_only

def run_mec_offloading():
    step = 0
    local_only_cost,remote_only_cost,total_cost=[],[],[]
    for episode in range(100):

        observation = reset()

        for time_1 in range(108):
            # print("当前状态值为：",observation)
            action = RL.choose_action(observation)
            print(action)

            observation_, reward ,local_only,remote_only= mec_step(observation,action,time_1)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 2000) and (step % 5 == 0):
                 RL.learn()
            if  step>2000 and step % 100 == 0:
                 local_only_cost.append(local_only)
                 remote_only_cost.append(remote_only)
                 total_cost.append(-reward)

            observation = observation_
            step += 1
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(local_only_cost)), local_only_cost,'b')
    plt.plot(np.arange(len(remote_only_cost)), remote_only_cost,'g')
    plt.plot(np.arange(len(total_cost)), total_cost,'r')
    plt.legend(("Execute_Local","Execute_Remote","Advanced_DQN"))
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    # end of game
    print('game over')

if __name__ == "__main__":
    # maze game

    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    run_mec_offloading()
    RL.plot_cost()