from RL_brain import DeepQNetwork
import time
def update():
    #生成随机状态（总共有8种状态）（总工作量，上行链路容量，下行链路容量，各个服务器可用核数）



def render():

def step():


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = update()

        while True:
            # fresh env
            render()
            #生成随机环境
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
            time.sleep(0.01)
        time.sleep(0.02)


    # end of game
    print('game over')

if __name__ == "__main__":
    # maze game
    RL = DeepQNetwork(n_actions=8, n_features=30,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #env.after(100, run_maze)
    run_maze()
    RL.plot_cost()