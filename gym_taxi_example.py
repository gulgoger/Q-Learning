import gym

env = gym.make("Taxi-v3",render_mode="human").env   #enviroment ile yapı kurulumu

env.reset()  #env başlat

env.render()   #show


#env.close()
#%%
print("State space:" ,env.observation_space)#500 (ortamın toplamda 500 farklı durum(state) olduğunu gösterir.)
print("Action space:" ,env.action_space)#6 taksi 6 farklı eylem gerçekleştirir aşağı,yuk,sağ,sol,yolcu indir,bindir.

#taxi row, taxi column, passenger index, destination
state = env.encode(3,1,2,3)
print("State number: ",state)

env.s = state
env.render()

#%%

"""
Actions:
    There are 6 discrete deterministic actions:
        - 0: move south
        - 1: move north
        - 2: move east
        - 3: move west
        - 4: pickup passenger
        - 5: dropoff passenger
"""
# probability, next state
env.P[331]

#%%
env.reset()
time_step = 0
total_reward = 0
list_visualize = []

while True:
    
    time_step += 1
    
    #choose action
    action = env.action_space.sample()  #yapacağımız actionı belirliyoruz
    
    #perform action and get reward
    state, reward, done,truncated, _ = env.step(action) #state = next state
        
    #total reward
    total_reward += reward
    
    #visualize
    list_visualize.append({"frame": env,
                           "state":state, "action":action, "reward":reward,
                           "Total Reward":total_reward})
    #env.render()
    
    if done:
        break

#env.close()
#%%
import time
for i, frame in enumerate(list_visualize):
    print(frame["frame"].render())
    print("Timestep:", i+1)
    print("Timestep:", frame["state"])
    print("action:", frame["action"])
    print("reward:", frame["reward"])
    print("Total Reward:", frame["Total Reward"])
    time.sleep(1)


env.close()











































