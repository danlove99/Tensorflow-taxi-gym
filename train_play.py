import gym
import random
import numpy as np
from numpy import sin, cos, pi
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
import time 

# setup enviroment and define variables
env = gym.make('Taxi-v3')
env.reset()
goal_steps = 100
score_requirement = 20
initial_games = 50000
LR = 1e-3


def model_data_preparation():

    training_data = []
    accepted_scores = []
    count = 0
    # for every game

    for game_index in range(initial_games):
        count += 1
        print('game: {}'.format(count))
        score = 0
        # temp memory that may be appended to training score
        game_memory = []
        #place holder for last obs
        previous_observation = []
        # for each step 
        for step_index in range(goal_steps):
            action = random.randrange(0, 6)
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = [observation]
            
            if (reward == 20):
                score = reward
            
            score += reward
            if done:
                break
        # now have final score for that game along with game memory of position(observation) and action
            
        # if score is good append score to accepted scores 
        # append that game data to training_data with one hot encoded action    
        if score >= score_requirement:
            print("win!")
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                elif data[1] == 3:
                    output = [1, 1, 0]
                elif data[1] == 4:
                    output = [0, 1, 1]
                elif data[1] == 5:
                    output = [1, 1, 1]
                training_data.append([data[0], output])
        
        env.reset()
    
    print(accepted_scores)
    
    return training_data
   


def build_model(input_size, output_size):
        model = Sequential()
        model.add(Dense(128, input_dim=input_size, activation='relu'))
        model.add(Dense(52, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())

        return model

def train_model(training_data):

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    print(X)
    print(y)
    model.fit(X, y, epochs=5)
    return model


data = model_data_preparation()



trained_model = train_model(data)


scores = []
choices = []
for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        # Uncomment this line if you want to see how our bot playing
        env.render()
        #time.sleep(0.2)
        # start with random step, then predict next step based on prev obs 
        # which turns into new obs every step
        if len(prev_obs)==0:
            action = random.randrange(0,6)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = np.array([new_observation])
        score+=reward
        if done:
            #print('WIN!!!')
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score:',sum(scores)/len(scores))
