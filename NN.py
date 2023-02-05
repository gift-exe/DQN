import tensorflow as tf
import numpy as np
from tensorflow import keras
import random

def feature_extractor(grid, agent):
    '''
        lol! you really thought?! lmao! I'm wheezing!!
    '''
    #extract state value for each spot on the env
    reshaped_grid = grid.reshape(98,)
    state_values = np.array([spot.state for spot in reshaped_grid])
    
    #extract position of the agent and map position to an int value ranging from (0, 97)
    agent_coordinates = agent.get_pos()
    agent_pos = (agent_coordinates[1] * 14) + agent_coordinates[0]

    state_values = np.append(state_values, agent_pos)

    return state_values.reshape(1, 99)

def neural_net(state, action_shape):
    '''
        maps state to action
        states is the array of the grid (7x14)
        actions shape is just a (1x8) array representing the actions:
        w, a, s, d, w+a, w+d, s+a, s+d
    '''
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform() #generate weights with uniform values
    model = keras.Sequential()
    model.add(keras.layers.Dense(53, input_shape=state, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def train(replay_memory, model, target_model, done):
    '''
        qs >> q-state value pair
    '''
    learning_rate = 0.7
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    
    batch_size = 64*2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([(transition[0], transition[1]) for transition in mini_batch])
    current_state_features = feature_extractor(current_states[0][0], current_states[0][1])
    current_qs_list = model.predict(current_state_features)
    new_current_states = np.array([(transition[4], transition[1]) for transition in mini_batch])
    new_current_state_features = feature_extractor(new_current_states[0][0], new_current_states[0][1])
    future_qs_list = target_model.predict(new_current_state_features)

    X = []
    Y = []

    for index, (observation, agent, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[0][index])
            print('\n\n',future_qs_list[0][index],'\n\n')
        else:
            max_future_q = reward
        
        current_qs = current_qs_list[0]
        current_qs[action] = (1-learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

'''
    Observation would have the state (position) of the agent, state (positions) of other cells in the env    

    State 
    Action
    New State
    Reward
    Done
'''