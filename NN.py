import tensorflow as tf
import numpy as np
from tensorflow import keras
import random

def neural_net(state_shape, action_shape=(1,8)):
    '''
        maps state to action
        states is the array of the grid (7x14)
        actions shape is just a (1x8) array representing the actions:
        w, a, s, d, w+a, w+d, s+a, s+d
    '''
    
    state_shape = state_shape.reshape(1, 98)
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform() #generate weights with uniform values
    model = keras.Sequential()
    model.add(keras.layers.Dense(53, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def train(env, replay_memory, model, target_model, done):
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
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []

    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        
        current_qs = current_qs_list[index]
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