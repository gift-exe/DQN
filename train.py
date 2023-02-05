from collections import deque

from environment import *
from NN import *


def main():
    object_spawn_interval = time.time()
    episode_time_limit = time.time()
    pygame.init()
    fps=30
    fpsclock=pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    grid = make_grid(ROWS, COLUMNS, WIN_WIDTH)
    agent = Agent(0, 0, 28, 0)
    
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    replay_memory = deque(maxlen=50_000)

    # Main Model (updated every 4 steps)
    model = neural_net((99,), 8)

    # Target Model (updated every 100 steps)
    target_model = neural_net((99,), 8)
    target_model.set_weights(model.get_weights())

    steps_to_update_target_model = 0

    for episode in range(300):
        done = False
        while not done:
            steps_to_update_target_model += 1

            exp_prob = np.random.rand()
            if exp_prob <= epsilon:
                #explore
                action = np.random.choice(agent.actions)
            else:
                #exploit
                curr_state = feature_extractor(grid, agent)
                prediction = model.predict(curr_state)
                action = np.argmax(prediction)

            new_state, reward, done = step(agent, action, grid, object_spawn_interval, episode_time_limit)
            replay_memory.append([grid, agent, action, reward, new_state, done]) #grid == state

            if steps_to_update_target_model % 4 == 0:
                train(replay_memory, model, target_model, done)
            
            grid = new_state

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(agent.score, episode, reward))
                

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            
            
            
            


    # while True:
    #     if time.time() - time_limit >= 120:
    #         print('time limit reached')
    #         print(f'score: {agent.score}')
    #         break
    #     if time.time() - start >= 0.4:
    #         chosen_spot = random_spot_chooser(grid, agent.get_pos())
    #         if chosen_spot == None:
    #             print('Mission Failed!! All cells have been occupied')
    #             print(f'score: {agent.score}')
    #             break
    #         grid[chosen_spot.row][chosen_spot.column].state = True
    #         start = time.time()
        
    #     grid = draw(SCREEN, grid, ROWS, COLUMNS, WIN_WIDTH, WIN_HEIGHT)
        
    #     agent.draw(SCREEN)
        
    #     event_listerners(agent)
        
    #     agent_object_picker(agent, grid)
        
    #     pygame.display.update()
    #     fpsclock.tick(fps)

if __name__ == '__main__':
    main()