from collections import deque

from .environment import *
from .NN import *


def main():
    pygame.init()
    fps=30
    fpsclock=pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    agent = Agent(0, 0, 28, 0)
    
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    replay_memory = deque(maxlen=1_000)

    # Main Model (updated every 4 steps)
    model = neural_net((99,), 8)

    # Target Model (updated every 100 steps)
    target_model = neural_net((99,), 8)
    target_model.set_weights(model.get_weights())

    steps_to_update_target_model = 0

    for episode in range(300):
        episode_time_limit = time.time()
        object_spawn_interval = time.time()
        grid = make_grid(ROWS, COLUMNS, WIN_WIDTH)
        agent.score = 0
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
                print('Exploiting')
                prediction = model.predict(np.array([curr_state]))
                action = np.argmax(prediction)
            
            new_state, reward, done, object_spawn_interval = step(agent, action, grid, object_spawn_interval, episode_time_limit)
            
            object_spawn_interval = object_spawn_interval


            grid = draw(SCREEN, new_state, ROWS, COLUMNS, WIN_WIDTH, WIN_HEIGHT)
            agent.draw(SCREEN)
            pygame.display.update()
            fpsclock.tick(fps)

            replay_memory.append([grid, agent, action, reward, new_state, done]) #grid == state

            if steps_to_update_target_model % 4 == 0 or done:
                history = train(replay_memory, model, target_model, done)
            
            grid = new_state
            agent.score += reward


            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(agent.score, episode, reward))
                

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    return model, history

if __name__ == '__main__':
    main()