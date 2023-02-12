from environment import *

def main():
    start = time.time()
    time_limit = time.time()
    pygame.init()
    fps=30
    fpsclock=pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    grid = make_grid(ROWS, COLUMNS, WIN_WIDTH)
    agent = Agent(0, 0, 28, 0)
    while True:
        if time.time() - time_limit >= 120:
            print('time limit reached')
            print(f'score: {agent.score}')
            break
        if time.time() - start >= 0.4:
            chosen_spot = random_spot_chooser(grid, agent.get_pos())
            if chosen_spot == None:
                print('Mission Failed!! All cells have been occupied')
                print(f'score: {agent.score}')
                break
            grid[chosen_spot.row][chosen_spot.column].state = True
            start = time.time()
        
        grid = draw(SCREEN, grid, ROWS, COLUMNS, WIN_WIDTH, WIN_HEIGHT)
        
        agent.draw(SCREEN)
        
        event_listerners(agent)
        
        agent.object_picker(grid)
        
        pygame.display.update()
        fpsclock.tick(fps)

if __name__ == '__main__':
    main()  