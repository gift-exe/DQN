import pygame
import random
import time
import numpy as np

global SCREEN
WIN_HEIGHT = 196
WIN_WIDTH = 392
ROWS = 7
COLUMNS = 14

WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)

AGENT = (0, 255, 0)
OBJECTIVE = (255, 0, 0)



class Spot():
    def __init__(self, row, column, width, total_rows):
        self.row = row
        self.column = column
        self.x = width * row
        self.y = width * column
        self.state = False
        self.spot_width = width
        self.total_rows = total_rows
        self.color = self.get_color()
    
    def __repr__(self):
        return f'(position: {(self.row, self.column)}, \n state: {self.state}\n)'
    
    def get_color(self):
        if self.state:
            color = OBJECTIVE
        else:
            color = BLACK
        return color

    def get_pos(self):
        return (self.row, self.column)
        
    def draw(self, win):
        pygame.draw.rect(win, self.get_color(), (self.x/2, self.y/2, self.spot_width/2, self.spot_width/2))
    
class Agent():
    def __init__(self, column, row, width, score):
        self.row = row
        self.column = column
        self.agent_width = width
        self.score = score
        self.color = AGENT
        self.actions_map = {'W':0, 'A':1, 'S':2, 'D':3, 'WA':4, 'WD':5, 'SA':6, 'SD':7}
        self.actions = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        
    def get_coordinates(self):
        x = self.agent_width * self.row
        y = self.agent_width * self.column
        return x, y

    def get_pos(self):
        return (self.row, self.column)
        
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.get_coordinates()[0], self.get_coordinates()[1], self.agent_width, self.agent_width))
    
    def move_up(self):
        if self.get_pos()[1] != 0:
            self.column = self.column -1
    def move_down(self):
        if self.get_pos()[1] != 6:
            self.column = self.column + 1
    def move_left(self):
        if self.get_pos()[0] != 0:
            self.row = self.row - 1
    def move_right(self):
        if self.get_pos()[0] != 13:
            self.row = self.row + 1
    
    def agent_listerner(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                self.move_left()
            if event.key == pygame.K_w:
                self.move_up()
            if event.key == pygame.K_s:
                self.move_down()
            if event.key == pygame.K_d:
                self.move_right()

    def object_picker(self, grid):
        current_pos = self.get_pos()
        if grid[current_pos[0]][current_pos[1]].state == True:
            self.score = self.score + 1
            grid[current_pos[0]][current_pos[1]].state = False
            return 1
        return 0


def make_grid(rows, columns, width):
    grid = []
    gap = width // rows
    for i in range(columns):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return np.array(grid)

def draw_grid(win, rows, columns, width, height):
    r_gap =  height // rows
    c_gap = width // columns
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * r_gap), (width, i * r_gap))
        for j in range(columns):
            pygame.draw.line(win, GREY, (j* c_gap, 0), (j * c_gap, height))

def draw(win, grid, rows, columns, width, height):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, columns, width, height)
    pygame.display.update()
    return grid

def random_spot_chooser(grid, agent_position):
    new_grid = []
    for row in grid:
        for spot in row:
            if spot.state == False and spot.get_pos() != agent_position:
                new_grid.append(spot)
            
    if len(new_grid)<=1:
        return None
    spot = random.choice(new_grid)

    return spot

def event_listerners(agent):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        agent.agent_listerner(event)

def event_listerners_for_ai():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.K_q:
            pygame.quit()
            quit()

def check_if_episode_time_limit_is_reached(agent, episode_time_limit):
    if time.time() - episode_time_limit >= 30:
        print('\n\ntime limit reached\n\n')
        print(f'score: {agent.score}')
        return True

def check_if_grid_filled_with_objects(agent, grid, object_spawn_interval):
    done = False
    if time.time() - object_spawn_interval >= 0.4:
        chosen_spot = random_spot_chooser(grid, agent.get_pos())
        if chosen_spot == None:
            print('\n\nMission Failed!! All cells have been occupied\n')
            print(f'score: {agent.score}\n\n')
            done = True
            return done, grid
        grid[chosen_spot.row][chosen_spot.column].state = True
        object_spawn_interval = time.time()
    return done, grid, object_spawn_interval

def check_if_current_episode_should_terminate(episode_time_limit, object_spawn_interval, agent, grid):
    done = check_if_episode_time_limit_is_reached(agent, episode_time_limit)
    if done:
        return done, grid, object_spawn_interval    
    done, grid, object_spawn_interval = check_if_grid_filled_with_objects(agent, grid, object_spawn_interval)
    return done, grid, object_spawn_interval

def ai_act(agent, action):
    if action == 0:
        agent.move_up()
    elif action == 1:
        agent.move_left()
    elif action == 2:
        agent.move_down()
    elif action == 3:
        agent.move_right()
    elif action == 4:
        agent.move_up()
        agent.move_left()
    elif action == 5:
        agent.move_up()
        agent.move_right()
    elif action == 6:
        agent.move_down()
        agent.move_left()
    elif action == 7:
        agent.move_down()
        agent.move_right()
    else:
        print(f'Action Value {action} Not Valid !!')

def step(agent, action, grid, object_spawn_interval, episode_time_limit):
    event_listerners_for_ai()
    ai_act(agent, action)   
    reward = agent.object_picker(grid)    
    done, curr_grid, object_spawn_interval = check_if_current_episode_should_terminate(episode_time_limit, object_spawn_interval, agent, grid)#new state after action
    return curr_grid, reward, done, object_spawn_interval
    


