import numpy as np
import random
import pygame
from pathlib import Path
import os


class Env:

    def __init__(self, id=0, width=10, start_state=0, random_start=True, goal_state=1, blocks=None, num_block=5):
        self.id = id
        self.width = width
        self.random_start = random_start
        if random_start is True:
            self.start_state = np.random.randint(0, width ** 2)
            while self.start_state in blocks:
                self.start_state = np.random.randint(0, width ** 2)
        else:
            self.start_state = start_state
        self.goal_state = goal_state
        if blocks is None:
            self.blocks = self.generate_block(num_block)
        else:
            self.blocks = blocks
        self.state = self.start_state
        self.done = False
        self.value = np.ones((width, width, 4))
        self.viewer = None
        self.FPSCLOCK = pygame.time.Clock()
        self.unit = 60
        self.screen_size = (width * (self.unit + 1), width * (self.unit + 1))
        self.path = []

        self.env_picture_path = "./env_picture/env_{}.jpg".format(self.id)
        if Path(self.env_picture_path).is_file():
            os.remove(self.env_picture_path)

    def generate_block(self, num_block):
        blocks = [self.start_state]
        while self.start_state in blocks or self.goal_state in blocks:
            blocks = random.sample(range(0, self.width ** 2), num_block)
        return blocks

    def reset(self):
        if self.random_start is True:
            self.start_state = np.random.randint(0, self.width ** 2)
            while self.start_state in self.blocks:
                self.start_state = np.random.randint(0, self.width ** 2)
        self.state = self.start_state
        self.done = False
        return self.state

    def assign_reward(self):
        if self.state == self.goal_state:
            reward = 1
        else:
            reward = 0
        return reward

    def step(self, action):
        # 0:move up, 1:move down, 2:move left, 3:move right
        state = self.state
        if action == 0 and state >= self.width:
            state -= self.width
        if action == 1 and state < self.width ** 2 - self.width:
            state += self.width
        if action == 2 and not state % self.width == 0:
            state -= 1
        if action == 3 and not (self.state % self.width) == self.width - 1:
            state += 1
        if state in self.blocks:
            pass
        else:
            self.state = state
        reward = self.assign_reward()
        if self.state == self.goal_state:
            self.done = True
        else:
            self.done = False
        return self.state, reward, self.done

    def render(self):
        # time.sleep(1)
        if self.viewer is None:
            pygame.init()
            # init a window
            self.viewer = pygame.display.set_mode(self.screen_size, 0)
            pygame.display.set_caption("GirdWorld_{}".format(self.id))
        self.viewer.fill((255, 255, 255))
        # draw the border of grids
        for i in range(self.width + 1):
            pygame.draw.lines(self.viewer, (0, 0, 0), True,
                              (((self.unit + 1) * i, 0), ((self.unit + 1) * i, self.width * (self.unit + 1))), 1)
            pygame.draw.lines(self.viewer, (0, 0, 0), True,
                              ((0, (self.unit + 1) * i), (self.width * (self.unit + 1), (self.unit + 1) * i)), 1)
        # draw blocks
        for i in range(len(self.blocks)):
            x = self.blocks[i] % self.width * (self.unit + 1) + 1
            y = self.blocks[i] // self.width * (self.unit + 1) + 1
            pygame.draw.rect(self.viewer, (0, 0, 0), pygame.Rect(x, y, self.unit, self.unit))
        # print q-values on grids
        font = pygame.font.SysFont('times', 10)
        for i in range(self.width):
            for j in range(self.width):
                surface = font.render('{0:''^5}'.format(value_format(self.value[i, j, 0])), True, (0, 0, 0))
                self.viewer.blit(surface, ((self.unit + 1) * i + (self.unit / 2 - 10), (self.unit + 1) * j + 1))
                surface = font.render('{0:''^5}'.format(value_format(self.value[i, j, 1])), True, (0, 0, 0))
                self.viewer.blit(surface,
                                 ((self.unit + 1) * i + (self.unit / 2 - 10), (self.unit + 1) * j + (self.unit - 10)))
                surface = font.render('{0:''<5}'.format(value_format(self.value[i, j, 2])), True, (0, 0, 0))
                self.viewer.blit(surface, ((self.unit + 1) * i + 2, (self.unit + 1) * j + (self.unit / 2 - 5)))
                surface = font.render('{0:''>5}'.format(value_format(self.value[i, j, 3])), True, (0, 0, 0))
                self.viewer.blit(surface,
                                 ((self.unit + 1) * i + (self.unit - 22), (self.unit + 1) * j + (self.unit / 2 - 5)))
        # draw the path
        for i in range(len(self.path)):
            if self.path[i] in self.path[i + 1:]:
                continue
            x = self.path[i] % self.width * (self.unit + 1)
            y = self.path[i] // self.width * (self.unit + 1)
            pygame.draw.rect(self.viewer, [255, 0, 0], [x, y, self.unit + 2, self.unit + 2], 2)
            surface = font.render(str(i), True, (255, 0, 0))
            self.viewer.blit(surface, (x + 5, y + 5))
        pygame.display.update()

        game_over()
        # time.sleep(1)
        self.FPSCLOCK.tick(30)

    def save_result(self):
        pygame.image.save(self.viewer, self.env_picture_path)

    def link_record(self, value, path):
        """
        :param value: get a q-value list from agent.
        :param path: get a path from the path record.

        link variables for updating the render
        """
        self.value = value
        self.path = path


def game_over():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()


def value_format(v):
    if v > 1:
        s = '{0:.4g}'.format(float(v))
    else:
        s = '{0:.3f}'.format(float(v))
    return float(s)