import pygame

from src.simulator.Environment import Environment

env = Environment(robot_start=(300, 300), goal=(400, 400))

import numpy as np
np.set_printoptions(precision=4, linewidth=180)
class Keyboard_Controller:
    def __init__(self, MODE):
        self.left = 0
        self.right = 0
        self.running = True

        self.MODE = MODE
        self.max_s = [-9E9 for i in range(6)]
        self.loop()


    def loop(self):
        while self.running:

            self.process_keys()

            state = env.step((self.left, self.right))
            s = state[0]
            for i in range(0,len(state)):
                if s[i] > self.max_s[i]:
                    self.max_s[i] = s[i]

            print(s)


            
            self.left = 0
            self.right = 0
                
    def process_keys(self):

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.running = False

        if MODE == "WESD":
            if keys[pygame.K_s]:
                self.left = -1
            if keys[pygame.K_w]:
                self.left = 1
            if keys[pygame.K_d]:
                self.right = -1
            if keys[pygame.K_e]:
                self.right = 1

        elif MODE == "WASD":
            if keys[pygame.K_w]:
                self.left = .5
                self.right = .5
            if keys[pygame.K_s]:
                self.left = -.5
                self.right = -.5
            if keys[pygame.K_a]:
                self.left = -.1
                self.right = .1
            if keys[pygame.K_d]:
                self.left = .1
                self.right = -.1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

MODE = "WASD"
kc = Keyboard_Controller(MODE)
