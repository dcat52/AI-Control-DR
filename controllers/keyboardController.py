from simulator.Environment import Environment
import pygame

env = Environment(robot_start=(300, 300), goal=(400, 400))

class Keyboard_Controller:
    def __init__(self, MODE):
        self.left = 0
        self.right = 0
        self.running = True

        self.MODE = MODE
        self.loop()

    def loop(self):
        while self.running:

            self.process_keys()

            state = env.step((self.left, self.right))
            print(state)
            
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
                self.left = 1
                self.right = 1
            if keys[pygame.K_s]:
                self.left = -1
                self.right = -1
            if keys[pygame.K_a]:
                self.left = -.5
                self.right = .5
            if keys[pygame.K_d]:
                self.left = .5
                self.right = -.5

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

MODE = "WASD"
kc = Keyboard_Controller(MODE)
