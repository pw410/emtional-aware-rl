import numpy as np

class GridWorld11x11:
    def __init__(self):
        self.size = 11
        self.start = (0, 0)
        self.goal = (10, 10)

        # risky cell (lava)
        self.lava = (5, 5)

        # actions: up, down, left, right
        self.action_space = 4  

        self.reset()
        self.state_dim = len(self._get_state())

    def reset(self):
        self.agent = list(self.start)
        return self._get_state()

    def step(self, action):
        x, y = self.agent

        if action == 0: x -= 1   # up
        elif action == 1: x += 1 # down
        elif action == 2: y -= 1 # left
        elif action == 3: y += 1 # right

        # boundary clip
        x = np.clip(x, 0, self.size-1)
        y = np.clip(y, 0, self.size-1)

        self.agent = [x, y]

        done = False
        reward = -0.1  # step penalty

        # lava check
        if (x, y) == self.lava:
            reward = -10.0
            done = True

        # goal check
        if (x, y) == self.goal:
            reward = +10.0
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        # state = [x, y, lava_x, lava_y, goal_x, goal_y]
        x, y = self.agent
        return np.array([x, y, 5, 5, 10, 10], dtype=np.float32)