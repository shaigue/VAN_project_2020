"""This is the implementation of agent.
"""

from abc import ABC, abstractmethod
import numpy as np
from Field import Field, plot_field, plot_observation, plot_occlusions
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

class Agent(ABC):
    """This is an abstruct class for Agent.
    
    Attributes:
        x: x coordinate
        y: y coordinate
        M: Width dimention of the grid
        N: Hight dimention of the grid
        belief_grid: estimated probability for each cell to be occupied by an object
        history: the history of observations, list of tuples `((x,y),observation_dict)`

    Methods:
        __init__():
        reset(M, N):
        update_belief(observation): updates the belif of the object hypothesis given `observation`
            observation: this is a dictionary with `(x,y)` coordinates as keys, 
            and classifier score `s` as value.
        predict(): predicts for each cell if it occupied or not
        step(): 

    """

    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0
        self.M = None
        self.N = None
        self.belief_grid = None
        self.history = None

    @abstractmethod
    def reset(self, M, N):
        self.x = 0
        self.y = 0
        self.M = M
        self.N = N
        self.belief_grid = np.full((M, N), 0.5)
        self.history = []

    @abstractmethod
    def update_belief(self, observation):
        if self.belief_grid is None:
            raise RuntimeError("The agent is not initialized")
        self.history.append(((self.x, self.y), observation))

    
    @abstractmethod
    def predict(self):
        if self.belief_grid is None:
            raise RuntimeError("The agent is not initialized")
        grid_est = np.zeros_like(self.belief_grid)
        grid_est[self.belief_grid > 0.5] = 1
        grid_est[self.belief_grid <= 0.5] = 0

        return grid_est
    
    @abstractmethod
    def step(self):
        if self.belief_grid is None:
            raise RuntimeError("The agent is not initialized")
        return True


class ScanningAgent(Agent):
    """A naive agent that assumes certianty in observations,
    and goes in a simple scan trajectory"""
    
    def __init__(self):
        super().__init__()

        # for movement control
        self.left = 0
        self.up = False

    def reset(self, M, N):
        super().reset(M, N)
        self.left = 2
        self.up = True

    def update_belief(self, observation):
        super().update_belief(observation)
        # TODO: decide on how to update the belief
        # do it in a modular way
        
        for (x,y),s in observation.items():
            if (s > 0.5):
                self.belief_grid[x,y] = 1
            else:
                self.belief_grid[x,y] = 0

    def predict(self):
        return super().predict()

    def step(self):
        super().step()

        # if reached the end
        if self.x == (self.M - 1) and self.y == (self.N - 1):
            return False
        
        # do we need to do left actions?
        if self.left > 0:
            self.left -= 1
            if (self.x < self.M - 1):
                self.x += 1
            return True

        # are we going down?
        if not self.up:
            # have we reached the end?
            if self.y > 0:
                self.y -= 1
            else:
                self.left = 3
                self.up = True
                if self.x < self.M - 1:
                    self.x += 1
            return True

        # are we going up?
        if self.up:
            # have we reached the end?
            if self.y < self.N - 1:
                self.y += 1
            else:
                self.left = 3
                self.up = False
                if (self.x < self.M - 1):
                    self.x += 1
            return True


def plot_trajectory(ax: Axes, agent: Agent, field: Field):
    traj_x = [h[0][0] + 0.5 for h in agent.history]
    traj_y = [h[0][1] + 0.5 for h in agent.history]
    ax.plot(traj_x, traj_y, c="r")

def plot_estimate(ax: Axes, agent: Agent, field: Field):
    """plot the estimates at the current point at were is there an object,
    and where there is none - overlayed on the gound truth."""
    est = agent.predict()
    est_obj_x = []
    est_obj_y = []
    for i in range(field.M):
        for j in range(field.N):
            if est[i,j] == 1:
                est_obj_x.append(i + 0.5)
                est_obj_y.append(j + 0.5)
    ax.scatter(est_obj_x, est_obj_y, s=50, c="b", marker="o", alpha=0.2)

def plot_belief(ax: Axes, agent: Agent, field: Field):
    """plot the probability of Y_i,j for the entire grid cell - 
    overlayed on the ground truth."""
    ax.imshow(agent.belief_grid.T,
        extent=(0,field.M,0,field.N),
        origin="lower", 
        alpha=0.5,
        cmap="Reds",
    )

if __name__ == "__main__":
    M = 20
    N = 20
    F = 3
    Q = 4
    noise_ind = 0.5

    # run a game with the ScanningAgent
    s_agent = ScanningAgent()
    s_agent.reset(M,N)
    field = Field(M, N, F, Q, noise_ind)
    while s_agent.step():
        s_agent.update_belief(field.generate_observation(s_agent.x, s_agent.y))
    
    fig, axes = plt.subplots(2,2)
    plot_field(axes[0,0],field, False)
    plot_trajectory(axes[0,0],s_agent, field)
    plot_belief(axes[0,0],s_agent,field)
    # plot_estimate(axes[0,0], s_agent, field)
    plt.show()

    # naive_agent = NaiveAgent()
    # naive_agent.reset(M, N)
    # field = Field(M, N, F, Q)

    # for t in range(M * N / 2):
    #     observation = field.generate_observation(naive_agent.x, naive_agent.y)
    #     naive_agent.update_belief(observation)
    #     naive_agent.step()

    # # plot the real board
    # plt.imshow(field.object_grid)
    # plt.show()
    # # plot the estimation results
    # plt.imshow(naive_agent.predict())
    # plt.show()

    # # plot the trajectory
    # img = np.zeros((M,N))
    # for x,y in naive_agent.trajectory:
    #     img[x,y] = 1
    #     plt.imshow(img)
    #     plt.pause(0.1)
        
    # plt.show()

