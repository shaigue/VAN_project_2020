"""This is the implementation of agent.
"""
from pprint import pprint
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from Models import Model, GNB_Fixed, parse_history, GP_corr
from Field import Field, plot_field, plot_observation, plot_occlusions, explore_field



class Agent(ABC):
    """This is an abstruct class for Agent.
    
    Attributes:
        x: x coordinate
        y: y coordinate
        M: Width dimention of the grid
        N: Hight dimention of the grid
        belief_grid: estimated probability for each cell to be occupied by an object
        history: the history of observations, list of tuples `((x,y),observation_dict)`
        model: a trained model that is ready to use

    Methods:
        __init__():
        reset(M, N):
        update_belief(observation): updates the belif of the object hypothesis given `observation`
            observation: this is a dictionary with `(x,y)` coordinates as keys, 
            and classifier score `s` as value.
        predict(): predicts for each cell if it occupied or not
        step(): 

    """

    def __init__(self, M: int, N: int, model: Model):
        super().__init__()
        self.x = 0
        self.y = 0
        self.M = M
        self.N = N
        self.history = None
        self.model = model

    @abstractmethod
    def reset(self) -> None:
        self.x = 0
        self.y = 0
        self.history = list()

    @abstractmethod
    def update_belief(self, observation) -> None:
        if self.history is None:
            raise RuntimeError("The agent is not initialized")
        self.history.append(((self.x, self.y), observation))

    
    @abstractmethod
    def predict(self) -> np.array:
        if self.history is None:
            raise RuntimeError("The agent is not initialized")

        return self.model.predict(self.history)

    @abstractmethod
    def predict_proba(self) -> np.array:
        if self.history is None:
            raise RuntimeError("The agent is not initialized")

        return self.model.predict_proba(self.history)
    
    @abstractmethod
    def step(self):
        if self.history is None:
            raise RuntimeError("The agent is not initialized")
        return True


class ScanningAgent(Agent):
    """A naive agent that assumes certianty in observations,
    and goes in a simple scan trajectory"""
    
    def __init__(self,  M: int, N: int, model: Model, steps: int = 3):
        super().__init__(M, N, model)

        # for movement control
        self.setps = steps
        self.left = 0
        self.up = False

    def reset(self):
        super().reset()
        self.left = int(self.setps / 2)
        self.up = True

    def update_belief(self, observation):
        super().update_belief(observation)

    def predict(self):
        return super().predict()
        
    def predict_proba(self):
        return super().predict_proba()

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
                self.left = self.setps - 1
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
                self.left = self.setps - 1
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
    belief = agent.predict_proba()
    ax.imshow(belief.T,
        extent=(0,field.M,0,field.N),
        origin="lower", 
        alpha=0.5,
        cmap="Reds",
    )

if __name__ == "__main__":
    M = 30
    N = 30
    F = 5
    Q = 20
    noise_ind = 0.2
    range_decay = 0.1
    lights = []
    obstacles = [] 
    num_alias = 20
    # place lights
    t1 = round(M/3)
    t2 = round(N/3)
    lights.append((t1,t2))
    lights.append((M-t1,M-t2))
    # place obstacles
    for i in [4, 15, 22]:
        for j in range(3,8):
            obstacles.append((i,j))
    for j in [1, 17, 27]:
        for i in range(9,14):
            obstacles.append((i,j)) 
    
    field = Field(M, N, F, Q, noise_ind, range_decay, lights, obstacles, num_alias)

    ind_model = GNB_Fixed()
    corr_model = GP_corr()
    ind_model.train(field)
    corr_model.train(field)
    
    ind_agent = ScanningAgent(M,N,ind_model, F * 2)
    corr_agent = ScanningAgent(M,N,corr_model, F * 2)


    field.randomize_objects()
    ind_agent.reset()
    while ind_agent.step():
      ind_agent.update_belief(field.generate_observation(ind_agent.x, ind_agent.y))

    corr_agent.reset()
    while corr_agent.step():
      corr_agent.update_belief(field.generate_observation(corr_agent.x, corr_agent.y))
    
    # we want 2 figures.
    # one of the field, with lighting
    fig, ax = plt.subplots(figsize=(15,15))
    plot_field(ax, field)
    ax.set_title("Field")
    
    # and 2 of the agents, independent, and correlated
    
    
    # independent agent
    fig_ind, ax_ind = plt.subplots(figsize=(15,15))
    ax_ind.set_title("Independent Observation Model")
    plot_field(ax_ind, field, light=False)
    plot_trajectory(ax_ind, ind_agent, field)
    plot_belief(ax_ind, ind_agent, field)
    plot_estimate(ax_ind, ind_agent, field)

    # Correlated agent
    fig_corr, ax_corr = plt.subplots(figsize=(15,15))
    ax_corr.set_title("Correlated Observation Model")
    plot_field(ax_corr, field, light=False)
    plot_trajectory(ax_corr, corr_agent, field)
    plot_belief(ax_corr, corr_agent, field)
    plot_estimate(ax_corr, corr_agent, field)
    
    plt.show()

