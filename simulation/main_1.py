# this is the main script for running the simulation

"""description - 
parameters:
M
N
Q
T
{(X_i, Y_i)}

we have a game here, where a 
N X M grid represents a N * meters X M * meters field,
and there are Q objects in total. (Q is unknown to the player)
in grid points (X_i, Y_i) i is in 0 to Q - 1.
the player has a time limit of T steps.

the agent can plan a trajectory and go at each time step one cell in each of the 6 adjacent cells,
and each time recieves an observation of its surrounding

we have an observation model that produces the observations
and a model that predicts
"""

from abc import ABC, abstractmethod
import numpy as np

def run_game(agent, field, T):
    """This is the function that runs the gameplay.
    
    Arguments:
        agent:  this is an object that represents the agent,
                it needs to have a method to agent.plan, and

        field:  object that represents the field
        
        T:      the game duration
    """

    # set the costs for true positive, true negative, false positive and false negative
    # and calculate the decision of the agent accordingly
    # cost is minimized!!!
    cost_dict = {
        "TP": -1,
        "TN": -0.05,
        "FP": 0.3,
        "FN": 5,
    }

    # initialize the agent state, belief, initial position (defualt to 0,0)
    # and field boundries
    agent.reset(M=field.M, N=field.N, cost_dict=cost_dict)

    # play for T steps
    for t in range(T):
        # generate an observation given the current pose
        observation = field.generate_observation(agent.x, agent.y)
        # update the agents state and belief according to the given observations
        agent.integrate_observation(observation)
        # plan and pick an action to do given that we have T - t actions left 
        agent.do_action(T - t)
    
    # check the result and evaluate the scores: 
    estimated_grid = agent.estimate_grid()
    real_grid = field.get_grid()
    score = evaluate_score(estimated_grid, real_grid, cost_dict)
    return score


if __name__ == "__main__":
    # run the different setting of the experiments
    pass
