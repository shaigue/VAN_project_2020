"""Here we do some experiments to get some metrices.

accuracy and MSDE."""
from pprint import pprint

import numpy as np

from Models import Model, GNB_Fixed, GP_corr
from Field import Field
from Agent import ScanningAgent


if __name__ == "__main__":
    # create a field
    # run some tests, and collect - MSDE, true/false positives/negatives.
    # summerize all the data
    """Field for experiment 1"""
    """
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
    """
    
    """Field for experiment 2 + 3
    M = 40
    N = 40
    F = 5
    Q = 50
    noise_ind = 0.2
    range_decay = 0.1
    lights = []
    obstacles = [] 
    num_alias = 50
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
    for i in [10, 25, 35]:
        for j in range(20,26):
            obstacles.append((i,j))
    for j in [24, 34, 15]:
        for i in range(30,36):
            obstacles.append((i,j)) 
    end field for experiment 2 + 3"""
    """Field for experiment 4 + 5"""
    M = 50
    N = 50
    F = 6
    Q = 100
    noise_ind = 0.2
    # exp 4
    # range_decay = 0.1
    # exp 5
    range_decay = 0.2
    lights = []
    obstacles = [] 
    num_alias = 100
    # place lights
    t1 = round(M/3)
    t2 = round(N/3)
    lights.append((t1,t2))
    lights.append((M-t1,t2))
    lights.append((t1,N-t2))
    # place obstacles
    for i in [4, 15, 22]:
        for j in range(3,8):
            obstacles.append((i,j))
    for j in [1, 17, 27]:
        for i in range(9,14):
            obstacles.append((i,j))
    for i in [10, 25, 35]:
        for j in range(20,26):
            obstacles.append((i,j))
    for j in [24, 34, 15]:
        for i in range(30,36):
            obstacles.append((i,j)) 
    # exp 5
    for i in [12,22,32,42]:
        for j in range(10, 18):
            obstacles.append((i,j))
        for j in range(38, 44):
            obstacles.append((i,j))
    """ end field for experiment 4 + 5"""
    field = Field(M, N, F, Q, noise_ind, range_decay, lights, obstacles, num_alias)

    ind_model = GNB_Fixed()
    corr_model = GP_corr()
    ind_model.train(field)
    corr_model.train(field)
    
    # experiment 2:
    # step_size = F
    # experiment 3:
    # step_size = round(3 * (F / 2))
    # experiment 4:
    step_size = F * 2

    ind_agent = ScanningAgent(M,N,ind_model, step_size)
    corr_agent = ScanningAgent(M,N,corr_model, step_size)

    num_trails = 50
    corr_stats = {'accuracy':[], 'MSDE':[]}
    ind_stats = {'accuracy':[], 'MSDE':[]}
    for i in range(num_trails):
        field.randomize_objects()
        ind_agent.reset()
        while ind_agent.step():
            ind_agent.update_belief(field.generate_observation(ind_agent.x, ind_agent.y))

        corr_agent.reset()
        while corr_agent.step():
            corr_agent.update_belief(field.generate_observation(corr_agent.x, corr_agent.y))
        
        corr_prob = corr_agent.predict_proba()
        ind_prob = ind_agent.predict_proba()
        corr_est = corr_agent.predict()
        ind_est = ind_agent.predict()

        objects_grid = np.zeros_like(corr_prob)
        for (i,j) in field.objects:
            objects_grid[i,j] = 1

        corr_acc = np.sum(corr_est == objects_grid) / corr_prob.size
        ind_acc = np.sum(ind_est == objects_grid) / ind_prob.size

        corr_msde = np.mean((objects_grid - corr_prob) ** 2)
        ind_msde = np.mean((objects_grid - ind_prob) ** 2)

        corr_stats['accuracy'].append(corr_acc) 
        corr_stats['MSDE'].append(corr_msde)
        ind_stats['accuracy'].append(ind_acc)
        ind_stats['MSDE'].append(ind_msde)

    corr_stats['mean_acc'] = np.mean(corr_stats['accuracy'])
    corr_stats['mean_msde'] = np.mean(corr_stats['MSDE'])
    ind_stats['mean_acc'] = np.mean(ind_stats['accuracy'])
    ind_stats['mean_msde'] = np.mean(ind_stats['MSDE'])

    print("***Corr***")
    pprint(corr_stats)
    print("***Ind***")
    pprint(ind_stats)

        

    

