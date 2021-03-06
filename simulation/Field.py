"""This is were the field class will be implemented.

The Field is the class that represents the environment, 
and produces the observations(noisy and correlated) and classifier scores.
"""
from pprint import pprint

import numpy as np
from numpy.random import randint, randn, rand
from numpy.linalg import norm 
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

def l2_norm_sqared(x0, y0, x1, y1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2

def l2_norm(x0, y0, x1, y1):
    return np.sqrt(l2_norm_sqared(x0, y0, x1, y1))

class Field:
    """Represents a field.

    Attributes:
        M: grid width

        N: grid hight

        F: distance in grid cells of sight range
        
        Q: number of objects to be placed
        
        noise_ind:      float in [0,1] that indicate the level of independent noise in the classifier.
                        0 - there is no noise, 1 - maximum noise.

        range_decay:    float in [0,1] that indicate how distance affects observations.
                        the farther away the object is, the less confident will be the classifier.
                        0 - there is no decay, 1 - maximal range decay.

        lights:         list of light sources(x,y). The closer that the object is to the light source, the 
                        more confident the classifier will be.
                        if =None then no light effects will take place.
        
        light_grid:     np.array(M,N), in [0,1]. Light value for each grid cell, the more light, the better.
                        0 - low light, 1 - high light 

        obstacles:      list of obstacles(x,y) that interfer with the field of view. 
                        Observations that are occluded by them will say that there is no object there.
                        if =None then no obsticles are in the scene.
        
        num_alias:      number of aliased objects(places in the map that seem like the object in a certain 
                        direction, but not really are)
        
        alias:          The colloction of aliased objects and their directions 

    Methods:
        __init__():
        check_occlusions():

    """

    def __init__(self, M, N, F, Q, 
    noise_ind = 0, range_decay = 0, lights = None, obstacles = None,
    num_alias = 0):
        self.M = M 
        self.N = N
        self.F = F
        self.Q = Q
        self.noise_ind = noise_ind
        self.range_decay = range_decay
        self.lights = lights
        self.obstacles = obstacles
        self.num_alias = num_alias

        # make a board containing `Q` objects, randomly placed, also aliased objects
        self.randomize_objects()
        
        self.object_grid = np.zeros((self.M, self.N))
        for obj in self.objects:
            self.object_grid[obj[0],obj[1]] = 1

        # construct the light grid, summing over all light, 
        # it is proportaional to the squared distance between them.
        if self.lights is None:
            self.light_grid = np.ones_like(self.object_grid)
        else:
            self.light_grid = np.zeros_like(self.object_grid)
            for x in range(0, M):
                for y in range(0, N):
                    for xl,yl in lights:
                        d = l2_norm_sqared(xl, yl, x, y)
                        l = 1 / (1 + (d / 100))
                        self.light_grid[x, y] += l
        # clip values that are bigger then one
        self.light_grid[self.light_grid > 1] = 1


    def randomize_objects(self):
        self.objects = set()    # this only containes the objects positions
        self.alias = dict()     # this contains the alias position and direction the aliasing is strongest 
        
        # objects
        while len(self.objects) < self.Q:
            x = randint(0, self.M)
            y = randint(0, self.N)
            if self.obstacles is None or (x,y) not in self.obstacles:
                self.objects.add((x,y))

        # alias
        while len(self.alias) < self.num_alias:
            x = randint(0, self.M)
            y = randint(0, self.N)
            if (self.obstacles is None or (x,y) not in self.obstacles) and \
                (x, y) not in self.objects:
                
                theta = rand(2) - 0.5
                theta = theta / norm(theta)

                self.alias[(x, y)] = theta

        

    def check_occlusion(self, x0: int, y0: int, x1: int, y1: int) -> int:
        """check if there is an obstacle between (x0,y0) and (x1,y1), not inclusive.
        if there is returns 1, else 0"""
        
        if self.obstacles is None:
            return 0

        if x0 == x1 and y0 == y1:
            return 0

        if x0 == x1:
            if y1 < y0:
                y0, y1 = y1, y0

            for y in range(y0, y1+1):
                if (x0, y) in self.obstacles:
                    return 1
            return 0

        if y0 == y1:
            if x1 < x0:
                x0, x1 = x1, x0

            for x in range(x0, x1+1):
                if (x, y0) in self.obstacles:
                    return 1
            return 0
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        # run on x
        if (dx > dy):
            if (x1 < x0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            m = (y1 - y0) / (x1 - x0)
            y = y0
            for x in range(x0, x1+1):
                y_tag = round(y)
                if (x, y_tag) in self.obstacles:
                    return 1
                y += m
        # run on y
        else:
            if (y1 < y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            m = (x1 - x0) / (y1 - y0)
            x = x0
            for y in range(y0, y1+1):
                x_tag = round(x)
                if (x_tag, y) in self.obstacles:
                    return 1
                x += m

        return 0

    def get_alias(self, x0: int, y0: int, x1: int, y1: int) -> float:
            """Assumes robot is in (x0, y0), and checks if (x1, y1) is an aliasing point.
            if not, then returns 0, if it is, calculates the `cos(theta)` between the direction 
            of the strongest aliasing, and the robot's view point, and returns it.
            (only for positive valuse).
            
            Assumes that aliased objects cannot appear with un aliassed objects."""
            if (x1,y1) == (x0,y0) or (x1,y1) not in self.alias.keys():
                return 0
            alias_dir = self.alias[(x1, y1)]
            line_of_sight = np.array([x0 - x1, y0 - y1])
            line_of_sight = line_of_sight / norm(line_of_sight)
            
            cos_theta = np.dot(line_of_sight, alias_dir)
            if cos_theta <= 0: 
                return 0
                
            return cos_theta


    def generate_observation(self, x, y):
        if x < 0 or x >= self.M or y < 0 or y > self.N:
            raise ValueError(f"x={x} and y={y} should be positive,\
                 and smaller then the grid size, {self.M}, {self.N}.")

        Z = dict()

        start_x = max(x - self.F, 0)
        start_y = max(y - self.F, 0)
        end_x = min(x + self.F + 1, self.M)
        end_y = min(y + self.F + 1, self.N)
        
        for x1 in range(start_x, end_x):
            for y1 in range(start_y, end_y):
                # is there an object?
                I_y = int((x1,y1) in self.objects)
                # is there occlusion?
                I_occ = self.check_occlusion(x, y, x1, y1)
                # level of aliasing
                I_alias = self.get_alias(x, y, x1, y1)
                # amount of light
                l = self.light_grid[x1, y1]
                # distance
                r = l2_norm(x, y, x1, y1)
                
                Z[(x1,y1)] = \
                    ((I_y + I_alias) * (1 - I_occ) * l * np.exp(-self.range_decay * r)) \
                     + self.noise_ind * randn()
        
        return Z

        


def plot_field(ax: Axes, field: Field, light: bool = True):
    # plot the lighting
    if light:
        ax.imshow(field.light_grid.T,
            extent=(0,field.M,0,field.N),
            origin="lower", 
            alpha=0.5,
            cmap="Greys_r"
        )

    # for making the entire grid show up each time:
    ax.set_xlim(0,field.M)
    ax.set_xlim(0,field.N)
    ax.grid(True)
    ax.set_xticks(range(0, field.M + 1))
    ax.set_yticks(range(0, field.N + 1))
    
    # plot the limits
    limits_x = [0, 0, field.M, field.M]
    limits_y = [0, field.N, 0, field.N]
    ax.scatter(limits_x, limits_y, s=10, c="k", marker="+")
    # plot the objects locations with black 'X'
    objects_x = [(obj[0] + 0.5) for obj in field.objects]
    objects_y = [(obj[1] + 0.5) for obj in field.objects]
    ax.scatter(objects_x, objects_y, s=20, c="k", marker="x")
    # plot the light sources
    if light:
        if field.lights is not None:
            lights_x = [(light[0] + 0.5) for light in field.lights]
            lights_y = [(light[1] + 0.5) for light in field.lights]
            ax.scatter(lights_x, lights_y, s=20, c="y", marker="d")
    # plot the blocking objects locations
    if field.obstacles is not None:
        block_x = [(block[0] + 0.5) for block in field.obstacles]
        block_y = [(block[1] + 0.5) for block in field.obstacles]
        ax.scatter(block_x, block_y, s=30, c="k", marker="s")
    # plot the aliasing and the aliasing directions
    alias_poses = field.alias.keys()
    alias_x = [k[0] + 0.5 for k in alias_poses]
    alias_y = [k[1] + 0.5 for k in alias_poses]
    directions = np.array(list(field.alias.values()))
    ax.scatter(alias_x, alias_y, s=20, c="r", marker="*")
    q = ax.quiver(alias_x, alias_y, directions[:,0],directions[:,1],
        scale=0.5, scale_units='xy',angles='xy', color='r', alpha=0.5)
    

def plot_observation(ax: Axes, field: Field, Z: dict, x: int, y: int):
    def Z_prob(i,j):
        if (i,j) in Z:
            return Z[(i,j)]
        return 0.5

    Z_img = np.zeros((field.M, field.N))
    for i in range(0, field.M):
        for j in range(0, field.N):
            if (i,j) in Z:
                Z_img[i,j] = Z[(i,j)]
            else:
                Z_img[i,j] = 0.5

    # plot the lighting
    ax.imshow(Z_img.T,
        extent=(0,field.M,0,field.N),
        origin="lower", 
        alpha=0.5,
        cmap="Reds"
    )

    # for making the entire grid show up each time:
    ax.set_xlim(0,field.M)
    ax.set_xlim(0,field.N)
    ax.grid(True)
    ax.set_xticks(range(0, field.M + 1))
    ax.set_yticks(range(0, field.N + 1))
    ax.set_title("Observation")
    # plot the limits
    limits_x = [0, 0, field.M, field.M]
    limits_y = [0, field.N, 0, field.N]
    ax.scatter(limits_x, limits_y, s=10, c="k", marker="+")
    # plot the robots location with black 'X'
    ax.scatter([x + 0.5], [y + 0.5], s=20, c="k", marker="x")

def plot_occlusions(ax: Axes, field: Field, x: int, y: int):
    def Z_prob(i,j):
        if (i,j) in Z:
            return Z[(i,j)]
        return 0.5

    Z_img = np.zeros((field.M, field.N))
    for i in range(0, field.M):
        for j in range(0, field.N):
            if field.check_occlusion(x,y,i,j):
                Z_img[i,j] = 1

    # plot the lighting
    ax.imshow(Z_img.T,
        extent=(0,field.M,0,field.N),
        origin="lower", 
        alpha=0.5,
        cmap="Reds"
    )

    # for making the entire grid show up each time:
    ax.set_xlim(0,field.M)
    ax.set_xlim(0,field.N)
    ax.grid(True)
    ax.set_xticks(range(0, field.M + 1))
    ax.set_yticks(range(0, field.N + 1))
    ax.set_title("Field of view")
    # plot the limits
    limits_x = [0, 0, field.M, field.M]
    limits_y = [0, field.N, 0, field.N]
    ax.scatter(limits_x, limits_y, s=10, c="k", marker="+")
    # plot the robots location with black 'X'
    ax.scatter([x + 0.5], [y + 0.5], s=20, c="k", marker="x")


def explore_field(field: Field):
    # print the field
    fig, axes = plt.subplots(2,2,figsize=(15,15))
    plot_field(axes[0,0], field)
    tx = round(field.M / 3)
    ty = round(field.N / 3)
    x1,y1 = tx,ty
    x2,y2 = field.M - tx,ty
    x3,y3 = tx,field.N - ty

    Z1 = field.generate_observation(x1,y1)
    Z2 = field.generate_observation(x2,y2)
    Z3 = field.generate_observation(x3,y3)
    plot_observation(axes[0,1], field, Z1, x1, y1)
    plot_observation(axes[1,0], field, Z2, x2, y2)
    plot_observation(axes[1,1], field, Z3, x3, y3)
    plt.show()
    # make an observation from the center of the field

def explore_field_occlusions(field: Field):
    fig, axes = plt.subplots(2,2)
    plot_field(axes[0,0], field)
    x1,y1 = 1,1
    x2,y2 = 5,5
    x3,y3 = 7,3
    plot_occlusions(axes[0,1], field, x1, y1)
    plot_occlusions(axes[1,0], field, x2, y2)
    plot_occlusions(axes[1,1], field, x3, y3)
    plt.show()

if __name__ == "__main__":
    f = Field(15, 15, 3, 8,
        noise_ind=0.1,
        range_decay=0.1,
        lights=[(5,5)],
        obstacles=[(3,1),(3,2),(3,3),(3,0),(3,4),(10,3),(10,4),(10,5),(10,6),(10,7)],
        num_alias=5)

    explore_field(f)
