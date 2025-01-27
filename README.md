## Introduction
This repository is created in order solve the optimization pricing problem at Nash equilibrium.

Suppose to have a network with $n_{od}$ origin-destination (OD) pairs, across which an amount of $n_i, 1\le i\le n_{od},$ users for each OD want to travel. The users can decide to take the trip using one of the common $n_{paths}$ toll paths or by the only toll-free path available for each OD.  
The problem can be modeled as a bilevel problem with an upper level consisting on a genetic algorithm, used to find the best costs **$T = (T_1, ..., T_{n_{paths}})$** to apply to the toll paths of the network and that maximize the following function

$$\sum_{1\le j\le n_{paths}}\sum_{1\le i\le n_{od}} n_i p^j_i T_j,$$

and a lower level, in which the probabilities $p_i^j$ of the users of the $i$-th OD of choosing the path $j$ on the network are calculated in order to minimize their time travel and cost, and to get a Nash equilibrium. 

Focusing on the upper level, the algorithms proposed for the project are the following:
- covariance matrix adaptation evolution strategies (CMA-ES);
- particle swarm optimization (PSO);
- fuzzy self tuning particle swarm optimization (FST-PSO);
- real-valued genetic algorithm with uniform mutation;
- real-valued genetic algorithm with gaussian mutation.

They are compared also with a vanilla algorithm.

## Structure
The main.py file is the executable.
By running it, an instance of the problem is initalised with a set of given values preset at the beginning of the file itself. After that, a class algorithm (real-valued genetic algorithm with uniform mutation by default, however another algorithm can be chosen) is called and its algorithm is run with the specific 
function for a predefined N_RUN times. The output is a dataframe that is then converted in a .csv file if the SAVE parameter at the beginning is True (it is False by default).

Each algorithm to run requires the instance.py file in the directory Instance and a file for the lower level. 

The vanilla (genetic_algorithm_torch.py in GA directory) and the real-valued genetic algorithms (in the GA/Project folder) are implemented using torch, while the others (in GA/Project) use numpy.
Concerning that, two different lower level files are proposed:
- lower_level_torch.py is used by vanilla and real-valued genetic algorithms;
- lover_level.py in GA/Project is used by CMA-ES, PSO and FST-PSO.

The code implemented for the graphs visulization can be found in Results/Graphs/Graphs.

The others directories do not concern the project.

## Requirements
The required packages and their versions can be found in requirements.txt.
