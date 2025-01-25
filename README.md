## Introduction
This repository is created in order solve the optimization pricing problem at Nash equilibrium.
The problem can be modeled as a bilevel problem with an upper level with a genetic algorithm, used to find
the best cost to apply to the toll paths of a network, and a lower level, in which the probabilities 
of the users to choose a good path on the network are calculated in order to get a Nash equilibrium. 

Focusing on the upper level, the algorithms proposed for the project are the following:
- covariance matrix adaptation evolution strategies (CMA-ES);
- particle swarm optimization (PSO);
- fuzzy self tuning particle swarm optimization (FST-PSO);
- real-valued genetic algorithm with uniform mutation;
- real-valued genetic algorithm with gaussian mutation.

They are compared also with a vanilla algorithm.

## Structure
The main.py file is the executable.
By running it, an instance of the problem is initalised with a set of given values preset 
at the beginning of the file itself.
After that, a class algorithm (real-valued genetic algorithm with uniform mutation by default, 
however another algorithm can be chosen) is called and its algorithm is run with the specific 
function for a predefined N_RUN times.
The output is a dataframe that is then converted in a .csv file if the SAVE parameter 
at the beginning is True (it is False by default).

Each algorithm to run requires the instance.py file in the directory Instance and a file for the 
lower level. 

The vanilla (genetic_algorithm_torch.py in GA directory) and the real-valued genetic algorithms 
(in the GA/Project folder) are implemented using torch, while the others (in GA/Project) use numpy.
Concerning that, two different lower level files are proposed:
- lower_level_torch.py is used by vanilla and real-valued genetic algorithms;
- lover_level.py in GA/Project is used by CMA-ES, PSO and FST-PSO.

The code implemented for the graphs visulization can be found in Results/Graphs/Graphs.

The others directories do not concern the project.

## Requirements
The required packages and their versions can be found in requirements.txt.
