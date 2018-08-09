# zeroth-order-HMM
## Description
All of the current libraries that model Hidden Markov Models(eg. hmmlearn in Python) assume that the HMM is of the 1st order i.e. the next state depends on the current state. 
This is not true all the time. There arise cases where the next state is independent of the current state. For eg, in Bayesian Stackelberg Games if we model the attacker types as states.
This repository contains two (python)classes which implement the above described version of Hidden Markov Models: One is for discrete observations labelled DiscreteHMM and one is for continuous observations labelled GaussianHMM

## Requirement
The only step that is required for using these classes is to import numpy as np

