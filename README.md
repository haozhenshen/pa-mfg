# Renewable Enrgy Cedit Principal Agent Mean Field Game

This repo contains the implementation of multiperiod pricipal agent mean field game by using deep neural networks to model Forward Backward Stochastic Differential Equations.

## Usage
```bash
usage: main.py [--runner RUNNER] [--config CONFIG] [--seed SEED]
               [--run RUN] [--doc DOC] [--test] [--resume_training] [-o PLOTS_FOLDER]
```
There are currentlly two runners Naive3p_Runner, and the Fb3p_Runner.

* `Naive3p_Runner` The plain three period model.
* `Fb3p_Runner` The finite banking three period model.
