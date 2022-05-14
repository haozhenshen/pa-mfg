#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --mail=user=haozhen.shen@mail.utoronto.ca
#SBATCH --mail-type=ALL

python main.py


