# Conformal novelty detection for replicate point patterns
This GitHub repository contains the code library used to create the results of the paper `Conformal novelty detection for replicate point patterns with FDR or FWER control` submitted to the Journal of the American Statistical Association, January 2024.

Author: Martin Voigt Vejling

E-Mails: mvv@math.aau.dk;
         mvv@es.aau.dk;
         martin.vejling@gmail.com

Co-authors: Christophe A. N. Biscio and Adrien Mazoyer

## Contents
### Modules
- simulation_study/MySimulate.R `Base functionality to simulate the considered point processes.`

### Simulation scripts
- simulation_study/CMMCTest.R `Run a simulation study for the proposed CMMCTest procedure in which conformal p-values are computed and saved.`
- simulation_study/MMCTest.R `Run a simulation study for the naïve MMCTest procedure in which independent p-values are computed and saved.`

### Recreating figures
- Figure 1: numerical_FWER/polyFWER.py
- Figures 3, 4, 9, and 10: simulation_study/power_across_alpha.py
- Figure 5: simulation_study/power_across_lambda.py

## Software Setup

### Python dependencies
```
python 3
numpy
matplotlib
rds2py
pandas
```
