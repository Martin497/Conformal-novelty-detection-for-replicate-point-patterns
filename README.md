# Conformal novelty detection for replicate point patterns
This GitHub repository contains the code library used to create the results of the paper `Conformal novelty detection for replicate point patterns with FDR or FWER control` submitted.

Author: Martin Voigt Vejling

E-Mails: mvv@math.aau.dk;
         mvv@es.aau.dk;
         martin.vejling@gmail.com

Co-authors: Christophe A. N. Biscio and Adrien Mazoyer

## Contents
### Modules
- simulation_study/MySimulate.R `Base functionality to simulate the considered point processes.`
- simulation_study/MyFit.R `Base functionality to estimate parameters of the considered point processes.`
- simulation_study/multiple_testing_module.py `Base functionality to run multiple testing procedures.`

### Simulation scripts
- simulation_study/CMMCTest.R `Run a simulation study for the proposed CMMCTest procedure in which conformal p-values are computed and saved.`
- simulation_study/MMCTest.R `Run a simulation study for the naïve MMCTest procedure in which independent p-values are computed and saved.`
- simulation_study/CMMCTest_estimate.R `Run a simulation study for the proposed CMMCTest procedure using parametric data augmentation in which conformal p-values are computed and saved.`
- simulation_study/MMCTest_estimate.R `Run a simulation study for the naïve MMCTest procedure using parametric data augmentation in which independent p-values are computed and saved.`
- simulation_study/multipleGET.R `Run a simulation study using the multiple global envelope test by concatenating the functional summary statistics.`

### Sweat gland data
- sweat_gland/MMCTest_sweat_gland.R `Implementation of the naïve MMCTest procedure on the sweat gland data in which independent p-values are computed and saved.`
- sweat_gland/CMMCTest_sweat_gland.R `Implementation of the proposed CMMCTest procedure on the sweat gland data in which conformal p-values are computed and saved.`
- sweat_gland/graphical_interpretation_sweat_gland.R `Computing data needed for the graphical interpretation.`

#### From https://github.com/mikkoku/SweatPaper
- sweat_gland/abc_mcmc_sampler.jl `Implementation of the ABC-MCMC sampler for the generative model.`
- sweat_gland/generative_model.jl `Implementation of the generative model.`
- sweat_gland/sequential_model_with_noise.jl `Implementation of the sequential model with noise.`
- sweat_gland/readpattern.jl `Loading the data file.`

### Recreating figures
- Figure 1: numerical_FWER/polyFWER.py
- Figures 3, 4, 9, and 10: simulation_study/power_across_alpha.py
- Figure 5: simulation_study/power_across_lambda.py
- Figures 6 and 7: simulation_study/test_statistic_alpha.py
- Figures 14 and 15: simulation_study/global_test_across_alpha.py
- Figure 8: simulation_study/power_across_alphanm.py
- Figure 12: sweat_gland/open_pvalues.py
- Figure 13: sweat_gland/graphical_interpretation_sweat_gland.py

## Software Setup

### Python dependencies
```
python 3
numpy
matplotlib
rds2py
pandas
scipy
```

### R dependencies
```
R 4
spatstat
GET
glue
hash
lhs
```

