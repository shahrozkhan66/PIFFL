# Physics-Informed Feature-to-Feature Learning (PIFFL)



<img src="assets/PIFFL.png?raw=true" width="600">



**Physics-Informed Feature-to-Feature Learning for Design-Space Dimensionality Reduction in Shape Optimisation**

[Shahroz Khan](https://www.shahrozkhan.info/)\*, [Andrea Serani](http://www.inm.cnr.it/people/andrea-serani/)\, [Matteo Diez](http://www.inm.cnr.it/people/matteo-diez/)\, [Panagiotis Kaklis](https://www.strath.ac.uk/staff/kaklispanagiotisprof/)

*American Institute of Aeronautics and Astronautics, Scitech 2021 Forum*

[[Paper]](https://drive.google.com/file/d/1xgluCc2a4qZWn0qVIYMTAayN5jPeR_BI/view?usp=sharing) [[Presentation]](-) [[Video]](-)


## Overview

This repository contains Matlab implementation of the algorithm framework for Physics-Informed Feature-to-Feature Learning for Dimensionality Reduction, including the implementation of Principal Component Analysis, Active-Subspace Method and Gaussian Process Regression.

## Test Pipelines

Following the different pipelines were tested:

<img src="assets/pipeline.png?raw=true" width="700">

This first pipeline is the Typical [Active Subspace Method](http://activesubspaces.org/), second pipeline is the proposed approach, which combaine widely used [Principal Compnent Analysis](https://doi.org/10.1016/j.cma.2014.10.042) and [Active Subspace Method](http://activesubspaces.org/) and last pipeline is the  [Physics-informed Principal Compnent Analysis](https://doi.org/10.2514/6.2017-3665) with [Active Subspace Method](http://activesubspaces.org/).
