# TGEC-NOC
NOC (Natal Optimization Code) is a stellar model optimization code using the TGEC (Toulouse-Geneva Evolution Code; 
Hui-Bon-Hoa, 2008) stellar evolution code. For oscillation frequency calculation, one can use PULSE (Brassard & 
Charpinet, 2008) or ADIPLS (Christensen-Dalsgaard, 2007) pulsation codes. The model optimization process is performed 
by the Levenberg-Marquardt (Levenberg, 1944; Marquardt, 1963) algorithm for least-squares estimation of nonlinear 
parameters.

## Requirements

* Python3, Numpy and Scipy.
* TGEC (Toulouse-Geneva Evolution Code): a free code for the calculation of the structure and evolution of solar-like stars;
* Pulse and/or Adipls: code for calculating adiabatic stellar oscillation modes (Adipls: http://astro.phys.au.dk/~jcd/adipack.n/);

## Installation

You must first install TGEC and Pulse and/or Adipls, please refer to their associated documentations.

Use `make` to install NOC.