To be able to use this library when running python scripts from the terminal you need to add the library to your
 pythonpath which you can do by:
 1. Opening your bash file, probably ~/.bash_profile
 2. Add the command:
    export PYTHONPATH=$PYTHONPATH:path-to-where-tools-is-located
    to the bash_profile and save it. For example, if tools is in my product-analytics folder I should write
    export PYTHONPATH=$PYTHONPATH:/Users/alexander/product-analytics
 3. Done!

A good tutorial on how to use the pymc3 library is found here: https://pymc-devs.github.io/pymc3/notebooks/BEST.html


Example of how to use this library for an A/B test where we think the data is exponentially distributed with a uniform prior:


import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
from bayesian_tests import exponential_test, exponential_mean_plot, exponential_diff_plot

# Let's pretend this is CPV for treatment and control
treatment_group = np.random.exponential(scale=250, size=10000)
control_group = np.random.exponential(scale=200, size=20000)

trace = exponential_test(
    control_data=control_group[control_group > 0],
    treatment_data=treatment_group[treatment_group > 0],
    prior=pm.Uniform,
    prior_kw={'lower': 0, 'upper': 10})

thinning = 10

plt.figure(1)
exponential_mean_plot(trace[::thinning])
plt.figure(2)
exponential_diff_plot(trace[::thinning])

plt.show()

