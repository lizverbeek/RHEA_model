# RHEA model

### Description
Python replication of the Risks and Hedonics in Empirical Agent-based (RHEA) land market model.
The RHEA model simulates the aggregated impact of household residential
location choices under natural hazard risks. The model consists of realtor
and household agents forming ask and bid prices from adaptive price
expectations. Households are heterogeneous in income, risk perceptions and
preferences for coastal amenities.

This implementation of the RHEA model is based on the original RHEA model written in NetLogo [[1]](#1) and on additional improvements on this model written in R [[2, 3, 4]](#2). The Python implementation is based on the original NetLogo and R source codes.

### Setup
**Requirements**: [Mesa](https://mesa.readthedocs.io/en/stable/), [NumPy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), [statsmodels](https://www.statsmodels.org/stable/index.html)

All required libraries can be installed with
```
pip install -r requirements.txt
```

**Usage instructions** To run the RHEA model, run the `run.py` file. In this file, the user can specify
- `years`: Number of years to run the RHEA model for
- `kY`: Number of timesteps per years (e.g. if `kY = 2`, each step spans a period of half a year)
- `runs`: Number of model replications
- `parcel_file`: CSV file containing parcel information

If desired, the input parameters of the RHEA model can also be varied from the run file.\
Input parameters include:
- `new_buyer_coef`: Ratio of new buyers vs. new sellers every timestep
- `HH_coastal_prefs` (mean, std): Distribution of household preference for coastal amenities
- `HH_RP_bias` (mean, std): Distribution of household risk perception bias
- `update_hedonics`: Boolean indicating if hedonic price estimation function parameters should be updated every timestep or not
- `seller_mode` ("Random" or "Least utility"): Indicates whether each timesteps households who will try to sell are selected randomly or based on their utility.

### Notes
The RHEA model is still under active development, further improvements, additions and structural changes can be expected in the near future.

### References
<a id="1">[1]</a> 
Filatova, T. (2015).
[Empirical agent-based land market: Integrating adaptive economic behavior in urban land-use models. Computers, environment and urban systems, 54, 397-413.](https://www.sciencedirect.com/science/article/pii/S0198971514000714)

<a id="2">[2]</a> 
de Koning, K., Filatova, T., & Bin, O. (2018).
[Improved methods for predicting property prices in hazard prone dynamic markets.
Environmental and resource economics, 69, 247-263.](https://link.springer.com/article/10.1007/s10640-016-0076-5)

<a id="3">[3]</a> 
de Koning, K., Filatova, T., & Bin, O. (2017).
[Bridging the gap between revealed and stated preferences in flood-prone housing markets. Ecological economics, 136, 1-13.](https://www.sciencedirect.com/science/article/pii/S1462901111000657)

<a id="4">[4]</a>
De Koning, K., & Filatova, T. (2020).
[Repetitive floods intensify outmigration and climate gentrification in coastal cities. Environmental research letters, 15(3), 034008.](https://iopscience.iop.org/article/10.1088/1748-9326/ab6668/pdf)
