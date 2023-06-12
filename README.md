# RHEA model

### Description
Python replication of the Risks and Hedonics in Empirical Agent-based (RHEA) land market model.
The RHEA model simulates the aggregated impact of household residential location choices under natural hazard risks in the United States. The model consists of a realtor agent providing adaptive price expectations for selling household agents in the housing market. Buyers evaluate available properties based on their own preferences for property characteristics, (flood) risk perceptions and budget. The trade process is a double auction in which sellers and buyers negotiate on a transaction price based on their ask and bid prices, respectively.

This implementation of the RHEA model is based on the original RHEA model written in NetLogo [[1]](#1) and on additional improvements on this model written in R [[2, 3, 4]](#2). The Python implementation is based on the original NetLogo and R source codes.

### Setup
**Requirements**: [Mesa](https://mesa.readthedocs.io/en/stable/), [NumPy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), [statsmodels](https://www.statsmodels.org/stable/index.html)

All required libraries can be installed with
```
pip install -r requirements.txt
```

**Usage instructions** To run the RHEA model, run the `run.py` file. In this file, the user can specify
- `years`: Number of years to run the RHEA model for
- `kY`: Number of timesteps per year (e.g. if `kY = 2`, each step spans a period of half a year)
- `runs`: Number of model replications
- `parcel_file`: CSV file containing parcel information. Column names should match the variables used in the specified price estimation and utility methods. For naming conventions, see attached example file `Parcel_chars.csv` or check the input lists specified in `parcel.py`.

If desired, the input parameters of the RHEA model can also be varied directly in the run file.\
Input parameters include:
- `F_sale` (mean, std): Fraction of properties becoming available each timestep. This may vary per region of interest. Default: (0.25, 0.02)
- `HH_coastal_prefs` (mean, std): Distribution of household preference for coastal amenities. Default: (0.5, 0.05)
- `HH_RP_bias` (mean, std): Distribution of household risk perception bias. Default: (0, 0)
- `update_hedonics`: Boolean indicating if hedonic price estimation function parameters should be updated every timestep or not
- `price_method`: Method to estimate property prices from transaction history. Options:
  - `regression`: Estimate property prices from historical transactions using a regression model, following the approach in [[1]](#1)
  - `regression kriging`: Estimate property prices from historical transactions using kriging, following the approach in [[3]](#2)
- `buyer_util_method`: Method to compute utility of properties for buyers to decide which property to bid on. Options:
  - `EU_v1`: expected utility based on preferences for spatial vs. composite goods and coastal amenities, following [[1]](#1).
  - `EU_v2`: expected utility based on, following [[3]](#3).
  - `PTnull`: utility function based on Prospect Theory; baseline, as described in [[[3]](#3).
  - `PT0`: utility function based on Prospect Theory, where the reference point is no floods experienced during residence.
  - `PT1`: Prospect Theory; reference point = single flood experienced during residence.
  - `PT3`: Prospect Theory: reference point = three floods experienced during residence.
- `seller_mode`: indicates how sellers are selected. Options:
  - `Random`: Sellers are selected randomly.
  - Least utility`: sellers are households with least utility in their current house.

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
