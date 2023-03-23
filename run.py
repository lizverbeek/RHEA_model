# -*- coding: utf-8 -*-

"""

@author: lizverbeek

This script is used to run the RHEA model and save model- and agent-level
variables for further analysis.

Please specify the timespan (years), number of timesteps per year (kY)
and number of runs below.

"""

import os

from model import RHEA_Model


years = 50                  # Number of years to run model for
kY = 2                      # Number of timesteps per year
                            # (e.g. if kY = 2, every timestep equals half a year)
steps = years * kY + 1
runs = 10

parcel_file = "Beaufort_parcels.csv"
random_seeds = range(0, runs)


for i, random_seed in enumerate(random_seeds):
    print("Run num", i, "with random seed", random_seed)
    print("Case file:", parcel_file)

    model = RHEA_Model(random_seed, parcel_file,
                       seller_mode="Random")

    for j in range(steps):
        print("# ------------ Step", j, "------------ #")
        model.step()

    model_vars = model.datacollector.get_model_vars_dataframe()
    agent_vars = model.datacollector.get_agent_vars_dataframe()

    # -- STORING OUTPUT -- #
    if not os.path.isdir("Results"):
        os.makedirs(("Results"))

    # Store pickles (faster)
    model_vars.to_pickle("Results/model_variables_[seed" +
                         str(random_seed) + "].pkl")
    agent_vars.to_pickle("Results/agent_variables_[seed" +
                         str(random_seed) + "].pkl")

    # # Store csv files (human-readable)
    # model_vars.to_csv("Results/model_variables_[seed" + str(random_seed) + "].csv")
    # agent_vars.to_csv("results/agent_variables_[seed" + str(random_seed) + "].csv")

    print()

print("# -------- FINISHED", len(random_seeds), "runs -------- #")