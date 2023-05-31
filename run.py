# -*- coding: utf-8 -*-

"""

@author: lizverbeek

This script is used to run the RHEA model and save model- and agent-level
variables for further analysis.

Please specify the timespan (years), number of timesteps per year (kY)
and number of runs below.

"""

import os
import time

from model import RHEA_Model


years = 30                      # Number of years to run model for
kY = 2                          # Number of timesteps per year
                                #   (e.g. kY = 2: every timestep equals half a year)
steps = years * kY
runs = 10
random_seeds = range(0, runs)

# Property price estimation method ("Regression" or "Regression kriging")
price_method = "Regression"
# Buyer utility method ("EU_v1", "EU_v2", "PTnull", "PT0", "PT1" or "PT3")
buyer_util_method = "EU_v1"
# CSV file with parcel characteristics (Beaufort or Greenville)
parcel_file = "Data/Parcel_chars_Beaufort.csv"

# Check compatibility of price method, utility function and parcel data
if parcel_file.endswith("Beaufort.csv") and buyer_util_method != "EU_v1":
    raise ValueError("Cannot apply utility function for the specified parcel data. "
                     "'EU_v2' and all versions of 'PT' can only be applied to the Greenville region."
                     "Please select a different method or region.")
elif parcel_file.endswith("Greenville.csv") and price_method == "Regression":
    raise ValueError("Cannot apply price model for the specified parcel data. "
                     "Initial full regression coefficients are only available for the Beaufort region."
                     "Please select a different method or region.")
elif parcel_file.endswith("Greenville.csv") and buyer_util_method == "EU_v1":
    raise ValueError("Cannot apply utility function for the specified parcel data. "
                     "Not all attributes of this utility function are known for the Greenville region."
                     "Please select a different method or region.")

# Adjust fraction of properties for sale each timestep to region
if parcel_file.endswith("Beaufort.csv"):
    F_sale = (0.25, 0.02)
elif parcel_file.endswith("Greenville.csv"):
    F_sale = (0.1, 0.01)

for i, random_seed in enumerate(random_seeds):
    print("Run num", i+1, "with random seed", random_seed)
    print("Case file:", parcel_file)

    tic = time.time()

    model = RHEA_Model(random_seed, parcel_file, F_sale=F_sale,
                       buyer_util_method=buyer_util_method,
                       price_method=price_method,
                       seller_mode="Random")

    for j in range(steps):
        print("# ------------ Step", j, "------------ #")
        model.step()

    toc = time.time()
    print("Model runtime:", toc-tic)

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

    # Store csv files (human-readable)
    model_vars.to_csv("Results/model_variables_[seed" + str(random_seed) + "].csv")
    agent_vars.to_csv("Results/agent_variables_[seed" + str(random_seed) + "].csv")

    print()

print("# -------- FINISHED", len(random_seeds), "runs -------- #")