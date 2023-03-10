# -*- coding: utf-8 -*-

"""

@author: lizverbeek

Model class for the Risk and Hedonics in Empirical Agent-based (RHEA)
land market model.
The RHEA model simulates the aggregated impact of household residential
location choices under natural hazard risks. The model consists of realtor
and household agents forming ask and bid prices from adaptive price
expectations. Households are heterogeneous in income, risk perceptions and
preferences for coastal amenities.

The implementation of the RHEA model is based on the Python Mesa framework
for agent-based modeling (https://mesa.readthedocs.io/en/stable/) and is
written in Python version 3.10. 

"""

import numpy as np
import pandas as pd
import heapq

from mesa import Model
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector

from household import Household
from parcel import Parcel
from realtor import Realtor


YEARS_MORTGAGE = 30         # Mortgage duration (years)
TRAVEL_COSTS = 0.284        # Travel costs per unit of distance
F_SALE_DIST = (0.25, 0.02)  # Fraction of houses for sale per timestep (mean, std)
F_INCOME = 0.3              # Max fraction of income spent on housin
F_LEAVING = 0.7             # Fraction of sellers leaving area after sale
MARKET_SUBSET = 5           # Subset size of properties seen by buyers
BUYER_TIMELIMIT = 2         # Max time on the market for buyers (in years)
SELLER_TIMELIMIT = 2        # Max time on the market for sellers
NOT_MOVING_TIME = 0.5       # Wait time after recently moving (in years)


# ------------------------------------------------ #
# ----------- DATACOLLECTION FUNCTIONS ----------- #
# ------------------------------------------------ #
def count_by_status(model, market_status):
    """Get agent count by market status ("Buyer", Seller" or "Inactive")."""
    agent_count = sum(1 for hh in model.households
                      if hh.market_status == market_status)
    return agent_count


class RHEA_Model(Model):
    """Model class for the RHEA model. """

    def __init__(self, random_seed, parcel_file, kY=2, new_buyer_coef=0.7,
                 HH_coastal_prefs=(0.5, 0.05), HH_RP_bias=(0, 0),
                 update_hedonics=True, seller_mode="Least utility",
                 insurance=False):
        """Initialization of the RHEA model.

        Args:
            random_seed         : Seed value for random number generation
            parcel_file         : Path to csv file with parcel attributes
            kY                  : Timesteps per year
            new_buyer_coef      : Ratio between newcoming buyers and sellers
            HH_coastal_prefs    :
            update_hedonics     : Boolean indicating adaptive hedonic function
            seller_mode         : Mode to select households who decide to sell
                                  Options: "Random" or "Least utility"
            insurance           : Boolean indicating households in flood zone
                                  can have flood insurance
        """

        # -- SCHEDULE INITIALIZATION -- #
        self.current_id = 0
        stage_list = ["stage0", "stage1", "stage2", "stage3", "stage4"]
        self.schedule = StagedActivation(self, stage_list)
        # -------------------------------

        # -- REGULATE STOCHASTICITY -- #
        
        # self.random_seed = random_seed
        
        # Separate random generator for household initialization
        self.rng_init = np.random.default_rng(random_seed)
        # Random generator for rest of dynamics
        self.rng = np.random.default_rng(random_seed + 1)
        if seller_mode == "Random":
            # Separate random generator if seller selection is random
            self.rng_random_sellers = np.random.default_rng(random_seed + 2)
        # ------------------------------

        # -- INITIALIZATION -- #
        # Initialize model parameters
        self.kY = kY
        self.seller_mode = seller_mode
        self.insurance = insurance
        self.new_buyer_coef = new_buyer_coef
        self.yM = YEARS_MORTGAGE
        self.F_sale = F_SALE_DIST
        self.F_income = F_INCOME
        self.F_leaving = F_LEAVING
        self.market_subset = MARKET_SUBSET
        self.buyer_timelimit = BUYER_TIMELIMIT
        self.seller_timelimit = SELLER_TIMELIMIT
        self.not_moving_steps = np.ceil(NOT_MOVING_TIME * kY)
        self.trav_costs = TRAVEL_COSTS
        self.F_flood_damage = 0.15
        # --------------------------------

        # -- AGENT INITIALIZATION -- #
        # Save agent attributes
        self.HH_coastal_prefs = HH_coastal_prefs
        self.HH_RP_bias = HH_RP_bias
        # Select case study
        parcel_data = pd.read_csv(parcel_file)
        # Store income distribution data
        bins_df = pd.read_csv("Income_distribution.csv")
        self.HH_income_params = bins_df["Bins"], bins_df["Probabilities"]
        # Create parcels and owners based on case data
        self.parcels = []
        self.households = []
        for i in range(len(parcel_data)):
            hh = self.add_household()
            parcel = self.add_parcel(hh, parcel_data.iloc[i])
            hh.property = parcel

        # Initialize realtor and transaction history
        self.update_hedonics = update_hedonics
        self.realtor = Realtor(self)
        self.schedule.add(self.realtor)
        self.transactions = []
        # -----------------------------

        # Get total and average parcel prices for monthly rent estimation
        avg_price = (sum(parcel.price for parcel in self.parcels) /
                     len(self.parcels)) 
        self.avg_rent_per_step = avg_price / self.yM / self.kY

        # -- Initialize output collection -- #
        model_reporters = {"Households": lambda m: len(m.households),
                           "Sellers": lambda m: count_by_status(m, "Seller"),
                           "Buyers": lambda m: count_by_status(m, "Buyer"),
                           "Transactions": lambda m: m.transactions
                           # "Transaction prop IDs": (lambda m: [tr["Property ID"]
                           #                          for tr in m.transactions]),
                           # "AVG ask price": (lambda m: np.mean([tr["Ask price"]
                           #                   for tr in m.transactions])),
                           # "AVG bid price": (lambda m: np.mean([tr["Bid price"]
                           #                   for tr in m.transactions])),
                           # "AVG transaction price": (lambda m:
                           #                           np.mean([tr["Transaction price"]
                           #                           for tr in m.transactions])),
                           }
        if self.update_hedonics:
            model_reporters["Regression coefs"] = (lambda m:
                                                   m.realtor.regression_coefs)

        agent_reporters = {# ---------- HOUSEHOLD VARIABLES ------------#
                           "Type": (lambda a: "Household"
                                    if type(a) == Household else Realtor),
                           "Property ID": (lambda a: a.property.unique_id
                                           if (type(a) == Household) and
                                              (a.property is not None) else None),
                           "Property price": (lambda a: a.property.price 
                                              if (type(a) == Household) and
                                                 (a.property is not None) else None),
                           # "Status": (lambda a: a.market_status
                           #            if type(a) == Household else None),
                           "Preferences": (lambda a: a.prefs
                                           if type(a) == Household else None),
                           "RP bias": (lambda a: a.RP_bias
                                       if type(a) == Household else None),
                           "Income": (lambda a: a.income
                                      if type(a) == Household else None)
                           }

        self.datacollector = DataCollector(model_reporters=model_reporters,
                                           agent_reporters=agent_reporters)

        # Collect data for initialization phase
        self.datacollector.collect(self)

    # ------------------------------------------------ #
    # -------- MODEL INITIALIZATION FUNCTIONS -------- #
    # ------------------------------------------------ #
    def add_parcel(self, owner, parcel_chars):
        """Create new parcel and add to the model. """
        parcel = Parcel(self, owner, parcel_chars)
        self.parcels.append(parcel)
        return parcel

    def add_household(self):
        """Create new household and add to the model. """
        hh = Household(self, self.HH_coastal_prefs, self.HH_RP_bias)
        self.schedule.add(hh)
        self.households.append(hh)
        return hh

    def remove_household(self, hh):
        """Remove a household object from the model. """
        self.households.remove(hh)
        self.schedule.remove(hh)

    def update_parcel_age(self):
        """Update age of all parcels in the model. """
        for parcel in self.parcels:
            parcel.age += 1

    def update_monthly_costs(self):
        """Update average monthly costs for households. """
        avg_price = (sum(parcel.price for parcel in self.parcels) /
                     len(self.parcels))
        self.avg_rent_per_step = avg_price / self.yM / self.kY


    # ------------------------------------------------ #
    # -------------- TRADING FUNCTIONS --------------- #
    # ------------------------------------------------ #
    def assign_sellers(self, F_sale, mode="Random"):
        """Puts fraction of properties on the market.

        Args:
            F_sale      : Tuple (mean, std): normal distribution from which
                                             number of new sellers is drawn
            mode        : "Random" = randomly select sellers
                        : "Least utility" = select sellers from households with
                                            lowest utility
        """

        # Draw number of sellers
        sale_frac = self.rng.normal(F_sale[0], F_sale[1])
        n_sellers = round(sale_frac * len(self.parcels) / self.kY)
        # Select all households who can become sellers this timestep
        owners = [hh for hh in self.households
                  if (hh.market_status == "Inactive" and
                  hh.moving_wait_time == 0)]

        # Select properties for sale
        if mode == "Random":
            # Select random "Inactive" households to become sellers
            # Avoid selecting households who have recently moved
            sellers = self.rng.choice(owners, n_sellers, replace=False)
            print(sellers)
        elif mode == "Least utility":
            # Select sellers for which current house gives least utility
            util_owners = {hh: hh.compute_utility(hh.property) for hh in owners}
            sellers = heapq.nsmallest(n_sellers, util_owners, key=util_owners.get)
        else:
            print("Error assigning sellers: mode not recognized. \
                   Please specify 'Random' or 'Least utility'.")
        
        # Change status of selling households
        for hh in sellers:
            hh.market_status = "Seller"

        # Return properties for sale
        return sellers

    def create_new_buyers(self, F_buyers):
        """Create new agents with status 'buyer' entering the market.

        Args:
            F_buyers        : Tuple (mean, std): normal dist. from which
                                                 number of new buyers is drawn
        """
        
        # Draw number of buyers
        buy_frac = self.rng.normal(F_buyers[0], F_buyers[1])
        n_buyers = round(self.new_buyer_coef *
                         buy_frac * len(self.parcels) / self.kY)

        for i in range(n_buyers):
            hh = self.add_household()
            hh.market_status = "Buyer"

    def register_transaction(self, prop, p_ask, p_bid, p_trans):
        """Register trade transaction and update property price.

        Args:
            prop            : Parcel object that was traded
            p_trans         : Price for which property was sold
            p_bid           : Buyer's original bid price
            p_ask           : Seller's original ask price
        """

        # Save property ID, bid and ask price, final transaction price
        # and property attributes used for hedonic analysis
        transaction = {"Property ID": prop.unique_id,
                       "Ask price": p_ask,
                       "Bid price": p_bid,
                       "Transaction price": p_trans,
                       }
        if self.update_hedonics == True:
            # If needed for updating hedonic function: save property attributes
            prop_vars = prop.get_hedonic_vars()
            transaction["Property attributes"] = prop_vars

        # Add transaction to transaction list for current timestep
        self.transactions.append(transaction)
        # Update property price
        prop.price = p_trans

    def step(self):
        """Defines a single RHEA Model timestep. """

        # Select new sellers from current inactive property owners
        sellers = self.assign_sellers(self.F_sale, mode=self.seller_mode)
        # Save all properties for sale for efficient ask price computation
        self.props_for_sale = [hh.property for hh in self.households if
                               hh.market_status == "Seller"]

        # Create new buyers to enter the housing market in this area
        self.create_new_buyers(self.F_sale)

        # Reset transaction storing for this timestep
        self.transactions = []

        # Model step
        self.schedule.step()

        # Update parcel age each year
        if (self.schedule.steps % self.kY) == 0:
            self.update_parcel_age()

        # Update monthly rent/cost estimation
        self.update_monthly_costs()

        # Collect data
        self.datacollector.collect(self)
        print()
