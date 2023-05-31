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
import matplotlib.pyplot as plt

from mesa import Model
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector

from household import Household
from parcel import Parcel
from realtor import Realtor

YEARS_MORTGAGE = 30         # Mortgage duration (years)
TRAVEL_COSTS = 0.284        # Travel costs per unit of distance
NEW_BUYER_COEF = 0.7        # Additional buyer/seller ratio parameter
F_INCOME = 0.3              # Max fraction of income spent on housin
F_LEAVING = 0.7             # Fraction of sellers leaving area after sale
F_FLOOD_DAMAGE = 0.15       # Percentage of house value damaged in case of flood
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

    def __init__(self, random_seed, parcel_file, kY=2, F_sale=(0.25, 0.02),
                 HH_coastal_prefs=(0.5, 0.05), HH_RP_bias=(0, 0),
                 update_hedonics=True, price_method="Regression",
                 buyer_util_method="EU_v1", seller_mode="Random"):
        """Initialization of the RHEA model.

        Args:
            random_seed (int)          : Seed value for random number generation
            parcel_file (string)       : Path to csv file with parcel attributes
            kY (int)                   : Timesteps per year
            F_sale (tuple)             : Distribution of fraction of houses
                                         becoming available each timestep
            HH_coastal_prefs (tuple)   : Distribution of household preferences
                                         for coastal amenities (mean, std)
            HH_RP_bias (tuple)         : Distribution of household risk perception
                                         bias (mean, std)
            update_hedonics (boolean)  : Indicates whether parameters of hedonic
                                         function are updated every timestep
            price_method (string)      : Method used in hedonic function
                                         Options: "Regression" or "Regression kriging"
            buyer_util_method (string) : Method for computing utility of buyers
                                         Options: "EU_v1", "EU_v2", "PTnull",
                                                  "PT0", "PT1", "PT3"
            seller_mode (string)       : Mode to select households who decide to sell
                                         Options: "Random" or "Least utility"
        """

        # -- SCHEDULE INITIALIZATION -- #
        self.current_id = 0
        stage_list = ["stage0", "stage1", "stage2", "stage3", "stage4"]
        self.schedule = StagedActivation(self, stage_list)
        # -------------------------------

        # -- REGULATE STOCHASTICITY -- #
        # Separate random generator for household initialization
        self.rng_init = np.random.default_rng(random_seed)
        # Random generator for rest of dynamics
        self.rng = np.random.default_rng(random_seed + 1)
        if seller_mode == "Random":
            # Separate random generator if seller selection is random
            self.rng_random_sellers = np.random.default_rng(random_seed + 2)
        # ------------------------------

        # -- INITIALIZATION -- #
        # Save model setting
        self.update_hedonics = update_hedonics
        self.price_method = price_method
        self.buyer_util_method = buyer_util_method

        # Initialize model parameters
        self.kY = kY
        self.seller_mode = seller_mode
        self.new_buyer_coef = NEW_BUYER_COEF
        self.yM = YEARS_MORTGAGE
        self.F_sale = F_sale
        self.F_income = F_INCOME
        self.F_leaving = F_LEAVING
        self.market_subset = MARKET_SUBSET
        self.buyer_timelimit = BUYER_TIMELIMIT
        self.seller_timelimit = SELLER_TIMELIMIT
        self.not_moving_steps = np.ceil(NOT_MOVING_TIME * kY)
        self.trav_costs = TRAVEL_COSTS
        self.F_flood_damage = F_FLOOD_DAMAGE
        # --------------------------------

        # -- AGENT INITIALIZATION -- #
        # Save agent attributes
        self.HH_coastal_prefs = HH_coastal_prefs
        self.HH_RP_bias = HH_RP_bias
        # Select case study
        parcel_data = pd.read_csv(parcel_file)
        # Store income distribution data
        bins_df = pd.read_csv("Data/Income_distribution.csv")
        self.HH_income_params = bins_df["Bins"], bins_df["Probabilities"]
        # Create parcels and owners based on case data
        self.parcels = []
        self.households = []
        for i in range(len(parcel_data)):
            hh = self.add_household()
            parcel = self.add_parcel(parcel_data.iloc[i])
        # Match properties to household given property price and household income
        self.match_HH_props()

        # Initialize realtor and transaction history
        self.realtor = Realtor(self)
        self.schedule.add(self.realtor)
        self.transactions = {}
        # -----------------------------

        # Get total and average parcel prices for monthly rent estimation
        avg_price = (sum(parcel.price for parcel in self.parcels) /
                     len(self.parcels)) 
        self.avg_rent_per_step = avg_price / self.yM / self.kY

        # -- Initialize output collection -- #
        model_reporters = {"Households": lambda m: len(m.households),
                           "N_sellers": lambda m: count_by_status(m, "Seller"),
                           "N_buyers": lambda m: count_by_status(m, "Buyer"),
                           "Sold properties": lambda m: [prop.unique_id for prop
                                                         in m.transactions.keys()],
                           "Successful sellers": lambda m: [trans["Seller ID"]
                                                 for trans in m.transactions.values()],
                           "Successful buyers": lambda m: [trans["Buyer ID"]
                                                for trans in m.transactions.values()],
                           "P_ask": lambda m: [trans["P_ask"] for trans
                                               in m.transactions.values()],
                           "P_bid": lambda m: [trans["P_bid"] for trans
                                               in m.transactions.values()],
                           "P_trans": lambda m: [trans["P_trans"] for trans
                                               in m.transactions.values()],
                           "Trans history": lambda m: m.realtor.history_count
                           }
        if self.update_hedonics:
            model_reporters["Regression coefs"] = (lambda m:
                                                   m.realtor.regression_coefs)
        agent_reporters = {# ---------- HOUSEHOLD VARIABLES ------------#
                           "Type": (lambda a: "Household"
                                    if type(a) == Household else "Realtor"),
                           "Market status": (lambda a: a.market_status
                                             if type(a) == Household else None),
                           "Income": (lambda a: a.income
                                      if type(a) == Household else None),
                           "Property ID": (lambda a: a.property.unique_id
                                           if (type(a) == Household) and
                                              (a.property is not None) else None),
                           "Property price": (lambda a: a.property.price 
                                              if (type(a) == Household) and
                                                 (a.property is not None) else None),
                           "Property N sales": (lambda a: a.property.N_sales
                                                if (type(a) == Household) and
                                                   (a.property is not None) else None),
                           "Coastal pref": (lambda a: a.prefs["coast"]
                                            if (type(a) == Household) and
                                                (a.model.buyer_util_method == "EU_v1")
                                            else None),
                           "RP bias": (lambda a: a.RP_bias
                                       if type(a) == Household else None),
                           }
        self.datacollector = DataCollector(model_reporters=model_reporters,
                                           agent_reporters=agent_reporters)

        # Collect data for initialization phase
        self.datacollector.collect(self)

    # ------------------------------------------------ #
    # -------- MODEL INITIALIZATION FUNCTIONS -------- #
    # ------------------------------------------------ #
    def add_parcel(self, parcel_chars):
        """Create new parcel and add to the model. """
        parcel = Parcel(self, parcel_chars)
        self.parcels.append(parcel)
        return parcel

    def add_household(self):
        """Create new household and add to the model. """
        hh = Household(self, self.HH_coastal_prefs, self.HH_RP_bias)
        self.schedule.add(hh)
        self.households.append(hh)
        return hh

    def match_HH_props(self):
        """Match households to properties they can afford at initialization. """

        # Sort properties and households for most efficient matching
        props_avail = sorted(self.parcels, key=lambda prop: prop.price)
        households = sorted(self.households, key=lambda hh: hh.budget)
        for hh in households:
            # Get available properties (without lower bound)
            props = hh.get_affordable(props_avail, lower_limit=False)
            # If no properties available, reinitialize household
            while len(props) == 0:
                self.households.remove(hh)
                hh = self.add_household()
                props = hh.get_affordable(props_avail, lower_limit=False)
            # Randomly select property from affordable ones
            prop = self.rng_init.choice(props)
            # Connect property and owner both ways
            hh.property = prop
            prop.owner = hh
            # Remove from available property list
            props_avail.remove(prop)

    def remove_household(self, hh):
        """Remove a household object from the model. """
        self.households.remove(hh)
        self.schedule.remove(hh)
        del hh

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

    def register_transaction(self, prop, seller_id, buyer_id,
                             p_ask, p_bid, p_trans):
        """Register trade transaction and update property price.

        Args:
            prop            : Parcel object that was traded
            seller_id       : ID of selling household
            buyer_id        : ID of buying household
            p_trans         : Price for which property was sold
            p_bid           : Buyer's original bid price
            p_ask           : Seller's original ask price
        """
        # Save ask, bid and final transaction price per property
        self.transactions[prop] = {"Seller ID": seller_id,
                                   "Buyer ID": buyer_id,
                                   "P_ask": p_ask,
                                   "P_bid": p_bid,
                                   "P_trans": p_trans}
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
        self.transactions = {}

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
