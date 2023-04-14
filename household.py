# -*- coding: utf-8 -*-

"""

Author: Liz Verbeek
Date: 09-03-2023

Household class of the RHEA model, based on the Agent class of the MESA library.
This class contains the "stages" for the MESA StagedActivation scheduler for
a household agent.

The Household class contains the following helper functions:
    - set_income()          : Draw household income from given income distribution

    For buyers:
    - compute_utility()     : Compute utility for given properties
    - get_affordable()      : Select properties within household budget range
    - find_best_prop()      : Select best property to bid on
    - place_bid()           : Determine bid price from property ask price

    For sellers:
    - negotiation()         : Trade negotiation process
    - transfer_property()   : Change property ownership from seller to buyer

"""

import math
import bisect
import numpy as np

from scipy.stats import truncnorm

from mesa import Agent

LOWER_BID_LIMIT = 0.8           # Lower limit of property_price/HH_budget
                                #   ratio to select properties to bid on
SPATIAL_PREFS = (0.3, 0.05)     # Household preferences for spatial goods


class Household(Agent):
    """Household agent class of the RHEA model. """

    def __init__(self, model, coastal_prefs, RP_bias):
        """Initialize household agent.

        Args:
            model (Model)           : Model object containing the household agent
            coastal_prefs (tuple)   : Distribution of household preferences
                                      for coastal amenities (mean, std)
            RP_bias (tuple)         : Distribution of flood risk perception bias
                                      (mean, std)
        """

        super().__init__(model.next_id(), model)

        # -- Financial attributes -- #
        # Draw household income values from predefined empirical distribution
        bins, cum_weights = self.model.HH_income_params
        self.set_income(bins, cum_weights, mode="Bins")
        
        # Total budget household can spend on housing
        self.budget = int(self.model.F_income * self.income * self.model.yM)
        
        # -- Housing market attributes -- #
        self.property = None
        self.market_status = "Inactive"
        self.n_tr_nosuccess = 0
        self.moving_wait_time = 0

        # -- Perceptions and preferences -- #
        # HH preferences of spatial over composite goods and coastal amenities
        pref_spat = self.model.rng_init.normal(SPATIAL_PREFS[0], SPATIAL_PREFS[1])
        pref_amen = self.model.rng_init.normal(coastal_prefs[0], coastal_prefs[1])
        self.prefs = {"spat": pref_spat,
                      "comp": 1 - pref_spat,
                      "coast": pref_amen}

        # Risk perception bias
        self.RP_bias = self.model.rng_init.normal(RP_bias[0], RP_bias[1])

    def set_income(self, *args, min_income=8362, mode="Bins"):
        """Assign income value to household.

        Args:
            *(mean, std)            : Mean and std for "Normal" mode
            *(bins, cum_weights)    : Bin ranges and cumulative weights
                                      that define empirical distribution
            min_income (int)        : HH income lower bound
            mode (string)           : Method to draw income values.
                                      Options: "Bins" or "Normal"
        """

        if mode == "Bins":
            # Draw income values from predefined empirical distribution.
            # Empirical distribution based on Filatova et al. (2015)

            # Select income bins based on given probablities
            bins, cum_weights = args
            idx = bisect.bisect(cum_weights, self.model.rng_init.random())
            # Draw random income value from corresponding bin
            self.income = (min_income if idx == 0 else
                           np.round(self.model.rng_init.integers(bins[idx - 1],
                                                                 bins[idx]), -1))

        elif mode == "Normal":
            # Draw income value from normal distribution
            mean, std = args[0]
            self.income = truncnorm.rvs(a=(min_income - mean)/std, b=np.inf,
                                        loc=mean, scale=std,
                                        random_state=self.model.rng_init)

    def compute_utility(self, prop, mode="EU"):
        """Compute the utility of a certain property.

        Args:
            prop (Parcel)        : Property to compute utility for
            mode (string)        : Method for utility estimation.
                                   Options:
                                   "EU" = expected utility
        """
        if mode == "EU":
            # Compute housing goods as geometric avg of
            # lot- and structure size (in feet)
            spat_goods = math.sqrt(prop.lot_size * 43560 * prop.house_size)
            
            # Compute composite goods
            TC = self.model.trav_costs * prop.dist_CBD
            comp_goods_nofl = (self.income - TC - prop.IP -
                               prop.price/self.model.yM)
            comp_goods_fl = (self.income - TC - prop.IP + prop.IC -
                             self.model.F_flood_damage * prop.price -
                             prop.price/self.model.yM)

            # Avoid composite goods < 0
            if comp_goods_fl < 0:
                comp_goods_fl = 1
            if comp_goods_nofl < 0:
                comp_goods_nofl = 1

            # Compute utility in case of flood and in case of no flood
            U_flood = (spat_goods**self.prefs["spat"] *
                       comp_goods_fl**self.prefs["comp"] *
                       prop.prox_amen**self.prefs["coast"])
            U_no_flood = (spat_goods**self.prefs["spat"] *
                          comp_goods_nofl**self.prefs["comp"] *
                          prop.prox_amen**self.prefs["coast"])

            # Compute expected utility from U_flood and U_no_flood
            flood_prob = (0.01 if prop.flood_prob_100 else 0.002
                          if prop.flood_prob_500 else 0)
            # When in flood plain: compute subjective risk perception
            if flood_prob > 0:
                RP = flood_prob + self.RP_bias
                # Avoid values below 0 or above 1
                if RP < 0:
                    RP = 0
                elif RP > 1:
                    RP = 1
            else:
                RP = flood_prob
            EU = (RP * U_flood + (1 - RP) * U_no_flood)

        return EU

    def check_lower_price(self):
        """Lower ask price for sellers with unsuccessful trade attempts. """
        if self.n_tr_nosuccess > 1:
            self.ask_price = self.ask_price * 0.99
            # Update property price
            self.property.price = self.ask_price

    def get_affordable(self, properties, lower_limit=True):
        """Get affordable properties based on household budget.

        Args:
            properties (list)       : List of available properties
            lower_limit (Boolean)   : If TRUE, select properties above lower limit
        Returns:
            props_afford (list)     : List of properties within budget range
        """

        if lower_limit:
            # Compare property prices to lower limit and household budget
            props_afford = filter(lambda prop:
                                  LOWER_BID_LIMIT <= prop.price/self.budget <= 1,
                                  properties)
        else:
            # Compare property prices only to household budget
            props_afford = filter(lambda prop: prop.price <= self.budget,
                                  properties)
        return list(props_afford)

    def find_best_prop(self, properties):
        """From specified list, find best property to bid on.

        Args:
            properties (list)       : List of properties to choose from
        Returns:
            best_prop (Parcel)      : Best property for this household
        """

        # Compute expected utility for given properties
        utils = {prop: round(self.compute_utility(prop), -3)
                 for prop in properties}
        # Get property with highest utility
        best_props = list(k for k, v in utils.items()
                          if v == max(utils.values()))
        # If multiple properties with highest utility: select cheapest
        if len(best_props) > 1:
            prices = {prop: prop.price for prop in best_props}
            best_prop = min(prices, key=prices.get)
        else:
            best_prop = best_props[0]

        return best_prop

    def place_bid(self, ask_price):
        """Place bid on desired property.
        
        Args:
            ask_price (int)     : Seller ask price
        Returns
            bid (int)           : Buyer bid price
        """
        
        # Draw random number between -5% and +5% from ask price,
        # constrained by household budget.
        bid_low = ask_price * 0.95
        bid_high = min(ask_price * 1.05, self.budget)
        bid = np.round(self.model.rng.integers(bid_low, bid_high), -3)
        return bid

    def negotiation(self, highest_bidder, highest_bid):
        """Trade negotiation process.

        Args:
            highest_bidder (Household)  : Household agent with the highest bid
            highest_bid (float)         : Value of the highest bid
        """
        if highest_bid >= self.ask_price:
            # If bid is higher than ask price: successful trade
            self.model.register_transaction(self.property,
                                            self.unique_id,
                                            highest_bidder.unique_id,
                                            self.ask_price,
                                            highest_bid,
                                            highest_bid)
            self.transfer_property(highest_bidder, self.property)
        else:
            diff = self.ask_price - highest_bid
            D_neg_buyer = self.model.avg_rent_per_step
            D_neg_seller = (self.property.price
                            / self.model.yM / self.model.kY
                            * (1 + self.n_tr_nosuccess))
            # If seller received multiple offers: buyer reconsiders
            if self.bids_received and diff <= D_neg_buyer:
                # Buyer accepts higher ask price
                self.model.register_transaction(self.property,
                                                self.unique_id,
                                                highest_bidder.unique_id,
                                                self.ask_price,
                                                highest_bid,
                                                self.ask_price)
                self.transfer_property(highest_bidder, self.property)
            
            # If seller received only one offer: seller reconsiders
            elif not self.bids_received and diff <= D_neg_seller:
                # Seller accepts lower bid price
                self.model.register_transaction(self.property,
                                                self.unique_id,
                                                highest_bidder.unique_id,
                                                self.ask_price,
                                                highest_bid,
                                                highest_bid)
                self.transfer_property(highest_bidder, self.property)
            else:
                # Unsuccessful attempt for seller and highest bidder
                self.n_tr_nosuccess += 1
                highest_bidder.n_tr_nosuccess += 1

    def transfer_property(self, buyer, prop):
        """Transfer a property from the current owner (self) to a buyer.

        Args:
            buyer (Household)   : Buyer to transfer property to
            prop (Parcel)       : Parcel object to transfer
        """

        # Transfer house to new owner
        prop.N_sales += 1
        prop.owner = buyer
        buyer.property = prop
        self.property = None

        # Reset buyers trade attributes and change market status
        del buyer.desired_prop
        buyer.n_tr_nosuccess = 0
        buyer.market_status = "Inactive"
        buyer.moving_wait_time = self.model.not_moving_steps

        # Sellers lave town with 70% probability, otherwise become buyers
        if self.model.rng.random() < self.model.F_leaving:
            self.model.remove_household(self)
            del self
        else:
            self.market_status = "Buyer"
            # If seller stays: reset seller trade attributes
            self.n_tr_nosuccess = 0
            del self.ask_price, self.bids_received

    def stage0(self):
        pass

    def stage1(self):
        """First stage of household step:
           1) Get market information and set ask prices (sellers).
        """

        # Check if household has recently moved and has to wait for new trade
        if self.moving_wait_time > 0:
            self.moving_wait_time -= 1

        # If household wants to sell, get market information and set ask price
        if self.market_status == "Seller":
            self.ask_price = self.property.price
            # Check if households should lower price
            self.check_lower_price()
            self.bids_received = {}

    def stage2(self):
        """Second stage of household step:
           1) Select affordable properties (buyers)
           2) Select desired property (highest utility) from subset of
              available properties (buyers)
           3) Place bid (buyers)
        """
        if self.market_status == "Buyer":
            # Get properties within buyers budget range
            props = self.get_affordable(self.model.props_for_sale)

            # Check if there are any properties to bid on
            if props:
                # Get random subset of affordable properties
                if len(props) > self.model.market_subset:
                    props = self.model.rng.choice(props, self.model.market_subset,
                                                  replace=False)
                # Find best property and place bid
                self.desired_prop = self.find_best_prop(props)
                bid = self.place_bid(self.desired_prop.price)
                self.desired_prop.owner.bids_received[self] = bid

            else:
                # If no affordable properties: register as unsuccessful
                self.n_tr_nosuccess += 1

    def stage3(self):
        """Third stage of household step:
            1) Check if any bids received (seller)
            2) Select highest bid (seller)
            3) Start negotiation process (seller --> buyer)
        """

        if self.market_status == "Seller":
            # Check if seller received any bids
            self.bids_received = self.bids_received
            if self.bids_received:
                # Get highest bid and bidder, register others as unsuccessful
                highest_bidder = max(self.bids_received,
                                     key=self.bids_received.get)
                highest_bid = self.bids_received.pop(highest_bidder)
                for buyer in self.bids_received.keys():
                    buyer.n_tr_nosuccess += 1
                id_hist = self.unique_id
                # Start negotiation between seller and highest bidder
                self.negotiation(highest_bidder, highest_bid)

            # If no bids are placed, register as unsuccessful for seller
            else:
                self.n_tr_nosuccess += 1

    def stage4(self):
        """Fourth stage of household step:
            1) If unsuccessful search > time limit, leave area (buyers)
            2) If unsuccessful sellin > time limit, give up (sellers)
        """

        buyer_limit = self.model.buyer_timelimit * self.model.kY
        seller_limit = self.model.seller_timelimit * self.model.kY

        # Buyers leave market after predefined timelimit
        if self.market_status == "Buyer" and self.n_tr_nosuccess >= buyer_limit:
            self.model.remove_household(self)
            del self

        # Sellers give up on selling after predefined timelimit
        elif (self.market_status == "Seller" and
              self.n_tr_nosuccess >= seller_limit):
            self.market_status = "Inactive"
            self.n_tr_nosuccess = 0
            self.moving_wait_time = self.model.not_moving_steps