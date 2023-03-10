# -*- coding: utf-8 -*-

"""

Author: Liz Verbeek
Date: 09-03-2023

Realtor agent class of the RHEA model, based on Agent class of the MESA library.
The Realtor class contains the "stages" for the MESA StagedActivation scheduler.
It contains helper functions to update coefficients of the hedonic
price function and estimate property prices using this price function.

"""

import numpy as np
import statsmodels.api as sm

from mesa import Agent


HEDONIC_UPDATE_HISTORY = 2      # Time (in years) of market history used
                                # to update hedonic price function

class Realtor(Agent):
    """Realtor agent class of the RHEA model. """

    def __init__(self, model):
        """Initialize realtor agent.

        Args:
            model       : RHEA_model containing the realtor agent
        """
        super().__init__(model.next_id(), model)

        # Keep track of market history per timestep
        self.market_history = {}

        # Original regression coefficients from Bin et al. (2008)
        self.regression_coefs = np.array([11.337, 0.108, -0.011, -0.01,
                                          0.000094, 0.001, -0.00011, 0.03,
                                          0.00019, -0.059, -0.022, -0.078,
                                          -0.062, 0.314, -0.106, -0.00038,
                                          0.005, -0.001])

    def update_hedonic_func(self, market_history):
        """Update regression coefficients for hedonic price estimation
           based on historical transactions in the area.

        Args:
            market_history (dict)       : Full market history
        """

        # Get desired historical timespan
        N_hist_steps = HEDONIC_UPDATE_HISTORY * self.model.kY
        start = int(self.model.schedule.time) - N_hist_steps
        # If available history < desired timespan: take full available history
        if start < 0:
            start = 0
        stop = int(self.model.schedule.time)
        # Extract relevant transactions from market_history
        market_history = [market_history.get(t) for t in range(start, stop)][0]

        # Fit regression model on transaction history
        Y = np.array([trans["Transaction price"] for trans in market_history])
        X = np.array([trans["Property attributes"] for trans in market_history])
        results = sm.OLS(np.log(Y), X).fit()
        # Adjust regression coefficients
        self.regression_coefs = results.params

    def estimate_prices(self, properties):
        """Estimate property prices from hedonic analysis.

        Args:
            properties (list)   : Properties to estimate prices for
        Returns:
            prices (dict)       : Dictionary with estimated prices per property
        """

        # Compute hedonic prices from estimated regression coefficients
        property_chars = [prop.get_hedonic_vars() for prop in properties]
        H = property_chars @ self.regression_coefs
        # Round and avoid zero prices
        prices = dict(zip(properties, np.maximum(1e2, np.round(np.exp(H), -2))))
        return prices

    def stage0(self):
        """First stage of Realtor timestep:
           1) Update price estimation regression coefficients
           2) Update prices for properties that are on sale
        """

        # Update price estimation from recent transactions
        if self.model.update_hedonics == True:
            # Do not update in first timestep: no history yet
            if self.model.schedule.steps > 0:
                self.update_hedonic_func(self.market_history)

        # Update property prices with updated hedonic function
        new_prices = self.estimate_prices(self.model.props_for_sale)
        for prop in self.model.props_for_sale:
            prop.price = new_prices[prop]

    def stage1(self):
        pass

    def stage2(self):
        pass

    def stage3(self):
        pass

    def stage4(self):
        """Fourth stage of Realtor timestep:
           1) Save transactions from current timestep in market history
        """
        self.market_history[self.model.schedule.steps] = self.model.transactions
