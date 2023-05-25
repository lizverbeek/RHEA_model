# -*- coding: utf-8 -*-

"""

Author: Liz Verbeek
Date: 09-03-2023

Realtor agent class of the RHEA model, based on Agent class of the MESA library.
The Realtor class contains the "stages" for the MESA StagedActivation scheduler.
It contains helper functions to update coefficients of the hedonic
price function and estimate property prices using this price function.

"""

import sys, os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from contextlib import contextmanager
from sklearn.linear_model import LinearRegression
from pykrige.rk import RegressionKriging
from mesa import Agent

YEARS_TRANS_HIST = 2                    # Time (in years) of market history used
                                        #   to update hedonic price function


@contextmanager
def suppress_stdout():
    """Suppress standard output from imported modules. """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class Realtor(Agent):
    """Realtor agent class of the RHEA model. """

    def __init__(self, model):
        """Initialize realtor agent.

        Args:
            model       : RHEA_model containing the realtor agent
        """
        super().__init__(model.next_id(), model)

        # Keep track of market history per timestep
        self.market_history = []
        # Keep track of number of timesteps used in updating price model
        self.history_count = 0

        # Initialize price estimation model
        method = self.model.price_method
        if method == "Regression":
            # Use regression coefficients from Bin et al. (2008)
            self.regression_coefs = np.array([11.337, 0.108, -0.011, -0.01,
                                              0.000094, 0.001, -0.00011, 0.03,
                                              0.00019, -0.059, -0.022, -0.078,
                                              -0.062, 0.314, -0.106, -0.00038,
                                              0.005, -0.001])
        elif method == "Regression kriging":
            # Initialize regression kriging model based on initial prices
            Y = np.array([prop.price for prop in self.model.parcels])
            X = np.array([prop.get_prop_chars(method) for prop in self.model.parcels])
            coords = np.array([prop.coords for prop in self.model.parcels])

            # Fit regression model
            m_regression = LinearRegression(fit_intercept=True)
            m_regression.fit(X, np.log(Y))
            results = sm.OLS(np.log(Y), sm.add_constant(X)).fit()
            # Save coefficients
            self.regression_coefs = np.append(m_regression.intercept_,
                                              m_regression.coef_)

            # Krige residuals from regression model
            with suppress_stdout():  # Avoid printing output from PyKrige library
                self.m_kriging = RegressionKriging(regression_model=m_regression,
                                                   n_closest_points=10,
                                                   variogram_model="spherical")
                self.m_kriging.fit(X, coords, np.log(Y))

        else:
            raise ValueError("Invalid price estimation method")

    def fit_price_model(self, steps, method):
        """Update regression coefficients for hedonic price estimation
           based on historical transactions in the area.

        Args:
            market_history (dict)   : Full market history
            method                  : Method used in hedonic function
                                      Options: "Regression" or "Regression kriging"
        """

        # Extract transactions for desired historical timespan
        T = self.model.schedule.steps
        transactions = {k: v for d in self.market_history[T - steps:]
                        for k, v in d.items()}

        # Get prices for transactions in market history
        Y = np.array([price["P_trans"] for price in transactions.values()])
        # Get property characteristics for specified price model
        X = np.array([prop.get_prop_chars(method) for prop in transactions])

        # Check significance of linear regression coefficients
        results = sm.OLS(np.log(Y), sm.add_constant(X)).fit()
        # Check if all coefficients are significant
        if np.all(results.pvalues < 0.05):
            print("Significant after", steps, "timesteps")
            # Update counter for saving number of timesteps used as history
            self.history_count = steps
            # Update regression coefficients
            self.regression_coefs = results.params

            # For regression kriging: also krige residuals
            if method == "Regression kriging":
                # Fit regression model
                m_regression = LinearRegression(fit_intercept=True)
                m_regression.fit(X, np.log(Y))
                # Krige residuals from regression model
                with suppress_stdout():  # Avoid printing output from PyKrige library
                    predict = m_regression.predict(X)
                    resid = np.log(Y) - predict
                    self.m_kriging = RegressionKriging(regression_model=m_regression,
                                                       n_closest_points=10,
                                                       variogram_model="spherical")
                    coords = np.array([prop.coords for prop in transactions])
                    self.m_kriging.fit(X, coords, resid)
        else:
            # Check if not already using full history
            if not T - steps <= 0:
                # Rerun price model fitting with more historical transactions
                self.fit_price_model(steps + 1, method)
            else:
                print("No significant fit found")

    def estimate_prices(self, props, method):
        """Estimate property prices from hedonic analysis.

        Args:
            props (list)        : Properties to estimate prices for
            method (string)     : Method used in hedonic function
                                  Options: "Regression" or "Regression kriging"
        Returns:
            prices (dict)       : Dictionary with estimated prices per property
        """

        # Collect property characteristics based on chosen method
        X = np.array([prop.get_prop_chars(method) for prop in props])

        if method == "Regression":
            # Compute hedonic prices from estimated regression coefficients
            H = sm.add_constant(X) @ self.regression_coefs
        elif method == "Regression kriging":
            # Regression part
            predict = sm.add_constant(X) @ self.regression_coefs
            # Kriging part
            coords = np.array([prop.coords for prop in props])
            H = self.m_kriging.krige_residual(coords) + predict

        else:
            raise ValueError("Invalid price estimation method")

        # Round and avoid zero prices
        price_dict = dict(zip(props, np.maximum(1e2, np.round(np.exp(H), -2))))
        return price_dict

    def stage0(self):
        """First stage of Realtor timestep:
           1) Update price estimation regression coefficients
           2) Update prices for properties that are on sale
        """

        price_method = self.model.price_method
        # Do not update in first timestep: no history yet
        if self.model.update_hedonics and self.model.schedule.steps > 0:
            # Update price estimation from recent transactions
            N_steps = int(YEARS_TRANS_HIST * self.model.kY)
            self.history_count = self.model.schedule.steps
            self.fit_price_model(N_steps, price_method)

        # Update property prices with updated hedonic function
        new_prices = self.estimate_prices(self.model.props_for_sale, price_method)

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
        self.market_history.append(self.model.transactions)
