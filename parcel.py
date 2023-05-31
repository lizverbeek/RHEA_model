# -*- coding: utf-8 -*-

"""

Author: Liz Verbeek
Date: 09-03-2023

Parcel class for the RHEA model.
This class represents a parcel owned by a household agent in the RHEA model.
Parcels are characterized from spatial input data.

This file contains helper functions to get the parcel characteristics for
hedonic price analysis in the desired format for the specified price model.

"""

import numpy as np

from scipy.special import comb

# Parcel characteristics for both price estimation methods (regression and kriging)
REGR_CHARS = ["AGE", "BATHROOMS", "HOUSESIZE", "LOTSIZE", "NEWHOME", "POSTFIRM",
              "FIRSTROW", "DISTAMEN", "DISTCBD", "DISTHWY", "DISTPARK", "PRICE_REGR"]
KRIGING_CHARS = ["COORDS_X", "COORDS_Y", "AGE", "HOUSESIZE", "LOTSIZE",
                 "BEDROOMS", "PRICE_KRIGING"]

# Parcel characteristics used in buyer utility functions
EU_V1_CHARS = ["PROXAMEN", "DISTCBD"]
EU_V2_CHARS = PT_CHARS = ["AGEnorm", "HOUSESIZEnorm", "LOTSIZEnorm",
                          "BEDROOMSnorm", "RESIDnorm"]

# Other constants
N_FLOODS = np.array([0,1,2,3])  # Flood experience scenarios (number of floods)


class Parcel():
    """Parcel class of the RHEA model. """

    def __init__(self, model, parcel_chars):
        """Initialize a parcel object.

        Args:
            model           : RHEA_model containing the parcel
            parcel_chars    : Parcel characteristics read from input data
        """

        self.model = model
        self.N_sales = 0

        # -- PARCEL CHARACTERISTICS -- #
        # Extract relevant parcel characteristics
        self.unique_id = parcel_chars["ID"]
        self.dflood_100 = parcel_chars.get("DFLOOD100")
        self.dflood_500 = parcel_chars.get("DFLOOD500")
        # Get flood probability from flood plain dummies
        self.flood_prob = 0
        if self.dflood_100:
            # Check if 1:100 flood plain exists in dataset
            self.flood_prob += 0.01 * self.dflood_100
        if self.dflood_500:
            # Check if 1:500 flood plain exists in dataset
            self.flood_prob += 0.002 * self.dflood_500

        # Read parcel characteristics for price estimation
        if self.model.price_method == "Regression":
            (self.age, self.n_bathrooms, self.house_size,
             self.lot_size, self.new_home, self.post_firm,
             self.coastal_front, self.dist_amen, self.dist_CBD,
             self.dist_hwy, self.dist_park, self.price) = parcel_chars[REGR_CHARS]
        elif self.model.price_method == "Regression kriging":
            (x, y, self.age, self.house_size, self.lot_size,
             self.n_bedrooms, self.price) = parcel_chars[KRIGING_CHARS]
            self.coords = (x, y)
        else:
            raise ValueError("Invalid price method. Please specify "
                             "'Regression' or 'Regression kriging'")

        # Read attributes for utility function
        util_method = self.model.buyer_util_method
        if util_method == "EU_v1":
            self.prox_amen, self.dist_CBD = parcel_chars[EU_V1_CHARS]
        elif util_method == "EU_v2":
            (self.age_norm, self.house_size_norm, self.lot_size_norm,
             self.n_bedrooms_norm, self.resid_norm) = parcel_chars[EU_V2_CHARS]
            # Get probability of experiencing one or more floods during residence
            self.P_floods = self.n_flood_prob()

        elif util_method.startswith("PT"):
            (self.age_norm, self.house_size_norm, self.lot_size_norm,
             self.n_bedrooms_norm, self.resid_norm) = parcel_chars[PT_CHARS]
            # Get probability of experiencing one or more floods during residence
            self.P_floods = self.n_flood_prob()
        else:
            raise ValueError("Invalid buyer utility method. Please specify "
                             "'EU_v1', 'EU_v2', 'PTnull', 'PT0', 'PT1' or 'PT3'")

    def n_flood_prob(self, N_floods=N_FLOODS):
        """Computes probability of N floods occuring during a given number of years.
        
        Args:
            N_floods (Vector)      : Number of flood occurences
        Returns:
            P_floods (list)        : Probability of flood occurences
        """

        # Compute (average) residence time from F_sale and length of timestep
        res_time = 1/(self.model.kY * self.model.F_sale[0])
        P_floods = (self.flood_prob**N_floods *
                    (1 - self.flood_prob)**(res_time - N_floods) *
                    comb(res_time, N_floods))
        return P_floods

    def get_prop_chars(self, method):
        """Get property characteristics used in hedonic price estimation.
        
        Args:
            method (string)       : Method used in hedonic function
                                    Options: "Regression" or "Regression kriging"
        Returns:
            prop_chars (list)     : List of hedonic function terms
        """

        if method == "Regression":
            # Return relevant property characteristics for regression
            prop_chars = [self.n_bathrooms, self.n_bathrooms**2,
                          self.age, self.age**2,
                          self.house_size, 1e-4*self.house_size**2,
                          self.lot_size, self.lot_size**2,
                          self.new_home,
                          self.post_firm, 
                          self.dflood_100,
                          self.dflood_500,
                          self.coastal_front,
                          np.log(self.dist_amen),
                          np.log(self.dist_CBD),
                          np.log(self.dist_hwy),
                          np.log(self.dist_park)]
        
        elif method == "Regression kriging":
            # Return relevant property characteristics for regression kriging
            prop_chars = [self.age,
                          np.log(self.house_size),
                          np.log(self.lot_size),
                          self.n_bedrooms,
                          np.ceil(self.flood_prob)]

        else:
            raise ValueError("Invalid price estimation method")

        return prop_chars
