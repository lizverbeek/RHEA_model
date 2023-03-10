# -*- coding: utf-8 -*-

"""

Author: Liz Verbeek
Date: 09-03-2023

Parcel class for the RHEA model.
This class represents a parcel owned by a household agent in the RHEA model.
Parcels are characterized from spatial input data.

This file contains helper functions to get the parcel characteristics
for the hedonic price analysis in the right format, and the set_insurance_values()
function to compute insurance premiums and coverage for this specific parcel.

"""

import numpy as np

class Parcel():
    """Parcel class of the RHEA model. """

    def __init__(self, model, owner, parcel_chars):
        """Initialize a parcel object.

        Args:
            model           : RHEA_model containing the parcel
            owner           : Owner of the property on this parcel
            parcel_chars    : Parcel characteristics from input data
        """

        self.model = model
        self.owner = owner

        # -- PARCEL CHARACTERISTICS -- #
        (self.unique_id, self.n_bathrooms, self.age, self.house_size,
         self.lot_size, self.new_home, self.post_firm,
         self.flood_prob_100, self.flood_prob_500, self.coastal_front,
         self.dist_amen, self.dist_CBD, self.dist_hwy, self.dist_park,
         self.price) = parcel_chars.values

        # Compute insurance premium and coverage in case of flood
        if self.model.insurance:
            self.IP, self.IC = self.set_insurance_values()
        else:
            self.IP = self.IC = 0

    def get_hedonic_vars(self):
        """Get property variables used in hedonic price estimation.
        
        Returns:
            hedonic_vars (list)     : List of hedonic function terms
        """

        # Return (predetermined) relevant variables, and, where
        # necessary: transformations of these variables.
        hedonic_vars = [1,
                        self.n_bathrooms, self.n_bathrooms**2,
                        self.age, self.age**2,
                        self.house_size, 1e-4*self.house_size**2,
                        self.lot_size, self.lot_size**2,
                        self.new_home, 
                        self.post_firm, 
                        self.flood_prob_100, 
                        self.flood_prob_500, 
                        self.coastal_front, 
                        np.log(self.dist_amen), 
                        np.log(self.dist_CBD), 
                        np.log(self.dist_hwy), 
                        np.log(self.dist_park)]
        return hedonic_vars

    def set_insurance_values(self):
        """Compute the annual insurance premium for a property based on
           flood probability and property price.
        
        Returns:
            IP (float)      : Insurance coverage in case of flood
            IC (float)      : Insurance premium
        """
        if self.flood_prob_100:
            IP = 526
            if self.price * 0.8 >= 5e4:
                IP += (0.8*self.price - 5e4) * 8e-4
            IC = IP * 0.763
        elif self.flood_prob_500:
            IP = 326
            if self.price * 0.8 >= 5e4:
                IP += (0.8*self.price - 5e4) * 14e-4
            IC = IP * 0.763
        else:
            IC = IP = 0
        return IP, IC
