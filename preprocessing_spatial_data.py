# -*- coding: utf-8 -*-

"""

@author: lizverbeek

Read shapefiles and extract relevant data for the RHEA model

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from pykrige.rk import RegressionKriging


def replace_zero_by_avg(df, column):
    avg = round(df[df[column] != 0][column].mean(), 2)
    df.loc[df[column] == 0, column] = avg
    return df

def norm_chars(df, variables):
    for var in variables:
        new_var = str(var) + "norm"
        df[new_var] = (df[var] - df[var].min())/(df[var] - df[var].min()).max()
        if var == "AGE":
            df[new_var] = 1 - df[new_var]
    return df

def initial_regression(parcels):
    """Compute hedonic price estimate based on property attributes.
       Regression coefficients are taken from Bin et al. (2008).

    Args:
        parcels        : Properties to estimate price for
    """

    # Add constant and second order terms
    parcels["CONSTANT"] = np.ones(len(parcels))
    parcels["BATHROOM_SQ"] = parcels["BATHROOMS"]**2
    parcels["AGE_SQ"] = parcels["AGE"]**2
    parcels["HOUSESIZE_SQ"] = parcels["HOUSESIZE"]**2*1e-4
    parcels["LOTSIZE_SQ"] = parcels["LOTSIZE"]**2
    parcels["LN(DISTAMEN)"] = np.log(parcels["DISTAMEN"])
    parcels["LN(DISTCBD)"] = np.log(parcels["DISTCBD"])
    parcels["LN(DISTHWY)"] = np.log(parcels["DISTHWY"])
    parcels["LN(DISTPARK)"] = np.log(parcels["DISTPARK"])

    # Regression coefficients established for the original RHEA model
    regression_coefs = np.array([11.337, 0.108, -0.011, -0.01, 0.000094, 0.001,
                                -0.00011, 0.03, 0.00019, -0.059, -0.022, -0.078,
                                -0.062, 0.314, -0.106, -0.00038, 0.005, -0.001])

    # Select relevant characteristics in right order
    chars = parcels[["CONSTANT", "BATHROOMS", "BATHROOM_SQ", "AGE", "AGE_SQ",
                     "HOUSESIZE", "HOUSESIZE_SQ", "LOTSIZE", "LOTSIZE_SQ",
                     "NEWHOME", "POSTFIRM", "DFLOOD100", "DFLOOD500",
                     "FIRSTROW", "LN(DISTAMEN)", "LN(DISTCBD)",
                     "LN(DISTHWY)", "LN(DISTPARK)"]].copy()

    # Compute all prices (matrix multiplication on dataframe)
    parcels["PRICE_REGR"] = np.round(np.exp(chars @ regression_coefs), -2)
    return parcels

def initial_kriging(parcels):
    """Compute hedonic price estimate from regression kriging

    Args:
        parcels         : Properties to estimate price for
    """

    # Preprocess parcel data
    parcels.loc[parcels["FLOOD_PROB"] > 0, "DFLOOD100"] = 1
    parcels.loc[parcels["FLOOD_PROB"] == 0, "DFLOOD100"] = 0
    parcels["LN(HOUSESIZE)"] = np.log(parcels["HOUSESIZE"])
    parcels["LN(LOTSIZE)"] = np.log(parcels["LOTSIZE"])

    # Select parcels with sale price and sale year
    parcels_sales = parcels[(parcels["SALE_PRI"] > 0) &
                            (~parcels["SALE_PRI"].isna()) &
                            (~parcels["SALE_YR"].isna())].copy()
    parcels_rest = parcels[~parcels.index.isin(parcels_sales.index)].copy()

    # Correct sale prices for year (GDP change)
    gdp_df = pd.read_csv("Data/GDP.csv").set_index("Country Name")
    gdp_df = gdp_df.drop(columns=["Country Code",
                                  "Indicator Name",
                                  "Indicator Code"])
    gdp_df = gdp_df.loc["United States"]
    gdp_df = gdp_df/gdp_df["2004"]
    price_correction = gdp_df[parcels_sales["SALE_YR"]].values
    # Get corrected known sale prices
    Y = parcels_sales["SALE_PRI"]/price_correction
    
    # Get parcel characteristics
    chars = ["AGE", "LN(HOUSESIZE)", "LN(LOTSIZE)", "BEDROOMS", "DFLOOD100"]
    X_sales = parcels_sales[chars].copy()
    coords_sales = parcels_sales[["COORDS_X", "COORDS_Y"]].values

    # Chech linear regression summary
    results = sm.OLS(np.log(Y), sm.add_constant(X_sales)).fit()
    print(results.summary())
    # # Fit linear regression
    m_regression = LinearRegression(fit_intercept=True)
    m_regression.fit(X_sales, np.log(Y))
    resid_sales = np.log(Y) - m_regression.predict(X_sales)
    parcels.loc[parcels_sales.index, "RESID"] = resid_sales

    # Fit kriging model
    m_kriging = RegressionKriging(regression_model=m_regression,
                                  n_closest_points=10,
                                  variogram_model="spherical")
    m_kriging.fit(X_sales, coords_sales, np.log(Y))

    # Predict unknown prices from: fit linear regression and krige residuals
    X_rest = parcels_rest[chars].copy()
    coords_rest = parcels_rest[["COORDS_X", "COORDS_Y"]].values
    if len(X_rest) > 0:
        # Save kriged residuals
        resid_rest = m_kriging.krige_residual(coords_rest)
        parcels.loc[parcels_rest.index, "RESID"] = resid_rest
        # Assign prices to original parcel dataframe
        prices = np.round(np.exp(m_kriging.predict(X_rest, coords_rest)), -2)
        parcels_rest["PRICE_KRIGING"] = prices
        parcels.loc[parcels_rest.index, "PRICE_KRIGING"] = parcels_rest["PRICE_KRIGING"]
    parcels.loc[parcels_sales.index, "PRICE_KRIGING"] = parcels_sales["SALE_PRI"]

    # Normalize parcel characteristics
    parcels = norm_chars(parcels, ["AGE", "HOUSESIZE", "LN(LOTSIZE)",
                                   "BEDROOMS", "RESID"])
    parcels = parcels.rename(columns={"AGEnorm": "AGEnorm",
                                      "HOUSESIZEnorm": "HOUSESIZEnorm",
                                      "LN(LOTSIZE)norm": "LOTSIZEnorm"})
    return parcels


# ------------------------------------- #
# ----------- PREPROCESSING ----------- #
# ------------------------------------- #

# # ------------ BEAUFORT DATASET ------- #
# # Extract coastal front info, save as temporary file
# df = gpd.read_file("Data/Beaufort_final_8.shp")
# water_shape = df[df["ID1"] == 6575]["geometry"].values[0]
# df["FIRSTROW_1"] = df["geometry"].intersects(water_shape) & (df["ID1"] != 6575)
# df.loc[df["ID1"] == 6575, "FIRSTROW_1"] = False
# df.to_file("Data/Beaufort_temp.shp")

# If already there, read new shapefile (otherwise very slow)
df = gpd.read_file("Data/Beaufort_temp.shp")
df = df.rename(columns={"TOT_SQ_F_1": "HOUSESIZE", "TOTAL_AC_1": "LOTSIZE",
                        "BATHROOM_1": "BATHROOMS", "DISTCBD1_1": "DISTCBD",
                        "DFLOOD_X_1": "DFLOOD500", "DFLOOD_A_1": "DFLOOD100"})
df.columns = df.columns.str.replace("_1", "")

# Select only parcels in Beaufort
df = df[df["DTOWN2"] == 1]
# Remove parcels with unrealistic data values (assuming these are not residential)
df = df[((df["AGE"] != 0) | (df["BEDROOMS"] != 0)) &
          ((df["AGE"] != 0) | (df["HOUSESIZE"] != 0)) &
          ((df["BEDROOMS"] != 0) |
           (df["BATHROOMS"] != 0) |
           (df["HOUSESIZE"] != 0)) &
          (df["BATHROOMS"] <= 6) &
          (df["HOUSESIZE"] <= 6108) &
          (df["LOTSIZE"] <= 46.3)]

# Only select parcels with price < 1,500,000 as residential parcels
df = df[df["SALE_PRI"] < 1.5e6]

# Replace inconsistent or missing values
df.loc[df["AGE"] == 2004, "AGE"] = 1    # Age = 2004 --> age = 1
for column in ["AGE", "HOUSESIZE", "LOTSIZE", "BATHROOMS"]:
    df = replace_zero_by_avg(df, column)

# Create categorical flood probability variable from dummies
df.loc[df["DFLOOD100"] == 1, "FLOOD_PROB"] = 0.01
df.loc[df["DFLOOD500"] == 1, "FLOOD_PROB"] = 0.002
df["FLOOD_PROB"] = df["FLOOD_PROB"].fillna(0)

# Convert distance to water and beach to combined distance to coastal amenities
df["DISTAMEN"] = df[["DISTWTR", "DISTBEAC"]].min(axis=1)
prox_beach = df["DISTBEAC"].max() + 1 - df["DISTBEAC"]
prox_water = df["DISTWTR"].max() + 1 - df["DISTWTR"]
df.loc[df["DISTBEAC"] < df["DISTWTR"], "PROXAMEN"] = prox_beach
df.loc[df["DISTWTR"] < df["DISTBEAC"], "PROXAMEN"] = prox_water


# ---------- PRICE ESTIMATION ------------ #
# Set parcel IDs as index
df = df.set_index(["ID"])
# Extract parcel coordinates to be used in kriging method for price estimation
df["COORDS_X"] = df["geometry"].centroid.x
df["COORDS_Y"] = df["geometry"].centroid.y

# Return dataframe with relevant characteristics and estimated prices
df_prices = initial_regression(df)
df_prices = initial_kriging(df)

# After filling missing values: again remove house prices above 1.5 million dollars
df_prices = df_prices[df_prices["PRICE_REGR"] < 1.5e6]
df_prices = df_prices[df_prices["PRICE_KRIGING"] < 1.5e6]

# Save all parcel characteristics for all price methods and utility functions
attrs = (["COORDS_X", "COORDS_Y", "AGE", "BATHROOMS", "BEDROOMS",
          "HOUSESIZE", "LOTSIZE", "NEWHOME", "POSTFIRM", "FIRSTROW",
          "DISTAMEN", "DISTCBD", "DISTHWY", "DISTPARK", "PROXAMEN",
          "AGEnorm", "HOUSESIZEnorm", "LOTSIZEnorm", "BEDROOMSnorm", "RESIDnorm",
          "DFLOOD100", "DFLOOD500", "PRICE_REGR", "PRICE_KRIGING"])

df_prices.to_csv("Data/Parcel_chars_Beaufort" + ".csv", columns=attrs)
# --------------------------------------------------------------------- #


# ------------------------- GREENVILLE DATASET ------------------------ #
df = gpd.read_file("Data/Parcels_Greenville.shp")

df = df.rename(columns={"TOTALSQFT": "HOUSESIZE","ACRES": "LOTSIZE",
                        "age": "AGE", "bedrm": "BEDROOMS", "bathrm": "BATHROOMS",
                        ""
                        "dflood": "FLOOD_PROB"})
df = df.assign(SALE_YR="2004")
df["SALE_PRI"] = np.exp(df["LNPRICE"])

# Set parcel IDs as index
df = df.set_index(["OBJECTID"])
df.index = df.index.rename("ID")

# Extract parcel coordinates to be used in kriging method for price estimation
df["COORDS_X"] = df["geometry"].centroid.x
df["COORDS_Y"] = df["geometry"].centroid.y

# df_prices = initial_regression(df)
df_prices = initial_kriging(df)

# Save relevant attributes and prices for all residential parcels
attrs = ["COORDS_X", "COORDS_Y", "AGE", "HOUSESIZE", "LOTSIZE",
         "BEDROOMS", "DFLOOD100", "RESID", "AGEnorm", "HOUSESIZEnorm",
         "LOTSIZEnorm", "BEDROOMSnorm", "RESIDnorm", "PRICE_KRIGING"]

df_prices.to_csv("Data/Parcel_chars_Greenville.csv", columns=attrs)
