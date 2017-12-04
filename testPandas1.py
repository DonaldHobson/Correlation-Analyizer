#import netCDF4
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("tos_O1_2001-2002.nc")
df = ds.to_dataframe()

