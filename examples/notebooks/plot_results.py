import argparse
import xarray
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import StringIO
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timedelta


def open_ensemble(f, domain, chunks={"time": 1}):
    time = xarray.open_dataset(f).time
    root = xarray.open_dataset(f, decode_times=False)

    config = json.loads(root.attrs['config'])

    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs = root.attrs
    return ds.assign_coords(time=time)


def plot_time_series(ds):
    # print(ds.t2m.shape)
    # print(ds.sel(npoints=0).t2m.shape)
    # exit()

    lead_time = np.array((pd.to_datetime(ds.time) - pd.to_datetime(ds.time)[0]).total_seconds()/ 3600)
    # nyc_lat = 40
    # nyc_lon = 360-74
    # NYC = ds.sel(lon = nyc_lon, lat = nyc_lat)
    NYC = ds.sel(npoints=0)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(311)
    ax.set_title('Ensemble members')
    ax.plot(lead_time, NYC.t2m.T)
    ax.set_ylabel('t2m [K]')

    ax = fig.add_subplot(312)
    ax.set_title('deviation from ensemble mean')
    ax.plot(lead_time, NYC.t2m.T-NYC.t2m.mean("ensemble"))
    ax.set_ylabel('t2m [K]')

    ax = fig.add_subplot(313)
    ax.set_title('ensemble spread')
    ax.plot(lead_time, NYC.t2m.std("ensemble"))
    ax.set_xlabel('lead_time [h]')
    ax.set_ylabel('std t2m [K]')
    plt.tight_layout()
    plt.savefig('./plot.png', format="png")

    return


def plot_results(dat_file):
    root = xarray.open_dataset(dat_file, decode_times=False)
    config = json.loads(root.attrs['config'])
    domain = config['weather_event']["domains"][1]['name']

    ds = open_ensemble(dat_file, domain)


    countries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='50m',
        facecolor='none',
        edgecolor='black'
    )

    # plot time series of points
    plot_time_series(ds)

    # plot map of window

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str)
    args = parser.parse_args()

    args.file = os.path.abspath(args.file)
    assert os.path.isfile(args.file), f'file {args.file} does not exist'

    plot_results(args.file)