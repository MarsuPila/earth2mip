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
    # config = json.loads(root.attrs['config'])

    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs = root.attrs
    return ds.assign_coords(time=time)


def plot_time_series_ensemble(ds, domain_type, selection, config):
    if domain_type == 'MultiPoint':
        loc = ds.sel(npoints=selection)
        _lat, _lon = float(loc.lat_point), float(loc.lon_point)
    elif domain_type == 'Window':
        loc = ds.sel(lat=selection[0], lon=selection[1])
        _lat, _lon = selection
    else:
        raise ValueError(f'plotting time series not implemented for domain type {domain_type}')

    lead_time = np.array((pd.to_datetime(ds.time) - pd.to_datetime(ds.time)[0]).total_seconds()/3600)
    fig = plt.figure(figsize=(9, 6))
    fig.suptitle(f'Model: {config["weather_model"]}\n lat={_lat}, lon={_lon}\ninitial condidtion: {config["weather_event"]["properties"]["start_time"]}')

    ax = fig.add_subplot(311)
    ax.set_title('Ensemble members')
    ax.plot(lead_time, loc.t2m.T)
    ax.set_ylabel('t2m [K]')

    ax = fig.add_subplot(312)
    ax.set_title('deviation from ensemble mean')
    ax.plot(lead_time, loc.t2m.T-loc.t2m.mean("ensemble"))
    ax.set_ylabel('t2m [K]')

    ax = fig.add_subplot(313)
    ax.set_title('ensemble spread')
    ax.plot(lead_time, loc.t2m.std("ensemble"))
    ax.set_xlabel('lead_time [h]')
    ax.set_ylabel('std t2m [K]')
    plt.tight_layout()
    plt.savefig('./time_series.png', format="png")

    return


def plot_map(ds, domain_type, config, time_step: int=-1):
    countries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='50m',
        facecolor='none',
        edgecolor='black'
    )

    assert domain_type == 'Window', print('only for Window domain')
    cen_lat = ds.lat.mean().item()
    cen_lon = ds.lon.mean().item()

    fig = plt.figure(figsize=(15, 6.5))
    plt.rcParams['figure.dpi'] = 100
    proj = ccrs.NearsidePerspective(central_longitude=cen_lon, central_latitude=cen_lat)
    lead_time = np.array((pd.to_datetime(ds.time) - pd.to_datetime(ds.time)[0]).total_seconds()/3600)
    fig.suptitle(f'Model: {config["weather_model"]}\n time: {str(ds.time[time_step].values)}\ninitial condidtion: {config["weather_event"]["properties"]["start_time"]}\nlead time: {lead_time[time_step]}h')

    data = ds.u10m[0,time_step,:,:]
    norm = mcolors.CenteredNorm(vcenter=0)
    ax = fig.add_subplot(131, projection=proj)
    ax.set_title('First ensemble member u10m [m/s]')
    img = ax.pcolormesh(ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), norm=norm, cmap="seismic")
    ax.coastlines(linewidth=1)
    ax.add_feature(countries, edgecolor='black', linewidth=0.25)
    plt.colorbar(img, ax=ax, shrink=0.80, norm=mcolors.CenteredNorm(vcenter=0))
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.bottom_labels=False
    gl.right_labels=False

    data = ds.u10m[-1,time_step,:,:]
    ax = fig.add_subplot(132, projection=proj)
    plt.rcParams['figure.dpi'] = 100
    proj = ccrs.NearsidePerspective(central_longitude=cen_lon, central_latitude=cen_lat)
    ax.set_title('Last ensemble member u10m [m/s]')
    img = ax.pcolormesh(ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), norm=norm, cmap="seismic")
    ax.coastlines(linewidth=1)
    ax.add_feature(countries, edgecolor='black', linewidth=0.25)
    plt.colorbar(img, ax=ax, shrink=0.80, norm=mcolors.CenteredNorm(vcenter=0))
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.bottom_labels=False
    gl.right_labels=False

    ds_ensemble_std = ds.std(dim = "ensemble")
    data = ds_ensemble_std.u10m[-1,:,:]
    proj = ccrs.NearsidePerspective(central_longitude=cen_lon, central_latitude=cen_lat)
    ax = fig.add_subplot(133, projection=proj)
    ax.set_title('ensemble std u10m [m/s]')
    img = ax.pcolormesh(ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), cmap="plasma")
    ax.coastlines(linewidth=1)
    ax.add_feature(countries, edgecolor='black', linewidth=0.25)
    plt.colorbar(img, ax=ax, shrink=0.80, norm=mcolors.CenteredNorm(vcenter=0))
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.bottom_labels=False
    gl.right_labels=False

    plt.tight_layout()
    plt.savefig('./map.png', format="png")

    return

def plot_results(dat_file, domain_idx: int=0):
    root = xarray.open_dataset(dat_file, decode_times=False)
    config = json.loads(root.attrs['config'])
    domain = config['weather_event']["domains"][domain_idx]

    ds = open_ensemble(dat_file, domain['name'])

    # plot time series of points
    if domain['type'] == 'MultiPoint':
        selection = 0 # id of point
    elif domain['type'] == 'Window':
        selection = (50.5, 13.5) # lat/lon coords
    plot_time_series_ensemble(ds, domain['type'], selection, config)

    # plot map of window
    if domain['type'] == 'Window':
        plot_map(ds, domain['type'], config)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str)
    args = parser.parse_args()

    args.file = os.path.abspath(args.file)
    assert os.path.isfile(args.file), f'file {args.file} does not exist'

    plot_results(args.file)