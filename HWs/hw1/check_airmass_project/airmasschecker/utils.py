# utils.py
# by Kaiwen Zhang

import pytz
import calendar
from datetime import datetime
import numpy as np
import astropy.units as u
import astropy.time as Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord


def generate_obs_time(hr, month, year):
    """
    Generate list of time objects for a given month and given year.
    :param hr: hr specified by user, typically 23
    :param month: month specified by user
    :param year: year specified by user, typically 2025
    :return: array of time objects for that month
    """

    mt_time_naive = datetime(year, month, 1, hr, 0)
    mt_tz = pytz.timezone("America/Denver")
    mt_time = mt_tz.localize(mt_time_naive)
    utc_time = mt_time.astimezone(pytz.UTC)
    obs_time = Time.Time(utc_time, scale='utc')

    days = calendar.monthrange(year, month)[1]
    obs_time_array = obs_time + np.arange(0, days, 1) * u.day
    return obs_time_array


def calculate_min_airmass(obs_time_array, ra, dec):
    """
    Calculate the best airmass based on the input time.
    :param obs_time_array: time object span over 1 month for calculating the airmass
    :param ra: RA for the quasars
    :param dec: DEC for the quasars
    :return: quasar index and the corresponding airmass for give month
    """

    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    expanded_coords = coords[:, np.newaxis]
    expanded_times = obs_time_array[np.newaxis, :]

    kpno = EarthLocation.of_site('kpno')
    az_frame = AltAz(location=kpno, obstime=expanded_times)

    airmass = np.array(expanded_coords.transform_to(az_frame).secz)
    airmass_filtered = np.where(airmass < 0, np.nan, airmass)
    return np.nanargmin(airmass_filtered, axis=0), np.round(np.nanmin(airmass_filtered, axis=0), 4)
