import os
import argparse
import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt


def cal_area_sq_deg(ra_min, ra_max, dec_min, dec_max):
    """
    Transform the (ra, dec, theta) into (x, y, z, 1-h) for spherical cap vector
    :param ra_min， ra_max: right accession range (degree)
    :param dec_min, dec_max: declination range (degree)
    :return: area bounded by ra/dec in square degrees
    """

    area = (np.sin(dec_max) - np.sin(dec_min)) * (ra_max - ra_min).to('radian').value * (180 / np.pi) ** 2
    return area.value


def uniform_scatter_on_sphere(number, ra_min, ra_max, dec_min, dec_max):
    """
    Generate uniform random points that populate the entire sphere and select points inside lat-lon
    :param number: number of points will be randomly sampled
    :param ra_min， ra_max: right accession range (degree)
    :param dec_min, dec_max: declination range (degree)
    :return: array of ra/dec inside lat-lon (radian)
    """

    ra = 2 * np.pi * (np.random.random(number) - 0.5)
    dec = np.arcsin(1. - np.random.random(number) * 2)

    # mask out the points inside lat-lon
    mask = (ra_min < ra) & (ra < ra_max) & (dec < dec_max) & (dec > dec_min)

    return ra[mask], dec[mask]


def plot_lat_lon(output_dir):
    """
    Generate plot for lat-lon
    :param output_dir: directory to save the plot
    """
    ra_min, ra_max = (0 * u.deg).to('radian'), (30 * u.deg).to('radian')
    dec_min, dec_max = (np.arange(0, 60, 15) * u.deg).to('radian'), (np.arange(15, 75, 15) * u.deg).to('radian')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='aitoff')
    ax.grid(color='k', linestyle='--', linewidth=0.5)
    xlab = [f'{str(i)}h' for i in np.arange(14, 36, 2) % 24]
    ax.set_xticklabels(xlab, weight=800)
    color = ['r-', 'b-', 'g-', 'y-']

    for i in range(len(dec_min)):
        # calculate the area for corresponding lat-lon
        area = cal_area_sq_deg(ra_min, ra_max, dec_min[i], dec_max[i])

        # plot boundary for corresponding lat-lon with different color
        ra_rad, dec_rad = np.linspace(ra_min, ra_max, 100), np.linspace(dec_min[i], dec_max[i], 100)
        ax.plot(ra_min * np.ones_like(dec_rad), dec_rad, color[i], label=f'{area:.2f} sq degree')
        ax.plot(ra_max * np.ones_like(dec_rad), dec_rad, color[i])
        ax.plot(ra_rad, (dec_min[i]) * np.ones_like(ra_rad), color[i])
        ax.plot(ra_rad, (dec_max[i]) * np.ones_like(ra_rad), color[i])
    plt.legend()
    output_path = os.path.join(output_dir, "lat_lon_plot.pdf")
    plt.savefig(output_path)
    plt.close()


def main():
    """Command-line interface for generating plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to save plots.")

    args = parser.parse_args()

    plot_lat_lon(args.output_dir)


if __name__ == "__main__":
    main()
