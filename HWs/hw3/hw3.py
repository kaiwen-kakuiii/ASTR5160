import os
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from sdssDR9query import sdssQuery
from time import sleep

SWEEP_FILE_PATH = '/d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0/'
FIRST_FILE_PATH = '/d/scratch/ASTR5160/data/first/first_08jul16.fits'


def get_sweep_files(ra, dec):
    # KK search sweep files needed for given ra & dec
    sweep_files = []
    ra_max, ra_min = np.max(ra), np.min(ra)
    dec_max, dec_min = np.max(dec), np.min(dec)

    ra = np.arange(int(ra_min // 10 * 10), int(ra_max // 10 * 10 + 10), 10)
    dec = np.arange(int(dec_min // 5 * 5), int(dec_max // 5 * 5 + 5), 5)

    # KK even use two for loops, but since ra & dec only contains a few elements,
    # this won;t take too much time to executed
    for i in ra:
        for j in dec:
            ra_1, ra_2 = str(i).rjust(3, '0'), str(i + 10).rjust(3, '0')
            dec_1, dec_2 = str(j).rjust(3, '0'), str(j + 5).rjust(3, '0')
            sweep_files.append(f'sweep-{ra_1}p{dec_1}-{ra_2}p{dec_2}.fits')
    return sweep_files


def query_sdss_server(ra, dec):
    # ADM initialize the query.
    qry = sdssQuery()

    # ADM the query to be executed. You can substitute any query, here!
    query = """SELECT top 1 ra,dec,u,g,r,i,z,GNOE.distance*60 FROM PhotoObj as PT
    JOIN dbo.fGetNearbyObjEq(""" + str(ra) + """,""" + str(dec) + """,0.02) as GNOE
    on PT.objID = GNOE.objID ORDER BY GNOE.distance"""

    # ADM execute the query.
    qry.query = query
    for line in qry.executeQuery():
        result = line.strip()

    # ADM NEVER remove this line! It won't speed up your code, it will
    # ADM merely overwhelm the SDSS server (a denial-of-service attack)!
    sleep(1)

    # ADM the server returns a byte-type string. Convert it to a string.
    return result.decode().split(',')


def main():
    # KK read FIRST survey data
    first_data = Table.read(FIRST_FILE_PATH)

    # KK construct Skycoord for match the target coord
    first_coord = SkyCoord(np.array(first_data['RA']) * u.deg,
                           np.array(first_data['DEC']) * u.deg)
    center_coord = SkyCoord([163] * u.deg, [50] * u.deg)

    # KK select sources within 3 degree from target
    index_target, _, _, _ = center_coord.search_around_sky(first_coord, 3 * u.deg)
    first_data_selected = first_data[index_target]
    first_coord_selected = first_coord[index_target]

    # KK get the name of sweep files needed for matching the given target
    sweep_files = get_sweep_files(np.array(first_data_selected['RA']),
                                  np.array(first_data_selected['DEC']))

    # KK load the SDSS legacy survey
    hdul = fits.open(os.path.join(SWEEP_FILE_PATH, sweep_files[0]))
    data = hdul[1].data
    hdul.close()

    all_sweep_data = np.vstack((data['RA'], data['DEC'], data['FLUX_R'],
                                data['FLUX_W1'], data['FLUX_W2']))
    for i in sweep_files[1:]:
        hdul = fits.open(os.path.join(SWEEP_FILE_PATH, i))
        data = hdul[1].data
        hdul.close()

        tem = np.vstack((data['RA'], data['DEC'], data['FLUX_R'],
                         data['FLUX_W1'], data['FLUX_W2']))
        all_sweep_data = np.hstack((all_sweep_data, tem))

    # KK cross-match the SDSS data with selected FIRST sources
    sweep_coord = SkyCoord(all_sweep_data[0, :] * u.deg, all_sweep_data[1, :] * u.deg)
    index_sweep2first, index_first, _, _ = first_coord_selected.search_around_sky(
        sweep_coord, 1 * u.arcsec)
    sweep_matched = all_sweep_data[:, index_sweep2first]
    first_matched = first_data_selected[index_first]

    # KK filter out the negative value and calculate the corresponding magnitude
    detected = (sweep_matched[2] > 0) & (sweep_matched[3] > 0) & (sweep_matched[4] > 0)
    R_mag = (lambda x: 22.5 - 2.5 * np.log10(x))(sweep_matched[2][detected])
    W1_mag = (lambda x: 22.5 - 2.5 * np.log10(x))(sweep_matched[3][detected])
    W2_mag = (lambda x: 22.5 - 2.5 * np.log10(x))(sweep_matched[4][detected])

    # KK make color cut and select coords for remaining targets
    masked = (R_mag < 22) & (W1_mag - W2_mag > 0.5)
    ra, dec = sweep_matched[0, detected][masked], sweep_matched[1, detected][masked]

    # KK print out how many objects found after the color and mag cut
    print(f'Found {len(ra)} objects after the color and mag cut.')

    # KK query the targets in SDSS server and ignore the No detection targets
    # also record the index of original array
    sdss_data = []
    for i in range(len(ra)):
        tmp = query_sdss_server(ra[i], dec[i])
        if 'No' in tmp[0]:
            continue
        tmp.append(i)
        sdss_data.append(tmp)

    # KK print out how many objects found in SDSS server
    print(f'Found {len(sdss_data)} in SDSS server based on ra & dec given.')

    # KK find the brightest source in SDSS sample
    idx_sdss_u_max = np.argmin(np.array(sdss_data)[:, 2].astype(float))

    # KK get sdss fluxes
    ubrite1 = np.array(sdss_data)[idx_sdss_u_max, :].astype(float)
    ubrite1_flux = list(10 ** ((ubrite1[2:7] - 22.5) / -2.5))

    # KK match the index at sweep catalog and get fluxes
    idx_sweep = int(np.array(sdss_data)[:, 8][idx_sdss_u_max])
    ubrite1_flux = ubrite1_flux + list(sweep_matched.T[detected][masked][idx_sweep][2:5])

    # KK match the index at first catalog and get the fluxes
    sweep_idx = index_sweep2first[detected][masked][idx_sweep]
    first_idx = np.argwhere(index_sweep2first == sweep_idx)[0][0]
    ubrite1_flux.append(first_matched['FINT'][first_idx] * 1e3 / 3.631)

    # KK filters are in order u, g, r, i, z, R, W1, W2, first
    print('Done selecting brightest target in SDSS u band')
    print('Begin to plot the SED of ubrite1')
    lamda = [355, 468, 616, 748, 893, 641, 3400, 4600, 2.14e8]

    plt.scatter(lamda, ubrite1_flux, marker='s', s=10)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('flux (nano maggies)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    plt.close()

    print(
        'The resshift of ubrite1 is 1.035 based on the prominent MgII broad emission line and other lines in the '
        'spectrum. This target seems like a quasar, even though it is labeled as star in SDSS website. I do not '
        'expect star having that broad MgII emission line.')
    return


if __name__ == "__main__":
    main()
