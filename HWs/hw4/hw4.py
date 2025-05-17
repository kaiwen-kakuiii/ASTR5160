import os
import argparse
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from sklearn import neighbors
from joblib import dump, load

QSO_FILE_PATH = '/d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits'
SWEEP_FILE_PATH = '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/'
SWEEP_FILES = ['sweep-170p025-180p030.fits', 'sweep-170p030-180p035.fits',
               'sweep-180p025-190p030.fits', 'sweep-180p030-190p035.fits']


def training_classifier():
    """
    Train and save a KNN classifier on given QSO dataset
    """

    # KZ Get all the QSO data and cross-match with sweep data
    QSO_data = Table.read(QSO_FILE_PATH)
    sweep_data = []
    for i in SWEEP_FILES:
        sweep_data.append(Table.read(os.path.join(SWEEP_FILE_PATH, i)))
    sweep_data = vstack(sweep_data)

    QSO_coord = SkyCoord(QSO_data['RA'] * u.deg, QSO_data['DEC'] * u.deg)
    sweep_coord = SkyCoord(sweep_data['RA'], sweep_data['DEC'])
    index_sweep, _, _, _ = QSO_coord.search_around_sky(sweep_coord, 0.5 * u.arcsec)
    obj = sweep_data[index_sweep]

    # KZ cut through MASKBITS on g, r, z, W1 bands
    flag_all_g = 2 ** 5
    flag_all_r = 2 ** 6
    flag_all_z = 2 ** 7
    flag_all_w1 = 2 ** 8

    good_exposure = ((obj['MASKBITS'] & flag_all_g == 0) & (obj['MASKBITS'] & flag_all_r == 0) \
                     & (obj['MASKBITS'] & flag_all_z == 0) & (obj['MASKBITS'] & flag_all_w1 == 0))
    obj = obj[good_exposure]

    # KZ convert flux to mag and cut on no detection and r < 19
    mag_r = (lambda x: 22.5 - 2.5 * np.log10(x))(obj['FLUX_R'])
    cut_rmag = mag_r <= 19
    obj = obj[cut_rmag]

    mag_g = 22.5 - 2.5 * np.log10(obj['FLUX_G'])
    mag_r = 22.5 - 2.5 * np.log10(obj['FLUX_R'])
    mag_i = 22.5 - 2.5 * np.log10(obj['FLUX_Z'])
    mag_w1 = 22.5 - 2.5 * np.log10(obj['FLUX_W1'])

    # KZ train KNN model and save the model
    data_mag = np.vstack((mag_g - mag_i, mag_r - mag_w1))
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(data_mag.transpose(), np.zeros(len(data_mag[0])))
    distances, _ = knn.kneighbors(data_mag.transpose())
    cut_qso = np.percentile(distances.mean(axis=1), 95)
    dump(knn, '../hw4/qso_classifier.joblib')
    return


def splendid_function(data_table):
    """
    Take input table data and determine if they are QSOs or not using KNN on color cut schema
    :param data_table: input data table include flux info
    :return: array of True or False indicating QSOs or not
    """

    # KZ cut no detection on input data
    detected = ((data_table['FLUX_G'] > 0) & (data_table['FLUX_R'] > 0)\
               & (data_table['FLUX_Z'] > 0) & (data_table['FLUX_W1'] > 0))
    data_table = data_table[detected]

    # KZ calculate the color for the input table
    mag_g = 22.5 - 2.5 * np.log10(data_table['FLUX_G'])
    mag_r = 22.5 - 2.5 * np.log10(data_table['FLUX_R'])
    mag_i = 22.5 - 2.5 * np.log10(data_table['FLUX_Z'])
    mag_w1 = 22.5 - 2.5 * np.log10(data_table['FLUX_W1'])
    input_mag = np.vstack((mag_g - mag_i, mag_r - mag_w1)).T

    # KZ identify and return the input data for QSOs, cut_qso value is determined from the training function
    knn = load("../hw4/qso_classifier.joblib")
    dist, _ = knn.kneighbors(input_mag)
    cut_qso = 0.2457
    dist, _ = knn.kneighbors(input_mag)
    return dist.mean(axis=1) <= cut_qso


def main():
    """Command-line interface for QSOs identification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to fits file")

    args = parser.parse_args()
    data_table = Table.read(args.path)

    res = splendid_function(data_table)
    print(res)
    return


if __name__ == "__main__":
    main()
