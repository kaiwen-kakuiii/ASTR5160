# __main__.py
# by Kaiwen Zhang

import os
import numpy as np
from utils import calculate_min_airmass, generate_obs_time
from cli import read_user_cli_args


def main():
    # Get the directory of the current script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths relative to the script
    FILE_PATH_RA_DEC = os.path.join(SCRIPT_DIR, "coord_all_quasar.txt")
    FILE_PATH_QUASAR = os.path.join(SCRIPT_DIR, "HW1quasarfile.txt")

    # Get obs_time object
    user_args = read_user_cli_args()

    # Load the quasar coord info
    all_quasar_coord = np.loadtxt(FILE_PATH_QUASAR, dtype=str)
    all_quasar = np.loadtxt(FILE_PATH_RA_DEC, dtype=str)
    ra, dec = all_quasar[:, 0], all_quasar[:, 1]

    # Generate the obs_time for given month
    obs_time = generate_obs_time(hr=23, month=user_args.month, year=2025)

    # Calculate the target with min airmass for obs_time
    idx_list, airmass_list = calculate_min_airmass(obs_time, ra, dec)

    # Write output
    print(list(zip(obs_time.isot, all_quasar_coord[idx_list], ra[idx_list], dec[idx_list], airmass_list)))


if __name__ == "__main__":
    main()
