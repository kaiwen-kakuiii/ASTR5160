# cli.py
# By Kaiwen Zhang

import argparse


def read_user_cli_args():
    """
    Handle the CLI arguments and options.
    :return: parsed arguments
    """

    parser = argparse.ArgumentParser(
        prog="airmasschecker", description="check the object with best airmass for a given month")

    parser.add_argument("-m", "--month", type=int, required=True,
                        help="Specify the observing month (1-12).")

    return parser.parse_args()
