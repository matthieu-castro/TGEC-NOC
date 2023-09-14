#!/usr/bin/python3
# coding: utf-8

import argparse

from nocpkg.NOCMain import NOCMain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='?', default='', help='Name of the model')
    parser.add_argument('-V', '--version',  action='store_true', default=False,
                        help="Prints the version of NOC.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-g', '--debug', action='store_true', default=False,
                        help="TGEC is run in debug mode.")

    result = NOCMain(parser)
