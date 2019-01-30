import os
import yaml
from argparse import ArgumentParser


def setup():
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    yaml_file = os.path.join(__location__, 'input.yml')
    content = open(yaml_file)
    config_dict = yaml.load(content)

    parser = ArgumentParser(description='Create SSHT plot')
    parser.add_argument('type', metavar='type', type=str, nargs='?', default='real', const='real', choices=[
                        'abs', 'real', 'imag', 'sum'], help='plotting method i.e. real')
    parser.add_argument('method', metavar='method', type=str, nargs='?', default='north', const='north', choices=[
                        'north', 'rotate', 'translate'], help='plotting method i.e. north')
    parser.add_argument('--alpha', '-a', metavar='alpha', type=float,
                        default=0.0, help='alpha/phi pi fraction')
    parser.add_argument('--beta', '-b', metavar='beta', type=float,
                        default=0.0, help='beta/theta pi fraction')

    return config_dict, parser
