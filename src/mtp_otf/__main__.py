import os
import argparse
from .otf_mtp import main


if __name__ == "__main__":
    print("Starting main.py")

    parser = argparse.ArgumentParser(prog=None, description="Utility to select structures for training se based on D-optimality criterion")

    parser.add_argument("extrapolative_dumps", nargs='+', help=" extrapolative_structures.dump", type=str)
    parser.add_argument("-p", "--potential", help="input potential YAML file name, will override input file 'potential' section", type=str, default="output_potential.yaml")
    parser.add_argument("-t", "--training_set", help="Training dataset file name(s), ex.: train.cfg", type=str, default="train.cfg")

    parser.add_argument("-g", "--gamma_tolerance", help="Gamma tolerance", default=1.010, type=float)
    parser.add_argument("-G", "--gamma_max", help="Gamma tolerance", default=0, type=float)

    parser.add_argument("-m", "--max_structures", help="Max structures selection", default=-1, type=int)
    parser.add_argument("-l", "--iteration_limit", help="Number of maximum iteration in training algorithm", default=300, type=int)

    args_parse = parser.parse_args()

    _env = {k: v for k, v in os.environ.items() if not k.startswith("OMPI_")}

    returncode = main(args_parse, _env)
    exit(returncode)
