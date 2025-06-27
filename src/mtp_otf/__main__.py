import os
import argparse
from .otf_mtp import main


if __name__ == "__main__":
    print("Starting main.py")

    parser = argparse.ArgumentParser(prog=None, description="Utility to select structures for training se based on D-optimality criterion")

    parser.add_argument("extrapolative_dumps", nargs='+', help=" extrapolative_structures.dump", type=str)
    parser.add_argument("-p", "--potential", help="input potential file name, will override input file 'potential' section", type=str, default="potential.almtp")
    parser.add_argument("-t", "--training_set", help="Training dataset file name, ex.: train.cfg", type=str, default="train.cfg")

    parser.add_argument("-P", "--no_preselection_filtering", help="Preselection filtering", dest='preselection_filtering', action='store_false')

    parser.add_argument("-g", "--gamma_tolerance", help="Gamma tolerance", default=1.010, type=float)
    parser.add_argument("-G", "--gamma_max", help="Gamma max", default=0, type=float)
    parser.add_argument("-D", "--gamma_max0", help="Gamma max_0", default=100000, type=float)
    parser.add_argument("-X", "--max_extrapolation_lock", help="Max extrapolation lock", default=5, type=int)

    parser.add_argument("-m", "--max_structures", help="Max structures selection", default=-1, type=int)
    parser.add_argument("-l", "--iteration_limit", help="Number of maximum iteration in training algorithm", default=300, type=int)

    args_parse = parser.parse_args()


    # Remove all OMPI_ environment variables to avoid issues with MPI
    save_env = os.environ.copy()
    del_env = {k: os.environ.pop(k) for k, v in os.environ.items() if k.startswith("OMPI_")}
    print("Removed OMPI_ environment variables: ", del_env)

    returncode = main(args_parse, os.environ)

    os.environ = save_env
    exit(returncode)
