import os
import subprocess

import numpy

import ase
import ase.io.lammpsrun

from .io_cfg import read_cfg, write_cfg
from evaluator import evaluator

mlp = os.environ["OTF_MTP_COMMAND"]
if mlp == "":
    print("Error OTF_MTP_COMMAND variable not set, set with export OTF_MTP_COMMAND=\"/path/to/mlp\" (in bash) before this script")

def preselected_dump2cfg(extrapolative_dumps, extrapolative_candidates_cfg, extrapolation_field="f_extrapolation_grade"):
    dumps = []
    for i, extrapolative_dump in enumerate(extrapolative_dumps):
        print(f"Reading extrapolative dump : ", extrapolative_dump)
        with open(extrapolative_dump) as dump_file:
            dumps += ase.io.lammpsrun.read_lammps_dump_text(dump_file, index=slice(None))
        with open(extrapolative_dump, mode="w") as dump_file:
            continue

    for dump in dumps:
        if dump.has(extrapolation_field): dump.set_array("nbh_grades", dump.get_array(extrapolation_field).flatten())

    with open(extrapolative_candidates_cfg, mode="w") as preselected_file:
        write_cfg(preselected_file, dumps)

def preselected_filter(preselected_cfg, gamma_tolerance, gamma_max, gamma_max0, max_extrapolation_lock, max_structures=-1):
    with open(preselected_cfg, mode="r") as preselected_file:
        cfgs = read_cfg(preselected_file)

    print("Preselected structures count: ", len(cfgs))

    def checkgrade(cfg):
        if "nbh_grades" in cfg.arrays:
            return cfg.arrays["nbh_grades"].max()

        if "features" in cfg.info and "MV_grade" in cfg.info["features"]:
            if "MV_grade" in cfg.info["features"]:
                if cfg.info["features"]["MV_grade"] < gamma_max0:
                    return cfg.info["features"]["MV_grade"]
                else:
                    return 0
        else:
            return 0

    filtred_cfgs = []
    gammas = numpy.array([checkgrade(cfg) for cfg in cfgs])
    cfgs = [cfgs[i] for i in numpy.where(gammas > gamma_tolerance)[0]]

    if numpy.any(gammas < gamma_max):
        filtred_cfgs = [cfgs[i] for i in numpy.where(gammas < gamma_max)[0]]
    elif numpy.all(gammas > gamma_max) and numpy.any(gammas < gamma_max0):
        filtred_cfgs = [cfgs[numpy.argmin(gammas)]]
        print("Selected structure with gamma = ", gammas[numpy.argmin(gammas)])
    else:
        print("No structures with gamma < {} found".format(gamma_max0))
        if len(cfgs) > max_extrapolation_lock:
            print("Warning : No structures with gamma < {} found, max_extrapolation_lock is smaller than {} structures. Selecting conf with gamma minimal gamma = {}".format(gamma_max0, len(cfgs), numpy.min(gammas)))
            filtred_cfgs = [cfgs[numpy.argmin(gammas)]]
        else:
            extrapolation_lock = 0
            if os.path.isfile("extrapolation.lock"):
                with open("extrapolation.lock", mode="r") as lock_file:
                    line = lock_file.read().strip()
                if line != '': extrapolation_lock = int(line)
                if extrapolation_lock < max_extrapolation_lock:
                    print("Extreme Warning : No structures with gamma < {} found, but extrapolation_lock = {} < max_extrapolation_lock = {}. Selecting conf with gamma minimal gamma = {}".format(gamma_max0, extrapolation_lock, max_extrapolation_lock, numpy.min(gammas)))
                    filtred_cfgs = [cfgs[numpy.argmin(gammas)]]
                else:
                    print("Extreme Warning : No structures with gamma < {} found and extrapolation_lock = {} >= max_extrapolation_lock = {}. Continuing without selection".format(gamma_max0, extrapolation_lock, max_extrapolation_lock))
            with open("extrapolation.lock", mode="w") as lock_file:
                print("extrapolation_lock incremented to ", extrapolation_lock + 1)
                lock_file.write(str(extrapolation_lock + 1))
            if extrapolation_lock > 99:
                print("Something is wrong, extrapolation_lock is too high, please check. Breaking exit.")
                exit(89)

    if max_structures > 0 and len(filtred_cfgs) > max_structures:
        rnd_selected = numpy.random.choice(len(filtered_cfgs), size=max_structures, replace=False)
        filtered_cfgs = [filtered_cfgs[i] for i in rnd_selected]

    with open(preselected_cfg, mode="w") as preselected_file:
        write_cfg(preselected_file, filtred_cfgs)

    print("Filtered structures count: ", len(filtred_cfgs))

def eval_structures(selected_extrapolative, training_set):
    with open(selected_extrapolative, mode="r") as selected_file:
        selected_structures = read_cfg(selected_file)

    for i, selected_structure in enumerate(selected_structures):
        print(f"Calculating structure {i+1}/{len(selected_structures)}")

        selected_structure = evaluator(selected_structure)

        with open(training_set, mode="r") as training_file:
            training_structure = read_cfg(training_file)
        training_structure += [selected_structure]
        with open(training_set, mode="w") as training_file:
            write_cfg(training_file, training_structure)

    return 0

def main(args_parse, _env):
    potential = args_parse.potential
    training_set = args_parse.training_set

    extrapolative_dumps = args_parse.extrapolative_dumps
    extrapolative_candidates = "preselected.cfg"
    selected_extrapolative = "selected.cfg"
    extrapolation_field = "f_extrapolation_grade"

    preselection_filtering = args_parse.preselection_filtering
    gamma_tolerance = args_parse.gamma_tolerance
    gamma_max = args_parse.gamma_max
    gamma_max0 = args_parse.gamma_max0
    max_extrapolation_lock = args_parse.max_extrapolation_lock
    max_structures = args_parse.max_structures
    iteration_limit = args_parse.iteration_limit

    preselected_dump2cfg(extrapolative_dumps, extrapolative_candidates, extrapolation_field)

    if preselection_filtering:
        # failsafe because sometimes lammps extrapolation fix-halt stops lammps before grade calculation
        args = ["mpirun", "-n", "1", mlp, "calculate_grade", potential, extrapolative_candidates, extrapolative_candidates[:-4] + ".calculate_grade"]
        print("running calculate_grade with args: ", args)
        result = subprocess.run([*args], text=True)
        if result.returncode == 0:
            os.replace(extrapolative_candidates[:-4] + ".calculate_grade.0", extrapolative_candidates)
            print("Successfully executed calculate_grade.")
        else:
            print("Failed to execute calculate_grade.")
            exit(result.returncode)

        preselected_filter(extrapolative_candidates, gamma_tolerance, gamma_max, gamma_max0=gamma_max0, max_extrapolation_lock=max_extrapolation_lock, max_structures=max_structures)

    args = ["mpirun", "-n", "1", mlp, "select_add", potential, training_set, extrapolative_candidates, selected_extrapolative]
    print("running select_add with args: ", args)
    result = subprocess.run([*args], text=True, check=True, env=_env)
    if result.returncode == 0:
        print("Successfully executed select_add.")
    else:
        print("Failed to execute select_add.")
        exit(result.returncode)


    returncode = eval_structures(selected_extrapolative, training_set)
    if returncode == 0:
        print("Successfully executed eval_structures.")
    else:
        print("Failed to execute eval_structures.")
        exit(returncode)

    # "taskset", "-c", "0-7",
    # "numactl", "--cpunodebind=0",
    args = ["mpirun", mlp, "train", potential, training_set, "--save_to=tmp_{}".format(potential), "--iteration_limit=" + str(iteration_limit), "--al_mode=nbh"]
    print("running training with args: ", args)
    result = subprocess.run([*args], text=True, check=True, env=_env)
    if result.returncode == 0:
        # replace potential by tmp potential
        os.replace("tmp_{}".format(potential), potential)
        print("Successfully executed trained.")
    else:
        print("Failed to execute train.")
        exit(result.returncode)


    # Active set generation (train update the selection set, so not needed)
    # args = [potential, training_set]
    # result = subprocess.run([mlp, "select", *args], text=True)
    # if result.returncode == 0:
    #     print("Successfully executed selection.")
    # else:
    #     print("Failed to execute selection.")
    #     exit(result.returncode)

    # return result.returncode
