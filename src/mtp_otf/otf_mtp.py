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

def preselected_filter(preselected_cfg, gamma_tolerance, gamma_max, max_structures=-1, gamma_max0=100000):
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
    if numpy.any(gammas < gamma_max):
        for i, cfg in enumerate(cfgs):
            if gammas[i] < gamma_max:
                filtred_cfgs += [cfg]
    elif numpy.all(gammas > gamma_max) and numpy.any(gammas < gamma_max0):
        filtred_cfgs = [cfgs[numpy.argmin(gammas)]]
        print("Selected structure with gamma = ", gammas[numpy.argmin(gammas)])
    else:
        print("No structures with gamma < {} found".format(gamma_max0))
        if len(cfgs) > 1:
            filtred_cfgs = [cfgs[numpy.argmin(gammas)]]

    if max_structures > 0 and len(filtred_cfgs) > max_structures:
        filtered_cfgs = filtered_cfgs[numpy.random.choice(len(filtered_cfgs), size=max_structures, replace=False)]

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

def main(args_parse):
    potential = args_parse.potential
    training_set = args_parse.training_set

    extrapolative_dumps = args_parse.extrapolative_dumps
    extrapolative_candidates = "preselected.cfg"
    selected_extrapolative = "selected.cfg"
    extrapolation_field = "f_extrapolation_grade"

    gamma_tolerance = args_parse.gamma_tolerance
    gamma_max = args_parse.gamma_max
    max_structures = args_parse.max_structures
    iteration_limit = args_parse.iteration_limit

    _env = {k: v for k, v in os.environ.items() if not k.startswith("OMPI_")}

    preselected_dump2cfg(extrapolative_dumps, extrapolative_candidates, extrapolation_field)

    # args = [potential, extrapolative_candidates, extrapolative_candidates + ""]
    # result = subprocess.run(["mpirun", "-n", "1", mlp, "calculate_grade", *args], text=True)
    # if result.returncode == 0:
    #     print("Successfully executed calculate_grade.")
    # else:
    #     print("Failed to execute calculate_grade.")
    #     exit(result.returncode)

    preselected_filter(extrapolative_candidates, gamma_tolerance, gamma_max, max_structures=max_structures, gamma_max0=100000)


    args = [potential, training_set, extrapolative_candidates, selected_extrapolative]
    result = subprocess.run(["mpirun", "-n", "1", mlp, "select_add", *args], text=True, check=True, env=_env)
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
    args = [potential, training_set, "--save_to=tmp_{}".format(potential), "--iteration_limit=" + str(iteration_limit), "--al_mode=nbh"]
    print("running training with args: ", args)
    result = subprocess.run(["mpirun", mlp, "train", *args], text=True, check=True, env=_env)
    if result.returncode == 0:
        # copy tmp potential
        os.rename("tmp_{}".format(potential), potential)
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
