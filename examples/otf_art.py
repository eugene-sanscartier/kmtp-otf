import subprocess

from lammps import lammps

def run_si_nvt_simulation():
    # Initialize LAMMPS
    lmp = lammps()

    # Define LAMMPS input script
    script = """
        units       metal
        dimension   3
        boundary    p p p
        atom_style  atomic
        atom_modify sort 0 1

        read_data conf.sw

        pair_style mtp/extrapolation sige.almtp

        mass 1 28.09
        mass 2 72.64

        fix extrapolation_grade all pair 1 mtp/extrapolation extrapolation 1
        compute max_grade all pair mtp/extrapolation
        variable max_grade equal c_max_grade[1]

        dump 1 all custom 1 lammps_run.dump id type x y z fx fy fz f_extrapolation_grade

        variable dump_skip equal "v_max_grade < {gamma_select}"
        dump extrapolative_structures_dump all custom 1 extrapolative_structures.dump id type x y z fx fy fz f_extrapolation_grade
        dump_modify extrapolative_structures_dump skip v_dump_skip

        fix extreme_extrapolation all halt 1 v_max_grade > {gamma_max}

        thermo 1
        thermo_style custom step pe v_max_grade

        plugin load /mnt/home/Doctorat/code_library/artn-plugin-mtp/build/libartn.so
        fix 10 all artn dmax 5.0
        timestep 0.001
        reset_timestep 0
        min_style fire
        minimize 1e-5 1e-5 5000 10000
    """.format(gamma_select=gamma_select, gamma_max=gamma_max)

    # Execute LAMMPS commands
    lmp.commands_list(script.strip().split("\n"))

    max_grade = lmp.extract_variable("max_grade")

    # Close LAMMPS instance
    lmp.close()

    return max_grade

gamma_select = 2.2
gamma_max = 5.5

while run_si_nvt_simulation() > gamma_max:
    # main.py [-h] [-t TRAINING_DATASET] [-p POTENTIAL] [-e ELEMENTS] [-g GAMMA_TOLERANCE] [-i MAXVOL_ITERS] [-r MAXVOL_REFINEMENT] extrapolative_dataset [extrapolative_dataset ...]
    args = ["--potential", "sige.almtp", "--training_set", "train.cfg", "extrapolative_structures.dump"]
    ret = subprocess.run(["python", "main.py", *args], text=True)

    if ret.returncode != 0:
        exit(ret.returncode)
