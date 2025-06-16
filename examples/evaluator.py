import os
import ase

os.environ["ASE_LAMMPSRUN_COMMAND"] = "/usr/bin/mpirun --np 1 /home/eugene/Doctorat/code_library/lammps/build/lmp"

# Evaluate selected structure
def evaluator(structure):
    lammps_params = {'pair_style': 'sw', 'pair_coeff': ['* * SiGe.sw Si Ge']}
    elements = ["Si", "Ge"]
    files = ["SiGe.sw"]

    def lammps_calc(): return ase.calculators.lammpsrun.LAMMPS(files=files, specorder=None, **lammps_params)

    structure.calc = lammps_calc()
    structure.get_potential_energy()
    structure.get_forces()
    structure.get_stress()

    return structure
