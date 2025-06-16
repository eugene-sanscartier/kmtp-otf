import sys
import os
import logging
import subprocess
import pickle
import argparse

# import pyace
import pandas
import numpy
import ase

# import pyace.aceselect
# import pyace.activelearning

import ase.io.lammpsrun
import ase.calculators.lammpsrun

os.environ["ASE_LAMMPSRUN_COMMAND"] = "/usr/bin/mpirun --np 1 /home/eugene/Doctorat/code_library/lammps/build/lmp"
mlp = "/mnt/home/Doctorat/code_library/mlip-3/bin/mlp"

from ase.atoms import Atoms

from main import write_cfg, read_cfg

# def read_xyz(fileobj, index):
#     lines = fileobj.readlines()
#     images = []
#     while len(lines) > 0:
#         symbols = []
#         positions = []
#         natoms = int(lines.pop(0))
#         lines.pop(0)  # Comment line; ignored
#         for _ in range(natoms):
#             line = lines.pop(0)
#             symbol, x, y, z = line.split()[:4]
#             symbol = symbol.lower().capitalize()
#             symbols.append(symbol)
#             positions.append([float(x), float(y), float(z)])
#         images.append(Atoms(symbols=symbols, positions=positions))
#     yield from images[index]


# def write_cfg(fileobj, images, fmt='%22.15f'):
#     for atoms in images:
#         fileobj.write("BEGIN_CFG\n")
#         fileobj.write(" Size\n")
#         fileobj.write("%9d\n" % len(atoms))
#         fileobj.write(" Supercell\n")
#         fileobj.write("    %s %s %s\n" % (fmt % atoms.get_cell()[0][0], fmt % atoms.get_cell()[0][1], fmt % atoms.get_cell()[0][2]))
#         fileobj.write("    %s %s %s\n" % (fmt % atoms.get_cell()[1][0], fmt % atoms.get_cell()[1][1], fmt % atoms.get_cell()[1][2]))
#         fileobj.write("    %s %s %s\n" % (fmt % atoms.get_cell()[2][0], fmt % atoms.get_cell()[2][1], fmt % atoms.get_cell()[2][2]))
#         fileobj.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z      fx      fy      fz\n")
#         for i, (symbol, (x, y, z)) in enumerate(zip(atoms.symbols, atoms.positions)):
#             fileobj.write("%9d %3d %s %s %s %s %s %s\n" % (i + 1, atoms.get_atomic_numbers()[i] - 1, fmt % x, fmt % y, fmt % z, fmt % atoms.get_forces()[i, 0], fmt % atoms.get_forces()[i, 1], fmt % atoms.get_forces()[i, 2]))
#         fileobj.write(" Energy\n")
#         fileobj.write("    %s\n" % (fmt % atoms.get_potential_energy()))
#         fileobj.write(" Feature    NotEvaluated 0\n")
#         fileobj.write("END_CFG\n")
#         fileobj.write("\n")


from lammps import lammps

def run_si_nvt_simulation():
    # Initialize LAMMPS
    lmp = lammps()
    script = """
        units       metal
        dimension   3
        boundary    p p p
        atom_style  atomic
        atom_modify sort 0 1

        lattice        diamond 5.5421217827
        region         box block 0 2 0 2 0 2
        create_box     2 box
        create_atoms   1 box basis 1 1 basis 2 1 basis 3 1 basis 4 1 basis 5 2 basis 6 2 basis 7 2 basis 8 2

        group target_atom id 42
        delete_atoms group target_atom

        pair_style sw
        pair_coeff * * SiGe.sw Si Ge

        mass 1 28.09
        mass 2 72.64
        timestep 0.001

        thermo 1
        thermo_style custom step pe ke etotal fnorm

        minimize 1.0e-16 1.0e-16 10000 100000

        dump dump all custom 1 startup_config.dump id type x y z fx fy fz
        run 0
    """

    lmp.commands_list(script.strip().split("\n"))

    lmp.close()

run_si_nvt_simulation()

# Setup calculator, load /create structure, calculate energy and forces, save to file
lammps_params = {'pair_style': 'sw', 'pair_coeff': ['* * SiGe.sw Si Ge']}
files = ["SiGe.sw"]

startup_config_filename = "startup_config.dump"

lammps_calc = ase.calculators.lammpsrun.LAMMPS(files=files, specorder=["H", "He"], **lammps_params)
with open(startup_config_filename) as config_file:
    structures = ase.io.lammpsrun.read_lammps_dump_text(config_file, index=slice(None))

for i, structure in enumerate(structures):
    print(f"Calculating structure {i+1}/{len(structures)}")

    structure.calc = lammps_calc
    structure.get_potential_energy()
    structure.get_forces()

    structure.get_stress()
    # structure.calc.results["stress"] *= structure.get_volume()

structures_filename = "train.cfg"
with open(structures_filename, "w") as f:
    write_cfg(f, structures)

args = ["16.almtp", "train.cfg", "--save_to=sige.almtp", "--iteration_limit=100", "--al_mode=nbh"]
result = subprocess.run(["mpirun", mlp, "train", *args], text=True)
