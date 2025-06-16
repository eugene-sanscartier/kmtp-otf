import numpy
import ase


def read_cfg(fileobj):
    lines = fileobj.readlines()
    images = []
    while len(lines) > 0:

        line = lines.pop(0)
        if line.strip() == "BEGIN_CFG":
            size = 0
            energy = None
            supercell, id, type, cartes, f, nbh_grades, stress = [], [], [], [], [], [], []
            calcs, features = {}, {}
            stress_type = None
        elif line.strip() == "Size":
            size = int(lines.pop(0).strip())
        elif line.strip() == "Supercell":
            for _ in range(3):
                line = lines.pop(0).strip()
                supercell += [[float(x) for x in line.split()]]
        elif line.strip().startswith("AtomData:"):
            fields = line.split()[1:]
            for i in range(size):
                fields_data = lines.pop(0).strip().split()
                if "id" in fields:
                    id += [int(fields_data[fields.index("id")])]
                if "type" in fields:
                    type += [int(fields_data[fields.index("type")]) + 1]
                if "cartes_x" in fields and "cartes_y" in fields and "cartes_z" in fields:
                    cartes += [[float(fields_data[fields.index("cartes_x")]), float(fields_data[fields.index("cartes_y")]), float(fields_data[fields.index("cartes_z")])]]
                if "fx" in fields and "fy" in fields and "fz" in fields:
                    f += [[float(fields_data[fields.index("fx")]), float(fields_data[fields.index("fy")]), float(fields_data[fields.index("fz")])]]
                if "nbh_grades" in fields:
                    nbh_grades += [float(fields_data[fields.index("nbh_grades")])]
        elif line.strip() == "Energy":
            energy = float(lines.pop(0).strip())
        elif line.strip().startswith("PlusStress:"):
            stress_type = "PlusStress"
            stress = [float(x) for x in lines.pop(0).strip().split()]
        elif line.strip().startswith("Feature"):
            _, feature_name, feature_value = line.strip().split()
            features[feature_name] = feature_value
        elif line.strip() == "END_CFG":
            if energy != None:
                calcs["energy"] = energy
            if f != []:
                calcs["forces"] = f
            if stress != []:
                if stress_type == "PlusStress":
                    stress = numpy.array(stress, dtype=float) / ase.Atoms(cell=supercell).get_volume() * -1
                calcs["stress"] = stress

            atoms = ase.Atoms(numbers=type, positions=cartes, cell=supercell, pbc=True)
            atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(atoms, **calcs)

            if nbh_grades != []:
                atoms.set_array("nbh_grades", numpy.array(nbh_grades))
                features["MV_grade"] = numpy.max(nbh_grades)
            if features != {}:
                atoms.info["features"] = features

            images += [atoms]

    return images

def write_cfg(fileobj, images, fmt='%12.6f'):
    def map2ranks(arr):
        rank_map = {val: i for i, val in enumerate(sorted(set(arr)))}
        return [rank_map[num] for num in arr]

    for atoms in images:
        fileobj.write("BEGIN_CFG\n")
        fileobj.write(" Size\n")
        fileobj.write("%9d\n" % len(atoms))
        fileobj.write(" Supercell\n")
        fileobj.write("    {} {} {}\n".format(fmt % atoms.get_cell()[0][0], fmt % atoms.get_cell()[0][1], fmt % atoms.get_cell()[0][2]))
        fileobj.write("    {} {} {}\n".format(fmt % atoms.get_cell()[1][0], fmt % atoms.get_cell()[1][1], fmt % atoms.get_cell()[1][2]))
        fileobj.write("    {} {} {}\n".format(fmt % atoms.get_cell()[2][0], fmt % atoms.get_cell()[2][1], fmt % atoms.get_cell()[2][2]))

        fields, fields_data = ["id", "type", "cartes_x", "cartes_y", "cartes_z"], {"id":[], "type":[], "cartes_x":[], "cartes_y":[], "cartes_z":[]}
        if atoms.calc != None and "forces" in atoms.calc.results:
            fields += ["fx", "fy", "fz"]
            fields_data["fx"], fields_data["fy"], fields_data["fz"] = [], [], []
        if "nbh_grades" in atoms.arrays:
            fields += ["nbh_grades"]
            fields_data["nbh_grades"] = []

        for i in range(len(atoms)):
            fields_data["id"] += [i+1]
            fields_data["type"] += [map2ranks(atoms.get_atomic_numbers())[i]]
            fields_data["cartes_x"] += [atoms.get_positions()[i][0]]
            fields_data["cartes_y"] += [atoms.get_positions()[i][1]]
            fields_data["cartes_z"] += [atoms.get_positions()[i][2]]
            if atoms.calc != None and "forces" in atoms.calc.results:
                fields_data["fx"] += [atoms.calc.results["forces"][i][0]]
                fields_data["fy"] += [atoms.calc.results["forces"][i][1]]
                fields_data["fz"] += [atoms.calc.results["forces"][i][2]]
            if "nbh_grades" in atoms.arrays:
                fields_data["nbh_grades"] += [atoms.arrays["nbh_grades"][i]]

        if "nbh_grades" in atoms.arrays:
            if "features" in atoms.info:
                atoms.info["features"]["MV_grade"] = numpy.max(atoms.arrays["nbh_grades"])
            else:
                atoms.info["features"] = {"MV_grade": numpy.max(atoms.arrays["nbh_grades"])}

        fileobj.write(" AtomData:  " + "    ".join(fields) + "\n")
        for i in range(len(atoms)):
            fileobj.write(" {:9d} {:3d} {} {} {} ".format(fields_data["id"][i], fields_data["type"][i], fmt % fields_data["cartes_x"][i], fmt % fields_data["cartes_y"][i], fmt % fields_data["cartes_z"][i]))
            if atoms.calc != None and "forces" in atoms.calc.results:
                fileobj.write(" {} {} {} ".format(fmt % fields_data["fx"][i], fmt % fields_data["fy"][i], fmt % fields_data["fz"][i]))
            if "nbh_grades" in atoms.arrays:
                fileobj.write(" {}".format(fmt % fields_data["nbh_grades"][i]))
            fileobj.write("\n")

        fileobj.write(" Energy\n")
        if atoms.calc != None and "energy" in atoms.calc.results:
            fileobj.write("    {}\n".format(fmt % atoms.calc.results["energy"]))
        else:
            fileobj.write("    {}\n".format(fmt % 0.0))

        if atoms.calc != None and "stress" in atoms.calc.results:
            stress_fields, stress_fields_data = ["xx", "yy", "zz", "yz", "xz", "xy"], {}
            for i, stress_field in enumerate(stress_fields): stress_fields_data[stress_field] = atoms.calc.results["stress"][i] * atoms.get_volume() * -1
            fileobj.write(" PlusStress:  " + "   ".join(stress_fields) + "\n")
            fileobj.write("    {} {} {} {} {} {}\n".format(fmt % stress_fields_data["xx"], fmt % stress_fields_data["yy"], fmt % stress_fields_data["zz"], fmt % stress_fields_data["yz"], fmt % stress_fields_data["xz"], fmt % stress_fields_data["xy"]))

        if "features" in atoms.info:
            for feature_name, feature_value in atoms.info["features"].items():
                fileobj.write(" Feature    {} {}\n".format(feature_name, feature_value))

        fileobj.write("END_CFG\n")
        fileobj.write("\n")
