#!/usr/bin/env python
# coding=utf-8
import numpy as np
from pymatgen.core.structure import Structure
from stopping_power_ml.rc import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import linalg

class Cell:
    """
    extract the crystal information that will be useful for later simulation
    """
    def __init__(self, atoms):
        """Create the class
        :param atoms: ase.Atoms, crystal structure being studied (including particle as the last atom)
        """
        # Store the main details
        self.atoms = atoms

        # Compute the primitive unit cell vectors
        self.simulation_cell = AseAtomsAdaptor.get_structure(atoms)
        spg = SpacegroupAnalyzer(self.simulation_cell)
        self.prim_strc = spg.find_primitive()
        self.conv_strc = spg.get_conventional_standard_structure()

        self.input_to_conv = np.round(linalg.solve(self.conv_strc.lattice.matrix, self.simulation_cell.lattice.matrix))

        # Compute the matrix that we will use to map lattice vectors in the conventional cell to ones in the primitive
        self.conv_to_prim = np.round(linalg.solve(self.prim_strc.lattice.matrix, self.conv_strc.lattice.matrix))

        logging.info(f"Original simulation cell (bohr)\n {self.simulation_cell}")
        logging.info(f"Conventional unit cell (bohr)\n {self.conv_strc}")
        logging.info(f"Primitive cell (bohr)\n {self.prim_strc}")
        logging.info(f"Supercell to conventional unit cell conversion\n {self.input_to_conv}")
        logging.info(f"Conventional unit cell to primitive cell conversion\n {self.conv_to_prim}")

    def input_cell_lattice(self):
        return self.simulation_cell.lattice.matrix
    
    def conventional_cell_lattice(self):
        return self.conv_strc.lattice.matrix

    def primitive_cell_lattice(self):
        return self.prim_strc.lattice.matrix
