#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice

# Define unit cell and supercell lattice
unitcell_lattice = Lattice.cubic(3.0)
supercell_matrix = np.array([[4, 0, 0],
                             [0, 3, 0],
                             [0, 0, 1]])  # Supercell is 2x2x1
supercell_lattice = Lattice(supercell_matrix @ unitcell_lattice.matrix)

# Contravariant vector in supercell basis
v_frac_supercell = np.array([0.75, 0.75, 0.0])

# Step 1: Convert to Cartesian
v_cart = supercell_lattice.get_cartesian_coords(v_frac_supercell)

# Step 2: Convert to unit cell fractional (contravariant)
v_frac_unitcell = unitcell_lattice.get_fractional_coords(v_cart)

print("Vector in unit cell fractional coordinates:", v_frac_unitcell)

