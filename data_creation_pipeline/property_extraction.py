import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Data, Batch
import os
import pubchempy as pcp
import periodictable as pt
import re
import periodictable as pt
import re


def parse_formula(formula):
    # Regular expression to match elements and their counts
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)

    # Create a dictionary to store element counts
    element_counts = {}
    for (element, count) in matches:
        count = int(count) if count else 1
        if element in element_counts:
            element_counts[element] += count
        else:
            element_counts[element] = count

    return element_counts


def calculate_molecular_weight(formula):
    element_counts = parse_formula(formula)
    molecular_weight = 0.0

    for element, count in element_counts.items():
        atomic_weight = pt.elements.symbol(element).mass
        molecular_weight += atomic_weight * count

    return molecular_weight


element_atomic_number = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
    'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56,
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
    'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88,
    'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
    'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}


element_atomic_radius = {
    'H': 53, 'He': 31,
    'Li': 167, 'Be': 112, 'B': 87, 'C': 67, 'N': 56, 'O': 48, 'F': 42, 'Ne': 38,
    'Na': 190, 'Mg': 145, 'Al': 118, 'Si': 111, 'P': 98, 'S': 88, 'Cl': 79, 'Ar': 71,
    'K': 243, 'Ca': 194, 'Sc': 184, 'Ti': 176, 'V': 171, 'Cr': 166, 'Mn': 161, 'Fe': 156, 'Co': 152, 'Ni': 149, 'Cu': 145, 'Zn': 142,
    'Ga': 136, 'Ge': 125, 'As': 114, 'Se': 103, 'Br': 94, 'Kr': 88,
    'Rb': 265, 'Sr': 219, 'Y': 212, 'Zr': 206, 'Nb': 198, 'Mo': 190, 'Tc': 183, 'Ru': 178, 'Rh': 173, 'Pd': 169, 'Ag': 165, 'Cd': 161,
    'In': 156, 'Sn': 145, 'Sb': 133, 'Te': 123, 'I': 115, 'Xe': 108,
    'Cs': 298, 'Ba': 253, 'La': 195, 'Ce': 185, 'Pr': 247, 'Nd': 206, 'Pm': 205, 'Sm': 238, 'Eu': 231, 'Gd': 233, 'Tb': 225, 'Dy': 228,
    'Ho': 226, 'Er': 226, 'Tm': 222, 'Yb': 222, 'Lu': 217,
    'Hf': 208, 'Ta': 200, 'W': 193, 'Re': 188, 'Os': 185, 'Ir': 180, 'Pt': 177, 'Au': 174, 'Hg': 171, 'Tl': 156, 'Pb': 154, 'Bi': 143,
    'Po': 135, 'At': 127, 'Rn': 120,
    'Fr': 260, 'Ra': 221, 'Ac': 215, 'Th': 206, 'Pa': 200, 'U': 196, 'Np': 190, 'Pu': 187, 'Am': 180, 'Cm': 173, 'Bk': 170, 'Cf': 167,
    'Es': 165, 'Fm': 167, 'Md': 173, 'No': 176, 'Lr': 161,
    'Rf': 157, 'Db': 149, 'Sg': 143, 'Bh': 141, 'Hs': 134, 'Mt': 129, 'Ds': 128,
    'Rg': 121, 'Cn': 122, 'Nh': 124, 'Fl': 124, 'Mc': 125, 'Lv': 126, 'Ts': 127, 'Og': 128
}

element_electronegativity = {
    'H': 2.20, 'He': None,
    'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': None, 'Sm': 1.17, 'Eu': None, 'Gd': 1.20, 'Tb': None, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': None, 'Lu': 1.27,
    'Hf': 1.3, 'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02,
    'Po': 2.0, 'At': 2.2, 'Rn': None,
    'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3, 'Bk': 1.3, 'Cf': 1.3,
    'Es': 1.3, 'Fm': 1.3, 'Md': 1.3, 'No': 1.3, 'Lr': 1.3,
    'Rf': None, 'Db': None, 'Sg': None, 'Bh': None, 'Hs': None, 'Mt': None, 'Ds': None,
    'Rg': None, 'Cn': None, 'Nh': None, 'Fl': None, 'Mc': None, 'Lv': None, 'Ts': None, 'Og': None
}


def extract_bond_values(sdf_data):
    # Split the data into lines
    lines = sdf_data.split('\n')

    # Find the start of the atom block
    atom_count_line = lines[3]
    atom_count = int(atom_count_line.split()[0])
    bond_count = int(atom_count_line.split()[1])

    # Calculate the start and end of the bond block
    bond_block_start = 4 + atom_count
    bond_block_end = bond_block_start + bond_count

    # Extract bond information
    bonds = []
    for line in lines[bond_block_start:bond_block_end]:
        parts = line.split()
        atom1 = int(parts[0])-1
        atom2 = int(parts[1])-1
        bond_type = int(parts[2])
        bonds.append((atom1, atom2, bond_type))
        bonds.append((atom2, atom1, bond_type))

    return bonds


def find_cations_anions(sdf_data, nr_of_atoms, string1='anion', string2='cation'):
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    lines = sdf_data.split('\n')
    # Flags to track when we are inside the relevant section
    in_pharmacophore_features = False

    cations = []
    anions = []

    for line in lines:
        if 'PUBCHEM_PHARMACOPHORE_FEATURES' in line:
            in_pharmacophore_features = True
        # Assuming each section ends with 'END'
        elif in_pharmacophore_features and len(line) == 0:
            break
        elif in_pharmacophore_features:
            if string1 in line:
                # Extract atom identifiers for cations
                cations.extend(line.split(' ')[1:-1])
            elif string2 in line:
                # Extract atom identifiers for anions
                anions.extend(line.split(' ')[1:-1])

    cations_anions = np.zeros((nr_of_atoms, 1))
    for cation in cations:
        cations_anions[int(cation)-1] = 1
    for anion in anions:
        cations_anions[int(anion)-1] = -1
    return cations_anions


def find_partial_charges(sdf_data, atom_nr):
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    lines = sdf_data.split('\n')
    # Flags to track when we are inside the partial charges section
    in_partial_charges_section = False
    partial_charges = []
    atom_index_list = []

    for line in lines:
        if 'PUBCHEM_MMFF94_PARTIAL_CHARGES' in line:  # Hypothetical section name
            in_partial_charges_section = True
        elif in_partial_charges_section and len(line) == 0:
            break
        elif in_partial_charges_section:
            if len(line.split()) > 1:
                # Assuming each line in this section has format: "atom_index charge"
                parts = line.split()
                atom_index = int(parts[0])
                charge = float(parts[1])
                atom_index_list.append(atom_index-1)
                partial_charges.append(charge)
    charges = np.zeros((atom_nr, 1))
    for i, index in enumerate(atom_index_list):
        charges[index] = partial_charges[i]
    return charges


def extract_number_of_atoms(sdf_data):
    lines = sdf_data.split('\n')  # Split the SDF data into lines
    if len(lines) > 3:  # Ensure there are enough lines to contain the counts line
        counts_line = lines[3]  # The fourth line contains the counts
        # Extract and convert the number of atoms
        num_atoms = int(counts_line[:3].strip())
        return num_atoms
    else:
        return "Error: SDF data does not contain enough lines."


# Step 3 & 4: Search for the target edge and find its index
def find_pair_index(edges, target_pair):
    # Transpose back to make searching easier if pairs are columns
    edges_transposed = edges.T
    # Convert target pair to a NumPy array for comparison
    target_array = np.array(target_pair)
    # Find the index
    for index, pair in enumerate(edges_transposed):
        if np.array_equal(pair, target_array):
            return index
    return -1


def get_graph_but_better(atoms, pos, nodes, bond_emb, physical_properties, property, target=None, cutoff=False, distance=8, ):
    '''
    Get a Data graph from the numpy coordinates, the type of atom and the target.
    '''
    # edge index
    a = np.arange(len(atoms))
    edges = np.array(np.meshgrid(a, a)).T.reshape(-1, 2).T
    # edges = torch.tensor(edges, dtype=torch.int64)

    atomic_nums = np.asarray([element_atomic_number[atom] for atom in atoms])[
        :, np.newaxis]  # keep as numpy for later use
    electroneg = torch.tensor(np.asarray([element_electronegativity[atom] for atom in atoms])[
                              :, np.newaxis], dtype=torch.float)
    atomic_radius = torch.tensor(np.asarray(
        [element_atomic_radius[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)

    pair_dist = pairwise_distances(pos)
    cm = (atomic_nums*atomic_nums.T) / pair_dist
    np.fill_diagonal(cm, 0.5*atomic_nums**2.4)

    cm = cm.flatten()[:, np.newaxis]
    bonds = torch.zeros((len(atoms)*len(atoms), 1))
    for i in range(len(bond_emb)):
        bond_temp = bond_emb[i]

        edge_1 = bond_temp[0]
        edge_2 = bond_temp[1]
        edge_to_find = (edge_1, edge_2)
        index = find_pair_index(edges, edge_to_find)
        # print(index)
        bonds[index] = bond_temp[2]
    bonds = torch.tensor(bonds, dtype=torch.float)
    edge_attr = torch.cat([torch.tensor(cm, dtype=torch.float), bonds, torch.tensor(
        pair_dist.flatten()[:, np.newaxis], dtype=torch.float)], dim=1)
    edges = torch.tensor(edges, dtype=torch.int64)
    if target:
        target = torch.tensor(target, dtype=torch.float)

    node_attrs = torch.cat(
        [torch.tensor(atomic_nums, dtype=torch.float), electroneg, atomic_radius], dim=1)
    node_attrs = torch.cat([node_attrs, nodes], dim=1)
    print("property", property)
    graph = Data(x=node_attrs,
                 pos=torch.tensor(pos, dtype=torch.float),
                 z=torch.tensor(atomic_nums.squeeze(), dtype=torch.int),
                 edge_index=edges,
                 edge_attr=edge_attr,
                #  dP=target,
                 #  p=target[1],
                 #  h=target[2],
                 physical_properties=torch.tensor(
                     physical_properties).reshape(1, 4)
                 )
    setattr(graph, property, target)

    return graph


def get_complexity(compound_name):
    compounds = pcp.get_compounds(compound_name, 'name')
    if compounds:
        compound = compounds[0]
        complexity = compound.complexity
        return complexity
    return None


def get_tpsa(compound_name):
    compounds = pcp.get_compounds(compound_name, 'name')
    if compounds:
        compound = compounds[0]
        tpsa = compound.tpsa
        return tpsa
    return None


def get_molecular_formula(compound_name):
    # Search for the compound by name
    compounds = pcp.get_compounds(compound_name, 'name')

    if compounds:
        # Get the molecular formula of the first matching compound
        molecular_formula = compounds[0].molecular_formula
        return molecular_formula
    else:
        return None


def get_xlog(compound_name):
    # Search for the compound by name
    compounds = pcp.get_compounds(compound_name, 'name')

    if compounds:
        # Get the density of the first matching compound
        # Note: PubChem does not provide density directly, but xlogp can be used as a proxy
        xlogp = compounds[0].xlogp
        return xlogp
    else:
        return None
