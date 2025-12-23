# %%
import warnings
from glob import *
# from rdkit.Chem import Descriptors
import requests
import pandas as pd
import re
# from bs4 import BeautifulSoup
import os
import torch
from helpers import *
# from helpers import get_graph
from property_extraction import *
abs_path = os.path.dirname(os.path.abspath(__file__))
extrs = list()

# suppress warnings
warnings.filterwarnings("ignore")
# %%


def extract_coordinates(sdf_data):
    lines = sdf_data.split('\n')
    atom_count_line_index = 3  # Assuming standard SDF format
    atom_count_line = lines[atom_count_line_index]
    # Extract number of atoms from the counts line
    num_atoms = int(atom_count_line.split()[0])

    coordinates = []
    for i in range(atom_count_line_index + 1, atom_count_line_index + 1 + num_atoms):
        atom_info = lines[i].split()
        # Convert string coordinates to float
        x, y, z = map(float, atom_info[:3])
        coordinates.append((x, y, z))

    return coordinates


def extract_atom_info(sdf_data):
    lines = sdf_data.split('\n')
    atom_count_line_index = 3  # Assuming standard SDF format
    atom_count_line = lines[atom_count_line_index]
    # Extract number of atoms from the counts line
    num_atoms = int(atom_count_line.split()[0])

    atom_info_list = []
    for i in range(atom_count_line_index + 1, atom_count_line_index + 1 + num_atoms):
        atom_line = lines[i].split()
        # Convert string coordinates to float
        x, y, z = map(float, atom_line[:3])
        atom_symbol = atom_line[3]  # Extract the atom symbol
        atom_info_list.append((atom_symbol))

    return atom_info_list


def get_molecule_coordinates_from_pubchem(chemical_name):
    # URL for PubChem's REST API to search for a compound by name and retrieve its 3D structure in SDF format
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/record/SDF/?record_type=3d"

    response = requests.get(url)
    if response.status_code == 200:
        # "found"
        print(chemical_name, " found")
        return response.text, True # Returns the SDF content with 3D coordinates
    else:
        print("Error: Could not retrieve data from PubChem.")
        return "Error: Could not retrieve data from PubChem.", False


# %%

# read the all_DN.csv file and drop all rows that have an empty "Donor number" value
# Read the all_DN.csv file and drop all rows that have an empty "Donor numberDN" value
def create_graphs(file, property):
    df = pd.read_csv('DN.csv')
    df = df.dropna(subset=[property])

    # Convert the "Donor numberDN" column to numeric, forcing non-numeric values to NaN
    df[property] = pd.to_numeric(df[property], errors='coerce')

    # Drop rows where "Donor numberDN" is NaN
    df = df.dropna(subset=[property])
    # print(df)
    data_found = data_found = pd.DataFrame(
        columns=['Name', 'Coordinates', 'Atoms', property, ''])
    counter = 0
    for index, row in df.iterrows():
        # if row[property] == 0:
        #     continue
        # print(index)
        i = re.split(r'[&@]', row['Name'])

        print(row['Name'])
        # print(i)
        for j in range(len(i)):
            i_temp= i[j].strip()
            print(i_temp)
            sdf_data, found_flag = get_molecule_coordinates_from_pubchem(i_temp)
            if found_flag:
                i  = i_temp
                break

        # print(sdf_data)
        try:
            atoms = extract_atom_info(sdf_data)
            nr_atoms = extract_number_of_atoms(sdf_data)
            bonds = extract_bond_values(sdf_data)
            charge = find_partial_charges(sdf_data, nr_atoms)
            anion_cation = find_cations_anions(
                sdf_data, nr_atoms, string1='anion', string2='cation')
            acceptor = find_cations_anions(
                sdf_data, nr_atoms, string1='acceptor', string2='donor')
            ring = find_cations_anions(
                sdf_data, nr_atoms, string1='ring', string2='SNEROCKS')
            node_properties = np.stack(
                (charge, anion_cation, acceptor, ring), axis=1)
            node_properties = torch.tensor(
                node_properties, dtype=torch.float).squeeze()
            edge_properties = bonds
            coordiantes = extract_coordinates(sdf_data)
            # print("Found: ", i)
            molecular_formula = get_molecular_formula(i)
            # print(molecular_formula)
            molecular_weight = calculate_molecular_weight(molecular_formula)
            # print(molecular_weight)
            xlog = get_xlog(i)
            tpsa = get_tpsa(i)
            complexity = get_complexity(i)
            # print(xlog, molecular_weight, tpsa, complexity)
            # physical_properties = torch.tensor([xlog, molecular_weight], dtype=torch.float)
            # print(molecular_weight)
            physical_properties = [xlog, molecular_weight, tpsa, complexity]

            new_row = {'Name': [i],
                    'Coordinates': [extract_coordinates(sdf_data)],
                    'Atoms': [atoms],
                    property: [row[property]],
                    'node_properties': [node_properties],
                    'edge_properties': [edge_properties],
                    'physical_properties': [physical_properties]}
            # data_found = data_found.append(new_row, ignore_index=True)
            data_found = pd.concat(
                [data_found, pd.DataFrame(new_row)], ignore_index=True)
            graph_temp = get_graph_but_better(atoms, extract_coordinates(sdf_data), node_properties,
                                            edge_properties, physical_properties, property, target=[row[property]], cutoff=False, distance=8, )
            print(graph_temp)
            graph_temp.name = row['Name']
            filename = f'/{i}.pt'
            torch.save(graph_temp, processed_data_path + filename)
            print("Saved: ", i, "in ", processed_data_path)
            if i == "dimethylsulfoxide":
                print(i, [row[property]])

        except Exception as e:
            print(f"Error in line {e.__traceback__.tb_lineno} of file {__file__}: {e}")
            # print("Not! Found: ", i)
            continue

# %%
