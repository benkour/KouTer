import os
import sys
import subprocess
import shutil

import pickle

from tqdm import tqdm
import numpy as np

import pandas as pd

import torch

from rdkit.Chem import EditableMol, Conformer
from rdkit import Chem
import pubchempy as pcp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

OCHEM_DATA_FILE = './ochem_dataset/ochem_data.csv'
OCHEM_METADATA_COLUMNS = ['SMILES', 'CASRN', 'EXTERNALID', 'N', 'NAME', 'ARTICLEID', 'PUBMEDID', 'PAGE', 'TABLE', 'ERROR','rdkit_smiles']
input_path = os.path.join(os.getcwd(), "graph_in")
output_path = os.path.join(os.getcwd(), "graph_out")
raw_descriptor_path = os.path.join(os.getcwd(), "raw_descriptors")
solvent_graphs_name = 'solvent'

descriptors_selection_thresholds = {
    'default' : [0.3, 0.001, 0.9, 0.5],
    'a': [0.3, 0.001, 0.9, 0.3],
}

def check_compound(compound, graph):
    '''Check if the compound matches the graph data'''
    # Check indices
    name = graph['name'].split('&')[0]
    for i in range(len(graph['z'])):
        if compound.to_dict()['atoms'][i]['number'] != graph['z'][i]:
            print(f"Atomic number mismatch in compound {name} at index {i}: {compound.to_dict()['atoms'][i]['number']} != {graph['z'][i]}")
            return False
    return True

def get_bonds(graph, gname):
    name = gname#graph['name'].split('&')[0]
    compounds = pcp.get_compounds(name, 'name')
    if len(compounds) == 0:
        print(f"Could not find {name} in pubchempy")
        return None
    compound = compounds[0]
    if not check_compound(compound, graph):
        return None
    return compounds[0].to_dict()['bonds']

def graph_to_mol(graph, gname):
    '''Convert graphs to RDKit Mol objects with 3D coordinates'''
    emol = EditableMol(Chem.Mol())
    # Add atoms
    atom_indices = []
    for atomic_num in graph['z']:
        atom = Chem.Atom(int(atomic_num))
        idx = emol.AddAtom(atom)
        atom_indices.append(idx)
    bonds = get_bonds(graph, gname)
    if bonds is None:
        return None
    for bond in bonds:
        emol.AddBond(bond['aid1'] - 1, bond['aid2'] - 1,
                     Chem.BondType.SINGLE if bond['order'] == 1 else Chem.BondType.DOUBLE)
    mol = emol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"Sanitization failed for {graph['name']}: {e}")
        return None
    conf = Conformer(len(atom_indices))
    for i, pos in enumerate(graph['pos']):
        pos_arr = np.array(pos, dtype=float).flatten()
        conf.SetAtomPosition(i, pos_arr.tolist())
    mol.AddConformer(conf, assignId=True)
    return mol
def add_descriptors_to_graphs(property):
    if not os.path.exists(OCHEM_DATA_FILE):
        raise FileNotFoundError("OCHEM dataset file 'ochem_data.csv' not found in './ochem_dataset/' directory.")

    ochem_datafile = OCHEM_DATA_FILE
    print(f"Loading ochem database from file: {ochem_datafile}")
    ochem_data = pd.read_csv(ochem_datafile)

    graph_targets = os.listdir(input_path)
    # keep graph targets that contain the property string
    graph_targets = [gt for gt in graph_targets if property or "solvent" in gt]
    if not graph_targets:
        raise FileNotFoundError("No graph data found in 'graph_in' directory.")
    # Put solvent graphs first because we add all other descriptors to solvent graphs at the end
    graph_targets.sort(key=lambda x: 0 if x.startswith(solvent_graphs_name) else 1)

    for target in graph_targets:
        target_name = target.split('_')[0]
        graph_data_path = os.path.join(input_path, f"{target}")
        desc_out_path = os.path.join(raw_descriptor_path, f"{target}")

        if not os.path.exists(desc_out_path):
            os.makedirs(desc_out_path)
        i = 1
        while os.path.exists(os.path.join(desc_out_path, f'log_{i}.txt')):
            i += 1
        log_file_path = os.path.join(desc_out_path, f'log_{i}.txt')
        ## This will redirect the output to a log file
        # Remove last log redirection if any
        if 'tee' in locals():
            # Restore original stdout/stderr before terminating tee
            os.dup2(original_stdout, sys.stdout.fileno())
            os.dup2(original_stderr, sys.stderr.fileno())
            tee.terminate()
            tee.wait()
        else:
            # Save original stdout/stderr for later restoration
            original_stdout = os.dup(sys.stdout.fileno())
            original_stderr = os.dup(sys.stderr.fileno())
        tee = subprocess.Popen(["tee", log_file_path], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

        print(f"Processing target: {target}")

        graph_files = [os.path.join(graph_data_path, f) for f in os.listdir(graph_data_path) if f.endswith('.pt')]
        print(len(graph_files), "graph files found in", graph_data_path)
        graphs = [torch.load(f, weights_only=False) for f in graph_files]
        graph_fn = [x.split('/')[-1].split('.')[0] for x in graph_files]

        print(f"Number of graphs: {len(graphs)}")

        # Load previously processed data if exists
        if os.path.exists(os.path.join(desc_out_path, f'processed_graphs_{target}.pkl')):
            assert os.path.exists(os.path.join(desc_out_path, f'processed_graph_names_{target}.pkl')), "Processed graph names file missing!"
            assert os.path.exists(os.path.join(desc_out_path, f'processed_graph_descriptors_{target}.pkl')), "Processed graph descriptors file missing!"
            with open(os.path.join(desc_out_path, f'processed_graphs_{target}.pkl'), 'rb') as f:
                processed_graphs = pickle.load(f)
            with open(os.path.join(desc_out_path, f'processed_graph_names_{target}.pkl'), 'rb') as f:
                processed_graph_names = pickle.load(f)
            with open(os.path.join(desc_out_path, f'processed_graph_descriptors_{target}.pkl'), 'rb') as f:
                processed_graph_descriptors = pickle.load(f)
            print(f"Loaded {len(processed_graphs)} previously processed graphs.")
        else:
            processed_graphs = [] # The graph objects
            processed_graph_names = [] # The graph file names
            processed_graph_descriptors = []

        proc, found = 0, 0
        for i, g in tqdm(enumerate(graphs), desc="Processing graphs", total=len(graphs)):
            fname = graph_fn[i]
            if fname in processed_graph_names:
                print(f"Graph {fname} already processed, skipping...")
                found += 1
                continue
            try:
                mol = graph_to_mol(g, fname)
            except Exception as e:
                print(f"Error processing graph {graph_fn[i]}: {e}\n")
                mol = None
            if mol is not None:
                proc += 1
                smiles = Chem.MolToSmiles(mol)
                ochem_vals = ochem_data.loc[ochem_data['rdkit_smiles'] == smiles]
                if ochem_vals.empty:
                    print(f"No OCHEM descriptors found for {smiles}")
                    continue
                else:
                    ochem_descriptors = [ochem_vals[col].values[0] for col in ochem_vals.columns if col not in OCHEM_METADATA_COLUMNS]
                processed_graphs.append(g)
                processed_graph_names.append(fname)
                processed_graph_descriptors.append(ochem_descriptors)

        descriptor_names = list([col for col in ochem_data.columns if col not in OCHEM_METADATA_COLUMNS])
        print(f"Total graphs: {len(graphs)}")
        print(f"Found {found} existing prcessed graphs, processed {proc} new files.")

        print(f"Processing complete for target: {target}")

        # Save pickled raw data to avoid re-processing
        if True:
            with open(os.path.join(desc_out_path, f'processed_graphs_{target}.pkl'), 'wb') as f:
                pickle.dump(processed_graphs, f)
            print(f"Processed graphs saved to {desc_out_path}")
            with open(os.path.join(desc_out_path, f'processed_graph_descriptors_{target}.pkl'), 'wb') as f:
                pickle.dump(processed_graph_descriptors, f)
            print(f"Processed graph descriptors saved to {desc_out_path}")
            with open(os.path.join(desc_out_path, f'processed_graph_names_{target}.pkl'), 'wb') as f:
                pickle.dump(processed_graph_names, f)
            print(f"Processed graph names saved to {desc_out_path}")


            with open(os.path.join(desc_out_path, f'descriptor_names_{target}.pkl'), 'wb') as f:
                pickle.dump(descriptor_names, f)
            print(f"Descriptor names saved to {desc_out_path}")

        if target_name == solvent_graphs_name:
            print(f"Target is solvent graphs, skipping descriptor selection.")
            continue

        ## Description selection
        thresholds = descriptors_selection_thresholds.get(target_name, descriptors_selection_thresholds['default'])
        threshold_zero = thresholds[0] # Remove any descriptors where fraction of graphs with absolute value less than 0.01 is more than 'threshold_zero'
        threshold_var = thresholds[1] # Remove any descriptors with variance less than 'threshold_var'
        threshold_self_corr = thresholds[2] # Remove descriptors with correlation within themselves more than 'threshold_self_corr'
        threshold_target_corr = thresholds[3] # Remove descriptors with correlation to less than 'threshold_target_corr' to target
        print("="*120)
        print("Starting descriptor selection for target:", target)

        # Remove sparsely populated descriptors
        graph_descriptors = np.array(processed_graph_descriptors)
        to_keep = np.sum(np.abs(graph_descriptors) <= 0.01, axis=0) / graph_descriptors.shape[0] < threshold_zero
        graph_descriptors = graph_descriptors[:, to_keep]
        descriptor_names = [name for i, name in enumerate(descriptor_names) if to_keep[i]]
        print(f"After removing descriptors with more than {threshold_zero*100:.2f}% zeros, {graph_descriptors.shape[1]} descriptors remain")
        # Low variance filter
        to_keep = np.var(graph_descriptors, axis=0) > threshold_var
        graph_descriptors = graph_descriptors[:, to_keep]
        descriptor_names = [name for i, name in enumerate(descriptor_names) if to_keep[i]]
        print(f"After removing descriptors with variance less than {threshold_var}, {graph_descriptors.shape[1]} descriptors remain")
        # Remove highly self-correlated descriptors
        corr_matrix = np.corrcoef(graph_descriptors, rowvar=False)
        to_remove = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i,j]) > threshold_self_corr:
                    to_remove.add(j)
        graph_descriptors = np.delete(graph_descriptors, list(to_remove), axis=1)
        descriptor_names = [name for i, name in enumerate(descriptor_names) if i not in to_remove]
        print(f"After removing descriptors with correlation more than {threshold_self_corr} within themselves, {graph_descriptors.shape[1]} descriptors remain")
        # Remove descriptors too weakly correlated to target
        target_values = np.array([graph.to_dict()[target_name].item() for graph in processed_graphs])
        corr_with_target = np.array([np.corrcoef(graph_descriptors[:,i], target_values)[0,1] for i in range(graph_descriptors.shape[1])])
        to_keep = np.abs(corr_with_target) > threshold_target_corr
        graph_descriptors = graph_descriptors[:, to_keep]
        descriptor_names = [name for i, name in enumerate(descriptor_names) if to_keep[i]]
        if len(descriptor_names) == 0:
            raise ValueError(f"No descriptors left after applying correlation threshold to target {target_name}. Consider lowering the threshold.")
        print(f"After removing descriptors with correlation less than {threshold_target_corr} to target, {graph_descriptors.shape[1]} descriptors remain")
        print("="*120)

        # Save the selected descriptors
        save_path = os.path.join(output_path, f'{target}')
        if os.path.exists(save_path):
            print(f"Output path {save_path} already exists, overwriting...")
            shutil.rmtree(save_path)
        os.makedirs(save_path)

        for i in range(len(processed_graphs)):
            graph = processed_graphs[i]
            descriptors = graph_descriptors[i].tolist()
            graph['descriptors'] = descriptors

            output_file = os.path.join(save_path, f'{processed_graph_names[i]}.pt')
            torch.save(graph, output_file)

        with open(os.path.join(save_path, f'descriptor_names.txt'), 'w') as f:
            for name in descriptor_names:
                f.write(f"{name}\n")
        print(f"Saved graphs with selected descriptors to {save_path}")

        solvent_desc_path = os.path.join(raw_descriptor_path, f"{solvent_graphs_name}_graphs")
        if os.path.exists(solvent_desc_path):
            print(f"Solvent graphs found, adding descriptors to solvent graphs...")
            solvent_output_path = os.path.join(output_path, f'{target_name}_{solvent_graphs_name}_graphs')
            with open(os.path.join(solvent_desc_path, f'processed_graphs_{solvent_graphs_name}_graphs.pkl'), 'rb') as f:
                solvent_graphs = pickle.load(f)
            with open(os.path.join(solvent_desc_path, f'processed_graph_descriptors_{solvent_graphs_name}_graphs.pkl'), 'rb') as f:
                solvent_descriptors = pickle.load(f)
            with open(os.path.join(solvent_desc_path, f'processed_graph_names_{solvent_graphs_name}_graphs.pkl'), 'rb') as f:
                solvent_names = pickle.load(f)
            with open(os.path.join(solvent_desc_path, f'descriptor_names_{solvent_graphs_name}_graphs.pkl'), 'rb') as f:
                solvent_descriptor_names = pickle.load(f)

            assert len(solvent_graphs) == len(solvent_descriptors) == len(solvent_names), "Mismatch in number of solvent graphs, descriptors, or names"

            solvent_descriptors = np.array(solvent_descriptors).astype(float)

            print(f"Loaded {len(solvent_graphs)} solvent graphs with {solvent_descriptors.shape[1]} descriptors for output")

            # Select the same descriptors as selected for the target graphs
            selected_descriptor_idx = [i for i, name in enumerate(solvent_descriptor_names) if name in descriptor_names]
            solvent_descriptors = solvent_descriptors[:, selected_descriptor_idx]
            solvent_descriptor_names = [solvent_descriptor_names[i] for i in selected_descriptor_idx]

            print(f"Final number of descriptors for solvent with target {target_name}: {solvent_descriptors.shape[1]}")

            if not os.path.exists(solvent_output_path):
                os.makedirs(solvent_output_path)

            for i in range(len(solvent_graphs)):
                graph = graphs[i]
                descriptors = solvent_descriptors[i].tolist()
                graph['descriptors'] = descriptors

                output_file = os.path.join(solvent_output_path, f'{solvent_names[i]}.pt')
                torch.save(graph, output_file)

            with open(os.path.join(solvent_output_path, f'descriptor_names.txt'), 'w') as f:
                for name in solvent_descriptor_names:
                    f.write(f"{name}\n")
