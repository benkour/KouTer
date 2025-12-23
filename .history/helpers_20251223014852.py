import os
import re
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Data, Batch
import random
from torch_geometric.data import DataLoader as GeoLoader
from add_dimenet_outputs import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_dimenet_outputs(loader, model_dimenet):
        model_dimenet.eval()
        with torch.no_grad():
            for idx, graph in enumerate(loader.dataset):
                graph = graph.to(device)
                # build a batch index for a single graph
                num_nodes = graph.num_nodes if hasattr(graph, "num_nodes") else graph.z.size(0)
                batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
                P_graph = model_dimenet(graph.z, graph.pos, batch_idx)
                # If model returns a shape (1, D) for a single graph, squeeze the batch dim
                # if P_graph.dim() > 1 and P_graph.size(0) == 1:
                    # P_graph = P_graph.squeeze(0)
                # store embedding on the graph (move to CPU to avoid device issues later)
                graph.dimenet_embedding = P_graph.cpu()
                # print(idx)

        return loader

def min_max_normalize(x, stand=False, log_processing=False, div_by_10=False):
    x = np.asarray(x)
    if log_processing:
        x = np.log10(x)
    if div_by_10:
        x = x / 10

    if x.ndim == 1:
        if stand:
            std, mean = np.std(x), np.mean(x)
            norm = (x - std) / mean
            return std, mean, norm
        else:
            min_val, max_val = np.min(x), np.max(x)
            norm = (x - min_val) / (max_val - min_val)
            return min_val, max_val, norm
    else:
        if stand:
            std = np.std(x, axis=0)
            mean = np.mean(x, axis=0)
            norm = (x - std) / mean
            return std, mean, norm
        else:
            min_val = np.min(x, axis=0)
            max_val = np.max(x, axis=0)
            norm = (x - min_val) / (max_val - min_val)
            return min_val, max_val, norm

def min_max_normalize_with_inputs(x, min_val, max_val, stand=False):
    x = np.asarray(x)
    min_val = np.asarray(min_val)
    max_val = np.asarray(max_val)
    if stand:
        norm = (x - min_val) / max_val
    else:
        norm = (x - min_val) / (max_val - min_val)

    return norm

def min_max_denormalize(x, min_val, max_val, stand=False, log_processing=False, div_by_10=False):
    x = np.asarray(x)

    min_val = np.asarray(min_val)
    max_val = np.asarray(max_val)
    if stand:
        denorm = x * max_val + min_val
    else:
        denorm = x * (max_val - min_val) + min_val
    if log_processing:
        denorm = 10**denorm
    if div_by_10:
        denorm = denorm * 10
    return denorm

# def min_max_normalize(x, stand=False):
#     x = np.log10(x)
#     if stand:
#         return np.std(x), np.mean(x), (x - np.std(x)) / np.mean(x)
#     else:
#         return min(x), max(x), (x - min(x)) / (max(x) - min(x))


# def min_max_denormalize(x, min_val, max_val, stand=False):
#     if stand:
#         return 10**(x * max_val + min_val)
#     else:
#         return 10**(x * (max_val - min_val) + min_val)



def get_graph(atoms, pos, target=None, distance=8, domain=None):
    '''
    Get a Data graph from the numpy coordinates, the type of atom and the target.
    '''
    # edge index
    a = np.arange(len(atoms))
    edges = np.array(np.meshgrid(a, a)).T.reshape(-1, 2).T
    edges = torch.tensor(edges, dtype=torch.int64)

    atom_to_num = {'C': 6, 'O': 8, 'Zn': 30,
                   'Pt': 78, 'Ni': 28}  # atom to atomic number
    atom_to_en = {'C': 2.55, 'O': 3.44, 'Zn': 1.65,
                  'Pt': 2.28, 'Ni': 1.91}  # atom to electronegativity
    atom_to_r = {'C': 70, 'O': 60, 'Zn': 135, 'Pt': 135, 'Ni': 135}
    # atom_to_num = {'C': 0, 'O':0.02777, 'Zn':0.3333, 'Pt':1} # atom to atomic number
    # atom_to_en = {'C': 0.5027932896, 'O':1, 'Zn':0, 'Pt':0.351955307} # atom to electronegativity
    # atom_to_r = {'C': 0.1333, 'O':0, 'Zn':1, 'Pt':1} # atom to radius

    atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[
        :, np.newaxis]  # keep as numpy for later use
    z = torch.tensor(np.asarray([atom_to_num[atom] for atom in atoms]))
    electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[
                              :, np.newaxis], dtype=torch.float)
    atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[
                                 :, np.newaxis], dtype=torch.float)

    # In the loop we extract the nodes' embeddings, edges connectivity
    # and label for a graph, process the information and put it in a Data
    # object, then we add the object to a list

    # Node features
    # atomic number abd electronegativity # TODO: add atomic radius

    # Edge features
    # shape [N', D'] N': number of edges, D': number of edge features
    # cm matrix and bond matrix
    pair_dist = pairwise_distances(pos)
    cm = (atomic_nums*atomic_nums.T) / pair_dist
    np.fill_diagonal(cm, 0.5*atomic_nums**2.4)
    edge_1 = []
    edge_2 = []
    cm_sparse = []
    dist_sparse = []
    for i in range(pair_dist.shape[0]):
        for j in range(pair_dist.shape[1]):
            if pair_dist[i, j] < 6:
                edge_1.append(i)
                edge_2.append(j)
                cm_sparse.append(cm[i, j])
                dist_sparse.append(pair_dist[i, j])
    cm_sparse = np.array(cm_sparse)
    dist_sparse = np.array(dist_sparse)
    cm_sparse = torch.tensor(
        cm_sparse.flatten()[:, np.newaxis], dtype=torch.float)
    dist_sparse = torch.tensor(dist_sparse.flatten()[
                               :, np.newaxis], dtype=torch.float)
    cm = cm.flatten()[:, np.newaxis]
    # edge_attr = torch.tensor(cm, dtype=torch.float)
    # print(cm.shape, pair_dist.shape)
    edge_attr = torch.cat([torch.tensor(cm, dtype=torch.float), torch.tensor(
        pair_dist.flatten()[:, np.newaxis], dtype=torch.float)], dim=1)
    # edge_attr = torch.cat([cm_sparse, dist_sparse], dim = 1)
    # edges = torch.tensor(np.array([edge_1, edge_2]), dtype=torch.int64)
    if target:
        target = torch.tensor(target, dtype=torch.float)
    # belonging = torch.zeros(pos.shape[0],1)
    # belonging[:-2] = 1
    # belonging[:-2,0] = 1
    # belonging[-2:,1] = 1

    node_attrs = torch.cat(
        [torch.tensor(atomic_nums, dtype=torch.float), electroneg, atomic_radius], dim=1)
    distances_co = pair_dist[-1:, :]
    nearby_nodes = np.argwhere(distances_co < distance)[:, 1]
    nearby_nodes = np.array(list(set(nearby_nodes)))
    mask = torch.zeros((pair_dist.shape[0]), dtype=torch.bool)
    mask[nearby_nodes] = 1
    pos = torch.tensor(pos, dtype=torch.float)

    graph = Data(x=node_attrs,
                 z=z,
                 pos=pos,
                 edge_index=edges,
                 edge_attr=edge_attr,
                 y=target,
                 mask=mask,
                 domain=domain)

    return graph


def save_points(name, positions, cluster_origin, nr):
    with open(name+'.txt', "w") as file:
        file.write(f" {positions.shape[0]+len(cluster_origin)}\n")
        file.write(f"Simulation {nr}\n")
        for atom, coordinate in cluster_origin:
            line = f"{atom} {coordinate[0]} {coordinate[1]} {coordinate[2]}\n"
            file.write(line)
        file.write("Zn\t")
        np.savetxt(file, positions, fmt='%.7f')
        file.write("\n######################################\n")


def save_atoms(root_path, txt_name, atoms):
    file_path = os.path.join(root_path, txt_name)
    if not os.path.isfile(file_path):
        np.savetxt(file_path, atoms, fmt='%s')
        print(f'File {txt_name} saved.')
    else:
        print(f'File {txt_name} exists')


def array_reshape(a):
    a = np.array(a)
    a = np.reshape(a, (a.shape[0]*a.shape[1], 1))
    return a


def data_preparation(folder='/data_processed/', device=device, types_to_include=['Pt3', 'Pt6', 'Pt9']):
    data_list = []
    data_dict = {t: {} for t in types_to_include}
    root_path = os.getcwd() + folder
    filenames = os.listdir(root_path)
    filenames.sort()
    pattern = re.compile(r'(.*)__(\d+)_data_(-?\d+)\.pt')

    for filename in filenames:
        if filename.endswith('.pt'):
            match = pattern.match(filename)
            if match:
                type_key, file_index, data_index = match.groups()
                type_key = type_key + '_'
                print('type_key', type_key)
                file_index = int(file_index)
                data_index = int(data_index)

                if type_key in types_to_include:
                    full_path = os.path.join(root_path, filename)
                    graph = torch.load(full_path)
                    graph.x = graph.x.to(device)
                    graph.edge_attr = graph.edge_attr.to(device)
                    graph.edge_index = graph.edge_index.to(device)
                    graph.y = torch.tensor(
                        graph.y, dtype=torch.float).to(device)
                    data_list.append(graph)

                    if file_index not in data_dict[type_key]:
                        data_dict[type_key][file_index] = {}
                    if data_index not in data_dict[type_key][file_index]:
                        data_dict[type_key][file_index][data_index] = []
                    data_dict[type_key][file_index][data_index].append(graph)

    return data_list, data_dict


def split_train_test(data_dict, train_ratio=0.8):
    file_index_list = []
    for type_key in data_dict:
        file_index_list.extend([(type_key, file_index)
                               for file_index in data_dict[type_key]])

    random.shuffle(file_index_list)

    train_file_count = int(len(file_index_list) * train_ratio)

    train_file_indices = file_index_list[:train_file_count]
    test_file_indices = file_index_list[train_file_count:]

    train_data = []
    for type_key, file_index in train_file_indices:
        for data_idx in [0, 3, -1]:
            train_data.extend(data_dict[type_key][file_index][data_idx])

    test_data = []
    for type_key, file_index in test_file_indices:
        for data_idx in [0, 3, -1]:
            test_data.extend(data_dict[type_key][file_index][data_idx])

    train_loader = GeoLoader(
        train_data, batch_size=len(train_data), shuffle=True)
    test_loader = GeoLoader(
        test_data, batch_size=len(test_data), shuffle=False)

    return train_loader, test_loader
