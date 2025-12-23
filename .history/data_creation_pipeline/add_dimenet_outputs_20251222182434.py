# %%
import warnings
import pandas as pd
import torch_geometric
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
import glob
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
import json
import torch
import numpy as np
import traceback
from models import SimpleTransformerModel, SimpleGPModel
# from train import train_model
# from plotting import plot_results, plot_error_bars, plot_histogram
from torch_geometric.datasets import QM9
from helpers import *
import gpytorch
from custom_dimenet import DimeNet
from training import *
from plotting import *
# Set seeds for reproducibility
import random

BATCHSIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# device = torch.device("cpu")models.py
print(f"Device: {device}")
# Load and prepare data
path = os.path.join(os.getcwd(), 'data', 'QM9')
dataset = QM9(path)
print('Loading model weights...')
with open('graph_parameters.json', 'r') as json_file:
    config = json.load(json_file)
current_property = config["prediction_target"]
dimenet_target_list = config["dimenet_target"]
# suppress warnings
warnings.filterwarnings("ignore")
# prediction_target = 'dP'


for prediction_target in current_property:
            graph_data_path = os.getcwd() + 'graph_out/'+str(prediction_target)
            save_folder_graphs = "graphs_with_dimenet_embeddings/"+str(prediction_target)+'/'
            graph_files = sorted(glob.glob(os.path.join(graph_data_path, '*.pt')))
            graphs = [torch.load(f) for f in graph_files]

            for i, f in enumerate(graph_files):
                try:
                    graphs[i].name = os.path.splitext(os.path.basename(f))[0]
                except Exception:
                    pass
            for dimenet_target in dimenet_target_list:
                print("Processing prediction target:", prediction_target, "with DimeNet target:", dimenet_target)
                graph_data_path = os.getcwd() + '/molecular_fingerprints/out_ver5/'+str(prediction_target)+'_graphs_out/'
                graph_files = sorted(glob.glob(os.path.join(graph_data_path, '*.pt')))
                graphs = [torch.load(f) for f in graph_files]
                for i, f in enumerate(graph_files):
                    try:
                        graphs[i].name = os.path.splitext(os.path.basename(f))[0]
                    except Exception:
                        pass
                print("Generating DimeNet embeddings for target:", dimenet_target)
                model_dimenet, datasets = DimeNet.from_qm9_pretrained(path, dataset, dimenet_target)
                loader = GeoLoader(graphs, batch_size=1, shuffle=False)
                save_specific_folder = save_folder_graphs + '/'+str(dimenet_target)+'/known_graphs/'
                if not os.path.exists(save_specific_folder):
                    os.makedirs(save_specific_folder)
                model_dimenet = model_dimenet.to(device)
                model_dimenet.eval()
                with torch.no_grad():
                    for idx, graph in enumerate(loader.dataset):
                        graph = graph.to(device)
                        # build a batch index for a single graph
                        num_nodes = graph.num_nodes if hasattr(graph, "num_nodes") else graph.z.size(0)
                        batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
                        P_graph = model_dimenet(graph.z, graph.pos, batch_idx)
                        graph.dimenet_embedding = P_graph.cpu()
                        graph = graph.to('cpu')
                        torch.save(graph, save_specific_folder + graph.name + '.pt')


                graph_data_path = os.getcwd() + '/molecular_fingerprints/out_ver5/solvent_'+ str(prediction_target) +'_graphs_out'

                graph_files = sorted(glob.glob(os.path.join(graph_data_path, '*.pt')))
                graphs = [torch.load(f) for f in graph_files]

                for i, f in enumerate(graph_files):
                    try:
                        graphs[i].name = os.path.splitext(os.path.basename(f))[0]
                    except Exception:
                        pass
                # save name in the graph
                for i in range(len(graph_files)):
                    graphs[i].name = graph_files[i].split('/')[-1].split('.')[0]

                properties_to_remove = ['dP', 'dH', 'dD',
                                        'dn', 'beta', 'property', 'alpha', 'DC']


                for graph in graphs:
                    graph = graph
                    for prop in properties_to_remove:
                        if hasattr(graph, prop):
                            delattr(graph, prop)
                            # pass
                        else:
                            pass
                            # graph.__setattr__(prop, torch.tensor([0]))

                prediction_loader = GeoLoader(graphs, batch_size=len(graphs), shuffle=False)
                # prediction_loader = add_dimenet_outputs(prediction_loader, model_dimenet)
                with torch.no_grad():
                    for idx, graph in enumerate(prediction_loader.dataset):
                        print(idx)
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
                        save_specific_folder = save_folder_graphs + '/'+str(dimenet_target)+'/unknown_graphs/'
                        if not os.path.exists(save_specific_folder):
                            os.makedirs(save_specific_folder)
                        graph = graph.to('cpu')
                        torch.save(graph, save_specific_folder + graph.name + '.pt')
                        # print("Saved graph with DimeNet target", dimenet_target, ":", graph.name)
                        # print(idx)
