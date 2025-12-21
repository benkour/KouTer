# %%
import torch
from torch.optim import Adam
import gpytorch
import os
import pandas as pd
import torch_geometric
from sklearn.metrics import r2_score
import os
import torch
import numpy as np
from torch_geometric.datasets import QM9
from helpers import *
import gpytorch
import json
# with open('parameters.json', 'r') as json_file:
#     config = json.load(json_file)
# %%
# log_flag = config["log_flag"]
# normalize_descriptors = config["normalize_descriptors"]

# from torch.optim import SGD, Adam
# transfomers_flag = config["transfomers_flag"]
# descriptor_flag = config["descriptor_flag"]
# validation_flag = config["validation"]
# solvent_prediction_flag = config["solvent_prediction_flag"]
# divided_by_10_flag = config["divided_by_10_flag"]
# # output_dim_model = config["output_dim"]
# # print("output_dim_model:", output_dim_model)
# save_folder = config["save_folder"]
# dimenet_flag = config.get("dimenet_flag")
# # Set seeds for reproducibility
# nr_of_solvent_predictions = config.get("nr_of_solvent_predictions")
import random

from tqdm import tqdm
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# %%


def train_model(config, descriptor_min, descriptor_max, property_min, property_max,model_dimenet, transformer_model, gp_model, likelihood, train_loader,  test_loader,val_loader,
                epochs,  device, prediction_target, dimenet_target, uncertainty, standardization, custom_line, output_dim_model, run):
    print("run number:", run)
    log_flag = config["log_flag"]
    normalize_descriptors = config["normalize_descriptors"]

    from torch.optim import SGD, Adam
    transfomers_flag = config["transfomers_flag"]
    descriptor_flag = config["descriptor_flag"]
    validation_flag = config["validation"]
    solvent_prediction_flag = config["solvent_prediction_flag"]
    divided_by_10_flag = config["divided_by_10_flag"]
    # output_dim_model = config["output_dim"]
    # print("output_dim_model:", output_dim_model)
    save_folder = config["save_folder"]
    dimenet_flag = config.get("dimenet_flag")
    # Set seeds for reproducibility
    nr_of_solvent_predictions = config.get("nr_of_solvent_predictions")
    if solvent_prediction_flag:
        # graph_data_path = os.getcwd() + '/solvent_graphs/'
        # graph_data_path = os.getcwd() + '/molecular_fingerprints/outv2/ver2_dataset/solvent_'+ str(prediction_target) +'_graphs_out'

        graph_data_path = os.getcwd() + '/data/'+str(prediction_target)+'/' + str(dimenet_target) + '/unknown_graphs/'

        graph_files = [os.path.join(graph_data_path, f) for f in os.listdir(graph_data_path)
                    if f.endswith('.pt')]
        graphs = [torch.load(f, weights_only = False) for f in graph_files]
        # save name in the graph
        for i in range(len(graph_files)):
            graphs[i].name = graph_files[i].split('/')[-1].split('.')[0]

        properties_to_remove = ['dP', 'dH', 'dD',
                                'dn', 'beta', 'property', 'alpha', 'DC']


        for graph in graphs:
            graph = graph
            # Remove the 7th descriptor (index 6)
            # descriptors = graph.descriptors
            # descriptors = np.delete(descriptors, 6)
            # graph.descriptors = torch.tensor(descriptors, dtype=torch.float32)
            if normalize_descriptors:
                d_temp = torch.tensor(min_max_normalize_with_inputs(
                    graph.descriptors, descriptor_min, descriptor_max, standardization))
                d_temp = np.reshape(d_temp, (1,len(d_temp)))
                # d_temp = torch.tensor(d_temp, dtype=torch.float32)
                d_temp = d_temp.reshape(1,d_temp.shape[1])
                graph.descriptors = d_temp
            for prop in properties_to_remove:
                if hasattr(graph, prop):
                    delattr(graph, prop)
                    # pass
                else:
                    pass
                    # graph.__setattr__(prop, torch.tensor([0]))

        prediction_loader = GeoLoader(graphs, batch_size=len(graphs), shuffle=False)
        # prediction_loader = test_loader
        print("Number of solvent graphs loaded for prediction:", len(graphs))
        # prediction_loader = add_dimenet_outputs(prediction_loader, model_dimenet)

    best_val_loss = float('inf')
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    all_parameters = list(transformer_model.parameters())
    if uncertainty:
        all_parameters += list() + \
            list(likelihood.parameters())+list(gp_model.parameters())
    else:
        gp_model = None
        likelihood = None
    optimizer = Adam(all_parameters, lr=0.01)
    # optimizer = SGD(all_parameters, lr=0.01, momentum=0.9, nesterov=True, weight_decay=0)
    print('Starting training', flush=True)
    # for epoch in tqdm(range(epochs), desc='Epochs'):
    for epoch in range(epochs):
        # model_dimenet = model_dimenet.to(device)
        transformer_model = transformer_model.to(device)
        # model_dimenet.train()
        transformer_model.train()
        if uncertainty:

            gp_model = gp_model.to(device)
            likelihood = likelihood.to(device)
            gp_model.train()
            likelihood.train()
            mll.train()

        total_train_loss = 0
        # for data in tqdm(train_loader, desc='Batches', leave=False):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # P = model_dimenet(data.z, data.pos, data.batch)
            P = data.dimenet_embedding
            # desc = torch.tensor(data.descriptors, device=P.device, dtype=torch.float)
            # desc = torch.reshape(desc, (P.shape[0], int(desc.shape[0]/P.shape[0])))


            if descriptor_flag:
                P = torch.cat([P, torch.tensor(data.descriptors, device=P.device, dtype=torch.float)], dim=-1)[:,:output_dim_model]
            # P = torch.cat([P, torch.tensor(data.descriptors, device=P.device, dtype=torch.float)], dim=-1)[:,:output_dim_model]
            if dimenet_flag == False:
                P = torch.tensor(data.descriptors, device=P.device, dtype=torch.float)[:,:output_dim_model]
            # print("P shape in training:", P.shape)
            if transfomers_flag or not uncertainty:
                transformer_output = transformer_model(P).squeeze(-1)
                transformer_output = transformer_output.to(device)
            else:
                transformer_output = P

            # print("shape of transformer_output:", transformer_output.shape)
            if not uncertainty:
                loss = torch.nn.MSELoss()(transformer_output, data.y)
            if uncertainty:
                gp_model.set_train_data(
                    inputs=transformer_output.detach(), targets=data.y.detach(), strict=False)
                gp_prediction = gp_model(transformer_output)
                loss = -mll(gp_prediction, data.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        print(f'Epoch: {epoch}, Train Loss: {total_train_loss/len(train_loader)}', flush=True)
        if validation_flag:
            # print('Validating...', flush=True)
            if epoch % 1 == 0:

                model_dimenet.eval()
                transformer_model.eval()
                if uncertainty:
                    gp_model.eval()
                    likelihood.eval()
                    mll.eval()
                with torch.no_grad():
                    avg_val_loss, _,_,_, _ = test_model(config,
                        val_loader, model_dimenet, transformer_model, 'cpu', gp_model, likelihood, uncertainty,output_dim_model)
                    # print(f'Epoch: {epoch}, Validation Loss: {avg_val_loss}')
                    if avg_val_loss < best_val_loss:
                        counter = 0
                        best_val_loss = avg_val_loss
                        print(f'New best validation loss: {best_val_loss}')
                        best_transformer_model = transformer_model
                        if uncertainty:
                            best_gp_model = gp_model
                            best_likelihood = likelihood
                        else:
                            best_gp_model = None
                            best_likelihood = None
                    else:
                        counter += 1
            if counter > 50:
                    print(f'Early stopping at epoch {epoch} with best validation loss: {best_val_loss}')
                    break
        #     avg_test_loss, truth, mean, std, _ = test_model(
        #         test_loader, model_dimenet, transformer_model, device, gp_model, likelihood, uncertainty,output_dim_model)
        #     print(
        #         f'Epoch: {epoch}, Train Loss: {total_train_loss/len(train_loader)}, Test Loss: {avg_test_loss}')
    if validation_flag:
        gp_model = best_gp_model
        likelihood = best_likelihood
        transformer_model = best_transformer_model
    print('Training complete')
    # print('device of gp_model:', next(gp_model.parameters()).device)
    # print('device of transformer_model:', next(transformer_model.parameters()).device)
    # print('device of model_dimenet:', next(model_dimenet.parameters()).device)
    # print('device of likelihood:', next(likelihood.parameters()).device)
    model_folder = os.path.join(save_folder, 'results_'+prediction_target, 'models')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(transformer_model, os.path.join(
        model_folder, 'transformers_model' + str(dimenet_target) + str(uncertainty) + '_'+str(standardization)+'_'+custom_line+ "run_nr_"+str(run)+  '.pth'))
    if uncertainty:
        torch.save(likelihood, os.path.join(
            model_folder, 'likelihood' + str(dimenet_target) + str(uncertainty) + '_'+str(standardization)+'_'+custom_line+ "run_nr_"+str(run)+ '.pth'))
        torch.save(gp_model, os.path.join(
            model_folder, 'gp_model' + str(dimenet_target) + str(uncertainty) + '_'+str(standardization)+'_'+custom_line+ "run_nr_"+str(run)+ '.pth'))

    if solvent_prediction_flag:
        print("Starting solvent predictions...")
        for i in range(nr_of_solvent_predictions):
            mean_solvents, std_solvents, names_solvents = use_model(config,
                prediction_loader, model_dimenet, transformer_model, 'cpu', gp_model, likelihood, uncertainty,output_dim_model)
            mean_solvents = np.array(mean_solvents)
            print("mean_solvents shape:", mean_solvents.shape)
            std_solvents = np.array(std_solvents)
            mean_solvents_un = min_max_denormalize(
                            mean_solvents, property_min, property_max,  standardization,log_processing=log_flag, div_by_10=divided_by_10_flag)
            # mean_solvents_un = min_max_denormalize(
            #     mean_solvents, property_min, property_max,  standardization, log_processing=log_flag, div_by_10=divided_by_10_flag)
            if uncertainty:
                std_solvents_un = min_max_denormalize(
                    std_solvents, property_min, property_max,  standardization, log_processing=log_flag, div_by_10=divided_by_10_flag)
                df = pd.DataFrame({
                    'mean': mean_solvents_un,
                    'std': std_solvents_un,
                    'name': names_solvents
                })
            else:
                df = pd.DataFrame({
                    'mean': mean_solvents_un,
                    'name': names_solvents
                })
            folder = os.path.join(save_folder, 'results_'+prediction_target, 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            df.to_csv(os.path.join(
                folder, 'results_' + str(dimenet_target) + '_'+str(uncertainty) + '_'+str(standardization) + custom_line + "run_nr_"+str(run)+ '_full_solvents.csv'), index=False)

            print('Results saved for solvents in ', 'results_' + str(dimenet_target) + '_'+str(uncertainty) + '_'+str(standardization) + custom_line + "run_nr_"+str(run)+  '_full_solvents.csv')

    avg_test_loss, truth, mean, std, names = test_model(config,
        test_loader, model_dimenet, transformer_model, 'cpu', gp_model, likelihood, uncertainty,output_dim_model)
    return truth, mean, std, names, transformer_model, gp_model, likelihood


def test_model(config, test_loader, model_dimenet, transformer_model, device, gp_model, likelihood, uncertainty=False,output_dim_model = 7):
    log_flag = config["log_flag"]
    normalize_descriptors = config["normalize_descriptors"]

    from torch.optim import SGD, Adam
    transfomers_flag = config["transfomers_flag"]
    descriptor_flag = config["descriptor_flag"]
    validation_flag = config["validation"]
    solvent_prediction_flag = config["solvent_prediction_flag"]
    divided_by_10_flag = config["divided_by_10_flag"]
    # output_dim_model = config["output_dim"]
    # print("output_dim_model:", output_dim_model)
    save_folder = config["save_folder"]
    dimenet_flag = config.get("dimenet_flag")
    # Set seeds for reproducibility
    nr_of_solvent_predictions = config.get("nr_of_solvent_predictions")
    model_dimenet = model_dimenet.to(device)
    transformer_model = transformer_model.to(device)
    model_dimenet.eval()
    transformer_model.eval()
    if uncertainty:
        gp_model.eval()
        likelihood.eval()
        gp_model = gp_model.to(device)
        likelihood = likelihood.to(device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)



    # print('device of gp_model:', next(gp_model.parameters()).device)
    # print('device of transformer_model:', next(transformer_model.parameters()).device)


    mean = []
    std = []
    truth = []
    names = []
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # P = model_dimenet(data.z, data.pos, data.batch)
            P = data.dimenet_embedding
            desc = torch.tensor(data.descriptors, device=P.device, dtype=torch.float)
            # desc = torch.reshape(desc, (P.shape[0], int(desc.shape[0]/P.shape[0])))

            # print(P.shape, data.descriptors.shape, "shape of P and descriptors")
            if descriptor_flag:
                P = torch.cat([P, torch.tensor(data.descriptors, device=P.device, dtype=torch.float)], dim=-1)[:,:output_dim_model]
            # P = torch.cat([P, torch.tensor(data.descriptors, device=P.device, dtype=torch.float)], dim=-1)[:,:output_dim_model]
            if dimenet_flag == False:
                P = torch.tensor(data.descriptors, device=P.device, dtype=torch.float)[:,:output_dim_model]
                # print("P shape in test_model:", P.shape)
            if transfomers_flag or not uncertainty:
                transformer_output = transformer_model(P).squeeze(-1)
                transformer_output = transformer_output.to(device)
            else:
                transformer_output = P
            # print('device of transformer_output:', transformer_output.device)
            try:
                truth.append(data.y.cpu().numpy())
            except AttributeError:
                # Handle case where data.y is not present
                truth.append(np.zeros_like(transformer_output.cpu().numpy()))

            names.append(data.name)
            if uncertainty:
                # gp_model.set_train_data(
                #     inputs=transformer_output.detach(), targets=data.y.detach(), strict=False)
                #print devices of likelihood and gp_model


                prediction = likelihood(gp_model(transformer_output))
                loss = -mll(prediction, data.y)
                total_test_loss += loss.item()
                mean.append(prediction.mean.cpu().numpy())
                std.append(prediction.stddev.cpu().numpy())
            else:
                prediction = transformer_output
                mean.append(prediction.cpu().numpy())
                loss = torch.nn.MSELoss()(transformer_output, data.y)
                std.append(-torch.ones_like(prediction).cpu().numpy())
                total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    return avg_test_loss, truth, mean, std, names


def use_model(config, test_loader, model_dimenet, transformer_model, device, gp_model, likelihood, uncertainty=False, output_dim_model = 7):
    # print(output_dim_model, "output_dim_model in use_model")
    # print('device in use_model:', device)
    log_flag = config["log_flag"]
    normalize_descriptors = config["normalize_descriptors"]

    from torch.optim import SGD, Adam
    transfomers_flag = config["transfomers_flag"]
    descriptor_flag = config["descriptor_flag"]
    validation_flag = config["validation"]
    solvent_prediction_flag = config["solvent_prediction_flag"]
    divided_by_10_flag = config["divided_by_10_flag"]
    # output_dim_model = config["output_dim"]
    # print("output_dim_model:", output_dim_model)
    save_folder = config["save_folder"]
    dimenet_flag = config.get("dimenet_flag")
    # Set seeds for reproducibility
    nr_of_solvent_predictions = config.get("nr_of_solvent_predictions")

    transformer_model.to(device)
        # print('device of likelihood:', likelihood.device)

    # print('device of transformer_model:', next(transformer_model.parameters()).device)

    transformer_model.eval()
    if uncertainty:
        gp_model.to(device)
        likelihood.to(device)
        gp_model.eval()
        likelihood.eval()
        # print('device of gp_model:', next(gp_model.parameters()).device)
        # print('device of likelihood:', next(likelihood.parameters()).device)


    mean = []
    std = []
    names = []
    with torch.no_grad():
        for data in test_loader:
            print("first")
            data = data.to(device)
            # P = model_dimenet(data.z, data.pos, data.batch)
            P = data.dimenet_embedding
            # gp_model.prediction_strategy = None  # Reset cached strategy
            desc = torch.tensor(data.descriptors, device=P.device, dtype=torch.float)
            # desc = torch.reshape(desc, (P.shape[0], int(desc.shape[0]/P.shape[0])))

            if descriptor_flag:
                P = torch.cat([P, torch.tensor(data.descriptors, device=P.device, dtype=torch.float)], dim=-1)[:,:output_dim_model]
            # P = torch.cat([P, torch.tensor(data.descriptors, device=P.device, dtype=torch.float)], dim=-1)[:,:output_dim_model]
            if dimenet_flag == False:
                P = torch.tensor(data.descriptors, device=P.device, dtype=torch.float)[:,:output_dim_model]
                # print("P shape in use_model:", P.shape)
            if transfomers_flag or not uncertainty:
                transformer_output = transformer_model(P).squeeze(-1)
                transformer_output = transformer_output.to(device)
            else:
                transformer_output = P

            if uncertainty:
                # print('device of transformer_output:', transformer_output.device)
                # print('device of transformer_model:', next(transformer_model.parameters()).device)
                # print('device of gp_model:', next(gp_model.parameters()).device)
                # print('device of likelihood:', next(likelihood.parameters()).device)
                # print('device of diment:', next(model_dimenet.parameters()).device)
                gp_model = gp_model.to(device)
                gp_model.eval()
                likelihood = likelihood.to(device)
                likelihood.eval()
                prediction = likelihood(gp_model(transformer_output))
                print("prediction obtained")
                print("SHAPE OF prediction.mean:", prediction.mean.cpu().numpy().shape)
                # print("device of prediction:", prediction.mean.device)
                mean.append(prediction.mean.cpu().numpy())
                std.append(prediction.stddev.cpu().numpy())
            else:
                # prediction = transformer_output
                prediction = transformer_output
                mean.append(prediction.cpu().numpy())
                std.append(-torch.ones_like(prediction).cpu().numpy())

            names.append(data.name)
    # flatten the lists
    mean = [item for sublist in mean for item in sublist]
    std = [item for sublist in std for item in sublist]
    names = [item for sublist in names for item in sublist]
    return mean, std, names
