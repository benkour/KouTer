# %%
import warnings
import pandas as pd
import torch_geometric
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
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
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
BATCHSIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# device = torch.device("cpu")models.py
print(f"Device: {device}")
# Load and prepare data
path = os.path.join(os.getcwd(), 'data', 'QM9')
dataset = QM9(path)
print('Loading model weights...')
with open('parameters.json', 'r') as json_file:
    config = json.load(json_file)
log_flag = config["log_flag"]
output_dim_model = config["output_dim"]
save_folder = config["save_folder"]
validation_flag = config["validation"]
transformer_flag = config["transfomers_flag"]
normalize_descriptors = config["normalize_descriptors"]
transformer_heads = config["transformer_heads"]
nr_of_mixtures = config["nr_of_mixtures"]
EPOCHS = config["EPOCHS"]
current_property = config["prediction_target"]
mean_module_list = config["mean_module"]
dimenet_target_list = config["dimenet_target"]
nr_of_runs = config.get("nr_of_runs", 1)
uncertainty_list = config.get("uncertainty")
standardization_list = config.get("standardization")
divided_by_10_flag = config.get("divided_by_10_flag", False)
dimenet_flag = config.get("dimenet_flag")
# suppress warnings
warnings.filterwarnings("ignore")
# prediction_target = 'dP'
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
# uncertainty = True
# standardization = True
# for prediction_target in ['DC', 'dP', 'dH', 'dD', 'beta', 'dn']:
# for prediction_target in ['beta', 'dn', 'DC']:
# for prediction_target in ['dP', 'dH', 'dD','beta', 'dn', 'DC']:
# for prediction_target in ['dP', 'dH', 'dD']:
# for prediction_target in ['DC']:
# fast dataset dn and beta
# slow datasets the rest
def mean_module_name(mean_module, output_dim_model):
    if mean_module == "ConstantMean":
        return ConstantMean(input_size=output_dim_model)
    if mean_module == "LinearMean":
        return LinearMean(input_size=output_dim_model)
    if mean_module == "ZeroMean":
        return ZeroMean()
    else:
        return "UnknownMean"

print("Mean module list:", mean_module_list)
for mean_module_str in mean_module_list:
    print("Using mean module:", mean_module_str)
    for prediction_target in current_property:

        results_folder = save_folder + '/results_'+str(prediction_target) +'/results'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        for standardization in standardization_list:
            for uncertainty in uncertainty_list:
                # custom_line = prediction_target +"5_molecular_embeddings_validation_CONSTANT_scaled_gmx_4_feed_forward_24_head_12"
                # custom_line = prediction_target + str(mean_module).split('.')[-1].split("'")[0] + 'scaled_gmx_4_feed_forward_24_head_5_3_layers_7_output_NO_TRANSFORMERS_NODESCRIPTORS_VALIDATION_50_early_stopping_paper_data'
                custom_line = prediction_target + str(mean_module_str).split('.')[-1].split("'")[0] + 'scaled_RBF_kernel_'
                # custom_line = "test"

                if descriptor_flag:
                    custom_line = custom_line + '_with_descriptors'
                if log_flag:
                    custom_line = 'log_processed_' + custom_line
                if divided_by_10_flag:
                    custom_line = 'divided_by_10_' + custom_line

                if transfomers_flag:
                    custom_line = 'with_transformers_heads_' +str(transformer_heads)+ custom_line

                else:
                    custom_line = 'no_transformers_' + custom_line
                if validation_flag:
                    custom_line = custom_line + '_with_validation'
                else:
                    custom_line = custom_line + '_no_validation'
                if descriptor_flag:
                    if normalize_descriptors:
                        custom_line = custom_line + '_normalized_descriptors'
                    else:
                        custom_line = custom_line + 'descriptors'
                if not dimenet_flag:
                    custom_line = custom_line + '_no_dimenet'
                if standardization:
                    custom_line = custom_line + '_standardized'
                else:
                    custom_line = custom_line + 'normalized'

                custom_line = custom_line + 'EPOCHS_' + str(EPOCHS)
                for dimenet_target in dimenet_target_list:

                    # graph_data_path = os.getcwd() + '/molecular_fingerprints/outv2/ver2_dataset/'+str(prediction_target)+'_graphs_out/'
                    graph_data_path = os.getcwd() + '/data/'+str(prediction_target)+'/' + str(dimenet_target) + '/known_graphs/'

                    graph_files = [os.path.join(graph_data_path, f) for f in os.listdir(graph_data_path)
                                if f.endswith('.pt')]
                    graphs = [torch.load(f) for f in graph_files]
                    # graphs = graphs[:1000]
                    temp_property = getattr(graphs[0], prediction_target)

                    # print((temp_property.numpy()))
                    # print("Number of graphs:", len(graphs), "nr of files:", len(graph_files))
                    # print(graphs[0])
                    if config["keep_all_descriptors"]:
                        print("Keeping all descriptors")
                        print(len(graphs[0].descriptors))
                        output_dim_model = 7 + len(graphs[0].descriptors)
                        if dimenet_flag == False:
                            output_dim_model = output_dim_model -7
                    if not config["descriptor_flag"]:
                        print("Not using descriptors")
                        output_dim_model = 7
                        # mean_module.input_size = output_dim_model
                    if transformer_heads == -1:
                        transformer_heads = output_dim_model
                    # print("mean module:", mean_module_str)
                    mean_module = mean_module_name(mean_module_str, output_dim_model)
                    # print("Using mean module:", mean_module)
                    # print("output_dim_model:", output_dim_model)
                    property = []
                    descriptors = []
                    to_delete = []
                    for i in range(len(graphs)):
                        temp_property = getattr(graphs[i], prediction_target)
                        if temp_property <= 0 or temp_property > 100 :
                            # remove the graph
                            to_delete.append(i)
                            continue
                        property.append(temp_property.cpu().numpy())
                        # Remove the 7th descriptor (index 6)
                        desc = graphs[i].descriptors
                        # desc = np.delete(desc, 6)
                        # graphs[i].descriptors = desc
                        descriptors.append(desc)
                        # print("appended", temp_property.numpy()[0], i)
                    # print(len(property))
                    property = np.array(property)
                    # print("Property shape:", property.shape)
                    # print("max property:", property.max() , "min property:", property.min() )
                    # print(property.numpy().max(),property.numpy().min())


                    descriptors = np.array(descriptors)
                    descriptors = np.reshape(descriptors, (len(graphs)-len(to_delete), len(desc)))
                    property_min, property_max, property_stand = min_max_normalize(
                        property, standardization, log_processing=log_flag, div_by_10=divided_by_10_flag)
                    descriptors_min, descriptors_max, descriptors_stand = min_max_normalize(
                        descriptors, standardization)

                    # print("Property min:", descriptors_min)
                    # print("Property max:", descriptors_max)
                    # print(len(property))
                    # save the property min and max in a csv
                    df = pd.DataFrame({
                        'property_min': [property_min],
                        'property_max': [property_max],
                        'descriptors_min': [descriptors_min.tolist()],
                        'descriptors_max': [descriptors_max.tolist()]
                    })
                    df.to_csv(results_folder + '/property_min_max_standardization'+ '_'+str(standardization)+'.csv', index=False)
                    # delete the graphs with 0 property
                    # reverse to_delete
                    to_delete = to_delete[::-1]
                    for i in to_delete:
                        del graphs[i]
                    for i in range(len(graphs)):
                        graphs[i].y = torch.tensor(property_stand[i])

                        if normalize_descriptors:
                            d_temp = descriptors_stand[i]
                            d_temp = np.reshape(d_temp, (1,len(d_temp)))
                            d_temp = torch.tensor(d_temp, dtype=torch.float32)
                            d_temp = d_temp.reshape(1,d_temp.shape[1])
                            graphs[i].descriptors = d_temp

                    print('dimenet_target:', dimenet_target, 'dimenet_target_list:', dimenet_target_list)

                    total_truth = []
                    total_mean = []
                    total_std = []
                    total_name = []
                    flag_valid_run = False
                    for run in range(nr_of_runs):
                        random.shuffle(graphs)
                        train_data, test_data = train_test_split(
                            graphs, test_size=0.1)
                        if validation_flag:
                            train_data, val_data = train_test_split(
                                train_data, test_size=0.1)
                        else:
                            val_data = train_data
                        # print("Train size:", len(train_data),
                            # "Validation size:", len(val_data), "Test size:", len(test_data), "for property:", prediction_target)
                        train_loader = GeoLoader(
                            train_data, batch_size=200, shuffle=True)
                        val_loader = GeoLoader(val_data, batch_size=100, shuffle=False)
                        test_loader = GeoLoader(
                            test_data, batch_size=len(test_data), shuffle=False)
                        print("Run:", run+1, "of", nr_of_runs)
                        print('Loading model for target:', dimenet_target,
                            "to predict", prediction_target, "with uncertainty:", uncertainty, "and standardization:", standardization)

                        pred = []
                        labels = []

                        # for param in model_dimenet.parameters():
                        #     param.requires_grad = False
                        if uncertainty:
                            transformer_model = SimpleTransformerModel(
                                input_dim=output_dim_model, num_heads=transformer_heads, dim_feedforward=24, output_dim=output_dim_model).to(device)
                        else:
                            transformer_model = SimpleTransformerModel(
                                input_dim=output_dim_model, num_heads=transformer_heads, dim_feedforward=24, output_dim=1).to(device)
                        likelihood = gpytorch.likelihoods.GaussianLikelihood()
                        gp_model = SimpleGPModel(train_x=None, train_y=None,
                                                likelihood=likelihood, mean_module = mean_module,  output_dim_model=output_dim_model).to(device)
                        #put the models to device
                        # model_dimenet = model_dimenet.to(device)
                        likelihood = likelihood.to(device)
                        gp_model = gp_model.to(device)
                        if transformer_flag == True:
                            transformer_model = transformer_model.to(device)
                        # # Training
                        # print("Property min:", property_min)
                        # print("Property max:", property_max)
                        # truth, mean, std, name, transformer_model, gp_model, likelihood = train_model(model_dimenet, transformer_model, gp_model, likelihood, train_loader,val_loader,  test_loader,
                        #                                                                             EPOCHS,  device, prediction_target, dimenet_target, uncertainty, standardization, custom_line, property_min, property_max)
                        # print(np.array(truth).shape)
                        # print("Truth:")

                            # Attach per-graph DimeNet embeddings to the graphs in the train_loader dataset
                        # train_loader = add_dimenet_outputs(train_loader, model_dimenet)
                        # val_loader = add_dimenet_outputs(val_loader, model_dimenet)
                        # test_loader = add_dimenet_outputs(test_loader, model_dimenet)
                        if gp_model is not None:
                            model_dimenet = gp_model
                        elif transformer_model is not None:
                            model_dimenet = transformer_model
                        try:
                            truth, mean, std, name, transformer_model, gp_model, likelihood = train_model(config,descriptors_min, descriptors_max, property_min, property_max,model_dimenet, transformer_model, gp_model, likelihood, train_loader,  test_loader,val_loader,
                                                                                        EPOCHS,  device, prediction_target, dimenet_target, uncertainty, standardization, custom_line, output_dim_model, run)
                            flag_valid_run = True
                        except Exception as e:
                            tb = traceback.extract_tb(e.__traceback__)
                            print(f"Error in file {tb[-1].filename}, line {tb[-1].lineno}: {e}")
                            if str(e) == "Matrix not positive definite after repeatedly adding jitter up to 1.0e-04.":
                                print("Skipping this configuration due to numerical issues.")
                                continue
                            else:
                                import sys

                                sys.exit(1)

                            # print('error:', e)
                            # # Skip to the next configuration if early stopping is triggered

                            # if str(e) == "Early stopping":
                            #     print("Early stopping triggered. Moving to next configuration.")
                            #     continue
                            # else:
                            #     raise e
                        print("exited the loop")
                        truth = np.concatenate([np.array(t).flatten() for t in truth])
                        mean = np.concatenate([np.array(m).flatten() for m in mean])
                        truth = truth.squeeze()
                        mean = mean.squeeze()
                        name = np.concatenate([np.array(n).flatten() for n in name])
                        name = name.squeeze()
                        # print("Truth shape:", truth.shape)
                        # print("Mean shape:", mean.shape)
                        if uncertainty:
                            std = np.concatenate([np.array(s).flatten() for s in std])
                            # print("Std shape:", std.shape)
                            std = std.squeeze()


                        truth_un = min_max_denormalize(
                            truth, property_min, property_max,  standardization,log_processing=log_flag, div_by_10=divided_by_10_flag)
                        mean_un = min_max_denormalize(
                            mean, property_min, property_max,  standardization,log_processing=log_flag, div_by_10=divided_by_10_flag)
                        if uncertainty:
                            std = np.array(std).squeeze()
                            std_un = min_max_denormalize(
                                property_min, property_max, std, standardization,log_processing=log_flag, div_by_10=divided_by_10_flag)
                            std_un = std_un.squeeze().tolist()

                        truth = np.concatenate([np.array(t).flatten() for t in truth])
                        mean = np.concatenate([np.array(m).flatten() for m in mean])
                        truth = truth.squeeze()
                        mean = mean.squeeze()
                        name = np.concatenate([np.array(n).flatten() for n in name])
                        name = name.squeeze()
                        # print("Truth shape:", truth.shape)
                        # print("Mean shape:", mean.shape)
                        if uncertainty:
                            std_un = np.concatenate([np.array(s).flatten() for s in std_un])
                            # print("Std shape:", std.shape)
                            std_un = std_un.squeeze()

                        total_truth.append(truth_un)
                        total_mean.append(mean_un)
                        total_name.append(name)
                        if uncertainty:
                            std_un = np.array(std_un).squeeze()
                            std_un = std_un.squeeze().tolist()
                            total_std.append(std_un)
                    if not flag_valid_run:
                            continue
                    total_truth = np.concatenate([np.array(t).flatten() for t in total_truth]).squeeze()
                    total_mean = np.concatenate([np.array(m).flatten() for m in total_mean]).squeeze()
                    total_name = np.concatenate([np.array(n).flatten() for n in total_name]).squeeze()
                    # print("Total truth shape:", total_truth.shape)
                    df = pd.DataFrame({'truth': total_truth, 'mean': total_mean, 'name': total_name})
                    if uncertainty:
                        total_std = np.concatenate([np.array(s).flatten() for s in total_std]).squeeze()
                        df['std'] = total_std

                    df.to_csv(results_folder + '/results_' + str(dimenet_target) + str(uncertainty) + '_'+str(standardization)+'_'+custom_line+'.csv', index=False)
                    # Plot results

                    plot_results(total_truth, total_mean, uncertainty,
                                dimenet_target, prediction_target, standardization, custom_line)
                    if not uncertainty:
                        break
                # %%
                # %%

    # %%
