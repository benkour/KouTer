import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
with open('parameters.json', 'r') as json_file:
    config = json.load(json_file)
save_folder = config["save_folder"]

def plot_results(truth, mean, uncertainty, dimenet_target, prediction_target, standardization, custom_line):
    results_folder = save_folder + '/results_'+str(prediction_target) +'/results'

    truth = np.array(truth)
    mean = np.array(mean)
    truth = truth.squeeze()
    mean = mean.squeeze()
    mse = mean_squared_error(truth, mean)
    x_min = np.minimum(mean, truth).min() - 0.1
    x_max = np.maximum(mean, truth).max() + 0.1
    x = np.linspace(x_min, x_max, 400).squeeze()
    y = x
    # put mse and r2 in the title
    # make the figure ratio one to one
    plt.figure(figsize=(6, 6))
    r2 = r2_score(truth, mean)
    plt.title(f'MSE: {mse:.4f} R2: {r2:.4f}')
    plt.plot(mean, truth, 'o')
    plt.plot(x, y, linestyle='dotted')
    plt.xlabel('Energy predictions (standardized)')
    plt.ylabel('True Energies (standardized)')
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.savefig(results_folder+'/' +str(dimenet_target) + str(uncertainty) + '_'+str(standardization)+'_'+custom_line+'.pdf')
    plt.show()
    plt.close()


def plot_error_bars(truth, mean, std, property_min, property_max, uncertainty, dimenet_target, prediction_target, normalization_flag=True):

    errors = std
    plt.errorbar(truth, mean, yerr=errors, fmt='o', ecolor='r', capsize=5,
                 linestyle='None', marker='o', markersize=5, label='Predictions with 95% CI')
    plt.plot([min(truth), max(truth)], [min(truth), max(truth)],
             'k--', label='Perfect Predictions')
    plt.legend()
    x = np.linspace(min(truth), max(truth), 100)
    plt.plot(x, x, 'k--')
    plt.title('Predictions vs. Actual Values with 95% Confidence Interval')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Mean')
    plt.savefig('results/'+str(prediction_target) +
                str(dimenet_target) + str(uncertainty)+'.pdf')
    plt.show()


def plot_histogram(mean):
    plt.hist(mean[-1], bins=30, color='skyblue', edgecolor='black')
    plt.show()
