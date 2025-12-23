from initial_graph_creation import *
from descriptor_creation_pipeline import *
from add_dimenet_outputs import *
import json
with open('graph_parameters.json', 'r') as json_file:
    config = json.load(json_file)
properties = config["property"]
copy_data_to_model_input_dir = config["copy_data_to_model_input_dir"]
files = config["file"]
initilal_processing = config["initial_processing"]
descriptor_processing = config["descriptor_processing"]
dimenet_outputs = config["dimenet_outputs"]
for i in range(len(properties)):
    property = properties[i]
    file = files[i]
    if initilal_processing:
        create_graphs(file, property)
    if descriptor_processing:
        add_descriptors_to_graphs(property)
    if dimenet_outputs:
        add_dimenet_outputs()

if copy_data_to_model_input_dir:
    #model_input directory is one level above and called data
    model_input_graph_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
    #copy contents of dimen