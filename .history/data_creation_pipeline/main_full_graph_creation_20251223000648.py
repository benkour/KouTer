from initial_graph_creation import *
from descriptor_creation_pipeline import *
from add_dimenet_outputs import *
import json
with open('graph_parameters.json', 'r') as json_file:
    config = json.load(json_file)
properties = config["property"]
file = config["file"]
initilal_processing = config["initial_processing"]
descriptor_processing = config["descriptor_processing"]
dimenet_outputs = config["dimenet_outputs"]
for property in properties:
    if initilal_processing:
        create_graphs(file, property)
    if descriptor_processing:
        add_descriptors_to_graphs()
    if dimenet_outputs:
        add_dimenet_outputs()