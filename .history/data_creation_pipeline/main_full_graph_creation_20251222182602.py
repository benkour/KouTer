from initial_graph_creation import *
from descriptor_creation_pipeline import *
import json
with open('graph_parameters.json', 'r') as json_file:
    config = json.load(json_file)
property = config["property"][0]
file = config["file"]
initilal_processing = config["initial_processing"]
descriptor_processing = config["descriptor_processing"]
add_dimenet_outputs = config["add_dimenet_outputs"]
if initilal_processing:
    create_graphs(file, property)
if descriptor_processing:
    add_descriptors_to_graphs()

add_dimenet_outputs()