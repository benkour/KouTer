from initial_graph_creation import *
from descriptor_creation_pipeline import *
import json
with open('graph_parameters.json', 'r') as json_file:
    config = json.load(json_file)
property = config["property"][0]
file = config["file"]
create_graphs(file, property)
descriptor_creation_pipeline()