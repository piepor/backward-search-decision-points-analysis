import os
import json
import pickle
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.importer import importer as pnml_importer
from c4dot5.DecisionTree import DecisionTree

    
def import_attributes(net_name: str) -> dict:
    attributes_map_file = f"{net_name}.attr"
    if attributes_map_file in os.listdir('dt-attributes'):
        with open(os.path.join('dt-attributes', attributes_map_file), 'r') as f:
            json_string = json.load(f)
            attributes_map = json.loads(json_string)
    else:
        raise FileNotFoundError('Create a configuration file for the decision tree before fitting it.')
    return attributes_map

def convert_attributes(attributes_map: dict, event_log: EventLog) -> EventLog:
    """ Convert event attributes in the Event Log in agreement with the attributes map """
    for trace in event_log:
        for event in trace:
            for attribute in event.keys():
                if attribute in attributes_map:
                    if attributes_map[attribute] == 'continuous':
                        event[attribute] = float(event[attribute])
                    elif attributes_map[attribute] == 'boolean':
                        event[attribute] = bool(event[attribute])
    return event_log

def import_petri_net(net_name: str) -> tuple[PetriNet, Marking, Marking]:
    try:
        net, im, fm = pnml_importer.apply("models/petri_nets/{}.pnml".format(net_name))
    except:
        raise FileNotFoundError("Model not found in ./models")
    return net, im, fm

def import_decision_tree(tree_name: str) -> DecisionTree:
    with open(f"decision-trees/{tree_name}.dt", 'rb') as file:
        decision_tree = pickle.load(file)
    return decision_tree

