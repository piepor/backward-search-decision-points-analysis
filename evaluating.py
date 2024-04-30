import os
from itertools import count
import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking
from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_align
from pm4py.objects.petri_net.importer import importer as pnml_importer
from c4dot5.importing import import_classifier

def evaluate_data_petri_net(petri_net: PetriNet, event_log: EventLog, decision_points: dict, initial_marking: Marking, final_marking: Marking):
    # returns also the name of the transitions
    dps_map = get_dps(petri_net)
    possible_attributes = get_possible_attributes(event_log)
    parameters = {pn_align.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligned_traces = pn_align.apply_log(event_log, petri_net, initial_marking, final_marking, parameters=parameters)
    metrics = []
    for i, alignment in enumerate(aligned_traces):
        trace = event_log[i]
        metrics.append(evaluate_trace(alignment, trace, decision_points, dps_map, possible_attributes))
    return sum(metrics) / len(event_log)

def evaluate_trace(alignment: dict, trace: Trace, decision_points: dict, dps_map: dict, possible_attributes: set) -> float:
    transitions = []
    prob_sequence = []
    for align in alignment['alignment']:
        trans = align[0][1]
        # TODO add the case of a transitions pointed by two dps
        if trans in dps_map:
            attributes = get_attributes(transitions, trace, possible_attributes)
            dp = dps_map[trans][0]
            classifier = decision_points[dp]
            _, pred_distribution = classifier.predict(attributes, distribution=True)
            if trans not in pred_distribution[0]:
                breakpoint()
            target_prob = pred_distribution[0][trans]
            prob_sequence.append(np.log(target_prob))
        transitions.append(align[1][1])
    return np.exp(sum(prob_sequence))

def get_possible_attributes(event_log: EventLog):
    attrs = set()
    for trace in event_log:
        for event in trace:
            for attribute in event:
                if not attribute in attrs:
                    attrs.add(attribute)
    return attrs

def get_attributes(trans: list, trace: Trace, possible_attributes: set):
    if not trans:
        return []
    attributes = {key: [None] for key in possible_attributes}
    visible_trans = [transition for transition in trans if not transition is None]
    last_trans = visible_trans[-1]
    count_trans = trans.count(last_trans)
    trace_seq = [event['concept:name'] for event in trace]
    occurrences = [ind for ind, ele in zip(count(), trace_seq) if ele == last_trans]
    if not occurrences:
        return []
    index_trans = occurrences[count_trans-1]
    for i in range(index_trans+1):
        event = trace[i]
        for event_attr in event.keys():
            attributes[event_attr] = [event[event_attr]]
    return pd.DataFrame.from_dict(attributes)
        
def get_dps(petri_net: PetriNet):
    dps = {}
    for place in petri_net.places:
        if len(place.out_arcs) > 1:
            for arc in place.out_arcs:
                trans = arc.target.name
                if not trans in dps.keys():
                    dps[trans] = []
                dps[trans].append(place.name)
    return dps

def validation(net_name: str, models_dir: str, data_path: str):
    log = pm4py.read_xes(data_path)
    net, im, fm = pnml_importer.apply(f'{models_dir}/petri_nets/{net_name}.pnml')
    decision_points_clfs = load_classifiers(net, models_dir)
    res = evaluate_data_petri_net(net, log, decision_points_clfs, im, fm)
    return res

def load_classifiers(net: PetriNet, models_dir:str) -> dict:
    decision_points_classifiers = {}
    for place in net.places:
        file_name = None
        if len(place.out_arcs) > 1:
            for classifier_name in os.listdir(f"{models_dir}/classifiers"):
                if place.name == classifier_name.split('-')[1]:
                    file_name = classifier_name
                    break
            if not file_name is None: 
                decision_points_classifiers[place.name] = import_classifier(f"{models_dir}/classifiers/{file_name}")
            else:
                decision_points_classifiers[place.name] = []
    return decision_points_classifiers

# log = pm4py.read_xes('logs/log-Road_Traffic_Fine_Management_Process.xes')
# net, im, fm = pnml_importer.apply('models/Road_Traffic_Fine_Management_Process.pnml')
# evaluate_data_petri_net(net, log, {}, im, fm)
net_name = 'Road_Traffic_Fine_Management_Process'
models_dir = './models'
data_path = f'./logs/log-{net_name}.xes'
res = validation(net_name, models_dir, data_path)
breakpoint()
print(f"Total likelihood: {res}")
