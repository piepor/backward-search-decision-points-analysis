import pm4py
import copy
import numpy as np
import datetime
import argparse
from tqdm import tqdm
from random import choice
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking
from pm4py.objects.log import obj as log_instance
from pm4py.util import xes_constants
from random import choice

# Argument (verbose and net_name)
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="show more info", default=False, type=bool)
parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str)

# parse arguments
args = parser.parse_args()
verbose = args.verbose
net_name = args.net_name
verboseprint = print if verbose else lambda *args, **kwargs: None
try:
    net, initial_marking, final_marking = pnml_importer.apply("models/{}.pnml".format(net_name))
except:
    raise Exception("File not found")

def get_ex_cont(model_name):
    if model_name == 'one-split-PetriNet':
        a = np.random.uniform(0, 10, 1)
        ex_cont = {"A": a[0]}
    elif model_name == 'one-split-PetriNet-categorical':
        choose_var_ex = np.random.uniform(0, 10, 1)
        if choose_var_ex[0] < 3.33:
            a = np.random.uniform(0, 10, 1)
            ex_cont = {"A": a[0], "cat": "None"}
        elif choose_var_ex[0] > 6.66:
            cat = np.random.uniform(0, 10, 1)
            if cat[0] > 5:
                ex_cont = {'A': -1, 'cat': "cat_1"}
            else:
                ex_cont = {'A': -1, 'cat': "cat_2"}
        else:
            a = np.random.uniform(0, 10, 1)
            if a[0] >= 5:
                ex_cont = {"A": a[0], "cat": "cat_1"}
            else:
                ex_cont = {"A": a[0], "cat": "cat_2"}
    elif model_name == 'running-example-Will-BPM':
        a = np.random.uniform(0, 1000, 1)
        status = choice(["approved", "rejected"])
        policy_type = choice(["normal", "premium"])
        ex_cont = {"amount": a[0], "policyType": policy_type, "status": status}
    elif model_name == 'running-example-Will-BPM-silent' or model_name == 'running-example-Will-BPM-silent-trace-attr':
        a = np.random.uniform(0, 1000, 1)
        status = choice(["approved", "rejected"])
        policy_type = choice(["normal", "premium"])
        communication = choice(["email", "letter"])
        ex_cont = {"amount": a[0], "policyType": policy_type, "status": status, "communication": communication}
    elif model_name == 'running-example-Will-BPM-silent-loops' or model_name == 'running-example-Will-BPM-silent-loops-silent':
        a = np.random.uniform(0, 1000, 1)
        status = choice(["approved", "rejected"])
        policy_type = choice(["normal", "premium"])
        communication = choice(["email", "letter"])
        appeal = choice([True, False])
        ex_cont = {"amount": a[0], "policyType": policy_type, "status": status,
                "communication": communication, "appeal": appeal}
    elif model_name == 'running-example-Will-BPM-silent-loops-silent-loopB':
        a = np.random.uniform(0, 1000, 1)
        status = choice(["approved", "rejected"])
        policy_type = choice(["normal", "premium"])
        communication = choice(["email", "letter"])
        appeal = choice([True, False])
        discarded = choice([True, False])
        ex_cont = {"amount": a[0], "policyType": policy_type, "status": status,
                "communication": communication, "appeal": appeal, "discarded": discarded}
    elif model_name == 'running-example-paper':
        is_present = choice([True, False])
        skip_everything = choice([True, False])
        a = np.random.uniform(0, 1000, 1)
        doc_is_updated = choice([True, False])
        loan_accepted = choice(["yes", "no", "recheck"])
        ex_cont = {"amount": a[0], "is_present": is_present, "skip_everything": skip_everything,
                "doc_is_updated": doc_is_updated, "loan_accepted": loan_accepted}
    else:
        raise Exception("Model name not implemented.")
    
    return ex_cont
        
# playout
max_trace_length = 100
NO_TRACES = 100
case_id_key = xes_constants.DEFAULT_TRACEID_KEY
activity_key = xes_constants.DEFAULT_NAME_KEY
timestamp_key = xes_constants.DEFAULT_TIMESTAMP_KEY
curr_timestamp = datetime.datetime.now()

# playout until you are in final marking or exceeded max length or you are in a deadlock
all_visited = []
for i in tqdm(range(NO_TRACES)):
    #breakpoint()
    # reset marking to initial
    dm = DataMarking()
    dm[list(initial_marking.keys())[0]] = initial_marking.get(list(initial_marking.keys())[0])
    visited_elements = []
    all_enabled_trans = [0]
    # execution context
    ex_cont_total = get_ex_cont(net_name)
    if "loops" in net_name:
        ex_cont_total["appeal"] = choice([True, False])
    #breakpoint()
    while dm != final_marking and len(visited_elements) < max_trace_length and len(all_enabled_trans) > 0:
        verboseprint(dm)
        all_enabled_trans = dpn_semantics.enabled_transitions(net, dm, ex_cont_total)
        #breakpoint()
        for enabled in list(all_enabled_trans):
            if "guard" in enabled.properties:
                if not dpn_semantics.evaluate_guard(enabled.properties["guard"], enabled.properties["readVariable"], ex_cont_total):
                    all_enabled_trans.discard(enabled)
        if len(all_enabled_trans) == 0:
            breakpoint()
        #breakpoint()
        trans = choice(list(all_enabled_trans))
        #breakpoint()
        if "readVariable" in trans.properties:
            for read_var in trans.properties["readVariable"]:
                ex_cont[read_var] = ex_cont_total[read_var]
        else:
            ex_cont = dict()
        #dm = dpn_semantics.execute(trans, net, dm, ex_cont_total)
        dm = dpn_semantics.execute(trans, net, dm, ex_cont)
        #breakpoint()
        if not trans.label is None:
            if 'paper' in net_name:
                if trans.name in ["trans_A"]:
                    ex_cont["amount"] = copy.copy(ex_cont_total["amount"])
                    ex_cont["skip_everything"] = copy.copy(ex_cont_total["skip_everything"])
                    ex_cont["doc_is_updated"] = copy.copy(ex_cont_total["doc_is_updated"])
            visited_elements.append(tuple([trans, ex_cont]))
        if 'loopB' in net_name:
            if trans.name in ["trans_S"]:
                ex_cont_total["appeal"] = False
        elif 'loops' in net_name:
            if trans.name in ["trans_R"]:
                ex_cont_total["appeal"] = False
        if 'paper' in net_name:
            #breakpoint()
            if trans.name in ["skip_4"]:
                ex_cont_total["loan_accepted"] = choice(['yes', 'no'])
                ex_cont_total["skip_everything"] = False
                ex_cont_total["doc_is_updated"] = True

    if dm == final_marking:
        verboseprint("Final marking reached!")
    elif len(all_enabled_trans) == 0:
        verboseprint("Block in deadlock!")
    else:
        verboseprint("Max length of traces permitted")

    verboseprint("Visited activities: {}".format(visited_elements))
    all_visited.append(tuple(visited_elements))

#breakpoint()
log = log_instance.EventLog()
for index, element_sequence in tqdm(enumerate(all_visited)):
    #breakpoint()
    trace = log_instance.Trace()
    trace.attributes[case_id_key] = str(index)
    #breakpoint()
    for element in element_sequence:
        if 'trace-attr' in net_name:
            if "policyType" in element[1].keys():
                trace.attributes["policyType"] = element[1]["policyType"]
            if "communication" in element[1].keys():
                trace.attributes["communication"] = element[1]["communication"]
        event_el = element[0]
        ex_cont = element[1]
        if type(event_el) is PetriNet.Transition:
            event = log_instance.Event()
            event[activity_key] = event_el.label
            event[timestamp_key] = curr_timestamp
            for attr in ex_cont.keys():
                if not (ex_cont[attr] == -1 or ex_cont[attr] == 'None') and not ('trace-attr' in net_name and (attr == "communication" or attr == "policyType")):
                    event[attr] = copy.copy(ex_cont[attr])
            trace.append(event)
            # increase 5 minutes
            curr_timestamp = curr_timestamp + datetime.timedelta(minutes=5)
    log.append(trace)
#breakpoint()
xes_exporter.apply(log, 'data/log-{}.xes'.format(net_name))
