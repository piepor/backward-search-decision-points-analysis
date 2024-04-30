from tqdm import tqdm
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet
from backward_search import get_all_dp_to_source_from_first_event, get_decision_points_and_targets, get_all_dp_from_sink_to_last_event
from model_utils import get_transition, get_sink
from processing import insert_trace_data
from processing import get_attributes_from_event
from diego_streamlit_utils import build_datasets

import numpy as np

class MultiplePaths:
    def __init__(self, petri_net: PetriNet, activities_to_trans_map: dict):
        self.petri_net = petri_net
        self.activities_to_trans_map = activities_to_trans_map
        self.sink = get_sink(self.petri_net)
        self.stored_dicts = dict()
        #self.dp_dict = dict()

    def activities_decision_points_map(self, variant: str) -> dict:
        transitions_sequence, activities = list(), list()
        dp_activities = dict()
        for i, activity in enumerate(variant.split(',')):
            trans_from_activity = self.activities_to_trans_map[activity]
            transitions_sequence.append(trans_from_activity)
            activities.append(activity)
            if len(transitions_sequence) > 1:
                dp_dict, self.stored_dicts = get_decision_points_and_targets(transitions_sequence, self.petri_net, self.stored_dicts)
                dp_activities['Activity_{}'.format(i+1)] = dp_dict
        # Final update of the current trace (from last event to snk)
        transition =  get_transition(self.petri_net, activity)
        dp_activities['End'] = get_all_dp_from_sink_to_last_event(transition, self.sink, dp_activities)
        return dp_activities

    def extract_decision_points_data(self, event_log: EventLog) -> dict:
        decision_points_data, event_attr = dict(), dict()
        variants = variants_filter.get_variants(event_log)
        # Decision points of interest are searched considering the variants only
        for variant in tqdm(variants):
            dp_events_sequence = self.activities_decision_points_map(variant)
            #breakpoint()
            traces = variants[variant]
            decision_points_data = insert_trace_data(
                    traces, decision_points_data, event_attr, dp_events_sequence, self.activities_to_trans_map)
        return decision_points_data

    def old_extract_decision_points_data(self, log):

        decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
        variants = variants_filter.get_variants(log)
# Decision points of interest are searched considering the variants only
        complexity = {'sequence_length': [], 'operations_number': []}
        for variant in tqdm(variants):
            transitions_sequence, events_sequence = list(), list()
            dp_events_sequence = dict()
            counters = []
            for i, event_name in enumerate(variant.split(',')):
                #trans_from_event = events_to_trans_map[event_name]
                # dealing with events not present in the net
                if event_name in self.activities_to_trans_map:
                    trans_from_event = self.activities_to_trans_map[event_name]
                else:
                    trans_from_event = "NotContainedInTheNet"
                transitions_sequence.append(trans_from_event)
                events_sequence.append(event_name)
                if len(transitions_sequence) > 1:
                    # dealing with events not present in the net
                    if not trans_from_event == "NotContainedInTheNet":
                        dp_dict, stored_dicts, counter = get_decision_points_and_targets(transitions_sequence, self.petri_net, stored_dicts)
                    else:
                        dp_dict = dict()
                    dp_events_sequence['Event_{}'.format(i+1)] = dp_dict
                    counters.append(counter)
                else:
                    # get dp from source to first event
                    transition = [trans for trans in self.petri_net.transitions if trans.label == event_name][0]
                    if transition:
                        dp_dict, counter = get_all_dp_to_source_from_first_event(transition)
                        dp_events_sequence['Event_{}'.format(i+1)] = dp_dict
                        counters.append(counter)
                #breakpoint()

            # Final update of the current trace (from last event to sink)
            if event_name in self.activities_to_trans_map:
                transition = [trans for trans in self.petri_net.transitions if trans.label == event_name][0]
            else:
                transition = None
            #dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, sink_complete_net, dp_events_sequence)
            # dealing with events not present in the net
            if transition:
                dp_events_sequence['End'], counter = get_all_dp_from_sink_to_last_event(transition, self.sink, dp_events_sequence)
            else:
                dp_events_sequence['End'] = dict()
                counter = 0
            counters.append(counter)
            complexity['sequence_length'].append(len(variant.split(',')))
            complexity['operations_number'].append(sum(counters))

            for trace in variants[variant]:
                # Storing the trace attributes (if any)
                if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                    event_attr.update(trace.attributes)

                # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
                event_attr = {k: np.nan for k in event_attr.keys()}

                transitions_sequence = list()
                for i, event in enumerate(trace):
                    #trans_from_event = events_to_trans_map[event["concept:name"]]
                    # dealing with events not present in the net
                    if event["concept:name"] in self.activities_to_trans_map:
                        trans_from_event = self.activities_to_trans_map[event["concept:name"]]
                    else:
                        trans_from_event = "NotContainedInTheNet"
                    transitions_sequence.append(trans_from_event)

                    # Appending the last attribute values to the decision point dictionary
                    #if len(transitions_sequence) > 1:
                    dp_dict = dp_events_sequence['Event_{}'.format(i+1)]
                    for dp in dp_dict.keys():
                        # Adding the decision point to the total dictionary if it is not already there
                        if dp not in decision_points_data.keys():
                            decision_points_data[dp] = {k: [] for k in ['target']}
                        for dp_target in dp_dict[dp]:
                            for a in event_attr.keys():
                                # Attribute not present and not nan: add it as new key and fill previous entries as nan
                                if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                    n_entries = len(decision_points_data[dp]['target'])
                                    decision_points_data[dp][a] = [np.nan] * n_entries
                                    decision_points_data[dp][a].append(event_attr[a])
                                # Attribute present: just append it to the existing list
                                elif a in decision_points_data[dp]:
                                    decision_points_data[dp][a].append(event_attr[a])
                            # Appending also the target transition label to the decision point dictionary
                            decision_points_data[dp]['target'].append(dp_target)

                    # Updating the attribute values dictionary with the values from the current event
                    event_attr.update(get_attributes_from_event(event))
                
                # Appending the last attribute values to the decision point dictionary (from last event to sink)
                if len(dp_events_sequence['End']) > 0:
                    for dp in dp_events_sequence['End'].keys():
                        if dp not in decision_points_data.keys():
                            decision_points_data[dp] = {k: [] for k in ['target']}
                        for dp_target in dp_events_sequence['End'][dp]:
                            for a in event_attr.keys():
                                if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                    n_entries = len(decision_points_data[dp]['target'])
                                    decision_points_data[dp][a] = [np.nan] * n_entries
                                    decision_points_data[dp][a].append(event_attr[a])
                                elif a in decision_points_data[dp]:
                                    decision_points_data[dp][a].append(event_attr[a])
                            decision_points_data[dp]['target'].append(dp_target)
        return decision_points_data, complexity

    def old_extract_decision_points_data_only_last_event(self, log):

        decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
        variants = variants_filter.get_variants(log)
# Decision points of interest are searched considering the variants only
        for variant in tqdm(variants):
            transitions_sequence, events_sequence = list(), list()
            dp_events_sequence = dict()
            for i, event_name in enumerate(variant.split(',')):
                #trans_from_event = events_to_trans_map[event_name]
                trans_from_event = self.activities_to_trans_map[event_name]
                transitions_sequence.append(trans_from_event)
                events_sequence.append(event_name)
                if len(transitions_sequence) > 1:
                    dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, self.petri_net, stored_dicts)
                    dp_events_sequence['Event_{}'.format(i+1)] = dp_dict

            # Final update of the current trace (from last event to sink)
            transition = [trans for trans in self.petri_net.transitions if trans.label == event_name][0]
            #dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, sink_complete_net, dp_events_sequence)
            dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, self.sink, dp_events_sequence)

            for trace in variants[variant]:
                # Storing the trace attributes (if any)
                if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                    event_attr.update(trace.attributes)

                # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
                event_attr = {k: np.nan for k in event_attr.keys()}

                transitions_sequence = list()
                for i, event in enumerate(trace):
                    #trans_from_event = events_to_trans_map[event["concept:name"]]
                    trans_from_event = self.activities_to_trans_map[event["concept:name"]]
                    transitions_sequence.append(trans_from_event)

                    # Appending the last attribute values to the decision point dictionary
                    if len(transitions_sequence) > 1:
                        dp_dict = dp_events_sequence['Event_{}'.format(i+1)]
                        for dp in dp_dict.keys():
                            # Adding the decision point to the total dictionary if it is not already there
                            if dp not in decision_points_data.keys():
                                decision_points_data[dp] = {k: [] for k in ['target']}
                            for dp_target in dp_dict[dp]:
                                for a in event_attr.keys():
                                    # Attribute not present and not nan: add it as new key and fill previous entries as nan
                                    if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                        n_entries = len(decision_points_data[dp]['target'])
                                        decision_points_data[dp][a] = [np.nan] * n_entries
                                        decision_points_data[dp][a].append(event_attr[a])
                                    # Attribute present: just append it to the existing list
                                    elif a in decision_points_data[dp]:
                                        decision_points_data[dp][a].append(event_attr[a])
                                # Appending also the target transition label to the decision point dictionary
                                decision_points_data[dp]['target'].append(dp_target)

                    # Resetting attribute values to keep only the last event's ones
                    event_attr = {k: np.nan for k in event_attr.keys()}
                    # Updating the attribute values dictionary with the values from the current event
                    event_attr.update(get_attributes_from_event(event))

                # Appending the last attribute values to the decision point dictionary (from last event to sink)
                if len(dp_events_sequence['End']) > 0:
                    for dp in dp_events_sequence['End'].keys():
                        if dp not in decision_points_data.keys():
                            decision_points_data[dp] = {k: [] for k in ['target']}
                        for dp_target in dp_events_sequence['End'][dp]:
                            for a in event_attr.keys():
                                if a not in decision_points_data[dp] and event_attr[a] is not np.nan:
                                    n_entries = len(decision_points_data[dp]['target'])
                                    decision_points_data[dp][a] = [np.nan] * n_entries
                                    decision_points_data[dp][a].append(event_attr[a])
                                elif a in decision_points_data[dp]:
                                    decision_points_data[dp][a].append(event_attr[a])
                            decision_points_data[dp]['target'].append(dp_target)
        return decision_points_data

    def build_dataset(self, event_log):
        return build_datasets(self.petri_net, event_log)
