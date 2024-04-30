import numpy as np
from pm4py.objects.log.obj import Trace
from copy import copy


def discard_source(data: dict):
    filtered_data = dict()
    for dp in data:
        if not dp == 'source':
            filtered_data[dp] = copy(data[dp])
    return filtered_data

def keep_only_desired_attributes(data: dict, attributes_map: dict):
    filtered_data = dict()
    for dp in data:
        filtered_data[dp] = dict()
        filtered_data[dp]['target'] = copy(data[dp]['target'])
        for attr in data[dp]:
            if attr in attributes_map:
                filtered_data[dp][attr] = copy(data[dp][attr])
    return filtered_data

def get_attributes_from_event(event) -> dict:
    """ Given an event from an event log, returns a dictionary containing the attributes values of the given event.

    Each attribute is set as key of the dictionary, and its value is the actual value of the attribute in the event.
    Attributes 'time:timestamp' and 'concept:name' are ignored and therefore not added to the dictionary.
    """

    attributes = dict()
    for attribute in event.keys():
        if attribute not in ['time:timestamp', 'concept:name']:
            attributes[attribute] = event[attribute]
    return attributes

def insert_attribute(attribute: str, data: dict, decision_point: str) -> dict:
    n_entries = len(data[decision_point]['target'])
    data[decision_point][attribute] = [np.nan] * n_entries
    return data

def insert_decision_point(data: dict, decision_point: str):
    data[decision_point] = {k: [] for k in ['target']}
    return data

def insert_nan(decision_points_data: dict, dp: str, event_attr: dict) -> dict:
    key_dp = set(decision_points_data[dp].keys())
    key_event = set(event_attr.keys())
    missing_columns = key_dp.difference(key_event)
    for column in missing_columns:
        if column != 'target':
            decision_points_data[dp][column].append(np.nan)
    return decision_points_data

def insert_data(event_dps: dict, decision_points_data: dict, event_attr: dict) -> dict:
    for dp in event_dps.keys():
        # Adding the decision point to the total dictionary if it is not already there
        if dp not in decision_points_data.keys():
            decision_points_data = insert_decision_point(decision_points_data, dp)
        for dp_target in event_dps[dp]:
            for attribute in event_attr.keys():
                # Attribute not present and not nan: add it as new key and fill previous entries as nan
                if attribute not in decision_points_data[dp] and event_attr[attribute] is not np.nan:
                    n_entries = len(decision_points_data[dp]['target'])
                    decision_points_data[dp][attribute] = [np.nan] * n_entries
                    decision_points_data[dp][attribute].append(event_attr[attribute])
                # Attribute present: just append it to the existing list
                elif attribute in decision_points_data[dp]:
                    decision_points_data[dp][attribute].append(event_attr[attribute])
            # Appending also the target transition label to the decision point dictionary
            decision_points_data[dp]['target'].append(dp_target)
#                if attribute not in decision_points_data[dp]:
#                    decision_points_data = insert_attribute(attribute, decision_points_data, dp)
#                # Attribute present: just append it to the existing list
#                decision_points_data[dp][attribute].append(event_attr[attribute])
            # insert attributes present in the dataset but not in the event
            #decision_points_data = insert_nan(decision_points_data, dp, event_attr)
            # Appending also the target transition label to the decision point dictionary
            #decision_points_data[dp]['target'].append(dp_target)
    return decision_points_data

def insert_trace_data(traces: list[Trace], data: dict, event_attr: dict, dp_events_sequence: dict, activities_to_trans_map: dict):
    for trace in traces:
        # Storing the trace attributes (if any)
        if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
            event_attr.update(trace.attributes)

        # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
        event_attr = {k: np.nan for k in event_attr.keys()}

        transitions_sequence = list()
        for i, event in enumerate(trace):
            trans_from_event = activities_to_trans_map[event["concept:name"]]
            transitions_sequence.append(trans_from_event)

            # Appending the last attribute values to the decision point dictionary
            if len(transitions_sequence) > 1:
                dp_dict = dp_events_sequence['Activity_{}'.format(i+1)]
                data = insert_data(dp_dict, data, event_attr)
            # Updating the attribute values dictionary with the values from the current event
            event_attr.update(get_attributes_from_event(event))

        # Appending the last attribute values to the decision point dictionary (from last event to sink)
        if len(dp_events_sequence['End']) > 0:
            data = insert_data(dp_events_sequence['End'], data, event_attr)
    return data
