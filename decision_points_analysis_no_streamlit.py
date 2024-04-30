import pm4py
import os
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import argcomplete
import argparse
import json
from sklearn import metrics
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from backward_search import get_decision_points_and_targets, get_all_dp_from_sink_to_last_event
from rules_extraction import sampling_dataset, extract_rules_with_pruning, pessimistic_pruning,\
    discover_overlapping_rules, shorten_rules_manually
from utils import get_attributes_from_event, get_map_events_to_transitions
from daikon_utils import discover_branching_conditions
from DecisionTreeC45 import DecisionTree


def main():
    def ModelCompleter(**kwargs):
        return [name.split('.')[0] for name in os.listdir('models')]
    # Argument (verbose and net_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str).completer = ModelCompleter
    argcomplete.autocomplete(parser)
    # parse arguments
    args = parser.parse_args()
    net_name = args.net_name

    # Importing the log
    log = xes_importer.apply('logs/log-{}.xes'.format(net_name))

    # Importing the attributes_map file
    attributes_map_file = f"{net_name}.attr"
    if attributes_map_file in os.listdir('dt-attributes'):
        with open(os.path.join('dt-attributes', attributes_map_file), 'r') as f:
            json_string = json.load(f)
            attributes_map = json.loads(json_string)
    else:
        raise FileNotFoundError('Create a configuration file for the decision tree before fitting it.')

    # Converting attributes types according to the attributes_map file
    for trace in log:
        for event in trace:
            for attribute in event.keys():
                if attribute in attributes_map:
                    if attributes_map[attribute] == 'continuous':
                        event[attribute] = float(event[attribute])
                    elif attributes_map[attribute] == 'boolean':
                        event[attribute] = bool(event[attribute])

    # Importing the Petri net model, if it exists
    try:
        net, im, fm = pnml_importer.apply("models/{}.pnml".format(net_name))
    except FileNotFoundError:
        print("Existing Petri Net model not found. Extracting one using the Inductive Miner...")
        net, im, fm = pm4py.discover_petri_net_inductive(log)

    # Stuff...
    sink_complete_net = [place for place in net.places if place.name == 'sink'][0]
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    # Dealing with loops and other stuff... needs cleaning
    events_to_trans_map = get_map_events_to_transitions(net)

    tic = time()
    # Scanning the log to get the logs related to decision points
    print('Extracting training logs from Event Log...')
    decision_points_data, event_attr, stored_dicts = dict(), dict(), dict()
    variants = variants_filter.get_variants(log)
    # Decision points of interest are searched considering the variants only
    for variant in tqdm(variants):
        transitions_sequence, events_sequence = list(), list()
        dp_events_sequence = dict()
        for i, event_name in enumerate(variant.split(',')):
            trans_from_event = events_to_trans_map[event_name]
            transitions_sequence.append(trans_from_event)
            events_sequence.append(event_name)
            if len(transitions_sequence) > 1:
                dp_dict, stored_dicts = get_decision_points_and_targets(transitions_sequence, net, stored_dicts)
                dp_events_sequence['Event_{}'.format(i+1)] = dp_dict

        # Final update of the current trace (from last event to sink)
        transition = [trans for trans in net.transitions if trans.label == event_name][0]
        dp_events_sequence['End'] = get_all_dp_from_sink_to_last_event(transition, sink_complete_net, dp_events_sequence)

        for trace in variants[variant]:
            # Storing the trace attributes (if any)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                event_attr.update(trace.attributes)

            # Keeping the same attributes observed in previously (to keep dictionaries at the same length)
            event_attr = {k: np.nan for k in event_attr.keys()}

            transitions_sequence = list()
            for i, event in enumerate(trace):
                trans_from_event = events_to_trans_map[event["concept:name"]]
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

    # Data has been gathered. For each decision point, fitting a decision tree on its logs and extracting the rules
    file_name = 'test.txt'
    for decision_point in decision_points_data.keys():
        print("\nDecision point: {}".format(decision_point))
        dataset = pd.DataFrame.from_dict(decision_points_data[decision_point])
        # Replacing ':' with '_' both in the dataset columns and in the attributes map since ':' creates problems
        dataset.columns = dataset.columns.str.replace(':', '_')
        attributes_map = {k.replace(':', '_'): attributes_map[k] for k in attributes_map}

        # Discovering branching conditions with Daikon - comment these four lines to go back to decision tree + pruning
        # rules = discover_branching_conditions(dataset)
        # rules = {k: rules[k].replace('_', ':') for k in rules}
        # print(rules)
        # continue

        print("Fitting a decision tree on the decision point's dataset...")
        accuracies, f_scores = list(), list()
        for i in tqdm(range(10)):
            # Sampling
            dataset = sampling_dataset(dataset)

            # Fitting
            dt = DecisionTree.DecisionTree(attributes_map)
            dt.fit(dataset)

            # Predict
            y_pred = dt.predict(dataset.drop(columns=['target']))

            # Accuracy
            accuracy = metrics.accuracy_score(dataset['target'], y_pred)
            accuracies.append(accuracy)

            # F1-score
            if len(dataset['target'].unique()) > 2:
                f1_score = metrics.f1_score(dataset['target'], y_pred, average='weighted')
            else:
                f1_score = metrics.f1_score(dataset['target'], y_pred, pos_label=dataset['target'].unique()[0])
            f_scores.append(f1_score)

        # Rules extraction
        if len(dt.get_nodes()) > 1:
            print("Training complete. Extracting rules...")
            with open(file_name, 'a') as f:
                f.write('{} - SUCCESS\n'.format(decision_point))
                f.write('Dataset target values counts:\n {}\n'.format(dataset['target'].value_counts()))

                print("Train accuracy: {}".format(sum(accuracies) / len(accuracies)))
                f.write('Accuracy: {}\n'.format(sum(accuracies) / len(accuracies)))
                print("F1 score: {}".format(sum(f_scores) / len(f_scores)))
                f.write('F1 score: {}\n'.format(sum(f_scores) / len(f_scores)))

                # Rule extraction without pruning
                rules = dt.extract_rules()

                # Rule extraction with pruning
                # rules = dt.extract_rules_with_pruning(dataset)

                # Alternative pruning (directly on tree)
                # dt.pessimistic_pruning(dataset)
                # rules = dt.extract_rules()

                # Overlapping rules discovery
                # rules = discover_overlapping_rules(dt, dataset, attributes_map, rules)

                rules = shorten_rules_manually(rules, attributes_map)
                rules = {k: rules[k].replace('_', ':') for k in rules}

                f.write('Rules:\n')
                for k in rules:
                    f.write('{}: {}\n'.format(k, rules[k]))
                f.write('\n')
            print(rules)
        else:
            with open(file_name, 'a') as f:
                f.write('{} - FAIL\n'.format(decision_point))
                f.write('Dataset target values counts: {}\n'.format(dataset['target'].value_counts()))

    toc = time()
    print("\nTotal time: {}".format(toc-tic))


if __name__ == '__main__':
    main()
