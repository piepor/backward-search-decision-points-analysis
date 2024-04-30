import argparse
from time import time
from pm4py.objects.log.importer.xes import importer as xes_importer
from importing import import_attributes, convert_attributes, import_petri_net
from model_utils import get_activities_to_transitions_map
from algo import MultiplePaths
from training import train
from processing import discard_source, keep_only_desired_attributes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str)    # parse arguments
    args = parser.parse_args()
    net_name = args.net_name

    # Importing the log
    log_name = "_".join(net_name.split("_")[:-1])
    log = xes_importer.apply('logs/log-{}.xes'.format(log_name))

    # Importing the attributes_map file
    attributes_map = import_attributes(net_name)

    # Converting attributes types according to the attributes_map file
    log = convert_attributes(attributes_map, log)

    # import model
    petri_net, init_mark, final_mark = import_petri_net(net_name)

#    gviz = pn_visualizer.apply(net, im, fm)
#    pn_visualizer.view(gviz)

    # Dealing with loops and other stuff... needs cleaning
    activities_to_trans_map = get_activities_to_transitions_map(petri_net)
    # TODO declare algo type

    tic = time()
    # Scanning the log to get the logs related to decision points
    print('Extracting training logs from Event Log...')
    algorithm = MultiplePaths(petri_net, activities_to_trans_map)
    training_data, complexity = algorithm.old_extract_decision_points_data(log)
    training_data = keep_only_desired_attributes(training_data, attributes_map)
    training_data = discard_source(training_data)
    #training_data = algorithm.old_extract_decision_points_data_only_last_event(log)
    # Data has been gathered. For each decision point, fitting a decision tree on its logs and extracting the rules
    train(training_data, attributes_map, net_name, './models')
    toc = time()
    print("\nTotal time: {}".format(toc-tic))


if __name__ == "__main__":
    main()
