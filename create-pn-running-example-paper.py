import pm4py
import copy
import numpy as np
from tqdm import tqdm
from random import choice
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties

# create empty petri net
net_name = "running-example-paper-pn"
net = PetriNet(net_name)

# create and add places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
p_3 = PetriNet.Place("p_3")
p_4 = PetriNet.Place("p_4")
p_5 = PetriNet.Place("p_5")
p_6 = PetriNet.Place("p_6")
p_7 = PetriNet.Place("p_7")
p_8 = PetriNet.Place("p_8")
net.places.add(source)
net.places.add(sink)
net.places.add(p_1)
net.places.add(p_2)
net.places.add(p_3)
net.places.add(p_4)
net.places.add(p_5)
net.places.add(p_6)
net.places.add(p_7)
net.places.add(p_8)

# create and add trasitions
t_A = PetriNet.Transition("trans_A", "Request loan")
t_B = PetriNet.Transition("trans_B", "Register")
skip_1 = PetriNet.Transition("skip_1", None)
tauSplit_1 = PetriNet.Transition("tauSplit_1", None)
tauJoin_1 = PetriNet.Transition("tauJoin_1", None)
t_C = PetriNet.Transition("trans_C", "Check")
skip_2 = PetriNet.Transition("skip_2", None)
t_D = PetriNet.Transition("trans_D", "Prepare documents")
skip_3 = PetriNet.Transition("skip_3", None)
skip_4 = PetriNet.Transition("skip_4", None)
t_E = PetriNet.Transition("trans_E", "Final check")
t_F = PetriNet.Transition("trans_F", "Authorize")
t_G = PetriNet.Transition("trans_G", "Don't authorize")
net.transitions.add(t_A)
net.transitions.add(t_B)
net.transitions.add(t_C)
net.transitions.add(t_D)
net.transitions.add(t_E)
net.transitions.add(t_F)
net.transitions.add(t_G)
net.transitions.add(skip_1)
net.transitions.add(skip_2)
net.transitions.add(skip_3)
net.transitions.add(skip_4)
net.transitions.add(tauSplit_1)
net.transitions.add(tauJoin_1)

# add arcs
petri_utils.add_arc_from_to(source, t_A, net)
petri_utils.add_arc_from_to(t_A, p_1, net)
petri_utils.add_arc_from_to(p_1, t_B, net)
petri_utils.add_arc_from_to(p_1, skip_1, net)
petri_utils.add_arc_from_to(t_B, p_2, net)
petri_utils.add_arc_from_to(skip_1, p_2, net)
petri_utils.add_arc_from_to(p_2, tauSplit_1, net)
petri_utils.add_arc_from_to(tauSplit_1, p_3, net)
petri_utils.add_arc_from_to(tauSplit_1, p_4, net)
petri_utils.add_arc_from_to(p_3, t_C, net)
petri_utils.add_arc_from_to(p_3, skip_2, net)
petri_utils.add_arc_from_to(p_4, t_D, net)
petri_utils.add_arc_from_to(p_4, skip_3, net)
petri_utils.add_arc_from_to(t_C, p_5, net)
petri_utils.add_arc_from_to(skip_2, p_5, net)
petri_utils.add_arc_from_to(t_D, p_6, net)
petri_utils.add_arc_from_to(skip_3, p_6, net)
petri_utils.add_arc_from_to(p_5, tauJoin_1, net)
petri_utils.add_arc_from_to(p_6, tauJoin_1, net)
petri_utils.add_arc_from_to(tauJoin_1, p_7, net)
petri_utils.add_arc_from_to(p_7, t_E, net)
petri_utils.add_arc_from_to(t_E, p_8, net)
petri_utils.add_arc_from_to(p_8, t_F, net)
petri_utils.add_arc_from_to(p_8, t_G, net)
petri_utils.add_arc_from_to(p_8, skip_4, net)
petri_utils.add_arc_from_to(skip_4, p_2, net)
petri_utils.add_arc_from_to(t_F, sink, net)
petri_utils.add_arc_from_to(t_G, sink, net)

# initial and final marking
initial_marking = Marking()
initial_marking[source] = 1
final_marking = Marking()
final_marking[sink] = 1

#breakpoint()
pnml_exporter.apply(net, initial_marking, "models/{}.pnml".format(net_name), final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "{}.svg".format(net_name))
