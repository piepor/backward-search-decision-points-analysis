from pm4py.objects.petri_net.obj import PetriNet


def get_sink(petri_net: PetriNet) -> PetriNet.Place:
    """ Return the sink place of the net """
    return [place for place in petri_net.places if place.name == 'sink'][0]

def get_transition(petri_net: PetriNet, activity: str) -> PetriNet.Transition:
    return [trans for trans in petri_net.transitions if trans.label == activity][0]

def get_activities_to_transitions_map(petri_net: PetriNet) -> dict:
    """ Compute a map of transitions name and events

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every event the corresponding transition name
    """
    # initialize
    map_trans_activities = dict()
    for trans in petri_net.transitions:
        map_trans_activities[trans.label] = trans.name
    map_trans_activities['None'] = 'None'
    return map_trans_activities

