import copy

import numpy as np
import pandas as pd
from sklearn.tree import export_text
from pm4py.objects.petri_net.obj import PetriNet

#from DecisionTree import DecisionTree
#from decision_tree_utils import extract_rules_from_leaf


def are_activities_parallel(first_activity, second_activity, parallel_branches) -> bool:
    """ Check if two activities are parallel in the net.

    The dictionary 'parallel_branches' is derived from the net simplification algorithm."""
    are_parallel = False
    for split in parallel_branches.keys():
        for branch in parallel_branches[split].keys():
            if first_activity in parallel_branches[split][branch] and not second_activity in parallel_branches[split][branch]:
                for other_branch in parallel_branches[split].keys():
                    if not other_branch == branch:
                        if second_activity in parallel_branches[split][other_branch]:
                            are_parallel = True
    return are_parallel


def get_decision_points_and_targets(sequence, loops, net, reachable_activities, stored_dicts) -> [dict, dict]:
    """ Returns a dictionary containing decision points and their targets.

    Starting from the last activity in the sequence, the algorithm selects the previous not
    parallel activity. Exploring the net backward, it returns all the decision points with their targets
    encountered on the path leading to the last activity in the sequence.
    """

    # search the first not parallel activity before the last in the sequence
    current_act_name = sequence[-1]
    current_act = [trans for trans in net.transitions if trans.name == current_act_name][0]
    previous_sequence = sequence[:-1]
    previous_sequence.reverse()
    previous_act_name = None
    for previous in previous_sequence:
        '''
        COMMENTED SINCE WE ARE TEMPORARILY USING MANUAL 'REACHABLE_ACTIVITIES' DICTIONARY
        # problems if the loop has some parts that cannot be simplified
        # TODO change the simplifying algo to be sure that all the parallel parts are recognized
        parallel = are_activities_parallel(current_act_name, previous, parallel_branches)
        if not parallel:
            previous_act_name = previous
            break
        '''
        previous_act = [trans for trans in net.transitions if trans.name == previous][0]
        reachable = current_act.label in reachable_activities[previous_act.label]
        if reachable:
            previous_act_name = previous_act.name
            break
    if previous_act_name is None:
        raise Exception("Can't find the previous not parallel activity")

    # TODO reachability becomes useless if _new_get_dp_to_previous_event is used (and also loops)
    reachability = dict()
    # check if the two activities are contained in the same loop and save it if exists
#    for loop in loops:
#        if loop.is_node_in_loop_complete_net(current_act_name) and loop.is_node_in_loop_complete_net(previous_act_name):
#            # check if the last activity is reachable from the previous one (if not it means that the loop is active)
#            reachability[loop.name] = loop.check_if_reachable(previous_act, current_act.name, False)

    # Extracting the decision points between previous and current activities, if not already computed
    prev_curr_key = ' ,'.join([previous_act_name, current_act_name])
    if prev_curr_key not in stored_dicts.keys():
        dp_dict, _, _ = _new_get_dp_to_previous_event(previous_act_name, current_act)

        # ---------- ONLY SHORTEST PATH APPROACH (comment method call above) ----------
        '''
        paths = get_all_paths(previous_act_name, current_act)
        min_length_value = min(len(paths[k]) for k in paths)
        min_length_path = [l for k, l in paths.items() if len(l) == min_length_value][0]
        dp_dict = dict(min_length_path)
        old_keys = list(dp_dict.keys()).copy()
        for dp in old_keys:
            if len(dp.out_arcs) < 2:
                del dp_dict[dp]
            else:
                dp_dict[dp.name] = dp_dict.pop(dp)
        '''
        # ---------- ONLY SHORTEST PATH APPROACH (comment method call above) ----------

        # TODO this is just for debugging
        print("\nPrevious event: {}".format(previous_act.label))
        print("Current event: {}".format(current_act.label))
        print("DPs")
        events_trans_map = get_map_events_transitions(net)
        for key in dp_dict.keys():
            print(" - {}".format(key))
            for inn_key in dp_dict[key]:
                if inn_key in events_trans_map.keys():
                    print("   - {}".format(events_trans_map[inn_key]))
                else:
                    print("   - {}".format(inn_key))

        stored_dicts[prev_curr_key] = dp_dict
    else:
        dp_dict = stored_dicts[prev_curr_key]

    return dp_dict, stored_dicts


def _new_get_dp_to_previous_event(previous, current, decision_points=None, decision_points_temp=None, passed_inn_arcs=None) -> [dict, dict, bool]:
    """ Extracts all the decision points that are traversed between two activities (previous and current), reporting the
    decision(s) that has been taken for each of them.

    Starting from the 'current' activity, the algorithm proceeds backwards on each incoming arc (in_arc). Then, it saves
    the so-called 'inner_arcs', which are the arcs between the previous place (in_arc.source) and its previous
    activities (in_arc.source.in_arcs).
    If the 'previous' activity is immediately before the previous place (so it is the source of one of the inner_arcs),
    then the algorithm set a boolean variable 'target_found' to True to signal that the target has been found, and it
    adds the corresponding decision point (in_arc.source) to the dictionary.
    In any case, all the other backward paths containing an invisible activity are explored recursively.
    Every time an 'inner_arc' is traversed for exploration, it is added to the 'passed_inn_arcs' list, to avoid looping
    endlessly. Indeed, before a new recursion on an 'inner_arc', the algorithm checks if it is present in that list:
        1) If it is not present, it simply goes on with the recursion, since it means that the specific path has not
           been explored yet.
        2) Otherwise, it means that the current recursive call has already passed through that 'inner_arc' and so it
           must stop the recursion. This also means that the path followed a loop, and so the decision points
           encountered along that path must be added to the dictionary, since the target activity ('previous') can be
           reached through that loop.
    Note that decision points are then added to the dictionary in a forward way: whenever the recursion cannot go on (no
    more invisible activities backward) it returns also signalling the 'target_found' value. This is True if the current
    path found the target activity (or an already explored 'inner_arc' during the same path, as explained before). The
    returned value is then put in disjunction with the current value of 'target_found' in case the target activity has
    been found by the actual instance and not by the recursive one.

    It returns a 'decision_points' dictionary which contains, for each decision point on every possible path between
    the 'current' activity and the 'previous' activity, the target value(s) to be followed.

    # TODO decision_points_temp is not used at the moment, it was just to start working on handling parallel activities.
    """

    if decision_points is None:
        decision_points = dict()
    if decision_points_temp is None:
        decision_points_temp = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    target_found = False
    for in_arc in current.in_arcs:
        # Preparing the lists containing inner_arcs towards invisible and non-invisible transitions
        inner_inv_acts, inner_in_arcs_names = set(), set()
        for inner_in_arc in in_arc.source.in_arcs:
            if inner_in_arc.source.label is None:
                inner_inv_acts.add(inner_in_arc)
            else:
                inner_in_arcs_names.add(inner_in_arc.source.name)

        # Base case: the target activity (previous) is one of the activities immediately before the current one
        if previous in inner_in_arcs_names:
            target_found = True
            decision_points = _add_dp_target(decision_points, in_arc.source, current.name, target_found)
        # Recursive case: follow every invisible activity backward
        for inner_in_arc in inner_inv_acts:
            if inner_in_arc not in passed_inn_arcs:
                passed_inn_arcs.add(inner_in_arc)
                decision_points, decision_points_temp, previous_found = _new_get_dp_to_previous_event(previous, inner_in_arc.source,
                                                                                decision_points, decision_points_temp, passed_inn_arcs)
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, previous_found)
                passed_inn_arcs.remove(inner_in_arc)
                target_found = target_found or previous_found
                '''
                if target_found:
                    for dp_temp in decision_points_temp.keys():
                        if dp_temp not in decision_points.keys():
                            decision_points[dp_temp] = set()
                        decision_points[dp_temp].update(decision_points_temp[dp_temp])
                    decision_points_temp.clear()
                '''
            else:
                target_found = True
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, target_found)
                # for decision_points_temp, comment target_found=True and add dp target to decision_points_temp

    return decision_points, decision_points_temp, target_found


def _add_dp_target(decision_points, dp, target, previous_found):
    """ Adds the decision point and its target activity to the 'decision_points' dictionary.

    Given the 'decision_points' dictionary, the place 'dp' (in_arc.source) and the target activity name (current.name),
    adds the target activity name to the set of targets related to the decision point. If not presents, adds the
    decision point to the dictionary keys first.
    This is done if the place is an actual decision point, and if the boolean variable 'previous_found' is True.
    """

    if previous_found and len(dp.out_arcs) > 1:
        if dp.name in decision_points.keys():
            decision_points[dp.name].add(target)
        else:
            decision_points[dp.name] = {target}
    return decision_points


def get_all_paths(previous, current, passed_places=None, passed_inn_arcs=None, paths=None):

    if passed_places is None:
        passed_places = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    if paths is None:
        paths = dict()

    for in_arc in current.in_arcs:
        # Preparing the lists containing inner_arcs towards invisible and non-invisible transitions
        inner_inv_acts, inner_in_arcs_names = set(), set()
        for inner_in_arc in in_arc.source.in_arcs:
            if inner_in_arc.source.label is None:
                inner_inv_acts.add(inner_in_arc)
            else:
                inner_in_arcs_names.add(inner_in_arc.source.name)

        if in_arc.source not in passed_places.keys():
            passed_places[in_arc.source] = set()
        passed_places[in_arc.source].add(current.name)

        # Base case: the target activity (previous) is one of the activities immediately before the current one
        if previous in inner_in_arcs_names:
            paths[len(paths)] = copy.deepcopy(passed_places)
        else:
            # Recursive case: follow every invisible activity backward
            for inner_in_arc in inner_inv_acts:
                if inner_in_arc not in passed_inn_arcs:
                    passed_inn_arcs.add(inner_in_arc)
                    paths = get_all_paths(previous, inner_in_arc.source, passed_places, passed_inn_arcs, paths)
                    passed_inn_arcs.remove(inner_in_arc)

        passed_places[in_arc.source].remove(current.name)
        if len(passed_places[in_arc.source]) == 0:
            del passed_places[in_arc.source]

    return paths


def get_dp_to_previous_event(previous, current, loops, common_loops, decision_points, reachability, passed_inv_act) -> [dict, bool, bool]:
    """ Recursively explores the net and saves the decision points encountered with the targets.

    The backward recursion is allowed only if there is an invisbile activity and the target activity has not been reached.
    When a loop input is encountered, if the two activities are in the same loop and the 'current' is reachable from
    the 'previous' the algorithm stops. If the two activities are not in the same loop, the algorithm chooses to exit from the loop
    otherwise if the two activities are in the same loop but 'current' is not reachable from 'previous' the algorithm chooses to remain in the
    loop (i.e. it goes back)
    """
    #print(current.in_arcs)
    #breakpoint()
    for in_arc in current.in_arcs:
        #print(in_arc)
        #breakpoint()
        # setting previous_reached to False because we want to explore ALL the possible paths
        previous_reached = False
        not_found = False
        # check if the previous is in the node inputs
        inner_in_arcs_names = [inner_in_arc.source.name for inner_in_arc in in_arc.source.in_arcs if not inner_in_arc.source.label is None]
        inv_act_names = [inner_in_arc.source.name for inner_in_arc in in_arc.source.in_arcs if inner_in_arc.source.label is None]
        # recurse if the target activity is not in the inputs of the place and there is at least one invisible activity
        if not previous in inner_in_arcs_names and len(inv_act_names) > 0:
            for inner_in_arc in in_arc.source.in_arcs:
                # inner_in_arc.source is a transition
                not_found = False
                previous_reached = False
                is_activity_in_some_loop = False
                # check if we are in a loop and if it is one of the loops containing both the activities
                for loop in loops:
                    go_on = False
                    stop_recursion = False
                    # check the conditions for the recursion
                    # for loops that contain both 'current' and 'previous'
                    if loop in common_loops:
                        # 1) the node is an input of the loop, the next (backward) activity is also in the loop AND the last activitiy in the initial sequence is NOT reachable from 'previous'
                        if loop.is_node_in_loop_complete_net(inner_in_arc.source.name) and loop.is_input_node_complete_net(in_arc.source.name) and not reachability[loop.name]:
                            go_on = True
                        # if we reached an input of the loop and the last activity in the initial sequence was reachable, the algorithm stops
                        elif loop.is_input_node_complete_net(in_arc.source.name) and reachability[loop.name]:
                            not_found = True
                        # 2) the node is not an input loop AND the next (backward) is also in the loop
                        elif loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                            go_on = True
                        # in all other cases the recursion must stop
                        else:
                            stop_recursion = True
                    # for all the other loops
                    if not loop in common_loops and len(common_loops) == 0:
                        # 3) the node is a loop input AND the next (backward) activity is NOT in the loop
                        if loop.is_input_node_complete_net(in_arc.source.name) and not loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                            go_on = True
                        # 4) both the node and the next (backward) transition are in the loop
                        elif not loop.is_input_node_complete_net(in_arc.source.name) and loop.is_node_in_loop_complete_net(in_arc.source.name) and loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                            go_on = True
                        else:
                        # in all other cases the recursion must stop
                        # TODO check this condition
                            stop_recursion = True
                    if go_on:
                        # if the transition is silent and we haven't already seen it
                        if not inner_in_arc.source.name == previous and inner_in_arc.source.label is None and not inner_in_arc.source.name in passed_inv_act.keys():
                            # current is a transition, in_arc.source is a place
                            if len(in_arc.source.out_arcs) > 1:
                                if in_arc.source.name in decision_points.keys():
                                    # using set ecause it is possible to pass multiple time through the same decision point
                                    decision_points[in_arc.source.name].add(current.name)
                                else:
                                    decision_points[in_arc.source.name] = {current.name}
                            decision_points, not_found, previous_reached = get_dp_to_previous_event(previous, inner_in_arc.source, loops, common_loops, decision_points, reachability, passed_inv_act)
                            # add the silent transition to the alread seen list
                            # I need to know if passing through the invisible activity already seen would have led me to the target activity
                            passed_inv_act[inner_in_arc.source.name] = {'previous_reached': previous_reached, 'not_found': not_found}
                            if not_found:
                                if in_arc.source.name in decision_points.keys():
                                    if current.name in decision_points[in_arc.source.name] and current.label is None:
                                        decision_points[in_arc.source.name].remove(current.name)
                        elif inner_in_arc.source.name == previous:
                            previous_reached = True
                            if len(in_arc.source.out_arcs) > 1:
                                if in_arc.source.name in decision_points.keys():
                                    # using set ecause it is possible to pass multiple time through the same decision point
                                    decision_points[in_arc.source.name].add(current.name)
                                else:
                                    decision_points[in_arc.source.name] = {current.name}
                        elif inner_in_arc.source.name in passed_inv_act.keys():
                            if passed_inv_act[inner_in_arc.source.name]['previous_reached']:
                                # TODO previous_reached = True?
                                if len(in_arc.source.out_arcs) > 1:
                                    if in_arc.source.name in decision_points.keys():
                                        # using set ecause it is possible to pass multiple time through the same decision point
                                        decision_points[in_arc.source.name].add(current.name)
                                    else:
                                        decision_points[in_arc.source.name] = {current.name}
                                previous_reached = True
                            elif passed_inv_act[inner_in_arc.source.name]['not_found']:
                                not_found = True
                        elif not inner_in_arc.source.label is None:
                            not_found = True
                    if loop.is_node_in_loop_complete_net(inner_in_arc.source.name):
                        is_activity_in_some_loop = True
                    # if we reached the target, stops the search of other loops
                    if previous_reached == True:
                        break
                # if there aren't loops or the activity doesn't belong to any of them and we didn't pass inside the previous selections, go on if invisible not alread seen
                if not is_activity_in_some_loop and not stop_recursion:
                    if not inner_in_arc.source.name == previous and inner_in_arc.source.label is None and not inner_in_arc.source.name in passed_inv_act.keys():
                        if len(in_arc.source.out_arcs) > 1:
                            if in_arc.source.name in decision_points.keys():
                                # using set ecause it is possible to pass multiple time through the same decision point
                                decision_points[in_arc.source.name].add(current.name)
                            else:
                                decision_points[in_arc.source.name] = {current.name}
                        decision_points, not_found, previous_reached = get_dp_to_previous_event(previous, inner_in_arc.source, loops, common_loops, decision_points, reachability, passed_inv_act)
                        # add the silent transition to the alread seen list
                        passed_inv_act.add(inner_in_arc.source.name)
                        #breakpoint()
                    elif inner_in_arc.source.name == previous:
                        previous_reached = True
                        if len(in_arc.source.out_arcs) > 1:
                            if in_arc.source.name in decision_points.keys():
                                # using set ecause it is possible to pass multiple time through the same decision point
                                decision_points[in_arc.source.name].add(current.name)
                            else:
                                decision_points[in_arc.source.name] = {current.name}
                # if not found, delete the target activity added at the begininng for the considered decision point
                # TODO check not found for each loop separately
                # TODO check this condition
                if not_found:
                    #if current.name in decision_points[in_arc.source.name]:
                    #    decision_points[in_arc.source.name].remove(current.name)
                    continue
            # if i finished to check inner arcs and there is at least one previous reached: previous_reached is TRue
            for inner_in_arc in in_arc.source.in_arcs:
                if inner_in_arc.source.name in passed_inv_act.keys():
                    if passed_inv_act[inner_in_arc.source.name]['previous_reached']:
                        previous_reached = True
                        not_found = False
        # if previous in the inputs, stop
        elif previous in inner_in_arcs_names:
            previous_reached = True
            if len(in_arc.source.out_arcs) > 1:
                if in_arc.source.name in decision_points.keys():
                    # using set ecause it is possible to pass multiple time through the same decision point
                    decision_points[in_arc.source.name].add(current.name)
                else:
                    decision_points[in_arc.source.name] = {current.name}
        else:
            not_found = True
            if in_arc.source.name in decision_points.keys():
                if current.name in decision_points[in_arc.source.name] and current.label is None:
                    decision_points[in_arc.source.name].remove(current.name)
        # stop if we find the target in one of the places going into a transition (they are parallel branches)
        if previous_reached:
            break

    for in_arc in current.in_arcs:
        if in_arc.source.name in passed_inv_act.keys():
            if passed_inv_act[in_arc.source.name]['previous_reached']:
                previous_reached = True
                not_found = False
    return decision_points, not_found, previous_reached

def print_matrix(matrix, row_names, col_names):
    row_cols_name = "   "
    for col_name in col_names:
        row_cols_name = "{}     {}".format(row_cols_name, col_name)
    print(row_cols_name)
    for i, row_name in enumerate(row_names):
        row = row_name
        for j, row_col in enumerate(col_names):
            if j == 0:
                row = "{}   {}".format(row, matrix[i, j])
            else:
                row = "{}     {}".format(row, matrix[i, j])
        print(row)

def get_map_place_to_events(net, loops) -> dict:
    """ Gets a mapping of decision point and their target transitions

    Given a Petri Net in the implementation of Pm4Py library and the loops inside,
    computes the target transtion for every decision point
    (i.e. a place with more than one out arcs). If a target is a silent transition,
    the next not silent transitions are taken as added targets, following rules regarding loops if present.
    """
    # initialize
    places = dict()
    for place in net.places:
        if len(place.out_arcs) >= 2:
            # dictionary containing for every decision point target categories
            places[place.name] = dict()
            # loop for out arcs
            for arc in place.out_arcs:
                # check if silent
                if not arc.target.label is None:
                    places[place.name][arc.target.name] = arc.target.label
                else:
                    # search for next not silent from the next places
                    silent_out_arcs = arc.target.out_arcs
                    next_not_silent = set()
                    for silent_out_arc in silent_out_arcs:
                        loop_name = 'None'
                        is_input = False
                        is_output = False
                        for loop in loops:
                            if loop.is_vertex_in_loop(place.name):
                                next_place_silent = silent_out_arc.target
                                next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, loops, [place.name], loop.name)
                            else:
                                next_place_silent = silent_out_arc.target
                                next_not_silent = get_next_not_silent(next_place_silent, next_not_silent, loops, [place.name], 'None')
                    # remove not silent transitions impossible to reach without activating a loop
                    next_not_silent_to_be_removed = set()
                    for not_silent in next_not_silent:
                        if not check_if_reachable_without_loops(loops, arc.target, not_silent, False):
                            next_not_silent_to_be_removed.add(not_silent)
                    # remove all transitions inside the loop if the place is an output place of the loop
                    for loop in loops:
                        if loop.is_vertex_output_loop(place.name):
                            next_not_silent_to_be_removed = next_not_silent_to_be_removed.union(set(loop.events))
                    places[place.name][arc.target.name] = next_not_silent.difference(next_not_silent_to_be_removed)
    return places

def get_next_not_silent(place, not_silent, loops, start_places, loop_name_start) -> list:
    """ Recursively compute the first not silent transition connected to a place

    Given a place and a list of not silent transition (i.e. without label) computes
    the next not silent transitions in order to correctly characterize the path through
    the considered place. The algorithm stops in presence of a joint-node (if not in a loop)
    or when all of the output transitions are not silent. If at least one transition is
    silent, the algorithm computes recursively the next not silent.
    """
    # first stop condition: joint node
    if place.name == 'sink':
        return not_silent
    is_input = None
    for loop in loops:
        if is_input == loop.is_vertex_input_loop(place.name):
            is_input = True
    is_input_a_skip = check_if_skip(place)
    if (len(place.in_arcs) > 1 and not (is_input or is_input_a_skip)) or place.name in start_places:
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    # second stop condition: all not silent outputs
    if not None in out_arcs_label:
        not_silent = not_silent.union(set(out_arcs_label))
        return not_silent
    for out_arc in place.out_arcs:
        # add activity if not silent
        if not out_arc.target.label is None:
            not_silent.add(out_arc.target.label)
        else:
            # recursive part if is silent
            for out_arc_inn in out_arc.target.out_arcs:
                added_start_place = False
                for loop in loops:
                    # add start place if it is an input node not already seen
                    if loop.is_vertex_input_loop(out_arc_inn.target.name) and not out_arc_inn.target.name in start_places:
                        start_places.append(out_arc_inn.target.name)
                        added_start_place = True
                # if the place is an input to another loop (nested loops), we propagate inside the other loop too
                if added_start_place:
                    for out_arc_inner_loop in out_arc_inn.target.out_arcs:
                        if not out_arc_inner_loop.target.label is None:
                            not_silent.add(out_arc_inner_loop.target.label)
                        else:
                            for next_out_arcs in out_arc_inner_loop.target.out_arcs:
                                next_place_silent = next_out_arcs.target
                                not_silent = get_next_not_silent(next_place_silent, not_silent, loops, start_places, loop_name_start)
                else:
                    not_silent = get_next_not_silent(out_arc_inn.target, not_silent, loops, start_places, loop_name_start)
    return not_silent

def get_place_from_event(places_map, event, dp_list) -> list:
    """ Returns the places that are decision points of a certain transition

    Given the dictionary mapping every decision point with its reference event(s),
    returns the list of decision point referred to the input event
    """

    places = list()
    for place in dp_list:
        for trans in places_map[place].keys():
            if event in places_map[place][trans]:
                places.append((place, trans))
    return places


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

def extract_rules(dt, feature_names) -> dict:
    """ Returns a readable version of a decision tree rules

    Given a sklearn decision tree object and the name of the features used, interprets
    the output text of export_text (sklearn function) in order to give the user a more
    human readable version of the rules defining the decision tree behavior. The output
    is a dictionary using as key the target class (leaves) of the decision tree.
    """
    text_rules = export_text(dt)
    for feature_name in feature_names.keys():
            text_rules = text_rules.replace(feature_names[feature_name], feature_name)
    text_rules = text_rules.split('\n')[:-1]
    extracted_rules = dict()
    one_complete_pass = ""
    tree_level = 0
    for text_rule in text_rules:
        single_rule = text_rule.split('|')[1:]
        if '---' in single_rule[0]:
            one_complete_pass = single_rule[0].split('--- ')[1]
        else:
            if 'class' in text_rule:
                label_name = text_rule.split(': ')[-1]
                if label_name in extracted_rules.keys():
                    extracted_rules[label_name].append(one_complete_pass)
                else:
                    extracted_rules[label_name] = list()
                    extracted_rules[label_name].append(one_complete_pass)
                reset_level_rule = one_complete_pass.split(' & ')
                if len(reset_level_rule) > 1:
                    one_complete_pass = reset_level_rule[:-1][0]
            else:
                single_rule = text_rule.split('|--- ')[1]
                one_complete_pass = "{} & {}".format(one_complete_pass, single_rule)
                tree_level += 1
    return extracted_rules

def get_feature_names(dataset) -> dict:
    """
    Returns a dictionary containing the numerated name of the features
    """
    if not isinstance(dataset, pd.DataFrame):
        raise Exception("Not a dataset object")
    features = dict()
    for index, feature in enumerate(dataset.drop(columns=['target']).columns):
        if not feature == 'target':
            features[feature] = "feature_{}".format(index)
    return features

def get_map_transitions_events(net) -> dict:
    """ Compute a map of transitions name and events

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every event the corresponding transition name
    """
    # initialize
    map_trans_events = dict()
    for trans in net.transitions:
        map_trans_events[trans.label] = trans.name
    map_trans_events['None'] = 'None'
    return map_trans_events

def get_map_events_transitions(net) -> dict:
    """ Compute a map of event name and transitions name

    Given a Petri Net in the implementation of Pm4Py library, the function creates
    a dictionary containing for every transition the corresponding event name
    """
    # initialize
    map_events_trans= dict()
    for trans in net.transitions:
        if not trans.label is None:
            map_events_trans[trans.name] = trans.label
    return map_events_trans

def check_if_reachable_without_loops(loops, start_trans, end_trans, reachable) -> bool:
    """ Check if a transition is reachable from another one without using loops

    Given a list of Loop ojects, recursively check if a transition is reachable from another one without looping
    and using only invisible transitions. This means that if a place is an output place for a loop, the algorithm
    chooses to follow the path exiting the loop
    """
    for out_arc in start_trans.out_arcs:
        if out_arc.target.name == "sink":
            return reachable
        else:
            is_a_loop_output = False
            loop_name = 'None'
            # check if it is an output
            for loop in loops:
                if loop.is_vertex_output_loop(out_arc.target.name):
                    is_a_loop_output = True
                    loop_name = loop.name
                    loop_selected = loop
            for out_arc_inn in out_arc.target.out_arcs:
                # if it is invisible and not output
                if out_arc_inn.target.label is None and not is_a_loop_output:
                    reachable = check_if_reachable_without_loops(loops, out_arc_inn.target, end_trans, reachable)
                # if it is invisible and the next vertex is not in the same loop
                elif out_arc_inn.target.label is None and is_a_loop_output and not out_arc_inn.target.name in loop_selected.vertex:
                    reachable = check_if_reachable_without_loops(loops, out_arc_inn.target, end_trans, reachable)
                else:
                    if out_arc_inn.target.label == end_trans:
                        reachable = True
                if reachable:
                    break
        if reachable:
            break
    return reachable

def update_places_map_dp_list_if_looping(net, dp_list, places_map, loops, event_sequence, number_of_loops, trans_events_map) -> [dict, list]:
    """ Updates the map of transitions related to a decision point in case a loop is active

    Given an event sequence, checks if there are active loops (i.e. the sequence is reproducible only
    passing two times from an input place. In case of acitve loops, in what part of the loop is located the current activity
    and assigns to the decision points the transitions that identify if the path passed through that decision point.
    For example if a loop is:
           -   A    -           -   B    -
    dp0 - |          | - dp1 - |          | - dp2 -
    |       - silent -           - silent -        |
    |                                              |
    ------------------- silent --------------------
    and the sequence is A - B - A, it means that the decision points to be added at the moment of the second A are:
    - from A to the start (forward): dp2
    - from the start to A (backward): dp0, dp1
    """
    transition_sequence = [trans_events_map[event] for event in event_sequence]
    loop_name = 'None'
    for loop in loops:
        loop.check_if_loop_is_active(net, transition_sequence)
        if loop.is_active():
            # update the dictionary only if the loop is actually "looping°
            # (i.e. the number of cycle is growing) -> because a decision point can be
            # chosen only one time per cycle
            loop.count_number_of_loops(net, transition_sequence.copy())
            if number_of_loops[loop.name] < loop.number_of_loops:
                number_of_loops[loop.name] = loop.number_of_loops
                # update decision points in the forward path
                for dp in loop.dp_forward.keys():
                    if not dp in dp_list:
                        dp_list.append(dp)
                    loop_reachable = loop.dp_forward.copy()
                    for trans in places_map[dp].keys():
                        if loop.is_vertex_in_loop(trans):
                            if trans in loop.dp_forward[dp].keys():
                                if event_sequence[-1] in loop.dp_forward[dp][trans]:
                                    # update only if the decision point is connected to an invisible activity
                                    # (i.e. related to a set of activities not just one)
                                    if isinstance(places_map[dp][trans], set):
                                        places_map[dp][trans] = places_map[dp][trans].difference(loop.events)
                                        places_map[dp][trans] = places_map[dp][trans].union(loop_reachable[dp][trans])
                # update decision points in the backward path
                for dp in loop.dp_backward.keys():
                    if not dp in dp_list:
                        dp_list.append(dp)
                    loop_reachable = loop.dp_backward.copy()
                    for trans in places_map[dp].keys():
                        if loop.is_vertex_in_loop(trans):
                            if trans in loop.dp_backward[dp].keys():
                                if event_sequence[-1] in loop.dp_backward[dp][trans]:
                                    # update only if the decision point is connected to an invisible activity
                                    # (i.e. related to a set of activities not just one)
                                    if isinstance(places_map[dp][trans], set):
                                        places_map[dp][trans] = places_map[dp][trans].difference(loop.events)
                                        places_map[dp][trans] = places_map[dp][trans].union(loop_reachable[dp][trans])
    return places_map, dp_list

def update_dp_list(places_from_event, dp_list) -> list:
    """ Updates the list of decision points if related with the event

    Removes the decision points related to the event in input because a decision point
    can be chosen only once
    """
    for place, trans_name in places_from_event:
        if place in dp_list:
            dp_list.remove(place)
    return dp_list


def get_all_dp_from_event_to_sink(transition, sink, decision_points_seen) -> dict:
    """ Returns all the decision points in the path from the 'transition' activity to the sink of the Petri net, passing
    only through invisible transitions.

    Starting from the sink, extracts all the transitions connected to the sink (the ones immediately before the sink).
    If 'transition' is one of them, there are no decision points to return, so it returns an empty dictionary.
    Otherwise, for each invisible transition among them, it calls method '_new_get_dp_to_previous_event' to retrieve
    all the decision points and related targets between 'transition' and the invisible transition currently considered.
    Discovered decision points for all the backward paths are put in the same 'decision_points' dictionary.
    """

    dp_seen = set()
    for event_key in decision_points_seen:
        for dp_key in decision_points_seen[event_key]:
            dp_seen.add(dp_key)

    sink_in_acts = [in_arc.source for in_arc in sink.in_arcs]
    if transition in sink_in_acts:
        return dict()
    else:
        decision_points = dict()

        for sink_in_act in sink_in_acts:
            if sink_in_act.label is None:
                decision_points = _get_dp_to_previous_event_from_sink(transition, sink_in_act, dp_seen, decision_points)

                # ---------- ONLY SHORTEST PATH APPROACH (comment method call above) ----------
                '''
                paths = get_all_paths(transition.name, sink_in_act)
                min_length_value = min(len(paths[k]) for k in paths)
                min_length_path = [l for k, l in paths.items() if len(l) == min_length_value][0]
                decision_points = dict(min_length_path)
                old_keys = list(decision_points.keys()).copy()
                for dp in old_keys:
                    if len(dp.out_arcs) < 2:
                        del decision_points[dp]
                    else:
                        decision_points[dp.name] = decision_points.pop(dp)
                '''
                # ---------- ONLY SHORTEST PATH APPROACH (comment method call above) ----------

        return decision_points


def _get_dp_to_previous_event_from_sink(previous, current, dp_seen, decision_points=None, passed_inn_arcs=None) -> dict:

    if decision_points is None:
        decision_points = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    for in_arc in current.in_arcs:
        # If decision point already seen in variant, stop following this path
        if in_arc.source.name in dp_seen:
            continue

        for inner_in_arc in in_arc.source.in_arcs:
            # If invisible activity backward, recurse only if 'inner_in_arc' has not been seen in this path yet
            if inner_in_arc.source.label is None:
                if inner_in_arc not in passed_inn_arcs:
                    passed_inn_arcs.add(inner_in_arc)
                    decision_points= _get_dp_to_previous_event_from_sink(previous, inner_in_arc.source, dp_seen,
                                                                         decision_points, passed_inn_arcs)
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
                    passed_inn_arcs.remove(inner_in_arc)
                else:
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
            # Otherwise, just add the decision point and its target activity
            else:
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)

    return decision_points


def check_if_skip(place) -> bool:
    """ Checks if a place is a 'skip'

    A place is a 'skip' if has N input arcs of which only one is an invisible activity and all
    are coming from the same place
    """
    #breakpoint()
    is_skip = False
    if len(place.in_arcs) > 1 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1:
        source_name = None
        source_name_count = 0
        for arc in place.in_arcs:
            if len(arc.source.in_arcs) == 1:
                for source_arc in arc.source.in_arcs:
                    if source_name is None:
                        source_name = source_arc.source.name
                    elif source_arc.source.name == source_name:
                        source_name_count += 1
        if source_name_count == len(place.in_arcs) - 1:
            is_skip = True
    return is_skip

def check_if_reducible(node):
    #breakpoint()
    is_reducible = False
    if len(node.in_arcs) > 1:
        source_name = None
        source_name_count = 0
        # check if the node incoming arcs are the same number as the outcoming from the previous one of the same type
        for arc in node.in_arcs:
            if len(arc.source.in_arcs) == 1:
                for source_arc in arc.source.in_arcs:
                    # check if the node has only the output going to the initial one
                    if len(source_arc.target.out_arcs) == 1:
                        if source_name is None:
                            source_name = source_arc.source.name
                            source_name_count = 1
                        elif source_arc.source.name == source_name:
                            source_name_count += 1
        if source_name_count == len(node.in_arcs) and source_name_count != 0:
            is_reducible = True
    # if there is only one incoming arcs, only the configuration place -> trans -> place is reducible
    elif len(node.in_arcs) == 1 and isinstance(node, PetriNet.Place):
        for arc in node.in_arcs:
            if len(arc.source.in_arcs) == 1:
                for arc_inner in arc.source.in_arcs:
                    if len(arc_inner.source.out_arcs) == 1:
                        is_reducible = True
    return is_reducible

def get_previous_same_type(node):
    #breakpoint()
    if len(node.in_arcs) > 0:
        previous = list(node.in_arcs)[0]
        if len(previous.source.in_arcs) > 1:
            print("Can't find a unique previous node of the same type of {}".format(node))
            previous_same_type = None
        elif len(previous.source.in_arcs) == 0:
            print("Previous node of the same type of {} not present".format(node))
            previous_same_type = None
        else:
            previous_same_type = list(previous.source.in_arcs)[0].source
    else:
        previous_same_type = None
    return previous_same_type

def get_previous(node):
    #breakpoint()
    previous = list(node.in_arcs)[0].source
    return previous


def discover_overlapping_rules(base_tree, dataset, attributes_map, original_rules):
    """ Discovers overlapping rules, if any.

    Given the fitted decision tree, extracts the training set instances that have been wrongly classified, i.e., for
    each leaf node, all those instances whose target is different from the leaf label. Then, it fits a new decision tree
    on those instances, builds a rules dictionary as before (disjunctions of conjunctions) and puts the resulting rules
    in disjunction with the original rules, according to the target value.
    Method taken by "Decision Mining Revisited - Discovering Overlapping Rules" by Felix Mannhardt, Massimiliano de
    Leoni, Hajo A. Reijers, Wil M.P. van der Aalst (2016).
    """

    rules = copy.deepcopy(original_rules)

    leaf_nodes = base_tree.get_leaves_nodes()
    leaf_nodes_with_wrong_instances = [ln for ln in leaf_nodes if len(ln.get_class_names()) > 1]

    for leaf_node in leaf_nodes_with_wrong_instances:
        vertical_rules = extract_rules_from_leaf(leaf_node)

        vertical_rules_query = ""
        for r in vertical_rules:
            r_attr, r_comp, r_value = r.split(' ')
            vertical_rules_query += r_attr
            if r_comp == '=':
                vertical_rules_query += ' == '
            else:
                vertical_rules_query += ' ' + r_comp + ' '
            if dataset.dtypes[r_attr] == 'float64' or dataset.dtypes[r_attr] == 'bool':
                vertical_rules_query += r_value
            else:
                vertical_rules_query += '"' + r_value + '"'
            if r != vertical_rules[-1]:
                vertical_rules_query += ' & '

        leaf_instances = dataset.query(vertical_rules_query)
        # TODO not considering missing values for now, so wrong_instances could be empty
        # This happens because all the wrongly classified instances have missing values for the query attribute(s)
        wrong_instances = (leaf_instances[leaf_instances['target'] != leaf_node._label_class]).copy()

        sub_tree = DecisionTree(attributes_map)
        sub_tree.fit(wrong_instances)

        sub_leaf_nodes = sub_tree.get_leaves_nodes()
        if len(sub_leaf_nodes) > 1:
            sub_rules = {}
            for sub_leaf_node in sub_leaf_nodes:
                new_rule = ' && '.join(vertical_rules + extract_rules_from_leaf(sub_leaf_node))
                if sub_leaf_node._label_class not in sub_rules.keys():
                    sub_rules[sub_leaf_node._label_class] = set()
                sub_rules[sub_leaf_node._label_class].add(new_rule)
            for sub_target_class in sub_rules.keys():
                sub_rules[sub_target_class] = ' || '.join(sub_rules[sub_target_class])
                if sub_target_class not in rules.keys():
                    rules[sub_target_class] = sub_rules[sub_target_class]
                else:
                    rules[sub_target_class] += ' || ' + sub_rules[sub_target_class]
        # Only root in sub_tree = could not find a suitable split of the root node -> most frequent target is chosen
        elif len(wrong_instances) > 0:  # length 0 could happen since we do not consider missing values for now
            sub_target_class = wrong_instances['target'].mode()[0]
            if sub_target_class not in rules.keys():
                rules[sub_target_class] = ' && '.join(vertical_rules)
            else:
                rules[sub_target_class] += ' || ' + ' && '.join(vertical_rules)

    return rules


def shorten_rules_manually(original_rules, attributes_map):
    """ Rewrites the final rules dictionary to compress many-valued categorical attributes equalities and continuous
    attributes inequalities.

    For example, the series "org:resource = 10 && org:resource = 144 && org:resource = 68" is rewritten as "org:resource
    one of [10, 68, 144]".
    The series "paymentAmount > 21.0 && paymentAmount <= 37.0 && paymentAmount <= 200.0 && amount > 84.0 && amount <=
    138.0 && amount > 39.35" is rewritten as "paymentAmount > 21.0 && paymentAmount <= 37.0 && amount <= 138.0 && amount
    84.0".
    The same reasoning is applied for atoms without '&&s' inside.
    """

    rules = copy.deepcopy(original_rules)

    for target_class in rules.keys():
        or_atoms = rules[target_class].split(' || ')
        new_target_rule = list()
        cat_atoms_same_attr_noand = dict()
        cont_atoms_same_attr_less_noand, cont_atoms_same_attr_greater_noand = dict(), dict()
        cont_comp_less_equal_noand, cont_comp_greater_equal_noand = dict(), dict()

        for or_atom in or_atoms:
            if ' && ' in or_atom:
                and_atoms = or_atom.split(' && ')
                cat_atoms_same_attr = dict()
                cont_atoms_same_attr_less, cont_atoms_same_attr_greater = dict(), dict()
                cont_comp_less_equal, cont_comp_greater_equal = dict(), dict()
                new_or_atom = list()

                for and_atom in and_atoms:
                    a_attr, a_comp, a_value = and_atom.split(' ')
                    # Storing information for many-values categorical attributes equalities
                    if attributes_map[a_attr] == 'categorical' and a_comp == '=':
                        if a_attr not in cat_atoms_same_attr.keys():
                            cat_atoms_same_attr[a_attr] = list()
                        cat_atoms_same_attr[a_attr].append(a_value)
                    # Storing information for continuous attributes inequalities (min/max value for each attribute and
                    # also if the inequality is strict or not)
                    elif attributes_map[a_attr] == 'continuous':
                        if a_comp in ['<', '<=']:
                            if a_attr not in cont_atoms_same_attr_less.keys() or float(a_value) <= float(cont_atoms_same_attr_less[a_attr]):
                                cont_atoms_same_attr_less[a_attr] = a_value
                                cont_comp_less_equal[a_attr] = True if a_comp == '<=' else False
                        elif a_comp in ['>', '>=']:
                            if a_attr not in cont_atoms_same_attr_greater.keys() or float(a_value) >= float(cont_atoms_same_attr_greater[a_attr]):
                                cont_atoms_same_attr_greater[a_attr] = a_value
                                cont_comp_greater_equal[a_attr] = True if a_comp == '>=' else False
                    else:
                        new_or_atom.append(and_atom)

                # Compressing many-values categorical attributes equalities
                for attr in cat_atoms_same_attr.keys():
                    if len(cat_atoms_same_attr[attr]) > 1:
                        new_or_atom.append(attr + ' one of [' + ', '.join(sorted(cat_atoms_same_attr[attr])) + ']')
                    else:
                        new_or_atom.append(attr + ' = ' + cat_atoms_same_attr[attr][0])

                # Compressing continuous attributes inequalities (< / <= and then > / >=)
                for attr in cont_atoms_same_attr_less.keys():
                    min_value = cont_atoms_same_attr_less[attr]
                    comp = ' <= ' if cont_comp_less_equal[attr] else ' < '
                    new_or_atom.append(attr + comp + min_value)

                for attr in cont_atoms_same_attr_greater.keys():
                    max_value = cont_atoms_same_attr_greater[attr]
                    comp = ' >= ' if cont_comp_greater_equal[attr] else ' > '
                    new_or_atom.append(attr + comp + max_value)

                # Or-atom analyzed: putting its new and-atoms in conjunction
                new_target_rule.append(' && ' .join(new_or_atom))

            # If the or_atom does not have &&s inside (single atom), just simplify attributes.
            # For example, the series "org:resource = 10 || org:resource = 144 || org:resource = 68" is rewritten as
            # "org:resource one of [10, 68, 144]". For continuous attributes, follows the same reasoning as before.
            else:
                a_attr, a_comp, a_value = or_atom.split(' ')
                # Storing information for many-values categorical attributes equalities
                if attributes_map[a_attr] == 'categorical' and a_comp == '=':
                    if a_attr not in cat_atoms_same_attr_noand.keys():
                        cat_atoms_same_attr_noand[a_attr] = list()
                    cat_atoms_same_attr_noand[a_attr].append(a_value)
                elif attributes_map[a_attr] == 'continuous':
                    if a_comp in ['<', '<=']:
                        if a_attr not in cont_atoms_same_attr_less_noand.keys() or float(a_value) <= float(cont_atoms_same_attr_less_noand[a_attr]):
                            cont_atoms_same_attr_less_noand[a_attr] = a_value
                            cont_comp_less_equal_noand[a_attr] = True if a_comp == '<=' else False
                    elif a_comp in ['>', '>=']:
                        if a_attr not in cont_atoms_same_attr_greater_noand.keys() or float(a_value) >= float(cont_atoms_same_attr_greater_noand[a_attr]):
                            cont_atoms_same_attr_greater_noand[a_attr] = a_value
                            cont_comp_greater_equal_noand[a_attr] = True if a_comp == '>=' else False
                else:
                    new_target_rule.append(or_atom)

        # Compressing many-values categorical attributes equalities for the 'no &&s' case
        for attr in cat_atoms_same_attr_noand.keys():
            if len(cat_atoms_same_attr_noand[attr]) > 1:
                new_target_rule.append(attr + ' one of [' + ', '.join(sorted(cat_atoms_same_attr_noand[attr])) + ']')
            else:
                new_target_rule.append(attr + ' = ' + cat_atoms_same_attr_noand[attr][0])

        # Compressing continuous attributes inequalities (< / <= and then > / >=) for the 'no &&s' case
        for attr in cont_atoms_same_attr_less_noand.keys():
            min_value = cont_atoms_same_attr_less_noand[attr]
            comp = ' <= ' if cont_comp_less_equal_noand[attr] else ' < '
            new_target_rule.append(attr + comp + min_value)

        for attr in cont_atoms_same_attr_greater_noand.keys():
            max_value = cont_atoms_same_attr_greater_noand[attr]
            comp = ' >= ' if cont_comp_greater_equal_noand[attr] else ' > '
            new_target_rule.append(attr + comp + max_value)

        # Rule for a target class analyzed: putting its new or-atoms in disjunction
        rules[target_class] = ' || '.join(new_target_rule)

    # Rules for all target classes analyzed: returning the new rules dictionary
    return rules


def sampling_dataset(dataset) -> pd.DataFrame:
    """ Performs sampling to obtain a balanced dataset, in terms of target values. """

    dataset = dataset.copy()

    groups = list()
    grouped_df = dataset.groupby('target')
    for target_value in dataset['target'].unique():
        groups.append(grouped_df.get_group(target_value))
    groups.sort(key=len)
    # Groups is a list containing a dataset for each target value, ordered by length
    # If the smaller datasets are less than the 35% of the total dataset length, then apply the sampling
    if sum(len(group) for group in groups[:-1]) / len(dataset) <= 0.35:
        samples = list()
        # Each smaller dataset is appended to the 'samples' list, along with a sampled dataset from the largest one
        for group in groups[:-1]:
            samples.append(group)
            samples.append(groups[-1].sample(len(group)))
        # The datasets in the 'samples' list are then concatenated together
        dataset = pd.concat(samples, ignore_index=True)

    return dataset
