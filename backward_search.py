def get_decision_points_and_targets(sequence, net, stored_dicts) -> [dict, dict]:
    """ Returns a dictionary containing decision points and their targets.

    Starting from the last activity in the sequence, the algorithm selects the previous not
    parallel activity. Exploring the net backward, it returns all the decision points with their targets
    encountered on the path leading to the last activity in the sequence.
    """

    # Current activity
    current_act = [trans for trans in net.transitions if trans.name == sequence[-1]][0]

    # Backward decision points search towards the previous reachable activity
    dp_dict = dict()
    counter = 0
    for previous_act in reversed(sequence[:-1]):
        prev_curr_key = ', '.join([previous_act, current_act.name])
        if prev_curr_key not in stored_dicts.keys():
            counter += 1
            dp_dict, event_found, counter = _backward_depth_first_search(previous_act, current_act, counter)
            if event_found:
                # ------------------------------ DEBUGGING ------------------------------
                # from utils import get_map_events_transitions
                # events_trans_map = get_map_events_transitions(net)
                # print("\nPrevious event: {}".format(events_trans_map[previous_act]))
                # print("Current event: {}".format(current_act.label))
                # print("DPs")
                # for key in dp_dict.keys():
                #     print(" - {}".format(key))
                #     for inn_key in dp_dict[key]:
                #         if inn_key in events_trans_map.keys():
                #             print("   - {}".format(events_trans_map[inn_key]))
                #         else:
                #             print("   - {}".format(inn_key))
                # ------------------------------ DEBUGGING ------------------------------
                stored_dicts[prev_curr_key] = dp_dict
                break
            elif current_act.name == previous_act:
                break
        else:
            dp_dict = stored_dicts[prev_curr_key]
            break

    return dp_dict, stored_dicts, counter


def _backward_depth_first_search(previous, current, counter, decision_points=None, passed_inn_arcs=None) -> tuple[dict, bool, int]:
    """ Extracts all the decision points that are traversed between two activities (previous and current), reporting the
    decision(s) that has been taken for each of them.

    Starting from the 'current' activity, the algorithm proceeds backwards on each incoming arc (in_arc). Then, it saves
    the so-called 'inner_arcs', which are the arcs between the previous place (in_arc.source) and its previous
    activities (in_arc.source.in_arcs).
    If the 'previous' activity is immediately before the previous place (so it is the source of one of the inner_arcs),
    then the algorithm sets a boolean variable 'target_found' to True to signal that the target has been found, and it
    adds the corresponding decision point (in_arc.source) to the dictionary.
    In any case, all the other backward paths containing an invisible activity are explored recursively.
    Every time an 'inner_arc' is traversed for exploration, it is added to the 'passed_inn_arcs' list, to avoid looping
    endlessly. Indeed, before a new recursion on an 'inner_arc', the algorithm checks if it is present in that list: if
    it is not present, it simply goes on with the recursion, since it means that the specific path has not been explored
    yet.
    Note that decision points are then added to the dictionary in a forward way: whenever the recursion cannot go on (no
    more invisible activities backward) it returns also signalling the 'target_found' value. This is True if the current
    path found the target activity (or an already explored 'inner_arc' during the same path, as explained before). The
    returned value is then put in disjunction with the current value of 'target_found' in case the target activity has
    been found by the actual instance and not by the recursive one.
    It returns a 'decision_points' dictionary which contains, for each decision point on every possible path between
    the 'current' activity and the 'previous' activity, the target value(s) to be followed.
    """

    if decision_points is None:
        decision_points = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    target_found = False
    for in_arc in current.in_arcs:
        counter += 1
        # Preparing the lists containing inner_arcs towards invisible and non-invisible transitions
        inner_inv_acts, inner_in_arcs_names = set(), set()
        for inner_in_arc in in_arc.source.in_arcs:
            counter += 1
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
                decision_points, previous_found, counter = _backward_depth_first_search(previous, inner_in_arc.source, counter,
                                                                               decision_points, passed_inn_arcs)
                passed_inn_arcs.remove(inner_in_arc)
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, previous_found)
                target_found = target_found or previous_found

    return decision_points, target_found, counter


def _add_dp_target(decision_points, dp, target, add_dp) -> dict:
    """ Adds the decision point and its target activity to the 'decision_points' dictionary.

    Given the 'decision_points' dictionary, the place 'dp' and the target activity name, adds the target activity name
    to the set of targets related to the decision point. If not present, adds the decision point to the dictionary keys
    first. This is done if the place is an actual decision point, and if the boolean variable 'add_dp' is True.
    """

    if add_dp and len(dp.out_arcs) > 1:
        if dp.name in decision_points.keys():
            decision_points[dp.name].add(target)
        else:
            decision_points[dp.name] = {target}
    return decision_points


def get_all_dp_from_sink_to_last_event(transition, sink, decision_points_seen) -> tuple[dict, int]:
    """ Returns all the decision points in the path from the 'transition' activity to the sink of the Petri net, passing
    only through invisible transitions.

    Starting from the sink, extracts all the transitions connected to the sink (the ones immediately before the sink).
    If 'transition' is one of them, there are no decision points to return, so it returns an empty dictionary.
    Otherwise, for each invisible transition among them, it calls method '_new_get_dp_to_previous_event' to retrieve
    all the decision points and related targets between 'transition' and the invisible transition currently considered.
    Discovered decision points for all the backward paths are put in the same 'decision_points' dictionary."""
    dp_seen = set()
    for event_key in decision_points_seen:
        for dp_key in decision_points_seen[event_key]:
            dp_seen.add(dp_key)

    sink_in_acts = [in_arc.source for in_arc in sink.in_arcs]
    counter = len(sink_in_acts)
    if transition in sink_in_acts:
        return dict(), counter
    else:
        decision_points = dict()

        for sink_in_act in sink_in_acts:
            if sink_in_act.label is None:
                counter += 1
                decision_points, counter = _backward_depth_first_search_from_sink(transition, sink_in_act, dp_seen, counter,
                                                                                  decision_points)
        return decision_points, counter


def _backward_depth_first_search_from_sink(previous, current, dp_seen, counter, decision_points=None, passed_inn_arcs=None) -> tuple[dict, int]:
    if decision_points is None:
        decision_points = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    for in_arc in current.in_arcs:
        counter += 1
        # If decision point already seen in variant, stop following this path
        if in_arc.source.name in dp_seen:
            continue

        for inner_in_arc in in_arc.source.in_arcs:
            counter += 1
            # If invisible activity backward, recurse only if 'inner_in_arc' has not been seen in this path yet
            if inner_in_arc.source.label is None:
                if inner_in_arc not in passed_inn_arcs:
                    passed_inn_arcs.add(inner_in_arc)
                    decision_points, counter = _backward_depth_first_search_from_sink(previous, inner_in_arc.source, dp_seen, counter,
                                                                             decision_points, passed_inn_arcs)
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
                    passed_inn_arcs.remove(inner_in_arc)
                else:
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
            # Otherwise, just add the decision point and its target activity
            else:
                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)

    return decision_points, counter


def get_all_dp_to_source_from_first_event(transition) -> tuple[dict, int]:
    """ Returns all the decision points in the path from the 'transition' activity to the sink of the Petri net, passing
    only through invisible transitions.

    Starting from the sink, extracts all the transitions connected to the sink (the ones immediately before the sink).
    If 'transition' is one of them, there are no decision points to return, so it returns an empty dictionary.
    Otherwise, for each invisible transition among them, it calls method '_new_get_dp_to_previous_event' to retrieve
    all the decision points and related targets between 'transition' and the invisible transition currently considered.
    Discovered decision points for all the backward paths are put in the same 'decision_points' dictionary."""

    decision_points = dict()
    passed_inn_arcs = set()
    target_found = False
    counter = 0
    for in_arc in transition.in_arcs:
        counter += 1
        if in_arc.source.name == 'source':
            target_found = True
            decision_points = _add_dp_target(decision_points, in_arc.source, transition.name, target_found)
        else:
            # Preparing the lists containing inner_arcs towards invisible and non-invisible transitions
            inner_inv_acts, inner_in_arcs_names = set(), set()
            for inner_in_arc in in_arc.source.in_arcs:
                counter += 1
                if inner_in_arc.source.label is None:
                    inner_inv_acts.add(inner_in_arc)
                else:
                    inner_in_arcs_names.add(inner_in_arc.source.name)

            # Base case: the target activity (previous) is one of the activities immediately before the current one
            #if previous in inner_in_arcs_names:
            #    target_found = True
            #    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, target_found)
            # Recursive case: follow every invisible activity backward
            for inner_in_arc in inner_inv_acts:
                if inner_in_arc not in passed_inn_arcs:
                    passed_inn_arcs.add(inner_in_arc)
                    decision_points, source_found, counter = _backward_depth_first_search_to_source(inner_in_arc.source, counter,
                                                                                   decision_points, passed_inn_arcs)
                    passed_inn_arcs.remove(inner_in_arc)
                    decision_points = _add_dp_target(decision_points, in_arc.source, transition.name, source_found)
                    target_found = target_found or source_found

    return decision_points, counter
#    dp_seen = set()
#    for event_key in decision_points_seen:
#        for dp_key in decision_points_seen[event_key]:
#            dp_seen.add(dp_key)
#
#    sink_in_acts = [in_arc.source for in_arc in sink.in_arcs]
#    counter = len(sink_in_acts)
#    if transition in sink_in_acts:
#        return dict(), counter
#    else:
#        decision_points = dict()
#
#        for sink_in_act in sink_in_acts:
#            if sink_in_act.label is None:
#                counter += 1
#                decision_points, counter = _backward_depth_first_search_to_source(transition, sink_in_act, dp_seen, counter,
#                                                                                  decision_points)
#        return decision_points, counter


def _backward_depth_first_search_to_source(current, counter, decision_points=None, passed_inn_arcs=None) -> tuple[dict, bool, int]:
    if decision_points is None:
        decision_points = dict()
    if passed_inn_arcs is None:
        passed_inn_arcs = set()
    target_found = False
    for in_arc in current.in_arcs:
        counter += 1
        # Preparing the lists containing inner_arcs towards invisible and non-invisible transitions
        if in_arc.source.name == 'source':
            target_found = True
            decision_points = _add_dp_target(decision_points, in_arc.source, current.name, target_found)
        else:

            inner_inv_acts, inner_in_arcs_names = set(), set()
            for inner_in_arc in in_arc.source.in_arcs:
                counter += 1
                if inner_in_arc.source.label is None:
                    inner_inv_acts.add(inner_in_arc)
                else:
                    inner_in_arcs_names.add(inner_in_arc.source.name)

            # Base case: the target activity (previous) is one of the activities immediately before the current one
    #        if previous in inner_in_arcs_names:
    #            target_found = True
    #            decision_points = _add_dp_target(decision_points, in_arc.source, current.name, target_found)
            # Recursive case: follow every invisible activity backward
            for inner_in_arc in inner_inv_acts:
                if inner_in_arc not in passed_inn_arcs:
                    passed_inn_arcs.add(inner_in_arc)
                    decision_points, previous_found, counter = _backward_depth_first_search(current, inner_in_arc.source, counter,
                                                                                   decision_points, passed_inn_arcs)
                    passed_inn_arcs.remove(inner_in_arc)
                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, previous_found)
                    target_found = target_found or previous_found

#    if decision_points is None:
#        decision_points = dict()
#    if passed_inn_arcs is None:
#        passed_inn_arcs = set()
#    for in_arc in current.in_arcs:
#        counter += 1
#        # If decision point already seen in variant, stop following this path
#        if in_arc.source.name in dp_seen:
#            continue
#
#        for inner_in_arc in in_arc.source.in_arcs:
#            counter += 1
#            # If invisible activity backward, recurse only if 'inner_in_arc' has not been seen in this path yet
#            if inner_in_arc.source.label is None:
#                if inner_in_arc not in passed_inn_arcs:
#                    passed_inn_arcs.add(inner_in_arc)
#                    decision_points, counter = _backward_depth_first_search_to_source(inner_in_arc.source, dp_seen, counter,
#                                                                             decision_points, passed_inn_arcs)
#                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
#                    passed_inn_arcs.remove(inner_in_arc)
#                else:
#                    decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)
#            # Otherwise, just add the decision point and its target activity
#            else:
#                decision_points = _add_dp_target(decision_points, in_arc.source, current.name, True)

    return decision_points, target_found, counter
