import copy
import pandas as pd
import multiprocessing
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from DecisionTreeC45 import DecisionTree
from DecisionTreeC45.Nodes import DecisionNode, LeafNode
from DecisionTreeC45.decision_tree_utils import extract_rules_from_leaf
from statsmodels.stats.proportion import proportion_confint


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


def extract_rules_with_pruning(dt, data_in) -> dict:
    """ Extracts the rules from the tree, one for each target transition.

    For each leaf node, takes the list of conditions from the root to the leaf, simplifies it if possible, and puts
    them in conjunction, adding the resulting rule to the dictionary at the corresponding target class key.
    Finally, all the rules related to different leaves with the same target class are put in disjunction.
    """

    # Starting with a p_value threshold for the Fisher's Exact Test (for rule pruning) of 0.01, create the rules
    # dictionary. If for some target all the rules have been pruned, repeat the process increasing the threshold.
    p_threshold = 0.01
    keep_rule = set()
    while p_threshold <= 1.0:
        rules = dict()

        leaves = dt.get_leaves_nodes()
        inputs = [(leaf, keep_rule, p_threshold, data_in) for leaf in leaves]
        print("Starting multiprocessing rules pruning on {} leaves...".format(str(len(leaves))))
        with multiprocessing.Pool() as pool:
            result = list(tqdm(pool.imap(_simplify_rule_multiprocess, inputs), total=len(leaves)))

        for (vertical_rules, leaf_class) in result:
            # Create the set corresponding to the target class in the rules dictionary, if not already present
            if leaf_class not in rules.keys():
                rules[leaf_class] = set()

            # If the resulting list is composed by at least one rule, put them in conjunction and add the result to
            # the dictionary of rules, at the corresponding class label
            if len(vertical_rules) > 0:
                vertical_rules = " && ".join(vertical_rules)
                rules[leaf_class].add(vertical_rules)

        # Put the rules for the same target class in disjunction. If there are no rules for some target class (they
        # have been pruned) then set the 'empty_rule' variable to True.
        empty_rule = False
        for target_class in rules.keys():
            if len(rules[target_class]) == 0:
                empty_rule = True
                break
            else:
                rules[target_class] = " || ".join(rules[target_class])

        # If 'empty_rule' is True, then increase the threshold and repeat the process. Otherwise, if two target
        # transitions have the same rule (because the original vertical rule has been pruned "too much"), repeat the
        # process but avoid simplifying the rule that originated the problem. This is done only if the 'new' rules
        # to be avoided are not all already present in 'keep_rule', otherwise it means that the process is looping.
        # If that happens, simply increase the threshold and repeat the process.
        # Otherwise, return the dictionary.
        if empty_rule:
            # TODO maybe increase more each time? This is precise but it may take long since the cap is 1
            keep_rule = set()
            p_threshold = round(p_threshold + 0.01, 2)
        elif len(rules.values()) != len(set(rules.values())):
            rules_to_add = [r for r in set(rules.values()) if list(rules.values()).count(r) > 1]
            if not all([r in keep_rule for r in rules_to_add]):
                keep_rule.update(rules_to_add)
            else:
                keep_rule = set()
                p_threshold = round(p_threshold + 0.01, 2)
        else:
            break

    return rules


def _simplify_rule_multiprocess(input):
    leaf_node, kr, p_threshold, data_in = input
    vertical_rules = extract_rules_from_leaf(leaf_node)

    # Simplify the list of rules, if possible (and if vertical_rules does not contain rules in keep_rule)
    if not any([r in vertical_rules for r in kr]):
        vertical_rules = _simplify_rule(vertical_rules, leaf_node._label_class, p_threshold, data_in)

    return vertical_rules, leaf_node._label_class


def _simplify_rule(vertical_rules, leaf_class, p_threshold, data_in) -> list:
    """ Simplifies the list of rules from the root to a leaf node.

    Given the list of vertical rules for a leaf, i.e. the list of rules from the root to the leaf node,
    drops the irrelevant rules recursively applying a Fisher's Exact Test and returns the remaining ones.
    In principle, all the rules could be removed: in that case, the result would be an empty list.
    Method taken from "Simplifying Decision Trees" by J.R. Quinlan (1986).
    """

    rules_candidates_remove = list()
    # For every rule in vertical_rules, check if it could be removed from vertical_rules.
    # This is true if the related p-value returned by the Fisher's Exact Test is higher than the threshold.
    # Indeed, a rule is considered relevant for the classification only if the null hypothesis (i.e. the two
    # variables - table rows and columns - are independent) can be rejected at the threshold*100% level or better.
    for rule in vertical_rules:
        other_rules = vertical_rules[:]
        other_rules.remove(rule)
        table = _create_fisher_table(rule, other_rules, leaf_class, data_in)
        (_, p_value) = stats.fisher_exact(table)
        if p_value > p_threshold:
            rules_candidates_remove.append((rule, p_value))

    # Among the candidates rules, remove the one with the highest p-value (the most irrelevant)
    if len(rules_candidates_remove) > 0:
        rule_to_remove = max(rules_candidates_remove, key=itemgetter(1))[0]
        vertical_rules.remove(rule_to_remove)
        # Then, recurse the process on the remaining rules
        _simplify_rule(vertical_rules, leaf_class, p_threshold, data_in)

    return vertical_rules


def _create_fisher_table(rule, other_rules, leaf_class, data_in) -> pd.DataFrame:
    """ Creates a 2x2 table to be used for the Fisher's Exact Test.

    Given a rule from the list of rules from the root to the leaf node, the other rules from that list, the leaf
    class and the training set, creates a 2x2 table containing the number of training examples that satisfy the
    other rules divided according to the satisfaction of the excluded rule and the belonging to target class.
    Missing values are not taken into account.
    """

    # Create a query string with all the rules in "other_rules" in conjunction (if there are other rules)
    # Get the examples in the training set that satisfy all the rules in other_rules in conjunction
    if len(other_rules) > 0:
        query_other = ""
        for r in other_rules:
            r_attr, r_comp, r_value = r.split(' ')
            query_other += r_attr
            if r_comp == '=':
                query_other += ' == '
            else:
                query_other += ' ' + r_comp + ' '
            if data_in.dtypes[r_attr] in ['float64', 'bool']:
                query_other += r_value
            else:
                query_other += '"' + r_value + '"'
            if r != other_rules[-1]:
                query_other += ' & '
        examples_satisfy_other = data_in.query(query_other)
    else:
        examples_satisfy_other = data_in.copy()

    # Create a query with the excluded rule
    rule_attr, rule_comp, rule_value = rule.split(' ')
    query_rule = rule_attr
    if rule_comp == '=':
        query_rule += ' == '
    else:
        query_rule += ' ' + rule_comp + ' '
    if data_in.dtypes[rule_attr] in ['float64', 'bool']:
        query_rule += rule_value
    else:
        query_rule += '"' + rule_value + '"'

    # Get the examples in the training set that satisfy the excluded rule
    examples_satisfy_other_and_rule = examples_satisfy_other.query(query_rule)

    # Get the examples in the training set that satisfy the other_rules in conjunction but not the excluded rule
    examples_satisfy_other_but_not_rule = examples_satisfy_other[
        ~examples_satisfy_other.apply(tuple, 1).isin(examples_satisfy_other_and_rule.apply(tuple, 1))]

    # Create the table which contains, for every target class and the satisfaction of the excluded rule,
    # the corresponding number of examples in the training set
    table = {k1: {k2: 0 for k2 in [leaf_class, 'not '+leaf_class]} for k1 in ['satisfies rule', 'does not satisfy rule']}

    count_other_and_rule = examples_satisfy_other_and_rule.groupby('target').count().iloc[:, 0]
    count_other_but_not_rule = examples_satisfy_other_but_not_rule.groupby('target').count().iloc[:, 0]

    for idx, value in count_other_and_rule.items():
        if idx == leaf_class:
            table['satisfies rule'][leaf_class] = value
        else:
            table['satisfies rule']['not '+leaf_class] = value
    for idx, value in count_other_but_not_rule.items():
        if idx == leaf_class:
            table['does not satisfy rule'][leaf_class] = value
        else:
            table['does not satisfy rule']['not '+leaf_class] = value

    table_df = pd.DataFrame.from_dict(table, orient='index')
    return table_df


def pessimistic_pruning(dt, data_in) -> None:
    """ Prunes the decision tree, substituting subtrees with leaves when possible.

    Given a subtree of the decision tree, computes the number of predicted errors keeping the subtree as it is.
    Then, for each target value among its leaves, computes the number of predicted errors substituting the subtree
    with a leaf having that target value. If the number of predicted errors after the substitution is lower than
    the one before the substitution, then replaces the subtree with a leaf having as target value the one which
    gave the smallest number of predicted errors.
    The procedure is repeated for every subtree in the decision tree, starting from the bottom. Every time the
    decision tree is pruned, the method is called recursively on the pruned decision tree in order to re-evaluate it
    for possibly further pruning.

    Method is taken by "Pruning Decision Trees" in "C4.5: Programs for Machine Learning" by J. R. Quinlan (1993).
    """

    subtrees_to_prune = set()
    subtrees_with_only_leaves = [node for node in dt.get_nodes() if isinstance(node, DecisionNode) and
                                 node.get_label() != 'root' and
                                 all(isinstance(child, LeafNode) for child in node.get_childs())]

    for subtree in subtrees_with_only_leaves:
        target_values = set()
        subtree_errors = 0
        # Number of predicted errors of the subtree
        for leaf in subtree.get_childs():
            subtree_errors += _compute_number_predicted_errors(leaf, leaf._label_class, data_in)
            target_values.add(leaf._label_class)

        # Number of predicted errors replacing the subtree with a leaf for every target value among its children
        tests_results = dict()
        for target_value in target_values:
            tests_results[target_value] = _compute_number_predicted_errors(subtree, target_value, data_in)

        # Storing the subtree if it needs to be pruned
        min_tests_item = min(tests_results.items(), key=lambda x: x[1])
        # TODO added or conditions: in case of 'nan', then prune
        if min_tests_item[1] < subtree_errors or np.isnan(subtree_errors) or np.isnan(min_tests_item[1]):
            subtrees_to_prune.add((subtree, min_tests_item[0]))

    # Pruning the stored subtrees
    for subtree, target_value in subtrees_to_prune:
        dt.delete_node(subtree)
        node = LeafNode({target_value: len(data_in[data_in['target'] == target_value])}, subtree.get_label(), subtree.get_level())
        dt.add_node(node, subtree.get_parent_node())

    # Recursion only if at least one subtree has been pruned
    if subtrees_to_prune:
        pessimistic_pruning(dt, data_in)


def _compute_number_predicted_errors(node, target_value, data_in) -> float:
    """ Computes the number of predicted errors given a node, a target value and the training set.

    First, it computes the number N of training instances covered by that node of the decision tree. Then, it
    isolates the ones E that have a target value different from the one prescribed by the node. Finally, it computes
    tha number of predicted errors of that node. This is computed as the product between N and the upper bound of a
    binomial distribution with N trials and E observed events.
    """

    branch_conditions = extract_rules_from_leaf(node)

    query = ""
    for r in branch_conditions:
        r_attr, r_comp, r_value = r.split(' ')
        query += r_attr
        if r_comp == '=':
            query += ' == '
        else:
            query += ' ' + r_comp + ' '
        if data_in.dtypes[r_attr] in ['float64', 'bool']:
            query += r_value
        else:
            query += '"' + r_value + '"'
        if r != branch_conditions[-1]:
            query += ' & '

    node_instances = data_in.query(query)
    wrong_instances = node_instances[node_instances['target'] != target_value]

    # TODO Sometimes both len are 0 so the upper bound is 'nan' (this also raises a warning since division by 0)
    return len(node_instances) * proportion_confint(len(wrong_instances), len(node_instances), method='beta', alpha=0.50)[1]


def discover_overlapping_rules(base_tree, dataset, attributes_map, original_rules) -> dict:
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

        sub_tree = DecisionTree.DecisionTree(attributes_map)
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


def shorten_rules_manually(original_rules, attributes_map) -> dict:
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
