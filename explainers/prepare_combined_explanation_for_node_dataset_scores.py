import torch


def identity_explanation_combined(node_explanations, explainer):
    """
    identity explanation, which does not change the input node_explanations
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    return node_explanations


def characterization_score_explanation_combined(node_explanations, explainer):
    """
    explanation for the characterization score, introduced in GNNExplainer
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if 'pos_weight_characterization_score' not in node_explanations.control_data:
        pos_weight = explainer.config['pos_weight_characterization_score']
        node_explanations.control_data.update(
            {'pos_weight_characterization_score': pos_weight})
    if 'neg_weight_characterization_score' not in node_explanations.control_data:
        neg_weight = explainer.config['neg_weight_characterization_score']
        node_explanations.control_data.update(
            {'neg_weight_characterization_score': neg_weight})
    return node_explanations


def get_multi_edge_threshold_auc(explainer):
    if explainer.config['edge_mask_hard_method'] == 'threshold':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['edge_mask_hard_method'] == 'auto_threshold':
        edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        threshold = [torch.quantile(edge_mask_concat, i) for i in
                     explainer.config['threshold_percentage_auc']]
        return threshold
    elif explainer.config['edge_mask_hard_method'] == 'original':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['edge_mask_hard_method'] == 'top_k':
        if explainer.config.get(['top_k_for_auc'], None) is not None:
            return explainer.config['top_k_for_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    else:
        raise ValueError('Invalid edge_mask_threshold_method: {}'.format(
            explainer.config['edge_mask_threshold_method']))


def get_multi_feature_threshold_auc(explainer):
    if explainer.config['feature_mask_hard_method'] == 'threshold':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['feature_mask_hard_method'] == 'auto_threshold':
        feature_mask = explainer.feature_mask_for_output
        threshold = [torch.quantile(feature_mask, i) for i in
                     explainer.config['threshold_percentage_auc']]
        return threshold
    elif explainer.config['feature_mask_hard_method'] == 'original':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['feature_mask_hard_method'] == 'top_k':
        if explainer.config.get(['top_k_for_auc'], None) is not None:
            return explainer.config['top_k_for_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    else:
        raise ValueError('Invalid feature_mask_threshold_method: {}'.format(
            explainer.config['feature_mask_threshold_method']))


def fidelity_curve_auc_explanation_combined(node_explanations, explainer):
    """
    explanation for the fidelity curve AUC score, introduced in GNNExplainer
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if "auc_threshold_values" not in node_explanations.control_data:
        if explainer.config.get("auc_use_feature_mask", False):
            threshold_values = get_multi_feature_threshold_auc(explainer)
        else:
            threshold_values = get_multi_edge_threshold_auc(explainer)
        node_explanations.control_data.update(
            {"auc_threshold_values": threshold_values})
    return node_explanations


def graph_exp_stability_feature_explanation_combined(node_explanations, explainer):
    """
    Set top_k_for_stability_feature in the control_data of the node_explanations

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if 'top_k_for_stability_feature' not in node_explanations.control_data:
        if 'top_k_for_stability_feature' in explainer.config:
            k = explainer.config['top_k_for_stability_feature']
        else:
            k = 0.25
        node_explanations.control_data.update({'top_k_for_stability_feature': k})
    return node_explanations


def graph_exp_stability_edge_explanation_combined(node_explanations, explainer):
    """
    Set top_k_for_stability_edge in the control_data of the node_explanations

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if 'top_k_for_stability_edge' not in node_explanations.control_data:
        if 'top_k_for_stability_edge' in explainer.config:
            k = explainer.config['top_k_for_stability_edge']
        else:
            k = 0.25
        node_explanations.control_data.update({'top_k_for_stability_edge': k})
    return node_explanations

def metapath_necessity_explanation_combined(node_explanations, explainer):
    """
    prepare for meta-path necessity study
    """
    if hasattr(node_explanations, 'metapath_removal_results'):
        return node_explanations

    print("Running meta-path necessity study (removing each meta-path)")

    # > original performace
    original_pred = explainer.model.forward()
    original_pred_hard = original_pred.argmax(dim=-1)
    test_nodes = explainer.model.dataset.labels[2]
    test_nodes_idx = [node[0] for node in test_nodes]
    test_labels = torch.tensor([node[1] for node in test_nodes]).to(explainer.device_string)

    original_acc = (original_pred_hard[test_nodes_idx] == test_labels).float().mean().item()

    # > individually remove meta paths
    metapath_removed_accuracies[[I = {}
    num_metapaths = len(explainer.model.get_metapaths()) if hasattr(explainer.model, 'get_metapaths') else 0

    for mp_id in range(num_metapaths):
        # > build mask: remove mp_id th mata path
        removed_pred = explainer.model.custom_forward(
            explainer.get_metapath_removal_handle_fn(mp_id)
        )
        removed_pred_hard = removed_pred.argmax(dim=-1)
        removed_acc = (removed_pred_hard[test_nodes_idx] == test_labels).float().mean().item()
        metapath_removed_accuracies[mp_id] = removed_acc

    # > save results
    node_explanations.metapath_removal_results = {
        'original_accuracy': original_acc,
        'metapath_removed_accuracies': metapath_removed_accuracies
    }

    return node_explanations


def uniform_attention_explanation_combined(node_explanations, explainer):
    """
    performance when attention is uniform
    """
    if hasattr(node_explanations, 'uniform_attention_results'):
        return node_explanations

    print("Testing uniform attention...")

    # > original performance
    original_pred = explainer.model.forward()
    original_pred_hard = original_pred.argmax(dim=-1)
    test_nodes = explainer.model.dataset.labels[2]
    test_nodes_idx = [node[0] for node in test_nodes]
    test_labels = torch.tensor([node[1] for node in test_nodes]).to(explainer.device_string)

    original_acc = (original_pred_hard[test_nodes_idx] == test_labels).float().mean().item()

    # > apply uniform attention
    uniform_pred = explainer.model.custom_forward(
        explainer.get_uniform_attention_handle_fn()
    )
    uniform_pred_hard = uniform_pred.argmax(dim=-1)
    uniform_acc = (uniform_pred_hard[test_nodes_idx] == test_labels).float().mean().item()

    node_explanations.uniform_attention_results = {
        'original_accuracy': original_acc,
        'uniform_attention_accuracy': uniform_acc
    }

    return node_explanations



prepare_combined_explanation_fn_for_node_dataset_scores = {
    'fidelity_neg': identity_explanation_combined,
    'fidelity_pos': identity_explanation_combined,
    'characterization_score': characterization_score_explanation_combined,
    'fidelity_curve_auc': fidelity_curve_auc_explanation_combined,
    'unfaithfulness': identity_explanation_combined,
    'sparsity': identity_explanation_combined,
    'graph_exp_stability_feature': graph_exp_stability_feature_explanation_combined,
    'graph_exp_stability_edge': graph_exp_stability_edge_explanation_combined,
    'Macro-F1': identity_explanation_combined,
    'Micro-F1': identity_explanation_combined,
    'roc_auc_score': identity_explanation_combined,
    'fidelity_neg_model': identity_explanation_combined,
    'fidelity_pos_model': identity_explanation_combined,
}
