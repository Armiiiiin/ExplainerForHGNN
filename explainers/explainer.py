import torch

class ExplainerCore:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = config.get("eval_metrics", None)

        self.device = "cpu"
        self.registered_modules_and_params = {}
        self.original_explainer = False

    def init_params(self):
        """Initialize parameters for the explainer."""
        pass

    def init_params_node_level(self):
        """
        Initialize parameters for node-level explanation.
        :return:
        """
        self.edge_mask = None
        self.feature_mask = None
        pass

    def init_params_graph_level(self):
        """
        Initialize parameters for graph-level explanation.
        :return:
        """
        pass

    def extract_neighbors_input(self):
        """
        Extract input of neighbors for the target node.
        :return:
        """
        self.neighbor_input = {"gs": None, "feature": None}
        self.n_hop = self.config.get("n_hop", 3)
        self.used_nodes = []
        self.recovery_dict = {}
        pass

    def explain(self, model, **kwargs):
        """
        Explain the model.
        :param model:
        :param kwargs:
        :return:
        """
        self.model = model
        self.model.eval()

        pass

    def node_level_explain(self, **kwargs):
        """
        Node-level explanation.
        :param kwargs:
        :return:
        """
        self.node_id = kwargs.get("node_id", None)
        if self.node_id is None:
            raise ValueError("Node ID is required for node-level explanation")
        pass

    def graph_level_explain(self):
        """
        Graph-level explanation.
        :return:
        """
        pass

    def visualize(self):
        """
        Visualize the explanation.
        :return:
        """
        pass

    def construct_explanation(self):
        """
        Construct the explanation for metrics and visualization.
        :return:
        """
        pass

    def get_required_fit_params(self):
        """
        Get the required fit parameters, to be used in the optimization process.
        :return:
        """
        pass

    def fit(self):
        """
        Fit the model. If it is a training process, the model will be trained.
        :return:
        """
        pass

    def fit_node_level(self):
        """
        Fit the model for node-level explanation.
        :return:
        """
        pass

    def fit_graph_level(self):
        """
        Fit the model for graph-level explanation.
        :return:
        """
        pass

    def get_loss(self, **kwargs):
        """
        Get the loss for the optimization process.
        :return:
        """
        pass

    def get_loss_node_level(self, **kwargs):
        """
        Get the loss for the optimization process in node-level explanation.
        :return:
        """
        pass

    def get_loss_graph_level(self, **kwargs):
        """
        Get the loss for the optimization process in graph-level explanation.
        :return:
        """
        pass

    def get_input_handle_fn(self):
        """
        Get the input handle function for the model.
        :return:
        """
        pass

    def get_input_handle_fn_node_level(self):
        """
        Get the input handle function for the model in node-level explanation.
        :return:
        """
        self.masked = {"gs": None, "feature": None}
        pass

    def get_input_handle_fn_graph_level(self):
        """
        Get the input handle function for the model in graph-level explanation.
        :return:
        """

        pass

    def forward(self, **kwargs):
        """
        Forward the model.
        :param kwargs:
        :return:
        """
        pass

    def forward_node_level(self, **kwargs):
        """
        Forward the model in node-level explanation.
        :param kwargs:
        :return:
        """
        pass

    def forward_graph_level(self, **kwargs):
        """
        Forward the model in graph-level explanation.
        :param kwargs:
        :return:
        """
        pass

    def build_optimizer(self):
        """
        Build the optimizer for the optimization process.
        :return:
        """
        pass

    def build_scheduler(self, optimizer):
        """
        Build the scheduler for the optimization process.
        :return:
        """
        pass

    @property
    def edge_mask_for_output(self):
        return self.edge_mask

    @property
    def feature_mask_for_output(self):
        return self.feature_mask

    def get_custom_input_handle_fn(self, **kwargs):
        """
        Get the custom input handle function for the model.
        :return:
        """
        pass

    def to(self, device):
        """
        Set the device for the explainer core.
        :param device:
        :return:
        """
        self.device = device
        for module in self.registered_modules_and_params.values():
            module.to(self.device_string)
        return self

    @property
    def device_string(self):
        return "cuda:{}".format(self.device) if self.device != "cpu" else self.device


class Explainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = config.get("metrics", None)
        self.device = config.get("device", "cpu")
        if self.device is None:
            self.device = "cpu"
        self.registered_modules_and_params = {}

    def explain(self, model):
        self.model = model
        self.model.eval()
        pass

    def node_level_explain(self, **kwargs):
        pass

    def graph_level_explain(self, **kwargs):
        pass

    def construct_explanation(self, result):
        pass

    def evaluate(self):
        pass

    def visualize(self):
        pass

    def get_summary(self):
        pass

    def save_summary(self):
        pass

    def to(self, device):
        """
        Set the device for the explainer.
        :param device:
        :return:
        """
        self.device = device
        for module in self.registered_modules_and_params.values():
            module.to(self.device_string)
        return self

    @property
    def device_string(self):
        return "cuda:{}".format(self.device) if self.device != "cpu" else self.device

    def save_explanation(self):
        pass

    def core_class(self):
        """
        Get the core class of the explainer.
        :return:
        """
        return ExplainerCore

    def get_metapath_removal_handle_fn(self, metapath_id_to_remove):
        """
        Generate input handle function for removing a specific meta-path.
        Used for meta-path necessity ablation study.

        :param metapath_id_to_remove: Index of the meta-path to remove
        :return: Input handle function
        """
        def input_handle_fn(model):
            gs, features = model.standard_input()

            # > Remove the specified meta-path by replacing with zero matrix
            masked_gs = []
            for i, g in enumerate(gs):
                if i == metapath_id_to_remove:
                    # Create zero matrix to remove this meta-path
                    masked_gs.append(torch.zeros_like(g).to(self.device_string))
                else:
                    masked_gs.append(g)

            return masked_gs, features

        return input_handle_fn
    def get_uniform_attention_handle_fn(self):
        """
        Generate input handle function for uniform attention.
        Used for testing the importance of learned attention mechanism.

        :return: Input handle function
        """
        def input_handle_fn(model):
            gs, features = model.standard_input()

            # > Set uniform attention weights for all meta paths
            num_metapaths = len(gs)
            if num_metapaths > 0:
                uniform_weights = torch.ones(num_metapaths) / num_metapaths
                uniform_weights = uniform_weights.to(self.device_string)

                # > Try different ways to set attention based on model implementation
                if hasattr(model, 'set_uniform_attention'):
                    model.set_uniform_attention()
                elif hasattr(model, 'set_attention_weights'):
                    model.set_attention_weights(uniform_weights)
                elif hasattr(model, 'attention_weights'):
                    model.attention_weights = uniform_weights

            return gs, features

        return input_handle_fn
