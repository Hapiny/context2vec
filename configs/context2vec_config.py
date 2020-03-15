from .config import Config


class Context2VecConfig(Config):
    def __init__(
            self,
            context_emb_size: int,
            target_emb_size: int,
            lstm_size: int,
            lstm_num_layers: int,
            mlp_size: int,
            mlp_num_layers: int,
            mlp_dropout: float,
            num_negative_samples: int,
            alpha: float
    ):
        self.context_emb_size = context_emb_size
        self.target_emb_size = target_emb_size
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_size = mlp_size
        self.mlp_num_layers = mlp_num_layers
        self.mlp_dropout = mlp_dropout
        self.num_negative_samples = num_negative_samples
        self.alpha = alpha
