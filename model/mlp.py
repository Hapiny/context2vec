import torch.nn as nn

from overrides import overrides


class MLP(nn.Module):
    def __init__(self, _input_size: int, _hidden_size: int, _output_size: int, _num_layers: int = 2,
                 _keep_prob: float = 1.0, _activate_func: str = "relu"):
        """"""
        super(MLP, self).__init__()
        assert _num_layers >= 1, f"Invalid number of layers: {_num_layers}"
        assert 0.0 <= _keep_prob <= 1.0, f"Invalid Dropout keep prob: {_keep_prob}"
        assert _activate_func in ["relu", "tanh"], f"Invalid activation function name: {_activate_func}"

        self.input_size = _input_size
        self.hidden_size = _hidden_size
        self.output_size = _output_size
        self.num_layers = _num_layers
        self.dropout = nn.Dropout(1 - _keep_prob)
        self.mlp = self.create_mlp()
        self.activate_func = (nn.ReLU() if _activate_func == "relu" else nn.Tanh())

    def create_mlp(self) -> nn.ModuleList:
        mlp = nn.ModuleList()
        if self.num_layers == 1:
            mlp.append(nn.Linear(self.input_size, self.output_size))
        else:
            mlp.append(nn.Linear(self.input_size, self.hidden_size))
            for _ in range(self.num_layers - 2):
                mlp.append(nn.Linear(self.hidden_size, self.hidden_size))
            mlp.append(nn.Linear(self.hidden_size, self.output_size))
        return mlp

    @overrides
    def forward(self, _input):
        output = _input
        for i in range(self.num_layers - 1):
            output = self.mlp[i](self.dropout(output))
            output = self.activate_func(output)
        output = self.mlp[-1](self.dropout(output))
        return output
