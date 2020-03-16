import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


from overrides import overrides


class MLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: Union[int, List[int]],
            output_size: int,
            num_layers: int = 2,
            dropout_rate: float = 0.0,
    ):
        """
        Class that implements Multi-Layer Perceptron module.
        Args:
            input_size: Size of the input Tensor.
            hidden_size: Size of the hidden states in the hidden layers.
                You can specify different hidden sizes for different layers
                by passing list of the size or pass one number, then all hidden
                layers will have same size.
            output_size: Size of the output Tensor.
            num_layers: Number of layers excepting input layer, e.g.
                number of hidden layers + output layer.
                It must be value greater than zero. If num_layers equal to 1,
                then MLP consists only input and output layers.
            dropout_rate: Probability that values of the Tensor could be zeroed.
        """
        super(MLP, self).__init__()
        assert num_layers >= 1, f"Invalid number of layers: {num_layers}"
        assert 0.0 <= dropout_rate <= 1.0, f"Invalid Dropout prob: {dropout_rate}"

        self.input_size = input_size
        self.hidden_sizes = hidden_size if isinstance(hidden_size, list) else [hidden_size] * (num_layers - 1)
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = self.create_mlp()

    def create_mlp(self) -> nn.ModuleList:
        """
        Method that creates a list of Linear layers for Perceptron.
        Returns:
            mlp: PyTorch list of the modules.
        """
        mlp = nn.ModuleList()
        if self.num_layers == 1:
            mlp.append(nn.Linear(self.input_size, self.output_size))
        else:
            mlp.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
            for i in range(1, self.num_layers - 1):
                mlp.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_size[i]))
            mlp.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
        return mlp

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Method that doing forward pass of the MLP.
        Args:
            inputs: Input tensors for processing with MLP.

        Returns:
            output: Processed tensor.
        """
        output = inputs
        for i in range(self.num_layers - 1):
            output = self.mlp[i](self.dropout(output))
            output = F.relu(output)
        output = self.mlp[-1](self.dropout(output))
        return output
