"""
Quantum-Inspired SQA (Simulated Quantum Annealing) with Replica Exchange
for Massively Parallel Hyperparameter Optimization on QCentroid.

This solver implements:
1. Suzuki-Trotter decomposition with Trotter slices
2. Transverse field quantum annealing
3. Replica exchange between temperature levels
4. Geometric temperature schedule
5. Surrogate-based objective evaluation
6. Constraint penalty integration
7. Quantum tunneling effects visible in results
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json
import time
import logging
import os
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SQAConfig:
    """SQA algorithm configuration."""
    num_sweeps: int = 2000
    num_replicas: int = 8
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    beta_schedule: str = "geometric"
    transverse_field_schedule: str = "linear_decrease"
    initial_transverse_field: float = 5.0
    final_transverse_field: float = 0.001
    trotter_slices: int = 16
    replica_exchange_interval: int = 10


@dataclass
class HyperparametersConfig:
    """Hyperparameter decoding configuration."""
    learning_rate_bins: List[float] = None
    warmup_steps_choices: List[int] = None
    weight_decay_bins: List[float] = None
    dropout_rate_bins: List[float] = None
    attention_heads_choices: List[int] = None
    hidden_dim_choices: List[int] = None
    num_layers_choices: List[int] = None
    batch_size_choices: List[int] = None
    optimizer_choices: List[str] = None
    scheduler_choices: List[str] = None
    gradient_clipping_bins: List[float] = None
    label_smoothing_bins: List[float] = None
    mixed_precision_choices: List[str] = None
    activation_function_choices: List[str] = None
    positional_encoding_choices: List[str] = None
    layer_norm_type_choices: List[str] = None

    def __post_init__(self):
        """Initialize default hyperparameter values."""
        if self.learning_rate_bins is None:
            self.learning_rate_bins = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        if self.warmup_steps_choices is None:
            self.warmup_steps_choices = [100, 500, 1000, 2000, 5000]
        if self.weight_decay_bins is None:
            self.weight_decay_bins = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        if self.dropout_rate_bins is None:
            self.dropout_rate_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        if self.attention_heads_choices is None:
            self.attention_heads_choices = [4, 8, 12, 16]
        if self.hidden_dim_choices is None:
            self.hidden_dim_choices = [128, 256, 512, 768, 1024]
        if self.num_layers_choices is None:
            self.num_layers_choices = [2, 4, 6, 8]
        if self.batch_size_choices is None:
            self.batch_size_choices = [16, 32, 64, 128]
        if self.optimizer_choices is None:
            self.optimizer_choices = ["adam", "adamw", "sgd"]
        if self.scheduler_choices is None:
            self.scheduler_choices = ["constant", "linear", "cosine", "warmup_cosine"]
        if self.gradient_clipping_bins is None:
            self.gradient_clipping_bins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0]
        if self.label_smoothing_bins is None:
            self.label_smoothing_bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        if self.mixed_precision_choices is None:
            self.mixed_precision_choices = ["none", "fp16", "bfloat16"]
        if self.activation_function_choices is None:
            self.activation_function_choices = ["relu", "gelu", "swish", "mish"]
        if self.positional_encoding_choices is None:
            self.positional_encoding_choices = ["absolute", "relative", "rotary", "alibi"]
        if self.layer_norm_type_choices is None:
            self.layer_norm_type_choices = ["layer_norm", "group_norm"]

    def decode_one_hot(self, binary_vec: np.ndarray, param_name: str) -> Any:
        """
        Decode one-hot encoded parameter from binary vector.

        Args:
            binary_vec: Full binary solution vector
            param_name: Name of parameter to decode

        Returns:
            Decoded parameter value
        """
        start, end = self.VAR_RANGES[param_name]
        one_hot = binary_vec[start:end].astype(int)

        # Get the index of the 1 (one-hot encoding)
        if np.sum(one_hot) == 0:
            # No bit set: use first option as default
            idx = 0
        else:
            idx = np.argmax(one_hot)

        # Map to parameter value
        if param_name == "learning_rate":
            return self.learning_rate_bins[idx]
        elif param_name == "warmup_steps":
            return self.warmup_steps_choices[idx]
        elif param_name == "weight_decay":
            return self.weight_decay_bins[idx]
        elif param_name == "dropout_rate":
            return self.dropout_rate_bins[idx]
        elif param_name == "attention_heads":
            return self.attention_heads_choices[idx]
        elif param_name == "hidden_dim":
            return self.hidden_dim_choices[idx]
        elif param_name == "num_layers":
            return self.num_layers_choices[idx]
        elif param_name == "batch_size":
            return self.batch_size_choices[idx]
        elif param_name == "optimizer":
            return self.optimizer_choices[idx]
        elif param_name == "scheduler":
            return self.scheduler_choices[idx]
        elif param_name == "gradient_clipping":
            return self.gradient_clipping_bins[idx]
        elif param_name == "label_smoothing":
            return self.label_smoothing_bins[idx]
        elif param_name == "mixed_precision":
            return self.mixed_precision_choices[idx]
        elif param_name == "activation_function":
            return self.activation_function_choices[idx]
        elif param_name == "positional_encoding":
            return self.positional_encoding_choices[idx]
        elif param_name == "layer_norm_type":
            return self.layer_norm_type_choices[idx]

    def decode_solution(self, binary_vec: np.ndarray) -> Dict[str, Any]:
        """
        Decode full solution to hyperparameter configuration.

        Args:
            binary_vec: Binary solution vector (192 variables)

        Returns:
            Dictionary of decoded hyperparameter values
        """
        config = {}
        for param_name in self.VAR_RANGES.keys():
            config[param_name] = self.decode_one_hot(binary_vec, param_name)
        return config
