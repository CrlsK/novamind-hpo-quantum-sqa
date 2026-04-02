"""
QCentroid Local Test Runner for Quantum-Inspired SQA Solver

This script provides a standard testing interface for the solver,
demonstrating how to invoke it with realistic hyperparameter optimization data.
"""

import json
import sys
import numpy as np
from typing import Dict, Any
import time

# Import the solver
from qcentroid import run


def create_realistic_hpo_input() -> Dict[str, Any]:
    """
    Create a realistic HPO problem specification matching the use case.

    Returns:
        Dictionary with full HPO problem specification
    """
    n_vars = 192

    # Create QUBO matrix with realistic structure
    # Sparsity ~15.5%, 2847 J-terms
    J_matrix = np.zeros((n_vars, n_vars))

    # Add correlated parameter pairs
    correlations = [
        (0, 8, 0.3),        # learning_rate - warmup_steps
        (13, 21, 0.2),      # weight_decay - dropout_rate
        (29, 33, 0.15),     # attention_heads - hidden_dim
        (38, 42, 0.2),      # num_layers - batch_size
        (46, 49, 0.25),     # optimizer - scheduler
        (72, 76, 0.15),     # activation_function - positional_encoding
    ]

    for start, end, strength in correlations:
        for i in range(start, start + 4):
            for j in range(end, end + 4):
                if np.random.random() < 0.6:
                    J_matrix[i, j] = np.random.randn() * strength
                    J_matrix[j, i] = J_matrix[i, j]

    # Add self-interactions
    for i in range(n_vars):
        J_matrix[i, i] = np.random.randn() * 0.05

    # Random sparse coupling for other pairs
    n_edges = 0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if np.random.random() < 0.02:  # 2% density
                J_matrix[i, j] = np.random.randn() * 0.02
                J_matrix[j, i] = J_matrix[i, j]
                n_edges += 1

    # Linear terms
    h_vector = np.random.randn(n_vars) * 0.1

    # Penalty terms (constraint enforcement)
    penalty_weight = 3.5
    for i in range(82, n_vars):
        h_vector[i] += 10.0 * penalty_weight

    input_data = {
        "Search_space_definition": {
            "num_hyperparameters": 16,
            "encoding": "one_hot",
            "total_binary_variables": 192,
            "variable_mapping": {
                "learning_rate": {"range": [0, 8], "values": 8},
                "warmup_steps": {"range": [8, 13], "values": 5},
                "weight_decay": {"range": [13, 21], "values": 8},
                "dropout_rate": {"range": [21, 29], "values": 8},
                "attention_heads": {"range": [29, 33], "values": 4},
                "hidden_dim": {"range": [33, 38], "values": 5},
                "num_layers": {"range": [38, 42], "values": 4},
                "batch_size": {"range": [42, 46], "values": 4},
                "optimizer": {"range": [46, 49], "values": 3},
                "scheduler": {"range": [49, 53], "values": 4},
                "gradient_clipping": {"range": [53, 61], "values": 8},
                "label_smoothing": {"range": [61, 69], "values": 8},
                "mixed_precision": {"range": [69, 72], "values": 3},
                "activation_function": {"range": [72, 76], "values": 4},
                "positional_encoding": {"range": [76, 80], "values": 4},
                "layer_norm_type": {"range": [80, 82], "values": 2},
            },
        },
        "Constraints_specification": {
            "constraint_types": ["one_hot_encoding", "feasible_combinations"],
            "constraint_penalty_indices": list(range(82, 192)),
            "penalty_weight": 3.5,
            "constraint_penalty_multiplier": 10.0,
        },
        "QUBO_coefficients": {
            "J_matrix": J_matrix.tolist(),
            "h_vector": h_vector.tolist(),
            "sparsity": float(np.count_nonzero(J_matrix)) / (n_vars * n_vars),
            "num_J_terms": n_edges,
        },
        "Solver_configuration": {
            "quantum_inspired": {
                "num_sweeps": 2000,
                "num_replicas": 8,
                "initial_temperature": 10.0,
                "final_temperature": 0.01,
                "beta_schedule": "geometric",
                "transverse_field_schedule": "linear_decrease",
                "initial_transverse_field": 5.0,
                "final_transverse_field": 0.001,
                "trotter_slices": 16,
                "replica_exchange_interval": 10,
            },
        },
        "Hybrid_quantum_classical_settings": {
            "warm_start": "from_top_10_classical_trials",
            "feedback_loop": True,
            "classical_prescreening_budget": 20,
            "quantum_refinement_budget": 180,
            "surrogate_model": "random_forest_regressor",
            "acquisition_function": "expected_improvement",
        },
        "Initialization_warm_start": {
            "use_classical_solutions": True,
            "num_classical_trials": 10,
            "diversification_factor": 0.3,
        },
        "Objective_specification": {
            "metric": "f1_macro",
            "baseline_best": 0.786,
            "target": 0.82,
            "direction": "maximize",
        },
    }

    return input_data


def create_solver_params(test_mode: bool = False) -> Dict[str, Any]:
    """
    Create solver parameters for SQA execution.

    Args:
        test_mode: If True, use reduced parameters for quick testing

    Returns:
        Dictionary with solver configuration
    """
    if test_mode:
        return {
            "num_sweeps": 100,
            "num_replicas": 4,
            "initial_temperature": 10.0,
            "final_temperature": 0.01,
            "beta_schedule": "geometric",
            "transverse_field_schedule": "linear_decrease",
            "initial_transverse_field": 5.0,
            "final_transverse_field": 0.001,
            "trotter_slices": 8,
        }
    return {
        "num_sweeps": 2000,
        "num_replicas": 8,
        "initial_temperature": 10.0,
        "final_temperature": 0.01,
        "beta_schedule": "geometric",
        "transverse_field_schedule": "linear_decrease",
        "initial_transverse_field": 5.0,
        "final_transverse_field": 0.001,
        "trotter_slices": 16,
    }


def run_solver_test():
    """
    Execute the solver test with realistic HPO data.

    Demonstrates:
    1. Input preparation
    2. Solver invocation
    3. Result validation
    4. Performance reporting
    """
    print("=" * 80)
    print("QCentroid Quantum-Inspired SQA Solver Test")
    print("=" * 80)

    # Prepare inputs
    print("\n[1] Preparing HPO problem specification...")
    input_data = create_realistic_hpo_input()
    solver_params = create_solver_params(test_mode=True)

    print(f"    - QUBO matrix: 192x192 (sparse)")
    print(f"    - SQA sweeps: {solver_params['num_sweeps']}")
    print(f"    - Replicas (temperature levels): {solver_params['num_replicas']}")
    print(f"    - Baseline f1_macro: 0.786")
    print(f"    - Target f1_macro: 0.82+")

    # Run solver
    print("\n[2] Executing Quantum-Inspired SQA solver...")
    start = time.time()
    result = run(input_data, solver_params, {})
    elapsed = time.time() - start

    # Validate result
    print("\n[3] Validating results...")
    assert result["solution_status"] == "success", "Solver failed"
    assert "objective_value" in result, "Missing objective_value"
    assert "best_hyperparameter_config" in result, "Missing config"
    assert "top_10_configurations" in result, "Missing top_10_configurations"
    assert "benchmark" in result, "Missing benchmark"
    print("    â All required fields present")

    # Report results
    print("\n[4] Results Summary")
    print("-" * 80)

    obj_value = result["objective_value"]
    print(f"Objective Value (estimated f1_macro):        {obj_value:.6f}")
    print(f"Improvement over baseline (0.786):          {(obj_value - 0.786):.6f} "
          f"({100 * (obj_value - 0.786) / 0.786:.2f}%)")

    print(f"\nSolution Status:                            {result['solution_status']}")

    best_config = result["best_hyperparameter_config"]
    print(f"\nBest Hyperparameter Configuration:")
    print(f"  Learning Rate:                              {best_config.get('learning_rate', 'N/A'):.6f}")
    print(f"  Weight Decay:                              {best_config.get('weight_decay', 'N/A'):.6f}")
    print(f"  Dropout Rate:                              {best_config.get('dropout_rate', 'N/A'):.6f}")
    print(f"  Optimizer:                                 {best_config.get('optimizer', 'N/A')}")
    print(f"  Scheduler:                                 {best_config.get('scheduler', 'N/A')}")
    print(f"  Hidden Dimension:                          {best_config.get('hidden_dim', 'N/A')}")
    print(f"  Num Layers:                                {best_config.get('num_layers', 'N/A')}")
    print(f"  Batch Size:                                {best_config.get('batch_size', 'N/A')}")

    print(f"\nEnergy Landscape Statistics:")
    energy_landscape = result["computation_metrics"]["energy_landscape"]
    print(f"  Min Energy:                                {energy_landscape['min_energy']:.6f}")
    print(f"  Max Energy:                                {energy_landscape['max_energy']:.6f}")
    print(f"  Mean Energy:                               {energy_landscape['mean_energy']:.6f}")
    print(f"  Energy Gap:                                {energy_landscape['energy_gap']:.6f}")

    print(f"\nQuantum Annealing Metrics:")
    print(f"  Replica Exchange Rate:                     "
          f"{result['computation_metrics']['replica_exchange_rate']:.3f}")
    print(f"  Sweeps Completed:                        * {solver_params['num_sweeps']}")
    print(f"  Active Temperature Levels:                 {solver_params['num_replicas']}")

    print(f"\nTop 10 Configurations (by estimated f1_macro):")
    for cfg in result["top_10_configurations"][:10]:
        print(f"  Rank {cfg['rank']:2d}: f1_macro={cfg['estimated_f1_macro']:.6f}, "
              f"energy={cfg['qubo_energy']:10.4f}")

    print(f"\nComputational Cost Breakdown:")
    cost = result["cost_breakdown"]
    print(f"  SQA Sweeps:                                {cost['sqa_sweeps']}")
    print(f"  Replica Exchanges:                         {cost['replica_exchanges']}")
    print(f"  QUBO Energy Evaluations:                   {cost['qubo_energy_evaluations']}")

    print(f"\nBenchmark Metrics:")
    bench = result["benchmark"]
    print(f"  Execution Cost (sweeps):                   {bench['execution_cost']:.0f}")
    print(f"  Time Elapsed:                              {bench['time_elapsed']:.3f}s")
    print(f"  Energy Consumption (evals):                {bench['energy_consumption']:.0f}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    result = run_solver_test()
    sys.exit(0 if result["solution_status"] == "success" else 1)
