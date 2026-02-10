This repository contains the data and code for the paper titled "Resilient Microgrid Planning for Socially Vulnerable Communities", published in Applied Energy Journal, 2026.
This repository contains a Python-based framework for modeling, solving, and analyzing a microgrid optimization problem. The project includes data processing, model formulation, decomposition methods, scenario analysis, and sensitivity studies.

Repository Structure
.
├── Data/                # Input data files
├── IMG/                 # Figures and visualization outputs
├── Results/             # Solution outputs and processed results
├── Scenarios/           # Scenario definitions and configurations
├── Solutions/           # Stored optimization solutions
├── Analysis.ipynb       # Post-solution analysis and visualization
├── Sensitivity.ipynb    # Sensitivity analysis experiments
├── Workspace.ipynb      # Interactive development and testing
├── Data.py              # Data loading and preprocessing utilities
├── Master.py            # Main optimization model (master problem)
├── Methods.py           # Solution methods and algorithm implementations
├── Separation.py        # Subproblem / cut generation logic
├── master_log.log       # Solver log file
└── README.md


Overview

The framework is designed to:

Formulate a large-scale optimization model for microgrid planning and operation

Solve the problem using decomposition-based methods

Evaluate system performance under multiple scenarios

Analyze results and perform sensitivity studies

The workflow follows a structured pipeline:

Load and preprocess input data

Define scenarios

Build and solve the optimization model

Store solutions

Perform analysis and visualization

Main Components
Model

Master.py
Implements the master optimization model.

Separation.py
Handles subproblems and cut generation (e.g., Benders-type separation).

Methods.py
Contains algorithmic procedures and solution strategies.

Data Handling

Data.py
Reads input data and prepares model parameters.

Experiments

Scenarios/
Different system configurations and uncertainty cases.

Sensitivity.ipynb
Parameter sensitivity analysis.

Results & Analysis

Results/ and Solutions/
Store model outputs and solution files.

Analysis.ipynb
Post-processing, visualization, and performance evaluation.

Requirements

Typical dependencies:

Python 3.9+

numpy

pandas

matplotlib

jupyter

gurobipy (or another supported solver)

Install basic packages:

pip install numpy pandas matplotlib jupyter


Make sure your optimization solver is installed and licensed (e.g., Gurobi).

How to Run
1. Prepare Data

Place required input files inside the Data/ directory.

2. Define Scenario

Configure or select a scenario from the Scenarios/ folder.

3. Run the Model
python Master.py


Outputs will be saved in:

Results/

Solutions/

master_log.log

4. Analyze Results

Open:

Analysis.ipynb for performance evaluation

Sensitivity.ipynb for parameter studies

Outputs

The framework generates:

Optimal decision variables

Objective values

Scenario performance metrics

Solver logs

Plots and figures (stored in IMG/)

Notes

Large experiments may require significant computation time.

Intermediate results and logs are stored automatically.

The repository is structured to support reproducible computational experiments.

Author

Farzane Ezzati
PhD Candidate, Industrial Engineering
University of Houston
