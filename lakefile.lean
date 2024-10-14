import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    require mathlib from git
    "https://github.com/leanprover-community/mathlib4.git" @ "master"

    require equational_theories from git
    "https://github.com/teorth/equational_theories.git" @ "main"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data
    
    lean_exe constants where
    root := `scripts.constants

    