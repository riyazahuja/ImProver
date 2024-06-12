import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    require mathlib from git
    "https://github.com/leanprover-community/mathlib4.git" @ "master"

    require Tests from git
    "https://github.com/riyazahuja/Tests.git" @ "acdb55e7de01fb3e62bb0757dcbabcb38ad2694f"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    lean_exe state_comments where
    root := `scripts.state_comments

    