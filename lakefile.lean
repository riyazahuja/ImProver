import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    require mathlib from git
    "https://github.com/leanprover-community/mathlib4.git" @ "cf8e23a62939ed7cc530fbb68e83539730f32f86"

    require Tests from git
    "https://github.com/riyazahuja/Tests.git" @ "bff613df11a1f8c3c812276ee93365c16980e174"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    lean_exe state_comments where
    root := `scripts.state_comments

    