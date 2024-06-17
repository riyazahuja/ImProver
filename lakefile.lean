import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    

    require Tests3 from git
    "https://github.com/riyazahuja/Tests3.git" @ "8c545b4691a61cdd218f5a0e1779f64383866356"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    lean_exe state_comments where
    root := `scripts.state_comments

    