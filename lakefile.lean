import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    

    require Tests3 from git
    "https://github.com/riyazahuja/Tests3.git" @ "7ede12b32cef52b0b3633fed06d3c81d436499a4"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    lean_exe state_comments where
    root := `scripts.state_comments

    