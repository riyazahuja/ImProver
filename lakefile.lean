import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    

    require Tests3 from git
    "https://github.com/riyazahuja/Tests3.git" @ "961344f4e153f35f6b65e1b6cdfc769fc9c5ba26"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    