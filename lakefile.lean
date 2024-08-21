import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    

    require compfiles from git
    "https://github.com/dwrensha/compfiles.git" @ "2ab6378aedaf379a1f58043199f89d40a5cda400"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data

    