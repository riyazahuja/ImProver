import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    

    require PrimeNumberTheoremAnd from git
    "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd.git" @ "main"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    lean_exe training_data where
    root := `scripts.training_data
    
    lean_exe constants where
    root := `scripts.constants

    