import Lake
    open Lake DSL

    package «lean-training-data» {
    -- add any package configuration options here
    }

    

    require braid_project from "/Users/ahuja/Desktop/braids_better"

    @[default_target]
    lean_lib TrainingData where

    lean_lib Examples where

    