import Lake
open Lake DSL

package «lean-training-data» {
  -- add any package configuration options here
}

require PFR from git
  "https://github.com/teorth/pfr.git" @ "71bddfb866c423f3285c6ee186a0d95e62126bb3"

@[default_target]
lean_lib TrainingData where

lean_lib Examples where

lean_exe training_data where
  root := `scripts.training_data

lean_exe full_proof_training_data where
  root := `scripts.full_proof_training_data

lean_exe state_comments where
  root := `scripts.state_comments

