import Lake
open Lake DSL

package «lean-training-data» {
  -- add any package configuration options here
}

-- require mathlib from git
--   "https://github.com/leanprover-community/mathlib4.git" @ "e9ae2a61ef5c99d6edac84f0d04f6324c5d97f67"

require mil from git
  "https://github.com/riyazahuja/Improver_MIL" @ "new"

require «doc-gen4» from git "https://github.com/leanprover/doc-gen4" @ "v4.15.0"

@[default_target]
lean_lib TrainingData where

lean_lib temp where

lean_lib Examples where

lean_lib scripts where


@[default_target]
lean_exe ImProver where
  root := `scripts.verifier
  supportInterpreter := true
