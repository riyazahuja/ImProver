import Lake
open Lake DSL

package «lean-training-data» {
  -- add any package configuration options here
}

-- require mathlib from git
--   "https://github.com/leanprover-community/mathlib4.git" @ "v4.17.0"

require mil from git
  "https://github.com/riyazahuja/Improver_MIL" @ "fixed"

require compfiles from git
  "https://github.com/dwrensha/compfiles" @ "e5870cdeef56b5731a94480d546ede69902d719d"

require PrimeNumberTheoremAnd from git
  "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd" @ "be80860a87757375652a0a92d2258c893dfe5002"


require PFR from git
  "https://github.com/teorth/pfr" @ "v4.17.0"


lean_lib TrainingData where

@[default_target]
lean_lib ImProver where


lean_lib temp where


lean_lib Examples where


lean_lib RAG where





lean_exe improver where
  root := `ImProver.improver

@[default_target]
lean_exe get_prompts where
  root := `ImProver.efficient.get_prompts
  supportInterpreter := true


lean_exe get_KG where
  root := `ImProver.C1Graph.getKG
  supportInterpreter := true

lean_exe eval_improver where
  root := `ImProver.efficient.eval_improver
  supportInterpreter := true




lean_exe StateComments where
  root := `scripts.standalone_state_comments

lean_exe extract_states where
  root := `scripts.extract_states
