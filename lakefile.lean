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

lean_lib Examples where


lean_lib RAG where


@[default_target]
lean_exe get_prompts where
  root := `ImProver.get_prompts
  supportInterpreter := true

@[default_target]
lean_exe get_class2 where
  root := `ImProver.KG.get_class2
  supportInterpreter := true

@[default_target]
lean_exe eval_improver where
  root := `ImProver.eval_improver
  supportInterpreter := true
