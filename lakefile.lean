import Lake
open Lake DSL

package «lean-training-data» {
  -- add any package configuration options here
}

-- require mathlib from git
--   "https://github.com/leanprover-community/mathlib4.git" @ "e9ae2a61ef5c99d6edac84f0d04f6324c5d97f67"

require mil from git
  "https://github.com/riyazahuja/Improver_MIL" @ "new"

require compfiles from git
  "https://github.com/dwrensha/compfiles" @ "c09159effc5eeb64903482db764e3ed5f14c8ee2"

-- require carleson from git
--   "https://github.com/fpvandoorn/carleson" @ "386a3c6e178f3c1b92ad5547a69b48ebfe1564ea"

-- require HepLean from git
--   "https://github.com/HEPLean/PhysLean" @ "656a3e422fe26c38c4f52081528c24fc1ec6b26c"

-- -- require htpi from git
-- --   "https://github.com/djvelleman/HTPILeanPackage"

-- require PrimeNumberTheoremAnd from git
--   "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd" @ "44863eeae4c5b9981af7e634e5531cb41a3135a0"



require «doc-gen4» from git "https://github.com/leanprover/doc-gen4" @ "v4.15.0"


lean_lib TrainingData where

@[default_target]
lean_lib ImProver where


lean_lib temp where


lean_lib Examples where


lean_lib RAG where


@[default_target]
lean_exe improver where
  root := `ImProver.improver
  supportInterpreter := true

lean_exe StateComments where
  root := `scripts.standalone_state_comments

lean_exe extract_states where
  root := `scripts.extract_states
