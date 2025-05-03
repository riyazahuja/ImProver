/-
Extracts proof states from constants in the environment
-/

import TrainingData.Frontend
import Mathlib.Data.Prod.Basic
import Mathlib.Data.Nat.Prime.Basic
import Lean
import Cli

open Lean Meta Core Cli

/-- Set this to false if you also want to extract proof states for definitions -/
def theoremsOnly := true

/-- Extracts the initial proof state of a constant. -/
def getInitialProofState (env : Environment) (ci : ConstantInfo) : IO String := do
  let (state, _, _) ← MetaM.toIO (ctxCore := { fileName := "", fileMap := default }) (sCore := { env }) do
    -- forallTelescope transforms ∀ n : Nat, 0 + n = n to _args = #[n : Nat] and typ = 0 + n = n
    forallTelescope ci.type fun _args typ => do
      let g ← mkFreshExprMVar typ
      g.mvarId!.withContext do
        -- We disable some pretty-printing options,
        -- e.g. Nat is not pretty-printed as ℕ
        -- HAdd.hAdd is not pretty-printed as +
        let state ← withOptions (fun o => o.set `pp.notation false |>.set `pp.fullNames true) <| Meta.ppGoal g.mvarId!
        return state.pretty (width := 100000000)
  return state

/-- Premises from a module whose name has one of the following as a component are not retrieved. -/
def moduleBlackList : Array String := #[
  "Aesop", "Auto", "Cli", "CodeAction", "DocGen4", "Duper", "ImportGraph", "Lake", "Lean", "LeanSearchClient", "Linter", "Mathport",
  "MD4Lean", "Plausible", "ProofWidgets", "Qq", "QuerySMT", "Tactic", "TacticExtra", "Test", "Testing", "UnicodeBasic", "Util"
]

/-- A premise whose name has one of the following as a component is not retrieved. -/
def nameBlackList : Array String := #["Lean", "Lake", "Qq"]

/-- A premise whose `type.getForallBody.getAppFn` is a constant that has this prefix is not retrieved. -/
def typePrefixBlackList : Array Name := #[`Lean]

def isBlackListedPremise (env : Environment) (name : Name) : Bool := Id.run do
  if name == ``sorryAx then return true
  if name.isInternalDetail then return true
  if nameBlackList.any (fun p => name.anyS (· == p)) then return true
  -- let some moduleName := env.getModuleFor? name | return true
  -- if moduleBlackList.any (fun p => moduleName.anyS (· == p)) then return true
  let some ci := env.find? name | return true
  match ci with
  | .thmInfo _ => pure ()
  | .defnInfo _ => if theoremsOnly then return true
  | _ => return true
  if let .const fnName _ := ci.type.getForallBody.getAppFn then
    if typePrefixBlackList.any (fun p => p.isPrefixOf fnName) then return true
  return false

structure Result where
  name : Name
  module : Name
  initialProofState : String
  decl : String
deriving Repr, ToJson

open Elab.IO in
def allProofStatesFromModule (targetModule : Name) (decls : Option (List Name)) (proofAsSorry? : Bool) : IO (Array Result) := do
  -- Don't extract anything from blacklisted modules
  if moduleBlackList.any (fun p => targetModule.anyS (· == p)) then
    return #[]

  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean targetModule).toString

  /- Replace all tactics with "sorry" for faster execution -/
  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)

  let options := if proofAsSorry? then proofAsSorry else {}
  let steps := processInput' (← moduleSource targetModule) none options fileName
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut results : Array Result := #[]

  for (cmd, ci) in targets do
    let ci_name_stem := ci.name.toString.splitOn "." |>.getLast! |>.toName
    if (decls.isSome && !(decls.get!.contains ci_name_stem)) then
      continue

    let env := cmd.after
    let name := ci.name
    unless isBlackListedPremise env name do
      let result : Result := {
        name := name,
        module := targetModule,
        initialProofState := ← getInitialProofState env ci,
        decl := toString cmd.src
      }
      results := results.push result

  return results

def extractStatesMain (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "module" |>.as! ModuleName
  let results ← allProofStatesFromModule module none true
  for result in results do
    IO.println (toJson result).compress
  return 0

def extract_states : Cmd := `[Cli|
  extract_states VIA extractStatesMain; ["0.0.1"]
  "Print a jsonl file of all theorems in a module along with their initial states and proofs."

  ARGS:
    module : ModuleName; "Lean module to compile and extract states from."
]

-- /-- `lake exe extract_states` -/
def main (args : List String) : IO UInt32 :=
  extract_states.validate args

-- #eval main ["Mathlib.Data.Nat.Prime.Basic"]
