import scripts.training_data
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import Batteries.Lean.Util.Path
import ImportGraph.RequiredModules

import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import TrainingData.TreeParser
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Batteries.Lean.Util.Path
import Batteries.Data.String.Basic
import Mathlib.Tactic.Change
import ImportGraph.RequiredModules
import Cli


open Lean Elab Term Command Frontend Parser
open Lean Elab IO Meta
open Cli System



def test_env (args: String) : IO UInt32 := do
  let options := Options.empty.insert `maxHeartbeats (0 : Nat)
  let module := args.toName
  searchPathRef.set compile_time_search_path%

  let steps : List CompilationStep ← compileModule module
  -- let consts_steps := steps.map (fun cs => cs.after.constants.map₁)
  let last := steps.getLast?
  match last with
  | none => return 0
  | some cs =>
    let consts := cs.after.constants.map₁
    println <| consts.toList.map (fun (n,_)=>n)



  -- let const_m := CoreM.withImportModules [module].toArray (options := options) do
  --   let env ← getEnv
  --   return env.constants.map₁

  -- println s!"STEPS:\n"
  -- for step in consts_steps do
  --   println <| step.toList.map (fun (n,_)=>n)
  --   println "=========================="

  -- println s!"Straight:\n"
  -- let duh ← const_m
  -- println <| duh.toList.map (fun (n,_)=>n)
  -- println "=========================="









  return 0

#eval test_env "Tests.tester"
