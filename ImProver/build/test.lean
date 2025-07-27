import Cli.Basic
import ImportGraph.Imports
import Mathlib.Lean.CoreM


open Lean Cli Environment NameMap

set_option autoImplicit true


def importGraphWithDepth (env : Environment) (depth : Nat) : NameMap (Array Name) := Id.run do
  let main := env.header.mainModule
  let initialImports := env.header.imports.map Import.module
  let mut m : NameMap (Array Name) := ({} : NameMap _).insert main initialImports
  let mut seen : NameSet := NameSet.empty.insert main
  let mut frontier : Array Name := initialImports

  for _ in [0:depth] do
    let mut nextFrontier : Array Name := #[]
    for mod in frontier do
      if !seen.contains mod then
        let modImports := env.importsOf mod
        m := m.insert mod modImports
        nextFrontier := nextFrontier ++ modImports
        seen := seen.insert mod
    frontier := nextFrontier
  m




def test (mods : List Name) (test_mod : Name) : IO UInt32 := do
  initSearchPath (← getLibDir (← findSysroot))

  let _ ← CoreM.withImportModules mods.toArray do
    let env ← getEnv
    let g := importGraph env
    let g := transitiveClosure g

    -- let graph_output := g.find? test_mod
    let gs :="\n".intercalate <| (g.toList.map (fun (n,imports) => s!"{n} -> {imports.toList}"))
    IO.println s!">> Graph: \n{gs}"


  return 0



def buildRagCLI (args : Cli.Parsed) : IO UInt32 := do

  let modules_raw := args.positionalArg! "modules" |>.as! String
  let modules : List String := if modules_raw.isEmpty then [] else modules_raw.splitOn ","
  let mods := modules.map (fun m => m.toName)

  test mods mods[0]!


def build_rag : Cmd := `[Cli|
  build_rag VIA buildRagCLI; ["0.0.1"]
"Test import graph."

  ARGS:
    modules : String; "List of modules to include in the prompts, separated by \",\"."
]


def main (args : List String) : IO UInt32 :=
  build_rag.validate args


#eval main ["Mathlib.Data.Set.Basic,Mathlib.Data.Real.Basic"]
