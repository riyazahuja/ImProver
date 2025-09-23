import Cli.Basic
import TrainingData.Utils.context

open Lean Cli Environment NameMap
open Lean Lean.Core Lean.Elab IO Lean.Elab.IO Lean.Meta Lean.Elab.Term Lean.Elab.Command Lean.Meta.Tactic
  Lean.Elab.Tactic Cli System String Cli.String

set_option autoImplicit true




structure RAGModule where
  module : Name
  imports : Array Name
  fullImports : Array Name
  depth : Nat
  deriving Inhabited, ToJson, FromJson








structure RAGDecl where
  decl : Name
  kind : String
  module : Name
  content : String
  deriving Inhabited, FromJson, ToJson





def test (mods : List Name) (test_mod : Name) : IO UInt32 := do
  initSearchPath (← getLibDir (← findSysroot))

  let _ ← CoreM.withImportModules mods.toArray do
    let env ← getEnv
    let g := importGraph env

    let graph_output := g.find? test_mod
    IO.println s!">> Graph: \n{graph_output}"


  return 0


/-

 I give dataset, we collect all the modules, import all of them, and then
 we get the importgraph (up to a user-specified depth) to have a big list of modules.
 For each module in the importgraph of the dataset, we note its depth from the dataset (idk how yet) and also what it directly imports and indirecly imports.
 Then in the main import environment, we also then look at all the constants we have access to.
 Of these constants, we filter to things in the importgraph and then we get the contents of these constants.
 I.e. take the `env.constants` and filter it by the importgraph, then given the filters, group by module source, and then get contents and kinds of each.

Save this into a database of each item being typed as `(module, imports, all_imports)` in one
table and `(module, decl, kind, contents)` in onther table.
-/
def getModuleData (mods : Array Name) : CoreM (Array RAGModule) := do
  let env ← getEnv
  let ig := importGraph env

  let tc := transitiveClosure ig


  -- Compute the depth of each module from the starting modules (mods)
  -- We'll use a BFS to assign depths, starting from mods (depth 0)
  let mut depthMap : NameMap Nat := {}
  let mut queue : Array (Name × Nat) := mods.map (fun m => (m, 0))
  let mut seen : NameSet := {}
  while !queue.isEmpty do
    let (curr, d) := Array.back! queue
    queue := queue.pop
    if seen.contains curr then
      continue
    seen := seen.insert curr
    if depthMap.contains curr then
      continue
    depthMap := depthMap.insert curr d
    if let some directImports := ig.find? curr then
      for imp in directImports do
        if !seen.contains imp then
          queue := queue.push (imp, d + 1)

  -- Now, for every module in ig, build a RAGModule with its depth (if present), direct imports, and full imports
  let out : Array RAGModule :=
    ig.toList.map (fun (mod, imports) =>
      let fullImports := tc.findD mod NameSet.empty |>.toArray
      let depth := depthMap.find? mod |>.getD (panic! s!"No depth for {mod}")
      { module := mod, imports := imports, fullImports := fullImports, depth := depth }
    ) |>.toArray

  return out

def getDeclData (mods : Array Name) (constants : Option (Array Name)): CoreM (Array RAGDecl) := do
  let env ← getEnv
  let decls := env.constants.map₁.toList.map (fun (n, cinfo) => (n, cinfo))
  IO.println s!"There are {decls.length} declarations in the environment"
  let decls_with_module := decls.map (fun (n, ci) => (n, ci, env.getModuleFor? n |>.getD .anonymous))
  let decls_filtered := decls_with_module.filter (fun (n, _, m) => mods.contains m && m != .anonymous && !isAuxLemma n && (constants.isNone || n ∈ constants.getD #[]))
  IO.println s!"There are {decls_filtered.length} declarations in the environment after filtering"
  let decls_with_kind := decls_filtered.map (fun (n, ci, m) => (n, ci, m, getKind' ci))

  let decls_with_range ← decls_with_kind.mapM (fun (n, ci, m, k) => do
    return (n, ci, m, k, (← findDeclarationRanges? n))
    )

  let total_non_null := decls_with_range.filter (fun (_,_,_,_,rgs) => rgs.isSome)
  IO.println s!"There are {total_non_null.length} declarations in the environment after filtering for human theorems"

  let mut decls_filtered_grouped := Std.HashMap.empty
  for (n, ci, m, k, rgs) in decls_with_range do

    if let some decls := decls_filtered_grouped[m]? then
      decls_filtered_grouped := decls_filtered_grouped.insert m (decls.push (n, ci, k, rgs))
    else
      decls_filtered_grouped :=decls_filtered_grouped.insert m #[(n, ci, k, rgs)]






  let mut output : Array RAGDecl := #[]

  for (m,decls) in decls_filtered_grouped do
    let modulePath ← findLean m
    let fileContents ← IO.FS.readFile modulePath.toString
    let fileMap := fileContents.toFileMap

    for (n, _, k, rgs?) in decls do
      if rgs?.isNone then
        continue
      let rgs := rgs?.get!.range

      -- Use proper position-based extraction instead of line-based
      let declText := fileMap.source.extract (fileMap.ofPosition rgs.pos) (fileMap.ofPosition rgs.endPos)
      output := output.push { decl := n, kind := k, module := m, content := declText }

  IO.println s!"There are {output.size} declarations in the output"

  return output







def buildRag (mod : Name) (constant : Name): IO Unit := do
  initSearchPath (← getLibDir (← findSysroot))



  let _ ← CoreM.withImportModules #[mod] do
    let env ← getEnv

    let ci := env.constants.map₁[constant]? |>.get!
    let module := env.getModuleFor? constant |>.get!
    let rgs := ( ← findDeclarationRanges? constant) |>.get! |>.range

    let modulePath ← findLean mod
    let fileContents ← IO.FS.readFile modulePath.toString

    -- IO.println fileContents

    let fileMap := fileContents.toFileMap


    -- Use proper position-based extraction instead of line-based
    let declText := fileMap.source.extract (fileMap.ofPosition rgs.pos) (fileMap.ofPosition rgs.endPos)

    -- Print the fileContents with position markers for the start of every line using fileMap
    for i in [:fileMap.positions.size] do
      let pos := fileMap.positions[i]!
      let nextPos := if i + 1 < fileMap.positions.size then fileMap.positions[i+1]! else fileMap.source.endPos
      let lineText := fileMap.source.extract pos nextPos
      IO.println s!"[pos {pos}] {lineText.trimRight}"


    IO.println s!"[{constant} | {mod}]\n\n{declText}\n\n{(fileMap.ofPosition rgs.pos)} -> {(fileMap.ofPosition rgs.endPos)}"


  return


#eval buildRag `Mathlib.GroupTheory.Index `Subgroup.index_map
