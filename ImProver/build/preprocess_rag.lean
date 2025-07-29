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
  deriving Inhabited

instance : ToJson (Array RAGModule) where
  toJson m :=
  let modules := m.map (fun m =>
    (m.module.toString, Json.mkObj [
      ("imports", toJson m.imports),
      ("fullImports", toJson m.fullImports),
      ("depth", toJson m.depth)]))

  Json.mkObj (modules.toList)






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
def getModuleData (mods : List Name) : CoreM (Array RAGModule) := do
  let env ← getEnv
  let ig := importGraph env

  let tc := transitiveClosure ig


  -- Compute the depth of each module from the starting modules (mods)
  -- We'll use a BFS to assign depths, starting from mods (depth 0)
  let mut depthMap : NameMap Nat := {}
  let mut queue : Array (Name × Nat) := mods.toArray.map (fun m => (m, 0))
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

def getDeclData (mods : List Name) : CoreM (Array RAGDecl) := do
  let env ← getEnv
  let decls := env.constants.map₁.toList.map (fun (n, cinfo) => (n, cinfo))
  IO.println s!"There are {decls.length} declarations in the environment"
  let decls_with_module := decls.map (fun (n, ci) => (n, ci, env.getModuleFor? n |>.getD .anonymous))
  let decls_filtered := decls_with_module.filter (fun (n, _, m) => mods.contains m && m != .anonymous && !isAuxLemma n)
  IO.println s!"There are {decls_filtered.length} declarations in the environment after filtering"
  let decls_with_kind := decls_filtered.map (fun (n, ci, m) => (n, ci, m, getKind' ci))

  let decls_with_range ← decls_with_kind.mapM (fun (n, ci, m, k) => do
    return (n, ci, m, k, (← findDeclarationRanges? n))
    )

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

    for (n, _, k, rgs?) in decls do
      if rgs?.isNone then
        continue
      let rgs := rgs?.get!.range


      let declTextList := if rgs.pos.line == 0 then [] else
        fileContents.splitOn "\n"
          |>.drop (rgs.pos.line - 1)
          |>.take (rgs.endPos.line - rgs.pos.line + 1)
      let declText := "\n".intercalate declTextList
      output := output.push { decl := n, kind := k, module := m, content := declText }

  IO.println s!"There are {output.size} declarations in the output"

  return output







def buildRag (mods : List Name) (output_dir : String) : IO Unit := do
  initSearchPath (← getLibDir (← findSysroot))

  let _ ← CoreM.withImportModules mods.toArray do

    let module_data : Array RAGModule ← getModuleData mods
    let all_modules := module_data.map (fun m => m.module)
    -- IO.println s!"all_modules: {all_modules}"
    let decl_data : Array RAGDecl ← getDeclData all_modules.toList
    -- IO.println s!"output_dir: [{output_dir}]"
    let module_data_path := output_dir ++ "/module_data.json"
    -- IO.println s!"module_data_path: {module_data_path}"
    let decl_data_path := output_dir ++ "/decl_data.json"
    IO.FS.writeFile module_data_path (toJson module_data |>.pretty)
    IO.FS.writeFile decl_data_path (toJson decl_data |>.pretty)



  return




def buildRagCLI (args : Cli.Parsed) : IO UInt32 := do

  let modules_raw := args.positionalArg! "modules" |>.as! String
  let modules : List String := if modules_raw.isEmpty then [] else modules_raw.splitOn ","
  let mods := modules.map (fun m => m.toName)


  let output_dir := args.positionalArg! "output_dir" |>.as! String

  buildRag mods output_dir

  return 0
  -- test mods mods[0]!


def build_rag : Cmd := `[Cli|
  build_rag VIA buildRagCLI; ["0.0.1"]
"Build a RAG from a list of modules."


  ARGS:
    modules : String; "List of modules to include in the prompts, separated by \",\"."
    output_dir : String; "Directory to write the RAG to."

]


def main (args : List String) : IO UInt32 :=
  build_rag.validate args


-- #eval main ["Mathlib.Data.Set.Basic,Mathlib.Data.Real.Basic", "rag/test_rag"]
