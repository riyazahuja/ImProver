import ImProver.online.prompting.state_comments
import ImProver.online.prompting.context
import Cli
import ImProver.online.prompting.prompts
import ImProver.online.prompting.rag
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.HumanTheorem
import ImportGraph.RequiredModules
import ImportGraph.Imports
import TrainingData.TreeParser
import TrainingData.ExtractGoal
import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import ImProver.online.c2
import ImProver.get_prompts.utils
import ImProver.get_prompts.where_with_end
import Lean.Elab.Command


open Lean Core Elab IO Meta Term Command Tactic Cli Environment NameMap Std

set_option autoImplicit true

structure InformalData where
  module : Name
  decl : String
  informal_statement : String
  informal_proof : String
  deriving Inhabited, ToJson, FromJson

def getInformalData (mods : List Name) (promptsDirectory : String) : IO (Except String (Std.HashMap Name (Std.HashMap String (String × String)))) := do
  -- let mut informal_data : Std.HashMap (Name × String) String := Std.HashMap.empty

  let database_path := promptsDirectory ++ "/informal_data.duckdb"
  let SQL_cmd : String := s!"SELECT module, decl, informal_statement, informal_proof FROM informal_data;"

  let output ← IO.Process.output {
    cmd := "duckdb"--"/home/riyaza/.local/bin/duckdb",
    args := #[database_path, "--readonly", "--json", "-c", SQL_cmd]}

  if output.exitCode != 0 then
    return .error s!"Error running duckdb: {output.stderr}"

  let json? : Option Json := Json.parse output.stdout |>.toOption
  if json?.isNone then
    return .error s!"Error parsing JSON: {output.stdout}"

  let json := json?.get!
  let informal_data_raw? := (Json.getArr? json |>.toOption.getD #[]).mapM (fun v => @FromJson.fromJson? InformalData _ v) |>.toOption
  if informal_data_raw?.isNone then
    return .error s!"Error parsing JSON to InformalData: {output.stdout}"

  let informal_data_raw : Array InformalData := informal_data_raw?.get!


  let mut output := Std.HashMap.empty
  for d in informal_data_raw do
    if mods.contains d.module then
      let in_output? := output[d.module]?
      match in_output? with
        | some declMap =>
          output := output.insert d.module (declMap.insert d.decl (d.informal_statement, d.informal_proof))
        | none =>
          output := output.insert d.module (Std.HashMap.empty.insert d.decl (d.informal_statement, d.informal_proof))





  return .ok output

namespace OLeanSearch
-- Written by Tate :D
open Lean System IO Core Meta Elab

def CoreM.withImportModules {α : Type} (modules : Array Name) (run : CoreM α)
    (searchPath : Option Lean.SearchPath := none) (options : Options := {})
    (trustLevel : UInt32 := 0) (fileName := "") :
    IO α := unsafe do
  if let some sp := searchPath then searchPathRef.set sp
  Lean.withImportModules (modules.map (fun m => Import.mk m false)) options (trustLevel := trustLevel) fun env =>
    let ctx := {fileName, options, fileMap := default}
    let state := {env}
    Prod.fst <$> (CoreM.toIO · ctx state) do
      run

structure DeclInfo where
  nameString : String
  moduleString : String
  kind : String
  src : String
deriving Repr, ToJson

def findLean (mod : Name) : IO FilePath := do
  let srcSearchPath : Lean.SearchPath ← initSrcSearchPath
  if let some fname ← srcSearchPath.findModuleWithExt "lean" mod then
    return fname
  else
    let fname := FilePath.mk ((← findOLean mod).toString.replace ".lake/build/lib/" "") |>.withExtension "lean"
    if !(← fname.pathExists) then
      throw <| IO.userError s!"Path to {mod} not found"
    return fname

def getAllConstantInfos (modules : Array Name) : IO (List DeclInfo) := do
  initSearchPath (← getLibDir (← findSysroot))
  unsafe Lean.enableInitializersExecution
  let env ← importModules (modules.map (Import.mk · false)) Options.empty (leakEnv := true)

  return ← CoreM.withImportModules modules do
    let mut infos := []
    for ⟨name, ci⟩ in env.constants do
      let mod ← Lean.findModuleOf? name
      match mod with
      | none => continue
      | some module =>
        -- if (ci.isTheorem || ci.isDefinition) && (module != Name.anonymous) && (!module.toString.startsWith "Lean") && (!module.toString.startsWith "Batteries") && (!module.toString.startsWith "Aesop") && (!module.toString.startsWith "Init") && (!module.toString.startsWith "Std") then
        if (ci.isTheorem || ci.isDefinition) && (module != Name.anonymous) && (modules.contains module) then

          let kind := if ci.isTheorem then "theorem" else "definition"
          match (← findDeclarationRanges? name) with
          | none => continue
          | some rgs =>
            let modulePath ← findLean module
            let fileContents ← IO.FS.readFile modulePath.toString

            let declTextList := if rgs.range.pos.line == 0 then [] else
              fileContents.splitOn "\n"
                |>.drop (rgs.range.pos.line - 1)
                |>.take (rgs.range.endPos.line - rgs.range.pos.line + 1)
            let declText := "\n".intercalate declTextList

            infos := (DeclInfo.mk name.toString module.toString kind declText) :: infos

    return infos
end OLeanSearch


def augmentData (mods : List Name) (promptsDirectory : String) : IO Unit := do
  searchPathRef.set compile_time_search_path%

  let graph : NameMap NameSet ← CoreM.withImportModules mods.toArray do
    return transitiveClosure (importGraph (← getEnv))

  -- let informal_data? ← getInformalData mods promptsDirectory

  -- if not informal_data?.isOk then
  --   let msg := match informal_data? with
  --     | .error e => e
  --     | .ok _ => "Unknown error"
  --   IO.println s!"[ERROR] {msg}"
  --   return

  -- let informal_data := informal_data?.toOption.get!
  let informal_data := Std.HashMap.empty.insert `Mathlib.Algebra.Group.Basic (Std.HashMap.empty.insert "div_eq_div_mul_div" ("informal_statement", "informal_proof"))


  for mod in mods do
    let json_path := promptsDirectory ++ "/src/" ++ mod.toString.replace "." "/" ++ ".json"
    let json_contents ← IO.FS.readFile json_path
    let json? := Json.parse json_contents |>.toOption
    if json?.isNone then
      IO.println s!"[ERROR] Error parsing JSON for {mod}: {json_contents}"
      continue

    let json := json?.get!
    let file_data? := @FromJson.fromJson? FileData _ json |>.toOption
    if file_data?.isNone then
      IO.println s!"[ERROR] Error parsing JSON for {mod}: {json_contents}"
      continue

    let file_data : FileData := file_data?.get!


    let declMap? := informal_data[mod]?
    if declMap?.isNone then
      IO.println s!"[ERROR] No informal data for {mod}"
      continue

    let declMap := declMap?.get!

    let augmented_theorems := file_data.theorems.map (fun thm =>
      let informal_data? := declMap[thm.id.name.toString]?
      match informal_data? with
        | some (informal_statement, informal_proof) =>
        -- TODO double check that this works if one of the fields is null
          {thm with informal_statement := some informal_statement, informal_proof := some informal_proof}
        | none => thm
    )

    let importgraph := graph.find? mod |>.map (fun x => x.toList)

    let augmented_file_data := {file_data with theorems := augmented_theorems, importGraph := importgraph}

    IO.FS.writeFile json_path (ToJson.toJson augmented_file_data |>.pretty)

  let all_descendants := graph.toList.map (fun (_, imports) => imports.toList) |>.flatten
  let all_descendants_deduped := all_descendants.eraseDups
  let all_descendants_deduped_trimmed := all_descendants_deduped.filter (fun mod => not (mods.contains mod))
  -- let all_descendants_deduped_trimmed := [`Mathlib.Algebra.Group.Basic]
  let all_declarations ← OLeanSearch.getAllConstantInfos all_descendants_deduped_trimmed.toArray

  match all_declarations.length with
  | 0 => return
  | _ =>
    let database_path := promptsDirectory ++ "/dependency_data.duckdb"
    -- This is an awful way to do this, but should be ok for now
    let SQL_cmd := "CREATE TABLE IF NOT EXISTS dependency_data (module TEXT, decl TEXT, kind TEXT); " ++
                  "INSERT INTO dependency_data (module, decl, kind) VALUES " ++
                  ((all_declarations.map (fun d =>
                    s!"('{d.moduleString.replace "'" "''"}', '{d.nameString.replace "'" "''"}', '{d.kind.replace "'" "''"}')")) |>.foldr (fun acc x => acc ++ ",\n" ++ x) "") ++ ";"

    let output ← IO.Process.output {
      cmd := "duckdb"
      args := #[database_path, "-c", SQL_cmd]}

    if output.exitCode != 0 then
      IO.println SQL_cmd
      IO.println s!"[ERROR] Error running dependency duckdb: {output.stderr}"
      return

-- #eval augmentData [`Mathlib.Algebra.Group.Basic, `Mathlib.Algebra.Group.Defs] "/home/trowney/ImProver/prompts/final_final_train"

-- -- Given: all_descendants_deduped_trimmed : List Name
-- -- Goal: For each module in this list, efficiently extract the source code (as a string) of every statement (def, theorem, etc.) in the module, using the .olean files.

-- open Lean

-- def getModuleStatementsFromOlean (mod : Name) : IO (List (Name × String)) := do
--   -- Find the .olean file for the module
--   let oleanPath ← Lean.findOLean mod
--   -- Read the module data from the .olean file
--   let (modData, _) ← Lean.readModuleData oleanPath
--   -- For each constant in the module, try to get its source code
--   let mut stmts : List (Name × String) := []
--   for cinfo in modData.constants do
--     let declName := cinfo.name
--     -- Try to get the source code for this declaration
--     -- This requires the .olean to have been built with source info (debug info)
--     -- If not available, we can only get the pretty-printed declaration
--     let src? ← try
--       -- Try to get the source location (if available)
--       match cinfo.value? with
--       | some val =>
--         match val.ctor with
--         | .thmInfo thmVal =>
--           match thmVal.source? with
--           | some src => pure (some src)
--           | none => pure none
--         | .defnInfo defnVal =>
--           match defnVal.source? with
--           | some src => pure (some src)
--           | none => pure none
--         | _ => pure none
--       | none => pure none
--     catch _ => pure none
--     -- If we have source code, use it; otherwise, pretty-print the declaration
--     let stmtStr ← match src? with
--       | some src => pure src
--       | none =>
--         -- Fallback: pretty-print the declaration as Lean code
--         pure (toString cinfo)
--     stmts := stmts ++ [(declName, stmtStr)]
--   return stmts

-- -- For all modules in all_descendants_deduped_trimmed, collect all statements
-- let allModuleStatementsIO : IO (List (Name × Name × String)) := do
--   let mut result : List (Name × Name × String) := []
--   for mod in all_descendants_deduped_trimmed do
--     let stmts ← getModuleStatementsFromOlean mod
--     for (declName, stmtStr) in stmts do
--       result := result ++ [(mod, declName, stmtStr)]
--   return result

-- -- Example usage: (uncomment to run)
-- -- let allStmts ← allModuleStatementsIO
-- -- for (mod, decl, src) in allStmts do
-- --   IO.println s!"Module: {mod}, Decl: {decl}\n{src}\n---"

-- -- Now you have, for each module in all_descendants_deduped_trimmed, a list of (declName, source code as string) for every statement in the module, efficiently extracted from the .olean files.







def preprocessRagCLI (args : Cli.Parsed) : IO UInt32 := do
  let outputDirectory := args.positionalArg! "outputDirectory" |>.as! String
  -- let modules_raw : String := match args.flag? "modules" with
  -- | some x => x |>.as! String
  -- | none => ""
  let modules_raw := args.positionalArg! "modules" |>.as! String
  let modules : List String := if modules_raw.isEmpty then [] else modules_raw.splitOn ","
  let mods := modules.map (fun m => m.toName)

  augmentData mods outputDirectory
  return 0


def preprocess_rag : Cmd := `[Cli|
  preprocess_rag VIA preprocessRagCLI; ["0.0.1"]
"Preprocess the RAG data."

  ARGS:
    modules : String; "List of modules to include in the prompts, separated by \",\"."
    outputDirectory : String; "Where to save the Json output."
]


def main (args : List String) : IO UInt32 :=
  preprocess_rag.validate args
