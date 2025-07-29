import ImProver.metrics.tagger
import metrics.examples

open Lean Core Elab IO Meta Term Command Tactic Cli



structure ExampleData extends TheoremData where
  improved : String := ""
  deriving ToJson, FromJson


def groupByTagName (targetData : List (CompilationStep × ConstantInfo × Name × Version))
  : IO (Array (Name × Array (CompilationStep × ConstantInfo × Version))) := do

  let mut grouped : Array (Name × Array (CompilationStep × ConstantInfo × Version)) := #[]

  for (cmd, ci, tagName, version) in targetData do
    let existingGroup? := grouped.findIdx? (fun (name, _) => name == tagName)
    match existingGroup? with
    | some idx =>
      let (name, existing) := grouped[idx]!

      -- push only if a singleton containing unoptimized or optimized.
      let version_subarray := existing.filterMap (fun (_, _, v) => if v == Version.unoptimized || v == Version.optimized then some v else none)
      if version_subarray.size == 1 && (
          (version == .unoptimized && version_subarray[0]! == .optimized)
        || (version == .optimized && version_subarray[0]! == .unoptimized)
        ) then
        grouped := grouped.set! idx (name, existing.push (cmd, ci, version))
      else
        IO.println s!"Unexpected versioning in {tagName}: {version} with existing versions: {version_subarray}, taking only first"

    | none =>
      grouped := grouped.push (tagName, #[(cmd, ci, version)])
  return grouped

def checkValidity (grouped : Array (Name × Array (CompilationStep × ConstantInfo × Version))) : IO (Array (Name × CompilationStep × ConstantInfo × String)) := do
  let mut validPairs : Array (Name × CompilationStep × ConstantInfo × String) := #[]

  for (tagName, items) in grouped do
    if items.size != 2 then
      IO.println s!"Error: tagName {tagName} has {items.size} versions, expected exactly 2 (optimized and unoptimized)"
      continue

    let versions := items.map (fun (_, _, v) => v)
    let hasUnoptimized := versions.any (· == Version.unoptimized)
    let hasOptimized := versions.any (· == Version.optimized)

    if !hasUnoptimized || !hasOptimized then
      IO.println s!"Error: tagName {tagName} missing required versions (needs both optimized and unoptimized)"
      continue

    let outputStr ← match items.find? (fun (_, _, v) => v == Version.optimized) with
    | some (cmd, _, _) => pure cmd.src.toString
    | none => do
      IO.println s!"Error: could not find optimized version for {tagName}"
      continue

    -- Find the unoptimized version to use as the base
    match items.find? (fun (_, _, v) => v == Version.unoptimized) with
    | some (cmd, ci, _) =>
      validPairs := validPairs.push (tagName, cmd, ci, outputStr)
    | none =>
      IO.println s!"Error: could not find unoptimized version for {tagName}"
      continue

  return validPairs

def getExamples (mod : Name) (outputFile : String) (python_cmd : String) (rag_id : String) (k : Nat) : IO Unit := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)
  let targets_flat ← targets.force
  IO.println s!"Found {targets_flat.length} compilation steps for module {mod}"
  for (cmd, ci) in targets_flat do
    -- IO.println s!">> Processing {ci.name.toString} in {mod.toString} with command {cmd.src.toString}"
  let final_env? := match targets_flat.getLast? with
  | none => none
  | some (cmd,_) => some cmd.after

  match final_env? with
  | none => pure ()
  | some final_env =>
    let declNames : List (Name × Name) := nameAttribute.ext.getState final_env |>.toList
    let declVersions : List (Name × Version) := versionAttribute.ext.getState final_env |>.toList
    let declVersionNames : List (Name × Name × Version) := declNames.filterMap (fun (n, tagName) => match declVersions.find? (fun (n', (_ : Version)) => n' == n) with
    | some items => (n, tagName, items.2)
    | none => none)
    let targetData :  List (CompilationStep × ConstantInfo × Name × Version) := targets_flat.filterMap (fun (cmd, ci) =>
      let data? := declVersionNames.find? (fun (n, _, _) => n == ci.name)
      match data? with
      | none => none
      | some (_, tagName, version) =>
        some (cmd, ci, tagName, version))
    -- want to group the previous first by tagName, and then as either auto, or an unoptimized, optimized pair.
    -- collect all the auto and unoptimized versions, get their prompts via getPromptsAux, and then reinsert the batched results into the object
    let grouped ← groupByTagName targetData


    -- ensure that the inner array contains two elements, one unoptimized and one optimized. if this is confirmed,
    -- then flatten this array to get a single item of type Name × CompilationStep × ConstantInfo × String, where the compilationstep, constantinfo, are from the item corresponding to the unoptimized version.
    -- if this is unable to be confirmed, print an error message on how this tagName has an unexpected number of versions, aind continue to the next tagName.

    let validPairs ← checkValidity grouped

    let raw_data : List (List TheoremData × Nat) ← getPromptsAux (validPairs.map (fun (_, cmd, ci, _) => (cmd, ci))) mod python_cmd fileName rag_id k
    let mut results : Array (Name × ExampleData) := #[]

    for i in [0:validPairs.size] do
      if h : i < validPairs.size then
        let (tagName, _, _, outputStr) := validPairs[i]
        match raw_data.get? i with
        | some (theoremDataList, _) =>
          let realTheoremData := theoremDataList.filter (fun td => !td.id.isExtracted) |>.get! 0
          results := results.push (tagName, ⟨realTheoremData, outputStr⟩)
         | none =>
          IO.println s!"Error: no raw_data found for index {i} (tagName: {tagName})"

    -- Write the results to JSON files
    let jsonOutput := Json.mkObj (results.toList.map (fun (tagName, data) =>
      (tagName.toString, toJson data)))

    let json_path := outputFile
    IO.println s!"Writing to {json_path}"

    if not (← System.FilePath.pathExists json_path) then
      let parent := System.FilePath.parent json_path
      match parent with
      | some path =>
        IO.println path
        IO.FS.createDirAll path
      | none => pure ()



    IO.FS.writeFile json_path (ToString.toString jsonOutput)







def getExamplesCLI (args : Cli.Parsed) : IO UInt32 := do
  let module := args.positionalArg! "file" |>.as! ModuleName
  let outputFile := args.positionalArg! "outputFile" |>.as! String
  let python_cmd := args.positionalArg! "pythonCommand" |>.as! String
  let mod :Name := module
  let rag_id := args.positionalArg! "rag_id" |>.as! String
  let k := args.positionalArg! "k" |>.as! Nat




  getExamples mod outputFile python_cmd rag_id k
  return 0


def get_examples : Cmd := `[Cli|
  get_examples VIA getExamplesCLI; ["0.0.1"]
"Generate examples for ImProver."

  ARGS:
    file : ModuleName; "Lean module to get prompts for."
    outputFile : String; "Where to save the Json output."
    pythonCommand : String; "Path to python executable."
    rag_id : String; "ID for the rag directory, used to identify the source of the prompts in the database."
    k : Nat; "Number of RAG results to use for each prompt."
]


def main (args : List String) : IO UInt32 :=
  get_examples.validate args
