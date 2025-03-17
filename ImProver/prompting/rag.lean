
import Cli
import ImProver.prompting.state_comments
import ImProver.prompting.context
import ImProver.utils
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules
import ImportGraph.Imports



import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
-- import Compfiles

open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true





def initialize_retrieval (config : ImProverConfig) (step: CompilationStep) : IO ImProverConfig := do
  let env : NameMap (Array Name) := step.after.importGraph
  let nodes := env.toList.map (fun (n, _) => n) |>.eraseDup
  let output : ImProverConfig := {
    config with
    retrievalFilter := nodes
  }
  return output

def retrieve (step : CompilationStep) (config: ImProverConfig) : IO (List String) := do
  let query : String ← insert_state_comments step

  let data : Json := Json.mkObj
    [("query", Json.str query),
      ("k", Json.num <| JsonNumber.fromNat config.rag?),
     ("imports", Json.arr <| List.toArray <| config.retrievalFilter.map (fun n => Json.str (n.toString)))
    ]
  IO.println data.compress
  let out ← IO.Process.output {
    cmd := "/home/riyaza/miniconda3/envs/env/bin/python3",
    args := #["ImProver/prompting/rag.py", data.compress]
  }

  let stdout := out.stdout.trim
  IO.println stdout
  IO.println "ERROR:"
  IO.println out.stderr
  let items := stdout.splitOn "<BREAK>"
  let items := if items.isEmpty then [] else items.take (items.length - 1)
  return items


def retrieve_batch (steps : Array CompilationStep) (config: ImProverConfig) : IO (Array (CompilationStep × (List String))) := do
  let queries : Array String ← steps.mapM insert_state_comments

  let data : Json := Json.mkObj
    [("queries", Json.arr <| queries.map (fun q => Json.str q)),
      ("k", Json.num <| JsonNumber.fromNat config.rag?),
    --  ("imports", Json.arr <| List.toArray <| config.retrievalFilter.map (fun n => Json.str (n.toString)))
    ]

  let out ← IO.Process.output {
    -- cmd := "/home/riyaza/miniconda3/envs/env/bin/python3",
    cmd := "/Users/ahuja/Desktop/ImProver_new/.venv/bin/python3",
    args := #["ImProver/prompting/rag_batched.py", data.compress]
  }

  let stdout := out.stdout.trim
  -- IO.println out.stde/rr
  -- IO.println "OUT"
  -- IO.println out.stdout
  let json? := Json.parse stdout |>.toOption
  let json := match json? with
  | some j => j
  | none => Json.mkObj []

  let data : Array (Array String) := fromJson? json |>.toOption |>.getD #[]
  let data := data.map (fun d => d.toList)
  let out := steps.zip data
  return out







-- there is 100% something already in mathlib/lean core to find a substring
-- but I am too lazy to find it. please replace if you know what it is.
partial def findString (s pattern : String) : String × String :=
  if pattern.isEmpty then (s, pattern) else
  if s.length < pattern.length then (s, pattern) else
  let candidatePos := s.find ("".push · == pattern.take 1)
  let notContains := {s.toSubstring with stopPos := candidatePos}.toString
  let rest := {s.toSubstring with startPos := candidatePos}.toString
  if rest.startsWith pattern then
    (notContains, rest)
  else
    let (init, tail) := findString (rest.drop 1) pattern
    (notContains ++ (pattern.take 1) ++ init, tail)

def containsString (s pattern : String) : Option Nat :=
  let find := findString s pattern
  if find.1 == s && pattern != "" then none else
    find.1.length


-- temporary naive implementation: fix with better splitting and metadata


def extract_rag_from_prompt (prompt:String) : IO (List String) := do
  let startPos := containsString prompt "<RETRIEVED>"
  let endPos := containsString prompt "</RETRIEVED>"
  if startPos.isNone || endPos.isNone then
    IO.println "No <RETRIEVED> tag found in prompt"
    return []
  else
    let start := startPos.get!
    let end_ := endPos.get!
    let ragContent := prompt.drop (start + "<RETRIEVED>".length) |>.take (end_ - start - "<RETRIEVED>".length)
    let items := ragContent.splitOn "</DOC>"
    let items := if items.isEmpty then [] else items.take (items.length - 1)
    let items := items.map (fun item => item.replace "<DOC>" "" |>.trim)
    return items

def calculate_utilization (prompt : String) (current : String) : IO Float := do
  let items ← extract_rag_from_prompt prompt
  let mut found_ids : List String := []

  for item in items do
    let lines := item.splitOn "\n"
    for line in lines do
      let trimmed := line.trim

      let keywords := ["theorem ", "lemma ", "def "]
      for keyword in keywords do
        if trimmed.startsWith keyword then
          let parts := trimmed.drop keyword.length |>.trim |>.splitOn " "
          if !parts.isEmpty then
            found_ids := parts.head! :: found_ids

  -- now we have a list of strings "found_ids" that
  -- we need to search for in the "current" string
  -- the utilization will be the number of found_ids that are present in current,
  -- divided by the total number of found_ids

  if found_ids.isEmpty then
    return 0.0
  else
    let mut count := 0
    for id in found_ids do
      if (containsString current id).isSome then
        count := count + 1
    let utilization := count.toFloat / found_ids.length.toFloat
    return utilization
