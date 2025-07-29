
import Cli
import ImProver.online.prompting.state_comments
import ImProver.online.prompting.context
import ImProver.online.utils
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





def initialize_retrieval (config : ImProverConfig) (step: CompilationStep) (python_cmd : String := "python"): IO ImProverConfig := do
  let env : NameMap (Array Name) := step.after.importGraph
  let nodes := env.toList.map (fun (n, _) => n) |>.eraseDup
  let output : ImProverConfig := {
    config with
    retrievalFilter := nodes
  }
  let _ ← IO.Process.output {
    cmd := python_cmd,
    args := #["ImProver/prompting/rag_new_initialize.py"]
  }
  return output


def retrieve (step : CompilationStep) (config: ImProverConfig)
  (python_cmd : String := "python"): IO (List String) := do
  let query : String ← insert_state_comments step

  let data : Json := Json.mkObj
    [("query", Json.str query),
      ("k", Json.num <| JsonNumber.fromNat config.rag?),
      ("imports", Json.arr <| List.toArray <| config.retrievalFilter.map (fun n => Json.str (n.toString))),
      ("prompt_id", Json.str config.prompt),
      ("name", Json.str ""), -- TODO
      ("module", Json.str step.after.mainModule.toString)
    ]
  IO.println data.compress
  let out ← IO.Process.output {
    cmd := python_cmd,
    args := #["ImProver/prompting/rag.py", data.compress]
  }

  let stdout := out.stdout.trim
  -- IO.println stdout
  -- IO.println "ERROR:"
  -- IO.println out.stderr
  let items := stdout.splitOn "<BREAK>"
  let items := if items.isEmpty then [] else items.take (items.length - 1)
  return items

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

-- test both on first step RAG or all steps RAG
def retrieve_batch (steps : Array (CompilationStep × ConstantInfo)) (config: ImProverConfig)
  (python_cmd : String := "python"): IO (Array (CompilationStep × (List String))) := do
  let queries : Array String ← steps.mapM (fun (cmd, ci) => do
    let env := cmd.after

    try
      getInitialProofState env ci
    catch e =>
      IO.println s!"Error getting initial proof state on {ci.name}: {e}"
      let tacs :=cmd.trees
        |>.flatMap InfoTree.retainTacticInfo
        |>.flatMap InfoTree.retainOriginal
        |>.flatMap InfoTree.retainSubstantive
        |>.flatMap InfoTree.tactics
      match tacs with
      | [] => pure ""
      | i::_ => pure <| ((← i.mainGoalStateBefore)).pretty 1000000

  )

  -- IO.println queries
  let data : Json := Json.mkObj
    [("queries", Json.arr <| queries.map (fun q => Json.str q)),
      ("k", Json.num <| JsonNumber.fromNat config.rag?),
     ("imports", Json.arr <| List.toArray <| config.retrievalFilter.map (fun n => Json.str (n.toString)))
    ]
  IO.println data.compress
  let out ← IO.Process.output {
    cmd := python_cmd,
    -- cmd := "/Users/ahuja/Desktop/ImProver_new/.venv/bin/python3",
    args := #["ImProver/prompting/rag_batched.py", data.compress]
  }

  let stdout := out.stdout.trim
  IO.println out.stderr
  IO.println "OUT"
  IO.println out.stdout
  let json? := Json.parse stdout |>.toOption
  let json := match json? with
  | some j => j
  | none => Json.mkObj []

  let data : Array (Array String) := fromJson? json |>.toOption |>.getD #[]
  let data := data.map (fun d => d.toList)
  let cmds := steps.map (fun (c, _) => c)
  let out := cmds.zip data
  return out



def retrieve_batch_indep (steps : Array (CompilationStep × ConstantInfo)) (mod : Name) (prompt_id : String)
  (python_cmd : String := "/home/riyaza/miniconda3/envs/env/bin/python") : IO (Array (CompilationStep × (List String))) := do
  IO.println "Retrieving batch independently"

  -- let queries : Array String ← steps.mapM (fun (cmd, ci) => do
  --   let env := cmd.after

  --   try
  --     getInitialProofState env ci
  --   catch e =>
  --     IO.println s!"Error getting initial proof state on {ci.name}: {e}"
  --     let tacs :=cmd.trees
  --       |>.flatMap InfoTree.retainTacticInfo
  --       |>.flatMap InfoTree.retainOriginal
  --       |>.flatMap InfoTree.retainSubstantive
  --       |>.flatMap InfoTree.tactics
  --     match tacs with
  --     | [] => pure ""
  --     | i::_ => pure <| ((← i.mainGoalStateBefore)).pretty 1000000

  -- )

  -- IO.println queries
  let data : Json := Json.mkObj
    [("queries", Json.arr <| steps.map (fun (cmd, ci) =>
      Json.mkObj
        [("module", Json.str mod.toString),
         ("name", Json.str ci.name.toString)]
    )),
      ("k", Json.num 10)
    ]
  IO.println data.compress
  let out ← IO.Process.output {
    cmd := python_cmd,
    -- cmd := "/Users/ahuja/Desktop/ImProver_new/.venv/bin/python3",
    args := #["ImProver/online/prompting/rag_batched.py",
      data.compress,
      "--prompt_id", prompt_id
    ]
  }

  let stdout := out.stdout.trim
  IO.println "================================================"
  IO.println "\n>>> RAG OUTPUT <<<\n"
  IO.println stdout
  IO.println "\n>>> RAG ERROR <<<\n"
  IO.println out.stderr
  IO.println "================================================"
  -- IO.println out.stderr
  -- IO.println "OUT"
  -- IO.println out.stdout

  let json? := Json.parse stdout |>.toOption
  let json := match json? with
  | some j => j
  | none => Json.mkObj []

  let data : Array (Array String) := fromJson? json |>.toOption |>.getD #[]
  let data := data.map (fun d => d.toList)
  let cmds := steps.map (fun (c, _) => c)
  let out := cmds.zip data
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
