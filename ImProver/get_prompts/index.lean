import Cli
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic


open Lean Core Elab IO Meta Term Command Tactic Cli

set_option autoImplicit true


def getKind (c : ConstantInfo) : String :=
  match c with
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"


def Lean.Name.isHumanTheorem (name : Name) : CoreM Bool := do
  let hasDeclRange := (← Lean.findDeclarationRanges? name).isSome
  -- let isTheorem ← Name.isTheorem name
  let notProjFn := !(← Lean.isProjectionFn name)
  return hasDeclRange --&& isTheorem
    && notProjFn

def String.splitAtString (s : String) (pattern : String): Option (String × String) :=
  if h : pattern.endPos.1 = 0 then none
  else
    have hPatt := Nat.zero_lt_of_ne_zero h
    let rec loop (pos : String.Pos) :=
      if h : pos.byteIdx + pattern.endPos.byteIdx > s.endPos.byteIdx then
        none
      else
        have := Nat.lt_of_lt_of_le (Nat.add_lt_add_left hPatt _) (Nat.ge_of_not_lt h)
        let eq := s.substrEq pos pattern 0 pattern.endPos.byteIdx
        if eq then
          -- Found a match, return split strings
          let before := s.extract 0 pos
          let after := s.extract (pos + pattern) s.endPos
          some (before, after)
        else
          have := Nat.sub_lt_sub_left this (lt_next s pos)
          loop (s.next pos)
      termination_by s.endPos.1 - pos.1
    loop 0


structure messageData where
  severity : String
  content : String
  deriving Repr, ToJson, FromJson

structure declarationData where
  name : Name
  content : String
  content_proof_replaced : String
  messages : Array messageData
  kind : String
  deriving Repr, ToJson, FromJson


def replaceProof (cmd : CompilationStep) (replacement : String) : Option String :=
  let tactics := InfoTree.tactics_new cmd.trees |>.map (fun t => (t.pp, FileMap.ofPosition t.ctx.fileMap t.range.1))
    if tactics.isEmpty then
      let splitAt? := cmd.src.toString.splitAtString ":="
      match splitAt? with
      | none => none
      | some (before, _) =>
        let new_thm := before ++ s!":= by {replacement}"
        some new_thm
    else
      let (_, range) := tactics[0]!

      let cmd_rng := cmd.stx.getPos?

      let sstr : Substring := ⟨cmd.src.str, cmd_rng.getD 0,  range⟩

      let new_thm := sstr.toString ++ replacement
      some new_thm

def getTheorems (payload : String) (replace_proof? : Option String): IO Unit := do
  searchPathRef.set compile_time_search_path%

  let steps := Lean.Elab.IO.processInput' (payload) none {} none

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut targets_new : Array (CompilationStep × ConstantInfo × String) := #[]

  for (cmd, ci) in targets do

    let pf_env := cmd.after
    let ctx : Core.Context := {fileName := "", fileMap := default}
    let state : Core.State := {env := pf_env}
    let isHuman := match (← CoreM.run (Lean.Name.isHumanTheorem ci.name) ctx state |>.toIO').toOption with
      | some x => x.1 -- currently modified to return something that is not necessarily a theorem
      | none => false
    if not isHuman then
      continue

    let kind := getKind ci

    targets_new := targets_new.push (cmd, ci, kind)

  let mut decls : Array declarationData := #[]
  for (cmd, ci, kind) in targets_new do
    let msgs := cmd.msgs
    let name := ci.name
    let content := cmd.src.toString
    let content_proof_replaced := match replace_proof? with
      | none => content
      | some rp => replaceProof cmd rp |>.getD content
    let toMessageData (m : Message) : IO messageData := do
      let sev := match m.severity with
      | .error => "error"
      | .warning => "warning"
      | .information => "information"
      let content ←  m.toString
      return {severity := sev, content := content}

    let messages ← msgs.toArray.mapM toMessageData
    decls := decls.push
      {name := name,
        content := content,
        content_proof_replaced := content_proof_replaced,
        messages := messages,
        kind := kind}
  let json_data := ToJson.toJson decls

  IO.println json_data.compress



def getTheoremsCLI (args : Cli.Parsed) : IO UInt32 := do
  let payload_bytes := args.positionalArg! "n" |>.as! Nat |>.toUSize
  let replace_proof := args.flag! "replace_proof" |>.as! String
  let replace_proof? := if replace_proof == "" then none else some replace_proof


  let payload_raw : ByteArray ← (← getStdin).read payload_bytes
  let payload_str? := String.fromUTF8? payload_raw
  match payload_str? with
  | none =>
    IO.println "Error: could not decode stdin as UTF-8"
    return 1
  | some payload =>
    getTheorems payload replace_proof?


  return 0


def get_theorems : Cmd := `[Cli|
  get_theorems VIA getTheoremsCLI; ["0.0.1"]
"Get theorems from a string."

  -- FLAGS:
  --   theorems : String; "List of theorems to include in the prompts, separated by \",\". If empty, all theorems in the module will be used."


  ARGS:
    n : Nat; "Bytes to read from stdin"
    replace_proof : String; "What to replace the proof with. Default is none"
  --   pythonCommand : String; "Path to python executable."
  --   rag_id : String; "ID for the rag directory, used to identify the source of the prompts in the database."
  --   k : Nat; "Number of RAG results to use for each prompt."

  EXTENSIONS:
    defaultValues! #[("replace_proof", "")]
]


def main (args : List String) : IO UInt32 :=
  get_theorems.validate args
