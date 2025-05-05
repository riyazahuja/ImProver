
import Cli
import TrainingData.InfoTree.Basic
import TrainingData.InfoTree.TacticInvocation.Basic
import ImportGraph.RequiredModules
import ImProver.utils
import ImProver.prompting.context
import ImProver.prompting.rag

import Lean.Util.SearchPath
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap
import TrainingData.Frontend
import TrainingData.InfoTree.ToJson
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.Utils.Range
import Mathlib.Data.String.Defs
import Mathlib.Lean.CoreM
import Mathlib.Tactic.Change
import Batteries.Lean.HashSet
import Batteries.Data.List.Basic
import Cli


open Lean Core Elab IO Meta Term Command Tactic Cli System

set_option autoImplicit true




def metric_length (cmd:CompilationStep) : IO Float :=
  return InfoTree.tactics_new cmd.trees |>.length |>.toFloat

def length_prompt : String := s!"Shorten the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem."


def metric_declarativity (cmd:CompilationStep) : IO Float :=
  let tac_stx := InfoTree.tactics_new (cmd.trees) |>.map (fun x => x.info.stx)
  let haves := tac_stx.filter (fun stx =>
    match stx with
    | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
    | _ => false)
  return haves.length |>.toFloat

def declarativity_prompt : String := s!"Shorten the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as declarative in style as possible. We define and measure declarativity as the number of explicitly typed \"have\" statements, which you will aim to maximize in order to construct a more readable, structured, and forward-reasoning approach to the proof as possible - while also ensuring that the output is still a correct proof of the theorem."



def metric_dependency (cmd:CompilationStep) : IO Float := do
  let context ← get_context cmd
  let external_deps := context.filter (fun c =>
    match c.kind with
    | "theorem (internal)" => true
    | "theorem" => true
    | _ => false)
  return external_deps.length |>.toFloat


def dependency_prompt : String := s!"Shorten the current Lean4 theorem (wrapped in <CURRENT>...</CURRENT>) to be as independent of external theorems and lemmas as possible. Namely, you aim to rewrite the proof to minimize the number of external dependencies - while also ensuring that the output is still a correct proof of the theorem."


def metric_completion (cmd:CompilationStep) : IO Float := do
  let msgs : List String ← cmd.msgs.filterMapM (fun msg : Message =>
      if msg.severity != .error then
        return none
      else do
        let m ← msg.data.toString
        return some (bombEmoji++m))
  return if cmd.trees.length == 0
    then 0
    else msgs.length |>.toFloat

def completion_prompt : String := s!"Prove the current theorem (wrapped in <CURRENT>...</CURRENT>) with a correct, formal, and complete (sorry-free) Lean4 proof."


-- def metric_readability (cmd:CompilationStep) :=
--   InfoTree.tactics_new cmd.trees |>.length |>.toFloat

-- def readability_prompt : String := s!"Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length - measured in the number of tactics in the proof - while also ensuring that the output is still a correct proof of the theorem."




/- Returns metric function from name (for ease of use from command line) -/
def get_metric (metric_name : String) : CompilationStep → IO Float :=
  match metric_name with
  | "length" => metric_length
  | "declarativity" => metric_declarativity
  | "dependency" => metric_dependency
  | "completion" => metric_completion
  -- | "readability" => metric_length
  | _ => fun _ => pure 0.0

/- Returns the prompt function from a name -/
def get_prompt (prompt_name : String)  (config : ImProverConfig) (cmd : CompilationStep) : IO String := do
  let main_prompt := match prompt_name with
  | "length" => length_prompt
  | "declarativity" => declarativity_prompt
  | "dependency" => dependency_prompt
  | "completion" => completion_prompt
  | _ => length_prompt

  let srcCommand := cmd.src.toString

  let annotation_prompt : String := s!" A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response."
  let annotation_string : String ← if config.annotation? then (insert_state_comments cmd) else pure ""

  let context_prompt : String := s!" The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>."
  let context_string : String ← if config.context? then do
      let context ← get_context cmd
      let data := context.map (fun c => s!"<ITEM>\n--name={c.name}\n--context_item_type={c.kind}\n{c.text}\n</ITEM>")
      pure <| "\n".intercalate data
    else pure ""


  let rag_prompt : String := s!" The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>."
  let rag_string : String ← if config.rag? > 0 then do
      let items ← retrieve cmd config
      let data := items.map (fun c => s!"<DOC>\n{c}\n</DOC>")
      pure <| "\n".intercalate data
    else pure ""

  let prompt : String := s!"{main_prompt}{if config.annotation? then annotation_prompt else ""}{if config.context? then context_prompt else ""}{if config.rag? != 0 then rag_prompt else ""} Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n{if config.context? then ("<CONTEXT>\n" ++ context_string ++ "\n</CONTEXT>\n\n") else ""}{if config.rag? != 0 then "<RETRIEVED>\n" ++ rag_string ++ "\n</RETRIEVED>\n\n" else ""}{if config.annotation? then "<ANNOTATION>\n" ++ annotation_string ++ "\n</ANNOTATION>\n\n" else ""}<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"
  return prompt


/- Returns the prompt function from a name -/
def get_prompt_batched (prompt_name : String)  (config : ImProverConfig) (cmds_ci : Array (CompilationStep ×ConstantInfo) ) : IO (Array  String) := do
  let main_prompt := match config.metric with
  | "length" => length_prompt
  | "declarativity" => declarativity_prompt
  | "dependency" => dependency_prompt
  | "completion" => completion_prompt
  | _ => length_prompt


  let annotation_prompt : String := s!" A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response."
  let context_prompt : String := s!" The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>."
  let rag_prompt : String := s!" The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>."

  let example_prompt : String := s!"Here are some examples of such optimization, as wrapped in <EXAMPLES>...</EXAMPLES>. Note that these examples are for illustrative purposes only and should not be copied directly. Instead, use them to understand the kind of optimization expected and apply similar techniques to the current theorem."
  let example_string : String ← if config.example_file.isSome then do
    IO.FS.readFile config.example_file.get!
    else
    pure ""

  let prompt_data ← cmds_ci.mapM (fun (cmd,_) => do

    let srcCommand := cmd.src.toString


    let annotation_string : String ← if config.annotation? then (insert_state_comments cmd) else pure ""

    let context_string : String ← if config.context? then do
        let context ← get_context cmd
        let data := context.map (fun c => s!"<ITEM>\n--name={c.name}\n--context_item_type={c.kind}\n{c.text}\n</ITEM>")
        pure <| "\n".intercalate data
      else pure ""

    return (cmd, srcCommand, annotation_string, context_string)
  )

  let rag_strings : Array String ← if config.rag? > 0 then do
      let items ← retrieve_batch cmds_ci config
      let data := items.map (fun (c,docs) => (c, "\n".intercalate <| docs.map (fun d => s!"<DOC>\n{d}\n</DOC>")))
      -- pure <| "\n".intercalate data
      pure <| data.map (fun (_,d) => d)
    else
      pure <| Array.range (cmds_ci.size) |>.map (fun _=> "")

  let prompt_data := prompt_data.zip rag_strings |>.map (fun ((_,srcCommand,annotation_string,context_string),rag_string) =>
    let prompt : String := s!"{main_prompt}{if config.annotation? then annotation_prompt else ""}{if config.context? then context_prompt else ""}{if config.rag? != 0 then rag_prompt else ""} Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n{if config.example_file.isSome then s!"{example_prompt}\n\n<EXAMPLES>\n{example_string}\n</EXAMPLES>\n\n" else ""}{if config.context? then ("<CONTEXT>\n" ++ context_string ++ "\n</CONTEXT>\n\n") else ""}{if config.rag? != 0 then "<RETRIEVED>\n" ++ rag_string ++ "\n</RETRIEVED>\n\n" else ""}{if config.annotation? then "<ANNOTATION>\n" ++ annotation_string ++ "\n</ANNOTATION>\n\n" else ""}<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"

    prompt
    )

  return prompt_data

def get_prompt_batched_anno (prompt_name : String)  (config : ImProverConfig) (cmds_ci : Array (CompilationStep ×ConstantInfo) ) : IO (Array  String) := do
  let main_prompt := match prompt_name with
  | "length" => length_prompt
  | "declarativity" => declarativity_prompt
  | "dependency" => dependency_prompt
  | "completion" => completion_prompt
  | _ => length_prompt


  let annotation_prompt : String := s!" A version of the current theorem with the goal states annotated has also been provided for reference. Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response."
  let context_prompt : String := s!" The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>."
  let rag_prompt : String := s!" The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>."


  let prompt_data ← cmds_ci.mapM (fun (cmd,_) => do

    let srcCommand := cmd.src.toString


    let annotation_string : String ← if config.annotation? then (insert_state_comments cmd) else pure ""

    let context_string : String ← if config.context? then do
        let context ← get_context cmd
        let data := context.map (fun c => s!"<ITEM>\n--name={c.name}\n--context_item_type={c.kind}\n{c.text}\n</ITEM>")
        pure <| "\n".intercalate data
      else pure ""

    return (cmd, srcCommand, annotation_string, context_string)
  )

  let rag_strings : Array String ← if config.rag? > 0 then do
      let items ← retrieve_batch cmds_ci config
      let data := items.map (fun (c,docs) => (c, "\n".intercalate <| docs.map (fun d => s!"<DOC>\n{d}\n</DOC>")))
      -- pure <| "\n".intercalate data
      pure <| data.map (fun (_,d) => d)
    else
      pure <| Array.range (cmds_ci.size) |>.map (fun _=> "")

  let prompt_data := prompt_data.zip rag_strings |>.map (fun ((_,srcCommand,annotation_string,context_string),rag_string) =>
    let prompt : String := s!"{main_prompt}{if config.annotation? then annotation_prompt else ""}{if config.context? then context_prompt else ""}{if config.rag? != 0 then rag_prompt else ""} Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n{if config.context? then ("<CONTEXT>\n" ++ context_string ++ "\n</CONTEXT>\n\n") else ""}{if config.rag? != 0 then "<RETRIEVED>\n" ++ rag_string ++ "\n</RETRIEVED>\n\n" else ""}{if config.annotation? then "<CURRENT>\n" ++ annotation_string ++ "\n</CURRENT>\n\n" else s!"<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"}"

    prompt
    )

  return prompt_data


def get_prompt_batched_ctx (prompt_name : String)  (config : ImProverConfig) (cmds_ci : Array (CompilationStep ×ConstantInfo) ) : IO (Array  String) := do
  let main_prompt := match prompt_name with
  | "length" => length_prompt
  | "declarativity" => declarativity_prompt
  | "dependency" => dependency_prompt
  | "completion" => completion_prompt
  | _ => length_prompt


  let annotation_prompt : String := s!" A version of the current theorem with the goal states annotated has also been provided for reference (wrapped in <ANNOTATED>...</ANNOTATED>). Namely, the goal states have been interleaved between tactics as comments to help you better understand the proof and ensure the correctness of your response. Do not include such state comments in your final response."
  -- let context_prompt : String := s!" The proof context, with relevant definitions and theorems, has additionally been provided to help you better understand the proof and ensure the correctness of your response. It is wrapped in <CONTEXT>...</CONTEXT>, with each item wrapped in <ITEM>...</ITEM>."
  let rag_prompt : String := s!" The following items have been retrieved from the knowledge base as they may be helpful in optimizing the proof. They are wrapped in <RETRIEVED>...</RETRIEVED> with each item being wrapped further in <DOC>...</DOC>."


  let prompt_data ← cmds_ci.mapM (fun (cmd,_) => do

    let srcCommand := cmd.src.toString


    let annotation_string : String ← if config.annotation? then (insert_state_comments cmd) else pure ""

    let context_string : String ← if config.context? then do
        let context ← get_context cmd
        let data := context.map (fun c => s!"<DOC>\n--name={c.name}\n{c.text}\n</DOC>")
        pure <| "\n".intercalate data
      else pure ""

    return (cmd, srcCommand, annotation_string, context_string)
  )

  let rag_strings : Array String ← if config.rag? > 0 then do
      let items ← retrieve_batch cmds_ci config
      let data := items.map (fun (c,docs) => (c, "\n".intercalate <| docs.map (fun d => s!"<DOC>\n{d}\n</DOC>")))
      -- pure <| "\n".intercalate data
      pure <| data.map (fun (_,d) => d)
    else
      pure <| Array.range (cmds_ci.size) |>.map (fun _=> "")

  let prompt_data := prompt_data.zip rag_strings |>.map (fun ((_,srcCommand,annotation_string,context_string),rag_string) =>
    let prompt : String := s!"{main_prompt}{if config.annotation? then annotation_prompt else ""}{if config.rag? != 0 || config.context? then rag_prompt else ""} Include the output in the <IMPROVED>...</IMPROVED> tag.\n\n{if config.rag? != 0 || config.context? then "<RETRIEVED>\n" ++ (if config.context? then context_string ++"\n" else "")++(if config.rag? != 0 then rag_string else "") ++ "\n</RETRIEVED>\n\n" else ""}{if config.annotation? then "<ANNOTATION>\n" ++ annotation_string ++ "\n</ANNOTATION>\n\n" else ""}<CURRENT>\n{srcCommand}\n</CURRENT>\n\n<IMPROVED>"

    prompt
    )

  return prompt_data
