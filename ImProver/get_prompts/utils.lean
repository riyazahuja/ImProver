import ImProver.get_prompts.where_with_end
import TrainingData.Utils.c2

open Lean Core Elab IO Meta Term Command Tactic

set_option autoImplicit true



def getNormalId (target : CompilationStep × ConstantInfo) (mod : Name): IO TheoremID := do
  let (cmd, ci) := target
  -- name : Name
  -- module : Name
  -- content : Option String := none
  -- compilationAlias : Option String := none
  -- isExtracted : Bool := false
  -- errorMsgs : Array String := #[]
  -- kind : String := "theorem"
  let msgs ← cmd.msgs.filterMapM (fun msg => do
        let m ← msg.data.toString
        if msg.severity != .error then
          return none
        return some m)
  let kind := getKind cmd.after.constants ci.name
  return {
    name := ci.name,
    module := mod,
    content := some cmd.src.toString,
    compilationAlias := some cmd.src.toString,
    isExtracted := false,
    errorMsgs := msgs.toArray,
    kind := kind
  }

open Lean.Elab.Command



def getScopes (cmd : CompilationStep) (fileName : String) : IO (String × String) := do

  let ctx : Command.Context := {
    fileName := fileName,
    fileMap := cmd.src.toString.toFileMap,
    tacticCache? := none,
    snap? := none,
    cancelTk? := none
  }
  let state := cmd.commandStateBefore

  let ((prescopes_raw, postscopes_raw), _) ← CommandElabM.toIO whereWithEndCore ctx state
  let prescopes ← prescopes_raw.toString
  let postscopes ← postscopes_raw.toString
  IO.println s!"==== Got scopes ===="
  IO.println s!"Prescopes: {prescopes}"
  IO.println s!"Postscopes: {postscopes}"
  return (prescopes, postscopes)


-- def getIDsRAG (targets_new : Array (CompilationStep × ConstantInfo))
--     (mod : Name) (python_cmd : String) (rag_id : String)
--     (informal_data : NameMap (String × String))
--     : IO (Array (CompilationStep × ConstantInfo × TheoremID)) := do



def getRagItems (targets_new : Array (CompilationStep × ConstantInfo))
    (mod : Name) (rag_id : String) (python_cmd : String) (k : Nat)
    : IO (Option (Array (String × String × (Array RagItem)))) := do

    let data : Json := Json.mkObj
        [("queries", Json.arr <| targets_new.map (fun (_, ci) =>
            Json.mkObj
            [("module", Json.str mod.toString),
                ("name", Json.str ci.name.toString)]
        )),
            ("k", Json.num k)
        ]
    IO.println data.pretty
    let out ← IO.Process.output {
    cmd := python_cmd,
    -- cmd := "/Users/ahuja/Desktop/ImProver_new/.venv/bin/python3",
    args := #["ImProver/get_prompts/rag.py",rag_id,
        data.compress
    ]
    }

    let stdout := out.stdout.trim

    let after_output := stdout.splitAtString "<OUTPUT>" |>.getD ("","") |>.2
    let output_raw := after_output.splitAtString "</OUTPUT>" |>.getD ("","") |>.1

    -- IO.println s!"STDOUT: {output_raw}\n\n"

    let json? := Json.parse output_raw.trim |>.toOption

    -- IO.println s!"JSON: {json?}"


    if json?.isNone then
        return none
    let json := json?.get!

    let outputs? := fromJson? json |>.toOption

    if outputs?.isNone then
        return none
    let outputs : Array RagOutput := outputs?.get!

    let outputs_trimmed := outputs.map (fun o => (o.informal_statement, o.informal_proof, o.results.toArray))
    return some outputs_trimmed




def proofAsSorry (cmd : CompilationStep) : Option String := do
  let tactics := InfoTree.tactics_new cmd.trees |>.map (fun t => (t.pp, FileMap.ofPosition t.ctx.fileMap t.range.1))
    if tactics.isEmpty then
      let splitAt? := cmd.src.toString.splitAtString ":="
      match splitAt? with
      | none => none
      | some (before, _) =>
        let new_thm := before ++ ":= by sorry"
        some new_thm
    else
      let (_, range) := tactics[0]!

      let cmd_rng := cmd.stx.getPos?

      let sstr : Substring := ⟨cmd.src.str, cmd_rng.getD 0,  range⟩

      let new_thm := sstr.toString ++ "sorry"
      some new_thm


def getPromptsAux (targets_new : Array (CompilationStep × ConstantInfo))
(mod : Name) (python_cmd : String) (fileName: String) (rag_id : Option String) (k : Nat)
: IO (List (List TheoremData × Nat)) := do

--   let informal_data : NameMap (String × String) := ( ← getInformalData rag_id mod ) |>.getD default

--   let targets_new_with_rag : Array (CompilationStep × ConstantInfo × TheoremID) := getIDsRAG targets_new mod python_cmd rag_id informal_data


  let io_rag_items : IO (Array (String × String × (Array RagItem))) := do

    if rag_id.isNone then
      return targets_new.map (fun _ => ("", "", #[]))

    let rag_items? ← getRagItems targets_new mod rag_id.get! python_cmd k
    if rag_items?.isNone then
      return targets_new.map (fun _ => ("", "", #[]))
    let rag_items := rag_items?.get!
    return rag_items

  let rag_items ← io_rag_items



--   let rag_strings : Array Json ← do
--     let items ← if targets_new.isEmpty then pure #[] else retrieve_batch_indep targets_new mod rag_id (python_cmd := python_cmd)
--     let x := items.map (fun (_, (b : List String)) => Json.arr <| b.map (fun x=> Json.str x) |>.toArray)
--     pure x


--   IO.println s!"==== Got {rag_strings.size} prompts from RAG ===="

--   let targets_new_with_id : Array (CompilationStep × ConstantInfo × (Array (CompilationStep × ConstantInfo))) :=
--     targets_new.mapIdx (fun i target => (target.1, target.2, targets_new.extract 0 i))
  let targets_new_with_id : Array (CompilationStep × ConstantInfo × TheoremID) ←
    targets_new.mapM (fun target => do
      let id ← getNormalId target mod
      return (target.1, target.2, id))

  let targets_new_cumulative : Array (CompilationStep × ConstantInfo × TheoremID × (Array TheoremID)) :=
    targets_new_with_id.mapIdx (fun i (cmd, ci, id) =>
      let deps := targets_new_with_id.extract 0 i |>.map (fun (_, _, dep_id) => dep_id)
      (cmd, ci, id, deps))


--   let targets_new_cumulative_with_informal : Array (CompilationStep × ConstantInfo × TheoremID × (Array TheoremID) × (Option String) × (Option String)) :=
--     targets_new_cumulative.map (fun i (cmd, ci, id, deps) =>

  let full_data := targets_new_cumulative.zip rag_items

  let mut outputs := []
  for (((cmd, ci, id, prev_ids), (informal_statement, informal_proof, rag_items)), target_idx) in (full_data).zipIdx do
    IO.println s!"Processing {ci.name.toString} in {mod.toString}"
    -- eventually want annotation on partial proofs, but for now, ignore
    let annotation_string : String ← insert_state_comments cmd


    -- let context_string : Json ← do
    --     let context ← get_context cmd
    --     pure <| Json.arr <| context.map (fun c : ExternalContext => Json.mkObj [
    --         ("name", Json.str c.name.toString),
    --         ("context_item_type", Json.str c.kind),
    --         ("content", Json.str c.text)
    --       ]) |>.toArray
    let (prescopes, postscopes) ← getScopes cmd fileName

    let pfAsSorry := proofAsSorry cmd |>.getD ""

    let initialGoal ←  getInitialProofState2 cmd


    let C1_raw ← get_context cmd --["theorem", "theorem (internal)"]
    let C1_dependencies : List TheoremID:= C1_raw.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text, kind:= ctx.kind, compilationAlias := some ctx.text})

    let C2_raw ← splitC2 fileName cmd "spawned"

    let mut extracted_thms : List TheoremData := []
    let mut C2_dependencies : List TheoremID := []
    -- errors = none means didn't compile, some [] means no errors, some [errors] means there were errors
    for ((pp_thm, stx_thm, deps, errors), idx) in C2_raw.zipIdx do

      let split_thm : TheoremID :=
        {name := s!"extracted_split_{ci.name}_{idx}".toName,
          module := mod,
          content := pp_thm,
          compilationAlias := stx_thm,
          isExtracted := true,
          errorMsgs := errors.toArray
        }

      C2_dependencies := split_thm :: C2_dependencies

      -- we don't get all the fancy data for splits bc we are lazy...
      let split_data : TheoremData :=
        { id := split_thm,
          prescopes := prescopes,
          postscopes := postscopes,
          C0_dependencies := prev_ids, --idk yet whether to keep this
          C1_dependencies := deps.map (fun ctx => {name := ctx.name, module := ctx.module, content := some ctx.text}) |>.toArray,
          C2_dependencies := #[]
        }
      extracted_thms := split_data :: extracted_thms

    -- let id : TheoremID := {name := ci.name, module := mod, content := some srcCommand, compilationAlias := some srcCommand}
    -- let id ← getNormalId (cmd, ci) mod

    let mainData : TheoremData :=
      { id := id,
        C0_dependencies := prev_ids,
        C1_dependencies := C1_dependencies.toArray,
        C2_dependencies := C2_dependencies.toArray,
        annotation := annotation_string,
        content_sorry := pfAsSorry,
        goal := initialGoal,
        rag := rag_items,
        prescopes := prescopes,
        postscopes := postscopes,
        informal_statement := informal_statement,
        informal_proof := informal_proof
        }

    outputs := ((mainData :: extracted_thms),target_idx) :: outputs

  return outputs
