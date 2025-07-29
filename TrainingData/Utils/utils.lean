import TrainingData.Utils.state_comments


open Lean Core Elab IO Meta Term Command Tactic System

set_option autoImplicit true


/- Helper function to return plaintext of a theorem annotated with goal states -/
def annotateTheorems (targetModule : Name) (decls : Option (List Name)) (proofAsSorry? : Bool) : IO (List String) := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean targetModule).toString

  /- Handle incomplete proofs with "sorry" in them -/
  let proofAsSorry := ({} : KVMap).insert `debug.byAsSorry (.ofBool true)
    |>.insert `linter.unusedVariables (.ofBool false)
    |>.insert `linter.unusedTactic (.ofBool false)
    |>.insert `linter.unreachableTactic (.ofBool false)

  /- Process the actual source code from our module -/
  let steps := Lean.Elab.IO.processInput' (← moduleSource targetModule) none (if proofAsSorry? then proofAsSorry else {}) fileName
  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  let mut annotatedTheorems := []

  for (cmd, ci) in targets do
    let ci_name_stem := ci.name.toString.splitOn "." |>.getLast! |>.toName
    if (decls.isSome && !(decls.get!.contains ci_name_stem)) then
      continue
    let state_comments ← insert_state_comments cmd
    annotatedTheorems := annotatedTheorems ++ [state_comments]
    IO.println s!"{state_comments}"

  return annotatedTheorems




/- Not sure what this is for but the file doesn't run without it :| -/
def _root_.Lean.Elab.Command.State.withOptions (state : Command.State) (options : Options) :=
  { state with
    scopes := state.scopes.map fun s : Scope =>
      { s with opts := Id.run do
          let mut opts := s.opts
          for (k, v) in options do
            opts := opts.insert k v
          opts } }


def String.splitAtString (s : String) (pattern : String): Option (String × String) :=
  if h : pattern.endPos.1 = 0 then none
  else
    have hPatt := Nat.zero_lt_of_ne_zero h
    let rec loop (pos : String.Pos) :=
      if h : pos.byteIdx + pattern.endPos.byteIdx > s.endPos.byteIdx then
        none
      else
        have := Nat.lt_of_lt_of_le (Nat.add_lt_add_left hPatt _) (Nat.ge_of_not_lt h)
        if s.substrEq pos pattern 0 pattern.endPos.byteIdx then
          -- Found a match, return split strings
          let before := s.extract 0 pos
          let after := s.extract (pos + pattern) s.endPos
          some (before, after)
        else
          have := Nat.sub_lt_sub_left this (lt_next s pos)
          loop (s.next pos)
      termination_by s.endPos.1 - pos.1
    loop 0




def getInitialProofState2 (cmd : CompilationStep) : IO String := do
  let env := cmd.after
  let ci? := cmd.diff.get? 0

  if ci?.isSome then
    try
      let ci := ci?.get!
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
    catch _ =>
      let backup:=  match InfoTree.tactics_new cmd.trees |>.get? 0 with
      | some t => t.mainGoalStateBefore
      | _ => pure default
      return (← backup).pretty (width := 100000000)

  else
    let backup:=  match InfoTree.tactics_new cmd.trees |>.get? 0 with
    | some t => t.mainGoalStateBefore
    | _ => pure default
    return (← backup).pretty (width := 100000000)
