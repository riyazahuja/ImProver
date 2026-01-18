import ProofWidgets.Component.GraphDisplay
import ProofWidgets.Component.HtmlDisplay
import ProofWidgets.Component.Basic
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser

/-!
# ProofTree Widget Demo

This file demonstrates how to use the ProofTree visualization widget.
Open this file in VS Code with the Lean extension to see the interactive graph.
-/

open Lean Elab Server Widget
open ProofWidgets
open scoped ProofWidgets.Jsx

/-!
## Widget Helper Functions (inline for demo)
-/

/-- Truncate a string to a maximum length -/
def truncateStr (s : String) (maxLen : Nat := 25) : String :=
  if s.length > maxLen then s.take (maxLen - 3) ++ "..." else s

/-- Format hypothesis list as a string -/
def fmtHyps (hyps : List Hypothesis) : String :=
  hyps.filterMap (fun h =>
    if h.isProof == "universe" then none
    else some s!"  {h.username} : {truncateStr h.type 50}"
  ) |> String.intercalate "\n"

/-- Format goal info as a string -/
def fmtGoal (goal : GoalInfo) : String :=
  let hypsStr := fmtHyps goal.hyps
  let goalStr := s!"  |- {goal.type}"
  if hypsStr.isEmpty then goalStr else hypsStr ++ "\n" ++ goalStr

/-- Format a list of goals as a string -/
def fmtGoalList (goals : List GoalInfo) : String :=
  goals.zipIdx.map (fun (g, i) => s!"[{i+1}] |- {truncateStr g.type 60}")
  |> String.intercalate "\n"

/-- Create HTML details for a ProofStep node -/
def stepDetails (step : ProofStep) : Html :=
  let tacticStr := s!"Tactic: {step.tacticString}"
  let goalBeforeStr := s!"Goal Before:\n{fmtGoal step.goalBefore}"
  let goalsAfterStr :=
    if step.goalsAfter.isEmpty && step.spawnedGoals.isEmpty then
      "Goals After: (none - proved!)"
    else
      let regular := if step.goalsAfter.isEmpty then ""
                     else s!"Goals After:\n{fmtGoalList step.goalsAfter}"
      let spawned := if step.spawnedGoals.isEmpty then ""
                     else s!"Spawned Goals:\n{fmtGoalList step.spawnedGoals}"
      if regular.isEmpty then spawned
      else if spawned.isEmpty then regular
      else regular ++ "\n" ++ spawned
  let depsStr :=
    if step.tacticDependsOn.isEmpty then ""
    else s!"Dependencies: {step.tacticDependsOn}"

  let fullText := String.intercalate "\n---------------------\n"
    ([tacticStr, goalBeforeStr, goalsAfterStr] ++
     if depsStr.isEmpty then [] else [depsStr])

  Html.element "pre"
    #[("style", json% {fontSize: "12px", whiteSpace: "pre-wrap", fontFamily: "monospace"})]
    #[.text fullText]

/-- Create an SVG label for a vertex -/
def mkLabel (tactic : String) (isSpawned : Bool := false) : Html :=
  let displayText := truncateStr tactic 30
  let textLen := displayText.length * 8 + 16
  let rectWidth : Int := max textLen 80
  let halfWidth : Int := rectWidth / 2
  let fillColor := if isSpawned then "#fff3e0" else "var(--vscode-editor-background)"
  let strokeColor := if isSpawned then "#ff9800" else "var(--vscode-editor-foreground)"

  Html.element "g" #[] #[
    Html.element "rect"
      #[("x", toJson (-halfWidth)), ("y", toJson (-15 : Int)),
        ("width", toJson rectWidth), ("height", toJson (30 : Nat)),
        ("rx", toJson (5 : Nat)), ("fill", toJson fillColor),
        ("stroke", toJson strokeColor), ("strokeWidth", toJson (1.5 : Float)),
        ("className", toJson "dim")]
      #[],
    Html.element "text"
      #[("textAnchor", toJson "middle"), ("dominantBaseline", toJson "middle"),
        ("fontSize", toJson "11"), ("fill", toJson "var(--vscode-editor-foreground)")]
      #[.text displayText]
  ]

/-- State for tree traversal -/
structure GraphState where
  vertices : Array GraphDisplay.Vertex
  edges : Array GraphDisplay.Edge
  nextId : Nat
  deriving Inhabited

/-- Convert ProofTree to graph vertices and edges -/
partial def treeToGraph (tree : ProofTree) (parentId : Option String := none)
    (isSpawned : Bool := false) : StateM GraphState Unit := do
  let state ← get
  let nodeId := s!"node_{state.nextId}"

  let textLen := tree.node.tacticString.length * 8 + 16
  let rectWidth : Float := Float.ofNat (max 80 textLen)
  let vertex : GraphDisplay.Vertex := {
    id := nodeId
    label := mkLabel tree.node.tacticString isSpawned
    boundingShape := .rect rectWidth 30
    details? := some (stepDetails tree.node)
  }

  let newEdges := match parentId with
    | none => state.edges
    | some pid =>
      let edgeAttrs : Array (String × Json) :=
        if isSpawned then
          #[("stroke", "#ff9800"), ("strokeDasharray", "5,5"),
            ("strokeWidth", (2 : Nat)), ("markerEnd", "url(#arrow)")]
        else
          #[("stroke", "var(--vscode-editor-foreground)"),
            ("strokeWidth", (2 : Nat)), ("markerEnd", "url(#arrow)")]
      state.edges.push { source := pid, target := nodeId, attrs := edgeAttrs }

  set { state with
    vertices := state.vertices.push vertex
    edges := newEdges
    nextId := state.nextId + 1
  }

  -- spawned_children is a subset of children, so only iterate children once
  -- and check if each child is in spawned_children to determine styling
  for child in tree.children do
    let childIsSpawned := tree.spawned_children.any (· == child)
    treeToGraph child (some nodeId) childIsSpawned

/-- Convert ProofTree to GraphDisplay.Props -/
def treeToProps (tree : ProofTree) : GraphDisplay.Props :=
  let initState : GraphState := { vertices := #[], edges := #[], nextId := 0 }
  let (_, finalState) := treeToGraph tree |>.run initState
  { vertices := finalState.vertices
    edges := finalState.edges
    showDetails := true
    forces := #[
      .link { distance? := some 100, strength? := some 0.8 },
      .collide { radius? := some 60 },
      .manyBody { strength? := some (-200) },
      .y { strength? := some 0.1 }
    ] }

/-- Generate HTML for displaying a ProofTree -/
def treeHtml (tree : IO ProofTree) : IO Html := do
  let props := treeToProps (← tree)
  pure <| Html.ofComponent GraphDisplay props #[]




open Lean Core Elab IO Meta Term Command Tactic System


def getCompilationSteps (mod : Name) (decl : Name) (new_proof : String) : IO (Option CompilationStep × Option CompilationStep) := do
  searchPathRef.set compile_time_search_path%

  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl

  match target with
  | none => return (none, none)
  | some (target_cmd, _) => do

    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})

    let new_target ← elaborated_steps.head?

    match new_target with
    | none =>
      return (target_cmd, none)
    | some new_target => do
      let correct := not <| new_target.msgs.any (fun m => m.severity == .error)
      return (some target_cmd, if correct then some new_target else none)



def extractProofTree (mod : Name) (decl : Name) (new_proof : String) : IO (Option ProofTree × Option ProofTree) := do
  let (old,new) ← getCompilationSteps mod decl new_proof
  let old_tree ← match old with
    | none => pure none
    | some old_cs => do
      let steps := (← old_cs.trees.filterMapM BetterParser).flatMap (fun x => x.steps)
      pure (getProofTree steps)
  let new_tree ← match new with
    | none => pure none
    | some new_cs => do
      let steps := (← new_cs.trees.filterMapM BetterParser).flatMap (fun x => x.steps)
      pure (getProofTree steps)
  return (old_tree, new_tree)



def new_proof := "lemma KD5_weakerThan_KD45 : (Hilbert.KD5 α) ≤ₛ (Hilbert.KD45 α) := by
  -- Introduce a lemma to handle the subset relationship between the axioms of KD5 and KD45
  have h₁ : (LO.Modal.Hilbert.KD5 α).axioms ⊆ (LO.Modal.Hilbert.KD45 α).axioms → (Hilbert.KD5 α) ≤ₛ (Hilbert.KD45 α) := by
    intro h
    -- Apply the lemma that establishes the weakening relation given a subset of axioms
    apply normal_weakerThan_of_subset
    -- Use the given subset relation to conclude the proof
    <;> assumption
  -- Prove the subset relation between the axioms of KD5 and KD45
  have h₂ : (LO.Modal.Hilbert.KD5 α).axioms ⊆ (LO.Modal.Hilbert.KD45 α).axioms := by
    -- Introduce the axioms of KD5 and verify they are included in the axioms of KD45
    intro φ hφ
    cases' hφ with hφ hφ
    <;> simp_all [LO.Modal.Hilbert.KD5, LO.Modal.Hilbert.KD45]
    <;> aesop
  -- Combine the results to conclude the weakening relation
  exact h₁ h₂"

-- #eval do extractProofTree `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof

def demoTree : IO ProofTree := do
  let (_, new_tree) ← extractProofTree `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof
  match new_tree with
  | some tree => return tree
  | none => throw <| IO.userError "Failed to extract proof tree"



#html do treeHtml demoTree
