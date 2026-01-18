import ProofWidgets.Component.GraphDisplay
import ProofWidgets.Component.HtmlDisplay
import ProofWidgets.Component.Basic
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser

/-!
# GoalTree Demo

A goal-centric view of proofs where:
- **Nodes** = Goal states (proof obligations)
- **Edges** = Tactics (labeled transitions between goals)
- **Green nodes** = Proved goals (leaves)
- **Dashed orange edges** = Spawned transitions (from `<;>` etc.)
-/

open Lean Core Elab IO Meta Term Command Tactic System Server Widget
open ProofWidgets
open scoped ProofWidgets.Jsx

/-!
## GoalTree Data Structure (inline)
-/

/-- A goal-centric proof tree where nodes are goal states -/
structure GoalTree where
  goal : GoalInfo
  tactic : Option String := none
  dependencies : List String := []
  children : Array GoalTree := #[]
  spawnedChildren : Array GoalTree := #[]
  deriving Inhabited

instance : ToString GoalTree where
  toString t :=
    let tacStr := match t.tactic with | none => "Proved!" | some s => s
    let depStr := if t.dependencies.isEmpty then "" else s!" Uses: {t.dependencies}"
    let childStr := if t.children.isEmpty && t.spawnedChildren.isEmpty then ""
      else s!" Children: {t.children.size} (spawned: {t.spawnedChildren.size})"
    s!"Goal: {t.goal.type}\nTactic: {tacStr}{depStr}{childStr}\n"

def GoalTree.isLeaf (t : GoalTree) : Bool :=
  t.children.isEmpty && t.spawnedChildren.isEmpty

/-- Convert ProofTree to GoalTree -/
partial def proofTreeToGoalTree (pt : ProofTree) : GoalTree :=
  let goal := pt.node.goalBefore
  let tactic := pt.node.tacticString
  let deps := pt.node.tacticDependsOn
  -- spawned_children ⊆ children, so filter properly
  let regularChildren := pt.children.filter (fun c => !pt.spawned_children.any (· == c))
    |>.map proofTreeToGoalTree
  let spawnedChildren := pt.spawned_children.map proofTreeToGoalTree
  let tacticOpt := if pt.children.isEmpty && pt.spawned_children.isEmpty
                   then none else some tactic
  { goal, tactic := tacticOpt, dependencies := deps,
    children := regularChildren, spawnedChildren }

/-!
## Widget helpers
-/

def truncStr (s : String) (n : Nat := 25) : String :=
  if s.length > n then s.take (n - 3) ++ "..." else s

def fmtHyps (hyps : List _root_.Hypothesis) : String :=
  hyps.filterMap (fun h =>
    if h.isProof == "universe" then none
    else some s!"  {h.username} : {truncStr h.type 50}"
  ) |> String.intercalate "\n"

def fmtGoalInfo (goal : GoalInfo) : String :=
  let hypsStr := fmtHyps goal.hyps
  let goalStr := s!"|- {goal.type}"
  if hypsStr.isEmpty then goalStr
  else s!"Hypotheses:\n{hypsStr}\n\nGoal:\n  {goalStr}"

def goalDetails (tree : GoalTree) : Html :=
  let goalStr := fmtGoalInfo tree.goal
  let tacticStr := match tree.tactic with
    | none => "Proved! (no subgoals)"
    | some t =>
      let n := tree.children.size + tree.spawnedChildren.size
      let sp := if tree.spawnedChildren.isEmpty then "" else s!" ({tree.spawnedChildren.size} spawned)"
      s!"Tactic: {t}\nProduces {n} subgoal(s){sp}"
  let depsStr := if tree.dependencies.isEmpty then "" else s!"\nUses: {tree.dependencies}"
  Html.element "pre"
    #[("style", json% {fontSize: "12px", whiteSpace: "pre-wrap", fontFamily: "monospace"})]
    #[.text (goalStr ++ "\n---------------------\n" ++ tacticStr ++ depsStr)]

def goalLabel (goal : GoalInfo) (isLeaf : Bool) : Html :=
  let txt := truncStr goal.type 35
  let w : Int := max (txt.length * 7 + 20) 100
  let fill := if isLeaf then "#e8f5e9" else "var(--vscode-editor-background)"
  let stroke := if isLeaf then "#4caf50" else "var(--vscode-editor-foreground)"
  Html.element "g" #[] #[
    Html.element "rect"
      #[("x", toJson (-w/2)), ("y", toJson (-18:Int)), ("width", toJson w),
        ("height", toJson (36:Nat)), ("rx", toJson (8:Nat)), ("fill", toJson fill),
        ("stroke", toJson stroke), ("strokeWidth", toJson (2.0:Float)), ("className", toJson "dim")]
      #[],
    Html.element "text"
      #[("textAnchor", toJson "middle"), ("dominantBaseline", toJson "middle"),
        ("fontSize", toJson "11"), ("fill", toJson "var(--vscode-editor-foreground)"),
        ("fontFamily", toJson "monospace")]
      #[.text txt]
  ]

/-!
## Graph construction
-/

structure GState where
  vertices : Array GraphDisplay.Vertex
  edges : Array GraphDisplay.Edge
  nextId : Nat
  deriving Inhabited

partial def goalTreeToGraph (tree : GoalTree) (parentId : Option String := none)
    (edgeLbl : Option String := none) (isSpawned : Bool := false) : StateM GState Unit := do
  let st ← get
  let nid := s!"goal_{st.nextId}"
  let w : Float := Float.ofNat (max 100 (tree.goal.type.length * 7 + 20))
  let v : GraphDisplay.Vertex := {
    id := nid, label := goalLabel tree.goal tree.isLeaf,
    boundingShape := .rect w 36, details? := some (goalDetails tree)
  }
  let newEdges := match parentId with
    | none => st.edges
    | some pid =>
      let attrs : Array (String × Json) :=
        if isSpawned then #[("stroke", "#ff9800"), ("strokeDasharray", "5,5"), ("strokeWidth", (2 : Nat)), ("markerEnd", "url(#arrow)")]
        else #[("stroke", "var(--vscode-editor-foreground)"), ("strokeWidth", (2 : Nat)), ("markerEnd", "url(#arrow)")]
      let lbl := edgeLbl.map fun l =>
        let t := truncStr l 12; let lw : Int := t.length * 6 + 10
        Html.element "g" #[] #[
          Html.element "rect" #[("x", toJson (-lw/2)), ("y", toJson (-8 : Int)), ("width", toJson lw),
            ("height", toJson (16 : Nat)), ("rx", toJson (3 : Nat)), ("fill", toJson "var(--vscode-editor-background)"),
            ("stroke", toJson "var(--vscode-editorWidget-border)"), ("strokeWidth", toJson (1 : Nat))] #[],
          Html.element "text" #[("textAnchor", toJson "middle"), ("dominantBaseline", toJson "middle"),
            ("fontSize", toJson "9"), ("fill", toJson "var(--vscode-editor-foreground)")] #[.text t]
        ]
      st.edges.push { source := pid, target := nid, attrs, label? := lbl }
  set { st with vertices := st.vertices.push v, edges := newEdges, nextId := st.nextId + 1 }
  let tLbl := tree.tactic
  for c in tree.children do goalTreeToGraph c (some nid) tLbl false
  for c in tree.spawnedChildren do goalTreeToGraph c (some nid) tLbl true

def goalTreeToProps (tree : GoalTree) : GraphDisplay.Props :=
  let (_, st) := goalTreeToGraph tree |>.run { vertices := #[], edges := #[], nextId := 0 }
  { vertices := st.vertices, edges := st.edges, showDetails := true,
    forces := #[.link {distance? := some 120, strength? := some 0.6},
                .collide {radius? := some 70}, .manyBody {strength? := some (-300)},
                .y {strength? := some 0.15}] }

def goalTreeHtml (tree : GoalTree) : Html :=
  Html.ofComponent GraphDisplay (goalTreeToProps tree) #[]

def proofTreeAsGoalHtml (pt : ProofTree) : Html :=
  goalTreeHtml (proofTreeToGoalTree pt)

/-!
## Proof extraction
-/

def getCompilationSteps (mod : Name) (decl : Name) (new_proof : String) : IO (Option CompilationStep × Option CompilationStep) := do
  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl
  match target with
  | none => return (none, none)
  | some (target_cmd, _) => do
    let bg := (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |> toString
    let elabSteps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (bg ++ new_proof) fileName)
      target_cmd.parserStateBefore (target_cmd.commandStateBefore.withOptions {})
    let new_target ← elabSteps.head?
    match new_target with
    | none => return (target_cmd, none)
    | some nt => return (some target_cmd, if nt.msgs.any (·.severity == .error) then none else some nt)

def extractProofTree (mod : Name) (decl : Name) (new_proof : String) : IO (Option ProofTree × Option ProofTree) := do
  let (old, new) ← getCompilationSteps mod decl new_proof
  let oldT ← match old with
    | none => pure none
    | some cs => pure (getProofTree ((← cs.trees.filterMapM BetterParser).flatMap (·.steps)))
  let newT ← match new with
    | none => pure none
    | some cs => pure (getProofTree ((← cs.trees.filterMapM BetterParser).flatMap (·.steps)))
  return (oldT, newT)

/-!
## Demo
-/

def new_proof := "lemma KD5_weakerThan_KD45 : (Hilbert.KD5 α) ≤ₛ (Hilbert.KD45 α) := by
  have h₁ : (LO.Modal.Hilbert.KD5 α).axioms ⊆ (LO.Modal.Hilbert.KD45 α).axioms → (Hilbert.KD5 α) ≤ₛ (Hilbert.KD45 α) := by
    intro h
    apply normal_weakerThan_of_subset
    <;> assumption
  have h₂ : (LO.Modal.Hilbert.KD5 α).axioms ⊆ (LO.Modal.Hilbert.KD45 α).axioms := by
    intro φ hφ
    cases' hφ with hφ hφ
    <;> simp_all [LO.Modal.Hilbert.KD5, LO.Modal.Hilbert.KD45]
    <;> aesop
  exact h₁ h₂"

def demoProofTree : IO ProofTree := do
  let (_, t) ← extractProofTree `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof
  match t with | some tree => return tree | none => throw <| IO.userError "Failed"

/-!
## GoalTree Visualization

Place cursor here to see goals as nodes, tactics as edge labels:
- **White nodes** = open goals
- **Green nodes** = proved goals (leaves)
- **Solid edges** = regular subgoals
- **Dashed orange edges** = spawned subgoals
-/

#html do pure <| proofTreeAsGoalHtml (← demoProofTree)

#eval do pure <| proofTreeToGoalTree (← demoProofTree)
