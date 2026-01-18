import ProofWidgets.Component.GraphDisplay
import ProofWidgets.Component.HtmlDisplay
import ProofWidgets.Component.Basic
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser

/-!
# GoalTree Widget Visualization

A goal-centric view of proofs where:
- **Nodes** = Goal states (the proof obligations)
- **Edges** = Tactics (transitions that transform goals into subgoals)

This is dual to ProofTree where nodes are tactics.
-/

open Lean Elab Server Widget
open ProofWidgets
open scoped ProofWidgets.Jsx

/-!
## GoalTree Data Structure

We use a simple recursive structure. Each node has:
- The goal state
- Optionally, a tactic that was applied and the resulting children
-/

/-- A goal-centric proof tree where nodes are goal states -/
structure GoalTree where
  /-- The goal at this node -/
  goal : GoalInfo
  /-- The tactic applied to this goal (if any) -/
  tactic : Option String := none
  /-- Hypotheses used by the tactic -/
  dependencies : List String := []
  /-- Regular subgoals produced by the tactic -/
  children : Array GoalTree := #[]
  /-- Spawned subgoals (from combinators like `<;>`) -/
  spawnedChildren : Array GoalTree := #[]
  deriving Inhabited

namespace GoalTree

/-- Check if this is a leaf (goal proved, no children) -/
def isLeaf (t : GoalTree) : Bool :=
  t.children.isEmpty && t.spawnedChildren.isEmpty

/-- Get all children (regular + spawned) -/
def allChildren (t : GoalTree) : Array GoalTree :=
  t.children ++ t.spawnedChildren

end GoalTree

/-!
## Conversion from ProofTree to GoalTree
-/

/-- Convert a ProofTree to a GoalTree -/
partial def proofTreeToGoalTree (pt : ProofTree) : GoalTree :=
  let goal := pt.node.goalBefore
  let tactic := pt.node.tacticString
  let deps := pt.node.tacticDependsOn

  -- Determine which children are spawned (spawned_children ⊆ children)
  let regularChildren := pt.children.filter (fun c => !pt.spawned_children.any (· == c))
    |>.map proofTreeToGoalTree
  let spawnedChildren := pt.spawned_children.map proofTreeToGoalTree

  -- If no children, tactic proved the goal directly
  let tacticOpt := if pt.children.isEmpty && pt.spawned_children.isEmpty
                   then none
                   else some tactic

  { goal := goal
    tactic := tacticOpt
    dependencies := deps
    children := regularChildren
    spawnedChildren := spawnedChildren }

/-!
## Widget Helper Functions
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

/-- Format goal info for display -/
def fmtGoalInfo (goal : GoalInfo) : String :=
  let hypsStr := fmtHyps goal.hyps
  let goalStr := s!"|- {goal.type}"
  if hypsStr.isEmpty then goalStr
  else s!"Hypotheses:\n{hypsStr}\n\nGoal:\n  {goalStr}"

/-- Create HTML details for a goal node -/
def goalDetails (tree : GoalTree) : Html :=
  let goalStr := fmtGoalInfo tree.goal
  let tacticStr := match tree.tactic with
    | none => "Proved! (no subgoals)"
    | some t =>
      let childCount := tree.children.size + tree.spawnedChildren.size
      let spawnedNote := if tree.spawnedChildren.isEmpty then ""
                         else s!" ({tree.spawnedChildren.size} spawned)"
      s!"Tactic: {t}\nProduces {childCount} subgoal(s){spawnedNote}"

  let depsStr :=
    if tree.dependencies.isEmpty then ""
    else s!"\nUses: {tree.dependencies}"

  let fullText := goalStr ++ "\n---------------------\n" ++ tacticStr ++ depsStr

  Html.element "pre"
    #[("style", json% {fontSize: "12px", whiteSpace: "pre-wrap", fontFamily: "monospace"})]
    #[.text fullText]

/-- Create an SVG label for a goal node -/
def goalLabel (goal : GoalInfo) (isLeaf : Bool) : Html :=
  let displayText := truncateStr goal.type 35
  let textLen := displayText.length * 7 + 20
  let rectWidth : Int := max textLen 100
  let halfWidth : Int := rectWidth / 2
  -- Leaf nodes (proved goals) get a green color
  let fillColor := if isLeaf then "#e8f5e9" else "var(--vscode-editor-background)"
  let strokeColor := if isLeaf then "#4caf50" else "var(--vscode-editor-foreground)"

  Html.element "g" #[] #[
    Html.element "rect"
      #[("x", toJson (-halfWidth)), ("y", toJson (-18 : Int)),
        ("width", toJson rectWidth), ("height", toJson (36 : Nat)),
        ("rx", toJson (8 : Nat)), ("fill", toJson fillColor),
        ("stroke", toJson strokeColor), ("strokeWidth", toJson (2.0 : Float)),
        ("className", toJson "dim")]
      #[],
    Html.element "text"
      #[("textAnchor", toJson "middle"), ("dominantBaseline", toJson "middle"),
        ("fontSize", toJson "11"), ("fill", toJson "var(--vscode-editor-foreground)"),
        ("fontFamily", toJson "monospace")]
      #[.text displayText]
  ]

/-!
## Graph Construction
-/

/-- State for tree traversal -/
structure GraphState where
  vertices : Array GraphDisplay.Vertex
  edges : Array GraphDisplay.Edge
  nextId : Nat
  deriving Inhabited

/-- Convert GoalTree to graph vertices and edges -/
partial def goalTreeToGraph (tree : GoalTree) (parentId : Option String := none)
    (edgeLabel : Option String := none) (isSpawned : Bool := false)
    : StateM GraphState Unit := do
  let state ← get
  let nodeId := s!"goal_{state.nextId}"

  let textLen := tree.goal.type.length * 7 + 20
  let rectWidth : Float := Float.ofNat (max 100 textLen)
  let vertex : GraphDisplay.Vertex := {
    id := nodeId
    label := goalLabel tree.goal tree.isLeaf
    boundingShape := .rect rectWidth 36
    details? := some (goalDetails tree)
  }

  -- Create edge from parent if exists
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
      -- Create edge with optional tactic label
      let labelHtml := edgeLabel.map fun lbl =>
        let truncLbl := truncateStr lbl 12
        let lblWidth : Int := truncLbl.length * 6 + 10
        Html.element "g" #[] #[
          Html.element "rect"
            #[("x", toJson (-lblWidth / 2)), ("y", toJson (-8 : Int)),
              ("width", toJson lblWidth), ("height", toJson (16 : Nat)),
              ("rx", toJson (3 : Nat)),
              ("fill", toJson "var(--vscode-editor-background)"),
              ("stroke", toJson "var(--vscode-editorWidget-border)"),
              ("strokeWidth", toJson (1 : Nat))]
            #[],
          Html.element "text"
            #[("textAnchor", toJson "middle"), ("dominantBaseline", toJson "middle"),
              ("fontSize", toJson "9"), ("fill", toJson "var(--vscode-editor-foreground)")]
            #[.text truncLbl]
        ]
      let edge : GraphDisplay.Edge := {
        source := pid
        target := nodeId
        attrs := edgeAttrs
        label? := labelHtml
      }
      state.edges.push edge

  set { state with
    vertices := state.vertices.push vertex
    edges := newEdges
    nextId := state.nextId + 1
  }

  -- Process children with tactic as edge label
  let tacticLabel := tree.tactic
  for child in tree.children do
    goalTreeToGraph child (some nodeId) tacticLabel false
  for child in tree.spawnedChildren do
    goalTreeToGraph child (some nodeId) tacticLabel true

/-- Convert GoalTree to GraphDisplay.Props -/
def goalTreeToProps (tree : GoalTree) : GraphDisplay.Props :=
  let initState : GraphState := { vertices := #[], edges := #[], nextId := 0 }
  let (_, finalState) := goalTreeToGraph tree |>.run initState
  { vertices := finalState.vertices
    edges := finalState.edges
    showDetails := true
    forces := #[
      .link { distance? := some 120, strength? := some 0.6 },
      .collide { radius? := some 70 },
      .manyBody { strength? := some (-300) },
      .y { strength? := some 0.15 }
    ] }

/-- Generate HTML for displaying a GoalTree -/
def goalTreeHtml (tree : GoalTree) : Html :=
  let props := goalTreeToProps tree
  Html.ofComponent GraphDisplay props #[]

/-- Convert ProofTree and display as GoalTree -/
def proofTreeAsGoalHtml (pt : ProofTree) : Html :=
  goalTreeHtml (proofTreeToGoalTree pt)

/-!
## IO variants for use with #html
-/

def goalTreeHtmlIO (tree : IO GoalTree) : IO Html := do
  pure <| goalTreeHtml (← tree)

def proofTreeAsGoalHtmlIO (pt : IO ProofTree) : IO Html := do
  pure <| proofTreeAsGoalHtml (← pt)
