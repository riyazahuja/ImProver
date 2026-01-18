import ProofWidgets.Component.GraphDisplay
import ProofWidgets.Component.HtmlDisplay
import ProofWidgets.Component.Basic
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser

open Lean Elab Server Widget
open ProofWidgets
open scoped ProofWidgets.Jsx

/-!
# ProofTree Widget Visualization

This module provides a widget to visualize `ProofTree` structures using
ProofWidgets4's `GraphDisplay` component.
-/

/-- Truncate a string to a maximum length, adding "..." if truncated -/
def truncateString (s : String) (maxLen : Nat := 25) : String :=
  if s.length > maxLen then
    s.take (maxLen - 3) ++ "..."
  else s

/-- Format hypothesis list as a string -/
def formatHypotheses (hyps : List Hypothesis) : String :=
  hyps.filterMap (fun h =>
    if h.isProof == "universe" then none
    else some s!"  {h.username} : {truncateString h.type 50}"
  ) |> String.intercalate "\n"

/-- Format goal info as a string -/
def formatGoalInfo (goal : GoalInfo) : String :=
  let hypsStr := formatHypotheses goal.hyps
  let goalStr := s!"  |- {goal.type}"
  if hypsStr.isEmpty then goalStr
  else hypsStr ++ "\n" ++ goalStr

/-- Format a list of goals as a string -/
def formatGoalList (goals : List GoalInfo) : String :=
  goals.zipIdx.map (fun (g, i) => s!"[{i+1}] |- {truncateString g.type 60}")
  |> String.intercalate "\n"

/-- Create HTML details for a ProofStep node -/
def proofStepDetailsHtml (step : ProofStep) : Html :=
  let tacticStr := s!"Tactic: {step.tacticString}"
  let goalBeforeStr := s!"Goal Before:\n{formatGoalInfo step.goalBefore}"
  let goalsAfterStr :=
    if step.goalsAfter.isEmpty && step.spawnedGoals.isEmpty then
      "Goals After: (none - proved!)"
    else
      let regular := if step.goalsAfter.isEmpty then ""
                     else s!"Goals After:\n{formatGoalList step.goalsAfter}"
      let spawned := if step.spawnedGoals.isEmpty then ""
                     else s!"Spawned Goals:\n{formatGoalList step.spawnedGoals}"
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

/-- Create an SVG label for a vertex showing the tactic string -/
def tacticLabel (tactic : String) (isSpawned : Bool := false) : Html :=
  let displayText := truncateString tactic 30
  let textLen := displayText.length * 8 + 16
  let rectWidth : Int := max textLen 80
  let halfWidth : Int := rectWidth / 2
  let fillColor := if isSpawned then "#fff3e0" else "var(--vscode-editor-background)"
  let strokeColor := if isSpawned then "#ff9800" else "var(--vscode-editor-foreground)"

  Html.element "g" #[] #[
    Html.element "rect"
      #[("x", toJson (-halfWidth)),
        ("y", toJson (-15 : Int)),
        ("width", toJson rectWidth),
        ("height", toJson (30 : Nat)),
        ("rx", toJson (5 : Nat)),
        ("fill", toJson fillColor),
        ("stroke", toJson strokeColor),
        ("strokeWidth", toJson (1.5 : Float)),
        ("className", toJson "dim")]
      #[],
    Html.element "text"
      #[("textAnchor", toJson "middle"),
        ("dominantBaseline", toJson "middle"),
        ("fontSize", toJson "11"),
        ("fill", toJson "var(--vscode-editor-foreground)")]
      #[.text displayText]
  ]

/-- State for tree traversal, tracking vertices and edges -/
structure TreeGraphState where
  vertices : Array GraphDisplay.Vertex
  edges : Array GraphDisplay.Edge
  nextId : Nat
  deriving Inhabited

/-- Convert a ProofTree to GraphDisplay vertices and edges recursively -/
partial def proofTreeToGraph (tree : ProofTree) (parentId : Option String := none)
    (isSpawned : Bool := false) : StateM TreeGraphState Unit := do
  let state ← get
  let nodeId := s!"node_{state.nextId}"

  -- Create vertex for this node
  let textLen := tree.node.tacticString.length * 8 + 16
  let rectWidth : Float := Float.ofNat (max 80 textLen)
  let vertex : GraphDisplay.Vertex := {
    id := nodeId
    label := tacticLabel tree.node.tacticString isSpawned
    boundingShape := .rect rectWidth 30
    details? := some (proofStepDetailsHtml tree.node)
  }

  -- Create edge from parent if exists
  let newEdges := match parentId with
    | none => state.edges
    | some pid =>
      let edgeAttrs : Array (String × Json) :=
        if isSpawned then
          #[("stroke", "#ff9800"),
            ("strokeDasharray", "5,5"),
            ("strokeWidth", (2 : Nat)),
            ("markerEnd", "url(#arrow)")]
        else
          #[("stroke", "var(--vscode-editor-foreground)"),
            ("strokeWidth", (2 : Nat)),
            ("markerEnd", "url(#arrow)")]
      state.edges.push {
        source := pid
        target := nodeId
        attrs := edgeAttrs
      }

  set { state with
    vertices := state.vertices.push vertex
    edges := newEdges
    nextId := state.nextId + 1
  }

  -- spawned_children is a subset of children, so only iterate children once
  -- and check if each child is in spawned_children to determine styling
  for child in tree.children do
    let childIsSpawned := tree.spawned_children.any (· == child)
    proofTreeToGraph child (some nodeId) childIsSpawned

/-- Convert a ProofTree to GraphDisplay.Props -/
def proofTreeToGraphProps (tree : ProofTree) : GraphDisplay.Props :=
  let initState : TreeGraphState := {
    vertices := #[]
    edges := #[]
    nextId := 0
  }
  let (_, finalState) := proofTreeToGraph tree |>.run initState
  {
    vertices := finalState.vertices
    edges := finalState.edges
    showDetails := true
    forces := #[
      .link { distance? := some 100, strength? := some 0.8 },
      .collide { radius? := some 60 },
      .manyBody { strength? := some (-200) },
      .y { strength? := some 0.1 }
    ]
  }

/-- Generate HTML for displaying a ProofTree as an interactive graph.
    Use this function with `#html` to visualize a proof tree. -/
def proofTreeGraphHtml (tree : ProofTree) : Html :=
  let props := proofTreeToGraphProps tree
  Html.ofComponent GraphDisplay props #[]

/-!
## Example Usage

To visualize a proof tree extracted from a proof:

```lean
import metrics.declarativity3.proof_tree_widget

-- Extract a proof tree (see tree_test.lean for full example)
#eval do
  let (_, new_tree) ← extractProofTree `Module.Name `decl_name proof_string
  match new_tree with
  | some tree =>
    -- Use #html to display
    return proofTreeGraphHtml tree
  | none => return Html.text "No tree found"
```

Or, for a more direct approach using JSX:

```lean
def props := proofTreeToGraphProps myTree

#html <GraphDisplay
    vertices={props.vertices}
    edges={props.edges}
    showDetails={true}
    forces={props.forces}
  />
```
-/
