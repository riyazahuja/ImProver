import Lean
import Lean.Meta.Basic
import Lean.Meta.CollectMVars
import Init.Data.String.Basic
import TrainingData.Utils.utils

open Lean Elab Server Std String

abbrev TypeKey := UInt64

structure Hypothesis where
  username : String
  type : String
  value : Option String
  -- unique identifier for the hypothesis, fvarId
  id : String
  fid      : FVarId
  isProof : String
  typeKey : TypeKey
  typeExpr : Expr
  deriving Inhabited--, ToJson, FromJson

structure GoalInfo where
  username : String
  type : String
  hyps : List Hypothesis
  -- unique identifier for the goal, mvarId
  id : MVarId
  typeKey : TypeKey
  typeExpr : Expr
  deriving Inhabited--, ToJson, FromJson

open Meta in
def typeKeyOf (e : Expr) : MetaM TypeKey := do
  let e ← instantiateMVars e
  let e ← whnf e
  pure e.hash -- `Expr.hash` is UInt64

/-- Compute a key for a local hypothesis type under the goal's lctx. -/
def hypTypeKey (printCtx : ContextInfo) (lctx : LocalContext) (hypDecl : LocalDecl) : IO TypeKey := do
  printCtx.runMetaM lctx do typeKeyOf hypDecl.type

/-- Compute a key for the goal type. -/
def goalTypeKey (printCtx : ContextInfo) (decl : MetavarDecl) : IO TypeKey := do
  printCtx.runMetaM decl.lctx do typeKeyOf decl.type



instance : BEq GoalInfo where
  beq g1 g2 := g1.id == g2.id

instance : Hashable GoalInfo where
  hash g := hash g.id

instance : ToJson Pos where
  toJson pos := Json.num pos.byteIdx

instance : FromJson Pos where
  fromJson? json := match json.getNat?.toOption with
    | some n => .ok { byteIdx := n}
    | _ => .error s!"expected Nat for Pos, got '{json}'"

structure ProofStep where
  tacticString : String
  goalBefore : GoalInfo
  goalsAfter : List GoalInfo
  tacticDependsOn : List String
  spawnedGoals : List GoalInfo
  pos : Option Pos := none
  tailPos : Option Pos := none
  deriving Inhabited--, ToJson, FromJson

def stepGoalsAfter (step : ProofStep) : List GoalInfo := step.goalsAfter ++ step.spawnedGoals

def noInEdgeGoals (allGoals : Std.HashSet GoalInfo) (steps : List ProofStep) : Std.HashSet GoalInfo :=
  -- Some of the orphaned goals might be matched by tactics in sibling subtrees, e.g. for tacticSeq.
  (steps.bind stepGoalsAfter).foldl HashSet.erase allGoals

/-
  Instead of doing parsing of what user wrote (it wouldn't work for linarith etc),
  let's do the following.
  We have assigned something to our goal in mctxAfter.
  All the fvars used in these assignments are what was actually used instead of what was in syntax.
-/
def findHypsUsedByTactic (goalId: MVarId) (goalDecl : MetavarDecl) (mctxAfter : MetavarContext) : MetaM (List String) := do
  let some expr := mctxAfter.eAssignment.find? goalId
    | return []

  -- Need to instantiate it to get all fvars
  let fullExpr ← instantiateExprMVars expr --|>.run
  let fvarIds := (collectFVars {} fullExpr).fvarIds
  let fvars := fvarIds.filterMap goalDecl.lctx.find?
  let proofFvars ← fvars.filterM (Meta.isProof ·.toExpr)
  -- let pretty := proofFvars.map (fun x => x.userName)
  -- dbg_trace s!"Used {pretty}"
  return proofFvars.map (fun x => x.fvarId.name.toString) |>.toList

-- This is used to match goalsBefore with goalsAfter to see what was assigned to what
def findMVarsAssigned (goalId : MVarId) (mctxAfter : MetavarContext) : MetaM (List MVarId) := do
  let some expr := mctxAfter.eAssignment.find? goalId
    | return []
  let (_, s) ← (Meta.collectMVars expr).run {}
  return s.result.toList

def mayBeProof (expr : Expr) : MetaM String := do
  let type : Expr ← Lean.Meta.inferType expr
  if ← Meta.isProof expr then
    return "proof"
  if type.isSort then
    return "universe"
  else
    return "data"

def printGoalInfo (printCtx : ContextInfo) (id : MVarId) : IO GoalInfo := do
  let some decl := printCtx.mctx.findDecl? id
    | panic! "printGoalInfo: goal not found in the mctx"
  -- to get tombstones in name ✝ for unreachable hypothesis
  let lctx := decl.lctx |>.sanitizeNames.run' {options := {}}
  let ppContext := printCtx.toPPContext lctx
  let hyps ← lctx.foldrM (init := []) (fun hypDecl acc => do
    if hypDecl.isAuxDecl || hypDecl.isImplementationDetail then
      return acc
    let type ← liftM (ppExprWithInfos ppContext hypDecl.type)
    let value ← liftM (hypDecl.value?.mapM (ppExprWithInfos ppContext))
    let isProof : String ← printCtx.runMetaM decl.lctx (mayBeProof hypDecl.toExpr)
    let tkey ← hypTypeKey printCtx decl.lctx hypDecl
    return ({
      username := hypDecl.userName.toString,
      type := type.fmt.pretty,
      value := value.map (·.fmt.pretty),
      fid := hypDecl.fvarId,
      id := hypDecl.fvarId.name.toString,
      isProof := isProof,
      typeKey := tkey,
      typeExpr := hypDecl.type
    } : Hypothesis) :: acc)
  let gkey ← goalTypeKey printCtx decl
  return ⟨ decl.userName.toString, (← ppExprWithInfos ppContext decl.type).fmt.pretty, hyps, id, gkey, decl.type⟩

-- Returns unassigned goals from the provided list of goals
def getUnassignedGoals (goals : List MVarId) (mctx : MetavarContext) : IO (List MVarId) := do
  goals.filterMapM fun id => do
    if let none := mctx.findDecl? id then
      return none
    if mctx.eAssignment.contains id ||
       mctx.dAssignment.contains id then
      return none
    return some id

structure Result where
  steps : List ProofStep
  allGoals : Std.HashSet GoalInfo
  deriving Inhabited

def getGoalsChange (ctx : ContextInfo) (tInfo : TacticInfo) : IO (List (List String × GoalInfo × List GoalInfo)) := do
  -- We want to filter out `focus` like tactics which don't do any assignments
  -- therefore we check all goals on whether they were assigned during the tactic
  let goalMVars := tInfo.goalsBefore ++ tInfo.goalsAfter
  -- For printing purposes we always need to use the latest mctx assignments. For example in
  -- have h := by calc
  --  3 ≤ 4 := by trivial
  --  4 ≤ 5 := by trivial
  -- at mctxBefore type of `h` is `?m.260`, but by the time calc is elaborated at mctxAfter
  -- it's known to be `3 ≤ 5`
  let printCtx := {ctx with mctx := tInfo.mctxAfter}
  let mut goalsBefore ← getUnassignedGoals goalMVars tInfo.mctxBefore
  let mut goalsAfter ← getUnassignedGoals goalMVars tInfo.mctxAfter
  let commonGoals := goalsBefore.filter fun g => goalsAfter.contains g
  goalsBefore := goalsBefore.filter (!commonGoals.contains ·)
  goalsAfter :=  goalsAfter.filter (!commonGoals.contains ·)
  -- We need to match them into (goalBefore, goalsAfter) pairs according to assignment.
  let mut result : List (List String × GoalInfo × List GoalInfo) := []
  for goalBefore in goalsBefore do
    if let some goalDecl := tInfo.mctxBefore.findDecl? goalBefore then
      let assignedMVars ← ctx.runMetaM goalDecl.lctx (findMVarsAssigned goalBefore tInfo.mctxAfter)
      let tacticDependsOn ← ctx.runMetaM goalDecl.lctx
          (findHypsUsedByTactic goalBefore goalDecl tInfo.mctxAfter)

      result := (
        tacticDependsOn,
        ← printGoalInfo printCtx goalBefore,
        ← goalsAfter.filter assignedMVars.contains |>.mapM (printGoalInfo printCtx)
      ) :: result
  return result

def prettifySteps (stx : Syntax) (steps : List ProofStep) : List ProofStep := Id.run do
  match stx with
  | `(tactic| rw [$_,*] $(_)?)
  | `(tactic| rewrite [$_,*] $(_)?) =>
    let prettify (tStr : String) :=
      let res := tStr.trim.dropRightWhile (· == ',')
      -- rw puts final rfl on the "]" token
      if res == "]" then "rfl" else res
    return steps.map fun a => { a with tacticString := s!"rw [{prettify a.tacticString}]" }
  | _ => return steps

-- Comparator for names, e.g. so that _uniq.34 and _uniq.102 go in the right order.
-- That's not completely right because it doesn't compare prefixes but
-- it's much shorter to write than correct version and serves the purpose.
def nameNumLt (n1 n2 : Name) : Bool :=
  match n1, n2 with
  | .num _ n₁, .num _ n₂ => n₁ < n₂
  | .num _ _,  _ => true
  | _, _ => false

partial def postNode (ctx : ContextInfo) (i : Info) (_: PersistentArray InfoTree) (res : List (Option Result)) : IO Result := do
    let res := res.filterMap id
    let some ctx := i.updateContext? ctx
      | panic! "unexpected context node"
    let steps := res.map (fun r => r.steps) |>.join
    let allSubGoals := Std.HashSet.empty.insertMany $ res.bind (·.allGoals.toList)
    if let .ofTacticInfo tInfo := i then
      -- shortcut if it's not a tactic user wrote
      -- \n trim to avoid empty lines/comments until next tactic,
      -- especially at the end of theorem it will capture comment for the next one
      let some tacticString := tInfo.stx.getSubstring?.map
             (·.toString |>.splitOn "\n" |>.head!.trim)
        | return {steps, allGoals := allSubGoals}

      let pos := tInfo.stx.getPos?
      let tailPos := tInfo.stx.getTailPos?

      let steps := prettifySteps tInfo.stx steps

      let proofTreeEdges ← getGoalsChange ctx tInfo
      let currentGoals := proofTreeEdges.map (fun ⟨ _, g₁, gs ⟩ => g₁ :: gs)  |>.join
      let allGoals := allSubGoals.insertMany $ currentGoals
      -- It's like tacticDependsOn but unnamed mvars instead of hyps.
      -- Important to sort for have := calc for example, e.g. calc 3 < 4 ... 4 < 5 ...
      let orphanedGoals := currentGoals.foldl Std.HashSet.erase (noInEdgeGoals allGoals steps)
        |>.toArray.insertionSort (nameNumLt ·.id.name ·.id.name) |>.toList

      let newSteps := proofTreeEdges.filterMap fun ⟨ tacticDependsOn, goalBefore, goalsAfter ⟩ =>
       -- Leave only steps which are not handled in the subtree.
        if steps.map (·.goalBefore) |>.elem goalBefore then
          none
        else
          some {
            tacticString,
            goalBefore,
            goalsAfter,
            tacticDependsOn,
            spawnedGoals := orphanedGoals
            pos := pos
            tailPos := tailPos
          }

      return { steps := newSteps ++ steps, allGoals }
    else
      return { steps, allGoals := allSubGoals}

partial def BetterParser (i : InfoTree) := i.visitM (postNode := postNode)




def filter_universe_hyp (gi : GoalInfo) : GoalInfo :=
    ⟨gi.username,
    gi.type,
    gi.hyps.filter (fun hyp => not <| hyp.isProof == "universe"),
    gi.id,
    gi.typeKey,
    gi.typeExpr
    ⟩

-- def filterBacktracking (steps : List ProofStep) : List ProofStep := Id.run do
--   let mut result : List ProofStep := []
--   let mut

def existsArrow (nodeA : ProofStep) (nodeB : ProofStep) : Bool :=
  let outputGoalsA := nodeA.goalsAfter ++ nodeA.spawnedGoals |>.map (fun goal=>goal.id)
  let inputGoalB := nodeB.goalBefore.id
  outputGoalsA.any (fun id => id==inputGoalB)

def existsSpawnedArrow (nodeA : ProofStep) (nodeB : ProofStep) : Bool :=
  let outputGoalsA := nodeA.spawnedGoals.map (fun goal=>goal.id)
  let inputGoalB := nodeB.goalBefore.id
  outputGoalsA.any (fun id => id==inputGoalB)

def getProofTree' (steps : List ProofStep) : List (String × (List Nat) × (List Nat)) :=

  let steps : List ProofStep := steps.map (fun step =>
    ⟨step.tacticString,
      filter_universe_hyp step.goalBefore,
      step.goalsAfter.map filter_universe_hyp,
      step.tacticDependsOn,
      step.spawnedGoals.map filter_universe_hyp,
      step.pos,
      step.tailPos
      ⟩)

  if steps.isEmpty then []
  else
    let combos := steps.enum.bind (fun (i,a) => steps.enum.filterMap (fun (j,b) => if i != j then some ((i,a),(j,b)) else none))
    let arrow_idx : List (Nat × Nat):= combos.filterMap (fun (nodeA,nodeB) => if existsArrow nodeA.2 nodeB.2 then some (nodeA.1,nodeB.1) else none)
    let spawned_arrow_idx : List (Nat × Nat):= combos.filterMap (fun (nodeA,nodeB) => if existsSpawnedArrow nodeA.2 nodeB.2 then some (nodeA.1,nodeB.1) else none)
    --let base_nodes : List (List Nat) := List.range (steps.length) |>.map (fun i => arrow_idx.filterMap (fun (j,k) => if i == j then some k else none))
    let base_nodes : List ((List Nat) × (List Nat)) := List.range (steps.length) |>.map
      (fun i => (arrow_idx.filterMap (fun (j,k) => if i == j then some k else none),
        spawned_arrow_idx.filterMap (fun (j,k) => if i == j then some k else none)))

    let idx_map := steps.map (fun ps => ps.tacticString)
    base_nodes.enum.map (fun (i,xs) => (idx_map.get! i, xs))



structure ProofTree where
  node : ProofStep
  children :  Array ProofTree
  spawned_children :  Array ProofTree
deriving Inhabited--, ToJson, FromJson

instance : BEq ProofTree where
  beq t1 t2 := t1.node.tacticString == t2.node.tacticString

partial def ptts_helper (t : ProofTree) (indent : String) (isFirst : Bool) (isSpawned : Bool) : String :=
  let prefix' := if isFirst then indent else indent ++ "└─ "
  let childIndent := if isFirst then indent else indent ++ "   "

  let nodeStr := if isSpawned then
                   s!"{prefix'}[*{t.node.tacticString} | {t.node.pos} -> {t.node.tailPos}]"
                 else
                   s!"{prefix'}[{t.node.tacticString} | {t.node.pos} -> {t.node.tailPos}]"

  let allChildren := t.children.toList ++ t.spawned_children.toList

  if allChildren.isEmpty then
    nodeStr
  else
    let childrenStrs := allChildren.enum.map fun (i, child) =>
      let isLast := i == allChildren.length - 1
      let nextIndent := if isLast then childIndent else childIndent ++ "│  "
      let isChildSpawned := t.spawned_children.toList.contains child
      "\n" ++ ptts_helper child nextIndent false isChildSpawned

    nodeStr ++ String.join childrenStrs

def ProofTree.toString (tree : ProofTree) : String :=
  ptts_helper tree "" true false

instance : ToString ProofTree where
  toString := ProofTree.toString



partial def ProofTree.getBreakpoints (tree : ProofTree)
  (breakpoint_type : String := "all_splits"): List ProofStep :=
  match breakpoint_type with
  | "all_tactics" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpoints breakpoint_type) |>.flatten
    tree.node :: recursive
  | "all_splits" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpoints breakpoint_type) |>.flatten
    if tree.children.size < 2 then
      recursive
    else
      tree.children.toList.map (fun c => c.node) ++ recursive
  | "spawned" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpoints breakpoint_type) |>.flatten
    if tree.spawned_children.size < 2 then
      recursive
    else
      tree.children.toList.map (fun c => c.node) ++ recursive
  | "bifurcated" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpoints breakpoint_type) |>.flatten
    if tree.children.size - tree.spawned_children.size < 2 then
      recursive
    else
      tree.children.toList.map (fun c => c.node) ++ recursive
  | _ => []



partial def ProofTree.getBreakpointsWithDescendents (tree : ProofTree)
  (breakpoint_type : String := "all_splits"): List ProofTree :=
  match breakpoint_type with
  | "all_tactics" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpointsWithDescendents breakpoint_type) |>.flatten
    tree :: recursive
  | "all_splits" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpointsWithDescendents breakpoint_type) |>.flatten
    if tree.children.size < 2 then
      recursive
    else
      tree.children.toList ++ recursive
  | "spawned" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpointsWithDescendents breakpoint_type) |>.flatten
    -- if tree.spawned_children.size < 2 then
    --   recursive
    -- else
    tree.spawned_children.toList ++ recursive
  | "bifurcated" =>
    let recursive := tree.children.toList.map (fun child => child.getBreakpointsWithDescendents breakpoint_type) |>.flatten
    if tree.children.size - tree.spawned_children.size < 2 then
      recursive
    else
      tree.children.toList ++ recursive
  | _ => []

partial def getLeaves (tree : ProofTree) : List ProofTree :=
  if tree.children.isEmpty && tree.spawned_children.isEmpty then
    [tree]
  else
    (tree.children.toList ++ tree.spawned_children.toList).flatMap getLeaves

-- def AugmentBreakpointsWithDescendents (text : String) (breakpoints : List ProofStep) : List (ProofStep × Substring) :=
--   breakpoints.filterMap (fun ps =>
--     if ps.pos.isNone || ps.tailPos.isNone then
--       none
--     else
--       let startPos := ps.pos.get!
--       let tailPos := ps.tailPos.get!
--       let substring := ⟨text, pos, tailPos⟩
--       some (ps, substring))

-- Now want a function that takes in a string representation of a theorem
-- and all the breakpoints, and replaces each breakpoint B with extract_goals; B
-- To do this, we proceed greedily. For B[0], split theorem at first instance into T0,R
-- with T0 before B[0], R after B[0]. Replace B[0] with extract_goals; B[0] and then recurse on R
-- with B[1]. We get back a list of strings T0, T1, ... and output append(T0, ...)


partial def insertBreakpoints (thm : String) (breakpoints : List ProofStep) : String :=
  match breakpoints with
  | [] => thm
  | currentBreakpoint :: rest =>
    let tacticToFind := currentBreakpoint.tacticString
    let opt := thm.splitAtString tacticToFind
    match opt with
    | some (T,R) =>
      let recursive := insertBreakpoints R rest
      s!"{T}better_extract_goal; {tacticToFind}{recursive}"
    | none => -- silent errors, i.e. Hydra <;>'s etc.
      let recursive := insertBreakpoints thm rest
      recursive

def insertBreakpointsFromTree (thm : String) (tree : ProofTree) (breakpointType : String := "all_splits") : String :=
  let breakpoints := tree.getBreakpoints breakpointType
  insertBreakpoints thm breakpoints

def insertBreakpointsFromTree' (thm : String) (breakpoints : List ProofStep) : String :=
  insertBreakpoints thm breakpoints


partial def buildProofTree (steps : List ProofStep) (output : List (ProofStep × List Nat × List Nat)) (rootIdx : Nat) : ProofTree :=
  let (node, children, spawnedChildren) := output[rootIdx]!
  let childTrees := children.map (buildProofTree steps output)
  let spawnedChildTrees := spawnedChildren.map (buildProofTree steps output)
  {
    node := node,
    children := childTrees.toArray,
    spawned_children := spawnedChildTrees.toArray
  }

def findRootIndices (output : List (ProofStep × List Nat × List Nat)) : List Nat :=
  let allChildren := output.bind (fun (_, children, spawnedChildren) => children ++ spawnedChildren)
  List.range output.length |>.filter (fun i => !allChildren.contains i)

def getProofTree (steps : List ProofStep) : Option ProofTree :=

  let steps : List ProofStep := steps.map (fun step =>
    ⟨step.tacticString,
      filter_universe_hyp step.goalBefore,
      step.goalsAfter.map filter_universe_hyp,
      step.tacticDependsOn,
      step.spawnedGoals.map filter_universe_hyp,
      step.pos,
      step.tailPos
      ⟩)

  if steps.isEmpty then none
  else
    let combos := steps.enum.bind (fun (i,a) => steps.enum.filterMap (fun (j,b) => if i != j then some ((i,a),(j,b)) else none))
    let arrow_idx : List (Nat × Nat):= combos.filterMap (fun (nodeA,nodeB) => if existsArrow nodeA.2 nodeB.2 then some (nodeA.1,nodeB.1) else none)
    let spawned_arrow_idx : List (Nat × Nat):= combos.filterMap (fun (nodeA,nodeB) => if existsSpawnedArrow nodeA.2 nodeB.2 then some (nodeA.1,nodeB.1) else none)
    --let base_nodes : List (List Nat) := List.range (steps.length) |>.map (fun i => arrow_idx.filterMap (fun (j,k) => if i == j then some k else none))
    let base_nodes : List ((List Nat) × (List Nat)) := List.range (steps.length) |>.map
      (fun i => (arrow_idx.filterMap (fun (j,k) => if i == j then some k else none),
        spawned_arrow_idx.filterMap (fun (j,k) => if i == j then some k else none)))

    let idx_map := steps--.map (fun ps => ps.tacticString)
    let output : List (ProofStep × List Nat × List Nat) := base_nodes.enum.map (fun (i,xs) => (idx_map.get! i, xs))
    let rootIndices := findRootIndices output
    if rootIndices.isEmpty then
      none
    else
      -- Assuming the first root index is the main root of the proof tree
      some (buildProofTree steps output rootIndices.head!)
