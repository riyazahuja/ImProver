
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System



/-
What makes a good declarative proof?

filter all spawned edges to remove spawned goals that are immediately or trivially

bad: trivial - solved immediately or in a single exact step or a rw/exact pair (i.e. anything < 2 lines of work)  or doesn't use anything from its context
bad: duplicate - the spawned goal has the same goal type as a preceding goal, where the spawned goal's context is a superset of the preceding goal's context (i.e. the spawned goal is a duplicate of a previous goal, modulo extra hypotheses)
bad: utility - the hypotheses used by the spawned goal must be used later in the proof, and we must judge this "used later" condition by if that spawned goal was removed from the proof, the proof will fail. We implement this by checking if any of the hypotheses introduced by the spawned goal are used in any later goal outside of the spawned goal's own subtree, or if there is no other steps after the spawned goal (i.e. like a final calc statement), we just consider it good.


first pass: mark spawned goals as ineffective if they are trivial or duplicates of a previous goal.

then make a dependency tree between each edge e: A->B, where (e : A->B) -> (e':A'->B') are connected if B' uses any hypothesis introduced by e. each node in this dependency graph is marked as being normal or effective spawned or ineffective spawned. we want to efficiently go through the effective spawned goals for remarking such that for each (currently) effective spawned goal e, if in_edges(e) contains all ineffective spawned goals, or out_edges(e) contains all ineffective spawned goals, we remark e as ineffective. we repeat this until a fixed point is reached.

In practice, we don't want to repeat this process up to a fixpoint, rather, we want to choose the order of processing spawned goals such that we can do it in one pass. We can process spawned goals in dependency order: first process all spawned goals that have no dependencies on other spawned goals, then process those that depend only on already-processed spawned goals. This ensures that when we evaluate whether a spawned goal's dependencies are all ineffective, we've already determined the effectiveness of all its dependencies. A topological sort on the dependency graph achieves this ordering.

-/



/-


Step 1, augment proof tree with an status map (ineffective, effective, normal) for each edge.
Step 2, Initialize status map by marking trivial and duplicate spawned goals as ineffective, and all other spawned goals as effective.
- need to make trivial and duplicate checker. I.e. check all preceding goals (in any branch) for duplicate goal/hyps context, and check if the spawned goal has < 2 lines of work or doesn't use any hypotheses/constants from its context.

Step 3, initialize dependency graph between edges
- to do this, need helper functions to get introduced hypotheses, and to track what hypotheses a proofstep uses.
- edges on this graph are directed from the edge that introduces the hypothesis to the edge that uses it. we want to transpose this direction so that we can do a topological sort from edges with no dependencies to edges that depend on other edges, and mark edges based on the status of the items that depend on them.

Step 4, topological sort the dependency graph transpose and process each effective spawned edge in order, marking it as ineffective if all things that depend on it are ineffective, or if nothing depends on it.
-/

-- Step 1: EdgeStatus type
inductive EdgeStatus
  | normal
  | effective
  | ineffective
  deriving BEq, Inhabited

-- Reason for ineffectiveness
inductive IneffectiveReason
  | trivial (workSteps : Nat) (hasDeps : Bool)
  | duplicate (precedingGoalType : String)
  | noDependents
  | allDependentsIneffective (dependentIds : List Nat)
  deriving Inhabited

-- Extended status with reason
structure EdgeStatusInfo where
  status : EdgeStatus
  tacticStr : String
  reason : Option IneffectiveReason
  deriving Inhabited

-- Edge identifier: unique key for each proof step
def EdgeId := Nat
  deriving BEq, Hashable, Inhabited, ToString

-- Create a unique identifier for a ProofTree node
def getEdgeId (tree : ProofTree) : EdgeId :=
  tree.node.goalBefore.id.name.hash.toNat

-- Map from EdgeId to EdgeStatusInfo
abbrev StatusMap := Std.HashMap EdgeId EdgeStatusInfo

-- Helper to get tactic string from ProofTree
def getTacticStr (tree : ProofTree) : String :=
  tree.node.tacticString

-- Step 2 Helpers: Trivial and Duplicate Checkers

/-- Stable key for local context (Γ) using hypothesis ids. -/
def hypsKey (g : GoalInfo) : List String :=
  (g.hyps.map (·.id)).toArray.insertionSort (fun a b => a < b) |>.toList

/-- Same Γ (approx). -/
def sameContext (a b : GoalInfo) : Bool :=
  hypsKey a == hypsKey b

/-- Check if child is a duplicate of any preceding goal, returning the preceding goal type if found -/
-- NOT CHECKING MODULO FORALL!!!!
partial def isDuplicateOfPreceding (child : ProofTree) (preceding : List ProofTree) : Option String :=
  preceding.findSome? fun p =>
    let childGoal := child.node.goalBefore
    let precedingGoal := p.node.goalBefore
    -- Same goal type and same or superset context
    if childGoal.type == precedingGoal.type &&
       (hypsKey precedingGoal).all (fun h => (hypsKey childGoal).contains h) then
      some precedingGoal.type
    else
      none

/-- Collect only *normal* (non-spawned) descendants. -/
partial def normalSubtreeNodes (t : ProofTree) : List ProofTree :=
  -- Filter out spawned children to only count normal work
  let normalChildren := t.children.toList.filter (fun c =>
    !t.spawned_children.toList.contains c)
  normalChildren.flatMap (fun c => c :: normalSubtreeNodes c)

/-- Number of normal steps under `child`. -/
def workStepsUnder (child : ProofTree) : Nat :=
  (normalSubtreeNodes child).length   -- +1 for the child itself

/-- Union of tactic dependencies in normal subtree (including child). -/
partial def depsInNormalSubtree (t : ProofTree)
    (acc : Std.HashSet String := {}): Std.HashSet String :=
  let acc' := t.node.tacticDependsOn.foldl (fun s x => s.insert x) acc
  t.children.foldl (fun s c => depsInNormalSubtree c s) acc'

/-- Check if spawned goal is trivial: ≤ 1 work steps or doesn't use any dependencies -/
def isTrivial (child : ProofTree) : Bool :=
  workStepsUnder child ≤ 1 || (depsInNormalSubtree child).isEmpty







/-- Collect all free variable ids in an expression. -/
partial def gatherFVarIds (e : Expr) (acc : Std.HashSet FVarId := {}) : Std.HashSet FVarId :=
  match e with
  | .fvar fid        => acc.insert fid
  | .app f a         => gatherFVarIds a (gatherFVarIds f acc)
  | .lam _ ty bd _   => gatherFVarIds bd (gatherFVarIds ty acc)
  | .forallE _ ty bd _ => gatherFVarIds bd (gatherFVarIds ty acc)
  | .letE _ ty v b _ => gatherFVarIds b (gatherFVarIds v (gatherFVarIds ty acc))
  | .mdata _ b       => gatherFVarIds b acc
  | .proj _ _ b      => gatherFVarIds b acc
  | _                => acc

/-- Replace fvars that are **not** in the current local context with fresh mvars.
    Uses a shared cache to ensure the same fvar maps to the same mvar across multiple calls. -/
partial def replaceUnknownFVarsWithMVarsShared (e : Expr)
    (cache : IO.Ref (Std.HashMap FVarId Expr)) : MetaM Expr := do
  let lctx ← getLCtx
  let rec go (e : Expr) : MetaM Expr := do
    match e with
    | .fvar fid =>
      match lctx.find? fid with
      | some _ => pure e
      | none   =>
        -- Check cache first
        let cacheMap ← cache.get
        match cacheMap.get? fid with
        | some mvar => pure mvar
        | none =>
          let u ← mkFreshLevelMVar
          let α ← mkFreshExprMVar (mkSort u)
          let mvar ← mkFreshExprMVar α
          cache.set (cacheMap.insert fid mvar)
          pure mvar
    | .app f a           => return .app (← go f) (← go a)
    | .lam n ty b bi     => return .lam n (← go ty) (← go b) bi
    | .forallE n ty b bi => return .forallE n (← go ty) (← go b) bi
    | .letE n ty v b nd  => return .letE n (← go ty) (← go v) (← go b) nd
    | .mdata md b        => return .mdata md (← go b)
    | .proj s i b        => return .proj s i (← go b)
    | e                  => pure e
  go e

/-- Make universe levels flexible everywhere (helps defEq). -/
partial def loosenLevels (e : Expr) : MetaM Expr := do
  let cacheRef ← IO.mkRef ({} : Std.HashMap Name (Array Level))
  let rec go (e : Expr) : MetaM Expr := do
    match e with
    | .const nm lvls =>
      let cache ← cacheRef.get
      match cache.get? nm with
      | some newLvls => return .const nm newLvls.toList
      | none =>
        let newLvls ← lvls.mapM (fun _ => mkFreshLevelMVar)
        cacheRef.set (cache.insert nm newLvls.toArray)
        return .const nm newLvls
    | .app f a           => return .app (← go f) (← go a)
    | .lam n ty b bi     => return .lam n (← go ty) (← go b) bi
    | .forallE n ty b bi => return .forallE n (← go ty) (← go b) bi
    | .letE n ty v b nd  => return .letE n (← go ty) (← go v) (← go b) nd
    | .mdata md b        => return .mdata md (← go b)
    | .proj s i b        => return .proj s i (← go b)
    | _                  => pure e
  go e

/-- Definitional equality "modulo ∀": peel all Π/∀, standardize parameters, then check `isDefEq`.

    Strategy:
    1. Replace external fvars with mvars using SHARED cache (ensures same fvar → same mvar)
    2. Loosen universe levels (helps unification)
    3. Peel ALL Π-binders (∀ and →) from both sides
    4. Abstract bodies into lambdas over peeled parameters
    5. Apply lambdas with fresh mvars (standardizes parameter names)
    6. Check definitional equality on the applications

    This handles cases like:
    - Child: `∀ C : Set α, M.Circuit C → C.Nonempty`
    - Parent: `C.Nonempty` (where C and M.Circuit C are in context)
    - After peeling and standardizing: both reduce to `?C.Nonempty` → Match!

    ALSO handles exact duplicates:
    - Child: `P ∧ Q → Q ∧ P`
    - Parent: `P ∧ Q → Q ∧ P`
    - Both peel to same body, so isDefEq succeeds -/
def defEqModuloForallMeta (a b : Expr) : MetaM Bool := do
  -- Step 1: Create shared cache for fvar → mvar replacement
  -- This ensures the SAME fvar gets the SAME mvar in both expressions
  let sharedCache ← IO.mkRef ({} : Std.HashMap FVarId Expr)

  -- Step 2: Replace all external fvars with fresh mvars using SHARED cache
  let a0 ← replaceUnknownFVarsWithMVarsShared a sharedCache
  let b0 ← replaceUnknownFVarsWithMVarsShared b sharedCache



  -- Step 3: Make universe levels flexible (helps unification)
  let a1 ← loosenLevels a0
  let b1 ← loosenLevels b0



  -- Step 4: Peel ALL Π-binders (∀ and →) from child
  forallTelescopeReducing a1 fun paramsA bodyA => do
    -- Step 5: Peel ALL Π-binders from parent (might have none)
    forallTelescopeReducing b1 fun paramsB bodyB => do
      -- Step 6: Abstract child body over its parameters (if any)
      let instA ← if paramsA.isEmpty then
        pure bodyA
      else
        -- RAISING GIANT ERROR BUT IDK HOW IT AFFECTS PERFORMANCE????
        let lamA ←--try
          --mkLambdaFVars paramsA bodyA
        --catch e =>
          pure bodyA  -- fallback
        let mvarsA ← paramsA.mapM (fun p => do
          let ty ← inferType p
          mkFreshExprMVar ty)

        pure (mkAppN lamA mvarsA)

      -- Step 7: Do same for parent (if it had any foralls)
      let instB ← if paramsB.isEmpty then
        pure bodyB
      else
        -- RAISING GIANT ERROR BUT IDK HOW IT AFFECTS PERFORMANCE????
        let lamB ← --try
          --mkLambdaFVars paramsB bodyB
        --catch e =>
          pure bodyB  -- fallback
        let mvarsB ← paramsB.mapM (fun p => do
          let ty ← inferType p
          mkFreshExprMVar ty)

        pure (mkAppN lamB mvarsB)



      -- Step 8: Check definitional equality
      -- instA and instB now have standardized parameters (mvars)
      try
        isDefEq instA instB
      catch _ =>
        -- Handle any type errors gracefully
        return false




def defEqModuloForall (env : Environment) (a b : Expr) : IO Bool := do
  let coreCtx : Core.Context := { fileName := "<internal>", fileMap := default, options := {} }
  let coreState : Core.State := { env := env }
  let m := defEqModuloForallMeta a b
  try
    let (output, _) ← (m.run').toIO coreCtx coreState
    return output
  catch _ =>
    -- Handle any panics gracefully (e.g. unknown metavariable)
    return false

-- edge depth i for e: A -> B means that B's depth is i
partial def getAllEdgesWithDepth (tree : ProofTree) (depth : Nat := 0) : List (Nat × EdgeStatus × ProofStep × ProofTree) := Id.run do
  let mut curr := []
  for child in tree.children.toList do
    let status := if tree.spawned_children.toList.contains child then EdgeStatus.effective else EdgeStatus.normal
    let recursive := getAllEdgesWithDepth child (depth + 1)
    curr := (depth, status, tree.node, child) :: curr ++ recursive
  curr



partial def initializeStatusMap (env : Environment) (tree : ProofTree) (enableLogging : Bool := false) : IO StatusMap := do
  let mut statusMap : StatusMap := {}

  -- Extract the root theorem goal for duplication checking
  let rootGoal := tree.node.goalBefore

  let allEdgesWithDepth := getAllEdgesWithDepth tree |>.toArray
  let depthMap : Std.HashMap Nat (List (EdgeStatus × ProofStep × ProofTree)) :=
    allEdgesWithDepth.foldl (fun dm (d, s, ps, pt) =>
      let existing := dm.getD d []
      dm.insert d ((s, ps, pt) :: existing)
    ) {}

  for (depth, status, parent, child) in allEdgesWithDepth do
    let edgeId := getEdgeId child
    let childTacticStr := getTacticStr child
    let tacticStr := s!"([{parent.tacticString}] -> [{childTacticStr}])"

    if status == EdgeStatus.effective then
      let childGoal := child.node.goalBefore
      let parentGoal := parent.goalBefore

      -- (0) Duplicate to root theorem, modulo forall-unrolling
      let dupToRoot ← defEqModuloForall env childGoal.typeExpr rootGoal.typeExpr
      if enableLogging then
        IO.println s!"[DEBUG] Checking root dup for {edgeId}"
        IO.println s!"  Child goal: {childGoal.type}"
        IO.println s!"  Root goal: {rootGoal.type}"
        IO.println s!"  Result: {dupToRoot}"
      if dupToRoot then
        let info : EdgeStatusInfo := {
          status := .ineffective
          tacticStr := tacticStr
          reason := some (IneffectiveReason.duplicate s!"{rootGoal.type} (root/mod ∀)")
        }
        statusMap := statusMap.insert edgeId info
        continue

      -- (1) Duplicate to parent, modulo forall-unrolling
      let dupToParent ← defEqModuloForall env childGoal.typeExpr parent.goalBefore.typeExpr
      if dupToParent then
        let info : EdgeStatusInfo := {
          status := .ineffective
          tacticStr := tacticStr
          reason := some (IneffectiveReason.duplicate s!"{parentGoal.type} (mod ∀)")
        }
        statusMap := statusMap.insert edgeId info
        continue

      -- (2) Duplicate to any preceding (≤ depth), modulo forall-unrolling
      let preceding : List (EdgeStatus × ProofStep × ProofTree) :=
        depthMap.toList.filter (fun (d, _) => d ≤ depth) |>.flatMap (·.2)

      let mut hit : Option String := none
      for (_, _, preTree) in preceding do
        if getEdgeId preTree == edgeId then
          continue
        let preGoal := preTree.node.goalBefore

        let ok ← defEqModuloForall env childGoal.typeExpr preGoal.typeExpr
        if ok then
          hit := some preGoal.type
          break

      match hit with
      | some typ =>
          let info : EdgeStatusInfo := {
            status := .ineffective
            tacticStr := tacticStr
            reason := some (IneffectiveReason.duplicate s!"{typ} (mod ∀)")
          }
          statusMap := statusMap.insert edgeId info
      | none =>
          -- (3) Not a duplicate: fall back to triviality gate
          let workSteps := workStepsUnder child
          let hasDeps := !(depsInNormalSubtree child).isEmpty
          if workSteps ≤ 1 || !hasDeps then
            let info : EdgeStatusInfo := {
              status := .ineffective
              tacticStr := tacticStr
              reason := some (IneffectiveReason.trivial workSteps hasDeps)
            }
            statusMap := statusMap.insert edgeId info
          else
            let info : EdgeStatusInfo := {
              status := .effective
              tacticStr := tacticStr
              reason := none
            }
            statusMap := statusMap.insert edgeId info

    else
      -- normal edge
      let info : EdgeStatusInfo := {
        status := .normal
        tacticStr := tacticStr
        reason := none
      }
      statusMap := statusMap.insert edgeId info

  return statusMap




-- Step 3: Build dependency graph

/-- Hash-set utilities. -/
def hypIdSet (hs : List _root_.Hypothesis) : Std.HashSet String :=
  hs.foldl (init := {}) (fun s h => s.insert h.id)

def unionHypIds (lists : List (List _root_.Hypothesis)) : Std.HashSet String :=
  lists.foldl (init := {}) (fun s hs => hs.foldl (fun s' h => s'.insert h.id) s)

/-- Hypothesis ids that become available *after* this step (approx). -/
def introducedHypIds (st : ProofStep) : Std.HashSet String :=
  let before := hypIdSet st.goalBefore.hyps
  -- Include both goalsAfter AND spawnedGoals
  let after  := unionHypIds ((st.goalsAfter ++ st.spawnedGoals).map (·.hyps))
  after.fold (fun acc id => if before.contains id then acc else acc.insert id) ({} : Std.HashSet String)

/-- Flatten all nodes in a proof tree. -/
partial def allNodes (t : ProofTree) : List ProofTree :=
  -- spawned_children is a subset of children, so only iterate children to avoid double-counting
  t :: (t.children.toList.flatMap allNodes)

/-- Check if `node` is a descendant of `ancestor` in the proof tree.
    Only checks NORMAL children, not spawned children, to avoid filtering siblings. -/
partial def isDescendantOf (nodeId : EdgeId) (ancestor : ProofTree) : Bool :=
  -- Only check normal children (filter out spawned children)
  -- This ensures we only look within the subtree, not at siblings
  let normalChildren := ancestor.children.toList.filter (fun c =>
    !ancestor.spawned_children.toList.contains c)
  normalChildren.any fun child =>
    getEdgeId child == nodeId || isDescendantOf nodeId child

/-- Build dependency graph: edge A->B depends on edge C->D if B uses hypotheses introduced by C->D -/
structure DependencyGraph where
  -- Map from EdgeId to list of EdgeIds it depends on (edges whose hypotheses it uses)
  dependencies : Std.HashMap EdgeId (List EdgeId)
  -- Map from EdgeId to list of EdgeIds that depend on it (reverse edges)
  dependents : Std.HashMap EdgeId (List EdgeId)
  deriving Inhabited


  /-- ToString instance for DependencyGraph -/
  instance : ToString DependencyGraph := ⟨fun graph =>
    let depLines := graph.dependencies.toList.map fun (id, deps) =>
      s!"  {id} depends on: {deps}"
    let depentLines := graph.dependents.toList.map fun (id, deps) =>
      s!"  {id} has dependents: {deps}"
    s!"Dependencies:\n{"\n".intercalate depLines}\nDependents:\n{"\n".intercalate depentLines}"⟩

  /-- Enhanced toString with tactic information -/
  def DependencyGraph.toString (graph : DependencyGraph) (statusMap : StatusMap) : String :=
    let depLines := graph.dependencies.toList.map fun (id, deps) =>
      let tacticStr := statusMap.get? id |>.map (·.tacticStr) |>.getD "unknown"
      let depTactics := deps.map fun depId =>
        let depTactic := statusMap.get? depId |>.map (·.tacticStr) |>.getD "unknown"
        s!"{depId}({depTactic})"
      s!"  {tacticStr}\n\tdepends on: {List.intercalate ["\n"] <| depTactics.map (fun s => [s])}"
    let depentLines := graph.dependents.toList.map fun (id, deps) =>
      let tacticStr := statusMap.get? id |>.map (·.tacticStr) |>.getD "unknown"
      let depTactics := deps.map fun depId =>
        let depTactic := statusMap.get? depId |>.map (·.tacticStr) |>.getD "unknown"
        s!"{depId}({depTactic})"
      s!"  [{id}]{tacticStr} has dependents: {depTactics}"
    s!"Dependencies:\n{"\n".intercalate depLines}\nDependents:\n{"\n".intercalate depentLines}"


/-- Collect all spawned edge IDs in the tree -/
partial def getAllSpawnedEdgeIds (tree : ProofTree) : Std.HashSet EdgeId := Id.run do
  let mut spawnedIds : Std.HashSet EdgeId := {}
  let allNodesInTree := allNodes tree
  for node in allNodesInTree do
    for child in node.spawned_children.toList do
      spawnedIds := spawnedIds.insert (getEdgeId child)
  return spawnedIds

/-- Build mapping from spawned edge ID to parent node ID -/
partial def buildSpawnedToParentMap (tree : ProofTree) : Std.HashMap EdgeId EdgeId := Id.run do
  let mut spawnedToParent : Std.HashMap EdgeId EdgeId := {}
  let allNodesInTree := allNodes tree
  for node in allNodesInTree do
    for child in node.spawned_children.toList do
      spawnedToParent := spawnedToParent.insert (getEdgeId child) (getEdgeId node)
  return spawnedToParent

/-- Build dependency graph from proof tree -/
partial def buildDependencyGraph (root : ProofTree) (spawnedEdgeIds : Std.HashSet EdgeId) (enableLogging : Bool := false) : IO DependencyGraph := do
  let allNodesInTree := allNodes root
  let mut dependencies : Std.HashMap EdgeId (List EdgeId) := {}
  let mut dependents : Std.HashMap EdgeId (List EdgeId) := {}

  -- Build a map from EdgeId to ProofTree for quick lookup
  let mut nodeMap : Std.HashMap EdgeId ProofTree := {}
  for node in allNodesInTree do
    nodeMap := nodeMap.insert (getEdgeId node) node

  if enableLogging then
    IO.println "\n[DEBUG] Building dependency graph..."

  -- For each node, check which other nodes' hypotheses it uses
  for node in allNodesInTree do
    let nodeId := getEdgeId node
    let nodeTactic := getTacticStr node
    let nodeDepends := node.node.tacticDependsOn
    let mut nodeDeps : List EdgeId := []

    if enableLogging && !nodeDepends.isEmpty then
      IO.println s!"[DEBUG] Node {nodeId} ({nodeTactic}) depends on fvars: {nodeDepends}"

    -- Check which preceding nodes introduced hypotheses used by this node
    for otherNode in allNodesInTree do
      let otherId := getEdgeId otherNode
      if otherId != nodeId then
        let introducedHyps := introducedHypIds otherNode.node
        -- Check if any dependency of current node was introduced by otherNode
        let matchingDeps := nodeDepends.filter (fun dep => introducedHyps.contains dep)
        if !matchingDeps.isEmpty then
          if enableLogging then
            IO.println s!"  -> Found that {otherId} ({getTacticStr otherNode}) introduced: {matchingDeps}"

          -- CRITICAL FILTER: Only add dependency if node is NOT within a spawned subtree
          -- Two cases to check:
          -- 1. If otherNode is a spawned edge itself, check if node is a descendant of otherNode
          -- 2. Otherwise, check if node is a descendant of any of otherNode's spawned children
          let isInSpawnedSubtree :=
            if spawnedEdgeIds.contains otherId then
              -- otherNode is a spawned edge, check if using node is its descendant
              nodeId != otherId && isDescendantOf nodeId otherNode
            else
              -- otherNode is not spawned, check its spawned children
              otherNode.spawned_children.toList.any (fun spawnedChild =>
                getEdgeId spawnedChild == nodeId || isDescendantOf nodeId spawnedChild)

          if enableLogging then
            let checkType := if spawnedEdgeIds.contains otherId then "spawned edge itself" else "spawned children"
            IO.println s!"  -> isInSpawnedSubtree({nodeId}, {otherId}'s {checkType}) = {isInSpawnedSubtree}"

          if !isInSpawnedSubtree then
            nodeDeps := otherId :: nodeDeps
            -- Update reverse edge
            let existing := dependents.getD otherId []
            dependents := dependents.insert otherId (nodeId :: existing)
            if enableLogging then
              IO.println s!"  -> Added dependency: {nodeId} depends on {otherId}"
          else
            if enableLogging then
              IO.println s!"  -> FILTERED OUT (descendant relationship)"

    dependencies := dependencies.insert nodeId nodeDeps

  if enableLogging then
    IO.println "\n[DEBUG] Final dependents map:"
    for (id, deps) in dependents.toList do
      IO.println s!"  {id}: {deps.length} dependents"

  return { dependencies, dependents }

-- Step 4: Topological sort and update status map

/-- Topological sort using Kahn's algorithm -/
def topologicalSort (graph : DependencyGraph) (nodes : List EdgeId) : List EdgeId := Id.run do
  let mut result : List EdgeId := []
  let mut inDegree : Std.HashMap EdgeId Nat := {}
  let mut queue : List EdgeId := []

  -- Calculate in-degrees
  for node in nodes do
    let deps := graph.dependencies.getD node []
    inDegree := inDegree.insert node deps.length
    if deps.isEmpty then
      queue := node :: queue

  -- Process queue
  let mut currentQueue := queue
  while !currentQueue.isEmpty do
    match currentQueue with
    | [] => break
    | node :: rest =>
      result := result ++ [node]
      currentQueue := rest

      -- Update dependents
      let deps := graph.dependents.getD node []
      for dep in deps do
        let currentDegree := inDegree.getD dep 0
        if currentDegree > 0 then
          let newDegree := currentDegree - 1
          inDegree := inDegree.insert dep newDegree
          if newDegree == 0 then
            currentQueue := currentQueue ++ [dep]

  result

/-- Process spawned goals in dependency order and update status -/
def processSpawnedGoalsInOrder (graph : DependencyGraph) (statusMap : StatusMap)
    (spawnedIds : List EdgeId) (spawnedToParent : Std.HashMap EdgeId EdgeId) : StatusMap :=
  let sorted := topologicalSort graph spawnedIds

  sorted.foldl (fun sm nodeId =>
    -- Check current status
    match sm.get? nodeId with
    | none => sm
    | some info =>
      match info.status with
      | EdgeStatus.normal => sm
      | EdgeStatus.ineffective => sm  -- Already ineffective
      | EdgeStatus.effective =>
        -- Key insight: The spawned edge is the proof, but the hypothesis is introduced by the parent node
        -- Check dependents of the PARENT node (the "have" statement), not the spawned edge itself
        let parentId := spawnedToParent.getD nodeId nodeId  -- fallback to self if no parent found
        let deps := graph.dependents.getD parentId []

        if deps.isEmpty then
          -- Nothing depends on parent's introduced hypothesis -> ineffective
          let updatedInfo : EdgeStatusInfo := {
            status := EdgeStatus.ineffective
            tacticStr := info.tacticStr
            reason := some IneffectiveReason.noDependents
          }
          sm.insert nodeId updatedInfo
        else
          -- Check if all dependents are ineffective spawned edges
          let ineffectiveDeps := deps.filter fun depId =>
            match sm.get? depId with
            | some depInfo => depInfo.status == EdgeStatus.ineffective
            | _ => false

          if ineffectiveDeps.length == deps.length then
            let updatedInfo : EdgeStatusInfo := {
              status := EdgeStatus.ineffective
              tacticStr := info.tacticStr
              reason := some (IneffectiveReason.allDependentsIneffective deps)
            }
            sm.insert nodeId updatedInfo
          else
            sm  -- Keep effective
  ) statusMap

/-- Count effective spawned edges in tree using status map -/
partial def countEffectiveSpawnsFromMap (tree : ProofTree) (statusMap : StatusMap) : Nat :=
  let localCount := tree.spawned_children.toList.filter (fun child =>
    let edgeId := getEdgeId child
    match statusMap.get? edgeId with
    | some info => info.status == EdgeStatus.effective
    | _ => false
  ) |>.length

  -- Only recurse into children (which includes spawned_children), don't double-count
  localCount
  + tree.children.toList.foldl (fun s c => s + countEffectiveSpawnsFromMap c statusMap) 0

/-- Format reason for logging -/
def formatReason (reason : IneffectiveReason) (statusMap : StatusMap) : String :=
  match reason with
  | IneffectiveReason.trivial workSteps hasDeps =>
    s!"TRIVIAL (workSteps={workSteps}, hasDeps={hasDeps})"
  | IneffectiveReason.duplicate precedingType =>
    s!"DUPLICATE (precedingType={precedingType})"
  | IneffectiveReason.noDependents =>
    "NO_DEPENDENTS (nothing uses hypotheses introduced by this edge)"
  | IneffectiveReason.allDependentsIneffective depIds =>
    let depTactics := depIds.filterMap (fun id =>
      statusMap.get? id |>.map (fun info => s!"{id}:{info.tacticStr}"))
    s!"ALL_DEPENDENTS_INEFFECTIVE (depends on: {depTactics})"

/-- Log spawned edge status -/
def logSpawnedEdgeStatus (edgeId : EdgeId) (info : EdgeStatusInfo) (statusMap : StatusMap) : IO Unit := do
  let statusStr := match info.status with
    | EdgeStatus.effective => "EFFECTIVE"
    | EdgeStatus.ineffective => "INEFFECTIVE"
    | EdgeStatus.normal => "NORMAL"

  let reasonStr := match info.reason with
    | none => "N/A"
    | some r => formatReason r statusMap

  IO.println s!"[EdgeId={edgeId}] Status={statusStr} | Tactic: {info.tacticStr} | Reason: {reasonStr}"

/-- Main function: compute effective spawned edges count -/
def computeEffectiveSpawns (env : Environment) (tree : ProofTree) (enableLogging : Bool := false) : IO Nat := do
  -- Step 2: Initialize status map
  let statusMap ← initializeStatusMap env tree enableLogging
  IO.println s!"Tree:\n{tree.toString}\n\n"
  if enableLogging then
    IO.println "\n========== INITIAL SPAWNED EDGE STATUS =========="
    for (edgeId, info) in statusMap.toList do
      logSpawnedEdgeStatus edgeId info statusMap
    IO.println "===============================================\n"

  -- Step 2.5: Collect all spawned edge IDs and build parent mapping
  let spawnedEdgeIds := getAllSpawnedEdgeIds tree
  let spawnedToParent := buildSpawnedToParentMap tree

  -- Step 3: Build dependency graph
  let graph ← buildDependencyGraph tree spawnedEdgeIds enableLogging

  -- if enableLogging then
  --   IO.println s!"\n[DEBUG] Dependency Graph:\n{graph.toString statusMap}\n\n"




  -- Get all spawned edge IDs
  let allNodesInTree := allNodes tree
  let mut spawnedIds : List EdgeId := []
  for node in allNodesInTree do
    for child in node.spawned_children.toList do
      let childId := getEdgeId child
      spawnedIds := childId :: spawnedIds

  let sorted := topologicalSort graph <| graph.dependencies.toList.map (·.1)
  -- if enableLogging then
    -- IO.println s!"[DEBUG] Topologically sorted spawned edge IDs: {List.intercalate ["\n"] <| sorted.map (fun s=> [statusMap[s]?.map (fun info => info.tacticStr) |>.getD ""])}\n"

  -- Step 4: Process in dependency order
  let finalStatusMap := processSpawnedGoalsInOrder graph statusMap (graph.dependencies.toList.map (·.1)) spawnedToParent

  -- Log results
  if enableLogging then
    IO.println "\n========== SPAWNED EDGE ANALYSIS =========="
    for (edgeId, info) in finalStatusMap.toList do
      logSpawnedEdgeStatus edgeId info finalStatusMap
    IO.println "============================================\n"

  -- Count effective spawns
  return countEffectiveSpawnsFromMap tree finalStatusMap

/-- Main scoring function -/
def declarativity2_score (cs : CompilationStep) : IO Float := do
  let steps := (← cs.trees.filterMapM BetterParser).flatMap (·.steps)
  IO.println s!"Total steps in compilation: {steps.length}"
  match getProofTree steps with
  | none => return 0.0
  | some tree =>
    -- Enable logging by setting to true
    -- Use the 'after' environment from the compilation step
    let eff ← computeEffectiveSpawns cs.after tree true
    return eff |>.toFloat
