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

/-- Replace fvars that are **not** in the current local context with fresh mvars. -/
partial def replaceUnknownFVarsWithMVars (e : Expr) : MetaM Expr := do
  let lctx ← getLCtx
  let rec go (e : Expr) : MetaM Expr := do
    match e with
    | .fvar fid =>
      match lctx.find? fid with
      | some _ => pure e
      | none   =>
        let u ← mkFreshLevelMVar
        let α ← mkFreshExprMVar (mkSort u)
        mkFreshExprMVar α
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
    1. Replace external fvars with mvars (prevents "unknown free variable" errors)
    2. Loosen universe levels (helps unification)
    3. Peel ALL Π-binders (∀ and →) from both sides
    4. Abstract bodies into lambdas over peeled parameters
    5. Apply lambdas with fresh mvars (standardizes parameter names)
    6. Check definitional equality on the applications

    This handles cases like:
    - Child: `∀ C : Set α, M.Circuit C → C.Nonempty`
    - Parent: `C.Nonempty` (where C and M.Circuit C are in context)
    - After peeling and standardizing: both reduce to `?C.Nonempty` → Match! -/
def defEqModuloForallMeta (a b : Expr) : MetaM Bool := do
  -- Step 1: Replace all external fvars with fresh mvars
  -- This handles fvars from the original proof context that don't exist in current lctx
  let a0 ← replaceUnknownFVarsWithMVars a
  let b0 ← replaceUnknownFVarsWithMVars b

  -- Step 2: Make universe levels flexible (helps unification)
  let a1 ← loosenLevels a0
  let b1 ← loosenLevels b0

  -- Step 3: Peel ALL Π-binders (∀ and →) from child
  forallTelescopeReducing a1 fun paramsA bodyA => do
    -- Child must have foralls to be a forall-abstraction duplicate
    if paramsA.isEmpty then
      return false

    -- Step 4: Peel ALL Π-binders from parent (might have none)
    forallTelescopeReducing b1 fun paramsB bodyB => do
      -- Step 5: Abstract child body over its parameters
      -- This creates: λ (p₁ : T₁) ... (pₙ : Tₙ) => bodyA
      let lamA ← mkLambdaFVars paramsA bodyA

      -- Step 6: Apply lambda with fresh mvars
      -- This gives: bodyA[p₁ := ?m₁, ..., pₙ := ?mₙ]
      let mvarsA ← paramsA.mapM (fun p => do
        let ty ← inferType p
        mkFreshExprMVar ty)
      let instA := mkAppN lamA mvarsA

      -- Step 7: Do same for parent (if it had any foralls)
      let instB ← if paramsB.isEmpty then
        -- Parent has no foralls, just use the body directly
        pure bodyB
      else
        -- Parent also had foralls, abstract and apply
        let lamB ← mkLambdaFVars paramsB bodyB
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
  IO.println s!"Checking defEqModuloForall between:\n A: {a}\n B: {b}"
  let coreCtx : Core.Context := { fileName := "<internal>", fileMap := default, options := {} }
  let coreState : Core.State := { env := env }
  let m := defEqModuloForallMeta a b
  let (output, _)← (m.run').toIO coreCtx coreState
  return output

-- edge depth i for e: A -> B means that B's depth is i
partial def getAllEdgesWithDepth (tree : ProofTree) (depth : Nat := 0) : List (Nat × EdgeStatus × ProofStep × ProofTree) := Id.run do
  let mut curr := []
  for child in tree.children.toList do
    let status := if tree.spawned_children.toList.contains child then EdgeStatus.effective else EdgeStatus.normal
    let recursive := getAllEdgesWithDepth child (depth + 1)
    curr := (depth, status, tree.node, child) :: curr ++ recursive
  curr



partial def initializeStatusMap (env : Environment) (tree : ProofTree) : IO StatusMap := do
  let mut statusMap : StatusMap := {}

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
          -- (3) Not a duplicate: fall back to your triviality gate
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
  let statusMap ← initializeStatusMap env tree

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

  if enableLogging then
    IO.println s!"\n[DEBUG] Dependency Graph:\n{graph.toString statusMap}\n\n"




  -- Get all spawned edge IDs
  let allNodesInTree := allNodes tree
  let mut spawnedIds : List EdgeId := []
  for node in allNodesInTree do
    for child in node.spawned_children.toList do
      let childId := getEdgeId child
      spawnedIds := childId :: spawnedIds

  let sorted := topologicalSort graph <| graph.dependencies.toList.map (·.1)
  if enableLogging then
    IO.println s!"[DEBUG] Topologically sorted spawned edge IDs: {List.intercalate ["\n"] <| sorted.map (fun s=> [statusMap[s]?.map (fun info => info.tacticStr) |>.getD ""])}\n"

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



def getScore2 (mod : Name) (decl : Name) (new_proof : String) : IO (Float× Float × Bool) := do
  let startTime ← IO.monoMsNow

  -- Don't set search path to avoid environment extension issues
  searchPathRef.set compile_time_search_path%
  let afterSearchPath ← IO.monoMsNow
  IO.println s!"[TIMING] Set search path: {afterSearchPath - startTime}ms"

  let fileName := (← findLean mod).toString
  -- IO.println s!"Found file: {fileName}"
  let afterFindLean ← IO.monoMsNow
  IO.println s!"[TIMING] Find lean file: {afterFindLean - afterSearchPath}ms"

  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  -- IO.println s!"{← moduleSource mod}"

  -- let forced ← steps.force
  -- for c in forced do
  --   IO.println s!"Compilation command: {c.src.toString}\n\n----------------------\n"

  let afterProcessInput ← IO.monoMsNow
  IO.println s!"[TIMING] Process input: {afterProcessInput - afterFindLean}ms"

  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let afterTargets ← IO.monoMsNow
  IO.println s!"[TIMING] Extract targets: {afterTargets - afterProcessInput}ms"

  let target := targets.find? fun (_, i) => i.name == decl

  match target with
  | none =>
    let endTime ← IO.monoMsNow
    IO.println s!"[TIMING] Total (no target found): {endTime - startTime}ms"
    IO.println s!"[ERROR] Target decls: {targets.map (fun (_, i) => i.name)}"
    return ((-1.0),(-1.0), false)
  | some (target_cmd, _) => do
    let beforeOgScore ← IO.monoMsNow
    IO.println s!"[TIMING] Find target: {beforeOgScore - afterTargets}ms"

    let og_score ← declarativity2_score target_cmd
    let afterOgScore ← IO.monoMsNow
    IO.println s!"[TIMING] Original declarativity score: {afterOgScore - beforeOgScore}ms"

    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString
    let afterBackgroundContent ← IO.monoMsNow
    IO.println s!"[TIMING] Extract background content: {afterBackgroundContent - afterOgScore}ms"

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})
    let afterElaboratedSteps ← IO.monoMsNow
    IO.println s!"[TIMING] Elaborate new steps: {afterElaboratedSteps - afterBackgroundContent}ms"

    let new_target ← elaborated_steps.head?
    let afterNewTarget ← IO.monoMsNow
    IO.println s!"[TIMING] Get new target: {afterNewTarget - afterElaboratedSteps}ms"

    match new_target with
    | none =>
      let endTime ← IO.monoMsNow
      IO.println s!"[TIMING] Total (no new target): {endTime - startTime}ms"
      return (og_score, (-1.0), false)
    | some new_target => do
      let beforeNewScore ← IO.monoMsNow
      let correct := not <| new_target.msgs.any (fun m => m.severity == .error)
      let afterCorrectCheck ← IO.monoMsNow
      IO.println s!"[TIMING] Check correctness: {afterCorrectCheck - beforeNewScore}ms"

      let new_score ← declarativity2_score new_target
      let endTime ← IO.monoMsNow
      IO.println s!"[TIMING] New declarativity score: {endTime - afterCorrectCheck}ms"
      IO.println s!"[TIMING] Total execution time: {endTime - startTime}ms"

      return (og_score, new_score, correct)



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
-- ((0.000000, 0.000000), (2.000000, 2.000000), true)
-- #eval do getScore2 `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof


def new_proof2 := "lemma TwoSumAssumptions.decomposition_isRegular_both {M₁ M₂ : Matroid α}
    (assumptions : TwoSumAssumptions M₁ M₂) (regularity : assumptions.build2sum.IsRegular) :
    M₁.IsRegular ∧ M₂.IsRegular := by
  -- Introduce a helper lemma to handle the regularity of each summand
  have h₁ : M₁.IsRegular → M₁.IsRegular ∧ M₂.IsRegular := by
    intro h
    -- Use the helper lemma to conclude the regularity of both summands
    exact ⟨h, assumptions.decomposition_isRegular_right regularity⟩ <;> aesop
  -- Similarly, introduce a helper lemma for the other summand
  have h₂ : M₂.IsRegular → M₁.IsRegular ∧ M₂.IsRegular := by
    intro h
    -- Use the helper lemma to conclude the regularity of both summands
    exact ⟨assumptions.decomposition_isRegular_left regularity, h⟩ <;> aesop
  -- Apply the helper lemmas to complete the proof
  have h₃ : M₁.IsRegular ∨ M₂.IsRegular → M₁.IsRegular ∧ M₂.IsRegular := by
    intro h
    -- Use the helper lemmas to conclude the proof
    rcases h with h <;> simp_all [h₁, h₂] <;> aesop
  -- Apply the final helper lemma to conclude the proof
  -- Regularity of the build2sum implies regularity of both summands
  exact ⟨assumptions.decomposition_isRegular_left regularity, assumptions.decomposition_isRegular_right regularity⟩ <;> aesop
"
-- ((0.000000, 0.000000), (3.000000, 1.000000), true)
-- #eval do getScore2 `Seymour.Matroid.Operations.Sum2.Regularity `TwoSumAssumptions.decomposition_isRegular_both new_proof2




def new_proof3 := "@[simp] lemma det_mul (a : R) : (mul R R a).det = a := by
  -- Lemma to handle the core determinant calculation
  have h₁ : ∀ a : R, (mul R R a).det = a → (mul R R a).det = a := by
    intro a h
    -- Apply the helper lemma to complete the proof
    rw [h]
  -- Lemma to handle the algebraic manipulation
  have h₂ : ∀ a : R, (mul R R a).det = a → (mul R R a).det = a := by
    intro a h
    -- Apply the helper lemma to complete the proof
    rw [h]
  -- Main lemma combining both steps
  classical
  rw [det_eq_det_toMatrix_of_finset (s := {1}) ⟨(Finsupp.LinearEquiv.finsuppUnique R R _).symm⟩, Matrix.det_unique]
  change a * _ = a
  simp
"
-- ((0.000000, 0.000000), (2.000000, 1.000000), true)
-- #eval do getScore2 `FLT.Mathlib.LinearAlgebra.Determinant `LinearMap.det_mul new_proof3


def unused := "lemma index_smul (a : G) (S : AddSubgroup A) : (a • S).index = S.index := by
  -- Introduce the first sub-proof to handle the general case of the bijection
  have h₁ : ∀ a : G, (a • S).index = S.index → (a • S).index = S.index := by
    intro a h_a
    -- Use the helper lemma to conclude the proof
    rw [h_a]
  -- Apply the helper lemma to complete the proof
  exact index_map_of_bijective _ (MulAction.bijective _)"
-- ((0.000000, 0.000000), (1.000000, 1.000000), true)
-- #eval do getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul unused


def repeats_and_unused := "lemma ofFieldOp_mul_ofFieldOp_eq_superCommute (φ φ' : 𝓕.FieldOp) :
    ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ
    + [ofFieldOp φ, ofFieldOp φ']ₛ := by
  -- Main lemma handling the general case
  have h₁ : ∀ φ φ' : 𝓕.FieldOp, ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ → ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ := by
    intro φ φ' h
    -- Apply the helper lemma to handle the specific case
    rw [h]
  -- Apply the main lemma to complete the proof
  have h₂ : ∀ φ φ' : 𝓕.FieldOp, ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ → ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ := by
    intro φ φ' h
    rw [h]
  -- Use the main lemma to conclude the proof
  have h₃ : ∀ φ φ' : 𝓕.FieldOp, ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ := by
    intro φ φ'
    rw [← ofFieldOpList_singleton, ← ofFieldOpList_singleton]
    rw [ofFieldOpList_mul_ofFieldOpList_eq_superCommute, ofFieldOpList_singleton]
    simp
  -- Apply the helper lemma to finalize the proof
  rw [h₃]"
-- ((0.000000, 0.000000), (3.000000, 2.000000), true)
-- #eval do getScore2 `HepLean.PerturbationTheory.FieldOpAlgebra.SuperCommute `FieldSpecification.FieldOpAlgebra.ofFieldOp_mul_ofFieldOp_eq_superCommute repeats_and_unused



def repeat_main_goal := "lemma L2norm_sq_eq {T : ℝ} [hT : Fact (0 < T)] (f : Lp ℂ 2 <| @haarAddCircle T hT) :
    ‖f‖ ^ 2 = ∫ (x : AddCircle T), ‖f x‖ ^ 2 ∂haarAddCircle := by
  -- Establish the basic framework of the proof
  have h1 : ‖f‖ ^ 2 = ∫ (x : AddCircle T), ‖f x‖ ^ 2 ∂haarAddCircle := by
    rw [@norm_sq_eq_inner ℂ, @L2.inner_def (AddCircle T) ℂ ℂ _ _ _ _ _ f f, ← integral_re (L2.integrable_inner f f)]
    simp only [← norm_sq_eq_inner]
  -- Conclude the proof by applying the established framework
  exact h1"
-- #eval do getScore2 `Carleson.Classical.SpectralProjectionBound `L2norm_sq_eq repeat_main_goal


theorem foo (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h₁ : P ∧ Q → Q ∧ P := by
    intro h
    constructor
    . exact h.2
    . exact h.1
  exact h₁

theorem foo2 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h₁ : ∀ P Q : Prop, P ∧ Q → Q ∧ P := by
    intro P Q h
    constructor
    . exact h.2
    . exact h.1
  exact h₁ P Q


def repeat_main_goal2 := "theorem foo (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h₁ : P ∧ Q → Q ∧ P := by
    intro h
    constructor
    . exact h.2
    . exact h.1
  exact h₁"



def repeat_main_goal3 := "theorem foo2 (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h₁ : ∀ P Q : Prop, P ∧ Q → Q ∧ P := by
    intro P Q h
    constructor
    . exact h.2
    . exact h.1
  exact h₁ P Q"

-- Test new_proof (should have 2 effective spawned edges: (0.0, 2.0, true))
-- #eval do
--   IO.println "\n=== Testing new_proof ==="
--   let result ← getScore2 `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof
--   IO.println s!"Result: {result}"

-- Test repeat_main_goal2 (exact duplicate: should be (0.0, 0.0, true))
-- #eval do
  -- IO.println "\n=== Testing repeat_main_goal2 ==="
  -- let result ← getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal2
  -- IO.println s!"Result: {result}"

-- Test repeat_main_goal3 (forall-abstraction duplicate: should be (0.0, 0.0, true))
-- #eval do
--   IO.println "\n=== Testing repeat_main_goal3 ==="
--   let result ← getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal3
--   IO.println s!"Result: {result}"



def repeat_main_goal4 := "theorem foo2 (n : ℕ) : n=n → n=n := by
  have h₁ : ∀ n : ℕ, n=n → n=n := by
    intro n hn
    have duh : ∀ m : ℕ, m=m := by
      intro m
      rfl
    exact duh n
  intro hn
  exact h₁ n hn"

-- #eval do
--   IO.println "\n=== Testing repeat_main_goal4 ==="
--   let result ← getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal4
--   IO.println s!"Result: {result}"






def matroid := "lemma Matroid.Circuit.nonempty {M : Matroid α} {C : Set α} (hC : M.Circuit C) : C.Nonempty := by
  -- Extract the property that a circuit must be nonempty
  have h_nonempty : ∀ C : Set α, M.Circuit C → C.Nonempty := by
    intro C hC
    -- Assume for contradiction that the circuit is empty
    by_contra! h_empty
    -- Rewrite the assumption to show the empty set cannot be a circuit
    rw [h_empty] at hC
    -- Derive a contradiction since an empty set cannot be a circuit
    exact hC.not_empty
  -- Apply the extracted property to conclude the proof
  apply h_nonempty
  exact hC"

#eval do
  IO.println "\n=== Testing matroid ==="
  let result ← getScore2 `Seymour.Matroid.Notions.Circuit `Matroid.Circuit.nonempty matroid
  IO.println s!"Result: {result}"


def singleton := "theorem op_eq_singleton_iff (x y : TSet γ) (z : TSet β) :
    op hβ hγ x y = singleton hβ z ↔ singleton hγ x = z ∧ singleton hγ y = z := by
  -- Define the equivalence for the operation op resulting in a singleton set
  have h1 : ∀ x y z, op hβ hγ x y = singleton hβ z ↔ singleton hγ x = z ∧ singleton hγ y = z := by
    intro x y z
    rw [op, up_eq_singleton_iff, and_congr_right_iff]
    rintro rfl
    simp only [up_eq_singleton_iff, true_and, singleton_inj]
  -- Apply the established equivalence
  exact h1 x y z"

-- #eval do
--   IO.println "\n=== Testing singleton ==="
--   let result ← getScore2 `ConNF.Model.Hailperin `ConNF.TSet.op_eq_singleton_iff singleton
--   IO.println s!"Result: {result}"
