import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System

-- may want to upgrade declarativity to use proof tree instead of just haves
-- def declarativity_score (cs : CompilationStep) : IO Float :=
--   let tac_stx := InfoTree.tactics_new (cs.trees) |>.map (fun x => x.info.stx)
--   let haves := tac_stx.filter (fun stx =>
--     match stx with
--     | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
--     | _ => false)
--   return haves.length |>.toFloat

partial def getSpawnedGoalsCount (tree : ProofTree) (acc : Nat := 0) : Nat :=
  let counts := tree.children.map (fun child => getSpawnedGoalsCount child acc)
  let curr := tree.spawned_children.size
  counts.foldl (fun a b => a + b) curr

def declarativity_score_old (cs : CompilationStep) : IO Float := do
  let tree? := getProofTree <| (← (cs.trees.filterMapM (BetterParser)) ).flatMap (fun result => result.steps)

  match tree? with
  | none => return (0 : Float)
  | some tree =>
    let spawned_goals_count := getSpawnedGoalsCount tree |>.toFloat
    return spawned_goals_count

  -- let tac_stx := InfoTree.tactics_new (cs.trees) |>.map (fun x => x.info.stx)
  -- let haves := tac_stx.filter (fun stx =>
  --   match stx with
  --   | Syntax.node _ `Lean.Parser.Tactic.tacticHave_ _ => true
  --   | _ => false)
  -- return haves.length |>.toFloat





/-- Stable key for local context (Γ) using hypothesis ids. -/
def hypsKey (g : GoalInfo) : List String :=
  -- order in `hyps` can vary; sort the fvar ids
  (g.hyps.map (·.id)).toArray.insertionSort (fun a b => a < b) |>.toList

/-- Same Γ (approx). -/
def sameContext (a b : GoalInfo) : Bool :=
  hypsKey a == hypsKey b

/-- Duplicate spawn (approx): same context and same pretty-printed goal type. -/
def isDuplicateSpawnApprox (parent child : ProofTree) : Bool :=
  sameContext parent.node.goalBefore child.node.goalBefore
  && parent.node.goalBefore.type == child.node.goalBefore.type

/-- Collect only *normal* (non-spawned) descendants. -/
partial def normalSubtreeNodes (t : ProofTree) : List ProofTree :=
  t.children.toList.flatMap (fun c => c :: normalSubtreeNodes c)

/-- Number of normal steps under `child`. -/
def workStepsUnder (child : ProofTree) : Nat :=
  (normalSubtreeNodes child).length

/-- Union of tactic dependencies in normal subtree (including child). -/
partial def depsInNormalSubtree (t : ProofTree)
    (acc : Std.HashSet String := {}): Std.HashSet String :=
  let acc' := t.node.tacticDependsOn.foldl (fun s x => s.insert x) acc
  t.children.foldl (fun s c => depsInNormalSubtree c s) acc'

/-- “Thin” = no normal work and no dependency use. -/
def isThin (child : ProofTree) : Bool :=
  workStepsUnder child == 0 && (depsInNormalSubtree child).isEmpty




/-- Find a hypothesis by fvar-id within the child’s local context. -/
def hypById (g : GoalInfo) (fid : String) : Option _root_.Hypothesis :=
  g.hyps.find? (·.id == fid)

/-- Is the child an alias-copy: `exact h` and `type(h) ≡ type(goal)`? -/
def isAliasCopySpawn (child : ProofTree) : Bool :=
  match child.node.tacticDependsOn with
  | [fid] =>
      match hypById child.node.goalBefore fid with
      | some hyp => hyp.type == child.node.goalBefore.type
      | none     => false
  | _ => false



/-- Hash-set utilities. -/
def hypIdSet (hs : List _root_.Hypothesis) : Std.HashSet String :=
  hs.foldl (init := {}) (fun s h => s.insert h.id)

def unionHypIds (lists : List (List _root_.Hypothesis)) : Std.HashSet String :=
  lists.foldl (init := {}) (fun s hs => hs.foldl (fun s' h => s'.insert h.id) s)

/-- Hypothesis ids that become available *after* this step (approx). -/
def introducedHypIds (st : ProofStep) : Std.HashSet String :=
  let before := hypIdSet st.goalBefore.hyps
  let after  := unionHypIds (st.goalsAfter.map (·.hyps))
  -- new hyps show up in `after` but were not in `before`
  after.fold (fun acc id => if before.contains id then acc else acc.insert id) ({} : Std.HashSet String)


/-- Flatten all nodes in a proof tree. -/
partial def allNodes (t : ProofTree) : List ProofTree :=
  t :: (t.children.toList.flatMap allNodes) ++ (t.spawned_children.toList.flatMap allNodes)

/-- Is node `b` textually after `a` (strictly), using positions? -/
def strictlyAfter (a b : ProofTree) : Bool :=
  match a.node.tailPos, b.node.pos with
  | some ta, some pb => ta.byteIdx ≤ pb.byteIdx
  | _, _             => false  -- if we lack positions, be conservative: not “after”

/-- Was the local produced by `child` ever used later (outside its own block)? -/
def usedLaterOnce (root child : ProofTree) : Bool :=
  let newIds := introducedHypIds child.node
  if newIds.isEmpty then
    false
  else
    (allNodes root).any (fun n =>
      strictlyAfter child n &&
      n.node.tacticDependsOn.any (fun fid => newIds.contains fid))


/-- Count a spawned edge parent→child iff it is not a dup and does some work or uses deps. -/
-- def admissibleSpawn (parent child : ProofTree) : Bool :=
--   let dup  := isDuplicateSpawnApprox parent child
--   let w    := workStepsUnder child
--   let deps := depsInNormalSubtree child
--   (!dup) && (w ≥ 1 || !deps.isEmpty)
def admissibleSpawn (root parent child : ProofTree) : Bool :=
  let dupParent := isDuplicateSpawnApprox parent child
  let aliasCopy := isAliasCopySpawn child
  let w    := workStepsUnder child
  let deps := depsInNormalSubtree child
  (!dupParent) && (!aliasCopy) && (w ≥ 1 || !deps.isEmpty) && usedLaterOnce root child


/-- Grow a linear chain following exactly one spawned child at each step. -/
partial def growSpawnChain (start : ProofTree) : List ProofTree :=
  let rec go (n : ProofTree) (acc : List ProofTree) : List ProofTree :=
    if n.spawned_children.size == 1 then
      let nxt := n.spawned_children.toList.head!
      go nxt (acc ++ [nxt])
    else acc
  go start []

/-- Suppress all but the head of any thin spawn-only chain of length ≥ ℓmin. -/
partial def collectSuppressedSpawnIds
  (t : ProofTree) (ℓmin : Nat := 3)
  (acc : Std.HashSet String := {}) : Std.HashSet String :=

  let acc1 :=
    if t.spawned_children.size ≥ 1 then
      let head := t.spawned_children.toList.head!
      let chain := head :: growSpawnChain head
      let allThin := chain.all isThin
      if allThin && chain.length ≥ ℓmin then
        -- suppress 2..end
        chain.drop 1 |>.foldl (fun s c => s.insert c.node.goalBefore.id.name.toString) acc
      else acc
    else acc

  let acc2 := t.children.foldl (fun s c => collectSuppressedSpawnIds c ℓmin s) acc1
  t.spawned_children.foldl (fun s c => collectSuppressedSpawnIds c ℓmin s) acc2



/-- Raw spawned-edge count (what you had, but totals everywhere). -/
partial def getRawSpawnCount (tree : ProofTree) : Nat :=
  tree.spawned_children.size
  + tree.children.toList.foldl (fun s c => s + getRawSpawnCount c) 0
  + tree.spawned_children.toList.foldl (fun s c => s + getRawSpawnCount c) 0

/-- Count only admissible spawned edges not suppressed by chain collapsing. -/
-- partial def countEffectiveSpawns (t : ProofTree) (suppress : Std.HashSet String) : Nat :=
--   let loc :=
--     t.spawned_children.toList
--       |>.filter (fun child =>
--         let key := child.node.goalBefore.id.name.toString
--         (!suppress.contains key) && admissibleSpawn t child)
--       |>.length
--   loc
--   + t.children.toList.foldl (fun s c => s + countEffectiveSpawns c suppress) 0
--   + t.spawned_children.toList.foldl (fun s c => s + countEffectiveSpawns c suppress) 0

partial def countEffectiveSpawns (root t : ProofTree) (suppress : Std.HashSet String) : Nat :=
  let loc :=
    t.spawned_children.toList
      |>.filter (fun child =>
        let key := child.node.goalBefore.id.name.toString
        (!suppress.contains key) && admissibleSpawn root t child)
      |>.length
  loc
  + t.children.toList.foldl (fun s c => s + countEffectiveSpawns root c suppress) 0
  + t.spawned_children.toList.foldl (fun s c => s + countEffectiveSpawns root c suppress) 0


/-- Reject proofs whose counted spawns are dominated by filtered ones. -/
def admissibleProofForModularity (raw eff : Nat) (ρ : Float := 0.5) : Bool :=
  if raw == 0 then true else
  let removed := raw - eff
  (Float.ofNat removed) / (Float.ofNat raw) ≤ ρ

def declarativity_score (cs : CompilationStep) : IO Float := do
  let steps := (← cs.trees.filterMapM BetterParser).flatMap (·.steps)
  match getProofTree steps with
  | none => return 0.0
  | some tree =>
    let raw := getRawSpawnCount tree
    let suppressed := collectSuppressedSpawnIds tree (3)   -- ℓmin = 3
    let eff := countEffectiveSpawns tree tree suppressed
    if admissibleProofForModularity raw eff 0.5 then
      return (Float.ofNat eff)
    else
      return 0.0
