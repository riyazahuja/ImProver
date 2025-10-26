import TrainingData.TreeParser
import TrainingData.Frontend

open Lean Core Elab IO Meta Term Command Tactic




partial def getRangeOfSubtree (tree : ProofTree) : (Option String.Pos × Option String.Pos) :=
  let (starting?,ending?) := (tree.node.pos.map (fun x=>x.byteIdx), tree.node.tailPos.map (fun x=>x.byteIdx))
  let childRanges := tree.children.map getRangeOfSubtree |>.map (fun (s?, e?) => (s?.map (fun x => x.byteIdx), e?.map (fun x => x.byteIdx)))
  let max_enclosing_range := childRanges.foldl
    (fun (curr_start?, curr_end?) (child_start?, child_end?) =>
      let new_start? := match (curr_start?, child_start?) with
        | (some a, some b) => some (min a b)
        | (some a, none) => some a
        | (none, some b) => some b
        | (none, none) => none
      let new_end? := match (curr_end?, child_end?) with
        | (some a, some b) => some (max a b)
        | (some a, none) => some a
        | (none, some b) => some b
        | (none, none) => none
      (new_start?, new_end?))
    (starting?, ending?)
  (max_enclosing_range.1.map (fun x=> ⟨x⟩),max_enclosing_range.2.map (fun x=> ⟨x⟩))


-- Helper to get unique key for a node based on goal ID (more unique than position because of <;>)
def nodeKey (tree : ProofTree) : MVarId :=
  tree.node.goalBefore.id

partial def getValue (tree : ProofTree) (memo : Std.HashMap MVarId Nat := Std.HashMap.empty) : Std.HashMap MVarId Nat :=
let key := nodeKey tree
if memo.contains key then
  memo
else
  match tree.children with
  | #[] => memo.insert key 0  -- leaf node has value 0
  | children => Id.run do
    let childMaps := children.map getValue

    let mut combinedMap := memo
    for childMap in childMaps do
      for (k, v) in childMap.toList do
        combinedMap := combinedMap.insert k (max v (combinedMap.getD k v))


    let childValues := children.map (fun child => combinedMap.get! (nodeKey child))
    let minChildValue := childValues.foldl Nat.min (childValues.get! 0)
    combinedMap.insert key (minChildValue + 1)



def bestFirstSearch (tree : ProofTree) : Array ProofTree := Id.run do
  let valueMap := getValue tree

  let mut result : Array ProofTree := #[]
  let mut visited : Std.HashSet MVarId := Std.HashSet.empty  -- Track visited by goal ID
  let mut queue : Array ProofTree := #[tree]

  while !queue.isEmpty do
    -- Find the node with minimum value in the queue
    let minValueNode := queue.foldl (fun acc node =>
      if valueMap.getD (nodeKey node) 0 < valueMap.getD (nodeKey acc) 0 then node else acc
    ) queue[0]!

    -- Remove the selected node from queue (by goal ID)
    queue := queue.filter (fun n => nodeKey n != nodeKey minValueNode)

    -- Mark as visited
    visited := visited.insert (nodeKey minValueNode)

    -- Only add to result if tactic string is not already present
    if !result.any (fun n => n.node.tacticString == minValueNode.node.tacticString) then
      result := result.push minValueNode

    -- Add children to queue (check visited set by goal ID)
    for child in minValueNode.children do
      if !visited.contains (nodeKey child) then
        queue := queue.push child

  result


-- Helper to collect all node IDs in a tree that have tactic strings in the revealed set
partial def getRevealedNodeIds (tree : ProofTree) (revealedTactics : Std.HashSet String) : Std.HashSet MVarId := Id.run do
  let mut result := Std.HashSet.empty
  if revealedTactics.contains tree.node.tacticString then
    result := result.insert (nodeKey tree)
  for child in tree.children do
    result := (getRevealedNodeIds child revealedTactics).fold (fun acc id => acc.insert id) result
  return result

-- Helper to check if a node is a descendant of another
partial def isDescendantOf (node : ProofTree) (ancestor : ProofTree) : Bool :=
  ancestor.children.any (fun child =>
    nodeKey child == nodeKey node || isDescendantOf node child)

-- Helper to find nodes that should be masked with "sorry"
-- These are nodes NOT in revealed set whose parent IS in revealed set
partial def getMaskingPoints (tree : ProofTree) (revealed : Std.HashSet MVarId) (parentRevealed : Bool := true) : Array ProofTree :=
  let isRevealed := revealed.contains (nodeKey tree)
  if parentRevealed && !isRevealed then
    -- This node should be masked
    #[tree]
  else if isRevealed then
    -- This node is revealed, check its children
    let childPoints := tree.children.flatMap (fun child => getMaskingPoints child revealed true)
    -- Filter out any node that is a descendant of another (to avoid overlapping ranges)
    childPoints.filter (fun point =>
      !childPoints.any (fun other =>
        nodeKey point != nodeKey other && isDescendantOf point other))
  else
    -- Parent not revealed, so we don't descend
    #[]




-- Debug helper to count all nodes
partial def countAllNodes (tree : ProofTree) : Nat :=
  1 + (tree.children.toList.map countAllNodes).foldl (·+·) 0 + (tree.spawned_children.toList.map countAllNodes).foldl (·+·) 0

def getDenoisingTrajectories (cs : CompilationStep): IO (Array String) := do
  let tree? := getProofTree <| (← (cs.trees.filterMapM (BetterParser)) ).flatMap (fun result => result.steps)

  match tree? with
  | none => return #[]
  | some tree =>
    let totalNodes := countAllNodes tree
    IO.println s!"Total nodes in tree (including spawned): {totalNodes}"
    IO.println s!"  Regular children count: {tree.children.size}"
    IO.println s!"  Spawned children count: {tree.spawned_children.size}"

    let val := getValue tree
    IO.println s!"Value map size: {val.size}"

    -- Debug: trace down to find simp_all and aesop
    IO.println s!"\nTracing tree structure:"
    IO.println s!"Root: {tree.node.tacticString} -> children: {tree.children.toList.map (·.node.tacticString)}"

    -- Follow first child (intro h)
    let introH := tree.children[0]!
    IO.println s!"  intro h -> children: {introH.children.toList.map (·.node.tacticString)}"

    -- Follow apply normal_weakerThan_of_subset
    if introH.children.size > 0 then
      let applyNode := introH.children[0]!
      IO.println s!"    apply... -> children: {applyNode.children.toList.map (·.node.tacticString)}"

    -- Follow second child (have h₂)
    let haveH2 := tree.children[1]!
    IO.println s!"  have h₂ -> children: {haveH2.children.toList.map (·.node.tacticString)}"

    -- Follow intro φ hφ
    if haveH2.children.size > 0 then
      let introNode := haveH2.children[0]!
      IO.println s!"    intro φ hφ -> children: {introNode.children.toList.map (·.node.tacticString)}"

      -- Follow cases' hφ
      if introNode.children.size > 0 then
        let casesNode := introNode.children[0]!
        IO.println s!"      cases' hφ -> children: {casesNode.children.toList.map (·.node.tacticString)}"

        -- Check both children of cases
        if casesNode.children.size > 0 then
          let simpNode1 := casesNode.children[0]!
          IO.println s!"        simp_all[0] pos={simpNode1.node.pos}, tail={simpNode1.node.tailPos} -> children: {simpNode1.children.toList.map (·.node.tacticString)}"
        if casesNode.children.size > 1 then
          let simpNode2 := casesNode.children[1]!
          IO.println s!"        simp_all[1] pos={simpNode2.node.pos}, tail={simpNode2.node.tailPos} -> children: {simpNode2.children.toList.map (·.node.tacticString)}"

    let trajectories := bestFirstSearch tree
    IO.println s!"Trajectories found: \n{trajectories.map (fun t => t.node.tacticString)}"

    -- Generate progressive proof steps with masking
    let source := cs.src.str
    let mut steps := #[]

    for i in [0:trajectories.size] do
      -- Build set of revealed tactic strings for step i
      let revealedTactics := (trajectories.extract 0 (i + 1)).map (·.node.tacticString) |>.foldl (fun s t => s.insert t) Std.HashSet.empty

      -- Get all node IDs in the tree that have revealed tactic strings
      let revealed := getRevealedNodeIds tree revealedTactics

      -- Find nodes to mask (not revealed but parent is revealed)
      let maskingPoints := getMaskingPoints tree revealed

      -- Get their ranges
      let maskRanges := maskingPoints.filterMap (fun node =>
        let (s?, e?) := getRangeOfSubtree node
        match (s?, e?) with
        | (some s, some e) => some (s, e)
        | _ => none)

      -- Deduplicate and filter overlapping ranges (keep only non-overlapping ranges)
      -- For overlapping ranges, keep the larger one
      let uniqueRanges := maskRanges.foldl (fun acc range =>
        -- Check if this range is identical to or overlapped by an existing range
        let isRedundant := acc.any (fun r =>
          (r.1.byteIdx == range.1.byteIdx && r.2.byteIdx == range.2.byteIdx) || -- identical
          (r.1.byteIdx <= range.1.byteIdx && r.2.byteIdx >= range.2.byteIdx))   -- range is contained in r
        if isRedundant then acc
        else
          -- Remove any existing ranges that are contained in this range
          let filtered := acc.filter (fun r =>
            !(range.1.byteIdx <= r.1.byteIdx && range.2.byteIdx >= r.2.byteIdx))
          filtered.push range) #[]

      -- Sort ranges by starting position (descending, to process from end)
      let sortedRanges := uniqueRanges.qsort (fun (s1, _) (s2, _) => s1.byteIdx > s2.byteIdx)

      -- Apply masking from end to start (to preserve positions)
      let mut masked := source
      for (start, ending) in sortedRanges do
        masked := (masked.extract ⟨0⟩ start) ++ "sorry" ++ (masked.extract ending masked.endPos)

      -- Extract the proof text for this step
      let stepText := masked.extract cs.src.startPos (cs.src.stopPos.min masked.endPos)
      steps := steps.push stepText


    return steps


--     -- -- Get all nodes in BFS order
--     -- let allNodes := flattenTreeBFS tree
--     -- let nodeToIdx : Std.HashMap String Nat :=
--     --   allNodes.zipIdx.foldl (fun map (node, idx) => map.insert node.node.tacticString idx) Std.HashMap.empty

--     -- let topLevel := getTopLevelNodes tree
--     -- let topLevelCount := topLevel.length

--     -- let mut trajectories : Array String := #[]

--     -- if fine then
--     --   -- Fine mode: reveal nodes one by one
--     --   -- Step 0: nothing revealed
--     --   let step0 := reconstructProofWithSorry source tree Std.HashSet.empty nodeToIdx
--     --   trajectories := trajectories.push step0

--     --   -- Steps 1 to n: reveal nodes progressively
--     --   for i in [0:allNodes.length] do
--     --     let revealedSet := Std.HashSet.empty.insertMany (List.range (i + 1))
--     --     let trajectory := reconstructProofWithSorry source tree revealedSet nodeToIdx
--     --     trajectories := trajectories.push trajectory
--     -- else
--     --   -- Regular mode: reveal in layers
--     --   -- Step 0: nothing revealed
--     --   let step0 := reconstructProofWithSorry source tree Std.HashSet.empty nodeToIdx
--     --   trajectories := trajectories.push step0

--     --   -- Step 1: root + top-level nodes revealed (but not their children)
--     --   if topLevelCount > 0 then
--     --     -- Root is at index 0, top-level nodes are at indices 1 to topLevelCount
--     --     let revealedSet := Std.HashSet.empty.insertMany (List.range (topLevelCount + 1))
--     --     let trajectory := reconstructProofWithSorry source tree revealedSet nodeToIdx
--     --     trajectories := trajectories.push trajectory

--     --   -- Step 2: everything revealed (just return original source)
--     --   trajectories := trajectories.push source

--     -- return trajectories











def debugTraj (mod : Name) (decl : Name) (new_proof : String) : IO (Array String) := do
  searchPathRef.set compile_time_search_path%

  let fileName := (← findLean mod).toString
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl
  match target with
  | none => do
    IO.println s!"[ERROR] Target decls: {targets.map (fun (_, i) => i.name)}"
    throw <| IO.userError s!"Declaration {decl} not found in module {mod}"
  | some (target_cmd, _) => do
    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})

    let new_target ← elaborated_steps.head?

    match new_target with
    | none =>
      throw <| IO.userError s!"Failed to elaborate new proof for {decl} in module {mod}"
    | some new_target => do
      let new_score ← getDenoisingTrajectories new_target
      for (traj, idx) in new_score.zipIdx do
        IO.println s!"==== Step {idx} ====\n{traj}\n"
      return new_score


-- [CORRECT!]
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

#eval do debugTraj `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof



-- [CORRECT!]
def repeat_main_goal2 := "theorem foo (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h₁ : P ∧ Q → Q ∧ P := by
    intro h
    constructor
    . exact h.2
    . exact h.1
  exact h₁"



-- #eval do debugTraj `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul repeat_main_goal2



-- [NOT CORRECT! : corrupted outputs and have statements not going first, want to write h1, h2, h3, rw and then start going inside.]
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
-- #eval do debugTraj `HepLean.PerturbationTheory.FieldOpAlgebra.SuperCommute `FieldSpecification.FieldOpAlgebra.ofFieldOp_mul_ofFieldOp_eq_superCommute repeats_and_unused




-- [NOT CORRECT! : have statemetns not going first, want to write h1, h2, h3, exact and then start going inside. Also weird stuff around the <;>'s.]
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
-- #eval do debugTraj `Seymour.Matroid.Operations.Sum2.Regularity `TwoSumAssumptions.decomposition_isRegular_both new_proof2



-- [NOT CORRECT! : Similar corruption and ordering bugs as previous ones]
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
-- #eval do debugTraj `FLT.Mathlib.LinearAlgebra.Determinant `LinearMap.det_mul new_proof3
