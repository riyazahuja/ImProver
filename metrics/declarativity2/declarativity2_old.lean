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

-- partial def getSpawnedGoalsCount (tree : ProofTree) (acc : Nat := 0) : Nat :=
--   let counts := tree.children.map (fun child => getSpawnedGoalsCount child acc)
--   let curr := tree.spawned_children.size
--   counts.foldl (fun a b => a + b) curr

-- def declarativity_score_old (cs : CompilationStep) : IO Float := do
--   let tree? := getProofTree <| (← (cs.trees.filterMapM (BetterParser)) ).flatMap (fun result => result.steps)

--   match tree? with
--   | none => return (0 : Float)
--   | some tree =>
--     let spawned_goals_count := getSpawnedGoalsCount tree |>.toFloat
--     return spawned_goals_count

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

def declarativity2_score (cs : CompilationStep) : IO Float := do
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





-- -- def background:= "/- This file contains helper lemmas. Either they should be replaced by a mathlib version if there is\n   one or they might be candidates to go there, possibly in a generalized form. -/\n\nimport Carleson.ToMathlib.Misc\nimport Mathlib.MeasureTheory.Integral.IntervalIntegral\n\nopen MeasureTheory\n\ntheorem Real.volume_uIoc {a b : ℝ} : volume (Set.uIoc a b) = ENNReal.ofReal |b - a| := by\n  /- Cf. proof of Real.volume_interval-/\n  rw [Set.uIoc, volume_Ioc, max_sub_min_eq_abs]\n\nlemma intervalIntegral.integral_conj' {μ : Measure ℝ} {𝕜 : Type} [RCLike 𝕜] {f : ℝ → 𝕜} {a b : ℝ}:\n    ∫ x in a..b, (starRingEnd 𝕜) (f x) ∂μ = (starRingEnd 𝕜) (∫ x in a..b, f x ∂μ) := by\n  rw [intervalIntegral_eq_integral_uIoc, integral_conj, intervalIntegral_eq_integral_uIoc,\n      RCLike.real_smul_eq_coe_mul, RCLike.real_smul_eq_coe_mul, map_mul, RCLike.conj_ofReal]\n\nlemma intervalIntegrable_of_bdd {a b : ℝ} {δ : ℝ} {g : ℝ → ℂ} (measurable_g : Measurable g) (bddg : ∀ x, ‖g x‖ ≤ δ) : IntervalIntegrable g volume a b := by\n  apply @IntervalIntegrable.mono_fun' _ _ _ _ _ _ (fun _ ↦ δ)\n  · exact intervalIntegrable_const\n  · exact measurable_g.aestronglyMeasurable\n  · rw [Filter.EventuallyLE, ae_restrict_iff_subtype measurableSet_uIoc]\n    apply Filter.Eventually.of_forall\n    rw [Subtype.forall]\n    exact fun x _ ↦ bddg x\n\nlemma IntervalIntegrable.bdd_mul {F : Type} [NormedDivisionRing F] {f g : ℝ → F} {a b : ℝ} {μ : Measure ℝ}\n    (hg : IntervalIntegrable g μ a b) (hm : AEStronglyMeasurable f μ) (hfbdd : ∃ C, ∀ x, ‖f x‖ ≤ C) : IntervalIntegrable (fun x ↦ f x * g x) μ a b := by\n  rw [intervalIntegrable_iff, IntegrableOn]\n  apply Integrable.bdd_mul _ hm.restrict hfbdd\n  rwa [← IntegrableOn, ← intervalIntegrable_iff]\n\nlemma IntervalIntegrable.mul_bdd {F : Type} [NormedField F] {f g : ℝ → F} {a b : ℝ} {μ : Measure ℝ}\n    (hf : IntervalIntegrable f μ a b) (hm : AEStronglyMeasurable g μ) (hgbdd : ∃ C, ∀ x, ‖g x‖ ≤ C) : IntervalIntegrable (fun x ↦ f x * g x) μ a b := by\n  conv => pattern (fun x ↦ f x * g x); ext x; rw [mul_comm]\n  exact hf.bdd_mul hm hgbdd\n\nlemma IntegrableOn.sub {α : Type} {β : Type} {m : MeasurableSpace α}\n    {μ : Measure α} [NormedAddCommGroup β] {s : Set α} {f g : α → β} (hf : IntegrableOn f s μ) (hg : IntegrableOn g s μ) : IntegrableOn (f - g) s μ := by\n  apply Integrable.sub <;> rwa [← IntegrableOn]\n\n\nlemma ConditionallyCompleteLattice.le_biSup {α : Type} [ConditionallyCompleteLinearOrder α] {ι : Type} [Nonempty ι]\n    {f : ι → α} {s : Set ι} {a : α} (hfs : BddAbove (f '' s)) (ha : ∃ i ∈ s, f i = a) :\n    a ≤ ⨆ i ∈ s, f i := by\n  apply ConditionallyCompleteLattice.le_csSup\n  · --TODO: improve this\n    rw [bddAbove_def] at *\n    rcases hfs with ⟨x, hx⟩\n    use (max x (sSup ∅))\n    intro y hy\n    simp only [Set.mem_range] at hy\n    rcases hy with ⟨z, hz⟩\n    rw [iSup] at hz\n    by_cases h : z ∈ s\n    · have : (@Set.range α (z ∈ s) fun _ ↦ f z) = {f z} := by\n        rw [Set.eq_singleton_iff_unique_mem]\n        exact ⟨Set.mem_range_self h, fun x hx ↦ hx.2.symm⟩\n      rw [this, csSup_singleton _] at hz\n      have : f z ≤ x := by\n        simp only [Set.mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂] at hx\n        exact hx z h\n      rw [hz] at this\n      exact le_max_of_le_left this\n    have : (@Set.range α (z ∈ s) fun _ ↦ f z) = ∅ := by simpa\n    rw [this] at hz\n    exact hz ▸ le_max_right x y\n  rw [Set.mem_range]\n  rcases ha with ⟨i, hi, fia⟩\n  use i\n  rw [iSup]\n  convert csSup_singleton _\n  rw [Set.eq_singleton_iff_unique_mem]\n  refine ⟨⟨hi, fia⟩, fun x hx ↦ ?_⟩\n  simp only [Set.mem_range, exists_prop] at hx\n  rwa [hx.2] at fia\n\n\n/-Adapted from mathlib Function.Periodic.exists_mem_Ico₀-/\ntheorem Function.Periodic.exists_mem_Ico₀' {α : Type} {β : Type} {f : α → β} {c : α}\n  [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c) (hc : 0 < c) (x : α) : ∃ (n : ℤ), (x - n • c) ∈ Set.Ico 0 c ∧ f x = f (x - n • c) :=\n  let ⟨n, H, _⟩ := existsUnique_zsmul_near_of_pos' hc x\n  ⟨n, H, (h.sub_zsmul_eq n).symm⟩\n\n/-Adapted from mathlib Function.Periodic.exists_mem_Ico₀-/\n"
-- def background := "import Mathlib.Data.Set.Lattice
-- import Mathlib.Data.Set.Function
-- import Mathlib.Analysis.SpecialFunctions.Log.Basic
-- import Mathlib.Data.Real.Basic
-- import Mathlib.Data.Nat.Factorization.Basic
-- import Mathlib.Data.Nat.Prime.Basic


-- theorem even_of_even_sqr {m : ℕ} (h : 2 ∣ m ^ 2) : 2 ∣ m := by
--   rw [pow_two, Nat.prime_two.dvd_mul] at h
--   cases h <;> assumption
-- "
-- def content := "
-- lemma inter_diff_assoc_NEGATIVE_EXAMPLE (s t u : Set α) :
--     s ∩ (t \\ u) = (s ∩ t) \\ u := by
--   ext x; constructor <;> intro hx
--   · rcases hx with ⟨hs, ht, hnotu⟩
--     exact ⟨⟨hs, ht⟩, hnotu⟩
--   · rcases hx with ⟨⟨hs, ht⟩, hnotu⟩
--     exact ⟨hs, ht, hnotu⟩
-- "

-- def content2 := "

-- lemma inter_diff_assoc_NEGATIVE_EXAMPLE (s t u : Set α) :
--   s ∩ (t \\ u) = (s ∩ t) \\ u := by
--   -- BAD: restate the goal and prove it inside a single mega-`have`
--   have everything : s ∩ (t \\ u) = (s ∩ t) \\ u := by
--     -- BAD: pointless, never-used facts
--     have hTrue : True := True.intro
--     have id₁ : s = s := rfl
--     have id₂ : t = t := rfl
--     have id₃ : u = u := rfl
--     have reflEq : (s ∩ (t \\ u)) = (s ∩ (t \\ u)) := rfl
--     -- bury both ⊆ directions inside the blob
--     ext x; constructor
--     · intro hx
--       -- BAD: trivial rebindings + unused junk
--       have hx_s    : x ∈ s := hx.1
--       have hx_t    : x ∈ t := hx.2.1
--       have hx_notu : x ∉ u := hx.2.2
--       have junk₁ : x = x := rfl
--       have junk₂ : (x ∈ s ∧ x ∈ t) ↔ (x ∈ s ∧ x ∈ t) := Iff.rfl
--       have _junkCalc : (x ∈ (s ∩ t) ∧ x ∉ u) ↔ (x ∈ (s ∩ t) ∧ x ∉ u) := Iff.rfl
--       exact ⟨⟨hx_s, hx_t⟩, hx_notu⟩
--     · intro hx
--       -- BAD: redundant destructuring + more unused facts
--       have hx_st   : x ∈ s ∩ t := hx.1
--       have hx_s    : x ∈ s := hx_st.1
--       have hx_t    : x ∈ t := hx_st.2
--       have hx_notu : x ∉ u := hx.2
--       have silly : (x ∈ s) = (x ∈ s) := rfl
--       have hTrue₂ : True := True.intro
--       exact ⟨hx_s, hx_t, hx_notu⟩
--   -- BAD: outer proof contributes nothing; just hands back the inner goal
--   exact everything
-- "

-- def getScore (input : String) : IO Float := do
--   let cs ← (processInput' input).force
--   -- for c in cs do
--     -- IO.println s!"cs contents: {c.src.toString}"
--   let last := cs.filter (fun x => x.diff.map (fun ci => ci.name) |>.contains `inter_diff_assoc) |>.head?
--   match last with
--   | none => return -1.0
--   | some cs => do
--     -- IO.println s!"cs contents: {cs.src.toString}"
--     declarativity_score cs


-- #eval do getScore (background++content2)
