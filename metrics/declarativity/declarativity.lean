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

def declarativity_score (cs : CompilationStep) : IO Float := do
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





-- def background:= "/- This file contains helper lemmas. Either they should be replaced by a mathlib version if there is\n   one or they might be candidates to go there, possibly in a generalized form. -/\n\nimport Carleson.ToMathlib.Misc\nimport Mathlib.MeasureTheory.Integral.IntervalIntegral\n\nopen MeasureTheory\n\ntheorem Real.volume_uIoc {a b : ℝ} : volume (Set.uIoc a b) = ENNReal.ofReal |b - a| := by\n  /- Cf. proof of Real.volume_interval-/\n  rw [Set.uIoc, volume_Ioc, max_sub_min_eq_abs]\n\nlemma intervalIntegral.integral_conj' {μ : Measure ℝ} {𝕜 : Type} [RCLike 𝕜] {f : ℝ → 𝕜} {a b : ℝ}:\n    ∫ x in a..b, (starRingEnd 𝕜) (f x) ∂μ = (starRingEnd 𝕜) (∫ x in a..b, f x ∂μ) := by\n  rw [intervalIntegral_eq_integral_uIoc, integral_conj, intervalIntegral_eq_integral_uIoc,\n      RCLike.real_smul_eq_coe_mul, RCLike.real_smul_eq_coe_mul, map_mul, RCLike.conj_ofReal]\n\nlemma intervalIntegrable_of_bdd {a b : ℝ} {δ : ℝ} {g : ℝ → ℂ} (measurable_g : Measurable g) (bddg : ∀ x, ‖g x‖ ≤ δ) : IntervalIntegrable g volume a b := by\n  apply @IntervalIntegrable.mono_fun' _ _ _ _ _ _ (fun _ ↦ δ)\n  · exact intervalIntegrable_const\n  · exact measurable_g.aestronglyMeasurable\n  · rw [Filter.EventuallyLE, ae_restrict_iff_subtype measurableSet_uIoc]\n    apply Filter.Eventually.of_forall\n    rw [Subtype.forall]\n    exact fun x _ ↦ bddg x\n\nlemma IntervalIntegrable.bdd_mul {F : Type} [NormedDivisionRing F] {f g : ℝ → F} {a b : ℝ} {μ : Measure ℝ}\n    (hg : IntervalIntegrable g μ a b) (hm : AEStronglyMeasurable f μ) (hfbdd : ∃ C, ∀ x, ‖f x‖ ≤ C) : IntervalIntegrable (fun x ↦ f x * g x) μ a b := by\n  rw [intervalIntegrable_iff, IntegrableOn]\n  apply Integrable.bdd_mul _ hm.restrict hfbdd\n  rwa [← IntegrableOn, ← intervalIntegrable_iff]\n\nlemma IntervalIntegrable.mul_bdd {F : Type} [NormedField F] {f g : ℝ → F} {a b : ℝ} {μ : Measure ℝ}\n    (hf : IntervalIntegrable f μ a b) (hm : AEStronglyMeasurable g μ) (hgbdd : ∃ C, ∀ x, ‖g x‖ ≤ C) : IntervalIntegrable (fun x ↦ f x * g x) μ a b := by\n  conv => pattern (fun x ↦ f x * g x); ext x; rw [mul_comm]\n  exact hf.bdd_mul hm hgbdd\n\nlemma IntegrableOn.sub {α : Type} {β : Type} {m : MeasurableSpace α}\n    {μ : Measure α} [NormedAddCommGroup β] {s : Set α} {f g : α → β} (hf : IntegrableOn f s μ) (hg : IntegrableOn g s μ) : IntegrableOn (f - g) s μ := by\n  apply Integrable.sub <;> rwa [← IntegrableOn]\n\n\nlemma ConditionallyCompleteLattice.le_biSup {α : Type} [ConditionallyCompleteLinearOrder α] {ι : Type} [Nonempty ι]\n    {f : ι → α} {s : Set ι} {a : α} (hfs : BddAbove (f '' s)) (ha : ∃ i ∈ s, f i = a) :\n    a ≤ ⨆ i ∈ s, f i := by\n  apply ConditionallyCompleteLattice.le_csSup\n  · --TODO: improve this\n    rw [bddAbove_def] at *\n    rcases hfs with ⟨x, hx⟩\n    use (max x (sSup ∅))\n    intro y hy\n    simp only [Set.mem_range] at hy\n    rcases hy with ⟨z, hz⟩\n    rw [iSup] at hz\n    by_cases h : z ∈ s\n    · have : (@Set.range α (z ∈ s) fun _ ↦ f z) = {f z} := by\n        rw [Set.eq_singleton_iff_unique_mem]\n        exact ⟨Set.mem_range_self h, fun x hx ↦ hx.2.symm⟩\n      rw [this, csSup_singleton _] at hz\n      have : f z ≤ x := by\n        simp only [Set.mem_image, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂] at hx\n        exact hx z h\n      rw [hz] at this\n      exact le_max_of_le_left this\n    have : (@Set.range α (z ∈ s) fun _ ↦ f z) = ∅ := by simpa\n    rw [this] at hz\n    exact hz ▸ le_max_right x y\n  rw [Set.mem_range]\n  rcases ha with ⟨i, hi, fia⟩\n  use i\n  rw [iSup]\n  convert csSup_singleton _\n  rw [Set.eq_singleton_iff_unique_mem]\n  refine ⟨⟨hi, fia⟩, fun x hx ↦ ?_⟩\n  simp only [Set.mem_range, exists_prop] at hx\n  rwa [hx.2] at fia\n\n\n/-Adapted from mathlib Function.Periodic.exists_mem_Ico₀-/\ntheorem Function.Periodic.exists_mem_Ico₀' {α : Type} {β : Type} {f : α → β} {c : α}\n  [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c) (hc : 0 < c) (x : α) : ∃ (n : ℤ), (x - n • c) ∈ Set.Ico 0 c ∧ f x = f (x - n • c) :=\n  let ⟨n, H, _⟩ := existsUnique_zsmul_near_of_pos' hc x\n  ⟨n, H, (h.sub_zsmul_eq n).symm⟩\n\n/-Adapted from mathlib Function.Periodic.exists_mem_Ico₀-/\n"
def background := "import ImProver.metrics.tagger
import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Data.Nat.Prime.Basic
"
def content := "
theorem foo {a b c d : ℝ} :
    max a b + max c d = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by
  have lemma_add_distrib : ∀ (x y z : ℝ), z + max x y = max (z + x) (z + y) := by
    intro x y z
    rcases le_total x y with h | h
    · rw [max_eq_right h, max_eq_right (add_le_add_left h z)]
    · rw [max_eq_left h, max_eq_left (add_le_add_left h z)]

  calc max a b + max c d
    _ = max c d + max a b := by rw [add_comm]
    _ = max (max c d + a) (max c d + b) := by rw [lemma_add_distrib]
    _ = max (a + max c d) (b + max c d) := by rw [add_comm (max c d) a, add_comm (max c d) b]
    _ = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by rw [lemma_add_distrib, lemma_add_distrib]

"

def content2 := "

theorem foo {a b c d : ℝ} :
    max a b + max c d = max (max (a + c) (a + d)) (max (b + c) (b + d)) := by
  rcases le_total a b with h_ab | h_ba
  · rcases le_total c d with h_cd | h_dc
    -- Case 1: a ≤ b and c ≤ d
    · calc max a b + max c d
        _ = b + d := by rw [max_eq_right h_ab, max_eq_right h_cd]
        _ = max (b+c) (b+d) := by rw [max_eq_right (add_le_add_left h_cd b)]
        _ = max (max (a+d) (b+c)) (b+d) := by
          rw [max_eq_right (add_le_add_left h_cd b)]
          apply symm
          apply @max_eq_right _ _ (max (a+d) (b+c)) (b+d)
          rw [max_le_iff]
          constructor
          . linarith
          . exact add_le_add_left h_cd b
        _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_cd]
    -- Case 2: a ≤ b and d ≤ c
    · calc max a b + max c d
        _ = b + c := by rw [max_eq_right h_ab, max_eq_left h_dc]
        _ = max (b+c) (b+d) := by rw [max_eq_left (add_le_add_left h_dc b)]
        _ = max (max (a+d) (b+c)) (b+d) := by
          rw [max_eq_left (add_le_add_left h_dc b)]
          apply symm
          rw [max_assoc (a+d) (b+c) (b+d), max_comm (b+c) (b+d), ← max_assoc (a+d) (b+d) (b+c)]
          apply @max_eq_right _ _ (max (a+d) (b+d)) (b+c)
          rw [max_le_iff]
          constructor
          . linarith
          . exact add_le_add_left h_dc b
        _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ab, h_dc]; linarith
  · rcases le_total c d with h_cd | h_dc
    -- Case 3: b ≤ a and c ≤ d
    · calc max a b + max c d
        _ = a + d := by rw [max_eq_left h_ba, max_eq_right h_cd]
        _ = max (a+c) (a+d) := by rw [max_eq_right (add_le_add_left h_cd a)]
        _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ba, h_cd]
    -- Case 4: b ≤ a and d ≤ c
    · calc max a b + max c d
          _ = a + c := by rw [max_eq_left h_ba, max_eq_left h_dc]
          _ = max (a+c) (a+d) := by rw [max_eq_left (add_le_add_left h_dc a)]
          _ = max (max (a+c) (a+d)) (max (b+c) (b+d)) := by simp [h_ba, h_dc]


"

def getScore (input : String) : IO ( Float) := do
  let cs ← (processInput' input).force
  for c in cs do
    IO.println s!"cs contents: {c.src.toString}"
  let last := cs.getLast?
  match last with
  | none => return (-1.0)
  | some cs => do
    IO.println s!"cs contents: {cs.src.toString}"
    declarativity_score cs

#eval do getScore (background ++ content)


def getScore2 (mod : Name) (decl : Name) (new_proof : String) : IO (Float× Float × Bool) := do

  searchPathRef.set compile_time_search_path%
  let fileName := (← findLean mod).toString
  -- let scope_import := "import ImProver.get_prompts.where_with_end\n"
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  let iosteps ← steps.force
  for s in iosteps do
    IO.println s!"step: {s.src.toString}"
    IO.println s!"diff: {s.diff.map (fun d => d.name)}"
    IO.println s!"raw: {s.after.constants.map₁.toList.map (fun (n,_)=> n)}"
    IO.println "----"

  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let target := targets.find? fun (_, i) => i.name == decl
  IO.println s!"possible targets: {targets.map (fun (_, i) => i.name)}"

  match target with
  | none => return ((-1.0),(-1.0), false)
  | some (target_cmd, _) => do
    let og_score ← declarativity2_score target_cmd

    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString
    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})
    let new_target ← elaborated_steps.head?
    match new_target with
    | none => return (og_score, (-1.0), false)
    | some new_target => do
      let correct := not <| new_target.msgs.any (fun m => m.severity == .error)
      return (og_score, ← declarativity2_score new_target, correct)



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
#eval do getScore2 `Foundation.Modal.Hilbert.WeakerThan.KD5_KD45 `LO.Modal.Hilbert.KD5_weakerThan_KD45 new_proof


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
#eval do getScore2 `Seymour.Matroid.Operations.Sum2.Regularity `TwoSumAssumptions.decomposition_isRegular_both new_proof2




-- def new_proof3 := "@[simp] lemma det_mul (a : R) : (mul R R a).det = a := by
--   -- Lemma to handle the core determinant calculation
--   have h₁ : ∀ a : R, (mul R R a).det = a → (mul R R a).det = a := by
--     intro a h
--     -- Apply the helper lemma to complete the proof
--     rw [h]
--   -- Lemma to handle the algebraic manipulation
--   have h₂ : ∀ a : R, (mul R R a).det = a → (mul R R a).det = a := by
--     intro a h
--     -- Apply the helper lemma to complete the proof
--     rw [h]
--   -- Main lemma combining both steps
--   classical
--   rw [det_eq_det_toMatrix_of_finset (s := {1}) ⟨(Finsupp.LinearEquiv.finsuppUnique R R _).symm⟩, Matrix.det_unique]
--   change a * _ = a
--   simp
-- "
-- -- ((0.000000, 0.000000), (2.000000, 1.000000), true)
-- #eval do getScore2 `FLT.Mathlib.LinearAlgebra.Determinant `LinearMap.det_mul new_proof3


-- def new_proof4 := "lemma C6_forest' (hkn : k ≤ n) :
--     ℭ₆ (X := X) k n j = ⋃ l ∈ Iio (4 * n + 12), ⋃ u ∈ 𝔘₄ k n j l, 𝔗₂ k n j u := by
--   -- Main lemma: Rewrite ℭ₆ using the C6_forest result
--   have h₁ : ∀ hkn : k ≤ n, ℭ₆ (X := X) k n j = ⋃ l ∈ Iio (4 * n + 12), ⋃ u ∈ 𝔘₄ k n j l, 𝔗₂ k n j u := by
--     intro hkn
--     rw [C6_forest, ← iUnion_𝔘₄ hkn]
--     simp
--   -- Apply the main lemma to the specific case
--   exact h₁ hkn"
-- -- ((0.000000, 0.000000), (1.000000, 1.000000), true)
-- #eval do getScore2 `Carleson.Discrete.ForestUnion `C6_forest' new_proof4

-- def unused := "lemma index_smul (a : G) (S : AddSubgroup A) : (a • S).index = S.index := by
--   -- Introduce the first sub-proof to handle the general case of the bijection
--   have h₁ : ∀ a : G, (a • S).index = S.index → (a • S).index = S.index := by
--     intro a h_a
--     -- Use the helper lemma to conclude the proof
--     rw [h_a]
--   -- Apply the helper lemma to complete the proof
--   exact index_map_of_bijective _ (MulAction.bijective _)"
-- -- ((0.000000, 0.000000), (1.000000, 1.000000), true)
-- #eval do getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul unused


-- def repeats_and_unused := "lemma ofFieldOp_mul_ofFieldOp_eq_superCommute (φ φ' : 𝓕.FieldOp) :
--     ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ
--     + [ofFieldOp φ, ofFieldOp φ']ₛ := by
--   -- Main lemma handling the general case
--   have h₁ : ∀ φ φ' : 𝓕.FieldOp, ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ → ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ := by
--     intro φ φ' h
--     -- Apply the helper lemma to handle the specific case
--     rw [h]
--   -- Apply the main lemma to complete the proof
--   have h₂ : ∀ φ φ' : 𝓕.FieldOp, ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ → ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ := by
--     intro φ φ' h
--     rw [h]
--   -- Use the main lemma to conclude the proof
--   have h₃ : ∀ φ φ' : 𝓕.FieldOp, ofFieldOp φ * ofFieldOp φ' = 𝓢(𝓕 |>ₛ φ, 𝓕 |>ₛ φ') • ofFieldOp φ' * ofFieldOp φ + [ofFieldOp φ, ofFieldOp φ']ₛ := by
--     intro φ φ'
--     rw [← ofFieldOpList_singleton, ← ofFieldOpList_singleton]
--     rw [ofFieldOpList_mul_ofFieldOpList_eq_superCommute, ofFieldOpList_singleton]
--     simp
--   -- Apply the helper lemma to finalize the proof
--   rw [h₃]"
-- -- ((0.000000, 0.000000), (3.000000, 2.000000), true)
-- #eval do getScore2 `HepLean.PerturbationTheory.FieldOpAlgebra.SuperCommute `FieldSpecification.FieldOpAlgebra.ofFieldOp_mul_ofFieldOp_eq_superCommute repeats_and_unused
