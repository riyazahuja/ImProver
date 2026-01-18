-- import metrics.declarativity2.declarativity2
import metrics.declarativity2.declarativity2_v1
-- import metrics.declarativity2.declarativity2_v0
-- import metrics.declarativity2.declarativity2_oldest
import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic
import TrainingData.TreeParser
open Lean Core Elab IO Meta Term Command Tactic System






set_option linter.unusedVariables false in
def getScore2 (mod : Name) (decl : Name) (new_proof : String) : IO (Float× Float × Bool) := do
  let startTime ← IO.monoMsNow

  -- Don't set search path to avoid environment extension issues
  searchPathRef.set compile_time_search_path%
  let afterSearchPath ← IO.monoMsNow
  -- IO.println s!"[TIMING] Set search path: {afterSearchPath - startTime}ms"

  let fileName := (← findLean mod).toString
  -- IO.println s!"Found file: {fileName}"
  let afterFindLean ← IO.monoMsNow
  -- IO.println s!"[TIMING] Find lean file: {afterFindLean - afterSearchPath}ms"

  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none {} fileName
  -- IO.println s!"{← moduleSource mod}"

  -- let forced ← steps.force
  -- for c in forced do
  --   IO.println s!"Compilation command: {c.src.toString}\n\n----------------------\n"

  let afterProcessInput ← IO.monoMsNow
  -- IO.println s!"[TIMING] Process input: {afterProcessInput - afterFindLean}ms"

  let targets ← (steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)).force
  let afterTargets ← IO.monoMsNow
  -- IO.println s!"[TIMING] Extract targets: {afterTargets - afterProcessInput}ms"

  let target := targets.find? fun (_, i) => i.name == decl

  match target with
  | none =>
    let endTime ← IO.monoMsNow
    -- IO.println s!"[TIMING] Total (no target found): {endTime - startTime}ms"
    IO.println s!"[ERROR] Target decls: {targets.map (fun (_, i) => i.name)}"
    return ((-1.0),(-1.0), false)
  | some (target_cmd, _) => do
    let beforeOgScore ← IO.monoMsNow
    -- IO.println s!"[TIMING] Find target: {beforeOgScore - afterTargets}ms"

    let og_score ← declarativity2_score target_cmd
    let afterOgScore ← IO.monoMsNow
    -- IO.println s!"[TIMING] Original declarativity score: {afterOgScore - beforeOgScore}ms"

    let background_content :=  (Substring.mk target_cmd.src.str 0 target_cmd.src.startPos) |>toString
    let afterBackgroundContent ← IO.monoMsNow
    -- IO.println s!"[TIMING] Extract background content: {afterBackgroundContent - afterOgScore}ms"

    let elaborated_steps := Lean.Elab.IO.compilationSteps
      (Parser.mkInputContext (background_content ++ new_proof) fileName)
      target_cmd.parserStateBefore
      (target_cmd.commandStateBefore.withOptions {})
    let afterElaboratedSteps ← IO.monoMsNow
    -- IO.println s!"[TIMING] Elaborate new steps: {afterElaboratedSteps - afterBackgroundContent}ms"

    let new_target ← elaborated_steps.head?
    let afterNewTarget ← IO.monoMsNow
    -- IO.println s!"[TIMING] Get new target: {afterNewTarget - afterElaboratedSteps}ms"

    match new_target with
    | none =>
      let endTime ← IO.monoMsNow
      -- IO.println s!"[TIMING] Total (no new target): {endTime - startTime}ms"
      return (og_score, (-1.0), false)
    | some new_target => do
      let beforeNewScore ← IO.monoMsNow
      let correct := not <| new_target.msgs.any (fun m => m.severity == .error)
      let afterCorrectCheck ← IO.monoMsNow
      -- IO.println s!"[TIMING] Check correctness: {afterCorrectCheck - beforeNewScore}ms"

      let new_score ← declarativity2_score new_target
      let endTime ← IO.monoMsNow
      -- IO.println s!"[TIMING] New declarativity score: {endTime - afterCorrectCheck}ms"
      -- IO.println s!"[TIMING] Total execution time: {endTime - startTime}ms"

      return (og_score, new_score, correct)




def sset := "lemma cc_subset_oo {x : X} {r₁ R₁ r₂ R₂ : ℝ} (hr : r₂ < r₁) (hR : R₁ < R₂) :
    cc x r₁ R₁ ⊆ oo x r₂ R₂ := by
  -- Introduce a helper lemma to handle the core subset proof
  have h₁ : ∀ a : X, a ∈ cc x r₁ R₁ → a ∈ oo x r₂ R₂ := by
    intro a ha
    -- Decompose the hypothesis into its components
    cases' ha with hr₁ hR₁
    -- Use the helper lemmas to prove the inclusion
    exact ⟨lt_of_lt_of_le hr (hr₁) , lt_of_le_of_lt hR₁ (hR)⟩
  -- Apply the helper lemma to complete the proof
  exact h₁"
#eval do getScore2 `Carleson.ToMathlib.Annulus `Set.Annulus.cc_subset_oo sset







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
-- expected: (0.000000, 2.000000, true)
-- because: h1 and h2 are effective
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
-- expected: (0.000000, 0.000000, true)
-- because: h1 and h2 are trivial, and moreover, h3 (the only place where h1 and h2 are used) is never used anywhere.
#eval do getScore2 `Seymour.Matroid.Operations.Sum2.Regularity `TwoSumAssumptions.decomposition_isRegular_both new_proof2



def c5 :="lemma C5_1_2_optimized_le' {a : ℕ} {q : ℝ≥0} (ha : 4 ≤ a) :
    C5_1_2_optimized a q ≤ C2_0_4_base a * 2 ^ (a ^ 3) / (q - 1) ^ 4 := by
  -- Main lemma: Bounding C5_1_2_optimized a q by C2_0_4_base a * 2 ^ (a ^ 3) / (q - 1) ^ 4
  have h₁ : ∀ a : ℕ, 4 ≤ a → C5_1_2_optimized a q ≤ C2_0_4_base a * 2 ^ (a ^ 3) / (q - 1) ^ 4 := by
    intro a ha
    -- Lemma to handle the core inequality
    have h₂ : C5_1_2_optimized a q = C2_0_4_base a * (2 ^ (a + 5/2 : ℝ) * 13009) / (q - 1) ^ 4 := by
      simp [C5_1_2_optimized, mul_assoc]
    rw [h₂]
    -- Apply the main bounding lemma
    gcongr
    simp only [← NNReal.coe_le_coe, NNReal.coe_mul, coe_rpow, NNReal.coe_ofNat]
    calc
      (2 : ℝ) ^ (a + 5 / 2 : ℝ) * 13009
      _ ≤ 2 ^ (a + 3 : ℝ) * 2 ^ 14 := by gcongr <;> norm_num
      _ = 2 ^ (a + 17) := by
        have : (a + 3 : ℝ) = (a + 3 : ℕ) := by norm_cast
        rw [this, Real.rpow_natCast, ← pow_add]
      _ ≤ 2 ^ (a ^ 3) := by
        apply pow_le_pow_right₀ one_le_two
        have : (4 : ℤ) ≤ a := mod_cast ha
        zify
        calc (a : ℤ) + 17
        _ ≤ a + 4 * (4 * 4 - 1) := by gcongr; norm_num
        _ ≤ a + a * (a * a - 1) := by gcongr
        _ = a ^ 3 := by ring
    -- End of lemma application
  -- Main goal achieved
  exact h₁ a ha

-- Helper lemma: Core inequality involving the main terms"

-- expected: (0.000000, 0.000000, true) - i think?
-- because: h1 is a duplicate of the original goal, h2 is trivial, and the two calcs are contained in an ineffective goal.
#eval do getScore2 `Carleson.Discrete.ForestUnion `C5_1_2_optimized_le' c5



def volume := "lemma Complex.volume_complex_smul (z : ℂ) (s : Set ℂ) : volume (z • s) = ‖z‖₊ ^ 2 * volume s := by
  -- Consider the case where z is zero
  have h₁ : ∀ z : ℂ, z = 0 → volume (z • s) = ‖z‖₊ ^ 2 * volume s := by
    intro z hz
    -- If z is zero, the set z · s collapses to a single point at the origin
    simp [(finite_zero.subset s.zero_smul_set_subset).measure_zero, hz]
  -- Consider the case where z is not zero
  have h₂ : ∀ z : ℂ, z ≠ 0 → volume (z • s) = ‖z‖₊ ^ 2 * volume s := by
    intro z hz
    -- If z is not zero, we can view z as an invertible element in the complex plane
    lift z to ℂˣ using hz.isUnit
    -- Apply the property that scaling a set by a complex number z scales its measure by ‖z‖₊ ^ 2
    rw [← ENNReal.coe_pow, ← distribHaarChar_complex, distribHaarChar_mul, Units.smul_def]
  -- Combine both cases
  cases' eq_or_ne z 0 with rfl rfl <;> simp_all [h₁, h₂]
"
-- expected: (0.000000, 1.000000, true)
-- because: h1 is trivial, h2 is effective
#eval do getScore2 `FLT.HaarMeasure.DistribHaarChar.RealComplex `Complex.volume_complex_smul volume



def invOneSub := "theorem one_sub_pow_add_mul_invOneSubPow_val_eq_one_sub_pow (e : ℕ) :
    (1 - X) ^ (d + e) * (invOneSubPow S e).val = (1 - X) ^ d := by
  -- Introduce an intermediate lemma to handle the core logic of simplifying the expression involving (1 - X) and its powers.
  have h₁ : ∀ e : ℕ, (1 - X) ^ (d + e) * (invOneSubPow S e).val = (1 - X) ^ d → (1 - X) ^ (d + e) * (invOneSubPow S e).val = (1 - X) ^ d := by
    intro e h
    -- Apply the lemma to simplify the expression step-by-step, ensuring that each transformation is clear and justified.
    rw [h]
  -- Use the intermediate lemma to conclude the proof, applying it to the specific case.
  have h₂ : (1 - X) ^ (d + e) * (invOneSubPow S e).val = (1 - X) ^ d := by
    simp [pow_add, mul_assoc, ← invOneSubPow_inv_eq_one_sub_pow S e]
  -- Apply the helper lemma to complete the proof, ensuring that the final steps are logically sound and maintain the flow of the argument.
  apply h₁
  -- Final simplification to confirm the equality, ensuring that all steps are clearly laid out and easy to follow.
  <;> simp [pow_add, mul_assoc, ← invOneSubPow_inv_eq_one_sub_pow S e]"
-- expected: (0.000000, 0.000000, true)
-- because: h1 is a duplicate of the original goal state, h2 is trivial
#eval do getScore2 `Mathlib.RingTheory.PowerSeries.WellKnown `PowerSeries.one_sub_pow_add_mul_invOneSubPow_val_eq_one_sub_pow invOneSub


def mem := "theorem typedMem_singleton_iff' {α β : Λ} (hβ : (β : TypeIndex) < α) (x y : TSet β) :
    y ∈[hβ] singleton hβ x ↔ y = x := by
  -- fix the ambient level once
  letI : Level := ⟨α⟩
  -- package the core equivalence as a named subproof to make the case-splitting
  -- on whether `β` coincides with the base level explicit and modular.
  have core : y ∈[hβ] singleton hβ x ↔ y = x := by
    by_cases h : (β : TypeIndex) = α
    · -- case: β = α
      -- provide the required `LeLevel` / `LtLevel` instances for this branch
      letI : LeLevel α := ⟨le_rfl⟩
      letI : LtLevel β := ⟨hβ⟩
      -- delegate to the general lemma which handles both subcases
      exact typedMem_singleton_iff hβ x y
    · -- case: β ≠ α
      letI : LeLevel α := ⟨le_rfl⟩
      letI : LtLevel β := ⟨hβ⟩
      exact typedMem_singleton_iff hβ x y
  exact core"
-- expected: (0.000000, 0.000000, true)
-- because: core is a duplicate of the original goal state
#eval do getScore2 `ConNF.Model.TTT `ConNF.typedMem_singleton_iff' mem


def KB_again := "lemma KB5_weakerThan_S5 : (Hilbert.KB5 ℕ) ≤ₛ (Hilbert.S5 ℕ) := by
  -- reduce the goal to showing a subset relation between frame classes
  apply Kripke.weakerThan_of_subset_FrameClass SymmetricEuclideanFrameClass ReflexiveEuclideanFrameClass;
  -- prove the subset: every reflexive+euclidean frame is symmetric+euclidean
  have subset : ReflexiveEuclideanFrameClass ⊆ SymmetricEuclideanFrameClass := by
    intro F hF
    rcases hF with ⟨h_refl, h_eucl⟩
    -- separate subproof establishing symmetry from reflexivity and euclideanity
    have symm : Symmetric F.Rel := by
      intro x y hxy
      -- using reflexivity at x and euclideanity to get y R x
      exact h_eucl (h_refl x) hxy
    -- assemble the required membership of SymmetricEuclideanFrameClass
    exact ⟨symm, h_eucl⟩
  exact subset"
-- expected: (0.000000, 1.000000, true)
-- because: symm is trivial and therefore ineffective, subset is effective.
#eval do getScore2 `Foundation.Modal.Hilbert.WeakerThan.KB5_S5 `LO.Modal.Hilbert.KB5_weakerThan_S5 KB_again



def XX := "lemma StandardRepresentation.Is3sumOf.interXX (hM : M.Is3sumOf M₁ M₂) :
    ∃ x₁ x₂ x₃ : α, M₁.X ∩ M₂.X = {x₁, x₂, x₃} := by
  -- extract the witnesses and the equality from the hypothesis
  obtain ⟨x₁, x₂, x₃, _, _, _, hXX, _⟩ := hM

  -- prove the set equality by extensionality, producing two meaningful subproofs
  have h_eq : M₁.X ∩ M₂.X = {x₁, x₂, x₃} := by
    apply Set.ext
    intro a
    constructor
    -- forward direction: element of intersection belongs to the explicit three-element set
    · intro ha
      rw [hXX] at ha
      exact ha
    -- backward direction: element of the explicit three-element set belongs to the intersection
    · intro ha
      rw [←hXX] at ha
      exact ha

  -- assemble the existential using the proved equality
  exact ⟨x₁, x₂, x₃, h_eq⟩"
-- expected: (0.000000, 1.000000, true)
-- because: h_eq is effective as it is nontrivial, used in the last exact statement, and not a duplicate of any prev goals.
#eval do getScore2 `Seymour.Matroid.Operations.MatrixSums.Sum3 `StandardRepresentation.Is3sumOf.interXX XX















-- Other new test cases


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
-- (0.000000, 0.000000, true)
#eval do getScore2 `FLT.Mathlib.LinearAlgebra.Determinant `LinearMap.det_mul new_proof3


def unused := "lemma index_smul (a : G) (S : AddSubgroup A) : (a • S).index = S.index := by
  -- Introduce the first sub-proof to handle the general case of the bijection
  have h₁ : ∀ a : G, (a • S).index = S.index → (a • S).index = S.index := by
    intro a h_a
    -- Use the helper lemma to conclude the proof
    rw [h_a]
  -- Apply the helper lemma to complete the proof
  exact index_map_of_bijective _ (MulAction.bijective _)"
-- (0.000000, 0.000000, true)
#eval do getScore2 `FLT.Mathlib.GroupTheory.Index `AddSubgroup.index_smul unused



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
-- (0.000000, 0.000000, true)
#eval getScore2 `Seymour.Matroid.Notions.Circuit `Matroid.Circuit.nonempty matroid


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
-- (0.000000, 0.000000, true)
#eval getScore2 `ConNF.Model.Hailperin `ConNF.TSet.op_eq_singleton_iff singleton




def c6 := "lemma C6_forest' (hkn : k ≤ n) :
    ℭ₆ (X := X) k n j = ⋃ l ∈ Iio (4 * n + 12), ⋃ u ∈ 𝔘₄ k n j l, 𝔗₂ k n j u := by
  -- Main lemma: Rewrite ℭ₆ using the C6_forest result
  have h₁ : ∀ hkn : k ≤ n, ℭ₆ (X := X) k n j = ⋃ l ∈ Iio (4 * n + 12), ⋃ u ∈ 𝔘₄ k n j l, 𝔗₂ k n j u := by
    intro hkn
    rw [C6_forest, ← iUnion_𝔘₄ hkn]
    simp
  -- Apply the main lemma to the specific case
  exact h₁ hkn"
-- (0.000000, 0.000000, true)
#eval getScore2 `Carleson.Discrete.ForestUnion `C6_forest' c6


def exists_inter := "theorem exists_inter (x y : TSet α) :
    ∃ w : TSet α, ∀ z : TSet β, z ∈[hβ] w ↔ z ∈[hβ] x ∧ z ∈[hβ] y := by
  -- reduce the problem to producing a symmetric support for the intersection predicate
  refine exists_of_symmetric {z | z ∈[hβ] x ∧ z ∈[hβ] y} hβ ?_
  -- obtain supports witnessing symmetry for x and y separately
  obtain ⟨S, hS⟩ := symmetric x hβ
  obtain ⟨T, hT⟩ := symmetric y hβ
  -- use the sum of the two supports for the intersection
  use S + T
  intro ρ hρ
  -- first subproof: specialize and simplify the symmetry witness for x wrt S ≤ S+T
  have hS_spec := by
    specialize hS ρ (smul_eq_of_le Support.le_add_right hρ)
    simp [Set.ext_iff, Set.mem_smul_set_iff_inv_smul_mem] at hS
    exact hS
  -- second subproof: specialize and simplify the symmetry witness for y wrt T ≤ S+T
  have hT_spec := by
    specialize hT ρ (smul_eq_of_le Support.le_add_left hρ)
    simp [Set.ext_iff, Set.mem_smul_set_iff_inv_smul_mem] at hT
    exact hT
  -- combine the two simplified equalities to finish the goal
  simp [Set.ext_iff, Set.mem_smul_set_iff_inv_smul_mem] at hS_spec hT_spec ⊢
  aesop"
-- (0.000000, 2.000000, true)
#eval getScore2 `ConNF.Model.Hailperin `ConNF.TSet.exists_inter exists_inter
