/-- One direction of the **Borel-Cantelli lemma**
(sometimes called the "*first* Borel-Cantelli lemma"):
if `(s i)` is a countable family of sets such that `∑' i, μ (s i)` is finite,
then the limit superior of the `s i` along the cofinite filter is a null set.

Note: for the *second* Borel-Cantelli lemma (applying to independent sets in a probability space),
see `ProbabilityTheory.measure_limsup_eq_one`. -/
theorem measure_limsup_cofinite_eq_zero {s : ι → Set α} (hs : ∑' i, μ (s i) ≠ ∞) :
    μ (limsup s cofinite) = 0 := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    inst✝ : Countable ι
    μ : F
    s : ι → Set α
    hs : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ Eq (μ (Filter.limsup s Filter.cofinite)) 0
  -/
  refine bot_unique <| ge_of_tendsto' (ENNReal.tendsto_tsum_compl_atTop_zero hs) fun t ↦ ?_
  calc
    μ (limsup s cofinite) ≤ μ (⋃ i : {i // i ∉ t}, s i) := by
      gcongr
      rw [hasBasis_cofinite.limsup_eq_iInf_iSup, iUnion_subtype]
      exact iInter₂_subset _ t.finite_toSet
    _ ≤ ∑' i : {i // i ∉ t}, μ (s i) := measure_iUnion_le _


/-- One direction of the **Borel-Cantelli lemma**
(sometimes called the "*first* Borel-Cantelli lemma"):
if `(s i)` is a sequence of sets such that `∑' i, μ (s i)` is finite,
then the limit superior of the `s i` along the `atTop` filter is a null set.

Note: for the *second* Borel-Cantelli lemma (applying to independent sets in a probability space),
see `ProbabilityTheory.measure_limsup_eq_one`. -/
theorem measure_limsup_atTop_eq_zero {s : ℕ → Set α} (hs : ∑' i, μ (s i) ≠ ∞) :
    μ (limsup s atTop) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    hs : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ Eq (μ (Filter.limsup s Filter.atTop)) 0
  -/
  rw [← Nat.cofinite_eq_atTop, measure_limsup_cofinite_eq_zero hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-01")]
alias measure_limsup_eq_zero := measure_limsup_atTop_eq_zero


/-- One direction of the **Borel-Cantelli lemma**
(sometimes called the "*first* Borel-Cantelli lemma"):
if `(s i)` is a countable family of sets such that `∑' i, μ (s i)` is finite,
then a.e. all points belong to finitely sets of the family. -/
theorem ae_finite_setOf_mem {s : ι → Set α} (h : ∑' i, μ (s i) ≠ ∞) :
    ∀ᵐ x ∂μ, {i | x ∈ s i}.Finite := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    inst✝ : Countable ι
    μ : F
    s : ι → Set α
    h : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ Filter.Eventually (fun x => (setOf fun i => Membership.mem (s i) x).Finite)  …
  -/
  rw [ae_iff, ← measure_limsup_cofinite_eq_zero h]
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    inst✝ : Countable ι
    μ : F
    s : ι → Set α
    h : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ Eq (μ (setOf fun a => Not (setOf fun i => Membership.mem (s i) a).Finite)) ( …
  -/
  congr 1 with x
  /-
    case h.e_6.h.h
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    inst✝ : Countable ι
    μ : F
    s : ι → Set α
    h : Ne (tsum fun i => μ (s i)) Top.top
    x : α
    ⊢ Iff (Membership.mem (setOf fun a => Not (setOf fun i => Membership.mem (s i) …
  -/
  simp [mem_limsup_iff_frequently_mem, Filter.Frequently]
  /-
    🎉 no goals
  -/


/-- A version of the **Borel-Cantelli lemma**: if `pᵢ` is a sequence of predicates such that
`∑' i, μ {x | pᵢ x}` is finite, then the measure of `x` such that `pᵢ x` holds frequently as `i → ∞`
(or equivalently, `pᵢ x` holds for infinitely many `i`) is equal to zero. -/
theorem measure_setOf_frequently_eq_zero {p : ℕ → α → Prop} (hp : ∑' i, μ { x | p i x } ≠ ∞) :
    μ { x | ∃ᶠ n in atTop, p n x } = 0 := by
  simpa only [limsup_eq_iInf_iSup_of_nat, frequently_atTop, ← bex_def, setOf_forall,
    setOf_exists] using measure_limsup_atTop_eq_zero hp


/-- A version of the **Borel-Cantelli lemma**: if `sᵢ` is a sequence of sets such that
`∑' i, μ sᵢ` is finite, then for almost all `x`, `x` does not belong to `sᵢ` for large `i`. -/
theorem ae_eventually_not_mem {s : ℕ → Set α} (hs : (∑' i, μ (s i)) ≠ ∞) :
    ∀ᵐ x ∂μ, ∀ᶠ n in atTop, x ∉ s n :=
  measure_setOf_frequently_eq_zero hs


theorem measure_liminf_cofinite_eq_zero [Infinite ι]  {s : ι → Set α} (h : ∑' i, μ (s i) ≠ ∞) :
    μ (liminf s cofinite) = 0 := by
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    inst✝¹ : Countable ι
    μ : F
    inst✝ : Infinite ι
    s : ι → Set α
    h : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ Eq (μ (Filter.liminf s Filter.cofinite)) 0
  -/
  rw [← le_zero_iff, ← measure_limsup_cofinite_eq_zero h]
  /-
    α : Type u_1
    ι : Type u_2
    F : Type u_3
    inst✝³ : FunLike F (Set α) ENNReal
    inst✝² : MeasureTheory.OuterMeasureClass F α
    inst✝¹ : Countable ι
    μ : F
    inst✝ : Infinite ι
    s : ι → Set α
    h : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ LE.le (μ (Filter.liminf s Filter.cofinite)) (μ (Filter.limsup s Filter.cofin …
  -/
  exact measure_mono liminf_le_limsup
  /-
    🎉 no goals
  -/


theorem measure_liminf_atTop_eq_zero {s : ℕ → Set α} (h : (∑' i, μ (s i)) ≠ ∞) :
    μ (liminf s atTop) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    h : Ne (tsum fun i => μ (s i)) Top.top
    ⊢ Eq (μ (Filter.liminf s Filter.atTop)) 0
  -/
  rw [← Nat.cofinite_eq_atTop, measure_liminf_cofinite_eq_zero h]
  /-
    🎉 no goals
  -/

-- TODO: the next 2 lemmas are true for any filter with countable intersections, not only `ae`.
-- Need to specify `α := Set α` below because of diamond; see https://github.com/leanprover-community/mathlib4/pull/19041

theorem limsup_ae_eq_of_forall_ae_eq (s : ℕ → Set α) {t : Set α}
    (h : ∀ n, s n =ᵐ[μ] t) : limsup (α := Set α) s atTop =ᵐ[μ] t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    t : Set α
    h : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (s n) t
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.limsup s Filter.atTop) t
  -/
  simp only [eventuallyEq_set, ← eventually_countable_forall] at h
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    t : Set α
    h : Filter.Eventually (fun x => ∀ (i : Nat), Iff (Membership.mem (s i) x) (Mem …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.limsup s Filter.atTop) t
  -/
  refine eventuallyEq_set.2 <| h.mono fun x hx ↦ ?_
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    t : Set α
    h : Filter.Eventually (fun x => ∀ (i : Nat), Iff (Membership.mem (s i) x) (Mem …
    x : α
    hx : ∀ (i : Nat), Iff (Membership.mem (s i) x) (Membership.mem t x)
    ⊢ Iff (Membership.mem (Filter.limsup s Filter.atTop) x) (Membership.mem t x)
  -/
  simp [mem_limsup_iff_frequently_mem, hx]
  /-
    🎉 no goals
  -/

-- Need to specify `α := Set α` above because of diamond; see https://github.com/leanprover-community/mathlib4/pull/19041

theorem liminf_ae_eq_of_forall_ae_eq (s : ℕ → Set α) {t : Set α}
    (h : ∀ n, s n =ᵐ[μ] t) : liminf (α := Set α) s atTop =ᵐ[μ] t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    t : Set α
    h : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyEq (s n) t
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.liminf s Filter.atTop) t
  -/
  simp only [eventuallyEq_set, ← eventually_countable_forall] at h
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    t : Set α
    h : Filter.Eventually (fun x => ∀ (i : Nat), Iff (Membership.mem (s i) x) (Mem …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Filter.liminf s Filter.atTop) t
  -/
  refine eventuallyEq_set.2 <| h.mono fun x hx ↦ ?_
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s : Nat → Set α
    t : Set α
    h : Filter.Eventually (fun x => ∀ (i : Nat), Iff (Membership.mem (s i) x) (Mem …
    x : α
    hx : ∀ (i : Nat), Iff (Membership.mem (s i) x) (Membership.mem t x)
    ⊢ Iff (Membership.mem (Filter.liminf s Filter.atTop) x) (Membership.mem t x)
  -/
  simp only [mem_liminf_iff_eventually_mem, hx, eventually_const]
  /-
    🎉 no goals
  -/


