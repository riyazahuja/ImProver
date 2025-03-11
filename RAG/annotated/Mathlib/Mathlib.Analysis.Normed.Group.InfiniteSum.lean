theorem cauchySeq_finset_iff_vanishing_norm {f : ι → E} :
    (CauchySeq fun s : Finset ι => ∑ i ∈ s, f i) ↔
      ∀ ε > (0 : ℝ), ∃ s : Finset ι, ∀ t, Disjoint t s → ‖∑ i ∈ t, f i‖ < ε := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    ⊢ Iff (CauchySeq fun s => s.sum fun i => f i) (∀ (ε : Real), GT.gt ε 0 → Exist …
  -/
  rw [cauchySeq_finset_iff_sum_vanishing, nhds_basis_ball.forall_iff]
    /-
      ι : Type u_1
      E : Type u_3
      inst✝ : SeminormedAddCommGroup E
      f : ι → E
      ⊢ Iff (∀ (i : Real), LT.lt 0 i → Exists fun s => ∀ (t : Finset ι), Disjoint t  …
    -/
  · simp only [ball_zero_eq, Set.mem_setOf_eq]
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      E : Type u_3
      inst✝ : SeminormedAddCommGroup E
      f : ι → E
      ⊢ ∀ ⦃s t : Set E⦄, HasSubset.Subset s t → (Exists fun s_1 => ∀ (t : Finset ι), …
    -/
  · rintro s t hst ⟨s', hs'⟩
    /-
      case intro
      ι : Type u_1
      E : Type u_3
      inst✝ : SeminormedAddCommGroup E
      f : ι → E
      s t : Set E
      hst : HasSubset.Subset s t
      s' : Finset ι
      hs' : ∀ (t : Finset ι), Disjoint t s' → Membership.mem s (t.sum fun b => f b)
      ⊢ Exists fun s => ∀ (t_1 : Finset ι), Disjoint t_1 s → Membership.mem t (t_1.s …
    -/
    exact ⟨s', fun t' ht' => hst <| hs' _ ht'⟩
    /-
      🎉 no goals
    -/


theorem summable_iff_vanishing_norm [CompleteSpace E] {f : ι → E} :
    Summable f ↔ ∀ ε > (0 : ℝ), ∃ s : Finset ι, ∀ t, Disjoint t s → ‖∑ i ∈ t, f i‖ < ε := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : ι → E
    ⊢ Iff (Summable f) (∀ (ε : Real), GT.gt ε 0 → Exists fun s => ∀ (t : Finset ι) …
  -/
  rw [summable_iff_cauchySeq_finset, cauchySeq_finset_iff_vanishing_norm]
  /-
    🎉 no goals
  -/


theorem cauchySeq_finset_of_norm_bounded_eventually {f : ι → E} {g : ι → ℝ} (hg : Summable g)
    (h : ∀ᶠ i in cofinite, ‖f i‖ ≤ g i) : CauchySeq fun s => ∑ i ∈ s, f i := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    g : ι → Real
    hg : Summable g
    h : Filter.Eventually (fun i => LE.le (Norm.norm (f i)) (g i)) Filter.cofinite
    ⊢ CauchySeq fun s => s.sum fun i => f i
  -/
  refine cauchySeq_finset_iff_vanishing_norm.2 fun ε hε => ?_
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    g : ι → Real
    hg : Summable g
    h : Filter.Eventually (fun i => LE.le (Norm.norm (f i)) (g i)) Filter.cofinite
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun s => ∀ (t : Finset ι), Disjoint t s → LT.lt (Norm.norm (t.sum fun …
  -/
  rcases summable_iff_vanishing_norm.1 hg ε hε with ⟨s, hs⟩
  classical
  refine ⟨s ∪ h.toFinset, fun t ht => ?_⟩
  have : ∀ i ∈ t, ‖f i‖ ≤ g i := by
    intro i hi
    simp only [disjoint_left, mem_union, not_or, h.mem_toFinset, Set.mem_compl_iff,
      Classical.not_not] at ht
    exact (ht hi).2
  calc
    ‖∑ i ∈ t, f i‖ ≤ ∑ i ∈ t, g i := norm_sum_le_of_le _ this
    _ ≤ ‖∑ i ∈ t, g i‖ := le_abs_self _
    _ < ε := hs _ (ht.mono_right le_sup_left)


theorem cauchySeq_finset_of_norm_bounded {f : ι → E} (g : ι → ℝ) (hg : Summable g)
    (h : ∀ i, ‖f i‖ ≤ g i) : CauchySeq fun s : Finset ι => ∑ i ∈ s, f i :=
  cauchySeq_finset_of_norm_bounded_eventually hg <| Eventually.of_forall h


/-- A version of the **direct comparison test** for conditionally convergent series.
See `cauchySeq_finset_of_norm_bounded` for the same statement about absolutely convergent ones. -/
theorem cauchySeq_range_of_norm_bounded {f : ℕ → E} (g : ℕ → ℝ)
    (hg : CauchySeq fun n => ∑ i ∈ range n, g i) (hf : ∀ i, ‖f i‖ ≤ g i) :
    CauchySeq fun n => ∑ i ∈ range n, f i := by
  /-
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : Nat → E
    g : Nat → Real
    hg : CauchySeq fun n => (Finset.range n).sum fun i => g i
    hf : ∀ (i : Nat), LE.le (Norm.norm (f i)) (g i)
    ⊢ CauchySeq fun n => (Finset.range n).sum fun i => f i
  -/
  refine Metric.cauchySeq_iff'.2 fun ε hε => ?_
  /-
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : Nat → E
    g : Nat → Real
    hg : CauchySeq fun n => (Finset.range n).sum fun i => g i
    hf : ∀ (i : Nat), LE.le (Norm.norm (f i)) (g i)
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist ((Finset.range n). …
  -/
  refine (Metric.cauchySeq_iff'.1 hg ε hε).imp fun N hg n hn => ?_
  /-
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : Nat → E
    g : Nat → Real
    hg✝ : CauchySeq fun n => (Finset.range n).sum fun i => g i
    hf : ∀ (i : Nat), LE.le (Norm.norm (f i)) (g i)
    ε : Real
    hε : GT.gt ε 0
    N : Nat
    hg : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist ((Finset.range n).sum fun i =>  …
    n : Nat
    hn : GE.ge n N
    ⊢ LT.lt (Dist.dist ((Finset.range n).sum fun i => f i) ((Finset.range N).sum f …
  -/
  specialize hg n hn
  /-
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : Nat → E
    g : Nat → Real
    hg✝ : CauchySeq fun n => (Finset.range n).sum fun i => g i
    hf : ∀ (i : Nat), LE.le (Norm.norm (f i)) (g i)
    ε : Real
    hε : GT.gt ε 0
    N n : Nat
    hn : GE.ge n N
    hg : LT.lt (Dist.dist ((Finset.range n).sum fun i => g i) ((Finset.range N).su …
    ⊢ LT.lt (Dist.dist ((Finset.range n).sum fun i => f i) ((Finset.range N).sum f …
  -/
  rw [dist_eq_norm, ← sum_Ico_eq_sub _ hn] at hg ⊢
  calc
    ‖∑ k ∈ Ico N n, f k‖ ≤ ∑ k ∈ _, ‖f k‖ := norm_sum_le _ _
    _ ≤ ∑ k ∈ _, g k := sum_le_sum fun x _ => hf x
    _ ≤ ‖∑ k ∈ _, g k‖ := le_abs_self _
    _ < ε := hg


theorem cauchySeq_finset_of_summable_norm {f : ι → E} (hf : Summable fun a => ‖f a‖) :
    CauchySeq fun s : Finset ι => ∑ a ∈ s, f a :=
  cauchySeq_finset_of_norm_bounded _ hf fun _i => le_rfl


/-- If a function `f` is summable in norm, and along some sequence of finsets exhausting the space
its sum is converging to a limit `a`, then this holds along all finsets, i.e., `f` is summable
with sum `a`. -/
theorem hasSum_of_subseq_of_summable {f : ι → E} (hf : Summable fun a => ‖f a‖) {s : α → Finset ι}
    {p : Filter α} [NeBot p] (hs : Tendsto s p atTop) {a : E}
    (ha : Tendsto (fun b => ∑ i ∈ s b, f i) p (𝓝 a)) : HasSum f a :=
  tendsto_nhds_of_cauchySeq_of_subseq (cauchySeq_finset_of_summable_norm hf) hs ha


theorem hasSum_iff_tendsto_nat_of_summable_norm {f : ℕ → E} {a : E} (hf : Summable fun i => ‖f i‖) :
    HasSum f a ↔ Tendsto (fun n : ℕ => ∑ i ∈ range n, f i) atTop (𝓝 a) :=
  ⟨fun h => h.tendsto_sum_nat, fun h => hasSum_of_subseq_of_summable hf tendsto_finset_range h⟩


/-- The direct comparison test for series:  if the norm of `f` is bounded by a real function `g`
which is summable, then `f` is summable. -/
theorem Summable.of_norm_bounded [CompleteSpace E] {f : ι → E} (g : ι → ℝ) (hg : Summable g)
    (h : ∀ i, ‖f i‖ ≤ g i) : Summable f := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : ι → E
    g : ι → Real
    hg : Summable g
    h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
    ⊢ Summable f
  -/
  rw [summable_iff_cauchySeq_finset]
  /-
    ι : Type u_1
    E : Type u_3
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : ι → E
    g : ι → Real
    hg : Summable g
    h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
    ⊢ CauchySeq fun s => s.sum fun b => f b
  -/
  exact cauchySeq_finset_of_norm_bounded g hg h
  /-
    🎉 no goals
  -/


theorem HasSum.norm_le_of_bounded {f : ι → E} {g : ι → ℝ} {a : E} {b : ℝ} (hf : HasSum f a)
    (hg : HasSum g b) (h : ∀ i, ‖f i‖ ≤ g i) : ‖a‖ ≤ b := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    g : ι → Real
    a : E
    b : Real
    hf : HasSum f a
    hg : HasSum g b
    h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
    ⊢ LE.le (Norm.norm a) b
  -/
  classical exact le_of_tendsto_of_tendsto' hf.norm hg fun _s ↦ norm_sum_le_of_le _ fun i _hi ↦ h i
  /-
    🎉 no goals
  -/


/-- Quantitative result associated to the direct comparison test for series:  If `∑' i, g i` is
summable, and for all `i`, `‖f i‖ ≤ g i`, then `‖∑' i, f i‖ ≤ ∑' i, g i`. Note that we do not
assume that `∑' i, f i` is summable, and it might not be the case if `α` is not a complete space. -/
theorem tsum_of_norm_bounded {f : ι → E} {g : ι → ℝ} {a : ℝ} (hg : HasSum g a)
    (h : ∀ i, ‖f i‖ ≤ g i) : ‖∑' i : ι, f i‖ ≤ a := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    g : ι → Real
    a : Real
    hg : HasSum g a
    h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
    ⊢ LE.le (Norm.norm (tsum fun i => f i)) a
  -/
  by_cases hf : Summable f
    /-
      case pos
      ι : Type u_1
      E : Type u_3
      inst✝ : SeminormedAddCommGroup E
      f : ι → E
      g : ι → Real
      a : Real
      hg : HasSum g a
      h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
      hf : Summable f
      ⊢ LE.le (Norm.norm (tsum fun i => f i)) a
    -/
  · exact hf.hasSum.norm_le_of_bounded hg h
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      E : Type u_3
      inst✝ : SeminormedAddCommGroup E
      f : ι → E
      g : ι → Real
      a : Real
      hg : HasSum g a
      h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
      hf : Not (Summable f)
      ⊢ LE.le (Norm.norm (tsum fun i => f i)) a
    -/
  · rw [tsum_eq_zero_of_not_summable hf, norm_zero]
    /-
      case neg
      ι : Type u_1
      E : Type u_3
      inst✝ : SeminormedAddCommGroup E
      f : ι → E
      g : ι → Real
      a : Real
      hg : HasSum g a
      h : ∀ (i : ι), LE.le (Norm.norm (f i)) (g i)
      hf : Not (Summable f)
      ⊢ LE.le 0 a
    -/
    classical exact ge_of_tendsto' hg fun s => sum_nonneg fun i _hi => (norm_nonneg _).trans (h i)
    /-
      🎉 no goals
    -/


/-- If `∑' i, ‖f i‖` is summable, then `‖∑' i, f i‖ ≤ (∑' i, ‖f i‖)`. Note that we do not assume
that `∑' i, f i` is summable, and it might not be the case if `α` is not a complete space. -/
theorem norm_tsum_le_tsum_norm {f : ι → E} (hf : Summable fun i => ‖f i‖) :
    ‖∑' i, f i‖ ≤ ∑' i, ‖f i‖ :=
  tsum_of_norm_bounded hf.hasSum fun _i => le_rfl


/-- Quantitative result associated to the direct comparison test for series: If `∑' i, g i` is
summable, and for all `i`, `‖f i‖₊ ≤ g i`, then `‖∑' i, f i‖₊ ≤ ∑' i, g i`. Note that we
do not assume that `∑' i, f i` is summable, and it might not be the case if `α` is not a complete
space. -/
theorem tsum_of_nnnorm_bounded {f : ι → E} {g : ι → ℝ≥0} {a : ℝ≥0} (hg : HasSum g a)
    (h : ∀ i, ‖f i‖₊ ≤ g i) : ‖∑' i : ι, f i‖₊ ≤ a := by
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    g : ι → NNReal
    a : NNReal
    hg : HasSum g a
    h : ∀ (i : ι), LE.le (NNNorm.nnnorm (f i)) (g i)
    ⊢ LE.le (NNNorm.nnnorm (tsum fun i => f i)) a
  -/
  simp only [← NNReal.coe_le_coe, ← NNReal.hasSum_coe, coe_nnnorm] at *
  /-
    ι : Type u_1
    E : Type u_3
    inst✝ : SeminormedAddCommGroup E
    f : ι → E
    g : ι → NNReal
    a : NNReal
    hg : HasSum (fun a => ↑(g a)) ↑a
    h : ∀ (i : ι), LE.le (Norm.norm (f i)) ↑(g i)
    ⊢ LE.le (Norm.norm (tsum fun i => f i)) ↑a
  -/
  exact tsum_of_norm_bounded hg h
  /-
    🎉 no goals
  -/


/-- If `∑' i, ‖f i‖₊` is summable, then `‖∑' i, f i‖₊ ≤ ∑' i, ‖f i‖₊`. Note that
we do not assume that `∑' i, f i` is summable, and it might not be the case if `α` is not a complete
space. -/
theorem nnnorm_tsum_le {f : ι → E} (hf : Summable fun i => ‖f i‖₊) : ‖∑' i, f i‖₊ ≤ ∑' i, ‖f i‖₊ :=
  tsum_of_nnnorm_bounded hf.hasSum fun _i => le_rfl


/-- Variant of the direct comparison test for series:  if the norm of `f` is eventually bounded by a
real function `g` which is summable, then `f` is summable. -/
theorem Summable.of_norm_bounded_eventually {f : ι → E} (g : ι → ℝ) (hg : Summable g)
    (h : ∀ᶠ i in cofinite, ‖f i‖ ≤ g i) : Summable f :=
  summable_iff_cauchySeq_finset.2 <| cauchySeq_finset_of_norm_bounded_eventually hg h


/-- Variant of the direct comparison test for series:  if the norm of `f` is eventually bounded by a
real function `g` which is summable, then `f` is summable. -/
theorem Summable.of_norm_bounded_eventually_nat {f : ℕ → E} (g : ℕ → ℝ) (hg : Summable g)
    (h : ∀ᶠ i in atTop, ‖f i‖ ≤ g i) : Summable f :=
  .of_norm_bounded_eventually g hg <| Nat.cofinite_eq_atTop ▸ h


theorem Summable.of_nnnorm_bounded {f : ι → E} (g : ι → ℝ≥0) (hg : Summable g)
    (h : ∀ i, ‖f i‖₊ ≤ g i) : Summable f :=
  .of_norm_bounded (fun i => (g i : ℝ)) (NNReal.summable_coe.2 hg) h


theorem Summable.of_norm {f : ι → E} (hf : Summable fun a => ‖f a‖) : Summable f :=
  .of_norm_bounded _ hf fun _i => le_rfl


theorem Summable.of_nnnorm {f : ι → E} (hf : Summable fun a => ‖f a‖₊) : Summable f :=
  .of_nnnorm_bounded _ hf fun _i => le_rfl

