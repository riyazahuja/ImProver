/-- A map is said to be `ContractingWith K`, if `K < 1` and `f` is `LipschitzWith K`. -/
def ContractingWith [EMetricSpace α] (K : ℝ≥0) (f : α → α) :=
  K < 1 ∧ LipschitzWith K f


theorem toLipschitzWith (hf : ContractingWith K f) : LipschitzWith K f := hf.2


                                                                             /-
                                                                               α : Type u_1
                                                                               inst✝ : EMetricSpace α
                                                                               K : NNReal
                                                                               f : α → α
                                                                               hf : ContractingWith K f
                                                                               ⊢ LT.lt 0 (HSub.hSub 1 ↑K)
                                                                             -/
theorem one_sub_K_pos' (hf : ContractingWith K f) : (0 : ℝ≥0∞) < 1 - K := by simp [hf.1]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem one_sub_K_ne_zero (hf : ContractingWith K f) : (1 : ℝ≥0∞) - K ≠ 0 :=
  ne_of_gt hf.one_sub_K_pos'


theorem one_sub_K_ne_top : (1 : ℝ≥0∞) - K ≠ ∞ := by
  /-
    K : NNReal
    ⊢ Ne (HSub.hSub 1 ↑K) Top.top
  -/
  norm_cast
  /-
    K : NNReal
    ⊢ Not (Eq (↑(HSub.hSub 1 K)) Top.top)
  -/
  exact ENNReal.coe_ne_top
  /-
    🎉 no goals
  -/


theorem edist_inequality (hf : ContractingWith K f) {x y} (h : edist x y ≠ ∞) :
    edist x y ≤ (edist x (f x) + edist y (f y)) / (1 - K) :=
  suffices edist x y ≤ edist x (f x) + edist y (f y) + K * edist x y by
    rwa [ENNReal.le_div_iff_mul_le (Or.inl hf.one_sub_K_ne_zero) (Or.inl one_sub_K_ne_top),
      mul_comm, ENNReal.sub_mul fun _ _ ↦ h, one_mul, tsub_le_iff_right]
  calc
    edist x y ≤ edist x (f x) + edist (f x) (f y) + edist (f y) y := edist_triangle4 _ _ _ _
                                                                /-
                                                                  α : Type u_1
                                                                  inst✝ : EMetricSpace α
                                                                  K : NNReal
                                                                  f : α → α
                                                                  hf : ContractingWith K f
                                                                  x y : α
                                                                  h : Ne (EDist.edist x y) Top.top
                                                                  ⊢ Eq (HAdd.hAdd (HAdd.hAdd (EDist.edist x (f x)) (EDist.edist (f x) (f y))) (E …
                                                                -/
    _ = edist x (f x) + edist y (f y) + edist (f x) (f y) := by rw [edist_comm y, add_right_comm]
                                                                /-
                                                                  🎉 no goals
                                                                -/
    _ ≤ edist x (f x) + edist y (f y) + K * edist x y := add_le_add le_rfl (hf.2 _ _)


theorem edist_le_of_fixedPoint (hf : ContractingWith K f) {x y} (h : edist x y ≠ ∞)
    (hy : IsFixedPt f y) : edist x y ≤ edist x (f x) / (1 - K) := by
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    x y : α
    h : Ne (EDist.edist x y) Top.top
    hy : Function.IsFixedPt f y
    ⊢ LE.le (EDist.edist x y) (HDiv.hDiv (EDist.edist x (f x)) (HSub.hSub 1 ↑K))
  -/
  simpa only [hy.eq, edist_self, add_zero] using hf.edist_inequality h
  /-
    🎉 no goals
  -/


theorem eq_or_edist_eq_top_of_fixedPoints (hf : ContractingWith K f) {x y} (hx : IsFixedPt f x)
    (hy : IsFixedPt f y) : x = y ∨ edist x y = ∞ := by
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    x y : α
    hx : Function.IsFixedPt f x
    hy : Function.IsFixedPt f y
    ⊢ Or (Eq x y) (Eq (EDist.edist x y) Top.top)
  -/
  refine or_iff_not_imp_right.2 fun h ↦ edist_le_zero.1 ?_
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    x y : α
    hx : Function.IsFixedPt f x
    hy : Function.IsFixedPt f y
    h : Not (Eq (EDist.edist x y) Top.top)
    ⊢ LE.le (EDist.edist x y) 0
  -/
  simpa only [hx.eq, edist_self, add_zero, ENNReal.zero_div] using hf.edist_le_of_fixedPoint h hy
  /-
    🎉 no goals
  -/


/-- If a map `f` is `ContractingWith K`, and `s` is a forward-invariant set, then
restriction of `f` to `s` is `ContractingWith K` as well. -/
theorem restrict (hf : ContractingWith K f) {s : Set α} (hs : MapsTo f s s) :
    ContractingWith K (hs.restrict f s s) :=
  ⟨hf.1, fun x y ↦ hf.2 x y⟩


/-- Banach fixed-point theorem, contraction mapping theorem, `EMetricSpace` version.
A contracting map on a complete metric space has a fixed point.
We include more conclusions in this theorem to avoid proving them again later.

The main API for this theorem are the functions `efixedPoint` and `fixedPoint`,
and lemmas about these functions. -/
theorem exists_fixedPoint (hf : ContractingWith K f) (x : α) (hx : edist x (f x) ≠ ∞) :
    ∃ y, IsFixedPt f y ∧ Tendsto (fun n ↦ f^[n] x) atTop (𝓝 y) ∧
      ∀ n : ℕ, edist (f^[n] x) y ≤ edist x (f x) * (K : ℝ≥0∞) ^ n / (1 - K) :=
  have : CauchySeq fun n ↦ f^[n] x :=
    cauchySeq_of_edist_le_geometric K (edist x (f x)) (ENNReal.coe_lt_one_iff.2 hf.1) hx
      (hf.toLipschitzWith.edist_iterate_succ_le_geometric x)
  let ⟨y, hy⟩ := cauchySeq_tendsto_of_complete this
  ⟨y, isFixedPt_of_tendsto_iterate hy hf.2.continuous.continuousAt, hy,
    edist_le_of_edist_le_geometric_of_tendsto K (edist x (f x))
      (hf.toLipschitzWith.edist_iterate_succ_le_geometric x) hy⟩


/-- Let `x` be a point of a complete emetric space. Suppose that `f` is a contracting map,
and `edist x (f x) ≠ ∞`. Then `efixedPoint` is the unique fixed point of `f`
in `EMetric.ball x ∞`. -/
noncomputable def efixedPoint (hf : ContractingWith K f) (x : α) (hx : edist x (f x) ≠ ∞) : α :=
  Classical.choose <| hf.exists_fixedPoint x hx


theorem efixedPoint_isFixedPt (hf : ContractingWith K f) {x : α} (hx : edist x (f x) ≠ ∞) :
    IsFixedPt f (efixedPoint f hf x hx) :=
  (Classical.choose_spec <| hf.exists_fixedPoint x hx).1


theorem tendsto_iterate_efixedPoint (hf : ContractingWith K f) {x : α} (hx : edist x (f x) ≠ ∞) :
    Tendsto (fun n ↦ f^[n] x) atTop (𝓝 <| efixedPoint f hf x hx) :=
  (Classical.choose_spec <| hf.exists_fixedPoint x hx).2.1


theorem apriori_edist_iterate_efixedPoint_le (hf : ContractingWith K f) {x : α}
    (hx : edist x (f x) ≠ ∞) (n : ℕ) :
    edist (f^[n] x) (efixedPoint f hf x hx) ≤ edist x (f x) * (K : ℝ≥0∞) ^ n / (1 - K) :=
  (Classical.choose_spec <| hf.exists_fixedPoint x hx).2.2 n


theorem edist_efixedPoint_le (hf : ContractingWith K f) {x : α} (hx : edist x (f x) ≠ ∞) :
    edist x (efixedPoint f hf x hx) ≤ edist x (f x) / (1 - K) := by
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    ⊢ LE.le (EDist.edist x (ContractingWith.efixedPoint f hf x hx)) (HDiv.hDiv (ED …
  -/
  convert hf.apriori_edist_iterate_efixedPoint_le hx 0
  /-
    case h.e'_4.h.e'_5
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    ⊢ Eq (EDist.edist x (f x)) (HMul.hMul (EDist.edist x (f x)) (HPow.hPow (↑K) 0))
  -/
  simp only [pow_zero, mul_one]
  /-
    🎉 no goals
  -/


theorem edist_efixedPoint_lt_top (hf : ContractingWith K f) {x : α} (hx : edist x (f x) ≠ ∞) :
    edist x (efixedPoint f hf x hx) < ∞ :=
  (hf.edist_efixedPoint_le hx).trans_lt
    (ENNReal.mul_ne_top hx <| ENNReal.inv_ne_top.2 hf.one_sub_K_ne_zero).lt_top


theorem efixedPoint_eq_of_edist_lt_top (hf : ContractingWith K f) {x : α} (hx : edist x (f x) ≠ ∞)
    {y : α} (hy : edist y (f y) ≠ ∞) (h : edist x y ≠ ∞) :
    efixedPoint f hf x hx = efixedPoint f hf y hy := by
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    y : α
    hy : Ne (EDist.edist y (f y)) Top.top
    h : Ne (EDist.edist x y) Top.top
    ⊢ Eq (ContractingWith.efixedPoint f hf x hx) (ContractingWith.efixedPoint f hf …
  -/
  refine (hf.eq_or_edist_eq_top_of_fixedPoints ?_ ?_).elim id fun h' ↦ False.elim (ne_of_lt ?_ h')
        /-
          case refine_1
          α : Type u_1
          inst✝¹ : EMetricSpace α
          K : NNReal
          f : α → α
          inst✝ : CompleteSpace α
          hf : ContractingWith K f
          x : α
          hx : Ne (EDist.edist x (f x)) Top.top
          y : α
          hy : Ne (EDist.edist y (f y)) Top.top
          h : Ne (EDist.edist x y) Top.top
          ⊢ Function.IsFixedPt f (ContractingWith.efixedPoint f hf x hx)
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
    <;> try apply efixedPoint_isFixedPt
  /-
    case refine_3
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    y : α
    hy : Ne (EDist.edist y (f y)) Top.top
    h : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
    ⊢ LT.lt (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
  -/
  change edistLtTopSetoid _ _
  /-
    case refine_3
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    y : α
    hy : Ne (EDist.edist y (f y)) Top.top
    h : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
    ⊢ EMetric.edistLtTopSetoid (ContractingWith.efixedPoint f hf x hx) (Contractin …
  -/
  trans x
    /-
      α : Type u_1
      inst✝¹ : EMetricSpace α
      K : NNReal
      f : α → α
      inst✝ : CompleteSpace α
      hf : ContractingWith K f
      x : α
      hx : Ne (EDist.edist x (f x)) Top.top
      y : α
      hy : Ne (EDist.edist y (f y)) Top.top
      h : Ne (EDist.edist x y) Top.top
      h' : Eq (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
      ⊢ EMetric.edistLtTopSetoid (ContractingWith.efixedPoint f hf x hx) x
    -/
  · apply Setoid.symm' -- Porting note: Originally `symm`
    /-
      case a
      α : Type u_1
      inst✝¹ : EMetricSpace α
      K : NNReal
      f : α → α
      inst✝ : CompleteSpace α
      hf : ContractingWith K f
      x : α
      hx : Ne (EDist.edist x (f x)) Top.top
      y : α
      hy : Ne (EDist.edist y (f y)) Top.top
      h : Ne (EDist.edist x y) Top.top
      h' : Eq (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
      ⊢ EMetric.edistLtTopSetoid x (ContractingWith.efixedPoint f hf x hx)
    -/
    exact hf.edist_efixedPoint_lt_top hx
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    y : α
    hy : Ne (EDist.edist y (f y)) Top.top
    h : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
    ⊢ EMetric.edistLtTopSetoid x (ContractingWith.efixedPoint f hf y hy)
  -/
  trans y
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    K : NNReal
    f : α → α
    inst✝ : CompleteSpace α
    hf : ContractingWith K f
    x : α
    hx : Ne (EDist.edist x (f x)) Top.top
    y : α
    hy : Ne (EDist.edist y (f y)) Top.top
    h : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint f hf x hx) (ContractingWith. …
    ⊢ EMetric.edistLtTopSetoid x y
  -/
  exacts [lt_top_iff_ne_top.2 h, hf.edist_efixedPoint_lt_top hy]
  /-
    🎉 no goals
  -/


/-- Banach fixed-point theorem for maps contracting on a complete subset. -/
theorem exists_fixedPoint' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    ∃ y ∈ s, IsFixedPt f y ∧ Tendsto (fun n ↦ f^[n] x) atTop (𝓝 y) ∧
      ∀ n : ℕ, edist (f^[n] x) y ≤ edist x (f x) * (K : ℝ≥0∞) ^ n / (1 - K) := by
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    ⊢ Exists fun y => And (Membership.mem s y) (And (Function.IsFixedPt f y) (And  …
  -/
  haveI := hsc.completeSpace_coe
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    this : CompleteSpace ↑s
    ⊢ Exists fun y => And (Membership.mem s y) (And (Function.IsFixedPt f y) (And  …
  -/
  rcases hf.exists_fixedPoint ⟨x, hxs⟩ hx with ⟨y, hfy, h_tendsto, hle⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    this : CompleteSpace ↑s
    y : ↑s
    hfy : Function.IsFixedPt (Set.MapsTo.restrict f s s hsf) y
    h_tendsto : Filter.Tendsto (fun n => Nat.iterate (Set.MapsTo.restrict f s s hs …
    hle : ∀ (n : Nat), LE.le (EDist.edist (Nat.iterate (Set.MapsTo.restrict f s s  …
    ⊢ Exists fun y => And (Membership.mem s y) (And (Function.IsFixedPt f y) (And  …
  -/
  refine ⟨y, y.2, Subtype.ext_iff_val.1 hfy, ?_, fun n ↦ ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      this : CompleteSpace ↑s
      y : ↑s
      hfy : Function.IsFixedPt (Set.MapsTo.restrict f s s hsf) y
      h_tendsto : Filter.Tendsto (fun n => Nat.iterate (Set.MapsTo.restrict f s s hs …
      hle : ∀ (n : Nat), LE.le (EDist.edist (Nat.iterate (Set.MapsTo.restrict f s s  …
      ⊢ Filter.Tendsto (fun n => Nat.iterate f n x) Filter.atTop (nhds ↑y)
    -/
  · convert (continuous_subtype_val.tendsto _).comp h_tendsto
    /-
      case h.e'_3.h
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      this : CompleteSpace ↑s
      y : ↑s
      hfy : Function.IsFixedPt (Set.MapsTo.restrict f s s hsf) y
      h_tendsto : Filter.Tendsto (fun n => Nat.iterate (Set.MapsTo.restrict f s s hs …
      hle : ∀ (n : Nat), LE.le (EDist.edist (Nat.iterate (Set.MapsTo.restrict f s s  …
      x✝ : Nat
      ⊢ Eq (Nat.iterate f x✝ x) (Function.comp Subtype.val (fun n => Nat.iterate (Se …
    -/
    simp only [(· ∘ ·), MapsTo.iterate_restrict, MapsTo.val_restrict_apply]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      this : CompleteSpace ↑s
      y : ↑s
      hfy : Function.IsFixedPt (Set.MapsTo.restrict f s s hsf) y
      h_tendsto : Filter.Tendsto (fun n => Nat.iterate (Set.MapsTo.restrict f s s hs …
      hle : ∀ (n : Nat), LE.le (EDist.edist (Nat.iterate (Set.MapsTo.restrict f s s  …
      n : Nat
      ⊢ LE.le (EDist.edist (Nat.iterate f n x) ↑y) (HDiv.hDiv (HMul.hMul (EDist.edis …
    -/
  · convert hle n
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      this : CompleteSpace ↑s
      y : ↑s
      hfy : Function.IsFixedPt (Set.MapsTo.restrict f s s hsf) y
      h_tendsto : Filter.Tendsto (fun n => Nat.iterate (Set.MapsTo.restrict f s s hs …
      hle : ∀ (n : Nat), LE.le (EDist.edist (Nat.iterate (Set.MapsTo.restrict f s s  …
      n : Nat
      ⊢ Eq (EDist.edist (Nat.iterate f n x) ↑y) (EDist.edist (Nat.iterate (Set.MapsT …
    -/
    rw [MapsTo.iterate_restrict]
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      this : CompleteSpace ↑s
      y : ↑s
      hfy : Function.IsFixedPt (Set.MapsTo.restrict f s s hsf) y
      h_tendsto : Filter.Tendsto (fun n => Nat.iterate (Set.MapsTo.restrict f s s hs …
      hle : ∀ (n : Nat), LE.le (EDist.edist (Nat.iterate (Set.MapsTo.restrict f s s  …
      n : Nat
      ⊢ Eq (EDist.edist (Nat.iterate f n x) ↑y) (EDist.edist (Set.MapsTo.restrict (N …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Let `s` be a complete forward-invariant set of a self-map `f`. If `f` contracts on `s`
and `x ∈ s` satisfies `edist x (f x) ≠ ∞`, then `efixedPoint'` is the unique fixed point
of the restriction of `f` to `s ∩ EMetric.ball x ∞`. -/
noncomputable def efixedPoint' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) (x : α) (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    α :=
  Classical.choose <| hf.exists_fixedPoint' hsc hsf hxs hx


theorem efixedPoint_mem' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    efixedPoint' f hsc hsf hf x hxs hx ∈ s :=
  (Classical.choose_spec <| hf.exists_fixedPoint' hsc hsf hxs hx).1


theorem efixedPoint_isFixedPt' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    IsFixedPt f (efixedPoint' f hsc hsf hf x hxs hx) :=
  (Classical.choose_spec <| hf.exists_fixedPoint' hsc hsf hxs hx).2.1


theorem tendsto_iterate_efixedPoint' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    Tendsto (fun n ↦ f^[n] x) atTop (𝓝 <| efixedPoint' f hsc hsf hf x hxs hx) :=
  (Classical.choose_spec <| hf.exists_fixedPoint' hsc hsf hxs hx).2.2.1


theorem apriori_edist_iterate_efixedPoint_le' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞)
    (n : ℕ) :
    edist (f^[n] x) (efixedPoint' f hsc hsf hf x hxs hx) ≤
      edist x (f x) * (K : ℝ≥0∞) ^ n / (1 - K) :=
  (Classical.choose_spec <| hf.exists_fixedPoint' hsc hsf hxs hx).2.2.2 n


theorem edist_efixedPoint_le' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    edist x (efixedPoint' f hsc hsf hf x hxs hx) ≤ edist x (f x) / (1 - K) := by
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    ⊢ LE.le (EDist.edist x (ContractingWith.efixedPoint' f hsc hsf hf x hxs hx)) ( …
  -/
  convert hf.apriori_edist_iterate_efixedPoint_le' hsc hsf hxs hx 0
  /-
    case h.e'_4.h.e'_5
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hf : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    ⊢ Eq (EDist.edist x (f x)) (HMul.hMul (EDist.edist x (f x)) (HPow.hPow (↑K) 0))
  -/
  rw [pow_zero, mul_one]
  /-
    🎉 no goals
  -/


theorem edist_efixedPoint_lt_top' {s : Set α} (hsc : IsComplete s) (hsf : MapsTo f s s)
    (hf : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s) (hx : edist x (f x) ≠ ∞) :
    edist x (efixedPoint' f hsc hsf hf x hxs hx) < ∞ :=
  (hf.edist_efixedPoint_le' hsc hsf hxs hx).trans_lt
    (ENNReal.mul_ne_top hx <| ENNReal.inv_ne_top.2 hf.one_sub_K_ne_zero).lt_top


/-- If a globally contracting map `f` has two complete forward-invariant sets `s`, `t`,
and `x ∈ s` is at a finite distance from `y ∈ t`, then the `efixedPoint'` constructed by `x`
is the same as the `efixedPoint'` constructed by `y`.

This lemma takes additional arguments stating that `f` contracts on `s` and `t` because this way
it can be used to prove the desired equality with non-trivial proofs of these facts. -/
theorem efixedPoint_eq_of_edist_lt_top' (hf : ContractingWith K f) {s : Set α} (hsc : IsComplete s)
    (hsf : MapsTo f s s) (hfs : ContractingWith K <| hsf.restrict f s s) {x : α} (hxs : x ∈ s)
    (hx : edist x (f x) ≠ ∞) {t : Set α} (htc : IsComplete t) (htf : MapsTo f t t)
    (hft : ContractingWith K <| htf.restrict f t t) {y : α} (hyt : y ∈ t) (hy : edist y (f y) ≠ ∞)
    (hxy : edist x y ≠ ∞) :
    efixedPoint' f hsc hsf hfs x hxs hx = efixedPoint' f htc htf hft y hyt hy := by
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    t : Set α
    htc : IsComplete t
    htf : Set.MapsTo f t t
    hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
    y : α
    hyt : Membership.mem t y
    hy : Ne (EDist.edist y (f y)) Top.top
    hxy : Ne (EDist.edist x y) Top.top
    ⊢ Eq (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (ContractingWith.ef …
  -/
  refine (hf.eq_or_edist_eq_top_of_fixedPoints ?_ ?_).elim id fun h' ↦ False.elim (ne_of_lt ?_ h')
        /-
          case refine_1
          α : Type u_1
          inst✝ : EMetricSpace α
          K : NNReal
          f : α → α
          hf : ContractingWith K f
          s : Set α
          hsc : IsComplete s
          hsf : Set.MapsTo f s s
          hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
          x : α
          hxs : Membership.mem s x
          hx : Ne (EDist.edist x (f x)) Top.top
          t : Set α
          htc : IsComplete t
          htf : Set.MapsTo f t t
          hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
          y : α
          hyt : Membership.mem t y
          hy : Ne (EDist.edist y (f y)) Top.top
          hxy : Ne (EDist.edist x y) Top.top
          ⊢ Function.IsFixedPt f (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx)
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
    <;> try apply efixedPoint_isFixedPt'
  /-
    case refine_3
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    t : Set α
    htc : IsComplete t
    htf : Set.MapsTo f t t
    hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
    y : α
    hyt : Membership.mem t y
    hy : Ne (EDist.edist y (f y)) Top.top
    hxy : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
    ⊢ LT.lt (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
  -/
  change edistLtTopSetoid _ _
  /-
    case refine_3
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    t : Set α
    htc : IsComplete t
    htf : Set.MapsTo f t t
    hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
    y : α
    hyt : Membership.mem t y
    hy : Ne (EDist.edist y (f y)) Top.top
    hxy : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
    ⊢ EMetric.edistLtTopSetoid (ContractingWith.efixedPoint' f hsc hsf hfs x hxs h …
  -/
  trans x
    /-
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      hf : ContractingWith K f
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      t : Set α
      htc : IsComplete t
      htf : Set.MapsTo f t t
      hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
      y : α
      hyt : Membership.mem t y
      hy : Ne (EDist.edist y (f y)) Top.top
      hxy : Ne (EDist.edist x y) Top.top
      h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
      ⊢ EMetric.edistLtTopSetoid (ContractingWith.efixedPoint' f hsc hsf hfs x hxs h …
    -/
  · apply Setoid.symm' -- Porting note: Originally `symm`
    /-
      case a
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      hf : ContractingWith K f
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      t : Set α
      htc : IsComplete t
      htf : Set.MapsTo f t t
      hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
      y : α
      hyt : Membership.mem t y
      hy : Ne (EDist.edist y (f y)) Top.top
      hxy : Ne (EDist.edist x y) Top.top
      h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
      ⊢ EMetric.edistLtTopSetoid x (ContractingWith.efixedPoint' f hsc hsf hfs x hxs …
    -/
    apply edist_efixedPoint_lt_top'
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : EMetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    s : Set α
    hsc : IsComplete s
    hsf : Set.MapsTo f s s
    hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
    x : α
    hxs : Membership.mem s x
    hx : Ne (EDist.edist x (f x)) Top.top
    t : Set α
    htc : IsComplete t
    htf : Set.MapsTo f t t
    hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
    y : α
    hyt : Membership.mem t y
    hy : Ne (EDist.edist y (f y)) Top.top
    hxy : Ne (EDist.edist x y) Top.top
    h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
    ⊢ EMetric.edistLtTopSetoid x (ContractingWith.efixedPoint' f htc htf hft y hyt …
  -/
  trans y
    /-
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      hf : ContractingWith K f
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      t : Set α
      htc : IsComplete t
      htf : Set.MapsTo f t t
      hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
      y : α
      hyt : Membership.mem t y
      hy : Ne (EDist.edist y (f y)) Top.top
      hxy : Ne (EDist.edist x y) Top.top
      h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
      ⊢ EMetric.edistLtTopSetoid x y
    -/
  · exact lt_top_iff_ne_top.2 hxy
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : EMetricSpace α
      K : NNReal
      f : α → α
      hf : ContractingWith K f
      s : Set α
      hsc : IsComplete s
      hsf : Set.MapsTo f s s
      hfs : ContractingWith K (Set.MapsTo.restrict f s s hsf)
      x : α
      hxs : Membership.mem s x
      hx : Ne (EDist.edist x (f x)) Top.top
      t : Set α
      htc : IsComplete t
      htf : Set.MapsTo f t t
      hft : ContractingWith K (Set.MapsTo.restrict f t t htf)
      y : α
      hyt : Membership.mem t y
      hy : Ne (EDist.edist y (f y)) Top.top
      hxy : Ne (EDist.edist x y) Top.top
      h' : Eq (EDist.edist (ContractingWith.efixedPoint' f hsc hsf hfs x hxs hx) (Co …
      ⊢ EMetric.edistLtTopSetoid y (ContractingWith.efixedPoint' f htc htf hft y hyt …
    -/
  · apply edist_efixedPoint_lt_top'
    /-
      🎉 no goals
    -/


theorem one_sub_K_pos (hf : ContractingWith K f) : (0 : ℝ) < 1 - K :=
  sub_pos.2 hf.1


theorem dist_le_mul (x y : α) : dist (f x) (f y) ≤ K * dist x y :=
  hf.toLipschitzWith.dist_le_mul x y


theorem dist_inequality (x y) : dist x y ≤ (dist x (f x) + dist y (f y)) / (1 - K) :=
  suffices dist x y ≤ dist x (f x) + dist y (f y) + K * dist x y by
    /-
      α : Type u_1
      inst✝ : MetricSpace α
      K : NNReal
      f : α → α
      hf : ContractingWith K f
      x y : α
      this : LE.le (Dist.dist x y) (HAdd.hAdd (HAdd.hAdd (Dist.dist x (f x)) (Dist.d …
      ⊢ LE.le (Dist.dist x y) (HDiv.hDiv (HAdd.hAdd (Dist.dist x (f x)) (Dist.dist y …
    -/
    rwa [le_div_iff₀ hf.one_sub_K_pos, mul_comm, _root_.sub_mul, one_mul, sub_le_iff_le_add]
    /-
      🎉 no goals
    -/
  calc
    dist x y ≤ dist x (f x) + dist y (f y) + dist (f x) (f y) := dist_triangle4_right _ _ _ _
    _ ≤ dist x (f x) + dist y (f y) + K * dist x y := add_le_add_left (hf.dist_le_mul _ _) _


theorem dist_le_of_fixedPoint (x) {y} (hy : IsFixedPt f y) : dist x y ≤ dist x (f x) / (1 - K) := by
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    x y : α
    hy : Function.IsFixedPt f y
    ⊢ LE.le (Dist.dist x y) (HDiv.hDiv (Dist.dist x (f x)) (HSub.hSub 1 ↑K))
  -/
  simpa only [hy.eq, dist_self, add_zero] using hf.dist_inequality x y
  /-
    🎉 no goals
  -/


theorem fixedPoint_unique' {x y} (hx : IsFixedPt f x) (hy : IsFixedPt f y) : x = y :=
  (hf.eq_or_edist_eq_top_of_fixedPoints hx hy).resolve_right (edist_ne_top _ _)


/-- Let `f` be a contracting map with constant `K`; let `g` be another map uniformly
`C`-close to `f`. If `x` and `y` are their fixed points, then `dist x y ≤ C / (1 - K)`. -/
theorem dist_fixedPoint_fixedPoint_of_dist_le' (g : α → α) {x y} (hx : IsFixedPt f x)
    (hy : IsFixedPt g y) {C} (hfg : ∀ z, dist (f z) (g z) ≤ C) : dist x y ≤ C / (1 - K) :=
  calc
    dist x y = dist y x := dist_comm x y
    _ ≤ dist y (f y) / (1 - K) := hf.dist_le_of_fixedPoint y hx
                                         /-
                                           α : Type u_1
                                           inst✝ : MetricSpace α
                                           K : NNReal
                                           f : α → α
                                           hf : ContractingWith K f
                                           g : α → α
                                           x y : α
                                           hx : Function.IsFixedPt f x
                                           hy : Function.IsFixedPt g y
                                           C : Real
                                           hfg : ∀ (z : α), LE.le (Dist.dist (f z) (g z)) C
                                           ⊢ Eq (HDiv.hDiv (Dist.dist y (f y)) (HSub.hSub 1 ↑K)) (HDiv.hDiv (Dist.dist (f …
                                         -/
    _ = dist (f y) (g y) / (1 - K) := by rw [hy.eq, dist_comm]
                                         /-
                                           🎉 no goals
                                         -/
    _ ≤ C / (1 - K) := (div_le_div_iff_of_pos_right hf.one_sub_K_pos).2 (hfg y)


/-- The unique fixed point of a contracting map in a nonempty complete metric space. -/
noncomputable def fixedPoint : α :=
  efixedPoint f hf _ (edist_ne_top (Classical.choice ‹Nonempty α›) _)


/-- The point provided by `ContractingWith.fixedPoint` is actually a fixed point. -/
theorem fixedPoint_isFixedPt : IsFixedPt f (fixedPoint f hf) :=
  hf.efixedPoint_isFixedPt _


theorem fixedPoint_unique {x} (hx : IsFixedPt f x) : x = fixedPoint f hf :=
  hf.fixedPoint_unique' hx hf.fixedPoint_isFixedPt


theorem dist_fixedPoint_le (x) : dist x (fixedPoint f hf) ≤ dist x (f x) / (1 - K) :=
  hf.dist_le_of_fixedPoint x hf.fixedPoint_isFixedPt


/-- Aposteriori estimates on the convergence of iterates to the fixed point. -/
theorem aposteriori_dist_iterate_fixedPoint_le (x n) :
    dist (f^[n] x) (fixedPoint f hf) ≤ dist (f^[n] x) (f^[n + 1] x) / (1 - K) := by
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    x : α
    n : Nat
    ⊢ LE.le (Dist.dist (Nat.iterate f n x) (ContractingWith.fixedPoint f hf)) (HDi …
  -/
  rw [iterate_succ']
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    x : α
    n : Nat
    ⊢ LE.le (Dist.dist (Nat.iterate f n x) (ContractingWith.fixedPoint f hf)) (HDi …
  -/
  apply hf.dist_fixedPoint_le
  /-
    🎉 no goals
  -/


theorem apriori_dist_iterate_fixedPoint_le (x n) :
    dist (f^[n] x) (fixedPoint f hf) ≤ dist x (f x) * (K : ℝ) ^ n / (1 - K) :=
  calc
    _ ≤ dist (f^[n] x) (f^[n + 1] x) / (1 - K) := hf.aposteriori_dist_iterate_fixedPoint_le x n
    _ ≤ _ := by
      /-
        α : Type u_1
        inst✝² : MetricSpace α
        K : NNReal
        f : α → α
        hf : ContractingWith K f
        inst✝¹ : Nonempty α
        inst✝ : CompleteSpace α
        x : α
        n : Nat
        ⊢ LE.le (HDiv.hDiv (Dist.dist (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n  …
      -/
      gcongr; exacts [hf.one_sub_K_pos.le, hf.toLipschitzWith.dist_iterate_succ_le_geometric x n]
              /-
                🎉 no goals
              -/


theorem tendsto_iterate_fixedPoint (x) :
    Tendsto (fun n ↦ f^[n] x) atTop (𝓝 <| fixedPoint f hf) := by
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    x : α
    ⊢ Filter.Tendsto (fun n => Nat.iterate f n x) Filter.atTop (nhds (ContractingW …
  -/
  convert tendsto_iterate_efixedPoint hf (edist_ne_top x _)
  /-
    case h.e'_5.h.e'_3
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    x : α
    ⊢ Eq (ContractingWith.fixedPoint f hf) (ContractingWith.efixedPoint f hf x ⋯)
  -/
  refine (fixedPoint_unique _ ?_).symm
  /-
    case h.e'_5.h.e'_3
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    hf : ContractingWith K f
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    x : α
    ⊢ Function.IsFixedPt f (ContractingWith.efixedPoint f hf x ⋯)
  -/
  apply efixedPoint_isFixedPt
  /-
    🎉 no goals
  -/


theorem fixedPoint_lipschitz_in_map {g : α → α} (hg : ContractingWith K g) {C}
    (hfg : ∀ z, dist (f z) (g z) ≤ C) : dist (fixedPoint f hf) (fixedPoint g hg) ≤ C / (1 - K) :=
  hf.dist_fixedPoint_fixedPoint_of_dist_le' g hf.fixedPoint_isFixedPt hg.fixedPoint_isFixedPt hfg


/-- If a map `f` has a contracting iterate `f^[n]`, then the fixed point of `f^[n]` is also a fixed
point of `f`. -/
theorem isFixedPt_fixedPoint_iterate {n : ℕ} (hf : ContractingWith K f^[n]) :
    IsFixedPt f (hf.fixedPoint f^[n]) := by
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    ⊢ Function.IsFixedPt f (ContractingWith.fixedPoint (Nat.iterate f n) hf)
  -/
  set x := hf.fixedPoint f^[n]
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    ⊢ Function.IsFixedPt f x
  -/
  have hx : f^[n] x = x := hf.fixedPoint_isFixedPt
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    ⊢ Function.IsFixedPt f x
  -/
  have := hf.toLipschitzWith.dist_le_mul x (f x)
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    this : LE.le (Dist.dist (Nat.iterate f n x) (Nat.iterate f n (f x))) (HMul.hMu …
    ⊢ Function.IsFixedPt f x
  -/
  rw [← iterate_succ_apply, iterate_succ_apply', hx] at this
  -- Porting note: Originally `contrapose! this`
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    this : LE.le (Dist.dist x (f x)) (HMul.hMul (↑K) (Dist.dist x (f x)))
    ⊢ Function.IsFixedPt f x
  -/
  revert this
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    ⊢ LE.le (Dist.dist x (f x)) (HMul.hMul (↑K) (Dist.dist x (f x))) → Function.Is …
  -/
  contrapose!
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    ⊢ Not (Function.IsFixedPt f x) → LT.lt (HMul.hMul (↑K) (Dist.dist x (f x))) (D …
  -/
  intro this
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    this : Not (Function.IsFixedPt f x)
    ⊢ LT.lt (HMul.hMul (↑K) (Dist.dist x (f x))) (Dist.dist x (f x))
  -/
  have := dist_pos.2 (Ne.symm this)
  /-
    α : Type u_1
    inst✝² : MetricSpace α
    K : NNReal
    f : α → α
    inst✝¹ : Nonempty α
    inst✝ : CompleteSpace α
    n : Nat
    hf : ContractingWith K (Nat.iterate f n)
    x : α := ContractingWith.fixedPoint (Nat.iterate f n) hf
    hx : Eq (Nat.iterate f n x) x
    this✝ : Not (Function.IsFixedPt f x)
    this : LT.lt 0 (Dist.dist x (f x))
    ⊢ LT.lt (HMul.hMul (↑K) (Dist.dist x (f x))) (Dist.dist x (f x))
  -/
  simpa only [NNReal.coe_one, one_mul, NNReal.val_eq_coe] using (mul_lt_mul_right this).mpr hf.left
  /-
    🎉 no goals
  -/


