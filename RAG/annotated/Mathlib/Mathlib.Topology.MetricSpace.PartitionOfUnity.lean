/-- Let `K : ι → Set X` be a locally finite family of closed sets in an emetric space. Let
`U : ι → Set X` be a family of open sets such that `K i ⊆ U i` for all `i`. Then for any point
`x : X`, for sufficiently small `r : ℝ≥0∞` and for `y` sufficiently close to `x`, for all `i`, if
`y ∈ K i`, then `EMetric.closedBall y r ⊆ U i`. -/
theorem eventually_nhds_zero_forall_closedBall_subset (hK : ∀ i, IsClosed (K i))
    (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) (x : X) :
    ∀ᶠ p : ℝ≥0∞ × X in 𝓝 0 ×ˢ 𝓝 x, ∀ i, p.2 ∈ K i → closedBall p.2 p.1 ⊆ U i := by
  suffices ∀ i, x ∈ K i → ∀ᶠ p : ℝ≥0∞ × X in 𝓝 0 ×ˢ 𝓝 x, closedBall p.2 p.1 ⊆ U i by
    apply mp_mem ((eventually_all_finite (hfin.point_finite x)).2 this)
      (mp_mem (@tendsto_snd ℝ≥0∞ _ (𝓝 0) _ _ (hfin.iInter_compl_mem_nhds hK x)) _)
    apply univ_mem'
    rintro ⟨r, y⟩ hxy hyU i hi
    simp only [mem_iInter, mem_compl_iff, not_imp_not, mem_preimage] at hxy
    exact hyU _ (hxy _ hi)
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    ⊢ ∀ (i : ι), Membership.mem (K i) x → Filter.Eventually (fun p => HasSubset.Su …
  -/
  intro i hi
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    i : ι
    hi : Membership.mem (K i) x
    ⊢ Filter.Eventually (fun p => HasSubset.Subset (EMetric.closedBall p.2 p.1) (U …
  -/
  rcases nhds_basis_closed_eball.mem_iff.1 ((hU i).mem_nhds <| hKU i hi) with ⟨R, hR₀, hR⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    i : ι
    hi : Membership.mem (K i) x
    R : ENNReal
    hR₀ : LT.lt 0 R
    hR : HasSubset.Subset (EMetric.closedBall x R) (U i)
    ⊢ Filter.Eventually (fun p => HasSubset.Subset (EMetric.closedBall p.2 p.1) (U …
  -/
  rcases ENNReal.lt_iff_exists_nnreal_btwn.mp hR₀ with ⟨r, hr₀, hrR⟩
  filter_upwards [prod_mem_prod (eventually_lt_nhds hr₀)
      (closedBall_mem_nhds x (tsub_pos_iff_lt.2 hrR))] with p hp z hz
  /-
    case h
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    i : ι
    hi : Membership.mem (K i) x
    R : ENNReal
    hR₀ : LT.lt 0 R
    hR : HasSubset.Subset (EMetric.closedBall x R) (U i)
    r : NNReal
    hr₀ : LT.lt 0 ↑r
    hrR : LT.lt (↑r) R
    p : Prod ENNReal X
    hp : Membership.mem (SProd.sprod (setOf fun x => LT.lt x ↑r) (EMetric.closedBa …
    z : X
    hz : Membership.mem (EMetric.closedBall p.2 p.1) z
    ⊢ Membership.mem (U i) z
  -/
  apply hR
  calc
    edist z x ≤ edist z p.2 + edist p.2 x := edist_triangle _ _ _
    _ ≤ p.1 + (R - p.1) := add_le_add hz <| le_trans hp.2 <| tsub_le_tsub_left hp.1.out.le _
    _ = R := add_tsub_cancel_of_le (lt_trans (by exact hp.1) hrR).le


theorem exists_forall_closedBall_subset_aux₁ (hK : ∀ i, IsClosed (K i)) (hU : ∀ i, IsOpen (U i))
    (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) (x : X) :
    ∃ r : ℝ, ∀ᶠ y in 𝓝 x,
      r ∈ Ioi (0 : ℝ) ∩ ENNReal.ofReal ⁻¹' ⋂ (i) (_ : y ∈ K i), { r | closedBall y r ⊆ U i } := by
  have := (ENNReal.continuous_ofReal.tendsto' 0 0 ENNReal.ofReal_zero).eventually
    (eventually_nhds_zero_forall_closedBall_subset hK hU hKU hfin x).curry
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    this : Filter.Eventually (fun x_1 => Filter.Eventually (fun y => ∀ (i : ι), Me …
    ⊢ Exists fun r => Filter.Eventually (fun y => Membership.mem (Inter.inter (Set …
  -/
  rcases this.exists_gt with ⟨r, hr0, hr⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    this : Filter.Eventually (fun x_1 => Filter.Eventually (fun y => ∀ (i : ι), Me …
    r : Real
    hr0 : GT.gt r 0
    hr : Filter.Eventually (fun y => ∀ (i : ι), Membership.mem (K i) { fst := ENNR …
    ⊢ Exists fun r => Filter.Eventually (fun y => Membership.mem (Inter.inter (Set …
  -/
  refine ⟨r, hr.mono fun y hy => ⟨hr0, ?_⟩⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    x : X
    this : Filter.Eventually (fun x_1 => Filter.Eventually (fun y => ∀ (i : ι), Me …
    r : Real
    hr0 : GT.gt r 0
    hr : Filter.Eventually (fun y => ∀ (i : ι), Membership.mem (K i) { fst := ENNR …
    y : X
    hy : ∀ (i : ι), Membership.mem (K i) { fst := ENNReal.ofReal r, snd := y }.2 → …
    ⊢ Membership.mem (Set.preimage ENNReal.ofReal (Set.iInter fun i => Set.iInter  …
  -/
  rwa [mem_preimage, mem_iInter₂]
  /-
    🎉 no goals
  -/


theorem exists_forall_closedBall_subset_aux₂ (y : X) :
    Convex ℝ
      (Ioi (0 : ℝ) ∩ ENNReal.ofReal ⁻¹' ⋂ (i) (_ : y ∈ K i), { r | closedBall y r ⊆ U i }) :=
  (convex_Ioi _).inter <| OrdConnected.convex <| OrdConnected.preimage_ennreal_ofReal <|
    ordConnected_iInter fun i => ordConnected_iInter fun (_ : y ∈ K i) =>
      ordConnected_setOf_closedBall_subset y (U i)


/-- Let `X` be an extended metric space. Let `K : ι → Set X` be a locally finite family of closed
sets, let `U : ι → Set X` be a family of open sets such that `K i ⊆ U i` for all `i`. Then there
exists a positive continuous function `δ : C(X, ℝ)` such that for any `i` and `x ∈ K i`,
we have `EMetric.closedBall x (ENNReal.ofReal (δ x)) ⊆ U i`. -/
theorem exists_continuous_real_forall_closedBall_subset (hK : ∀ i, IsClosed (K i))
    (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) :
    ∃ δ : C(X, ℝ), (∀ x, 0 < δ x) ∧
      ∀ (i), ∀ x ∈ K i, closedBall x (ENNReal.ofReal <| δ x) ⊆ U i := by
  simpa only [mem_inter_iff, forall_and, mem_preimage, mem_iInter, @forall_swap ι X] using
    exists_continuous_forall_mem_convex_of_local_const exists_forall_closedBall_subset_aux₂
      (exists_forall_closedBall_subset_aux₁ hK hU hKU hfin)


/-- Let `X` be an extended metric space. Let `K : ι → Set X` be a locally finite family of closed
sets, let `U : ι → Set X` be a family of open sets such that `K i ⊆ U i` for all `i`. Then there
exists a positive continuous function `δ : C(X, ℝ≥0)` such that for any `i` and `x ∈ K i`,
we have `EMetric.closedBall x (δ x) ⊆ U i`. -/
theorem exists_continuous_nnreal_forall_closedBall_subset (hK : ∀ i, IsClosed (K i))
    (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) :
    ∃ δ : C(X, ℝ≥0), (∀ x, 0 < δ x) ∧ ∀ (i), ∀ x ∈ K i, closedBall x (δ x) ⊆ U i := by
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    ⊢ Exists fun δ => And (∀ (x : X), LT.lt 0 (δ x)) (∀ (i : ι) (x : X), Membershi …
  -/
  rcases exists_continuous_real_forall_closedBall_subset hK hU hKU hfin with ⟨δ, hδ₀, hδ⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContinuousMap X Real
    hδ₀ : ∀ (x : X), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : X), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    ⊢ Exists fun δ => And (∀ (x : X), LT.lt 0 (δ x)) (∀ (i : ι) (x : X), Membershi …
  -/
  lift δ to C(X, ℝ≥0) using fun x => (hδ₀ x).le
  /-
    case intro.intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContinuousMap X NNReal
    hδ₀ : ∀ (x : X), LT.lt 0 ((ContinuousMap.coeNNRealReal.comp δ) x)
    hδ : ∀ (i : ι) (x : X), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    ⊢ Exists fun δ => And (∀ (x : X), LT.lt 0 (δ x)) (∀ (i : ι) (x : X), Membershi …
  -/
  refine ⟨δ, hδ₀, fun i x hi => ?_⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : EMetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContinuousMap X NNReal
    hδ₀ : ∀ (x : X), LT.lt 0 ((ContinuousMap.coeNNRealReal.comp δ) x)
    hδ : ∀ (i : ι) (x : X), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    i : ι
    x : X
    hi : Membership.mem (K i) x
    ⊢ HasSubset.Subset (EMetric.closedBall x ↑(δ x)) (U i)
  -/
  simpa only [← ENNReal.ofReal_coe_nnreal] using hδ i x hi
  /-
    🎉 no goals
  -/


/-- Let `X` be an extended metric space. Let `K : ι → Set X` be a locally finite family of closed
sets, let `U : ι → Set X` be a family of open sets such that `K i ⊆ U i` for all `i`. Then there
exists a positive continuous function `δ : C(X, ℝ≥0∞)` such that for any `i` and `x ∈ K i`,
we have `EMetric.closedBall x (δ x) ⊆ U i`. -/
theorem exists_continuous_eNNReal_forall_closedBall_subset (hK : ∀ i, IsClosed (K i))
    (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) :
    ∃ δ : C(X, ℝ≥0∞), (∀ x, 0 < δ x) ∧ ∀ (i), ∀ x ∈ K i, closedBall x (δ x) ⊆ U i :=
  let ⟨δ, hδ₀, hδ⟩ := exists_continuous_nnreal_forall_closedBall_subset hK hU hKU hfin
  ⟨ContinuousMap.comp ⟨Coe.coe, ENNReal.continuous_coe⟩ δ, fun x => ENNReal.coe_pos.2 (hδ₀ x), hδ⟩


/-- Let `X` be a metric space. Let `K : ι → Set X` be a locally finite family of closed sets, let
`U : ι → Set X` be a family of open sets such that `K i ⊆ U i` for all `i`. Then there exists a
positive continuous function `δ : C(X, ℝ≥0)` such that for any `i` and `x ∈ K i`, we have
`Metric.closedBall x (δ x) ⊆ U i`. -/
theorem exists_continuous_nnreal_forall_closedBall_subset (hK : ∀ i, IsClosed (K i))
    (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) :
    ∃ δ : C(X, ℝ≥0), (∀ x, 0 < δ x) ∧ ∀ (i), ∀ x ∈ K i, closedBall x (δ x) ⊆ U i := by
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : MetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    ⊢ Exists fun δ => And (∀ (x : X), LT.lt 0 (δ x)) (∀ (i : ι) (x : X), Membershi …
  -/
  rcases EMetric.exists_continuous_nnreal_forall_closedBall_subset hK hU hKU hfin with ⟨δ, hδ0, hδ⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : MetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContinuousMap X NNReal
    hδ0 : ∀ (x : X), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : X), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    ⊢ Exists fun δ => And (∀ (x : X), LT.lt 0 (δ x)) (∀ (i : ι) (x : X), Membershi …
  -/
  refine ⟨δ, hδ0, fun i x hx => ?_⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : MetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContinuousMap X NNReal
    hδ0 : ∀ (x : X), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : X), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    i : ι
    x : X
    hx : Membership.mem (K i) x
    ⊢ HasSubset.Subset (Metric.closedBall x ↑(δ x)) (U i)
  -/
  rw [← emetric_closedBall_nnreal]
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝ : MetricSpace X
    K U : ι → Set X
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContinuousMap X NNReal
    hδ0 : ∀ (x : X), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : X), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    i : ι
    x : X
    hx : Membership.mem (K i) x
    ⊢ HasSubset.Subset (EMetric.closedBall x ↑(δ x)) (U i)
  -/
  exact hδ i x hx
  /-
    🎉 no goals
  -/


/-- Let `X` be a metric space. Let `K : ι → Set X` be a locally finite family of closed sets, let
`U : ι → Set X` be a family of open sets such that `K i ⊆ U i` for all `i`. Then there exists a
positive continuous function `δ : C(X, ℝ)` such that for any `i` and `x ∈ K i`, we have
`Metric.closedBall x (δ x) ⊆ U i`. -/
theorem exists_continuous_real_forall_closedBall_subset (hK : ∀ i, IsClosed (K i))
    (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i) (hfin : LocallyFinite K) :
    ∃ δ : C(X, ℝ), (∀ x, 0 < δ x) ∧ ∀ (i), ∀ x ∈ K i, closedBall x (δ x) ⊆ U i :=
  let ⟨δ, hδ₀, hδ⟩ := exists_continuous_nnreal_forall_closedBall_subset hK hU hKU hfin
  ⟨ContinuousMap.comp ⟨Coe.coe, NNReal.continuous_coe⟩ δ, hδ₀, hδ⟩


