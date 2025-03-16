/-- The interval `[0,1]` as a compact set with non-empty interior. -/
def TopologicalSpace.PositiveCompacts.Icc01 : PositiveCompacts ℝ where
  carrier := Icc 0 1
  isCompact' := isCompact_Icc
                           /-
                             ⊢ (interior { carrier := Set.Icc 0 1, isCompact' := ⋯ }.carrier).Nonempty
                           -/
  interior_nonempty' := by simp_rw [interior_Icc, nonempty_Ioo, zero_lt_one]
                           /-
                             🎉 no goals
                           -/


/-- The set `[0,1]^ι` as a compact set with non-empty interior. -/
def TopologicalSpace.PositiveCompacts.piIcc01 (ι : Type*) [Finite ι] :
    PositiveCompacts (ι → ℝ) where
  carrier := pi univ fun _ => Icc 0 1
  isCompact' := isCompact_univ_pi fun _ => isCompact_Icc
  interior_nonempty' := by
    simp only [interior_pi_set, Set.toFinite, interior_Icc, univ_pi_nonempty_iff, nonempty_Ioo,
      imp_true_iff, zero_lt_one]


/-- The parallelepiped formed from the standard basis for `ι → ℝ` is `[0,1]^ι` -/
theorem Basis.parallelepiped_basisFun (ι : Type*) [Fintype ι] :
    (Pi.basisFun ℝ ι).parallelepiped = TopologicalSpace.PositiveCompacts.piIcc01 ι :=
  SetLike.coe_injective <| by
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      ⊢ Eq ↑(Pi.basisFun Real ι).parallelepiped ↑(TopologicalSpace.PositiveCompacts. …
    -/
    refine Eq.trans ?_ ((uIcc_of_le ?_).trans (Set.pi_univ_Icc _ _).symm)
      /-
        case refine_1
        ι : Type u_1
        inst✝ : Fintype ι
        ⊢ Eq (↑(Pi.basisFun Real ι).parallelepiped) (Set.uIcc (fun i => 0) fun i => 1)
      -/
    · classical convert parallelepiped_single (ι := ι) 1
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        ι : Type u_1
        inst✝ : Fintype ι
        ⊢ LE.le (fun i => 0) fun i => 1
      -/
    · exact zero_le_one
      /-
        🎉 no goals
      -/


/-- A parallelepiped can be expressed on the standard basis. -/
theorem Basis.parallelepiped_eq_map  {ι E : Type*} [Fintype ι] [NormedAddCommGroup E]
    [NormedSpace ℝ E] (b : Basis ι ℝ E) :
    b.parallelepiped = (PositiveCompacts.piIcc01 ι).map b.equivFun.symm
      b.equivFunL.symm.continuous b.equivFunL.symm.isOpenMap := by
  classical
  rw [← Basis.parallelepiped_basisFun, ← Basis.parallelepiped_map]
  congr with x
  simp [Pi.single_apply]


theorem Basis.map_addHaar {ι E F : Type*} [Fintype ι] [NormedAddCommGroup E] [NormedAddCommGroup F]
    [NormedSpace ℝ E] [NormedSpace ℝ F] [MeasurableSpace E] [MeasurableSpace F] [BorelSpace E]
    [BorelSpace F] [SecondCountableTopology F] [SigmaCompactSpace F]
    (b : Basis ι ℝ E) (f : E ≃L[ℝ] F) :
    map f b.addHaar = (b.map f.toLinearEquiv).addHaar := by
  have : IsAddHaarMeasure (map f b.addHaar) :=
    AddEquiv.isAddHaarMeasure_map b.addHaar f.toAddEquiv f.continuous f.symm.continuous
  rw [eq_comm, Basis.addHaar_eq_iff, Measure.map_apply f.continuous.measurable
    (PositiveCompacts.isCompact _).measurableSet, Basis.coe_parallelepiped, Basis.coe_map]
  /-
    ι : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : Fintype ι
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : NormedSpace Real F
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : MeasurableSpace F
    inst✝³ : BorelSpace E
    inst✝² : BorelSpace F
    inst✝¹ : SecondCountableTopology F
    inst✝ : SigmaCompactSpace F
    b : Basis ι Real E
    f : ContinuousLinearEquiv (RingHom.id Real) E F
    this : (MeasureTheory.Measure.map (⇑f) b.addHaar).IsAddHaarMeasure
    ⊢ Eq (b.addHaar (Set.preimage (⇑f) (_root_.parallelepiped (Function.comp ⇑f.to …
  -/
  erw [← image_parallelepiped, f.toEquiv.preimage_image, addHaar_self]
  /-
    🎉 no goals
  -/


/-- The Haar measure equals the Lebesgue measure on `ℝ`. -/
theorem addHaarMeasure_eq_volume : addHaarMeasure Icc01 = volume := by
  /-
    ⊢ Eq (MeasureTheory.Measure.addHaarMeasure TopologicalSpace.PositiveCompacts.I …
  -/
  convert (addHaarMeasure_unique volume Icc01).symm; simp [Icc01]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The Haar measure equals the Lebesgue measure on `ℝ^ι`. -/
theorem addHaarMeasure_eq_volume_pi (ι : Type*) [Fintype ι] :
    addHaarMeasure (piIcc01 ι) = volume := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    ⊢ Eq (MeasureTheory.Measure.addHaarMeasure (TopologicalSpace.PositiveCompacts. …
  -/
  convert (addHaarMeasure_unique volume (piIcc01 ι)).symm
  simp only [piIcc01, volume_pi_pi fun _ => Icc (0 : ℝ) 1, PositiveCompacts.coe_mk,
    Compacts.coe_mk, Finset.prod_const_one, ENNReal.ofReal_one, Real.volume_Icc, one_smul, sub_zero]

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: remove this instance?

instance isAddHaarMeasure_volume_pi (ι : Type*) [Fintype ι] :
    IsAddHaarMeasure (volume : Measure (ι → ℝ)) :=
  inferInstance


/-- If a set is disjoint of its translates by infinitely many bounded vectors, then it has measure
zero. This auxiliary lemma proves this assuming additionally that the set is bounded. -/
theorem addHaar_eq_zero_of_disjoint_translates_aux {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] (μ : Measure E)
    [IsAddHaarMeasure μ] {s : Set E} (u : ℕ → E) (sb : IsBounded s) (hu : IsBounded (range u))
    (hs : Pairwise (Disjoint on fun n => {u n} + s)) (h's : MeasurableSet s) : μ s = 0 := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    u : Nat → E
    sb : Bornology.IsBounded s
    hu : Bornology.IsBounded (Set.range u)
    hs : Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton …
    h's : MeasurableSet s
    ⊢ Eq (μ s) 0
  -/
  by_contra h
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    u : Nat → E
    sb : Bornology.IsBounded s
    hu : Bornology.IsBounded (Set.range u)
    hs : Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton …
    h's : MeasurableSet s
    h : Not (Eq (μ s) 0)
    ⊢ False
  -/
  apply lt_irrefl ∞
  calc
    ∞ = ∑' _ : ℕ, μ s := (ENNReal.tsum_const_eq_top_of_ne_zero h).symm
    _ = ∑' n : ℕ, μ ({u n} + s) := by
      congr 1; ext1 n; simp only [image_add_left, measure_preimage_add, singleton_add]
    _ = μ (⋃ n, {u n} + s) := Eq.symm <| measure_iUnion hs fun n => by
      simpa only [image_add_left, singleton_add] using measurable_id.const_add _ h's
    _ = μ (range u + s) := by rw [← iUnion_add, iUnion_singleton_eq_range]
    _ < ∞ := (hu.add sb).measure_lt_top


/-- If a set is disjoint of its translates by infinitely many bounded vectors, then it has measure
zero. -/
theorem addHaar_eq_zero_of_disjoint_translates {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] (μ : Measure E)
    [IsAddHaarMeasure μ] {s : Set E} (u : ℕ → E) (hu : IsBounded (range u))
    (hs : Pairwise (Disjoint on fun n => {u n} + s)) (h's : MeasurableSet s) : μ s = 0 := by
  suffices H : ∀ R, μ (s ∩ closedBall 0 R) = 0 by
    apply le_antisymm _ (zero_le _)
    calc
      μ s ≤ ∑' n : ℕ, μ (s ∩ closedBall 0 n) := by
        conv_lhs => rw [← iUnion_inter_closedBall_nat s 0]
        exact measure_iUnion_le _
      _ = 0 := by simp only [H, tsum_zero]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    u : Nat → E
    hu : Bornology.IsBounded (Set.range u)
    hs : Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton …
    h's : MeasurableSet s
    ⊢ ∀ (R : Real), Eq (μ (Inter.inter s (Metric.closedBall 0 R))) 0
  -/
  intro R
  apply addHaar_eq_zero_of_disjoint_translates_aux μ u
    (isBounded_closedBall.subset inter_subset_right) hu _ (h's.inter measurableSet_closedBall)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    u : Nat → E
    hu : Bornology.IsBounded (Set.range u)
    hs : Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton …
    h's : MeasurableSet s
    R : Real
    ⊢ Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton (u …
  -/
  refine pairwise_disjoint_mono hs fun n => ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    u : Nat → E
    hu : Bornology.IsBounded (Set.range u)
    hs : Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton …
    h's : MeasurableSet s
    R : Real
    n : Nat
    ⊢ LE.le (HAdd.hAdd (Singleton.singleton (u n)) (Inter.inter s (Metric.closedBa …
  -/
  exact add_subset_add Subset.rfl inter_subset_left
  /-
    🎉 no goals
  -/


/-- A strict vector subspace has measure zero. -/
theorem addHaar_submodule {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [MeasurableSpace E]
    [BorelSpace E] [FiniteDimensional ℝ E] (μ : Measure E) [IsAddHaarMeasure μ] (s : Submodule ℝ E)
    (hs : s ≠ ⊤) : μ s = 0 := by
  obtain ⟨x, hx⟩ : ∃ x, x ∉ s := by
    simpa only [Submodule.eq_top_iff', not_exists, Ne, not_forall] using hs
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Submodule Real E
    hs : Ne s Top.top
    x : E
    hx : Not (Membership.mem s x)
    ⊢ Eq (μ ↑s) 0
  -/
  obtain ⟨c, cpos, cone⟩ : ∃ c : ℝ, 0 < c ∧ c < 1 := ⟨1 / 2, by norm_num, by norm_num⟩
  have A : IsBounded (range fun n : ℕ => c ^ n • x) :=
    have : Tendsto (fun n : ℕ => c ^ n • x) atTop (𝓝 ((0 : ℝ) • x)) :=
      (tendsto_pow_atTop_nhds_zero_of_lt_one cpos.le cone).smul_const x
    isBounded_range_of_tendsto _ this
  apply addHaar_eq_zero_of_disjoint_translates μ _ A _
    (Submodule.closed_of_finiteDimensional s).measurableSet
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Submodule Real E
    hs : Ne s Top.top
    x : E
    hx : Not (Membership.mem s x)
    c : Real
    cpos : LT.lt 0 c
    cone : LT.lt c 1
    A : Bornology.IsBounded (Set.range fun n => HSMul.hSMul (HPow.hPow c n) x)
    ⊢ Pairwise (Function.onFun Disjoint fun n => HAdd.hAdd (Singleton.singleton (H …
  -/
  intro m n hmn
  simp only [Function.onFun, image_add_left, singleton_add, disjoint_left, mem_preimage,
    SetLike.mem_coe]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Submodule Real E
    hs : Ne s Top.top
    x : E
    hx : Not (Membership.mem s x)
    c : Real
    cpos : LT.lt 0 c
    cone : LT.lt c 1
    A : Bornology.IsBounded (Set.range fun n => HSMul.hSMul (HPow.hPow c n) x)
    m n : Nat
    hmn : Ne m n
    ⊢ ∀ ⦃a : E⦄, Membership.mem s (HAdd.hAdd (Neg.neg (HSMul.hSMul (HPow.hPow c m) …
  -/
  intro y hym hyn
  have A : (c ^ n - c ^ m) • x ∈ s := by
    convert s.sub_mem hym hyn using 1
    simp only [sub_smul, neg_sub_neg, add_sub_add_right_eq_sub]
  have H : c ^ n - c ^ m ≠ 0 := by
    simpa only [sub_eq_zero, Ne] using (pow_right_strictAnti₀ cpos cone).injective.ne hmn.symm
  have : x ∈ s := by
    convert s.smul_mem (c ^ n - c ^ m)⁻¹ A
    rw [smul_smul, inv_mul_cancel₀ H, one_smul]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Submodule Real E
    hs : Ne s Top.top
    x : E
    hx : Not (Membership.mem s x)
    c : Real
    cpos : LT.lt 0 c
    cone : LT.lt c 1
    A✝ : Bornology.IsBounded (Set.range fun n => HSMul.hSMul (HPow.hPow c n) x)
    m n : Nat
    hmn : Ne m n
    y : E
    hym : Membership.mem s (HAdd.hAdd (Neg.neg (HSMul.hSMul (HPow.hPow c m) x)) y)
    hyn : Membership.mem s (HAdd.hAdd (Neg.neg (HSMul.hSMul (HPow.hPow c n) x)) y)
    A : Membership.mem s (HSMul.hSMul (HSub.hSub (HPow.hPow c n) (HPow.hPow c m)) x)
    H : Ne (HSub.hSub (HPow.hPow c n) (HPow.hPow c m)) 0
    this : Membership.mem s x
    ⊢ False
  -/
  exact hx this
  /-
    🎉 no goals
  -/


/-- A strict affine subspace has measure zero. -/
theorem addHaar_affineSubspace {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] (μ : Measure E) [IsAddHaarMeasure μ]
    (s : AffineSubspace ℝ E) (hs : s ≠ ⊤) : μ s = 0 := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : AffineSubspace Real E
    hs : Ne s Top.top
    ⊢ Eq (μ ↑s) 0
  -/
  rcases s.eq_bot_or_nonempty with (rfl | hne)
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hs : Ne Bot.bot Top.top
      ⊢ Eq (μ ↑Bot.bot) 0
    -/
  · rw [AffineSubspace.bot_coe, measure_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : AffineSubspace Real E
    hs : Ne s Top.top
    hne : (↑s).Nonempty
    ⊢ Eq (μ ↑s) 0
  -/
  rw [Ne, ← AffineSubspace.direction_eq_top_iff_of_nonempty hne] at hs
  /-
    case inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : AffineSubspace Real E
    hs : Not (Eq s.direction Top.top)
    hne : (↑s).Nonempty
    ⊢ Eq (μ ↑s) 0
  -/
  rcases hne with ⟨x, hx : x ∈ s⟩
  simpa only [AffineSubspace.coe_direction_eq_vsub_set_right hx, vsub_eq_sub, sub_eq_add_neg,
    image_add_right, neg_neg, measure_preimage_add_right] using addHaar_submodule μ s.direction hs


theorem map_linearMap_addHaar_pi_eq_smul_addHaar {ι : Type*} [Finite ι] {f : (ι → ℝ) →ₗ[ℝ] ι → ℝ}
    (hf : LinearMap.det f ≠ 0) (μ : Measure (ι → ℝ)) [IsAddHaarMeasure μ] :
    Measure.map f μ = ENNReal.ofReal (abs (LinearMap.det f)⁻¹) • μ := by
  /-
    ι : Type u_1
    inst✝¹ : Finite ι
    f : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det f) 0
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  cases nonempty_fintype ι
  /- We have already proved the result for the Lebesgue product measure, using matrices.
    We deduce it for any Haar measure by uniqueness (up to scalar multiplication). -/
  /-
    case intro
    ι : Type u_1
    inst✝¹ : Finite ι
    f : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det f) 0
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : μ.IsAddHaarMeasure
    val✝ : Fintype ι
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  have := addHaarMeasure_unique μ (piIcc01 ι)
  rw [this, addHaarMeasure_eq_volume_pi, Measure.map_smul,
    Real.map_linearMap_volume_pi_eq_smul_volume_pi hf, smul_comm]


theorem map_linearMap_addHaar_eq_smul_addHaar {f : E →ₗ[ℝ] E} (hf : LinearMap.det f ≠ 0) :
    Measure.map f μ = ENNReal.ofReal |(LinearMap.det f)⁻¹| • μ := by
  -- we reduce to the case of `E = ι → ℝ`, for which we have already proved the result using
  -- matrices in `map_linearMap_addHaar_pi_eq_smul_addHaar`.
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  let ι := Fin (finrank ℝ E)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ι : Type := Fin (Module.finrank Real E)
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  haveI : FiniteDimensional ℝ (ι → ℝ) := by infer_instance
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ι : Type := Fin (Module.finrank Real E)
    this : FiniteDimensional Real (ι → Real)
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  have : finrank ℝ E = finrank ℝ (ι → ℝ) := by simp [ι]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  have e : E ≃ₗ[ℝ] ι → ℝ := LinearEquiv.ofFinrankEq E (ι → ℝ) this
  -- next line is to avoid `g` getting reduced by `simp`.
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  obtain ⟨g, hg⟩ : ∃ g, g = (e : E →ₗ[ℝ] ι → ℝ).comp (f.comp (e.symm : (ι → ℝ) →ₗ[ℝ] E)) := ⟨_, rfl⟩
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  have gdet : LinearMap.det g = LinearMap.det f := by rw [hg]; exact LinearMap.det_conj f e
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  rw [← gdet] at hf ⊢
  have fg : f = (e.symm : (ι → ℝ) →ₗ[ℝ] E).comp (g.comp (e : E →ₗ[ℝ] ι → ℝ)) := by
    ext x
    simp only [LinearEquiv.coe_coe, Function.comp_apply, LinearMap.coe_comp,
      LinearEquiv.symm_apply_apply, hg]
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det g) 0
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    fg : Eq f ((↑e.symm).comp (g.comp ↑e))
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  simp only [fg, LinearEquiv.coe_coe, LinearMap.coe_comp]
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det g) 0
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    fg : Eq f ((↑e.symm).comp (g.comp ↑e))
    ⊢ Eq (MeasureTheory.Measure.map (Function.comp (⇑e.symm) (Function.comp ⇑g ⇑e) …
  -/
  have Ce : Continuous e := (e : E →ₗ[ℝ] ι → ℝ).continuous_of_finiteDimensional
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det g) 0
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    fg : Eq f ((↑e.symm).comp (g.comp ↑e))
    Ce : Continuous ⇑e
    ⊢ Eq (MeasureTheory.Measure.map (Function.comp (⇑e.symm) (Function.comp ⇑g ⇑e) …
  -/
  have Cg : Continuous g := LinearMap.continuous_of_finiteDimensional g
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det g) 0
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    fg : Eq f ((↑e.symm).comp (g.comp ↑e))
    Ce : Continuous ⇑e
    Cg : Continuous ⇑g
    ⊢ Eq (MeasureTheory.Measure.map (Function.comp (⇑e.symm) (Function.comp ⇑g ⇑e) …
  -/
  have Cesymm : Continuous e.symm := (e.symm : (ι → ℝ) →ₗ[ℝ] E).continuous_of_finiteDimensional
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det g) 0
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    fg : Eq f ((↑e.symm).comp (g.comp ↑e))
    Ce : Continuous ⇑e
    Cg : Continuous ⇑g
    Cesymm : Continuous ⇑e.symm
    ⊢ Eq (MeasureTheory.Measure.map (Function.comp (⇑e.symm) (Function.comp ⇑g ⇑e) …
  -/
  rw [← map_map Cesymm.measurable (Cg.comp Ce).measurable, ← map_map Cg.measurable Ce.measurable]
  /-
    case intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    ι : Type := Fin (Module.finrank Real E)
    this✝ : FiniteDimensional Real (ι → Real)
    this : Eq (Module.finrank Real E) (Module.finrank Real (ι → Real))
    e : LinearEquiv (RingHom.id Real) E (ι → Real)
    g : LinearMap (RingHom.id Real) (ι → Real) (ι → Real)
    hf : Ne (LinearMap.det g) 0
    hg : Eq g ((↑e).comp (f.comp ↑e.symm))
    gdet : Eq (LinearMap.det g) (LinearMap.det f)
    fg : Eq f ((↑e.symm).comp (g.comp ↑e))
    Ce : Continuous ⇑e
    Cg : Continuous ⇑g
    Cesymm : Continuous ⇑e.symm
    ⊢ Eq (MeasureTheory.Measure.map (⇑e.symm) (MeasureTheory.Measure.map (⇑g) (Mea …
  -/
  haveI : IsAddHaarMeasure (map e μ) := (e : E ≃+ (ι → ℝ)).isAddHaarMeasure_map μ Ce Cesymm
  have ecomp : e.symm ∘ e = id := by
    ext x; simp only [id, Function.comp_apply, LinearEquiv.symm_apply_apply]
  rw [map_linearMap_addHaar_pi_eq_smul_addHaar hf (map e μ), Measure.map_smul,
    map_map Cesymm.measurable Ce.measurable, ecomp, Measure.map_id]


/-- The preimage of a set `s` under a linear map `f` with nonzero determinant has measure
equal to `μ s` times the absolute value of the inverse of the determinant of `f`. -/
@[simp]
theorem addHaar_preimage_linearMap {f : E →ₗ[ℝ] E} (hf : LinearMap.det f ≠ 0) (s : Set E) :
    μ (f ⁻¹' s) = ENNReal.ofReal |(LinearMap.det f)⁻¹| * μ s :=
  calc
    μ (f ⁻¹' s) = Measure.map f μ s :=
      ((f.equivOfDetNeZero hf).toContinuousLinearEquiv.toHomeomorph.toMeasurableEquiv.map_apply
          s).symm
    _ = ENNReal.ofReal |(LinearMap.det f)⁻¹| * μ s := by
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        f : LinearMap (RingHom.id Real) E E
        hf : Ne (LinearMap.det f) 0
        s : Set E
        ⊢ Eq ((MeasureTheory.Measure.map (⇑f) μ) s) (HMul.hMul (ENNReal.ofReal (abs (I …
      -/
      rw [map_linearMap_addHaar_eq_smul_addHaar μ hf]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The preimage of a set `s` under a continuous linear map `f` with nonzero determinant has measure
equal to `μ s` times the absolute value of the inverse of the determinant of `f`. -/
@[simp]
theorem addHaar_preimage_continuousLinearMap {f : E →L[ℝ] E}
    (hf : LinearMap.det (f : E →ₗ[ℝ] E) ≠ 0) (s : Set E) :
    μ (f ⁻¹' s) = ENNReal.ofReal (abs (LinearMap.det (f : E →ₗ[ℝ] E))⁻¹) * μ s :=
  addHaar_preimage_linearMap μ hf s


/-- The preimage of a set `s` under a linear equiv `f` has measure
equal to `μ s` times the absolute value of the inverse of the determinant of `f`. -/
@[simp]
theorem addHaar_preimage_linearEquiv (f : E ≃ₗ[ℝ] E) (s : Set E) :
    μ (f ⁻¹' s) = ENNReal.ofReal |LinearMap.det (f.symm : E →ₗ[ℝ] E)| * μ s := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearEquiv (RingHom.id Real) E E
    s : Set E
    ⊢ Eq (μ (Set.preimage (⇑f) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det  …
  -/
  have A : LinearMap.det (f : E →ₗ[ℝ] E) ≠ 0 := (LinearEquiv.isUnit_det' f).ne_zero
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearEquiv (RingHom.id Real) E E
    s : Set E
    A : Ne (LinearMap.det ↑f) 0
    ⊢ Eq (μ (Set.preimage (⇑f) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det  …
  -/
  convert addHaar_preimage_linearMap μ A s
  /-
    case h.e'_3.h.e'_5.h.e'_1.h.e'_4
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearEquiv (RingHom.id Real) E E
    s : Set E
    A : Ne (LinearMap.det ↑f) 0
    ⊢ Eq (LinearMap.det ↑f.symm) (Inv.inv (LinearMap.det ↑f))
  -/
  simp only [LinearEquiv.det_coe_symm]
  /-
    🎉 no goals
  -/


/-- The preimage of a set `s` under a continuous linear equiv `f` has measure
equal to `μ s` times the absolute value of the inverse of the determinant of `f`. -/
@[simp]
theorem addHaar_preimage_continuousLinearEquiv (f : E ≃L[ℝ] E) (s : Set E) :
    μ (f ⁻¹' s) = ENNReal.ofReal |LinearMap.det (f.symm : E →ₗ[ℝ] E)| * μ s :=
  addHaar_preimage_linearEquiv μ _ s


/-- The image of a set `s` under a linear map `f` has measure
equal to `μ s` times the absolute value of the determinant of `f`. -/
@[simp]
theorem addHaar_image_linearMap (f : E →ₗ[ℝ] E) (s : Set E) :
    μ (f '' s) = ENNReal.ofReal |LinearMap.det f| * μ s := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    s : Set E
    ⊢ Eq (μ (Set.image (⇑f) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det f)) …
  -/
  rcases ne_or_eq (LinearMap.det f) 0 with (hf | hf)
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      f : LinearMap (RingHom.id Real) E E
      s : Set E
      hf : Ne (LinearMap.det f) 0
      ⊢ Eq (μ (Set.image (⇑f) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det f)) …
    -/
  · let g := (f.equivOfDetNeZero hf).toContinuousLinearEquiv
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      f : LinearMap (RingHom.id Real) E E
      s : Set E
      hf : Ne (LinearMap.det f) 0
      g : ContinuousLinearEquiv (RingHom.id Real) E E := (f.equivOfDetNeZero hf).toC …
      ⊢ Eq (μ (Set.image (⇑f) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det f)) …
    -/
    change μ (g '' s) = _
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      f : LinearMap (RingHom.id Real) E E
      s : Set E
      hf : Ne (LinearMap.det f) 0
      g : ContinuousLinearEquiv (RingHom.id Real) E E := (f.equivOfDetNeZero hf).toC …
      ⊢ Eq (μ (Set.image (⇑g) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det f)) …
    -/
    rw [ContinuousLinearEquiv.image_eq_preimage g s, addHaar_preimage_continuousLinearEquiv]
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      f : LinearMap (RingHom.id Real) E E
      s : Set E
      hf : Ne (LinearMap.det f) 0
      g : ContinuousLinearEquiv (RingHom.id Real) E E := (f.equivOfDetNeZero hf).toC …
      ⊢ Eq (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det ↑↑g.symm.symm))) (μ s)) (H …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      f : LinearMap (RingHom.id Real) E E
      s : Set E
      hf : Eq (LinearMap.det f) 0
      ⊢ Eq (μ (Set.image (⇑f) s)) (HMul.hMul (ENNReal.ofReal (abs (LinearMap.det f)) …
    -/
  · simp only [hf, zero_mul, ENNReal.ofReal_zero, abs_zero]
    have : μ (LinearMap.range f) = 0 :=
      addHaar_submodule μ _ (LinearMap.range_lt_top_of_det_eq_zero hf).ne
    /-
      case inr
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      f : LinearMap (RingHom.id Real) E E
      s : Set E
      hf : Eq (LinearMap.det f) 0
      this : Eq (μ ↑(LinearMap.range f)) 0
      ⊢ Eq (μ (Set.image (⇑f) s)) 0
    -/
    exact le_antisymm (le_trans (measure_mono (image_subset_range _ _)) this.le) (zero_le _)
    /-
      🎉 no goals
    -/


/-- The image of a set `s` under a continuous linear map `f` has measure
equal to `μ s` times the absolute value of the determinant of `f`. -/
@[simp]
theorem addHaar_image_continuousLinearMap (f : E →L[ℝ] E) (s : Set E) :
    μ (f '' s) = ENNReal.ofReal |LinearMap.det (f : E →ₗ[ℝ] E)| * μ s :=
  addHaar_image_linearMap μ _ s


/-- The image of a set `s` under a continuous linear equiv `f` has measure
equal to `μ s` times the absolute value of the determinant of `f`. -/
@[simp]
theorem addHaar_image_continuousLinearEquiv (f : E ≃L[ℝ] E) (s : Set E) :
    μ (f '' s) = ENNReal.ofReal |LinearMap.det (f : E →ₗ[ℝ] E)| * μ s :=
  μ.addHaar_image_linearMap (f : E →ₗ[ℝ] E) s


theorem LinearMap.quasiMeasurePreserving (f : E →ₗ[ℝ] E) (hf : LinearMap.det f ≠ 0) :
    QuasiMeasurePreserving f μ μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (⇑f) μ μ
  -/
  refine ⟨f.continuous_of_finiteDimensional.measurable, ?_⟩
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ⊢ (MeasureTheory.Measure.map (⇑f) μ).AbsolutelyContinuous μ
  -/
  rw [map_linearMap_addHaar_eq_smul_addHaar μ hf]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : LinearMap (RingHom.id Real) E E
    hf : Ne (LinearMap.det f) 0
    ⊢ (HSMul.hSMul (ENNReal.ofReal (abs (Inv.inv (LinearMap.det f)))) μ).Absolutel …
  -/
  exact smul_absolutelyContinuous
  /-
    🎉 no goals
  -/


theorem ContinuousLinearMap.quasiMeasurePreserving (f : E →L[ℝ] E) (hf : f.det ≠ 0) :
    QuasiMeasurePreserving f μ μ :=
  LinearMap.quasiMeasurePreserving μ (f : E →ₗ[ℝ] E) hf


theorem map_addHaar_smul {r : ℝ} (hr : r ≠ 0) :
    Measure.map (r • ·) μ = ENNReal.ofReal (abs (r ^ finrank ℝ E)⁻¹) • μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : Ne r 0
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HSMul.hSMul r x) μ) (HSMul.hSMul (EN …
  -/
  let f : E →ₗ[ℝ] E := r • (1 : E →ₗ[ℝ] E)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : Ne r 0
    f : LinearMap (RingHom.id Real) E E := HSMul.hSMul r 1
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HSMul.hSMul r x) μ) (HSMul.hSMul (EN …
  -/
  change Measure.map f μ = _
  have hf : LinearMap.det f ≠ 0 := by
    simp only [f, mul_one, LinearMap.det_smul, Ne, MonoidHom.map_one]
    intro h
    exact hr (pow_eq_zero h)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : Ne r 0
    f : LinearMap (RingHom.id Real) E E := HSMul.hSMul r 1
    hf : Ne (LinearMap.det f) 0
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) μ) (HSMul.hSMul (ENNReal.ofReal (abs (Inv …
  -/
  simp only [f, map_linearMap_addHaar_eq_smul_addHaar μ hf, mul_one, LinearMap.det_smul, map_one]
  /-
    🎉 no goals
  -/


theorem quasiMeasurePreserving_smul {r : ℝ} (hr : r ≠ 0) :
    QuasiMeasurePreserving (r • ·) μ μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : Ne r 0
    ⊢ MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMul.hSMul r x) μ μ
  -/
  refine ⟨measurable_const_smul r, ?_⟩
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : Ne r 0
    ⊢ (MeasureTheory.Measure.map (fun x => HSMul.hSMul r x) μ).AbsolutelyContinuou …
  -/
  rw [map_addHaar_smul μ hr]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : Ne r 0
    ⊢ (HSMul.hSMul (ENNReal.ofReal (abs (Inv.inv (HPow.hPow r (Module.finrank Real …
  -/
  exact smul_absolutelyContinuous
  /-
    🎉 no goals
  -/


@[simp]
theorem addHaar_preimage_smul {r : ℝ} (hr : r ≠ 0) (s : Set E) :
    μ ((r • ·) ⁻¹' s) = ENNReal.ofReal (abs (r ^ finrank ℝ E)⁻¹) * μ s :=
  calc
    μ ((r • ·) ⁻¹' s) = Measure.map (r • ·) μ s :=
      ((Homeomorph.smul (isUnit_iff_ne_zero.2 hr).unit).toMeasurableEquiv.map_apply s).symm
    _ = ENNReal.ofReal (abs (r ^ finrank ℝ E)⁻¹) * μ s := by
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        r : Real
        hr : Ne r 0
        s : Set E
        ⊢ Eq ((MeasureTheory.Measure.map (fun x => HSMul.hSMul r x) μ) s) (HMul.hMul ( …
      -/
      rw [map_addHaar_smul μ hr, coe_smul, Pi.smul_apply, smul_eq_mul]
      /-
        🎉 no goals
      -/


/-- Rescaling a set by a factor `r` multiplies its measure by `abs (r ^ dim)`. -/
@[simp]
theorem addHaar_smul (r : ℝ) (s : Set E) :
    μ (r • s) = ENNReal.ofReal (abs (r ^ finrank ℝ E)) * μ s := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    s : Set E
    ⊢ Eq (μ (HSMul.hSMul r s)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow r (Modul …
  -/
  rcases ne_or_eq r 0 with (h | rfl)
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      r : Real
      s : Set E
      h : Ne r 0
      ⊢ Eq (μ (HSMul.hSMul r s)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow r (Modul …
    -/
  · rw [← preimage_smul_inv₀ h, addHaar_preimage_smul μ (inv_ne_zero h), inv_pow, inv_inv]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    ⊢ Eq (μ (HSMul.hSMul 0 s)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow 0 (Modul …
  -/
  rcases eq_empty_or_nonempty s with (rfl | hs)
    /-
      case inr.inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      ⊢ Eq (μ (HSMul.hSMul 0 EmptyCollection.emptyCollection)) (HMul.hMul (ENNReal.o …
    -/
  · simp only [measure_empty, mul_zero, smul_set_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : s.Nonempty
    ⊢ Eq (μ (HSMul.hSMul 0 s)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow 0 (Modul …
  -/
  rw [zero_smul_set hs, ← singleton_zero]
  /-
    case inr.inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : s.Nonempty
    ⊢ Eq (μ (Singleton.singleton 0)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow 0  …
  -/
  by_cases h : finrank ℝ E = 0
    /-
      case pos
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      s : Set E
      hs : s.Nonempty
      h : Eq (Module.finrank Real E) 0
      ⊢ Eq (μ (Singleton.singleton 0)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow 0  …
    -/
  · haveI : Subsingleton E := finrank_zero_iff.1 h
    simp only [h, one_mul, ENNReal.ofReal_one, abs_one, Subsingleton.eq_univ_of_nonempty hs,
      pow_zero, Subsingleton.eq_univ_of_nonempty (singleton_nonempty (0 : E))]
    /-
      case neg
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      s : Set E
      hs : s.Nonempty
      h : Not (Eq (Module.finrank Real E) 0)
      ⊢ Eq (μ (Singleton.singleton 0)) (HMul.hMul (ENNReal.ofReal (abs (HPow.hPow 0  …
    -/
  · haveI : Nontrivial E := nontrivial_of_finrank_pos (bot_lt_iff_ne_bot.2 h)
    simp only [h, zero_mul, ENNReal.ofReal_zero, abs_zero, Ne, not_false_iff,
      zero_pow, measure_singleton]


theorem addHaar_smul_of_nonneg {r : ℝ} (hr : 0 ≤ r) (s : Set E) :
    μ (r • s) = ENNReal.ofReal (r ^ finrank ℝ E) * μ s := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hr : LE.le 0 r
    s : Set E
    ⊢ Eq (μ (HSMul.hSMul r s)) (HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fin …
  -/
  rw [addHaar_smul, abs_pow, abs_of_nonneg hr]
  /-
    🎉 no goals
  -/


theorem NullMeasurableSet.const_smul (hs : NullMeasurableSet s μ) (r : ℝ) :
    NullMeasurableSet (r • s) μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasureTheory.NullMeasurableSet s μ
    r : Real
    ⊢ MeasureTheory.NullMeasurableSet (HSMul.hSMul r s) μ
  -/
  obtain rfl | hs' := s.eq_empty_or_nonempty
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      r : Real
      hs : MeasureTheory.NullMeasurableSet EmptyCollection.emptyCollection μ
      ⊢ MeasureTheory.NullMeasurableSet (HSMul.hSMul r EmptyCollection.emptyCollecti …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasureTheory.NullMeasurableSet s μ
    r : Real
    hs' : s.Nonempty
    ⊢ MeasureTheory.NullMeasurableSet (HSMul.hSMul r s) μ
  -/
  obtain rfl | hr := eq_or_ne r 0
    /-
      case inr.inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      s : Set E
      hs : MeasureTheory.NullMeasurableSet s μ
      hs' : s.Nonempty
      ⊢ MeasureTheory.NullMeasurableSet (HSMul.hSMul 0 s) μ
    -/
  · simpa [zero_smul_set hs'] using nullMeasurableSet_singleton _
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasureTheory.NullMeasurableSet s μ
    r : Real
    hs' : s.Nonempty
    hr : Ne r 0
    ⊢ MeasureTheory.NullMeasurableSet (HSMul.hSMul r s) μ
  -/
  obtain ⟨t, ht, hst⟩ := hs
  /-
    case inr.inr.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    r : Real
    hs' : s.Nonempty
    hr : Ne r 0
    t : Set E
    ht : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ MeasureTheory.NullMeasurableSet (HSMul.hSMul r s) μ
  -/
  refine ⟨_, ht.const_smul_of_ne_zero hr, ?_⟩
  /-
    case inr.inr.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    r : Real
    hs' : s.Nonempty
    hr : Ne r 0
    t : Set E
    ht : MeasurableSet t
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul r s) (HSMul.hSMul r t)
  -/
  rw [← measure_symmDiff_eq_zero_iff] at hst ⊢
  /-
    case inr.inr.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    r : Real
    hs' : s.Nonempty
    hr : Ne r 0
    t : Set E
    ht : MeasurableSet t
    hst : Eq (μ (symmDiff s t)) 0
    ⊢ Eq (μ (symmDiff (HSMul.hSMul r s) (HSMul.hSMul r t))) 0
  -/
  rw [← smul_set_symmDiff₀ hr, addHaar_smul μ, hst, mul_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem addHaar_image_homothety (x : E) (r : ℝ) (s : Set E) :
    μ (AffineMap.homothety x r '' s) = ENNReal.ofReal (abs (r ^ finrank ℝ E)) * μ s :=
  calc
    μ (AffineMap.homothety x r '' s) = μ ((fun y => y + x) '' (r • (fun y => y + -x) '' s)) := by
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        x : E
        r : Real
        s : Set E
        ⊢ Eq (μ (Set.image (⇑(AffineMap.homothety x r)) s)) (μ (Set.image (fun y => HA …
      -/
      simp only [← image_smul, image_image, ← sub_eq_add_neg]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/
    _ = ENNReal.ofReal (abs (r ^ finrank ℝ E)) * μ s := by
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        x : E
        r : Real
        s : Set E
        ⊢ Eq (μ (Set.image (fun y => HAdd.hAdd y x) (HSMul.hSMul r (Set.image (fun y = …
      -/
      simp only [image_add_right, measure_preimage_add_right, addHaar_smul]
      /-
        🎉 no goals
      -/


theorem addHaar_ball_center {E : Type*} [NormedAddCommGroup E] [MeasurableSpace E] [BorelSpace E]
    (μ : Measure E) [IsAddHaarMeasure μ] (x : E) (r : ℝ) : μ (ball x r) = μ (ball (0 : E) r) := by
  /-
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    ⊢ Eq (μ (Metric.ball x r)) (μ (Metric.ball 0 r))
  -/
  have : ball (0 : E) r = (x + ·) ⁻¹' ball x r := by simp [preimage_add_ball]
  /-
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    this : Eq (Metric.ball 0 r) (Set.preimage (fun x_1 => HAdd.hAdd x x_1) (Metric …
    ⊢ Eq (μ (Metric.ball x r)) (μ (Metric.ball 0 r))
  -/
  rw [this, measure_preimage_add]
  /-
    🎉 no goals
  -/


theorem addHaar_closedBall_center {E : Type*} [NormedAddCommGroup E] [MeasurableSpace E]
    [BorelSpace E] (μ : Measure E) [IsAddHaarMeasure μ] (x : E) (r : ℝ) :
    μ (closedBall x r) = μ (closedBall (0 : E) r) := by
  /-
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    ⊢ Eq (μ (Metric.closedBall x r)) (μ (Metric.closedBall 0 r))
  -/
  have : closedBall (0 : E) r = (x + ·) ⁻¹' closedBall x r := by simp [preimage_add_closedBall]
  /-
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    this : Eq (Metric.closedBall 0 r) (Set.preimage (fun x_1 => HAdd.hAdd x x_1) ( …
    ⊢ Eq (μ (Metric.closedBall x r)) (μ (Metric.closedBall 0 r))
  -/
  rw [this, measure_preimage_add]
  /-
    🎉 no goals
  -/


theorem addHaar_ball_mul_of_pos (x : E) {r : ℝ} (hr : 0 < r) (s : ℝ) :
    μ (ball x (r * s)) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (ball 0 s) := by
  have : ball (0 : E) (r * s) = r • ball (0 : E) s := by
    simp only [_root_.smul_ball hr.ne' (0 : E) s, Real.norm_eq_abs, abs_of_nonneg hr.le, smul_zero]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : LT.lt 0 r
    s : Real
    this : Eq (Metric.ball 0 (HMul.hMul r s)) (HSMul.hSMul r (Metric.ball 0 s))
    ⊢ Eq (μ (Metric.ball x (HMul.hMul r s))) (HMul.hMul (ENNReal.ofReal (HPow.hPow …
  -/
  simp only [this, addHaar_smul, abs_of_nonneg hr.le, addHaar_ball_center, abs_pow]
  /-
    🎉 no goals
  -/


theorem addHaar_ball_of_pos (x : E) {r : ℝ} (hr : 0 < r) :
    μ (ball x r) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (ball 0 1) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (μ (Metric.ball x r)) (HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fin …
  -/
  rw [← addHaar_ball_mul_of_pos μ x hr, mul_one]
  /-
    🎉 no goals
  -/


theorem addHaar_ball_mul [Nontrivial E] (x : E) {r : ℝ} (hr : 0 ≤ r) (s : ℝ) :
    μ (ball x (r * s)) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (ball 0 s) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    inst✝ : Nontrivial E
    x : E
    r : Real
    hr : LE.le 0 r
    s : Real
    ⊢ Eq (μ (Metric.ball x (HMul.hMul r s))) (HMul.hMul (ENNReal.ofReal (HPow.hPow …
  -/
  rcases hr.eq_or_lt with (rfl | h)
  · simp only [zero_pow (finrank_pos (R := ℝ) (M := E)).ne', measure_empty, zero_mul,
      ENNReal.ofReal_zero, ball_zero]
    /-
      case inr
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝¹ : μ.IsAddHaarMeasure
      inst✝ : Nontrivial E
      x : E
      r : Real
      hr : LE.le 0 r
      s : Real
      h : LT.lt 0 r
      ⊢ Eq (μ (Metric.ball x (HMul.hMul r s))) (HMul.hMul (ENNReal.ofReal (HPow.hPow …
    -/
  · exact addHaar_ball_mul_of_pos μ x h s
    /-
      🎉 no goals
    -/


theorem addHaar_ball [Nontrivial E] (x : E) {r : ℝ} (hr : 0 ≤ r) :
    μ (ball x r) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (ball 0 1) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    inst✝ : Nontrivial E
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (μ (Metric.ball x r)) (HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fin …
  -/
  rw [← addHaar_ball_mul μ x hr, mul_one]
  /-
    🎉 no goals
  -/


theorem addHaar_closedBall_mul_of_pos (x : E) {r : ℝ} (hr : 0 < r) (s : ℝ) :
    μ (closedBall x (r * s)) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (closedBall 0 s) := by
  have : closedBall (0 : E) (r * s) = r • closedBall (0 : E) s := by
    simp [smul_closedBall' hr.ne' (0 : E), abs_of_nonneg hr.le]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : LT.lt 0 r
    s : Real
    this : Eq (Metric.closedBall 0 (HMul.hMul r s)) (HSMul.hSMul r (Metric.closedB …
    ⊢ Eq (μ (Metric.closedBall x (HMul.hMul r s))) (HMul.hMul (ENNReal.ofReal (HPo …
  -/
  simp only [this, addHaar_smul, abs_of_nonneg hr.le, addHaar_closedBall_center, abs_pow]
  /-
    🎉 no goals
  -/


theorem addHaar_closedBall_mul (x : E) {r : ℝ} (hr : 0 ≤ r) {s : ℝ} (hs : 0 ≤ s) :
    μ (closedBall x (r * s)) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (closedBall 0 s) := by
  have : closedBall (0 : E) (r * s) = r • closedBall (0 : E) s := by
    simp [smul_closedBall r (0 : E) hs, abs_of_nonneg hr]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : LE.le 0 r
    s : Real
    hs : LE.le 0 s
    this : Eq (Metric.closedBall 0 (HMul.hMul r s)) (HSMul.hSMul r (Metric.closedB …
    ⊢ Eq (μ (Metric.closedBall x (HMul.hMul r s))) (HMul.hMul (ENNReal.ofReal (HPo …
  -/
  simp only [this, addHaar_smul, abs_of_nonneg hr, addHaar_closedBall_center, abs_pow]
  /-
    🎉 no goals
  -/


/-- The measure of a closed ball can be expressed in terms of the measure of the closed unit ball.
Use instead `addHaar_closedBall`, which uses the measure of the open unit ball as a standard
form. -/
theorem addHaar_closedBall' (x : E) {r : ℝ} (hr : 0 ≤ r) :
    μ (closedBall x r) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (closedBall 0 1) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (μ (Metric.closedBall x r)) (HMul.hMul (ENNReal.ofReal (HPow.hPow r (Modu …
  -/
  rw [← addHaar_closedBall_mul μ x hr zero_le_one, mul_one]
  /-
    🎉 no goals
  -/


theorem addHaar_unitClosedBall_eq_addHaar_unitBall :
    μ (closedBall (0 : E) 1) = μ (ball 0 1) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Eq (μ (Metric.closedBall 0 1)) (μ (Metric.ball 0 1))
  -/
  apply le_antisymm _ (measure_mono ball_subset_closedBall)
  have A : Tendsto
      (fun r : ℝ => ENNReal.ofReal (r ^ finrank ℝ E) * μ (closedBall (0 : E) 1)) (𝓝[<] 1)
        (𝓝 (ENNReal.ofReal ((1 : ℝ) ^ finrank ℝ E) * μ (closedBall (0 : E) 1))) := by
    refine ENNReal.Tendsto.mul ?_ (by simp) tendsto_const_nhds (by simp)
    exact ENNReal.tendsto_ofReal ((tendsto_id'.2 nhdsWithin_le_nhds).pow _)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : Filter.Tendsto (fun r => HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fi …
    ⊢ LE.le (μ (Metric.closedBall 0 1)) (μ (Metric.ball 0 1))
  -/
  simp only [one_pow, one_mul, ENNReal.ofReal_one] at A
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : Filter.Tendsto (fun r => HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fi …
    ⊢ LE.le (μ (Metric.closedBall 0 1)) (μ (Metric.ball 0 1))
  -/
  refine le_of_tendsto A ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : Filter.Tendsto (fun r => HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fi …
    ⊢ Filter.Eventually (fun c => LE.le (HMul.hMul (ENNReal.ofReal (HPow.hPow c (M …
  -/
  filter_upwards [Ioo_mem_nhdsLT zero_lt_one] with r hr
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : Filter.Tendsto (fun r => HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fi …
    r : Real
    hr : Membership.mem (Set.Ioo 0 1) r
    ⊢ LE.le (HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.finrank Real E))) (μ ( …
  -/
  rw [← addHaar_closedBall' μ (0 : E) hr.1.le]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : Filter.Tendsto (fun r => HMul.hMul (ENNReal.ofReal (HPow.hPow r (Module.fi …
    r : Real
    hr : Membership.mem (Set.Ioo 0 1) r
    ⊢ LE.le (μ (Metric.closedBall 0 r)) (μ (Metric.ball 0 1))
  -/
  exact measure_mono (closedBall_subset_ball hr.2)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-01")]
alias addHaar_closed_unit_ball_eq_addHaar_unit_ball := addHaar_unitClosedBall_eq_addHaar_unitBall


theorem addHaar_closedBall (x : E) {r : ℝ} (hr : 0 ≤ r) :
    μ (closedBall x r) = ENNReal.ofReal (r ^ finrank ℝ E) * μ (ball 0 1) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (μ (Metric.closedBall x r)) (HMul.hMul (ENNReal.ofReal (HPow.hPow r (Modu …
  -/
  rw [addHaar_closedBall' μ x hr, addHaar_unitClosedBall_eq_addHaar_unitBall]
  /-
    🎉 no goals
  -/


theorem addHaar_closedBall_eq_addHaar_ball [Nontrivial E] (x : E) (r : ℝ) :
    μ (closedBall x r) = μ (ball x r) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (μ (Metric.closedBall x r)) (μ (Metric.ball x r))
  -/
  by_cases h : r < 0
    /-
      case pos
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝¹ : μ.IsAddHaarMeasure
      inst✝ : Nontrivial E
      x : E
      r : Real
      h : LT.lt r 0
      ⊢ Eq (μ (Metric.closedBall x r)) (μ (Metric.ball x r))
    -/
  · rw [Metric.closedBall_eq_empty.mpr h, Metric.ball_eq_empty.mpr h.le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    inst✝ : Nontrivial E
    x : E
    r : Real
    h : Not (LT.lt r 0)
    ⊢ Eq (μ (Metric.closedBall x r)) (μ (Metric.ball x r))
  -/
  push_neg at h
  /-
    case neg
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    inst✝ : Nontrivial E
    x : E
    r : Real
    h : LE.le 0 r
    ⊢ Eq (μ (Metric.closedBall x r)) (μ (Metric.ball x r))
  -/
  rw [addHaar_closedBall μ x h, addHaar_ball μ x h]
  /-
    🎉 no goals
  -/


theorem addHaar_sphere_of_ne_zero (x : E) {r : ℝ} (hr : r ≠ 0) : μ (sphere x r) = 0 := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (μ (Metric.sphere x r)) 0
  -/
  rcases hr.lt_or_lt with (h | h)
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      x : E
      r : Real
      hr : Ne r 0
      h : LT.lt r 0
      ⊢ Eq (μ (Metric.sphere x r)) 0
    -/
  · simp only [empty_diff, measure_empty, ← closedBall_diff_ball, closedBall_eq_empty.2 h]
    /-
      🎉 no goals
    -/
  · rw [← closedBall_diff_ball,
      measure_diff ball_subset_closedBall measurableSet_ball.nullMeasurableSet
        measure_ball_lt_top.ne,
      addHaar_ball_of_pos μ _ h, addHaar_closedBall μ _ h.le, tsub_self]


theorem addHaar_sphere [Nontrivial E] (x : E) (r : ℝ) : μ (sphere x r) = 0 := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝¹ : μ.IsAddHaarMeasure
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (μ (Metric.sphere x r)) 0
  -/
  rcases eq_or_ne r 0 with (rfl | h)
    /-
      case inl
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝¹ : μ.IsAddHaarMeasure
      inst✝ : Nontrivial E
      x : E
      ⊢ Eq (μ (Metric.sphere x 0)) 0
    -/
  · rw [sphere_zero, measure_singleton]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      inst✝² : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝¹ : μ.IsAddHaarMeasure
      inst✝ : Nontrivial E
      x : E
      r : Real
      h : Ne r 0
      ⊢ Eq (μ (Metric.sphere x r)) 0
    -/
  · exact addHaar_sphere_of_ne_zero μ x h
    /-
      🎉 no goals
    -/


theorem addHaar_singleton_add_smul_div_singleton_add_smul {r : ℝ} (hr : r ≠ 0) (x y : E)
    (s t : Set E) : μ ({x} + r • s) / μ ({y} + r • t) = μ s / μ t :=
  calc
    μ ({x} + r • s) / μ ({y} + r • t) = ENNReal.ofReal (|r| ^ finrank ℝ E) * μ s *
        (ENNReal.ofReal (|r| ^ finrank ℝ E) * μ t)⁻¹ := by
      simp only [div_eq_mul_inv, addHaar_smul, image_add_left, measure_preimage_add, abs_pow,
        singleton_add]
    _ = ENNReal.ofReal (|r| ^ finrank ℝ E) * (ENNReal.ofReal (|r| ^ finrank ℝ E))⁻¹ *
          (μ s * (μ t)⁻¹) := by
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        r : Real
        hr : Ne r 0
        x y : E
        s t : Set E
        ⊢ Eq (HMul.hMul (HMul.hMul (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank  …
      -/
      rw [ENNReal.mul_inv]
        /-
          E : Type u_1
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace Real E
          inst✝³ : MeasurableSpace E
          inst✝² : BorelSpace E
          inst✝¹ : FiniteDimensional Real E
          μ : MeasureTheory.Measure E
          inst✝ : μ.IsAddHaarMeasure
          r : Real
          hr : Ne r 0
          x y : E
          s t : Set E
          ⊢ Eq (HMul.hMul (HMul.hMul (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank  …
        -/
      · ring
        /-
          🎉 no goals
        -/
        /-
          case ha
          E : Type u_1
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace Real E
          inst✝³ : MeasurableSpace E
          inst✝² : BorelSpace E
          inst✝¹ : FiniteDimensional Real E
          μ : MeasureTheory.Measure E
          inst✝ : μ.IsAddHaarMeasure
          r : Real
          hr : Ne r 0
          x y : E
          s t : Set E
          ⊢ Or (Ne (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank Real E))) 0) (Ne ( …
        -/
      · simp only [pow_pos (abs_pos.mpr hr), ENNReal.ofReal_eq_zero, not_le, Ne, true_or]
        /-
          🎉 no goals
        -/
        /-
          case hb
          E : Type u_1
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace Real E
          inst✝³ : MeasurableSpace E
          inst✝² : BorelSpace E
          inst✝¹ : FiniteDimensional Real E
          μ : MeasureTheory.Measure E
          inst✝ : μ.IsAddHaarMeasure
          r : Real
          hr : Ne r 0
          x y : E
          s t : Set E
          ⊢ Or (Ne (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank Real E))) Top.top) …
        -/
      · simp only [ENNReal.ofReal_ne_top, true_or, Ne, not_false_iff]
        /-
          🎉 no goals
        -/
    _ = μ s / μ t := by
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝ : μ.IsAddHaarMeasure
        r : Real
        hr : Ne r 0
        x y : E
        s t : Set E
        ⊢ Eq (HMul.hMul (HMul.hMul (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank  …
      -/
      rw [ENNReal.mul_inv_cancel, one_mul, div_eq_mul_inv]
        /-
          case h0
          E : Type u_1
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace Real E
          inst✝³ : MeasurableSpace E
          inst✝² : BorelSpace E
          inst✝¹ : FiniteDimensional Real E
          μ : MeasureTheory.Measure E
          inst✝ : μ.IsAddHaarMeasure
          r : Real
          hr : Ne r 0
          x y : E
          s t : Set E
          ⊢ Ne (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank Real E))) 0
        -/
      · simp only [pow_pos (abs_pos.mpr hr), ENNReal.ofReal_eq_zero, not_le, Ne]
        /-
          🎉 no goals
        -/
        /-
          case ht
          E : Type u_1
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace Real E
          inst✝³ : MeasurableSpace E
          inst✝² : BorelSpace E
          inst✝¹ : FiniteDimensional Real E
          μ : MeasureTheory.Measure E
          inst✝ : μ.IsAddHaarMeasure
          r : Real
          hr : Ne r 0
          x y : E
          s t : Set E
          ⊢ Ne (ENNReal.ofReal (HPow.hPow (abs r) (Module.finrank Real E))) Top.top
        -/
      · simp only [ENNReal.ofReal_ne_top, Ne, not_false_iff]
        /-
          🎉 no goals
        -/


instance (priority := 100) isUnifLocDoublingMeasureOfIsAddHaarMeasure :
    IsUnifLocDoublingMeasure μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    ⊢ IsUnifLocDoublingMeasure μ
  -/
  refine ⟨⟨(2 : ℝ≥0) ^ finrank ℝ E, ?_⟩⟩
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    ⊢ Filter.Eventually (fun ε => ∀ (x : E), LE.le (μ (Metric.closedBall x (HMul.h …
  -/
  filter_upwards [self_mem_nhdsWithin] with r hr x
  rw [addHaar_closedBall_mul μ x zero_le_two (le_of_lt hr), addHaar_closedBall_center μ x,
    ENNReal.ofReal, Real.toNNReal_pow zero_le_two]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    r : Real
    hr : Membership.mem (Set.Ioi 0) r
    x : E
    ⊢ LE.le (HMul.hMul (↑(HPow.hPow (Real.toNNReal 2) (Module.finrank Real E))) (μ …
  -/
  simp only [Real.toNNReal_ofNat, le_refl]
  /-
    🎉 no goals
  -/


theorem addHaar_parallelepiped (b : Basis ι ℝ G) (v : ι → G) :
    b.addHaar (parallelepiped v) = ENNReal.ofReal |b.det v| := by
  /-
    ι : Type u_2
    G : Type u_3
    inst✝⁵ : Fintype ι
    inst✝⁴ : DecidableEq ι
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    inst✝¹ : MeasurableSpace G
    inst✝ : BorelSpace G
    b : Basis ι Real G
    v : ι → G
    ⊢ Eq (b.addHaar (parallelepiped v)) (ENNReal.ofReal (abs (b.det v)))
  -/
  have : FiniteDimensional ℝ G := FiniteDimensional.of_fintype_basis b
  have A : parallelepiped v = b.constr ℕ v '' parallelepiped b := by
    rw [image_parallelepiped]
    -- Porting note: was `congr 1 with i` but Lean 4 `congr` applies `ext` first
    refine congr_arg _ <| funext fun i ↦ ?_
    exact (b.constr_basis ℕ v i).symm
  rw [A, addHaar_image_linearMap, b.addHaar_self, mul_one, ← LinearMap.det_toMatrix b,
    ← Basis.toMatrix_eq_toMatrix_constr, Basis.det_apply]


/-- The Lebesgue measure associated to an alternating map. It gives measure `|ω v|` to the
parallelepiped spanned by the vectors `v₁, ..., vₙ`. Note that it is not always a Haar measure,
as it can be zero, but it is always locally finite and translation invariant. -/
noncomputable irreducible_def _root_.AlternatingMap.measure (ω : G [⋀^Fin n]→ₗ[ℝ] ℝ) :
    Measure G :=
  ‖ω (finBasisOfFinrankEq ℝ G _i.out)‖₊ • (finBasisOfFinrankEq ℝ G _i.out).addHaar


theorem _root_.AlternatingMap.measure_parallelepiped (ω : G [⋀^Fin n]→ₗ[ℝ] ℝ)
    (v : Fin n → G) : ω.measure (parallelepiped v) = ENNReal.ofReal |ω v| := by
  /-
    G : Type u_3
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : FiniteDimensional Real G
    n : Nat
    _i : Fact (Eq (Module.finrank Real G) n)
    ω : AlternatingMap Real G Real (Fin n)
    v : Fin n → G
    ⊢ Eq (ω.measure (parallelepiped v)) (ENNReal.ofReal (abs (ω v)))
  -/
  conv_rhs => rw [ω.eq_smul_basis_det (finBasisOfFinrankEq ℝ G _i.out)]
  simp only [addHaar_parallelepiped, AlternatingMap.measure, coe_nnreal_smul_apply,
    AlternatingMap.smul_apply, Algebra.id.smul_eq_mul, abs_mul, ENNReal.ofReal_mul (abs_nonneg _),
    Real.ennnorm_eq_ofReal_abs]


instance (ω : G [⋀^Fin n]→ₗ[ℝ] ℝ) : IsAddLeftInvariant ω.measure := by
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝⁷ : μ.IsAddHaarMeasure
    s : Set E
    ι : Type u_2
    G : Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : FiniteDimensional Real G
    n : Nat
    _i : Fact (Eq (Module.finrank Real G) n)
    ω : AlternatingMap Real G Real (Fin n)
    ⊢ ω.measure.IsAddLeftInvariant
  -/
  rw [AlternatingMap.measure]; infer_instance
                               /-
                                 🎉 no goals
                               -/


instance (ω : G [⋀^Fin n]→ₗ[ℝ] ℝ) : IsLocallyFiniteMeasure ω.measure := by
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝⁷ : μ.IsAddHaarMeasure
    s : Set E
    ι : Type u_2
    G : Type u_3
    inst✝⁶ : Fintype ι
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace Real G
    inst✝² : MeasurableSpace G
    inst✝¹ : BorelSpace G
    inst✝ : FiniteDimensional Real G
    n : Nat
    _i : Fact (Eq (Module.finrank Real G) n)
    ω : AlternatingMap Real G Real (Fin n)
    ⊢ MeasureTheory.IsLocallyFiniteMeasure ω.measure
  -/
  rw [AlternatingMap.measure]; infer_instance
                               /-
                                 🎉 no goals
                               -/


theorem tendsto_addHaar_inter_smul_zero_of_density_zero_aux1 (s : Set E) (x : E)
    (h : Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 0)) (t : Set E)
    (u : Set E) (h'u : μ u ≠ 0) (t_bound : t ⊆ closedBall 0 1) :
    Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t)) / μ ({x} + r • u)) (𝓝[>] 0) (𝓝 0) := by
  have A : Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t)) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 0) := by
    apply
      tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h
        (Eventually.of_forall fun b => zero_le _)
    filter_upwards [self_mem_nhdsWithin]
    rintro r (rpos : 0 < r)
    rw [← affinity_unitClosedBall rpos.le, singleton_add, ← image_vadd]
    gcongr
  have B :
    Tendsto (fun r : ℝ => μ (closedBall x r) / μ ({x} + r • u)) (𝓝[>] 0)
      (𝓝 (μ (closedBall x 1) / μ ({x} + u))) := by
    apply tendsto_const_nhds.congr' _
    filter_upwards [self_mem_nhdsWithin]
    rintro r (rpos : 0 < r)
    have : closedBall x r = {x} + r • closedBall (0 : E) 1 := by
      simp only [_root_.smul_closedBall, Real.norm_of_nonneg rpos.le, zero_le_one, add_zero,
        mul_one, singleton_add_closedBall, smul_zero]
    simp only [this, addHaar_singleton_add_smul_div_singleton_add_smul μ rpos.ne']
    simp only [addHaar_closedBall_center, image_add_left, measure_preimage_add, singleton_add]
  have C : Tendsto (fun r : ℝ =>
        μ (s ∩ ({x} + r • t)) / μ (closedBall x r) * (μ (closedBall x r) / μ ({x} + r • u)))
      (𝓝[>] 0) (𝓝 (0 * (μ (closedBall x 1) / μ ({x} + u)))) := by
    apply ENNReal.Tendsto.mul A _ B (Or.inr ENNReal.zero_ne_top)
    simp only [ne_eq, not_true, singleton_add, image_add_left, measure_preimage_add, false_or,
      ENNReal.div_eq_top, h'u, not_and, and_false]
    intro aux
    exact (measure_closedBall_lt_top.ne aux).elim
    -- Porting note: it used to be enough to pass `measure_closedBall_lt_top.ne` to `simp`
    -- and avoid the `intro; exact` dance.
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    t_bound : HasSubset.Subset t (Metric.closedBall 0 1)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HDiv.hDiv (μ (Metric.closedBall x r)) (μ (HAdd.hA …
    C : Filter.Tendsto (fun r => HMul.hMul (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  simp only [zero_mul] at C
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    t_bound : HasSubset.Subset t (Metric.closedBall 0 1)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HDiv.hDiv (μ (Metric.closedBall x r)) (μ (HAdd.hA …
    C : Filter.Tendsto (fun r => HMul.hMul (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  apply C.congr' _
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    t_bound : HasSubset.Subset t (Metric.closedBall 0 1)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HDiv.hDiv (μ (Metric.closedBall x r)) (μ (HAdd.hA …
    C : Filter.Tendsto (fun r => HMul.hMul (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd …
    ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (fun r => HMul.hMul (HDiv.hDiv (μ (I …
  -/
  filter_upwards [self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    t_bound : HasSubset.Subset t (Metric.closedBall 0 1)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HDiv.hDiv (μ (Metric.closedBall x r)) (μ (HAdd.hA …
    C : Filter.Tendsto (fun r => HMul.hMul (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd …
    ⊢ ∀ (a : Real), Membership.mem (Set.Ioi 0) a → Eq (HMul.hMul (HDiv.hDiv (μ (In …
  -/
  rintro r (rpos : 0 < r)
  calc
    μ (s ∩ ({x} + r • t)) / μ (closedBall x r) * (μ (closedBall x r) / μ ({x} + r • u)) =
        μ (closedBall x r) * (μ (closedBall x r))⁻¹ * (μ (s ∩ ({x} + r • t)) / μ ({x} + r • u)) :=
      by simp only [div_eq_mul_inv]; ring
    _ = μ (s ∩ ({x} + r • t)) / μ ({x} + r • u) := by
      rw [ENNReal.mul_inv_cancel (measure_closedBall_pos μ x rpos).ne'
          measure_closedBall_lt_top.ne,
        one_mul]


theorem tendsto_addHaar_inter_smul_zero_of_density_zero_aux2 (s : Set E) (x : E)
    (h : Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 0)) (t : Set E)
    (u : Set E) (h'u : μ u ≠ 0) (R : ℝ) (Rpos : 0 < R) (t_bound : t ⊆ closedBall 0 R) :
    Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t)) / μ ({x} + r • u)) (𝓝[>] 0) (𝓝 0) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  set t' := R⁻¹ • t with ht'
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  set u' := R⁻¹ • u with hu'
  have A : Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t')) / μ ({x} + r • u')) (𝓝[>] 0) (𝓝 0) := by
    apply tendsto_addHaar_inter_smul_zero_of_density_zero_aux1 μ s x h t' u'
    · simp only [u', h'u, (pow_pos Rpos _).ne', abs_nonpos_iff, addHaar_smul, not_false_iff,
        ENNReal.ofReal_eq_zero, inv_eq_zero, inv_pow, Ne, or_self_iff, mul_eq_zero]
    · refine (smul_set_mono t_bound).trans_eq ?_
      rw [smul_closedBall _ _ Rpos.le, smul_zero, Real.norm_of_nonneg (inv_nonneg.2 Rpos.le),
        inv_mul_cancel₀ Rpos.ne']
  have B : Tendsto (fun r : ℝ => R * r) (𝓝[>] 0) (𝓝[>] (R * 0)) := by
    apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
    · exact (tendsto_const_nhds.mul tendsto_id).mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin]
      intro r rpos
      rw [mul_zero]
      exact mul_pos Rpos rpos
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    u' : Set E := HSMul.hSMul (Inv.inv R) u
    hu' : Eq u' (HSMul.hSMul (Inv.inv R) u)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HMul.hMul R r) (nhdsWithin 0 (Set.Ioi 0)) (nhdsWi …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  rw [mul_zero] at B
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    u' : Set E := HSMul.hSMul (Inv.inv R) u
    hu' : Eq u' (HSMul.hSMul (Inv.inv R) u)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HMul.hMul R r) (nhdsWithin 0 (Set.Ioi 0)) (nhdsWi …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  apply (A.comp B).congr' _
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    u' : Set E := HSMul.hSMul (Inv.inv R) u
    hu' : Eq u' (HSMul.hSMul (Inv.inv R) u)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HMul.hMul R r) (nhdsWithin 0 (Set.Ioi 0)) (nhdsWi …
    ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (Function.comp (fun r => HDiv.hDiv ( …
  -/
  filter_upwards [self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    u' : Set E := HSMul.hSMul (Inv.inv R) u
    hu' : Eq u' (HSMul.hSMul (Inv.inv R) u)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HMul.hMul R r) (nhdsWithin 0 (Set.Ioi 0)) (nhdsWi …
    ⊢ ∀ (a : Real), Membership.mem (Set.Ioi 0) a → Eq (Function.comp (fun r => HDi …
  -/
  rintro r -
  have T : (R * r) • t' = r • t := by
    rw [mul_comm, ht', smul_smul, mul_assoc, mul_inv_cancel₀ Rpos.ne', mul_one]
  have U : (R * r) • u' = r • u := by
    rw [mul_comm, hu', smul_smul, mul_assoc, mul_inv_cancel₀ Rpos.ne', mul_one]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    u' : Set E := HSMul.hSMul (Inv.inv R) u
    hu' : Eq u' (HSMul.hSMul (Inv.inv R) u)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HMul.hMul R r) (nhdsWithin 0 (Set.Ioi 0)) (nhdsWi …
    r : Real
    T : Eq (HSMul.hSMul (HMul.hMul R r) t') (HSMul.hSMul r t)
    U : Eq (HSMul.hSMul (HMul.hMul R r) u') (HSMul.hSMul r u)
    ⊢ Eq (Function.comp (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleto …
  -/
  dsimp
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t u : Set E
    h'u : Ne (μ u) 0
    R : Real
    Rpos : LT.lt 0 R
    t_bound : HasSubset.Subset t (Metric.closedBall 0 R)
    t' : Set E := HSMul.hSMul (Inv.inv R) t
    ht' : Eq t' (HSMul.hSMul (Inv.inv R) t)
    u' : Set E := HSMul.hSMul (Inv.inv R) u
    hu' : Eq u' (HSMul.hSMul (Inv.inv R) u)
    A : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    B : Filter.Tendsto (fun r => HMul.hMul R r) (nhdsWithin 0 (Set.Ioi 0)) (nhdsWi …
    r : Real
    T : Eq (HSMul.hSMul (HMul.hMul R r) t') (HSMul.hSMul r t)
    U : Eq (HSMul.hSMul (HMul.hMul R r) u') (HSMul.hSMul r u)
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hS …
  -/
  rw [T, U]
  /-
    🎉 no goals
  -/


/-- Consider a point `x` at which a set `s` has density zero, with respect to closed balls. Then it
also has density zero with respect to any measurable set `t`: the proportion of points in `s`
belonging to a rescaled copy `{x} + r • t` of `t` tends to zero as `r` tends to zero. -/
theorem tendsto_addHaar_inter_smul_zero_of_density_zero (s : Set E) (x : E)
    (h : Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 0)) (t : Set E)
    (ht : MeasurableSet t) (h''t : μ t ≠ ∞) :
    Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t)) / μ ({x} + r • t)) (𝓝[>] 0) (𝓝 0) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h''t : Ne (μ t) Top.top
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  refine tendsto_order.2 ⟨fun a' ha' => (ENNReal.not_lt_zero ha').elim, fun ε (εpos : 0 < ε) => ?_⟩
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h''t : Ne (μ t) Top.top
    ε : ENNReal
    εpos : LT.lt 0 ε
    ⊢ Filter.Eventually (fun b => LT.lt (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (S …
  -/
  rcases eq_or_ne (μ t) 0 with (h't | h't)
    /-
      case inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      s : Set E
      x : E
      h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
      t : Set E
      ht : MeasurableSet t
      h''t : Ne (μ t) Top.top
      ε : ENNReal
      εpos : LT.lt 0 ε
      h't : Eq (μ t) 0
      ⊢ Filter.Eventually (fun b => LT.lt (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (S …
    -/
  · filter_upwards with r
    suffices H : μ (s ∩ ({x} + r • t)) = 0 by
      rw [H]; simpa only [ENNReal.zero_div] using εpos
    /-
      case inl.h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      s : Set E
      x : E
      h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
      t : Set E
      ht : MeasurableSet t
      h''t : Ne (μ t) Top.top
      ε : ENNReal
      εpos : LT.lt 0 ε
      h't : Eq (μ t) 0
      r : Real
      ⊢ Eq (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t)))) 0
    -/
    apply le_antisymm _ (zero_le _)
    calc
      μ (s ∩ ({x} + r • t)) ≤ μ ({x} + r • t) := measure_mono inter_subset_right
      _ = 0 := by
        simp only [h't, addHaar_smul, image_add_left, measure_preimage_add, singleton_add,
          mul_zero]
  obtain ⟨n, npos, hn⟩ : ∃ n : ℕ, 0 < n ∧ μ (t \ closedBall 0 n) < ε / 2 * μ t := by
    have A :
      Tendsto (fun n : ℕ => μ (t \ closedBall 0 n)) atTop
        (𝓝 (μ (⋂ n : ℕ, t \ closedBall 0 n))) := by
      have N : ∃ n : ℕ, μ (t \ closedBall 0 n) ≠ ∞ :=
        ⟨0, ((measure_mono diff_subset).trans_lt h''t.lt_top).ne⟩
      refine tendsto_measure_iInter_atTop
        (fun n ↦ (ht.diff measurableSet_closedBall).nullMeasurableSet) (fun m n hmn ↦ ?_) N
      exact diff_subset_diff Subset.rfl (closedBall_subset_closedBall (Nat.cast_le.2 hmn))
    have : ⋂ n : ℕ, t \ closedBall 0 n = ∅ := by
      simp_rw [diff_eq, ← inter_iInter, iInter_eq_compl_iUnion_compl, compl_compl,
        iUnion_closedBall_nat, compl_univ, inter_empty]
    simp only [this, measure_empty] at A
    have I : 0 < ε / 2 * μ t := ENNReal.mul_pos (ENNReal.half_pos εpos.ne').ne' h't
    exact (Eventually.and (Ioi_mem_atTop 0) ((tendsto_order.1 A).2 _ I)).exists
  have L :
    Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • (t ∩ closedBall 0 n))) / μ ({x} + r • t)) (𝓝[>] 0)
      (𝓝 0) :=
    tendsto_addHaar_inter_smul_zero_of_density_zero_aux2 μ s x h _ t h't n (Nat.cast_pos.2 npos)
      inter_subset_right
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h''t : Ne (μ t) Top.top
    ε : ENNReal
    εpos : LT.lt 0 ε
    h't : Ne (μ t) 0
    n : Nat
    npos : LT.lt 0 n
    hn : LT.lt (μ (SDiff.sdiff t (Metric.closedBall 0 ↑n))) (HMul.hMul (HDiv.hDiv  …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    ⊢ Filter.Eventually (fun b => LT.lt (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (S …
  -/
  filter_upwards [(tendsto_order.1 L).2 _ (ENNReal.half_pos εpos.ne'), self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h''t : Ne (μ t) Top.top
    ε : ENNReal
    εpos : LT.lt 0 ε
    h't : Ne (μ t) 0
    n : Nat
    npos : LT.lt 0 n
    hn : LT.lt (μ (SDiff.sdiff t (Metric.closedBall 0 ↑n))) (HMul.hMul (HDiv.hDiv  …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton …
    ⊢ ∀ (a : Real), LT.lt (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.singl …
  -/
  rintro r hr (rpos : 0 < r)
  have I :
    μ (s ∩ ({x} + r • t)) ≤
      μ (s ∩ ({x} + r • (t ∩ closedBall 0 n))) + μ ({x} + r • (t \ closedBall 0 n)) :=
    calc
      μ (s ∩ ({x} + r • t)) =
          μ (s ∩ ({x} + r • (t ∩ closedBall 0 n)) ∪ s ∩ ({x} + r • (t \ closedBall 0 n))) := by
        rw [← inter_union_distrib_left, ← add_union, ← smul_set_union, inter_union_diff]
      _ ≤ μ (s ∩ ({x} + r • (t ∩ closedBall 0 n))) + μ (s ∩ ({x} + r • (t \ closedBall 0 n))) :=
        measure_union_le _ _
      _ ≤ μ (s ∩ ({x} + r • (t ∩ closedBall 0 n))) + μ ({x} + r • (t \ closedBall 0 n)) := by
        gcongr; apply inter_subset_right
  calc
    μ (s ∩ ({x} + r • t)) / μ ({x} + r • t) ≤
        (μ (s ∩ ({x} + r • (t ∩ closedBall 0 n))) + μ ({x} + r • (t \ closedBall 0 n))) /
          μ ({x} + r • t) := by gcongr
    _ < ε / 2 + ε / 2 := by
      rw [ENNReal.add_div]
      apply ENNReal.add_lt_add hr _
      rwa [addHaar_singleton_add_smul_div_singleton_add_smul μ rpos.ne',
        ENNReal.div_lt_iff (Or.inl h't) (Or.inl h''t)]
    _ = ε := ENNReal.add_halves _


theorem tendsto_addHaar_inter_smul_one_of_density_one_aux (s : Set E) (hs : MeasurableSet s)
    (x : E) (h : Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 1))
    (t : Set E) (ht : MeasurableSet t) (h't : μ t ≠ 0) (h''t : μ t ≠ ∞) :
    Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t)) / μ ({x} + r • t)) (𝓝[>] 0) (𝓝 1) := by
  have I : ∀ u v, μ u ≠ 0 → μ u ≠ ∞ → MeasurableSet v →
    μ u / μ u - μ (vᶜ ∩ u) / μ u = μ (v ∩ u) / μ u := by
    intro u v uzero utop vmeas
    simp_rw [div_eq_mul_inv]
    rw [← ENNReal.sub_mul]; swap
    · simp only [uzero, ENNReal.inv_eq_top, imp_true_iff, Ne, not_false_iff]
    congr 1
    rw [inter_comm _ u, inter_comm _ u, eq_comm]
    exact ENNReal.eq_sub_of_add_eq' utop (measure_inter_add_diff u vmeas)
  have L : Tendsto (fun r => μ (sᶜ ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 0) := by
    have A : Tendsto (fun r => μ (closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 1) := by
      apply tendsto_const_nhds.congr' _
      filter_upwards [self_mem_nhdsWithin]
      intro r hr
      rw [div_eq_mul_inv, ENNReal.mul_inv_cancel]
      · exact (measure_closedBall_pos μ _ hr).ne'
      · exact measure_closedBall_lt_top.ne
    have B := ENNReal.Tendsto.sub A h (Or.inl ENNReal.one_ne_top)
    simp only [tsub_self] at B
    apply B.congr' _
    filter_upwards [self_mem_nhdsWithin]
    rintro r (rpos : 0 < r)
    convert I (closedBall x r) sᶜ (measure_closedBall_pos μ _ rpos).ne'
      measure_closedBall_lt_top.ne hs.compl
    rw [compl_compl]
  have L' : Tendsto (fun r : ℝ => μ (sᶜ ∩ ({x} + r • t)) / μ ({x} + r • t)) (𝓝[>] 0) (𝓝 0) :=
    tendsto_addHaar_inter_smul_zero_of_density_zero μ sᶜ x L t ht h''t
  have L'' : Tendsto (fun r : ℝ => μ ({x} + r • t) / μ ({x} + r • t)) (𝓝[>] 0) (𝓝 1) := by
    apply tendsto_const_nhds.congr' _
    filter_upwards [self_mem_nhdsWithin]
    rintro r (rpos : 0 < r)
    rw [addHaar_singleton_add_smul_div_singleton_add_smul μ rpos.ne', ENNReal.div_self h't h''t]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasurableSet s
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    I : ∀ (u v : Set E), Ne (μ u) 0 → Ne (μ u) Top.top → MeasurableSet v → Eq (HSu …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (Met …
    L' : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (HA …
    L'' : Filter.Tendsto (fun r => HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  have := ENNReal.Tendsto.sub L'' L' (Or.inl ENNReal.one_ne_top)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasurableSet s
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    I : ∀ (u v : Set E), Ne (μ u) 0 → Ne (μ u) Top.top → MeasurableSet v → Eq (HSu …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (Met …
    L' : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (HA …
    L'' : Filter.Tendsto (fun r => HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) …
    this : Filter.Tendsto (fun a => HSub.hSub (HDiv.hDiv (μ (HAdd.hAdd (Singleton. …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  simp only [tsub_zero] at this
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasurableSet s
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    I : ∀ (u v : Set E), Ne (μ u) 0 → Ne (μ u) Top.top → MeasurableSet v → Eq (HSu …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (Met …
    L' : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (HA …
    L'' : Filter.Tendsto (fun r => HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) …
    this : Filter.Tendsto (fun a => HSub.hSub (HDiv.hDiv (μ (HAdd.hAdd (Singleton. …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  apply this.congr' _
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasurableSet s
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    I : ∀ (u v : Set E), Ne (μ u) 0 → Ne (μ u) Top.top → MeasurableSet v → Eq (HSu …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (Met …
    L' : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (HA …
    L'' : Filter.Tendsto (fun r => HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) …
    this : Filter.Tendsto (fun a => HSub.hSub (HDiv.hDiv (μ (HAdd.hAdd (Singleton. …
    ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyEq (fun a => HSub.hSub (HDiv.hDiv (μ (H …
  -/
  filter_upwards [self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasurableSet s
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    I : ∀ (u v : Set E), Ne (μ u) 0 → Ne (μ u) Top.top → MeasurableSet v → Eq (HSu …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (Met …
    L' : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (HA …
    L'' : Filter.Tendsto (fun r => HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) …
    this : Filter.Tendsto (fun a => HSub.hSub (HDiv.hDiv (μ (HAdd.hAdd (Singleton. …
    ⊢ ∀ (a : Real), Membership.mem (Set.Ioi 0) a → Eq (HSub.hSub (HDiv.hDiv (μ (HA …
  -/
  rintro r (rpos : 0 < r)
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    hs : MeasurableSet s
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    I : ∀ (u v : Set E), Ne (μ u) 0 → Ne (μ u) Top.top → MeasurableSet v → Eq (HSu …
    L : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (Met …
    L' : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (HasCompl.compl s) (HA …
    L'' : Filter.Tendsto (fun r => HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) …
    this : Filter.Tendsto (fun a => HSub.hSub (HDiv.hDiv (μ (HAdd.hAdd (Singleton. …
    r : Real
    rpos : LT.lt 0 r
    ⊢ Eq (HSub.hSub (HDiv.hDiv (μ (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul  …
  -/
  refine I ({x} + r • t) s ?_ ?_ hs
  · simp only [h't, abs_of_nonneg rpos.le, pow_pos rpos, addHaar_smul, image_add_left,
      ENNReal.ofReal_eq_zero, not_le, or_false, Ne, measure_preimage_add, abs_pow,
      singleton_add, mul_eq_zero]
  · simp [h''t, ENNReal.ofReal_ne_top, addHaar_smul, image_add_left, ENNReal.mul_eq_top,
      Ne, not_false_iff, measure_preimage_add, singleton_add, or_self_iff]


/-- Consider a point `x` at which a set `s` has density one, with respect to closed balls (i.e.,
a Lebesgue density point of `s`). Then `s` has also density one at `x` with respect to any
measurable set `t`: the proportion of points in `s` belonging to a rescaled copy `{x} + r • t`
of `t` tends to one as `r` tends to zero. -/
theorem tendsto_addHaar_inter_smul_one_of_density_one (s : Set E) (x : E)
    (h : Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 1)) (t : Set E)
    (ht : MeasurableSet t) (h't : μ t ≠ 0) (h''t : μ t ≠ ∞) :
    Tendsto (fun r : ℝ => μ (s ∩ ({x} + r • t)) / μ ({x} + r • t)) (𝓝[>] 0) (𝓝 1) := by
  have : Tendsto (fun r : ℝ => μ (toMeasurable μ s ∩ ({x} + r • t)) / μ ({x} + r • t))
    (𝓝[>] 0) (𝓝 1) := by
    apply
      tendsto_addHaar_inter_smul_one_of_density_one_aux μ _ (measurableSet_toMeasurable _ _) _ _
        t ht h't h''t
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' h tendsto_const_nhds
    · refine Eventually.of_forall fun r ↦ ?_
      gcongr
      apply subset_toMeasurable
    · filter_upwards [self_mem_nhdsWithin]
      rintro r -
      apply ENNReal.div_le_of_le_mul
      rw [one_mul]
      exact measure_mono inter_subset_right
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    this : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (MeasureTheory.toMea …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.s …
  -/
  refine this.congr fun r => ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    this : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (MeasureTheory.toMea …
    r : Real
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter (MeasureTheory.toMeasurable μ s) (HAdd.hAdd (S …
  -/
  congr 1
  /-
    case e_a
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    this : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (MeasureTheory.toMea …
    r : Real
    ⊢ Eq (μ (Inter.inter (MeasureTheory.toMeasurable μ s) (HAdd.hAdd (Singleton.si …
  -/
  apply measure_toMeasurable_inter_of_sFinite
  /-
    case e_a.hs
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    this : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (MeasureTheory.toMea …
    r : Real
    ⊢ MeasurableSet (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t))
  -/
  simp only [image_add_left, singleton_add]
  /-
    case e_a.hs
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    h''t : Ne (μ t) Top.top
    this : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter (MeasureTheory.toMea …
    r : Real
    ⊢ MeasurableSet (Set.preimage (fun x_1 => HAdd.hAdd (Neg.neg x) x_1) (HSMul.hS …
  -/
  apply (continuous_add_left (-x)).measurable (ht.const_smul₀ r)
  /-
    🎉 no goals
  -/


/-- Consider a point `x` at which a set `s` has density one, with respect to closed balls (i.e.,
a Lebesgue density point of `s`). Then `s` intersects the rescaled copies `{x} + r • t` of a given
set `t` with positive measure, for any small enough `r`. -/
theorem eventually_nonempty_inter_smul_of_density_one (s : Set E) (x : E)
    (h : Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 1)) (t : Set E)
    (ht : MeasurableSet t) (h't : μ t ≠ 0) :
    ∀ᶠ r in 𝓝[>] (0 : ℝ), (s ∩ ({x} + r • t)).Nonempty := by
  obtain ⟨t', t'_meas, t't, t'pos, t'top⟩ : ∃ t', MeasurableSet t' ∧ t' ⊆ t ∧ 0 < μ t' ∧ μ t' < ⊤ :=
    exists_subset_measure_lt_top ht h't.bot_lt
  filter_upwards [(tendsto_order.1
          (tendsto_addHaar_inter_smul_one_of_density_one μ s x h t' t'_meas t'pos.ne' t'top.ne)).1
      0 zero_lt_one]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    t' : Set E
    t'_meas : MeasurableSet t'
    t't : HasSubset.Subset t' t
    t'pos : LT.lt 0 (μ t')
    t'top : LT.lt (μ t') Top.top
    ⊢ ∀ (a : Real), LT.lt 0 (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.sin …
  -/
  intro r hr
  have : μ (s ∩ ({x} + r • t')) ≠ 0 := fun h' => by
    simp only [ENNReal.not_lt_zero, ENNReal.zero_div, h'] at hr
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    t' : Set E
    t'_meas : MeasurableSet t'
    t't : HasSubset.Subset t' t
    t'pos : LT.lt 0 (μ t')
    t'top : LT.lt (μ t') Top.top
    r : Real
    hr : LT.lt 0 (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) ( …
    this : Ne (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r  …
    ⊢ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t))).Nonempty
  -/
  have : (s ∩ ({x} + r • t')).Nonempty := nonempty_of_measure_ne_zero this
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    t' : Set E
    t'_meas : MeasurableSet t'
    t't : HasSubset.Subset t' t
    t'pos : LT.lt 0 (μ t')
    t'top : LT.lt (μ t') Top.top
    r : Real
    hr : LT.lt 0 (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) ( …
    this✝ : Ne (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r …
    this : (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t'))). …
    ⊢ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t))).Nonempty
  -/
  apply this.mono (inter_subset_inter Subset.rfl _)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    s : Set E
    x : E
    h : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x  …
    t : Set E
    ht : MeasurableSet t
    h't : Ne (μ t) 0
    t' : Set E
    t'_meas : MeasurableSet t'
    t't : HasSubset.Subset t' t
    t'pos : LT.lt 0 (μ t')
    t'top : LT.lt (μ t') Top.top
    r : Real
    hr : LT.lt 0 (HDiv.hDiv (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) ( …
    this✝ : Ne (μ (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r …
    this : (Inter.inter s (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t'))). …
    ⊢ HasSubset.Subset (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r t')) (HAd …
  -/
  exact add_subset_add Subset.rfl (smul_set_mono t't)
  /-
    🎉 no goals
  -/


