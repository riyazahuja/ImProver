/-- In this file we normalise the measure on `ℝ / ℤ` to have total volume 1. -/
local instance : MeasureSpace UnitAddCircle := ⟨AddCircle.haarAddCircle⟩


/-- The measure on `ℝ / ℤ` is a Haar measure. -/
local instance : Measure.IsAddHaarMeasure (volume : Measure UnitAddCircle) :=
  inferInstanceAs (Measure.IsAddHaarMeasure AddCircle.haarAddCircle)


/-- The measure on `ℝ / ℤ` is a probability measure. -/
local instance : IsProbabilityMeasure (volume : Measure UnitAddCircle) :=
  inferInstanceAs (IsProbabilityMeasure AddCircle.haarAddCircle)


/-- The product of finitely many copies of the unit circle, indexed by `d`. -/
abbrev UnitAddTorus (d : Type*) := d → UnitAddCircle


/-- Exponential monomials in `d` variables. -/
def mFourier : C(UnitAddTorus d, ℂ) where
  toFun x := ∏ i : d, fourier (n i) (x i)
  continuous_toFun := continuous_finset_prod _
    fun i _ ↦ (fourier (n i)).continuous.comp (continuous_apply i)


lemma mFourier_neg : mFourier (-n) x = conj (mFourier n x) := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    n : d → Int
    x : UnitAddTorus d
    ⊢ Eq ((UnitAddTorus.mFourier (Neg.neg n)) x) ((starRingEnd Complex) ((UnitAddT …
  -/
  simp only [mFourier, Pi.neg_apply, fourier_neg, ContinuousMap.coe_mk, map_prod]
  /-
    🎉 no goals
  -/


lemma mFourier_add {m : d → ℤ} : mFourier (m + n) x = mFourier m x * mFourier n x := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    n : d → Int
    x : UnitAddTorus d
    m : d → Int
    ⊢ Eq ((UnitAddTorus.mFourier (HAdd.hAdd m n)) x) (HMul.hMul ((UnitAddTorus.mFo …
  -/
  simp only [mFourier, Pi.add_apply, fourier_add, ContinuousMap.coe_mk, ← Finset.prod_mul_distrib]
  /-
    🎉 no goals
  -/


lemma mFourier_zero : mFourier (0 : d → ℤ) = 1 := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    ⊢ Eq (UnitAddTorus.mFourier 0) 1
  -/
  ext x
  simp only [mFourier, Pi.zero_apply, fourier_zero, Finset.prod_const_one, ContinuousMap.coe_mk,
    ContinuousMap.one_apply]


lemma mFourier_norm : ‖mFourier n‖ = 1 := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    n : d → Int
    ⊢ Eq (Norm.norm (UnitAddTorus.mFourier n)) 1
  -/
  apply le_antisymm
    /-
      case a
      d : Type u_1
      inst✝ : Fintype d
      n : d → Int
      ⊢ LE.le (Norm.norm (UnitAddTorus.mFourier n)) 1
    -/
  · refine (ContinuousMap.norm_le _ zero_le_one).mpr fun i ↦ ?_
    simp only [mFourier, fourier_apply, ContinuousMap.coe_mk, norm_prod, Complex.norm_eq_abs,
      Circle.abs_coe, Finset.prod_const_one, le_rfl]
    /-
      case a
      d : Type u_1
      inst✝ : Fintype d
      n : d → Int
      ⊢ LE.le 1 (Norm.norm (UnitAddTorus.mFourier n))
    -/
  · refine (le_of_eq ?_).trans ((mFourier n).norm_coe_le_norm fun _ ↦ 0)
    simp only [mFourier, ContinuousMap.coe_mk, fourier_eval_zero, Finset.prod_const_one,
      CStarRing.norm_one]


lemma mFourier_single [DecidableEq d] (z : d → AddCircle (1 : ℝ)) (i : d) :
    mFourier (Pi.single i 1) z = fourier 1 (z i) := by
  /-
    d : Type u_1
    inst✝¹ : Fintype d
    inst✝ : DecidableEq d
    z : d → AddCircle 1
    i : d
    ⊢ Eq ((UnitAddTorus.mFourier (Pi.single i 1)) z) ((fourier 1) (z i))
  -/
  simp_rw [mFourier, ContinuousMap.coe_mk]
  /-
    d : Type u_1
    inst✝¹ : Fintype d
    inst✝ : DecidableEq d
    z : d → AddCircle 1
    i : d
    ⊢ Eq (Finset.univ.prod fun i_1 => (fourier (Pi.single i 1 i_1)) (z i_1)) ((fou …
  -/
  have := Finset.prod_mul_prod_compl {i} (fun j ↦ fourier ((Pi.single i (1 : ℤ) : d → ℤ) j) (z j))
  /-
    d : Type u_1
    inst✝¹ : Fintype d
    inst✝ : DecidableEq d
    z : d → AddCircle 1
    i : d
    this : Eq (HMul.hMul ((Singleton.singleton i).prod fun i_1 => (fourier (Pi.sin …
    ⊢ Eq (Finset.univ.prod fun i_1 => (fourier (Pi.single i 1 i_1)) (z i_1)) ((fou …
  -/
  rw [Finset.prod_singleton, Finset.prod_congr rfl (fun j hj ↦ ?_)] at this
    /-
      d : Type u_1
      inst✝¹ : Fintype d
      inst✝ : DecidableEq d
      z : d → AddCircle 1
      i : d
      this✝ : Eq (HMul.hMul ((fourier (Pi.single i 1 i)) (z i)) ((HasCompl.compl (Si …
      this : Eq (HMul.hMul ((fourier (Pi.single i 1 i)) (z i)) ((HasCompl.compl (Sin …
      ⊢ Eq (Finset.univ.prod fun i_1 => (fourier (Pi.single i 1 i_1)) (z i_1)) ((fou …
    -/
  · rw [← this, Finset.prod_const_one, mul_one, Pi.single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      d : Type u_1
      inst✝¹ : Fintype d
      inst✝ : DecidableEq d
      z : d → AddCircle 1
      i : d
      this : Eq (HMul.hMul ((fourier (Pi.single i 1 i)) (z i)) ((HasCompl.compl (Sin …
      j : d
      hj : Membership.mem (HasCompl.compl (Singleton.singleton i)) j
      ⊢ Eq ((fourier (Pi.single i 1 j)) (z j)) 1
    -/
  · rw [Finset.mem_compl, Finset.mem_singleton] at hj
    /-
      d : Type u_1
      inst✝¹ : Fintype d
      inst✝ : DecidableEq d
      z : d → AddCircle 1
      i : d
      this : Eq (HMul.hMul ((fourier (Pi.single i 1 i)) (z i)) ((HasCompl.compl (Sin …
      j : d
      hj : Not (Eq j i)
      ⊢ Eq ((fourier (Pi.single i 1 j)) (z j)) 1
    -/
    simp only [Pi.single_eq_of_ne hj, fourier_zero]
    /-
      🎉 no goals
    -/


/-- The star subalgebra of `C(UnitAddTorus d, ℂ)` generated by `mFourier n` for `n ∈ ℤᵈ`. -/
def mFourierSubalgebra (d : Type*) [Fintype d] : StarSubalgebra ℂ C(UnitAddTorus d, ℂ) where
  toSubalgebra := Algebra.adjoin ℂ (range mFourier)
  star_mem' := by
    /-
      d✝ : Type u_1
      inst✝¹ : Fintype d✝
      d : Type u_2
      inst✝ : Fintype d
      ⊢ ∀ {a : ContinuousMap (UnitAddTorus d) Complex}, Membership.mem (Algebra.adjo …
    -/
    show Algebra.adjoin ℂ (range mFourier) ≤ star (Algebra.adjoin ℂ (range mFourier))
    /-
      d✝ : Type u_1
      inst✝¹ : Fintype d✝
      d : Type u_2
      inst✝ : Fintype d
      ⊢ LE.le (Algebra.adjoin Complex (Set.range UnitAddTorus.mFourier)) (Star.star  …
    -/
    refine adjoin_le ?_
    /-
      d✝ : Type u_1
      inst✝¹ : Fintype d✝
      d : Type u_2
      inst✝ : Fintype d
      ⊢ HasSubset.Subset (Set.range UnitAddTorus.mFourier) ↑(Star.star (Algebra.adjo …
    -/
    rintro _ ⟨n, rfl⟩
    /-
      case intro
      d✝ : Type u_1
      inst✝¹ : Fintype d✝
      d : Type u_2
      inst✝ : Fintype d
      n : d → Int
      ⊢ Membership.mem (↑(Star.star (Algebra.adjoin Complex (Set.range UnitAddTorus. …
    -/
    refine subset_adjoin ⟨-n, ?_⟩
    /-
      case intro
      d✝ : Type u_1
      inst✝¹ : Fintype d✝
      d : Type u_2
      inst✝ : Fintype d
      n : d → Int
      ⊢ Eq (UnitAddTorus.mFourier (Neg.neg n)) (Star.star (UnitAddTorus.mFourier n))
    -/
    ext1 x
    /-
      case intro.h
      d✝ : Type u_1
      inst✝¹ : Fintype d✝
      d : Type u_2
      inst✝ : Fintype d
      n : d → Int
      x : UnitAddTorus d
      ⊢ Eq ((UnitAddTorus.mFourier (Neg.neg n)) x) ((Star.star (UnitAddTorus.mFourie …
    -/
    simp only [mFourier_neg, starRingEnd_apply, ContinuousMap.star_apply]
    /-
      🎉 no goals
    -/


/-- The star subalgebra of `C(UnitAddTorus d, ℂ)` generated by `mFourier n` for `n ∈ ℤᵈ` is in fact
the linear span of these functions. -/
theorem mFourierSubalgebra_coe :
    (mFourierSubalgebra d).toSubalgebra.toSubmodule = span ℂ (range mFourier) := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    ⊢ Eq (Subalgebra.toSubmodule (UnitAddTorus.mFourierSubalgebra d).toSubalgebra) …
  -/
  apply adjoin_eq_span_of_subset
  /-
    case hs
    d : Type u_1
    inst✝ : Fintype d
    ⊢ HasSubset.Subset ↑(Submonoid.closure (Set.range UnitAddTorus.mFourier)) ↑(Su …
  -/
  refine .trans (fun x ↦ Submonoid.closure_induction (fun _ ↦ id) ⟨0, ?_⟩ ?_) subset_span
    /-
      case hs.refine_1
      d : Type u_1
      inst✝ : Fintype d
      x : ContinuousMap (UnitAddTorus d) Complex
      ⊢ Eq (UnitAddTorus.mFourier 0) 1
    -/
  · ext z
    simp only [mFourier, Pi.zero_apply, fourier_zero, Finset.prod_const, one_pow,
      ContinuousMap.coe_mk, ContinuousMap.one_apply]
    /-
      case hs.refine_2
      d : Type u_1
      inst✝ : Fintype d
      x : ContinuousMap (UnitAddTorus d) Complex
      ⊢ ∀ (x y : ContinuousMap (UnitAddTorus d) Complex), Membership.mem (Submonoid. …
    -/
  · rintro _ _ _ _ ⟨m, rfl⟩ ⟨n, rfl⟩
    /-
      case hs.refine_2.intro.intro
      d : Type u_1
      inst✝ : Fintype d
      x : ContinuousMap (UnitAddTorus d) Complex
      m : d → Int
      hx✝ : Membership.mem (Submonoid.closure (Set.range UnitAddTorus.mFourier)) (Un …
      n : d → Int
      hy✝ : Membership.mem (Submonoid.closure (Set.range UnitAddTorus.mFourier)) (Un …
      ⊢ Membership.mem (Set.range UnitAddTorus.mFourier) (HMul.hMul (UnitAddTorus.mF …
    -/
    refine ⟨m + n, ?_⟩
    /-
      case hs.refine_2.intro.intro
      d : Type u_1
      inst✝ : Fintype d
      x : ContinuousMap (UnitAddTorus d) Complex
      m : d → Int
      hx✝ : Membership.mem (Submonoid.closure (Set.range UnitAddTorus.mFourier)) (Un …
      n : d → Int
      hy✝ : Membership.mem (Submonoid.closure (Set.range UnitAddTorus.mFourier)) (Un …
      ⊢ Eq (UnitAddTorus.mFourier (HAdd.hAdd m n)) (HMul.hMul (UnitAddTorus.mFourier …
    -/
    ext z
    simp only [mFourier, Pi.add_apply, fourier_apply, fourier_add', Finset.prod_mul_distrib,
      ContinuousMap.coe_mk, ContinuousMap.mul_apply]


/-- The subalgebra of `C(UnitAddTorus d, ℂ)` generated by `mFourier n` for `n ∈ ℤᵈ` separates
points. -/
theorem mFourierSubalgebra_separatesPoints : (mFourierSubalgebra d).SeparatesPoints := by
  classical
  intro x y hxy
  rw [Ne, funext_iff, not_forall] at hxy
  obtain ⟨i, hi⟩ := hxy
  refine ⟨_, ⟨mFourier (Pi.single i 1), subset_adjoin ⟨Pi.single i 1, rfl⟩, rfl⟩, ?_⟩
  dsimp only
  rw [mFourier_single, mFourier_single, fourier_one, fourier_one, Ne, Subtype.coe_inj]
  contrapose! hi
  exact AddCircle.injective_toCircle one_ne_zero hi


/-- The subalgebra of `C(UnitAddTorus d, ℂ)` generated by `mFourier n` for `n : d → ℤ` is dense. -/
theorem mFourierSubalgebra_closure_eq_top : (mFourierSubalgebra d).topologicalClosure = ⊤ :=
  ContinuousMap.starSubalgebra_topologicalClosure_eq_top_of_separatesPoints _
    mFourierSubalgebra_separatesPoints


/-- The linear span of the monomials `mFourier n` is dense in `C(UnitAddTorus d, ℂ)`. -/
theorem span_mFourier_closure_eq_top :
    (span ℂ (range <| mFourier (d := d))).topologicalClosure = ⊤ := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    ⊢ Eq (Submodule.span Complex (Set.range UnitAddTorus.mFourier)).topologicalClo …
  -/
  rw [← mFourierSubalgebra_coe]
  exact congr_arg (Subalgebra.toSubmodule <| StarSubalgebra.toSubalgebra ·)
    mFourierSubalgebra_closure_eq_top


/-- The family of monomials `mFourier n`, parametrized by `n : ℤᵈ` and considered as
elements of the `Lp` space of functions `UnitAddTorus d → ℂ`. -/
abbrev mFourierLp (p : ℝ≥0∞) [Fact (1 ≤ p)] (n : d → ℤ) :
    Lp ℂ p (volume : Measure (UnitAddTorus d)) :=
  ContinuousMap.toLp (E := ℂ) p volume ℂ (mFourier n)


theorem coeFn_mFourierLp (p : ℝ≥0∞) [Fact (1 ≤ p)] (n : d → ℤ) :
    mFourierLp p n =ᵐ[volume] mFourier n :=
  ContinuousMap.coeFn_toLp volume (mFourier n)


/-- For each `1 ≤ p < ∞`, the linear span of the monomials `mFourier n` is dense in the `Lᵖ` space
of functions on `UnitAddTorus d`. -/
theorem span_mFourierLp_closure_eq_top {p : ℝ≥0∞} [Fact (1 ≤ p)] (hp : p ≠ ∞) :
    (span ℂ (range (@mFourierLp d _ p _))).topologicalClosure = ⊤ := by
  simpa only [map_span, ContinuousLinearMap.coe_coe, ← range_comp, Function.comp_def] using
    (ContinuousMap.toLp_denseRange ℂ volume ℂ hp).topologicalClosure_map_submodule
      span_mFourier_closure_eq_top


/-- The monomials `mFourierLp 2 n` are an orthonormal set in `L²`. -/
theorem orthonormal_mFourier : Orthonormal ℂ (mFourierLp (d := d) 2) := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    ⊢ Orthonormal Complex (UnitAddTorus.mFourierLp 2)
  -/
  rw [orthonormal_iff_ite]
  /-
    d : Type u_1
    inst✝ : Fintype d
    ⊢ ∀ (i j : d → Int), Eq (Inner.inner (UnitAddTorus.mFourierLp 2 i) (UnitAddTor …
  -/
  intro m n
  /-
    d : Type u_1
    inst✝ : Fintype d
    m n : d → Int
    ⊢ Eq (Inner.inner (UnitAddTorus.mFourierLp 2 m) (UnitAddTorus.mFourierLp 2 n)) …
  -/
  simp only [ContinuousMap.inner_toLp, ← mFourier_neg, ← mFourier_add]
  /-
    d : Type u_1
    inst✝ : Fintype d
    m n : d → Int
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (UnitA …
  -/
  split_ifs with h
  · simpa only [h, neg_add_cancel, mFourier_zero, measure_univ, ENNReal.one_toReal, one_smul] using
      integral_const (α := UnitAddTorus d) (μ := volume) (1 : ℂ)
  /-
    case neg
    d : Type u_1
    inst✝ : Fintype d
    m n : d → Int
    h : Not (Eq m n)
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (UnitA …
  -/
  rw [mFourier, ContinuousMap.coe_mk, MeasureTheory.integral_fintype_prod_eq_prod]
  /-
    case neg
    d : Type u_1
    inst✝ : Fintype d
    m n : d → Int
    h : Not (Eq m n)
    ⊢ Eq (Finset.univ.prod fun i => MeasureTheory.integral MeasureTheory.MeasureSp …
  -/
  obtain ⟨i, hi⟩ := Function.ne_iff.mp h
  /-
    case neg.intro
    d : Type u_1
    inst✝ : Fintype d
    m n : d → Int
    h : Not (Eq m n)
    i : d
    hi : Ne (m i) (n i)
    ⊢ Eq (Finset.univ.prod fun i => MeasureTheory.integral MeasureTheory.MeasureSp …
  -/
  apply Finset.prod_eq_zero (Finset.mem_univ i)
  simpa only [eq_false_intro hi, if_false, ContinuousMap.inner_toLp, ← fourier_neg,
    ← fourier_add] using (orthonormal_iff_ite.mp <| orthonormal_fourier) (m i) (n i)


/-- The `n`-th Fourier coefficient of a function `UnitAddTorus d → E`, for `E` a complete normed
`ℂ`-vector space, defined as the integral over `UnitAddTorus d` of `mFourier (-n) t • f t`. -/
def mFourierCoeff (f : UnitAddTorus d → E) (n : d → ℤ) : E := ∫ t, mFourier (-n) t • f t


local notation "L²(" α ")" => Lp ℂ 2 (volume : Measure α)


/-- We define `mFourierBasis` to be a `ℤᵈ`-indexed Hilbert basis for the `L²` space of functions
on `UnitAddTorus d`, which by definition is an isometric isomorphism from `L²(UnitAddTorus d)`
to `ℓ²(ℤᵈ, ℂ)`. -/
def mFourierBasis : HilbertBasis (d → ℤ) ℂ L²(UnitAddTorus d) :=
                                                                           /-
                                                                             d : Type u_1
                                                                             inst✝ : Fintype d
                                                                             ⊢ Ne 2 Top.top
                                                                           -/
  HilbertBasis.mk orthonormal_mFourier (span_mFourierLp_closure_eq_top (by norm_num)).ge
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The elements of the Hilbert basis `mFourierBasis` are the functions `mFourierLp 2`, i.e. the
monomials `mFourier n` on `UnitAddTorus d` considered as elements of `L²`. -/
@[simp]
theorem coe_mFourierBasis : ⇑(mFourierBasis (d := d)) = mFourierLp 2 := HilbertBasis.coe_mk _ _


/-- Under the isometric isomorphism `mFourierBasis` from `L²(UnitAddTorus d)` to `ℓ²(ℤᵈ, ℂ)`,
the `i`-th coefficient is `mFourierCoeff f i`. -/
theorem mFourierBasis_repr (f : L²(UnitAddTorus d)) (i : d → ℤ) :
    mFourierBasis.repr f i = mFourierCoeff f i := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
    i : d → Int
    ⊢ Eq (↑(UnitAddTorus.mFourierBasis.repr f) i) (UnitAddTorus.mFourierCoeff (↑↑f …
  -/
  trans ∫ t, conj (mFourierLp 2 i t) * f t
    /-
      d : Type u_1
      inst✝ : Fintype d
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
      i : d → Int
      ⊢ Eq (↑(UnitAddTorus.mFourierBasis.repr f) i) (MeasureTheory.integral MeasureT …
    -/
  · rw [mFourierBasis.repr_apply_apply f i, MeasureTheory.L2.inner_def, coe_mFourierBasis]
    /-
      d : Type u_1
      inst✝ : Fintype d
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
      i : d → Int
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun a => Inner. …
    -/
    simp only [RCLike.inner_apply]
    /-
      🎉 no goals
    -/
    /-
      d : Type u_1
      inst✝ : Fintype d
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
      i : d → Int
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun t => HMul.h …
    -/
  · apply integral_congr_ae
    /-
      case h
      d : Type u_1
      inst✝ : Fintype d
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
      i : d → Int
      ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (fun a =>  …
    -/
    filter_upwards [coeFn_mFourierLp 2 i] with _ ht
    /-
      case h
      d : Type u_1
      inst✝ : Fintype d
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
      i : d → Int
      a✝ : UnitAddTorus d
      ht : Eq (↑↑(UnitAddTorus.mFourierLp 2 i) a✝) ((UnitAddTorus.mFourier i) a✝)
      ⊢ Eq (HMul.hMul ((starRingEnd Complex) (↑↑(UnitAddTorus.mFourierLp 2 i) a✝)) ( …
    -/
    rw [ht, ← mFourier_neg, smul_eq_mul]
    /-
      🎉 no goals
    -/


/-- The Fourier series of an `L2` function `f` sums to `f` in the `L²` norm. -/
theorem hasSum_mFourier_series_L2 (f : L²(UnitAddTorus d)) :
    HasSum (fun i ↦ mFourierCoeff f i • mFourierLp 2 i) f := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheory. …
    ⊢ HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (↑↑f) i) (UnitAddTo …
  -/
  simpa [← coe_mFourierBasis, mFourierBasis_repr] using mFourierBasis.hasSum_repr f
  /-
    🎉 no goals
  -/


/-- **Parseval's identity** for inner products: for `L²` functions `f, g` on `UnitAddTorus d`, the
inner product of the Fourier coefficients of `f` and `g` is the inner product of `f` and `g`. -/
theorem hasSum_prod_mFourierCoeff (f g : L²(UnitAddTorus d)) :
    HasSum (fun i ↦ conj (mFourierCoeff f i) * (mFourierCoeff g i)) (∫ t, conj (f t) * g t) := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheor …
    ⊢ HasSum (fun i => HMul.hMul ((starRingEnd Complex) (UnitAddTorus.mFourierCoef …
  -/
  refine HasSum.congr_fun (mFourierBasis.hasSum_inner_mul_inner f g) (fun n ↦ ?_)
  /-
    d : Type u_1
    inst✝ : Fintype d
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 MeasureTheor …
    n : d → Int
    ⊢ Eq (HMul.hMul ((starRingEnd Complex) (UnitAddTorus.mFourierCoeff (↑↑f) n)) ( …
  -/
  simp only [← mFourierBasis_repr, HilbertBasis.repr_apply_apply, inner_conj_symm]
  /-
    🎉 no goals
  -/


/-- **Parseval's identity** for norms: for an `L²` function `f` on `UnitAddTorus d`, the sum of the
squared norms of the Fourier coefficients equals the `L²` norm of `f`. -/
theorem hasSum_sq_mFourierCoeff (f : L²(UnitAddTorus d)) :
    HasSum (fun i ↦ ‖mFourierCoeff f i‖ ^ 2) (∫ t, ‖f t‖ ^ 2) := by
  simpa only [← RCLike.inner_apply, inner_self_eq_norm_sq, ← integral_re
    (L2.integrable_inner f f)] using RCLike.hasSum_re ℂ (hasSum_prod_mFourierCoeff f f)


theorem mFourierCoeff_toLp (n : d → ℤ) :
    mFourierCoeff (f.toLp 2 volume ℂ) n = mFourierCoeff f n :=
  integral_congr_ae (ae_eq_rfl.mul <| f.coeFn_toAEEqFun _)


/-- If the sequence of Fourier coefficients of `f` is summable, then the Fourier series converges
uniformly to `f`. -/
theorem hasSum_mFourier_series_of_summable (h : Summable (mFourierCoeff f)) :
    HasSum (fun i ↦ mFourierCoeff f i • mFourier i) f := by
  /-
    d : Type u_1
    inst✝ : Fintype d
    f : ContinuousMap (UnitAddTorus d) Complex
    h : Summable (UnitAddTorus.mFourierCoeff ⇑f)
    ⊢ HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (⇑f) i) (UnitAddTor …
  -/
  have sum_L2 := hasSum_mFourier_series_L2 (ContinuousMap.toLp 2 volume ℂ f)
  /-
    d : Type u_1
    inst✝ : Fintype d
    f : ContinuousMap (UnitAddTorus d) Complex
    h : Summable (UnitAddTorus.mFourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (↑↑((Continu …
    ⊢ HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (⇑f) i) (UnitAddTor …
  -/
  simp only [mFourierCoeff_toLp] at sum_L2
  /-
    d : Type u_1
    inst✝ : Fintype d
    f : ContinuousMap (UnitAddTorus d) Complex
    h : Summable (UnitAddTorus.mFourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (⇑f) i) (Uni …
    ⊢ HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (⇑f) i) (UnitAddTor …
  -/
  refine ContinuousMap.hasSum_of_hasSum_Lp (.of_norm ?_) sum_L2
  /-
    d : Type u_1
    inst✝ : Fintype d
    f : ContinuousMap (UnitAddTorus d) Complex
    h : Summable (UnitAddTorus.mFourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (UnitAddTorus.mFourierCoeff (⇑f) i) (Uni …
    ⊢ Summable fun a => Norm.norm (HSMul.hSMul (UnitAddTorus.mFourierCoeff (⇑f) a) …
  -/
  simpa only [norm_smul, mFourier_norm, mul_one] using h.norm
  /-
    🎉 no goals
  -/


/-- If the sequence of Fourier coefficients of `f` is summable, then the Fourier series of `f`
converges everywhere pointwise to `f`. -/
theorem hasSum_mFourier_series_apply_of_summable (h : Summable (mFourierCoeff f))
    (x : UnitAddTorus d) : HasSum (fun i ↦ mFourierCoeff f i • mFourier i x) (f x) := by
  simpa only [_root_.map_smul] using (ContinuousMap.evalCLM ℂ x).hasSum
    (hasSum_mFourier_series_of_summable h)


