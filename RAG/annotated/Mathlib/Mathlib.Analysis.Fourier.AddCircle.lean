/-- Haar measure on the additive circle, normalised to have total measure 1. -/
def haarAddCircle : Measure (AddCircle T) :=
  addHaarMeasure ⊤

-- Porting note: was `deriving IsAddHaarMeasure` on `haarAddCircle`

instance : IsAddHaarMeasure (@haarAddCircle T _) :=
  Measure.isAddHaarMeasure_addHaarMeasure ⊤


instance : IsProbabilityMeasure (@haarAddCircle T _) :=
  IsProbabilityMeasure.mk addHaarMeasure_self


theorem volume_eq_smul_haarAddCircle :
    (volume : Measure (AddCircle T)) = ENNReal.ofReal T • (@haarAddCircle T _) :=
  rfl


/-- The family of exponential monomials `fun x => exp (2 π i n x / T)`, parametrized by `n : ℤ` and
considered as bundled continuous maps from `ℝ / ℤ • T` to `ℂ`. -/
def fourier (n : ℤ) : C(AddCircle T, ℂ) where
  toFun x := toCircle (n • x :)
  continuous_toFun := continuous_induced_dom.comp <| continuous_toCircle.comp <| continuous_zsmul _


@[simp]
theorem fourier_apply {n : ℤ} {x : AddCircle T} : fourier n x = toCircle (n • x :) :=
  rfl

-- simp normal form is `fourier_coe_apply'`

theorem fourier_coe_apply {n : ℤ} {x : ℝ} :
    fourier n (x : AddCircle T) = Complex.exp (2 * π * Complex.I * n * x / T) := by
  rw [fourier_apply, ← QuotientAddGroup.mk_zsmul, toCircle, Function.Periodic.lift_coe,
    Circle.coe_exp, Complex.ofReal_mul, Complex.ofReal_div, Complex.ofReal_mul, zsmul_eq_mul,
    Complex.ofReal_mul, Complex.ofReal_intCast]
  /-
    T : Real
    n : Int
    x : Real
    ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul ↑2 ↑Real.pi) ↑T) …
  -/
  norm_num
  /-
    T : Real
    n : Int
    x : Real
    ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul 2 ↑Real.pi) ↑T)  …
  -/
  congr 1; ring
           /-
             🎉 no goals
           -/


@[simp]
theorem fourier_coe_apply' {n : ℤ} {x : ℝ} :
    toCircle (n • (x : AddCircle T) :) = Complex.exp (2 * π * Complex.I * n * x / T) := by
  /-
    T : Real
    n : Int
    x : Real
    ⊢ Eq (↑(AddCircle.toCircle (HSMul.hSMul n ↑x))) (Complex.exp (HDiv.hDiv (HMul. …
  -/
  rw [← fourier_apply]; exact fourier_coe_apply
                        /-
                          🎉 no goals
                        -/

-- simp normal form is `fourier_zero'`

theorem fourier_zero {x : AddCircle T} : fourier 0 x = 1 := by
  /-
    T : Real
    x : AddCircle T
    ⊢ Eq ((fourier 0) x) 1
  -/
  induction x using QuotientAddGroup.induction_on
  /-
    case H
    T z✝ : Real
    ⊢ Eq ((fourier 0) ↑z✝) 1
  -/
  simp only [fourier_coe_apply]
  /-
    case H
    T z✝ : Real
    ⊢ Eq (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Re …
  -/
  norm_num
  /-
    🎉 no goals
  -/


theorem fourier_zero' {x : AddCircle T} : @toCircle T 0 = (1 : ℂ) := by
  /-
    T : Real
    x : AddCircle T
    ⊢ Eq (↑(AddCircle.toCircle 0)) 1
  -/
  have : fourier 0 x = @toCircle T 0 := by rw [fourier_apply, zero_smul]
  /-
    T : Real
    x : AddCircle T
    this : Eq ((fourier 0) x) ↑(AddCircle.toCircle 0)
    ⊢ Eq (↑(AddCircle.toCircle 0)) 1
  -/
  rw [← this]; exact fourier_zero
               /-
                 🎉 no goals
               -/

-- simp normal form is *also* `fourier_zero'`

theorem fourier_eval_zero (n : ℤ) : fourier n (0 : AddCircle T) = 1 := by
  rw [← QuotientAddGroup.mk_zero, fourier_coe_apply, Complex.ofReal_zero, mul_zero,
    zero_div, Complex.exp_zero]


                                                                       /-
                                                                         T : Real
                                                                         x : AddCircle T
                                                                         ⊢ Eq ((fourier 1) x) ↑x.toCircle
                                                                       -/
theorem fourier_one {x : AddCircle T} : fourier 1 x = toCircle x := by rw [fourier_apply, one_zsmul]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

-- simp normal form is `fourier_neg'`

theorem fourier_neg {n : ℤ} {x : AddCircle T} : fourier (-n) x = conj (fourier n x) := by
  /-
    T : Real
    n : Int
    x : AddCircle T
    ⊢ Eq ((fourier (Neg.neg n)) x) ((starRingEnd Complex) ((fourier n) x))
  -/
  induction x using QuotientAddGroup.induction_on
  /-
    case H
    T : Real
    n : Int
    z✝ : Real
    ⊢ Eq ((fourier (Neg.neg n)) ↑z✝) ((starRingEnd Complex) ((fourier n) ↑z✝))
  -/
  simp_rw [fourier_apply, toCircle]
  /-
    case H
    T : Real
    n : Int
    z✝ : Real
    ⊢ Eq (↑(⋯.lift (HSMul.hSMul (Neg.neg n) ↑z✝))) ((starRingEnd Complex) ↑(⋯.lift …
  -/
  rw [← QuotientAddGroup.mk_zsmul, ← QuotientAddGroup.mk_zsmul]
  simp_rw [Function.Periodic.lift_coe, ← Circle.coe_inv_eq_conj, ← Circle.exp_neg,
    neg_smul, mul_neg]


@[simp]
theorem fourier_neg' {n : ℤ} {x : AddCircle T} : @toCircle T (-(n • x)) = conj (fourier n x) := by
  /-
    T : Real
    n : Int
    x : AddCircle T
    ⊢ Eq (↑(Neg.neg (HSMul.hSMul n x)).toCircle) ((starRingEnd Complex) ((fourier  …
  -/
  rw [← neg_smul, ← fourier_apply]; exact fourier_neg
                                    /-
                                      🎉 no goals
                                    -/

-- simp normal form is `fourier_add'`

theorem fourier_add {m n : ℤ} {x : AddCircle T} : fourier (m+n) x = fourier m x * fourier n x := by
  /-
    T : Real
    m n : Int
    x : AddCircle T
    ⊢ Eq ((fourier (HAdd.hAdd m n)) x) (HMul.hMul ((fourier m) x) ((fourier n) x))
  -/
  simp_rw [fourier_apply, add_zsmul, toCircle_add, Circle.coe_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem fourier_add' {m n : ℤ} {x : AddCircle T} :
    toCircle ((m + n) • x :) = fourier m x * fourier n x := by
  /-
    T : Real
    m n : Int
    x : AddCircle T
    ⊢ Eq (↑(HSMul.hSMul (HAdd.hAdd m n) x).toCircle) (HMul.hMul ((fourier m) x) (( …
  -/
  rw [← fourier_apply]; exact fourier_add
                        /-
                          🎉 no goals
                        -/


theorem fourier_norm [Fact (0 < T)] (n : ℤ) : ‖@fourier T n‖ = 1 := by
  /-
    T : Real
    inst✝ : Fact (LT.lt 0 T)
    n : Int
    ⊢ Eq (Norm.norm (fourier n)) 1
  -/
  rw [ContinuousMap.norm_eq_iSup_norm]
  /-
    T : Real
    inst✝ : Fact (LT.lt 0 T)
    n : Int
    ⊢ Eq (iSup fun x => Norm.norm ((fourier n) x)) 1
  -/
  have : ∀ x : AddCircle T, ‖fourier n x‖ = 1 := fun x => Circle.abs_coe _
  /-
    T : Real
    inst✝ : Fact (LT.lt 0 T)
    n : Int
    this : ∀ (x : AddCircle T), Eq (Norm.norm ((fourier n) x)) 1
    ⊢ Eq (iSup fun x => Norm.norm ((fourier n) x)) 1
  -/
  simp_rw [this]
  /-
    T : Real
    inst✝ : Fact (LT.lt 0 T)
    n : Int
    this : ∀ (x : AddCircle T), Eq (Norm.norm ((fourier n) x)) 1
    ⊢ Eq (iSup fun x => 1) 1
  -/
  exact @ciSup_const _ _ _ Zero.instNonempty _
  /-
    🎉 no goals
  -/


/-- For `n ≠ 0`, a translation by `T / 2 / n` negates the function `fourier n`. -/
theorem fourier_add_half_inv_index {n : ℤ} (hn : n ≠ 0) (hT : 0 < T) (x : AddCircle T) :
    @fourier T n (x + ↑(T / 2 / n)) = -fourier n x := by
  /-
    T : Real
    n : Int
    hn : Ne n 0
    hT : LT.lt 0 T
    x : AddCircle T
    ⊢ Eq ((fourier n) (HAdd.hAdd x ↑(HDiv.hDiv (HDiv.hDiv T 2) ↑n))) (Neg.neg ((fo …
  -/
  rw [fourier_apply, zsmul_add, ← QuotientAddGroup.mk_zsmul, toCircle_add, coe_mul_unitSphere]
  /-
    T : Real
    n : Int
    hn : Ne n 0
    hT : LT.lt 0 T
    x : AddCircle T
    ⊢ Eq (HMul.hMul ↑(HSMul.hSMul n x).toCircle ↑(AddCircle.toCircle ↑(HSMul.hSMul …
  -/
  have : (n : ℂ) ≠ 0 := by simpa using hn
  have : (@toCircle T (n • (T / 2 / n) : ℝ) : ℂ) = -1 := by
    rw [zsmul_eq_mul, toCircle, Function.Periodic.lift_coe, Circle.coe_exp]
    replace hT := Complex.ofReal_ne_zero.mpr hT.ne'
    convert Complex.exp_pi_mul_I using 3
    field_simp; ring
  /-
    T : Real
    n : Int
    hn : Ne n 0
    hT : LT.lt 0 T
    x : AddCircle T
    this✝ : Ne (↑n) 0
    this : Eq (↑(AddCircle.toCircle ↑(HSMul.hSMul n (HDiv.hDiv (HDiv.hDiv T 2) ↑n) …
    ⊢ Eq (HMul.hMul ↑(HSMul.hSMul n x).toCircle ↑(AddCircle.toCircle ↑(HSMul.hSMul …
  -/
  rw [this]; simp
             /-
               🎉 no goals
             -/


/-- The star subalgebra of `C(AddCircle T, ℂ)` generated by `fourier n` for `n ∈ ℤ` . -/
def fourierSubalgebra : StarSubalgebra ℂ C(AddCircle T, ℂ) where
  toSubalgebra := Algebra.adjoin ℂ (range fourier)
  star_mem' := by
    show Algebra.adjoin ℂ (range (fourier (T := T))) ≤
      star (Algebra.adjoin ℂ (range (fourier (T := T))))
    /-
      T : Real
      ⊢ LE.le (Algebra.adjoin Complex (Set.range fourier)) (Star.star (Algebra.adjoi …
    -/
    refine adjoin_le ?_
    /-
      T : Real
      ⊢ HasSubset.Subset (Set.range fourier) ↑(Star.star (Algebra.adjoin Complex (Se …
    -/
    rintro - ⟨n, rfl⟩
    /-
      case intro
      T : Real
      n : Int
      ⊢ Membership.mem (↑(Star.star (Algebra.adjoin Complex (Set.range fourier)))) ( …
    -/
    exact subset_adjoin ⟨-n, ext fun _ => fourier_neg⟩
    /-
      🎉 no goals
    -/


/-- The star subalgebra of `C(AddCircle T, ℂ)` generated by `fourier n` for `n ∈ ℤ` is in fact the
linear span of these functions. -/
theorem fourierSubalgebra_coe :
    Subalgebra.toSubmodule (@fourierSubalgebra T).toSubalgebra = span ℂ (range (@fourier T)) := by
  /-
    T : Real
    ⊢ Eq (Subalgebra.toSubmodule fourierSubalgebra.toSubalgebra) (Submodule.span C …
  -/
  apply adjoin_eq_span_of_subset
  /-
    case hs
    T : Real
    ⊢ HasSubset.Subset ↑(Submonoid.closure (Set.range fourier)) ↑(Submodule.span C …
  -/
  refine Subset.trans ?_ Submodule.subset_span
  /-
    case hs
    T : Real
    ⊢ HasSubset.Subset (↑(Submonoid.closure (Set.range fourier))) (Set.range fouri …
  -/
  intro x hx
  /-
    case hs
    T : Real
    x : ContinuousMap (AddCircle T) Complex
    hx : Membership.mem (↑(Submonoid.closure (Set.range fourier))) x
    ⊢ Membership.mem (Set.range fourier) x
  -/
  refine Submonoid.closure_induction (fun _ => id) ⟨0, ?_⟩ ?_ hx
    /-
      case hs.refine_1
      T : Real
      x : ContinuousMap (AddCircle T) Complex
      hx : Membership.mem (↑(Submonoid.closure (Set.range fourier))) x
      ⊢ Eq (fourier 0) 1
    -/
  · ext1 z; exact fourier_zero
            /-
              🎉 no goals
            -/
    /-
      case hs.refine_2
      T : Real
      x : ContinuousMap (AddCircle T) Complex
      hx : Membership.mem (↑(Submonoid.closure (Set.range fourier))) x
      ⊢ ∀ (x y : ContinuousMap (AddCircle T) Complex), Membership.mem (Submonoid.clo …
    -/
  · rintro - - - - ⟨m, rfl⟩ ⟨n, rfl⟩
    /-
      case hs.refine_2.intro.intro
      T : Real
      x : ContinuousMap (AddCircle T) Complex
      hx : Membership.mem (↑(Submonoid.closure (Set.range fourier))) x
      m n : Int
      ⊢ Membership.mem (Set.range fourier) (HMul.hMul (fourier m) (fourier n))
    -/
    refine ⟨m + n, ?_⟩
    /-
      case hs.refine_2.intro.intro
      T : Real
      x : ContinuousMap (AddCircle T) Complex
      hx : Membership.mem (↑(Submonoid.closure (Set.range fourier))) x
      m n : Int
      ⊢ Eq (fourier (HAdd.hAdd m n)) (HMul.hMul (fourier m) (fourier n))
    -/
    ext1 z
    /-
      case hs.refine_2.intro.intro.h
      T : Real
      x : ContinuousMap (AddCircle T) Complex
      hx : Membership.mem (↑(Submonoid.closure (Set.range fourier))) x
      m n : Int
      z : AddCircle T
      ⊢ Eq ((fourier (HAdd.hAdd m n)) z) ((HMul.hMul (fourier m) (fourier n)) z)
    -/
    exact fourier_add
    /-
      🎉 no goals
    -/

/- a post-port refactor made `fourierSubalgebra` into a `StarSubalgebra`, and eliminated
`conjInvariantSubalgebra` entirely, making this lemma irrelevant. -/


/-- The subalgebra of `C(AddCircle T, ℂ)` generated by `fourier n` for `n ∈ ℤ`
separates points. -/
theorem fourierSubalgebra_separatesPoints : (@fourierSubalgebra T).SeparatesPoints := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ fourierSubalgebra.SeparatesPoints
  -/
  intro x y hxy
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x y : AddCircle T
    hxy : Ne x y
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑fourierSubalge …
  -/
  refine ⟨_, ⟨fourier 1, subset_adjoin ⟨1, rfl⟩, rfl⟩, ?_⟩
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x y : AddCircle T
    hxy : Ne x y
    ⊢ Ne ((fun f => ⇑f) (fourier 1) x) ((fun f => ⇑f) (fourier 1) y)
  -/
  dsimp only; rw [fourier_one, fourier_one]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x y : AddCircle T
    hxy : Ne x y
    ⊢ Ne ↑x.toCircle ↑y.toCircle
  -/
  contrapose! hxy
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x y : AddCircle T
    hxy : Eq ↑x.toCircle ↑y.toCircle
    ⊢ Eq x y
  -/
  rw [Subtype.coe_inj] at hxy
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x y : AddCircle T
    hxy : Eq x.toCircle y.toCircle
    ⊢ Eq x y
  -/
  exact injective_toCircle hT.elim.ne' hxy
  /-
    🎉 no goals
  -/


/-- The subalgebra of `C(AddCircle T, ℂ)` generated by `fourier n` for `n ∈ ℤ` is dense. -/
theorem fourierSubalgebra_closure_eq_top : (@fourierSubalgebra T).topologicalClosure = ⊤ :=
  ContinuousMap.starSubalgebra_topologicalClosure_eq_top_of_separatesPoints fourierSubalgebra
    fourierSubalgebra_separatesPoints


/-- The linear span of the monomials `fourier n` is dense in `C(AddCircle T, ℂ)`. -/
theorem span_fourier_closure_eq_top : (span ℂ (range <| @fourier T)).topologicalClosure = ⊤ := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ Eq (Submodule.span Complex (Set.range fourier)).topologicalClosure Top.top
  -/
  rw [← fourierSubalgebra_coe]
  exact congr_arg (Subalgebra.toSubmodule <| StarSubalgebra.toSubalgebra ·)
    fourierSubalgebra_closure_eq_top


/-- The family of monomials `fourier n`, parametrized by `n : ℤ` and considered as
elements of the `Lp` space of functions `AddCircle T → ℂ`. -/
abbrev fourierLp (p : ℝ≥0∞) [Fact (1 ≤ p)] (n : ℤ) : Lp ℂ p (@haarAddCircle T hT) :=
  toLp (E := ℂ) p haarAddCircle ℂ (fourier n)


theorem coeFn_fourierLp (p : ℝ≥0∞) [Fact (1 ≤ p)] (n : ℤ) :
    @fourierLp T hT p _ n =ᵐ[haarAddCircle] fourier n :=
  coeFn_toLp haarAddCircle (fourier n)


/-- For each `1 ≤ p < ∞`, the linear span of the monomials `fourier n` is dense in
`Lp ℂ p haarAddCircle`. -/
theorem span_fourierLp_closure_eq_top {p : ℝ≥0∞} [Fact (1 ≤ p)] (hp : p ≠ ∞) :
    (span ℂ (range (@fourierLp T _ p _))).topologicalClosure = ⊤ := by
  convert
    (ContinuousMap.toLp_denseRange ℂ (@haarAddCircle T hT) ℂ hp).topologicalClosure_map_submodule
      span_fourier_closure_eq_top
  /-
    case h.e'_2.h.e'_9.h.h
    T : Real
    hT : Fact (LT.lt 0 T)
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    e_3✝ : Eq Complex.instSemiring DivisionSemiring.toSemiring
    he✝ : Eq MeasureTheory.Lp.instModule MeasureTheory.Lp.instModule
    ⊢ Eq (Submodule.span Complex (Set.range (fourierLp p))) (Submodule.map (↑(Cont …
  -/
  erw [map_span, range_comp]
  /-
    case h.e'_2.h.e'_9.h.h
    T : Real
    hT : Fact (LT.lt 0 T)
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    hp : Ne p Top.top
    e_3✝ : Eq Complex.instSemiring DivisionSemiring.toSemiring
    he✝ : Eq MeasureTheory.Lp.instModule MeasureTheory.Lp.instModule
    ⊢ Eq (Submodule.span Complex (Set.image (⇑(ContinuousMap.toLp p AddCircle.haar …
  -/
  simp only [ContinuousLinearMap.coe_coe]
  /-
    🎉 no goals
  -/


/-- The monomials `fourier n` are an orthonormal set with respect to normalised Haar measure. -/
theorem orthonormal_fourier : Orthonormal ℂ (@fourierLp T _ 2 _) := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ Orthonormal Complex (fourierLp 2)
  -/
  rw [orthonormal_iff_ite]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ⊢ ∀ (i j : Int), Eq (Inner.inner (fourierLp 2 i) (fourierLp 2 j)) (ite (Eq i j …
  -/
  intro i j
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    i j : Int
    ⊢ Eq (Inner.inner (fourierLp 2 i) (fourierLp 2 j)) (ite (Eq i j) 1 0)
  -/
  rw [ContinuousMap.inner_toLp (@haarAddCircle T hT) (fourier i) (fourier j)]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    i j : Int
    ⊢ Eq (MeasureTheory.integral AddCircle.haarAddCircle fun x => HMul.hMul ((star …
  -/
  simp_rw [← fourier_neg, ← fourier_add]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    i j : Int
    ⊢ Eq (MeasureTheory.integral AddCircle.haarAddCircle fun x => (fourier (HAdd.h …
  -/
  split_ifs with h
    /-
      case pos
      T : Real
      hT : Fact (LT.lt 0 T)
      i j : Int
      h : Eq i j
      ⊢ Eq (MeasureTheory.integral AddCircle.haarAddCircle fun x => (fourier (HAdd.h …
    -/
  · simp_rw [h, neg_add_cancel]
    /-
      case pos
      T : Real
      hT : Fact (LT.lt 0 T)
      i j : Int
      h : Eq i j
      ⊢ Eq (MeasureTheory.integral AddCircle.haarAddCircle fun x => (fourier 0) x) 1
    -/
    have : ⇑(@fourier T 0) = (fun _ => 1 : AddCircle T → ℂ) := by ext1; exact fourier_zero
    rw [this, integral_const, measure_univ, ENNReal.one_toReal, Complex.real_smul,
      Complex.ofReal_one, mul_one]
  have hij : -i + j ≠ 0 := by
    rw [add_comm]
    exact sub_ne_zero.mpr (Ne.symm h)
  convert integral_eq_zero_of_add_right_eq_neg (μ := haarAddCircle)
    (fourier_add_half_inv_index hij hT.elim)


/-- The `n`-th Fourier coefficient of a function `AddCircle T → E`, for `E` a complete normed
`ℂ`-vector space, defined as the integral over `AddCircle T` of `fourier (-n) t • f t`. -/
def fourierCoeff (f : AddCircle T → E) (n : ℤ) : E :=
  ∫ t : AddCircle T, fourier (-n) t • f t ∂haarAddCircle


/-- The Fourier coefficients of a function on `AddCircle T` can be computed as an integral
over `[a, a + T]`, for any real `a`. -/
theorem fourierCoeff_eq_intervalIntegral (f : AddCircle T → E) (n : ℤ) (a : ℝ) :
    fourierCoeff f n = (1 / T) • ∫ x in a..a + T, @fourier T (-n) x • f x := by
  have : ∀ x : ℝ, @fourier T (-n) x • f x = (fun z : AddCircle T => @fourier T (-n) z • f z) x := by
    intro x; rfl
  -- After https://github.com/leanprover/lean4/pull/3124, we need to add `singlePass := true` to avoid an infinite loop.
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : AddCircle T → E
    n : Int
    a : Real
    this : ∀ (x : Real), Eq (HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (f ↑x)) ((fun  …
    ⊢ Eq (fourierCoeff f n) (HSMul.hSMul (HDiv.hDiv 1 T) (intervalIntegral (fun x  …
  -/
  simp_rw (config := {singlePass := true}) [this]
  rw [fourierCoeff, AddCircle.intervalIntegral_preimage T a (fun z => _ • _),
    volume_eq_smul_haarAddCircle, integral_smul_measure, ENNReal.toReal_ofReal hT.out.le,
    ← smul_assoc, smul_eq_mul, one_div_mul_cancel hT.out.ne', one_smul]


theorem fourierCoeff.const_smul (f : AddCircle T → E) (c : ℂ) (n : ℤ) :
    fourierCoeff (c • f :) n = c • fourierCoeff f n := by
  simp_rw [fourierCoeff, Pi.smul_apply, ← smul_assoc, smul_eq_mul, mul_comm, ← smul_eq_mul,
    smul_assoc, integral_smul]


theorem fourierCoeff.const_mul (f : AddCircle T → ℂ) (c : ℂ) (n : ℤ) :
    fourierCoeff (fun x => c * f x) n = c * fourierCoeff f n :=
  fourierCoeff.const_smul f c n


/-- For a function on `ℝ`, the Fourier coefficients of `f` on `[a, b]` are defined as the
Fourier coefficients of the unique periodic function agreeing with `f` on `Ioc a b`. -/
def fourierCoeffOn {a b : ℝ} (hab : a < b) (f : ℝ → E) (n : ℤ) : E :=
                       /-
                         T : Real
                         hT : Fact (LT.lt 0 T)
                         E : Type
                         inst✝¹ : NormedAddCommGroup E
                         inst✝ : NormedSpace Complex E
                         a b : Real
                         hab : LT.lt a b
                         f : Real → E
                         n : Int
                         ⊢ LT.lt 0 (HSub.hSub b a)
                       -/
  haveI := Fact.mk (by linarith : 0 < b - a)
                       /-
                         🎉 no goals
                       -/
  fourierCoeff (AddCircle.liftIoc (b - a) a f) n


theorem fourierCoeffOn_eq_integral {a b : ℝ} (f : ℝ → E) (n : ℤ) (hab : a < b) :
    fourierCoeffOn hab f n =
      (1 / (b - a)) • ∫ x in a..b, fourier (-n) (x : AddCircle (b - a)) • f x := by
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    ⊢ Eq (fourierCoeffOn hab f n) (HSMul.hSMul (HDiv.hDiv 1 (HSub.hSub b a)) (inte …
  -/
  haveI := Fact.mk (by linarith : 0 < b - a)
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (fourierCoeffOn hab f n) (HSMul.hSMul (HDiv.hDiv 1 (HSub.hSub b a)) (inte …
  -/
  rw [fourierCoeffOn, fourierCoeff_eq_intervalIntegral _ _ a, add_sub, add_sub_cancel_left]
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv 1 (HSub.hSub b a)) (intervalIntegral (fun x => HS …
  -/
  congr 1
  /-
    case e_a
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (intervalIntegral (fun x => HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (AddCi …
  -/
  simp_rw [intervalIntegral.integral_of_le hab.le]
  /-
    case e_a
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioc fun x hx => ?_
  /-
    case e_a
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    x : Real
    hx : Membership.mem (Set.Ioc a b) x
    ⊢ Eq (HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (AddCircle.liftIoc (HSub.hSub b a …
  -/
  rw [liftIoc_coe_apply]
  /-
    case e_a
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    x : Real
    hx : Membership.mem (Set.Ioc a b) x
    ⊢ Membership.mem (Set.Ioc a (HAdd.hAdd a (HSub.hSub b a))) x
  -/
  rwa [add_sub, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


theorem fourierCoeffOn.const_smul {a b : ℝ} (f : ℝ → E) (c : ℂ) (n : ℤ) (hab : a < b) :
    fourierCoeffOn hab (c • f) n = c • fourierCoeffOn hab f n := by
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    c : Complex
    n : Int
    hab : LT.lt a b
    ⊢ Eq (fourierCoeffOn hab (HSMul.hSMul c f) n) (HSMul.hSMul c (fourierCoeffOn h …
  -/
  haveI := Fact.mk (by linarith : 0 < b - a)
  /-
    E : Type
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b : Real
    f : Real → E
    c : Complex
    n : Int
    hab : LT.lt a b
    this : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (fourierCoeffOn hab (HSMul.hSMul c f) n) (HSMul.hSMul c (fourierCoeffOn h …
  -/
  apply fourierCoeff.const_smul
  /-
    🎉 no goals
  -/


theorem fourierCoeffOn.const_mul {a b : ℝ} (f : ℝ → ℂ) (c : ℂ) (n : ℤ) (hab : a < b) :
    fourierCoeffOn hab (fun x => c * f x) n = c * fourierCoeffOn hab f n :=
  fourierCoeffOn.const_smul _ _ _ _


theorem fourierCoeff_liftIoc_eq {a : ℝ} (f : ℝ → ℂ) (n : ℤ) :
    fourierCoeff (AddCircle.liftIoc T a f) n =
    fourierCoeffOn (lt_add_of_pos_right a hT.out) f n := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    ⊢ Eq (fourierCoeff (AddCircle.liftIoc T a f) n) (fourierCoeffOn ⋯ f n)
  -/
  rw [fourierCoeffOn_eq_integral, fourierCoeff_eq_intervalIntegral, add_sub_cancel_left a T]
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      a : Real
      f : Real → Complex
      n : Int
      ⊢ Eq (HSMul.hSMul (HDiv.hDiv 1 T) (intervalIntegral (fun x => HSMul.hSMul ((fo …
    -/
  · congr 1
    /-
      case e_a
      T : Real
      hT : Fact (LT.lt 0 T)
      a : Real
      f : Real → Complex
      n : Int
      ⊢ Eq (intervalIntegral (fun x => HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (AddCi …
    -/
    refine intervalIntegral.integral_congr_ae (ae_of_all _ fun x hx => ?_)
    /-
      case e_a
      T : Real
      hT : Fact (LT.lt 0 T)
      a : Real
      f : Real → Complex
      n : Int
      x : Real
      hx : Membership.mem (Set.uIoc a (HAdd.hAdd a T)) x
      ⊢ Eq (HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (AddCircle.liftIoc T a f ↑x)) (HS …
    -/
    rw [liftIoc_coe_apply]
    /-
      case e_a
      T : Real
      hT : Fact (LT.lt 0 T)
      a : Real
      f : Real → Complex
      n : Int
      x : Real
      hx : Membership.mem (Set.uIoc a (HAdd.hAdd a T)) x
      ⊢ Membership.mem (Set.Ioc a (HAdd.hAdd a T)) x
    -/
    rwa [uIoc_of_le (lt_add_of_pos_right a hT.out).le] at hx
    /-
      🎉 no goals
    -/


theorem fourierCoeff_liftIco_eq {a : ℝ} (f : ℝ → ℂ) (n : ℤ) :
    fourierCoeff (AddCircle.liftIco T a f) n =
    fourierCoeffOn (lt_add_of_pos_right a hT.out) f n := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    ⊢ Eq (fourierCoeff (AddCircle.liftIco T a f) n) (fourierCoeffOn ⋯ f n)
  -/
  rw [fourierCoeffOn_eq_integral, fourierCoeff_eq_intervalIntegral _ _ a, add_sub_cancel_left a T]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv 1 T) (intervalIntegral (fun x => HSMul.hSMul ((fo …
  -/
  congr 1
  /-
    case e_a
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    ⊢ Eq (intervalIntegral (fun x => HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (AddCi …
  -/
  simp_rw [intervalIntegral.integral_of_le (lt_add_of_pos_right a hT.out).le]
  /-
    case e_a
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  iterate 2 rw [integral_Ioc_eq_integral_Ioo]
  /-
    case e_a
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioo fun x hx => ?_
  /-
    case e_a
    T : Real
    hT : Fact (LT.lt 0 T)
    a : Real
    f : Real → Complex
    n : Int
    x : Real
    hx : Membership.mem (Set.Ioo a (HAdd.hAdd a T)) x
    ⊢ Eq (HSMul.hSMul ((fourier (Neg.neg n)) ↑x) (AddCircle.liftIco T a f ↑x)) (HS …
  -/
  rw [liftIco_coe_apply (Ioo_subset_Ico_self hx)]
  /-
    🎉 no goals
  -/


/-- We define `fourierBasis` to be a `ℤ`-indexed Hilbert basis for `Lp ℂ 2 haarAddCircle`,
which by definition is an isometric isomorphism from `Lp ℂ 2 haarAddCircle` to `ℓ²(ℤ, ℂ)`. -/
def fourierBasis : HilbertBasis ℤ ℂ (Lp ℂ 2 <| @haarAddCircle T hT) :=
                                                                         /-
                                                                           T : Real
                                                                           hT : Fact (LT.lt 0 T)
                                                                           ⊢ Ne 2 Top.top
                                                                         -/
  HilbertBasis.mk orthonormal_fourier (span_fourierLp_closure_eq_top (by norm_num)).ge
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The elements of the Hilbert basis `fourierBasis` are the functions `fourierLp 2`, i.e. the
monomials `fourier n` on the circle considered as elements of `L²`. -/
@[simp]
theorem coe_fourierBasis : ⇑(@fourierBasis T hT) = @fourierLp T hT 2 _ :=
  HilbertBasis.coe_mk _ _


/-- Under the isometric isomorphism `fourierBasis` from `Lp ℂ 2 haarAddCircle` to `ℓ²(ℤ, ℂ)`, the
`i`-th coefficient is `fourierCoeff f i`, i.e., the integral over `AddCircle T` of
`fun t => fourier (-i) t * f t` with respect to the Haar measure of total mass 1. -/
theorem fourierBasis_repr (f : Lp ℂ 2 <| @haarAddCircle T hT) (i : ℤ) :
    fourierBasis.repr f i = fourierCoeff f i := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    i : Int
    ⊢ Eq (↑(fourierBasis.repr f) i) (fourierCoeff (↑↑f) i)
  -/
  trans ∫ t : AddCircle T, conj ((@fourierLp T hT 2 _ i : AddCircle T → ℂ) t) * f t ∂haarAddCircle
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      i : Int
      ⊢ Eq (↑(fourierBasis.repr f) i) (MeasureTheory.integral AddCircle.haarAddCircl …
    -/
  · rw [fourierBasis.repr_apply_apply f i, MeasureTheory.L2.inner_def, coe_fourierBasis]
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      i : Int
      ⊢ Eq (MeasureTheory.integral AddCircle.haarAddCircle fun a => Inner.inner (↑↑( …
    -/
    simp only [RCLike.inner_apply]
    /-
      🎉 no goals
    -/
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      i : Int
      ⊢ Eq (MeasureTheory.integral AddCircle.haarAddCircle fun t => HMul.hMul ((star …
    -/
  · apply integral_congr_ae
    /-
      case h
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      i : Int
      ⊢ (MeasureTheory.ae AddCircle.haarAddCircle).EventuallyEq (fun a => HMul.hMul  …
    -/
    filter_upwards [coeFn_fourierLp 2 i] with _ ht
    /-
      case h
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      i : Int
      a✝ : AddCircle T
      ht : Eq (↑↑(fourierLp 2 i) a✝) ((fourier i) a✝)
      ⊢ Eq (HMul.hMul ((starRingEnd Complex) (↑↑(fourierLp 2 i) a✝)) (↑↑f a✝)) (HSMu …
    -/
    rw [ht, ← fourier_neg, smul_eq_mul]
    /-
      🎉 no goals
    -/


/-- The Fourier series of an `L2` function `f` sums to `f`, in the `L²` space of `AddCircle T`. -/
theorem hasSum_fourier_series_L2 (f : Lp ℂ 2 <| @haarAddCircle T hT) :
    HasSum (fun i => fourierCoeff f i • fourierLp 2 i) f := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    ⊢ HasSum (fun i => HSMul.hSMul (fourierCoeff (↑↑f) i) (fourierLp 2 i)) f
  -/
  simp_rw [← fourierBasis_repr]; rw [← coe_fourierBasis]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    ⊢ HasSum (fun i => HSMul.hSMul (↑(fourierBasis.repr f) i) ((fun i => fourierBa …
  -/
  exact HilbertBasis.hasSum_repr fourierBasis f
  /-
    🎉 no goals
  -/


/-- **Parseval's identity**: for an `L²` function `f` on `AddCircle T`, the sum of the squared
norms of the Fourier coefficients equals the `L²` norm of `f`. -/
theorem tsum_sq_fourierCoeff (f : Lp ℂ 2 <| @haarAddCircle T hT) :
    ∑' i : ℤ, ‖fourierCoeff f i‖ ^ 2 = ∫ t : AddCircle T, ‖f t‖ ^ 2 ∂haarAddCircle := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    ⊢ Eq (tsum fun i => HPow.hPow (Norm.norm (fourierCoeff (↑↑f) i)) 2) (MeasureTh …
  -/
  simp_rw [← fourierBasis_repr]
  have H₁ : ‖fourierBasis.repr f‖ ^ 2 = ∑' i, ‖fourierBasis.repr f i‖ ^ 2 := by
    apply_mod_cast lp.norm_rpow_eq_tsum ?_ (fourierBasis.repr f)
    norm_num
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    H₁ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (tsum fun i => HPow.hP …
    ⊢ Eq (tsum fun i => HPow.hPow (Norm.norm (↑(fourierBasis.repr f) i)) 2) (Measu …
  -/
  have H₂ : ‖fourierBasis.repr f‖ ^ 2 = ‖f‖ ^ 2 := by simp
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    H₁ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (tsum fun i => HPow.hP …
    H₂ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (HPow.hPow (Norm.norm  …
    ⊢ Eq (tsum fun i => HPow.hPow (Norm.norm (↑(fourierBasis.repr f) i)) 2) (Measu …
  -/
  have H₃ := congr_arg RCLike.re (@L2.inner_def (AddCircle T) ℂ ℂ _ _ _ _ _ f f)
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
    H₁ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (tsum fun i => HPow.hP …
    H₂ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (HPow.hPow (Norm.norm  …
    H₃ : Eq (RCLike.re (Inner.inner f f)) (RCLike.re (MeasureTheory.integral AddCi …
    ⊢ Eq (tsum fun i => HPow.hPow (Norm.norm (↑(fourierBasis.repr f) i)) 2) (Measu …
  -/
  rw [← integral_re] at H₃
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      H₁ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (tsum fun i => HPow.hP …
      H₂ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (HPow.hPow (Norm.norm  …
      H₃ : Eq (RCLike.re (Inner.inner f f)) (MeasureTheory.integral AddCircle.haarAd …
      ⊢ Eq (tsum fun i => HPow.hPow (Norm.norm (↑(fourierBasis.repr f) i)) 2) (Measu …
    -/
  · simp only [← norm_sq_eq_inner] at H₃
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      H₁ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (tsum fun i => HPow.hP …
      H₂ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (HPow.hPow (Norm.norm  …
      H₃ : Eq (HPow.hPow (Norm.norm f) 2) (MeasureTheory.integral AddCircle.haarAddC …
      ⊢ Eq (tsum fun i => HPow.hPow (Norm.norm (↑(fourierBasis.repr f) i)) 2) (Measu …
    -/
    rw [← H₁, H₂, H₃]
    /-
      🎉 no goals
    -/
    /-
      T : Real
      hT : Fact (LT.lt 0 T)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Complex 2 AddCircle.haar …
      H₁ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (tsum fun i => HPow.hP …
      H₂ : Eq (HPow.hPow (Norm.norm (fourierBasis.repr f)) 2) (HPow.hPow (Norm.norm  …
      H₃ : Eq (RCLike.re (Inner.inner f f)) (RCLike.re (MeasureTheory.integral AddCi …
      ⊢ MeasureTheory.Integrable (fun a => Inner.inner (↑↑f a) (↑↑f a)) AddCircle.ha …
    -/
  · exact L2.integrable_inner f f
    /-
      🎉 no goals
    -/


theorem fourierCoeff_toLp (n : ℤ) :
    fourierCoeff (toLp (E := ℂ) 2 haarAddCircle ℂ f) n = fourierCoeff f n :=
                                                                              /-
                                                                                T : Real
                                                                                hT : Fact (LT.lt 0 T)
                                                                                f : ContinuousMap (AddCircle T) Complex
                                                                                n : Int
                                                                                ⊢ ∀ (x : AddCircle T), Eq ((fourier (Neg.neg n)) x) ((fourier (Neg.neg n)) x)
                                                                              -/
  integral_congr_ae (Filter.EventuallyEq.mul (Filter.Eventually.of_forall (by tauto))
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    (ContinuousMap.coeFn_toAEEqFun haarAddCircle f))


/-- If the sequence of Fourier coefficients of `f` is summable, then the Fourier series converges
uniformly to `f`. -/
theorem hasSum_fourier_series_of_summable (h : Summable (fourierCoeff f)) :
    HasSum (fun i => fourierCoeff f i • fourier i) f := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : ContinuousMap (AddCircle T) Complex
    h : Summable (fourierCoeff ⇑f)
    ⊢ HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) (fourier i)) f
  -/
  have sum_L2 := hasSum_fourier_series_L2 (toLp (E := ℂ) 2 haarAddCircle ℂ f)
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : ContinuousMap (AddCircle T) Complex
    h : Summable (fourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (fourierCoeff (↑↑((ContinuousMap.toLp 2  …
    ⊢ HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) (fourier i)) f
  -/
  simp_rw [fourierCoeff_toLp] at sum_L2
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : ContinuousMap (AddCircle T) Complex
    h : Summable (fourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) (fourierLp 2 i)) ( …
    ⊢ HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) (fourier i)) f
  -/
  refine ContinuousMap.hasSum_of_hasSum_Lp (.of_norm ?_) sum_L2
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : ContinuousMap (AddCircle T) Complex
    h : Summable (fourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) (fourierLp 2 i)) ( …
    ⊢ Summable fun a => Norm.norm (HSMul.hSMul (fourierCoeff (⇑f) a) (fourier a))
  -/
  simp_rw [norm_smul, fourier_norm, mul_one]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : ContinuousMap (AddCircle T) Complex
    h : Summable (fourierCoeff ⇑f)
    sum_L2 : HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) (fourierLp 2 i)) ( …
    ⊢ Summable fun a => Norm.norm (fourierCoeff (⇑f) a)
  -/
  exact h.norm
  /-
    🎉 no goals
  -/


/-- If the sequence of Fourier coefficients of `f` is summable, then the Fourier series of `f`
converges everywhere pointwise to `f`. -/
theorem has_pointwise_sum_fourier_series_of_summable (h : Summable (fourierCoeff f))
    (x : AddCircle T) : HasSum (fun i => fourierCoeff f i • fourier i x) (f x) := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    f : ContinuousMap (AddCircle T) Complex
    h : Summable (fourierCoeff ⇑f)
    x : AddCircle T
    ⊢ HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑f) i) ((fourier i) x)) (f x)
  -/
  convert (ContinuousMap.evalCLM ℂ x).hasSum (hasSum_fourier_series_of_summable h)
  /-
    🎉 no goals
  -/


theorem hasDerivAt_fourier (n : ℤ) (x : ℝ) :
    HasDerivAt (fun y : ℝ => fourier n (y : AddCircle T))
      (2 * π * I * n / T * fourier n (x : AddCircle T)) x := by
  /-
    T : Real
    n : Int
    x : Real
    ⊢ HasDerivAt (fun y => (fourier n) ↑y) (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul. …
  -/
  simp_rw [fourier_coe_apply]
  /-
    T : Real
    n : Int
    x : Real
    ⊢ HasDerivAt (fun y => Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul …
  -/
  refine (?_ : HasDerivAt (fun y => exp (2 * π * I * n * y / T)) _ _).comp_ofReal
  /-
    T : Real
    n : Int
    x : Real
    ⊢ HasDerivAt (fun y => Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul …
  -/
  rw [(fun α β => by ring : ∀ α β : ℂ, α * exp β = exp β * α)]
  /-
    T : Real
    n : Int
    x : Real
    ⊢ HasDerivAt (fun y => Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul …
  -/
  refine (hasDerivAt_exp _).comp (x : ℂ) ?_
  /-
    T : Real
    n : Int
    x : Real
    ⊢ HasDerivAt (fun y => HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 …
  -/
  convert hasDerivAt_mul_const (2 * ↑π * I * ↑n / T) using 1
  /-
    case h.e'_8
    T : Real
    n : Int
    x : Real
    ⊢ Eq (fun y => HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.p …
  -/
  ext1 y; ring
          /-
            🎉 no goals
          -/


theorem hasDerivAt_fourier_neg (n : ℤ) (x : ℝ) :
    HasDerivAt (fun y : ℝ => fourier (-n) (y : AddCircle T))
      (-2 * π * I * n / T * fourier (-n) (x : AddCircle T)) x := by
  /-
    T : Real
    n : Int
    x : Real
    ⊢ HasDerivAt (fun y => (fourier (Neg.neg n)) ↑y) (HMul.hMul (HDiv.hDiv (HMul.h …
  -/
  simpa using hasDerivAt_fourier T (-n) x
  /-
    🎉 no goals
  -/


theorem has_antideriv_at_fourier_neg (hT : Fact (0 < T)) {n : ℤ} (hn : n ≠ 0) (x : ℝ) :
    HasDerivAt (fun y : ℝ => (T : ℂ) / (-2 * π * I * n) * fourier (-n) (y : AddCircle T))
      (fourier (-n) (x : AddCircle T)) x := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    n : Int
    hn : Ne n 0
    x : Real
    ⊢ HasDerivAt (fun y => HMul.hMul (HDiv.hDiv (↑T) (HMul.hMul (HMul.hMul (HMul.h …
  -/
  convert (hasDerivAt_fourier_neg T n x).div_const (-2 * π * I * n / T) using 1
    /-
      case h.e'_8
      T : Real
      hT : Fact (LT.lt 0 T)
      n : Int
      hn : Ne n 0
      x : Real
      ⊢ Eq (fun y => HMul.hMul (HDiv.hDiv (↑T) (HMul.hMul (HMul.hMul (HMul.hMul (-2) …
    -/
  · ext1 y; rw [div_div_eq_mul_div]; ring
                                     /-
                                       🎉 no goals
                                     -/
    /-
      case h.e'_9
      T : Real
      hT : Fact (LT.lt 0 T)
      n : Int
      hn : Ne n 0
      x : Real
      ⊢ Eq ((fourier (Neg.neg n)) ↑x) (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HMul.hMul (H …
    -/
  · simp [mul_div_cancel_left₀, hn, (Fact.out : 0 < T).ne', Real.pi_pos.ne']
    /-
      🎉 no goals
    -/


/-- Express Fourier coefficients of `f` on an interval in terms of those of its derivative. -/
theorem fourierCoeffOn_of_hasDeriv_right {a b : ℝ} (hab : a < b) {f f' : ℝ → ℂ}
    {n : ℤ} (hn : n ≠ 0)
    (hf : ContinuousOn f [[a, b]])
    (hff' : ∀ x, x ∈ Ioo (min a b) (max a b) → HasDerivWithinAt f (f' x) (Ioi x) x)
    (hf' : IntervalIntegrable f' volume a b) :
    fourierCoeffOn hab f n = 1 / (-2 * π * I * n) *
      (fourier (-n) (a : AddCircle (b - a)) * (f b - f a) - (b - a) * fourierCoeffOn hab f' n) := by
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    ⊢ Eq (fourierCoeffOn hab f n) (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (H …
  -/
  rw [← ofReal_sub]
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    ⊢ Eq (fourierCoeffOn hab f n) (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (H …
  -/
  have hT : Fact (0 < b - a) := ⟨by linarith⟩
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (fourierCoeffOn hab f n) (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (H …
  -/
  simp_rw [fourierCoeffOn_eq_integral, smul_eq_mul, real_smul, ofReal_div, ofReal_one]
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (intervalIntegral (fun x => HMu …
  -/
  conv => pattern (occs := 1 2 3) fourier _ _ * _ <;> (rw [mul_comm])
  rw [integral_mul_deriv_eq_deriv_mul_of_hasDeriv_right hf
    (fun x _ ↦ has_antideriv_at_fourier_neg hT hn x |>.continuousAt |>.continuousWithinAt) hff'
    (fun x _ ↦ has_antideriv_at_fourier_neg hT hn x |>.hasDerivWithinAt) hf'
    (((map_continuous (fourier (-n))).comp (AddCircle.continuous_mk' _)).intervalIntegrable _ _)]
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HSub.hSub (HSub.hSub (HMul.hMu …
  -/
  have : ∀ u v w : ℂ, u * ((b - a : ℝ) / v * w) = (b - a : ℝ) / v * (u * w) := by intros; ring
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HSub.hSub (HSub.hSub (HMul.hMu …
  -/
  conv in intervalIntegral _ _ _ _ => congr; ext; rw [this]
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HSub.hSub (HSub.hSub (HMul.hMu …
  -/
  rw [(by ring : ((b - a : ℝ) : ℂ) / (-2 * π * I * n) = ((b - a : ℝ) : ℂ) * (1 / (-2 * π * I * n)))]
  have s2 : (b : AddCircle (b - a)) = (a : AddCircle (b - a)) := by
    simpa using coe_add_period (b - a) a
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
    s2 : Eq ↑b ↑a
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HSub.hSub (HSub.hSub (HMul.hMu …
  -/
  rw [s2, integral_const_mul, ← sub_mul, mul_sub, mul_sub]
  /-
    a b : Real
    hab : LT.lt a b
    f f' : Real → Complex
    n : Int
    hn : Ne n 0
    hf : ContinuousOn f (Set.uIcc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
    hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    hT : Fact (LT.lt 0 (HSub.hSub b a))
    this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
    s2 : Eq ↑b ↑a
    ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HMul.hMul (HSub.hSu …
  -/
  congr 1
    /-
      case e_a
      a b : Real
      hab : LT.lt a b
      f f' : Real → Complex
      n : Int
      hn : Ne n 0
      hf : ContinuousOn f (Set.uIcc a b)
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
      hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hT : Fact (LT.lt 0 (HSub.hSub b a))
      this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
      s2 : Eq ↑b ↑a
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HMul.hMul (HSub.hSub (f b) (f  …
    -/
  · conv_lhs => rw [mul_comm, mul_div, mul_one]
    /-
      case e_a
      a b : Real
      hab : LT.lt a b
      f f' : Real → Complex
      n : Int
      hn : Ne n 0
      hf : ContinuousOn f (Set.uIcc a b)
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
      hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hT : Fact (LT.lt 0 (HSub.hSub b a))
      this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
      s2 : Eq ↑b ↑a
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HSub.hSub (f b) (f a)) (HMul.hMul (HMul.hMul (↑(HS …
    -/
    rw [div_eq_iff (ofReal_ne_zero.mpr hT.out.ne')]
    /-
      case e_a
      a b : Real
      hab : LT.lt a b
      f f' : Real → Complex
      n : Int
      hn : Ne n 0
      hf : ContinuousOn f (Set.uIcc a b)
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
      hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hT : Fact (LT.lt 0 (HSub.hSub b a))
      this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
      s2 : Eq ↑b ↑a
      ⊢ Eq (HMul.hMul (HSub.hSub (f b) (f a)) (HMul.hMul (HMul.hMul (↑(HSub.hSub b a …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case e_a
      a b : Real
      hab : LT.lt a b
      f f' : Real → Complex
      n : Int
      hn : Ne n 0
      hf : ContinuousOn f (Set.uIcc a b)
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo (Min.min a b) (Max.max a b)) x →  …
      hf' : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hT : Fact (LT.lt 0 (HSub.hSub b a))
      this : ∀ (u v w : Complex), Eq (HMul.hMul u (HMul.hMul (HDiv.hDiv (↑(HSub.hSub …
      s2 : Eq ↑b ↑a
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 ↑(HSub.hSub b a)) (HMul.hMul (HMul.hMul (↑(HSub.h …
    -/
  · ring
    /-
      🎉 no goals
    -/


/-- Express Fourier coefficients of `f` on an interval in terms of those of its derivative. -/
theorem fourierCoeffOn_of_hasDerivAt_Ioo {a b : ℝ} (hab : a < b) {f f' : ℝ → ℂ}
    {n : ℤ} (hn : n ≠ 0)
    (hf : ContinuousOn f [[a, b]])
    (hff' : ∀ x, x ∈ Ioo (min a b) (max a b) → HasDerivAt f (f' x) x)
    (hf' : IntervalIntegrable f' volume a b) :
    fourierCoeffOn hab f n = 1 / (-2 * π * I * n) *
      (fourier (-n) (a : AddCircle (b - a)) * (f b - f a) - (b - a) * fourierCoeffOn hab f' n) :=
  fourierCoeffOn_of_hasDeriv_right hab hn hf (fun x hx ↦ hff' x hx |>.hasDerivWithinAt) hf'


/-- Express Fourier coefficients of `f` on an interval in terms of those of its derivative. -/
theorem fourierCoeffOn_of_hasDerivAt {a b : ℝ} (hab : a < b) {f f' : ℝ → ℂ} {n : ℤ} (hn : n ≠ 0)
    (hf : ∀ x, x ∈ [[a, b]] → HasDerivAt f (f' x) x) (hf' : IntervalIntegrable f' volume a b) :
    fourierCoeffOn hab f n = 1 / (-2 * π * I * n) *
      (fourier (-n) (a : AddCircle (b - a)) * (f b - f a) - (b - a) * fourierCoeffOn hab f' n) :=
  fourierCoeffOn_of_hasDerivAt_Ioo hab hn
    (fun x hx ↦ hf x hx |>.continuousAt.continuousWithinAt)
    (fun x hx ↦ hf x <| mem_Icc_of_Ioo hx)
    hf'


