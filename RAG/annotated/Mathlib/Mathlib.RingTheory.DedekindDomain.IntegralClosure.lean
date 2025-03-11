/-- If `L` is an algebraic extension of `K = Frac(A)` and `L` has no zero smul divisors by `A`,
then `L` is the localization of the integral closure `C` of `A` in `L` at `A⁰`. -/
theorem IsIntegralClosure.isLocalization [IsDomain A] [Algebra.IsAlgebraic K L] :
    IsLocalization (Algebra.algebraMapSubmonoid C A⁰) L := by
  haveI : IsDomain C :=
    (IsIntegralClosure.equiv A C L (integralClosure A L)).toMulEquiv.isDomain (integralClosure A L)
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : Algebra A K
    inst✝¹¹ : IsFractionRing A K
    L : Type u_3
    inst✝¹⁰ : Field L
    C : Type u_4
    inst✝⁹ : CommRing C
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : Algebra C L
    inst✝⁴ : IsIntegralClosure C A L
    inst✝³ : Algebra A C
    inst✝² : IsScalarTower A C L
    inst✝¹ : IsDomain A
    inst✝ : Algebra.IsAlgebraic K L
    this : IsDomain C
    ⊢ IsLocalization (Algebra.algebraMapSubmonoid C (nonZeroDivisors A)) L
  -/
  haveI : NoZeroSMulDivisors A L := NoZeroSMulDivisors.trans A K L
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : Algebra A K
    inst✝¹¹ : IsFractionRing A K
    L : Type u_3
    inst✝¹⁰ : Field L
    C : Type u_4
    inst✝⁹ : CommRing C
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : Algebra C L
    inst✝⁴ : IsIntegralClosure C A L
    inst✝³ : Algebra A C
    inst✝² : IsScalarTower A C L
    inst✝¹ : IsDomain A
    inst✝ : Algebra.IsAlgebraic K L
    this✝ : IsDomain C
    this : NoZeroSMulDivisors A L
    ⊢ IsLocalization (Algebra.algebraMapSubmonoid C (nonZeroDivisors A)) L
  -/
  haveI : NoZeroSMulDivisors A C := IsIntegralClosure.noZeroSMulDivisors A L
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : Algebra A K
    inst✝¹¹ : IsFractionRing A K
    L : Type u_3
    inst✝¹⁰ : Field L
    C : Type u_4
    inst✝⁹ : CommRing C
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : Algebra C L
    inst✝⁴ : IsIntegralClosure C A L
    inst✝³ : Algebra A C
    inst✝² : IsScalarTower A C L
    inst✝¹ : IsDomain A
    inst✝ : Algebra.IsAlgebraic K L
    this✝¹ : IsDomain C
    this✝ : NoZeroSMulDivisors A L
    this : NoZeroSMulDivisors A C
    ⊢ IsLocalization (Algebra.algebraMapSubmonoid C (nonZeroDivisors A)) L
  -/
  refine ⟨?_, fun z => ?_, fun {x y} h => ⟨1, ?_⟩⟩
    /-
      case refine_1
      A : Type u_1
      K : Type u_2
      inst✝¹⁴ : CommRing A
      inst✝¹³ : Field K
      inst✝¹² : Algebra A K
      inst✝¹¹ : IsFractionRing A K
      L : Type u_3
      inst✝¹⁰ : Field L
      C : Type u_4
      inst✝⁹ : CommRing C
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra A L
      inst✝⁶ : IsScalarTower A K L
      inst✝⁵ : Algebra C L
      inst✝⁴ : IsIntegralClosure C A L
      inst✝³ : Algebra A C
      inst✝² : IsScalarTower A C L
      inst✝¹ : IsDomain A
      inst✝ : Algebra.IsAlgebraic K L
      this✝¹ : IsDomain C
      this✝ : NoZeroSMulDivisors A L
      this : NoZeroSMulDivisors A C
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid C (nonZe …
    -/
  · rintro ⟨_, x, hx, rfl⟩
    rw [isUnit_iff_ne_zero, map_ne_zero_iff _ (IsIntegralClosure.algebraMap_injective C A L),
      Subtype.coe_mk, map_ne_zero_iff _ (NoZeroSMulDivisors.algebraMap_injective A C)]
    /-
      case refine_1.mk.intro.intro
      A : Type u_1
      K : Type u_2
      inst✝¹⁴ : CommRing A
      inst✝¹³ : Field K
      inst✝¹² : Algebra A K
      inst✝¹¹ : IsFractionRing A K
      L : Type u_3
      inst✝¹⁰ : Field L
      C : Type u_4
      inst✝⁹ : CommRing C
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra A L
      inst✝⁶ : IsScalarTower A K L
      inst✝⁵ : Algebra C L
      inst✝⁴ : IsIntegralClosure C A L
      inst✝³ : Algebra A C
      inst✝² : IsScalarTower A C L
      inst✝¹ : IsDomain A
      inst✝ : Algebra.IsAlgebraic K L
      this✝¹ : IsDomain C
      this✝ : NoZeroSMulDivisors A L
      this : NoZeroSMulDivisors A C
      x : A
      hx : Membership.mem (↑(nonZeroDivisors A)) x
      ⊢ Ne x 0
    -/
    exact mem_nonZeroDivisors_iff_ne_zero.mp hx
    /-
      🎉 no goals
    -/
  · obtain ⟨m, hm⟩ :=
      IsIntegral.exists_multiple_integral_of_isLocalization A⁰ z
        (Algebra.IsIntegral.isIntegral (R := K) z)
    /-
      case refine_2.intro
      A : Type u_1
      K : Type u_2
      inst✝¹⁴ : CommRing A
      inst✝¹³ : Field K
      inst✝¹² : Algebra A K
      inst✝¹¹ : IsFractionRing A K
      L : Type u_3
      inst✝¹⁰ : Field L
      C : Type u_4
      inst✝⁹ : CommRing C
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra A L
      inst✝⁶ : IsScalarTower A K L
      inst✝⁵ : Algebra C L
      inst✝⁴ : IsIntegralClosure C A L
      inst✝³ : Algebra A C
      inst✝² : IsScalarTower A C L
      inst✝¹ : IsDomain A
      inst✝ : Algebra.IsAlgebraic K L
      this✝¹ : IsDomain C
      this✝ : NoZeroSMulDivisors A L
      this : NoZeroSMulDivisors A C
      z : L
      m : Subtype fun x => Membership.mem (nonZeroDivisors A) x
      hm : IsIntegral A (HSMul.hSMul m z)
      ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap C L) ↑x.2)) ((algebraMap C L) x …
    -/
    obtain ⟨x, hx⟩ : ∃ x, algebraMap C L x = m • z := IsIntegralClosure.isIntegral_iff.mp hm
    /-
      case refine_2.intro.intro
      A : Type u_1
      K : Type u_2
      inst✝¹⁴ : CommRing A
      inst✝¹³ : Field K
      inst✝¹² : Algebra A K
      inst✝¹¹ : IsFractionRing A K
      L : Type u_3
      inst✝¹⁰ : Field L
      C : Type u_4
      inst✝⁹ : CommRing C
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra A L
      inst✝⁶ : IsScalarTower A K L
      inst✝⁵ : Algebra C L
      inst✝⁴ : IsIntegralClosure C A L
      inst✝³ : Algebra A C
      inst✝² : IsScalarTower A C L
      inst✝¹ : IsDomain A
      inst✝ : Algebra.IsAlgebraic K L
      this✝¹ : IsDomain C
      this✝ : NoZeroSMulDivisors A L
      this : NoZeroSMulDivisors A C
      z : L
      m : Subtype fun x => Membership.mem (nonZeroDivisors A) x
      hm : IsIntegral A (HSMul.hSMul m z)
      x : C
      hx : Eq ((algebraMap C L) x) (HSMul.hSMul m z)
      ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap C L) ↑x.2)) ((algebraMap C L) x …
    -/
    refine ⟨⟨x, algebraMap A C m, m, SetLike.coe_mem m, rfl⟩, ?_⟩
    rw [Subtype.coe_mk, ← IsScalarTower.algebraMap_apply, hx, mul_comm, Submonoid.smul_def,
      smul_def]
    /-
      case refine_3
      A : Type u_1
      K : Type u_2
      inst✝¹⁴ : CommRing A
      inst✝¹³ : Field K
      inst✝¹² : Algebra A K
      inst✝¹¹ : IsFractionRing A K
      L : Type u_3
      inst✝¹⁰ : Field L
      C : Type u_4
      inst✝⁹ : CommRing C
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra A L
      inst✝⁶ : IsScalarTower A K L
      inst✝⁵ : Algebra C L
      inst✝⁴ : IsIntegralClosure C A L
      inst✝³ : Algebra A C
      inst✝² : IsScalarTower A C L
      inst✝¹ : IsDomain A
      inst✝ : Algebra.IsAlgebraic K L
      this✝¹ : IsDomain C
      this✝ : NoZeroSMulDivisors A L
      this : NoZeroSMulDivisors A C
      x y : C
      h : Eq ((algebraMap C L) x) ((algebraMap C L) y)
      ⊢ Eq (HMul.hMul (↑1) x) (HMul.hMul (↑1) y)
    -/
  · simp only [IsIntegralClosure.algebraMap_injective C A L h]
    /-
      🎉 no goals
    -/


theorem IsIntegralClosure.isLocalization_of_isSeparable [IsDomain A] [Algebra.IsSeparable K L] :
    IsLocalization (Algebra.algebraMapSubmonoid C A⁰) L :=
  IsIntegralClosure.isLocalization A K L C


theorem IsIntegralClosure.range_le_span_dualBasis [Algebra.IsSeparable K L] {ι : Type*} [Fintype ι]
    [DecidableEq ι] (b : Basis ι K L) (hb_int : ∀ i, IsIntegral A (b i)) [IsIntegrallyClosed A] :
    LinearMap.range ((Algebra.linearMap C L).restrictScalars A) ≤
    Submodule.span A (Set.range <| (traceForm K L).dualBasis (traceForm_nondegenerate K L) b) := by
  rw [← LinearMap.BilinForm.dualSubmodule_span_of_basis,
    ← LinearMap.BilinForm.le_flip_dualSubmodule, Submodule.span_le]
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : Algebra.IsSeparable K L
    ι : Type u_5
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    b : Basis ι K L
    hb_int : ∀ (i : ι), IsIntegral A (b i)
    inst✝ : IsIntegrallyClosed A
    ⊢ HasSubset.Subset (Set.range ⇑b) ↑((Algebra.traceForm K L).flip.dualSubmodule …
  -/
  rintro _ ⟨i, rfl⟩ _ ⟨y, rfl⟩
  simp only [LinearMap.coe_restrictScalars, linearMap_apply, LinearMap.BilinForm.flip_apply,
    traceForm_apply]
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : Algebra.IsSeparable K L
    ι : Type u_5
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    b : Basis ι K L
    hb_int : ∀ (i : ι), IsIntegral A (b i)
    inst✝ : IsIntegrallyClosed A
    i : ι
    y : C
    ⊢ Membership.mem 1 ((Algebra.trace K L) (HMul.hMul ((algebraMap C L) y) (b i)))
  -/
  refine Submodule.mem_one.mpr <| IsIntegrallyClosed.isIntegral_iff.mp ?_
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : Algebra.IsSeparable K L
    ι : Type u_5
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    b : Basis ι K L
    hb_int : ∀ (i : ι), IsIntegral A (b i)
    inst✝ : IsIntegrallyClosed A
    i : ι
    y : C
    ⊢ IsIntegral A ((Algebra.trace K L) (HMul.hMul ((algebraMap C L) y) (b i)))
  -/
  exact isIntegral_trace ((IsIntegralClosure.isIntegral A L y).algebraMap.mul (hb_int i))
  /-
    🎉 no goals
  -/


theorem integralClosure_le_span_dualBasis [Algebra.IsSeparable K L] {ι : Type*} [Fintype ι]
    [DecidableEq ι] (b : Basis ι K L) (hb_int : ∀ i, IsIntegral A (b i)) [IsIntegrallyClosed A] :
    Subalgebra.toSubmodule (integralClosure A L) ≤
    Submodule.span A (Set.range <| (traceForm K L).dualBasis (traceForm_nondegenerate K L) b) := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹² : CommRing A
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    L : Type u_3
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : Algebra.IsSeparable K L
    ι : Type u_5
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    b : Basis ι K L
    hb_int : ∀ (i : ι), IsIntegral A (b i)
    inst✝ : IsIntegrallyClosed A
    ⊢ LE.le (Subalgebra.toSubmodule (integralClosure A L)) (Submodule.span A (Set. …
  -/
  refine le_trans ?_ (IsIntegralClosure.range_le_span_dualBasis (integralClosure A L) b hb_int)
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹² : CommRing A
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    L : Type u_3
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : Algebra.IsSeparable K L
    ι : Type u_5
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    b : Basis ι K L
    hb_int : ∀ (i : ι), IsIntegral A (b i)
    inst✝ : IsIntegrallyClosed A
    ⊢ LE.le (Subalgebra.toSubmodule (integralClosure A L)) (LinearMap.range (↑A (A …
  -/
  intro x hx
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹² : CommRing A
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    L : Type u_3
    inst✝⁸ : Field L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : Algebra.IsSeparable K L
    ι : Type u_5
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    b : Basis ι K L
    hb_int : ∀ (i : ι), IsIntegral A (b i)
    inst✝ : IsIntegrallyClosed A
    x : L
    hx : Membership.mem (Subalgebra.toSubmodule (integralClosure A L)) x
    ⊢ Membership.mem (LinearMap.range (↑A (Algebra.linearMap (Subtype fun x => Mem …
  -/
  exact ⟨⟨x, hx⟩, rfl⟩
  /-
    🎉 no goals
  -/


/-- Send a set of `x`s in a finite extension `L` of the fraction field of `R`
to `(y : R) • x ∈ integralClosure R L`. -/
theorem exists_integral_multiples (s : Finset L) :
    ∃ y ≠ (0 : A), ∀ x ∈ s, IsIntegral A (y • x) :=
  have := IsLocalization.isAlgebraic K (nonZeroDivisors A)
  have := Algebra.IsAlgebraic.trans' A (algebraMap K L).injective
  Algebra.IsAlgebraic.exists_integral_multiples (IsScalarTower.algebraMap_eq A K L ▸
    (algebraMap K L).injective.comp (IsFractionRing.injective _ _)) _


/-- If `L` is a finite extension of `K = Frac(A)`,
then `L` has a basis over `A` consisting of integral elements. -/
theorem FiniteDimensional.exists_is_basis_integral :
    ∃ (s : Finset L) (b : Basis s K L), ∀ x, IsIntegral A (b x) := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Field K
    inst✝⁷ : Algebra A K
    inst✝⁶ : IsFractionRing A K
    L : Type u_3
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsDomain A
    ⊢ Exists fun s => Exists fun b => ∀ (x : Subtype fun x => Membership.mem s x), …
  -/
  letI := Classical.decEq L
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Field K
    inst✝⁷ : Algebra A K
    inst✝⁶ : IsFractionRing A K
    L : Type u_3
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsDomain A
    this : DecidableEq L := Classical.decEq L
    ⊢ Exists fun s => Exists fun b => ∀ (x : Subtype fun x => Membership.mem s x), …
  -/
  letI : IsNoetherian K L := IsNoetherian.iff_fg.2 inferInstance
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Field K
    inst✝⁷ : Algebra A K
    inst✝⁶ : IsFractionRing A K
    L : Type u_3
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsDomain A
    this✝ : DecidableEq L := Classical.decEq L
    this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
    ⊢ Exists fun s => Exists fun b => ∀ (x : Subtype fun x => Membership.mem s x), …
  -/
  let s' := IsNoetherian.finsetBasisIndex K L
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Field K
    inst✝⁷ : Algebra A K
    inst✝⁶ : IsFractionRing A K
    L : Type u_3
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsDomain A
    this✝ : DecidableEq L := Classical.decEq L
    this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
    s' : Finset L := IsNoetherian.finsetBasisIndex K L
    ⊢ Exists fun s => Exists fun b => ∀ (x : Subtype fun x => Membership.mem s x), …
  -/
  let bs' := IsNoetherian.finsetBasis K L
  /-
    A : Type u_1
    K : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : Field K
    inst✝⁷ : Algebra A K
    inst✝⁶ : IsFractionRing A K
    L : Type u_3
    inst✝⁵ : Field L
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra A L
    inst✝² : IsScalarTower A K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsDomain A
    this✝ : DecidableEq L := Classical.decEq L
    this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
    s' : Finset L := IsNoetherian.finsetBasisIndex K L
    bs' : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K  …
    ⊢ Exists fun s => Exists fun b => ∀ (x : Subtype fun x => Membership.mem s x), …
  -/
  obtain ⟨y, hy, his'⟩ := exists_integral_multiples A K (Finset.univ.image bs')
  have hy' : algebraMap A L y ≠ 0 := by
    refine mt ((injective_iff_map_eq_zero (algebraMap A L)).mp ?_ _) hy
    rw [IsScalarTower.algebraMap_eq A K L]
    exact (algebraMap K L).injective.comp (IsFractionRing.injective A K)
  refine ⟨s', bs'.map {Algebra.lmul _ _ (algebraMap A L y) with
    toFun := fun x => algebraMap A L y * x
    invFun := fun x => (algebraMap A L y)⁻¹ * x
    left_inv := ?_
    right_inv := ?_}, ?_⟩
    /-
      case intro.intro.refine_1
      A : Type u_1
      K : Type u_2
      inst✝⁹ : CommRing A
      inst✝⁸ : Field K
      inst✝⁷ : Algebra A K
      inst✝⁶ : IsFractionRing A K
      L : Type u_3
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A L
      inst✝² : IsScalarTower A K L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsDomain A
      this✝ : DecidableEq L := Classical.decEq L
      this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
      s' : Finset L := IsNoetherian.finsetBasisIndex K L
      bs' : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K  …
      y : A
      hy : Ne y 0
      his' : ∀ (x : L), Membership.mem (Finset.image (⇑bs') Finset.univ) x → IsInteg …
      hy' : Ne ((algebraMap A L) y) 0
      ⊢ Function.LeftInverse (fun x => HMul.hMul (Inv.inv ((algebraMap A L) y)) x) { …
    -/
  · intro x; simp only [inv_mul_cancel_left₀ hy']
             /-
               🎉 no goals
             -/
    /-
      case intro.intro.refine_2
      A : Type u_1
      K : Type u_2
      inst✝⁹ : CommRing A
      inst✝⁸ : Field K
      inst✝⁷ : Algebra A K
      inst✝⁶ : IsFractionRing A K
      L : Type u_3
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A L
      inst✝² : IsScalarTower A K L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsDomain A
      this✝ : DecidableEq L := Classical.decEq L
      this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
      s' : Finset L := IsNoetherian.finsetBasisIndex K L
      bs' : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K  …
      y : A
      hy : Ne y 0
      his' : ∀ (x : L), Membership.mem (Finset.image (⇑bs') Finset.univ) x → IsInteg …
      hy' : Ne ((algebraMap A L) y) 0
      ⊢ Function.RightInverse (fun x => HMul.hMul (Inv.inv ((algebraMap A L) y)) x)  …
    -/
  · intro x; simp only [mul_inv_cancel_left₀ hy']
             /-
               🎉 no goals
             -/
    /-
      case intro.intro.refine_3
      A : Type u_1
      K : Type u_2
      inst✝⁹ : CommRing A
      inst✝⁸ : Field K
      inst✝⁷ : Algebra A K
      inst✝⁶ : IsFractionRing A K
      L : Type u_3
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A L
      inst✝² : IsScalarTower A K L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsDomain A
      this✝ : DecidableEq L := Classical.decEq L
      this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
      s' : Finset L := IsNoetherian.finsetBasisIndex K L
      bs' : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K  …
      y : A
      hy : Ne y 0
      his' : ∀ (x : L), Membership.mem (Finset.image (⇑bs') Finset.univ) x → IsInteg …
      hy' : Ne ((algebraMap A L) y) 0
      ⊢ ∀ (x : Subtype fun x => Membership.mem s' x),
          IsIntegral A
            ((bs'.map
                (let __src := (Algebra.lmul K L) ((algebraMap A L) y);
                { toFun := fun x => HMul.hMul ((algebraMap A L) y) x, map_add' := ⋯, …
              x)
    -/
  · rintro ⟨x', hx'⟩
    simp only [Algebra.smul_def, Finset.mem_image, exists_prop, Finset.mem_univ,
      true_and] at his'
    /-
      case intro.intro.refine_3.mk
      A : Type u_1
      K : Type u_2
      inst✝⁹ : CommRing A
      inst✝⁸ : Field K
      inst✝⁷ : Algebra A K
      inst✝⁶ : IsFractionRing A K
      L : Type u_3
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A L
      inst✝² : IsScalarTower A K L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsDomain A
      this✝ : DecidableEq L := Classical.decEq L
      this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
      s' : Finset L := IsNoetherian.finsetBasisIndex K L
      bs' : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K  …
      y : A
      hy : Ne y 0
      hy' : Ne ((algebraMap A L) y) 0
      x' : L
      hx' : Membership.mem s' x'
      his' : ∀ (x : L), (Exists fun a => Eq (bs' a) x) → IsIntegral A (HMul.hMul ((a …
      ⊢ IsIntegral A
          ((bs'.map
              (let __src := (Algebra.lmul K L) ((algebraMap A L) y);
              { toFun := fun x => HMul.hMul ((algebraMap A L) y) x, map_add' := ⋯, m …
            ⟨x', hx'⟩)
    -/
    simp only [Basis.map_apply, LinearEquiv.coe_mk]
    /-
      case intro.intro.refine_3.mk
      A : Type u_1
      K : Type u_2
      inst✝⁹ : CommRing A
      inst✝⁸ : Field K
      inst✝⁷ : Algebra A K
      inst✝⁶ : IsFractionRing A K
      L : Type u_3
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A L
      inst✝² : IsScalarTower A K L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsDomain A
      this✝ : DecidableEq L := Classical.decEq L
      this : IsNoetherian K L := IsNoetherian.iff_fg.mpr inferInstance
      s' : Finset L := IsNoetherian.finsetBasisIndex K L
      bs' : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K  …
      y : A
      hy : Ne y 0
      hy' : Ne ((algebraMap A L) y) 0
      x' : L
      hx' : Membership.mem s' x'
      his' : ∀ (x : L), (Exists fun a => Eq (bs' a) x) → IsIntegral A (HMul.hMul ((a …
      ⊢ IsIntegral A (HMul.hMul ((algebraMap A L) y) (bs' ⟨x', hx'⟩))
    -/
    exact his' _ ⟨_, rfl⟩
    /-
      🎉 no goals
    -/


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is
integrally closed and Noetherian, the integral closure `C` of `A` in `L` is
Noetherian over `A`. -/
theorem IsIntegralClosure.isNoetherian [IsIntegrallyClosed A] [IsNoetherianRing A] :
    IsNoetherian A C := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    ⊢ IsNoetherian A C
  -/
  haveI := Classical.decEq L
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this : DecidableEq L
    ⊢ IsNoetherian A C
  -/
  obtain ⟨s, b, hb_int⟩ := FiniteDimensional.exists_is_basis_integral A K L
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this : DecidableEq L
    s : Finset L
    b : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral A (b x)
    ⊢ IsNoetherian A C
  -/
  let b' := (traceForm K L).dualBasis (traceForm_nondegenerate K L) b
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this : DecidableEq L
    s : Finset L
    b : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral A (b x)
    b' : Basis (Subtype fun x => Membership.mem s x) K L := (Algebra.traceForm K L …
    ⊢ IsNoetherian A C
  -/
  letI := isNoetherian_span_of_finite A (Set.finite_range b')
  let f : C →ₗ[A] Submodule.span A (Set.range b') :=
    (Submodule.inclusion (IsIntegralClosure.range_le_span_dualBasis C b hb_int)).comp
      ((Algebra.linearMap C L).restrictScalars A).rangeRestrict
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this✝ : DecidableEq L
    s : Finset L
    b : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral A (b x)
    b' : Basis (Subtype fun x => Membership.mem s x) K L := (Algebra.traceForm K L …
    this : IsNoetherian A (Subtype fun x => Membership.mem (Submodule.span A (Set. …
    f : LinearMap (RingHom.id A) C (Subtype fun x => Membership.mem (Submodule.spa …
    ⊢ IsNoetherian A C
  -/
  refine isNoetherian_of_ker_bot f ?_
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this✝ : DecidableEq L
    s : Finset L
    b : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral A (b x)
    b' : Basis (Subtype fun x => Membership.mem s x) K L := (Algebra.traceForm K L …
    this : IsNoetherian A (Subtype fun x => Membership.mem (Submodule.span A (Set. …
    f : LinearMap (RingHom.id A) C (Subtype fun x => Membership.mem (Submodule.spa …
    ⊢ Eq (LinearMap.ker f) Bot.bot
  -/
  rw [LinearMap.ker_comp, Submodule.ker_inclusion, Submodule.comap_bot, LinearMap.ker_codRestrict]
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this✝ : DecidableEq L
    s : Finset L
    b : Basis (Subtype fun x => Membership.mem s x) K L
    hb_int : ∀ (x : Subtype fun x => Membership.mem s x), IsIntegral A (b x)
    b' : Basis (Subtype fun x => Membership.mem s x) K L := (Algebra.traceForm K L …
    this : IsNoetherian A (Subtype fun x => Membership.mem (Submodule.span A (Set. …
    f : LinearMap (RingHom.id A) C (Subtype fun x => Membership.mem (Submodule.spa …
    ⊢ Eq (LinearMap.ker (↑A (Algebra.linearMap C L))) Bot.bot
  -/
  exact LinearMap.ker_eq_bot_of_injective (IsIntegralClosure.algebraMap_injective C A L)
  /-
    🎉 no goals
  -/


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is
integrally closed and Noetherian, the integral closure `C` of `A` in `L` is
Noetherian. -/
theorem IsIntegralClosure.isNoetherianRing [IsIntegrallyClosed A] [IsNoetherianRing A] :
    IsNoetherianRing C :=
  isNoetherianRing_iff.mpr <| isNoetherian_of_tower A (IsIntegralClosure.isNoetherian A K L C)


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is
integrally closed and Noetherian, the integral closure `C` of `A` in `L` is
finite over `A`. -/
theorem IsIntegralClosure.finite [IsIntegrallyClosed A] [IsNoetherianRing A] :
    Module.Finite A C := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    ⊢ Module.Finite A C
  -/
  haveI := IsIntegralClosure.isNoetherian A K L C
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsNoetherianRing A
    this : IsNoetherian A C
    ⊢ Module.Finite A C
  -/
  exact Module.IsNoetherian.finite A C
  /-
    🎉 no goals
  -/


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is a principal ring
and `L` has no zero smul divisors by `A`, the integral closure `C` of `A` in `L` is
a free `A`-module. -/
theorem IsIntegralClosure.module_free [NoZeroSMulDivisors A L] [IsPrincipalIdealRing A] :
    Module.Free A C :=
  haveI : NoZeroSMulDivisors A C := IsIntegralClosure.noZeroSMulDivisors A L
  haveI : IsNoetherian A C := IsIntegralClosure.isNoetherian A K L _
  inferInstance


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is a principal ring
and `L` has no zero smul divisors by `A`, the `A`-rank of the integral closure `C` of `A` in `L`
is equal to the `K`-rank of `L`. -/
theorem IsIntegralClosure.rank [IsPrincipalIdealRing A] [NoZeroSMulDivisors A L] :
    Module.finrank A C = Module.finrank K L := by
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsPrincipalIdealRing A
    inst✝ : NoZeroSMulDivisors A L
    ⊢ Eq (Module.finrank A C) (Module.finrank K L)
  -/
  haveI : Module.Free A C := IsIntegralClosure.module_free A K L C
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsPrincipalIdealRing A
    inst✝ : NoZeroSMulDivisors A L
    this : Module.Free A C
    ⊢ Eq (Module.finrank A C) (Module.finrank K L)
  -/
  haveI : IsNoetherian A C := IsIntegralClosure.isNoetherian A K L C
  haveI : IsLocalization (Algebra.algebraMapSubmonoid C A⁰) L :=
    IsIntegralClosure.isLocalization A K L C
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsPrincipalIdealRing A
    inst✝ : NoZeroSMulDivisors A L
    this✝¹ : Module.Free A C
    this✝ : IsNoetherian A C
    this : IsLocalization (Algebra.algebraMapSubmonoid C (nonZeroDivisors A)) L
    ⊢ Eq (Module.finrank A C) (Module.finrank K L)
  -/
  let b := Basis.localizationLocalization K A⁰ L (Module.Free.chooseBasis A C)
  /-
    A : Type u_1
    K : Type u_2
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    L : Type u_3
    inst✝¹³ : Field L
    C : Type u_4
    inst✝¹² : CommRing C
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : Algebra C L
    inst✝⁷ : IsIntegralClosure C A L
    inst✝⁶ : Algebra A C
    inst✝⁵ : IsScalarTower A C L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsDomain A
    inst✝² : Algebra.IsSeparable K L
    inst✝¹ : IsPrincipalIdealRing A
    inst✝ : NoZeroSMulDivisors A L
    this✝¹ : Module.Free A C
    this✝ : IsNoetherian A C
    this : IsLocalization (Algebra.algebraMapSubmonoid C (nonZeroDivisors A)) L
    b : Basis (Module.Free.ChooseBasisIndex A C) K L := Basis.localizationLocaliza …
    ⊢ Eq (Module.finrank A C) (Module.finrank K L)
  -/
  rw [Module.finrank_eq_card_chooseBasisIndex, Module.finrank_eq_card_basis b]
  /-
    🎉 no goals
  -/


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is
integrally closed and Noetherian, the integral closure of `A` in `L` is
Noetherian. -/
theorem integralClosure.isNoetherianRing [IsIntegrallyClosed A] [IsNoetherianRing A] :
    IsNoetherianRing (integralClosure A L) :=
  IsIntegralClosure.isNoetherianRing A K L (integralClosure A L)


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is a Dedekind domain,
the integral closure `C` of `A` in `L` is a Dedekind domain.

This cannot be an instance since `A`, `K` or `L` can't be inferred. See also the instance
`integralClosure.isDedekindDomain_fractionRing` where `K := FractionRing A`
and `C := integralClosure A L`. -/
theorem IsIntegralClosure.isDedekindDomain [IsDedekindDomain A] : IsDedekindDomain C :=
  have : IsFractionRing C L := IsIntegralClosure.isFractionRing_of_finite_extension A K L C
  have : Algebra.IsIntegral A C := IsIntegralClosure.isIntegral_algebra A L
  { IsIntegralClosure.isNoetherianRing A K L C,
    Ring.DimensionLEOne.isIntegralClosure A L C,
    (isIntegrallyClosed_iff L).mpr fun {x} hx =>
      ⟨IsIntegralClosure.mk' C x (isIntegral_trans (R := A) _ hx),
        IsIntegralClosure.algebraMap_mk' _ _ _⟩ with : IsDedekindDomain C }


/-- If `L` is a finite separable extension of `K = Frac(A)`, where `A` is a Dedekind domain,
the integral closure of `A` in `L` is a Dedekind domain.

This cannot be an instance since `K` can't be inferred. See also the instance
`integralClosure.isDedekindDomain_fractionRing` where `K := FractionRing A`. -/
theorem integralClosure.isDedekindDomain [IsDedekindDomain A] :
    IsDedekindDomain (integralClosure A L) :=
  IsIntegralClosure.isDedekindDomain A K L (integralClosure A L)


/-- If `L` is a finite separable extension of `Frac(A)`, where `A` is a Dedekind domain,
the integral closure of `A` in `L` is a Dedekind domain.

See also the lemma `integralClosure.isDedekindDomain` where you can choose
the field of fractions yourself. -/
instance integralClosure.isDedekindDomain_fractionRing [IsDedekindDomain A] :
    IsDedekindDomain (integralClosure A L) :=
  integralClosure.isDedekindDomain A (FractionRing A) L


