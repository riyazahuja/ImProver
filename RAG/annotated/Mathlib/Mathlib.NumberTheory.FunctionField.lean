/-- `F` is a function field over the finite field `Fq` if it is a finite
extension of the field of rational functions in one variable over `Fq`.

Note that `F` can be a function field over multiple, non-isomorphic, `Fq`.
-/
abbrev FunctionField [Algebra (RatFunc Fq) F] : Prop :=
  FiniteDimensional (RatFunc Fq) F


/-- `F` is a function field over `Fq` iff it is a finite extension of `Fq(t)`. -/
theorem functionField_iff (Fqt : Type*) [Field Fqt] [Algebra Fq[X] Fqt]
    [IsFractionRing Fq[X] Fqt] [Algebra (RatFunc Fq) F] [Algebra Fqt F] [Algebra Fq[X] F]
    [IsScalarTower Fq[X] Fqt F] [IsScalarTower Fq[X] (RatFunc Fq) F] :
    FunctionField Fq F ↔ FiniteDimensional Fqt F := by
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁹ : Field Fq
    inst✝⁸ : Field F
    Fqt : Type u_3
    inst✝⁷ : Field Fqt
    inst✝⁶ : Algebra (Polynomial Fq) Fqt
    inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
    inst✝⁴ : Algebra (RatFunc Fq) F
    inst✝³ : Algebra Fqt F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    ⊢ Iff (FunctionField Fq F) (FiniteDimensional Fqt F)
  -/
  let e := IsLocalization.algEquiv Fq[X]⁰ (RatFunc Fq) Fqt
  have : ∀ (c) (x : F), e c • x = c • x := by
    intro c x
    rw [Algebra.smul_def, Algebra.smul_def]
    congr
    refine congr_fun (f := fun c => algebraMap Fqt F (e c)) ?_ c -- Porting note: Added `(f := _)`
    refine IsLocalization.ext (nonZeroDivisors Fq[X]) _ _ ?_ ?_ ?_ ?_ ?_ <;> intros <;>
      simp only [map_one, map_mul, AlgEquiv.commutes, ← IsScalarTower.algebraMap_apply]
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁹ : Field Fq
    inst✝⁸ : Field F
    Fqt : Type u_3
    inst✝⁷ : Field Fqt
    inst✝⁶ : Algebra (Polynomial Fq) Fqt
    inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
    inst✝⁴ : Algebra (RatFunc Fq) F
    inst✝³ : Algebra Fqt F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    e : AlgEquiv (Polynomial Fq) (RatFunc Fq) Fqt := IsLocalization.algEquiv (nonZ …
    this : ∀ (c : RatFunc Fq) (x : F), Eq (HSMul.hSMul (e c) x) (HSMul.hSMul c x)
    ⊢ Iff (FunctionField Fq F) (FiniteDimensional Fqt F)
  -/
  constructor <;> intro h
    /-
      case mp
      Fq : Type u_1
      F : Type u_2
      inst✝⁹ : Field Fq
      inst✝⁸ : Field F
      Fqt : Type u_3
      inst✝⁷ : Field Fqt
      inst✝⁶ : Algebra (Polynomial Fq) Fqt
      inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
      inst✝⁴ : Algebra (RatFunc Fq) F
      inst✝³ : Algebra Fqt F
      inst✝² : Algebra (Polynomial Fq) F
      inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
      inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
      e : AlgEquiv (Polynomial Fq) (RatFunc Fq) Fqt := IsLocalization.algEquiv (nonZ …
      this : ∀ (c : RatFunc Fq) (x : F), Eq (HSMul.hSMul (e c) x) (HSMul.hSMul c x)
      h : FunctionField Fq F
      ⊢ FiniteDimensional Fqt F
    -/
  · let b := Module.finBasis (RatFunc Fq) F
    /-
      case mp
      Fq : Type u_1
      F : Type u_2
      inst✝⁹ : Field Fq
      inst✝⁸ : Field F
      Fqt : Type u_3
      inst✝⁷ : Field Fqt
      inst✝⁶ : Algebra (Polynomial Fq) Fqt
      inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
      inst✝⁴ : Algebra (RatFunc Fq) F
      inst✝³ : Algebra Fqt F
      inst✝² : Algebra (Polynomial Fq) F
      inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
      inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
      e : AlgEquiv (Polynomial Fq) (RatFunc Fq) Fqt := IsLocalization.algEquiv (nonZ …
      this : ∀ (c : RatFunc Fq) (x : F), Eq (HSMul.hSMul (e c) x) (HSMul.hSMul c x)
      h : FunctionField Fq F
      b : Basis (Fin (Module.finrank (RatFunc Fq) F)) (RatFunc Fq) F := Module.finBa …
      ⊢ FiniteDimensional Fqt F
    -/
    exact FiniteDimensional.of_fintype_basis (b.mapCoeffs e this)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Fq : Type u_1
      F : Type u_2
      inst✝⁹ : Field Fq
      inst✝⁸ : Field F
      Fqt : Type u_3
      inst✝⁷ : Field Fqt
      inst✝⁶ : Algebra (Polynomial Fq) Fqt
      inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
      inst✝⁴ : Algebra (RatFunc Fq) F
      inst✝³ : Algebra Fqt F
      inst✝² : Algebra (Polynomial Fq) F
      inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
      inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
      e : AlgEquiv (Polynomial Fq) (RatFunc Fq) Fqt := IsLocalization.algEquiv (nonZ …
      this : ∀ (c : RatFunc Fq) (x : F), Eq (HSMul.hSMul (e c) x) (HSMul.hSMul c x)
      h : FiniteDimensional Fqt F
      ⊢ FunctionField Fq F
    -/
  · let b := Module.finBasis Fqt F
    /-
      case mpr
      Fq : Type u_1
      F : Type u_2
      inst✝⁹ : Field Fq
      inst✝⁸ : Field F
      Fqt : Type u_3
      inst✝⁷ : Field Fqt
      inst✝⁶ : Algebra (Polynomial Fq) Fqt
      inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
      inst✝⁴ : Algebra (RatFunc Fq) F
      inst✝³ : Algebra Fqt F
      inst✝² : Algebra (Polynomial Fq) F
      inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
      inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
      e : AlgEquiv (Polynomial Fq) (RatFunc Fq) Fqt := IsLocalization.algEquiv (nonZ …
      this : ∀ (c : RatFunc Fq) (x : F), Eq (HSMul.hSMul (e c) x) (HSMul.hSMul c x)
      h : FiniteDimensional Fqt F
      b : Basis (Fin (Module.finrank Fqt F)) Fqt F := Module.finBasis Fqt F
      ⊢ FunctionField Fq F
    -/
    refine FiniteDimensional.of_fintype_basis (b.mapCoeffs e.symm ?_)
    /-
      case mpr
      Fq : Type u_1
      F : Type u_2
      inst✝⁹ : Field Fq
      inst✝⁸ : Field F
      Fqt : Type u_3
      inst✝⁷ : Field Fqt
      inst✝⁶ : Algebra (Polynomial Fq) Fqt
      inst✝⁵ : IsFractionRing (Polynomial Fq) Fqt
      inst✝⁴ : Algebra (RatFunc Fq) F
      inst✝³ : Algebra Fqt F
      inst✝² : Algebra (Polynomial Fq) F
      inst✝¹ : IsScalarTower (Polynomial Fq) Fqt F
      inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
      e : AlgEquiv (Polynomial Fq) (RatFunc Fq) Fqt := IsLocalization.algEquiv (nonZ …
      this : ∀ (c : RatFunc Fq) (x : F), Eq (HSMul.hSMul (e c) x) (HSMul.hSMul c x)
      h : FiniteDimensional Fqt F
      b : Basis (Fin (Module.finrank Fqt F)) Fqt F := Module.finBasis Fqt F
      ⊢ ∀ (c : Fqt) (x : F), Eq (HSMul.hSMul (↑e.symm c) x) (HSMul.hSMul c x)
    -/
    intro c x; convert (this (e.symm c) x).symm; simp only [e.apply_symm_apply]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem algebraMap_injective [Algebra Fq[X] F] [Algebra (RatFunc Fq) F]
    [IsScalarTower Fq[X] (RatFunc Fq) F] : Function.Injective (⇑(algebraMap Fq[X] F)) := by
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    ⊢ Function.Injective ⇑(algebraMap (Polynomial Fq) F)
  -/
  rw [IsScalarTower.algebraMap_eq Fq[X] (RatFunc Fq) F]
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    ⊢ Function.Injective ⇑((algebraMap (RatFunc Fq) F).comp (algebraMap (Polynomia …
  -/
  exact (algebraMap (RatFunc Fq) F).injective.comp (IsFractionRing.injective Fq[X] (RatFunc Fq))
  /-
    🎉 no goals
  -/


/-- The function field analogue of `NumberField.ringOfIntegers`:
`FunctionField.ringOfIntegers Fq Fqt F` is the integral closure of `Fq[t]` in `F`.

We don't actually assume `F` is a function field over `Fq` in the definition,
only when proving its properties.
-/
def ringOfIntegers [Algebra Fq[X] F] :=
  integralClosure Fq[X] F


instance : IsDomain (ringOfIntegers Fq F) :=
  (ringOfIntegers Fq F).isDomain


instance : IsIntegralClosure (ringOfIntegers Fq F) Fq[X] F :=
  integralClosure.isIntegralClosure _ _


theorem algebraMap_injective : Function.Injective (⇑(algebraMap Fq[X] (ringOfIntegers Fq F))) := by
  have hinj : Function.Injective (⇑(algebraMap Fq[X] F)) := by
    rw [IsScalarTower.algebraMap_eq Fq[X] (RatFunc Fq) F]
    exact (algebraMap (RatFunc Fq) F).injective.comp (IsFractionRing.injective Fq[X] (RatFunc Fq))
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    hinj : Function.Injective ⇑(algebraMap (Polynomial Fq) F)
    ⊢ Function.Injective ⇑(algebraMap (Polynomial Fq) (Subtype fun x => Membership …
  -/
  rw [injective_iff_map_eq_zero (algebraMap Fq[X] (↥(ringOfIntegers Fq F)))]
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    hinj : Function.Injective ⇑(algebraMap (Polynomial Fq) F)
    ⊢ ∀ (a : Polynomial Fq), Eq ((algebraMap (Polynomial Fq) (Subtype fun x => Mem …
  -/
  intro p hp
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    hinj : Function.Injective ⇑(algebraMap (Polynomial Fq) F)
    p : Polynomial Fq
    hp : Eq ((algebraMap (Polynomial Fq) (Subtype fun x => Membership.mem (Functio …
    ⊢ Eq p 0
  -/
  rw [← Subtype.coe_inj, Subalgebra.coe_zero] at hp
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    hinj : Function.Injective ⇑(algebraMap (Polynomial Fq) F)
    p : Polynomial Fq
    hp : Eq (↑((algebraMap (Polynomial Fq) (Subtype fun x => Membership.mem (Funct …
    ⊢ Eq p 0
  -/
  rw [injective_iff_map_eq_zero (algebraMap Fq[X] F)] at hinj
  /-
    Fq : Type u_1
    F : Type u_2
    inst✝⁴ : Field Fq
    inst✝³ : Field F
    inst✝² : Algebra (Polynomial Fq) F
    inst✝¹ : Algebra (RatFunc Fq) F
    inst✝ : IsScalarTower (Polynomial Fq) (RatFunc Fq) F
    hinj : ∀ (a : Polynomial Fq), Eq ((algebraMap (Polynomial Fq) F) a) 0 → Eq a 0
    p : Polynomial Fq
    hp : Eq (↑((algebraMap (Polynomial Fq) (Subtype fun x => Membership.mem (Funct …
    ⊢ Eq p 0
  -/
  exact hinj p hp
  /-
    🎉 no goals
  -/


theorem not_isField : ¬IsField (ringOfIntegers Fq F) := by
  simpa [← (IsIntegralClosure.isIntegral_algebra Fq[X] F).isField_iff_isField
      (algebraMap_injective Fq F)] using
    Polynomial.not_isField Fq


instance : IsFractionRing (ringOfIntegers Fq F) F :=
  integralClosure.isFractionRing_of_finite_extension (RatFunc Fq) F


instance : IsIntegrallyClosed (ringOfIntegers Fq F) :=
  integralClosure.isIntegrallyClosedOfFiniteExtension (RatFunc Fq)


instance [Algebra.IsSeparable (RatFunc Fq) F] : IsNoetherian Fq[X] (ringOfIntegers Fq F) :=
  IsIntegralClosure.isNoetherian _ (RatFunc Fq) F _


instance [Algebra.IsSeparable (RatFunc Fq) F] : IsDedekindDomain (ringOfIntegers Fq F) :=
  IsIntegralClosure.isDedekindDomain Fq[X] (RatFunc Fq) F _


/-- The valuation at infinity is the nonarchimedean valuation on `Fq(t)` with uniformizer `1/t`.
Explicitly, if `f/g ∈ Fq(t)` is a nonzero quotient of polynomials, its valuation at infinity is
`Multiplicative.ofAdd(degree(f) - degree(g))`. -/
def inftyValuationDef (r : RatFunc Fq) : ℤₘ₀ :=
  if r = 0 then 0 else ↑(Multiplicative.ofAdd r.intDegree)


theorem InftyValuation.map_zero' : inftyValuationDef Fq 0 = 0 :=
  if_pos rfl


theorem InftyValuation.map_one' : inftyValuationDef Fq 1 = 1 :=
                                   /-
                                     Fq : Type u_1
                                     inst✝¹ : Field Fq
                                     inst✝ : DecidableEq (RatFunc Fq)
                                     ⊢ Eq (↑(Multiplicative.ofAdd (RatFunc.intDegree 1))) 1
                                   -/
  (if_neg one_ne_zero).trans <| by rw [RatFunc.intDegree_one, ofAdd_zero, WithZero.coe_one]
                                   /-
                                     🎉 no goals
                                   -/


theorem InftyValuation.map_mul' (x y : RatFunc Fq) :
    inftyValuationDef Fq (x * y) = inftyValuationDef Fq x * inftyValuationDef Fq y := by
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    x y : RatFunc Fq
    ⊢ Eq (FunctionField.inftyValuationDef Fq (HMul.hMul x y)) (HMul.hMul (Function …
  -/
  rw [inftyValuationDef, inftyValuationDef, inftyValuationDef]
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    x y : RatFunc Fq
    ⊢ Eq (ite (Eq (HMul.hMul x y) 0) 0 ↑(Multiplicative.ofAdd (HMul.hMul x y).intD …
  -/
  by_cases hx : x = 0
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Field Fq
      inst✝ : DecidableEq (RatFunc Fq)
      x y : RatFunc Fq
      hx : Eq x 0
      ⊢ Eq (ite (Eq (HMul.hMul x y) 0) 0 ↑(Multiplicative.ofAdd (HMul.hMul x y).intD …
    -/
  · rw [hx, zero_mul, if_pos (Eq.refl _), zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Fq : Type u_1
      inst✝¹ : Field Fq
      inst✝ : DecidableEq (RatFunc Fq)
      x y : RatFunc Fq
      hx : Not (Eq x 0)
      ⊢ Eq (ite (Eq (HMul.hMul x y) 0) 0 ↑(Multiplicative.ofAdd (HMul.hMul x y).intD …
    -/
  · by_cases hy : y = 0
      /-
        case pos
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : DecidableEq (RatFunc Fq)
        x y : RatFunc Fq
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Eq (ite (Eq (HMul.hMul x y) 0) 0 ↑(Multiplicative.ofAdd (HMul.hMul x y).intD …
      -/
    · rw [hy, mul_zero, if_pos (Eq.refl _), mul_zero]
      /-
        🎉 no goals
      -/
    · rw [if_neg hx, if_neg hy, if_neg (mul_ne_zero hx hy), ← WithZero.coe_mul, WithZero.coe_inj,
        ← ofAdd_add, RatFunc.intDegree_mul hx hy]


theorem InftyValuation.map_add_le_max' (x y : RatFunc Fq) :
    inftyValuationDef Fq (x + y) ≤ max (inftyValuationDef Fq x) (inftyValuationDef Fq y) := by
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    x y : RatFunc Fq
    ⊢ LE.le (FunctionField.inftyValuationDef Fq (HAdd.hAdd x y)) (Max.max (Functio …
  -/
  by_cases hx : x = 0
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Field Fq
      inst✝ : DecidableEq (RatFunc Fq)
      x y : RatFunc Fq
      hx : Eq x 0
      ⊢ LE.le (FunctionField.inftyValuationDef Fq (HAdd.hAdd x y)) (Max.max (Functio …
    -/
  · rw [hx, zero_add]
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Field Fq
      inst✝ : DecidableEq (RatFunc Fq)
      x y : RatFunc Fq
      hx : Eq x 0
      ⊢ LE.le (FunctionField.inftyValuationDef Fq y) (Max.max (FunctionField.inftyVa …
    -/
    conv_rhs => rw [inftyValuationDef, if_pos (Eq.refl _)]
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Field Fq
      inst✝ : DecidableEq (RatFunc Fq)
      x y : RatFunc Fq
      hx : Eq x 0
      ⊢ LE.le (FunctionField.inftyValuationDef Fq y) (Max.max 0 (FunctionField.infty …
    -/
    rw [max_eq_right (WithZero.zero_le (inftyValuationDef Fq y))]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Fq : Type u_1
      inst✝¹ : Field Fq
      inst✝ : DecidableEq (RatFunc Fq)
      x y : RatFunc Fq
      hx : Not (Eq x 0)
      ⊢ LE.le (FunctionField.inftyValuationDef Fq (HAdd.hAdd x y)) (Max.max (Functio …
    -/
  · by_cases hy : y = 0
      /-
        case pos
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : DecidableEq (RatFunc Fq)
        x y : RatFunc Fq
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ LE.le (FunctionField.inftyValuationDef Fq (HAdd.hAdd x y)) (Max.max (Functio …
      -/
    · rw [hy, add_zero]
      /-
        case pos
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : DecidableEq (RatFunc Fq)
        x y : RatFunc Fq
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ LE.le (FunctionField.inftyValuationDef Fq x) (Max.max (FunctionField.inftyVa …
      -/
      conv_rhs => rw [max_comm, inftyValuationDef, if_pos (Eq.refl _)]
      /-
        case pos
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : DecidableEq (RatFunc Fq)
        x y : RatFunc Fq
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ LE.le (FunctionField.inftyValuationDef Fq x) (Max.max 0 (FunctionField.infty …
      -/
      rw [max_eq_right (WithZero.zero_le (inftyValuationDef Fq x))]
      /-
        🎉 no goals
      -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : DecidableEq (RatFunc Fq)
        x y : RatFunc Fq
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ LE.le (FunctionField.inftyValuationDef Fq (HAdd.hAdd x y)) (Max.max (Functio …
      -/
    · by_cases hxy : x + y = 0
        /-
          case pos
          Fq : Type u_1
          inst✝¹ : Field Fq
          inst✝ : DecidableEq (RatFunc Fq)
          x y : RatFunc Fq
          hx : Not (Eq x 0)
          hy : Not (Eq y 0)
          hxy : Eq (HAdd.hAdd x y) 0
          ⊢ LE.le (FunctionField.inftyValuationDef Fq (HAdd.hAdd x y)) (Max.max (Functio …
        -/
      · rw [inftyValuationDef, if_pos hxy]; exact zero_le'
                                            /-
                                              🎉 no goals
                                            -/
      · rw [inftyValuationDef, inftyValuationDef, inftyValuationDef, if_neg hx, if_neg hy,
          if_neg hxy]
        rw [le_max_iff, WithZero.coe_le_coe, Multiplicative.ofAdd_le, WithZero.coe_le_coe,
          Multiplicative.ofAdd_le, ← le_max_iff]
        /-
          case neg
          Fq : Type u_1
          inst✝¹ : Field Fq
          inst✝ : DecidableEq (RatFunc Fq)
          x y : RatFunc Fq
          hx : Not (Eq x 0)
          hy : Not (Eq y 0)
          hxy : Not (Eq (HAdd.hAdd x y) 0)
          ⊢ LE.le (HAdd.hAdd x y).intDegree (Max.max x.intDegree y.intDegree)
        -/
        exact RatFunc.intDegree_add_le hy hxy
        /-
          🎉 no goals
        -/


@[simp]
theorem inftyValuation_of_nonzero {x : RatFunc Fq} (hx : x ≠ 0) :
    inftyValuationDef Fq x = Multiplicative.ofAdd x.intDegree := by
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    x : RatFunc Fq
    hx : Ne x 0
    ⊢ Eq (FunctionField.inftyValuationDef Fq x) ↑(Multiplicative.ofAdd x.intDegree)
  -/
  rw [inftyValuationDef, if_neg hx]
  /-
    🎉 no goals
  -/


/-- The valuation at infinity on `Fq(t)`. -/
def inftyValuation : Valuation (RatFunc Fq) ℤₘ₀ where
  toFun := inftyValuationDef Fq
  map_zero' := InftyValuation.map_zero' Fq
  map_one' := InftyValuation.map_one' Fq
  map_mul' := InftyValuation.map_mul' Fq
  map_add_le_max' := InftyValuation.map_add_le_max' Fq


@[simp]
theorem inftyValuation_apply {x : RatFunc Fq} : inftyValuation Fq x = inftyValuationDef Fq x :=
  rfl


@[simp]
theorem inftyValuation.C {k : Fq} (hk : k ≠ 0) :
    inftyValuationDef Fq (RatFunc.C k) = Multiplicative.ofAdd (0 : ℤ) := by
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    k : Fq
    hk : Ne k 0
    ⊢ Eq (FunctionField.inftyValuationDef Fq (RatFunc.C k)) ↑(Multiplicative.ofAdd …
  -/
  have hCk : RatFunc.C k ≠ 0 := (map_ne_zero _).mpr hk
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    k : Fq
    hk : Ne k 0
    hCk : Ne (RatFunc.C k) 0
    ⊢ Eq (FunctionField.inftyValuationDef Fq (RatFunc.C k)) ↑(Multiplicative.ofAdd …
  -/
  rw [inftyValuationDef, if_neg hCk, RatFunc.intDegree_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem inftyValuation.X : inftyValuationDef Fq RatFunc.X = Multiplicative.ofAdd (1 : ℤ) := by
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    ⊢ Eq (FunctionField.inftyValuationDef Fq RatFunc.X) ↑(Multiplicative.ofAdd 1)
  -/
  rw [inftyValuationDef, if_neg RatFunc.X_ne_zero, RatFunc.intDegree_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem inftyValuation.polynomial {p : Fq[X]} (hp : p ≠ 0) :
    inftyValuationDef Fq (algebraMap Fq[X] (RatFunc Fq) p) =
      Multiplicative.ofAdd (p.natDegree : ℤ) := by
  have hp' : algebraMap Fq[X] (RatFunc Fq) p ≠ 0 := by
    rw [Ne, NoZeroSMulDivisors.algebraMap_eq_zero_iff]; exact hp
  /-
    Fq : Type u_1
    inst✝¹ : Field Fq
    inst✝ : DecidableEq (RatFunc Fq)
    p : Polynomial Fq
    hp : Ne p 0
    hp' : Ne ((algebraMap (Polynomial Fq) (RatFunc Fq)) p) 0
    ⊢ Eq (FunctionField.inftyValuationDef Fq ((algebraMap (Polynomial Fq) (RatFunc …
  -/
  rw [inftyValuationDef, if_neg hp', RatFunc.intDegree_polynomial]
  /-
    🎉 no goals
  -/


/-- The valued field `Fq(t)` with the valuation at infinity. -/
def inftyValuedFqt : Valued (RatFunc Fq) ℤₘ₀ :=
  Valued.mk' <| inftyValuation Fq


theorem inftyValuedFqt.def {x : RatFunc Fq} :
    @Valued.v (RatFunc Fq) _ _ _ (inftyValuedFqt Fq) x = inftyValuationDef Fq x :=
  rfl


/-- The completion `Fq((t⁻¹))` of `Fq(t)` with respect to the valuation at infinity. -/
def FqtInfty :=
  @UniformSpace.Completion (RatFunc Fq) <| (inftyValuedFqt Fq).toUniformSpace


instance : Field (FqtInfty Fq) :=
  letI := inftyValuedFqt Fq
  UniformSpace.Completion.instField


instance : Inhabited (FqtInfty Fq) :=
  ⟨(0 : FqtInfty Fq)⟩


/-- The valuation at infinity on `k(t)` extends to a valuation on `FqtInfty`. -/
instance valuedFqtInfty : Valued (FqtInfty Fq) ℤₘ₀ :=
  @Valued.valuedCompletion _ _ _ _ (inftyValuedFqt Fq)


theorem valuedFqtInfty.def {x : FqtInfty Fq} :
    Valued.v x = @Valued.extension (RatFunc Fq) _ _ _ (inftyValuedFqt Fq) x :=
  rfl


