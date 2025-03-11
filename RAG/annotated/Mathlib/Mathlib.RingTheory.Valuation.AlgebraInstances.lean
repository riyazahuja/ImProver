instance : Algebra v.valuationSubring L := Algebra.ofSubring v.valuationSubring.toSubring


theorem algebraMap_injective : Injective (algebraMap v.valuationSubring L) :=
  (NoZeroSMulDivisors.algebraMap_injective K L).comp (IsFractionRing.injective _ _)


theorem isIntegral_of_mem_ringOfIntegers {x : L} (hx : x ∈ integralClosure v.valuationSubring L) :
    IsIntegral v.valuationSubring (⟨x, hx⟩ : integralClosure v.valuationSubring L) := by
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : Membership.mem (integralClosure (Subtype fun x => Membership.mem v.valuat …
    ⊢ IsIntegral (Subtype fun x => Membership.mem v.valuationSubring x) ⟨x, hx⟩
  -/
  obtain ⟨P, hPm, hP⟩ := hx
  /-
    case intro.intro
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    P : Polynomial (Subtype fun x => Membership.mem v.valuationSubring x)
    hPm : P.Monic
    hP : Eq (Polynomial.eval₂ (algebraMap (Subtype fun x => Membership.mem v.valua …
    ⊢ IsIntegral (Subtype fun x => Membership.mem v.valuationSubring x) ⟨x, ⋯⟩
  -/
  refine ⟨P, hPm, ?_⟩
  rw [← Polynomial.aeval_def, ← Subalgebra.coe_eq_zero, Polynomial.aeval_subalgebra_coe,
    Polynomial.aeval_def, Subtype.coe_mk, hP]


theorem isIntegral_of_mem_ringOfIntegers' {x : (integralClosure v.valuationSubring L)} :
    IsIntegral v.valuationSubring (x : integralClosure v.valuationSubring L) := by
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : Subtype fun x => Membership.mem (integralClosure (Subtype fun x => Members …
    ⊢ IsIntegral (Subtype fun x => Membership.mem v.valuationSubring x) x
  -/
  apply isIntegral_of_mem_ringOfIntegers
  /-
    🎉 no goals
  -/


instance : IsScalarTower v.valuationSubring L E := Subring.instIsScalarTowerSubtypeMem _


/-- Given an algebra between two field extensions `L` and `E` of a field `K` with a valuation `v`,
  create an algebra between their two rings of integers. -/
instance algebra :
    Algebra (integralClosure v.valuationSubring L) (integralClosure v.valuationSubring E) :=
  RingHom.toAlgebra
    { toFun := fun k => ⟨algebraMap L E k, IsIntegral.algebraMap k.2⟩
      map_zero' :=
                          /-
                            K : Type u_1
                            inst✝⁶ : Field K
                            v : Valuation K (WithZero (Multiplicative Int))
                            L : Type u_2
                            inst✝⁵ : Field L
                            inst✝⁴ : Algebra K L
                            E : Type ?u.17387
                            inst✝³ : Field E
                            inst✝² : Algebra K E
                            inst✝¹ : Algebra L E
                            inst✝ : IsScalarTower K L E
                            ⊢ Eq ↑((↑{ toFun := fun k => ⟨(algebraMap L E) ↑k, ⋯⟩, map_one' := ⋯, map_mul' …
                          -/
                                    /-
                                      K : Type u_1
                                      inst✝⁶ : Field K
                                      v : Valuation K (WithZero (Multiplicative Int))
                                      L : Type u_2
                                      inst✝⁵ : Field L
                                      inst✝⁴ : Algebra K L
                                      E : Type ?u.17387
                                      inst✝³ : Field E
                                      inst✝² : Algebra K E
                                      inst✝¹ : Algebra L E
                                      inst✝ : IsScalarTower K L E
                                      ⊢ Eq ↑((fun k => ⟨(algebraMap L E) ↑k, ⋯⟩) 1) ↑1
                                    -/
        Subtype.ext <| by simp only [Subtype.coe_mk, Subalgebra.coe_zero, _root_.map_zero]
                                    /-
                                      🎉 no goals
                                    -/
                          /-
                            🎉 no goals
                          -/
      map_one' := Subtype.ext <| by simp only [Subtype.coe_mk, Subalgebra.coe_one, _root_.map_one]
      map_add' := fun x y =>
                          /-
                            K : Type u_1
                            inst✝⁶ : Field K
                            v : Valuation K (WithZero (Multiplicative Int))
                            L : Type u_2
                            inst✝⁵ : Field L
                            inst✝⁴ : Algebra K L
                            E : Type ?u.17387
                            inst✝³ : Field E
                            inst✝² : Algebra K E
                            inst✝¹ : Algebra L E
                            inst✝ : IsScalarTower K L E
                            x y : Subtype fun x => Membership.mem (integralClosure (Subtype fun x => Membe …
                            ⊢ Eq ↑({ toFun := fun k => ⟨(algebraMap L E) ↑k, ⋯⟩, map_one' := ⋯ }.toFun (HM …
                          -/
                          /-
                            K : Type u_1
                            inst✝⁶ : Field K
                            v : Valuation K (WithZero (Multiplicative Int))
                            L : Type u_2
                            inst✝⁵ : Field L
                            inst✝⁴ : Algebra K L
                            E : Type ?u.17387
                            inst✝³ : Field E
                            inst✝² : Algebra K E
                            inst✝¹ : Algebra L E
                            inst✝ : IsScalarTower K L E
                            x y : Subtype fun x => Membership.mem (integralClosure (Subtype fun x => Membe …
                            ⊢ Eq ↑((↑{ toFun := fun k => ⟨(algebraMap L E) ↑k, ⋯⟩, map_one' := ⋯, map_mul' …
                          -/
                          /-
                            🎉 no goals
                          -/
        Subtype.ext <| by simp only [_root_.map_add, Subalgebra.coe_add, Subtype.coe_mk]
                          /-
                            🎉 no goals
                          -/
      map_mul' := fun x y =>
        Subtype.ext <| by simp only [Subalgebra.coe_mul, _root_.map_mul, Subtype.coe_mk] }


/-- A ring equivalence between the integral closure of the valuation subring of `K` in `L`
  and a ring `R` satisfying `isIntegralClosure R v.valuationSubring L`. -/
protected noncomputable def equiv (R : Type*) [CommRing R] [Algebra v.valuationSubring R]
    [Algebra R L] [IsScalarTower v.valuationSubring R L]
    [IsIntegralClosure R v.valuationSubring L] : integralClosure v.valuationSubring L ≃+* R := by
  have := IsScalarTower.subalgebra' (valuationSubring v) L L
    (integralClosure (valuationSubring v) L)
  exact (IsIntegralClosure.equiv v.valuationSubring R L
    (integralClosure v.valuationSubring L)).symm.toRingEquiv


theorem integralClosure_algebraMap_injective :
    Injective (algebraMap v.valuationSubring (integralClosure v.valuationSubring L)) := by
  have hinj : Injective ⇑(algebraMap v.valuationSubring L) :=
    ValuationSubring.algebraMap_injective v L
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    hinj : Function.Injective ⇑(algebraMap (Subtype fun x => Membership.mem v.valu …
    ⊢ Function.Injective ⇑(algebraMap (Subtype fun x => Membership.mem v.valuation …
  -/
  rw [injective_iff_map_eq_zero (algebraMap v.valuationSubring _)]
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    hinj : Function.Injective ⇑(algebraMap (Subtype fun x => Membership.mem v.valu …
    ⊢ ∀ (a : Subtype fun x => Membership.mem v.valuationSubring x), Eq ((algebraMa …
  -/
  intro x hx
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    hinj : Function.Injective ⇑(algebraMap (Subtype fun x => Membership.mem v.valu …
    x : Subtype fun x => Membership.mem v.valuationSubring x
    hx : Eq ((algebraMap (Subtype fun x => Membership.mem v.valuationSubring x) (S …
    ⊢ Eq x 0
  -/
  rw [← Subtype.coe_inj, Subalgebra.coe_zero] at hx
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    hinj : Function.Injective ⇑(algebraMap (Subtype fun x => Membership.mem v.valu …
    x : Subtype fun x => Membership.mem v.valuationSubring x
    hx : Eq (↑((algebraMap (Subtype fun x => Membership.mem v.valuationSubring x)  …
    ⊢ Eq x 0
  -/
  rw [injective_iff_map_eq_zero (algebraMap v.valuationSubring L)] at hinj
  /-
    K : Type u_1
    inst✝² : Field K
    v : Valuation K (WithZero (Multiplicative Int))
    L : Type u_2
    inst✝¹ : Field L
    inst✝ : Algebra K L
    hinj : ∀ (a : Subtype fun x => Membership.mem v.valuationSubring x), Eq ((alge …
    x : Subtype fun x => Membership.mem v.valuationSubring x
    hx : Eq (↑((algebraMap (Subtype fun x => Membership.mem v.valuationSubring x)  …
    ⊢ Eq x 0
  -/
  exact hinj x hx
  /-
    🎉 no goals
  -/


