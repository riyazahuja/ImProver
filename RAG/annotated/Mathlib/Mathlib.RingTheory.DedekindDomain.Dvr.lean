/-- A Dedekind domain is an integral domain that is Noetherian, and the
localization at every nonzero prime is a discrete valuation ring.

This is equivalent to `IsDedekindDomain`.
-/
class IsDedekindDomainDvr extends IsNoetherian A A : Prop where
  is_dvr_at_nonzero_prime : ∀ P ≠ (⊥ : Ideal A), ∀ _ : P.IsPrime,
    IsDiscreteValuationRing (Localization.AtPrime P)


/-- Localizing a domain of Krull dimension `≤ 1` gives another ring of Krull dimension `≤ 1`.

Note that the same proof can/should be generalized to preserving any Krull dimension,
once we have a suitable definition.
-/
theorem Ring.DimensionLEOne.localization {R : Type*} (Rₘ : Type*) [CommRing R] [IsDomain R]
    [CommRing Rₘ] [Algebra R Rₘ] {M : Submonoid R} [IsLocalization M Rₘ] (hM : M ≤ R⁰)
    [h : Ring.DimensionLEOne R] : Ring.DimensionLEOne Rₘ := ⟨by
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    ⊢ ∀ {p : Ideal Rₘ}, Ne p Bot.bot → p.IsPrime → p.IsMaximal
  -/
  intro p hp0 hpp
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    p : Ideal Rₘ
    hp0 : Ne p Bot.bot
    hpp : p.IsPrime
    ⊢ p.IsMaximal
  -/
  refine Ideal.isMaximal_def.mpr ⟨hpp.ne_top, Ideal.maximal_of_no_maximal fun P hpP hPm => ?_⟩
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    p : Ideal Rₘ
    hp0 : Ne p Bot.bot
    hpp : p.IsPrime
    P : Ideal Rₘ
    hpP : LT.lt p P
    hPm : P.IsMaximal
    ⊢ False
  -/
  have hpP' : (⟨p, hpp⟩ : { p : Ideal Rₘ // p.IsPrime }) < ⟨P, hPm.isPrime⟩ := hpP
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    p : Ideal Rₘ
    hp0 : Ne p Bot.bot
    hpp : p.IsPrime
    P : Ideal Rₘ
    hpP : LT.lt p P
    hPm : P.IsMaximal
    hpP' : LT.lt ⟨p, hpp⟩ ⟨P, ⋯⟩
    ⊢ False
  -/
  rw [← (IsLocalization.orderIsoOfPrime M Rₘ).lt_iff_lt] at hpP'
  haveI : Ideal.IsPrime (Ideal.comap (algebraMap R Rₘ) p) :=
    ((IsLocalization.orderIsoOfPrime M Rₘ) ⟨p, hpp⟩).2.1
  haveI : Ideal.IsPrime (Ideal.comap (algebraMap R Rₘ) P) :=
    ((IsLocalization.orderIsoOfPrime M Rₘ) ⟨P, hPm.isPrime⟩).2.1
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    p : Ideal Rₘ
    hp0 : Ne p Bot.bot
    hpp : p.IsPrime
    P : Ideal Rₘ
    hpP : LT.lt p P
    hPm : P.IsMaximal
    hpP' : LT.lt ((IsLocalization.orderIsoOfPrime M Rₘ) ⟨p, hpp⟩) ((IsLocalization …
    this✝ : (Ideal.comap (algebraMap R Rₘ) p).IsPrime
    this : (Ideal.comap (algebraMap R Rₘ) P).IsPrime
    ⊢ False
  -/
  have hlt : Ideal.comap (algebraMap R Rₘ) p < Ideal.comap (algebraMap R Rₘ) P := hpP'
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    p : Ideal Rₘ
    hp0 : Ne p Bot.bot
    hpp : p.IsPrime
    P : Ideal Rₘ
    hpP : LT.lt p P
    hPm : P.IsMaximal
    hpP' : LT.lt ((IsLocalization.orderIsoOfPrime M Rₘ) ⟨p, hpp⟩) ((IsLocalization …
    this✝ : (Ideal.comap (algebraMap R Rₘ) p).IsPrime
    this : (Ideal.comap (algebraMap R Rₘ) P).IsPrime
    hlt : LT.lt (Ideal.comap (algebraMap R Rₘ) p) (Ideal.comap (algebraMap R Rₘ) P)
    ⊢ False
  -/
  refine h.not_lt_lt ⊥ (Ideal.comap _ _) (Ideal.comap _ _) ⟨?_, hlt⟩
  /-
    R : Type u_2
    Rₘ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    M : Submonoid R
    inst✝ : IsLocalization M Rₘ
    hM : LE.le M (nonZeroDivisors R)
    h : Ring.DimensionLEOne R
    p : Ideal Rₘ
    hp0 : Ne p Bot.bot
    hpp : p.IsPrime
    P : Ideal Rₘ
    hpP : LT.lt p P
    hPm : P.IsMaximal
    hpP' : LT.lt ((IsLocalization.orderIsoOfPrime M Rₘ) ⟨p, hpp⟩) ((IsLocalization …
    this✝ : (Ideal.comap (algebraMap R Rₘ) p).IsPrime
    this : (Ideal.comap (algebraMap R Rₘ) P).IsPrime
    hlt : LT.lt (Ideal.comap (algebraMap R Rₘ) p) (Ideal.comap (algebraMap R Rₘ) P)
    ⊢ LT.lt Bot.bot (Ideal.comap (algebraMap R Rₘ) p)
  -/
  exact IsLocalization.bot_lt_comap_prime _ _ hM _ hp0⟩
  /-
    🎉 no goals
  -/


/-- The localization of a Dedekind domain is a Dedekind domain. -/
theorem IsLocalization.isDedekindDomain [IsDedekindDomain A] {M : Submonoid A} (hM : M ≤ A⁰)
    (Aₘ : Type*) [CommRing Aₘ] [IsDomain Aₘ] [Algebra A Aₘ] [IsLocalization M Aₘ] :
    IsDedekindDomain Aₘ := by
  have h : ∀ y : M, IsUnit (algebraMap A (FractionRing A) y) := by
    rintro ⟨y, hy⟩
    exact IsUnit.mk0 _ (mt IsFractionRing.to_map_eq_zero_iff.mp (nonZeroDivisors.ne_zero (hM hy)))
  /-
    A : Type u_1
    inst✝⁶ : CommRing A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDedekindDomain A
    M : Submonoid A
    hM : LE.le M (nonZeroDivisors A)
    Aₘ : Type u_2
    inst✝³ : CommRing Aₘ
    inst✝² : IsDomain Aₘ
    inst✝¹ : Algebra A Aₘ
    inst✝ : IsLocalization M Aₘ
    h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
    ⊢ IsDedekindDomain Aₘ
  -/
  letI : Algebra Aₘ (FractionRing A) := RingHom.toAlgebra (IsLocalization.lift h)
  haveI : IsScalarTower A Aₘ (FractionRing A) :=
    IsScalarTower.of_algebraMap_eq fun x => (IsLocalization.lift_eq h x).symm
  haveI : IsFractionRing Aₘ (FractionRing A) :=
    IsFractionRing.isFractionRing_of_isDomain_of_isLocalization M _ _
  /-
    A : Type u_1
    inst✝⁶ : CommRing A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDedekindDomain A
    M : Submonoid A
    hM : LE.le M (nonZeroDivisors A)
    Aₘ : Type u_2
    inst✝³ : CommRing Aₘ
    inst✝² : IsDomain Aₘ
    inst✝¹ : Algebra A Aₘ
    inst✝ : IsLocalization M Aₘ
    h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
    this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
    this✝ : IsScalarTower A Aₘ (FractionRing A)
    this : IsFractionRing Aₘ (FractionRing A)
    ⊢ IsDedekindDomain Aₘ
  -/
  refine (isDedekindDomain_iff _ (FractionRing A)).mpr ⟨?_, ?_, ?_, ?_⟩
    /-
      case refine_1
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      ⊢ IsDomain Aₘ
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      ⊢ IsNoetherianRing Aₘ
    -/
  · exact IsLocalization.isNoetherianRing M _ inferInstance
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      ⊢ Ring.DimensionLEOne Aₘ
    -/
  · exact Ring.DimensionLEOne.localization Aₘ hM
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      ⊢ ∀ {x : FractionRing A}, IsIntegral Aₘ x → Exists fun y => Eq ((algebraMap Aₘ …
    -/
  · intro x hx
    /-
      case refine_4
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      x : FractionRing A
      hx : IsIntegral Aₘ x
      ⊢ Exists fun y => Eq ((algebraMap Aₘ (FractionRing A)) y) x
    -/
    obtain ⟨⟨y, y_mem⟩, hy⟩ := hx.exists_multiple_integral_of_isLocalization M _
    /-
      case refine_4.intro.mk
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      x : FractionRing A
      hx : IsIntegral Aₘ x
      y : A
      y_mem : Membership.mem M y
      hy : IsIntegral A (HSMul.hSMul ⟨y, y_mem⟩ x)
      ⊢ Exists fun y => Eq ((algebraMap Aₘ (FractionRing A)) y) x
    -/
    obtain ⟨z, hz⟩ := (isIntegrallyClosed_iff _).mp IsDedekindRing.toIsIntegralClosure hy
    /-
      case refine_4.intro.mk.intro
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      x : FractionRing A
      hx : IsIntegral Aₘ x
      y : A
      y_mem : Membership.mem M y
      hy : IsIntegral A (HSMul.hSMul ⟨y, y_mem⟩ x)
      z : A
      hz : Eq ((algebraMap A (FractionRing A)) z) (HSMul.hSMul ⟨y, y_mem⟩ x)
      ⊢ Exists fun y => Eq ((algebraMap Aₘ (FractionRing A)) y) x
    -/
    refine ⟨IsLocalization.mk' Aₘ z ⟨y, y_mem⟩, (IsLocalization.lift_mk'_spec _ _ _ _).mpr ?_⟩
    /-
      case refine_4.intro.mk.intro
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      x : FractionRing A
      hx : IsIntegral Aₘ x
      y : A
      y_mem : Membership.mem M y
      hy : IsIntegral A (HSMul.hSMul ⟨y, y_mem⟩ x)
      z : A
      hz : Eq ((algebraMap A (FractionRing A)) z) (HSMul.hSMul ⟨y, y_mem⟩ x)
      ⊢ Eq ((algebraMap A (FractionRing A)) z) (HMul.hMul ((algebraMap A (FractionRi …
    -/
    rw [hz, ← Algebra.smul_def]
    /-
      case refine_4.intro.mk.intro
      A : Type u_1
      inst✝⁶ : CommRing A
      inst✝⁵ : IsDomain A
      inst✝⁴ : IsDedekindDomain A
      M : Submonoid A
      hM : LE.le M (nonZeroDivisors A)
      Aₘ : Type u_2
      inst✝³ : CommRing Aₘ
      inst✝² : IsDomain Aₘ
      inst✝¹ : Algebra A Aₘ
      inst✝ : IsLocalization M Aₘ
      h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap A (Fracti …
      this✝¹ : Algebra Aₘ (FractionRing A) := (IsLocalization.lift h).toAlgebra
      this✝ : IsScalarTower A Aₘ (FractionRing A)
      this : IsFractionRing Aₘ (FractionRing A)
      x : FractionRing A
      hx : IsIntegral Aₘ x
      y : A
      y_mem : Membership.mem M y
      hy : IsIntegral A (HSMul.hSMul ⟨y, y_mem⟩ x)
      z : A
      hz : Eq ((algebraMap A (FractionRing A)) z) (HSMul.hSMul ⟨y, y_mem⟩ x)
      ⊢ Eq (HSMul.hSMul ⟨y, y_mem⟩ x) (HSMul.hSMul (↑⟨y, y_mem⟩) x)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The localization of a Dedekind domain at every nonzero prime ideal is a Dedekind domain. -/
theorem IsLocalization.AtPrime.isDedekindDomain [IsDedekindDomain A] (P : Ideal A) [P.IsPrime]
    (Aₘ : Type*) [CommRing Aₘ] [IsDomain Aₘ] [Algebra A Aₘ] [IsLocalization.AtPrime Aₘ P] :
    IsDedekindDomain Aₘ :=
  IsLocalization.isDedekindDomain A P.primeCompl_le_nonZeroDivisors Aₘ


instance Localization.AtPrime.isDedekindDomain [IsDedekindDomain A] (P : Ideal A) [P.IsPrime] :
    IsDedekindDomain (Localization.AtPrime P) :=
  IsLocalization.AtPrime.isDedekindDomain A P _


theorem IsLocalization.AtPrime.not_isField {P : Ideal A} (hP : P ≠ ⊥) [pP : P.IsPrime] (Aₘ : Type*)
    [CommRing Aₘ] [Algebra A Aₘ] [IsLocalization.AtPrime Aₘ P] : ¬ IsField Aₘ := by
  /-
    A : Type u_1
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    P : Ideal A
    hP : Ne P Bot.bot
    pP : P.IsPrime
    Aₘ : Type u_2
    inst✝² : CommRing Aₘ
    inst✝¹ : Algebra A Aₘ
    inst✝ : IsLocalization.AtPrime Aₘ P
    ⊢ Not (IsField Aₘ)
  -/
  intro h
  /-
    A : Type u_1
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    P : Ideal A
    hP : Ne P Bot.bot
    pP : P.IsPrime
    Aₘ : Type u_2
    inst✝² : CommRing Aₘ
    inst✝¹ : Algebra A Aₘ
    inst✝ : IsLocalization.AtPrime Aₘ P
    h : IsField Aₘ
    ⊢ False
  -/
  letI := h.toField
  /-
    A : Type u_1
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    P : Ideal A
    hP : Ne P Bot.bot
    pP : P.IsPrime
    Aₘ : Type u_2
    inst✝² : CommRing Aₘ
    inst✝¹ : Algebra A Aₘ
    inst✝ : IsLocalization.AtPrime Aₘ P
    h : IsField Aₘ
    this : Field Aₘ := h.toField
    ⊢ False
  -/
  obtain ⟨x, x_mem, x_ne⟩ := P.ne_bot_iff.mp hP
  exact
    (IsLocalRing.maximalIdeal.isMaximal _).ne_top
      (Ideal.eq_top_of_isUnit_mem _
        ((IsLocalization.AtPrime.to_map_mem_maximal_iff Aₘ P _).mpr x_mem)
        (isUnit_iff_ne_zero.mpr
          ((map_ne_zero_iff (algebraMap A Aₘ)
                (IsLocalization.injective Aₘ P.primeCompl_le_nonZeroDivisors)).mpr
            x_ne)))


/-- In a Dedekind domain, the localization at every nonzero prime ideal is a DVR. -/
theorem IsLocalization.AtPrime.isDiscreteValuationRing_of_dedekind_domain [IsDedekindDomain A]
    {P : Ideal A} (hP : P ≠ ⊥) [pP : P.IsPrime] (Aₘ : Type*) [CommRing Aₘ] [IsDomain Aₘ]
    [Algebra A Aₘ] [IsLocalization.AtPrime Aₘ P] : IsDiscreteValuationRing Aₘ := by
  classical
  letI : IsNoetherianRing Aₘ :=
    IsLocalization.isNoetherianRing P.primeCompl _ IsDedekindRing.toIsNoetherian
  letI : IsLocalRing Aₘ := IsLocalization.AtPrime.isLocalRing Aₘ P
  have hnf := IsLocalization.AtPrime.not_isField A hP Aₘ
  exact
    ((IsDiscreteValuationRing.TFAE Aₘ hnf).out 0 2).mpr
      (IsLocalization.AtPrime.isDedekindDomain A P _)


/-- Dedekind domains, in the sense of Noetherian integrally closed domains of Krull dimension ≤ 1,
are also Dedekind domains in the sense of Noetherian domains where the localization at every
nonzero prime ideal is a DVR. -/
instance IsDedekindDomain.isDedekindDomainDvr [IsDedekindDomain A] : IsDedekindDomainDvr A where
  is_dvr_at_nonzero_prime := fun _ hP _ =>
    IsLocalization.AtPrime.isDiscreteValuationRing_of_dedekind_domain A hP _


instance IsDedekindDomainDvr.ring_dimensionLEOne [h : IsDedekindDomainDvr A] :
    Ring.DimensionLEOne A where
  maximalOfPrime := by
    /-
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      ⊢ ∀ {p : Ideal A}, Ne p Bot.bot → p.IsPrime → p.IsMaximal
    -/
    intro p hp hpp
    /-
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      p : Ideal A
      hp : Ne p Bot.bot
      hpp : p.IsPrime
      ⊢ p.IsMaximal
    -/
    rcases p.exists_le_maximal (Ideal.IsPrime.ne_top hpp) with ⟨q, hq, hpq⟩
    /-
      case intro.intro
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      p : Ideal A
      hp : Ne p Bot.bot
      hpp : p.IsPrime
      q : Ideal A
      hq : q.IsMaximal
      hpq : LE.le p q
      ⊢ p.IsMaximal
    -/
    let f := (IsLocalization.orderIsoOfPrime q.primeCompl (Localization.AtPrime q)).symm
    /-
      case intro.intro
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      p : Ideal A
      hp : Ne p Bot.bot
      hpp : p.IsPrime
      q : Ideal A
      hq : q.IsMaximal
      hpq : LE.le p q
      f : OrderIso (Subtype fun p => And p.IsPrime (Disjoint ↑q.primeCompl ↑p)) (Sub …
      ⊢ p.IsMaximal
    -/
    let P := f ⟨p, hpp, hpq.disjoint_compl_left⟩
    /-
      case intro.intro
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      p : Ideal A
      hp : Ne p Bot.bot
      hpp : p.IsPrime
      q : Ideal A
      hq : q.IsMaximal
      hpq : LE.le p q
      f : OrderIso (Subtype fun p => And p.IsPrime (Disjoint ↑q.primeCompl ↑p)) (Sub …
      P : Subtype fun p => p.IsPrime := f ⟨p, ⋯⟩
      ⊢ p.IsMaximal
    -/
    let Q := f ⟨q, hq.isPrime, Set.disjoint_left.mpr fun _ a => a⟩
    have hinj : Function.Injective (algebraMap A (Localization.AtPrime q)) :=
      IsLocalization.injective (Localization.AtPrime q) q.primeCompl_le_nonZeroDivisors
    /-
      case intro.intro
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      p : Ideal A
      hp : Ne p Bot.bot
      hpp : p.IsPrime
      q : Ideal A
      hq : q.IsMaximal
      hpq : LE.le p q
      f : OrderIso (Subtype fun p => And p.IsPrime (Disjoint ↑q.primeCompl ↑p)) (Sub …
      P : Subtype fun p => p.IsPrime := f ⟨p, ⋯⟩
      Q : Subtype fun p => p.IsPrime := f ⟨q, ⋯⟩
      hinj : Function.Injective ⇑(algebraMap A (Localization.AtPrime q))
      ⊢ p.IsMaximal
    -/
    have hp1 : P.1 ≠ ⊥ := fun x => hp ((p.map_eq_bot_iff_of_injective hinj).mp x)
    have hq1 : Q.1 ≠ ⊥ :=
      fun x => (ne_bot_of_le_ne_bot hp hpq) ((q.map_eq_bot_iff_of_injective hinj).mp x)
    rcases (IsDiscreteValuationRing.iff_pid_with_one_nonzero_prime (Localization.AtPrime q)).mp
      (h.is_dvr_at_nonzero_prime q (ne_bot_of_le_ne_bot hp hpq) hq.isPrime) with ⟨_, huq⟩
    rw [show p = q from Subtype.val_inj.mpr <| f.injective <|
      Subtype.val_inj.mp (huq.unique ⟨hp1, P.2⟩ ⟨hq1, Q.2⟩)]
    /-
      case intro.intro.intro
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      h : IsDedekindDomainDvr A
      p : Ideal A
      hp : Ne p Bot.bot
      hpp : p.IsPrime
      q : Ideal A
      hq : q.IsMaximal
      hpq : LE.le p q
      f : OrderIso (Subtype fun p => And p.IsPrime (Disjoint ↑q.primeCompl ↑p)) (Sub …
      P : Subtype fun p => p.IsPrime := f ⟨p, ⋯⟩
      Q : Subtype fun p => p.IsPrime := f ⟨q, ⋯⟩
      hinj : Function.Injective ⇑(algebraMap A (Localization.AtPrime q))
      hp1 : Ne (↑P) Bot.bot
      hq1 : Ne (↑Q) Bot.bot
      left✝ : IsPrincipalIdealRing (Localization.AtPrime q)
      huq : ExistsUnique fun P => And (Ne P Bot.bot) P.IsPrime
      ⊢ q.IsMaximal
    -/
    exact hq
    /-
      🎉 no goals
    -/


instance IsDedekindDomainDvr.isIntegrallyClosed [h : IsDedekindDomainDvr A] :
    IsIntegrallyClosed A :=
  IsIntegrallyClosed.of_localization_maximal <| fun p hp0 hpm ↦
    let ⟨_, _⟩ := (IsDiscreteValuationRing.iff_pid_with_one_nonzero_prime
      (Localization.AtPrime p)).mp (h.is_dvr_at_nonzero_prime p hp0 hpm.isPrime)
    inferInstance


/-- If an integral domain is Noetherian, and the localization at every nonzero prime is
a discrete valuation ring, then it is a Dedekind domain. -/
instance IsDedekindDomainDvr.isDedekindDomain [IsDedekindDomainDvr A] : IsDedekindDomain A where

