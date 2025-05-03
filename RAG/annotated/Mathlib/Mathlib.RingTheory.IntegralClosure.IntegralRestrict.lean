/-- The lift `End(B/A) → End(L/K)` in an ALKB setup.
This is inverse to the restriction. See `galRestrictHom`. -/
noncomputable
def galLift (σ : B →ₐ[A] B) : L →ₐ[K] L :=
  haveI := (IsFractionRing.injective A K).isDomain
  haveI := NoZeroSMulDivisors.trans A K L
  haveI := IsIntegralClosure.isLocalization A K L B
  haveI H : ∀ (y :  Algebra.algebraMapSubmonoid B A⁰),
      IsUnit (((algebraMap B L).comp σ) (y : B)) := by
    /-
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ : AlgHom A B B
      this✝¹ : IsDomain A
      this✝ : NoZeroSMulDivisors A L
      this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid B (nonZe …
    -/
    rintro ⟨_, x, hx, rfl⟩
    simpa only [RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply, AlgHom.commutes,
      isUnit_iff_ne_zero, ne_eq, map_eq_zero_iff _ (NoZeroSMulDivisors.algebraMap_injective _ _),
      ← IsScalarTower.algebraMap_apply] using nonZeroDivisors.ne_zero hx
  haveI H_eq : (IsLocalization.lift (S := L) H).comp (algebraMap K L) = (algebraMap K L) := by
    /-
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ : AlgHom A B B
      this✝¹ : IsDomain A
      this✝ : NoZeroSMulDivisors A L
      this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
      H : ∀ (y : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid B (non …
      ⊢ Eq ((IsLocalization.lift H).comp (algebraMap K L)) (algebraMap K L)
    -/
    apply IsLocalization.ringHom_ext A⁰
    /-
      case h
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ : AlgHom A B B
      this✝¹ : IsDomain A
      this✝ : NoZeroSMulDivisors A L
      this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
      H : ∀ (y : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid B (non …
      ⊢ Eq (((IsLocalization.lift H).comp (algebraMap K L)).comp (algebraMap A K)) ( …
    -/
    ext
    simp only [RingHom.coe_comp, Function.comp_apply, ← IsScalarTower.algebraMap_apply A K L,
      IsScalarTower.algebraMap_apply A B L, IsLocalization.lift_eq,
      RingHom.coe_coe, AlgHom.commutes]
  { IsLocalization.lift (S := L) H with commutes' := DFunLike.congr_fun H_eq }


/-- The restriction `End(L/K) → End(B/A)` in an AKLB setup.
Also see `galRestrict` for the `AlgEquiv` version. -/
noncomputable
def galRestrictHom : (L →ₐ[K] L) ≃* (B →ₐ[A] B) where
  toFun := fun f ↦ (IsIntegralClosure.equiv A (integralClosure A L) L B).toAlgHom.comp
      (((f.restrictScalars A).comp (IsScalarTower.toAlgHom A B L)).codRestrict
        (integralClosure A L) (fun x ↦ IsIntegral.map _ (IsIntegralClosure.isIntegral A L x)))
  map_mul' := by
    /-
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      ⊢ ∀ (x y : AlgHom K L L), Eq ({ toFun := fun f => (↑(IsIntegralClosure.equiv A …
    -/
    intros σ₁ σ₂
    /-
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ₁ σ₂ : AlgHom K L L
      ⊢ Eq ({ toFun := fun f => (↑(IsIntegralClosure.equiv A (Subtype fun x => Membe …
    -/
    ext x
    /-
      case H
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ₁ σ₂ : AlgHom K L L
      x : B
      ⊢ Eq (({ toFun := fun f => (↑(IsIntegralClosure.equiv A (Subtype fun x => Memb …
    -/
    apply (IsIntegralClosure.equiv A (integralClosure A L) L B).symm.injective
    /-
      case H.a
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ₁ σ₂ : AlgHom K L L
      x : B
      ⊢ Eq ((IsIntegralClosure.equiv A (Subtype fun x => Membership.mem (integralClo …
    -/
    ext
    /-
      case H.a.a
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ₁ σ₂ : AlgHom K L L
      x : B
      ⊢ Eq ↑((IsIntegralClosure.equiv A (Subtype fun x => Membership.mem (integralCl …
    -/
    dsimp
    simp only [AlgEquiv.symm_apply_apply, AlgHom.coe_codRestrict, AlgHom.coe_restrictScalars',
      AlgHom.coe_comp, AlgHom.restrictDomain, IsScalarTower.coe_toAlgHom', Function.comp_apply,
      AlgHom.mul_apply, IsIntegralClosure.algebraMap_equiv, Subalgebra.algebraMap_eq]
    /-
      case H.a.a
      A : Type u_1
      K : Type u_2
      L : Type u_3
      B : Type u_4
      inst✝¹³ : CommRing A
      inst✝¹² : CommRing B
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Field K
      inst✝⁹ : Field L
      inst✝⁸ : Algebra A K
      inst✝⁷ : IsFractionRing A K
      inst✝⁶ : Algebra B L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra A L
      inst✝³ : IsScalarTower A B L
      inst✝² : IsScalarTower A K L
      inst✝¹ : IsIntegralClosure B A L
      inst✝ : Algebra.IsAlgebraic K L
      σ₁ σ₂ : AlgHom K L L
      x : B
      ⊢ Eq (σ₁ (σ₂ ((algebraMap B L) x))) (σ₁ (((algebraMap L L).comp ↑(integralClos …
    -/
                                /-
                                  A : Type u_1
                                  K : Type u_2
                                  L : Type u_3
                                  B : Type u_4
                                  inst✝¹³ : CommRing A
                                  inst✝¹² : CommRing B
                                  inst✝¹¹ : Algebra A B
                                  inst✝¹⁰ : Field K
                                  inst✝⁹ : Field L
                                  inst✝⁸ : Algebra A K
                                  inst✝⁷ : IsFractionRing A K
                                  inst✝⁶ : Algebra B L
                                  inst✝⁵ : Algebra K L
                                  inst✝⁴ : Algebra A L
                                  inst✝³ : IsScalarTower A B L
                                  inst✝² : IsScalarTower A K L
                                  inst✝¹ : IsIntegralClosure B A L
                                  inst✝ : Algebra.IsAlgebraic K L
                                  σ : AlgHom K L L
                                  this✝ : IsDomain A
                                  this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
                                  x : B
                                  ⊢ Eq (((↑(galLift A K L B ((fun f => (↑(IsIntegralClosure.equiv A (Subtype fun …
                                -/
    rfl
                                /-
                                  🎉 no goals
                                -/
    /-
      🎉 no goals
    -/
  invFun := galLift A K L B
  left_inv σ :=
    have := (IsFractionRing.injective A K).isDomain
          /-
            A : Type u_1
            K : Type u_2
            L : Type u_3
            B : Type u_4
            inst✝¹³ : CommRing A
            inst✝¹² : CommRing B
            inst✝¹¹ : Algebra A B
            inst✝¹⁰ : Field K
            inst✝⁹ : Field L
            inst✝⁸ : Algebra A K
            inst✝⁷ : IsFractionRing A K
            inst✝⁶ : Algebra B L
            inst✝⁵ : Algebra K L
            inst✝⁴ : Algebra A L
            inst✝³ : IsScalarTower A B L
            inst✝² : IsScalarTower A K L
            inst✝¹ : IsIntegralClosure B A L
            inst✝ : Algebra.IsAlgebraic K L
            σ : AlgHom A B B
            this✝ : IsDomain A
            this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
            x : B
            ⊢ Eq ((algebraMap B L) (((fun f => (↑(IsIntegralClosure.equiv A (Subtype fun x …
          -/
    have := IsIntegralClosure.isLocalization A K L B
          /-
            🎉 no goals
          -/
    AlgHom.coe_ringHom_injective <| IsLocalization.ringHom_ext (Algebra.algebraMapSubmonoid B A⁰)
      <| RingHom.ext fun x ↦ by simp [Subalgebra.algebraMap_eq, AlgHom.restrictDomain, galLift]
  right_inv σ :=
    have := (IsFractionRing.injective A K).isDomain
    have := IsIntegralClosure.isLocalization A K L B
    AlgHom.ext fun x ↦ IsIntegralClosure.algebraMap_injective B A L
      (by simp [AlgHom.restrictDomain, Subalgebra.algebraMap_eq, galLift])


@[simp]
lemma algebraMap_galRestrictHom_apply (σ : L →ₐ[K] L) (x : B) :
    algebraMap B L (galRestrictHom A K L B σ x) = σ (algebraMap B L x) := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹³ : CommRing A
    inst✝¹² : CommRing B
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra A K
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra A L
    inst✝³ : IsScalarTower A B L
    inst✝² : IsScalarTower A K L
    inst✝¹ : IsIntegralClosure B A L
    inst✝ : Algebra.IsAlgebraic K L
    σ : AlgHom K L L
    x : B
    ⊢ Eq ((algebraMap B L) (((galRestrictHom A K L B) σ) x)) (σ ((algebraMap B L)  …
  -/
  simp [galRestrictHom, Subalgebra.algebraMap_eq, AlgHom.restrictDomain]
  /-
    🎉 no goals
  -/


@[simp, nolint unusedHavesSuffices] -- false positive from unfolding galRestrictHom
lemma galRestrictHom_symm_algebraMap_apply (σ : B →ₐ[A] B) (x : B) :
    (galRestrictHom A K L B).symm σ (algebraMap B L x) = algebraMap B L (σ x) := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹³ : CommRing A
    inst✝¹² : CommRing B
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra A K
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra A L
    inst✝³ : IsScalarTower A B L
    inst✝² : IsScalarTower A K L
    inst✝¹ : IsIntegralClosure B A L
    inst✝ : Algebra.IsAlgebraic K L
    σ : AlgHom A B B
    x : B
    ⊢ Eq (((galRestrictHom A K L B).symm σ) ((algebraMap B L) x)) ((algebraMap B L …
  -/
  have := (IsFractionRing.injective A K).isDomain
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹³ : CommRing A
    inst✝¹² : CommRing B
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra A K
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra A L
    inst✝³ : IsScalarTower A B L
    inst✝² : IsScalarTower A K L
    inst✝¹ : IsIntegralClosure B A L
    inst✝ : Algebra.IsAlgebraic K L
    σ : AlgHom A B B
    x : B
    this : IsDomain A
    ⊢ Eq (((galRestrictHom A K L B).symm σ) ((algebraMap B L) x)) ((algebraMap B L …
  -/
  have := IsIntegralClosure.isLocalization A K L B
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹³ : CommRing A
    inst✝¹² : CommRing B
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra A K
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : Algebra B L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra A L
    inst✝³ : IsScalarTower A B L
    inst✝² : IsScalarTower A K L
    inst✝¹ : IsIntegralClosure B A L
    inst✝ : Algebra.IsAlgebraic K L
    σ : AlgHom A B B
    x : B
    this✝ : IsDomain A
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    ⊢ Eq (((galRestrictHom A K L B).symm σ) ((algebraMap B L) x)) ((algebraMap B L …
  -/
  simp [galRestrictHom, galLift, Subalgebra.algebraMap_eq]
  /-
    🎉 no goals
  -/


/-- The restriction `Aut(L/K) → Aut(B/A)` in an AKLB setup. -/
noncomputable
def galRestrict : (L ≃ₐ[K] L) ≃* (B ≃ₐ[A] B) :=
  (AlgEquiv.algHomUnitsEquiv K L).symm.trans
    ((Units.mapEquiv <| galRestrictHom A K L B).trans (AlgEquiv.algHomUnitsEquiv A B))


lemma coe_galRestrict_apply (σ : L ≃ₐ[K] L) :
    (galRestrict A K L B σ : B →ₐ[A] B) = galRestrictHom A K L B σ := rfl


lemma galRestrict_apply (σ : L ≃ₐ[K] L) (x : B) :
    galRestrict A K L B σ x = galRestrictHom A K L B σ x := rfl


lemma algebraMap_galRestrict_apply (σ : L ≃ₐ[K] L) (x : B) :
    algebraMap B L (galRestrict A K L B σ x) = σ (algebraMap B L x) :=
  algebraMap_galRestrictHom_apply A K L B σ.toAlgHom x


lemma prod_galRestrict_eq_norm [IsGalois K L] [IsIntegrallyClosed A] (x : B) :
    (∏ σ : L ≃ₐ[K] L, galRestrict A K L B σ x) =
    algebraMap A B (IsIntegralClosure.mk' (R := A) A (Algebra.norm K <| algebraMap B L x)
      (Algebra.isIntegral_norm K (IsIntegralClosure.isIntegral A L x).algebraMap)) := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : CommRing B
    inst✝¹³ : Algebra A B
    inst✝¹² : Field K
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    inst✝⁸ : Algebra B L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsScalarTower A K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : FiniteDimensional K L
    inst✝¹ : IsGalois K L
    inst✝ : IsIntegrallyClosed A
    x : B
    ⊢ Eq (Finset.univ.prod fun σ => ((galRestrict A K L B) σ) x) ((algebraMap A B) …
  -/
  apply IsIntegralClosure.algebraMap_injective B A L
  /-
    case a
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : CommRing B
    inst✝¹³ : Algebra A B
    inst✝¹² : Field K
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    inst✝⁸ : Algebra B L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsScalarTower A K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : FiniteDimensional K L
    inst✝¹ : IsGalois K L
    inst✝ : IsIntegrallyClosed A
    x : B
    ⊢ Eq ((algebraMap B L) (Finset.univ.prod fun σ => ((galRestrict A K L B) σ) x) …
  -/
  rw [← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_eq A K L]
  simp only [map_prod, algebraMap_galRestrict_apply, IsIntegralClosure.algebraMap_mk',
    Algebra.norm_eq_prod_automorphisms, AlgHom.coe_coe, RingHom.coe_comp, Function.comp_apply]


noncomputable
instance (priority := 900) [IsDomain A] [IsDomain B] [IsIntegrallyClosed B]
    [Module.Finite A B] [NoZeroSMulDivisors A B] : Fintype (B ≃ₐ[A] B) :=
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (Algebra.algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  Fintype.ofEquiv _ (galRestrict A (FractionRing A) (FractionRing B) B).toEquiv


/-- The restriction of the trace on `L/K` restricted onto `B/A` in an AKLB setup.
See `Algebra.intTrace` instead. -/
noncomputable
def Algebra.intTraceAux [IsIntegrallyClosed A] :
    B →ₗ[A] A :=
  (IsIntegralClosure.equiv A (integralClosure A K) K A).toLinearMap.comp
    ((((Algebra.trace K L).restrictScalars A).comp
      (IsScalarTower.toAlgHom A B L).toLinearMap).codRestrict
        (Subalgebra.toSubmodule <| integralClosure A K) (fun x ↦ isIntegral_trace
          (IsIntegral.algebraMap (IsIntegralClosure.isIntegral A L x))))


lemma Algebra.map_intTraceAux [IsIntegrallyClosed A] (x : B) :
    algebraMap A K (Algebra.intTraceAux A K L B x) = Algebra.trace K L (algebraMap B L x) :=
  IsIntegralClosure.algebraMap_equiv A (integralClosure A K) K A _


/-- The trace of a finite extension of integrally closed domains `B/A` is the restriction of
the trace on `Frac(B)/Frac(A)` onto `B/A`. See `Algebra.algebraMap_intTrace`. -/
noncomputable
def Algebra.intTrace : B →ₗ[A] A :=
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  Algebra.intTraceAux A (FractionRing A) (FractionRing B) B


lemma Algebra.algebraMap_intTrace (x : B) :
    algebraMap A K (Algebra.intTrace A B x) = Algebra.trace K L (algebraMap B L x) := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : IsFractionRing A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsIntegrallyClosed A
    inst✝³ : IsDomain B
    inst✝² : IsIntegrallyClosed B
    inst✝¹ : Module.Finite A B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A K) ((Algebra.intTrace A B) x)) ((Algebra.trace K L) ((alge …
  -/
  haveI := IsIntegralClosure.isFractionRing_of_finite_extension A K L B
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : IsFractionRing A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsIntegrallyClosed A
    inst✝³ : IsDomain B
    inst✝² : IsIntegrallyClosed B
    inst✝¹ : Module.Finite A B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    this✝² : IsIntegralClosure B A (FractionRing B)
    this✝¹ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (F …
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : IsFractionRing B L
    ⊢ Eq ((algebraMap A K) ((Algebra.intTrace A B) x)) ((Algebra.trace K L) ((alge …
  -/
  apply (FractionRing.algEquiv A K).symm.injective
  rw [AlgEquiv.commutes, Algebra.intTrace, Algebra.map_intTraceAux,
    ← AlgEquiv.commutes (FractionRing.algEquiv B L)]
  apply Algebra.trace_eq_of_equiv_equiv (FractionRing.algEquiv A K).toRingEquiv
    (FractionRing.algEquiv B L).toRingEquiv
  /-
    case a.he
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    inst✝¹⁶ : Field K
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : IsFractionRing A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsIntegrallyClosed A
    inst✝³ : IsDomain B
    inst✝² : IsIntegrallyClosed B
    inst✝¹ : Module.Finite A B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    this✝² : IsIntegralClosure B A (FractionRing B)
    this✝¹ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (F …
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : IsFractionRing B L
    ⊢ Eq ((algebraMap K L).comp ↑(FractionRing.algEquiv A K).toRingEquiv) ((↑(Frac …
  -/
  apply IsLocalization.ringHom_ext A⁰
  simp only [AlgEquiv.toRingEquiv_eq_coe, ← AlgEquiv.coe_ringHom_commutes, RingHom.comp_assoc,
    AlgHom.comp_algebraMap_of_tower, ← IsScalarTower.algebraMap_eq, RingHom.comp_assoc]


lemma Algebra.algebraMap_intTrace_fractionRing (x : B) :
    algebraMap A (FractionRing A) (Algebra.intTrace A B x) =
      Algebra.trace (FractionRing A) (FractionRing B) (algebraMap B _ x) := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra A B
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsIntegrallyClosed A
    inst✝³ : IsDomain B
    inst✝² : IsIntegrallyClosed B
    inst✝¹ : Module.Finite A B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A (FractionRing A)) ((Algebra.intTrace A B) x)) ((Algebra.tr …
  -/
  exact Algebra.map_intTraceAux x
  /-
    🎉 no goals
  -/


lemma Algebra.intTrace_eq_trace [Module.Free A B] : Algebra.intTrace A B = Algebra.trace A B := by
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Module.Free A B
    ⊢ Eq (Algebra.intTrace A B) (Algebra.trace A B)
  -/
  ext x
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  /-
    case h
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Module.Free A B
    x : B
    this✝ : IsIntegralClosure B A (FractionRing B)
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fra …
    ⊢ Eq ((Algebra.intTrace A B) x) ((Algebra.trace A B) x)
  -/
  apply IsFractionRing.injective A (FractionRing A)
  /-
    case h.a
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Module.Free A B
    x : B
    this✝ : IsIntegralClosure B A (FractionRing B)
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fra …
    ⊢ Eq ((algebraMap A (FractionRing A)) ((Algebra.intTrace A B) x)) ((algebraMap …
  -/
  rw [Algebra.algebraMap_intTrace_fractionRing, Algebra.trace_localization A A⁰]
  /-
    🎉 no goals
  -/


include M in
lemma Algebra.intTrace_eq_of_isLocalization
    (x : B) :
    algebraMap A Aₘ (Algebra.intTrace A B x) = Algebra.intTrace Aₘ Bₘ (algebraMap B Bₘ x) := by
  /-
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  by_cases hM : 0 ∈ M
    /-
      case pos
      A : Type u_1
      B : Type u_4
      inst✝²⁴ : CommRing A
      inst✝²³ : CommRing B
      inst✝²² : Algebra A B
      Aₘ : Type u_5
      Bₘ : Type u_6
      inst✝²¹ : CommRing Aₘ
      inst✝²⁰ : CommRing Bₘ
      inst✝¹⁹ : Algebra Aₘ Bₘ
      inst✝¹⁸ : Algebra A Aₘ
      inst✝¹⁷ : Algebra B Bₘ
      inst✝¹⁶ : Algebra A Bₘ
      inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
      inst✝¹⁴ : IsScalarTower A B Bₘ
      M : Submonoid A
      inst✝¹³ : IsLocalization M Aₘ
      inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
      inst✝¹¹ : IsDomain A
      inst✝¹⁰ : IsIntegrallyClosed A
      inst✝⁹ : IsDomain B
      inst✝⁸ : IsIntegrallyClosed B
      inst✝⁷ : Module.Finite A B
      inst✝⁶ : NoZeroSMulDivisors A B
      inst✝⁵ : IsDomain Aₘ
      inst✝⁴ : IsIntegrallyClosed Aₘ
      inst✝³ : IsDomain Bₘ
      inst✝² : IsIntegrallyClosed Bₘ
      inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
      inst✝ : Module.Finite Aₘ Bₘ
      x : B
      hM : Membership.mem M 0
      ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
    -/
  · subsingleton [IsLocalization.uniqueOfZeroMem (S := Aₘ) hM]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : Not (Membership.mem M 0)
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  replace hM : M ≤ A⁰ := fun x hx ↦ mem_nonZeroDivisors_iff_ne_zero.mpr (fun e ↦ hM (e ▸ hx))
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  let K := FractionRing A
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  let L := FractionRing B
  have : IsIntegralClosure B A L :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  have : IsLocalization (algebraMapSubmonoid B A⁰) L :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝ : IsIntegralClosure B A L
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  let f : Aₘ →+* K := IsLocalization.map _ (T := A⁰) (RingHom.id A) hM
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝ : IsIntegralClosure B A L
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  letI := f.toAlgebra
  have : IsScalarTower A Aₘ K := IsScalarTower.of_algebraMap_eq'
    (by rw [RingHom.algebraMap_toAlgebra, IsLocalization.map_comp, RingHomCompTriple.comp_eq])
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝² : IsIntegralClosure B A L
    this✝¹ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝ : Algebra Aₘ K := f.toAlgebra
    this : IsScalarTower A Aₘ K
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  letI := IsFractionRing.isFractionRing_of_isDomain_of_isLocalization M Aₘ K
  let g : Bₘ →+* L := IsLocalization.map _
      (M := algebraMapSubmonoid B M) (T := algebraMapSubmonoid B A⁰)
      (RingHom.id B) (Submonoid.monotone_map hM)
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝³ : IsIntegralClosure B A L
    this✝² : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝¹ : Algebra Aₘ K := f.toAlgebra
    this✝ : IsScalarTower A Aₘ K
    this : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_isL …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  letI := g.toAlgebra
  have : IsScalarTower B Bₘ L := IsScalarTower.of_algebraMap_eq'
    (by rw [RingHom.algebraMap_toAlgebra, IsLocalization.map_comp, RingHomCompTriple.comp_eq])
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝⁵ : IsIntegralClosure B A L
    this✝⁴ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝³ : Algebra Aₘ K := f.toAlgebra
    this✝² : IsScalarTower A Aₘ K
    this✝¹ : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝ : Algebra Bₘ L := g.toAlgebra
    this : IsScalarTower B Bₘ L
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  letI := ((algebraMap K L).comp f).toAlgebra
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝⁶ : IsIntegralClosure B A L
    this✝⁵ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝⁴ : Algebra Aₘ K := f.toAlgebra
    this✝³ : IsScalarTower A Aₘ K
    this✝² : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝¹ : Algebra Bₘ L := g.toAlgebra
    this✝ : IsScalarTower B Bₘ L
    this : Algebra Aₘ L := ((algebraMap K L).comp f).toAlgebra
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  have : IsScalarTower Aₘ K L := IsScalarTower.of_algebraMap_eq' rfl
  have : IsScalarTower Aₘ Bₘ L := by
    apply IsScalarTower.of_algebraMap_eq'
    apply IsLocalization.ringHom_ext M
    rw [RingHom.algebraMap_toAlgebra, RingHom.algebraMap_toAlgebra (R := Bₘ), RingHom.comp_assoc,
      RingHom.comp_assoc, ← IsScalarTower.algebraMap_eq, IsScalarTower.algebraMap_eq A B Bₘ,
      IsLocalization.map_comp, RingHom.comp_id, ← RingHom.comp_assoc, IsLocalization.map_comp,
      RingHom.comp_id, ← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq]
  letI := IsFractionRing.isFractionRing_of_isDomain_of_isLocalization
    (algebraMapSubmonoid B M) Bₘ L
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝⁹ : IsIntegralClosure B A L
    this✝⁸ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝⁷ : Algebra Aₘ K := f.toAlgebra
    this✝⁶ : IsScalarTower A Aₘ K
    this✝⁵ : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝⁴ : Algebra Bₘ L := g.toAlgebra
    this✝³ : IsScalarTower B Bₘ L
    this✝² : Algebra Aₘ L := ((algebraMap K L).comp f).toAlgebra
    this✝¹ : IsScalarTower Aₘ K L
    this✝ : IsScalarTower Aₘ Bₘ L
    this : IsFractionRing Bₘ L := IsFractionRing.isFractionRing_of_isDomain_of_isL …
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  have : FiniteDimensional K L := Module.Finite_of_isLocalization A B _ _ A⁰
  have : IsIntegralClosure Bₘ Aₘ L :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁴ : CommRing A
    inst✝²³ : CommRing B
    inst✝²² : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²¹ : CommRing Aₘ
    inst✝²⁰ : CommRing Bₘ
    inst✝¹⁹ : Algebra Aₘ Bₘ
    inst✝¹⁸ : Algebra A Aₘ
    inst✝¹⁷ : Algebra B Bₘ
    inst✝¹⁶ : Algebra A Bₘ
    inst✝¹⁵ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁴ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹³ : IsLocalization M Aₘ
    inst✝¹² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsIntegrallyClosed A
    inst✝⁹ : IsDomain B
    inst✝⁸ : IsIntegrallyClosed B
    inst✝⁷ : Module.Finite A B
    inst✝⁶ : NoZeroSMulDivisors A B
    inst✝⁵ : IsDomain Aₘ
    inst✝⁴ : IsIntegrallyClosed Aₘ
    inst✝³ : IsDomain Bₘ
    inst✝² : IsIntegrallyClosed Bₘ
    inst✝¹ : NoZeroSMulDivisors Aₘ Bₘ
    inst✝ : Module.Finite Aₘ Bₘ
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝¹¹ : IsIntegralClosure B A L
    this✝¹⁰ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝⁹ : Algebra Aₘ K := f.toAlgebra
    this✝⁸ : IsScalarTower A Aₘ K
    this✝⁷ : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝⁶ : Algebra Bₘ L := g.toAlgebra
    this✝⁵ : IsScalarTower B Bₘ L
    this✝⁴ : Algebra Aₘ L := ((algebraMap K L).comp f).toAlgebra
    this✝³ : IsScalarTower Aₘ K L
    this✝² : IsScalarTower Aₘ Bₘ L
    this✝¹ : IsFractionRing Bₘ L := IsFractionRing.isFractionRing_of_isDomain_of_i …
    this✝ : FiniteDimensional K L
    this : IsIntegralClosure Bₘ Aₘ L
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intTrace A B) x)) ((Algebra.intTrace Aₘ Bₘ)  …
  -/
  apply IsFractionRing.injective Aₘ K
  rw [← IsScalarTower.algebraMap_apply, Algebra.algebraMap_intTrace_fractionRing,
    Algebra.algebraMap_intTrace (L := L), ← IsScalarTower.algebraMap_apply]


/-- The restriction of the norm on `L/K` restricted onto `B/A` in an AKLB setup.
See `Algebra.intNorm` instead. -/
noncomputable
def Algebra.intNormAux [Algebra.IsSeparable K L] :
    B →* A where
  toFun := fun s ↦ IsIntegralClosure.mk' (R := A) A (Algebra.norm K (algebraMap B L s))
    (isIntegral_norm K <| IsIntegral.map (IsScalarTower.toAlgHom A B L)
      (IsIntegralClosure.isIntegral A L s))
                 /-
                   A : Type u_1
                   K : Type u_2
                   L : Type u_3
                   B : Type u_4
                   inst✝²⁵ : CommRing A
                   inst✝²⁴ : CommRing B
                   inst✝²³ : Algebra A B
                   inst✝²² : Field K
                   inst✝²¹ : Field L
                   inst✝²⁰ : Algebra A K
                   inst✝¹⁹ : IsFractionRing A K
                   inst✝¹⁸ : Algebra B L
                   inst✝¹⁷ : Algebra K L
                   inst✝¹⁶ : Algebra A L
                   inst✝¹⁵ : IsScalarTower A B L
                   inst✝¹⁴ : IsScalarTower A K L
                   inst✝¹³ : IsIntegralClosure B A L
                   inst✝¹² : FiniteDimensional K L
                   Aₘ : Type ?u.297971
                   Bₘ : Type ?u.297974
                   inst✝¹¹ : CommRing Aₘ
                   inst✝¹⁰ : CommRing Bₘ
                   inst✝⁹ : Algebra Aₘ Bₘ
                   inst✝⁸ : Algebra A Aₘ
                   inst✝⁷ : Algebra B Bₘ
                   inst✝⁶ : Algebra A Bₘ
                   inst✝⁵ : IsScalarTower A Aₘ Bₘ
                   inst✝⁴ : IsScalarTower A B Bₘ
                   M : Submonoid A
                   inst✝³ : IsLocalization M Aₘ
                   inst✝² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
                   inst✝¹ : IsIntegrallyClosed A
                   inst✝ : Algebra.IsSeparable K L
                   ⊢ Eq ((fun s => IsIntegralClosure.mk' A ((Algebra.norm K) ((algebraMap B L) s) …
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
                           /-
                             A : Type u_1
                             K : Type u_2
                             L : Type u_3
                             B : Type u_4
                             inst✝²⁵ : CommRing A
                             inst✝²⁴ : CommRing B
                             inst✝²³ : Algebra A B
                             inst✝²² : Field K
                             inst✝²¹ : Field L
                             inst✝²⁰ : Algebra A K
                             inst✝¹⁹ : IsFractionRing A K
                             inst✝¹⁸ : Algebra B L
                             inst✝¹⁷ : Algebra K L
                             inst✝¹⁶ : Algebra A L
                             inst✝¹⁵ : IsScalarTower A B L
                             inst✝¹⁴ : IsScalarTower A K L
                             inst✝¹³ : IsIntegralClosure B A L
                             inst✝¹² : FiniteDimensional K L
                             Aₘ : Type ?u.297971
                             Bₘ : Type ?u.297974
                             inst✝¹¹ : CommRing Aₘ
                             inst✝¹⁰ : CommRing Bₘ
                             inst✝⁹ : Algebra Aₘ Bₘ
                             inst✝⁸ : Algebra A Aₘ
                             inst✝⁷ : Algebra B Bₘ
                             inst✝⁶ : Algebra A Bₘ
                             inst✝⁵ : IsScalarTower A Aₘ Bₘ
                             inst✝⁴ : IsScalarTower A B Bₘ
                             M : Submonoid A
                             inst✝³ : IsLocalization M Aₘ
                             inst✝² : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
                             inst✝¹ : IsIntegrallyClosed A
                             inst✝ : Algebra.IsSeparable K L
                             x y : B
                             ⊢ Eq ({ toFun := fun s => IsIntegralClosure.mk' A ((Algebra.norm K) ((algebraM …
                           -/
  map_mul' := fun x y ↦ by simpa using IsIntegralClosure.mk'_mul _ _ _ _ _
                           /-
                             🎉 no goals
                           -/


lemma Algebra.map_intNormAux [Algebra.IsSeparable K L] (x : B) :
    algebraMap A K (Algebra.intNormAux A K L B x) = Algebra.norm K (algebraMap B L x) := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : CommRing B
    inst✝¹³ : Algebra A B
    inst✝¹² : Field K
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    inst✝⁸ : Algebra B L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsScalarTower A K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : FiniteDimensional K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : Algebra.IsSeparable K L
    x : B
    ⊢ Eq ((algebraMap A K) ((Algebra.intNormAux A K L B) x)) ((Algebra.norm K) ((a …
  -/
  dsimp [Algebra.intNormAux]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : CommRing B
    inst✝¹³ : Algebra A B
    inst✝¹² : Field K
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : IsFractionRing A K
    inst✝⁸ : Algebra B L
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsScalarTower A K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : FiniteDimensional K L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : Algebra.IsSeparable K L
    x : B
    ⊢ Eq ((algebraMap A K) (IsIntegralClosure.mk' A ((Algebra.norm K) ((algebraMap …
  -/
  exact IsIntegralClosure.algebraMap_mk' _ _ _
  /-
    🎉 no goals
  -/


/-- The norm of a finite extension of integrally closed domains `B/A` is the restriction of
the norm on `Frac(B)/Frac(A)` onto `B/A`. See `Algebra.algebraMap_intNorm`. -/
noncomputable
def Algebra.intNorm : B →* A :=
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  Algebra.intNormAux A (FractionRing A) (FractionRing B) B


lemma Algebra.algebraMap_intNorm (x : B) :
    algebraMap A K (Algebra.intNorm A B x) = Algebra.norm K (algebraMap B L x) := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝²⁰ : CommRing A
    inst✝¹⁹ : CommRing B
    inst✝¹⁸ : Algebra A B
    inst✝¹⁷ : Field K
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A B L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsIntegralClosure B A L
    inst✝⁷ : FiniteDimensional K L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A K) ((Algebra.intNorm A B) x)) ((Algebra.norm K) ((algebraM …
  -/
  haveI := IsIntegralClosure.isFractionRing_of_finite_extension A K L B
  /-
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝²⁰ : CommRing A
    inst✝¹⁹ : CommRing B
    inst✝¹⁸ : Algebra A B
    inst✝¹⁷ : Field K
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A B L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsIntegralClosure B A L
    inst✝⁷ : FiniteDimensional K L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    x : B
    this✝² : IsIntegralClosure B A (FractionRing B)
    this✝¹ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (F …
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : IsFractionRing B L
    ⊢ Eq ((algebraMap A K) ((Algebra.intNorm A B) x)) ((Algebra.norm K) ((algebraM …
  -/
  apply (FractionRing.algEquiv A K).symm.injective
  rw [AlgEquiv.commutes, Algebra.intNorm, Algebra.map_intNormAux,
    ← AlgEquiv.commutes (FractionRing.algEquiv B L)]
  apply Algebra.norm_eq_of_equiv_equiv (FractionRing.algEquiv A K).toRingEquiv
    (FractionRing.algEquiv B L).toRingEquiv
  /-
    case a.he
    A : Type u_1
    K : Type u_2
    L : Type u_3
    B : Type u_4
    inst✝²⁰ : CommRing A
    inst✝¹⁹ : CommRing B
    inst✝¹⁸ : Algebra A B
    inst✝¹⁷ : Field K
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : IsFractionRing A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A B L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsIntegralClosure B A L
    inst✝⁷ : FiniteDimensional K L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    x : B
    this✝² : IsIntegralClosure B A (FractionRing B)
    this✝¹ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (F …
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : IsFractionRing B L
    ⊢ Eq ((algebraMap K L).comp ↑(FractionRing.algEquiv A K).toRingEquiv) ((↑(Frac …
  -/
  apply IsLocalization.ringHom_ext A⁰
  simp only [AlgEquiv.toRingEquiv_eq_coe, ← AlgEquiv.coe_ringHom_commutes, RingHom.comp_assoc,
    AlgHom.comp_algebraMap_of_tower, ← IsScalarTower.algebraMap_eq, RingHom.comp_assoc]


@[simp]
lemma Algebra.algebraMap_intNorm_fractionRing (x : B) :
    algebraMap A (FractionRing A) (Algebra.intNorm A B x) =
      Algebra.norm (FractionRing A) (algebraMap B (FractionRing B) x) := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A (FractionRing A)) ((Algebra.intNorm A B) x)) ((Algebra.nor …
  -/
  exact Algebra.map_intNormAux x
  /-
    🎉 no goals
  -/


lemma Algebra.intNorm_eq_norm [Module.Free A B] : Algebra.intNorm A B = Algebra.norm A := by
  /-
    A : Type u_1
    B : Type u_4
    inst✝¹⁰ : CommRing A
    inst✝⁹ : CommRing B
    inst✝⁸ : Algebra A B
    inst✝⁷ : IsIntegrallyClosed A
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsDomain B
    inst✝⁴ : IsIntegrallyClosed B
    inst✝³ : Module.Finite A B
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝ : Module.Free A B
    ⊢ Eq (Algebra.intNorm A B) (Algebra.norm A)
  -/
  ext x
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  /-
    case h
    A : Type u_1
    B : Type u_4
    inst✝¹⁰ : CommRing A
    inst✝⁹ : CommRing B
    inst✝⁸ : Algebra A B
    inst✝⁷ : IsIntegrallyClosed A
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsDomain B
    inst✝⁴ : IsIntegrallyClosed B
    inst✝³ : Module.Finite A B
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝ : Module.Free A B
    x : B
    this✝ : IsIntegralClosure B A (FractionRing B)
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fra …
    ⊢ Eq ((Algebra.intNorm A B) x) ((Algebra.norm A) x)
  -/
  apply IsFractionRing.injective A (FractionRing A)
  /-
    case h.a
    A : Type u_1
    B : Type u_4
    inst✝¹⁰ : CommRing A
    inst✝⁹ : CommRing B
    inst✝⁸ : Algebra A B
    inst✝⁷ : IsIntegrallyClosed A
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsDomain B
    inst✝⁴ : IsIntegrallyClosed B
    inst✝³ : Module.Finite A B
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝ : Module.Free A B
    x : B
    this✝ : IsIntegralClosure B A (FractionRing B)
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fra …
    ⊢ Eq ((algebraMap A (FractionRing A)) ((Algebra.intNorm A B) x)) ((algebraMap  …
  -/
  rw [Algebra.algebraMap_intNorm_fractionRing, Algebra.norm_localization A A⁰]
  /-
    🎉 no goals
  -/


@[simp]
lemma Algebra.intNorm_zero : Algebra.intNorm A B 0 = 0 := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((Algebra.intNorm A B) 0) 0
  -/
  apply IsFractionRing.injective A (FractionRing A)
  /-
    case a
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A (FractionRing A)) ((Algebra.intNorm A B) 0)) ((algebraMap  …
  -/
  simp only [algebraMap_intNorm_fractionRing, map_zero, norm_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma Algebra.intNorm_eq_zero {x : B} : Algebra.intNorm A B x = 0 ↔ x = 0 := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  rw [← (IsFractionRing.injective A (FractionRing A)).eq_iff,
    ← (IsFractionRing.injective B (FractionRing B)).eq_iff]
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDomain A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Iff (Eq ((algebraMap A (FractionRing A)) ((Algebra.intNorm A B) x)) ((algebr …
  -/
  simp only [algebraMap_intNorm_fractionRing, map_zero, norm_eq_zero_iff]
  /-
    🎉 no goals
  -/


                                                                                /-
                                                                                  A : Type u_1
                                                                                  B : Type u_4
                                                                                  inst✝⁹ : CommRing A
                                                                                  inst✝⁸ : CommRing B
                                                                                  inst✝⁷ : Algebra A B
                                                                                  inst✝⁶ : IsIntegrallyClosed A
                                                                                  inst✝⁵ : IsDomain A
                                                                                  inst✝⁴ : IsDomain B
                                                                                  inst✝³ : IsIntegrallyClosed B
                                                                                  inst✝² : Module.Finite A B
                                                                                  inst✝¹ : NoZeroSMulDivisors A B
                                                                                  inst✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
                                                                                  x : B
                                                                                  ⊢ Iff (Ne ((Algebra.intNorm A B) x) 0) (Ne x 0)
                                                                                -/
lemma Algebra.intNorm_ne_zero {x : B} : Algebra.intNorm A B x ≠ 0 ↔ x ≠ 0 := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


include M in
lemma Algebra.intNorm_eq_of_isLocalization (x : B) :
    algebraMap A Aₘ (Algebra.intNorm A B x) = Algebra.intNorm Aₘ Bₘ (algebraMap B Bₘ x) := by
  /-
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  by_cases hM : 0 ∈ M
    /-
      case pos
      A : Type u_1
      B : Type u_4
      inst✝²⁶ : CommRing A
      inst✝²⁵ : CommRing B
      inst✝²⁴ : Algebra A B
      Aₘ : Type u_5
      Bₘ : Type u_6
      inst✝²³ : CommRing Aₘ
      inst✝²² : CommRing Bₘ
      inst✝²¹ : Algebra Aₘ Bₘ
      inst✝²⁰ : Algebra A Aₘ
      inst✝¹⁹ : Algebra B Bₘ
      inst✝¹⁸ : Algebra A Bₘ
      inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
      inst✝¹⁶ : IsScalarTower A B Bₘ
      M : Submonoid A
      inst✝¹⁵ : IsLocalization M Aₘ
      inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
      inst✝¹³ : IsIntegrallyClosed A
      inst✝¹² : IsDomain A
      inst✝¹¹ : IsDomain B
      inst✝¹⁰ : IsIntegrallyClosed B
      inst✝⁹ : Module.Finite A B
      inst✝⁸ : NoZeroSMulDivisors A B
      inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
      inst✝⁶ : IsDomain Aₘ
      inst✝⁵ : IsIntegrallyClosed Aₘ
      inst✝⁴ : IsDomain Bₘ
      inst✝³ : IsIntegrallyClosed Bₘ
      inst✝² : NoZeroSMulDivisors Aₘ Bₘ
      inst✝¹ : Module.Finite Aₘ Bₘ
      inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
      x : B
      hM : Membership.mem M 0
      ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
    -/
  · subsingleton [IsLocalization.uniqueOfZeroMem (S := Aₘ) hM]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : Not (Membership.mem M 0)
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  replace hM : M ≤ A⁰ := fun x hx ↦ mem_nonZeroDivisors_iff_ne_zero.mpr (fun e ↦ hM (e ▸ hx))
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  let K := FractionRing A
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  let L := FractionRing B
  have : IsIntegralClosure B A L :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  have : IsLocalization (algebraMapSubmonoid B A⁰) L :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝ : IsIntegralClosure B A L
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  let f : Aₘ →+* K := IsLocalization.map _ (T := A⁰) (RingHom.id A) hM
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝ : IsIntegralClosure B A L
    this : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  letI := f.toAlgebra
  have : IsScalarTower A Aₘ K := IsScalarTower.of_algebraMap_eq'
    (by rw [RingHom.algebraMap_toAlgebra, IsLocalization.map_comp, RingHomCompTriple.comp_eq])
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝² : IsIntegralClosure B A L
    this✝¹ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝ : Algebra Aₘ K := f.toAlgebra
    this : IsScalarTower A Aₘ K
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  letI := IsFractionRing.isFractionRing_of_isDomain_of_isLocalization M Aₘ K
  let g : Bₘ →+* L := IsLocalization.map _
      (M := algebraMapSubmonoid B M) (T := algebraMapSubmonoid B A⁰)
      (RingHom.id B) (Submonoid.monotone_map hM)
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝³ : IsIntegralClosure B A L
    this✝² : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝¹ : Algebra Aₘ K := f.toAlgebra
    this✝ : IsScalarTower A Aₘ K
    this : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_isL …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  letI := g.toAlgebra
  have : IsScalarTower B Bₘ L := IsScalarTower.of_algebraMap_eq'
    (by rw [RingHom.algebraMap_toAlgebra, IsLocalization.map_comp, RingHomCompTriple.comp_eq])
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝⁵ : IsIntegralClosure B A L
    this✝⁴ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝³ : Algebra Aₘ K := f.toAlgebra
    this✝² : IsScalarTower A Aₘ K
    this✝¹ : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝ : Algebra Bₘ L := g.toAlgebra
    this : IsScalarTower B Bₘ L
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  letI := ((algebraMap K L).comp f).toAlgebra
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝⁶ : IsIntegralClosure B A L
    this✝⁵ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝⁴ : Algebra Aₘ K := f.toAlgebra
    this✝³ : IsScalarTower A Aₘ K
    this✝² : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝¹ : Algebra Bₘ L := g.toAlgebra
    this✝ : IsScalarTower B Bₘ L
    this : Algebra Aₘ L := ((algebraMap K L).comp f).toAlgebra
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  have : IsScalarTower Aₘ K L := IsScalarTower.of_algebraMap_eq' rfl
  have : IsScalarTower Aₘ Bₘ L := by
    apply IsScalarTower.of_algebraMap_eq'
    apply IsLocalization.ringHom_ext M
    rw [RingHom.algebraMap_toAlgebra, RingHom.algebraMap_toAlgebra (R := Bₘ), RingHom.comp_assoc,
      RingHom.comp_assoc, ← IsScalarTower.algebraMap_eq, IsScalarTower.algebraMap_eq A B Bₘ,
      IsLocalization.map_comp, RingHom.comp_id, ← RingHom.comp_assoc, IsLocalization.map_comp,
      RingHom.comp_id, ← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq]
  letI := IsFractionRing.isFractionRing_of_isDomain_of_isLocalization
    (algebraMapSubmonoid B M) Bₘ L
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝⁹ : IsIntegralClosure B A L
    this✝⁸ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝⁷ : Algebra Aₘ K := f.toAlgebra
    this✝⁶ : IsScalarTower A Aₘ K
    this✝⁵ : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝⁴ : Algebra Bₘ L := g.toAlgebra
    this✝³ : IsScalarTower B Bₘ L
    this✝² : Algebra Aₘ L := ((algebraMap K L).comp f).toAlgebra
    this✝¹ : IsScalarTower Aₘ K L
    this✝ : IsScalarTower Aₘ Bₘ L
    this : IsFractionRing Bₘ L := IsFractionRing.isFractionRing_of_isDomain_of_isL …
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  have : FiniteDimensional K L := Module.Finite_of_isLocalization A B _ _ A⁰
  have : IsIntegralClosure Bₘ Aₘ L :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  /-
    case neg
    A : Type u_1
    B : Type u_4
    inst✝²⁶ : CommRing A
    inst✝²⁵ : CommRing B
    inst✝²⁴ : Algebra A B
    Aₘ : Type u_5
    Bₘ : Type u_6
    inst✝²³ : CommRing Aₘ
    inst✝²² : CommRing Bₘ
    inst✝²¹ : Algebra Aₘ Bₘ
    inst✝²⁰ : Algebra A Aₘ
    inst✝¹⁹ : Algebra B Bₘ
    inst✝¹⁸ : Algebra A Bₘ
    inst✝¹⁷ : IsScalarTower A Aₘ Bₘ
    inst✝¹⁶ : IsScalarTower A B Bₘ
    M : Submonoid A
    inst✝¹⁵ : IsLocalization M Aₘ
    inst✝¹⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₘ
    inst✝¹³ : IsIntegrallyClosed A
    inst✝¹² : IsDomain A
    inst✝¹¹ : IsDomain B
    inst✝¹⁰ : IsIntegrallyClosed B
    inst✝⁹ : Module.Finite A B
    inst✝⁸ : NoZeroSMulDivisors A B
    inst✝⁷ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝⁶ : IsDomain Aₘ
    inst✝⁵ : IsIntegrallyClosed Aₘ
    inst✝⁴ : IsDomain Bₘ
    inst✝³ : IsIntegrallyClosed Bₘ
    inst✝² : NoZeroSMulDivisors Aₘ Bₘ
    inst✝¹ : Module.Finite Aₘ Bₘ
    inst✝ : Algebra.IsSeparable (FractionRing Aₘ) (FractionRing Bₘ)
    x : B
    hM : LE.le M (nonZeroDivisors A)
    K : Type u_1 := FractionRing A
    L : Type u_4 := FractionRing B
    this✝¹¹ : IsIntegralClosure B A L
    this✝¹⁰ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) L
    f : RingHom Aₘ K := IsLocalization.map K (RingHom.id A) hM
    this✝⁹ : Algebra Aₘ K := f.toAlgebra
    this✝⁸ : IsScalarTower A Aₘ K
    this✝⁷ : IsFractionRing Aₘ K := IsFractionRing.isFractionRing_of_isDomain_of_i …
    g : RingHom Bₘ L := IsLocalization.map L (RingHom.id B) ⋯
    this✝⁶ : Algebra Bₘ L := g.toAlgebra
    this✝⁵ : IsScalarTower B Bₘ L
    this✝⁴ : Algebra Aₘ L := ((algebraMap K L).comp f).toAlgebra
    this✝³ : IsScalarTower Aₘ K L
    this✝² : IsScalarTower Aₘ Bₘ L
    this✝¹ : IsFractionRing Bₘ L := IsFractionRing.isFractionRing_of_isDomain_of_i …
    this✝ : FiniteDimensional K L
    this : IsIntegralClosure Bₘ Aₘ L
    ⊢ Eq ((algebraMap A Aₘ) ((Algebra.intNorm A B) x)) ((Algebra.intNorm Aₘ Bₘ) (( …
  -/
  apply IsFractionRing.injective Aₘ K
  rw [← IsScalarTower.algebraMap_apply, Algebra.algebraMap_intNorm_fractionRing,
    Algebra.algebraMap_intNorm (L := L), ← IsScalarTower.algebraMap_apply]


lemma Algebra.algebraMap_intNorm_of_isGalois
    [IsDomain A] [IsIntegrallyClosed A] [IsDomain B] [IsIntegrallyClosed B]
    [Module.Finite A B] [NoZeroSMulDivisors A B] [IsGalois (FractionRing A) (FractionRing B)]
    {x : B} :
    algebraMap A B (Algebra.intNorm A B x) = ∏ σ : B ≃ₐ[A] B, σ x := by
  haveI : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  haveI : IsLocalization (Algebra.algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  haveI : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : IsGalois (FractionRing A) (FractionRing B)
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A B) ((Algebra.intNorm A B) x)) (Finset.univ.prod fun σ => σ …
  -/
  rw [← (galRestrict A (FractionRing A) (FractionRing B) B).toEquiv.prod_comp]
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : IsGalois (FractionRing A) (FractionRing B)
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A B) ((Algebra.intNorm A B) x)) (Finset.univ.prod fun i => ( …
  -/
  simp only [MulEquiv.toEquiv_eq_coe, EquivLike.coe_coe]
  /-
    A : Type u_1
    B : Type u_4
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDomain B
    inst✝³ : IsIntegrallyClosed B
    inst✝² : Module.Finite A B
    inst✝¹ : NoZeroSMulDivisors A B
    inst✝ : IsGalois (FractionRing A) (FractionRing B)
    x : B
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq ((algebraMap A B) ((Algebra.intNorm A B) x)) (Finset.univ.prod fun x_1 => …
  -/
  convert (prod_galRestrict_eq_norm A (FractionRing A) (FractionRing B) B x).symm
  /-
    🎉 no goals
  -/

