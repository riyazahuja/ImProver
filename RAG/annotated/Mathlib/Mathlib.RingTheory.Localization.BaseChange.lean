/-- The forward direction of `isLocalizedModule_iff_isBaseChange`. It is also used to prove the
other direction. -/
theorem IsLocalizedModule.isBaseChange [IsLocalizedModule S f] : IsBaseChange A f :=
  .of_lift_unique _ fun Q _ _ _ _ g ↦ by
    obtain ⟨ℓ, rfl, h₂⟩ := IsLocalizedModule.is_universal S f g fun s ↦ by
      rw [← (Algebra.lsmul R (A := A) R Q).commutes]; exact (IsLocalization.map_units A s).map _
    /-
      case intro.intro
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsLocalization S A
      M : Type u_3
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      M' : Type u_4
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      inst✝ : IsLocalizedModule S f
      Q : Type (max u_3 u_4 u_2)
      x✝³ : AddCommMonoid Q
      x✝² : Module R Q
      x✝¹ : Module A Q
      x✝ : IsScalarTower R A Q
      ℓ : LinearMap (RingHom.id R) M' Q
      h₂ : ∀ (y : LinearMap (RingHom.id R) M' Q), (fun l => Eq (l.comp f) (ℓ.comp f) …
      ⊢ ExistsUnique fun g' => Eq ((↑R g').comp f) (ℓ.comp f)
    -/
    refine ⟨ℓ.extendScalarsOfIsLocalization S A, by simp, fun g'' h ↦ ?_⟩
    /-
      case intro.intro
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      inst✝⁹ : CommSemiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : IsLocalization S A
      M : Type u_3
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      M' : Type u_4
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      inst✝² : Module A M'
      inst✝¹ : IsScalarTower R A M'
      f : LinearMap (RingHom.id R) M M'
      inst✝ : IsLocalizedModule S f
      Q : Type (max u_3 u_4 u_2)
      x✝³ : AddCommMonoid Q
      x✝² : Module R Q
      x✝¹ : Module A Q
      x✝ : IsScalarTower R A Q
      ℓ : LinearMap (RingHom.id R) M' Q
      h₂ : ∀ (y : LinearMap (RingHom.id R) M' Q), (fun l => Eq (l.comp f) (ℓ.comp f) …
      g'' : LinearMap (RingHom.id A) M' Q
      h : (fun g' => Eq ((↑R g').comp f) (ℓ.comp f)) g''
      ⊢ Eq g'' (LinearMap.extendScalarsOfIsLocalization S A ℓ)
    -/
    cases h₂ (LinearMap.restrictScalars R g'') h; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The map `(f : M →ₗ[R] M')` is a localization of modules iff the map
`(Localization S) × M → N, (s, m) ↦ s • f m` is the tensor product (insomuch as it is the universal
bilinear map).
In particular, there is an isomorphism between `LocalizedModule S M` and `(Localization S) ⊗[R] M`
given by `m/s ↦ (1/s) ⊗ₜ m`.
-/
theorem isLocalizedModule_iff_isBaseChange : IsLocalizedModule S f ↔ IsBaseChange A f := by
  /-
    R : Type u_1
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : IsLocalization S A
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_4
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    inst✝¹ : Module A M'
    inst✝ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ Iff (IsLocalizedModule S f) (IsBaseChange A f)
  -/
  refine ⟨fun _ ↦ IsLocalizedModule.isBaseChange S A f, fun h ↦ ?_⟩
  /-
    R : Type u_1
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : IsLocalization S A
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_4
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    inst✝¹ : Module A M'
    inst✝ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    h : IsBaseChange A f
    ⊢ IsLocalizedModule S f
  -/
  have : IsBaseChange A (LocalizedModule.mkLinearMap S M) := IsLocalizedModule.isBaseChange S A _
  /-
    R : Type u_1
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : IsLocalization S A
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_4
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    inst✝¹ : Module A M'
    inst✝ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    h : IsBaseChange A f
    this : IsBaseChange A (LocalizedModule.mkLinearMap S M)
    ⊢ IsLocalizedModule S f
  -/
  let e := (this.equiv.symm.trans h.equiv).restrictScalars R
  /-
    R : Type u_1
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : IsLocalization S A
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_4
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    inst✝¹ : Module A M'
    inst✝ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    h : IsBaseChange A f
    this : IsBaseChange A (LocalizedModule.mkLinearMap S M)
    e : LinearEquiv (RingHom.id R) (LocalizedModule S M) M' := LinearEquiv.restric …
    ⊢ IsLocalizedModule S f
  -/
  convert IsLocalizedModule.of_linearEquiv S (LocalizedModule.mkLinearMap S M) e
  /-
    case h.e'_10
    R : Type u_1
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : IsLocalization S A
    M : Type u_3
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M' : Type u_4
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    inst✝¹ : Module A M'
    inst✝ : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    h : IsBaseChange A f
    this : IsBaseChange A (LocalizedModule.mkLinearMap S M)
    e : LinearEquiv (RingHom.id R) (LocalizedModule S M) M' := LinearEquiv.restric …
    ⊢ Eq f ((↑e).comp (LocalizedModule.mkLinearMap S M))
  -/
  ext
  rw [LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    LinearEquiv.restrictScalars_apply, LinearEquiv.trans_apply, IsBaseChange.equiv_symm_apply,
    IsBaseChange.equiv_tmul, one_smul]


instance tensorProduct_isLocalizedModule : IsLocalizedModule S (TensorProduct.mk R A M 1) :=
  (isLocalizedModule_iff_isBaseChange _ A _).mpr (TensorProduct.isBaseChange _ _ _)


theorem tensorProduct_compatibleSMul : CompatibleSMul R A M₁ M₂ where
  smul_tmul a _ _ := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      inst✝¹⁰ : CommSemiring A
      inst✝⁹ : Algebra R A
      inst✝⁸ : IsLocalization S A
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : Module R M₁
      inst✝⁴ : Module R M₂
      inst✝³ : Module A M₁
      inst✝² : Module A M₂
      inst✝¹ : IsScalarTower R A M₁
      inst✝ : IsScalarTower R A M₂
      a : A
      x✝¹ : M₁
      x✝ : M₂
      ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul a x✝¹) x✝) (TensorProduct.tmul R x✝¹ ( …
    -/
    obtain ⟨r, s, rfl⟩ := mk'_surjective S a
    /-
      case intro.intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      inst✝¹⁰ : CommSemiring A
      inst✝⁹ : Algebra R A
      inst✝⁸ : IsLocalization S A
      M₁ : Type u_5
      M₂ : Type u_6
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : Module R M₁
      inst✝⁴ : Module R M₂
      inst✝³ : Module A M₁
      inst✝² : Module A M₂
      inst✝¹ : IsScalarTower R A M₁
      inst✝ : IsScalarTower R A M₂
      x✝¹ : M₁
      x✝ : M₂
      r : R
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul (IsLocalization.mk' A r s) x✝¹) x✝) (T …
    -/
    rw [← (map_units A s).smul_left_cancel]
    simp_rw [algebraMap_smul, smul_tmul', ← smul_assoc, smul_tmul, ← smul_assoc, smul_mk'_self,
      algebraMap_smul, smul_tmul]


/-- If `A` is a localization of `R`, tensoring two `A`-modules over `A` is the same as
tensoring them over `R`. -/
noncomputable def moduleTensorEquiv : M₁ ⊗[A] M₂ ≃ₗ[A] M₁ ⊗[R] M₂ :=
  have := tensorProduct_compatibleSMul S A M₁ M₂
  equivOfCompatibleSMul R A M₁ M₂


/-- If `A` is a localization of `R`, tensoring an `A`-module with `A` over `R` does nothing. -/
noncomputable def moduleLid : A ⊗[R] M₁ ≃ₗ[A] M₁ :=
  have := tensorProduct_compatibleSMul S A A M₁
  (equivOfCompatibleSMul R A A M₁).symm ≪≫ₗ TensorProduct.lid _ _


/-- If `A` is a localization of `R`, tensoring two `A`-algebras over `A` is the same as
tensoring them over `R`. -/
noncomputable def algebraTensorEquiv : B ⊗[A] C ≃ₐ[A] B ⊗[R] C :=
  have := tensorProduct_compatibleSMul S A B C
  Algebra.TensorProduct.equivOfCompatibleSMul R A B C


/-- If `A` is a localization of `R`, tensoring an `A`-algebra with `A` over `R` does nothing. -/
noncomputable def algebraLid : A ⊗[R] B ≃ₐ[A] B :=
  have := tensorProduct_compatibleSMul S A A B
  Algebra.TensorProduct.lidOfCompatibleSMul R A B


@[deprecated (since := "2024-12-01")] alias tensorSelfAlgEquiv := algebraLid


set_option linter.docPrime false in
theorem bijective_linearMap_mul' : Function.Bijective (LinearMap.mul' R A) :=
  have := tensorProduct_compatibleSMul S A A A
  (Algebra.TensorProduct.lmulEquiv R A).bijective


lemma Algebra.isPushout_of_isLocalization [IsLocalization (Algebra.algebraMapSubmonoid T S) B] :
    Algebra.IsPushout R T A B := by
  /-
    R : Type u_1
    inst✝¹² : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝¹¹ : CommSemiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : IsLocalization S A
    T : Type u_5
    B : Type u_6
    inst✝⁸ : CommSemiring T
    inst✝⁷ : CommSemiring B
    inst✝⁶ : Algebra R T
    inst✝⁵ : Algebra T B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R T B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsLocalization (Algebra.algebraMapSubmonoid T S) B
    ⊢ Algebra.IsPushout R T A B
  -/
  rw [Algebra.IsPushout.comm, Algebra.isPushout_iff]
  /-
    R : Type u_1
    inst✝¹² : CommSemiring R
    S : Submonoid R
    A : Type u_2
    inst✝¹¹ : CommSemiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : IsLocalization S A
    T : Type u_5
    B : Type u_6
    inst✝⁸ : CommSemiring T
    inst✝⁷ : CommSemiring B
    inst✝⁶ : Algebra R T
    inst✝⁵ : Algebra T B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R T B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsLocalization (Algebra.algebraMapSubmonoid T S) B
    ⊢ IsBaseChange A (IsScalarTower.toAlgHom R T B).toLinearMap
  -/
  apply IsLocalizedModule.isBaseChange S
  /-
    🎉 no goals
  -/


open TensorProduct in
instance (R M : Type*) [CommRing R] [AddCommGroup M] [Module R M]
    {α} (S : Submonoid R) {Mₛ} [AddCommGroup Mₛ] [Module R Mₛ] (f : M →ₗ[R] Mₛ)
    [IsLocalizedModule S f] : IsLocalizedModule S (Finsupp.mapRange.linearMap (α := α) f) := by
  classical
  let e : Localization S ⊗[R] M ≃ₗ[R] Mₛ :=
    (IsLocalizedModule.isBaseChange S (Localization S)
      (LocalizedModule.mkLinearMap S M)).equiv.restrictScalars R ≪≫ₗ IsLocalizedModule.iso S f
  let e' : Localization S ⊗[R] (α →₀ M) ≃ₗ[R] (α →₀ Mₛ) :=
    finsuppRight R (Localization S) M α ≪≫ₗ Finsupp.mapRange.linearEquiv e
  suffices IsLocalizedModule S (e'.symm.toLinearMap ∘ₗ Finsupp.mapRange.linearMap f) by
    convert this.of_linearEquiv (e := e')
    ext
    simp
  rw [isLocalizedModule_iff_isBaseChange S (Localization S)]
  convert TensorProduct.isBaseChange R (α →₀ M) (Localization S) using 1
  ext a m
  apply (finsuppRight R (Localization S) M α).injective
  ext b
  apply e.injective
  suffices (if a = b then f m else 0) = e (1 ⊗ₜ[R] if a = b then m else 0) by
    simpa [e', Finsupp.single_apply, -EmbeddingLike.apply_eq_iff_eq, apply_ite e]
  split_ifs with h
  · simp [e, IsBaseChange.equiv_tmul]
  · simp only [tmul_zero, LinearEquiv.trans_apply, LinearEquiv.restrictScalars_apply, map_zero]


/-- `S⁻¹M ⊗[R] N = S⁻¹(M ⊗[R] N)`. -/
instance IsLocalizedModule.rTensor (g : M →ₗ[A] M') [h : IsLocalizedModule S g] :
    IsLocalizedModule S (AlgebraTensorModule.rTensor R N g) := by
  /-
    R : Type u_1
    inst✝²¹ : CommSemiring R
    S✝ : Submonoid R
    A : Type u_2
    inst✝²⁰ : CommSemiring A
    inst✝¹⁹ : Algebra R A
    inst✝¹⁸ : IsLocalization S✝ A
    M : Type u_3
    inst✝¹⁷ : AddCommMonoid M
    inst✝¹⁶ : Module R M
    M' : Type u_4
    inst✝¹⁵ : AddCommMonoid M'
    inst✝¹⁴ : Module R M'
    inst✝¹³ : Module A M'
    inst✝¹² : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    T : Type u_5
    B : Type u_6
    inst✝¹¹ : CommSemiring T
    inst✝¹⁰ : CommSemiring B
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra T B
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : IsScalarTower R T B
    inst✝⁴ : IsScalarTower R A B
    S : Submonoid A
    N : Type u_7
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    ⊢ IsLocalizedModule S ((TensorProduct.AlgebraTensorModule.rTensor R N) g)
  -/
  let Aₚ := Localization S
  /-
    R : Type u_1
    inst✝²¹ : CommSemiring R
    S✝ : Submonoid R
    A : Type u_2
    inst✝²⁰ : CommSemiring A
    inst✝¹⁹ : Algebra R A
    inst✝¹⁸ : IsLocalization S✝ A
    M : Type u_3
    inst✝¹⁷ : AddCommMonoid M
    inst✝¹⁶ : Module R M
    M' : Type u_4
    inst✝¹⁵ : AddCommMonoid M'
    inst✝¹⁴ : Module R M'
    inst✝¹³ : Module A M'
    inst✝¹² : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    T : Type u_5
    B : Type u_6
    inst✝¹¹ : CommSemiring T
    inst✝¹⁰ : CommSemiring B
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra T B
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : IsScalarTower R T B
    inst✝⁴ : IsScalarTower R A B
    S : Submonoid A
    N : Type u_7
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    Aₚ : Type u_2 := Localization S
    ⊢ IsLocalizedModule S ((TensorProduct.AlgebraTensorModule.rTensor R N) g)
  -/
  letI : Module Aₚ M' := (IsLocalizedModule.iso S g).symm.toAddEquiv.module Aₚ
  /-
    R : Type u_1
    inst✝²¹ : CommSemiring R
    S✝ : Submonoid R
    A : Type u_2
    inst✝²⁰ : CommSemiring A
    inst✝¹⁹ : Algebra R A
    inst✝¹⁸ : IsLocalization S✝ A
    M : Type u_3
    inst✝¹⁷ : AddCommMonoid M
    inst✝¹⁶ : Module R M
    M' : Type u_4
    inst✝¹⁵ : AddCommMonoid M'
    inst✝¹⁴ : Module R M'
    inst✝¹³ : Module A M'
    inst✝¹² : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    T : Type u_5
    B : Type u_6
    inst✝¹¹ : CommSemiring T
    inst✝¹⁰ : CommSemiring B
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra T B
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : IsScalarTower R T B
    inst✝⁴ : IsScalarTower R A B
    S : Submonoid A
    N : Type u_7
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    Aₚ : Type u_2 := Localization S
    this : Module Aₚ M' := AddEquiv.module Aₚ (IsLocalizedModule.iso S g).symm.toA …
    ⊢ IsLocalizedModule S ((TensorProduct.AlgebraTensorModule.rTensor R N) g)
  -/
  haveI : IsScalarTower A Aₚ M' := (IsLocalizedModule.iso S g).symm.isScalarTower Aₚ
  haveI : IsScalarTower R Aₚ M' :=
    IsScalarTower.of_algebraMap_smul <| fun r x ↦ by simp [IsScalarTower.algebraMap_apply R A Aₚ]
  /-
    R : Type u_1
    inst✝²¹ : CommSemiring R
    S✝ : Submonoid R
    A : Type u_2
    inst✝²⁰ : CommSemiring A
    inst✝¹⁹ : Algebra R A
    inst✝¹⁸ : IsLocalization S✝ A
    M : Type u_3
    inst✝¹⁷ : AddCommMonoid M
    inst✝¹⁶ : Module R M
    M' : Type u_4
    inst✝¹⁵ : AddCommMonoid M'
    inst✝¹⁴ : Module R M'
    inst✝¹³ : Module A M'
    inst✝¹² : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    T : Type u_5
    B : Type u_6
    inst✝¹¹ : CommSemiring T
    inst✝¹⁰ : CommSemiring B
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra T B
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : IsScalarTower R T B
    inst✝⁴ : IsScalarTower R A B
    S : Submonoid A
    N : Type u_7
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    Aₚ : Type u_2 := Localization S
    this✝¹ : Module Aₚ M' := AddEquiv.module Aₚ (IsLocalizedModule.iso S g).symm.t …
    this✝ : IsScalarTower A Aₚ M'
    this : IsScalarTower R Aₚ M'
    ⊢ IsLocalizedModule S ((TensorProduct.AlgebraTensorModule.rTensor R N) g)
  -/
  rw [isLocalizedModule_iff_isBaseChange (S := S) (A := Aₚ)] at h ⊢
  /-
    R : Type u_1
    inst✝²¹ : CommSemiring R
    S✝ : Submonoid R
    A : Type u_2
    inst✝²⁰ : CommSemiring A
    inst✝¹⁹ : Algebra R A
    inst✝¹⁸ : IsLocalization S✝ A
    M : Type u_3
    inst✝¹⁷ : AddCommMonoid M
    inst✝¹⁶ : Module R M
    M' : Type u_4
    inst✝¹⁵ : AddCommMonoid M'
    inst✝¹⁴ : Module R M'
    inst✝¹³ : Module A M'
    inst✝¹² : IsScalarTower R A M'
    f : LinearMap (RingHom.id R) M M'
    T : Type u_5
    B : Type u_6
    inst✝¹¹ : CommSemiring T
    inst✝¹⁰ : CommSemiring B
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra T B
    inst✝⁷ : Algebra R B
    inst✝⁶ : Algebra A B
    inst✝⁵ : IsScalarTower R T B
    inst✝⁴ : IsScalarTower R A B
    S : Submonoid A
    N : Type u_7
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    g : LinearMap (RingHom.id A) M M'
    h✝ : IsLocalizedModule S g
    Aₚ : Type u_2 := Localization S
    this✝¹ : Module Aₚ M' := AddEquiv.module Aₚ (IsLocalizedModule.iso S g).symm.t …
    this✝ : IsScalarTower A Aₚ M'
    h : IsBaseChange Aₚ g
    this : IsScalarTower R Aₚ M'
    ⊢ IsBaseChange Aₚ ((TensorProduct.AlgebraTensorModule.rTensor R N) g)
  -/
  exact isBaseChange_tensorProduct_map _ h
  /-
    🎉 no goals
  -/


lemma IsLocalizedModule.map_lTensor (g : M →ₗ[A] M') [h : IsLocalizedModule S g] :
    IsLocalizedModule.map S (AlgebraTensorModule.rTensor R N g) (AlgebraTensorModule.rTensor R P g)
      (AlgebraTensorModule.lTensor A M f) = AlgebraTensorModule.lTensor A M' f := by
  /-
    R : Type u_1
    inst✝¹⁴ : CommSemiring R
    A : Type u_2
    inst✝¹³ : CommSemiring A
    inst✝¹² : Algebra R A
    M : Type u_3
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : Module R M
    M' : Type u_4
    inst✝⁹ : AddCommMonoid M'
    inst✝⁸ : Module R M'
    inst✝⁷ : Module A M'
    inst✝⁶ : IsScalarTower R A M'
    S : Submonoid A
    N : Type u_7
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    P : Type u_8
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    ⊢ Eq ((IsLocalizedModule.map S ((TensorProduct.AlgebraTensorModule.rTensor R N …
  -/
  apply linearMap_ext S (AlgebraTensorModule.rTensor R N g) (AlgebraTensorModule.rTensor R P g)
  /-
    case h
    R : Type u_1
    inst✝¹⁴ : CommSemiring R
    A : Type u_2
    inst✝¹³ : CommSemiring A
    inst✝¹² : Algebra R A
    M : Type u_3
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : Module R M
    M' : Type u_4
    inst✝⁹ : AddCommMonoid M'
    inst✝⁸ : Module R M'
    inst✝⁷ : Module A M'
    inst✝⁶ : IsScalarTower R A M'
    S : Submonoid A
    N : Type u_7
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    P : Type u_8
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    ⊢ Eq (((IsLocalizedModule.map S ((TensorProduct.AlgebraTensorModule.rTensor R  …
  -/
  rw [map_comp]
  /-
    case h
    R : Type u_1
    inst✝¹⁴ : CommSemiring R
    A : Type u_2
    inst✝¹³ : CommSemiring A
    inst✝¹² : Algebra R A
    M : Type u_3
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : Module R M
    M' : Type u_4
    inst✝⁹ : AddCommMonoid M'
    inst✝⁸ : Module R M'
    inst✝⁷ : Module A M'
    inst✝⁶ : IsScalarTower R A M'
    S : Submonoid A
    N : Type u_7
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    P : Type u_8
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.rTensor R P) g).comp ((TensorProduct …
  -/
  ext
  /-
    case h.a.h.h
    R : Type u_1
    inst✝¹⁴ : CommSemiring R
    A : Type u_2
    inst✝¹³ : CommSemiring A
    inst✝¹² : Algebra R A
    M : Type u_3
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : Module R M
    M' : Type u_4
    inst✝⁹ : AddCommMonoid M'
    inst✝⁸ : Module R M'
    inst✝⁷ : Module A M'
    inst✝⁶ : IsScalarTower R A M'
    S : Submonoid A
    N : Type u_7
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module A M
    inst✝² : IsScalarTower R A M
    P : Type u_8
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) N P
    g : LinearMap (RingHom.id A) M M'
    h : IsLocalizedModule S g
    x✝¹ : M
    x✝ : N
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  simp
  /-
    🎉 no goals
  -/


