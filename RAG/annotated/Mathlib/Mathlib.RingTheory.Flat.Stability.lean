private noncomputable abbrev auxRightMul (I : Ideal R) : M ⊗[R] I →ₗ[S] M := by
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    I : Ideal R
    ⊢ LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.mem …
  -/
  letI i : M ⊗[R] I →ₗ[S] M ⊗[R] R := AlgebraTensorModule.map LinearMap.id I.subtype
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    I : Ideal R
    i : LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.m …
    ⊢ LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.mem …
  -/
  letI e' : M ⊗[R] R →ₗ[S] M := AlgebraTensorModule.rid R S M
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    I : Ideal R
    i : LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.m …
    e' : LinearMap (RingHom.id S) (TensorProduct R M R) M := ↑(TensorProduct.Algeb …
    ⊢ LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.mem …
  -/
  exact AlgebraTensorModule.rid R S M ∘ₗ i
  /-
    🎉 no goals
  -/


private noncomputable abbrev J (I : Ideal R) : Ideal S := LinearMap.range (auxRightMul R S S I)


private noncomputable abbrev auxIso [Flat R S] {I : Ideal R} :
    S ⊗[R] I ≃ₗ[S] J R S I := by
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    ⊢ LinearEquiv (RingHom.id S) (TensorProduct R S (Subtype fun x => Membership.m …
  -/
  apply LinearEquiv.ofInjective (auxRightMul R S S I)
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    ⊢ Function.Injective ⇑(Module.Flat.auxRightMul R S S I)
  -/
  simp only [LinearMap.coe_comp, LinearEquiv.coe_coe, EquivLike.comp_injective]
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    ⊢ Function.Injective ⇑(TensorProduct.AlgebraTensorModule.map LinearMap.id (Sub …
  -/
  exact (Flat.iff_lTensor_injective' R S).mp inferInstance I
  /-
    🎉 no goals
  -/


private noncomputable abbrev auxLTensor [Flat R S] (I : Ideal R) :
    M ⊗[R] I →ₗ[S] M := by
  letI e1 : M ⊗[R] I ≃ₗ[S] M ⊗[S] (S ⊗[R] I) :=
    (AlgebraTensorModule.cancelBaseChange R S S M I).symm
  letI e2 : M ⊗[S] (S ⊗[R] I) ≃ₗ[S] M ⊗[S] (J R S I) :=
    TensorProduct.congr (LinearEquiv.refl S M) (auxIso R S)
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    e1 : LinearEquiv (RingHom.id S) (TensorProduct R M (Subtype fun x => Membershi …
    e2 : LinearEquiv (RingHom.id S) (TensorProduct S M (TensorProduct R S (Subtype …
    ⊢ LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.mem …
  -/
  letI e3 : M ⊗[S] (J R S I) →ₗ[S] M ⊗[S] S := lTensor M (J R S I).subtype
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    e1 : LinearEquiv (RingHom.id S) (TensorProduct R M (Subtype fun x => Membershi …
    e2 : LinearEquiv (RingHom.id S) (TensorProduct S M (TensorProduct R S (Subtype …
    e3 : LinearMap (RingHom.id S) (TensorProduct S M (Subtype fun x => Membership. …
    ⊢ LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.mem …
  -/
  letI e4 : M ⊗[S] S →ₗ[S] M := TensorProduct.rid S M
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    e1 : LinearEquiv (RingHom.id S) (TensorProduct R M (Subtype fun x => Membershi …
    e2 : LinearEquiv (RingHom.id S) (TensorProduct S M (TensorProduct R S (Subtype …
    e3 : LinearMap (RingHom.id S) (TensorProduct S M (Subtype fun x => Membership. …
    e4 : LinearMap (RingHom.id S) (TensorProduct S M S) M := ↑(TensorProduct.rid S …
    ⊢ LinearMap (RingHom.id S) (TensorProduct R M (Subtype fun x => Membership.mem …
  -/
  exact e4 ∘ₗ e3 ∘ₗ (e1 ≪≫ₗ e2)
  /-
    🎉 no goals
  -/


private lemma auxLTensor_eq [Flat R S] {I : Ideal R} :
    (auxLTensor R S M I : M ⊗[R] I →ₗ[R] M) =
    TensorProduct.rid R M ∘ₗ lTensor M I.subtype := by
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    ⊢ Eq (↑R (Module.Flat.auxLTensor R S M I)) ((↑(TensorProduct.rid R M)).comp (L …
  -/
  apply TensorProduct.ext'
  /-
    case H
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    ⊢ ∀ (x : M) (y : Subtype fun x => Membership.mem I x), Eq ((↑R (Module.Flat.au …
  -/
  intro m x
  /-
    case H
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    m : M
    x : Subtype fun x => Membership.mem I x
    ⊢ Eq ((↑R (Module.Flat.auxLTensor R S M I)) (TensorProduct.tmul R m x)) (((↑(T …
  -/
  erw [TensorProduct.rid_tmul]
  /-
    case H
    R : Type u
    S : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    inst✝ : Module.Flat R S
    I : Ideal R
    m : M
    x : Subtype fun x => Membership.mem I x
    ⊢ Eq (HSMul.hSMul ((Submodule.subtype (Module.Flat.J R S I)) { fst := ↑(Linear …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `S` is a flat `R`-algebra, then any flat `S`-Module is also `R`-flat. -/
theorem trans [Flat R S] [Flat S M] : Flat R M := by
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.Flat R S
    inst✝ : Module.Flat S M
    ⊢ Module.Flat R M
  -/
  rw [Flat.iff_lTensor_injective']
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.Flat R S
    inst✝ : Module.Flat S M
    ⊢ ∀ (I : Ideal R), Function.Injective ⇑(LinearMap.lTensor M (Submodule.subtype …
  -/
  intro I
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.Flat R S
    inst✝ : Module.Flat S M
    I : Ideal R
    ⊢ Function.Injective ⇑(LinearMap.lTensor M (Submodule.subtype I))
  -/
  rw [← EquivLike.comp_injective _ (TensorProduct.rid R M)]
  haveI h : TensorProduct.rid R M ∘ lTensor M I.subtype =
    TensorProduct.rid R M ∘ₗ lTensor M I.subtype := rfl
  simp only [h, ← auxLTensor_eq R S M, LinearMap.coe_restrictScalars, LinearMap.coe_comp,
    LinearEquiv.coe_coe, EquivLike.comp_injective, EquivLike.injective_comp]
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.Flat R S
    inst✝ : Module.Flat S M
    I : Ideal R
    h : Eq (Function.comp ⇑(TensorProduct.rid R M) ⇑(LinearMap.lTensor M (Submodul …
    ⊢ Function.Injective ⇑(LinearMap.lTensor M (Submodule.subtype (Module.Flat.J R …
  -/
  exact (Flat.iff_lTensor_injective' S M).mp inferInstance _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-03")] alias comp := trans


private noncomputable abbrev auxRTensorBaseChange (I : Ideal S) :
    I ⊗[S] (S ⊗[R] M) →ₗ[S] S ⊗[S] (S ⊗[R] M) :=
  letI e1 : I ⊗[S] (S ⊗[R] M) ≃ₗ[S] I ⊗[R] M :=
    AlgebraTensorModule.cancelBaseChange R S S I M
  letI e2 : S ⊗[S] (S ⊗[R] M) ≃ₗ[S] S ⊗[R] M :=
    AlgebraTensorModule.cancelBaseChange R S S S M
  letI f : I ⊗[R] M →ₗ[S] S ⊗[R] M := AlgebraTensorModule.map I.subtype LinearMap.id
  e2.symm.toLinearMap ∘ₗ f ∘ₗ e1.toLinearMap


private lemma auxRTensorBaseChange_eq (I : Ideal S) :
    auxRTensorBaseChange R S M I = rTensor (S ⊗[R] M) I.subtype := by
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal S
    ⊢ Eq (Module.Flat.auxRTensorBaseChange R S M I) (LinearMap.rTensor (TensorProd …
  -/
  ext
  /-
    case a.h.a.h.h
    R : Type u
    S : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal S
    x✝¹ : Subtype fun x => Membership.mem I x
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry ((TensorProduct.AlgebraTensorM …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `M` is a flat `R`-module and `S` is any `R`-algebra, `S ⊗[R] M` is `S`-flat. -/
instance baseChange [Flat R M] : Flat S (S ⊗[R] M) := by
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Flat R M
    ⊢ Module.Flat S (TensorProduct R S M)
  -/
  rw [Flat.iff_rTensor_injective']
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Flat R M
    ⊢ ∀ (I : Ideal S), Function.Injective ⇑(LinearMap.rTensor (TensorProduct R S M …
  -/
  intro I
  simp only [← auxRTensorBaseChange_eq, auxRTensorBaseChange, LinearMap.coe_comp,
    LinearEquiv.coe_coe, EmbeddingLike.comp_injective, EquivLike.injective_comp]
  /-
    R : Type u
    S : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Flat R M
    I : Ideal S
    ⊢ Function.Injective ⇑(TensorProduct.AlgebraTensorModule.map (Submodule.subtyp …
  -/
  exact rTensor_preserves_injective_linearMap (I.subtype : I →ₗ[R] S) Subtype.val_injective
  /-
    🎉 no goals
  -/


/-- A base change of a flat module is flat. -/
theorem isBaseChange [Flat R M] (N : Type t) [AddCommGroup N] [Module R N] [Module S N]
    [IsScalarTower R S N] {f : M →ₗ[R] N} (h : IsBaseChange S f) :
    Flat S N :=
  of_linearEquiv S (S ⊗[R] M) N (IsBaseChange.equiv h).symm


instance localizedModule [Flat R M] (S : Submonoid R) :
    Flat (Localization S) (LocalizedModule S M) := by
  apply Flat.isBaseChange (R := R) (S := Localization S)
    (f := LocalizedModule.mkLinearMap S M)
  /-
    R : Type u
    M : Type u_1
    Mp : Type u_2
    Rp : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : CommRing Rp
    inst✝⁵ : Algebra R Rp
    inst✝⁴ : AddCommGroup Mp
    inst✝³ : Module R Mp
    inst✝² : Module Rp Mp
    inst✝¹ : IsScalarTower R Rp Mp
    inst✝ : Module.Flat R M
    S : Submonoid R
    ⊢ IsBaseChange (Localization S) (LocalizedModule.mkLinearMap S M)
  -/
  rw [← isLocalizedModule_iff_isBaseChange S]
  /-
    R : Type u
    M : Type u_1
    Mp : Type u_2
    Rp : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : CommRing Rp
    inst✝⁵ : Algebra R Rp
    inst✝⁴ : AddCommGroup Mp
    inst✝³ : Module R Mp
    inst✝² : Module Rp Mp
    inst✝¹ : IsScalarTower R Rp Mp
    inst✝ : Module.Flat R M
    S : Submonoid R
    ⊢ IsLocalizedModule S (LocalizedModule.mkLinearMap S M)
  -/
  exact localizedModuleIsLocalizedModule S
  /-
    🎉 no goals
  -/


theorem of_isLocalizedModule [Flat R M] (S : Submonoid R) [IsLocalization S Rp]
    (f : M →ₗ[R] Mp) [h : IsLocalizedModule S f] : Flat Rp Mp := by
  /-
    R : Type u
    M : Type u_1
    Mp : Type u_2
    Rp : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : CommRing Rp
    inst✝⁶ : Algebra R Rp
    inst✝⁵ : AddCommGroup Mp
    inst✝⁴ : Module R Mp
    inst✝³ : Module Rp Mp
    inst✝² : IsScalarTower R Rp Mp
    inst✝¹ : Module.Flat R M
    S : Submonoid R
    inst✝ : IsLocalization S Rp
    f : LinearMap (RingHom.id R) M Mp
    h : IsLocalizedModule S f
    ⊢ Module.Flat Rp Mp
  -/
  fapply Flat.isBaseChange (R := R) (M := M) (S := Rp) (N := Mp)
  /-
    R : Type u
    M : Type u_1
    Mp : Type u_2
    Rp : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : CommRing Rp
    inst✝⁶ : Algebra R Rp
    inst✝⁵ : AddCommGroup Mp
    inst✝⁴ : Module R Mp
    inst✝³ : Module Rp Mp
    inst✝² : IsScalarTower R Rp Mp
    inst✝¹ : Module.Flat R M
    S : Submonoid R
    inst✝ : IsLocalization S Rp
    f : LinearMap (RingHom.id R) M Mp
    h : IsLocalizedModule S f
    ⊢ IsBaseChange Rp ?m.122440
  -/
  exact (isLocalizedModule_iff_isBaseChange S Rp f).mp h
  /-
    🎉 no goals
  -/


