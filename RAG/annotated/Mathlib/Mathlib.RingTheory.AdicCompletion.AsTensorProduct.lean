private
def ofTensorProductBil : AdicCompletion I R →ₗ[AdicCompletion I R] M →ₗ[R] AdicCompletion I M where
  toFun r := LinearMap.lsmul (AdicCompletion I R) (AdicCompletion I M) r ∘ₗ of I M
  map_add' x y := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      x y : AdicCompletion I R
      ⊢ Eq ((fun r => (↑R ((LinearMap.lsmul (AdicCompletion I R) (AdicCompletion I M …
    -/
    apply LinearMap.ext
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      x y : AdicCompletion I R
      ⊢ ∀ (x_1 : M), Eq (((fun r => (↑R ((LinearMap.lsmul (AdicCompletion I R) (Adic …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_smul' r x := by
    /-
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      r x : AdicCompletion I R
      ⊢ Eq ({ toFun := fun r => (↑R ((LinearMap.lsmul (AdicCompletion I R) (AdicComp …
    -/
    apply LinearMap.ext
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      r x : AdicCompletion I R
      ⊢ ∀ (x_1 : M), Eq (({ toFun := fun r => (↑R ((LinearMap.lsmul (AdicCompletion  …
    -/
    intro y
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      r x : AdicCompletion I R
      y : M
      ⊢ Eq (({ toFun := fun r => (↑R ((LinearMap.lsmul (AdicCompletion I R) (AdicCom …
    -/
    ext n
    /-
      case h.h
      R : Type u_1
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      r x : AdicCompletion I R
      y : M
      n : Nat
      ⊢ Eq (↑(({ toFun := fun r => (↑R ((LinearMap.lsmul (AdicCompletion I R) (AdicC …
    -/
    simp [mul_smul (r.val n)]
    /-
      🎉 no goals
    -/


@[simp]
private lemma ofTensorProductBil_apply_apply (r : AdicCompletion I R) (x : M) :
    ((AdicCompletion.ofTensorProductBil I M) r) x = r • (of I M) x :=
  rfl


/-- The natural `AdicCompletion I R`-linear map from `AdicCompletion I R ⊗[R] M` to
the adic completion of `M`. -/
def ofTensorProduct : AdicCompletion I R ⊗[R] M →ₗ[AdicCompletion I R] AdicCompletion I M :=
  TensorProduct.AlgebraTensorModule.lift (ofTensorProductBil I M)


@[simp]
lemma ofTensorProduct_tmul (r : AdicCompletion I R) (x : M) :
    ofTensorProduct I M (r ⊗ₜ x) = r • of I M x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    r : AdicCompletion I R
    x : M
    ⊢ Eq ((AdicCompletion.ofTensorProduct I M) (TensorProduct.tmul R r x)) (HSMul. …
  -/
  simp [ofTensorProduct]
  /-
    🎉 no goals
  -/


variable {M} in
/-- `ofTensorProduct` is functorial in `M`. -/
lemma ofTensorProduct_naturality (f : M →ₗ[R] N) :
    map I f ∘ₗ ofTensorProduct I M =
      ofTensorProduct I N ∘ₗ AlgebraTensorModule.map LinearMap.id f := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq ((AdicCompletion.map I f).comp (AdicCompletion.ofTensorProduct I M)) ((Ad …
  -/
  ext
  /-
    case a.h.h.h
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    x✝ : M
    n✝ : Nat
    ⊢ Eq (↑(((TensorProduct.AlgebraTensorModule.curry ((AdicCompletion.map I f).co …
  -/
  simp
  /-
    🎉 no goals
  -/


private lemma piEquivOfFintype_comp_ofTensorProduct_eq :
    piEquivOfFintype I (fun _ : ι ↦ R) ∘ₗ ofTensorProduct I (ι → R) =
      (TensorProduct.piScalarRight R (AdicCompletion I R) (AdicCompletion I R) ι).toLinearMap := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((↑(AdicCompletion.piEquivOfFintype I fun x => R)).comp (AdicCompletion.o …
  -/
  ext i j k
  suffices h : (if j = i then 1 else 0) = (if j = i then 1 else 0 : AdicCompletion I R).val k by
    simpa [Pi.single_apply, -smul_eq_mul, -Algebra.id.smul_eq_mul]
  /-
    case a.h.h.h.h.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    i j : ι
    k : Nat
    ⊢ Eq (ite (Eq j i) 1 0) (↑(ite (Eq j i) 1 0) k)
  -/
            /-
              🎉 no goals
            -/
  split <;> simp
            /-
              🎉 no goals
            -/


private lemma ofTensorProduct_eq :
    ofTensorProduct I (ι → R) = (piEquivOfFintype I (ι := ι) (fun _ : ι ↦ R)).symm.toLinearMap ∘ₗ
      (TensorProduct.piScalarRight R (AdicCompletion I R) (AdicCompletion I R) ι).toLinearMap := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq (AdicCompletion.ofTensorProduct I (ι → R)) ((↑(AdicCompletion.piEquivOfFi …
  -/
  rw [← piEquivOfFintype_comp_ofTensorProduct_eq I ι, ← LinearMap.comp_assoc]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq (AdicCompletion.ofTensorProduct I (ι → R)) (((↑(AdicCompletion.piEquivOfF …
  -/
  simp
  /-
    🎉 no goals
  -/

/- If `M = R^ι` and `ι` is finite, we may construct an inverse to `ofTensorProduct I (ι → R)`. -/

private def ofTensorProductInvOfPiFintype :
    AdicCompletion I (ι → R) ≃ₗ[AdicCompletion I R] AdicCompletion I R ⊗[R] (ι → R) :=
  letI f := piEquivOfFintype I (fun _ : ι ↦ R)
  letI g := (TensorProduct.piScalarRight R (AdicCompletion I R) (AdicCompletion I R) ι).symm
  f.trans g


private lemma ofTensorProductInvOfPiFintype_comp_ofTensorProduct :
    ofTensorProductInvOfPiFintype I ι ∘ₗ ofTensorProduct I (ι → R) = LinearMap.id := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((↑(AdicCompletion.ofTensorProductInvOfPiFintype I ι)).comp (AdicCompleti …
  -/
  dsimp only [ofTensorProductInvOfPiFintype]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((↑((AdicCompletion.piEquivOfFintype I fun x => R).trans (TensorProduct.p …
  -/
  rw [LinearEquiv.coe_trans, LinearMap.comp_assoc, piEquivOfFintype_comp_ofTensorProduct_eq]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((↑(TensorProduct.piScalarRight R (AdicCompletion I R) (AdicCompletion I  …
  -/
  simp
  /-
    🎉 no goals
  -/


private lemma ofTensorProduct_comp_ofTensorProductInvOfPiFintype :
    ofTensorProduct I (ι → R) ∘ₗ ofTensorProductInvOfPiFintype I ι = LinearMap.id := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((AdicCompletion.ofTensorProduct I (ι → R)).comp ↑(AdicCompletion.ofTenso …
  -/
  dsimp only [ofTensorProductInvOfPiFintype]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((AdicCompletion.ofTensorProduct I (ι → R)).comp ↑((AdicCompletion.piEqui …
  -/
  rw [LinearEquiv.coe_trans, ofTensorProduct_eq, LinearMap.comp_assoc]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((↑(AdicCompletion.piEquivOfFintype I fun x => R).symm).comp ((↑(TensorPr …
  -/
  nth_rw 2 [← LinearMap.comp_assoc]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((↑(AdicCompletion.piEquivOfFintype I fun x => R).symm).comp (((↑(TensorP …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `ofTensorProduct` as an equiv in the case of `M = R^ι` where `ι` is finite. -/
def ofTensorProductEquivOfPiFintype :
    AdicCompletion I R ⊗[R] (ι → R) ≃ₗ[AdicCompletion I R] AdicCompletion I (ι → R) :=
  LinearEquiv.ofLinear
    (ofTensorProduct I (ι → R))
    (ofTensorProductInvOfPiFintype I ι)
    (ofTensorProduct_comp_ofTensorProductInvOfPiFintype I ι)
    (ofTensorProductInvOfPiFintype_comp_ofTensorProduct I ι)


/-- If `M = R^ι`, `ofTensorProduct` is bijective. -/
lemma ofTensorProduct_bijective_of_pi_of_fintype [Finite ι] :
    Function.Bijective (ofTensorProduct I (ι → R)) := by
  classical
  cases nonempty_fintype ι
  exact EquivLike.bijective (ofTensorProductEquivOfPiFintype I ι)


/-- If `M` is a finite `R`-module, then the canonical map
`AdicCompletion I R ⊗[R] M →ₗ AdicCompletion I M` is surjective. -/
lemma ofTensorProduct_surjective_of_finite [Module.Finite R M] :
    Function.Surjective (ofTensorProduct I M) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    ⊢ Function.Surjective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  obtain ⟨n, p, hp⟩ := Module.Finite.exists_fin' R M
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    n : Nat
    p : LinearMap (RingHom.id R) (Fin n → R) M
    hp : Function.Surjective ⇑p
    ⊢ Function.Surjective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  let f := ofTensorProduct I M ∘ₗ p.baseChange (AdicCompletion I R)
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    n : Nat
    p : LinearMap (RingHom.id R) (Fin n → R) M
    hp : Function.Surjective ⇑p
    f : LinearMap (RingHom.id (AdicCompletion I R)) (TensorProduct R (AdicCompleti …
    ⊢ Function.Surjective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  let g := map I p ∘ₗ ofTensorProduct I (Fin n → R)
  have hfg : f = g := by
    ext
    simp [f, g]
  have hf : Function.Surjective f := by
    simp only [hfg, LinearMap.coe_comp, g]
    apply Function.Surjective.comp
    · exact AdicCompletion.map_surjective I hp
    · exact (ofTensorProduct_bijective_of_pi_of_fintype I (Fin n)).surjective
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    n : Nat
    p : LinearMap (RingHom.id R) (Fin n → R) M
    hp : Function.Surjective ⇑p
    f : LinearMap (RingHom.id (AdicCompletion I R)) (TensorProduct R (AdicCompleti …
    g : LinearMap (RingHom.id (AdicCompletion I R)) (TensorProduct R (AdicCompleti …
    hfg : Eq f g
    hf : Function.Surjective ⇑f
    ⊢ Function.Surjective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  exact Function.Surjective.of_comp hf
  /-
    🎉 no goals
  -/


private
def lTensorKerIncl : AdicCompletion I R ⊗[R] LinearMap.ker f →ₗ[AdicCompletion I R]
    AdicCompletion I R ⊗[R] (ι → R) :=
  AlgebraTensorModule.map LinearMap.id (LinearMap.ker f).subtype

/- The second horizontal arrow in the top row. -/

private def lTensorf :
    AdicCompletion I R ⊗[R] (ι → R) →ₗ[AdicCompletion I R] AdicCompletion I R ⊗[R] M :=
  AlgebraTensorModule.map LinearMap.id f


private lemma tens_exact : Function.Exact (lTensorKerIncl I M f) (lTensorf I M f) :=
  lTensor_exact (AdicCompletion I R) (f.exact_subtype_ker_map) hf


private lemma tens_surj : Function.Surjective (lTensorf I M f) :=
  LinearMap.lTensor_surjective (AdicCompletion I R) hf


private lemma adic_exact [IsNoetherianRing R] [Fintype ι] :
    Function.Exact (map I (LinearMap.ker f).subtype) (map I f) :=
  map_exact (Submodule.injective_subtype _) (f.exact_subtype_ker_map) hf


private lemma adic_surj : Function.Surjective (map I f) :=
  map_surjective I hf


private instance : AddCommGroup (AdicCompletion I R ⊗[R] (LinearMap.ker f)) :=
  inferInstance


private def firstRow : ComposableArrows (ModuleCat (AdicCompletion I R)) 4 :=
  ComposableArrows.mk₄
    (ModuleCat.ofHom <| lTensorKerIncl I M f)
    (ModuleCat.ofHom <| lTensorf I M f)
    (ModuleCat.ofHom (0 : AdicCompletion I R ⊗[R] M →ₗ[AdicCompletion I R] PUnit))
    (ModuleCat.ofHom (0 : _ →ₗ[AdicCompletion I R] PUnit))


private def secondRow : ComposableArrows (ModuleCat (AdicCompletion I R)) 4 :=
  ComposableArrows.mk₄
    (ModuleCat.ofHom (map I <| (LinearMap.ker f).subtype))
    (ModuleCat.ofHom (map I f))
    (ModuleCat.ofHom (0 : _ →ₗ[AdicCompletion I R] PUnit))
    (ModuleCat.ofHom (0 : _ →ₗ[AdicCompletion I R] PUnit))


private lemma firstRow_exact : (firstRow I M f).Exact where
  zero k _ := match k with
    | 0 => ModuleCat.hom_ext (tens_exact I M f hf).linearMap_comp_eq_zero
    | 1 => ModuleCat.hom_ext (LinearMap.zero_comp _)
    | 2 => ModuleCat.hom_ext (LinearMap.zero_comp 0)
  exact k _ := by
    /-
      R : Type u
      inst✝² : CommRing R
      I : Ideal R
      M : Type u
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      k : Nat
      x✝ : autoParam (LE.le (HAdd.hAdd k 2) 4) _auto✝
      ⊢ ((AdicCompletion.firstRow I M f).sc ⋯ k ⋯).Exact
    -/
    rw [ShortComplex.moduleCat_exact_iff]
    match k with
    | 0 => intro x hx; exact (tens_exact I M f hf x).mp hx
    | 1 => intro x _; exact (tens_surj I M f hf) x
    | 2 => intro _ _; exact ⟨0, rfl⟩


private lemma secondRow_exact [Fintype ι] [IsNoetherianRing R] : (secondRow I M f).Exact where
  zero k _ := match k with
    | 0 => ModuleCat.hom_ext (adic_exact I M f hf).linearMap_comp_eq_zero
    | 1 => ModuleCat.hom_ext (LinearMap.zero_comp (map I f))
    | 2 => ModuleCat.hom_ext (LinearMap.zero_comp 0)
  exact k _ := by
    /-
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      k : Nat
      x✝ : autoParam (LE.le (HAdd.hAdd k 2) 4) _auto✝
      ⊢ ((AdicCompletion.secondRow I M f).sc ⋯ k ⋯).Exact
    -/
    rw [ShortComplex.moduleCat_exact_iff]
    match k with
    | 0 => intro x hx; exact (adic_exact I M f hf x).mp hx
    | 1 => intro x _; exact (adic_surj I M f hf) x
    | 2 => intro _ _; exact ⟨0, rfl⟩

/- The compatible vertical maps between the first and the second row. -/

private def firstRowToSecondRow : firstRow I M f ⟶ secondRow I M f :=
  ComposableArrows.homMk₄
    (ModuleCat.ofHom (ofTensorProduct I (LinearMap.ker f)))
    (ModuleCat.ofHom (ofTensorProduct I (ι → R)))
    (ModuleCat.ofHom (ofTensorProduct I M))
    (ModuleCat.ofHom 0)
    (ModuleCat.ofHom 0)
    (ModuleCat.hom_ext (ofTensorProduct_naturality I <| (LinearMap.ker f).subtype).symm)
    (ModuleCat.hom_ext (ofTensorProduct_naturality I f).symm)
    rfl
    rfl


private lemma ofTensorProduct_iso [Fintype ι] [IsNoetherianRing R] :
    IsIso (ModuleCat.ofHom (ofTensorProduct I M)) := by
  refine Abelian.isIso_of_epi_of_isIso_of_isIso_of_mono
    (firstRow_exact I M f hf) (secondRow_exact I M f hf) (firstRowToSecondRow I M f) ?_ ?_ ?_ ?_
    /-
      case refine_1
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' (AdicCompletion.fir …
    -/
  · apply ConcreteCategory.epi_of_surjective
    /-
      case refine_1.s
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ Function.Surjective ⇑(CategoryTheory.ComposableArrows.app' (AdicCompletion.f …
    -/
    exact ofTensorProduct_surjective_of_finite I (LinearMap.ker f)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' (AdicCompletion.f …
    -/
  · apply (ConcreteCategory.isIso_iff_bijective _).mpr
    /-
      case refine_2
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ Function.Bijective ((CategoryTheory.forget (ModuleCat (AdicCompletion I R))) …
    -/
    exact ofTensorProduct_bijective_of_pi_of_fintype I ι
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' (AdicCompletion.f …
    -/
  · show IsIso (ModuleCat.ofHom 0)
    /-
      case refine_3
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ CategoryTheory.IsIso (ModuleCat.ofHom 0)
    -/
    apply Limits.isIso_of_isTerminal
          /-
            case refine_3.hX
            R : Type u
            inst✝⁴ : CommRing R
            I : Ideal R
            M : Type u
            inst✝³ : AddCommGroup M
            inst✝² : Module R M
            ι : Type
            f : LinearMap (RingHom.id R) (ι → R) M
            hf : Function.Surjective ⇑f
            inst✝¹ : Fintype ι
            inst✝ : IsNoetherianRing R
            ⊢ CategoryTheory.Limits.IsTerminal (ModuleCat.of (AdicCompletion I R) PUnit.{u …
          -/
          /-
            🎉 no goals
          -/
      <;> exact Limits.IsZero.isTerminal (ModuleCat.isZero_of_subsingleton _)
          /-
            🎉 no goals
          -/
    /-
      case refine_4
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' (AdicCompletion.fi …
    -/
  · apply ConcreteCategory.mono_of_injective
    /-
      case refine_4.i
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      ⊢ Function.Injective ⇑(CategoryTheory.ComposableArrows.app' (AdicCompletion.fi …
    -/
    intro x y _
    /-
      case refine_4.i
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type
      f : LinearMap (RingHom.id R) (ι → R) M
      hf : Function.Surjective ⇑f
      inst✝¹ : Fintype ι
      inst✝ : IsNoetherianRing R
      x y : (CategoryTheory.forget (ModuleCat (AdicCompletion I R))).obj ((AdicCompl …
      a✝ : Eq ((CategoryTheory.ComposableArrows.app' (AdicCompletion.firstRowToSecon …
      ⊢ Eq x y
    -/
    rfl
    /-
      🎉 no goals
    -/


private
lemma ofTensorProduct_bijective_of_map_from_fin [Fintype ι] [IsNoetherianRing R] :
    Function.Bijective (ofTensorProduct I M) := by
  have : IsIso (ModuleCat.ofHom (ofTensorProduct I M)) :=
    ofTensorProduct_iso I M f hf
  /-
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type
    f : LinearMap (RingHom.id R) (ι → R) M
    hf : Function.Surjective ⇑f
    inst✝¹ : Fintype ι
    inst✝ : IsNoetherianRing R
    this : CategoryTheory.IsIso (ModuleCat.ofHom (AdicCompletion.ofTensorProduct I …
    ⊢ Function.Bijective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  exact ConcreteCategory.bijective_of_isIso (ModuleCat.ofHom (ofTensorProduct I M))
  /-
    🎉 no goals
  -/


/-- If `R` is a Noetherian ring and `M` is a finite `R`-module, then the natural map
given by `AdicCompletion.ofTensorProduct` is an isomorphism. -/
theorem ofTensorProduct_bijective_of_finite_of_isNoetherian
    [Module.Finite R M] :
    Function.Bijective (ofTensorProduct I M) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    ⊢ Function.Bijective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  obtain ⟨n, f, hf⟩ := Module.Finite.exists_fin' R M
  /-
    case intro.intro
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    n : Nat
    f : LinearMap (RingHom.id R) (Fin n → R) M
    hf : Function.Surjective ⇑f
    ⊢ Function.Bijective ⇑(AdicCompletion.ofTensorProduct I M)
  -/
  exact ofTensorProduct_bijective_of_map_from_fin I M f hf
  /-
    🎉 no goals
  -/


/-- `ofTensorProduct` packaged as linear equiv if `M` is a finite `R`-module and `R` is
Noetherian. -/
def ofTensorProductEquivOfFiniteNoetherian [Module.Finite R M] :
    AdicCompletion I R ⊗[R] M ≃ₗ[AdicCompletion I R] AdicCompletion I M :=
  LinearEquiv.ofBijective (ofTensorProduct I M)
    (ofTensorProduct_bijective_of_finite_of_isNoetherian I M)


@[simp]
lemma ofTensorProductEquivOfFiniteNoetherian_apply [Module.Finite R M]
    (x : AdicCompletion I R ⊗[R] M) :
    ofTensorProductEquivOfFiniteNoetherian I M x = ofTensorProduct I M x :=
  rfl


@[simp]
lemma ofTensorProductEquivOfFiniteNoetherian_symm_of
    [Module.Finite R M] (x : M) :
    (ofTensorProductEquivOfFiniteNoetherian I M).symm ((of I M) x) = 1 ⊗ₜ x := by
  have h : (of I M) x = ofTensorProductEquivOfFiniteNoetherian I M (1 ⊗ₜ x) := by
    simp
  /-
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    x : M
    h : Eq ((AdicCompletion.of I M) x) ((AdicCompletion.ofTensorProductEquivOfFini …
    ⊢ Eq ((AdicCompletion.ofTensorProductEquivOfFiniteNoetherian I M).symm ((AdicC …
  -/
  rw [h, LinearEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


lemma tensor_map_id_left_eq_map :
    (AlgebraTensorModule.map LinearMap.id f) =
      (ofTensorProductEquivOfFiniteNoetherian I N).symm.toLinearMap ∘ₗ
      map I f ∘ₗ
      (ofTensorProductEquivOfFiniteNoetherian I M).toLinearMap := by
  /-
    R : Type u
    inst✝⁷ : CommRing R
    I : Ideal R
    inst✝⁶ : IsNoetherianRing R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Finite R N
    ⊢ Eq (TensorProduct.AlgebraTensorModule.map LinearMap.id f) ((↑(AdicCompletion …
  -/
  erw [ofTensorProduct_naturality I f]
  /-
    R : Type u
    inst✝⁷ : CommRing R
    I : Ideal R
    inst✝⁶ : IsNoetherianRing R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Finite R N
    ⊢ Eq (TensorProduct.AlgebraTensorModule.map LinearMap.id f) ((↑(AdicCompletion …
  -/
  ext x
  /-
    case a.h.h
    R : Type u
    inst✝⁷ : CommRing R
    I : Ideal R
    inst✝⁶ : IsNoetherianRing R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Finite R N
    x : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (TensorProduct.AlgebraTensorMo …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma tensor_map_id_left_injective_of_injective (hf : Function.Injective f) :
    Function.Injective (AlgebraTensorModule.map LinearMap.id f :
        AdicCompletion I R ⊗[R] M →ₗ[AdicCompletion I R] AdicCompletion I R ⊗[R] N) := by
  /-
    R : Type u
    inst✝⁷ : CommRing R
    I : Ideal R
    inst✝⁶ : IsNoetherianRing R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Finite R N
    hf : Function.Injective ⇑f
    ⊢ Function.Injective ⇑(TensorProduct.AlgebraTensorModule.map LinearMap.id f)
  -/
  rw [tensor_map_id_left_eq_map I f]
  simp only [LinearMap.coe_comp, LinearEquiv.coe_coe, EmbeddingLike.comp_injective,
    EquivLike.injective_comp]
  /-
    R : Type u
    inst✝⁷ : CommRing R
    I : Ideal R
    inst✝⁶ : IsNoetherianRing R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    f : LinearMap (RingHom.id R) M N
    inst✝¹ : Module.Finite R M
    inst✝ : Module.Finite R N
    hf : Function.Injective ⇑f
    ⊢ Function.Injective ⇑(AdicCompletion.map I f)
  -/
  exact map_injective I hf
  /-
    🎉 no goals
  -/


/-- Adic completion of a Noetherian ring `R` is flat over `R`. -/
instance flat_of_isNoetherian [IsNoetherianRing R] : Module.Flat R (AdicCompletion I R) :=
  (Module.Flat.iff_lTensor_injective' R (AdicCompletion I R)).mpr fun J ↦
    tensor_map_id_left_injective_of_injective I (Submodule.injective_subtype J)


