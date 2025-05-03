                                                                                         /-
                                                                                           C : Type u
                                                                                           inst✝² : CategoryTheory.Category.{v, u} C
                                                                                           inst✝¹ : CategoryTheory.Abelian C
                                                                                           inst✝ : CategoryTheory.HasExt C
                                                                                           S : CategoryTheory.ShortComplex C
                                                                                           ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp (CategoryTheory.Abelian.Ext.mk …
                                                                                         -/
lemma ext_mk₀_f_comp_ext_mk₀_g : (Ext.mk₀ S.f).comp (Ext.mk₀ S.g) (zero_add 0) = 0 := by simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


local notation "W" => HomologicalComplex.quasiIso C (ComplexShape.up ℤ)

local notation "S'" => S.map (CochainComplex.singleFunctor C 0)

local notation "hS'" => hS.map_of_exact (HomologicalComplex.single _ _ _)

local notation "K" => CochainComplex.mappingCone (ShortComplex.f S')

local notation "qis" => CochainComplex.mappingCone.descShortComplex S'

local notation "hqis" => CochainComplex.mappingCone.quasiIso_descShortComplex hS'

local notation "δ" => Triangle.mor₃ (CochainComplex.mappingCone.triangle (ShortComplex.f S'))


instance : HasSmallLocalizedShiftedHom.{w} W ℤ (S').X₃ (S').X₁ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedShiftedHom (HomologicalComplex. …
  -/
  dsimp
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedShiftedHom (HomologicalComplex. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


include hS in
private lemma hasSmallLocalizedHom_S'_X₃_K :
    HasSmallLocalizedHom.{w} W (S').X₃ K := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedHom (HomologicalComplex.quasiIs …
  -/
  rw [Localization.hasSmallLocalizedHom_iff_target W (S').X₃ qis hqis]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedHom (HomologicalComplex.quasiIs …
  -/
  dsimp
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedHom (HomologicalComplex.quasiIs …
  -/
  apply Localization.hasSmallLocalizedHom_of_hasSmallLocalizedShiftedHom₀ (M := ℤ)
  /-
    🎉 no goals
  -/


include hS in
private lemma hasSmallLocalizedShiftedHom_K_S'_X₁ :
    HasSmallLocalizedShiftedHom.{w} W ℤ K (S').X₁ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedShiftedHom (HomologicalComplex. …
  -/
  rw [Localization.hasSmallLocalizedShiftedHom_iff_source.{w} W ℤ qis hqis (S').X₁]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Localization.HasSmallLocalizedShiftedHom (HomologicalComplex. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The class in `Ext S.X₃ S.X₁ 1` that is attached to a short exact
short complex `S` in an abelian category. -/
noncomputable def extClass : Ext.{w} S.X₃ S.X₁ 1 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Abelian.Ext S.X₃ S.X₁ 1
  -/
  have := hS.hasSmallLocalizedHom_S'_X₃_K
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : CategoryTheory.Localization.HasSmallLocalizedHom (HomologicalComplex.qu …
    ⊢ CategoryTheory.Abelian.Ext S.X₃ S.X₁ 1
  -/
  have := hS.hasSmallLocalizedShiftedHom_K_S'_X₁
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this✝ : CategoryTheory.Localization.HasSmallLocalizedHom (HomologicalComplex.q …
    this : CategoryTheory.Localization.HasSmallLocalizedShiftedHom (HomologicalCom …
    ⊢ CategoryTheory.Abelian.Ext S.X₃ S.X₁ 1
  -/
  change SmallHom W (S').X₃ ((S').X₁⟦(1 : ℤ)⟧)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this✝ : CategoryTheory.Localization.HasSmallLocalizedHom (HomologicalComplex.q …
    this : CategoryTheory.Localization.HasSmallLocalizedShiftedHom (HomologicalCom …
    ⊢ CategoryTheory.Localization.SmallHom (HomologicalComplex.quasiIso C (Complex …
  -/
  exact (SmallHom.mkInv qis hqis).comp (SmallHom.mk W δ)
  /-
    🎉 no goals
  -/


@[simp]
lemma extClass_hom [HasDerivedCategory.{w'} C] : hS.extClass.hom = hS.singleδ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq hS.extClass.hom hS.singleδ
  -/
  change SmallShiftedHom.equiv W Q hS.extClass = _
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq ((CategoryTheory.Localization.SmallShiftedHom.equiv (HomologicalComplex.q …
  -/
  dsimp [extClass, SmallShiftedHom.equiv]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq (((DerivedCategory.Q.commShiftIso 1).app ((CochainComplex.singleFunctor C …
  -/
  erw [SmallHom.equiv_comp, Iso.homToEquiv_apply]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [SmallHom.equiv_mkInv, SmallHom.equiv_mk]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [singleδ, triangleOfSESδ]
  rw [Category.assoc, Category.assoc, Category.assoc,
    singleFunctorsPostcompQIso_hom_hom, singleFunctorsPostcompQIso_inv_hom]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.isoOfHom …
  -/
  erw [Category.id_comp, Functor.map_id, Category.comp_id]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.isoOfHom …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_extClass : (Ext.mk₀ S.g).comp hS.extClass (zero_add 1) = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp hS.extClass ⋯) 0
  -/
  letI := HasDerivedCategory.standard C
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp hS.extClass ⋯) 0
  -/
  ext
  simp only [Ext.comp_hom, Ext.mk₀_hom, extClass_hom, Ext.zero_hom,
    ShiftedHom.mk₀_comp]
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((DerivedCategory.singleFunctor C 0). …
  -/
  exact comp_distTriang_mor_zero₂₃ _ hS.singleTriangle_distinguished
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_extClass_assoc {Y : C} {n : ℕ} (γ : Ext S.X₁ Y n) {n' : ℕ} (h : 1 + n = n') :
    (Ext.mk₀ S.g).comp (hS.extClass.comp γ h) (zero_add n') = 0 := by
  rw [← Ext.comp_assoc (a₁₂ := 1) _ _ _ (by omega) (by omega) (by omega),
    comp_extClass, Ext.zero_comp]


@[simp]
lemma extClass_comp : hS.extClass.comp (Ext.mk₀ S.f) (add_zero 1) = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ Eq (hS.extClass.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) 0
  -/
  letI := HasDerivedCategory.standard C
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    ⊢ Eq (hS.extClass.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) 0
  -/
  ext
  simp only [Ext.comp_hom, Ext.mk₀_hom, extClass_hom, Ext.zero_hom,
    ShiftedHom.comp_mk₀]
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp hS.singleδ ((CategoryTheory.shiftFunc …
  -/
  exact comp_distTriang_mor_zero₃₁ _ hS.singleTriangle_distinguished
  /-
    🎉 no goals
  -/


@[simp]
lemma extClass_comp_assoc {Y : C} {n : ℕ} (γ : Ext S.X₂ Y n) {n' : ℕ} {h : 1 + n = n'} :
    hS.extClass.comp ((Ext.mk₀ S.f).comp γ (zero_add n)) h = 0 := by
  rw [← Ext.comp_assoc (a₁₂ := 1) _ _ _ (by omega) (by omega) (by omega),
    extClass_comp, Ext.zero_comp]


