/-- No point making this an instance, as it requires `i`. -/
theorem hasKernels [PreservesFiniteLimits G] (i : F ⋙ G ≅ 𝟭 C) : HasKernels C :=
  { has_limit := fun f => by
      /-
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁴ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝¹ : G.PreservesZeroMorphisms
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        ⊢ CategoryTheory.Limits.HasKernel f
      -/
      have := NatIso.naturality_1 i f
      simp? at this says
        simp only [Functor.id_obj, Functor.comp_obj, Functor.comp_map, Functor.id_map] at this
      /-
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁴ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝¹ : G.PreservesZeroMorphisms
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this : Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X✝) (CategoryTheory.C …
        ⊢ CategoryTheory.Limits.HasKernel f
      -/
      rw [← this]
      /-
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁴ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝¹ : G.PreservesZeroMorphisms
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this : Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X✝) (CategoryTheory.C …
        ⊢ CategoryTheory.Limits.HasKernel (CategoryTheory.CategoryStruct.comp (i.inv.a …
      -/
      haveI : HasKernel (G.map (F.map f) ≫ i.hom.app _) := Limits.hasKernel_comp_mono _ _
      /-
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁴ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝¹ : G.PreservesZeroMorphisms
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this✝ : Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X✝) (CategoryTheory. …
        this : CategoryTheory.Limits.HasKernel (CategoryTheory.CategoryStruct.comp (G. …
        ⊢ CategoryTheory.Limits.HasKernel (CategoryTheory.CategoryStruct.comp (i.inv.a …
      -/
      apply Limits.hasKernel_iso_comp }
      /-
        🎉 no goals
      -/


/-- No point making this an instance, as it requires `i` and `adj`. -/
theorem hasCokernels (i : F ⋙ G ≅ 𝟭 C) (adj : G ⊣ F) : HasCokernels C :=
  { has_colimit := fun f => by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝ : G.PreservesZeroMorphisms
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        adj : CategoryTheory.Adjunction G F
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        ⊢ CategoryTheory.Limits.HasCokernel f
      -/
      have : PreservesColimits G := adj.leftAdjoint_preservesColimits
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝ : G.PreservesZeroMorphisms
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        adj : CategoryTheory.Adjunction G F
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this : CategoryTheory.Limits.PreservesColimits G
        ⊢ CategoryTheory.Limits.HasCokernel f
      -/
      have := NatIso.naturality_1 i f
      simp? at this says
        simp only [Functor.id_obj, Functor.comp_obj, Functor.comp_map, Functor.id_map] at this
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝ : G.PreservesZeroMorphisms
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        adj : CategoryTheory.Adjunction G F
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this✝ : CategoryTheory.Limits.PreservesColimits G
        this : Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X✝) (CategoryTheory.C …
        ⊢ CategoryTheory.Limits.HasCokernel f
      -/
      rw [← this]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝ : G.PreservesZeroMorphisms
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        adj : CategoryTheory.Adjunction G F
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this✝ : CategoryTheory.Limits.PreservesColimits G
        this : Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X✝) (CategoryTheory.C …
        ⊢ CategoryTheory.Limits.HasCokernel (CategoryTheory.CategoryStruct.comp (i.inv …
      -/
      haveI : HasCokernel (G.map (F.map f) ≫ i.hom.app _) := Limits.hasCokernel_comp_iso _ _
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Abelian D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        inst✝ : G.PreservesZeroMorphisms
        i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
        adj : CategoryTheory.Adjunction G F
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        this✝¹ : CategoryTheory.Limits.PreservesColimits G
        this✝ : Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X✝) (CategoryTheory. …
        this : CategoryTheory.Limits.HasCokernel (CategoryTheory.CategoryStruct.comp ( …
        ⊢ CategoryTheory.Limits.HasCokernel (CategoryTheory.CategoryStruct.comp (i.inv …
      -/
      apply Limits.hasCokernel_epi_comp }
      /-
        🎉 no goals
      -/


/-- Auxiliary construction for `coimageIsoImage` -/
def cokernelIso (i : F ⋙ G ≅ 𝟭 C) (adj : G ⊣ F) {X Y : C} (f : X ⟶ Y) :
    G.obj (cokernel (F.map f)) ≅ cokernel f := by
  -- We have to write an explicit `PreservesColimits` type here,
  -- as `leftAdjointPreservesColimits` has universe variables.
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    inst✝¹ : G.PreservesZeroMorphisms
    inst✝ : CategoryTheory.Limits.HasCokernels C
    i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
    adj : CategoryTheory.Adjunction G F
    X Y : C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Iso (G.obj (CategoryTheory.Limits.cokernel (F.map f))) (Categ …
  -/
  have : PreservesColimits G := adj.leftAdjoint_preservesColimits
  calc
    G.obj (cokernel (F.map f)) ≅ cokernel (G.map (F.map f)) :=
      (asIso (cokernelComparison _ G)).symm
    _ ≅ cokernel (i.hom.app X ≫ f ≫ i.inv.app Y) := cokernelIsoOfEq (NatIso.naturality_2 i f).symm
    _ ≅ cokernel (f ≫ i.inv.app Y) := cokernelEpiComp (i.hom.app X) (f ≫ i.inv.app Y)
    _ ≅ cokernel f := cokernelCompIsIso f (i.inv.app Y)


/-- Auxiliary construction for `coimageIsoImage` -/
def coimageIsoImageAux (i : F ⋙ G ≅ 𝟭 C) (adj : G ⊣ F) {X Y : C} (f : X ⟶ Y) :
    kernel (G.map (cokernel.π (F.map f))) ≅ kernel (cokernel.π f) := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    inst✝³ : G.PreservesZeroMorphisms
    inst✝² : CategoryTheory.Limits.HasCokernels C
    inst✝¹ : CategoryTheory.Limits.HasKernels C
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
    i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
    adj : CategoryTheory.Adjunction G F
    X Y : C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.kernel (G.map (CategoryTheory.Limi …
  -/
  have : PreservesColimits G := adj.leftAdjoint_preservesColimits
  calc
    kernel (G.map (cokernel.π (F.map f))) ≅
        kernel (cokernel.π (G.map (F.map f)) ≫ cokernelComparison (F.map f) G) :=
      kernelIsoOfEq (π_comp_cokernelComparison _ _).symm
    _ ≅ kernel (cokernel.π (G.map (F.map f))) := kernelCompMono _ _
    _ ≅ kernel (cokernel.π (_ ≫ f ≫ _) ≫ (cokernelIsoOfEq _).hom) :=
      (kernelIsoOfEq (π_comp_cokernelIsoOfEq_hom (NatIso.naturality_2 i f)).symm)
    _ ≅ kernel (cokernel.π (_ ≫ f ≫ _)) := kernelCompMono _ _
    _ ≅ kernel (cokernel.π (f ≫ i.inv.app Y) ≫ (cokernelEpiComp (i.hom.app X) _).inv) :=
      (kernelIsoOfEq (by simp only [cokernel.π_desc, cokernelEpiComp_inv]))
    _ ≅ kernel (cokernel.π (f ≫ _)) := kernelCompMono _ _
    _ ≅ kernel (inv (i.inv.app Y) ≫ cokernel.π f ≫ (cokernelCompIsIso f (i.inv.app Y)).inv) :=
      (kernelIsoOfEq
        (by simp only [cokernel.π_desc, cokernelCompIsIso_inv, Iso.hom_inv_id_app_assoc,
          NatIso.inv_inv_app]))
    _ ≅ kernel (cokernel.π f ≫ _) := kernelIsIsoComp _ _
    _ ≅ kernel (cokernel.π f) := kernelCompMono _ _


/-- Auxiliary definition: the abelian coimage and abelian image agree.
We still need to check that this agrees with the canonical morphism.
-/
def coimageIsoImage (i : F ⋙ G ≅ 𝟭 C) (adj : G ⊣ F) {X Y : C} (f : X ⟶ Y) :
    Abelian.coimage f ≅ Abelian.image f := by
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁷ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁵ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    inst✝⁴ : G.PreservesZeroMorphisms
    inst✝³ : CategoryTheory.Limits.HasCokernels C
    inst✝² : CategoryTheory.Limits.HasKernels C
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits G
    inst✝ : F.PreservesZeroMorphisms
    i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
    adj : CategoryTheory.Adjunction G F
    X Y : C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Iso (CategoryTheory.Abelian.coimage f) (CategoryTheory.Abelia …
  -/
  have : PreservesLimits F := adj.rightAdjoint_preservesLimits
  calc
    Abelian.coimage f ≅ cokernel (kernel.ι f) := Iso.refl _
    _ ≅ G.obj (cokernel (F.map (kernel.ι f))) := (cokernelIso _ _ i adj _).symm
    _ ≅ G.obj (cokernel (kernelComparison f F ≫ kernel.ι (F.map f))) :=
      (G.mapIso (cokernelIsoOfEq (by simp)))
    _ ≅ G.obj (cokernel (kernel.ι (F.map f))) := G.mapIso (cokernelEpiComp _ _)
    _ ≅ G.obj (Abelian.coimage (F.map f)) := Iso.refl _
    _ ≅ G.obj (Abelian.image (F.map f)) := G.mapIso (Abelian.coimageIsoImage _)
    _ ≅ G.obj (kernel (cokernel.π (F.map f))) := Iso.refl _
    _ ≅ kernel (G.map (cokernel.π (F.map f))) := PreservesKernel.iso _ _
    _ ≅ kernel (cokernel.π f) := coimageIsoImageAux F G i adj f
    _ ≅ Abelian.image f := Iso.refl _

-- The account of this proof in the Stacks project omits this calculation.

theorem coimageIsoImage_hom (i : F ⋙ G ≅ 𝟭 C) (adj : G ⊣ F) {X Y : C} (f : X ⟶ Y) :
    (coimageIsoImage F G i adj f).hom = Abelian.coimageImageComparison f := by
  dsimp [coimageIsoImage, cokernelIso, cokernelEpiComp, cokernelCompIsIso_inv,
    coimageIsoImageAux, kernelCompMono]
  simpa only [← cancel_mono (Abelian.image.ι f), ← cancel_epi (Abelian.coimage.π f),
    Category.assoc, Category.id_comp, cokernel.π_desc_assoc,
    π_comp_cokernelIsoOfEq_inv_assoc, PreservesKernel.iso_hom,
    π_comp_cokernelComparison_assoc, ← G.map_comp_assoc, kernel.lift_ι,
    Abelian.coimage_image_factorisation, lift_comp_kernelIsoOfEq_hom_assoc,
    kernelIsIsoComp_hom, kernel.lift_ι_assoc, kernelIsoOfEq_hom_comp_ι_assoc,
    kernelComparison_comp_ι_assoc, π_comp_cokernelIsoOfEq_hom_assoc,
    asIso_hom, NatIso.inv_inv_app] using NatIso.naturality_1 i f


/-- If `C` is an additive category, `D` is an abelian category,
we have `F : C ⥤ D` `G : D ⥤ C` (both preserving zero morphisms),
`G` is left exact (that is, preserves finite limits),
and further we have `adj : G ⊣ F` and `i : F ⋙ G ≅ 𝟭 C`,
then `C` is also abelian.

See <https://stacks.math.columbia.edu/tag/03A3>
-/
def abelianOfAdjunction {C : Type u₁} [Category.{v₁} C] [Preadditive C] [HasFiniteProducts C]
    {D : Type u₂} [Category.{v₂} D] [Abelian D] (F : C ⥤ D) [Functor.PreservesZeroMorphisms F]
    (G : D ⥤ C) [Functor.PreservesZeroMorphisms G] [PreservesFiniteLimits G] (i : F ⋙ G ≅ 𝟭 C)
    (adj : G ⊣ F) : Abelian C := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Limits.HasFiniteProducts C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    G : CategoryTheory.Functor D C
    inst✝¹ : G.PreservesZeroMorphisms
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
    i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
    adj : CategoryTheory.Adjunction G F
    ⊢ CategoryTheory.Abelian C
  -/
  haveI := hasKernels F G i
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Limits.HasFiniteProducts C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    G : CategoryTheory.Functor D C
    inst✝¹ : G.PreservesZeroMorphisms
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
    i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
    adj : CategoryTheory.Adjunction G F
    this : CategoryTheory.Limits.HasKernels C
    ⊢ CategoryTheory.Abelian C
  -/
  haveI := hasCokernels F G i adj
  have : ∀ {X Y : C} (f : X ⟶ Y), IsIso (Abelian.coimageImageComparison f) := by
    intro X Y f
    rw [← coimageIsoImage_hom F G i adj f]
    infer_instance
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Limits.HasFiniteProducts C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    G : CategoryTheory.Functor D C
    inst✝¹ : G.PreservesZeroMorphisms
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G
    i : CategoryTheory.Iso (F.comp G) (CategoryTheory.Functor.id C)
    adj : CategoryTheory.Adjunction G F
    this✝¹ : CategoryTheory.Limits.HasKernels C
    this✝ : CategoryTheory.Limits.HasCokernels C
    this : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheory. …
    ⊢ CategoryTheory.Abelian C
  -/
  apply Abelian.ofCoimageImageComparisonIsIso
  /-
    🎉 no goals
  -/


/-- If `C` is an additive category equivalent to an abelian category `D`
via a functor that preserves zero morphisms,
then `C` is also abelian.
-/
def abelianOfEquivalence {C : Type u₁} [Category.{v₁} C] [Preadditive C] [HasFiniteProducts C]
    {D : Type u₂} [Category.{v₂} D] [Abelian D] (F : C ⥤ D) [Functor.PreservesZeroMorphisms F]
    [F.IsEquivalence] : Abelian C :=
  abelianOfAdjunction F F.inv F.asEquivalence.unitIso.symm F.asEquivalence.symm.toAdjunction


noncomputable instance homGroup (P Q : ShrinkHoms C) : AddCommGroup (P ⟶ Q : Type w) :=
  Equiv.addCommGroup (equivShrink _).symm


lemma functor_map_add {P Q : C} (f g : P ⟶ Q) :
    (functor C).map (f + g) =
      (functor C).map f + (functor C).map g := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.LocallySmall.{w, u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    P Q : C
    f g : Quiver.Hom P Q
    ⊢ Eq ((CategoryTheory.ShrinkHoms.functor C).map (HAdd.hAdd f g)) (HAdd.hAdd (( …
  -/
  exact map_add (equivShrink.{w} (P ⟶ Q)).symm.addEquiv.symm f g
  /-
    🎉 no goals
  -/


lemma inverse_map_add {P Q : ShrinkHoms C} (f g : P ⟶ Q) :
    (inverse C).map (f + g) =
      (inverse C).map f + (ShrinkHoms.inverse C).map g :=
  map_add (equivShrink.{w} (P.fromShrinkHoms ⟶ Q.fromShrinkHoms)).symm.addEquiv f g


noncomputable instance preadditive :
    Preadditive.{w} (ShrinkHoms C) where
  homGroup := homGroup
  add_comp _ _ _ _ _ _ := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.133013, u_1} C
      inst✝¹ : CategoryTheory.LocallySmall.{w, ?u.133013, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      x✝⁵ x✝⁴ x✝³ : CategoryTheory.ShrinkHoms.{u_1} C
      x✝² x✝¹ : Quiver.Hom x✝⁵ x✝⁴
      x✝ : Quiver.Hom x✝⁴ x✝³
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (C …
    -/
    apply (inverse C).map_injective
    /-
      case a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.133013, u_1} C
      inst✝¹ : CategoryTheory.LocallySmall.{w, ?u.133013, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      x✝⁵ x✝⁴ x✝³ : CategoryTheory.ShrinkHoms.{u_1} C
      x✝² x✝¹ : Quiver.Hom x✝⁵ x✝⁴
      x✝ : Quiver.Hom x✝⁴ x✝³
      ⊢ Eq ((CategoryTheory.ShrinkHoms.inverse C).map (CategoryTheory.CategoryStruct …
    -/
    simp only [inverse_map_add, Functor.map_comp, Preadditive.add_comp]
    /-
      🎉 no goals
    -/
  comp_add _ _ _ _ _ _ := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.133013, u_1} C
      inst✝¹ : CategoryTheory.LocallySmall.{w, ?u.133013, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      x✝⁵ x✝⁴ x✝³ : CategoryTheory.ShrinkHoms.{u_1} C
      x✝² : Quiver.Hom x✝⁵ x✝⁴
      x✝¹ x✝ : Quiver.Hom x✝⁴ x✝³
      ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (C …
    -/
    apply (inverse C).map_injective
    /-
      case a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.133013, u_1} C
      inst✝¹ : CategoryTheory.LocallySmall.{w, ?u.133013, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      x✝⁵ x✝⁴ x✝³ : CategoryTheory.ShrinkHoms.{u_1} C
      x✝² : Quiver.Hom x✝⁵ x✝⁴
      x✝¹ x✝ : Quiver.Hom x✝⁴ x✝³
      ⊢ Eq ((CategoryTheory.ShrinkHoms.inverse C).map (CategoryTheory.CategoryStruct …
    -/
    simp only [inverse_map_add, Functor.map_comp, Preadditive.comp_add]
    /-
      🎉 no goals
    -/


instance : (inverse C).Additive where
                /-
                  C : Type u_1
                  inst✝² : CategoryTheory.Category.{u_2, u_1} C
                  inst✝¹ : CategoryTheory.LocallySmall.{w, u_2, u_1} C
                  inst✝ : CategoryTheory.Preadditive C
                  ⊢ ∀ {X Y : CategoryTheory.ShrinkHoms.{u_1} C} {f g : Quiver.Hom X Y}, Eq ((Cat …
                -/
  map_add := by apply inverse_map_add
                /-
                  🎉 no goals
                -/


instance : (functor C).Additive where
                /-
                  C : Type u_1
                  inst✝² : CategoryTheory.Category.{u_2, u_1} C
                  inst✝¹ : CategoryTheory.LocallySmall.{w, u_2, u_1} C
                  inst✝ : CategoryTheory.Preadditive C
                  ⊢ ∀ {X Y : C} {f g : Quiver.Hom X Y}, Eq ((CategoryTheory.ShrinkHoms.functor C …
                -/
  map_add := by apply functor_map_add
                /-
                  🎉 no goals
                -/


instance hasLimitsOfShape (J : Type*) [Category J]
    [HasLimitsOfShape J C] : HasLimitsOfShape.{_, _, w} J (ShrinkHoms C) :=
  Adjunction.hasLimitsOfShape_of_equivalence (inverse C)


instance hasFiniteLimits [HasFiniteLimits C] :
    HasFiniteLimits.{w} (ShrinkHoms C) := ⟨fun _ => inferInstance⟩


variable (C) in
noncomputable instance abelian [Abelian C] :
    Abelian.{w} (ShrinkHoms C) := abelianOfEquivalence (inverse C)


