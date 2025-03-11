/--
The light condensed object associated to a presheaf on `LightProfinite` which preserves finite
products and satisfies the equalizer condition.
-/
@[simps]
noncomputable def ofSheafLightProfinite (F : LightProfinite.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts F]
    (hF : EqualizerCondition F) : LightCondensed A where
  val := F
  cond := by
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} A
      F : CategoryTheory.Functor (Opposite LightProfinite) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      hF : CategoryTheory.regularTopology.EqualizerCondition F
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology LightProfin …
    -/
    rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition F]
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} A
      F : CategoryTheory.Functor (Opposite LightProfinite) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      hF : CategoryTheory.regularTopology.EqualizerCondition F
      ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.regula …
    -/
    exact ⟨⟨fun _ _ ↦ inferInstance⟩, hF⟩
    /-
      🎉 no goals
    -/


/--
The light condensed object associated to a presheaf on `LightProfinite` whose postcomposition with
the forgetful functor preserves finite products and satisfies the equalizer condition.
-/
@[simps]
noncomputable def ofSheafForgetLightProfinite
    [ConcreteCategory A] [ReflectsFiniteLimits (CategoryTheory.forget A)]
    (F : LightProfinite.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts (F ⋙ CategoryTheory.forget A)]
    (hF : EqualizerCondition (F ⋙ CategoryTheory.forget A)) : LightCondensed A where
  val := F
  cond := by
    /-
      A : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.2407, u_1} A
      inst✝² : CategoryTheory.ConcreteCategory A
      inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
      F : CategoryTheory.Functor (Opposite LightProfinite) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
      hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology LightProfin …
    -/
    apply isSheaf_coherent_of_hasPullbacks_of_comp F (CategoryTheory.forget A)
    /-
      A : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.2407, u_1} A
      inst✝² : CategoryTheory.ConcreteCategory A
      inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
      F : CategoryTheory.Functor (Opposite LightProfinite) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
      hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology LightProfin …
    -/
    rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition]
    /-
      A : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.2407, u_1} A
      inst✝² : CategoryTheory.ConcreteCategory A
      inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
      F : CategoryTheory.Functor (Opposite LightProfinite) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
      hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
      ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory.f …
    -/
    exact ⟨⟨fun _ _ ↦ inferInstance⟩, hF⟩
    /-
      🎉 no goals
    -/


/-- A light condensed object satisfies the equalizer condition. -/
theorem equalizerCondition (X : LightCondensed A) : EqualizerCondition X.val :=
  isSheaf_iff_preservesFiniteProducts_and_equalizerCondition X.val |>.mp X.cond |>.2


/-- A light condensed object preserves finite products. -/
noncomputable instance (X : LightCondensed A) : PreservesFiniteProducts X.val :=
  isSheaf_iff_preservesFiniteProducts_and_equalizerCondition X.val |>.mp X.cond |>.1


/-- A `LightCondSet` version of `LightCondensed.ofSheafLightProfinite`. -/
noncomputable abbrev ofSheafLightProfinite (F : LightProfinite.{u}ᵒᵖ ⥤ Type u)
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : LightCondSet :=
  LightCondensed.ofSheafLightProfinite F hF


/-- A `LightCondAb` version of `LightCondensed.ofSheafLightProfinite`. -/
noncomputable abbrev ofSheafLightProfinite (F : LightProfinite.{u}ᵒᵖ ⥤ ModuleCat.{u} R)
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : LightCondMod.{u} R :=
  LightCondensed.ofSheafLightProfinite F hF


/-- A `LightCondAb` version of `LightCondensed.ofSheafLightProfinite`. -/
noncomputable abbrev ofSheafLightProfinite (F : LightProfiniteᵒᵖ ⥤ ModuleCat ℤ)
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : LightCondAb :=
  LightCondMod.ofSheafLightProfinite ℤ F hF


