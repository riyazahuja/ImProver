/-- The condensed object associated to a finite-product-preserving presheaf on `Stonean`. -/
noncomputable def ofSheafStonean
    [∀ X, HasLimitsOfShape (StructuredArrow X Stonean.toCompHaus.op) A]
    (F : Stonean.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts F] :
    Condensed A :=
  StoneanCompHaus.equivalence A |>.functor.obj {
    val := F
    cond := by
      /-
        A : Type u_1
        inst✝² : CategoryTheory.Category.{?u.13, u_1} A
        inst✝¹ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        F : CategoryTheory.Functor (Opposite Stonean) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology Stonean) F
      -/
      rw [isSheaf_iff_preservesFiniteProducts_of_projective F]
      /-
        A : Type u_1
        inst✝² : CategoryTheory.Category.{?u.13, u_1} A
        inst✝¹ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        F : CategoryTheory.Functor (Opposite Stonean) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
        ⊢ CategoryTheory.Limits.PreservesFiniteProducts F
      -/
      exact ⟨fun _ _ ↦ inferInstance⟩ }
      /-
        🎉 no goals
      -/


/--
The condensed object associated to a presheaf on `Stonean` whose postcomposition with the
forgetful functor preserves finite products.
-/
noncomputable def ofSheafForgetStonean
    [∀ X, HasLimitsOfShape (StructuredArrow X Stonean.toCompHaus.op) A]
    [ConcreteCategory A] [ReflectsFiniteProducts (CategoryTheory.forget A)]
    (F : Stonean.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts (F ⋙ CategoryTheory.forget A)] :
    Condensed A :=
  StoneanCompHaus.equivalence A |>.functor.obj {
    val := F
    cond := by
      /-
        A : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.2065, u_1} A
        inst✝³ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        inst✝² : CategoryTheory.ConcreteCategory A
        inst✝¹ : CategoryTheory.Limits.ReflectsFiniteProducts (CategoryTheory.forget A)
        F : CategoryTheory.Functor (Opposite Stonean) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology Stonean) F
      -/
      apply isSheaf_coherent_of_projective_of_comp F (CategoryTheory.forget A)
      /-
        A : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.2065, u_1} A
        inst✝³ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        inst✝² : CategoryTheory.ConcreteCategory A
        inst✝¹ : CategoryTheory.Limits.ReflectsFiniteProducts (CategoryTheory.forget A)
        F : CategoryTheory.Functor (Opposite Stonean) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology Stonean) (F …
      -/
      rw [isSheaf_iff_preservesFiniteProducts_of_projective]
      /-
        A : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.2065, u_1} A
        inst✝³ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        inst✝² : CategoryTheory.ConcreteCategory A
        inst✝¹ : CategoryTheory.Limits.ReflectsFiniteProducts (CategoryTheory.forget A)
        F : CategoryTheory.Functor (Opposite Stonean) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
        ⊢ CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory.forget …
      -/
      exact ⟨fun _ _ ↦ inferInstance⟩ }
      /-
        🎉 no goals
      -/


/--
The condensed object associated to a presheaf on `Profinite` which preserves finite products and
satisfies the equalizer condition.
-/
noncomputable def ofSheafProfinite
    [∀ X, HasLimitsOfShape (StructuredArrow X profiniteToCompHaus.op) A]
    (F : Profinite.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts F]
    (hF : EqualizerCondition F) : Condensed A :=
  ProfiniteCompHaus.equivalence A |>.functor.obj {
    val := F
    cond := by
      /-
        A : Type u_1
        inst✝² : CategoryTheory.Category.{?u.6208, u_1} A
        inst✝¹ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        F : CategoryTheory.Functor (Opposite Profinite) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
        hF : CategoryTheory.regularTopology.EqualizerCondition F
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology Profinite) F
      -/
      rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition F]
      /-
        A : Type u_1
        inst✝² : CategoryTheory.Category.{?u.6208, u_1} A
        inst✝¹ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        F : CategoryTheory.Functor (Opposite Profinite) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
        hF : CategoryTheory.regularTopology.EqualizerCondition F
        ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.regula …
      -/
      exact ⟨⟨fun _ _ ↦ inferInstance⟩, hF⟩ }
      /-
        🎉 no goals
      -/


/--
The condensed object associated to a presheaf on `Profinite` whose postcomposition with the
forgetful functor preserves finite products and satisfies the equalizer condition.
-/
noncomputable def ofSheafForgetProfinite
    [∀ X, HasLimitsOfShape (StructuredArrow X profiniteToCompHaus.op) A]
    [ConcreteCategory A] [ReflectsFiniteLimits (CategoryTheory.forget A)]
    (F : Profinite.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts (F ⋙ CategoryTheory.forget A)]
    (hF : EqualizerCondition (F ⋙ CategoryTheory.forget A)) :
    Condensed A :=
  ProfiniteCompHaus.equivalence A |>.functor.obj {
    val := F
    cond := by
      /-
        A : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.9269, u_1} A
        inst✝³ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        inst✝² : CategoryTheory.ConcreteCategory A
        inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
        F : CategoryTheory.Functor (Opposite Profinite) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
        hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology Profinite) F
      -/
      apply isSheaf_coherent_of_hasPullbacks_of_comp F (CategoryTheory.forget A)
      /-
        A : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.9269, u_1} A
        inst✝³ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        inst✝² : CategoryTheory.ConcreteCategory A
        inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
        F : CategoryTheory.Functor (Opposite Profinite) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
        hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
        ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology Profinite)  …
      -/
      rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition]
      /-
        A : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.9269, u_1} A
        inst✝³ : ∀ (X : Opposite CompHaus), CategoryTheory.Limits.HasLimitsOfShape (Ca …
        inst✝² : CategoryTheory.ConcreteCategory A
        inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
        F : CategoryTheory.Functor (Opposite Profinite) A
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
        hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
        ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory.f …
      -/
      exact ⟨⟨fun _ _ ↦ inferInstance⟩, hF⟩ }
      /-
        🎉 no goals
      -/


/--
The condensed object associated to a presheaf on `CompHaus` which preserves finite products and
satisfies the equalizer condition.
-/
noncomputable def ofSheafCompHaus
    (F : CompHaus.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts F]
    (hF : EqualizerCondition F) : Condensed A where
  val := F
  cond := by
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.15107, u_1} A
      F : CategoryTheory.Functor (Opposite CompHaus) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      hF : CategoryTheory.regularTopology.EqualizerCondition F
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology CompHaus) F
    -/
    rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition F]
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.15107, u_1} A
      F : CategoryTheory.Functor (Opposite CompHaus) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      hF : CategoryTheory.regularTopology.EqualizerCondition F
      ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.regula …
    -/
    exact ⟨⟨fun _ _ ↦ inferInstance⟩, hF⟩
    /-
      🎉 no goals
    -/


/--
The condensed object associated to a presheaf on `CompHaus` whose postcomposition with the
forgetful functor preserves finite products and satisfies the equalizer condition.
-/
noncomputable def ofSheafForgetCompHaus
    [ConcreteCategory A] [ReflectsFiniteLimits (CategoryTheory.forget A)]
    (F : CompHaus.{u}ᵒᵖ ⥤ A) [PreservesFiniteProducts (F ⋙ CategoryTheory.forget A)]
    (hF : EqualizerCondition (F ⋙ CategoryTheory.forget A)) : Condensed A where
  val := F
  cond := by
    /-
      A : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.17187, u_1} A
      inst✝² : CategoryTheory.ConcreteCategory A
      inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
      F : CategoryTheory.Functor (Opposite CompHaus) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
      hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology CompHaus) F
    -/
    apply isSheaf_coherent_of_hasPullbacks_of_comp F (CategoryTheory.forget A)
    /-
      A : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.17187, u_1} A
      inst✝² : CategoryTheory.ConcreteCategory A
      inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
      F : CategoryTheory.Functor (Opposite CompHaus) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
      hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology CompHaus) ( …
    -/
    rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition]
    /-
      A : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.17187, u_1} A
      inst✝² : CategoryTheory.ConcreteCategory A
      inst✝¹ : CategoryTheory.Limits.ReflectsFiniteLimits (CategoryTheory.forget A)
      F : CategoryTheory.Functor (Opposite CompHaus) A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory. …
      hF : CategoryTheory.regularTopology.EqualizerCondition (F.comp (CategoryTheory …
      ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryTheory.f …
    -/
    exact ⟨⟨fun _ _ ↦ inferInstance⟩, hF⟩
    /-
      🎉 no goals
    -/


/-- A condensed object satisfies the equalizer condition. -/
theorem equalizerCondition (X : Condensed A) : EqualizerCondition X.val :=
  isSheaf_iff_preservesFiniteProducts_and_equalizerCondition X.val |>.mp X.cond |>.2


/-- A condensed object preserves finite products. -/
noncomputable instance (X : Condensed A) : PreservesFiniteProducts X.val :=
  isSheaf_iff_preservesFiniteProducts_and_equalizerCondition X.val |>.mp
    X.cond |>.1


/-- A condensed object regarded as a sheaf on `Profinite` preserves finite products. -/
noncomputable instance (X : Sheaf (coherentTopology Profinite.{u}) A) :
    PreservesFiniteProducts X.val :=
  isSheaf_iff_preservesFiniteProducts_and_equalizerCondition X.val |>.mp
    X.cond |>.1


/-- A condensed object regarded as a sheaf on `Profinite` satisfies the equalizer condition. -/
theorem equalizerCondition_profinite (X : Sheaf (coherentTopology Profinite.{u}) A) :
    EqualizerCondition X.val :=
  isSheaf_iff_preservesFiniteProducts_and_equalizerCondition X.val |>.mp X.cond |>.2


/-- A condensed object regarded as a sheaf on `Stonean` preserves finite products. -/
noncomputable instance (X : Sheaf (coherentTopology Stonean.{u}) A) :
    PreservesFiniteProducts X.val :=
  isSheaf_iff_preservesFiniteProducts_of_projective X.val |>.mp X.cond


/-- A `CondensedSet` version of `Condensed.ofSheafStonean`. -/
noncomputable abbrev ofSheafStonean (F : Stonean.{u}ᵒᵖ ⥤ Type (u+1)) [PreservesFiniteProducts F] :
    CondensedSet :=
  Condensed.ofSheafStonean F


/-- A `CondensedSet` version of `Condensed.ofSheafProfinite`. -/
noncomputable abbrev ofSheafProfinite (F : Profinite.{u}ᵒᵖ ⥤ Type (u+1))
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : CondensedSet :=
  Condensed.ofSheafProfinite F hF


/-- A `CondensedSet` version of `Condensed.ofSheafCompHaus`. -/
noncomputable abbrev ofSheafCompHaus (F : CompHaus.{u}ᵒᵖ ⥤ Type (u+1))
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : CondensedSet :=
  Condensed.ofSheafCompHaus F hF


/-- A `CondensedMod` version of `Condensed.ofSheafStonean`. -/
noncomputable abbrev ofSheafStonean (F : Stonean.{u}ᵒᵖ ⥤ ModuleCat.{u+1} R)
    [PreservesFiniteProducts F] : CondensedMod R :=
  haveI : HasLimitsOfSize.{u, u+1} (ModuleCat R) := hasLimitsOfSizeShrink.{u, u+1, u+1, u+1} _
  Condensed.ofSheafStonean F


/-- A `CondensedMod` version of `Condensed.ofSheafProfinite`. -/
noncomputable abbrev ofSheafProfinite (F : Profinite.{u}ᵒᵖ ⥤ ModuleCat.{u+1} R)
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : CondensedMod R :=
  haveI : HasLimitsOfSize.{u, u+1} (ModuleCat R) := hasLimitsOfSizeShrink.{u, u+1, u+1, u+1} _
  Condensed.ofSheafProfinite F hF


/-- A `CondensedMod` version of `Condensed.ofSheafCompHaus`. -/
noncomputable abbrev ofSheafCompHaus (F : CompHaus.{u}ᵒᵖ ⥤ ModuleCat.{u+1} R)
    [PreservesFiniteProducts F] (hF : EqualizerCondition F) : CondensedMod R :=
  Condensed.ofSheafCompHaus F hF


