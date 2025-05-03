/-- The functor `G : D ⥤ C` satisfies the *solution set condition* if for every `A : C`, there is a
family of morphisms `{f_i : A ⟶ G (B_i) // i ∈ ι}` such that given any morphism `h : A ⟶ G X`,
there is some `i ∈ ι` such that `h` factors through `f_i`.

The key part of this definition is that the indexing set `ι` lives in `Type v`, where `v` is the
universe of morphisms of the category: this is the "smallness" condition which allows the general
adjoint functor theorem to go through.
-/
def SolutionSetCondition {D : Type u} [Category.{v} D] (G : D ⥤ C) : Prop :=
  ∀ A : C,
    ∃ (ι : Type v) (B : ι → D) (f : ∀ i : ι, A ⟶ G.obj (B i)),
      ∀ (X) (h : A ⟶ G.obj X), ∃ (i : ι) (g : B i ⟶ X), f i ≫ G.map g = h


/-- If `G : D ⥤ C` is a right adjoint it satisfies the solution set condition. -/
theorem solutionSetCondition_of_isRightAdjoint [G.IsRightAdjoint] : SolutionSetCondition G := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝ : G.IsRightAdjoint
    ⊢ CategoryTheory.SolutionSetCondition G
  -/
  intro A
  refine
    ⟨PUnit, fun _ => G.leftAdjoint.obj A, fun _ => (Adjunction.ofIsRightAdjoint G).unit.app A, ?_⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝ : G.IsRightAdjoint
    A : C
    ⊢ ∀ (X : D) (h : Quiver.Hom A (G.obj X)), Exists fun i => Exists fun g => Eq ( …
  -/
  intro B h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝ : G.IsRightAdjoint
    A : C
    B : D
    h : Quiver.Hom A (G.obj B)
    ⊢ Exists fun i => Exists fun g => Eq (CategoryTheory.CategoryStruct.comp ((fun …
  -/
  refine ⟨PUnit.unit, ((Adjunction.ofIsRightAdjoint G).homEquiv _ _).symm h, ?_⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝ : G.IsRightAdjoint
    A : C
    B : D
    h : Quiver.Hom A (G.obj B)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => (CategoryTheory.Adjunction …
  -/
  rw [← Adjunction.homEquiv_unit, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- The general adjoint functor theorem says that if `G : D ⥤ C` preserves limits and `D` has them,
if `G` satisfies the solution set condition then `G` is a right adjoint.
-/
lemma isRightAdjoint_of_preservesLimits_of_solutionSetCondition [HasLimits D]
    [PreservesLimits G] (hG : SolutionSetCondition G) : G.IsRightAdjoint := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    hG : CategoryTheory.SolutionSetCondition G
    ⊢ G.IsRightAdjoint
  -/
  refine @isRightAdjointOfStructuredArrowInitials _ _ _ _ G ?_
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    hG : CategoryTheory.SolutionSetCondition G
    ⊢ ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.StructuredArrow  …
  -/
  intro A
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    hG : CategoryTheory.SolutionSetCondition G
    A : C
    ⊢ CategoryTheory.Limits.HasInitial (CategoryTheory.StructuredArrow A G)
  -/
  specialize hG A
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    A : C
    hG : Exists fun ι => Exists fun B => Exists fun f => ∀ (X : D) (h : Quiver.Hom …
    ⊢ CategoryTheory.Limits.HasInitial (CategoryTheory.StructuredArrow A G)
  -/
  choose ι B f g using hG
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    A : C
    ι : Type v
    B : ι → D
    f : (i : ι) → Quiver.Hom A (G.obj (B i))
    g : ∀ (X : D) (h : Quiver.Hom A (G.obj X)), Exists fun i => Exists fun g => Eq …
    ⊢ CategoryTheory.Limits.HasInitial (CategoryTheory.StructuredArrow A G)
  -/
  let B' : ι → StructuredArrow A G := fun i => StructuredArrow.mk (f i)
  have hB' : ∀ A' : StructuredArrow A G, ∃ i, Nonempty (B' i ⟶ A') := by
    intro A'
    obtain ⟨i, _, t⟩ := g _ A'.hom
    exact ⟨i, ⟨StructuredArrow.homMk _ t⟩⟩
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    A : C
    ι : Type v
    B : ι → D
    f : (i : ι) → Quiver.Hom A (G.obj (B i))
    g : ∀ (X : D) (h : Quiver.Hom A (G.obj X)), Exists fun i => Exists fun g => Eq …
    B' : ι → CategoryTheory.StructuredArrow A G := fun i => CategoryTheory.Structu …
    hB' : ∀ (A' : CategoryTheory.StructuredArrow A G), Exists fun i => Nonempty (Q …
    ⊢ CategoryTheory.Limits.HasInitial (CategoryTheory.StructuredArrow A G)
  -/
  obtain ⟨T, hT⟩ := has_weakly_initial_of_weakly_initial_set_and_hasProducts hB'
  /-
    case intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u
    inst✝² : CategoryTheory.Category.{v, u} D
    G : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.Limits.HasLimits D
    inst✝ : CategoryTheory.Limits.PreservesLimits G
    A : C
    ι : Type v
    B : ι → D
    f : (i : ι) → Quiver.Hom A (G.obj (B i))
    g : ∀ (X : D) (h : Quiver.Hom A (G.obj X)), Exists fun i => Exists fun g => Eq …
    B' : ι → CategoryTheory.StructuredArrow A G := fun i => CategoryTheory.Structu …
    hB' : ∀ (A' : CategoryTheory.StructuredArrow A G), Exists fun i => Nonempty (Q …
    T : CategoryTheory.StructuredArrow A G
    hT : ∀ (X : CategoryTheory.StructuredArrow A G), Nonempty (Quiver.Hom T X)
    ⊢ CategoryTheory.Limits.HasInitial (CategoryTheory.StructuredArrow A G)
  -/
  apply hasInitial_of_weakly_initial_and_hasWideEqualizers hT
  /-
    🎉 no goals
  -/


/-- The special adjoint functor theorem: if `G : D ⥤ C` preserves limits and `D` is complete,
well-powered and has a small coseparating set, then `G` has a left adjoint.
-/
lemma isRightAdjoint_of_preservesLimits_of_isCoseparating [HasLimits D] [WellPowered.{v} D]
    {𝒢 : Set D} [Small.{v} 𝒢] (h𝒢 : IsCoseparating 𝒢) (G : D ⥤ C) [PreservesLimits G] :
    G.IsRightAdjoint :=
  have : ∀ A, HasInitial (StructuredArrow A G) := fun A =>
    hasInitial_of_isCoseparating (StructuredArrow.isCoseparating_proj_preimage A G h𝒢)
  isRightAdjointOfStructuredArrowInitials _


/-- The special adjoint functor theorem: if `F : C ⥤ D` preserves colimits and `C` is cocomplete,
well-copowered and has a small separating set, then `F` has a right adjoint.
-/
lemma isLeftAdjoint_of_preservesColimits_of_isSeparating [HasColimits C] [WellPowered.{v} Cᵒᵖ]
    {𝒢 : Set C} [Small.{v} 𝒢] (h𝒢 : IsSeparating 𝒢) (F : C ⥤ D) [PreservesColimits F] :
    F.IsLeftAdjoint :=
  have : ∀ A, HasTerminal (CostructuredArrow F A) := fun A =>
    hasTerminal_of_isSeparating (CostructuredArrow.isSeparating_proj_preimage F A h𝒢)
  isLeftAdjoint_of_costructuredArrowTerminals _


/-- A consequence of the special adjoint functor theorem: if `C` is complete, well-powered and
    has a small coseparating set, then it is cocomplete. -/
theorem hasColimits_of_hasLimits_of_isCoseparating [HasLimits C] [WellPowered.{v} C] {𝒢 : Set C}
    [Small.{v} 𝒢] (h𝒢 : IsCoseparating 𝒢) : HasColimits C :=
  { has_colimits_of_shape := fun _ _ =>
      hasColimitsOfShape_iff_isRightAdjoint_const.2
        (isRightAdjoint_of_preservesLimits_of_isCoseparating h𝒢 _) }


/-- A consequence of the special adjoint functor theorem: if `C` is cocomplete, well-copowered and
    has a small separating set, then it is complete. -/
theorem hasLimits_of_hasColimits_of_isSeparating [HasColimits C] [WellPowered.{v} Cᵒᵖ] {𝒢 : Set C}
    [Small.{v} 𝒢] (h𝒢 : IsSeparating 𝒢) : HasLimits C :=
  { has_limits_of_shape := fun _ _ =>
      hasLimitsOfShape_iff_isLeftAdjoint_const.2
        (isLeftAdjoint_of_preservesColimits_of_isSeparating h𝒢 _) }


/-- A consequence of the special adjoint functor theorem: if `C` is complete, well-powered and
    has a separator, then it is complete. -/
theorem hasLimits_of_hasColimits_of_hasSeparator [HasColimits C] [HasSeparator C]
    [WellPowered.{v} Cᵒᵖ] : HasLimits C :=
  hasLimits_of_hasColimits_of_isSeparating <| isSeparator_separator C


/-- A consequence of the special adjoint functor theorem: if `C` is complete, well-powered and
    has a coseparator, then it is cocomplete. -/
theorem hasColimits_of_hasLimits_of_hasCoseparator [HasLimits C] [HasCoseparator C]
    [WellPowered.{v} C] : HasColimits C :=
  hasColimits_of_hasLimits_of_isCoseparating <| isCoseparator_coseparator C


