variable (C) in
/-- The category of Ind-objects of `C`. -/
def Ind : Type (max u (v + 1)) :=
  ShrinkHoms (FullSubcategory (IsIndObject (C := C)))


noncomputable instance : Category.{v} (Ind C) :=
  inferInstanceAs <| Category.{v} (ShrinkHoms (FullSubcategory (IsIndObject (C := C))))


variable (C) in
/-- The defining properties of `Ind C` are that its morphisms live in `v` and that it is equivalent
to the full subcategory of `Cᵒᵖ ⥤ Type v` containing the ind-objects. -/
noncomputable def Ind.equivalence : Ind C ≌ FullSubcategory (IsIndObject (C := C)) :=
  (ShrinkHoms.equivalence _).symm


variable (C) in
/-- The canonical inclusion of ind-objects into presheaves. -/
protected noncomputable def Ind.inclusion : Ind C ⥤ Cᵒᵖ ⥤ Type v :=
  (Ind.equivalence C).functor ⋙ fullSubcategoryInclusion _


instance : (Ind.inclusion C).Full :=
  inferInstanceAs <| ((Ind.equivalence C).functor ⋙ fullSubcategoryInclusion _).Full


instance : (Ind.inclusion C).Faithful :=
  inferInstanceAs <| ((Ind.equivalence C).functor ⋙ fullSubcategoryInclusion _).Faithful


/-- The functor `Ind C ⥤ Cᵒᵖ ⥤ Type v` is fully faithful. -/
protected noncomputable def Ind.inclusion.fullyFaithful : (Ind.inclusion C).FullyFaithful :=
  .ofFullyFaithful _


/-- The inclusion of `C` into `Ind C` induced by the Yoneda embedding. -/
protected noncomputable def Ind.yoneda : C ⥤ Ind C :=
  FullSubcategory.lift _ CategoryTheory.yoneda isIndObject_yoneda ⋙ (Ind.equivalence C).inverse


instance : (Ind.yoneda (C := C)).Full :=
  inferInstanceAs <| Functor.Full <|
    FullSubcategory.lift _ CategoryTheory.yoneda isIndObject_yoneda ⋙ (Ind.equivalence C).inverse


instance : (Ind.yoneda (C := C)).Faithful :=
  inferInstanceAs <| Functor.Faithful <|
    FullSubcategory.lift _ CategoryTheory.yoneda isIndObject_yoneda ⋙ (Ind.equivalence C).inverse


/-- The functor `C ⥤ Ind C` is fully faithful. -/
protected noncomputable def Ind.yoneda.fullyFaithful : (Ind.yoneda (C := C)).FullyFaithful :=
  .ofFullyFaithful _


/-- The composition `C ⥤ Ind C ⥤ (Cᵒᵖ ⥤ Type v)` is just the Yoneda embedding. -/
noncomputable def Ind.yonedaCompInclusion : Ind.yoneda ⋙ Ind.inclusion C ≅ CategoryTheory.yoneda :=
  isoWhiskerLeft (FullSubcategory.lift _ _ _)
    (isoWhiskerRight (Ind.equivalence C).counitIso (fullSubcategoryInclusion _))


noncomputable instance {J : Type v} [SmallCategory J] [IsFiltered J] :
    CreatesColimitsOfShape J (Ind.inclusion C) :=
  letI _ : CreatesColimitsOfShape J (fullSubcategoryInclusion (IsIndObject (C := C))) :=
    createsColimitsOfShapeFullSubcategoryInclusion (closedUnderColimitsOfShape_of_colimit
      (isIndObject_colimit _ _))
  inferInstanceAs <|
    CreatesColimitsOfShape J ((Ind.equivalence C).functor ⋙ fullSubcategoryInclusion _)


instance : HasFilteredColimits (Ind C) where
  HasColimitsOfShape _ _ _ :=
    hasColimitsOfShape_of_hasColimitsOfShape_createsColimitsOfShape (Ind.inclusion C)


theorem Ind.isIndObject_inclusion_obj (X : Ind C) : IsIndObject ((Ind.inclusion C).obj X) :=
  X.2


/-- Pick a presentation of an ind-object `X` using choice. -/
noncomputable def Ind.presentation (X : Ind C) : IndObjectPresentation ((Ind.inclusion C).obj X) :=
  X.isIndObject_inclusion_obj.presentation


/-- An ind-object `X` is the colimit (in `Ind C`!) of the filtered diagram presenting it. -/
noncomputable def Ind.colimitPresentationCompYoneda (X : Ind C) :
    colimit (X.presentation.F ⋙ Ind.yoneda) ≅ X :=
  Ind.inclusion.fullyFaithful.isoEquiv.symm <| calc
    (Ind.inclusion C).obj (colimit (X.presentation.F ⋙ Ind.yoneda))
      ≅ colimit (X.presentation.F ⋙ Ind.yoneda ⋙ Ind.inclusion C) := preservesColimitIso _ _
    _ ≅ colimit (X.presentation.F ⋙ yoneda) :=
          HasColimit.isoOfNatIso (isoWhiskerLeft X.presentation.F Ind.yonedaCompInclusion)
    _ ≅ (Ind.inclusion C).obj X :=
          IsColimit.coconePointUniqueUpToIso (colimit.isColimit _) X.presentation.isColimit


instance : RepresentablyCoflat (Ind.yoneda (C := C)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.RepresentablyCoflat CategoryTheory.Ind.yoneda
  -/
  refine ⟨fun X => ?_⟩
  suffices IsFiltered (CostructuredArrow yoneda ((Ind.inclusion C).obj X)) from
    IsFiltered.of_equivalence
      ((CostructuredArrow.post Ind.yoneda (Ind.inclusion C) X).asEquivalence.trans
      (CostructuredArrow.mapNatIso Ind.yonedaCompInclusion)).symm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.Ind C
    ⊢ CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTheory.y …
  -/
  exact ((isIndObject_iff _).1 (Ind.isIndObject_inclusion_obj X)).1
  /-
    🎉 no goals
  -/


noncomputable instance : PreservesFiniteColimits (Ind.yoneda (C := C)) :=
  preservesFiniteColimits_of_coflat _


instance {α : Type v} [Finite α] [HasColimitsOfShape (Discrete α) C] :
    HasColimitsOfShape (Discrete α) (Ind C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    α : Type v
    inst✝¹ : Finite α
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) (Catego …
  -/
  refine ⟨fun F => ?_⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    α : Type v
    inst✝¹ : Finite α
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    F : CategoryTheory.Functor (CategoryTheory.Discrete α) (CategoryTheory.Ind C)
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  let I : α → Type v := fun s => (F.obj ⟨s⟩).presentation.I
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    α : Type v
    inst✝¹ : Finite α
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    F : CategoryTheory.Functor (CategoryTheory.Discrete α) (CategoryTheory.Ind C)
    I : α → Type v := fun s => (F.obj { as := s }).presentation.I
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  let G : ∀ s, I s ⥤ C := fun s => (F.obj ⟨s⟩).presentation.F
  let iso : Discrete.functor (fun s => Pi.eval I s ⋙ G s) ⋙
      (whiskeringRight _ _ _).obj Ind.yoneda ⋙ colim ≅ F := by
    refine Discrete.natIso (fun s => ?_)
    refine (Functor.Final.colimitIso (Pi.eval I s.as) (G s.as ⋙ Ind.yoneda)) ≪≫ ?_
    exact Ind.colimitPresentationCompYoneda _
  -- The actual proof happens during typeclass resolution in the following line, which deduces
  -- ```
  -- HasColimit Discrete.functor (fun s => Pi.eval I s ⋙ G s) ⋙
  --    (whiskeringRight _ _ _).obj Ind.yoneda ⋙ colim
  -- ```
  -- from the fact that finite limits commute with filtered colimits and from the fact that
  -- `Ind.yoneda` preserves finite colimits.
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    α : Type v
    inst✝¹ : Finite α
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    F : CategoryTheory.Functor (CategoryTheory.Discrete α) (CategoryTheory.Ind C)
    I : α → Type v := fun s => (F.obj { as := s }).presentation.I
    G : (s : α) → CategoryTheory.Functor (I s) C := fun s => (F.obj { as := s }).p …
    iso : CategoryTheory.Iso ((CategoryTheory.Discrete.functor fun s => (CategoryT …
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  apply hasColimitOfIso iso.symm
  /-
    🎉 no goals
  -/


instance [HasFiniteCoproducts C] : HasCoproducts.{v} (Ind C) :=
  have : HasFiniteCoproducts (Ind C) :=
    ⟨fun _ => hasColimitsOfShape_of_equivalence (Discrete.equivalence Equiv.ulift)⟩
  hasCoproducts_of_finite_and_filtered


