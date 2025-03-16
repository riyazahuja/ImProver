/-- The data that witnesses that a presheaf `A` is an ind-object. It consists of a small
    filtered indexing category `I`, a diagram `F : I ⥤ C` and the data for a colimit cocone on
    `F ⋙ yoneda : I ⥤ Cᵒᵖ ⥤ Type v` with cocone point `A`. -/
structure IndObjectPresentation (A : Cᵒᵖ ⥤ Type v) where
  /-- The indexing category of the filtered colimit presentation -/
  I : Type v
  /-- The indexing category of the filtered colimit presentation -/
  [ℐ : SmallCategory I]
  [hI : IsFiltered I]
  /-- The diagram of the filtered colimit presentation -/
  F : I ⥤ C
  /-- Use `IndObjectPresentation.cocone` instead. -/
  ι : F ⋙ yoneda ⟶ (Functor.const I).obj A
  /-- Use `IndObjectPresentation.coconeIsColimit` instead. -/
  isColimit : IsColimit (Cocone.mk A ι)


/-- Alternative constructor for `IndObjectPresentation` taking a cocone instead of its defining
    natural transformation. -/
@[simps]
def ofCocone {I : Type v} [SmallCategory I] [IsFiltered I] {F : I ⥤ C}
    (c : Cocone (F ⋙ yoneda)) (hc : IsColimit c) : IndObjectPresentation c.pt where
  I := I
  F := F
  ι := c.ι
  isColimit := hc


instance : SmallCategory P.I := P.ℐ

instance : IsFiltered P.I := P.hI


/-- The (colimit) cocone with cocone point `A`. -/
@[simps pt]
def cocone : Cocone (P.F ⋙ yoneda) where
  pt := A
  ι := P.ι


/-- `P.cocone` is a colimit cocone. -/
def coconeIsColimit : IsColimit P.cocone :=
  P.isColimit


/-- If `A` and `B` are isomorphic, then an ind-object presentation of `A` can be extended to an
    ind-object presentation of `B`. -/
@[simps!]
noncomputable def extend {A B : Cᵒᵖ ⥤ Type v} (P : IndObjectPresentation A) (η : A ⟶ B) [IsIso η] :
    IndObjectPresentation B :=
                                                                 /-
                                                                   C : Type u
                                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                   A✝ : CategoryTheory.Functor (Opposite C) (Type v)
                                                                   P✝ : CategoryTheory.Limits.IndObjectPresentation A✝
                                                                   A B : CategoryTheory.Functor (Opposite C) (Type v)
                                                                   P : CategoryTheory.Limits.IndObjectPresentation A
                                                                   η : Quiver.Hom A B
                                                                   inst✝ : CategoryTheory.IsIso η
                                                                   ⊢ Quiver.Hom P.cocone.pt ?m.8288
                                                                 -/
  .ofCocone (P.cocone.extend η) (P.coconeIsColimit.extendIso (by exact η))
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The canonical comparison functor between the indexing category of the presentation and the
    comma category `CostructuredArrow yoneda A`. This functor is always final. -/
@[simps! obj_left obj_right_as obj_hom map_left]
def toCostructuredArrow : P.I ⥤ CostructuredArrow yoneda A :=
  P.cocone.toCostructuredArrow ⋙ CostructuredArrow.pre _ _ _


instance : P.toCostructuredArrow.Final :=
  Presheaf.final_toCostructuredArrow_comp_pre _ P.coconeIsColimit


/-- Representable presheaves are (trivially) ind-objects. -/
@[simps]
def yoneda (X : C) : IndObjectPresentation (yoneda.obj X) where
  I := Discrete PUnit.{v + 1}
  F := Functor.fromPUnit X
  ι := { app := fun _ => 𝟙 _ }
  isColimit :=
    { desc := fun s => s.ι.app ⟨PUnit.unit⟩
      uniq := fun _ _ h => h ⟨PUnit.unit⟩ }


/-- A presheaf is called an ind-object if it can be written as a filtered colimit of representable
    presheaves. -/
structure IsIndObject (A : Cᵒᵖ ⥤ Type v) : Prop where
  mk' :: nonempty_presentation : Nonempty (IndObjectPresentation A)


theorem IsIndObject.mk {A : Cᵒᵖ ⥤ Type v} (P : IndObjectPresentation A) : IsIndObject A :=
  ⟨⟨P⟩⟩


/-- Representable presheaves are (trivially) ind-objects. -/
theorem isIndObject_yoneda (X : C) : IsIndObject (yoneda.obj X) :=
  .mk <| IndObjectPresentation.yoneda X


theorem map {A B : Cᵒᵖ ⥤ Type v} (η : A ⟶ B) [IsIso η] : IsIndObject A → IsIndObject B
  | ⟨⟨P⟩⟩ => ⟨⟨P.extend η⟩⟩


theorem iff_of_iso {A B : Cᵒᵖ ⥤ Type v} (η : A ⟶ B) [IsIso η] : IsIndObject A ↔ IsIndObject B :=
  ⟨.map η, .map (inv η)⟩


instance : ClosedUnderIsomorphisms (IsIndObject (C := C)) where
  of_iso i h := h.map i.hom


/-- Pick a presentation for an ind-object using choice. -/
noncomputable def presentation : IsIndObject A → IndObjectPresentation A
  | ⟨P⟩ => P.some


theorem isFiltered (h : IsIndObject A) : IsFiltered (CostructuredArrow yoneda A) :=
  IsFiltered.of_final h.presentation.toCostructuredArrow


theorem finallySmall (h : IsIndObject A) : FinallySmall.{v} (CostructuredArrow yoneda A) :=
  FinallySmall.mk' h.presentation.toCostructuredArrow


theorem isIndObject_of_isFiltered_of_finallySmall (A : Cᵒᵖ ⥤ Type v)
    [IsFiltered (CostructuredArrow yoneda A)] [FinallySmall.{v} (CostructuredArrow yoneda A)] :
    IsIndObject A := by
  have h₁ : (factoring (fromFinalModel (CostructuredArrow yoneda A)) ⋙
      inclusion (fromFinalModel (CostructuredArrow yoneda A))).Final := Functor.final_of_natIso
    (factoringCompInclusion (fromFinalModel <| CostructuredArrow yoneda A)).symm
  have h₂ : Functor.Final (inclusion (fromFinalModel (CostructuredArrow yoneda A))) :=
    Functor.final_of_comp_full_faithful' (factoring _) (inclusion _)
  let c := (Presheaf.tautologicalCocone A).whisker
    (inclusion (fromFinalModel (CostructuredArrow yoneda A)))
  let hc : IsColimit c := (Functor.Final.isColimitWhiskerEquiv _ _).symm
    (Presheaf.isColimitTautologicalCocone A)
  have hq : Nonempty (FinalModel (CostructuredArrow yoneda A)) := Nonempty.map
    (Functor.Final.lift (fromFinalModel (CostructuredArrow yoneda A))) IsFiltered.nonempty
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    inst✝¹ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryT …
    inst✝ : CategoryTheory.FinallySmall (CategoryTheory.CostructuredArrow Category …
    h₁ : ((CategoryTheory.IsFiltered.SmallFilteredIntermediate.factoring (Category …
    h₂ : (CategoryTheory.IsFiltered.SmallFilteredIntermediate.inclusion (CategoryT …
    c : CategoryTheory.Limits.Cocone ((CategoryTheory.IsFiltered.SmallFilteredInte …
    hc : CategoryTheory.Limits.IsColimit c := (CategoryTheory.Functor.Final.isColi …
    hq : Nonempty (CategoryTheory.FinalModel (CategoryTheory.CostructuredArrow Cat …
    ⊢ CategoryTheory.Limits.IsIndObject A
  -/
  exact ⟨_, inclusion (fromFinalModel _) ⋙ CostructuredArrow.proj yoneda A, c.ι, hc⟩
  /-
    🎉 no goals
  -/


/-- The recognition theorem for ind-objects: `A : Cᵒᵖ ⥤ Type v` is an ind-object if and only if
    `CostructuredArrow yoneda A` is filtered and finally `v`-small.

    Theorem 6.1.5 of [Kashiwara2006] -/
theorem isIndObject_iff (A : Cᵒᵖ ⥤ Type v) : IsIndObject A ↔
    (IsFiltered (CostructuredArrow yoneda A) ∧ FinallySmall.{v} (CostructuredArrow yoneda A)) :=
  ⟨fun h => ⟨h.isFiltered, h.finallySmall⟩,
   fun ⟨_, _⟩ => isIndObject_of_isFiltered_of_finallySmall A⟩


/-- If a limit already exists in `C`, then the limit of the image of the diagram under the Yoneda
embedding is an ind-object. -/
theorem isIndObject_limit_comp_yoneda {J : Type u'} [Category.{v'} J] (F : J ⥤ C) [HasLimit F] :
    IsIndObject (limit (F ⋙ yoneda)) :=
  IsIndObject.map (preservesLimitIso yoneda F).hom (isIndObject_yoneda (limit F))


