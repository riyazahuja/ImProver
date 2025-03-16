/--
Suppose `F : J ⥤ I ⥤ C` is a finite diagram in the functor category `I ⥤ C`, where `I` is small
and filtered. If `i : I`, we can apply the Yoneda embedding to `F(·, i)` to obtain a
diagram of presheaves `J ⥤ Cᵒᵖ ⥤ Type v`. Suppose that the limits of this diagram is always an
ind-object.

For `j : J` we can apply the Yoneda embedding to `F(j, ·)` and take colimits to obtain a finite
diagram `J ⥤ Cᵒᵖ ⥤ Type v` (which is actually a diagram `J ⥤ Ind C`). The theorem states that
the limit of this diagram is an ind-object.

This theorem will be used to construct equalizers in the category of ind-objects. It can be
interpreted as saying that ind-objects are closed under finite limits as long as the diagram
we are taking the limit of comes from a diagram in a functor category `I ⥤ C`. We will show (TODO)
that this is the case for any parallel pair of morphisms in `Ind C` and deduce that ind-objects
are closed under equalizers.

This is Proposition 6.1.16(i) in [Kashiwara2006].
-/
theorem isIndObject_limit_comp_yoneda_comp_colim
    (hF : ∀ i, IsIndObject (limit (F.flip.obj i ⋙ yoneda))) :
    IsIndObject (limit (F ⋙ (whiskeringRight _ _ _).obj yoneda ⋙ colim)) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝³ : CategoryTheory.SmallCategory I
    inst✝² : CategoryTheory.IsFiltered I
    J : Type
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor J (CategoryTheory.Functor I C)
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit …
    ⊢ CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit (F.comp (((Ca …
  -/
  let G : J ⥤ I ⥤ (Cᵒᵖ ⥤ Type v) := F ⋙ (whiskeringRight _ _ _).obj yoneda
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝³ : CategoryTheory.SmallCategory I
    inst✝² : CategoryTheory.IsFiltered I
    J : Type
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor J (CategoryTheory.Functor I C)
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit …
    G : CategoryTheory.Functor J (CategoryTheory.Functor I (CategoryTheory.Functor …
    ⊢ CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit (F.comp (((Ca …
  -/
  apply IsIndObject.map (HasLimit.isoOfNatIso (colimitFlipIsoCompColim G)).hom
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝³ : CategoryTheory.SmallCategory I
    inst✝² : CategoryTheory.IsFiltered I
    J : Type
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor J (CategoryTheory.Functor I C)
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit …
    G : CategoryTheory.Functor J (CategoryTheory.Functor I (CategoryTheory.Functor …
    ⊢ CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit (CategoryTheo …
  -/
  apply IsIndObject.map (colimitLimitIso G).hom
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝³ : CategoryTheory.SmallCategory I
    inst✝² : CategoryTheory.IsFiltered I
    J : Type
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor J (CategoryTheory.Functor I C)
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit …
    G : CategoryTheory.Functor J (CategoryTheory.Functor I (CategoryTheory.Functor …
    ⊢ CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.colimit (CategoryTh …
  -/
  apply isIndObject_colimit
  /-
    case hF
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    I : Type v
    inst✝³ : CategoryTheory.SmallCategory I
    inst✝² : CategoryTheory.IsFiltered I
    J : Type
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor J (CategoryTheory.Functor I C)
    hF : ∀ (i : I), CategoryTheory.Limits.IsIndObject (CategoryTheory.Limits.limit …
    G : CategoryTheory.Functor J (CategoryTheory.Functor I (CategoryTheory.Functor …
    ⊢ ∀ (i : I), CategoryTheory.Limits.IsIndObject ((CategoryTheory.Limits.limit G …
  -/
  exact fun i => IsIndObject.map (limitObjIsoLimitCompEvaluation _ _).inv (hF i)
  /-
    🎉 no goals
  -/


