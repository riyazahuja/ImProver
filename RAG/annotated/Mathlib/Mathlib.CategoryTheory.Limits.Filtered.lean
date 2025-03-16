/-- `C` is filtered if and only if for every functor `F : J ⥤ C` from a finite category there is
    some `X : C` such that `lim Hom(F·, X)` is nonempty.

    Lemma 3.1.2 of [Kashiwara2006] -/
theorem IsFiltered.iff_nonempty_limit : IsFiltered C ↔
    ∀ {J : Type v} [SmallCategory J] [FinCategory J] (F : J ⥤ C),
      ∃ (X : C), Nonempty (limit (F.op ⋙ yoneda.obj X)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (CategoryTheory.IsFiltered C) (∀ {J : Type v} [inst : CategoryTheory.Sma …
  -/
  rw [IsFiltered.iff_cocone_nonempty.{v}]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : Catego …
  -/
  refine ⟨fun h J _ _ F => ?_, fun h J _ _ F => ?_⟩
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (F.op.comp (CategoryTh …
    -/
  · obtain ⟨c⟩ := h F
    /-
      case refine_1.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cocone F
      ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (F.op.comp (CategoryTh …
    -/
    exact ⟨c.pt, ⟨(limitCompYonedaIsoCocone F c.pt).inv c.ι⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      ⊢ Nonempty (CategoryTheory.Limits.Cocone F)
    -/
  · obtain ⟨pt, ⟨ι⟩⟩ := h F
    /-
      case refine_2.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      pt : C
      ι : CategoryTheory.Limits.limit (F.op.comp (CategoryTheory.yoneda.obj pt))
      ⊢ Nonempty (CategoryTheory.Limits.Cocone F)
    -/
    exact ⟨⟨pt, (limitCompYonedaIsoCocone F pt).hom ι⟩⟩
    /-
      🎉 no goals
    -/


/-- `C` is cofiltered if and only if for every functor `F : J ⥤ C` from a finite category there is
    some `X : C` such that `lim Hom(X, F·)` is nonempty. -/
theorem IsCofiltered.iff_nonempty_limit : IsCofiltered C ↔
    ∀ {J : Type v} [SmallCategory J] [FinCategory J] (F : J ⥤ C),
      ∃ (X : C), Nonempty (limit (F ⋙ coyoneda.obj (op X))) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (CategoryTheory.IsCofiltered C) (∀ {J : Type v} [inst : CategoryTheory.S …
  -/
  rw [IsCofiltered.iff_cone_nonempty.{v}]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : Catego …
  -/
  refine ⟨fun h J _ _ F => ?_, fun h J _ _ F => ?_⟩
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (F.comp (CategoryTheor …
    -/
  · obtain ⟨c⟩ := h F
    /-
      case refine_1.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      ⊢ Exists fun X => Nonempty (CategoryTheory.Limits.limit (F.comp (CategoryTheor …
    -/
    exact ⟨c.pt, ⟨(limitCompCoyonedaIsoCone F c.pt).inv c.π⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      ⊢ Nonempty (CategoryTheory.Limits.Cone F)
    -/
  · obtain ⟨pt, ⟨π⟩⟩ := h F
    /-
      case refine_2.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
      J : Type v
      x✝¹ : CategoryTheory.SmallCategory J
      x✝ : CategoryTheory.FinCategory J
      F : CategoryTheory.Functor J C
      pt : C
      π : CategoryTheory.Limits.limit (F.comp (CategoryTheory.coyoneda.obj { unop := …
      ⊢ Nonempty (CategoryTheory.Limits.Cone F)
    -/
    exact ⟨⟨pt, (limitCompCoyonedaIsoCone F pt).hom π⟩⟩
    /-
      🎉 no goals
    -/


/-- Class for having all cofiltered limits of a given size. -/
@[pp_with_univ]
class HasCofilteredLimitsOfSize : Prop where
  /-- For all filtered types of size `w`, we have limits -/
  HasLimitsOfShape : ∀ (I : Type w) [Category.{w'} I] [IsCofiltered I], HasLimitsOfShape I C


/-- Class for having all filtered colimits of a given size. -/
@[pp_with_univ]
class HasFilteredColimitsOfSize : Prop where
  /-- For all filtered types of a size `w`, we have colimits -/
  HasColimitsOfShape : ∀ (I : Type w) [Category.{w'} I] [IsFiltered I], HasColimitsOfShape I C


/-- Class for having cofiltered limits. -/
abbrev HasCofilteredLimits := HasCofilteredLimitsOfSize.{v, v} C


/-- Class for having filtered colimits. -/
abbrev HasFilteredColimits := HasFilteredColimitsOfSize.{v, v} C


instance (priority := 100) hasFilteredColimitsOfSize_of_hasColimitsOfSize
    [HasColimitsOfSize.{w', w} C] : HasFilteredColimitsOfSize.{w', w} C where
  HasColimitsOfShape _ _ _ := inferInstance


instance (priority := 100) hasCofilteredLimitsOfSize_of_hasLimitsOfSize
    [HasLimitsOfSize.{w', w} C] : HasCofilteredLimitsOfSize.{w', w} C where
  HasLimitsOfShape _ _ _ := inferInstance


instance (priority := 100) hasLimitsOfShape_of_has_cofiltered_limits
    [HasCofilteredLimitsOfSize.{w', w} C] (I : Type w) [Category.{w'} I] [IsCofiltered I] :
    HasLimitsOfShape I C :=
  HasCofilteredLimitsOfSize.HasLimitsOfShape _


instance (priority := 100) hasColimitsOfShape_of_has_filtered_colimits
    [HasFilteredColimitsOfSize.{w', w} C] (I : Type w) [Category.{w'} I] [IsFiltered I] :
    HasColimitsOfShape I C :=
  HasFilteredColimitsOfSize.HasColimitsOfShape _


lemma hasCofilteredLimitsOfSize_of_univLE [UnivLE.{w, w₂}] [UnivLE.{w', w₂'}]
    [HasCofilteredLimitsOfSize.{w₂', w₂} C] :
    HasCofilteredLimitsOfSize.{w', w} C where
  HasLimitsOfShape J :=
    haveI := IsCofiltered.of_equivalence ((ShrinkHoms.equivalence.{w₂'} J).trans <|
      Shrink.equivalence.{w₂} (ShrinkHoms.{w} J))
    hasLimitsOfShape_of_equivalence ((ShrinkHoms.equivalence.{w₂'} J).trans <|
      Shrink.equivalence.{w₂} (ShrinkHoms.{w} J)).symm


lemma hasCofilteredLimitsOfSize_shrink [HasCofilteredLimitsOfSize.{max w' w₂', max w w₂} C] :
    HasCofilteredLimitsOfSize.{w', w} C :=
  hasCofilteredLimitsOfSize_of_univLE.{w', w, max w' w₂', max w w₂}


lemma hasFilteredColimitsOfSize_of_univLE [UnivLE.{w, w₂}] [UnivLE.{w', w₂'}]
    [HasFilteredColimitsOfSize.{w₂', w₂} C] :
    HasFilteredColimitsOfSize.{w', w} C where
  HasColimitsOfShape J :=
    haveI := IsFiltered.of_equivalence ((ShrinkHoms.equivalence.{w₂'} J).trans <|
      Shrink.equivalence.{w₂} (ShrinkHoms.{w} J))
    hasColimitsOfShape_of_equivalence ((ShrinkHoms.equivalence.{w₂'} J).trans <|
      Shrink.equivalence.{w₂} (ShrinkHoms.{w} J)).symm


lemma hasFilteredColimitsOfSize_shrink [HasFilteredColimitsOfSize.{max w' w₂', max w w₂} C] :
    HasFilteredColimitsOfSize.{w', w} C :=
  hasFilteredColimitsOfSize_of_univLE.{w', w, max w' w₂', max w w₂}


