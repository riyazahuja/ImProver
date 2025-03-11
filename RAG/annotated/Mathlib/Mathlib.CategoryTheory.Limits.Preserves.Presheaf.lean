/-- If `C` is a finitely cocomplete category and `A : Cᵒᵖ ⥤ Type u` is a presheaf that preserves
finite limites, then `CostructuredArrow yoneda A` is filtered.

One direction of Proposition 3.3.13 of [Kashiwara2006].
-/
theorem isFiltered_costructuredArrow_yoneda_of_preservesFiniteLimits
    [PreservesFiniteLimits A] : IsFiltered (CostructuredArrow yoneda A) := by
  suffices IsCofiltered A.Elements from
    IsFiltered.of_equivalence (CategoryOfElements.costructuredArrowYonedaEquivalence _)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits A
    ⊢ CategoryTheory.IsCofiltered A.Elements
  -/
  suffices HasFiniteLimits A.Elements from IsCofiltered.of_hasFiniteLimits A.Elements
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type v)
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits A
    ⊢ CategoryTheory.Limits.HasFiniteLimits A.Elements
  -/
  exact ⟨fun J _ _ => inferInstance⟩
  /-
    🎉 no goals
  -/


/-- (Implementation) This is the bifunctor we will apply "filtered colimits commute with finite
limits" to. -/
def functorToInterchange : J ⥤ CostructuredArrow yoneda A ⥤ Type u :=
  K ⋙ coyoneda ⋙ (whiskeringLeft _ _ _).obj (CostructuredArrow.proj _ _)


/-- (Implementation) The definition of `functorToInterchange`. -/
@[simps!]
def functorToInterchangeIso : functorToInterchange A K ≅
    K ⋙ coyoneda ⋙ (whiskeringLeft _ _ _).obj (CostructuredArrow.proj _ _) :=
  Iso.refl _


/-- (Implementation) One way to express the flipped version of our functor. We choose this
association because the type of `Presheaf.tautologicalCocone` is
`Cocone (CostructuredArrow.proj yoneda P ⋙ yoneda)`, so this association will show up in the
proof.-/
@[simps!]
def flipFunctorToInterchange : (functorToInterchange A K).flip ≅
    ((CostructuredArrow.proj yoneda A ⋙ yoneda) ⋙ (whiskeringLeft J Cᵒᵖ (Type u)).obj K) :=
  Iso.refl _


/-- (Implementation) A natural isomorphism we will need to construct `iso`. -/
@[simps! (config := { fullyApplied := false }) hom_app]
noncomputable def isoAux :
    (CostructuredArrow.proj yoneda A ⋙ yoneda ⋙ (evaluation Cᵒᵖ (Type u)).obj (limit K)) ≅
      ((coyoneda ⋙ (whiskeringLeft (CostructuredArrow yoneda A) C (Type u)).obj
        (CostructuredArrow.proj yoneda A)).obj (limit K)) :=
  Iso.refl _


/-- (Implementation) The isomorphism that proves that `A` preserves finite limits. -/
noncomputable def iso [IsFiltered (CostructuredArrow yoneda A)] :
    A.obj (limit K) ≅ limit (K ⋙ A) := calc
  A.obj (limit K) ≅ (colimit (CostructuredArrow.proj yoneda A ⋙ yoneda)).obj (limit K) :=
      (IsColimit.coconePointUniqueUpToIso
        (Presheaf.isColimitTautologicalCocone A) (colimit.isColimit _)).app _
  _ ≅ colimit (((CostructuredArrow.proj yoneda A) ⋙ yoneda) ⋙ (evaluation _ _).obj (limit K)) :=
      (colimitObjIsoColimitCompEvaluation _ _)
  _ ≅ colimit ((CostructuredArrow.proj yoneda A) ⋙ yoneda ⋙ (evaluation _ _).obj (limit K)) :=
      HasColimit.isoOfNatIso (Functor.associator _ _ _)
  _ ≅ colimit
      ((coyoneda ⋙ (whiskeringLeft _ _ _).obj (CostructuredArrow.proj yoneda A)).obj (limit K)) :=
        HasColimit.isoOfNatIso (isoAux A K)
  _ ≅ colimit (limit (K ⋙ (coyoneda ⋙ (whiskeringLeft _ _ _).obj (CostructuredArrow.proj _ _)))) :=
        HasColimit.isoOfNatIso (preservesLimitIso _ _)
  _ ≅ colimit (limit (functorToInterchange A K)) :=
        HasColimit.isoOfNatIso (HasLimit.isoOfNatIso (functorToInterchangeIso A K).symm)
  _ ≅ limit (colimit (functorToInterchange A K).flip) := colimitLimitIso _
  _ ≅ limit (colimit
        ((CostructuredArrow.proj yoneda A ⋙ yoneda) ⋙ (whiskeringLeft _ _ _).obj K)) :=
          HasLimit.isoOfNatIso (HasColimit.isoOfNatIso (flipFunctorToInterchange A K))
  _ ≅ limit (K ⋙ colimit (CostructuredArrow.proj yoneda A ⋙ yoneda)) :=
        HasLimit.isoOfNatIso
          (colimitCompWhiskeringLeftIsoCompColimit (CostructuredArrow.proj yoneda A ⋙ yoneda) K)
  _ ≅ limit (K ⋙ A) := HasLimit.isoOfNatIso (isoWhiskerLeft _
        (IsColimit.coconePointUniqueUpToIso
          (colimit.isColimit _) (Presheaf.isColimitTautologicalCocone A)))


theorem iso_hom [IsFiltered (CostructuredArrow yoneda A)] : (iso A K).hom = limit.post K A := by
  -- We will have to use `ι_colimitLimitIso_limit_π` eventually, so let's start by
  -- transforming the goal into something from a colimit to a limit so that we can apply
  -- `limit.hom_ext` and `colimit.hom_ext`.
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    ⊢ Eq (CategoryTheory.Limits.PreservesFiniteLimitsOfIsFilteredCostructuredArrow …
  -/
  dsimp [iso, -Iso.app_hom]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Presheaf.isColimitT …
  -/
  rw [Eq.comm, ← Iso.inv_comp_eq, ← Iso.inv_comp_eq]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitObjIsoC …
  -/
  refine limit.hom_ext (fun j => colimit.hom_ext (fun i => ?_))
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    j : J
    i : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (((C …
  -/
  simp only [Category.assoc]

  -- `simp` is not too helpful here because we will need to apply `NatTrans.comp_app_assoc`
  -- backwards at certain points, so we rewrite the term manually.
  rw [HasLimit.isoOfNatIso_hom_π, HasLimit.isoOfNatIso_hom_π_assoc, limit.post_π,
    colimitObjIsoColimitCompEvaluation_ι_inv_assoc (CostructuredArrow.proj yoneda A ⋙ yoneda),
    Iso.app_inv, ← NatTrans.comp_app_assoc, colimit.comp_coconePointUniqueUpToIso_inv,
    Presheaf.tautologicalCocone_ι_app, HasColimit.isoOfNatIso_ι_hom_assoc,
    HasLimit.isoOfNatIso_hom_π_assoc, HasColimit.isoOfNatIso_ι_hom_assoc,
    HasColimit.isoOfNatIso_ι_hom_assoc, HasColimit.isoOfNatIso_ι_hom_assoc,
    ι_colimitLimitIso_limit_π_assoc, isoAux_hom_app, ← NatTrans.comp_app_assoc,
    ← NatTrans.comp_app_assoc, Category.assoc, HasLimit.isoOfNatIso_hom_π,
    preservesLimitIso_hom_π_assoc, Iso.symm_hom,
    ← NatTrans.comp_app_assoc, HasColimit.isoOfNatIso_ι_hom,
    ← NatTrans.comp_app_assoc, Category.assoc,
    ι_colimitCompWhiskeringLeftIsoCompColimit_hom,
    NatTrans.comp_app, Category.assoc, isoWhiskerLeft_hom, NatTrans.comp_app, Category.assoc,
    ← NatTrans.comp_app, ← whiskerLeft_comp, colimit.comp_coconePointUniqueUpToIso_hom]

  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    j : J
    i : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.hom.app (CategoryTheory.Limits.lim …
  -/
  have := i.hom.naturality (limit.π K j)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    j : J
    i : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj i.le …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.hom.app (CategoryTheory.Limits.lim …
  -/
  dsimp only [yoneda_obj_obj, Functor.const_obj_obj] at this
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    j : J
    i : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj i.le …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.hom.app (CategoryTheory.Limits.lim …
  -/
  rw [← this]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    j : J
    i : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj i.le …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj i.left).m …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝⁴ : CategoryTheory.SmallCategory C
    inst✝³ : CategoryTheory.Limits.HasFiniteColimits C
    A : CategoryTheory.Functor (Opposite C) (Type u)
    J : Type
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J (Opposite C)
    inst✝ : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow CategoryTh …
    j : J
    i : CategoryTheory.CostructuredArrow CategoryTheory.yoneda A
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj i.le …
    a✝ : Quiver.Hom (Opposite.unop (CategoryTheory.Limits.limit K)) i.left
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yoneda.obj i.left).m …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isIso_post [IsFiltered (CostructuredArrow yoneda A)] : IsIso (limit.post K A) :=
  iso_hom A K ▸ inferInstance


/-- If `C` is a small finitely cocomplete category and `A : Cᵒᵖ ⥤ Type u` is a presheaf such that
`CostructuredArrow yoneda A` is filtered, then `A` preserves finite limits.

One direction of Proposition 3.3.13 of [Kashiwara2006].
-/
lemma preservesFiniteLimits_of_isFiltered_costructuredArrow_yoneda
    [IsFiltered (CostructuredArrow yoneda A)] : PreservesFiniteLimits A where
  preservesFiniteLimits _ _ _ := ⟨fun {_} => preservesLimit_of_isIso_post _ _⟩


/-- If `C` is a small finitely cocomplete category and `A : Cᵒᵖ ⥤ Type u` is a presheaf, then
`CostructuredArrow yoneda A` is filtered if and only if `A` preserves finite limits.

Proposition 3.3.13 of [Kashiwara2006].
-/
theorem isFiltered_costructuredArrow_yoneda_iff_nonempty_preservesFiniteLimits :
    IsFiltered (CostructuredArrow yoneda A) ↔ Nonempty (PreservesFiniteLimits A) :=
  ⟨fun _ => ⟨preservesFiniteLimits_of_isFiltered_costructuredArrow_yoneda A⟩,
   fun ⟨_⟩ => isFiltered_costructuredArrow_yoneda_of_preservesFiniteLimits A⟩


