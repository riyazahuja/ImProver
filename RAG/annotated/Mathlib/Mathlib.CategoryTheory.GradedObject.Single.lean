/-- The functor which sends `X : C` to the graded object which is `X` in degree `j`
and the initial object in other degrees. -/
noncomputable def single (j : J) : C ⥤ GradedObject J C where
  obj X i := if i = j then X else ⊥_ C
  map {X₁ X₂} f i :=
    if h : i = j then eqToHom (if_pos h) ≫ f ≫ eqToHom (if_pos h).symm
                     /-
                       J : Type u_1
                       C : Type u_2
                       inst✝² : CategoryTheory.Category.{?u.44, u_2} C
                       inst✝¹ : CategoryTheory.Limits.HasInitial C
                       inst✝ : DecidableEq J
                       j : J
                       X₁ X₂ : C
                       f : Quiver.Hom X₁ X₂
                       i : J
                       h : Not (Eq i j)
                       ⊢ Eq ((fun X i => ite (Eq i j) X (CategoryTheory.Limits.initial C)) X₁ i) ((fu …
                     -/
    else eqToHom (by dsimp; rw [if_neg h, if_neg h])
                            /-
                              🎉 no goals
                            -/


variable (J) in
/-- The functor which sends `X : C` to the graded object which is `X` in degree `0`
and the initial object in nonzero degrees. -/
noncomputable abbrev single₀ [Zero J] : C ⥤ GradedObject J C := single 0


/-- The canonical isomorphism `(single j).obj X i ≅ X` when `i = j`. -/
noncomputable def singleObjApplyIsoOfEq (j : J) (X : C) (i : J) (h : i = j) :
    (single j).obj X i ≅ X := eqToIso (if_pos h)


/-- The canonical isomorphism `(single j).obj X j ≅ X`. -/
noncomputable abbrev singleObjApplyIso (j : J) (X : C) :
    (single j).obj X j ≅ X := singleObjApplyIsoOfEq j X j rfl


/-- The object `(single j).obj X i` is initial when `i ≠ j`. -/
noncomputable def isInitialSingleObjApply (j : J) (X : C) (i : J) (h : i ≠ j) :
    IsInitial ((single j).obj X i) := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{?u.28977, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X : C
    i : J
    h : Ne i j
    ⊢ CategoryTheory.Limits.IsInitial ((CategoryTheory.GradedObject.single j).obj  …
  -/
  dsimp [single]
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{?u.28977, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X : C
    i : J
    h : Ne i j
    ⊢ CategoryTheory.Limits.IsInitial (ite (Eq i j) X (CategoryTheory.Limits.initi …
  -/
  rw [if_neg h]
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{?u.28977, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X : C
    i : J
    h : Ne i j
    ⊢ CategoryTheory.Limits.IsInitial (CategoryTheory.Limits.initial C)
  -/
  exact initialIsInitial
  /-
    🎉 no goals
  -/


lemma singleObjApplyIsoOfEq_inv_single_map (j : J) {X Y : C} (f : X ⟶ Y) (i : J) (h : i = j) :
    (singleObjApplyIsoOfEq j X i h).inv ≫ (single j).map f i =
      f ≫ (singleObjApplyIsoOfEq j Y i h).inv := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X Y : C
    f : Quiver.Hom X Y
    i : J
    h : Eq i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.singleOb …
  -/
  subst h
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    X Y : C
    f : Quiver.Hom X Y
    i : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.singleOb …
  -/
  simp [singleObjApplyIsoOfEq, single]
  /-
    🎉 no goals
  -/


lemma single_map_singleObjApplyIsoOfEq_hom (j : J) {X Y : C} (f : X ⟶ Y) (i : J) (h : i = j) :
    (single j).map f i ≫ (singleObjApplyIsoOfEq j Y i h).hom =
      (singleObjApplyIsoOfEq j X i h).hom ≫ f := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X Y : C
    f : Quiver.Hom X Y
    i : J
    h : Eq i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GradedObject.single  …
  -/
  subst h
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    X Y : C
    f : Quiver.Hom X Y
    i : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GradedObject.single  …
  -/
  simp [singleObjApplyIsoOfEq, single]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma singleObjApplyIso_inv_single_map (j : J) {X Y : C} (f : X ⟶ Y) :
    (singleObjApplyIso j X).inv ≫ (single j).map f j = f ≫ (singleObjApplyIso j Y).inv := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.singleOb …
  -/
  apply singleObjApplyIsoOfEq_inv_single_map
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma single_map_singleObjApplyIso_hom (j : J) {X Y : C} (f : X ⟶ Y) :
    (single j).map f j ≫ (singleObjApplyIso j Y).hom = (singleObjApplyIso j X).hom ≫ f := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : DecidableEq J
    j : J
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GradedObject.single  …
  -/
  apply single_map_singleObjApplyIsoOfEq_hom
  /-
    🎉 no goals
  -/


variable (C) in
/-- The composition of the single functor `single j : C ⥤ GradedObject J C` and the
evaluation functor `eval j` identifies to the identity functor. -/
@[simps!]
noncomputable def singleCompEval (j : J) : single j ⋙ eval j ≅ 𝟭 C :=
                                                /-
                                                  J : Type u_1
                                                  C : Type u_2
                                                  inst✝² : CategoryTheory.Category.{?u.41170, u_2} C
                                                  inst✝¹ : CategoryTheory.Limits.HasInitial C
                                                  inst✝ : DecidableEq J
                                                  j : J
                                                  ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                -/
  NatIso.ofComponents (singleObjApplyIso j) (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


