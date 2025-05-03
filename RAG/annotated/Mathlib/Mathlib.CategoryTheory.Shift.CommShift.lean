/-- For any functor `F : C ⥤ D`, this is the obvious isomorphism
`shiftFunctor C (0 : A) ⋙ F ≅ F ⋙ shiftFunctor D (0 : A)` deduced from the
isomorphisms `shiftFunctorZero` on both categories `C` and `D`. -/
@[simps!]
noncomputable def isoZero : shiftFunctor C (0 : A) ⋙ F ≅ F ⋙ shiftFunctor D (0 : A) :=
  isoWhiskerRight (shiftFunctorZero C A) F ≪≫ F.leftUnitor ≪≫
     F.rightUnitor.symm ≪≫ isoWhiskerLeft F (shiftFunctorZero D A).symm


/-- For any functor `F : C ⥤ D` and any `a` in `A` such that `a = 0`,
this is the obvious isomorphism `shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` deduced from the
isomorphisms `shiftFunctorZero'` on both categories `C` and `D`. -/
@[simps!]
noncomputable def isoZero' (a : A) (ha : a = 0) : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a :=
  isoWhiskerRight (shiftFunctorZero' C a ha) F ≪≫ F.leftUnitor ≪≫
     F.rightUnitor.symm ≪≫ isoWhiskerLeft F (shiftFunctorZero' D a ha).symm


@[simp]
lemma isoZero'_eq_isoZero : isoZero' F A 0 rfl = isoZero F A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    ⊢ Eq (CategoryTheory.Functor.CommShift.isoZero' F A 0 ⋯) (CategoryTheory.Funct …
  -/
  ext; simp [isoZero', shiftFunctorZero']
       /-
         🎉 no goals
       -/


/-- If a functor `F : C ⥤ D` is equipped with "commutation isomorphisms" with the
shifts by `a` and `b`, then there is a commutation isomorphism with the shift by `c` when
`a + b = c`. -/
@[simps!]
noncomputable def isoAdd' {a b c : A} (h : a + b = c)
    (e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a)
    (e₂ : shiftFunctor C b ⋙ F ≅ F ⋙ shiftFunctor D b) :
    shiftFunctor C c ⋙ F ≅ F ⋙ shiftFunctor D c :=
  isoWhiskerRight (shiftFunctorAdd' C _ _ _ h) F ≪≫ Functor.associator _ _ _ ≪≫
    isoWhiskerLeft _ e₂ ≪≫ (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight e₁ _ ≪≫
      Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (shiftFunctorAdd' D _ _ _ h).symm


/-- If a functor `F : C ⥤ D` is equipped with "commutation isomorphisms" with the
shifts by `a` and `b`, then there is a commutation isomorphism with the shift by `a + b`. -/
noncomputable def isoAdd {a b : A}
    (e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a)
    (e₂ : shiftFunctor C b ⋙ F ≅ F ⋙ shiftFunctor D b) :
    shiftFunctor C (a + b) ⋙ F ≅ F ⋙ shiftFunctor D (a + b) :=
  CommShift.isoAdd' rfl e₁ e₂


@[simp]
lemma isoAdd_hom_app {a b : A}
    (e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a)
    (e₂ : shiftFunctor C b ⋙ F ≅ F ⋙ shiftFunctor D b) (X : C) :
      (CommShift.isoAdd e₁ e₂).hom.app X =
        F.map ((shiftFunctorAdd C a b).hom.app X) ≫ e₂.hom.app ((shiftFunctor C a).obj X) ≫
          (shiftFunctor D b).map (e₁.hom.app X) ≫ (shiftFunctorAdd D a b).inv.app (F.obj X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    X : C
    ⊢ Eq ((CategoryTheory.Functor.CommShift.isoAdd e₁ e₂).hom.app X) (CategoryTheo …
  -/
  simp only [isoAdd, isoAdd'_hom_app, shiftFunctorAdd'_eq_shiftFunctorAdd]
  /-
    🎉 no goals
  -/


@[simp]
lemma isoAdd_inv_app {a b : A}
    (e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a)
    (e₂ : shiftFunctor C b ⋙ F ≅ F ⋙ shiftFunctor D b) (X : C) :
      (CommShift.isoAdd e₁ e₂).inv.app X = (shiftFunctorAdd D a b).hom.app (F.obj X) ≫
        (shiftFunctor D b).map (e₁.inv.app X) ≫ e₂.inv.app ((shiftFunctor C a).obj X) ≫
        F.map ((shiftFunctorAdd C a b).inv.app X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    X : C
    ⊢ Eq ((CategoryTheory.Functor.CommShift.isoAdd e₁ e₂).inv.app X) (CategoryTheo …
  -/
  simp only [isoAdd, isoAdd'_inv_app, shiftFunctorAdd'_eq_shiftFunctorAdd]
  /-
    🎉 no goals
  -/


/-- A functor `F` commutes with the shift by a monoid `A` if it is equipped with
commutation isomorphisms with the shifts by all `a : A`, and these isomorphisms
satisfy coherence properties with respect to `0 : A` and the addition in `A`. -/
class CommShift where
  iso (a : A) : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a
  zero : iso 0 = CommShift.isoZero F A := by aesop_cat
  add (a b : A) : iso (a + b) = CommShift.isoAdd (iso a) (iso b) := by aesop_cat


/-- If a functor `F` commutes with the shift by `A` (i.e. `[F.CommShift A]`), then
`F.commShiftIso a` is the given isomorphism `shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a`. -/
def commShiftIso (a : A) :
    shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a :=
  CommShift.iso a

-- Note: The following two lemmas are introduced in order to have more proofs work `by simp`.
-- Indeed, `simp only [(F.commShiftIso a).hom.naturality f]` would almost never work because
-- of the compositions of functors which appear in both the source and target of
-- `F.commShiftIso a`. Otherwise, we would be forced to use `erw [NatTrans.naturality]`.


@[reassoc (attr := simp)]
lemma commShiftIso_hom_naturality {X Y : C} (f : X ⟶ Y) (a : A) :
    F.map (f⟦a⟧') ≫ (F.commShiftIso a).hom.app Y =
      (F.commShiftIso a).hom.app X ≫ (F.map f)⟦a⟧' :=
  (F.commShiftIso a).hom.naturality f


@[reassoc (attr := simp)]
lemma commShiftIso_inv_naturality {X Y : C} (f : X ⟶ Y) (a : A) :
    (F.map f)⟦a⟧' ≫ (F.commShiftIso a).inv.app Y =
      (F.commShiftIso a).inv.app X ≫ F.map (f⟦a⟧') :=
  (F.commShiftIso a).inv.naturality f


lemma commShiftIso_zero :
    F.commShiftIso (0 : A) = CommShift.isoZero F A :=
  CommShift.zero


set_option linter.docPrime false in
lemma commShiftIso_zero' (a : A) (h : a = 0) :
    F.commShiftIso a = CommShift.isoZero' F A a h := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a : A
    h : Eq a 0
    ⊢ Eq (F.commShiftIso a) (CategoryTheory.Functor.CommShift.isoZero' F A a h)
  -/
  subst h; rw [CommShift.isoZero'_eq_isoZero, commShiftIso_zero]
           /-
             🎉 no goals
           -/


lemma commShiftIso_add (a b : A) :
    F.commShiftIso (a + b) = CommShift.isoAdd (F.commShiftIso a) (F.commShiftIso b) :=
  CommShift.add a b


lemma commShiftIso_add' {a b c : A} (h : a + b = c) :
    F.commShiftIso c = CommShift.isoAdd' h (F.commShiftIso a) (F.commShiftIso b) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a b c : A
    h : Eq (HAdd.hAdd a b) c
    ⊢ Eq (F.commShiftIso c) (CategoryTheory.Functor.CommShift.isoAdd' h (F.commShi …
  -/
  subst h
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a b : A
    ⊢ Eq (F.commShiftIso (HAdd.hAdd a b)) (CategoryTheory.Functor.CommShift.isoAdd …
  -/
  simp only [commShiftIso_add, CommShift.isoAdd]
  /-
    🎉 no goals
  -/


variable (C) in
instance id : CommShift (𝟭 C) A where
  iso := fun _ => rightUnitor _ ≪≫ (leftUnitor _).symm


instance comp [F.CommShift A] [G.CommShift A] : (F ⋙ G).CommShift A where
  iso a := (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight (F.commShiftIso a) _ ≪≫
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (G.commShiftIso a) ≪≫
    (Functor.associator _ _ _).symm
  zero := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      ⊢ Eq ((fun a => ((CategoryTheory.shiftFunctor C a).associator F G).symm.trans  …
    -/
    ext X
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      X : C
      ⊢ Eq (((fun a => ((CategoryTheory.shiftFunctor C a).associator F G).symm.trans …
    -/
    dsimp
    simp only [id_comp, comp_id, commShiftIso_zero, isoZero_hom_app, ← Functor.map_comp_assoc,
      assoc, Iso.inv_hom_id_app, id_obj, comp_map, comp_obj]
  add := fun a b => by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      a b : A
      ⊢ Eq ((fun a => ((CategoryTheory.shiftFunctor C a).associator F G).symm.trans  …
    -/
    ext X
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      a b : A
      X : C
      ⊢ Eq (((fun a => ((CategoryTheory.shiftFunctor C a).associator F G).symm.trans …
    -/
    dsimp
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
    -/
    simp only [commShiftIso_add, isoAdd_hom_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
    -/
    dsimp
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
    -/
    simp only [comp_id, id_comp, assoc, ← Functor.map_comp_assoc, Iso.inv_hom_id_app, comp_obj]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝¹¹ : CategoryTheory.Category.{?u.58894, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{?u.58898, u_2} D
      inst✝⁹ : CategoryTheory.Category.{?u.58902, u_3} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      A : Type u_4
      B : Type u_5
      inst✝⁸ : AddMonoid A
      inst✝⁷ : AddCommMonoid B
      inst✝⁶ : CategoryTheory.HasShift C A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift C B
      inst✝² : CategoryTheory.HasShift D B
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
    -/
    simp only [map_comp, assoc, commShiftIso_hom_naturality_assoc]
    /-
      🎉 no goals
    -/


@[simp]
lemma commShiftIso_id_hom_app (a : A) (X : C) :
    (commShiftIso (𝟭 C) a).hom.app X = 𝟙 _ := comp_id _


@[simp]
lemma commShiftIso_id_inv_app (a : A) (X : C) :
    (commShiftIso (𝟭 C) a).inv.app X = 𝟙 _ := comp_id _


lemma commShiftIso_comp_hom_app [F.CommShift A] [G.CommShift A] (a : A) (X : C) :
    (commShiftIso (F ⋙ G) a).hom.app X =
      G.map ((commShiftIso F a).hom.app X) ≫ (commShiftIso G a).hom.app (F.obj X) := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁶ : CategoryTheory.Category.{u_8, u_3} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    A : Type u_4
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : F.CommShift A
    inst✝ : G.CommShift A
    a : A
    X : C
    ⊢ Eq (((F.comp G).commShiftIso a).hom.app X) (CategoryTheory.CategoryStruct.co …
  -/
  simp [commShiftIso, CommShift.iso]
  /-
    🎉 no goals
  -/


lemma commShiftIso_comp_inv_app [F.CommShift A] [G.CommShift A] (a : A) (X : C) :
    (commShiftIso (F ⋙ G) a).inv.app X =
      (commShiftIso G a).inv.app (F.obj X) ≫ G.map ((commShiftIso F a).inv.app X) := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁶ : CategoryTheory.Category.{u_8, u_3} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    A : Type u_4
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : CategoryTheory.HasShift E A
    inst✝¹ : F.CommShift A
    inst✝ : G.CommShift A
    a : A
    X : C
    ⊢ Eq (((F.comp G).commShiftIso a).inv.app X) (CategoryTheory.CategoryStruct.co …
  -/
  simp [commShiftIso, CommShift.iso]
  /-
    🎉 no goals
  -/


lemma map_shiftFunctorComm_hom_app [F.CommShift B] (X : C) (a b : B) :
    F.map ((shiftFunctorComm C a b).hom.app X) = (F.commShiftIso b).hom.app (X⟦a⟧) ≫
      ((F.commShiftIso a).hom.app X)⟦b⟧' ≫ (shiftFunctorComm D a b).hom.app (F.obj X) ≫
      ((F.commShiftIso b).inv.app X)⟦a⟧' ≫ (F.commShiftIso a).inv.app (X⟦b⟧) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    B : Type u_5
    inst✝³ : AddCommMonoid B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.HasShift D B
    inst✝ : F.CommShift B
    X : C
    a b : B
    ⊢ Eq (F.map ((CategoryTheory.shiftFunctorComm C a b).hom.app X)) (CategoryTheo …
  -/
  have eq := NatTrans.congr_app (congr_arg Iso.hom (F.commShiftIso_add a b)) X
  simp only [comp_obj, CommShift.isoAdd_hom_app,
    ← cancel_epi (F.map ((shiftFunctorAdd C a b).inv.app X)), Category.assoc,
    ← F.map_comp_assoc, Iso.inv_hom_id_app, F.map_id, Category.id_comp, F.map_comp] at eq
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    B : Type u_5
    inst✝³ : AddCommMonoid B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.HasShift D B
    inst✝ : F.CommShift B
    X : C
    a b : B
    eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
    ⊢ Eq (F.map ((CategoryTheory.shiftFunctorComm C a b).hom.app X)) (CategoryTheo …
  -/
  simp only [shiftFunctorComm_eq D a b _ rfl]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    B : Type u_5
    inst✝³ : AddCommMonoid B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.HasShift D B
    inst✝ : F.CommShift B
    X : C
    a b : B
    eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
    ⊢ Eq (F.map ((CategoryTheory.shiftFunctorComm C a b).hom.app X)) (CategoryTheo …
  -/
  dsimp
  simp only [Functor.map_comp, shiftFunctorAdd'_eq_shiftFunctorAdd, Category.assoc,
    ← reassoc_of% eq, shiftFunctorComm_eq C a b _ rfl]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    B : Type u_5
    inst✝³ : AddCommMonoid B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.HasShift D B
    inst✝ : F.CommShift B
    X : C
    a b : B
    eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
    ⊢ Eq (F.map (((CategoryTheory.shiftFunctorAdd C a b).symm.trans (CategoryTheor …
  -/
  dsimp
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    B : Type u_5
    inst✝³ : AddCommMonoid B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.HasShift D B
    inst✝ : F.CommShift B
    X : C
    a b : B
    eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorA …
  -/
  rw [Functor.map_comp]
  simp only [NatTrans.congr_app (congr_arg Iso.hom (F.commShiftIso_add' (add_comm b a))) X,
    CommShift.isoAdd'_hom_app, Category.assoc, Iso.inv_hom_id_app_assoc,
    ← Functor.map_comp_assoc, Iso.hom_inv_id_app,
    Functor.map_id, Category.id_comp, comp_obj, Category.comp_id]


@[simp, reassoc]
lemma map_shiftFunctorCompIsoId_hom_app [F.CommShift A] (X : C) (a b : A) (h : a + b = 0) :
    F.map ((shiftFunctorCompIsoId C a b h).hom.app X) =
      (F.commShiftIso b).hom.app (X⟦a⟧) ≫ ((F.commShiftIso a).hom.app X)⟦b⟧' ≫
        (shiftFunctorCompIsoId D a b h).hom.app (F.obj X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    X : C
    a b : A
    h : Eq (HAdd.hAdd a b) 0
    ⊢ Eq (F.map ((CategoryTheory.shiftFunctorCompIsoId C a b h).hom.app X)) (Categ …
  -/
  dsimp [shiftFunctorCompIsoId]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    X : C
    a b : A
    h : Eq (HAdd.hAdd a b) 0
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorA …
  -/
  have eq := NatTrans.congr_app (congr_arg Iso.hom (F.commShiftIso_add' h)) X
  simp only [commShiftIso_zero, comp_obj, CommShift.isoZero_hom_app,
    CommShift.isoAdd'_hom_app] at eq
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    X : C
    a b : A
    h : Eq (HAdd.hAdd a b) 0
    eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorA …
  -/
  rw [← cancel_epi (F.map ((shiftFunctorAdd' C a b 0 h).hom.app X)), ← reassoc_of% eq, F.map_comp]
  simp only [Iso.inv_hom_id_app, id_obj, Category.comp_id, ← F.map_comp_assoc, Iso.hom_inv_id_app,
    F.map_id, Category.id_comp]


@[simp, reassoc]
lemma map_shiftFunctorCompIsoId_inv_app [F.CommShift A] (X : C) (a b : A) (h : a + b = 0) :
    F.map ((shiftFunctorCompIsoId C a b h).inv.app X) =
      (shiftFunctorCompIsoId D a b h).inv.app (F.obj X) ≫
        ((F.commShiftIso a).inv.app X)⟦b⟧' ≫ (F.commShiftIso b).inv.app (X⟦a⟧) := by
  rw [← cancel_epi (F.map ((shiftFunctorCompIsoId C a b h).hom.app X)), ← F.map_comp,
    Iso.hom_inv_id_app, F.map_id, map_shiftFunctorCompIsoId_hom_app]
  simp only [comp_obj, id_obj, Category.assoc, Iso.hom_inv_id_app_assoc,
    ← Functor.map_comp_assoc, Iso.hom_inv_id_app, Functor.map_id, Category.id_comp]


/-- If `τ : F₁ ⟶ F₂` is a natural transformation between two functors
which commute with a shift by an additive monoid `A`, this typeclass
asserts a compatibility of `τ` with these shifts. -/
class CommShift : Prop where
  shift_comm (a : A) : (F₁.commShiftIso a).hom ≫ whiskerRight τ _ =
    whiskerLeft _ τ ≫ (F₂.commShiftIso a).hom := by aesop_cat


@[reassoc]
lemma shift_comm (a : A) :
    (F₁.commShiftIso a).hom ≫ whiskerRight τ _ =
      whiskerLeft _ τ ≫ (F₂.commShiftIso a).hom := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_6, u_2} D
    F₁ F₂ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    A : Type u_5
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : F₁.CommShift A
    inst✝¹ : F₂.CommShift A
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.commShiftIso a).hom (CategoryTheo …
  -/
  apply CommShift.shift_comm
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shift_app_comm (a : A) (X : C) :
    (F₁.commShiftIso a).hom.app X ≫ (τ.app X)⟦a⟧' =
      τ.app (X⟦a⟧) ≫ (F₂.commShiftIso a).hom.app X :=
  congr_app (shift_comm τ a) X


@[reassoc]
lemma shift_app (a : A) (X : C) :
    (τ.app X)⟦a⟧' = (F₁.commShiftIso a).inv.app X ≫
      τ.app (X⟦a⟧) ≫ (F₂.commShiftIso a).hom.app X := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_6, u_2} D
    F₁ F₂ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    A : Type u_5
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : F₁.CommShift A
    inst✝¹ : F₂.CommShift A
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor D a).map (τ.app X)) (CategoryTheory.Categor …
  -/
  rw [← shift_app_comm, Iso.inv_hom_id_app_assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma app_shift (a : A) (X : C) :
    τ.app (X⟦a⟧) = (F₁.commShiftIso a).hom.app X ≫ (τ.app X)⟦a⟧' ≫
      (F₂.commShiftIso a).inv.app X := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_6, u_2} D
    F₁ F₂ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    A : Type u_5
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : F₁.CommShift A
    inst✝¹ : F₂.CommShift A
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    a : A
    X : C
    ⊢ Eq (τ.app ((CategoryTheory.shiftFunctor C a).obj X)) (CategoryTheory.Categor …
  -/
  simp [shift_app_comm_assoc τ a X]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-31")] alias CommShift.comm' := shift_comm

@[deprecated (since := "2024-12-31")] alias CommShift.comm := shift_comm

@[deprecated (since := "2024-12-31")] alias CommShift.comm_app := shift_app_comm

@[deprecated (since := "2024-12-31")] alias CommShift.shift_app := shift_app

@[deprecated (since := "2024-12-31")] alias CommShift.app_shift := app_shift


instance of_iso_inv [NatTrans.CommShift e.hom A] :
  NatTrans.CommShift e.inv A := ⟨fun a => by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    J : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_7, u_2} D
    inst✝¹³ : CategoryTheory.Category.{?u.125912, u_3} E
    inst✝¹² : CategoryTheory.Category.{?u.125916, u_4} J
    F₁ F₂ F₃ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    τ' : Quiver.Hom F₂ F₃
    e : CategoryTheory.Iso F₁ F₂
    G G' : CategoryTheory.Functor D E
    τ'' : Quiver.Hom G G'
    H : CategoryTheory.Functor E J
    A : Type u_5
    inst✝¹¹ : AddMonoid A
    inst✝¹⁰ : CategoryTheory.HasShift C A
    inst✝⁹ : CategoryTheory.HasShift D A
    inst✝⁸ : CategoryTheory.HasShift E A
    inst✝⁷ : CategoryTheory.HasShift J A
    inst✝⁶ : F₁.CommShift A
    inst✝⁵ : F₂.CommShift A
    inst✝⁴ : F₃.CommShift A
    inst✝³ : G.CommShift A
    inst✝² : G'.CommShift A
    inst✝¹ : H.CommShift A
    inst✝ : CategoryTheory.NatTrans.CommShift e.hom A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₂.commShiftIso a).hom (CategoryTheo …
  -/
  ext X
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    E : Type u_3
    J : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_7, u_2} D
    inst✝¹³ : CategoryTheory.Category.{?u.125912, u_3} E
    inst✝¹² : CategoryTheory.Category.{?u.125916, u_4} J
    F₁ F₂ F₃ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    τ' : Quiver.Hom F₂ F₃
    e : CategoryTheory.Iso F₁ F₂
    G G' : CategoryTheory.Functor D E
    τ'' : Quiver.Hom G G'
    H : CategoryTheory.Functor E J
    A : Type u_5
    inst✝¹¹ : AddMonoid A
    inst✝¹⁰ : CategoryTheory.HasShift C A
    inst✝⁹ : CategoryTheory.HasShift D A
    inst✝⁸ : CategoryTheory.HasShift E A
    inst✝⁷ : CategoryTheory.HasShift J A
    inst✝⁶ : F₁.CommShift A
    inst✝⁵ : F₂.CommShift A
    inst✝⁴ : F₃.CommShift A
    inst✝³ : G.CommShift A
    inst✝² : G'.CommShift A
    inst✝¹ : H.CommShift A
    inst✝ : CategoryTheory.NatTrans.CommShift e.hom A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F₂.commShiftIso a).hom (CategoryThe …
  -/
  dsimp
  rw [← cancel_epi (e.hom.app (X⟦a⟧)), e.hom_inv_id_app_assoc, ← shift_app_comm_assoc,
    ← Functor.map_comp, e.hom_inv_id_app, Functor.map_id, Category.comp_id]⟩


lemma of_isIso [IsIso τ] [NatTrans.CommShift τ A] :
    NatTrans.CommShift (inv τ) A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} D
    F₁ F₂ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    A : Type u_5
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    inst✝⁴ : CategoryTheory.HasShift D A
    inst✝³ : F₁.CommShift A
    inst✝² : F₂.CommShift A
    inst✝¹ : CategoryTheory.IsIso τ
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    ⊢ CategoryTheory.NatTrans.CommShift (CategoryTheory.inv τ) A
  -/
  haveI : NatTrans.CommShift (asIso τ).hom A := by assumption
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} D
    F₁ F₂ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    A : Type u_5
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    inst✝⁴ : CategoryTheory.HasShift D A
    inst✝³ : F₁.CommShift A
    inst✝² : F₂.CommShift A
    inst✝¹ : CategoryTheory.IsIso τ
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    this : CategoryTheory.NatTrans.CommShift (CategoryTheory.asIso τ).hom A
    ⊢ CategoryTheory.NatTrans.CommShift (CategoryTheory.inv τ) A
  -/
  change NatTrans.CommShift (asIso τ).inv A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} D
    F₁ F₂ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    A : Type u_5
    inst✝⁶ : AddMonoid A
    inst✝⁵ : CategoryTheory.HasShift C A
    inst✝⁴ : CategoryTheory.HasShift D A
    inst✝³ : F₁.CommShift A
    inst✝² : F₂.CommShift A
    inst✝¹ : CategoryTheory.IsIso τ
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    this : CategoryTheory.NatTrans.CommShift (CategoryTheory.asIso τ).hom A
    ⊢ CategoryTheory.NatTrans.CommShift (CategoryTheory.asIso τ).inv A
  -/
  infer_instance
  /-
    🎉 no goals
  -/


variable (F₁) in
instance id : NatTrans.CommShift (𝟙 F₁) A where


instance comp [NatTrans.CommShift τ A] [NatTrans.CommShift τ' A] :
    NatTrans.CommShift (τ ≫ τ') A where


instance whiskerRight [NatTrans.CommShift τ A] :
    NatTrans.CommShift (whiskerRight τ G) A := ⟨fun a => by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    J : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{u_6, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_7, u_2} D
    inst✝¹³ : CategoryTheory.Category.{u_8, u_3} E
    inst✝¹² : CategoryTheory.Category.{?u.142126, u_4} J
    F₁ F₂ F₃ : CategoryTheory.Functor C D
    τ : Quiver.Hom F₁ F₂
    τ' : Quiver.Hom F₂ F₃
    e : CategoryTheory.Iso F₁ F₂
    G G' : CategoryTheory.Functor D E
    τ'' : Quiver.Hom G G'
    H : CategoryTheory.Functor E J
    A : Type u_5
    inst✝¹¹ : AddMonoid A
    inst✝¹⁰ : CategoryTheory.HasShift C A
    inst✝⁹ : CategoryTheory.HasShift D A
    inst✝⁸ : CategoryTheory.HasShift E A
    inst✝⁷ : CategoryTheory.HasShift J A
    inst✝⁶ : F₁.CommShift A
    inst✝⁵ : F₂.CommShift A
    inst✝⁴ : F₃.CommShift A
    inst✝³ : G.CommShift A
    inst✝² : G'.CommShift A
    inst✝¹ : H.CommShift A
    inst✝ : CategoryTheory.NatTrans.CommShift τ A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F₁.comp G).commShiftIso a).hom (Cat …
  -/
  ext X
  simp only [whiskerRight_twice, comp_app,
    whiskerRight_app, Functor.comp_map, whiskerLeft_app,
    Functor.commShiftIso_comp_hom_app, Category.assoc,
    ← Functor.commShiftIso_hom_naturality,
    ← G.map_comp_assoc, shift_app_comm]⟩


instance whiskerLeft [NatTrans.CommShift τ'' A] :
    NatTrans.CommShift (whiskerLeft F₁ τ'') A where


instance associator : CommShift (Functor.associator F₁ G H).hom A where


instance leftUnitor : CommShift F₁.leftUnitor.hom A where


instance rightUnitor : CommShift F₁.rightUnitor.hom A where


/-- If `e : F ≅ G` is an isomorphism of functors and if `F` commutes with the
shift, then `G` also commutes with the shift. -/
def ofIso : G.CommShift A where
  iso a := isoWhiskerLeft _ e.symm ≪≫ F.commShiftIso a ≪≫ isoWhiskerRight e _
  zero := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.172674, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.172678, u_2} D
      F G : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F G
      A : Type u_4
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      ⊢ Eq ((fun a => (CategoryTheory.isoWhiskerLeft (CategoryTheory.shiftFunctor C  …
    -/
    ext X
    simp only [comp_obj, F.commShiftIso_zero A, Iso.trans_hom, isoWhiskerLeft_hom,
      Iso.symm_hom, isoWhiskerRight_hom, NatTrans.comp_app, whiskerLeft_app,
      isoZero_hom_app, whiskerRight_app, assoc]
    erw [← e.inv.naturality_assoc, ← NatTrans.naturality,
      e.inv_hom_id_app_assoc]
  add a b := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.172674, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.172678, u_2} D
      F G : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F G
      A : Type u_4
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      ⊢ Eq ((fun a => (CategoryTheory.isoWhiskerLeft (CategoryTheory.shiftFunctor C  …
    -/
    ext X
    simp only [comp_obj, F.commShiftIso_add, Iso.trans_hom, isoWhiskerLeft_hom,
      Iso.symm_hom, isoWhiskerRight_hom, NatTrans.comp_app, whiskerLeft_app,
      isoAdd_hom_app, whiskerRight_app, assoc, map_comp, NatTrans.naturality_assoc,
      NatIso.cancel_natIso_inv_left]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.172674, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.172678, u_2} D
      F G : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F G
      A : Type u_4
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorA …
    -/
    simp only [← Functor.map_comp_assoc, e.hom_inv_id_app_assoc]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.172674, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.172678, u_2} D
      F G : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F G
      A : Type u_4
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorA …
    -/
    simp only [← NatTrans.naturality, comp_obj, comp_map, map_comp, assoc]
    /-
      🎉 no goals
    -/


lemma ofIso_compatibility :
    letI := ofIso e A
    NatTrans.CommShift e.hom A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    ⊢ CategoryTheory.NatTrans.CommShift e.hom A
  -/
  letI := ofIso e A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    this : G.CommShift A := CategoryTheory.Functor.CommShift.ofIso e A
    ⊢ CategoryTheory.NatTrans.CommShift e.hom A
  -/
  refine ⟨fun a => ?_⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    this : G.CommShift A := CategoryTheory.Functor.CommShift.ofIso e A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.commShiftIso a).hom (CategoryTheor …
  -/
  dsimp [commShiftIso, ofIso]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    this : G.CommShift A := CategoryTheory.Functor.CommShift.ofIso e A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.CommShift.iso …
  -/
  rw [← whiskerLeft_comp_assoc, e.hom_inv_id, whiskerLeft_id', id_comp]
  /-
    🎉 no goals
  -/


/--
Assume that we have a diagram of categories
```
C₁ ⥤ D₁
‖     ‖
v     v
C₂ ⥤ D₂
‖     ‖
v     v
C₃ ⥤ D₃
```
with functors `F₁₂ : C₁ ⥤ C₂`, `F₂₃ : C₂ ⥤ C₃` and `F₁₃ : C₁ ⥤ C₃` on the first
column that are related by a natural transformation `α : F₁₃ ⟶ F₁₂ ⋙ F₂₃`
and similarly `β : G₁₂ ⋙ G₂₃ ⟶ G₁₃` on the second column. Assume that we have
natural transformations
`e₁₂ : F₁₂ ⋙ L₂ ⟶ L₁ ⋙ G₁₂` (top square), `e₂₃ : F₂₃ ⋙ L₃ ⟶ L₂ ⋙ G₂₃` (bottom square),
and `e₁₃ : F₁₃ ⋙ L₃ ⟶ L₁ ⋙ G₁₃` (outer square), where the horizontal functors
are denoted `L₁`, `L₂` and `L₃`. Assume that `e₁₃` is determined by the other
natural transformations `α`, `e₂₃`, `e₁₂` and `β`. Then, if all these categories
are equipped with a shift by an additive monoid `A`, and all these functors commute with
these shifts, then the natural transformation `e₁₃` of the outer square commutes with the
shift if all `α`, `e₂₃`, `e₁₂` and `β` do. -/
lemma NatTrans.CommShift.verticalComposition {C₁ C₂ C₃ D₁ D₂ D₃ : Type*}
    [Category C₁] [Category C₂] [Category C₃] [Category D₁] [Category D₂] [Category D₃]
    {F₁₂ : C₁ ⥤ C₂} {F₂₃ : C₂ ⥤ C₃} {F₁₃ : C₁ ⥤ C₃} (α : F₁₃ ⟶ F₁₂ ⋙ F₂₃)
    {G₁₂ : D₁ ⥤ D₂} {G₂₃ : D₂ ⥤ D₃} {G₁₃ : D₁ ⥤ D₃} (β : G₁₂ ⋙ G₂₃ ⟶ G₁₃)
    {L₁ : C₁ ⥤ D₁} {L₂ : C₂ ⥤ D₂} {L₃ : C₃ ⥤ D₃}
    (e₁₂ : F₁₂ ⋙ L₂ ⟶ L₁ ⋙ G₁₂) (e₂₃ : F₂₃ ⋙ L₃ ⟶ L₂ ⋙ G₂₃) (e₁₃ : F₁₃ ⋙ L₃ ⟶ L₁ ⋙ G₁₃)
    (A : Type*) [AddMonoid A] [HasShift C₁ A] [HasShift C₂ A] [HasShift C₃ A]
    [HasShift D₁ A] [HasShift D₂ A] [HasShift D₃ A]
    [F₁₂.CommShift A] [F₂₃.CommShift A] [F₁₃.CommShift A] [CommShift α A]
    [G₁₂.CommShift A] [G₂₃.CommShift A] [G₁₃.CommShift A] [CommShift β A]
    [L₁.CommShift A] [L₂.CommShift A] [L₃.CommShift A]
    [CommShift e₁₂ A] [CommShift e₂₃ A]
    (h₁₃ : e₁₃ = CategoryTheory.whiskerRight α L₃ ≫ (Functor.associator _ _ _).hom ≫
      CategoryTheory.whiskerLeft F₁₂ e₂₃ ≫ (Functor.associator _ _ _).inv ≫
        CategoryTheory.whiskerRight e₁₂ G₂₃ ≫ (Functor.associator _ _ _).hom ≫
          CategoryTheory.whiskerLeft L₁ β) : CommShift e₁₃ A := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝²⁵ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝²⁴ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝²³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝²² : CategoryTheory.Category.{u_11, u_4} D₁
    inst✝²¹ : CategoryTheory.Category.{u_12, u_5} D₂
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} D₃
    F₁₂ : CategoryTheory.Functor C₁ C₂
    F₂₃ : CategoryTheory.Functor C₂ C₃
    F₁₃ : CategoryTheory.Functor C₁ C₃
    α : Quiver.Hom F₁₃ (F₁₂.comp F₂₃)
    G₁₂ : CategoryTheory.Functor D₁ D₂
    G₂₃ : CategoryTheory.Functor D₂ D₃
    G₁₃ : CategoryTheory.Functor D₁ D₃
    β : Quiver.Hom (G₁₂.comp G₂₃) G₁₃
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    L₃ : CategoryTheory.Functor C₃ D₃
    e₁₂ : Quiver.Hom (F₁₂.comp L₂) (L₁.comp G₁₂)
    e₂₃ : Quiver.Hom (F₂₃.comp L₃) (L₂.comp G₂₃)
    e₁₃ : Quiver.Hom (F₁₃.comp L₃) (L₁.comp G₁₃)
    A : Type u_7
    inst✝¹⁹ : AddMonoid A
    inst✝¹⁸ : CategoryTheory.HasShift C₁ A
    inst✝¹⁷ : CategoryTheory.HasShift C₂ A
    inst✝¹⁶ : CategoryTheory.HasShift C₃ A
    inst✝¹⁵ : CategoryTheory.HasShift D₁ A
    inst✝¹⁴ : CategoryTheory.HasShift D₂ A
    inst✝¹³ : CategoryTheory.HasShift D₃ A
    inst✝¹² : F₁₂.CommShift A
    inst✝¹¹ : F₂₃.CommShift A
    inst✝¹⁰ : F₁₃.CommShift A
    inst✝⁹ : CategoryTheory.NatTrans.CommShift α A
    inst✝⁸ : G₁₂.CommShift A
    inst✝⁷ : G₂₃.CommShift A
    inst✝⁶ : G₁₃.CommShift A
    inst✝⁵ : CategoryTheory.NatTrans.CommShift β A
    inst✝⁴ : L₁.CommShift A
    inst✝³ : L₂.CommShift A
    inst✝² : L₃.CommShift A
    inst✝¹ : CategoryTheory.NatTrans.CommShift e₁₂ A
    inst✝ : CategoryTheory.NatTrans.CommShift e₂₃ A
    h₁₃ : Eq e₁₃ (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight  …
    ⊢ CategoryTheory.NatTrans.CommShift e₁₃ A
  -/
  subst h₁₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝²⁵ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝²⁴ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝²³ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝²² : CategoryTheory.Category.{u_11, u_4} D₁
    inst✝²¹ : CategoryTheory.Category.{u_12, u_5} D₂
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} D₃
    F₁₂ : CategoryTheory.Functor C₁ C₂
    F₂₃ : CategoryTheory.Functor C₂ C₃
    F₁₃ : CategoryTheory.Functor C₁ C₃
    α : Quiver.Hom F₁₃ (F₁₂.comp F₂₃)
    G₁₂ : CategoryTheory.Functor D₁ D₂
    G₂₃ : CategoryTheory.Functor D₂ D₃
    G₁₃ : CategoryTheory.Functor D₁ D₃
    β : Quiver.Hom (G₁₂.comp G₂₃) G₁₃
    L₁ : CategoryTheory.Functor C₁ D₁
    L₂ : CategoryTheory.Functor C₂ D₂
    L₃ : CategoryTheory.Functor C₃ D₃
    e₁₂ : Quiver.Hom (F₁₂.comp L₂) (L₁.comp G₁₂)
    e₂₃ : Quiver.Hom (F₂₃.comp L₃) (L₂.comp G₂₃)
    A : Type u_7
    inst✝¹⁹ : AddMonoid A
    inst✝¹⁸ : CategoryTheory.HasShift C₁ A
    inst✝¹⁷ : CategoryTheory.HasShift C₂ A
    inst✝¹⁶ : CategoryTheory.HasShift C₃ A
    inst✝¹⁵ : CategoryTheory.HasShift D₁ A
    inst✝¹⁴ : CategoryTheory.HasShift D₂ A
    inst✝¹³ : CategoryTheory.HasShift D₃ A
    inst✝¹² : F₁₂.CommShift A
    inst✝¹¹ : F₂₃.CommShift A
    inst✝¹⁰ : F₁₃.CommShift A
    inst✝⁹ : CategoryTheory.NatTrans.CommShift α A
    inst✝⁸ : G₁₂.CommShift A
    inst✝⁷ : G₂₃.CommShift A
    inst✝⁶ : G₁₃.CommShift A
    inst✝⁵ : CategoryTheory.NatTrans.CommShift β A
    inst✝⁴ : L₁.CommShift A
    inst✝³ : L₂.CommShift A
    inst✝² : L₃.CommShift A
    inst✝¹ : CategoryTheory.NatTrans.CommShift e₁₂ A
    inst✝ : CategoryTheory.NatTrans.CommShift e₂₃ A
    ⊢ CategoryTheory.NatTrans.CommShift (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


