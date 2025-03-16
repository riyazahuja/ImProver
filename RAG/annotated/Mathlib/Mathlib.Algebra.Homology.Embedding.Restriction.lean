/-- Given `K : HomologicalComplex C c'` and `e : c.Embedding c'` (satisfying `[e.IsRelIff]`),
this is the homological complex in `HomologicalComplex C c` obtained by restriction. -/
@[simps]
def restriction : HomologicalComplex C c where
  X i := K.X (e.f i)
  d _ _ := K.d _ _
                                   /-
                                     ι : Type u_1
                                     ι' : Type u_2
                                     c : ComplexShape ι
                                     c' : ComplexShape ι'
                                     C : Type u_3
                                     inst✝² : CategoryTheory.Category.{?u.305, u_3} C
                                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                     K L M : HomologicalComplex C c'
                                     φ : Quiver.Hom K L
                                     φ' : Quiver.Hom L M
                                     e : c.Embedding c'
                                     inst✝ : e.IsRelIff
                                     i j : ι
                                     hij : Not (c.Rel i j)
                                     ⊢ Not (c'.Rel (e.f i) (e.f j))
                                   -/
  shape i j hij := K.shape _ _ (by simpa only [← e.rel_iff] using hij)
                                   /-
                                     🎉 no goals
                                   -/


/-- The isomorphism `(K.restriction e).X i ≅ K.X i'` when `e.f i = i'`. -/
def restrictionXIso {i : ι} {i' : ι'} (h : e.f i = i') :
    (K.restriction e).X i ≅ K.X i' :=
  eqToIso (h ▸ rfl)


@[reassoc]
lemma restriction_d_eq {i j : ι} {i' j' : ι'} (hi : e.f i = i') (hj : e.f j = j') :
    (K.restriction e).d i j = (K.restrictionXIso e hi).hom ≫ K.d i' j' ≫
      (K.restrictionXIso e hj).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    i' j' : ι'
    hi : Eq (e.f i) i'
    hj : Eq (e.f j) j'
    ⊢ Eq ((K.restriction e).d i j) (CategoryTheory.CategoryStruct.comp (K.restrict …
  -/
  subst hi hj
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i j : ι
    ⊢ Eq ((K.restriction e).d i j) (CategoryTheory.CategoryStruct.comp (K.restrict …
  -/
  simp [restrictionXIso]
  /-
    🎉 no goals
  -/


/-- The morphism `K.restriction e ⟶ L.restriction e` induced by a morphism `φ : K ⟶ L`. -/
@[simps]
def restrictionMap : K.restriction e ⟶ L.restriction e where
  f i := φ.f (e.f i)


@[reassoc]
lemma restrictionMap_f' {i : ι} {i' : ι'} (hi : e.f i = i') :
    (restrictionMap φ e).f i = (K.restrictionXIso e hi).hom ≫
      φ.f i' ≫ (L.restrictionXIso e hi).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i : ι
    i' : ι'
    hi : Eq (e.f i) i'
    ⊢ Eq ((HomologicalComplex.restrictionMap φ e).f i) (CategoryTheory.CategoryStr …
  -/
  subst hi
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i : ι
    ⊢ Eq ((HomologicalComplex.restrictionMap φ e).f i) (CategoryTheory.CategoryStr …
  -/
  simp [restrictionXIso]
  /-
    🎉 no goals
  -/


@[simp]
lemma restrictionMap_id : restrictionMap (𝟙 K) e = 𝟙 _ := rfl


@[simp, reassoc]
lemma restrictionMap_comp :
    restrictionMap (φ ≫ φ') e = restrictionMap φ e ≫ restrictionMap φ' e := rfl


/-- Given `e : ComplexShape.Embedding c c'`, this is the restriction
functor `HomologicalComplex C c' ⥤ HomologicalComplex C c`. -/
@[simps]
noncomputable def restrictionFunctor [HasZeroMorphisms C] :
    HomologicalComplex C c' ⥤ HomologicalComplex C c where
  obj K := K.restriction e
  map φ := HomologicalComplex.restrictionMap φ e


instance [HasZeroMorphisms C] : (e.restrictionFunctor C).PreservesZeroMorphisms where


instance [Preadditive C] : (e.restrictionFunctor C).Additive where


