/-- The type of families of functors `A → C ⥤ D` which are compatible with
the shift by `A` on the category `D`. -/
structure SingleFunctors where
  /-- a family of functors `C ⥤ D` indexed by the elements of the additive monoid `A` -/
  functor (a : A) : C ⥤ D
  /-- the isomorphism `functor a' ⋙ shiftFunctor D n ≅ functor a` when `n + a = a'` -/
  shiftIso (n a a' : A) (ha' : n + a = a') : functor a' ⋙ shiftFunctor D n ≅ functor a
  /-- `shiftIso 0` is the obvious isomorphism. -/
  shiftIso_zero (a : A) :
    shiftIso 0 a a (zero_add a) = isoWhiskerLeft _ (shiftFunctorZero D A)
  /-- `shiftIso (m + n)` is determined by `shiftIso m` and `shiftIso n`. -/
  shiftIso_add (n m a a' a'' : A) (ha' : n + a = a') (ha'' : m + a' = a'') :
                               /-
                                 C : Type u_1
                                 D : Type u_2
                                 E : Type u_3
                                 E' : Type u_4
                                 inst✝⁷ : CategoryTheory.Category.{?u.122, u_1} C
                                 inst✝⁶ : CategoryTheory.Category.{?u.126, u_2} D
                                 inst✝⁵ : CategoryTheory.Category.{?u.130, u_3} E
                                 inst✝⁴ : CategoryTheory.Category.{?u.134, u_4} E'
                                 A : Type u_5
                                 inst✝³ : AddMonoid A
                                 inst✝² : CategoryTheory.HasShift D A
                                 inst✝¹ : CategoryTheory.HasShift E A
                                 inst✝ : CategoryTheory.HasShift E' A
                                 functor : A → CategoryTheory.Functor C D
                                 shiftIso : (n a a' : A) → Eq (HAdd.hAdd n a) a' → CategoryTheory.Iso ((functor …
                                 shiftIso_zero : ∀ (a : A), Eq (shiftIso 0 a a ⋯) (CategoryTheory.isoWhiskerLef …
                                 n m a a' a'' : A
                                 ha' : Eq (HAdd.hAdd n a) a'
                                 ha'' : Eq (HAdd.hAdd m a') a''
                                 ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                               -/
    shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha'']) =
                               /-
                                 🎉 no goals
                               -/
      isoWhiskerLeft _ (shiftFunctorAdd D m n) ≪≫ (Functor.associator _ _ _).symm ≪≫
        isoWhiskerRight (shiftIso m a' a'' ha'') _ ≪≫ shiftIso n a a' ha'


lemma shiftIso_add_hom_app (n m a a' a'' : A) (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                                  /-
                                    C : Type u_1
                                    D : Type u_2
                                    E : Type u_3
                                    E' : Type u_4
                                    inst✝⁷ : CategoryTheory.Category.{?u.4857, u_1} C
                                    inst✝⁶ : CategoryTheory.Category.{?u.4861, u_2} D
                                    inst✝⁵ : CategoryTheory.Category.{?u.4865, u_3} E
                                    inst✝⁴ : CategoryTheory.Category.{?u.4869, u_4} E'
                                    A : Type u_5
                                    inst✝³ : AddMonoid A
                                    inst✝² : CategoryTheory.HasShift D A
                                    inst✝¹ : CategoryTheory.HasShift E A
                                    inst✝ : CategoryTheory.HasShift E' A
                                    F G H : CategoryTheory.SingleFunctors C D A
                                    n m a a' a'' : A
                                    ha' : Eq (HAdd.hAdd n a) a'
                                    ha'' : Eq (HAdd.hAdd m a') a''
                                    X : C
                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                                  -/
    (F.shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha''])).hom.app X =
                                  /-
                                    🎉 no goals
                                  -/
      (shiftFunctorAdd D m n).hom.app ((F.functor a'').obj X) ≫
        ((F.shiftIso m a' a'' ha'').hom.app X)⟦n⟧' ≫
        (F.shiftIso n a a' ha').hom.app X := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    n m a a' a'' : A
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso (HAdd.hAdd m n) a a'' ⋯).hom.app X) (CategoryTheory.Category …
  -/
  simp [F.shiftIso_add n m a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


lemma shiftIso_add_inv_app (n m a a' a'' : A) (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                                  /-
                                    C : Type u_1
                                    D : Type u_2
                                    E : Type u_3
                                    E' : Type u_4
                                    inst✝⁷ : CategoryTheory.Category.{?u.9373, u_1} C
                                    inst✝⁶ : CategoryTheory.Category.{?u.9377, u_2} D
                                    inst✝⁵ : CategoryTheory.Category.{?u.9381, u_3} E
                                    inst✝⁴ : CategoryTheory.Category.{?u.9385, u_4} E'
                                    A : Type u_5
                                    inst✝³ : AddMonoid A
                                    inst✝² : CategoryTheory.HasShift D A
                                    inst✝¹ : CategoryTheory.HasShift E A
                                    inst✝ : CategoryTheory.HasShift E' A
                                    F G H : CategoryTheory.SingleFunctors C D A
                                    n m a a' a'' : A
                                    ha' : Eq (HAdd.hAdd n a) a'
                                    ha'' : Eq (HAdd.hAdd m a') a''
                                    X : C
                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                                  -/
    (F.shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha''])).inv.app X =
                                  /-
                                    🎉 no goals
                                  -/
      (F.shiftIso n a a' ha').inv.app X ≫
      ((F.shiftIso m a' a'' ha'').inv.app X)⟦n⟧' ≫
      (shiftFunctorAdd D m n).inv.app ((F.functor a'').obj X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    n m a a' a'' : A
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso (HAdd.hAdd m n) a a'' ⋯).inv.app X) (CategoryTheory.Category …
  -/
  simp [F.shiftIso_add n m a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


lemma shiftIso_add' (n m mn : A) (hnm : m + n = mn) (a a' a'' : A)
    (ha' : n + a = a') (ha'' : m + a' = a'') :
                            /-
                              C : Type u_1
                              D : Type u_2
                              E : Type u_3
                              E' : Type u_4
                              inst✝⁷ : CategoryTheory.Category.{?u.13893, u_1} C
                              inst✝⁶ : CategoryTheory.Category.{?u.13897, u_2} D
                              inst✝⁵ : CategoryTheory.Category.{?u.13901, u_3} E
                              inst✝⁴ : CategoryTheory.Category.{?u.13905, u_4} E'
                              A : Type u_5
                              inst✝³ : AddMonoid A
                              inst✝² : CategoryTheory.HasShift D A
                              inst✝¹ : CategoryTheory.HasShift E A
                              inst✝ : CategoryTheory.HasShift E' A
                              F G H : CategoryTheory.SingleFunctors C D A
                              n m mn : A
                              hnm : Eq (HAdd.hAdd m n) mn
                              a a' a'' : A
                              ha' : Eq (HAdd.hAdd n a) a'
                              ha'' : Eq (HAdd.hAdd m a') a''
                              ⊢ Eq (HAdd.hAdd mn a) a''
                            -/
    F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc]) =
                            /-
                              🎉 no goals
                            -/
      isoWhiskerLeft _ (shiftFunctorAdd' D m n mn hnm) ≪≫ (Functor.associator _ _ _).symm ≪≫
        isoWhiskerRight (F.shiftIso m a' a'' ha'') _ ≪≫ F.shiftIso n a a' ha' := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    n m mn : A
    hnm : Eq (HAdd.hAdd m n) mn
    a a' a'' : A
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    ⊢ Eq (F.shiftIso mn a a'' ⋯) ((CategoryTheory.isoWhiskerLeft (F.functor a'') ( …
  -/
  subst hnm
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    n m a a' a'' : A
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    ⊢ Eq (F.shiftIso (HAdd.hAdd m n) a a'' ⋯) ((CategoryTheory.isoWhiskerLeft (F.f …
  -/
  rw [shiftFunctorAdd'_eq_shiftFunctorAdd, shiftIso_add]
  /-
    🎉 no goals
  -/


lemma shiftIso_add'_hom_app (n m mn : A) (hnm : m + n = mn) (a a' a'' : A)
    (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                             /-
                               C : Type u_1
                               D : Type u_2
                               E : Type u_3
                               E' : Type u_4
                               inst✝⁷ : CategoryTheory.Category.{?u.16840, u_1} C
                               inst✝⁶ : CategoryTheory.Category.{?u.16844, u_2} D
                               inst✝⁵ : CategoryTheory.Category.{?u.16848, u_3} E
                               inst✝⁴ : CategoryTheory.Category.{?u.16852, u_4} E'
                               A : Type u_5
                               inst✝³ : AddMonoid A
                               inst✝² : CategoryTheory.HasShift D A
                               inst✝¹ : CategoryTheory.HasShift E A
                               inst✝ : CategoryTheory.HasShift E' A
                               F G H : CategoryTheory.SingleFunctors C D A
                               n m mn : A
                               hnm : Eq (HAdd.hAdd m n) mn
                               a a' a'' : A
                               ha' : Eq (HAdd.hAdd n a) a'
                               ha'' : Eq (HAdd.hAdd m a') a''
                               X : C
                               ⊢ Eq (HAdd.hAdd mn a) a''
                             -/
    (F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc])).hom.app X =
                             /-
                               🎉 no goals
                             -/
      (shiftFunctorAdd' D m n mn hnm).hom.app ((F.functor a'').obj X) ≫
        ((F.shiftIso m a' a'' ha'').hom.app X)⟦n⟧' ≫ (F.shiftIso n a a' ha').hom.app X := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    n m mn : A
    hnm : Eq (HAdd.hAdd m n) mn
    a a' a'' : A
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso mn a a'' ⋯).hom.app X) (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [F.shiftIso_add' n m mn hnm a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


lemma shiftIso_add'_inv_app (n m mn : A) (hnm : m + n = mn) (a a' a'' : A)
    (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                             /-
                               C : Type u_1
                               D : Type u_2
                               E : Type u_3
                               E' : Type u_4
                               inst✝⁷ : CategoryTheory.Category.{?u.21417, u_1} C
                               inst✝⁶ : CategoryTheory.Category.{?u.21421, u_2} D
                               inst✝⁵ : CategoryTheory.Category.{?u.21425, u_3} E
                               inst✝⁴ : CategoryTheory.Category.{?u.21429, u_4} E'
                               A : Type u_5
                               inst✝³ : AddMonoid A
                               inst✝² : CategoryTheory.HasShift D A
                               inst✝¹ : CategoryTheory.HasShift E A
                               inst✝ : CategoryTheory.HasShift E' A
                               F G H : CategoryTheory.SingleFunctors C D A
                               n m mn : A
                               hnm : Eq (HAdd.hAdd m n) mn
                               a a' a'' : A
                               ha' : Eq (HAdd.hAdd n a) a'
                               ha'' : Eq (HAdd.hAdd m a') a''
                               X : C
                               ⊢ Eq (HAdd.hAdd mn a) a''
                             -/
    (F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc])).inv.app X =
                             /-
                               🎉 no goals
                             -/
        (F.shiftIso n a a' ha').inv.app X ≫
        ((F.shiftIso m a' a'' ha'').inv.app X)⟦n⟧' ≫
      (shiftFunctorAdd' D m n mn hnm).inv.app ((F.functor a'').obj X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    n m mn : A
    hnm : Eq (HAdd.hAdd m n) mn
    a a' a'' : A
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso mn a a'' ⋯).inv.app X) (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [F.shiftIso_add' n m mn hnm a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftIso_zero_hom_app (a : A) (X : C) :
    (F.shiftIso 0 a a (zero_add a)).hom.app X = (shiftFunctorZero D A).hom.app _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    a : A
    X : C
    ⊢ Eq ((F.shiftIso 0 a a ⋯).hom.app X) ((CategoryTheory.shiftFunctorZero D A).h …
  -/
  rw [shiftIso_zero]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.isoWhiskerLeft (F.functor a) (CategoryTheory.shiftFuncto …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftIso_zero_inv_app (a : A) (X : C) :
    (F.shiftIso 0 a a (zero_add a)).inv.app X = (shiftFunctorZero D A).inv.app _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    a : A
    X : C
    ⊢ Eq ((F.shiftIso 0 a a ⋯).inv.app X) ((CategoryTheory.shiftFunctorZero D A).i …
  -/
  rw [shiftIso_zero]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.SingleFunctors C D A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.isoWhiskerLeft (F.functor a) (CategoryTheory.shiftFuncto …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The morphisms in the category `SingleFunctors C D A` -/
@[ext]
structure Hom where
  /-- a family of natural transformations `F.functor a ⟶ G.functor a` -/
  hom (a : A) : F.functor a ⟶ G.functor a
  comm (n a a' : A) (ha' : n + a = a') : (F.shiftIso n a a' ha').hom ≫ hom a =
    whiskerRight (hom a') (shiftFunctor D n) ≫ (G.shiftIso n a a' ha').hom := by aesop_cat


attribute [reassoc] comm

/-- The identity morphism in `SingleFunctors C D A`. -/
@[simps]
def id : Hom F F where
  hom _ := 𝟙 _


/-- The composition of morphisms in `SingleFunctors C D A`. -/
@[simps]
def comp (α : Hom F G) (β : Hom G H) : Hom F H where
  hom a := α.hom a ≫ β.hom a


instance : Category (SingleFunctors C D A) where
  Hom := Hom
  id := Hom.id
  comp := Hom.comp


@[simp]
lemma id_hom (a : A) : Hom.hom (𝟙 F) a = 𝟙 _ := rfl


@[simp, reassoc]
lemma comp_hom (f : F ⟶ G) (g : G ⟶ H) (a : A) : (f ≫ g).hom a = f.hom a ≫ g.hom a := rfl


@[ext]
lemma hom_ext (f g : F ⟶ G) (h : f.hom = g.hom) : f = g := Hom.ext h


/-- Construct an isomorphism in `SingleFunctors C D A` by giving
level-wise isomorphisms and checking compatibility only in the forward direction. -/
@[simps]
def isoMk (iso : ∀ a, (F.functor a ≅ G.functor a))
    (comm : ∀ (n a a' : A) (ha' : n + a = a'), (F.shiftIso n a a' ha').hom ≫ (iso a).hom =
      whiskerRight (iso a').hom (shiftFunctor D n) ≫ (G.shiftIso n a a' ha').hom) :
    F ≅ G where
  hom :=
    { hom := fun a => (iso a).hom
      comm := comm }
  inv :=
    { hom := fun a => (iso a).inv
      comm := fun n a a' ha' => by
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          E' : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.42456, u_1} C
          inst✝⁶ : CategoryTheory.Category.{?u.42460, u_2} D
          inst✝⁵ : CategoryTheory.Category.{?u.42464, u_3} E
          inst✝⁴ : CategoryTheory.Category.{?u.42468, u_4} E'
          A : Type u_5
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift D A
          inst✝¹ : CategoryTheory.HasShift E A
          inst✝ : CategoryTheory.HasShift E' A
          F G H : CategoryTheory.SingleFunctors C D A
          iso : (a : A) → CategoryTheory.Iso (F.functor a) (G.functor a)
          comm : ∀ (n a a' : A) (ha' : Eq (HAdd.hAdd n a) a'), Eq (CategoryTheory.Catego …
          n a a' : A
          ha' : Eq (HAdd.hAdd n a) a'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.shiftIso n a a' ha').hom ((fun a = …
        -/
        dsimp only
        rw [← cancel_mono (iso a).hom, assoc, assoc, Iso.inv_hom_id, comp_id, comm,
          ← whiskerRight_comp_assoc, Iso.inv_hom_id, whiskerRight_id', id_comp] }


/-- The evaluation `SingleFunctors C D A ⥤ C ⥤ D` for some `a : A`. -/
@[simps]
def evaluation (a : A) : SingleFunctors C D A ⥤ C ⥤ D where
  obj F := F.functor a
  map {_ _} φ := φ.hom a


@[reassoc (attr := simp)]
lemma hom_inv_id_hom (e : F ≅ G) (n : A) : e.hom.hom n ≫ e.inv.hom n = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F G : CategoryTheory.SingleFunctors C D A
    e : CategoryTheory.Iso F G
    n : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.hom.hom n) (e.inv.hom n)) (Categor …
  -/
  rw [← comp_hom, e.hom_inv_id, id_hom]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_hom_id_hom (e : F ≅ G) (n : A) : e.inv.hom n ≫ e.hom.hom n = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F G : CategoryTheory.SingleFunctors C D A
    e : CategoryTheory.Iso F G
    n : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.inv.hom n) (e.hom.hom n)) (Categor …
  -/
  rw [← comp_hom, e.inv_hom_id, id_hom]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma hom_inv_id_hom_app (e : F ≅ G) (n : A) (X : C) :
    (e.hom.hom n).app X ≫ (e.inv.hom n).app X = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F G : CategoryTheory.SingleFunctors C D A
    e : CategoryTheory.Iso F G
    n : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e.hom.hom n).app X) ((e.inv.hom n). …
  -/
  rw [← NatTrans.comp_app, hom_inv_id_hom, NatTrans.id_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_hom_id_hom_app (e : F ≅ G) (n : A) (X : C) :
    (e.inv.hom n).app X ≫ (e.hom.hom n).app X = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    A : Type u_5
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F G : CategoryTheory.SingleFunctors C D A
    e : CategoryTheory.Iso F G
    n : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e.inv.hom n).app X) ((e.hom.hom n). …
  -/
  rw [← NatTrans.comp_app, inv_hom_id_hom, NatTrans.id_app]
  /-
    🎉 no goals
  -/


instance (f : F ⟶ G) [IsIso f] (n : A) : IsIso (f.hom n) :=
  (inferInstance : IsIso ((evaluation C D n).map f))


/-- Given `F : SingleFunctors C D A`, and a functor `G : D ⥤ E` which commutes
with the shift by `A`, this is the "composition" of `F` and `G` in `SingleFunctors C E A`. -/
@[simps! functor shiftIso_hom_app shiftIso_inv_app]
def postcomp (G : D ⥤ E) [G.CommShift A] :
    SingleFunctors C E A where
  functor a := F.functor a ⋙ G
  shiftIso n a a' ha' :=
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (G.commShiftIso n).symm ≪≫
      (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight (F.shiftIso n a a' ha') G
  shiftIso_zero a := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁸ : CategoryTheory.Category.{?u.64646, u_1} C
      inst✝⁷ : CategoryTheory.Category.{?u.64650, u_2} D
      inst✝⁶ : CategoryTheory.Category.{?u.64654, u_3} E
      inst✝⁵ : CategoryTheory.Category.{?u.64658, u_4} E'
      A : Type u_5
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      inst✝ : G.CommShift A
      a : A
      ⊢ Eq ((fun n a a' ha' => ((F.functor a').associator G (CategoryTheory.shiftFun …
    -/
    ext X
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁸ : CategoryTheory.Category.{?u.64646, u_1} C
      inst✝⁷ : CategoryTheory.Category.{?u.64650, u_2} D
      inst✝⁶ : CategoryTheory.Category.{?u.64654, u_3} E
      inst✝⁵ : CategoryTheory.Category.{?u.64658, u_4} E'
      A : Type u_5
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      inst✝ : G.CommShift A
      a : A
      X : C
      ⊢ Eq (((fun n a a' ha' => ((F.functor a').associator G (CategoryTheory.shiftFu …
    -/
    dsimp
    simp only [Functor.commShiftIso_zero, Functor.CommShift.isoZero_inv_app,
      SingleFunctors.shiftIso_zero_hom_app,id_comp, assoc, ← G.map_comp, Iso.inv_hom_id_app,
      Functor.map_id, Functor.id_obj, comp_id]
  shiftIso_add n m a a' a'' ha' ha'' := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁸ : CategoryTheory.Category.{?u.64646, u_1} C
      inst✝⁷ : CategoryTheory.Category.{?u.64650, u_2} D
      inst✝⁶ : CategoryTheory.Category.{?u.64654, u_3} E
      inst✝⁵ : CategoryTheory.Category.{?u.64658, u_4} E'
      A : Type u_5
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      inst✝ : G.CommShift A
      n m a a' a'' : A
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq ((fun n a a' ha' => ((F.functor a').associator G (CategoryTheory.shiftFun …
    -/
    ext X
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁸ : CategoryTheory.Category.{?u.64646, u_1} C
      inst✝⁷ : CategoryTheory.Category.{?u.64650, u_2} D
      inst✝⁶ : CategoryTheory.Category.{?u.64654, u_3} E
      inst✝⁵ : CategoryTheory.Category.{?u.64658, u_4} E'
      A : Type u_5
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      inst✝ : G.CommShift A
      n m a a' a'' : A
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      X : C
      ⊢ Eq (((fun n a a' ha' => ((F.functor a').associator G (CategoryTheory.shiftFu …
    -/
    dsimp
    simp only [F.shiftIso_add_hom_app n m a a' a'' ha' ha'', Functor.commShiftIso_add,
      Functor.CommShift.isoAdd_inv_app, Functor.map_comp, id_comp, assoc,
      Functor.commShiftIso_inv_naturality_assoc]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁸ : CategoryTheory.Category.{?u.64646, u_1} C
      inst✝⁷ : CategoryTheory.Category.{?u.64650, u_2} D
      inst✝⁶ : CategoryTheory.Category.{?u.64654, u_3} E
      inst✝⁵ : CategoryTheory.Category.{?u.64658, u_4} E'
      A : Type u_5
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift D A
      inst✝² : CategoryTheory.HasShift E A
      inst✝¹ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      inst✝ : G.CommShift A
      n m a a' a'' : A
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd E m  …
    -/
    simp only [← G.map_comp, Iso.inv_hom_id_app_assoc]
    /-
      🎉 no goals
    -/


/-- The functor `SingleFunctors C D A ⥤ SingleFunctors C E A` given by the postcomposition
by a functor `G : D ⥤ E` which commutes with the shift. -/
def postcompFunctor (G : D ⥤ E) [G.CommShift A] :
    SingleFunctors C D A ⥤ SingleFunctors C E A where
  obj F := F.postcomp G
  map {F₁ F₂} φ :=
    { hom := fun a => whiskerRight (φ.hom a) G
      comm := fun n a a' ha' => by
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          E' : Type u_4
          inst✝⁸ : CategoryTheory.Category.{?u.78662, u_1} C
          inst✝⁷ : CategoryTheory.Category.{?u.78666, u_2} D
          inst✝⁶ : CategoryTheory.Category.{?u.78670, u_3} E
          inst✝⁵ : CategoryTheory.Category.{?u.78674, u_4} E'
          A : Type u_5
          inst✝⁴ : AddMonoid A
          inst✝³ : CategoryTheory.HasShift D A
          inst✝² : CategoryTheory.HasShift E A
          inst✝¹ : CategoryTheory.HasShift E' A
          F G✝ H : CategoryTheory.SingleFunctors C D A
          G : CategoryTheory.Functor D E
          inst✝ : G.CommShift A
          F₁ F₂ : CategoryTheory.SingleFunctors C D A
          φ : Quiver.Hom F₁ F₂
          n a a' : A
          ha' : Eq (HAdd.hAdd n a) a'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun F => F.postcomp G) F₁).shiftIs …
        -/
        ext X
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          E : Type u_3
          E' : Type u_4
          inst✝⁸ : CategoryTheory.Category.{?u.78662, u_1} C
          inst✝⁷ : CategoryTheory.Category.{?u.78666, u_2} D
          inst✝⁶ : CategoryTheory.Category.{?u.78670, u_3} E
          inst✝⁵ : CategoryTheory.Category.{?u.78674, u_4} E'
          A : Type u_5
          inst✝⁴ : AddMonoid A
          inst✝³ : CategoryTheory.HasShift D A
          inst✝² : CategoryTheory.HasShift E A
          inst✝¹ : CategoryTheory.HasShift E' A
          F G✝ H : CategoryTheory.SingleFunctors C D A
          G : CategoryTheory.Functor D E
          inst✝ : G.CommShift A
          F₁ F₂ : CategoryTheory.SingleFunctors C D A
          φ : Quiver.Hom F₁ F₂
          n a a' : A
          ha' : Eq (HAdd.hAdd n a) a'
          X : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((fun F => F.postcomp G) F₁).shiftI …
        -/
        simpa using G.congr_map (congr_app (φ.comm n a a' ha') X) }
        /-
          🎉 no goals
        -/


/-- The canonical isomorphism `(F.postcomp G).postcomp G' ≅ F.postcomp (G ⋙ G')`. -/
@[simps!]
def postcompPostcompIso (G : D ⥤ E) (G' : E ⥤ E') [G.CommShift A] [G'.CommShift A] :
    (F.postcomp G).postcomp G' ≅ F.postcomp (G ⋙ G') :=
  isoMk (fun _ => Functor.associator _ _ _) (fun n a a' ha' => by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁹ : CategoryTheory.Category.{?u.94200, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.94204, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.94208, u_3} E
      inst✝⁶ : CategoryTheory.Category.{?u.94212, u_4} E'
      A : Type u_5
      inst✝⁵ : AddMonoid A
      inst✝⁴ : CategoryTheory.HasShift D A
      inst✝³ : CategoryTheory.HasShift E A
      inst✝² : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E E'
      inst✝¹ : G.CommShift A
      inst✝ : G'.CommShift A
      n a a' : A
      ha' : Eq (HAdd.hAdd n a) a'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.postcomp G).postcomp G').shiftIs …
    -/
    ext X
    /-
      case w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝⁹ : CategoryTheory.Category.{?u.94200, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.94204, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.94208, u_3} E
      inst✝⁶ : CategoryTheory.Category.{?u.94212, u_4} E'
      A : Type u_5
      inst✝⁵ : AddMonoid A
      inst✝⁴ : CategoryTheory.HasShift D A
      inst✝³ : CategoryTheory.HasShift E A
      inst✝² : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G : CategoryTheory.Functor D E
      G' : CategoryTheory.Functor E E'
      inst✝¹ : G.CommShift A
      inst✝ : G'.CommShift A
      n a a' : A
      ha' : Eq (HAdd.hAdd n a) a'
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((F.postcomp G).postcomp G').shiftI …
    -/
    simp [Functor.commShiftIso_comp_inv_app])
    /-
      🎉 no goals
    -/


/-- The isomorphism `F.postcomp G ≅ F.postcomp G'` induced by an isomorphism `e : G ≅ G'`
which commutes with the shift. -/
@[simps!]
def postcompIsoOfIso {G G' : D ⥤ E} (e : G ≅ G') [G.CommShift A] [G'.CommShift A]
    [NatTrans.CommShift e.hom A] :
    F.postcomp G ≅ F.postcomp G' :=
  isoMk (fun a => isoWhiskerLeft (F.functor a) e) (fun n a a' ha' => by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝¹⁰ : CategoryTheory.Category.{?u.103247, u_1} C
      inst✝⁹ : CategoryTheory.Category.{?u.103251, u_2} D
      inst✝⁸ : CategoryTheory.Category.{?u.103255, u_3} E
      inst✝⁷ : CategoryTheory.Category.{?u.103259, u_4} E'
      A : Type u_5
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G G' : CategoryTheory.Functor D E
      e : CategoryTheory.Iso G G'
      inst✝² : G.CommShift A
      inst✝¹ : G'.CommShift A
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom A
      n a a' : A
      ha' : Eq (HAdd.hAdd n a) a'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.postcomp G).shiftIso n a a' ha'). …
    -/
    ext X
    /-
      case w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝¹⁰ : CategoryTheory.Category.{?u.103247, u_1} C
      inst✝⁹ : CategoryTheory.Category.{?u.103251, u_2} D
      inst✝⁸ : CategoryTheory.Category.{?u.103255, u_3} E
      inst✝⁷ : CategoryTheory.Category.{?u.103259, u_4} E'
      A : Type u_5
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G G' : CategoryTheory.Functor D E
      e : CategoryTheory.Iso G G'
      inst✝² : G.CommShift A
      inst✝¹ : G'.CommShift A
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom A
      n a a' : A
      ha' : Eq (HAdd.hAdd n a) a'
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((F.postcomp G).shiftIso n a a' ha') …
    -/
    dsimp
    /-
      case w.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      E' : Type u_4
      inst✝¹⁰ : CategoryTheory.Category.{?u.103247, u_1} C
      inst✝⁹ : CategoryTheory.Category.{?u.103251, u_2} D
      inst✝⁸ : CategoryTheory.Category.{?u.103255, u_3} E
      inst✝⁷ : CategoryTheory.Category.{?u.103259, u_4} E'
      A : Type u_5
      inst✝⁶ : AddMonoid A
      inst✝⁵ : CategoryTheory.HasShift D A
      inst✝⁴ : CategoryTheory.HasShift E A
      inst✝³ : CategoryTheory.HasShift E' A
      F G✝ H : CategoryTheory.SingleFunctors C D A
      G G' : CategoryTheory.Functor D E
      e : CategoryTheory.Iso G G'
      inst✝² : G.CommShift A
      inst✝¹ : G'.CommShift A
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom A
      n a a' : A
      ha' : Eq (HAdd.hAdd n a) a'
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((F.postcomp G).shiftIso n a a' ha') …
    -/
    simp [NatTrans.shift_app e.hom n])
    /-
      🎉 no goals
    -/


