/-- A shift sequence for a functor `F : C ⥤ A` when `C` is equipped with a shift
by a monoid `M` involves a sequence of functor `sequence n : C ⥤ A` for all `n : M`
which behave like `shiftFunctor C n ⋙ F`. -/
class ShiftSequence where
  /-- a sequence of functors -/
  sequence : M → C ⥤ A
  /-- `sequence 0` identifies to the given functor -/
  isoZero : sequence 0 ≅ F
  /-- compatibility isomorphism with the shift -/
  shiftIso (n a a' : M) (ha' : n + a = a') : shiftFunctor C n ⋙ sequence a ≅ sequence a'
  shiftIso_zero (a : M) : shiftIso 0 a a (zero_add a) =
    isoWhiskerRight (shiftFunctorZero C M) _ ≪≫ leftUnitor _
  shiftIso_add : ∀ (n m a a' a'' : M) (ha' : n + a = a') (ha'' : m + a' = a''),
                               /-
                                 C : Type u_1
                                 A : Type u_2
                                 inst✝⁵ : CategoryTheory.Category.{?u.229, u_1} C
                                 inst✝⁴ : CategoryTheory.Category.{?u.233, u_2} A
                                 F : CategoryTheory.Functor C A
                                 M : Type u_3
                                 inst✝³ : AddMonoid M
                                 inst✝² : CategoryTheory.HasShift C M
                                 G : Type u_4
                                 inst✝¹ : AddGroup G
                                 inst✝ : CategoryTheory.HasShift C G
                                 sequence : M → CategoryTheory.Functor C A
                                 isoZero : CategoryTheory.Iso (sequence 0) F
                                 shiftIso : (n a a' : M) → Eq (HAdd.hAdd n a) a' → CategoryTheory.Iso ((Categor …
                                 shiftIso_zero : ∀ (a : M), Eq (shiftIso 0 a a ⋯) ((CategoryTheory.isoWhiskerRi …
                                 n m a a' a'' : M
                                 ha' : Eq (HAdd.hAdd n a) a'
                                 ha'' : Eq (HAdd.hAdd m a') a''
                                 ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                               -/
    shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha'']) =
                               /-
                                 🎉 no goals
                               -/
      isoWhiskerRight (shiftFunctorAdd C m n) _ ≪≫ Functor.associator _ _ _ ≪≫
        isoWhiskerLeft _ (shiftIso n a a' ha') ≪≫ shiftIso m a' a'' ha''


/-- The tautological shift sequence on a functor. -/
noncomputable def ShiftSequence.tautological : ShiftSequence F M where
  sequence n := shiftFunctor C n ⋙ F
  isoZero := isoWhiskerRight (shiftFunctorZero C M) F ≪≫ F.rightUnitor
  shiftIso n a a' ha' := (Functor.associator _ _ _).symm ≪≫
    isoWhiskerRight (shiftFunctorAdd' C n a a' ha').symm _
  shiftIso_zero a := by
    /-
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      a : M
      ⊢ Eq ((fun n a a' ha' => ((CategoryTheory.shiftFunctor C n).associator (Catego …
    -/
    dsimp
    /-
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      a : M
      ⊢ Eq (((CategoryTheory.shiftFunctor C 0).associator (CategoryTheory.shiftFunct …
    -/
    rw [shiftFunctorAdd'_zero_add]
    /-
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      a : M
      ⊢ Eq (((CategoryTheory.shiftFunctor C 0).associator (CategoryTheory.shiftFunct …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  shiftIso_add n m a a' a'' ha' ha'' := by
    /-
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq ((fun n a a' ha' => ((CategoryTheory.shiftFunctor C n).associator (Catego …
    -/
    ext X
    /-
      case w.w.h
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      X : C
      ⊢ Eq (((fun n a a' ha' => ((CategoryTheory.shiftFunctor C n).associator (Categ …
    -/
    dsimp
    /-
      case w.w.h
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (F. …
    -/
    simp only [id_comp, ← Functor.map_comp]
    /-
      case w.w.h
      C : Type u_1
      A : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.6237, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.6241, u_2} A
      F : CategoryTheory.Functor C A
      M : Type u_3
      inst✝³ : AddMonoid M
      inst✝² : CategoryTheory.HasShift C M
      G : Type u_4
      inst✝¹ : AddGroup G
      inst✝ : CategoryTheory.HasShift C G
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      X : C
      ⊢ Eq (F.map ((CategoryTheory.shiftFunctorAdd' C (HAdd.hAdd m n) a a'' ⋯).inv.a …
    -/
    congr
    simpa only [← cancel_epi ((shiftFunctor C a).map ((shiftFunctorAdd C m n).hom.app X)),
      shiftFunctorAdd'_eq_shiftFunctorAdd, ← Functor.map_comp_assoc, Iso.hom_inv_id_app,
      Functor.map_id, id_comp] using shiftFunctorAdd'_assoc_inv_app m n a (m+n) a' a'' rfl ha'
        (by rw [← ha'', ← ha', add_assoc]) X


/-- The shifted functors given by the shift sequence. -/
def shift (n : M) : C ⥤ A := ShiftSequence.sequence F n


/-- Compatibility isomorphism `shiftFunctor C n ⋙ F.shift a ≅ F.shift a'` when `n + a = a'`. -/
def shiftIso (n a a' : M) (ha' : n + a = a') :
    shiftFunctor C n ⋙ F.shift a ≅ F.shift a' :=
  ShiftSequence.shiftIso n a a' ha'


@[reassoc (attr := simp 1100)]
lemma shiftIso_hom_naturality {X Y : C} (n a a' : M) (ha' : n + a = a') (f : X ⟶ Y) :
    (shift F a).map (f⟦n⟧') ≫ (shiftIso F n a a' ha').hom.app Y =
      (shiftIso F n a a' ha').hom.app X ≫ (shift F a').map f :=
  (F.shiftIso n a a' ha').hom.naturality f


@[reassoc (attr := simp 1100)]
lemma shiftIso_inv_naturality {X Y : C} (n a a' : M) (ha' : n + a = a') (f : X ⟶ Y) :
    (shift F a').map f ≫ (shiftIso F n a a' ha').inv.app Y =
      (shiftIso F n a a' ha').inv.app X ≫ (shift F a).map (f⟦n⟧') :=
  (F.shiftIso n a a' ha').inv.naturality f


/-- The canonical isomorphism `F.shift 0 ≅ F`. -/
def isoShiftZero : F.shift (0 : M) ≅ F := ShiftSequence.isoZero


/-- The canonical isomorphism `shiftFunctor C n ⋙ F ≅ F.shift n`. -/
def isoShift (n : M) : shiftFunctor C n ⋙ F ≅ F.shift n :=
  isoWhiskerLeft _ (F.isoShiftZero M).symm ≪≫ F.shiftIso _ _ _ (add_zero n)


@[reassoc]
lemma isoShift_hom_naturality (n : M) {X Y : C} (f : X ⟶ Y) :
    F.map (f⟦n⟧') ≫ (F.isoShift n).hom.app Y =
      (F.isoShift n).hom.app X ≫ (F.shift n).map f :=
  (F.isoShift n).hom.naturality f


@[reassoc]
lemma isoShift_inv_naturality (n : M) {X Y : C} (f : X ⟶ Y) :
    (F.shift n).map f ≫ (F.isoShift n).inv.app Y =
      (F.isoShift n).inv.app X ≫ F.map (f⟦n⟧') :=
  (F.isoShift n).inv.naturality f


lemma shiftIso_zero (a : M) :
    F.shiftIso 0 a a (zero_add a) =
      isoWhiskerRight (shiftFunctorZero C M) _ ≪≫ leftUnitor _ :=
  ShiftSequence.shiftIso_zero a


@[simp]
lemma shiftIso_zero_hom_app (a : M) (X : C) :
    (F.shiftIso 0 a a (zero_add a)).hom.app X =
      (shift F a).map ((shiftFunctorZero C M).hom.app X) := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    a : M
    X : C
    ⊢ Eq ((F.shiftIso 0 a a ⋯).hom.app X) ((F.shift a).map ((CategoryTheory.shiftF …
  -/
  simp [F.shiftIso_zero a]
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftIso_zero_inv_app (a : M) (X : C) :
    (F.shiftIso 0 a a (zero_add a)).inv.app X =
      (shift F a).map ((shiftFunctorZero C M).inv.app X) := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    a : M
    X : C
    ⊢ Eq ((F.shiftIso 0 a a ⋯).inv.app X) ((F.shift a).map ((CategoryTheory.shiftF …
  -/
  simp [F.shiftIso_zero a]
  /-
    🎉 no goals
  -/


lemma shiftIso_add (n m a a' a'' : M) (ha' : n + a = a') (ha'' : m + a' = a'') :
                                 /-
                                   C : Type u_1
                                   A : Type u_2
                                   inst✝⁶ : CategoryTheory.Category.{?u.40636, u_1} C
                                   inst✝⁵ : CategoryTheory.Category.{?u.40640, u_2} A
                                   F : CategoryTheory.Functor C A
                                   M : Type u_3
                                   inst✝⁴ : AddMonoid M
                                   inst✝³ : CategoryTheory.HasShift C M
                                   G : Type u_4
                                   inst✝² : AddGroup G
                                   inst✝¹ : CategoryTheory.HasShift C G
                                   inst✝ : F.ShiftSequence M
                                   n m a a' a'' : M
                                   ha' : Eq (HAdd.hAdd n a) a'
                                   ha'' : Eq (HAdd.hAdd m a') a''
                                   ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                                 -/
    F.shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha'']) =
                                 /-
                                   🎉 no goals
                                 -/
      isoWhiskerRight (shiftFunctorAdd C m n) _ ≪≫ Functor.associator _ _ _ ≪≫
        isoWhiskerLeft _ (F.shiftIso n a a' ha') ≪≫ F.shiftIso m a' a'' ha'' :=
  ShiftSequence.shiftIso_add _ _ _ _ _ _ _


lemma shiftIso_add_hom_app (n m a a' a'' : M) (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                                  /-
                                    C : Type u_1
                                    A : Type u_2
                                    inst✝⁶ : CategoryTheory.Category.{?u.43981, u_1} C
                                    inst✝⁵ : CategoryTheory.Category.{?u.43985, u_2} A
                                    F : CategoryTheory.Functor C A
                                    M : Type u_3
                                    inst✝⁴ : AddMonoid M
                                    inst✝³ : CategoryTheory.HasShift C M
                                    G : Type u_4
                                    inst✝² : AddGroup G
                                    inst✝¹ : CategoryTheory.HasShift C G
                                    inst✝ : F.ShiftSequence M
                                    n m a a' a'' : M
                                    ha' : Eq (HAdd.hAdd n a) a'
                                    ha'' : Eq (HAdd.hAdd m a') a''
                                    X : C
                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                                  -/
    (F.shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha''])).hom.app X =
                                  /-
                                    🎉 no goals
                                  -/
      (shift F a).map ((shiftFunctorAdd C m n).hom.app X) ≫
        (shiftIso F n a a' ha').hom.app ((shiftFunctor C m).obj X) ≫
          (shiftIso F m a' a'' ha'').hom.app X := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    n m a a' a'' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso (HAdd.hAdd m n) a a'' ⋯).hom.app X) (CategoryTheory.Category …
  -/
  simp [F.shiftIso_add n m a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


lemma shiftIso_add_inv_app (n m a a' a'' : M) (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                                  /-
                                    C : Type u_1
                                    A : Type u_2
                                    inst✝⁶ : CategoryTheory.Category.{?u.48706, u_1} C
                                    inst✝⁵ : CategoryTheory.Category.{?u.48710, u_2} A
                                    F : CategoryTheory.Functor C A
                                    M : Type u_3
                                    inst✝⁴ : AddMonoid M
                                    inst✝³ : CategoryTheory.HasShift C M
                                    G : Type u_4
                                    inst✝² : AddGroup G
                                    inst✝¹ : CategoryTheory.HasShift C G
                                    inst✝ : F.ShiftSequence M
                                    n m a a' a'' : M
                                    ha' : Eq (HAdd.hAdd n a) a'
                                    ha'' : Eq (HAdd.hAdd m a') a''
                                    X : C
                                    ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                                  -/
    (F.shiftIso (m + n) a a'' (by rw [add_assoc, ha', ha''])).inv.app X =
                                  /-
                                    🎉 no goals
                                  -/
      (shiftIso F m a' a'' ha'').inv.app X ≫
        (shiftIso F n a a' ha').inv.app ((shiftFunctor C m).obj X) ≫
          (shift F a).map ((shiftFunctorAdd C m n).inv.app X) := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    n m a a' a'' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso (HAdd.hAdd m n) a a'' ⋯).inv.app X) (CategoryTheory.Category …
  -/
  simp [F.shiftIso_add n m a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


lemma shiftIso_add' (n m mn : M) (hnm : m + n = mn) (a a' a'' : M)
    (ha' : n + a = a') (ha'' : m + a' = a'') :
                            /-
                              C : Type u_1
                              A : Type u_2
                              inst✝⁶ : CategoryTheory.Category.{?u.52530, u_1} C
                              inst✝⁵ : CategoryTheory.Category.{?u.52534, u_2} A
                              F : CategoryTheory.Functor C A
                              M : Type u_3
                              inst✝⁴ : AddMonoid M
                              inst✝³ : CategoryTheory.HasShift C M
                              G : Type u_4
                              inst✝² : AddGroup G
                              inst✝¹ : CategoryTheory.HasShift C G
                              inst✝ : F.ShiftSequence M
                              n m mn : M
                              hnm : Eq (HAdd.hAdd m n) mn
                              a a' a'' : M
                              ha' : Eq (HAdd.hAdd n a) a'
                              ha'' : Eq (HAdd.hAdd m a') a''
                              ⊢ Eq (HAdd.hAdd mn a) a''
                            -/
    F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc]) =
                            /-
                              🎉 no goals
                            -/
      isoWhiskerRight (shiftFunctorAdd' C m n _ hnm) _ ≪≫ Functor.associator _ _ _ ≪≫
        isoWhiskerLeft _ (F.shiftIso n a a' ha') ≪≫ F.shiftIso m a' a'' ha'' := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    n m mn : M
    hnm : Eq (HAdd.hAdd m n) mn
    a a' a'' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    ⊢ Eq (F.shiftIso mn a a'' ⋯) ((CategoryTheory.isoWhiskerRight (CategoryTheory. …
  -/
  subst hnm
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    n m a a' a'' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    ⊢ Eq (F.shiftIso (HAdd.hAdd m n) a a'' ⋯) ((CategoryTheory.isoWhiskerRight (Ca …
  -/
  rw [shiftFunctorAdd'_eq_shiftFunctorAdd, shiftIso_add]
  /-
    🎉 no goals
  -/


lemma shiftIso_add'_hom_app (n m mn : M) (hnm : m + n = mn) (a a' a'' : M)
    (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                             /-
                               C : Type u_1
                               A : Type u_2
                               inst✝⁶ : CategoryTheory.Category.{?u.56467, u_1} C
                               inst✝⁵ : CategoryTheory.Category.{?u.56471, u_2} A
                               F : CategoryTheory.Functor C A
                               M : Type u_3
                               inst✝⁴ : AddMonoid M
                               inst✝³ : CategoryTheory.HasShift C M
                               G : Type u_4
                               inst✝² : AddGroup G
                               inst✝¹ : CategoryTheory.HasShift C G
                               inst✝ : F.ShiftSequence M
                               n m mn : M
                               hnm : Eq (HAdd.hAdd m n) mn
                               a a' a'' : M
                               ha' : Eq (HAdd.hAdd n a) a'
                               ha'' : Eq (HAdd.hAdd m a') a''
                               X : C
                               ⊢ Eq (HAdd.hAdd mn a) a''
                             -/
    (F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc])).hom.app X =
                             /-
                               🎉 no goals
                             -/
      (shift F a).map ((shiftFunctorAdd' C m n mn hnm).hom.app X) ≫
        (shiftIso F n a a' ha').hom.app ((shiftFunctor C m).obj X) ≫
          (shiftIso F m a' a'' ha'').hom.app X := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    n m mn : M
    hnm : Eq (HAdd.hAdd m n) mn
    a a' a'' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso mn a a'' ⋯).hom.app X) (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [F.shiftIso_add' n m mn hnm a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


lemma shiftIso_add'_inv_app (n m mn : M) (hnm : m + n = mn) (a a' a'' : M)
    (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
                             /-
                               C : Type u_1
                               A : Type u_2
                               inst✝⁶ : CategoryTheory.Category.{?u.61227, u_1} C
                               inst✝⁵ : CategoryTheory.Category.{?u.61231, u_2} A
                               F : CategoryTheory.Functor C A
                               M : Type u_3
                               inst✝⁴ : AddMonoid M
                               inst✝³ : CategoryTheory.HasShift C M
                               G : Type u_4
                               inst✝² : AddGroup G
                               inst✝¹ : CategoryTheory.HasShift C G
                               inst✝ : F.ShiftSequence M
                               n m mn : M
                               hnm : Eq (HAdd.hAdd m n) mn
                               a a' a'' : M
                               ha' : Eq (HAdd.hAdd n a) a'
                               ha'' : Eq (HAdd.hAdd m a') a''
                               X : C
                               ⊢ Eq (HAdd.hAdd mn a) a''
                             -/
    (F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc])).inv.app X =
                             /-
                               🎉 no goals
                             -/
      (shiftIso F m a' a'' ha'').inv.app X ≫
        (shiftIso F n a a' ha').inv.app ((shiftFunctor C m).obj X) ≫
        (shift F a).map ((shiftFunctorAdd' C m n mn hnm).inv.app X) := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    n m mn : M
    hnm : Eq (HAdd.hAdd m n) mn
    a a' a'' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    X : C
    ⊢ Eq ((F.shiftIso mn a a'' ⋯).inv.app X) (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [F.shiftIso_add' n m mn hnm a a' a'' ha' ha'']
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shiftIso_hom_app_comp (n m mn : M) (hnm : m + n = mn)
    (a a' a'' : M) (ha' : n + a = a') (ha'' : m + a' = a'') (X : C) :
    (shiftIso F n a a' ha').hom.app ((shiftFunctor C m).obj X) ≫
      (shiftIso F m a' a'' ha'').hom.app X =
        (shift F a).map ((shiftFunctorAdd' C m n mn hnm).inv.app X) ≫
                                   /-
                                     C : Type u_1
                                     A : Type u_2
                                     inst✝⁶ : CategoryTheory.Category.{?u.65086, u_1} C
                                     inst✝⁵ : CategoryTheory.Category.{?u.65090, u_2} A
                                     F : CategoryTheory.Functor C A
                                     M : Type u_3
                                     inst✝⁴ : AddMonoid M
                                     inst✝³ : CategoryTheory.HasShift C M
                                     G : Type u_4
                                     inst✝² : AddGroup G
                                     inst✝¹ : CategoryTheory.HasShift C G
                                     inst✝ : F.ShiftSequence M
                                     n m mn : M
                                     hnm : Eq (HAdd.hAdd m n) mn
                                     a a' a'' : M
                                     ha' : Eq (HAdd.hAdd n a) a'
                                     ha'' : Eq (HAdd.hAdd m a') a''
                                     X : C
                                     ⊢ Eq (HAdd.hAdd mn a) a''
                                   -/
          (F.shiftIso mn a a'' (by rw [← hnm, ← ha'', ← ha', add_assoc])).hom.app X := by
                                   /-
                                     🎉 no goals
                                   -/
  rw [F.shiftIso_add'_hom_app n m mn hnm a a' a'' ha' ha'', ← Functor.map_comp_assoc,
    Iso.inv_hom_id_app, Functor.map_id, id_comp]


/-- The morphism `(F.shift a).obj X ⟶ (F.shift a').obj Y` induced by a morphism
`f : X ⟶ Y⟦n⟧` when `n + a = a'`. -/
def shiftMap {X Y : C} {n : M} (f : X ⟶ Y⟦n⟧) (a a' : M) (ha' : n + a = a') :
    (F.shift a).obj X ⟶ (F.shift a').obj Y :=
  (F.shift a).map f ≫ (F.shiftIso _ _ _ ha').hom.app Y


@[reassoc]
lemma shiftMap_comp {X Y Z : C} {n : M} (f : X ⟶ Y⟦n⟧) (g : Y ⟶ Z) (a a' : M) (ha' : n + a = a') :
    F.shiftMap (f ≫ g⟦n⟧') a a' ha' = F.shiftMap f a a' ha' ≫ (F.shift a').map g := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    X Y Z : C
    n : M
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    g : Quiver.Hom Y Z
    a a' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (F.shiftMap (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.shiftF …
  -/
  simp [shiftMap]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shiftMap_comp' {X Y Z : C} {n : M} (f : X ⟶ Y) (g : Y ⟶ Z⟦n⟧) (a a' : M) (ha' : n + a = a') :
    F.shiftMap (f ≫ g) a a' ha' = (F.shift a).map f ≫ F.shiftMap g a a' ha' := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    inst✝ : F.ShiftSequence M
    X Y Z : C
    n : M
    f : Quiver.Hom X Y
    g : Quiver.Hom Y ((CategoryTheory.shiftFunctor C n).obj Z)
    a a' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (F.shiftMap (CategoryTheory.CategoryStruct.comp f g) a a' ha') (CategoryT …
  -/
  simp [shiftMap]
  /-
    🎉 no goals
  -/


/--
When `f : X ⟶ Y⟦m⟧`, `m + n = mn`, `n + a = a'` and `ha'' : m + a' = a''`, this lemma
relates the two morphisms `F.shiftMap f a' a'' ha''` and `(F.shift a).map (f⟦n⟧')`. Indeed,
via canonical isomorphisms, they both identity to morphisms
`(F.shift a').obj X ⟶ (F.shift a'').obj Y`.
-/
lemma shiftIso_hom_app_comp_shiftMap {X Y : C} {m : M} (f : X ⟶ Y⟦m⟧) (n mn : M) (hnm : m + n = mn)
    (a a' a'' : M) (ha' : n + a = a') (ha'' : m + a' = a'') :
    (F.shiftIso n a a' ha').hom.app X ≫ F.shiftMap f a' a'' ha'' =
      (F.shift a).map (f⟦n⟧') ≫ (F.shift a).map ((shiftFunctorAdd' C m n mn hnm).inv.app Y) ≫
                                 /-
                                   C : Type u_1
                                   A : Type u_2
                                   inst✝⁶ : CategoryTheory.Category.{?u.80997, u_1} C
                                   inst✝⁵ : CategoryTheory.Category.{?u.81001, u_2} A
                                   F : CategoryTheory.Functor C A
                                   M : Type u_3
                                   inst✝⁴ : AddMonoid M
                                   inst✝³ : CategoryTheory.HasShift C M
                                   G : Type u_4
                                   inst✝² : AddGroup G
                                   inst✝¹ : CategoryTheory.HasShift C G
                                   inst✝ : F.ShiftSequence M
                                   X Y : C
                                   m : M
                                   f : Quiver.Hom X ((CategoryTheory.shiftFunctor C m).obj Y)
                                   n mn : M
                                   hnm : Eq (HAdd.hAdd m n) mn
                                   a a' a'' : M
                                   ha' : Eq (HAdd.hAdd n a) a'
                                   ha'' : Eq (HAdd.hAdd m a') a''
                                   ⊢ Eq (HAdd.hAdd mn a) a''
                                 -/
        (F.shiftIso mn a a'' (by rw [← ha'', ← ha', ← hnm, add_assoc])).hom.app Y := by
                                 /-
                                   🎉 no goals
                                 -/
  simp only [F.shiftIso_add'_hom_app n m mn hnm a a' a'' ha' ha'' Y,
    ← Functor.map_comp_assoc, Iso.inv_hom_id_app, Functor.map_id,
    id_comp, comp_obj, shiftIso_hom_naturality_assoc, shiftMap]


/--
If `f : X ⟶ Y⟦m⟧`, `n + m = 0` and `ha' : m + a = a'`, this lemma relates the two
morphisms `F.shiftMap f a a' ha'` and `(F.shift a').map (f⟦n⟧')`. Indeed,
via canonical isomorphisms, they both identify to morphisms
`(F.shift a).obj X ⟶ (F.shift a').obj Y`.
-/
lemma shiftIso_hom_app_comp_shiftMap_of_add_eq_zero [F.ShiftSequence G]
    {X Y : C} {m : G} (f : X ⟶ Y⟦m⟧)
    (n : G) (hnm : n + m = 0) (a a' : G) (ha' : m + a = a') :
                           /-
                             C : Type u_1
                             A : Type u_2
                             inst✝⁷ : CategoryTheory.Category.{?u.85124, u_1} C
                             inst✝⁶ : CategoryTheory.Category.{?u.85128, u_2} A
                             F : CategoryTheory.Functor C A
                             M : Type u_3
                             inst✝⁵ : AddMonoid M
                             inst✝⁴ : CategoryTheory.HasShift C M
                             G : Type u_4
                             inst✝³ : AddGroup G
                             inst✝² : CategoryTheory.HasShift C G
                             inst✝¹ : F.ShiftSequence M
                             inst✝ : F.ShiftSequence G
                             X Y : C
                             m : G
                             f : Quiver.Hom X ((CategoryTheory.shiftFunctor C m).obj Y)
                             n : G
                             hnm : Eq (HAdd.hAdd n m) 0
                             a a' : G
                             ha' : Eq (HAdd.hAdd m a) a'
                             ⊢ Eq (HAdd.hAdd n a') a
                           -/
    (F.shiftIso n a' a (by rw [← ha', ← add_assoc, hnm, zero_add])).hom.app X ≫
                           /-
                             🎉 no goals
                           -/
      F.shiftMap f a a' ha' =
    (F.shift a').map (f⟦n⟧' ≫ (shiftFunctorCompIsoId C m n
          /-
            C : Type u_1
            A : Type u_2
            inst✝⁷ : CategoryTheory.Category.{?u.85124, u_1} C
            inst✝⁶ : CategoryTheory.Category.{?u.85128, u_2} A
            F : CategoryTheory.Functor C A
            M : Type u_3
            inst✝⁵ : AddMonoid M
            inst✝⁴ : CategoryTheory.HasShift C M
            G : Type u_4
            inst✝³ : AddGroup G
            inst✝² : CategoryTheory.HasShift C G
            inst✝¹ : F.ShiftSequence M
            inst✝ : F.ShiftSequence G
            X Y : C
            m : G
            f : Quiver.Hom X ((CategoryTheory.shiftFunctor C m).obj Y)
            n : G
            hnm : Eq (HAdd.hAdd n m) 0
            a a' : G
            ha' : Eq (HAdd.hAdd m a) a'
            ⊢ Eq (HAdd.hAdd m n) 0
          -/
      (by rw [← add_left_inj m, add_assoc, hnm, zero_add, add_zero])).hom.app Y) := by
          /-
            🎉 no goals
          -/
  have hnm' : m + n = 0 := by
    rw [← add_left_inj m, add_assoc, hnm, zero_add, add_zero]
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_2} A
    F : CategoryTheory.Functor C A
    G : Type u_4
    inst✝² : AddGroup G
    inst✝¹ : CategoryTheory.HasShift C G
    inst✝ : F.ShiftSequence G
    X Y : C
    m : G
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C m).obj Y)
    n : G
    hnm : Eq (HAdd.hAdd n m) 0
    a a' : G
    ha' : Eq (HAdd.hAdd m a) a'
    hnm' : Eq (HAdd.hAdd m n) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shiftIso n a' a ⋯).hom.app X) (F. …
  -/
  dsimp
  simp [F.shiftIso_hom_app_comp_shiftMap f n 0 hnm' a' a, shiftIso_zero_hom_app,
    shiftFunctorCompIsoId]


instance (n : M) : (F.shift n).PreservesZeroMorphisms :=
  preservesZeroMorphisms_of_iso (F.isoShift n)


@[simp]
lemma shiftMap_zero (X Y : C) (n a a' : M) (ha' : n + a = a') :
    F.shiftMap (0 : X ⟶ Y⟦n⟧) a a' ha' = 0 := by
  /-
    C : Type u_1
    A : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_6, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_5, u_2} A
    F : CategoryTheory.Functor C A
    M : Type u_3
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : F.ShiftSequence M
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms A
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : ∀ (n : M), (CategoryTheory.shiftFunctor C n).PreservesZeroMorphisms
    X Y : C
    n a a' : M
    ha' : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (F.shiftMap 0 a a' ha') 0
  -/
  simp [shiftMap]
  /-
    🎉 no goals
  -/


instance (n : M) : (F.shift n).Additive := additive_of_iso (F.isoShift n)


