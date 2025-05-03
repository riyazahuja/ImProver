/-- The collection of all single functors `C ⥤ CochainComplex C ℤ` along with
their compatibilites with shifts. (This definition has purposely no `simps`
attribute, as the generated lemmas would not be very useful.) -/
noncomputable def singleFunctors : SingleFunctors C (CochainComplex C ℤ) ℤ where
  functor n := single _ _ n
  shiftIso n a a' ha' := NatIso.ofComponents
              /-
                C : Type u
                inst✝² : CategoryTheory.Category.{v, u} C
                inst✝¹ : CategoryTheory.Preadditive C
                inst✝ : CategoryTheory.Limits.HasZeroObject C
                n a a' : Int
                ha' : Eq (HAdd.hAdd n a) a'
                X : C
                ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
              -/
    (fun X => Hom.isoOfComponents
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          n a a' : Int
          ha' : Eq (HAdd.hAdd n a) a'
          X : C
          i : Int
          ⊢ Eq (((((fun n => HomologicalComplex.single C (ComplexShape.up Int) n) a').co …
        -/
              /-
                🎉 no goals
              -/
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          n a : Int
          X : C
          i : Int
          ha' : Eq (HAdd.hAdd n a) (HAdd.hAdd a n)
          ⊢ Eq (((((fun n => HomologicalComplex.single C (ComplexShape.up Int) n) (HAdd. …
        -/
      (fun i => eqToIso (by
          /-
            case pos
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            n a : Int
            X : C
            i : Int
            ha' : Eq (HAdd.hAdd n a) (HAdd.hAdd a n)
            h : Eq i a
            ⊢ Eq (((((fun n => HomologicalComplex.single C (ComplexShape.up Int) n) (HAdd. …
          -/
        obtain rfl : a' = a + n := by omega
          /-
            case pos
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            n : Int
            X : C
            i : Int
            ha' : Eq (HAdd.hAdd n i) (HAdd.hAdd i n)
            ⊢ Eq (((((fun n => HomologicalComplex.single C (ComplexShape.up Int) n) (HAdd. …
          -/
        by_cases h : i = a
          /-
            🎉 no goals
          -/
          /-
            case neg
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            n a : Int
            X : C
            i : Int
            ha' : Eq (HAdd.hAdd n a) (HAdd.hAdd a n)
            h : Not (Eq i a)
            ⊢ Eq (((((fun n => HomologicalComplex.single C (ComplexShape.up Int) n) (HAdd. …
          -/
        · subst h
          /-
            case neg
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            n a : Int
            X : C
            i : Int
            ha' : Eq (HAdd.hAdd n a) (HAdd.hAdd a n)
            h : Not (Eq i a)
            ⊢ Eq (ite (Eq (HAdd.hAdd i n) (HAdd.hAdd a n)) X 0) (ite (Eq i a) X 0)
          -/
          simp only [Functor.comp_obj, shiftFunctor_obj_X', single_obj_X_self]
          /-
            🎉 no goals
          -/
        · dsimp [single]
          rw [if_neg h, if_neg (fun h' => h (by omega))])))
    (fun {X Y} f => by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        n a a' : Int
        ha' : Eq (HAdd.hAdd n a) a'
        X Y : C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun n => HomologicalComplex.singl …
      -/
      obtain rfl : a' = a + n := by omega
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        n a : Int
        X Y : C
        f : Quiver.Hom X Y
        ha' : Eq (HAdd.hAdd n a) (HAdd.hAdd a n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun n => HomologicalComplex.singl …
      -/
      ext
      /-
        case hfg
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        n a : Int
        X Y : C
        f : Quiver.Hom X Y
        ha' : Eq (HAdd.hAdd n a) (HAdd.hAdd a n)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((((fun n => HomologicalComplex.sing …
      -/
      simp [single])
      /-
        🎉 no goals
      -/
  shiftIso_zero a := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      a : Int
      ⊢ Eq ((fun n a a' ha' => CategoryTheory.NatIso.ofComponents (fun X => Homologi …
    -/
    ext
    /-
      case w.w.h.hfg
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      a : Int
      x✝ : C
      ⊢ Eq ((((fun n a a' ha' => CategoryTheory.NatIso.ofComponents (fun X => Homolo …
    -/
    dsimp
    simp only [single, shiftFunctorZero_eq, shiftFunctorZero'_hom_app_f,
      XIsoOfEq, eqToIso.hom]
  shiftIso_add n m a a' a'' ha' ha'' := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      n m a a' a'' : Int
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq ((fun n a a' ha' => CategoryTheory.NatIso.ofComponents (fun X => Homologi …
    -/
    ext
    /-
      case w.w.h.hfg
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      n m a a' a'' : Int
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      x✝ : C
      ⊢ Eq ((((fun n a a' ha' => CategoryTheory.NatIso.ofComponents (fun X => Homolo …
    -/
    dsimp
    simp only [shiftFunctorAdd_eq, shiftFunctorAdd'_hom_app_f, XIsoOfEq,
      eqToIso.hom, eqToHom_trans, id_comp]


instance (n : ℤ) : ((singleFunctors C).functor n).Additive := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    n : Int
    ⊢ ((CochainComplex.singleFunctors C).functor n).Additive
  -/
  dsimp only [singleFunctors]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    n : Int
    ⊢ (HomologicalComplex.single C (ComplexShape.up Int) n).Additive
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The single functor `C ⥤ CochainComplex C ℤ` which sends `X` to the complex
consisting of `X` in degree `n : ℤ` and zero otherwise.
(This is definitionally equal to `HomologicalComplex.single C (up ℤ) n`,
but `singleFunctor C n` is the preferred term when interactions with shifts are relevant.) -/
noncomputable abbrev singleFunctor (n : ℤ) := (singleFunctors C).functor n


/-- The collection of all single functors `C ⥤ HomotopyCategory C (ComplexShape.up ℤ))`
for `n : ℤ` along with their compatibilites with shifts. -/
noncomputable def singleFunctors : SingleFunctors C (HomotopyCategory C (ComplexShape.up ℤ)) ℤ :=
  (CochainComplex.singleFunctors C).postcomp (HomotopyCategory.quotient _ _)


/-- The single functor `C ⥤ HomotopyCategory C (ComplexShape.up ℤ)`
which sends `X` to the complex consisting of `X` in degree `n : ℤ` and zero otherwise. -/
noncomputable abbrev singleFunctor (n : ℤ) :
    C ⥤ HomotopyCategory C (ComplexShape.up ℤ) :=
  (singleFunctors C).functor n


instance (n : ℤ) : (singleFunctor C n).Additive := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    n : Int
    ⊢ (HomotopyCategory.singleFunctor C n).Additive
  -/
  dsimp only [singleFunctor, singleFunctors, SingleFunctors.postcomp]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    n : Int
    ⊢ (((CochainComplex.singleFunctors C).functor n).comp (HomotopyCategory.quotie …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The isomorphism given by the very definition of `singleFunctors C`. -/
noncomputable def singleFunctorsPostcompQuotientIso :
    singleFunctors C ≅
      (CochainComplex.singleFunctors C).postcomp (HomotopyCategory.quotient _ _) :=
  Iso.refl _


/-- `HomotopyCategory.singleFunctor C n` is induced by `CochainComplex.singleFunctor C n`. -/
noncomputable def singleFunctorPostcompQuotientIso (n : ℤ) :
    singleFunctor C n ≅ CochainComplex.singleFunctor C n ⋙ quotient _ _ :=
  (SingleFunctors.evaluation _ _ n).mapIso (singleFunctorsPostcompQuotientIso C)


