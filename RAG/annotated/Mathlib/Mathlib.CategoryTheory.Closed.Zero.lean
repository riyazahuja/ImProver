/-- If a cartesian closed category has an initial object which is isomorphic to the terminal object,
then each homset has exactly one element.
-/
def uniqueHomsetOfInitialIsoUnit [HasInitial C] (i : ⊥_ C ≅ 𝟙_ C) (X Y : C) : Unique (X ⟶ Y) :=
  Equiv.unique <|
    calc
      (X ⟶ Y) ≃ (X ⊗ 𝟙_ C ⟶ Y) := Iso.homCongr (rightUnitor _).symm (Iso.refl _)
      _ ≃ (X ⊗ ⊥_ C ⟶ Y) := (Iso.homCongr ((Iso.refl _) ⊗ i.symm) (Iso.refl _))
      _ ≃ (⊥_ C ⟶ Y ^^ X) := (exp.adjunction _).homEquiv _ _


/-- If a cartesian closed category has a zero object, each homset has exactly one element. -/
def uniqueHomsetOfZero [HasZeroObject C] (X Y : C) : Unique (X ⟶ Y) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.CartesianClosed C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    ⊢ Unique (Quiver.Hom X Y)
  -/
  haveI : HasInitial C := HasZeroObject.hasInitial
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.CartesianClosed C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    this : CategoryTheory.Limits.HasInitial C
    ⊢ Unique (Quiver.Hom X Y)
  -/
  apply uniqueHomsetOfInitialIsoUnit _ X Y
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.CartesianClosed C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    this : CategoryTheory.Limits.HasInitial C
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.initial C) CategoryTheory.Monoidal …
  -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  refine ⟨default, (default : 𝟙_ C ⟶ 0) ≫ default, ?_, ?_⟩ <;> simp [eq_iff_true_of_subsingleton]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A cartesian closed category with a zero object is equivalent to the category with one object and
one morphism.
-/
def equivPUnit [HasZeroObject C] : C ≌ Discrete PUnit.{w + 1} where
  functor := Functor.star C
  inverse := Functor.fromPUnit 0
  unitIso := NatIso.ofComponents
      (fun X =>
        { hom := default
          inv := default })
      fun _ => Subsingleton.elim _ _
  counitIso := Functor.punitExt _ _


