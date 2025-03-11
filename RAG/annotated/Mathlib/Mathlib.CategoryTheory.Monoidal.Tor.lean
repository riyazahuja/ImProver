/-- We define `Tor C n : C ⥤ C ⥤ C` by left-deriving in the second factor of `(X, Y) ↦ X ⊗ Y`. -/
@[simps]
def Tor (n : ℕ) : C ⥤ C ⥤ C where
  obj X := Functor.leftDerived ((tensoringLeft C).obj X) n
  map f := NatTrans.leftDerived ((tensoringLeft C).map f) n


/-- An alternative definition of `Tor`, where we left-derive in the first factor instead. -/
@[simps! obj_obj]
def Tor' (n : ℕ) : C ⥤ C ⥤ C :=
  Functor.flip
    { obj := fun X => Functor.leftDerived ((tensoringRight C).obj X) n
      map := fun f => NatTrans.leftDerived ((tensoringRight C).map f) n }

-- Porting note: the `checkType` linter complains about the automatically generated
-- lemma `Tor'_map_app`, but not about this one

@[simp]
lemma Tor'_map_app' (n : ℕ) {X Y : C} (f : X ⟶ Y) (Z : C) :
    ((Tor' C n).map f).app Z = (Functor.leftDerived ((tensoringRight C).obj Z) n).map f := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.MonoidalPreadditive C
    inst✝ : CategoryTheory.HasProjectiveResolutions C
    n : Nat
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    ⊢ Eq (((CategoryTheory.Tor' C n).map f).app Z) ((((CategoryTheory.MonoidalCate …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: this specific lemma was added because otherwise the internals of
-- `NatTrans.leftDerived` leaks into the RHS (it was already so in mathlib)

@[simp]
lemma Tor'_obj_map (n : ℕ) {X Y : C} (Z : C) (f : X ⟶ Y) :
    ((Tor' C n).obj Z).map f = (NatTrans.leftDerived ((tensoringRight C).map f) n).app Z := rfl


/-- The higher `Tor` groups for `X` and `Y` are zero if `Y` is projective. -/
lemma isZero_Tor_succ_of_projective (X Y : C) [Projective Y] (n : ℕ) :
    IsZero (((Tor C (n + 1)).obj X).obj Y) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.HasProjectiveResolutions C
    X Y : C
    inst✝ : CategoryTheory.Projective Y
    n : Nat
    ⊢ CategoryTheory.Limits.IsZero (((CategoryTheory.Tor C (HAdd.hAdd n 1)).obj X) …
  -/
  apply Functor.isZero_leftDerived_obj_projective_succ
  /-
    🎉 no goals
  -/


/-- The higher `Tor'` groups for `X` and `Y` are zero if `X` is projective. -/
lemma isZero_Tor'_succ_of_projective (X Y : C) [Projective X] (n : ℕ) :
    IsZero (((Tor' C (n + 1)).obj X).obj Y) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.MonoidalCategory C
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.MonoidalPreadditive C
    inst✝¹ : CategoryTheory.HasProjectiveResolutions C
    X Y : C
    inst✝ : CategoryTheory.Projective X
    n : Nat
    ⊢ CategoryTheory.Limits.IsZero (((CategoryTheory.Tor' C (HAdd.hAdd n 1)).obj X …
  -/
  apply Functor.isZero_leftDerived_obj_projective_succ
  /-
    🎉 no goals
  -/


