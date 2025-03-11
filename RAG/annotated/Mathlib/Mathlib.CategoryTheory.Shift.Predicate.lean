/-- Given a predicate `P : C → Prop` on objects of a category equipped with a shift by `A`,
this is the predicate which is satisfied by `X` if `P (X⟦a⟧)`. -/
def PredicateShift (a : A) : C → Prop := fun X => P (X⟦a⟧)


lemma predicateShift_iff (a : A) (X : C) : PredicateShift P a X ↔ P (X⟦a⟧) := Iff.rfl


instance predicateShift_closedUnderIsomorphisms (a : A) [ClosedUnderIsomorphisms P] :
    ClosedUnderIsomorphisms (PredicateShift P a) where
  of_iso e hX := mem_of_iso P ((shiftFunctor C a).mapIso e) hX


@[simp]
lemma predicateShift_zero [ClosedUnderIsomorphisms P] : PredicateShift P (0 : A) = P := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
    ⊢ Eq (CategoryTheory.PredicateShift P 0) P
  -/
  ext X
  /-
    case h.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
    X : C
    ⊢ Iff (CategoryTheory.PredicateShift P 0 X) (P X)
  -/
  exact mem_iff_of_iso P ((shiftFunctorZero C A).app X)
  /-
    🎉 no goals
  -/


lemma predicateShift_predicateShift (a b c : A) (h : a + b = c) [ClosedUnderIsomorphisms P] :
    PredicateShift (PredicateShift P b) a = PredicateShift P c := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    a b c : A
    h : Eq (HAdd.hAdd a b) c
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
    ⊢ Eq (CategoryTheory.PredicateShift (CategoryTheory.PredicateShift P b) a) (Ca …
  -/
  ext X
  /-
    case h.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    P : C → Prop
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    a b c : A
    h : Eq (HAdd.hAdd a b) c
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
    X : C
    ⊢ Iff (CategoryTheory.PredicateShift (CategoryTheory.PredicateShift P b) a X)  …
  -/
  exact mem_iff_of_iso _ ((shiftFunctorAdd' C a b c h).symm.app X)
  /-
    🎉 no goals
  -/


