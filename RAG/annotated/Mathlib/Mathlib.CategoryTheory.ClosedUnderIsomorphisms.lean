/-- A predicate `C → Prop` on the objects of a category is closed under isomorphisms
if whenever `P X`, then all the objects `Y` that are isomorphic to `X` also satisfy `P Y`. -/
class ClosedUnderIsomorphisms : Prop where
  of_iso {X Y : C} (_ : X ≅ Y) (_ : P X) : P Y


lemma mem_of_iso [ClosedUnderIsomorphisms P] {X Y : C} (e : X ≅ Y) (hX : P X) : P Y :=
  ClosedUnderIsomorphisms.of_iso e hX


lemma mem_iff_of_iso [ClosedUnderIsomorphisms P] {X Y : C} (e : X ≅ Y) : P X ↔ P Y :=
  ⟨mem_of_iso P e, mem_of_iso P e.symm⟩


lemma mem_of_isIso [ClosedUnderIsomorphisms P] {X Y : C} (f : X ⟶ Y) [IsIso f] (hX : P X) : P Y :=
  mem_of_iso P (asIso f) hX


lemma mem_iff_of_isIso [ClosedUnderIsomorphisms P] {X Y : C} (f : X ⟶ Y) [IsIso f] : P X ↔ P Y :=
  mem_iff_of_iso P (asIso f)


/-- The closure by isomorphisms of a predicate on objects in a category. -/
def isoClosure : C → Prop := fun X => ∃ (Y : C) (_ : P Y), Nonempty (X ≅ Y)


lemma mem_isoClosure_iff (X : C) :
                                                                 /-
                                                                   C : Type u_1
                                                                   inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                                   P : C → Prop
                                                                   X : C
                                                                   ⊢ Iff (CategoryTheory.isoClosure P X) (Exists fun Y => Exists fun x => Nonempt …
                                                                 -/
    isoClosure P X ↔ ∃ (Y : C) (_ : P Y), Nonempty (X ≅ Y) := by rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


variable {P} in
lemma mem_isoClosure {X Y : C} (h : P X) (e : X ⟶ Y) [IsIso e] : isoClosure P Y :=
  ⟨X, h, ⟨(asIso e).symm⟩⟩


lemma le_isoClosure : P ≤ isoClosure P :=
  fun X hX => ⟨X, hX, ⟨Iso.refl X⟩⟩


variable {P Q} in
lemma monotone_isoClosure (h : P ≤ Q) : isoClosure P ≤ isoClosure Q := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : C → Prop
    h : LE.le P Q
    ⊢ LE.le (CategoryTheory.isoClosure P) (CategoryTheory.isoClosure Q)
  -/
  rintro X ⟨X', hX', ⟨e⟩⟩
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : C → Prop
    h : LE.le P Q
    X X' : C
    hX' : P X'
    e : CategoryTheory.Iso X X'
    ⊢ CategoryTheory.isoClosure Q X
  -/
  exact ⟨X', h _ hX', ⟨e⟩⟩
  /-
    🎉 no goals
  -/


lemma isoClosure_eq_self [ClosedUnderIsomorphisms P] : isoClosure P = P := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    P : C → Prop
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
    ⊢ Eq (CategoryTheory.isoClosure P) P
  -/
  apply le_antisymm
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      P : C → Prop
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
      ⊢ LE.le (CategoryTheory.isoClosure P) P
    -/
  · intro X ⟨Y, hY, ⟨e⟩⟩
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      P : C → Prop
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
      X Y : C
      hY : P Y
      e : CategoryTheory.Iso X Y
      ⊢ P X
    -/
    exact mem_of_iso P e.symm hY
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      P : C → Prop
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms P
      ⊢ LE.le P (CategoryTheory.isoClosure P)
    -/
  · exact le_isoClosure P
    /-
      🎉 no goals
    -/


lemma isoClosure_le_iff [ClosedUnderIsomorphisms Q] : isoClosure P ≤ Q ↔ P ≤ Q :=
  ⟨(le_isoClosure P).trans,
                                               /-
                                                 C : Type u_1
                                                 inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                                 P Q : C → Prop
                                                 inst✝ : CategoryTheory.ClosedUnderIsomorphisms Q
                                                 h : LE.le P Q
                                                 ⊢ LE.le (CategoryTheory.isoClosure Q) Q
                                               -/
    fun h => (monotone_isoClosure h).trans (by rw [isoClosure_eq_self])⟩
                                               /-
                                                 🎉 no goals
                                               -/


instance : ClosedUnderIsomorphisms (isoClosure P) where
  of_iso := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      P Q : C → Prop
      ⊢ ∀ {X Y : C}, CategoryTheory.Iso X Y → CategoryTheory.isoClosure P X → Catego …
    -/
    rintro X Y e ⟨Z, hZ, ⟨f⟩⟩
    /-
      case intro.intro.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      P Q : C → Prop
      X Y : C
      e : CategoryTheory.Iso X Y
      Z : C
      hZ : P Z
      f : CategoryTheory.Iso X Z
      ⊢ CategoryTheory.isoClosure P Y
    -/
    exact ⟨Z, hZ, ⟨e.symm.trans f⟩⟩
    /-
      🎉 no goals
    -/


