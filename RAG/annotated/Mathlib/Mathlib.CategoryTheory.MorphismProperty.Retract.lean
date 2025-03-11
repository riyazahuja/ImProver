/-- A class of morphisms is stable under retracts if a retract of such a morphism still
lies in the class. -/
class IsStableUnderRetracts (P : MorphismProperty C) : Prop where
  of_retract {X Y Z W : C} {f : X ⟶ Y} {g : Z ⟶ W} (h : RetractArrow f g) (hg : P g) : P f


lemma of_retract {P : MorphismProperty C} [P.IsStableUnderRetracts]
    {X Y Z W : C} {f : X ⟶ Y} {g : Z ⟶ W} (h : RetractArrow f g) (hg : P g) : P f :=
  IsStableUnderRetracts.of_retract h hg


instance IsStableUnderRetracts.monomorphisms : (monomorphisms C).IsStableUnderRetracts where
  of_retract {_ _ _ _ f g} h (hg : Mono g) := ⟨fun α β w ↦ by
    rw [← cancel_mono h.i.left, ← cancel_mono g, Category.assoc, Category.assoc,
      h.i_w, reassoc_of% w]⟩


instance IsStableUnderRetracts.epimorphisms : (epimorphisms C).IsStableUnderRetracts where
  of_retract {_ _ _ _ f g} h (hg : Epi g) := ⟨fun α β w ↦ by
    rw [← cancel_epi h.r.right, ← cancel_epi g, ← Category.assoc, ← Category.assoc, ← h.r_w,
      Category.assoc, Category.assoc, w]⟩


instance IsStableUnderRetracts.isomorphisms : (isomorphisms C).IsStableUnderRetracts where
  of_retract {X Y Z W f g} h (_ : IsIso _) := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y Z W : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z W
      h : CategoryTheory.RetractArrow f g
      x✝ : CategoryTheory.IsIso g
      ⊢ CategoryTheory.MorphismProperty.isomorphisms C f
    -/
    refine ⟨h.i.right ≫ inv g ≫ h.r.left, ?_, ?_⟩
      /-
        case refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Z W
        h : CategoryTheory.RetractArrow f g
        x✝ : CategoryTheory.IsIso g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
    · rw [← h.i_w_assoc, IsIso.hom_inv_id_assoc, h.retract_left]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y Z W : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Z W
        h : CategoryTheory.RetractArrow f g
        x✝ : CategoryTheory.IsIso g
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
      -/
    · rw [Category.assoc, Category.assoc, h.r_w, IsIso.inv_hom_id_assoc, h.retract_right]
      /-
        🎉 no goals
      -/


