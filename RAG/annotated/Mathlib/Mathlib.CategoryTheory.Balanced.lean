/-- A category is called balanced if any morphism that is both monic and epic is an isomorphism. -/
class Balanced : Prop where
  isIso_of_mono_of_epi : ∀ {X Y : C} (f : X ⟶ Y) [Mono f] [Epi f], IsIso f


theorem isIso_of_mono_of_epi [Balanced C] {X Y : C} (f : X ⟶ Y) [Mono f] [Epi f] : IsIso f :=
  Balanced.isIso_of_mono_of_epi _


theorem isIso_iff_mono_and_epi [Balanced C] {X Y : C} (f : X ⟶ Y) : IsIso f ↔ Mono f ∧ Epi f :=
  ⟨fun _ => ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ => isIso_of_mono_of_epi _⟩


theorem balanced_opposite [Balanced C] : Balanced Cᵒᵖ :=
  { isIso_of_mono_of_epi := fun f fmono fepi => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Balanced C
        X✝ Y✝ : Opposite C
        f : Quiver.Hom X✝ Y✝
        fmono : CategoryTheory.Mono f
        fepi : CategoryTheory.Epi f
        ⊢ CategoryTheory.IsIso f
      -/
      rw [← Quiver.Hom.op_unop f]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Balanced C
        X✝ Y✝ : Opposite C
        f : Quiver.Hom X✝ Y✝
        fmono : CategoryTheory.Mono f
        fepi : CategoryTheory.Epi f
        ⊢ CategoryTheory.IsIso f.unop.op
      -/
      exact isIso_of_op _ }
      /-
        🎉 no goals
      -/


