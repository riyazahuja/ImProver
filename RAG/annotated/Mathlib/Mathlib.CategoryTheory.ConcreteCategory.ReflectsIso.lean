instance : (forget (Type u)).ReflectsIsomorphisms where reflects _ _ _ {i} := i


/-- A `forget₂ C D` forgetful functor between concrete categories `C` and `D`
where `forget C` reflects isomorphisms, itself reflects isomorphisms.
-/
theorem reflectsIsomorphisms_forget₂ [HasForget₂ C D] [(forget C).ReflectsIsomorphisms] :
    (forget₂ C D).ReflectsIsomorphisms :=
  { reflects := fun X Y f {i} => by
      /-
        C : Type (u + 1)
        inst✝⁵ : CategoryTheory.Category.{u_1, u + 1} C
        inst✝⁴ : CategoryTheory.ConcreteCategory C
        D : Type (u + 1)
        inst✝³ : CategoryTheory.Category.{u_2, u + 1} D
        inst✝² : CategoryTheory.ConcreteCategory D
        inst✝¹ : CategoryTheory.HasForget₂ C D
        inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
        X Y : C
        f : Quiver.Hom X Y
        i : CategoryTheory.IsIso ((CategoryTheory.forget₂ C D).map f)
        ⊢ CategoryTheory.IsIso f
      -/
      haveI i' : IsIso ((forget D).map ((forget₂ C D).map f)) := Functor.map_isIso (forget D) _
      haveI : IsIso ((forget C).map f) := by
        have := @HasForget₂.forget_comp C D
        rwa [← this]
      /-
        C : Type (u + 1)
        inst✝⁵ : CategoryTheory.Category.{u_1, u + 1} C
        inst✝⁴ : CategoryTheory.ConcreteCategory C
        D : Type (u + 1)
        inst✝³ : CategoryTheory.Category.{u_2, u + 1} D
        inst✝² : CategoryTheory.ConcreteCategory D
        inst✝¹ : CategoryTheory.HasForget₂ C D
        inst✝ : (CategoryTheory.forget C).ReflectsIsomorphisms
        X Y : C
        f : Quiver.Hom X Y
        i : CategoryTheory.IsIso ((CategoryTheory.forget₂ C D).map f)
        i' : CategoryTheory.IsIso ((CategoryTheory.forget D).map ((CategoryTheory.forg …
        this : CategoryTheory.IsIso ((CategoryTheory.forget C).map f)
        ⊢ CategoryTheory.IsIso f
      -/
      apply isIso_of_reflects_iso f (forget C) }
      /-
        🎉 no goals
      -/


