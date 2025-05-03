include F in
lemma Functor.reflects_preregular : Preregular C where
  exists_fac f g _ := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.EffectivelyEnough
      inst✝² : CategoryTheory.Preregular D
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Z✝ Y✝
      x✝ : CategoryTheory.EffectiveEpi g
      ⊢ Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Category …
    -/
    obtain ⟨W, f', _, i, w⟩ := Preregular.exists_fac (F.map f) (F.map g)
    refine ⟨_, F.preimage (F.effectiveEpiOver W ≫ f'),
      ⟨F.effectiveEpi_of_map _ ?_, F.preimage (F.effectiveEpiOver W ≫ i), ?_⟩⟩
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Preregular D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        X✝ Y✝ Z✝ : C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Z✝ Y✝
        x✝ : CategoryTheory.EffectiveEpi g
        W : D
        f' : Quiver.Hom W (F.obj X✝)
        w✝ : CategoryTheory.EffectiveEpi f'
        i : Quiver.Hom W (F.obj Z✝)
        w : Eq (CategoryTheory.CategoryStruct.comp i (F.map g)) (CategoryTheory.Catego …
        ⊢ CategoryTheory.EffectiveEpi (F.map (F.preimage (CategoryTheory.CategoryStruc …
      -/
    · simp only [Functor.map_preimage]
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Preregular D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        X✝ Y✝ Z✝ : C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Z✝ Y✝
        x✝ : CategoryTheory.EffectiveEpi g
        W : D
        f' : Quiver.Hom W (F.obj X✝)
        w✝ : CategoryTheory.EffectiveEpi f'
        i : Quiver.Hom W (F.obj Z✝)
        w : Eq (CategoryTheory.CategoryStruct.comp i (F.map g)) (CategoryTheory.Catego …
        ⊢ CategoryTheory.EffectiveEpi (CategoryTheory.CategoryStruct.comp (F.effective …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Preregular D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        X✝ Y✝ Z✝ : C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Z✝ Y✝
        x✝ : CategoryTheory.EffectiveEpi g
        W : D
        f' : Quiver.Hom W (F.obj X✝)
        w✝ : CategoryTheory.EffectiveEpi f'
        i : Quiver.Hom W (F.obj Z✝)
        w : Eq (CategoryTheory.CategoryStruct.comp i (F.map g)) (CategoryTheory.Catego …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.preimage (CategoryTheory.CategoryS …
      -/
    · apply F.map_injective
      /-
        case intro.intro.intro.intro.refine_2.a
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Preregular D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        X✝ Y✝ Z✝ : C
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Z✝ Y✝
        x✝ : CategoryTheory.EffectiveEpi g
        W : D
        f' : Quiver.Hom W (F.obj X✝)
        w✝ : CategoryTheory.EffectiveEpi f'
        i : Quiver.Hom W (F.obj Z✝)
        w : Eq (CategoryTheory.CategoryStruct.comp i (F.map g)) (CategoryTheory.Catego …
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (F.preimage (CategoryTheory.Ca …
      -/
      simp [w]
      /-
        🎉 no goals
      -/


