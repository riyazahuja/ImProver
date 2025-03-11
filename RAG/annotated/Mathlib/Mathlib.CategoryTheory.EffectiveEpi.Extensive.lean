theorem effectiveEpi_desc_iff_effectiveEpiFamily {α : Type} [Finite α]
    {B : C} (X : α → C) (π : (a : α) → X a ⟶ B) :
    EffectiveEpi (Sigma.desc π) ↔ EffectiveEpiFamily X π := by
  exact ⟨fun h ↦ ⟨⟨@effectiveEpiFamilyStructOfEffectiveEpiDesc _ _ _ _ X π _ h _ _ (fun g ↦
    (FinitaryPreExtensive.sigma_desc_iso (fun a ↦ Sigma.ι X a) g inferInstance).epi_of_iso)⟩⟩,
    fun _ ↦ inferInstance⟩


instance [F.ReflectsEffectiveEpis] : F.ReflectsFiniteEffectiveEpiFamilies where
  reflects {α _ B} X π h := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.FinitaryPreExtensive C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      inst✝² : CategoryTheory.FinitaryPreExtensive D
      F : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteCoproducts F
      inst✝ : F.ReflectsEffectiveEpis
      α : Type
      x✝ : Finite α
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map (π …
      ⊢ CategoryTheory.EffectiveEpiFamily X π
    -/
    simp only [← effectiveEpi_desc_iff_effectiveEpiFamily]
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.FinitaryPreExtensive C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      inst✝² : CategoryTheory.FinitaryPreExtensive D
      F : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteCoproducts F
      inst✝ : F.ReflectsEffectiveEpis
      α : Type
      x✝ : Finite α
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map (π …
      ⊢ CategoryTheory.EffectiveEpi (CategoryTheory.Limits.Sigma.desc π)
    -/
    apply F.effectiveEpi_of_map
    convert (inferInstance :
      EffectiveEpi (inv (sigmaComparison F X) ≫ (Sigma.desc (fun a ↦ F.map (π a)))))
    /-
      case h.e'_5
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.FinitaryPreExtensive C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      inst✝² : CategoryTheory.FinitaryPreExtensive D
      F : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteCoproducts F
      inst✝ : F.ReflectsEffectiveEpis
      α : Type
      x✝ : Finite α
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map (π …
      ⊢ Eq (F.map (CategoryTheory.Limits.Sigma.desc π)) (CategoryTheory.CategoryStru …
    -/
    simp
    /-
      🎉 no goals
    -/


instance [F.PreservesEffectiveEpis] : F.PreservesFiniteEffectiveEpiFamilies where
  preserves {α _ B} X π h := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.FinitaryPreExtensive C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      inst✝² : CategoryTheory.FinitaryPreExtensive D
      F : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteCoproducts F
      inst✝ : F.PreservesEffectiveEpis
      α : Type
      x✝ : Finite α
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily X π
      ⊢ CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map (π a)
    -/
    simp only [← effectiveEpi_desc_iff_effectiveEpiFamily]
    convert (inferInstance :
      EffectiveEpi ((sigmaComparison F X) ≫ (F.map (Sigma.desc π))))
    /-
      case h.e'_5
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.FinitaryPreExtensive C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      inst✝² : CategoryTheory.FinitaryPreExtensive D
      F : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteCoproducts F
      inst✝ : F.PreservesEffectiveEpis
      α : Type
      x✝ : Finite α
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily X π
      ⊢ Eq (CategoryTheory.Limits.Sigma.desc fun a => F.map (π a)) (CategoryTheory.C …
    -/
    simp
    /-
      🎉 no goals
    -/


