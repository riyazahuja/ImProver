include F in
lemma Functor.reflects_precoherent : Precoherent C where
  pullback {B₁ B₂} f α _ X₁ π₁ _ := by
    obtain ⟨β, _, Y₂, τ₂, H, i, ι, hh⟩ := Precoherent.pullback (F.map f) _ _
      (fun a ↦ F.map (π₁ a)) inferInstance
    refine ⟨β, inferInstance, _, fun b ↦ F.preimage (F.effectiveEpiOver (Y₂ b) ≫ τ₂ b),
      F.finite_effectiveEpiFamily_of_map _ _ ?_,
        ⟨i, fun b ↦ F.preimage (F.effectiveEpiOver (Y₂ b) ≫ ι b), ?_⟩⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Precoherent D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        x✝ : CategoryTheory.EffectiveEpiFamily X₁ π₁
        β : Type
        w✝ : Finite β
        Y₂ : β → D
        τ₂ : (b : β) → Quiver.Hom (Y₂ b) (F.obj B₂)
        H : CategoryTheory.EffectiveEpiFamily Y₂ τ₂
        i : β → α
        ι : (b : β) → Quiver.Hom (Y₂ b) (F.obj (X₁ (i b)))
        hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (F.map (π₁ (i b)) …
        ⊢ CategoryTheory.EffectiveEpiFamily (fun a => F.obj ⋯.some.p) fun a => F.map ( …
      -/
    · simp only [Functor.map_preimage]
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Precoherent D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        x✝ : CategoryTheory.EffectiveEpiFamily X₁ π₁
        β : Type
        w✝ : Finite β
        Y₂ : β → D
        τ₂ : (b : β) → Quiver.Hom (Y₂ b) (F.obj B₂)
        H : CategoryTheory.EffectiveEpiFamily Y₂ τ₂
        i : β → α
        ι : (b : β) → Quiver.Hom (Y₂ b) (F.obj (X₁ (i b)))
        hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (F.map (π₁ (i b)) …
        ⊢ CategoryTheory.EffectiveEpiFamily (fun a => F.obj ⋯.some.p) fun a => Categor …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Precoherent D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        x✝ : CategoryTheory.EffectiveEpiFamily X₁ π₁
        β : Type
        w✝ : Finite β
        Y₂ : β → D
        τ₂ : (b : β) → Quiver.Hom (Y₂ b) (F.obj B₂)
        H : CategoryTheory.EffectiveEpiFamily Y₂ τ₂
        i : β → α
        ι : (b : β) → Quiver.Hom (Y₂ b) (F.obj (X₁ (i b)))
        hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (F.map (π₁ (i b)) …
        ⊢ ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp ((fun b => F.preimage (Cat …
      -/
    · intro b
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Precoherent D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        x✝ : CategoryTheory.EffectiveEpiFamily X₁ π₁
        β : Type
        w✝ : Finite β
        Y₂ : β → D
        τ₂ : (b : β) → Quiver.Hom (Y₂ b) (F.obj B₂)
        H : CategoryTheory.EffectiveEpiFamily Y₂ τ₂
        i : β → α
        ι : (b : β) → Quiver.Hom (Y₂ b) (F.obj (X₁ (i b)))
        hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (F.map (π₁ (i b)) …
        b : β
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun b => F.preimage (CategoryTheory …
      -/
      apply F.map_injective
      /-
        case intro.intro.intro.intro.intro.intro.intro.refine_2.a
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{?u.234, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.EffectivelyEnough
        inst✝² : CategoryTheory.Precoherent D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        x✝ : CategoryTheory.EffectiveEpiFamily X₁ π₁
        β : Type
        w✝ : Finite β
        Y₂ : β → D
        τ₂ : (b : β) → Quiver.Hom (Y₂ b) (F.obj B₂)
        H : CategoryTheory.EffectiveEpiFamily Y₂ τ₂
        i : β → α
        ι : (b : β) → Quiver.Hom (Y₂ b) (F.obj (X₁ (i b)))
        hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (F.map (π₁ (i b)) …
        b : β
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((fun b => F.preimage (Categor …
      -/
      simp [hh b]
      /-
        🎉 no goals
      -/


