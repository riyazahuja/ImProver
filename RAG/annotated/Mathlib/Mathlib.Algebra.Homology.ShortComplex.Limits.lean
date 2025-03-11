/-- If a cone with values in `ShortComplex C` is such that it becomes limit
when we apply the three projections `ShortComplex C ⥤ C`, then it is limit. -/
def isLimitOfIsLimitπ (c : Cone F)
    (h₁ : IsLimit (π₁.mapCone c)) (h₂ : IsLimit (π₂.mapCone c))
    (h₃ : IsLimit (π₃.mapCone c)) : IsLimit c where
  lift s :=
    { τ₁ := h₁.lift (π₁.mapCone s)
      τ₂ := h₂.lift (π₂.mapCone s)
      τ₃ := h₃.lift (π₃.mapCone s)
      comm₁₂ := h₂.hom_ext (fun j => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₁ := h₁.fac (π₁.mapCone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₁.lift (CategoryTheo …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₂ := h₂.fac (π₂.mapCone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₁.lift (CategoryTheo …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₁₂ := fun j => (c.π.app j).comm₁₂
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₁.lift (CategoryTheo …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₁₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).τ₁ (F.obj …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₁₂' := fun j => (s.π.app j).comm₁₂
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₁.lift (CategoryTheo …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₁₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).τ₁ (F.obj …
          eq₁₂' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.π.app j).τ₁ (F.ob …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        dsimp at eq₁ eq₂ eq₁₂ eq₁₂' ⊢
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₁.lift (CategoryTheo …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₁₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).τ₁ (F.obj …
          eq₁₂' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.π.app j).τ₁ (F.ob …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [assoc, assoc, ← eq₁₂, reassoc_of% eq₁, eq₂, eq₁₂'])
        /-
          🎉 no goals
        -/
      comm₂₃ := h₃.hom_ext (fun j => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₂ := h₂.fac (π₂.mapCone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₃ := h₃.fac (π₃.mapCone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₃.lift (CategoryTheo …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₂₃ := fun j => (c.π.app j).comm₂₃
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₃.lift (CategoryTheo …
          eq₂₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).τ₂ (F.obj …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq₂₃' := fun j => (s.π.app j).comm₂₃
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₃.lift (CategoryTheo …
          eq₂₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).τ₂ (F.obj …
          eq₂₃' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.π.app j).τ₂ (F.ob …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        dsimp at eq₂ eq₃ eq₂₃ eq₂₃' ⊢
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.103, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cone F
          h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
          h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
          h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
          s : CategoryTheory.Limits.Cone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₂.lift (CategoryTheo …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h₃.lift (CategoryTheo …
          eq₂₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).τ₂ (F.obj …
          eq₂₃' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.π.app j).τ₂ (F.ob …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [assoc, assoc, ← eq₂₃, reassoc_of% eq₂, eq₃, eq₂₃']) }
        /-
          🎉 no goals
        -/
                /-
                  J : Type u_1
                  C : Type u_2
                  inst✝² : CategoryTheory.Category.{?u.103, u_1} J
                  inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                  F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                  c : CategoryTheory.Limits.Cone F
                  h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
                  h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
                  h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
                  s : CategoryTheory.Limits.Cone F
                  j : J
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => { τ₁ := h₁.lift (CategoryT …
                -/
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
  fac s j := by ext <;> apply IsLimit.fac
                        /-
                          🎉 no goals
                        -/
  uniq s m hm := by
    /-
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.103, u_1} J
      inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
      c : CategoryTheory.Limits.Cone F
      h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
      h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
      h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      ⊢ Eq m ((fun s => { τ₁ := h₁.lift (CategoryTheory.ShortComplex.π₁.mapCone s),  …
    -/
    ext
      /-
        case h₁
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.103, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cone F
        h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
        h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
        h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt c.pt
        hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
        ⊢ Eq m.τ₁ ((fun s => { τ₁ := h₁.lift (CategoryTheory.ShortComplex.π₁.mapCone s …
      -/
    · exact h₁.uniq (π₁.mapCone s) _ (fun j => π₁.congr_map (hm j))
      /-
        🎉 no goals
      -/
      /-
        case h₂
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.103, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cone F
        h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
        h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
        h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt c.pt
        hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
        ⊢ Eq m.τ₂ ((fun s => { τ₁ := h₁.lift (CategoryTheory.ShortComplex.π₁.mapCone s …
      -/
    · exact h₂.uniq (π₂.mapCone s) _ (fun j => π₂.congr_map (hm j))
      /-
        🎉 no goals
      -/
      /-
        case h₃
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.103, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.107, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cone F
        h₁ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₁.mapCone c)
        h₂ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₂.mapCone c)
        h₃ : CategoryTheory.Limits.IsLimit (CategoryTheory.ShortComplex.π₃.mapCone c)
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt c.pt
        hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
        ⊢ Eq m.τ₃ ((fun s => { τ₁ := h₁.lift (CategoryTheory.ShortComplex.π₁.mapCone s …
      -/
    · exact h₃.uniq (π₃.mapCone s) _ (fun j => π₃.congr_map (hm j))
      /-
        🎉 no goals
      -/


/-- Construction of a limit cone for a functor `J ⥤ ShortComplex C` using the limits
of the three components `J ⥤ C`. -/
noncomputable def limitCone : Cone F :=
  Cone.mk (ShortComplex.mk (limMap (whiskerLeft F π₁Toπ₂)) (limMap (whiskerLeft F π₂Toπ₃))
          /-
            J : Type u_1
            C : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.19826, u_1} J
            inst✝⁴ : CategoryTheory.Category.{?u.19830, u_2} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
            inst✝² : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₁)
            inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₂)
            inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₃)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap (Catego …
          -/
      (by aesop_cat))
          /-
            🎉 no goals
          -/
    { app := fun j => Hom.mk (limit.π _ _) (limit.π _ _) (limit.π _ _)
            /-
              J : Type u_1
              C : Type u_2
              inst✝⁵ : CategoryTheory.Category.{?u.19826, u_1} J
              inst✝⁴ : CategoryTheory.Category.{?u.19830, u_2} C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
              inst✝² : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₁)
              inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₂)
              inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₃)
              j : J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
            -/
            /-
              🎉 no goals
            -/
        (by aesop_cat) (by aesop_cat)
                           /-
                             🎉 no goals
                           -/
      naturality := fun _ _ f => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝⁵ : CategoryTheory.Category.{?u.19826, u_1} J
          inst✝⁴ : CategoryTheory.Category.{?u.19830, u_2} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          inst✝² : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₁)
          inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₂)
          inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₃)
          x✝¹ x✝ : J
          f : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext
        all_goals
          dsimp
          erw [id_comp, limit.w] }


/-- `limitCone F` becomes limit after the application of `π₁ : ShortComplex C ⥤ C`. -/
noncomputable def isLimitπ₁MapConeLimitCone : IsLimit (π₁.mapCone (limitCone F)) :=
                                                                    /-
                                                                      J : Type u_1
                                                                      C : Type u_2
                                                                      inst✝⁵ : CategoryTheory.Category.{?u.36538, u_1} J
                                                                      inst✝⁴ : CategoryTheory.Category.{?u.36542, u_2} C
                                                                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                      F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                                                      inst✝² : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₁)
                                                                      inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₂)
                                                                      inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₃)
                                                                      ⊢ ∀ (j : J), Eq ((CategoryTheory.Limits.limit.cone (F.comp CategoryTheory.Shor …
                                                                    -/
  (IsLimit.ofIsoLimit (limit.isLimit _) (Cones.ext (Iso.refl _) (by aesop_cat)))
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- `limitCone F` becomes limit after the application of `π₂ : ShortComplex C ⥤ C`. -/
noncomputable def isLimitπ₂MapConeLimitCone : IsLimit (π₂.mapCone (limitCone F)) :=
                                                                    /-
                                                                      J : Type u_1
                                                                      C : Type u_2
                                                                      inst✝⁵ : CategoryTheory.Category.{?u.41997, u_1} J
                                                                      inst✝⁴ : CategoryTheory.Category.{?u.42001, u_2} C
                                                                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                      F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                                                      inst✝² : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₁)
                                                                      inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₂)
                                                                      inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₃)
                                                                      ⊢ ∀ (j : J), Eq ((CategoryTheory.Limits.limit.cone (F.comp CategoryTheory.Shor …
                                                                    -/
  (IsLimit.ofIsoLimit (limit.isLimit _) (Cones.ext (Iso.refl _) (by aesop_cat)))
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- `limitCone F` becomes limit after the application of `π₃ : ShortComplex C ⥤ C`. -/
noncomputable def isLimitπ₃MapConeLimitCone : IsLimit (π₃.mapCone (limitCone F)) :=
                                                                    /-
                                                                      J : Type u_1
                                                                      C : Type u_2
                                                                      inst✝⁵ : CategoryTheory.Category.{?u.47456, u_1} J
                                                                      inst✝⁴ : CategoryTheory.Category.{?u.47460, u_2} C
                                                                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                      F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                                                      inst✝² : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₁)
                                                                      inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₂)
                                                                      inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.ShortComplex.π₃)
                                                                      ⊢ ∀ (j : J), Eq ((CategoryTheory.Limits.limit.cone (F.comp CategoryTheory.Shor …
                                                                    -/
  (IsLimit.ofIsoLimit (limit.isLimit _) (Cones.ext (Iso.refl _) (by aesop_cat)))
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- `limitCone F` is limit. -/
noncomputable def isLimitLimitCone : IsLimit (limitCone F) :=
  isLimitOfIsLimitπ _ (isLimitπ₁MapConeLimitCone F)
    (isLimitπ₂MapConeLimitCone F) (isLimitπ₃MapConeLimitCone F)


instance hasLimit_of_hasLimitπ : HasLimit F := ⟨⟨⟨_, isLimitLimitCone _⟩⟩⟩


noncomputable instance : PreservesLimit F π₁ :=
  preservesLimit_of_preserves_limit_cone (isLimitLimitCone F) (isLimitπ₁MapConeLimitCone F)


noncomputable instance : PreservesLimit F π₂ :=
  preservesLimit_of_preserves_limit_cone (isLimitLimitCone F) (isLimitπ₂MapConeLimitCone F)


noncomputable instance : PreservesLimit F π₃ :=
  preservesLimit_of_preserves_limit_cone (isLimitLimitCone F) (isLimitπ₃MapConeLimitCone F)


instance hasLimitsOfShape :
    HasLimitsOfShape J (ShortComplex C) where


noncomputable instance : PreservesLimitsOfShape J (π₁ : _ ⥤ C) where


noncomputable instance : PreservesLimitsOfShape J (π₂ : _ ⥤ C) where


noncomputable instance : PreservesLimitsOfShape J (π₃ : _ ⥤ C) where


instance hasFiniteLimits : HasFiniteLimits (ShortComplex C) :=
  ⟨fun _ _ _ => inferInstance⟩


noncomputable instance : PreservesFiniteLimits (π₁ : _ ⥤ C) :=
  ⟨fun _ _ _ => inferInstance⟩


noncomputable instance : PreservesFiniteLimits (π₂ : _ ⥤ C) :=
  ⟨fun _ _ _ => inferInstance⟩


noncomputable instance : PreservesFiniteLimits (π₃ : _ ⥤ C) :=
  ⟨fun _ _ _ => inferInstance⟩


instance preservesMonomorphisms_π₁ :
    Functor.PreservesMonomorphisms (π₁ : _ ⥤ C) :=
  CategoryTheory.preservesMonomorphisms_of_preservesLimitsOfShape _


instance preservesMonomorphisms_π₂ :
    Functor.PreservesMonomorphisms (π₂ : _ ⥤ C) :=
  CategoryTheory.preservesMonomorphisms_of_preservesLimitsOfShape _


instance preservesMonomorphisms_π₃ :
    Functor.PreservesMonomorphisms (π₃ : _ ⥤ C) :=
  CategoryTheory.preservesMonomorphisms_of_preservesLimitsOfShape _


/-- If a cocone with values in `ShortComplex C` is such that it becomes colimit
when we apply the three projections `ShortComplex C ⥤ C`, then it is colimit. -/
def isColimitOfIsColimitπ (c : Cocone F)
    (h₁ : IsColimit (π₁.mapCocone c)) (h₂ : IsColimit (π₂.mapCocone c))
    (h₃ : IsColimit (π₃.mapCocone c)) : IsColimit c where
  desc s :=
    { τ₁ := h₁.desc (π₁.mapCocone s)
      τ₂ := h₂.desc (π₂.mapCocone s)
      τ₃ := h₃.desc (π₃.mapCocone s)
      comm₁₂ := h₁.hom_ext (fun j => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₁.mapC …
        -/
        have eq₁ := h₁.fac (π₁.mapCocone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₁.mapC …
        -/
        have eq₂ := h₂.fac (π₂.mapCocone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₁.mapC …
        -/
        have eq₁₂ := fun j => (c.ι.app j).comm₁₂
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₁₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₁ (((Cat …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₁.mapC …
        -/
        have eq₁₂' := fun j => (s.ι.app j).comm₁₂
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₁₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₁ (((Cat …
          eq₁₂' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j).τ₁ (((Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₁.mapC …
        -/
        dsimp at eq₁ eq₂ eq₁₂ eq₁₂' ⊢
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₁ (h₁.des …
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₂ (h₂.des …
          eq₁₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₁ c.pt.f …
          eq₁₂' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j).τ₁ s.pt. …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₁ (CategoryTheory.Catego …
        -/
        rw [reassoc_of% (eq₁ j), eq₁₂', reassoc_of% eq₁₂, eq₂])
        /-
          🎉 no goals
        -/
      comm₂₃ := h₂.hom_ext (fun j => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₂.mapC …
        -/
        have eq₂ := h₂.fac (π₂.mapCocone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₂.mapC …
        -/
        have eq₃ := h₃.fac (π₃.mapCocone s)
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₂.mapC …
        -/
        have eq₂₃ := fun j => (c.ι.app j).comm₂₃
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₂₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₂ (((Cat …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₂.mapC …
        -/
        have eq₂₃' := fun j => (s.ι.app j).comm₂₃
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Short …
          eq₂₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₂ (((Cat …
          eq₂₃' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j).τ₂ (((Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShortComplex.π₂.mapC …
        -/
        dsimp at eq₂ eq₃ eq₂₃ eq₂₃' ⊢
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          c : CategoryTheory.Limits.Cocone F
          h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
          h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
          h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
          s : CategoryTheory.Limits.Cocone F
          j : J
          eq₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₂ (h₂.des …
          eq₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₃ (h₃.des …
          eq₂₃ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₂ c.pt.g …
          eq₂₃' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j).τ₂ s.pt. …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j).τ₂ (CategoryTheory.Catego …
        -/
        rw [reassoc_of% (eq₂ j), eq₂₃', reassoc_of% eq₂₃, eq₃]) }
        /-
          🎉 no goals
        -/
  fac s j := by
    /-
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
      inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
      c : CategoryTheory.Limits.Cocone F
      h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
      h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
      h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { τ₁ := h₁.des …
    -/
    ext
      /-
        case h₁
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cocone F
        h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
        h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
        h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
        s : CategoryTheory.Limits.Cocone F
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { τ₁ := h₁.des …
      -/
    · apply IsColimit.fac h₁
      /-
        🎉 no goals
      -/
      /-
        case h₂
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cocone F
        h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
        h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
        h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
        s : CategoryTheory.Limits.Cocone F
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { τ₁ := h₁.des …
      -/
    · apply IsColimit.fac h₂
      /-
        🎉 no goals
      -/
      /-
        case h₃
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cocone F
        h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
        h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
        h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
        s : CategoryTheory.Limits.Cocone F
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { τ₁ := h₁.des …
      -/
    · apply IsColimit.fac h₃
      /-
        🎉 no goals
      -/
  uniq s m hm := by
    /-
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
      inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
      c : CategoryTheory.Limits.Cocone F
      h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
      h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
      h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
      ⊢ Eq m ((fun s => { τ₁ := h₁.desc (CategoryTheory.ShortComplex.π₁.mapCocone s) …
    -/
    ext
      /-
        case h₁
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cocone F
        h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
        h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
        h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom c.pt s.pt
        hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
        ⊢ Eq m.τ₁ ((fun s => { τ₁ := h₁.desc (CategoryTheory.ShortComplex.π₁.mapCocone …
      -/
    · exact h₁.uniq (π₁.mapCocone s) _ (fun j => π₁.congr_map (hm j))
      /-
        🎉 no goals
      -/
      /-
        case h₂
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cocone F
        h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
        h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
        h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom c.pt s.pt
        hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
        ⊢ Eq m.τ₂ ((fun s => { τ₁ := h₁.desc (CategoryTheory.ShortComplex.π₁.mapCocone …
      -/
    · exact h₂.uniq (π₂.mapCocone s) _ (fun j => π₂.congr_map (hm j))
      /-
        🎉 no goals
      -/
      /-
        case h₃
        J : Type u_1
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.88585, u_1} J
        inst✝¹ : CategoryTheory.Category.{?u.88589, u_2} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
        c : CategoryTheory.Limits.Cocone F
        h₁ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₁.mapCocone …
        h₂ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₂.mapCocone …
        h₃ : CategoryTheory.Limits.IsColimit (CategoryTheory.ShortComplex.π₃.mapCocone …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom c.pt s.pt
        hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app …
        ⊢ Eq m.τ₃ ((fun s => { τ₁ := h₁.desc (CategoryTheory.ShortComplex.π₁.mapCocone …
      -/
    · exact h₃.uniq (π₃.mapCocone s) _ (fun j => π₃.congr_map (hm j))
      /-
        🎉 no goals
      -/


/-- Construction of a colimit cocone for a functor `J ⥤ ShortComplex C` using the colimits
of the three components `J ⥤ C`. -/
noncomputable def colimitCocone : Cocone F :=
  Cocone.mk (ShortComplex.mk (colimMap (whiskerLeft F π₁Toπ₂)) (colimMap (whiskerLeft F π₂Toπ₃))
          /-
            J : Type u_1
            C : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.107060, u_1} J
            inst✝⁴ : CategoryTheory.Category.{?u.107064, u_2} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
            inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap (Cate …
          -/
      (by aesop_cat))
          /-
            🎉 no goals
          -/
    { app := fun j => Hom.mk (colimit.ι (F ⋙ π₁) _) (colimit.ι (F ⋙ π₂) _)
                                   /-
                                     J : Type u_1
                                     C : Type u_2
                                     inst✝⁵ : CategoryTheory.Category.{?u.107060, u_1} J
                                     inst✝⁴ : CategoryTheory.Category.{?u.107064, u_2} C
                                     inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                     F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                     inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                     inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                     inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
                                     j : J
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
        (colimit.ι (F ⋙ π₃) _) (by aesop_cat) (by aesop_cat)
                                                  /-
                                                    🎉 no goals
                                                  -/
      naturality := fun _ _ f => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝⁵ : CategoryTheory.Category.{?u.107060, u_1} J
          inst✝⁴ : CategoryTheory.Category.{?u.107064, u_2} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
          inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
          inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
          inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
          x✝¹ x✝ : J
          f : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { τ₁ := Category …
        -/
        ext
          /-
            case h₁
            J : Type u_1
            C : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.107060, u_1} J
            inst✝⁴ : CategoryTheory.Category.{?u.107064, u_2} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
            inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
            x✝¹ x✝ : J
            f : Quiver.Hom x✝¹ x✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { τ₁ := Category …
          -/
        · dsimp; erw [comp_id, colimit.w (F ⋙ π₁)]
                 /-
                   🎉 no goals
                 -/
          /-
            case h₂
            J : Type u_1
            C : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.107060, u_1} J
            inst✝⁴ : CategoryTheory.Category.{?u.107064, u_2} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
            inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
            x✝¹ x✝ : J
            f : Quiver.Hom x✝¹ x✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { τ₁ := Category …
          -/
        · dsimp; erw [comp_id, colimit.w (F ⋙ π₂)]
                 /-
                   🎉 no goals
                 -/
          /-
            case h₃
            J : Type u_1
            C : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.107060, u_1} J
            inst✝⁴ : CategoryTheory.Category.{?u.107064, u_2} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
            inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
            inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
            x✝¹ x✝ : J
            f : Quiver.Hom x✝¹ x✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => { τ₁ := Category …
          -/
        · dsimp; erw [comp_id, colimit.w (F ⋙ π₃)] }
                 /-
                   🎉 no goals
                 -/


/-- `colimitCocone F` becomes colimit after the application of `π₁ : ShortComplex C ⥤ C`. -/
noncomputable def isColimitπ₁MapCoconeColimitCocone :
    IsColimit (π₁.mapCocone (colimitCocone F)) :=
                                                                              /-
                                                                                J : Type u_1
                                                                                C : Type u_2
                                                                                inst✝⁵ : CategoryTheory.Category.{?u.125175, u_1} J
                                                                                inst✝⁴ : CategoryTheory.Category.{?u.125179, u_2} C
                                                                                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                                                                inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                                                                inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                                                                inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
                                                                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.co …
                                                                              -/
  (IsColimit.ofIsoColimit (colimit.isColimit _) (Cocones.ext (Iso.refl _) (by aesop_cat)))
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- `colimitCocone F` becomes colimit after the application of `π₂ : ShortComplex C ⥤ C`. -/
noncomputable def isColimitπ₂MapCoconeColimitCocone :
    IsColimit (π₂.mapCocone (colimitCocone F)) :=
                                                                              /-
                                                                                J : Type u_1
                                                                                C : Type u_2
                                                                                inst✝⁵ : CategoryTheory.Category.{?u.130683, u_1} J
                                                                                inst✝⁴ : CategoryTheory.Category.{?u.130687, u_2} C
                                                                                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                                                                inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                                                                inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                                                                inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
                                                                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.co …
                                                                              -/
  (IsColimit.ofIsoColimit (colimit.isColimit _) (Cocones.ext (Iso.refl _) (by aesop_cat)))
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- `colimitCocone F` becomes colimit after the application of `π₃ : ShortComplex C ⥤ C`. -/
noncomputable def isColimitπ₃MapCoconeColimitCocone :
    IsColimit (π₃.mapCocone (colimitCocone F)) :=
                                                                              /-
                                                                                J : Type u_1
                                                                                C : Type u_2
                                                                                inst✝⁵ : CategoryTheory.Category.{?u.136191, u_1} J
                                                                                inst✝⁴ : CategoryTheory.Category.{?u.136195, u_2} C
                                                                                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                F : CategoryTheory.Functor J (CategoryTheory.ShortComplex C)
                                                                                inst✝² : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                                                                inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex. …
                                                                                inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.ShortComplex.π₃)
                                                                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.co …
                                                                              -/
  (IsColimit.ofIsoColimit (colimit.isColimit _) (Cocones.ext (Iso.refl _) (by aesop_cat)))
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- `colimitCocone F` is colimit. -/
noncomputable def isColimitColimitCocone : IsColimit (colimitCocone F) :=
  isColimitOfIsColimitπ _ (isColimitπ₁MapCoconeColimitCocone F)
    (isColimitπ₂MapCoconeColimitCocone F) (isColimitπ₃MapCoconeColimitCocone F)


instance hasColimit_of_hasColimitπ : HasColimit F := ⟨⟨⟨_, isColimitColimitCocone _⟩⟩⟩


noncomputable instance : PreservesColimit F π₁ :=
  preservesColimit_of_preserves_colimit_cocone (isColimitColimitCocone F)
    (isColimitπ₁MapCoconeColimitCocone F)


noncomputable instance : PreservesColimit F π₂ :=
  preservesColimit_of_preserves_colimit_cocone (isColimitColimitCocone F)
    (isColimitπ₂MapCoconeColimitCocone F)


noncomputable instance : PreservesColimit F π₃ :=
  preservesColimit_of_preserves_colimit_cocone (isColimitColimitCocone F)
    (isColimitπ₃MapCoconeColimitCocone F)


instance hasColimitsOfShape :
    HasColimitsOfShape J (ShortComplex C) where


noncomputable instance : PreservesColimitsOfShape J (π₁ : _ ⥤ C) where


noncomputable instance : PreservesColimitsOfShape J (π₂ : _ ⥤ C) where


noncomputable instance : PreservesColimitsOfShape J (π₃ : _ ⥤ C) where


instance hasFiniteColimits : HasFiniteColimits (ShortComplex C) :=
  ⟨fun _ _ _ => inferInstance⟩


noncomputable instance : PreservesFiniteColimits (π₁ : _ ⥤ C) :=
  ⟨fun _ _ _ => inferInstance⟩


noncomputable instance : PreservesFiniteColimits (π₂ : _ ⥤ C) :=
  ⟨fun _ _ _ => inferInstance⟩


noncomputable instance : PreservesFiniteColimits (π₃ : _ ⥤ C) :=
  ⟨fun _ _ _ => inferInstance⟩


instance preservesEpimorphisms_π₁ :
    Functor.PreservesEpimorphisms (π₁ : _ ⥤ C) :=
  CategoryTheory.preservesEpimorphisms_of_preservesColimitsOfShape _


instance preservesEpimorphisms_π₂ :
    Functor.PreservesEpimorphisms (π₂ : _ ⥤ C) :=
  CategoryTheory.preservesEpimorphisms_of_preservesColimitsOfShape _


instance preservesEpimorphisms_π₃ :
    Functor.PreservesEpimorphisms (π₃ : _ ⥤ C) :=
  CategoryTheory.preservesEpimorphisms_of_preservesColimitsOfShape _


