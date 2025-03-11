/-- The cokernel cofork attached to a commutative square in a preadditive category. -/
noncomputable abbrev CommSq.cokernelCofork (sq : CommSq f g inl inr) :
    CokernelCofork (biprod.lift f (-g)) :=
                                               /-
                                                 C : Type u_1
                                                 inst✝² : CategoryTheory.Category.{?u.591, u_1} C
                                                 inst✝¹ : CategoryTheory.Preadditive C
                                                 X₁ X₂ X₃ X₄ : C
                                                 inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
                                                 f : Quiver.Hom X₁ X₂
                                                 g : Quiver.Hom X₁ X₃
                                                 inl : Quiver.Hom X₂ X₄
                                                 inr : Quiver.Hom X₃ X₄
                                                 sq : CategoryTheory.CommSq f g inl inr
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift f  …
                                               -/
  CokernelCofork.ofπ (biprod.desc inl inr) (by simp [sq.w])
                                               /-
                                                 🎉 no goals
                                               -/


/-- A commutative square in a preadditive category is a pushout square iff
the corresponding diagram `X₁ ⟶ X₂ ⊞ X₃ ⟶ X₄ ⟶ 0` makes `X₄` a cokernel. -/
noncomputable def CommSq.isColimitEquivIsColimitCokernelCofork (sq : CommSq f g inl inr) :
    IsColimit (PushoutCocone.mk _ _ sq.w) ≃ IsColimit sq.cokernelCofork where
  toFun h :=
    Cofork.IsColimit.mk _
      (fun s ↦ PushoutCocone.IsColimit.desc h
        (biprod.inl ≫ s.π) (biprod.inr ≫ s.π) (by
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
          -/
          rw [← sub_eq_zero, ← assoc, ← assoc, ← Preadditive.sub_comp]
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
          -/
                                  /-
                                    🎉 no goals
                                  -/
          convert s.condition <;> aesop_cat))
                                  /-
                                    🎉 no goals
                                  -/
      (fun s ↦ by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          f : Quiver.Hom X₁ X₂
          g : Quiver.Hom X₁ X₃
          inl : Quiver.Hom X₂ X₄
          inr : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq f g inl inr
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
          s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π sq.co …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          f : Quiver.Hom X₁ X₂
          g : Quiver.Hom X₁ X₃
          inl : Quiver.Hom X₂ X₄
          inr : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq f g inl inr
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
          s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.desc in …
        -/
        ext
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (Cat …
          -/
        · simp only [biprod.inl_desc_assoc]
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inl (CategoryTheory.Limits.PushoutCoc …
          -/
          apply PushoutCocone.IsColimit.inl_desc h
          /-
            🎉 no goals
          -/
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
          -/
        · simp only [biprod.inr_desc_assoc]
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inr (CategoryTheory.Limits.PushoutCoc …
          -/
          apply PushoutCocone.IsColimit.inr_desc h)
          /-
            🎉 no goals
          -/
      (fun s m hm ↦ by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          f : Quiver.Hom X₁ X₂
          g : Quiver.Hom X₁ X₃
          inl : Quiver.Hom X₂ X₄
          inr : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq f g inl inr
          h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
          s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
          m : Quiver.Hom sq.cokernelCofork.pt s.pt
          hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π sq …
          ⊢ Eq m ((fun s => CategoryTheory.Limits.PushoutCocone.IsColimit.desc h (Catego …
        -/
        apply PushoutCocone.IsColimit.hom_ext h
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π sq …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
          -/
        · replace hm := biprod.inl ≫= hm
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
          -/
          dsimp at hm ⊢
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inl m) (CategoryTheory.CategoryStruct …
          -/
          simp only [biprod.inl_desc_assoc] at hm
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp inl m) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inl m) (CategoryTheory.CategoryStruct …
          -/
          rw [hm]
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp inl m) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl s.π) …
          -/
          symm
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp inl m) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inl (CategoryTheory.Limits.PushoutCoc …
          -/
          apply PushoutCocone.IsColimit.inl_desc h
          /-
            🎉 no goals
          -/
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π sq …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
          -/
        · replace hm := biprod.inr ≫= hm
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr ( …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
          -/
          dsimp at hm ⊢
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr ( …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inr m) (CategoryTheory.CategoryStruct …
          -/
          simp only [biprod.inr_desc_assoc] at hm
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp inr m) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inr m) (CategoryTheory.CategoryStruct …
          -/
          rw [hm]
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp inr m) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr s.π) …
          -/
          symm
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            f : Quiver.Hom X₁ X₂
            g : Quiver.Hom X₁ X₃
            inl : Quiver.Hom X₂ X₄
            inr : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq f g inl inr
            h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk in …
            s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
            m : Quiver.Hom sq.cokernelCofork.pt s.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp inr m) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp inr (CategoryTheory.Limits.PushoutCoc …
          -/
          apply PushoutCocone.IsColimit.inr_desc h)
          /-
            🎉 no goals
          -/
  invFun h :=
    PushoutCocone.IsColimit.mk _
      (fun s ↦ h.desc (CokernelCofork.ofπ (biprod.desc s.inl s.inr)
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
                inst✝¹ : CategoryTheory.Preadditive C
                X₁ X₂ X₃ X₄ : C
                inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
                f : Quiver.Hom X₁ X₂
                g : Quiver.Hom X₁ X₃
                inl : Quiver.Hom X₂ X₄
                inr : Quiver.Hom X₃ X₄
                sq : CategoryTheory.CommSq f g inl inr
                h : CategoryTheory.Limits.IsColimit sq.cokernelCofork
                s : CategoryTheory.Limits.PushoutCocone f g
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift f  …
              -/
          (by simp [s.condition])))
              /-
                🎉 no goals
              -/
      (fun s ↦ by simpa using biprod.inl ≫=
                h.fac (CokernelCofork.ofπ (biprod.desc s.inl s.inr)
                  (by simp [s.condition])) .one)
      (fun s ↦ by simpa using biprod.inr ≫=
                h.fac (CokernelCofork.ofπ (biprod.desc s.inl s.inr)
                  (by simp [s.condition])) .one)
      (fun s m hm₁ hm₂ ↦ by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          f : Quiver.Hom X₁ X₂
          g : Quiver.Hom X₁ X₃
          inl : Quiver.Hom X₂ X₄
          inr : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq f g inl inr
          h : CategoryTheory.Limits.IsColimit sq.cokernelCofork
          s : CategoryTheory.Limits.PushoutCocone f g
          m : Quiver.Hom X₄ s.pt
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp inl m) s.inl
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp inr m) s.inr
          ⊢ Eq m ((fun s => h.desc (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTh …
        -/
        apply Cofork.IsColimit.hom_ext h
        convert (h.fac (CokernelCofork.ofπ (biprod.desc s.inl s.inr)
          (by simp [s.condition])) .one).symm
        /-
          case h.e'_2.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.2811, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          f : Quiver.Hom X₁ X₂
          g : Quiver.Hom X₁ X₃
          inl : Quiver.Hom X₂ X₄
          inr : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq f g inl inr
          h : CategoryTheory.Limits.IsColimit sq.cokernelCofork
          s : CategoryTheory.Limits.PushoutCocone f g
          m : Quiver.Hom X₄ s.pt
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp inl m) s.inl
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp inr m) s.inr
          e_1✝ : Eq (Quiver.Hom ((CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π sq.co …
        -/
        aesop_cat)
        /-
          🎉 no goals
        -/
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- The colimit cokernel cofork attached to a pushout square. -/
noncomputable def IsPushout.isColimitCokernelCofork (h : IsPushout f g inl inr) :
    IsColimit h.cokernelCofork :=
  h.isColimitEquivIsColimitCokernelCofork h.isColimit


/-- The kernel fork attached to a commutative square in a preadditive category. -/
noncomputable abbrev CommSq.kernelFork (sq : CommSq fst snd f g) :
    KernelFork (biprod.desc f (-g)) :=
                                           /-
                                             C : Type u_1
                                             inst✝² : CategoryTheory.Category.{?u.50012, u_1} C
                                             inst✝¹ : CategoryTheory.Preadditive C
                                             X₁ X₂ X₃ X₄ : C
                                             inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
                                             fst : Quiver.Hom X₁ X₂
                                             snd : Quiver.Hom X₁ X₃
                                             f : Quiver.Hom X₂ X₄
                                             g : Quiver.Hom X₃ X₄
                                             sq : CategoryTheory.CommSq fst snd f g
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift fs …
                                           -/
  KernelFork.ofι (biprod.lift fst snd) (by simp [sq.w])
                                           /-
                                             🎉 no goals
                                           -/


/-- A commutative square in a preadditive category is a pullback square iff
the corresponding diagram `0 ⟶ X₁ ⟶ X₂ ⊞ X₃ ⟶ X₄ ⟶ 0` makes `X₁` a kernel. -/
noncomputable def CommSq.isLimitEquivIsLimitKernelFork (sq : CommSq fst snd f g) :
    IsLimit (PullbackCone.mk _ _ sq.w) ≃ IsLimit sq.kernelFork where
  toFun h :=
    Fork.IsLimit.mk _
      (fun s ↦ PullbackCone.IsLimit.lift h
        (s.ι ≫ biprod.fst) (s.ι ≫ biprod.snd) (by
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
          -/
          rw [← sub_eq_zero, assoc, assoc, ← Preadditive.comp_sub]
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι (HSub.hSub (CategoryTheory.Catego …
          -/
                                  /-
                                    🎉 no goals
                                  -/
          convert s.condition <;> aesop_cat))
                                  /-
                                    🎉 no goals
                                  -/
      (fun s ↦ by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          fst : Quiver.Hom X₁ X₂
          snd : Quiver.Hom X₁ X₃
          f : Quiver.Hom X₂ X₄
          g : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq fst snd f g
          h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
          s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.Pull …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          fst : Quiver.Hom X₁ X₂
          snd : Quiver.Hom X₁ X₃
          f : Quiver.Hom X₂ X₄
          g : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq fst snd f g
          h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
          s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackCone.I …
        -/
        ext
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp only [assoc, biprod.lift_fst]
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackCone.I …
          -/
          apply PullbackCone.IsLimit.lift_fst h
          /-
            🎉 no goals
          -/
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp only [assoc, biprod.lift_snd]
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackCone.I …
          -/
          apply PullbackCone.IsLimit.lift_snd h)
          /-
            🎉 no goals
          -/
      (fun s m hm ↦ by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          fst : Quiver.Hom X₁ X₂
          snd : Quiver.Hom X₁ X₃
          f : Quiver.Hom X₂ X₄
          g : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq fst snd f g
          h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
          s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
          m : Quiver.Hom s.pt sq.kernelFork.pt
          hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι sq …
          ⊢ Eq m ((fun s => CategoryTheory.Limits.PullbackCone.IsLimit.lift h (CategoryT …
        -/
        apply PullbackCone.IsLimit.hom_ext h
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι sq …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackCone …
          -/
        · replace hm := hm =≫ biprod.fst
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackCone …
          -/
          dsimp at hm ⊢
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m fst) (CategoryTheory.CategoryStruct …
          -/
          simp only [assoc, biprod.lift_fst] at hm
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m fst) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m fst) (CategoryTheory.CategoryStruct …
          -/
          rw [hm]
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m fst) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι CategoryTheory.Limits.biprod.fst) …
          -/
          symm
          /-
            case h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m fst) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackCone.I …
          -/
          apply PullbackCone.IsLimit.lift_fst h
          /-
            🎉 no goals
          -/
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι sq …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackCone …
          -/
        · replace hm := hm =≫ biprod.snd
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackCone …
          -/
          dsimp at hm ⊢
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m snd) (CategoryTheory.CategoryStruct …
          -/
          simp only [assoc, biprod.lift_snd] at hm
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m snd) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m snd) (CategoryTheory.CategoryStruct …
          -/
          rw [hm]
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m snd) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι CategoryTheory.Limits.biprod.snd) …
          -/
          symm
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            X₁ X₂ X₃ X₄ : C
            inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
            fst : Quiver.Hom X₁ X₂
            snd : Quiver.Hom X₁ X₃
            f : Quiver.Hom X₂ X₄
            g : Quiver.Hom X₃ X₄
            sq : CategoryTheory.CommSq fst snd f g
            h : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk fst s …
            s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
            m : Quiver.Hom s.pt sq.kernelFork.pt
            hm : Eq (CategoryTheory.CategoryStruct.comp m snd) (CategoryTheory.CategoryStr …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PullbackCone.I …
          -/
          apply PullbackCone.IsLimit.lift_snd h)
          /-
            🎉 no goals
          -/
  invFun h :=
    PullbackCone.IsLimit.mk _
      (fun s ↦ h.lift (KernelFork.ofι (biprod.lift s.fst s.snd)
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
                inst✝¹ : CategoryTheory.Preadditive C
                X₁ X₂ X₃ X₄ : C
                inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
                fst : Quiver.Hom X₁ X₂
                snd : Quiver.Hom X₁ X₃
                f : Quiver.Hom X₂ X₄
                g : Quiver.Hom X₃ X₄
                sq : CategoryTheory.CommSq fst snd f g
                h : CategoryTheory.Limits.IsLimit sq.kernelFork
                s : CategoryTheory.Limits.PullbackCone f g
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift s. …
              -/
          (by simp [s.condition])))
              /-
                🎉 no goals
              -/
      (fun s ↦ by simpa using h.fac (KernelFork.ofι (biprod.lift s.fst s.snd)
        (by simp [s.condition])) .zero =≫ biprod.fst)
      (fun s ↦ by simpa using h.fac (KernelFork.ofι (biprod.lift s.fst s.snd)
        (by simp [s.condition])) .zero =≫ biprod.snd)
      (fun s m hm₁ hm₂ ↦ by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          fst : Quiver.Hom X₁ X₂
          snd : Quiver.Hom X₁ X₃
          f : Quiver.Hom X₂ X₄
          g : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq fst snd f g
          h : CategoryTheory.Limits.IsLimit sq.kernelFork
          s : CategoryTheory.Limits.PullbackCone f g
          m : Quiver.Hom s.pt X₁
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp m fst) s.fst
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp m snd) s.snd
          ⊢ Eq m ((fun s => h.lift (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory …
        -/
        apply Fork.IsLimit.hom_ext h
        convert (h.fac (KernelFork.ofι (biprod.lift s.fst s.snd)
          (by simp [s.condition])) .zero).symm
        /-
          case h.e'_2.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.52227, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          X₁ X₂ X₃ X₄ : C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₂ X₃
          fst : Quiver.Hom X₁ X₂
          snd : Quiver.Hom X₁ X₃
          f : Quiver.Hom X₂ X₄
          g : Quiver.Hom X₃ X₄
          sq : CategoryTheory.CommSq fst snd f g
          h : CategoryTheory.Limits.IsLimit sq.kernelFork
          s : CategoryTheory.Limits.PullbackCone f g
          m : Quiver.Hom s.pt X₁
          hm₁ : Eq (CategoryTheory.CategoryStruct.comp m fst) s.fst
          hm₂ : Eq (CategoryTheory.CategoryStruct.comp m snd) s.snd
          e_1✝ : Eq (Quiver.Hom s.pt ((CategoryTheory.Limits.parallelPair (CategoryTheor …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι sq.ke …
        -/
        aesop_cat)
        /-
          🎉 no goals
        -/
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- The limit kernel fork attached to a pullback square. -/
noncomputable def IsPullback.isLimitKernelFork (h : IsPullback fst snd f g) :
    IsLimit h.kernelFork :=
  h.isLimitEquivIsLimitKernelFork h.isLimit


