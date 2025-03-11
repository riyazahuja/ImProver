/-- The cokernel cofork attached to a commutative square in a preadditive category. -/
noncomputable abbrev cokernelCofork  :
    CokernelCofork (biprod.lift sq.f₁₂ (-sq.f₁₃)) :=
                                                     /-
                                                       C : Type u_1
                                                       inst✝² : CategoryTheory.Category.{?u.233, u_1} C
                                                       inst✝¹ : CategoryTheory.Preadditive C
                                                       sq : CategoryTheory.Square C
                                                       inst✝ : CategoryTheory.Limits.HasBinaryBiproduct sq.X₂ sq.X₃
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift sq …
                                                     -/
  CokernelCofork.ofπ (biprod.desc sq.f₂₄ sq.f₃₄) (by simp [sq.fac])
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- A commutative square in a preadditive category is a pushout square iff
the corresponding diagram `X₁ ⟶ X₂ ⊞ X₃ ⟶ X₄ ⟶ 0` makes `X₄` a cokernel. -/
noncomputable def isPushoutEquivIsColimitCokernelCofork :
    sq.IsPushout ≃ IsColimit sq.cokernelCofork :=
  Equiv.trans
    { toFun := fun h ↦ h.isColimit
      invFun := fun h ↦ IsPushout.mk _ h
      left_inv := fun _ ↦ rfl
      right_inv := fun _ ↦ Subsingleton.elim _ _ }
    sq.commSq.isColimitEquivIsColimitCokernelCofork


variable {sq} in
/-- The colimit cokernel cofork attached to a pushout square. -/
noncomputable def IsPushout.isColimitCokernelCofork (h : sq.IsPushout) :
    IsColimit sq.cokernelCofork :=
  h.isColimitEquivIsColimitCokernelCofork h.isColimit


/-- The kernel fork attached to a commutative square in a preadditive category. -/
noncomputable abbrev kernelFork  :
    KernelFork (biprod.desc sq.f₂₄ (-sq.f₃₄)) :=
                                                 /-
                                                   C : Type u_1
                                                   inst✝² : CategoryTheory.Category.{?u.4335, u_1} C
                                                   inst✝¹ : CategoryTheory.Preadditive C
                                                   sq : CategoryTheory.Square C
                                                   inst✝ : CategoryTheory.Limits.HasBinaryBiproduct sq.X₂ sq.X₃
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift sq …
                                                 -/
  KernelFork.ofι (biprod.lift sq.f₁₂ sq.f₁₃) (by simp [sq.fac])
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A commutative square in a preadditive category is a pullback square iff
the corresponding diagram `0 ⟶ X₁ ⟶ X₂ ⊞ X₃ ⟶ X₄ ⟶ 0` makes `X₁` a kernel. -/
noncomputable def isPullbackEquivIsLimitKernelFork :
    sq.IsPullback ≃ IsLimit sq.kernelFork :=
  Equiv.trans
    { toFun := fun h ↦ h.isLimit
      invFun := fun h ↦ IsPullback.mk _ h
      left_inv := fun _ ↦ rfl
      right_inv := fun _ ↦ Subsingleton.elim _ _ }
    sq.commSq.isLimitEquivIsLimitKernelFork


variable {sq} in
/-- The limit kernel fork attached to a pullback square. -/
noncomputable def IsPullback.isLimitKernelFork (h : sq.IsPullback) :
    IsLimit sq.kernelFork :=
  h.isLimitEquivIsLimitKernelFork h.isLimit


