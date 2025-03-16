theorem upperBounds_image {s : Set α} : upperBounds (f '' s) = f '' upperBounds s :=
  Subset.antisymm
    (fun x hx =>
      ⟨f.symm x, fun _ hy => f.le_symm_apply.2 (hx <| mem_image_of_mem _ hy), f.apply_symm_apply x⟩)
    f.monotone.image_upperBounds_subset_upperBounds_image


theorem lowerBounds_image {s : Set α} : lowerBounds (f '' s) = f '' lowerBounds s :=
  @upperBounds_image αᵒᵈ βᵒᵈ _ _ f.dual _

-- Porting note: by simps were `fun _ _ => f.le_iff_le` and `fun _ _ => f.symm.le_iff_le`

@[simp]
theorem isLUB_image {s : Set α} {x : β} : IsLUB (f '' s) x ↔ IsLUB s (f.symm x) :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 inst✝¹ : Preorder α
                                 inst✝ : Preorder β
                                 f : OrderIso α β
                                 s : Set α
                                 x : β
                                 h : IsLUB (Set.image (⇑f) s) x
                                 ⊢ ∀ {x y : α}, Iff (LE.le (f x) (f y)) (LE.le x y)
                               -/
  ⟨fun h => IsLUB.of_image (by simp) ((f.apply_symm_apply x).symm ▸ h), fun h =>
                               /-
                                 🎉 no goals
                               -/
                        /-
                          α : Type u_1
                          β : Type u_2
                          inst✝¹ : Preorder α
                          inst✝ : Preorder β
                          f : OrderIso α β
                          s : Set α
                          x : β
                          h : IsLUB s (f.symm x)
                          ⊢ ∀ {x y : β}, Iff (LE.le (f.symm x) (f.symm y)) (LE.le x y)
                        -/
    (IsLUB.of_image (by simp)) <| (f.symm_image_image s).symm ▸ h⟩
                        /-
                          🎉 no goals
                        -/


theorem isLUB_image' {s : Set α} {x : α} : IsLUB (f '' s) (f x) ↔ IsLUB s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    s : Set α
    x : α
    ⊢ Iff (IsLUB (Set.image (⇑f) s) (f x)) (IsLUB s x)
  -/
  rw [isLUB_image, f.symm_apply_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem isGLB_image {s : Set α} {x : β} : IsGLB (f '' s) x ↔ IsGLB s (f.symm x) :=
  f.dual.isLUB_image


theorem isGLB_image' {s : Set α} {x : α} : IsGLB (f '' s) (f x) ↔ IsGLB s x :=
  f.dual.isLUB_image'


@[simp]
theorem isLUB_preimage {s : Set β} {x : α} : IsLUB (f ⁻¹' s) x ↔ IsLUB s (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    s : Set β
    x : α
    ⊢ Iff (IsLUB (Set.preimage (⇑f) s) x) (IsLUB s (f x))
  -/
  rw [← f.symm_symm, ← image_eq_preimage, isLUB_image]
  /-
    🎉 no goals
  -/


theorem isLUB_preimage' {s : Set β} {x : β} : IsLUB (f ⁻¹' s) (f.symm x) ↔ IsLUB s x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderIso α β
    s : Set β
    x : β
    ⊢ Iff (IsLUB (Set.preimage (⇑f) s) (f.symm x)) (IsLUB s x)
  -/
  rw [isLUB_preimage, f.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem isGLB_preimage {s : Set β} {x : α} : IsGLB (f ⁻¹' s) x ↔ IsGLB s (f x) :=
  f.dual.isLUB_preimage


theorem isGLB_preimage' {s : Set β} {x : β} : IsGLB (f ⁻¹' s) (f.symm x) ↔ IsGLB s x :=
  f.dual.isLUB_preimage'


