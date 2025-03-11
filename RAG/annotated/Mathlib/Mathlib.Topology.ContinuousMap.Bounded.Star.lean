instance instStarAddMonoid : StarAddMonoid (α →ᵇ β) where
  star f := f.comp star starNormedAddGroupHom.lipschitz
  star_involutive f := ext fun x => star_star (f x)
  star_add f g := ext fun x => star_add (f x) (g x)


/-- The right-hand side of this equality can be parsed `star ∘ ⇑f` because of the
instance `Pi.instStarForAll`. Upon inspecting the goal, one sees `⊢ ↑(star f) = star ↑f`. -/
@[simp]
theorem coe_star (f : α →ᵇ β) : ⇑(star f) = star (⇑f) := rfl


@[simp]
theorem star_apply (f : α →ᵇ β) (x : α) : star f x = star (f x) := rfl


instance instNormedStarGroup : NormedStarGroup (α →ᵇ β) where
                    /-
                      F : Type u_1
                      α : Type u
                      β : Type v
                      γ : Type w
                      𝕜 : Type u_2
                      inst✝⁷ : NormedField 𝕜
                      inst✝⁶ : StarRing 𝕜
                      inst✝⁵ : TopologicalSpace α
                      inst✝⁴ : SeminormedAddCommGroup β
                      inst✝³ : StarAddMonoid β
                      inst✝² : NormedStarGroup β
                      inst✝¹ : NormedSpace 𝕜 β
                      inst✝ : StarModule 𝕜 β
                      f : BoundedContinuousFunction α β
                      ⊢ Eq (Norm.norm (Star.star f)) (Norm.norm f)
                    -/
  norm_star f := by simp only [norm_eq, star_apply, norm_star]
                    /-
                      🎉 no goals
                    -/


instance instStarModule : StarModule 𝕜 (α →ᵇ β) where
  star_smul k f := ext fun x => star_smul k (f x)


instance instStarRing [NormedStarGroup β] : StarRing (α →ᵇ β) where
  __ := instStarAddMonoid
  star_mul f g := ext fun x ↦ star_mul (f x) (g x)


instance instCStarRing : CStarRing (α →ᵇ β) where
  norm_mul_self_le f := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : BoundedContinuousFunction α β
      ⊢ LE.le (HMul.hMul (Norm.norm f) (Norm.norm f)) (Norm.norm (HMul.hMul (Star.st …
    -/
    rw [← sq, ← Real.le_sqrt (norm_nonneg _) (norm_nonneg _), norm_le (Real.sqrt_nonneg _)]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : BoundedContinuousFunction α β
      ⊢ ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm (HMul.hMul (Star.star f) f)).s …
    -/
    intro x
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (Norm.norm (f x)) (Norm.norm (HMul.hMul (Star.star f) f)).sqrt
    -/
    rw [Real.le_sqrt (norm_nonneg _) (norm_nonneg _), sq, ← CStarRing.norm_star_mul_self]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (Norm.norm (HMul.hMul (Star.star (f x)) (f x))) (Norm.norm (HMul.hMul  …
    -/
    exact norm_coe_le_norm (star f * f) x
    /-
      🎉 no goals
    -/


