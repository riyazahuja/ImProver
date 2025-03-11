/-- Class that expresses that a ring homomorphism is in fact the identity. -/
-- This at first seems not very useful. However we need this when considering
-- modules over some diagram in the category of rings,
-- e.g. when defining presheaves over a presheaf of rings.
-- See `Mathlib.Algebra.Category.ModuleCat.Presheaf`.
class RingHomId {R : Type*} [Semiring R] (σ : R →+* R) : Prop where
  eq_id : σ = RingHom.id R


instance {R : Type*} [Semiring R] : RingHomId (RingHom.id R) where
  eq_id := rfl


/-- Class that expresses the fact that three ring homomorphisms form a composition triple. This is
used to handle composition of semilinear maps. -/
class RingHomCompTriple (σ₁₂ : R₁ →+* R₂) (σ₂₃ : R₂ →+* R₃) (σ₁₃ : outParam (R₁ →+* R₃)) :
  Prop where
  /-- The morphisms form a commutative triangle -/
  comp_eq : σ₂₃.comp σ₁₂ = σ₁₃


@[simp]
theorem comp_apply [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] {x : R₁} : σ₂₃ (σ₁₂ x) = σ₁₃ x :=
  RingHom.congr_fun comp_eq x


/-- Class that expresses the fact that two ring homomorphisms are inverses of each other. This is
used to handle `symm` for semilinear equivalences. -/
class RingHomInvPair (σ : R₁ →+* R₂) (σ' : outParam (R₂ →+* R₁)) : Prop where
  /-- `σ'` is a left inverse of `σ` -/
  comp_eq : σ'.comp σ = RingHom.id R₁
  /-- `σ'` is a left inverse of `σ'` -/
  comp_eq₂ : σ.comp σ' = RingHom.id R₂


theorem comp_apply_eq {x : R₁} : σ' (σ x) = x := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝² : Semiring R₁
    inst✝¹ : Semiring R₂
    σ : RingHom R₁ R₂
    σ' : RingHom R₂ R₁
    inst✝ : RingHomInvPair σ σ'
    x : R₁
    ⊢ Eq (σ' (σ x)) x
  -/
  rw [← RingHom.comp_apply, comp_eq]
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝² : Semiring R₁
    inst✝¹ : Semiring R₂
    σ : RingHom R₁ R₂
    σ' : RingHom R₂ R₁
    inst✝ : RingHomInvPair σ σ'
    x : R₁
    ⊢ Eq ((RingHom.id R₁) x) x
  -/
  simp
  /-
    🎉 no goals
  -/


theorem comp_apply_eq₂ {x : R₂} : σ (σ' x) = x := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝² : Semiring R₁
    inst✝¹ : Semiring R₂
    σ : RingHom R₁ R₂
    σ' : RingHom R₂ R₁
    inst✝ : RingHomInvPair σ σ'
    x : R₂
    ⊢ Eq (σ (σ' x)) x
  -/
  rw [← RingHom.comp_apply, comp_eq₂]
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝² : Semiring R₁
    inst✝¹ : Semiring R₂
    σ : RingHom R₁ R₂
    σ' : RingHom R₂ R₁
    inst✝ : RingHomInvPair σ σ'
    x : R₂
    ⊢ Eq ((RingHom.id R₂) x) x
  -/
  simp
  /-
    🎉 no goals
  -/


instance ids : RingHomInvPair (RingHom.id R₁) (RingHom.id R₁) :=
  ⟨rfl, rfl⟩


instance triples {σ₂₁ : R₂ →+* R₁} [RingHomInvPair σ₁₂ σ₂₁] :
    RingHomCompTriple σ₁₂ σ₂₁ (RingHom.id R₁) :=
      /-
        R₁ : Type u_1
        R₂ : Type u_2
        R₃ : Type u_3
        inst✝⁴ : Semiring R₁
        inst✝³ : Semiring R₂
        inst✝² : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R₁ R₃
        σ : RingHom R₁ R₂
        σ' : RingHom R₂ R₁
        inst✝¹ : RingHomInvPair σ σ'
        σ₂₁ : RingHom R₂ R₁
        inst✝ : RingHomInvPair σ₁₂ σ₂₁
        ⊢ Eq (σ₂₁.comp σ₁₂) (RingHom.id R₁)
      -/
  ⟨by simp only [comp_eq]⟩
      /-
        🎉 no goals
      -/


instance triples₂ {σ₂₁ : R₂ →+* R₁} [RingHomInvPair σ₁₂ σ₂₁] :
    RingHomCompTriple σ₂₁ σ₁₂ (RingHom.id R₂) :=
      /-
        R₁ : Type u_1
        R₂ : Type u_2
        R₃ : Type u_3
        inst✝⁴ : Semiring R₁
        inst✝³ : Semiring R₂
        inst✝² : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R₁ R₃
        σ : RingHom R₁ R₂
        σ' : RingHom R₂ R₁
        inst✝¹ : RingHomInvPair σ σ'
        σ₂₁ : RingHom R₂ R₁
        inst✝ : RingHomInvPair σ₁₂ σ₂₁
        ⊢ Eq (σ₁₂.comp σ₂₁) (RingHom.id R₂)
      -/
  ⟨by simp only [comp_eq₂]⟩
      /-
        🎉 no goals
      -/


/-- Construct a `RingHomInvPair` from both directions of a ring equiv.

This is not an instance, as for equivalences that are involutions, a better instance
would be `RingHomInvPair e e`. Indeed, this declaration is not currently used in mathlib.
-/
theorem of_ringEquiv (e : R₁ ≃+* R₂) : RingHomInvPair (↑e : R₁ →+* R₂) ↑e.symm :=
  ⟨e.symm_toRingHom_comp_toRingHom, e.symm.symm_toRingHom_comp_toRingHom⟩


/--
Swap the direction of a `RingHomInvPair`. This is not an instance as it would loop, and better
instances are often available and may often be preferable to using this one. Indeed, this
declaration is not currently used in mathlib.
-/
theorem symm (σ₁₂ : R₁ →+* R₂) (σ₂₁ : R₂ →+* R₁) [RingHomInvPair σ₁₂ σ₂₁] :
    RingHomInvPair σ₂₁ σ₁₂ :=
  ⟨RingHomInvPair.comp_eq₂, RingHomInvPair.comp_eq⟩


instance ids : RingHomCompTriple (RingHom.id R₁) σ₁₂ σ₁₂ :=
  ⟨by
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝² : Semiring R₁
      inst✝¹ : Semiring R₂
      inst✝ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ : RingHom R₁ R₂
      σ' : RingHom R₂ R₁
      ⊢ Eq (σ₁₂.comp (RingHom.id R₁)) σ₁₂
    -/
    ext
    /-
      case a
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝² : Semiring R₁
      inst✝¹ : Semiring R₂
      inst✝ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ : RingHom R₁ R₂
      σ' : RingHom R₂ R₁
      x✝ : R₁
      ⊢ Eq ((σ₁₂.comp (RingHom.id R₁)) x✝) (σ₁₂ x✝)
    -/
    simp⟩
    /-
      🎉 no goals
    -/


instance right_ids : RingHomCompTriple σ₁₂ (RingHom.id R₂) σ₁₂ :=
  ⟨by
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝² : Semiring R₁
      inst✝¹ : Semiring R₂
      inst✝ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ : RingHom R₁ R₂
      σ' : RingHom R₂ R₁
      ⊢ Eq ((RingHom.id R₂).comp σ₁₂) σ₁₂
    -/
    ext
    /-
      case a
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝² : Semiring R₁
      inst✝¹ : Semiring R₂
      inst✝ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ : RingHom R₁ R₂
      σ' : RingHom R₂ R₁
      x✝ : R₁
      ⊢ Eq (((RingHom.id R₂).comp σ₁₂) x✝) (σ₁₂ x✝)
    -/
    simp⟩
    /-
      🎉 no goals
    -/


/-- Class expressing the fact that a `RingHom` is surjective. This is needed in the context
of semilinear maps, where some lemmas require this. -/
class RingHomSurjective (σ : R₁ →+* R₂) : Prop where
  /-- The ring homomorphism is surjective -/
  is_surjective : Function.Surjective σ


theorem RingHom.surjective (σ : R₁ →+* R₂) [t : RingHomSurjective σ] : Function.Surjective σ :=
  t.is_surjective


instance (priority := 100) invPair {σ₁ : R₁ →+* R₂} {σ₂ : R₂ →+* R₁} [RingHomInvPair σ₁ σ₂] :
    RingHomSurjective σ₁ :=
  ⟨fun x => ⟨σ₂ x, RingHomInvPair.comp_apply_eq₂⟩⟩


instance ids : RingHomSurjective (RingHom.id R₁) :=
  ⟨is_surjective⟩


/-- This cannot be an instance as there is no way to infer `σ₁₂` and `σ₂₃`. -/
theorem comp [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] [RingHomSurjective σ₁₂] [RingHomSurjective σ₂₃] :
    RingHomSurjective σ₁₃ :=
  { is_surjective := by
      /-
        R₁ : Type u_1
        R₂ : Type u_2
        R₃ : Type u_3
        inst✝⁵ : Semiring R₁
        inst✝⁴ : Semiring R₂
        inst✝³ : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R₁ R₃
        inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝¹ : RingHomSurjective σ₁₂
        inst✝ : RingHomSurjective σ₂₃
        ⊢ Function.Surjective ⇑σ₁₃
      -/
      have := σ₂₃.surjective.comp σ₁₂.surjective
      /-
        R₁ : Type u_1
        R₂ : Type u_2
        R₃ : Type u_3
        inst✝⁵ : Semiring R₁
        inst✝⁴ : Semiring R₂
        inst✝³ : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R₁ R₃
        inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝¹ : RingHomSurjective σ₁₂
        inst✝ : RingHomSurjective σ₂₃
        this : Function.Surjective (Function.comp ⇑σ₂₃ ⇑σ₁₂)
        ⊢ Function.Surjective ⇑σ₁₃
      -/
      rwa [← RingHom.coe_comp, RingHomCompTriple.comp_eq] at this }
      /-
        🎉 no goals
      -/


instance (σ : R₁ ≃+* R₂) : RingHomSurjective (σ : R₁ →+* R₂) := ⟨σ.surjective⟩


