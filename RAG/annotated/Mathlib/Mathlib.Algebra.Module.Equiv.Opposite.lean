@[ext high]
theorem LinearMap.ext_ring_op
    {σ : Rᵐᵒᵖ →+* S} {f g : R →ₛₗ[σ] M} (h : f (1 : R) = g (1 : R)) :
    f = g :=
  ext fun x ↦ by
    -- Porting note: replaced the oneliner `rw` proof with a partially term-mode proof
    -- because `rw` was giving "motive is type incorrect" errors
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      σ : RingHom (MulOpposite R) S
      f g : LinearMap σ R M
      h : Eq (f 1) (g 1)
      x : R
      ⊢ Eq (f x) (g x)
    -/
    rw [← one_mul x, ← op_smul_eq_mul]
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      σ : RingHom (MulOpposite R) S
      f g : LinearMap σ R M
      h : Eq (f 1) (g 1)
      x : R
      ⊢ Eq (f (HSMul.hSMul (MulOpposite.op x) 1)) (g (HSMul.hSMul (MulOpposite.op x) …
    -/
    refine (f.map_smulₛₗ (MulOpposite.op x) 1).trans ?_
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      σ : RingHom (MulOpposite R) S
      f g : LinearMap σ R M
      h : Eq (f 1) (g 1)
      x : R
      ⊢ Eq (HSMul.hSMul (σ (MulOpposite.op x)) (f 1)) (g (HSMul.hSMul (MulOpposite.o …
    -/
    rw [h]
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      σ : RingHom (MulOpposite R) S
      f g : LinearMap σ R M
      h : Eq (f 1) (g 1)
      x : R
      ⊢ Eq (HSMul.hSMul (σ (MulOpposite.op x)) (g 1)) (g (HSMul.hSMul (MulOpposite.o …
    -/
    exact (g.map_smulₛₗ (MulOpposite.op x) 1).symm
    /-
      🎉 no goals
    -/


/-- The function `op` is a linear equivalence. -/
def opLinearEquiv : M ≃ₗ[R] Mᵐᵒᵖ :=
  { opAddEquiv with map_smul' := MulOpposite.op_smul }


@[simp]
theorem coe_opLinearEquiv : (opLinearEquiv R : M → Mᵐᵒᵖ) = op :=
  rfl


@[simp]
theorem coe_opLinearEquiv_symm : ((opLinearEquiv R).symm : Mᵐᵒᵖ → M) = unop :=
  rfl


@[simp]
theorem coe_opLinearEquiv_toLinearMap : ((opLinearEquiv R).toLinearMap : M → Mᵐᵒᵖ) = op :=
  rfl


@[simp]
theorem coe_opLinearEquiv_symm_toLinearMap :
    ((opLinearEquiv R).symm.toLinearMap : Mᵐᵒᵖ → M) = unop :=
  rfl

-- Porting note: LHS simplifies; added new simp lemma below @[simp]

theorem opLinearEquiv_toAddEquiv : (opLinearEquiv R : M ≃ₗ[R] Mᵐᵒᵖ).toAddEquiv = opAddEquiv :=
  rfl


@[simp]
theorem coe_opLinearEquiv_addEquiv : ((opLinearEquiv R : M ≃ₗ[R] Mᵐᵒᵖ) : M ≃+ Mᵐᵒᵖ) = opAddEquiv :=
  rfl

-- Porting note: LHS simplifies; added new simp lemma below @[simp]

theorem opLinearEquiv_symm_toAddEquiv :
    (opLinearEquiv R : M ≃ₗ[R] Mᵐᵒᵖ).symm.toAddEquiv = opAddEquiv.symm :=
  rfl


@[simp]
theorem coe_opLinearEquiv_symm_addEquiv :
    ((opLinearEquiv R : M ≃ₗ[R] Mᵐᵒᵖ).symm : Mᵐᵒᵖ ≃+ M) = opAddEquiv.symm :=
  rfl


