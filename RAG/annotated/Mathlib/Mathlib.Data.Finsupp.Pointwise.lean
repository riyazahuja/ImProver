/-- The product of `f g : α →₀ β` is the finitely supported function
  whose value at `a` is `f a * g a`. -/
instance : Mul (α →₀ β) :=
  ⟨zipWith (· * ·) (mul_zero 0)⟩


theorem coe_mul (g₁ g₂ : α →₀ β) : ⇑(g₁ * g₂) = g₁ * g₂ :=
  rfl


@[simp]
theorem mul_apply {g₁ g₂ : α →₀ β} {a : α} : (g₁ * g₂) a = g₁ a * g₂ a :=
  rfl


@[simp]
theorem single_mul (a : α) (b₁ b₂ : β) : single a (b₁ * b₂) = single a b₁ * single a b₂ :=
  (zipWith_single_single _ _ _ _ _).symm


theorem support_mul [DecidableEq α] {g₁ g₂ : α →₀ β} :
    (g₁ * g₂).support ⊆ g₁.support ∩ g₂.support := by
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    ⊢ HasSubset.Subset (HMul.hMul g₁ g₂).support (Inter.inter g₁.support g₂.support)
  -/
  intro a h
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    a : α
    h : Membership.mem (HMul.hMul g₁ g₂).support a
    ⊢ Membership.mem (Inter.inter g₁.support g₂.support) a
  -/
  simp only [mul_apply, mem_support_iff] at h
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    a : α
    h : Ne (HMul.hMul (g₁ a) (g₂ a)) 0
    ⊢ Membership.mem (Inter.inter g₁.support g₂.support) a
  -/
  simp only [mem_support_iff, mem_inter, Ne]
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    a : α
    h : Ne (HMul.hMul (g₁ a) (g₂ a)) 0
    ⊢ And (Not (Eq (g₁ a) 0)) (Not (Eq (g₂ a) 0))
  -/
  rw [← not_or]
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    a : α
    h : Ne (HMul.hMul (g₁ a) (g₂ a)) 0
    ⊢ Not (Or (Eq (g₁ a) 0) (Eq (g₂ a) 0))
  -/
  intro w
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    a : α
    h : Ne (HMul.hMul (g₁ a) (g₂ a)) 0
    w : Or (Eq (g₁ a) 0) (Eq (g₂ a) 0)
    ⊢ False
  -/
  apply h
  /-
    α : Type u₁
    β : Type u₂
    inst✝¹ : MulZeroClass β
    inst✝ : DecidableEq α
    g₁ g₂ : Finsupp α β
    a : α
    h : Ne (HMul.hMul (g₁ a) (g₂ a)) 0
    w : Or (Eq (g₁ a) 0) (Eq (g₂ a) 0)
    ⊢ Eq (HMul.hMul (g₁ a) (g₂ a)) 0
  -/
                                 /-
                                   🎉 no goals
                                 -/
  cases' w with w w <;> (rw [w]; simp)
                                 /-
                                   🎉 no goals
                                 -/


instance : MulZeroClass (α →₀ β) :=
  DFunLike.coe_injective.mulZeroClass _ coe_zero coe_mul


instance [SemigroupWithZero β] : SemigroupWithZero (α →₀ β) :=
  DFunLike.coe_injective.semigroupWithZero _ coe_zero coe_mul


instance [NonUnitalNonAssocSemiring β] : NonUnitalNonAssocSemiring (α →₀ β) :=
  DFunLike.coe_injective.nonUnitalNonAssocSemiring _ coe_zero coe_add coe_mul fun _ _ ↦ rfl


instance [NonUnitalSemiring β] : NonUnitalSemiring (α →₀ β) :=
  DFunLike.coe_injective.nonUnitalSemiring _ coe_zero coe_add coe_mul fun _ _ ↦ rfl


instance [NonUnitalCommSemiring β] : NonUnitalCommSemiring (α →₀ β) :=
  DFunLike.coe_injective.nonUnitalCommSemiring _ coe_zero coe_add coe_mul fun _ _ ↦ rfl


instance [NonUnitalNonAssocRing β] : NonUnitalNonAssocRing (α →₀ β) :=
  DFunLike.coe_injective.nonUnitalNonAssocRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ ↦ rfl) fun _ _ ↦ rfl


instance [NonUnitalRing β] : NonUnitalRing (α →₀ β) :=
  DFunLike.coe_injective.nonUnitalRing _ coe_zero coe_add coe_mul coe_neg coe_sub (fun _ _ ↦ rfl)
    fun _ _ ↦ rfl


instance [NonUnitalCommRing β] : NonUnitalCommRing (α →₀ β) :=
  DFunLike.coe_injective.nonUnitalCommRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ ↦ rfl) fun _ _ ↦ rfl

-- TODO can this be generalized in the direction of `Pi.smul'`
-- (i.e. dependent functions and finsupps)
-- TODO in theory this could be generalised, we only really need `smul_zero` for the definition

instance pointwiseScalar [Semiring β] : SMul (α → β) (α →₀ β) where
  smul f g :=
    Finsupp.ofSupportFinite (fun a ↦ f a • g a) (by
      /-
        α : Type u₁
        β : Type u₂
        γ : Type u₃
        δ : Type u₄
        ι : Type u₅
        inst✝ : Semiring β
        f : α → β
        g : Finsupp α β
        ⊢ (Function.support fun a => HSMul.hSMul (f a) (g a)).Finite
      -/
      apply Set.Finite.subset g.finite_support
      simp only [Function.support_subset_iff, Finsupp.mem_support_iff, Ne,
        Finsupp.fun_support_eq, Finset.mem_coe]
      /-
        α : Type u₁
        β : Type u₂
        γ : Type u₃
        δ : Type u₄
        ι : Type u₅
        inst✝ : Semiring β
        f : α → β
        g : Finsupp α β
        ⊢ ∀ (x : α), Not (Eq (HSMul.hSMul (f x) (g x)) 0) → Not (Eq (g x) 0)
      -/
      intro x hx h
      /-
        α : Type u₁
        β : Type u₂
        γ : Type u₃
        δ : Type u₄
        ι : Type u₅
        inst✝ : Semiring β
        f : α → β
        g : Finsupp α β
        x : α
        hx : Not (Eq (HSMul.hSMul (f x) (g x)) 0)
        h : Eq (g x) 0
        ⊢ False
      -/
      apply hx
      /-
        α : Type u₁
        β : Type u₂
        γ : Type u₃
        δ : Type u₄
        ι : Type u₅
        inst✝ : Semiring β
        f : α → β
        g : Finsupp α β
        x : α
        hx : Not (Eq (HSMul.hSMul (f x) (g x)) 0)
        h : Eq (g x) 0
        ⊢ Eq (HSMul.hSMul (f x) (g x)) 0
      -/
      rw [h, smul_zero])
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_pointwise_smul [Semiring β] (f : α → β) (g : α →₀ β) : ⇑(f • g) = f • ⇑g :=
  rfl


/-- The pointwise multiplicative action of functions on finitely supported functions -/
instance pointwiseModule [Semiring β] : Module (α → β) (α →₀ β) :=
  Function.Injective.module _ coeFnAddHom DFunLike.coe_injective coe_pointwise_smul


