@[to_additive Sum.hasVAdd]
instance : SMul M (α ⊕ β) :=
  ⟨fun a => Sum.map (a • ·) (a • ·)⟩


@[to_additive]
theorem smul_def : a • x = x.map (a • ·) (a • ·) :=
  rfl


@[to_additive (attr := simp)]
theorem smul_inl : a • (inl b : α ⊕ β) = inl (a • b) :=
  rfl


@[to_additive (attr := simp)]
theorem smul_inr : a • (inr c : α ⊕ β) = inr (a • c) :=
  rfl


@[to_additive (attr := simp)]
                                                    /-
                                                      M : Type u_1
                                                      α : Type u_3
                                                      β : Type u_4
                                                      inst✝¹ : SMul M α
                                                      inst✝ : SMul M β
                                                      a : M
                                                      x : Sum α β
                                                      ⊢ Eq (HSMul.hSMul a x).swap (HSMul.hSMul a x.swap)
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem smul_swap : (a • x).swap = a • x.swap := by cases x <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


instance [SMul M N] [IsScalarTower M N α] [IsScalarTower M N β] : IsScalarTower M N (α ⊕ β) :=
  ⟨fun a b x => by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁶ : SMul M α
      inst✝⁵ : SMul M β
      inst✝⁴ : SMul N α
      inst✝³ : SMul N β
      a✝ : M
      b✝ : α
      c : β
      x✝ : Sum α β
      inst✝² : SMul M N
      inst✝¹ : IsScalarTower M N α
      inst✝ : IsScalarTower M N β
      a : M
      b : N
      x : Sum α β
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) x) (HSMul.hSMul a (HSMul.hSMul b x))
    -/
    cases x
    /-
      case inl
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁶ : SMul M α
      inst✝⁵ : SMul M β
      inst✝⁴ : SMul N α
      inst✝³ : SMul N β
      a✝ : M
      b✝ : α
      c : β
      x : Sum α β
      inst✝² : SMul M N
      inst✝¹ : IsScalarTower M N α
      inst✝ : IsScalarTower M N β
      a : M
      b : N
      val✝ : α
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) (Sum.inl val✝)) (HSMul.hSMul a (HSMul.hSMu …
    -/
    exacts [congr_arg inl (smul_assoc _ _ _), congr_arg inr (smul_assoc _ _ _)]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMulCommClass M N α] [SMulCommClass M N β] : SMulCommClass M N (α ⊕ β) :=
  ⟨fun a b x => by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : SMul M α
      inst✝⁴ : SMul M β
      inst✝³ : SMul N α
      inst✝² : SMul N β
      a✝ : M
      b✝ : α
      c : β
      x✝ : Sum α β
      inst✝¹ : SMulCommClass M N α
      inst✝ : SMulCommClass M N β
      a : M
      b : N
      x : Sum α β
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul b x)) (HSMul.hSMul b (HSMul.hSMul a x))
    -/
    cases x
    /-
      case inl
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : SMul M α
      inst✝⁴ : SMul M β
      inst✝³ : SMul N α
      inst✝² : SMul N β
      a✝ : M
      b✝ : α
      c : β
      x : Sum α β
      inst✝¹ : SMulCommClass M N α
      inst✝ : SMulCommClass M N β
      a : M
      b : N
      val✝ : α
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul b (Sum.inl val✝))) (HSMul.hSMul b (HSMul.hSMu …
    -/
    exacts [congr_arg inl (smul_comm _ _ _), congr_arg inr (smul_comm _ _ _)]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMul Mᵐᵒᵖ α] [SMul Mᵐᵒᵖ β] [IsCentralScalar M α] [IsCentralScalar M β] :
    IsCentralScalar M (α ⊕ β) :=
  ⟨fun a x => by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁷ : SMul M α
      inst✝⁶ : SMul M β
      inst✝⁵ : SMul N α
      inst✝⁴ : SMul N β
      a✝ : M
      b : α
      c : β
      x✝ : Sum α β
      inst✝³ : SMul (MulOpposite M) α
      inst✝² : SMul (MulOpposite M) β
      inst✝¹ : IsCentralScalar M α
      inst✝ : IsCentralScalar M β
      a : M
      x : Sum α β
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) x) (HSMul.hSMul a x)
    -/
    cases x
    /-
      case inl
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁷ : SMul M α
      inst✝⁶ : SMul M β
      inst✝⁵ : SMul N α
      inst✝⁴ : SMul N β
      a✝ : M
      b : α
      c : β
      x : Sum α β
      inst✝³ : SMul (MulOpposite M) α
      inst✝² : SMul (MulOpposite M) β
      inst✝¹ : IsCentralScalar M α
      inst✝ : IsCentralScalar M β
      a : M
      val✝ : α
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) (Sum.inl val✝)) (HSMul.hSMul a (Sum.inl v …
    -/
    exacts [congr_arg inl (op_smul_eq_smul _ _), congr_arg inr (op_smul_eq_smul _ _)]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance FaithfulSMulLeft [FaithfulSMul M α] : FaithfulSMul M (α ⊕ β) :=
                                               /-
                                                 M : Type u_1
                                                 N : Type u_2
                                                 α : Type u_3
                                                 β : Type u_4
                                                 inst✝⁴ : SMul M α
                                                 inst✝³ : SMul M β
                                                 inst✝² : SMul N α
                                                 inst✝¹ : SMul N β
                                                 a✝ : M
                                                 b : α
                                                 c : β
                                                 x : Sum α β
                                                 inst✝ : FaithfulSMul M α
                                                 m₁✝ m₂✝ : M
                                                 h : ∀ (a : Sum α β), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                                 a : α
                                                 ⊢ Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                               -/
  ⟨fun h => eq_of_smul_eq_smul fun a : α => by injection h (inl a)⟩
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
instance FaithfulSMulRight [FaithfulSMul M β] : FaithfulSMul M (α ⊕ β) :=
                                               /-
                                                 M : Type u_1
                                                 N : Type u_2
                                                 α : Type u_3
                                                 β : Type u_4
                                                 inst✝⁴ : SMul M α
                                                 inst✝³ : SMul M β
                                                 inst✝² : SMul N α
                                                 inst✝¹ : SMul N β
                                                 a : M
                                                 b✝ : α
                                                 c : β
                                                 x : Sum α β
                                                 inst✝ : FaithfulSMul M β
                                                 m₁✝ m₂✝ : M
                                                 h : ∀ (a : Sum α β), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                                 b : β
                                                 ⊢ Eq (HSMul.hSMul m₁✝ b) (HSMul.hSMul m₂✝ b)
                                               -/
  ⟨fun h => eq_of_smul_eq_smul fun b : β => by injection h (inr b)⟩
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
instance {m : Monoid M} [MulAction M α] [MulAction M β] :
    MulAction M (α ⊕ β) where
  mul_smul a b x := by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      m : Monoid M
      inst✝¹ : MulAction M α
      inst✝ : MulAction M β
      a b : M
      x : Sum α β
      ⊢ Eq (HSMul.hSMul (HMul.hMul a b) x) (HSMul.hSMul a (HSMul.hSMul b x))
    -/
    cases x
    /-
      case inl
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      m : Monoid M
      inst✝¹ : MulAction M α
      inst✝ : MulAction M β
      a b : M
      val✝ : α
      ⊢ Eq (HSMul.hSMul (HMul.hMul a b) (Sum.inl val✝)) (HSMul.hSMul a (HSMul.hSMul  …
    -/
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      m : Monoid M
      inst✝¹ : MulAction M α
      inst✝ : MulAction M β
      x : Sum α β
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    exacts [congr_arg inl (mul_smul _ _ _), congr_arg inr (mul_smul _ _ _)]
    /-
      case inl
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      m : Monoid M
      inst✝¹ : MulAction M α
      inst✝ : MulAction M β
      val✝ : α
      ⊢ Eq (HSMul.hSMul 1 (Sum.inl val✝)) (Sum.inl val✝)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  one_smul x := by
    cases x
    exacts [congr_arg inl (one_smul _ _), congr_arg inr (one_smul _ _)]


