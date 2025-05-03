@[to_additive Option.VAdd]
instance : SMul M (Option α) :=
  ⟨fun a => Option.map <| (a • ·)⟩


@[to_additive]
theorem smul_def : a • x = x.map (a • ·) :=
  rfl


@[to_additive (attr := simp)]
theorem smul_none : a • (none : Option α) = none :=
  rfl


@[to_additive (attr := simp)]
theorem smul_some : a • some b = some (a • b) :=
  rfl


@[to_additive]
instance instIsScalarTowerOfSMul [SMul M N] [IsScalarTower M N α] : IsScalarTower M N (Option α) :=
  ⟨fun a b x => by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝³ : SMul M α
      inst✝² : SMul N α
      a✝ : M
      b✝ : α
      x✝ : Option α
      inst✝¹ : SMul M N
      inst✝ : IsScalarTower M N α
      a : M
      b : N
      x : Option α
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) x) (HSMul.hSMul a (HSMul.hSMul b x))
    -/
    cases x
    /-
      case none
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝³ : SMul M α
      inst✝² : SMul N α
      a✝ : M
      b✝ : α
      x : Option α
      inst✝¹ : SMul M N
      inst✝ : IsScalarTower M N α
      a : M
      b : N
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) Option.none) (HSMul.hSMul a (HSMul.hSMul b …
    -/
    exacts [rfl, congr_arg some (smul_assoc _ _ _)]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMulCommClass M N α] : SMulCommClass M N (Option α) :=
  ⟨fun _ _ => Function.Commute.option_map <| smul_comm _ _⟩


@[to_additive]
instance [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] : IsCentralScalar M (Option α) :=
  ⟨fun a x => by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝³ : SMul M α
      inst✝² : SMul N α
      a✝ : M
      b : α
      x✝ : Option α
      inst✝¹ : SMul (MulOpposite M) α
      inst✝ : IsCentralScalar M α
      a : M
      x : Option α
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) x) (HSMul.hSMul a x)
    -/
    cases x
    /-
      case none
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝³ : SMul M α
      inst✝² : SMul N α
      a✝ : M
      b : α
      x : Option α
      inst✝¹ : SMul (MulOpposite M) α
      inst✝ : IsCentralScalar M α
      a : M
      ⊢ Eq (HSMul.hSMul (MulOpposite.op a) Option.none) (HSMul.hSMul a Option.none)
    -/
    exacts [rfl, congr_arg some (op_smul_eq_smul _ _)]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [FaithfulSMul M α] : FaithfulSMul M (Option α) :=
                                               /-
                                                 M : Type u_1
                                                 N : Type u_2
                                                 α : Type u_3
                                                 inst✝² : SMul M α
                                                 inst✝¹ : SMul N α
                                                 a : M
                                                 b✝ : α
                                                 x : Option α
                                                 inst✝ : FaithfulSMul M α
                                                 m₁✝ m₂✝ : M
                                                 h : ∀ (a : Option α), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                                 b : α
                                                 ⊢ Eq (HSMul.hSMul m₁✝ b) (HSMul.hSMul m₂✝ b)
                                               -/
  ⟨fun h => eq_of_smul_eq_smul fun b : α => by injection h (some b)⟩
                                               /-
                                                 🎉 no goals
                                               -/


instance [Monoid M] [MulAction M α] :
    MulAction M (Option α) where
  smul := (· • ·)
  one_smul b := by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝¹ : Monoid M
      inst✝ : MulAction M α
      b : Option α
      ⊢ Eq (HSMul.hSMul 1 b) b
    -/
    cases b
    /-
      case none
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝¹ : Monoid M
      inst✝ : MulAction M α
      ⊢ Eq (HSMul.hSMul 1 Option.none) Option.none
    -/
    exacts [rfl, congr_arg some (one_smul _ _)]
    /-
      🎉 no goals
    -/
  mul_smul a₁ a₂ b := by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝¹ : Monoid M
      inst✝ : MulAction M α
      a₁ a₂ : M
      b : Option α
      ⊢ Eq (HSMul.hSMul (HMul.hMul a₁ a₂) b) (HSMul.hSMul a₁ (HSMul.hSMul a₂ b))
    -/
    cases b
    /-
      case none
      M : Type u_1
      N : Type u_2
      α : Type u_3
      inst✝¹ : Monoid M
      inst✝ : MulAction M α
      a₁ a₂ : M
      ⊢ Eq (HSMul.hSMul (HMul.hMul a₁ a₂) Option.none) (HSMul.hSMul a₁ (HSMul.hSMul  …
    -/
    exacts [rfl, congr_arg some (mul_smul _ _ _)]
    /-
      🎉 no goals
    -/


