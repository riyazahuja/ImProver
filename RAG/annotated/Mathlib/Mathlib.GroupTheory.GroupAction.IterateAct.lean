/-- A structure with a single field `val : ℕ`
that additively acts on `α` by `⟨n⟩ +ᵥ x = f^[n] x`. -/
structure IterateAddAct {α : Type*} (f : α → α) where
  /-- The value of `n : IterateAddAct f`. -/
  val : ℕ


/-- A structure with a single field `val : ℕ` that acts on `α` by `⟨n⟩ • x = f^[n] x`. -/
@[to_additive (attr := ext)]
structure IterateMulAct {α : Type*} (f : α → α) where
  /-- The value of `n : IterateMulAct f`. -/
  val : ℕ


@[to_additive]
instance instCountable : Countable (IterateMulAct f) :=
  Function.Injective.countable fun _ _ ↦ IterateMulAct.ext


@[to_additive]
instance instCommMonoid : CommMonoid (IterateMulAct f) where
  one := ⟨0⟩
  mul m n := ⟨m.1 + n.1⟩
                        /-
                          α : Type u_1
                          f : α → α
                          a b c : IterateMulAct f
                          ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
                        -/
  mul_assoc a b c := by ext; apply Nat.add_assoc
                             /-
                               🎉 no goals
                             -/
                  /-
                    α : Type u_1
                    f : α → α
                    x✝ : IterateMulAct f
                    ⊢ Eq (HMul.hMul 1 x✝) x✝
                  -/
  one_mul _ := by ext; apply Nat.zero_add
                       /-
                         🎉 no goals
                       -/
  mul_one _ := rfl
                     /-
                       α : Type u_1
                       f : α → α
                       x✝¹ x✝ : IterateMulAct f
                       ⊢ Eq (HMul.hMul x✝¹ x✝) (HMul.hMul x✝ x✝¹)
                     -/
  mul_comm _ _ := by ext; apply Nat.add_comm
                    /-
                      α : Type u_1
                      f : α → α
                      x✝ : IterateMulAct f
                      ⊢ Eq ((fun n a => { val := HMul.hMul n a.val }) 0 x✝) 1
                    -/
                          /-
                            🎉 no goals
                          -/
                         /-
                           🎉 no goals
                         -/
                      /-
                        α : Type u_1
                        f : α → α
                        n : Nat
                        a : IterateMulAct f
                        ⊢ Eq ((fun n a => { val := HMul.hMul n a.val }) (HAdd.hAdd n 1) a) (HMul.hMul  …
                      -/
  npow n a := ⟨n * a.val⟩
                           /-
                             🎉 no goals
                           -/
  npow_zero _ := by ext; apply Nat.zero_mul
  npow_succ n a := by ext; apply Nat.succ_mul


@[to_additive]
instance instMulAction : MulAction (IterateMulAct f) α where
  smul n x := f^[n.val] x
  one_smul _ := rfl
  mul_smul _ _ := Function.iterate_add_apply f _ _


@[to_additive (attr := simp)]
theorem mk_smul (n : ℕ) (x : α) : mk (f := f) n • x = f^[n] x := rfl


