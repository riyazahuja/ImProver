/-- `IsSymmOp op` where `op : α → α → β` says that `op` is a symmetric operation,
i.e. `op a b = op b a`.
It is the natural generalisation of `Std.Commutative` (`β = α`) and `IsSymm` (`β = Prop`). -/
class IsSymmOp (op : α → α → β) : Prop where
  /-- A symmetric operation satisfies `op a b = op b a`. -/
  symm_op : ∀ a b, op a b = op b a


/-- `LeftCommutative op` where `op : α → β → β` says that `op` is a left-commutative operation,
i.e. `op a₁ (op a₂ b) = op a₂ (op a₁ b)`. -/
class LeftCommutative (op : α → β → β) : Prop where
  /-- A left-commutative operation satisfies `op a₁ (op a₂ b) = op a₂ (op a₁ b)`. -/
  left_comm : (a₁ a₂ : α) → (b : β) → op a₁ (op a₂ b) = op a₂ (op a₁ b)


/-- `RightCommutative op` where `op : β → α → β` says that `op` is a right-commutative operation,
i.e. `op (op b a₁) a₂ = op (op b a₂) a₁`. -/
class RightCommutative (op : β → α → β) : Prop where
  /-- A right-commutative operation satisfies `op (op b a₁) a₂ = op (op b a₂) a₁`. -/
  right_comm : (b : β) → (a₁ a₂ : α) → op (op b a₁) a₂ = op (op b a₂) a₁


instance (priority := 100) isSymmOp_of_isCommutative (α : Sort u) (op : α → α → α)
    [Std.Commutative op] : IsSymmOp op where symm_op := Std.Commutative.comm


theorem IsSymmOp.flip_eq (op : α → α → β) [IsSymmOp op] : flip op = op :=
  funext fun a ↦ funext fun b ↦ (IsSymmOp.symm_op a b).symm


instance {f : α → β → β} [h : LeftCommutative f] : RightCommutative (fun x y ↦ f y x) :=
  ⟨fun _ _ _ ↦ (h.left_comm _ _ _).symm⟩


instance {f : β → α → β} [h : RightCommutative f] : LeftCommutative (fun x y ↦ f y x) :=
  ⟨fun _ _ _ ↦ (h.right_comm _ _ _).symm⟩


instance {f : α → α → α} [hc : Std.Commutative f] [ha : Std.Associative f] : LeftCommutative f :=
                  /-
                    α : Sort u
                    β : Sort v
                    f : α → α → α
                    hc : Std.Commutative f
                    ha : Std.Associative f
                    a b c : α
                    ⊢ Eq (f a (f b c)) (f b (f a c))
                  -/
  ⟨fun a b c ↦ by rw [← ha.assoc, hc.comm a, ha.assoc]⟩
                  /-
                    🎉 no goals
                  -/


instance {f : α → α → α} [hc : Std.Commutative f] [ha : Std.Associative f] : RightCommutative f :=
                  /-
                    α : Sort u
                    β : Sort v
                    f : α → α → α
                    hc : Std.Commutative f
                    ha : Std.Associative f
                    a b c : α
                    ⊢ Eq (f (f a b) c) (f (f a c) b)
                  -/
  ⟨fun a b c ↦ by rw [ha.assoc, hc.comm b, ha.assoc]⟩
                  /-
                    🎉 no goals
                  -/

