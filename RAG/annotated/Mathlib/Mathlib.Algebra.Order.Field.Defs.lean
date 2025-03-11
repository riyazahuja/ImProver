/-- A linear ordered semifield is a field with a linear order respecting the operations. -/
class LinearOrderedSemifield (α : Type*) extends LinearOrderedCommSemiring α, Semifield α


/-- A linear ordered field is a field with a linear order respecting the operations. -/
class LinearOrderedField (α : Type*) extends LinearOrderedCommRing α, Field α

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedField.toLinearOrderedSemifield [LinearOrderedField α] :
    LinearOrderedSemifield α :=
  { LinearOrderedRing.toLinearOrderedSemiring, ‹LinearOrderedField α› with }


/-- Equality holds when `a ≠ 0`. See `mul_inv_cancel`. -/
                                         /-
                                           α : Type u_1
                                           inst✝ : LinearOrderedSemifield α
                                           a : α
                                           ⊢ LE.le (HMul.hMul a (Inv.inv a)) 1
                                         -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
lemma mul_inv_le_one : a * a⁻¹ ≤ 1 := by obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Equality holds when `a ≠ 0`. See `inv_mul_cancel`. -/
                                         /-
                                           α : Type u_1
                                           inst✝ : LinearOrderedSemifield α
                                           a : α
                                           ⊢ LE.le (HMul.hMul (Inv.inv a) a) 1
                                         -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
lemma inv_mul_le_one : a⁻¹ * a ≤ 1 := by obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Equality holds when `a ≠ 0`. See `mul_inv_cancel_left`. -/
lemma mul_inv_left_le (hb : 0 ≤ b) : a * (a⁻¹ * b) ≤ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    hb : LE.le 0 b
    ⊢ LE.le (HMul.hMul a (HMul.hMul (Inv.inv a) b)) b
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `a ≠ 0`. See `mul_inv_cancel_left`. -/
lemma le_mul_inv_left (hb : b ≤ 0) : b ≤ a * (a⁻¹ * b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    hb : LE.le b 0
    ⊢ LE.le b (HMul.hMul a (HMul.hMul (Inv.inv a) b))
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `a ≠ 0`. See `inv_mul_cancel_left`. -/
lemma inv_mul_left_le (hb : 0 ≤ b) : a⁻¹ * (a * b) ≤ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    hb : LE.le 0 b
    ⊢ LE.le (HMul.hMul (Inv.inv a) (HMul.hMul a b)) b
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `a ≠ 0`. See `inv_mul_cancel_left`. -/
lemma le_inv_mul_left (hb : b ≤ 0) : b ≤ a⁻¹ * (a * b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    hb : LE.le b 0
    ⊢ LE.le b (HMul.hMul (Inv.inv a) (HMul.hMul a b))
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | ha := eq_or_ne a 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `b ≠ 0`. See `mul_inv_cancel_right`. -/
lemma mul_inv_right_le (ha : 0 ≤ a) : a * b * b⁻¹ ≤ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    ha : LE.le 0 a
    ⊢ LE.le (HMul.hMul (HMul.hMul a b) (Inv.inv b)) a
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | hb := eq_or_ne b 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `b ≠ 0`. See `mul_inv_cancel_right`. -/
lemma le_mul_inv_right (ha : a ≤ 0) : a ≤ a * b * b⁻¹ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    ha : LE.le a 0
    ⊢ LE.le a (HMul.hMul (HMul.hMul a b) (Inv.inv b))
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | hb := eq_or_ne b 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `b ≠ 0`. See `inv_mul_cancel_right`. -/
lemma inv_mul_right_le (ha : 0 ≤ a) : a * b⁻¹ * b ≤ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    ha : LE.le 0 a
    ⊢ LE.le (HMul.hMul (HMul.hMul a (Inv.inv b)) b) a
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | hb := eq_or_ne b 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `b ≠ 0`. See `inv_mul_cancel_right`. -/
lemma le_inv_mul_right (ha : a ≤ 0) : a ≤ a * b⁻¹ * b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b : α
    ha : LE.le a 0
    ⊢ LE.le a (HMul.hMul (HMul.hMul a (Inv.inv b)) b)
  -/
                                      /-
                                        🎉 no goals
                                      -/
  obtain rfl | hb := eq_or_ne b 0 <;> simp [*]
                                      /-
                                        🎉 no goals
                                      -/


/-- Equality holds when `c ≠ 0`. See `mul_div_mul_left`. -/
lemma mul_div_mul_left_le (h : 0 ≤ a / b) : c * a / (c * b) ≤ a / b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b c : α
    h : LE.le 0 (HDiv.hDiv a b)
    ⊢ LE.le (HDiv.hDiv (HMul.hMul c a) (HMul.hMul c b)) (HDiv.hDiv a b)
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b : α
      h : LE.le 0 (HDiv.hDiv a b)
      ⊢ LE.le (HDiv.hDiv (HMul.hMul 0 a) (HMul.hMul 0 b)) (HDiv.hDiv a b)
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b c : α
      h : LE.le 0 (HDiv.hDiv a b)
      hc : Ne c 0
      ⊢ LE.le (HDiv.hDiv (HMul.hMul c a) (HMul.hMul c b)) (HDiv.hDiv a b)
    -/
  · rw [mul_div_mul_left _ _ hc]
    /-
      🎉 no goals
    -/


/-- Equality holds when `c ≠ 0`. See `mul_div_mul_left`. -/
lemma le_mul_div_mul_left (h : a / b ≤ 0) : a / b ≤ c * a / (c * b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b c : α
    h : LE.le (HDiv.hDiv a b) 0
    ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul c a) (HMul.hMul c b))
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b : α
      h : LE.le (HDiv.hDiv a b) 0
      ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul 0 a) (HMul.hMul 0 b))
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b c : α
      h : LE.le (HDiv.hDiv a b) 0
      hc : Ne c 0
      ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul c a) (HMul.hMul c b))
    -/
  · rw [mul_div_mul_left _ _ hc]
    /-
      🎉 no goals
    -/


/-- Equality holds when `c ≠ 0`. See `mul_div_mul_right`. -/
lemma mul_div_mul_right_le (h : 0 ≤ a / b) : a * c / (b * c) ≤ a / b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b c : α
    h : LE.le 0 (HDiv.hDiv a b)
    ⊢ LE.le (HDiv.hDiv (HMul.hMul a c) (HMul.hMul b c)) (HDiv.hDiv a b)
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b : α
      h : LE.le 0 (HDiv.hDiv a b)
      ⊢ LE.le (HDiv.hDiv (HMul.hMul a 0) (HMul.hMul b 0)) (HDiv.hDiv a b)
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b c : α
      h : LE.le 0 (HDiv.hDiv a b)
      hc : Ne c 0
      ⊢ LE.le (HDiv.hDiv (HMul.hMul a c) (HMul.hMul b c)) (HDiv.hDiv a b)
    -/
  · rw [mul_div_mul_right _ _ hc]
    /-
      🎉 no goals
    -/


/-- Equality holds when `c ≠ 0`. See `mul_div_mul_right`. -/
lemma le_mul_div_mul_right (h : a / b ≤ 0) : a / b ≤ a * c / (b * c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a b c : α
    h : LE.le (HDiv.hDiv a b) 0
    ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul a c) (HMul.hMul b c))
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b : α
      h : LE.le (HDiv.hDiv a b) 0
      ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul a 0) (HMul.hMul b 0))
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      a b c : α
      h : LE.le (HDiv.hDiv a b) 0
      hc : Ne c 0
      ⊢ LE.le (HDiv.hDiv a b) (HDiv.hDiv (HMul.hMul a c) (HMul.hMul b c))
    -/
  · rw [mul_div_mul_right _ _ hc]
    /-
      🎉 no goals
    -/

