/-- A linearly ordered commutative monoid with a zero element. -/
class LinearOrderedCommMonoidWithZero (α : Type*) extends LinearOrderedCommMonoid α,
  CommMonoidWithZero α where
  /-- `0 ≤ 1` in any linearly ordered commutative monoid. -/
  zero_le_one : (0 : α) ≤ 1


/-- A linearly ordered commutative group with a zero element. -/
class LinearOrderedCommGroupWithZero (α : Type*) extends LinearOrderedCommMonoidWithZero α,
  CommGroupWithZero α


instance (priority := 100) LinearOrderedCommMonoidWithZero.toZeroLeOneClass
    [LinearOrderedCommMonoidWithZero α] : ZeroLEOneClass α :=
  { ‹LinearOrderedCommMonoidWithZero α› with }


instance (priority := 100) canonicallyOrderedAddCommMonoid.toZeroLeOneClass
    [CanonicallyOrderedAddCommMonoid α] [One α] : ZeroLEOneClass α :=
  ⟨zero_le 1⟩


/-- Pullback a `LinearOrderedCommMonoidWithZero` under an injective map.
See note [reducible non-instances]. -/
abbrev Function.Injective.linearOrderedCommMonoidWithZero {β : Type*} [Zero β] [One β] [Mul β]
    [Pow β ℕ] [Max β] [Min β] (f : β → α) (hf : Function.Injective f) (zero : f 0 = 0)
    (one : f 1 = 1) (mul : ∀ x y, f (x * y) = f x * f y) (npow : ∀ (x) (n : ℕ), f (x ^ n) = f x ^ n)
    (hsup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (hinf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrderedCommMonoidWithZero β :=
  { LinearOrder.lift f hf hsup hinf, hf.orderedCommMonoid f one mul npow,
    hf.commMonoidWithZero f zero one mul npow with
    zero_le_one :=
                        /-
                          α : Type u_1
                          inst✝⁶ : LinearOrderedCommMonoidWithZero α
                          a b : α
                          n : Nat
                          β : Type u_2
                          inst✝⁵ : Zero β
                          inst✝⁴ : One β
                          inst✝³ : Mul β
                          inst✝² : Pow β Nat
                          inst✝¹ : Max β
                          inst✝ : Min β
                          f : β → α
                          hf : Function.Injective f
                          zero : Eq (f 0) 0
                          one : Eq (f 1) 1
                          mul : ∀ (x y : β), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                          npow : ∀ (x : β) (n : Nat), Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
                          hsup : ∀ (x y : β), Eq (f (Max.max x y)) (Max.max (f x) (f y))
                          hinf : ∀ (x y : β), Eq (f (Min.min x y)) (Min.min (f x) (f y))
                          ⊢ LE.le (f 0) (f 1)
                        -/
      show f 0 ≤ f 1 by simp only [zero, one, LinearOrderedCommMonoidWithZero.zero_le_one] }
                        /-
                          🎉 no goals
                        -/


@[simp] lemma zero_le' : 0 ≤ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommMonoidWithZero α
    a : α
    ⊢ LE.le 0 a
  -/
  simpa only [mul_zero, mul_one] using mul_le_mul_left' (zero_le_one' α) a
  /-
    🎉 no goals
  -/


@[simp]
theorem not_lt_zero' : ¬a < 0 :=
  not_lt_of_le zero_le'


@[simp]
theorem le_zero_iff : a ≤ 0 ↔ a = 0 :=
  ⟨fun h ↦ le_antisymm h zero_le', fun h ↦ h ▸ le_rfl⟩


theorem zero_lt_iff : 0 < a ↔ a ≠ 0 :=
  ⟨ne_of_gt, fun h ↦ lt_of_le_of_ne zero_le' h.symm⟩


theorem ne_zero_of_lt (h : b < a) : a ≠ 0 := fun h1 ↦ not_lt_zero' <| show b < 0 from h1 ▸ h


instance instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual :
    LinearOrderedAddCommMonoidWithTop (Additive αᵒᵈ) where
  top := (0 : α)
  top_add' := fun a ↦ zero_mul a.toMul
  le_top := fun _ ↦ zero_le'


instance instLinearOrderedAddCommMonoidWithTopOrderDualAdditive :
    LinearOrderedAddCommMonoidWithTop (Additive α)ᵒᵈ where
  top := OrderDual.toDual (Additive.ofMul 0)
  top_add' := fun a ↦ zero_mul (Additive.toMul (OrderDual.ofDual a))
  le_top := fun a ↦ @zero_le' _ _ (Additive.toMul (OrderDual.ofDual a))


                                                         /-
                                                           α : Type u_1
                                                           inst✝¹ : LinearOrderedCommMonoidWithZero α
                                                           a : α
                                                           n : Nat
                                                           inst✝ : NoZeroDivisors α
                                                           hn : Ne n 0
                                                           ⊢ Iff (LT.lt 0 (HPow.hPow a n)) (LT.lt 0 a)
                                                         -/
lemma pow_pos_iff (hn : n ≠ 0) : 0 < a ^ n ↔ 0 < a := by simp_rw [zero_lt_iff, pow_ne_zero_iff hn]
                                                         /-
                                                           🎉 no goals
                                                         -/


instance (priority := 100) LinearOrderedCommGroupWithZero.toMulPosMono : MulPosMono α where
  elim _a _b _c hbc := mul_le_mul_right' hbc _

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedCommGroupWithZero.toPosMulMono : PosMulMono α where
  elim _a _b _c hbc := mul_le_mul_left' hbc _

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedCommGroupWithZero.toPosMulReflectLE :
    PosMulReflectLE α where
                       /-
                         α : Type u_1
                         inst✝ : LinearOrderedCommGroupWithZero α
                         a✝ b✝ c✝ d : α
                         m n : Nat
                         a : Subtype fun x => LT.lt 0 x
                         b c : α
                         hbc : LE.le (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
                         ⊢ LE.le b c
                       -/
  elim a b c hbc := by simpa [a.2.ne'] using mul_le_mul_left' hbc a⁻¹
                       /-
                         🎉 no goals
                       -/

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedCommGroupWithZero.toMulPosReflectLE :
    MulPosReflectLE α where
                       /-
                         α : Type u_1
                         inst✝ : LinearOrderedCommGroupWithZero α
                         a✝ b✝ c✝ d : α
                         m n : Nat
                         a : Subtype fun x => LT.lt 0 x
                         b c : α
                         hbc : LE.le (HMul.hMul b ↑a) (HMul.hMul c ↑a)
                         ⊢ LE.le b c
                       -/
  elim a b c hbc := by simpa [a.2.ne'] using mul_le_mul_right' hbc a⁻¹
                       /-
                         🎉 no goals
                       -/

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedCommGroupWithZero.toPosMulReflectLT :
    PosMulReflectLT α where elim _a _b _c := lt_of_mul_lt_mul_left'

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedCommGroupWithZero.toPosMulStrictMono :
    PosMulStrictMono α where
                       /-
                         α : Type u_1
                         inst✝ : LinearOrderedCommGroupWithZero α
                         a✝ b✝ c✝ d : α
                         m n : Nat
                         a : Subtype fun x => LT.lt 0 x
                         b c : α
                         hbc : LT.lt b c
                         ⊢ LT.lt (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
                       -/
  elim a b c hbc := by by_contra! h; exact hbc.not_le <| (mul_le_mul_left a.2).1 h
                                     /-
                                       🎉 no goals
                                     -/

-- See note [lower instance priority]

instance (priority := 100) LinearOrderedCommGroupWithZero.toMulPosStrictMono :
    MulPosStrictMono α where
                       /-
                         α : Type u_1
                         inst✝ : LinearOrderedCommGroupWithZero α
                         a✝ b✝ c✝ d : α
                         m n : Nat
                         a : Subtype fun x => LT.lt 0 x
                         b c : α
                         hbc : LT.lt b c
                         ⊢ LT.lt (HMul.hMul b ↑a) (HMul.hMul c ↑a)
                       -/
  elim a b c hbc := by by_contra! h; exact hbc.not_le <| (mul_le_mul_right a.2).1 h
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated mul_inv_le_of_le_mul₀ (since := "2024-11-18")]
theorem mul_inv_le_of_le_mul (hab : a ≤ b * c) : a * c⁻¹ ≤ b :=
  mul_inv_le_of_le_mul₀ zero_le' zero_le' hab


@[simp]
theorem Units.zero_lt (u : αˣ) : (0 : α) < u :=
  zero_lt_iff.2 u.ne_zero


@[deprecated mul_lt_mul_of_le_of_lt_of_nonneg_of_pos (since := "2024-11-18")]
theorem mul_lt_mul_of_lt_of_le₀ (hab : a ≤ b) (hb : b ≠ 0) (hcd : c < d) : a * c < b * d :=
  mul_lt_mul_of_le_of_lt_of_nonneg_of_pos hab hcd zero_le' (zero_lt_iff.2 hb)


@[deprecated mul_lt_mul'' (since := "2024-11-18")]
theorem mul_lt_mul₀ (hab : a < b) (hcd : c < d) : a * c < b * d :=
  mul_lt_mul'' hab hcd zero_le' zero_le'


theorem mul_inv_lt_of_lt_mul₀ (h : a < b * c) : a * c⁻¹ < b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a b c : α
    h : LT.lt a (HMul.hMul b c)
    ⊢ LT.lt (HMul.hMul a (Inv.inv c)) b
  -/
  contrapose! h
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a b c : α
    h : LE.le b (HMul.hMul a (Inv.inv c))
    ⊢ LE.le (HMul.hMul b c) a
  -/
  simpa only [inv_inv] using mul_inv_le_of_le_mul₀ zero_le' zero_le' h
  /-
    🎉 no goals
  -/


theorem inv_mul_lt_of_lt_mul₀ (h : a < b * c) : b⁻¹ * a < c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a b c : α
    h : LT.lt a (HMul.hMul b c)
    ⊢ LT.lt (HMul.hMul (Inv.inv b) a) c
  -/
  rw [mul_comm] at *
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a b c : α
    h : LT.lt a (HMul.hMul c b)
    ⊢ LT.lt (HMul.hMul a (Inv.inv b)) c
  -/
  exact mul_inv_lt_of_lt_mul₀ h
  /-
    🎉 no goals
  -/


theorem lt_of_mul_lt_mul_of_le₀ (h : a * b < c * d) (hc : 0 < c) (hh : c ≤ a) : b < d := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a b c d : α
    h : LT.lt (HMul.hMul a b) (HMul.hMul c d)
    hc : LT.lt 0 c
    hh : LE.le c a
    ⊢ LT.lt b d
  -/
  have ha : a ≠ 0 := ne_of_gt (lt_of_lt_of_le hc hh)
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a b c d : α
    h : LT.lt (HMul.hMul a b) (HMul.hMul c d)
    hc : LT.lt 0 c
    hh : LE.le c a
    ha : Ne a 0
    ⊢ LT.lt b d
  -/
  rw [← inv_le_inv₀ (zero_lt_iff.2 ha) hc] at hh
  simpa [inv_mul_cancel_left₀ ha, inv_mul_cancel_left₀ hc.ne']
    using mul_lt_mul_of_le_of_lt_of_nonneg_of_pos hh  h zero_le' (inv_pos.2 hc)


@[deprecated div_le_div_iff_of_pos_right (since := "2024-11-18")]
theorem div_le_div_right₀ (hc : c ≠ 0) : a / c ≤ b / c ↔ a ≤ b :=
  div_le_div_iff_of_pos_right (zero_lt_iff.2 hc)


@[deprecated div_le_div_iff_of_pos_left (since := "2024-11-18")]
theorem div_le_div_left₀ (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a / b ≤ a / c ↔ c ≤ b :=
  div_le_div_iff_of_pos_left (zero_lt_iff.2 ha) (zero_lt_iff.2 hb) (zero_lt_iff.2 hc)


/-- `Equiv.mulLeft₀` as an `OrderIso` on a `LinearOrderedCommGroupWithZero.`. -/
@[simps! (config := { simpRhs := true }) apply toEquiv,
deprecated OrderIso.mulLeft₀ (since := "2024-11-18")]
def OrderIso.mulLeft₀' {a : α} (ha : a ≠ 0) : α ≃o α := .mulLeft₀ a (zero_lt_iff.2 ha)


set_option linter.deprecated false in
@[deprecated OrderIso.mulLeft₀_symm (since := "2024-11-18")]
theorem OrderIso.mulLeft₀'_symm {a : α} (ha : a ≠ 0) :
    (OrderIso.mulLeft₀' ha).symm = OrderIso.mulLeft₀' (inv_ne_zero ha) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a : α
    ha : Ne a 0
    ⊢ Eq (OrderIso.mulLeft₀' ha).symm (OrderIso.mulLeft₀' ⋯)
  -/
  ext
  /-
    case h.h
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a : α
    ha : Ne a 0
    x✝ : α
    ⊢ Eq ((OrderIso.mulLeft₀' ha).symm x✝) ((OrderIso.mulLeft₀' ⋯) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Equiv.mulRight₀` as an `OrderIso` on a `LinearOrderedCommGroupWithZero.`. -/
@[simps! (config := { simpRhs := true }) apply toEquiv,
deprecated OrderIso.mulRight₀ (since := "2024-11-18")]
def OrderIso.mulRight₀' {a : α} (ha : a ≠ 0) : α ≃o α := .mulRight₀ a (zero_lt_iff.2 ha)


set_option linter.deprecated false in
@[deprecated OrderIso.mulRight₀_symm (since := "2024-11-18")]
theorem OrderIso.mulRight₀'_symm {a : α} (ha : a ≠ 0) :
    (OrderIso.mulRight₀' ha).symm = OrderIso.mulRight₀' (inv_ne_zero ha) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a : α
    ha : Ne a 0
    ⊢ Eq (OrderIso.mulRight₀' ha).symm (OrderIso.mulRight₀' ⋯)
  -/
  ext
  /-
    case h.h
    α : Type u_1
    inst✝ : LinearOrderedCommGroupWithZero α
    a : α
    ha : Ne a 0
    x✝ : α
    ⊢ Eq ((OrderIso.mulRight₀' ha).symm x✝) ((OrderIso.mulRight₀' ⋯) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : LinearOrderedAddCommGroupWithTop (Additive αᵒᵈ) where
  __ := Additive.subNegMonoid
  __ := instLinearOrderedAddCommMonoidWithTopAdditiveOrderDual
  neg_top := inv_zero (G₀ := α)
  add_neg_cancel := fun a ha ↦ mul_inv_cancel₀ (G₀ := α) (id ha : a.toMul ≠ 0)


instance : LinearOrderedAddCommGroupWithTop (Additive α)ᵒᵈ where
  __ := instSubNegAddMonoidOrderDual
  __ := instLinearOrderedAddCommMonoidWithTopOrderDualAdditive
  neg_top := inv_zero (G₀ := α)
  add_neg_cancel := fun a ha ↦ mul_inv_cancel₀ (G₀ := α) (id ha : a.toMul ≠ 0)


@[deprecated pow_lt_pow_right₀ (since := "2024-11-18")]
lemma pow_lt_pow_succ (ha : 1 < a) : a ^ n < a ^ n.succ := pow_lt_pow_right₀ ha n.lt_succ_self


instance instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual
    [LinearOrderedAddCommMonoidWithTop α] :
    LinearOrderedCommMonoidWithZero (Multiplicative αᵒᵈ) where
  __ := Multiplicative.orderedCommMonoid
  __ := Multiplicative.linearOrder
  zero := Multiplicative.ofAdd (OrderDual.toDual ⊤)
  zero_mul := @top_add _ (_)
  -- Porting note:  Here and elsewhere in the file, just `zero_mul` worked in Lean 3. See
  -- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Type.20synonyms
  mul_zero := @add_top _ (_)
  zero_le_one := (le_top : (0 : α) ≤ ⊤)


@[simp]
theorem ofAdd_toDual_eq_zero_iff [LinearOrderedAddCommMonoidWithTop α]
    (x : α) : Multiplicative.ofAdd (OrderDual.toDual x) = 0 ↔ x = ⊤ := Iff.rfl


@[simp]
theorem ofDual_toAdd_eq_top_iff [LinearOrderedAddCommMonoidWithTop α]
    (x : Multiplicative αᵒᵈ) : OrderDual.ofDual x.toAdd = ⊤ ↔ x = 0 := Iff.rfl


@[simp]
theorem ofAdd_bot [LinearOrderedAddCommMonoidWithTop α] :
    Multiplicative.ofAdd ⊥ = (0 : Multiplicative αᵒᵈ) := rfl


@[simp]
theorem ofDual_toAdd_zero [LinearOrderedAddCommMonoidWithTop α] :
    OrderDual.ofDual (0 : Multiplicative αᵒᵈ).toAdd = ⊤ := rfl


instance [LinearOrderedAddCommGroupWithTop α] :
    LinearOrderedCommGroupWithZero (Multiplicative αᵒᵈ) :=
  { Multiplicative.divInvMonoid, instLinearOrderedCommMonoidWithZeroMultiplicativeOrderDual,
    Multiplicative.instNontrivial with
    inv_zero := @LinearOrderedAddCommGroupWithTop.neg_top _ (_)
    mul_inv_cancel := @LinearOrderedAddCommGroupWithTop.add_neg_cancel _ (_) }


instance preorder : Preorder (WithZero α) := WithBot.preorder

instance orderBot : OrderBot (WithZero α) := WithBot.orderBot


lemma zero_le (a : WithZero α) : 0 ≤ a := bot_le


lemma zero_lt_coe (a : α) : (0 : WithZero α) < a := WithBot.bot_lt_coe a


lemma zero_eq_bot : (0 : WithZero α) = ⊥ := rfl


@[simp, norm_cast] lemma coe_lt_coe : (a : WithZero α) < b ↔ a < b := WithBot.coe_lt_coe


@[simp, norm_cast] lemma coe_le_coe : (a : WithZero α) ≤ b ↔ a ≤ b := WithBot.coe_le_coe


@[simp, norm_cast] lemma one_lt_coe [One α] : 1 < (a : WithZero α) ↔ 1 < a := coe_lt_coe


@[simp, norm_cast] lemma one_le_coe [One α] : 1 ≤ (a : WithZero α) ↔ 1 ≤ a := coe_le_coe


@[simp, norm_cast] lemma coe_lt_one [One α] : (a : WithZero α) < 1 ↔ a < 1 := coe_lt_coe


@[simp, norm_cast] lemma coe_le_one [One α] : (a : WithZero α) ≤ 1 ↔ a ≤ 1 := coe_le_coe


theorem coe_le_iff {x : WithZero α} : (a : WithZero α) ≤ x ↔ ∃ b : α, x = b ∧ a ≤ b :=
  WithBot.coe_le_iff


@[simp] lemma unzero_le_unzero {a b : WithZero α} (ha hb) :
    unzero (x := a) ha ≤ unzero (x := b) hb ↔ a ≤ b := by
  -- TODO: Fix `lift` so that it doesn't try to clear the hypotheses I give it when it is
  -- impossible to do so. See https://github.com/leanprover-community/mathlib4/issues/19160
  /-
    α : Type u_1
    inst✝ : Preorder α
    a b : WithZero α
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Iff (LE.le (WithZero.unzero ha) (WithZero.unzero hb)) (LE.le a b)
  -/
  lift a to α using id ha
  /-
    case intro
    α : Type u_1
    inst✝ : Preorder α
    b : WithZero α
    hb : Ne b 0
    a : α
    ha : Ne (↑a) 0
    ⊢ Iff (LE.le (WithZero.unzero ha) (WithZero.unzero hb)) (LE.le (↑a) b)
  -/
  lift b to α using id hb
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ha : Ne (↑a) 0
    b : α
    hb : Ne (↑b) 0
    ⊢ Iff (LE.le (WithZero.unzero ha) (WithZero.unzero hb)) (LE.le ↑a ↑b)
  -/
  simp
  /-
    🎉 no goals
  -/


instance mulLeftMono [Mul α] [MulLeftMono α] :
    MulLeftMono (WithZero α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    a b : α
    inst✝¹ : Mul α
    inst✝ : MulLeftMono α
    ⊢ MulLeftMono (WithZero α)
  -/
  refine ⟨fun a b c hbc => ?_⟩
  /-
    α : Type u_1
    inst✝² : Preorder α
    a✝ b✝ : α
    inst✝¹ : Mul α
    inst✝ : MulLeftMono α
    a b c : WithZero α
    hbc : LE.le b c
    ⊢ LE.le (HMul.hMul a b) (HMul.hMul a c)
  -/
  induction a; · exact zero_le _
                 /-
                   🎉 no goals
                 -/
  /-
    case h₂
    α : Type u_1
    inst✝² : Preorder α
    a b✝ : α
    inst✝¹ : Mul α
    inst✝ : MulLeftMono α
    b c : WithZero α
    hbc : LE.le b c
    a✝ : α
    ⊢ LE.le (HMul.hMul (↑a✝) b) (HMul.hMul (↑a✝) c)
  -/
  induction b; · exact zero_le _
                 /-
                   🎉 no goals
                 -/
  /-
    case h₂.h₂
    α : Type u_1
    inst✝² : Preorder α
    a b : α
    inst✝¹ : Mul α
    inst✝ : MulLeftMono α
    c : WithZero α
    a✝¹ a✝ : α
    hbc : LE.le (↑a✝) c
    ⊢ LE.le (HMul.hMul ↑a✝¹ ↑a✝) (HMul.hMul (↑a✝¹) c)
  -/
  rcases WithZero.coe_le_iff.1 hbc with ⟨c, rfl, hbc'⟩
  /-
    case h₂.h₂.intro.intro
    α : Type u_1
    inst✝² : Preorder α
    a b : α
    inst✝¹ : Mul α
    inst✝ : MulLeftMono α
    a✝¹ a✝ c : α
    hbc' : LE.le a✝ c
    hbc : LE.le ↑a✝ ↑c
    ⊢ LE.le (HMul.hMul ↑a✝¹ ↑a✝) (HMul.hMul ↑a✝¹ ↑c)
  -/
  rw [← coe_mul _ c, ← coe_mul, coe_le_coe]
  /-
    case h₂.h₂.intro.intro
    α : Type u_1
    inst✝² : Preorder α
    a b : α
    inst✝¹ : Mul α
    inst✝ : MulLeftMono α
    a✝¹ a✝ c : α
    hbc' : LE.le a✝ c
    hbc : LE.le ↑a✝ ↑c
    ⊢ LE.le (HMul.hMul a✝¹ a✝) (HMul.hMul a✝¹ c)
  -/
  exact mul_le_mul_left' hbc' _
  /-
    🎉 no goals
  -/


protected lemma addLeftMono [AddZeroClass α] [AddLeftMono α]
    (h : ∀ a : α, 0 ≤ a) : AddLeftMono (WithZero α) := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : AddZeroClass α
    inst✝ : AddLeftMono α
    h : ∀ (a : α), LE.le 0 a
    ⊢ AddLeftMono (WithZero α)
  -/
  refine ⟨fun a b c hbc => ?_⟩
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : AddZeroClass α
    inst✝ : AddLeftMono α
    h : ∀ (a : α), LE.le 0 a
    a b c : WithZero α
    hbc : LE.le b c
    ⊢ LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
  -/
  induction a
    /-
      case h₁
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : AddZeroClass α
      inst✝ : AddLeftMono α
      h : ∀ (a : α), LE.le 0 a
      b c : WithZero α
      hbc : LE.le b c
      ⊢ LE.le (HAdd.hAdd 0 b) (HAdd.hAdd 0 c)
    -/
  · rwa [zero_add, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case h₂
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : AddZeroClass α
    inst✝ : AddLeftMono α
    h : ∀ (a : α), LE.le 0 a
    b c : WithZero α
    hbc : LE.le b c
    a✝ : α
    ⊢ LE.le (HAdd.hAdd (↑a✝) b) (HAdd.hAdd (↑a✝) c)
  -/
  induction b
    /-
      case h₂.h₁
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : AddZeroClass α
      inst✝ : AddLeftMono α
      h : ∀ (a : α), LE.le 0 a
      c : WithZero α
      a✝ : α
      hbc : LE.le 0 c
      ⊢ LE.le (HAdd.hAdd (↑a✝) 0) (HAdd.hAdd (↑a✝) c)
    -/
  · rw [add_zero]
    /-
      case h₂.h₁
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : AddZeroClass α
      inst✝ : AddLeftMono α
      h : ∀ (a : α), LE.le 0 a
      c : WithZero α
      a✝ : α
      hbc : LE.le 0 c
      ⊢ LE.le (↑a✝) (HAdd.hAdd (↑a✝) c)
    -/
    induction c
      /-
        case h₂.h₁.h₁
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : AddZeroClass α
        inst✝ : AddLeftMono α
        h : ∀ (a : α), LE.le 0 a
        a✝ : α
        hbc : LE.le 0 0
        ⊢ LE.le (↑a✝) (HAdd.hAdd (↑a✝) 0)
      -/
    · rw [add_zero]
      /-
        🎉 no goals
      -/
      /-
        case h₂.h₁.h₂
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : AddZeroClass α
        inst✝ : AddLeftMono α
        h : ∀ (a : α), LE.le 0 a
        a✝¹ a✝ : α
        hbc : LE.le 0 ↑a✝
        ⊢ LE.le (↑a✝¹) (HAdd.hAdd ↑a✝¹ ↑a✝)
      -/
    · rw [← coe_add, coe_le_coe]
      /-
        case h₂.h₁.h₂
        α : Type u_1
        inst✝² : Preorder α
        inst✝¹ : AddZeroClass α
        inst✝ : AddLeftMono α
        h : ∀ (a : α), LE.le 0 a
        a✝¹ a✝ : α
        hbc : LE.le 0 ↑a✝
        ⊢ LE.le a✝¹ (HAdd.hAdd a✝¹ a✝)
      -/
      exact le_add_of_nonneg_right (h _)
      /-
        🎉 no goals
      -/
    /-
      case h₂.h₂
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : AddZeroClass α
      inst✝ : AddLeftMono α
      h : ∀ (a : α), LE.le 0 a
      c : WithZero α
      a✝¹ a✝ : α
      hbc : LE.le (↑a✝) c
      ⊢ LE.le (HAdd.hAdd ↑a✝¹ ↑a✝) (HAdd.hAdd (↑a✝¹) c)
    -/
  · rcases WithZero.coe_le_iff.1 hbc with ⟨c, rfl, hbc'⟩
    /-
      case h₂.h₂.intro.intro
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : AddZeroClass α
      inst✝ : AddLeftMono α
      h : ∀ (a : α), LE.le 0 a
      a✝¹ a✝ c : α
      hbc' : LE.le a✝ c
      hbc : LE.le ↑a✝ ↑c
      ⊢ LE.le (HAdd.hAdd ↑a✝¹ ↑a✝) (HAdd.hAdd ↑a✝¹ ↑c)
    -/
    rw [← coe_add, ← coe_add _ c, coe_le_coe]
    /-
      case h₂.h₂.intro.intro
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : AddZeroClass α
      inst✝ : AddLeftMono α
      h : ∀ (a : α), LE.le 0 a
      a✝¹ a✝ c : α
      hbc' : LE.le a✝ c
      hbc : LE.le ↑a✝ ↑c
      ⊢ LE.le (HAdd.hAdd a✝¹ a✝) (HAdd.hAdd a✝¹ c)
    -/
    exact add_le_add_left hbc' _
    /-
      🎉 no goals
    -/


instance existsAddOfLE [Add α] [ExistsAddOfLE α] : ExistsAddOfLE (WithZero α) :=
  ⟨fun {a b} => by
    /-
      α : Type u_1
      inst✝² : Preorder α
      a✝ b✝ : α
      inst✝¹ : Add α
      inst✝ : ExistsAddOfLE α
      a b : WithZero α
      ⊢ LE.le a b → Exists fun c => Eq b (HAdd.hAdd a c)
    -/
    induction a
      /-
        case h₁
        α : Type u_1
        inst✝² : Preorder α
        a b✝ : α
        inst✝¹ : Add α
        inst✝ : ExistsAddOfLE α
        b : WithZero α
        ⊢ LE.le 0 b → Exists fun c => Eq b (HAdd.hAdd 0 c)
      -/
    · exact fun _ => ⟨b, (zero_add b).symm⟩
      /-
        🎉 no goals
      -/
    /-
      case h₂
      α : Type u_1
      inst✝² : Preorder α
      a b✝ : α
      inst✝¹ : Add α
      inst✝ : ExistsAddOfLE α
      b : WithZero α
      a✝ : α
      ⊢ LE.le (↑a✝) b → Exists fun c => Eq b (HAdd.hAdd (↑a✝) c)
    -/
    induction b
      /-
        case h₂.h₁
        α : Type u_1
        inst✝² : Preorder α
        a b : α
        inst✝¹ : Add α
        inst✝ : ExistsAddOfLE α
        a✝ : α
        ⊢ LE.le (↑a✝) 0 → Exists fun c => Eq 0 (HAdd.hAdd (↑a✝) c)
      -/
    · exact fun h => (WithBot.not_coe_le_bot _ h).elim
      /-
        🎉 no goals
      -/
    /-
      case h₂.h₂
      α : Type u_1
      inst✝² : Preorder α
      a b : α
      inst✝¹ : Add α
      inst✝ : ExistsAddOfLE α
      a✝¹ a✝ : α
      ⊢ LE.le ↑a✝¹ ↑a✝ → Exists fun c => Eq (↑a✝) (HAdd.hAdd (↑a✝¹) c)
    -/
    intro h
    /-
      case h₂.h₂
      α : Type u_1
      inst✝² : Preorder α
      a b : α
      inst✝¹ : Add α
      inst✝ : ExistsAddOfLE α
      a✝¹ a✝ : α
      h : LE.le ↑a✝¹ ↑a✝
      ⊢ Exists fun c => Eq (↑a✝) (HAdd.hAdd (↑a✝¹) c)
    -/
    obtain ⟨c, rfl⟩ := exists_add_of_le (WithZero.coe_le_coe.1 h)
    /-
      case h₂.h₂.intro
      α : Type u_1
      inst✝² : Preorder α
      a b : α
      inst✝¹ : Add α
      inst✝ : ExistsAddOfLE α
      a✝ c : α
      h : LE.le ↑a✝ ↑(HAdd.hAdd a✝ c)
      ⊢ Exists fun c_1 => Eq (↑(HAdd.hAdd a✝ c)) (HAdd.hAdd (↑a✝) c_1)
    -/
    exact ⟨c, rfl⟩⟩
    /-
      🎉 no goals
    -/


instance partialOrder : PartialOrder (WithZero α) := WithBot.partialOrder


instance mulLeftReflectLT [Mul α] [MulLeftReflectLT α] :
    MulLeftReflectLT (WithZero α) := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : Mul α
    inst✝ : MulLeftReflectLT α
    ⊢ MulLeftReflectLT (WithZero α)
  -/
  refine ⟨fun a b c h => ?_⟩
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : Mul α
    inst✝ : MulLeftReflectLT α
    a b c : WithZero α
    h : LT.lt (HMul.hMul a b) (HMul.hMul a c)
    ⊢ LT.lt b c
  -/
  have := ((zero_le _).trans_lt h).ne'
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : Mul α
    inst✝ : MulLeftReflectLT α
    a b c : WithZero α
    h : LT.lt (HMul.hMul a b) (HMul.hMul a c)
    this : Ne (HMul.hMul a c) 0
    ⊢ LT.lt b c
  -/
  induction a
    /-
      case h₁
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : Mul α
      inst✝ : MulLeftReflectLT α
      b c : WithZero α
      h : LT.lt (HMul.hMul 0 b) (HMul.hMul 0 c)
      this : Ne (HMul.hMul 0 c) 0
      ⊢ LT.lt b c
    -/
  · simp at this
    /-
      🎉 no goals
    -/
  /-
    case h₂
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : Mul α
    inst✝ : MulLeftReflectLT α
    b c : WithZero α
    a✝ : α
    h : LT.lt (HMul.hMul (↑a✝) b) (HMul.hMul (↑a✝) c)
    this : Ne (HMul.hMul (↑a✝) c) 0
    ⊢ LT.lt b c
  -/
  induction c
    /-
      case h₂.h₁
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : Mul α
      inst✝ : MulLeftReflectLT α
      b : WithZero α
      a✝ : α
      h : LT.lt (HMul.hMul (↑a✝) b) (HMul.hMul (↑a✝) 0)
      this : Ne (HMul.hMul (↑a✝) 0) 0
      ⊢ LT.lt b 0
    -/
  · simp at this
    /-
      🎉 no goals
    -/
  /-
    case h₂.h₂
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : Mul α
    inst✝ : MulLeftReflectLT α
    b : WithZero α
    a✝¹ a✝ : α
    h : LT.lt (HMul.hMul (↑a✝¹) b) (HMul.hMul ↑a✝¹ ↑a✝)
    this : Ne (HMul.hMul ↑a✝¹ ↑a✝) 0
    ⊢ LT.lt b ↑a✝
  -/
  induction b
  /-
    case h₂.h₂.h₁
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : Mul α
    inst✝ : MulLeftReflectLT α
    a✝¹ a✝ : α
    this : Ne (HMul.hMul ↑a✝¹ ↑a✝) 0
    h : LT.lt (HMul.hMul (↑a✝¹) 0) (HMul.hMul ↑a✝¹ ↑a✝)
    ⊢ LT.lt 0 ↑a✝
  -/
  exacts [zero_lt_coe _, coe_lt_coe.mpr (lt_of_mul_lt_mul_left' <| coe_lt_coe.mp h)]
  /-
    🎉 no goals
  -/


instance lattice [Lattice α] : Lattice (WithZero α) := WithBot.lattice


instance linearOrder : LinearOrder (WithZero α) := WithBot.linearOrder


protected lemma le_max_iff : (a : WithZero α) ≤ max (b : WithZero α) c ↔ a ≤ max b c := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Iff (LE.le (↑a) (Max.max ↑b ↑c)) (LE.le a (Max.max b c))
  -/
  simp only [WithZero.coe_le_coe, le_max_iff]
  /-
    🎉 no goals
  -/


protected lemma min_le_iff : min (a : WithZero α) b ≤ c ↔ min a b ≤ c := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Iff (LE.le (Min.min ↑a ↑b) ↑c) (LE.le (Min.min a b) c)
  -/
  simp only [WithZero.coe_le_coe, min_le_iff]
  /-
    🎉 no goals
  -/


instance orderedCommMonoid [OrderedCommMonoid α] : OrderedCommMonoid (WithZero α) :=
  { WithZero.commMonoidWithZero.toCommMonoid, WithZero.partialOrder with
    mul_le_mul_left := fun _ _ => mul_le_mul_left' }

/-
Note 1 : the below is not an instance because it requires `zero_le`. It seems
like a rather pathological definition because α already has a zero.
Note 2 : there is no multiplicative analogue because it does not seem necessary.
Mathematicians might be more likely to use the order-dual version, where all
elements are ≤ 1 and then 1 is the top element.
-/

/-- If `0` is the least element in `α`, then `WithZero α` is an `OrderedAddCommMonoid`. -/
-- See note [reducible non-instances]
protected abbrev orderedAddCommMonoid [OrderedAddCommMonoid α] (zero_le : ∀ a : α, 0 ≤ a) :
    OrderedAddCommMonoid (WithZero α) :=
  { WithZero.partialOrder, WithZero.addCommMonoid with
    add_le_add_left := @add_le_add_left _ _ _ (WithZero.addLeftMono zero_le).. }

-- This instance looks absurd: a monoid already has a zero

/-- Adding a new zero to a canonically ordered additive monoid produces another one. -/
instance canonicallyOrderedAddCommMonoid [CanonicallyOrderedAddCommMonoid α] :
    CanonicallyOrderedAddCommMonoid (WithZero α) :=
  { WithZero.orderBot,
    WithZero.orderedAddCommMonoid _root_.zero_le,
    WithZero.existsAddOfLE with
    le_self_add := fun a b => by
      /-
        α : Type u_1
        inst✝ : CanonicallyOrderedAddCommMonoid α
        a b : WithZero α
        ⊢ LE.le a (HAdd.hAdd a b)
      -/
      induction a
        /-
          case h₁
          α : Type u_1
          inst✝ : CanonicallyOrderedAddCommMonoid α
          b : WithZero α
          ⊢ LE.le 0 (HAdd.hAdd 0 b)
        -/
      · exact bot_le
        /-
          🎉 no goals
        -/
      /-
        case h₂
        α : Type u_1
        inst✝ : CanonicallyOrderedAddCommMonoid α
        b : WithZero α
        a✝ : α
        ⊢ LE.le (↑a✝) (HAdd.hAdd (↑a✝) b)
      -/
      induction b
        /-
          case h₂.h₁
          α : Type u_1
          inst✝ : CanonicallyOrderedAddCommMonoid α
          a✝ : α
          ⊢ LE.le (↑a✝) (HAdd.hAdd (↑a✝) 0)
        -/
      · exact le_rfl
        /-
          🎉 no goals
        -/
        /-
          case h₂.h₂
          α : Type u_1
          inst✝ : CanonicallyOrderedAddCommMonoid α
          a✝¹ a✝ : α
          ⊢ LE.le (↑a✝¹) (HAdd.hAdd ↑a✝¹ ↑a✝)
        -/
      · exact WithZero.coe_le_coe.2 le_self_add }
        /-
          🎉 no goals
        -/


instance canonicallyLinearOrderedAddCommMonoid [CanonicallyLinearOrderedAddCommMonoid α] :
    CanonicallyLinearOrderedAddCommMonoid (WithZero α) :=
  { WithZero.canonicallyOrderedAddCommMonoid, WithZero.linearOrder with }


instance instLinearOrderedCommMonoidWithZero [LinearOrderedCommMonoid α] :
    LinearOrderedCommMonoidWithZero (WithZero α) :=
  { WithZero.linearOrder, WithZero.commMonoidWithZero with
    mul_le_mul_left := fun _ _ ↦ mul_le_mul_left', zero_le_one := WithZero.zero_le _ }


instance instLinearOrderedCommGroupWithZero [LinearOrderedCommGroup α] :
    LinearOrderedCommGroupWithZero (WithZero α) where
  __ := instLinearOrderedCommMonoidWithZero
  __ := commGroupWithZero


/-- Notation for `WithZero (Multiplicative ℕ)` -/
scoped[Multiplicative] notation "ℕₘ₀" => WithZero (Multiplicative ℕ)


/-- Notation for `WithZero (Multiplicative ℤ)` -/
scoped[Multiplicative] notation "ℤₘ₀" => WithZero (Multiplicative ℤ)


