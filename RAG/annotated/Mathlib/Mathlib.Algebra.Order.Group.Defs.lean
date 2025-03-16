/-- An ordered additive commutative group is an additive commutative group
with a partial order in which addition is strictly monotone. -/
class OrderedAddCommGroup (α : Type u) extends AddCommGroup α, PartialOrder α where
  /-- Addition is monotone in an ordered additive commutative group. -/
  protected add_le_add_left : ∀ a b : α, a ≤ b → ∀ c : α, c + a ≤ c + b


/-- An ordered commutative group is a commutative group
with a partial order in which multiplication is strictly monotone. -/
class OrderedCommGroup (α : Type u) extends CommGroup α, PartialOrder α where
  /-- Multiplication is monotone in an ordered commutative group. -/
  protected mul_le_mul_left : ∀ a b : α, a ≤ b → ∀ c : α, c * a ≤ c * b


@[to_additive]
instance OrderedCommGroup.toMulLeftMono (α : Type u) [OrderedCommGroup α] :
    MulLeftMono α where
      elim a b c bc := OrderedCommGroup.mul_le_mul_left b c bc a

-- See note [lower instance priority]

@[to_additive OrderedAddCommGroup.toOrderedCancelAddCommMonoid]
instance (priority := 100) OrderedCommGroup.toOrderedCancelCommMonoid [OrderedCommGroup α] :
    OrderedCancelCommMonoid α :=
{ ‹OrderedCommGroup α› with le_of_mul_le_mul_left := fun _ _ _ ↦ le_of_mul_le_mul_left' }


/-- A choice-free shortcut instance. -/
@[to_additive "A choice-free shortcut instance."]
theorem OrderedCommGroup.toMulLeftReflectLE (α : Type u) [OrderedCommGroup α] :
    MulLeftReflectLE α where
                          /-
                            α : Type u
                            inst✝ : OrderedCommGroup α
                            a b c : α
                            bc : LE.le (HMul.hMul a b) (HMul.hMul a c)
                            ⊢ LE.le b c
                          -/
      elim a b c bc := by simpa using mul_le_mul_left' bc a⁻¹
                          /-
                            🎉 no goals
                          -/

-- Porting note: this instance is not used,
-- and causes timeouts after https://github.com/leanprover/lean4/pull/2210.
-- See further explanation on `OrderedCommGroup.toMulLeftReflectLE`.

/-- A choice-free shortcut instance. -/
@[to_additive "A choice-free shortcut instance."]
theorem OrderedCommGroup.toMulRightReflectLE (α : Type u) [OrderedCommGroup α] :
    MulRightReflectLE α where
                          /-
                            α : Type u
                            inst✝ : OrderedCommGroup α
                            a b c : α
                            bc : LE.le (Function.swap (fun x1 x2 => HMul.hMul x1 x2) a b) (Function.swap ( …
                            ⊢ LE.le b c
                          -/
      elim a b c bc := by simpa using mul_le_mul_right' bc a⁻¹
                          /-
                            🎉 no goals
                          -/


alias OrderedCommGroup.mul_lt_mul_left' := mul_lt_mul_left'


attribute [to_additive OrderedAddCommGroup.add_lt_add_left] OrderedCommGroup.mul_lt_mul_left'


alias OrderedCommGroup.le_of_mul_le_mul_left := le_of_mul_le_mul_left'


attribute [to_additive] OrderedCommGroup.le_of_mul_le_mul_left


alias OrderedCommGroup.lt_of_mul_lt_mul_left := lt_of_mul_lt_mul_left'


attribute [to_additive] OrderedCommGroup.lt_of_mul_lt_mul_left



/-- A linearly ordered additive commutative group is an
additive commutative group with a linear order in which
addition is monotone. -/
class LinearOrderedAddCommGroup (α : Type u) extends OrderedAddCommGroup α, LinearOrder α


/-- A linearly ordered commutative group is a
commutative group with a linear order in which
multiplication is monotone. -/
@[to_additive]
class LinearOrderedCommGroup (α : Type u) extends OrderedCommGroup α, LinearOrder α


@[to_additive LinearOrderedAddCommGroup.add_lt_add_left]
theorem LinearOrderedCommGroup.mul_lt_mul_left' (a b : α) (h : a < b) (c : α) : c * a < c * b :=
  _root_.mul_lt_mul_left' h c


@[to_additive eq_zero_of_neg_eq]
theorem eq_one_of_inv_eq' (h : a⁻¹ = a) : a = 1 :=
  match lt_trichotomy a 1 with
  | Or.inl h₁ =>
    have : 1 < a := h ▸ one_lt_inv_of_inv h₁
    absurd h₁ this.asymm
  | Or.inr (Or.inl h₁) => h₁
  | Or.inr (Or.inr h₁) =>
    have : a < 1 := h ▸ inv_lt_one'.mpr h₁
    absurd h₁ this.asymm


@[to_additive exists_zero_lt]
theorem exists_one_lt' [Nontrivial α] : ∃ a : α, 1 < a := by
  /-
    α : Type u
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : Nontrivial α
    ⊢ Exists fun a => LT.lt 1 a
  -/
  obtain ⟨y, hy⟩ := Decidable.exists_ne (1 : α)
  /-
    case intro
    α : Type u
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : Nontrivial α
    y : α
    hy : Ne y 1
    ⊢ Exists fun a => LT.lt 1 a
  -/
  obtain h|h := hy.lt_or_lt
    /-
      case intro.inl
      α : Type u
      inst✝¹ : LinearOrderedCommGroup α
      inst✝ : Nontrivial α
      y : α
      hy : Ne y 1
      h : LT.lt y 1
      ⊢ Exists fun a => LT.lt 1 a
    -/
  · exact ⟨y⁻¹, one_lt_inv'.mpr h⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u
      inst✝¹ : LinearOrderedCommGroup α
      inst✝ : Nontrivial α
      y : α
      hy : Ne y 1
      h : LT.lt 1 y
      ⊢ Exists fun a => LT.lt 1 a
    -/
  · exact ⟨y, h⟩
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

@[to_additive]
instance (priority := 100) LinearOrderedCommGroup.to_noMaxOrder [Nontrivial α] : NoMaxOrder α :=
  ⟨by
    /-
      α : Type u
      inst✝¹ : LinearOrderedCommGroup α
      a : α
      inst✝ : Nontrivial α
      ⊢ ∀ (a : α), Exists fun b => LT.lt a b
    -/
    obtain ⟨y, hy⟩ : ∃ a : α, 1 < a := exists_one_lt'
    /-
      case intro
      α : Type u
      inst✝¹ : LinearOrderedCommGroup α
      a : α
      inst✝ : Nontrivial α
      y : α
      hy : LT.lt 1 y
      ⊢ ∀ (a : α), Exists fun b => LT.lt a b
    -/
    exact fun a => ⟨a * y, lt_mul_of_one_lt_right' a hy⟩⟩
    /-
      🎉 no goals
    -/

-- see Note [lower instance priority]

@[to_additive]
instance (priority := 100) LinearOrderedCommGroup.to_noMinOrder [Nontrivial α] : NoMinOrder α :=
  ⟨by
    /-
      α : Type u
      inst✝¹ : LinearOrderedCommGroup α
      a : α
      inst✝ : Nontrivial α
      ⊢ ∀ (a : α), Exists fun b => LT.lt b a
    -/
    obtain ⟨y, hy⟩ : ∃ a : α, 1 < a := exists_one_lt'
    /-
      case intro
      α : Type u
      inst✝¹ : LinearOrderedCommGroup α
      a : α
      inst✝ : Nontrivial α
      y : α
      hy : LT.lt 1 y
      ⊢ ∀ (a : α), Exists fun b => LT.lt b a
    -/
    exact fun a => ⟨a / y, (div_lt_self_iff a).mpr hy⟩⟩
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) LinearOrderedCommGroup.toLinearOrderedCancelCommMonoid
    [LinearOrderedCommGroup α] : LinearOrderedCancelCommMonoid α :=
{ ‹LinearOrderedCommGroup α›, OrderedCommGroup.toOrderedCancelCommMonoid with }


@[to_additive (attr := simp)]
                                                /-
                                                  α : Type u
                                                  inst✝ : LinearOrderedCommGroup α
                                                  a : α
                                                  ⊢ Iff (LE.le (Inv.inv a) a) (LE.le 1 a)
                                                -/
theorem inv_le_self_iff : a⁻¹ ≤ a ↔ 1 ≤ a := by simp [inv_le_iff_one_le_mul']
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := simp)]
                                                /-
                                                  α : Type u
                                                  inst✝ : LinearOrderedCommGroup α
                                                  a : α
                                                  ⊢ Iff (LT.lt (Inv.inv a) a) (LT.lt 1 a)
                                                -/
theorem inv_lt_self_iff : a⁻¹ < a ↔ 1 < a := by simp [inv_lt_iff_one_lt_mul]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := simp)]
                                                /-
                                                  α : Type u
                                                  inst✝ : LinearOrderedCommGroup α
                                                  a : α
                                                  ⊢ Iff (LE.le a (Inv.inv a)) (LE.le a 1)
                                                -/
theorem le_inv_self_iff : a ≤ a⁻¹ ↔ a ≤ 1 := by simp [← not_iff_not]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := simp)]
                                                /-
                                                  α : Type u
                                                  inst✝ : LinearOrderedCommGroup α
                                                  a : α
                                                  ⊢ Iff (LT.lt a (Inv.inv a)) (LT.lt a 1)
                                                -/
theorem lt_inv_self_iff : a < a⁻¹ ↔ a < 1 := by simp [← not_iff_not]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := gcongr) neg_le_neg]
theorem inv_le_inv' : a ≤ b → b⁻¹ ≤ a⁻¹ :=
  inv_le_inv_iff.mpr


@[to_additive (attr := gcongr) neg_lt_neg]
theorem inv_lt_inv' : a < b → b⁻¹ < a⁻¹ :=
  inv_lt_inv_iff.mpr

--  The additive version is also a `linarith` lemma.

@[to_additive]
theorem inv_lt_one_of_one_lt : 1 < a → a⁻¹ < 1 :=
  inv_lt_one_iff_one_lt.mpr

--  The additive version is also a `linarith` lemma.

@[to_additive]
theorem inv_le_one_of_one_le : 1 ≤ a → a⁻¹ ≤ 1 :=
  inv_le_one'.mpr


@[to_additive neg_nonneg_of_nonpos]
theorem one_le_inv_of_le_one : a ≤ 1 → 1 ≤ a⁻¹ :=
  one_le_inv'.mpr


