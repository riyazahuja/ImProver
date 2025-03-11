/-- Local notation for `m a b`. -/
local notation a " <" m:51 "> " b => m a b


/-- `IsUnital m e` expresses that `e : X` is a left and right unit
for the binary operation `m : X → X → X`. -/
structure IsUnital (m : X → X → X) (e : X) extends Std.LawfulIdentity m e : Prop


@[to_additive EckmannHilton.AddZeroClass.IsUnital]
theorem MulOneClass.isUnital [_G : MulOneClass X] : IsUnital (· * ·) (1 : X) :=
  IsUnital.mk { left_id := MulOneClass.one_mul,
                right_id := MulOneClass.mul_one }


/-- If a type carries two unital binary operations that distribute over each other,
then they have the same unit elements.

In fact, the two operations are the same, and give a commutative monoid structure,
see `eckmann_hilton.CommMonoid`. -/
theorem one : e₁ = e₂ := by
  /-
    X : Type u
    m₁ m₂ : X → X → X
    e₁ e₂ : X
    h₁ : EckmannHilton.IsUnital m₁ e₁
    h₂ : EckmannHilton.IsUnital m₂ e₂
    distrib : ∀ (a b c d : X), Eq (m₁ (m₂ a b) (m₂ c d)) (m₂ (m₁ a c) (m₁ b d))
    ⊢ Eq e₁ e₂
  -/
  simpa only [h₁.left_id, h₁.right_id, h₂.left_id, h₂.right_id] using distrib e₂ e₁ e₁ e₂
  /-
    🎉 no goals
  -/


/-- If a type carries two unital binary operations that distribute over each other,
then these operations are equal.

In fact, they give a commutative monoid structure, see `eckmann_hilton.CommMonoid`. -/
theorem mul : m₁ = m₂ := by
  /-
    X : Type u
    m₁ m₂ : X → X → X
    e₁ e₂ : X
    h₁ : EckmannHilton.IsUnital m₁ e₁
    h₂ : EckmannHilton.IsUnital m₂ e₂
    distrib : ∀ (a b c d : X), Eq (m₁ (m₂ a b) (m₂ c d)) (m₂ (m₁ a c) (m₁ b d))
    ⊢ Eq m₁ m₂
  -/
  funext a b
  calc
    m₁ a b = m₁ (m₂ a e₁) (m₂ e₁ b) := by
      { simp only [one h₁ h₂ distrib, h₁.left_id, h₁.right_id, h₂.left_id, h₂.right_id] }
    _ = m₂ a b := by simp only [distrib, h₁.left_id, h₁.right_id, h₂.left_id, h₂.right_id]


/-- If a type carries two unital binary operations that distribute over each other,
then these operations are commutative.

In fact, they give a commutative monoid structure, see `eckmann_hilton.CommMonoid`. -/
theorem mul_comm : Std.Commutative m₂ :=
                 /-
                   X : Type u
                   m₁ m₂ : X → X → X
                   e₁ e₂ : X
                   h₁ : EckmannHilton.IsUnital m₁ e₁
                   h₂ : EckmannHilton.IsUnital m₂ e₂
                   distrib : ∀ (a b c d : X), Eq (m₁ (m₂ a b) (m₂ c d)) (m₂ (m₁ a c) (m₁ b d))
                   a b : X
                   ⊢ Eq (m₂ a b) (m₂ b a)
                 -/
  ⟨fun a b => by simpa [mul h₁ h₂ distrib, h₂.left_id, h₂.right_id] using distrib e₂ a b e₂⟩
                 /-
                   🎉 no goals
                 -/


/-- If a type carries two unital binary operations that distribute over each other,
then these operations are associative.

In fact, they give a commutative monoid structure, see `eckmann_hilton.CommMonoid`. -/
theorem mul_assoc : Std.Associative m₂ :=
                   /-
                     X : Type u
                     m₁ m₂ : X → X → X
                     e₁ e₂ : X
                     h₁ : EckmannHilton.IsUnital m₁ e₁
                     h₂ : EckmannHilton.IsUnital m₂ e₂
                     distrib : ∀ (a b c d : X), Eq (m₁ (m₂ a b) (m₂ c d)) (m₂ (m₁ a c) (m₁ b d))
                     a b c : X
                     ⊢ Eq (m₂ (m₂ a b) c) (m₂ a (m₂ b c))
                   -/
  ⟨fun a b c => by simpa [mul h₁ h₂ distrib, h₂.left_id, h₂.right_id] using distrib a b e₂ c⟩
                   /-
                     🎉 no goals
                   -/


/-- If a type carries a unital magma structure that distributes over a unital binary
operation, then the magma structure is a commutative monoid. -/
@[to_additive
      "If a type carries a unital additive magma structure that distributes over a unital binary
      operation, then the additive magma structure is a commutative additive monoid."]
abbrev commMonoid [h : MulOneClass X]
    (distrib : ∀ a b c d, ((a * b) <m₁> c * d) = (a <m₁> c) * b <m₁> d) : CommMonoid X :=
  { h with
      mul_comm := (mul_comm h₁ MulOneClass.isUnital distrib).comm,
      mul_assoc := (mul_assoc h₁ MulOneClass.isUnital distrib).assoc }


/-- If a type carries a group structure that distributes over a unital binary operation,
then the group is commutative. -/
@[to_additive
      "If a type carries an additive group structure that distributes over a unital binary
      operation, then the additive group is commutative."]
abbrev commGroup [G : Group X]
    (distrib : ∀ a b c d, ((a * b) <m₁> c * d) = (a <m₁> c) * b <m₁> d) : CommGroup X :=
  { EckmannHilton.commMonoid h₁ distrib, G with .. }


