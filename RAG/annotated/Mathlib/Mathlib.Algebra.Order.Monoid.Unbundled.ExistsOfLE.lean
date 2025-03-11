/-- An `OrderedAddCommMonoid` with one-sided 'subtraction' in the sense that
if `a ≤ b`, then there is some `c` for which `a + c = b`. This is a weaker version
of the condition on canonical orderings defined by `CanonicallyOrderedAddCommMonoid`. -/
class ExistsAddOfLE (α : Type u) [Add α] [LE α] : Prop where
  /-- For `a ≤ b`, there is a `c` so `b = a + c`. -/
  exists_add_of_le : ∀ {a b : α}, a ≤ b → ∃ c : α, b = a + c


/-- An `OrderedCommMonoid` with one-sided 'division' in the sense that
if `a ≤ b`, there is some `c` for which `a * c = b`. This is a weaker version
of the condition on canonical orderings defined by `CanonicallyOrderedCommMonoid`. -/
@[to_additive]
class ExistsMulOfLE (α : Type u) [Mul α] [LE α] : Prop where
  /-- For `a ≤ b`, `a` left divides `b` -/
  exists_mul_of_le : ∀ {a b : α}, a ≤ b → ∃ c : α, b = a * c


@[to_additive]
instance (priority := 100) Group.existsMulOfLE (α : Type u) [Group α] [LE α] : ExistsMulOfLE α :=
  ⟨fun {a b} _ => ⟨a⁻¹ * b, (mul_inv_cancel_left _ _).symm⟩⟩


@[to_additive] lemma exists_one_le_mul_of_le [MulLeftReflectLE α] (h : a ≤ b) :
    ∃ c, 1 ≤ c ∧ a * c = b := by
  /-
    α : Type u
    inst✝³ : MulOneClass α
    inst✝² : Preorder α
    inst✝¹ : ExistsMulOfLE α
    a b : α
    inst✝ : MulLeftReflectLE α
    h : LE.le a b
    ⊢ Exists fun c => And (LE.le 1 c) (Eq (HMul.hMul a c) b)
  -/
  obtain ⟨c, rfl⟩ := exists_mul_of_le h; exact ⟨c, one_le_of_le_mul_right h, rfl⟩
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive] lemma exists_one_lt_mul_of_lt' [MulLeftReflectLT α] (h : a < b) :
    ∃ c, 1 < c ∧ a * c = b := by
  /-
    α : Type u
    inst✝³ : MulOneClass α
    inst✝² : Preorder α
    inst✝¹ : ExistsMulOfLE α
    a b : α
    inst✝ : MulLeftReflectLT α
    h : LT.lt a b
    ⊢ Exists fun c => And (LT.lt 1 c) (Eq (HMul.hMul a c) b)
  -/
  obtain ⟨c, rfl⟩ := exists_mul_of_le h.le; exact ⟨c, one_lt_of_lt_mul_right h, rfl⟩
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive] lemma le_iff_exists_one_le_mul [MulLeftMono α]
    [MulLeftReflectLE α] : a ≤ b ↔ ∃ c, 1 ≤ c ∧ a * c = b :=
                               /-
                                 α : Type u
                                 inst✝⁴ : MulOneClass α
                                 inst✝³ : Preorder α
                                 inst✝² : ExistsMulOfLE α
                                 a b : α
                                 inst✝¹ : MulLeftMono α
                                 inst✝ : MulLeftReflectLE α
                                 ⊢ (Exists fun c => And (LE.le 1 c) (Eq (HMul.hMul a c) b)) → LE.le a b
                               -/
  ⟨exists_one_le_mul_of_le, by rintro ⟨c, hc, rfl⟩; exact le_mul_of_one_le_right' hc⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive] lemma lt_iff_exists_one_lt_mul [MulLeftStrictMono α]
    [MulLeftReflectLT α] : a < b ↔ ∃ c, 1 < c ∧ a * c = b :=
                                /-
                                  α : Type u
                                  inst✝⁴ : MulOneClass α
                                  inst✝³ : Preorder α
                                  inst✝² : ExistsMulOfLE α
                                  a b : α
                                  inst✝¹ : MulLeftStrictMono α
                                  inst✝ : MulLeftReflectLT α
                                  ⊢ (Exists fun c => And (LT.lt 1 c) (Eq (HMul.hMul a c) b)) → LT.lt a b
                                -/
  ⟨exists_one_lt_mul_of_lt', by rintro ⟨c, hc, rfl⟩; exact lt_mul_of_one_lt_right' _ hc⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
theorem le_of_forall_one_lt_le_mul (h : ∀ ε : α, 1 < ε → a ≤ b * ε) : a ≤ b :=
  le_of_forall_le_of_dense fun x hxb => by
    /-
      α : Type u
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : Monoid α
      inst✝¹ : ExistsMulOfLE α
      inst✝ : MulLeftReflectLT α
      a b : α
      h : ∀ (ε : α), LT.lt 1 ε → LE.le a (HMul.hMul b ε)
      x : α
      hxb : LT.lt b x
      ⊢ LE.le a x
    -/
    obtain ⟨ε, rfl⟩ := exists_mul_of_le hxb.le
    /-
      case intro
      α : Type u
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : Monoid α
      inst✝¹ : ExistsMulOfLE α
      inst✝ : MulLeftReflectLT α
      a b : α
      h : ∀ (ε : α), LT.lt 1 ε → LE.le a (HMul.hMul b ε)
      ε : α
      hxb : LT.lt b (HMul.hMul b ε)
      ⊢ LE.le a (HMul.hMul b ε)
    -/
    exact h _ (one_lt_of_lt_mul_right hxb)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem le_of_forall_one_lt_lt_mul' (h : ∀ ε : α, 1 < ε → a < b * ε) : a ≤ b :=
  le_of_forall_one_lt_le_mul fun ε hε => (h ε hε).le


@[to_additive]
theorem le_iff_forall_one_lt_lt_mul' [MulLeftStrictMono α] :
    a ≤ b ↔ ∀ ε, 1 < ε → a < b * ε :=
  ⟨fun h _ => lt_mul_of_le_of_one_lt h, le_of_forall_one_lt_lt_mul'⟩


@[to_additive]
theorem le_iff_forall_one_lt_le_mul [MulLeftStrictMono α] :
    a ≤ b ↔ ∀ ε, 1 < ε → a ≤ b * ε :=
  ⟨fun h _ hε ↦ lt_mul_of_le_of_one_lt h hε |>.le, le_of_forall_one_lt_le_mul⟩


