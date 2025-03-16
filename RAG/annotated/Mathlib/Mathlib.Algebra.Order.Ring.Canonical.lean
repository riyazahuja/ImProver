/-- A canonically ordered commutative semiring is an ordered, commutative semiring in which `a ≤ b`
iff there exists `c` with `b = a + c`. This is satisfied by the natural numbers, for example, but
not the integers or other ordered groups. -/
class CanonicallyOrderedCommSemiring (α : Type*) extends CanonicallyOrderedAddCommMonoid α,
    CommSemiring α where
  /-- No zero divisors. -/
  protected eq_zero_or_eq_zero_of_mul_eq_zero : ∀ {a b : α}, a * b = 0 → a = 0 ∨ b = 0


                                                   /-
                                                     α : Type u
                                                     inst✝¹ : CanonicallyOrderedCommSemiring α
                                                     a : α
                                                     inst✝ : Nontrivial α
                                                     ⊢ Odd a → LT.lt 0 a
                                                   -/
lemma Odd.pos [Nontrivial α] : Odd a → 0 < a := by rintro ⟨k, rfl⟩; simp [pos_iff_ne_zero]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance (priority := 100) toNoZeroDivisors : NoZeroDivisors α :=
  ⟨CanonicallyOrderedCommSemiring.eq_zero_or_eq_zero_of_mul_eq_zero⟩

-- see Note [lower instance priority]

instance (priority := 100) toMulLeftMono : MulLeftMono α := by
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommSemiring α
    a b c d : α
    ⊢ MulLeftMono α
  -/
  refine ⟨fun a b c h => ?_⟩; dsimp
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommSemiring α
    a✝ b✝ c✝ d a b c : α
    h : LE.le b c
    ⊢ LE.le (HMul.hMul a b) (HMul.hMul a c)
  -/
  rcases exists_add_of_le h with ⟨c, rfl⟩
  /-
    case intro
    α : Type u
    inst✝ : CanonicallyOrderedCommSemiring α
    a✝ b✝ c✝ d a b c : α
    h : LE.le b (HAdd.hAdd b c)
    ⊢ LE.le (HMul.hMul a b) (HMul.hMul a (HAdd.hAdd b c))
  -/
  rw [mul_add]
  /-
    case intro
    α : Type u
    inst✝ : CanonicallyOrderedCommSemiring α
    a✝ b✝ c✝ d a b c : α
    h : LE.le b (HAdd.hAdd b c)
    ⊢ LE.le (HMul.hMul a b) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
  -/
  apply self_le_add_right
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) toOrderedCommMonoid : OrderedCommMonoid α where
  mul_le_mul_left := fun _ _ => mul_le_mul_left'

-- see Note [lower instance priority]

instance (priority := 100) toOrderedCommSemiring : OrderedCommSemiring α :=
  { ‹CanonicallyOrderedCommSemiring α› with
    zero_le_one := zero_le _,
    mul_le_mul_of_nonneg_left := fun _ _ _ h _ => mul_le_mul_left' h _,
    mul_le_mul_of_nonneg_right := fun _ _ _ h _ => mul_le_mul_right' h _ }


@[simp]
protected theorem mul_pos : 0 < a * b ↔ 0 < a ∧ 0 < b := by
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommSemiring α
    a b : α
    ⊢ Iff (LT.lt 0 (HMul.hMul a b)) (And (LT.lt 0 a) (LT.lt 0 b))
  -/
  simp only [pos_iff_ne_zero, ne_eq, mul_eq_zero, not_or]
  /-
    🎉 no goals
  -/


lemma pow_pos (ha : 0 < a) (n : ℕ) : 0 < a ^ n := pos_iff_ne_zero.2 <| pow_ne_zero _ ha.ne'


protected lemma mul_lt_mul_of_lt_of_lt [PosMulStrictMono α] (hab : a < b) (hcd : c < d) :
    a * c < b * d := by
  -- TODO: This should be an instance but it currently times out
  /-
    α : Type u
    inst✝¹ : CanonicallyOrderedCommSemiring α
    a b c d : α
    inst✝ : PosMulStrictMono α
    hab : LT.lt a b
    hcd : LT.lt c d
    ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    α : Type u
    inst✝¹ : CanonicallyOrderedCommSemiring α
    a b c d : α
    inst✝ : PosMulStrictMono α
    hab : LT.lt a b
    hcd : LT.lt c d
    this : MulPosStrictMono α
    ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
  -/
  obtain rfl | hc := eq_zero_or_pos c
    /-
      case inl
      α : Type u
      inst✝¹ : CanonicallyOrderedCommSemiring α
      a b d : α
      inst✝ : PosMulStrictMono α
      hab : LT.lt a b
      this : MulPosStrictMono α
      hcd : LT.lt 0 d
      ⊢ LT.lt (HMul.hMul a 0) (HMul.hMul b d)
    -/
  · rw [mul_zero]
    /-
      case inl
      α : Type u
      inst✝¹ : CanonicallyOrderedCommSemiring α
      a b d : α
      inst✝ : PosMulStrictMono α
      hab : LT.lt a b
      this : MulPosStrictMono α
      hcd : LT.lt 0 d
      ⊢ LT.lt 0 (HMul.hMul b d)
    -/
    exact mul_pos ((zero_le _).trans_lt hab) hcd
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝¹ : CanonicallyOrderedCommSemiring α
      a b c d : α
      inst✝ : PosMulStrictMono α
      hab : LT.lt a b
      hcd : LT.lt c d
      this : MulPosStrictMono α
      hc : LT.lt 0 c
      ⊢ LT.lt (HMul.hMul a c) (HMul.hMul b d)
    -/
  · exact mul_lt_mul_of_pos' hab hcd hc ((zero_le _).trans_lt hab)
    /-
      🎉 no goals
    -/


protected theorem mul_tsub (h : AddLECancellable (a * c)) : a * (b - c) = a * b - a * c := by
  /-
    α : Type u
    inst✝³ : CanonicallyOrderedCommSemiring α
    a b c : α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    h : AddLECancellable (HMul.hMul a c)
    ⊢ Eq (HMul.hMul a (HSub.hSub b c)) (HSub.hSub (HMul.hMul a b) (HMul.hMul a c))
  -/
  cases' total_of (· ≤ ·) b c with hbc hcb
    /-
      case inl
      α : Type u
      inst✝³ : CanonicallyOrderedCommSemiring α
      a b c : α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
      h : AddLECancellable (HMul.hMul a c)
      hbc : LE.le b c
      ⊢ Eq (HMul.hMul a (HSub.hSub b c)) (HSub.hSub (HMul.hMul a b) (HMul.hMul a c))
    -/
  · rw [tsub_eq_zero_iff_le.2 hbc, mul_zero, tsub_eq_zero_iff_le.2 (mul_le_mul_left' hbc a)]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝³ : CanonicallyOrderedCommSemiring α
      a b c : α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
      h : AddLECancellable (HMul.hMul a c)
      hcb : LE.le c b
      ⊢ Eq (HMul.hMul a (HSub.hSub b c)) (HSub.hSub (HMul.hMul a b) (HMul.hMul a c))
    -/
  · apply h.eq_tsub_of_add_eq
    /-
      case inr
      α : Type u
      inst✝³ : CanonicallyOrderedCommSemiring α
      a b c : α
      inst✝² : Sub α
      inst✝¹ : OrderedSub α
      inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
      h : AddLECancellable (HMul.hMul a c)
      hcb : LE.le c b
      ⊢ Eq (HAdd.hAdd (HMul.hMul a (HSub.hSub b c)) (HMul.hMul a c)) (HMul.hMul a b)
    -/
    rw [← mul_add, tsub_add_cancel_of_le hcb]
    /-
      🎉 no goals
    -/


protected theorem tsub_mul (h : AddLECancellable (b * c)) : (a - b) * c = a * c - b * c := by
  /-
    α : Type u
    inst✝³ : CanonicallyOrderedCommSemiring α
    a b c : α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    h : AddLECancellable (HMul.hMul b c)
    ⊢ Eq (HMul.hMul (HSub.hSub a b) c) (HSub.hSub (HMul.hMul a c) (HMul.hMul b c))
  -/
  simp only [mul_comm _ c] at *
  /-
    α : Type u
    inst✝³ : CanonicallyOrderedCommSemiring α
    a b c : α
    inst✝² : Sub α
    inst✝¹ : OrderedSub α
    inst✝ : IsTotal α fun x1 x2 => LE.le x1 x2
    h : AddLECancellable (HMul.hMul c b)
    ⊢ Eq (HMul.hMul c (HSub.hSub a b)) (HSub.hSub (HMul.hMul c a) (HMul.hMul c b))
  -/
  exact h.mul_tsub
  /-
    🎉 no goals
  -/


theorem mul_tsub (a b c : α) : a * (b - c) = a * b - a * c :=
  Contravariant.AddLECancellable.mul_tsub


theorem tsub_mul (a b c : α) : (a - b) * c = a * c - b * c :=
  Contravariant.AddLECancellable.tsub_mul


                                                             /-
                                                               α : Type u
                                                               inst✝⁴ : CanonicallyOrderedCommSemiring α
                                                               inst✝³ : Sub α
                                                               inst✝² : OrderedSub α
                                                               inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
                                                               inst✝ : AddLeftReflectLE α
                                                               a b : α
                                                               ⊢ Eq (HMul.hMul a (HSub.hSub b 1)) (HSub.hSub (HMul.hMul a b) a)
                                                             -/
lemma mul_tsub_one (a b : α) : a * (b - 1) = a * b - a := by rw [mul_tsub, mul_one]
                                                             /-
                                                               🎉 no goals
                                                             -/

                                                             /-
                                                               α : Type u
                                                               inst✝⁴ : CanonicallyOrderedCommSemiring α
                                                               inst✝³ : Sub α
                                                               inst✝² : OrderedSub α
                                                               inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
                                                               inst✝ : AddLeftReflectLE α
                                                               a b : α
                                                               ⊢ Eq (HMul.hMul (HSub.hSub a 1) b) (HSub.hSub (HMul.hMul a b) b)
                                                             -/
lemma tsub_one_mul (a b : α) : (a - 1) * b = a * b - b := by rw [tsub_mul, one_mul]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The `tsub` version of `mul_self_sub_mul_self`. Notably, this holds for `Nat` and `NNReal`. -/
theorem mul_self_tsub_mul_self (a b : α) : a * a - b * b = (a + b) * (a - b) := by
  /-
    α : Type u
    inst✝⁴ : CanonicallyOrderedCommSemiring α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : AddLeftReflectLE α
    a b : α
    ⊢ Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul b b)) (HMul.hMul (HAdd.hAdd a b) (H …
  -/
  rw [mul_tsub, add_mul, add_mul, tsub_add_eq_tsub_tsub, mul_comm b a, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


/-- The `tsub` version of `sq_sub_sq`. Notably, this holds for `Nat` and `NNReal`. -/
theorem sq_tsub_sq (a b : α) : a ^ 2 - b ^ 2 = (a + b) * (a - b) := by
  /-
    α : Type u
    inst✝⁴ : CanonicallyOrderedCommSemiring α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : AddLeftReflectLE α
    a b : α
    ⊢ Eq (HSub.hSub (HPow.hPow a 2) (HPow.hPow b 2)) (HMul.hMul (HAdd.hAdd a b) (H …
  -/
  rw [sq, sq, mul_self_tsub_mul_self]
  /-
    🎉 no goals
  -/


theorem mul_self_tsub_one (a : α) : a * a - 1 = (a + 1) * (a - 1) := by
  /-
    α : Type u
    inst✝⁴ : CanonicallyOrderedCommSemiring α
    inst✝³ : Sub α
    inst✝² : OrderedSub α
    inst✝¹ : IsTotal α fun x1 x2 => LE.le x1 x2
    inst✝ : AddLeftReflectLE α
    a : α
    ⊢ Eq (HSub.hSub (HMul.hMul a a) 1) (HMul.hMul (HAdd.hAdd a 1) (HSub.hSub a 1))
  -/
  rw [← mul_self_tsub_mul_self, mul_one]
  /-
    🎉 no goals
  -/


