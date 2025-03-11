/-- A canonically ordered additive monoid is an ordered commutative additive monoid
  in which the ordering coincides with the subtractibility relation,
  which is to say, `a ≤ b` iff there exists `c` with `b = a + c`.
  This is satisfied by the natural numbers, for example, but not
  the integers or other nontrivial `OrderedAddCommGroup`s. -/
class CanonicallyOrderedAddCommMonoid (α : Type*) extends OrderedAddCommMonoid α, OrderBot α where
  /-- For `a ≤ b`, there is a `c` so `b = a + c`. -/
  protected exists_add_of_le : ∀ {a b : α}, a ≤ b → ∃ c, b = a + c
  /-- For any `a` and `b`, `a ≤ a + b` -/
  protected le_self_add : ∀ a b : α, a ≤ a + b

-- see Note [lower instance priority]

/-- A canonically ordered monoid is an ordered commutative monoid
  in which the ordering coincides with the divisibility relation,
  which is to say, `a ≤ b` iff there exists `c` with `b = a * c`.
  Examples seem rare; it seems more likely that the `OrderDual`
  of a naturally-occurring lattice satisfies this than the lattice
  itself (for example, dual of the lattice of ideals of a PID or
  Dedekind domain satisfy this; collections of all things ≤ 1 seem to
  be more natural that collections of all things ≥ 1).
-/
@[to_additive]
class CanonicallyOrderedCommMonoid (α : Type*) extends OrderedCommMonoid α, OrderBot α where
  /-- For `a ≤ b`, there is a `c` so `b = a * c`. -/
  protected exists_mul_of_le : ∀ {a b : α}, a ≤ b → ∃ c, b = a * c
  /-- For any `a` and `b`, `a ≤ a * b` -/
  protected le_self_mul : ∀ a b : α, a ≤ a * b

-- see Note [lower instance priority]

@[to_additive]
instance (priority := 100) CanonicallyOrderedCommMonoid.existsMulOfLE (α : Type u)
    [h : CanonicallyOrderedCommMonoid α] : ExistsMulOfLE α :=
  { h with }


@[to_additive]
theorem le_self_mul : a ≤ a * c :=
  CanonicallyOrderedCommMonoid.le_self_mul _ _


@[to_additive]
theorem le_mul_self : a ≤ b * a := by
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    ⊢ LE.le a (HMul.hMul b a)
  -/
  rw [mul_comm]
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    ⊢ LE.le a (HMul.hMul a b)
  -/
  exact le_self_mul
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem self_le_mul_right (a b : α) : a ≤ a * b :=
  le_self_mul


@[to_additive (attr := simp)]
theorem self_le_mul_left (a b : α) : a ≤ b * a :=
  le_mul_self


@[to_additive]
theorem le_of_mul_le_left : a * b ≤ c → a ≤ c :=
  le_self_mul.trans


@[to_additive]
theorem le_of_mul_le_right : a * b ≤ c → b ≤ c :=
  le_mul_self.trans


@[to_additive]
theorem le_mul_of_le_left : a ≤ b → a ≤ b * c :=
  le_self_mul.trans'


@[to_additive]
theorem le_mul_of_le_right : a ≤ c → a ≤ b * c :=
  le_mul_self.trans'


@[to_additive]
theorem le_iff_exists_mul : a ≤ b ↔ ∃ c, b = a * c :=
  ⟨exists_mul_of_le, by
    /-
      α : Type u
      inst✝ : CanonicallyOrderedCommMonoid α
      a b : α
      ⊢ (Exists fun c => Eq b (HMul.hMul a c)) → LE.le a b
    -/
    rintro ⟨c, rfl⟩
    /-
      case intro
      α : Type u
      inst✝ : CanonicallyOrderedCommMonoid α
      a c : α
      ⊢ LE.le a (HMul.hMul a c)
    -/
    exact le_self_mul⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem le_iff_exists_mul' : a ≤ b ↔ ∃ c, b = c * a := by
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    ⊢ Iff (LE.le a b) (Exists fun c => Eq b (HMul.hMul c a))
  -/
  simp only [mul_comm _ a, le_iff_exists_mul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) zero_le]
theorem one_le (a : α) : 1 ≤ a :=
  le_iff_exists_mul.mpr ⟨a, (one_mul _).symm⟩


@[to_additive]
theorem bot_eq_one : (⊥ : α) = 1 :=
  le_antisymm bot_le (one_le ⊥)


@[to_additive] instance CanonicallyOrderedCommMonoid.toUniqueUnits : Unique αˣ where
  uniq a := Units.ext ((mul_eq_one_iff_of_one_le (α := α) (one_le _) <| one_le _).1 a.mul_inv).1


@[deprecated (since := "2024-07-24")] alias mul_eq_one_iff := mul_eq_one

@[deprecated (since := "2024-07-24")] alias add_eq_zero_iff := add_eq_zero


@[to_additive (attr := simp)]
theorem le_one_iff_eq_one : a ≤ 1 ↔ a = 1 :=
  (one_le a).le_iff_eq


@[to_additive]
theorem one_lt_iff_ne_one : 1 < a ↔ a ≠ 1 :=
  (one_le a).lt_iff_ne.trans ne_comm


@[to_additive]
theorem eq_one_or_one_lt (a : α) : a = 1 ∨ 1 < a := (one_le a).eq_or_lt.imp_left Eq.symm


@[to_additive]
lemma one_not_mem_iff {s : Set α} : 1 ∉ s ↔ ∀ x ∈ s, 1 < x :=
  bot_eq_one (α := α) ▸ bot_not_mem_iff


@[to_additive (attr := simp) add_pos_iff]
theorem one_lt_mul_iff : 1 < a * b ↔ 1 < a ∨ 1 < b := by
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    ⊢ Iff (LT.lt 1 (HMul.hMul a b)) (Or (LT.lt 1 a) (LT.lt 1 b))
  -/
  simp only [one_lt_iff_ne_one, Ne, mul_eq_one, not_and_or]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_one_lt_mul_of_lt (h : a < b) : ∃ (c : _) (_ : 1 < c), a * c = b := by
  /-
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    h : LT.lt a b
    ⊢ Exists fun c => Exists fun x => Eq (HMul.hMul a c) b
  -/
  obtain ⟨c, hc⟩ := le_iff_exists_mul.1 h.le
  /-
    case intro
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    h : LT.lt a b
    c : α
    hc : Eq b (HMul.hMul a c)
    ⊢ Exists fun c => Exists fun x => Eq (HMul.hMul a c) b
  -/
  refine ⟨c, one_lt_iff_ne_one.2 ?_, hc.symm⟩
  /-
    case intro
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    h : LT.lt a b
    c : α
    hc : Eq b (HMul.hMul a c)
    ⊢ Ne c 1
  -/
  rintro rfl
  /-
    case intro
    α : Type u
    inst✝ : CanonicallyOrderedCommMonoid α
    a b : α
    h : LT.lt a b
    hc : Eq b (HMul.hMul a 1)
    ⊢ False
  -/
  simp [hc, lt_irrefl] at h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_mul_left (h : a ≤ c) : a ≤ b * c :=
  calc
                    /-
                      α : Type u
                      inst✝ : CanonicallyOrderedCommMonoid α
                      a b c : α
                      h : LE.le a c
                      ⊢ Eq a (HMul.hMul 1 a)
                    -/
    a = 1 * a := by simp
                    /-
                      🎉 no goals
                    -/
    _ ≤ b * c := mul_le_mul' (one_le _) h


@[to_additive]
theorem le_mul_right (h : a ≤ b) : a ≤ b * c :=
  calc
                    /-
                      α : Type u
                      inst✝ : CanonicallyOrderedCommMonoid α
                      a b c : α
                      h : LE.le a b
                      ⊢ Eq a (HMul.hMul a 1)
                    -/
    a = a * 1 := by simp
                    /-
                      🎉 no goals
                    -/
    _ ≤ b * c := mul_le_mul' h (one_le _)


@[to_additive]
theorem lt_iff_exists_mul [MulLeftStrictMono α] : a < b ↔ ∃ c > 1, b = a * c := by
  /-
    α : Type u
    inst✝¹ : CanonicallyOrderedCommMonoid α
    a b : α
    inst✝ : MulLeftStrictMono α
    ⊢ Iff (LT.lt a b) (Exists fun c => And (GT.gt c 1) (Eq b (HMul.hMul a c)))
  -/
  rw [lt_iff_le_and_ne, le_iff_exists_mul, ← exists_and_right]
  /-
    α : Type u
    inst✝¹ : CanonicallyOrderedCommMonoid α
    a b : α
    inst✝ : MulLeftStrictMono α
    ⊢ Iff (Exists fun x => And (Eq b (HMul.hMul a x)) (Ne a b)) (Exists fun c => A …
  -/
  apply exists_congr
  /-
    case h
    α : Type u
    inst✝¹ : CanonicallyOrderedCommMonoid α
    a b : α
    inst✝ : MulLeftStrictMono α
    ⊢ ∀ (a_1 : α), Iff (And (Eq b (HMul.hMul a a_1)) (Ne a b)) (And (GT.gt a_1 1)  …
  -/
  intro c
  /-
    case h
    α : Type u
    inst✝¹ : CanonicallyOrderedCommMonoid α
    a b : α
    inst✝ : MulLeftStrictMono α
    c : α
    ⊢ Iff (And (Eq b (HMul.hMul a c)) (Ne a b)) (And (GT.gt c 1) (Eq b (HMul.hMul  …
  -/
  rw [and_comm, and_congr_left_iff, gt_iff_lt]
  /-
    case h
    α : Type u
    inst✝¹ : CanonicallyOrderedCommMonoid α
    a b : α
    inst✝ : MulLeftStrictMono α
    c : α
    ⊢ Eq b (HMul.hMul a c) → Iff (Ne a b) (LT.lt 1 c)
  -/
  rintro rfl
  /-
    case h
    α : Type u
    inst✝¹ : CanonicallyOrderedCommMonoid α
    a : α
    inst✝ : MulLeftStrictMono α
    c : α
    ⊢ Iff (Ne a (HMul.hMul a c)) (LT.lt 1 c)
  -/
  constructor
    /-
      case h.mp
      α : Type u
      inst✝¹ : CanonicallyOrderedCommMonoid α
      a : α
      inst✝ : MulLeftStrictMono α
      c : α
      ⊢ Ne a (HMul.hMul a c) → LT.lt 1 c
    -/
  · rw [one_lt_iff_ne_one]
    /-
      case h.mp
      α : Type u
      inst✝¹ : CanonicallyOrderedCommMonoid α
      a : α
      inst✝ : MulLeftStrictMono α
      c : α
      ⊢ Ne a (HMul.hMul a c) → Ne c 1
    -/
    apply mt
    /-
      case h.mp.h₁
      α : Type u
      inst✝¹ : CanonicallyOrderedCommMonoid α
      a : α
      inst✝ : MulLeftStrictMono α
      c : α
      ⊢ Eq c 1 → Eq a (HMul.hMul a c)
    -/
    rintro rfl
    /-
      case h.mp.h₁
      α : Type u
      inst✝¹ : CanonicallyOrderedCommMonoid α
      a : α
      inst✝ : MulLeftStrictMono α
      ⊢ Eq a (HMul.hMul a 1)
    -/
    rw [mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u
      inst✝¹ : CanonicallyOrderedCommMonoid α
      a : α
      inst✝ : MulLeftStrictMono α
      c : α
      ⊢ LT.lt 1 c → Ne a (HMul.hMul a c)
    -/
  · rw [← (self_le_mul_right a c).lt_iff_ne]
    /-
      case h.mpr
      α : Type u
      inst✝¹ : CanonicallyOrderedCommMonoid α
      a : α
      inst✝ : MulLeftStrictMono α
      c : α
      ⊢ LT.lt 1 c → LT.lt a (HMul.hMul a c)
    -/
    apply lt_mul_of_one_lt_right'
    /-
      🎉 no goals
    -/


theorem pos_of_gt {M : Type*} [CanonicallyOrderedAddCommMonoid M] {n m : M} (h : n < m) : 0 < m :=
  lt_of_le_of_lt (zero_le _) h


theorem pos {M} (a : M) [CanonicallyOrderedAddCommMonoid M] [NeZero a] : 0 < a :=
  (zero_le a).lt_of_ne <| NeZero.out.symm


theorem of_gt {M} [CanonicallyOrderedAddCommMonoid M] {x y : M} (h : x < y) : NeZero y :=
  of_pos <| pos_of_gt h

-- 1 < p is still an often-used `Fact`, due to `Nat.Prime` implying it, and it implying `Nontrivial`
-- on `ZMod`'s ring structure. We cannot just set this to be any `x < y`, else that becomes a
-- metavariable and it will hugely slow down typeclass inference.

instance (priority := 10) of_gt' {M : Type*} [CanonicallyOrderedAddCommMonoid M] [One M] {y : M}
  -- Porting note: Fact.out has different type signature from mathlib3
  [Fact (1 < y)] : NeZero y := of_gt <| @Fact.out (1 < y) _


/-- A canonically linear-ordered additive monoid is a canonically ordered additive monoid
    whose ordering is a linear order. -/
class CanonicallyLinearOrderedAddCommMonoid (α : Type*)
  extends CanonicallyOrderedAddCommMonoid α, LinearOrderedAddCommMonoid α


/-- A canonically linear-ordered monoid is a canonically ordered monoid
    whose ordering is a linear order. -/
@[to_additive]
class CanonicallyLinearOrderedCommMonoid (α : Type*)
  extends CanonicallyOrderedCommMonoid α, LinearOrderedCommMonoid α


@[to_additive]
instance (priority := 100) CanonicallyLinearOrderedCommMonoid.semilatticeSup : SemilatticeSup α :=
  { LinearOrder.toLattice with }


@[to_additive]
theorem min_mul_distrib (a b c : α) : min a (b * c) = min a (min a b * min a c) := by
  /-
    α : Type u
    inst✝ : CanonicallyLinearOrderedCommMonoid α
    a b c : α
    ⊢ Eq (Min.min a (HMul.hMul b c)) (Min.min a (HMul.hMul (Min.min a b) (Min.min  …
  -/
  rcases le_total a b with hb | hb
    /-
      case inl
      α : Type u
      inst✝ : CanonicallyLinearOrderedCommMonoid α
      a b c : α
      hb : LE.le a b
      ⊢ Eq (Min.min a (HMul.hMul b c)) (Min.min a (HMul.hMul (Min.min a b) (Min.min  …
    -/
  · simp [hb, le_mul_right]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : CanonicallyLinearOrderedCommMonoid α
      a b c : α
      hb : LE.le b a
      ⊢ Eq (Min.min a (HMul.hMul b c)) (Min.min a (HMul.hMul (Min.min a b) (Min.min  …
    -/
  · rcases le_total a c with hc | hc
      /-
        case inr.inl
        α : Type u
        inst✝ : CanonicallyLinearOrderedCommMonoid α
        a b c : α
        hb : LE.le b a
        hc : LE.le a c
        ⊢ Eq (Min.min a (HMul.hMul b c)) (Min.min a (HMul.hMul (Min.min a b) (Min.min  …
      -/
    · simp [hc, le_mul_left]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        α : Type u
        inst✝ : CanonicallyLinearOrderedCommMonoid α
        a b c : α
        hb : LE.le b a
        hc : LE.le c a
        ⊢ Eq (Min.min a (HMul.hMul b c)) (Min.min a (HMul.hMul (Min.min a b) (Min.min  …
      -/
    · simp [hb, hc]
      /-
        🎉 no goals
      -/


@[to_additive]
theorem min_mul_distrib' (a b c : α) : min (a * b) c = min (min a c * min b c) c := by
  /-
    α : Type u
    inst✝ : CanonicallyLinearOrderedCommMonoid α
    a b c : α
    ⊢ Eq (Min.min (HMul.hMul a b) c) (Min.min (HMul.hMul (Min.min a c) (Min.min b  …
  -/
  simpa [min_comm _ c] using min_mul_distrib c a b
  /-
    🎉 no goals
  -/


@[to_additive]
theorem one_min (a : α) : min 1 a = 1 :=
  min_eq_left (one_le a)


@[to_additive]
theorem min_one (a : α) : min a 1 = 1 :=
  min_eq_right (one_le a)


/-- In a linearly ordered monoid, we are happy for `bot_eq_one` to be a `@[simp]` lemma. -/
@[to_additive (attr := simp)
  "In a linearly ordered monoid, we are happy for `bot_eq_zero` to be a `@[simp]` lemma"]
theorem bot_eq_one' : (⊥ : α) = 1 :=
  bot_eq_one


