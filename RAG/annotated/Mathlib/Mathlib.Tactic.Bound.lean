lemma mul_lt_mul_left_of_pos_of_lt {α : Type} {a b c : α} [Mul α] [Zero α] [Preorder α]
    [PosMulStrictMono α] [PosMulReflectLT α] (a0 : 0 < a) : b < c → a * b < a * c :=
  (mul_lt_mul_left a0).mpr


lemma mul_lt_mul_right_of_pos_of_lt {α : Type} {a b c : α} [Mul α] [Zero α] [Preorder α]
    [MulPosStrictMono α] [MulPosReflectLT α] (c0 : 0 < c) : a < b → a * c < b * c :=
  (mul_lt_mul_right c0).mpr


lemma Nat.cast_pos_of_pos {R : Type} [OrderedSemiring R] [Nontrivial R] {n : ℕ} :
    0 < n → 0 < (n : R) :=
  Nat.cast_pos.mpr


lemma Nat.one_le_cast_of_le {α : Type} [AddCommMonoidWithOne α] [PartialOrder α]
    [AddLeftMono α] [ZeroLEOneClass α]
    [CharZero α] {n : ℕ} : 1 ≤ n → 1 ≤ (n : α) :=
  Nat.one_le_cast.mpr


lemma le_max_of_le_left_or_le_right : a ≤ b ∨ a ≤ c → a ≤ max b c := le_max_iff.mpr

lemma lt_max_of_lt_left_or_lt_right : a < b ∨ a < c → a < max b c := lt_max_iff.mpr

lemma min_le_of_left_le_or_right_le : a ≤ c ∨ b ≤ c → min a b ≤ c := min_le_iff.mpr

lemma min_lt_of_left_lt_or_right_lt : a < c ∨ b < c → min a b < c := min_lt_iff.mpr

-- Register guessing rules

/-- Close numerical goals with `norm_num` -/
def boundNormNum : Aesop.RuleTac :=
  Aesop.SingleRuleTac.toRuleTac fun i => do
    let tac := do Mathlib.Meta.NormNum.elabNormNum .missing .missing .missing
    let goals ← Lean.Elab.Tactic.run i.goal tac |>.run'
    if !goals.isEmpty then failure
    return (#[], none, some .hundred)

/-- Close numerical and other goals with `linarith` -/
def boundLinarith : Aesop.RuleTac :=
  Aesop.SingleRuleTac.toRuleTac fun i => do
    Linarith.linarith false [] {} i.goal
    return (#[], none, some .hundred)

/-- Aesop configuration for `bound` -/
def boundConfig : Aesop.Options := {
  enableSimp := false
}


/-- `bound` tactic for proving inequalities via straightforward recursion on expression structure.

An example use case is

```
-- Calc example: A weak lower bound for `z ↦ z^2 + c`
lemma le_sqr_add {c z : ℂ} (cz : abs c ≤ abs z) (z3 : 3 ≤ abs z) :
    2 * abs z ≤ abs (z^2 + c) := by
  calc abs (z^2 + c)
    _ ≥ abs (z^2) - abs c := by bound
    _ ≥ abs (z^2) - abs z := by bound
    _ ≥ (abs z - 1) * abs z := by rw [mul_comm, mul_sub_one, ← pow_two, ← abs.map_pow]
    _ ≥ 2 * abs z := by bound
```

`bound` is built on top of `aesop`, and uses
1. Apply lemmas registered via the `@[bound]` attribute
2. Forward lemmas registered via the `@[bound_forward]` attribute
3. Local hypotheses from the context
4. Optionally: additional hypotheses provided as `bound [h₀, h₁]` or similar. These are added to the
   context as if by `have := hᵢ`.

The functionality of `bound` overlaps with `positivity` and `gcongr`, but can jump back and forth
between `0 ≤ x` and `x ≤ y`-type inequalities.  For example, `bound` proves
  `0 ≤ c → b ≤ a → 0 ≤ a * c - b * c`
by turning the goal into `b * c ≤ a * c`, then using `mul_le_mul_of_nonneg_right`.  `bound` also
contains lemmas for goals of the form `1 ≤ x, 1 < x, x ≤ 1, x < 1`.  Conversely, `gcongr` can prove
inequalities for more types of relations, supports all `positivity` functionality, and is likely
faster since it is more specialized (not built atop `aesop`). -/
syntax "bound " (" [" term,* "]")? : tactic

-- Plain `bound` elaboration, with no hypotheses

elab_rules : tactic
  | `(tactic| bound) => do
    let tac ← `(tactic| aesop (rule_sets := [Bound, -default]) (config := Bound.boundConfig))
    liftMetaTactic fun g ↦ do return (← Lean.Elab.runTactic g tac.raw).1

-- Rewrite `bound [h₀, h₁]` into `have := h₀, have := h₁, bound`, and similar

macro_rules
  | `(tactic| bound%$tk [$[$ts],*]) => do
    let haves ← ts.mapM fun (t : Term) => withRef t `(tactic| have := $t)
    `(tactic| ($haves;*; bound%$tk))

