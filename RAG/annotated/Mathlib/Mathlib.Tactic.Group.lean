@[to_additive]
theorem zpow_trick {G : Type*} [Group G] (a b : G) (n m : ℤ) :
                                              /-
                                                G : Type u_1
                                                inst✝ : Group G
                                                a b : G
                                                n m : Int
                                                ⊢ Eq (HMul.hMul (HMul.hMul a (HPow.hPow b n)) (HPow.hPow b m)) (HMul.hMul a (H …
                                              -/
    a * b ^ n * b ^ m = a * b ^ (n + m) := by rw [mul_assoc, ← zpow_add]
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
theorem zpow_trick_one {G : Type*} [Group G] (a b : G) (m : ℤ) :
                                          /-
                                            G : Type u_1
                                            inst✝ : Group G
                                            a b : G
                                            m : Int
                                            ⊢ Eq (HMul.hMul (HMul.hMul a b) (HPow.hPow b m)) (HMul.hMul a (HPow.hPow b (HA …
                                          -/
    a * b * b ^ m = a * b ^ (m + 1) := by rw [mul_assoc, mul_self_zpow]
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem zpow_trick_one' {G : Type*} [Group G] (a b : G) (n : ℤ) :
                                          /-
                                            G : Type u_1
                                            inst✝ : Group G
                                            a b : G
                                            n : Int
                                            ⊢ Eq (HMul.hMul (HMul.hMul a (HPow.hPow b n)) b) (HMul.hMul a (HPow.hPow b (HA …
                                          -/
    a * b ^ n * b = a * b ^ (n + 1) := by rw [mul_assoc, mul_zpow_self]
                                          /-
                                            🎉 no goals
                                          -/


/-- Auxiliary tactic for the `group` tactic. Calls the simplifier only. -/
syntax (name := aux_group₁) "aux_group₁" (location)? : tactic


macro_rules
| `(tactic| aux_group₁ $[at $location]?) =>
  `(tactic| simp -decide -failIfUnchanged only
    [commutatorElement_def, mul_one, one_mul,
      ← zpow_neg_one, ← zpow_natCast, ← zpow_mul,
      Int.ofNat_add, Int.ofNat_mul,
      Int.mul_neg, Int.neg_mul, neg_neg,
      one_zpow, zpow_zero, zpow_one, mul_zpow_neg_one,
      ← mul_assoc,
      ← zpow_add, ← zpow_add_one, ← zpow_one_add, zpow_trick, zpow_trick_one, zpow_trick_one',
      tsub_self, sub_self, add_neg_cancel, neg_add_cancel]
  $[at $location]?)


/-- Auxiliary tactic for the `group` tactic. Calls `ring_nf` to normalize exponents. -/
syntax (name := aux_group₂) "aux_group₂" (location)? : tactic


macro_rules
| `(tactic| aux_group₂ $[at $location]?) =>
  `(tactic| ring_nf $[at $location]?)


/-- Tactic for normalizing expressions in multiplicative groups, without assuming
commutativity, using only the group axioms without any information about which group
is manipulated.

(For additive commutative groups, use the `abel` tactic instead.)

Example:
```lean
example {G : Type} [Group G] (a b c d : G) (h : c = (a*b^2)*((b*b)⁻¹*a⁻¹)*d) : a*c*d⁻¹ = a := by
  group at h -- normalizes `h` which becomes `h : c = d`
  rw [h]     -- the goal is now `a*d*d⁻¹ = a`
  group      -- which then normalized and closed
```
-/
syntax (name := group) "group" (location)? : tactic


macro_rules
| `(tactic| group $[$loc]?) =>
  `(tactic| repeat (fail_if_no_progress (aux_group₁ $[$loc]? <;> aux_group₂ $[$loc]?)))


