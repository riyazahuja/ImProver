/--
The `rify` tactic is used to shift propositions from `ℕ`, `ℤ` or `ℚ` to `ℝ`.
Although less useful than its cousins `zify` and `qify`, it can be useful when your
goal or context already involves real numbers.

In the example below, assumption `hn` is about natural numbers, `hk` is about integers
and involves casting a natural number to `ℤ`, and the conclusion is about real numbers.
The proof uses `rify` to lift both assumptions to `ℝ` before calling `linarith`.
```
example {n : ℕ} {k : ℤ} (hn : 8 ≤ n) (hk : 2 * k ≤ n + 2) :
    (0 : ℝ) < n - k - 1 := by
  rify at hn hk /- Now have hn : 8 ≤ (n : ℝ)   hk : 2 * (k : ℝ) ≤ (n : ℝ) + 2-/
  linarith
```

`rify` makes use of the `@[zify_simps]`, `@[qify_simps]` and `@[rify_simps]` attributes to move
propositions, and the `push_cast` tactic to simplify the `ℝ`-valued expressions.

`rify` can be given extra lemmas to use in simplification. This is especially useful in the
presence of nat subtraction: passing `≤` arguments will allow `push_cast` to do more work.
```
example (a b c : ℕ) (h : a - b < c) (hab : b ≤ a) : a < b + c := by
  rify [hab] at h ⊢
  linarith
```
Note that `zify` or `qify` would work just as well in the above example (and `zify` is the natural
choice since it is enough to get rid of the pathological `ℕ` subtraction). -/
syntax (name := rify) "rify" (simpArgs)? (location)? : tactic


macro_rules
| `(tactic| rify $[[$simpArgs,*]]? $[at $location]?) =>
  let args := simpArgs.map (·.getElems) |>.getD #[]
  `(tactic|
    simp -decide only [zify_simps, qify_simps, rify_simps, push_cast, $args,*]
      $[at $location]?)


                                                                           /-
                                                                             a b : Rat
                                                                             ⊢ Iff (Eq a b) (Eq ↑a ↑b)
                                                                           -/
@[rify_simps] lemma ratCast_eq (a b : ℚ) : a = b ↔ (a : ℝ) = (b : ℝ) := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             a b : Rat
                                                                             ⊢ Iff (LE.le a b) (LE.le ↑a ↑b)
                                                                           -/
@[rify_simps] lemma ratCast_le (a b : ℚ) : a ≤ b ↔ (a : ℝ) ≤ (b : ℝ) := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             a b : Rat
                                                                             ⊢ Iff (LT.lt a b) (LT.lt ↑a ↑b)
                                                                           -/
@[rify_simps] lemma ratCast_lt (a b : ℚ) : a < b ↔ (a : ℝ) < (b : ℝ) := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                           /-
                                                                             a b : Rat
                                                                             ⊢ Iff (Ne a b) (Ne ↑a ↑b)
                                                                           -/
@[rify_simps] lemma ratCast_ne (a b : ℚ) : a ≠ b ↔ (a : ℝ) ≠ (b : ℝ) := by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[deprecated (since := "2024-04-17")]
alias rat_cast_ne := ratCast_ne


@[rify_simps] lemma ofNat_rat_real (a : ℕ) [a.AtLeastTwo] :
    ((ofNat(a) : ℚ) : ℝ) = (ofNat(a) : ℝ) := rfl


