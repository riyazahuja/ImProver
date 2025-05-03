/-- Infinitely many natural numbers are equal to `d` mod `n`. -/
theorem frequently_modEq {n : ℕ} (h : n ≠ 0) (d : ℕ) : ∃ᶠ m in atTop, m ≡ d [MOD n] :=
  ((tendsto_add_atTop_nat d).comp (tendsto_id.nsmul_atTop h.bot_lt)).frequently <|
                                     /-
                                       n : Nat
                                       h : Ne n 0
                                       d m : Nat
                                       ⊢ n.ModEq (Function.comp (fun a => HAdd.hAdd a d) (fun x => HSMul.hSMul n (id  …
                                     -/
    Frequently.of_forall fun m => by simp [Nat.modEq_iff_dvd, ← sub_sub]
                                     /-
                                       🎉 no goals
                                     -/


theorem frequently_mod_eq {d n : ℕ} (h : d < n) : ∃ᶠ m in atTop, m % n = d := by
  /-
    d n : Nat
    h : LT.lt d n
    ⊢ Filter.Frequently (fun m => Eq (HMod.hMod m n) d) Filter.atTop
  -/
  simpa only [Nat.ModEq, mod_eq_of_lt h] using frequently_modEq h.ne_bot d
  /-
    🎉 no goals
  -/


theorem frequently_even : ∃ᶠ m : ℕ in atTop, Even m := by
  /-
    ⊢ Filter.Frequently (fun m => Even m) Filter.atTop
  -/
  simpa only [even_iff] using frequently_mod_eq zero_lt_two
  /-
    🎉 no goals
  -/


theorem frequently_odd : ∃ᶠ m : ℕ in atTop, Odd m := by
  /-
    ⊢ Filter.Frequently (fun m => Odd m) Filter.atTop
  -/
  simpa only [odd_iff] using frequently_mod_eq one_lt_two
  /-
    🎉 no goals
  -/


theorem Filter.nonneg_of_eventually_pow_nonneg {α : Type*} [LinearOrderedRing α] {a : α}
    (h : ∀ᶠ n in atTop, 0 ≤ a ^ (n : ℕ)) : 0 ≤ a :=
  let ⟨_n, ho, hn⟩ := (Nat.frequently_odd.and_eventually h).exists
  ho.pow_nonneg_iff.1 hn

