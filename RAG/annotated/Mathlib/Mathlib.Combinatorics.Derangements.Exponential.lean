theorem numDerangements_tendsto_inv_e :
    Tendsto (fun n => (numDerangements n : ℝ) / n.factorial) atTop (𝓝 (Real.exp (-1))) := by
  -- we show that d(n)/n! is the partial sum of exp(-1), but offset by 1.
  -- this isn't entirely obvious, since we have to ensure that asc_factorial and
  -- factorial interact in the right way, e.g., that k ≤ n always
  /-
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv ↑(numDerangements n) ↑n.factorial) Filter …
  -/
  let s : ℕ → ℝ := fun n => ∑ k ∈ Finset.range n, (-1 : ℝ) ^ k / k.factorial
  suffices ∀ n : ℕ, (numDerangements n : ℝ) / n.factorial = s (n + 1) by
    simp_rw [this]
    -- shift the function by 1, and then use the fact that the partial sums
    -- converge to the infinite sum
    rw [tendsto_add_atTop_iff_nat
      (f := fun n => ∑ k ∈ Finset.range n, (-1 : ℝ) ^ k / k.factorial) 1]
    apply HasSum.tendsto_sum_nat
    -- there's no specific lemma for ℝ that ∑ x^k/k! sums to exp(x), but it's
    -- true in more general fields, so use that lemma
    rw [Real.exp_eq_exp_ℝ]
    exact expSeries_div_hasSum_exp ℝ (-1 : ℝ)
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    ⊢ ∀ (n : Nat), Eq (HDiv.hDiv ↑(numDerangements n) ↑n.factorial) (s (HAdd.hAdd  …
  -/
  intro n
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n : Nat
    ⊢ Eq (HDiv.hDiv ↑(numDerangements n) ↑n.factorial) (s (HAdd.hAdd n 1))
  -/
  rw [← Int.cast_natCast, numDerangements_sum]
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n : Nat
    ⊢ Eq (HDiv.hDiv ↑((Finset.range (HAdd.hAdd n 1)).sum fun k => HMul.hMul (HPow. …
  -/
  push_cast
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n : Nat
    ⊢ Eq (HDiv.hDiv ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HPow.h …
  -/
  rw [Finset.sum_div]
  -- get down to individual terms
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HDiv.hDiv (HMul.hMul (HPow.h …
  -/
  refine Finset.sum_congr (refl _) ?_
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (HDiv.hDiv …
  -/
  intro k hk
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) k) ↑((HAdd.hAdd k 1).ascFactorial ( …
  -/
  have h_le : k ≤ n := Finset.mem_range_succ_iff.mp hk
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    h_le : LE.le k n
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) k) ↑((HAdd.hAdd k 1).ascFactorial ( …
  -/
  rw [Nat.ascFactorial_eq_div, add_tsub_cancel_of_le h_le]
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    h_le : LE.le k n
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) k) ↑(HDiv.hDiv n.factorial k.factor …
  -/
  push_cast [Nat.factorial_dvd_factorial h_le]
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    h_le : LE.le k n
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) k) (HDiv.hDiv ↑n.factorial ↑k.facto …
  -/
  field_simp [Nat.factorial_ne_zero]
  /-
    s : Nat → Real := fun n => (Finset.range n).sum fun k => HDiv.hDiv (HPow.hPow  …
    n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    h_le : LE.le k n
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) k) ↑n.factorial) ↑k.factorial) (HMu …
  -/
  ring
  /-
    🎉 no goals
  -/

