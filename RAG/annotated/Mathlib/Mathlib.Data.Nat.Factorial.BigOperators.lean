lemma monotone_factorial : Monotone factorial := fun _ _ => factorial_le


                                                       /-
                                                         α : Type u_1
                                                         s : Finset α
                                                         f : α → Nat
                                                         ⊢ LT.lt 0 (s.prod fun i => (f i).factorial)
                                                       -/
theorem prod_factorial_pos : 0 < ∏ i ∈ s, (f i)! := by positivity
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem prod_factorial_dvd_factorial_sum : (∏ i ∈ s, (f i)!) ∣ (∑ i ∈ s, f i)! := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Nat
    ⊢ Dvd.dvd (s.prod fun i => (f i).factorial) (s.sum fun i => f i).factorial
  -/
  induction' s using Finset.cons_induction_on with a s has ih
    /-
      case h₁
      α : Type u_1
      s : Finset α
      f : α → Nat
      ⊢ Dvd.dvd (EmptyCollection.emptyCollection.prod fun i => (f i).factorial) (Emp …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      s✝ : Finset α
      f : α → Nat
      a : α
      s : Finset α
      has : Not (Membership.mem s a)
      ih : Dvd.dvd (s.prod fun i => (f i).factorial) (s.sum fun i => f i).factorial
      ⊢ Dvd.dvd ((Finset.cons a s has).prod fun i => (f i).factorial) ((Finset.cons  …
    -/
  · rw [prod_cons, Finset.sum_cons]
    /-
      case h₂
      α : Type u_1
      s✝ : Finset α
      f : α → Nat
      a : α
      s : Finset α
      has : Not (Membership.mem s a)
      ih : Dvd.dvd (s.prod fun i => (f i).factorial) (s.sum fun i => f i).factorial
      ⊢ Dvd.dvd (HMul.hMul (f a).factorial (s.prod fun x => (f x).factorial)) (HAdd. …
    -/
    exact (mul_dvd_mul_left _ ih).trans (Nat.factorial_mul_factorial_dvd_factorial_add _ _)
    /-
      🎉 no goals
    -/


theorem ascFactorial_eq_prod_range (n : ℕ) : ∀ k, n.ascFactorial k = ∏ i ∈ range k, (n + i)
  | 0 => rfl
                /-
                  n k : Nat
                  ⊢ Eq (n.ascFactorial (HAdd.hAdd k 1)) ((Finset.range (HAdd.hAdd k 1)).prod fun …
                -/
  | k + 1 => by rw [ascFactorial, prod_range_succ, mul_comm, ascFactorial_eq_prod_range n k]
                /-
                  🎉 no goals
                -/


theorem descFactorial_eq_prod_range (n : ℕ) : ∀ k, n.descFactorial k = ∏ i ∈ range k, (n - i)
  | 0 => rfl
                /-
                  n k : Nat
                  ⊢ Eq (n.descFactorial (HAdd.hAdd k 1)) ((Finset.range (HAdd.hAdd k 1)).prod fu …
                -/
  | k + 1 => by rw [descFactorial, prod_range_succ, mul_comm, descFactorial_eq_prod_range n k]
                /-
                  🎉 no goals
                -/


