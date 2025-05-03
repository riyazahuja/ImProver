/-- **Chebyshev's Sum Inequality**: When `f` and `g` monovary together (eg they are both
monotone/antitone), the scalar product of their sum is less than the size of the set times their
scalar product. -/
theorem MonovaryOn.sum_smul_sum_le_card_smul_sum (hfg : MonovaryOn f g s) :
    (∑ i ∈ s, f i) • ∑ i ∈ s, g i ≤ #s • ∑ i ∈ s, f i • g i := by
  classical
  obtain ⟨σ, hσ, hs⟩ := s.countable_toSet.exists_cycleOn
  rw [← card_range #s, sum_smul_sum_eq_sum_perm hσ]
  exact sum_le_card_nsmul _ _ _ fun n _ ↦
    hfg.sum_smul_comp_perm_le_sum_smul fun x hx ↦ hs fun h ↦ hx <| IsFixedPt.perm_pow h _


/-- **Chebyshev's Sum Inequality**: When `f` and `g` antivary together (eg one is monotone, the
other is antitone), the scalar product of their sum is less than the size of the set times their
scalar product. -/
theorem AntivaryOn.card_smul_sum_le_sum_smul_sum (hfg : AntivaryOn f g s) :
    #s • ∑ i ∈ s, f i • g i ≤ (∑ i ∈ s, f i) • ∑ i ∈ s, g i :=
  hfg.dual_right.sum_smul_sum_le_card_smul_sum


/-- **Chebyshev's Sum Inequality**: When `f` and `g` monovary together (eg they are both
monotone/antitone), the scalar product of their sum is less than the size of the set times their
scalar product. -/
theorem Monovary.sum_smul_sum_le_card_smul_sum (hfg : Monovary f g) :
    (∑ i, f i) • ∑ i, g i ≤ Fintype.card ι • ∑ i, f i • g i :=
  (hfg.monovaryOn _).sum_smul_sum_le_card_smul_sum


/-- **Chebyshev's Sum Inequality**: When `f` and `g` antivary together (eg one is monotone, the
other is antitone), the scalar product of their sum is less than the size of the set times their
scalar product. -/
theorem Antivary.card_smul_sum_le_sum_smul_sum (hfg : Antivary f g) :
    Fintype.card ι • ∑ i, f i • g i ≤ (∑ i, f i) • ∑ i, g i :=
  (hfg.dual_right.monovaryOn _).sum_smul_sum_le_card_smul_sum


/-- **Chebyshev's Sum Inequality**: When `f` and `g` monovary together (eg they are both
monotone/antitone), the product of their sum is less than the size of the set times their scalar
product. -/
theorem MonovaryOn.sum_mul_sum_le_card_mul_sum (hfg : MonovaryOn f g s) :
    (∑ i ∈ s, f i) * ∑ i ∈ s, g i ≤ #s * ∑ i ∈ s, f i * g i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f g : ι → α
    hfg : MonovaryOn f g ↑s
    ⊢ LE.le (HMul.hMul (s.sum fun i => f i) (s.sum fun i => g i)) (HMul.hMul (↑s.c …
  -/
  rw [← nsmul_eq_mul]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f g : ι → α
    hfg : MonovaryOn f g ↑s
    ⊢ LE.le (HMul.hMul (s.sum fun i => f i) (s.sum fun i => g i)) (HSMul.hSMul s.c …
  -/
  exact hfg.sum_smul_sum_le_card_smul_sum
  /-
    🎉 no goals
  -/


/-- **Chebyshev's Sum Inequality**: When `f` and `g` antivary together (eg one is monotone, the
other is antitone), the product of their sum is greater than the size of the set times their scalar
product. -/
theorem AntivaryOn.card_mul_sum_le_sum_mul_sum (hfg : AntivaryOn f g s) :
    (#s : α) * ∑ i ∈ s, f i * g i ≤ (∑ i ∈ s, f i) * ∑ i ∈ s, g i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f g : ι → α
    hfg : AntivaryOn f g ↑s
    ⊢ LE.le (HMul.hMul (↑s.card) (s.sum fun i => HMul.hMul (f i) (g i))) (HMul.hMu …
  -/
  rw [← nsmul_eq_mul]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f g : ι → α
    hfg : AntivaryOn f g ↑s
    ⊢ LE.le (HSMul.hSMul s.card (s.sum fun i => HMul.hMul (f i) (g i))) (HMul.hMul …
  -/
  exact hfg.card_smul_sum_le_sum_smul_sum
  /-
    🎉 no goals
  -/


/-- Special case of **Jensen's inequality** for sums of powers. -/
lemma pow_sum_le_card_mul_sum_pow (hf : ∀ i ∈ s, 0 ≤ f i) :
    ∀ n, (∑ i ∈ s, f i) ^ (n + 1) ≤ (#s : α) ^ n * ∑ i ∈ s, f i ^ (n + 1)
            /-
              ι : Type u_1
              α : Type u_2
              inst✝¹ : LinearOrderedSemiring α
              inst✝ : ExistsAddOfLE α
              s : Finset ι
              f : ι → α
              hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
              ⊢ LE.le (HPow.hPow (s.sum fun i => f i) (HAdd.hAdd 0 1)) (HMul.hMul (HPow.hPow …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 =>
    calc
                                                        /-
                                                          ι : Type u_1
                                                          α : Type u_2
                                                          inst✝¹ : LinearOrderedSemiring α
                                                          inst✝ : ExistsAddOfLE α
                                                          s : Finset ι
                                                          f : ι → α
                                                          hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
                                                          n : Nat
                                                          ⊢ Eq (HPow.hPow (s.sum fun i => f i) (HAdd.hAdd (HAdd.hAdd n 1) 1)) (HMul.hMul …
                                                        -/
      _ = (∑ i ∈ s, f i) ^ (n + 1) * ∑ i ∈ s, f i := by rw [pow_succ]
                                                        /-
                                                          🎉 no goals
                                                        -/
      _ ≤ (#s ^ n * ∑ i ∈ s, f i ^ (n + 1)) * ∑ i ∈ s, f i := by
        /-
          ι : Type u_1
          α : Type u_2
          inst✝¹ : LinearOrderedSemiring α
          inst✝ : ExistsAddOfLE α
          s : Finset ι
          f : ι → α
          hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
          n : Nat
          ⊢ LE.le (HMul.hMul (HPow.hPow (s.sum fun i => f i) (HAdd.hAdd n 1)) (s.sum fun …
        -/
        gcongr
        /-
          case a0
          ι : Type u_1
          α : Type u_2
          inst✝¹ : LinearOrderedSemiring α
          inst✝ : ExistsAddOfLE α
          s : Finset ι
          f : ι → α
          hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
          n : Nat
          ⊢ LE.le 0 (s.sum fun i => f i)
        -/
        exacts [sum_nonneg hf, pow_sum_le_card_mul_sum_pow hf _]
        /-
          🎉 no goals
        -/
                                                                   /-
                                                                     ι : Type u_1
                                                                     α : Type u_2
                                                                     inst✝¹ : LinearOrderedSemiring α
                                                                     inst✝ : ExistsAddOfLE α
                                                                     s : Finset ι
                                                                     f : ι → α
                                                                     hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
                                                                     n : Nat
                                                                     ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑s.card) n) (s.sum fun i => HPow.hPow ( …
                                                                   -/
      _ = #s ^ n * ((∑ i ∈ s, f i ^ (n + 1)) * ∑ i ∈ s, f i) := by rw [mul_assoc]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
      _ ≤ #s ^ n * (#s * ∑ i ∈ s, f i ^ (n + 1) * f i) := by
        /-
          ι : Type u_1
          α : Type u_2
          inst✝¹ : LinearOrderedSemiring α
          inst✝ : ExistsAddOfLE α
          s : Finset ι
          f : ι → α
          hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
          n : Nat
          ⊢ LE.le (HMul.hMul (HPow.hPow (↑s.card) n) (HMul.hMul (s.sum fun i => HPow.hPo …
        -/
        gcongr _ * ?_
        /-
          case h
          ι : Type u_1
          α : Type u_2
          inst✝¹ : LinearOrderedSemiring α
          inst✝ : ExistsAddOfLE α
          s : Finset ι
          f : ι → α
          hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
          n : Nat
          ⊢ LE.le (HMul.hMul (s.sum fun i => HPow.hPow (f i) (HAdd.hAdd n 1)) (s.sum fun …
        -/
        exact ((monovaryOn_self ..).pow_left₀ hf _).sum_mul_sum_le_card_mul_sum
        /-
          🎉 no goals
        -/
                  /-
                    ι : Type u_1
                    α : Type u_2
                    inst✝¹ : LinearOrderedSemiring α
                    inst✝ : ExistsAddOfLE α
                    s : Finset ι
                    f : ι → α
                    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
                    n : Nat
                    ⊢ Eq (HMul.hMul (HPow.hPow (↑s.card) n) (HMul.hMul (↑s.card) (s.sum fun i => H …
                  -/
      _ = _ := by simp_rw [← mul_assoc, ← pow_succ]
                  /-
                    🎉 no goals
                  -/


/-- Special case of **Chebyshev's Sum Inequality** or the **Cauchy-Schwarz Inequality**: The square
of the sum is less than the size of the set times the sum of the squares. -/
theorem sq_sum_le_card_mul_sum_sq : (∑ i ∈ s, f i) ^ 2 ≤ #s * ∑ i ∈ s, f i ^ 2 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) 2) (HMul.hMul (↑s.card) (s.sum fun i = …
  -/
  simp_rw [sq]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    ⊢ LE.le (HMul.hMul (s.sum fun i => f i) (s.sum fun i => f i)) (HMul.hMul (↑s.c …
  -/
  exact (monovaryOn_self _ _).sum_mul_sum_le_card_mul_sum
  /-
    🎉 no goals
  -/


/-- **Chebyshev's Sum Inequality**: When `f` and `g` monovary together (eg they are both
monotone/antitone), the product of their sum is less than the size of the set times their scalar
product. -/
theorem Monovary.sum_mul_sum_le_card_mul_sum (hfg : Monovary f g) :
    (∑ i, f i) * ∑ i, g i ≤ Fintype.card ι * ∑ i, f i * g i :=
  (hfg.monovaryOn _).sum_mul_sum_le_card_mul_sum


/-- **Chebyshev's Sum Inequality**: When `f` and `g` antivary together (eg one is monotone, the
other is antitone), the product of their sum is less than the size of the set times their scalar
product. -/
theorem Antivary.card_mul_sum_le_sum_mul_sum (hfg : Antivary f g) :
    Fintype.card ι * ∑ i, f i * g i ≤ (∑ i, f i) * ∑ i, g i :=
  (hfg.antivaryOn _).card_mul_sum_le_sum_mul_sum


/-- Special case of **Jensen's inequality** for sums of powers. -/
lemma pow_sum_div_card_le_sum_pow (hf : ∀ i ∈ s, 0 ≤ f i) (n : ℕ) :
    (∑ i ∈ s, f i) ^ (n + 1) / #s ^ n ≤ ∑ i ∈ s, f i ^ (n + 1) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    n : Nat
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (s.sum fun i => f i) (HAdd.hAdd n 1)) (HPow.hPow …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      ι : Type u_1
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : ExistsAddOfLE α
      f : ι → α
      n : Nat
      hf : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → LE.le 0 (f i)
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (EmptyCollection.emptyCollection.sum fun i => f  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    n : Nat
    hs : s.Nonempty
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (s.sum fun i => f i) (HAdd.hAdd n 1)) (HPow.hPow …
  -/
  rw [div_le_iff₀' (by positivity)]
  /-
    case inr
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    n : Nat
    hs : s.Nonempty
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow …
  -/
  exact pow_sum_le_card_mul_sum_pow hf _
  /-
    🎉 no goals
  -/


theorem sum_div_card_sq_le_sum_sq_div_card :
    ((∑ i ∈ s, f i) / #s) ^ 2 ≤ (∑ i ∈ s, f i ^ 2) / #s := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    ⊢ LE.le (HPow.hPow (HDiv.hDiv (s.sum fun i => f i) ↑s.card) 2) (HDiv.hDiv (s.s …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      ι : Type u_1
      α : Type u_2
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : ExistsAddOfLE α
      f : ι → α
      ⊢ LE.le (HPow.hPow (HDiv.hDiv (EmptyCollection.emptyCollection.sum fun i => f  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  rw [div_pow, div_le_div_iff₀ (by positivity) (by positivity), sq (#s : α), mul_left_comm,
    ← mul_assoc]
  /-
    case inr
    ι : Type u_1
    α : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : ExistsAddOfLE α
    s : Finset ι
    f : ι → α
    hs : s.Nonempty
    ⊢ LE.le (HMul.hMul (HPow.hPow (s.sum fun i => f i) 2) ↑s.card) (HMul.hMul (HMu …
  -/
  exact mul_le_mul_of_nonneg_right sq_sum_le_card_mul_sum_sq (by positivity)
  /-
    🎉 no goals
  -/

