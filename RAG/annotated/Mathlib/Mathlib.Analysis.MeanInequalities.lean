/-- **AM-GM inequality**: The geometric mean is less than or equal to the arithmetic mean, weighted
version for real-valued nonnegative functions. -/
theorem geom_mean_le_arith_mean_weighted (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) :
    ∏ i ∈ s, z i ^ w i ≤ ∑ i ∈ s, w i * z i := by
  -- If some number `z i` equals zero and has non-zero weight, then LHS is 0 and RHS is nonnegative.
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
  -/
  by_cases A : ∃ i ∈ s, z i = 0 ∧ w i ≠ 0
    /-
      case pos
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : Exists fun i => And (Membership.mem s i) (And (Eq (z i) 0) (Ne (w i) 0))
      ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
  · rcases A with ⟨i, his, hzi, hwi⟩
    /-
      case pos.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      i : ι
      his : Membership.mem s i
      hzi : Eq (z i) 0
      hwi : Ne (w i) 0
      ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
    rw [prod_eq_zero his]
      /-
        case pos.intro.intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        i : ι
        his : Membership.mem s i
        hzi : Eq (z i) 0
        hwi : Ne (w i) 0
        ⊢ LE.le 0 (s.sum fun i => HMul.hMul (w i) (z i))
      -/
    · exact sum_nonneg fun j hj => mul_nonneg (hw j hj) (hz j hj)
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        i : ι
        his : Membership.mem s i
        hzi : Eq (z i) 0
        hwi : Ne (w i) 0
        ⊢ Eq (HPow.hPow (z i) (w i)) 0
      -/
    · rw [hzi]
      /-
        case pos.intro.intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        i : ι
        his : Membership.mem s i
        hzi : Eq (z i) 0
        hwi : Ne (w i) 0
        ⊢ Eq (HPow.hPow 0 (w i)) 0
      -/
      exact zero_rpow hwi
      /-
        🎉 no goals
      -/
  -- If all numbers `z i` with non-zero weight are positive, then we apply Jensen's inequality
  -- for `exp` and numbers `log (z i)` with weights `w i`.
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : Not (Exists fun i => And (Membership.mem s i) (And (Eq (z i) 0) (Ne (w i)  …
      ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
  · simp only [not_exists, not_and, Ne, Classical.not_not] at A
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
      ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
    have := convexOn_exp.map_sum_le hw hw' fun i _ => Set.mem_univ <| log (z i)
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
      this : LE.le (Real.exp (s.sum fun i => HSMul.hSMul (w i) (Real.log (z i)))) (s …
      ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
    simp only [exp_sum, smul_eq_mul, mul_comm (w _) (log _)] at this
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
      this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
      ⊢ LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
    convert this using 1 <;> [apply prod_congr rfl;apply sum_congr rfl] <;> intro i hi
      /-
        case h.e'_3
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
        this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HPow.hPow (z i) (w i)) (Real.exp (HMul.hMul (Real.log (z i)) (w i)))
      -/
    · cases' eq_or_lt_of_le (hz i hi) with hz hz
        /-
          case h.e'_3.inl
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz✝ : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
          this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
          i : ι
          hi : Membership.mem s i
          hz : Eq 0 (z i)
          ⊢ Eq (HPow.hPow (z i) (w i)) (Real.exp (HMul.hMul (Real.log (z i)) (w i)))
        -/
      · simp [A i hi hz.symm]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3.inr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz✝ : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
          this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
          i : ι
          hi : Membership.mem s i
          hz : LT.lt 0 (z i)
          ⊢ Eq (HPow.hPow (z i) (w i)) (Real.exp (HMul.hMul (Real.log (z i)) (w i)))
        -/
      · exact rpow_def_of_pos hz _
        /-
          🎉 no goals
        -/
      /-
        case h.e'_4
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
        this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HMul.hMul (w i) (z i)) (HMul.hMul (w i) (Real.exp (Real.log (z i))))
      -/
    · cases' eq_or_lt_of_le (hz i hi) with hz hz
        /-
          case h.e'_4.inl
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz✝ : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
          this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
          i : ι
          hi : Membership.mem s i
          hz : Eq 0 (z i)
          ⊢ Eq (HMul.hMul (w i) (z i)) (HMul.hMul (w i) (Real.exp (Real.log (z i))))
        -/
      · simp [A i hi hz.symm]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_4.inr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz✝ : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Eq (w x) 0
          this : LE.le (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) (s. …
          i : ι
          hi : Membership.mem s i
          hz : LT.lt 0 (z i)
          ⊢ Eq (HMul.hMul (w i) (z i)) (HMul.hMul (w i) (Real.exp (Real.log (z i))))
        -/
      · rw [exp_log hz]
        /-
          🎉 no goals
        -/


/-- **AM-GM inequality**: The **geometric mean is less than or equal to the arithmetic mean. -/
theorem geom_mean_le_arith_mean {ι : Type*} (s : Finset ι) (w : ι → ℝ) (z : ι → ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hw' : 0 < ∑ i ∈ s, w i) (hz : ∀ i ∈ s, 0 ≤ z i) :
    (∏ i ∈ s, z i ^ w i) ^ (∑ i ∈ s, w i)⁻¹  ≤  (∑ i ∈ s, w i * z i) / (∑ i ∈ s, w i) := by
  /-
    ι : Type u_1
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : LT.lt 0 (s.sum fun i => w i)
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    ⊢ LE.le (HPow.hPow (s.prod fun i => HPow.hPow (z i) (w i)) (Inv.inv (s.sum fun …
  -/
  convert geom_mean_le_arith_mean_weighted s (fun i => (w i) / ∑ i ∈ s, w i) z ?_ ?_ hz using 2
    /-
      case h.e'_3
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ Eq (HPow.hPow (s.prod fun i => HPow.hPow (z i) (w i)) (Inv.inv (s.sum fun i  …
    -/
  · rw [← finset_prod_rpow _ _ (fun i hi => rpow_nonneg (hz _ hi) _) _]
    /-
      case h.e'_3
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ Eq (s.prod fun i => HPow.hPow (HPow.hPow (z i) (w i)) (Inv.inv (s.sum fun i  …
    -/
    refine Finset.prod_congr rfl (fun _ ih => ?_)
    /-
      case h.e'_3
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      x✝ : ι
      ih : Membership.mem s x✝
      ⊢ Eq (HPow.hPow (HPow.hPow (z x✝) (w x✝)) (Inv.inv (s.sum fun i => w i))) (HPo …
    -/
    rw [div_eq_mul_inv, rpow_mul (hz _ ih)]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ Eq (HDiv.hDiv (s.sum fun i => HMul.hMul (w i) (z i)) (s.sum fun i => w i)) ( …
    -/
  · simp_rw [div_eq_mul_inv, mul_assoc, mul_comm, ← mul_assoc, ← Finset.sum_mul, mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case convert_1
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ ∀ (i : ι), Membership.mem s i → LE.le 0 ((fun i => HDiv.hDiv (w i) (s.sum fu …
    -/
  · exact fun _ hi => div_nonneg (hw _ hi) (le_of_lt hw')
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ Eq (s.sum fun i => (fun i => HDiv.hDiv (w i) (s.sum fun i => w i)) i) 1
    -/
  · simp_rw [div_eq_mul_inv, ← Finset.sum_mul]
    /-
      case convert_2
      ι : Type u_1
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ Eq (HMul.hMul (s.sum fun i => w i) (Inv.inv (s.sum fun i => w i))) 1
    -/
    exact mul_inv_cancel₀ (by linarith)
    /-
      🎉 no goals
    -/


theorem geom_mean_weighted_of_constant (w z : ι → ℝ) (x : ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) (hx : ∀ i ∈ s, w i ≠ 0 → z i = x) :
    ∏ i ∈ s, z i ^ w i = x :=
  calc
    ∏ i ∈ s, z i ^ w i = ∏ i ∈ s, x ^ w i := by
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        ⊢ Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.prod fun i => HPow.hPow x (w i))
      -/
      refine prod_congr rfl fun i hi => ?_
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HPow.hPow (z i) (w i)) (HPow.hPow x (w i))
      -/
      rcases eq_or_ne (w i) 0 with h₀ | h₀
        /-
          case inl
          ι : Type u
          s : Finset ι
          w z : ι → Real
          x : Real
          hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
          i : ι
          hi : Membership.mem s i
          h₀ : Eq (w i) 0
          ⊢ Eq (HPow.hPow (z i) (w i)) (HPow.hPow x (w i))
        -/
      · rw [h₀, rpow_zero, rpow_zero]
        /-
          🎉 no goals
        -/
        /-
          case inr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          x : Real
          hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
          i : ι
          hi : Membership.mem s i
          h₀ : Ne (w i) 0
          ⊢ Eq (HPow.hPow (z i) (w i)) (HPow.hPow x (w i))
        -/
      · rw [hx i hi h₀]
        /-
          🎉 no goals
        -/
    _ = x := by
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        ⊢ Eq (s.prod fun i => HPow.hPow x (w i)) x
      -/
      rw [← rpow_sum_of_nonneg _ hw, hw', rpow_one]
      have : (∑ i ∈ s, w i) ≠ 0 := by
        rw [hw']
        exact one_ne_zero
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        this : Ne (s.sum fun i => w i) 0
        ⊢ LE.le 0 x
      -/
      obtain ⟨i, his, hi⟩ := exists_ne_zero_of_sum_ne_zero this
      /-
        case intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        this : Ne (s.sum fun i => w i) 0
        i : ι
        his : Membership.mem s i
        hi : Ne (w i) 0
        ⊢ LE.le 0 x
      -/
      rw [← hx i his hi]
      /-
        case intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        this : Ne (s.sum fun i => w i) 0
        i : ι
        his : Membership.mem s i
        hi : Ne (w i) 0
        ⊢ LE.le 0 (z i)
      -/
      exact hz i his
      /-
        🎉 no goals
      -/


theorem arith_mean_weighted_of_constant (w z : ι → ℝ) (x : ℝ) (hw' : ∑ i ∈ s, w i = 1)
    (hx : ∀ i ∈ s, w i ≠ 0 → z i = x) : ∑ i ∈ s, w i * z i = x :=
  calc
    ∑ i ∈ s, w i * z i = ∑ i ∈ s, w i * x := by
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw' : Eq (s.sum fun i => w i) 1
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        ⊢ Eq (s.sum fun i => HMul.hMul (w i) (z i)) (s.sum fun i => HMul.hMul (w i) x)
      -/
      refine sum_congr rfl fun i hi => ?_
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        x : Real
        hw' : Eq (s.sum fun i => w i) 1
        hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HMul.hMul (w i) (z i)) (HMul.hMul (w i) x)
      -/
      rcases eq_or_ne (w i) 0 with hwi | hwi
        /-
          case inl
          ι : Type u
          s : Finset ι
          w z : ι → Real
          x : Real
          hw' : Eq (s.sum fun i => w i) 1
          hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
          i : ι
          hi : Membership.mem s i
          hwi : Eq (w i) 0
          ⊢ Eq (HMul.hMul (w i) (z i)) (HMul.hMul (w i) x)
        -/
      · rw [hwi, zero_mul, zero_mul]
        /-
          🎉 no goals
        -/
        /-
          case inr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          x : Real
          hw' : Eq (s.sum fun i => w i) 1
          hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
          i : ι
          hi : Membership.mem s i
          hwi : Ne (w i) 0
          ⊢ Eq (HMul.hMul (w i) (z i)) (HMul.hMul (w i) x)
        -/
      · rw [hx i hi hwi]
        /-
          🎉 no goals
        -/
                /-
                  ι : Type u
                  s : Finset ι
                  w z : ι → Real
                  x : Real
                  hw' : Eq (s.sum fun i => w i) 1
                  hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
                  ⊢ Eq (s.sum fun i => HMul.hMul (w i) x) x
                -/
    _ = x := by rw [← sum_mul, hw', one_mul]
                /-
                  🎉 no goals
                -/


theorem geom_mean_eq_arith_mean_weighted_of_constant (w z : ι → ℝ) (x : ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) (hx : ∀ i ∈ s, w i ≠ 0 → z i = x) :
    ∏ i ∈ s, z i ^ w i = ∑ i ∈ s, w i * z i := by
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    x : Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    hx : ∀ (i : ι), Membership.mem s i → Ne (w i) 0 → Eq (z i) x
    ⊢ Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i) ( …
  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  rw [geom_mean_weighted_of_constant, arith_mean_weighted_of_constant] <;> assumption
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- **AM-GM inequality - equality condition**: This theorem provides the equality condition for the
*positive* weighted version of the AM-GM inequality for real-valued nonnegative functions. -/
theorem geom_mean_eq_arith_mean_weighted_iff' (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 < w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) :
    ∏ i ∈ s, z i ^ w i = ∑ i ∈ s, w i * z i ↔ ∀ j ∈ s, z j = ∑ i ∈ s, w i * z i := by
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
  -/
  by_cases A : ∃ i ∈ s, z i = 0 ∧ w i ≠ 0
    /-
      case pos
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : Exists fun i => And (Membership.mem s i) (And (Eq (z i) 0) (Ne (w i) 0))
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
  · rcases A with ⟨i, his, hzi, hwi⟩
    /-
      case pos.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      i : ι
      his : Membership.mem s i
      hzi : Eq (z i) 0
      hwi : Ne (w i) 0
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
    rw [prod_eq_zero his]
      /-
        case pos.intro.intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        i : ι
        his : Membership.mem s i
        hzi : Eq (z i) 0
        hwi : Ne (w i) 0
        ⊢ Iff (Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))) (∀ (j : ι), Membership.mem …
      -/
    · constructor
        /-
          case pos.intro.intro.intro.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          ⊢ Eq 0 (s.sum fun i => HMul.hMul (w i) (z i)) → ∀ (j : ι), Membership.mem s j  …
        -/
      · intro h
        /-
          case pos.intro.intro.intro.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))
          ⊢ ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) (z  …
        -/
        rw [← h]
        /-
          case pos.intro.intro.intro.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))
          ⊢ ∀ (j : ι), Membership.mem s j → Eq (z j) 0
        -/
        intro j hj
        /-
          case pos.intro.intro.intro.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))
          j : ι
          hj : Membership.mem s j
          ⊢ Eq (z j) 0
        -/
        apply eq_zero_of_ne_zero_of_mul_left_eq_zero (ne_of_lt (hw j hj)).symm
        /-
          case pos.intro.intro.intro.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))
          j : ι
          hj : Membership.mem s j
          ⊢ Eq (HMul.hMul (w j) (z j)) 0
        -/
        apply (sum_eq_zero_iff_of_nonneg ?_).mp h.symm j hj
        /-
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))
          j : ι
          hj : Membership.mem s j
          ⊢ ∀ (i : ι), Membership.mem s i → LE.le 0 (HMul.hMul (w i) (z i))
        -/
        exact fun i hi => (mul_nonneg_iff_of_pos_left (hw i hi)).mpr (hz i hi)
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.intro.intro.mpr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          ⊢ (∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) (z …
        -/
      · intro h
        /-
          case pos.intro.intro.intro.mpr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
          ⊢ Eq 0 (s.sum fun i => HMul.hMul (w i) (z i))
        -/
        convert h i his
        /-
          case h.e'_2
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          i : ι
          his : Membership.mem s i
          hzi : Eq (z i) 0
          hwi : Ne (w i) 0
          h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
          ⊢ Eq 0 (z i)
        -/
        exact hzi.symm
        /-
          🎉 no goals
        -/
      /-
        case pos.intro.intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        i : ι
        his : Membership.mem s i
        hzi : Eq (z i) 0
        hwi : Ne (w i) 0
        ⊢ Eq (HPow.hPow (z i) (w i)) 0
      -/
    · rw [hzi]
      /-
        case pos.intro.intro.intro
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        i : ι
        his : Membership.mem s i
        hzi : Eq (z i) 0
        hwi : Ne (w i) 0
        ⊢ Eq (HPow.hPow 0 (w i)) 0
      -/
      exact zero_rpow hwi
      /-
        🎉 no goals
      -/
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : Not (Exists fun i => And (Membership.mem s i) (And (Eq (z i) 0) (Ne (w i)  …
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
  · simp only [not_exists, not_and] at A
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
    have hz' := fun i h => lt_of_le_of_ne (hz i h) (fun a => (A i h a.symm) (ne_of_gt (hw i h)))
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
      hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
    have := strictConvexOn_exp.map_sum_eq_iff hw hw' fun i _ => Set.mem_univ <| log (z i)
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
      hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      this : Iff (Eq (Real.exp (s.sum fun i => HSMul.hSMul (w i) (Real.log (z i))))  …
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
    simp only [exp_sum, smul_eq_mul, mul_comm (w _) (log _)] at this
    /-
      case neg
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
      hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
      ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
    -/
    convert this using 1
    · apply Eq.congr <;>
      [apply prod_congr rfl; apply sum_congr rfl] <;>
      /-
        case h.e'_1.a.h₁
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
        hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
        this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
        ⊢ ∀ (x : ι), Membership.mem s x → Eq (HPow.hPow (z x) (w x)) (Real.exp (HMul.h …
      -/
      intro i hi <;>
      /-
        case h.e'_1.a.h₁
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
        hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
        this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HPow.hPow (z i) (w i)) (Real.exp (HMul.hMul (Real.log (z i)) (w i)))
      -/
      /-
        🎉 no goals
      -/
      simp only [exp_mul, exp_log (hz' i hi)]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.a
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
        hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
        this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
        ⊢ Iff (∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i …
      -/
    · constructor <;> intro h j hj
        /-
          case h.e'_2.a.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
          j : ι
          hj : Membership.mem s j
          ⊢ Eq (Real.log (z j)) (s.sum fun x => HMul.hMul (Real.log (z x)) (w x))
        -/
      · rw [← arith_mean_weighted_of_constant s w _ (log (z j)) hw' fun i _ => congrFun rfl]
        /-
          case h.e'_2.a.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
          j : ι
          hj : Membership.mem s j
          ⊢ Eq (s.sum fun i => HMul.hMul (w i) (Real.log (z j))) (s.sum fun x => HMul.hM …
        -/
        apply sum_congr rfl
        /-
          case h.e'_2.a.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
          j : ι
          hj : Membership.mem s j
          ⊢ ∀ (x : ι), Membership.mem s x → Eq (HMul.hMul (w x) (Real.log (z j))) (HMul. …
        -/
        intro x hx
        /-
          case h.e'_2.a.mp
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
          j : ι
          hj : Membership.mem s j
          x : ι
          hx : Membership.mem s x
          ⊢ Eq (HMul.hMul (w x) (Real.log (z j))) (HMul.hMul (Real.log (z x)) (w x))
        -/
        simp only [mul_comm, h j hj, h x hx]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.a.mpr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (Real.log (z j)) (s.sum fun x => HMul.h …
          j : ι
          hj : Membership.mem s j
          ⊢ Eq (z j) (s.sum fun i => HMul.hMul (w i) (z i))
        -/
      · rw [← arith_mean_weighted_of_constant s w _ (z j) hw' fun i _ => congrFun rfl]
        /-
          case h.e'_2.a.mpr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (Real.log (z j)) (s.sum fun x => HMul.h …
          j : ι
          hj : Membership.mem s j
          ⊢ Eq (s.sum fun i => HMul.hMul (w i) (z j)) (s.sum fun i => HMul.hMul (w i) (z …
        -/
        apply sum_congr rfl
        /-
          case h.e'_2.a.mpr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (Real.log (z j)) (s.sum fun x => HMul.h …
          j : ι
          hj : Membership.mem s j
          ⊢ ∀ (x : ι), Membership.mem s x → Eq (HMul.hMul (w x) (z j)) (HMul.hMul (w x)  …
        -/
        intro x hx
        /-
          case h.e'_2.a.mpr
          ι : Type u
          s : Finset ι
          w z : ι → Real
          hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
          hw' : Eq (s.sum fun i => w i) 1
          hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
          A : ∀ (x : ι), Membership.mem s x → Eq (z x) 0 → Not (Ne (w x) 0)
          hz' : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
          this : Iff (Eq (s.prod fun x => Real.exp (HMul.hMul (Real.log (z x)) (w x))) ( …
          h : ∀ (j : ι), Membership.mem s j → Eq (Real.log (z j)) (s.sum fun x => HMul.h …
          j : ι
          hj : Membership.mem s j
          x : ι
          hx : Membership.mem s x
          ⊢ Eq (HMul.hMul (w x) (z j)) (HMul.hMul (w x) (z x))
        -/
        simp only [log_injOn_pos (hz' j hj) (hz' x hx), h j hj, h x hx]
        /-
          🎉 no goals
        -/


/-- **AM-GM inequality - equality condition**: This theorem provides the equality condition for the
weighted version of the AM-GM inequality for real-valued nonnegative functions. -/
theorem geom_mean_eq_arith_mean_weighted_iff (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) :
    ∏ i ∈ s, z i ^ w i = ∑ i ∈ s, w i * z i ↔ ∀ j ∈ s, w j ≠ 0 → z j = ∑ i ∈ s, w i * z i := by
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
  -/
  have h (i) (_ : i ∈ s) : w i * z i ≠ 0 → w i ≠ 0 := by apply left_ne_zero_of_mul
  have h' (i) (_ : i ∈ s) : z i ^ w i ≠ 1 → w i ≠ 0 := by
    by_contra!
    obtain ⟨h1, h2⟩ := this
    simp only [h2, rpow_zero, ne_self_iff_false] at h1
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    h : ∀ (i : ι), Membership.mem s i → Ne (HMul.hMul (w i) (z i)) 0 → Ne (w i) 0
    h' : ∀ (i : ι), Membership.mem s i → Ne (HPow.hPow (z i) (w i)) 1 → Ne (w i) 0
    ⊢ Iff (Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
  -/
  rw [← sum_filter_of_ne h, ← prod_filter_of_ne h', geom_mean_eq_arith_mean_weighted_iff']
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      h : ∀ (i : ι), Membership.mem s i → Ne (HMul.hMul (w i) (z i)) 0 → Ne (w i) 0
      h' : ∀ (i : ι), Membership.mem s i → Ne (HPow.hPow (z i) (w i)) 1 → Ne (w i) 0
      ⊢ Iff (∀ (j : ι), Membership.mem (Finset.filter (fun x => Ne (w x) 0) s) j → E …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hw
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      h : ∀ (i : ι), Membership.mem s i → Ne (HMul.hMul (w i) (z i)) 0 → Ne (w i) 0
      h' : ∀ (i : ι), Membership.mem s i → Ne (HPow.hPow (z i) (w i)) 1 → Ne (w i) 0
      ⊢ ∀ (i : ι), Membership.mem (Finset.filter (fun x => Ne (w x) 0) s) i → LT.lt  …
    -/
  · simp +contextual [(hw _ _).gt_iff_ne]
    /-
      🎉 no goals
    -/
    /-
      case hw'
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      h : ∀ (i : ι), Membership.mem s i → Ne (HMul.hMul (w i) (z i)) 0 → Ne (w i) 0
      h' : ∀ (i : ι), Membership.mem s i → Ne (HPow.hPow (z i) (w i)) 1 → Ne (w i) 0
      ⊢ Eq ((Finset.filter (fun x => Ne (w x) 0) s).sum fun i => w i) 1
    -/
  · rwa [sum_filter_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case hz
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LE.le 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      h : ∀ (i : ι), Membership.mem s i → Ne (HMul.hMul (w i) (z i)) 0 → Ne (w i) 0
      h' : ∀ (i : ι), Membership.mem s i → Ne (HPow.hPow (z i) (w i)) 1 → Ne (w i) 0
      ⊢ ∀ (i : ι), Membership.mem (Finset.filter (fun x => Ne (w x) 0) s) i → LE.le  …
    -/
  · simp_all only [ne_eq, mul_eq_zero, not_or, not_false_eq_true, and_imp, implies_true, mem_filter]
    /-
      🎉 no goals
    -/


/-- **AM-GM inequality - strict inequality condition**: This theorem provides the strict inequality
condition for the *positive* weighted version of the AM-GM inequality for real-valued nonnegative
functions. -/
theorem geom_mean_lt_arith_mean_weighted_iff_of_pos (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 < w i)
    (hw' : ∑ i ∈ s, w i = 1) (hz : ∀ i ∈ s, 0 ≤ z i) :
    ∏ i ∈ s, z i ^ w i < ∑ i ∈ s, w i * z i ↔ ∃ j ∈ s, ∃ k ∈ s, z j ≠ z k:= by
  /-
    ι : Type u
    s : Finset ι
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
    hw' : Eq (s.sum fun i => w i) 1
    hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
    ⊢ Iff (LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul …
  -/
  constructor
    /-
      case mp
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
  · intro h
    /-
      case mp
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      h : LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
      ⊢ Exists fun j => And (Membership.mem s j) (Exists fun k => And (Membership.me …
    -/
    by_contra! h_contra
    /-
      case mp
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      h : LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
      h_contra : ∀ (j : ι), Membership.mem s j → ∀ (k : ι), Membership.mem s k → Eq  …
      ⊢ False
    -/
    rw [(geom_mean_eq_arith_mean_weighted_iff' s w z hw hw' hz).mpr ?_] at h
      /-
        case mp
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        h : LT.lt (s.sum fun i => HMul.hMul (w i) (z i)) (s.sum fun i => HMul.hMul (w  …
        h_contra : ∀ (j : ι), Membership.mem s j → ∀ (k : ι), Membership.mem s k → Eq  …
        ⊢ False
      -/
    · exact (lt_self_iff_false _).mp h
      /-
        🎉 no goals
      -/
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        h : LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
        h_contra : ∀ (j : ι), Membership.mem s j → ∀ (k : ι), Membership.mem s k → Eq  …
        ⊢ ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) (z  …
      -/
    · intro j hjs
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        h : LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
        h_contra : ∀ (j : ι), Membership.mem s j → ∀ (k : ι), Membership.mem s k → Eq  …
        j : ι
        hjs : Membership.mem s j
        ⊢ Eq (z j) (s.sum fun i => HMul.hMul (w i) (z i))
      -/
      rw [← arith_mean_weighted_of_constant s w (fun _ => z j) (z j) hw' fun _ _ => congrFun rfl]
      /-
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
        h : LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w …
        h_contra : ∀ (j : ι), Membership.mem s j → ∀ (k : ι), Membership.mem s k → Eq  …
        j : ι
        hjs : Membership.mem s j
        ⊢ Eq (s.sum fun i => HMul.hMul (w i) (z j)) (s.sum fun i => HMul.hMul (w i) (z …
      -/
      apply sum_congr rfl (fun x a => congrArg (HMul.hMul (w x)) (h_contra j hjs x a))
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      ⊢ (Exists fun j => And (Membership.mem s j) (Exists fun k => And (Membership.m …
    -/
  · rintro ⟨j, hjs, k, hks, hzjk⟩
    /-
      case mpr.intro.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      j : ι
      hjs : Membership.mem s j
      k : ι
      hks : Membership.mem s k
      hzjk : Ne (z j) (z k)
      ⊢ LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
    have := geom_mean_le_arith_mean_weighted s w z (fun i a => le_of_lt (hw i a)) hw' hz
    /-
      case mpr.intro.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      j : ι
      hjs : Membership.mem s j
      k : ι
      hks : Membership.mem s k
      hzjk : Ne (z j) (z k)
      this : LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul …
      ⊢ LT.lt (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i …
    -/
    by_contra! h
    /-
      case mpr.intro.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      j : ι
      hjs : Membership.mem s j
      k : ι
      hks : Membership.mem s k
      hzjk : Ne (z j) (z k)
      this : LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul …
      h : LE.le (s.sum fun i => HMul.hMul (w i) (z i)) (s.prod fun i => HPow.hPow (z …
      ⊢ False
    -/
    apply le_antisymm this at h
    /-
      case mpr.intro.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      j : ι
      hjs : Membership.mem s j
      k : ι
      hks : Membership.mem s k
      hzjk : Ne (z j) (z k)
      this : LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul …
      h : Eq (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul (w i) …
      ⊢ False
    -/
    apply (geom_mean_eq_arith_mean_weighted_iff' s w z hw hw' hz).mp at h
    /-
      case mpr.intro.intro.intro.intro
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LE.le 0 (z i)
      j : ι
      hjs : Membership.mem s j
      k : ι
      hks : Membership.mem s k
      hzjk : Ne (z j) (z k)
      this : LE.le (s.prod fun i => HPow.hPow (z i) (w i)) (s.sum fun i => HMul.hMul …
      h : ∀ (j : ι), Membership.mem s j → Eq (z j) (s.sum fun i => HMul.hMul (w i) ( …
      ⊢ False
    -/
    simp only [h j hjs, h k hks, ne_eq, not_true_eq_false] at hzjk
    /-
      🎉 no goals
    -/


/-- **AM-GM inequality**: The geometric mean is less than or equal to the arithmetic mean, weighted
version for `NNReal`-valued functions. -/
theorem geom_mean_le_arith_mean_weighted (w z : ι → ℝ≥0) (hw' : ∑ i ∈ s, w i = 1) :
    (∏ i ∈ s, z i ^ (w i : ℝ)) ≤ ∑ i ∈ s, w i * z i :=
  mod_cast
    Real.geom_mean_le_arith_mean_weighted _ _ _ (fun i _ => (w i).coe_nonneg)
          /-
            ι : Type u
            s : Finset ι
            w z : ι → NNReal
            hw' : Eq (s.sum fun i => w i) 1
            ⊢ Eq (s.sum fun i => ↑(w i)) 1
          -/
      (by assumption_mod_cast) fun i _ => (z i).coe_nonneg
          /-
            🎉 no goals
          -/


/-- **AM-GM inequality**: The geometric mean is less than or equal to the arithmetic mean, weighted
version for two `NNReal` numbers. -/
theorem geom_mean_le_arith_mean2_weighted (w₁ w₂ p₁ p₂ : ℝ≥0) :
    w₁ + w₂ = 1 → p₁ ^ (w₁ : ℝ) * p₂ ^ (w₂ : ℝ) ≤ w₁ * p₁ + w₂ * p₂ := by
  simpa only [Fin.prod_univ_succ, Fin.sum_univ_succ, Finset.prod_empty, Finset.sum_empty,
    Finset.univ_eq_empty, Fin.cons_succ, Fin.cons_zero, add_zero, mul_one] using
    geom_mean_le_arith_mean_weighted univ ![w₁, w₂] ![p₁, p₂]


theorem geom_mean_le_arith_mean3_weighted (w₁ w₂ w₃ p₁ p₂ p₃ : ℝ≥0) :
    w₁ + w₂ + w₃ = 1 →
      p₁ ^ (w₁ : ℝ) * p₂ ^ (w₂ : ℝ) * p₃ ^ (w₃ : ℝ) ≤ w₁ * p₁ + w₂ * p₂ + w₃ * p₃ := by
  simpa only [Fin.prod_univ_succ, Fin.sum_univ_succ, Finset.prod_empty, Finset.sum_empty,
    Finset.univ_eq_empty, Fin.cons_succ, Fin.cons_zero, add_zero, mul_one, ← add_assoc,
    mul_assoc] using geom_mean_le_arith_mean_weighted univ ![w₁, w₂, w₃] ![p₁, p₂, p₃]


theorem geom_mean_le_arith_mean4_weighted (w₁ w₂ w₃ w₄ p₁ p₂ p₃ p₄ : ℝ≥0) :
    w₁ + w₂ + w₃ + w₄ = 1 →
      p₁ ^ (w₁ : ℝ) * p₂ ^ (w₂ : ℝ) * p₃ ^ (w₃ : ℝ) * p₄ ^ (w₄ : ℝ) ≤
        w₁ * p₁ + w₂ * p₂ + w₃ * p₃ + w₄ * p₄ := by
  simpa only [Fin.prod_univ_succ, Fin.sum_univ_succ, Finset.prod_empty, Finset.sum_empty,
    Finset.univ_eq_empty, Fin.cons_succ, Fin.cons_zero, add_zero, mul_one, ← add_assoc,
    mul_assoc] using geom_mean_le_arith_mean_weighted univ ![w₁, w₂, w₃, w₄] ![p₁, p₂, p₃, p₄]


theorem geom_mean_le_arith_mean2_weighted {w₁ w₂ p₁ p₂ : ℝ} (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂)
    (hp₁ : 0 ≤ p₁) (hp₂ : 0 ≤ p₂) (hw : w₁ + w₂ = 1) : p₁ ^ w₁ * p₂ ^ w₂ ≤ w₁ * p₁ + w₂ * p₂ :=
  NNReal.geom_mean_le_arith_mean2_weighted ⟨w₁, hw₁⟩ ⟨w₂, hw₂⟩ ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩ <|
                           /-
                             w₁ w₂ p₁ p₂ : Real
                             hw₁ : LE.le 0 w₁
                             hw₂ : LE.le 0 w₂
                             hp₁ : LE.le 0 p₁
                             hp₂ : LE.le 0 p₂
                             hw : Eq (HAdd.hAdd w₁ w₂) 1
                             ⊢ Eq ↑(HAdd.hAdd ⟨w₁, hw₁⟩ ⟨w₂, hw₂⟩) ↑1
                           -/
    NNReal.coe_inj.1 <| by assumption
                           /-
                             🎉 no goals
                           -/


theorem geom_mean_le_arith_mean3_weighted {w₁ w₂ w₃ p₁ p₂ p₃ : ℝ} (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hp₁ : 0 ≤ p₁) (hp₂ : 0 ≤ p₂) (hp₃ : 0 ≤ p₃) (hw : w₁ + w₂ + w₃ = 1) :
    p₁ ^ w₁ * p₂ ^ w₂ * p₃ ^ w₃ ≤ w₁ * p₁ + w₂ * p₂ + w₃ * p₃ :=
  NNReal.geom_mean_le_arith_mean3_weighted ⟨w₁, hw₁⟩ ⟨w₂, hw₂⟩ ⟨w₃, hw₃⟩ ⟨p₁, hp₁⟩ ⟨p₂, hp₂⟩
      ⟨p₃, hp₃⟩ <|
    NNReal.coe_inj.1 hw


theorem geom_mean_le_arith_mean4_weighted {w₁ w₂ w₃ w₄ p₁ p₂ p₃ p₄ : ℝ} (hw₁ : 0 ≤ w₁)
    (hw₂ : 0 ≤ w₂) (hw₃ : 0 ≤ w₃) (hw₄ : 0 ≤ w₄) (hp₁ : 0 ≤ p₁) (hp₂ : 0 ≤ p₂) (hp₃ : 0 ≤ p₃)
    (hp₄ : 0 ≤ p₄) (hw : w₁ + w₂ + w₃ + w₄ = 1) :
    p₁ ^ w₁ * p₂ ^ w₂ * p₃ ^ w₃ * p₄ ^ w₄ ≤ w₁ * p₁ + w₂ * p₂ + w₃ * p₃ + w₄ * p₄ :=
  NNReal.geom_mean_le_arith_mean4_weighted ⟨w₁, hw₁⟩ ⟨w₂, hw₂⟩ ⟨w₃, hw₃⟩ ⟨w₄, hw₄⟩ ⟨p₁, hp₁⟩
      ⟨p₂, hp₂⟩ ⟨p₃, hp₃⟩ ⟨p₄, hp₄⟩ <|
                           /-
                             w₁ w₂ w₃ w₄ p₁ p₂ p₃ p₄ : Real
                             hw₁ : LE.le 0 w₁
                             hw₂ : LE.le 0 w₂
                             hw₃ : LE.le 0 w₃
                             hw₄ : LE.le 0 w₄
                             hp₁ : LE.le 0 p₁
                             hp₂ : LE.le 0 p₂
                             hp₃ : LE.le 0 p₃
                             hp₄ : LE.le 0 p₄
                             hw : Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd w₁ w₂) w₃) w₄) 1
                             ⊢ Eq ↑(HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ⟨w₁, hw₁⟩ ⟨w₂, hw₂⟩) ⟨w₃, hw₃⟩) ⟨w₄, hw …
                           -/
    NNReal.coe_inj.1 <| by assumption
                           /-
                             🎉 no goals
                           -/


/-- **HM-GM inequality**: The harmonic mean is less than or equal to the geometric mean, weighted
version for real-valued nonnegative functions. -/
theorem harm_mean_le_geom_mean_weighted (w z : ι → ℝ) (hs : s.Nonempty) (hw : ∀ i ∈ s, 0 < w i)
    (hw' : ∑ i in s, w i = 1) (hz : ∀ i ∈ s, 0 < z i) :
    (∑ i in s, w i / z i)⁻¹ ≤ ∏ i in s, z i ^ w i  := by
    have : ∏ i in s, (1 / z) i ^ w i ≤ ∑ i in s, w i * (1 / z) i :=
      geom_mean_le_arith_mean_weighted s w (1/z) (fun i hi ↦ le_of_lt (hw i hi)) hw'
      (fun i hi ↦ one_div_nonneg.2 (le_of_lt (hz i hi)))
    have p_pos : 0 < ∏ i in s, (z i)⁻¹ ^ w i :=
      prod_pos fun i hi => rpow_pos_of_pos (inv_pos.2 (hz i hi)) _
    have s_pos : 0 < ∑ i in s, w i * (z i)⁻¹ :=
      sum_pos (fun i hi => mul_pos (hw i hi) (inv_pos.2 (hz i hi))) hs
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hs : s.Nonempty
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      this : LE.le (s.prod fun i => HPow.hPow (HDiv.hDiv 1 z i) (w i)) (s.sum fun i  …
      p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
      s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
      ⊢ LE.le (Inv.inv (s.sum fun i => HDiv.hDiv (w i) (z i))) (s.prod fun i => HPow …
    -/
    norm_num at this
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hs : s.Nonempty
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
      s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
      this : LE.le (s.prod fun x => HPow.hPow (Inv.inv (z x)) (w x)) (s.sum fun x => …
      ⊢ LE.le (Inv.inv (s.sum fun i => HDiv.hDiv (w i) (z i))) (s.prod fun i => HPow …
    -/
    rw [← inv_le_inv₀ s_pos p_pos] at this
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hs : s.Nonempty
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
      s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
      this : LE.le (Inv.inv (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))) (Inv.i …
      ⊢ LE.le (Inv.inv (s.sum fun i => HDiv.hDiv (w i) (z i))) (s.prod fun i => HPow …
    -/
    apply le_trans this
    have p_pos₂ : 0 < (∏ i in s, (z i) ^ w i)⁻¹ :=
      inv_pos.2 (prod_pos fun i hi => rpow_pos_of_pos ((hz i hi)) _ )
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hs : s.Nonempty
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
      s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
      this : LE.le (Inv.inv (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))) (Inv.i …
      p_pos₂ : LT.lt 0 (Inv.inv (s.prod fun i => HPow.hPow (z i) (w i)))
      ⊢ LE.le (Inv.inv (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))) (s.prod fu …
    -/
    rw [← inv_inv (∏ i in s, z i ^ w i), inv_le_inv₀ p_pos p_pos₂, ← Finset.prod_inv_distrib]
    /-
      ι : Type u
      s : Finset ι
      w z : ι → Real
      hs : s.Nonempty
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : Eq (s.sum fun i => w i) 1
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
      s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
      this : LE.le (Inv.inv (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))) (Inv.i …
      p_pos₂ : LT.lt 0 (Inv.inv (s.prod fun i => HPow.hPow (z i) (w i)))
      ⊢ LE.le (s.prod fun x => Inv.inv (HPow.hPow (z x) (w x))) (s.prod fun i => HPo …
    -/
    gcongr
      /-
        case h0
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hs : s.Nonempty
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
        p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
        s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
        this : LE.le (Inv.inv (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))) (Inv.i …
        p_pos₂ : LT.lt 0 (Inv.inv (s.prod fun i => HPow.hPow (z i) (w i)))
        ⊢ ∀ (i : ι), Membership.mem s i → LE.le 0 (Inv.inv (HPow.hPow (z i) (w i)))
      -/
    · exact fun i hi ↦ inv_nonneg.mpr (Real.rpow_nonneg (le_of_lt (hz i hi)) _)
      /-
        🎉 no goals
      -/
      /-
        case h1
        ι : Type u
        s : Finset ι
        w z : ι → Real
        hs : s.Nonempty
        hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
        hw' : Eq (s.sum fun i => w i) 1
        hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
        p_pos : LT.lt 0 (s.prod fun i => HPow.hPow (Inv.inv (z i)) (w i))
        s_pos : LT.lt 0 (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))
        this : LE.le (Inv.inv (s.sum fun i => HMul.hMul (w i) (Inv.inv (z i)))) (Inv.i …
        p_pos₂ : LT.lt 0 (Inv.inv (s.prod fun i => HPow.hPow (z i) (w i)))
        i✝ : ι
        a✝ : Membership.mem s i✝
        ⊢ LE.le (Inv.inv (HPow.hPow (z i✝) (w i✝))) (HPow.hPow (Inv.inv (z i✝)) (w i✝))
      -/
    · rw [Real.inv_rpow]; apply fun i hi ↦ le_of_lt (hz i hi); assumption
                                                               /-
                                                                 🎉 no goals
                                                               -/



/-- **HM-GM inequality**: The **harmonic mean is less than or equal to the geometric mean. -/
theorem harm_mean_le_geom_mean {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (w : ι → ℝ)
    (z : ι → ℝ) (hw : ∀ i ∈ s, 0 < w i) (hw' : 0 < ∑ i in s, w i) (hz : ∀ i ∈ s, 0 < z i) :
    (∑ i in s, w i) / (∑ i in s, w i / z i) ≤ (∏ i in s, z i ^ w i) ^ (∑ i in s, w i)⁻¹ := by
  /-
    ι : Type u_1
    s : Finset ι
    hs : s.Nonempty
    w z : ι → Real
    hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
    hw' : LT.lt 0 (s.sum fun i => w i)
    hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
    ⊢ LE.le (HDiv.hDiv (s.sum fun i => w i) (s.sum fun i => HDiv.hDiv (w i) (z i)) …
  -/
  have := harm_mean_le_geom_mean_weighted s (fun i => (w i) / ∑ i in s, w i) z hs ?_ ?_ hz
    /-
      case refine_3
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      this : LE.le (Inv.inv (s.sum fun i => HDiv.hDiv ((fun i => HDiv.hDiv (w i) (s. …
      ⊢ LE.le (HDiv.hDiv (s.sum fun i => w i) (s.sum fun i => HDiv.hDiv (w i) (z i)) …
    -/
  · simp only at this
    /-
      case refine_3
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) (s.sum fun i  …
      ⊢ LE.le (HDiv.hDiv (s.sum fun i => w i) (s.sum fun i => HDiv.hDiv (w i) (z i)) …
    -/
    set n := ∑ i in s, w i
    /-
      case refine_3
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      n : Real := s.sum fun i => w i
      hw' : LT.lt 0 n
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (s …
      ⊢ LE.le (HDiv.hDiv n (s.sum fun i => HDiv.hDiv (w i) (z i))) (HPow.hPow (s.pro …
    -/
    nth_rw 1 [div_eq_mul_inv, (show n = (n⁻¹)⁻¹ by norm_num), ← mul_inv, Finset.mul_sum _ _ n⁻¹]
    /-
      case refine_3
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      n : Real := s.sum fun i => w i
      hw' : LT.lt 0 n
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (s …
      ⊢ LE.le (Inv.inv (s.sum fun i => HMul.hMul (Inv.inv n) (HDiv.hDiv (w i) (z i)) …
    -/
    simp_rw [inv_mul_eq_div n ((w _)/(z _)), div_right_comm _ _ n]
    /-
      case refine_3
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      n : Real := s.sum fun i => w i
      hw' : LT.lt 0 n
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (s …
      ⊢ LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (HPow.h …
    -/
    convert this
    /-
      case h.e'_4
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      n : Real := s.sum fun i => w i
      hw' : LT.lt 0 n
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (s …
      ⊢ Eq (HPow.hPow (s.prod fun i => HPow.hPow (z i) (w i)) (Inv.inv n)) (s.prod f …
    -/
    rw [← Real.finset_prod_rpow s _ (fun i hi ↦ Real.rpow_nonneg (le_of_lt <| hz i hi) _)]
    /-
      case h.e'_4
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      n : Real := s.sum fun i => w i
      hw' : LT.lt 0 n
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (s …
      ⊢ Eq (s.prod fun i => HPow.hPow (HPow.hPow (z i) (w i)) (Inv.inv n)) (s.prod f …
    -/
    refine Finset.prod_congr rfl (fun i hi => ?_)
    /-
      case h.e'_4
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      n : Real := s.sum fun i => w i
      hw' : LT.lt 0 n
      this : LE.le (Inv.inv (s.sum fun x => HDiv.hDiv (HDiv.hDiv (w x) n) (z x))) (s …
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (HPow.hPow (HPow.hPow (z i) (w i)) (Inv.inv n)) (HPow.hPow (z i) (HDiv.hD …
    -/
    rw [← Real.rpow_mul (le_of_lt <| hz i hi) (w _) n⁻¹, div_eq_mul_inv (w _) n]
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      ⊢ ∀ (i : ι), Membership.mem s i → LT.lt 0 ((fun i => HDiv.hDiv (w i) (s.sum fu …
    -/
  · exact fun i hi ↦ div_pos (hw i hi) hw'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      ⊢ Eq (s.sum fun i => (fun i => HDiv.hDiv (w i) (s.sum fun i => w i)) i) 1
    -/
  · simp_rw [div_eq_mul_inv (w _) (∑ i in s, w i), ← Finset.sum_mul _ _ (∑ i in s, w i)⁻¹]
    /-
      case refine_2
      ι : Type u_1
      s : Finset ι
      hs : s.Nonempty
      w z : ι → Real
      hw : ∀ (i : ι), Membership.mem s i → LT.lt 0 (w i)
      hw' : LT.lt 0 (s.sum fun i => w i)
      hz : ∀ (i : ι), Membership.mem s i → LT.lt 0 (z i)
      ⊢ Eq (HMul.hMul (s.sum fun i => w i) (Inv.inv (s.sum fun i => w i))) 1
    -/
    exact mul_inv_cancel₀ hw'.ne'
    /-
      🎉 no goals
    -/


/-- **Young's inequality**, a version for nonnegative real numbers. -/
theorem young_inequality_of_nonneg {a b p q : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hpq : p.IsConjExponent q) : a * b ≤ a ^ p / p + b ^ q / q := by
  simpa [← rpow_mul, ha, hb, hpq.ne_zero, hpq.symm.ne_zero, _root_.div_eq_inv_mul] using
    geom_mean_le_arith_mean2_weighted hpq.inv_nonneg hpq.symm.inv_nonneg
      (rpow_nonneg ha p) (rpow_nonneg hb q) hpq.inv_add_inv_conj


/-- **Young's inequality**, a version for arbitrary real numbers. -/
theorem young_inequality (a b : ℝ) {p q : ℝ} (hpq : p.IsConjExponent q) :
    a * b ≤ |a| ^ p / p + |b| ^ q / q :=
  calc
    a * b ≤ |a * b| := le_abs_self (a * b)
    _ = |a| * |b| := abs_mul a b
    _ ≤ |a| ^ p / p + |b| ^ q / q :=
      Real.young_inequality_of_nonneg (abs_nonneg a) (abs_nonneg b) hpq


/-- **Young's inequality**, `ℝ≥0` version. We use `{p q : ℝ≥0}` in order to avoid constructing
witnesses of `0 ≤ p` and `0 ≤ q` for the denominators. -/
theorem young_inequality (a b : ℝ≥0) {p q : ℝ≥0} (hpq : p.IsConjExponent q) :
    a * b ≤ a ^ (p : ℝ) / p + b ^ (q : ℝ) / q :=
  Real.young_inequality_of_nonneg a.coe_nonneg b.coe_nonneg hpq.coe


/-- **Young's inequality**, `ℝ≥0` version with real conjugate exponents. -/
theorem young_inequality_real (a b : ℝ≥0) {p q : ℝ} (hpq : p.IsConjExponent q) :
    a * b ≤ a ^ p / Real.toNNReal p + b ^ q / Real.toNNReal q := by
  /-
    a b : NNReal
    p q : Real
    hpq : p.IsConjExponent q
    ⊢ LE.le (HMul.hMul a b) (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) p.toNNReal) (HDi …
  -/
  simpa [Real.coe_toNNReal, hpq.nonneg, hpq.symm.nonneg] using young_inequality a b hpq.toNNReal
  /-
    🎉 no goals
  -/


/-- **Young's inequality**, `ℝ≥0∞` version with real conjugate exponents. -/
theorem young_inequality (a b : ℝ≥0∞) {p q : ℝ} (hpq : p.IsConjExponent q) :
    a * b ≤ a ^ p / ENNReal.ofReal p + b ^ q / ENNReal.ofReal q := by
  /-
    a b : ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    ⊢ LE.le (HMul.hMul a b) (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) (ENNReal.ofReal  …
  -/
  by_cases h : a = ⊤ ∨ b = ⊤
    /-
      case pos
      a b : ENNReal
      p q : Real
      hpq : p.IsConjExponent q
      h : Or (Eq a Top.top) (Eq b Top.top)
      ⊢ LE.le (HMul.hMul a b) (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) (ENNReal.ofReal  …
    -/
  · refine le_trans le_top (le_of_eq ?_)
    /-
      case pos
      a b : ENNReal
      p q : Real
      hpq : p.IsConjExponent q
      h : Or (Eq a Top.top) (Eq b Top.top)
      ⊢ Eq Top.top (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) (ENNReal.ofReal p)) (HDiv.h …
    -/
    repeat rw [div_eq_mul_inv]
    /-
      case pos
      a b : ENNReal
      p q : Real
      hpq : p.IsConjExponent q
      h : Or (Eq a Top.top) (Eq b Top.top)
      ⊢ Eq Top.top (HAdd.hAdd (HMul.hMul (HPow.hPow a p) (Inv.inv (ENNReal.ofReal p) …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    cases' h with h h <;> rw [h] <;> simp [h, hpq.pos, hpq.symm.pos]
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case neg
    a b : ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    h : Not (Or (Eq a Top.top) (Eq b Top.top))
    ⊢ LE.le (HMul.hMul a b) (HAdd.hAdd (HDiv.hDiv (HPow.hPow a p) (ENNReal.ofReal  …
  -/
  push_neg at h
  -- if a ≠ ⊤ and b ≠ ⊤, use the nnreal version: nnreal.young_inequality_real
  rw [← coe_toNNReal h.left, ← coe_toNNReal h.right, ← coe_mul, ← coe_rpow_of_nonneg _ hpq.nonneg,
    ← coe_rpow_of_nonneg _ hpq.symm.nonneg, ENNReal.ofReal, ENNReal.ofReal, ←
    @coe_div (Real.toNNReal p) _ (by simp [hpq.pos]), ←
    @coe_div (Real.toNNReal q) _ (by simp [hpq.symm.pos]), ← coe_add, coe_le_coe]
  /-
    case neg
    a b : ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    h : And (Ne a Top.top) (Ne b Top.top)
    ⊢ LE.le (HMul.hMul a.toNNReal b.toNNReal) (HAdd.hAdd (HDiv.hDiv (HPow.hPow a.t …
  -/
  exact NNReal.young_inequality_real a.toNNReal b.toNNReal hpq
  /-
    🎉 no goals
  -/


private theorem inner_le_Lp_mul_Lp_of_norm_le_one (f g : ι → ℝ≥0) {p q : ℝ}
    (hpq : p.IsConjExponent q) (hf : ∑ i ∈ s, f i ^ p ≤ 1) (hg : ∑ i ∈ s, g i ^ q ≤ 1) :
    ∑ i ∈ s, f i * g i ≤ 1 := by
  /-
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : LE.le (s.sum fun i => HPow.hPow (f i) p) 1
    hg : LE.le (s.sum fun i => HPow.hPow (g i) q) 1
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) 1
  -/
  have hp : 0 < p.toNNReal := zero_lt_one.trans hpq.toNNReal.one_lt
  /-
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : LE.le (s.sum fun i => HPow.hPow (f i) p) 1
    hg : LE.le (s.sum fun i => HPow.hPow (g i) q) 1
    hp : LT.lt 0 p.toNNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) 1
  -/
  have hq : 0 < q.toNNReal := zero_lt_one.trans hpq.toNNReal.symm.one_lt
  calc
    ∑ i ∈ s, f i * g i ≤ ∑ i ∈ s, (f i ^ p / Real.toNNReal p + g i ^ q / Real.toNNReal q) :=
      Finset.sum_le_sum fun i _ => young_inequality_real (f i) (g i) hpq
    _ = (∑ i ∈ s, f i ^ p) / Real.toNNReal p + (∑ i ∈ s, g i ^ q) / Real.toNNReal q := by
      rw [sum_add_distrib, sum_div, sum_div]
    _ ≤ 1 / Real.toNNReal p + 1 / Real.toNNReal q := by
      refine add_le_add ?_ ?_ <;> rwa [div_le_iff₀, div_mul_cancel₀] <;> positivity
    _ = 1 := by simp_rw [one_div, hpq.toNNReal.inv_add_inv_conj]


private theorem inner_le_Lp_mul_Lp_of_norm_eq_zero (f g : ι → ℝ≥0) {p q : ℝ}
    (hpq : p.IsConjExponent q) (hf : ∑ i ∈ s, f i ^ p = 0) :
    ∑ i ∈ s, f i * g i ≤ (∑ i ∈ s, f i ^ p) ^ (1 / p) * (∑ i ∈ s, g i ^ q) ^ (1 / q) := by
  simp only [hf, hpq.ne_zero, one_div, sum_eq_zero_iff, zero_rpow, zero_mul,
    inv_eq_zero, Ne, not_false_iff, le_zero_iff, mul_eq_zero]
  /-
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : Eq (s.sum fun i => HPow.hPow (f i) p) 0
    ⊢ ∀ (x : ι), Membership.mem s x → Or (Eq (f x) 0) (Eq (g x) 0)
  -/
  intro i his
  /-
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : Eq (s.sum fun i => HPow.hPow (f i) p) 0
    i : ι
    his : Membership.mem s i
    ⊢ Or (Eq (f i) 0) (Eq (g i) 0)
  -/
  left
  /-
    case h
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : Eq (s.sum fun i => HPow.hPow (f i) p) 0
    i : ι
    his : Membership.mem s i
    ⊢ Eq (f i) 0
  -/
  rw [sum_eq_zero_iff] at hf
  /-
    case h
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : ∀ (x : ι), Membership.mem s x → Eq (HPow.hPow (f x) p) 0
    i : ι
    his : Membership.mem s i
    ⊢ Eq (f i) 0
  -/
  exact (rpow_eq_zero_iff.mp (hf i his)).left
  /-
    🎉 no goals
  -/


/-- **Hölder inequality**: The scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. Version for sums over finite sets,
with `ℝ≥0`-valued functions. -/
theorem inner_le_Lp_mul_Lq (f g : ι → ℝ≥0) {p q : ℝ} (hpq : p.IsConjExponent q) :
    ∑ i ∈ s, f i * g i ≤ (∑ i ∈ s, f i ^ p) ^ (1 / p) * (∑ i ∈ s, g i ^ q) ^ (1 / q) := by
  /-
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  obtain hf | hf := eq_zero_or_pos (∑ i ∈ s, f i ^ p)
    /-
      case inl
      ι : Type u
      s : Finset ι
      f g : ι → NNReal
      p q : Real
      hpq : p.IsConjExponent q
      hf : Eq (s.sum fun i => HPow.hPow (f i) p) 0
      ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
  · exact inner_le_Lp_mul_Lp_of_norm_eq_zero s f g hpq hf
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  obtain hg | hg := eq_zero_or_pos (∑ i ∈ s, g i ^ q)
  · calc
      ∑ i ∈ s, f i * g i = ∑ i ∈ s, g i * f i := by
        congr with i
        rw [mul_comm]
      _ ≤ (∑ i ∈ s, g i ^ q) ^ (1 / q) * (∑ i ∈ s, f i ^ p) ^ (1 / p) :=
        (inner_le_Lp_mul_Lp_of_norm_eq_zero s g f hpq.symm hg)
      _ = (∑ i ∈ s, f i ^ p) ^ (1 / p) * (∑ i ∈ s, g i ^ q) ^ (1 / q) := mul_comm _ _
  /-
    case inr.inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
    hg : LT.lt 0 (s.sum fun i => HPow.hPow (g i) q)
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  let f' i := f i / (∑ i ∈ s, f i ^ p) ^ (1 / p)
  /-
    case inr.inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
    hg : LT.lt 0 (s.sum fun i => HPow.hPow (g i) q)
    f' : ι → NNReal := fun i => HDiv.hDiv (f i) (HPow.hPow (s.sum fun i => HPow.hP …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  let g' i := g i / (∑ i ∈ s, g i ^ q) ^ (1 / q)
  suffices (∑ i ∈ s, f' i * g' i) ≤ 1 by
    simp_rw [f', g', div_mul_div_comm, ← sum_div] at this
    rwa [div_le_iff₀, one_mul] at this
    -- TODO: We are missing a positivity  extension here
    exact mul_pos (rpow_pos hf) (rpow_pos hg)
  /-
    case inr.inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
    hg : LT.lt 0 (s.sum fun i => HPow.hPow (g i) q)
    f' : ι → NNReal := fun i => HDiv.hDiv (f i) (HPow.hPow (s.sum fun i => HPow.hP …
    g' : ι → NNReal := fun i => HDiv.hDiv (g i) (HPow.hPow (s.sum fun i => HPow.hP …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f' i) (g' i)) 1
  -/
  refine inner_le_Lp_mul_Lp_of_norm_le_one s f' g' hpq (le_of_eq ?_) (le_of_eq ?_)
  · simp_rw [f', div_rpow, ← sum_div, ← rpow_mul, one_div, inv_mul_cancel₀ hpq.ne_zero, rpow_one,
      div_self hf.ne']
  · simp_rw [g', div_rpow, ← sum_div, ← rpow_mul, one_div, inv_mul_cancel₀ hpq.symm.ne_zero,
      rpow_one, div_self hg.ne']


/-- **Weighted Hölder inequality**. -/
lemma inner_le_weight_mul_Lp (s : Finset ι) {p : ℝ} (hp : 1 ≤ p) (w f : ι → ℝ≥0) :
    ∑ i ∈ s, w i * f i ≤ (∑ i ∈ s, w i) ^ (1 - p⁻¹) * (∑ i ∈ s, w i * f i ^ p) ^ p⁻¹ := by
  /-
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → NNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  obtain rfl | hp := hp.eq_or_lt
    /-
      case inl
      ι : Type u
      s : Finset ι
      w f : ι → NNReal
      hp : LE.le 1 1
      ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
  · simp
    /-
      🎉 no goals
    -/
  calc
    _ = ∑ i ∈ s, w i ^ (1 - p⁻¹) * (w i ^ p⁻¹ * f i) := ?_
    _ ≤ (∑ i ∈ s, (w i ^ (1 - p⁻¹)) ^ (1 - p⁻¹)⁻¹) ^ (1 / (1 - p⁻¹)⁻¹) *
          (∑ i ∈ s, (w i ^ p⁻¹ * f i) ^ p) ^ (1 / p) :=
        inner_le_Lp_mul_Lq _ _ _ (.symm ⟨hp, by simp⟩)
    _ = _ := ?_
    /-
      case inr.calc_1
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → NNReal
      hp : LT.lt 1 p
      ⊢ Eq (s.sum fun i => HMul.hMul (w i) (f i)) (s.sum fun i => HMul.hMul (HPow.hP …
    -/
  · congr with i
    /-
      case inr.calc_1.e_f.h.a
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → NNReal
      hp : LT.lt 1 p
      i : ι
      ⊢ Eq ↑(HMul.hMul (w i) (f i)) ↑(HMul.hMul (HPow.hPow (w i) (HSub.hSub 1 (Inv.i …
    -/
    rw [← mul_assoc, ← rpow_of_add_eq _ one_ne_zero, rpow_one]
    /-
      case inr.calc_1.e_f.h.a
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → NNReal
      hp : LT.lt 1 p
      i : ι
      ⊢ Eq (HAdd.hAdd (HSub.hSub 1 (Inv.inv p)) (Inv.inv p)) 1
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr.calc_2
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → NNReal
      hp : LT.lt 1 p
      ⊢ Eq (HMul.hMul (HPow.hPow (s.sum fun i => HPow.hPow (HPow.hPow (w i) (HSub.hS …
    -/
  · have hp₀ : p ≠ 0 := by positivity
    /-
      case inr.calc_2
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → NNReal
      hp : LT.lt 1 p
      hp₀ : Ne p 0
      ⊢ Eq (HMul.hMul (HPow.hPow (s.sum fun i => HPow.hPow (HPow.hPow (w i) (HSub.hS …
    -/
    have hp₁ : 1 - p⁻¹ ≠ 0 := by simp [sub_eq_zero, hp.ne']
    /-
      case inr.calc_2
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → NNReal
      hp : LT.lt 1 p
      hp₀ : Ne p 0
      hp₁ : Ne (HSub.hSub 1 (Inv.inv p)) 0
      ⊢ Eq (HMul.hMul (HPow.hPow (s.sum fun i => HPow.hPow (HPow.hPow (w i) (HSub.hS …
    -/
    simp [mul_rpow, div_inv_eq_mul, one_mul, one_div, hp₀, hp₁]
    /-
      🎉 no goals
    -/


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. A version for `NNReal`-valued
functions. For an alternative version, convenient if the infinite sums are already expressed as
`p`-th powers, see `inner_le_Lp_mul_Lq_hasSum`. -/
theorem inner_le_Lp_mul_Lq_tsum {f g : ι → ℝ≥0} {p q : ℝ} (hpq : p.IsConjExponent q)
    (hf : Summable fun i => f i ^ p) (hg : Summable fun i => g i ^ q) :
    (Summable fun i => f i * g i) ∧
      ∑' i, f i * g i ≤ (∑' i, f i ^ p) ^ (1 / p) * (∑' i, g i ^ q) ^ (1 / q) := by
  have H₁ : ∀ s : Finset ι,
      ∑ i ∈ s, f i * g i ≤ (∑' i, f i ^ p) ^ (1 / p) * (∑' i, g i ^ q) ^ (1 / q) := by
    intro s
    refine le_trans (inner_le_Lp_mul_Lq s f g hpq) (mul_le_mul ?_ ?_ bot_le bot_le)
    · rw [NNReal.rpow_le_rpow_iff (one_div_pos.mpr hpq.pos)]
      exact sum_le_tsum _ (fun _ _ => zero_le _) hf
    · rw [NNReal.rpow_le_rpow_iff (one_div_pos.mpr hpq.symm.pos)]
      exact sum_le_tsum _ (fun _ _ => zero_le _) hg
  have bdd : BddAbove (Set.range fun s => ∑ i ∈ s, f i * g i) := by
    refine ⟨(∑' i, f i ^ p) ^ (1 / p) * (∑' i, g i ^ q) ^ (1 / q), ?_⟩
    rintro a ⟨s, rfl⟩
    exact H₁ s
  /-
    ι : Type u
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) q
    H₁ : ∀ (s : Finset ι), LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul …
    bdd : BddAbove (Set.range fun s => s.sum fun i => HMul.hMul (f i) (g i))
    ⊢ And (Summable fun i => HMul.hMul (f i) (g i)) (LE.le (tsum fun i => HMul.hMu …
  -/
  have H₂ : Summable _ := (hasSum_of_isLUB _ (isLUB_ciSup bdd)).summable
  /-
    ι : Type u
    f g : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) q
    H₁ : ∀ (s : Finset ι), LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul …
    bdd : BddAbove (Set.range fun s => s.sum fun i => HMul.hMul (f i) (g i))
    H₂ : Summable fun i => HMul.hMul (f i) (g i)
    ⊢ And (Summable fun i => HMul.hMul (f i) (g i)) (LE.le (tsum fun i => HMul.hMu …
  -/
  exact ⟨H₂, tsum_le_of_sum_le H₂ H₁⟩
  /-
    🎉 no goals
  -/


theorem summable_mul_of_Lp_Lq {f g : ι → ℝ≥0} {p q : ℝ} (hpq : p.IsConjExponent q)
    (hf : Summable fun i => f i ^ p) (hg : Summable fun i => g i ^ q) :
    Summable fun i => f i * g i :=
  (inner_le_Lp_mul_Lq_tsum hpq hf hg).1


theorem inner_le_Lp_mul_Lq_tsum' {f g : ι → ℝ≥0} {p q : ℝ} (hpq : p.IsConjExponent q)
    (hf : Summable fun i => f i ^ p) (hg : Summable fun i => g i ^ q) :
    ∑' i, f i * g i ≤ (∑' i, f i ^ p) ^ (1 / p) * (∑' i, g i ^ q) ^ (1 / q) :=
  (inner_le_Lp_mul_Lq_tsum hpq hf hg).2


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. A version for `NNReal`-valued
functions. For an alternative version, convenient if the infinite sums are not already expressed as
`p`-th powers, see `inner_le_Lp_mul_Lq_tsum`. -/
theorem inner_le_Lp_mul_Lq_hasSum {f g : ι → ℝ≥0} {A B : ℝ≥0} {p q : ℝ}
    (hpq : p.IsConjExponent q) (hf : HasSum (fun i => f i ^ p) (A ^ p))
    (hg : HasSum (fun i => g i ^ q) (B ^ q)) : ∃ C, C ≤ A * B ∧ HasSum (fun i => f i * g i) C := by
  /-
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
    ⊢ Exists fun C => And (LE.le C (HMul.hMul A B)) (HasSum (fun i => HMul.hMul (f …
  -/
  obtain ⟨H₁, H₂⟩ := inner_le_Lp_mul_Lq_tsum hpq hf.summable hg.summable
  /-
    case intro
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
    H₁ : Summable fun i => HMul.hMul (f i) (g i)
    H₂ : LE.le (tsum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (tsum f …
    ⊢ Exists fun C => And (LE.le C (HMul.hMul A B)) (HasSum (fun i => HMul.hMul (f …
  -/
  have hA : A = (∑' i : ι, f i ^ p) ^ (1 / p) := by rw [hf.tsum_eq, rpow_inv_rpow_self hpq.ne_zero]
  have hB : B = (∑' i : ι, g i ^ q) ^ (1 / q) := by
    rw [hg.tsum_eq, rpow_inv_rpow_self hpq.symm.ne_zero]
  /-
    case intro
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p q : Real
    hpq : p.IsConjExponent q
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
    H₁ : Summable fun i => HMul.hMul (f i) (g i)
    H₂ : LE.le (tsum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (tsum f …
    hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
    hB : Eq B (HPow.hPow (tsum fun i => HPow.hPow (g i) q) (HDiv.hDiv 1 q))
    ⊢ Exists fun C => And (LE.le C (HMul.hMul A B)) (HasSum (fun i => HMul.hMul (f …
  -/
  refine ⟨∑' i, f i * g i, ?_, ?_⟩
    /-
      case intro.refine_1
      ι : Type u
      f g : ι → NNReal
      A B : NNReal
      p q : Real
      hpq : p.IsConjExponent q
      hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
      hg : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
      H₁ : Summable fun i => HMul.hMul (f i) (g i)
      H₂ : LE.le (tsum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (tsum f …
      hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
      hB : Eq B (HPow.hPow (tsum fun i => HPow.hPow (g i) q) (HDiv.hDiv 1 q))
      ⊢ LE.le (tsum fun i => HMul.hMul (f i) (g i)) (HMul.hMul A B)
    -/
  · simpa [hA, hB] using H₂
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      ι : Type u
      f g : ι → NNReal
      A B : NNReal
      p q : Real
      hpq : p.IsConjExponent q
      hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
      hg : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
      H₁ : Summable fun i => HMul.hMul (f i) (g i)
      H₂ : LE.le (tsum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (tsum f …
      hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
      hB : Eq B (HPow.hPow (tsum fun i => HPow.hPow (g i) q) (HDiv.hDiv 1 q))
      ⊢ HasSum (fun i => HMul.hMul (f i) (g i)) (tsum fun i => HMul.hMul (f i) (g i))
    -/
  · simpa only [rpow_self_rpow_inv hpq.ne_zero] using H₁.hasSum
    /-
      🎉 no goals
    -/


/-- For `1 ≤ p`, the `p`-th power of the sum of `f i` is bounded above by a constant times the
sum of the `p`-th powers of `f i`. Version for sums over finite sets, with `ℝ≥0`-valued functions.
-/
theorem rpow_sum_le_const_mul_sum_rpow (f : ι → ℝ≥0) {p : ℝ} (hp : 1 ≤ p) :
    (∑ i ∈ s, f i) ^ p ≤ (#s : ℝ≥0) ^ (p - 1) * ∑ i ∈ s, f i ^ p := by
  /-
    ι : Type u
    s : Finset ι
    f : ι → NNReal
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  cases' eq_or_lt_of_le hp with hp hp
    /-
      case inl
      ι : Type u
      s : Finset ι
      f : ι → NNReal
      p : Real
      hp✝ : LE.le 1 p
      hp : Eq 1 p
      ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
    -/
  · simp [← hp]
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u
    s : Finset ι
    f : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  let q : ℝ := p / (p - 1)
  /-
    case inr
    ι : Type u
    s : Finset ι
    f : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    q : Real := HDiv.hDiv p (HSub.hSub p 1)
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  have hpq : p.IsConjExponent q := .conjExponent hp
  /-
    case inr
    ι : Type u
    s : Finset ι
    f : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    q : Real := HDiv.hDiv p (HSub.hSub p 1)
    hpq : p.IsConjExponent q
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  have hp₁ : 1 / p * p = 1 := one_div_mul_cancel hpq.ne_zero
  have hq : 1 / q * p = p - 1 := by
    rw [← hpq.div_conj_eq_sub_one]
    ring
  simpa only [NNReal.mul_rpow, ← NNReal.rpow_mul, hp₁, hq, one_mul, one_rpow, rpow_one,
    Pi.one_apply, sum_const, Nat.smul_one_eq_cast] using
    NNReal.rpow_le_rpow (inner_le_Lp_mul_Lq s 1 f hpq.symm) hpq.nonneg


/-- The `L_p` seminorm of a vector `f` is the greatest value of the inner product
`∑ i ∈ s, f i * g i` over functions `g` of `L_q` seminorm less than or equal to one. -/
theorem isGreatest_Lp (f : ι → ℝ≥0) {p q : ℝ} (hpq : p.IsConjExponent q) :
    IsGreatest ((fun g : ι → ℝ≥0 => ∑ i ∈ s, f i * g i) '' { g | ∑ i ∈ s, g i ^ q ≤ 1 })
      ((∑ i ∈ s, f i ^ p) ^ (1 / p)) := by
  /-
    ι : Type u
    s : Finset ι
    f : ι → NNReal
    p q : Real
    hpq : p.IsConjExponent q
    ⊢ IsGreatest (Set.image (fun g => s.sum fun i => HMul.hMul (f i) (g i)) (setOf …
  -/
  constructor
    /-
      case left
      ι : Type u
      s : Finset ι
      f : ι → NNReal
      p q : Real
      hpq : p.IsConjExponent q
      ⊢ Membership.mem (Set.image (fun g => s.sum fun i => HMul.hMul (f i) (g i)) (s …
    -/
  · use fun i => f i ^ p / f i / (∑ i ∈ s, f i ^ p) ^ (1 / q)
    /-
      case h
      ι : Type u
      s : Finset ι
      f : ι → NNReal
      p q : Real
      hpq : p.IsConjExponent q
      ⊢ And (Membership.mem (setOf fun g => LE.le (s.sum fun i => HPow.hPow (g i) q) …
    -/
    obtain hf | hf := eq_zero_or_pos (∑ i ∈ s, f i ^ p)
      /-
        case h.inl
        ι : Type u
        s : Finset ι
        f : ι → NNReal
        p q : Real
        hpq : p.IsConjExponent q
        hf : Eq (s.sum fun i => HPow.hPow (f i) p) 0
        ⊢ And (Membership.mem (setOf fun g => LE.le (s.sum fun i => HPow.hPow (g i) q) …
      -/
    · simp [hf, hpq.ne_zero, hpq.symm.ne_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        ι : Type u
        s : Finset ι
        f : ι → NNReal
        p q : Real
        hpq : p.IsConjExponent q
        hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
        ⊢ And (Membership.mem (setOf fun g => LE.le (s.sum fun i => HPow.hPow (g i) q) …
      -/
    · have A : p + q - q ≠ 0 := by simp [hpq.ne_zero]
      have B : ∀ y : ℝ≥0, y * y ^ p / y = y ^ p := by
        refine fun y => mul_div_cancel_left_of_imp fun h => ?_
        simp [h, hpq.ne_zero]
      simp only [Set.mem_setOf_eq, div_rpow, ← sum_div, ← rpow_mul,
        div_mul_cancel₀ _ hpq.symm.ne_zero, rpow_one, div_le_iff₀ hf, one_mul, hpq.mul_eq_add, ←
        rpow_sub' A, add_sub_cancel_right, le_refl, true_and, ← mul_div_assoc, B]
      /-
        case h.inr
        ι : Type u
        s : Finset ι
        f : ι → NNReal
        p q : Real
        hpq : p.IsConjExponent q
        hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
        A : Ne (HSub.hSub (HAdd.hAdd p q) q) 0
        B : ∀ (y : NNReal), Eq (HDiv.hDiv (HMul.hMul y (HPow.hPow y p)) y) (HPow.hPow  …
        ⊢ Eq (HDiv.hDiv (s.sum fun i => HPow.hPow (f i) p) (HPow.hPow (s.sum fun i =>  …
      -/
      rw [div_eq_iff, ← rpow_add hf.ne', one_div, one_div, hpq.inv_add_inv_conj, rpow_one]
      /-
        case h.inr
        ι : Type u
        s : Finset ι
        f : ι → NNReal
        p q : Real
        hpq : p.IsConjExponent q
        hf : LT.lt 0 (s.sum fun i => HPow.hPow (f i) p)
        A : Ne (HSub.hSub (HAdd.hAdd p q) q) 0
        B : ∀ (y : NNReal), Eq (HDiv.hDiv (HMul.hMul y (HPow.hPow y p)) y) (HPow.hPow  …
        ⊢ Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 q)) 0
      -/
      simpa [hpq.symm.ne_zero] using hf.ne'
      /-
        🎉 no goals
      -/
    /-
      case right
      ι : Type u
      s : Finset ι
      f : ι → NNReal
      p q : Real
      hpq : p.IsConjExponent q
      ⊢ Membership.mem (upperBounds (Set.image (fun g => s.sum fun i => HMul.hMul (f …
    -/
  · rintro _ ⟨g, hg, rfl⟩
    /-
      case right.intro.intro
      ι : Type u
      s : Finset ι
      f : ι → NNReal
      p q : Real
      hpq : p.IsConjExponent q
      g : ι → NNReal
      hg : Membership.mem (setOf fun g => LE.le (s.sum fun i => HPow.hPow (g i) q) 1 …
      ⊢ LE.le ((fun g => s.sum fun i => HMul.hMul (f i) (g i)) g) (HPow.hPow (s.sum  …
    -/
    apply le_trans (inner_le_Lp_mul_Lq s f g hpq)
    simpa only [mul_one] using
      mul_le_mul_left' (NNReal.rpow_le_one hg (le_of_lt hpq.symm.one_div_pos)) _


/-- **Minkowski inequality**: the `L_p` seminorm of the sum of two vectors is less than or equal
to the sum of the `L_p`-seminorms of the summands. A version for `NNReal`-valued functions. -/
theorem Lp_add_le (f g : ι → ℝ≥0) {p : ℝ} (hp : 1 ≤ p) :
    (∑ i ∈ s, (f i + g i) ^ p) ^ (1 / p) ≤
      (∑ i ∈ s, f i ^ p) ^ (1 / p) + (∑ i ∈ s, g i ^ p) ^ (1 / p) := by
  -- The result is trivial when `p = 1`, so we can assume `1 < p`.
  /-
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  rcases eq_or_lt_of_le hp with (rfl | hp)
    /-
      case inl
      ι : Type u
      s : Finset ι
      f g : ι → NNReal
      hp : LE.le 1 1
      ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) 1) (1 / 1 …
    -/
  · simp [Finset.sum_add_distrib]
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  have hpq := Real.IsConjExponent.conjExponent hp
  /-
    case inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    hpq : p.IsConjExponent p.conjExponent
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  have := isGreatest_Lp s (f + g) hpq
  /-
    case inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    hpq : p.IsConjExponent p.conjExponent
    this : IsGreatest (Set.image (fun g_1 => s.sum fun i => HMul.hMul (HAdd.hAdd f …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  simp only [Pi.add_apply, add_mul, sum_add_distrib] at this
  /-
    case inr
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    hpq : p.IsConjExponent p.conjExponent
    this : IsGreatest (Set.image (fun a => HAdd.hAdd (s.sum fun x => HMul.hMul (f  …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  rcases this.1 with ⟨φ, hφ, H⟩
  /-
    case inr.intro.intro
    ι : Type u
    s : Finset ι
    f g : ι → NNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    hpq : p.IsConjExponent p.conjExponent
    this : IsGreatest (Set.image (fun a => HAdd.hAdd (s.sum fun x => HMul.hMul (f  …
    φ : ι → NNReal
    hφ : Membership.mem (setOf fun g => LE.le (s.sum fun i => HPow.hPow (g i) p.co …
    H : Eq ((fun a => HAdd.hAdd (s.sum fun x => HMul.hMul (f x) (a x)) (s.sum fun  …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  rw [← H]
  exact
    add_le_add ((isGreatest_Lp s f hpq).2 ⟨φ, hφ, rfl⟩) ((isGreatest_Lp s g hpq).2 ⟨φ, hφ, rfl⟩)


/-- **Minkowski inequality**: the `L_p` seminorm of the infinite sum of two vectors is less than or
equal to the infinite sum of the `L_p`-seminorms of the summands, if these infinite sums both
exist. A version for `NNReal`-valued functions. For an alternative version, convenient if the
infinite sums are already expressed as `p`-th powers, see `Lp_add_le_hasSum_of_nonneg`. -/
theorem Lp_add_le_tsum {f g : ι → ℝ≥0} {p : ℝ} (hp : 1 ≤ p) (hf : Summable fun i => f i ^ p)
    (hg : Summable fun i => g i ^ p) :
    (Summable fun i => (f i + g i) ^ p) ∧
      (∑' i, (f i + g i) ^ p) ^ (1 / p) ≤
        (∑' i, f i ^ p) ^ (1 / p) + (∑' i, g i ^ p) ^ (1 / p) := by
  /-
    ι : Type u
    f g : ι → NNReal
    p : Real
    hp : LE.le 1 p
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) p
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (LE.le (HPow.hPo …
  -/
  have pos : 0 < p := lt_of_lt_of_le zero_lt_one hp
  have H₁ : ∀ s : Finset ι,
      (∑ i ∈ s, (f i + g i) ^ p) ≤
        ((∑' i, f i ^ p) ^ (1 / p) + (∑' i, g i ^ p) ^ (1 / p)) ^ p := by
    intro s
    rw [one_div, ← NNReal.rpow_inv_le_iff pos, ← one_div]
    refine le_trans (Lp_add_le s f g hp) (add_le_add ?_ ?_) <;>
        rw [NNReal.rpow_le_rpow_iff (one_div_pos.mpr pos)] <;>
      refine sum_le_tsum _ (fun _ _ => zero_le _) ?_
    exacts [hf, hg]
  have bdd : BddAbove (Set.range fun s => ∑ i ∈ s, (f i + g i) ^ p) := by
    refine ⟨((∑' i, f i ^ p) ^ (1 / p) + (∑' i, g i ^ p) ^ (1 / p)) ^ p, ?_⟩
    rintro a ⟨s, rfl⟩
    exact H₁ s
  /-
    ι : Type u
    f g : ι → NNReal
    p : Real
    hp : LE.le 1 p
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) p
    pos : LT.lt 0 p
    H₁ : ∀ (s : Finset ι), LE.le (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) …
    bdd : BddAbove (Set.range fun s => s.sum fun i => HPow.hPow (HAdd.hAdd (f i) ( …
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (LE.le (HPow.hPo …
  -/
  have H₂ : Summable _ := (hasSum_of_isLUB _ (isLUB_ciSup bdd)).summable
  /-
    ι : Type u
    f g : ι → NNReal
    p : Real
    hp : LE.le 1 p
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) p
    pos : LT.lt 0 p
    H₁ : ∀ (s : Finset ι), LE.le (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) …
    bdd : BddAbove (Set.range fun s => s.sum fun i => HPow.hPow (HAdd.hAdd (f i) ( …
    H₂ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (LE.le (HPow.hPo …
  -/
  refine ⟨H₂, ?_⟩
  /-
    ι : Type u
    f g : ι → NNReal
    p : Real
    hp : LE.le 1 p
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) p
    pos : LT.lt 0 p
    H₁ : ∀ (s : Finset ι), LE.le (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) …
    bdd : BddAbove (Set.range fun s => s.sum fun i => HPow.hPow (HAdd.hAdd (f i) ( …
    H₂ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
    ⊢ LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv.h …
  -/
  rw [one_div, NNReal.rpow_inv_le_iff pos, ← one_div]
  /-
    ι : Type u
    f g : ι → NNReal
    p : Real
    hp : LE.le 1 p
    hf : Summable fun i => HPow.hPow (f i) p
    hg : Summable fun i => HPow.hPow (g i) p
    pos : LT.lt 0 p
    H₁ : ∀ (s : Finset ι), LE.le (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) …
    bdd : BddAbove (Set.range fun s => s.sum fun i => HPow.hPow (HAdd.hAdd (f i) ( …
    H₂ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
    ⊢ LE.le (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow (HAdd.h …
  -/
  exact tsum_le_of_sum_le H₂ H₁
  /-
    🎉 no goals
  -/


theorem summable_Lp_add {f g : ι → ℝ≥0} {p : ℝ} (hp : 1 ≤ p) (hf : Summable fun i => f i ^ p)
    (hg : Summable fun i => g i ^ p) : Summable fun i => (f i + g i) ^ p :=
  (Lp_add_le_tsum hp hf hg).1


theorem Lp_add_le_tsum' {f g : ι → ℝ≥0} {p : ℝ} (hp : 1 ≤ p) (hf : Summable fun i => f i ^ p)
    (hg : Summable fun i => g i ^ p) :
    (∑' i, (f i + g i) ^ p) ^ (1 / p) ≤ (∑' i, f i ^ p) ^ (1 / p) + (∑' i, g i ^ p) ^ (1 / p) :=
  (Lp_add_le_tsum hp hf hg).2


/-- **Minkowski inequality**: the `L_p` seminorm of the infinite sum of two vectors is less than or
equal to the infinite sum of the `L_p`-seminorms of the summands, if these infinite sums both
exist. A version for `NNReal`-valued functions. For an alternative version, convenient if the
infinite sums are not already expressed as `p`-th powers, see `Lp_add_le_tsum_of_nonneg`. -/
theorem Lp_add_le_hasSum {f g : ι → ℝ≥0} {A B : ℝ≥0} {p : ℝ} (hp : 1 ≤ p)
    (hf : HasSum (fun i => f i ^ p) (A ^ p)) (hg : HasSum (fun i => g i ^ p) (B ^ p)) :
    ∃ C, C ≤ A + B ∧ HasSum (fun i => (f i + g i) ^ p) (C ^ p) := by
  /-
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p : Real
    hp : LE.le 1 p
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    ⊢ Exists fun C => And (LE.le C (HAdd.hAdd A B)) (HasSum (fun i => HPow.hPow (H …
  -/
  have hp' : p ≠ 0 := (lt_of_lt_of_le zero_lt_one hp).ne'
  /-
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p : Real
    hp : LE.le 1 p
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    hp' : Ne p 0
    ⊢ Exists fun C => And (LE.le C (HAdd.hAdd A B)) (HasSum (fun i => HPow.hPow (H …
  -/
  obtain ⟨H₁, H₂⟩ := Lp_add_le_tsum hp hf.summable hg.summable
  /-
    case intro
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p : Real
    hp : LE.le 1 p
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    hp' : Ne p 0
    H₁ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
    H₂ : LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDi …
    ⊢ Exists fun C => And (LE.le C (HAdd.hAdd A B)) (HasSum (fun i => HPow.hPow (H …
  -/
  have hA : A = (∑' i : ι, f i ^ p) ^ (1 / p) := by rw [hf.tsum_eq, rpow_inv_rpow_self hp']
  /-
    case intro
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p : Real
    hp : LE.le 1 p
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    hp' : Ne p 0
    H₁ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
    H₂ : LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDi …
    hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
    ⊢ Exists fun C => And (LE.le C (HAdd.hAdd A B)) (HasSum (fun i => HPow.hPow (H …
  -/
  have hB : B = (∑' i : ι, g i ^ p) ^ (1 / p) := by rw [hg.tsum_eq, rpow_inv_rpow_self hp']
  /-
    case intro
    ι : Type u
    f g : ι → NNReal
    A B : NNReal
    p : Real
    hp : LE.le 1 p
    hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    hp' : Ne p 0
    H₁ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
    H₂ : LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDi …
    hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
    hB : Eq B (HPow.hPow (tsum fun i => HPow.hPow (g i) p) (HDiv.hDiv 1 p))
    ⊢ Exists fun C => And (LE.le C (HAdd.hAdd A B)) (HasSum (fun i => HPow.hPow (H …
  -/
  refine ⟨(∑' i, (f i + g i) ^ p) ^ (1 / p), ?_, ?_⟩
    /-
      case intro.refine_1
      ι : Type u
      f g : ι → NNReal
      A B : NNReal
      p : Real
      hp : LE.le 1 p
      hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
      hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
      hp' : Ne p 0
      H₁ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
      H₂ : LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDi …
      hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
      hB : Eq B (HPow.hPow (tsum fun i => HPow.hPow (g i) p) (HDiv.hDiv 1 p))
      ⊢ LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv.h …
    -/
  · simpa [hA, hB] using H₂
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      ι : Type u
      f g : ι → NNReal
      A B : NNReal
      p : Real
      hp : LE.le 1 p
      hf : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
      hg : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
      hp' : Ne p 0
      H₁ : Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p
      H₂ : LE.le (HPow.hPow (tsum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDi …
      hA : Eq A (HPow.hPow (tsum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p))
      hB : Eq B (HPow.hPow (tsum fun i => HPow.hPow (g i) p) (HDiv.hDiv 1 p))
      ⊢ HasSum (fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow (HPow.hPow  …
    -/
  · simpa only [rpow_self_rpow_inv hp'] using H₁.hasSum
    /-
      🎉 no goals
    -/


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. Version for sums over finite sets,
with real-valued functions. -/
theorem inner_le_Lp_mul_Lq (hpq : IsConjExponent p q) :
    ∑ i ∈ s, f i * g i ≤ (∑ i ∈ s, |f i| ^ p) ^ (1 / p) * (∑ i ∈ s, |g i| ^ q) ^ (1 / q) := by
  have :=
    NNReal.coe_le_coe.2
      (NNReal.inner_le_Lp_mul_Lq s (fun i => ⟨_, abs_nonneg (f i)⟩) (fun i => ⟨_, abs_nonneg (g i)⟩)
        hpq)
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    this : LE.le ↑(s.sum fun i => HMul.hMul ((fun i => ⟨abs (f i), ⋯⟩) i) ((fun i  …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  push_cast at this
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    this : LE.le (s.sum fun x => HMul.hMul (abs (f x)) (abs (g x))) (HMul.hMul (HP …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  refine le_trans (sum_le_sum fun i _ => ?_) this
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    this : LE.le (s.sum fun x => HMul.hMul (abs (f x)) (abs (g x))) (HMul.hMul (HP …
    i : ι
    x✝ : Membership.mem s i
    ⊢ LE.le (HMul.hMul (f i) (g i)) (HMul.hMul (abs (f i)) (abs (g i)))
  -/
  simp only [← abs_mul, le_abs_self]
  /-
    🎉 no goals
  -/


/-- For `1 ≤ p`, the `p`-th power of the sum of `f i` is bounded above by a constant times the
sum of the `p`-th powers of `f i`. Version for sums over finite sets, with `ℝ`-valued functions. -/
theorem rpow_sum_le_const_mul_sum_rpow (hp : 1 ≤ p) :
    (∑ i ∈ s, |f i|) ^ p ≤ (#s : ℝ) ^ (p - 1) * ∑ i ∈ s, |f i| ^ p := by
  have :=
    NNReal.coe_le_coe.2
      (NNReal.rpow_sum_le_const_mul_sum_rpow s (fun i => ⟨_, abs_nonneg (f i)⟩) hp)
  /-
    ι : Type u
    s : Finset ι
    f : ι → Real
    p : Real
    hp : LE.le 1 p
    this : LE.le ↑(HPow.hPow (s.sum fun i => (fun i => ⟨abs (f i), ⋯⟩) i) p) ↑(HMu …
    ⊢ LE.le (HPow.hPow (s.sum fun i => abs (f i)) p) (HMul.hMul (HPow.hPow (↑s.car …
  -/
  push_cast at this
  /-
    ι : Type u
    s : Finset ι
    f : ι → Real
    p : Real
    hp : LE.le 1 p
    this : LE.le (HPow.hPow (s.sum fun x => abs (f x)) p) (HMul.hMul (HPow.hPow (↑ …
    ⊢ LE.le (HPow.hPow (s.sum fun i => abs (f i)) p) (HMul.hMul (HPow.hPow (↑s.car …
  -/
  exact this
  /-
    🎉 no goals
  -/

-- for some reason `exact_mod_cast` can't replace this argument

/-- **Minkowski inequality**: the `L_p` seminorm of the sum of two vectors is less than or equal
to the sum of the `L_p`-seminorms of the summands. A version for `Real`-valued functions. -/
theorem Lp_add_le (hp : 1 ≤ p) :
    (∑ i ∈ s, |f i + g i| ^ p) ^ (1 / p) ≤
      (∑ i ∈ s, |f i| ^ p) ^ (1 / p) + (∑ i ∈ s, |g i| ^ p) ^ (1 / p) := by
  have :=
    NNReal.coe_le_coe.2
      (NNReal.Lp_add_le s (fun i => ⟨_, abs_nonneg (f i)⟩) (fun i => ⟨_, abs_nonneg (g i)⟩) hp)
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p : Real
    hp : LE.le 1 p
    this : LE.le ↑(HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd ((fun i => ⟨abs  …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (abs (HAdd.hAdd (f i) (g i))) p)  …
  -/
  push_cast at this
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p : Real
    hp : LE.le 1 p
    this : LE.le (HPow.hPow (s.sum fun x => HPow.hPow (HAdd.hAdd (abs (f x)) (abs  …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (abs (HAdd.hAdd (f i) (g i))) p)  …
  -/
  refine le_trans (rpow_le_rpow ?_ (sum_le_sum fun i _ => ?_) ?_) this <;>
    simp [sum_nonneg, rpow_nonneg, abs_nonneg, le_trans zero_le_one hp, abs_add,
      rpow_le_rpow]


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. Version for sums over finite sets,
with real-valued nonnegative functions. -/
theorem inner_le_Lp_mul_Lq_of_nonneg (hpq : IsConjExponent p q) (hf : ∀ i ∈ s, 0 ≤ f i)
    (hg : ∀ i ∈ s, 0 ≤ g i) :
    ∑ i ∈ s, f i * g i ≤ (∑ i ∈ s, f i ^ p) ^ (1 / p) * (∑ i ∈ s, g i ^ q) ^ (1 / q) := by
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  convert inner_le_Lp_mul_Lq s f g hpq using 3 <;> apply sum_congr rfl <;> intro i hi <;>
    /-
      case h.e'_4.h.e'_5.h.e'_5
      ι : Type u
      s : Finset ι
      f g : ι → Real
      p q : Real
      hpq : p.IsConjExponent q
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (HPow.hPow (f i) p) (HPow.hPow (abs (f i)) p)
    -/
    /-
      🎉 no goals
    -/
    simp only [abs_of_nonneg, hf i hi, hg i hi]
    /-
      🎉 no goals
    -/


/-- **Weighted Hölder inequality**. -/
lemma inner_le_weight_mul_Lp_of_nonneg (s : Finset ι) {p : ℝ} (hp : 1 ≤ p) (w f : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hf : ∀ i, 0 ≤ f i) :
    ∑ i ∈ s, w i * f i ≤ (∑ i ∈ s, w i) ^ (1 - p⁻¹) * (∑ i ∈ s, w i * f i ^ p) ^ p⁻¹ := by
  /-
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → Real
    hw : ∀ (i : ι), LE.le 0 (w i)
    hf : ∀ (i : ι), LE.le 0 (f i)
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  lift w to ι → ℝ≥0 using hw
  /-
    case intro
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    f : ι → Real
    hf : ∀ (i : ι), LE.le 0 (f i)
    w : ι → NNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul ((fun i => ↑(w i)) i) (f i)) (HMul.hMul (HPo …
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro.intro
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → NNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul ((fun i => ↑(w i)) i) ((fun i => ↑(f i)) i)) …
  -/
  beta_reduce at *
  /-
    case intro.intro
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → NNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul ↑(w i) ↑(f i)) (HMul.hMul (HPow.hPow (s.sum  …
  -/
  norm_cast at *
  /-
    case intro.intro
    ι : Type u
    s : Finset ι
    p : Real
    w f : ι → NNReal
    hp : LE.le 1 p
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  exact NNReal.inner_le_weight_mul_Lp _ hp _ _
  /-
    🎉 no goals
  -/


/-- **Weighted Hölder inequality** in terms of `Finset.expect`. -/
lemma compact_inner_le_weight_mul_Lp_of_nonneg (s : Finset ι) {p : ℝ} (hp : 1 ≤ p) {w f : ι → ℝ}
    (hw : ∀ i, 0 ≤ w i) (hf : ∀ i, 0 ≤ f i) :
    𝔼 i ∈ s, w i * f i ≤ (𝔼 i ∈ s, w i) ^ (1 - p⁻¹) * (𝔼 i ∈ s, w i * f i ^ p) ^ p⁻¹ := by
  /-
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → Real
    hw : ∀ (i : ι), LE.le 0 (w i)
    hf : ∀ (i : ι), LE.le 0 (f i)
    ⊢ LE.le (s.expect fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.exp …
  -/
  simp_rw [expect_eq_sum_div_card]
  /-
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → Real
    hw : ∀ (i : ι), LE.le 0 (w i)
    hf : ∀ (i : ι), LE.le 0 (f i)
    ⊢ LE.le (HDiv.hDiv (s.sum fun i => HMul.hMul (w i) (f i)) ↑s.card) (HMul.hMul  …
  -/
  rw [div_rpow, div_rpow, div_mul_div_comm, ← rpow_add', sub_add_cancel, rpow_one]
    /-
      ι : Type u
      s : Finset ι
      p : Real
      hp : LE.le 1 p
      w f : ι → Real
      hw : ∀ (i : ι), LE.le 0 (w i)
      hf : ∀ (i : ι), LE.le 0 (f i)
      ⊢ LE.le (HDiv.hDiv (s.sum fun i => HMul.hMul (w i) (f i)) ↑s.card) (HDiv.hDiv  …
    -/
  · gcongr
    /-
      case hab
      ι : Type u
      s : Finset ι
      p : Real
      hp : LE.le 1 p
      w f : ι → Real
      hw : ∀ (i : ι), LE.le 0 (w i)
      hf : ∀ (i : ι), LE.le 0 (f i)
      ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
    exact inner_le_weight_mul_Lp_of_nonneg s hp _ _ hw hf
    /-
      🎉 no goals
    -/
  /-
    case hx
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → Real
    hw : ∀ (i : ι), LE.le 0 (w i)
    hf : ∀ (i : ι), LE.le 0 (f i)
    ⊢ LE.le 0 ↑s.card
  -/
  any_goals simp
    /-
      case hx
      ι : Type u
      s : Finset ι
      p : Real
      hp : LE.le 1 p
      w f : ι → Real
      hw : ∀ (i : ι), LE.le 0 (w i)
      hf : ∀ (i : ι), LE.le 0 (f i)
      ⊢ LE.le 0 (s.sum fun i => HMul.hMul (w i) (HPow.hPow (f i) p))
    -/
  · exact sum_nonneg fun i _ ↦ by have := hw i; have := hf i; positivity
    /-
      🎉 no goals
    -/
    /-
      case hx
      ι : Type u
      s : Finset ι
      p : Real
      hp : LE.le 1 p
      w f : ι → Real
      hw : ∀ (i : ι), LE.le 0 (w i)
      hf : ∀ (i : ι), LE.le 0 (f i)
      ⊢ LE.le 0 (s.sum fun i => w i)
    -/
  · exact sum_nonneg fun i _ ↦ by have := hw i; positivity
    /-
      🎉 no goals
    -/


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. A version for `ℝ`-valued functions.
For an alternative version, convenient if the infinite sums are already expressed as `p`-th powers,
see `inner_le_Lp_mul_Lq_hasSum_of_nonneg`. -/
theorem inner_le_Lp_mul_Lq_tsum_of_nonneg (hpq : p.IsConjExponent q) (hf : ∀ i, 0 ≤ f i)
    (hg : ∀ i, 0 ≤ g i) (hf_sum : Summable fun i => f i ^ p) (hg_sum : Summable fun i => g i ^ q) :
    (Summable fun i => f i * g i) ∧
      ∑' i, f i * g i ≤ (∑' i, f i ^ p) ^ (1 / p) * (∑' i, g i ^ q) ^ (1 / q) := by
  /-
    ι : Type u
    f g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    hf : ∀ (i : ι), LE.le 0 (f i)
    hg : ∀ (i : ι), LE.le 0 (g i)
    hf_sum : Summable fun i => HPow.hPow (f i) p
    hg_sum : Summable fun i => HPow.hPow (g i) q
    ⊢ And (Summable fun i => HMul.hMul (f i) (g i)) (LE.le (tsum fun i => HMul.hMu …
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro
    ι : Type u
    g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    hg : ∀ (i : ι), LE.le 0 (g i)
    hg_sum : Summable fun i => HPow.hPow (g i) q
    f : ι → NNReal
    hf_sum : Summable fun i => HPow.hPow ((fun i => ↑(f i)) i) p
    ⊢ And (Summable fun i => HMul.hMul ((fun i => ↑(f i)) i) (g i)) (LE.le (tsum f …
  -/
  lift g to ι → ℝ≥0 using hg
  -- After https://github.com/leanprover/lean4/pull/2734, `norm_cast` needs help with beta reduction.
  /-
    case intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f : ι → NNReal
    hf_sum : Summable fun i => HPow.hPow ((fun i => ↑(f i)) i) p
    g : ι → NNReal
    hg_sum : Summable fun i => HPow.hPow ((fun i => ↑(g i)) i) q
    ⊢ And (Summable fun i => HMul.hMul ((fun i => ↑(f i)) i) ((fun i => ↑(g i)) i) …
  -/
  beta_reduce at *
  /-
    case intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f : ι → NNReal
    hf_sum : Summable fun i => HPow.hPow (↑(f i)) p
    g : ι → NNReal
    hg_sum : Summable fun i => HPow.hPow (↑(g i)) q
    ⊢ And (Summable fun i => HMul.hMul ↑(f i) ↑(g i)) (LE.le (tsum fun i => HMul.h …
  -/
  norm_cast at *
  /-
    case intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f g : ι → NNReal
    hf_sum : Summable fun a => HPow.hPow (f a) p
    hg_sum : Summable fun a => HPow.hPow (g a) q
    ⊢ And (Summable fun a => HMul.hMul (f a) (g a)) (LE.le (tsum fun a => HMul.hMu …
  -/
  exact NNReal.inner_le_Lp_mul_Lq_tsum hpq hf_sum hg_sum
  /-
    🎉 no goals
  -/


theorem summable_mul_of_Lp_Lq_of_nonneg (hpq : p.IsConjExponent q) (hf : ∀ i, 0 ≤ f i)
    (hg : ∀ i, 0 ≤ g i) (hf_sum : Summable fun i => f i ^ p) (hg_sum : Summable fun i => g i ^ q) :
    Summable fun i => f i * g i :=
  (inner_le_Lp_mul_Lq_tsum_of_nonneg hpq hf hg hf_sum hg_sum).1


theorem inner_le_Lp_mul_Lq_tsum_of_nonneg' (hpq : p.IsConjExponent q) (hf : ∀ i, 0 ≤ f i)
    (hg : ∀ i, 0 ≤ g i) (hf_sum : Summable fun i => f i ^ p) (hg_sum : Summable fun i => g i ^ q) :
    ∑' i, f i * g i ≤ (∑' i, f i ^ p) ^ (1 / p) * (∑' i, g i ^ q) ^ (1 / q) :=
  (inner_le_Lp_mul_Lq_tsum_of_nonneg hpq hf hg hf_sum hg_sum).2


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. A version for `NNReal`-valued
functions. For an alternative version, convenient if the infinite sums are not already expressed as
`p`-th powers, see `inner_le_Lp_mul_Lq_tsum_of_nonneg`. -/
theorem inner_le_Lp_mul_Lq_hasSum_of_nonneg (hpq : p.IsConjExponent q) {A B : ℝ} (hA : 0 ≤ A)
    (hB : 0 ≤ B) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i)
    (hf_sum : HasSum (fun i => f i ^ p) (A ^ p)) (hg_sum : HasSum (fun i => g i ^ q) (B ^ q)) :
    ∃ C : ℝ, 0 ≤ C ∧ C ≤ A * B ∧ HasSum (fun i => f i * g i) C := by
  /-
    ι : Type u
    f g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    A B : Real
    hA : LE.le 0 A
    hB : LE.le 0 B
    hf : ∀ (i : ι), LE.le 0 (f i)
    hg : ∀ (i : ι), LE.le 0 (g i)
    hf_sum : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hg_sum : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul A B)) (HasSum (fun  …
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro
    ι : Type u
    g : ι → Real
    p q : Real
    hpq : p.IsConjExponent q
    A B : Real
    hA : LE.le 0 A
    hB : LE.le 0 B
    hg : ∀ (i : ι), LE.le 0 (g i)
    hg_sum : HasSum (fun i => HPow.hPow (g i) q) (HPow.hPow B q)
    f : ι → NNReal
    hf_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow A p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul A B)) (HasSum (fun  …
  -/
  lift g to ι → ℝ≥0 using hg
  /-
    case intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    A B : Real
    hA : LE.le 0 A
    hB : LE.le 0 B
    f : ι → NNReal
    hf_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow A p)
    g : ι → NNReal
    hg_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(g i)) i) q) (HPow.hPow B q)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul A B)) (HasSum (fun  …
  -/
  lift A to ℝ≥0 using hA
  /-
    case intro.intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    B : Real
    hB : LE.le 0 B
    f g : ι → NNReal
    hg_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(g i)) i) q) (HPow.hPow B q)
    A : NNReal
    hf_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow (↑A) p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul (↑A) B)) (HasSum (f …
  -/
  lift B to ℝ≥0 using hB
  -- After https://github.com/leanprover/lean4/pull/2734, `norm_cast` needs help with beta reduction.
  /-
    case intro.intro.intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f g : ι → NNReal
    A : NNReal
    hf_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow (↑A) p)
    B : NNReal
    hg_sum : HasSum (fun i => HPow.hPow ((fun i => ↑(g i)) i) q) (HPow.hPow (↑B) q)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul ↑A ↑B)) (HasSum (fu …
  -/
  beta_reduce at *
  /-
    case intro.intro.intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f g : ι → NNReal
    A : NNReal
    hf_sum : HasSum (fun i => HPow.hPow (↑(f i)) p) (HPow.hPow (↑A) p)
    B : NNReal
    hg_sum : HasSum (fun i => HPow.hPow (↑(g i)) q) (HPow.hPow (↑B) q)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul ↑A ↑B)) (HasSum (fu …
  -/
  norm_cast at hf_sum hg_sum
  /-
    case intro.intro.intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f g : ι → NNReal
    A B : NNReal
    hf_sum : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hg_sum : HasSum (fun a => HPow.hPow (g a) q) (HPow.hPow B q)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul ↑A ↑B)) (HasSum (fu …
  -/
  obtain ⟨C, hC, H⟩ := NNReal.inner_le_Lp_mul_Lq_hasSum hpq hf_sum hg_sum
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f g : ι → NNReal
    A B : NNReal
    hf_sum : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hg_sum : HasSum (fun a => HPow.hPow (g a) q) (HPow.hPow B q)
    C : NNReal
    hC : LE.le C (HMul.hMul A B)
    H : HasSum (fun i => HMul.hMul (f i) (g i)) C
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HMul.hMul ↑A ↑B)) (HasSum (fu …
  -/
  refine ⟨C, C.prop, hC, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    p q : Real
    hpq : p.IsConjExponent q
    f g : ι → NNReal
    A B : NNReal
    hf_sum : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hg_sum : HasSum (fun a => HPow.hPow (g a) q) (HPow.hPow B q)
    C : NNReal
    hC : LE.le C (HMul.hMul A B)
    H : HasSum (fun i => HMul.hMul (f i) (g i)) C
    ⊢ HasSum (fun i => HMul.hMul ↑(f i) ↑(g i)) ↑C
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- For `1 ≤ p`, the `p`-th power of the sum of `f i` is bounded above by a constant times the
sum of the `p`-th powers of `f i`. Version for sums over finite sets, with nonnegative `ℝ`-valued
functions. -/
theorem rpow_sum_le_const_mul_sum_rpow_of_nonneg (hp : 1 ≤ p) (hf : ∀ i ∈ s, 0 ≤ f i) :
    (∑ i ∈ s, f i) ^ p ≤ (#s : ℝ) ^ (p - 1) * ∑ i ∈ s, f i ^ p := by
  /-
    ι : Type u
    s : Finset ι
    f : ι → Real
    p : Real
    hp : LE.le 1 p
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  convert rpow_sum_le_const_mul_sum_rpow s f hp using 2 <;> apply sum_congr rfl <;> intro i hi <;>
    /-
      case h.e'_3.h.e'_5
      ι : Type u
      s : Finset ι
      f : ι → Real
      p : Real
      hp : LE.le 1 p
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (f i) (abs (f i))
    -/
    /-
      🎉 no goals
    -/
    simp only [abs_of_nonneg, hf i hi]
    /-
      🎉 no goals
    -/


/-- **Minkowski inequality**: the `L_p` seminorm of the sum of two vectors is less than or equal
to the sum of the `L_p`-seminorms of the summands. A version for `ℝ`-valued nonnegative
functions. -/
theorem Lp_add_le_of_nonneg (hp : 1 ≤ p) (hf : ∀ i ∈ s, 0 ≤ f i) (hg : ∀ i ∈ s, 0 ≤ g i) :
    (∑ i ∈ s, (f i + g i) ^ p) ^ (1 / p) ≤
      (∑ i ∈ s, f i ^ p) ^ (1 / p) + (∑ i ∈ s, g i ^ p) ^ (1 / p) := by
  /-
    ι : Type u
    s : Finset ι
    f g : ι → Real
    p : Real
    hp : LE.le 1 p
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  convert Lp_add_le s f g hp using 2 <;> [skip;congr 1;congr 1] <;> apply sum_congr rfl <;>
      /-
        case h.e'_3.h.e'_5
        ι : Type u
        s : Finset ι
        f g : ι → Real
        p : Real
        hp : LE.le 1 p
        hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
        hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
        ⊢ ∀ (x : ι), Membership.mem s x → Eq (HPow.hPow (HAdd.hAdd (f x) (g x)) p) (HP …
      -/
      intro i hi <;>
    /-
      case h.e'_3.h.e'_5
      ι : Type u
      s : Finset ι
      f g : ι → Real
      p : Real
      hp : LE.le 1 p
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow (abs (HAdd.hAdd (f i) (g …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp only [abs_of_nonneg, hf i hi, hg i hi, add_nonneg]
    /-
      🎉 no goals
    -/


/-- **Minkowski inequality**: the `L_p` seminorm of the infinite sum of two vectors is less than or
equal to the infinite sum of the `L_p`-seminorms of the summands, if these infinite sums both
exist. A version for `ℝ`-valued functions. For an alternative version, convenient if the infinite
sums are already expressed as `p`-th powers, see `Lp_add_le_hasSum_of_nonneg`. -/
theorem Lp_add_le_tsum_of_nonneg (hp : 1 ≤ p) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i)
    (hf_sum : Summable fun i => f i ^ p) (hg_sum : Summable fun i => g i ^ p) :
    (Summable fun i => (f i + g i) ^ p) ∧
      (∑' i, (f i + g i) ^ p) ^ (1 / p) ≤
        (∑' i, f i ^ p) ^ (1 / p) + (∑' i, g i ^ p) ^ (1 / p) := by
  /-
    ι : Type u
    f g : ι → Real
    p : Real
    hp : LE.le 1 p
    hf : ∀ (i : ι), LE.le 0 (f i)
    hg : ∀ (i : ι), LE.le 0 (g i)
    hf_sum : Summable fun i => HPow.hPow (f i) p
    hg_sum : Summable fun i => HPow.hPow (g i) p
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (LE.le (HPow.hPo …
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro
    ι : Type u
    g : ι → Real
    p : Real
    hp : LE.le 1 p
    hg : ∀ (i : ι), LE.le 0 (g i)
    hg_sum : Summable fun i => HPow.hPow (g i) p
    f : ι → NNReal
    hf_sum : Summable fun i => HPow.hPow ((fun i => ↑(f i)) i) p
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd ((fun i => ↑(f i)) i) (g i)) p)  …
  -/
  lift g to ι → ℝ≥0 using hg
  -- After https://github.com/leanprover/lean4/pull/2734, `norm_cast` needs help with beta reduction.
  /-
    case intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f : ι → NNReal
    hf_sum : Summable fun i => HPow.hPow ((fun i => ↑(f i)) i) p
    g : ι → NNReal
    hg_sum : Summable fun i => HPow.hPow ((fun i => ↑(g i)) i) p
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd ((fun i => ↑(f i)) i) ((fun i => …
  -/
  beta_reduce at *
  /-
    case intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f : ι → NNReal
    hf_sum : Summable fun i => HPow.hPow (↑(f i)) p
    g : ι → NNReal
    hg_sum : Summable fun i => HPow.hPow (↑(g i)) p
    ⊢ And (Summable fun i => HPow.hPow (HAdd.hAdd ↑(f i) ↑(g i)) p) (LE.le (HPow.h …
  -/
  norm_cast0 at *
  /-
    case intro.intro
    ι : Type u
    p : Real
    f g : ι → NNReal
    hp : LE.le 1 p
    hf_sum : Summable fun a => HPow.hPow (f a) p
    hg_sum : Summable fun a => HPow.hPow (g a) p
    ⊢ And (Summable fun a => HPow.hPow (HAdd.hAdd (f a) (g a)) p) (LE.le (HPow.hPo …
  -/
  exact NNReal.Lp_add_le_tsum hp hf_sum hg_sum
  /-
    🎉 no goals
  -/


theorem summable_Lp_add_of_nonneg (hp : 1 ≤ p) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i)
    (hf_sum : Summable fun i => f i ^ p) (hg_sum : Summable fun i => g i ^ p) :
    Summable fun i => (f i + g i) ^ p :=
  (Lp_add_le_tsum_of_nonneg hp hf hg hf_sum hg_sum).1


theorem Lp_add_le_tsum_of_nonneg' (hp : 1 ≤ p) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i)
    (hf_sum : Summable fun i => f i ^ p) (hg_sum : Summable fun i => g i ^ p) :
    (∑' i, (f i + g i) ^ p) ^ (1 / p) ≤ (∑' i, f i ^ p) ^ (1 / p) + (∑' i, g i ^ p) ^ (1 / p) :=
  (Lp_add_le_tsum_of_nonneg hp hf hg hf_sum hg_sum).2


/-- **Minkowski inequality**: the `L_p` seminorm of the infinite sum of two vectors is less than or
equal to the infinite sum of the `L_p`-seminorms of the summands, if these infinite sums both
exist. A version for `ℝ`-valued functions. For an alternative version, convenient if the infinite
sums are not already expressed as `p`-th powers, see `Lp_add_le_tsum_of_nonneg`. -/
theorem Lp_add_le_hasSum_of_nonneg (hp : 1 ≤ p) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i) {A B : ℝ}
    (hA : 0 ≤ A) (hB : 0 ≤ B) (hfA : HasSum (fun i => f i ^ p) (A ^ p))
    (hgB : HasSum (fun i => g i ^ p) (B ^ p)) :
    ∃ C, 0 ≤ C ∧ C ≤ A + B ∧ HasSum (fun i => (f i + g i) ^ p) (C ^ p) := by
  /-
    ι : Type u
    f g : ι → Real
    p : Real
    hp : LE.le 1 p
    hf : ∀ (i : ι), LE.le 0 (f i)
    hg : ∀ (i : ι), LE.le 0 (g i)
    A B : Real
    hA : LE.le 0 A
    hB : LE.le 0 B
    hfA : HasSum (fun i => HPow.hPow (f i) p) (HPow.hPow A p)
    hgB : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd A B)) (HasSum (fun  …
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro
    ι : Type u
    g : ι → Real
    p : Real
    hp : LE.le 1 p
    hg : ∀ (i : ι), LE.le 0 (g i)
    A B : Real
    hA : LE.le 0 A
    hB : LE.le 0 B
    hgB : HasSum (fun i => HPow.hPow (g i) p) (HPow.hPow B p)
    f : ι → NNReal
    hfA : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow A p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd A B)) (HasSum (fun  …
  -/
  lift g to ι → ℝ≥0 using hg
  /-
    case intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    A B : Real
    hA : LE.le 0 A
    hB : LE.le 0 B
    f : ι → NNReal
    hfA : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow A p)
    g : ι → NNReal
    hgB : HasSum (fun i => HPow.hPow ((fun i => ↑(g i)) i) p) (HPow.hPow B p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd A B)) (HasSum (fun  …
  -/
  lift A to ℝ≥0 using hA
  /-
    case intro.intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    B : Real
    hB : LE.le 0 B
    f g : ι → NNReal
    hgB : HasSum (fun i => HPow.hPow ((fun i => ↑(g i)) i) p) (HPow.hPow B p)
    A : NNReal
    hfA : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow (↑A) p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd (↑A) B)) (HasSum (f …
  -/
  lift B to ℝ≥0 using hB
  -- After https://github.com/leanprover/lean4/pull/2734, `norm_cast` needs help with beta reduction.
  /-
    case intro.intro.intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A : NNReal
    hfA : HasSum (fun i => HPow.hPow ((fun i => ↑(f i)) i) p) (HPow.hPow (↑A) p)
    B : NNReal
    hgB : HasSum (fun i => HPow.hPow ((fun i => ↑(g i)) i) p) (HPow.hPow (↑B) p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd ↑A ↑B)) (HasSum (fu …
  -/
  beta_reduce at hfA hgB
  /-
    case intro.intro.intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A : NNReal
    hfA : HasSum (fun i => HPow.hPow (↑(f i)) p) (HPow.hPow (↑A) p)
    B : NNReal
    hgB : HasSum (fun i => HPow.hPow (↑(g i)) p) (HPow.hPow (↑B) p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd ↑A ↑B)) (HasSum (fu …
  -/
  norm_cast at hfA hgB
  /-
    case intro.intro.intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A B : NNReal
    hfA : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hgB : HasSum (fun a => HPow.hPow (g a) p) (HPow.hPow B p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd ↑A ↑B)) (HasSum (fu …
  -/
  obtain ⟨C, hC₁, hC₂⟩ := NNReal.Lp_add_le_hasSum hp hfA hgB
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A B : NNReal
    hfA : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hgB : HasSum (fun a => HPow.hPow (g a) p) (HPow.hPow B p)
    C : NNReal
    hC₁ : LE.le C (HAdd.hAdd A B)
    hC₂ : HasSum (fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow C p)
    ⊢ Exists fun C => And (LE.le 0 C) (And (LE.le C (HAdd.hAdd ↑A ↑B)) (HasSum (fu …
  -/
  use C
  -- After https://github.com/leanprover/lean4/pull/2734, `norm_cast` needs help with beta reduction.
  /-
    case h
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A B : NNReal
    hfA : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hgB : HasSum (fun a => HPow.hPow (g a) p) (HPow.hPow B p)
    C : NNReal
    hC₁ : LE.le C (HAdd.hAdd A B)
    hC₂ : HasSum (fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow C p)
    ⊢ And (LE.le 0 ↑C) (And (LE.le (↑C) (HAdd.hAdd ↑A ↑B)) (HasSum (fun i => HPow. …
  -/
  beta_reduce
  /-
    case h
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A B : NNReal
    hfA : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hgB : HasSum (fun a => HPow.hPow (g a) p) (HPow.hPow B p)
    C : NNReal
    hC₁ : LE.le C (HAdd.hAdd A B)
    hC₂ : HasSum (fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow C p)
    ⊢ And (LE.le 0 ↑C) (And (LE.le (↑C) (HAdd.hAdd ↑A ↑B)) (HasSum (fun i => HPow. …
  -/
  norm_cast
  /-
    case h
    ι : Type u
    p : Real
    hp : LE.le 1 p
    f g : ι → NNReal
    A B : NNReal
    hfA : HasSum (fun a => HPow.hPow (f a) p) (HPow.hPow A p)
    hgB : HasSum (fun a => HPow.hPow (g a) p) (HPow.hPow B p)
    C : NNReal
    hC₁ : LE.le C (HAdd.hAdd A B)
    hC₂ : HasSum (fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow C p)
    ⊢ And (LE.le 0 C) (And (LE.le C (HAdd.hAdd A B)) (HasSum (fun a => HPow.hPow ( …
  -/
  exact ⟨zero_le _, hC₁, hC₂⟩
  /-
    🎉 no goals
  -/


/-- **Hölder inequality**: the scalar product of two functions is bounded by the product of their
`L^p` and `L^q` norms when `p` and `q` are conjugate exponents. Version for sums over finite sets,
with `ℝ≥0∞`-valued functions. -/
theorem inner_le_Lp_mul_Lq (hpq : p.IsConjExponent q) :
    ∑ i ∈ s, f i * g i ≤ (∑ i ∈ s, f i ^ p) ^ (1 / p) * (∑ i ∈ s, g i ^ q) ^ (1 / q) := by
  /-
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  by_cases H : (∑ i ∈ s, f i ^ p) ^ (1 / p) = 0 ∨ (∑ i ∈ s, g i ^ q) ^ (1 / q) = 0
  · replace H : (∀ i ∈ s, f i = 0) ∨ ∀ i ∈ s, g i = 0 := by
      simpa [ENNReal.rpow_eq_zero_iff, hpq.pos, hpq.symm.pos, asymm hpq.pos, asymm hpq.symm.pos,
        sum_eq_zero_iff_of_nonneg] using H
    /-
      case pos
      ι : Type u
      s : Finset ι
      f g : ι → ENNReal
      p q : Real
      hpq : p.IsConjExponent q
      H : Or (∀ (i : ι), Membership.mem s i → Eq (f i) 0) (∀ (i : ι), Membership.mem …
      ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
    have : ∀ i ∈ s, f i * g i = 0 := fun i hi => by cases' H with H H <;> simp [H i hi]
    /-
      case pos
      ι : Type u
      s : Finset ι
      f g : ι → ENNReal
      p q : Real
      hpq : p.IsConjExponent q
      H : Or (∀ (i : ι), Membership.mem s i → Eq (f i) 0) (∀ (i : ι), Membership.mem …
      this : ∀ (i : ι), Membership.mem s i → Eq (HMul.hMul (f i) (g i)) 0
      ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
    simp [sum_eq_zero this]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    H : Not (Or (Eq (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  push_neg at H
  /-
    case neg
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    H : And (Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) 0)  …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  by_cases H' : (∑ i ∈ s, f i ^ p) ^ (1 / p) = ⊤ ∨ (∑ i ∈ s, g i ^ q) ^ (1 / q) = ⊤
    /-
      case pos
      ι : Type u
      s : Finset ι
      f g : ι → ENNReal
      p q : Real
      hpq : p.IsConjExponent q
      H : And (Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) 0)  …
      H' : Or (Eq (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) Top …
      ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
                             /-
                               🎉 no goals
                             -/
  · cases' H' with H' H' <;> simp [H', -one_div, -sum_eq_zero_iff, -rpow_eq_zero_iff, H]
                             /-
                               🎉 no goals
                             -/
  replace H' : (∀ i ∈ s, f i ≠ ⊤) ∧ ∀ i ∈ s, g i ≠ ⊤ := by
    simpa [ENNReal.rpow_eq_top_iff, asymm hpq.pos, asymm hpq.symm.pos, hpq.pos, hpq.symm.pos,
      ENNReal.sum_eq_top, not_or] using H'
  have := ENNReal.coe_le_coe.2 (@NNReal.inner_le_Lp_mul_Lq _ s (fun i => ENNReal.toNNReal (f i))
    (fun i => ENNReal.toNNReal (g i)) _ _ hpq)
  simp [ENNReal.coe_rpow_of_nonneg, hpq.pos.le, hpq.one_div_pos.le, hpq.symm.pos.le,
    hpq.symm.one_div_pos.le] at this
  /-
    case neg
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p q : Real
    hpq : p.IsConjExponent q
    H : And (Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) 0)  …
    H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
    this : LE.le (s.sum fun x => HMul.hMul ↑(f x).toNNReal ↑(g x).toNNReal) (HMul. …
    ⊢ LE.le (s.sum fun i => HMul.hMul (f i) (g i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  convert this using 1 <;> [skip; congr 2] <;> [skip; skip; simp; skip; simp] <;>
      /-
        case h.e'_3
        ι : Type u
        s : Finset ι
        f g : ι → ENNReal
        p q : Real
        hpq : p.IsConjExponent q
        H : And (Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) 0)  …
        H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
        this : LE.le (s.sum fun x => HMul.hMul ↑(f x).toNNReal ↑(g x).toNNReal) (HMul. …
        ⊢ Eq (s.sum fun i => HMul.hMul (f i) (g i)) (s.sum fun x => HMul.hMul ↑(f x).t …
      -/
      /-
        case h.e'_3
        ι : Type u
        s : Finset ι
        f g : ι → ENNReal
        p q : Real
        hpq : p.IsConjExponent q
        H : And (Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) 0)  …
        H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
        this : LE.le (s.sum fun x => HMul.hMul ↑(f x).toNNReal ↑(g x).toNNReal) (HMul. …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HMul.hMul (f i) (g i)) (HMul.hMul ↑(f i).toNNReal ↑(g i).toNNReal)
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4.e_a.e_a
        ι : Type u
        s : Finset ι
        f g : ι → ENNReal
        p q : Real
        hpq : p.IsConjExponent q
        H : And (Ne (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) 0)  …
        H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
        this : LE.le (s.sum fun x => HMul.hMul ↑(f x).toNNReal ↑(g x).toNNReal) (HMul. …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HPow.hPow (g i) q) (HPow.hPow (↑(g i).toNNReal) q)
      -/
      simp [H'.1 i hi, H'.2 i hi, -WithZero.coe_mul]
      /-
        🎉 no goals
      -/


/-- **Weighted Hölder inequality**. -/
lemma inner_le_weight_mul_Lp_of_nonneg (s : Finset ι) {p : ℝ} (hp : 1 ≤ p) (w f : ι → ℝ≥0∞) :
    ∑ i ∈ s, w i * f i ≤ (∑ i ∈ s, w i) ^ (1 - p⁻¹) * (∑ i ∈ s, w i * f i ^ p) ^ p⁻¹ := by
  /-
    ι : Type u
    s : Finset ι
    p : Real
    hp : LE.le 1 p
    w f : ι → ENNReal
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  obtain rfl | hp := hp.eq_or_lt
    /-
      case inl
      ι : Type u
      s : Finset ι
      w f : ι → ENNReal
      hp : LE.le 1 1
      ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  have hp₀ : 0 < p := by positivity
  /-
    case inr
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  have hp₁ : p⁻¹ < 1 := inv_lt_one_of_one_lt₀ hp
  /-
    case inr
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt (Inv.inv p) 1
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  by_cases H : (∑ i ∈ s, w i) ^ (1 - p⁻¹) = 0 ∨ (∑ i ∈ s, w i * f i ^ p) ^ p⁻¹ = 0
  · replace H : (∀ i ∈ s, w i = 0) ∨ ∀ i ∈ s, w i = 0 ∨ f i = 0 := by
      simpa [hp₀, hp₁, hp₀.not_lt, hp₁.not_lt, sum_eq_zero_iff_of_nonneg] using H
    /-
      case pos
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : Or (∀ (i : ι), Membership.mem s i → Eq (w i) 0) (∀ (i : ι), Membership.mem …
      ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
    have (i) (hi : i ∈ s) : w i * f i = 0 := by cases' H with H H <;> simp [H i hi]
    /-
      case pos
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : Or (∀ (i : ι), Membership.mem s i → Eq (w i) 0) (∀ (i : ι), Membership.mem …
      this : ∀ (i : ι), Membership.mem s i → Eq (HMul.hMul (w i) (f i)) 0
      ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
    simp [sum_eq_zero this]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt (Inv.inv p) 1
    H : Not (Or (Eq (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0)  …
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  push_neg at H
  /-
    case neg
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt (Inv.inv p) 1
    H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  by_cases H' : (∑ i ∈ s, w i) ^ (1 - p⁻¹) = ⊤ ∨ (∑ i ∈ s, w i * f i ^ p) ^ p⁻¹ = ⊤
    /-
      case pos
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
      H' : Or (Eq (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) Top.top …
      ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
    -/
                             /-
                               🎉 no goals
                             -/
  · cases' H' with H' H' <;> simp [H', -one_div, -sum_eq_zero_iff, -rpow_eq_zero_iff, H]
                             /-
                               🎉 no goals
                             -/
  replace H' : (∀ i ∈ s, w i ≠ ⊤) ∧ ∀ i ∈ s, w i * f i ^ p ≠ ⊤ := by
    simpa [rpow_eq_top_iff,hp₀, hp₁, hp₀.not_lt, hp₁.not_lt, sum_eq_top, not_or] using H'
  have := coe_le_coe.2 <| NNReal.inner_le_weight_mul_Lp s hp.le (fun i ↦ ENNReal.toNNReal (w i))
    fun i ↦ ENNReal.toNNReal (f i)
  /-
    case neg
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt (Inv.inv p) 1
    H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
    H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
    this : LE.le ↑(s.sum fun i => HMul.hMul (w i).toNNReal (f i).toNNReal) ↑(HMul. …
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  rw [coe_mul] at this
  simp_rw [coe_rpow_of_nonneg _ <| inv_nonneg.2 hp₀.le, coe_finset_sum, ← ENNReal.toNNReal_rpow,
    ← ENNReal.toNNReal_mul, sum_congr rfl fun i hi ↦ coe_toNNReal (H'.2 i hi)] at this
  /-
    case neg
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt (Inv.inv p) 1
    H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
    H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
    this : LE.le (s.sum fun x => ↑(HMul.hMul (w x) (f x)).toNNReal) (HMul.hMul (↑( …
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  simp [ENNReal.coe_rpow_of_nonneg, hp₀.le, hp₁.le] at this
  /-
    case neg
    ι : Type u
    s : Finset ι
    p : Real
    hp✝ : LE.le 1 p
    w f : ι → ENNReal
    hp : LT.lt 1 p
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt (Inv.inv p) 1
    H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
    H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
    this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
    ⊢ LE.le (s.sum fun i => HMul.hMul (w i) (f i)) (HMul.hMul (HPow.hPow (s.sum fu …
  -/
  convert this using 2 with i hi
    /-
      case h.e'_3.a
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
      H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
      this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (HMul.hMul (w i) (f i)) (HMul.hMul ↑(w i).toNNReal ↑(f i).toNNReal)
    -/
  · obtain hw | hw := eq_or_ne (w i) 0
      /-
        case h.e'_3.a.inl
        ι : Type u
        s : Finset ι
        p : Real
        hp✝ : LE.le 1 p
        w f : ι → ENNReal
        hp : LT.lt 1 p
        hp₀ : LT.lt 0 p
        hp₁ : LT.lt (Inv.inv p) 1
        H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
        H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
        this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
        i : ι
        hi : Membership.mem s i
        hw : Eq (w i) 0
        ⊢ Eq (HMul.hMul (w i) (f i)) (HMul.hMul ↑(w i).toNNReal ↑(f i).toNNReal)
      -/
    · simp [hw]
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3.a.inr
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
      H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
      this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
      i : ι
      hi : Membership.mem s i
      hw : Ne (w i) 0
      ⊢ Eq (HMul.hMul (w i) (f i)) (HMul.hMul ↑(w i).toNNReal ↑(f i).toNNReal)
    -/
    rw [coe_toNNReal (H'.1 _ hi), coe_toNNReal]
    /-
      case h.e'_3.a.inr
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
      H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
      this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
      i : ι
      hi : Membership.mem s i
      hw : Ne (w i) 0
      ⊢ Ne (f i) Top.top
    -/
    simpa [mul_eq_top, hw, hp₀, hp₀.not_lt, H'.1 _ hi] using H'.2 _ hi
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_5
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
      H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
      this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
      ⊢ Eq (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) (HPow.hPow (s. …
    -/
  · convert rfl with i hi
    /-
      case h.e'_3.h.e'_5.a
      ι : Type u
      s : Finset ι
      p : Real
      hp✝ : LE.le 1 p
      w f : ι → ENNReal
      hp : LT.lt 1 p
      hp₀ : LT.lt 0 p
      hp₁ : LT.lt (Inv.inv p) 1
      H : And (Ne (HPow.hPow (s.sum fun i => w i) (HSub.hSub 1 (Inv.inv p))) 0) (Ne  …
      H' : And (∀ (i : ι), Membership.mem s i → Ne (w i) Top.top) (∀ (i : ι), Member …
      this : LE.le (s.sum fun x => HMul.hMul ↑(w x).toNNReal ↑(f x).toNNReal) (HMul. …
      i : ι
      hi : Membership.mem s i
      ⊢ Eq (↑(w i).toNNReal) (w i)
    -/
    exact coe_toNNReal (H'.1 _ hi)
    /-
      🎉 no goals
    -/


/-- For `1 ≤ p`, the `p`-th power of the sum of `f i` is bounded above by a constant times the
sum of the `p`-th powers of `f i`. Version for sums over finite sets, with `ℝ≥0∞`-valued functions.
-/
theorem rpow_sum_le_const_mul_sum_rpow (hp : 1 ≤ p) :
    (∑ i ∈ s, f i) ^ p ≤ (card s : ℝ≥0∞) ^ (p - 1) * ∑ i ∈ s, f i ^ p := by
  /-
    ι : Type u
    s : Finset ι
    f : ι → ENNReal
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  cases' eq_or_lt_of_le hp with hp hp
    /-
      case inl
      ι : Type u
      s : Finset ι
      f : ι → ENNReal
      p : Real
      hp✝ : LE.le 1 p
      hp : Eq 1 p
      ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
    -/
  · simp [← hp]
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u
    s : Finset ι
    f : ι → ENNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  let q : ℝ := p / (p - 1)
  /-
    case inr
    ι : Type u
    s : Finset ι
    f : ι → ENNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    q : Real := HDiv.hDiv p (HSub.hSub p 1)
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  have hpq : p.IsConjExponent q := .conjExponent hp
  /-
    case inr
    ι : Type u
    s : Finset ι
    f : ι → ENNReal
    p : Real
    hp✝ : LE.le 1 p
    hp : LT.lt 1 p
    q : Real := HDiv.hDiv p (HSub.hSub p 1)
    hpq : p.IsConjExponent q
    ⊢ LE.le (HPow.hPow (s.sum fun i => f i) p) (HMul.hMul (HPow.hPow (↑s.card) (HS …
  -/
  have hp₁ : 1 / p * p = 1 := one_div_mul_cancel hpq.ne_zero
  have hq : 1 / q * p = p - 1 := by
    rw [← hpq.div_conj_eq_sub_one]
    ring
  simpa only [ENNReal.mul_rpow_of_nonneg _ _ hpq.nonneg, ← ENNReal.rpow_mul, hp₁, hq, coe_one,
    one_mul, one_rpow, rpow_one, Pi.one_apply, sum_const, Nat.smul_one_eq_cast] using
    ENNReal.rpow_le_rpow (inner_le_Lp_mul_Lq s 1 f hpq.symm) hpq.nonneg


/-- **Minkowski inequality**: the `L_p` seminorm of the sum of two vectors is less than or equal
to the sum of the `L_p`-seminorms of the summands. A version for `ℝ≥0∞` valued nonnegative
functions. -/
theorem Lp_add_le (hp : 1 ≤ p) :
    (∑ i ∈ s, (f i + g i) ^ p) ^ (1 / p) ≤
      (∑ i ∈ s, f i ^ p) ^ (1 / p) + (∑ i ∈ s, g i ^ p) ^ (1 / p) := by
  /-
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p : Real
    hp : LE.le 1 p
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  by_cases H' : (∑ i ∈ s, f i ^ p) ^ (1 / p) = ⊤ ∨ (∑ i ∈ s, g i ^ p) ^ (1 / p) = ⊤
    /-
      case pos
      ι : Type u
      s : Finset ι
      f g : ι → ENNReal
      p : Real
      hp : LE.le 1 p
      H' : Or (Eq (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p)) Top …
      ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
    -/
                             /-
                               🎉 no goals
                             -/
  · cases' H' with H' H' <;> simp [H', -one_div]
                             /-
                               🎉 no goals
                             -/
  /-
    case neg
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p : Real
    hp : LE.le 1 p
    H' : Not (Or (Eq (HPow.hPow (s.sum fun i => HPow.hPow (f i) p) (HDiv.hDiv 1 p) …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  have pos : 0 < p := lt_of_lt_of_le zero_lt_one hp
  replace H' : (∀ i ∈ s, f i ≠ ⊤) ∧ ∀ i ∈ s, g i ≠ ⊤ := by
    simpa [ENNReal.rpow_eq_top_iff, asymm pos, pos, ENNReal.sum_eq_top, not_or] using H'
  have :=
    ENNReal.coe_le_coe.2
      (@NNReal.Lp_add_le _ s (fun i => ENNReal.toNNReal (f i)) (fun i => ENNReal.toNNReal (g i)) _
        hp)
  /-
    case neg
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p : Real
    hp : LE.le 1 p
    pos : LT.lt 0 p
    H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
    this : LE.le ↑(HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd ((fun i => (f i) …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  push_cast [ENNReal.coe_rpow_of_nonneg, le_of_lt pos, le_of_lt (one_div_pos.2 pos)] at this
  /-
    case neg
    ι : Type u
    s : Finset ι
    f g : ι → ENNReal
    p : Real
    hp : LE.le 1 p
    pos : LT.lt 0 p
    H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
    this : LE.le (HPow.hPow (s.sum fun x => HPow.hPow (HAdd.hAdd ↑(f x).toNNReal ↑ …
    ⊢ LE.le (HPow.hPow (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HDiv. …
  -/
  convert this using 2 <;> [skip; congr 1; congr 1] <;>
      /-
        case h.e'_3.h.e'_5
        ι : Type u
        s : Finset ι
        f g : ι → ENNReal
        p : Real
        hp : LE.le 1 p
        pos : LT.lt 0 p
        H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
        this : LE.le (HPow.hPow (s.sum fun x => HPow.hPow (HAdd.hAdd ↑(f x).toNNReal ↑ …
        ⊢ Eq (s.sum fun i => HPow.hPow (HAdd.hAdd (f i) (g i)) p) (s.sum fun x => HPow …
      -/
      /-
        case h.e'_3.h.e'_5
        ι : Type u
        s : Finset ι
        f g : ι → ENNReal
        p : Real
        hp : LE.le 1 p
        pos : LT.lt 0 p
        H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
        this : LE.le (HPow.hPow (s.sum fun x => HPow.hPow (HAdd.hAdd ↑(f x).toNNReal ↑ …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HPow.hPow (HAdd.hAdd (f i) (g i)) p) (HPow.hPow (HAdd.hAdd ↑(f i).toNNRe …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4.h.e'_6.e_a
        ι : Type u
        s : Finset ι
        f g : ι → ENNReal
        p : Real
        hp : LE.le 1 p
        pos : LT.lt 0 p
        H' : And (∀ (i : ι), Membership.mem s i → Ne (f i) Top.top) (∀ (i : ι), Member …
        this : LE.le (HPow.hPow (s.sum fun x => HPow.hPow (HAdd.hAdd ↑(f x).toNNReal ↑ …
        i : ι
        hi : Membership.mem s i
        ⊢ Eq (HPow.hPow (g i) p) (HPow.hPow (↑(g i).toNNReal) p)
      -/
      simp [H'.1 i hi, H'.2 i hi]
      /-
        🎉 no goals
      -/


