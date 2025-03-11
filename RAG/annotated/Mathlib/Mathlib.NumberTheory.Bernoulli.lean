/-- The Bernoulli numbers:
the $n$-th Bernoulli number $B_n$ is defined recursively via
$$B_n = 1 - \sum_{k < n} \binom{n}{k}\frac{B_k}{n+1-k}$$ -/
def bernoulli' : ℕ → ℚ :=
  WellFounded.fix Nat.lt_wfRel.wf fun n bernoulli' =>
    1 - ∑ k : Fin n, n.choose k / (n - k + 1) * bernoulli' k k.2


theorem bernoulli'_def' (n : ℕ) :
    bernoulli' n = 1 - ∑ k : Fin n, n.choose k / (n - k + 1) * bernoulli' k :=
  WellFounded.fix_eq _ _ _


theorem bernoulli'_def (n : ℕ) :
    bernoulli' n = 1 - ∑ k ∈ range n, n.choose k / (n - k + 1) * bernoulli' k := by
  /-
    n : Nat
    ⊢ Eq (bernoulli' n) (HSub.hSub 1 ((Finset.range n).sum fun k => HMul.hMul (HDi …
  -/
  rw [bernoulli'_def', ← Fin.sum_univ_eq_sum_range]
  /-
    🎉 no goals
  -/


theorem bernoulli'_spec (n : ℕ) :
    (∑ k ∈ range n.succ, (n.choose (n - k) : ℚ) / (n - k + 1) * bernoulli' k) = 1 := by
  rw [sum_range_succ_comm, bernoulli'_def n, tsub_self, choose_zero_right, sub_self, zero_add,
    div_one, cast_one, one_mul, sub_add, ← sum_sub_distrib, ← sub_eq_zero, sub_sub_cancel_left,
    neg_eq_zero]
  /-
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun x => HSub.hSub (HMul.hMul (HDiv.hDiv (↑(n.choos …
  -/
  exact Finset.sum_eq_zero (fun x hx => by rw [choose_symm (le_of_lt (mem_range.1 hx)), sub_self])
  /-
    🎉 no goals
  -/


theorem bernoulli'_spec' (n : ℕ) :
    (∑ k ∈ antidiagonal n, ((k.1 + k.2).choose k.2 : ℚ) / (k.2 + 1) * bernoulli' k.1) = 1 := by
  /-
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun k => HMul.hMul (HDiv.hDi …
  -/
  refine ((sum_antidiagonal_eq_sum_range_succ_mk _ n).trans ?_).trans (bernoulli'_spec n)
  /-
    n : Nat
    ⊢ Eq ((Finset.range n.succ).sum fun k => HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd {  …
  -/
  refine sum_congr rfl fun x hx => ?_
  /-
    n x : Nat
    hx : Membership.mem (Finset.range n.succ) x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := x, snd := HSub.hSub n x }.1  …
  -/
  simp only [add_tsub_cancel_of_le, mem_range_succ_iff.mp hx, cast_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem bernoulli'_zero : bernoulli' 0 = 1 := by
  /-
    ⊢ Eq (bernoulli' 0) 1
  -/
  rw [bernoulli'_def]
  /-
    ⊢ Eq (HSub.hSub 1 ((Finset.range 0).sum fun k => HMul.hMul (HDiv.hDiv (↑(Nat.c …
  -/
  norm_num
  /-
    🎉 no goals
  -/


@[simp]
theorem bernoulli'_one : bernoulli' 1 = 1 / 2 := by
  /-
    ⊢ Eq (bernoulli' 1) (1 / 2)
  -/
  rw [bernoulli'_def]
  /-
    ⊢ Eq (HSub.hSub 1 ((Finset.range 1).sum fun k => HMul.hMul (HDiv.hDiv (↑(Nat.c …
  -/
  norm_num
  /-
    🎉 no goals
  -/


@[simp]
theorem bernoulli'_two : bernoulli' 2 = 1 / 6 := by
  /-
    ⊢ Eq (bernoulli' 2) (1 / 6)
  -/
  rw [bernoulli'_def]
  /-
    ⊢ Eq (HSub.hSub 1 ((Finset.range 2).sum fun k => HMul.hMul (HDiv.hDiv (↑(Nat.c …
  -/
  norm_num [sum_range_succ, sum_range_succ, sum_range_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem bernoulli'_three : bernoulli' 3 = 0 := by
  /-
    ⊢ Eq (bernoulli' 3) 0
  -/
  rw [bernoulli'_def]
  /-
    ⊢ Eq (HSub.hSub 1 ((Finset.range 3).sum fun k => HMul.hMul (HDiv.hDiv (↑(Nat.c …
  -/
  norm_num [sum_range_succ, sum_range_succ, sum_range_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem bernoulli'_four : bernoulli' 4 = -1 / 30 := by
  /-
    ⊢ Eq (bernoulli' 4) (-1 / 30)
  -/
  have : Nat.choose 4 2 = 6 := by decide -- shrug
  /-
    this : Eq (Nat.choose 4 2) 6
    ⊢ Eq (bernoulli' 4) (-1 / 30)
  -/
  rw [bernoulli'_def]
  /-
    this : Eq (Nat.choose 4 2) 6
    ⊢ Eq (HSub.hSub 1 ((Finset.range 4).sum fun k => HMul.hMul (HDiv.hDiv (↑(Nat.c …
  -/
  norm_num [sum_range_succ, sum_range_succ, sum_range_zero, this]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_bernoulli' (n : ℕ) : (∑ k ∈ range n, (n.choose k : ℚ) * bernoulli' k) = n := by
  /-
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.choose k)) (bernoulli' k)) ↑n
  -/
  cases' n with n
    /-
      case zero
      ⊢ Eq ((Finset.range 0).sum fun k => HMul.hMul (↑(Nat.choose 0 k)) (bernoulli'  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  suffices
    ((n + 1 : ℚ) * ∑ k ∈ range n, ↑(n.choose k) / (n - k + 1) * bernoulli' k) =
      ∑ x ∈ range n, ↑(n.succ.choose x) * bernoulli' x by
    rw_mod_cast [sum_range_succ, bernoulli'_def, ← this, choose_succ_self_right]
    ring
  /-
    case succ
    n : Nat
    ⊢ Eq (HMul.hMul (HAdd.hAdd (↑n) 1) ((Finset.range n).sum fun k => HMul.hMul (H …
  -/
  simp_rw [mul_sum, ← mul_assoc]
  /-
    case succ
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun x => HMul.hMul (HMul.hMul (HAdd.hAdd (↑n) 1) (H …
  -/
  refine sum_congr rfl fun k hk => ?_
  /-
    case succ
    n k : Nat
    hk : Membership.mem (Finset.range n) k
    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (↑n) 1) (HDiv.hDiv (↑(n.choose k)) (HAdd …
  -/
  congr
  /-
    case succ.e_a
    n k : Nat
    hk : Membership.mem (Finset.range n) k
    ⊢ Eq (HMul.hMul (HAdd.hAdd (↑n) 1) (HDiv.hDiv (↑(n.choose k)) (HAdd.hAdd (HSub …
  -/
  have : ((n - k : ℕ) : ℚ) + 1 ≠ 0 := by norm_cast
  /-
    case succ.e_a
    n k : Nat
    hk : Membership.mem (Finset.range n) k
    this : Ne (HAdd.hAdd (↑(HSub.hSub n k)) 1) 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (↑n) 1) (HDiv.hDiv (↑(n.choose k)) (HAdd.hAdd (HSub …
  -/
  field_simp [← cast_sub (mem_range.1 hk).le, mul_comm]
  /-
    case succ.e_a
    n k : Nat
    hk : Membership.mem (Finset.range n) k
    this : Ne (HAdd.hAdd (↑(HSub.hSub n k)) 1) 0
    ⊢ Eq (HMul.hMul (↑(n.choose k)) (HAdd.hAdd (↑n) 1)) (HMul.hMul (↑((HAdd.hAdd n …
  -/
  rw_mod_cast [tsub_add_eq_add_tsub (mem_range.1 hk).le, choose_mul_succ_eq]
  /-
    🎉 no goals
  -/


/-- The exponential generating function for the Bernoulli numbers `bernoulli' n`. -/
def bernoulli'PowerSeries :=
  mk fun n => algebraMap ℚ A (bernoulli' n / n !)


theorem bernoulli'PowerSeries_mul_exp_sub_one :
    bernoulli'PowerSeries A * (exp A - 1) = X * exp A := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    ⊢ Eq (HMul.hMul (bernoulli'PowerSeries A) (HSub.hSub (PowerSeries.exp A) 1)) ( …
  -/
  ext n
  -- constant coefficient is a special case
  /-
    case h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((PowerSeries.coeff A n) (HMul.hMul (bernoulli'PowerSeries A) (HSub.hSub  …
  -/
  cases' n with n
    /-
      case h.zero
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      ⊢ Eq ((PowerSeries.coeff A 0) (HMul.hMul (bernoulli'PowerSeries A) (HSub.hSub  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((PowerSeries.coeff A (HAdd.hAdd n 1)) (HMul.hMul (bernoulli'PowerSeries  …
  -/
  rw [bernoulli'PowerSeries, coeff_mul, mul_comm X, sum_antidiagonal_succ']
  suffices (∑ p ∈ antidiagonal n,
      bernoulli' p.1 / p.1! * ((p.2 + 1) * p.2! : ℚ)⁻¹) = (n ! : ℚ)⁻¹ by
    simpa [map_sum, Nat.factorial] using congr_arg (algebraMap ℚ A) this
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun p => HMul.hMul (HDiv.hDi …
  -/
  apply eq_inv_of_mul_eq_one_left
  /-
    case h.succ.h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.HasAntidiagonal.antidiagonal n).sum fun p => HMul.hMu …
  -/
  rw [sum_mul]
  /-
    case h.succ.h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun i => HMul.hMul (HMul.hMu …
  -/
  convert bernoulli'_spec' n using 1
  /-
    case h.e'_2
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun i => HMul.hMul (HMul.hMu …
  -/
  apply sum_congr rfl
  /-
    case h.e'_2
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
  -/
  simp_rw [mem_antidiagonal]
  /-
    case h.e'_2
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ ∀ (x : Prod Nat Nat), Eq (HAdd.hAdd x.1 x.2) n → Eq (HMul.hMul (HMul.hMul (H …
  -/
  rintro ⟨i, j⟩ rfl
  /-
    case h.e'_2.mk
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    i j : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (bernoulli' { fst := i, snd := j }.1) ↑{ …
  -/
  have := factorial_mul_factorial_dvd_factorial_add i j
  /-
    case h.e'_2.mk
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    i j : Nat
    this : Dvd.dvd (HMul.hMul i.factorial j.factorial) (HAdd.hAdd i j).factorial
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (bernoulli' { fst := i, snd := j }.1) ↑{ …
  -/
  field_simp [mul_comm _ (bernoulli' i), mul_assoc, add_choose]
  /-
    case h.e'_2.mk
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    i j : Nat
    this : Dvd.dvd (HMul.hMul i.factorial j.factorial) (HAdd.hAdd i j).factorial
    ⊢ Or (Or (Or (Eq (HMul.hMul (↑j.factorial) (HAdd.hAdd (↑j) 1)) (HMul.hMul (HAd …
  -/
  norm_cast
  /-
    case h.e'_2.mk
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    i j : Nat
    this : Dvd.dvd (HMul.hMul i.factorial j.factorial) (HAdd.hAdd i j).factorial
    ⊢ Or (Or (Or (Eq (HMul.hMul j.factorial (HAdd.hAdd j 1)) (HMul.hMul (HAdd.hAdd …
  -/
  simp [mul_comm (j + 1)]
  /-
    🎉 no goals
  -/


/-- Odd Bernoulli numbers (greater than 1) are zero. -/
theorem bernoulli'_odd_eq_zero {n : ℕ} (h_odd : Odd n) (hlt : 1 < n) : bernoulli' n = 0 := by
  /-
    n : Nat
    h_odd : Odd n
    hlt : LT.lt 1 n
    ⊢ Eq (bernoulli' n) 0
  -/
  let B := mk fun n => bernoulli' n / (n ! : ℚ)
  suffices (B - evalNegHom B) * (exp ℚ - 1) = X * (exp ℚ - 1) by
    cases' mul_eq_mul_right_iff.mp this with h h <;>
      simp only [PowerSeries.ext_iff, evalNegHom, coeff_X] at h
    · apply eq_zero_of_neg_eq
      specialize h n
      split_ifs at h <;> simp_all [B, h_odd.neg_one_pow, factorial_ne_zero]
    · simpa +decide [Nat.factorial] using h 1
  have h : B * (exp ℚ - 1) = X * exp ℚ := by
    simpa [bernoulli'PowerSeries] using bernoulli'PowerSeries_mul_exp_sub_one ℚ
  /-
    n : Nat
    h_odd : Odd n
    hlt : LT.lt 1 n
    B : PowerSeries Rat := PowerSeries.mk fun n => HDiv.hDiv (bernoulli' n) ↑n.fac …
    h : Eq (HMul.hMul B (HSub.hSub (PowerSeries.exp Rat) 1)) (HMul.hMul PowerSerie …
    ⊢ Eq (HMul.hMul (HSub.hSub B (PowerSeries.evalNegHom B)) (HSub.hSub (PowerSeri …
  -/
  rw [sub_mul, h, mul_sub X, sub_right_inj, ← neg_sub, mul_neg, neg_eq_iff_eq_neg]
  suffices evalNegHom (B * (exp ℚ - 1)) * exp ℚ = evalNegHom (X * exp ℚ) * exp ℚ by
    rw [map_mul, map_mul] at this -- Porting note: Why doesn't simp do this?
    simpa [mul_assoc, sub_mul, mul_comm (evalNegHom (exp ℚ)), exp_mul_exp_neg_eq_one]
  /-
    n : Nat
    h_odd : Odd n
    hlt : LT.lt 1 n
    B : PowerSeries Rat := PowerSeries.mk fun n => HDiv.hDiv (bernoulli' n) ↑n.fac …
    h : Eq (HMul.hMul B (HSub.hSub (PowerSeries.exp Rat) 1)) (HMul.hMul PowerSerie …
    ⊢ Eq (HMul.hMul (PowerSeries.evalNegHom (HMul.hMul B (HSub.hSub (PowerSeries.e …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The Bernoulli numbers are defined to be `bernoulli'` with a parity sign. -/
def bernoulli (n : ℕ) : ℚ :=
  (-1) ^ n * bernoulli' n


theorem bernoulli'_eq_bernoulli (n : ℕ) : bernoulli' n = (-1) ^ n * bernoulli n := by
  /-
    n : Nat
    ⊢ Eq (bernoulli' n) (HMul.hMul (HPow.hPow (-1) n) (bernoulli n))
  -/
  simp [bernoulli, ← mul_assoc, ← sq, ← pow_mul, mul_comm n 2]
  /-
    🎉 no goals
  -/


@[simp]
                                               /-
                                                 ⊢ Eq (bernoulli 0) 1
                                               -/
theorem bernoulli_zero : bernoulli 0 = 1 := by simp [bernoulli]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                   /-
                                                     ⊢ Eq (bernoulli 1) (-1 / 2)
                                                   -/
theorem bernoulli_one : bernoulli 1 = -1 / 2 := by norm_num [bernoulli]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem bernoulli_eq_bernoulli'_of_ne_one {n : ℕ} (hn : n ≠ 1) : bernoulli n = bernoulli' n := by
  /-
    n : Nat
    hn : Ne n 1
    ⊢ Eq (bernoulli n) (bernoulli' n)
  -/
  by_cases h0 : n = 0; · simp [h0]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    n : Nat
    hn : Ne n 1
    h0 : Not (Eq n 0)
    ⊢ Eq (bernoulli n) (bernoulli' n)
  -/
  rw [bernoulli, neg_one_pow_eq_pow_mod_two]
  /-
    case neg
    n : Nat
    hn : Ne n 1
    h0 : Not (Eq n 0)
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HMod.hMod n 2)) (bernoulli' n)) (bernoulli' n)
  -/
  cases' mod_two_eq_zero_or_one n with h h
    /-
      case neg.inl
      n : Nat
      hn : Ne n 1
      h0 : Not (Eq n 0)
      h : Eq (HMod.hMod n 2) 0
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HMod.hMod n 2)) (bernoulli' n)) (bernoulli' n)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      n : Nat
      hn : Ne n 1
      h0 : Not (Eq n 0)
      h : Eq (HMod.hMod n 2) 1
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HMod.hMod n 2)) (bernoulli' n)) (bernoulli' n)
    -/
  · simp [bernoulli'_odd_eq_zero (odd_iff.mpr h) (one_lt_iff_ne_zero_and_ne_one.mpr ⟨h0, hn⟩)]
    /-
      🎉 no goals
    -/


@[simp]
theorem sum_bernoulli (n : ℕ) :
    (∑ k ∈ range n, (n.choose k : ℚ) * bernoulli k) = if n = 1 then 1 else 0 := by
  /-
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.choose k)) (bernoulli k)) ( …
  -/
  cases' n with n
    /-
      case zero
      ⊢ Eq ((Finset.range 0).sum fun k => HMul.hMul (↑(Nat.choose 0 k)) (bernoulli k …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => HMul.hMul (↑((HAdd.hAdd n 1) …
  -/
  cases' n with n
    /-
      case succ.zero
      ⊢ Eq ((Finset.range (HAdd.hAdd 0 1)).sum fun k => HMul.hMul (↑((HAdd.hAdd 0 1) …
    -/
  · rw [sum_range_one]
    /-
      case succ.zero
      ⊢ Eq (HMul.hMul (↑((HAdd.hAdd 0 1).choose 0)) (bernoulli 0)) (ite (Eq (HAdd.hA …
    -/
    simp
    /-
      🎉 no goals
    -/
  suffices (∑ i ∈ range n, ↑((n + 2).choose (i + 2)) * bernoulli (i + 2)) = n / 2 by
    simp only [this, sum_range_succ', cast_succ, bernoulli_one, bernoulli_zero, choose_one_right,
      mul_one, choose_zero_right, cast_zero, if_false, zero_add, succ_succ_ne_one]
    ring
  /-
    case succ.succ
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (↑((HAdd.hAdd n 2).choose (HAdd. …
  -/
  have f := sum_bernoulli' n.succ.succ
  /-
    case succ.succ
    n : Nat
    f : Eq ((Finset.range n.succ.succ).sum fun k => HMul.hMul (↑(n.succ.succ.choos …
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (↑((HAdd.hAdd n 2).choose (HAdd. …
  -/
  simp_rw [sum_range_succ', cast_succ, ← eq_sub_iff_add_eq] at f
  -- Porting note: was `convert f`
  /-
    case succ.succ
    n : Nat
    f : Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.succ.succ.choose (HAdd.hA …
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (↑((HAdd.hAdd n 2).choose (HAdd. …
  -/
  refine Eq.trans ?_ (Eq.trans f ?_)
    /-
      case succ.succ.refine_1
      n : Nat
      f : Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.succ.succ.choose (HAdd.hA …
      ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (↑((HAdd.hAdd n 2).choose (HAdd. …
    -/
  · congr
    /-
      case succ.succ.refine_1.e_f
      n : Nat
      f : Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.succ.succ.choose (HAdd.hA …
      ⊢ Eq (fun i => HMul.hMul (↑((HAdd.hAdd n 2).choose (HAdd.hAdd i 2))) (bernoull …
    -/
    funext x
    /-
      case succ.succ.refine_1.e_f.h
      n : Nat
      f : Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.succ.succ.choose (HAdd.hA …
      x : Nat
      ⊢ Eq (HMul.hMul (↑((HAdd.hAdd n 2).choose (HAdd.hAdd x 2))) (bernoulli (HAdd.h …
    -/
    rw [bernoulli_eq_bernoulli'_of_ne_one (succ_ne_zero x ∘ succ.inj)]
    /-
      🎉 no goals
    -/
  · simp only [one_div, mul_one, bernoulli'_zero, cast_one, choose_zero_right, add_sub_cancel_right,
      zero_add, choose_one_right, cast_succ, cast_add, cast_one, bernoulli'_one, one_div]
    /-
      case succ.succ.refine_2
      n : Nat
      f : Eq ((Finset.range n).sum fun k => HMul.hMul (↑(n.succ.succ.choose (HAdd.hA …
      ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (↑n) 1) 1) (HAdd.hAdd (↑0) 1) …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem bernoulli_spec' (n : ℕ) :
    (∑ k ∈ antidiagonal n, ((k.1 + k.2).choose k.2 : ℚ) / (k.2 + 1) * bernoulli k.1) =
      if n = 0 then 1 else 0 := by
  /-
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun k => HMul.hMul (HDiv.hDi …
  -/
  cases' n with n
    /-
      case zero
      ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal 0).sum fun k => HMul.hMul (HDiv.hDi …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  rw [if_neg (succ_ne_zero _)]
  -- algebra facts
  /-
    case succ
    n : Nat
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  have h₁ : (1, n) ∈ antidiagonal n.succ := by simp [mem_antidiagonal, add_comm]
  /-
    case succ
    n : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  have h₂ : (n : ℚ) + 1 ≠ 0 := by norm_cast
  /-
    case succ
    n : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
    h₂ : Ne (HAdd.hAdd (↑n) 1) 0
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  have h₃ : (1 + n).choose n = n + 1 := by simp [add_comm]
  -- key equation: the corresponding fact for `bernoulli'`
  /-
    case succ
    n : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
    h₂ : Ne (HAdd.hAdd (↑n) 1) 0
    h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  have H := bernoulli'_spec' n.succ
  -- massage it to match the structure of the goal, then convert piece by piece
  /-
    case succ
    n : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
    h₂ : Ne (HAdd.hAdd (↑n) 1) 0
    h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
    H : Eq ((Finset.HasAntidiagonal.antidiagonal n.succ).sum fun k => HMul.hMul (H …
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  rw [sum_eq_add_sum_diff_singleton h₁] at H ⊢
  /-
    case succ
    n : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
    h₂ : Ne (HAdd.hAdd (↑n) 1) 0
    h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
    H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 { …
  -/
  apply add_eq_of_eq_sub'
  /-
    case succ.h
    n : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
    h₂ : Ne (HAdd.hAdd (↑n) 1) 0
    h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
    H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
    ⊢ Eq ((SDiff.sdiff (Finset.HasAntidiagonal.antidiagonal n.succ) (Singleton.sin …
  -/
  convert eq_sub_of_add_eq' H using 1
    /-
      case h.e'_2
      n : Nat
      h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
      h₂ : Ne (HAdd.hAdd (↑n) 1) 0
      h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
      H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
      ⊢ Eq ((SDiff.sdiff (Finset.HasAntidiagonal.antidiagonal n.succ) (Singleton.sin …
    -/
  · refine sum_congr rfl fun p h => ?_
    /-
      case h.e'_2
      n : Nat
      h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
      h₂ : Ne (HAdd.hAdd (↑n) 1) 0
      h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
      H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
      p : Prod Nat Nat
      h : Membership.mem (SDiff.sdiff (Finset.HasAntidiagonal.antidiagonal n.succ) ( …
      ⊢ Eq (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd p.1 p.2).choose p.2)) (HAdd.hAdd (↑p. …
    -/
    obtain ⟨h', h''⟩ : p ∈ _ ∧ p ≠ _ := by rwa [mem_sdiff, mem_singleton] at h
    /-
      case h.e'_2.intro
      n : Nat
      h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
      h₂ : Ne (HAdd.hAdd (↑n) 1) 0
      h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
      H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
      p : Prod Nat Nat
      h : Membership.mem (SDiff.sdiff (Finset.HasAntidiagonal.antidiagonal n.succ) ( …
      h' : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) p
      h'' : Ne p { fst := 1, snd := n }
      ⊢ Eq (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd p.1 p.2).choose p.2)) (HAdd.hAdd (↑p. …
    -/
    simp [bernoulli_eq_bernoulli'_of_ne_one ((not_congr (antidiagonal_congr h' h₁)).mp h'')]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      n : Nat
      h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
      h₂ : Ne (HAdd.hAdd (↑n) 1) 0
      h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
      H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
      ⊢ Eq (HSub.hSub 0 (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
    -/
  · field_simp [h₃]
    /-
      case h.e'_3
      n : Nat
      h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) { fst := 1, s …
      h₂ : Ne (HAdd.hAdd (↑n) 1) 0
      h₃ : Eq ((HAdd.hAdd 1 n).choose n) (HAdd.hAdd n 1)
      H : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (↑((HAdd.hAdd { fst := 1, snd := n }.1 …
      ⊢ Eq 1 (HSub.hSub 2 1)
    -/
    norm_num
    /-
      🎉 no goals
    -/


/-- The exponential generating function for the Bernoulli numbers `bernoulli n`. -/
def bernoulliPowerSeries :=
  mk fun n => algebraMap ℚ A (bernoulli n / n !)


theorem bernoulliPowerSeries_mul_exp_sub_one : bernoulliPowerSeries A * (exp A - 1) = X := by
  /-
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    ⊢ Eq (HMul.hMul (bernoulliPowerSeries A) (HSub.hSub (PowerSeries.exp A) 1)) Po …
  -/
  ext n
  -- constant coefficient is a special case
  /-
    case h
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq ((PowerSeries.coeff A n) (HMul.hMul (bernoulliPowerSeries A) (HSub.hSub ( …
  -/
  cases' n with n
    /-
      case h.zero
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      ⊢ Eq ((PowerSeries.coeff A 0) (HMul.hMul (bernoulliPowerSeries A) (HSub.hSub ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
  simp only [bernoulliPowerSeries, coeff_mul, coeff_X, sum_antidiagonal_succ', one_div, coeff_mk,
    coeff_one, coeff_exp, LinearMap.map_sub, factorial, if_pos, cast_succ, cast_one, cast_mul,
    sub_zero, RingHom.map_one, add_eq_zero, if_false, _root_.inv_one, zero_add, one_ne_zero,
    mul_zero, and_false, sub_self, ← RingHom.map_mul, ← map_sum]
  /-
    case h.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv (bernoulli (HAdd.hAd …
  -/
  cases' n with n
    /-
      case h.succ.zero
      A : Type u_1
      inst✝¹ : CommRing A
      inst✝ : Algebra Rat A
      ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv (bernoulli (HAdd.hAd …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv (bernoulli (HAdd.hAd …
  -/
  rw [if_neg n.succ_succ_ne_one]
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv (bernoulli (HAdd.hAd …
  -/
  have hfact : ∀ m, (m ! : ℚ) ≠ 0 := fun m => mod_cast factorial_ne_zero m
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv (bernoulli (HAdd.hAd …
  -/
  have hite2 : ite (n.succ = 0) 1 0 = (0 : ℚ) := if_neg n.succ_ne_zero
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap Rat A) (HDiv.hDiv (bernoulli (HAdd.hAd …
  -/
  simp only [CharP.cast_eq_zero, zero_add, inv_one, map_one, sub_self, mul_zero, add_eq]
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    ⊢ Eq ((algebraMap Rat A) ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1) …
  -/
  rw [← map_zero (algebraMap ℚ A), ← zero_div (n.succ ! : ℚ), ← hite2, ← bernoulli_spec', sum_div]
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    ⊢ Eq ((algebraMap Rat A) ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd n 1) …
  -/
  refine congr_arg (algebraMap ℚ A) (sum_congr rfl fun x h => eq_div_of_mul_eq (hfact n.succ) ?_)
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    x : Prod Nat Nat
    h : Membership.mem (Finset.HasAntidiagonal.antidiagonal n.succ) x
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (bernoulli x.1) ↑x.1.factorial) (Inv.inv …
  -/
  rw [mem_antidiagonal] at h
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    x : Prod Nat Nat
    h : Eq (HAdd.hAdd x.1 x.2) n.succ
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (bernoulli x.1) ↑x.1.factorial) (Inv.inv …
  -/
  rw [← h, add_choose, cast_div_charZero (factorial_mul_factorial_dvd_factorial_add _ _)]
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    x : Prod Nat Nat
    h : Eq (HAdd.hAdd x.1 x.2) n.succ
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (bernoulli x.1) ↑x.1.factorial) (Inv.inv …
  -/
  field_simp [hfact x.1, mul_comm _ (bernoulli x.1), mul_assoc]
  /-
    case h.succ.succ
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra Rat A
    n : Nat
    hfact : ∀ (m : Nat), Ne (↑m.factorial) 0
    hite2 : Eq (ite (Eq n.succ 0) 1 0) 0
    x : Prod Nat Nat
    h : Eq (HAdd.hAdd x.1 x.2) n.succ
    ⊢ Or (Or (Eq (HMul.hMul (↑x.2.factorial) (HAdd.hAdd (↑x.2) 1)) (HMul.hMul (HAd …
  -/
  left; left; ring
              /-
                🎉 no goals
              -/


/-- **Faulhaber's theorem** relating the **sum of p-th powers** to the Bernoulli numbers:
$$\sum_{k=0}^{n-1} k^p = \sum_{i=0}^p B_i\binom{p+1}{i}\frac{n^{p+1-i}}{p+1}.$$
See https://proofwiki.org/wiki/Faulhaber%27s_Formula and [orosi2018faulhaber] for
the proof provided here. -/
theorem sum_range_pow (n p : ℕ) :
    (∑ k ∈ range n, (k : ℚ) ^ p) =
      ∑ i ∈ range (p + 1), bernoulli i * ((p + 1).choose i) * (n : ℚ) ^ (p + 1 - i) / (p + 1) := by
  /-
    n p : Nat
    ⊢ Eq ((Finset.range n).sum fun k => HPow.hPow (↑k) p) ((Finset.range (HAdd.hAd …
  -/
  have hne : ∀ m : ℕ, (m ! : ℚ) ≠ 0 := fun m => mod_cast factorial_ne_zero m
  -- compute the Cauchy product of two power series
  have h_cauchy :
    ((mk fun p => bernoulli p / p !) * mk fun q => coeff ℚ (q + 1) (exp ℚ ^ n)) =
      mk fun p => ∑ i ∈ range (p + 1),
          bernoulli i * (p + 1).choose i * (n : ℚ) ^ (p + 1 - i) / (p + 1)! := by
    ext q : 1
    let f a b := bernoulli a / a ! * coeff ℚ (b + 1) (exp ℚ ^ n)
    -- key step: use `PowerSeries.coeff_mul` and then rewrite sums
    simp only [f, coeff_mul, coeff_mk, cast_mul, sum_antidiagonal_eq_sum_range_succ f]
    apply sum_congr rfl
    intros m h
    simp only [f, exp_pow_eq_rescale_exp, rescale, one_div, coeff_mk, RingHom.coe_mk, coeff_exp,
      RingHom.id_apply, cast_mul, Algebra.id.map_eq_id]
    -- manipulate factorials and binomial coefficients
    simp? at h says simp only [succ_eq_add_one, mem_range] at h
    rw [choose_eq_factorial_div_factorial h.le, eq_comm, div_eq_iff (hne q.succ), succ_eq_add_one,
      mul_assoc _ _ (q.succ ! : ℚ), mul_comm _ (q.succ ! : ℚ), ← mul_assoc, div_mul_eq_mul_div]
    simp only [add_eq, add_zero, IsUnit.mul_iff, Nat.isUnit_iff, succ.injEq, cast_mul,
      cast_succ, MonoidHom.coe_mk, OneHom.coe_mk, coeff_exp, Algebra.id.map_eq_id, one_div,
      map_inv₀, map_natCast, coeff_mk, mul_inv_rev]
    rw [mul_comm ((n : ℚ) ^ (q - m + 1)), ← mul_assoc _ _ ((n : ℚ) ^ (q - m + 1)), ← one_div,
      mul_one_div, div_div, tsub_add_eq_add_tsub (le_of_lt_succ h), cast_div, cast_mul]
    · ring
    · exact factorial_mul_factorial_dvd_factorial h.le
    · simp [hne, factorial_ne_zero]
  -- same as our goal except we pull out `p!` for convenience
  have hps :
    (∑ k ∈ range n, (k : ℚ) ^ p) =
      (∑ i ∈ range (p + 1),
          bernoulli i * (p + 1).choose i * (n : ℚ) ^ (p + 1 - i) / (p + 1)!) * p ! := by
    suffices
      (mk fun p => ∑ k ∈ range n, (k : ℚ) ^ p * algebraMap ℚ ℚ p !⁻¹) =
        mk fun p =>
          ∑ i ∈ range (p + 1), bernoulli i * (p + 1).choose i * (n : ℚ) ^ (p + 1 - i) / (p + 1)! by
      rw [← div_eq_iff (hne p), div_eq_mul_inv, sum_mul]
      rw [PowerSeries.ext_iff] at this
      simpa using this p
    -- the power series `exp ℚ - 1` is non-zero, a fact we need in order to use `mul_right_inj'`
    have hexp : exp ℚ - 1 ≠ 0 := by
      simp only [exp, PowerSeries.ext_iff, Ne, not_forall]
      use 1
      simp [factorial_ne_zero]
    have h_r : exp ℚ ^ n - 1 = X * mk fun p => coeff ℚ (p + 1) (exp ℚ ^ n) := by
      have h_const : C ℚ (constantCoeff ℚ (exp ℚ ^ n)) = 1 := by simp
      rw [← h_const, sub_const_eq_X_mul_shift]
    -- key step: a chain of equalities of power series
    -- Porting note: altered proof slightly
    rw [← mul_right_inj' hexp, mul_comm]
    rw [← exp_pow_sum, geom_sum_mul, h_r, ← bernoulliPowerSeries_mul_exp_sub_one,
      bernoulliPowerSeries, mul_right_comm]
    simp only [mul_comm, mul_eq_mul_left_iff, hexp, or_false]
    refine Eq.trans (mul_eq_mul_right_iff.mpr ?_) (Eq.trans h_cauchy ?_)
    · left
      congr
    · simp only [mul_comm, factorial, cast_succ, cast_pow]

  -- massage `hps` into our goal
  /-
    n p : Nat
    hne : ∀ (m : Nat), Ne (↑m.factorial) 0
    h_cauchy : Eq (HMul.hMul (PowerSeries.mk fun p => HDiv.hDiv (bernoulli p) ↑p.f …
    hps : Eq ((Finset.range n).sum fun k => HPow.hPow (↑k) p) (HMul.hMul ((Finset. …
    ⊢ Eq ((Finset.range n).sum fun k => HPow.hPow (↑k) p) ((Finset.range (HAdd.hAd …
  -/
  rw [hps, sum_mul]
  /-
    n p : Nat
    hne : ∀ (m : Nat), Ne (↑m.factorial) 0
    h_cauchy : Eq (HMul.hMul (PowerSeries.mk fun p => HDiv.hDiv (bernoulli p) ↑p.f …
    hps : Eq ((Finset.range n).sum fun k => HPow.hPow (↑k) p) (HMul.hMul ((Finset. …
    ⊢ Eq ((Finset.range (HAdd.hAdd p 1)).sum fun i => HMul.hMul (HDiv.hDiv (HMul.h …
  -/
  refine sum_congr rfl fun x _ => ?_
  /-
    n p : Nat
    hne : ∀ (m : Nat), Ne (↑m.factorial) 0
    h_cauchy : Eq (HMul.hMul (PowerSeries.mk fun p => HDiv.hDiv (bernoulli p) ↑p.f …
    hps : Eq ((Finset.range n).sum fun k => HPow.hPow (↑k) p) (HMul.hMul ((Finset. …
    x : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd p 1)) x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli x) ↑((HAdd.hAdd p  …
  -/
  field_simp [mul_right_comm _ ↑p !, ← mul_assoc _ _ ↑p !, factorial]
  /-
    n p : Nat
    hne : ∀ (m : Nat), Ne (↑m.factorial) 0
    h_cauchy : Eq (HMul.hMul (PowerSeries.mk fun p => HDiv.hDiv (bernoulli p) ↑p.f …
    hps : Eq ((Finset.range n).sum fun k => HPow.hPow (↑k) p) (HMul.hMul ((Finset. …
    x : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd p 1)) x
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (bernoulli x) ↑((HAdd.hAdd p  …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Alternate form of **Faulhaber's theorem**, relating the sum of p-th powers to the Bernoulli
numbers: $$\sum_{k=1}^{n} k^p = \sum_{i=0}^p (-1)^iB_i\binom{p+1}{i}\frac{n^{p+1-i}}{p+1}.$$
Deduced from `sum_range_pow`. -/
theorem sum_Ico_pow (n p : ℕ) :
    (∑ k ∈ Ico 1 (n + 1), (k : ℚ) ^ p) =
      ∑ i ∈ range (p + 1), bernoulli' i * (p + 1).choose i * (n : ℚ) ^ (p + 1 - i) / (p + 1) := by
  /-
    n p : Nat
    ⊢ Eq ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun k => HPow.hPow (↑k) p) ((Finset.r …
  -/
  rw [← Nat.cast_succ]
  -- dispose of the trivial case
  /-
    n p : Nat
    ⊢ Eq ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun k => HPow.hPow (↑k) p) ((Finset.r …
  -/
  cases' p with p
    /-
      case zero
      n : Nat
      ⊢ Eq ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun k => HPow.hPow (↑k) 0) ((Finset.r …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n p : Nat
    ⊢ Eq ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun k => HPow.hPow (↑k) (HAdd.hAdd p  …
  -/
  let f i := bernoulli i * p.succ.succ.choose i * (n : ℚ) ^ (p.succ.succ - i) / p.succ.succ
  /-
    case succ
    n p : Nat
    f : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli i) ↑(p.su …
    ⊢ Eq ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun k => HPow.hPow (↑k) (HAdd.hAdd p  …
  -/
  let f' i := bernoulli' i * p.succ.succ.choose i * (n : ℚ) ^ (p.succ.succ - i) / p.succ.succ
  /-
    case succ
    n p : Nat
    f : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli i) ↑(p.su …
    f' : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli' i) ↑(p. …
    ⊢ Eq ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun k => HPow.hPow (↑k) (HAdd.hAdd p  …
  -/
  suffices (∑ k ∈ Ico 1 n.succ, (k : ℚ) ^ p.succ) = ∑ i ∈ range p.succ.succ, f' i by convert this
  -- prove some algebraic facts that will make things easier for us later on
  /-
    case succ
    n p : Nat
    f : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli i) ↑(p.su …
    f' : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli' i) ↑(p. …
    ⊢ Eq ((Finset.Ico 1 n.succ).sum fun k => HPow.hPow (↑k) p.succ) ((Finset.range …
  -/
  have hle := Nat.le_add_left 1 n
  /-
    case succ
    n p : Nat
    f : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli i) ↑(p.su …
    f' : Nat → Rat := fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (bernoulli' i) ↑(p. …
    hle : LE.le 1 (HAdd.hAdd n 1)
    ⊢ Eq ((Finset.Ico 1 n.succ).sum fun k => HPow.hPow (↑k) p.succ) ((Finset.range …
  -/
  have hne : (p + 1 + 1 : ℚ) ≠ 0 := by norm_cast
  have h1 : ∀ r : ℚ, r * (p + 1 + 1) * (n : ℚ) ^ p.succ / (p + 1 + 1 : ℚ) = r * (n : ℚ) ^ p.succ :=
      fun r => by rw [mul_div_right_comm, mul_div_cancel_right₀ _ hne]
  have h2 : f 1 + (n : ℚ) ^ p.succ = 1 / 2 * (n : ℚ) ^ p.succ := by
    simp_rw [f, bernoulli_one, choose_one_right, succ_sub_succ_eq_sub, cast_succ, tsub_zero, h1]
    ring
  have :
    (∑ i ∈ range p, bernoulli (i + 2) * (p + 2).choose (i + 2) * (n : ℚ) ^ (p - i) / ↑(p + 2)) =
      ∑ i ∈ range p, bernoulli' (i + 2) * (p + 2).choose (i + 2) * (n : ℚ) ^ (p - i) / ↑(p + 2) :=
    sum_congr rfl fun i _ => by rw [bernoulli_eq_bernoulli'_of_ne_one (succ_succ_ne_one i)]
  calc
    (-- replace sum over `Ico` with sum over `range` and simplify
        ∑ k ∈ Ico 1 n.succ, (k : ℚ) ^ p.succ)
    _ = ∑ k ∈ range n.succ, (k : ℚ) ^ p.succ := by simp [sum_Ico_eq_sub _ hle, succ_ne_zero]
    -- extract the last term of the sum
    _ = (∑ k ∈ range n, (k : ℚ) ^ p.succ) + (n : ℚ) ^ p.succ := by rw [sum_range_succ]
    -- apply the key lemma, `sum_range_pow`
    _ = (∑ i ∈ range p.succ.succ, f i) + (n : ℚ) ^ p.succ := by simp [f, sum_range_pow]
    -- extract the first two terms of the sum
    _ = (∑ i ∈ range p, f i.succ.succ) + f 1 + f 0 + (n : ℚ) ^ p.succ := by
      simp_rw [sum_range_succ']
    _ = (∑ i ∈ range p, f i.succ.succ) + (f 1 + (n : ℚ) ^ p.succ) + f 0 := by ring
    _ = (∑ i ∈ range p, f i.succ.succ) + 1 / 2 * (n : ℚ) ^ p.succ + f 0 := by rw [h2]
    -- convert from `bernoulli` to `bernoulli'`
    _ = (∑ i ∈ range p, f' i.succ.succ) + f' 1 + f' 0 := by
      simpa [f, f', h1, fun i => show i + 2 = i + 1 + 1 from rfl]
    -- rejoin the first two terms of the sum
    _ = ∑ i ∈ range p.succ.succ, f' i := by simp_rw [sum_range_succ']


