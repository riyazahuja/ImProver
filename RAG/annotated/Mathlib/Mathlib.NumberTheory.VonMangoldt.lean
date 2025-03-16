/-- `log` as an arithmetic function `ℕ → ℝ`. Note this is in the `ArithmeticFunction`
namespace to indicate that it is bundled as an `ArithmeticFunction` rather than being the usual
real logarithm. -/
noncomputable def log : ArithmeticFunction ℝ :=
                           /-
                             ⊢ Eq ((fun n => Real.log ↑n) 0) 0
                           -/
  ⟨fun n => Real.log n, by simp⟩
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem log_apply {n : ℕ} : log n = Real.log n :=
  rfl


/--
The `vonMangoldt` function is the function on natural numbers that returns `log p` if the input can
be expressed as `p^k` for a prime `p`.
In the case when `n` is a prime power, `Nat.minFac` will give the appropriate prime, as it is the
smallest prime factor.

In the `ArithmeticFunction` locale, we have the notation `Λ` for this function.
This is also available in the `ArithmeticFunction.vonMangoldt` locale, allowing for selective
access to the notation.
-/
noncomputable def vonMangoldt : ArithmeticFunction ℝ :=
  ⟨fun n => if IsPrimePow n then Real.log (minFac n) else 0, if_neg not_isPrimePow_zero⟩


@[inherit_doc] scoped[ArithmeticFunction] notation "Λ" => ArithmeticFunction.vonMangoldt


@[inherit_doc] scoped[ArithmeticFunction.vonMangoldt] notation "Λ" =>
  ArithmeticFunction.vonMangoldt


theorem vonMangoldt_apply {n : ℕ} : Λ n = if IsPrimePow n then Real.log (minFac n) else 0 :=
  rfl


@[simp]
                                              /-
                                                ⊢ Eq (ArithmeticFunction.vonMangoldt 1) 0
                                              -/
theorem vonMangoldt_apply_one : Λ 1 = 0 := by simp [vonMangoldt_apply]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem vonMangoldt_nonneg {n : ℕ} : 0 ≤ Λ n := by
  /-
    n : Nat
    ⊢ LE.le 0 (ArithmeticFunction.vonMangoldt n)
  -/
  rw [vonMangoldt_apply]
  /-
    n : Nat
    ⊢ LE.le 0 (ite (IsPrimePow n) (Real.log ↑n.minFac) 0)
  -/
  split_ifs
    /-
      case pos
      n : Nat
      h✝ : IsPrimePow n
      ⊢ LE.le 0 (Real.log ↑n.minFac)
    -/
  · exact Real.log_nonneg (one_le_cast.2 (Nat.minFac_pos n))
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    h✝ : Not (IsPrimePow n)
    ⊢ LE.le 0 0
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem vonMangoldt_apply_pow {n k : ℕ} (hk : k ≠ 0) : Λ (n ^ k) = Λ n := by
  /-
    n k : Nat
    hk : Ne k 0
    ⊢ Eq (ArithmeticFunction.vonMangoldt (HPow.hPow n k)) (ArithmeticFunction.vonM …
  -/
  simp only [vonMangoldt_apply, isPrimePow_pow_iff hk, pow_minFac hk]
  /-
    🎉 no goals
  -/


theorem vonMangoldt_apply_prime {p : ℕ} (hp : p.Prime) : Λ p = Real.log p := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq (ArithmeticFunction.vonMangoldt p) (Real.log ↑p)
  -/
  rw [vonMangoldt_apply, Prime.minFac_eq hp, if_pos hp.prime.isPrimePow]
  /-
    🎉 no goals
  -/


theorem vonMangoldt_ne_zero_iff {n : ℕ} : Λ n ≠ 0 ↔ IsPrimePow n := by
  /-
    n : Nat
    ⊢ Iff (Ne (ArithmeticFunction.vonMangoldt n) 0) (IsPrimePow n)
  -/
  rcases eq_or_ne n 1 with (rfl | hn); · simp [not_isPrimePow_one]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    n : Nat
    hn : Ne n 1
    ⊢ Iff (Ne (ArithmeticFunction.vonMangoldt n) 0) (IsPrimePow n)
  -/
  exact (Real.log_pos (one_lt_cast.2 (minFac_prime hn).one_lt)).ne'.ite_ne_right_iff
  /-
    🎉 no goals
  -/


theorem vonMangoldt_pos_iff {n : ℕ} : 0 < Λ n ↔ IsPrimePow n :=
  vonMangoldt_nonneg.lt_iff_ne.trans (ne_comm.trans vonMangoldt_ne_zero_iff)


theorem vonMangoldt_eq_zero_iff {n : ℕ} : Λ n = 0 ↔ ¬IsPrimePow n :=
  vonMangoldt_ne_zero_iff.not_right


theorem vonMangoldt_sum {n : ℕ} : ∑ i ∈ n.divisors, Λ i = Real.log n := by
  /-
    n : Nat
    ⊢ Eq (n.divisors.sum fun i => ArithmeticFunction.vonMangoldt i) (Real.log ↑n)
  -/
  refine recOnPrimeCoprime ?_ ?_ ?_ n
    /-
      case refine_1
      n : Nat
      ⊢ Eq ((Nat.divisors 0).sum fun i => ArithmeticFunction.vonMangoldt i) (Real.lo …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      ⊢ ∀ (p n : Nat), Nat.Prime p → Eq ((HPow.hPow p n).divisors.sum fun i => Arith …
    -/
  · intro p k hp
    rw [sum_divisors_prime_pow hp, cast_pow, Real.log_pow, Finset.sum_range_succ', Nat.pow_zero,
      vonMangoldt_apply_one]
    /-
      case refine_2
      n p k : Nat
      hp : Nat.Prime p
      ⊢ Eq (HAdd.hAdd ((Finset.range k).sum fun k => ArithmeticFunction.vonMangoldt  …
    -/
    simp [vonMangoldt_apply_pow (Nat.succ_ne_zero _), vonMangoldt_apply_prime hp]
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    n : Nat
    ⊢ ∀ (a b : Nat), LT.lt 1 a → LT.lt 1 b → a.Coprime b → Eq (a.divisors.sum fun  …
  -/
  intro a b ha' hb' hab ha hb
  /-
    case refine_3
    n a b : Nat
    ha' : LT.lt 1 a
    hb' : LT.lt 1 b
    hab : a.Coprime b
    ha : Eq (a.divisors.sum fun i => ArithmeticFunction.vonMangoldt i) (Real.log ↑a)
    hb : Eq (b.divisors.sum fun i => ArithmeticFunction.vonMangoldt i) (Real.log ↑b)
    ⊢ Eq ((HMul.hMul a b).divisors.sum fun i => ArithmeticFunction.vonMangoldt i)  …
  -/
  simp only [vonMangoldt_apply, ← sum_filter] at ha hb ⊢
  rw [mul_divisors_filter_prime_pow hab, filter_union,
    sum_union (disjoint_divisors_filter_isPrimePow hab), ha, hb, Nat.cast_mul,
    Real.log_mul (cast_ne_zero.2 (pos_of_gt ha').ne') (cast_ne_zero.2 (pos_of_gt hb').ne')]


@[simp]
theorem vonMangoldt_mul_zeta : Λ * ζ = log := by
  /-
    ⊢ Eq (HMul.hMul ArithmeticFunction.vonMangoldt ↑ArithmeticFunction.zeta) Arith …
  -/
  ext n; rw [coe_mul_zeta_apply, vonMangoldt_sum]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
                                                                          /-
                                                                            ⊢ Eq (HMul.hMul (↑ArithmeticFunction.zeta) ArithmeticFunction.vonMangoldt) Ari …
                                                                          -/
theorem zeta_mul_vonMangoldt : (ζ : ArithmeticFunction ℝ) * Λ = log := by rw [mul_comm]; simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem log_mul_moebius_eq_vonMangoldt : log * μ = Λ := by
  /-
    ⊢ Eq (HMul.hMul ArithmeticFunction.log ↑ArithmeticFunction.moebius) Arithmetic …
  -/
  rw [← vonMangoldt_mul_zeta, mul_assoc, coe_zeta_mul_coe_moebius, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem moebius_mul_log_eq_vonMangoldt : (μ : ArithmeticFunction ℝ) * log = Λ := by
  /-
    ⊢ Eq (HMul.hMul (↑ArithmeticFunction.moebius) ArithmeticFunction.log) Arithmet …
  -/
  rw [mul_comm]; simp
                 /-
                   🎉 no goals
                 -/


theorem sum_moebius_mul_log_eq {n : ℕ} : (∑ d ∈ n.divisors, (μ d : ℝ) * log d) = -Λ n := by
  simp only [← log_mul_moebius_eq_vonMangoldt, mul_comm log, mul_apply, log_apply, intCoe_apply, ←
    Finset.sum_neg_distrib, neg_mul_eq_mul_neg]
  /-
    n : Nat
    ⊢ Eq (n.divisors.sum fun x => HMul.hMul (↑(ArithmeticFunction.moebius x)) (Rea …
  -/
  rw [sum_divisorsAntidiagonal fun i j => (μ i : ℝ) * -Real.log j]
  have : (∑ i ∈ n.divisors, (μ i : ℝ) * -Real.log (n / i : ℕ)) =
      ∑ i ∈ n.divisors, ((μ i : ℝ) * Real.log i - μ i * Real.log n) := by
    apply sum_congr rfl
    simp only [and_imp, Int.cast_eq_zero, mul_eq_mul_left_iff, Ne, neg_inj, mem_divisors]
    intro m mn hn
    have : (m : ℝ) ≠ 0 := by
      rw [cast_ne_zero]
      rintro rfl
      exact hn (by simpa using mn)
    rw [Nat.cast_div mn this, Real.log_div (cast_ne_zero.2 hn) this, neg_sub, mul_sub]
  rw [this, sum_sub_distrib, ← sum_mul, ← Int.cast_sum, ← coe_mul_zeta_apply, eq_comm, sub_eq_self,
    moebius_mul_coe_zeta]
  /-
    n : Nat
    this : Eq (n.divisors.sum fun i => HMul.hMul (↑(ArithmeticFunction.moebius i)) …
    ⊢ Eq (HMul.hMul (↑(1 n)) (Real.log ↑n)) 0
  -/
                                         /-
                                           🎉 no goals
                                         -/
  rcases eq_or_ne n 1 with (hn | hn) <;> simp [hn]
                                         /-
                                           🎉 no goals
                                         -/


theorem vonMangoldt_le_log : ∀ {n : ℕ}, Λ n ≤ Real.log (n : ℝ)
            /-
              ⊢ LE.le (ArithmeticFunction.vonMangoldt 0) (Real.log ↑0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      n : Nat
      ⊢ LE.le (ArithmeticFunction.vonMangoldt (HAdd.hAdd n 1)) (Real.log ↑(HAdd.hAdd …
    -/
    rw [← vonMangoldt_sum]
    exact single_le_sum (by exact fun _ _ => vonMangoldt_nonneg)
      (mem_divisors_self _ n.succ_ne_zero)


