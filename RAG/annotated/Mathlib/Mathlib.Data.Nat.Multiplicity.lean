/-- The multiplicity of `m` in `n` is the number of positive natural numbers `i` such that `m ^ i`
divides `n`. This set is expressed by filtering `Ico 1 b` where `b` is any bound greater than
`log m n`. -/
theorem emultiplicity_eq_card_pow_dvd {m n b : ℕ} (hm : m ≠ 1) (hn : 0 < n) (hb : log m n < b) :
    emultiplicity m n = #{i ∈ Ico 1 b | m ^ i ∣ n} :=
  have fin := Nat.finiteMultiplicity_iff.2 ⟨hm, hn⟩
  calc
    emultiplicity m n = #(Ico 1 <| multiplicity m n + 1) := by
      /-
        m n b : Nat
        hm : Ne m 1
        hn : LT.lt 0 n
        hb : LT.lt (Nat.log m n) b
        fin : FiniteMultiplicity m n
        ⊢ Eq (emultiplicity m n) ↑(Finset.Ico 1 (HAdd.hAdd (multiplicity m n) 1)).card
      -/
      simp [fin.emultiplicity_eq_multiplicity]
      /-
        🎉 no goals
      -/
    _ = #{i ∈ Ico 1 b | m ^ i ∣ n} :=
      congr_arg _ <|
        congr_arg card <|
          Finset.ext fun i => by
            simp only [mem_Ico, Nat.lt_succ_iff,
              fin.pow_dvd_iff_le_multiplicity, mem_filter,
              and_assoc, and_congr_right_iff, iff_and_self]
            /-
              m n b : Nat
              hm : Ne m 1
              hn : LT.lt 0 n
              hb : LT.lt (Nat.log m n) b
              fin : FiniteMultiplicity m n
              i : Nat
              ⊢ LE.le 1 i → LE.le i (multiplicity m n) → LT.lt i b
            -/
            intro hi h
            /-
              m n b : Nat
              hm : Ne m 1
              hn : LT.lt 0 n
              hb : LT.lt (Nat.log m n) b
              fin : FiniteMultiplicity m n
              i : Nat
              hi : LE.le 1 i
              h : LE.le i (multiplicity m n)
              ⊢ LT.lt i b
            -/
            rw [← fin.pow_dvd_iff_le_multiplicity] at h
            /-
              m n b : Nat
              hm : Ne m 1
              hn : LT.lt 0 n
              hb : LT.lt (Nat.log m n) b
              fin : FiniteMultiplicity m n
              i : Nat
              hi : LE.le 1 i
              h : Dvd.dvd (HPow.hPow m i) n
              ⊢ LT.lt i b
            -/
            cases' m with m
              /-
                case zero
                n b : Nat
                hn : LT.lt 0 n
                i : Nat
                hi : LE.le 1 i
                hm : Ne 0 1
                hb : LT.lt (Nat.log 0 n) b
                fin : FiniteMultiplicity 0 n
                h : Dvd.dvd (HPow.hPow 0 i) n
                ⊢ LT.lt i b
              -/
            · rw [zero_pow, zero_dvd_iff] at h
              /-
                case zero
                n b : Nat
                hn : LT.lt 0 n
                i : Nat
                hi : LE.le 1 i
                hm : Ne 0 1
                hb : LT.lt (Nat.log 0 n) b
                fin : FiniteMultiplicity 0 n
                h : Eq n 0
                ⊢ LT.lt i b
              -/
              exacts [(hn.ne' h).elim, one_le_iff_ne_zero.1 hi]
              /-
                🎉 no goals
              -/
            /-
              case succ
              n b : Nat
              hn : LT.lt 0 n
              i : Nat
              hi : LE.le 1 i
              m : Nat
              hm : Ne (HAdd.hAdd m 1) 1
              hb : LT.lt (Nat.log (HAdd.hAdd m 1) n) b
              fin : FiniteMultiplicity (HAdd.hAdd m 1) n
              h : Dvd.dvd (HPow.hPow (HAdd.hAdd m 1) i) n
              ⊢ LT.lt i b
            -/
            refine LE.le.trans_lt ?_ hb
            exact le_log_of_pow_le (one_lt_iff_ne_zero_and_ne_one.2 ⟨m.succ_ne_zero, hm⟩)
                (le_of_dvd hn h)


theorem emultiplicity_one {p : ℕ} (hp : p.Prime) : emultiplicity p 1 = 0 :=
  emultiplicity_of_one_right hp.prime.not_unit


theorem emultiplicity_mul {p m n : ℕ} (hp : p.Prime) :
    emultiplicity p (m * n) = emultiplicity p m + emultiplicity p n :=
  _root_.emultiplicity_mul hp.prime


theorem emultiplicity_pow {p m n : ℕ} (hp : p.Prime) :
    emultiplicity p (m ^ n) = n * emultiplicity p m :=
  _root_.emultiplicity_pow hp.prime


theorem emultiplicity_self {p : ℕ} (hp : p.Prime) : emultiplicity p p = 1 :=
  (Nat.finiteMultiplicity_iff.2 ⟨hp.ne_one, hp.pos⟩).emultiplicity_self


theorem emultiplicity_pow_self {p n : ℕ} (hp : p.Prime) : emultiplicity p (p ^ n) = n :=
  _root_.emultiplicity_pow_self hp.ne_zero hp.prime.not_unit n


/-- **Legendre's Theorem**

The multiplicity of a prime in `n!` is the sum of the quotients `n / p ^ i`. This sum is expressed
over the finset `Ico 1 b` where `b` is any bound greater than `log p n`. -/
theorem emultiplicity_factorial {p : ℕ} (hp : p.Prime) :
    ∀ {n b : ℕ}, log p n < b → emultiplicity p n ! = (∑ i ∈ Ico 1 b, n / p ^ i : ℕ)
                  /-
                    p : Nat
                    hp : Nat.Prime p
                    b : Nat
                    x✝ : LT.lt (Nat.log p 0) b
                    ⊢ Eq (emultiplicity p (Nat.factorial 0)) ↑((Finset.Ico 1 b).sum fun i => HDiv. …
                  -/
  | 0, b, _ => by simp [Ico, hp.emultiplicity_one]
                  /-
                    🎉 no goals
                  -/
  | n + 1, b, hb =>
    calc
      emultiplicity p (n + 1)! = emultiplicity p n ! + emultiplicity p (n + 1) := by
        /-
          p : Nat
          hp : Nat.Prime p
          n b : Nat
          hb : LT.lt (Nat.log p (HAdd.hAdd n 1)) b
          ⊢ Eq (emultiplicity p (HAdd.hAdd n 1).factorial) (HAdd.hAdd (emultiplicity p n …
        -/
        rw [factorial_succ, hp.emultiplicity_mul, add_comm]
        /-
          🎉 no goals
        -/
      _ = (∑ i ∈ Ico 1 b, n / p ^ i : ℕ) + #{i ∈ Ico 1 b | p ^ i ∣ n + 1} := by
        rw [emultiplicity_factorial hp ((log_mono_right <| le_succ _).trans_lt hb), ←
          emultiplicity_eq_card_pow_dvd hp.ne_one (succ_pos _) hb]
      _ = (∑ i ∈ Ico 1 b, (n / p ^ i + if p ^ i ∣ n + 1 then 1 else 0) : ℕ) := by
        /-
          p : Nat
          hp : Nat.Prime p
          n b : Nat
          hb : LT.lt (Nat.log p (HAdd.hAdd n 1)) b
          ⊢ Eq (HAdd.hAdd ↑((Finset.Ico 1 b).sum fun i => HDiv.hDiv n (HPow.hPow p i)) ↑ …
        -/
        rw [sum_add_distrib, sum_boole]
        /-
          p : Nat
          hp : Nat.Prime p
          n b : Nat
          hb : LT.lt (Nat.log p (HAdd.hAdd n 1)) b
          ⊢ Eq (HAdd.hAdd ↑((Finset.Ico 1 b).sum fun i => HDiv.hDiv n (HPow.hPow p i)) ↑ …
        -/
        simp
        /-
          🎉 no goals
        -/
      _ = (∑ i ∈ Ico 1 b, (n + 1) / p ^ i : ℕ) :=
        congr_arg _ <| Finset.sum_congr rfl fun _ _ => (succ_div _ _).symm


/-- For a prime number `p`, taking `(p - 1)` times the multiplicity of `p` in `n!` equals `n` minus
the sum of base `p` digits of `n`. -/
 theorem sub_one_mul_multiplicity_factorial {n p : ℕ} (hp : p.Prime) :
     (p - 1) * multiplicity p n ! =
     n - (p.digits n).sum := by
  simp only [multiplicity_eq_of_emultiplicity_eq_some <|
      emultiplicity_factorial hp <| lt_succ_of_lt <| lt.base (log p n),
    ← Finset.sum_Ico_add' _ 0 _ 1, Ico_zero_eq_range, ←
    sub_one_mul_sum_log_div_pow_eq_sub_sum_digits]


/-- The multiplicity of `p` in `(p * (n + 1))!` is one more than the sum
  of the multiplicities of `p` in `(p * n)!` and `n + 1`. -/
theorem emultiplicity_factorial_mul_succ {n p : ℕ} (hp : p.Prime) :
    emultiplicity p (p * (n + 1))! = emultiplicity p (p * n)! + emultiplicity p (n + 1) + 1 := by
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ Eq (emultiplicity p (HMul.hMul p (HAdd.hAdd n 1)).factorial) (HAdd.hAdd (HAd …
  -/
  have hp' := hp.prime
  /-
    n p : Nat
    hp : Nat.Prime p
    hp' : _root_.Prime p
    ⊢ Eq (emultiplicity p (HMul.hMul p (HAdd.hAdd n 1)).factorial) (HAdd.hAdd (HAd …
  -/
  have h0 : 2 ≤ p := hp.two_le
  /-
    n p : Nat
    hp : Nat.Prime p
    hp' : _root_.Prime p
    h0 : LE.le 2 p
    ⊢ Eq (emultiplicity p (HMul.hMul p (HAdd.hAdd n 1)).factorial) (HAdd.hAdd (HAd …
  -/
  have h1 : 1 ≤ p * n + 1 := Nat.le_add_left _ _
  /-
    n p : Nat
    hp : Nat.Prime p
    hp' : _root_.Prime p
    h0 : LE.le 2 p
    h1 : LE.le 1 (HAdd.hAdd (HMul.hMul p n) 1)
    ⊢ Eq (emultiplicity p (HMul.hMul p (HAdd.hAdd n 1)).factorial) (HAdd.hAdd (HAd …
  -/
  have h2 : p * n + 1 ≤ p * (n + 1) := by linarith
  /-
    n p : Nat
    hp : Nat.Prime p
    hp' : _root_.Prime p
    h0 : LE.le 2 p
    h1 : LE.le 1 (HAdd.hAdd (HMul.hMul p n) 1)
    h2 : LE.le (HAdd.hAdd (HMul.hMul p n) 1) (HMul.hMul p (HAdd.hAdd n 1))
    ⊢ Eq (emultiplicity p (HMul.hMul p (HAdd.hAdd n 1)).factorial) (HAdd.hAdd (HAd …
  -/
  have h3 : p * n + 1 ≤ p * (n + 1) + 1 := by omega
  have hm : emultiplicity p (p * n)! ≠ ⊤ := by
    rw [Ne, emultiplicity_eq_top, Classical.not_not, Nat.finiteMultiplicity_iff]
    exact ⟨hp.ne_one, factorial_pos _⟩
  /-
    n p : Nat
    hp : Nat.Prime p
    hp' : _root_.Prime p
    h0 : LE.le 2 p
    h1 : LE.le 1 (HAdd.hAdd (HMul.hMul p n) 1)
    h2 : LE.le (HAdd.hAdd (HMul.hMul p n) 1) (HMul.hMul p (HAdd.hAdd n 1))
    h3 : LE.le (HAdd.hAdd (HMul.hMul p n) 1) (HAdd.hAdd (HMul.hMul p (HAdd.hAdd n  …
    hm : Ne (emultiplicity p (HMul.hMul p n).factorial) Top.top
    ⊢ Eq (emultiplicity p (HMul.hMul p (HAdd.hAdd n 1)).factorial) (HAdd.hAdd (HAd …
  -/
  revert hm
  have h4 : ∀ m ∈ Ico (p * n + 1) (p * (n + 1)), emultiplicity p m = 0 := by
    intro m hm
    rw [emultiplicity_eq_zero, ← not_dvd_iff_between_consec_multiples _ hp.pos]
    rw [mem_Ico] at hm
    exact ⟨n, lt_of_succ_le hm.1, hm.2⟩
  simp_rw [← prod_Ico_id_eq_factorial, Finset.emultiplicity_prod hp', ← sum_Ico_consecutive _ h1 h3,
    add_assoc]
  /-
    n p : Nat
    hp : Nat.Prime p
    hp' : _root_.Prime p
    h0 : LE.le 2 p
    h1 : LE.le 1 (HAdd.hAdd (HMul.hMul p n) 1)
    h2 : LE.le (HAdd.hAdd (HMul.hMul p n) 1) (HMul.hMul p (HAdd.hAdd n 1))
    h3 : LE.le (HAdd.hAdd (HMul.hMul p n) 1) (HAdd.hAdd (HMul.hMul p (HAdd.hAdd n  …
    h4 : ∀ (m : Nat), Membership.mem (Finset.Ico (HAdd.hAdd (HMul.hMul p n) 1) (HM …
    ⊢ Ne ((Finset.Ico 1 (HAdd.hAdd (HMul.hMul p n) 1)).sum fun x => emultiplicity  …
  -/
  intro h
  rw [WithTop.add_left_cancel_iff h, sum_Ico_succ_top h2, hp.emultiplicity_mul,
    hp.emultiplicity_self, sum_congr rfl h4, sum_const_zero, zero_add, add_comm 1]


/-- The multiplicity of `p` in `(p * n)!` is `n` more than that of `n!`. -/
theorem emultiplicity_factorial_mul {n p : ℕ} (hp : p.Prime) :
    emultiplicity p (p * n)! = emultiplicity p n ! + n := by
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ Eq (emultiplicity p (HMul.hMul p n).factorial) (HAdd.hAdd (emultiplicity p n …
  -/
  induction' n with n ih
    /-
      case zero
      p : Nat
      hp : Nat.Prime p
      ⊢ Eq (emultiplicity p (HMul.hMul p 0).factorial) (HAdd.hAdd (emultiplicity p ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp only [hp, emultiplicity_factorial_mul_succ, ih, factorial_succ, emultiplicity_mul,
    cast_add, cast_one, ← add_assoc]
    /-
      case succ
      p : Nat
      hp : Nat.Prime p
      n : Nat
      ih : Eq (emultiplicity p (HMul.hMul p n).factorial) (HAdd.hAdd (emultiplicity  …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (emultiplicity p n.factorial) ↑n) (emult …
    -/
    congr 1
    /-
      case succ.e_a
      p : Nat
      hp : Nat.Prime p
      n : Nat
      ih : Eq (emultiplicity p (HMul.hMul p n).factorial) (HAdd.hAdd (emultiplicity  …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (emultiplicity p n.factorial) ↑n) (emultiplicity p  …
    -/
    rw [add_comm, add_assoc]
    /-
      🎉 no goals
    -/


/-- A prime power divides `n!` iff it is at most the sum of the quotients `n / p ^ i`.
  This sum is expressed over the set `Ico 1 b` where `b` is any bound greater than `log p n` -/
theorem pow_dvd_factorial_iff {p : ℕ} {n r b : ℕ} (hp : p.Prime) (hbn : log p n < b) :
    p ^ r ∣ n ! ↔ r ≤ ∑ i ∈ Ico 1 b, n / p ^ i := by
  rw [← WithTop.coe_le_coe, ENat.some_eq_coe, ← hp.emultiplicity_factorial hbn,
    pow_dvd_iff_le_emultiplicity]


theorem emultiplicity_factorial_le_div_pred {p : ℕ} (hp : p.Prime) (n : ℕ) :
    emultiplicity p n ! ≤ (n / (p - 1) : ℕ) := by
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ LE.le (emultiplicity p n.factorial) ↑(HDiv.hDiv n (HSub.hSub p 1))
  -/
  rw [hp.emultiplicity_factorial (lt_succ_self _)]
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ LE.le ↑((Finset.Ico 1 (Nat.log p n).succ).sum fun i => HDiv.hDiv n (HPow.hPo …
  -/
  apply WithTop.coe_mono
  /-
    case a
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ LE.le ((Finset.Ico 1 (Nat.log p n).succ).sum fun i => HDiv.hDiv n (HPow.hPow …
  -/
  exact Nat.geom_sum_Ico_le hp.two_le _ _
  /-
    🎉 no goals
  -/


theorem multiplicity_choose_aux {p n b k : ℕ} (hp : p.Prime) (hkn : k ≤ n) :
    ∑ i ∈ Finset.Ico 1 b, n / p ^ i =
      ((∑ i ∈ Finset.Ico 1 b, k / p ^ i) + ∑ i ∈ Finset.Ico 1 b, (n - k) / p ^ i) +
        #{i ∈ Ico 1 b | p ^ i ≤ k % p ^ i + (n - k) % p ^ i} :=
  calc
    ∑ i ∈ Finset.Ico 1 b, n / p ^ i = ∑ i ∈ Finset.Ico 1 b, (k + (n - k)) / p ^ i := by
      /-
        p n b k : Nat
        hp : Nat.Prime p
        hkn : LE.le k n
        ⊢ Eq ((Finset.Ico 1 b).sum fun i => HDiv.hDiv n (HPow.hPow p i)) ((Finset.Ico  …
      -/
      simp only [add_tsub_cancel_of_le hkn]
      /-
        🎉 no goals
      -/
    _ = ∑ i ∈ Finset.Ico 1 b,
          (k / p ^ i + (n - k) / p ^ i + if p ^ i ≤ k % p ^ i + (n - k) % p ^ i then 1 else 0) := by
      /-
        p n b k : Nat
        hp : Nat.Prime p
        hkn : LE.le k n
        ⊢ Eq ((Finset.Ico 1 b).sum fun i => HDiv.hDiv (HAdd.hAdd k (HSub.hSub n k)) (H …
      -/
      simp only [Nat.add_div (pow_pos hp.pos _)]
      /-
        🎉 no goals
      -/
                /-
                  p n b k : Nat
                  hp : Nat.Prime p
                  hkn : LE.le k n
                  ⊢ Eq ((Finset.Ico 1 b).sum fun i => HAdd.hAdd (HAdd.hAdd (HDiv.hDiv k (HPow.hP …
                -/
    _ = _ := by simp [sum_add_distrib, sum_boole]
                /-
                  🎉 no goals
                -/


/-- The multiplicity of `p` in `choose (n + k) k` is the number of carries when `k` and `n`
  are added in base `p`. The set is expressed by filtering `Ico 1 b` where `b`
  is any bound greater than `log p (n + k)`. -/
theorem emultiplicity_choose' {p n k b : ℕ} (hp : p.Prime) (hnb : log p (n + k) < b) :
    emultiplicity p (choose (n + k) k) = #{i ∈ Ico 1 b | p ^ i ≤ k % p ^ i + n % p ^ i} := by
  have h₁ :
      emultiplicity p (choose (n + k) k) + emultiplicity p (k ! * n !) =
        #{i ∈ Ico 1 b | p ^ i ≤ k % p ^ i + n % p ^ i} + emultiplicity p (k ! * n !) := by
    rw [← hp.emultiplicity_mul, ← mul_assoc]
    have := (add_tsub_cancel_right n k) ▸ choose_mul_factorial_mul_factorial (le_add_left k n)
    rw [this, hp.emultiplicity_factorial hnb, hp.emultiplicity_mul,
      hp.emultiplicity_factorial ((log_mono_right (le_add_left k n)).trans_lt hnb),
      hp.emultiplicity_factorial ((log_mono_right (le_add_left n k)).trans_lt
      (add_comm n k ▸ hnb)), multiplicity_choose_aux hp (le_add_left k n)]
    simp [add_comm]
  /-
    p n k b : Nat
    hp : Nat.Prime p
    hnb : LT.lt (Nat.log p (HAdd.hAdd n k)) b
    h₁ : Eq (HAdd.hAdd (emultiplicity p ((HAdd.hAdd n k).choose k)) (emultiplicity …
    ⊢ Eq (emultiplicity p ((HAdd.hAdd n k).choose k)) ↑(Finset.filter (fun i => LE …
  -/
  refine (WithTop.add_right_cancel_iff ?_).1 h₁
  /-
    p n k b : Nat
    hp : Nat.Prime p
    hnb : LT.lt (Nat.log p (HAdd.hAdd n k)) b
    h₁ : Eq (HAdd.hAdd (emultiplicity p ((HAdd.hAdd n k).choose k)) (emultiplicity …
    ⊢ Ne (emultiplicity p (HMul.hMul k.factorial n.factorial)) Top.top
  -/
  apply finiteMultiplicity_iff_emultiplicity_ne_top.1
  /-
    p n k b : Nat
    hp : Nat.Prime p
    hnb : LT.lt (Nat.log p (HAdd.hAdd n k)) b
    h₁ : Eq (HAdd.hAdd (emultiplicity p ((HAdd.hAdd n k).choose k)) (emultiplicity …
    ⊢ FiniteMultiplicity p (HMul.hMul k.factorial n.factorial)
  -/
  exact Nat.finiteMultiplicity_iff.2 ⟨hp.ne_one, mul_pos (factorial_pos k) (factorial_pos n)⟩
  /-
    🎉 no goals
  -/


/-- The multiplicity of `p` in `choose n k` is the number of carries when `k` and `n - k`
  are added in base `p`. The set is expressed by filtering `Ico 1 b` where `b`
  is any bound greater than `log p n`. -/
theorem emultiplicity_choose {p n k b : ℕ} (hp : p.Prime) (hkn : k ≤ n) (hnb : log p n < b) :
    emultiplicity p (choose n k) = #{i ∈ Ico 1 b | p ^ i ≤ k % p ^ i + (n - k) % p ^ i} := by
  /-
    p n k b : Nat
    hp : Nat.Prime p
    hkn : LE.le k n
    hnb : LT.lt (Nat.log p n) b
    ⊢ Eq (emultiplicity p (n.choose k)) ↑(Finset.filter (fun i => LE.le (HPow.hPow …
  -/
  have := Nat.sub_add_cancel hkn
  /-
    p n k b : Nat
    hp : Nat.Prime p
    hkn : LE.le k n
    hnb : LT.lt (Nat.log p n) b
    this : Eq (HAdd.hAdd (HSub.hSub n k) k) n
    ⊢ Eq (emultiplicity p (n.choose k)) ↑(Finset.filter (fun i => LE.le (HPow.hPow …
  -/
  convert @emultiplicity_choose' p (n - k) k b hp _
    /-
      case h.e'_2.h.e'_4.h.e'_1
      p n k b : Nat
      hp : Nat.Prime p
      hkn : LE.le k n
      hnb : LT.lt (Nat.log p n) b
      this : Eq (HAdd.hAdd (HSub.hSub n k) k) n
      ⊢ Eq n (HAdd.hAdd (HSub.hSub n k) k)
    -/
  · rw [this]
    /-
      🎉 no goals
    -/
  /-
    p n k b : Nat
    hp : Nat.Prime p
    hkn : LE.le k n
    hnb : LT.lt (Nat.log p n) b
    this : Eq (HAdd.hAdd (HSub.hSub n k) k) n
    ⊢ LT.lt (Nat.log p (HAdd.hAdd (HSub.hSub n k) k)) b
  -/
  exact this.symm ▸ hnb
  /-
    🎉 no goals
  -/


/-- A lower bound on the multiplicity of `p` in `choose n k`. -/
theorem emultiplicity_le_emultiplicity_choose_add {p : ℕ} (hp : p.Prime) :
    ∀ n k : ℕ, emultiplicity p n ≤ emultiplicity p (choose n k) + emultiplicity p k
               /-
                 p : Nat
                 hp : Nat.Prime p
                 x✝ : Nat
                 ⊢ LE.le (emultiplicity p x✝) (HAdd.hAdd (emultiplicity p (x✝.choose 0)) (emult …
               -/
  | _, 0 => by simp
               /-
                 🎉 no goals
               -/
                   /-
                     p : Nat
                     hp : Nat.Prime p
                     n✝ : Nat
                     ⊢ LE.le (emultiplicity p 0) (HAdd.hAdd (emultiplicity p (Nat.choose 0 (HAdd.hA …
                   -/
  | 0, _ + 1 => by simp
                   /-
                     🎉 no goals
                   -/
  | n + 1, k + 1 => by
    /-
      p : Nat
      hp : Nat.Prime p
      n k : Nat
      ⊢ LE.le (emultiplicity p (HAdd.hAdd n 1)) (HAdd.hAdd (emultiplicity p ((HAdd.h …
    -/
    rw [← hp.emultiplicity_mul]
    /-
      p : Nat
      hp : Nat.Prime p
      n k : Nat
      ⊢ LE.le (emultiplicity p (HAdd.hAdd n 1)) (emultiplicity p (HMul.hMul ((HAdd.h …
    -/
    refine emultiplicity_le_emultiplicity_of_dvd_right ?_
    /-
      p : Nat
      hp : Nat.Prime p
      n k : Nat
      ⊢ Dvd.dvd (HAdd.hAdd n 1) (HMul.hMul ((HAdd.hAdd n 1).choose (HAdd.hAdd k 1))  …
    -/
    rw [← succ_mul_choose_eq]
    /-
      p : Nat
      hp : Nat.Prime p
      n k : Nat
      ⊢ Dvd.dvd (HAdd.hAdd n 1) (HMul.hMul n.succ (n.choose k))
    -/
    exact dvd_mul_right _ _
    /-
      🎉 no goals
    -/


theorem emultiplicity_choose_prime_pow_add_emultiplicity (hp : p.Prime) (hkn : k ≤ p ^ n)
    (hk0 : k ≠ 0) : emultiplicity p (choose (p ^ n) k) + emultiplicity p k = n :=
  le_antisymm
    (by
      have hdisj :
        Disjoint {i ∈ Ico 1 n.succ | p ^ i ≤ k % p ^ i + (p ^ n - k) % p ^ i}
          {i ∈ Ico 1 n.succ | p ^ i ∣ k} := by
        simp +contextual [disjoint_right, *, dvd_iff_mod_eq_zero,
          Nat.mod_lt _ (pow_pos hp.pos _)]
      rw [emultiplicity_choose hp hkn (lt_succ_self _),
        emultiplicity_eq_card_pow_dvd (ne_of_gt hp.one_lt) hk0.bot_lt
          (lt_succ_of_le (log_mono_right hkn)),
        ← Nat.cast_add]
      /-
        p n k : Nat
        hp : Nat.Prime p
        hkn : LE.le k (HPow.hPow p n)
        hk0 : Ne k 0
        hdisj : Disjoint (Finset.filter (fun i => LE.le (HPow.hPow p i) (HAdd.hAdd (HM …
        ⊢ LE.le ↑(HAdd.hAdd (Finset.filter (fun i => LE.le (HPow.hPow p i) (HAdd.hAdd  …
      -/
      apply WithTop.coe_mono
      /-
        case a
        p n k : Nat
        hp : Nat.Prime p
        hkn : LE.le k (HPow.hPow p n)
        hk0 : Ne k 0
        hdisj : Disjoint (Finset.filter (fun i => LE.le (HPow.hPow p i) (HAdd.hAdd (HM …
        ⊢ LE.le (↑(HAdd.hAdd (Finset.filter (fun i => LE.le (HPow.hPow p i) (HAdd.hAdd …
      -/
      rw [log_pow hp.one_lt, ← card_union_of_disjoint hdisj, filter_union_right]
      have filter_le_Ico := (Ico 1 n.succ).card_filter_le
        fun x => p ^ x ≤ k % p ^ x + (p ^ n - k) % p ^ x ∨ p ^ x ∣ k
      /-
        case a
        p n k : Nat
        hp : Nat.Prime p
        hkn : LE.le k (HPow.hPow p n)
        hk0 : Ne k 0
        hdisj : Disjoint (Finset.filter (fun i => LE.le (HPow.hPow p i) (HAdd.hAdd (HM …
        filter_le_Ico : LE.le (Finset.filter (fun x => Or (LE.le (HPow.hPow p x) (HAdd …
        ⊢ LE.le (↑(Finset.filter (fun x => Or (LE.le (HPow.hPow p x) (HAdd.hAdd (HMod. …
      -/
      rwa [card_Ico 1 n.succ] at filter_le_Ico)
      /-
        🎉 no goals
      -/
        /-
          p n k : Nat
          hp : Nat.Prime p
          hkn : LE.le k (HPow.hPow p n)
          hk0 : Ne k 0
          ⊢ LE.le (↑n) (HAdd.hAdd (emultiplicity p ((HPow.hPow p n).choose k)) (emultipl …
        -/
    (by rw [← hp.emultiplicity_pow_self]; exact emultiplicity_le_emultiplicity_choose_add hp _ _)
                                          /-
                                            🎉 no goals
                                          -/


theorem emultiplicity_choose_prime_pow {p n k : ℕ} (hp : p.Prime) (hkn : k ≤ p ^ n) (hk0 : k ≠ 0) :
    emultiplicity p (choose (p ^ n) k) = ↑(n - multiplicity p k) := by
  /-
    p n k : Nat
    hp : Nat.Prime p
    hkn : LE.le k (HPow.hPow p n)
    hk0 : Ne k 0
    ⊢ Eq (emultiplicity p ((HPow.hPow p n).choose k)) ↑(HSub.hSub n (multiplicity  …
  -/
  push_cast
  rw [← emultiplicity_choose_prime_pow_add_emultiplicity hp hkn hk0,
    (finiteMultiplicity_iff.2 ⟨hp.ne_one, Nat.pos_of_ne_zero hk0⟩).emultiplicity_eq_multiplicity,
    (finiteMultiplicity_iff.2 ⟨hp.ne_one, choose_pos hkn⟩).emultiplicity_eq_multiplicity]
  /-
    p n k : Nat
    hp : Nat.Prime p
    hkn : LE.le k (HPow.hPow p n)
    hk0 : Ne k 0
    ⊢ Eq (↑(multiplicity p ((HPow.hPow p n).choose k))) (HSub.hSub (HAdd.hAdd ↑(mu …
  -/
  norm_cast
  /-
    p n k : Nat
    hp : Nat.Prime p
    hkn : LE.le k (HPow.hPow p n)
    hk0 : Ne k 0
    ⊢ Eq (multiplicity p ((HPow.hPow p n).choose k)) (HSub.hSub (HAdd.hAdd (multip …
  -/
  rw [Nat.add_sub_cancel_right]
  /-
    🎉 no goals
  -/


theorem dvd_choose_pow (hp : Prime p) (hk : k ≠ 0) (hkp : k ≠ p ^ n) : p ∣ (p ^ n).choose k := by
  /-
    p n k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    hkp : Ne k (HPow.hPow p n)
    ⊢ Dvd.dvd p ((HPow.hPow p n).choose k)
  -/
  obtain hkp | hkp := hkp.symm.lt_or_lt
    /-
      case inl
      p n k : Nat
      hp : Nat.Prime p
      hk : Ne k 0
      hkp✝ : Ne k (HPow.hPow p n)
      hkp : LT.lt (HPow.hPow p n) k
      ⊢ Dvd.dvd p ((HPow.hPow p n).choose k)
    -/
  · simp [choose_eq_zero_of_lt hkp]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p n k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    hkp✝ : Ne k (HPow.hPow p n)
    hkp : LT.lt k (HPow.hPow p n)
    ⊢ Dvd.dvd p ((HPow.hPow p n).choose k)
  -/
  refine emultiplicity_ne_zero.1 fun h => hkp.not_le <| Nat.le_of_dvd hk.bot_lt ?_
  /-
    case inr
    p n k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    hkp✝ : Ne k (HPow.hPow p n)
    hkp : LT.lt k (HPow.hPow p n)
    h : Eq (emultiplicity p ((HPow.hPow p n).choose k)) 0
    ⊢ Dvd.dvd (HPow.hPow p n) k
  -/
  have H := hp.emultiplicity_choose_prime_pow_add_emultiplicity hkp.le hk
  /-
    case inr
    p n k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    hkp✝ : Ne k (HPow.hPow p n)
    hkp : LT.lt k (HPow.hPow p n)
    h : Eq (emultiplicity p ((HPow.hPow p n).choose k)) 0
    H : Eq (HAdd.hAdd (emultiplicity p ((HPow.hPow p n).choose k)) (emultiplicity  …
    ⊢ Dvd.dvd (HPow.hPow p n) k
  -/
  rw [h, zero_add, emultiplicity_eq_coe] at H
  /-
    case inr
    p n k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    hkp✝ : Ne k (HPow.hPow p n)
    hkp : LT.lt k (HPow.hPow p n)
    h : Eq (emultiplicity p ((HPow.hPow p n).choose k)) 0
    H : And (Dvd.dvd (HPow.hPow p n) k) (Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1) …
    ⊢ Dvd.dvd (HPow.hPow p n) k
  -/
  exact H.1
  /-
    🎉 no goals
  -/


theorem dvd_choose_pow_iff (hp : Prime p) : p ∣ (p ^ n).choose k ↔ k ≠ 0 ∧ k ≠ p ^ n := by
  /-
    p n k : Nat
    hp : Nat.Prime p
    ⊢ Iff (Dvd.dvd p ((HPow.hPow p n).choose k)) (And (Ne k 0) (Ne k (HPow.hPow p  …
  -/
  refine ⟨fun h => ⟨?_, ?_⟩, fun h => dvd_choose_pow hp h.1 h.2⟩ <;> rintro rfl <;>
    /-
      case refine_1
      p n : Nat
      hp : Nat.Prime p
      h : Dvd.dvd p ((HPow.hPow p n).choose 0)
      ⊢ False
    -/
    /-
      🎉 no goals
    -/
    simp [hp.ne_one] at h
    /-
      🎉 no goals
    -/


theorem emultiplicity_two_factorial_lt : ∀ {n : ℕ} (_ : n ≠ 0), emultiplicity 2 n ! < n := by
  /-
    ⊢ ∀ {n : Nat}, Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
  -/
  have h2 := prime_two.prime
  /-
    h2 : _root_.Prime 2
    ⊢ ∀ {n : Nat}, Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
  -/
  refine binaryRec ?_ ?_
    /-
      case refine_1
      h2 : _root_.Prime 2
      ⊢ Ne 0 0 → LT.lt (emultiplicity 2 (Nat.factorial 0)) ↑0
    -/
  · exact fun h => False.elim <| h rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      h2 : _root_.Prime 2
      ⊢ ∀ (b : Bool) (n : Nat), (Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n) →  …
    -/
  · intro b n ih h
    /-
      case refine_2
      h2 : _root_.Prime 2
      b : Bool
      n : Nat
      ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
      h : Ne (Nat.bit b n) 0
      ⊢ LT.lt (emultiplicity 2 (Nat.bit b n).factorial) ↑(Nat.bit b n)
    -/
    by_cases hn : n = 0
      /-
        case pos
        h2 : _root_.Prime 2
        b : Bool
        n : Nat
        ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
        h : Ne (Nat.bit b n) 0
        hn : Eq n 0
        ⊢ LT.lt (emultiplicity 2 (Nat.bit b n).factorial) ↑(Nat.bit b n)
      -/
    · subst hn
      /-
        case pos
        h2 : _root_.Prime 2
        b : Bool
        ih : Ne 0 0 → LT.lt (emultiplicity 2 (Nat.factorial 0)) ↑0
        h : Ne (Nat.bit b 0) 0
        ⊢ LT.lt (emultiplicity 2 (Nat.bit b 0).factorial) ↑(Nat.bit b 0)
      -/
      simp only [ne_eq, bit_eq_zero_iff, true_and, Bool.not_eq_false] at h
      /-
        case pos
        h2 : _root_.Prime 2
        b : Bool
        ih : Ne 0 0 → LT.lt (emultiplicity 2 (Nat.factorial 0)) ↑0
        h : Eq b Bool.true
        ⊢ LT.lt (emultiplicity 2 (Nat.bit b 0).factorial) ↑(Nat.bit b 0)
      -/
      simp only [bit, h, cond_true, mul_zero, zero_add, factorial_one]
      /-
        case pos
        h2 : _root_.Prime 2
        b : Bool
        ih : Ne 0 0 → LT.lt (emultiplicity 2 (Nat.factorial 0)) ↑0
        h : Eq b Bool.true
        ⊢ LT.lt (emultiplicity 2 1) ↑1
      -/
      rw [Prime.emultiplicity_one]
        /-
          case pos
          h2 : _root_.Prime 2
          b : Bool
          ih : Ne 0 0 → LT.lt (emultiplicity 2 (Nat.factorial 0)) ↑0
          h : Eq b Bool.true
          ⊢ LT.lt 0 ↑1
        -/
      · exact zero_lt_one
        /-
          🎉 no goals
        -/
        /-
          case pos
          h2 : _root_.Prime 2
          b : Bool
          ih : Ne 0 0 → LT.lt (emultiplicity 2 (Nat.factorial 0)) ↑0
          h : Eq b Bool.true
          ⊢ Nat.Prime 2
        -/
      · decide
        /-
          🎉 no goals
        -/
    have : emultiplicity 2 (2 * n)! < (2 * n : ℕ) := by
      rw [prime_two.emultiplicity_factorial_mul]
      rw [two_mul]
      push_cast
      apply WithTop.add_lt_add_right _ (ih hn)
      exact Ne.symm nofun
    /-
      case neg
      h2 : _root_.Prime 2
      b : Bool
      n : Nat
      ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
      h : Ne (Nat.bit b n) 0
      hn : Not (Eq n 0)
      this : LT.lt (emultiplicity 2 (HMul.hMul 2 n).factorial) ↑(HMul.hMul 2 n)
      ⊢ LT.lt (emultiplicity 2 (Nat.bit b n).factorial) ↑(Nat.bit b n)
    -/
    cases b
      /-
        case neg.false
        h2 : _root_.Prime 2
        n : Nat
        ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
        hn : Not (Eq n 0)
        this : LT.lt (emultiplicity 2 (HMul.hMul 2 n).factorial) ↑(HMul.hMul 2 n)
        h : Ne (Nat.bit Bool.false n) 0
        ⊢ LT.lt (emultiplicity 2 (Nat.bit Bool.false n).factorial) ↑(Nat.bit Bool.fals …
      -/
    · simpa
      /-
        🎉 no goals
      -/
    · suffices emultiplicity 2 (2 * n + 1) + emultiplicity 2 (2 * n)! < ↑(2 * n) + 1 by
        simpa [emultiplicity_mul, h2, prime_two, bit, factorial]
      /-
        case neg.true
        h2 : _root_.Prime 2
        n : Nat
        ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
        hn : Not (Eq n 0)
        this : LT.lt (emultiplicity 2 (HMul.hMul 2 n).factorial) ↑(HMul.hMul 2 n)
        h : Ne (Nat.bit Bool.true n) 0
        ⊢ LT.lt (HAdd.hAdd (emultiplicity 2 (HAdd.hAdd (HMul.hMul 2 n) 1)) (emultiplic …
      -/
      rw [emultiplicity_eq_zero.2 (two_not_dvd_two_mul_add_one n), zero_add]
      /-
        case neg.true
        h2 : _root_.Prime 2
        n : Nat
        ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
        hn : Not (Eq n 0)
        this : LT.lt (emultiplicity 2 (HMul.hMul 2 n).factorial) ↑(HMul.hMul 2 n)
        h : Ne (Nat.bit Bool.true n) 0
        ⊢ LT.lt (emultiplicity 2 (HMul.hMul 2 n).factorial) (HAdd.hAdd (↑(HMul.hMul 2  …
      -/
      refine this.trans ?_
      /-
        case neg.true
        h2 : _root_.Prime 2
        n : Nat
        ih : Ne n 0 → LT.lt (emultiplicity 2 n.factorial) ↑n
        hn : Not (Eq n 0)
        this : LT.lt (emultiplicity 2 (HMul.hMul 2 n).factorial) ↑(HMul.hMul 2 n)
        h : Ne (Nat.bit Bool.true n) 0
        ⊢ LT.lt (↑(HMul.hMul 2 n)) (HAdd.hAdd (↑(HMul.hMul 2 n)) 1)
      -/
      exact mod_cast lt_succ_self _
      /-
        🎉 no goals
      -/


