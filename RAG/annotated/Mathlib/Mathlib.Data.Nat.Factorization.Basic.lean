theorem factorization_eq_zero_of_lt {n p : ℕ} (h : n < p) : n.factorization p = 0 :=
  Finsupp.not_mem_support_iff.mp (mt le_of_mem_primeFactors (not_le_of_lt h))


@[simp]
theorem factorization_one_right (n : ℕ) : n.factorization 1 = 0 :=
  factorization_eq_zero_of_non_prime _ not_prime_one


theorem dvd_of_factorization_pos {n p : ℕ} (hn : n.factorization p ≠ 0) : p ∣ n :=
  dvd_of_mem_primeFactorsList <| mem_primeFactors_iff_mem_primeFactorsList.1 <| mem_support_iff.2 hn


theorem factorization_eq_zero_iff_remainder {p r : ℕ} (i : ℕ) (pp : p.Prime) (hr0 : r ≠ 0) :
    ¬p ∣ r ↔ (p * i + r).factorization p = 0 := by
  /-
    p r i : Nat
    pp : Nat.Prime p
    hr0 : Ne r 0
    ⊢ Iff (Not (Dvd.dvd p r)) (Eq ((HAdd.hAdd (HMul.hMul p i) r).factorization p) 0)
  -/
  refine ⟨factorization_eq_zero_of_remainder i, fun h => ?_⟩
  /-
    p r i : Nat
    pp : Nat.Prime p
    hr0 : Ne r 0
    h : Eq ((HAdd.hAdd (HMul.hMul p i) r).factorization p) 0
    ⊢ Not (Dvd.dvd p r)
  -/
  rw [factorization_eq_zero_iff] at h
  /-
    p r i : Nat
    pp : Nat.Prime p
    hr0 : Ne r 0
    h : Or (Not (Nat.Prime p)) (Or (Not (Dvd.dvd p (HAdd.hAdd (HMul.hMul p i) r))) …
    ⊢ Not (Dvd.dvd p r)
  -/
  contrapose! h
  /-
    p r i : Nat
    pp : Nat.Prime p
    hr0 : Ne r 0
    h : Dvd.dvd p r
    ⊢ And (Nat.Prime p) (And (Dvd.dvd p (HAdd.hAdd (HMul.hMul p i) r)) (Ne (HAdd.h …
  -/
  refine ⟨pp, ?_, ?_⟩
    /-
      case refine_1
      p r i : Nat
      pp : Nat.Prime p
      hr0 : Ne r 0
      h : Dvd.dvd p r
      ⊢ Dvd.dvd p (HAdd.hAdd (HMul.hMul p i) r)
    -/
  · rwa [← Nat.dvd_add_iff_right (dvd_mul_right p i)]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p r i : Nat
      pp : Nat.Prime p
      hr0 : Ne r 0
      h : Dvd.dvd p r
      ⊢ Ne (HAdd.hAdd (HMul.hMul p i) r) 0
    -/
  · contrapose! hr0
    /-
      case refine_2
      p r i : Nat
      pp : Nat.Prime p
      h : Dvd.dvd p r
      hr0 : Eq (HAdd.hAdd (HMul.hMul p i) r) 0
      ⊢ Eq r 0
    -/
    exact (add_eq_zero.1 hr0).2
    /-
      🎉 no goals
    -/


/-- The only numbers with empty prime factorization are `0` and `1` -/
theorem factorization_eq_zero_iff' (n : ℕ) : n.factorization = 0 ↔ n = 0 ∨ n = 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.factorization 0) (Or (Eq n 0) (Eq n 1))
  -/
  rw [factorization_eq_primeFactorsList_multiset n]
  /-
    n : Nat
    ⊢ Iff (Eq (Multiset.toFinsupp ↑n.primeFactorsList) 0) (Or (Eq n 0) (Eq n 1))
  -/
  simp [factorization, AddEquiv.map_eq_zero_iff, Multiset.coe_eq_zero]
  /-
    🎉 no goals
  -/


/-- A product over `n.factorization` can be written as a product over `n.primeFactors`; -/
lemma prod_factorization_eq_prod_primeFactors {β : Type*} [CommMonoid β] (f : ℕ → ℕ → β) :
    n.factorization.prod f = ∏ p ∈ n.primeFactors, f p (n.factorization p) := rfl


/-- A product over `n.primeFactors` can be written as a product over `n.factorization`; -/
lemma prod_primeFactors_prod_factorization {β : Type*} [CommMonoid β] (f : ℕ → β) :
    ∏ p ∈ n.primeFactors, f p = n.factorization.prod (fun p _ ↦ f p) := rfl


/-- The multiplicity of prime `p` in `p` is `1` -/
@[simp]
                                                                                      /-
                                                                                        p : Nat
                                                                                        hp : Nat.Prime p
                                                                                        ⊢ Eq (p.factorization p) 1
                                                                                      -/
theorem Prime.factorization_self {p : ℕ} (hp : Prime p) : p.factorization p = 1 := by simp [hp]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- If the factorization of `n` contains just one number `p` then `n` is a power of `p` -/
theorem eq_pow_of_factorization_eq_single {n p k : ℕ} (hn : n ≠ 0)
    (h : n.factorization = Finsupp.single p k) : n = p ^ k := by
  -- Porting note: explicitly added `Finsupp.prod_single_index`
  /-
    n p k : Nat
    hn : Ne n 0
    h : Eq n.factorization (Finsupp.single p k)
    ⊢ Eq n (HPow.hPow p k)
  -/
  rw [← Nat.factorization_prod_pow_eq_self hn, h, Finsupp.prod_single_index]
  /-
    n p k : Nat
    hn : Ne n 0
    h : Eq n.factorization (Finsupp.single p k)
    ⊢ Eq (HPow.hPow p 0) 1
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The only prime factor of prime `p` is `p` itself. -/
theorem Prime.eq_of_factorization_pos {p q : ℕ} (hp : Prime p) (h : p.factorization q ≠ 0) :
                /-
                  p q : Nat
                  hp : Nat.Prime p
                  h : Ne (p.factorization q) 0
                  ⊢ Eq p q
                -/
    p = q := by simpa [hp.factorization, single_apply] using h
                /-
                  🎉 no goals
                -/


theorem eq_factorization_iff {n : ℕ} {f : ℕ →₀ ℕ} (hn : n ≠ 0) (hf : ∀ p ∈ f.support, Prime p) :
    f = n.factorization ↔ f.prod (· ^ ·) = n :=
               /-
                 n : Nat
                 f : Finsupp Nat Nat
                 hn : Ne n 0
                 hf : ∀ (p : Nat), Membership.mem f.support p → Nat.Prime p
                 h : Eq f n.factorization
                 ⊢ Eq (f.prod fun x1 x2 => HPow.hPow x1 x2) n
               -/
  ⟨fun h => by rw [h, factorization_prod_pow_eq_self hn], fun h => by
               /-
                 🎉 no goals
               -/
    /-
      n : Nat
      f : Finsupp Nat Nat
      hn : Ne n 0
      hf : ∀ (p : Nat), Membership.mem f.support p → Nat.Prime p
      h : Eq (f.prod fun x1 x2 => HPow.hPow x1 x2) n
      ⊢ Eq f n.factorization
    -/
    rw [← h, prod_pow_factorization_eq_self hf]⟩
    /-
      🎉 no goals
    -/


theorem factorizationEquiv_inv_apply {f : ℕ →₀ ℕ} (hf : ∀ p ∈ f.support, Prime p) :
    (factorizationEquiv.symm ⟨f, hf⟩).1 = f.prod (· ^ ·) :=
  rfl


@[simp]
theorem ordProj_of_not_prime (n p : ℕ) (hp : ¬p.Prime) : ordProj[p] n = 1 := by
  /-
    n p : Nat
    hp : Not (Nat.Prime p)
    ⊢ Eq (HPow.hPow p (n.factorization p)) 1
  -/
  simp [factorization_eq_zero_of_non_prime n hp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")] alias ord_proj_of_not_prime := ordProj_of_not_prime


@[simp]
theorem ordCompl_of_not_prime (n p : ℕ) (hp : ¬p.Prime) : ordCompl[p] n = n := by
  /-
    n p : Nat
    hp : Not (Nat.Prime p)
    ⊢ Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) n
  -/
  simp [factorization_eq_zero_of_non_prime n hp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")] alias ord_compl_of_not_prime := ordCompl_of_not_prime


theorem ordCompl_dvd (n p : ℕ) : ordCompl[p] n ∣ n :=
  div_dvd_of_dvd (ordProj_dvd n p)


@[deprecated (since := "2024-10-24")] alias ord_compl_dvd := ordCompl_dvd


theorem ordProj_pos (n p : ℕ) : 0 < ordProj[p] n := by
  /-
    n p : Nat
    ⊢ LT.lt 0 (HPow.hPow p (n.factorization p))
  -/
  if pp : p.Prime then simp [pow_pos pp.pos] else simp [pp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")] alias ord_proj_pos := ordProj_pos


theorem ordProj_le {n : ℕ} (p : ℕ) (hn : n ≠ 0) : ordProj[p] n ≤ n :=
  le_of_dvd hn.bot_lt (Nat.ordProj_dvd n p)


@[deprecated (since := "2024-10-24")] alias ord_proj_le := ordProj_le


theorem ordCompl_pos {n : ℕ} (p : ℕ) (hn : n ≠ 0) : 0 < ordCompl[p] n := by
  if pp : p.Prime then
    exact Nat.div_pos (ordProj_le p hn) (ordProj_pos n p)
  else
    simpa [Nat.factorization_eq_zero_of_non_prime n pp] using hn.bot_lt


@[deprecated (since := "2024-10-24")] alias ord_compl_pos := ordCompl_pos


theorem ordCompl_le (n p : ℕ) : ordCompl[p] n ≤ n :=
  Nat.div_le_self _ _


@[deprecated (since := "2024-10-24")] alias ord_compl_le := ordCompl_le


theorem ordProj_mul_ordCompl_eq_self (n p : ℕ) : ordProj[p] n * ordCompl[p] n = n :=
  Nat.mul_div_cancel' (ordProj_dvd n p)


@[deprecated (since := "2024-10-24")]
alias ord_proj_mul_ord_compl_eq_self := ordProj_mul_ordCompl_eq_self


theorem ordProj_mul {a b : ℕ} (p : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) :
    ordProj[p] (a * b) = ordProj[p] a * ordProj[p] b := by
  /-
    a b p : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HPow.hPow p ((HMul.hMul a b).factorization p)) (HMul.hMul (HPow.hPow p ( …
  -/
  simp [factorization_mul ha hb, pow_add]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")] alias ord_proj_mul := ordProj_mul


theorem ordCompl_mul (a b p : ℕ) : ordCompl[p] (a * b) = ordCompl[p] a * ordCompl[p] b := by
  if ha : a = 0 then simp [ha] else
  if hb : b = 0 then simp [hb] else
  simp only [ordProj_mul p ha hb]
  rw [div_mul_div_comm (ordProj_dvd a p) (ordProj_dvd b p)]


@[deprecated (since := "2024-10-24")] alias ord_compl_mul := ordCompl_mul


/-- A crude upper bound on `n.factorization p` -/
theorem factorization_lt {n : ℕ} (p : ℕ) (hn : n ≠ 0) : n.factorization p < n := by
  /-
    n p : Nat
    hn : Ne n 0
    ⊢ LT.lt (n.factorization p) n
  -/
  by_cases pp : p.Prime
  · exact (Nat.pow_lt_pow_iff_right pp.one_lt).1 <| (ordProj_le p hn).trans_lt <|
      Nat.lt_pow_self pp.one_lt
    /-
      case neg
      n p : Nat
      hn : Ne n 0
      pp : Not (Nat.Prime p)
      ⊢ LT.lt (n.factorization p) n
    -/
  · simpa only [factorization_eq_zero_of_non_prime n pp] using hn.bot_lt
    /-
      🎉 no goals
    -/


/-- An upper bound on `n.factorization p` -/
theorem factorization_le_of_le_pow {n p b : ℕ} (hb : n ≤ p ^ b) : n.factorization p ≤ b := by
  if hn : n = 0 then simp [hn] else
  if pp : p.Prime then
    exact (Nat.pow_le_pow_iff_right pp.one_lt).1 ((ordProj_le p hn).trans hb)
  else
    simp [factorization_eq_zero_of_non_prime n pp]


theorem factorization_prime_le_iff_dvd {d n : ℕ} (hd : d ≠ 0) (hn : n ≠ 0) :
    (∀ p : ℕ, p.Prime → d.factorization p ≤ n.factorization p) ↔ d ∣ n := by
  /-
    d n : Nat
    hd : Ne d 0
    hn : Ne n 0
    ⊢ Iff (∀ (p : Nat), Nat.Prime p → LE.le (d.factorization p) (n.factorization p …
  -/
  rw [← factorization_le_iff_dvd hd hn]
  /-
    d n : Nat
    hd : Ne d 0
    hn : Ne n 0
    ⊢ Iff (∀ (p : Nat), Nat.Prime p → LE.le (d.factorization p) (n.factorization p …
  -/
  refine ⟨fun h p => (em p.Prime).elim (h p) fun hp => ?_, fun h p _ => h p⟩
  /-
    d n : Nat
    hd : Ne d 0
    hn : Ne n 0
    h : ∀ (p : Nat), Nat.Prime p → LE.le (d.factorization p) (n.factorization p)
    p : Nat
    hp : Not (Nat.Prime p)
    ⊢ LE.le (d.factorization p) (n.factorization p)
  -/
  simp_rw [factorization_eq_zero_of_non_prime _ hp]
  /-
    d n : Nat
    hd : Ne d 0
    hn : Ne n 0
    h : ∀ (p : Nat), Nat.Prime p → LE.le (d.factorization p) (n.factorization p)
    p : Nat
    hp : Not (Nat.Prime p)
    ⊢ LE.le 0 0
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem factorization_le_factorization_mul_left {a b : ℕ} (hb : b ≠ 0) :
    a.factorization ≤ (a * b).factorization := by
  /-
    a b : Nat
    hb : Ne b 0
    ⊢ LE.le a.factorization (HMul.hMul a b).factorization
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      b : Nat
      hb : Ne b 0
      ⊢ LE.le (Nat.factorization 0) (HMul.hMul 0 b).factorization
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    hb : Ne b 0
    ha : Ne a 0
    ⊢ LE.le a.factorization (HMul.hMul a b).factorization
  -/
  rw [factorization_le_iff_dvd ha <| mul_ne_zero ha hb]
  /-
    case inr
    a b : Nat
    hb : Ne b 0
    ha : Ne a 0
    ⊢ Dvd.dvd a (HMul.hMul a b)
  -/
  exact Dvd.intro b rfl
  /-
    🎉 no goals
  -/


theorem factorization_le_factorization_mul_right {a b : ℕ} (ha : a ≠ 0) :
    b.factorization ≤ (a * b).factorization := by
  /-
    a b : Nat
    ha : Ne a 0
    ⊢ LE.le b.factorization (HMul.hMul a b).factorization
  -/
  rw [mul_comm]
  /-
    a b : Nat
    ha : Ne a 0
    ⊢ LE.le b.factorization (HMul.hMul b a).factorization
  -/
  apply factorization_le_factorization_mul_left ha
  /-
    🎉 no goals
  -/


theorem Prime.pow_dvd_iff_le_factorization {p k n : ℕ} (pp : Prime p) (hn : n ≠ 0) :
    p ^ k ∣ n ↔ k ≤ n.factorization p := by
  /-
    p k n : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    ⊢ Iff (Dvd.dvd (HPow.hPow p k) n) (LE.le k (n.factorization p))
  -/
  rw [← factorization_le_iff_dvd (pow_pos pp.pos k).ne' hn, pp.factorization_pow, single_le_iff]
  /-
    🎉 no goals
  -/


theorem Prime.pow_dvd_iff_dvd_ordProj {p k n : ℕ} (pp : Prime p) (hn : n ≠ 0) :
    p ^ k ∣ n ↔ p ^ k ∣ ordProj[p] n := by
  /-
    p k n : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    ⊢ Iff (Dvd.dvd (HPow.hPow p k) n) (Dvd.dvd (HPow.hPow p k) (HPow.hPow p (n.fac …
  -/
  rw [pow_dvd_pow_iff_le_right pp.one_lt, pp.pow_dvd_iff_le_factorization hn]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias Prime.pow_dvd_iff_dvd_ord_proj := Prime.pow_dvd_iff_dvd_ordProj


theorem Prime.dvd_iff_one_le_factorization {p n : ℕ} (pp : Prime p) (hn : n ≠ 0) :
    p ∣ n ↔ 1 ≤ n.factorization p :=
                /-
                  p n : Nat
                  pp : Nat.Prime p
                  hn : Ne n 0
                  ⊢ Iff (Dvd.dvd p n) (Dvd.dvd (HPow.hPow p 1) n)
                -/
  Iff.trans (by simp) (pp.pow_dvd_iff_le_factorization hn)
                /-
                  🎉 no goals
                -/


theorem exists_factorization_lt_of_lt {a b : ℕ} (ha : a ≠ 0) (hab : a < b) :
    ∃ p : ℕ, a.factorization p < b.factorization p := by
  /-
    a b : Nat
    ha : Ne a 0
    hab : LT.lt a b
    ⊢ Exists fun p => LT.lt (a.factorization p) (b.factorization p)
  -/
  have hb : b ≠ 0 := (ha.bot_lt.trans hab).ne'
  /-
    a b : Nat
    ha : Ne a 0
    hab : LT.lt a b
    hb : Ne b 0
    ⊢ Exists fun p => LT.lt (a.factorization p) (b.factorization p)
  -/
  contrapose! hab
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    hab : ∀ (p : Nat), LE.le (b.factorization p) (a.factorization p)
    ⊢ LE.le b a
  -/
  rw [← Finsupp.le_def, factorization_le_iff_dvd hb ha] at hab
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    hab : Dvd.dvd b a
    ⊢ LE.le b a
  -/
  exact le_of_dvd ha.bot_lt hab
  /-
    🎉 no goals
  -/


@[simp]
theorem factorization_div {d n : ℕ} (h : d ∣ n) :
    (n / d).factorization = n.factorization - d.factorization := by
  /-
    d n : Nat
    h : Dvd.dvd d n
    ⊢ Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization)
  -/
  rcases eq_or_ne d 0 with (rfl | hd); · simp [zero_dvd_iff.mp h]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    d n : Nat
    h : Dvd.dvd d n
    hd : Ne d 0
    ⊢ Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization)
  -/
  rcases eq_or_ne n 0 with (rfl | hn); · simp [tsub_eq_zero_of_le]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr.inr
    d n : Nat
    h : Dvd.dvd d n
    hd : Ne d 0
    hn : Ne n 0
    ⊢ Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization)
  -/
  apply add_left_injective d.factorization
  /-
    case inr.inr.a
    d n : Nat
    h : Dvd.dvd d n
    hd : Ne d 0
    hn : Ne n 0
    ⊢ Eq ((fun x => HAdd.hAdd x d.factorization) (HDiv.hDiv n d).factorization) (( …
  -/
  simp only
  rw [tsub_add_cancel_of_le <| (Nat.factorization_le_iff_dvd hd hn).mpr h, ←
    Nat.factorization_mul (Nat.div_pos (Nat.le_of_dvd hn.bot_lt h) hd.bot_lt).ne' hd,
    Nat.div_mul_cancel h]


theorem dvd_ordProj_of_dvd {n p : ℕ} (hn : n ≠ 0) (pp : p.Prime) (h : p ∣ n) : p ∣ ordProj[p] n :=
  dvd_pow_self p (Prime.factorization_pos_of_dvd pp hn h).ne'


@[deprecated (since := "2024-10-24")] alias dvd_ord_proj_of_dvd := dvd_ordProj_of_dvd


theorem not_dvd_ordCompl {n p : ℕ} (hp : Prime p) (hn : n ≠ 0) : ¬p ∣ ordCompl[p] n := by
  /-
    n p : Nat
    hp : Nat.Prime p
    hn : Ne n 0
    ⊢ Not (Dvd.dvd p (HDiv.hDiv n (HPow.hPow p (n.factorization p))))
  -/
  rw [Nat.Prime.dvd_iff_one_le_factorization hp (ordCompl_pos p hn).ne']
  /-
    n p : Nat
    hp : Nat.Prime p
    hn : Ne n 0
    ⊢ Not (LE.le 1 ((HDiv.hDiv n (HPow.hPow p (n.factorization p))).factorization  …
  -/
  rw [Nat.factorization_div (Nat.ordProj_dvd n p)]
  /-
    n p : Nat
    hp : Nat.Prime p
    hn : Ne n 0
    ⊢ Not (LE.le 1 ((HSub.hSub n.factorization (HPow.hPow p (n.factorization p)).f …
  -/
  simp [hp.factorization]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")] alias not_dvd_ord_compl := not_dvd_ordCompl


theorem coprime_ordCompl {n p : ℕ} (hp : Prime p) (hn : n ≠ 0) : Coprime p (ordCompl[p] n) :=
  (or_iff_left (not_dvd_ordCompl hp hn)).mp <| coprime_or_dvd_of_prime hp _


@[deprecated (since := "2024-10-24")] alias coprime_ord_compl := coprime_ordCompl


theorem factorization_ordCompl (n p : ℕ) :
    (ordCompl[p] n).factorization = n.factorization.erase p := by
  if hn : n = 0 then simp [hn] else
  if pp : p.Prime then ?_ else
    -- Porting note: needed to solve side goal explicitly
    rw [Finsupp.erase_of_not_mem_support] <;> simp [pp]
  ext q
  rcases eq_or_ne q p with (rfl | hqp)
  · simp only [Finsupp.erase_same, factorization_eq_zero_iff, not_dvd_ordCompl pp hn]
    simp
  · rw [Finsupp.erase_ne hqp, factorization_div (ordProj_dvd n p)]
    simp [pp.factorization, hqp.symm]


@[deprecated (since := "2024-10-24")] alias factorization_ord_compl := factorization_ordCompl

-- `ordCompl[p] n` is the largest divisor of `n` not divisible by `p`.

theorem dvd_ordCompl_of_dvd_not_dvd {p d n : ℕ} (hdn : d ∣ n) (hpd : ¬p ∣ d) :
    d ∣ ordCompl[p] n := by
  if hn0 : n = 0 then simp [hn0] else
  if hd0 : d = 0 then simp [hd0] at hpd else
  rw [← factorization_le_iff_dvd hd0 (ordCompl_pos p hn0).ne', factorization_ordCompl]
  intro q
  if hqp : q = p then
    simp [factorization_eq_zero_iff, hqp, hpd]
  else
    simp [hqp, (factorization_le_iff_dvd hd0 hn0).2 hdn q]


@[deprecated (since := "2024-10-24")]
alias dvd_ord_compl_of_dvd_not_dvd := dvd_ordCompl_of_dvd_not_dvd


/-- If `n` is a nonzero natural number and `p ≠ 1`, then there are natural numbers `e`
and `n'` such that `n'` is not divisible by `p` and `n = p^e * n'`. -/
theorem exists_eq_pow_mul_and_not_dvd {n : ℕ} (hn : n ≠ 0) (p : ℕ) (hp : p ≠ 1) :
    ∃ e n' : ℕ, ¬p ∣ n' ∧ n = p ^ e * n' :=
  let ⟨a', h₁, h₂⟩ :=
    (Nat.finiteMultiplicity_iff.mpr ⟨hp, Nat.pos_of_ne_zero hn⟩).exists_eq_pow_mul_and_not_dvd
  ⟨_, a', h₂, h₁⟩


/-- Any nonzero natural number is the product of an odd part `m` and a power of
two `2 ^ k`. -/
theorem exists_eq_two_pow_mul_odd {n : ℕ} (hn : n ≠ 0) :
    ∃ k m : ℕ, Odd m ∧ n = 2 ^ k * m :=
  let ⟨k, m, hm, hn⟩ := exists_eq_pow_mul_and_not_dvd hn 2 (succ_ne_self 1)
  ⟨k, m, not_even_iff_odd.1 (mt Even.two_dvd hm), hn⟩


theorem dvd_iff_div_factorization_eq_tsub {d n : ℕ} (hd : d ≠ 0) (hdn : d ≤ n) :
    d ∣ n ↔ (n / d).factorization = n.factorization - d.factorization := by
  /-
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    ⊢ Iff (Dvd.dvd d n) (Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorizati …
  -/
  refine ⟨factorization_div, ?_⟩
  /-
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    ⊢ Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization) …
  -/
  rcases eq_or_lt_of_le hdn with (rfl | hd_lt_n); · simp
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    case inr
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    hd_lt_n : LT.lt d n
    ⊢ Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization) …
  -/
  have h1 : n / d ≠ 0 := by simp [*]
  /-
    case inr
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    hd_lt_n : LT.lt d n
    h1 : Ne (HDiv.hDiv n d) 0
    ⊢ Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization) …
  -/
  intro h
  /-
    case inr
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    hd_lt_n : LT.lt d n
    h1 : Ne (HDiv.hDiv n d) 0
    h : Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization)
    ⊢ Dvd.dvd d n
  -/
  rw [dvd_iff_le_div_mul n d]
  /-
    case inr
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    hd_lt_n : LT.lt d n
    h1 : Ne (HDiv.hDiv n d) 0
    h : Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization)
    ⊢ LE.le n (HMul.hMul (HDiv.hDiv n d) d)
  -/
  by_contra h2
  /-
    case inr
    d n : Nat
    hd : Ne d 0
    hdn : LE.le d n
    hd_lt_n : LT.lt d n
    h1 : Ne (HDiv.hDiv n d) 0
    h : Eq (HDiv.hDiv n d).factorization (HSub.hSub n.factorization d.factorization)
    h2 : Not (LE.le n (HMul.hMul (HDiv.hDiv n d) d))
    ⊢ False
  -/
  cases' exists_factorization_lt_of_lt (mul_ne_zero h1 hd) (not_le.mp h2) with p hp
  rwa [factorization_mul h1 hd, add_apply, ← lt_tsub_iff_right, h, tsub_apply,
   lt_self_iff_false] at hp


theorem ordProj_dvd_ordProj_of_dvd {a b : ℕ} (hb0 : b ≠ 0) (hab : a ∣ b) (p : ℕ) :
    ordProj[p] a ∣ ordProj[p] b := by
  /-
    a b : Nat
    hb0 : Ne b 0
    hab : Dvd.dvd a b
    p : Nat
    ⊢ Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.factorization p))
  -/
  rcases em' p.Prime with (pp | pp); · simp [pp]
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case inr
    a b : Nat
    hb0 : Ne b 0
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    ⊢ Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.factorization p))
  -/
  rcases eq_or_ne a 0 with (rfl | ha0); · simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr.inr
    a b : Nat
    hb0 : Ne b 0
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    ha0 : Ne a 0
    ⊢ Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.factorization p))
  -/
  rw [pow_dvd_pow_iff_le_right pp.one_lt]
  /-
    case inr.inr
    a b : Nat
    hb0 : Ne b 0
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    ha0 : Ne a 0
    ⊢ LE.le (a.factorization p) (b.factorization p)
  -/
  exact (factorization_le_iff_dvd ha0 hb0).2 hab p
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias ord_proj_dvd_ord_proj_of_dvd := ordProj_dvd_ordProj_of_dvd


theorem ordProj_dvd_ordProj_iff_dvd {a b : ℕ} (ha0 : a ≠ 0) (hb0 : b ≠ 0) :
    (∀ p : ℕ, ordProj[p] a ∣ ordProj[p] b) ↔ a ∣ b := by
  /-
    a b : Nat
    ha0 : Ne a 0
    hb0 : Ne b 0
    ⊢ Iff (∀ (p : Nat), Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b. …
  -/
  refine ⟨fun h => ?_, fun hab p => ordProj_dvd_ordProj_of_dvd hb0 hab p⟩
  /-
    a b : Nat
    ha0 : Ne a 0
    hb0 : Ne b 0
    h : ∀ (p : Nat), Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.fac …
    ⊢ Dvd.dvd a b
  -/
  rw [← factorization_le_iff_dvd ha0 hb0]
  /-
    a b : Nat
    ha0 : Ne a 0
    hb0 : Ne b 0
    h : ∀ (p : Nat), Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.fac …
    ⊢ LE.le a.factorization b.factorization
  -/
  intro q
  /-
    a b : Nat
    ha0 : Ne a 0
    hb0 : Ne b 0
    h : ∀ (p : Nat), Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.fac …
    q : Nat
    ⊢ LE.le (a.factorization q) (b.factorization q)
  -/
  rcases le_or_lt q 1 with (hq_le | hq1)
    /-
      case inl
      a b : Nat
      ha0 : Ne a 0
      hb0 : Ne b 0
      h : ∀ (p : Nat), Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.fac …
      q : Nat
      hq_le : LE.le q 1
      ⊢ LE.le (a.factorization q) (b.factorization q)
    -/
                         /-
                           🎉 no goals
                         -/
  · interval_cases q <;> simp
                         /-
                           🎉 no goals
                         -/
  /-
    case inr
    a b : Nat
    ha0 : Ne a 0
    hb0 : Ne b 0
    h : ∀ (p : Nat), Dvd.dvd (HPow.hPow p (a.factorization p)) (HPow.hPow p (b.fac …
    q : Nat
    hq1 : LT.lt 1 q
    ⊢ LE.le (a.factorization q) (b.factorization q)
  -/
  exact (pow_dvd_pow_iff_le_right hq1).1 (h q)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias ord_proj_dvd_ord_proj_iff_dvd := ordProj_dvd_ordProj_iff_dvd


theorem ordCompl_dvd_ordCompl_of_dvd {a b : ℕ} (hab : a ∣ b) (p : ℕ) :
    ordCompl[p] a ∣ ordCompl[p] b := by
  /-
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
  -/
  rcases em' p.Prime with (pp | pp)
    /-
      case inl
      a b : Nat
      hab : Dvd.dvd a b
      p : Nat
      pp : Not (Nat.Prime p)
      ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
    -/
  · simp [pp, hab]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
  -/
  rcases eq_or_ne b 0 with (rfl | hb0)
    /-
      case inr.inl
      a p : Nat
      pp : Nat.Prime p
      hab : Dvd.dvd a 0
      ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv 0 (HPow.h …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
  -/
  rcases eq_or_ne a 0 with (rfl | ha0)
    /-
      case inr.inr.inl
      b p : Nat
      pp : Nat.Prime p
      hb0 : Ne b 0
      hab : Dvd.dvd 0 b
      ⊢ Dvd.dvd (HDiv.hDiv 0 (HPow.hPow p ((Nat.factorization 0) p))) (HDiv.hDiv b ( …
    -/
  · cases hb0 (zero_dvd_iff.1 hab)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
  -/
  have ha := (Nat.div_pos (ordProj_le p ha0) (ordProj_pos a p)).ne'
  /-
    case inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ha : Ne (HDiv.hDiv a (HPow.hPow p (a.factorization p))) 0
    ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
  -/
  have hb := (Nat.div_pos (ordProj_le p hb0) (ordProj_pos b p)).ne'
  /-
    case inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ha : Ne (HDiv.hDiv a (HPow.hPow p (a.factorization p))) 0
    hb : Ne (HDiv.hDiv b (HPow.hPow p (b.factorization p))) 0
    ⊢ Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv.hDiv b (HPow.h …
  -/
  rw [← factorization_le_iff_dvd ha hb, factorization_ordCompl a p, factorization_ordCompl b p]
  /-
    case inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ha : Ne (HDiv.hDiv a (HPow.hPow p (a.factorization p))) 0
    hb : Ne (HDiv.hDiv b (HPow.hPow p (b.factorization p))) 0
    ⊢ LE.le (Finsupp.erase p a.factorization) (Finsupp.erase p b.factorization)
  -/
  intro q
  /-
    case inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ha : Ne (HDiv.hDiv a (HPow.hPow p (a.factorization p))) 0
    hb : Ne (HDiv.hDiv b (HPow.hPow p (b.factorization p))) 0
    q : Nat
    ⊢ LE.le ((Finsupp.erase p a.factorization) q) ((Finsupp.erase p b.factorizatio …
  -/
  rcases eq_or_ne q p with (rfl | hqp)
    /-
      case inr.inr.inr.inl
      a b : Nat
      hab : Dvd.dvd a b
      hb0 : Ne b 0
      ha0 : Ne a 0
      q : Nat
      pp : Nat.Prime q
      ha : Ne (HDiv.hDiv a (HPow.hPow q (a.factorization q))) 0
      hb : Ne (HDiv.hDiv b (HPow.hPow q (b.factorization q))) 0
      ⊢ LE.le ((Finsupp.erase q a.factorization) q) ((Finsupp.erase q b.factorizatio …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ha : Ne (HDiv.hDiv a (HPow.hPow p (a.factorization p))) 0
    hb : Ne (HDiv.hDiv b (HPow.hPow p (b.factorization p))) 0
    q : Nat
    hqp : Ne q p
    ⊢ LE.le ((Finsupp.erase p a.factorization) q) ((Finsupp.erase p b.factorizatio …
  -/
  simp_rw [erase_ne hqp]
  /-
    case inr.inr.inr.inr
    a b : Nat
    hab : Dvd.dvd a b
    p : Nat
    pp : Nat.Prime p
    hb0 : Ne b 0
    ha0 : Ne a 0
    ha : Ne (HDiv.hDiv a (HPow.hPow p (a.factorization p))) 0
    hb : Ne (HDiv.hDiv b (HPow.hPow p (b.factorization p))) 0
    q : Nat
    hqp : Ne q p
    ⊢ LE.le (a.factorization q) (b.factorization q)
  -/
  exact (factorization_le_iff_dvd ha0 hb0).2 hab q
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias ord_compl_dvd_ord_compl_of_dvd := ordCompl_dvd_ordCompl_of_dvd


theorem ordCompl_dvd_ordCompl_iff_dvd (a b : ℕ) :
    (∀ p : ℕ, ordCompl[p] a ∣ ordCompl[p] b) ↔ a ∣ b := by
  /-
    a b : Nat
    ⊢ Iff (∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (H …
  -/
  refine ⟨fun h => ?_, fun hab p => ordCompl_dvd_ordCompl_of_dvd hab p⟩
  /-
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    ⊢ Dvd.dvd a b
  -/
  rcases eq_or_ne b 0 with (rfl | hb0)
    /-
      case inl
      a : Nat
      h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
      ⊢ Dvd.dvd a 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    ⊢ Dvd.dvd a b
  -/
  if pa : a.Prime then ?_ else simpa [pa] using h a
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    pa : Nat.Prime a
    ⊢ Dvd.dvd a b
  -/
  if pb : b.Prime then ?_ else simpa [pb] using h b
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    pa : Nat.Prime a
    pb : Nat.Prime b
    ⊢ Dvd.dvd a b
  -/
  rw [prime_dvd_prime_iff_eq pa pb]
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    pa : Nat.Prime a
    pb : Nat.Prime b
    ⊢ Eq a b
  -/
  by_contra hab
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    pa : Nat.Prime a
    pb : Nat.Prime b
    hab : Not (Eq a b)
    ⊢ False
  -/
  apply pa.ne_one
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    pa : Nat.Prime a
    pb : Nat.Prime b
    hab : Not (Eq a b)
    ⊢ Eq a 1
  -/
  rw [← Nat.dvd_one, ← Nat.mul_dvd_mul_iff_left hb0.bot_lt, mul_one]
  /-
    case inr
    a b : Nat
    h : ∀ (p : Nat), Dvd.dvd (HDiv.hDiv a (HPow.hPow p (a.factorization p))) (HDiv …
    hb0 : Ne b 0
    pa : Nat.Prime a
    pb : Nat.Prime b
    hab : Not (Eq a b)
    ⊢ Dvd.dvd (HMul.hMul b a) b
  -/
  simpa [Prime.factorization_self pb, Prime.factorization pa, hab] using h b
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias ord_compl_dvd_ord_compl_iff_dvd := ordCompl_dvd_ordCompl_iff_dvd


theorem dvd_iff_prime_pow_dvd_dvd (n d : ℕ) :
    d ∣ n ↔ ∀ p k : ℕ, Prime p → p ^ k ∣ d → p ^ k ∣ n := by
  /-
    n d : Nat
    ⊢ Iff (Dvd.dvd d n) (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d →  …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      d : Nat
      ⊢ Iff (Dvd.dvd d 0) (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d →  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n d : Nat
    hn : Ne n 0
    ⊢ Iff (Dvd.dvd d n) (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d →  …
  -/
  rcases eq_or_ne d 0 with (rfl | hd)
    /-
      case inr.inl
      n : Nat
      hn : Ne n 0
      ⊢ Iff (Dvd.dvd 0 n) (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) 0 →  …
    -/
  · simp only [zero_dvd_iff, hn, false_iff, not_forall]
    /-
      case inr.inl
      n : Nat
      hn : Ne n 0
      ⊢ Exists fun x => Exists fun x_1 => Exists fun h => Exists fun x_2 => Not (Dvd …
    -/
    exact ⟨2, n, prime_two, dvd_zero _, mt (le_of_dvd hn.bot_lt) (n.lt_two_pow_self).not_le⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    n d : Nat
    hn : Ne n 0
    hd : Ne d 0
    ⊢ Iff (Dvd.dvd d n) (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d →  …
  -/
  refine ⟨fun h p k _ hpkd => dvd_trans hpkd h, ?_⟩
  /-
    case inr.inr
    n d : Nat
    hn : Ne n 0
    hd : Ne d 0
    ⊢ (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d → Dvd.dvd (HPow.hPow …
  -/
  rw [← factorization_prime_le_iff_dvd hd hn]
  /-
    case inr.inr
    n d : Nat
    hn : Ne n 0
    hd : Ne d 0
    ⊢ (∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d → Dvd.dvd (HPow.hPow …
  -/
  intro h p pp
  /-
    case inr.inr
    n d : Nat
    hn : Ne n 0
    hd : Ne d 0
    h : ∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d → Dvd.dvd (HPow.hPo …
    p : Nat
    pp : Nat.Prime p
    ⊢ LE.le (d.factorization p) (n.factorization p)
  -/
  simp_rw [← pp.pow_dvd_iff_le_factorization hn]
  /-
    case inr.inr
    n d : Nat
    hn : Ne n 0
    hd : Ne d 0
    h : ∀ (p k : Nat), Nat.Prime p → Dvd.dvd (HPow.hPow p k) d → Dvd.dvd (HPow.hPo …
    p : Nat
    pp : Nat.Prime p
    ⊢ Dvd.dvd (HPow.hPow p (d.factorization p)) n
  -/
  exact h p _ pp (ordProj_dvd _ _)
  /-
    🎉 no goals
  -/


theorem prod_primeFactors_dvd (n : ℕ) : ∏ p ∈ n.primeFactors, p ∣ n := by
  /-
    n : Nat
    ⊢ Dvd.dvd (n.primeFactors.prod fun p => p) n
  -/
  by_cases hn : n = 0
    /-
      case pos
      n : Nat
      hn : Eq n 0
      ⊢ Dvd.dvd (n.primeFactors.prod fun p => p) n
    -/
  · subst hn
    /-
      case pos
      ⊢ Dvd.dvd ((Nat.primeFactors 0).prod fun p => p) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      hn : Not (Eq n 0)
      ⊢ Dvd.dvd (n.primeFactors.prod fun p => p) n
    -/
  · simpa [prod_primeFactorsList hn] using (n.primeFactorsList : Multiset ℕ).toFinset_prod_dvd_prod
    /-
      🎉 no goals
    -/


theorem factorization_gcd {a b : ℕ} (ha_pos : a ≠ 0) (hb_pos : b ≠ 0) :
    (gcd a b).factorization = a.factorization ⊓ b.factorization := by
  /-
    a b : Nat
    ha_pos : Ne a 0
    hb_pos : Ne b 0
    ⊢ Eq (a.gcd b).factorization (Min.min a.factorization b.factorization)
  -/
  let dfac := a.factorization ⊓ b.factorization
  /-
    a b : Nat
    ha_pos : Ne a 0
    hb_pos : Ne b 0
    dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
    ⊢ Eq (a.gcd b).factorization (Min.min a.factorization b.factorization)
  -/
  let d := dfac.prod (· ^ ·)
  have dfac_prime : ∀ p : ℕ, p ∈ dfac.support → Prime p := by
    intro p hp
    have : p ∈ a.primeFactorsList ∧ p ∈ b.primeFactorsList := by simpa [dfac] using hp
    exact prime_of_mem_primeFactorsList this.1
  /-
    a b : Nat
    ha_pos : Ne a 0
    hb_pos : Ne b 0
    dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
    d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
    dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
    ⊢ Eq (a.gcd b).factorization (Min.min a.factorization b.factorization)
  -/
  have h1 : d.factorization = dfac := prod_pow_factorization_eq_self dfac_prime
  /-
    a b : Nat
    ha_pos : Ne a 0
    hb_pos : Ne b 0
    dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
    d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
    dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
    h1 : Eq d.factorization dfac
    ⊢ Eq (a.gcd b).factorization (Min.min a.factorization b.factorization)
  -/
  have hd_pos : d ≠ 0 := (factorizationEquiv.invFun ⟨dfac, dfac_prime⟩).2.ne'
  /-
    a b : Nat
    ha_pos : Ne a 0
    hb_pos : Ne b 0
    dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
    d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
    dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
    h1 : Eq d.factorization dfac
    hd_pos : Ne d 0
    ⊢ Eq (a.gcd b).factorization (Min.min a.factorization b.factorization)
  -/
  suffices d = gcd a b by rwa [← this]
  /-
    a b : Nat
    ha_pos : Ne a 0
    hb_pos : Ne b 0
    dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
    d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
    dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
    h1 : Eq d.factorization dfac
    hd_pos : Ne d 0
    ⊢ Eq d (a.gcd b)
  -/
  apply gcd_greatest
    /-
      case hda
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      ⊢ Dvd.dvd d a
    -/
  · rw [← factorization_le_iff_dvd hd_pos ha_pos, h1]
    /-
      case hda
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      ⊢ LE.le dfac a.factorization
    -/
    exact inf_le_left
    /-
      🎉 no goals
    -/
    /-
      case hdb
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      ⊢ Dvd.dvd d b
    -/
  · rw [← factorization_le_iff_dvd hd_pos hb_pos, h1]
    /-
      case hdb
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      ⊢ LE.le dfac b.factorization
    -/
    exact inf_le_right
    /-
      🎉 no goals
    -/
    /-
      case hd
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      ⊢ ∀ (e : Nat), Dvd.dvd e a → Dvd.dvd e b → Dvd.dvd e d
    -/
  · intro e hea heb
    /-
      case hd
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      e : Nat
      hea : Dvd.dvd e a
      heb : Dvd.dvd e b
      ⊢ Dvd.dvd e d
    -/
    rcases Decidable.eq_or_ne e 0 with (rfl | he_pos)
      /-
        case hd.inl
        a b : Nat
        ha_pos : Ne a 0
        hb_pos : Ne b 0
        dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
        d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
        dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
        h1 : Eq d.factorization dfac
        hd_pos : Ne d 0
        hea : Dvd.dvd 0 a
        heb : Dvd.dvd 0 b
        ⊢ Dvd.dvd 0 d
      -/
    · simp only [zero_dvd_iff] at hea
      /-
        case hd.inl
        a b : Nat
        ha_pos : Ne a 0
        hb_pos : Ne b 0
        dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
        d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
        dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
        h1 : Eq d.factorization dfac
        hd_pos : Ne d 0
        heb : Dvd.dvd 0 b
        hea : Eq a 0
        ⊢ Dvd.dvd 0 d
      -/
      contradiction
      /-
        🎉 no goals
      -/
    /-
      case hd.inr
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      e : Nat
      hea : Dvd.dvd e a
      heb : Dvd.dvd e b
      he_pos : Ne e 0
      ⊢ Dvd.dvd e d
    -/
    have hea' := (factorization_le_iff_dvd he_pos ha_pos).mpr hea
    /-
      case hd.inr
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      e : Nat
      hea : Dvd.dvd e a
      heb : Dvd.dvd e b
      he_pos : Ne e 0
      hea' : LE.le e.factorization a.factorization
      ⊢ Dvd.dvd e d
    -/
    have heb' := (factorization_le_iff_dvd he_pos hb_pos).mpr heb
    /-
      case hd.inr
      a b : Nat
      ha_pos : Ne a 0
      hb_pos : Ne b 0
      dfac : Finsupp Nat Nat := Min.min a.factorization b.factorization
      d : Nat := dfac.prod fun x1 x2 => HPow.hPow x1 x2
      dfac_prime : ∀ (p : Nat), Membership.mem dfac.support p → Nat.Prime p
      h1 : Eq d.factorization dfac
      hd_pos : Ne d 0
      e : Nat
      hea : Dvd.dvd e a
      heb : Dvd.dvd e b
      he_pos : Ne e 0
      hea' : LE.le e.factorization a.factorization
      heb' : LE.le e.factorization b.factorization
      ⊢ Dvd.dvd e d
    -/
    simp [dfac, ← factorization_le_iff_dvd he_pos hd_pos, h1, hea', heb']
    /-
      🎉 no goals
    -/


theorem factorization_lcm {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0) :
    (a.lcm b).factorization = a.factorization ⊔ b.factorization := by
  rw [← add_right_inj (a.gcd b).factorization, ←
    factorization_mul (mt gcd_eq_zero_iff.1 fun h => ha h.1) (lcm_ne_zero ha hb), gcd_mul_lcm,
    factorization_gcd ha hb, factorization_mul ha hb]
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HAdd.hAdd a.factorization b.factorization) (HAdd.hAdd (Min.min a.factori …
  -/
  ext1
  /-
    case h
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    a✝ : Nat
    ⊢ Eq ((HAdd.hAdd a.factorization b.factorization) a✝) ((HAdd.hAdd (Min.min a.f …
  -/
  exact (min_add_max _ _).symm
  /-
    🎉 no goals
  -/


@[simp]
lemma factorizationLCMLeft_zero_left : factorizationLCMLeft 0 b = 1 := by
  /-
    b : Nat
    ⊢ Eq (Nat.factorizationLCMLeft 0 b) 1
  -/
  simp [factorizationLCMLeft]
  /-
    🎉 no goals
  -/


@[simp]
lemma factorizationLCMLeft_zero_right : factorizationLCMLeft a 0 = 1 := by
  /-
    a : Nat
    ⊢ Eq (a.factorizationLCMLeft 0) 1
  -/
  simp [factorizationLCMLeft]
  /-
    🎉 no goals
  -/


@[simp]
lemma factorizationLCRight_zero_left : factorizationLCMRight 0 b = 1 := by
  /-
    b : Nat
    ⊢ Eq (Nat.factorizationLCMRight 0 b) 1
  -/
  simp [factorizationLCMRight]
  /-
    🎉 no goals
  -/

@[simp]
lemma factorizationLCMRight_zero_right : factorizationLCMRight a 0 = 1 := by
  /-
    a : Nat
    ⊢ Eq (a.factorizationLCMRight 0) 1
  -/
  simp [factorizationLCMRight]
  /-
    🎉 no goals
  -/


lemma factorizationLCMLeft_pos :
    0 < factorizationLCMLeft a b := by
  /-
    a b : Nat
    ⊢ LT.lt 0 (a.factorizationLCMLeft b)
  -/
  apply Nat.pos_of_ne_zero
  /-
    case a
    a b : Nat
    ⊢ Ne (a.factorizationLCMLeft b) 0
  -/
  rw [factorizationLCMLeft, Finsupp.prod_ne_zero_iff]
  /-
    case a
    a b : Nat
    ⊢ ∀ (i : Nat), Membership.mem (a.lcm b).factorization.support i → Ne (ite (LE. …
  -/
  intro p _ H
  /-
    case a
    a b p : Nat
    a✝ : Membership.mem (a.lcm b).factorization.support p
    H : Eq (ite (LE.le (b.factorization p) (a.factorization p)) (HPow.hPow p ((a.l …
    ⊢ False
  -/
  by_cases h : b.factorization p ≤ a.factorization p
    /-
      case pos
      a b p : Nat
      a✝ : Membership.mem (a.lcm b).factorization.support p
      H : Eq (ite (LE.le (b.factorization p) (a.factorization p)) (HPow.hPow p ((a.l …
      h : LE.le (b.factorization p) (a.factorization p)
      ⊢ False
    -/
  · simp only [h, reduceIte, pow_eq_zero_iff', ne_eq] at H
    /-
      case pos
      a b p : Nat
      a✝ : Membership.mem (a.lcm b).factorization.support p
      h : LE.le (b.factorization p) (a.factorization p)
      H : And (Eq p 0) (Not (Eq ((a.lcm b).factorization p) 0))
      ⊢ False
    -/
    simpa [H.1] using H.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b p : Nat
      a✝ : Membership.mem (a.lcm b).factorization.support p
      H : Eq (ite (LE.le (b.factorization p) (a.factorization p)) (HPow.hPow p ((a.l …
      h : Not (LE.le (b.factorization p) (a.factorization p))
      ⊢ False
    -/
  · simp only [h, reduceIte, one_ne_zero] at H
    /-
      🎉 no goals
    -/


lemma factorizationLCMRight_pos :
    0 < factorizationLCMRight a b := by
  /-
    a b : Nat
    ⊢ LT.lt 0 (a.factorizationLCMRight b)
  -/
  apply Nat.pos_of_ne_zero
  /-
    case a
    a b : Nat
    ⊢ Ne (a.factorizationLCMRight b) 0
  -/
  rw [factorizationLCMRight, Finsupp.prod_ne_zero_iff]
  /-
    case a
    a b : Nat
    ⊢ ∀ (i : Nat), Membership.mem (a.lcm b).factorization.support i → Ne (ite (LE. …
  -/
  intro p _ H
  /-
    case a
    a b p : Nat
    a✝ : Membership.mem (a.lcm b).factorization.support p
    H : Eq (ite (LE.le (b.factorization p) (a.factorization p)) 1 (HPow.hPow p ((a …
    ⊢ False
  -/
  by_cases h : b.factorization p ≤ a.factorization p
    /-
      case pos
      a b p : Nat
      a✝ : Membership.mem (a.lcm b).factorization.support p
      H : Eq (ite (LE.le (b.factorization p) (a.factorization p)) 1 (HPow.hPow p ((a …
      h : LE.le (b.factorization p) (a.factorization p)
      ⊢ False
    -/
  · simp only [h, reduceIte, pow_eq_zero_iff', ne_eq, reduceCtorEq] at H
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b p : Nat
      a✝ : Membership.mem (a.lcm b).factorization.support p
      H : Eq (ite (LE.le (b.factorization p) (a.factorization p)) 1 (HPow.hPow p ((a …
      h : Not (LE.le (b.factorization p) (a.factorization p))
      ⊢ False
    -/
  · simp only [h, ↓reduceIte, pow_eq_zero_iff', ne_eq] at H
    /-
      case neg
      a b p : Nat
      a✝ : Membership.mem (a.lcm b).factorization.support p
      h : Not (LE.le (b.factorization p) (a.factorization p))
      H : And (Eq p 0) (Not (Eq ((a.lcm b).factorization p) 0))
      ⊢ False
    -/
    simpa [H.1] using H.2
    /-
      🎉 no goals
    -/


lemma coprime_factorizationLCMLeft_factorizationLCMRight :
    (factorizationLCMLeft a b).Coprime (factorizationLCMRight a b) := by
  /-
    a b : Nat
    ⊢ (a.factorizationLCMLeft b).Coprime (a.factorizationLCMRight b)
  -/
  rw [factorizationLCMLeft, factorizationLCMRight]
  /-
    a b : Nat
    ⊢ ((a.lcm b).factorization.prod fun p n => ite (LE.le (b.factorization p) (a.f …
  -/
  refine coprime_prod_left_iff.mpr fun p hp ↦ coprime_prod_right_iff.mpr fun q hq ↦ ?_
  /-
    a b p : Nat
    hp : Membership.mem (a.lcm b).factorization.support p
    q : Nat
    hq : Membership.mem (a.lcm b).factorization.support q
    ⊢ ((fun p n => ite (LE.le (b.factorization p) (a.factorization p)) (HPow.hPow  …
  -/
  dsimp only; split_ifs with h h'
  /-
    case pos
    a b p : Nat
    hp : Membership.mem (a.lcm b).factorization.support p
    q : Nat
    hq : Membership.mem (a.lcm b).factorization.support q
    h : LE.le (b.factorization p) (a.factorization p)
    h' : LE.le (b.factorization q) (a.factorization q)
    ⊢ (HPow.hPow p ((a.lcm b).factorization p)).Coprime 1
  -/
  any_goals simp only [coprime_one_right_eq_true, coprime_one_left_eq_true]
  /-
    case neg
    a b p : Nat
    hp : Membership.mem (a.lcm b).factorization.support p
    q : Nat
    hq : Membership.mem (a.lcm b).factorization.support q
    h : LE.le (b.factorization p) (a.factorization p)
    h' : Not (LE.le (b.factorization q) (a.factorization q))
    ⊢ (HPow.hPow p ((a.lcm b).factorization p)).Coprime (HPow.hPow q ((a.lcm b).fa …
  -/
  refine coprime_pow_primes _ _ (prime_of_mem_primeFactors hp) (prime_of_mem_primeFactors hq) ?_
  /-
    case neg
    a b p : Nat
    hp : Membership.mem (a.lcm b).factorization.support p
    q : Nat
    hq : Membership.mem (a.lcm b).factorization.support q
    h : LE.le (b.factorization p) (a.factorization p)
    h' : Not (LE.le (b.factorization q) (a.factorization q))
    ⊢ Ne p q
  -/
  contrapose! h'; rwa [← h']
                  /-
                    🎉 no goals
                  -/


lemma factorizationLCMLeft_mul_factorizationLCMRight (ha : a ≠ 0) (hb : b ≠ 0) :
    (factorizationLCMLeft a b) * (factorizationLCMRight a b) = lcm a b := by
  rw [← factorization_prod_pow_eq_self (lcm_ne_zero ha hb), factorizationLCMLeft,
    factorizationLCMRight, ← prod_mul]
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq ((a.lcm b).factorization.prod fun a_1 b_1 => HMul.hMul (ite (LE.le (b.fac …
  -/
                                /-
                                  🎉 no goals
                                -/
  congr; ext p n; split_ifs <;> simp
                                /-
                                  🎉 no goals
                                -/


lemma factorizationLCMLeft_dvd_left : factorizationLCMLeft a b ∣ a := by
  /-
    a b : Nat
    ⊢ Dvd.dvd (a.factorizationLCMLeft b) a
  -/
  rcases eq_or_ne a 0 with rfl | ha
    /-
      case inl
      b : Nat
      ⊢ Dvd.dvd (Nat.factorizationLCMLeft 0 b) 0
    -/
  · simp only [dvd_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    ha : Ne a 0
    ⊢ Dvd.dvd (a.factorizationLCMLeft b) a
  -/
  rcases eq_or_ne b 0 with rfl | hb
    /-
      case inr.inl
      a : Nat
      ha : Ne a 0
      ⊢ Dvd.dvd (a.factorizationLCMLeft 0) a
    -/
  · simp [factorizationLCMLeft]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Dvd.dvd (a.factorizationLCMLeft b) a
  -/
  nth_rewrite 2 [← factorization_prod_pow_eq_self ha]
  /-
    case inr.inr
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Dvd.dvd (a.factorizationLCMLeft b) (a.factorization.prod fun x1 x2 => HPow.h …
  -/
  rw [prod_of_support_subset (s := (lcm a b).factorization.support)]
    /-
      case inr.inr
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Dvd.dvd (a.factorizationLCMLeft b) ((a.lcm b).factorization.support.prod fun …
    -/
  · apply prod_dvd_prod_of_dvd; rintro p -; dsimp only; split_ifs with le
      /-
        case pos
        a b : Nat
        ha : Ne a 0
        hb : Ne b 0
        p : Nat
        le : LE.le (b.factorization p) (a.factorization p)
        ⊢ Dvd.dvd (HPow.hPow p ((a.lcm b).factorization p)) (HPow.hPow p (a.factorizat …
      -/
    · rw [factorization_lcm ha hb]; apply pow_dvd_pow; exact sup_le le_rfl le
                                                       /-
                                                         🎉 no goals
                                                       -/
      /-
        case neg
        a b : Nat
        ha : Ne a 0
        hb : Ne b 0
        p : Nat
        le : Not (LE.le (b.factorization p) (a.factorization p))
        ⊢ Dvd.dvd 1 (HPow.hPow p (a.factorization p))
      -/
    · apply one_dvd
      /-
        🎉 no goals
      -/
    /-
      case inr.inr.hs
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ HasSubset.Subset a.factorization.support (a.lcm b).factorization.support
    -/
  · intro p hp; rw [mem_support_iff] at hp ⊢
    /-
      case inr.inr.hs
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      p : Nat
      hp : Ne (a.factorization p) 0
      ⊢ Ne ((a.lcm b).factorization p) 0
    -/
    rw [factorization_lcm ha hb]; exact (lt_sup_iff.mpr <| .inl <| Nat.pos_of_ne_zero hp).ne'
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case inr.inr.h
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ ∀ (i : Nat), Membership.mem (a.lcm b).factorization.support i → Eq (HPow.hPo …
    -/
  · intros; rw [pow_zero]
            /-
              🎉 no goals
            -/


lemma factorizationLCMRight_dvd_right : factorizationLCMRight a b ∣ b := by
  /-
    a b : Nat
    ⊢ Dvd.dvd (a.factorizationLCMRight b) b
  -/
  rcases eq_or_ne a 0 with rfl | ha
    /-
      case inl
      b : Nat
      ⊢ Dvd.dvd (Nat.factorizationLCMRight 0 b) b
    -/
  · simp [factorizationLCMRight]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    ha : Ne a 0
    ⊢ Dvd.dvd (a.factorizationLCMRight b) b
  -/
  rcases eq_or_ne b 0 with rfl | hb
    /-
      case inr.inl
      a : Nat
      ha : Ne a 0
      ⊢ Dvd.dvd (a.factorizationLCMRight 0) 0
    -/
  · simp only [dvd_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Dvd.dvd (a.factorizationLCMRight b) b
  -/
  nth_rewrite 2 [← factorization_prod_pow_eq_self hb]
  /-
    case inr.inr
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Dvd.dvd (a.factorizationLCMRight b) (b.factorization.prod fun x1 x2 => HPow. …
  -/
  rw [prod_of_support_subset (s := (lcm a b).factorization.support)]
    /-
      case inr.inr
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Dvd.dvd (a.factorizationLCMRight b) ((a.lcm b).factorization.support.prod fu …
    -/
  · apply Finset.prod_dvd_prod_of_dvd; rintro p -; dsimp only; split_ifs with le
      /-
        case pos
        a b : Nat
        ha : Ne a 0
        hb : Ne b 0
        p : Nat
        le : LE.le (b.factorization p) (a.factorization p)
        ⊢ Dvd.dvd 1 (HPow.hPow p (b.factorization p))
      -/
    · apply one_dvd
      /-
        🎉 no goals
      -/
      /-
        case neg
        a b : Nat
        ha : Ne a 0
        hb : Ne b 0
        p : Nat
        le : Not (LE.le (b.factorization p) (a.factorization p))
        ⊢ Dvd.dvd (HPow.hPow p ((a.lcm b).factorization p)) (HPow.hPow p (b.factorizat …
      -/
    · rw [factorization_lcm ha hb]; apply pow_dvd_pow; exact sup_le (not_le.1 le).le le_rfl
                                                       /-
                                                         🎉 no goals
                                                       -/
    /-
      case inr.inr.hs
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ HasSubset.Subset b.factorization.support (a.lcm b).factorization.support
    -/
  · intro p hp; rw [mem_support_iff] at hp ⊢
    /-
      case inr.inr.hs
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      p : Nat
      hp : Ne (b.factorization p) 0
      ⊢ Ne ((a.lcm b).factorization p) 0
    -/
    rw [factorization_lcm ha hb]; exact (lt_sup_iff.mpr <| .inr <| Nat.pos_of_ne_zero hp).ne'
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case inr.inr.h
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ ∀ (i : Nat), Membership.mem (a.lcm b).factorization.support i → Eq (HPow.hPo …
    -/
  · intros; rw [pow_zero]
            /-
              🎉 no goals
            -/


@[to_additive sum_primeFactors_gcd_add_sum_primeFactors_mul]
theorem prod_primeFactors_gcd_mul_prod_primeFactors_mul {β : Type*} [CommMonoid β] (m n : ℕ)
    (f : ℕ → β) :
    (m.gcd n).primeFactors.prod f * (m * n).primeFactors.prod f =
      m.primeFactors.prod f * n.primeFactors.prod f := by
  /-
    β : Type u_1
    inst✝ : CommMonoid β
    m n : Nat
    f : Nat → β
    ⊢ Eq (HMul.hMul ((m.gcd n).primeFactors.prod f) ((HMul.hMul m n).primeFactors. …
  -/
  obtain rfl | hm₀ := eq_or_ne m 0
    /-
      case inl
      β : Type u_1
      inst✝ : CommMonoid β
      n : Nat
      f : Nat → β
      ⊢ Eq (HMul.hMul ((Nat.gcd 0 n).primeFactors.prod f) ((HMul.hMul 0 n).primeFact …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    β : Type u_1
    inst✝ : CommMonoid β
    m n : Nat
    f : Nat → β
    hm₀ : Ne m 0
    ⊢ Eq (HMul.hMul ((m.gcd n).primeFactors.prod f) ((HMul.hMul m n).primeFactors. …
  -/
  obtain rfl | hn₀ := eq_or_ne n 0
    /-
      case inr.inl
      β : Type u_1
      inst✝ : CommMonoid β
      m : Nat
      f : Nat → β
      hm₀ : Ne m 0
      ⊢ Eq (HMul.hMul ((m.gcd 0).primeFactors.prod f) ((HMul.hMul m 0).primeFactors. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      β : Type u_1
      inst✝ : CommMonoid β
      m n : Nat
      f : Nat → β
      hm₀ : Ne m 0
      hn₀ : Ne n 0
      ⊢ Eq (HMul.hMul ((m.gcd n).primeFactors.prod f) ((HMul.hMul m n).primeFactors. …
    -/
  · rw [primeFactors_mul hm₀ hn₀, primeFactors_gcd hm₀ hn₀, mul_comm, Finset.prod_union_inter]
    /-
      🎉 no goals
    -/


theorem setOf_pow_dvd_eq_Icc_factorization {n p : ℕ} (pp : p.Prime) (hn : n ≠ 0) :
    { i : ℕ | i ≠ 0 ∧ p ^ i ∣ n } = Set.Icc 1 (n.factorization p) := by
  /-
    n p : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    ⊢ Eq (setOf fun i => And (Ne i 0) (Dvd.dvd (HPow.hPow p i) n)) (Set.Icc 1 (n.f …
  -/
  ext
  /-
    case h
    n p : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    x✝ : Nat
    ⊢ Iff (Membership.mem (setOf fun i => And (Ne i 0) (Dvd.dvd (HPow.hPow p i) n) …
  -/
  simp [Nat.lt_succ_iff, one_le_iff_ne_zero, pp.pow_dvd_iff_le_factorization hn]
  /-
    🎉 no goals
  -/


/-- The set of positive powers of prime `p` that divide `n` is exactly the set of
positive natural numbers up to `n.factorization p`. -/
theorem Icc_factorization_eq_pow_dvd (n : ℕ) {p : ℕ} (pp : Prime p) :
    Icc 1 (n.factorization p) = {i ∈ Ico 1 n | p ^ i ∣ n} := by
  /-
    n p : Nat
    pp : Nat.Prime p
    ⊢ Eq (Finset.Icc 1 (n.factorization p)) (Finset.filter (fun i => Dvd.dvd (HPow …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      p : Nat
      pp : Nat.Prime p
      ⊢ Eq (Finset.Icc 1 ((Nat.factorization 0) p)) (Finset.filter (fun i => Dvd.dvd …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n p : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    ⊢ Eq (Finset.Icc 1 (n.factorization p)) (Finset.filter (fun i => Dvd.dvd (HPow …
  -/
  ext x
  simp only [mem_Icc, Finset.mem_filter, mem_Ico, and_assoc, and_congr_right_iff,
    pp.pow_dvd_iff_le_factorization hn, iff_and_self]
  /-
    case inr.h
    n p : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    x : Nat
    ⊢ LE.le 1 x → LE.le x (n.factorization p) → LT.lt x n
  -/
  exact fun _ H => lt_of_le_of_lt H (factorization_lt p hn)
  /-
    🎉 no goals
  -/


theorem factorization_eq_card_pow_dvd (n : ℕ) {p : ℕ} (pp : p.Prime) :
    n.factorization p = #{i ∈ Ico 1 n | p ^ i ∣ n} := by
  /-
    n p : Nat
    pp : Nat.Prime p
    ⊢ Eq (n.factorization p) (Finset.filter (fun i => Dvd.dvd (HPow.hPow p i) n) ( …
  -/
  simp [← Icc_factorization_eq_pow_dvd n pp]
  /-
    🎉 no goals
  -/


theorem Ico_filter_pow_dvd_eq {n p b : ℕ} (pp : p.Prime) (hn : n ≠ 0) (hb : n ≤ p ^ b) :
    {i ∈ Ico 1 n | p ^ i ∣ n} = {i ∈ Icc 1 b | p ^ i ∣ n} := by
  /-
    n p b : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    hb : LE.le n (HPow.hPow p b)
    ⊢ Eq (Finset.filter (fun i => Dvd.dvd (HPow.hPow p i) n) (Finset.Ico 1 n)) (Fi …
  -/
  ext x
  /-
    case h
    n p b : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    hb : LE.le n (HPow.hPow p b)
    x : Nat
    ⊢ Iff (Membership.mem (Finset.filter (fun i => Dvd.dvd (HPow.hPow p i) n) (Fin …
  -/
  simp only [Finset.mem_filter, mem_Ico, mem_Icc, and_congr_left_iff, and_congr_right_iff]
  /-
    case h
    n p b : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    hb : LE.le n (HPow.hPow p b)
    x : Nat
    ⊢ Dvd.dvd (HPow.hPow p x) n → LE.le 1 x → Iff (LT.lt x n) (LE.le x b)
  -/
  rintro h1 -
  exact iff_of_true (lt_of_pow_dvd_right hn pp.two_le h1) <|
    (Nat.pow_le_pow_iff_right pp.one_lt).1 <| (le_of_dvd hn.bot_lt h1).trans hb


/-- If `p` is a prime factor of `a` then the power of `p` in `a` is the same that in `a * b`,
for any `b` coprime to `a`. -/
theorem factorization_eq_of_coprime_left {p a b : ℕ} (hab : Coprime a b)
    (hpa : p ∈ a.primeFactorsList) : (a * b).factorization p = a.factorization p := by
  rw [factorization_mul_apply_of_coprime hab, ← primeFactorsList_count_eq,
    ← primeFactorsList_count_eq,
    count_eq_zero_of_not_mem (coprime_primeFactorsList_disjoint hab hpa), add_zero]


/-- If `p` is a prime factor of `b` then the power of `p` in `b` is the same that in `a * b`,
for any `a` coprime to `b`. -/
theorem factorization_eq_of_coprime_right {p a b : ℕ} (hab : Coprime a b)
    (hpb : p ∈ b.primeFactorsList) : (a * b).factorization p = b.factorization p := by
  /-
    p a b : Nat
    hab : a.Coprime b
    hpb : Membership.mem b.primeFactorsList p
    ⊢ Eq ((HMul.hMul a b).factorization p) (b.factorization p)
  -/
  rw [mul_comm]
  /-
    p a b : Nat
    hab : a.Coprime b
    hpb : Membership.mem b.primeFactorsList p
    ⊢ Eq ((HMul.hMul b a).factorization p) (b.factorization p)
  -/
  exact factorization_eq_of_coprime_left (coprime_comm.mp hab) hpb
  /-
    🎉 no goals
  -/


/-- Two positive naturals are equal if their prime padic valuations are equal -/
theorem eq_iff_prime_padicValNat_eq (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) :
    a = b ↔ ∀ p : ℕ, p.Prime → padicValNat p a = padicValNat p b := by
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Iff (Eq a b) (∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p …
  -/
  constructor
    /-
      case mp
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq a b → ∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p b)
    -/
  · rintro rfl
    /-
      case mp
      a : Nat
      ha hb : Ne a 0
      ⊢ ∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p a)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ (∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p b)) → Eq a b
    -/
  · intro h
    /-
      case mpr
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      h : ∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p b)
      ⊢ Eq a b
    -/
    refine eq_of_factorization_eq ha hb fun p => ?_
    /-
      case mpr
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      h : ∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p b)
      p : Nat
      ⊢ Eq (a.factorization p) (b.factorization p)
    -/
    by_cases pp : p.Prime
      /-
        case pos
        a b : Nat
        ha : Ne a 0
        hb : Ne b 0
        h : ∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p b)
        p : Nat
        pp : Nat.Prime p
        ⊢ Eq (a.factorization p) (b.factorization p)
      -/
    · simp [factorization_def, pp, h p pp]
      /-
        🎉 no goals
      -/
      /-
        case neg
        a b : Nat
        ha : Ne a 0
        hb : Ne b 0
        h : ∀ (p : Nat), Nat.Prime p → Eq (padicValNat p a) (padicValNat p b)
        p : Nat
        pp : Not (Nat.Prime p)
        ⊢ Eq (a.factorization p) (b.factorization p)
      -/
    · simp [factorization_eq_zero_of_non_prime, pp]
      /-
        🎉 no goals
      -/


theorem prod_pow_prime_padicValNat (n : Nat) (hn : n ≠ 0) (m : Nat) (pr : n < m) :
    ∏ p ∈ range m with p.Prime, p ^ padicValNat p n = n := by
  -- Porting note: was `nth_rw_rhs`
  conv =>
    rhs
    rw [← factorization_prod_pow_eq_self hn]
  /-
    n : Nat
    hn : Ne n 0
    m : Nat
    pr : LT.lt n m
    ⊢ Eq ((Finset.filter (fun p => Nat.Prime p) (Finset.range m)).prod fun p => HP …
  -/
  rw [eq_comm]
  /-
    n : Nat
    hn : Ne n 0
    m : Nat
    pr : LT.lt n m
    ⊢ Eq (n.factorization.prod fun x1 x2 => HPow.hPow x1 x2) ((Finset.filter (fun  …
  -/
  apply Finset.prod_subset_one_on_sdiff
  · exact fun p hp => Finset.mem_filter.mpr ⟨Finset.mem_range.2 <| pr.trans_le' <|
      le_of_mem_primeFactors hp, prime_of_mem_primeFactors hp⟩
    /-
      case hg
      n : Nat
      hn : Ne n 0
      m : Nat
      pr : LT.lt n m
      ⊢ ∀ (x : Nat), Membership.mem (SDiff.sdiff (Finset.filter (fun p => Nat.Prime  …
    -/
  · intro p hp
    /-
      case hg
      n : Nat
      hn : Ne n 0
      m : Nat
      pr : LT.lt n m
      p : Nat
      hp : Membership.mem (SDiff.sdiff (Finset.filter (fun p => Nat.Prime p) (Finset …
      ⊢ Eq (HPow.hPow p (padicValNat p n)) 1
    -/
    cases' Finset.mem_sdiff.mp hp with hp1 hp2
    /-
      case hg.intro
      n : Nat
      hn : Ne n 0
      m : Nat
      pr : LT.lt n m
      p : Nat
      hp : Membership.mem (SDiff.sdiff (Finset.filter (fun p => Nat.Prime p) (Finset …
      hp1 : Membership.mem (Finset.filter (fun p => Nat.Prime p) (Finset.range m)) p
      hp2 : Not (Membership.mem n.factorization.support p)
      ⊢ Eq (HPow.hPow p (padicValNat p n)) 1
    -/
    rw [← factorization_def n (Finset.mem_filter.mp hp1).2]
    /-
      case hg.intro
      n : Nat
      hn : Ne n 0
      m : Nat
      pr : LT.lt n m
      p : Nat
      hp : Membership.mem (SDiff.sdiff (Finset.filter (fun p => Nat.Prime p) (Finset …
      hp1 : Membership.mem (Finset.filter (fun p => Nat.Prime p) (Finset.range m)) p
      hp2 : Not (Membership.mem n.factorization.support p)
      ⊢ Eq (HPow.hPow p (n.factorization p)) 1
    -/
    simp [Finsupp.not_mem_support_iff.mp hp2]
    /-
      🎉 no goals
    -/
    /-
      case hfg
      n : Nat
      hn : Ne n 0
      m : Nat
      pr : LT.lt n m
      ⊢ ∀ (x : Nat), Membership.mem n.factorization.support x → Eq ((fun x1 x2 => HP …
    -/
  · intro p hp
    /-
      case hfg
      n : Nat
      hn : Ne n 0
      m : Nat
      pr : LT.lt n m
      p : Nat
      hp : Membership.mem n.factorization.support p
      ⊢ Eq ((fun x1 x2 => HPow.hPow x1 x2) p (n.factorization p)) (HPow.hPow p (padi …
    -/
    simp [factorization_def n (prime_of_mem_primeFactors hp)]
    /-
      🎉 no goals
    -/


/-- Exactly `n / p` naturals in `[1, n]` are multiples of `p`.
See `Nat.card_multiples'` for an alternative spelling of the statement. -/
theorem card_multiples (n p : ℕ) : #{e ∈ range n | p ∣ e + 1} = n / p := by
  /-
    n p : Nat
    ⊢ Eq (Finset.filter (fun e => Dvd.dvd p (HAdd.hAdd e 1)) (Finset.range n)).car …
  -/
  induction' n with n hn
    /-
      case zero
      p : Nat
      ⊢ Eq (Finset.filter (fun e => Dvd.dvd p (HAdd.hAdd e 1)) (Finset.range 0)).car …
    -/
  · simp
    /-
      🎉 no goals
    -/
  simp [Nat.succ_div, add_ite, add_zero, Finset.range_succ, filter_insert, apply_ite card,
    card_insert_of_not_mem, hn]


/-- Exactly `n / p` naturals in `(0, n]` are multiples of `p`. -/
theorem Ioc_filter_dvd_card_eq_div (n p : ℕ) : #{x ∈ Ioc 0 n | p ∣ x} = n / p := by
  /-
    n p : Nat
    ⊢ Eq (Finset.filter (fun x => Dvd.dvd p x) (Finset.Ioc 0 n)).card (HDiv.hDiv n …
  -/
  induction' n with n IH
    /-
      case zero
      p : Nat
      ⊢ Eq (Finset.filter (fun x => Dvd.dvd p x) (Finset.Ioc 0 0)).card (HDiv.hDiv 0 …
    -/
  · simp
    /-
      🎉 no goals
    -/
  -- TODO: Golf away `h1` after Yaël PRs a lemma asserting this
  have h1 : Ioc 0 n.succ = insert n.succ (Ioc 0 n) := by
    rcases n.eq_zero_or_pos with (rfl | hn)
    · simp
    simp_rw [← Ico_succ_succ, Ico_insert_right (succ_le_succ hn.le), Ico_succ_right]
  simp [Nat.succ_div, add_ite, add_zero, h1, filter_insert, apply_ite card, card_insert_eq_ite, IH,
    Finset.mem_filter, mem_Ioc, not_le.2 (lt_add_one n)]


/-- There are exactly `⌊N/n⌋` positive multiples of `n` that are `≤ N`.
See `Nat.card_multiples` for a "shifted-by-one" version. -/
lemma card_multiples' (N n : ℕ) : #{k ∈ range N.succ | k ≠ 0 ∧ n ∣ k} = N / n := by
  induction N with
    | zero => simp [Finset.filter_false_of_mem]
    | succ N ih =>
        rw [Finset.range_succ, Finset.filter_insert]
        by_cases h : n ∣ N.succ
        · simp [h, succ_div_of_dvd, ih]
        · simp [h, succ_div_of_not_dvd, ih]


