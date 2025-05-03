theorem IsPrimePow.minFac_pow_factorization_eq {n : ℕ} (hn : IsPrimePow n) :
    n.minFac ^ n.factorization n.minFac = n := by
  /-
    n : Nat
    hn : IsPrimePow n
    ⊢ Eq (HPow.hPow n.minFac (n.factorization n.minFac)) n
  -/
  obtain ⟨p, k, hp, hk, rfl⟩ := hn
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : Prime p
    hk : LT.lt 0 k
    ⊢ Eq (HPow.hPow (HPow.hPow p k).minFac ((HPow.hPow p k).factorization (HPow.hP …
  -/
  rw [← Nat.prime_iff] at hp
  /-
    case intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    ⊢ Eq (HPow.hPow (HPow.hPow p k).minFac ((HPow.hPow p k).factorization (HPow.hP …
  -/
  rw [hp.pow_minFac hk.ne', hp.factorization_pow, Finsupp.single_eq_same]
  /-
    🎉 no goals
  -/


theorem isPrimePow_of_minFac_pow_factorization_eq {n : ℕ}
    (h : n.minFac ^ n.factorization n.minFac = n) (hn : n ≠ 1) : IsPrimePow n := by
  /-
    n : Nat
    h : Eq (HPow.hPow n.minFac (n.factorization n.minFac)) n
    hn : Ne n 1
    ⊢ IsPrimePow n
  -/
  rcases eq_or_ne n 0 with (rfl | hn')
    /-
      case inl
      h : Eq (HPow.hPow (Nat.minFac 0) ((Nat.factorization 0) (Nat.minFac 0))) 0
      hn : Ne 0 1
      ⊢ IsPrimePow 0
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    h : Eq (HPow.hPow n.minFac (n.factorization n.minFac)) n
    hn : Ne n 1
    hn' : Ne n 0
    ⊢ IsPrimePow n
  -/
  refine ⟨_, _, (Nat.minFac_prime hn).prime, ?_, h⟩
  simp [pos_iff_ne_zero, ← Finsupp.mem_support_iff, Nat.support_factorization, hn',
    Nat.minFac_prime hn, Nat.minFac_dvd]


theorem isPrimePow_iff_minFac_pow_factorization_eq {n : ℕ} (hn : n ≠ 1) :
    IsPrimePow n ↔ n.minFac ^ n.factorization n.minFac = n :=
  ⟨fun h => h.minFac_pow_factorization_eq, fun h => isPrimePow_of_minFac_pow_factorization_eq h hn⟩


theorem isPrimePow_iff_factorization_eq_single {n : ℕ} :
    IsPrimePow n ↔ ∃ p k : ℕ, 0 < k ∧ n.factorization = Finsupp.single p k := by
  /-
    n : Nat
    ⊢ Iff (IsPrimePow n) (Exists fun p => Exists fun k => And (LT.lt 0 k) (Eq n.fa …
  -/
  rw [isPrimePow_nat_iff]
  /-
    n : Nat
    ⊢ Iff (Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq  …
  -/
  refine exists₂_congr fun p k => ?_
  /-
    n p k : Nat
    ⊢ Iff (And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow.hPow p k) n))) (And (LT.lt …
  -/
  constructor
    /-
      case mp
      n p k : Nat
      ⊢ And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow.hPow p k) n)) → And (LT.lt 0 k) …
    -/
  · rintro ⟨hp, hk, hn⟩
    /-
      case mp.intro.intro
      n p k : Nat
      hp : Nat.Prime p
      hk : LT.lt 0 k
      hn : Eq (HPow.hPow p k) n
      ⊢ And (LT.lt 0 k) (Eq n.factorization (Finsupp.single p k))
    -/
    exact ⟨hk, by rw [← hn, Nat.Prime.factorization_pow hp]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n p k : Nat
      ⊢ And (LT.lt 0 k) (Eq n.factorization (Finsupp.single p k)) → And (Nat.Prime p …
    -/
  · rintro ⟨hk, hn⟩
    have hn0 : n ≠ 0 := by
      rintro rfl
      simp_all only [Finsupp.single_eq_zero, eq_comm, Nat.factorization_zero, hk.ne']
    /-
      case mpr.intro
      n p k : Nat
      hk : LT.lt 0 k
      hn : Eq n.factorization (Finsupp.single p k)
      hn0 : Ne n 0
      ⊢ And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow.hPow p k) n))
    -/
    rw [Nat.eq_pow_of_factorization_eq_single hn0 hn]
    exact ⟨Nat.prime_of_mem_primeFactors <|
      Finsupp.mem_support_iff.2 (by simp [hn, hk.ne'] : n.factorization p ≠ 0), hk, rfl⟩


theorem isPrimePow_iff_card_primeFactors_eq_one {n : ℕ} :
    IsPrimePow n ↔ n.primeFactors.card = 1 := by
  simp_rw [isPrimePow_iff_factorization_eq_single, ← Nat.support_factorization,
    Finsupp.card_support_eq_one', pos_iff_ne_zero]


theorem IsPrimePow.exists_ordCompl_eq_one {n : ℕ} (h : IsPrimePow n) :
    ∃ p : ℕ, p.Prime ∧ ordCompl[p] n = 1 := by
  /-
    n : Nat
    h : IsPrimePow n
    ⊢ Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow.hPow p (n.factoriza …
  -/
  rcases eq_or_ne n 0 with (rfl | hn0); · cases not_isPrimePow_zero h
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    n : Nat
    h : IsPrimePow n
    hn0 : Ne n 0
    ⊢ Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow.hPow p (n.factoriza …
  -/
  rcases isPrimePow_iff_factorization_eq_single.mp h with ⟨p, k, hk0, h1⟩
  /-
    case inr.intro.intro.intro
    n : Nat
    h : IsPrimePow n
    hn0 : Ne n 0
    p k : Nat
    hk0 : LT.lt 0 k
    h1 : Eq n.factorization (Finsupp.single p k)
    ⊢ Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow.hPow p (n.factoriza …
  -/
  rcases em' p.Prime with (pp | pp)
    /-
      case inr.intro.intro.intro.inl
      n : Nat
      h : IsPrimePow n
      hn0 : Ne n 0
      p k : Nat
      hk0 : LT.lt 0 k
      h1 : Eq n.factorization (Finsupp.single p k)
      pp : Not (Nat.Prime p)
      ⊢ Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow.hPow p (n.factoriza …
    -/
  · refine absurd ?_ hk0.ne'
    /-
      case inr.intro.intro.intro.inl
      n : Nat
      h : IsPrimePow n
      hn0 : Ne n 0
      p k : Nat
      hk0 : LT.lt 0 k
      h1 : Eq n.factorization (Finsupp.single p k)
      pp : Not (Nat.Prime p)
      ⊢ Eq k 0
    -/
    simp [← Nat.factorization_eq_zero_of_non_prime n pp, h1]
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro.inr
    n : Nat
    h : IsPrimePow n
    hn0 : Ne n 0
    p k : Nat
    hk0 : LT.lt 0 k
    h1 : Eq n.factorization (Finsupp.single p k)
    pp : Nat.Prime p
    ⊢ Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow.hPow p (n.factoriza …
  -/
  refine ⟨p, pp, ?_⟩
  /-
    case inr.intro.intro.intro.inr
    n : Nat
    h : IsPrimePow n
    hn0 : Ne n 0
    p k : Nat
    hk0 : LT.lt 0 k
    h1 : Eq n.factorization (Finsupp.single p k)
    pp : Nat.Prime p
    ⊢ Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) 1
  -/
  refine Nat.eq_of_factorization_eq (Nat.ordCompl_pos p hn0).ne' (by simp) fun q => ?_
  /-
    case inr.intro.intro.intro.inr
    n : Nat
    h : IsPrimePow n
    hn0 : Ne n 0
    p k : Nat
    hk0 : LT.lt 0 k
    h1 : Eq n.factorization (Finsupp.single p k)
    pp : Nat.Prime p
    q : Nat
    ⊢ Eq ((HDiv.hDiv n (HPow.hPow p (n.factorization p))).factorization q) ((Nat.f …
  -/
  rw [Nat.factorization_ordCompl n p, h1]
  /-
    case inr.intro.intro.intro.inr
    n : Nat
    h : IsPrimePow n
    hn0 : Ne n 0
    p k : Nat
    hk0 : LT.lt 0 k
    h1 : Eq n.factorization (Finsupp.single p k)
    pp : Nat.Prime p
    q : Nat
    ⊢ Eq ((Finsupp.erase p (Finsupp.single p k)) q) ((Nat.factorization 1) q)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias IsPrimePow.exists_ord_compl_eq_one := IsPrimePow.exists_ordCompl_eq_one


theorem exists_ordCompl_eq_one_iff_isPrimePow {n : ℕ} (hn : n ≠ 1) :
    IsPrimePow n ↔ ∃ p : ℕ, p.Prime ∧ ordCompl[p] n = 1 := by
  /-
    n : Nat
    hn : Ne n 1
    ⊢ Iff (IsPrimePow n) (Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow …
  -/
  refine ⟨fun h => IsPrimePow.exists_ordCompl_eq_one h, fun h => ?_⟩
  /-
    n : Nat
    hn : Ne n 1
    h : Exists fun p => And (Nat.Prime p) (Eq (HDiv.hDiv n (HPow.hPow p (n.factori …
    ⊢ IsPrimePow n
  -/
  rcases h with ⟨p, pp, h⟩
  /-
    case intro.intro
    n : Nat
    hn : Ne n 1
    p : Nat
    pp : Nat.Prime p
    h : Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) 1
    ⊢ IsPrimePow n
  -/
  rw [isPrimePow_nat_iff]
  /-
    case intro.intro
    n : Nat
    hn : Ne n 1
    p : Nat
    pp : Nat.Prime p
    h : Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) 1
    ⊢ Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow …
  -/
  rw [← Nat.eq_of_dvd_of_div_eq_one (Nat.ordProj_dvd n p) h] at hn ⊢
  /-
    case intro.intro
    n p : Nat
    hn : Ne (HPow.hPow p (n.factorization p)) 1
    pp : Nat.Prime p
    h : Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) 1
    ⊢ Exists fun p_1 => Exists fun k => And (Nat.Prime p_1) (And (LT.lt 0 k) (Eq ( …
  -/
  refine ⟨p, n.factorization p, pp, ?_, by simp⟩
  /-
    case intro.intro
    n p : Nat
    hn : Ne (HPow.hPow p (n.factorization p)) 1
    pp : Nat.Prime p
    h : Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) 1
    ⊢ LT.lt 0 (n.factorization p)
  -/
  contrapose! hn
  /-
    case intro.intro
    n p : Nat
    pp : Nat.Prime p
    h : Eq (HDiv.hDiv n (HPow.hPow p (n.factorization p))) 1
    hn : LE.le (n.factorization p) 0
    ⊢ Eq (HPow.hPow p (n.factorization p)) 1
  -/
  simp [Nat.le_zero.1 hn]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")]
alias exists_ord_compl_eq_one_iff_isPrimePow := exists_ordCompl_eq_one_iff_isPrimePow


/-- An equivalent definition for prime powers: `n` is a prime power iff there is a unique prime
dividing it. -/
theorem isPrimePow_iff_unique_prime_dvd {n : ℕ} : IsPrimePow n ↔ ∃! p : ℕ, p.Prime ∧ p ∣ n := by
  /-
    n : Nat
    ⊢ Iff (IsPrimePow n) (ExistsUnique fun p => And (Nat.Prime p) (Dvd.dvd p n))
  -/
  rw [isPrimePow_nat_iff]
  /-
    n : Nat
    ⊢ Iff (Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq  …
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ (Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPo …
    -/
  · rintro ⟨p, k, hp, hk, rfl⟩
    /-
      case mp.intro.intro.intro.intro
      p k : Nat
      hp : Nat.Prime p
      hk : LT.lt 0 k
      ⊢ ExistsUnique fun p_1 => And (Nat.Prime p_1) (Dvd.dvd p_1 (HPow.hPow p k))
    -/
    refine ⟨p, ⟨hp, dvd_pow_self _ hk.ne'⟩, ?_⟩
    /-
      case mp.intro.intro.intro.intro
      p k : Nat
      hp : Nat.Prime p
      hk : LT.lt 0 k
      ⊢ ∀ (y : Nat), (fun p_1 => And (Nat.Prime p_1) (Dvd.dvd p_1 (HPow.hPow p k)))  …
    -/
    rintro q ⟨hq, hq'⟩
    /-
      case mp.intro.intro.intro.intro.intro
      p k : Nat
      hp : Nat.Prime p
      hk : LT.lt 0 k
      q : Nat
      hq : Nat.Prime q
      hq' : Dvd.dvd q (HPow.hPow p k)
      ⊢ Eq q p
    -/
    exact (Nat.prime_dvd_prime_iff_eq hq hp).1 (hq.dvd_of_dvd_pow hq')
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    ⊢ (ExistsUnique fun p => And (Nat.Prime p) (Dvd.dvd p n)) → Exists fun p => Ex …
  -/
  rintro ⟨p, ⟨hp, hn⟩, hq⟩
  /-
    case mpr.intro.intro.intro
    n p : Nat
    hq : ∀ (y : Nat), (fun p => And (Nat.Prime p) (Dvd.dvd p n)) y → Eq y p
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    ⊢ Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow …
  -/
  rcases eq_or_ne n 0 with (rfl | hn₀)
    /-
      case mpr.intro.intro.intro.inl
      p : Nat
      hp : Nat.Prime p
      hq : ∀ (y : Nat), (fun p => And (Nat.Prime p) (Dvd.dvd p 0)) y → Eq y p
      hn : Dvd.dvd p 0
      ⊢ Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow …
    -/
  · cases (hq 2 ⟨Nat.prime_two, dvd_zero 2⟩).trans (hq 3 ⟨Nat.prime_three, dvd_zero 3⟩).symm
    /-
      🎉 no goals
    -/
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hq : ∀ (y : Nat), (fun p => And (Nat.Prime p) (Dvd.dvd p n)) y → Eq y p
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    ⊢ Exists fun p => Exists fun k => And (Nat.Prime p) (And (LT.lt 0 k) (Eq (HPow …
  -/
  refine ⟨p, n.factorization p, hp, hp.factorization_pos_of_dvd hn₀ hn, ?_⟩
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hq : ∀ (y : Nat), (fun p => And (Nat.Prime p) (Dvd.dvd p n)) y → Eq y p
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    ⊢ Eq (HPow.hPow p (n.factorization p)) n
  -/
  simp only [and_imp] at hq
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    ⊢ Eq (HPow.hPow p (n.factorization p)) n
  -/
  apply Nat.dvd_antisymm (Nat.ordProj_dvd _ _)
  -- We need to show n ∣ p ^ n.factorization p
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    ⊢ Dvd.dvd n (HPow.hPow p (n.factorization p))
  -/
  apply Nat.dvd_of_primeFactorsList_subperm hn₀
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    ⊢ n.primeFactorsList.Subperm (HPow.hPow p (n.factorization p)).primeFactorsList
  -/
  rw [hp.primeFactorsList_pow, List.subperm_ext_iff]
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    ⊢ ∀ (x : Nat), Membership.mem n.primeFactorsList x → LE.le (List.count x n.pri …
  -/
  intro q hq'
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    q : Nat
    hq' : Membership.mem n.primeFactorsList q
    ⊢ LE.le (List.count q n.primeFactorsList) (List.count q (List.replicate (n.fac …
  -/
  rw [Nat.mem_primeFactorsList hn₀] at hq'
  /-
    case mpr.intro.intro.intro.inr
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    q : Nat
    hq' : And (Nat.Prime q) (Dvd.dvd q n)
    ⊢ LE.le (List.count q n.primeFactorsList) (List.count q (List.replicate (n.fac …
  -/
  cases hq _ hq'.1 hq'.2
  /-
    case mpr.intro.intro.intro.inr.refl
    n p : Nat
    hp : Nat.Prime p
    hn : Dvd.dvd p n
    hn₀ : Ne n 0
    hq : ∀ (y : Nat), Nat.Prime y → Dvd.dvd y n → Eq y p
    hq' : And (Nat.Prime p) (Dvd.dvd p n)
    ⊢ LE.le (List.count p n.primeFactorsList) (List.count p (List.replicate (n.fac …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isPrimePow_pow_iff {n k : ℕ} (hk : k ≠ 0) : IsPrimePow (n ^ k) ↔ IsPrimePow n := by
  /-
    n k : Nat
    hk : Ne k 0
    ⊢ Iff (IsPrimePow (HPow.hPow n k)) (IsPrimePow n)
  -/
  simp only [isPrimePow_iff_unique_prime_dvd]
  /-
    n k : Nat
    hk : Ne k 0
    ⊢ Iff (ExistsUnique fun p => And (Nat.Prime p) (Dvd.dvd p (HPow.hPow n k))) (E …
  -/
  apply existsUnique_congr
  /-
    case h
    n k : Nat
    hk : Ne k 0
    ⊢ ∀ (a : Nat), Iff (And (Nat.Prime a) (Dvd.dvd a (HPow.hPow n k))) (And (Nat.P …
  -/
  simp only [and_congr_right_iff]
  /-
    case h
    n k : Nat
    hk : Ne k 0
    ⊢ ∀ (a : Nat), Nat.Prime a → Iff (Dvd.dvd a (HPow.hPow n k)) (Dvd.dvd a n)
  -/
  intro p hp
  /-
    case h
    n k : Nat
    hk : Ne k 0
    p : Nat
    hp : Nat.Prime p
    ⊢ Iff (Dvd.dvd p (HPow.hPow n k)) (Dvd.dvd p n)
  -/
  exact ⟨hp.dvd_of_dvd_pow, fun t => t.trans (dvd_pow_self _ hk)⟩
  /-
    🎉 no goals
  -/


theorem Nat.Coprime.isPrimePow_dvd_mul {n a b : ℕ} (hab : Nat.Coprime a b) (hn : IsPrimePow n) :
    n ∣ a * b ↔ n ∣ a ∨ n ∣ b := by
  /-
    n a b : Nat
    hab : a.Coprime b
    hn : IsPrimePow n
    ⊢ Iff (Dvd.dvd n (HMul.hMul a b)) (Or (Dvd.dvd n a) (Dvd.dvd n b))
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      n b : Nat
      hn : IsPrimePow n
      hab : Nat.Coprime 0 b
      ⊢ Iff (Dvd.dvd n (HMul.hMul 0 b)) (Or (Dvd.dvd n 0) (Dvd.dvd n b))
    -/
  · simp only [Nat.coprime_zero_left] at hab
    /-
      case inl
      n b : Nat
      hn : IsPrimePow n
      hab : Eq b 1
      ⊢ Iff (Dvd.dvd n (HMul.hMul 0 b)) (Or (Dvd.dvd n 0) (Dvd.dvd n b))
    -/
    simp [hab, Finset.filter_singleton, not_isPrimePow_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n a b : Nat
    hab : a.Coprime b
    hn : IsPrimePow n
    ha : Ne a 0
    ⊢ Iff (Dvd.dvd n (HMul.hMul a b)) (Or (Dvd.dvd n a) (Dvd.dvd n b))
  -/
  rcases eq_or_ne b 0 with (rfl | hb)
    /-
      case inr.inl
      n a : Nat
      hn : IsPrimePow n
      ha : Ne a 0
      hab : a.Coprime 0
      ⊢ Iff (Dvd.dvd n (HMul.hMul a 0)) (Or (Dvd.dvd n a) (Dvd.dvd n 0))
    -/
  · simp only [Nat.coprime_zero_right] at hab
    /-
      case inr.inl
      n a : Nat
      hn : IsPrimePow n
      ha : Ne a 0
      hab : Eq a 1
      ⊢ Iff (Dvd.dvd n (HMul.hMul a 0)) (Or (Dvd.dvd n a) (Dvd.dvd n 0))
    -/
    simp [hab, Finset.filter_singleton, not_isPrimePow_one]
    /-
      🎉 no goals
    -/
  refine
    ⟨?_, fun h =>
      Or.elim h (fun i => i.trans ((@dvd_mul_right a b a hab).mpr (dvd_refl a)))
          fun i => i.trans ((@dvd_mul_left a b b hab.symm).mpr (dvd_refl b))⟩
  /-
    case inr.inr
    n a b : Nat
    hab : a.Coprime b
    hn : IsPrimePow n
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Dvd.dvd n (HMul.hMul a b) → Or (Dvd.dvd n a) (Dvd.dvd n b)
  -/
  obtain ⟨p, k, hp, _, rfl⟩ := (isPrimePow_nat_iff _).1 hn
  simp only [hp.pow_dvd_iff_le_factorization (mul_ne_zero ha hb), Nat.factorization_mul ha hb,
    hp.pow_dvd_iff_le_factorization ha, hp.pow_dvd_iff_le_factorization hb, Pi.add_apply,
    Finsupp.coe_add]
  have : a.factorization p = 0 ∨ b.factorization p = 0 := by
    rw [← Finsupp.not_mem_support_iff, ← Finsupp.not_mem_support_iff, ← not_and_or, ←
      Finset.mem_inter]
    intro t -- Porting note: used to be `exact` below, but the definition of `∈` has changed.
    simpa using hab.disjoint_primeFactors.le_bot t
  /-
    case inr.inr.intro.intro.intro.intro
    a b : Nat
    hab : a.Coprime b
    ha : Ne a 0
    hb : Ne b 0
    p k : Nat
    hp : Nat.Prime p
    left✝ : LT.lt 0 k
    hn : IsPrimePow (HPow.hPow p k)
    this : Or (Eq (a.factorization p) 0) (Eq (b.factorization p) 0)
    ⊢ LE.le k (HAdd.hAdd (a.factorization p) (b.factorization p)) → Or (LE.le k (a …
  -/
                           /-
                             🎉 no goals
                           -/
  cases' this with h h <;> simp [h, imp_or]
                           /-
                             🎉 no goals
                           -/


theorem Nat.mul_divisors_filter_prime_pow {a b : ℕ} (hab : a.Coprime b) :
    (a * b).divisors.filter IsPrimePow = (a.divisors ∪ b.divisors).filter IsPrimePow := by
  /-
    a b : Nat
    hab : a.Coprime b
    ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul a b).divisors) (Finset.filter IsPrim …
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      b : Nat
      hab : Nat.Coprime 0 b
      ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul 0 b).divisors) (Finset.filter IsPrim …
    -/
  · simp only [Nat.coprime_zero_left] at hab
    /-
      case inl
      b : Nat
      hab : Eq b 1
      ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul 0 b).divisors) (Finset.filter IsPrim …
    -/
    simp [hab, Finset.filter_singleton, not_isPrimePow_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    hab : a.Coprime b
    ha : Ne a 0
    ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul a b).divisors) (Finset.filter IsPrim …
  -/
  rcases eq_or_ne b 0 with (rfl | hb)
    /-
      case inr.inl
      a : Nat
      ha : Ne a 0
      hab : a.Coprime 0
      ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul a 0).divisors) (Finset.filter IsPrim …
    -/
  · simp only [Nat.coprime_zero_right] at hab
    /-
      case inr.inl
      a : Nat
      ha : Ne a 0
      hab : Eq a 1
      ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul a 0).divisors) (Finset.filter IsPrim …
    -/
    simp [hab, Finset.filter_singleton, not_isPrimePow_one]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    hab : a.Coprime b
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (Finset.filter IsPrimePow (HMul.hMul a b).divisors) (Finset.filter IsPrim …
  -/
  ext n
  simp only [ha, hb, Finset.mem_union, Finset.mem_filter, Nat.mul_eq_zero, and_true, Ne,
    and_congr_left_iff, not_false_iff, Nat.mem_divisors, or_self_iff]
  /-
    case inr.inr.h
    a b : Nat
    hab : a.Coprime b
    ha : Ne a 0
    hb : Ne b 0
    n : Nat
    ⊢ IsPrimePow n → Iff (Dvd.dvd n (HMul.hMul a b)) (Or (Dvd.dvd n a) (Dvd.dvd n  …
  -/
  apply hab.isPrimePow_dvd_mul
  /-
    🎉 no goals
  -/


lemma IsPrimePow.factorization_minFac_ne_zero {n : ℕ} (hn : IsPrimePow n) :
    n.factorization n.minFac ≠ 0 := by
  /-
    n : Nat
    hn : IsPrimePow n
    ⊢ Ne (n.factorization n.minFac) 0
  -/
  refine mt (Nat.factorization_eq_zero_iff _ _).mp ?_
  /-
    n : Nat
    hn : IsPrimePow n
    ⊢ Not (Or (Not (Nat.Prime n.minFac)) (Or (Not (Dvd.dvd n.minFac n)) (Eq n 0)))
  -/
  push_neg
  /-
    n : Nat
    hn : IsPrimePow n
    ⊢ And (Nat.Prime n.minFac) (And (Dvd.dvd n.minFac n) (Ne n 0))
  -/
  exact ⟨n.minFac_prime hn.ne_one, n.minFac_dvd, hn.ne_zero⟩
  /-
    🎉 no goals
  -/


/-- The canonical equivalence between pairs `(p, k)` with `p` a prime and `k : ℕ`
and the set of prime powers given by `(p, k) ↦ p^(k+1)`. -/
def Nat.Primes.prodNatEquiv : Nat.Primes × ℕ ≃ {n : ℕ // IsPrimePow n} where
  toFun pk :=
    ⟨pk.1 ^ (pk.2 + 1), ⟨pk.1, pk.2 + 1, prime_iff.mp pk.1.prop, pk.2.add_one_pos, rfl⟩⟩
  invFun n :=
    (⟨n.val.minFac, minFac_prime n.prop.ne_one⟩, n.val.factorization n.val.minFac - 1)
  left_inv := fun (p, k) ↦ by
    simp only [p.prop.pow_minFac k.add_one_ne_zero, Subtype.coe_eta, factorization_pow, p.prop,
      Prime.factorization, Finsupp.smul_single, smul_eq_mul, mul_one, Finsupp.single_add,
      Finsupp.coe_add, Pi.add_apply, Finsupp.single_eq_same, add_tsub_cancel_right]
  right_inv n := by
    /-
      n : Subtype fun n => IsPrimePow n
      ⊢ Eq ((fun pk => ⟨HPow.hPow (↑pk.1) (HAdd.hAdd pk.2 1), ⋯⟩) ((fun n => { fst : …
    -/
    ext1
    /-
      case a
      n : Subtype fun n => IsPrimePow n
      ⊢ Eq ↑((fun pk => ⟨HPow.hPow (↑pk.1) (HAdd.hAdd pk.2 1), ⋯⟩) ((fun n => { fst  …
    -/
    dsimp only
    /-
      case a
      n : Subtype fun n => IsPrimePow n
      ⊢ Eq (HPow.hPow (↑n).minFac (HAdd.hAdd (HSub.hSub ((↑n).factorization (↑n).min …
    -/
    rw [sub_one_add_one n.prop.factorization_minFac_ne_zero, n.prop.minFac_pow_factorization_eq]
    /-
      🎉 no goals
    -/


@[simp]
lemma Nat.Primes.prodNatEquiv_apply (p : Nat.Primes) (k : ℕ) :
    prodNatEquiv (p, k) = ⟨p ^ (k + 1), p, k + 1, prime_iff.mp p.prop, k.add_one_pos, rfl⟩ := by
  /-
    p : Nat.Primes
    k : Nat
    ⊢ Eq (Nat.Primes.prodNatEquiv { fst := p, snd := k }) ⟨HPow.hPow (↑p) (HAdd.hA …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma Nat.Primes.coe_prodNatEquiv_apply (p : Nat.Primes) (k : ℕ) :
    (prodNatEquiv (p, k) : ℕ) = p ^ (k + 1) :=
  rfl


@[simp]
lemma Nat.Primes.prodNatEquiv_symm_apply {n : ℕ} (hn : IsPrimePow n) :
    prodNatEquiv.symm ⟨n, hn⟩ =
      (⟨n.minFac, minFac_prime hn.ne_one⟩, n.factorization n.minFac - 1) :=
  rfl

