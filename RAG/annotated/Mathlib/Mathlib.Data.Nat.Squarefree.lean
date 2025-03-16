theorem squarefree_iff_nodup_primeFactorsList {n : ℕ} (h0 : n ≠ 0) :
    Squarefree n ↔ n.primeFactorsList.Nodup := by
  /-
    n : Nat
    h0 : Ne n 0
    ⊢ Iff (Squarefree n) n.primeFactorsList.Nodup
  -/
  rw [UniqueFactorizationMonoid.squarefree_iff_nodup_normalizedFactors h0, Nat.factors_eq]
  /-
    n : Nat
    h0 : Ne n 0
    ⊢ Iff (↑n.primeFactorsList).Nodup n.primeFactorsList.Nodup
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-17")]
alias squarefree_iff_nodup_factors := squarefree_iff_nodup_primeFactorsList


theorem Squarefree.nodup_primeFactorsList {n : ℕ} (hn : Squarefree n) : n.primeFactorsList.Nodup :=
  (Nat.squarefree_iff_nodup_primeFactorsList hn.ne_zero).mp hn


@[deprecated (since := "2024-07-17")]
alias Squarefree.nodup_factors := Squarefree.nodup_primeFactorsList


theorem squarefree_iff_prime_squarefree {n : ℕ} : Squarefree n ↔ ∀ x, Prime x → ¬x * x ∣ n :=
  squarefree_iff_irreducible_sq_not_dvd_of_exists_irreducible ⟨_, prime_two⟩


theorem _root_.Squarefree.natFactorization_le_one {n : ℕ} (p : ℕ) (hn : Squarefree n) :
    n.factorization p ≤ 1 := by
  /-
    n p : Nat
    hn : Squarefree n
    ⊢ LE.le (n.factorization p) 1
  -/
  rcases eq_or_ne n 0 with (rfl | hn')
    /-
      case inl
      p : Nat
      hn : Squarefree 0
      ⊢ LE.le ((Nat.factorization 0) p) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n p : Nat
    hn : Squarefree n
    hn' : Ne n 0
    ⊢ LE.le (n.factorization p) 1
  -/
  rw [squarefree_iff_emultiplicity_le_one] at hn
  /-
    case inr
    n p : Nat
    hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
    hn' : Ne n 0
    ⊢ LE.le (n.factorization p) 1
  -/
  by_cases hp : p.Prime
    /-
      case pos
      n p : Nat
      hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
      hn' : Ne n 0
      hp : Nat.Prime p
      ⊢ LE.le (n.factorization p) 1
    -/
  · have := hn p
    /-
      case pos
      n p : Nat
      hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
      hn' : Ne n 0
      hp : Nat.Prime p
      this : Or (LE.le (emultiplicity p n) 1) (IsUnit p)
      ⊢ LE.le (n.factorization p) 1
    -/
    rw [← multiplicity_eq_factorization hp hn']
    /-
      case pos
      n p : Nat
      hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
      hn' : Ne n 0
      hp : Nat.Prime p
      this : Or (LE.le (emultiplicity p n) 1) (IsUnit p)
      ⊢ LE.le (multiplicity p n) 1
    -/
    simp only [Nat.isUnit_iff, hp.ne_one, or_false] at this
    /-
      case pos
      n p : Nat
      hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
      hn' : Ne n 0
      hp : Nat.Prime p
      this : LE.le (emultiplicity p n) 1
      ⊢ LE.le (multiplicity p n) 1
    -/
    exact multiplicity_le_of_emultiplicity_le this
    /-
      🎉 no goals
    -/
    /-
      case neg
      n p : Nat
      hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
      hn' : Ne n 0
      hp : Not (Nat.Prime p)
      ⊢ LE.le (n.factorization p) 1
    -/
  · rw [factorization_eq_zero_of_non_prime _ hp]
    /-
      case neg
      n p : Nat
      hn : ∀ (x : Nat), Or (LE.le (emultiplicity x n) 1) (IsUnit x)
      hn' : Ne n 0
      hp : Not (Nat.Prime p)
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/


lemma factorization_eq_one_of_squarefree (hn : Squarefree n) (hp : p.Prime) (hpn : p ∣ n) :
    factorization n p = 1 :=
  (hn.natFactorization_le_one _).antisymm <| (hp.dvd_iff_one_le_factorization hn.ne_zero).1 hpn


theorem squarefree_of_factorization_le_one {n : ℕ} (hn : n ≠ 0) (hn' : ∀ p, n.factorization p ≤ 1) :
    Squarefree n := by
  /-
    n : Nat
    hn : Ne n 0
    hn' : ∀ (p : Nat), LE.le (n.factorization p) 1
    ⊢ Squarefree n
  -/
  rw [squarefree_iff_nodup_primeFactorsList hn, List.nodup_iff_count_le_one]
  /-
    n : Nat
    hn : Ne n 0
    hn' : ∀ (p : Nat), LE.le (n.factorization p) 1
    ⊢ ∀ (a : Nat), LE.le (List.count a n.primeFactorsList) 1
  -/
  intro a
  /-
    n : Nat
    hn : Ne n 0
    hn' : ∀ (p : Nat), LE.le (n.factorization p) 1
    a : Nat
    ⊢ LE.le (List.count a n.primeFactorsList) 1
  -/
  rw [primeFactorsList_count_eq]
  /-
    n : Nat
    hn : Ne n 0
    hn' : ∀ (p : Nat), LE.le (n.factorization p) 1
    a : Nat
    ⊢ LE.le (n.factorization a) 1
  -/
  apply hn'
  /-
    🎉 no goals
  -/


theorem squarefree_iff_factorization_le_one {n : ℕ} (hn : n ≠ 0) :
    Squarefree n ↔ ∀ p, n.factorization p ≤ 1 :=
  ⟨fun hn => hn.natFactorization_le_one, squarefree_of_factorization_le_one hn⟩


theorem Squarefree.ext_iff {n m : ℕ} (hn : Squarefree n) (hm : Squarefree m) :
    n = m ↔ ∀ p, Prime p → (p ∣ n ↔ p ∣ m) := by
  /-
    n m : Nat
    hn : Squarefree n
    hm : Squarefree m
    ⊢ Iff (Eq n m) (∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m))
  -/
  refine ⟨by rintro rfl; simp, fun h => eq_of_factorization_eq hn.ne_zero hm.ne_zero fun p => ?_⟩
  /-
    n m : Nat
    hn : Squarefree n
    hm : Squarefree m
    h : ∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m)
    p : Nat
    ⊢ Eq (n.factorization p) (m.factorization p)
  -/
  by_cases hp : p.Prime
    /-
      case pos
      n m : Nat
      hn : Squarefree n
      hm : Squarefree m
      h : ∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m)
      p : Nat
      hp : Nat.Prime p
      ⊢ Eq (n.factorization p) (m.factorization p)
    -/
  · have h₁ := h _ hp
    rw [← not_iff_not, hp.dvd_iff_one_le_factorization hn.ne_zero, not_le, lt_one_iff,
      hp.dvd_iff_one_le_factorization hm.ne_zero, not_le, lt_one_iff] at h₁
    /-
      case pos
      n m : Nat
      hn : Squarefree n
      hm : Squarefree m
      h : ∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m)
      p : Nat
      hp : Nat.Prime p
      h₁ : Iff (Eq (n.factorization p) 0) (Eq (m.factorization p) 0)
      ⊢ Eq (n.factorization p) (m.factorization p)
    -/
    have h₂ := hn.natFactorization_le_one p
    /-
      case pos
      n m : Nat
      hn : Squarefree n
      hm : Squarefree m
      h : ∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m)
      p : Nat
      hp : Nat.Prime p
      h₁ : Iff (Eq (n.factorization p) 0) (Eq (m.factorization p) 0)
      h₂ : LE.le (n.factorization p) 1
      ⊢ Eq (n.factorization p) (m.factorization p)
    -/
    have h₃ := hm.natFactorization_le_one p
    /-
      case pos
      n m : Nat
      hn : Squarefree n
      hm : Squarefree m
      h : ∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m)
      p : Nat
      hp : Nat.Prime p
      h₁ : Iff (Eq (n.factorization p) 0) (Eq (m.factorization p) 0)
      h₂ : LE.le (n.factorization p) 1
      h₃ : LE.le (m.factorization p) 1
      ⊢ Eq (n.factorization p) (m.factorization p)
    -/
    omega
    /-
      🎉 no goals
    -/
  /-
    case neg
    n m : Nat
    hn : Squarefree n
    hm : Squarefree m
    h : ∀ (p : Nat), Nat.Prime p → Iff (Dvd.dvd p n) (Dvd.dvd p m)
    p : Nat
    hp : Not (Nat.Prime p)
    ⊢ Eq (n.factorization p) (m.factorization p)
  -/
  rw [factorization_eq_zero_of_non_prime _ hp, factorization_eq_zero_of_non_prime _ hp]
  /-
    🎉 no goals
  -/


theorem squarefree_pow_iff {n k : ℕ} (hn : n ≠ 1) (hk : k ≠ 0) :
    Squarefree (n ^ k) ↔ Squarefree n ∧ k = 1 := by
  /-
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    ⊢ Iff (Squarefree (HPow.hPow n k)) (And (Squarefree n) (Eq k 1))
  -/
  refine ⟨fun h => ?_, by rintro ⟨hn, rfl⟩; simpa⟩
  /-
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    h : Squarefree (HPow.hPow n k)
    ⊢ And (Squarefree n) (Eq k 1)
  -/
  rcases eq_or_ne n 0 with (rfl | -)
    /-
      case inl
      k : Nat
      hk : Ne k 0
      hn : Ne 0 1
      h : Squarefree (HPow.hPow 0 k)
      ⊢ And (Squarefree 0) (Eq k 1)
    -/
  · simp [zero_pow hk] at h
    /-
      🎉 no goals
    -/
  /-
    case inr
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    h : Squarefree (HPow.hPow n k)
    ⊢ And (Squarefree n) (Eq k 1)
  -/
  refine ⟨h.squarefree_of_dvd (dvd_pow_self _ hk), by_contradiction fun h₁ => ?_⟩
  /-
    case inr
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    h : Squarefree (HPow.hPow n k)
    h₁ : Not (Eq k 1)
    ⊢ False
  -/
  have : 2 ≤ k := k.two_le_iff.mpr ⟨hk, h₁⟩
  /-
    case inr
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    h : Squarefree (HPow.hPow n k)
    h₁ : Not (Eq k 1)
    this : LE.le 2 k
    ⊢ False
  -/
  apply hn (Nat.isUnit_iff.1 (h _ _))
  /-
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    h : Squarefree (HPow.hPow n k)
    h₁ : Not (Eq k 1)
    this : LE.le 2 k
    ⊢ Dvd.dvd (HMul.hMul n n) (HPow.hPow n k)
  -/
  rw [← sq]
  /-
    n k : Nat
    hn : Ne n 1
    hk : Ne k 0
    h : Squarefree (HPow.hPow n k)
    h₁ : Not (Eq k 1)
    this : LE.le 2 k
    ⊢ Dvd.dvd (HPow.hPow n 2) (HPow.hPow n k)
  -/
  exact pow_dvd_pow _ this
  /-
    🎉 no goals
  -/


theorem squarefree_and_prime_pow_iff_prime {n : ℕ} : Squarefree n ∧ IsPrimePow n ↔ Prime n := by
  /-
    n : Nat
    ⊢ Iff (And (Squarefree n) (IsPrimePow n)) (Nat.Prime n)
  -/
  refine ⟨?_, fun hn => ⟨hn.squarefree, hn.isPrimePow⟩⟩
  /-
    n : Nat
    ⊢ And (Squarefree n) (IsPrimePow n) → Nat.Prime n
  -/
  rw [isPrimePow_nat_iff]
  /-
    n : Nat
    ⊢ And (Squarefree n) (Exists fun p => Exists fun k => And (Nat.Prime p) (And ( …
  -/
  rintro ⟨h, p, k, hp, hk, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    h : Squarefree (HPow.hPow p k)
    ⊢ Nat.Prime (HPow.hPow p k)
  -/
  rw [squarefree_pow_iff hp.ne_one hk.ne'] at h
  /-
    case intro.intro.intro.intro.intro
    p k : Nat
    hp : Nat.Prime p
    hk : LT.lt 0 k
    h : And (Squarefree p) (Eq k 1)
    ⊢ Nat.Prime (HPow.hPow p k)
  -/
  rwa [h.2, pow_one]
  /-
    🎉 no goals
  -/


/-- Assuming that `n` has no factors less than `k`, returns the smallest prime `p` such that
  `p^2 ∣ n`. -/
def minSqFacAux : ℕ → ℕ → Option ℕ
  | n, k =>
    if h : n < k * k then none
    else
      have : Nat.sqrt n - k < Nat.sqrt n + 2 - k := by
        /-
          s : Finset Nat
          m n✝ p n k : Nat
          h : Not (LT.lt n (HMul.hMul k k))
          ⊢ LT.lt (HSub.hSub n.sqrt k) (HSub.hSub (HAdd.hAdd n.sqrt 2) k)
        -/
        exact Nat.minFac_lemma n k h
        /-
          🎉 no goals
        -/
      if k ∣ n then
        let n' := n / k
        have : Nat.sqrt n' - k < Nat.sqrt n + 2 - k :=
        lt_of_le_of_lt (Nat.sub_le_sub_right (Nat.sqrt_le_sqrt <| Nat.div_le_self _ _) k) this
        if k ∣ n' then some k else minSqFacAux n' (k + 2)
      else minSqFacAux n (k + 2)
termination_by n k => sqrt n + 2 - k


/-- Returns the smallest prime factor `p` of `n` such that `p^2 ∣ n`, or `none` if there is no
  such `p` (that is, `n` is squarefree). See also `Nat.squarefree_iff_minSqFac`. -/
def minSqFac (n : ℕ) : Option ℕ :=
  if 2 ∣ n then
    let n' := n / 2
    if 2 ∣ n' then some 2 else minSqFacAux n' 3
  else minSqFacAux n 3


/-- The correctness property of the return value of `minSqFac`.
  * If `none`, then `n` is squarefree;
  * If `some d`, then `d` is a minimal square factor of `n` -/
def MinSqFacProp (n : ℕ) : Option ℕ → Prop
  | none => Squarefree n
  | some d => Prime d ∧ d * d ∣ n ∧ ∀ p, Prime p → p * p ∣ n → d ≤ p


theorem minSqFacProp_div (n) {k} (pk : Prime k) (dk : k ∣ n) (dkk : ¬k * k ∣ n) {o}
    (H : MinSqFacProp (n / k) o) : MinSqFacProp n o := by
  have : ∀ p, Prime p → p * p ∣ n → k * (p * p) ∣ n := fun p pp dp =>
    have :=
      (coprime_primes pk pp).2 fun e => by
        subst e
        contradiction
    (coprime_mul_iff_right.2 ⟨this, this⟩).mul_dvd_of_dvd_of_dvd dk dp
  /-
    n k : Nat
    pk : Nat.Prime k
    dk : Dvd.dvd k n
    dkk : Not (Dvd.dvd (HMul.hMul k k) n)
    o : Option Nat
    H : (HDiv.hDiv n k).MinSqFacProp o
    this : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) n → Dvd.dvd (HMul.hM …
    ⊢ n.MinSqFacProp o
  -/
  cases' o with d
    /-
      case none
      n k : Nat
      pk : Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd (HMul.hMul k k) n)
      this : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) n → Dvd.dvd (HMul.hM …
      H : (HDiv.hDiv n k).MinSqFacProp Option.none
      ⊢ n.MinSqFacProp Option.none
    -/
  · rw [MinSqFacProp, squarefree_iff_prime_squarefree] at H ⊢
    /-
      case none
      n k : Nat
      pk : Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd (HMul.hMul k k) n)
      this : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) n → Dvd.dvd (HMul.hM …
      H : ∀ (x : Nat), Nat.Prime x → Not (Dvd.dvd (HMul.hMul x x) (HDiv.hDiv n k))
      ⊢ ∀ (x : Nat), Nat.Prime x → Not (Dvd.dvd (HMul.hMul x x) n)
    -/
    exact fun p pp dp => H p pp ((dvd_div_iff_mul_dvd dk).2 (this _ pp dp))
    /-
      🎉 no goals
    -/
    /-
      case some
      n k : Nat
      pk : Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd (HMul.hMul k k) n)
      this : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) n → Dvd.dvd (HMul.hM …
      d : Nat
      H : (HDiv.hDiv n k).MinSqFacProp (Option.some d)
      ⊢ n.MinSqFacProp (Option.some d)
    -/
  · obtain ⟨H1, H2, H3⟩ := H
    /-
      case some.intro.intro
      n k : Nat
      pk : Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd (HMul.hMul k k) n)
      this : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) n → Dvd.dvd (HMul.hM …
      d : Nat
      H1 : Nat.Prime d
      H2 : Dvd.dvd (HMul.hMul d d) (HDiv.hDiv n k)
      H3 : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) (HDiv.hDiv n k) → LE.l …
      ⊢ n.MinSqFacProp (Option.some d)
    -/
    simp only [dvd_div_iff_mul_dvd dk] at H2 H3
    /-
      case some.intro.intro
      n k : Nat
      pk : Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd (HMul.hMul k k) n)
      this : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul p p) n → Dvd.dvd (HMul.hM …
      d : Nat
      H1 : Nat.Prime d
      H2 : Dvd.dvd (HMul.hMul k (HMul.hMul d d)) n
      H3 : ∀ (p : Nat), Nat.Prime p → Dvd.dvd (HMul.hMul k (HMul.hMul p p)) n → LE.l …
      ⊢ n.MinSqFacProp (Option.some d)
    -/
    exact ⟨H1, dvd_trans (dvd_mul_left _ _) H2, fun p pp dp => H3 _ pp (this _ pp dp)⟩
    /-
      🎉 no goals
    -/


theorem minSqFacAux_has_prop {n : ℕ} (k) (n0 : 0 < n) (i) (e : k = 2 * i + 3)
    (ih : ∀ m, Prime m → m ∣ n → k ≤ m) : MinSqFacProp n (minSqFacAux n k) := by
  /-
    n k : Nat
    n0 : LT.lt 0 n
    i : Nat
    e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
    ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
    ⊢ n.MinSqFacProp (n.minSqFacAux k)
  -/
  rw [minSqFacAux]
  /-
    n k : Nat
    n0 : LT.lt 0 n
    i : Nat
    e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
    ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
    ⊢ n.MinSqFacProp
        (dite (LT.lt n (HMul.hMul k k)) (fun h => Option.none) fun h =>
          letFun ⋯ fun this =>
            ite (Dvd.dvd k n)
              (let n' := HDiv.hDiv n k;
              letFun ⋯ fun this => ite (Dvd.dvd k n') (Option.some k) (n'.minSqFac …
              (n.minSqFacAux (HAdd.hAdd k 2)))
  -/
  by_cases h : n < k * k <;> simp only [h, ↓reduceDIte]
    /-
      case pos
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : LT.lt n (HMul.hMul k k)
      ⊢ n.MinSqFacProp Option.none
    -/
  · refine squarefree_iff_prime_squarefree.2 fun p pp d => ?_
    /-
      case pos
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : LT.lt n (HMul.hMul k k)
      p : Nat
      pp : Nat.Prime p
      d : Dvd.dvd (HMul.hMul p p) n
      ⊢ False
    -/
    have := ih p pp (dvd_trans ⟨_, rfl⟩ d)
    /-
      case pos
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : LT.lt n (HMul.hMul k k)
      p : Nat
      pp : Nat.Prime p
      d : Dvd.dvd (HMul.hMul p p) n
      this : LE.le k p
      ⊢ False
    -/
    have := Nat.mul_le_mul this this
    /-
      case pos
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : LT.lt n (HMul.hMul k k)
      p : Nat
      pp : Nat.Prime p
      d : Dvd.dvd (HMul.hMul p p) n
      this✝ : LE.le k p
      this : LE.le (HMul.hMul k k) (HMul.hMul p p)
      ⊢ False
    -/
    exact not_le_of_lt h (le_trans this (le_of_dvd n0 d))
    /-
      🎉 no goals
    -/
  /-
    case neg
    n k : Nat
    n0 : LT.lt 0 n
    i : Nat
    e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
    ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
    h : Not (LT.lt n (HMul.hMul k k))
    ⊢ n.MinSqFacProp (ite (Dvd.dvd k n) (ite (Dvd.dvd k (HDiv.hDiv n k)) (Option.s …
  -/
  have k2 : 2 ≤ k := by omega
  /-
    case neg
    n k : Nat
    n0 : LT.lt 0 n
    i : Nat
    e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
    ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
    h : Not (LT.lt n (HMul.hMul k k))
    k2 : LE.le 2 k
    ⊢ n.MinSqFacProp (ite (Dvd.dvd k n) (ite (Dvd.dvd k (HDiv.hDiv n k)) (Option.s …
  -/
  have k0 : 0 < k := lt_of_lt_of_le (by decide) k2
  have IH : ∀ n', n' ∣ n → ¬k ∣ n' → MinSqFacProp n' (n'.minSqFacAux (k + 2)) := by
    intro n' nd' nk
    have hn' := le_of_dvd n0 nd'
    refine
      have : Nat.sqrt n' - k < Nat.sqrt n + 2 - k :=
        lt_of_le_of_lt (Nat.sub_le_sub_right (Nat.sqrt_le_sqrt hn') _) (Nat.minFac_lemma n k h)
      @minSqFacAux_has_prop n' (k + 2) (pos_of_dvd_of_pos nd' n0) (i + 1)
        (by simp [e, left_distrib]) fun m m2 d => ?_
    rcases Nat.eq_or_lt_of_le (ih m m2 (dvd_trans d nd')) with me | ml
    · subst me
      contradiction
    apply (Nat.eq_or_lt_of_le ml).resolve_left
    intro me
    rw [← me, e] at d
    change 2 * (i + 2) ∣ n' at d
    have := ih _ prime_two (dvd_trans (dvd_of_mul_right_dvd d) nd')
    rw [e] at this
    exact absurd this (by omega)
  have pk : k ∣ n → Prime k := by
    refine fun dk => prime_def_minFac.2 ⟨k2, le_antisymm (minFac_le k0) ?_⟩
    exact ih _ (minFac_prime (ne_of_gt k2)) (dvd_trans (minFac_dvd _) dk)
  /-
    case neg
    n k : Nat
    n0 : LT.lt 0 n
    i : Nat
    e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
    ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
    h : Not (LT.lt n (HMul.hMul k k))
    k2 : LE.le 2 k
    k0 : LT.lt 0 k
    IH : ∀ (n' : Nat), Dvd.dvd n' n → Not (Dvd.dvd k n') → n'.MinSqFacProp (n'.min …
    pk : Dvd.dvd k n → Nat.Prime k
    ⊢ n.MinSqFacProp (ite (Dvd.dvd k n) (ite (Dvd.dvd k (HDiv.hDiv n k)) (Option.s …
  -/
  split_ifs with dk dkk
    /-
      case pos
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : Not (LT.lt n (HMul.hMul k k))
      k2 : LE.le 2 k
      k0 : LT.lt 0 k
      IH : ∀ (n' : Nat), Dvd.dvd n' n → Not (Dvd.dvd k n') → n'.MinSqFacProp (n'.min …
      pk : Dvd.dvd k n → Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Dvd.dvd k (HDiv.hDiv n k)
      ⊢ n.MinSqFacProp (Option.some k)
    -/
  · exact ⟨pk dk, (Nat.dvd_div_iff_mul_dvd dk).1 dkk, fun p pp d => ih p pp (dvd_trans ⟨_, rfl⟩ d)⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : Not (LT.lt n (HMul.hMul k k))
      k2 : LE.le 2 k
      k0 : LT.lt 0 k
      IH : ∀ (n' : Nat), Dvd.dvd n' n → Not (Dvd.dvd k n') → n'.MinSqFacProp (n'.min …
      pk : Dvd.dvd k n → Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd k (HDiv.hDiv n k))
      ⊢ n.MinSqFacProp ((HDiv.hDiv n k).minSqFacAux (HAdd.hAdd k 2))
    -/
  · specialize IH (n / k) (div_dvd_of_dvd dk) dkk
    /-
      case neg
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : Not (LT.lt n (HMul.hMul k k))
      k2 : LE.le 2 k
      k0 : LT.lt 0 k
      pk : Dvd.dvd k n → Nat.Prime k
      dk : Dvd.dvd k n
      dkk : Not (Dvd.dvd k (HDiv.hDiv n k))
      IH : (HDiv.hDiv n k).MinSqFacProp ((HDiv.hDiv n k).minSqFacAux (HAdd.hAdd k 2))
      ⊢ n.MinSqFacProp ((HDiv.hDiv n k).minSqFacAux (HAdd.hAdd k 2))
    -/
    exact minSqFacProp_div _ (pk dk) dk (mt (Nat.dvd_div_iff_mul_dvd dk).2 dkk) IH
    /-
      🎉 no goals
    -/
    /-
      case neg
      n k : Nat
      n0 : LT.lt 0 n
      i : Nat
      e : Eq k (HAdd.hAdd (HMul.hMul 2 i) 3)
      ih : ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le k m
      h : Not (LT.lt n (HMul.hMul k k))
      k2 : LE.le 2 k
      k0 : LT.lt 0 k
      IH : ∀ (n' : Nat), Dvd.dvd n' n → Not (Dvd.dvd k n') → n'.MinSqFacProp (n'.min …
      pk : Dvd.dvd k n → Nat.Prime k
      dk : Not (Dvd.dvd k n)
      ⊢ n.MinSqFacProp (n.minSqFacAux (HAdd.hAdd k 2))
    -/
  · exact IH n (dvd_refl _) dk
    /-
      🎉 no goals
    -/
termination_by n.sqrt + 2 - k


theorem minSqFac_has_prop (n : ℕ) : MinSqFacProp n (minSqFac n) := by
  /-
    n : Nat
    ⊢ n.MinSqFacProp n.minSqFac
  -/
  dsimp only [minSqFac]; split_ifs with d2 d4
    /-
      case pos
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Dvd.dvd 2 (HDiv.hDiv n 2)
      ⊢ n.MinSqFacProp (Option.some 2)
    -/
  · exact ⟨prime_two, (dvd_div_iff_mul_dvd d2).1 d4, fun p pp _ => pp.two_le⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
      ⊢ n.MinSqFacProp ((HDiv.hDiv n 2).minSqFacAux 3)
    -/
  · rcases Nat.eq_zero_or_pos n with n0 | n0
      /-
        case neg.inl
        n : Nat
        d2 : Dvd.dvd 2 n
        d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
        n0 : Eq n 0
        ⊢ n.MinSqFacProp ((HDiv.hDiv n 2).minSqFacAux 3)
      -/
    · subst n0
      /-
        case neg.inl
        d2 : Dvd.dvd 2 0
        d4 : Not (Dvd.dvd 2 (0 / 2))
        ⊢ Nat.MinSqFacProp 0 ((0 / 2).minSqFacAux 3)
      -/
      cases d4 (by decide)
      /-
        🎉 no goals
      -/
    /-
      case neg.inr
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
      n0 : GT.gt n 0
      ⊢ n.MinSqFacProp ((HDiv.hDiv n 2).minSqFacAux 3)
    -/
    refine minSqFacProp_div _ prime_two d2 (mt (dvd_div_iff_mul_dvd d2).2 d4) ?_
    /-
      case neg.inr
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
      n0 : GT.gt n 0
      ⊢ (HDiv.hDiv n 2).MinSqFacProp ((HDiv.hDiv n 2).minSqFacAux 3)
    -/
    refine minSqFacAux_has_prop 3 (Nat.div_pos (le_of_dvd n0 d2) (by decide)) 0 rfl ?_
    /-
      case neg.inr
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
      n0 : GT.gt n 0
      ⊢ ∀ (m : Nat), Nat.Prime m → Dvd.dvd m (HDiv.hDiv n 2) → LE.le 3 m
    -/
    refine fun p pp dp => succ_le_of_lt (lt_of_le_of_ne pp.two_le ?_)
    /-
      case neg.inr
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
      n0 : GT.gt n 0
      p : Nat
      pp : Nat.Prime p
      dp : Dvd.dvd p (HDiv.hDiv n 2)
      ⊢ Ne 2 p
    -/
    rintro rfl
    /-
      case neg.inr
      n : Nat
      d2 : Dvd.dvd 2 n
      d4 : Not (Dvd.dvd 2 (HDiv.hDiv n 2))
      n0 : GT.gt n 0
      pp : Nat.Prime 2
      dp : Dvd.dvd 2 (HDiv.hDiv n 2)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      d2 : Not (Dvd.dvd 2 n)
      ⊢ n.MinSqFacProp (n.minSqFacAux 3)
    -/
  · rcases Nat.eq_zero_or_pos n with n0 | n0
      /-
        case neg.inl
        n : Nat
        d2 : Not (Dvd.dvd 2 n)
        n0 : Eq n 0
        ⊢ n.MinSqFacProp (n.minSqFacAux 3)
      -/
    · subst n0
      /-
        case neg.inl
        d2 : Not (Dvd.dvd 2 0)
        ⊢ Nat.MinSqFacProp 0 (Nat.minSqFacAux 0 3)
      -/
      cases d2 (by decide)
      /-
        🎉 no goals
      -/
    /-
      case neg.inr
      n : Nat
      d2 : Not (Dvd.dvd 2 n)
      n0 : GT.gt n 0
      ⊢ n.MinSqFacProp (n.minSqFacAux 3)
    -/
    refine minSqFacAux_has_prop _ n0 0 rfl ?_
    /-
      case neg.inr
      n : Nat
      d2 : Not (Dvd.dvd 2 n)
      n0 : GT.gt n 0
      ⊢ ∀ (m : Nat), Nat.Prime m → Dvd.dvd m n → LE.le 3 m
    -/
    refine fun p pp dp => succ_le_of_lt (lt_of_le_of_ne pp.two_le ?_)
    /-
      case neg.inr
      n : Nat
      d2 : Not (Dvd.dvd 2 n)
      n0 : GT.gt n 0
      p : Nat
      pp : Nat.Prime p
      dp : Dvd.dvd p n
      ⊢ Ne 2 p
    -/
    rintro rfl
    /-
      case neg.inr
      n : Nat
      d2 : Not (Dvd.dvd 2 n)
      n0 : GT.gt n 0
      pp : Nat.Prime 2
      dp : Dvd.dvd 2 n
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem minSqFac_prime {n d : ℕ} (h : n.minSqFac = some d) : Prime d := by
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    ⊢ Nat.Prime d
  -/
  have := minSqFac_has_prop n
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    this : n.MinSqFacProp n.minSqFac
    ⊢ Nat.Prime d
  -/
  rw [h] at this
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    this : n.MinSqFacProp (Option.some d)
    ⊢ Nat.Prime d
  -/
  exact this.1
  /-
    🎉 no goals
  -/


theorem minSqFac_dvd {n d : ℕ} (h : n.minSqFac = some d) : d * d ∣ n := by
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    ⊢ Dvd.dvd (HMul.hMul d d) n
  -/
  have := minSqFac_has_prop n
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    this : n.MinSqFacProp n.minSqFac
    ⊢ Dvd.dvd (HMul.hMul d d) n
  -/
  rw [h] at this
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    this : n.MinSqFacProp (Option.some d)
    ⊢ Dvd.dvd (HMul.hMul d d) n
  -/
  exact this.2.1
  /-
    🎉 no goals
  -/


theorem minSqFac_le_of_dvd {n d : ℕ} (h : n.minSqFac = some d) {m} (m2 : 2 ≤ m) (md : m * m ∣ n) :
    d ≤ m := by
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    m : Nat
    m2 : LE.le 2 m
    md : Dvd.dvd (HMul.hMul m m) n
    ⊢ LE.le d m
  -/
  have := minSqFac_has_prop n; rw [h] at this
  /-
    n d : Nat
    h : Eq n.minSqFac (Option.some d)
    m : Nat
    m2 : LE.le 2 m
    md : Dvd.dvd (HMul.hMul m m) n
    this : n.MinSqFacProp (Option.some d)
    ⊢ LE.le d m
  -/
  have fd := minFac_dvd m
  exact
    le_trans (this.2.2 _ (minFac_prime <| ne_of_gt m2) (dvd_trans (mul_dvd_mul fd fd) md))
      (minFac_le <| lt_of_lt_of_le (by decide) m2)


theorem squarefree_iff_minSqFac {n : ℕ} : Squarefree n ↔ n.minSqFac = none := by
  /-
    n : Nat
    ⊢ Iff (Squarefree n) (Eq n.minSqFac Option.none)
  -/
  have := minSqFac_has_prop n
  /-
    n : Nat
    this : n.MinSqFacProp n.minSqFac
    ⊢ Iff (Squarefree n) (Eq n.minSqFac Option.none)
  -/
  constructor <;> intro H
    /-
      case mp
      n : Nat
      this : n.MinSqFacProp n.minSqFac
      H : Squarefree n
      ⊢ Eq n.minSqFac Option.none
    -/
  · cases' e : n.minSqFac with d
      /-
        case mp.none
        n : Nat
        this : n.MinSqFacProp n.minSqFac
        H : Squarefree n
        e : Eq n.minSqFac Option.none
        ⊢ Eq Option.none Option.none
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case mp.some
      n : Nat
      this : n.MinSqFacProp n.minSqFac
      H : Squarefree n
      d : Nat
      e : Eq n.minSqFac (Option.some d)
      ⊢ Eq (Option.some d) Option.none
    -/
    rw [e] at this
    /-
      case mp.some
      n : Nat
      H : Squarefree n
      d : Nat
      this : n.MinSqFacProp (Option.some d)
      e : Eq n.minSqFac (Option.some d)
      ⊢ Eq (Option.some d) Option.none
    -/
    cases squarefree_iff_prime_squarefree.1 H _ this.1 this.2.1
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      this : n.MinSqFacProp n.minSqFac
      H : Eq n.minSqFac Option.none
      ⊢ Squarefree n
    -/
  · rwa [H] at this
    /-
      🎉 no goals
    -/


instance : DecidablePred (Squarefree : ℕ → Prop) := fun _ =>
  decidable_of_iff' _ squarefree_iff_minSqFac


theorem squarefree_two : Squarefree 2 := by
  /-
    ⊢ Squarefree 2
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  rw [squarefree_iff_nodup_primeFactorsList] <;> simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem divisors_filter_squarefree_of_squarefree {n : ℕ} (hn : Squarefree n) :
    {d ∈ n.divisors | Squarefree d} = n.divisors :=
  Finset.ext fun d => ⟨@Finset.filter_subset _ _ _ _ d, fun hd =>
    Finset.mem_filter.mpr ⟨hd, hn.squarefree_of_dvd (Nat.dvd_of_mem_divisors hd) ⟩⟩


theorem divisors_filter_squarefree {n : ℕ} (h0 : n ≠ 0) :
    {d ∈ n.divisors | Squarefree d}.val =
      (UniqueFactorizationMonoid.normalizedFactors n).toFinset.powerset.val.map fun x =>
        x.val.prod := by
  /-
    n : Nat
    h0 : Ne n 0
    ⊢ Eq (Finset.filter (fun d => Squarefree d) n.divisors).val (Multiset.map (fun …
  -/
  rw [(Finset.nodup _).ext ((Finset.nodup _).map_on _)]
    /-
      n : Nat
      h0 : Ne n 0
      ⊢ ∀ (a : Nat), Iff (Membership.mem (Finset.filter (fun d => Squarefree d) n.di …
    -/
  · intro a
    simp only [Multiset.mem_filter, id, Multiset.mem_map, Finset.filter_val, ← Finset.mem_def,
      mem_divisors]
    /-
      n : Nat
      h0 : Ne n 0
      a : Nat
      ⊢ Iff (And (And (Dvd.dvd a n) (Ne n 0)) (Squarefree a)) (Exists fun a_1 => And …
    -/
    constructor
      /-
        case mp
        n : Nat
        h0 : Ne n 0
        a : Nat
        ⊢ And (And (Dvd.dvd a n) (Ne n 0)) (Squarefree a) → Exists fun a_2 => And (Mem …
      -/
    · rintro ⟨⟨an, h0⟩, hsq⟩
      /-
        case mp.intro.intro
        n : Nat
        h0✝ : Ne n 0
        a : Nat
        hsq : Squarefree a
        an : Dvd.dvd a n
        h0 : Ne n 0
        ⊢ Exists fun a_1 => And (Membership.mem (UniqueFactorizationMonoid.normalizedF …
      -/
      use (UniqueFactorizationMonoid.normalizedFactors a).toFinset
      /-
        case h
        n : Nat
        h0✝ : Ne n 0
        a : Nat
        hsq : Squarefree a
        an : Dvd.dvd a n
        h0 : Ne n 0
        ⊢ And (Membership.mem (UniqueFactorizationMonoid.normalizedFactors n).toFinset …
      -/
      simp only [id, Finset.mem_powerset]
      /-
        case h
        n : Nat
        h0✝ : Ne n 0
        a : Nat
        hsq : Squarefree a
        an : Dvd.dvd a n
        h0 : Ne n 0
        ⊢ And (HasSubset.Subset (UniqueFactorizationMonoid.normalizedFactors a).toFins …
      -/
      rcases an with ⟨b, rfl⟩
      /-
        case h.intro
        a : Nat
        hsq : Squarefree a
        b : Nat
        h0✝ h0 : Ne (HMul.hMul a b) 0
        ⊢ And (HasSubset.Subset (UniqueFactorizationMonoid.normalizedFactors a).toFins …
      -/
      rw [mul_ne_zero_iff] at h0
      /-
        case h.intro
        a : Nat
        hsq : Squarefree a
        b : Nat
        h0✝ : Ne (HMul.hMul a b) 0
        h0 : And (Ne a 0) (Ne b 0)
        ⊢ And (HasSubset.Subset (UniqueFactorizationMonoid.normalizedFactors a).toFins …
      -/
      rw [UniqueFactorizationMonoid.squarefree_iff_nodup_normalizedFactors h0.1] at hsq
      rw [Multiset.toFinset_subset, Multiset.toFinset_val, hsq.dedup, ← associated_iff_eq,
        normalizedFactors_mul h0.1 h0.2]
      /-
        case h.intro
        a : Nat
        hsq : (UniqueFactorizationMonoid.normalizedFactors a).Nodup
        b : Nat
        h0✝ : Ne (HMul.hMul a b) 0
        h0 : And (Ne a 0) (Ne b 0)
        ⊢ And (HasSubset.Subset (UniqueFactorizationMonoid.normalizedFactors a) (HAdd. …
      -/
      exact ⟨Multiset.subset_of_le (Multiset.le_add_right _ _), prod_normalizedFactors h0.1⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        n : Nat
        h0 : Ne n 0
        a : Nat
        ⊢ (Exists fun a_1 => And (Membership.mem (UniqueFactorizationMonoid.normalized …
      -/
    · rintro ⟨s, hs, rfl⟩
      /-
        case mpr.intro.intro
        n : Nat
        h0 : Ne n 0
        s : Finset Nat
        hs : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n).toFinset.p …
        ⊢ And (And (Dvd.dvd s.val.prod n) (Ne n 0)) (Squarefree s.val.prod)
      -/
      rw [Finset.mem_powerset, ← Finset.val_le_iff, Multiset.toFinset_val] at hs
      have hs0 : s.val.prod ≠ 0 := by
        rw [Ne, Multiset.prod_eq_zero_iff]
        intro con
        apply
          not_irreducible_zero
            (irreducible_of_normalized_factor 0 (Multiset.mem_dedup.1 (Multiset.mem_of_le hs con)))
      /-
        case mpr.intro.intro
        n : Nat
        h0 : Ne n 0
        s : Finset Nat
        hs : LE.le s.val (UniqueFactorizationMonoid.normalizedFactors n).dedup
        hs0 : Ne s.val.prod 0
        ⊢ And (And (Dvd.dvd s.val.prod n) (Ne n 0)) (Squarefree s.val.prod)
      -/
      rw [(prod_normalizedFactors h0).symm.dvd_iff_dvd_right]
      /-
        case mpr.intro.intro
        n : Nat
        h0 : Ne n 0
        s : Finset Nat
        hs : LE.le s.val (UniqueFactorizationMonoid.normalizedFactors n).dedup
        hs0 : Ne s.val.prod 0
        ⊢ And (And (Dvd.dvd s.val.prod (UniqueFactorizationMonoid.normalizedFactors n) …
      -/
      refine ⟨⟨Multiset.prod_dvd_prod_of_le (le_trans hs (Multiset.dedup_le _)), h0⟩, ?_⟩
      have h :=
        UniqueFactorizationMonoid.factors_unique irreducible_of_normalized_factor
          (fun x hx =>
            irreducible_of_normalized_factor x
              (Multiset.mem_of_le (le_trans hs (Multiset.dedup_le _)) hx))
          (prod_normalizedFactors hs0)
      /-
        case mpr.intro.intro
        n : Nat
        h0 : Ne n 0
        s : Finset Nat
        hs : LE.le s.val (UniqueFactorizationMonoid.normalizedFactors n).dedup
        hs0 : Ne s.val.prod 0
        h : Multiset.Rel Associated (UniqueFactorizationMonoid.normalizedFactors s.val …
        ⊢ Squarefree s.val.prod
      -/
      rw [associated_eq_eq, Multiset.rel_eq] at h
      /-
        case mpr.intro.intro
        n : Nat
        h0 : Ne n 0
        s : Finset Nat
        hs : LE.le s.val (UniqueFactorizationMonoid.normalizedFactors n).dedup
        hs0 : Ne s.val.prod 0
        h : Eq (UniqueFactorizationMonoid.normalizedFactors s.val.prod) s.val
        ⊢ Squarefree s.val.prod
      -/
      rw [UniqueFactorizationMonoid.squarefree_iff_nodup_normalizedFactors hs0, h]
      /-
        case mpr.intro.intro
        n : Nat
        h0 : Ne n 0
        s : Finset Nat
        hs : LE.le s.val (UniqueFactorizationMonoid.normalizedFactors n).dedup
        hs0 : Ne s.val.prod 0
        h : Eq (UniqueFactorizationMonoid.normalizedFactors s.val.prod) s.val
        ⊢ s.val.Nodup
      -/
      apply s.nodup
      /-
        🎉 no goals
      -/
    /-
      n : Nat
      h0 : Ne n 0
      ⊢ ∀ (x : Finset Nat), Membership.mem (UniqueFactorizationMonoid.normalizedFact …
    -/
  · intro x hx y hy h
    /-
      n : Nat
      h0 : Ne n 0
      x : Finset Nat
      hx : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n).toFinset.p …
      y : Finset Nat
      hy : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n).toFinset.p …
      h : Eq x.val.prod y.val.prod
      ⊢ Eq x y
    -/
    rw [← Finset.val_inj, ← Multiset.rel_eq, ← associated_eq_eq]
    /-
      n : Nat
      h0 : Ne n 0
      x : Finset Nat
      hx : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n).toFinset.p …
      y : Finset Nat
      hy : Membership.mem (UniqueFactorizationMonoid.normalizedFactors n).toFinset.p …
      h : Eq x.val.prod y.val.prod
      ⊢ Multiset.Rel (fun x1 x2 => Associated x1 x2) x.val y.val
    -/
    rw [← Finset.mem_def, Finset.mem_powerset] at hx hy
    /-
      n : Nat
      h0 : Ne n 0
      x : Finset Nat
      hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
      y : Finset Nat
      hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
      h : Eq x.val.prod y.val.prod
      ⊢ Multiset.Rel (fun x1 x2 => Associated x1 x2) x.val y.val
    -/
    apply UniqueFactorizationMonoid.factors_unique _ _ (associated_iff_eq.2 h)
      /-
        n : Nat
        h0 : Ne n 0
        x : Finset Nat
        hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        y : Finset Nat
        hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        h : Eq x.val.prod y.val.prod
        ⊢ ∀ (x_1 : Nat), Membership.mem x.val x_1 → Irreducible x_1
      -/
    · intro z hz
      /-
        n : Nat
        h0 : Ne n 0
        x : Finset Nat
        hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        y : Finset Nat
        hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        h : Eq x.val.prod y.val.prod
        z : Nat
        hz : Membership.mem x.val z
        ⊢ Irreducible z
      -/
      apply irreducible_of_normalized_factor z
        /-
          n : Nat
          h0 : Ne n 0
          x : Finset Nat
          hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          y : Finset Nat
          hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          h : Eq x.val.prod y.val.prod
          z : Nat
          hz : Membership.mem x.val z
          ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors ?m.57482) z
        -/
      · rw [← Multiset.mem_toFinset]
        /-
          n : Nat
          h0 : Ne n 0
          x : Finset Nat
          hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          y : Finset Nat
          hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          h : Eq x.val.prod y.val.prod
          z : Nat
          hz : Membership.mem x.val z
          ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors ?m.57482).toFins …
        -/
        apply hx hz
        /-
          🎉 no goals
        -/
      /-
        n : Nat
        h0 : Ne n 0
        x : Finset Nat
        hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        y : Finset Nat
        hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        h : Eq x.val.prod y.val.prod
        ⊢ ∀ (x : Nat), Membership.mem y.val x → Irreducible x
      -/
    · intro z hz
      /-
        n : Nat
        h0 : Ne n 0
        x : Finset Nat
        hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        y : Finset Nat
        hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
        h : Eq x.val.prod y.val.prod
        z : Nat
        hz : Membership.mem y.val z
        ⊢ Irreducible z
      -/
      apply irreducible_of_normalized_factor z
        /-
          n : Nat
          h0 : Ne n 0
          x : Finset Nat
          hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          y : Finset Nat
          hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          h : Eq x.val.prod y.val.prod
          z : Nat
          hz : Membership.mem y.val z
          ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors ?m.57695) z
        -/
      · rw [← Multiset.mem_toFinset]
        /-
          n : Nat
          h0 : Ne n 0
          x : Finset Nat
          hx : HasSubset.Subset x (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          y : Finset Nat
          hy : HasSubset.Subset y (UniqueFactorizationMonoid.normalizedFactors n).toFinset
          h : Eq x.val.prod y.val.prod
          z : Nat
          hz : Membership.mem y.val z
          ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors ?m.57695).toFins …
        -/
        apply hy hz
        /-
          🎉 no goals
        -/


theorem sum_divisors_filter_squarefree {n : ℕ} (h0 : n ≠ 0) {α : Type*} [AddCommMonoid α]
    {f : ℕ → α} :
    ∑ d ∈ n.divisors with Squarefree d, f d =
      ∑ i ∈ (UniqueFactorizationMonoid.normalizedFactors n).toFinset.powerset, f i.val.prod := by
  rw [Finset.sum_eq_multiset_sum, divisors_filter_squarefree h0, Multiset.map_map,
    Finset.sum_eq_multiset_sum]
  /-
    n : Nat
    h0 : Ne n 0
    α : Type u_1
    inst✝ : AddCommMonoid α
    f : Nat → α
    ⊢ Eq (Multiset.map (Function.comp f fun x => x.val.prod) (UniqueFactorizationM …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem sq_mul_squarefree_of_pos {n : ℕ} (hn : 0 < n) :
    ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ b ^ 2 * a = n ∧ Squarefree a := by
  classical -- Porting note: This line is not needed in Lean 3
  set S := {s ∈ range (n + 1) | s ∣ n ∧ ∃ x, s = x ^ 2}
  have hSne : S.Nonempty := by
    use 1
    have h1 : 0 < n ∧ ∃ x : ℕ, 1 = x ^ 2 := ⟨hn, ⟨1, (one_pow 2).symm⟩⟩
    simp [S, h1]
  let s := Finset.max' S hSne
  have hs : s ∈ S := Finset.max'_mem S hSne
  simp only [S, Finset.mem_filter, Finset.mem_range] at hs
  obtain ⟨-, ⟨a, hsa⟩, ⟨b, hsb⟩⟩ := hs
  rw [hsa] at hn
  obtain ⟨hlts, hlta⟩ := CanonicallyOrderedCommSemiring.mul_pos.mp hn
  rw [hsb] at hsa hn hlts
  refine ⟨a, b, hlta, (pow_pos_iff two_ne_zero).mp hlts, hsa.symm, ?_⟩
  rintro x ⟨y, hy⟩
  rw [Nat.isUnit_iff]
  by_contra hx
  refine Nat.lt_le_asymm ?_ (Finset.le_max' S ((b * x) ^ 2) ?_)
  -- Porting note: these two goals were in the opposite order in Lean 3
  · convert lt_mul_of_one_lt_right hlts
      (one_lt_pow two_ne_zero (one_lt_iff_ne_zero_and_ne_one.mpr ⟨fun h => by simp_all, hx⟩))
      using 1
    rw [mul_pow]
  · simp_rw [S, hsa, Finset.mem_filter, Finset.mem_range]
    refine ⟨Nat.lt_succ_iff.mpr (le_of_dvd hn ?_), ?_, ⟨b * x, rfl⟩⟩ <;> use y <;> rw [hy] <;> ring


theorem sq_mul_squarefree_of_pos' {n : ℕ} (h : 0 < n) :
    ∃ a b : ℕ, (b + 1) ^ 2 * (a + 1) = n ∧ Squarefree (a + 1) := by
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Exists fun a => Exists fun b => And (Eq (HMul.hMul (HPow.hPow (HAdd.hAdd b 1 …
  -/
  obtain ⟨a₁, b₁, ha₁, hb₁, hab₁, hab₂⟩ := sq_mul_squarefree_of_pos h
  /-
    case intro.intro.intro.intro.intro
    n : Nat
    h : LT.lt 0 n
    a₁ b₁ : Nat
    ha₁ : LT.lt 0 a₁
    hb₁ : LT.lt 0 b₁
    hab₁ : Eq (HMul.hMul (HPow.hPow b₁ 2) a₁) n
    hab₂ : Squarefree a₁
    ⊢ Exists fun a => Exists fun b => And (Eq (HMul.hMul (HPow.hPow (HAdd.hAdd b 1 …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  refine ⟨a₁.pred, b₁.pred, ?_, ?_⟩ <;> simpa only [add_one, succ_pred_eq_of_pos, ha₁, hb₁]
                                        /-
                                          🎉 no goals
                                        -/


theorem sq_mul_squarefree (n : ℕ) : ∃ a b : ℕ, b ^ 2 * a = n ∧ Squarefree a := by
  /-
    n : Nat
    ⊢ Exists fun a => Exists fun b => And (Eq (HMul.hMul (HPow.hPow b 2) a) n) (Sq …
  -/
  cases' n with n
    /-
      case zero
      ⊢ Exists fun a => Exists fun b => And (Eq (HMul.hMul (HPow.hPow b 2) a) 0) (Sq …
    -/
  · exact ⟨1, 0, by simp, squarefree_one⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ⊢ Exists fun a => Exists fun b => And (Eq (HMul.hMul (HPow.hPow b 2) a) (HAdd. …
    -/
  · obtain ⟨a, b, -, -, h₁, h₂⟩ := sq_mul_squarefree_of_pos (succ_pos n)
    /-
      case succ.intro.intro.intro.intro.intro
      n a b : Nat
      h₁ : Eq (HMul.hMul (HPow.hPow b 2) a) n.succ
      h₂ : Squarefree a
      ⊢ Exists fun a => Exists fun b => And (Eq (HMul.hMul (HPow.hPow b 2) a) (HAdd. …
    -/
    exact ⟨a, b, h₁, h₂⟩
    /-
      🎉 no goals
    -/


/-- `Squarefree` is multiplicative. Note that the → direction does not require `hmn`
and generalizes to arbitrary commutative monoids. See `Squarefree.of_mul_left` and
`Squarefree.of_mul_right` above for auxiliary lemmas. -/
theorem squarefree_mul {m n : ℕ} (hmn : m.Coprime n) :
    Squarefree (m * n) ↔ Squarefree m ∧ Squarefree n := by
  /-
    m n : Nat
    hmn : m.Coprime n
    ⊢ Iff (Squarefree (HMul.hMul m n)) (And (Squarefree m) (Squarefree n))
  -/
  simp only [squarefree_iff_prime_squarefree, ← sq, ← forall_and]
  /-
    m n : Nat
    hmn : m.Coprime n
    ⊢ Iff (∀ (x : Nat), Nat.Prime x → Not (Dvd.dvd (HPow.hPow x 2) (HMul.hMul m n) …
  -/
  refine forall₂_congr fun p hp => ?_
  /-
    m n : Nat
    hmn : m.Coprime n
    p : Nat
    hp : Nat.Prime p
    ⊢ Iff (Not (Dvd.dvd (HPow.hPow p 2) (HMul.hMul m n))) (And (Not (Dvd.dvd (HPow …
  -/
  simp only [hmn.isPrimePow_dvd_mul (hp.isPrimePow.pow two_ne_zero), not_or]
  /-
    🎉 no goals
  -/


theorem coprime_of_squarefree_mul {m n : ℕ} (h : Squarefree (m * n)) : m.Coprime n :=
  coprime_of_dvd fun p hp hm hn => squarefree_iff_prime_squarefree.mp h p hp (mul_dvd_mul hm hn)


theorem squarefree_mul_iff {m n : ℕ} :
    Squarefree (m * n) ↔ m.Coprime n ∧ Squarefree m ∧ Squarefree n :=
  ⟨fun h => ⟨coprime_of_squarefree_mul h, (squarefree_mul <| coprime_of_squarefree_mul h).mp h⟩,
    fun h => (squarefree_mul h.1).mpr h.2⟩


lemma coprime_div_gcd_of_squarefree (hm : Squarefree m) (hn : n ≠ 0) : Coprime (m / gcd m n) n := by
  have : Coprime (m / gcd m n) (gcd m n) :=
    coprime_of_squarefree_mul <| by simpa [Nat.div_mul_cancel, gcd_dvd_left]
  simpa [Nat.div_mul_cancel, gcd_dvd_right] using
    (coprime_div_gcd_div_gcd (m := m) (gcd_ne_zero_right hn).bot_lt).mul_right this


lemma prod_primeFactors_of_squarefree (hn : Squarefree n) : ∏ p ∈ n.primeFactors, p = n := by
  rw [← toFinset_factors, List.prod_toFinset _ hn.nodup_primeFactorsList,
    List.map_id', Nat.prod_primeFactorsList hn.ne_zero]


lemma primeFactors_prod (hs : ∀ p ∈ s, p.Prime) : primeFactors (∏ p ∈ s, p) = s := by
  /-
    s : Finset Nat
    hs : ∀ (p : Nat), Membership.mem s p → Nat.Prime p
    ⊢ Eq (s.prod fun p => p).primeFactors s
  -/
  have hn : ∏ p ∈ s, p ≠ 0 := prod_ne_zero_iff.2 fun p hp ↦ (hs _ hp).ne_zero
  /-
    s : Finset Nat
    hs : ∀ (p : Nat), Membership.mem s p → Nat.Prime p
    hn : Ne (s.prod fun p => p) 0
    ⊢ Eq (s.prod fun p => p).primeFactors s
  -/
  ext p
  /-
    case h
    s : Finset Nat
    hs : ∀ (p : Nat), Membership.mem s p → Nat.Prime p
    hn : Ne (s.prod fun p => p) 0
    p : Nat
    ⊢ Iff (Membership.mem (s.prod fun p => p).primeFactors p) (Membership.mem s p)
  -/
  rw [mem_primeFactors_of_ne_zero hn, and_congr_right (fun hp ↦ hp.prime.dvd_finset_prod_iff _)]
  /-
    case h
    s : Finset Nat
    hs : ∀ (p : Nat), Membership.mem s p → Nat.Prime p
    hn : Ne (s.prod fun p => p) 0
    p : Nat
    ⊢ Iff (And (Nat.Prime p) (Exists fun a => And (Membership.mem s a) (Dvd.dvd p  …
  -/
  refine ⟨?_, fun hp ↦ ⟨hs _ hp, _, hp, dvd_rfl⟩⟩
  /-
    case h
    s : Finset Nat
    hs : ∀ (p : Nat), Membership.mem s p → Nat.Prime p
    hn : Ne (s.prod fun p => p) 0
    p : Nat
    ⊢ And (Nat.Prime p) (Exists fun a => And (Membership.mem s a) (Dvd.dvd p a)) → …
  -/
  rintro ⟨hp, q, hq, hpq⟩
  /-
    case h.intro.intro.intro
    s : Finset Nat
    hs : ∀ (p : Nat), Membership.mem s p → Nat.Prime p
    hn : Ne (s.prod fun p => p) 0
    p : Nat
    hp : Nat.Prime p
    q : Nat
    hq : Membership.mem s q
    hpq : Dvd.dvd p q
    ⊢ Membership.mem s p
  -/
  rwa [← ((hs _ hq).dvd_iff_eq hp.ne_one).1 hpq]
  /-
    🎉 no goals
  -/


lemma primeFactors_div_gcd (hm : Squarefree m) (hn : n ≠ 0) :
    primeFactors (m / m.gcd n) = primeFactors m \ primeFactors n := by
  /-
    m n : Nat
    hm : Squarefree m
    hn : Ne n 0
    ⊢ Eq (HDiv.hDiv m (m.gcd n)).primeFactors (SDiff.sdiff m.primeFactors n.primeF …
  -/
  ext p
  /-
    case h
    m n : Nat
    hm : Squarefree m
    hn : Ne n 0
    p : Nat
    ⊢ Iff (Membership.mem (HDiv.hDiv m (m.gcd n)).primeFactors p) (Membership.mem  …
  -/
  have : m / m.gcd n ≠ 0 := by simp [gcd_ne_zero_right hn, gcd_le_left _ hm.ne_zero.bot_lt]
  simp only [mem_primeFactors, ne_eq, this, not_false_eq_true, and_true, not_and, mem_sdiff,
    hm.ne_zero, hn, dvd_div_iff_mul_dvd (gcd_dvd_left _ _)]
  refine ⟨fun hp ↦ ⟨⟨hp.1, dvd_of_mul_left_dvd hp.2⟩, fun _ hpn ↦ hp.1.not_unit <| hm _ <|
    (mul_dvd_mul_right (dvd_gcd (dvd_of_mul_left_dvd hp.2) hpn) _).trans hp.2⟩, fun hp ↦
      ⟨hp.1.1, Coprime.mul_dvd_of_dvd_of_dvd ?_ (gcd_dvd_left _ _) hp.1.2⟩⟩
  /-
    case h
    m n : Nat
    hm : Squarefree m
    hn : Ne n 0
    p : Nat
    this : Ne (HDiv.hDiv m (m.gcd n)) 0
    hp : And (And (Nat.Prime p) (Dvd.dvd p m)) (Nat.Prime p → Not (Dvd.dvd p n))
    ⊢ (m.gcd n).Coprime p
  -/
  rw [coprime_comm, hp.1.1.coprime_iff_not_dvd]
  /-
    case h
    m n : Nat
    hm : Squarefree m
    hn : Ne n 0
    p : Nat
    this : Ne (HDiv.hDiv m (m.gcd n)) 0
    hp : And (And (Nat.Prime p) (Dvd.dvd p m)) (Nat.Prime p → Not (Dvd.dvd p n))
    ⊢ Not (Dvd.dvd p (m.gcd n))
  -/
  exact fun hpn ↦ hp.2 hp.1.1 <| hpn.trans <| gcd_dvd_right _ _
  /-
    🎉 no goals
  -/


lemma prod_primeFactors_invOn_squarefree :
    Set.InvOn (fun n : ℕ ↦ (factorization n).support) (fun s ↦ ∏ p ∈ s, p)
      {s | ∀ p ∈ s, p.Prime} {n | Squarefree n} :=
  ⟨fun _s ↦ primeFactors_prod, fun _n ↦ prod_primeFactors_of_squarefree⟩


theorem prod_primeFactors_sdiff_of_squarefree {n : ℕ} (hn : Squarefree n) {t : Finset ℕ}
    (ht : t ⊆ n.primeFactors) :
    ∏ a ∈ (n.primeFactors \ t), a = n / ∏ a ∈ t, a := by
  refine symm <| Nat.div_eq_of_eq_mul_left (Finset.prod_pos
    fun p hp => (prime_of_mem_primeFactorsList (List.mem_toFinset.mp (ht hp))).pos) ?_
  /-
    n : Nat
    hn : Squarefree n
    t : Finset Nat
    ht : HasSubset.Subset t n.primeFactors
    ⊢ Eq n (HMul.hMul ((SDiff.sdiff n.primeFactors t).prod fun a => a) (t.prod fun …
  -/
  rw [Finset.prod_sdiff ht, prod_primeFactors_of_squarefree hn]
  /-
    🎉 no goals
  -/


