theorem prime_pow_coprime_prod_of_coprime_insert [DecidableEq α] {s : Finset α} (i : α → ℕ) (p : α)
    (hps : p ∉ s) (is_prime : ∀ q ∈ insert p s, Prime q)
    (is_coprime : ∀ᵉ (q ∈ insert p s) (q' ∈ insert p s), q ∣ q' → q = q') :
    IsRelPrime (p ^ i p) (∏ p' ∈ s, p' ^ i p') := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    ⊢ IsRelPrime (HPow.hPow p (i p)) (s.prod fun p' => HPow.hPow p' (i p'))
  -/
  have hp := is_prime _ (Finset.mem_insert_self _ _)
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    ⊢ IsRelPrime (HPow.hPow p (i p)) (s.prod fun p' => HPow.hPow p' (i p'))
  -/
  refine (isRelPrime_iff_no_prime_factors <| pow_ne_zero _ hp.ne_zero).mpr ?_
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    ⊢ ∀ ⦃d : α⦄, Dvd.dvd d (HPow.hPow p (i p)) → Dvd.dvd d (s.prod fun p' => HPow. …
  -/
  intro d hdp hdprod hd
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdp : Dvd.dvd d (HPow.hPow p (i p))
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    ⊢ False
  -/
  apply hps
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdp : Dvd.dvd d (HPow.hPow p (i p))
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    ⊢ Membership.mem s p
  -/
  replace hdp := hd.dvd_of_dvd_pow hdp
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    hdp : Dvd.dvd d p
    ⊢ Membership.mem s p
  -/
  obtain ⟨q, q_mem', hdq⟩ := hd.exists_mem_multiset_dvd hdprod
  /-
    case intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    hdp : Dvd.dvd d p
    q : α
    q_mem' : Membership.mem (Multiset.map (fun p' => HPow.hPow p' (i p')) s.val) q
    hdq : Dvd.dvd d q
    ⊢ Membership.mem s p
  -/
  obtain ⟨q, q_mem, rfl⟩ := Multiset.mem_map.mp q_mem'
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    hdp : Dvd.dvd d p
    q : α
    q_mem : Membership.mem s.val q
    q_mem' : Membership.mem (Multiset.map (fun p' => HPow.hPow p' (i p')) s.val) ( …
    hdq : Dvd.dvd d (HPow.hPow q (i q))
    ⊢ Membership.mem s p
  -/
  replace hdq := hd.dvd_of_dvd_pow hdq
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    hdp : Dvd.dvd d p
    q : α
    q_mem : Membership.mem s.val q
    q_mem' : Membership.mem (Multiset.map (fun p' => HPow.hPow p' (i p')) s.val) ( …
    hdq : Dvd.dvd d q
    ⊢ Membership.mem s p
  -/
  have : p ∣ q := dvd_trans (hd.irreducible.dvd_symm hp.irreducible hdp) hdq
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq α
    s : Finset α
    i : α → Nat
    p : α
    hps : Not (Membership.mem s p)
    is_prime : ∀ (q : α), Membership.mem (Insert.insert p s) q → Prime q
    is_coprime : ∀ (q : α), Membership.mem (Insert.insert p s) q → ∀ (q' : α), Mem …
    hp : Prime p
    d : α
    hdprod : Dvd.dvd d (s.prod fun p' => HPow.hPow p' (i p'))
    hd : Prime d
    hdp : Dvd.dvd d p
    q : α
    q_mem : Membership.mem s.val q
    q_mem' : Membership.mem (Multiset.map (fun p' => HPow.hPow p' (i p')) s.val) ( …
    hdq : Dvd.dvd d q
    this : Dvd.dvd p q
    ⊢ Membership.mem s p
  -/
  convert q_mem using 0
  rw [Finset.mem_val,
    is_coprime _ (Finset.mem_insert_self p s) _ (Finset.mem_insert_of_mem q_mem) this]


/-- If `P` holds for units and powers of primes,
and `P x ∧ P y` for coprime `x, y` implies `P (x * y)`,
then `P` holds on a product of powers of distinct primes. -/
@[elab_as_elim]
theorem induction_on_prime_power {P : α → Prop} (s : Finset α) (i : α → ℕ)
    (is_prime : ∀ p ∈ s, Prime p) (is_coprime : ∀ᵉ (p ∈ s) (q ∈ s), p ∣ q → p = q)
    (h1 : ∀ {x}, IsUnit x → P x) (hpr : ∀ {p} (i : ℕ), Prime p → P (p ^ i))
    (hcp : ∀ {x y}, IsRelPrime x y → P x → P y → P (x * y)) :
    P (∏ p ∈ s, p ^ i p) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    s : Finset α
    i : α → Nat
    is_prime : ∀ (p : α), Membership.mem s p → Prime p
    is_coprime : ∀ (p : α), Membership.mem s p → ∀ (q : α), Membership.mem s q → D …
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    ⊢ P (s.prod fun p => HPow.hPow p (i p))
  -/
  letI := Classical.decEq α
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    s : Finset α
    i : α → Nat
    is_prime : ∀ (p : α), Membership.mem s p → Prime p
    is_coprime : ∀ (p : α), Membership.mem s p → ∀ (q : α), Membership.mem s q → D …
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this : DecidableEq α := Classical.decEq α
    ⊢ P (s.prod fun p => HPow.hPow p (i p))
  -/
  induction' s using Finset.induction_on with p f' hpf' ih
    /-
      case empty
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      P : α → Prop
      i : α → Nat
      h1 : ∀ {x : α}, IsUnit x → P x
      hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
      hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
      this : DecidableEq α := Classical.decEq α
      is_prime : ∀ (p : α), Membership.mem EmptyCollection.emptyCollection p → Prime p
      is_coprime : ∀ (p : α), Membership.mem EmptyCollection.emptyCollection p → ∀ ( …
      ⊢ P (EmptyCollection.emptyCollection.prod fun p => HPow.hPow p (i p))
    -/
  · simpa using h1 isUnit_one
    /-
      🎉 no goals
    -/
  /-
    case insert
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    i : α → Nat
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this : DecidableEq α := Classical.decEq α
    p : α
    f' : Finset α
    hpf' : Not (Membership.mem f' p)
    ih : (∀ (p : α), Membership.mem f' p → Prime p) → (∀ (p : α), Membership.mem f …
    is_prime : ∀ (p_1 : α), Membership.mem (Insert.insert p f') p_1 → Prime p_1
    is_coprime : ∀ (p_1 : α), Membership.mem (Insert.insert p f') p_1 → ∀ (q : α), …
    ⊢ P ((Insert.insert p f').prod fun p => HPow.hPow p (i p))
  -/
  rw [Finset.prod_insert hpf']
  exact
    hcp (prime_pow_coprime_prod_of_coprime_insert i p hpf' is_prime is_coprime)
      (hpr (i p) (is_prime _ (Finset.mem_insert_self _ _)))
      (ih (fun q hq => is_prime _ (Finset.mem_insert_of_mem hq)) fun q hq q' hq' =>
        is_coprime _ (Finset.mem_insert_of_mem hq) _ (Finset.mem_insert_of_mem hq'))


/-- If `P` holds for `0`, units and powers of primes,
and `P x ∧ P y` for coprime `x, y` implies `P (x * y)`,
then `P` holds on all `a : α`. -/
@[elab_as_elim]
theorem induction_on_coprime {P : α → Prop} (a : α) (h0 : P 0) (h1 : ∀ {x}, IsUnit x → P x)
    (hpr : ∀ {p} (i : ℕ), Prime p → P (p ^ i))
    (hcp : ∀ {x y}, IsRelPrime x y → P x → P y → P (x * y)) : P a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    ⊢ P a
  -/
  letI := Classical.decEq α
  have P_of_associated : ∀ {x y}, Associated x y → P x → P y := by
    rintro x y ⟨u, rfl⟩ hx
    exact hcp (fun p _ hpx => isUnit_of_dvd_unit hpx u.isUnit) hx (h1 u.isUnit)
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this : DecidableEq α := Classical.decEq α
    P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
    ⊢ P a
  -/
  by_cases ha0 : a = 0
    /-
      case pos
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      P : α → Prop
      a : α
      h0 : P 0
      h1 : ∀ {x : α}, IsUnit x → P x
      hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
      hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
      this : DecidableEq α := Classical.decEq α
      P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
      ha0 : Eq a 0
      ⊢ P a
    -/
  · rwa [ha0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this : DecidableEq α := Classical.decEq α
    P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
    ha0 : Not (Eq a 0)
    ⊢ P a
  -/
  haveI : Nontrivial α := ⟨⟨_, _, ha0⟩⟩
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this✝ : DecidableEq α := Classical.decEq α
    P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
    ha0 : Not (Eq a 0)
    this : Nontrivial α
    ⊢ P a
  -/
  letI : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this✝¹ : DecidableEq α := Classical.decEq α
    P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
    ha0 : Not (Eq a 0)
    this✝ : Nontrivial α
    this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
    ⊢ P a
  -/
  refine P_of_associated (prod_normalizedFactors ha0) ?_
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this✝¹ : DecidableEq α := Classical.decEq α
    P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
    ha0 : Not (Eq a 0)
    this✝ : Nontrivial α
    this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
    ⊢ P (UniqueFactorizationMonoid.normalizedFactors a).prod
  -/
  rw [← (normalizedFactors a).map_id, Finset.prod_multiset_map_count]
  /-
    case neg
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    P : α → Prop
    a : α
    h0 : P 0
    h1 : ∀ {x : α}, IsUnit x → P x
    hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
    hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
    this✝¹ : DecidableEq α := Classical.decEq α
    P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
    ha0 : Not (Eq a 0)
    this✝ : Nontrivial α
    this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
    ⊢ P ((UniqueFactorizationMonoid.normalizedFactors a).toFinset.prod fun m => HP …
  -/
  refine induction_on_prime_power _ _ ?_ ?_ @h1 @hpr @hcp <;> simp only [Multiset.mem_toFinset]
    /-
      case neg.refine_1
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      P : α → Prop
      a : α
      h0 : P 0
      h1 : ∀ {x : α}, IsUnit x → P x
      hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
      hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
      this✝¹ : DecidableEq α := Classical.decEq α
      P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
      ha0 : Not (Eq a 0)
      this✝ : Nontrivial α
      this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
      ⊢ ∀ (p : α), Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p  …
    -/
  · apply prime_of_normalized_factor
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      P : α → Prop
      a : α
      h0 : P 0
      h1 : ∀ {x : α}, IsUnit x → P x
      hpr : ∀ {p : α} (i : Nat), Prime p → P (HPow.hPow p i)
      hcp : ∀ {x y : α}, IsRelPrime x y → P x → P y → P (HMul.hMul x y)
      this✝¹ : DecidableEq α := Classical.decEq α
      P_of_associated : ∀ {x y : α}, Associated x y → P x → P y
      ha0 : Not (Eq a 0)
      this✝ : Nontrivial α
      this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
      ⊢ ∀ (p : α), Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p  …
    -/
  · apply normalizedFactors_eq_of_dvd
    /-
      🎉 no goals
    -/


/-- If `f` maps `p ^ i` to `(f p) ^ i` for primes `p`, and `f`
is multiplicative on coprime elements, then `f` is multiplicative on all products of primes. -/
theorem multiplicative_prime_power {f : α → β} (s : Finset α) (i j : α → ℕ)
    (is_prime : ∀ p ∈ s, Prime p) (is_coprime : ∀ᵉ (p ∈ s) (q ∈ s), p ∣ q → p = q)
    (h1 : ∀ {x y}, IsUnit y → f (x * y) = f x * f y)
    (hpr : ∀ {p} (i : ℕ), Prime p → f (p ^ i) = f p ^ i)
    (hcp : ∀ {x y}, IsRelPrime x y → f (x * y) = f x * f y) :
    f (∏ p ∈ s, p ^ (i p + j p)) = f (∏ p ∈ s, p ^ i p) * f (∏ p ∈ s, p ^ j p) := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    s : Finset α
    i j : α → Nat
    is_prime : ∀ (p : α), Membership.mem s p → Prime p
    is_coprime : ∀ (p : α), Membership.mem s p → ∀ (q : α), Membership.mem s q → D …
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    ⊢ Eq (f (s.prod fun p => HPow.hPow p (HAdd.hAdd (i p) (j p)))) (HMul.hMul (f ( …
  -/
  letI := Classical.decEq α
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    s : Finset α
    i j : α → Nat
    is_prime : ∀ (p : α), Membership.mem s p → Prime p
    is_coprime : ∀ (p : α), Membership.mem s p → ∀ (q : α), Membership.mem s q → D …
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    ⊢ Eq (f (s.prod fun p => HPow.hPow p (HAdd.hAdd (i p) (j p)))) (HMul.hMul (f ( …
  -/
  induction' s using Finset.induction_on with p s hps ih
    /-
      case empty
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      β : Type u_3
      inst✝ : CancelCommMonoidWithZero β
      f : α → β
      i j : α → Nat
      h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
      hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
      this : DecidableEq α := Classical.decEq α
      is_prime : ∀ (p : α), Membership.mem EmptyCollection.emptyCollection p → Prime p
      is_coprime : ∀ (p : α), Membership.mem EmptyCollection.emptyCollection p → ∀ ( …
      ⊢ Eq (f (EmptyCollection.emptyCollection.prod fun p => HPow.hPow p (HAdd.hAdd  …
    -/
  · simpa using h1 isUnit_one
    /-
      🎉 no goals
    -/
  /-
    case insert
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    i j : α → Nat
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    p : α
    s : Finset α
    hps : Not (Membership.mem s p)
    ih : (∀ (p : α), Membership.mem s p → Prime p) → (∀ (p : α), Membership.mem s  …
    is_prime : ∀ (p_1 : α), Membership.mem (Insert.insert p s) p_1 → Prime p_1
    is_coprime : ∀ (p_1 : α), Membership.mem (Insert.insert p s) p_1 → ∀ (q : α),  …
    ⊢ Eq (f ((Insert.insert p s).prod fun p => HPow.hPow p (HAdd.hAdd (i p) (j p)) …
  -/
  have hpr_p := is_prime _ (Finset.mem_insert_self _ _)
  /-
    case insert
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    i j : α → Nat
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    p : α
    s : Finset α
    hps : Not (Membership.mem s p)
    ih : (∀ (p : α), Membership.mem s p → Prime p) → (∀ (p : α), Membership.mem s  …
    is_prime : ∀ (p_1 : α), Membership.mem (Insert.insert p s) p_1 → Prime p_1
    is_coprime : ∀ (p_1 : α), Membership.mem (Insert.insert p s) p_1 → ∀ (q : α),  …
    hpr_p : Prime p
    ⊢ Eq (f ((Insert.insert p s).prod fun p => HPow.hPow p (HAdd.hAdd (i p) (j p)) …
  -/
  have hpr_s : ∀ p ∈ s, Prime p := fun p hp => is_prime _ (Finset.mem_insert_of_mem hp)
  /-
    case insert
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    i j : α → Nat
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    p : α
    s : Finset α
    hps : Not (Membership.mem s p)
    ih : (∀ (p : α), Membership.mem s p → Prime p) → (∀ (p : α), Membership.mem s  …
    is_prime : ∀ (p_1 : α), Membership.mem (Insert.insert p s) p_1 → Prime p_1
    is_coprime : ∀ (p_1 : α), Membership.mem (Insert.insert p s) p_1 → ∀ (q : α),  …
    hpr_p : Prime p
    hpr_s : ∀ (p : α), Membership.mem s p → Prime p
    ⊢ Eq (f ((Insert.insert p s).prod fun p => HPow.hPow p (HAdd.hAdd (i p) (j p)) …
  -/
  have hcp_p := fun i => prime_pow_coprime_prod_of_coprime_insert i p hps is_prime is_coprime
  have hcp_s : ∀ᵉ (p ∈ s) (q ∈ s), p ∣ q → p = q := fun p hp q hq =>
    is_coprime p (Finset.mem_insert_of_mem hp) q (Finset.mem_insert_of_mem hq)
  rw [Finset.prod_insert hps, Finset.prod_insert hps, Finset.prod_insert hps, hcp (hcp_p _),
    hpr _ hpr_p, hcp (hcp_p _), hpr _ hpr_p, hcp (hcp_p (fun p => i p + j p)), hpr _ hpr_p,
    ih hpr_s hcp_s, pow_add, mul_assoc, mul_left_comm (f p ^ j p), mul_assoc]


/-- If `f` maps `p ^ i` to `(f p) ^ i` for primes `p`, and `f`
is multiplicative on coprime elements, then `f` is multiplicative everywhere. -/
theorem multiplicative_of_coprime (f : α → β) (a b : α) (h0 : f 0 = 0)
    (h1 : ∀ {x y}, IsUnit y → f (x * y) = f x * f y)
    (hpr : ∀ {p} (i : ℕ), Prime p → f (p ^ i) = f p ^ i)
    (hcp : ∀ {x y}, IsRelPrime x y → f (x * y) = f x * f y) :
    f (a * b) = f a * f b := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
  -/
  letI := Classical.decEq α
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
  -/
  by_cases ha0 : a = 0
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      β : Type u_3
      inst✝ : CancelCommMonoidWithZero β
      f : α → β
      a b : α
      h0 : Eq (f 0) 0
      h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
      hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
      this : DecidableEq α := Classical.decEq α
      ha0 : Eq a 0
      ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
    -/
  · rw [ha0, zero_mul, h0, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    ha0 : Not (Eq a 0)
    ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
  -/
  by_cases hb0 : b = 0
    /-
      case pos
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      β : Type u_3
      inst✝ : CancelCommMonoidWithZero β
      f : α → β
      a b : α
      h0 : Eq (f 0) 0
      h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
      hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
      this : DecidableEq α := Classical.decEq α
      ha0 : Not (Eq a 0)
      hb0 : Eq b 0
      ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
    -/
  · rw [hb0, mul_zero, h0, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    ha0 : Not (Eq a 0)
    hb0 : Not (Eq b 0)
    ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
  -/
  by_cases hf1 : f 1 = 0
  · calc
      f (a * b) = f (a * b * 1) := by rw [mul_one]
      _ = 0 := by simp only [h1 isUnit_one, hf1, mul_zero]
      _ = f a * f (b * 1) := by simp only [h1 isUnit_one, hf1, mul_zero]
      _ = f a * f b := by rw [mul_one]
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this : DecidableEq α := Classical.decEq α
    ha0 : Not (Eq a 0)
    hb0 : Not (Eq b 0)
    hf1 : Not (Eq (f 1) 0)
    ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
  -/
  haveI : Nontrivial α := ⟨⟨_, _, ha0⟩⟩
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this✝ : DecidableEq α := Classical.decEq α
    ha0 : Not (Eq a 0)
    hb0 : Not (Eq b 0)
    hf1 : Not (Eq (f 1) 0)
    this : Nontrivial α
    ⊢ Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
  -/
  letI : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
  suffices
      f (∏ p ∈ (normalizedFactors a).toFinset ∪ (normalizedFactors b).toFinset,
        p ^ ((normalizedFactors a).count p + (normalizedFactors b).count p)) =
      f (∏ p ∈ (normalizedFactors a).toFinset ∪ (normalizedFactors b).toFinset,
        p ^ (normalizedFactors a).count p) *
      f (∏ p ∈ (normalizedFactors a).toFinset ∪ (normalizedFactors b).toFinset,
        p ^ (normalizedFactors b).count p) by
    obtain ⟨ua, a_eq⟩ := prod_normalizedFactors ha0
    obtain ⟨ub, b_eq⟩ := prod_normalizedFactors hb0
    rw [← a_eq, ← b_eq, mul_right_comm (Multiset.prod (normalizedFactors a)) ua
        (Multiset.prod (normalizedFactors b) * ub), h1 ua.isUnit, h1 ub.isUnit, h1 ua.isUnit, ←
      mul_assoc, h1 ub.isUnit, mul_right_comm _ (f ua), ← mul_assoc]
    congr
    rw [← (normalizedFactors a).map_id, ← (normalizedFactors b).map_id,
      Finset.prod_multiset_map_count, Finset.prod_multiset_map_count,
      Finset.prod_subset (Finset.subset_union_left (s₂ := (normalizedFactors b).toFinset)),
      Finset.prod_subset (Finset.subset_union_right (s₂ := (normalizedFactors b).toFinset)), ←
      Finset.prod_mul_distrib]
    · simp_rw [id, ← pow_add, this]
    all_goals simp only [Multiset.mem_toFinset]
    · intro p _ hpb
      simp [hpb]
    · intro p _ hpa
      simp [hpa]
  /-
    case neg
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this✝¹ : DecidableEq α := Classical.decEq α
    ha0 : Not (Eq a 0)
    hb0 : Not (Eq b 0)
    hf1 : Not (Eq (f 1) 0)
    this✝ : Nontrivial α
    this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
    ⊢ Eq (f ((Union.union (UniqueFactorizationMonoid.normalizedFactors a).toFinset …
  -/
  refine multiplicative_prime_power _ _ _ ?_ ?_ @h1 @hpr @hcp
  /-
    case neg.refine_1
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    β : Type u_3
    inst✝ : CancelCommMonoidWithZero β
    f : α → β
    a b : α
    h0 : Eq (f 0) 0
    h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
    hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
    this✝¹ : DecidableEq α := Classical.decEq α
    ha0 : Not (Eq a 0)
    hb0 : Not (Eq b 0)
    hf1 : Not (Eq (f 1) 0)
    this✝ : Nontrivial α
    this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
    ⊢ ∀ (p : α), Membership.mem (Union.union (UniqueFactorizationMonoid.normalized …
  -/
  all_goals simp only [Multiset.mem_toFinset, Finset.mem_union]
    /-
      case neg.refine_1
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      β : Type u_3
      inst✝ : CancelCommMonoidWithZero β
      f : α → β
      a b : α
      h0 : Eq (f 0) 0
      h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
      hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
      this✝¹ : DecidableEq α := Classical.decEq α
      ha0 : Not (Eq a 0)
      hb0 : Not (Eq b 0)
      hf1 : Not (Eq (f 1) 0)
      this✝ : Nontrivial α
      this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
      ⊢ ∀ (p : α), Or (Membership.mem (UniqueFactorizationMonoid.normalizedFactors a …
    -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  · rintro p (hpa | hpb) <;> apply prime_of_normalized_factor <;> assumption
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    /-
      case neg.refine_2
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      β : Type u_3
      inst✝ : CancelCommMonoidWithZero β
      f : α → β
      a b : α
      h0 : Eq (f 0) 0
      h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
      hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
      hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
      this✝¹ : DecidableEq α := Classical.decEq α
      ha0 : Not (Eq a 0)
      hb0 : Not (Eq b 0)
      hf1 : Not (Eq (f 1) 0)
      this✝ : Nontrivial α
      this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
      ⊢ ∀ (p : α), Or (Membership.mem (UniqueFactorizationMonoid.normalizedFactors a …
    -/
  · rintro p (hp | hp) q (hq | hq) hdvd <;>
      /-
        case neg.refine_2.inl.inl
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        β : Type u_3
        inst✝ : CancelCommMonoidWithZero β
        f : α → β
        a b : α
        h0 : Eq (f 0) 0
        h1 : ∀ {x y : α}, IsUnit y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        hpr : ∀ {p : α} (i : Nat), Prime p → Eq (f (HPow.hPow p i)) (HPow.hPow (f p) i)
        hcp : ∀ {x y : α}, IsRelPrime x y → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f …
        this✝¹ : DecidableEq α := Classical.decEq α
        ha0 : Not (Eq a 0)
        hb0 : Not (Eq b 0)
        hf1 : Not (Eq (f 1) 0)
        this✝ : Nontrivial α
        this : NormalizationMonoid α := UniqueFactorizationMonoid.normalizationMonoid
        p : α
        hp : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) p
        q : α
        hq : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) q
        hdvd : Dvd.dvd p q
        ⊢ Eq p q
      -/
      rw [← normalize_normalized_factor _ hp, ← normalize_normalized_factor _ hq] <;>
      exact
        normalize_eq_normalize hdvd
          ((prime_of_normalized_factor _ hp).irreducible.dvd_symm
            (prime_of_normalized_factor _ hq).irreducible hdvd)


