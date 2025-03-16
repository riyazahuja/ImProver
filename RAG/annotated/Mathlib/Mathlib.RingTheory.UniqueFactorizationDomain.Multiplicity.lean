local infixl:50 " ~ᵤ " => Associated


theorem WfDvdMonoid.max_power_factor' [CommMonoidWithZero α] [WfDvdMonoid α] {a₀ x : α}
    (h : a₀ ≠ 0) (hx : ¬IsUnit x) : ∃ (n : ℕ) (a : α), ¬x ∣ a ∧ a₀ = x ^ n * a := by
  obtain ⟨a, ⟨n, rfl⟩, hm⟩ := wellFounded_dvdNotUnit.has_min
    {a | ∃ n, x ^ n * a = a₀} ⟨a₀, 0, by rw [pow_zero, one_mul]⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : CommMonoidWithZero α
    inst✝ : WfDvdMonoid α
    x : α
    hx : Not (IsUnit x)
    a : α
    n : Nat
    h : Ne (HMul.hMul (HPow.hPow x n) a) 0
    hm : ∀ (x_1 : α), Membership.mem (setOf fun a_1 => Exists fun n_1 => Eq (HMul. …
    ⊢ Exists fun n_1 => Exists fun a_1 => And (Not (Dvd.dvd x a_1)) (Eq (HMul.hMul …
  -/
  refine ⟨n, a, ?_, rfl⟩; rintro ⟨d, rfl⟩
  exact hm d ⟨n + 1, by rw [pow_succ, mul_assoc]⟩
    ⟨(right_ne_zero_of_mul <| right_ne_zero_of_mul h), x, hx, mul_comm _ _⟩


theorem WfDvdMonoid.max_power_factor [CommMonoidWithZero α] [WfDvdMonoid α] {a₀ x : α}
    (h : a₀ ≠ 0) (hx : Irreducible x) : ∃ (n : ℕ) (a : α), ¬x ∣ a ∧ a₀ = x ^ n * a :=
  max_power_factor' h hx.not_unit


theorem FiniteMultiplicity.of_not_isUnit [CancelCommMonoidWithZero α] [WfDvdMonoid α]
    {a b : α} (ha : ¬IsUnit a) (hb : b ≠ 0) : FiniteMultiplicity a b := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : WfDvdMonoid α
    a b : α
    ha : Not (IsUnit a)
    hb : Ne b 0
    ⊢ FiniteMultiplicity a b
  -/
  obtain ⟨n, c, ndvd, rfl⟩ := WfDvdMonoid.max_power_factor' hb ha
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : WfDvdMonoid α
    a : α
    ha : Not (IsUnit a)
    n : Nat
    c : α
    ndvd : Not (Dvd.dvd a c)
    hb : Ne (HMul.hMul (HPow.hPow a n) c) 0
    ⊢ FiniteMultiplicity a (HMul.hMul (HPow.hPow a n) c)
  -/
  exact ⟨n, by rwa [pow_succ, mul_dvd_mul_iff_left (left_ne_zero_of_mul hb)]⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity.finite_of_not_isUnit := FiniteMultiplicity.of_not_isUnit


theorem FiniteMultiplicity.of_prime_left [CancelCommMonoidWithZero α] [WfDvdMonoid α]
    {a b : α} (ha : Prime a) (hb : b ≠ 0) : FiniteMultiplicity a b :=
  .of_not_isUnit ha.not_unit hb


@[deprecated (since := "2024-11-30")]
alias multiplicity.finite_prime_left := FiniteMultiplicity.of_prime_left


theorem le_emultiplicity_iff_replicate_le_normalizedFactors {a b : R} {n : ℕ} (ha : Irreducible a)
    (hb : b ≠ 0) :
    ↑n ≤ emultiplicity a b ↔ replicate n (normalize a) ≤ normalizedFactors b := by
  /-
    R : Type u_2
    inst✝² : CancelCommMonoidWithZero R
    inst✝¹ : UniqueFactorizationMonoid R
    inst✝ : NormalizationMonoid R
    a b : R
    n : Nat
    ha : Irreducible a
    hb : Ne b 0
    ⊢ Iff (LE.le (↑n) (emultiplicity a b)) (LE.le (Multiset.replicate n (normalize …
  -/
  rw [← pow_dvd_iff_le_emultiplicity]
  /-
    R : Type u_2
    inst✝² : CancelCommMonoidWithZero R
    inst✝¹ : UniqueFactorizationMonoid R
    inst✝ : NormalizationMonoid R
    a b : R
    n : Nat
    ha : Irreducible a
    hb : Ne b 0
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.replicate n (normalize a))  …
  -/
  revert b
  /-
    R : Type u_2
    inst✝² : CancelCommMonoidWithZero R
    inst✝¹ : UniqueFactorizationMonoid R
    inst✝ : NormalizationMonoid R
    a : R
    n : Nat
    ha : Irreducible a
    ⊢ ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.replica …
  -/
  induction' n with n ih; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    R : Type u_2
    inst✝² : CancelCommMonoidWithZero R
    inst✝¹ : UniqueFactorizationMonoid R
    inst✝ : NormalizationMonoid R
    a : R
    ha : Irreducible a
    n : Nat
    ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
    ⊢ ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b) (LE.le (Mu …
  -/
  intro b hb
  /-
    case succ
    R : Type u_2
    inst✝² : CancelCommMonoidWithZero R
    inst✝¹ : UniqueFactorizationMonoid R
    inst✝ : NormalizationMonoid R
    a : R
    ha : Irreducible a
    n : Nat
    ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
    b : R
    hb : Ne b 0
    ⊢ Iff (Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b) (LE.le (Multiset.replicate (HA …
  -/
  constructor
    /-
      case succ.mp
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      b : R
      hb : Ne b 0
      ⊢ Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b → LE.le (Multiset.replicate (HAdd.hA …
    -/
  · rintro ⟨c, rfl⟩
    /-
      case succ.mp.intro
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      c : R
      hb : Ne (HMul.hMul (HPow.hPow a (HAdd.hAdd n 1)) c) 0
      ⊢ LE.le (Multiset.replicate (HAdd.hAdd n 1) (normalize a)) (UniqueFactorizatio …
    -/
    rw [Ne, pow_succ', mul_assoc, mul_eq_zero, not_or] at hb
    rw [pow_succ', mul_assoc, normalizedFactors_mul hb.1 hb.2, replicate_succ,
      normalizedFactors_irreducible ha, singleton_add, cons_le_cons_iff, ← ih hb.2]
    /-
      case succ.mp.intro
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      c : R
      hb : And (Not (Eq a 0)) (Not (Eq (HMul.hMul (HPow.hPow a n) c) 0))
      ⊢ Dvd.dvd (HPow.hPow a n) (HMul.hMul (HPow.hPow a n) c)
    -/
    apply Dvd.intro _ rfl
    /-
      🎉 no goals
    -/
    /-
      case succ.mpr
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      b : R
      hb : Ne b 0
      ⊢ LE.le (Multiset.replicate (HAdd.hAdd n 1) (normalize a)) (UniqueFactorizatio …
    -/
  · rw [Multiset.le_iff_exists_add]
    /-
      case succ.mpr
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      b : R
      hb : Ne b 0
      ⊢ (Exists fun u => Eq (UniqueFactorizationMonoid.normalizedFactors b) (HAdd.hA …
    -/
    rintro ⟨u, hu⟩
    /-
      case succ.mpr.intro
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      b : R
      hb : Ne b 0
      u : Multiset R
      hu : Eq (UniqueFactorizationMonoid.normalizedFactors b) (HAdd.hAdd (Multiset.r …
      ⊢ Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) b
    -/
    rw [← (prod_normalizedFactors hb).dvd_iff_dvd_right, hu, prod_add, prod_replicate]
    /-
      case succ.mpr.intro
      R : Type u_2
      inst✝² : CancelCommMonoidWithZero R
      inst✝¹ : UniqueFactorizationMonoid R
      inst✝ : NormalizationMonoid R
      a : R
      ha : Irreducible a
      n : Nat
      ih : ∀ {b : R}, Ne b 0 → Iff (Dvd.dvd (HPow.hPow a n) b) (LE.le (Multiset.repl …
      b : R
      hb : Ne b 0
      u : Multiset R
      hu : Eq (UniqueFactorizationMonoid.normalizedFactors b) (HAdd.hAdd (Multiset.r …
      ⊢ Dvd.dvd (HPow.hPow a (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow (normalize a) (H …
    -/
    exact (Associated.pow_pow <| associated_normalize a).dvd.trans (Dvd.intro u.prod rfl)
    /-
      🎉 no goals
    -/


/-- The multiplicity of an irreducible factor of a nonzero element is exactly the number of times
the normalized factor occurs in the `normalizedFactors`.

See also `count_normalizedFactors_eq` which expands the definition of `multiplicity`
to produce a specification for `count (normalizedFactors _) _`..
-/
theorem emultiplicity_eq_count_normalizedFactors [DecidableEq R] {a b : R} (ha : Irreducible a)
    (hb : b ≠ 0) : emultiplicity a b = (normalizedFactors b).count (normalize a) := by
  /-
    R : Type u_2
    inst✝³ : CancelCommMonoidWithZero R
    inst✝² : UniqueFactorizationMonoid R
    inst✝¹ : NormalizationMonoid R
    inst✝ : DecidableEq R
    a b : R
    ha : Irreducible a
    hb : Ne b 0
    ⊢ Eq (emultiplicity a b) ↑(Multiset.count (normalize a) (UniqueFactorizationMo …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_2
      inst✝³ : CancelCommMonoidWithZero R
      inst✝² : UniqueFactorizationMonoid R
      inst✝¹ : NormalizationMonoid R
      inst✝ : DecidableEq R
      a b : R
      ha : Irreducible a
      hb : Ne b 0
      ⊢ LE.le (emultiplicity a b) ↑(Multiset.count (normalize a) (UniqueFactorizatio …
    -/
  · apply Order.le_of_lt_add_one
    rw [← Nat.cast_one, ← Nat.cast_add, lt_iff_not_ge, ge_iff_le,
      le_emultiplicity_iff_replicate_le_normalizedFactors ha hb, ← le_count_iff_replicate_le]
    /-
      case a.h
      R : Type u_2
      inst✝³ : CancelCommMonoidWithZero R
      inst✝² : UniqueFactorizationMonoid R
      inst✝¹ : NormalizationMonoid R
      inst✝ : DecidableEq R
      a b : R
      ha : Irreducible a
      hb : Ne b 0
      ⊢ Not (LE.le (HAdd.hAdd (Multiset.count (normalize a) (UniqueFactorizationMono …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case a
    R : Type u_2
    inst✝³ : CancelCommMonoidWithZero R
    inst✝² : UniqueFactorizationMonoid R
    inst✝¹ : NormalizationMonoid R
    inst✝ : DecidableEq R
    a b : R
    ha : Irreducible a
    hb : Ne b 0
    ⊢ LE.le (↑(Multiset.count (normalize a) (UniqueFactorizationMonoid.normalizedF …
  -/
  rw [le_emultiplicity_iff_replicate_le_normalizedFactors ha hb, ← le_count_iff_replicate_le]
  /-
    🎉 no goals
  -/


/-- The number of times an irreducible factor `p` appears in `normalizedFactors x` is defined by
the number of times it divides `x`.

See also `multiplicity_eq_count_normalizedFactors` if `n` is given by `multiplicity p x`.
-/
theorem count_normalizedFactors_eq [DecidableEq R] {p x : R} (hp : Irreducible p)
    (hnorm : normalize p = p) {n : ℕ} (hle : p ^ n ∣ x) (hlt : ¬p ^ (n + 1) ∣ x) :
    (normalizedFactors x).count p = n := by classical
  by_cases hx0 : x = 0
  · simp [hx0] at hlt
  apply Nat.cast_injective (R := ℕ∞)
  convert (emultiplicity_eq_count_normalizedFactors hp hx0).symm
  · exact hnorm.symm
  exact (emultiplicity_eq_coe.mpr ⟨hle, hlt⟩).symm


/-- The number of times an irreducible factor `p` appears in `normalizedFactors x` is defined by
the number of times it divides `x`. This is a slightly more general version of
`UniqueFactorizationMonoid.count_normalizedFactors_eq` that allows `p = 0`.

See also `multiplicity_eq_count_normalizedFactors` if `n` is given by `multiplicity p x`.
-/
theorem count_normalizedFactors_eq' [DecidableEq R] {p x : R} (hp : p = 0 ∨ Irreducible p)
    (hnorm : normalize p = p) {n : ℕ} (hle : p ^ n ∣ x) (hlt : ¬p ^ (n + 1) ∣ x) :
    (normalizedFactors x).count p = n := by
  /-
    R : Type u_2
    inst✝³ : CancelCommMonoidWithZero R
    inst✝² : UniqueFactorizationMonoid R
    inst✝¹ : NormalizationMonoid R
    inst✝ : DecidableEq R
    p x : R
    hp : Or (Eq p 0) (Irreducible p)
    hnorm : Eq (normalize p) p
    n : Nat
    hle : Dvd.dvd (HPow.hPow p n) x
    hlt : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) x)
    ⊢ Eq (Multiset.count p (UniqueFactorizationMonoid.normalizedFactors x)) n
  -/
  rcases hp with (rfl | hp)
    /-
      case inl
      R : Type u_2
      inst✝³ : CancelCommMonoidWithZero R
      inst✝² : UniqueFactorizationMonoid R
      inst✝¹ : NormalizationMonoid R
      inst✝ : DecidableEq R
      x : R
      n : Nat
      hnorm : Eq (normalize 0) 0
      hle : Dvd.dvd (HPow.hPow 0 n) x
      hlt : Not (Dvd.dvd (HPow.hPow 0 (HAdd.hAdd n 1)) x)
      ⊢ Eq (Multiset.count 0 (UniqueFactorizationMonoid.normalizedFactors x)) n
    -/
  · cases n
      /-
        case inl.zero
        R : Type u_2
        inst✝³ : CancelCommMonoidWithZero R
        inst✝² : UniqueFactorizationMonoid R
        inst✝¹ : NormalizationMonoid R
        inst✝ : DecidableEq R
        x : R
        hnorm : Eq (normalize 0) 0
        hle : Dvd.dvd (HPow.hPow 0 0) x
        hlt : Not (Dvd.dvd (HPow.hPow 0 (HAdd.hAdd 0 1)) x)
        ⊢ Eq (Multiset.count 0 (UniqueFactorizationMonoid.normalizedFactors x)) 0
      -/
    · exact count_eq_zero.2 (zero_not_mem_normalizedFactors _)
      /-
        🎉 no goals
      -/
      /-
        case inl.succ
        R : Type u_2
        inst✝³ : CancelCommMonoidWithZero R
        inst✝² : UniqueFactorizationMonoid R
        inst✝¹ : NormalizationMonoid R
        inst✝ : DecidableEq R
        x : R
        hnorm : Eq (normalize 0) 0
        n✝ : Nat
        hle : Dvd.dvd (HPow.hPow 0 (HAdd.hAdd n✝ 1)) x
        hlt : Not (Dvd.dvd (HPow.hPow 0 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)) x)
        ⊢ Eq (Multiset.count 0 (UniqueFactorizationMonoid.normalizedFactors x)) (HAdd. …
      -/
    · rw [zero_pow (Nat.succ_ne_zero _)] at hle hlt
      /-
        case inl.succ
        R : Type u_2
        inst✝³ : CancelCommMonoidWithZero R
        inst✝² : UniqueFactorizationMonoid R
        inst✝¹ : NormalizationMonoid R
        inst✝ : DecidableEq R
        x : R
        hnorm : Eq (normalize 0) 0
        n✝ : Nat
        hle : Dvd.dvd 0 x
        hlt : Not (Dvd.dvd 0 x)
        ⊢ Eq (Multiset.count 0 (UniqueFactorizationMonoid.normalizedFactors x)) (HAdd. …
      -/
      exact absurd hle hlt
      /-
        🎉 no goals
      -/
    /-
      case inr
      R : Type u_2
      inst✝³ : CancelCommMonoidWithZero R
      inst✝² : UniqueFactorizationMonoid R
      inst✝¹ : NormalizationMonoid R
      inst✝ : DecidableEq R
      p x : R
      hnorm : Eq (normalize p) p
      n : Nat
      hle : Dvd.dvd (HPow.hPow p n) x
      hlt : Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd n 1)) x)
      hp : Irreducible p
      ⊢ Eq (Multiset.count p (UniqueFactorizationMonoid.normalizedFactors x)) n
    -/
  · exact count_normalizedFactors_eq hp hnorm hle hlt
    /-
      🎉 no goals
    -/


/-- Deprecated. Use `WfDvdMonoid.max_power_factor` instead. -/
@[deprecated WfDvdMonoid.max_power_factor (since := "2024-03-01")]
theorem max_power_factor {a₀ x : R} (h : a₀ ≠ 0) (hx : Irreducible x) :
    ∃ n : ℕ, ∃ a : R, ¬x ∣ a ∧ a₀ = x ^ n * a := WfDvdMonoid.max_power_factor h hx


