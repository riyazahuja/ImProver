/-- The finite set of prime factors of an element in a unique factorization monoid. -/
def primeFactors (a : M) : Finset M :=
  (normalizedFactors a).toFinset


theorem _root_.Associated.primeFactors_eq {a b : M} (h : Associated a b) :
    primeFactors a = primeFactors b := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a b : M
    h : Associated a b
    ⊢ Eq (UniqueFactorizationMonoid.primeFactors a) (UniqueFactorizationMonoid.pri …
  -/
  unfold primeFactors
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a b : M
    h : Associated a b
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors a).toFinset (UniqueFactoriza …
  -/
  rw [h.normalizedFactors_eq]
  /-
    🎉 no goals
  -/



/--
Radical of an element `a` in a unique factorization monoid is the product of
the prime factors of `a`.
-/
def radical (a : M) : M :=
  (primeFactors a).prod id


@[simp]
theorem radical_zero_eq : radical (0 : M) = 1 := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    ⊢ Eq (UniqueFactorizationMonoid.radical 0) 1
  -/
  rw [radical, primeFactors, normalizedFactors_zero, Multiset.toFinset_zero, Finset.prod_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem radical_one_eq : radical (1 : M) = 1 := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    ⊢ Eq (UniqueFactorizationMonoid.radical 1) 1
  -/
  rw [radical, primeFactors, normalizedFactors_one, Multiset.toFinset_zero, Finset.prod_empty]
  /-
    🎉 no goals
  -/


theorem radical_eq_of_associated {a b : M} (h : Associated a b) : radical a = radical b := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a b : M
    h : Associated a b
    ⊢ Eq (UniqueFactorizationMonoid.radical a) (UniqueFactorizationMonoid.radical b)
  -/
  unfold radical
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a b : M
    h : Associated a b
    ⊢ Eq ((UniqueFactorizationMonoid.primeFactors a).prod id) ((UniqueFactorizatio …
  -/
  rw [h.primeFactors_eq]
  /-
    🎉 no goals
  -/


theorem radical_of_isUnit {a : M} (h : IsUnit a) : radical a = 1 :=
  (radical_eq_of_associated (associated_one_iff_isUnit.mpr h)).trans radical_one_eq


theorem radical_mul_of_isUnit_left {a u : M} (h : IsUnit u) : radical (u * a) = radical a :=
  radical_eq_of_associated (associated_unit_mul_left _ _ h)


theorem radical_mul_of_isUnit_right {a u : M} (h : IsUnit u) : radical (a * u) = radical a :=
  radical_eq_of_associated (associated_mul_unit_left _ _ h)


theorem primeFactors_pow (a : M) {n : ℕ} (hn : 0 < n) : primeFactors (a ^ n) = primeFactors a := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (UniqueFactorizationMonoid.primeFactors (HPow.hPow a n)) (UniqueFactoriza …
  -/
  simp_rw [primeFactors, normalizedFactors_pow, Multiset.toFinset_nsmul _ _ hn.ne']
  /-
    🎉 no goals
  -/


theorem radical_pow (a : M) {n : Nat} (hn : 0 < n) : radical (a ^ n) = radical a := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (UniqueFactorizationMonoid.radical (HPow.hPow a n)) (UniqueFactorizationM …
  -/
  simp_rw [radical, primeFactors_pow a hn]
  /-
    🎉 no goals
  -/


theorem radical_dvd_self (a : M) : radical a ∣ a := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    ⊢ Dvd.dvd (UniqueFactorizationMonoid.radical a) a
  -/
  by_cases ha : a = 0
    /-
      case pos
      M : Type u_1
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : NormalizationMonoid M
      inst✝ : UniqueFactorizationMonoid M
      a : M
      ha : Eq a 0
      ⊢ Dvd.dvd (UniqueFactorizationMonoid.radical a) a
    -/
  · rw [ha]
    /-
      case pos
      M : Type u_1
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : NormalizationMonoid M
      inst✝ : UniqueFactorizationMonoid M
      a : M
      ha : Eq a 0
      ⊢ Dvd.dvd (UniqueFactorizationMonoid.radical 0) 0
    -/
    apply dvd_zero
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_1
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : NormalizationMonoid M
      inst✝ : UniqueFactorizationMonoid M
      a : M
      ha : Not (Eq a 0)
      ⊢ Dvd.dvd (UniqueFactorizationMonoid.radical a) a
    -/
  · rw [radical, ← Finset.prod_val, ← (prod_normalizedFactors ha).dvd_iff_dvd_right]
    /-
      case neg
      M : Type u_1
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : NormalizationMonoid M
      inst✝ : UniqueFactorizationMonoid M
      a : M
      ha : Not (Eq a 0)
      ⊢ Dvd.dvd (UniqueFactorizationMonoid.primeFactors a).val.prod (UniqueFactoriza …
    -/
    apply Multiset.prod_dvd_prod_of_le
    /-
      case neg.h
      M : Type u_1
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : NormalizationMonoid M
      inst✝ : UniqueFactorizationMonoid M
      a : M
      ha : Not (Eq a 0)
      ⊢ LE.le (UniqueFactorizationMonoid.primeFactors a).val (UniqueFactorizationMon …
    -/
    rw [primeFactors, Multiset.toFinset_val]
    /-
      case neg.h
      M : Type u_1
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : NormalizationMonoid M
      inst✝ : UniqueFactorizationMonoid M
      a : M
      ha : Not (Eq a 0)
      ⊢ LE.le (UniqueFactorizationMonoid.normalizedFactors a).dedup (UniqueFactoriza …
    -/
    apply Multiset.dedup_le
    /-
      🎉 no goals
    -/


theorem radical_of_prime {a : M} (ha : Prime a) : radical a = normalize a := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    ha : Prime a
    ⊢ Eq (UniqueFactorizationMonoid.radical a) (normalize a)
  -/
  rw [radical, primeFactors]
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    ha : Prime a
    ⊢ Eq ((UniqueFactorizationMonoid.normalizedFactors a).toFinset.prod id) (norma …
  -/
  rw [normalizedFactors_irreducible ha.irreducible]
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    ha : Prime a
    ⊢ Eq ((Singleton.singleton (normalize a)).toFinset.prod id) (normalize a)
  -/
  simp only [Multiset.toFinset_singleton, id, Finset.prod_singleton]
  /-
    🎉 no goals
  -/


theorem radical_pow_of_prime {a : M} (ha : Prime a) {n : ℕ} (hn : 0 < n) :
    radical (a ^ n) = normalize a := by
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    ha : Prime a
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (UniqueFactorizationMonoid.radical (HPow.hPow a n)) (normalize a)
  -/
  rw [radical_pow a hn]
  /-
    M : Type u_1
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : NormalizationMonoid M
    inst✝ : UniqueFactorizationMonoid M
    a : M
    ha : Prime a
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (UniqueFactorizationMonoid.radical a) (normalize a)
  -/
  exact radical_of_prime ha
  /-
    🎉 no goals
  -/


theorem radical_ne_zero (a : M) [Nontrivial M] : radical a ≠ 0 := by
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    inst✝² : NormalizationMonoid M
    inst✝¹ : UniqueFactorizationMonoid M
    a : M
    inst✝ : Nontrivial M
    ⊢ Ne (UniqueFactorizationMonoid.radical a) 0
  -/
  rw [radical, ← Finset.prod_val]
  /-
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    inst✝² : NormalizationMonoid M
    inst✝¹ : UniqueFactorizationMonoid M
    a : M
    inst✝ : Nontrivial M
    ⊢ Ne (UniqueFactorizationMonoid.primeFactors a).val.prod 0
  -/
  apply Multiset.prod_ne_zero
  /-
    case h
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    inst✝² : NormalizationMonoid M
    inst✝¹ : UniqueFactorizationMonoid M
    a : M
    inst✝ : Nontrivial M
    ⊢ Not (Membership.mem (UniqueFactorizationMonoid.primeFactors a).val 0)
  -/
  rw [primeFactors]
  /-
    case h
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    inst✝² : NormalizationMonoid M
    inst✝¹ : UniqueFactorizationMonoid M
    a : M
    inst✝ : Nontrivial M
    ⊢ Not (Membership.mem (UniqueFactorizationMonoid.normalizedFactors a).toFinset …
  -/
  simp only [Multiset.toFinset_val, Multiset.mem_dedup]
  /-
    case h
    M : Type u_1
    inst✝³ : CancelCommMonoidWithZero M
    inst✝² : NormalizationMonoid M
    inst✝¹ : UniqueFactorizationMonoid M
    a : M
    inst✝ : Nontrivial M
    ⊢ Not (Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) 0)
  -/
  exact zero_not_mem_normalizedFactors _
  /-
    🎉 no goals
  -/


/-- Coprime elements have disjoint prime factors (as multisets). -/
theorem disjoint_normalizedFactors {a b : R} (hc : IsCoprime a b) :
    Disjoint (normalizedFactors a) (normalizedFactors b) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ⊢ Disjoint (UniqueFactorizationMonoid.normalizedFactors a) (UniqueFactorizatio …
  -/
  rw [Multiset.disjoint_left]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ⊢ ∀ {a_1 : R}, Membership.mem (UniqueFactorizationMonoid.normalizedFactors a)  …
  -/
  intro x hxa hxb
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    x : R
    hxa : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) x
    hxb : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) x
    ⊢ False
  -/
  have x_dvd_a := dvd_of_mem_normalizedFactors hxa
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    x : R
    hxa : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) x
    hxb : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) x
    x_dvd_a : Dvd.dvd x a
    ⊢ False
  -/
  have x_dvd_b := dvd_of_mem_normalizedFactors hxb
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    x : R
    hxa : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) x
    hxb : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) x
    x_dvd_a : Dvd.dvd x a
    x_dvd_b : Dvd.dvd x b
    ⊢ False
  -/
  have xp := prime_of_normalized_factor x hxa
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    x : R
    hxa : Membership.mem (UniqueFactorizationMonoid.normalizedFactors a) x
    hxb : Membership.mem (UniqueFactorizationMonoid.normalizedFactors b) x
    x_dvd_a : Dvd.dvd x a
    x_dvd_b : Dvd.dvd x b
    xp : Prime x
    ⊢ False
  -/
  exact xp.not_unit (hc.isUnit_of_dvd' x_dvd_a x_dvd_b)
  /-
    🎉 no goals
  -/


/-- Coprime elements have disjoint prime factors (as finsets). -/
theorem disjoint_primeFactors {a b : R} (hc : IsCoprime a b) :
    Disjoint (primeFactors a) (primeFactors b) :=
  Multiset.disjoint_toFinset.mpr (disjoint_normalizedFactors hc)


theorem mul_primeFactors_disjUnion {a b : R} (ha : a ≠ 0) (hb : b ≠ 0)
    (hc : IsCoprime a b) :
    primeFactors (a * b) =
    (primeFactors a).disjUnion (primeFactors b) (disjoint_primeFactors hc) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    hc : IsCoprime a b
    ⊢ Eq (UniqueFactorizationMonoid.primeFactors (HMul.hMul a b)) ((UniqueFactoriz …
  -/
  rw [Finset.disjUnion_eq_union]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    hc : IsCoprime a b
    ⊢ Eq (UniqueFactorizationMonoid.primeFactors (HMul.hMul a b)) (Union.union (Un …
  -/
  simp_rw [primeFactors]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    ha : Ne a 0
    hb : Ne b 0
    hc : IsCoprime a b
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HMul.hMul a b)).toFinset (U …
  -/
  rw [normalizedFactors_mul ha hb, Multiset.toFinset_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem radical_neg_one : radical (-1 : R) = 1 :=
  radical_of_isUnit isUnit_one.neg


/-- Radical is multiplicative for coprime elements. -/
theorem radical_mul {a b : R} (hc : IsCoprime a b) :
    radical (a * b) = radical a * radical b := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul a b)) (HMul.hMul (UniqueFac …
  -/
  by_cases ha : a = 0
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : NormalizationMonoid R
      inst✝ : UniqueFactorizationMonoid R
      a b : R
      hc : IsCoprime a b
      ha : Eq a 0
      ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul a b)) (HMul.hMul (UniqueFac …
    -/
  · subst ha; rw [isCoprime_zero_left] at hc
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : NormalizationMonoid R
      inst✝ : UniqueFactorizationMonoid R
      b : R
      hc : IsUnit b
      ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul 0 b)) (HMul.hMul (UniqueFac …
    -/
    simp only [zero_mul, radical_zero_eq, one_mul, radical_of_isUnit hc]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ha : Not (Eq a 0)
    ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul a b)) (HMul.hMul (UniqueFac …
  -/
  by_cases hb : b = 0
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : NormalizationMonoid R
      inst✝ : UniqueFactorizationMonoid R
      a b : R
      hc : IsCoprime a b
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul a b)) (HMul.hMul (UniqueFac …
    -/
  · subst hb; rw [isCoprime_zero_right] at hc
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : NormalizationMonoid R
      inst✝ : UniqueFactorizationMonoid R
      a : R
      ha : Not (Eq a 0)
      hc : IsUnit a
      ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul a 0)) (HMul.hMul (UniqueFac …
    -/
    simp only [mul_zero, radical_zero_eq, mul_one, radical_of_isUnit hc]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Eq (UniqueFactorizationMonoid.radical (HMul.hMul a b)) (HMul.hMul (UniqueFac …
  -/
  simp_rw [radical]
  /-
    case neg
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Eq ((UniqueFactorizationMonoid.primeFactors (HMul.hMul a b)).prod id) (HMul. …
  -/
  rw [mul_primeFactors_disjUnion ha hb hc]
  /-
    case neg
    R : Type u_1
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : NormalizationMonoid R
    inst✝ : UniqueFactorizationMonoid R
    a b : R
    hc : IsCoprime a b
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Eq (((UniqueFactorizationMonoid.primeFactors a).disjUnion (UniqueFactorizati …
  -/
  rw [Finset.prod_disjUnion (disjoint_primeFactors hc)]
  /-
    🎉 no goals
  -/


theorem radical_neg {a : R} : radical (-a) = radical a :=
  radical_eq_of_associated Associated.rfl.neg_left


/-- Division of an element by its radical in an Euclidean domain. -/
def divRadical (a : E) : E := a / radical a


theorem radical_mul_divRadical (a : E) : radical a * divRadical a = a := by
  rw [divRadical, ← EuclideanDomain.mul_div_assoc _ (radical_dvd_self a),
    mul_div_cancel_left₀ _ (radical_ne_zero a)]


theorem divRadical_mul_radical (a : E) : divRadical a * radical a = a := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a : E
    ⊢ Eq (HMul.hMul (EuclideanDomain.divRadical a) (UniqueFactorizationMonoid.radi …
  -/
  rw [mul_comm]
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a : E
    ⊢ Eq (HMul.hMul (UniqueFactorizationMonoid.radical a) (EuclideanDomain.divRadi …
  -/
  exact radical_mul_divRadical a
  /-
    🎉 no goals
  -/


theorem divRadical_ne_zero {a : E} (ha : a ≠ 0) : divRadical a ≠ 0 := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a : E
    ha : Ne a 0
    ⊢ Ne (EuclideanDomain.divRadical a) 0
  -/
  rw [← radical_mul_divRadical a] at ha
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a : E
    ha : Ne (HMul.hMul (UniqueFactorizationMonoid.radical a) (EuclideanDomain.divR …
    ⊢ Ne (EuclideanDomain.divRadical a) 0
  -/
  exact right_ne_zero_of_mul ha
  /-
    🎉 no goals
  -/


theorem divRadical_isUnit {u : E} (hu : IsUnit u) : IsUnit (divRadical u) := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    u : E
    hu : IsUnit u
    ⊢ IsUnit (EuclideanDomain.divRadical u)
  -/
  rwa [divRadical, radical_of_isUnit hu, EuclideanDomain.div_one]
  /-
    🎉 no goals
  -/


theorem eq_divRadical {a x : E} (h : radical a * x = a) : x = divRadical a := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a x : E
    h : Eq (HMul.hMul (UniqueFactorizationMonoid.radical a) x) a
    ⊢ Eq x (EuclideanDomain.divRadical a)
  -/
  apply EuclideanDomain.eq_div_of_mul_eq_left (radical_ne_zero a)
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a x : E
    h : Eq (HMul.hMul (UniqueFactorizationMonoid.radical a) x) a
    ⊢ Eq (HMul.hMul x (UniqueFactorizationMonoid.radical a)) a
  -/
  rwa [mul_comm]
  /-
    🎉 no goals
  -/


theorem divRadical_mul {a b : E} (hab : IsCoprime a b) :
    divRadical (a * b) = divRadical a * divRadical b := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a b : E
    hab : IsCoprime a b
    ⊢ Eq (EuclideanDomain.divRadical (HMul.hMul a b)) (HMul.hMul (EuclideanDomain. …
  -/
  symm; apply eq_divRadical
  /-
    case h
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a b : E
    hab : IsCoprime a b
    ⊢ Eq (HMul.hMul (UniqueFactorizationMonoid.radical (HMul.hMul a b)) (HMul.hMul …
  -/
  rw [radical_mul hab]
  /-
    case h
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a b : E
    hab : IsCoprime a b
    ⊢ Eq (HMul.hMul (HMul.hMul (UniqueFactorizationMonoid.radical a) (UniqueFactor …
  -/
  rw [mul_mul_mul_comm, radical_mul_divRadical, radical_mul_divRadical]
  /-
    🎉 no goals
  -/


theorem divRadical_dvd_self (a : E) : divRadical a ∣ a := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a : E
    ⊢ Dvd.dvd (EuclideanDomain.divRadical a) a
  -/
  exact Dvd.intro (radical a) (divRadical_mul_radical a)
  /-
    🎉 no goals
  -/


theorem _root_.IsCoprime.divRadical {a b : E} (h : IsCoprime a b) :
    IsCoprime (divRadical a) (divRadical b) := by
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a b : E
    h : IsCoprime a b
    ⊢ IsCoprime (EuclideanDomain.divRadical a) (EuclideanDomain.divRadical b)
  -/
  rw [← radical_mul_divRadical a] at h
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a b : E
    h : IsCoprime (HMul.hMul (UniqueFactorizationMonoid.radical a) (EuclideanDomai …
    ⊢ IsCoprime (EuclideanDomain.divRadical a) (EuclideanDomain.divRadical b)
  -/
  rw [← radical_mul_divRadical b] at h
  /-
    E : Type u_1
    inst✝² : EuclideanDomain E
    inst✝¹ : NormalizationMonoid E
    inst✝ : UniqueFactorizationMonoid E
    a b : E
    h : IsCoprime (HMul.hMul (UniqueFactorizationMonoid.radical a) (EuclideanDomai …
    ⊢ IsCoprime (EuclideanDomain.divRadical a) (EuclideanDomain.divRadical b)
  -/
  exact h.of_mul_left_right.of_mul_right_right
  /-
    🎉 no goals
  -/


