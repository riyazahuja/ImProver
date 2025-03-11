/-- Flooring root of a natural number. This divides the valuation of every prime number rounding
down.

Eg if `n = 2`, `a = 2^3 * 3^2 * 5`, then `floorRoot n a = 2 * 3`.

In order theory terms, this is the upper or right adjoint of the map `a ↦ a ^ n : ℕ → ℕ` where `ℕ`
is ordered by divisibility.

To ensure that the adjunction (`Nat.pow_dvd_iff_dvd_floorRoot`) holds in as many cases as possible,
we special-case the following values:
* `floorRoot 0 a = 0`
* `floorRoot n 0 = 0`
-/
def floorRoot (n a : ℕ) : ℕ :=
  if n = 0 ∨ a = 0 then 0 else a.factorization.prod fun p k ↦ p ^ (k / n)


/-- The RHS is a noncomputable version of `Nat.floorRoot` with better order theoretical
properties. -/
lemma floorRoot_def :
    floorRoot n a = if n = 0 ∨ a = 0 then 0 else (a.factorization ⌊/⌋ n).prod (· ^ ·) := by
  /-
    a n : Nat
    ⊢ Eq (n.floorRoot a) (ite (Or (Eq n 0) (Eq a 0)) 0 ((FloorDiv.floorDiv a.facto …
  -/
                                         /-
                                           🎉 no goals
                                         -/
  unfold floorRoot; split_ifs with h <;> simp [Finsupp.floorDiv_def, prod_mapRange_index pow_zero]
                                         /-
                                           🎉 no goals
                                         -/


                                                                    /-
                                                                      a : Nat
                                                                      ⊢ Eq (Nat.floorRoot 0 a) 0
                                                                    -/
@[simp] lemma floorRoot_zero_left (a : ℕ) : floorRoot 0 a = 0 := by simp [floorRoot]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

                                                                     /-
                                                                       n : Nat
                                                                       ⊢ Eq (n.floorRoot 0) 0
                                                                     -/
@[simp] lemma floorRoot_zero_right (n : ℕ) : floorRoot n 0 = 0 := by simp [floorRoot]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/

@[simp] lemma floorRoot_one_left (a : ℕ) : floorRoot 1 a = a := by
  /-
    a : Nat
    ⊢ Eq (Nat.floorRoot 1 a) a
  -/
                                  /-
                                    🎉 no goals
                                  -/
  simp [floorRoot]; split_ifs <;> simp [*]
                                  /-
                                    🎉 no goals
                                  -/

                                                                         /-
                                                                           n : Nat
                                                                           hn : Ne n 0
                                                                           ⊢ Eq (n.floorRoot 1) 1
                                                                         -/
@[simp] lemma floorRoot_one_right (hn : n ≠ 0) : floorRoot n 1 = 1 := by simp [floorRoot, hn]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma floorRoot_pow_self (hn : n ≠ 0) (a : ℕ) : floorRoot n (a ^ n) = a := by
  /-
    n : Nat
    hn : Ne n 0
    a : Nat
    ⊢ Eq (n.floorRoot (HPow.hPow a n)) a
  -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  simp [floorRoot_def, pos_iff_ne_zero.2, hn]; split_ifs <;> simp [*]
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma floorRoot_ne_zero : floorRoot n a ≠ 0 ↔ n ≠ 0 ∧ a ≠ 0 := by
  /-
    a n : Nat
    ⊢ Iff (Ne (n.floorRoot a) 0) (And (Ne n 0) (Ne a 0))
  -/
  simp +contextual [floorRoot, not_imp_not, not_or]
  /-
    🎉 no goals
  -/


@[simp] lemma floorRoot_eq_zero : floorRoot n a = 0 ↔ n = 0 ∨ a = 0 :=
                                          /-
                                            a n : Nat
                                            ⊢ Iff (Not (And (Ne n 0) (Ne a 0))) (Or (Eq n 0) (Eq a 0))
                                          -/
  floorRoot_ne_zero.not_right.trans <| by simp only [not_and_or, ne_eq, not_not]
                                          /-
                                            🎉 no goals
                                          -/


@[simp] lemma factorization_floorRoot (n a : ℕ) :
    (floorRoot n a).factorization = a.factorization ⌊/⌋ n := by
  /-
    n a : Nat
    ⊢ Eq (n.floorRoot a).factorization (FloorDiv.floorDiv a.factorization n)
  -/
  rw [floorRoot_def]
  /-
    n a : Nat
    ⊢ Eq (ite (Or (Eq n 0) (Eq a 0)) 0 ((FloorDiv.floorDiv a.factorization n).prod …
  -/
  split_ifs with h
    /-
      case pos
      n a : Nat
      h : Or (Eq n 0) (Eq a 0)
      ⊢ Eq (Nat.factorization 0) (FloorDiv.floorDiv a.factorization n)
    -/
                              /-
                                🎉 no goals
                              -/
  · obtain rfl | rfl := h <;> simp
                              /-
                                🎉 no goals
                              -/
  /-
    case neg
    n a : Nat
    h : Not (Or (Eq n 0) (Eq a 0))
    ⊢ Eq ((FloorDiv.floorDiv a.factorization n).prod fun x1 x2 => HPow.hPow x1 x2) …
  -/
  refine prod_pow_factorization_eq_self fun p hp ↦ ?_
  /-
    case neg
    n a : Nat
    h : Not (Or (Eq n 0) (Eq a 0))
    p : Nat
    hp : Membership.mem (FloorDiv.floorDiv a.factorization n).support p
    ⊢ Nat.Prime p
  -/
  have : p.Prime ∧ p ∣ a ∧ ¬a = 0 := by simpa using support_floorDiv_subset hp
  /-
    case neg
    n a : Nat
    h : Not (Or (Eq n 0) (Eq a 0))
    p : Nat
    hp : Membership.mem (FloorDiv.floorDiv a.factorization n).support p
    this : And (Nat.Prime p) (And (Dvd.dvd p a) (Not (Eq a 0)))
    ⊢ Nat.Prime p
  -/
  exact this.1
  /-
    🎉 no goals
  -/


/-- Galois connection between `a ↦ a ^ n : ℕ → ℕ` and `floorRoot n : ℕ → ℕ` where `ℕ` is ordered
by divisibility. -/
lemma pow_dvd_iff_dvd_floorRoot : a ^ n ∣ b ↔ a ∣ floorRoot n b := by
  /-
    a b n : Nat
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) b) (Dvd.dvd a (n.floorRoot b))
  -/
  obtain rfl | hn := eq_or_ne n 0
    /-
      case inl
      a b : Nat
      ⊢ Iff (Dvd.dvd (HPow.hPow a 0) b) (Dvd.dvd a (Nat.floorRoot 0 b))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b n : Nat
    hn : Ne n 0
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) b) (Dvd.dvd a (n.floorRoot b))
  -/
  obtain rfl | hb := eq_or_ne b 0
    /-
      case inr.inl
      a n : Nat
      hn : Ne n 0
      ⊢ Iff (Dvd.dvd (HPow.hPow a n) 0) (Dvd.dvd a (n.floorRoot 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b n : Nat
    hn : Ne n 0
    hb : Ne b 0
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) b) (Dvd.dvd a (n.floorRoot b))
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inr.inr.inl
      b n : Nat
      hn : Ne n 0
      hb : Ne b 0
      ⊢ Iff (Dvd.dvd (HPow.hPow 0 n) b) (Dvd.dvd 0 (n.floorRoot b))
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  rw [← factorization_le_iff_dvd (pow_ne_zero _ ha) hb,
    ← factorization_le_iff_dvd ha (floorRoot_ne_zero.2 ⟨hn, hb⟩), factorization_pow,
    factorization_floorRoot, le_floorDiv_iff_smul_le (β := ℕ →₀ ℕ) (pos_iff_ne_zero.2 hn)]


lemma floorRoot_pow_dvd : floorRoot n a ^ n ∣ a := pow_dvd_iff_dvd_floorRoot.2 dvd_rfl


/-- Ceiling root of a natural number. This divides the valuation of every prime number rounding up.

Eg if `n = 3`, `a = 2^4 * 3^2 * 5`, then `ceilRoot n a = 2^2 * 3 * 5`.

In order theory terms, this is the lower or left adjoint of the map `a ↦ a ^ n : ℕ → ℕ` where `ℕ`
is ordered by divisibility.

To ensure that the adjunction (`Nat.dvd_pow_iff_ceilRoot_dvd`) holds in as many cases as possible,
we special-case the following values:
* `ceilRoot 0 a = 0` (this one is not strictly necessary)
* `ceilRoot n 0 = 0`
-/
def ceilRoot (n a : ℕ) : ℕ :=
  if n = 0 ∨ a = 0 then 0 else a.factorization.prod fun p k ↦ p ^ ((k + n - 1) / n)


/-- The RHS is a noncomputable version of `Nat.ceilRoot` with better order theoretical
properties. -/
lemma ceilRoot_def :
    ceilRoot n a = if n = 0 ∨ a = 0 then 0 else (a.factorization ⌈/⌉ n).prod (· ^ ·) := by
  /-
    a n : Nat
    ⊢ Eq (n.ceilRoot a) (ite (Or (Eq n 0) (Eq a 0)) 0 ((CeilDiv.ceilDiv a.factoriz …
  -/
  unfold ceilRoot
  /-
    a n : Nat
    ⊢ Eq (ite (Or (Eq n 0) (Eq a 0)) 0 (a.factorization.prod fun p k => HPow.hPow  …
  -/
  split_ifs with h <;>
    /-
      case pos
      a n : Nat
      h : Or (Eq n 0) (Eq a 0)
      ⊢ Eq 0 0
    -/
    /-
      🎉 no goals
    -/
    simp [Finsupp.ceilDiv_def, prod_mapRange_index pow_zero, Nat.ceilDiv_eq_add_pred_div]
    /-
      🎉 no goals
    -/


                                                                  /-
                                                                    a : Nat
                                                                    ⊢ Eq (Nat.ceilRoot 0 a) 0
                                                                  -/
@[simp] lemma ceilRoot_zero_left (a : ℕ) : ceilRoot 0 a = 0 := by simp [ceilRoot]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

                                                                   /-
                                                                     n : Nat
                                                                     ⊢ Eq (n.ceilRoot 0) 0
                                                                   -/
@[simp] lemma ceilRoot_zero_right (n : ℕ) : ceilRoot n 0 = 0 := by simp [ceilRoot]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

@[simp] lemma ceilRoot_one_left (a : ℕ) : ceilRoot 1 a = a := by
  /-
    a : Nat
    ⊢ Eq (Nat.ceilRoot 1 a) a
  -/
                                 /-
                                   🎉 no goals
                                 -/
  simp [ceilRoot]; split_ifs <;> simp [*]
                                 /-
                                   🎉 no goals
                                 -/

                                                                       /-
                                                                         n : Nat
                                                                         hn : Ne n 0
                                                                         ⊢ Eq (n.ceilRoot 1) 1
                                                                       -/
@[simp] lemma ceilRoot_one_right (hn : n ≠ 0) : ceilRoot n 1 = 1 := by simp [ceilRoot, hn]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp] lemma ceilRoot_pow_self (hn : n ≠ 0) (a : ℕ) : ceilRoot n (a ^ n) = a := by
  /-
    n : Nat
    hn : Ne n 0
    a : Nat
    ⊢ Eq (n.ceilRoot (HPow.hPow a n)) a
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  simp [ceilRoot_def, pos_iff_ne_zero.2, hn]; split_ifs <;> simp [*]
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma ceilRoot_ne_zero : ceilRoot n a ≠ 0 ↔ n ≠ 0 ∧ a ≠ 0 := by
  /-
    a n : Nat
    ⊢ Iff (Ne (n.ceilRoot a) 0) (And (Ne n 0) (Ne a 0))
  -/
  simp +contextual [ceilRoot_def, not_imp_not, not_or]
  /-
    🎉 no goals
  -/


@[simp] lemma ceilRoot_eq_zero : ceilRoot n a = 0 ↔ n = 0 ∨ a = 0 :=
                                         /-
                                           a n : Nat
                                           ⊢ Iff (Not (And (Ne n 0) (Ne a 0))) (Or (Eq n 0) (Eq a 0))
                                         -/
  ceilRoot_ne_zero.not_right.trans <| by simp only [not_and_or, ne_eq, not_not]
                                         /-
                                           🎉 no goals
                                         -/


@[simp] lemma factorization_ceilRoot (n a : ℕ) :
    (ceilRoot n a).factorization = a.factorization ⌈/⌉ n := by
  /-
    n a : Nat
    ⊢ Eq (n.ceilRoot a).factorization (CeilDiv.ceilDiv a.factorization n)
  -/
  rw [ceilRoot_def]
  /-
    n a : Nat
    ⊢ Eq (ite (Or (Eq n 0) (Eq a 0)) 0 ((CeilDiv.ceilDiv a.factorization n).prod f …
  -/
  split_ifs with h
    /-
      case pos
      n a : Nat
      h : Or (Eq n 0) (Eq a 0)
      ⊢ Eq (Nat.factorization 0) (CeilDiv.ceilDiv a.factorization n)
    -/
                              /-
                                🎉 no goals
                              -/
  · obtain rfl | rfl := h <;> simp
                              /-
                                🎉 no goals
                              -/
  /-
    case neg
    n a : Nat
    h : Not (Or (Eq n 0) (Eq a 0))
    ⊢ Eq ((CeilDiv.ceilDiv a.factorization n).prod fun x1 x2 => HPow.hPow x1 x2).f …
  -/
  refine prod_pow_factorization_eq_self fun p hp ↦ ?_
  /-
    case neg
    n a : Nat
    h : Not (Or (Eq n 0) (Eq a 0))
    p : Nat
    hp : Membership.mem (CeilDiv.ceilDiv a.factorization n).support p
    ⊢ Nat.Prime p
  -/
  have : p.Prime ∧ p ∣ a ∧ ¬a = 0 := by simpa using support_ceilDiv_subset hp
  /-
    case neg
    n a : Nat
    h : Not (Or (Eq n 0) (Eq a 0))
    p : Nat
    hp : Membership.mem (CeilDiv.ceilDiv a.factorization n).support p
    this : And (Nat.Prime p) (And (Dvd.dvd p a) (Not (Eq a 0)))
    ⊢ Nat.Prime p
  -/
  exact this.1
  /-
    🎉 no goals
  -/


/-- Galois connection between `ceilRoot n : ℕ → ℕ` and `a ↦ a ^ n : ℕ → ℕ` where `ℕ` is ordered
by divisibility.

Note that this cannot possibly hold for `n = 0`, regardless of the value of `ceilRoot 0 a`, because
the statement reduces to `a = 1 ↔ ceilRoot 0 a ∣ b`, which is false for eg `a = 0`,
`b = ceilRoot 0 a`. -/
lemma dvd_pow_iff_ceilRoot_dvd (hn : n ≠ 0) : a ∣ b ^ n ↔ ceilRoot n a ∣ b := by
  /-
    a b n : Nat
    hn : Ne n 0
    ⊢ Iff (Dvd.dvd a (HPow.hPow b n)) (Dvd.dvd (n.ceilRoot a) b)
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      b n : Nat
      hn : Ne n 0
      ⊢ Iff (Dvd.dvd 0 (HPow.hPow b n)) (Dvd.dvd (n.ceilRoot 0) b)
    -/
  · aesop
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b n : Nat
    hn : Ne n 0
    ha : Ne a 0
    ⊢ Iff (Dvd.dvd a (HPow.hPow b n)) (Dvd.dvd (n.ceilRoot a) b)
  -/
  obtain rfl | hb := eq_or_ne b 0
    /-
      case inr.inl
      a n : Nat
      hn : Ne n 0
      ha : Ne a 0
      ⊢ Iff (Dvd.dvd a (HPow.hPow 0 n)) (Dvd.dvd (n.ceilRoot a) 0)
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  rw [← factorization_le_iff_dvd ha (pow_ne_zero _ hb),
    ← factorization_le_iff_dvd (ceilRoot_ne_zero.2 ⟨hn, ha⟩) hb, factorization_pow,
    factorization_ceilRoot, ceilDiv_le_iff_le_smul (β := ℕ →₀ ℕ) (pos_iff_ne_zero.2 hn)]


lemma dvd_ceilRoot_pow (hn : n ≠ 0) : a ∣ ceilRoot n a ^ n :=
  (dvd_pow_iff_ceilRoot_dvd hn).2 dvd_rfl


