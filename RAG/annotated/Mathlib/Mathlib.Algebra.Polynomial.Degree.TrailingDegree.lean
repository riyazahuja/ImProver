/-- `trailingDegree p` is the multiplicity of `x` in the polynomial `p`, i.e. the smallest
`X`-exponent in `p`.
`trailingDegree p = some n` when `p ≠ 0` and `n` is the smallest power of `X` that appears
in `p`, otherwise
`trailingDegree 0 = ⊤`. -/
def trailingDegree (p : R[X]) : ℕ∞ :=
  p.support.min


theorem trailingDegree_lt_wf : WellFounded fun p q : R[X] => trailingDegree p < trailingDegree q :=
  InvImage.wf trailingDegree wellFounded_lt


/-- `natTrailingDegree p` forces `trailingDegree p` to `ℕ`, by defining
`natTrailingDegree ⊤ = 0`. -/
def natTrailingDegree (p : R[X]) : ℕ :=
  ENat.toNat (trailingDegree p)


/-- `trailingCoeff p` gives the coefficient of the smallest power of `X` in `p`-/
def trailingCoeff (p : R[X]) : R :=
  coeff p (natTrailingDegree p)


/-- a polynomial is `monic_at` if its trailing coefficient is 1 -/
def TrailingMonic (p : R[X]) :=
  trailingCoeff p = (1 : R)


theorem TrailingMonic.def : TrailingMonic p ↔ trailingCoeff p = 1 :=
  Iff.rfl


instance TrailingMonic.decidable [DecidableEq R] : Decidable (TrailingMonic p) :=
  inferInstanceAs <| Decidable (trailingCoeff p = (1 : R))


@[simp]
theorem TrailingMonic.trailingCoeff {p : R[X]} (hp : p.TrailingMonic) : trailingCoeff p = 1 :=
  hp


@[simp]
theorem trailingDegree_zero : trailingDegree (0 : R[X]) = ⊤ :=
  rfl


@[simp]
theorem trailingCoeff_zero : trailingCoeff (0 : R[X]) = 0 :=
  rfl


@[simp]
theorem natTrailingDegree_zero : natTrailingDegree (0 : R[X]) = 0 :=
  rfl


@[simp]
theorem trailingDegree_eq_top : trailingDegree p = ⊤ ↔ p = 0 :=
                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : Semiring R
                                                                      p : Polynomial R
                                                                      h : Eq p 0
                                                                      ⊢ Eq p.trailingDegree Top.top
                                                                    -/
  ⟨fun h => support_eq_empty.1 (Finset.min_eq_top.1 h), fun h => by simp [h]⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem trailingDegree_eq_natTrailingDegree (hp : p ≠ 0) :
    trailingDegree p = (natTrailingDegree p : ℕ∞) :=
  .symm <| ENat.coe_toNat <| mt trailingDegree_eq_top.1 hp


theorem trailingDegree_eq_iff_natTrailingDegree_eq {p : R[X]} {n : ℕ} (hp : p ≠ 0) :
    p.trailingDegree = n ↔ p.natTrailingDegree = n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hp : Ne p 0
    ⊢ Iff (Eq p.trailingDegree ↑n) (Eq p.natTrailingDegree n)
  -/
  rw [trailingDegree_eq_natTrailingDegree hp, Nat.cast_inj]
  /-
    🎉 no goals
  -/


theorem trailingDegree_eq_iff_natTrailingDegree_eq_of_pos {p : R[X]} {n : ℕ} (hn : n ≠ 0) :
    p.trailingDegree = n ↔ p.natTrailingDegree = n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq p.trailingDegree ↑n) (Eq p.natTrailingDegree n)
  -/
  rw [natTrailingDegree, ENat.toNat_eq_iff hn]
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_eq_of_trailingDegree_eq_some {p : R[X]} {n : ℕ}
    (h : trailingDegree p = n) : natTrailingDegree p = n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : Eq p.trailingDegree ↑n
    ⊢ Eq p.natTrailingDegree n
  -/
  simp [natTrailingDegree, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem natTrailingDegree_le_trailingDegree : ↑(natTrailingDegree p) ≤ trailingDegree p :=
  ENat.coe_toNat_le_self _


theorem natTrailingDegree_eq_of_trailingDegree_eq [Semiring S] {q : S[X]}
    (h : trailingDegree p = trailingDegree q) : natTrailingDegree p = natTrailingDegree q := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    q : Polynomial S
    h : Eq p.trailingDegree q.trailingDegree
    ⊢ Eq p.natTrailingDegree q.natTrailingDegree
  -/
  unfold natTrailingDegree
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    q : Polynomial S
    h : Eq p.trailingDegree q.trailingDegree
    ⊢ Eq p.trailingDegree.toNat q.trailingDegree.toNat
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem trailingDegree_le_of_ne_zero (h : coeff p n ≠ 0) : trailingDegree p ≤ n :=
  min_le (mem_support_iff.2 h)


theorem natTrailingDegree_le_of_ne_zero (h : coeff p n ≠ 0) : natTrailingDegree p ≤ n :=
  ENat.toNat_le_of_le_coe <| trailingDegree_le_of_ne_zero h


@[simp] lemma coeff_natTrailingDegree_eq_zero : coeff p p.natTrailingDegree = 0 ↔ p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq (p.coeff p.natTrailingDegree) 0) (Eq p 0)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ Eq (p.coeff p.natTrailingDegree) 0 → Eq p 0
    -/
  · rintro h
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : Eq (p.coeff p.natTrailingDegree) 0
      ⊢ Eq p 0
    -/
    by_contra hp
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : Eq (p.coeff p.natTrailingDegree) 0
      hp : Not (Eq p 0)
      ⊢ False
    -/
    obtain ⟨n, hpn, hn⟩ := by simpa using min_mem_image_coe <| support_nonempty.2 hp
    /-
      case mp.intro.intro
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : Eq (p.coeff p.natTrailingDegree) 0
      hp : Not (Eq p 0)
      n : Nat
      hpn : Not (Eq (p.coeff n) 0)
      hn : Eq (↑n) p.support.min
      ⊢ False
    -/
    obtain rfl := (trailingDegree_eq_iff_natTrailingDegree_eq hp).1 hn.symm
    /-
      case mp.intro.intro
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : Eq (p.coeff p.natTrailingDegree) 0
      hp : Not (Eq p 0)
      hpn : Not (Eq (p.coeff p.natTrailingDegree) 0)
      hn : Eq (↑p.natTrailingDegree) p.support.min
      ⊢ False
    -/
    exact hpn h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ Eq p 0 → Eq (p.coeff p.natTrailingDegree) 0
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      ⊢ Eq (Polynomial.coeff 0 (Polynomial.natTrailingDegree 0)) 0
    -/
    simp
    /-
      🎉 no goals
    -/


lemma coeff_natTrailingDegree_ne_zero : coeff p p.natTrailingDegree ≠ 0 ↔ p ≠ 0 :=
  coeff_natTrailingDegree_eq_zero.not


@[simp]
lemma trailingDegree_eq_zero : trailingDegree p = 0 ↔ coeff p 0 ≠ 0 :=
  Finset.min_eq_bot.trans mem_support_iff


@[simp] lemma natTrailingDegree_eq_zero : natTrailingDegree p = 0 ↔ p = 0 ∨ coeff p 0 ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq p.natTrailingDegree 0) (Or (Eq p 0) (Ne (p.coeff 0) 0))
  -/
  simp [natTrailingDegree, or_comm]
  /-
    🎉 no goals
  -/


lemma natTrailingDegree_ne_zero : natTrailingDegree p ≠ 0 ↔ p ≠ 0 ∧ coeff p 0 = 0 :=
                                            /-
                                              R : Type u
                                              inst✝ : Semiring R
                                              p : Polynomial R
                                              ⊢ Iff (Not (Or (Eq p 0) (Ne (p.coeff 0) 0))) (And (Ne p 0) (Eq (p.coeff 0) 0))
                                            -/
  natTrailingDegree_eq_zero.not.trans <| by rw [not_or, not_ne_iff]
                                            /-
                                              🎉 no goals
                                            -/


lemma trailingDegree_ne_zero : trailingDegree p ≠ 0 ↔ coeff p 0 = 0 :=
  trailingDegree_eq_zero.not_left


@[simp] theorem trailingDegree_le_trailingDegree (h : coeff q (natTrailingDegree p) ≠ 0) :
    trailingDegree q ≤ trailingDegree p :=
  (trailingDegree_le_of_ne_zero h).trans natTrailingDegree_le_trailingDegree


theorem trailingDegree_ne_of_natTrailingDegree_ne {n : ℕ} :
    p.natTrailingDegree ≠ n → trailingDegree p ≠ n :=
                 /-
                   R : Type u
                   inst✝ : Semiring R
                   p : Polynomial R
                   n : Nat
                   h : Eq p.trailingDegree ↑n
                   ⊢ Eq p.natTrailingDegree n
                 -/
  mt fun h => by rw [natTrailingDegree, h, ENat.toNat_coe]
                 /-
                   🎉 no goals
                 -/


theorem natTrailingDegree_le_of_trailingDegree_le {n : ℕ} {hp : p ≠ 0}
    (H : (n : ℕ∞) ≤ trailingDegree p) : n ≤ natTrailingDegree p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hp : Ne p 0
    H : LE.le (↑n) p.trailingDegree
    ⊢ LE.le n p.natTrailingDegree
  -/
  rwa [trailingDegree_eq_natTrailingDegree hp, Nat.cast_le] at H
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_le_natTrailingDegree (hq : q ≠ 0)
    (hpq : p.trailingDegree ≤ q.trailingDegree) : p.natTrailingDegree ≤ q.natTrailingDegree :=
                                /-
                                  R : Type u
                                  inst✝ : Semiring R
                                  p q : Polynomial R
                                  hq : Ne q 0
                                  hpq : LE.le p.trailingDegree q.trailingDegree
                                  ⊢ Ne q.trailingDegree Top.top
                                -/
  ENat.toNat_le_toNat hpq <| by simpa
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem trailingDegree_monomial (ha : a ≠ 0) : trailingDegree (monomial n a) = n := by
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq ((Polynomial.monomial n) a).trailingDegree ↑n
  -/
  rw [trailingDegree, support_monomial n ha, min_singleton]
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq ↑n ↑n
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_monomial (ha : a ≠ 0) : natTrailingDegree (monomial n a) = n := by
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq ((Polynomial.monomial n) a).natTrailingDegree n
  -/
  rw [natTrailingDegree, trailingDegree_monomial ha]
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ Eq (↑n).toNat n
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_monomial_le : natTrailingDegree (monomial n a) ≤ n :=
  letI := Classical.decEq R
                        /-
                          R : Type u
                          a : R
                          n : Nat
                          inst✝ : Semiring R
                          this : DecidableEq R := Classical.decEq R
                          ha : Eq a 0
                          ⊢ LE.le ((Polynomial.monomial n) a).natTrailingDegree n
                        -/
  if ha : a = 0 then by simp [ha] else (natTrailingDegree_monomial ha).le
                        /-
                          🎉 no goals
                        -/


theorem le_trailingDegree_monomial : ↑n ≤ trailingDegree (monomial n a) :=
  letI := Classical.decEq R
                        /-
                          R : Type u
                          a : R
                          n : Nat
                          inst✝ : Semiring R
                          this : DecidableEq R := Classical.decEq R
                          ha : Eq a 0
                          ⊢ LE.le (↑n) ((Polynomial.monomial n) a).trailingDegree
                        -/
  if ha : a = 0 then by simp [ha] else (trailingDegree_monomial ha).ge
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem trailingDegree_C (ha : a ≠ 0) : trailingDegree (C a) = (0 : ℕ∞) :=
  trailingDegree_monomial ha


theorem le_trailingDegree_C : (0 : ℕ∞) ≤ trailingDegree (C a) :=
  le_trailingDegree_monomial


theorem trailingDegree_one_le : (0 : ℕ∞) ≤ trailingDegree (1 : R[X]) := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ LE.le 0 (Polynomial.trailingDegree 1)
  -/
  rw [← C_1]
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ LE.le 0 (Polynomial.C 1).trailingDegree
  -/
  exact le_trailingDegree_C
  /-
    🎉 no goals
  -/


@[simp]
theorem natTrailingDegree_C (a : R) : natTrailingDegree (C a) = 0 :=
  nonpos_iff_eq_zero.1 natTrailingDegree_monomial_le


@[simp]
theorem natTrailingDegree_one : natTrailingDegree (1 : R[X]) = 0 :=
  natTrailingDegree_C 1


@[simp]
theorem natTrailingDegree_natCast (n : ℕ) : natTrailingDegree (n : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (↑n).natTrailingDegree 0
  -/
  simp only [← C_eq_natCast, natTrailingDegree_C]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias natTrailingDegree_nat_cast := natTrailingDegree_natCast


@[simp]
theorem trailingDegree_C_mul_X_pow (n : ℕ) (ha : a ≠ 0) : trailingDegree (C a * X ^ n) = n := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    n : Nat
    ha : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).trailingDegree ↑n
  -/
  rw [C_mul_X_pow_eq_monomial, trailingDegree_monomial ha]
  /-
    🎉 no goals
  -/


theorem le_trailingDegree_C_mul_X_pow (n : ℕ) (a : R) :
    (n : ℕ∞) ≤ trailingDegree (C a * X ^ n) := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ LE.le (↑n) (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).trailingD …
  -/
  rw [C_mul_X_pow_eq_monomial]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ LE.le (↑n) ((Polynomial.monomial n) a).trailingDegree
  -/
  exact le_trailingDegree_monomial
  /-
    🎉 no goals
  -/


theorem coeff_eq_zero_of_lt_trailingDegree (h : (n : ℕ∞) < trailingDegree p) : coeff p n = 0 :=
  Classical.not_not.1 (mt trailingDegree_le_of_ne_zero (not_le_of_gt h))


theorem coeff_eq_zero_of_lt_natTrailingDegree {p : R[X]} {n : ℕ} (h : n < p.natTrailingDegree) :
    p.coeff n = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt n p.natTrailingDegree
    ⊢ Eq (p.coeff n) 0
  -/
  apply coeff_eq_zero_of_lt_trailingDegree
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt n p.natTrailingDegree
    ⊢ LT.lt (↑n) p.trailingDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : LT.lt n p.natTrailingDegree
      hp : Eq p 0
      ⊢ LT.lt (↑n) p.trailingDegree
    -/
  · rw [hp, trailingDegree_zero]
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : LT.lt n p.natTrailingDegree
      hp : Eq p 0
      ⊢ LT.lt (↑n) Top.top
    -/
    exact WithTop.coe_lt_top n
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : LT.lt n p.natTrailingDegree
      hp : Not (Eq p 0)
      ⊢ LT.lt (↑n) p.trailingDegree
    -/
  · rw [trailingDegree_eq_natTrailingDegree hp]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : LT.lt n p.natTrailingDegree
      hp : Not (Eq p 0)
      ⊢ LT.lt ↑n ↑p.natTrailingDegree
    -/
    exact WithTop.coe_lt_coe.2 h
    /-
      🎉 no goals
    -/


@[simp]
theorem coeff_natTrailingDegree_pred_eq_zero {p : R[X]} {hp : (0 : ℕ∞) < natTrailingDegree p} :
    p.coeff (p.natTrailingDegree - 1) = 0 :=
  coeff_eq_zero_of_lt_natTrailingDegree <|
    Nat.sub_lt (WithTop.coe_pos.mp hp) Nat.one_pos


theorem le_trailingDegree_X_pow (n : ℕ) : (n : ℕ∞) ≤ trailingDegree (X ^ n : R[X]) := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ LE.le (↑n) (HPow.hPow Polynomial.X n).trailingDegree
  -/
  simpa only [C_1, one_mul] using le_trailingDegree_C_mul_X_pow n (1 : R)
  /-
    🎉 no goals
  -/


theorem le_trailingDegree_X : (1 : ℕ∞) ≤ trailingDegree (X : R[X]) :=
  le_trailingDegree_monomial


theorem natTrailingDegree_X_le : (X : R[X]).natTrailingDegree ≤ 1 :=
  natTrailingDegree_monomial_le


@[simp]
theorem trailingCoeff_eq_zero : trailingCoeff p = 0 ↔ p = 0 :=
  ⟨fun h =>
    _root_.by_contradiction fun hp =>
      mt mem_support_iff.1 (Classical.not_not.2 h)
        (mem_of_min (trailingDegree_eq_natTrailingDegree hp)),
    fun h => h.symm ▸ leadingCoeff_zero⟩


theorem trailingCoeff_nonzero_iff_nonzero : trailingCoeff p ≠ 0 ↔ p ≠ 0 :=
  not_congr trailingCoeff_eq_zero


theorem natTrailingDegree_mem_support_of_nonzero : p ≠ 0 → natTrailingDegree p ∈ p.support :=
  mem_support_iff.mpr ∘ trailingCoeff_nonzero_iff_nonzero.mpr


theorem natTrailingDegree_le_of_mem_supp (a : ℕ) : a ∈ p.support → natTrailingDegree p ≤ a :=
  natTrailingDegree_le_of_ne_zero ∘ mem_support_iff.mp


theorem natTrailingDegree_eq_support_min' (h : p ≠ 0) :
    natTrailingDegree p = p.support.min' (nonempty_support_iff.mpr h) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne p 0
    ⊢ Eq p.natTrailingDegree (p.support.min' ⋯)
  -/
  rw [natTrailingDegree, trailingDegree, ← Finset.coe_min', ENat.some_eq_coe, ENat.toNat_coe]
  /-
    🎉 no goals
  -/


theorem le_natTrailingDegree (hp : p ≠ 0) (hn : ∀ m < n, p.coeff m = 0) :
    n ≤ p.natTrailingDegree := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    hn : ∀ (m : Nat), LT.lt m n → Eq (p.coeff m) 0
    ⊢ LE.le n p.natTrailingDegree
  -/
  rw [natTrailingDegree_eq_support_min' hp]
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    hn : ∀ (m : Nat), LT.lt m n → Eq (p.coeff m) 0
    ⊢ LE.le n (p.support.min' ⋯)
  -/
  exact Finset.le_min' _ _ _ fun m hm => not_lt.1 fun hmn => mem_support_iff.1 hm <| hn _ hmn
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_le_natDegree (p : R[X]) : p.natTrailingDegree ≤ p.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ LE.le p.natTrailingDegree p.natDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p 0
      ⊢ LE.le p.natTrailingDegree p.natDegree
    -/
  · rw [hp, natDegree_zero, natTrailingDegree_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Not (Eq p 0)
      ⊢ LE.le p.natTrailingDegree p.natDegree
    -/
  · exact le_natDegree_of_ne_zero (mt trailingCoeff_eq_zero.mp hp)
    /-
      🎉 no goals
    -/


theorem natTrailingDegree_mul_X_pow {p : R[X]} (hp : p ≠ 0) (n : ℕ) :
    (p * X ^ n).natTrailingDegree = p.natTrailingDegree + n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    n : Nat
    ⊢ Eq (HMul.hMul p (HPow.hPow Polynomial.X n)).natTrailingDegree (HAdd.hAdd p.n …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      n : Nat
      ⊢ LE.le (HMul.hMul p (HPow.hPow Polynomial.X n)).natTrailingDegree (HAdd.hAdd  …
    -/
  · refine natTrailingDegree_le_of_ne_zero fun h => mt trailingCoeff_eq_zero.mp hp ?_
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      n : Nat
      h : Eq ((HMul.hMul p (HPow.hPow Polynomial.X n)).coeff (HAdd.hAdd p.natTrailin …
      ⊢ Eq p.trailingCoeff 0
    -/
    rwa [trailingCoeff, ← coeff_mul_X_pow]
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      n : Nat
      ⊢ LE.le (HAdd.hAdd p.natTrailingDegree n) (HMul.hMul p (HPow.hPow Polynomial.X …
    -/
  · rw [natTrailingDegree_eq_support_min' fun h => hp (mul_X_pow_eq_zero h), Finset.le_min'_iff]
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      n : Nat
      ⊢ ∀ (y : Nat), Membership.mem (HMul.hMul p (HPow.hPow Polynomial.X n)).support …
    -/
    intro y hy
    have key : n ≤ y := by
      rw [mem_support_iff, coeff_mul_X_pow'] at hy
      exact by_contra fun h => hy (if_neg h)
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      n y : Nat
      hy : Membership.mem (HMul.hMul p (HPow.hPow Polynomial.X n)).support y
      key : LE.le n y
      ⊢ LE.le (HAdd.hAdd p.natTrailingDegree n) y
    -/
    rw [mem_support_iff, coeff_mul_X_pow', if_pos key] at hy
    /-
      case a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      n y : Nat
      hy : Ne (p.coeff (HSub.hSub y n)) 0
      key : LE.le n y
      ⊢ LE.le (HAdd.hAdd p.natTrailingDegree n) y
    -/
    exact (le_tsub_iff_right key).mp (natTrailingDegree_le_of_ne_zero hy)
    /-
      🎉 no goals
    -/


theorem le_trailingDegree_mul : p.trailingDegree + q.trailingDegree ≤ (p * q).trailingDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ LE.le (HAdd.hAdd p.trailingDegree q.trailingDegree) (HMul.hMul p q).trailing …
  -/
  refine Finset.le_min fun n hn => ?_
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    hn : Membership.mem (HMul.hMul p q).support n
    ⊢ LE.le (HAdd.hAdd p.trailingDegree q.trailingDegree) ↑n
  -/
  rw [mem_support_iff, coeff_mul] at hn
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    hn : Ne ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HMul.hMul (p.coe …
    ⊢ LE.le (HAdd.hAdd p.trailingDegree q.trailingDegree) ↑n
  -/
  obtain ⟨⟨i, j⟩, hij, hpq⟩ := exists_ne_zero_of_sum_ne_zero hn
  refine
    (add_le_add (min_le (mem_support_iff.mpr (left_ne_zero_of_mul hpq)))
          (min_le (mem_support_iff.mpr (right_ne_zero_of_mul hpq)))).trans
      (le_of_eq ?_)
  /-
    case intro.mk.intro
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    hn : Ne ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HMul.hMul (p.coe …
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) { fst := i, snd : …
    hpq : Ne (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, sn …
    ⊢ Eq (HAdd.hAdd ↑{ fst := i, snd := j }.1 ↑{ fst := i, snd := j }.2) ↑n
  -/
  rwa [← WithTop.coe_add, WithTop.coe_eq_coe, ← mem_antidiagonal]
  /-
    🎉 no goals
  -/


theorem le_natTrailingDegree_mul (h : p * q ≠ 0) :
    p.natTrailingDegree + q.natTrailingDegree ≤ (p * q).natTrailingDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p q) 0
    ⊢ LE.le (HAdd.hAdd p.natTrailingDegree q.natTrailingDegree) (HMul.hMul p q).na …
  -/
  have hp : p ≠ 0 := fun hp => h (by rw [hp, zero_mul])
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p q) 0
    hp : Ne p 0
    ⊢ LE.le (HAdd.hAdd p.natTrailingDegree q.natTrailingDegree) (HMul.hMul p q).na …
  -/
  have hq : q ≠ 0 := fun hq => h (by rw [hq, mul_zero])
  -- Porting note: Needed to account for different coercion behaviour & add the lemma below
  have : ∀ (p : R[X]), WithTop.some (natTrailingDegree p) = Nat.cast (natTrailingDegree p) :=
    fun p ↦ rfl
  rw [← WithTop.coe_le_coe, WithTop.coe_add, this p, this q, this (p * q),
    ← trailingDegree_eq_natTrailingDegree hp, ← trailingDegree_eq_natTrailingDegree hq,
    ← trailingDegree_eq_natTrailingDegree h]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p q) 0
    hp : Ne p 0
    hq : Ne q 0
    this : ∀ (p : Polynomial R), Eq ↑p.natTrailingDegree ↑p.natTrailingDegree
    ⊢ LE.le (HAdd.hAdd p.trailingDegree q.trailingDegree) (HMul.hMul p q).trailing …
  -/
  exact le_trailingDegree_mul
  /-
    🎉 no goals
  -/


theorem coeff_mul_natTrailingDegree_add_natTrailingDegree : (p * q).coeff
    (p.natTrailingDegree + q.natTrailingDegree) = p.trailingCoeff * q.trailingCoeff := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ Eq ((HMul.hMul p q).coeff (HAdd.hAdd p.natTrailingDegree q.natTrailingDegree …
  -/
  rw [coeff_mul]
  refine
    Finset.sum_eq_single (p.natTrailingDegree, q.natTrailingDegree) ?_ fun h =>
      (h (mem_antidiagonal.mpr rfl)).elim
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
  -/
  rintro ⟨i, j⟩ h₁ h₂
  /-
    case mk
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    i j : Nat
    h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natTrail …
    h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
    ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
  -/
  rw [mem_antidiagonal] at h₁
  /-
    case mk
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    i j : Nat
    h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
    h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
    ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
  -/
  by_cases hi : i < p.natTrailingDegree
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      i j : Nat
      h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
      h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
      hi : LT.lt i p.natTrailingDegree
      ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
    -/
  · rw [coeff_eq_zero_of_lt_natTrailingDegree hi, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    i j : Nat
    h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
    h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
    hi : Not (LT.lt i p.natTrailingDegree)
    ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
  -/
  by_cases hj : j < q.natTrailingDegree
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      i j : Nat
      h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
      h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
      hi : Not (LT.lt i p.natTrailingDegree)
      hj : LT.lt j q.natTrailingDegree
      ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
    -/
  · rw [coeff_eq_zero_of_lt_natTrailingDegree hj, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    i j : Nat
    h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
    h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
    hi : Not (LT.lt i p.natTrailingDegree)
    hj : Not (LT.lt j q.natTrailingDegree)
    ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
  -/
  rw [not_lt] at hi hj
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    i j : Nat
    h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
    h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
    hi : LE.le p.natTrailingDegree i
    hj : LE.le q.natTrailingDegree j
    ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
  -/
  refine (h₂ (Prod.ext_iff.mpr ?_).symm).elim
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    i j : Nat
    h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
    h₂ : Ne { fst := i, snd := j } { fst := p.natTrailingDegree, snd := q.natTrail …
    hi : LE.le p.natTrailingDegree i
    hj : LE.le q.natTrailingDegree j
    ⊢ And (Eq { fst := p.natTrailingDegree, snd := q.natTrailingDegree }.1 { fst : …
  -/
  exact (add_eq_add_iff_eq_and_eq hi hj).mp h₁.symm
  /-
    🎉 no goals
  -/


theorem trailingDegree_mul' (h : p.trailingCoeff * q.trailingCoeff ≠ 0) :
    (p * q).trailingDegree = p.trailingDegree + q.trailingDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
  -/
  have hp : p ≠ 0 := fun hp => h (by rw [hp, trailingCoeff_zero, zero_mul])
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
  -/
  have hq : q ≠ 0 := fun hq => h (by rw [hq, trailingCoeff_zero, mul_zero])
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
  -/
  refine le_antisymm ?_ le_trailingDegree_mul
  rw [trailingDegree_eq_natTrailingDegree hp, trailingDegree_eq_natTrailingDegree hq, ←
    ENat.coe_add]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    hq : Ne q 0
    ⊢ LE.le (HMul.hMul p q).trailingDegree ↑(HAdd.hAdd p.natTrailingDegree q.natTr …
  -/
  apply trailingDegree_le_of_ne_zero
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Ne ((HMul.hMul p q).coeff (HAdd.hAdd p.natTrailingDegree q.natTrailingDegree …
  -/
  rwa [coeff_mul_natTrailingDegree_add_natTrailingDegree]
  /-
    🎉 no goals
  -/


theorem natTrailingDegree_mul' (h : p.trailingCoeff * q.trailingCoeff ≠ 0) :
    (p * q).natTrailingDegree = p.natTrailingDegree + q.natTrailingDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    ⊢ Eq (HMul.hMul p q).natTrailingDegree (HAdd.hAdd p.natTrailingDegree q.natTra …
  -/
  have hp : p ≠ 0 := fun hp => h (by rw [hp, trailingCoeff_zero, zero_mul])
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    ⊢ Eq (HMul.hMul p q).natTrailingDegree (HAdd.hAdd p.natTrailingDegree q.natTra …
  -/
  have hq : q ≠ 0 := fun hq => h (by rw [hq, trailingCoeff_zero, mul_zero])
  -- Porting note: Needed to account for different coercion behaviour & add the lemmas below
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    hq : Ne q 0
    ⊢ Eq (HMul.hMul p q).natTrailingDegree (HAdd.hAdd p.natTrailingDegree q.natTra …
  -/
  have aux1 : ∀ n, Nat.cast n = WithTop.some (n) := fun n ↦ rfl
  have aux2 : ∀ (p : R[X]), WithTop.some (natTrailingDegree p) = Nat.cast (natTrailingDegree p) :=
    fun p ↦ rfl
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.trailingCoeff q.trailingCoeff) 0
    hp : Ne p 0
    hq : Ne q 0
    aux1 : ∀ (n : Nat), Eq ↑n ↑n
    aux2 : ∀ (p : Polynomial R), Eq ↑p.natTrailingDegree ↑p.natTrailingDegree
    ⊢ Eq (HMul.hMul p q).natTrailingDegree (HAdd.hAdd p.natTrailingDegree q.natTra …
  -/
  apply natTrailingDegree_eq_of_trailingDegree_eq_some
  rw [trailingDegree_mul' h, aux1 (natTrailingDegree p + natTrailingDegree q),
    WithTop.coe_add, aux2 p, aux2 q, ← trailingDegree_eq_natTrailingDegree hp, ←
    trailingDegree_eq_natTrailingDegree hq]


theorem natTrailingDegree_mul [NoZeroDivisors R] (hp : p ≠ 0) (hq : q ≠ 0) :
    (p * q).natTrailingDegree = p.natTrailingDegree + q.natTrailingDegree :=
  natTrailingDegree_mul'
    (mul_ne_zero (mt trailingCoeff_eq_zero.mp hp) (mt trailingCoeff_eq_zero.mp hq))


@[simp]
theorem trailingDegree_one : trailingDegree (1 : R[X]) = (0 : ℕ∞) :=
  trailingDegree_C one_ne_zero


@[simp]
theorem trailingDegree_X : trailingDegree (X : R[X]) = 1 :=
  trailingDegree_monomial one_ne_zero


@[simp]
theorem natTrailingDegree_X : (X : R[X]).natTrailingDegree = 1 :=
  natTrailingDegree_monomial one_ne_zero


@[simp]
lemma trailingDegree_X_pow (n : ℕ) :
    (X ^ n : R[X]).trailingDegree = n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).trailingDegree ↑n
  -/
  rw [X_pow_eq_monomial, trailingDegree_monomial one_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma natTrailingDegree_X_pow (n : ℕ) :
    (X ^ n : R[X]).natTrailingDegree = n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).natTrailingDegree n
  -/
  rw [X_pow_eq_monomial, natTrailingDegree_monomial one_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem trailingDegree_neg (p : R[X]) : trailingDegree (-p) = trailingDegree p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).trailingDegree p.trailingDegree
  -/
  unfold trailingDegree
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).support.min p.support.min
  -/
  rw [support_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem natTrailingDegree_neg (p : R[X]) : natTrailingDegree (-p) = natTrailingDegree p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).natTrailingDegree p.natTrailingDegree
  -/
  simp [natTrailingDegree]
  /-
    🎉 no goals
  -/


@[simp]
theorem natTrailingDegree_intCast (n : ℤ) : natTrailingDegree (n : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    ⊢ Eq (↑n).natTrailingDegree 0
  -/
  simp only [← C_eq_intCast, natTrailingDegree_C]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias natTrailingDegree_int_cast := natTrailingDegree_intCast


/-- The second-lowest coefficient, or 0 for constants -/
def nextCoeffUp (p : R[X]) : R :=
  if p.natTrailingDegree = 0 then 0 else p.coeff (p.natTrailingDegree + 1)


                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : Semiring R
                                                                    ⊢ Eq (Polynomial.nextCoeffUp 0) 0
                                                                  -/
@[simp] lemma nextCoeffUp_zero : nextCoeffUp (0 : R[X]) = 0 := by simp [nextCoeffUp]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem nextCoeffUp_C_eq_zero (c : R) : nextCoeffUp (C c) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    ⊢ Eq (Polynomial.C c).nextCoeffUp 0
  -/
  rw [nextCoeffUp]
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    ⊢ Eq (ite (Eq (Polynomial.C c).natTrailingDegree 0) 0 ((Polynomial.C c).coeff  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nextCoeffUp_of_constantCoeff_eq_zero (p : R[X]) (hp : coeff p 0 = 0) :
    nextCoeffUp p = p.coeff (p.natTrailingDegree + 1) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Eq (p.coeff 0) 0
    ⊢ Eq p.nextCoeffUp (p.coeff (HAdd.hAdd p.natTrailingDegree 1))
  -/
  obtain rfl | hp₀ := eq_or_ne p 0
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      hp : Eq (Polynomial.coeff 0 0) 0
      ⊢ Eq (Polynomial.nextCoeffUp 0) (Polynomial.coeff 0 (HAdd.hAdd (Polynomial.nat …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq (p.coeff 0) 0
      hp₀ : Ne p 0
      ⊢ Eq p.nextCoeffUp (p.coeff (HAdd.hAdd p.natTrailingDegree 1))
    -/
  · rw [nextCoeffUp, if_neg (natTrailingDegree_ne_zero.2 ⟨hp₀, hp⟩)]
    /-
      🎉 no goals
    -/


theorem coeff_natTrailingDegree_eq_zero_of_trailingDegree_lt
    (h : trailingDegree p < trailingDegree q) : coeff q (natTrailingDegree p) = 0 :=
  coeff_eq_zero_of_lt_trailingDegree <| natTrailingDegree_le_trailingDegree.trans_lt h


theorem ne_zero_of_trailingDegree_lt {n : ℕ∞} (h : trailingDegree p < n) : p ≠ 0 := fun h₀ =>
               /-
                 R : Type u
                 inst✝ : Semiring R
                 p : Polynomial R
                 n : ENat
                 h : LT.lt p.trailingDegree n
                 h₀ : Eq p 0
                 ⊢ LE.le n p.trailingDegree
               -/
  h.not_le (by simp [h₀])
               /-
                 🎉 no goals
               -/


lemma natTrailingDegree_eq_zero_of_constantCoeff_ne_zero (h : constantCoeff p ≠ 0) :
    p.natTrailingDegree = 0 :=
  le_antisymm (natTrailingDegree_le_of_ne_zero h) zero_le'


lemma eq_X_pow_iff_natDegree_le_natTrailingDegree (h₁ : p.Monic) :
    p = X ^ p.natDegree ↔ p.natDegree ≤ p.natTrailingDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h₁ : p.Monic
    ⊢ Iff (Eq p (HPow.hPow Polynomial.X p.natDegree)) (LE.le p.natDegree p.natTrai …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h₁ : p.Monic
      h : Eq p (HPow.hPow Polynomial.X p.natDegree)
      ⊢ LE.le p.natDegree p.natTrailingDegree
    -/
  · nontriviality R
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h₁ : p.Monic
      h : Eq p (HPow.hPow Polynomial.X p.natDegree)
      a✝ : Nontrivial R
      ⊢ LE.le p.natDegree p.natTrailingDegree
    -/
    rw [h, natTrailingDegree_X_pow, ← h]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h₁ : p.Monic
      h : LE.le p.natDegree p.natTrailingDegree
      ⊢ Eq p (HPow.hPow Polynomial.X p.natDegree)
    -/
  · ext n
    /-
      case refine_2.a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h₁ : p.Monic
      h : LE.le p.natDegree p.natTrailingDegree
      n : Nat
      ⊢ Eq (p.coeff n) ((HPow.hPow Polynomial.X p.natDegree).coeff n)
    -/
    rw [coeff_X_pow]
    /-
      case refine_2.a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h₁ : p.Monic
      h : LE.le p.natDegree p.natTrailingDegree
      n : Nat
      ⊢ Eq (p.coeff n) (ite (Eq n p.natDegree) 1 0)
    -/
    obtain hn | rfl | hn := lt_trichotomy n p.natDegree
      /-
        case refine_2.a.inl
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        h₁ : p.Monic
        h : LE.le p.natDegree p.natTrailingDegree
        n : Nat
        hn : LT.lt n p.natDegree
        ⊢ Eq (p.coeff n) (ite (Eq n p.natDegree) 1 0)
      -/
    · rw [if_neg hn.ne, coeff_eq_zero_of_lt_natTrailingDegree (hn.trans_le h)]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.a.inr.inl
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        h₁ : p.Monic
        h : LE.le p.natDegree p.natTrailingDegree
        ⊢ Eq (p.coeff p.natDegree) (ite (Eq p.natDegree p.natDegree) 1 0)
      -/
    · simpa only [if_pos rfl] using h₁.leadingCoeff
      /-
        🎉 no goals
      -/
      /-
        case refine_2.a.inr.inr
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        h₁ : p.Monic
        h : LE.le p.natDegree p.natTrailingDegree
        n : Nat
        hn : LT.lt p.natDegree n
        ⊢ Eq (p.coeff n) (ite (Eq n p.natDegree) 1 0)
      -/
    · rw [if_neg hn.ne', coeff_eq_zero_of_natDegree_lt hn]
      /-
        🎉 no goals
      -/


lemma eq_X_pow_iff_natTrailingDegree_eq_natDegree (h₁ : p.Monic) :
    p = X ^ p.natDegree ↔ p.natTrailingDegree = p.natDegree :=
  h₁.eq_X_pow_iff_natDegree_le_natTrailingDegree.trans (natTrailingDegree_le_natDegree p).ge_iff_eq


@[deprecated (since := "2024-04-26")]
alias ⟨_, eq_X_pow_of_natTrailingDegree_eq_natDegree⟩ := eq_X_pow_iff_natTrailingDegree_eq_natDegree


