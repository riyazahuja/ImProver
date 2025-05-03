protected theorem add_pow_prime_pow_eq (h : Commute x y) (n : ℕ) :
    (x + y) ^ p ^ n =
      x ^ p ^ n + y ^ p ^ n +
        p * ∑ k ∈ Ioo 0 (p ^ n), x ^ k * y ^ (p ^ n - k) * ↑((p ^ n).choose k / p) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Nat
    hp : Nat.Prime p
    x y : R
    h : Commute x y
    n : Nat
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) (HPow.hPow p n)) (HAdd.hAdd (HAdd.hAdd (HPow.h …
  -/
  trans x ^ p ^ n + y ^ p ^ n + ∑ k ∈ Ioo 0 (p ^ n), x ^ k * y ^ (p ^ n - k) * (p ^ n).choose k
  · simp_rw [h.add_pow, ← Nat.Ico_zero_eq_range, Nat.Ico_succ_right, Icc_eq_cons_Ico (zero_le _),
      Finset.sum_cons, Ico_eq_cons_Ioo (pow_pos hp.pos _), Finset.sum_cons, tsub_self, tsub_zero,
      pow_zero, Nat.choose_zero_right, Nat.choose_self, Nat.cast_one, mul_one, one_mul, ← add_assoc]
    /-
      R : Type u_1
      inst✝ : Semiring R
      p : Nat
      hp : Nat.Prime p
      x y : R
      h : Commute x y
      n : Nat
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow x (HPow.hPow p n)) (HPow.hPow y (HPow.hP …
    -/
  · congr 1
    /-
      case e_a
      R : Type u_1
      inst✝ : Semiring R
      p : Nat
      hp : Nat.Prime p
      x y : R
      h : Commute x y
      n : Nat
      ⊢ Eq ((Finset.Ioo 0 (HPow.hPow p n)).sum fun k => HMul.hMul (HMul.hMul (HPow.h …
    -/
    simp_rw [Finset.mul_sum, Nat.cast_comm, mul_assoc _ _ (p : R), ← Nat.cast_mul]
    /-
      case e_a
      R : Type u_1
      inst✝ : Semiring R
      p : Nat
      hp : Nat.Prime p
      x y : R
      h : Commute x y
      n : Nat
      ⊢ Eq ((Finset.Ioo 0 (HPow.hPow p n)).sum fun k => HMul.hMul (HMul.hMul (HPow.h …
    -/
    refine Finset.sum_congr rfl fun i hi => ?_
    /-
      case e_a
      R : Type u_1
      inst✝ : Semiring R
      p : Nat
      hp : Nat.Prime p
      x y : R
      h : Commute x y
      n i : Nat
      hi : Membership.mem (Finset.Ioo 0 (HPow.hPow p n)) i
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x i) (HPow.hPow y (HSub.hSub (HPow.hPow  …
    -/
    rw [mem_Ioo] at hi
    /-
      case e_a
      R : Type u_1
      inst✝ : Semiring R
      p : Nat
      hp : Nat.Prime p
      x y : R
      h : Commute x y
      n i : Nat
      hi : And (LT.lt 0 i) (LT.lt i (HPow.hPow p n))
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x i) (HPow.hPow y (HSub.hSub (HPow.hPow  …
    -/
    rw [Nat.div_mul_cancel (hp.dvd_choose_pow hi.1.ne' hi.2.ne)]
    /-
      🎉 no goals
    -/


protected theorem add_pow_prime_eq (h : Commute x y) :
    (x + y) ^ p =
      x ^ p + y ^ p + p * ∑ k ∈ Finset.Ioo 0 p, x ^ k * y ^ (p - k) * ↑(p.choose k / p) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Nat
    hp : Nat.Prime p
    x y : R
    h : Commute x y
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) p) (HAdd.hAdd (HAdd.hAdd (HPow.hPow x p) (HPow …
  -/
  simpa using h.add_pow_prime_pow_eq hp 1
  /-
    🎉 no goals
  -/


protected theorem exists_add_pow_prime_pow_eq (h : Commute x y) (n : ℕ) :
    ∃ r, (x + y) ^ p ^ n = x ^ p ^ n + y ^ p ^ n + p * r :=
  ⟨_, h.add_pow_prime_pow_eq hp n⟩


protected theorem exists_add_pow_prime_eq (h : Commute x y) :
    ∃ r, (x + y) ^ p = x ^ p + y ^ p + p * r :=
  ⟨_, h.add_pow_prime_eq hp⟩


theorem add_pow_prime_pow_eq :
    (x + y) ^ p ^ n =
      x ^ p ^ n + y ^ p ^ n +
        p * ∑ k ∈ Finset.Ioo 0 (p ^ n), x ^ k * y ^ (p ^ n - k) * ↑((p ^ n).choose k / p) :=
  (Commute.all x y).add_pow_prime_pow_eq hp n


theorem add_pow_prime_eq :
    (x + y) ^ p =
      x ^ p + y ^ p + p * ∑ k ∈ Finset.Ioo 0 p, x ^ k * y ^ (p - k) * ↑(p.choose k / p) :=
  (Commute.all x y).add_pow_prime_eq hp


theorem exists_add_pow_prime_pow_eq :
    ∃ r, (x + y) ^ p ^ n = x ^ p ^ n + y ^ p ^ n + p * r :=
  (Commute.all x y).exists_add_pow_prime_pow_eq hp n


theorem exists_add_pow_prime_eq :
    ∃ r, (x + y) ^ p = x ^ p + y ^ p + p * r :=
  (Commute.all x y).exists_add_pow_prime_eq hp


lemma add_pow_expChar_of_commute (h : Commute x y) : (x + y) ^ p = x ^ p + y ^ p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    p : Nat
    hR : ExpChar R p
    h : Commute x y
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) p) (HAdd.hAdd (HPow.hPow x p) (HPow.hPow y p))
  -/
  obtain _ | hprime := hR
    /-
      case zero
      R : Type u_1
      inst✝¹ : Semiring R
      x y : R
      h : Commute x y
      inst✝ : CharZero R
      ⊢ Eq (HPow.hPow (HAdd.hAdd x y) 1) (HAdd.hAdd (HPow.hPow x 1) (HPow.hPow y 1))
    -/
  · simp only [pow_one]
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝ : Semiring R
      x y : R
      p : Nat
      h : Commute x y
      hprime : Nat.Prime p
      hchar✝ : CharP R p
      ⊢ Eq (HPow.hPow (HAdd.hAdd x y) p) (HAdd.hAdd (HPow.hPow x p) (HPow.hPow y p))
    -/
  · let ⟨r, hr⟩ := h.exists_add_pow_prime_eq hprime
    /-
      case prime
      R : Type u_1
      inst✝ : Semiring R
      x y : R
      p : Nat
      h : Commute x y
      hprime : Nat.Prime p
      hchar✝ : CharP R p
      r : R
      hr : Eq (HPow.hPow (HAdd.hAdd x y) p) (HAdd.hAdd (HAdd.hAdd (HPow.hPow x p) (H …
      ⊢ Eq (HPow.hPow (HAdd.hAdd x y) p) (HAdd.hAdd (HPow.hPow x p) (HPow.hPow y p))
    -/
    simp [hr]
    /-
      🎉 no goals
    -/


lemma add_pow_expChar_pow_of_commute (h : Commute x y) :
    (x + y) ^ p ^ n = x ^ p ^ n + y ^ p ^ n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    p n : Nat
    hR : ExpChar R p
    h : Commute x y
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) (HPow.hPow p n)) (HAdd.hAdd (HPow.hPow x (HPow …
  -/
  obtain _ | hprime := hR
    /-
      case zero
      R : Type u_1
      inst✝¹ : Semiring R
      x y : R
      n : Nat
      h : Commute x y
      inst✝ : CharZero R
      ⊢ Eq (HPow.hPow (HAdd.hAdd x y) (HPow.hPow 1 n)) (HAdd.hAdd (HPow.hPow x (HPow …
    -/
  · simp only [one_pow, pow_one]
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝ : Semiring R
      x y : R
      p n : Nat
      h : Commute x y
      hprime : Nat.Prime p
      hchar✝ : CharP R p
      ⊢ Eq (HPow.hPow (HAdd.hAdd x y) (HPow.hPow p n)) (HAdd.hAdd (HPow.hPow x (HPow …
    -/
  · let ⟨r, hr⟩ := h.exists_add_pow_prime_pow_eq hprime n
    /-
      case prime
      R : Type u_1
      inst✝ : Semiring R
      x y : R
      p n : Nat
      h : Commute x y
      hprime : Nat.Prime p
      hchar✝ : CharP R p
      r : R
      hr : Eq (HPow.hPow (HAdd.hAdd x y) (HPow.hPow p n)) (HAdd.hAdd (HAdd.hAdd (HPo …
      ⊢ Eq (HPow.hPow (HAdd.hAdd x y) (HPow.hPow p n)) (HAdd.hAdd (HPow.hPow x (HPow …
    -/
    simp [hr]
    /-
      🎉 no goals
    -/


lemma add_pow_eq_mul_pow_add_pow_div_expChar_of_commute (h : Commute x y) :
    (x + y) ^ n = (x + y) ^ (n % p) * (x ^ p + y ^ p) ^ (n / p) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    p n : Nat
    hR : ExpChar R p
    h : Commute x y
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) n) (HMul.hMul (HPow.hPow (HAdd.hAdd x y) (HMod …
  -/
  rw [← add_pow_expChar_of_commute _ h, ← pow_mul, ← pow_add, Nat.mod_add_div]
  /-
    🎉 no goals
  -/


lemma add_pow_char_of_commute (h : Commute x y) : (x + y) ^ p = x ^ p + y ^ p :=
  add_pow_expChar_of_commute _ h


lemma add_pow_char_pow_of_commute (h : Commute x y) : (x + y) ^ p ^ n = x ^ p ^ n + y ^ p ^ n :=
  add_pow_expChar_pow_of_commute _ _ h


lemma add_pow_eq_mul_pow_add_pow_div_char_of_commute (h : Commute x y) :
    (x + y) ^ n = (x + y) ^ (n % p) * (x ^ p + y ^ p) ^ (n / p) :=
  add_pow_eq_mul_pow_add_pow_div_expChar_of_commute _ _ h


lemma add_pow_expChar : (x + y) ^ p = x ^ p + y ^ p := add_pow_expChar_of_commute _ <| .all ..


lemma add_pow_expChar_pow : (x + y) ^ p ^ n = x ^ p ^ n + y ^ p ^ n :=
  add_pow_expChar_pow_of_commute _ _ <| .all ..


lemma add_pow_eq_mul_pow_add_pow_div_expChar :
    (x + y) ^ n = (x + y) ^ (n % p) * (x ^ p + y ^ p) ^ (n / p) :=
  add_pow_eq_mul_pow_add_pow_div_expChar_of_commute _ _ <| .all ..


lemma add_pow_char : (x + y) ^ p = x ^ p + y ^ p := add_pow_expChar ..

lemma add_pow_char_pow : (x + y) ^ p ^ n = x ^ p ^ n + y ^ p ^ n := add_pow_expChar_pow ..


lemma add_pow_eq_mul_pow_add_pow_div_char :
    (x + y) ^ n = (x + y) ^ (n % p) * (x ^ p + y ^ p) ^ (n / p) :=
  add_pow_eq_mul_pow_add_pow_div_expChar ..


@[deprecated (since := "2024-10-21")]
alias add_pow_eq_add_pow_mod_mul_pow_add_pow_div := add_pow_eq_mul_pow_add_pow_div_char


lemma sub_pow_expChar_of_commute (h : Commute x y) : (x - y) ^ p = x ^ p - y ^ p := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    p : Nat
    hR : ExpChar R p
    h : Commute x y
    ⊢ Eq (HPow.hPow (HSub.hSub x y) p) (HSub.hSub (HPow.hPow x p) (HPow.hPow y p))
  -/
  simp [eq_sub_iff_add_eq, ← add_pow_expChar_of_commute _ (h.sub_left rfl)]
  /-
    🎉 no goals
  -/


lemma sub_pow_expChar_pow_of_commute (h : Commute x y) :
    (x - y) ^ p ^ n = x ^ p ^ n - y ^ p ^ n := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    p n : Nat
    hR : ExpChar R p
    h : Commute x y
    ⊢ Eq (HPow.hPow (HSub.hSub x y) (HPow.hPow p n)) (HSub.hSub (HPow.hPow x (HPow …
  -/
  simp [eq_sub_iff_add_eq, ← add_pow_expChar_pow_of_commute _ _ (h.sub_left rfl)]
  /-
    🎉 no goals
  -/


lemma sub_pow_eq_mul_pow_sub_pow_div_expChar_of_commute (h : Commute x y) :
    (x - y) ^ n = (x - y) ^ (n % p) * (x ^ p - y ^ p) ^ (n / p) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    p n : Nat
    hR : ExpChar R p
    h : Commute x y
    ⊢ Eq (HPow.hPow (HSub.hSub x y) n) (HMul.hMul (HPow.hPow (HSub.hSub x y) (HMod …
  -/
  rw [← sub_pow_expChar_of_commute _ h, ← pow_mul, ← pow_add, Nat.mod_add_div]
  /-
    🎉 no goals
  -/


lemma neg_one_pow_expChar : (-1 : R) ^ p = -1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p : Nat
    hR : ExpChar R p
    ⊢ Eq (HPow.hPow (-1) p) (-1)
  -/
  rw [eq_neg_iff_add_eq_zero]
  /-
    R : Type u_1
    inst✝ : Ring R
    p : Nat
    hR : ExpChar R p
    ⊢ Eq (HAdd.hAdd (HPow.hPow (-1) p) 1) 0
  -/
  nth_rw 2 [← one_pow p]
  rw [← add_pow_expChar_of_commute _ (Commute.one_right _), neg_add_cancel,
    zero_pow (expChar_ne_zero R p)]


lemma neg_one_pow_expChar_pow : (-1 : R) ^ p ^ n = -1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p n : Nat
    hR : ExpChar R p
    ⊢ Eq (HPow.hPow (-1) (HPow.hPow p n)) (-1)
  -/
  rw [eq_neg_iff_add_eq_zero]
  /-
    R : Type u_1
    inst✝ : Ring R
    p n : Nat
    hR : ExpChar R p
    ⊢ Eq (HAdd.hAdd (HPow.hPow (-1) (HPow.hPow p n)) 1) 0
  -/
  nth_rw 2 [← one_pow (p ^ n)]
  rw [← add_pow_expChar_pow_of_commute _ _ (Commute.one_right _), neg_add_cancel,
    zero_pow (pow_ne_zero _ <| expChar_ne_zero R p)]


lemma sub_pow_char_of_commute (h : Commute x y) : (x - y) ^ p = x ^ p - y ^ p :=
  sub_pow_expChar_of_commute _ h


lemma sub_pow_char_pow_of_commute (h : Commute x y) : (x - y) ^ p ^ n = x ^ p ^ n - y ^ p ^ n :=
  sub_pow_expChar_pow_of_commute _ _ h


lemma neg_one_pow_char : (-1 : R) ^ p = -1 := neg_one_pow_expChar ..


lemma neg_one_pow_char_pow : (-1 : R) ^ p ^ n = -1 := neg_one_pow_expChar_pow ..


lemma sub_pow_eq_mul_pow_sub_pow_div_char_of_commute (h : Commute x y) :
    (x - y) ^ n = (x - y) ^ (n % p) * (x ^ p - y ^ p) ^ (n / p) :=
  sub_pow_eq_mul_pow_sub_pow_div_expChar_of_commute _ _ h


lemma sub_pow_expChar : (x - y) ^ p = x ^ p - y ^ p := sub_pow_expChar_of_commute _ <| .all ..


lemma sub_pow_expChar_pow : (x - y) ^ p ^ n = x ^ p ^ n - y ^ p ^ n :=
  sub_pow_expChar_pow_of_commute _ _ <| .all ..


lemma sub_pow_eq_mul_pow_sub_pow_div_expChar :
    (x - y) ^ n = (x - y) ^ (n % p) * (x ^ p - y ^ p) ^ (n / p) :=
  sub_pow_eq_mul_pow_sub_pow_div_expChar_of_commute _ _ <| .all ..


lemma sub_pow_char : (x - y) ^ p = x ^ p - y ^ p := sub_pow_expChar ..

lemma sub_pow_char_pow : (x - y) ^ p ^ n = x ^ p ^ n - y ^ p ^ n := sub_pow_expChar_pow ..


lemma sub_pow_eq_mul_pow_sub_pow_div_char :
    (x - y) ^ n = (x - y) ^ (n % p) * (x ^ p - y ^ p) ^ (n / p) :=
  sub_pow_eq_mul_pow_sub_pow_div_expChar ..


lemma Nat.Prime.dvd_add_pow_sub_pow_of_dvd (hpri : p.Prime) {r : R} (h₁ : r ∣ x ^ p)
    (h₂ : r ∣ p * x) : r ∣ (x + y) ^ p - y ^ p := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    p : Nat
    hpri : Nat.Prime p
    r : R
    h₁ : Dvd.dvd r (HPow.hPow x p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) x)
    ⊢ Dvd.dvd r (HSub.hSub (HPow.hPow (HAdd.hAdd x y) p) (HPow.hPow y p))
  -/
  rw [add_pow_prime_eq hpri, add_right_comm, add_assoc, add_sub_assoc, add_sub_cancel_right]
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    p : Nat
    hpri : Nat.Prime p
    r : R
    h₁ : Dvd.dvd r (HPow.hPow x p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) x)
    ⊢ Dvd.dvd r (HAdd.hAdd (HPow.hPow x p) (HMul.hMul (↑p) ((Finset.Ioo 0 p).sum f …
  -/
  apply dvd_add h₁ (h₂.trans <| mul_dvd_mul_left _ <| Finset.dvd_sum _)
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    p : Nat
    hpri : Nat.Prime p
    r : R
    h₁ : Dvd.dvd r (HPow.hPow x p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) x)
    ⊢ ∀ (i : Nat), Membership.mem (Finset.Ioo 0 p) i → Dvd.dvd x (HMul.hMul (HMul. …
  -/
  simp only [Finset.mem_Ioo, and_imp, mul_assoc]
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    p : Nat
    hpri : Nat.Prime p
    r : R
    h₁ : Dvd.dvd r (HPow.hPow x p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) x)
    ⊢ ∀ (i : Nat), LT.lt 0 i → LT.lt i p → Dvd.dvd x (HMul.hMul (HPow.hPow x i) (H …
  -/
  intro i hi0 _
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    p : Nat
    hpri : Nat.Prime p
    r : R
    h₁ : Dvd.dvd r (HPow.hPow x p)
    h₂ : Dvd.dvd r (HMul.hMul (↑p) x)
    i : Nat
    hi0 : LT.lt 0 i
    a✝ : LT.lt i p
    ⊢ Dvd.dvd x (HMul.hMul (HPow.hPow x i) (HMul.hMul (HPow.hPow y (HSub.hSub p i) …
  -/
  exact dvd_mul_of_dvd_left (dvd_rfl.pow hi0.ne') _
  /-
    🎉 no goals
  -/


/-- The characteristic of a finite ring cannot be zero. -/
theorem char_ne_zero_of_finite (p : ℕ) [CharP R p] [Finite R] : p ≠ 0 := by
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : Finite R
    ⊢ Ne p 0
  -/
  rintro rfl
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : Finite R
    inst✝ : CharP R 0
    ⊢ False
  -/
  haveI : CharZero R := charP_to_charZero R
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : Finite R
    inst✝ : CharP R 0
    this : CharZero R
    ⊢ False
  -/
  cases nonempty_fintype R
  /-
    case intro
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : Finite R
    inst✝ : CharP R 0
    this : CharZero R
    val✝ : Fintype R
    ⊢ False
  -/
  exact absurd Nat.cast_injective (not_injective_infinite_finite ((↑) : ℕ → R))
  /-
    🎉 no goals
  -/


theorem ringChar_ne_zero_of_finite [Finite R] : ringChar R ≠ 0 :=
  char_ne_zero_of_finite R (ringChar R)


theorem char_is_prime (p : ℕ) [CharP R p] : p.Prime :=
  Or.resolve_right (char_is_prime_or_zero R p) (char_ne_zero_of_finite R p)


variable (R) in
/-- The frobenius map `x ↦ x ^ p`. -/
def frobenius : R →+* R where
  __ := powMonoidHom p
  map_zero' := zero_pow (expChar_pos R p).ne'
  map_add' _ _ := add_pow_expChar ..


variable (R) in
/-- The iterated frobenius map `x ↦ x ^ p ^ n`. -/
def iterateFrobenius : R →+* R where
  __ := powMonoidHom (p ^ n)
  map_zero' := zero_pow (expChar_pow_pos R p n).ne'
  map_add' _ _ := add_pow_expChar_pow ..



lemma frobenius_def : frobenius R p x = x ^ p := rfl


lemma iterateFrobenius_def : iterateFrobenius R p n x = x ^ p ^ n := rfl


lemma iterate_frobenius : (frobenius R p)^[n] x = x ^ p ^ n := congr_fun (pow_iterate p n) x


lemma coe_iterateFrobenius : iterateFrobenius R p n = (frobenius R p)^[n] :=
  (pow_iterate p n).symm


lemma iterateFrobenius_one_apply : iterateFrobenius R p 1 x = x ^ p := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    p : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Eq ((iterateFrobenius R p 1) x) (HPow.hPow x p)
  -/
  rw [iterateFrobenius_def, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma iterateFrobenius_one : iterateFrobenius R p 1 = frobenius R p :=
  RingHom.ext (iterateFrobenius_one_apply R p)


lemma iterateFrobenius_zero_apply : iterateFrobenius R p 0 x = x := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    p : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Eq ((iterateFrobenius R p 0) x) x
  -/
  rw [iterateFrobenius_def, pow_zero, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma iterateFrobenius_zero : iterateFrobenius R p 0 = RingHom.id R :=
  RingHom.ext (iterateFrobenius_zero_apply R p)


lemma iterateFrobenius_add_apply :
    iterateFrobenius R p (m + n) x = iterateFrobenius R p m (iterateFrobenius R p n x) := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    p m n : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Eq ((iterateFrobenius R p (HAdd.hAdd m n)) x) ((iterateFrobenius R p m) ((it …
  -/
  simp_rw [iterateFrobenius_def, add_comm m n, pow_add, pow_mul]
  /-
    🎉 no goals
  -/


lemma iterateFrobenius_add :
    iterateFrobenius R p (m + n) = (iterateFrobenius R p m).comp (iterateFrobenius R p n) :=
  RingHom.ext (iterateFrobenius_add_apply R p m n)


lemma iterateFrobenius_mul_apply :
    iterateFrobenius R p (m * n) x = (iterateFrobenius R p m)^[n] x := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    p m n : Nat
    inst✝ : ExpChar R p
    x : R
    ⊢ Eq ((iterateFrobenius R p (HMul.hMul m n)) x) (Nat.iterate (⇑(iterateFrobeni …
  -/
  simp_rw [coe_iterateFrobenius, Function.iterate_mul]
  /-
    🎉 no goals
  -/


lemma coe_iterateFrobenius_mul : iterateFrobenius R p (m * n) = (iterateFrobenius R p m)^[n] :=
  funext (iterateFrobenius_mul_apply R p m n)


lemma frobenius_mul : frobenius R p (x * y) = frobenius R p x * frobenius R p y :=
  map_mul (frobenius R p) x y


lemma frobenius_one : frobenius R p 1 = 1 := one_pow _


lemma MonoidHom.map_frobenius : f (frobenius R p x) = frobenius S p (f x) := map_pow f x p

lemma RingHom.map_frobenius : g (frobenius R p x) = frobenius S p (g x) := map_pow g x p


lemma MonoidHom.map_iterate_frobenius (n : ℕ) :
    f ((frobenius R p)^[n] x) = (frobenius S p)^[n] (f x) :=
  Function.Semiconj.iterate_right (f.map_frobenius p) n x


lemma RingHom.map_iterate_frobenius (n : ℕ) :
    g ((frobenius R p)^[n] x) = (frobenius S p)^[n] (g x) :=
  g.toMonoidHom.map_iterate_frobenius p x n


lemma MonoidHom.iterate_map_frobenius (f : R →* R) (p : ℕ) [ExpChar R p] (n : ℕ) :
    f^[n] (frobenius R p x) = frobenius R p (f^[n] x) :=
  iterate_map_pow f _ _ _


lemma RingHom.iterate_map_frobenius (f : R →+* R) (p : ℕ) [ExpChar R p] (n : ℕ) :
    f^[n] (frobenius R p x) = frobenius R p (f^[n] x) := iterate_map_pow f _ _ _


lemma list_sum_pow_char (l : List R) : l.sum ^ p = (l.map (· ^ p : R → R)).sum :=
  map_list_sum (frobenius R p) _


lemma multiset_sum_pow_char (s : Multiset R) : s.sum ^ p = (s.map (· ^ p : R → R)).sum :=
  map_multiset_sum (frobenius R p) _


lemma sum_pow_char {ι : Type*} (s : Finset ι) (f : ι → R) : (∑ i ∈ s, f i) ^ p = ∑ i ∈ s, f i ^ p :=
  map_sum (frobenius R p) _ _


lemma list_sum_pow_char_pow (l : List R) : l.sum ^ p ^ n = (l.map (· ^ p ^ n : R → R)).sum :=
  map_list_sum (iterateFrobenius R p n) _


lemma multiset_sum_pow_char_pow (s : Multiset R) :
    s.sum ^ p ^ n = (s.map (· ^ p ^ n : R → R)).sum := map_multiset_sum (iterateFrobenius R p n) _


lemma sum_pow_char_pow {ι : Type*} (s : Finset ι) (f : ι → R) :
    (∑ i ∈ s, f i) ^ p ^ n = ∑ i ∈ s, f i ^ p ^ n := map_sum (iterateFrobenius R p n) _ _


lemma frobenius_neg : frobenius R p (-x) = -frobenius R p x := map_neg ..


lemma frobenius_sub : frobenius R p (x - y) = frobenius R p x - frobenius R p y := map_sub ..


