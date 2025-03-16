/-- In a local ring the characteristics is either zero or a prime power. -/
theorem charP_zero_or_prime_power (R : Type*) [CommRing R] [IsLocalRing R] (q : ℕ)
    [char_R_q : CharP R q] : q = 0 ∨ IsPrimePow q := by
  -- Assume `q := char(R)` is not zero.
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    ⊢ Or (Eq q 0) (IsPrimePow q)
  -/
  apply or_iff_not_imp_left.2
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    ⊢ Not (Eq q 0) → IsPrimePow q
  -/
  intro q_pos
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    q_pos : Not (Eq q 0)
    ⊢ IsPrimePow q
  -/
  let K := IsLocalRing.ResidueField R
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    q_pos : Not (Eq q 0)
    K : Type u_1 := IsLocalRing.ResidueField R
    ⊢ IsPrimePow q
  -/
  haveI RM_char := ringChar.charP K
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    q_pos : Not (Eq q 0)
    K : Type u_1 := IsLocalRing.ResidueField R
    RM_char : CharP K (ringChar K)
    ⊢ IsPrimePow q
  -/
  let r := ringChar K
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    q_pos : Not (Eq q 0)
    K : Type u_1 := IsLocalRing.ResidueField R
    RM_char : CharP K (ringChar K)
    r : Nat := ringChar K
    ⊢ IsPrimePow q
  -/
  let n := q.factorization r
  -- `r := char(R/m)` is either prime or zero:
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsLocalRing R
    q : Nat
    char_R_q : CharP R q
    q_pos : Not (Eq q 0)
    K : Type u_1 := IsLocalRing.ResidueField R
    RM_char : CharP K (ringChar K)
    r : Nat := ringChar K
    n : Nat := q.factorization r
    ⊢ IsPrimePow q
  -/
  rcases CharP.char_is_prime_or_zero K r with r_prime | r_zero
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      ⊢ IsPrimePow q
    -/
  · let a := q / r ^ n
    -- If `r` is prime, we can write it as `r = a * q^n` ...
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      a : Nat := HDiv.hDiv q (HPow.hPow r n)
      ⊢ IsPrimePow q
    -/
    have q_eq_a_mul_rn : q = r ^ n * a := by rw [Nat.mul_div_cancel' (Nat.ordProj_dvd q r)]
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      a : Nat := HDiv.hDiv q (HPow.hPow r n)
      q_eq_a_mul_rn : Eq q (HMul.hMul (HPow.hPow r n) a)
      ⊢ IsPrimePow q
    -/
    have r_ne_dvd_a := Nat.not_dvd_ordCompl r_prime q_pos
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      a : Nat := HDiv.hDiv q (HPow.hPow r n)
      q_eq_a_mul_rn : Eq q (HMul.hMul (HPow.hPow r n) a)
      r_ne_dvd_a : Not (Dvd.dvd r (HDiv.hDiv q (HPow.hPow r (q.factorization r))))
      ⊢ IsPrimePow q
    -/
    have rn_dvd_q : r ^ n ∣ q := ⟨a, q_eq_a_mul_rn⟩
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      a : Nat := HDiv.hDiv q (HPow.hPow r n)
      q_eq_a_mul_rn : Eq q (HMul.hMul (HPow.hPow r n) a)
      r_ne_dvd_a : Not (Dvd.dvd r (HDiv.hDiv q (HPow.hPow r (q.factorization r))))
      rn_dvd_q : Dvd.dvd (HPow.hPow r n) q
      ⊢ IsPrimePow q
    -/
    rw [mul_comm] at q_eq_a_mul_rn
    -- ... where `a` is a unit.
    have a_unit : IsUnit (a : R) := by
      by_contra g
      rw [← mem_nonunits_iff] at g
      rw [← IsLocalRing.mem_maximalIdeal] at g
      have a_cast_zero := Ideal.Quotient.eq_zero_iff_mem.2 g
      rw [map_natCast] at a_cast_zero
      have r_dvd_a := (ringChar.spec K a).1 a_cast_zero
      exact absurd r_dvd_a r_ne_dvd_a
    -- Let `b` be the inverse of `a`.
    have rn_cast_zero : ↑(r ^ n) = (0 : R) := by
      rw [← @mul_one R _ ↑(r ^ n), mul_comm, ← Classical.choose_spec a_unit.exists_left_inv,
        mul_assoc, ← Nat.cast_mul, ← q_eq_a_mul_rn, CharP.cast_eq_zero R q]
      simp
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      a : Nat := HDiv.hDiv q (HPow.hPow r n)
      q_eq_a_mul_rn : Eq q (HMul.hMul a (HPow.hPow r n))
      r_ne_dvd_a : Not (Dvd.dvd r (HDiv.hDiv q (HPow.hPow r (q.factorization r))))
      rn_dvd_q : Dvd.dvd (HPow.hPow r n) q
      a_unit : IsUnit ↑a
      rn_cast_zero : Eq (↑(HPow.hPow r n)) 0
      ⊢ IsPrimePow q
    -/
    have q_eq_rn := Nat.dvd_antisymm ((CharP.cast_eq_zero_iff R q (r ^ n)).mp rn_cast_zero) rn_dvd_q
    have n_pos : n ≠ 0 := fun n_zero =>
      absurd (by simpa [n_zero] using q_eq_rn) (CharP.char_ne_one R q)
    -- Definition of prime power: `∃ r n, Prime r ∧ 0 < n ∧ r ^ n = q`.
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_prime : Nat.Prime r
      a : Nat := HDiv.hDiv q (HPow.hPow r n)
      q_eq_a_mul_rn : Eq q (HMul.hMul a (HPow.hPow r n))
      r_ne_dvd_a : Not (Dvd.dvd r (HDiv.hDiv q (HPow.hPow r (q.factorization r))))
      rn_dvd_q : Dvd.dvd (HPow.hPow r n) q
      a_unit : IsUnit ↑a
      rn_cast_zero : Eq (↑(HPow.hPow r n)) 0
      q_eq_rn : Eq q (HPow.hPow r n)
      n_pos : Ne n 0
      ⊢ IsPrimePow q
    -/
    exact ⟨r, ⟨n, ⟨r_prime.prime, ⟨pos_iff_ne_zero.mpr n_pos, q_eq_rn.symm⟩⟩⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_zero : Eq r 0
      ⊢ IsPrimePow q
    -/
  · haveI K_char_p_0 := ringChar.of_eq r_zero
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_zero : Eq r 0
      K_char_p_0 : CharP K 0
      ⊢ IsPrimePow q
    -/
    haveI K_char_zero : CharZero K := CharP.charP_to_charZero K
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_zero : Eq r 0
      K_char_p_0 : CharP K 0
      K_char_zero : CharZero K
      ⊢ IsPrimePow q
    -/
    haveI R_char_zero := RingHom.charZero (IsLocalRing.residue R)
    -- Finally, `r = 0` would lead to a contradiction:
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_zero : Eq r 0
      K_char_p_0 : CharP K 0
      K_char_zero : CharZero K
      R_char_zero : CharZero R
      ⊢ IsPrimePow q
    -/
    have q_zero := CharP.eq R char_R_q (CharP.ofCharZero R)
    /-
      case inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsLocalRing R
      q : Nat
      char_R_q : CharP R q
      q_pos : Not (Eq q 0)
      K : Type u_1 := IsLocalRing.ResidueField R
      RM_char : CharP K (ringChar K)
      r : Nat := ringChar K
      n : Nat := q.factorization r
      r_zero : Eq r 0
      K_char_p_0 : CharP K 0
      K_char_zero : CharZero K
      R_char_zero : CharZero R
      q_zero : Eq q 0
      ⊢ IsPrimePow q
    -/
    exact absurd q_zero q_pos
    /-
      🎉 no goals
    -/

