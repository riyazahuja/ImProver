/-- `legendreSym p 2` is given by `χ₈ p`. -/
theorem at_two (hp : p ≠ 2) : legendreSym p 2 = χ₈ p := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    ⊢ Eq (legendreSym p 2) (ZMod.χ₈ ↑p)
  -/
  have : (2 : ZMod p) = (2 : ℤ) := by norm_cast
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    this : Eq 2 ↑2
    ⊢ Eq (legendreSym p 2) (ZMod.χ₈ ↑p)
  -/
  rw [legendreSym, ← this, quadraticChar_two ((ringChar_zmod_n p).substr hp), card p]
  /-
    🎉 no goals
  -/


/-- `legendreSym p (-2)` is given by `χ₈' p`. -/
theorem at_neg_two (hp : p ≠ 2) : legendreSym p (-2) = χ₈' p := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    ⊢ Eq (legendreSym p (-2)) (ZMod.χ₈' ↑p)
  -/
  have : (-2 : ZMod p) = (-2 : ℤ) := by norm_cast
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    this : Eq (-2) ↑(-2)
    ⊢ Eq (legendreSym p (-2)) (ZMod.χ₈' ↑p)
  -/
  rw [legendreSym, ← this, quadraticChar_neg_two ((ringChar_zmod_n p).substr hp), card p]
  /-
    🎉 no goals
  -/


/-- `2` is a square modulo an odd prime `p` iff `p` is congruent to `1` or `7` mod `8`. -/
theorem exists_sq_eq_two_iff (hp : p ≠ 2) : IsSquare (2 : ZMod p) ↔ p % 8 = 1 ∨ p % 8 = 7 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    ⊢ Iff (IsSquare 2) (Or (Eq (HMod.hMod p 8) 1) (Eq (HMod.hMod p 8) 7))
  -/
  rw [FiniteField.isSquare_two_iff, card p]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    ⊢ Iff (And (Ne (HMod.hMod p 8) 3) (Ne (HMod.hMod p 8) 5)) (Or (Eq (HMod.hMod p …
  -/
  have h₁ := Prime.mod_two_eq_one_iff_ne_two.mpr hp
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    h₁ : Eq (HMod.hMod p 2) 1
    ⊢ Iff (And (Ne (HMod.hMod p 8) 3) (Ne (HMod.hMod p 8) 5)) (Or (Eq (HMod.hMod p …
  -/
  omega
  /-
    🎉 no goals
  -/


/-- `-2` is a square modulo an odd prime `p` iff `p` is congruent to `1` or `3` mod `8`. -/
theorem exists_sq_eq_neg_two_iff (hp : p ≠ 2) : IsSquare (-2 : ZMod p) ↔ p % 8 = 1 ∨ p % 8 = 3 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    ⊢ Iff (IsSquare (-2)) (Or (Eq (HMod.hMod p 8) 1) (Eq (HMod.hMod p 8) 3))
  -/
  rw [FiniteField.isSquare_neg_two_iff, card p]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    ⊢ Iff (And (Ne (HMod.hMod p 8) 5) (Ne (HMod.hMod p 8) 7)) (Or (Eq (HMod.hMod p …
  -/
  have h₁ := Prime.mod_two_eq_one_iff_ne_two.mpr hp
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hp : Ne p 2
    h₁ : Eq (HMod.hMod p 2) 1
    ⊢ Iff (And (Ne (HMod.hMod p 8) 5) (Ne (HMod.hMod p 8) 7)) (Or (Eq (HMod.hMod p …
  -/
  omega
  /-
    🎉 no goals
  -/


/-- **The Law of Quadratic Reciprocity**: if `p` and `q` are distinct odd primes, then
`(q / p) * (p / q) = (-1)^((p-1)(q-1)/4)`. -/
theorem quadratic_reciprocity (hp : p ≠ 2) (hq : q ≠ 2) (hpq : p ≠ q) :
    legendreSym q p * legendreSym p q = (-1) ^ (p / 2 * (q / 2)) := by
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    hpq : Ne p q
    ⊢ Eq (HMul.hMul (legendreSym q ↑p) (legendreSym p ↑q)) (HPow.hPow (-1) (HMul.h …
  -/
  have hp₁ := (Prime.eq_two_or_odd <| @Fact.out p.Prime _).resolve_left hp
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    hpq : Ne p q
    hp₁ : Eq (HMod.hMod p 2) 1
    ⊢ Eq (HMul.hMul (legendreSym q ↑p) (legendreSym p ↑q)) (HPow.hPow (-1) (HMul.h …
  -/
  have hq₁ := (Prime.eq_two_or_odd <| @Fact.out q.Prime _).resolve_left hq
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    hpq : Ne p q
    hp₁ : Eq (HMod.hMod p 2) 1
    hq₁ : Eq (HMod.hMod q 2) 1
    ⊢ Eq (HMul.hMul (legendreSym q ↑p) (legendreSym p ↑q)) (HPow.hPow (-1) (HMul.h …
  -/
  have hq₂ : ringChar (ZMod q) ≠ 2 := (ringChar_zmod_n q).substr hq
  have h :=
    quadraticChar_odd_prime ((ringChar_zmod_n p).substr hp) hq ((ringChar_zmod_n p).substr hpq)
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    hpq : Ne p q
    hp₁ : Eq (HMod.hMod p 2) 1
    hq₁ : Eq (HMod.hMod q 2) 1
    hq₂ : Ne (ringChar (ZMod q)) 2
    h : Eq ((quadraticChar (ZMod p)) ↑q) ((quadraticChar (ZMod q)) (HMul.hMul ↑(ZM …
    ⊢ Eq (HMul.hMul (legendreSym q ↑p) (legendreSym p ↑q)) (HPow.hPow (-1) (HMul.h …
  -/
  rw [card p] at h
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    hpq : Ne p q
    hp₁ : Eq (HMod.hMod p 2) 1
    hq₁ : Eq (HMod.hMod q 2) 1
    hq₂ : Ne (ringChar (ZMod q)) 2
    h : Eq ((quadraticChar (ZMod p)) ↑q) ((quadraticChar (ZMod q)) (HMul.hMul ↑(ZM …
    ⊢ Eq (HMul.hMul (legendreSym q ↑p) (legendreSym p ↑q)) (HPow.hPow (-1) (HMul.h …
  -/
  have nc : ∀ n r : ℕ, ((n : ℤ) : ZMod r) = n := fun n r => by norm_cast
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    hpq : Ne p q
    hp₁ : Eq (HMod.hMod p 2) 1
    hq₁ : Eq (HMod.hMod q 2) 1
    hq₂ : Ne (ringChar (ZMod q)) 2
    h : Eq ((quadraticChar (ZMod p)) ↑q) ((quadraticChar (ZMod q)) (HMul.hMul ↑(ZM …
    nc : ∀ (n r : Nat), Eq ↑↑n ↑n
    ⊢ Eq (HMul.hMul (legendreSym q ↑p) (legendreSym p ↑q)) (HPow.hPow (-1) (HMul.h …
  -/
  have nc' : (((-1) ^ (p / 2) : ℤ) : ZMod q) = (-1) ^ (p / 2) := by norm_cast
  rw [legendreSym, legendreSym, nc, nc, h, map_mul, mul_rotate', mul_comm (p / 2), ← pow_two,
    quadraticChar_sq_one (prime_ne_zero q p hpq.symm), mul_one, pow_mul, χ₄_eq_neg_one_pow hp₁, nc',
    map_pow, quadraticChar_neg_one hq₂, card q, χ₄_eq_neg_one_pow hq₁]


/-- The Law of Quadratic Reciprocity: if `p` and `q` are odd primes, then
`(q / p) = (-1)^((p-1)(q-1)/4) * (p / q)`. -/
theorem quadratic_reciprocity' (hp : p ≠ 2) (hq : q ≠ 2) :
    legendreSym q p = (-1) ^ (p / 2 * (q / 2)) * legendreSym p q := by
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Ne p 2
    hq : Ne q 2
    ⊢ Eq (legendreSym q ↑p) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv p 2)  …
  -/
  rcases eq_or_ne p q with h | h
    /-
      case inl
      p q : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Nat.Prime q)
      hp : Ne p 2
      hq : Ne q 2
      h : Eq p q
      ⊢ Eq (legendreSym q ↑p) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv p 2)  …
    -/
  · subst p
    /-
      case inl
      q : Nat
      inst✝¹ : Fact (Nat.Prime q)
      hq : Ne q 2
      inst✝ : Fact (Nat.Prime q)
      hp : Ne q 2
      ⊢ Eq (legendreSym q ↑q) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv q 2)  …
    -/
    rw [(eq_zero_iff q q).mpr (mod_cast natCast_self q), mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      p q : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Nat.Prime q)
      hp : Ne p 2
      hq : Ne q 2
      h : Ne p q
      ⊢ Eq (legendreSym q ↑p) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv p 2)  …
    -/
  · have qr := congr_arg (· * legendreSym p q) (quadratic_reciprocity hp hq h)
    /-
      case inr
      p q : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Nat.Prime q)
      hp : Ne p 2
      hq : Ne q 2
      h : Ne p q
      qr : Eq ((fun x => HMul.hMul x (legendreSym p ↑q)) (HMul.hMul (legendreSym q ↑ …
      ⊢ Eq (legendreSym q ↑p) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv p 2)  …
    -/
    have : ((q : ℤ) : ZMod p) ≠ 0 := mod_cast prime_ne_zero p q h
    /-
      case inr
      p q : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Nat.Prime q)
      hp : Ne p 2
      hq : Ne q 2
      h : Ne p q
      qr : Eq ((fun x => HMul.hMul x (legendreSym p ↑q)) (HMul.hMul (legendreSym q ↑ …
      this : Ne (↑↑q) 0
      ⊢ Eq (legendreSym q ↑p) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv p 2)  …
    -/
    simpa only [mul_assoc, ← pow_two, sq_one p this, mul_one] using qr
    /-
      🎉 no goals
    -/


/-- The Law of Quadratic Reciprocity: if `p` and `q` are odd primes and `p % 4 = 1`,
then `(q / p) = (p / q)`. -/
theorem quadratic_reciprocity_one_mod_four (hp : p % 4 = 1) (hq : q ≠ 2) :
    legendreSym q p = legendreSym p q := by
  rw [quadratic_reciprocity' (Prime.mod_two_eq_one_iff_ne_two.mp (odd_of_mod_four_eq_one hp)) hq,
    pow_mul, neg_one_pow_div_two_of_one_mod_four hp, one_pow, one_mul]


/-- The Law of Quadratic Reciprocity: if `p` and `q` are primes that are both congruent
to `3` mod `4`, then `(q / p) = -(p / q)`. -/
theorem quadratic_reciprocity_three_mod_four (hp : p % 4 = 3) (hq : q % 4 = 3) :
    legendreSym q p = -legendreSym p q := by
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Eq (HMod.hMod p 4) 3
    hq : Eq (HMod.hMod q 4) 3
    ⊢ Eq (legendreSym q ↑p) (Neg.neg (legendreSym p ↑q))
  -/
  let nop := @neg_one_pow_div_two_of_three_mod_four
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Eq (HMod.hMod p 4) 3
    hq : Eq (HMod.hMod q 4) 3
    nop : ∀ {n : Nat}, Eq (HMod.hMod n 4) 3 → Eq (HPow.hPow (-1) (HDiv.hDiv n 2))  …
    ⊢ Eq (legendreSym q ↑p) (Neg.neg (legendreSym p ↑q))
  -/
  rw [quadratic_reciprocity', pow_mul, nop hp, nop hq, neg_one_mul] <;>
  /-
    case hp
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp : Eq (HMod.hMod p 4) 3
    hq : Eq (HMod.hMod q 4) 3
    nop : ∀ {n : Nat}, Eq (HMod.hMod n 4) 3 → Eq (HPow.hPow (-1) (HDiv.hDiv n 2))  …
    ⊢ Ne p 2
  -/
  /-
    🎉 no goals
  -/
  rwa [← Prime.mod_two_eq_one_iff_ne_two, odd_of_mod_four_eq_three]
  /-
    🎉 no goals
  -/


/-- If `p` and `q` are odd primes and `p % 4 = 1`, then `q` is a square mod `p` iff
`p` is a square mod `q`. -/
theorem exists_sq_eq_prime_iff_of_mod_four_eq_one (hp1 : p % 4 = 1) (hq1 : q ≠ 2) :
    IsSquare (q : ZMod p) ↔ IsSquare (p : ZMod q) := by
  /-
    p q : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Fact (Nat.Prime q)
    hp1 : Eq (HMod.hMod p 4) 1
    hq1 : Ne q 2
    ⊢ Iff (IsSquare ↑q) (IsSquare ↑p)
  -/
  rcases eq_or_ne p q with h | h
    /-
      case inl
      p q : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : Fact (Nat.Prime q)
      hp1 : Eq (HMod.hMod p 4) 1
      hq1 : Ne q 2
      h : Eq p q
      ⊢ Iff (IsSquare ↑q) (IsSquare ↑p)
    -/
  · subst p; rfl
             /-
               🎉 no goals
             -/
  · rw [← eq_one_iff' p (prime_ne_zero p q h), ← eq_one_iff' q (prime_ne_zero q p h.symm),
      quadratic_reciprocity_one_mod_four hp1 hq1]


/-- If `p` and `q` are distinct primes that are both congruent to `3` mod `4`, then `q` is
a square mod `p` iff `p` is a nonsquare mod `q`. -/
theorem exists_sq_eq_prime_iff_of_mod_four_eq_three (hp3 : p % 4 = 3) (hq3 : q % 4 = 3)
    (hpq : p ≠ q) : IsSquare (q : ZMod p) ↔ ¬IsSquare (p : ZMod q) := by
  rw [← eq_one_iff' p (prime_ne_zero p q hpq), ← eq_neg_one_iff' q,
    quadratic_reciprocity_three_mod_four hp3 hq3, neg_inj]


