/-- A prime `p` is a unit in a commutative ring `R` of nonzero characteristic iff it does not divide
the characteristic. -/
theorem isUnit_iff_not_dvd_char_of_ringChar_ne_zero (R : Type*) [CommRing R] (p : ℕ) [Fact p.Prime]
    (hR : ringChar R ≠ 0) : IsUnit (p : R) ↔ ¬p ∣ ringChar R := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hR : Ne (ringChar R) 0
    ⊢ Iff (IsUnit ↑p) (Not (Dvd.dvd p (ringChar R)))
  -/
  have hch := CharP.cast_eq_zero R (ringChar R)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hR : Ne (ringChar R) 0
    hch : Eq (↑(ringChar R)) 0
    ⊢ Iff (IsUnit ↑p) (Not (Dvd.dvd p (ringChar R)))
  -/
  have hp : p.Prime := Fact.out
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    hR : Ne (ringChar R) 0
    hch : Eq (↑(ringChar R)) 0
    hp : Nat.Prime p
    ⊢ Iff (IsUnit ↑p) (Not (Dvd.dvd p (ringChar R)))
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      ⊢ IsUnit ↑p → Not (Dvd.dvd p (ringChar R))
    -/
  · rintro h₁ ⟨q, hq⟩
    /-
      case mp.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      hq : Eq (ringChar R) (HMul.hMul p q)
      ⊢ False
    -/
    rcases IsUnit.exists_left_inv h₁ with ⟨a, ha⟩
    have h₃ : ¬ringChar R ∣ q := by
      rintro ⟨r, hr⟩
      rw [hr, ← mul_assoc, mul_comm p, mul_assoc] at hq
      nth_rw 1 [← mul_one (ringChar R)] at hq
      exact Nat.Prime.not_dvd_one hp ⟨r, mul_left_cancel₀ hR hq⟩
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      hq : Eq (ringChar R) (HMul.hMul p q)
      a : R
      ha : Eq (HMul.hMul a ↑p) 1
      h₃ : Not (Dvd.dvd (ringChar R) q)
      ⊢ False
    -/
    have h₄ := mt (CharP.intCast_eq_zero_iff R (ringChar R) q).mp
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      hq : Eq (ringChar R) (HMul.hMul p q)
      a : R
      ha : Eq (HMul.hMul a ↑p) 1
      h₃ : Not (Dvd.dvd (ringChar R) q)
      h₄ : Not (Dvd.dvd ↑(ringChar R) ↑q) → Not (Eq (↑↑q) 0)
      ⊢ False
    -/
    apply_fun ((↑) : ℕ → R) at hq
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      a : R
      ha : Eq (HMul.hMul a ↑p) 1
      h₃ : Not (Dvd.dvd (ringChar R) q)
      h₄ : Not (Dvd.dvd ↑(ringChar R) ↑q) → Not (Eq (↑↑q) 0)
      hq : Eq ↑(ringChar R) ↑(HMul.hMul p q)
      ⊢ False
    -/
    apply_fun (· * ·) a at hq
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      a : R
      ha : Eq (HMul.hMul a ↑p) 1
      h₃ : Not (Dvd.dvd (ringChar R) q)
      h₄ : Not (Dvd.dvd ↑(ringChar R) ↑q) → Not (Eq (↑↑q) 0)
      hq : Eq (HMul.hMul a ↑(ringChar R)) (HMul.hMul a ↑(HMul.hMul p q))
      ⊢ False
    -/
    rw [Nat.cast_mul, hch, mul_zero, ← mul_assoc, ha, one_mul] at hq
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      a : R
      ha : Eq (HMul.hMul a ↑p) 1
      h₃ : Not (Dvd.dvd (ringChar R) q)
      h₄ : Not (Dvd.dvd ↑(ringChar R) ↑q) → Not (Eq (↑↑q) 0)
      hq : Eq 0 ↑q
      ⊢ False
    -/
    norm_cast at h₄
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h₁ : IsUnit ↑p
      q : Nat
      a : R
      ha : Eq (HMul.hMul a ↑p) 1
      h₃ : Not (Dvd.dvd (ringChar R) q)
      hq : Eq 0 ↑q
      h₄ : Not (Dvd.dvd (ringChar R) q) → Not (Eq (↑q) 0)
      ⊢ False
    -/
    exact h₄ h₃ hq.symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      ⊢ Not (Dvd.dvd p (ringChar R)) → IsUnit ↑p
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h : Not (Dvd.dvd p (ringChar R))
      ⊢ IsUnit ↑p
    -/
    rcases (hp.coprime_iff_not_dvd.mpr h).isCoprime with ⟨a, b, hab⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h : Not (Dvd.dvd p (ringChar R))
      a b : Int
      hab : Eq (HAdd.hAdd (HMul.hMul a ↑p) (HMul.hMul b ↑(ringChar R))) 1
      ⊢ IsUnit ↑p
    -/
    apply_fun ((↑) : ℤ → R) at hab
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h : Not (Dvd.dvd p (ringChar R))
      a b : Int
      hab : Eq ↑(HAdd.hAdd (HMul.hMul a ↑p) (HMul.hMul b ↑(ringChar R))) ↑1
      ⊢ IsUnit ↑p
    -/
    push_cast at hab
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h : Not (Dvd.dvd p (ringChar R))
      a b : Int
      hab : Eq (HAdd.hAdd (HMul.hMul ↑a ↑p) (HMul.hMul ↑b ↑(ringChar R))) 1
      ⊢ IsUnit ↑p
    -/
    rw [hch, mul_zero, add_zero, mul_comm] at hab
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      hR : Ne (ringChar R) 0
      hch : Eq (↑(ringChar R)) 0
      hp : Nat.Prime p
      h : Not (Dvd.dvd p (ringChar R))
      a b : Int
      hab : Eq (HMul.hMul ↑p ↑a) 1
      ⊢ IsUnit ↑p
    -/
    exact isUnit_of_mul_eq_one (p : R) a hab
    /-
      🎉 no goals
    -/


/-- A prime `p` is a unit in a finite commutative ring `R`
iff it does not divide the characteristic. -/
theorem isUnit_iff_not_dvd_char (R : Type*) [CommRing R] (p : ℕ) [Fact p.Prime] [Finite R] :
    IsUnit (p : R) ↔ ¬p ∣ ringChar R :=
  isUnit_iff_not_dvd_char_of_ringChar_ne_zero R p <| CharP.char_ne_zero_of_finite R (ringChar R)


/-- The prime divisors of the characteristic of a finite commutative ring are exactly
the prime divisors of its cardinality. -/
theorem prime_dvd_char_iff_dvd_card {R : Type*} [CommRing R] [Fintype R] (p : ℕ) [Fact p.Prime] :
    p ∣ ringChar R ↔ p ∣ Fintype.card R := by
  refine
    ⟨fun h =>
      h.trans <|
        Int.natCast_dvd_natCast.mp <|
          (CharP.intCast_eq_zero_iff R (ringChar R) (Fintype.card R)).mp <|
            mod_cast Nat.cast_card_eq_zero R,
      fun h => ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    ⊢ Dvd.dvd p (ringChar R)
  -/
  by_contra h₀
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    ⊢ False
  -/
  rcases exists_prime_addOrderOf_dvd_card p h with ⟨r, hr⟩
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    r : R
    hr : Eq (addOrderOf r) p
    ⊢ False
  -/
  have hr₁ := addOrderOf_nsmul_eq_zero r
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    r : R
    hr : Eq (addOrderOf r) p
    hr₁ : Eq (HSMul.hSMul (addOrderOf r) r) 0
    ⊢ False
  -/
  rw [hr, nsmul_eq_mul] at hr₁
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    r : R
    hr : Eq (addOrderOf r) p
    hr₁ : Eq (HMul.hMul (↑p) r) 0
    ⊢ False
  -/
  rcases IsUnit.exists_left_inv ((isUnit_iff_not_dvd_char R p).mpr h₀) with ⟨u, hu⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    r : R
    hr : Eq (addOrderOf r) p
    hr₁ : Eq (HMul.hMul (↑p) r) 0
    u : R
    hu : Eq (HMul.hMul u ↑p) 1
    ⊢ False
  -/
  apply_fun (· * ·) u at hr₁
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    r : R
    hr : Eq (addOrderOf r) p
    u : R
    hu : Eq (HMul.hMul u ↑p) 1
    hr₁ : Eq (HMul.hMul u (HMul.hMul (↑p) r)) (HMul.hMul u 0)
    ⊢ False
  -/
  rw [mul_zero, ← mul_assoc, hu, one_mul] at hr₁
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    h : Dvd.dvd p (Fintype.card R)
    h₀ : Not (Dvd.dvd p (ringChar R))
    r : R
    hr : Eq (addOrderOf r) p
    u : R
    hu : Eq (HMul.hMul u ↑p) 1
    hr₁ : Eq r 0
    ⊢ False
  -/
  exact mt AddMonoid.addOrderOf_eq_one_iff.mpr (ne_of_eq_of_ne hr (Nat.Prime.ne_one Fact.out)) hr₁
  /-
    🎉 no goals
  -/


/-- A prime that divides the cardinality of a finite commutative ring `R`
isn't a unit in `R`. -/
theorem not_isUnit_prime_of_dvd_card {R : Type*} [CommRing R] [Fintype R] (p : ℕ) [Fact p.Prime]
    (hp : p ∣ Fintype.card R) : ¬IsUnit (p : R) :=
  mt (isUnit_iff_not_dvd_char R p).mp
    (Classical.not_not.mpr ((prime_dvd_char_iff_dvd_card p).mpr hp))


lemma charP_of_card_eq_prime {R : Type*} [NonAssocRing R] [Fintype R] (p : ℕ) [hp : Fact p.Prime]
    (hR : Fintype.card R = p) : CharP R p :=
  have := Fintype.one_lt_card_iff_nontrivial.1 (hR ▸ hp.1.one_lt)
  (CharP.charP_iff_prime_eq_zero hp.1).2 (hR ▸ Nat.cast_card_eq_zero R)

