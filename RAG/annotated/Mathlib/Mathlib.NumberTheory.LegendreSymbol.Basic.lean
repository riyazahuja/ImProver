/-- Euler's Criterion: A unit `x` of `ZMod p` is a square if and only if `x ^ (p / 2) = 1`. -/
theorem euler_criterion_units (x : (ZMod p)ˣ) : (∃ y : (ZMod p)ˣ, y ^ 2 = x) ↔ x ^ (p / 2) = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Units (ZMod p)
    ⊢ Iff (Exists fun y => Eq (HPow.hPow y 2) x) (Eq (HPow.hPow x (HDiv.hDiv p 2)) …
  -/
  by_cases hc : p = 2
    /-
      case pos
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x : Units (ZMod p)
      hc : Eq p 2
      ⊢ Iff (Exists fun y => Eq (HPow.hPow y 2) x) (Eq (HPow.hPow x (HDiv.hDiv p 2)) …
    -/
  · subst hc
    /-
      case pos
      inst✝ : Fact (Nat.Prime 2)
      x : Units (ZMod 2)
      ⊢ Iff (Exists fun y => Eq (HPow.hPow y 2) x) (Eq (HPow.hPow x (2 / 2)) 1)
    -/
    simp only [eq_iff_true_of_subsingleton, exists_const]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x : Units (ZMod p)
      hc : Not (Eq p 2)
      ⊢ Iff (Exists fun y => Eq (HPow.hPow y 2) x) (Eq (HPow.hPow x (HDiv.hDiv p 2)) …
    -/
  · have h₀ := FiniteField.unit_isSquare_iff (by rwa [ringChar_zmod_n]) x
    have hs : (∃ y : (ZMod p)ˣ, y ^ 2 = x) ↔ IsSquare x := by
      rw [isSquare_iff_exists_sq x]
      simp_rw [eq_comm]
    /-
      case neg
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x : Units (ZMod p)
      hc : Not (Eq p 2)
      h₀ : Iff (IsSquare x) (Eq (HPow.hPow x (HDiv.hDiv (Fintype.card (ZMod p)) 2)) 1)
      hs : Iff (Exists fun y => Eq (HPow.hPow y 2) x) (IsSquare x)
      ⊢ Iff (Exists fun y => Eq (HPow.hPow y 2) x) (Eq (HPow.hPow x (HDiv.hDiv p 2)) …
    -/
    rw [hs]
    /-
      case neg
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      x : Units (ZMod p)
      hc : Not (Eq p 2)
      h₀ : Iff (IsSquare x) (Eq (HPow.hPow x (HDiv.hDiv (Fintype.card (ZMod p)) 2)) 1)
      hs : Iff (Exists fun y => Eq (HPow.hPow y 2) x) (IsSquare x)
      ⊢ Iff (IsSquare x) (Eq (HPow.hPow x (HDiv.hDiv p 2)) 1)
    -/
    rwa [card p] at h₀
    /-
      🎉 no goals
    -/


/-- Euler's Criterion: a nonzero `a : ZMod p` is a square if and only if `x ^ (p / 2) = 1`. -/
theorem euler_criterion {a : ZMod p} (ha : a ≠ 0) : IsSquare (a : ZMod p) ↔ a ^ (p / 2) = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ha : Ne a 0
    ⊢ Iff (IsSquare a) (Eq (HPow.hPow a (HDiv.hDiv p 2)) 1)
  -/
  apply (iff_congr _ (by simp [Units.ext_iff])).mp (euler_criterion_units p (Units.mk0 a ha))
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ha : Ne a 0
    ⊢ Iff (Exists fun y => Eq (HPow.hPow y 2) (Units.mk0 a ha)) (IsSquare a)
  -/
  simp only [Units.ext_iff, sq, Units.val_mk0, Units.val_mul]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ha : Ne a 0
    ⊢ Iff (Exists fun y => Eq (HMul.hMul ↑y ↑y) a) (IsSquare a)
  -/
  constructor
    /-
      case mp
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Ne a 0
      ⊢ (Exists fun y => Eq (HMul.hMul ↑y ↑y) a) → IsSquare a
    -/
  · rintro ⟨y, hy⟩; exact ⟨y, hy.symm⟩
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Ne a 0
      ⊢ IsSquare a → Exists fun y => Eq (HMul.hMul ↑y ↑y) a
    -/
  · rintro ⟨y, rfl⟩
    have hy : y ≠ 0 := by
      rintro rfl
      simp [zero_pow, mul_zero, ne_eq, not_true] at ha
    /-
      case mpr.intro
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      y : ZMod p
      ha : Ne (HMul.hMul y y) 0
      hy : Ne y 0
      ⊢ Exists fun y_1 => Eq (HMul.hMul ↑y_1 ↑y_1) (HMul.hMul y y)
    -/
    refine ⟨Units.mk0 y hy, ?_⟩; simp
                                 /-
                                   🎉 no goals
                                 -/


/-- If `a : ZMod p` is nonzero, then `a^(p/2)` is either `1` or `-1`. -/
theorem pow_div_two_eq_neg_one_or_one {a : ZMod p} (ha : a ≠ 0) :
    a ^ (p / 2) = 1 ∨ a ^ (p / 2) = -1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ha : Ne a 0
    ⊢ Or (Eq (HPow.hPow a (HDiv.hDiv p 2)) 1) (Eq (HPow.hPow a (HDiv.hDiv p 2)) (- …
  -/
  cases' Prime.eq_two_or_odd (@Fact.out p.Prime _) with hp2 hp_odd
    /-
      case inl
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Ne a 0
      hp2 : Eq p 2
      ⊢ Or (Eq (HPow.hPow a (HDiv.hDiv p 2)) 1) (Eq (HPow.hPow a (HDiv.hDiv p 2)) (- …
    -/
  · subst p; revert a ha; intro a; fin_cases a
      /-
        case inl.«0»
        inst✝ : Fact (Nat.Prime 2)
        ⊢ Ne ((fun i => i) ⟨0, ⋯⟩) 0 → Or (Eq (HPow.hPow ((fun i => i) ⟨0, ⋯⟩) (2 / 2) …
      -/
    · tauto
      /-
        🎉 no goals
      -/
      /-
        case inl.«1»
        inst✝ : Fact (Nat.Prime 2)
        ⊢ Ne ((fun i => i) ⟨1, ⋯⟩) 0 → Or (Eq (HPow.hPow ((fun i => i) ⟨1, ⋯⟩) (2 / 2) …
      -/
    · simp
      /-
        🎉 no goals
      -/
  /-
    case inr
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ha : Ne a 0
    hp_odd : Eq (HMod.hMod p 2) 1
    ⊢ Or (Eq (HPow.hPow a (HDiv.hDiv p 2)) 1) (Eq (HPow.hPow a (HDiv.hDiv p 2)) (- …
  -/
  rw [← mul_self_eq_one_iff, ← pow_add, ← two_mul, two_mul_odd_div_two hp_odd]
  /-
    case inr
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ha : Ne a 0
    hp_odd : Eq (HMod.hMod p 2) 1
    ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) 1
  -/
  exact pow_card_sub_one_eq_one ha
  /-
    🎉 no goals
  -/


/-- The Legendre symbol of `a : ℤ` and a prime `p`, `legendreSym p a`,
is an integer defined as

* `0` if `a` is `0` modulo `p`;
* `1` if `a` is a nonzero square modulo `p`
* `-1` otherwise.

Note the order of the arguments! The advantage of the order chosen here is
that `legendreSym p` is a multiplicative function `ℤ → ℤ`.
-/
def legendreSym (a : ℤ) : ℤ :=
  quadraticChar (ZMod p) a


/-- We have the congruence `legendreSym p a ≡ a ^ (p / 2) mod p`. -/
theorem eq_pow (a : ℤ) : (legendreSym p a : ZMod p) = (a : ZMod p) ^ (p / 2) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ⊢ Eq (↑(legendreSym p a)) (HPow.hPow (↑a) (HDiv.hDiv p 2))
  -/
  rcases eq_or_ne (ringChar (ZMod p)) 2 with hc | hc
    /-
      case inl
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      hc : Eq (ringChar (ZMod p)) 2
      ⊢ Eq (↑(legendreSym p a)) (HPow.hPow (↑a) (HDiv.hDiv p 2))
    -/
  · by_cases ha : (a : ZMod p) = 0
    · rw [legendreSym, ha, quadraticChar_zero,
        zero_pow (Nat.div_pos (@Fact.out p.Prime).two_le (succ_pos 1)).ne']
      /-
        case pos
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        a : Int
        hc : Eq (ringChar (ZMod p)) 2
        ha : Eq (↑a) 0
        ⊢ Eq (↑0) 0
      -/
      norm_cast
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        a : Int
        hc : Eq (ringChar (ZMod p)) 2
        ha : Not (Eq (↑a) 0)
        ⊢ Eq (↑(legendreSym p a)) (HPow.hPow (↑a) (HDiv.hDiv p 2))
      -/
    · have := (ringChar_zmod_n p).symm.trans hc
      -- p = 2
      /-
        case neg
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        a : Int
        hc : Eq (ringChar (ZMod p)) 2
        ha : Not (Eq (↑a) 0)
        this : Eq p 2
        ⊢ Eq (↑(legendreSym p a)) (HPow.hPow (↑a) (HDiv.hDiv p 2))
      -/
      subst p
      /-
        case neg
        a : Int
        inst✝ : Fact (Nat.Prime 2)
        hc : Eq (ringChar (ZMod 2)) 2
        ha : Not (Eq (↑a) 0)
        ⊢ Eq (↑(legendreSym 2 a)) (HPow.hPow (↑a) (2 / 2))
      -/
      rw [legendreSym, quadraticChar_eq_one_of_char_two hc ha]
      /-
        case neg
        a : Int
        inst✝ : Fact (Nat.Prime 2)
        hc : Eq (ringChar (ZMod 2)) 2
        ha : Not (Eq (↑a) 0)
        ⊢ Eq (↑1) (HPow.hPow (↑a) (2 / 2))
      -/
      revert ha
      /-
        case neg
        a : Int
        inst✝ : Fact (Nat.Prime 2)
        hc : Eq (ringChar (ZMod 2)) 2
        ⊢ Not (Eq (↑a) 0) → Eq (↑1) (HPow.hPow (↑a) (2 / 2))
      -/
      push_cast
      /-
        case neg
        a : Int
        inst✝ : Fact (Nat.Prime 2)
        hc : Eq (ringChar (ZMod 2)) 2
        ⊢ Not (Eq (↑a) 0) → Eq 1 (HPow.hPow (↑a) 1)
      -/
      generalize (a : ZMod 2) = b; fin_cases b
        /-
          case neg.«_@»._hyg.683.«0»
          a : Int
          inst✝ : Fact (Nat.Prime 2)
          hc : Eq (ringChar (ZMod 2)) 2
          ⊢ Not (Eq ((fun i => i) ⟨0, ⋯⟩) 0) → Eq 1 (HPow.hPow ((fun i => i) ⟨0, ⋯⟩) 1)
        -/
      · tauto
        /-
          🎉 no goals
        -/
        /-
          case neg.«_@»._hyg.683.«1»
          a : Int
          inst✝ : Fact (Nat.Prime 2)
          hc : Eq (ringChar (ZMod 2)) 2
          ⊢ Not (Eq ((fun i => i) ⟨1, ⋯⟩) 0) → Eq 1 (HPow.hPow ((fun i => i) ⟨1, ⋯⟩) 1)
        -/
      · simp
        /-
          🎉 no goals
        -/
    /-
      case inr
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      hc : Ne (ringChar (ZMod p)) 2
      ⊢ Eq (↑(legendreSym p a)) (HPow.hPow (↑a) (HDiv.hDiv p 2))
    -/
  · convert quadraticChar_eq_pow_of_char_ne_two' hc (a : ZMod p)
    /-
      case h.e'_3.h.e'_6.h.e'_5
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      hc : Ne (ringChar (ZMod p)) 2
      ⊢ Eq p (Fintype.card (ZMod p))
    -/
    exact (card p).symm
    /-
      🎉 no goals
    -/


/-- If `p ∤ a`, then `legendreSym p a` is `1` or `-1`. -/
theorem eq_one_or_neg_one {a : ℤ} (ha : (a : ZMod p) ≠ 0) :
    legendreSym p a = 1 ∨ legendreSym p a = -1 :=
  quadraticChar_dichotomy ha


theorem eq_neg_one_iff_not_one {a : ℤ} (ha : (a : ZMod p) ≠ 0) :
    legendreSym p a = -1 ↔ ¬legendreSym p a = 1 :=
  quadraticChar_eq_neg_one_iff_not_one ha


/-- The Legendre symbol of `p` and `a` is zero iff `p ∣ a`. -/
theorem eq_zero_iff (a : ℤ) : legendreSym p a = 0 ↔ (a : ZMod p) = 0 :=
  quadraticChar_eq_zero_iff


@[simp]
                                            /-
                                              p : Nat
                                              inst✝ : Fact (Nat.Prime p)
                                              ⊢ Eq (legendreSym p 0) 0
                                            -/
theorem at_zero : legendreSym p 0 = 0 := by rw [legendreSym, Int.cast_zero, MulChar.map_zero]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                           /-
                                             p : Nat
                                             inst✝ : Fact (Nat.Prime p)
                                             ⊢ Eq (legendreSym p 1) 1
                                           -/
theorem at_one : legendreSym p 1 = 1 := by rw [legendreSym, Int.cast_one, MulChar.map_one]
                                           /-
                                             🎉 no goals
                                           -/


/-- The Legendre symbol is multiplicative in `a` for `p` fixed. -/
protected theorem mul (a b : ℤ) : legendreSym p (a * b) = legendreSym p a * legendreSym p b := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a b : Int
    ⊢ Eq (legendreSym p (HMul.hMul a b)) (HMul.hMul (legendreSym p a) (legendreSym …
  -/
  simp [legendreSym, Int.cast_mul, map_mul, quadraticCharFun_mul]
  /-
    🎉 no goals
  -/


/-- The Legendre symbol is a homomorphism of monoids with zero. -/
@[simps]
def hom : ℤ →*₀ ℤ where
  toFun := legendreSym p
  map_zero' := at_zero p
  map_one' := at_one p
  map_mul' := legendreSym.mul p


/-- The square of the symbol is 1 if `p ∤ a`. -/
theorem sq_one {a : ℤ} (ha : (a : ZMod p) ≠ 0) : legendreSym p a ^ 2 = 1 :=
  quadraticChar_sq_one ha


/-- The Legendre symbol of `a^2` at `p` is 1 if `p ∤ a`. -/
theorem sq_one' {a : ℤ} (ha : (a : ZMod p) ≠ 0) : legendreSym p (a ^ 2) = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    ⊢ Eq (legendreSym p (HPow.hPow a 2)) 1
  -/
  dsimp only [legendreSym]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    ⊢ Eq ((quadraticChar (ZMod p)) ↑(HPow.hPow a 2)) 1
  -/
  rw [Int.cast_pow]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    ⊢ Eq ((quadraticChar (ZMod p)) (HPow.hPow (↑a) 2)) 1
  -/
  exact quadraticChar_sq_one' ha
  /-
    🎉 no goals
  -/


/-- The Legendre symbol depends only on `a` mod `p`. -/
protected theorem mod (a : ℤ) : legendreSym p a = legendreSym p (a % p) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ⊢ Eq (legendreSym p a) (legendreSym p (HMod.hMod a ↑p))
  -/
  simp only [legendreSym, intCast_mod]
  /-
    🎉 no goals
  -/


/-- When `p ∤ a`, then `legendreSym p a = 1` iff `a` is a square mod `p`. -/
theorem eq_one_iff {a : ℤ} (ha0 : (a : ZMod p) ≠ 0) : legendreSym p a = 1 ↔ IsSquare (a : ZMod p) :=
  quadraticChar_one_iff_isSquare ha0


theorem eq_one_iff' {a : ℕ} (ha0 : (a : ZMod p) ≠ 0) :
    legendreSym p a = 1 ↔ IsSquare (a : ZMod p) := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        a : Nat
        ha0 : Ne (↑a) 0
        ⊢ Iff (Eq (legendreSym p ↑a) 1) (IsSquare ↑a)
      -/
      rw [eq_one_iff]
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          a : Nat
          ha0 : Ne (↑a) 0
          ⊢ Iff (IsSquare ↑↑a) (IsSquare ↑a)
        -/
      · norm_cast
        /-
          🎉 no goals
        -/
        /-
          case ha0
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          a : Nat
          ha0 : Ne (↑a) 0
          ⊢ Ne (↑↑a) 0
        -/
      · exact mod_cast ha0
        /-
          🎉 no goals
        -/


/-- `legendreSym p a = -1` iff `a` is a nonsquare mod `p`. -/
theorem eq_neg_one_iff {a : ℤ} : legendreSym p a = -1 ↔ ¬IsSquare (a : ZMod p) :=
  quadraticChar_neg_one_iff_not_isSquare


theorem eq_neg_one_iff' {a : ℕ} : legendreSym p a = -1 ↔ ¬IsSquare (a : ZMod p) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Nat
    ⊢ Iff (Eq (legendreSym p ↑a) (-1)) (Not (IsSquare ↑a))
  -/
  rw [eq_neg_one_iff]; norm_cast
                       /-
                         🎉 no goals
                       -/


/-- The number of square roots of `a` modulo `p` is determined by the Legendre symbol. -/
theorem card_sqrts (hp : p ≠ 2) (a : ℤ) :
    ↑{x : ZMod p | x ^ 2 = a}.toFinset.card = legendreSym p a + 1 :=
  quadraticChar_card_sqrts ((ringChar_zmod_n p).substr hp) a


/-- The Legendre symbol `legendreSym p a = 1` if there is a solution in `ℤ/pℤ`
of the equation `x^2 - a*y^2 = 0` with `y ≠ 0`. -/
theorem eq_one_of_sq_sub_mul_sq_eq_zero {p : ℕ} [Fact p.Prime] {a : ℤ} (ha : (a : ZMod p) ≠ 0)
    {x y : ZMod p} (hy : y ≠ 0) (hxy : x ^ 2 - a * y ^ 2 = 0) : legendreSym p a = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    x y : ZMod p
    hy : Ne y 0
    hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
    ⊢ Eq (legendreSym p a) 1
  -/
  apply_fun (· * y⁻¹ ^ 2) at hxy
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    x y : ZMod p
    hy : Ne y 0
    hxy : Eq (HMul.hMul (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2) …
    ⊢ Eq (legendreSym p a) 1
  -/
  simp only [zero_mul] at hxy
  rw [(by ring : (x ^ 2 - ↑a * y ^ 2) * y⁻¹ ^ 2 = (x * y⁻¹) ^ 2 - a * (y * y⁻¹) ^ 2),
    mul_inv_cancel₀ hy, one_pow, mul_one, sub_eq_zero, pow_two] at hxy
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    x y : ZMod p
    hy : Ne y 0
    hxy : Eq (HMul.hMul (HMul.hMul x (Inv.inv y)) (HMul.hMul x (Inv.inv y))) ↑a
    ⊢ Eq (legendreSym p a) 1
  -/
  exact (eq_one_iff p ha).mpr ⟨x * y⁻¹, hxy.symm⟩
  /-
    🎉 no goals
  -/


/-- The Legendre symbol `legendreSym p a = 1` if there is a solution in `ℤ/pℤ`
of the equation `x^2 - a*y^2 = 0` with `x ≠ 0`. -/
theorem eq_one_of_sq_sub_mul_sq_eq_zero' {p : ℕ} [Fact p.Prime] {a : ℤ} (ha : (a : ZMod p) ≠ 0)
    {x y : ZMod p} (hx : x ≠ 0) (hxy : x ^ 2 - a * y ^ 2 = 0) : legendreSym p a = 1 := by
  haveI hy : y ≠ 0 := by
    rintro rfl
    rw [zero_pow two_ne_zero, mul_zero, sub_zero, sq_eq_zero_iff] at hxy
    exact hx hxy
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    ha : Ne (↑a) 0
    x y : ZMod p
    hx : Ne x 0
    hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
    hy : Ne y 0
    ⊢ Eq (legendreSym p a) 1
  -/
  exact eq_one_of_sq_sub_mul_sq_eq_zero ha hy hxy
  /-
    🎉 no goals
  -/


/-- If `legendreSym p a = -1`, then the only solution of `x^2 - a*y^2 = 0` in `ℤ/pℤ`
is the trivial one. -/
theorem eq_zero_mod_of_eq_neg_one {p : ℕ} [Fact p.Prime] {a : ℤ} (h : legendreSym p a = -1)
    {x y : ZMod p} (hxy : x ^ 2 - a * y ^ 2 = 0) : x = 0 ∧ y = 0 := by
  have ha : (a : ZMod p) ≠ 0 := by
    intro hf
    rw [(eq_zero_iff p a).mpr hf] at h
    simp at h
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (legendreSym p a) (-1)
    x y : ZMod p
    hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
    ha : Ne (↑a) 0
    ⊢ And (Eq x 0) (Eq y 0)
  -/
  by_contra hf
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (legendreSym p a) (-1)
    x y : ZMod p
    hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
    ha : Ne (↑a) 0
    hf : Not (And (Eq x 0) (Eq y 0))
    ⊢ False
  -/
  cases' imp_iff_or_not.mp (not_and'.mp hf) with hx hy
    /-
      case inl
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      h : Eq (legendreSym p a) (-1)
      x y : ZMod p
      hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
      ha : Ne (↑a) 0
      hf : Not (And (Eq x 0) (Eq y 0))
      hx : Not (Eq x 0)
      ⊢ False
    -/
  · rw [eq_one_of_sq_sub_mul_sq_eq_zero' ha hx hxy, CharZero.eq_neg_self_iff] at h
    /-
      case inl
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      h : Eq 1 0
      x y : ZMod p
      hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
      ha : Ne (↑a) 0
      hf : Not (And (Eq x 0) (Eq y 0))
      hx : Not (Eq x 0)
      ⊢ False
    -/
    exact one_ne_zero h
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      h : Eq (legendreSym p a) (-1)
      x y : ZMod p
      hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
      ha : Ne (↑a) 0
      hf : Not (And (Eq x 0) (Eq y 0))
      hy : Not (Eq y 0)
      ⊢ False
    -/
  · rw [eq_one_of_sq_sub_mul_sq_eq_zero ha hy hxy, CharZero.eq_neg_self_iff] at h
    /-
      case inr
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : Int
      h : Eq 1 0
      x y : ZMod p
      hxy : Eq (HSub.hSub (HPow.hPow x 2) (HMul.hMul (↑a) (HPow.hPow y 2))) 0
      ha : Ne (↑a) 0
      hf : Not (And (Eq x 0) (Eq y 0))
      hy : Not (Eq y 0)
      ⊢ False
    -/
    exact one_ne_zero h
    /-
      🎉 no goals
    -/


/-- If `legendreSym p a = -1` and `p` divides `x^2 - a*y^2`, then `p` must divide `x` and `y`. -/
theorem prime_dvd_of_eq_neg_one {p : ℕ} [Fact p.Prime] {a : ℤ} (h : legendreSym p a = -1) {x y : ℤ}
    (hxy : (p : ℤ) ∣ x ^ 2 - a * y ^ 2) : ↑p ∣ x ∧ ↑p ∣ y := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (legendreSym p a) (-1)
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow x 2) (HMul.hMul a (HPow.hPow y 2)))
    ⊢ And (Dvd.dvd (↑p) x) (Dvd.dvd (↑p) y)
  -/
  simp_rw [← ZMod.intCast_zmod_eq_zero_iff_dvd] at hxy ⊢
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (legendreSym p a) (-1)
    x y : Int
    hxy : Eq (↑(HSub.hSub (HPow.hPow x 2) (HMul.hMul a (HPow.hPow y 2)))) 0
    ⊢ And (Eq (↑x) 0) (Eq (↑y) 0)
  -/
  push_cast at hxy
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (legendreSym p a) (-1)
    x y : Int
    hxy : Eq (HSub.hSub (HPow.hPow (↑x) 2) (HMul.hMul (↑a) (HPow.hPow (↑y) 2))) 0
    ⊢ And (Eq (↑x) 0) (Eq (↑y) 0)
  -/
  exact eq_zero_mod_of_eq_neg_one h hxy
  /-
    🎉 no goals
  -/


/-- `legendreSym p (-1)` is given by `χ₄ p`. -/
theorem legendreSym.at_neg_one (hp : p ≠ 2) : legendreSym p (-1) = χ₄ p := by
  simp only [legendreSym, card p, quadraticChar_neg_one ((ringChar_zmod_n p).substr hp),
    Int.cast_neg, Int.cast_one]


/-- `-1` is a square in `ZMod p` iff `p` is not congruent to `3` mod `4`. -/
theorem exists_sq_eq_neg_one_iff : IsSquare (-1 : ZMod p) ↔ p % 4 ≠ 3 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Iff (IsSquare (-1)) (Ne (HMod.hMod p 4) 3)
  -/
  rw [FiniteField.isSquare_neg_one_iff, card p]
  /-
    🎉 no goals
  -/


theorem mod_four_ne_three_of_sq_eq_neg_one {y : ZMod p} (hy : y ^ 2 = -1) : p % 4 ≠ 3 :=
  exists_sq_eq_neg_one_iff.1 ⟨y, hy ▸ pow_two y⟩


/-- If two nonzero squares are negatives of each other in `ZMod p`, then `p % 4 ≠ 3`. -/
theorem mod_four_ne_three_of_sq_eq_neg_sq' {x y : ZMod p} (hy : y ≠ 0) (hxy : x ^ 2 = -y ^ 2) :
    p % 4 ≠ 3 :=
  @mod_four_ne_three_of_sq_eq_neg_one p _ (x / y)
    (by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        x y : ZMod p
        hy : Ne y 0
        hxy : Eq (HPow.hPow x 2) (Neg.neg (HPow.hPow y 2))
        ⊢ Eq (HPow.hPow (HDiv.hDiv x y) 2) (-1)
      -/
      apply_fun fun z => z / y ^ 2 at hxy
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        x y : ZMod p
        hy : Ne y 0
        hxy : Eq (HDiv.hDiv (HPow.hPow x 2) (HPow.hPow y 2)) (HDiv.hDiv (Neg.neg (HPow …
        ⊢ Eq (HPow.hPow (HDiv.hDiv x y) 2) (-1)
      -/
      rwa [neg_div, ← div_pow, ← div_pow, div_self hy, one_pow] at hxy)
      /-
        🎉 no goals
      -/


theorem mod_four_ne_three_of_sq_eq_neg_sq {x y : ZMod p} (hx : x ≠ 0) (hxy : x ^ 2 = -y ^ 2) :
    p % 4 ≠ 3 :=
  mod_four_ne_three_of_sq_eq_neg_sq' hx (neg_eq_iff_eq_neg.mpr hxy).symm


