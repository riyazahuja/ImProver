/-- `modPart p r` is an integer that satisfies
`‖(r - modPart p r : ℚ_[p])‖ < 1` when `‖(r : ℚ_[p])‖ ≤ 1`,
see `PadicInt.norm_sub_modPart`.
It is the unique non-negative integer that is `< p` with this property.

(Note that this definition assumes `r : ℚ`.
See `PadicInt.zmodRepr` for a version that takes values in `ℕ`
and works for arbitrary `x : ℤ_[p]`.) -/
def modPart : ℤ :=
  r.num * gcdA r.den p % p


theorem modPart_lt_p : modPart p r < p := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    ⊢ LT.lt (PadicInt.modPart p r) ↑p
  -/
  convert Int.emod_lt _ _
    /-
      case h.e'_4
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      ⊢ Eq (↑p) (abs ↑p)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      ⊢ Ne (↑p) 0
    -/
  · exact mod_cast hp_prime.1.ne_zero
    /-
      🎉 no goals
    -/


theorem modPart_nonneg : 0 ≤ modPart p r :=
  Int.emod_nonneg _ <| mod_cast hp_prime.1.ne_zero


theorem isUnit_den (r : ℚ) (h : ‖(r : ℚ_[p])‖ ≤ 1) : IsUnit (r.den : ℤ_[p]) := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ IsUnit ↑r.den
  -/
  rw [isUnit_iff]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ Eq (Norm.norm ↑r.den) 1
  -/
  apply le_antisymm (r.den : ℤ_[p]).2
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ LE.le 1 (Norm.norm ↑↑r.den)
  -/
  rw [← not_lt, coe_natCast]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ Not (LT.lt (Norm.norm ↑r.den) 1)
  -/
  intro norm_denom_lt
  have hr : ‖(r * r.den : ℚ_[p])‖ = ‖(r.num : ℚ_[p])‖ := by
    congr
    rw_mod_cast [@Rat.mul_den_eq_num r]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    norm_denom_lt : LT.lt (Norm.norm ↑r.den) 1
    hr : Eq (Norm.norm (HMul.hMul ↑r ↑r.den)) (Norm.norm ↑r.num)
    ⊢ False
  -/
  rw [padicNormE.mul] at hr
  have key : ‖(r.num : ℚ_[p])‖ < 1 := by
    calc
      _ = _ := hr.symm
      _ < 1 * 1 := mul_lt_mul' h norm_denom_lt (norm_nonneg _) zero_lt_one
      _ = 1 := mul_one 1

  have : ↑p ∣ r.num ∧ (p : ℤ) ∣ r.den := by
    simp only [← norm_int_lt_one_iff_dvd, ← padic_norm_e_of_padicInt]
    exact ⟨key, norm_denom_lt⟩
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    norm_denom_lt : LT.lt (Norm.norm ↑r.den) 1
    hr : Eq (HMul.hMul (Norm.norm ↑r) (Norm.norm ↑r.den)) (Norm.norm ↑r.num)
    key : LT.lt (Norm.norm ↑r.num) 1
    this : And (Dvd.dvd (↑p) r.num) (Dvd.dvd ↑p ↑r.den)
    ⊢ False
  -/
  apply hp_prime.1.not_dvd_one
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    norm_denom_lt : LT.lt (Norm.norm ↑r.den) 1
    hr : Eq (HMul.hMul (Norm.norm ↑r) (Norm.norm ↑r.den)) (Norm.norm ↑r.num)
    key : LT.lt (Norm.norm ↑r.num) 1
    this : And (Dvd.dvd (↑p) r.num) (Dvd.dvd ↑p ↑r.den)
    ⊢ Dvd.dvd p 1
  -/
  rwa [← r.reduced.gcd_eq_one, Nat.dvd_gcd_iff, ← Int.natCast_dvd, ← Int.natCast_dvd_natCast]
  /-
    🎉 no goals
  -/


theorem norm_sub_modPart_aux (r : ℚ) (h : ‖(r : ℚ_[p])‖ ≤ 1) :
    ↑p ∣ r.num - r.num * r.den.gcdA p % p * ↑r.den := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ Dvd.dvd (↑p) (HSub.hSub r.num (HMul.hMul (HMod.hMod (HMul.hMul r.num (r.den. …
  -/
  rw [← ZMod.intCast_zmod_eq_zero_iff_dvd]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ Eq (↑(HSub.hSub r.num (HMul.hMul (HMod.hMod (HMul.hMul r.num (r.den.gcdA p)) …
  -/
  simp only [Int.cast_natCast, ZMod.natCast_mod, Int.cast_mul, Int.cast_sub]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ Eq (HSub.hSub (↑r.num) (HMul.hMul ↑(HMod.hMod (HMul.hMul r.num (r.den.gcdA p …
  -/
  have := congr_arg (fun x => x % p : ℤ → ZMod p) (gcd_eq_gcd_ab r.den p)
  simp only [Int.cast_natCast, CharP.cast_eq_zero, EuclideanDomain.mod_zero, Int.cast_add,
    Int.cast_mul, zero_mul, add_zero] at this
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ Eq (HSub.hSub (↑r.num) (HMul.hMul ↑(HMod.hMod (HMul.hMul r.num (r.den.gcdA p …
  -/
  push_cast
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ Eq (HSub.hSub (↑r.num) (HMul.hMul (HMul.hMul ↑r.num ↑(r.den.gcdA p)) ↑r.den) …
  -/
  rw [mul_right_comm, mul_assoc, ← this]
  suffices rdcp : r.den.Coprime p by
    rw [rdcp.gcd_eq_one]
    simp only [mul_one, cast_one, sub_self]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ r.den.Coprime p
  -/
  apply Coprime.symm
  /-
    case a
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ p.Coprime r.den
  -/
  apply (coprime_or_dvd_of_prime hp_prime.1 _).resolve_right
  /-
    case a
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ Not (Dvd.dvd p r.den)
  -/
  rw [← Int.natCast_dvd_natCast, ← norm_int_lt_one_iff_dvd, not_lt]
  /-
    case a
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ LE.le 1 (Norm.norm ↑↑r.den)
  -/
  apply ge_of_eq
  /-
    case a.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ Eq (Norm.norm ↑↑r.den) 1
  -/
  rw [← isUnit_iff]
  /-
    case a.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    this : Eq (↑(r.den.gcd p)) (HMul.hMul ↑r.den ↑(r.den.gcdA p))
    ⊢ IsUnit ↑↑r.den
  -/
  exact isUnit_den r h
  /-
    🎉 no goals
  -/


theorem norm_sub_modPart (h : ‖(r : ℚ_[p])‖ ≤ 1) : ‖(⟨r, h⟩ - modPart p r : ℤ_[p])‖ < 1 := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    ⊢ LT.lt (Norm.norm (HSub.hSub ⟨↑r, h⟩ ↑(PadicInt.modPart p r))) 1
  -/
  let n := modPart p r
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    n : Int := PadicInt.modPart p r
    ⊢ LT.lt (Norm.norm (HSub.hSub ⟨↑r, h⟩ ↑(PadicInt.modPart p r))) 1
  -/
  rw [norm_lt_one_iff_dvd, ← (isUnit_den r h).dvd_mul_right]
  suffices ↑p ∣ r.num - n * r.den by
    convert (Int.castRingHom ℤ_[p]).map_dvd this
    simp only [n, sub_mul, Int.cast_natCast, eq_intCast, Int.cast_mul, sub_left_inj,
      Int.cast_sub]
    apply Subtype.coe_injective
    simp only [coe_mul, Subtype.coe_mk, coe_natCast]
    rw_mod_cast [@Rat.mul_den_eq_num r]
    rfl
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    r : Rat
    h : LE.le (Norm.norm ↑r) 1
    n : Int := PadicInt.modPart p r
    ⊢ Dvd.dvd (↑p) (HSub.hSub r.num (HMul.hMul n ↑r.den))
  -/
  exact norm_sub_modPart_aux r h
  /-
    🎉 no goals
  -/


theorem exists_mem_range_of_norm_rat_le_one (h : ‖(r : ℚ_[p])‖ ≤ 1) :
    ∃ n : ℤ, 0 ≤ n ∧ n < p ∧ ‖(⟨r, h⟩ - n : ℤ_[p])‖ < 1 :=
  ⟨modPart p r, modPart_nonneg _, modPart_lt_p _, norm_sub_modPart _ h⟩


theorem zmod_congr_of_sub_mem_span_aux (n : ℕ) (x : ℤ_[p]) (a b : ℤ)
    (ha : x - a ∈ (Ideal.span {(p : ℤ_[p]) ^ n}))
    (hb : x - b ∈ (Ideal.span {(p : ℤ_[p]) ^ n})) : (a : ZMod (p ^ n)) = b := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    a b : Int
    ha : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSu …
    hb : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSu …
    ⊢ Eq ↑a ↑b
  -/
  rw [Ideal.mem_span_singleton] at ha hb
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    a b : Int
    ha : Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑a)
    hb : Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑b)
    ⊢ Eq ↑a ↑b
  -/
  rw [← sub_eq_zero, ← Int.cast_sub, ZMod.intCast_zmod_eq_zero_iff_dvd, Int.natCast_pow]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    a b : Int
    ha : Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑a)
    hb : Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑b)
    ⊢ Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub a b)
  -/
  rw [← dvd_neg, neg_sub] at ha
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    a b : Int
    ha : Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub (↑a) x)
    hb : Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑b)
    ⊢ Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub a b)
  -/
  have := dvd_add ha hb
  rwa [sub_eq_add_neg, sub_eq_add_neg, add_assoc, neg_add_cancel_left, ← sub_eq_add_neg, ←
    Int.cast_sub, pow_p_dvd_int_iff] at this


theorem zmod_congr_of_sub_mem_span (n : ℕ) (x : ℤ_[p]) (a b : ℕ)
    (ha : x - a ∈ (Ideal.span {(p : ℤ_[p]) ^ n}))
    (hb : x - b ∈ (Ideal.span {(p : ℤ_[p]) ^ n})) : (a : ZMod (p ^ n)) = b := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    a b : Nat
    ha : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSu …
    hb : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSu …
    ⊢ Eq ↑a ↑b
  -/
  simpa using zmod_congr_of_sub_mem_span_aux n x a b ha hb
  /-
    🎉 no goals
  -/


theorem zmod_congr_of_sub_mem_max_ideal (x : ℤ_[p]) (m n : ℕ) (hm : x - m ∈ maximalIdeal ℤ_[p])
    (hn : x - n ∈ maximalIdeal ℤ_[p]) : (m : ZMod p) = n := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑m)
    hn : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑n)
    ⊢ Eq ↑m ↑n
  -/
  rw [maximalIdeal_eq_span_p] at hm hn
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑m)
    hn : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑n)
    ⊢ Eq ↑m ↑n
  -/
  have := zmod_congr_of_sub_mem_span_aux 1 x m n
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑m)
    hn : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑n)
    this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) 1))) (H …
    ⊢ Eq ↑m ↑n
  -/
  simp only [pow_one] at this
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑m)
    hn : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑n)
    this : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑↑m)  …
    ⊢ Eq ↑m ↑n
  -/
  specialize this hm hn
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑m)
    hn : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑n)
    this : Eq ↑↑m ↑↑n
    ⊢ Eq ↑m ↑n
  -/
  apply_fun ZMod.castHom (show p ∣ p ^ 1 by rw [pow_one]) (ZMod p) at this
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑m)
    hn : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑n)
    this : Eq ((ZMod.castHom ⋯ (ZMod p)) ↑↑m) ((ZMod.castHom ⋯ (ZMod p)) ↑↑n)
    ⊢ Eq ↑m ↑n
  -/
  simp only [map_intCast] at this
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    hm : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑m)
    hn : Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HSub.hSub x ↑n)
    this : Eq ↑↑m ↑↑n
    ⊢ Eq ↑m ↑n
  -/
  simpa only [Int.cast_natCast] using this
  /-
    🎉 no goals
  -/


theorem exists_mem_range : ∃ n : ℕ, n < p ∧ x - n ∈ maximalIdeal ℤ_[p] := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Exists fun n => And (LT.lt n p) (Membership.mem (IsLocalRing.maximalIdeal (P …
  -/
  simp only [maximalIdeal_eq_span_p, Ideal.mem_span_singleton, ← norm_lt_one_iff_dvd]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Exists fun n => And (LT.lt n p) (LT.lt (Norm.norm (HSub.hSub x ↑n)) 1)
  -/
  obtain ⟨r, hr⟩ := rat_dense p (x : ℚ_[p]) zero_lt_one
  have H : ‖(r : ℚ_[p])‖ ≤ 1 := by
    rw [norm_sub_rev] at hr
    calc
      _ = ‖(r : ℚ_[p]) - x + x‖ := by ring_nf
      _ ≤ _ := padicNormE.nonarchimedean _ _
      _ ≤ _ := max_le (le_of_lt hr) x.2

  /-
    case intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    ⊢ Exists fun n => And (LT.lt n p) (LT.lt (Norm.norm (HSub.hSub x ↑n)) 1)
  -/
  obtain ⟨n, hzn, hnp, hn⟩ := exists_mem_range_of_norm_rat_le_one r H
  /-
    case intro.intro.intro.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Int
    hzn : LE.le 0 n
    hnp : LT.lt n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ⟨↑r, H⟩ ↑n)) 1
    ⊢ Exists fun n => And (LT.lt n p) (LT.lt (Norm.norm (HSub.hSub x ↑n)) 1)
  -/
  lift n to ℕ using hzn
  /-
    case intro.intro.intro.intro.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ⟨↑r, H⟩ ↑↑n)) 1
    ⊢ Exists fun n => And (LT.lt n p) (LT.lt (Norm.norm (HSub.hSub x ↑n)) 1)
  -/
  use n
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ⟨↑r, H⟩ ↑↑n)) 1
    ⊢ And (LT.lt n p) (LT.lt (Norm.norm (HSub.hSub x ↑n)) 1)
  -/
  constructor
    /-
      case h.left
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      r : Rat
      hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
      H : LE.le (Norm.norm ↑r) 1
      n : Nat
      hnp : LT.lt ↑n ↑p
      hn : LT.lt (Norm.norm (HSub.hSub ⟨↑r, H⟩ ↑↑n)) 1
      ⊢ LT.lt n p
    -/
  · exact mod_cast hnp
    /-
      🎉 no goals
    -/
  /-
    case h.right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ⟨↑r, H⟩ ↑↑n)) 1
    ⊢ LT.lt (Norm.norm (HSub.hSub x ↑n)) 1
  -/
  simp only [norm_def, coe_sub, Subtype.coe_mk, coe_natCast] at hn ⊢
  /-
    case h.right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ↑r ↑↑↑n)) 1
    ⊢ LT.lt (Norm.norm (HSub.hSub ↑x ↑n)) 1
  -/
  rw [show (x - n : ℚ_[p]) = x - r + (r - n) by ring]
  /-
    case h.right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ↑r ↑↑↑n)) 1
    ⊢ LT.lt (Norm.norm (HAdd.hAdd (HSub.hSub ↑x ↑r) (HSub.hSub ↑r ↑n))) 1
  -/
  apply lt_of_le_of_lt (padicNormE.nonarchimedean _ _)
  /-
    case h.right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ↑r ↑↑↑n)) 1
    ⊢ LT.lt (Max.max (Norm.norm (HSub.hSub ↑x ↑r)) (Norm.norm (HSub.hSub ↑r ↑n))) 1
  -/
  apply max_lt hr
  /-
    case h.right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    r : Rat
    hr : LT.lt (Norm.norm (HSub.hSub ↑x ↑r)) 1
    H : LE.le (Norm.norm ↑r) 1
    n : Nat
    hnp : LT.lt ↑n ↑p
    hn : LT.lt (Norm.norm (HSub.hSub ↑r ↑↑↑n)) 1
    ⊢ LT.lt (Norm.norm (HSub.hSub ↑r ↑n)) 1
  -/
  simpa using hn
  /-
    🎉 no goals
  -/


theorem existsUnique_mem_range : ∃! n : ℕ, n < p ∧ x - n ∈ maximalIdeal ℤ_[p] := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ ExistsUnique fun n => And (LT.lt n p) (Membership.mem (IsLocalRing.maximalId …
  -/
  obtain ⟨n, hn₁, hn₂⟩ := exists_mem_range x
  /-
    case intro.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    hn₁ : LT.lt n p
    hn₂ : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑n)
    ⊢ ExistsUnique fun n => And (LT.lt n p) (Membership.mem (IsLocalRing.maximalId …
  -/
  use n, ⟨hn₁, hn₂⟩, fun m ⟨hm₁, hm₂⟩ ↦ ?_
  /-
    case right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    hn₁ : LT.lt n p
    hn₂ : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑n)
    m : Nat
    x✝ : (fun n => And (LT.lt n p) (Membership.mem (IsLocalRing.maximalIdeal (Padi …
    hm₁ : LT.lt m p
    hm₂ : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑m)
    ⊢ Eq m n
  -/
  have := (zmod_congr_of_sub_mem_max_ideal x n m hn₂ hm₂).symm
  /-
    case right
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    hn₁ : LT.lt n p
    hn₂ : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑n)
    m : Nat
    x✝ : (fun n => And (LT.lt n p) (Membership.mem (IsLocalRing.maximalIdeal (Padi …
    hm₁ : LT.lt m p
    hm₂ : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑m)
    this : Eq ↑m ↑n
    ⊢ Eq m n
  -/
  rwa [ZMod.natCast_eq_natCast_iff, ModEq, mod_eq_of_lt hn₁, mod_eq_of_lt hm₁] at this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-17")] alias exists_unique_mem_range := existsUnique_mem_range


/-- `zmod_repr x` is the unique natural number smaller than `p`
satisfying `‖(x - zmod_repr x : ℤ_[p])‖ < 1`.
-/
def zmodRepr : ℕ :=
  Classical.choose (existsUnique_mem_range x).exists


theorem zmodRepr_spec : zmodRepr x < p ∧ x - zmodRepr x ∈ maximalIdeal ℤ_[p] :=
  Classical.choose_spec (existsUnique_mem_range x).exists


theorem zmodRepr_unique (y : ℕ) (hy₁ : y < p) (hy₂ : x - y ∈ maximalIdeal ℤ_[p]) : y = zmodRepr x :=
  have h := (Classical.choose_spec (existsUnique_mem_range x)).right
  (h y ⟨hy₁, hy₂⟩).trans (h (zmodRepr x) (zmodRepr_spec x)).symm


theorem zmodRepr_lt_p : zmodRepr x < p :=
  (zmodRepr_spec _).1


theorem sub_zmodRepr_mem : x - zmodRepr x ∈ maximalIdeal ℤ_[p] :=
  (zmodRepr_spec _).2


/-- `toZModHom` is an auxiliary constructor for creating ring homs from `ℤ_[p]` to `ZMod v`.
-/
def toZModHom (v : ℕ) (f : ℤ_[p] → ℕ) (f_spec : ∀ x, x - f x ∈ (Ideal.span {↑v} : Ideal ℤ_[p]))
    (f_congr :
      ∀ (x : ℤ_[p]) (a b : ℕ),
        x - a ∈ (Ideal.span {↑v} : Ideal ℤ_[p]) →
          x - b ∈ (Ideal.span {↑v} : Ideal ℤ_[p]) → (a : ZMod v) = b) :
    ℤ_[p] →+* ZMod v where
  toFun x := f x
  map_zero' := by
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      ⊢ Eq ((↑{ toFun := fun x => ↑(f x), map_one' := ⋯, map_mul' := ⋯ }).toFun 0) 0
    -/
    dsimp only
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      ⊢ Eq (↑(f 0)) 0
    -/
    rw [f_congr (0 : ℤ_[p]) _ 0, cast_zero]
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub 0 ↑(f 0))
      -/
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      ⊢ Eq ((fun x => ↑(f x)) 1) 1
    -/
    · exact f_spec _
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      ⊢ Eq (↑(f 1)) 1
    -/
      /-
        🎉 no goals
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub 1 ↑(f 1))
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub 0 ↑0)
      -/
      /-
        🎉 no goals
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub 1 ↑1)
      -/
    · simp only [sub_zero, cast_zero, Submodule.zero_mem]
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
  map_one' := by
    dsimp only
    rw [f_congr (1 : ℤ_[p]) _ 1, cast_one]
    · exact f_spec _
    · simp only [sub_self, cast_one, Submodule.zero_mem]
  map_add' := by
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      ⊢ ∀ (x y : PadicInt p), Eq ((↑{ toFun := fun x => ↑(f x), map_one' := ⋯, map_m …
    -/
    intro x y
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      ⊢ ∀ (x y : PadicInt p), Eq ({ toFun := fun x => ↑(f x), map_one' := ⋯ }.toFun  …
    -/
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x✝ : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      x y : PadicInt p
      ⊢ Eq ((↑{ toFun := fun x => ↑(f x), map_one' := ⋯, map_mul' := ⋯ }).toFun (HAd …
    -/
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x✝ : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      x y : PadicInt p
      ⊢ Eq ({ toFun := fun x => ↑(f x), map_one' := ⋯ }.toFun (HMul.hMul x y)) (HMul …
    -/
    dsimp only
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x✝ : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      x y : PadicInt p
      ⊢ Eq (↑(f (HMul.hMul x y))) (HMul.hMul ↑(f x) ↑(f y))
    -/
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      r : Rat
      x✝ : PadicInt p
      v : Nat
      f : PadicInt p → Nat
      f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
      f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
      x y : PadicInt p
      ⊢ Eq (↑(f (HAdd.hAdd x y))) (HAdd.hAdd ↑(f x) ↑(f y))
    -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub (HMul.hMul x …
      -/
    rw [f_congr (x + y) _ (f x + f y), cast_add]
      /-
        🎉 no goals
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub (HMul.hMul x …
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub (HAdd.hAdd x …
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        I : Ideal (PadicInt p) := Ideal.span (Singleton.singleton ↑v)
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub (HMul.hMul x …
      -/
    · exact f_spec _
      /-
        case h.e'_5
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        I : Ideal (PadicInt p) := Ideal.span (Singleton.singleton ↑v)
        ⊢ Eq (HSub.hSub (HMul.hMul x y) ↑(HMul.hMul (f x) (f y))) (HAdd.hAdd (HMul.hMu …
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        I : Ideal (PadicInt p) := Ideal.span (Singleton.singleton ↑v)
        ⊢ Eq (HSub.hSub (HMul.hMul x y) (HMul.hMul ↑(f x) ↑(f y))) (HAdd.hAdd (HMul.hM …
      -/
      /-
        case a
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑v)) (HSub.hSub (HAdd.hAdd x …
      -/
      /-
        🎉 no goals
      -/
    · convert Ideal.add_mem _ (f_spec x) (f_spec y) using 1
      /-
        case h.e'_5
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        ⊢ Eq (HSub.hSub (HAdd.hAdd x y) ↑(HAdd.hAdd (f x) (f y))) (HAdd.hAdd (HSub.hSu …
      -/
      rw [cast_add]
      /-
        case h.e'_5
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        v : Nat
        f : PadicInt p → Nat
        f_spec : ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑ …
        f_congr : ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleto …
        x y : PadicInt p
        ⊢ Eq (HSub.hSub (HAdd.hAdd x y) (HAdd.hAdd ↑(f x) ↑(f y))) (HAdd.hAdd (HSub.hS …
      -/
      ring
      /-
        🎉 no goals
      -/
  map_mul' := by
    intro x y
    dsimp only
    rw [f_congr (x * y) _ (f x * f y), cast_mul]
    · exact f_spec _
    · let I : Ideal ℤ_[p] := Ideal.span {↑v}
      convert I.add_mem (I.mul_mem_left x (f_spec y)) (I.mul_mem_right ↑(f y) (f_spec x)) using 1
      rw [cast_mul]
      ring


/-- `toZMod` is a ring hom from `ℤ_[p]` to `ZMod p`,
with the equality `toZMod x = (zmodRepr x : ZMod p)`.
-/
def toZMod : ℤ_[p] →+* ZMod p :=
  toZModHom p zmodRepr
    (by
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        ⊢ ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑p)) (HS …
      -/
      rw [← maximalIdeal_eq_span_p]
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        ⊢ ∀ (x : PadicInt p), Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) ( …
      -/
      exact sub_zmodRepr_mem)
      /-
        🎉 no goals
      -/
    (by
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        ⊢ ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleton.single …
      -/
      rw [← maximalIdeal_eq_span_p]
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        ⊢ ∀ (x : PadicInt p) (a b : Nat), Membership.mem (IsLocalRing.maximalIdeal (Pa …
      -/
      exact zmod_congr_of_sub_mem_max_ideal)
      /-
        🎉 no goals
      -/


/-- `z - (toZMod z : ℤ_[p])` is contained in the maximal ideal of `ℤ_[p]`, for every `z : ℤ_[p]`.

The coercion from `ZMod p` to `ℤ_[p]` is `ZMod.cast`,
which coerces `ZMod p` into arbitrary rings.
This is unfortunate, but a consequence of the fact that we allow `ZMod p`
to coerce to rings of arbitrary characteristic, instead of only rings of characteristic `p`.
This coercion is only a ring homomorphism if it coerces into a ring whose characteristic divides
`p`. While this is not the case here we can still make use of the coercion.
-/
theorem toZMod_spec : x - (ZMod.cast (toZMod x) : ℤ_[p]) ∈ maximalIdeal ℤ_[p] := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x (PadicIn …
  -/
  convert sub_zmodRepr_mem x using 2
  /-
    case h.e'_5.h.e'_6
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Eq (PadicInt.toZMod x).cast ↑x.zmodRepr
  -/
  dsimp [toZMod, toZModHom]
  /-
    case h.e'_5.h.e'_6
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Eq (↑x.zmodRepr).cast ↑x.zmodRepr
  -/
  rcases Nat.exists_eq_add_of_lt hp_prime.1.pos with ⟨p', rfl⟩
  /-
    case h.e'_5.h.e'_6.intro
    p' : Nat
    hp_prime : Fact (Nat.Prime (HAdd.hAdd (HAdd.hAdd 0 p') 1))
    x : PadicInt (HAdd.hAdd (HAdd.hAdd 0 p') 1)
    ⊢ Eq (↑x.zmodRepr).cast ↑x.zmodRepr
  -/
  change ↑((_ : ZMod (0 + p' + 1)).val) = (_ : ℤ_[0 + p' + 1])
  /-
    case h.e'_5.h.e'_6.intro
    p' : Nat
    hp_prime : Fact (Nat.Prime (HAdd.hAdd (HAdd.hAdd 0 p') 1))
    x : PadicInt (HAdd.hAdd (HAdd.hAdd 0 p') 1)
    ⊢ Eq ↑(↑x.zmodRepr).val ↑x.zmodRepr
  -/
  rw [Nat.cast_inj]
  /-
    case h.e'_5.h.e'_6.intro
    p' : Nat
    hp_prime : Fact (Nat.Prime (HAdd.hAdd (HAdd.hAdd 0 p') 1))
    x : PadicInt (HAdd.hAdd (HAdd.hAdd 0 p') 1)
    ⊢ Eq (↑x.zmodRepr).val x.zmodRepr
  -/
  apply mod_eq_of_lt
  /-
    case h.e'_5.h.e'_6.intro.h
    p' : Nat
    hp_prime : Fact (Nat.Prime (HAdd.hAdd (HAdd.hAdd 0 p') 1))
    x : PadicInt (HAdd.hAdd (HAdd.hAdd 0 p') 1)
    ⊢ LT.lt x.zmodRepr (HAdd.hAdd 0 p').succ
  -/
  simpa only [zero_add] using zmodRepr_lt_p x
  /-
    🎉 no goals
  -/


theorem ker_toZMod : RingHom.ker (toZMod : ℤ_[p] →+* ZMod p) = maximalIdeal ℤ_[p] := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    ⊢ Eq (RingHom.ker PadicInt.toZMod) (IsLocalRing.maximalIdeal (PadicInt p))
  -/
  ext x
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Iff (Membership.mem (RingHom.ker PadicInt.toZMod) x) (Membership.mem (IsLoca …
  -/
  rw [RingHom.mem_ker]
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Iff (Eq (PadicInt.toZMod x) 0) (Membership.mem (IsLocalRing.maximalIdeal (Pa …
  -/
  constructor
    /-
      case h.mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      ⊢ Eq (PadicInt.toZMod x) 0 → Membership.mem (IsLocalRing.maximalIdeal (PadicIn …
    -/
  · intro h
    /-
      case h.mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      h : Eq (PadicInt.toZMod x) 0
      ⊢ Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) x
    -/
    simpa only [h, ZMod.cast_zero, sub_zero] using toZMod_spec x
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      ⊢ Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) x → Eq (PadicInt.toZM …
    -/
  · intro h
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      h : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) x
      ⊢ Eq (PadicInt.toZMod x) 0
    -/
    rw [← sub_zero x] at h
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      h : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x 0)
      ⊢ Eq (PadicInt.toZMod x) 0
    -/
    dsimp [toZMod, toZModHom]
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      h : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x 0)
      ⊢ Eq (↑x.zmodRepr) 0
    -/
    convert zmod_congr_of_sub_mem_max_ideal x _ 0 _ h
      /-
        case h.e'_3
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        x : PadicInt p
        h : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x 0)
        ⊢ Eq 0 ↑0
      -/
    · norm_cast
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.convert_2
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        x : PadicInt p
        h : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x 0)
        ⊢ Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) (HSub.hSub x ↑x.zmodR …
      -/
    · apply sub_zmodRepr_mem
      /-
        🎉 no goals
      -/


/-- The equivalence between the residue field of the `p`-adic integers and `ℤ/pℤ` -/
def residueField : IsLocalRing.ResidueField ℤ_[p] ≃+* ZMod p := by
  exact_mod_cast (@PadicInt.ker_toZMod p _) ▸ RingHom.quotientKerEquivOfSurjective
    (ZMod.ringHom_surjective PadicInt.toZMod)


/-- `appr n x` gives a value `v : ℕ` such that `x` and `↑v : ℤ_p` are congruent mod `p^n`.
See `appr_spec`. -/
-- Porting note: removing irreducible solves a lot of problems
noncomputable def appr : ℤ_[p] → ℕ → ℕ
  | _x, 0 => 0
  | x, n + 1 =>
    let y := x - appr x n
    if hy : y = 0 then appr x n
    else
      let u := (unitCoeff hy : ℤ_[p])
      appr x n + p ^ n * (toZMod ((u * (p : ℤ_[p]) ^ (y.valuation - n : ℤ).natAbs) : ℤ_[p])).val


theorem appr_lt (x : ℤ_[p]) (n : ℕ) : x.appr n < p ^ n := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    ⊢ LT.lt (x.appr n) (HPow.hPow p n)
  -/
  induction' n with n ih generalizing x
    /-
      case zero
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      ⊢ LT.lt (x.appr 0) (HPow.hPow p 0)
    -/
  · simp only [appr, zero_eq, _root_.pow_zero, zero_lt_one]
    /-
      🎉 no goals
    -/
  /-
    case succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
    x : PadicInt p
    ⊢ LT.lt (x.appr (HAdd.hAdd n 1)) (HPow.hPow p (HAdd.hAdd n 1))
  -/
  simp only [appr, map_natCast, ZMod.natCast_self, RingHom.map_pow, Int.natAbs, RingHom.map_mul]
  /-
    case succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
    x : PadicInt p
    ⊢ LT.lt (dite (Eq (HSub.hSub x ↑(x.appr n)) 0) (fun h => x.appr n) fun h => HA …
  -/
  have hp : p ^ n < p ^ (n + 1) := by apply Nat.pow_lt_pow_right hp_prime.1.one_lt n.lt_add_one
  /-
    case succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
    x : PadicInt p
    hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
    ⊢ LT.lt (dite (Eq (HSub.hSub x ↑(x.appr n)) 0) (fun h => x.appr n) fun h => HA …
  -/
  split_ifs with h
    /-
      case pos
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
      x : PadicInt p
      hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
      h : Eq (HSub.hSub x ↑(x.appr n)) 0
      ⊢ LT.lt (x.appr n) (HPow.hPow p (HAdd.hAdd n 1))
    -/
  · apply lt_trans (ih _) hp
    /-
      🎉 no goals
    -/
  · calc
      _ < p ^ n + p ^ n * (p - 1) := ?_
      _ = p ^ (n + 1) := ?_

      /-
        case neg.calc_1
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        n : Nat
        ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
        x : PadicInt p
        hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
        h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
        ⊢ LT.lt (HAdd.hAdd (x.appr n) (HMul.hMul (HPow.hPow p n) (HMul.hMul (PadicInt. …
      -/
    · apply add_lt_add_of_lt_of_le (ih _)
      /-
        case neg.calc_1
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        n : Nat
        ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
        x : PadicInt p
        hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
        h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
        ⊢ LE.le (HMul.hMul (HPow.hPow p n) (HMul.hMul (PadicInt.toZMod ↑(PadicInt.unit …
      -/
      apply Nat.mul_le_mul_left
      /-
        case neg.calc_1.h
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        n : Nat
        ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
        x : PadicInt p
        hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
        h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
        ⊢ LE.le (HMul.hMul (PadicInt.toZMod ↑(PadicInt.unitCoeff ⋯)) (HPow.hPow 0 (Int …
      -/
      apply le_pred_of_lt
      /-
        case neg.calc_1.h.h
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        n : Nat
        ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
        x : PadicInt p
        hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
        h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
        ⊢ LT.lt (HMul.hMul (PadicInt.toZMod ↑(PadicInt.unitCoeff ⋯)) (HPow.hPow 0 (Int …
      -/
      apply ZMod.val_lt
      /-
        🎉 no goals
      -/
      /-
        case neg.calc_2
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        n : Nat
        ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
        x : PadicInt p
        hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
        h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
        ⊢ Eq (HAdd.hAdd (HPow.hPow p n) (HMul.hMul (HPow.hPow p n) (HSub.hSub p 1))) ( …
      -/
    · rw [mul_tsub, mul_one, ← _root_.pow_succ]
      /-
        case neg.calc_2
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        n : Nat
        ih : ∀ (x : PadicInt p), LT.lt (x.appr n) (HPow.hPow p n)
        x : PadicInt p
        hp : LT.lt (HPow.hPow p n) (HPow.hPow p (HAdd.hAdd n 1))
        h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
        ⊢ Eq (HAdd.hAdd (HPow.hPow p n) (HSub.hSub (HPow.hPow p (HAdd.hAdd n 1)) (HPow …
      -/
      apply add_tsub_cancel_of_le (le_of_lt hp)
      /-
        🎉 no goals
      -/


theorem appr_mono (x : ℤ_[p]) : Monotone x.appr := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Monotone x.appr
  -/
  apply monotone_nat_of_le_succ
  /-
    case hf
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ ∀ (n : Nat), LE.le (x.appr n) (x.appr (HAdd.hAdd n 1))
  -/
  intro n
  /-
    case hf
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    ⊢ LE.le (x.appr n) (x.appr (HAdd.hAdd n 1))
  -/
  dsimp [appr]
  /-
    case hf
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    ⊢ LE.le (x.appr n) (dite (Eq (HSub.hSub x ↑(x.appr n)) 0) (fun hy => x.appr n) …
  -/
  split_ifs; · rfl
               /-
                 🎉 no goals
               -/
  /-
    case neg
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    h✝ : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    ⊢ LE.le (x.appr n) (HAdd.hAdd (x.appr n) (HMul.hMul (HPow.hPow p n) (PadicInt. …
  -/
  apply Nat.le_add_right
  /-
    🎉 no goals
  -/


theorem dvd_appr_sub_appr (x : ℤ_[p]) (m n : ℕ) (h : m ≤ n) : p ^ m ∣ x.appr n - x.appr m := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m n : Nat
    h : LE.le m n
    ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr n) (x.appr m))
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h; clear h
  /-
    case intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
  -/
  induction' k with k ih
    /-
      case intro.zero
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      m : Nat
      ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m 0)) (x.appr m))
    -/
  · simp only [zero_eq, add_zero, le_refl, tsub_eq_zero_of_le, ne_eq, Nat.isUnit_iff, dvd_zero]
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m (HAdd.hAdd k 1))) (x …
  -/
  rw [← add_assoc]
  /-
    case intro.succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd (HAdd.hAdd m k) 1)) (x …
  -/
  dsimp [appr]
  /-
    case intro.succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (dite (Eq (HSub.hSub x ↑(x.appr (HAdd.hAd …
  -/
  split_ifs with h
    /-
      case pos
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      m k : Nat
      ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
      h : Eq (HSub.hSub x ↑(x.appr (HAdd.hAdd m k))) 0
      ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    -/
  · exact ih
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    h : Not (Eq (HSub.hSub x ↑(x.appr (HAdd.hAdd m k))) 0)
    ⊢ Dvd.dvd (HPow.hPow p m) (HSub.hSub (HAdd.hAdd (x.appr (HAdd.hAdd m k)) (HMul …
  -/
  rw [add_comm, add_tsub_assoc_of_le (appr_mono _ (Nat.le_add_right m k))]
  /-
    case neg
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    h : Not (Eq (HSub.hSub x ↑(x.appr (HAdd.hAdd m k))) 0)
    ⊢ Dvd.dvd (HPow.hPow p m) (HAdd.hAdd (HMul.hMul (HPow.hPow p (HAdd.hAdd m k))  …
  -/
  apply dvd_add _ ih
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    h : Not (Eq (HSub.hSub x ↑(x.appr (HAdd.hAdd m k))) 0)
    ⊢ Dvd.dvd (HPow.hPow p m) (HMul.hMul (HPow.hPow p (HAdd.hAdd m k)) (PadicInt.t …
  -/
  apply dvd_mul_of_dvd_left
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    m k : Nat
    ih : Dvd.dvd (HPow.hPow p m) (HSub.hSub (x.appr (HAdd.hAdd m k)) (x.appr m))
    h : Not (Eq (HSub.hSub x ↑(x.appr (HAdd.hAdd m k))) 0)
    ⊢ Dvd.dvd (HPow.hPow p m) (HPow.hPow p (HAdd.hAdd m k))
  -/
  apply pow_dvd_pow _ (Nat.le_add_right m k)
  /-
    🎉 no goals
  -/


theorem appr_spec (n : ℕ) : ∀ x : ℤ_[p], x - appr x n ∈ Ideal.span {(p : ℤ_[p]) ^ n} := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ⊢ ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton (HPow.hP …
  -/
  simp only [Ideal.mem_span_singleton]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ⊢ ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
  -/
  induction' n with n ih
    /-
      case zero
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      ⊢ ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) 0) (HSub.hSub x ↑(x.appr 0))
    -/
  · simp only [zero_eq, _root_.pow_zero, isUnit_one, IsUnit.dvd, forall_const]
    /-
      🎉 no goals
    -/
  /-
    case succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    ⊢ ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub x ↑( …
  -/
  intro x
  /-
    case succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub x ↑(x.appr (HAdd.hAdd n  …
  -/
  dsimp only [appr]
  /-
    case succ
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub x ↑(dite (Eq (HSub.hSub  …
  -/
  split_ifs with h
    /-
      case pos
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Eq (HSub.hSub x ↑(x.appr n)) 0
      ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub x ↑(x.appr n))
    -/
  · rw [h]
    /-
      case pos
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Eq (HSub.hSub x ↑(x.appr n)) 0
      ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) 0
    -/
    apply dvd_zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub x ↑(HAdd.hAdd (x.appr n) …
  -/
  push_cast
  /-
    case neg
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub x (HAdd.hAdd (↑(x.appr n …
  -/
  rw [sub_add_eq_sub_sub]
  /-
    case neg
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub (HSub.hSub x ↑(x.appr n) …
  -/
  obtain ⟨c, hc⟩ := ih x
  /-
    case neg.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    c : PadicInt p
    hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub (HSub.hSub x ↑(x.appr n) …
  -/
  simp only [map_natCast, ZMod.natCast_self, RingHom.map_pow, RingHom.map_mul, ZMod.natCast_val]
  have hc' : c ≠ 0 := by
    rintro rfl
    simp only [mul_zero] at hc
    contradiction
  conv_rhs =>
    congr
    simp only [hc]
  /-
    case neg.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    c : PadicInt p
    hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
    hc' : Ne c 0
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub (HMul.hMul (HPow.hPow (↑ …
  -/
  rw [show (x - (appr x n : ℤ_[p])).valuation = ((p : ℤ_[p]) ^ n * c).valuation by rw [hc]]
  /-
    case neg.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    c : PadicInt p
    hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
    hc' : Ne c 0
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd n 1)) (HSub.hSub (HMul.hMul (HPow.hPow (↑ …
  -/
  rw [valuation_p_pow_mul _ _ hc', Nat.cast_add, add_sub_cancel_left, _root_.pow_succ, ← mul_sub]
  /-
    case neg.intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    c : PadicInt p
    hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
    hc' : Ne c 0
    ⊢ Dvd.dvd (HMul.hMul (HPow.hPow (↑p) n) ↑p) (HMul.hMul (HPow.hPow (↑p) n) (HSu …
  -/
  apply mul_dvd_mul_left
  /-
    case neg.intro.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
    x : PadicInt p
    h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
    c : PadicInt p
    hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
    hc' : Ne c 0
    ⊢ Dvd.dvd (↑p) (HSub.hSub c (HMul.hMul (PadicInt.toZMod ↑(PadicInt.unitCoeff h …
  -/
  obtain hc0 | hc0 := eq_or_ne c.valuation 0
    /-
      case neg.intro.h.inl
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : PadicInt p
      hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
      hc' : Ne c 0
      hc0 : Eq c.valuation 0
      ⊢ Dvd.dvd (↑p) (HSub.hSub c (HMul.hMul (PadicInt.toZMod ↑(PadicInt.unitCoeff h …
    -/
  · simp only [hc0, mul_one, _root_.pow_zero, Nat.cast_zero, Int.natAbs_zero]
    /-
      case neg.intro.h.inl
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : PadicInt p
      hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
      hc' : Ne c 0
      hc0 : Eq c.valuation 0
      ⊢ Dvd.dvd (↑p) (HSub.hSub c (PadicInt.toZMod ↑(PadicInt.unitCoeff h)).cast)
    -/
    rw [mul_comm, unitCoeff_spec h] at hc
    suffices c = unitCoeff h by
      rw [← this, ← Ideal.mem_span_singleton, ← maximalIdeal_eq_span_p]
      apply toZMod_spec
    /-
      case neg.intro.h.inl
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : PadicInt p
      hc : Eq (HMul.hMul (↑(PadicInt.unitCoeff h)) (HPow.hPow (↑p) (HSub.hSub x ↑(x. …
      hc' : Ne c 0
      hc0 : Eq c.valuation 0
      ⊢ Eq c ↑(PadicInt.unitCoeff h)
    -/
    lift c to ℤ_[p]ˣ using by simp [isUnit_iff, norm_eq_zpow_neg_valuation hc', hc0]
    /-
      case neg.intro.h.inl.intro
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : Units (PadicInt p)
      hc : Eq (HMul.hMul (↑(PadicInt.unitCoeff h)) (HPow.hPow (↑p) (HSub.hSub x ↑(x. …
      hc' : Ne (↑c) 0
      hc0 : Eq (↑c).valuation 0
      ⊢ Eq ↑c ↑(PadicInt.unitCoeff h)
    -/
    rw [IsDiscreteValuationRing.unit_mul_pow_congr_unit _ _ _ _ _ hc]
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : Units (PadicInt p)
      hc : Eq (HMul.hMul (↑(PadicInt.unitCoeff h)) (HPow.hPow (↑p) (HSub.hSub x ↑(x. …
      hc' : Ne (↑c) 0
      hc0 : Eq (↑c).valuation 0
      ⊢ Irreducible ↑p
    -/
    exact irreducible_p
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.h.inr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : PadicInt p
      hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
      hc' : Ne c 0
      hc0 : Ne c.valuation 0
      ⊢ Dvd.dvd (↑p) (HSub.hSub c (HMul.hMul (PadicInt.toZMod ↑(PadicInt.unitCoeff h …
    -/
  · simp only [Int.natAbs_ofNat, zero_pow hc0, sub_zero, ZMod.cast_zero, mul_zero]
    /-
      case neg.intro.h.inr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : PadicInt p
      hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
      hc' : Ne c 0
      hc0 : Ne c.valuation 0
      ⊢ Dvd.dvd (↑p) c
    -/
    rw [unitCoeff_spec hc']
    /-
      case neg.intro.h.inr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      ih : ∀ (x : PadicInt p), Dvd.dvd (HPow.hPow (↑p) n) (HSub.hSub x ↑(x.appr n))
      x : PadicInt p
      h : Not (Eq (HSub.hSub x ↑(x.appr n)) 0)
      c : PadicInt p
      hc : Eq (HSub.hSub x ↑(x.appr n)) (HMul.hMul (HPow.hPow (↑p) n) c)
      hc' : Ne c 0
      hc0 : Ne c.valuation 0
      ⊢ Dvd.dvd (↑p) (HMul.hMul (↑(PadicInt.unitCoeff hc')) (HPow.hPow (↑p) c.valuat …
    -/
    exact (dvd_pow_self (p : ℤ_[p]) hc0).mul_left _
    /-
      🎉 no goals
    -/


/-- A ring hom from `ℤ_[p]` to `ZMod (p^n)`, with underlying function `PadicInt.appr n`. -/
def toZModPow (n : ℕ) : ℤ_[p] →+* ZMod (p ^ n) :=
  toZModHom (p ^ n) (fun x => appr x n)
    (by
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        n : Nat
        ⊢ ∀ (x : PadicInt p), Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.h …
      -/
      intros
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        n : Nat
        x✝ : PadicInt p
        ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub.hSu …
      -/
      rw [Nat.cast_pow]
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        n : Nat
        x✝ : PadicInt p
        ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
      -/
      exact appr_spec n _)
      /-
        🎉 no goals
      -/
    (by
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x : PadicInt p
        n : Nat
        ⊢ ∀ (x : PadicInt p) (a b : Nat), Membership.mem (Ideal.span (Singleton.single …
      -/
      intro x a b ha hb
      /-
        p : Nat
        hp_prime : Fact (Nat.Prime p)
        r : Rat
        x✝ : PadicInt p
        n : Nat
        x : PadicInt p
        a b : Nat
        ha : Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub. …
        hb : Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub. …
        ⊢ Eq ↑a ↑b
      -/
      apply zmod_congr_of_sub_mem_span n x a b
        /-
          case ha
          p : Nat
          hp_prime : Fact (Nat.Prime p)
          r : Rat
          x✝ : PadicInt p
          n : Nat
          x : PadicInt p
          a b : Nat
          ha : Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub. …
          hb : Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub. …
          ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
        -/
      · simpa using ha
        /-
          🎉 no goals
        -/
        /-
          case hb
          p : Nat
          hp_prime : Fact (Nat.Prime p)
          r : Rat
          x✝ : PadicInt p
          n : Nat
          x : PadicInt p
          a b : Nat
          ha : Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub. …
          hb : Membership.mem (Ideal.span (Singleton.singleton ↑(HPow.hPow p n))) (HSub. …
          ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
        -/
      · simpa using hb)
        /-
          🎉 no goals
        -/


theorem ker_toZModPow (n : ℕ) :
    RingHom.ker (toZModPow n : ℤ_[p] →+* ZMod (p ^ n)) = Ideal.span {(p : ℤ_[p]) ^ n} := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (RingHom.ker (PadicInt.toZModPow n)) (Ideal.span (Singleton.singleton (HP …
  -/
  ext x
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    ⊢ Iff (Membership.mem (RingHom.ker (PadicInt.toZModPow n)) x) (Membership.mem  …
  -/
  rw [RingHom.mem_ker]
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    n : Nat
    x : PadicInt p
    ⊢ Iff (Eq ((PadicInt.toZModPow n) x) 0) (Membership.mem (Ideal.span (Singleton …
  -/
  constructor
    /-
      case h.mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      ⊢ Eq ((PadicInt.toZModPow n) x) 0 → Membership.mem (Ideal.span (Singleton.sing …
    -/
  · intro h
    suffices x.appr n = 0 by
      convert appr_spec n x
      simp only [this, sub_zero, cast_zero]
    /-
      case h.mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Eq ((PadicInt.toZModPow n) x) 0
      ⊢ Eq (x.appr n) 0
    -/
    dsimp [toZModPow, toZModHom] at h
    /-
      case h.mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Eq (↑(x.appr n)) 0
      ⊢ Eq (x.appr n) 0
    -/
    rw [ZMod.natCast_zmod_eq_zero_iff_dvd] at h
    /-
      case h.mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Dvd.dvd (HPow.hPow p n) (x.appr n)
      ⊢ Eq (x.appr n) 0
    -/
    apply eq_zero_of_dvd_of_lt h (appr_lt _ _)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) x → Eq  …
    -/
  · intro h
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) x
      ⊢ Eq ((PadicInt.toZModPow n) x) 0
    -/
    rw [← sub_zero x] at h
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub …
      ⊢ Eq ((PadicInt.toZModPow n) x) 0
    -/
    dsimp [toZModPow, toZModHom]
    /-
      case h.mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub …
      ⊢ Eq (↑(x.appr n)) 0
    -/
    rw [zmod_congr_of_sub_mem_span n x _ 0 _ h, cast_zero]
    /-
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      n : Nat
      x : PadicInt p
      h : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub …
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
    -/
    apply appr_spec
    /-
      🎉 no goals
    -/

-- This is not a simp lemma; simp can't match the LHS.

theorem zmod_cast_comp_toZModPow (m n : ℕ) (h : m ≤ n) :
    (ZMod.castHom (pow_dvd_pow p h) (ZMod (p ^ m))).comp (@toZModPow p _ n) = @toZModPow p _ m := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    ⊢ Eq ((ZMod.castHom ⋯ (ZMod (HPow.hPow p m))).comp (PadicInt.toZModPow n)) (Pa …
  -/
  apply ZMod.ringHom_eq_of_ker_eq
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    ⊢ Eq (RingHom.ker ((ZMod.castHom ⋯ (ZMod (HPow.hPow p m))).comp (PadicInt.toZM …
  -/
  ext x
  /-
    case h.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    x : PadicInt p
    ⊢ Iff (Membership.mem (RingHom.ker ((ZMod.castHom ⋯ (ZMod (HPow.hPow p m))).co …
  -/
  rw [RingHom.mem_ker, RingHom.mem_ker]
  /-
    case h.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    x : PadicInt p
    ⊢ Iff (Eq (((ZMod.castHom ⋯ (ZMod (HPow.hPow p m))).comp (PadicInt.toZModPow n …
  -/
  simp only [Function.comp_apply, ZMod.castHom_apply, RingHom.coe_comp]
  /-
    case h.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    x : PadicInt p
    ⊢ Iff (Eq ((PadicInt.toZModPow n) x).cast 0) (Eq ((PadicInt.toZModPow m) x) 0)
  -/
  simp only [toZModPow, toZModHom, RingHom.coe_mk]
  /-
    case h.h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    x : PadicInt p
    ⊢ Iff (Eq ({ toFun := fun x => ↑(x.appr n), map_one' := ⋯, map_mul' := ⋯ } x). …
  -/
  dsimp
  rw [ZMod.cast_natCast (pow_dvd_pow p h),
    zmod_congr_of_sub_mem_span m (x.appr n) (x.appr n) (x.appr m)]
    /-
      case h.h.ha
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      m n : Nat
      h : LE.le m n
      x : PadicInt p
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) m))) (HSub.h …
    -/
  · rw [sub_self]
    /-
      case h.h.ha
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      m n : Nat
      h : LE.le m n
      x : PadicInt p
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) m))) 0
    -/
    apply Ideal.zero_mem _
    /-
      🎉 no goals
    -/
    /-
      case h.h.hb
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      m n : Nat
      h : LE.le m n
      x : PadicInt p
      ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) m))) (HSub.h …
    -/
  · rw [Ideal.mem_span_singleton]
    /-
      case h.h.hb
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      m n : Nat
      h : LE.le m n
      x : PadicInt p
      ⊢ Dvd.dvd (HPow.hPow (↑p) m) (HSub.hSub ↑(x.appr n) ↑(x.appr m))
    -/
    rcases dvd_appr_sub_appr x m n h with ⟨c, hc⟩
    /-
      case h.h.hb.intro
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      m n : Nat
      h : LE.le m n
      x : PadicInt p
      c : Nat
      hc : Eq (HSub.hSub (x.appr n) (x.appr m)) (HMul.hMul (HPow.hPow p m) c)
      ⊢ Dvd.dvd (HPow.hPow (↑p) m) (HSub.hSub ↑(x.appr n) ↑(x.appr m))
    -/
    use c
    /-
      case h
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      m n : Nat
      h : LE.le m n
      x : PadicInt p
      c : Nat
      hc : Eq (HSub.hSub (x.appr n) (x.appr m)) (HMul.hMul (HPow.hPow p m) c)
      ⊢ Eq (HSub.hSub ↑(x.appr n) ↑(x.appr m)) (HMul.hMul (HPow.hPow (↑p) m) ↑c)
    -/
    rw [← Nat.cast_sub (appr_mono _ h), hc, Nat.cast_mul, Nat.cast_pow]
    /-
      🎉 no goals
    -/


@[simp]
theorem cast_toZModPow (m n : ℕ) (h : m ≤ n) (x : ℤ_[p]) :
    ZMod.cast (toZModPow n x) = toZModPow m x := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    x : PadicInt p
    ⊢ Eq ((PadicInt.toZModPow n) x).cast ((PadicInt.toZModPow m) x)
  -/
  rw [← zmod_cast_comp_toZModPow _ _ h]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    m n : Nat
    h : LE.le m n
    x : PadicInt p
    ⊢ Eq ((PadicInt.toZModPow n) x).cast (((ZMod.castHom ⋯ (ZMod (HPow.hPow p m))) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem denseRange_natCast : DenseRange (Nat.cast : ℕ → ℤ_[p]) := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    ⊢ DenseRange Nat.cast
  -/
  intro x
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Membership.mem (closure (Set.range Nat.cast)) x
  -/
  rw [Metric.mem_closure_range_iff]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun k => LT.lt (Dist.dist x ↑k) ε
  -/
  intro ε hε
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun k => LT.lt (Dist.dist x ↑k) ε
  -/
  obtain ⟨n, hn⟩ := exists_pow_neg_lt p hε
  /-
    case intro
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ε : Real
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ Exists fun k => LT.lt (Dist.dist x ↑k) ε
  -/
  use x.appr n
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ε : Real
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ LT.lt (Dist.dist x ↑(x.appr n)) ε
  -/
  rw [dist_eq_norm]
  /-
    case h
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ε : Real
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ LT.lt (Norm.norm (HSub.hSub x ↑(x.appr n))) ε
  -/
  apply lt_of_le_of_lt _ hn
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ε : Real
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ LE.le (Norm.norm (HSub.hSub x ↑(x.appr n))) (HPow.hPow (↑p) (Neg.neg ↑n))
  -/
  rw [norm_le_pow_iff_mem_span_pow]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ε : Real
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  apply appr_spec
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias denseRange_nat_cast := denseRange_natCast


theorem denseRange_intCast : DenseRange (Int.cast : ℤ → ℤ_[p]) := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    ⊢ DenseRange Int.cast
  -/
  intro x
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Membership.mem (closure (Set.range Int.cast)) x
  -/
  refine DenseRange.induction_on denseRange_natCast x ?_ ?_
    /-
      case refine_1
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      ⊢ IsClosed (setOf fun b => Membership.mem (closure (Set.range Int.cast)) b)
    -/
  · exact isClosed_closure
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      ⊢ ∀ (a : Nat), Membership.mem (closure (Set.range Int.cast)) ↑a
    -/
  · intro a
    /-
      case refine_2
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      a : Nat
      ⊢ Membership.mem (closure (Set.range Int.cast)) ↑a
    -/
    apply subset_closure
    /-
      case refine_2.a
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      a : Nat
      ⊢ Membership.mem (Set.range Int.cast) ↑a
    -/
    exact Set.mem_range_self _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias denseRange_int_cast := denseRange_intCast


/-- Given a family of ring homs `f : Π n : ℕ, R →+* ZMod (p ^ n)`,
`nthHom f r` is an integer-valued sequence
whose `n`th value is the unique integer `k` such that `0 ≤ k < p ^ n`
and `f n r = (k : ZMod (p ^ n))`.
-/
def nthHom (r : R) : ℕ → ℤ := fun n => (f n r : ZMod (p ^ n)).val


@[simp]
theorem nthHom_zero : nthHom f 0 = 0 := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    ⊢ Eq (PadicInt.nthHom f 0) 0
  -/
  simp (config := { unfoldPartialApp := true }) [nthHom]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    ⊢ Eq (fun n => 0) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem pow_dvd_nthHom_sub (r : R) (i j : ℕ) (h : i ≤ j) :
    (p : ℤ) ^ i ∣ nthHom f r j - nthHom f r i := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    i j : Nat
    h : LE.le i j
    ⊢ Dvd.dvd (HPow.hPow (↑p) i) (HSub.hSub (PadicInt.nthHom f r j) (PadicInt.nthH …
  -/
  specialize f_compat i j h
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    r : R
    i j : Nat
    h : LE.le i j
    f_compat : Eq ((ZMod.castHom ⋯ (ZMod (HPow.hPow p i))).comp (f j)) (f i)
    ⊢ Dvd.dvd (HPow.hPow (↑p) i) (HSub.hSub (PadicInt.nthHom f r j) (PadicInt.nthH …
  -/
  rw [← Int.natCast_pow, ← ZMod.intCast_zmod_eq_zero_iff_dvd, Int.cast_sub]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    r : R
    i j : Nat
    h : LE.le i j
    f_compat : Eq ((ZMod.castHom ⋯ (ZMod (HPow.hPow p i))).comp (f j)) (f i)
    ⊢ Eq (HSub.hSub ↑(PadicInt.nthHom f r j) ↑(PadicInt.nthHom f r i)) 0
  -/
  dsimp [nthHom]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    r : R
    i j : Nat
    h : LE.le i j
    f_compat : Eq ((ZMod.castHom ⋯ (ZMod (HPow.hPow p i))).comp (f j)) (f i)
    ⊢ Eq (HSub.hSub ↑↑((f j) r).val ↑↑((f i) r).val) 0
  -/
  rw [← f_compat, RingHom.comp_apply]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    r : R
    i j : Nat
    h : LE.le i j
    f_compat : Eq ((ZMod.castHom ⋯ (ZMod (HPow.hPow p i))).comp (f j)) (f i)
    ⊢ Eq (HSub.hSub ↑↑((f j) r).val ↑↑((ZMod.castHom ⋯ (ZMod (HPow.hPow p i))) ((f …
  -/
  simp only [ZMod.cast_id, ZMod.castHom_apply, sub_self, ZMod.natCast_val, ZMod.intCast_cast]
  /-
    🎉 no goals
  -/


theorem isCauSeq_nthHom (r : R) : IsCauSeq (padicNorm p) fun n => nthHom f r n := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ⊢ IsCauSeq (padicNorm p) fun n => ↑(PadicInt.nthHom f r n)
  -/
  intro ε hε
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (HSub.hSub ((fun …
  -/
  obtain ⟨k, hk⟩ : ∃ k : ℕ, (p : ℚ) ^ (-((k : ℕ) : ℤ)) < ε := exists_pow_neg_lt_rat p hε
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (HSub.hSub ((fun …
  -/
  use k
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    ⊢ ∀ (j : Nat), GE.ge j k → LT.lt (padicNorm p (HSub.hSub ((fun n => ↑(PadicInt …
  -/
  intro j hj
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    j : Nat
    hj : GE.ge j k
    ⊢ LT.lt (padicNorm p (HSub.hSub ((fun n => ↑(PadicInt.nthHom f r n)) j) ((fun  …
  -/
  refine lt_of_le_of_lt ?_ hk
  -- Need to do beta reduction first, as `norm_cast` doesn't.
  -- Added to adapt to https://github.com/leanprover/lean4/pull/2734.
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    j : Nat
    hj : GE.ge j k
    ⊢ LE.le (padicNorm p (HSub.hSub ((fun n => ↑(PadicInt.nthHom f r n)) j) ((fun  …
  -/
  beta_reduce
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    j : Nat
    hj : GE.ge j k
    ⊢ LE.le (padicNorm p (HSub.hSub ↑(PadicInt.nthHom f r j) ↑(PadicInt.nthHom f r …
  -/
  norm_cast
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    j : Nat
    hj : GE.ge j k
    ⊢ LE.le (padicNorm p ↑(HSub.hSub (PadicInt.nthHom f r j) (PadicInt.nthHom f r  …
  -/
  rw [← padicNorm.dvd_iff_norm_le]
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Rat
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    j : Nat
    hj : GE.ge j k
    ⊢ Dvd.dvd (↑(HPow.hPow p k)) (HSub.hSub (PadicInt.nthHom f r j) (PadicInt.nthH …
  -/
  exact mod_cast pow_dvd_nthHom_sub f_compat r k j hj
  /-
    🎉 no goals
  -/


/-- `nthHomSeq f_compat r` bundles `PadicInt.nthHom f r`
as a Cauchy sequence of rationals with respect to the `p`-adic norm.
The `n`th value of the sequence is `((f n r).val : ℚ)`.
-/
def nthHomSeq (r : R) : PadicSeq p :=
  ⟨fun n => nthHom f r n, isCauSeq_nthHom f_compat r⟩

-- this lemma ran into issues after changing to `NeZero` and I'm not sure why.

theorem nthHomSeq_one : nthHomSeq f_compat 1 ≈ 1 := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ⊢ HasEquiv.Equiv (PadicInt.nthHomSeq f_compat 1) 1
  -/
  intro ε hε
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub (Pa …
  -/
  change _ < _ at hε
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub (Pa …
  -/
  use 1
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ ∀ (j : Nat), GE.ge j 1 → LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq …
  -/
  intro j hj
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ε : Rat
    hε : LT.lt 0 ε
    j : Nat
    hj : GE.ge j 1
    ⊢ LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq f_compat 1) 1) j)) ε
  -/
  haveI : Fact (1 < p ^ j) := ⟨Nat.one_lt_pow (by omega) hp_prime.1.one_lt⟩
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ε : Rat
    hε : LT.lt 0 ε
    j : Nat
    hj : GE.ge j 1
    this : Fact (LT.lt 1 (HPow.hPow p j))
    ⊢ LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq f_compat 1) 1) j)) ε
  -/
  suffices (ZMod.cast (1 : ZMod (p ^ j)) : ℚ) = 1 by simp [nthHomSeq, nthHom, this, hε]
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    ε : Rat
    hε : LT.lt 0 ε
    j : Nat
    hj : GE.ge j 1
    this : Fact (LT.lt 1 (HPow.hPow p j))
    ⊢ Eq (ZMod.cast 1) 1
  -/
  rw [ZMod.cast_eq_val, ZMod.val_one, Nat.cast_one]
  /-
    🎉 no goals
  -/


theorem nthHomSeq_add (r s : R) :
    nthHomSeq f_compat (r + s) ≈ nthHomSeq f_compat r + nthHomSeq f_compat s := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ⊢ HasEquiv.Equiv (PadicInt.nthHomSeq f_compat (HAdd.hAdd r s)) (HAdd.hAdd (Pad …
  -/
  intro ε hε
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub (Pa …
  -/
  obtain ⟨n, hn⟩ := exists_pow_neg_lt_rat p hε
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub (Pa …
  -/
  use n
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ ∀ (j : Nat), GE.ge j n → LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq …
  -/
  intro j hj
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq f_compat (HAdd.hAdd r s) …
  -/
  dsimp [nthHomSeq]
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ LT.lt (padicNorm p (HSub.hSub (↑(PadicInt.nthHom f (HAdd.hAdd r s) j)) (HAdd …
  -/
  apply lt_of_le_of_lt _ hn
  rw [← Int.cast_add, ← Int.cast_sub, ← padicNorm.dvd_iff_norm_le, ←
    ZMod.intCast_zmod_eq_zero_iff_dvd]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (↑(HSub.hSub (PadicInt.nthHom f (HAdd.hAdd r s) j) (HAdd.hAdd (PadicInt.n …
  -/
  dsimp [nthHom]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (↑(HSub.hSub (↑((f j) (HAdd.hAdd r s)).val) (HAdd.hAdd ↑((f j) r).val ↑(( …
  -/
  simp only [ZMod.natCast_val, RingHom.map_add, Int.cast_sub, ZMod.intCast_cast, Int.cast_add]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((f j) r) ((f j) s)).cast (HAdd.hAdd ((f j) r).cast …
  -/
  rw [ZMod.cast_add (show p ^ n ∣ p ^ j from pow_dvd_pow _ hj)]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (HSub.hSub (HAdd.hAdd ((f j) r).cast ((f j) s).cast) (HAdd.hAdd ((f j) r) …
  -/
  simp only [cast_add, ZMod.natCast_val, Int.cast_add, ZMod.intCast_cast, sub_self]
  /-
    🎉 no goals
  -/


theorem nthHomSeq_mul (r s : R) :
    nthHomSeq f_compat (r * s) ≈ nthHomSeq f_compat r * nthHomSeq f_compat s := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ⊢ HasEquiv.Equiv (PadicInt.nthHomSeq f_compat (HMul.hMul r s)) (HMul.hMul (Pad …
  -/
  intro ε hε
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub (Pa …
  -/
  obtain ⟨n, hn⟩ := exists_pow_neg_lt_rat p hε
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (padicNorm p (↑(HSub.hSub (Pa …
  -/
  use n
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ ∀ (j : Nat), GE.ge j n → LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq …
  -/
  intro j hj
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ LT.lt (padicNorm p (↑(HSub.hSub (PadicInt.nthHomSeq f_compat (HMul.hMul r s) …
  -/
  dsimp [nthHomSeq]
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ LT.lt (padicNorm p (HSub.hSub (↑(PadicInt.nthHom f (HMul.hMul r s) j)) (HMul …
  -/
  apply lt_of_le_of_lt _ hn
  rw [← Int.cast_mul, ← Int.cast_sub, ← padicNorm.dvd_iff_norm_le, ←
    ZMod.intCast_zmod_eq_zero_iff_dvd]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (↑(HSub.hSub (PadicInt.nthHom f (HMul.hMul r s) j) (HMul.hMul (PadicInt.n …
  -/
  dsimp [nthHom]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (↑(HSub.hSub (↑((f j) (HMul.hMul r s)).val) (HMul.hMul ↑((f j) r).val ↑(( …
  -/
  simp only [ZMod.natCast_val, RingHom.map_mul, Int.cast_sub, ZMod.intCast_cast, Int.cast_mul]
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r s : R
    ε : Rat
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    j : Nat
    hj : GE.ge j n
    ⊢ Eq (HSub.hSub (HMul.hMul ((f j) r) ((f j) s)).cast (HMul.hMul ((f j) r).cast …
  -/
  rw [ZMod.cast_mul (show p ^ n ∣ p ^ j from pow_dvd_pow _ hj), sub_self]
  /-
    🎉 no goals
  -/


/--
`limNthHom f_compat r` is the limit of a sequence `f` of compatible ring homs `R →+* ZMod (p^k)`.
This is itself a ring hom: see `PadicInt.lift`.
-/
def limNthHom (r : R) : ℤ_[p] :=
  ofIntSeq (nthHom f r) (isCauSeq_nthHom f_compat r)


theorem limNthHom_spec (r : R) :
    ∀ ε : ℝ, 0 < ε → ∃ N : ℕ, ∀ n ≥ N, ‖limNthHom f_compat r - nthHom f r n‖ < ε := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (No …
  -/
  intro ε hε
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Norm.norm (HSub.hSub (PadicI …
  -/
  obtain ⟨ε', hε'0, hε'⟩ : ∃ v : ℚ, (0 : ℝ) < v ∧ ↑v < ε := exists_rat_btwn hε
  /-
    case intro.intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε'0 : LT.lt 0 ↑ε'
    hε' : LT.lt (↑ε') ε
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Norm.norm (HSub.hSub (PadicI …
  -/
  norm_cast at hε'0
  /-
    case intro.intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Norm.norm (HSub.hSub (PadicI …
  -/
  obtain ⟨N, hN⟩ := padicNormE.defn (nthHomSeq f_compat r) hε'0
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic.mk (PadicInt …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Norm.norm (HSub.hSub (PadicI …
  -/
  use N
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic.mk (PadicInt …
    ⊢ ∀ (n : Nat), GE.ge n N → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNthHom f_c …
  -/
  intro n hn
  /-
    case h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic.mk (PadicInt …
    n : Nat
    hn : GE.ge n N
    ⊢ LT.lt (Norm.norm (HSub.hSub (PadicInt.limNthHom f_compat r) ↑(PadicInt.nthHo …
  -/
  apply _root_.lt_trans _ hε'
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic.mk (PadicInt …
    n : Nat
    hn : GE.ge n N
    ⊢ LT.lt (Norm.norm (HSub.hSub (PadicInt.limNthHom f_compat r) ↑(PadicInt.nthHo …
  -/
  change (padicNormE _  : ℝ) < _
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic.mk (PadicInt …
    n : Nat
    hn : GE.ge n N
    ⊢ LT.lt ↑(padicNormE ↑(HSub.hSub (PadicInt.limNthHom f_compat r) ↑(PadicInt.nt …
  -/
  norm_cast
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    ε : Real
    hε : LT.lt 0 ε
    ε' : Rat
    hε' : LT.lt (↑ε') ε
    hε'0 : LT.lt 0 ε'
    N : Nat
    hN : ∀ (i : Nat), GE.ge i N → LT.lt (padicNormE (HSub.hSub (Padic.mk (PadicInt …
    n : Nat
    hn : GE.ge n N
    ⊢ LT.lt (padicNormE ↑(HSub.hSub (PadicInt.limNthHom f_compat r) ↑(PadicInt.nth …
  -/
  exact hN _ hn
  /-
    🎉 no goals
  -/


                                                        /-
                                                          R : Type u_1
                                                          inst✝ : NonAssocSemiring R
                                                          p : Nat
                                                          f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
                                                          hp_prime : Fact (Nat.Prime p)
                                                          f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
                                                          ⊢ Eq (PadicInt.limNthHom f_compat 0) 0
                                                        -/
theorem limNthHom_zero : limNthHom f_compat 0 = 0 := by simp [limNthHom]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem limNthHom_one : limNthHom f_compat 1 = 1 :=
  Subtype.ext <| Quot.sound <| nthHomSeq_one f_compat


theorem limNthHom_add (r s : R) :
    limNthHom f_compat (r + s) = limNthHom f_compat r + limNthHom f_compat s :=
  Subtype.ext <| Quot.sound <| nthHomSeq_add f_compat _ _


theorem limNthHom_mul (r s : R) :
    limNthHom f_compat (r * s) = limNthHom f_compat r * limNthHom f_compat s :=
  Subtype.ext <| Quot.sound <| nthHomSeq_mul f_compat _ _

-- TODO: generalize this to arbitrary complete discrete valuation rings

/-- `lift f_compat` is the limit of a sequence `f` of compatible ring homs `R →+* ZMod (p^k)`,
with the equality `lift f_compat r = PadicInt.limNthHom f_compat r`.
-/
def lift : R →+* ℤ_[p] where
  toFun := limNthHom f_compat
  map_one' := limNthHom_one f_compat
  map_mul' := limNthHom_mul f_compat
  map_zero' := limNthHom_zero f_compat
  map_add' := limNthHom_add f_compat


theorem lift_sub_val_mem_span (r : R) (n : ℕ) :
    lift f_compat r - (f n r).val ∈ (Ideal.span {(p : ℤ_[p]) ^ n}) := by
  obtain ⟨k, hk⟩ :=
    limNthHom_spec f_compat r _
      (show (0 : ℝ) < (p : ℝ) ^ (-n : ℤ) from zpow_pos (mod_cast hp_prime.1.pos) _)
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    n k : Nat
    hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  have := le_of_lt (hk (max n k) (le_max_right _ _))
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    n k : Nat
    hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
    this : LE.le (Norm.norm (HSub.hSub (PadicInt.limNthHom f_compat r) ↑(PadicInt. …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  rw [norm_le_pow_iff_mem_span_pow] at this
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    n k : Nat
    hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
    this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (H …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  dsimp [lift]
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    n k : Nat
    hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
    this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (H …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  rw [sub_eq_sub_add_sub (limNthHom f_compat r) _ ↑(nthHom f r (max n k))]
  /-
    case intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    n k : Nat
    hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
    this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (H …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HAdd.h …
  -/
  apply Ideal.add_mem _ _ this
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    r : R
    n k : Nat
    hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
    this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (H …
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  rw [Ideal.mem_span_singleton]
  convert
    (Int.castRingHom ℤ_[p]).map_dvd (pow_dvd_nthHom_sub f_compat r n (max n k) (le_max_left _ _))
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : NonAssocSemiring R
      p : Nat
      f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
      hp_prime : Fact (Nat.Prime p)
      f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
      r : R
      n k : Nat
      hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
      this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (H …
      ⊢ Eq (HPow.hPow (↑p) n) ((Int.castRingHom (PadicInt p)) (HPow.hPow (↑p) n))
    -/
  · rw [map_pow]; rfl
                  /-
                    🎉 no goals
                  -/
    /-
      case h.e'_4
      R : Type u_1
      inst✝ : NonAssocSemiring R
      p : Nat
      f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
      hp_prime : Fact (Nat.Prime p)
      f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
      r : R
      n k : Nat
      hk : ∀ (n_1 : Nat), GE.ge n_1 k → LT.lt (Norm.norm (HSub.hSub (PadicInt.limNth …
      this : Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (H …
      ⊢ Eq (HSub.hSub ↑(PadicInt.nthHom f r (Max.max n k)) ↑((f n) r).val) ((Int.cas …
    -/
  · rw [map_sub]; rfl
                  /-
                    🎉 no goals
                  -/


/-- One part of the universal property of `ℤ_[p]` as a projective limit.
See also `PadicInt.lift_unique`.
-/
theorem lift_spec (n : ℕ) : (toZModPow n).comp (lift f_compat) = f n := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    n : Nat
    ⊢ Eq ((PadicInt.toZModPow n).comp (PadicInt.lift f_compat)) (f n)
  -/
  ext r
  rw [RingHom.comp_apply, ← ZMod.natCast_zmod_val (f n r), ← map_natCast <| toZModPow n, ←
    sub_eq_zero, ← RingHom.map_sub, ← RingHom.mem_ker, ker_toZModPow]
  /-
    case a
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    n : Nat
    r : R
    ⊢ Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) (HSub.h …
  -/
  apply lift_sub_val_mem_span
  /-
    🎉 no goals
  -/


/-- One part of the universal property of `ℤ_[p]` as a projective limit.
See also `PadicInt.lift_spec`.
-/
theorem lift_unique (g : R →+* ℤ_[p]) (hg : ∀ n, (toZModPow n).comp g = f n) :
    lift f_compat = g := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    g : RingHom R (PadicInt p)
    hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) (f n)
    ⊢ Eq (PadicInt.lift f_compat) g
  -/
  ext1 r
  /-
    case a
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    g : RingHom R (PadicInt p)
    hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) (f n)
    r : R
    ⊢ Eq ((PadicInt.lift f_compat) r) (g r)
  -/
  apply eq_of_forall_dist_le
  /-
    case a.h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    g : RingHom R (PadicInt p)
    hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) (f n)
    r : R
    ⊢ ∀ (ε : Real), GT.gt ε 0 → LE.le (Dist.dist ((PadicInt.lift f_compat) r) (g r …
  -/
  intro ε hε
  /-
    case a.h
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    g : RingHom R (PadicInt p)
    hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) (f n)
    r : R
    ε : Real
    hε : GT.gt ε 0
    ⊢ LE.le (Dist.dist ((PadicInt.lift f_compat) r) (g r)) ε
  -/
  obtain ⟨n, hn⟩ := exists_pow_neg_lt p hε
  /-
    case a.h.intro
    R : Type u_1
    inst✝ : NonAssocSemiring R
    p : Nat
    f : (k : Nat) → RingHom R (ZMod (HPow.hPow p k))
    hp_prime : Fact (Nat.Prime p)
    f_compat : ∀ (k1 k2 : Nat) (hk : LE.le k1 k2), Eq ((ZMod.castHom ⋯ (ZMod (HPow …
    g : RingHom R (PadicInt p)
    hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) (f n)
    r : R
    ε : Real
    hε : GT.gt ε 0
    n : Nat
    hn : LT.lt (HPow.hPow (↑p) (Neg.neg ↑n)) ε
    ⊢ LE.le (Dist.dist ((PadicInt.lift f_compat) r) (g r)) ε
  -/
  apply le_trans _ (le_of_lt hn)
  rw [dist_eq_norm, norm_le_pow_iff_mem_span_pow, ← ker_toZModPow, RingHom.mem_ker,
    RingHom.map_sub, ← RingHom.comp_apply, ← RingHom.comp_apply, lift_spec, hg, sub_self]


@[simp]
theorem lift_self (z : ℤ_[p]) : lift zmod_cast_comp_toZModPow z = z := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    z : PadicInt p
    ⊢ Eq ((PadicInt.lift ⋯) z) z
  -/
  show _ = RingHom.id _ z
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    z : PadicInt p
    ⊢ Eq ((PadicInt.lift ⋯) z) ((RingHom.id (PadicInt p)) z)
  -/
  rw [lift_unique zmod_cast_comp_toZModPow (RingHom.id ℤ_[p])]
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    z : PadicInt p
    ⊢ ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp (RingHom.id (PadicInt p))) (Pad …
  -/
  intro; rw [RingHom.comp_id]
         /-
           🎉 no goals
         -/


theorem ext_of_toZModPow {x y : ℤ_[p]} : (∀ n, toZModPow n x = toZModPow n y) ↔ x = y := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    x y : PadicInt p
    ⊢ Iff (∀ (n : Nat), Eq ((PadicInt.toZModPow n) x) ((PadicInt.toZModPow n) y))  …
  -/
  constructor
    /-
      case mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x y : PadicInt p
      ⊢ (∀ (n : Nat), Eq ((PadicInt.toZModPow n) x) ((PadicInt.toZModPow n) y)) → Eq …
    -/
  · intro h
    /-
      case mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x y : PadicInt p
      h : ∀ (n : Nat), Eq ((PadicInt.toZModPow n) x) ((PadicInt.toZModPow n) y)
      ⊢ Eq x y
    -/
    rw [← lift_self x, ← lift_self y]
    /-
      case mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x y : PadicInt p
      h : ∀ (n : Nat), Eq ((PadicInt.toZModPow n) x) ((PadicInt.toZModPow n) y)
      ⊢ Eq ((PadicInt.lift ⋯) x) ((PadicInt.lift ⋯) y)
    -/
    simp (config := { unfoldPartialApp := true }) [lift, limNthHom, nthHom, h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x y : PadicInt p
      ⊢ Eq x y → ∀ (n : Nat), Eq ((PadicInt.toZModPow n) x) ((PadicInt.toZModPow n) y)
    -/
  · rintro rfl _
    /-
      case mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      x : PadicInt p
      n✝ : Nat
      ⊢ Eq ((PadicInt.toZModPow n✝) x) ((PadicInt.toZModPow n✝) x)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem toZModPow_eq_iff_ext {R : Type*} [NonAssocSemiring R] {g g' : R →+* ℤ_[p]} :
    (∀ n, (toZModPow n).comp g = (toZModPow n).comp g') ↔ g = g' := by
  /-
    p : Nat
    hp_prime : Fact (Nat.Prime p)
    R : Type u_1
    inst✝ : NonAssocSemiring R
    g g' : RingHom R (PadicInt p)
    ⊢ Iff (∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n) …
  -/
  constructor
    /-
      case mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      ⊢ (∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n).com …
    -/
  · intro hg
    /-
      case mp
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n).c …
      ⊢ Eq g g'
    -/
    ext x : 1
    /-
      case mp.a
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n).c …
      x : R
      ⊢ Eq (g x) (g' x)
    -/
    apply ext_of_toZModPow.mp
    /-
      case mp.a
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n).c …
      x : R
      ⊢ ∀ (n : Nat), Eq ((PadicInt.toZModPow n) (g x)) ((PadicInt.toZModPow n) (g' x))
    -/
    intro n
    /-
      case mp.a
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n).c …
      x : R
      n : Nat
      ⊢ Eq ((PadicInt.toZModPow n) (g x)) ((PadicInt.toZModPow n) (g' x))
    -/
    show (toZModPow n).comp g x = (toZModPow n).comp g' x
    /-
      case mp.a
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      hg : ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModPow n).c …
      x : R
      n : Nat
      ⊢ Eq (((PadicInt.toZModPow n).comp g) x) (((PadicInt.toZModPow n).comp g') x)
    -/
    rw [hg n]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g g' : RingHom R (PadicInt p)
      ⊢ Eq g g' → ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp g) ((PadicInt.toZModP …
    -/
  · rintro rfl _
    /-
      case mpr
      p : Nat
      hp_prime : Fact (Nat.Prime p)
      R : Type u_1
      inst✝ : NonAssocSemiring R
      g : RingHom R (PadicInt p)
      n✝ : Nat
      ⊢ Eq ((PadicInt.toZModPow n✝).comp g) ((PadicInt.toZModPow n✝).comp g)
    -/
    rfl
    /-
      🎉 no goals
    -/


