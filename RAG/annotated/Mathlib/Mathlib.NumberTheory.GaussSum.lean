/-- Definition of the Gauss sum associated to a multiplicative and an additive character. -/
def gaussSum (χ : MulChar R R') (ψ : AddChar R R') : R' :=
  ∑ a, χ a * ψ a


/-- Replacing `ψ` by `mulShift ψ a` and multiplying the Gauss sum by `χ a` does not change it. -/
theorem gaussSum_mulShift (χ : MulChar R R') (ψ : AddChar R R') (a : Rˣ) :
    χ a * gaussSum χ (mulShift ψ a) = gaussSum χ ψ := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ : MulChar R R'
    ψ : AddChar R R'
    a : Units R
    ⊢ Eq (HMul.hMul (χ ↑a) (gaussSum χ (ψ.mulShift ↑a))) (gaussSum χ ψ)
  -/
  simp only [gaussSum, mulShift_apply, Finset.mul_sum]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ : MulChar R R'
    ψ : AddChar R R'
    a : Units R
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (χ ↑a) (HMul.hMul (χ i) (ψ (HMul.hMul …
  -/
  simp_rw [← mul_assoc, ← map_mul]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ : MulChar R R'
    ψ : AddChar R R'
    a : Units R
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (χ (HMul.hMul (↑a) x)) (ψ (HMul.hMul  …
  -/
  exact Fintype.sum_bijective _ a.mulLeft_bijective _ _ fun x ↦ rfl
  /-
    🎉 no goals
  -/


open Finset in
/-- A formula for the product of two Gauss sums with the same additive character. -/
lemma gaussSum_mul {R : Type u} [CommRing R] [Fintype R] {R' : Type v} [CommRing R']
    (χ φ : MulChar R R') (ψ : AddChar R R') :
    gaussSum χ ψ * gaussSum φ ψ = ∑ t : R, ∑ x : R, χ x * φ (t - x) * ψ t := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ φ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (HMul.hMul (gaussSum χ ψ) (gaussSum φ ψ)) (Finset.univ.sum fun t => Finse …
  -/
  rw [gaussSum, gaussSum, sum_mul_sum]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ φ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (Finset.univ.sum fun i => Finset.univ.sum fun j => HMul.hMul (HMul.hMul ( …
  -/
  conv => enter [1, 2, x, 2, x_1]; rw [mul_mul_mul_comm]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ φ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HMul.hMul (HMul.hMul …
  -/
  simp only [← ψ.map_add_eq_mul]
  have sum_eq x : ∑ y : R, χ x * φ y * ψ (x + y) = ∑ y : R, χ x * φ (y - x) * ψ y := by
    rw [sum_bij (fun a _ ↦ a + x)]
    · simp only [mem_univ, forall_true_left, forall_const]
    · simp only [mem_univ, add_left_inj, imp_self, forall_const]
    · exact fun b _ ↦ ⟨b - x, mem_univ _, by rw [sub_add_cancel]⟩
    · exact fun a _ ↦ by rw [add_sub_cancel_right, add_comm]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ φ : MulChar R R'
    ψ : AddChar R R'
    sum_eq : ∀ (x : R), Eq (Finset.univ.sum fun y => HMul.hMul (HMul.hMul (χ x) (φ …
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HMul.hMul (HMul.hMul …
  -/
  rw [sum_congr rfl fun x _ ↦ sum_eq x, sum_comm]
  /-
    🎉 no goals
  -/

-- In the following, we need `R` to be a finite field.

lemma mul_gaussSum_inv_eq_gaussSum (χ : MulChar R R') (ψ : AddChar R R') :
    χ (-1) * gaussSum χ ψ⁻¹ = gaussSum χ ψ := by
  /-
    R : Type u
    inst✝² : Field R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (HMul.hMul (χ (-1)) (gaussSum χ (Inv.inv ψ))) (gaussSum χ ψ)
  -/
  rw [ψ.inv_mulShift, ← Units.coe_neg_one]
  /-
    R : Type u
    inst✝² : Field R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    χ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (HMul.hMul (χ ↑(-1)) (gaussSum χ (ψ.mulShift ↑(-1)))) (gaussSum χ ψ)
  -/
  exact gaussSum_mulShift χ ψ (-1)
  /-
    🎉 no goals
  -/


private theorem gaussSum_mul_aux {χ : MulChar R R'} (hχ : χ ≠ 1) (ψ : AddChar R R')
    (b : R) :
    ∑ a, χ (a * b⁻¹) * ψ (a - b) = ∑ c, χ c * ψ (b * (c - 1)) := by
  /-
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    ψ : AddChar R R'
    b : R
    ⊢ Eq (Finset.univ.sum fun a => HMul.hMul (χ (HMul.hMul a (Inv.inv b))) (ψ (HSu …
  -/
  rcases eq_or_ne b 0 with hb | hb
  · -- case `b = 0`
    simp only [hb, inv_zero, mul_zero, MulChar.map_zero, zero_mul,
      Finset.sum_const_zero, map_zero_eq_one, mul_one, χ.sum_eq_zero_of_ne_one hχ]
  · -- case `b ≠ 0`
    /-
      case inr
      R : Type u
      inst✝³ : Field R
      inst✝² : Fintype R
      R' : Type v
      inst✝¹ : CommRing R'
      inst✝ : IsDomain R'
      χ : MulChar R R'
      hχ : Ne χ 1
      ψ : AddChar R R'
      b : R
      hb : Ne b 0
      ⊢ Eq (Finset.univ.sum fun a => HMul.hMul (χ (HMul.hMul a (Inv.inv b))) (ψ (HSu …
    -/
    refine (Fintype.sum_bijective _ (mulLeft_bijective₀ b hb) _ _ fun x ↦ ?_).symm
    /-
      case inr
      R : Type u
      inst✝³ : Field R
      inst✝² : Fintype R
      R' : Type v
      inst✝¹ : CommRing R'
      inst✝ : IsDomain R'
      χ : MulChar R R'
      hχ : Ne χ 1
      ψ : AddChar R R'
      b : R
      hb : Ne b 0
      x : R
      ⊢ Eq (HMul.hMul (χ x) (ψ (HMul.hMul b (HSub.hSub x 1)))) (HMul.hMul (χ (HMul.h …
    -/
    rw [mul_assoc, mul_comm x, ← mul_assoc, mul_inv_cancel₀ hb, one_mul, mul_sub, mul_one]
    /-
      🎉 no goals
    -/


/-- We have `gaussSum χ ψ * gaussSum χ⁻¹ ψ⁻¹ = Fintype.card R`
when `χ` is nontrivial and `ψ` is primitive (and `R` is a field). -/
theorem gaussSum_mul_gaussSum_eq_card {χ : MulChar R R'} (hχ : χ ≠ 1) {ψ : AddChar R R'}
    (hψ : IsPrimitive ψ) :
    gaussSum χ ψ * gaussSum χ⁻¹ ψ⁻¹ = Fintype.card R := by
  /-
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (HMul.hMul (gaussSum χ ψ) (gaussSum (Inv.inv χ) (Inv.inv ψ))) ↑(Fintype.c …
  -/
  simp only [gaussSum, AddChar.inv_apply, Finset.sum_mul, Finset.mul_sum, MulChar.inv_apply']
  conv =>
    enter [1, 2, x, 2, y]
    rw [mul_mul_mul_comm, ← map_mul, ← map_add_eq_mul, ← sub_eq_add_neg]
--  conv in _ * _ * (_ * _) => rw [mul_mul_mul_comm, ← map_mul, ← map_add_eq_mul, ← sub_eq_add_neg]
  /-
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun y => HMul.hMul (χ (HMul.hMu …
  -/
  simp_rw [gaussSum_mul_aux hχ ψ]
  /-
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : Ne χ 1
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun c => HMul.hMul (χ c) (ψ (HM …
  -/
  rw [Finset.sum_comm]
  classical -- to get `[DecidableEq R]` for `sum_mulShift`
  simp_rw [← Finset.mul_sum, sum_mulShift _ hψ, sub_eq_zero, apply_ite, Nat.cast_zero, mul_zero]
  rw [Finset.sum_ite_eq' Finset.univ (1 : R)]
  simp only [Finset.mem_univ, map_one, one_mul, if_true]


/-- If `χ` is a multiplicative character of order `n` on a finite field `F`,
then `g(χ) * g(χ^(n-1)) = χ(-1)*#F` -/
lemma gaussSum_mul_gaussSum_pow_orderOf_sub_one {χ : MulChar R R'} {ψ : AddChar R R'}
    (hχ : χ ≠ 1) (hψ : ψ.IsPrimitive) :
    gaussSum χ ψ * gaussSum (χ ^ (orderOf χ - 1)) ψ = χ (-1) * Fintype.card R := by
  have h : χ ^ (orderOf χ - 1) = χ⁻¹ := by
    refine (inv_eq_of_mul_eq_one_right ?_).symm
    rw [← pow_succ', Nat.sub_one_add_one_eq_of_pos χ.orderOf_pos, pow_orderOf_eq_one]
  rw [h, ← mul_gaussSum_inv_eq_gaussSum χ⁻¹, mul_left_comm, gaussSum_mul_gaussSum_eq_card hχ hψ,
    MulChar.inv_apply', inv_neg_one]


/-- The Gauss sum of a nontrivial character on a finite field does not vanish. -/
lemma gaussSum_ne_zero_of_nontrivial (h : (Fintype.card R : R') ≠ 0) {χ : MulChar R R'}
    (hχ : χ ≠ 1) {ψ : AddChar R R'} (hψ : ψ.IsPrimitive) :
    gaussSum χ ψ ≠ 0 :=
  fun H ↦ h.symm <| zero_mul (gaussSum χ⁻¹ _) ▸ H ▸ gaussSum_mul_gaussSum_eq_card hχ hψ


/-- When `χ` is a nontrivial quadratic character, then the square of `gaussSum χ ψ`
is `χ(-1)` times the cardinality of `R`. -/
theorem gaussSum_sq {χ : MulChar R R'} (hχ₁ : χ ≠ 1) (hχ₂ : IsQuadratic χ)
    {ψ : AddChar R R'} (hψ : IsPrimitive ψ) :
    gaussSum χ ψ ^ 2 = χ (-1) * Fintype.card R := by
  /-
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card R))
  -/
  rw [pow_two, ← gaussSum_mul_gaussSum_eq_card hχ₁ hψ, hχ₂.inv, mul_rotate']
  /-
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (HMul.hMul (gaussSum χ ψ) (gaussSum χ ψ)) (HMul.hMul (gaussSum χ ψ) (HMul …
  -/
  congr
  /-
    case e_a
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (gaussSum χ ψ) (HMul.hMul (gaussSum χ (Inv.inv ψ)) (χ (-1)))
  -/
  rw [mul_comm, ← gaussSum_mulShift _ _ (-1 : Rˣ), inv_mulShift]
  /-
    case e_a
    R : Type u
    inst✝³ : Field R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    ψ : AddChar R R'
    hψ : ψ.IsPrimitive
    ⊢ Eq (HMul.hMul (χ ↑(-1)) (gaussSum χ (ψ.mulShift ↑(-1)))) (HMul.hMul (χ (-1)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- When `R'` has prime characteristic `p`, then the `p`th power of the Gauss sum
of `χ` and `ψ` is the Gauss sum of `χ^p` and `ψ^p`. -/
theorem gaussSum_frob (χ : MulChar R R') (ψ : AddChar R R') :
    gaussSum χ ψ ^ p = gaussSum (χ ^ p) (ψ ^ p) := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    p : Nat
    fp : Fact (Nat.Prime p)
    hch : CharP R' p
    χ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) p) (gaussSum (HPow.hPow χ p) (HPow.hPow ψ p))
  -/
  rw [← frobenius_def, gaussSum, gaussSum, map_sum]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    p : Nat
    fp : Fact (Nat.Prime p)
    hch : CharP R' p
    χ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (Finset.univ.sum fun x => (frobenius R' p) (HMul.hMul (χ x) (ψ x))) (Fins …
  -/
  simp_rw [pow_apply' χ fp.1.ne_zero, map_mul, frobenius_def]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    p : Nat
    fp : Fact (Nat.Prime p)
    hch : CharP R' p
    χ : MulChar R R'
    ψ : AddChar R R'
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (HPow.hPow (χ x) p) (HPow.hPow (ψ x)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- For a quadratic character `χ` and when the characteristic `p` of the target ring
is a unit in the source ring, the `p`th power of the Gauss sum of`χ` and `ψ` is
`χ p` times the original Gauss sum. -/
theorem MulChar.IsQuadratic.gaussSum_frob (hp : IsUnit (p : R)) {χ : MulChar R R'}
    (hχ : IsQuadratic χ) (ψ : AddChar R R') :
    gaussSum χ ψ ^ p = χ p * gaussSum χ ψ := by
  rw [_root_.gaussSum_frob, pow_mulShift, hχ.pow_char p, ← gaussSum_mulShift χ ψ hp.unit,
    ← mul_assoc, hp.unit_spec, ← pow_two, ← pow_apply' _ two_ne_zero, hχ.sq_eq_one, ← hp.unit_spec,
    one_apply_coe, one_mul]


/-- For a quadratic character `χ` and when the characteristic `p` of the target ring
is a unit in the source ring and `n` is a natural number, the `p^n`th power of the Gauss
sum of`χ` and `ψ` is `χ (p^n)` times the original Gauss sum. -/
theorem MulChar.IsQuadratic.gaussSum_frob_iter (n : ℕ) (hp : IsUnit (p : R)) {χ : MulChar R R'}
    (hχ : IsQuadratic χ) (ψ : AddChar R R') :
    gaussSum χ ψ ^ p ^ n = χ ((p : R) ^ n) * gaussSum χ ψ := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Fintype R
    R' : Type v
    inst✝ : CommRing R'
    p : Nat
    fp : Fact (Nat.Prime p)
    hch : CharP R' p
    n : Nat
    hp : IsUnit ↑p
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    ψ : AddChar R R'
    ⊢ Eq (HPow.hPow (gaussSum χ ψ) (HPow.hPow p n)) (HMul.hMul (χ (HPow.hPow (↑p)  …
  -/
  induction' n with n ih
    /-
      case zero
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : Fintype R
      R' : Type v
      inst✝ : CommRing R'
      p : Nat
      fp : Fact (Nat.Prime p)
      hch : CharP R' p
      hp : IsUnit ↑p
      χ : MulChar R R'
      hχ : χ.IsQuadratic
      ψ : AddChar R R'
      ⊢ Eq (HPow.hPow (gaussSum χ ψ) (HPow.hPow p 0)) (HMul.hMul (χ (HPow.hPow (↑p)  …
    -/
  · rw [pow_zero, pow_one, pow_zero, MulChar.map_one, one_mul]
    /-
      🎉 no goals
    -/
  · rw [pow_succ, pow_mul, ih, mul_pow, hχ.gaussSum_frob _ hp, ← mul_assoc, pow_succ, map_mul,
      ← pow_apply' χ fp.1.ne_zero ((p : R) ^ n), hχ.pow_char p]


/-- If the square of the Gauss sum of a quadratic character is `χ(-1) * #R`,
then we get, for all `n : ℕ`, the relation `(χ(-1) * #R) ^ (p^n/2) = χ(p^n)`,
where `p` is the (odd) characteristic of the target ring `R'`.
This version can be used when `R` is not a field, e.g., `ℤ/8ℤ`. -/
theorem Char.card_pow_char_pow {χ : MulChar R R'} (hχ : IsQuadratic χ) (ψ : AddChar R R') (p n : ℕ)
    [fp : Fact p.Prime] [hch : CharP R' p] (hp : IsUnit (p : R)) (hp' : p ≠ 2)
    (hg : gaussSum χ ψ ^ 2 = χ (-1) * Fintype.card R) :
    (χ (-1) * Fintype.card R) ^ (p ^ n / 2) = χ ((p : R) ^ n) := by
  have : gaussSum χ ψ ≠ 0 := by
    intro hf
    rw [hf, zero_pow two_ne_zero, eq_comm, mul_eq_zero] at hg
    exact not_isUnit_prime_of_dvd_card p
        ((CharP.cast_eq_zero_iff R' p _).mp <| hg.resolve_left (isUnit_one.neg.map χ).ne_zero) hp
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    ψ : AddChar R R'
    p n : Nat
    fp : Fact (Nat.Prime p)
    hch : CharP R' p
    hp : IsUnit ↑p
    hp' : Ne p 2
    hg : Eq (HPow.hPow (gaussSum χ ψ) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card R))
    this : Ne (gaussSum χ ψ) 0
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card R)) (HDiv.hDiv (HPow.hPow p …
  -/
  rw [← hg]
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : Fintype R
    R' : Type v
    inst✝¹ : CommRing R'
    inst✝ : IsDomain R'
    χ : MulChar R R'
    hχ : χ.IsQuadratic
    ψ : AddChar R R'
    p n : Nat
    fp : Fact (Nat.Prime p)
    hch : CharP R' p
    hp : IsUnit ↑p
    hp' : Ne p 2
    hg : Eq (HPow.hPow (gaussSum χ ψ) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card R))
    this : Ne (gaussSum χ ψ) 0
    ⊢ Eq (HPow.hPow (HPow.hPow (gaussSum χ ψ) 2) (HDiv.hDiv (HPow.hPow p n) 2)) (χ …
  -/
  apply mul_right_cancel₀ this
  rw [← hχ.gaussSum_frob_iter p n hp ψ, ← pow_mul, ← pow_succ,
    Nat.two_mul_div_two_add_one_of_odd (fp.1.eq_two_or_odd'.resolve_left hp').pow]


/-- When `F` and `F'` are finite fields and `χ : F → F'` is a nontrivial quadratic character,
then `(χ(-1) * #F)^(#F'/2) = χ #F'`. -/
theorem Char.card_pow_card {F : Type*} [Field F] [Fintype F] {F' : Type*} [Field F'] [Fintype F']
    {χ : MulChar F F'} (hχ₁ : χ ≠ 1) (hχ₂ : IsQuadratic χ)
    (hch₁ : ringChar F' ≠ ringChar F) (hch₂ : ringChar F' ≠ 2) :
    (χ (-1) * Fintype.card F) ^ (Fintype.card F' / 2) = χ (Fintype.card F') := by
  /-
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (HDiv.hDiv (Fintype.car …
  -/
  obtain ⟨n, hp, hc⟩ := FiniteField.card F (ringChar F)
  /-
    case intro.intro
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (HDiv.hDiv (Fintype.car …
  -/
  obtain ⟨n', hp', hc'⟩ := FiniteField.card F' (ringChar F')
  /-
    case intro.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (HDiv.hDiv (Fintype.car …
  -/
  let ψ := FiniteField.primitiveChar F F' hch₁
  /-
    case intro.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (HDiv.hDiv (Fintype.car …
  -/
  let FF' := CyclotomicField ψ.n F'
  /-
    case intro.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (HDiv.hDiv (Fintype.car …
  -/
  have hchar := Algebra.ringChar_eq F' FF'
  /-
    case intro.intro.intro.intro
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    hchar : Eq (ringChar F') (ringChar FF')
    ⊢ Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (HDiv.hDiv (Fintype.car …
  -/
  apply (algebraMap F' FF').injective
  /-
    case intro.intro.intro.intro.a
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    hchar : Eq (ringChar F') (ringChar FF')
    ⊢ Eq ((algebraMap F' FF') (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card F)) (H …
  -/
  rw [map_pow, map_mul, map_natCast, hc', hchar, Nat.cast_pow]
  /-
    case intro.intro.intro.intro.a
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    hchar : Eq (ringChar F') (ringChar FF')
    ⊢ Eq (HPow.hPow (HMul.hMul ((algebraMap F' FF') (χ (-1))) ↑(Fintype.card F)) ( …
  -/
  simp only [← MulChar.ringHomComp_apply]
  /-
    case intro.intro.intro.intro.a
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    hchar : Eq (ringChar F') (ringChar FF')
    ⊢ Eq (HPow.hPow (HMul.hMul ((χ.ringHomComp (algebraMap F' FF')) (-1)) ↑(Fintyp …
  -/
  have := Fact.mk hp'
  /-
    case intro.intro.intro.intro.a
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    hchar : Eq (ringChar F') (ringChar FF')
    this : Fact (Nat.Prime (ringChar F'))
    ⊢ Eq (HPow.hPow (HMul.hMul ((χ.ringHomComp (algebraMap F' FF')) (-1)) ↑(Fintyp …
  -/
  have := Fact.mk (hchar.subst hp')
  /-
    case intro.intro.intro.intro.a
    F : Type u_1
    inst✝³ : Field F
    inst✝² : Fintype F
    F' : Type u_2
    inst✝¹ : Field F'
    inst✝ : Fintype F'
    χ : MulChar F F'
    hχ₁ : Ne χ 1
    hχ₂ : χ.IsQuadratic
    hch₁ : Ne (ringChar F') (ringChar F)
    hch₂ : Ne (ringChar F') 2
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    n' : PNat
    hp' : Nat.Prime (ringChar F')
    hc' : Eq (Fintype.card F') (HPow.hPow (ringChar F') ↑n')
    ψ : AddChar.PrimitiveAddChar F F' := AddChar.FiniteField.primitiveChar F F' hch₁
    FF' : Type u_2 := CyclotomicField ψ.n F'
    hchar : Eq (ringChar F') (ringChar FF')
    this✝ : Fact (Nat.Prime (ringChar F'))
    this : Fact (Nat.Prime (ringChar FF'))
    ⊢ Eq (HPow.hPow (HMul.hMul ((χ.ringHomComp (algebraMap F' FF')) (-1)) ↑(Fintyp …
  -/
  rw [Ne, ← Nat.prime_dvd_prime_iff_eq hp' hp, ← isUnit_iff_not_dvd_char, hchar] at hch₁
  exact Char.card_pow_char_pow (hχ₂.comp _) ψ.char (ringChar FF') n' hch₁ (hchar ▸ hch₂)
       (gaussSum_sq ((ringHomComp_ne_one_iff (RingHom.injective _)).mpr hχ₁) (hχ₂.comp _) ψ.prim)


/-- For every finite field `F` of odd characteristic, we have `2^(#F/2) = χ₈ #F` in `F`. -/
theorem FiniteField.two_pow_card {F : Type*} [Fintype F] [Field F] (hF : ringChar F ≠ 2) :
    (2 : F) ^ (Fintype.card F / 2) = χ₈ (Fintype.card F) := by
  /-
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have hp2 (n : ℕ) : (2 ^ n : F) ≠ 0 := pow_ne_zero n (Ring.two_ne_zero hF)
  /-
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  obtain ⟨n, hp, hc⟩ := FiniteField.card F (ringChar F)

  -- we work in `FF`, the eighth cyclotomic field extension of `F`
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  let FF := CyclotomicField 8 F
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have hchar := Algebra.ringChar_eq F FF
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have FFp := hchar.subst hp
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have := Fact.mk FFp
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have hFF := hchar ▸ hF -- `ringChar FF ≠ 2`
  have hu : IsUnit (ringChar FF : ZMod 8) := by
    rw [isUnit_iff_not_dvd_char, ringChar_zmod_n]
    rw [Ne, ← Nat.prime_dvd_prime_iff_eq FFp Nat.prime_two] at hFF
    change ¬_ ∣ 2 ^ 3
    exact mt FFp.dvd_of_dvd_pow hFF

  -- there is a primitive additive character `ℤ/8ℤ → FF`, sending `a + 8ℤ ↦ τ^a`
  -- with a primitive eighth root of unity `τ`
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  let ψ₈ := primitiveZModChar 8 F (by convert hp2 3 using 1; norm_cast)
  -- We cast from `AddChar (ZMod (8 : ℕ+)) FF` to `AddChar (ZMod 8) FF`
  -- This is needed to make `simp_rw [← h₁]` below work.
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  let ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  let τ : FF := ψ₈char 1
  have τ_spec : τ ^ 4 = -1 := by
    rw [show τ = ψ₈.char 1 from rfl] -- to make `rw [ψ₈.prim.zmod_char_eq_one_iff]` work
    refine (sq_eq_one_iff.1 ?_).resolve_left ?_
    · rw [← pow_mul, ← map_nsmul_eq_pow ψ₈.char, ψ₈.prim.zmod_char_eq_one_iff]
      decide
    · rw [← map_nsmul_eq_pow ψ₈.char, ψ₈.prim.zmod_char_eq_one_iff]
      decide

  -- we consider `χ₈` as a multiplicative character `ℤ/8ℤ → FF`
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  let χ := χ₈.ringHomComp (Int.castRingHom FF)
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have hχ : χ (-1) = 1 := Int.cast_one
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    hχ : Eq (χ (-1)) 1
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have hq : IsQuadratic χ := isQuadratic_χ₈.comp _

  -- we now show that the Gauss sum of `χ` and `ψ₈` has the relevant property
  have h₁ : (fun (a : Fin 8) ↦ ↑(χ₈ a) * τ ^ (a : ℕ)) = fun a ↦ χ a * ↑(ψ₈char a) := by
    ext1; congr; apply pow_one
  have hg₁ : gaussSum χ ψ₈char = 2 * (τ - τ ^ 3) := by
    rw [gaussSum, ← h₁, Fin.sum_univ_eight,
      -- evaluate `χ₈`
      show χ₈ 0 = 0 from rfl, show χ₈ 1 = 1 from rfl, show χ₈ 2 = 0 from rfl,
      show χ₈ 3 = -1 from rfl, show χ₈ 4 = 0 from rfl, show χ₈ 5 = -1 from rfl,
      show χ₈ 6 = 0 from rfl, show χ₈ 7 = 1 from rfl,
      -- normalize exponents
      show ((3 : Fin 8) : ℕ) = 3 from rfl, show ((5 : Fin 8) : ℕ) = 5 from rfl,
      show ((7 : Fin 8) : ℕ) = 7 from rfl]
    simp only [Int.cast_zero, zero_mul, Int.cast_one, Fin.val_one, pow_one, one_mul, zero_add,
      Fin.val_two, add_zero, Int.reduceNeg, Int.cast_neg, neg_mul]
    linear_combination (τ ^ 3 - τ) * τ_spec
  have hg : gaussSum χ ψ₈char ^ 2 = χ (-1) * Fintype.card (ZMod 8) := by
    rw [hχ, one_mul, ZMod.card, Nat.cast_ofNat, hg₁]
    linear_combination (4 * τ ^ 2 - 8) * τ_spec

  -- this allows us to apply `card_pow_char_pow` to our situation
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    hχ : Eq (χ (-1)) 1
    hq : χ.IsQuadratic
    h₁ : Eq (fun a => HMul.hMul (↑(ZMod.χ₈ a)) (HPow.hPow τ ↑a)) fun a => HMul.hMu …
    hg₁ : Eq (gaussSum χ ψ₈char) (HMul.hMul 2 (HSub.hSub τ (HPow.hPow τ 3)))
    hg : Eq (HPow.hPow (gaussSum χ ψ₈char) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card ( …
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  have h := Char.card_pow_char_pow (R := ZMod 8) hq ψ₈char (ringChar FF) n hu hFF hg
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    hχ : Eq (χ (-1)) 1
    hq : χ.IsQuadratic
    h₁ : Eq (fun a => HMul.hMul (↑(ZMod.χ₈ a)) (HPow.hPow τ ↑a)) fun a => HMul.hMu …
    hg₁ : Eq (gaussSum χ ψ₈char) (HMul.hMul 2 (HSub.hSub τ (HPow.hPow τ 3)))
    hg : Eq (HPow.hPow (gaussSum χ ψ₈char) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card ( …
    h : Eq (HPow.hPow (HMul.hMul (χ (-1)) ↑(Fintype.card (ZMod 8))) (HDiv.hDiv (HP …
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  rw [ZMod.card, ← hchar, hχ, one_mul, ← hc, ← Nat.cast_pow (ringChar F), ← hc] at h

  -- finally, we change `2` to `8` on the left hand side
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    hχ : Eq (χ (-1)) 1
    hq : χ.IsQuadratic
    h₁ : Eq (fun a => HMul.hMul (↑(ZMod.χ₈ a)) (HPow.hPow τ ↑a)) fun a => HMul.hMu …
    hg₁ : Eq (gaussSum χ ψ₈char) (HMul.hMul 2 (HSub.hSub τ (HPow.hPow τ 3)))
    hg : Eq (HPow.hPow (gaussSum χ ψ₈char) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card ( …
    h : Eq (HPow.hPow (↑8) (HDiv.hDiv (Fintype.card F) 2)) (χ ↑(Fintype.card F))
    ⊢ Eq (HPow.hPow 2 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  convert_to (8 : F) ^ (Fintype.card F / 2) = _
  · rw [(by norm_num : (8 : F) = 2 ^ 2 * 2), mul_pow,
      (FiniteField.isSquare_iff hF <| hp2 2).mp ⟨2, pow_two 2⟩, one_mul]
  /-
    case intro.intro.convert_2
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    hχ : Eq (χ (-1)) 1
    hq : χ.IsQuadratic
    h₁ : Eq (fun a => HMul.hMul (↑(ZMod.χ₈ a)) (HPow.hPow τ ↑a)) fun a => HMul.hMu …
    hg₁ : Eq (gaussSum χ ψ₈char) (HMul.hMul 2 (HSub.hSub τ (HPow.hPow τ 3)))
    hg : Eq (HPow.hPow (gaussSum χ ψ₈char) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card ( …
    h : Eq (HPow.hPow (↑8) (HDiv.hDiv (Fintype.card F) 2)) (χ ↑(Fintype.card F))
    ⊢ Eq (HPow.hPow 8 (HDiv.hDiv (Fintype.card F) 2)) ↑(ZMod.χ₈ ↑(Fintype.card F))
  -/
  apply (algebraMap F FF).injective
  /-
    case intro.intro.convert_2.a
    F : Type u_1
    inst✝¹ : Fintype F
    inst✝ : Field F
    hF : Ne (ringChar F) 2
    hp2 : ∀ (n : Nat), Ne (HPow.hPow 2 n) 0
    n : PNat
    hp : Nat.Prime (ringChar F)
    hc : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    FF : Type u_1 := CyclotomicField 8 F
    hchar : Eq (ringChar F) (ringChar FF)
    FFp : Nat.Prime (ringChar FF)
    this : Fact (Nat.Prime (ringChar FF))
    hFF : Ne (ringChar FF) 2
    hu : IsUnit ↑(ringChar FF)
    ψ₈ : AddChar.PrimitiveAddChar (ZMod ↑8) F := AddChar.primitiveZModChar 8 F ⋯
    ψ₈char : AddChar (ZMod 8) FF := ψ₈.char
    τ : FF := ψ₈char 1
    τ_spec : Eq (HPow.hPow τ 4) (-1)
    χ : MulChar (ZMod 8) FF := ZMod.χ₈.ringHomComp (Int.castRingHom FF)
    hχ : Eq (χ (-1)) 1
    hq : χ.IsQuadratic
    h₁ : Eq (fun a => HMul.hMul (↑(ZMod.χ₈ a)) (HPow.hPow τ ↑a)) fun a => HMul.hMu …
    hg₁ : Eq (gaussSum χ ψ₈char) (HMul.hMul 2 (HSub.hSub τ (HPow.hPow τ 3)))
    hg : Eq (HPow.hPow (gaussSum χ ψ₈char) 2) (HMul.hMul (χ (-1)) ↑(Fintype.card ( …
    h : Eq (HPow.hPow (↑8) (HDiv.hDiv (Fintype.card F) 2)) (χ ↑(Fintype.card F))
    ⊢ Eq ((algebraMap F FF) (HPow.hPow 8 (HDiv.hDiv (Fintype.card F) 2))) ((algebr …
  -/
  simpa only [map_pow, map_ofNat, map_intCast, Nat.cast_ofNat] using h
  /-
    🎉 no goals
  -/


