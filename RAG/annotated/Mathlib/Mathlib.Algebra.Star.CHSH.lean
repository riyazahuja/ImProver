/-- A CHSH tuple in a *-monoid consists of 4 self-adjoint involutions `A₀ A₁ B₀ B₁` such that
the `Aᵢ` commute with the `Bⱼ`.

The physical interpretation is that `A₀` and `A₁` are a pair of boolean observables which
are spacelike separated from another pair `B₀` and `B₁` of boolean observables.
-/
--@[nolint has_nonempty_instance] Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
structure IsCHSHTuple {R} [Monoid R] [StarMul R] (A₀ A₁ B₀ B₁ : R) : Prop where
  A₀_inv : A₀ ^ 2 = 1
  A₁_inv : A₁ ^ 2 = 1
  B₀_inv : B₀ ^ 2 = 1
  B₁_inv : B₁ ^ 2 = 1
  A₀_sa : star A₀ = A₀
  A₁_sa : star A₁ = A₁
  B₀_sa : star B₀ = B₀
  B₁_sa : star B₁ = B₁
  A₀B₀_commutes : A₀ * B₀ = B₀ * A₀
  A₀B₁_commutes : A₀ * B₁ = B₁ * A₀
  A₁B₀_commutes : A₁ * B₀ = B₀ * A₁
  A₁B₁_commutes : A₁ * B₁ = B₁ * A₁


theorem CHSH_id [CommRing R] {A₀ A₁ B₀ B₁ : R} (A₀_inv : A₀ ^ 2 = 1) (A₁_inv : A₁ ^ 2 = 1)
    (B₀_inv : B₀ ^ 2 = 1) (B₁_inv : B₁ ^ 2 = 1) :
    (2 - A₀ * B₀ - A₀ * B₁ - A₁ * B₀ + A₁ * B₁) * (2 - A₀ * B₀ - A₀ * B₁ - A₁ * B₀ + A₁ * B₁) =
      4 * (2 - A₀ * B₀ - A₀ * B₁ - A₁ * B₀ + A₁ * B₁) := by
  -- polyrith suggests:
  linear_combination
    (2 * B₀ * B₁ + 2) * A₀_inv + (B₀ ^ 2 - 2 * B₀ * B₁ + B₁ ^ 2) * A₁_inv +
        (A₀ ^ 2 + 2 * A₀ * A₁ + 1) * B₀_inv +
      (A₀ ^ 2 - 2 * A₀ * A₁ + 1) * B₁_inv


/-- Given a CHSH tuple (A₀, A₁, B₀, B₁) in a *commutative* ordered `*`-algebra over ℝ,
`A₀ * B₀ + A₀ * B₁ + A₁ * B₀ - A₁ * B₁ ≤ 2`.

(We could work over ℤ[⅟2] if we wanted to!)
-/
theorem CHSH_inequality_of_comm [OrderedCommRing R] [StarRing R] [StarOrderedRing R] [Algebra ℝ R]
    [OrderedSMul ℝ R] (A₀ A₁ B₀ B₁ : R) (T : IsCHSHTuple A₀ A₁ B₀ B₁) :
    A₀ * B₀ + A₀ * B₁ + A₁ * B₀ - A₁ * B₁ ≤ 2 := by
  /-
    R : Type u
    inst✝⁴ : OrderedCommRing R
    inst✝³ : StarRing R
    inst✝² : StarOrderedRing R
    inst✝¹ : Algebra Real R
    inst✝ : OrderedSMul Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul A₀ B₀) (HMul.hMul A₀ B₁))  …
  -/
  let P := 2 - A₀ * B₀ - A₀ * B₁ - A₁ * B₀ + A₁ * B₁
  have i₁ : 0 ≤ P := by
    have idem : P * P = 4 * P := CHSH_id T.A₀_inv T.A₁_inv T.B₀_inv T.B₁_inv
    have idem' : P = (1 / 4 : ℝ) • (P * P) := by
      have h : 4 * P = (4 : ℝ) • P := by simp [map_ofNat, Algebra.smul_def]
      rw [idem, h, ← mul_smul]
      norm_num
    have sa : star P = P := by
      dsimp [P]
      simp only [star_add, star_sub, star_mul, star_ofNat, star_one, T.A₀_sa, T.A₁_sa, T.B₀_sa,
        T.B₁_sa, mul_comm B₀, mul_comm B₁]
    simpa only [← idem', sa]
      using smul_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 4) (star_mul_self_nonneg P)
  /-
    R : Type u
    inst✝⁴ : OrderedCommRing R
    inst✝³ : StarRing R
    inst✝² : StarOrderedRing R
    inst✝¹ : Algebra Real R
    inst✝ : OrderedSMul Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    P : R := HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub 2 (HMul.hMul A₀ B₀)) (HMul …
    i₁ : LE.le 0 P
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul A₀ B₀) (HMul.hMul A₀ B₁))  …
  -/
  apply le_of_sub_nonneg
  /-
    case a
    R : Type u
    inst✝⁴ : OrderedCommRing R
    inst✝³ : StarRing R
    inst✝² : StarOrderedRing R
    inst✝¹ : Algebra Real R
    inst✝ : OrderedSMul Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    P : R := HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub 2 (HMul.hMul A₀ B₀)) (HMul …
    i₁ : LE.le 0 P
    ⊢ LE.le 0 (HSub.hSub 2 (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul A₀ B₀) (HMu …
  -/
  simpa only [sub_add_eq_sub_sub, ← sub_add] using i₁
  /-
    🎉 no goals
  -/


theorem tsirelson_inequality_aux : √2 * √2 ^ 3 = √2 * (2 * (√2)⁻¹ + 4 * ((√2)⁻¹ * 2⁻¹)) := by
  /-
    ⊢ Eq (HMul.hMul (Real.sqrt 2) (HPow.hPow (Real.sqrt 2) 3)) (HMul.hMul (Real.sq …
  -/
  ring_nf
  /-
    ⊢ Eq (HPow.hPow (Real.sqrt 2) 4) (HMul.hMul (HMul.hMul (Real.sqrt 2) (Inv.inv  …
  -/
  rw [mul_inv_cancel₀ (ne_of_gt (Real.sqrt_pos.2 (show (2 : ℝ) > 0 by norm_num)))]
  /-
    ⊢ Eq (HPow.hPow (Real.sqrt 2) 4) (HMul.hMul 1 4)
  -/
  convert congr_arg (· ^ 2) (@Real.sq_sqrt 2 (by norm_num)) using 1 <;>
     /-
       case h.e'_2
       ⊢ Eq (HPow.hPow (Real.sqrt 2) 4) (HPow.hPow (HPow.hPow (Real.sqrt 2) 2) 2)
     -/
                                    /-
                                      🎉 no goals
                                    -/
    (try simp only [← pow_mul]) <;> norm_num
                                    /-
                                      🎉 no goals
                                    -/


theorem sqrt_two_inv_mul_self : (√2)⁻¹ * (√2)⁻¹ = (2⁻¹ : ℝ) := by
  /-
    ⊢ Eq (HMul.hMul (Inv.inv (Real.sqrt 2)) (Inv.inv (Real.sqrt 2))) (Inv.inv 2)
  -/
  rw [← mul_inv]
  /-
    ⊢ Eq (Inv.inv (HMul.hMul (Real.sqrt 2) (Real.sqrt 2))) (Inv.inv 2)
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- In a noncommutative ordered `*`-algebra over ℝ,
Tsirelson's bound for a CHSH tuple (A₀, A₁, B₀, B₁) is
`A₀ * B₀ + A₀ * B₁ + A₁ * B₀ - A₁ * B₁ ≤ 2^(3/2) • 1`.

We prove this by providing an explicit sum-of-squares decomposition
of the difference.

(We could work over `ℤ[2^(1/2), 2^(-1/2)]` if we really wanted to!)
-/
theorem tsirelson_inequality [OrderedRing R] [StarRing R] [StarOrderedRing R] [Algebra ℝ R]
    [OrderedSMul ℝ R] [StarModule ℝ R] (A₀ A₁ B₀ B₁ : R) (T : IsCHSHTuple A₀ A₁ B₀ B₁) :
    A₀ * B₀ + A₀ * B₁ + A₁ * B₀ - A₁ * B₁ ≤ √2 ^ 3 • (1 : R) := by
  -- abel will create `ℤ` multiplication. We will `simp` them away to `ℝ` multiplication.
  have M : ∀ (m : ℤ) (a : ℝ) (x : R), m • a • x = ((m : ℝ) * a) • x := fun m a x => by
    rw [← Int.cast_smul_eq_zsmul ℝ, ← mul_smul]
  /-
    R : Type u
    inst✝⁵ : OrderedRing R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : Algebra Real R
    inst✝¹ : OrderedSMul Real R
    inst✝ : StarModule Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    M : ∀ (m : Int) (a : Real) (x : R), Eq (HSMul.hSMul m (HSMul.hSMul a x)) (HSMu …
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul A₀ B₀) (HMul.hMul A₀ B₁))  …
  -/
  let P := (√2)⁻¹ • (A₁ + A₀) - B₀
  /-
    R : Type u
    inst✝⁵ : OrderedRing R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : Algebra Real R
    inst✝¹ : OrderedSMul Real R
    inst✝ : StarModule Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    M : ∀ (m : Int) (a : Real) (x : R), Eq (HSMul.hSMul m (HSMul.hSMul a x)) (HSMu …
    P : R := HSub.hSub (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HAdd.hAdd A₁ A₀)) B₀
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul A₀ B₀) (HMul.hMul A₀ B₁))  …
  -/
  let Q := (√2)⁻¹ • (A₁ - A₀) + B₁
  have w : √2 ^ 3 • (1 : R) - A₀ * B₀ - A₀ * B₁ - A₁ * B₀ + A₁ * B₁ = (√2)⁻¹ • (P ^ 2 + Q ^ 2) := by
    dsimp [P, Q]
    -- distribute out all the powers and products appearing on the RHS
    simp only [sq, sub_mul, mul_sub, add_mul, mul_add, smul_add, smul_sub]
    -- pull all coefficients out to the front, and combine `√2`s where possible
    simp only [Algebra.mul_smul_comm, Algebra.smul_mul_assoc, ← mul_smul, sqrt_two_inv_mul_self]
    -- replace Aᵢ * Aᵢ = 1 and Bᵢ * Bᵢ = 1
    simp only [← sq, T.A₀_inv, T.A₁_inv, T.B₀_inv, T.B₁_inv]
    -- move Aᵢ to the left of Bᵢ
    simp only [← T.A₀B₀_commutes, ← T.A₀B₁_commutes, ← T.A₁B₀_commutes, ← T.A₁B₁_commutes]
    -- collect terms, simplify coefficients, and collect terms again:
    abel_nf
    -- all terms coincide, but the last one. Simplify all other terms
    simp only [M]
    simp only [neg_mul, one_mul, mul_inv_cancel_of_invertible, Int.cast_one, add_assoc, add_comm,
      add_left_comm, one_smul, Int.cast_neg, neg_smul, Int.cast_ofNat]
    simp only [← add_assoc, ← add_smul]
    -- just look at the coefficients now:
    congr
    exact mul_left_cancel₀ (by norm_num) tsirelson_inequality_aux
  have pos : 0 ≤ (√2)⁻¹ • (P ^ 2 + Q ^ 2) := by
    have P_sa : star P = P := by
      simp only [P, star_smul, star_add, star_sub, star_id_of_comm, T.A₀_sa, T.A₁_sa, T.B₀_sa,
        T.B₁_sa]
    have Q_sa : star Q = Q := by
      simp only [Q, star_smul, star_add, star_sub, star_id_of_comm, T.A₀_sa, T.A₁_sa, T.B₀_sa,
        T.B₁_sa]
    have P2_nonneg : 0 ≤ P ^ 2 := by simpa only [P_sa, sq] using star_mul_self_nonneg P
    have Q2_nonneg : 0 ≤ Q ^ 2 := by simpa only [Q_sa, sq] using star_mul_self_nonneg Q
    exact smul_nonneg (by positivity) (add_nonneg P2_nonneg Q2_nonneg)
  /-
    R : Type u
    inst✝⁵ : OrderedRing R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : Algebra Real R
    inst✝¹ : OrderedSMul Real R
    inst✝ : StarModule Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    M : ∀ (m : Int) (a : Real) (x : R), Eq (HSMul.hSMul m (HSMul.hSMul a x)) (HSMu …
    P : R := HSub.hSub (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HAdd.hAdd A₁ A₀)) B₀
    Q : R := HAdd.hAdd (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HSub.hSub A₁ A₀)) B₁
    w : Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub (HSMul.hSMul (HPow.hPow (Re …
    pos : LE.le 0 (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HAdd.hAdd (HPow.hPow P 2)  …
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul A₀ B₀) (HMul.hMul A₀ B₁))  …
  -/
  apply le_of_sub_nonneg
  /-
    case a
    R : Type u
    inst✝⁵ : OrderedRing R
    inst✝⁴ : StarRing R
    inst✝³ : StarOrderedRing R
    inst✝² : Algebra Real R
    inst✝¹ : OrderedSMul Real R
    inst✝ : StarModule Real R
    A₀ A₁ B₀ B₁ : R
    T : IsCHSHTuple A₀ A₁ B₀ B₁
    M : ∀ (m : Int) (a : Real) (x : R), Eq (HSMul.hSMul m (HSMul.hSMul a x)) (HSMu …
    P : R := HSub.hSub (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HAdd.hAdd A₁ A₀)) B₀
    Q : R := HAdd.hAdd (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HSub.hSub A₁ A₀)) B₁
    w : Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub (HSMul.hSMul (HPow.hPow (Re …
    pos : LE.le 0 (HSMul.hSMul (Inv.inv (Real.sqrt 2)) (HAdd.hAdd (HPow.hPow P 2)  …
    ⊢ LE.le 0 (HSub.hSub (HSMul.hSMul (HPow.hPow (Real.sqrt 2) 3) 1) (HSub.hSub (H …
  -/
  simpa only [sub_add_eq_sub_sub, ← sub_add, w, Nat.cast_zero] using pos
  /-
    🎉 no goals
  -/

