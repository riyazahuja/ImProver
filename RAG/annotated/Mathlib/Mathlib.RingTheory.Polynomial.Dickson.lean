/-- `dickson` is the `n`-th (generalised) Dickson polynomial of the `k`-th kind associated to the
element `a ∈ R`. -/
noncomputable def dickson : ℕ → R[X]
  | 0 => 3 - k
  | 1 => X
  | n + 2 => X * dickson (n + 1) - C a * dickson n


@[simp]
theorem dickson_zero : dickson k a 0 = 3 - k :=
  rfl


@[simp]
theorem dickson_one : dickson k a 1 = X :=
  rfl


theorem dickson_two : dickson k a 2 = X ^ 2 - C a * (3 - k : R[X]) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    a : R
    ⊢ Eq (Polynomial.dickson k a 2) (HSub.hSub (HPow.hPow Polynomial.X 2) (HMul.hM …
  -/
  simp only [dickson, sq]
  /-
    🎉 no goals
  -/


@[simp]
theorem dickson_add_two (n : ℕ) :
                                                                              /-
                                                                                R : Type u_1
                                                                                inst✝ : CommRing R
                                                                                k : Nat
                                                                                a : R
                                                                                n : Nat
                                                                                ⊢ Eq (Polynomial.dickson k a (HAdd.hAdd n 2)) (HSub.hSub (HMul.hMul Polynomial …
                                                                              -/
    dickson k a (n + 2) = X * dickson k a (n + 1) - C a * dickson k a n := by rw [dickson]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem dickson_of_two_le {n : ℕ} (h : 2 ≤ n) :
    dickson k a n = X * dickson k a (n - 1) - C a * dickson k a (n - 2) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    a : R
    n : Nat
    h : LE.le 2 n
    ⊢ Eq (Polynomial.dickson k a n) (HSub.hSub (HMul.hMul Polynomial.X (Polynomial …
  -/
  obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le h
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    a : R
    n : Nat
    h : LE.le 2 (HAdd.hAdd 2 n)
    ⊢ Eq (Polynomial.dickson k a (HAdd.hAdd 2 n)) (HSub.hSub (HMul.hMul Polynomial …
  -/
  rw [add_comm]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    k : Nat
    a : R
    n : Nat
    h : LE.le 2 (HAdd.hAdd 2 n)
    ⊢ Eq (Polynomial.dickson k a (HAdd.hAdd n 2)) (HSub.hSub (HMul.hMul Polynomial …
  -/
  exact dickson_add_two k a n
  /-
    🎉 no goals
  -/


theorem map_dickson (f : R →+* S) : ∀ n : ℕ, map f (dickson k a n) = dickson k (f a) n
  | 0 => by
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      k : Nat
      a : R
      f : RingHom R S
      ⊢ Eq (Polynomial.map f (Polynomial.dickson k a 0)) (Polynomial.dickson k (f a) …
    -/
    simp_rw [dickson_zero, Polynomial.map_sub, Polynomial.map_natCast, Polynomial.map_ofNat]
    /-
      🎉 no goals
    -/
            /-
              R : Type u_1
              S : Type u_2
              inst✝¹ : CommRing R
              inst✝ : CommRing S
              k : Nat
              a : R
              f : RingHom R S
              ⊢ Eq (Polynomial.map f (Polynomial.dickson k a 1)) (Polynomial.dickson k (f a) …
            -/
  | 1 => by simp only [dickson_one, map_X]
            /-
              🎉 no goals
            -/
  | n + 2 => by
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      k : Nat
      a : R
      f : RingHom R S
      n : Nat
      ⊢ Eq (Polynomial.map f (Polynomial.dickson k a (HAdd.hAdd n 2))) (Polynomial.d …
    -/
    simp only [dickson_add_two, Polynomial.map_sub, Polynomial.map_mul, map_X, map_C]
    /-
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      k : Nat
      a : R
      f : RingHom R S
      n : Nat
      ⊢ Eq (HSub.hSub (HMul.hMul Polynomial.X (Polynomial.map f (Polynomial.dickson  …
    -/
    rw [map_dickson f n, map_dickson f (n + 1)]
    /-
      🎉 no goals
    -/


@[simp]
theorem dickson_two_zero : ∀ n : ℕ, dickson 2 (0 : R) n = X ^ n
  | 0 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (Polynomial.dickson 2 0 0) (HPow.hPow Polynomial.X 0)
    -/
    simp only [dickson_zero, pow_zero]
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (HSub.hSub 3 ↑2) 1
    -/
    norm_num
    /-
      🎉 no goals
    -/
            /-
              R : Type u_1
              inst✝ : CommRing R
              ⊢ Eq (Polynomial.dickson 2 0 1) (HPow.hPow Polynomial.X 1)
            -/
  | 1 => by simp only [dickson_one, pow_one]
            /-
              🎉 no goals
            -/
  | n + 2 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ Eq (Polynomial.dickson 2 0 (HAdd.hAdd n 2)) (HPow.hPow Polynomial.X (HAdd.hA …
    -/
    simp only [dickson_add_two, C_0, zero_mul, sub_zero]
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ Eq (HMul.hMul Polynomial.X (Polynomial.dickson 2 0 (HAdd.hAdd n 1))) (HPow.h …
    -/
    rw [dickson_two_zero (n + 1), pow_add X (n + 1) 1, mul_comm, pow_one]
    /-
      🎉 no goals
    -/


theorem dickson_one_one_eval_add_inv (x y : R) (h : x * y = 1) :
    ∀ n, (dickson 1 (1 : R) n).eval (x + y) = x ^ n + y ^ n
  | 0 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      x y : R
      h : Eq (HMul.hMul x y) 1
      ⊢ Eq (Polynomial.eval (HAdd.hAdd x y) (Polynomial.dickson 1 1 0)) (HAdd.hAdd ( …
    -/
    simp only [eval_one, eval_add, pow_zero, dickson_zero]; norm_num
                                                            /-
                                                              🎉 no goals
                                                            -/
            /-
              R : Type u_1
              inst✝ : CommRing R
              x y : R
              h : Eq (HMul.hMul x y) 1
              ⊢ Eq (Polynomial.eval (HAdd.hAdd x y) (Polynomial.dickson 1 1 1)) (HAdd.hAdd ( …
            -/
  | 1 => by simp only [eval_X, dickson_one, pow_one]
            /-
              🎉 no goals
            -/
  | n + 2 => by
    simp only [eval_sub, eval_mul, dickson_one_one_eval_add_inv x y h _, eval_X, dickson_add_two,
      C_1, eval_one]
    /-
      R : Type u_1
      inst✝ : CommRing R
      x y : R
      h : Eq (HMul.hMul x y) 1
      n : Nat
      ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd x y) (HAdd.hAdd (HPow.hPow x (HAdd.hAdd  …
    -/
    conv_lhs => simp only [pow_succ', add_mul, mul_add, h, ← mul_assoc, mul_comm y x, one_mul]
    /-
      R : Type u_1
      inst✝ : CommRing R
      x y : R
      h : Eq (HMul.hMul x y) 1
      n : Nat
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul x x) (HPow.hPow x  …
    -/
    ring
    /-
      🎉 no goals
    -/


private theorem two_mul_C_half_eq_one [Invertible (2 : R)] : 2 * C (⅟ 2 : R) = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible 2
    ⊢ Eq (HMul.hMul 2 (Polynomial.C (Invertible.invOf 2))) 1
  -/
  rw [two_mul, ← C_add, invOf_two_add_invOf_two, C_1]
  /-
    🎉 no goals
  -/


private theorem C_half_mul_two_eq_one [Invertible (2 : R)] : C (⅟ 2 : R) * 2 = 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Invertible 2
    ⊢ Eq (HMul.hMul (Polynomial.C (Invertible.invOf 2)) 2) 1
  -/
  rw [mul_comm, two_mul_C_half_eq_one]
  /-
    🎉 no goals
  -/


theorem dickson_one_one_eq_chebyshev_C : ∀ n, dickson 1 (1 : R) n = Chebyshev.C R n
  | 0 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (Polynomial.dickson 1 1 0) (Polynomial.Chebyshev.C R ↑0)
    -/
    simp only [Chebyshev.C_zero, mul_one, one_comp, dickson_zero]
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (HSub.hSub 3 ↑1) (Polynomial.Chebyshev.C R ↑0)
    -/
    norm_num
    /-
      🎉 no goals
    -/
  | 1 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (Polynomial.dickson 1 1 1) (Polynomial.Chebyshev.C R ↑1)
    -/
    rw [dickson_one, Nat.cast_one, Chebyshev.C_one]
    /-
      🎉 no goals
    -/
  | n + 2 => by
    rw [dickson_add_two, C_1, Nat.cast_add, Nat.cast_two, Chebyshev.C_add_two,
      dickson_one_one_eq_chebyshev_C (n + 1), dickson_one_one_eq_chebyshev_C n]
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ Eq (HSub.hSub (HMul.hMul Polynomial.X (Polynomial.Chebyshev.C R ↑(HAdd.hAdd  …
    -/
    push_cast
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ Eq (HSub.hSub (HMul.hMul Polynomial.X (Polynomial.Chebyshev.C R (HAdd.hAdd ( …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem dickson_one_one_eq_chebyshev_T [Invertible (2 : R)] (n : ℕ) :
    dickson 1 (1 : R) n = 2 * (Chebyshev.T R n).comp (C (⅟ 2) * X) :=
  (dickson_one_one_eq_chebyshev_C R n).trans (Chebyshev.C_eq_two_mul_T_comp_half_mul_X R n)


theorem chebyshev_T_eq_dickson_one_one [Invertible (2 : R)] (n : ℕ) :
    Chebyshev.T R n = C (⅟ 2) * (dickson 1 1 n).comp (2 * X) :=
  dickson_one_one_eq_chebyshev_C R n ▸ Chebyshev.T_eq_half_mul_C_comp_two_mul_X R n


theorem dickson_two_one_eq_chebyshev_S : ∀ n, dickson 2 (1 : R) n = Chebyshev.S R n
  | 0 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (Polynomial.dickson 2 1 0) (Polynomial.Chebyshev.S R ↑0)
    -/
    simp only [Chebyshev.S_zero, mul_one, one_comp, dickson_zero]
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (HSub.hSub 3 ↑2) (Polynomial.Chebyshev.S R ↑0)
    -/
    norm_num
    /-
      🎉 no goals
    -/
  | 1 => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ Eq (Polynomial.dickson 2 1 1) (Polynomial.Chebyshev.S R ↑1)
    -/
    rw [dickson_one, Nat.cast_one, Chebyshev.S_one]
    /-
      🎉 no goals
    -/
  | n + 2 => by
    rw [dickson_add_two, C_1, Nat.cast_add, Nat.cast_two, Chebyshev.S_add_two,
      dickson_two_one_eq_chebyshev_S (n + 1), dickson_two_one_eq_chebyshev_S n]
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ Eq (HSub.hSub (HMul.hMul Polynomial.X (Polynomial.Chebyshev.S R ↑(HAdd.hAdd  …
    -/
    push_cast
    /-
      R : Type u_1
      inst✝ : CommRing R
      n : Nat
      ⊢ Eq (HSub.hSub (HMul.hMul Polynomial.X (Polynomial.Chebyshev.S R (HAdd.hAdd ( …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem dickson_two_one_eq_chebyshev_U [Invertible (2 : R)] (n : ℕ) :
    dickson 2 (1 : R) n = (Chebyshev.U R n).comp (C (⅟ 2) * X) :=
  (dickson_two_one_eq_chebyshev_S R n).trans (Chebyshev.S_eq_U_comp_half_mul_X R n)


theorem chebyshev_U_eq_dickson_two_one (n : ℕ) :
    Chebyshev.U R n = (dickson 2 (1 : R) n).comp (2 * X) :=
  dickson_two_one_eq_chebyshev_S R n ▸ (Chebyshev.S_comp_two_mul_X R n).symm


/-- The `(m * n)`-th Dickson polynomial of the first kind is the composition of the `m`-th and
`n`-th. -/
theorem dickson_one_one_mul (m n : ℕ) :
    dickson 1 (1 : R) (m * n) = (dickson 1 1 m).comp (dickson 1 1 n) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    ⊢ Eq (Polynomial.dickson 1 1 (HMul.hMul m n)) ((Polynomial.dickson 1 1 m).comp …
  -/
  have h : (1 : R) = Int.castRingHom R 1 := by simp only [eq_intCast, Int.cast_one]
  /-
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (Polynomial.dickson 1 1 (HMul.hMul m n)) ((Polynomial.dickson 1 1 m).comp …
  -/
  rw [h]
  /-
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (Polynomial.dickson 1 ((Int.castRingHom R) 1) (HMul.hMul m n)) ((Polynomi …
  -/
  simp only [← map_dickson (Int.castRingHom R), ← map_comp]
  /-
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (Polynomial.map (Int.castRingHom R) (Polynomial.dickson 1 1 (HMul.hMul m  …
  -/
  congr 1
  /-
    case e_a
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (Polynomial.dickson 1 1 (HMul.hMul m n)) ((Polynomial.dickson 1 1 m).comp …
  -/
  apply map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [map_dickson, map_comp, eq_intCast, Int.cast_one, dickson_one_one_eq_chebyshev_T,
    Nat.cast_mul, Chebyshev.T_mul, two_mul, ← add_comp]
  /-
    case e_a.a
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (((HAdd.hAdd (Polynomial.Chebyshev.T Rat ↑m) (Polynomial.Chebyshev.T Rat  …
  -/
  simp only [← two_mul, ← comp_assoc]
  /-
    case e_a.a
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (((HMul.hMul 2 (Polynomial.Chebyshev.T Rat ↑m)).comp (Polynomial.Chebyshe …
  -/
  apply eval₂_congr rfl rfl
  /-
    case e_a.a
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq ((HMul.hMul 2 (Polynomial.Chebyshev.T Rat ↑m)).comp (Polynomial.Chebyshev …
  -/
  rw [comp_assoc]
  /-
    case e_a.a
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq ((HMul.hMul 2 (Polynomial.Chebyshev.T Rat ↑m)).comp (Polynomial.Chebyshev …
  -/
  apply eval₂_congr rfl _ rfl
  /-
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    h : Eq 1 ((Int.castRingHom R) 1)
    ⊢ Eq (Polynomial.Chebyshev.T Rat ↑n) ((HMul.hMul (Polynomial.C (Invertible.inv …
  -/
  rw [mul_comp, C_comp, X_comp, ← mul_assoc, C_half_mul_two_eq_one, one_mul]
  /-
    🎉 no goals
  -/


theorem dickson_one_one_comp_comm (m n : ℕ) :
    (dickson 1 (1 : R) m).comp (dickson 1 1 n) = (dickson 1 1 n).comp (dickson 1 1 m) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    m n : Nat
    ⊢ Eq ((Polynomial.dickson 1 1 m).comp (Polynomial.dickson 1 1 n)) ((Polynomial …
  -/
  rw [← dickson_one_one_mul, mul_comm, dickson_one_one_mul]
  /-
    🎉 no goals
  -/


theorem dickson_one_one_zmod_p (p : ℕ) [Fact p.Prime] : dickson 1 (1 : ZMod p) p = X ^ p := by
  -- Recall that `dickson_one_one_eval_add_inv` characterises `dickson 1 1 p`
  -- as a polynomial that maps `x + x⁻¹` to `x ^ p + (x⁻¹) ^ p`.
  -- Since `X ^ p` also satisfies this property in characteristic `p`,
  -- we can use a variant on `Polynomial.funext` to conclude that these polynomials are equal.
  -- For this argument, we need an arbitrary infinite field of characteristic `p`.
  obtain ⟨K, _, _, H⟩ : ∃ (K : Type) (_ : Field K), ∃ _ : CharP K p, Infinite K := by
    let K := FractionRing (Polynomial (ZMod p))
    let f : ZMod p →+* K := (algebraMap _ (FractionRing _)).comp C
    have : CharP K p := by
      rw [← f.charP_iff_charP]
      infer_instance
    haveI : Infinite K :=
      Infinite.of_injective (algebraMap (Polynomial (ZMod p)) (FractionRing (Polynomial (ZMod p))))
        (IsFractionRing.injective _ _)
    refine ⟨K, ?_, ?_, ?_⟩ <;> infer_instance
  /-
    case intro.intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    K : Type
    w✝¹ : Field K
    w✝ : CharP K p
    H : Infinite K
    ⊢ Eq (Polynomial.dickson 1 1 p) (HPow.hPow Polynomial.X p)
  -/
  apply map_injective (ZMod.castHom (dvd_refl p) K) (RingHom.injective _)
  /-
    case intro.intro.intro.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    K : Type
    w✝¹ : Field K
    w✝ : CharP K p
    H : Infinite K
    ⊢ Eq (Polynomial.map (ZMod.castHom ⋯ K) (Polynomial.dickson 1 1 p)) (Polynomia …
  -/
  rw [map_dickson, Polynomial.map_pow, map_X]
  /-
    case intro.intro.intro.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    K : Type
    w✝¹ : Field K
    w✝ : CharP K p
    H : Infinite K
    ⊢ Eq (Polynomial.dickson 1 ((ZMod.castHom ⋯ K) 1) p) (HPow.hPow Polynomial.X p)
  -/
  apply eq_of_infinite_eval_eq
  -- The two polynomials agree on all `x` of the form `x = y + y⁻¹`.
  /-
    case intro.intro.intro.a.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    K : Type
    w✝¹ : Field K
    w✝ : CharP K p
    H : Infinite K
    ⊢ (setOf fun x => Eq (Polynomial.eval x (Polynomial.dickson 1 ((ZMod.castHom ⋯ …
  -/
  apply @Set.Infinite.mono _ { x : K | ∃ y, x = y + y⁻¹ ∧ y ≠ 0 }
    /-
      case intro.intro.intro.a.h.h
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Infinite K
      ⊢ HasSubset.Subset (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv …
    -/
  · rintro _ ⟨x, rfl, hx⟩
    simp only [eval_X, eval_pow, Set.mem_setOf_eq, ZMod.cast_one', add_pow_char,
      dickson_one_one_eval_add_inv _ _ (mul_inv_cancel₀ hx), ZMod.castHom_apply]
  -- Now we need to show that the set of such `x` is infinite.
  -- If the set is finite, then we will show that `K` is also finite.
    /-
      case intro.intro.intro.a.h.a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Infinite K
      ⊢ (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y 0 …
    -/
  · intro h
    /-
      case intro.intro.intro.a.h.a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Infinite K
      h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
      ⊢ False
    -/
    rw [← Set.infinite_univ_iff] at H
    /-
      case intro.intro.intro.a.h.a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Set.univ.Infinite
      h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
      ⊢ False
    -/
    apply H
    -- To each `x` of the form `x = y + y⁻¹`
    -- we `bind` the set of `y` that solve the equation `x = y + y⁻¹`.
    -- For every `x`, that set is finite (since it is governed by a quadratic equation).
    -- For the moment, we claim that all these sets together cover `K`.
    suffices (Set.univ : Set K) =
        ⋃ x ∈ { x : K | ∃ y : K, x = y + y⁻¹ ∧ y ≠ 0 }, { y | x = y + y⁻¹ ∨ y = 0 }  by
      rw [this]
      clear this
      refine h.biUnion fun x _ => ?_
      -- The following quadratic polynomial has as solutions the `y` for which `x = y + y⁻¹`.
      let φ : K[X] := X ^ 2 - C x * X + 1
      have hφ : φ ≠ 0 := by
        intro H
        have : φ.eval 0 = 0 := by rw [H, eval_zero]
        simpa [φ, eval_X, eval_one, eval_pow, eval_sub, sub_zero, eval_add, eval_mul,
          mul_zero, sq, zero_add, one_ne_zero]
      classical
        convert (φ.roots ∪ {0}).toFinset.finite_toSet using 1
        ext1 y
        simp only [φ, Multiset.mem_toFinset, Set.mem_setOf_eq, Finset.mem_coe, Multiset.mem_union,
          mem_roots hφ, IsRoot, eval_add, eval_sub, eval_pow, eval_mul, eval_X, eval_C, eval_one,
          Multiset.mem_singleton]
        by_cases hy : y = 0
        · simp only [hy, eq_self_iff_true, or_true]
        apply or_congr _ Iff.rfl
        rw [← mul_left_inj' hy, eq_comm, ← sub_eq_zero, add_mul, inv_mul_cancel₀ hy]
        apply eq_iff_eq_cancel_right.mpr
        ring
    -- Finally, we prove the claim that our finite union of finite sets covers all of `K`.
    /-
      case intro.intro.intro.a.h.a
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Set.univ.Infinite
      h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
      ⊢ Eq Set.univ (Set.iUnion fun x => Set.iUnion fun h => setOf fun y => Or (Eq x …
    -/
    apply (Set.eq_univ_of_forall _).symm
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Set.univ.Infinite
      h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
      ⊢ ∀ (x : K), Membership.mem (Set.iUnion fun x => Set.iUnion fun h => setOf fun …
    -/
    intro x
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Set.univ.Infinite
      h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
      x : K
      ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => setOf fun y => Or (E …
    -/
    simp only [exists_prop, Set.mem_iUnion, Ne, Set.mem_setOf_eq]
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      K : Type
      w✝¹ : Field K
      w✝ : CharP K p
      H : Set.univ.Infinite
      h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
      x : K
      ⊢ Exists fun i => And (Exists fun y => And (Eq i (HAdd.hAdd y (Inv.inv y))) (N …
    -/
    by_cases hx : x = 0
      /-
        case pos
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        K : Type
        w✝¹ : Field K
        w✝ : CharP K p
        H : Set.univ.Infinite
        h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
        x : K
        hx : Eq x 0
        ⊢ Exists fun i => And (Exists fun y => And (Eq i (HAdd.hAdd y (Inv.inv y))) (N …
      -/
    · simp only [hx, and_true, eq_self_iff_true, inv_zero, or_true]
      /-
        case pos
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        K : Type
        w✝¹ : Field K
        w✝ : CharP K p
        H : Set.univ.Infinite
        h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
        x : K
        hx : Eq x 0
        ⊢ Exists fun i => Exists fun y => And (Eq i (HAdd.hAdd y (Inv.inv y))) (Not (E …
      -/
      exact ⟨_, 1, rfl, one_ne_zero⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        K : Type
        w✝¹ : Field K
        w✝ : CharP K p
        H : Set.univ.Infinite
        h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
        x : K
        hx : Not (Eq x 0)
        ⊢ Exists fun i => And (Exists fun y => And (Eq i (HAdd.hAdd y (Inv.inv y))) (N …
      -/
    · simp only [hx, or_false, exists_eq_right]
      /-
        case neg
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        K : Type
        w✝¹ : Field K
        w✝ : CharP K p
        H : Set.univ.Infinite
        h : (setOf fun x => Exists fun y => And (Eq x (HAdd.hAdd y (Inv.inv y))) (Ne y …
        x : K
        hx : Not (Eq x 0)
        ⊢ Exists fun y => And (Eq (HAdd.hAdd x (Inv.inv x)) (HAdd.hAdd y (Inv.inv y))) …
      -/
      exact ⟨_, rfl, hx⟩
      /-
        🎉 no goals
      -/


theorem dickson_one_one_charP (p : ℕ) [Fact p.Prime] [CharP R p] : dickson 1 (1 : R) p = X ^ p := by
  have h : (1 : R) = ZMod.castHom (dvd_refl p) R 1 := by
    simp only [ZMod.castHom_apply, ZMod.cast_one']
  rw [h, ← map_dickson (ZMod.castHom (dvd_refl p) R), dickson_one_one_zmod_p, Polynomial.map_pow,
    map_X]


