theorem sq_ne_two_fin_zmod_four (z : ZMod 4) : z * z ≠ 2 := by
  /-
    z : ZMod 4
    ⊢ Ne (HMul.hMul z z) 2
  -/
  change Fin 4 at z
  /-
    z : Fin 4
    ⊢ Ne (HMul.hMul z z) 2
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases z <;> decide
                  /-
                    🎉 no goals
                  -/


theorem Int.sq_ne_two_mod_four (z : ℤ) : z * z % 4 ≠ 2 := by
  /-
    z : Int
    ⊢ Ne (HMod.hMod (HMul.hMul z z) 4) 2
  -/
  suffices ¬z * z % (4 : ℕ) = 2 % (4 : ℕ) by exact this
  /-
    z : Int
    ⊢ Not (Eq (HMod.hMod (HMul.hMul z z) ↑4) (HMod.hMod 2 ↑4))
  -/
  rw [← ZMod.intCast_eq_intCast_iff']
  /-
    z : Int
    ⊢ Not (Eq ↑(HMul.hMul z z) ↑2)
  -/
  simpa using sq_ne_two_fin_zmod_four _
  /-
    🎉 no goals
  -/


/-- Three integers `x`, `y`, and `z` form a Pythagorean triple if `x * x + y * y = z * z`. -/
def PythagoreanTriple (x y z : ℤ) : Prop :=
  x * x + y * y = z * z


/-- Pythagorean triples are interchangeable, i.e `x * x + y * y = y * y + x * x = z * z`.
This comes from additive commutativity. -/
theorem pythagoreanTriple_comm {x y z : ℤ} : PythagoreanTriple x y z ↔ PythagoreanTriple y x z := by
  /-
    x y z : Int
    ⊢ Iff (PythagoreanTriple x y z) (PythagoreanTriple y x z)
  -/
  delta PythagoreanTriple
  /-
    x y z : Int
    ⊢ Iff (Eq (HAdd.hAdd (HMul.hMul x x) (HMul.hMul y y)) (HMul.hMul z z)) (Eq (HA …
  -/
  rw [add_comm]
  /-
    🎉 no goals
  -/


/-- The zeroth Pythagorean triple is all zeros. -/
theorem PythagoreanTriple.zero : PythagoreanTriple 0 0 0 := by
  /-
    ⊢ PythagoreanTriple 0 0 0
  -/
  simp only [PythagoreanTriple, zero_mul, zero_add]
  /-
    🎉 no goals
  -/


theorem eq (h : PythagoreanTriple x y z) : x * x + y * y = z * z :=
  h


@[symm]
theorem symm (h : PythagoreanTriple x y z) : PythagoreanTriple y x z := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    ⊢ PythagoreanTriple y x z
  -/
  rwa [pythagoreanTriple_comm]
  /-
    🎉 no goals
  -/


/-- A triple is still a triple if you multiply `x`, `y` and `z`
by a constant `k`. -/
theorem mul (h : PythagoreanTriple x y z) (k : ℤ) : PythagoreanTriple (k * x) (k * y) (k * z) :=
  calc
                                                                      /-
                                                                        x y z : Int
                                                                        h : PythagoreanTriple x y z
                                                                        k : Int
                                                                        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul k x) (HMul.hMul k x)) (HMul.hMul (HMul.h …
                                                                      -/
    k * x * (k * x) + k * y * (k * y) = k ^ 2 * (x * x + y * y) := by ring
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                              /-
                                x y z : Int
                                h : PythagoreanTriple x y z
                                k : Int
                                ⊢ Eq (HMul.hMul (HPow.hPow k 2) (HAdd.hAdd (HMul.hMul x x) (HMul.hMul y y))) ( …
                              -/
    _ = k ^ 2 * (z * z) := by rw [h.eq]
                              /-
                                🎉 no goals
                              -/
                              /-
                                x y z : Int
                                h : PythagoreanTriple x y z
                                k : Int
                                ⊢ Eq (HMul.hMul (HPow.hPow k 2) (HMul.hMul z z)) (HMul.hMul (HMul.hMul k z) (H …
                              -/
    _ = k * z * (k * z) := by ring
                              /-
                                🎉 no goals
                              -/


/-- `(k*x, k*y, k*z)` is a Pythagorean triple if and only if
`(x, y, z)` is also a triple. -/
theorem mul_iff (k : ℤ) (hk : k ≠ 0) :
    PythagoreanTriple (k * x) (k * y) (k * z) ↔ PythagoreanTriple x y z := by
  /-
    x y z k : Int
    hk : Ne k 0
    ⊢ Iff (PythagoreanTriple (HMul.hMul k x) (HMul.hMul k y) (HMul.hMul k z)) (Pyt …
  -/
  refine ⟨?_, fun h => h.mul k⟩
  /-
    x y z k : Int
    hk : Ne k 0
    ⊢ PythagoreanTriple (HMul.hMul k x) (HMul.hMul k y) (HMul.hMul k z) → Pythagor …
  -/
  simp only [PythagoreanTriple]
  /-
    x y z k : Int
    hk : Ne k 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul k x) (HMul.hMul k x)) (HMul.hMul (HMul.h …
  -/
  intro h
  /-
    x y z k : Int
    hk : Ne k 0
    h : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul k x) (HMul.hMul k x)) (HMul.hMul (HMul …
    ⊢ Eq (HAdd.hAdd (HMul.hMul x x) (HMul.hMul y y)) (HMul.hMul z z)
  -/
  rw [← mul_left_inj' (mul_ne_zero hk hk)]
  /-
    x y z k : Int
    hk : Ne k 0
    h : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul k x) (HMul.hMul k x)) (HMul.hMul (HMul …
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul x x) (HMul.hMul y y)) (HMul.hMul k k)) ( …
  -/
                        /-
                          🎉 no goals
                        -/
  convert h using 1 <;> ring
                        /-
                          🎉 no goals
                        -/


/-- A Pythagorean triple `x, y, z` is “classified” if there exist integers `k, m, n` such that
either
 * `x = k * (m ^ 2 - n ^ 2)` and `y = k * (2 * m * n)`, or
 * `x = k * (2 * m * n)` and `y = k * (m ^ 2 - n ^ 2)`. -/
@[nolint unusedArguments]
def IsClassified (_ : PythagoreanTriple x y z) :=
  ∃ k m n : ℤ,
    (x = k * (m ^ 2 - n ^ 2) ∧ y = k * (2 * m * n) ∨
        x = k * (2 * m * n) ∧ y = k * (m ^ 2 - n ^ 2)) ∧
      Int.gcd m n = 1


/-- A primitive Pythagorean triple `x, y, z` is a Pythagorean triple with `x` and `y` coprime.
 Such a triple is “primitively classified” if there exist coprime integers `m, n` such that either
 * `x = m ^ 2 - n ^ 2` and `y = 2 * m * n`, or
 * `x = 2 * m * n` and `y = m ^ 2 - n ^ 2`.
-/
@[nolint unusedArguments]
def IsPrimitiveClassified (_ : PythagoreanTriple x y z) :=
  ∃ m n : ℤ,
    (x = m ^ 2 - n ^ 2 ∧ y = 2 * m * n ∨ x = 2 * m * n ∧ y = m ^ 2 - n ^ 2) ∧
      Int.gcd m n = 1 ∧ (m % 2 = 0 ∧ n % 2 = 1 ∨ m % 2 = 1 ∧ n % 2 = 0)


theorem mul_isClassified (k : ℤ) (hc : h.IsClassified) : (h.mul k).IsClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    k : Int
    hc : h.IsClassified
    ⊢ ⋯.IsClassified
  -/
  obtain ⟨l, m, n, ⟨⟨rfl, rfl⟩ | ⟨rfl, rfl⟩, co⟩⟩ := hc
    /-
      case intro.intro.intro.intro.inl.intro
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
      ⊢ ⋯.IsClassified
    -/
  · use k * l, m, n
    /-
      case h
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
      ⊢ And (Or (And (Eq (HMul.hMul k (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow. …
    -/
    apply And.intro _ co
    /-
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
      ⊢ Or (And (Eq (HMul.hMul k (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow.hPow  …
    -/
    left
    /-
      case h
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
      ⊢ And (Eq (HMul.hMul k (HMul.hMul l (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2) …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> ring
                    /-
                      🎉 no goals
                    -/
    /-
      case intro.intro.intro.intro.inr.intro
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul l …
      ⊢ ⋯.IsClassified
    -/
  · use k * l, m, n
    /-
      case h
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul l …
      ⊢ And (Or (And (Eq (HMul.hMul k (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n))) ( …
    -/
    apply And.intro _ co
    /-
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul l …
      ⊢ Or (And (Eq (HMul.hMul k (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n))) (HMul. …
    -/
    right
    /-
      case h
      z k l m n : Int
      co : Eq (m.gcd n) 1
      h : PythagoreanTriple (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul l …
      ⊢ And (Eq (HMul.hMul k (HMul.hMul l (HMul.hMul (HMul.hMul 2 m) n))) (HMul.hMul …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> ring
                    /-
                      🎉 no goals
                    -/


theorem even_odd_of_coprime (hc : Int.gcd x y = 1) :
    x % 2 = 0 ∧ y % 2 = 1 ∨ x % 2 = 1 ∧ y % 2 = 0 := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    ⊢ Or (And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)) (And (Eq (HMod.hMod x …
  -/
  cases' Int.emod_two_eq_zero_or_one x with hx hx <;>
    /-
      case inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 0
      ⊢ Or (And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)) (And (Eq (HMod.hMod x …
    -/
    cases' Int.emod_two_eq_zero_or_one y with hy hy
  -- x even, y even
    /-
      case inl.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 0
      hy : Eq (HMod.hMod y 2) 0
      ⊢ Or (And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)) (And (Eq (HMod.hMod x …
    -/
  · exfalso
    /-
      case inl.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 0
      hy : Eq (HMod.hMod y 2) 0
      ⊢ False
    -/
    apply Nat.not_coprime_of_dvd_of_dvd (by decide : 1 < 2) _ _ hc
      /-
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hx : Eq (HMod.hMod x 2) 0
        hy : Eq (HMod.hMod y 2) 0
        ⊢ Dvd.dvd 2 x.natAbs
      -/
    · apply Int.natCast_dvd.1
      /-
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hx : Eq (HMod.hMod x 2) 0
        hy : Eq (HMod.hMod y 2) 0
        ⊢ Dvd.dvd (↑2) x
      -/
      apply Int.dvd_of_emod_eq_zero hx
      /-
        🎉 no goals
      -/
      /-
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hx : Eq (HMod.hMod x 2) 0
        hy : Eq (HMod.hMod y 2) 0
        ⊢ Dvd.dvd 2 y.natAbs
      -/
    · apply Int.natCast_dvd.1
      /-
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hx : Eq (HMod.hMod x 2) 0
        hy : Eq (HMod.hMod y 2) 0
        ⊢ Dvd.dvd (↑2) y
      -/
      apply Int.dvd_of_emod_eq_zero hy
      /-
        🎉 no goals
      -/
  -- x even, y odd
    /-
      case inl.inr
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 0
      hy : Eq (HMod.hMod y 2) 1
      ⊢ Or (And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)) (And (Eq (HMod.hMod x …
    -/
  · left
    /-
      case inl.inr.h
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 0
      hy : Eq (HMod.hMod y 2) 1
      ⊢ And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)
    -/
    exact ⟨hx, hy⟩
    /-
      🎉 no goals
    -/
  -- x odd, y even
    /-
      case inr.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 1
      hy : Eq (HMod.hMod y 2) 0
      ⊢ Or (And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)) (And (Eq (HMod.hMod x …
    -/
  · right
    /-
      case inr.inl.h
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 1
      hy : Eq (HMod.hMod y 2) 0
      ⊢ And (Eq (HMod.hMod x 2) 1) (Eq (HMod.hMod y 2) 0)
    -/
    exact ⟨hx, hy⟩
    /-
      🎉 no goals
    -/
  -- x odd, y odd
    /-
      case inr.inr
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hx : Eq (HMod.hMod x 2) 1
      hy : Eq (HMod.hMod y 2) 1
      ⊢ Or (And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)) (And (Eq (HMod.hMod x …
    -/
  · exfalso
    obtain ⟨x0, y0, rfl, rfl⟩ : ∃ x0 y0, x = x0 * 2 + 1 ∧ y = y0 * 2 + 1 := by
      cases' exists_eq_mul_left_of_dvd (Int.dvd_sub_of_emod_eq hx) with x0 hx2
      cases' exists_eq_mul_left_of_dvd (Int.dvd_sub_of_emod_eq hy) with y0 hy2
      rw [sub_eq_iff_eq_add] at hx2 hy2
      exact ⟨x0, y0, hx2, hy2⟩
    /-
      case inr.inr.intro.intro.intro
      z x0 y0 : Int
      hx : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul x0 2) 1) 2) 1
      hy : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul y0 2) 1) 2) 1
      h : PythagoreanTriple (HAdd.hAdd (HMul.hMul x0 2) 1) (HAdd.hAdd (HMul.hMul y0  …
      hc : Eq ((HAdd.hAdd (HMul.hMul x0 2) 1).gcd (HAdd.hAdd (HMul.hMul y0 2) 1)) 1
      ⊢ False
    -/
    apply Int.sq_ne_two_mod_four z
    rw [show z * z = 4 * (x0 * x0 + x0 + y0 * y0 + y0) + 2 by
        rw [← h.eq]
        ring]
    /-
      case inr.inr.intro.intro.intro
      z x0 y0 : Int
      hx : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul x0 2) 1) 2) 1
      hy : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul y0 2) 1) 2) 1
      h : PythagoreanTriple (HAdd.hAdd (HMul.hMul x0 2) 1) (HAdd.hAdd (HMul.hMul y0  …
      hc : Eq ((HAdd.hAdd (HMul.hMul x0 2) 1).gcd (HAdd.hAdd (HMul.hMul y0 2) 1)) 1
      ⊢ Eq (HMod.hMod (HAdd.hAdd (HMul.hMul 4 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul …
    -/
    simp only [Int.add_emod, Int.mul_emod_right, zero_add]
    /-
      case inr.inr.intro.intro.intro
      z x0 y0 : Int
      hx : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul x0 2) 1) 2) 1
      hy : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul y0 2) 1) 2) 1
      h : PythagoreanTriple (HAdd.hAdd (HMul.hMul x0 2) 1) (HAdd.hAdd (HMul.hMul y0  …
      hc : Eq ((HAdd.hAdd (HMul.hMul x0 2) 1).gcd (HAdd.hAdd (HMul.hMul y0 2) 1)) 1
      ⊢ Eq (HMod.hMod (HMod.hMod 2 4) 4) 2
    -/
    decide
    /-
      🎉 no goals
    -/


theorem gcd_dvd : (Int.gcd x y : ℤ) ∣ z := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    ⊢ Dvd.dvd (↑(x.gcd y)) z
  -/
  by_cases h0 : Int.gcd x y = 0
  · have hx : x = 0 := by
      apply Int.natAbs_eq_zero.mp
      apply Nat.eq_zero_of_gcd_eq_zero_left h0
    have hy : y = 0 := by
      apply Int.natAbs_eq_zero.mp
      apply Nat.eq_zero_of_gcd_eq_zero_right h0
    have hz : z = 0 := by
      simpa only [PythagoreanTriple, hx, hy, add_zero, zero_eq_mul, mul_zero,
        or_self_iff] using h
    /-
      case pos
      x y z : Int
      h : PythagoreanTriple x y z
      h0 : Eq (x.gcd y) 0
      hx : Eq x 0
      hy : Eq y 0
      hz : Eq z 0
      ⊢ Dvd.dvd (↑(x.gcd y)) z
    -/
    simp only [hz, dvd_zero]
    /-
      🎉 no goals
    -/
  obtain ⟨k, x0, y0, _, h2, rfl, rfl⟩ :
    ∃ (k : ℕ) (x0 y0 : _), 0 < k ∧ Int.gcd x0 y0 = 1 ∧ x = x0 * k ∧ y = y0 * k :=
    Int.exists_gcd_one' (Nat.pos_of_ne_zero h0)
  /-
    case neg.intro.intro.intro.intro.intro.intro
    z : Int
    k : Nat
    x0 y0 : Int
    left✝ : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h : PythagoreanTriple (HMul.hMul x0 ↑k) (HMul.hMul y0 ↑k) z
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    ⊢ Dvd.dvd (↑((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k))) z
  -/
  rw [Int.gcd_mul_right, h2, Int.natAbs_ofNat, one_mul]
  /-
    case neg.intro.intro.intro.intro.intro.intro
    z : Int
    k : Nat
    x0 y0 : Int
    left✝ : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h : PythagoreanTriple (HMul.hMul x0 ↑k) (HMul.hMul y0 ↑k) z
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    ⊢ Dvd.dvd (↑k) z
  -/
  rw [← Int.pow_dvd_pow_iff two_ne_zero, sq z, ← h.eq]
  /-
    case neg.intro.intro.intro.intro.intro.intro
    z : Int
    k : Nat
    x0 y0 : Int
    left✝ : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h : PythagoreanTriple (HMul.hMul x0 ↑k) (HMul.hMul y0 ↑k) z
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    ⊢ Dvd.dvd (HPow.hPow (↑k) 2) (HAdd.hAdd (HMul.hMul (HMul.hMul x0 ↑k) (HMul.hMu …
  -/
  rw [(by ring : x0 * k * (x0 * k) + y0 * k * (y0 * k) = (k : ℤ) ^ 2 * (x0 * x0 + y0 * y0))]
  /-
    case neg.intro.intro.intro.intro.intro.intro
    z : Int
    k : Nat
    x0 y0 : Int
    left✝ : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h : PythagoreanTriple (HMul.hMul x0 ↑k) (HMul.hMul y0 ↑k) z
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    ⊢ Dvd.dvd (HPow.hPow (↑k) 2) (HMul.hMul (HPow.hPow (↑k) 2) (HAdd.hAdd (HMul.hM …
  -/
  exact dvd_mul_right _ _
  /-
    🎉 no goals
  -/


theorem normalize : PythagoreanTriple (x / Int.gcd x y) (y / Int.gcd x y) (z / Int.gcd x y) := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    ⊢ PythagoreanTriple (HDiv.hDiv x ↑(x.gcd y)) (HDiv.hDiv y ↑(x.gcd y)) (HDiv.hD …
  -/
  by_cases h0 : Int.gcd x y = 0
  · have hx : x = 0 := by
      apply Int.natAbs_eq_zero.mp
      apply Nat.eq_zero_of_gcd_eq_zero_left h0
    have hy : y = 0 := by
      apply Int.natAbs_eq_zero.mp
      apply Nat.eq_zero_of_gcd_eq_zero_right h0
    have hz : z = 0 := by
      simpa only [PythagoreanTriple, hx, hy, add_zero, zero_eq_mul, mul_zero,
        or_self_iff] using h
    /-
      case pos
      x y z : Int
      h : PythagoreanTriple x y z
      h0 : Eq (x.gcd y) 0
      hx : Eq x 0
      hy : Eq y 0
      hz : Eq z 0
      ⊢ PythagoreanTriple (HDiv.hDiv x ↑(x.gcd y)) (HDiv.hDiv y ↑(x.gcd y)) (HDiv.hD …
    -/
    simp only [hx, hy, hz]
    /-
      case pos
      x y z : Int
      h : PythagoreanTriple x y z
      h0 : Eq (x.gcd y) 0
      hx : Eq x 0
      hy : Eq y 0
      hz : Eq z 0
      ⊢ PythagoreanTriple (HDiv.hDiv 0 ↑(Int.gcd 0 0)) (HDiv.hDiv 0 ↑(Int.gcd 0 0))  …
    -/
    exact zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    h0 : Not (Eq (x.gcd y) 0)
    ⊢ PythagoreanTriple (HDiv.hDiv x ↑(x.gcd y)) (HDiv.hDiv y ↑(x.gcd y)) (HDiv.hD …
  -/
  rcases h.gcd_dvd with ⟨z0, rfl⟩
  obtain ⟨k, x0, y0, k0, h2, rfl, rfl⟩ :
    ∃ (k : ℕ) (x0 y0 : _), 0 < k ∧ Int.gcd x0 y0 = 1 ∧ x = x0 * k ∧ y = y0 * k :=
    Int.exists_gcd_one' (Nat.pos_of_ne_zero h0)
  have hk : (k : ℤ) ≠ 0 := by
    norm_cast
    rwa [pos_iff_ne_zero] at k0
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    z0 : Int
    k : Nat
    x0 y0 : Int
    k0 : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    h : PythagoreanTriple (HMul.hMul x0 ↑k) (HMul.hMul y0 ↑k) (HMul.hMul (↑((HMul. …
    hk : Ne (↑k) 0
    ⊢ PythagoreanTriple (HDiv.hDiv (HMul.hMul x0 ↑k) ↑((HMul.hMul x0 ↑k).gcd (HMul …
  -/
  rw [Int.gcd_mul_right, h2, Int.natAbs_ofNat, one_mul] at h ⊢
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    z0 : Int
    k : Nat
    x0 y0 : Int
    k0 : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    h : PythagoreanTriple (HMul.hMul x0 ↑k) (HMul.hMul y0 ↑k) (HMul.hMul (↑k) z0)
    hk : Ne (↑k) 0
    ⊢ PythagoreanTriple (HDiv.hDiv (HMul.hMul x0 ↑k) ↑k) (HDiv.hDiv (HMul.hMul y0  …
  -/
  rw [mul_comm x0, mul_comm y0, mul_iff k hk] at h
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    z0 : Int
    k : Nat
    x0 y0 : Int
    k0 : LT.lt 0 k
    h2 : Eq (x0.gcd y0) 1
    h0 : Not (Eq ((HMul.hMul x0 ↑k).gcd (HMul.hMul y0 ↑k)) 0)
    h : PythagoreanTriple x0 y0 z0
    hk : Ne (↑k) 0
    ⊢ PythagoreanTriple (HDiv.hDiv (HMul.hMul x0 ↑k) ↑k) (HDiv.hDiv (HMul.hMul y0  …
  -/
  rwa [Int.mul_ediv_cancel _ hk, Int.mul_ediv_cancel _ hk, Int.mul_ediv_cancel_left _ hk]
  /-
    🎉 no goals
  -/


theorem isClassified_of_isPrimitiveClassified (hp : h.IsPrimitiveClassified) : h.IsClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hp : h.IsPrimitiveClassified
    ⊢ h.IsClassified
  -/
  obtain ⟨m, n, H⟩ := hp
  /-
    case intro.intro
    x y z : Int
    h : PythagoreanTriple x y z
    m n : Int
    H : And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMu …
    ⊢ h.IsClassified
  -/
  use 1, m, n
  /-
    case h
    x y z : Int
    h : PythagoreanTriple x y z
    m n : Int
    H : And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMu …
    ⊢ And (Or (And (Eq x (HMul.hMul 1 (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem isClassified_of_normalize_isPrimitiveClassified (hc : h.normalize.IsPrimitiveClassified) :
    h.IsClassified := by
  convert h.normalize.mul_isClassified (Int.gcd x y)
        (isClassified_of_isPrimitiveClassified h.normalize hc) <;>
    /-
      case h.e'_1
      x y z : Int
      h : PythagoreanTriple x y z
      hc : ⋯.IsPrimitiveClassified
      ⊢ Eq x (HMul.hMul (↑(x.gcd y)) (HDiv.hDiv x ↑(x.gcd y)))
    -/
    rw [Int.mul_ediv_cancel']
    /-
      case h.e'_1
      x y z : Int
      h : PythagoreanTriple x y z
      hc : ⋯.IsPrimitiveClassified
      ⊢ Dvd.dvd (↑(x.gcd y)) x
    -/
  · exact Int.gcd_dvd_left
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2
      x y z : Int
      h : PythagoreanTriple x y z
      hc : ⋯.IsPrimitiveClassified
      ⊢ Dvd.dvd (↑(x.gcd y)) y
    -/
  · exact Int.gcd_dvd_right
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      x y z : Int
      h : PythagoreanTriple x y z
      hc : ⋯.IsPrimitiveClassified
      ⊢ Dvd.dvd (↑(x.gcd y)) z
    -/
  · exact h.gcd_dvd
    /-
      🎉 no goals
    -/


theorem ne_zero_of_coprime (hc : Int.gcd x y = 1) : z ≠ 0 := by
  suffices 0 < z * z by
    rintro rfl
    norm_num at this
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    ⊢ LT.lt 0 (HMul.hMul z z)
  -/
  rw [← h.eq, ← sq, ← sq]
  have hc' : Int.gcd x y ≠ 0 := by
    rw [hc]
    exact one_ne_zero
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hc' : Ne (x.gcd y) 0
    ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
  -/
  cases' Int.ne_zero_of_gcd hc' with hxz hyz
    /-
      case inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hc' : Ne (x.gcd y) 0
      hxz : Ne x 0
      ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    -/
  · apply lt_add_of_pos_of_le (sq_pos_of_ne_zero hxz) (sq_nonneg y)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hc' : Ne (x.gcd y) 0
      hyz : Ne y 0
      ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
    -/
  · apply lt_add_of_le_of_pos (sq_nonneg x) (sq_pos_of_ne_zero hyz)
    /-
      🎉 no goals
    -/


theorem isPrimitiveClassified_of_coprime_of_zero_left (hc : Int.gcd x y = 1) (hx : x = 0) :
    h.IsPrimitiveClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hx : Eq x 0
    ⊢ h.IsPrimitiveClassified
  -/
  subst x
  /-
    y z : Int
    h : PythagoreanTriple 0 y z
    hc : Eq (Int.gcd 0 y) 1
    ⊢ h.IsPrimitiveClassified
  -/
  change Nat.gcd 0 (Int.natAbs y) = 1 at hc
  /-
    y z : Int
    h : PythagoreanTriple 0 y z
    hc : Eq (Nat.gcd 0 y.natAbs) 1
    ⊢ h.IsPrimitiveClassified
  -/
  rw [Nat.gcd_zero_left (Int.natAbs y)] at hc
  /-
    y z : Int
    h : PythagoreanTriple 0 y z
    hc : Eq y.natAbs 1
    ⊢ h.IsPrimitiveClassified
  -/
  cases' Int.natAbs_eq y with hy hy
    /-
      case inl
      y z : Int
      h : PythagoreanTriple 0 y z
      hc : Eq y.natAbs 1
      hy : Eq y ↑y.natAbs
      ⊢ h.IsPrimitiveClassified
    -/
  · use 1, 0
    /-
      case h
      y z : Int
      h : PythagoreanTriple 0 y z
      hc : Eq y.natAbs 1
      hy : Eq y ↑y.natAbs
      ⊢ And (Or (And (Eq 0 (HSub.hSub (HPow.hPow 1 2) (HPow.hPow 0 2))) (Eq y (HMul. …
    -/
    rw [hy, hc, Int.gcd_zero_right]
    /-
      case h
      y z : Int
      h : PythagoreanTriple 0 y z
      hc : Eq y.natAbs 1
      hy : Eq y ↑y.natAbs
      ⊢ And (Or (And (Eq 0 (HSub.hSub (HPow.hPow 1 2) (HPow.hPow 0 2))) (Eq (↑1) (HM …
    -/
    decide
    /-
      🎉 no goals
    -/
    /-
      case inr
      y z : Int
      h : PythagoreanTriple 0 y z
      hc : Eq y.natAbs 1
      hy : Eq y (Neg.neg ↑y.natAbs)
      ⊢ h.IsPrimitiveClassified
    -/
  · use 0, 1
    /-
      case h
      y z : Int
      h : PythagoreanTriple 0 y z
      hc : Eq y.natAbs 1
      hy : Eq y (Neg.neg ↑y.natAbs)
      ⊢ And (Or (And (Eq 0 (HSub.hSub (HPow.hPow 0 2) (HPow.hPow 1 2))) (Eq y (HMul. …
    -/
    rw [hy, hc, Int.gcd_zero_left]
    /-
      case h
      y z : Int
      h : PythagoreanTriple 0 y z
      hc : Eq y.natAbs 1
      hy : Eq y (Neg.neg ↑y.natAbs)
      ⊢ And (Or (And (Eq 0 (HSub.hSub (HPow.hPow 0 2) (HPow.hPow 1 2))) (Eq (Neg.neg …
    -/
    decide
    /-
      🎉 no goals
    -/


theorem coprime_of_coprime (hc : Int.gcd x y = 1) : Int.gcd y z = 1 := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    ⊢ Eq (y.gcd z) 1
  -/
  by_contra H
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    ⊢ False
  -/
  obtain ⟨p, hp, hpy, hpz⟩ := Nat.Prime.not_coprime_iff_dvd.mp H
  /-
    case intro.intro.intro
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p y.natAbs
    hpz : Dvd.dvd p z.natAbs
    ⊢ False
  -/
  apply hp.not_dvd_one
  /-
    case intro.intro.intro
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p y.natAbs
    hpz : Dvd.dvd p z.natAbs
    ⊢ Dvd.dvd p 1
  -/
  rw [← hc]
  /-
    case intro.intro.intro
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p y.natAbs
    hpz : Dvd.dvd p z.natAbs
    ⊢ Dvd.dvd p (x.gcd y)
  -/
  apply Nat.dvd_gcd (Int.Prime.dvd_natAbs_of_coe_dvd_sq hp _ _) hpy
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p y.natAbs
    hpz : Dvd.dvd p z.natAbs
    ⊢ Dvd.dvd (↑p) (HPow.hPow x 2)
  -/
  rw [sq, eq_sub_of_add_eq h]
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p y.natAbs
    hpz : Dvd.dvd p z.natAbs
    ⊢ Dvd.dvd (↑p) (HSub.hSub (HMul.hMul z z) (HMul.hMul y y))
  -/
  rw [← Int.natCast_dvd] at hpy hpz
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    H : Not (Eq (y.gcd z) 1)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd (↑p) y
    hpz : Dvd.dvd (↑p) z
    ⊢ Dvd.dvd (↑p) (HSub.hSub (HMul.hMul z z) (HMul.hMul y y))
  -/
  exact dvd_sub (hpz.mul_right _) (hpy.mul_right _)
  /-
    🎉 no goals
  -/


/-- A parameterization of the unit circle that is useful for classifying Pythagorean triples.
 (To be applied in the case where `K = ℚ`.) -/
def circleEquivGen (hk : ∀ x : K, 1 + x ^ 2 ≠ 0) :
    K ≃ { p : K × K // p.1 ^ 2 + p.2 ^ 2 = 1 ∧ p.2 ≠ -1 } where
  toFun x :=
    ⟨⟨2 * x / (1 + x ^ 2), (1 - x ^ 2) / (1 + x ^ 2)⟩, by
      /-
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x : K
        ⊢ Eq (HAdd.hAdd (HPow.hPow { fst := HDiv.hDiv (HMul.hMul 2 x) (HAdd.hAdd 1 (HP …
      -/
      field_simp [hk x, div_pow]
      /-
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x : K
        ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul 2 x) 2) (HPow.hPow (HSub.hSub 1 (HPow.hP …
      -/
      ring, by
      /-
        🎉 no goals
      -/
      /-
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x : K
        ⊢ Ne { fst := HDiv.hDiv (HMul.hMul 2 x) (HAdd.hAdd 1 (HPow.hPow x 2)), snd :=  …
      -/
      simp only [Ne, div_eq_iff (hk x), neg_mul, one_mul, neg_add, sub_eq_add_neg, add_left_inj]
      /-
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x : K
        ⊢ Not (Eq 1 (-1))
      -/
      simpa only [eq_neg_iff_add_eq_zero, one_pow] using hk 1⟩
      /-
        🎉 no goals
      -/
  invFun p := (p : K × K).1 / ((p : K × K).2 + 1)
  left_inv x := by
    /-
      K : Type u_1
      inst✝ : Field K
      hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      x : K
      ⊢ Eq ((fun p => HDiv.hDiv (↑p).1 (HAdd.hAdd (↑p).2 1)) ((fun x => ⟨{ fst := HD …
    -/
    have h2 : (1 + 1 : K) = 2 := by norm_num -- Porting note: rfl is not enough to close this
    have h3 : (2 : K) ≠ 0 := by
      convert hk 1
      rw [one_pow 2, h2]
    /-
      K : Type u_1
      inst✝ : Field K
      hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      x : K
      h2 : Eq (HAdd.hAdd 1 1) 2
      h3 : Ne 2 0
      ⊢ Eq ((fun p => HDiv.hDiv (↑p).1 (HAdd.hAdd (↑p).2 1)) ((fun x => ⟨{ fst := HD …
    -/
    field_simp [hk x, h2, add_assoc, add_comm, add_sub_cancel, mul_comm]
    /-
      🎉 no goals
    -/
  right_inv := fun ⟨⟨x, y⟩, hxy, hy⟩ => by
    /-
      K : Type u_1
      inst✝ : Field K
      hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
      x y : K
      hxy : Eq (HAdd.hAdd (HPow.hPow { fst := x, snd := y }.1 2) (HPow.hPow { fst := …
      hy : Ne { fst := x, snd := y }.2 (-1)
      ⊢ Eq ((fun x => ⟨{ fst := HDiv.hDiv (HMul.hMul 2 x) (HAdd.hAdd 1 (HPow.hPow x  …
    -/
    change x ^ 2 + y ^ 2 = 1 at hxy
    /-
      K : Type u_1
      inst✝ : Field K
      hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
      x y : K
      hy : Ne { fst := x, snd := y }.2 (-1)
      hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
      ⊢ Eq ((fun x => ⟨{ fst := HDiv.hDiv (HMul.hMul 2 x) (HAdd.hAdd 1 (HPow.hPow x  …
    -/
    have h2 : y + 1 ≠ 0 := mt eq_neg_of_add_eq_zero_left hy
    have h3 : (y + 1) ^ 2 + x ^ 2 = 2 * (y + 1) := by
      rw [(add_neg_eq_iff_eq_add.mpr hxy.symm).symm]
      ring
    have h4 : (2 : K) ≠ 0 := by
      convert hk 1
      rw [one_pow 2]
      ring -- Porting note: rfl is not enough to close this
    /-
      K : Type u_1
      inst✝ : Field K
      hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
      x y : K
      hy : Ne { fst := x, snd := y }.2 (-1)
      hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
      h2 : Ne (HAdd.hAdd y 1) 0
      h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
      h4 : Ne 2 0
      ⊢ Eq ((fun x => ⟨{ fst := HDiv.hDiv (HMul.hMul 2 x) (HAdd.hAdd 1 (HPow.hPow x  …
    -/
    simp only [Prod.mk.inj_iff, Subtype.mk_eq_mk]
    /-
      K : Type u_1
      inst✝ : Field K
      hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
      x y : K
      hy : Ne { fst := x, snd := y }.2 (-1)
      hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
      h2 : Ne (HAdd.hAdd y 1) 0
      h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
      h4 : Ne 2 0
      ⊢ And (Eq (HDiv.hDiv (HMul.hMul 2 (HDiv.hDiv x (HAdd.hAdd y 1))) (HAdd.hAdd 1  …
    -/
    constructor
      /-
        case left
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
        x y : K
        hy : Ne { fst := x, snd := y }.2 (-1)
        hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
        h2 : Ne (HAdd.hAdd y 1) 0
        h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
        h4 : Ne 2 0
        ⊢ Eq (HDiv.hDiv (HMul.hMul 2 (HDiv.hDiv x (HAdd.hAdd y 1))) (HAdd.hAdd 1 (HPow …
      -/
    · field_simp [h3]
      /-
        case left
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
        x y : K
        hy : Ne { fst := x, snd := y }.2 (-1)
        hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
        h2 : Ne (HAdd.hAdd y 1) 0
        h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
        h4 : Ne 2 0
        ⊢ Eq (HMul.hMul (HMul.hMul 2 x) (HPow.hPow (HAdd.hAdd y 1) 2)) (HMul.hMul x (H …
      -/
      ring
      /-
        🎉 no goals
      -/
      /-
        case right
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
        x y : K
        hy : Ne { fst := x, snd := y }.2 (-1)
        hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
        h2 : Ne (HAdd.hAdd y 1) 0
        h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
        h4 : Ne 2 0
        ⊢ Eq (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (HDiv.hDiv x (HAdd.hAdd y 1)) 2)) (HAd …
      -/
    · field_simp [h3]
      /-
        case right
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
        x y : K
        hy : Ne { fst := x, snd := y }.2 (-1)
        hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
        h2 : Ne (HAdd.hAdd y 1) 0
        h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
        h4 : Ne 2 0
        ⊢ Eq (HSub.hSub (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul y (H …
      -/
      rw [← add_neg_eq_iff_eq_add.mpr hxy.symm]
      /-
        case right
        K : Type u_1
        inst✝ : Field K
        hk : ∀ (x : K), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        x✝ : Subtype fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow.hPow p.2 2))  …
        x y : K
        hy : Ne { fst := x, snd := y }.2 (-1)
        hxy : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)) 1
        h2 : Ne (HAdd.hAdd y 1) 0
        h3 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd y 1) 2) (HPow.hPow x 2)) (HMul.hMul 2 …
        h4 : Ne 2 0
        ⊢ Eq (HSub.hSub (HPow.hPow (HAdd.hAdd y 1) 2) (HAdd.hAdd 1 (Neg.neg (HPow.hPow …
      -/
      ring
      /-
        🎉 no goals
      -/


@[simp]
theorem circleEquivGen_apply (hk : ∀ x : K, 1 + x ^ 2 ≠ 0) (x : K) :
    (circleEquivGen hk x : K × K) = ⟨2 * x / (1 + x ^ 2), (1 - x ^ 2) / (1 + x ^ 2)⟩ :=
  rfl


@[simp]
theorem circleEquivGen_symm_apply (hk : ∀ x : K, 1 + x ^ 2 ≠ 0)
    (v : { p : K × K // p.1 ^ 2 + p.2 ^ 2 = 1 ∧ p.2 ≠ -1 }) :
    (circleEquivGen hk).symm v = (v : K × K).1 / ((v : K × K).2 + 1) :=
  rfl


private theorem coprime_sq_sub_sq_add_of_even_odd {m n : ℤ} (h : Int.gcd m n = 1) (hm : m % 2 = 0)
    (hn : n % 2 = 1) : Int.gcd (m ^ 2 - n ^ 2) (m ^ 2 + n ^ 2) = 1 := by
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow m  …
  -/
  by_contra H
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
    ⊢ False
  -/
  obtain ⟨p, hp, hp1, hp2⟩ := Nat.Prime.not_coprime_iff_dvd.mp H
  /-
    case intro.intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd p (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).natAbs
    hp2 : Dvd.dvd p (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)).natAbs
    ⊢ False
  -/
  rw [← Int.natCast_dvd] at hp1 hp2
  have h2m : (p : ℤ) ∣ 2 * m ^ 2 := by
    convert dvd_add hp2 hp1 using 1
    ring
  have h2n : (p : ℤ) ∣ 2 * n ^ 2 := by
    convert dvd_sub hp2 hp1 using 1
    ring
  /-
    case intro.intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
    h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
    ⊢ False
  -/
  have hmc : p = 2 ∨ p ∣ Int.natAbs m := prime_two_or_dvd_of_dvd_two_mul_pow_self_two hp h2m
  /-
    case intro.intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
    h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
    hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
    ⊢ False
  -/
  have hnc : p = 2 ∨ p ∣ Int.natAbs n := prime_two_or_dvd_of_dvd_two_mul_pow_self_two hp h2n
  /-
    case intro.intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
    h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
    hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
    hnc : Or (Eq p 2) (Dvd.dvd p n.natAbs)
    ⊢ False
  -/
  by_cases h2 : p = 2
  -- Porting note: norm_num is not enough to close h3
  · have h3 : (m ^ 2 + n ^ 2) % 2 = 1 := by
      simp only [sq, Int.add_emod, Int.mul_emod, hm, hn, dvd_refl, Int.emod_emod_of_dvd]
      decide
    have h4 : (m ^ 2 + n ^ 2) % 2 = 0 := by
      apply Int.emod_eq_zero_of_dvd
      rwa [h2] at hp2
    /-
      case pos
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
      hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
      hnc : Or (Eq p 2) (Dvd.dvd p n.natAbs)
      h2 : Eq p 2
      h3 : Eq (HMod.hMod (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)) 2) 1
      h4 : Eq (HMod.hMod (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)) 2) 0
      ⊢ False
    -/
    rw [h4] at h3
    /-
      case pos
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
      hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
      hnc : Or (Eq p 2) (Dvd.dvd p n.natAbs)
      h2 : Eq p 2
      h3 : Eq 0 1
      h4 : Eq (HMod.hMod (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)) 2) 0
      ⊢ False
    -/
    exact zero_ne_one h3
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
      hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
      hnc : Or (Eq p 2) (Dvd.dvd p n.natAbs)
      h2 : Not (Eq p 2)
      ⊢ False
    -/
  · apply hp.not_dvd_one
    /-
      case neg
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
      hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
      hnc : Or (Eq p 2) (Dvd.dvd p n.natAbs)
      h2 : Not (Eq p 2)
      ⊢ Dvd.dvd p 1
    -/
    rw [← h]
    /-
      case neg
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      h2m : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow m 2))
      h2n : Dvd.dvd (↑p) (HMul.hMul 2 (HPow.hPow n 2))
      hmc : Or (Eq p 2) (Dvd.dvd p m.natAbs)
      hnc : Or (Eq p 2) (Dvd.dvd p n.natAbs)
      h2 : Not (Eq p 2)
      ⊢ Dvd.dvd p (m.gcd n)
    -/
    exact Nat.dvd_gcd (Or.resolve_left hmc h2) (Or.resolve_left hnc h2)
    /-
      🎉 no goals
    -/


private theorem coprime_sq_sub_sq_add_of_odd_even {m n : ℤ} (h : Int.gcd m n = 1) (hm : m % 2 = 1)
    (hn : n % 2 = 0) : Int.gcd (m ^ 2 - n ^ 2) (m ^ 2 + n ^ 2) = 1 := by
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 0
    ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow m  …
  -/
  rw [Int.gcd, ← Int.natAbs_neg (m ^ 2 - n ^ 2)]
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 0
    ⊢ Eq ((Neg.neg (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))).natAbs.gcd (HAdd.h …
  -/
  rw [(by ring : -(m ^ 2 - n ^ 2) = n ^ 2 - m ^ 2), add_comm]
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 0
    ⊢ Eq ((HSub.hSub (HPow.hPow n 2) (HPow.hPow m 2)).natAbs.gcd (HAdd.hAdd (HPow. …
  -/
  apply coprime_sq_sub_sq_add_of_even_odd _ hn hm; rwa [Int.gcd_comm]
                                                   /-
                                                     🎉 no goals
                                                   -/


private theorem coprime_sq_sub_mul_of_even_odd {m n : ℤ} (h : Int.gcd m n = 1) (hm : m % 2 = 0)
    (hn : n % 2 = 1) : Int.gcd (m ^ 2 - n ^ 2) (2 * m * n) = 1 := by
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul.hMul 2  …
  -/
  by_contra H
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    ⊢ False
  -/
  obtain ⟨p, hp, hp1, hp2⟩ := Nat.Prime.not_coprime_iff_dvd.mp H
  /-
    case intro.intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd p (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).natAbs
    hp2 : Dvd.dvd p (HMul.hMul (HMul.hMul 2 m) n).natAbs
    ⊢ False
  -/
  rw [← Int.natCast_dvd] at hp1 hp2
  have hnp : ¬(p : ℤ) ∣ Int.gcd m n := by
    rw [h]
    norm_cast
    exact mt Nat.dvd_one.mp (Nat.Prime.ne_one hp)
  /-
    case intro.intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
    ⊢ False
  -/
  cases' Int.Prime.dvd_mul hp hp2 with hp2m hpn
    /-
      case intro.intro.intro.inl
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul 2 m).natAbs
      ⊢ False
    -/
  · rw [Int.natAbs_mul] at hp2m
    /-
      case intro.intro.intro.inl
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
      ⊢ False
    -/
    cases' (Nat.Prime.dvd_mul hp).mp hp2m with hp2 hpm
      /-
        case intro.intro.intro.inl.inl
        m n : Int
        h : Eq (m.gcd n) 1
        hm : Eq (HMod.hMod m 2) 0
        hn : Eq (HMod.hMod n 2) 1
        H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
        p : Nat
        hp : Nat.Prime p
        hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
        hp2✝ : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
        hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
        hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
        hp2 : Dvd.dvd p (Int.natAbs 2)
        ⊢ False
      -/
    · have hp2' : p = 2 := (Nat.le_of_dvd zero_lt_two hp2).antisymm hp.two_le
      /-
        case intro.intro.intro.inl.inl
        m n : Int
        h : Eq (m.gcd n) 1
        hm : Eq (HMod.hMod m 2) 0
        hn : Eq (HMod.hMod n 2) 1
        H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
        p : Nat
        hp : Nat.Prime p
        hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
        hp2✝ : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
        hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
        hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
        hp2 : Dvd.dvd p (Int.natAbs 2)
        hp2' : Eq p 2
        ⊢ False
      -/
      revert hp1
      /-
        case intro.intro.intro.inl.inl
        m n : Int
        h : Eq (m.gcd n) 1
        hm : Eq (HMod.hMod m 2) 0
        hn : Eq (HMod.hMod n 2) 1
        H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
        p : Nat
        hp : Nat.Prime p
        hp2✝ : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
        hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
        hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
        hp2 : Dvd.dvd p (Int.natAbs 2)
        hp2' : Eq p 2
        ⊢ Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) → False
      -/
      rw [hp2']
      /-
        case intro.intro.intro.inl.inl
        m n : Int
        h : Eq (m.gcd n) 1
        hm : Eq (HMod.hMod m 2) 0
        hn : Eq (HMod.hMod n 2) 1
        H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
        p : Nat
        hp : Nat.Prime p
        hp2✝ : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
        hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
        hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
        hp2 : Dvd.dvd p (Int.natAbs 2)
        hp2' : Eq p 2
        ⊢ Dvd.dvd (↑2) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) → False
      -/
      apply mt Int.emod_eq_zero_of_dvd
      -- Porting note: norm_num is not enough to close this
      simp only [sq, Nat.cast_ofNat, Int.sub_emod, Int.mul_emod, hm, hn,
        mul_zero, EuclideanDomain.zero_mod, mul_one, zero_sub]
      /-
        case intro.intro.intro.inl.inl
        m n : Int
        h : Eq (m.gcd n) 1
        hm : Eq (HMod.hMod m 2) 0
        hn : Eq (HMod.hMod n 2) 1
        H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
        p : Nat
        hp : Nat.Prime p
        hp2✝ : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
        hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
        hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
        hp2 : Dvd.dvd p (Int.natAbs 2)
        hp2' : Eq p 2
        ⊢ Not (Eq (HMod.hMod (Neg.neg (HMod.hMod 1 2)) 2) 0)
      -/
      decide
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.inl.inr
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
      hpm : Dvd.dvd p m.natAbs
      ⊢ False
    -/
    apply mt (Int.dvd_gcd (Int.natCast_dvd.mpr hpm)) hnp
    /-
      case intro.intro.intro.inl.inr
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
      hpm : Dvd.dvd p m.natAbs
      ⊢ Dvd.dvd (↑p) n
    -/
    apply or_self_iff.mp
    /-
      case intro.intro.intro.inl.inr
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
      hpm : Dvd.dvd p m.natAbs
      ⊢ Or (Dvd.dvd (↑p) n) (Dvd.dvd (↑p) n)
    -/
    apply Int.Prime.dvd_mul' hp
    /-
      case intro.intro.intro.inl.inr
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
      hpm : Dvd.dvd p m.natAbs
      ⊢ Dvd.dvd (↑p) (HMul.hMul n n)
    -/
    rw [(by ring : n * n = -(m ^ 2 - n ^ 2) + m * m)]
    /-
      case intro.intro.intro.inl.inr
      m n : Int
      h : Eq (m.gcd n) 1
      hm : Eq (HMod.hMod m 2) 0
      hn : Eq (HMod.hMod n 2) 1
      H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
      hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
      hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
      hp2m : Dvd.dvd p (HMul.hMul (Int.natAbs 2) m.natAbs)
      hpm : Dvd.dvd p m.natAbs
      ⊢ Dvd.dvd (↑p) (HAdd.hAdd (Neg.neg (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
    -/
    exact hp1.neg_right.add ((Int.natCast_dvd.2 hpm).mul_right _)
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(m.gcd n))
    hpn : Dvd.dvd p n.natAbs
    ⊢ False
  -/
  rw [Int.gcd_comm] at hnp
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(n.gcd m))
    hpn : Dvd.dvd p n.natAbs
    ⊢ False
  -/
  apply mt (Int.dvd_gcd (Int.natCast_dvd.mpr hpn)) hnp
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(n.gcd m))
    hpn : Dvd.dvd p n.natAbs
    ⊢ Dvd.dvd (↑p) m
  -/
  apply or_self_iff.mp
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(n.gcd m))
    hpn : Dvd.dvd p n.natAbs
    ⊢ Or (Dvd.dvd (↑p) m) (Dvd.dvd (↑p) m)
  -/
  apply Int.Prime.dvd_mul' hp
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(n.gcd m))
    hpn : Dvd.dvd p n.natAbs
    ⊢ Dvd.dvd (↑p) (HMul.hMul m m)
  -/
  rw [(by ring : m * m = m ^ 2 - n ^ 2 + n * n)]
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(n.gcd m))
    hpn : Dvd.dvd p n.natAbs
    ⊢ Dvd.dvd (↑p) (HAdd.hAdd (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) (HMul.hM …
  -/
  apply dvd_add hp1
  /-
    case intro.intro.intro.inr
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 0
    hn : Eq (HMod.hMod n 2) 1
    H : Not (Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul. …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))
    hp2 : Dvd.dvd (↑p) (HMul.hMul (HMul.hMul 2 m) n)
    hnp : Not (Dvd.dvd ↑p ↑(n.gcd m))
    hpn : Dvd.dvd p n.natAbs
    ⊢ Dvd.dvd (↑p) (HMul.hMul n n)
  -/
  exact (Int.natCast_dvd.mpr hpn).mul_right n
  /-
    🎉 no goals
  -/


private theorem coprime_sq_sub_mul_of_odd_even {m n : ℤ} (h : Int.gcd m n = 1) (hm : m % 2 = 1)
    (hn : n % 2 = 0) : Int.gcd (m ^ 2 - n ^ 2) (2 * m * n) = 1 := by
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 0
    ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul.hMul 2  …
  -/
  rw [Int.gcd, ← Int.natAbs_neg (m ^ 2 - n ^ 2)]
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 0
    ⊢ Eq ((Neg.neg (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))).natAbs.gcd (HMul.h …
  -/
  rw [(by ring : 2 * m * n = 2 * n * m), (by ring : -(m ^ 2 - n ^ 2) = n ^ 2 - m ^ 2)]
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 0
    ⊢ Eq ((HSub.hSub (HPow.hPow n 2) (HPow.hPow m 2)).natAbs.gcd (HMul.hMul (HMul. …
  -/
  apply coprime_sq_sub_mul_of_even_odd _ hn hm; rwa [Int.gcd_comm]
                                                /-
                                                  🎉 no goals
                                                -/


private theorem coprime_sq_sub_mul {m n : ℤ} (h : Int.gcd m n = 1)
    (hmn : m % 2 = 0 ∧ n % 2 = 1 ∨ m % 2 = 1 ∧ n % 2 = 0) :
    Int.gcd (m ^ 2 - n ^ 2) (2 * m * n) = 1 := by
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hmn : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul.hMul 2  …
  -/
  cases' hmn with h1 h2
    /-
      case inl
      m n : Int
      h : Eq (m.gcd n) 1
      h1 : And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)
      ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul.hMul 2  …
    -/
  · exact coprime_sq_sub_mul_of_even_odd h h1.left h1.right
    /-
      🎉 no goals
    -/
    /-
      case inr
      m n : Int
      h : Eq (m.gcd n) 1
      h2 : And (Eq (HMod.hMod m 2) 1) (Eq (HMod.hMod n 2) 0)
      ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HMul.hMul (HMul.hMul 2  …
    -/
  · exact coprime_sq_sub_mul_of_odd_even h h2.left h2.right
    /-
      🎉 no goals
    -/


private theorem coprime_sq_sub_sq_sum_of_odd_odd {m n : ℤ} (h : Int.gcd m n = 1) (hm : m % 2 = 1)
    (hn : n % 2 = 1) :
    2 ∣ m ^ 2 + n ^ 2 ∧
      2 ∣ m ^ 2 - n ^ 2 ∧
        (m ^ 2 - n ^ 2) / 2 % 2 = 0 ∧ Int.gcd ((m ^ 2 - n ^ 2) / 2) ((m ^ 2 + n ^ 2) / 2) = 1 := by
  /-
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 1
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Dvd.dvd 2  …
  -/
  cases' exists_eq_mul_left_of_dvd (Int.dvd_sub_of_emod_eq hm) with m0 hm2
  /-
    case intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 1
    m0 : Int
    hm2 : Eq (HSub.hSub m 1) (HMul.hMul m0 2)
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Dvd.dvd 2  …
  -/
  cases' exists_eq_mul_left_of_dvd (Int.dvd_sub_of_emod_eq hn) with n0 hn2
  /-
    case intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 1
    m0 : Int
    hm2 : Eq (HSub.hSub m 1) (HMul.hMul m0 2)
    n0 : Int
    hn2 : Eq (HSub.hSub n 1) (HMul.hMul n0 2)
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Dvd.dvd 2  …
  -/
  rw [sub_eq_iff_eq_add] at hm2 hn2
  /-
    case intro.intro
    m n : Int
    h : Eq (m.gcd n) 1
    hm : Eq (HMod.hMod m 2) 1
    hn : Eq (HMod.hMod n 2) 1
    m0 : Int
    hm2 : Eq m (HAdd.hAdd (HMul.hMul m0 2) 1)
    n0 : Int
    hn2 : Eq n (HAdd.hAdd (HMul.hMul n0 2) 1)
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Dvd.dvd 2  …
  -/
  subst m
  /-
    case intro.intro
    n : Int
    hn : Eq (HMod.hMod n 2) 1
    m0 n0 : Int
    hn2 : Eq n (HAdd.hAdd (HMul.hMul n0 2) 1)
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd n) 1
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow …
  -/
  subst n
  have h1 : (m0 * 2 + 1) ^ 2 + (n0 * 2 + 1) ^ 2 = 2 * (2 * (m0 ^ 2 + n0 ^ 2 + m0 + n0) + 1) := by
    ring
  /-
    case intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow …
  -/
  have h2 : (m0 * 2 + 1) ^ 2 - (n0 * 2 + 1) ^ 2 = 2 * (2 * (m0 ^ 2 - n0 ^ 2 + m0 - n0)) := by ring
  have h3 : ((m0 * 2 + 1) ^ 2 - (n0 * 2 + 1) ^ 2) / 2 % 2 = 0 := by
    rw [h2, Int.mul_ediv_cancel_left, Int.mul_emod_right]
    decide
  /-
    case intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    ⊢ And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow …
  -/
  refine ⟨⟨_, h1⟩, ⟨_, h2⟩, h3, ?_⟩
  /-
    case intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    ⊢ Eq ((HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow …
  -/
  have h20 : (2 : ℤ) ≠ 0 := by decide
  /-
    case intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    ⊢ Eq ((HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow …
  -/
  rw [h1, h2, Int.mul_ediv_cancel_left _ h20, Int.mul_ediv_cancel_left _ h20]
  /-
    case intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    ⊢ Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) (HPow.hPo …
  -/
  by_contra h4
  /-
    case intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
    ⊢ False
  -/
  obtain ⟨p, hp, hp1, hp2⟩ := Nat.Prime.not_coprime_iff_dvd.mp h4
  /-
    case intro.intro.intro.intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd p (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) …
    hp2 : Dvd.dvd p (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow …
    ⊢ False
  -/
  apply hp.not_dvd_one
  /-
    case intro.intro.intro.intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd p (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) …
    hp2 : Dvd.dvd p (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow …
    ⊢ Dvd.dvd p 1
  -/
  rw [← h]
  /-
    case intro.intro.intro.intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd p (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) …
    hp2 : Dvd.dvd p (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow …
    ⊢ Dvd.dvd p ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1))
  -/
  rw [← Int.natCast_dvd] at hp1 hp2
  /-
    case intro.intro.intro.intro.intro
    m0 n0 : Int
    hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
    hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
    h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
    h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
    h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
    h20 : Ne 2 0
    h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
    p : Nat
    hp : Nat.Prime p
    hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
    hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
    ⊢ Dvd.dvd p ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1))
  -/
  apply Nat.dvd_gcd
    /-
      case intro.intro.intro.intro.intro.a
      m0 n0 : Int
      hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
      hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
      h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
      h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
      h20 : Ne 2 0
      h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
      ⊢ Dvd.dvd p (HAdd.hAdd (HMul.hMul m0 2) 1).natAbs
    -/
  · apply Int.Prime.dvd_natAbs_of_coe_dvd_sq hp
    /-
      case intro.intro.intro.intro.intro.a.h
      m0 n0 : Int
      hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
      hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
      h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
      h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
      h20 : Ne 2 0
      h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
      ⊢ Dvd.dvd (↑p) (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2)
    -/
    convert dvd_add hp1 hp2
    /-
      case h.e'_4
      m0 n0 : Int
      hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
      hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
      h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
      h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
      h20 : Ne 2 0
      h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
      ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HAdd.hAdd (HMul.hMul 2 (HSu …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.a
      m0 n0 : Int
      hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
      hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
      h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
      h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
      h20 : Ne 2 0
      h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
      ⊢ Dvd.dvd p (HAdd.hAdd (HMul.hMul n0 2) 1).natAbs
    -/
  · apply Int.Prime.dvd_natAbs_of_coe_dvd_sq hp
    /-
      case intro.intro.intro.intro.intro.a.h
      m0 n0 : Int
      hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
      hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
      h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
      h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
      h20 : Ne 2 0
      h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
      ⊢ Dvd.dvd (↑p) (HPow.hPow (HAdd.hAdd (HMul.hMul n0 2) 1) 2)
    -/
    convert dvd_sub hp2 hp1
    /-
      case h.e'_4
      m0 n0 : Int
      hm : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul m0 2) 1) 2) 1
      hn : Eq (HMod.hMod (HAdd.hAdd (HMul.hMul n0 2) 1) 2) 1
      h : Eq ((HAdd.hAdd (HMul.hMul m0 2) 1).gcd (HAdd.hAdd (HMul.hMul n0 2) 1)) 1
      h1 : Eq (HAdd.hAdd (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h2 : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2) 1) 2) (HPow.hPow (HA …
      h3 : Eq (HMod.hMod (HDiv.hDiv (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul m0 2 …
      h20 : Ne 2 0
      h4 : Not (Eq ((HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 2) ( …
      p : Nat
      hp : Nat.Prime p
      hp1 : Dvd.dvd (↑p) (HMul.hMul 2 (HSub.hSub (HAdd.hAdd (HSub.hSub (HPow.hPow m0 …
      hp2 : Dvd.dvd (↑p) (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (H …
      ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul n0 2) 1) 2) (HSub.hSub (HAdd.hAdd (HMul. …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem isPrimitiveClassified_aux (hc : x.gcd y = 1) (hzpos : 0 < z) {m n : ℤ}
    (hm2n2 : 0 < m ^ 2 + n ^ 2) (hv2 : (x : ℚ) / z = 2 * m * n / ((m : ℚ) ^ 2 + (n : ℚ) ^ 2))
    (hw2 : (y : ℚ) / z = ((m : ℚ) ^ 2 - (n : ℚ) ^ 2) / ((m : ℚ) ^ 2 + (n : ℚ) ^ 2))
    (H : Int.gcd (m ^ 2 - n ^ 2) (m ^ 2 + n ^ 2) = 1) (co : Int.gcd m n = 1)
    (pp : m % 2 = 0 ∧ n % 2 = 1 ∨ m % 2 = 1 ∧ n % 2 = 0) : h.IsPrimitiveClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    ⊢ h.IsPrimitiveClassified
  -/
  have hz : z ≠ 0 := ne_of_gt hzpos
  have h2 : y = m ^ 2 - n ^ 2 ∧ z = m ^ 2 + n ^ 2 := by
    apply Rat.div_int_inj hzpos hm2n2 (h.coprime_of_coprime hc) H
    rw [hw2]
    norm_cast
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    hz : Ne z 0
    h2 : And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (HAdd.hAdd ( …
    ⊢ h.IsPrimitiveClassified
  -/
  use m, n
  /-
    case h
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    hz : Ne z 0
    h2 : And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (HAdd.hAdd ( …
    ⊢ And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul. …
  -/
  apply And.intro _ (And.intro co pp)
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    hz : Ne z 0
    h2 : And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (HAdd.hAdd ( …
    ⊢ Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMul  …
  -/
  right
  /-
    case h
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    hz : Ne z 0
    h2 : And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (HAdd.hAdd ( …
    ⊢ And (Eq x (HMul.hMul (HMul.hMul 2 m) n)) (Eq y (HSub.hSub (HPow.hPow m 2) (H …
  -/
  refine ⟨?_, h2.left⟩
  /-
    case h
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    hz : Ne z 0
    h2 : And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (HAdd.hAdd ( …
    ⊢ Eq x (HMul.hMul (HMul.hMul 2 m) n)
  -/
  rw [← Rat.coe_int_inj _ _, ← div_left_inj' ((mt (Rat.coe_int_inj z 0).mp) hz), hv2, h2.right]
  /-
    case h
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    m n : Int
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hv2 : Eq (HDiv.hDiv ↑x ↑z) (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hA …
    hw2 : Eq (HDiv.hDiv ↑y ↑z) (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow …
    H : Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow  …
    co : Eq (m.gcd n) 1
    pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
    hz : Ne z 0
    h2 : And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (HAdd.hAdd ( …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑m) 2)  …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem isPrimitiveClassified_of_coprime_of_odd_of_pos (hc : Int.gcd x y = 1) (hyo : y % 2 = 1)
    (hzpos : 0 < z) : h.IsPrimitiveClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    ⊢ h.IsPrimitiveClassified
  -/
  by_cases h0 : x = 0
    /-
      case pos
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Eq x 0
      ⊢ h.IsPrimitiveClassified
    -/
  · exact h.isPrimitiveClassified_of_coprime_of_zero_left hc h0
    /-
      🎉 no goals
    -/
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    ⊢ h.IsPrimitiveClassified
  -/
  let v := (x : ℚ) / z
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    ⊢ h.IsPrimitiveClassified
  -/
  let w := (y : ℚ) / z
  have hq : v ^ 2 + w ^ 2 = 1 := by
    field_simp [v, w, sq]
    norm_cast
  have hvz : v ≠ 0 := by
    field_simp [v]
    exact h0
  have hw1 : w ≠ -1 := by
    contrapose! hvz with hw1
    -- Porting note: `contrapose` unfolds local names, refold them
    replace hw1 : w = -1 := hw1; show v = 0
    rw [hw1, neg_sq, one_pow, add_left_eq_self] at hq
    exact pow_eq_zero hq
  have hQ : ∀ x : ℚ, 1 + x ^ 2 ≠ 0 := by
    intro q
    apply ne_of_gt
    exact lt_add_of_pos_of_le zero_lt_one (sq_nonneg q)
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    ⊢ h.IsPrimitiveClassified
  -/
  have hp : (⟨v, w⟩ : ℚ × ℚ) ∈ { p : ℚ × ℚ | p.1 ^ 2 + p.2 ^ 2 = 1 ∧ p.2 ≠ -1 } := ⟨hq, hw1⟩
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    ⊢ h.IsPrimitiveClassified
  -/
  let q := (circleEquivGen hQ).symm ⟨⟨v, w⟩, hp⟩
  have ht4 : v = 2 * q / (1 + q ^ 2) ∧ w = (1 - q ^ 2) / (1 + q ^ 2) := by
    apply Prod.mk.inj
    exact congr_arg Subtype.val ((circleEquivGen hQ).apply_symm_apply ⟨⟨v, w⟩, hp⟩).symm
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    ⊢ h.IsPrimitiveClassified
  -/
  let m := (q.den : ℤ)
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    m : Int := ↑q.den
    ⊢ h.IsPrimitiveClassified
  -/
  let n := q.num
  have hm0 : m ≠ 0 := by
    -- Added to adapt to https://github.com/leanprover/lean4/pull/2734.
    -- Without `unfold`, `norm_cast` can't see the coercion.
    -- One might try `zeta := true` in `Tactic.NormCast.derive`,
    -- but that seems to break many other things.
    unfold m
    norm_cast
    apply Rat.den_nz q
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    m : Int := ↑q.den
    n : Int := q.num
    hm0 : Ne m 0
    ⊢ h.IsPrimitiveClassified
  -/
  have hq2 : q = n / m := (Rat.num_div_den q).symm
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    m : Int := ↑q.den
    n : Int := q.num
    hm0 : Ne m 0
    hq2 : Eq q (HDiv.hDiv ↑n ↑m)
    ⊢ h.IsPrimitiveClassified
  -/
  have hm2n2 : 0 < m ^ 2 + n ^ 2 := by positivity
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    m : Int := ↑q.den
    n : Int := q.num
    hm0 : Ne m 0
    hq2 : Eq q (HDiv.hDiv ↑n ↑m)
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    ⊢ h.IsPrimitiveClassified
  -/
  have hm2n20 : (m ^ 2 + n ^ 2 : ℚ) ≠ 0 := by positivity
  have hx1 {j k : ℚ} (h₁ : k ≠ 0) (h₂ : k ^ 2 + j ^ 2 ≠ 0) :
      (1 - (j / k) ^ 2) / (1 + (j / k) ^ 2) = (k ^ 2 - j ^ 2) / (k ^ 2 + j ^ 2) := by
    field_simp
  have hw2 : w = ((m : ℚ) ^ 2 - (n : ℚ) ^ 2) / ((m : ℚ) ^ 2 + (n : ℚ) ^ 2) := by
    calc
      w = (1 - q ^ 2) / (1 + q ^ 2) := by apply ht4.2
      _ = (1 - (↑n / ↑m) ^ 2) / (1 + (↑n / ↑m) ^ 2) := by rw [hq2]
      _ = _ := by exact hx1 (Int.cast_ne_zero.mpr hm0) hm2n20
  have hx2 {j k : ℚ} (h₁ : k ≠ 0) (h₂ : k ^ 2 + j ^ 2 ≠ 0) :
      2 * (j / k) / (1 + (j / k) ^ 2) = 2 * k * j / (k ^ 2 + j ^ 2) :=
    have h₃ : k * (k ^ 2 + j ^ 2) ≠ 0 := mul_ne_zero h₁ h₂
    by field_simp; ring
  have hv2 : v = 2 * m * n / ((m : ℚ) ^ 2 + (n : ℚ) ^ 2) := by
    calc
      v = 2 * q / (1 + q ^ 2) := by apply ht4.1
      _ = 2 * (n / m) / (1 + (↑n / ↑m) ^ 2) := by rw [hq2]
      _ = _ := by exact hx2 (Int.cast_ne_zero.mpr hm0) hm2n20
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    m : Int := ↑q.den
    n : Int := q.num
    hm0 : Ne m 0
    hq2 : Eq q (HDiv.hDiv ↑n ↑m)
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
    hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
    hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
    hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
    hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
    ⊢ h.IsPrimitiveClassified
  -/
  have hnmcp : Int.gcd n m = 1 := q.reduced
  have hmncp : Int.gcd m n = 1 := by
    rw [Int.gcd_comm]
    exact hnmcp
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hyo : Eq (HMod.hMod y 2) 1
    hzpos : LT.lt 0 z
    h0 : Not (Eq x 0)
    v : Rat := HDiv.hDiv ↑x ↑z
    w : Rat := HDiv.hDiv ↑y ↑z
    hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
    hvz : Ne v 0
    hw1 : Ne w (-1)
    hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
    q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
    ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
    m : Int := ↑q.den
    n : Int := q.num
    hm0 : Ne m 0
    hq2 : Eq q (HDiv.hDiv ↑n ↑m)
    hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
    hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
    hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
    hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
    hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
    hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
    hnmcp : Eq (n.gcd m) 1
    hmncp : Eq (m.gcd n) 1
    ⊢ h.IsPrimitiveClassified
  -/
  cases' Int.emod_two_eq_zero_or_one m with hm2 hm2 <;>
    /-
      case neg.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 0
      ⊢ h.IsPrimitiveClassified
    -/
    cases' Int.emod_two_eq_zero_or_one n with hn2 hn2
  · -- m even, n even
    /-
      case neg.inl.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 0
      hn2 : Eq (HMod.hMod n 2) 0
      ⊢ h.IsPrimitiveClassified
    -/
    exfalso
    have h1 : 2 ∣ (Int.gcd n m : ℤ) :=
      Int.dvd_gcd (Int.dvd_of_emod_eq_zero hn2) (Int.dvd_of_emod_eq_zero hm2)
    /-
      case neg.inl.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 0
      hn2 : Eq (HMod.hMod n 2) 0
      h1 : Dvd.dvd 2 ↑(n.gcd m)
      ⊢ False
    -/
    rw [hnmcp] at h1
    /-
      case neg.inl.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 0
      hn2 : Eq (HMod.hMod n 2) 0
      h1 : Dvd.dvd 2 ↑1
      ⊢ False
    -/
    revert h1
    /-
      case neg.inl.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 0
      hn2 : Eq (HMod.hMod n 2) 0
      ⊢ Dvd.dvd 2 ↑1 → False
    -/
    decide
    /-
      🎉 no goals
    -/
  · -- m even, n odd
    /-
      case neg.inl.inr
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 0
      hn2 : Eq (HMod.hMod n 2) 1
      ⊢ h.IsPrimitiveClassified
    -/
    apply h.isPrimitiveClassified_aux hc hzpos hm2n2 hv2 hw2 _ hmncp
      /-
        case neg.inl.inr
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hyo : Eq (HMod.hMod y 2) 1
        hzpos : LT.lt 0 z
        h0 : Not (Eq x 0)
        v : Rat := HDiv.hDiv ↑x ↑z
        w : Rat := HDiv.hDiv ↑y ↑z
        hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
        hvz : Ne v 0
        hw1 : Ne w (-1)
        hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
        q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
        ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
        m : Int := ↑q.den
        n : Int := q.num
        hm0 : Ne m 0
        hq2 : Eq q (HDiv.hDiv ↑n ↑m)
        hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
        hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
        hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
        hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
        hnmcp : Eq (n.gcd m) 1
        hmncp : Eq (m.gcd n) 1
        hm2 : Eq (HMod.hMod m 2) 0
        hn2 : Eq (HMod.hMod n 2) 1
        ⊢ Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMod m …
      -/
    · apply Or.intro_left
      /-
        case neg.inl.inr.h
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hyo : Eq (HMod.hMod y 2) 1
        hzpos : LT.lt 0 z
        h0 : Not (Eq x 0)
        v : Rat := HDiv.hDiv ↑x ↑z
        w : Rat := HDiv.hDiv ↑y ↑z
        hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
        hvz : Ne v 0
        hw1 : Ne w (-1)
        hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
        q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
        ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
        m : Int := ↑q.den
        n : Int := q.num
        hm0 : Ne m 0
        hq2 : Eq q (HDiv.hDiv ↑n ↑m)
        hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
        hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
        hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
        hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
        hnmcp : Eq (n.gcd m) 1
        hmncp : Eq (m.gcd n) 1
        hm2 : Eq (HMod.hMod m 2) 0
        hn2 : Eq (HMod.hMod n 2) 1
        ⊢ And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)
      -/
      exact And.intro hm2 hn2
      /-
        🎉 no goals
      -/
      /-
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hyo : Eq (HMod.hMod y 2) 1
        hzpos : LT.lt 0 z
        h0 : Not (Eq x 0)
        v : Rat := HDiv.hDiv ↑x ↑z
        w : Rat := HDiv.hDiv ↑y ↑z
        hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
        hvz : Ne v 0
        hw1 : Ne w (-1)
        hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
        q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
        ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
        m : Int := ↑q.den
        n : Int := q.num
        hm0 : Ne m 0
        hq2 : Eq q (HDiv.hDiv ↑n ↑m)
        hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
        hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
        hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
        hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
        hnmcp : Eq (n.gcd m) 1
        hmncp : Eq (m.gcd n) 1
        hm2 : Eq (HMod.hMod m 2) 0
        hn2 : Eq (HMod.hMod n 2) 1
        ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow m  …
      -/
    · apply coprime_sq_sub_sq_add_of_even_odd hmncp hm2 hn2
      /-
        🎉 no goals
      -/
  · -- m odd, n even
    /-
      case neg.inr.inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 1
      hn2 : Eq (HMod.hMod n 2) 0
      ⊢ h.IsPrimitiveClassified
    -/
    apply h.isPrimitiveClassified_aux hc hzpos hm2n2 hv2 hw2 _ hmncp
      /-
        case neg.inr.inl
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hyo : Eq (HMod.hMod y 2) 1
        hzpos : LT.lt 0 z
        h0 : Not (Eq x 0)
        v : Rat := HDiv.hDiv ↑x ↑z
        w : Rat := HDiv.hDiv ↑y ↑z
        hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
        hvz : Ne v 0
        hw1 : Ne w (-1)
        hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
        q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
        ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
        m : Int := ↑q.den
        n : Int := q.num
        hm0 : Ne m 0
        hq2 : Eq q (HDiv.hDiv ↑n ↑m)
        hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
        hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
        hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
        hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
        hnmcp : Eq (n.gcd m) 1
        hmncp : Eq (m.gcd n) 1
        hm2 : Eq (HMod.hMod m 2) 1
        hn2 : Eq (HMod.hMod n 2) 0
        ⊢ Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMod m …
      -/
    · apply Or.intro_right
      /-
        case neg.inr.inl.h
        x y z : Int
        h : PythagoreanTriple x y z
        hc : Eq (x.gcd y) 1
        hyo : Eq (HMod.hMod y 2) 1
        hzpos : LT.lt 0 z
        h0 : Not (Eq x 0)
        v : Rat := HDiv.hDiv ↑x ↑z
        w : Rat := HDiv.hDiv ↑y ↑z
        hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
        hvz : Ne v 0
        hw1 : Ne w (-1)
        hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
        hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
        q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
        ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
        m : Int := ↑q.den
        n : Int := q.num
        hm0 : Ne m 0
        hq2 : Eq q (HDiv.hDiv ↑n ↑m)
        hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
        hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
        hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
        hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
        hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
        hnmcp : Eq (n.gcd m) 1
        hmncp : Eq (m.gcd n) 1
        hm2 : Eq (HMod.hMod m 2) 1
        hn2 : Eq (HMod.hMod n 2) 0
        ⊢ And (Eq (HMod.hMod m 2) 1) (Eq (HMod.hMod n 2) 0)
      -/
      exact And.intro hm2 hn2
      /-
        🎉 no goals
      -/
    /-
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 1
      hn2 : Eq (HMod.hMod n 2) 0
      ⊢ Eq ((HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)).gcd (HAdd.hAdd (HPow.hPow m  …
    -/
    apply coprime_sq_sub_sq_add_of_odd_even hmncp hm2 hn2
    /-
      🎉 no goals
    -/
  · -- m odd, n odd
    /-
      case neg.inr.inr
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 1
      hn2 : Eq (HMod.hMod n 2) 1
      ⊢ h.IsPrimitiveClassified
    -/
    exfalso
    have h1 :
      2 ∣ m ^ 2 + n ^ 2 ∧
        2 ∣ m ^ 2 - n ^ 2 ∧
          (m ^ 2 - n ^ 2) / 2 % 2 = 0 ∧ Int.gcd ((m ^ 2 - n ^ 2) / 2) ((m ^ 2 + n ^ 2) / 2) = 1 :=
      coprime_sq_sub_sq_sum_of_odd_odd hmncp hm2 hn2
    have h2 : y = (m ^ 2 - n ^ 2) / 2 ∧ z = (m ^ 2 + n ^ 2) / 2 := by
      apply Rat.div_int_inj hzpos _ (h.coprime_of_coprime hc) h1.2.2.2
      · show w = _
        rw [← Rat.divInt_eq_div, ← Rat.divInt_mul_right (by norm_num : (2 : ℤ) ≠ 0)]
        rw [Int.ediv_mul_cancel h1.1, Int.ediv_mul_cancel h1.2.1, hw2, Rat.divInt_eq_div]
        norm_cast
      · apply (mul_lt_mul_right (by norm_num : 0 < (2 : ℤ))).mp
        rw [Int.ediv_mul_cancel h1.1, zero_mul]
        exact hm2n2
    /-
      case neg.inr.inr
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hyo : Eq (HMod.hMod y 2) 1
      hzpos : LT.lt 0 z
      h0 : Not (Eq x 0)
      v : Rat := HDiv.hDiv ↑x ↑z
      w : Rat := HDiv.hDiv ↑y ↑z
      hq : Eq (HAdd.hAdd (HPow.hPow v 2) (HPow.hPow w 2)) 1
      hvz : Ne v 0
      hw1 : Ne w (-1)
      hQ : ∀ (x : Rat), Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
      hp : Membership.mem (setOf fun p => And (Eq (HAdd.hAdd (HPow.hPow p.1 2) (HPow …
      q : Rat := (circleEquivGen hQ).symm ⟨{ fst := v, snd := w }, hp⟩
      ht4 : And (Eq v (HDiv.hDiv (HMul.hMul 2 q) (HAdd.hAdd 1 (HPow.hPow q 2)))) (Eq …
      m : Int := ↑q.den
      n : Int := q.num
      hm0 : Ne m 0
      hq2 : Eq q (HDiv.hDiv ↑n ↑m)
      hm2n2 : LT.lt 0 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
      hm2n20 : Ne (HAdd.hAdd (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) 0
      hx1 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hw2 : Eq w (HDiv.hDiv (HSub.hSub (HPow.hPow (↑m) 2) (HPow.hPow (↑n) 2)) (HAdd. …
      hx2 : ∀ {j k : Rat}, Ne k 0 → Ne (HAdd.hAdd (HPow.hPow k 2) (HPow.hPow j 2)) 0 …
      hv2 : Eq v (HDiv.hDiv (HMul.hMul (HMul.hMul 2 ↑m) ↑n) (HAdd.hAdd (HPow.hPow (↑ …
      hnmcp : Eq (n.gcd m) 1
      hmncp : Eq (m.gcd n) 1
      hm2 : Eq (HMod.hMod m 2) 1
      hn2 : Eq (HMod.hMod n 2) 1
      h1 : And (Dvd.dvd 2 (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Dvd.dvd …
      h2 : And (Eq y (HDiv.hDiv (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) 2)) (Eq  …
      ⊢ False
    -/
    norm_num [h2.1, h1.2.2.1] at hyo
    /-
      🎉 no goals
    -/


theorem isPrimitiveClassified_of_coprime_of_pos (hc : Int.gcd x y = 1) (hzpos : 0 < z) :
    h.IsPrimitiveClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    ⊢ h.IsPrimitiveClassified
  -/
  cases' h.even_odd_of_coprime hc with h1 h2
    /-
      case inl
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hzpos : LT.lt 0 z
      h1 : And (Eq (HMod.hMod x 2) 0) (Eq (HMod.hMod y 2) 1)
      ⊢ h.IsPrimitiveClassified
    -/
  · exact h.isPrimitiveClassified_of_coprime_of_odd_of_pos hc h1.right hzpos
    /-
      🎉 no goals
    -/
  /-
    case inr
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hzpos : LT.lt 0 z
    h2 : And (Eq (HMod.hMod x 2) 1) (Eq (HMod.hMod y 2) 0)
    ⊢ h.IsPrimitiveClassified
  -/
  rw [Int.gcd_comm] at hc
  /-
    case inr
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (y.gcd x) 1
    hzpos : LT.lt 0 z
    h2 : And (Eq (HMod.hMod x 2) 1) (Eq (HMod.hMod y 2) 0)
    ⊢ h.IsPrimitiveClassified
  -/
  obtain ⟨m, n, H⟩ := h.symm.isPrimitiveClassified_of_coprime_of_odd_of_pos hc h2.left hzpos
  /-
    case inr.intro.intro
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (y.gcd x) 1
    hzpos : LT.lt 0 z
    h2 : And (Eq (HMod.hMod x 2) 1) (Eq (HMod.hMod y 2) 0)
    m n : Int
    H : And (Or (And (Eq y (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq x (HMu …
    ⊢ h.IsPrimitiveClassified
  -/
  use m, n; tauto
            /-
              🎉 no goals
            -/


theorem isPrimitiveClassified_of_coprime (hc : Int.gcd x y = 1) : h.IsPrimitiveClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    ⊢ h.IsPrimitiveClassified
  -/
  by_cases hz : 0 < z
    /-
      case pos
      x y z : Int
      h : PythagoreanTriple x y z
      hc : Eq (x.gcd y) 1
      hz : LT.lt 0 z
      ⊢ h.IsPrimitiveClassified
    -/
  · exact h.isPrimitiveClassified_of_coprime_of_pos hc hz
    /-
      🎉 no goals
    -/
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hz : Not (LT.lt 0 z)
    ⊢ h.IsPrimitiveClassified
  -/
  have h' : PythagoreanTriple x y (-z) := by simpa [PythagoreanTriple, neg_mul_neg] using h.eq
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hz : Not (LT.lt 0 z)
    h' : PythagoreanTriple x y (Neg.neg z)
    ⊢ h.IsPrimitiveClassified
  -/
  apply h'.isPrimitiveClassified_of_coprime_of_pos hc
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hz : Not (LT.lt 0 z)
    h' : PythagoreanTriple x y (Neg.neg z)
    ⊢ LT.lt 0 (Neg.neg z)
  -/
  apply lt_of_le_of_ne _ (h'.ne_zero_of_coprime hc).symm
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    hc : Eq (x.gcd y) 1
    hz : Not (LT.lt 0 z)
    h' : PythagoreanTriple x y (Neg.neg z)
    ⊢ LE.le 0 (Neg.neg z)
  -/
  exact le_neg.mp (not_lt.mp hz)
  /-
    🎉 no goals
  -/


theorem classified : h.IsClassified := by
  /-
    x y z : Int
    h : PythagoreanTriple x y z
    ⊢ h.IsClassified
  -/
  by_cases h0 : Int.gcd x y = 0
  · have hx : x = 0 := by
      apply Int.natAbs_eq_zero.mp
      apply Nat.eq_zero_of_gcd_eq_zero_left h0
    have hy : y = 0 := by
      apply Int.natAbs_eq_zero.mp
      apply Nat.eq_zero_of_gcd_eq_zero_right h0
    /-
      case pos
      x y z : Int
      h : PythagoreanTriple x y z
      h0 : Eq (x.gcd y) 0
      hx : Eq x 0
      hy : Eq y 0
      ⊢ h.IsClassified
    -/
    use 0, 1, 0
    /-
      case h
      x y z : Int
      h : PythagoreanTriple x y z
      h0 : Eq (x.gcd y) 0
      hx : Eq x 0
      hy : Eq y 0
      ⊢ And (Or (And (Eq x (HMul.hMul 0 (HSub.hSub (HPow.hPow 1 2) (HPow.hPow 0 2))) …
    -/
    field_simp [hx, hy]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    h0 : Not (Eq (x.gcd y) 0)
    ⊢ h.IsClassified
  -/
  apply h.isClassified_of_normalize_isPrimitiveClassified
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    h0 : Not (Eq (x.gcd y) 0)
    ⊢ ⋯.IsPrimitiveClassified
  -/
  apply h.normalize.isPrimitiveClassified_of_coprime
  /-
    case neg
    x y z : Int
    h : PythagoreanTriple x y z
    h0 : Not (Eq (x.gcd y) 0)
    ⊢ Eq ((HDiv.hDiv x ↑(x.gcd y)).gcd (HDiv.hDiv y ↑(x.gcd y))) 1
  -/
  apply Int.gcd_div_gcd_div_gcd (Nat.pos_of_ne_zero h0)
  /-
    🎉 no goals
  -/


theorem coprime_classification :
    PythagoreanTriple x y z ∧ Int.gcd x y = 1 ↔
      ∃ m n,
        (x = m ^ 2 - n ^ 2 ∧ y = 2 * m * n ∨ x = 2 * m * n ∧ y = m ^ 2 - n ^ 2) ∧
          (z = m ^ 2 + n ^ 2 ∨ z = -(m ^ 2 + n ^ 2)) ∧
            Int.gcd m n = 1 ∧ (m % 2 = 0 ∧ n % 2 = 1 ∨ m % 2 = 1 ∧ n % 2 = 0) := by
  /-
    x y z : Int
    ⊢ Iff (And (PythagoreanTriple x y z) (Eq (x.gcd y) 1)) (Exists fun m => Exists …
  -/
  constructor
    /-
      case mp
      x y z : Int
      ⊢ And (PythagoreanTriple x y z) (Eq (x.gcd y) 1) → Exists fun m => Exists fun  …
    -/
  · intro h
    /-
      case mp
      x y z : Int
      h : And (PythagoreanTriple x y z) (Eq (x.gcd y) 1)
      ⊢ Exists fun m => Exists fun n => And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2 …
    -/
    obtain ⟨m, n, H⟩ := h.left.isPrimitiveClassified_of_coprime h.right
    /-
      case mp.intro.intro
      x y z : Int
      h : And (PythagoreanTriple x y z) (Eq (x.gcd y) 1)
      m n : Int
      H : And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMu …
      ⊢ Exists fun m => Exists fun n => And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2 …
    -/
    use m, n
    /-
      case h
      x y z : Int
      h : And (PythagoreanTriple x y z) (Eq (x.gcd y) 1)
      m n : Int
      H : And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMu …
      ⊢ And (Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul. …
    -/
    rcases H with ⟨⟨rfl, rfl⟩ | ⟨rfl, rfl⟩, co, pp⟩
      /-
        case h.intro.inl.intro.intro
        z m n : Int
        h : And (PythagoreanTriple (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) (HMul.h …
        co : Eq (m.gcd n) 1
        pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
        ⊢ And (Or (And (Eq (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) (HSub.hSub (HPo …
      -/
    · refine ⟨Or.inl ⟨rfl, rfl⟩, ?_, co, pp⟩
      have : z ^ 2 = (m ^ 2 + n ^ 2) ^ 2 := by
        rw [sq, ← h.left.eq]
        ring
      /-
        case h.intro.inl.intro.intro
        z m n : Int
        h : And (PythagoreanTriple (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) (HMul.h …
        co : Eq (m.gcd n) 1
        pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
        this : Eq (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2 …
        ⊢ Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HAdd.h …
      -/
      simpa using eq_or_eq_neg_of_sq_eq_sq _ _ this
      /-
        🎉 no goals
      -/
      /-
        case h.intro.inr.intro.intro
        z m n : Int
        h : And (PythagoreanTriple (HMul.hMul (HMul.hMul 2 m) n) (HSub.hSub (HPow.hPow …
        co : Eq (m.gcd n) 1
        pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
        ⊢ And (Or (And (Eq (HMul.hMul (HMul.hMul 2 m) n) (HSub.hSub (HPow.hPow m 2) (H …
      -/
    · refine ⟨Or.inr ⟨rfl, rfl⟩, ?_, co, pp⟩
      have : z ^ 2 = (m ^ 2 + n ^ 2) ^ 2 := by
        rw [sq, ← h.left.eq]
        ring
      /-
        case h.intro.inr.intro.intro
        z m n : Int
        h : And (PythagoreanTriple (HMul.hMul (HMul.hMul 2 m) n) (HSub.hSub (HPow.hPow …
        co : Eq (m.gcd n) 1
        pp : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hMo …
        this : Eq (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2 …
        ⊢ Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HAdd.h …
      -/
      simpa using eq_or_eq_neg_of_sq_eq_sq _ _ this
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x y z : Int
      ⊢ (Exists fun m => Exists fun n => And (Or (And (Eq x (HSub.hSub (HPow.hPow m  …
    -/
  · delta PythagoreanTriple
    /-
      case mpr
      x y z : Int
      ⊢ (Exists fun m => Exists fun n => And (Or (And (Eq x (HSub.hSub (HPow.hPow m  …
    -/
    rintro ⟨m, n, ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩, rfl | rfl, co, pp⟩ <;>
      first
      | constructor; ring; exact coprime_sq_sub_mul co pp
      | constructor; ring; rw [Int.gcd_comm]; exact coprime_sq_sub_mul co pp


/-- by assuming `x` is odd and `z` is positive we get a slightly more precise classification of
the Pythagorean triple `x ^ 2 + y ^ 2 = z ^ 2`-/
theorem coprime_classification' {x y z : ℤ} (h : PythagoreanTriple x y z)
    (h_coprime : Int.gcd x y = 1) (h_parity : x % 2 = 1) (h_pos : 0 < z) :
    ∃ m n,
      x = m ^ 2 - n ^ 2 ∧
        y = 2 * m * n ∧
          z = m ^ 2 + n ^ 2 ∧
            Int.gcd m n = 1 ∧ (m % 2 = 0 ∧ n % 2 = 1 ∨ m % 2 = 1 ∧ n % 2 = 0) ∧ 0 ≤ m := by
  obtain ⟨m, n, ht1, ht2, ht3, ht4⟩ :=
    PythagoreanTriple.coprime_classification.mp (And.intro h h_coprime)
  /-
    case intro.intro.intro.intro.intro
    x y z : Int
    h : PythagoreanTriple x y z
    h_coprime : Eq (x.gcd y) 1
    h_parity : Eq (HMod.hMod x 2) 1
    h_pos : LT.lt 0 z
    m n : Int
    ht1 : Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.h …
    ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
    ht3 : Eq (m.gcd n) 1
    ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
    ⊢ Exists fun m => Exists fun n => And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.h …
  -/
  rcases le_or_lt 0 m with hm | hm
    /-
      case intro.intro.intro.intro.intro.inl
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht1 : Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.h …
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LE.le 0 m
      ⊢ Exists fun m => Exists fun n => And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.h …
    -/
  · use m, n
    /-
      case h
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht1 : Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.h …
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LE.le 0 m
      ⊢ And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq y (HMul.hMul …
    -/
    cases' ht1 with h_odd h_even
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LE.le 0 m
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq y (HMul.hMul …
      -/
    · apply And.intro h_odd.1
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LE.le 0 m
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq y (HMul.hMul (HMul.hMul 2 m) n)) (And (Eq z (HAdd.hAdd (HPow.hPow m  …
      -/
      apply And.intro h_odd.2
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LE.le 0 m
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq (m.gcd n) 1) …
      -/
      cases' ht2 with h_pos h_neg
        /-
          case h.inl.inl
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos✝ : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LE.le 0 m
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
          ⊢ And (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq (m.gcd n) 1) …
        -/
      · apply And.intro h_pos (And.intro ht3 (And.intro ht4 hm))
        /-
          🎉 no goals
        -/
        /-
          case h.inl.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LE.le 0 m
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ And (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq (m.gcd n) 1) …
        -/
      · exfalso
        /-
          case h.inl.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LE.le 0 m
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ False
        -/
        revert h_pos
        /-
          case h.inl.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LE.le 0 m
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ LT.lt 0 z → False
        -/
        rw [h_neg]
        /-
          case h.inl.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LE.le 0 m
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ LT.lt 0 (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) → False
        -/
        exact imp_false.mpr (not_lt.mpr (neg_nonpos.mpr (add_nonneg (sq_nonneg m) (sq_nonneg n))))
        /-
          🎉 no goals
        -/
    /-
      case h.inr
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LE.le 0 m
      h_even : And (Eq x (HMul.hMul (HMul.hMul 2 m) n)) (Eq y (HSub.hSub (HPow.hPow  …
      ⊢ And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq y (HMul.hMul …
    -/
    exfalso
    /-
      case h.inr
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LE.le 0 m
      h_even : And (Eq x (HMul.hMul (HMul.hMul 2 m) n)) (Eq y (HSub.hSub (HPow.hPow  …
      ⊢ False
    -/
    rcases h_even with ⟨rfl, -⟩
    /-
      case h.inr.intro
      y z : Int
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LE.le 0 m
      h : PythagoreanTriple (HMul.hMul (HMul.hMul 2 m) n) y z
      h_coprime : Eq ((HMul.hMul (HMul.hMul 2 m) n).gcd y) 1
      h_parity : Eq (HMod.hMod (HMul.hMul (HMul.hMul 2 m) n) 2) 1
      ⊢ False
    -/
    rw [mul_assoc, Int.mul_emod_right] at h_parity
    /-
      case h.inr.intro
      y z : Int
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LE.le 0 m
      h : PythagoreanTriple (HMul.hMul (HMul.hMul 2 m) n) y z
      h_coprime : Eq ((HMul.hMul (HMul.hMul 2 m) n).gcd y) 1
      h_parity : Eq 0 1
      ⊢ False
    -/
    exact zero_ne_one h_parity
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.inr
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht1 : Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.h …
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LT.lt m 0
      ⊢ Exists fun m => Exists fun n => And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.h …
    -/
  · use -m, -n
    /-
      case h
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht1 : Or (And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.h …
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LT.lt m 0
      ⊢ And (Eq x (HSub.hSub (HPow.hPow (Neg.neg m) 2) (HPow.hPow (Neg.neg n) 2))) ( …
    -/
    cases' ht1 with h_odd h_even
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LT.lt m 0
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq x (HSub.hSub (HPow.hPow (Neg.neg m) 2) (HPow.hPow (Neg.neg n) 2))) ( …
      -/
    · rw [neg_sq m]
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LT.lt m 0
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow (Neg.neg n) 2))) (And (Eq y  …
      -/
      rw [neg_sq n]
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LT.lt m 0
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq y (HMul.hMul …
      -/
      apply And.intro h_odd.1
      /-
        case h.inl
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LT.lt m 0
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq y (HMul.hMul (HMul.hMul 2 (Neg.neg m)) (Neg.neg n))) (And (Eq z (HAd …
      -/
      constructor
        /-
          case h.inl.left
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos : LT.lt 0 z
          m n : Int
          ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          ⊢ Eq y (HMul.hMul (HMul.hMul 2 (Neg.neg m)) (Neg.neg n))
        -/
      · rw [h_odd.2]
        /-
          case h.inl.left
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos : LT.lt 0 z
          m n : Int
          ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          ⊢ Eq (HMul.hMul (HMul.hMul 2 m) n) (HMul.hMul (HMul.hMul 2 (Neg.neg m)) (Neg.n …
        -/
        ring
        /-
          🎉 no goals
        -/
      /-
        case h.inl.right
        x y z : Int
        h : PythagoreanTriple x y z
        h_coprime : Eq (x.gcd y) 1
        h_parity : Eq (HMod.hMod x 2) 1
        h_pos : LT.lt 0 z
        m n : Int
        ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
        ht3 : Eq (m.gcd n) 1
        ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
        hm : LT.lt m 0
        h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
        ⊢ And (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq ((Neg.neg m) …
      -/
      cases' ht2 with h_pos h_neg
        /-
          case h.inl.right.inl
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos✝ : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
          ⊢ And (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq ((Neg.neg m) …
        -/
      · apply And.intro h_pos
        /-
          case h.inl.right.inl
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos✝ : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
          ⊢ And (Eq ((Neg.neg m).gcd (Neg.neg n)) 1) (And (Or (And (Eq (HMod.hMod (Neg.n …
        -/
        constructor
          /-
            case h.inl.right.inl.left
            x y z : Int
            h : PythagoreanTriple x y z
            h_coprime : Eq (x.gcd y) 1
            h_parity : Eq (HMod.hMod x 2) 1
            h_pos✝ : LT.lt 0 z
            m n : Int
            ht3 : Eq (m.gcd n) 1
            ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
            hm : LT.lt m 0
            h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
            h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
            ⊢ Eq ((Neg.neg m).gcd (Neg.neg n)) 1
          -/
        · delta Int.gcd
          /-
            case h.inl.right.inl.left
            x y z : Int
            h : PythagoreanTriple x y z
            h_coprime : Eq (x.gcd y) 1
            h_parity : Eq (HMod.hMod x 2) 1
            h_pos✝ : LT.lt 0 z
            m n : Int
            ht3 : Eq (m.gcd n) 1
            ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
            hm : LT.lt m 0
            h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
            h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
            ⊢ Eq ((Neg.neg m).natAbs.gcd (Neg.neg n).natAbs) 1
          -/
          rw [Int.natAbs_neg, Int.natAbs_neg]
          /-
            case h.inl.right.inl.left
            x y z : Int
            h : PythagoreanTriple x y z
            h_coprime : Eq (x.gcd y) 1
            h_parity : Eq (HMod.hMod x 2) 1
            h_pos✝ : LT.lt 0 z
            m n : Int
            ht3 : Eq (m.gcd n) 1
            ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
            hm : LT.lt m 0
            h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
            h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
            ⊢ Eq (m.natAbs.gcd n.natAbs) 1
          -/
          exact ht3
          /-
            🎉 no goals
          -/
          /-
            case h.inl.right.inl.right
            x y z : Int
            h : PythagoreanTriple x y z
            h_coprime : Eq (x.gcd y) 1
            h_parity : Eq (HMod.hMod x 2) 1
            h_pos✝ : LT.lt 0 z
            m n : Int
            ht3 : Eq (m.gcd n) 1
            ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
            hm : LT.lt m 0
            h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
            h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
            ⊢ And (Or (And (Eq (HMod.hMod (Neg.neg m) 2) 0) (Eq (HMod.hMod (Neg.neg n) 2)  …
          -/
        · rw [Int.neg_emod_two, Int.neg_emod_two]
          /-
            case h.inl.right.inl.right
            x y z : Int
            h : PythagoreanTriple x y z
            h_coprime : Eq (x.gcd y) 1
            h_parity : Eq (HMod.hMod x 2) 1
            h_pos✝ : LT.lt 0 z
            m n : Int
            ht3 : Eq (m.gcd n) 1
            ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
            hm : LT.lt m 0
            h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
            h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
            ⊢ And (Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.h …
          -/
          apply And.intro ht4
          /-
            case h.inl.right.inl.right
            x y z : Int
            h : PythagoreanTriple x y z
            h_coprime : Eq (x.gcd y) 1
            h_parity : Eq (HMod.hMod x 2) 1
            h_pos✝ : LT.lt 0 z
            m n : Int
            ht3 : Eq (m.gcd n) 1
            ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
            hm : LT.lt m 0
            h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
            h_pos : Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))
            ⊢ LE.le 0 (Neg.neg m)
          -/
          omega
          /-
            🎉 no goals
          -/
        /-
          case h.inl.right.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ And (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (And (Eq ((Neg.neg m) …
        -/
      · exfalso
        /-
          case h.inl.right.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          h_pos : LT.lt 0 z
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ False
        -/
        revert h_pos
        /-
          case h.inl.right.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ LT.lt 0 z → False
        -/
        rw [h_neg]
        /-
          case h.inl.right.inr
          x y z : Int
          h : PythagoreanTriple x y z
          h_coprime : Eq (x.gcd y) 1
          h_parity : Eq (HMod.hMod x 2) 1
          m n : Int
          ht3 : Eq (m.gcd n) 1
          ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
          hm : LT.lt m 0
          h_odd : And (Eq x (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) (Eq y (HMul.hMu …
          h_neg : Eq z (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))
          ⊢ LT.lt 0 (Neg.neg (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) → False
        -/
        exact imp_false.mpr (not_lt.mpr (neg_nonpos.mpr (add_nonneg (sq_nonneg m) (sq_nonneg n))))
        /-
          🎉 no goals
        -/
    /-
      case h.inr
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LT.lt m 0
      h_even : And (Eq x (HMul.hMul (HMul.hMul 2 m) n)) (Eq y (HSub.hSub (HPow.hPow  …
      ⊢ And (Eq x (HSub.hSub (HPow.hPow (Neg.neg m) 2) (HPow.hPow (Neg.neg n) 2))) ( …
    -/
    exfalso
    /-
      case h.inr
      x y z : Int
      h : PythagoreanTriple x y z
      h_coprime : Eq (x.gcd y) 1
      h_parity : Eq (HMod.hMod x 2) 1
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LT.lt m 0
      h_even : And (Eq x (HMul.hMul (HMul.hMul 2 m) n)) (Eq y (HSub.hSub (HPow.hPow  …
      ⊢ False
    -/
    rcases h_even with ⟨rfl, -⟩
    /-
      case h.inr.intro
      y z : Int
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LT.lt m 0
      h : PythagoreanTriple (HMul.hMul (HMul.hMul 2 m) n) y z
      h_coprime : Eq ((HMul.hMul (HMul.hMul 2 m) n).gcd y) 1
      h_parity : Eq (HMod.hMod (HMul.hMul (HMul.hMul 2 m) n) 2) 1
      ⊢ False
    -/
    rw [mul_assoc, Int.mul_emod_right] at h_parity
    /-
      case h.inr.intro
      y z : Int
      h_pos : LT.lt 0 z
      m n : Int
      ht2 : Or (Eq z (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2))) (Eq z (Neg.neg (HA …
      ht3 : Eq (m.gcd n) 1
      ht4 : Or (And (Eq (HMod.hMod m 2) 0) (Eq (HMod.hMod n 2) 1)) (And (Eq (HMod.hM …
      hm : LT.lt m 0
      h : PythagoreanTriple (HMul.hMul (HMul.hMul 2 m) n) y z
      h_coprime : Eq ((HMul.hMul (HMul.hMul 2 m) n).gcd y) 1
      h_parity : Eq 0 1
      ⊢ False
    -/
    exact zero_ne_one h_parity
    /-
      🎉 no goals
    -/


/-- **Formula for Pythagorean Triples** -/
theorem classification :
    PythagoreanTriple x y z ↔
      ∃ k m n,
        (x = k * (m ^ 2 - n ^ 2) ∧ y = k * (2 * m * n) ∨
            x = k * (2 * m * n) ∧ y = k * (m ^ 2 - n ^ 2)) ∧
          (z = k * (m ^ 2 + n ^ 2) ∨ z = -k * (m ^ 2 + n ^ 2)) := by
  /-
    x y z : Int
    ⊢ Iff (PythagoreanTriple x y z) (Exists fun k => Exists fun m => Exists fun n  …
  -/
  constructor
    /-
      case mp
      x y z : Int
      ⊢ PythagoreanTriple x y z → Exists fun k => Exists fun m => Exists fun n => An …
    -/
  · intro h
    /-
      case mp
      x y z : Int
      h : PythagoreanTriple x y z
      ⊢ Exists fun k => Exists fun m => Exists fun n => And (Or (And (Eq x (HMul.hMu …
    -/
    obtain ⟨k, m, n, H⟩ := h.classified
    /-
      case mp.intro.intro.intro
      x y z : Int
      h : PythagoreanTriple x y z
      k m n : Int
      H : And (Or (And (Eq x (HMul.hMul k (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2) …
      ⊢ Exists fun k => Exists fun m => Exists fun n => And (Or (And (Eq x (HMul.hMu …
    -/
    use k, m, n
    /-
      case h
      x y z : Int
      h : PythagoreanTriple x y z
      k m n : Int
      H : And (Or (And (Eq x (HMul.hMul k (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2) …
      ⊢ And (Or (And (Eq x (HMul.hMul k (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) …
    -/
    rcases H with (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
      /-
        case h.intro.inl.intro
        z k m n : Int
        right✝ : Eq (m.gcd n) 1
        h : PythagoreanTriple (HMul.hMul k (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
        ⊢ And (Or (And (Eq (HMul.hMul k (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2))) ( …
      -/
    · refine ⟨Or.inl ⟨rfl, rfl⟩, ?_⟩
      have : z ^ 2 = (k * (m ^ 2 + n ^ 2)) ^ 2 := by
        rw [sq, ← h.eq]
        ring
      /-
        case h.intro.inl.intro
        z k m n : Int
        right✝ : Eq (m.gcd n) 1
        h : PythagoreanTriple (HMul.hMul k (HSub.hSub (HPow.hPow m 2) (HPow.hPow n 2)) …
        this : Eq (HPow.hPow z 2) (HPow.hPow (HMul.hMul k (HAdd.hAdd (HPow.hPow m 2) ( …
        ⊢ Or (Eq z (HMul.hMul k (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))) (Eq z (H …
      -/
      simpa using eq_or_eq_neg_of_sq_eq_sq _ _ this
      /-
        🎉 no goals
      -/
      /-
        case h.intro.inr.intro
        z k m n : Int
        right✝ : Eq (m.gcd n) 1
        h : PythagoreanTriple (HMul.hMul k (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul k …
        ⊢ And (Or (And (Eq (HMul.hMul k (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul k (H …
      -/
    · refine ⟨Or.inr ⟨rfl, rfl⟩, ?_⟩
      have : z ^ 2 = (k * (m ^ 2 + n ^ 2)) ^ 2 := by
        rw [sq, ← h.eq]
        ring
      /-
        case h.intro.inr.intro
        z k m n : Int
        right✝ : Eq (m.gcd n) 1
        h : PythagoreanTriple (HMul.hMul k (HMul.hMul (HMul.hMul 2 m) n)) (HMul.hMul k …
        this : Eq (HPow.hPow z 2) (HPow.hPow (HMul.hMul k (HAdd.hAdd (HPow.hPow m 2) ( …
        ⊢ Or (Eq z (HMul.hMul k (HAdd.hAdd (HPow.hPow m 2) (HPow.hPow n 2)))) (Eq z (H …
      -/
      simpa using eq_or_eq_neg_of_sq_eq_sq _ _ this
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x y z : Int
      ⊢ (Exists fun k => Exists fun m => Exists fun n => And (Or (And (Eq x (HMul.hM …
    -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
  · rintro ⟨k, m, n, ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩, rfl | rfl⟩ <;> delta PythagoreanTriple <;> ring
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


