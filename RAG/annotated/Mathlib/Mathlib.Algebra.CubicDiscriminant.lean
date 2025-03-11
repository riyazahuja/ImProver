/-- The structure representing a cubic polynomial. -/
@[ext]
structure Cubic (R : Type*) where
  (a b c d : R)


instance [Inhabited R] : Inhabited (Cubic R) :=
  ⟨⟨default, default, default, default⟩⟩


instance [Zero R] : Zero (Cubic R) :=
  ⟨⟨0, 0, 0, 0⟩⟩


/-- Convert a cubic polynomial to a polynomial. -/
def toPoly (P : Cubic R) : R[X] :=
  C P.a * X ^ 3 + C P.b * X ^ 2 + C P.c * X + C P.d


theorem C_mul_prod_X_sub_C_eq [CommRing S] {w x y z : S} :
    C w * (X - C x) * (X - C y) * (X - C z) =
      toPoly ⟨w, w * -(x + y + z), w * (x * y + x * z + y * z), w * -(x * y * z)⟩ := by
  /-
    S : Type u_2
    inst✝ : CommRing S
    w x y z : S
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Polynomial.C w) (HSub.hSub Polynomial.X …
  -/
  simp only [toPoly, C_neg, C_add, C_mul]
  /-
    S : Type u_2
    inst✝ : CommRing S
    w x y z : S
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Polynomial.C w) (HSub.hSub Polynomial.X …
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem prod_X_sub_C_eq [CommRing S] {x y z : S} :
    (X - C x) * (X - C y) * (X - C z) =
      toPoly ⟨1, -(x + y + z), x * y + x * z + y * z, -(x * y * z)⟩ := by
  /-
    S : Type u_2
    inst✝ : CommRing S
    x y z : S
    ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C x)) (HSub.hSu …
  -/
  rw [← one_mul <| X - C x, ← C_1, C_mul_prod_X_sub_C_eq, one_mul, one_mul, one_mul]
  /-
    🎉 no goals
  -/


private theorem coeffs : (∀ n > 3, P.toPoly.coeff n = 0) ∧ P.toPoly.coeff 3 = P.a ∧
    P.toPoly.coeff 2 = P.b ∧ P.toPoly.coeff 1 = P.c ∧ P.toPoly.coeff 0 = P.d := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ⊢ And (∀ (n : Nat), GT.gt n 3 → Eq (P.toPoly.coeff n) 0) (And (Eq (P.toPoly.co …
  -/
  simp only [toPoly, coeff_add, coeff_C, coeff_C_mul_X, coeff_C_mul_X_pow]
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ⊢ And (∀ (n : Nat), GT.gt n 3 → Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (ite (Eq n …
  -/
  norm_num
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ⊢ ∀ (n : Nat), LT.lt 3 n → Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (ite (Eq n 3) P …
  -/
  intro n hn
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    n : Nat
    hn : LT.lt 3 n
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (ite (Eq n 3) P.a 0) (ite (Eq n 2) P.b 0 …
  -/
  repeat' rw [if_neg]
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    n : Nat
    hn : LT.lt 3 n
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd 0 0) 0) 0) 0
  -/
  any_goals omega
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    n : Nat
    hn : LT.lt 3 n
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd 0 0) 0) 0) 0
  -/
  repeat' rw [zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_eq_zero {n : ℕ} (hn : 3 < n) : P.toPoly.coeff n = 0 :=
  coeffs.1 n hn


@[simp]
theorem coeff_eq_a : P.toPoly.coeff 3 = P.a :=
  coeffs.2.1


@[simp]
theorem coeff_eq_b : P.toPoly.coeff 2 = P.b :=
  coeffs.2.2.1


@[simp]
theorem coeff_eq_c : P.toPoly.coeff 1 = P.c :=
  coeffs.2.2.2.1


@[simp]
theorem coeff_eq_d : P.toPoly.coeff 0 = P.d :=
  coeffs.2.2.2.2


                                                            /-
                                                              R : Type u_1
                                                              P Q : Cubic R
                                                              inst✝ : Semiring R
                                                              h : Eq P.toPoly Q.toPoly
                                                              ⊢ Eq P.a Q.a
                                                            -/
theorem a_of_eq (h : P.toPoly = Q.toPoly) : P.a = Q.a := by rw [← coeff_eq_a, h, coeff_eq_a]
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                            /-
                                                              R : Type u_1
                                                              P Q : Cubic R
                                                              inst✝ : Semiring R
                                                              h : Eq P.toPoly Q.toPoly
                                                              ⊢ Eq P.b Q.b
                                                            -/
theorem b_of_eq (h : P.toPoly = Q.toPoly) : P.b = Q.b := by rw [← coeff_eq_b, h, coeff_eq_b]
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                            /-
                                                              R : Type u_1
                                                              P Q : Cubic R
                                                              inst✝ : Semiring R
                                                              h : Eq P.toPoly Q.toPoly
                                                              ⊢ Eq P.c Q.c
                                                            -/
theorem c_of_eq (h : P.toPoly = Q.toPoly) : P.c = Q.c := by rw [← coeff_eq_c, h, coeff_eq_c]
                                                            /-
                                                              🎉 no goals
                                                            -/


                                                            /-
                                                              R : Type u_1
                                                              P Q : Cubic R
                                                              inst✝ : Semiring R
                                                              h : Eq P.toPoly Q.toPoly
                                                              ⊢ Eq P.d Q.d
                                                            -/
theorem d_of_eq (h : P.toPoly = Q.toPoly) : P.d = Q.d := by rw [← coeff_eq_d, h, coeff_eq_d]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem toPoly_injective (P Q : Cubic R) : P.toPoly = Q.toPoly ↔ P = Q :=
  ⟨fun h ↦ Cubic.ext (a_of_eq h) (b_of_eq h) (c_of_eq h) (d_of_eq h), congr_arg toPoly⟩


theorem of_a_eq_zero (ha : P.a = 0) : P.toPoly = C P.b * X ^ 2 + C P.c * X + C P.d := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    ⊢ Eq P.toPoly (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C P.b) (HPow.hPow P …
  -/
  rw [toPoly, ha, C_0, zero_mul, zero_add]
  /-
    🎉 no goals
  -/


theorem of_a_eq_zero' : toPoly ⟨0, b, c, d⟩ = C b * X ^ 2 + C c * X + C d :=
  of_a_eq_zero rfl


theorem of_b_eq_zero (ha : P.a = 0) (hb : P.b = 0) : P.toPoly = C P.c * X + C P.d := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    ⊢ Eq P.toPoly (HAdd.hAdd (HMul.hMul (Polynomial.C P.c) Polynomial.X) (Polynomi …
  -/
  rw [of_a_eq_zero ha, hb, C_0, zero_mul, zero_add]
  /-
    🎉 no goals
  -/


theorem of_b_eq_zero' : toPoly ⟨0, 0, c, d⟩ = C c * X + C d :=
  of_b_eq_zero rfl rfl


theorem of_c_eq_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) : P.toPoly = C P.d := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    ⊢ Eq P.toPoly (Polynomial.C P.d)
  -/
  rw [of_b_eq_zero ha hb, hc, C_0, zero_mul, zero_add]
  /-
    🎉 no goals
  -/


theorem of_c_eq_zero' : toPoly ⟨0, 0, 0, d⟩ = C d :=
  of_c_eq_zero rfl rfl rfl


theorem of_d_eq_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) (hd : P.d = 0) :
    P.toPoly = 0 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    hd : Eq P.d 0
    ⊢ Eq P.toPoly 0
  -/
  rw [of_c_eq_zero ha hb hc, hd, C_0]
  /-
    🎉 no goals
  -/


theorem of_d_eq_zero' : (⟨0, 0, 0, 0⟩ : Cubic R).toPoly = 0 :=
  of_d_eq_zero rfl rfl rfl rfl


theorem zero : (0 : Cubic R).toPoly = 0 :=
  of_d_eq_zero'


theorem toPoly_eq_zero_iff (P : Cubic R) : P.toPoly = 0 ↔ P = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    P : Cubic R
    ⊢ Iff (Eq P.toPoly 0) (Eq P 0)
  -/
  rw [← zero, toPoly_injective]
  /-
    🎉 no goals
  -/


private theorem ne_zero (h0 : P.a ≠ 0 ∨ P.b ≠ 0 ∨ P.c ≠ 0 ∨ P.d ≠ 0) : P.toPoly ≠ 0 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    h0 : Or (Ne P.a 0) (Or (Ne P.b 0) (Or (Ne P.c 0) (Ne P.d 0)))
    ⊢ Ne P.toPoly 0
  -/
  contrapose! h0
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    h0 : Eq P.toPoly 0
    ⊢ And (Eq P.a 0) (And (Eq P.b 0) (And (Eq P.c 0) (Eq P.d 0)))
  -/
  rw [(toPoly_eq_zero_iff P).mp h0]
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    h0 : Eq P.toPoly 0
    ⊢ And (Eq (Cubic.a 0) 0) (And (Eq (Cubic.b 0) 0) (And (Eq (Cubic.c 0) 0) (Eq ( …
  -/
  exact ⟨rfl, rfl, rfl, rfl⟩
  /-
    🎉 no goals
  -/


theorem ne_zero_of_a_ne_zero (ha : P.a ≠ 0) : P.toPoly ≠ 0 :=
  (or_imp.mp ne_zero).1 ha


theorem ne_zero_of_b_ne_zero (hb : P.b ≠ 0) : P.toPoly ≠ 0 :=
  (or_imp.mp (or_imp.mp ne_zero).2).1 hb


theorem ne_zero_of_c_ne_zero (hc : P.c ≠ 0) : P.toPoly ≠ 0 :=
  (or_imp.mp (or_imp.mp (or_imp.mp ne_zero).2).2).1 hc


theorem ne_zero_of_d_ne_zero (hd : P.d ≠ 0) : P.toPoly ≠ 0 :=
  (or_imp.mp (or_imp.mp (or_imp.mp ne_zero).2).2).2 hd


@[simp]
theorem leadingCoeff_of_a_ne_zero (ha : P.a ≠ 0) : P.toPoly.leadingCoeff = P.a :=
  leadingCoeff_cubic ha


@[simp]
theorem leadingCoeff_of_a_ne_zero' (ha : a ≠ 0) : (toPoly ⟨a, b, c, d⟩).leadingCoeff = a :=
  leadingCoeff_of_a_ne_zero ha


@[simp]
theorem leadingCoeff_of_b_ne_zero (ha : P.a = 0) (hb : P.b ≠ 0) : P.toPoly.leadingCoeff = P.b := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Ne P.b 0
    ⊢ Eq P.toPoly.leadingCoeff P.b
  -/
  rw [of_a_eq_zero ha, leadingCoeff_quadratic hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_of_b_ne_zero' (hb : b ≠ 0) : (toPoly ⟨0, b, c, d⟩).leadingCoeff = b :=
  leadingCoeff_of_b_ne_zero rfl hb


@[simp]
theorem leadingCoeff_of_c_ne_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c ≠ 0) :
    P.toPoly.leadingCoeff = P.c := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Ne P.c 0
    ⊢ Eq P.toPoly.leadingCoeff P.c
  -/
  rw [of_b_eq_zero ha hb, leadingCoeff_linear hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_of_c_ne_zero' (hc : c ≠ 0) : (toPoly ⟨0, 0, c, d⟩).leadingCoeff = c :=
  leadingCoeff_of_c_ne_zero rfl rfl hc


@[simp]
theorem leadingCoeff_of_c_eq_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) :
    P.toPoly.leadingCoeff = P.d := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    ⊢ Eq P.toPoly.leadingCoeff P.d
  -/
  rw [of_c_eq_zero ha hb hc, leadingCoeff_C]
  /-
    🎉 no goals
  -/


theorem leadingCoeff_of_c_eq_zero' : (toPoly ⟨0, 0, 0, d⟩).leadingCoeff = d :=
  leadingCoeff_of_c_eq_zero rfl rfl rfl


theorem monic_of_a_eq_one (ha : P.a = 1) : P.toPoly.Monic := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 1
    ⊢ P.toPoly.Monic
  -/
  nontriviality R
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 1
    a✝ : Nontrivial R
    ⊢ P.toPoly.Monic
  -/
  rw [Monic, leadingCoeff_of_a_ne_zero (ha ▸ one_ne_zero), ha]
  /-
    🎉 no goals
  -/


theorem monic_of_a_eq_one' : (toPoly ⟨1, b, c, d⟩).Monic :=
  monic_of_a_eq_one rfl


theorem monic_of_b_eq_one (ha : P.a = 0) (hb : P.b = 1) : P.toPoly.Monic := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 1
    ⊢ P.toPoly.Monic
  -/
  nontriviality R
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 1
    a✝ : Nontrivial R
    ⊢ P.toPoly.Monic
  -/
  rw [Monic, leadingCoeff_of_b_ne_zero ha (hb ▸ one_ne_zero), hb]
  /-
    🎉 no goals
  -/


theorem monic_of_b_eq_one' : (toPoly ⟨0, 1, c, d⟩).Monic :=
  monic_of_b_eq_one rfl rfl


theorem monic_of_c_eq_one (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 1) : P.toPoly.Monic := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 1
    ⊢ P.toPoly.Monic
  -/
  nontriviality R
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 1
    a✝ : Nontrivial R
    ⊢ P.toPoly.Monic
  -/
  rw [Monic, leadingCoeff_of_c_ne_zero ha hb (hc ▸ one_ne_zero), hc]
  /-
    🎉 no goals
  -/


theorem monic_of_c_eq_one' : (toPoly ⟨0, 0, 1, d⟩).Monic :=
  monic_of_c_eq_one rfl rfl rfl


theorem monic_of_d_eq_one (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) (hd : P.d = 1) :
    P.toPoly.Monic := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    hd : Eq P.d 1
    ⊢ P.toPoly.Monic
  -/
  rw [Monic, leadingCoeff_of_c_eq_zero ha hb hc, hd]
  /-
    🎉 no goals
  -/


theorem monic_of_d_eq_one' : (toPoly ⟨0, 0, 0, 1⟩).Monic :=
  monic_of_d_eq_one rfl rfl rfl rfl


/-- The equivalence between cubic polynomials and polynomials of degree at most three. -/
@[simps]
def equiv : Cubic R ≃ { p : R[X] // p.degree ≤ 3 } where
  toFun P := ⟨P.toPoly, degree_cubic_le⟩
  invFun f := ⟨coeff f 3, coeff f 2, coeff f 1, coeff f 0⟩
                   /-
                     R : Type u_1
                     S : Type u_2
                     F : Type u_3
                     K : Type u_4
                     P✝ Q : Cubic R
                     a b c d a' b' c' d' : R
                     inst✝ : Semiring R
                     P : Cubic R
                     ⊢ Eq ((fun f => { a := (↑f).coeff 3, b := (↑f).coeff 2, c := (↑f).coeff 1, d : …
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
  left_inv P := by ext <;> simp only [Subtype.coe_mk, coeffs]
                           /-
                             🎉 no goals
                           -/
  right_inv f := by
    -- Porting note: Added `simp only [Nat.succ_eq_add_one] <;> ring_nf`
    -- There's probably a better way to do this.
    /-
      R : Type u_1
      S : Type u_2
      F : Type u_3
      K : Type u_4
      P Q : Cubic R
      a b c d a' b' c' d' : R
      inst✝ : Semiring R
      f : Subtype fun p => LE.le p.degree 3
      ⊢ Eq ((fun P => ⟨P.toPoly, ⋯⟩) ((fun f => { a := (↑f).coeff 3, b := (↑f).coeff …
    -/
    ext (_ | _ | _ | _ | n) <;> simp only [Nat.succ_eq_add_one] <;> ring_nf
          /-
            case a.a.zero
            R : Type u_1
            S : Type u_2
            F : Type u_3
            K : Type u_4
            P Q : Cubic R
            a b c d a' b' c' d' : R
            inst✝ : Semiring R
            f : Subtype fun p => LE.le p.degree 3
            ⊢ Eq ({ a := (↑f).coeff 3, b := (↑f).coeff 2, c := (↑f).coeff 1, d := (↑f).coe …
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
          /-
            🎉 no goals
          -/
      <;> try simp only [coeffs]
    /-
      case a.a.succ.succ.succ.succ
      R : Type u_1
      S : Type u_2
      F : Type u_3
      K : Type u_4
      P Q : Cubic R
      a b c d a' b' c' d' : R
      inst✝ : Semiring R
      f : Subtype fun p => LE.le p.degree 3
      n : Nat
      ⊢ Eq ({ a := (↑f).coeff 3, b := (↑f).coeff 2, c := (↑f).coeff 1, d := (↑f).coe …
    -/
    have h3 : 3 < 4 + n := by linarith only
    rw [coeff_eq_zero h3,
      (degree_le_iff_coeff_zero (f : R[X]) 3).mp f.2 _ <| WithBot.coe_lt_coe.mpr (by exact h3)]


@[simp]
theorem degree_of_a_ne_zero (ha : P.a ≠ 0) : P.toPoly.degree = 3 :=
  degree_cubic ha


@[simp]
theorem degree_of_a_ne_zero' (ha : a ≠ 0) : (toPoly ⟨a, b, c, d⟩).degree = 3 :=
  degree_of_a_ne_zero ha


theorem degree_of_a_eq_zero (ha : P.a = 0) : P.toPoly.degree ≤ 2 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    ⊢ LE.le P.toPoly.degree 2
  -/
  simpa only [of_a_eq_zero ha] using degree_quadratic_le
  /-
    🎉 no goals
  -/


theorem degree_of_a_eq_zero' : (toPoly ⟨0, b, c, d⟩).degree ≤ 2 :=
  degree_of_a_eq_zero rfl


@[simp]
theorem degree_of_b_ne_zero (ha : P.a = 0) (hb : P.b ≠ 0) : P.toPoly.degree = 2 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Ne P.b 0
    ⊢ Eq P.toPoly.degree 2
  -/
  rw [of_a_eq_zero ha, degree_quadratic hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_of_b_ne_zero' (hb : b ≠ 0) : (toPoly ⟨0, b, c, d⟩).degree = 2 :=
  degree_of_b_ne_zero rfl hb


theorem degree_of_b_eq_zero (ha : P.a = 0) (hb : P.b = 0) : P.toPoly.degree ≤ 1 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    ⊢ LE.le P.toPoly.degree 1
  -/
  simpa only [of_b_eq_zero ha hb] using degree_linear_le
  /-
    🎉 no goals
  -/


theorem degree_of_b_eq_zero' : (toPoly ⟨0, 0, c, d⟩).degree ≤ 1 :=
  degree_of_b_eq_zero rfl rfl


@[simp]
theorem degree_of_c_ne_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c ≠ 0) : P.toPoly.degree = 1 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Ne P.c 0
    ⊢ Eq P.toPoly.degree 1
  -/
  rw [of_b_eq_zero ha hb, degree_linear hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_of_c_ne_zero' (hc : c ≠ 0) : (toPoly ⟨0, 0, c, d⟩).degree = 1 :=
  degree_of_c_ne_zero rfl rfl hc


theorem degree_of_c_eq_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) : P.toPoly.degree ≤ 0 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    ⊢ LE.le P.toPoly.degree 0
  -/
  simpa only [of_c_eq_zero ha hb hc] using degree_C_le
  /-
    🎉 no goals
  -/


theorem degree_of_c_eq_zero' : (toPoly ⟨0, 0, 0, d⟩).degree ≤ 0 :=
  degree_of_c_eq_zero rfl rfl rfl


@[simp]
theorem degree_of_d_ne_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) (hd : P.d ≠ 0) :
    P.toPoly.degree = 0 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    hd : Ne P.d 0
    ⊢ Eq P.toPoly.degree 0
  -/
  rw [of_c_eq_zero ha hb hc, degree_C hd]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_of_d_ne_zero' (hd : d ≠ 0) : (toPoly ⟨0, 0, 0, d⟩).degree = 0 :=
  degree_of_d_ne_zero rfl rfl rfl hd


@[simp]
theorem degree_of_d_eq_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) (hd : P.d = 0) :
    P.toPoly.degree = ⊥ := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    hd : Eq P.d 0
    ⊢ Eq P.toPoly.degree Bot.bot
  -/
  rw [of_d_eq_zero ha hb hc hd, degree_zero]
  /-
    🎉 no goals
  -/


theorem degree_of_d_eq_zero' : (⟨0, 0, 0, 0⟩ : Cubic R).toPoly.degree = ⊥ :=
  degree_of_d_eq_zero rfl rfl rfl rfl


@[simp]
theorem degree_of_zero : (0 : Cubic R).toPoly.degree = ⊥ :=
  degree_of_d_eq_zero'


@[simp]
theorem natDegree_of_a_ne_zero (ha : P.a ≠ 0) : P.toPoly.natDegree = 3 :=
  natDegree_cubic ha


@[simp]
theorem natDegree_of_a_ne_zero' (ha : a ≠ 0) : (toPoly ⟨a, b, c, d⟩).natDegree = 3 :=
  natDegree_of_a_ne_zero ha


theorem natDegree_of_a_eq_zero (ha : P.a = 0) : P.toPoly.natDegree ≤ 2 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    ⊢ LE.le P.toPoly.natDegree 2
  -/
  simpa only [of_a_eq_zero ha] using natDegree_quadratic_le
  /-
    🎉 no goals
  -/


theorem natDegree_of_a_eq_zero' : (toPoly ⟨0, b, c, d⟩).natDegree ≤ 2 :=
  natDegree_of_a_eq_zero rfl


@[simp]
theorem natDegree_of_b_ne_zero (ha : P.a = 0) (hb : P.b ≠ 0) : P.toPoly.natDegree = 2 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Ne P.b 0
    ⊢ Eq P.toPoly.natDegree 2
  -/
  rw [of_a_eq_zero ha, natDegree_quadratic hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_of_b_ne_zero' (hb : b ≠ 0) : (toPoly ⟨0, b, c, d⟩).natDegree = 2 :=
  natDegree_of_b_ne_zero rfl hb


theorem natDegree_of_b_eq_zero (ha : P.a = 0) (hb : P.b = 0) : P.toPoly.natDegree ≤ 1 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    ⊢ LE.le P.toPoly.natDegree 1
  -/
  simpa only [of_b_eq_zero ha hb] using natDegree_linear_le
  /-
    🎉 no goals
  -/


theorem natDegree_of_b_eq_zero' : (toPoly ⟨0, 0, c, d⟩).natDegree ≤ 1 :=
  natDegree_of_b_eq_zero rfl rfl


@[simp]
theorem natDegree_of_c_ne_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c ≠ 0) :
    P.toPoly.natDegree = 1 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Ne P.c 0
    ⊢ Eq P.toPoly.natDegree 1
  -/
  rw [of_b_eq_zero ha hb, natDegree_linear hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_of_c_ne_zero' (hc : c ≠ 0) : (toPoly ⟨0, 0, c, d⟩).natDegree = 1 :=
  natDegree_of_c_ne_zero rfl rfl hc


@[simp]
theorem natDegree_of_c_eq_zero (ha : P.a = 0) (hb : P.b = 0) (hc : P.c = 0) :
    P.toPoly.natDegree = 0 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝ : Semiring R
    ha : Eq P.a 0
    hb : Eq P.b 0
    hc : Eq P.c 0
    ⊢ Eq P.toPoly.natDegree 0
  -/
  rw [of_c_eq_zero ha hb hc, natDegree_C]
  /-
    🎉 no goals
  -/


theorem natDegree_of_c_eq_zero' : (toPoly ⟨0, 0, 0, d⟩).natDegree = 0 :=
  natDegree_of_c_eq_zero rfl rfl rfl


@[simp]
theorem natDegree_of_zero : (0 : Cubic R).toPoly.natDegree = 0 :=
  natDegree_of_c_eq_zero'


/-- Map a cubic polynomial across a semiring homomorphism. -/
def map (φ : R →+* S) (P : Cubic R) : Cubic S :=
  ⟨φ P.a, φ P.b, φ P.c, φ P.d⟩


theorem map_toPoly : (map φ P).toPoly = Polynomial.map φ P.toPoly := by
  /-
    R : Type u_1
    S : Type u_2
    P : Cubic R
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    φ : RingHom R S
    ⊢ Eq (Cubic.map φ P).toPoly (Polynomial.map φ P.toPoly)
  -/
  simp only [map, toPoly, map_C, map_X, Polynomial.map_add, Polynomial.map_mul, Polynomial.map_pow]
  /-
    🎉 no goals
  -/


/-- The roots of a cubic polynomial. -/
def roots [IsDomain R] (P : Cubic R) : Multiset R :=
  P.toPoly.roots


theorem map_roots [IsDomain S] : (map φ P).roots = (Polynomial.map φ P.toPoly).roots := by
  /-
    R : Type u_1
    S : Type u_2
    P : Cubic R
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    φ : RingHom R S
    inst✝ : IsDomain S
    ⊢ Eq (Cubic.map φ P).roots (Polynomial.map φ P.toPoly).roots
  -/
  rw [roots, map_toPoly]
  /-
    🎉 no goals
  -/


theorem mem_roots_iff [IsDomain R] (h0 : P.toPoly ≠ 0) (x : R) :
    x ∈ P.roots ↔ P.a * x ^ 3 + P.b * x ^ 2 + P.c * x + P.d = 0 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h0 : Ne P.toPoly 0
    x : R
    ⊢ Iff (Membership.mem P.roots x) (Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hM …
  -/
  rw [roots, mem_roots h0, IsRoot, toPoly]
  /-
    R : Type u_1
    P : Cubic R
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h0 : Ne P.toPoly 0
    x : R
    ⊢ Iff (Eq (Polynomial.eval x (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Poly …
  -/
  simp only [eval_C, eval_X, eval_add, eval_mul, eval_pow]
  /-
    🎉 no goals
  -/


theorem card_roots_le [IsDomain R] [DecidableEq R] : P.roots.toFinset.card ≤ 3 := by
  /-
    R : Type u_1
    P : Cubic R
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : DecidableEq R
    ⊢ LE.le P.roots.toFinset.card 3
  -/
  apply (toFinset_card_le P.toPoly.roots).trans
  /-
    R : Type u_1
    P : Cubic R
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : DecidableEq R
    ⊢ LE.le P.toPoly.roots.card 3
  -/
  by_cases hP : P.toPoly = 0
    /-
      case pos
      R : Type u_1
      P : Cubic R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      hP : Eq P.toPoly 0
      ⊢ LE.le P.toPoly.roots.card 3
    -/
  · exact (card_roots' P.toPoly).trans (by rw [hP, natDegree_zero]; exact zero_le 3)
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      P : Cubic R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      hP : Not (Eq P.toPoly 0)
      ⊢ LE.le P.toPoly.roots.card 3
    -/
  · exact WithBot.coe_le_coe.1 ((card_roots hP).trans degree_cubic_le)
    /-
      🎉 no goals
    -/


theorem splits_iff_card_roots (ha : P.a ≠ 0) :
    Splits φ P.toPoly ↔ Multiset.card (map φ P).roots = 3 := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    ha : Ne P.a 0
    ⊢ Iff (Polynomial.Splits φ P.toPoly) (Eq (Cubic.map φ P).roots.card 3)
  -/
  replace ha : (map φ P).a ≠ 0 := (_root_.map_ne_zero φ).mpr ha
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    ha : Ne (Cubic.map φ P).a 0
    ⊢ Iff (Polynomial.Splits φ P.toPoly) (Eq (Cubic.map φ P).roots.card 3)
  -/
  nth_rw 1 [← RingHom.id_comp φ]
  rw [roots, ← splits_map_iff, ← map_toPoly, Polynomial.splits_iff_card_roots,
    ← ((degree_eq_iff_natDegree_eq <| ne_zero_of_a_ne_zero ha).1 <| degree_of_a_ne_zero ha : _ = 3)]


theorem splits_iff_roots_eq_three (ha : P.a ≠ 0) :
    Splits φ P.toPoly ↔ ∃ x y z : K, (map φ P).roots = {x, y, z} := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    ha : Ne P.a 0
    ⊢ Iff (Polynomial.Splits φ P.toPoly) (Exists fun x => Exists fun y => Exists f …
  -/
  rw [splits_iff_card_roots ha, card_eq_three]
  /-
    🎉 no goals
  -/


theorem eq_prod_three_roots (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    (map φ P).toPoly = C (φ P.a) * (X - C x) * (X - C y) * (X - C z) := by
  rw [map_toPoly,
    eq_prod_roots_of_splits <|
      (splits_iff_roots_eq_three ha).mpr <| Exists.intro x <| Exists.intro y <| Exists.intro z h3,
    leadingCoeff_of_a_ne_zero ha, ← map_roots, h3]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (HMul.hMul (Polynomial.C (φ P.a)) (Multiset.map (fun a => HSub.hSub Polyn …
  -/
  change C (φ P.a) * ((X - C x) ::ₘ (X - C y) ::ₘ {X - C z}).prod = _
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (HMul.hMul (Polynomial.C (φ P.a)) (Multiset.cons (HSub.hSub Polynomial.X  …
  -/
  rw [prod_cons, prod_cons, prod_singleton, mul_assoc, mul_assoc]
  /-
    🎉 no goals
  -/


theorem eq_sum_three_roots (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    map φ P =
      ⟨φ P.a, φ P.a * -(x + y + z), φ P.a * (x * y + x * z + y * z), φ P.a * -(x * y * z)⟩ := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (Cubic.map φ P) { a := φ P.a, b := HMul.hMul (φ P.a) (Neg.neg (HAdd.hAdd  …
  -/
  apply_fun @toPoly _ _
    /-
      F : Type u_3
      K : Type u_4
      P : Cubic F
      inst✝¹ : Field F
      inst✝ : Field K
      φ : RingHom F K
      x y z : K
      ha : Ne P.a 0
      h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
      ⊢ Eq (Cubic.map φ P).toPoly { a := φ P.a, b := HMul.hMul (φ P.a) (Neg.neg (HAd …
    -/
  · rw [eq_prod_three_roots ha h3, C_mul_prod_X_sub_C_eq]
    /-
      🎉 no goals
    -/
    /-
      case inj
      F : Type u_3
      K : Type u_4
      P : Cubic F
      inst✝¹ : Field F
      inst✝ : Field K
      φ : RingHom F K
      x y z : K
      ha : Ne P.a 0
      h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
      ⊢ Function.Injective Cubic.toPoly
    -/
  · exact fun P Q ↦ (toPoly_injective P Q).mp
    /-
      🎉 no goals
    -/


theorem b_eq_three_roots (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    φ P.b = φ P.a * -(x + y + z) := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (φ P.b) (HMul.hMul (φ P.a) (Neg.neg (HAdd.hAdd (HAdd.hAdd x y) z)))
  -/
  injection eq_sum_three_roots ha h3
  /-
    🎉 no goals
  -/


theorem c_eq_three_roots (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    φ P.c = φ P.a * (x * y + x * z + y * z) := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (φ P.c) (HMul.hMul (φ P.a) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x y) (HMul.hM …
  -/
  injection eq_sum_three_roots ha h3
  /-
    🎉 no goals
  -/


theorem d_eq_three_roots (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    φ P.d = φ P.a * -(x * y * z) := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (φ P.d) (HMul.hMul (φ P.a) (Neg.neg (HMul.hMul (HMul.hMul x y) z)))
  -/
  injection eq_sum_three_roots ha h3
  /-
    🎉 no goals
  -/


/-- The discriminant of a cubic polynomial. -/
def disc {R : Type*} [Ring R] (P : Cubic R) : R :=
  P.b ^ 2 * P.c ^ 2 - 4 * P.a * P.c ^ 3 - 4 * P.b ^ 3 * P.d - 27 * P.a ^ 2 * P.d ^ 2 +
    18 * P.a * P.b * P.c * P.d


theorem disc_eq_prod_three_roots (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    φ P.disc = (φ P.a * φ P.a * (x - y) * (x - z) * (y - z)) ^ 2 := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (φ P.disc) (HPow.hPow (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (φ P.a) …
  -/
  simp only [disc, RingHom.map_add, RingHom.map_sub, RingHom.map_mul, map_pow, map_ofNat]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub (HMul.hMul (HPow.hPow (φ P.b) …
  -/
  rw [b_eq_three_roots ha h3, c_eq_three_roots ha h3, d_eq_three_roots ha h3]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HSub.hSub (HMul.hMul (HPow.hPow (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem disc_ne_zero_iff_roots_ne (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    P.disc ≠ 0 ↔ x ≠ y ∧ x ≠ z ∧ y ≠ z := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Iff (Ne P.disc 0) (And (Ne x y) (And (Ne x z) (Ne y z)))
  -/
  rw [← _root_.map_ne_zero φ, disc_eq_prod_three_roots ha h3, pow_two]
  simp_rw [mul_ne_zero_iff, sub_ne_zero, _root_.map_ne_zero, and_self_iff, and_iff_right ha,
    and_assoc]


theorem disc_ne_zero_iff_roots_nodup (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z}) :
    P.disc ≠ 0 ↔ (map φ P).roots.Nodup := by
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Iff (Ne P.disc 0) (Cubic.map φ P).roots.Nodup
  -/
  rw [disc_ne_zero_iff_roots_ne ha h3, h3]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Iff (And (Ne x y) (And (Ne x z) (Ne y z))) (Insert.insert x (Insert.insert y …
  -/
  change _ ↔ (x ::ₘ y ::ₘ {z}).Nodup
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Iff (And (Ne x y) (And (Ne x z) (Ne y z))) (Multiset.cons x (Multiset.cons y …
  -/
  rw [nodup_cons, nodup_cons, mem_cons, mem_singleton, mem_singleton]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Iff (And (Ne x y) (And (Ne x z) (Ne y z))) (And (Not (Or (Eq x y) (Eq x z))) …
  -/
  simp only [nodup_singleton]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝¹ : Field F
    inst✝ : Field K
    φ : RingHom F K
    x y z : K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Iff (And (Ne x y) (And (Ne x z) (Ne y z))) (And (Not (Or (Eq x y) (Eq x z))) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem card_roots_of_disc_ne_zero [DecidableEq K] (ha : P.a ≠ 0) (h3 : (map φ P).roots = {x, y, z})
    (hd : P.disc ≠ 0) : (map φ P).roots.toFinset.card = 3 := by
  rw [toFinset_card_of_nodup <| (disc_ne_zero_iff_roots_nodup ha h3).mp hd,
    ← splits_iff_card_roots ha, splits_iff_roots_eq_three ha]
  /-
    F : Type u_3
    K : Type u_4
    P : Cubic F
    inst✝² : Field F
    inst✝¹ : Field K
    φ : RingHom F K
    x y z : K
    inst✝ : DecidableEq K
    ha : Ne P.a 0
    h3 : Eq (Cubic.map φ P).roots (Insert.insert x (Insert.insert y (Singleton.sin …
    hd : Ne P.disc 0
    ⊢ Exists fun x => Exists fun y => Exists fun z => Eq (Cubic.map φ P).roots (In …
  -/
  exact ⟨x, ⟨y, ⟨z, h3⟩⟩⟩
  /-
    🎉 no goals
  -/


