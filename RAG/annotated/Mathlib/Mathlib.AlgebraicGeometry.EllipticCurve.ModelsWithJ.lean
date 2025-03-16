/-- The Weierstrass curve $Y^2 + Y = X^3$. It is of j-invariant 0 if it is an elliptic curve. -/
def ofJ0 : WeierstrassCurve R :=
  ⟨0, 0, 1, 0, 0⟩


lemma ofJ0_c₄ : (ofJ0 R).c₄ = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (WeierstrassCurve.ofJ0 R).c₄ 0
  -/
  rw [ofJ0, c₄, b₂, b₄]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HPow.hPow { a₁ := 0, a₂ := 0, a₃ := 1,  …
  -/
  norm_num1
  /-
    🎉 no goals
  -/


lemma ofJ0_Δ : (ofJ0 R).Δ = -27 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (WeierstrassCurve.ofJ0 R).Δ (-27)
  -/
  rw [ofJ0, Δ, b₂, b₄, b₆, b₈]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HAdd.hAd …
  -/
  norm_num1
  /-
    🎉 no goals
  -/


/-- The Weierstrass curve $Y^2 = X^3 + X$. It is of j-invariant 1728 if it is an elliptic curve. -/
def ofJ1728 : WeierstrassCurve R :=
  ⟨0, 0, 0, 1, 0⟩


lemma ofJ1728_c₄ : (ofJ1728 R).c₄ = -48 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (WeierstrassCurve.ofJ1728 R).c₄ (-48)
  -/
  rw [ofJ1728, c₄, b₂, b₄]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HPow.hPow { a₁ := 0, a₂ := 0, a₃ := 0,  …
  -/
  norm_num1
  /-
    🎉 no goals
  -/


lemma ofJ1728_Δ : (ofJ1728 R).Δ = -64 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (WeierstrassCurve.ofJ1728 R).Δ (-64)
  -/
  rw [ofJ1728, Δ, b₂, b₄, b₆, b₈]
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HAdd.hAd …
  -/
  norm_num1
  /-
    🎉 no goals
  -/


/-- The Weierstrass curve $Y^2 + (j - 1728)XY = X^3 - 36(j - 1728)^3X - (j - 1728)^5$.
It is a modification of the curve in [silverman2009], Chapter III, Proposition 1.4 (c) to avoid
denominators. It is of j-invariant j if it is an elliptic curve. -/
def ofJNe0Or1728 : WeierstrassCurve R :=
  ⟨j - 1728, 0, 0, -36 * (j - 1728) ^ 3, -(j - 1728) ^ 5⟩


lemma ofJNe0Or1728_c₄ : (ofJNe0Or1728 j).c₄ = j * (j - 1728) ^ 3 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    j : R
    ⊢ Eq (WeierstrassCurve.ofJNe0Or1728 j).c₄ (HMul.hMul j (HPow.hPow (HSub.hSub j …
  -/
  simp only [ofJNe0Or1728, c₄, b₂, b₄]
  /-
    R : Type u_1
    inst✝ : CommRing R
    j : R
    ⊢ Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HPow.hPow (HSub.hSub j 1728) 2) (HMul.h …
  -/
  ring1
  /-
    🎉 no goals
  -/


lemma ofJNe0Or1728_Δ : (ofJNe0Or1728 j).Δ = j ^ 2 * (j - 1728) ^ 9 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    j : R
    ⊢ Eq (WeierstrassCurve.ofJNe0Or1728 j).Δ (HMul.hMul (HPow.hPow j 2) (HPow.hPow …
  -/
  simp only [ofJNe0Or1728, Δ, b₂, b₄, b₆, b₈]
  /-
    R : Type u_1
    inst✝ : CommRing R
    j : R
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HAdd.hAd …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- When 3 is a unit, $Y^2 + Y = X^3$ is an elliptic curve.
It is of j-invariant 0 (see `WeierstrassCurve.ofJ0_j`). -/
instance [hu : Fact (IsUnit (3 : R))] : (ofJ0 R).IsElliptic := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j : R
    inst✝ : W.IsElliptic
    hu : Fact (IsUnit 3)
    ⊢ (WeierstrassCurve.ofJ0 R).IsElliptic
  -/
  rw [isElliptic_iff, ofJ0_Δ]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j : R
    inst✝ : W.IsElliptic
    hu : Fact (IsUnit 3)
    ⊢ IsUnit (-27)
  -/
  convert (hu.out.pow 3).neg
  /-
    case h.e'_3.h.e'_3
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j : R
    inst✝ : W.IsElliptic
    hu : Fact (IsUnit 3)
    ⊢ Eq 27 (HPow.hPow 3 3)
  -/
  norm_num1
  /-
    🎉 no goals
  -/

-- TODO: change to `[IsUnit ...]` once #17458 is merged

lemma ofJ0_j [Fact (IsUnit (3 : R))] : (ofJ0 R).j = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (IsUnit 3)
    ⊢ Eq (WeierstrassCurve.ofJ0 R).j 0
  -/
  rw [j, ofJ0_c₄]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (IsUnit 3)
    ⊢ Eq (HMul.hMul (↑(Inv.inv (WeierstrassCurve.ofJ0 R).Δ')) (HPow.hPow 0 3)) 0
  -/
  ring1
  /-
    🎉 no goals
  -/

-- TODO: change to `[IsUnit ...]` once #17458 is merged

/-- When 2 is a unit, $Y^2 = X^3 + X$ is an elliptic curve.
It is of j-invariant 1728 (see `WeierstrassCurve.ofJ1728_j`). -/
instance [hu : Fact (IsUnit (2 : R))] : (ofJ1728 R).IsElliptic := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j : R
    inst✝ : W.IsElliptic
    hu : Fact (IsUnit 2)
    ⊢ (WeierstrassCurve.ofJ1728 R).IsElliptic
  -/
  rw [isElliptic_iff, ofJ1728_Δ]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j : R
    inst✝ : W.IsElliptic
    hu : Fact (IsUnit 2)
    ⊢ IsUnit (-64)
  -/
  convert (hu.out.pow 6).neg
  /-
    case h.e'_3.h.e'_3
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j : R
    inst✝ : W.IsElliptic
    hu : Fact (IsUnit 2)
    ⊢ Eq 64 (HPow.hPow 2 6)
  -/
  norm_num1
  /-
    🎉 no goals
  -/

-- TODO: change to `[IsUnit ...]` once #17458 is merged

lemma ofJ1728_j [Fact (IsUnit (2 : R))] : (ofJ1728 R).j = 1728 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (IsUnit 2)
    ⊢ Eq (WeierstrassCurve.ofJ1728 R).j 1728
  -/
  rw [j, Units.inv_mul_eq_iff_eq_mul, ofJ1728_c₄, coe_Δ', ofJ1728_Δ]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (IsUnit 2)
    ⊢ Eq (HPow.hPow (-48) 3) (HMul.hMul (-64) 1728)
  -/
  norm_num1
  /-
    🎉 no goals
  -/


/-- When j and j - 1728 are both units,
$Y^2 + (j - 1728)XY = X^3 - 36(j - 1728)^3X - (j - 1728)^5$ is an elliptic curve.
It is of j-invariant j (see `WeierstrassCurve.ofJNe0Or1728_j`). -/
instance (j : R) [h1 : Fact (IsUnit j)] [h2 : Fact (IsUnit (j - 1728))] :
    (ofJNe0Or1728 j).IsElliptic := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j✝ : R
    inst✝ : W.IsElliptic
    j : R
    h1 : Fact (IsUnit j)
    h2 : Fact (IsUnit (HSub.hSub j 1728))
    ⊢ (WeierstrassCurve.ofJNe0Or1728 j).IsElliptic
  -/
  rw [isElliptic_iff, ofJNe0Or1728_Δ]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    j✝ : R
    inst✝ : W.IsElliptic
    j : R
    h1 : Fact (IsUnit j)
    h2 : Fact (IsUnit (HSub.hSub j 1728))
    ⊢ IsUnit (HMul.hMul (HPow.hPow j 2) (HPow.hPow (HSub.hSub j 1728) 9))
  -/
  exact (h1.out.pow 2).mul (h2.out.pow 9)
  /-
    🎉 no goals
  -/

-- TODO: change to `[IsUnit ...]` once #17458 is merged

lemma ofJNe0Or1728_j (j : R) [Fact (IsUnit j)] [Fact (IsUnit (j - 1728))] :
    (ofJNe0Or1728 j).j = j := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    j : R
    inst✝¹ : Fact (IsUnit j)
    inst✝ : Fact (IsUnit (HSub.hSub j 1728))
    ⊢ Eq (WeierstrassCurve.ofJNe0Or1728 j).j j
  -/
  rw [WeierstrassCurve.j, Units.inv_mul_eq_iff_eq_mul, ofJNe0Or1728_c₄, coe_Δ', ofJNe0Or1728_Δ]
  /-
    R : Type u_1
    inst✝² : CommRing R
    j : R
    inst✝¹ : Fact (IsUnit j)
    inst✝ : Fact (IsUnit (HSub.hSub j 1728))
    ⊢ Eq (HPow.hPow (HMul.hMul j (HPow.hPow (HSub.hSub j 1728) 3)) 3) (HMul.hMul ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- For any element j of a field $F$, there exists an elliptic curve over $F$
with j-invariant equal to j (see `WeierstrassCurve.ofJ_j`).
Its coefficients are given explicitly (see `WeierstrassCurve.ofJ0`, `WeierstrassCurve.ofJ1728`
and `WeierstrassCurve.ofJNe0Or1728`). -/
def ofJ : WeierstrassCurve F :=
  if j = 0 then if (3 : F) = 0 then ofJ1728 F else ofJ0 F
  else if j = 1728 then ofJ1728 F else ofJNe0Or1728 j


lemma ofJ_0_of_three_ne_zero (h3 : (3 : F) ≠ 0) : ofJ 0 = ofJ0 F := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    h3 : Ne 3 0
    ⊢ Eq (WeierstrassCurve.ofJ 0) (WeierstrassCurve.ofJ0 F)
  -/
  rw [ofJ, if_pos rfl, if_neg h3]
  /-
    🎉 no goals
  -/


lemma ofJ_0_of_three_eq_zero (h3 : (3 : F) = 0) : ofJ 0 = ofJ1728 F := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    h3 : Eq 3 0
    ⊢ Eq (WeierstrassCurve.ofJ 0) (WeierstrassCurve.ofJ1728 F)
  -/
  rw [ofJ, if_pos rfl, if_pos h3]
  /-
    🎉 no goals
  -/


lemma ofJ_0_of_two_eq_zero (h2 : (2 : F) = 0) : ofJ 0 = ofJ0 F := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    h2 : Eq 2 0
    ⊢ Eq (WeierstrassCurve.ofJ 0) (WeierstrassCurve.ofJ0 F)
  -/
  rw [ofJ, if_pos rfl, if_neg ((show (3 : F) = 1 by linear_combination h2) ▸ one_ne_zero)]
  /-
    🎉 no goals
  -/


lemma ofJ_1728_of_three_eq_zero (h3 : (3 : F) = 0) : ofJ 1728 = ofJ1728 F := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    h3 : Eq 3 0
    ⊢ Eq (WeierstrassCurve.ofJ 1728) (WeierstrassCurve.ofJ1728 F)
  -/
  rw [ofJ, if_pos (by linear_combination 576 * h3), if_pos h3]
  /-
    🎉 no goals
  -/


lemma ofJ_1728_of_two_ne_zero (h2 : (2 : F) ≠ 0) : ofJ 1728 = ofJ1728 F := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : DecidableEq F
    h2 : Ne 2 0
    ⊢ Eq (WeierstrassCurve.ofJ 1728) (WeierstrassCurve.ofJ1728 F)
  -/
  by_cases h3 : (3 : F) = 0
    /-
      case pos
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : DecidableEq F
      h2 : Ne 2 0
      h3 : Eq 3 0
      ⊢ Eq (WeierstrassCurve.ofJ 1728) (WeierstrassCurve.ofJ1728 F)
    -/
  · exact ofJ_1728_of_three_eq_zero h3
    /-
      🎉 no goals
    -/
  · rw [ofJ, show (1728 : F) = 2 ^ 6 * 3 ^ 3 by norm_num1,
      if_neg (mul_ne_zero (pow_ne_zero 6 h2) (pow_ne_zero 3 h3)), if_pos rfl]


lemma ofJ_1728_of_two_eq_zero (h2 : (2 : F) = 0) : ofJ 1728 = ofJ0 F := by
  rw [ofJ, if_pos (by linear_combination 864 * h2),
    if_neg ((show (3 : F) = 1 by linear_combination h2) ▸ one_ne_zero)]


lemma ofJ_ne_0_ne_1728 (h0 : j ≠ 0) (h1728 : j ≠ 1728) : ofJ j = ofJNe0Or1728 j := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    j : F
    inst✝ : DecidableEq F
    h0 : Ne j 0
    h1728 : Ne j 1728
    ⊢ Eq (WeierstrassCurve.ofJ j) (WeierstrassCurve.ofJNe0Or1728 j)
  -/
  rw [ofJ, if_neg h0, if_neg h1728]
  /-
    🎉 no goals
  -/


instance : (ofJ j).IsElliptic := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    W : WeierstrassCurve R
    j✝ : R
    inst✝² : W.IsElliptic
    F : Type u_2
    inst✝¹ : Field F
    j : F
    inst✝ : DecidableEq F
    ⊢ (WeierstrassCurve.ofJ j).IsElliptic
  -/
  by_cases h0 : j = 0
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      W : WeierstrassCurve R
      j✝ : R
      inst✝² : W.IsElliptic
      F : Type u_2
      inst✝¹ : Field F
      j : F
      inst✝ : DecidableEq F
      h0 : Eq j 0
      ⊢ (WeierstrassCurve.ofJ j).IsElliptic
    -/
  · by_cases h3 : (3 : F) = 0
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Eq 3 0
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
    · have := Fact.mk (isUnit_of_mul_eq_one (2 : F) 2 (by linear_combination h3))
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Eq 3 0
        this : Fact (IsUnit 2)
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
      rw [h0, ofJ_0_of_three_eq_zero h3]
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Eq 3 0
        this : Fact (IsUnit 2)
        ⊢ (WeierstrassCurve.ofJ1728 F).IsElliptic
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Not (Eq 3 0)
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
    · have := Fact.mk (Ne.isUnit h3)
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Not (Eq 3 0)
        this : Fact (IsUnit 3)
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
      rw [h0, ofJ_0_of_three_ne_zero h3]
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Not (Eq 3 0)
        this : Fact (IsUnit 3)
        ⊢ (WeierstrassCurve.ofJ0 F).IsElliptic
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      W : WeierstrassCurve R
      j✝ : R
      inst✝² : W.IsElliptic
      F : Type u_2
      inst✝¹ : Field F
      j : F
      inst✝ : DecidableEq F
      h0 : Not (Eq j 0)
      ⊢ (WeierstrassCurve.ofJ j).IsElliptic
    -/
  · by_cases h1728 : j = 1728
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
    · have h2 : (2 : F) ≠ 0 := fun h ↦ h0 (by linear_combination h1728 + 864 * h)
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        h2 : Ne 2 0
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
      have := Fact.mk h2.isUnit
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        h2 : Ne 2 0
        this : Fact (IsUnit 2)
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
      rw [h1728, ofJ_1728_of_two_ne_zero h2]
      /-
        case pos
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        h2 : Ne 2 0
        this : Fact (IsUnit 2)
        ⊢ (WeierstrassCurve.ofJ1728 F).IsElliptic
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
    · have := Fact.mk (Ne.isUnit h0)
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        this : Fact (IsUnit j)
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
      have := Fact.mk (sub_ne_zero.2 h1728).isUnit
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        this✝ : Fact (IsUnit j)
        this : Fact (IsUnit (HSub.hSub j 1728))
        ⊢ (WeierstrassCurve.ofJ j).IsElliptic
      -/
      rw [ofJ_ne_0_ne_1728 j h0 h1728]
      /-
        case neg
        R : Type u_1
        inst✝³ : CommRing R
        W : WeierstrassCurve R
        j✝ : R
        inst✝² : W.IsElliptic
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        this✝ : Fact (IsUnit j)
        this : Fact (IsUnit (HSub.hSub j 1728))
        ⊢ (WeierstrassCurve.ofJNe0Or1728 j).IsElliptic
      -/
      infer_instance
      /-
        🎉 no goals
      -/


lemma ofJ_j : (ofJ j).j = j := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    j : F
    inst✝ : DecidableEq F
    ⊢ Eq (WeierstrassCurve.ofJ j).j j
  -/
  by_cases h0 : j = 0
    /-
      case pos
      F : Type u_2
      inst✝¹ : Field F
      j : F
      inst✝ : DecidableEq F
      h0 : Eq j 0
      ⊢ Eq (WeierstrassCurve.ofJ j).j j
    -/
  · by_cases h3 : (3 : F) = 0
      /-
        case pos
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Eq 3 0
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
    · have := Fact.mk (isUnit_of_mul_eq_one (2 : F) 2 (by linear_combination h3))
      /-
        case pos
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Eq 3 0
        this : Fact (IsUnit 2)
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
      simp_rw [h0, ofJ_0_of_three_eq_zero h3, ofJ1728_j]
      /-
        case pos
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Eq 3 0
        this : Fact (IsUnit 2)
        ⊢ Eq 1728 0
      -/
      linear_combination 576 * h3
      /-
        🎉 no goals
      -/
      /-
        case neg
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Not (Eq 3 0)
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
    · have := Fact.mk (Ne.isUnit h3)
      /-
        case neg
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Eq j 0
        h3 : Not (Eq 3 0)
        this : Fact (IsUnit 3)
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
      simp_rw [h0, ofJ_0_of_three_ne_zero h3, ofJ0_j]
      /-
        🎉 no goals
      -/
    /-
      case neg
      F : Type u_2
      inst✝¹ : Field F
      j : F
      inst✝ : DecidableEq F
      h0 : Not (Eq j 0)
      ⊢ Eq (WeierstrassCurve.ofJ j).j j
    -/
  · by_cases h1728 : j = 1728
      /-
        case pos
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
    · have h2 : (2 : F) ≠ 0 := fun h ↦ h0 (by linear_combination h1728 + 864 * h)
      /-
        case pos
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        h2 : Ne 2 0
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
      have := Fact.mk h2.isUnit
      /-
        case pos
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Eq j 1728
        h2 : Ne 2 0
        this : Fact (IsUnit 2)
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
      simp_rw [h1728, ofJ_1728_of_two_ne_zero h2, ofJ1728_j]
      /-
        🎉 no goals
      -/
      /-
        case neg
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
    · have := Fact.mk (Ne.isUnit h0)
      /-
        case neg
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        this : Fact (IsUnit j)
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
      have := Fact.mk (sub_ne_zero.2 h1728).isUnit
      /-
        case neg
        F : Type u_2
        inst✝¹ : Field F
        j : F
        inst✝ : DecidableEq F
        h0 : Not (Eq j 0)
        h1728 : Not (Eq j 1728)
        this✝ : Fact (IsUnit j)
        this : Fact (IsUnit (HSub.hSub j 1728))
        ⊢ Eq (WeierstrassCurve.ofJ j).j j
      -/
      simp_rw [ofJ_ne_0_ne_1728 j h0 h1728, ofJNe0Or1728_j]
      /-
        🎉 no goals
      -/


instance : Inhabited { W : WeierstrassCurve F // W.IsElliptic } := ⟨⟨ofJ 37, inferInstance⟩⟩


