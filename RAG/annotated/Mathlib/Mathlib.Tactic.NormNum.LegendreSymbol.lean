/-- The Jacobi symbol restricted to natural numbers in both arguments. -/
def jacobiSymNat (a b : ℕ) : ℤ :=
  jacobiSym a b


/-- Base cases: `b = 0`, `b = 1`, `a = 0`, `a = 1`. -/
theorem jacobiSymNat.zero_right (a : ℕ) : jacobiSymNat a 0 = 1 := by
  /-
    a : Nat
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a 0) 1
  -/
  rw [jacobiSymNat, jacobiSym.zero_right]
  /-
    🎉 no goals
  -/


theorem jacobiSymNat.one_right (a : ℕ) : jacobiSymNat a 1 = 1 := by
  /-
    a : Nat
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a 1) 1
  -/
  rw [jacobiSymNat, jacobiSym.one_right]
  /-
    🎉 no goals
  -/


theorem jacobiSymNat.zero_left (b : ℕ) (hb : Nat.beq (b / 2) 0 = false) : jacobiSymNat 0 b = 0 := by
  /-
    b : Nat
    hb : Eq ((HDiv.hDiv b 2).beq 0) Bool.false
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat 0 b) 0
  -/
  rw [jacobiSymNat, Nat.cast_zero, jacobiSym.zero_left ?_]
  calc
    1 < 2 * 1       := by decide
    _ ≤ 2 * (b / 2) :=
      Nat.mul_le_mul_left _ (Nat.succ_le.mpr (Nat.pos_of_ne_zero (Nat.ne_of_beq_eq_false hb)))
    _ ≤ b           := Nat.mul_div_le b 2


theorem jacobiSymNat.one_left (b : ℕ) : jacobiSymNat 1 b = 1 := by
  /-
    b : Nat
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat 1 b) 1
  -/
  rw [jacobiSymNat, Nat.cast_one, jacobiSym.one_left]
  /-
    🎉 no goals
  -/


/-- Turn a Legendre symbol into a Jacobi symbol. -/
theorem LegendreSym.to_jacobiSym (p : ℕ) (pp : Fact p.Prime) (a r : ℤ)
    (hr : IsInt (jacobiSym a p) r) : IsInt (legendreSym p a) r := by
  /-
    p : Nat
    pp : Fact (Nat.Prime p)
    a r : Int
    hr : Mathlib.Meta.NormNum.IsInt (jacobiSym a p) r
    ⊢ Mathlib.Meta.NormNum.IsInt (legendreSym p a) r
  -/
  rwa [@jacobiSym.legendreSym.to_jacobiSym p pp a]
  /-
    🎉 no goals
  -/


/-- The value depends only on the residue class of `a` mod `b`. -/
theorem JacobiSym.mod_left (a : ℤ) (b ab' : ℕ) (ab r b' : ℤ) (hb' : (b : ℤ) = b')
    (hab : a % b' = ab) (h : (ab' : ℤ) = ab) (hr : jacobiSymNat ab' b = r) : jacobiSym a b = r := by
  /-
    a : Int
    b ab' : Nat
    ab r b' : Int
    hb' : Eq (↑b) b'
    hab : Eq (HMod.hMod a b') ab
    h : Eq (↑ab') ab
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat ab' b) r
    ⊢ Eq (jacobiSym a b) r
  -/
  rw [← hr, jacobiSymNat, jacobiSym.mod_left, hb', hab, ← h]
  /-
    🎉 no goals
  -/


theorem jacobiSymNat.mod_left (a b ab : ℕ) (r : ℤ) (hab : a % b = ab) (hr : jacobiSymNat ab b = r) :
    jacobiSymNat a b = r := by
  /-
    a b ab : Nat
    r : Int
    hab : Eq (HMod.hMod a b) ab
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat ab b) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  rw [← hr, jacobiSymNat, jacobiSymNat, _root_.jacobiSym.mod_left a b, ← hab]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The symbol vanishes when both entries are even (and `b / 2 ≠ 0`). -/
theorem jacobiSymNat.even_even (a b : ℕ) (hb₀ : Nat.beq (b / 2) 0 = false) (ha : a % 2 = 0)
    (hb₁ : b % 2 = 0) : jacobiSymNat a b = 0 := by
  refine jacobiSym.eq_zero_iff.mpr
    ⟨ne_of_gt ((Nat.pos_of_ne_zero (Nat.ne_of_beq_eq_false hb₀)).trans_le (Nat.div_le_self b 2)),
      fun hf => ?_⟩
  /-
    a b : Nat
    hb₀ : Eq ((HDiv.hDiv b 2).beq 0) Bool.false
    ha : Eq (HMod.hMod a 2) 0
    hb₁ : Eq (HMod.hMod b 2) 0
    hf : Eq ((↑a).gcd ↑b) 1
    ⊢ False
  -/
  have h : 2 ∣ a.gcd b := Nat.dvd_gcd (Nat.dvd_of_mod_eq_zero ha) (Nat.dvd_of_mod_eq_zero hb₁)
  /-
    a b : Nat
    hb₀ : Eq ((HDiv.hDiv b 2).beq 0) Bool.false
    ha : Eq (HMod.hMod a 2) 0
    hb₁ : Eq (HMod.hMod b 2) 0
    hf : Eq ((↑a).gcd ↑b) 1
    h : Dvd.dvd 2 (a.gcd b)
    ⊢ False
  -/
  change 2 ∣ (a : ℤ).gcd b at h
  /-
    a b : Nat
    hb₀ : Eq ((HDiv.hDiv b 2).beq 0) Bool.false
    ha : Eq (HMod.hMod a 2) 0
    hb₁ : Eq (HMod.hMod b 2) 0
    hf : Eq ((↑a).gcd ↑b) 1
    h : Dvd.dvd 2 ((↑a).gcd ↑b)
    ⊢ False
  -/
  rw [hf, ← even_iff_two_dvd] at h
  /-
    a b : Nat
    hb₀ : Eq ((HDiv.hDiv b 2).beq 0) Bool.false
    ha : Eq (HMod.hMod a 2) 0
    hb₁ : Eq (HMod.hMod b 2) 0
    hf : Eq ((↑a).gcd ↑b) 1
    h : Even 1
    ⊢ False
  -/
  exact Nat.not_even_one h
  /-
    🎉 no goals
  -/


/-- When `a` is odd and `b` is even, we can replace `b` by `b / 2`. -/
theorem jacobiSymNat.odd_even (a b c : ℕ) (r : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 0) (hc : b / 2 = c)
    (hr : jacobiSymNat a c = r) : jacobiSymNat a b = r := by
  have ha' : legendreSym 2 a = 1 := by
    simp only [legendreSym.mod 2 a, Int.ofNat_mod_ofNat, ha]
    decide
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 1
    hb : Eq (HMod.hMod b 2) 0
    hc : Eq (HDiv.hDiv b 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat a c) r
    ha' : Eq (legendreSym 2 ↑a) 1
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  rcases eq_or_ne c 0 with (rfl | hc')
    /-
      case inl
      a b : Nat
      r : Int
      ha : Eq (HMod.hMod a 2) 1
      hb : Eq (HMod.hMod b 2) 0
      ha' : Eq (legendreSym 2 ↑a) 1
      hc : Eq (HDiv.hDiv b 2) 0
      hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat a 0) r
      ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
    -/
  · rw [← hr, Nat.eq_zero_of_dvd_of_div_eq_zero (Nat.dvd_of_mod_eq_zero hb) hc]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b c : Nat
      r : Int
      ha : Eq (HMod.hMod a 2) 1
      hb : Eq (HMod.hMod b 2) 0
      hc : Eq (HDiv.hDiv b 2) c
      hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat a c) r
      ha' : Eq (legendreSym 2 ↑a) 1
      hc' : Ne c 0
      ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
    -/
  · haveI : NeZero c := ⟨hc'⟩
    -- for `jacobiSym.mul_right`
    rwa [← Nat.mod_add_div b 2, hb, hc, Nat.zero_add, jacobiSymNat, jacobiSym.mul_right,
      ← jacobiSym.legendreSym.to_jacobiSym, ha', one_mul]


/-- If `a` is divisible by `4` and `b` is odd, then we can remove the factor `4` from `a`. -/
theorem jacobiSymNat.double_even (a b c : ℕ) (r : ℤ) (ha : a % 4 = 0) (hb : b % 2 = 1)
    (hc : a / 4 = c) (hr : jacobiSymNat c b = r) : jacobiSymNat a b = r := by
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 4) 0
    hb : Eq (HMod.hMod b 2) 1
    hc : Eq (HDiv.hDiv a 4) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  simp only [jacobiSymNat, ← hr, ← hc, Int.ofNat_ediv, Nat.cast_ofNat]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 4) 0
    hb : Eq (HMod.hMod b 2) 1
    hc : Eq (HDiv.hDiv a 4) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (HDiv.hDiv (↑a) 4) b)
  -/
  exact (jacobiSym.div_four_left (mod_cast ha) hb).symm
  /-
    🎉 no goals
  -/


/-- If `a` is even and `b` is odd, then we can remove a factor `2` from `a`,
but we may have to change the sign, depending on `b % 8`.
We give one version for each of the four odd residue classes mod `8`. -/
theorem jacobiSymNat.even_odd₁ (a b c : ℕ) (r : ℤ) (ha : a % 2 = 0) (hb : b % 8 = 1)
    (hc : a / 2 = c) (hr : jacobiSymNat c b = r) : jacobiSymNat a b = r := by
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 1
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  simp only [jacobiSymNat, ← hr, ← hc, Int.ofNat_ediv, Nat.cast_ofNat]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 1
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (HDiv.hDiv (↑a) 2) b)
  -/
  rw [← jacobiSym.even_odd (mod_cast ha), if_neg (by simp [hb])]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 1
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (HMod.hMod b 2) 1
  -/
  rw [← Nat.mod_mod_of_dvd, hb]; norm_num
                                 /-
                                   🎉 no goals
                                 -/


theorem jacobiSymNat.even_odd₇ (a b c : ℕ) (r : ℤ) (ha : a % 2 = 0) (hb : b % 8 = 7)
    (hc : a / 2 = c) (hr : jacobiSymNat c b = r) : jacobiSymNat a b = r := by
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 7
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  simp only [jacobiSymNat, ← hr, ← hc, Int.ofNat_ediv, Nat.cast_ofNat]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 7
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (HDiv.hDiv (↑a) 2) b)
  -/
  rw [← jacobiSym.even_odd (mod_cast ha), if_neg (by simp [hb])]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 7
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (HMod.hMod b 2) 1
  -/
  rw [← Nat.mod_mod_of_dvd, hb]; norm_num
                                 /-
                                   🎉 no goals
                                 -/


theorem jacobiSymNat.even_odd₃ (a b c : ℕ) (r : ℤ) (ha : a % 2 = 0) (hb : b % 8 = 3)
    (hc : a / 2 = c) (hr : jacobiSymNat c b = r) : jacobiSymNat a b = -r := by
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 3
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) (Neg.neg r)
  -/
  simp only [jacobiSymNat, ← hr, ← hc, Int.ofNat_ediv, Nat.cast_ofNat]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 3
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (jacobiSym (↑a) b) (Neg.neg (jacobiSym (HDiv.hDiv (↑a) 2) b))
  -/
  rw [← jacobiSym.even_odd (mod_cast ha), if_pos (by simp [hb])]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 3
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (HMod.hMod b 2) 1
  -/
  rw [← Nat.mod_mod_of_dvd, hb]; norm_num
                                 /-
                                   🎉 no goals
                                 -/


theorem jacobiSymNat.even_odd₅ (a b c : ℕ) (r : ℤ) (ha : a % 2 = 0) (hb : b % 8 = 5)
    (hc : a / 2 = c) (hr : jacobiSymNat c b = r) : jacobiSymNat a b = -r := by
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 5
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) (Neg.neg r)
  -/
  simp only [jacobiSymNat, ← hr, ← hc, Int.ofNat_ediv, Nat.cast_ofNat]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 5
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (jacobiSym (↑a) b) (Neg.neg (jacobiSym (HDiv.hDiv (↑a) 2) b))
  -/
  rw [← jacobiSym.even_odd (mod_cast ha), if_pos (by simp [hb])]
  /-
    a b c : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 0
    hb : Eq (HMod.hMod b 8) 5
    hc : Eq (HDiv.hDiv a 2) c
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat c b) r
    ⊢ Eq (HMod.hMod b 2) 1
  -/
  rw [← Nat.mod_mod_of_dvd, hb]; norm_num
                                 /-
                                   🎉 no goals
                                 -/


/-- Use quadratic reciproity to reduce to smaller `b`. -/
theorem jacobiSymNat.qr₁ (a b : ℕ) (r : ℤ) (ha : a % 4 = 1) (hb : b % 2 = 1)
    (hr : jacobiSymNat b a = r) : jacobiSymNat a b = r := by
  /-
    a b : Nat
    r : Int
    ha : Eq (HMod.hMod a 4) 1
    hb : Eq (HMod.hMod b 2) 1
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat b a) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  rwa [jacobiSymNat, jacobiSym.quadratic_reciprocity_one_mod_four ha (Nat.odd_iff.mpr hb)]
  /-
    🎉 no goals
  -/


theorem jacobiSymNat.qr₁_mod (a b ab : ℕ) (r : ℤ) (ha : a % 4 = 1) (hb : b % 2 = 1)
    (hab : b % a = ab) (hr : jacobiSymNat ab a = r) : jacobiSymNat a b = r :=
  jacobiSymNat.qr₁ _ _ _ ha hb <| jacobiSymNat.mod_left _ _ ab r hab hr


theorem jacobiSymNat.qr₁' (a b : ℕ) (r : ℤ) (ha : a % 2 = 1) (hb : b % 4 = 1)
    (hr : jacobiSymNat b a = r) : jacobiSymNat a b = r := by
  /-
    a b : Nat
    r : Int
    ha : Eq (HMod.hMod a 2) 1
    hb : Eq (HMod.hMod b 4) 1
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat b a) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) r
  -/
  rwa [jacobiSymNat, ← jacobiSym.quadratic_reciprocity_one_mod_four hb (Nat.odd_iff.mpr ha)]
  /-
    🎉 no goals
  -/


theorem jacobiSymNat.qr₁'_mod (a b ab : ℕ) (r : ℤ) (ha : a % 2 = 1) (hb : b % 4 = 1)
    (hab : b % a = ab) (hr : jacobiSymNat ab a = r) : jacobiSymNat a b = r :=
  jacobiSymNat.qr₁' _ _ _ ha hb <| jacobiSymNat.mod_left _ _ ab r hab hr


theorem jacobiSymNat.qr₃ (a b : ℕ) (r : ℤ) (ha : a % 4 = 3) (hb : b % 4 = 3)
    (hr : jacobiSymNat b a = r) : jacobiSymNat a b = -r := by
  /-
    a b : Nat
    r : Int
    ha : Eq (HMod.hMod a 4) 3
    hb : Eq (HMod.hMod b 4) 3
    hr : Eq (Mathlib.Meta.NormNum.jacobiSymNat b a) r
    ⊢ Eq (Mathlib.Meta.NormNum.jacobiSymNat a b) (Neg.neg r)
  -/
  rwa [jacobiSymNat, jacobiSym.quadratic_reciprocity_three_mod_four ha hb, neg_inj]
  /-
    🎉 no goals
  -/


theorem jacobiSymNat.qr₃_mod (a b ab : ℕ) (r : ℤ) (ha : a % 4 = 3) (hb : b % 4 = 3)
    (hab : b % a = ab) (hr : jacobiSymNat ab a = r) : jacobiSymNat a b = -r :=
  jacobiSymNat.qr₃ _ _ _ ha hb <| jacobiSymNat.mod_left _ _ ab r hab hr


theorem isInt_jacobiSym : {a na : ℤ} → {b nb : ℕ} → {r : ℤ} →
    IsInt a na → IsNat b nb → jacobiSym na nb = r → IsInt (jacobiSym a b) r
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨rfl⟩


theorem isInt_jacobiSymNat : {a na : ℕ} → {b nb : ℕ} → {r : ℤ} →
    IsNat a na → IsNat b nb → jacobiSymNat na nb = r → IsInt (jacobiSymNat a b) r
  | _, _, _, _, _, ⟨rfl⟩, ⟨rfl⟩, rfl => ⟨rfl⟩



/-- This evaluates `r := jacobiSymNat a b` recursively using quadratic reciprocity
and produces a proof term for the equality, assuming that `a < b` and `b` is odd. -/
partial def proveJacobiSymOdd (ea eb : Q(ℕ)) : (er : Q(ℤ)) × Q(jacobiSymNat $ea $eb = $er) :=
  match eb.natLit! with
  | 1 =>
    haveI : $eb =Q 1 := ⟨⟩
    ⟨mkRawIntLit 1, q(jacobiSymNat.one_right $ea)⟩
  | b =>
    match ea.natLit! with
    | 0 =>
      haveI : $ea =Q 0 := ⟨⟩
      have hb : Q(Nat.beq ($eb / 2) 0 = false) := (q(Eq.refl false) : Expr)
      ⟨mkRawIntLit 0, q(jacobiSymNat.zero_left $eb $hb)⟩
    | 1 =>
      haveI : $ea =Q 1 := ⟨⟩
      ⟨mkRawIntLit 1, q(jacobiSymNat.one_left $eb)⟩
    | a =>
      match a % 2 with
      | 0 =>
        match a % 4 with
        | 0 =>
          have ha : Q(Nat.mod $ea 4 = 0) := (q(Eq.refl 0) : Expr)
          have hb : Q(Nat.mod $eb 2 = 1) := (q(Eq.refl 1) : Expr)
          have ec : Q(ℕ) := mkRawNatLit (a / 4)
          have hc : Q(Nat.div $ea 4 = $ec) := (q(Eq.refl $ec) : Expr)
          have ⟨er, p⟩ := proveJacobiSymOdd ec eb
          ⟨er, q(jacobiSymNat.double_even $ea $eb $ec $er $ha $hb $hc $p)⟩
        | _ =>
          have ha : Q(Nat.mod $ea 2 = 0) := (q(Eq.refl 0) : Expr)
          have ec : Q(ℕ) := mkRawNatLit (a / 2)
          have hc : Q(Nat.div $ea 2 = $ec) := (q(Eq.refl $ec) : Expr)
          have ⟨er, p⟩ := proveJacobiSymOdd ec eb
          match b % 8 with
          | 1 =>
            have hb : Q(Nat.mod $eb 8 = 1) := (q(Eq.refl 1) : Expr)
            ⟨er, q(jacobiSymNat.even_odd₁ $ea $eb $ec $er $ha $hb $hc $p)⟩
          | 3 =>
            have er' := mkRawIntLit (-er.intLit!)
            have hb : Q(Nat.mod $eb 8 = 3) := (q(Eq.refl 3) : Expr)
            show (_ : Q(ℤ)) × Q(jacobiSymNat $ea $eb = -$er) from
              ⟨er', q(jacobiSymNat.even_odd₃ $ea $eb $ec $er $ha $hb $hc $p)⟩
          | 5 =>
            have er' := mkRawIntLit (-er.intLit!)
            haveI : $er' =Q -$er := ⟨⟩
            have hb : Q(Nat.mod $eb 8 = 5) := (q(Eq.refl 5) : Expr)
            ⟨er', q(jacobiSymNat.even_odd₅ $ea $eb $ec $er $ha $hb $hc $p)⟩
          | _ =>
            have hb : Q(Nat.mod $eb 8 = 7) := (q(Eq.refl 7) : Expr)
            ⟨er, q(jacobiSymNat.even_odd₇ $ea $eb $ec $er $ha $hb $hc $p)⟩
      | _ =>
        have eab : Q(ℕ) := mkRawNatLit (b % a)
        have hab : Q(Nat.mod $eb $ea = $eab) := (q(Eq.refl $eab) : Expr)
        have ⟨er, p⟩ := proveJacobiSymOdd eab ea
        match a % 4 with
        | 1 =>
          have ha : Q(Nat.mod $ea 4 = 1) := (q(Eq.refl 1) : Expr)
          have hb : Q(Nat.mod $eb 2 = 1) := (q(Eq.refl 1) : Expr)
          ⟨er, q(jacobiSymNat.qr₁_mod $ea $eb $eab $er $ha $hb $hab $p)⟩
        | _ =>
          match b % 4 with
          | 1 =>
            have ha : Q(Nat.mod $ea 2 = 1) := (q(Eq.refl 1) : Expr)
            have hb : Q(Nat.mod $eb 4 = 1) := (q(Eq.refl 1) : Expr)
            ⟨er, q(jacobiSymNat.qr₁'_mod $ea $eb $eab $er $ha $hb $hab $p)⟩
          | _ =>
            have er' := mkRawIntLit (-er.intLit!)
            haveI : $er' =Q -$er := ⟨⟩
            have ha : Q(Nat.mod $ea 4 = 3) := (q(Eq.refl 3) : Expr)
            have hb : Q(Nat.mod $eb 4 = 3) := (q(Eq.refl 3) : Expr)
            ⟨er', q(jacobiSymNat.qr₃_mod $ea $eb $eab $er $ha $hb $hab $p)⟩


/-- This evaluates `r := jacobiSymNat a b` and produces a proof term for the equality
by removing powers of `2` from `b` and then calling `proveJacobiSymOdd`. -/
partial def proveJacobiSymNat (ea eb : Q(ℕ)) : (er : Q(ℤ)) × Q(jacobiSymNat $ea $eb = $er) :=
  match eb.natLit! with
  | 0 =>
    haveI : $eb =Q 0 := ⟨⟩
    ⟨mkRawIntLit 1, q(jacobiSymNat.zero_right $ea)⟩
  | 1 =>
    haveI : $eb =Q 1 := ⟨⟩
    ⟨mkRawIntLit 1, q(jacobiSymNat.one_right $ea)⟩
  | b =>
    match b % 2 with
    | 0 =>
      match ea.natLit! with
      | 0 =>
        have hb : Q(Nat.beq ($eb / 2) 0 = false) := (q(Eq.refl false) : Expr)
        show (er : Q(ℤ)) × Q(jacobiSymNat 0 $eb = $er) from
          ⟨mkRawIntLit 0, q(jacobiSymNat.zero_left $eb $hb)⟩
      | 1 =>
        show (er : Q(ℤ)) × Q(jacobiSymNat 1 $eb = $er) from
          ⟨mkRawIntLit 1, q(jacobiSymNat.one_left $eb)⟩
      | a =>
        match a % 2 with
        | 0 =>
          have hb₀ : Q(Nat.beq ($eb / 2) 0 = false) := (q(Eq.refl false) : Expr)
          have ha : Q(Nat.mod $ea 2 = 0) := (q(Eq.refl 0) : Expr)
          have hb₁ : Q(Nat.mod $eb 2 = 0) := (q(Eq.refl 0) : Expr)
          ⟨mkRawIntLit 0, q(jacobiSymNat.even_even $ea $eb $hb₀ $ha $hb₁)⟩
        | _ =>
          have ha : Q(Nat.mod $ea 2 = 1) := (q(Eq.refl 1) : Expr)
          have hb : Q(Nat.mod $eb 2 = 0) := (q(Eq.refl 0) : Expr)
          have ec : Q(ℕ) := mkRawNatLit (b / 2)
          have hc : Q(Nat.div $eb 2 = $ec) := (q(Eq.refl $ec) : Expr)
          have ⟨er, p⟩ := proveJacobiSymOdd ea ec
          ⟨er, q(jacobiSymNat.odd_even $ea $eb $ec $er $ha $hb $hc $p)⟩
    | _ =>
      have a := ea.natLit!
      if b ≤ a then
        have eab : Q(ℕ) := mkRawNatLit (a % b)
        have hab : Q(Nat.mod $ea $eb = $eab) := (q(Eq.refl $eab) : Expr)
        have ⟨er, p⟩ := proveJacobiSymOdd eab eb
        ⟨er, q(jacobiSymNat.mod_left $ea $eb $eab $er $hab $p)⟩
      else
        proveJacobiSymOdd ea eb


/-- This evaluates `r := jacobiSym a b` and produces a proof term for the equality.
This is done by reducing to `r := jacobiSymNat (a % b) b`. -/
partial def proveJacobiSym (ea : Q(ℤ)) (eb : Q(ℕ)) : (er : Q(ℤ)) × Q(jacobiSym $ea $eb = $er) :=
  match eb.natLit! with
  | 0 =>
    haveI : $eb =Q 0 := ⟨⟩
    ⟨mkRawIntLit 1, q(jacobiSym.zero_right $ea)⟩
  | 1 =>
    haveI : $eb =Q 1 := ⟨⟩
    ⟨mkRawIntLit 1, q(jacobiSym.one_right $ea)⟩
  | b =>
    have eb' := mkRawIntLit b
    have hb' : Q(($eb : ℤ) = $eb') := (q(Eq.refl $eb') : Expr)
    have ab := ea.intLit! % b
    have eab := mkRawIntLit ab
    have hab : Q(Int.emod $ea $eb' = $eab) := (q(Eq.refl $eab) : Expr)
    have eab' : Q(ℕ) := mkRawNatLit ab.toNat
    have hab' : Q(($eab' : ℤ) = $eab) := (q(Eq.refl $eab) : Expr)
    have ⟨er, p⟩ := proveJacobiSymNat eab' eb
    ⟨er, q(JacobiSym.mod_left $ea $eb $eab' $eab $er $eb' $hb' $hab $hab' $p)⟩


/-- This is the `norm_num` plug-in that evaluates Jacobi symbols. -/
@[norm_num jacobiSym _ _]
def evalJacobiSym : NormNumExt where eval {u α} e := do
    let .app (.app _ (a : Q(ℤ))) (b : Q(ℕ)) ← Meta.whnfR e | failure
    let ⟨ea, pa⟩ ← deriveInt a _
    let ⟨eb, pb⟩ ← deriveNat b _
    haveI' : u =QL 0 := ⟨⟩ haveI' : $α =Q ℤ := ⟨⟩
    have ⟨er, pr⟩ := proveJacobiSym ea eb
    haveI' : $e =Q jacobiSym $a $b := ⟨⟩
    return .isInt _ er er.intLit! q(isInt_jacobiSym $pa $pb $pr)


/-- This is the `norm_num` plug-in that evaluates Jacobi symbols on natural numbers. -/
@[norm_num jacobiSymNat _ _]
def evalJacobiSymNat : NormNumExt where eval {u α} e := do
    let .app (.app _ (a : Q(ℕ))) (b : Q(ℕ)) ← Meta.whnfR e | failure
    let ⟨ea, pa⟩ ← deriveNat a _
    let ⟨eb, pb⟩ ← deriveNat b _
    haveI' : u =QL 0 := ⟨⟩ haveI' : $α =Q ℤ := ⟨⟩
    have ⟨er, pr⟩ := proveJacobiSymNat ea eb
    haveI' : $e =Q jacobiSymNat $a $b := ⟨⟩
    return .isInt _ er er.intLit!  q(isInt_jacobiSymNat $pa $pb $pr)


/-- This is the `norm_num` plug-in that evaluates Legendre symbols. -/
@[norm_num legendreSym _ _]
def evalLegendreSym : NormNumExt where eval {u α} e := do
    let .app (.app (.app _ (p : Q(ℕ))) (fp : Q(Fact (Nat.Prime $p)))) (a : Q(ℤ)) ← Meta.whnfR e |
      failure
    let ⟨ea, pa⟩ ← deriveInt a _
    let ⟨ep, pp⟩ ← deriveNat p _
    haveI' : u =QL 0 := ⟨⟩ haveI' : $α =Q ℤ := ⟨⟩
    have ⟨er, pr⟩ := proveJacobiSym ea ep
    haveI' : $e =Q legendreSym $p $a := ⟨⟩
    return .isInt _ er er.intLit!
      q(LegendreSym.to_jacobiSym $p $fp $a $er (isInt_jacobiSym $pa $pp $pr))


