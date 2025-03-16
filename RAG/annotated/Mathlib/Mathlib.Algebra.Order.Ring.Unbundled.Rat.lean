@[simp] lemma divInt_nonneg_iff_of_pos_right {a b : ℤ} (hb : 0 < b) : 0 ≤ a /. b ↔ 0 ≤ a := by
  /-
    a b : Int
    hb : LT.lt 0 b
    ⊢ Iff (LE.le 0 (Rat.divInt a b)) (LE.le 0 a)
  -/
  cases' hab : a /. b with n d hd hnd
  /-
    case mk'
    a b : Int
    hb : LT.lt 0 b
    n : Int
    d : Nat
    hd : Ne d 0
    hnd : n.natAbs.Coprime d
    hab : Eq (Rat.divInt a b) { num := n, den := d, den_nz := hd, reduced := hnd }
    ⊢ Iff (LE.le 0 { num := n, den := d, den_nz := hd, reduced := hnd }) (LE.le 0 a)
  -/
  rw [mk'_eq_divInt, divInt_eq_iff hb.ne' (mod_cast hd)] at hab
  rw [← num_nonneg, ← Int.mul_nonneg_iff_of_pos_right hb, ← hab,
    Int.mul_nonneg_iff_of_pos_right (mod_cast Nat.pos_of_ne_zero hd)]


@[simp] lemma divInt_nonneg {a b : ℤ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a /. b := by
  /-
    a b : Int
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le 0 (Rat.divInt a b)
  -/
  obtain rfl | hb := hb.eq_or_lt
    /-
      case inl
      a : Int
      ha : LE.le 0 a
      hb : LE.le 0 0
      ⊢ LE.le 0 (Rat.divInt a 0)
    -/
  · simp
    /-
      case inl
      a : Int
      ha : LE.le 0 a
      hb : LE.le 0 0
      ⊢ LE.le 0 0
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Int
    ha : LE.le 0 a
    hb✝ : LE.le 0 b
    hb : LT.lt 0 b
    ⊢ LE.le 0 (Rat.divInt a b)
  -/
  rwa [divInt_nonneg_iff_of_pos_right hb]
  /-
    🎉 no goals
  -/


@[simp] lemma mkRat_nonneg {a : ℤ} (ha : 0 ≤ a) (b : ℕ) : 0 ≤ mkRat a b := by
  /-
    a : Int
    ha : LE.le 0 a
    b : Nat
    ⊢ LE.le 0 (mkRat a b)
  -/
  simpa using divInt_nonneg ha (Int.natCast_nonneg _)
  /-
    🎉 no goals
  -/


theorem ofScientific_nonneg (m : ℕ) (s : Bool) (e : ℕ) :
    0 ≤ Rat.ofScientific m s e := by
  /-
    m : Nat
    s : Bool
    e : Nat
    ⊢ LE.le 0 (Rat.ofScientific m s e)
  -/
  rw [Rat.ofScientific]
  /-
    m : Nat
    s : Bool
    e : Nat
    ⊢ LE.le 0 (ite (Eq s Bool.true) (Rat.normalize (↑m) (HPow.hPow 10 e) ⋯) ↑(HMul …
  -/
  cases s
    /-
      case false
      m e : Nat
      ⊢ LE.le 0 (ite (Eq Bool.false Bool.true) (Rat.normalize (↑m) (HPow.hPow 10 e)  …
    -/
  · rw [if_neg (by decide)]
    /-
      case false
      m e : Nat
      ⊢ LE.le 0 ↑(HMul.hMul m (HPow.hPow 10 e))
    -/
    refine num_nonneg.mp ?_
    /-
      case false
      m e : Nat
      ⊢ LE.le 0 (↑(HMul.hMul m (HPow.hPow 10 e))).num
    -/
    rw [num_natCast]
    /-
      case false
      m e : Nat
      ⊢ LE.le 0 ↑(HMul.hMul m (HPow.hPow 10 e))
    -/
    exact Int.natCast_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case true
      m e : Nat
      ⊢ LE.le 0 (ite (Eq Bool.true Bool.true) (Rat.normalize (↑m) (HPow.hPow 10 e) ⋯ …
    -/
  · rw [if_pos rfl, normalize_eq_mkRat]
    /-
      case true
      m e : Nat
      ⊢ LE.le 0 (mkRat (↑m) (HPow.hPow 10 e))
    -/
    exact Rat.mkRat_nonneg (Int.natCast_nonneg _) _
    /-
      🎉 no goals
    -/


instance _root_.NNRatCast.toOfScientific {K} [NNRatCast K] : OfScientific K where
  ofScientific (m : ℕ) (b : Bool) (d : ℕ) :=
    NNRat.cast ⟨Rat.ofScientific m b d, ofScientific_nonneg m b d⟩


/-- Casting a scientific literal via `ℚ≥0` is the same as casting directly. -/
@[simp, norm_cast]
theorem _root_.NNRat.cast_ofScientific {K} [NNRatCast K] (m : ℕ) (s : Bool) (e : ℕ) :
    (OfScientific.ofScientific m s e : ℚ≥0) = (OfScientific.ofScientific m s e : K) :=
  rfl


protected lemma add_nonneg : 0 ≤ a → 0 ≤ b → 0 ≤ a + b :=
  numDenCasesOn' a fun n₁ d₁ h₁ ↦ numDenCasesOn' b fun n₂ d₂ h₂ ↦ by
    /-
      a b : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      ⊢ LE.le 0 (Rat.divInt n₁ ↑d₁) → LE.le 0 (Rat.divInt n₂ ↑d₂) → LE.le 0 (HAdd.hA …
    -/
    have d₁0 : 0 < (d₁ : ℤ) := mod_cast Nat.pos_of_ne_zero h₁
    /-
      a b : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      d₁0 : LT.lt 0 ↑d₁
      ⊢ LE.le 0 (Rat.divInt n₁ ↑d₁) → LE.le 0 (Rat.divInt n₂ ↑d₂) → LE.le 0 (HAdd.hA …
    -/
    have d₂0 : 0 < (d₂ : ℤ) := mod_cast Nat.pos_of_ne_zero h₂
    simp only [d₁0, d₂0, h₁, h₂, Int.mul_pos, divInt_nonneg_iff_of_pos_right, divInt_add_divInt, Ne,
      Nat.cast_eq_zero, not_false_iff]
    /-
      a b : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      d₁0 : LT.lt 0 ↑d₁
      d₂0 : LT.lt 0 ↑d₂
      ⊢ LE.le 0 n₁ → LE.le 0 n₂ → LE.le 0 (HAdd.hAdd (HMul.hMul n₁ ↑d₂) (HMul.hMul n …
    -/
    intro n₁0 n₂0
    /-
      a b : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      d₁0 : LT.lt 0 ↑d₁
      d₂0 : LT.lt 0 ↑d₂
      n₁0 : LE.le 0 n₁
      n₂0 : LE.le 0 n₂
      ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul n₁ ↑d₂) (HMul.hMul n₂ ↑d₁))
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
    apply Int.add_nonneg <;> apply Int.mul_nonneg <;> · first | assumption | apply Int.ofNat_zero_le
                                                        /-
                                                          🎉 no goals
                                                        -/


protected lemma mul_nonneg : 0 ≤ a → 0 ≤ b → 0 ≤ a * b :=
  numDenCasesOn' a fun n₁ d₁ h₁ =>
    numDenCasesOn' b fun n₂ d₂ h₂ => by
      /-
        a b : Rat
        n₁ : Int
        d₁ : Nat
        h₁ : Ne d₁ 0
        n₂ : Int
        d₂ : Nat
        h₂ : Ne d₂ 0
        ⊢ LE.le 0 (Rat.divInt n₁ ↑d₁) → LE.le 0 (Rat.divInt n₂ ↑d₂) → LE.le 0 (HMul.hM …
      -/
      have d₁0 : 0 < (d₁ : ℤ) := mod_cast Nat.pos_of_ne_zero h₁
      /-
        a b : Rat
        n₁ : Int
        d₁ : Nat
        h₁ : Ne d₁ 0
        n₂ : Int
        d₂ : Nat
        h₂ : Ne d₂ 0
        d₁0 : LT.lt 0 ↑d₁
        ⊢ LE.le 0 (Rat.divInt n₁ ↑d₁) → LE.le 0 (Rat.divInt n₂ ↑d₂) → LE.le 0 (HMul.hM …
      -/
      have d₂0 : 0 < (d₂ : ℤ) := mod_cast Nat.pos_of_ne_zero h₂
      simp only [d₁0, d₂0, Int.mul_pos, divInt_nonneg_iff_of_pos_right,
        divInt_mul_divInt _ _ d₁0.ne' d₂0.ne']
      /-
        a b : Rat
        n₁ : Int
        d₁ : Nat
        h₁ : Ne d₁ 0
        n₂ : Int
        d₂ : Nat
        h₂ : Ne d₂ 0
        d₁0 : LT.lt 0 ↑d₁
        d₂0 : LT.lt 0 ↑d₂
        ⊢ LE.le 0 n₁ → LE.le 0 n₂ → LE.le 0 (HMul.hMul n₁ n₂)
      -/
      apply Int.mul_nonneg
      /-
        🎉 no goals
      -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO can this be shortened?

protected theorem le_iff_sub_nonneg (a b : ℚ) : a ≤ b ↔ 0 ≤ b - a :=
  numDenCasesOn'' a fun na da ha hared =>
    numDenCasesOn'' b fun nb db hb hbred => by
      /-
        a b : Rat
        na : Int
        da : Nat
        ha : Ne da 0
        hared : na.natAbs.Coprime da
        nb : Int
        db : Nat
        hb : Ne db 0
        hbred : nb.natAbs.Coprime db
        ⊢ Iff (LE.le { num := na, den := da, den_nz := ha, reduced := hared } { num := …
      -/
      change Rat.blt _ _ = false ↔ _
      /-
        a b : Rat
        na : Int
        da : Nat
        ha : Ne da 0
        hared : na.natAbs.Coprime da
        nb : Int
        db : Nat
        hb : Ne db 0
        hbred : nb.natAbs.Coprime db
        ⊢ Iff (Eq ({ num := nb, den := db, den_nz := hb, reduced := hbred }.blt { num  …
      -/
      unfold Rat.blt
      simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.ite_eq_false_distrib,
        decide_eq_false_iff_not, not_lt, ite_eq_left_iff, not_and, not_le, ← num_nonneg]
      /-
        a b : Rat
        na : Int
        da : Nat
        ha : Ne da 0
        hared : na.natAbs.Coprime da
        nb : Int
        db : Nat
        hb : Ne db 0
        hbred : nb.natAbs.Coprime db
        ⊢ Iff (ite (And (LT.lt nb 0) (LE.le 0 na)) (Eq Bool.true Bool.false) (ite (Eq  …
      -/
      split_ifs with h h'
        /-
          case pos
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : And (LT.lt nb 0) (LE.le 0 na)
          ⊢ Iff False (LE.le 0 (HSub.hSub { num := nb, den := db, den_nz := hb, reduced  …
        -/
      · rw [Rat.sub_def]
        /-
          case pos
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : And (LT.lt nb 0) (LE.le 0 na)
          ⊢ Iff False (LE.le 0 (Rat.normalize (HSub.hSub (HMul.hMul { num := nb, den :=  …
        -/
        simp only [false_iff, not_le, reduceCtorEq]
        /-
          case pos
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : And (LT.lt nb 0) (LE.le 0 na)
          ⊢ LT.lt (Rat.normalize (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) (HMul …
        -/
        simp only [normalize_eq]
        /-
          case pos
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : And (LT.lt nb 0) (LE.le 0 na)
          ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) ↑((HSub.h …
        -/
        apply Int.ediv_neg'
          /-
            case pos.Ha
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : And (LT.lt nb 0) (LE.le 0 na)
            ⊢ LT.lt (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) 0
          -/
        · rw [sub_neg]
          /-
            case pos.Ha
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : And (LT.lt nb 0) (LE.le 0 na)
            ⊢ LT.lt (HMul.hMul nb ↑da) (HMul.hMul na ↑db)
          -/
          apply lt_of_lt_of_le
            /-
              case pos.Ha.hab
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : And (LT.lt nb 0) (LE.le 0 na)
              ⊢ LT.lt (HMul.hMul nb ↑da) ?pos.Ha.b✝
            -/
          · apply Int.mul_neg_of_neg_of_pos h.1
            /-
              case pos.Ha.hab
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : And (LT.lt nb 0) (LE.le 0 na)
              ⊢ LT.lt 0 ↑da
            -/
            rwa [Int.natCast_pos, Nat.pos_iff_ne_zero]
            /-
              🎉 no goals
            -/
            /-
              case pos.Ha.hbc
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : And (LT.lt nb 0) (LE.le 0 na)
              ⊢ LE.le 0 (HMul.hMul na ↑db)
            -/
          · apply Int.mul_nonneg h.2 (Int.natCast_nonneg _)
            /-
              🎉 no goals
            -/
          /-
            case pos.Hb
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : And (LT.lt nb 0) (LE.le 0 na)
            ⊢ LT.lt 0 ↑((HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)).natAbs.gcd (HMul …
          -/
        · simp only [Int.natCast_pos, Nat.pos_iff_ne_zero]
          /-
            case pos.Hb
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : And (LT.lt nb 0) (LE.le 0 na)
            ⊢ Ne ((HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)).natAbs.gcd (HMul.hMul  …
          -/
          exact Nat.gcd_ne_zero_right (Nat.mul_ne_zero hb ha)
          /-
            🎉 no goals
          -/
        /-
          case pos
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : Not (And (LT.lt nb 0) (LE.le 0 na))
          h' : Eq nb 0
          ⊢ Iff (LE.le na 0) (LE.le 0 (HSub.hSub { num := nb, den := db, den_nz := hb, r …
        -/
      · simp [h']
        /-
          🎉 no goals
        -/
        /-
          case neg
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : Not (And (LT.lt nb 0) (LE.le 0 na))
          h' : Not (Eq nb 0)
          ⊢ Iff ((LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da) …
        -/
      · simp only [Rat.sub_def, normalize_eq]
        /-
          case neg
          a b : Rat
          na : Int
          da : Nat
          ha : Ne da 0
          hared : na.natAbs.Coprime da
          nb : Int
          db : Nat
          hb : Ne db 0
          hbred : nb.natAbs.Coprime db
          h : Not (And (LT.lt nb 0) (LE.le 0 na))
          h' : Not (Eq nb 0)
          ⊢ Iff ((LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da) …
        -/
        refine ⟨fun H => ?_, fun H _ => ?_⟩
          /-
            case neg.refine_1
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
            ⊢ LE.le 0 (HDiv.hDiv (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) ↑((HSub …
          -/
        · refine Int.ediv_nonneg ?_ (Int.natCast_nonneg _)
          /-
            case neg.refine_1
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
            ⊢ LE.le 0 (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db))
          -/
          rw [Int.sub_nonneg]
          /-
            case neg.refine_1
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
            ⊢ LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
          -/
          obtain hb|hb := Ne.lt_or_lt h'
            /-
              case neg.refine_1.inl
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb✝ : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : Not (And (LT.lt nb 0) (LE.le 0 na))
              h' : Not (Eq nb 0)
              H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
              hb : LT.lt nb 0
              ⊢ LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
            -/
          · apply H
            /-
              case neg.refine_1.inl
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb✝ : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : Not (And (LT.lt nb 0) (LE.le 0 na))
              h' : Not (Eq nb 0)
              H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
              hb : LT.lt nb 0
              ⊢ LT.lt 0 nb → LT.lt 0 na
            -/
            intro H'
            /-
              case neg.refine_1.inl
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb✝ : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : Not (And (LT.lt nb 0) (LE.le 0 na))
              h' : Not (Eq nb 0)
              H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
              hb : LT.lt nb 0
              H' : LT.lt 0 nb
              ⊢ LT.lt 0 na
            -/
            exact (hb.trans H').false.elim
            /-
              🎉 no goals
            -/
            /-
              case neg.refine_1.inr
              a b : Rat
              na : Int
              da : Nat
              ha : Ne da 0
              hared : na.natAbs.Coprime da
              nb : Int
              db : Nat
              hb✝ : Ne db 0
              hbred : nb.natAbs.Coprime db
              h : Not (And (LT.lt nb 0) (LE.le 0 na))
              h' : Not (Eq nb 0)
              H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
              hb : LT.lt 0 nb
              ⊢ LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
            -/
          · obtain ha|ha := le_or_lt na 0
              /-
                case neg.refine_1.inr.inl
                a b : Rat
                na : Int
                da : Nat
                ha✝ : Ne da 0
                hared : na.natAbs.Coprime da
                nb : Int
                db : Nat
                hb✝ : Ne db 0
                hbred : nb.natAbs.Coprime db
                h : Not (And (LT.lt nb 0) (LE.le 0 na))
                h' : Not (Eq nb 0)
                H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
                hb : LT.lt 0 nb
                ha : LE.le na 0
                ⊢ LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
              -/
            · apply le_trans <| Int.mul_nonpos_of_nonpos_of_nonneg ha (Int.natCast_nonneg _)
              /-
                case neg.refine_1.inr.inl
                a b : Rat
                na : Int
                da : Nat
                ha✝ : Ne da 0
                hared : na.natAbs.Coprime da
                nb : Int
                db : Nat
                hb✝ : Ne db 0
                hbred : nb.natAbs.Coprime db
                h : Not (And (LT.lt nb 0) (LE.le 0 na))
                h' : Not (Eq nb 0)
                H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
                hb : LT.lt 0 nb
                ha : LE.le na 0
                ⊢ LE.le 0 (HMul.hMul nb ↑da)
              -/
              exact Int.mul_nonneg hb.le (Int.natCast_nonneg _)
              /-
                🎉 no goals
              -/
              /-
                case neg.refine_1.inr.inr
                a b : Rat
                na : Int
                da : Nat
                ha✝ : Ne da 0
                hared : na.natAbs.Coprime da
                nb : Int
                db : Nat
                hb✝ : Ne db 0
                hbred : nb.natAbs.Coprime db
                h : Not (And (LT.lt nb 0) (LE.le 0 na))
                h' : Not (Eq nb 0)
                H : (LT.lt 0 nb → LT.lt 0 na) → LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
                hb : LT.lt 0 nb
                ha : LT.lt 0 na
                ⊢ LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
              -/
            · exact H (fun _ => ha)
              /-
                🎉 no goals
              -/
          /-
            case neg.refine_2
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            H : LE.le 0 (HDiv.hDiv (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) ↑((HS …
            x✝ : LT.lt 0 nb → LT.lt 0 na
            ⊢ LE.le (HMul.hMul na ↑db) (HMul.hMul nb ↑da)
          -/
        · rw [← Int.sub_nonneg]
          /-
            case neg.refine_2
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            H : LE.le 0 (HDiv.hDiv (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) ↑((HS …
            x✝ : LT.lt 0 nb → LT.lt 0 na
            ⊢ LE.le 0 (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db))
          -/
          contrapose! H
          /-
            case neg.refine_2
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            x✝ : LT.lt 0 nb → LT.lt 0 na
            H : LT.lt (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) 0
            ⊢ LT.lt (HDiv.hDiv (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) ↑((HSub.h …
          -/
          apply Int.ediv_neg' H
          /-
            case neg.refine_2
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            x✝ : LT.lt 0 nb → LT.lt 0 na
            H : LT.lt (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) 0
            ⊢ LT.lt 0 ↑((HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)).natAbs.gcd (HMul …
          -/
          simp only [Int.natCast_pos, Nat.pos_iff_ne_zero]
          /-
            case neg.refine_2
            a b : Rat
            na : Int
            da : Nat
            ha : Ne da 0
            hared : na.natAbs.Coprime da
            nb : Int
            db : Nat
            hb : Ne db 0
            hbred : nb.natAbs.Coprime db
            h : Not (And (LT.lt nb 0) (LE.le 0 na))
            h' : Not (Eq nb 0)
            x✝ : LT.lt 0 nb → LT.lt 0 na
            H : LT.lt (HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)) 0
            ⊢ Ne ((HSub.hSub (HMul.hMul nb ↑da) (HMul.hMul na ↑db)).natAbs.gcd (HMul.hMul  …
          -/
          exact Nat.gcd_ne_zero_right (Nat.mul_ne_zero hb ha)
          /-
            🎉 no goals
          -/


protected lemma divInt_le_divInt {a b c d : ℤ} (b0 : 0 < b) (d0 : 0 < d) :
    a /. b ≤ c /. d ↔ a * d ≤ c * b := by
  /-
    a b c d : Int
    b0 : LT.lt 0 b
    d0 : LT.lt 0 d
    ⊢ Iff (LE.le (Rat.divInt a b) (Rat.divInt c d)) (LE.le (HMul.hMul a d) (HMul.h …
  -/
  rw [Rat.le_iff_sub_nonneg, ← Int.sub_nonneg]
  /-
    a b c d : Int
    b0 : LT.lt 0 b
    d0 : LT.lt 0 d
    ⊢ Iff (LE.le 0 (HSub.hSub (Rat.divInt c d) (Rat.divInt a b))) (LE.le 0 (HSub.h …
  -/
  simp [sub_eq_add_neg, ne_of_gt b0, ne_of_gt d0, Int.mul_pos d0 b0]
  /-
    🎉 no goals
  -/


protected lemma le_total : a ≤ b ∨ b ≤ a := by
  /-
    a b : Rat
    ⊢ Or (LE.le a b) (LE.le b a)
  -/
  simpa only [← Rat.le_iff_sub_nonneg, neg_sub] using Rat.nonneg_total (b - a)
  /-
    🎉 no goals
  -/


protected theorem not_le {a b : ℚ} : ¬a ≤ b ↔ b < a := (Bool.not_eq_false _).to_iff


instance linearOrder : LinearOrder ℚ where
                  /-
                    a✝ b p q a : Rat
                    ⊢ LE.le a a
                  -/
  le_refl a := by rw [Rat.le_iff_sub_nonneg, ← num_nonneg]; simp
                                                            /-
                                                              🎉 no goals
                                                            -/
  le_trans a b c hab hbc := by
    /-
      a✝ b✝ p q a b c : Rat
      hab : LE.le a b
      hbc : LE.le b c
      ⊢ LE.le a c
    -/
    rw [Rat.le_iff_sub_nonneg] at hab hbc
    /-
      a✝ b✝ p q a b c : Rat
      hab : LE.le 0 (HSub.hSub b a)
      hbc : LE.le 0 (HSub.hSub c b)
      ⊢ LE.le a c
    -/
    have := Rat.add_nonneg hab hbc
    simp_rw [sub_eq_add_neg, add_left_comm (b + -a) c (-b), add_comm (b + -a) (-b),
      add_left_comm (-b) b (-a), add_comm (-b) (-a), add_neg_cancel_comm_assoc,
      ← sub_eq_add_neg] at this
    /-
      a✝ b✝ p q a b c : Rat
      hab : LE.le 0 (HSub.hSub b a)
      hbc : LE.le 0 (HSub.hSub c b)
      this : LE.le 0 (HSub.hSub c a)
      ⊢ LE.le a c
    -/
    rwa [Rat.le_iff_sub_nonneg]
    /-
      🎉 no goals
    -/
  le_antisymm a b hab hba := by
    /-
      a✝ b✝ p q a b : Rat
      hab : LE.le a b
      hba : LE.le b a
      ⊢ Eq a b
    -/
    rw [Rat.le_iff_sub_nonneg] at hab hba
    /-
      a✝ b✝ p q a b : Rat
      hab : LE.le 0 (HSub.hSub b a)
      hba : LE.le 0 (HSub.hSub a b)
      ⊢ Eq a b
    -/
    rw [sub_eq_add_neg] at hba
    /-
      a✝ b✝ p q a b : Rat
      hab : LE.le 0 (HSub.hSub b a)
      hba : LE.le 0 (HAdd.hAdd a (Neg.neg b))
      ⊢ Eq a b
    -/
    rw [← neg_sub, sub_eq_add_neg] at hab
    /-
      a✝ b✝ p q a b : Rat
      hab : LE.le 0 (Neg.neg (HAdd.hAdd a (Neg.neg b)))
      hba : LE.le 0 (HAdd.hAdd a (Neg.neg b))
      ⊢ Eq a b
    -/
    have := eq_neg_of_add_eq_zero_left (Rat.nonneg_antisymm hba hab)
    /-
      a✝ b✝ p q a b : Rat
      hab : LE.le 0 (Neg.neg (HAdd.hAdd a (Neg.neg b)))
      hba : LE.le 0 (HAdd.hAdd a (Neg.neg b))
      this : Eq a (Neg.neg (Neg.neg b))
      ⊢ Eq a b
    -/
                             /-
                               a b p q x✝¹ x✝ : Rat
                               ⊢ Iff (LT.lt x✝¹ x✝) (And (LE.le x✝¹ x✝) (Not (LE.le x✝ x✝¹)))
                             -/
    rwa [neg_neg] at this
                             /-
                               🎉 no goals
                             -/
    /-
      🎉 no goals
    -/
  le_total _ _ := Rat.le_total
  decidableEq := inferInstance
  decidableLE := inferInstance
  decidableLT := inferInstance
  lt_iff_le_not_le _ _ := by rw [← Rat.not_le, and_iff_right_of_imp Rat.le_total.resolve_left]


instance instDistribLattice : DistribLattice ℚ := inferInstance

instance instLattice        : Lattice ℚ        := inferInstance

instance instSemilatticeInf : SemilatticeInf ℚ := inferInstance

instance instSemilatticeSup : SemilatticeSup ℚ := inferInstance

instance instInf            : Min ℚ            := inferInstance

instance instSup            : Max ℚ            := inferInstance

instance instPartialOrder   : PartialOrder ℚ   := inferInstance

instance instPreorder       : Preorder ℚ       := inferInstance


protected lemma le_def : p ≤ q ↔ p.num * q.den ≤ q.num * p.den := by
  /-
    p q : Rat
    ⊢ Iff (LE.le p q) (LE.le (HMul.hMul p.num ↑q.den) (HMul.hMul q.num ↑p.den))
  -/
  rw [← num_divInt_den q, ← num_divInt_den p]
  /-
    p q : Rat
    ⊢ Iff (LE.le (Rat.divInt p.num ↑p.den) (Rat.divInt q.num ↑q.den)) (LE.le (HMul …
  -/
  conv_rhs => simp only [num_divInt_den]
  /-
    p q : Rat
    ⊢ Iff (LE.le (Rat.divInt p.num ↑p.den) (Rat.divInt q.num ↑q.den)) (LE.le (HMul …
  -/
  exact Rat.divInt_le_divInt (mod_cast p.pos) (mod_cast q.pos)
  /-
    🎉 no goals
  -/


protected lemma lt_def : p < q ↔ p.num * q.den < q.num * p.den := by
  /-
    p q : Rat
    ⊢ Iff (LT.lt p q) (LT.lt (HMul.hMul p.num ↑q.den) (HMul.hMul q.num ↑p.den))
  -/
  rw [lt_iff_le_and_ne, Rat.le_def]
  suffices p ≠ q ↔ p.num * q.den ≠ q.num * p.den by
    constructor <;> intro h
    · exact lt_iff_le_and_ne.mpr ⟨h.left, this.mp h.right⟩
    · have tmp := lt_iff_le_and_ne.mp h
      exact ⟨tmp.left, this.mpr tmp.right⟩
  /-
    p q : Rat
    ⊢ Iff (Ne p q) (Ne (HMul.hMul p.num ↑q.den) (HMul.hMul q.num ↑p.den))
  -/
  exact not_iff_not.mpr eq_iff_mul_eq_mul
  /-
    🎉 no goals
  -/


protected theorem add_le_add_left {a b c : ℚ} : c + a ≤ c + b ↔ a ≤ b := by
  /-
    a b c : Rat
    ⊢ Iff (LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)) (LE.le a b)
  -/
  rw [Rat.le_iff_sub_nonneg, add_sub_add_left_eq_sub, ← Rat.le_iff_sub_nonneg]
  /-
    🎉 no goals
  -/


instance : AddLeftMono ℚ where
  elim := fun _ _ _ h => Rat.add_le_add_left.2 h


@[simp] lemma num_nonpos {a : ℚ} : a.num ≤ 0 ↔ a ≤ 0 := by
  /-
    a : Rat
    ⊢ Iff (LE.le a.num 0) (LE.le a 0)
  -/
  simp [Int.le_iff_lt_or_eq, instLE, Rat.blt, Int.not_lt]
  /-
    🎉 no goals
  -/

@[simp] lemma num_pos {a : ℚ} : 0 < a.num ↔ 0 < a := lt_iff_lt_of_le_iff_le num_nonpos

@[simp] lemma num_neg {a : ℚ} : a.num < 0 ↔ a < 0 := lt_iff_lt_of_le_iff_le num_nonneg


@[deprecated (since := "2024-02-16")] alias num_nonneg_iff_zero_le := num_nonneg

@[deprecated (since := "2024-02-16")] alias num_pos_iff_pos := num_pos


theorem div_lt_div_iff_mul_lt_mul {a b c d : ℤ} (b_pos : 0 < b) (d_pos : 0 < d) :
    (a : ℚ) / b < c / d ↔ a * d < c * b := by
  /-
    a b c d : Int
    b_pos : LT.lt 0 b
    d_pos : LT.lt 0 d
    ⊢ Iff (LT.lt (HDiv.hDiv ↑a ↑b) (HDiv.hDiv ↑c ↑d)) (LT.lt (HMul.hMul a d) (HMul …
  -/
  simp only [lt_iff_le_not_le]
  /-
    a b c d : Int
    b_pos : LT.lt 0 b
    d_pos : LT.lt 0 d
    ⊢ Iff (And (LE.le (HDiv.hDiv ↑a ↑b) (HDiv.hDiv ↑c ↑d)) (Not (LE.le (HDiv.hDiv  …
  -/
  apply and_congr
    /-
      case h₁
      a b c d : Int
      b_pos : LT.lt 0 b
      d_pos : LT.lt 0 d
      ⊢ Iff (LE.le (HDiv.hDiv ↑a ↑b) (HDiv.hDiv ↑c ↑d)) (LE.le (HMul.hMul a d) (HMul …
    -/
  · simp [div_def', Rat.divInt_le_divInt b_pos d_pos]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      a b c d : Int
      b_pos : LT.lt 0 b
      d_pos : LT.lt 0 d
      ⊢ Iff (Not (LE.le (HDiv.hDiv ↑c ↑d) (HDiv.hDiv ↑a ↑b))) (Not (LE.le (HMul.hMul …
    -/
  · apply not_congr
    /-
      case h₂.h
      a b c d : Int
      b_pos : LT.lt 0 b
      d_pos : LT.lt 0 d
      ⊢ Iff (LE.le (HDiv.hDiv ↑c ↑d) (HDiv.hDiv ↑a ↑b)) (LE.le (HMul.hMul c b) (HMul …
    -/
    simp [div_def', Rat.divInt_le_divInt d_pos b_pos]
    /-
      🎉 no goals
    -/


                                                                      /-
                                                                        q : Rat
                                                                        ⊢ Iff (LT.lt q 1) (LT.lt q.num ↑q.den)
                                                                      -/
theorem lt_one_iff_num_lt_denom {q : ℚ} : q < 1 ↔ q.num < q.den := by simp [Rat.lt_def]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem abs_def (q : ℚ) : |q| = q.num.natAbs /. q.den := by
  /-
    q : Rat
    ⊢ Eq (abs q) (Rat.divInt ↑q.num.natAbs ↑q.den)
  -/
  rcases le_total q 0 with hq | hq
    /-
      case inl
      q : Rat
      hq : LE.le q 0
      ⊢ Eq (abs q) (Rat.divInt ↑q.num.natAbs ↑q.den)
    -/
  · rw [abs_of_nonpos hq]
    rw [← num_divInt_den q, ← zero_divInt, Rat.divInt_le_divInt (mod_cast q.pos) Int.zero_lt_one,
      mul_one, zero_mul] at hq
    /-
      case inl
      q : Rat
      hq✝ : LE.le (Rat.divInt q.num ↑q.den) 0
      hq : LE.le q.num 0
      ⊢ Eq (Neg.neg q) (Rat.divInt ↑q.num.natAbs ↑q.den)
    -/
    rw [Int.ofNat_natAbs_of_nonpos hq, ← neg_def]
    /-
      🎉 no goals
    -/
    /-
      case inr
      q : Rat
      hq : LE.le 0 q
      ⊢ Eq (abs q) (Rat.divInt ↑q.num.natAbs ↑q.den)
    -/
  · rw [abs_of_nonneg hq]
    rw [← num_divInt_den q, ← zero_divInt, Rat.divInt_le_divInt Int.zero_lt_one (mod_cast q.pos),
      mul_one, zero_mul] at hq
    /-
      case inr
      q : Rat
      hq✝ : LE.le 0 (Rat.divInt q.num ↑q.den)
      hq : LE.le 0 q.num
      ⊢ Eq q (Rat.divInt ↑q.num.natAbs ↑q.den)
    -/
    rw [Int.natAbs_of_nonneg hq, num_divInt_den]
    /-
      🎉 no goals
    -/


