/-- `cardPowDegree` is the absolute value on `𝔽_q[t]` sending `f` to `q ^ degree f`.

`cardPowDegree 0` is defined to be `0`. -/
noncomputable def cardPowDegree : AbsoluteValue Fq[X] ℤ :=
  have card_pos : 0 < Fintype.card Fq := Fintype.card_pos_iff.mpr inferInstance
  have pow_pos : ∀ n, 0 < (Fintype.card Fq : ℤ) ^ n := fun n =>
    pow_pos (Int.natCast_pos.mpr card_pos) n
  letI := Classical.decEq Fq
  { toFun := fun p => if p = 0 then 0 else (Fintype.card Fq : ℤ) ^ p.natDegree
    nonneg' := fun p => by
      /-
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p : Polynomial Fq
        ⊢ LE.le 0 ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq))  …
      -/
      dsimp
      /-
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p : Polynomial Fq
        ⊢ LE.le 0 (ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p.natDegree))
      -/
      split_ifs
        /-
          case pos
          Fq : Type u_1
          inst✝¹ : Field Fq
          inst✝ : Fintype Fq
          card_pos : LT.lt 0 (Fintype.card Fq)
          pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
          this : DecidableEq Fq := Classical.decEq Fq
          p : Polynomial Fq
          h✝ : Eq p 0
          ⊢ LE.le 0 0
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p : Polynomial Fq
        h✝ : Not (Eq p 0)
        ⊢ LE.le 0 (HPow.hPow (↑(Fintype.card Fq)) p.natDegree)
      -/
      exact pow_nonneg (Int.ofNat_zero_le _) _
      /-
        🎉 no goals
      -/
    eq_zero' := fun p =>
      ite_eq_left_iff.trans
        ⟨fun h => by
          /-
            Fq : Type u_1
            inst✝¹ : Field Fq
            inst✝ : Fintype Fq
            card_pos : LT.lt 0 (Fintype.card Fq)
            pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
            this : DecidableEq Fq := Classical.decEq Fq
            p : Polynomial Fq
            h : Not (Eq p 0) → Eq (HPow.hPow (↑(Fintype.card Fq)) p.natDegree) 0
            ⊢ Eq p 0
          -/
          contrapose! h
          /-
            Fq : Type u_1
            inst✝¹ : Field Fq
            inst✝ : Fintype Fq
            card_pos : LT.lt 0 (Fintype.card Fq)
            pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
            this : DecidableEq Fq := Classical.decEq Fq
            p : Polynomial Fq
            h : Ne p 0
            ⊢ And (Ne p 0) (Ne (HPow.hPow (↑(Fintype.card Fq)) p.natDegree) 0)
          -/
          exact ⟨h, (pow_pos _).ne'⟩, absurd⟩
          /-
            🎉 no goals
          -/
    add_le' := fun p q => by
      /-
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        ⊢ LE.le ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p. …
      -/
      by_cases hp : p = 0; · simp [hp]
                             /-
                               🎉 no goals
                             -/
      /-
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        ⊢ Eq ((fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p.natDegree)) ( …
      -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        ⊢ LE.le ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p. …
      -/
                             /-
                               🎉 no goals
                             -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        ⊢ Eq ((fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p.natDegree)) ( …
      -/
      by_cases hq : q = 0; · simp [hq]
                             /-
                               🎉 no goals
                             -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        ⊢ Eq ((fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p.natDegree)) ( …
      -/
                             /-
                               🎉 no goals
                             -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        ⊢ LE.le ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p. …
      -/
      by_cases hpq : p + q = 0
        /-
          case pos
          Fq : Type u_1
          inst✝¹ : Field Fq
          inst✝ : Fintype Fq
          card_pos : LT.lt 0 (Fintype.card Fq)
          pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
          this : DecidableEq Fq := Classical.decEq Fq
          p q : Polynomial Fq
          hp : Not (Eq p 0)
          hq : Not (Eq q 0)
          hpq : Eq (HAdd.hAdd p q) 0
          ⊢ LE.le ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p. …
        -/
      · simp only [hpq, hp, hq, eq_self_iff_true, if_true, if_false]
        /-
          case pos
          Fq : Type u_1
          inst✝¹ : Field Fq
          inst✝ : Fintype Fq
          card_pos : LT.lt 0 (Fintype.card Fq)
          pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
          this : DecidableEq Fq := Classical.decEq Fq
          p q : Polynomial Fq
          hp : Not (Eq p 0)
          hq : Not (Eq q 0)
          hpq : Eq (HAdd.hAdd p q) 0
          ⊢ LE.le 0 (HAdd.hAdd (HPow.hPow (↑(Fintype.card Fq)) p.natDegree) (HPow.hPow ( …
        -/
        exact add_nonneg (pow_pos _).le (pow_pos _).le
        /-
          🎉 no goals
        -/
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        hpq : Not (Eq (HAdd.hAdd p q) 0)
        ⊢ LE.le ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p. …
      -/
      simp only [hpq, hp, hq, if_false]
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        hpq : Not (Eq (HAdd.hAdd p q) 0)
        ⊢ LE.le (HPow.hPow (↑(Fintype.card Fq)) (HAdd.hAdd p q).natDegree) (HAdd.hAdd  …
      -/
      refine le_trans (pow_right_mono₀ (by omega) (Polynomial.natDegree_add_le _ _)) ?_
      refine
        le_trans (le_max_iff.mpr ?_)
          (max_le_add_of_nonneg (pow_nonneg (by omega) _) (pow_nonneg (by omega) _))
      /-
        case neg
        Fq : Type u_1
        inst✝¹ : Field Fq
        inst✝ : Fintype Fq
        card_pos : LT.lt 0 (Fintype.card Fq)
        pow_pos : ∀ (n : Nat), LT.lt 0 (HPow.hPow (↑(Fintype.card Fq)) n)
        this : DecidableEq Fq := Classical.decEq Fq
        p q : Polynomial Fq
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        hpq : Not (Eq (HAdd.hAdd p q) 0)
        ⊢ Or (LE.le ((fun x => HPow.hPow (↑(Fintype.card Fq)) x) (Max.max p.natDegree  …
      -/
      exact (max_choice p.natDegree q.natDegree).imp (fun h => by rw [h]) fun h => by rw [h]
      /-
        🎉 no goals
      -/
    map_mul' := fun p q => by
      by_cases hp : p = 0; · simp [hp]
      by_cases hq : q = 0; · simp [hq]
      have hpq : p * q ≠ 0 := mul_ne_zero hp hq
      simp only [hpq, hp, hq, eq_self_iff_true, if_true, if_false, Polynomial.natDegree_mul hp hq,
        pow_add] }


theorem cardPowDegree_apply [DecidableEq Fq] (p : Fq[X]) :
    cardPowDegree p = if p = 0 then 0 else (Fintype.card Fq : ℤ) ^ natDegree p := by
  /-
    Fq : Type u_1
    inst✝² : Field Fq
    inst✝¹ : Fintype Fq
    inst✝ : DecidableEq Fq
    p : Polynomial Fq
    ⊢ Eq (Polynomial.cardPowDegree p) (ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card F …
  -/
  rw [cardPowDegree]
  /-
    Fq : Type u_1
    inst✝² : Field Fq
    inst✝¹ : Fintype Fq
    inst✝ : DecidableEq Fq
    p : Polynomial Fq
    ⊢ Eq ({ toFun := fun p => ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p.nat …
  -/
  dsimp
  /-
    Fq : Type u_1
    inst✝² : Field Fq
    inst✝¹ : Fintype Fq
    inst✝ : DecidableEq Fq
    p : Polynomial Fq
    ⊢ Eq (ite (Eq p 0) 0 (HPow.hPow (↑(Fintype.card Fq)) p.natDegree)) (ite (Eq p  …
  -/
  convert rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem cardPowDegree_zero : cardPowDegree (0 : Fq[X]) = 0 := rfl


@[simp]
theorem cardPowDegree_nonzero (p : Fq[X]) (hp : p ≠ 0) :
    cardPowDegree p = (Fintype.card Fq : ℤ) ^ p.natDegree :=
  if_neg hp


theorem cardPowDegree_isEuclidean : IsEuclidean (cardPowDegree : AbsoluteValue Fq[X] ℤ) :=
  have card_pos : 0 < Fintype.card Fq := Fintype.card_pos_iff.mpr inferInstance
  have pow_pos : ∀ n, 0 < (Fintype.card Fq : ℤ) ^ n := fun n =>
    pow_pos (Int.natCast_pos.mpr card_pos) n
  { map_lt_map_iff' := fun {p q} => by
      classical
      show cardPowDegree p < cardPowDegree q ↔ degree p < degree q
      simp only [cardPowDegree_apply]
      split_ifs with hp hq hq
      · simp only [hp, hq, lt_self_iff_false]
      · simp only [hp, hq, degree_zero, Ne, bot_lt_iff_ne_bot, degree_eq_bot, pow_pos,
          not_false_iff]
      · simp only [hp, hq, degree_zero, not_lt_bot, (pow_pos _).not_lt]
      · rw [degree_eq_natDegree hp, degree_eq_natDegree hq, Nat.cast_lt, pow_lt_pow_iff_right₀]
        exact mod_cast @Fintype.one_lt_card Fq _ _ }


