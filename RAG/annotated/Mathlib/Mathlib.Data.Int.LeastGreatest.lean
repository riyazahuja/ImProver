/-- A computable version of `exists_least_of_bdd`: given a decidable predicate on the
integers, with an explicit lower bound and a proof that it is somewhere true, return
the least value for which the predicate is true. -/
def leastOfBdd {P : ℤ → Prop} [DecidablePred P] (b : ℤ) (Hb : ∀ z : ℤ, P z → b ≤ z)
    (Hinh : ∃ z : ℤ, P z) : { lb : ℤ // P lb ∧ ∀ z : ℤ, P z → lb ≤ z } :=
  have EX : ∃ n : ℕ, P (b + n) :=
    let ⟨elt, Helt⟩ := Hinh
    match elt, le.dest (Hb _ Helt), Helt with
    | _, ⟨n, rfl⟩, Hn => ⟨n, Hn⟩
  ⟨b + (Nat.find EX : ℤ), Nat.find_spec EX, fun z h =>
    match z, le.dest (Hb _ h), h with
    | _, ⟨_, rfl⟩, h => add_le_add_left (Int.ofNat_le.2 <| Nat.find_min' _ h) _⟩


/-- `Int.leastOfBdd` is the least integer satisfying a predicate which is false for all `z : ℤ` with
`z < b` for some fixed `b : ℤ`. -/
lemma isLeast_coe_leastOfBdd {P : ℤ → Prop} [DecidablePred P] (b : ℤ) (Hb : ∀ z : ℤ, P z → b ≤ z)
    (Hinh : ∃ z : ℤ, P z) : IsLeast {z | P z} (leastOfBdd b Hb Hinh : ℤ) :=
  (leastOfBdd b Hb Hinh).2


/--
    If `P : ℤ → Prop` is a predicate such that the set `{m : P m}` is bounded below and nonempty,
    then this set has the least element. This lemma uses classical logic to avoid assumption
    `[DecidablePred P]`. See `Int.leastOfBdd` for a constructive counterpart. -/
theorem exists_least_of_bdd
    {P : ℤ → Prop}
    (Hbdd : ∃ b : ℤ , ∀ z : ℤ , P z → b ≤ z)
    (Hinh : ∃ z : ℤ , P z) : ∃ lb : ℤ , P lb ∧ ∀ z : ℤ , P z → lb ≤ z := by
  classical
  let ⟨b , Hb⟩ := Hbdd
  let ⟨lb , H⟩ := leastOfBdd b Hb Hinh
  exact ⟨lb , H⟩


theorem coe_leastOfBdd_eq {P : ℤ → Prop} [DecidablePred P] {b b' : ℤ} (Hb : ∀ z : ℤ, P z → b ≤ z)
    (Hb' : ∀ z : ℤ, P z → b' ≤ z) (Hinh : ∃ z : ℤ, P z) :
    (leastOfBdd b Hb Hinh : ℤ) = leastOfBdd b' Hb' Hinh := by
  /-
    P : Int → Prop
    inst✝ : DecidablePred P
    b b' : Int
    Hb : ∀ (z : Int), P z → LE.le b z
    Hb' : ∀ (z : Int), P z → LE.le b' z
    Hinh : Exists fun z => P z
    ⊢ Eq ↑(b.leastOfBdd Hb Hinh) ↑(b'.leastOfBdd Hb' Hinh)
  -/
  rcases leastOfBdd b Hb Hinh with ⟨n, hn, h2n⟩
  /-
    case mk.intro
    P : Int → Prop
    inst✝ : DecidablePred P
    b b' : Int
    Hb : ∀ (z : Int), P z → LE.le b z
    Hb' : ∀ (z : Int), P z → LE.le b' z
    Hinh : Exists fun z => P z
    n : Int
    hn : P n
    h2n : ∀ (z : Int), P z → LE.le n z
    ⊢ Eq ↑⟨n, ⋯⟩ ↑(b'.leastOfBdd Hb' Hinh)
  -/
  rcases leastOfBdd b' Hb' Hinh with ⟨n', hn', h2n'⟩
  /-
    case mk.intro.mk.intro
    P : Int → Prop
    inst✝ : DecidablePred P
    b b' : Int
    Hb : ∀ (z : Int), P z → LE.le b z
    Hb' : ∀ (z : Int), P z → LE.le b' z
    Hinh : Exists fun z => P z
    n : Int
    hn : P n
    h2n : ∀ (z : Int), P z → LE.le n z
    n' : Int
    hn' : P n'
    h2n' : ∀ (z : Int), P z → LE.le n' z
    ⊢ Eq ↑⟨n, ⋯⟩ ↑⟨n', ⋯⟩
  -/
  exact le_antisymm (h2n _ hn') (h2n' _ hn)
  /-
    🎉 no goals
  -/


/-- A computable version of `exists_greatest_of_bdd`: given a decidable predicate on the
integers, with an explicit upper bound and a proof that it is somewhere true, return
the greatest value for which the predicate is true. -/
def greatestOfBdd {P : ℤ → Prop} [DecidablePred P] (b : ℤ) (Hb : ∀ z : ℤ, P z → z ≤ b)
    (Hinh : ∃ z : ℤ, P z) : { ub : ℤ // P ub ∧ ∀ z : ℤ, P z → z ≤ ub } :=
  have Hbdd' : ∀ z : ℤ, P (-z) → -b ≤ z := fun _ h => neg_le.1 (Hb _ h)
  have Hinh' : ∃ z : ℤ, P (-z) :=
    let ⟨elt, Helt⟩ := Hinh
              /-
                P : Int → Prop
                inst✝ : DecidablePred P
                b : Int
                Hb : ∀ (z : Int), P z → LE.le z b
                Hinh : Exists fun z => P z
                Hbdd' : ∀ (z : Int), P (Neg.neg z) → LE.le (Neg.neg b) z
                elt : Int
                Helt : P elt
                ⊢ P (Neg.neg (Neg.neg elt))
              -/
    ⟨-elt, by rw [neg_neg]; exact Helt⟩
                            /-
                              🎉 no goals
                            -/
  let ⟨lb, Plb, al⟩ := leastOfBdd (-b) Hbdd' Hinh'
                                               /-
                                                 P : Int → Prop
                                                 inst✝ : DecidablePred P
                                                 b : Int
                                                 Hb : ∀ (z : Int), P z → LE.le z b
                                                 Hinh : Exists fun z => P z
                                                 Hbdd' : ∀ (z : Int), P (Neg.neg z) → LE.le (Neg.neg b) z
                                                 Hinh' : Exists fun z => P (Neg.neg z)
                                                 lb : Int
                                                 Plb : P (Neg.neg lb)
                                                 al : ∀ (z : Int), P (Neg.neg z) → LE.le lb z
                                                 z : Int
                                                 h : P z
                                                 ⊢ P (Neg.neg (Neg.neg z))
                                               -/
  ⟨-lb, Plb, fun z h => le_neg.1 <| al _ <| by rwa [neg_neg]⟩
                                               /-
                                                 🎉 no goals
                                               -/


/-- `Int.greatestOfBdd` is the greatest integer satisfying a predicate which is false for all
`z : ℤ` with `b < z` for some fixed `b : ℤ`. -/
lemma isGreatest_coe_greatestOfBdd {P : ℤ → Prop} [DecidablePred P] (b : ℤ)
    (Hb : ∀ z : ℤ, P z → z ≤ b) (Hinh : ∃ z : ℤ, P z) :
    IsGreatest {z | P z} (greatestOfBdd b Hb Hinh : ℤ) :=
  (greatestOfBdd b Hb Hinh).2


/--
    If `P : ℤ → Prop` is a predicate such that the set `{m : P m}` is bounded above and nonempty,
    then this set has the greatest element. This lemma uses classical logic to avoid assumption
    `[DecidablePred P]`. See `Int.greatestOfBdd` for a constructive counterpart. -/
theorem exists_greatest_of_bdd
    {P : ℤ → Prop}
    (Hbdd : ∃ b : ℤ , ∀ z : ℤ , P z → z ≤ b)
    (Hinh : ∃ z : ℤ , P z) : ∃ ub : ℤ , P ub ∧ ∀ z : ℤ , P z → z ≤ ub := by
  classical
  let ⟨b, Hb⟩ := Hbdd
  let ⟨lb, H⟩ := greatestOfBdd b Hb Hinh
  exact ⟨lb, H⟩


theorem coe_greatestOfBdd_eq {P : ℤ → Prop} [DecidablePred P] {b b' : ℤ}
    (Hb : ∀ z : ℤ, P z → z ≤ b) (Hb' : ∀ z : ℤ, P z → z ≤ b') (Hinh : ∃ z : ℤ, P z) :
    (greatestOfBdd b Hb Hinh : ℤ) = greatestOfBdd b' Hb' Hinh := by
  /-
    P : Int → Prop
    inst✝ : DecidablePred P
    b b' : Int
    Hb : ∀ (z : Int), P z → LE.le z b
    Hb' : ∀ (z : Int), P z → LE.le z b'
    Hinh : Exists fun z => P z
    ⊢ Eq ↑(b.greatestOfBdd Hb Hinh) ↑(b'.greatestOfBdd Hb' Hinh)
  -/
  rcases greatestOfBdd b Hb Hinh with ⟨n, hn, h2n⟩
  /-
    case mk.intro
    P : Int → Prop
    inst✝ : DecidablePred P
    b b' : Int
    Hb : ∀ (z : Int), P z → LE.le z b
    Hb' : ∀ (z : Int), P z → LE.le z b'
    Hinh : Exists fun z => P z
    n : Int
    hn : P n
    h2n : ∀ (z : Int), P z → LE.le z n
    ⊢ Eq ↑⟨n, ⋯⟩ ↑(b'.greatestOfBdd Hb' Hinh)
  -/
  rcases greatestOfBdd b' Hb' Hinh with ⟨n', hn', h2n'⟩
  /-
    case mk.intro.mk.intro
    P : Int → Prop
    inst✝ : DecidablePred P
    b b' : Int
    Hb : ∀ (z : Int), P z → LE.le z b
    Hb' : ∀ (z : Int), P z → LE.le z b'
    Hinh : Exists fun z => P z
    n : Int
    hn : P n
    h2n : ∀ (z : Int), P z → LE.le z n
    n' : Int
    hn' : P n'
    h2n' : ∀ (z : Int), P z → LE.le z n'
    ⊢ Eq ↑⟨n, ⋯⟩ ↑⟨n', ⋯⟩
  -/
  exact le_antisymm (h2n' _ hn) (h2n _ hn')
  /-
    🎉 no goals
  -/


