instance instConditionallyCompleteLinearOrder : ConditionallyCompleteLinearOrder ℤ where
  __ := instLinearOrder
  __ := LinearOrder.toLattice
  sSup s :=
    if h : s.Nonempty ∧ BddAbove s then
      greatestOfBdd (Classical.choose h.2) (Classical.choose_spec h.2) h.1
    else 0
  sInf s :=
    if h : s.Nonempty ∧ BddBelow s then
      leastOfBdd (Classical.choose h.2) (Classical.choose_spec h.2) h.1
    else 0
  le_csSup s n hs hns := by
    /-
      s : Set Int
      n : Int
      hs : BddAbove s
      hns : Membership.mem s n
      ⊢ LE.le n (SupSet.sSup s)
    -/
    have : s.Nonempty ∧ BddAbove s := ⟨⟨n, hns⟩, hs⟩
    -- Porting note: this was `rw [dif_pos this]`
    /-
      s : Set Int
      n : Int
      hs : BddAbove s
      hns : Membership.mem s n
      this : And s.Nonempty (BddAbove s)
      ⊢ LE.le n (SupSet.sSup s)
    -/
    simp only [this, and_self, dite_true]
    /-
      s : Set Int
      n : Int
      hs : BddAbove s
      hns : Membership.mem s n
      this : And s.Nonempty (BddAbove s)
      ⊢ LE.le n ↑((Classical.choose ⋯).greatestOfBdd ⋯ ⋯)
    -/
    exact (greatestOfBdd _ _ _).2.2 n hns
    /-
      🎉 no goals
    -/
  csSup_le s n hs hns := by
    /-
      s : Set Int
      n : Int
      hs : s.Nonempty
      hns : Membership.mem (upperBounds s) n
      ⊢ LE.le (SupSet.sSup s) n
    -/
    have : s.Nonempty ∧ BddAbove s := ⟨hs, ⟨n, hns⟩⟩
    -- Porting note: this was `rw [dif_pos this]`
    /-
      s : Set Int
      n : Int
      hs : s.Nonempty
      hns : Membership.mem (upperBounds s) n
      this : And s.Nonempty (BddAbove s)
      ⊢ LE.le (SupSet.sSup s) n
    -/
    simp only [this, and_self, dite_true]
    /-
      s : Set Int
      n : Int
      hs : s.Nonempty
      hns : Membership.mem (upperBounds s) n
      this : And s.Nonempty (BddAbove s)
      ⊢ LE.le (↑((Classical.choose ⋯).greatestOfBdd ⋯ ⋯)) n
    -/
    exact hns (greatestOfBdd _ (Classical.choose_spec this.2) _).2.1
    /-
      🎉 no goals
    -/
  csInf_le s n hs hns := by
    /-
      s : Set Int
      n : Int
      hs : BddBelow s
      hns : Membership.mem s n
      ⊢ LE.le (InfSet.sInf s) n
    -/
    have : s.Nonempty ∧ BddBelow s := ⟨⟨n, hns⟩, hs⟩
    -- Porting note: this was `rw [dif_pos this]`
    /-
      s : Set Int
      n : Int
      hs : BddBelow s
      hns : Membership.mem s n
      this : And s.Nonempty (BddBelow s)
      ⊢ LE.le (InfSet.sInf s) n
    -/
    simp only [this, and_self, dite_true]
    /-
      s : Set Int
      n : Int
      hs : BddBelow s
      hns : Membership.mem s n
      this : And s.Nonempty (BddBelow s)
      ⊢ LE.le (↑((Classical.choose ⋯).leastOfBdd ⋯ ⋯)) n
    -/
    exact (leastOfBdd _ _ _).2.2 n hns
    /-
      🎉 no goals
    -/
  le_csInf s n hs hns := by
    /-
      s : Set Int
      n : Int
      hs : s.Nonempty
      hns : Membership.mem (lowerBounds s) n
      ⊢ LE.le n (InfSet.sInf s)
    -/
    have : s.Nonempty ∧ BddBelow s := ⟨hs, ⟨n, hns⟩⟩
    -- Porting note: this was `rw [dif_pos this]`
    /-
      s : Set Int
      n : Int
      hs : s.Nonempty
      hns : Membership.mem (lowerBounds s) n
      this : And s.Nonempty (BddBelow s)
      ⊢ LE.le n (InfSet.sInf s)
    -/
    simp only [this, and_self, dite_true]
    /-
      s : Set Int
      n : Int
      hs : s.Nonempty
      hns : Membership.mem (lowerBounds s) n
      this : And s.Nonempty (BddBelow s)
      ⊢ LE.le n ↑((Classical.choose ⋯).leastOfBdd ⋯ ⋯)
    -/
    exact hns (leastOfBdd _ (Classical.choose_spec this.2) _).2.1
    /-
      🎉 no goals
    -/
                                         /-
                                           s : Set Int
                                           hs : Not (BddAbove s)
                                           ⊢ Eq (SupSet.sSup s) (SupSet.sSup EmptyCollection.emptyCollection)
                                         -/
  csSup_of_not_bddAbove := fun s hs ↦ by simp [hs]
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           s : Set Int
                                           hs : Not (BddBelow s)
                                           ⊢ Eq (InfSet.sInf s) (InfSet.sInf EmptyCollection.emptyCollection)
                                         -/
  csInf_of_not_bddBelow := fun s hs ↦ by simp [hs]
                                         /-
                                           🎉 no goals
                                         -/


theorem csSup_eq_greatest_of_bdd {s : Set ℤ} [DecidablePred (· ∈ s)] (b : ℤ) (Hb : ∀ z ∈ s, z ≤ b)
    (Hinh : ∃ z : ℤ, z ∈ s) : sSup s = greatestOfBdd b Hb Hinh := by
  /-
    s : Set Int
    inst✝ : DecidablePred fun x => Membership.mem s x
    b : Int
    Hb : ∀ (z : Int), Membership.mem s z → LE.le z b
    Hinh : Exists fun z => Membership.mem s z
    ⊢ Eq (SupSet.sSup s) ↑(b.greatestOfBdd Hb Hinh)
  -/
  have : s.Nonempty ∧ BddAbove s := ⟨Hinh, b, Hb⟩
  /-
    s : Set Int
    inst✝ : DecidablePred fun x => Membership.mem s x
    b : Int
    Hb : ∀ (z : Int), Membership.mem s z → LE.le z b
    Hinh : Exists fun z => Membership.mem s z
    this : And s.Nonempty (BddAbove s)
    ⊢ Eq (SupSet.sSup s) ↑(b.greatestOfBdd Hb Hinh)
  -/
  simp only [sSup, this, and_self, dite_true]
  /-
    s : Set Int
    inst✝ : DecidablePred fun x => Membership.mem s x
    b : Int
    Hb : ∀ (z : Int), Membership.mem s z → LE.le z b
    Hinh : Exists fun z => Membership.mem s z
    this : And s.Nonempty (BddAbove s)
    ⊢ Eq ↑((Classical.choose ⋯).greatestOfBdd ⋯ ⋯) ↑(b.greatestOfBdd Hb Hinh)
  -/
  convert (coe_greatestOfBdd_eq Hb (Classical.choose_spec (⟨b, Hb⟩ : BddAbove s)) Hinh).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem csSup_empty : sSup (∅ : Set ℤ) = 0 :=
              /-
                ⊢ Not (And EmptyCollection.emptyCollection.Nonempty (BddAbove EmptyCollection. …
              -/
  dif_neg (by simp)
              /-
                🎉 no goals
              -/


theorem csSup_of_not_bdd_above {s : Set ℤ} (h : ¬BddAbove s) : sSup s = 0 :=
              /-
                s : Set Int
                h : Not (BddAbove s)
                ⊢ Not (And s.Nonempty (BddAbove s))
              -/
  dif_neg (by simp [h])
              /-
                🎉 no goals
              -/

-- Porting note: mathlib3 proof uses `convert dif_pos _ using 1`

theorem csInf_eq_least_of_bdd {s : Set ℤ} [DecidablePred (· ∈ s)] (b : ℤ) (Hb : ∀ z ∈ s, b ≤ z)
    (Hinh : ∃ z : ℤ, z ∈ s) : sInf s = leastOfBdd b Hb Hinh := by
  /-
    s : Set Int
    inst✝ : DecidablePred fun x => Membership.mem s x
    b : Int
    Hb : ∀ (z : Int), Membership.mem s z → LE.le b z
    Hinh : Exists fun z => Membership.mem s z
    ⊢ Eq (InfSet.sInf s) ↑(b.leastOfBdd Hb Hinh)
  -/
  have : s.Nonempty ∧ BddBelow s := ⟨Hinh, b, Hb⟩
  /-
    s : Set Int
    inst✝ : DecidablePred fun x => Membership.mem s x
    b : Int
    Hb : ∀ (z : Int), Membership.mem s z → LE.le b z
    Hinh : Exists fun z => Membership.mem s z
    this : And s.Nonempty (BddBelow s)
    ⊢ Eq (InfSet.sInf s) ↑(b.leastOfBdd Hb Hinh)
  -/
  simp only [sInf, this, and_self, dite_true]
  /-
    s : Set Int
    inst✝ : DecidablePred fun x => Membership.mem s x
    b : Int
    Hb : ∀ (z : Int), Membership.mem s z → LE.le b z
    Hinh : Exists fun z => Membership.mem s z
    this : And s.Nonempty (BddBelow s)
    ⊢ Eq ↑((Classical.choose ⋯).leastOfBdd ⋯ ⋯) ↑(b.leastOfBdd Hb Hinh)
  -/
  convert (coe_leastOfBdd_eq Hb (Classical.choose_spec (⟨b, Hb⟩ : BddBelow s)) Hinh).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem csInf_empty : sInf (∅ : Set ℤ) = 0 :=
              /-
                ⊢ Not (And EmptyCollection.emptyCollection.Nonempty (BddBelow EmptyCollection. …
              -/
  dif_neg (by simp)
              /-
                🎉 no goals
              -/


theorem csInf_of_not_bdd_below {s : Set ℤ} (h : ¬BddBelow s) : sInf s = 0 :=
              /-
                s : Set Int
                h : Not (BddBelow s)
                ⊢ Not (And s.Nonempty (BddBelow s))
              -/
  dif_neg (by simp [h])
              /-
                🎉 no goals
              -/


theorem csSup_mem {s : Set ℤ} (h1 : s.Nonempty) (h2 : BddAbove s) : sSup s ∈ s := by
  /-
    s : Set Int
    h1 : s.Nonempty
    h2 : BddAbove s
    ⊢ Membership.mem s (SupSet.sSup s)
  -/
  convert (greatestOfBdd _ (Classical.choose_spec h2) h1).2.1
  /-
    case h.e'_5
    s : Set Int
    h1 : s.Nonempty
    h2 : BddAbove s
    ⊢ Eq (SupSet.sSup s) ↑((Classical.choose h2).greatestOfBdd ⋯ h1)
  -/
  exact dif_pos ⟨h1, h2⟩
  /-
    🎉 no goals
  -/


theorem csInf_mem {s : Set ℤ} (h1 : s.Nonempty) (h2 : BddBelow s) : sInf s ∈ s := by
  /-
    s : Set Int
    h1 : s.Nonempty
    h2 : BddBelow s
    ⊢ Membership.mem s (InfSet.sInf s)
  -/
  convert (leastOfBdd _ (Classical.choose_spec h2) h1).2.1
  /-
    case h.e'_5
    s : Set Int
    h1 : s.Nonempty
    h2 : BddBelow s
    ⊢ Eq (InfSet.sInf s) ↑((Classical.choose h2).leastOfBdd ⋯ h1)
  -/
  exact dif_pos ⟨h1, h2⟩
  /-
    🎉 no goals
  -/


