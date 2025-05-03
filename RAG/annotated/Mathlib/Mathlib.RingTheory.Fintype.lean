lemma Finset.univ_of_card_le_two (h : Fintype.card R ≤ 2) :
    (univ : Finset R) = {0, 1} := by
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : Fintype R
    inst✝ : DecidableEq R
    h : LE.le (Fintype.card R) 2
    ⊢ Eq Finset.univ (Insert.insert 0 (Singleton.singleton 1))
  -/
  rcases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h : LE.le (Fintype.card R) 2
      h✝ : Subsingleton R
      ⊢ Eq Finset.univ (Insert.insert 0 (Singleton.singleton 1))
    -/
  · exact le_antisymm (fun a _ ↦ by simp [Subsingleton.elim a 0]) (Finset.subset_univ _)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h : LE.le (Fintype.card R) 2
      h✝ : Nontrivial R
      ⊢ Eq Finset.univ (Insert.insert 0 (Singleton.singleton 1))
    -/
  · refine (eq_of_subset_of_card_le (subset_univ _) ?_).symm
    /-
      case inr
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h : LE.le (Fintype.card R) 2
      h✝ : Nontrivial R
      ⊢ LE.le Finset.univ.card (Insert.insert 0 (Singleton.singleton 1)).card
    -/
    convert h
    /-
      case h.e'_4
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h : LE.le (Fintype.card R) 2
      h✝ : Nontrivial R
      ⊢ Eq (Insert.insert 0 (Singleton.singleton 1)).card 2
    -/
    simp
    /-
      🎉 no goals
    -/


lemma Finset.univ_of_card_le_three (h : Fintype.card R ≤ 3) :
    (univ : Finset R) = {0, 1, -1} := by
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : Fintype R
    inst✝ : DecidableEq R
    h : LE.le (Fintype.card R) 3
    ⊢ Eq Finset.univ (Insert.insert 0 (Insert.insert 1 (Singleton.singleton (-1))))
  -/
  refine (eq_of_subset_of_card_le (subset_univ _) ?_).symm
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : Fintype R
    inst✝ : DecidableEq R
    h : LE.le (Fintype.card R) 3
    ⊢ LE.le Finset.univ.card (Insert.insert 0 (Insert.insert 1 (Singleton.singleto …
  -/
  rcases lt_or_eq_of_le h with h | h
    /-
      case inl
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h✝ : LE.le (Fintype.card R) 3
      h : LT.lt (Fintype.card R) 3
      ⊢ LE.le Finset.univ.card (Insert.insert 0 (Insert.insert 1 (Singleton.singleto …
    -/
  · apply card_le_card
    /-
      case inl.a
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h✝ : LE.le (Fintype.card R) 3
      h : LT.lt (Fintype.card R) 3
      ⊢ HasSubset.Subset Finset.univ (Insert.insert 0 (Insert.insert 1 (Singleton.si …
    -/
    rw [Finset.univ_of_card_le_two (Nat.lt_succ_iff.1 h)]
    /-
      case inl.a
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h✝ : LE.le (Fintype.card R) 3
      h : LT.lt (Fintype.card R) 3
      ⊢ HasSubset.Subset (Insert.insert 0 (Singleton.singleton 1)) (Insert.insert 0  …
    -/
    intro a ha
    /-
      case inl.a
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h✝ : LE.le (Fintype.card R) 3
      h : LT.lt (Fintype.card R) 3
      a : R
      ha : Membership.mem (Insert.insert 0 (Singleton.singleton 1)) a
      ⊢ Membership.mem (Insert.insert 0 (Insert.insert 1 (Singleton.singleton (-1))) …
    -/
    simp only [mem_insert, mem_singleton] at ha
    /-
      case inl.a
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h✝ : LE.le (Fintype.card R) 3
      h : LT.lt (Fintype.card R) 3
      a : R
      ha : Or (Eq a 0) (Eq a 1)
      ⊢ Membership.mem (Insert.insert 0 (Insert.insert 1 (Singleton.singleton (-1))) …
    -/
                                 /-
                                   🎉 no goals
                                 -/
    rcases ha with rfl | rfl <;> simp
                                 /-
                                   🎉 no goals
                                 -/
  · have : Nontrivial R := by
      refine Fintype.one_lt_card_iff_nontrivial.1 ?_
      rw [h]
      norm_num
    /-
      case inr
      R : Type u_1
      inst✝² : Ring R
      inst✝¹ : Fintype R
      inst✝ : DecidableEq R
      h✝ : LE.le (Fintype.card R) 3
      h : Eq (Fintype.card R) 3
      this : Nontrivial R
      ⊢ LE.le Finset.univ.card (Insert.insert 0 (Insert.insert 1 (Singleton.singleto …
    -/
    rw [card_univ, h, card_insert_of_not_mem, card_insert_of_not_mem, card_singleton]
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        ⊢ Not (Membership.mem (Singleton.singleton (-1)) 1)
      -/
    · rw [mem_singleton]
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        ⊢ Not (Eq 1 (-1))
      -/
      intro H
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        H : Eq 1 (-1)
        ⊢ False
      -/
      rw [← add_eq_zero_iff_eq_neg, one_add_one_eq_two] at H
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        H : Eq 2 0
        ⊢ False
      -/
      apply_fun (ringEquivOfPrime R Nat.prime_three h).symm at H
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        H : Eq ((ZMod.ringEquivOfPrime R Nat.prime_three h).symm 2) ((ZMod.ringEquivOf …
        ⊢ False
      -/
      simp only [map_ofNat, map_zero] at H
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        H : Eq 2 0
        ⊢ False
      -/
      replace H : ((2 : ℕ) : ZMod 3) = 0 := H
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        H : Eq (↑2) 0
        ⊢ False
      -/
      rw [natCast_zmod_eq_zero_iff_dvd] at H
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        H : Dvd.dvd 3 2
        ⊢ False
      -/
      norm_num at H
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝ : LE.le (Fintype.card R) 3
        h : Eq (Fintype.card R) 3
        this : Nontrivial R
        ⊢ Not (Membership.mem (Insert.insert 1 (Singleton.singleton (-1))) 0)
      -/
    · intro h
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝¹ : LE.le (Fintype.card R) 3
        h✝ : Eq (Fintype.card R) 3
        this : Nontrivial R
        h : Membership.mem (Insert.insert 1 (Singleton.singleton (-1))) 0
        ⊢ False
      -/
      simp only [mem_insert, mem_singleton, zero_eq_neg] at h
      /-
        case inr
        R : Type u_1
        inst✝² : Ring R
        inst✝¹ : Fintype R
        inst✝ : DecidableEq R
        h✝¹ : LE.le (Fintype.card R) 3
        h✝ : Eq (Fintype.card R) 3
        this : Nontrivial R
        h : Or (Eq 0 1) (Eq 1 0)
        ⊢ False
      -/
      rcases h with (h | h)
        /-
          case inr.inl
          R : Type u_1
          inst✝² : Ring R
          inst✝¹ : Fintype R
          inst✝ : DecidableEq R
          h✝¹ : LE.le (Fintype.card R) 3
          h✝ : Eq (Fintype.card R) 3
          this : Nontrivial R
          h : Eq 0 1
          ⊢ False
        -/
      · exact zero_ne_one h
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          R : Type u_1
          inst✝² : Ring R
          inst✝¹ : Fintype R
          inst✝ : DecidableEq R
          h✝¹ : LE.le (Fintype.card R) 3
          h✝ : Eq (Fintype.card R) 3
          this : Nontrivial R
          h : Eq 1 0
          ⊢ False
        -/
      · exact zero_ne_one h.symm
        /-
          🎉 no goals
        -/


theorem card_units_lt (M₀ : Type*) [MonoidWithZero M₀] [Nontrivial M₀] [Fintype M₀] :
    Fintype.card M₀ˣ < Fintype.card M₀ :=
  Fintype.card_lt_of_injective_of_not_mem Units.val Units.ext not_isUnit_zero

