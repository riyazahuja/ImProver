/-- If `f : ι → α` and `g : ι → β` are monovarying, then `MonovaryOrder f g` is a linear order on
`ι` that makes `f` and `g` simultaneously monotone.
We define `i < j` if `f i < f j`, or if `f i = f j` and `g i < g j`, breaking ties arbitrarily. -/
def MonovaryOrder (i j : ι) : Prop :=
  Prod.Lex (· < ·) (Prod.Lex (· < ·) WellOrderingRel) (f i, g i, i) (f j, g j, j)


instance : IsStrictTotalOrder ι (MonovaryOrder f g)
    where
  trichotomous i j := by
    /-
      ι : Type u_1
      ι' : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : ι → α
      g : ι → β
      s : Set ι
      i j : ι
      ⊢ Or (MonovaryOrder f g i j) (Or (Eq i j) (MonovaryOrder f g j i))
    -/
    convert trichotomous_of (Prod.Lex (· < ·) <| Prod.Lex (· < ·) WellOrderingRel) _ _
      /-
        case h.e'_2.h.e'_1.a
        ι : Type u_1
        ι' : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : ι → α
        g : ι → β
        s : Set ι
        i j : ι
        ⊢ Iff (Eq i j) (Eq { fst := f i, snd := { fst := g i, snd := i } } { fst := f  …
      -/
    · simp only [Prod.ext_iff, ← and_assoc, imp_and, eq_iff_iff, iff_and_self]
      /-
        case h.e'_2.h.e'_1.a
        ι : Type u_1
        ι' : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : ι → α
        g : ι → β
        s : Set ι
        i j : ι
        ⊢ And (Eq i j → Eq (f i) (f j)) (Eq i j → Eq (g i) (g j))
      -/
      exact ⟨congr_arg _, congr_arg _⟩
      /-
        🎉 no goals
      -/
      /-
        case convert_6
        ι : Type u_1
        ι' : Type u_2
        α : Type u_3
        β : Type u_4
        γ : Type u_5
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : ι → α
        g : ι → β
        s : Set ι
        i j : ι
        ⊢ IsTrichotomous (Prod α (Prod β ι)) (Prod.Lex (fun x1 x2 => LT.lt x1 x2) (Pro …
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
                 /-
                   ι : Type u_1
                   ι' : Type u_2
                   α : Type u_3
                   β : Type u_4
                   γ : Type u_5
                   inst✝¹ : LinearOrder α
                   inst✝ : LinearOrder β
                   f : ι → α
                   g : ι → β
                   s : Set ι
                   i : ι
                   ⊢ Not (MonovaryOrder f g i i)
                 -/
  irrefl i := by rw [MonovaryOrder]; exact irrefl _
                                     /-
                                       🎉 no goals
                                     -/
                    /-
                      ι : Type u_1
                      ι' : Type u_2
                      α : Type u_3
                      β : Type u_4
                      γ : Type u_5
                      inst✝¹ : LinearOrder α
                      inst✝ : LinearOrder β
                      f : ι → α
                      g : ι → β
                      s : Set ι
                      i j k : ι
                      ⊢ MonovaryOrder f g i j → MonovaryOrder f g j k → MonovaryOrder f g i k
                    -/
  trans i j k := by rw [MonovaryOrder]; exact _root_.trans
                                        /-
                                          🎉 no goals
                                        -/


lemma monovaryOn_iff_exists_monotoneOn :
    MonovaryOn f g s ↔ ∃ (_ : LinearOrder ι), MonotoneOn f s ∧ MonotoneOn g s := by
  classical
  letI := linearOrderOfSTO (MonovaryOrder f g)
  refine ⟨fun hfg => ⟨‹_›, monotoneOn_iff_forall_lt.2 fun i hi j hj hij => ?_,
    monotoneOn_iff_forall_lt.2 fun i hi j hj hij => ?_⟩, ?_⟩
  · obtain h | ⟨h, -⟩ := Prod.lex_iff.1 hij <;> exact h.le
  · obtain h | ⟨-, h⟩ := Prod.lex_iff.1 hij
    · exact hfg.symm hi hj h
    obtain h | ⟨h, -⟩ := Prod.lex_iff.1 h <;> exact h.le
  · rintro ⟨_, hf, hg⟩
    exact hf.monovaryOn hg


lemma antivaryOn_iff_exists_monotoneOn_antitoneOn :
    AntivaryOn f g s ↔ ∃ (_ : LinearOrder ι), MonotoneOn f s ∧ AntitoneOn g s := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : ι → α
    g : ι → β
    s : Set ι
    ⊢ Iff (AntivaryOn f g s) (Exists fun x => And (MonotoneOn f s) (AntitoneOn g s))
  -/
  simp_rw [← monovaryOn_toDual_right, monovaryOn_iff_exists_monotoneOn, monotoneOn_toDual_comp_iff]
  /-
    🎉 no goals
  -/


lemma monovaryOn_iff_exists_antitoneOn :
    MonovaryOn f g s ↔ ∃ (_ : LinearOrder ι), AntitoneOn f s ∧ AntitoneOn g s := by
  simp_rw [← antivaryOn_toDual_left, antivaryOn_iff_exists_monotoneOn_antitoneOn,
    monotoneOn_toDual_comp_iff]


lemma antivaryOn_iff_exists_antitoneOn_monotoneOn :
    AntivaryOn f g s ↔ ∃ (_ : LinearOrder ι), AntitoneOn f s ∧ MonotoneOn g s := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : ι → α
    g : ι → β
    s : Set ι
    ⊢ Iff (AntivaryOn f g s) (Exists fun x => And (AntitoneOn f s) (MonotoneOn g s))
  -/
  simp_rw [← monovaryOn_toDual_left, monovaryOn_iff_exists_monotoneOn, monotoneOn_toDual_comp_iff]
  /-
    🎉 no goals
  -/


lemma monovary_iff_exists_monotone :
    Monovary f g ↔ ∃ (_ : LinearOrder ι), Monotone f ∧ Monotone g := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : ι → α
    g : ι → β
    ⊢ Iff (Monovary f g) (Exists fun x => And (Monotone f) (Monotone g))
  -/
  simp [← monovaryOn_univ, monovaryOn_iff_exists_monotoneOn]
  /-
    🎉 no goals
  -/


lemma monovary_iff_exists_antitone :
    Monovary f g ↔ ∃ (_ : LinearOrder ι), Antitone f ∧ Antitone g := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : ι → α
    g : ι → β
    ⊢ Iff (Monovary f g) (Exists fun x => And (Antitone f) (Antitone g))
  -/
  simp [← monovaryOn_univ, monovaryOn_iff_exists_antitoneOn]
  /-
    🎉 no goals
  -/


lemma antivary_iff_exists_monotone_antitone :
    Antivary f g ↔ ∃ (_ : LinearOrder ι), Monotone f ∧ Antitone g := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : ι → α
    g : ι → β
    ⊢ Iff (Antivary f g) (Exists fun x => And (Monotone f) (Antitone g))
  -/
  simp [← antivaryOn_univ, antivaryOn_iff_exists_monotoneOn_antitoneOn]
  /-
    🎉 no goals
  -/


lemma antivary_iff_exists_antitone_monotone :
    Antivary f g ↔ ∃ (_ : LinearOrder ι), Antitone f ∧ Monotone g := by
  /-
    ι : Type u_1
    α : Type u_3
    β : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : ι → α
    g : ι → β
    ⊢ Iff (Antivary f g) (Exists fun x => And (Antitone f) (Monotone g))
  -/
  simp [← antivaryOn_univ, antivaryOn_iff_exists_antitoneOn_monotoneOn]
  /-
    🎉 no goals
  -/


alias ⟨MonovaryOn.exists_monotoneOn, _⟩ := monovaryOn_iff_exists_monotoneOn

alias ⟨MonovaryOn.exists_antitoneOn, _⟩ := monovaryOn_iff_exists_antitoneOn

alias ⟨AntivaryOn.exists_monotoneOn_antitoneOn, _⟩ := antivaryOn_iff_exists_monotoneOn_antitoneOn

alias ⟨AntivaryOn.exists_antitoneOn_monotoneOn, _⟩ := antivaryOn_iff_exists_antitoneOn_monotoneOn

alias ⟨Monovary.exists_monotone, _⟩ := monovary_iff_exists_monotone

alias ⟨Monovary.exists_antitone, _⟩ := monovary_iff_exists_antitone

alias ⟨Antivary.exists_monotone_antitone, _⟩ := antivary_iff_exists_monotone_antitone

alias ⟨Antivary.exists_antitone_monotone, _⟩ := antivary_iff_exists_antitone_monotone


