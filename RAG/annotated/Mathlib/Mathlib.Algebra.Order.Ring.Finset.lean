set_option linter.docPrime false in
@[simp, norm_cast]
lemma cast_finsetSup' (f : ι → ℕ) (hs) : ((s.sup' hs f : ℕ) : R) = s.sup' hs fun i ↦ (f i : R) :=
  comp_sup'_eq_sup'_comp _ _ cast_max


set_option linter.docPrime false in
@[simp, norm_cast]
lemma cast_finsetInf' (f : ι → ℕ) (hs) : (↑(s.inf' hs f) : R) = s.inf' hs fun i ↦ (f i : R) :=
  comp_inf'_eq_inf'_comp _ _ cast_min


@[simp, norm_cast]
lemma cast_finsetSup [CanonicallyLinearOrderedSemifield R] (s : Finset ι) (f : ι → ℕ) :
    (↑(s.sup f) : R) = s.sup fun i ↦ (f i : R) :=
                                      /-
                                        ι : Type u_1
                                        R : Type u_2
                                        inst✝ : CanonicallyLinearOrderedSemifield R
                                        s : Finset ι
                                        f : ι → Nat
                                        ⊢ Eq (↑Bot.bot) Bot.bot
                                      -/
  comp_sup_eq_sup_comp _ cast_max (by simp)
                                      /-
                                        🎉 no goals
                                      -/


