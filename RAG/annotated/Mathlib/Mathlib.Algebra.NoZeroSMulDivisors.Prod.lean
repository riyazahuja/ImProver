instance noZeroSMulDivisors [Zero R] [Zero M] [Zero N]
    [SMulWithZero R M] [SMulWithZero R N] [NoZeroSMulDivisors R M] [NoZeroSMulDivisors R N] :
    NoZeroSMulDivisors R (M × N) :=
  { eq_zero_or_eq_zero_of_smul_eq_zero := by -- Porting note: in mathlib3 there is no need for `by`/
      -- `intro`/`exact`, i.e. the following works:
      -- ⟨fun c ⟨x, y⟩ h =>
      --   or_iff_not_imp_left.mpr fun hc =>
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        inst✝⁶ : Zero R
        inst✝⁵ : Zero M
        inst✝⁴ : Zero N
        inst✝³ : SMulWithZero R M
        inst✝² : SMulWithZero R N
        inst✝¹ : NoZeroSMulDivisors R M
        inst✝ : NoZeroSMulDivisors R N
        ⊢ ∀ {c : R} {x : Prod M N}, Eq (HSMul.hSMul c x) 0 → Or (Eq c 0) (Eq x 0)
      -/
      intro c ⟨x, y⟩ h
      exact or_iff_not_imp_left.mpr fun hc =>
        mk.inj_iff.mpr
          ⟨(smul_eq_zero.mp (congr_arg fst h)).resolve_left hc,
            (smul_eq_zero.mp (congr_arg snd h)).resolve_left hc⟩ }


