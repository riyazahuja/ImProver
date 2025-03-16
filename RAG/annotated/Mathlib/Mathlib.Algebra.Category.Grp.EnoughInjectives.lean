instance enoughInjectives : EnoughInjectives AddCommGrp.{u} where
  presentation A_ := Nonempty.intro
    { J := of <| (CharacterModule A_) → ULift.{u} (AddCircle (1 : ℚ))
      injective := injective_of_divisible _
                                          /-
                                            A_ : AddCommGrp
                                            ⊢ Eq ((fun a i => { down := i a }) 0) 0
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
      f := ⟨⟨fun a i ↦ ULift.up (i a), by aesop⟩, by aesop⟩
                                                     /-
                                                       🎉 no goals
                                                     -/
      mono := (AddCommGrp.mono_iff_injective _).mpr <| (injective_iff_map_eq_zero _).mpr
        fun _ h0 ↦ eq_zero_of_character_apply (congr_arg ULift.down <| congr_fun h0 ·) }


instance enoughInjectives : EnoughInjectives CommGrp.{u} :=
  EnoughInjectives.of_equivalence commGroupAddCommGroupEquivalence.functor


