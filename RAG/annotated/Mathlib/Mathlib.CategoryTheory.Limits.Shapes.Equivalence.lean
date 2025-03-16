theorem hasInitial_of_equivalence (e : D ⥤ C) [e.IsEquivalence] [HasInitial C] : HasInitial D :=
  Adjunction.hasColimitsOfShape_of_equivalence e


theorem Equivalence.hasInitial_iff (e : C ≌ D) : HasInitial C ↔ HasInitial D :=
  ⟨fun (_ : HasInitial C) => hasInitial_of_equivalence e.inverse,
    fun (_ : HasInitial D) => hasInitial_of_equivalence e.functor⟩


theorem hasTerminal_of_equivalence (e : D ⥤ C) [e.IsEquivalence] [HasTerminal C] : HasTerminal D :=
  Adjunction.hasLimitsOfShape_of_equivalence e


theorem Equivalence.hasTerminal_iff (e : C ≌ D) : HasTerminal C ↔ HasTerminal D :=
  ⟨fun (_ : HasTerminal C) => hasTerminal_of_equivalence e.inverse,
    fun (_ : HasTerminal D) => hasTerminal_of_equivalence e.functor⟩


