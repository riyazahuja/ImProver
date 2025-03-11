instance (priority := 100) Countable.toSmall (α : Type v) [Countable α] : Small.{w} α :=
  let ⟨_, hf⟩ := exists_injective_nat α
  small_of_injective hf


@[deprecated (since := "2024-03-20"), nolint defLemma]
alias small_of_countable := Countable.toSmall


@[deprecated (since := "2024-03-20"), nolint defLemma]
alias small_of_fintype := Countable.toSmall

