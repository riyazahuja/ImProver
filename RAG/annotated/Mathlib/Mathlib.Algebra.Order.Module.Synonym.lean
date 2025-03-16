instance instModule [Semiring α] [AddCommMonoid β] [Module α β] : Module αᵒᵈ β where
  add_smul := add_smul (R := α)
  zero_smul := zero_smul _


instance instModule' [Semiring α] [AddCommMonoid β] [Module α β] : Module α βᵒᵈ where
  add_smul := add_smul (M := β)
  zero_smul := zero_smul _


instance instModule [Semiring α] [AddCommMonoid β] [Module α β] : Module (Lex α) β :=
  ‹Module α β›


instance instModule' [Semiring α] [AddCommMonoid β] [Module α β] : Module α (Lex β) :=
  ‹Module α β›


