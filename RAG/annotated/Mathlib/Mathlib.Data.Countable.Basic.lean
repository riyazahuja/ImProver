instance : Countable ℤ :=
  Countable.of_equiv ℕ Equiv.intEquivNat.symm


theorem countable_iff_nonempty_embedding : Countable α ↔ Nonempty (α ↪ ℕ) :=
  ⟨fun ⟨⟨f, hf⟩⟩ => ⟨⟨f, hf⟩⟩, fun ⟨f⟩ => ⟨⟨f, f.2⟩⟩⟩


theorem uncountable_iff_isEmpty_embedding : Uncountable α ↔ IsEmpty (α ↪ ℕ) := by
  /-
    α : Sort u
    ⊢ Iff (Uncountable α) (IsEmpty (Function.Embedding α Nat))
  -/
  rw [← not_countable_iff, countable_iff_nonempty_embedding, not_nonempty_iff]
  /-
    🎉 no goals
  -/


theorem nonempty_embedding_nat (α) [Countable α] : Nonempty (α ↪ ℕ) :=
  countable_iff_nonempty_embedding.1 ‹_›


protected theorem Function.Embedding.countable [Countable β] (f : α ↪ β) : Countable α :=
  f.injective.countable


protected lemma Function.Embedding.uncountable [Uncountable α] (f : α ↪ β) : Uncountable β :=
  f.injective.uncountable


instance [Countable α] [Countable β] : Countable (α ⊕ β) := by
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : Countable β
    ⊢ Countable (Sum α β)
  -/
  rcases exists_injective_nat α with ⟨f, hf⟩
  /-
    case intro
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : Countable β
    f : α → Nat
    hf : Function.Injective f
    ⊢ Countable (Sum α β)
  -/
  rcases exists_injective_nat β with ⟨g, hg⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : Countable β
    f : α → Nat
    hf : Function.Injective f
    g : β → Nat
    hg : Function.Injective g
    ⊢ Countable (Sum α β)
  -/
  exact (Equiv.natSumNatEquivNat.injective.comp <| hf.sum_map hg).countable
  /-
    🎉 no goals
  -/


instance Sum.uncountable_inl [Uncountable α] : Uncountable (α ⊕ β) :=
  inl_injective.uncountable


instance Sum.uncountable_inr [Uncountable β] : Uncountable (α ⊕ β) :=
  inr_injective.uncountable


instance Option.instCountable [Countable α] : Countable (Option α) :=
  Countable.of_equiv _ (Equiv.optionEquivSumPUnit.{0, _} α).symm


instance WithTop.instCountable [Countable α] : Countable (WithTop α) := Option.instCountable

instance WithBot.instCountable [Countable α] : Countable (WithBot α) := Option.instCountable

instance ENat.instCountable : Countable ℕ∞ := Option.instCountable


instance Option.instUncountable [Uncountable α] : Uncountable (Option α) :=
  Injective.uncountable fun _ _ ↦ Option.some_inj.1


instance WithTop.instUncountable [Uncountable α] : Uncountable (WithTop α) := Option.instUncountable

instance WithBot.instUncountable [Uncountable α] : Uncountable (WithBot α) := Option.instUncountable


instance [Countable α] [Countable β] : Countable (α × β) := by
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : Countable β
    ⊢ Countable (Prod α β)
  -/
  rcases exists_injective_nat α with ⟨f, hf⟩
  /-
    case intro
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : Countable β
    f : α → Nat
    hf : Function.Injective f
    ⊢ Countable (Prod α β)
  -/
  rcases exists_injective_nat β with ⟨g, hg⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : Countable β
    f : α → Nat
    hf : Function.Injective f
    g : β → Nat
    hg : Function.Injective g
    ⊢ Countable (Prod α β)
  -/
  exact (Nat.pairEquiv.injective.comp <| hf.prodMap hg).countable
  /-
    🎉 no goals
  -/


instance [Uncountable α] [Nonempty β] : Uncountable (α × β) := by
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Uncountable α
    inst✝ : Nonempty β
    ⊢ Uncountable (Prod α β)
  -/
  inhabit β
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Uncountable α
    inst✝ : Nonempty β
    inhabited_h : Inhabited β
    ⊢ Uncountable (Prod α β)
  -/
  exact (Prod.mk.inj_right default).uncountable
  /-
    🎉 no goals
  -/


instance [Nonempty α] [Uncountable β] : Uncountable (α × β) := by
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Nonempty α
    inst✝ : Uncountable β
    ⊢ Uncountable (Prod α β)
  -/
  inhabit α
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Nonempty α
    inst✝ : Uncountable β
    inhabited_h : Inhabited α
    ⊢ Uncountable (Prod α β)
  -/
  exact (Prod.mk.inj_left default).uncountable
  /-
    🎉 no goals
  -/


lemma countable_left_of_prod_of_nonempty [Nonempty β] (h : Countable (α × β)) : Countable α := by
  /-
    α : Type u
    β : Type v
    inst✝ : Nonempty β
    h : Countable (Prod α β)
    ⊢ Countable α
  -/
  contrapose h
  /-
    α : Type u
    β : Type v
    inst✝ : Nonempty β
    h : Not (Countable α)
    ⊢ Not (Countable (Prod α β))
  -/
  rw [not_countable_iff] at *
  /-
    α : Type u
    β : Type v
    inst✝ : Nonempty β
    h : Uncountable α
    ⊢ Uncountable (Prod α β)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma countable_right_of_prod_of_nonempty [Nonempty α] (h : Countable (α × β)) : Countable β := by
  /-
    α : Type u
    β : Type v
    inst✝ : Nonempty α
    h : Countable (Prod α β)
    ⊢ Countable β
  -/
  contrapose h
  /-
    α : Type u
    β : Type v
    inst✝ : Nonempty α
    h : Not (Countable β)
    ⊢ Not (Countable (Prod α β))
  -/
  rw [not_countable_iff] at *
  /-
    α : Type u
    β : Type v
    inst✝ : Nonempty α
    h : Uncountable β
    ⊢ Uncountable (Prod α β)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma countable_prod_swap [Countable (α × β)] : Countable (β × α) :=
  Countable.of_equiv _ (Equiv.prodComm α β)


instance [Countable α] [∀ a, Countable (π a)] : Countable (Sigma π) := by
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : ∀ (a : α), Countable (π a)
    ⊢ Countable (Sigma π)
  -/
  rcases exists_injective_nat α with ⟨f, hf⟩
  /-
    case intro
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : ∀ (a : α), Countable (π a)
    f : α → Nat
    hf : Function.Injective f
    ⊢ Countable (Sigma π)
  -/
  choose g hg using fun a => exists_injective_nat (π a)
  /-
    case intro
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Countable α
    inst✝ : ∀ (a : α), Countable (π a)
    f : α → Nat
    hf : Function.Injective f
    g : (a : α) → π a → Nat
    hg : ∀ (a : α), Function.Injective (g a)
    ⊢ Countable (Sigma π)
  -/
  exact ((Equiv.sigmaEquivProd ℕ ℕ).injective.comp <| hf.sigma_map hg).countable
  /-
    🎉 no goals
  -/


lemma Sigma.uncountable (a : α) [Uncountable (π a)] : Uncountable (Sigma π) :=
  (sigma_mk_injective (i := a)).uncountable


instance [Nonempty α] [∀ a, Uncountable (π a)] : Uncountable (Sigma π) := by
  /-
    α : Type u
    β : Type v
    π : α → Type w
    inst✝¹ : Nonempty α
    inst✝ : ∀ (a : α), Uncountable (π a)
    ⊢ Uncountable (Sigma π)
  -/
  inhabit α; exact Sigma.uncountable default
             /-
               🎉 no goals
             -/


instance (priority := 500) SetCoe.countable [Countable α] (s : Set α) : Countable s :=
  Subtype.countable


instance [Countable α] [Countable β] : Countable (α ⊕' β) :=
  Countable.of_equiv (PLift α ⊕ PLift β) (Equiv.plift.sumPSum Equiv.plift)


instance [Countable α] [Countable β] : Countable (PProd α β) :=
  Countable.of_equiv (PLift α × PLift β) (Equiv.plift.prodPProd Equiv.plift)


instance [Countable α] [∀ a, Countable (π a)] : Countable (PSigma π) :=
  Countable.of_equiv (Σa : PLift α, PLift (π a.down)) (Equiv.psigmaEquivSigmaPLift π).symm


instance [Finite α] [∀ a, Countable (π a)] : Countable (∀ a, π a) := by
  have : ∀ n, Countable (Fin n → ℕ) := by
    intro n
    induction' n with n ihn
    · change Countable (Fin 0 → ℕ); infer_instance
    · haveI := ihn
      exact Countable.of_equiv (ℕ × (Fin n → ℕ)) (Fin.consEquiv fun _ ↦ ℕ)
  /-
    α : Sort u
    β : Sort v
    π : α → Sort w
    inst✝¹ : Finite α
    inst✝ : ∀ (a : α), Countable (π a)
    this : ∀ (n : Nat), Countable (Fin n → Nat)
    ⊢ Countable ((a : α) → π a)
  -/
  rcases Finite.exists_equiv_fin α with ⟨n, ⟨e⟩⟩
  /-
    case intro.intro
    α : Sort u
    β : Sort v
    π : α → Sort w
    inst✝¹ : Finite α
    inst✝ : ∀ (a : α), Countable (π a)
    this : ∀ (n : Nat), Countable (Fin n → Nat)
    n : Nat
    e : Equiv α (Fin n)
    ⊢ Countable ((a : α) → π a)
  -/
  have f := fun a => (nonempty_embedding_nat (π a)).some
  /-
    case intro.intro
    α : Sort u
    β : Sort v
    π : α → Sort w
    inst✝¹ : Finite α
    inst✝ : ∀ (a : α), Countable (π a)
    this : ∀ (n : Nat), Countable (Fin n → Nat)
    n : Nat
    e : Equiv α (Fin n)
    f : (a : α) → Function.Embedding (π a) Nat
    ⊢ Countable ((a : α) → π a)
  -/
  exact ((Embedding.piCongrRight f).trans (Equiv.piCongrLeft' _ e).toEmbedding).countable
  /-
    🎉 no goals
  -/


