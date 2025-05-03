local notation "|" x "|" => Finset.card x


local notation "‖" x "‖" => Fintype.card x


theorem card_embedding_eq_of_unique {α β : Type*} [Unique α] [Fintype β] [Fintype (α ↪ β)] :
    ‖α ↪ β‖ = ‖β‖ :=
  card_congr Equiv.uniqueEmbeddingEquivResult

-- Establishes the cardinality of the type of all injections between two finite types.
-- Porting note: `induction'` is broken so instead we make an ugly refine and `dsimp` a lot.

@[simp]
theorem card_embedding_eq {α β : Type*} [Fintype α] [Fintype β] [emb : Fintype (α ↪ β)] :
    ‖α ↪ β‖ = ‖β‖.descFactorial ‖α‖ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    emb : Fintype (Function.Embedding α β)
    ⊢ Eq (Fintype.card (Function.Embedding α β)) ((Fintype.card β).descFactorial ( …
  -/
  rw [Subsingleton.elim emb Embedding.fintype]
  refine Fintype.induction_empty_option (P := fun t ↦ ‖t ↪ β‖ = ‖β‖.descFactorial ‖t‖)
                                                              /-
                                                                case refine_1
                                                                α : Type u_1
                                                                β : Type u_2
                                                                inst✝¹ : Fintype α
                                                                inst✝ : Fintype β
                                                                emb : Fintype (Function.Embedding α β)
                                                                α₁ α₂ : Type u_1
                                                                h₂ : Fintype α₂
                                                                e : Equiv α₁ α₂
                                                                ih : (fun t [Fintype t] => Eq (Fintype.card (Function.Embedding t β)) ((Fintyp …
                                                                ⊢ (fun t [Fintype t] => Eq (Fintype.card (Function.Embedding t β)) ((Fintype.c …
                                                              -/
        (fun α₁ α₂ h₂ e ih ↦ ?_) (?_) (fun γ h ih ↦ ?_) α <;> dsimp only <;> clear! α
    /-
      case refine_1
      β : Type u_2
      inst✝ : Fintype β
      α₁ α₂ : Type u_1
      h₂ : Fintype α₂
      e : Equiv α₁ α₂
      ih : (fun t [Fintype t] => Eq (Fintype.card (Function.Embedding t β)) ((Fintyp …
      ⊢ Eq (Fintype.card (Function.Embedding α₂ β)) ((Fintype.card β).descFactorial  …
    -/
  · letI := Fintype.ofEquiv _ e.symm
    /-
      case refine_1
      β : Type u_2
      inst✝ : Fintype β
      α₁ α₂ : Type u_1
      h₂ : Fintype α₂
      e : Equiv α₁ α₂
      ih : (fun t [Fintype t] => Eq (Fintype.card (Function.Embedding t β)) ((Fintyp …
      this : Fintype α₁ := Fintype.ofEquiv α₂ e.symm
      ⊢ Eq (Fintype.card (Function.Embedding α₂ β)) ((Fintype.card β).descFactorial  …
    -/
    rw [← card_congr (Equiv.embeddingCongr e (Equiv.refl β)), ih, card_congr e]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      β : Type u_2
      inst✝ : Fintype β
      ⊢ Eq (Fintype.card (Function.Embedding PEmpty.{u_1 + 1} β)) ((Fintype.card β). …
    -/
  · rw [card_pempty, Nat.descFactorial_zero, card_eq_one_iff]
    /-
      case refine_2
      β : Type u_2
      inst✝ : Fintype β
      ⊢ Exists fun x => ∀ (y : Function.Embedding PEmpty.{u_1 + 1} β), Eq y x
    -/
    exact ⟨Embedding.ofIsEmpty, fun x ↦ DFunLike.ext _ _ isEmptyElim⟩
    /-
      🎉 no goals
    -/
  · classical
    dsimp only at ih
    rw [card_option, Nat.descFactorial_succ, card_congr (Embedding.optionEmbeddingEquiv γ β),
        card_sigma, ← ih]
    simp only [Fintype.card_compl_set, Fintype.card_range, Finset.sum_const, Finset.card_univ,
      Nat.nsmul_eq_mul, mul_comm]


/-- The cardinality of embeddings from an infinite type to a finite type is zero.
This is a re-statement of the pigeonhole principle. -/
theorem card_embedding_eq_of_infinite {α β : Type*} [Infinite α] [Finite β] [Fintype (α ↪ β)] :
    ‖α ↪ β‖ = 0 :=
  card_eq_zero


