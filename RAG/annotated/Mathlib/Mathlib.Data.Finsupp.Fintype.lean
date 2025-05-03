noncomputable instance Finsupp.fintype : Fintype (ι →₀ α) :=
  Fintype.ofEquiv _ Finsupp.equivFunOnFinite.symm


instance Finsupp.infinite_of_left [Nontrivial α] [Infinite ι] : Infinite (ι →₀ α) :=
  let ⟨_, hm⟩ := exists_ne (0 : α)
  Infinite.of_injective _ <| Finsupp.single_left_injective hm


instance Finsupp.infinite_of_right [Infinite α] [Nonempty ι] : Infinite (ι →₀ α) :=
  Infinite.of_injective (fun i => Finsupp.single (Classical.arbitrary ι) i)
    (Finsupp.single_injective (Classical.arbitrary ι))


variable (ι α) in
@[simp] lemma Fintype.card_finsupp : card (ι →₀ α) = card α ^ card ι := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    inst✝¹ : Zero α
    inst✝ : Fintype α
    ⊢ Eq (Fintype.card (Finsupp ι α)) (HPow.hPow (Fintype.card α) (Fintype.card ι))
  -/
  simp [card_congr Finsupp.equivFunOnFinite]
  /-
    🎉 no goals
  -/

