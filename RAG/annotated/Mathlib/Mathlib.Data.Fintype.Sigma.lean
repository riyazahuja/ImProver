lemma Set.biUnion_finsetSigma_univ (s : Finset ι) (f : Sigma κ → Set α) :
                                                                            /-
                                                                              ι : Type u_1
                                                                              α : Type u_2
                                                                              κ : ι → Type u_3
                                                                              inst✝ : (i : ι) → Fintype (κ i)
                                                                              s : Finset ι
                                                                              f : Sigma κ → Set α
                                                                              ⊢ Eq (Set.iUnion fun ij => Set.iUnion fun h => f ij) (Set.iUnion fun i => Set. …
                                                                            -/
    ⋃ ij ∈ s.sigma fun _ ↦ Finset.univ, f ij = ⋃ i ∈ s, ⋃ j, f ⟨i, j⟩ := by aesop
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


lemma Set.biUnion_finsetSigma_univ' (s : Finset ι) (f : Π i, κ i → Set α) :
                                                                                /-
                                                                                  ι : Type u_1
                                                                                  α : Type u_2
                                                                                  κ : ι → Type u_3
                                                                                  inst✝ : (i : ι) → Fintype (κ i)
                                                                                  s : Finset ι
                                                                                  f : (i : ι) → κ i → Set α
                                                                                  ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.iUnion fun j => f i j) (Set. …
                                                                                -/
    ⋃ i ∈ s, ⋃ j, f i j = ⋃ ij ∈ s.sigma fun _ ↦ Finset.univ, f ij.1 ij.2 := by aesop
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


lemma Set.biInter_finsetSigma_univ (s : Finset ι) (f : Sigma κ → Set α) :
                                                                            /-
                                                                              ι : Type u_1
                                                                              α : Type u_2
                                                                              κ : ι → Type u_3
                                                                              inst✝ : (i : ι) → Fintype (κ i)
                                                                              s : Finset ι
                                                                              f : Sigma κ → Set α
                                                                              ⊢ Eq (Set.iInter fun ij => Set.iInter fun h => f ij) (Set.iInter fun i => Set. …
                                                                            -/
    ⋂ ij ∈ s.sigma fun _ ↦ Finset.univ, f ij = ⋂ i ∈ s, ⋂ j, f ⟨i, j⟩ := by aesop
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


attribute [local simp] Sigma.forall in
lemma Set.biInter_finsetSigma_univ' (s : Finset ι) (f : Π i, κ i → Set α) :
                                                                                /-
                                                                                  ι : Type u_1
                                                                                  α : Type u_2
                                                                                  κ : ι → Type u_3
                                                                                  inst✝ : (i : ι) → Fintype (κ i)
                                                                                  s : Finset ι
                                                                                  f : (i : ι) → κ i → Set α
                                                                                  ⊢ Eq (Set.iInter fun i => Set.iInter fun h => Set.iInter fun j => f i j) (Set. …
                                                                                -/
    ⋂ i ∈ s, ⋂ j, f i j = ⋂ ij ∈ s.sigma fun _ ↦ Finset.univ, f ij.1 ij.2 := by aesop
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


                                                                                /-
                                                                                  ι : Type u_1
                                                                                  α : Type u_2
                                                                                  κ : ι → Type u_3
                                                                                  inst✝¹ : (i : ι) → Fintype (κ i)
                                                                                  inst✝ : Fintype ι
                                                                                  ⊢ ∀ (x : Sigma fun i => κ i), Membership.mem (Finset.univ.sigma fun x => Finse …
                                                                                -/
instance Sigma.instFintype : Fintype (Σ i, κ i) := ⟨univ.sigma fun _ ↦ univ, by simp⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

instance PSigma.instFintype : Fintype (Σ' i, κ i) := .ofEquiv _ (Equiv.psigmaEquivSigma _).symm


@[simp] lemma Finset.univ_sigma_univ : univ.sigma (fun _ ↦ univ) = (univ : Finset (Σ i, κ i)) := rfl

