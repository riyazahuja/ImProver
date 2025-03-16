/-- The sum of an indexed collection of matroids, as a matroid on the sigma-type. -/
protected def sigma (M : (i : ι) → Matroid (α i)) : Matroid ((i : ι) × α i) where
  E := univ.sigma (fun i ↦ (M i).E)
  Indep I := ∀ i, (M i).Indep (Sigma.mk i ⁻¹' I)
  Base B := ∀ i, (M i).Base (Sigma.mk i ⁻¹' B)

  indep_iff' I := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      I : Set (Sigma fun i => α i)
      ⊢ Iff ((fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I) (Exi …
    -/
    refine ⟨fun h ↦ ?_, fun ⟨B, hB, hIB⟩ i ↦ (hB i).indep.subset (preimage_mono hIB)⟩
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      I : Set (Sigma fun i => α i)
      h : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      ⊢ Exists fun B => And ((fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk …
    -/
    choose Bs hBs using fun i ↦ (h i).exists_base_superset
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      I : Set (Sigma fun i => α i)
      h : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      Bs : (i : ι) → Set (α i)
      hBs : ∀ (i : ι), And ((M i).Base (Bs i)) (HasSubset.Subset (Set.preimage (Sigm …
      ⊢ Exists fun B => And ((fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk …
    -/
    refine ⟨univ.sigma Bs, fun i ↦ by simpa using (hBs i).1, ?_⟩
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      I : Set (Sigma fun i => α i)
      h : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      Bs : (i : ι) → Set (α i)
      hBs : ∀ (i : ι), And ((M i).Base (Bs i)) (HasSubset.Subset (Set.preimage (Sigm …
      ⊢ HasSubset.Subset I (Set.univ.sigma Bs)
    -/
    rw [← univ_sigma_preimage_mk I]
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      I : Set (Sigma fun i => α i)
      h : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      Bs : (i : ι) → Set (α i)
      hBs : ∀ (i : ι), And ((M i).Base (Bs i)) (HasSubset.Subset (Set.preimage (Sigm …
      ⊢ HasSubset.Subset (Set.univ.sigma fun i => Set.preimage (Sigma.mk i) I) (Set. …
    -/
    refine sigma_mono rfl.subset fun i ↦ (hBs i).2
    /-
      🎉 no goals
    -/

  exists_base := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      ⊢ Exists fun B => (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B …
    -/
    choose B hB using fun i ↦ (M i).exists_base
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B : (i : ι) → Set (α i)
      hB : ∀ (i : ι), (M i).Base (B i)
      ⊢ Exists fun B => (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B …
    -/
    exact ⟨univ.sigma B, by simpa⟩
    /-
      🎉 no goals
    -/

  base_exchange B₁ B₂ h₁ h₂ := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      ⊢ ∀ (a : Sigma fun i => α i), Membership.mem (SDiff.sdiff B₁ B₂) a → Exists fu …
    -/
    simp only [mem_diff, Sigma.exists, and_imp, Sigma.forall]
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      ⊢ ∀ (a : ι) (b : α a), Membership.mem B₁ ⟨a, b⟩ → Not (Membership.mem B₂ ⟨a, b …
    -/
    intro i e he₁ he₂
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      i : ι
      e : α i
      he₁ : Membership.mem B₁ ⟨i, e⟩
      he₂ : Not (Membership.mem B₂ ⟨i, e⟩)
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem B₂ ⟨a, b⟩) (Not (Me …
    -/
    have hf_ex := (h₁ i).exchange (h₂ i) ⟨he₁, by simpa⟩
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      i : ι
      e : α i
      he₁ : Membership.mem B₁ ⟨i, e⟩
      he₂ : Not (Membership.mem B₂ ⟨i, e⟩)
      hf_ex : Exists fun y => And (Membership.mem (SDiff.sdiff (Set.preimage (Sigma. …
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem B₂ ⟨a, b⟩) (Not (Me …
    -/
    obtain ⟨f, ⟨hf₁, hf₂⟩, hfB⟩ := hf_ex
    /-
      case intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      i : ι
      e : α i
      he₁ : Membership.mem B₁ ⟨i, e⟩
      he₂ : Not (Membership.mem B₂ ⟨i, e⟩)
      f : α i
      hfB : (M i).Base (Insert.insert f (SDiff.sdiff (Set.preimage (Sigma.mk i) B₁)  …
      hf₁ : Membership.mem (Set.preimage (Sigma.mk i) B₂) f
      hf₂ : Not (Membership.mem (Set.preimage (Sigma.mk i) B₁) f)
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem B₂ ⟨a, b⟩) (Not (Me …
    -/
    refine ⟨i, f, ⟨hf₁, hf₂⟩, fun j ↦ ?_⟩
    /-
      case intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      i : ι
      e : α i
      he₁ : Membership.mem B₁ ⟨i, e⟩
      he₂ : Not (Membership.mem B₂ ⟨i, e⟩)
      f : α i
      hfB : (M i).Base (Insert.insert f (SDiff.sdiff (Set.preimage (Sigma.mk i) B₁)  …
      hf₁ : Membership.mem (Set.preimage (Sigma.mk i) B₂) f
      hf₂ : Not (Membership.mem (Set.preimage (Sigma.mk i) B₁) f)
      j : ι
      ⊢ (M j).Base (Set.preimage (Sigma.mk j) (Insert.insert ⟨i, f⟩ (SDiff.sdiff B₁  …
    -/
    rw [← union_singleton, preimage_union, preimage_diff]
    /-
      case intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      i : ι
      e : α i
      he₁ : Membership.mem B₁ ⟨i, e⟩
      he₂ : Not (Membership.mem B₂ ⟨i, e⟩)
      f : α i
      hfB : (M i).Base (Insert.insert f (SDiff.sdiff (Set.preimage (Sigma.mk i) B₁)  …
      hf₁ : Membership.mem (Set.preimage (Sigma.mk i) B₂) f
      hf₂ : Not (Membership.mem (Set.preimage (Sigma.mk i) B₁) f)
      j : ι
      ⊢ (M j).Base (Union.union (SDiff.sdiff (Set.preimage (Sigma.mk j) B₁) (Set.pre …
    -/
    obtain (rfl | hne) := eq_or_ne i j
    · simpa only [ show ∀ x, {⟨i,x⟩} = Sigma.mk i '' {x} by simp,
        preimage_image_eq _ sigma_mk_injective, union_singleton]
    rw [preimage_singleton_eq_empty.2 (by simpa), preimage_singleton_eq_empty.2 (by simpa),
      diff_empty, union_empty]
    /-
      case intro.intro.intro.inr
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B₁ B₂ : Set (Sigma fun i => α i)
      h₁ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₁
      h₂ : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B₂
      i : ι
      e : α i
      he₁ : Membership.mem B₁ ⟨i, e⟩
      he₂ : Not (Membership.mem B₂ ⟨i, e⟩)
      f : α i
      hfB : (M i).Base (Insert.insert f (SDiff.sdiff (Set.preimage (Sigma.mk i) B₁)  …
      hf₁ : Membership.mem (Set.preimage (Sigma.mk i) B₂) f
      hf₂ : Not (Membership.mem (Set.preimage (Sigma.mk i) B₁) f)
      j : ι
      hne : Ne i j
      ⊢ (M j).Base (Set.preimage (Sigma.mk j) B₁)
    -/
    exact h₁ j
    /-
      🎉 no goals
    -/

  maximality X _ I hI hIX := by
    choose Js hJs using
      fun i ↦ (hI i).subset_basis'_of_subset (preimage_mono (f := Sigma.mk i) hIX)

    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      ⊢ Exists fun J => And (HasSubset.Subset I J) (Maximal (fun K => And ((fun I => …
    -/
    use univ.sigma Js
    /-
      case h
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      ⊢ And (HasSubset.Subset I (Set.univ.sigma Js)) (Maximal (fun K => And ((fun I  …
    -/
    simp only [maximal_subset_iff', mem_univ, mk_preimage_sigma, le_eq_subset, and_imp]
    /-
      case h
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      ⊢ And (HasSubset.Subset I (Set.univ.sigma Js)) (And (And (∀ (i : ι), (M i).Ind …
    -/
    refine ⟨?_, ⟨fun i ↦ (hJs i).1.indep, ?_⟩, fun S hS hSX hJS ↦ ?_⟩
      /-
        case h.refine_1
        ι : Type u_1
        α : ι → Type u_2
        M✝ M : (i : ι) → Matroid (α i)
        X : Set (Sigma fun i => α i)
        x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
        I : Set (Sigma fun i => α i)
        hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
        hIX : HasSubset.Subset I X
        Js : (i : ι) → Set (α i)
        hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
        ⊢ HasSubset.Subset I (Set.univ.sigma Js)
      -/
    · rw [← univ_sigma_preimage_mk I]
      /-
        case h.refine_1
        ι : Type u_1
        α : ι → Type u_2
        M✝ M : (i : ι) → Matroid (α i)
        X : Set (Sigma fun i => α i)
        x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
        I : Set (Sigma fun i => α i)
        hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
        hIX : HasSubset.Subset I X
        Js : (i : ι) → Set (α i)
        hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
        ⊢ HasSubset.Subset (Set.univ.sigma fun i => Set.preimage (Sigma.mk i) I) (Set. …
      -/
      exact sigma_mono rfl.subset fun i ↦ (hJs i).2
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        ι : Type u_1
        α : ι → Type u_2
        M✝ M : (i : ι) → Matroid (α i)
        X : Set (Sigma fun i => α i)
        x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
        I : Set (Sigma fun i => α i)
        hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
        hIX : HasSubset.Subset I X
        Js : (i : ι) → Set (α i)
        hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
        ⊢ HasSubset.Subset (Set.univ.sigma Js) X
      -/
    · rw [← univ_sigma_preimage_mk X]
      /-
        case h.refine_2
        ι : Type u_1
        α : ι → Type u_2
        M✝ M : (i : ι) → Matroid (α i)
        X : Set (Sigma fun i => α i)
        x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
        I : Set (Sigma fun i => α i)
        hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
        hIX : HasSubset.Subset I X
        Js : (i : ι) → Set (α i)
        hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
        ⊢ HasSubset.Subset (Set.univ.sigma Js) (Set.univ.sigma fun i => Set.preimage ( …
      -/
      exact sigma_mono rfl.subset fun i ↦ (hJs i).1.subset
      /-
        🎉 no goals
      -/
    /-
      case h.refine_3
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      S : Set (Sigma fun i => α i)
      hS : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) S)
      hSX : HasSubset.Subset S X
      hJS : HasSubset.Subset (Set.univ.sigma Js) S
      ⊢ HasSubset.Subset S (Set.univ.sigma Js)
    -/
    rw [← univ_sigma_preimage_mk S]
    /-
      case h.refine_3
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      S : Set (Sigma fun i => α i)
      hS : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) S)
      hSX : HasSubset.Subset S X
      hJS : HasSubset.Subset (Set.univ.sigma Js) S
      ⊢ HasSubset.Subset (Set.univ.sigma fun i => Set.preimage (Sigma.mk i) S) (Set. …
    -/
    refine sigma_mono rfl.subset fun i ↦ ?_
    /-
      case h.refine_3
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      S : Set (Sigma fun i => α i)
      hS : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) S)
      hSX : HasSubset.Subset S X
      hJS : HasSubset.Subset (Set.univ.sigma Js) S
      i : ι
      ⊢ HasSubset.Subset (Set.preimage (Sigma.mk i) S) (Js i)
    -/
    rw [sigma_subset_iff] at hJS
    /-
      case h.refine_3
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      S : Set (Sigma fun i => α i)
      hS : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) S)
      hSX : HasSubset.Subset S X
      hJS : ∀ ⦃i : ι⦄, Membership.mem Set.univ i → ∀ ⦃a : α i⦄, Membership.mem (Js i …
      i : ι
      ⊢ HasSubset.Subset (Set.preimage (Sigma.mk i) S) (Js i)
    -/
    rw [(hJs i).1.eq_of_subset_indep (hS i) (hJS <| mem_univ i)]
    /-
      case h.refine_3
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      X : Set (Sigma fun i => α i)
      x✝ : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      I : Set (Sigma fun i => α i)
      hI : (fun I => ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)) I
      hIX : HasSubset.Subset I X
      Js : (i : ι) → Set (α i)
      hJs : ∀ (i : ι), And ((M i).Basis' (Js i) (Set.preimage (Sigma.mk i) X)) (HasS …
      S : Set (Sigma fun i => α i)
      hS : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) S)
      hSX : HasSubset.Subset S X
      hJS : ∀ ⦃i : ι⦄, Membership.mem Set.univ i → ∀ ⦃a : α i⦄, Membership.mem (Js i …
      i : ι
      ⊢ HasSubset.Subset (Set.preimage (Sigma.mk i) S) (Set.preimage (Sigma.mk i) X)
    -/
    exact preimage_mono hSX
    /-
      🎉 no goals
    -/

  subset_ground B hB := by
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B : Set (Sigma fun i => α i)
      hB : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B
      ⊢ HasSubset.Subset B (Set.univ.sigma fun i => (M i).E)
    -/
    rw [← univ_sigma_preimage_mk B]
    /-
      ι : Type u_1
      α : ι → Type u_2
      M✝ M : (i : ι) → Matroid (α i)
      B : Set (Sigma fun i => α i)
      hB : (fun B => ∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) B)) B
      ⊢ HasSubset.Subset (Set.univ.sigma fun i => Set.preimage (Sigma.mk i) B) (Set. …
    -/
    apply sigma_mono Subset.rfl fun i ↦ (hB i).subset_ground
    /-
      🎉 no goals
    -/


@[simp] lemma sigma_indep_iff {I} :
    (Matroid.sigma M).Indep I ↔ ∀ i, (M i).Indep (Sigma.mk i ⁻¹' I) := Iff.rfl


@[simp] lemma sigma_base_iff {B} :
    (Matroid.sigma M).Base B ↔ ∀ i, (M i).Base (Sigma.mk i ⁻¹' B) := Iff.rfl


@[simp] lemma sigma_ground_eq : (Matroid.sigma M).E = univ.sigma fun i ↦ (M i).E := rfl


@[simp] lemma sigma_basis_iff {I X} :
    (Matroid.sigma M).Basis I X ↔ ∀ i, (M i).Basis (Sigma.mk i ⁻¹' I) (Sigma.mk i ⁻¹' X) := by
  simp only [Basis, sigma_indep_iff, maximal_subset_iff, and_imp, and_assoc, sigma_ground_eq,
    forall_and, and_congr_right_iff]
  refine fun hI ↦ ⟨fun ⟨hIX, h, h'⟩ ↦ ⟨fun i ↦ preimage_mono hIX, fun i I₀ hI₀ hI₀X hII₀ ↦ ?_, ?_⟩,
    fun ⟨hIX, h', h''⟩ ↦ ⟨?_, ?_, ?_⟩⟩
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h : ∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i : ι), (M i).Indep (Set.preimage (S …
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      ⊢ Eq (Set.preimage (Sigma.mk i) I) I₀
    -/
  · refine hII₀.antisymm ?_
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h : ∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i : ι), (M i).Indep (Set.preimage (S …
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      ⊢ HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) I)
    -/
    specialize h (t := I ∪ Sigma.mk i '' I₀)
    simp only [preimage_union, union_subset_iff, hIX, image_subset_iff, hI₀X, and_self,
      subset_union_left, true_implies] at h
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
      ⊢ HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) I)
    -/
    rw [h, preimage_union, sigma_mk_preimage_image_eq_self]
      /-
        case refine_1
        ι : Type u_1
        α : ι → Type u_2
        M : (i : ι) → Matroid (α i)
        I X : Set (Sigma fun i => α i)
        hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
        x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
        hIX : HasSubset.Subset I X
        h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
        i : ι
        I₀ : Set (α i)
        hI₀ : (M i).Indep I₀
        hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
        hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
        h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
        ⊢ HasSubset.Subset I₀ (Union.union (Set.preimage (Sigma.mk i) I) I₀)
      -/
    · exact subset_union_right
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
      ⊢ ∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) (Set …
    -/
    intro j
    /-
      case refine_1
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
      j : ι
      ⊢ (M j).Indep (Union.union (Set.preimage (Sigma.mk j) I) (Set.preimage (Sigma. …
    -/
    obtain (rfl | hij) := eq_or_ne i j
      /-
        case refine_1.inl
        ι : Type u_1
        α : ι → Type u_2
        M : (i : ι) → Matroid (α i)
        I X : Set (Sigma fun i => α i)
        hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
        x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
        hIX : HasSubset.Subset I X
        h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
        i : ι
        I₀ : Set (α i)
        hI₀ : (M i).Indep I₀
        hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
        hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
        h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
        ⊢ (M i).Indep (Union.union (Set.preimage (Sigma.mk i) I) (Set.preimage (Sigma. …
      -/
    · rwa [sigma_mk_preimage_image_eq_self, union_eq_self_of_subset_left hII₀]
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
      j : ι
      hij : Ne i j
      ⊢ (M j).Indep (Union.union (Set.preimage (Sigma.mk j) I) (Set.preimage (Sigma. …
    -/
    rw [sigma_mk_preimage_image' hij, union_empty]
    /-
      case refine_1.inr
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      i : ι
      I₀ : Set (α i)
      hI₀ : (M i).Indep I₀
      hI₀X : HasSubset.Subset I₀ (Set.preimage (Sigma.mk i) X)
      hII₀ : HasSubset.Subset (Set.preimage (Sigma.mk i) I) I₀
      h : (∀ (i_1 : ι), (M i_1).Indep (Union.union (Set.preimage (Sigma.mk i_1) I) ( …
      j : ι
      hij : Ne i j
      ⊢ (M j).Indep (Set.preimage (Sigma.mk j) I)
    -/
    apply hI
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (HasSubset.Subset I X) (And (∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i  …
      hIX : HasSubset.Subset I X
      h : ∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i : ι), (M i).Indep (Set.preimage (S …
      h' : HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
      ⊢ ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) X) (M x).E
    -/
  · exact fun i ↦ by simpa using preimage_mono (f := Sigma.mk i) h'
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preim …
      hIX : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preimage  …
      h' : ∀ (x : ι) ⦃t : Set (α x)⦄, (M x).Indep t → HasSubset.Subset t (Set.preima …
      h'' : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) X) (M x).E
      ⊢ HasSubset.Subset I X
    -/
  · exact fun ⟨i, x⟩ hx ↦ by simpa using hIX i hx
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝ : And (∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preim …
      hIX : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preimage  …
      h' : ∀ (x : ι) ⦃t : Set (α x)⦄, (M x).Indep t → HasSubset.Subset t (Set.preima …
      h'' : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) X) (M x).E
      ⊢ ∀ ⦃t : Set (Sigma fun i => α i)⦄, (∀ (i : ι), (M i).Indep (Set.preimage (Sig …
    -/
  · refine fun J hJ hJX hIJ ↦ hIJ.antisymm fun ⟨i,x⟩ hx ↦ ?_
    /-
      case refine_4
      ι : Type u_1
      α : ι → Type u_2
      M : (i : ι) → Matroid (α i)
      I X : Set (Sigma fun i => α i)
      hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
      x✝¹ : And (∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.prei …
      hIX : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preimage  …
      h' : ∀ (x : ι) ⦃t : Set (α x)⦄, (M x).Indep t → HasSubset.Subset t (Set.preima …
      h'' : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) X) (M x).E
      J : Set (Sigma fun i => α i)
      hJ : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) J)
      hJX : HasSubset.Subset J X
      hIJ : HasSubset.Subset I J
      x✝ : Sigma fun i => α i
      i : ι
      x : α i
      hx : Membership.mem J ⟨i, x⟩
      ⊢ Membership.mem I ⟨i, x⟩
    -/
    simpa using (h' i (hJ i) (preimage_mono hJX) (preimage_mono hIJ)).symm.subset hx
    /-
      🎉 no goals
    -/
  /-
    case refine_5
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    I X : Set (Sigma fun i => α i)
    hI : ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
    x✝ : And (∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preim …
    hIX : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) I) (Set.preimage  …
    h' : ∀ (x : ι) ⦃t : Set (α x)⦄, (M x).Indep t → HasSubset.Subset t (Set.preima …
    h'' : ∀ (x : ι), HasSubset.Subset (Set.preimage (Sigma.mk x) X) (M x).E
    ⊢ HasSubset.Subset X (Set.univ.sigma fun i => (M i).E)
  -/
  exact fun ⟨i,x⟩ hx ↦ by simpa using h'' i hx
  /-
    🎉 no goals
  -/


lemma Finitary.sigma (h : ∀ i, (M i).Finitary) : (Matroid.sigma M).Finitary := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    ⊢ (Matroid.sigma M).Finitary
  -/
  refine ⟨fun I hI ↦ ?_⟩
  /-
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    I : Set (Sigma fun i => α i)
    hI : ∀ (J : Set (Sigma fun i => α i)), HasSubset.Subset J I → J.Finite → (Matr …
    ⊢ (Matroid.sigma M).Indep I
  -/
  simp only [sigma_indep_iff] at hI ⊢
  /-
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    I : Set (Sigma fun i => α i)
    hI : ∀ (J : Set (Sigma fun i => α i)), HasSubset.Subset J I → J.Finite → ∀ (i  …
    ⊢ ∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) I)
  -/
  intro i
  /-
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    I : Set (Sigma fun i => α i)
    hI : ∀ (J : Set (Sigma fun i => α i)), HasSubset.Subset J I → J.Finite → ∀ (i  …
    i : ι
    ⊢ (M i).Indep (Set.preimage (Sigma.mk i) I)
  -/
  apply indep_of_forall_finite_subset_indep
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    I : Set (Sigma fun i => α i)
    hI : ∀ (J : Set (Sigma fun i => α i)), HasSubset.Subset J I → J.Finite → ∀ (i  …
    i : ι
    ⊢ ∀ (J : Set (α i)), HasSubset.Subset J (Set.preimage (Sigma.mk i) I) → J.Fini …
  -/
  intro J hJI hJ
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    I : Set (Sigma fun i => α i)
    hI : ∀ (J : Set (Sigma fun i => α i)), HasSubset.Subset J I → J.Finite → ∀ (i  …
    i : ι
    J : Set (α i)
    hJI : HasSubset.Subset J (Set.preimage (Sigma.mk i) I)
    hJ : J.Finite
    ⊢ (M i).Indep J
  -/
  convert hI (Sigma.mk i '' J) (by simpa) (hJ.image _) i
  /-
    case h.e'_3
    ι : Type u_1
    α : ι → Type u_2
    M : (i : ι) → Matroid (α i)
    h : ∀ (i : ι), (M i).Finitary
    I : Set (Sigma fun i => α i)
    hI : ∀ (J : Set (Sigma fun i => α i)), HasSubset.Subset J I → J.Finite → ∀ (i  …
    i : ι
    J : Set (α i)
    hJI : HasSubset.Subset J (Set.preimage (Sigma.mk i) I)
    hJ : J.Finite
    ⊢ Eq J (Set.preimage (Sigma.mk i) (Set.image (Sigma.mk i) J))
  -/
  rw [sigma_mk_preimage_image_eq_self]
  /-
    🎉 no goals
  -/


/-- The sum of an indexed family `M : ι → Matroid α` of matroids on the same type,
as a matroid on the product type `ι × α`. -/
protected def sum' (M : ι → Matroid α) : Matroid (ι × α) :=
  (Matroid.sigma M).mapEquiv <| Equiv.sigmaEquivProd ι α


@[simp] lemma sum'_indep_iff {I} :
    (Matroid.sum' M).Indep I ↔ ∀ i, (M i).Indep (Prod.mk i ⁻¹' I) := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I : Set (Prod ι α)
    ⊢ Iff ((Matroid.sum' M).Indep I) (∀ (i : ι), (M i).Indep (Set.preimage (Prod.m …
  -/
  simp only [Matroid.sum', mapEquiv_indep_iff, Equiv.sigmaEquivProd_symm_apply, sigma_indep_iff]
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I : Set (Prod ι α)
    ⊢ Iff (∀ (i : ι), (M i).Indep (Set.preimage (Sigma.mk i) (Set.image (fun a =>  …
  -/
  convert Iff.rfl
  /-
    case h.e'_2.h.h.e'_3
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I : Set (Prod ι α)
    a✝ : ι
    ⊢ Eq (Set.preimage (Prod.mk a✝) I) (Set.preimage (Sigma.mk a✝) (Set.image (fun …
  -/
  ext
  /-
    case h.e'_2.h.h.e'_3.h
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I : Set (Prod ι α)
    a✝ : ι
    x✝ : α
    ⊢ Iff (Membership.mem (Set.preimage (Prod.mk a✝) I) x✝) (Membership.mem (Set.p …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] lemma sum'_ground_eq (M : ι → Matroid α) :
    (Matroid.sum' M).E = ⋃ i, Prod.mk i '' (M i).E := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    ⊢ Eq (Matroid.sum' M).E (Set.iUnion fun i => Set.image (Prod.mk i) (M i).E)
  -/
  ext
  /-
    case h
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    x✝ : Prod ι α
    ⊢ Iff (Membership.mem (Matroid.sum' M).E x✝) (Membership.mem (Set.iUnion fun i …
  -/
  simp [Matroid.sum']
  /-
    🎉 no goals
  -/


@[simp] lemma sum'_base_iff {B} : (Matroid.sum' M).Base B ↔ ∀ i, (M i).Base (Prod.mk i ⁻¹' B) := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    B : Set (Prod ι α)
    ⊢ Iff ((Matroid.sum' M).Base B) (∀ (i : ι), (M i).Base (Set.preimage (Prod.mk  …
  -/
  simp only [Matroid.sum', mapEquiv_base_iff, Equiv.sigmaEquivProd_symm_apply, sigma_base_iff]
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    B : Set (Prod ι α)
    ⊢ Iff (∀ (i : ι), (M i).Base (Set.preimage (Sigma.mk i) (Set.image (fun a => ⟨ …
  -/
  convert Iff.rfl
  /-
    case h.e'_2.h.h.e'_3
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    B : Set (Prod ι α)
    a✝ : ι
    ⊢ Eq (Set.preimage (Prod.mk a✝) B) (Set.preimage (Sigma.mk a✝) (Set.image (fun …
  -/
  ext
  /-
    case h.e'_2.h.h.e'_3.h
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    B : Set (Prod ι α)
    a✝ : ι
    x✝ : α
    ⊢ Iff (Membership.mem (Set.preimage (Prod.mk a✝) B) x✝) (Membership.mem (Set.p …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] lemma sum'_basis_iff {I X} :
    (Matroid.sum' M).Basis I X ↔ ∀ i, (M i).Basis (Prod.mk i ⁻¹' I) (Prod.mk i ⁻¹' X) := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I X : Set (Prod ι α)
    ⊢ Iff ((Matroid.sum' M).Basis I X) (∀ (i : ι), (M i).Basis (Set.preimage (Prod …
  -/
  simp [Matroid.sum']
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I X : Set (Prod ι α)
    ⊢ Iff (∀ (i : ι), (M i).Basis (Set.preimage (Sigma.mk i) (Set.image (fun a =>  …
  -/
  convert Iff.rfl <;>
  /-
    case h.e'_2.h.h.e'_3
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    I X : Set (Prod ι α)
    a✝ : ι
    ⊢ Eq (Set.preimage (Prod.mk a✝) I) (Set.preimage (Sigma.mk a✝) (Set.image (fun …
  -/
  /-
    🎉 no goals
  -/
  exact ext <| by simp
  /-
    🎉 no goals
  -/


lemma Finitary.sum' (h : ∀ i, (M i).Finitary) : (Matroid.sum' M).Finitary := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : ∀ (i : ι), (M i).Finitary
    ⊢ (Matroid.sum' M).Finitary
  -/
  have := Finitary.sigma h
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : ∀ (i : ι), (M i).Finitary
    this : (Matroid.sigma M).Finitary
    ⊢ (Matroid.sum' M).Finitary
  -/
  rw [Matroid.sum']
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : ∀ (i : ι), (M i).Finitary
    this : (Matroid.sigma M).Finitary
    ⊢ ((Matroid.sigma M).mapEquiv (Equiv.sigmaEquivProd ι α)).Finitary
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The sum of an indexed collection of matroids on `α` with pairwise disjoint ground sets,
as a matroid on `α` -/
protected def disjointSigma (M : ι → Matroid α) (h : Pairwise (Disjoint on fun i ↦ (M i).E)) :
    Matroid α :=
  (Matroid.sigma (fun i ↦ (M i).restrictSubtype (M i).E)).mapEmbedding
    (Function.Embedding.sigmaSet h)


@[simp] lemma disjointSigma_ground_eq {h} : (Matroid.disjointSigma M h).E = ⋃ i : ι, (M i).E := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : Pairwise (Function.onFun Disjoint fun i => (M i).E)
    ⊢ Eq (Matroid.disjointSigma M h).E (Set.iUnion fun i => (M i).E)
  -/
  ext; simp [Matroid.disjointSigma, mapEmbedding, restrictSubtype]
       /-
         🎉 no goals
       -/


@[simp] lemma disjointSigma_indep_iff {h I} :
    (Matroid.disjointSigma M h).Indep I ↔
      (∀ i, (M i).Indep (I ∩ (M i).E)) ∧ I ⊆ ⋃ i, (M i).E := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : Pairwise (Function.onFun Disjoint fun i => (M i).E)
    I : Set α
    ⊢ Iff ((Matroid.disjointSigma M h).Indep I) (And (∀ (i : ι), (M i).Indep (Inte …
  -/
  simp [Matroid.disjointSigma, (Function.Embedding.sigmaSet_preimage h)]
  /-
    🎉 no goals
  -/


@[simp] lemma disjointSigma_base_iff {h B} :
    (Matroid.disjointSigma M h).Base B ↔
      (∀ i, (M i).Base (B ∩ (M i).E)) ∧ B ⊆ ⋃ i, (M i).E := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : Pairwise (Function.onFun Disjoint fun i => (M i).E)
    B : Set α
    ⊢ Iff ((Matroid.disjointSigma M h).Base B) (And (∀ (i : ι), (M i).Base (Inter. …
  -/
  simp [Matroid.disjointSigma, (Function.Embedding.sigmaSet_preimage h)]
  /-
    🎉 no goals
  -/


@[simp] lemma disjointSigma_basis_iff {h I X} :
    (Matroid.disjointSigma M h).Basis I X ↔
      (∀ i, (M i).Basis (I ∩ (M i).E) (X ∩ (M i).E)) ∧ I ⊆ X ∧ X ⊆ ⋃ i, (M i).E := by
  /-
    α : Type u_1
    ι : Type u_2
    M : ι → Matroid α
    h : Pairwise (Function.onFun Disjoint fun i => (M i).E)
    I X : Set α
    ⊢ Iff ((Matroid.disjointSigma M h).Basis I X) (And (∀ (i : ι), (M i).Basis (In …
  -/
  simp [Matroid.disjointSigma, Function.Embedding.sigmaSet_preimage h]
  /-
    🎉 no goals
  -/


/-- The sum of two matroids as a matroid on the sum type. -/
protected def sum (M : Matroid α) (N : Matroid β) : Matroid (α ⊕ β) :=
  let S := Matroid.sigma (Bool.rec (M.mapEquiv Equiv.ulift.symm) (N.mapEquiv Equiv.ulift.symm))
  let e := Equiv.sumEquivSigmaBool (ULift.{v} α) (ULift.{u} β)
  (S.mapEquiv e.symm).mapEquiv (Equiv.sumCongr Equiv.ulift Equiv.ulift)


@[simp] lemma sum_ground (M : Matroid α) (N : Matroid β) :
    (M.sum N).E = (.inl '' M.E) ∪ (.inr '' N.E) := by
  /-
    α : Type u
    β : Type v
    M : Matroid α
    N : Matroid β
    ⊢ Eq (M.sum N).E (Union.union (Set.image Sum.inl M.E) (Set.image Sum.inr N.E))
  -/
  simp [Matroid.sum, Set.ext_iff, mapEquiv, mapEmbedding, Equiv.ulift, Equiv.sumEquivSigmaBool]
  /-
    🎉 no goals
  -/


@[simp] lemma sum_indep_iff (M : Matroid α) (N : Matroid β) {I : Set (α ⊕ β)} :
    (M.sum N).Indep I ↔ M.Indep (.inl ⁻¹' I) ∧ N.Indep (.inr ⁻¹' I) := by
  simp only [Matroid.sum, mapEquiv_indep_iff, Equiv.sumCongr_symm, Equiv.sumCongr_apply,
    Equiv.symm_symm, sigma_indep_iff, Bool.forall_bool, Equiv.ulift_apply]
  /-
    α : Type u
    β : Type v
    M : Matroid α
    N : Matroid β
    I : Set (Sum α β)
    ⊢ Iff (And (M.Indep (Set.image (fun a => a.down) (Set.preimage (Sigma.mk Bool. …
  -/
  convert Iff.rfl <;>
    /-
      case h.e'_2.h.e'_1.h.e'_3
      α : Type u
      β : Type v
      M : Matroid α
      N : Matroid β
      I : Set (Sum α β)
      ⊢ Eq (Set.preimage Sum.inl I) (Set.image (fun a => a.down) (Set.preimage (Sigm …
    -/
    /-
      🎉 no goals
    -/
    simp [Set.ext_iff, Equiv.ulift, Equiv.sumEquivSigmaBool]
    /-
      🎉 no goals
    -/


@[simp] lemma sum_base_iff {M : Matroid α} {N : Matroid β} {B : Set (α ⊕ β)} :
    (M.sum N).Base B ↔ M.Base (.inl ⁻¹' B) ∧ N.Base (.inr ⁻¹' B) := by
  simp only [Matroid.sum, mapEquiv_base_iff, Equiv.sumCongr_symm, Equiv.sumCongr_apply,
    Equiv.symm_symm, sigma_base_iff, Bool.forall_bool, Equiv.ulift_apply]
  /-
    α : Type u
    β : Type v
    M : Matroid α
    N : Matroid β
    B : Set (Sum α β)
    ⊢ Iff (And (M.Base (Set.image (fun a => a.down) (Set.preimage (Sigma.mk Bool.f …
  -/
  convert Iff.rfl <;>
    /-
      case h.e'_2.h.e'_1.h.e'_3
      α : Type u
      β : Type v
      M : Matroid α
      N : Matroid β
      B : Set (Sum α β)
      ⊢ Eq (Set.preimage Sum.inl B) (Set.image (fun a => a.down) (Set.preimage (Sigm …
    -/
    /-
      🎉 no goals
    -/
    simp [Set.ext_iff, Equiv.ulift, Equiv.sumEquivSigmaBool]
    /-
      🎉 no goals
    -/


@[simp] lemma sum_basis_iff {M : Matroid α} {N : Matroid β} {I X : Set (α ⊕ β)} :
    (M.sum N).Basis I X ↔
      (M.Basis (Sum.inl ⁻¹' I) (Sum.inl ⁻¹' X) ∧ N.Basis (Sum.inr ⁻¹' I) (Sum.inr ⁻¹' X)) := by
  simp only [Matroid.sum, mapEquiv_basis_iff, Equiv.sumCongr_symm,
    Equiv.sumCongr_apply, Equiv.symm_symm, sigma_basis_iff, Bool.forall_bool, Equiv.ulift_apply,
    Equiv.sumEquivSigmaBool, Equiv.coe_fn_mk, Equiv.ulift]
  /-
    α : Type u
    β : Type v
    M : Matroid α
    N : Matroid β
    I X : Set (Sum α β)
    ⊢ Iff (And (M.Basis (Set.image (fun a => a.down) (Set.preimage (Sigma.mk Bool. …
  -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
  convert Iff.rfl <;> exact ext <| by simp
                      /-
                        🎉 no goals
                      -/


/-- The sum of two matroids on `α` with disjoint ground sets, as a `Matroid α`. -/
def disjointSum (M N : Matroid α) (h : Disjoint M.E N.E) : Matroid α :=
  ((M.restrictSubtype M.E).sum (N.restrictSubtype N.E)).mapEmbedding <| Function.Embedding.sumSet h


@[simp] lemma disjointSum_ground_eq {h} : (M.disjointSum N h).E = M.E ∪ N.E := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    ⊢ Eq (M.disjointSum N h).E (Union.union M.E N.E)
  -/
  simp [disjointSum, restrictSubtype, mapEmbedding]
  /-
    🎉 no goals
  -/


@[simp] lemma disjointSum_indep_iff {h I} :
    (M.disjointSum N h).Indep I ↔ M.Indep (I ∩ M.E) ∧ N.Indep (I ∩ N.E) ∧ I ⊆ M.E ∪ N.E := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    I : Set α
    ⊢ Iff ((M.disjointSum N h).Indep I) (And (M.Indep (Inter.inter I M.E)) (And (N …
  -/
  simp [disjointSum, and_assoc]
  /-
    🎉 no goals
  -/


@[simp] lemma disjointSum_base_iff {h B} :
    (M.disjointSum N h).Base B ↔ M.Base (B ∩ M.E) ∧ N.Base (B ∩ N.E) ∧ B ⊆ M.E ∪ N.E := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    B : Set α
    ⊢ Iff ((M.disjointSum N h).Base B) (And (M.Base (Inter.inter B M.E)) (And (N.B …
  -/
  simp [disjointSum, and_assoc]
  /-
    🎉 no goals
  -/


@[simp] lemma disjointSum_basis_iff {h I X} :
    (M.disjointSum N h).Basis I X ↔ M.Basis (I ∩ M.E) (X ∩ M.E) ∧
      N.Basis (I ∩ N.E) (X ∩ N.E) ∧ I ⊆ X ∧ X ⊆ M.E ∪ N.E := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    I X : Set α
    ⊢ Iff ((M.disjointSum N h).Basis I X) (And (M.Basis (Inter.inter I M.E) (Inter …
  -/
  simp [disjointSum, and_assoc]
  /-
    🎉 no goals
  -/


lemma disjointSum_comm {h} : M.disjointSum N h = N.disjointSum M h.symm := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    ⊢ Eq (M.disjointSum N h) (N.disjointSum M ⋯)
  -/
  ext
    /-
      case hE.h
      α : Type u_1
      M N : Matroid α
      h : Disjoint M.E N.E
      x✝ : α
      ⊢ Iff (Membership.mem (M.disjointSum N h).E x✝) (Membership.mem (N.disjointSum …
    -/
  · simp [union_comm]
    /-
      🎉 no goals
    -/
  repeat simpa [union_comm] using ⟨fun ⟨m, n, h⟩ ↦ ⟨n, m, M.E.union_comm N.E ▸ h⟩,
    fun ⟨n, m, h⟩ ↦ ⟨m, n, M.E.union_comm N.E ▸ h⟩⟩


lemma Indep.eq_union_image_of_disjointSum {h I} (hI : (disjointSum M N h).Indep I) :
    ∃ IM IN, M.Indep IM ∧ N.Indep IN ∧ Disjoint IM IN ∧ I = IM ∪ IN := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    I : Set α
    hI : (M.disjointSum N h).Indep I
    ⊢ Exists fun IM => Exists fun IN => And (M.Indep IM) (And (N.Indep IN) (And (D …
  -/
  rw [disjointSum_indep_iff] at hI
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    I : Set α
    hI : And (M.Indep (Inter.inter I M.E)) (And (N.Indep (Inter.inter I N.E)) (Has …
    ⊢ Exists fun IM => Exists fun IN => And (M.Indep IM) (And (N.Indep IN) (And (D …
  -/
  refine ⟨_, _, hI.1, hI.2.1, h.mono inter_subset_right inter_subset_right, ?_⟩
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    I : Set α
    hI : And (M.Indep (Inter.inter I M.E)) (And (N.Indep (Inter.inter I N.E)) (Has …
    ⊢ Eq I (Union.union (Inter.inter I M.E) (Inter.inter I N.E))
  -/
  rw [← inter_union_distrib_left, inter_eq_self_of_subset_left hI.2.2]
  /-
    🎉 no goals
  -/


lemma Base.eq_union_image_of_disjointSum {h B} (hB : (disjointSum M N h).Base B) :
    ∃ BM BN, M.Base BM ∧ N.Base BN ∧ Disjoint BM BN ∧ B = BM ∪ BN := by
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    B : Set α
    hB : (M.disjointSum N h).Base B
    ⊢ Exists fun BM => Exists fun BN => And (M.Base BM) (And (N.Base BN) (And (Dis …
  -/
  rw [disjointSum_base_iff] at hB
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    B : Set α
    hB : And (M.Base (Inter.inter B M.E)) (And (N.Base (Inter.inter B N.E)) (HasSu …
    ⊢ Exists fun BM => Exists fun BN => And (M.Base BM) (And (N.Base BN) (And (Dis …
  -/
  refine ⟨_, _, hB.1, hB.2.1, h.mono inter_subset_right inter_subset_right, ?_⟩
  /-
    α : Type u_1
    M N : Matroid α
    h : Disjoint M.E N.E
    B : Set α
    hB : And (M.Base (Inter.inter B M.E)) (And (N.Base (Inter.inter B N.E)) (HasSu …
    ⊢ Eq B (Union.union (Inter.inter B M.E) (Inter.inter B N.E))
  -/
  rw [← inter_union_distrib_left, inter_eq_self_of_subset_left hB.2.2]
  /-
    🎉 no goals
  -/


