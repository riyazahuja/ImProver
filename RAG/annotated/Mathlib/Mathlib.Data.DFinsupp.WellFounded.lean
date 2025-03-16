/-- This key lemma says that if a finitely supported dependent function `x₀` is obtained by merging
  two such functions `x₁` and `x₂`, and if we evolve `x₀` down the `DFinsupp.Lex` relation one
  step and get `x`, we can always evolve one of `x₁` and `x₂` down the `DFinsupp.Lex` relation
  one step while keeping the other unchanged, and merge them back (possibly in a different way)
  to get back `x`. In other words, the two parts evolve essentially independently under
  `DFinsupp.Lex`. This is used to show that a function `x` is accessible if
  `DFinsupp.single i (x i)` is accessible for each `i` in the (finite) support of `x`
  (`DFinsupp.Lex.acc_of_single`). -/
theorem lex_fibration [∀ (i) (s : Set ι), Decidable (i ∈ s)] :
    Fibration (InvImage (GameAdd (DFinsupp.Lex r s) (DFinsupp.Lex r s)) snd) (DFinsupp.Lex r s)
      fun x => piecewise x.2.1 x.2.2 x.1 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
    ⊢ Relation.Fibration (InvImage (Prod.GameAdd (DFinsupp.Lex r s) (DFinsupp.Lex  …
  -/
  rintro ⟨p, x₁, x₂⟩ x ⟨i, hr, hs⟩
  /-
    case mk.mk.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
    p : Set ι
    x₁ x₂ x : DFinsupp fun i => α i
    i : ι
    hr : ∀ (j : ι), r j i → Eq (x j) (((fun x => x.2.1.piecewise x.2.2 x.1) { fst  …
    hs : s i (x i) (((fun x => x.2.1.piecewise x.2.2 x.1) { fst := p, snd := { fst …
    ⊢ Exists fun a' => And (InvImage (Prod.GameAdd (DFinsupp.Lex r s) (DFinsupp.Le …
  -/
  simp_rw [piecewise_apply] at hs hr
  /-
    case mk.mk.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
    p : Set ι
    x₁ x₂ x : DFinsupp fun i => α i
    i : ι
    hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
    hs : s i (x i) (ite (Membership.mem p i) (x₁ i) (x₂ i))
    ⊢ Exists fun a' => And (InvImage (Prod.GameAdd (DFinsupp.Lex r s) (DFinsupp.Le …
  -/
  split_ifs at hs with hp
  · refine ⟨⟨{ j | r j i → j ∈ p }, piecewise x₁ x { j | r j i }, x₂⟩,
                                           /-
                                             case pos.refine_1
                                             ι : Type u_1
                                             α : ι → Type u_2
                                             inst✝¹ : (i : ι) → Zero (α i)
                                             r : ι → ι → Prop
                                             s : (i : ι) → α i → α i → Prop
                                             inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
                                             p : Set ι
                                             x₁ x₂ x : DFinsupp fun i => α i
                                             i : ι
                                             hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
                                             hp : Membership.mem p i
                                             hs : s i (x i) (x₁ i)
                                             j : ι
                                             hj : r j i
                                             ⊢ Eq ((x₁.piecewise x (setOf fun j => r j i)) j) (x₁ j)
                                           -/
      .fst ⟨i, fun j hj ↦ ?_, ?_⟩, ?_⟩ <;> simp only [piecewise_apply, Set.mem_setOf_eq]
      /-
        case pos.refine_1
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Membership.mem p i
        hs : s i (x i) (x₁ i)
        j : ι
        hj : r j i
        ⊢ Eq (ite (r j i) (x₁ j) (x j)) (x₁ j)
      -/
    · simp only [if_pos hj]
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Membership.mem p i
        hs : s i (x i) (x₁ i)
        ⊢ s i (ite (r i i) (x₁ i) (x i)) (x₁ i)
      -/
    · split_ifs with hi
        /-
          case pos
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Membership.mem p i
          hs : s i (x i) (x₁ i)
          hi : r i i
          ⊢ s i (x₁ i) (x₁ i)
        -/
      · rwa [hr i hi, if_pos hp] at hs
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Membership.mem p i
          hs : s i (x i) (x₁ i)
          hi : Not (r i i)
          ⊢ s i (x i) (x₁ i)
        -/
      · assumption
        /-
          🎉 no goals
        -/
      /-
        case pos.refine_3
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Membership.mem p i
        hs : s i (x i) (x₁ i)
        ⊢ Eq ((x₁.piecewise x (setOf fun j => r j i)).piecewise x₂ (setOf fun j => r j …
      -/
    · ext1 j
      /-
        case pos.refine_3.h
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Membership.mem p i
        hs : s i (x i) (x₁ i)
        j : ι
        ⊢ Eq (((x₁.piecewise x (setOf fun j => r j i)).piecewise x₂ (setOf fun j => r  …
      -/
      simp only [piecewise_apply, Set.mem_setOf_eq]
      /-
        case pos.refine_3.h
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Membership.mem p i
        hs : s i (x i) (x₁ i)
        j : ι
        ⊢ Eq (ite (r j i → Membership.mem p j) (ite (r j i) (x₁ j) (x j)) (x₂ j)) (x j)
      -/
                               /-
                                 🎉 no goals
                               -/
      split_ifs with h₁ h₂ <;> try rfl
        /-
          case pos
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Membership.mem p i
          hs : s i (x i) (x₁ i)
          j : ι
          h₁ : r j i → Membership.mem p j
          h₂ : r j i
          ⊢ Eq (x₁ j) (x j)
        -/
      · rw [hr j h₂, if_pos (h₁ h₂)]
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Membership.mem p i
          hs : s i (x i) (x₁ i)
          j : ι
          h₁ : Not (r j i → Membership.mem p j)
          ⊢ Eq (x₂ j) (x j)
        -/
      · rw [Classical.not_imp] at h₁
        /-
          case neg
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Membership.mem p i
          hs : s i (x i) (x₁ i)
          j : ι
          h₁ : And (r j i) (Not (Membership.mem p j))
          ⊢ Eq (x₂ j) (x j)
        -/
        rw [hr j h₁.1, if_neg h₁.2]
        /-
          🎉 no goals
        -/
  · refine ⟨⟨{ j | r j i ∧ j ∈ p }, x₁, piecewise x₂ x { j | r j i }⟩,
                                           /-
                                             case neg.refine_1
                                             ι : Type u_1
                                             α : ι → Type u_2
                                             inst✝¹ : (i : ι) → Zero (α i)
                                             r : ι → ι → Prop
                                             s : (i : ι) → α i → α i → Prop
                                             inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
                                             p : Set ι
                                             x₁ x₂ x : DFinsupp fun i => α i
                                             i : ι
                                             hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
                                             hp : Not (Membership.mem p i)
                                             hs : s i (x i) (x₂ i)
                                             j : ι
                                             hj : r j i
                                             ⊢ Eq ((x₂.piecewise x (setOf fun j => r j i)) j) (x₂ j)
                                           -/
      .snd ⟨i, fun j hj ↦ ?_, ?_⟩, ?_⟩ <;> simp only [piecewise_apply, Set.mem_setOf_eq]
      /-
        case neg.refine_1
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Not (Membership.mem p i)
        hs : s i (x i) (x₂ i)
        j : ι
        hj : r j i
        ⊢ Eq (ite (r j i) (x₂ j) (x j)) (x₂ j)
      -/
    · exact if_pos hj
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Not (Membership.mem p i)
        hs : s i (x i) (x₂ i)
        ⊢ s i (ite (r i i) (x₂ i) (x i)) (x₂ i)
      -/
    · split_ifs with hi
        /-
          case pos
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Not (Membership.mem p i)
          hs : s i (x i) (x₂ i)
          hi : r i i
          ⊢ s i (x₂ i) (x₂ i)
        -/
      · rwa [hr i hi, if_neg hp] at hs
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Not (Membership.mem p i)
          hs : s i (x i) (x₂ i)
          hi : Not (r i i)
          ⊢ s i (x i) (x₂ i)
        -/
      · assumption
        /-
          🎉 no goals
        -/
      /-
        case neg.refine_3
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Not (Membership.mem p i)
        hs : s i (x i) (x₂ i)
        ⊢ Eq (x₁.piecewise (x₂.piecewise x (setOf fun j => r j i)) (setOf fun j => And …
      -/
    · ext1 j
      /-
        case neg.refine_3.h
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Not (Membership.mem p i)
        hs : s i (x i) (x₂ i)
        j : ι
        ⊢ Eq ((x₁.piecewise (x₂.piecewise x (setOf fun j => r j i)) (setOf fun j => An …
      -/
      simp only [piecewise_apply, Set.mem_setOf_eq]
      /-
        case neg.refine_3.h
        ι : Type u_1
        α : ι → Type u_2
        inst✝¹ : (i : ι) → Zero (α i)
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
        p : Set ι
        x₁ x₂ x : DFinsupp fun i => α i
        i : ι
        hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
        hp : Not (Membership.mem p i)
        hs : s i (x i) (x₂ i)
        j : ι
        ⊢ Eq (ite (And (r j i) (Membership.mem p j)) (x₁ j) (ite (r j i) (x₂ j) (x j)) …
      -/
      split_ifs with h₁ h₂ <;> try rfl
                               /-
                                 🎉 no goals
                               -/
        /-
          case pos
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Not (Membership.mem p i)
          hs : s i (x i) (x₂ i)
          j : ι
          h₁ : And (r j i) (Membership.mem p j)
          ⊢ Eq (x₁ j) (x j)
        -/
      · rw [hr j h₁.1, if_pos h₁.2]
        /-
          🎉 no goals
        -/
        /-
          case pos
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Not (Membership.mem p i)
          hs : s i (x i) (x₂ i)
          j : ι
          h₁ : Not (And (r j i) (Membership.mem p j))
          h₂ : r j i
          ⊢ Eq (x₂ j) (x j)
        -/
      · rw [hr j h₂, if_neg]
        /-
          case pos.hnc
          ι : Type u_1
          α : ι → Type u_2
          inst✝¹ : (i : ι) → Zero (α i)
          r : ι → ι → Prop
          s : (i : ι) → α i → α i → Prop
          inst✝ : (i : ι) → (s : Set ι) → Decidable (Membership.mem s i)
          p : Set ι
          x₁ x₂ x : DFinsupp fun i => α i
          i : ι
          hr : ∀ (j : ι), r j i → Eq (x j) (ite (Membership.mem p j) (x₁ j) (x₂ j))
          hp : Not (Membership.mem p i)
          hs : s i (x i) (x₂ i)
          j : ι
          h₁ : Not (And (r j i) (Membership.mem p j))
          h₂ : r j i
          ⊢ Not (Membership.mem p j)
        -/
        simpa [h₂] using h₁
        /-
          🎉 no goals
        -/


theorem Lex.acc_of_single_erase [DecidableEq ι] {x : Π₀ i, α i} (i : ι)
    (hs : Acc (DFinsupp.Lex r s) <| single i (x i)) (hu : Acc (DFinsupp.Lex r s) <| x.erase i) :
    Acc (DFinsupp.Lex r s) x := by
  classical
    convert ← @Acc.of_fibration _ _ _ _ _ (lex_fibration r s) ⟨{i}, _⟩
      (InvImage.accessible snd <| hs.prod_gameAdd hu)
    convert piecewise_single_erase x i



theorem Lex.acc_zero (hbot : ∀ ⦃i a⦄, ¬s i a 0) : Acc (DFinsupp.Lex r s) 0 :=
  Acc.intro 0 fun _ ⟨_, _, h⟩ => (hbot h).elim


theorem Lex.acc_of_single (hbot : ∀ ⦃i a⦄, ¬s i a 0) [DecidableEq ι]
    [∀ (i) (x : α i), Decidable (x ≠ 0)] (x : Π₀ i, α i) :
    (∀ i ∈ x.support, Acc (DFinsupp.Lex r s) <| single i (x i)) → Acc (DFinsupp.Lex r s) x := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → (x : α i) → Decidable (Ne x 0)
    x : DFinsupp fun i => α i
    ⊢ (∀ (i : ι), Membership.mem x.support i → Acc (DFinsupp.Lex r s) (DFinsupp.si …
  -/
  generalize ht : x.support = t; revert x
  classical
    induction' t using Finset.induction with b t hb ih
    · intro x ht
      rw [support_eq_empty.1 ht]
      exact fun _ => Lex.acc_zero hbot
    refine fun x ht h => Lex.acc_of_single_erase b (h b <| t.mem_insert_self b) ?_
    refine ih _ (by rw [support_erase, ht, Finset.erase_insert hb]) fun a ha => ?_
    rw [erase_ne (ha.ne_of_not_mem hb)]
    exact h a (Finset.mem_insert_of_mem ha)


theorem Lex.acc_single (hbot : ∀ ⦃i a⦄, ¬s i a 0) (hs : ∀ i, WellFounded (s i))
    [DecidableEq ι] {i : ι} (hi : Acc (rᶜ ⊓ (· ≠ ·)) i) :
    ∀ a, Acc (DFinsupp.Lex r s) (single i a) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i : ι
    hi : Acc (Min.min (HasCompl.compl r) fun x1 x2 => Ne x1 x2) i
    ⊢ ∀ (a : α i), Acc (DFinsupp.Lex r s) (DFinsupp.single i a)
  -/
  induction' hi with i _ ih
  refine fun a => WellFounded.induction (hs i)
    (C := fun x ↦ Acc (DFinsupp.Lex r s) (single i x)) a fun a ha ↦ ?_
  /-
    case intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i✝ i : ι
    h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
    ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
    a✝ a : α i
    ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
    ⊢ (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single i x)) a
  -/
  refine Acc.intro _ fun x ↦ ?_
  /-
    case intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i✝ i : ι
    h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
    ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
    a✝ a : α i
    ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
    x : DFinsupp fun i => α i
    ⊢ DFinsupp.Lex r s x (DFinsupp.single i a) → Acc (DFinsupp.Lex r s) x
  -/
  rintro ⟨k, hr, hs⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs✝ : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i✝ i : ι
    h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
    ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
    a✝ a : α i
    ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
    x : DFinsupp fun i => α i
    k : ι
    hr : ∀ (j : ι), r j k → Eq (x j) ((DFinsupp.single i a) j)
    hs : s k (x k) ((DFinsupp.single i a) k)
    ⊢ Acc (DFinsupp.Lex r s) x
  -/
  rw [single_apply] at hs
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs✝ : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i✝ i : ι
    h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
    ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
    a✝ a : α i
    ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
    x : DFinsupp fun i => α i
    k : ι
    hr : ∀ (j : ι), r j k → Eq (x j) ((DFinsupp.single i a) j)
    hs : s k (x k) (dite (Eq i k) (fun h => Eq.recOn h a) fun h => 0)
    ⊢ Acc (DFinsupp.Lex r s) x
  -/
  split_ifs at hs with hik
  /-
    case pos
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs✝ : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i✝ i : ι
    h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
    ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
    a✝ a : α i
    ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
    x : DFinsupp fun i => α i
    k : ι
    hr : ∀ (j : ι), r j k → Eq (x j) ((DFinsupp.single i a) j)
    hik : Eq i k
    hs : s k (x k) (Eq.recOn hik a)
    ⊢ Acc (DFinsupp.Lex r s) x
  -/
  swap
    /-
      case neg
      ι : Type u_1
      α : ι → Type u_2
      inst✝¹ : (i : ι) → Zero (α i)
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
      hs✝ : ∀ (i : ι), WellFounded (s i)
      inst✝ : DecidableEq ι
      i✝ i : ι
      h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
      ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
      a✝ a : α i
      ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
      x : DFinsupp fun i => α i
      k : ι
      hr : ∀ (j : ι), r j k → Eq (x j) ((DFinsupp.single i a) j)
      hik : Not (Eq i k)
      hs : s k (x k) 0
      ⊢ Acc (DFinsupp.Lex r s) x
    -/
  · exact (hbot hs).elim
    /-
      🎉 no goals
    -/
  /-
    case pos
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → Zero (α i)
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
    hs✝ : ∀ (i : ι), WellFounded (s i)
    inst✝ : DecidableEq ι
    i✝ i : ι
    h✝ : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → Acc ( …
    ih : ∀ (y : ι), Min.min (HasCompl.compl r) (fun x1 x2 => Ne x1 x2) y i → ∀ (a  …
    a✝ a : α i
    ha : ∀ (y : α i), s i y a → (fun x => Acc (DFinsupp.Lex r s) (DFinsupp.single  …
    x : DFinsupp fun i => α i
    k : ι
    hr : ∀ (j : ι), r j k → Eq (x j) ((DFinsupp.single i a) j)
    hik : Eq i k
    hs : s k (x k) (Eq.recOn hik a)
    ⊢ Acc (DFinsupp.Lex r s) x
  -/
  subst hik
  classical
    refine Lex.acc_of_single hbot x fun j hj ↦ ?_
    obtain rfl | hij := eq_or_ne i j
    · exact ha _ hs
    by_cases h : r j i
    · rw [hr j h, single_eq_of_ne hij, single_zero]
      exact Lex.acc_zero hbot
    · exact ih _ ⟨h, hij.symm⟩ _


theorem Lex.acc (hbot : ∀ ⦃i a⦄, ¬s i a 0) (hs : ∀ i, WellFounded (s i))
    [DecidableEq ι] [∀ (i) (x : α i), Decidable (x ≠ 0)] (x : Π₀ i, α i)
    (h : ∀ i ∈ x.support, Acc (rᶜ ⊓ (· ≠ ·)) i) : Acc (DFinsupp.Lex r s) x :=
  Lex.acc_of_single hbot x fun i hi => Lex.acc_single hbot hs (h i hi) _


theorem Lex.wellFounded (hbot : ∀ ⦃i a⦄, ¬s i a 0) (hs : ∀ i, WellFounded (s i))
    (hr : WellFounded <| rᶜ ⊓ (· ≠ ·)) : WellFounded (DFinsupp.Lex r s) :=
               /-
                 ι : Type u_1
                 α : ι → Type u_2
                 inst✝ : (i : ι) → Zero (α i)
                 r : ι → ι → Prop
                 s : (i : ι) → α i → α i → Prop
                 hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (s i a 0)
                 hs : ∀ (i : ι), WellFounded (s i)
                 hr : WellFounded (Min.min (HasCompl.compl r) fun x1 x2 => Ne x1 x2)
                 x : DFinsupp fun i => α i
                 ⊢ Acc (DFinsupp.Lex r s) x
               -/
  ⟨fun x => by classical exact Lex.acc hbot hs x fun i _ => hr.apply i⟩
               /-
                 🎉 no goals
               -/


theorem Lex.wellFounded' (hbot : ∀ ⦃i a⦄, ¬s i a 0) (hs : ∀ i, WellFounded (s i))
    [IsTrichotomous ι r] (hr : WellFounded (Function.swap r)) :
    WellFounded (DFinsupp.Lex r s) :=
  Lex.wellFounded hbot hs <| Subrelation.wf
   (fun {i j} h => ((@IsTrichotomous.trichotomous ι r _ i j).resolve_left h.1).resolve_left h.2) hr


instance Lex.wellFoundedLT [LT ι] [IsTrichotomous ι (· < ·)] [hι : WellFoundedGT ι]
    [∀ i, CanonicallyOrderedAddCommMonoid (α i)] [hα : ∀ i, WellFoundedLT (α i)] :
    WellFoundedLT (Lex (Π₀ i, α i)) :=
  ⟨Lex.wellFounded' (fun _ a => (zero_le a).not_lt) (fun i => (hα i).wf) hι.wf⟩


theorem Pi.Lex.wellFounded [IsStrictTotalOrder ι r] [Finite ι] (hs : ∀ i, WellFounded (s i)) :
    WellFounded (Pi.Lex r (fun {i} ↦ s i)) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝¹ : IsStrictTotalOrder ι r
    inst✝ : Finite ι
    hs : ∀ (i : ι), WellFounded (s i)
    ⊢ WellFounded (Pi.Lex r fun {i} => s i)
  -/
  obtain h | ⟨⟨x⟩⟩ := isEmpty_or_nonempty (∀ i, α i)
    /-
      case inl
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      inst✝¹ : IsStrictTotalOrder ι r
      inst✝ : Finite ι
      hs : ∀ (i : ι), WellFounded (s i)
      h : IsEmpty ((i : ι) → α i)
      ⊢ WellFounded (Pi.Lex r fun {i} => s i)
    -/
  · convert emptyWf.wf
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝¹ : IsStrictTotalOrder ι r
    inst✝ : Finite ι
    hs : ∀ (i : ι), WellFounded (s i)
    x : (i : ι) → α i
    ⊢ WellFounded (Pi.Lex r fun {i} => s i)
  -/
  letI : ∀ i, Zero (α i) := fun i => ⟨(hs i).min ⊤ ⟨x i, trivial⟩⟩
  /-
    case inr.intro
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝¹ : IsStrictTotalOrder ι r
    inst✝ : Finite ι
    hs : ∀ (i : ι), WellFounded (s i)
    x : (i : ι) → α i
    this : (i : ι) → Zero (α i) := fun i => { zero := ⋯.min Top.top ⋯ }
    ⊢ WellFounded (Pi.Lex r fun {i} => s i)
  -/
  haveI := IsTrans.swap r; haveI := IsIrrefl.swap r; haveI := Fintype.ofFinite ι
  /-
    case inr.intro
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝¹ : IsStrictTotalOrder ι r
    inst✝ : Finite ι
    hs : ∀ (i : ι), WellFounded (s i)
    x : (i : ι) → α i
    this✝² : (i : ι) → Zero (α i) := fun i => { zero := ⋯.min Top.top ⋯ }
    this✝¹ : IsTrans ι (Function.swap r)
    this✝ : IsIrrefl ι (Function.swap r)
    this : Fintype ι
    ⊢ WellFounded (Pi.Lex r fun {i} => s i)
  -/
  refine InvImage.wf equivFunOnFintype.symm (Lex.wellFounded' (fun i a => ?_) hs ?_)
  /-
    case inr.intro.refine_1
    ι : Type u_1
    α : ι → Type u_2
    r : ι → ι → Prop
    s : (i : ι) → α i → α i → Prop
    inst✝¹ : IsStrictTotalOrder ι r
    inst✝ : Finite ι
    hs : ∀ (i : ι), WellFounded (s i)
    x : (i : ι) → α i
    this✝² : (i : ι) → Zero (α i) := fun i => { zero := ⋯.min Top.top ⋯ }
    this✝¹ : IsTrans ι (Function.swap r)
    this✝ : IsIrrefl ι (Function.swap r)
    this : Fintype ι
    i : ι
    a : α i
    ⊢ Not (s i a 0)
  -/
  exacts [(hs i).not_lt_min ⊤ _ trivial, Finite.wellFounded_of_trans_of_irrefl (Function.swap r)]
  /-
    🎉 no goals
  -/


instance Pi.Lex.wellFoundedLT [LinearOrder ι] [Finite ι] [∀ i, LT (α i)]
    [hwf : ∀ i, WellFoundedLT (α i)] : WellFoundedLT (Lex (∀ i, α i)) :=
  ⟨Pi.Lex.wellFounded (· < ·) fun i => (hwf i).1⟩


instance Function.Lex.wellFoundedLT {α} [LinearOrder ι] [Finite ι] [LT α] [WellFoundedLT α] :
    WellFoundedLT (Lex (ι → α)) :=
  Pi.Lex.wellFoundedLT


theorem DFinsupp.Lex.wellFounded_of_finite [IsStrictTotalOrder ι r] [Finite ι] [∀ i, Zero (α i)]
    (hs : ∀ i, WellFounded (s i)) : WellFounded (DFinsupp.Lex r s) :=
  have := Fintype.ofFinite ι
  InvImage.wf equivFunOnFintype (Pi.Lex.wellFounded r hs)


instance DFinsupp.Lex.wellFoundedLT_of_finite [LinearOrder ι] [Finite ι] [∀ i, Zero (α i)]
    [∀ i, LT (α i)] [hwf : ∀ i, WellFoundedLT (α i)] : WellFoundedLT (Lex (Π₀ i, α i)) :=
  ⟨DFinsupp.Lex.wellFounded_of_finite (· < ·) fun i => (hwf i).1⟩


protected theorem DFinsupp.wellFoundedLT [∀ i, Zero (α i)] [∀ i, Preorder (α i)]
    [∀ i, WellFoundedLT (α i)] (hbot : ∀ ⦃i⦄ ⦃a : α i⦄, ¬a < 0) : WellFoundedLT (Π₀ i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    set β := fun i ↦ Antisymmetrization (α i) (· ≤ ·)
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      β : ι → Type u_2 := fun i => Antisymmetrization (α i) fun x1 x2 => LE.le x1 x2
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    set e : (i : ι) → α i → β i := fun i ↦ toAntisymmetrization (· ≤ ·)
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      β : ι → Type u_2 := fun i => Antisymmetrization (α i) fun x1 x2 => LE.le x1 x2
      e : (i : ι) → α i → β i := fun i => toAntisymmetrization fun x1 x2 => LE.le x1 …
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    let _ : ∀ i, Zero (β i) := fun i ↦ ⟨e i 0⟩
    have : WellFounded (DFinsupp.Lex (Function.swap <| @WellOrderingRel ι)
        (fun _ ↦ (· < ·) : (i : ι) → β i → β i → Prop)) := by
      have := IsTrichotomous.swap (@WellOrderingRel ι)
      refine Lex.wellFounded' ?_ (fun i ↦ IsWellFounded.wf) ?_
      · rintro i ⟨a⟩
        apply hbot
      · #adaptation_note /-- nightly-2024-03-16: simp was
        simp (config := { unfoldPartialApp := true }) only [Function.swap] -/
        simp only [Function.swap_def]
        exact IsWellFounded.wf
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      β : ι → Type u_2 := fun i => Antisymmetrization (α i) fun x1 x2 => LE.le x1 x2
      e : (i : ι) → α i → β i := fun i => toAntisymmetrization fun x1 x2 => LE.le x1 …
      x✝ : (i : ι) → Zero (β i) := fun i => { zero := e i 0 }
      this : WellFounded (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2 = …
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    refine Subrelation.wf (fun h => ?_) <| InvImage.wf (mapRange e fun _ ↦ rfl) this
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      β : ι → Type u_2 := fun i => Antisymmetrization (α i) fun x1 x2 => LE.le x1 x2
      e : (i : ι) → α i → β i := fun i => toAntisymmetrization fun x1 x2 => LE.le x1 …
      x✝¹ : (i : ι) → Zero (β i) := fun i => { zero := e i 0 }
      this : WellFounded (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2 = …
      x✝ y✝ : DFinsupp fun i => α i
      h : LT.lt x✝ y✝
      ⊢ InvImage (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2 => LT.lt  …
    -/
    have := IsStrictOrder.swap (@WellOrderingRel ι)
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      β : ι → Type u_2 := fun i => Antisymmetrization (α i) fun x1 x2 => LE.le x1 x2
      e : (i : ι) → α i → β i := fun i => toAntisymmetrization fun x1 x2 => LE.le x1 …
      x✝¹ : (i : ι) → Zero (β i) := fun i => { zero := e i 0 }
      this✝ : WellFounded (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2  …
      x✝ y✝ : DFinsupp fun i => α i
      h : LT.lt x✝ y✝
      this : IsStrictOrder ι (Function.swap WellOrderingRel)
      ⊢ InvImage (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2 => LT.lt  …
    -/
    obtain ⟨i, he, hl⟩ := lex_lt_of_lt_of_preorder (Function.swap WellOrderingRel) h
    /-
      case intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : (i : ι) → Preorder (α i)
      inst✝ : ∀ (i : ι), WellFoundedLT (α i)
      hbot : ∀ ⦃i : ι⦄ ⦃a : α i⦄, Not (LT.lt a 0)
      β : ι → Type u_2 := fun i => Antisymmetrization (α i) fun x1 x2 => LE.le x1 x2
      e : (i : ι) → α i → β i := fun i => toAntisymmetrization fun x1 x2 => LE.le x1 …
      x✝¹ : (i : ι) → Zero (β i) := fun i => { zero := e i 0 }
      this✝ : WellFounded (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2  …
      x✝ y✝ : DFinsupp fun i => α i
      h : LT.lt x✝ y✝
      this : IsStrictOrder ι (Function.swap WellOrderingRel)
      i : ι
      he : ∀ (j : ι), Function.swap WellOrderingRel j i → And (LE.le (x✝ j) (y✝ j))  …
      hl : LT.lt (x✝ i) (y✝ i)
      ⊢ InvImage (DFinsupp.Lex (Function.swap WellOrderingRel) fun x x1 x2 => LT.lt  …
    -/
    exact ⟨i, fun j hj ↦ Quot.sound (he j hj), hl⟩⟩
    /-
      🎉 no goals
    -/


instance DFinsupp.wellFoundedLT' [∀ i, CanonicallyOrderedAddCommMonoid (α i)]
    [∀ i, WellFoundedLT (α i)] : WellFoundedLT (Π₀ i, α i) :=
  DFinsupp.wellFoundedLT fun _i a => (zero_le a).not_lt


instance Pi.wellFoundedLT [Finite ι] [∀ i, Preorder (α i)] [hw : ∀ i, WellFoundedLT (α i)] :
    WellFoundedLT (∀ i, α i) :=
  ⟨by
    /-
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Preorder (α i)
      hw : ∀ (i : ι), WellFoundedLT (α i)
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    obtain h | ⟨⟨x⟩⟩ := isEmpty_or_nonempty (∀ i, α i)
      /-
        case inl
        ι : Type u_1
        α : ι → Type u_2
        r : ι → ι → Prop
        s : (i : ι) → α i → α i → Prop
        inst✝¹ : Finite ι
        inst✝ : (i : ι) → Preorder (α i)
        hw : ∀ (i : ι), WellFoundedLT (α i)
        h : IsEmpty ((i : ι) → α i)
        ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
      -/
    · convert emptyWf.wf
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Preorder (α i)
      hw : ∀ (i : ι), WellFoundedLT (α i)
      x : (i : ι) → α i
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    letI : ∀ i, Zero (α i) := fun i => ⟨(hw i).wf.min ⊤ ⟨x i, trivial⟩⟩
    /-
      case inr.intro
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Preorder (α i)
      hw : ∀ (i : ι), WellFoundedLT (α i)
      x : (i : ι) → α i
      this : (i : ι) → Zero (α i) := fun i => { zero := ⋯.min Top.top ⋯ }
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    haveI := Fintype.ofFinite ι
    /-
      case inr.intro
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Preorder (α i)
      hw : ∀ (i : ι), WellFoundedLT (α i)
      x : (i : ι) → α i
      this✝ : (i : ι) → Zero (α i) := fun i => { zero := ⋯.min Top.top ⋯ }
      this : Fintype ι
      ⊢ WellFounded fun x1 x2 => LT.lt x1 x2
    -/
    refine InvImage.wf equivFunOnFintype.symm (DFinsupp.wellFoundedLT fun i a => ?_).wf
    /-
      case inr.intro
      ι : Type u_1
      α : ι → Type u_2
      r : ι → ι → Prop
      s : (i : ι) → α i → α i → Prop
      inst✝¹ : Finite ι
      inst✝ : (i : ι) → Preorder (α i)
      hw : ∀ (i : ι), WellFoundedLT (α i)
      x : (i : ι) → α i
      this✝ : (i : ι) → Zero (α i) := fun i => { zero := ⋯.min Top.top ⋯ }
      this : Fintype ι
      i : ι
      a : α i
      ⊢ Not (LT.lt a 0)
    -/
    exact (hw i).wf.not_lt_min ⊤ _ trivial⟩
    /-
      🎉 no goals
    -/


instance Function.wellFoundedLT {α} [Finite ι] [Preorder α] [WellFoundedLT α] :
    WellFoundedLT (ι → α) :=
  Pi.wellFoundedLT


instance DFinsupp.wellFoundedLT_of_finite [Finite ι] [∀ i, Zero (α i)] [∀ i, Preorder (α i)]
    [∀ i, WellFoundedLT (α i)] : WellFoundedLT (Π₀ i, α i) :=
  have := Fintype.ofFinite ι
  ⟨InvImage.wf equivFunOnFintype Pi.wellFoundedLT.wf⟩

