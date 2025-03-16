@[simp]
theorem add_closure_iUnion_range_single :
    AddSubmonoid.closure (⋃ i : ι, Set.range (single i : β i → Π₀ i, β i)) = ⊤ :=
  top_unique fun x _ => by
    /-
      ι : Type u
      β : ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → AddZeroClass (β i)
      x : DFinsupp fun i => β i
      x✝ : Membership.mem Top.top x
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => Set.range (DFinsup …
    -/
    apply DFinsupp.induction x
      /-
        case h0
        ι : Type u
        β : ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → AddZeroClass (β i)
        x : DFinsupp fun i => β i
        x✝ : Membership.mem Top.top x
        ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => Set.range (DFinsup …
      -/
    · exact AddSubmonoid.zero_mem _
      /-
        🎉 no goals
      -/
    exact fun a b f _ _ hf =>
      AddSubmonoid.add_mem _
        (AddSubmonoid.subset_closure <| Set.mem_iUnion.2 ⟨a, Set.mem_range_self _⟩) hf


/-- If two additive homomorphisms from `Π₀ i, β i` are equal on each `single a b`, then
they are equal. -/
theorem addHom_ext {γ : Type w} [AddZeroClass γ] ⦃f g : (Π₀ i, β i) →+ γ⦄
    (H : ∀ (i : ι) (y : β i), f (single i y) = g (single i y)) : f = g := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    γ : Type w
    inst✝ : AddZeroClass γ
    f g : AddMonoidHom (DFinsupp fun i => β i) γ
    H : ∀ (i : ι) (y : β i), Eq (f (DFinsupp.single i y)) (g (DFinsupp.single i y))
    ⊢ Eq f g
  -/
  refine AddMonoidHom.eq_of_eqOn_denseM add_closure_iUnion_range_single fun f hf => ?_
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    γ : Type w
    inst✝ : AddZeroClass γ
    f✝ g : AddMonoidHom (DFinsupp fun i => β i) γ
    H : ∀ (i : ι) (y : β i), Eq (f✝ (DFinsupp.single i y)) (g (DFinsupp.single i y))
    f : DFinsupp fun i => β i
    hf : Membership.mem (Set.iUnion fun i => Set.range (DFinsupp.single i)) f
    ⊢ Eq (f✝ f) (g f)
  -/
  simp only [Set.mem_iUnion, Set.mem_range] at hf
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    γ : Type w
    inst✝ : AddZeroClass γ
    f✝ g : AddMonoidHom (DFinsupp fun i => β i) γ
    H : ∀ (i : ι) (y : β i), Eq (f✝ (DFinsupp.single i y)) (g (DFinsupp.single i y))
    f : DFinsupp fun i => β i
    hf : Exists fun i => Exists fun y => Eq (DFinsupp.single i y) f
    ⊢ Eq (f✝ f) (g f)
  -/
  rcases hf with ⟨x, y, rfl⟩
  /-
    case intro.intro
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    γ : Type w
    inst✝ : AddZeroClass γ
    f g : AddMonoidHom (DFinsupp fun i => β i) γ
    H : ∀ (i : ι) (y : β i), Eq (f (DFinsupp.single i y)) (g (DFinsupp.single i y))
    x : ι
    y : β x
    ⊢ Eq (f (DFinsupp.single x y)) (g (DFinsupp.single x y))
  -/
  apply H
  /-
    🎉 no goals
  -/


/-- If two additive homomorphisms from `Π₀ i, β i` are equal on each `single a b`, then
they are equal.

See note [partially-applied ext lemmas]. -/
@[ext]
theorem addHom_ext' {γ : Type w} [AddZeroClass γ] ⦃f g : (Π₀ i, β i) →+ γ⦄
    (H : ∀ x, f.comp (singleAddHom β x) = g.comp (singleAddHom β x)) : f = g :=
  addHom_ext fun x => DFunLike.congr_fun (H x)


