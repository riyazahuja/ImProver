/-- Let `f` and `g` be two maps to the same commutative monoid. This lemma gives a sufficient
condition for comparison of the filter `atTop.map (fun s ↦ ∏ b ∈ s, f b)` with
`atTop.map (fun s ↦ ∏ b ∈ s, g b)`. This is useful to compare the set of limit points of
`Π b in s, f b` as `s → atTop` with the similar set for `g`. -/
@[to_additive "Let `f` and `g` be two maps to the same commutative additive monoid. This lemma gives
a sufficient condition for comparison of the filter `atTop.map (fun s ↦ ∑ b ∈ s, f b)` with
`atTop.map (fun s ↦ ∑ b ∈ s, g b)`. This is useful to compare the set of limit points of
`∑ b ∈ s, f b` as `s → atTop` with the similar set for `g`."]
theorem Filter.map_atTop_finset_prod_le_of_prod_eq {f : α → M} {g : β → M}
    (h_eq : ∀ u : Finset β,
      ∃ v : Finset α, ∀ v', v ⊆ v' → ∃ u', u ⊆ u' ∧ ∏ x ∈ u', g x = ∏ b ∈ v', f b) :
    (atTop.map fun s : Finset α => ∏ b ∈ s, f b) ≤
      atTop.map fun s : Finset β => ∏ x ∈ s, g x := by
  classical
    refine ((atTop_basis.map _).le_basis_iff (atTop_basis.map _)).2 fun b _ => ?_
    let ⟨v, hv⟩ := h_eq b
    refine ⟨v, trivial, ?_⟩
    simpa [Finset.image_subset_iff] using hv


/-- Let `g : γ → β` be an injective function and `f : β → α` be a function from the codomain of `g`
to a commutative monoid. Suppose that `f x = 1` outside of the range of `g`. Then the filters
`atTop.map (fun s ↦ ∏ i ∈ s, f (g i))` and `atTop.map (fun s ↦ ∏ i ∈ s, f i)` coincide.

The additive version of this lemma is used to prove the equality `∑' x, f (g x) = ∑' y, f y` under
the same assumptions. -/
@[to_additive]
theorem Function.Injective.map_atTop_finset_prod_eq {g : α → β}
    (hg : Function.Injective g) {f : β → M} (hf : ∀ x, x ∉ Set.range g → f x = 1) :
    map (fun s => ∏ i ∈ s, f (g i)) atTop = map (fun s => ∏ i ∈ s, f i) atTop := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝ : CommMonoid M
    g : α → β
    hg : Function.Injective g
    f : β → M
    hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
    ⊢ Eq (Filter.map (fun s => s.prod fun i => f (g i)) Filter.atTop) (Filter.map  …
  -/
  haveI := Classical.decEq β
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝ : CommMonoid M
    g : α → β
    hg : Function.Injective g
    f : β → M
    hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
    this : DecidableEq β
    ⊢ Eq (Filter.map (fun s => s.prod fun i => f (g i)) Filter.atTop) (Filter.map  …
  -/
  apply le_antisymm <;> refine map_atTop_finset_prod_le_of_prod_eq fun s => ?_
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      ⊢ Exists fun v => ∀ (v' : Finset α), HasSubset.Subset v v' → Exists fun u' =>  …
    -/
  · refine ⟨s.preimage g hg.injOn, fun t ht => ?_⟩
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      ⊢ Exists fun u' => And (HasSubset.Subset s u') (Eq (u'.prod fun x => f x) (t.p …
    -/
    refine ⟨t.image g ∪ s, Finset.subset_union_right, ?_⟩
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      ⊢ Eq ((Union.union (Finset.image g t) s).prod fun x => f x) (t.prod fun b => f …
    -/
    rw [← Finset.prod_image hg.injOn]
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      ⊢ Eq ((Union.union (Finset.image g t) s).prod fun x => f x) ((Finset.image g t …
    -/
    refine (prod_subset subset_union_left ?_).symm
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      ⊢ ∀ (x : β), Membership.mem (Union.union (Finset.image g t) s) x → Not (Member …
    -/
    simp only [Finset.mem_union, Finset.mem_image]
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      ⊢ ∀ (x : β), Or (Exists fun a => And (Membership.mem t a) (Eq (g a) x)) (Membe …
    -/
    refine fun y hy hyt => hf y (mt ?_ hyt)
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      y : β
      hy : Or (Exists fun a => And (Membership.mem t a) (Eq (g a) y)) (Membership.me …
      hyt : Not (Exists fun a => And (Membership.mem t a) (Eq (g a) y))
      ⊢ Membership.mem (Set.range g) y → Exists fun a => And (Membership.mem t a) (E …
    -/
    rintro ⟨x, rfl⟩
    /-
      case a.intro
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset β
      t : Finset α
      ht : HasSubset.Subset (s.preimage g ⋯) t
      x : α
      hy : Or (Exists fun a => And (Membership.mem t a) (Eq (g a) (g x))) (Membershi …
      hyt : Not (Exists fun a => And (Membership.mem t a) (Eq (g a) (g x)))
      ⊢ Exists fun a => And (Membership.mem t a) (Eq (g a) (g x))
    -/
    exact ⟨x, ht (Finset.mem_preimage.2 <| hy.resolve_left hyt), rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset α
      ⊢ Exists fun v => ∀ (v' : Finset β), HasSubset.Subset v v' → Exists fun u' =>  …
    -/
  · refine ⟨s.image g, fun t ht => ?_⟩
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset α
      t : Finset β
      ht : HasSubset.Subset (Finset.image g s) t
      ⊢ Exists fun u' => And (HasSubset.Subset s u') (Eq (u'.prod fun x => f (g x))  …
    -/
    simp only [← prod_preimage _ _ hg.injOn _ fun x _ => hf x]
    /-
      case a
      α : Type u_1
      β : Type u_2
      M : Type u_3
      inst✝ : CommMonoid M
      g : α → β
      hg : Function.Injective g
      f : β → M
      hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
      this : DecidableEq β
      s : Finset α
      t : Finset β
      ht : HasSubset.Subset (Finset.image g s) t
      ⊢ Exists fun u' => And (HasSubset.Subset s u') (Eq (u'.prod fun x => f (g x))  …
    -/
    exact ⟨_, (image_subset_iff_subset_preimage _).1 ht, rfl⟩
    /-
      🎉 no goals
    -/


