/-- `HasProd f a` means that the (potentially infinite) product of the `f b` for `b : β` converges
to `a`.

The `atTop` filter on `Finset β` is the limit of all finite sets towards the entire type. So we take
the product over bigger and bigger sets. This product operation is invariant under reordering.

For the definition and many statements, `α` does not need to be a topological monoid. We only add
this assumption later, for the lemmas where it is relevant.

These are defined in an identical way to infinite sums (`HasSum`). For example, we say that
the function `ℕ → ℝ` sending `n` to `1 / 2` has a product of `0`, rather than saying that it does
not converge as some authors would. -/
@[to_additive "`HasSum f a` means that the (potentially infinite) sum of the `f b` for `b : β`
converges to `a`.

The `atTop` filter on `Finset β` is the limit of all finite sets towards the entire type. So we sum
up bigger and bigger sets. This sum operation is invariant under reordering. In particular,
the function `ℕ → ℝ` sending `n` to `(-1)^n / (n+1)` does not have a
sum for this definition, but a series which is absolutely convergent will have the correct sum.

This is based on Mario Carneiro's
[infinite sum `df-tsms` in Metamath](http://us.metamath.org/mpeuni/df-tsms.html).

For the definition and many statements, `α` does not need to be a topological monoid. We only add
this assumption later, for the lemmas where it is relevant."]
def HasProd (f : β → α) (a : α) : Prop :=
  Tendsto (fun s : Finset β ↦ ∏ b ∈ s, f b) atTop (𝓝 a)


/-- `Multipliable f` means that `f` has some (infinite) product. Use `tprod` to get the value. -/
@[to_additive "`Summable f` means that `f` has some (infinite) sum. Use `tsum` to get the value."]
def Multipliable (f : β → α) : Prop :=
  ∃ a, HasProd f a


open scoped Classical in
/-- `∏' i, f i` is the product of `f` if it exists and is unconditionally convergent,
or 1 otherwise. -/
@[to_additive "`∑' i, f i` is the sum of `f` if it exists and is unconditionally convergent,
or 0 otherwise."]
noncomputable irreducible_def tprod {β} (f : β → α) :=
  if h : Multipliable f then
  /- Note that the product might not be uniquely defined if the topology is not separated.
  When the multiplicative support of `f` is finite, we make the most reasonable choice to use the
  product over the multiplicative support. Otherwise, we choose arbitrarily an `a` satisfying
  `HasProd f a`. -/
    if (mulSupport f).Finite then finprod f
    else h.choose
  else 1

-- see Note [operator precedence of big operators]

@[inherit_doc tprod]
notation3 "∏' "(...)", "r:67:(scoped f => tprod f) => r

@[inherit_doc tsum]
notation3 "∑' "(...)", "r:67:(scoped f => tsum f) => r


@[to_additive]
theorem HasProd.multipliable (h : HasProd f a) : Multipliable f :=
  ⟨a, h⟩


@[to_additive]
theorem tprod_eq_one_of_not_multipliable (h : ¬Multipliable f) : ∏' b, f b = 1 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    f : β → α
    h : Not (Multipliable f)
    ⊢ Eq (tprod fun b => f b) 1
  -/
  simp [tprod_def, h]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Function.Injective.hasProd_iff {g : γ → β} (hg : Injective g)
    (hf : ∀ x, x ∉ Set.range g → f x = 1) : HasProd (f ∘ g) a ↔ HasProd f a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    f : β → α
    a : α
    g : γ → β
    hg : Function.Injective g
    hf : ∀ (x : β), Not (Membership.mem (Set.range g) x) → Eq (f x) 1
    ⊢ Iff (HasProd (Function.comp f g) a) (HasProd f a)
  -/
  simp only [HasProd, Tendsto, comp_apply, hg.map_atTop_finset_prod_eq hf]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem hasProd_subtype_iff_of_mulSupport_subset {s : Set β} (hf : mulSupport f ⊆ s) :
    HasProd (f ∘ (↑) : s → α) a ↔ HasProd f a :=
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            inst✝¹ : CommMonoid α
                                            inst✝ : TopologicalSpace α
                                            f : β → α
                                            a : α
                                            s : Set β
                                            hf : HasSubset.Subset (Function.mulSupport f) s
                                            ⊢ ∀ (x : β), Not (Membership.mem (Set.range fun a => ↑a) x) → Eq (f x) 1
                                          -/
  Subtype.coe_injective.hasProd_iff <| by simpa using mulSupport_subset_iff'.1 hf
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem hasProd_fintype [Fintype β] (f : β → α) : HasProd f (∏ b, f b) :=
  OrderTop.tendsto_atTop_nhds _


@[to_additive]
protected theorem Finset.hasProd (s : Finset β) (f : β → α) :
    HasProd (f ∘ (↑) : (↑s : Set β) → α) (∏ b ∈ s, f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    s : Finset β
    f : β → α
    ⊢ HasProd (Function.comp f Subtype.val) (s.prod fun b => f b)
  -/
  rw [← prod_attach]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    s : Finset β
    f : β → α
    ⊢ HasProd (Function.comp f Subtype.val) (s.attach.prod fun x => f ↑x)
  -/
  exact hasProd_fintype _
  /-
    🎉 no goals
  -/


/-- If a function `f` is `1` outside of a finite set `s`, then it `HasProd` `∏ b ∈ s, f b`. -/
@[to_additive "If a function `f` vanishes outside of a finite set `s`, then it `HasSum`
`∑ b ∈ s, f b`."]
theorem hasProd_prod_of_ne_finset_one (hf : ∀ b ∉ s, f b = 1) :
    HasProd f (∏ b ∈ s, f b) :=
  (hasProd_subtype_iff_of_mulSupport_subset <| mulSupport_subset_iff'.2 hf).1 <| s.hasProd f


@[to_additive]
theorem multipliable_of_ne_finset_one (hf : ∀ b ∉ s, f b = 1) : Multipliable f :=
  (hasProd_prod_of_ne_finset_one hf).multipliable


@[to_additive]
theorem Multipliable.hasProd (ha : Multipliable f) : HasProd f (∏' b, f b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    f : β → α
    ha : Multipliable f
    ⊢ HasProd f (tprod fun b => f b)
  -/
  simp only [tprod_def, ha, dite_true]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : TopologicalSpace α
    f : β → α
    ha : Multipliable f
    ⊢ HasProd f (ite (Function.mulSupport fun b => f b).Finite (finprod fun b => f …
  -/
  by_cases H : (mulSupport f).Finite
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : CommMonoid α
      inst✝ : TopologicalSpace α
      f : β → α
      ha : Multipliable f
      H : (Function.mulSupport f).Finite
      ⊢ HasProd f (ite (Function.mulSupport fun b => f b).Finite (finprod fun b => f …
    -/
  · simp [H, hasProd_prod_of_ne_finset_one, finprod_eq_prod]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : CommMonoid α
      inst✝ : TopologicalSpace α
      f : β → α
      ha : Multipliable f
      H : Not (Function.mulSupport f).Finite
      ⊢ HasProd f (ite (Function.mulSupport fun b => f b).Finite (finprod fun b => f …
    -/
  · simpa [H] using ha.choose_spec
    /-
      🎉 no goals
    -/


@[to_additive]
theorem HasProd.unique {a₁ a₂ : α} [T2Space α] : HasProd f a₁ → HasProd f a₂ → a₁ = a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : CommMonoid α
    inst✝¹ : TopologicalSpace α
    f : β → α
    a₁ a₂ : α
    inst✝ : T2Space α
    ⊢ HasProd f a₁ → HasProd f a₂ → Eq a₁ a₂
  -/
  classical exact tendsto_nhds_unique
  /-
    🎉 no goals
  -/


@[to_additive]
theorem HasProd.tprod_eq (ha : HasProd f a) : ∏' b, f b = a :=
  (Multipliable.hasProd ⟨a, ha⟩).unique ha


@[to_additive]
theorem Multipliable.hasProd_iff (h : Multipliable f) : HasProd f a ↔ ∏' b, f b = a :=
  Iff.intro HasProd.tprod_eq fun eq ↦ eq ▸ h.hasProd


