@[to_additive]
lemma hasProd_le_of_prod_le [ClosedIicTopology α]
    (hf : HasProd f a) (h : ∀ s, ∏ i ∈ s, f i ≤ c) : a ≤ c :=
  le_of_tendsto' hf h


@[to_additive]
theorem le_hasProd_of_le_prod [ClosedIciTopology α]
    (hf : HasProd f a) (h : ∀ s, c ≤ ∏ i ∈ s, f i) : c ≤ a :=
  ge_of_tendsto' hf h


@[to_additive]
theorem tprod_le_of_prod_range_le [ClosedIicTopology α] {f : ℕ → α} (hf : Multipliable f)
    (h : ∀ n, ∏ i ∈ range n, f i ≤ c) : ∏' n, f n ≤ c :=
  le_of_tendsto' hf.hasProd.tendsto_prod_nat h


@[to_additive]
theorem hasProd_le (h : ∀ i, f i ≤ g i) (hf : HasProd f a₁) (hg : HasProd g a₂) : a₁ ≤ a₂ :=
  le_of_tendsto_of_tendsto' hf hg fun _ ↦ prod_le_prod' fun i _ ↦ h i


@[to_additive]
theorem hasProd_mono (hf : HasProd f a₁) (hg : HasProd g a₂) (h : f ≤ g) : a₁ ≤ a₂ :=
  hasProd_le h hf hg


@[to_additive]
theorem hasProd_le_inj {g : κ → α} (e : ι → κ) (he : Injective e)
    (hs : ∀ c, c ∉ Set.range e → 1 ≤ g c) (h : ∀ i, f i ≤ g (e i)) (hf : HasProd f a₁)
    (hg : HasProd g a₂) : a₁ ≤ a₂ := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    a₁ a₂ : α
    g : κ → α
    e : ι → κ
    he : Function.Injective e
    hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
    h : ∀ (i : ι), LE.le (f i) (g (e i))
    hf : HasProd f a₁
    hg : HasProd g a₂
    ⊢ LE.le a₁ a₂
  -/
  rw [← hasProd_extend_one he] at hf
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    a₁ a₂ : α
    g : κ → α
    e : ι → κ
    he : Function.Injective e
    hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
    h : ∀ (i : ι), LE.le (f i) (g (e i))
    hf : HasProd (Function.extend e f 1) a₁
    hg : HasProd g a₂
    ⊢ LE.le a₁ a₂
  -/
  refine hasProd_le (fun c ↦ ?_) hf hg
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    a₁ a₂ : α
    g : κ → α
    e : ι → κ
    he : Function.Injective e
    hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
    h : ∀ (i : ι), LE.le (f i) (g (e i))
    hf : HasProd (Function.extend e f 1) a₁
    hg : HasProd g a₂
    c : κ
    ⊢ LE.le (Function.extend e f 1 c) (g c)
  -/
  obtain ⟨i, rfl⟩ | h := em (c ∈ Set.range e)
    /-
      case inl.intro
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₁ a₂ : α
      g : κ → α
      e : ι → κ
      he : Function.Injective e
      hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
      h : ∀ (i : ι), LE.le (f i) (g (e i))
      hf : HasProd (Function.extend e f 1) a₁
      hg : HasProd g a₂
      i : ι
      ⊢ LE.le (Function.extend e f 1 (e i)) (g (e i))
    -/
  · rw [he.extend_apply]
    /-
      case inl.intro
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₁ a₂ : α
      g : κ → α
      e : ι → κ
      he : Function.Injective e
      hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
      h : ∀ (i : ι), LE.le (f i) (g (e i))
      hf : HasProd (Function.extend e f 1) a₁
      hg : HasProd g a₂
      i : ι
      ⊢ LE.le (f i) (g (e i))
    -/
    exact h _
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₁ a₂ : α
      g : κ → α
      e : ι → κ
      he : Function.Injective e
      hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
      h✝ : ∀ (i : ι), LE.le (f i) (g (e i))
      hf : HasProd (Function.extend e f 1) a₁
      hg : HasProd g a₂
      c : κ
      h : Not (Membership.mem (Set.range e) c)
      ⊢ LE.le (Function.extend e f 1 c) (g c)
    -/
  · rw [extend_apply' _ _ _ h]
    /-
      case inr
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₁ a₂ : α
      g : κ → α
      e : ι → κ
      he : Function.Injective e
      hs : ∀ (c : κ), Not (Membership.mem (Set.range e) c) → LE.le 1 (g c)
      h✝ : ∀ (i : ι), LE.le (f i) (g (e i))
      hf : HasProd (Function.extend e f 1) a₁
      hg : HasProd g a₂
      c : κ
      h : Not (Membership.mem (Set.range e) c)
      ⊢ LE.le (1 c) (g c)
    -/
    exact hs _ h
    /-
      🎉 no goals
    -/


@[to_additive]
theorem tprod_le_tprod_of_inj {g : κ → α} (e : ι → κ) (he : Injective e)
    (hs : ∀ c, c ∉ Set.range e → 1 ≤ g c) (h : ∀ i, f i ≤ g (e i)) (hf : Multipliable f)
    (hg : Multipliable g) : tprod f ≤ tprod g :=
  hasProd_le_inj _ he hs h hf.hasProd hg.hasProd


@[to_additive]
lemma tprod_subtype_le {κ γ : Type*} [OrderedCommGroup γ] [UniformSpace γ] [UniformGroup γ]
    [OrderClosedTopology γ] [CompleteSpace γ] (f : κ → γ) (β : Set κ) (h : ∀ a : κ, 1 ≤ f a)
    (hf : Multipliable f) : (∏' (b : β), f b) ≤ (∏' (a : κ), f a) := by
  apply tprod_le_tprod_of_inj _
    (Subtype.coe_injective)
    (by simp only [Subtype.range_coe_subtype, Set.setOf_mem_eq, h, implies_true])
    (by simp only [le_refl, Subtype.forall, implies_true])
    (by apply hf.subtype)
  /-
    κ : Type u_4
    γ : Type u_5
    inst✝⁴ : OrderedCommGroup γ
    inst✝³ : UniformSpace γ
    inst✝² : UniformGroup γ
    inst✝¹ : OrderClosedTopology γ
    inst✝ : CompleteSpace γ
    f : κ → γ
    β : Set κ
    h : ∀ (a : κ), LE.le 1 (f a)
    hf : Multipliable f
    ⊢ Multipliable fun a => f a
  -/
  apply hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_le_hasProd (s : Finset ι) (hs : ∀ i, i ∉ s → 1 ≤ f i) (hf : HasProd f a) :
    ∏ i ∈ s, f i ≤ a :=
  ge_of_tendsto hf (eventually_atTop.2
    ⟨s, fun _t hst ↦ prod_le_prod_of_subset_of_one_le' hst fun i _ hbs ↦ hs i hbs⟩)


@[to_additive]
theorem isLUB_hasProd (h : ∀ i, 1 ≤ f i) (hf : HasProd f a) :
    IsLUB (Set.range fun s ↦ ∏ i ∈ s, f i) a := by
  classical
  exact isLUB_of_tendsto_atTop (Finset.prod_mono_set_of_one_le' h) hf


@[to_additive]
theorem le_hasProd (hf : HasProd f a) (i : ι) (hb : ∀ j, j ≠ i → 1 ≤ f j) : f i ≤ a :=
  calc
                               /-
                                 ι : Type u_1
                                 α : Type u_3
                                 inst✝² : OrderedCommMonoid α
                                 inst✝¹ : TopologicalSpace α
                                 inst✝ : OrderClosedTopology α
                                 f : ι → α
                                 a : α
                                 hf : HasProd f a
                                 i : ι
                                 hb : ∀ (j : ι), Ne j i → LE.le 1 (f j)
                                 ⊢ Eq (f i) ((Singleton.singleton i).prod fun i => f i)
                               -/
    f i = ∏ i ∈ {i}, f i := by rw [prod_singleton]
                               /-
                                 🎉 no goals
                               -/
                                   /-
                                     ι : Type u_1
                                     α : Type u_3
                                     inst✝² : OrderedCommMonoid α
                                     inst✝¹ : TopologicalSpace α
                                     inst✝ : OrderClosedTopology α
                                     f : ι → α
                                     a : α
                                     hf : HasProd f a
                                     i : ι
                                     hb : ∀ (j : ι), Ne j i → LE.le 1 (f j)
                                     ⊢ ∀ (i_1 : ι), Not (Membership.mem (Singleton.singleton i) i_1) → LE.le 1 (f i …
                                   -/
    _ ≤ a := prod_le_hasProd _ (by simpa) hf
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
theorem prod_le_tprod {f : ι → α} (s : Finset ι) (hs : ∀ i, i ∉ s → 1 ≤ f i) (hf : Multipliable f) :
    ∏ i ∈ s, f i ≤ ∏' i, f i :=
  prod_le_hasProd s hs hf.hasProd


@[to_additive]
theorem le_tprod (hf : Multipliable f) (i : ι) (hb : ∀ j, j ≠ i → 1 ≤ f j) : f i ≤ ∏' i, f i :=
  le_hasProd hf.hasProd i hb


@[to_additive]
theorem tprod_le_tprod (h : ∀ i, f i ≤ g i) (hf : Multipliable f) (hg : Multipliable g) :
    ∏' i, f i ≤ ∏' i, g i :=
  hasProd_le h hf.hasProd hg.hasProd


@[to_additive (attr := mono)]
theorem tprod_mono (hf : Multipliable f) (hg : Multipliable g) (h : f ≤ g) :
    ∏' n, f n ≤ ∏' n, g n :=
  tprod_le_tprod h hf hg


@[to_additive]
theorem tprod_le_of_prod_le (hf : Multipliable f) (h : ∀ s, ∏ i ∈ s, f i ≤ a₂) : ∏' i, f i ≤ a₂ :=
  hasProd_le_of_prod_le hf.hasProd h


@[to_additive]
theorem tprod_le_of_prod_le' (ha₂ : 1 ≤ a₂) (h : ∀ s, ∏ i ∈ s, f i ≤ a₂) : ∏' i, f i ≤ a₂ := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    a₂ : α
    ha₂ : LE.le 1 a₂
    h : ∀ (s : Finset ι), LE.le (s.prod fun i => f i) a₂
    ⊢ LE.le (tprod fun i => f i) a₂
  -/
  by_cases hf : Multipliable f
    /-
      case pos
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₂ : α
      ha₂ : LE.le 1 a₂
      h : ∀ (s : Finset ι), LE.le (s.prod fun i => f i) a₂
      hf : Multipliable f
      ⊢ LE.le (tprod fun i => f i) a₂
    -/
  · exact tprod_le_of_prod_le hf h
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₂ : α
      ha₂ : LE.le 1 a₂
      h : ∀ (s : Finset ι), LE.le (s.prod fun i => f i) a₂
      hf : Not (Multipliable f)
      ⊢ LE.le (tprod fun i => f i) a₂
    -/
  · rw [tprod_eq_one_of_not_multipliable hf]
    /-
      case neg
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      a₂ : α
      ha₂ : LE.le 1 a₂
      h : ∀ (s : Finset ι), LE.le (s.prod fun i => f i) a₂
      hf : Not (Multipliable f)
      ⊢ LE.le 1 a₂
    -/
    exact ha₂
    /-
      🎉 no goals
    -/


@[to_additive]
theorem HasProd.one_le (h : ∀ i, 1 ≤ g i) (ha : HasProd g a) : 1 ≤ a :=
  hasProd_le h hasProd_one ha


@[to_additive]
theorem HasProd.le_one (h : ∀ i, g i ≤ 1) (ha : HasProd g a) : a ≤ 1 :=
  hasProd_le h ha hasProd_one


@[to_additive tsum_nonneg]
theorem one_le_tprod (h : ∀ i, 1 ≤ g i) : 1 ≤ ∏' i, g i := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    g : ι → α
    h : ∀ (i : ι), LE.le 1 (g i)
    ⊢ LE.le 1 (tprod fun i => g i)
  -/
  by_cases hg : Multipliable g
    /-
      case pos
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      g : ι → α
      h : ∀ (i : ι), LE.le 1 (g i)
      hg : Multipliable g
      ⊢ LE.le 1 (tprod fun i => g i)
    -/
  · exact hg.hasProd.one_le h
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      g : ι → α
      h : ∀ (i : ι), LE.le 1 (g i)
      hg : Not (Multipliable g)
      ⊢ LE.le 1 (tprod fun i => g i)
    -/
  · rw [tprod_eq_one_of_not_multipliable hg]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem tprod_le_one (h : ∀ i, f i ≤ 1) : ∏' i, f i ≤ 1 := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    h : ∀ (i : ι), LE.le (f i) 1
    ⊢ LE.le (tprod fun i => f i) 1
  -/
  by_cases hf : Multipliable f
    /-
      case pos
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      h : ∀ (i : ι), LE.le (f i) 1
      hf : Multipliable f
      ⊢ LE.le (tprod fun i => f i) 1
    -/
  · exact hf.hasProd.le_one h
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      h : ∀ (i : ι), LE.le (f i) 1
      hf : Not (Multipliable f)
      ⊢ LE.le (tprod fun i => f i) 1
    -/
  · rw [tprod_eq_one_of_not_multipliable hf]
    /-
      🎉 no goals
    -/

-- Porting note: generalized from `OrderedAddCommGroup` to `OrderedAddCommMonoid`

@[to_additive]
theorem hasProd_one_iff_of_one_le (hf : ∀ i, 1 ≤ f i) : HasProd f 1 ↔ f = 1 := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : OrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    hf : ∀ (i : ι), LE.le 1 (f i)
    ⊢ Iff (HasProd f 1) (Eq f 1)
  -/
  refine ⟨fun hf' ↦ ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      hf : ∀ (i : ι), LE.le 1 (f i)
      hf' : HasProd f 1
      ⊢ Eq f 1
    -/
  · ext i
    /-
      case refine_1.h
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      hf : ∀ (i : ι), LE.le 1 (f i)
      hf' : HasProd f 1
      i : ι
      ⊢ Eq (f i) (1 i)
    -/
    exact (hf i).antisymm' (le_hasProd hf' _ fun j _ ↦ hf j)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      f : ι → α
      hf : ∀ (i : ι), LE.le 1 (f i)
      ⊢ Eq f 1 → HasProd f 1
    -/
  · rintro rfl
    /-
      case refine_2
      ι : Type u_1
      α : Type u_3
      inst✝² : OrderedCommMonoid α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      hf : ∀ (i : ι), LE.le 1 (1 i)
      ⊢ HasProd 1 1
    -/
    exact hasProd_one
    /-
      🎉 no goals
    -/


@[to_additive]
theorem hasProd_lt (h : f ≤ g) (hi : f i < g i) (hf : HasProd f a₁) (hg : HasProd g a₂) :
    a₁ < a₂ := by
  classical
  have : update f i 1 ≤ update g i 1 := update_le_update_iff.mpr ⟨rfl.le, fun i _ ↦ h i⟩
  have : 1 / f i * a₁ ≤ 1 / g i * a₂ := hasProd_le this (hf.update i 1) (hg.update i 1)
  simpa only [one_div, mul_inv_cancel_left] using mul_lt_mul_of_lt_of_le hi this


@[to_additive (attr := mono)]
theorem hasProd_strict_mono (hf : HasProd f a₁) (hg : HasProd g a₂) (h : f < g) : a₁ < a₂ :=
  let ⟨hle, _i, hi⟩ := Pi.lt_def.mp h
  hasProd_lt hle hi hf hg


@[to_additive]
theorem tprod_lt_tprod (h : f ≤ g) (hi : f i < g i) (hf : Multipliable f) (hg : Multipliable g) :
    ∏' n, f n < ∏' n, g n :=
  hasProd_lt h hi hf.hasProd hg.hasProd


@[to_additive (attr := mono)]
theorem tprod_strict_mono (hf : Multipliable f) (hg : Multipliable g) (h : f < g) :
    ∏' n, f n < ∏' n, g n :=
  let ⟨hle, _i, hi⟩ := Pi.lt_def.mp h
  tprod_lt_tprod hle hi hf hg


@[to_additive tsum_pos]
theorem one_lt_tprod (hsum : Multipliable g) (hg : ∀ i, 1 ≤ g i) (i : ι) (hi : 1 < g i) :
    1 < ∏' i, g i := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝³ : OrderedCommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalGroup α
    inst✝ : OrderClosedTopology α
    g : ι → α
    hsum : Multipliable g
    hg : ∀ (i : ι), LE.le 1 (g i)
    i : ι
    hi : LT.lt 1 (g i)
    ⊢ LT.lt 1 (tprod fun i => g i)
  -/
  rw [← tprod_one]
  /-
    ι : Type u_1
    α : Type u_3
    inst✝³ : OrderedCommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalGroup α
    inst✝ : OrderClosedTopology α
    g : ι → α
    hsum : Multipliable g
    hg : ∀ (i : ι), LE.le 1 (g i)
    i : ι
    hi : LT.lt 1 (g i)
    ⊢ LT.lt (tprod fun x => 1) (tprod fun i => g i)
  -/
  exact tprod_lt_tprod hg hi multipliable_one hsum
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_hasProd' (hf : HasProd f a) (i : ι) : f i ≤ a :=
  le_hasProd hf i fun _ _ ↦ one_le _


@[to_additive]
theorem le_tprod' (hf : Multipliable f) (i : ι) : f i ≤ ∏' i, f i :=
  le_tprod hf i fun _ _ ↦ one_le _


@[to_additive]
theorem hasProd_one_iff : HasProd f 1 ↔ ∀ x, f x = 1 :=
  (hasProd_one_iff_of_one_le fun _ ↦ one_le _).trans funext_iff


@[to_additive]
theorem tprod_eq_one_iff (hf : Multipliable f) : ∏' i, f i = 1 ↔ ∀ x, f x = 1 := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : CanonicallyOrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    hf : Multipliable f
    ⊢ Iff (Eq (tprod fun i => f i) 1) (∀ (x : ι), Eq (f x) 1)
  -/
  rw [← hasProd_one_iff, hf.hasProd_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tprod_ne_one_iff (hf : Multipliable f) : ∏' i, f i ≠ 1 ↔ ∃ x, f x ≠ 1 := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : CanonicallyOrderedCommMonoid α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    f : ι → α
    hf : Multipliable f
    ⊢ Iff (Ne (tprod fun i => f i) 1) (Exists fun x => Ne (f x) 1)
  -/
  rw [Ne, tprod_eq_one_iff hf, not_forall]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isLUB_hasProd' (hf : HasProd f a) : IsLUB (Set.range fun s ↦ ∏ i ∈ s, f i) a := by
  classical
  exact isLUB_of_tendsto_atTop (Finset.prod_mono_set' f) hf


@[to_additive]
theorem hasProd_of_isLUB_of_one_le [LinearOrderedCommMonoid α] [TopologicalSpace α]
    [OrderTopology α] {f : ι → α} (i : α) (h : ∀ i, 1 ≤ f i)
    (hf : IsLUB (Set.range fun s ↦ ∏ i ∈ s, f i) i) : HasProd f i :=
  tendsto_atTop_isLUB (Finset.prod_mono_set_of_one_le' h) hf


@[to_additive]
theorem hasProd_of_isLUB [CanonicallyLinearOrderedCommMonoid α] [TopologicalSpace α]
    [OrderTopology α] {f : ι → α} (b : α) (hf : IsLUB (Set.range fun s ↦ ∏ i ∈ s, f i) b) :
    HasProd f b :=
  tendsto_atTop_isLUB (Finset.prod_mono_set' f) hf


@[to_additive]
theorem multipliable_mabs_iff [LinearOrderedCommGroup α] [UniformSpace α] [UniformGroup α]
    [CompleteSpace α] {f : ι → α} : (Multipliable fun x ↦ mabs (f x)) ↔ Multipliable f :=
  let s := { x | 1 ≤ f x }
  have h1 : ∀ x : s, mabs (f x) = f x := fun x ↦ mabs_of_one_le x.2
  have h2 : ∀ x : ↑sᶜ, mabs (f x) = (f x)⁻¹ := fun x ↦ mabs_of_lt_one (not_le.1 x.2)
  calc (Multipliable fun x ↦ mabs (f x)) ↔
      (Multipliable fun x : s ↦ mabs (f x)) ∧ Multipliable fun x : ↑sᶜ ↦ mabs (f x) :=
        multipliable_subtype_and_compl.symm
                                                                                /-
                                                                                  ι : Type u_1
                                                                                  α : Type u_3
                                                                                  inst✝³ : LinearOrderedCommGroup α
                                                                                  inst✝² : UniformSpace α
                                                                                  inst✝¹ : UniformGroup α
                                                                                  inst✝ : CompleteSpace α
                                                                                  f : ι → α
                                                                                  s : Set ι := setOf fun x => LE.le 1 (f x)
                                                                                  h1 : ∀ (x : ↑s), Eq (mabs (f ↑x)) (f ↑x)
                                                                                  h2 : ∀ (x : ↑(HasCompl.compl s)), Eq (mabs (f ↑x)) (Inv.inv (f ↑x))
                                                                                  ⊢ Iff (And (Multipliable fun x => mabs (f ↑x)) (Multipliable fun x => mabs (f  …
                                                                                -/
  _ ↔ (Multipliable fun x : s ↦ f x) ∧ Multipliable fun x : ↑sᶜ ↦ (f x)⁻¹ := by simp only [h1, h2]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                           /-
                             ι : Type u_1
                             α : Type u_3
                             inst✝³ : LinearOrderedCommGroup α
                             inst✝² : UniformSpace α
                             inst✝¹ : UniformGroup α
                             inst✝ : CompleteSpace α
                             f : ι → α
                             s : Set ι := setOf fun x => LE.le 1 (f x)
                             h1 : ∀ (x : ↑s), Eq (mabs (f ↑x)) (f ↑x)
                             h2 : ∀ (x : ↑(HasCompl.compl s)), Eq (mabs (f ↑x)) (Inv.inv (f ↑x))
                             ⊢ Iff (And (Multipliable fun x => f ↑x) (Multipliable fun x => Inv.inv (f ↑x)) …
                           -/
  _ ↔ Multipliable f := by simp only [multipliable_inv_iff, multipliable_subtype_and_compl]
                           /-
                             🎉 no goals
                           -/


alias ⟨Summable.of_abs, Summable.abs⟩ := summable_abs_iff


theorem Finite.of_summable_const [LinearOrderedAddCommGroup α] [TopologicalSpace α] [Archimedean α]
    [OrderClosedTopology α] {b : α} (hb : 0 < b) (hf : Summable fun _ : ι ↦ b) :
    Finite ι := by
  have H : ∀ s : Finset ι, #s • b ≤ ∑' _ : ι, b := fun s ↦ by
    simpa using sum_le_hasSum s (fun a _ ↦ hb.le) hf.hasSum
  /-
    ι : Type u_1
    α : Type u_3
    inst✝³ : LinearOrderedAddCommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : Archimedean α
    inst✝ : OrderClosedTopology α
    b : α
    hb : LT.lt 0 b
    hf : Summable fun x => b
    H : ∀ (s : Finset ι), LE.le (HSMul.hSMul s.card b) (tsum fun x => b)
    ⊢ Finite ι
  -/
  obtain ⟨n, hn⟩ := Archimedean.arch (∑' _ : ι, b) hb
  have : ∀ s : Finset ι, #s ≤ n := fun s ↦ by
    simpa [nsmul_le_nsmul_iff_left hb] using (H s).trans hn
  /-
    case intro
    ι : Type u_1
    α : Type u_3
    inst✝³ : LinearOrderedAddCommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : Archimedean α
    inst✝ : OrderClosedTopology α
    b : α
    hb : LT.lt 0 b
    hf : Summable fun x => b
    H : ∀ (s : Finset ι), LE.le (HSMul.hSMul s.card b) (tsum fun x => b)
    n : Nat
    hn : LE.le (tsum fun x => b) (HSMul.hSMul n b)
    this : ∀ (s : Finset ι), LE.le s.card n
    ⊢ Finite ι
  -/
  have : Fintype ι := fintypeOfFinsetCardLe n this
  /-
    case intro
    ι : Type u_1
    α : Type u_3
    inst✝³ : LinearOrderedAddCommGroup α
    inst✝² : TopologicalSpace α
    inst✝¹ : Archimedean α
    inst✝ : OrderClosedTopology α
    b : α
    hb : LT.lt 0 b
    hf : Summable fun x => b
    H : ∀ (s : Finset ι), LE.le (HSMul.hSMul s.card b) (tsum fun x => b)
    n : Nat
    hn : LE.le (tsum fun x => b) (HSMul.hSMul n b)
    this✝ : ∀ (s : Finset ι), LE.le s.card n
    this : Fintype ι
    ⊢ Finite ι
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem Set.Finite.of_summable_const [LinearOrderedAddCommGroup α] [TopologicalSpace α]
    [Archimedean α] [OrderClosedTopology α] {b : α} (hb : 0 < b) (hf : Summable fun _ : ι ↦ b) :
    (Set.univ : Set ι).Finite :=
  finite_univ_iff.2 <| .of_summable_const hb hf


nonrec theorem HasProd.abs (hfx : HasProd f x) : HasProd (|f ·|) |x| := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : LinearOrderedCommRing α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    f : ι → α
    x : α
    hfx : HasProd f x
    ⊢ HasProd (fun x => _root_.abs (f x)) (_root_.abs x)
  -/
  simpa only [HasProd, ← abs_prod] using hfx.abs
  /-
    🎉 no goals
  -/


theorem Multipliable.abs (hf : Multipliable f) : Multipliable (|f ·|) :=
  let ⟨x, hx⟩ := hf; ⟨|x|, hx.abs⟩


theorem abs_tprod (hf : Multipliable f) : |∏' i, f i| = ∏' i, |f i| :=
  hf.hasProd.abs.tprod_eq.symm


theorem Summable.tendsto_atTop_of_pos [LinearOrderedField α] [TopologicalSpace α] [OrderTopology α]
    {f : ℕ → α} (hf : Summable f⁻¹) (hf' : ∀ n, 0 < f n) : Tendsto f atTop atTop :=
  inv_inv f ▸ Filter.Tendsto.inv_tendsto_nhdsGT_zero <|
    tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ hf.tendsto_atTop_zero <|
      Eventually.of_forall fun _ ↦ inv_pos.2 (hf' _)

