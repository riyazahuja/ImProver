/-- If `f : ℕ → M` has product `m`, then the partial products `∏ i ∈ range n, f i` converge
to `m`. -/
@[to_additive "If `f : ℕ → M` has sum `m`, then the partial sums `∑ i ∈ range n, f i` converge
to `m`."]
theorem tendsto_prod_nat {f : ℕ → M} (h : HasProd f m) :
    Tendsto (fun n ↦ ∏ i ∈ range n, f i) atTop (𝓝 m) :=
  h.comp tendsto_finset_range


/-- If `f : ℕ → M` is multipliable, then the partial products `∏ i ∈ range n, f i` converge
to `∏' i, f i`. -/
@[to_additive "If `f : ℕ → M` is summable, then the partial sums `∑ i ∈ range n, f i` converge
to `∑' i, f i`."]
theorem Multipliable.tendsto_prod_tprod_nat {f : ℕ → M} (h : Multipliable f) :
    Tendsto (fun n ↦ ∏ i ∈ range n, f i) atTop (𝓝 (∏' i, f i)) :=
  tendsto_prod_nat h.hasProd


@[to_additive]
theorem prod_range_mul {f : ℕ → M} {k : ℕ} (h : HasProd (fun n ↦ f (n + k)) m) :
    HasProd f ((∏ i ∈ range k, f i) * m) := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Nat → M
    k : Nat
    h : HasProd (fun n => f (HAdd.hAdd n k)) m
    ⊢ HasProd f (HMul.hMul ((Finset.range k).prod fun i => f i) m)
  -/
  refine ((range k).hasProd f).mul_compl ?_
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Nat → M
    k : Nat
    h : HasProd (fun n => f (HAdd.hAdd n k)) m
    ⊢ HasProd (Function.comp f Subtype.val) m
  -/
  rwa [← (notMemRangeEquiv k).symm.hasProd_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem zero_mul {f : ℕ → M} (h : HasProd (fun n ↦ f (n + 1)) m) :
    HasProd f (f 0 * m) := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Nat → M
    h : HasProd (fun n => f (HAdd.hAdd n 1)) m
    ⊢ HasProd f (HMul.hMul (f 0) m)
  -/
  simpa only [prod_range_one] using h.prod_range_mul
  /-
    🎉 no goals
  -/


@[to_additive]
theorem even_mul_odd {f : ℕ → M} (he : HasProd (fun k ↦ f (2 * k)) m)
    (ho : HasProd (fun k ↦ f (2 * k + 1)) m') : HasProd f (m * m') := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m m' : M
    inst✝ : ContinuousMul M
    f : Nat → M
    he : HasProd (fun k => f (HMul.hMul 2 k)) m
    ho : HasProd (fun k => f (HAdd.hAdd (HMul.hMul 2 k) 1)) m'
    ⊢ HasProd f (HMul.hMul m m')
  -/
  have := mul_right_injective₀ (two_ne_zero' ℕ)
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m m' : M
    inst✝ : ContinuousMul M
    f : Nat → M
    he : HasProd (fun k => f (HMul.hMul 2 k)) m
    ho : HasProd (fun k => f (HAdd.hAdd (HMul.hMul 2 k) 1)) m'
    this : Function.Injective fun x => HMul.hMul 2 x
    ⊢ HasProd f (HMul.hMul m m')
  -/
  replace ho := ((add_left_injective 1).comp this).hasProd_range_iff.2 ho
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m m' : M
    inst✝ : ContinuousMul M
    f : Nat → M
    he : HasProd (fun k => f (HMul.hMul 2 k)) m
    this : Function.Injective fun x => HMul.hMul 2 x
    ho : HasProd (fun x => f ↑x) m'
    ⊢ HasProd f (HMul.hMul m m')
  -/
  refine (this.hasProd_range_iff.2 he).mul_isCompl ?_ ho
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m m' : M
    inst✝ : ContinuousMul M
    f : Nat → M
    he : HasProd (fun k => f (HMul.hMul 2 k)) m
    this : Function.Injective fun x => HMul.hMul 2 x
    ho : HasProd (fun x => f ↑x) m'
    ⊢ IsCompl (Set.range fun x => HMul.hMul 2 x) (Set.range (Function.comp (fun x  …
  -/
  simpa [Function.comp_def] using Nat.isCompl_even_odd
  /-
    🎉 no goals
  -/


@[to_additive]
theorem hasProd_iff_tendsto_nat [T2Space M] {f : ℕ → M} (hf : Multipliable f) :
    HasProd f m ↔ Tendsto (fun n : ℕ ↦ ∏ i ∈ range n, f i) atTop (𝓝 m) := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : T2Space M
    f : Nat → M
    hf : Multipliable f
    ⊢ Iff (HasProd f m) (Filter.Tendsto (fun n => (Finset.range n).prod fun i => f …
  -/
  refine ⟨fun h ↦ h.tendsto_prod_nat, fun h ↦ ?_⟩
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : T2Space M
    f : Nat → M
    hf : Multipliable f
    h : Filter.Tendsto (fun n => (Finset.range n).prod fun i => f i) Filter.atTop  …
    ⊢ HasProd f m
  -/
  rw [tendsto_nhds_unique h hf.hasProd.tendsto_prod_nat]
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : T2Space M
    f : Nat → M
    hf : Multipliable f
    h : Filter.Tendsto (fun n => (Finset.range n).prod fun i => f i) Filter.atTop  …
    ⊢ HasProd f (tprod fun b => f b)
  -/
  exact hf.hasProd
  /-
    🎉 no goals
  -/


@[to_additive]
theorem comp_nat_add {f : ℕ → M} {k : ℕ} (h : Multipliable fun n ↦ f (n + k)) : Multipliable f :=
  h.hasProd.prod_range_mul.multipliable


@[to_additive]
theorem even_mul_odd {f : ℕ → M} (he : Multipliable fun k ↦ f (2 * k))
    (ho : Multipliable fun k ↦ f (2 * k + 1)) : Multipliable f :=
  (he.hasProd.even_mul_odd ho.hasProd).multipliable


/-- You can compute a product over an encodable type by multiplying over the natural numbers and
taking a supremum. -/
@[to_additive "You can compute a sum over an encodable type by summing over the natural numbers and
  taking a supremum. This is useful for outer measures."]
theorem tprod_iSup_decode₂ [CompleteLattice α] (m : α → M) (m0 : m ⊥ = 1) (s : β → α) :
    ∏' i : ℕ, m (⨆ b ∈ decode₂ β i, s b) = ∏' b : β, m (s b) := by
  /-
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    α : Type u_3
    β : Type u_4
    inst✝¹ : Encodable β
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    s : β → α
    ⊢ Eq (tprod fun i => m (iSup fun b => iSup fun h => s b)) (tprod fun b => m (s …
  -/
  rw [← tprod_extend_one (@encode_injective β _)]
  /-
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    α : Type u_3
    β : Type u_4
    inst✝¹ : Encodable β
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    s : β → α
    ⊢ Eq (tprod fun i => m (iSup fun b => iSup fun h => s b)) (tprod fun y => Func …
  -/
  refine tprod_congr fun n ↦ ?_
  /-
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    α : Type u_3
    β : Type u_4
    inst✝¹ : Encodable β
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    s : β → α
    n : Nat
    ⊢ Eq (m (iSup fun b => iSup fun h => s b)) (Function.extend Encodable.encode ( …
  -/
  rcases em (n ∈ Set.range (encode : β → ℕ)) with ⟨a, rfl⟩ | hn
    /-
      case inl.intro
      M : Type u_1
      inst✝³ : CommMonoid M
      inst✝² : TopologicalSpace M
      α : Type u_3
      β : Type u_4
      inst✝¹ : Encodable β
      inst✝ : CompleteLattice α
      m : α → M
      m0 : Eq (m Bot.bot) 1
      s : β → α
      a : β
      ⊢ Eq (m (iSup fun b => iSup fun h => s b)) (Function.extend Encodable.encode ( …
    -/
  · simp [encode_injective.extend_apply]
    /-
      🎉 no goals
    -/
    /-
      case inr
      M : Type u_1
      inst✝³ : CommMonoid M
      inst✝² : TopologicalSpace M
      α : Type u_3
      β : Type u_4
      inst✝¹ : Encodable β
      inst✝ : CompleteLattice α
      m : α → M
      m0 : Eq (m Bot.bot) 1
      s : β → α
      n : Nat
      hn : Not (Membership.mem (Set.range Encodable.encode) n)
      ⊢ Eq (m (iSup fun b => iSup fun h => s b)) (Function.extend Encodable.encode ( …
    -/
  · rw [extend_apply' _ _ _ hn]
    /-
      case inr
      M : Type u_1
      inst✝³ : CommMonoid M
      inst✝² : TopologicalSpace M
      α : Type u_3
      β : Type u_4
      inst✝¹ : Encodable β
      inst✝ : CompleteLattice α
      m : α → M
      m0 : Eq (m Bot.bot) 1
      s : β → α
      n : Nat
      hn : Not (Membership.mem (Set.range Encodable.encode) n)
      ⊢ Eq (m (iSup fun b => iSup fun h => s b)) (1 n)
    -/
    rw [← decode₂_ne_none_iff, ne_eq, not_not] at hn
    /-
      case inr
      M : Type u_1
      inst✝³ : CommMonoid M
      inst✝² : TopologicalSpace M
      α : Type u_3
      β : Type u_4
      inst✝¹ : Encodable β
      inst✝ : CompleteLattice α
      m : α → M
      m0 : Eq (m Bot.bot) 1
      s : β → α
      n : Nat
      hn : Eq (Encodable.decode₂ β n) Option.none
      ⊢ Eq (m (iSup fun b => iSup fun h => s b)) (1 n)
    -/
    simp [hn, m0]
    /-
      🎉 no goals
    -/


/-- `tprod_iSup_decode₂` specialized to the complete lattice of sets. -/
@[to_additive "`tsum_iSup_decode₂` specialized to the complete lattice of sets."]
theorem tprod_iUnion_decode₂ (m : Set α → M) (m0 : m ∅ = 1) (s : β → Set α) :
    ∏' i, m (⋃ b ∈ decode₂ β i, s b) = ∏' b, m (s b) :=
  tprod_iSup_decode₂ m m0 s


/-- If a function is countably sub-multiplicative then it is sub-multiplicative on countable
types -/
@[to_additive "If a function is countably sub-additive then it is sub-additive on countable types"]
theorem rel_iSup_tprod [CompleteLattice α] (m : α → M) (m0 : m ⊥ = 1) (R : M → M → Prop)
    (m_iSup : ∀ s : ℕ → α, R (m (⨆ i, s i)) (∏' i, m (s i))) (s : β → α) :
    R (m (⨆ b : β, s b)) (∏' b : β, m (s b)) := by
  /-
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    α : Type u_3
    β : Type u_4
    inst✝¹ : Countable β
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    R : M → M → Prop
    m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
    s : β → α
    ⊢ R (m (iSup fun b => s b)) (tprod fun b => m (s b))
  -/
  cases nonempty_encodable β
  /-
    case intro
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    α : Type u_3
    β : Type u_4
    inst✝¹ : Countable β
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    R : M → M → Prop
    m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
    s : β → α
    val✝ : Encodable β
    ⊢ R (m (iSup fun b => s b)) (tprod fun b => m (s b))
  -/
  rw [← iSup_decode₂, ← tprod_iSup_decode₂ _ m0 s]
  /-
    case intro
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    α : Type u_3
    β : Type u_4
    inst✝¹ : Countable β
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    R : M → M → Prop
    m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
    s : β → α
    val✝ : Encodable β
    ⊢ R (m (iSup fun i => iSup fun b => iSup fun h => s b)) (tprod fun i => m (iSu …
  -/
  exact m_iSup _
  /-
    🎉 no goals
  -/


/-- If a function is countably sub-multiplicative then it is sub-multiplicative on finite sets -/
@[to_additive "If a function is countably sub-additive then it is sub-additive on finite sets"]
theorem rel_iSup_prod [CompleteLattice α] (m : α → M) (m0 : m ⊥ = 1) (R : M → M → Prop)
    (m_iSup : ∀ s : ℕ → α, R (m (⨆ i, s i)) (∏' i, m (s i))) (s : γ → α) (t : Finset γ) :
    R (m (⨆ d ∈ t, s d)) (∏ d ∈ t, m (s d)) := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    α : Type u_3
    γ : Type u_5
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    R : M → M → Prop
    m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
    s : γ → α
    t : Finset γ
    ⊢ R (m (iSup fun d => iSup fun h => s d)) (t.prod fun d => m (s d))
  -/
  rw [iSup_subtype', ← Finset.tprod_subtype]
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    α : Type u_3
    γ : Type u_5
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    R : M → M → Prop
    m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
    s : γ → α
    t : Finset γ
    ⊢ R (m (iSup fun x => s ↑x)) (tprod fun x => m (s ↑x))
  -/
  exact rel_iSup_tprod m m0 R m_iSup _
  /-
    🎉 no goals
  -/


/-- If a function is countably sub-multiplicative then it is binary sub-multiplicative -/
@[to_additive "If a function is countably sub-additive then it is binary sub-additive"]
theorem rel_sup_mul [CompleteLattice α] (m : α → M) (m0 : m ⊥ = 1) (R : M → M → Prop)
    (m_iSup : ∀ s : ℕ → α, R (m (⨆ i, s i)) (∏' i, m (s i))) (s₁ s₂ : α) :
    R (m (s₁ ⊔ s₂)) (m s₁ * m s₂) := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    α : Type u_3
    inst✝ : CompleteLattice α
    m : α → M
    m0 : Eq (m Bot.bot) 1
    R : M → M → Prop
    m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
    s₁ s₂ : α
    ⊢ R (m (Max.max s₁ s₂)) (HMul.hMul (m s₁) (m s₂))
  -/
  convert rel_iSup_tprod m m0 R m_iSup fun b ↦ cond b s₁ s₂
    /-
      case h.e'_1.h.e'_1
      M : Type u_1
      inst✝² : CommMonoid M
      inst✝¹ : TopologicalSpace M
      α : Type u_3
      inst✝ : CompleteLattice α
      m : α → M
      m0 : Eq (m Bot.bot) 1
      R : M → M → Prop
      m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
      s₁ s₂ : α
      ⊢ Eq (Max.max s₁ s₂) (iSup fun b => cond b s₁ s₂)
    -/
  · simp only [iSup_bool_eq, cond]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2
      M : Type u_1
      inst✝² : CommMonoid M
      inst✝¹ : TopologicalSpace M
      α : Type u_3
      inst✝ : CompleteLattice α
      m : α → M
      m0 : Eq (m Bot.bot) 1
      R : M → M → Prop
      m_iSup : ∀ (s : Nat → α), R (m (iSup fun i => s i)) (tprod fun i => m (s i))
      s₁ s₂ : α
      ⊢ Eq (HMul.hMul (m s₁) (m s₂)) (tprod fun b => m (cond b s₁ s₂))
    -/
  · rw [tprod_fintype, Fintype.prod_bool, cond, cond]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_mul_tprod_nat_mul'
    {f : ℕ → M} {k : ℕ} (h : Multipliable (fun n ↦ f (n + k))) :
    ((∏ i ∈ range k, f i) * ∏' i, f (i + k)) = ∏' i, f i :=
  h.hasProd.prod_range_mul.tprod_eq.symm


@[to_additive]
theorem tprod_eq_zero_mul'
    {f : ℕ → M} (hf : Multipliable (fun n ↦ f (n + 1))) :
    ∏' b, f b = f 0 * ∏' b, f (b + 1) := by
  /-
    M : Type u_1
    inst✝³ : CommMonoid M
    inst✝² : TopologicalSpace M
    inst✝¹ : T2Space M
    inst✝ : ContinuousMul M
    f : Nat → M
    hf : Multipliable fun n => f (HAdd.hAdd n 1)
    ⊢ Eq (tprod fun b => f b) (HMul.hMul (f 0) (tprod fun b => f (HAdd.hAdd b 1)))
  -/
  simpa only [prod_range_one] using (prod_mul_tprod_nat_mul' hf).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tprod_even_mul_odd {f : ℕ → M} (he : Multipliable fun k ↦ f (2 * k))
    (ho : Multipliable fun k ↦ f (2 * k + 1)) :
    (∏' k, f (2 * k)) * ∏' k, f (2 * k + 1) = ∏' k, f k :=
  (he.hasProd.even_mul_odd ho.hasProd).tprod_eq.symm


@[to_additive]
theorem hasProd_nat_add_iff {f : ℕ → G} (k : ℕ) :
    HasProd (fun n ↦ f (n + k)) g ↔ HasProd f (g * ∏ i ∈ range k, f i) := by
  /-
    G : Type u_2
    inst✝² : CommGroup G
    g : G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Nat → G
    k : Nat
    ⊢ Iff (HasProd (fun n => f (HAdd.hAdd n k)) g) (HasProd f (HMul.hMul g ((Finse …
  -/
  refine Iff.trans ?_ (range k).hasProd_compl_iff
  /-
    G : Type u_2
    inst✝² : CommGroup G
    g : G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Nat → G
    k : Nat
    ⊢ Iff (HasProd (fun n => f (HAdd.hAdd n k)) g) (HasProd (fun x => f ↑x) g)
  -/
  rw [← (notMemRangeEquiv k).symm.hasProd_iff, Function.comp_def, coe_notMemRangeEquiv_symm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multipliable_nat_add_iff {f : ℕ → G} (k : ℕ) :
    (Multipliable fun n ↦ f (n + k)) ↔ Multipliable f :=
  Iff.symm <|
    (Equiv.mulRight (∏ i ∈ range k, f i)).surjective.multipliable_iff_of_hasProd_iff
      (hasProd_nat_add_iff k).symm


@[to_additive]
theorem hasProd_nat_add_iff' {f : ℕ → G} (k : ℕ) :
    HasProd (fun n ↦ f (n + k)) (g / ∏ i ∈ range k, f i) ↔ HasProd f g := by
  /-
    G : Type u_2
    inst✝² : CommGroup G
    g : G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Nat → G
    k : Nat
    ⊢ Iff (HasProd (fun n => f (HAdd.hAdd n k)) (HDiv.hDiv g ((Finset.range k).pro …
  -/
  simp [hasProd_nat_add_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_mul_tprod_nat_add [T2Space G] {f : ℕ → G} (k : ℕ) (h : Multipliable f) :
    ((∏ i ∈ range k, f i) * ∏' i, f (i + k)) = ∏' i, f i :=
  prod_mul_tprod_nat_mul' <| (multipliable_nat_add_iff k).2 h


@[to_additive]
theorem tprod_eq_zero_mul [T2Space G] {f : ℕ → G} (hf : Multipliable f) :
    ∏' b, f b = f 0 * ∏' b, f (b + 1) :=
  tprod_eq_zero_mul' <| (multipliable_nat_add_iff 1).2 hf


/-- For `f : ℕ → G`, the product `∏' k, f (k + i)` tends to one. This does not require a
multipliability assumption on `f`, as otherwise all such products are one. -/
@[to_additive "For `f : ℕ → G`, the sum `∑' k, f (k + i)` tends to zero. This does not require a
summability assumption on `f`, as otherwise all such sums are zero."]
theorem tendsto_prod_nat_add [T2Space G] (f : ℕ → G) :
    Tendsto (fun i ↦ ∏' k, f (k + i)) atTop (𝓝 1) := by
  /-
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : T2Space G
    f : Nat → G
    ⊢ Filter.Tendsto (fun i => tprod fun k => f (HAdd.hAdd k i)) Filter.atTop (nhd …
  -/
  by_cases hf : Multipliable f
  · have h₀ : (fun i ↦ (∏' i, f i) / ∏ j ∈ range i, f j) = fun i ↦ ∏' k : ℕ, f (k + i) := by
      ext1 i
      rw [div_eq_iff_eq_mul, mul_comm, prod_mul_tprod_nat_add i hf]
    /-
      case pos
      G : Type u_2
      inst✝³ : CommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      f : Nat → G
      hf : Multipliable f
      h₀ : Eq (fun i => HDiv.hDiv (tprod fun i => f i) ((Finset.range i).prod fun j  …
      ⊢ Filter.Tendsto (fun i => tprod fun k => f (HAdd.hAdd k i)) Filter.atTop (nhd …
    -/
    have h₁ : Tendsto (fun _ : ℕ ↦ ∏' i, f i) atTop (𝓝 (∏' i, f i)) := tendsto_const_nhds
    /-
      case pos
      G : Type u_2
      inst✝³ : CommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      f : Nat → G
      hf : Multipliable f
      h₀ : Eq (fun i => HDiv.hDiv (tprod fun i => f i) ((Finset.range i).prod fun j  …
      h₁ : Filter.Tendsto (fun x => tprod fun i => f i) Filter.atTop (nhds (tprod fu …
      ⊢ Filter.Tendsto (fun i => tprod fun k => f (HAdd.hAdd k i)) Filter.atTop (nhd …
    -/
    simpa only [h₀, div_self'] using Tendsto.div' h₁ hf.hasProd.tendsto_prod_nat
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_2
      inst✝³ : CommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      f : Nat → G
      hf : Not (Multipliable f)
      ⊢ Filter.Tendsto (fun i => tprod fun k => f (HAdd.hAdd k i)) Filter.atTop (nhd …
    -/
  · refine tendsto_const_nhds.congr fun n ↦ (tprod_eq_one_of_not_multipliable ?_).symm
    /-
      case neg
      G : Type u_2
      inst✝³ : CommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : T2Space G
      f : Nat → G
      hf : Not (Multipliable f)
      n : Nat
      ⊢ Not (Multipliable fun k => f (HAdd.hAdd k n))
    -/
    rwa [multipliable_nat_add_iff n]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem cauchySeq_finset_iff_nat_tprod_vanishing {f : ℕ → G} :
    (CauchySeq fun s : Finset ℕ ↦ ∏ n ∈ s, f n) ↔
      ∀ e ∈ 𝓝 (1 : G), ∃ N : ℕ, ∀ t ⊆ {n | N ≤ n}, (∏' n : t, f n) ∈ e := by
  /-
    G : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : UniformSpace G
    inst✝ : UniformGroup G
    f : Nat → G
    ⊢ Iff (CauchySeq fun s => s.prod fun n => f n) (∀ (e : Set G), Membership.mem  …
  -/
  refine cauchySeq_finset_iff_tprod_vanishing.trans ⟨fun vanish e he ↦ ?_, fun vanish e he ↦ ?_⟩
    /-
      case refine_1
      G : Type u_2
      inst✝² : CommGroup G
      inst✝¹ : UniformSpace G
      inst✝ : UniformGroup G
      f : Nat → G
      vanish : ∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Set …
      e : Set G
      he : Membership.mem (nhds 1) e
      ⊢ Exists fun N => ∀ (t : Set Nat), HasSubset.Subset t (setOf fun n => LE.le N  …
    -/
  · obtain ⟨s, hs⟩ := vanish e he
    refine ⟨if h : s.Nonempty then s.max' h + 1 else 0,
      fun t ht ↦ hs _ <| Set.disjoint_left.mpr ?_⟩
    /-
      case refine_1.intro
      G : Type u_2
      inst✝² : CommGroup G
      inst✝¹ : UniformSpace G
      inst✝ : UniformGroup G
      f : Nat → G
      vanish : ∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Set …
      e : Set G
      he : Membership.mem (nhds 1) e
      s : Finset Nat
      hs : ∀ (t : Set Nat), Disjoint t ↑s → Membership.mem e (tprod fun b => f ↑b)
      t : Set Nat
      ht : HasSubset.Subset t (setOf fun n => LE.le (dite s.Nonempty (fun h => HAdd. …
      ⊢ ∀ ⦃a : Nat⦄, Membership.mem t a → Not (Membership.mem (↑s) a)
    -/
    split_ifs at ht with h
      /-
        case pos
        G : Type u_2
        inst✝² : CommGroup G
        inst✝¹ : UniformSpace G
        inst✝ : UniformGroup G
        f : Nat → G
        vanish : ∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Set …
        e : Set G
        he : Membership.mem (nhds 1) e
        s : Finset Nat
        hs : ∀ (t : Set Nat), Disjoint t ↑s → Membership.mem e (tprod fun b => f ↑b)
        t : Set Nat
        h : s.Nonempty
        ht : HasSubset.Subset t (setOf fun n => LE.le (HAdd.hAdd (s.max' h) 1) n)
        ⊢ ∀ ⦃a : Nat⦄, Membership.mem t a → Not (Membership.mem (↑s) a)
      -/
    · exact fun m hmt hms ↦ (s.le_max' _ hms).not_lt (Nat.succ_le_iff.mp <| ht hmt)
      /-
        🎉 no goals
      -/
      /-
        case neg
        G : Type u_2
        inst✝² : CommGroup G
        inst✝¹ : UniformSpace G
        inst✝ : UniformGroup G
        f : Nat → G
        vanish : ∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Set …
        e : Set G
        he : Membership.mem (nhds 1) e
        s : Finset Nat
        hs : ∀ (t : Set Nat), Disjoint t ↑s → Membership.mem e (tprod fun b => f ↑b)
        t : Set Nat
        h : Not s.Nonempty
        ht : HasSubset.Subset t (setOf fun n => LE.le 0 n)
        ⊢ ∀ ⦃a : Nat⦄, Membership.mem t a → Not (Membership.mem (↑s) a)
      -/
    · exact fun _ _ hs ↦ h ⟨_, hs⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      G : Type u_2
      inst✝² : CommGroup G
      inst✝¹ : UniformSpace G
      inst✝ : UniformGroup G
      f : Nat → G
      vanish : ∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun N => ∀ (t : Set …
      e : Set G
      he : Membership.mem (nhds 1) e
      ⊢ Exists fun s => ∀ (t : Set Nat), Disjoint t ↑s → Membership.mem e (tprod fun …
    -/
  · obtain ⟨N, hN⟩ := vanish e he
    exact ⟨range N, fun t ht ↦ hN _ fun n hnt ↦
      le_of_not_lt fun h ↦ Set.disjoint_left.mp ht hnt (mem_range.mpr h)⟩


@[to_additive]
theorem multipliable_iff_nat_tprod_vanishing {f : ℕ → G} : Multipliable f ↔
    ∀ e ∈ 𝓝 1, ∃ N : ℕ, ∀ t ⊆ {n | N ≤ n}, (∏' n : t, f n) ∈ e := by
  /-
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Nat → G
    ⊢ Iff (Multipliable f) (∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun  …
  -/
  rw [multipliable_iff_cauchySeq_finset, cauchySeq_finset_iff_nat_tprod_vanishing]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multipliable.nat_tprod_vanishing {f : ℕ → G} (hf : Multipliable f) ⦃e : Set G⦄
    (he : e ∈ 𝓝 1) : ∃ N : ℕ, ∀ t ⊆ {n | N ≤ n}, (∏' n : t, f n) ∈ e :=
  letI : UniformSpace G := TopologicalGroup.toUniformSpace G
  have : UniformGroup G := comm_topologicalGroup_is_uniform
  cauchySeq_finset_iff_nat_tprod_vanishing.1 hf.hasProd.cauchySeq e he


@[to_additive]
theorem Multipliable.tendsto_atTop_one {f : ℕ → G} (hf : Multipliable f) :
    Tendsto f atTop (𝓝 1) := by
  /-
    G : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Nat → G
    hf : Multipliable f
    ⊢ Filter.Tendsto f Filter.atTop (nhds 1)
  -/
  rw [← Nat.cofinite_eq_atTop]
  /-
    G : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Nat → G
    hf : Multipliable f
    ⊢ Filter.Tendsto f Filter.cofinite (nhds 1)
  -/
  exact hf.tendsto_cofinite_one
  /-
    🎉 no goals
  -/


@[to_additive HasSum.nat_add_neg_add_one]
lemma HasProd.nat_mul_neg_add_one {f : ℤ → M} (hf : HasProd f m) :
    HasProd (fun n : ℕ ↦ f n * f (-(n + 1))) m := by
  /-
    M : Type u_1
    inst✝¹ : CommMonoid M
    inst✝ : TopologicalSpace M
    m : M
    f : Int → M
    hf : HasProd f m
    ⊢ HasProd (fun n => HMul.hMul (f ↑n) (f (Neg.neg (HAdd.hAdd (↑n) 1)))) m
  -/
  change HasProd (fun n : ℕ ↦ f n * f (Int.negSucc n)) m
  /-
    M : Type u_1
    inst✝¹ : CommMonoid M
    inst✝ : TopologicalSpace M
    m : M
    f : Int → M
    hf : HasProd f m
    ⊢ HasProd (fun n => HMul.hMul (f ↑n) (f (Int.negSucc n))) m
  -/
  have : Injective Int.negSucc := @Int.negSucc.inj
  /-
    M : Type u_1
    inst✝¹ : CommMonoid M
    inst✝ : TopologicalSpace M
    m : M
    f : Int → M
    hf : HasProd f m
    this : Function.Injective Int.negSucc
    ⊢ HasProd (fun n => HMul.hMul (f ↑n) (f (Int.negSucc n))) m
  -/
  refine hf.hasProd_of_prod_eq fun u ↦ ?_
  refine ⟨u.preimage _ Nat.cast_injective.injOn ∪ u.preimage _ this.injOn,
      fun v' hv' ↦ ⟨v'.image Nat.cast ∪ v'.image Int.negSucc, fun x hx ↦ ?_, ?_⟩⟩
    /-
      case refine_1
      M : Type u_1
      inst✝¹ : CommMonoid M
      inst✝ : TopologicalSpace M
      m : M
      f : Int → M
      hf : HasProd f m
      this : Function.Injective Int.negSucc
      u : Finset Int
      v' : Finset Nat
      hv' : HasSubset.Subset (Union.union (u.preimage Nat.cast ⋯) (u.preimage Int.ne …
      x : Int
      hx : Membership.mem u x
      ⊢ Membership.mem (Union.union (Finset.image Nat.cast v') (Finset.image Int.neg …
    -/
  · simp only [mem_union, mem_image]
    /-
      case refine_1
      M : Type u_1
      inst✝¹ : CommMonoid M
      inst✝ : TopologicalSpace M
      m : M
      f : Int → M
      hf : HasProd f m
      this : Function.Injective Int.negSucc
      u : Finset Int
      v' : Finset Nat
      hv' : HasSubset.Subset (Union.union (u.preimage Nat.cast ⋯) (u.preimage Int.ne …
      x : Int
      hx : Membership.mem u x
      ⊢ Or (Exists fun a => And (Membership.mem v' a) (Eq (↑a) x)) (Exists fun a =>  …
    -/
    cases x
      /-
        case refine_1.ofNat
        M : Type u_1
        inst✝¹ : CommMonoid M
        inst✝ : TopologicalSpace M
        m : M
        f : Int → M
        hf : HasProd f m
        this : Function.Injective Int.negSucc
        u : Finset Int
        v' : Finset Nat
        hv' : HasSubset.Subset (Union.union (u.preimage Nat.cast ⋯) (u.preimage Int.ne …
        a✝ : Nat
        hx : Membership.mem u (Int.ofNat a✝)
        ⊢ Or (Exists fun a => And (Membership.mem v' a) (Eq (↑a) (Int.ofNat a✝))) (Exi …
      -/
    · exact Or.inl ⟨_, hv' (by simpa using Or.inl hx), rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.negSucc
        M : Type u_1
        inst✝¹ : CommMonoid M
        inst✝ : TopologicalSpace M
        m : M
        f : Int → M
        hf : HasProd f m
        this : Function.Injective Int.negSucc
        u : Finset Int
        v' : Finset Nat
        hv' : HasSubset.Subset (Union.union (u.preimage Nat.cast ⋯) (u.preimage Int.ne …
        a✝ : Nat
        hx : Membership.mem u (Int.negSucc a✝)
        ⊢ Or (Exists fun a => And (Membership.mem v' a) (Eq (↑a) (Int.negSucc a✝))) (E …
      -/
    · exact Or.inr ⟨_, hv' (by simpa using Or.inr hx), rfl⟩
      /-
        🎉 no goals
      -/
  · rw [prod_union, prod_image Nat.cast_injective.injOn, prod_image this.injOn,
      prod_mul_distrib]
    simp only [disjoint_iff_ne, mem_image, ne_eq, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂, not_false_eq_true, implies_true, forall_const, reduceCtorEq]


@[to_additive Summable.nat_add_neg_add_one]
lemma Multipliable.nat_mul_neg_add_one {f : ℤ → M} (hf : Multipliable f) :
    Multipliable (fun n : ℕ ↦ f n * f (-(n + 1))) :=
  hf.hasProd.nat_mul_neg_add_one.multipliable


@[to_additive tsum_nat_add_neg_add_one]
lemma tprod_nat_mul_neg_add_one [T2Space M] {f : ℤ → M} (hf : Multipliable f) :
    ∏' (n : ℕ), (f n * f (-(n + 1))) = ∏' (n : ℤ), f n :=
  hf.hasProd.nat_mul_neg_add_one.tprod_eq


@[to_additive HasSum.of_nat_of_neg_add_one]
lemma HasProd.of_nat_of_neg_add_one {f : ℤ → M}
    (hf₁ : HasProd (fun n : ℕ ↦ f n) m) (hf₂ : HasProd (fun n : ℕ ↦ f (-(n + 1))) m') :
    HasProd f (m * m') := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m m' : M
    inst✝ : ContinuousMul M
    f : Int → M
    hf₁ : HasProd (fun n => f ↑n) m
    hf₂ : HasProd (fun n => f (Neg.neg (HAdd.hAdd (↑n) 1))) m'
    ⊢ HasProd f (HMul.hMul m m')
  -/
  have hi₂ : Injective Int.negSucc := @Int.negSucc.inj
  have : IsCompl (Set.range ((↑) : ℕ → ℤ)) (Set.range Int.negSucc) := by
    constructor
    · rw [disjoint_iff_inf_le]
      rintro _ ⟨⟨i, rfl⟩, ⟨j, ⟨⟩⟩⟩
    · rw [codisjoint_iff_le_sup]
      rintro (i | j) <;> simp
  exact (Nat.cast_injective.hasProd_range_iff.mpr hf₁).mul_isCompl
    this (hi₂.hasProd_range_iff.mpr hf₂)


@[deprecated (since := "2024-03-04")] alias HasSum.nonneg_add_neg := HasSum.of_nat_of_neg_add_one


@[to_additive Summable.of_nat_of_neg_add_one]
lemma Multipliable.of_nat_of_neg_add_one {f : ℤ → M}
    (hf₁ : Multipliable fun n : ℕ ↦ f n) (hf₂ : Multipliable fun n : ℕ ↦ f (-(n + 1))) :
    Multipliable f :=
  (hf₁.hasProd.of_nat_of_neg_add_one hf₂.hasProd).multipliable


@[to_additive tsum_of_nat_of_neg_add_one]
lemma tprod_of_nat_of_neg_add_one [T2Space M] {f : ℤ → M}
    (hf₁ : Multipliable fun n : ℕ ↦ f n) (hf₂ : Multipliable fun n : ℕ ↦ f (-(n + 1))) :
    ∏' n : ℤ, f n = (∏' n : ℕ, f n) * ∏' n : ℕ, f (-(n + 1)) :=
  (hf₁.hasProd.of_nat_of_neg_add_one hf₂.hasProd).tprod_eq


/-- If `f₀, f₁, f₂, ...` and `g₀, g₁, g₂, ...` have products `a`, `b` respectively, then
the `ℤ`-indexed sequence: `..., g₂, g₁, g₀, f₀, f₁, f₂, ...` (with `f₀` at the `0`-th position) has
product `a + b`. -/
@[to_additive "If `f₀, f₁, f₂, ...` and `g₀, g₁, g₂, ...` have sums `a`, `b` respectively, then
the `ℤ`-indexed sequence: `..., g₂, g₁, g₀, f₀, f₁, f₂, ...` (with `f₀` at the `0`-th position) has
sum `a + b`."]
lemma HasProd.int_rec {f g : ℕ → M} (hf : HasProd f m) (hg : HasProd g m') :
    HasProd (Int.rec f g) (m * m') :=
  HasProd.of_nat_of_neg_add_one hf hg


/-- If `f₀, f₁, f₂, ...` and `g₀, g₁, g₂, ...` are both multipliable then so is the
`ℤ`-indexed sequence: `..., g₂, g₁, g₀, f₀, f₁, f₂, ...` (with `f₀` at the `0`-th position). -/
@[to_additive "If `f₀, f₁, f₂, ...` and `g₀, g₁, g₂, ...` are both summable then so is the
`ℤ`-indexed sequence: `..., g₂, g₁, g₀, f₀, f₁, f₂, ...` (with `f₀` at the `0`-th position)."]
lemma Multipliable.int_rec {f g : ℕ → M} (hf : Multipliable f) (hg : Multipliable g) :
    Multipliable (Int.rec f g) :=
  .of_nat_of_neg_add_one hf hg


/-- If `f₀, f₁, f₂, ...` and `g₀, g₁, g₂, ...` are both multipliable, then the product of the
`ℤ`-indexed sequence: `..., g₂, g₁, g₀, f₀, f₁, f₂, ...` (with `f₀` at the `0`-th position) is
`(∏' n, f n) * ∏' n, g n`. -/
@[to_additive "If `f₀, f₁, f₂, ...` and `g₀, g₁, g₂, ...` are both summable, then the sum of the
`ℤ`-indexed sequence: `..., g₂, g₁, g₀, f₀, f₁, f₂, ...` (with `f₀` at the `0`-th position) is
`∑' n, f n + ∑' n, g n`."]
lemma tprod_int_rec [T2Space M] {f g : ℕ → M} (hf : Multipliable f) (hg : Multipliable g) :
    ∏' n : ℤ, Int.rec f g n = (∏' n : ℕ, f n) * ∏' n : ℕ, g n :=
  (hf.hasProd.int_rec hg.hasProd).tprod_eq


@[to_additive]
theorem HasProd.nat_mul_neg {f : ℤ → M} (hf : HasProd f m) :
    HasProd (fun n : ℕ ↦ f n * f (-n)) (m * f 0) := by
  -- Note this is much easier to prove if you assume more about the target space, but we have to
  -- work hard to prove it under the very minimal assumptions here.
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Int → M
    hf : HasProd f m
    ⊢ HasProd (fun n => HMul.hMul (f ↑n) (f (Neg.neg ↑n))) (HMul.hMul m (f 0))
  -/
  apply (hf.mul (hasProd_ite_eq (0 : ℤ) (f 0))).hasProd_of_prod_eq fun u ↦ ?_
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Int → M
    hf : HasProd f m
    u : Finset Int
    ⊢ Exists fun v => ∀ (v' : Finset Nat), HasSubset.Subset v v' → Exists fun u' = …
  -/
  refine ⟨u.image Int.natAbs, fun v' hv' ↦ ?_⟩
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Int → M
    hf : HasProd f m
    u : Finset Int
    v' : Finset Nat
    hv' : HasSubset.Subset (Finset.image Int.natAbs u) v'
    ⊢ Exists fun u' => And (HasSubset.Subset u u') (Eq (u'.prod fun x => HMul.hMul …
  -/
  let u1 := v'.image fun x : ℕ ↦ (x : ℤ)
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    m : M
    inst✝ : ContinuousMul M
    f : Int → M
    hf : HasProd f m
    u : Finset Int
    v' : Finset Nat
    hv' : HasSubset.Subset (Finset.image Int.natAbs u) v'
    u1 : Finset Int := Finset.image (fun x => ↑x) v'
    ⊢ Exists fun u' => And (HasSubset.Subset u u') (Eq (u'.prod fun x => HMul.hMul …
  -/
  let u2 := v'.image fun x : ℕ ↦ -(x : ℤ)
  have A : u ⊆ u1 ∪ u2 := by
    intro x hx
    simp only [u1, u2, mem_union, mem_image, exists_prop]
    rcases le_total 0 x with (h'x | h'x)
    · refine Or.inl ⟨_, hv' <| mem_image.mpr ⟨x, hx, rfl⟩, ?_⟩
      simp only [Int.natCast_natAbs, abs_eq_self, h'x]
    · refine Or.inr ⟨_, hv' <| mem_image.mpr ⟨x, hx, rfl⟩, ?_⟩
      simp only [abs_of_nonpos h'x, Int.natCast_natAbs, neg_neg]
  exact ⟨_, A, calc
    (∏ x ∈ u1 ∪ u2, (f x * if x = 0 then f 0 else 1)) =
        (∏ x ∈ u1 ∪ u2, f x) * ∏ x ∈ u1 ∩ u2, f x := by
      rw [prod_mul_distrib]
      congr 1
      refine (prod_subset_one_on_sdiff inter_subset_union ?_ ?_).symm
      · intro x hx
        suffices x ≠ 0 by simp only [this, if_false]
        rintro rfl
        simp only [mem_sdiff, mem_union, mem_image, Nat.cast_eq_zero, exists_eq_right, neg_eq_zero,
          or_self, mem_inter, and_self, and_not_self, u1, u2] at hx
      · intro x hx
        simp only [u1, u2, mem_inter, mem_image, exists_prop] at hx
        suffices x = 0 by simp only [this, eq_self_iff_true, if_true]
        apply le_antisymm
        · rcases hx.2 with ⟨a, _, rfl⟩
          simp only [Right.neg_nonpos_iff, Nat.cast_nonneg]
        · rcases hx.1 with ⟨a, _, rfl⟩
          simp only [Nat.cast_nonneg]
    _ = (∏ x ∈ u1, f x) * ∏ x ∈ u2, f x := prod_union_inter
    _ = (∏ b ∈ v', f b) * ∏ b ∈ v', f (-b) := by
      simp only [u1, u2, Nat.cast_inj, imp_self, implies_true, forall_const, prod_image, neg_inj]
    _ = ∏ b ∈ v', (f b * f (-b)) := prod_mul_distrib.symm⟩


@[deprecated HasSum.nat_add_neg (since := "2024-03-04")]
alias HasSum.sum_nat_of_sum_int := HasSum.nat_add_neg


@[to_additive]
theorem Multipliable.nat_mul_neg {f : ℤ → M} (hf : Multipliable f) :
    Multipliable fun n : ℕ ↦ f n * f (-n) :=
  hf.hasProd.nat_mul_neg.multipliable


@[to_additive]
lemma tprod_nat_mul_neg [T2Space M] {f : ℤ → M} (hf : Multipliable f) :
    ∏' n : ℕ, (f n * f (-n)) = (∏' n : ℤ, f n) * f 0 :=
  hf.hasProd.nat_mul_neg.tprod_eq


@[to_additive HasSum.of_add_one_of_neg_add_one]
theorem HasProd.of_add_one_of_neg_add_one {f : ℤ → M}
    (hf₁ : HasProd (fun n : ℕ ↦ f (n + 1)) m) (hf₂ : HasProd (fun n : ℕ ↦ f (-(n + 1))) m') :
    HasProd f (m * f 0 * m') :=
  HasProd.of_nat_of_neg_add_one (mul_comm _ m ▸ HasProd.zero_mul hf₁) hf₂


@[deprecated HasSum.of_add_one_of_neg_add_one (since := "2024-03-04")]
alias HasSum.pos_add_zero_add_neg := HasSum.of_add_one_of_neg_add_one


@[to_additive Summable.of_add_one_of_neg_add_one]
lemma Multipliable.of_add_one_of_neg_add_one {f : ℤ → M}
    (hf₁ : Multipliable fun n : ℕ ↦ f (n + 1)) (hf₂ : Multipliable fun n : ℕ ↦ f (-(n + 1))) :
    Multipliable f :=
  (hf₁.hasProd.of_add_one_of_neg_add_one hf₂.hasProd).multipliable


@[to_additive tsum_of_add_one_of_neg_add_one]
lemma tprod_of_add_one_of_neg_add_one [T2Space M] {f : ℤ → M}
    (hf₁ : Multipliable fun n : ℕ ↦ f (n + 1)) (hf₂ : Multipliable fun n : ℕ ↦ f (-(n + 1))) :
    ∏' n : ℤ, f n = (∏' n : ℕ, f (n + 1)) * f 0 * ∏' n : ℕ, f (-(n + 1)) :=
  (hf₁.hasProd.of_add_one_of_neg_add_one hf₂.hasProd).tprod_eq


@[to_additive]
lemma HasProd.of_nat_of_neg {f : ℤ → G} (hf₁ : HasProd (fun n : ℕ ↦ f n) g)
    (hf₂ : HasProd (fun n : ℕ ↦ f (-n)) g') : HasProd f (g * g' / f 0) := by
  /-
    G : Type u_2
    inst✝² : CommGroup G
    g g' : G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Int → G
    hf₁ : HasProd (fun n => f ↑n) g
    hf₂ : HasProd (fun n => f (Neg.neg ↑n)) g'
    ⊢ HasProd f (HDiv.hDiv (HMul.hMul g g') (f 0))
  -/
  refine mul_div_assoc' g .. ▸ hf₁.of_nat_of_neg_add_one (m' := g' / f 0) ?_
  /-
    G : Type u_2
    inst✝² : CommGroup G
    g g' : G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    f : Int → G
    hf₁ : HasProd (fun n => f ↑n) g
    hf₂ : HasProd (fun n => f (Neg.neg ↑n)) g'
    ⊢ HasProd (fun n => f (Neg.neg (HAdd.hAdd (↑n) 1))) (HDiv.hDiv g' (f 0))
  -/
  rwa [← hasProd_nat_add_iff' 1, prod_range_one, Nat.cast_zero, neg_zero] at hf₂
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Multipliable.of_nat_of_neg {f : ℤ → G} (hf₁ : Multipliable fun n : ℕ ↦ f n)
    (hf₂ : Multipliable fun n : ℕ ↦ f (-n)) : Multipliable f :=
  (hf₁.hasProd.of_nat_of_neg hf₂.hasProd).multipliable


@[deprecated Summable.of_nat_of_neg (since := "2024-03-04")]
alias summable_int_of_summable_nat := Summable.of_nat_of_neg


@[to_additive]
lemma tprod_of_nat_of_neg [T2Space G] {f : ℤ → G}
    (hf₁ : Multipliable fun n : ℕ ↦ f n) (hf₂ : Multipliable fun n : ℕ ↦ f (-n)) :
    ∏' n : ℤ, f n = (∏' n : ℕ, f n) * (∏' n : ℕ, f (-n)) / f 0 :=
  (hf₁.hasProd.of_nat_of_neg hf₂.hasProd).tprod_eq


/-- "iff" version of `Multipliable.of_nat_of_neg_add_one`. -/
@[to_additive "\"iff\" version of `Summable.of_nat_of_neg_add_one`."]
lemma multipliable_int_iff_multipliable_nat_and_neg_add_one {f : ℤ → G} : Multipliable f ↔
    (Multipliable fun n : ℕ ↦ f n) ∧ (Multipliable fun n : ℕ ↦ f (-(n + 1))) := by
  /-
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Int → G
    ⊢ Iff (Multipliable f) (And (Multipliable fun n => f ↑n) (Multipliable fun n = …
  -/
  refine ⟨fun p ↦ ⟨?_, ?_⟩, fun ⟨hf₁, hf₂⟩ ↦ Multipliable.of_nat_of_neg_add_one hf₁ hf₂⟩ <;>
  /-
    case refine_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Int → G
    p : Multipliable f
    ⊢ Multipliable fun n => f ↑n
  -/
  apply p.comp_injective
  /-
    case refine_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Int → G
    p : Multipliable f
    ⊢ Function.Injective Nat.cast
  -/
  exacts [Nat.cast_injective, @Int.negSucc.inj]
  /-
    🎉 no goals
  -/


/-- "iff" version of `Multipliable.of_nat_of_neg`. -/
@[to_additive "\"iff\" version of `Summable.of_nat_of_neg`."]
lemma multipliable_int_iff_multipliable_nat_and_neg {f : ℤ → G} :
    Multipliable f ↔ (Multipliable fun n : ℕ ↦ f n) ∧ (Multipliable fun n : ℕ ↦ f (-n)) := by
  /-
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Int → G
    ⊢ Iff (Multipliable f) (And (Multipliable fun n => f ↑n) (Multipliable fun n = …
  -/
  refine ⟨fun p ↦ ⟨?_, ?_⟩, fun ⟨hf₁, hf₂⟩ ↦ Multipliable.of_nat_of_neg hf₁ hf₂⟩ <;>
  /-
    case refine_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Int → G
    p : Multipliable f
    ⊢ Multipliable fun n => f ↑n
  -/
  apply p.comp_injective
  /-
    case refine_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : CompleteSpace G
    f : Int → G
    p : Multipliable f
    ⊢ Function.Injective Nat.cast
  -/
  exacts [Nat.cast_injective, neg_injective.comp Nat.cast_injective]
  /-
    🎉 no goals
  -/


