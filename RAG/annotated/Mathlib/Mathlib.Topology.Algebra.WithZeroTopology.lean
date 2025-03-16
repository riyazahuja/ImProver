/-- The topology on a linearly ordered commutative group with a zero element adjoined.
A subset U is open if 0 ∉ U or if there is an invertible element γ₀ such that {γ | γ < γ₀} ⊆ U. -/
scoped instance (priority := 100) topologicalSpace : TopologicalSpace Γ₀ :=
  nhdsAdjoint 0 <| ⨅ γ ≠ 0, 𝓟 (Iio γ)


theorem nhds_eq_update : (𝓝 : Γ₀ → Filter Γ₀) = update pure 0 (⨅ γ ≠ 0, 𝓟 (Iio γ)) := by
   /-
     Γ₀ : Type u_2
     inst✝ : LinearOrderedCommGroupWithZero Γ₀
     ⊢ Eq nhds (Function.update Pure.pure 0 (iInf fun γ => iInf fun h => Filter.pri …
   -/
   rw [nhds_nhdsAdjoint, sup_of_le_right]
   /-
     Γ₀ : Type u_2
     inst✝ : LinearOrderedCommGroupWithZero Γ₀
     ⊢ LE.le (Pure.pure 0) (iInf fun γ => iInf fun h => Filter.principal (Set.Iio γ))
   -/
   exact le_iInf₂ fun γ hγ ↦ le_principal_iff.2 <| zero_lt_iff.2 hγ
   /-
     🎉 no goals
   -/


theorem nhds_zero : 𝓝 (0 : Γ₀) = ⨅ γ ≠ 0, 𝓟 (Iio γ) := by
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    ⊢ Eq (nhds 0) (iInf fun γ => iInf fun h => Filter.principal (Set.Iio γ))
  -/
  rw [nhds_eq_update, update_self]
  /-
    🎉 no goals
  -/


/-- In a linearly ordered group with zero element adjoined, `U` is a neighbourhood of `0` if and
only if there exists a nonzero element `γ₀` such that `Iio γ₀ ⊆ U`. -/
theorem hasBasis_nhds_zero : (𝓝 (0 : Γ₀)).HasBasis (fun γ : Γ₀ => γ ≠ 0) Iio := by
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    ⊢ (nhds 0).HasBasis (fun γ => Ne γ 0) Set.Iio
  -/
  rw [nhds_zero]
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    ⊢ (iInf fun γ => iInf fun h => Filter.principal (Set.Iio γ)).HasBasis (fun γ = …
  -/
  refine hasBasis_biInf_principal ?_ ⟨1, one_ne_zero⟩
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    ⊢ DirectedOn (Order.Preimage Set.Iio fun x1 x2 => GE.ge x1 x2) fun γ => Eq γ 0 …
  -/
  exact directedOn_iff_directed.2 (Monotone.directed_ge fun a b hab => Iio_subset_Iio hab)
  /-
    🎉 no goals
  -/


theorem Iio_mem_nhds_zero (hγ : γ ≠ 0) : Iio γ ∈ 𝓝 (0 : Γ₀) :=
  hasBasis_nhds_zero.mem_of_mem hγ


/-- If `γ` is an invertible element of a linearly ordered group with zero element adjoined, then
`Iio (γ : Γ₀)` is a neighbourhood of `0`. -/
theorem nhds_zero_of_units (γ : Γ₀ˣ) : Iio ↑γ ∈ 𝓝 (0 : Γ₀) :=
  Iio_mem_nhds_zero γ.ne_zero


theorem tendsto_zero : Tendsto f l (𝓝 (0 : Γ₀)) ↔ ∀ (γ₀) (_ : γ₀ ≠ 0), ∀ᶠ x in l, f x < γ₀ := by
  /-
    α : Type u_1
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    l : Filter α
    f : α → Γ₀
    ⊢ Iff (Filter.Tendsto f l (nhds 0)) (∀ (γ₀ : Γ₀), Ne γ₀ 0 → Filter.Eventually  …
  -/
  simp [nhds_zero]
  /-
    🎉 no goals
  -/


/-- The neighbourhood filter of a nonzero element consists of all sets containing that
element. -/
@[simp]
theorem nhds_of_ne_zero {γ : Γ₀} (h₀ : γ ≠ 0) : 𝓝 γ = pure γ :=
  nhds_nhdsAdjoint_of_ne _ h₀


/-- The neighbourhood filter of an invertible element consists of all sets containing that
element. -/
theorem nhds_coe_units (γ : Γ₀ˣ) : 𝓝 (γ : Γ₀) = pure (γ : Γ₀) :=
  nhds_of_ne_zero γ.ne_zero


/-- If `γ` is an invertible element of a linearly ordered group with zero element adjoined, then
`{γ}` is a neighbourhood of `γ`. -/
                                                                                   /-
                                                                                     Γ₀ : Type u_2
                                                                                     inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                                                                     γ : Units Γ₀
                                                                                     ⊢ Membership.mem (nhds ↑γ) (Singleton.singleton ↑γ)
                                                                                   -/
theorem singleton_mem_nhds_of_units (γ : Γ₀ˣ) : ({↑γ} : Set Γ₀) ∈ 𝓝 (γ : Γ₀) := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- If `γ` is a nonzero element of a linearly ordered group with zero element adjoined, then `{γ}`
is a neighbourhood of `γ`. -/
                                                                                      /-
                                                                                        Γ₀ : Type u_2
                                                                                        inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                                                                        γ : Γ₀
                                                                                        h : Ne γ 0
                                                                                        ⊢ Membership.mem (nhds γ) (Singleton.singleton γ)
                                                                                      -/
theorem singleton_mem_nhds_of_ne_zero (h : γ ≠ 0) : ({γ} : Set Γ₀) ∈ 𝓝 (γ : Γ₀) := by simp [h]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem hasBasis_nhds_of_ne_zero {x : Γ₀} (h : x ≠ 0) :
    HasBasis (𝓝 x) (fun _ : Unit => True) fun _ => {x} := by
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    x : Γ₀
    h : Ne x 0
    ⊢ (nhds x).HasBasis (fun x => True) fun x_1 => Singleton.singleton x
  -/
  rw [nhds_of_ne_zero h]
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    x : Γ₀
    h : Ne x 0
    ⊢ (Pure.pure x).HasBasis (fun x => True) fun x_1 => Singleton.singleton x
  -/
  exact hasBasis_pure _
  /-
    🎉 no goals
  -/


theorem hasBasis_nhds_units (γ : Γ₀ˣ) :
    HasBasis (𝓝 (γ : Γ₀)) (fun _ : Unit => True) fun _ => {↑γ} :=
  hasBasis_nhds_of_ne_zero γ.ne_zero


theorem tendsto_of_ne_zero {γ : Γ₀} (h : γ ≠ 0) : Tendsto f l (𝓝 γ) ↔ ∀ᶠ x in l, f x = γ := by
  /-
    α : Type u_1
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    l : Filter α
    f : α → Γ₀
    γ : Γ₀
    h : Ne γ 0
    ⊢ Iff (Filter.Tendsto f l (nhds γ)) (Filter.Eventually (fun x => Eq (f x) γ) l)
  -/
  rw [nhds_of_ne_zero h, tendsto_pure]
  /-
    🎉 no goals
  -/


theorem tendsto_units {γ₀ : Γ₀ˣ} : Tendsto f l (𝓝 (γ₀ : Γ₀)) ↔ ∀ᶠ x in l, f x = γ₀ :=
  tendsto_of_ne_zero γ₀.ne_zero


theorem Iio_mem_nhds (h : γ₁ < γ₂) : Iio γ₂ ∈ 𝓝 γ₁ := by
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    γ₁ γ₂ : Γ₀
    h : LT.lt γ₁ γ₂
    ⊢ Membership.mem (nhds γ₁) (Set.Iio γ₂)
  -/
                                           /-
                                             🎉 no goals
                                           -/
  rcases eq_or_ne γ₁ 0 with (rfl | h₀) <;> simp [*, h.ne', Iio_mem_nhds_zero]
                                           /-
                                             🎉 no goals
                                           -/


theorem isOpen_iff {s : Set Γ₀} : IsOpen s ↔ (0 : Γ₀) ∉ s ∨ ∃ γ, γ ≠ 0 ∧ Iio γ ⊆ s := by
  /-
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    s : Set Γ₀
    ⊢ Iff (IsOpen s) (Or (Not (Membership.mem s 0)) (Exists fun γ => And (Ne γ 0)  …
  -/
  rw [isOpen_iff_mem_nhds, ← and_forall_ne (0 : Γ₀)]
  simp +contextual [nhds_of_ne_zero, imp_iff_not_or,
    hasBasis_nhds_zero.mem_iff]


theorem isClosed_iff {s : Set Γ₀} : IsClosed s ↔ (0 : Γ₀) ∈ s ∨ ∃ γ, γ ≠ 0 ∧ s ⊆ Ici γ := by
  simp only [← isOpen_compl_iff, isOpen_iff, mem_compl_iff, not_not, ← compl_Ici,
    compl_subset_compl]


theorem isOpen_Iio {a : Γ₀} : IsOpen (Iio a) :=
  isOpen_iff.mpr <| imp_iff_not_or.mp fun ha => ⟨a, ne_of_gt ha, Subset.rfl⟩


/-- The topology on a linearly ordered group with zero element adjoined is compatible with the order
structure: the set `{p : Γ₀ × Γ₀ | p.1 ≤ p.2}` is closed. -/
@[nolint defLemma]
scoped instance (priority := 100) orderClosedTopology : OrderClosedTopology Γ₀ where
  isClosed_le' := by
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      ⊢ IsClosed (setOf fun p => LE.le p.1 p.2)
    -/
    simp only [← isOpen_compl_iff, compl_setOf, not_le, isOpen_iff_mem_nhds]
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      ⊢ ∀ (x : Prod Γ₀ Γ₀), Membership.mem (setOf fun a => LT.lt a.2 a.1) x → Member …
    -/
    rintro ⟨a, b⟩ (hab : b < a)
    /-
      case mk
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      a b : Γ₀
      hab : LT.lt b a
      ⊢ Membership.mem (nhds { fst := a, snd := b }) (setOf fun a => LT.lt a.2 a.1)
    -/
    rw [nhds_prod_eq, nhds_of_ne_zero (zero_le'.trans_lt hab).ne', pure_prod]
    /-
      case mk
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      a b : Γ₀
      hab : LT.lt b a
      ⊢ Membership.mem (Filter.map (Prod.mk a) (nhds b)) (setOf fun a => LT.lt a.2 a …
    -/
    exact Iio_mem_nhds hab
    /-
      🎉 no goals
    -/


/-- The topology on a linearly ordered group with zero element adjoined is T₅. -/
@[nolint defLemma]
scoped instance (priority := 100) t5Space : T5Space Γ₀ where
  completely_normal := fun s t h₁ h₂ => by
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      s t : Set Γ₀
      h₁ : Disjoint (closure s) t
      h₂ : Disjoint s (closure t)
      ⊢ Disjoint (nhdsSet s) (nhdsSet t)
    -/
    by_cases hs : 0 ∈ s
      /-
        case pos
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        s t : Set Γ₀
        h₁ : Disjoint (closure s) t
        h₂ : Disjoint s (closure t)
        hs : Membership.mem s 0
        ⊢ Disjoint (nhdsSet s) (nhdsSet t)
      -/
    · have ht : 0 ∉ t := fun ht => disjoint_left.1 h₁ (subset_closure hs) ht
      /-
        case pos
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        s t : Set Γ₀
        h₁ : Disjoint (closure s) t
        h₂ : Disjoint s (closure t)
        hs : Membership.mem s 0
        ht : Not (Membership.mem t 0)
        ⊢ Disjoint (nhdsSet s) (nhdsSet t)
      -/
      rwa [(isOpen_iff.2 (.inl ht)).nhdsSet_eq, disjoint_nhdsSet_principal]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        s t : Set Γ₀
        h₁ : Disjoint (closure s) t
        h₂ : Disjoint s (closure t)
        hs : Not (Membership.mem s 0)
        ⊢ Disjoint (nhdsSet s) (nhdsSet t)
      -/
    · rwa [(isOpen_iff.2 (.inl hs)).nhdsSet_eq, disjoint_principal_nhdsSet]
      /-
        🎉 no goals
      -/


/-- The topology on a linearly ordered group with zero element adjoined makes it a topological
monoid. -/
@[nolint defLemma]
scoped instance (priority := 100) : ContinuousMul Γ₀ where
  continuous_mul := by
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    simp only [continuous_iff_continuousAt, ContinuousAt]
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      ⊢ ∀ (x : Prod Γ₀ Γ₀), Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds x) (nh …
    -/
    rintro ⟨x, y⟩
    /-
      case mk
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      x y : Γ₀
      ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := x, snd := y }) (n …
    -/
    wlog hle : x ≤ y generalizing x y
      /-
        case mk.inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        x y : Γ₀
        this : ∀ (x y : Γ₀), LE.le x y → Filter.Tendsto (fun p => HMul.hMul p.1 p.2) ( …
        hle : Not (LE.le x y)
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := x, snd := y }) (n …
      -/
    · have := (this y x (le_of_not_le hle)).comp (continuous_swap.tendsto (x, y))
      /-
        case mk.inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        x y : Γ₀
        this✝ : ∀ (x y : Γ₀), LE.le x y → Filter.Tendsto (fun p => HMul.hMul p.1 p.2)  …
        hle : Not (LE.le x y)
        this : Filter.Tendsto (Function.comp (fun p => HMul.hMul p.1 p.2) Prod.swap) ( …
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := x, snd := y }) (n …
      -/
      simpa only [mul_comm, Function.comp_def, Prod.swap] using this
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      x y : Γ₀
      hle : LE.le x y
      ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := x, snd := y }) (n …
    -/
    rcases eq_or_ne x 0 with (rfl | hx) <;> [rcases eq_or_ne y 0 with (rfl | hy); skip]
      /-
        case inl.inl
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        hle : LE.le 0 0
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := 0, snd := 0 }) (n …
      -/
    · rw [zero_mul]
      refine ((hasBasis_nhds_zero.prod_nhds hasBasis_nhds_zero).tendsto_iff hasBasis_nhds_zero).2
        fun γ hγ => ⟨(γ, 1), ⟨hγ, one_ne_zero⟩, ?_⟩
      /-
        case inl.inl
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ✝ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        hle : LE.le 0 0
        γ : Γ₀
        hγ : Ne γ 0
        ⊢ ∀ (x : Prod Γ₀ Γ₀), Membership.mem (SProd.sprod (Set.Iio { fst := γ, snd :=  …
      -/
      rintro ⟨x, y⟩ ⟨hx : x < γ, hy : y < 1⟩
      /-
        case inl.inl.mk.intro
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ✝ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        hle : LE.le 0 0
        γ : Γ₀
        hγ : Ne γ 0
        x y : Γ₀
        hx : LT.lt x γ
        hy : LT.lt y 1
        ⊢ Membership.mem (Set.Iio γ) (HMul.hMul { fst := x, snd := y }.1 { fst := x, s …
      -/
      exact (mul_lt_mul'' hx hy zero_le' zero_le').trans_eq (mul_one γ)
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        y : Γ₀
        hle : LE.le 0 y
        hy : Ne y 0
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := 0, snd := y }) (n …
      -/
    · rw [zero_mul, nhds_prod_eq, nhds_of_ne_zero hy, prod_pure, tendsto_map'_iff]
      /-
        case inl.inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        y : Γ₀
        hle : LE.le 0 y
        hy : Ne y 0
        ⊢ Filter.Tendsto (Function.comp (fun p => HMul.hMul p.1 p.2) fun a => { fst := …
      -/
      refine (hasBasis_nhds_zero.tendsto_iff hasBasis_nhds_zero).2 fun γ hγ => ?_
      /-
        case inl.inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ✝ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        y : Γ₀
        hle : LE.le 0 y
        hy : Ne y 0
        γ : Γ₀
        hγ : Ne γ 0
        ⊢ Exists fun ia => And (Ne ia 0) (∀ (x : Γ₀), Membership.mem (Set.Iio ia) x →  …
      -/
      refine ⟨γ / y, div_ne_zero hγ hy, fun x hx => ?_⟩
      calc x * y < γ / y * y := mul_lt_mul_of_pos_right hx (zero_lt_iff.2 hy)
      _ = γ := div_mul_cancel₀ _ hy
      /-
        case inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        x y : Γ₀
        hle : LE.le x y
        hx : Ne x 0
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := x, snd := y }) (n …
      -/
    · have hy : y ≠ 0 := ((zero_lt_iff.mpr hx).trans_le hle).ne'
      /-
        case inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        x y : Γ₀
        hle : LE.le x y
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := x, snd := y }) (n …
      -/
      rw [nhds_prod_eq, nhds_of_ne_zero hx, nhds_of_ne_zero hy, prod_pure_pure]
      /-
        case inr
        α : Type u_1
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        γ γ₁ γ₂ : Γ₀
        l : Filter α
        f : α → Γ₀
        x y : Γ₀
        hle : LE.le x y
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (Pure.pure { fst := x, snd := y  …
      -/
      exact pure_le_nhds (x * y)
      /-
        🎉 no goals
      -/


@[nolint defLemma]
scoped instance (priority := 100) : HasContinuousInv₀ Γ₀ :=
  ⟨fun γ h => by
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ✝ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      γ : Γ₀
      h : Ne γ 0
      ⊢ ContinuousAt Inv.inv γ
    -/
    rw [ContinuousAt, nhds_of_ne_zero h]
    /-
      α : Type u_1
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      γ✝ γ₁ γ₂ : Γ₀
      l : Filter α
      f : α → Γ₀
      γ : Γ₀
      h : Ne γ 0
      ⊢ Filter.Tendsto Inv.inv (Pure.pure γ) (nhds (Inv.inv γ))
    -/
    exact pure_le_nhds γ⁻¹⟩
    /-
      🎉 no goals
    -/


