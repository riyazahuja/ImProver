/-- For a regular topological space with second countable topology,
there exists an inducing map to `l^∞ = ℕ →ᵇ ℝ`. -/
theorem exists_isInducing_l_infty : ∃ f : X → ℕ →ᵇ ℝ, IsInducing f := by
  -- Choose a countable basis, and consider the set `s` of pairs of set `(U, V)` such that `U ∈ B`,
  -- `V ∈ B`, and `closure U ⊆ V`.
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  rcases exists_countable_basis X with ⟨B, hBc, -, hB⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  let s : Set (Set X × Set X) := { UV ∈ B ×ˢ B | closure UV.1 ⊆ UV.2 }
  -- `s` is a countable set.
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  haveI : Encodable s := ((hBc.prod hBc).mono inter_subset_left).toEncodable
  -- We don't have the space of bounded (possibly discontinuous) functions, so we equip `s`
  -- with the discrete topology and deal with `s →ᵇ ℝ` instead.
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    this : Encodable ↑s
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  letI : TopologicalSpace s := ⊥
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    this✝ : Encodable ↑s
    this : TopologicalSpace ↑s := Bot.bot
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  haveI : DiscreteTopology s := ⟨rfl⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    this✝¹ : Encodable ↑s
    this✝ : TopologicalSpace ↑s := Bot.bot
    this : DiscreteTopology ↑s
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  rsuffices ⟨f, hf⟩ : ∃ f : X → s →ᵇ ℝ, IsInducing f
  · exact ⟨fun x => (f x).extend (Encodable.encode' s) 0,
      (BoundedContinuousFunction.isometry_extend (Encodable.encode' s)
        (0 : ℕ →ᵇ ℝ)).isEmbedding.isInducing.comp hf⟩
  have hd : ∀ UV : s, Disjoint (closure UV.1.1) UV.1.2ᶜ :=
    fun UV => disjoint_compl_right.mono_right (compl_subset_compl.2 UV.2.2)
  -- Choose a sequence of `εₙ > 0`, `n : s`, that is bounded above by `1` and tends to zero
  -- along the `cofinite` filter.
  obtain ⟨ε, ε01, hε⟩ : ∃ ε : s → ℝ, (∀ UV, ε UV ∈ Ioc (0 : ℝ) 1) ∧ Tendsto ε cofinite (𝓝 0) := by
    rcases posSumOfEncodable zero_lt_one s with ⟨ε, ε0, c, hεc, hc1⟩
    refine ⟨ε, fun UV => ⟨ε0 UV, ?_⟩, hεc.summable.tendsto_cofinite_zero⟩
    exact (le_hasSum hεc UV fun _ _ => (ε0 _).le).trans hc1
  /- For each `UV = (U, V) ∈ s` we use Urysohn's lemma to choose a function `f UV` that is equal to
    zero on `U` and is equal to `ε UV` on the complement to `V`. -/
  have : ∀ UV : s, ∃ f : C(X, ℝ),
      EqOn f 0 UV.1.1 ∧ EqOn f (fun _ => ε UV) UV.1.2ᶜ ∧ ∀ x, f x ∈ Icc 0 (ε UV) := by
    intro UV
    rcases exists_continuous_zero_one_of_isClosed isClosed_closure
        (hB.isOpen UV.2.1.2).isClosed_compl (hd UV) with
      ⟨f, hf₀, hf₁, hf01⟩
    exact ⟨ε UV • f, fun x hx => by simp [hf₀ (subset_closure hx)], fun x hx => by simp [hf₁ hx],
      fun x => ⟨mul_nonneg (ε01 _).1.le (hf01 _).1, mul_le_of_le_one_right (ε01 _).1.le (hf01 _).2⟩⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    this✝² : Encodable ↑s
    this✝¹ : TopologicalSpace ↑s := Bot.bot
    this✝ : DiscreteTopology ↑s
    hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
    ε : ↑s → Real
    ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
    hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
    this : ∀ (UV : ↑s), Exists fun f => And (Set.EqOn (⇑f) 0 (↑UV).1) (And (Set.Eq …
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  choose f hf0 hfε hf0ε using this
  have hf01 : ∀ UV x, f UV x ∈ Icc (0 : ℝ) 1 :=
    fun UV x => Icc_subset_Icc_right (ε01 _).2 (hf0ε _ _)
  -- The embedding is given by `F x UV = f UV x`.
  set F : X → s →ᵇ ℝ := fun x =>
    ⟨⟨fun UV => f UV x, continuous_of_discreteTopology⟩, 1,
      fun UV₁ UV₂ => Real.dist_le_of_mem_Icc_01 (hf01 _ _) (hf01 _ _)⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    this✝¹ : Encodable ↑s
    this✝ : TopologicalSpace ↑s := Bot.bot
    this : DiscreteTopology ↑s
    hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
    ε : ↑s → Real
    ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
    hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
    f : ↑s → ContinuousMap X Real
    hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
    hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
    hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
    hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
    F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  have hF : ∀ x UV, F x UV = f UV x := fun _ _ => rfl
  /-
    case intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : SecondCountableTopology X
    B : Set (Set X)
    hBc : B.Countable
    hB : TopologicalSpace.IsTopologicalBasis B
    s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
    this✝¹ : Encodable ↑s
    this✝ : TopologicalSpace ↑s := Bot.bot
    this : DiscreteTopology ↑s
    hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
    ε : ↑s → Real
    ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
    hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
    f : ↑s → ContinuousMap X Real
    hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
    hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
    hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
    hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
    F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
    hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
    ⊢ Exists fun f => Topology.IsInducing f
  -/
  refine ⟨F, isInducing_iff_nhds.2 fun x => le_antisymm ?_ ?_⟩
  · /- First we prove that `F` is continuous. Given `δ > 0`, consider the set `T` of `(U, V) ∈ s`
    such that `ε (U, V) ≥ δ`. Since `ε` tends to zero, `T` is finite. Since each `f` is continuous,
    we can choose a neighborhood such that `dist (F y (U, V)) (F x (U, V)) ≤ δ` for any
    `(U, V) ∈ T`. For `(U, V) ∉ T`, the same inequality is true because both `F y (U, V)` and
    `F x (U, V)` belong to the interval `[0, ε (U, V)]`. -/
    /-
      case intro.intro.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      ⊢ LE.le (nhds x) (Filter.comap F (nhds (F x)))
    -/
    refine (nhds_basis_closedBall.comap _).ge_iff.2 fun δ δ0 => ?_
    /-
      case intro.intro.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      δ : Real
      δ0 : LT.lt 0 δ
      ⊢ Membership.mem (nhds x) (Set.preimage F (Metric.closedBall (F x) δ))
    -/
    have h_fin : { UV : s | δ ≤ ε UV }.Finite := by simpa only [← not_lt] using hε (gt_mem_nhds δ0)
    have : ∀ᶠ y in 𝓝 x, ∀ UV, δ ≤ ε UV → dist (F y UV) (F x UV) ≤ δ := by
      refine (eventually_all_finite h_fin).2 fun UV _ => ?_
      exact (f UV).continuous.tendsto x (closedBall_mem_nhds _ δ0)
    /-
      case intro.intro.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝² : Encodable ↑s
      this✝¹ : TopologicalSpace ↑s := Bot.bot
      this✝ : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      δ : Real
      δ0 : LT.lt 0 δ
      h_fin : (setOf fun UV => LE.le δ (ε UV)).Finite
      this : Filter.Eventually (fun y => ∀ (UV : ↑s), LE.le δ (ε UV) → LE.le (Dist.d …
      ⊢ Membership.mem (nhds x) (Set.preimage F (Metric.closedBall (F x) δ))
    -/
    refine this.mono fun y hy => (BoundedContinuousFunction.dist_le δ0.le).2 fun UV => ?_
    /-
      case intro.intro.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝² : Encodable ↑s
      this✝¹ : TopologicalSpace ↑s := Bot.bot
      this✝ : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      δ : Real
      δ0 : LT.lt 0 δ
      h_fin : (setOf fun UV => LE.le δ (ε UV)).Finite
      this : Filter.Eventually (fun y => ∀ (UV : ↑s), LE.le δ (ε UV) → LE.le (Dist.d …
      y : X
      hy : ∀ (UV : ↑s), LE.le δ (ε UV) → LE.le (Dist.dist ((F y) UV) ((F x) UV)) δ
      UV : ↑s
      ⊢ LE.le (Dist.dist ((F y) UV) ((F x) UV)) δ
    -/
    rcases le_total δ (ε UV) with hle | hle
    /-
      case intro.intro.refine_1.inl
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝² : Encodable ↑s
      this✝¹ : TopologicalSpace ↑s := Bot.bot
      this✝ : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      δ : Real
      δ0 : LT.lt 0 δ
      h_fin : (setOf fun UV => LE.le δ (ε UV)).Finite
      this : Filter.Eventually (fun y => ∀ (UV : ↑s), LE.le δ (ε UV) → LE.le (Dist.d …
      y : X
      hy : ∀ (UV : ↑s), LE.le δ (ε UV) → LE.le (Dist.dist ((F y) UV) ((F x) UV)) δ
      UV : ↑s
      hle : LE.le δ (ε UV)
      ⊢ LE.le (Dist.dist ((F y) UV) ((F x) UV)) δ
    -/
    exacts [hy _ hle, (Real.dist_le_of_mem_Icc (hf0ε _ _) (hf0ε _ _)).trans (by rwa [sub_zero])]
    /-
      🎉 no goals
    -/
  · /- Finally, we prove that each neighborhood `V` of `x : X`
    includes a preimage of a neighborhood of `F x` under `F`.
    Without loss of generality, `V` belongs to `B`.
    Choose `U ∈ B` such that `x ∈ V` and `closure V ⊆ U`.
    Then the preimage of the `(ε (U, V))`-neighborhood of `F x` is included by `V`. -/
    /-
      case intro.intro.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      ⊢ LE.le (Filter.comap F (nhds (F x))) (nhds x)
    -/
    refine ((nhds_basis_ball.comap _).le_basis_iff hB.nhds_hasBasis).2 ?_
    /-
      case intro.intro.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      ⊢ ∀ (i' : Set X), And (Membership.mem B i') (Membership.mem i' x) → Exists fun …
    -/
    rintro V ⟨hVB, hxV⟩
    /-
      case intro.intro.refine_2.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      V : Set X
      hVB : Membership.mem B V
      hxV : Membership.mem V x
      ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Set.preimage F (Metric.ba …
    -/
    rcases hB.exists_closure_subset (hB.mem_nhds hVB hxV) with ⟨U, hUB, hxU, hUV⟩
    /-
      case intro.intro.refine_2.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      V : Set X
      hVB : Membership.mem B V
      hxV : Membership.mem V x
      U : Set X
      hUB : Membership.mem B U
      hxU : Membership.mem U x
      hUV : HasSubset.Subset (closure U) V
      ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Set.preimage F (Metric.ba …
    -/
    set UV : ↥s := ⟨(U, V), ⟨hUB, hVB⟩, hUV⟩
    /-
      case intro.intro.refine_2.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      V : Set X
      hVB : Membership.mem B V
      hxV : Membership.mem V x
      U : Set X
      hUB : Membership.mem B U
      hxU : Membership.mem U x
      hUV : HasSubset.Subset (closure U) V
      UV : ↑s := ⟨{ fst := U, snd := V }, ⋯⟩
      ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Set.preimage F (Metric.ba …
    -/
    refine ⟨ε UV, (ε01 UV).1, fun y (hy : dist (F y) (F x) < ε UV) => ?_⟩
    replace hy : dist (F y UV) (F x UV) < ε UV :=
      (BoundedContinuousFunction.dist_coe_le_dist _).trans_lt hy
    /-
      case intro.intro.refine_2.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      V : Set X
      hVB : Membership.mem B V
      hxV : Membership.mem V x
      U : Set X
      hUB : Membership.mem B U
      hxU : Membership.mem U x
      hUV : HasSubset.Subset (closure U) V
      UV : ↑s := ⟨{ fst := U, snd := V }, ⋯⟩
      y : X
      hy : LT.lt (Dist.dist ((F y) UV) ((F x) UV)) (ε UV)
      ⊢ Membership.mem V y
    -/
    contrapose! hy
    /-
      case intro.intro.refine_2.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      V : Set X
      hVB : Membership.mem B V
      hxV : Membership.mem V x
      U : Set X
      hUB : Membership.mem B U
      hxU : Membership.mem U x
      hUV : HasSubset.Subset (closure U) V
      UV : ↑s := ⟨{ fst := U, snd := V }, ⋯⟩
      y : X
      hy : Not (Membership.mem V y)
      ⊢ LE.le (ε UV) (Dist.dist ((F y) UV) ((F x) UV))
    -/
    rw [hF, hF, hfε UV hy, hf0 UV hxU, Pi.zero_apply, dist_zero_right]
    /-
      case intro.intro.refine_2.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : SecondCountableTopology X
      B : Set (Set X)
      hBc : B.Countable
      hB : TopologicalSpace.IsTopologicalBasis B
      s : Set (Prod (Set X) (Set X)) := setOf fun UV => And (Membership.mem (SProd.s …
      this✝¹ : Encodable ↑s
      this✝ : TopologicalSpace ↑s := Bot.bot
      this : DiscreteTopology ↑s
      hd : ∀ (UV : ↑s), Disjoint (closure (↑UV).1) (HasCompl.compl (↑UV).2)
      ε : ↑s → Real
      ε01 : ∀ (UV : ↑s), Membership.mem (Set.Ioc 0 1) (ε UV)
      hε : Filter.Tendsto ε Filter.cofinite (nhds 0)
      f : ↑s → ContinuousMap X Real
      hf0 : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) 0 (↑UV).1
      hfε : ∀ (UV : ↑s), Set.EqOn (⇑(f UV)) (fun x => ε UV) (HasCompl.compl (↑UV).2)
      hf0ε : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 (ε UV)) ((f UV) x)
      hf01 : ∀ (UV : ↑s) (x : X), Membership.mem (Set.Icc 0 1) ((f UV) x)
      F : X → BoundedContinuousFunction (↑s) Real := fun x => { toFun := fun UV => ( …
      hF : ∀ (x : X) (UV : ↑s), Eq ((F x) UV) ((f UV) x)
      x : X
      V : Set X
      hVB : Membership.mem B V
      hxV : Membership.mem V x
      U : Set X
      hUB : Membership.mem B U
      hxU : Membership.mem U x
      hUV : HasSubset.Subset (closure U) V
      UV : ↑s := ⟨{ fst := U, snd := V }, ⋯⟩
      y : X
      hy : Not (Membership.mem V y)
      ⊢ LE.le (ε UV) (Norm.norm ((fun x => ε UV) y))
    -/
    exact le_abs_self _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")] alias exists_inducing_l_infty := exists_isInducing_l_infty


/-- *Urysohn's metrization theorem* (Tychonoff's version):
a regular topological space with second countable topology `X` is metrizable,
i.e., there exists a pseudometric space structure that generates the same topology. -/
instance (priority := 90) PseudoMetrizableSpace.of_regularSpace_secondCountableTopology :
    PseudoMetrizableSpace X :=
  let ⟨_, hf⟩ := exists_isInducing_l_infty X
  hf.pseudoMetrizableSpace


/-- A T₃ topological space with second countable topology can be embedded into `l^∞ = ℕ →ᵇ ℝ`. -/
theorem exists_embedding_l_infty : ∃ f : X → ℕ →ᵇ ℝ, IsEmbedding f :=
  let ⟨f, hf⟩ := exists_isInducing_l_infty X; ⟨f, hf.isEmbedding⟩


/-- *Urysohn's metrization theorem* (Tychonoff's version): a T₃ topological space with second
countable topology `X` is metrizable, i.e., there exists a metric space structure that generates the
same topology. -/
instance (priority := 90) metrizableSpace_of_t3_secondCountable : MetrizableSpace X :=
  let ⟨_, hf⟩ := exists_embedding_l_infty X
  hf.metrizableSpace

-- The `alias` command creates a definition, triggering the defLemma linter.

@[nolint defLemma, deprecated (since := "2024-11-13")] alias
metrizableSpace_of_t3_second_countable := metrizableSpace_of_t3_secondCountable


