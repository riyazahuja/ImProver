/-- A metric space can be embedded in `l^∞(ℝ)` via the distances to points in
a fixed countable set, if this set is dense. This map is given in `kuratowskiEmbedding`,
without density assumptions. -/
def embeddingOfSubset : ℓ^∞(ℕ) :=
  ⟨fun n => dist a (x n) - dist (x 0) (x n), by
    /-
      α : Type u
      n : Nat
      inst✝ : MetricSpace α
      x : Nat → α
      a : α
      ⊢ Membership.mem (lp (fun i => Real) Top.top) fun n => HSub.hSub (Dist.dist a  …
    -/
    apply memℓp_infty
    /-
      case hf
      α : Type u
      n : Nat
      inst✝ : MetricSpace α
      x : Nat → α
      a : α
      ⊢ BddAbove (Set.range fun i => Norm.norm (HSub.hSub (Dist.dist a (x i)) (Dist. …
    -/
    use dist a (x 0)
    /-
      case h
      α : Type u
      n : Nat
      inst✝ : MetricSpace α
      x : Nat → α
      a : α
      ⊢ Membership.mem (upperBounds (Set.range fun i => Norm.norm (HSub.hSub (Dist.d …
    -/
    rintro - ⟨n, rfl⟩
    /-
      case h.intro
      α : Type u
      n✝ : Nat
      inst✝ : MetricSpace α
      x : Nat → α
      a : α
      n : Nat
      ⊢ LE.le ((fun i => Norm.norm (HSub.hSub (Dist.dist a (x i)) (Dist.dist (x 0) ( …
    -/
    exact abs_dist_sub_le _ _ _⟩
    /-
      🎉 no goals
    -/


theorem embeddingOfSubset_coe : embeddingOfSubset x a n = dist a (x n) - dist (x 0) (x n) :=
  rfl


/-- The embedding map is always a semi-contraction. -/
theorem embeddingOfSubset_dist_le (a b : α) :
    dist (embeddingOfSubset x a) (embeddingOfSubset x b) ≤ dist a b := by
  /-
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    a b : α
    ⊢ LE.le (Dist.dist (KuratowskiEmbedding.embeddingOfSubset x a) (KuratowskiEmbe …
  -/
  refine lp.norm_le_of_forall_le dist_nonneg fun n => ?_
  /-
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    a b : α
    n : Nat
    ⊢ LE.le (Norm.norm (↑(HSub.hSub (KuratowskiEmbedding.embeddingOfSubset x a) (K …
  -/
  simp only [lp.coeFn_sub, Pi.sub_apply, embeddingOfSubset_coe, Real.dist_eq]
  /-
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    a b : α
    n : Nat
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (Dist.dist a (x n)) (Dist.dist (x 0)  …
  -/
  convert abs_dist_sub_le a b (x n) using 2
  /-
    case h.e'_3.h.e'_1
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    a b : α
    n : Nat
    ⊢ Eq (HSub.hSub (HSub.hSub (Dist.dist a (x n)) (Dist.dist (x 0) (x n))) (HSub. …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- When the reference set is dense, the embedding map is an isometry on its image. -/
theorem embeddingOfSubset_isometry (H : DenseRange x) : Isometry (embeddingOfSubset x) := by
  /-
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    H : DenseRange x
    ⊢ Isometry (KuratowskiEmbedding.embeddingOfSubset x)
  -/
  refine Isometry.of_dist_eq fun a b => ?_
  /-
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    H : DenseRange x
    a b : α
    ⊢ Eq (Dist.dist (KuratowskiEmbedding.embeddingOfSubset x a) (KuratowskiEmbeddi …
  -/
  refine (embeddingOfSubset_dist_le x a b).antisymm (le_of_forall_pos_le_add fun e epos => ?_)
  -- First step: find n with dist a (x n) < e
  /-
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    H : DenseRange x
    a b : α
    e : Real
    epos : LT.lt 0 e
    ⊢ LE.le (Dist.dist a b) (HAdd.hAdd (Dist.dist (KuratowskiEmbedding.embeddingOf …
  -/
  rcases Metric.mem_closure_range_iff.1 (H a) (e / 2) (half_pos epos) with ⟨n, hn⟩
  -- Second step: use the norm control at index n to conclude
  have C : dist b (x n) - dist a (x n) = embeddingOfSubset x b n - embeddingOfSubset x a n := by
    simp only [embeddingOfSubset_coe, sub_sub_sub_cancel_right]
  have :=
    calc
      dist a b ≤ dist a (x n) + dist (x n) b := dist_triangle _ _ _
      _ = 2 * dist a (x n) + (dist b (x n) - dist a (x n)) := by simp [dist_comm]; ring
      _ ≤ 2 * dist a (x n) + |dist b (x n) - dist a (x n)| := by
        apply_rules [add_le_add_left, le_abs_self]
      _ ≤ 2 * (e / 2) + |embeddingOfSubset x b n - embeddingOfSubset x a n| := by
        rw [C]
        gcongr
      _ ≤ 2 * (e / 2) + dist (embeddingOfSubset x b) (embeddingOfSubset x a) := by
        gcongr
        simp only [dist_eq_norm]
        exact lp.norm_apply_le_norm ENNReal.top_ne_zero
          (embeddingOfSubset x b - embeddingOfSubset x a) n
      _ = dist (embeddingOfSubset x b) (embeddingOfSubset x a) + e := by ring
  /-
    case intro
    α : Type u
    inst✝ : MetricSpace α
    x : Nat → α
    H : DenseRange x
    a b : α
    e : Real
    epos : LT.lt 0 e
    n : Nat
    hn : LT.lt (Dist.dist a (x n)) (HDiv.hDiv e 2)
    C : Eq (HSub.hSub (Dist.dist b (x n)) (Dist.dist a (x n))) (HSub.hSub (↑(Kurat …
    this : LE.le (Dist.dist a b) (HAdd.hAdd (Dist.dist (KuratowskiEmbedding.embedd …
    ⊢ LE.le (Dist.dist a b) (HAdd.hAdd (Dist.dist (KuratowskiEmbedding.embeddingOf …
  -/
  simpa [dist_comm] using this
  /-
    🎉 no goals
  -/


/-- Every separable metric space embeds isometrically in `ℓ^∞(ℕ)`. -/
theorem exists_isometric_embedding (α : Type u) [MetricSpace α] [SeparableSpace α] :
    ∃ f : α → ℓ^∞(ℕ), Isometry f := by
  /-
    α : Type u
    inst✝¹ : MetricSpace α
    inst✝ : TopologicalSpace.SeparableSpace α
    ⊢ Exists fun f => Isometry f
  -/
  rcases (univ : Set α).eq_empty_or_nonempty with h | h
    /-
      case inl
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      h : Eq Set.univ EmptyCollection.emptyCollection
      ⊢ Exists fun f => Isometry f
    -/
  · use fun _ => 0; intro x; exact absurd h (Nonempty.ne_empty ⟨x, mem_univ x⟩)
                             /-
                               🎉 no goals
                             -/
  · -- We construct a map x : ℕ → α with dense image
    /-
      case inr
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      h : Set.univ.Nonempty
      ⊢ Exists fun f => Isometry f
    -/
    rcases h with ⟨basepoint⟩
    /-
      case inr.intro
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      basepoint : α
      h✝ : Membership.mem Set.univ basepoint
      ⊢ Exists fun f => Isometry f
    -/
    haveI : Inhabited α := ⟨basepoint⟩
    /-
      case inr.intro
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      basepoint : α
      h✝ : Membership.mem Set.univ basepoint
      this : Inhabited α
      ⊢ Exists fun f => Isometry f
    -/
    have : ∃ s : Set α, s.Countable ∧ Dense s := exists_countable_dense α
    /-
      case inr.intro
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      basepoint : α
      h✝ : Membership.mem Set.univ basepoint
      this✝ : Inhabited α
      this : Exists fun s => And s.Countable (Dense s)
      ⊢ Exists fun f => Isometry f
    -/
    rcases this with ⟨S, ⟨S_countable, S_dense⟩⟩
    /-
      case inr.intro.intro.intro
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      basepoint : α
      h✝ : Membership.mem Set.univ basepoint
      this : Inhabited α
      S : Set α
      S_countable : S.Countable
      S_dense : Dense S
      ⊢ Exists fun f => Isometry f
    -/
    rcases Set.countable_iff_exists_subset_range.1 S_countable with ⟨x, x_range⟩
    -- Use embeddingOfSubset to construct the desired isometry
    /-
      case inr.intro.intro.intro.intro
      α : Type u
      inst✝¹ : MetricSpace α
      inst✝ : TopologicalSpace.SeparableSpace α
      basepoint : α
      h✝ : Membership.mem Set.univ basepoint
      this : Inhabited α
      S : Set α
      S_countable : S.Countable
      S_dense : Dense S
      x : Nat → α
      x_range : HasSubset.Subset S (Set.range x)
      ⊢ Exists fun f => Isometry f
    -/
    exact ⟨embeddingOfSubset x, embeddingOfSubset_isometry x (S_dense.mono x_range)⟩
    /-
      🎉 no goals
    -/


/-- The Kuratowski embedding is an isometric embedding of a separable metric space in `ℓ^∞(ℕ, ℝ)`.
-/
def kuratowskiEmbedding (α : Type u) [MetricSpace α] [SeparableSpace α] : α → ℓ^∞(ℕ) :=
  Classical.choose (KuratowskiEmbedding.exists_isometric_embedding α)


/--
The Kuratowski embedding is an isometry.
Theorem 2.1 of [Assaf Naor, *Metric Embeddings and Lipschitz Extensions*][Naor-2015]. -/
protected theorem kuratowskiEmbedding.isometry (α : Type u) [MetricSpace α] [SeparableSpace α] :
    Isometry (kuratowskiEmbedding α) :=
  Classical.choose_spec (exists_isometric_embedding α)


/-- Version of the Kuratowski embedding for nonempty compacts -/
nonrec def NonemptyCompacts.kuratowskiEmbedding (α : Type u) [MetricSpace α] [CompactSpace α]
    [Nonempty α] : NonemptyCompacts ℓ^∞(ℕ) where
  carrier := range (kuratowskiEmbedding α)
  isCompact' := isCompact_range (kuratowskiEmbedding.isometry α).continuous
  nonempty' := range_nonempty _


/--
A function `f : α → ℓ^∞(ι, ℝ)` which is `K`-Lipschitz on a subset `s` admits a `K`-Lipschitz
extension to the whole space.

Theorem 2.2 of [Assaf Naor, *Metric Embeddings and Lipschitz Extensions*][Naor-2015]

The same result for the case of a finite type `ι` is implemented in
`LipschitzOnWith.extend_pi`.
-/
theorem LipschitzOnWith.extend_lp_infty [PseudoMetricSpace α] {s : Set α} {ι : Type*}
    {f : α → ℓ^∞(ι)} {K : ℝ≥0} (hfl : LipschitzOnWith K f s) :
    ∃ g : α → ℓ^∞(ι), LipschitzWith K g ∧ EqOn f g s := by
  -- Construct the coordinate-wise extensions
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    ι : Type u_1
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    K : NNReal
    hfl : LipschitzOnWith K f s
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  rw [LipschitzOnWith.coordinate] at hfl
  have (i : ι) : ∃ g : α → ℝ, LipschitzWith K g ∧ EqOn (fun x => f x i) g s :=
    LipschitzOnWith.extend_real (hfl i) -- use the nonlinear Hahn-Banach theorem here!
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    ι : Type u_1
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    K : NNReal
    hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
    this : ∀ (i : ι), Exists fun g => And (LipschitzWith K g) (Set.EqOn (fun x =>  …
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  choose g hgl hgeq using this
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    ι : Type u_1
    f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
    K : NNReal
    hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
    g : ι → α → Real
    hgl : ∀ (i : ι), LipschitzWith K (g i)
    hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  rcases s.eq_empty_or_nonempty with rfl | ⟨a₀, ha₀_in_s⟩
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      ι : Type u_1
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      K : NNReal
      g : ι → α → Real
      hgl : ∀ (i : ι), LipschitzWith K (g i)
      hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) EmptyCollection.emptyCo …
      hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) EmptyCollection.emptyColl …
      ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g EmptyCollection.emptyC …
    -/
  · exact ⟨0, LipschitzWith.const' 0, by simp⟩
    /-
      🎉 no goals
    -/
  · -- Show that the extensions are uniformly bounded
    have hf_extb : ∀ a : α, Memℓp (swap g a) ∞ := by
      apply LipschitzWith.uniformly_bounded (swap g) hgl a₀
      use ‖f a₀‖
      rintro - ⟨i, rfl⟩
      simp_rw [← hgeq i ha₀_in_s]
      exact lp.norm_apply_le_norm top_ne_zero (f a₀) i
    -- Construct witness by bundling the function with its certificate of membership in ℓ^∞
    /-
      case inr.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      ι : Type u_1
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      K : NNReal
      hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
      g : ι → α → Real
      hgl : ∀ (i : ι), LipschitzWith K (g i)
      hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
      a₀ : α
      ha₀_in_s : Membership.mem s a₀
      hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
      ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
    -/
    let f_ext' : α → ℓ^∞(ι) := fun i ↦ ⟨swap g i, hf_extb i⟩
    /-
      case inr.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      ι : Type u_1
      f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
      K : NNReal
      hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
      g : ι → α → Real
      hgl : ∀ (i : ι), LipschitzWith K (g i)
      hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
      a₀ : α
      ha₀_in_s : Membership.mem s a₀
      hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
      f_ext' : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := …
      ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
    -/
    refine ⟨f_ext', ?_, ?_⟩
      /-
        case inr.intro.refine_1
        α : Type u
        inst✝ : PseudoMetricSpace α
        s : Set α
        ι : Type u_1
        f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
        K : NNReal
        hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
        g : ι → α → Real
        hgl : ∀ (i : ι), LipschitzWith K (g i)
        hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
        a₀ : α
        ha₀_in_s : Membership.mem s a₀
        hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
        f_ext' : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := …
        ⊢ LipschitzWith K f_ext'
      -/
    · rw [LipschitzWith.coordinate]
      /-
        case inr.intro.refine_1
        α : Type u
        inst✝ : PseudoMetricSpace α
        s : Set α
        ι : Type u_1
        f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
        K : NNReal
        hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
        g : ι → α → Real
        hgl : ∀ (i : ι), LipschitzWith K (g i)
        hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
        a₀ : α
        ha₀_in_s : Membership.mem s a₀
        hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
        f_ext' : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := …
        ⊢ ∀ (i : ι), LipschitzWith K fun a => ↑(f_ext' a) i
      -/
      exact hgl
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.refine_2
        α : Type u
        inst✝ : PseudoMetricSpace α
        s : Set α
        ι : Type u_1
        f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
        K : NNReal
        hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
        g : ι → α → Real
        hgl : ∀ (i : ι), LipschitzWith K (g i)
        hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
        a₀ : α
        ha₀_in_s : Membership.mem s a₀
        hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
        f_ext' : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := …
        ⊢ Set.EqOn f f_ext' s
      -/
    · intro a hyp
      /-
        case inr.intro.refine_2
        α : Type u
        inst✝ : PseudoMetricSpace α
        s : Set α
        ι : Type u_1
        f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
        K : NNReal
        hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
        g : ι → α → Real
        hgl : ∀ (i : ι), LipschitzWith K (g i)
        hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
        a₀ : α
        ha₀_in_s : Membership.mem s a₀
        hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
        f_ext' : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := …
        a : α
        hyp : Membership.mem s a
        ⊢ Eq (f a) (f_ext' a)
      -/
      ext i
      /-
        case inr.intro.refine_2.h.h
        α : Type u
        inst✝ : PseudoMetricSpace α
        s : Set α
        ι : Type u_1
        f : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x
        K : NNReal
        hfl : ∀ (i : ι), LipschitzOnWith K (fun a => ↑(f a) i) s
        g : ι → α → Real
        hgl : ∀ (i : ι), LipschitzWith K (g i)
        hgeq : ∀ (i : ι), Set.EqOn (fun x => ↑(f x) i) (g i) s
        a₀ : α
        ha₀_in_s : Membership.mem s a₀
        hf_extb : ∀ (a : α), Memℓp (Function.swap g a) Top.top
        f_ext' : α → Subtype fun x => Membership.mem (lp (fun i => Real) Top.top) x := …
        a : α
        hyp : Membership.mem s a
        i : ι
        ⊢ Eq (↑(f a) i) (↑(f_ext' a) i)
      -/
      exact (hgeq i) hyp
      /-
        🎉 no goals
      -/

