/-- A very useful criterion to show that a space is complete is to show that all sequences
which satisfy a bound of the form `dist (u n) (u m) < B N` for all `n m ≥ N` are
converging. This is often applied for `B N = 2^{-N}`, i.e., with a very fast convergence to
`0`, which makes it possible to use arguments of converging series, while this is impossible
to do in general for arbitrary Cauchy sequences. -/
theorem Metric.complete_of_convergent_controlled_sequences (B : ℕ → Real) (hB : ∀ n, 0 < B n)
    (H : ∀ u : ℕ → α, (∀ N n m : ℕ, N ≤ n → N ≤ m → dist (u n) (u m) < B N) →
      ∃ x, Tendsto u atTop (𝓝 x)) :
    CompleteSpace α :=
  UniformSpace.complete_of_convergent_controlled_sequences
    (fun n => { p : α × α | dist p.1 p.2 < B n }) (fun n => dist_mem_uniformity <| hB n) H


/-- A pseudo-metric space is complete iff every Cauchy sequence converges. -/
theorem Metric.complete_of_cauchySeq_tendsto :
    (∀ u : ℕ → α, CauchySeq u → ∃ a, Tendsto u atTop (𝓝 a)) → CompleteSpace α :=
  EMetric.complete_of_cauchySeq_tendsto


/-- In a pseudometric space, Cauchy sequences are characterized by the fact that, eventually,
the distance between its elements is arbitrarily small -/
-- Porting note: @[nolint ge_or_gt] doesn't exist
theorem Metric.cauchySeq_iff {u : β → α} :
    CauchySeq u ↔ ∀ ε > 0, ∃ N, ∀ m ≥ N, ∀ n ≥ N, dist (u m) (u n) < ε :=
  uniformity_basis_dist.cauchySeq_iff


/-- A variation around the pseudometric characterization of Cauchy sequences -/
theorem Metric.cauchySeq_iff' {u : β → α} :
    CauchySeq u ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, dist (u n) (u N) < ε :=
  uniformity_basis_dist.cauchySeq_iff'

-- see Note [nolint_ge]

/-- In a pseudometric space, uniform Cauchy sequences are characterized by the fact that,
eventually, the distance between all its elements is uniformly, arbitrarily small. -/
-- Porting note: no attr @[nolint ge_or_gt]
theorem Metric.uniformCauchySeqOn_iff {γ : Type*} {F : β → γ → α} {s : Set γ} :
    UniformCauchySeqOn F atTop s ↔ ∀ ε > (0 : ℝ),
      ∃ N : β, ∀ m ≥ N, ∀ n ≥ N, ∀ x ∈ s, dist (F m x) (F n x) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝² : PseudoMetricSpace α
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    γ : Type u_3
    F : β → γ → α
    s : Set γ
    ⊢ Iff (UniformCauchySeqOn F Filter.atTop s) (∀ (ε : Real), GT.gt ε 0 → Exists  …
  -/
  constructor
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      ⊢ UniformCauchySeqOn F Filter.atTop s → ∀ (ε : Real), GT.gt ε 0 → Exists fun N …
    -/
  · intro h ε hε
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : UniformCauchySeqOn F Filter.atTop s
      ε : Real
      hε : GT.gt ε 0
      ⊢ Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Mem …
    -/
    let u := { a : α × α | dist a.fst a.snd < ε }
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : UniformCauchySeqOn F Filter.atTop s
      ε : Real
      hε : GT.gt ε 0
      u : Set (Prod α α) := setOf fun a => LT.lt (Dist.dist a.1 a.2) ε
      ⊢ Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Mem …
    -/
    have hu : u ∈ 𝓤 α := Metric.mem_uniformity_dist.mpr ⟨ε, hε, by simp [u]⟩
    rw [← Filter.eventually_atTop_prod_self' (p := fun m =>
      ∀ x ∈ s, dist (F m.fst x) (F m.snd x) < ε)]
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : UniformCauchySeqOn F Filter.atTop s
      ε : Real
      hε : GT.gt ε 0
      u : Set (Prod α α) := setOf fun a => LT.lt (Dist.dist a.1 a.2) ε
      hu : Membership.mem (uniformity α) u
      ⊢ Filter.Eventually (fun x => ∀ (x_1 : γ), Membership.mem s x_1 → LT.lt (Dist. …
    -/
    specialize h u hu
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      ε : Real
      hε : GT.gt ε 0
      u : Set (Prod α α) := setOf fun a => LT.lt (Dist.dist a.1 a.2) ε
      hu : Membership.mem (uniformity α) u
      h : Filter.Eventually (fun m => ∀ (x : γ), Membership.mem s x → Membership.mem …
      ⊢ Filter.Eventually (fun x => ∀ (x_1 : γ), Membership.mem s x_1 → LT.lt (Dist. …
    -/
    rw [prod_atTop_atTop_eq] at h
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      ε : Real
      hε : GT.gt ε 0
      u : Set (Prod α α) := setOf fun a => LT.lt (Dist.dist a.1 a.2) ε
      hu : Membership.mem (uniformity α) u
      h : Filter.Eventually (fun m => ∀ (x : γ), Membership.mem s x → Membership.mem …
      ⊢ Filter.Eventually (fun x => ∀ (x_1 : γ), Membership.mem s x_1 → LT.lt (Dist. …
    -/
    exact h.mono fun n h x hx => h x hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      ⊢ (∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β), …
    -/
  · intro h u hu
    /-
      case mpr
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ⊢ Filter.Eventually (fun m => ∀ (x : γ), Membership.mem s x → Membership.mem u …
    -/
    rcases Metric.mem_uniformity_dist.mp hu with ⟨ε, hε, hab⟩
    /-
      case mpr.intro.intro
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ε : Real
      hε : GT.gt ε 0
      hab : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd  …
      ⊢ Filter.Eventually (fun m => ∀ (x : γ), Membership.mem s x → Membership.mem u …
    -/
    rcases h ε hε with ⟨N, hN⟩
    /-
      case mpr.intro.intro.intro
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ε : Real
      hε : GT.gt ε 0
      hab : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd  …
      N : β
      hN : ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Membership.mem s …
      ⊢ Filter.Eventually (fun m => ∀ (x : γ), Membership.mem s x → Membership.mem u …
    -/
    rw [prod_atTop_atTop_eq, eventually_atTop]
    /-
      case mpr.intro.intro.intro
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ε : Real
      hε : GT.gt ε 0
      hab : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd  …
      N : β
      hN : ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Membership.mem s …
      ⊢ Exists fun a => ∀ (b : Prod β β), GE.ge b a → ∀ (x : γ), Membership.mem s x  …
    -/
    use (N, N)
    /-
      case h
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ε : Real
      hε : GT.gt ε 0
      hab : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd  …
      N : β
      hN : ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Membership.mem s …
      ⊢ ∀ (b : Prod β β), GE.ge b { fst := N, snd := N } → ∀ (x : γ), Membership.mem …
    -/
    intro b hb x hx
    /-
      case h
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ε : Real
      hε : GT.gt ε 0
      hab : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd  …
      N : β
      hN : ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Membership.mem s …
      b : Prod β β
      hb : GE.ge b { fst := N, snd := N }
      x : γ
      hx : Membership.mem s x
      ⊢ Membership.mem u { fst := F b.1 x, snd := F b.2 x }
    -/
    rcases hb with ⟨hbl, hbr⟩
    /-
      case h.intro
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      γ : Type u_3
      F : β → γ → α
      s : Set γ
      h : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : β), GE.ge m N → ∀ (n : β) …
      u : Set (Prod α α)
      hu : Membership.mem (uniformity α) u
      ε : Real
      hε : GT.gt ε 0
      hab : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem u { fst := a, snd  …
      N : β
      hN : ∀ (m : β), GE.ge m N → ∀ (n : β), GE.ge n N → ∀ (x : γ), Membership.mem s …
      b : Prod β β
      x : γ
      hx : Membership.mem s x
      hbl : LE.le { fst := N, snd := N }.1 b.1
      hbr : LE.le { fst := N, snd := N }.2 b.2
      ⊢ Membership.mem u { fst := F b.1 x, snd := F b.2 x }
    -/
    exact hab (hN b.fst hbl.ge b.snd hbr.ge x hx)
    /-
      🎉 no goals
    -/


/-- If the distance between `s n` and `s m`, `n ≤ m` is bounded above by `b n`
and `b` converges to zero, then `s` is a Cauchy sequence. -/
theorem cauchySeq_of_le_tendsto_0' {s : β → α} (b : β → ℝ)
    (h : ∀ n m : β, n ≤ m → dist (s n) (s m) ≤ b n) (h₀ : Tendsto b atTop (𝓝 0)) : CauchySeq s :=
  Metric.cauchySeq_iff'.2 fun ε ε0 => (h₀.eventually (gt_mem_nhds ε0)).exists.imp fun N hN n hn =>
    calc dist (s n) (s N) = dist (s N) (s n) := dist_comm _ _
    _ ≤ b N := h _ _ hn
    _ < ε := hN


/-- If the distance between `s n` and `s m`, `n, m ≥ N` is bounded above by `b N`
and `b` converges to zero, then `s` is a Cauchy sequence. -/
theorem cauchySeq_of_le_tendsto_0 {s : β → α} (b : β → ℝ)
    (h : ∀ n m N : β, N ≤ n → N ≤ m → dist (s n) (s m) ≤ b N) (h₀ : Tendsto b atTop (𝓝 0)) :
    CauchySeq s :=
  cauchySeq_of_le_tendsto_0' b (fun _n _m hnm => h _ _ _ le_rfl hnm) h₀


/-- A Cauchy sequence on the natural numbers is bounded. -/
theorem cauchySeq_bdd {u : ℕ → α} (hu : CauchySeq u) : ∃ R > 0, ∀ m n, dist (u m) (u n) < R := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (m n : Nat), LT.lt (Dist.dist (u m) (u n) …
  -/
  rcases Metric.cauchySeq_iff'.1 hu 1 zero_lt_one with ⟨N, hN⟩
  /-
    case intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (m n : Nat), LT.lt (Dist.dist (u m) (u n) …
  -/
  rsuffices ⟨R, R0, H⟩ : ∃ R > 0, ∀ n, dist (u n) (u N) < R
  · exact ⟨_, add_pos R0 R0, fun m n =>
      lt_of_le_of_lt (dist_triangle_right _ _ _) (add_lt_add (H m) (H n))⟩
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (n : Nat), LT.lt (Dist.dist (u n) (u N)) R)
  -/
  let R := Finset.sup (Finset.range N) fun n => nndist (u n) (u N)
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
    R : NNReal := (Finset.range N).sup fun n => NNDist.nndist (u n) (u N)
    ⊢ Exists fun R => And (GT.gt R 0) (∀ (n : Nat), LT.lt (Dist.dist (u n) (u N)) R)
  -/
  refine ⟨↑R + 1, add_pos_of_nonneg_of_pos R.2 zero_lt_one, fun n => ?_⟩
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    N : Nat
    hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
    R : NNReal := (Finset.range N).sup fun n => NNDist.nndist (u n) (u N)
    n : Nat
    ⊢ LT.lt (Dist.dist (u n) (u N)) (HAdd.hAdd (↑R) 1)
  -/
  rcases le_or_lt N n with h | h
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      u : Nat → α
      hu : CauchySeq u
      N : Nat
      hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
      R : NNReal := (Finset.range N).sup fun n => NNDist.nndist (u n) (u N)
      n : Nat
      h : LE.le N n
      ⊢ LT.lt (Dist.dist (u n) (u N)) (HAdd.hAdd (↑R) 1)
    -/
  · exact lt_of_lt_of_le (hN _ h) (le_add_of_nonneg_left R.2)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      u : Nat → α
      hu : CauchySeq u
      N : Nat
      hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
      R : NNReal := (Finset.range N).sup fun n => NNDist.nndist (u n) (u N)
      n : Nat
      h : LT.lt n N
      ⊢ LT.lt (Dist.dist (u n) (u N)) (HAdd.hAdd (↑R) 1)
    -/
  · have : _ ≤ R := Finset.le_sup (Finset.mem_range.2 h)
    /-
      case inr
      α : Type u
      inst✝ : PseudoMetricSpace α
      u : Nat → α
      hu : CauchySeq u
      N : Nat
      hN : ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (u N)) 1
      R : NNReal := (Finset.range N).sup fun n => NNDist.nndist (u n) (u N)
      n : Nat
      h : LT.lt n N
      this : LE.le (NNDist.nndist (u n) (u N)) R
      ⊢ LT.lt (Dist.dist (u n) (u N)) (HAdd.hAdd (↑R) 1)
    -/
    exact lt_of_le_of_lt this (lt_add_of_pos_right _ zero_lt_one)
    /-
      🎉 no goals
    -/


/-- Yet another metric characterization of Cauchy sequences on integers. This one is often the
most efficient. -/
theorem cauchySeq_iff_le_tendsto_0 {s : ℕ → α} :
    CauchySeq s ↔
      ∃ b : ℕ → ℝ,
        (∀ n, 0 ≤ b n) ∧
          (∀ n m N : ℕ, N ≤ n → N ≤ m → dist (s n) (s m) ≤ b N) ∧ Tendsto b atTop (𝓝 0) :=
  ⟨fun hs => by
    /- `s` is a Cauchy sequence. The sequence `b` will be constructed by taking
      the supremum of the distances between `s n` and `s m` for `n m ≥ N`.
      First, we prove that all these distances are bounded, as otherwise the Sup
      would not make sense. -/
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      ⊢ Exists fun b => And (∀ (n : Nat), LE.le 0 (b n)) (And (∀ (n m N : Nat), LE.l …
    -/
    let S N := (fun p : ℕ × ℕ => dist (s p.1) (s p.2)) '' { p | p.1 ≥ N ∧ p.2 ≥ N }
    have hS : ∀ N, ∃ x, ∀ y ∈ S N, y ≤ x := by
      rcases cauchySeq_bdd hs with ⟨R, -, hR⟩
      refine fun N => ⟨R, ?_⟩
      rintro _ ⟨⟨m, n⟩, _, rfl⟩
      exact le_of_lt (hR m n)
    -- Prove that it bounds the distances of points in the Cauchy sequence
    have ub : ∀ m n N, N ≤ m → N ≤ n → dist (s m) (s n) ≤ sSup (S N) := fun m n N hm hn =>
      le_csSup (hS N) ⟨⟨_, _⟩, ⟨hm, hn⟩, rfl⟩
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      ⊢ Exists fun b => And (∀ (n : Nat), LE.le 0 (b n)) (And (∀ (n m N : Nat), LE.l …
    -/
    have S0m : ∀ n, (0 : ℝ) ∈ S n := fun n => ⟨⟨n, n⟩, ⟨le_rfl, le_rfl⟩, dist_self _⟩
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      ⊢ Exists fun b => And (∀ (n : Nat), LE.le 0 (b n)) (And (∀ (n m N : Nat), LE.l …
    -/
    have S0 := fun n => le_csSup (hS n) (S0m n)
    -- Prove that it tends to `0`, by using the Cauchy property of `s`
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      S0 : ∀ (n : Nat), LE.le 0 (SupSet.sSup (S n))
      ⊢ Exists fun b => And (∀ (n : Nat), LE.le 0 (b n)) (And (∀ (n m N : Nat), LE.l …
    -/
    refine ⟨fun N => sSup (S N), S0, ub, Metric.tendsto_atTop.2 fun ε ε0 => ?_⟩
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      S0 : ∀ (n : Nat), LE.le 0 (SupSet.sSup (S n))
      ε : Real
      ε0 : GT.gt ε 0
      ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (SupSet.sSup (S n) …
    -/
    refine (Metric.cauchySeq_iff.1 hs (ε / 2) (half_pos ε0)).imp fun N hN n hn => ?_
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      S0 : ∀ (n : Nat), LE.le 0 (SupSet.sSup (S n))
      ε : Real
      ε0 : GT.gt ε 0
      N : Nat
      hN : ∀ (m : Nat), GE.ge m N → ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (s m)  …
      n : Nat
      hn : GE.ge n N
      ⊢ LT.lt (Dist.dist (SupSet.sSup (S n)) 0) ε
    -/
    rw [Real.dist_0_eq_abs, abs_of_nonneg (S0 n)]
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      S0 : ∀ (n : Nat), LE.le 0 (SupSet.sSup (S n))
      ε : Real
      ε0 : GT.gt ε 0
      N : Nat
      hN : ∀ (m : Nat), GE.ge m N → ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (s m)  …
      n : Nat
      hn : GE.ge n N
      ⊢ LT.lt (SupSet.sSup (S n)) ε
    -/
    refine lt_of_le_of_lt (csSup_le ⟨_, S0m _⟩ ?_) (half_lt_self ε0)
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      S0 : ∀ (n : Nat), LE.le 0 (SupSet.sSup (S n))
      ε : Real
      ε0 : GT.gt ε 0
      N : Nat
      hN : ∀ (m : Nat), GE.ge m N → ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (s m)  …
      n : Nat
      hn : GE.ge n N
      ⊢ ∀ (b : Real), Membership.mem (S n) b → LE.le b (HDiv.hDiv ε 2)
    -/
    rintro _ ⟨⟨m', n'⟩, ⟨hm', hn'⟩, rfl⟩
    /-
      case intro.mk.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Nat → α
      hs : CauchySeq s
      S : Nat → Set Real := fun N => Set.image (fun p => Dist.dist (s p.1) (s p.2))  …
      hS : ∀ (N : Nat), Exists fun x => ∀ (y : Real), Membership.mem (S N) y → LE.le …
      ub : ∀ (m n N : Nat), LE.le N m → LE.le N n → LE.le (Dist.dist (s m) (s n)) (S …
      S0m : ∀ (n : Nat), Membership.mem (S n) 0
      S0 : ∀ (n : Nat), LE.le 0 (SupSet.sSup (S n))
      ε : Real
      ε0 : GT.gt ε 0
      N : Nat
      hN : ∀ (m : Nat), GE.ge m N → ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (s m)  …
      n : Nat
      hn : GE.ge n N
      m' n' : Nat
      hm' : GE.ge { fst := m', snd := n' }.1 n
      hn' : GE.ge { fst := m', snd := n' }.2 n
      ⊢ LE.le ((fun p => Dist.dist (s p.1) (s p.2)) { fst := m', snd := n' }) (HDiv. …
    -/
    exact le_of_lt (hN _ (le_trans hn hm') _ (le_trans hn hn')),
    /-
      🎉 no goals
    -/
   fun ⟨b, _, b_bound, b_lim⟩ => cauchySeq_of_le_tendsto_0 b b_bound b_lim⟩


lemma Metric.exists_subseq_bounded_of_cauchySeq (u : ℕ → α) (hu : CauchySeq u) (b : ℕ → ℝ)
    (hb : ∀ n, 0 < b n) :
    ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, ∀ m ≥ f n, dist (u m) (u (f n)) < b n := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    b : Nat → Real
    hb : ∀ (n : Nat), LT.lt 0 (b n)
    ⊢ Exists fun f => And (StrictMono f) (∀ (n m : Nat), GE.ge m (f n) → LT.lt (Di …
  -/
  rw [cauchySeq_iff] at hu
  have hu' : ∀ k, ∀ᶠ (n : ℕ) in atTop, ∀ m ≥ n, dist (u m) (u n) < b k := by
    intro k
    rw [eventually_atTop]
    obtain ⟨N, hN⟩ := hu (b k) (hb k)
    exact ⟨N, fun m hm r hr => hN r (hm.trans hr) m hm⟩
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (m : Nat), GE.ge m N → ∀ (n : …
    b : Nat → Real
    hb : ∀ (n : Nat), LT.lt 0 (b n)
    hu' : ∀ (k : Nat), Filter.Eventually (fun n => ∀ (m : Nat), GE.ge m n → LT.lt  …
    ⊢ Exists fun f => And (StrictMono f) (∀ (n m : Nat), GE.ge m (f n) → LT.lt (Di …
  -/
  exact Filter.extraction_forall_of_eventually hu'
  /-
    🎉 no goals
  -/


