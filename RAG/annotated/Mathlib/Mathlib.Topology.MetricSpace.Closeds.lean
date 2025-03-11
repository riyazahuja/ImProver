/-- In emetric spaces, the Hausdorff edistance defines an emetric space structure
on the type of closed subsets -/
instance Closeds.emetricSpace : EMetricSpace (Closeds α) where
  edist s t := hausdorffEdist (s : Set α) t
  edist_self _ := hausdorffEdist_self
  edist_comm _ _ := hausdorffEdist_comm
  edist_triangle _ _ _ := hausdorffEdist_triangle
  eq_of_edist_eq_zero {s t} h :=
    Closeds.ext <| (hausdorffEdist_zero_iff_eq_of_closed s.closed t.closed).1 h


/-- The edistance to a closed set depends continuously on the point and the set -/
theorem continuous_infEdist_hausdorffEdist :
    Continuous fun p : α × Closeds α => infEdist p.1 p.2 := by
  /-
    α : Type u
    inst✝ : EMetricSpace α
    ⊢ Continuous fun p => EMetric.infEdist p.1 ↑p.2
  -/
  refine continuous_of_le_add_edist 2 (by simp) ?_
  /-
    α : Type u
    inst✝ : EMetricSpace α
    ⊢ ∀ (x y : Prod α (TopologicalSpace.Closeds α)), LE.le (EMetric.infEdist x.1 ↑ …
  -/
  rintro ⟨x, s⟩ ⟨y, t⟩
  calc
    infEdist x s ≤ infEdist x t + hausdorffEdist (t : Set α) s :=
      infEdist_le_infEdist_add_hausdorffEdist
    _ ≤ infEdist y t + edist x y + hausdorffEdist (t : Set α) s :=
      (add_le_add_right infEdist_le_infEdist_add_edist _)
    _ = infEdist y t + (edist x y + hausdorffEdist (s : Set α) t) := by
      rw [add_assoc, hausdorffEdist_comm]
    _ ≤ infEdist y t + (edist (x, s) (y, t) + edist (x, s) (y, t)) :=
      (add_le_add_left (add_le_add (le_max_left _ _) (le_max_right _ _)) _)
    _ = infEdist y t + 2 * edist (x, s) (y, t) := by rw [← mul_two, mul_comm]


/-- Subsets of a given closed subset form a closed set -/
theorem isClosed_subsets_of_isClosed (hs : IsClosed s) :
    IsClosed { t : Closeds α | (t : Set α) ⊆ s } := by
  refine isClosed_of_closure_subset fun
    (t : Closeds α) (ht : t ∈ closure {t : Closeds α | (t : Set α) ⊆ s}) (x : α) (hx : x ∈ t) => ?_
  have : x ∈ closure s := by
    refine mem_closure_iff.2 fun ε εpos => ?_
    obtain ⟨u : Closeds α, hu : u ∈ {t : Closeds α | (t : Set α) ⊆ s}, Dtu : edist t u < ε⟩ :=
      mem_closure_iff.1 ht ε εpos
    obtain ⟨y : α, hy : y ∈ u, Dxy : edist x y < ε⟩ := exists_edist_lt_of_hausdorffEdist_lt hx Dtu
    exact ⟨y, hu hy, Dxy⟩
  /-
    α : Type u
    inst✝ : EMetricSpace α
    s : Set α
    hs : IsClosed s
    t : TopologicalSpace.Closeds α
    ht : Membership.mem (closure (setOf fun t => HasSubset.Subset (↑t) s)) t
    x : α
    hx : Membership.mem t x
    this : Membership.mem (closure s) x
    ⊢ Membership.mem s x
  -/
  rwa [hs.closure_eq] at this
  /-
    🎉 no goals
  -/


/-- By definition, the edistance on `Closeds α` is given by the Hausdorff edistance -/
theorem Closeds.edist_eq {s t : Closeds α} : edist s t = hausdorffEdist (s : Set α) t :=
  rfl


/-- In a complete space, the type of closed subsets is complete for the
Hausdorff edistance. -/
instance Closeds.completeSpace [CompleteSpace α] : CompleteSpace (Closeds α) := by
  /- We will show that, if a sequence of sets `s n` satisfies
    `edist (s n) (s (n+1)) < 2^{-n}`, then it converges. This is enough to guarantee
    completeness, by a standard completeness criterion.
    We use the shorthand `B n = 2^{-n}` in ennreal. -/
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : CompleteSpace α
    ⊢ CompleteSpace (TopologicalSpace.Closeds α)
  -/
  let B : ℕ → ℝ≥0∞ := fun n => 2⁻¹ ^ n
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    ⊢ CompleteSpace (TopologicalSpace.Closeds α)
  -/
  have B_pos : ∀ n, (0 : ℝ≥0∞) < B n := by simp [B, ENNReal.pow_pos]
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    ⊢ CompleteSpace (TopologicalSpace.Closeds α)
  -/
  have B_ne_top : ∀ n, B n ≠ ⊤ := by simp [B, ENNReal.pow_ne_top]
  /- Consider a sequence of closed sets `s n` with `edist (s n) (s (n+1)) < B n`.
    We will show that it converges. The limit set is `t0 = ⋂n, closure (⋃m≥n, s m)`.
    We will have to show that a point in `s n` is close to a point in `t0`, and a point
    in `t0` is close to a point in `s n`. The completeness then follows from a
    standard criterion. -/
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    ⊢ CompleteSpace (TopologicalSpace.Closeds α)
  -/
  refine complete_of_convergent_controlled_sequences B B_pos fun s hs => ?_
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s✝ : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    s : Nat → TopologicalSpace.Closeds α
    hs : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (EDist.edist (s n) (s m))  …
    ⊢ Exists fun x => Filter.Tendsto s Filter.atTop (nhds x)
  -/
  let t0 := ⋂ n, closure (⋃ m ≥ n, s m : Set α)
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s✝ : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    s : Nat → TopologicalSpace.Closeds α
    hs : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (EDist.edist (s n) (s m))  …
    t0 : Set α := Set.iInter fun n => closure (Set.iUnion fun m => Set.iUnion fun  …
    ⊢ Exists fun x => Filter.Tendsto s Filter.atTop (nhds x)
  -/
  let t : Closeds α := ⟨t0, isClosed_iInter fun _ => isClosed_closure⟩
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    s✝ : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    s : Nat → TopologicalSpace.Closeds α
    hs : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (EDist.edist (s n) (s m))  …
    t0 : Set α := Set.iInter fun n => closure (Set.iUnion fun m => Set.iUnion fun  …
    t : TopologicalSpace.Closeds α := { carrier := t0, closed' := ⋯ }
    ⊢ Exists fun x => Filter.Tendsto s Filter.atTop (nhds x)
  -/
  use t
  -- The inequality is written this way to agree with `edist_le_of_edist_le_geometric_of_tendsto₀`
  have I1 : ∀ n, ∀ x ∈ s n, ∃ y ∈ t0, edist x y ≤ 2 * B n := by
    /- This is the main difficulty of the proof. Starting from `x ∈ s n`, we want
           to find a point in `t0` which is close to `x`. Define inductively a sequence of
           points `z m` with `z n = x` and `z m ∈ s m` and `edist (z m) (z (m+1)) ≤ B m`. This is
           possible since the Hausdorff distance between `s m` and `s (m+1)` is at most `B m`.
           This sequence is a Cauchy sequence, therefore converging as the space is complete, to
           a limit which satisfies the required properties. -/
    intro n x hx
    obtain ⟨z, hz₀, hz⟩ :
      ∃ z : ∀ l, s (n + l), (z 0 : α) = x ∧ ∀ k, edist (z k : α) (z (k + 1) : α) ≤ B n / 2 ^ k := by
      -- We prove existence of the sequence by induction.
      have : ∀ (l) (z : s (n + l)), ∃ z' : s (n + l + 1), edist (z : α) z' ≤ B n / 2 ^ l := by
        intro l z
        obtain ⟨z', z'_mem, hz'⟩ : ∃ z' ∈ s (n + l + 1), edist (z : α) z' < B n / 2 ^ l := by
          refine exists_edist_lt_of_hausdorffEdist_lt (s := s (n + l)) z.2 ?_
          simp only [ENNReal.inv_pow, div_eq_mul_inv]
          rw [← pow_add]
          apply hs <;> simp
        exact ⟨⟨z', z'_mem⟩, le_of_lt hz'⟩
      use fun k => Nat.recOn k ⟨x, hx⟩ fun l z => (this l z).choose
      simp only [Nat.add_zero, Nat.rec_zero, Nat.rec_add_one, true_and]
      exact fun k => (this k _).choose_spec
    -- it follows from the previous bound that `z` is a Cauchy sequence
    have : CauchySeq fun k => (z k : α) := cauchySeq_of_edist_le_geometric_two (B n) (B_ne_top n) hz
    -- therefore, it converges
    rcases cauchySeq_tendsto_of_complete this with ⟨y, y_lim⟩
    use y
    -- the limit point `y` will be the desired point, in `t0` and close to our initial point `x`.
    -- First, we check it belongs to `t0`.
    have : y ∈ t0 :=
      mem_iInter.2 fun k =>
        mem_closure_of_tendsto y_lim
          (by
            simp only [exists_prop, Set.mem_iUnion, Filter.eventually_atTop, Set.mem_preimage,
              Set.preimage_iUnion]
            exact ⟨k, fun m hm => ⟨n + m, zero_add k ▸ add_le_add (zero_le n) hm, (z m).2⟩⟩)
    use this
    -- Then, we check that `y` is close to `x = z n`. This follows from the fact that `y`
    -- is the limit of `z k`, and the distance between `z n` and `z k` has already been estimated.
    rw [← hz₀]
    exact edist_le_of_edist_le_geometric_two_of_tendsto₀ (B n) hz y_lim
  have I2 : ∀ n, ∀ x ∈ t0, ∃ y ∈ s n, edist x y ≤ 2 * B n := by
    /- For the (much easier) reverse inequality, we start from a point `x ∈ t0` and we want
            to find a point `y ∈ s n` which is close to `x`.
            `x` belongs to `t0`, the intersection of the closures. In particular, it is well
            approximated by a point `z` in `⋃m≥n, s m`, say in `s m`. Since `s m` and
            `s n` are close, this point is itself well approximated by a point `y` in `s n`,
            as required. -/
    intro n x xt0
    have : x ∈ closure (⋃ m ≥ n, s m : Set α) := by apply mem_iInter.1 xt0 n
    obtain ⟨z : α, hz, Dxz : edist x z < B n⟩ := mem_closure_iff.1 this (B n) (B_pos n)
    simp only [exists_prop, Set.mem_iUnion] at hz
    obtain ⟨m : ℕ, m_ge_n : m ≥ n, hm : z ∈ (s m : Set α)⟩ := hz
    have : hausdorffEdist (s m : Set α) (s n) < B n := hs n m n m_ge_n (le_refl n)
    obtain ⟨y : α, hy : y ∈ (s n : Set α), Dzy : edist z y < B n⟩ :=
      exists_edist_lt_of_hausdorffEdist_lt hm this
    exact
      ⟨y, hy,
        calc
          edist x y ≤ edist x z + edist z y := edist_triangle _ _ _
          _ ≤ B n + B n := add_le_add (le_of_lt Dxz) (le_of_lt Dzy)
          _ = 2 * B n := (two_mul _).symm
          ⟩
  -- Deduce from the above inequalities that the distance between `s n` and `t0` is at most `2 B n`.
  have main : ∀ n : ℕ, edist (s n) t ≤ 2 * B n := fun n =>
    hausdorffEdist_le_of_mem_edist (I1 n) (I2 n)
  -- from this, the convergence of `s n` to `t0` follows.
  /-
    case h
    α : Type u
    inst✝¹ : EMetricSpace α
    s✝ : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    s : Nat → TopologicalSpace.Closeds α
    hs : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (EDist.edist (s n) (s m))  …
    t0 : Set α := Set.iInter fun n => closure (Set.iUnion fun m => Set.iUnion fun  …
    t : TopologicalSpace.Closeds α := { carrier := t0, closed' := ⋯ }
    I1 : ∀ (n : Nat) (x : α), Membership.mem (s n) x → Exists fun y => And (Member …
    I2 : ∀ (n : Nat) (x : α), Membership.mem t0 x → Exists fun y => And (Membershi …
    main : ∀ (n : Nat), LE.le (EDist.edist (s n) t) (HMul.hMul 2 (B n))
    ⊢ Filter.Tendsto s Filter.atTop (nhds t)
  -/
  refine tendsto_atTop.2 fun ε εpos => ?_
  have : Tendsto (fun n => 2 * B n) atTop (𝓝 (2 * 0)) :=
    ENNReal.Tendsto.const_mul (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one <|
      by simp [ENNReal.one_lt_two]) (Or.inr <| by simp)
  /-
    case h
    α : Type u
    inst✝¹ : EMetricSpace α
    s✝ : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    s : Nat → TopologicalSpace.Closeds α
    hs : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (EDist.edist (s n) (s m))  …
    t0 : Set α := Set.iInter fun n => closure (Set.iUnion fun m => Set.iUnion fun  …
    t : TopologicalSpace.Closeds α := { carrier := t0, closed' := ⋯ }
    I1 : ∀ (n : Nat) (x : α), Membership.mem (s n) x → Exists fun y => And (Member …
    I2 : ∀ (n : Nat) (x : α), Membership.mem t0 x → Exists fun y => And (Membershi …
    main : ∀ (n : Nat), LE.le (EDist.edist (s n) t) (HMul.hMul 2 (B n))
    ε : ENNReal
    εpos : GT.gt ε 0
    this : Filter.Tendsto (fun n => HMul.hMul 2 (B n)) Filter.atTop (nhds (HMul.hM …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (EDist.edist (s n) t) ε
  -/
  rw [mul_zero] at this
  obtain ⟨N, hN⟩ : ∃ N, ∀ b ≥ N, ε > 2 * B b :=
    ((tendsto_order.1 this).2 ε εpos).exists_forall_of_atTop
  /-
    case h.intro
    α : Type u
    inst✝¹ : EMetricSpace α
    s✝ : Set α
    inst✝ : CompleteSpace α
    B : Nat → ENNReal := fun n => HPow.hPow (Inv.inv 2) n
    B_pos : ∀ (n : Nat), LT.lt 0 (B n)
    B_ne_top : ∀ (n : Nat), Ne (B n) Top.top
    s : Nat → TopologicalSpace.Closeds α
    hs : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (EDist.edist (s n) (s m))  …
    t0 : Set α := Set.iInter fun n => closure (Set.iUnion fun m => Set.iUnion fun  …
    t : TopologicalSpace.Closeds α := { carrier := t0, closed' := ⋯ }
    I1 : ∀ (n : Nat) (x : α), Membership.mem (s n) x → Exists fun y => And (Member …
    I2 : ∀ (n : Nat) (x : α), Membership.mem t0 x → Exists fun y => And (Membershi …
    main : ∀ (n : Nat), LE.le (EDist.edist (s n) t) (HMul.hMul 2 (B n))
    ε : ENNReal
    εpos : GT.gt ε 0
    this : Filter.Tendsto (fun n => HMul.hMul 2 (B n)) Filter.atTop (nhds 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → GT.gt ε (HMul.hMul 2 (B b))
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (EDist.edist (s n) t) ε
  -/
  exact ⟨N, fun n hn => lt_of_le_of_lt (main n) (hN n hn)⟩
  /-
    🎉 no goals
  -/


/-- In a compact space, the type of closed subsets is compact. -/
instance Closeds.compactSpace [CompactSpace α] : CompactSpace (Closeds α) :=
  ⟨by
    /- by completeness, it suffices to show that it is totally bounded,
        i.e., for all ε>0, there is a finite set which is ε-dense.
        start from a set `s` which is ε-dense in α. Then the subsets of `s`
        are finitely many, and ε-dense for the Hausdorff distance. -/
    refine
      isCompact_of_totallyBounded_isClosed (EMetric.totallyBounded_iff.2 fun ε εpos => ?_)
        isClosed_univ
    /-
      α : Type u
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : CompactSpace α
      ε : ENNReal
      εpos : GT.gt ε 0
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    rcases exists_between εpos with ⟨δ, δpos, δlt⟩
    obtain ⟨s : Set α, fs : s.Finite, hs : univ ⊆ ⋃ y ∈ s, ball y δ⟩ :=
      EMetric.totallyBounded_iff.1
        (isCompact_iff_totallyBounded_isComplete.1 (@isCompact_univ α _ _)).1 δ δpos
    -- we first show that any set is well approximated by a subset of `s`.
    have main : ∀ u : Set α, ∃ v ⊆ s, hausdorffEdist u v ≤ δ := by
      intro u
      let v := { x : α | x ∈ s ∧ ∃ y ∈ u, edist x y < δ }
      exists v, (fun x hx => hx.1 : v ⊆ s)
      refine hausdorffEdist_le_of_mem_edist ?_ ?_
      · intro x hx
        have : x ∈ ⋃ y ∈ s, ball y δ := hs (by simp)
        rcases mem_iUnion₂.1 this with ⟨y, ys, dy⟩
        have : edist y x < δ := by simpa [edist_comm]
        exact ⟨y, ⟨ys, ⟨x, hx, this⟩⟩, le_of_lt dy⟩
      · rintro x ⟨_, ⟨y, yu, hy⟩⟩
        exact ⟨y, yu, le_of_lt hy⟩
    -- introduce the set F of all subsets of `s` (seen as members of `Closeds α`).
    /-
      case intro.intro.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      s✝ : Set α
      inst✝ : CompactSpace α
      ε : ENNReal
      εpos : GT.gt ε 0
      δ : ENNReal
      δpos : LT.lt 0 δ
      δlt : LT.lt δ ε
      s : Set α
      fs : s.Finite
      hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
      main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    let F := { f : Closeds α | (f : Set α) ⊆ s }
    /-
      case intro.intro.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      s✝ : Set α
      inst✝ : CompactSpace α
      ε : ENNReal
      εpos : GT.gt ε 0
      δ : ENNReal
      δpos : LT.lt 0 δ
      δlt : LT.lt δ ε
      s : Set α
      fs : s.Finite
      hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
      main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
      F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    refine ⟨F, ?_, fun u _ => ?_⟩
    -- `F` is finite
      /-
        case intro.intro.intro.intro.refine_1
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        ⊢ F.Finite
      -/
    · apply @Finite.of_finite_image _ _ F _
        /-
          case intro.intro.intro.intro.refine_1.h
          α : Type u
          inst✝¹ : EMetricSpace α
          s✝ : Set α
          inst✝ : CompactSpace α
          ε : ENNReal
          εpos : GT.gt ε 0
          δ : ENNReal
          δpos : LT.lt 0 δ
          δlt : LT.lt δ ε
          s : Set α
          fs : s.Finite
          hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
          main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
          F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
          ⊢ (Set.image ?m.42665 F).Finite
        -/
      · apply fs.finite_subsets.subset fun b => _
          /-
            α : Type u
            inst✝¹ : EMetricSpace α
            s✝ : Set α
            inst✝ : CompactSpace α
            ε : ENNReal
            εpos : GT.gt ε 0
            δ : ENNReal
            δpos : LT.lt 0 δ
            δlt : LT.lt δ ε
            s : Set α
            fs : s.Finite
            hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
            main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
            F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
            ⊢ TopologicalSpace.Closeds α → Set α
          -/
        · exact fun s => (s : Set α)
          /-
            🎉 no goals
          -/
        /-
          α : Type u
          inst✝¹ : EMetricSpace α
          s✝ : Set α
          inst✝ : CompactSpace α
          ε : ENNReal
          εpos : GT.gt ε 0
          δ : ENNReal
          δpos : LT.lt 0 δ
          δlt : LT.lt δ ε
          s : Set α
          fs : s.Finite
          hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
          main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
          F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
          ⊢ ∀ (b : Set α), Membership.mem (Set.image (fun s => ↑s) F) b → Membership.mem …
        -/
        simp only [F, and_imp, Set.mem_image, Set.mem_setOf_eq, exists_imp]
        /-
          α : Type u
          inst✝¹ : EMetricSpace α
          s✝ : Set α
          inst✝ : CompactSpace α
          ε : ENNReal
          εpos : GT.gt ε 0
          δ : ENNReal
          δpos : LT.lt 0 δ
          δlt : LT.lt δ ε
          s : Set α
          fs : s.Finite
          hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
          main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
          F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
          ⊢ ∀ (b : Set α) (x : TopologicalSpace.Closeds α), HasSubset.Subset (↑x) s → Eq …
        -/
        intro _ x hx hx'
        /-
          α : Type u
          inst✝¹ : EMetricSpace α
          s✝ : Set α
          inst✝ : CompactSpace α
          ε : ENNReal
          εpos : GT.gt ε 0
          δ : ENNReal
          δpos : LT.lt 0 δ
          δlt : LT.lt δ ε
          s : Set α
          fs : s.Finite
          hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
          main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
          F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
          b✝ : Set α
          x : TopologicalSpace.Closeds α
          hx : HasSubset.Subset (↑x) s
          hx' : Eq (↑x) b✝
          ⊢ HasSubset.Subset b✝ s
        -/
        rwa [hx'] at hx
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.refine_1.hi
          α : Type u
          inst✝¹ : EMetricSpace α
          s✝ : Set α
          inst✝ : CompactSpace α
          ε : ENNReal
          εpos : GT.gt ε 0
          δ : ENNReal
          δpos : LT.lt 0 δ
          δlt : LT.lt δ ε
          s : Set α
          fs : s.Finite
          hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
          main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
          F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
          ⊢ Set.InjOn (fun s => ↑s) F
        -/
      · exact SetLike.coe_injective.injOn
        /-
          🎉 no goals
        -/
    -- `F` is ε-dense
      /-
        case intro.intro.intro.intro.refine_2
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) u
      -/
    · obtain ⟨t0, t0s, Dut0⟩ := main u
      /-
        case intro.intro.intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        t0 : Set α
        t0s : HasSubset.Subset t0 s
        Dut0 : LE.le (EMetric.hausdorffEdist (↑u) t0) δ
        ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) u
      -/
      have : IsClosed t0 := (fs.subset t0s).isCompact.isClosed
      /-
        case intro.intro.intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        t0 : Set α
        t0s : HasSubset.Subset t0 s
        Dut0 : LE.le (EMetric.hausdorffEdist (↑u) t0) δ
        this : IsClosed t0
        ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) u
      -/
      let t : Closeds α := ⟨t0, this⟩
      /-
        case intro.intro.intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        t0 : Set α
        t0s : HasSubset.Subset t0 s
        Dut0 : LE.le (EMetric.hausdorffEdist (↑u) t0) δ
        this : IsClosed t0
        t : TopologicalSpace.Closeds α := { carrier := t0, closed' := this }
        ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) u
      -/
      have : t ∈ F := t0s
      /-
        case intro.intro.intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        t0 : Set α
        t0s : HasSubset.Subset t0 s
        Dut0 : LE.le (EMetric.hausdorffEdist (↑u) t0) δ
        this✝ : IsClosed t0
        t : TopologicalSpace.Closeds α := { carrier := t0, closed' := this✝ }
        this : Membership.mem F t
        ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) u
      -/
      have : edist u t < ε := lt_of_le_of_lt Dut0 δlt
      /-
        case intro.intro.intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        t0 : Set α
        t0s : HasSubset.Subset t0 s
        Dut0 : LE.le (EMetric.hausdorffEdist (↑u) t0) δ
        this✝¹ : IsClosed t0
        t : TopologicalSpace.Closeds α := { carrier := t0, closed' := this✝¹ }
        this✝ : Membership.mem F t
        this : LT.lt (EDist.edist u t) ε
        ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) u
      -/
      apply mem_iUnion₂.2
      /-
        case intro.intro.intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : CompactSpace α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        s : Set α
        fs : s.Finite
        hs : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => EMetri …
        main : ∀ (u : Set α), Exists fun v => And (HasSubset.Subset v s) (LE.le (EMetr …
        F : Set (TopologicalSpace.Closeds α) := setOf fun f => HasSubset.Subset (↑f) s
        u : TopologicalSpace.Closeds α
        x✝ : Membership.mem Set.univ u
        t0 : Set α
        t0s : HasSubset.Subset t0 s
        Dut0 : LE.le (EMetric.hausdorffEdist (↑u) t0) δ
        this✝¹ : IsClosed t0
        t : TopologicalSpace.Closeds α := { carrier := t0, closed' := this✝¹ }
        this✝ : Membership.mem F t
        this : LT.lt (EDist.edist u t) ε
        ⊢ Exists fun i => Exists fun j => Membership.mem (EMetric.ball i ε) u
      -/
      exact ⟨t, ‹t ∈ F›, this⟩⟩
      /-
        🎉 no goals
      -/


/-- In an emetric space, the type of non-empty compact subsets is an emetric space,
where the edistance is the Hausdorff edistance -/
instance NonemptyCompacts.emetricSpace : EMetricSpace (NonemptyCompacts α) where
  edist s t := hausdorffEdist (s : Set α) t
  edist_self _ := hausdorffEdist_self
  edist_comm _ _ := hausdorffEdist_comm
  edist_triangle _ _ _ := hausdorffEdist_triangle
  eq_of_edist_eq_zero {s t} h := NonemptyCompacts.ext <| by
    /-
      α : Type u
      inst✝ : EMetricSpace α
      s✝ : Set α
      s t : TopologicalSpace.NonemptyCompacts α
      h : Eq (EDist.edist s t) 0
      ⊢ Eq ↑s ↑t
    -/
    have : closure (s : Set α) = closure t := hausdorffEdist_zero_iff_closure_eq_closure.1 h
    /-
      α : Type u
      inst✝ : EMetricSpace α
      s✝ : Set α
      s t : TopologicalSpace.NonemptyCompacts α
      h : Eq (EDist.edist s t) 0
      this : Eq (closure ↑s) (closure ↑t)
      ⊢ Eq ↑s ↑t
    -/
    rwa [s.isCompact.isClosed.closure_eq, t.isCompact.isClosed.closure_eq] at this
    /-
      🎉 no goals
    -/


/-- `NonemptyCompacts.toCloseds` is a uniform embedding (as it is an isometry) -/
theorem NonemptyCompacts.ToCloseds.isUniformEmbedding :
    IsUniformEmbedding (@NonemptyCompacts.toCloseds α _ _) :=
  Isometry.isUniformEmbedding fun _ _ => rfl


@[deprecated (since := "2024-10-01")]
alias NonemptyCompacts.ToCloseds.uniformEmbedding := NonemptyCompacts.ToCloseds.isUniformEmbedding


/-- The range of `NonemptyCompacts.toCloseds` is closed in a complete space -/
theorem NonemptyCompacts.isClosed_in_closeds [CompleteSpace α] :
    IsClosed (range <| @NonemptyCompacts.toCloseds α _ _) := by
  have :
    range NonemptyCompacts.toCloseds =
      { s : Closeds α | (s : Set α).Nonempty ∧ IsCompact (s : Set α) } := by
    ext s
    refine ⟨?_, fun h => ⟨⟨⟨s, h.2⟩, h.1⟩, Closeds.ext rfl⟩⟩
    rintro ⟨s, hs, rfl⟩
    exact ⟨s.nonempty, s.isCompact⟩
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    inst✝ : CompleteSpace α
    this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
    ⊢ IsClosed (Set.range TopologicalSpace.NonemptyCompacts.toCloseds)
  -/
  rw [this]
  /-
    α : Type u
    inst✝¹ : EMetricSpace α
    inst✝ : CompleteSpace α
    this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
    ⊢ IsClosed (setOf fun s => And (↑s).Nonempty (IsCompact ↑s))
  -/
  refine isClosed_of_closure_subset fun s hs => ⟨?_, ?_⟩
  · -- take a set t which is nonempty and at a finite distance of s
    /-
      case refine_1
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ⊢ (↑s).Nonempty
    -/
    rcases mem_closure_iff.1 hs ⊤ ENNReal.coe_lt_top with ⟨t, ht, Dst⟩
    /-
      case refine_1.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      t : TopologicalSpace.Closeds α
      ht : Membership.mem (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) t
      Dst : LT.lt (EDist.edist s t) Top.top
      ⊢ (↑s).Nonempty
    -/
    rw [edist_comm] at Dst
    -- since `t` is nonempty, so is `s`
    /-
      case refine_1.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      t : TopologicalSpace.Closeds α
      ht : Membership.mem (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) t
      Dst : LT.lt (EDist.edist t s) Top.top
      ⊢ (↑s).Nonempty
    -/
    exact nonempty_of_hausdorffEdist_ne_top ht.1 (ne_of_lt Dst)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ⊢ IsCompact ↑s
    -/
  · refine isCompact_iff_totallyBounded_isComplete.2 ⟨?_, s.closed.isComplete⟩
    /-
      case refine_2
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ⊢ TotallyBounded ↑s
    -/
    refine totallyBounded_iff.2 fun ε (εpos : 0 < ε) => ?_
    -- we have to show that s is covered by finitely many eballs of radius ε
    -- pick a nonempty compact set t at distance at most ε/2 of s
    /-
      case refine_2
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ε : ENNReal
      εpos : LT.lt 0 ε
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset (↑s) (Set.iUnion fun y => Set …
    -/
    rcases mem_closure_iff.1 hs (ε / 2) (ENNReal.half_pos εpos.ne') with ⟨t, ht, Dst⟩
    -- cover this space with finitely many balls of radius ε/2
    rcases totallyBounded_iff.1 (isCompact_iff_totallyBounded_isComplete.1 ht.2).1 (ε / 2)
        (ENNReal.half_pos εpos.ne') with
      ⟨u, fu, ut⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ε : ENNReal
      εpos : LT.lt 0 ε
      t : TopologicalSpace.Closeds α
      ht : Membership.mem (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) t
      Dst : LT.lt (EDist.edist s t) (HDiv.hDiv ε 2)
      u : Set α
      fu : u.Finite
      ut : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset (↑s) (Set.iUnion fun y => Set …
    -/
    refine ⟨u, ⟨fu, fun x hx => ?_⟩⟩
    -- u : set α, fu : u.finite, ut : t ⊆ ⋃ (y : α) (H : y ∈ u), eball y (ε / 2)
    -- then s is covered by the union of the balls centered at u of radius ε
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ε : ENNReal
      εpos : LT.lt 0 ε
      t : TopologicalSpace.Closeds α
      ht : Membership.mem (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) t
      Dst : LT.lt (EDist.edist s t) (HDiv.hDiv ε 2)
      u : Set α
      fu : u.Finite
      ut : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
      x : α
      hx : Membership.mem (↑s) x
      ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) x
    -/
    rcases exists_edist_lt_of_hausdorffEdist_lt hx Dst with ⟨z, hz, Dxz⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun s …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ε : ENNReal
      εpos : LT.lt 0 ε
      t : TopologicalSpace.Closeds α
      ht : Membership.mem (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) t
      Dst : LT.lt (EDist.edist s t) (HDiv.hDiv ε 2)
      u : Set α
      fu : u.Finite
      ut : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
      x : α
      hx : Membership.mem (↑s) x
      z : α
      hz : Membership.mem (↑t) z
      Dxz : LT.lt (EDist.edist x z) (HDiv.hDiv ε 2)
      ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) x
    -/
    rcases mem_iUnion₂.1 (ut hz) with ⟨y, hy, Dzy⟩
    have : edist x y < ε :=
      calc
        edist x y ≤ edist x z + edist z y := edist_triangle _ _ _
        _ < ε / 2 + ε / 2 := ENNReal.add_lt_add Dxz Dzy
        _ = ε := ENNReal.add_halves _
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      inst✝ : CompleteSpace α
      this✝ : Eq (Set.range TopologicalSpace.NonemptyCompacts.toCloseds) (setOf fun  …
      s : TopologicalSpace.Closeds α
      hs : Membership.mem (closure (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) …
      ε : ENNReal
      εpos : LT.lt 0 ε
      t : TopologicalSpace.Closeds α
      ht : Membership.mem (setOf fun s => And (↑s).Nonempty (IsCompact ↑s)) t
      Dst : LT.lt (EDist.edist s t) (HDiv.hDiv ε 2)
      u : Set α
      fu : u.Finite
      ut : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
      x : α
      hx : Membership.mem (↑s) x
      z : α
      hz : Membership.mem (↑t) z
      Dxz : LT.lt (EDist.edist x z) (HDiv.hDiv ε 2)
      y : α
      hy : Membership.mem u y
      Dzy : Membership.mem (EMetric.ball y (HDiv.hDiv ε 2)) z
      this : LT.lt (EDist.edist x y) ε
      ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => EMetric.ball y ε) x
    -/
    exact mem_biUnion hy this
    /-
      🎉 no goals
    -/


/-- In a complete space, the type of nonempty compact subsets is complete. This follows
from the same statement for closed subsets -/
instance NonemptyCompacts.completeSpace [CompleteSpace α] : CompleteSpace (NonemptyCompacts α) :=
  (completeSpace_iff_isComplete_range
        NonemptyCompacts.ToCloseds.isUniformEmbedding.isUniformInducing).2 <|
    NonemptyCompacts.isClosed_in_closeds.isComplete


/-- In a compact space, the type of nonempty compact subsets is compact. This follows from
the same statement for closed subsets -/
instance NonemptyCompacts.compactSpace [CompactSpace α] : CompactSpace (NonemptyCompacts α) :=
  ⟨by
    /-
      α : Type u
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : CompactSpace α
      ⊢ IsCompact Set.univ
    -/
    rw [NonemptyCompacts.ToCloseds.isUniformEmbedding.isEmbedding.isCompact_iff, image_univ]
    /-
      α : Type u
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : CompactSpace α
      ⊢ IsCompact (Set.range TopologicalSpace.NonemptyCompacts.toCloseds)
    -/
    exact NonemptyCompacts.isClosed_in_closeds.isCompact⟩
    /-
      🎉 no goals
    -/


/-- In a second countable space, the type of nonempty compact subsets is second countable -/
instance NonemptyCompacts.secondCountableTopology [SecondCountableTopology α] :
    SecondCountableTopology (NonemptyCompacts α) :=
  haveI : SeparableSpace (NonemptyCompacts α) := by
    /- To obtain a countable dense subset of `NonemptyCompacts α`, start from
        a countable dense subset `s` of α, and then consider all its finite nonempty subsets.
        This set is countable and made of nonempty compact sets. It turns out to be dense:
        by total boundedness, any compact set `t` can be covered by finitely many small balls, and
        approximations in `s` of the centers of these balls give the required finite approximation
        of `t`. -/
    /-
      α : Type u
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : SecondCountableTopology α
      ⊢ TopologicalSpace.SeparableSpace (TopologicalSpace.NonemptyCompacts α)
    -/
    rcases exists_countable_dense α with ⟨s, cs, s_dense⟩
    /-
      case intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      s✝ : Set α
      inst✝ : SecondCountableTopology α
      s : Set α
      cs : s.Countable
      s_dense : Dense s
      ⊢ TopologicalSpace.SeparableSpace (TopologicalSpace.NonemptyCompacts α)
    -/
    let v0 := { t : Set α | t.Finite ∧ t ⊆ s }
    /-
      case intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      s✝ : Set α
      inst✝ : SecondCountableTopology α
      s : Set α
      cs : s.Countable
      s_dense : Dense s
      v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
      ⊢ TopologicalSpace.SeparableSpace (TopologicalSpace.NonemptyCompacts α)
    -/
    let v : Set (NonemptyCompacts α) := { t : NonemptyCompacts α | (t : Set α) ∈ v0 }
    /-
      case intro.intro
      α : Type u
      inst✝¹ : EMetricSpace α
      s✝ : Set α
      inst✝ : SecondCountableTopology α
      s : Set α
      cs : s.Countable
      s_dense : Dense s
      v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
      v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
      ⊢ TopologicalSpace.SeparableSpace (TopologicalSpace.NonemptyCompacts α)
    -/
    refine ⟨⟨v, ?_, ?_⟩⟩
      /-
        case intro.intro.refine_1
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        ⊢ v.Countable
      -/
    · have : v0.Countable := countable_setOf_finite_subset cs
      /-
        case intro.intro.refine_1
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        this : v0.Countable
        ⊢ v.Countable
      -/
      exact this.preimage SetLike.coe_injective
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        ⊢ Dense v
      -/
    · refine fun t => mem_closure_iff.2 fun ε εpos => ?_
      -- t is a compact nonempty set, that we have to approximate uniformly by a a set in `v`.
      /-
        case intro.intro.refine_2
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      rcases exists_between εpos with ⟨δ, δpos, δlt⟩
      /-
        case intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have δpos' : 0 < δ / 2 := ENNReal.half_pos δpos.ne'
      -- construct a map F associating to a point in α an approximating point in s, up to δ/2.
      have Exy : ∀ x, ∃ y, y ∈ s ∧ edist x y < δ / 2 := by
        intro x
        rcases mem_closure_iff.1 (s_dense x) (δ / 2) δpos' with ⟨y, ys, hy⟩
        exact ⟨y, ⟨ys, hy⟩⟩
      /-
        case intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      let F x := (Exy x).choose
      /-
        case intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have Fspec : ∀ x, F x ∈ s ∧ edist x (F x) < δ / 2 := fun x => (Exy x).choose_spec
      -- cover `t` with finitely many balls. Their centers form a set `a`
      /-
        case intro.intro.refine_2.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have : TotallyBounded (t : Set α) := t.isCompact.totallyBounded
      obtain ⟨a : Set α, af : Set.Finite a, ta : (t : Set α) ⊆ ⋃ y ∈ a, ball y (δ / 2)⟩ :=
        totallyBounded_iff.1 this (δ / 2) δpos'
      -- replace each center by a nearby approximation in `s`, giving a new set `b`
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      let b := F '' a
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have : b.Finite := af.image _
      have tb : ∀ x ∈ t, ∃ y ∈ b, edist x y < δ := by
        intro x hx
        rcases mem_iUnion₂.1 (ta hx) with ⟨z, za, Dxz⟩
        exists F z, mem_image_of_mem _ za
        calc
          edist x (F z) ≤ edist x z + edist z (F z) := edist_triangle _ _ _
          _ < δ / 2 + δ / 2 := ENNReal.add_lt_add Dxz (Fspec z).2
          _ = δ := ENNReal.add_halves _
      -- keep only the points in `b` that are close to point in `t`, yielding a new set `c`
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝ : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      let c := { y ∈ b | ∃ x ∈ t, edist x y < δ }
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝ : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have : c.Finite := ‹b.Finite›.subset fun x hx => hx.1
      -- points in `t` are well approximated by points in `c`
      have tc : ∀ x ∈ t, ∃ y ∈ c, edist x y ≤ δ := by
        intro x hx
        rcases tb x hx with ⟨y, yv, Dxy⟩
        have : y ∈ c := by simpa [c, -mem_image] using ⟨yv, ⟨x, hx, Dxy⟩⟩
        exact ⟨y, this, le_of_lt Dxy⟩
      -- points in `c` are well approximated by points in `t`
      have ct : ∀ y ∈ c, ∃ x ∈ t, edist y x ≤ δ := by
        rintro y ⟨_, x, xt, Dyx⟩
        have : edist y x ≤ δ :=
          calc
            edist y x = edist x y := edist_comm _ _
            _ ≤ δ := le_of_lt Dyx
        exact ⟨x, xt, this⟩
      -- it follows that their Hausdorff distance is small
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝¹ : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this✝ : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        this : c.Finite
        tc : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem c y)  …
        ct : ∀ (y : α), Membership.mem c y → Exists fun x => And (Membership.mem t x)  …
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have : hausdorffEdist (t : Set α) c ≤ δ := hausdorffEdist_le_of_mem_edist tc ct
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝² : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this✝¹ : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        this✝ : c.Finite
        tc : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem c y)  …
        ct : ∀ (y : α), Membership.mem c y → Exists fun x => And (Membership.mem t x)  …
        this : LE.le (EMetric.hausdorffEdist (↑t) c) δ
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have Dtc : hausdorffEdist (t : Set α) c < ε := this.trans_lt δlt
      -- the set `c` is not empty, as it is well approximated by a nonempty set
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝² : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this✝¹ : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        this✝ : c.Finite
        tc : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem c y)  …
        ct : ∀ (y : α), Membership.mem c y → Exists fun x => And (Membership.mem t x)  …
        this : LE.le (EMetric.hausdorffEdist (↑t) c) δ
        Dtc : LT.lt (EMetric.hausdorffEdist (↑t) c) ε
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have hc : c.Nonempty := nonempty_of_hausdorffEdist_ne_top t.nonempty (ne_top_of_lt Dtc)
      -- let `d` be the version of `c` in the type `NonemptyCompacts α`
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝² : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this✝¹ : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        this✝ : c.Finite
        tc : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem c y)  …
        ct : ∀ (y : α), Membership.mem c y → Exists fun x => And (Membership.mem t x)  …
        this : LE.le (EMetric.hausdorffEdist (↑t) c) δ
        Dtc : LT.lt (EMetric.hausdorffEdist (↑t) c) ε
        hc : c.Nonempty
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      let d : NonemptyCompacts α := ⟨⟨c, ‹c.Finite›.isCompact⟩, hc⟩
      have : c ⊆ s := by
        intro x hx
        rcases (mem_image _ _ _).1 hx.1 with ⟨y, ⟨_, yx⟩⟩
        rw [← yx]
        exact (Fspec y).1
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝³ : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this✝² : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        this✝¹ : c.Finite
        tc : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem c y)  …
        ct : ∀ (y : α), Membership.mem c y → Exists fun x => And (Membership.mem t x)  …
        this✝ : LE.le (EMetric.hausdorffEdist (↑t) c) δ
        Dtc : LT.lt (EMetric.hausdorffEdist (↑t) c) ε
        hc : c.Nonempty
        d : TopologicalSpace.NonemptyCompacts α := { carrier := c, isCompact' := ⋯, no …
        this : HasSubset.Subset c s
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      have : d ∈ v := ⟨‹c.Finite›, this⟩
      -- we have proved that `d` is a good approximation of `t` as requested
      /-
        case intro.intro.refine_2.intro.intro.intro.intro
        α : Type u
        inst✝¹ : EMetricSpace α
        s✝ : Set α
        inst✝ : SecondCountableTopology α
        s : Set α
        cs : s.Countable
        s_dense : Dense s
        v0 : Set (Set α) := setOf fun t => And t.Finite (HasSubset.Subset t s)
        v : Set (TopologicalSpace.NonemptyCompacts α) := setOf fun t => Membership.mem …
        t : TopologicalSpace.NonemptyCompacts α
        ε : ENNReal
        εpos : GT.gt ε 0
        δ : ENNReal
        δpos : LT.lt 0 δ
        δlt : LT.lt δ ε
        δpos' : LT.lt 0 (HDiv.hDiv δ 2)
        Exy : ∀ (x : α), Exists fun y => And (Membership.mem s y) (LT.lt (EDist.edist  …
        F : α → α := fun x => ⋯.choose
        Fspec : ∀ (x : α), And (Membership.mem s (F x)) (LT.lt (EDist.edist x (F x)) ( …
        this✝⁴ : TotallyBounded ↑t
        a : Set α
        af : a.Finite
        ta : HasSubset.Subset (↑t) (Set.iUnion fun y => Set.iUnion fun h => EMetric.ba …
        b : Set α := Set.image F a
        this✝³ : b.Finite
        tb : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem b y)  …
        c : Set α := setOf fun y => And (Membership.mem b y) (Exists fun x => And (Mem …
        this✝² : c.Finite
        tc : ∀ (x : α), Membership.mem t x → Exists fun y => And (Membership.mem c y)  …
        ct : ∀ (y : α), Membership.mem c y → Exists fun x => And (Membership.mem t x)  …
        this✝¹ : LE.le (EMetric.hausdorffEdist (↑t) c) δ
        Dtc : LT.lt (EMetric.hausdorffEdist (↑t) c) ε
        hc : c.Nonempty
        d : TopologicalSpace.NonemptyCompacts α := { carrier := c, isCompact' := ⋯, no …
        this✝ : HasSubset.Subset c s
        this : Membership.mem v d
        ⊢ Exists fun y => And (Membership.mem v y) (LT.lt (EDist.edist t y) ε)
      -/
      exact ⟨d, ‹d ∈ v›, Dtc⟩
      /-
        🎉 no goals
      -/
  UniformSpace.secondCountable_of_separable (NonemptyCompacts α)


/-- `NonemptyCompacts α` inherits a metric space structure, as the Hausdorff
edistance between two such sets is finite. -/
instance NonemptyCompacts.metricSpace : MetricSpace (NonemptyCompacts α) :=
  EMetricSpace.toMetricSpace fun x y =>
    hausdorffEdist_ne_top_of_nonempty_of_bounded x.nonempty y.nonempty x.isCompact.isBounded
      y.isCompact.isBounded


/-- The distance on `NonemptyCompacts α` is the Hausdorff distance, by construction -/
theorem NonemptyCompacts.dist_eq {x y : NonemptyCompacts α} :
    dist x y = hausdorffDist (x : Set α) y :=
  rfl


theorem lipschitz_infDist_set (x : α) : LipschitzWith 1 fun s : NonemptyCompacts α => infDist x s :=
  LipschitzWith.of_le_add fun s t => by
    /-
      α : Type u
      inst✝ : MetricSpace α
      x : α
      s t : TopologicalSpace.NonemptyCompacts α
      ⊢ LE.le (Metric.infDist x ↑s) (HAdd.hAdd (Metric.infDist x ↑t) (Dist.dist s t))
    -/
    rw [dist_comm]
    /-
      α : Type u
      inst✝ : MetricSpace α
      x : α
      s t : TopologicalSpace.NonemptyCompacts α
      ⊢ LE.le (Metric.infDist x ↑s) (HAdd.hAdd (Metric.infDist x ↑t) (Dist.dist t s))
    -/
    exact infDist_le_infDist_add_hausdorffDist (edist_ne_top t s)
    /-
      🎉 no goals
    -/


theorem lipschitz_infDist : LipschitzWith 2 fun p : α × NonemptyCompacts α => infDist p.1 p.2 := by
  -- Porting note: Changed tactic from `exact` to `convert`, because Lean had trouble with 2 = 1 + 1
  convert @LipschitzWith.uncurry α (NonemptyCompacts α) ℝ _ _ _
    (fun (x : α) (s : NonemptyCompacts α) => infDist x s) 1 1
    (fun s => lipschitz_infDist_pt ↑s) lipschitz_infDist_set
  /-
    case h.e'_5
    α : Type u
    inst✝ : MetricSpace α
    ⊢ Eq 2 (HAdd.hAdd 1 1)
  -/
  norm_num
  /-
    🎉 no goals
  -/


theorem uniformContinuous_infDist_Hausdorff_dist :
    UniformContinuous fun p : α × NonemptyCompacts α => infDist p.1 p.2 :=
  lipschitz_infDist.uniformContinuous


