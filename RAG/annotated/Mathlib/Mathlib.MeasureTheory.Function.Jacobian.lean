/-- Assume that a function `f` has a derivative at every point of a set `s`. Then one may cover `s`
with countably many closed sets `t n` on which `f` is well approximated by linear maps `A n`. -/
theorem exists_closed_cover_approximatesLinearOn_of_hasFDerivWithinAt [SecondCountableTopology F]
    (f : E → F) (s : Set E) (f' : E → E →L[ℝ] F) (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x)
    (r : (E →L[ℝ] F) → ℝ≥0) (rpos : ∀ A, r A ≠ 0) :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] F),
      (∀ n, IsClosed (t n)) ∧
        (s ⊆ ⋃ n, t n) ∧
          (∀ n, ApproximatesLinearOn f (A n) (s ∩ t n) (r (A n))) ∧
            (s.Nonempty → ∀ n, ∃ y ∈ s, A n = f' y) := by
  /- Choose countably many linear maps `f' z`. For every such map, if `f` has a derivative at `x`
    close enough to `f' z`, then `f y - f x` is well approximated by `f' z (y - x)` for `y` close
    enough to `x`, say on a ball of radius `r` (or even `u n` for some `n`, where `u` is a fixed
    sequence tending to `0`).
    Let `M n z` be the points where this happens. Then this set is relatively closed inside `s`,
    and moreover in every closed ball of radius `u n / 3` inside it the map is well approximated by
    `f' z`. Using countably many closed balls to split `M n z` into small diameter subsets
    `K n z p`, one obtains the desired sets `t q` after reindexing.
    -/
  -- exclude the trivial case where `s` is empty
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : SecondCountableTopology F
    f : E → F
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F), Ne (r A) 0
    ⊢ Exists fun t => Exists fun A => And (∀ (n : Nat), IsClosed (t n)) (And (HasS …
  -/
  rcases eq_empty_or_nonempty s with (rfl | hs)
    /-
      case inl
      E : Type u_1
      F : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      inst✝ : SecondCountableTopology F
      f : E → F
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      r : ContinuousLinearMap (RingHom.id Real) E F → NNReal
      rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F), Ne (r A) 0
      hf' : ∀ (x : E), Membership.mem EmptyCollection.emptyCollection x → HasFDerivW …
      ⊢ Exists fun t => Exists fun A => And (∀ (n : Nat), IsClosed (t n)) (And (HasS …
    -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  · refine ⟨fun _ => ∅, fun _ => 0, ?_, ?_, ?_, ?_⟩ <;> simp
                                                        /-
                                                          🎉 no goals
                                                        -/
  -- we will use countably many linear maps. Select these from all the derivatives since the
  -- space of linear maps is second-countable
  obtain ⟨T, T_count, hT⟩ :
    ∃ T : Set s,
      T.Countable ∧ ⋃ x ∈ T, ball (f' (x : E)) (r (f' x)) = ⋃ x : s, ball (f' x) (r (f' x)) :=
    TopologicalSpace.isOpen_iUnion_countable _ fun x => isOpen_ball
  -- fix a sequence `u` of positive reals tending to zero.
  obtain ⟨u, _, u_pos, u_lim⟩ :
    ∃ u : ℕ → ℝ, StrictAnti u ∧ (∀ n : ℕ, 0 < u n) ∧ Tendsto u atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  -- `M n z` is the set of points `x` such that `f y - f x` is close to `f' z (y - x)` for `y`
  -- in the ball of radius `u n` around `x`.
  let M : ℕ → T → Set E := fun n z =>
    {x | x ∈ s ∧ ∀ y ∈ s ∩ ball x (u n), ‖f y - f x - f' z (y - x)‖ ≤ r (f' z) * ‖y - x‖}
  -- As `f` is differentiable everywhere on `s`, the sets `M n z` cover `s` by design.
  have s_subset : ∀ x ∈ s, ∃ (n : ℕ) (z : T), x ∈ M n z := by
    intro x xs
    obtain ⟨z, zT, hz⟩ : ∃ z ∈ T, f' x ∈ ball (f' (z : E)) (r (f' z)) := by
      have : f' x ∈ ⋃ z ∈ T, ball (f' (z : E)) (r (f' z)) := by
        rw [hT]
        refine mem_iUnion.2 ⟨⟨x, xs⟩, ?_⟩
        simpa only [mem_ball, Subtype.coe_mk, dist_self] using (rpos (f' x)).bot_lt
      rwa [mem_iUnion₂, bex_def] at this
    obtain ⟨ε, εpos, hε⟩ : ∃ ε : ℝ, 0 < ε ∧ ‖f' x - f' z‖ + ε ≤ r (f' z) := by
      refine ⟨r (f' z) - ‖f' x - f' z‖, ?_, le_of_eq (by abel)⟩
      simpa only [sub_pos] using mem_ball_iff_norm.mp hz
    obtain ⟨δ, δpos, hδ⟩ :
      ∃ (δ : ℝ), 0 < δ ∧ ball x δ ∩ s ⊆ {y | ‖f y - f x - (f' x) (y - x)‖ ≤ ε * ‖y - x‖} :=
      Metric.mem_nhdsWithin_iff.1 ((hf' x xs).isLittleO.def εpos)
    obtain ⟨n, hn⟩ : ∃ n, u n < δ := ((tendsto_order.1 u_lim).2 _ δpos).exists
    refine ⟨n, ⟨z, zT⟩, ⟨xs, ?_⟩⟩
    intro y hy
    calc
      ‖f y - f x - (f' z) (y - x)‖ = ‖f y - f x - (f' x) (y - x) + (f' x - f' z) (y - x)‖ := by
        congr 1
        simp only [ContinuousLinearMap.coe_sub', map_sub, Pi.sub_apply]
        abel
      _ ≤ ‖f y - f x - (f' x) (y - x)‖ + ‖(f' x - f' z) (y - x)‖ := norm_add_le _ _
      _ ≤ ε * ‖y - x‖ + ‖f' x - f' z‖ * ‖y - x‖ := by
        refine add_le_add (hδ ?_) (ContinuousLinearMap.le_opNorm _ _)
        rw [inter_comm]
        exact inter_subset_inter_right _ (ball_subset_ball hn.le) hy
      _ ≤ r (f' z) * ‖y - x‖ := by
        rw [← add_mul, add_comm]
        gcongr
  -- the sets `M n z` are relatively closed in `s`, as all the conditions defining it are clearly
  -- closed
  have closure_M_subset : ∀ n z, s ∩ closure (M n z) ⊆ M n z := by
    rintro n z x ⟨xs, hx⟩
    refine ⟨xs, fun y hy => ?_⟩
    obtain ⟨a, aM, a_lim⟩ : ∃ a : ℕ → E, (∀ k, a k ∈ M n z) ∧ Tendsto a atTop (𝓝 x) :=
      mem_closure_iff_seq_limit.1 hx
    have L1 :
      Tendsto (fun k : ℕ => ‖f y - f (a k) - (f' z) (y - a k)‖) atTop
        (𝓝 ‖f y - f x - (f' z) (y - x)‖) := by
      apply Tendsto.norm
      have L : Tendsto (fun k => f (a k)) atTop (𝓝 (f x)) := by
        apply (hf' x xs).continuousWithinAt.tendsto.comp
        apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ a_lim
        exact Eventually.of_forall fun k => (aM k).1
      apply Tendsto.sub (tendsto_const_nhds.sub L)
      exact ((f' z).continuous.tendsto _).comp (tendsto_const_nhds.sub a_lim)
    have L2 : Tendsto (fun k : ℕ => (r (f' z) : ℝ) * ‖y - a k‖) atTop (𝓝 (r (f' z) * ‖y - x‖)) :=
      (tendsto_const_nhds.sub a_lim).norm.const_mul _
    have I : ∀ᶠ k in atTop, ‖f y - f (a k) - (f' z) (y - a k)‖ ≤ r (f' z) * ‖y - a k‖ := by
      have L : Tendsto (fun k => dist y (a k)) atTop (𝓝 (dist y x)) :=
        tendsto_const_nhds.dist a_lim
      filter_upwards [(tendsto_order.1 L).2 _ hy.2]
      intro k hk
      exact (aM k).2 y ⟨hy.1, hk⟩
    exact le_of_tendsto_of_tendsto L1 L2 I
  -- choose a dense sequence `d p`
  /-
    case inr.intro.intro.intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : SecondCountableTopology F
    f : E → F
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    ⊢ Exists fun t => Exists fun A => And (∀ (n : Nat), IsClosed (t n)) (And (HasS …
  -/
  rcases TopologicalSpace.exists_dense_seq E with ⟨d, hd⟩
  -- split `M n z` into subsets `K n z p` of small diameters by intersecting with the ball
  -- `closedBall (d p) (u n / 3)`.
  /-
    case inr.intro.intro.intro.intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : SecondCountableTopology F
    f : E → F
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    d : Nat → E
    hd : DenseRange d
    ⊢ Exists fun t => Exists fun A => And (∀ (n : Nat), IsClosed (t n)) (And (HasS …
  -/
  let K : ℕ → T → ℕ → Set E := fun n z p => closure (M n z) ∩ closedBall (d p) (u n / 3)
  -- on the sets `K n z p`, the map `f` is well approximated by `f' z` by design.
  have K_approx : ∀ (n) (z : T) (p), ApproximatesLinearOn f (f' z) (s ∩ K n z p) (r (f' z)) := by
    intro n z p x hx y hy
    have yM : y ∈ M n z := closure_M_subset _ _ ⟨hy.1, hy.2.1⟩
    refine yM.2 _ ⟨hx.1, ?_⟩
    calc
      dist x y ≤ dist x (d p) + dist y (d p) := dist_triangle_right _ _ _
      _ ≤ u n / 3 + u n / 3 := add_le_add hx.2.2 hy.2.2
      _ < u n := by linarith [u_pos n]
  -- the sets `K n z p` are also closed, again by design.
  have K_closed : ∀ (n) (z : T) (p), IsClosed (K n z p) := fun n z p =>
    isClosed_closure.inter isClosed_ball
  -- reindex the sets `K n z p`, to let them only depend on an integer parameter `q`.
  obtain ⟨F, hF⟩ : ∃ F : ℕ → ℕ × T × ℕ, Function.Surjective F := by
    haveI : Encodable T := T_count.toEncodable
    have : Nonempty T := by
      rcases hs with ⟨x, xs⟩
      rcases s_subset x xs with ⟨n, z, _⟩
      exact ⟨z⟩
    inhabit ↥T
    exact ⟨_, Encodable.surjective_decode_iget (ℕ × T × ℕ)⟩
  -- these sets `t q = K n z p` will do
  refine
    ⟨fun q => K (F q).1 (F q).2.1 (F q).2.2, fun q => f' (F q).2.1, fun n => K_closed _ _ _,
      fun x xs => ?_, fun q => K_approx _ _ _, fun _ q => ⟨(F q).2.1, (F q).2.1.1.2, rfl⟩⟩
  -- the only fact that needs further checking is that they cover `s`.
  -- we already know that any point `x ∈ s` belongs to a set `M n z`.
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    F✝ : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F✝
    inst✝¹ : NormedSpace Real F✝
    inst✝ : SecondCountableTopology F✝
    f : E → F✝
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F✝
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F✝ → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F✝), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    d : Nat → E
    hd : DenseRange d
    K : Nat → ↑T → Nat → Set E := fun n z p => Inter.inter (closure (M n z)) (Metr …
    K_approx : ∀ (n : Nat) (z : ↑T) (p : Nat), ApproximatesLinearOn f (f' ↑↑z) (In …
    K_closed : ∀ (n : Nat) (z : ↑T) (p : Nat), IsClosed (K n z p)
    F : Nat → Prod Nat (Prod (↑T) Nat)
    hF : Function.Surjective F
    x : E
    xs : Membership.mem s x
    ⊢ Membership.mem (Set.iUnion fun n => (fun q => K (F q).1 (F q).2.1 (F q).2.2) …
  -/
  obtain ⟨n, z, hnz⟩ : ∃ (n : ℕ) (z : T), x ∈ M n z := s_subset x xs
  -- by density, it also belongs to a ball `closedBall (d p) (u n / 3)`.
  obtain ⟨p, hp⟩ : ∃ p : ℕ, x ∈ closedBall (d p) (u n / 3) := by
    have : Set.Nonempty (ball x (u n / 3)) := by simp only [nonempty_ball]; linarith [u_pos n]
    obtain ⟨p, hp⟩ : ∃ p : ℕ, d p ∈ ball x (u n / 3) := hd.exists_mem_open isOpen_ball this
    exact ⟨p, (mem_ball'.1 hp).le⟩
  -- choose `q` for which `t q = K n z p`.
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    F✝ : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F✝
    inst✝¹ : NormedSpace Real F✝
    inst✝ : SecondCountableTopology F✝
    f : E → F✝
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F✝
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F✝ → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F✝), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    d : Nat → E
    hd : DenseRange d
    K : Nat → ↑T → Nat → Set E := fun n z p => Inter.inter (closure (M n z)) (Metr …
    K_approx : ∀ (n : Nat) (z : ↑T) (p : Nat), ApproximatesLinearOn f (f' ↑↑z) (In …
    K_closed : ∀ (n : Nat) (z : ↑T) (p : Nat), IsClosed (K n z p)
    F : Nat → Prod Nat (Prod (↑T) Nat)
    hF : Function.Surjective F
    x : E
    xs : Membership.mem s x
    n : Nat
    z : ↑T
    hnz : Membership.mem (M n z) x
    p : Nat
    hp : Membership.mem (Metric.closedBall (d p) (HDiv.hDiv (u n) 3)) x
    ⊢ Membership.mem (Set.iUnion fun n => (fun q => K (F q).1 (F q).2.1 (F q).2.2) …
  -/
  obtain ⟨q, hq⟩ : ∃ q, F q = (n, z, p) := hF _
  -- then `x` belongs to `t q`.
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    F✝ : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F✝
    inst✝¹ : NormedSpace Real F✝
    inst✝ : SecondCountableTopology F✝
    f : E → F✝
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F✝
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F✝ → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F✝), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    d : Nat → E
    hd : DenseRange d
    K : Nat → ↑T → Nat → Set E := fun n z p => Inter.inter (closure (M n z)) (Metr …
    K_approx : ∀ (n : Nat) (z : ↑T) (p : Nat), ApproximatesLinearOn f (f' ↑↑z) (In …
    K_closed : ∀ (n : Nat) (z : ↑T) (p : Nat), IsClosed (K n z p)
    F : Nat → Prod Nat (Prod (↑T) Nat)
    hF : Function.Surjective F
    x : E
    xs : Membership.mem s x
    n : Nat
    z : ↑T
    hnz : Membership.mem (M n z) x
    p : Nat
    hp : Membership.mem (Metric.closedBall (d p) (HDiv.hDiv (u n) 3)) x
    q : Nat
    hq : Eq (F q) { fst := n, snd := { fst := z, snd := p } }
    ⊢ Membership.mem (Set.iUnion fun n => (fun q => K (F q).1 (F q).2.1 (F q).2.2) …
  -/
  apply mem_iUnion.2 ⟨q, _⟩
  /-
    E : Type u_1
    F✝ : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F✝
    inst✝¹ : NormedSpace Real F✝
    inst✝ : SecondCountableTopology F✝
    f : E → F✝
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F✝
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F✝ → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F✝), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    d : Nat → E
    hd : DenseRange d
    K : Nat → ↑T → Nat → Set E := fun n z p => Inter.inter (closure (M n z)) (Metr …
    K_approx : ∀ (n : Nat) (z : ↑T) (p : Nat), ApproximatesLinearOn f (f' ↑↑z) (In …
    K_closed : ∀ (n : Nat) (z : ↑T) (p : Nat), IsClosed (K n z p)
    F : Nat → Prod Nat (Prod (↑T) Nat)
    hF : Function.Surjective F
    x : E
    xs : Membership.mem s x
    n : Nat
    z : ↑T
    hnz : Membership.mem (M n z) x
    p : Nat
    hp : Membership.mem (Metric.closedBall (d p) (HDiv.hDiv (u n) 3)) x
    q : Nat
    hq : Eq (F q) { fst := n, snd := { fst := z, snd := p } }
    ⊢ Membership.mem (K (F q).1 (F q).2.1 (F q).2.2) x
  -/
  simp (config := { zeta := false }) only [K, hq, mem_inter_iff, hp, and_true]
  /-
    E : Type u_1
    F✝ : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : NormedAddCommGroup F✝
    inst✝¹ : NormedSpace Real F✝
    inst✝ : SecondCountableTopology F✝
    f : E → F✝
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id Real) E F✝
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    r : ContinuousLinearMap (RingHom.id Real) E F✝ → NNReal
    rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F✝), Ne (r A) 0
    hs : s.Nonempty
    T : Set ↑s
    T_count : T.Countable
    hT : Eq (Set.iUnion fun x => Set.iUnion fun h => Metric.ball (f' ↑x) ↑(r (f' ↑ …
    u : Nat → Real
    left✝ : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    M : Nat → ↑T → Set E := fun n z => setOf fun x => And (Membership.mem s x) (∀  …
    s_subset : ∀ (x : E), Membership.mem s x → Exists fun n => Exists fun z => Mem …
    closure_M_subset : ∀ (n : Nat) (z : ↑T), HasSubset.Subset (Inter.inter s (clos …
    d : Nat → E
    hd : DenseRange d
    K : Nat → ↑T → Nat → Set E := fun n z p => Inter.inter (closure (M n z)) (Metr …
    K_approx : ∀ (n : Nat) (z : ↑T) (p : Nat), ApproximatesLinearOn f (f' ↑↑z) (In …
    K_closed : ∀ (n : Nat) (z : ↑T) (p : Nat), IsClosed (K n z p)
    F : Nat → Prod Nat (Prod (↑T) Nat)
    hF : Function.Surjective F
    x : E
    xs : Membership.mem s x
    n : Nat
    z : ↑T
    hnz : Membership.mem (M n z) x
    p : Nat
    hp : Membership.mem (Metric.closedBall (d p) (HDiv.hDiv (u n) 3)) x
    q : Nat
    hq : Eq (F q) { fst := n, snd := { fst := z, snd := p } }
    ⊢ Membership.mem (closure (M n z)) x
  -/
  exact subset_closure hnz
  /-
    🎉 no goals
  -/


/-- Assume that a function `f` has a derivative at every point of a set `s`. Then one may
partition `s` into countably many disjoint relatively measurable sets (i.e., intersections
of `s` with measurable sets `t n`) on which `f` is well approximated by linear maps `A n`. -/
theorem exists_partition_approximatesLinearOn_of_hasFDerivWithinAt [SecondCountableTopology F]
    (f : E → F) (s : Set E) (f' : E → E →L[ℝ] F) (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x)
    (r : (E →L[ℝ] F) → ℝ≥0) (rpos : ∀ A, r A ≠ 0) :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] F),
      Pairwise (Disjoint on t) ∧
        (∀ n, MeasurableSet (t n)) ∧
          (s ⊆ ⋃ n, t n) ∧
            (∀ n, ApproximatesLinearOn f (A n) (s ∩ t n) (r (A n))) ∧
              (s.Nonempty → ∀ n, ∃ y ∈ s, A n = f' y) := by
  rcases exists_closed_cover_approximatesLinearOn_of_hasFDerivWithinAt f s f' hf' r rpos with
    ⟨t, A, t_closed, st, t_approx, ht⟩
  refine
    ⟨disjointed t, A, disjoint_disjointed _,
      MeasurableSet.disjointed fun n => (t_closed n).measurableSet, ?_, ?_, ht⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      E : Type u_1
      F : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : SecondCountableTopology F
      f : E → F
      s : Set E
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      r : ContinuousLinearMap (RingHom.id Real) E F → NNReal
      rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F), Ne (r A) 0
      t : Nat → Set E
      A : Nat → ContinuousLinearMap (RingHom.id Real) E F
      t_closed : ∀ (n : Nat), IsClosed (t n)
      st : HasSubset.Subset s (Set.iUnion fun n => t n)
      t_approx : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) (r  …
      ht : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (Eq (A …
      ⊢ HasSubset.Subset s (Set.iUnion fun n => disjointed t n)
    -/
  · rw [iUnion_disjointed]; exact st
                            /-
                              🎉 no goals
                            -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      E : Type u_1
      F : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : SecondCountableTopology F
      f : E → F
      s : Set E
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      r : ContinuousLinearMap (RingHom.id Real) E F → NNReal
      rpos : ∀ (A : ContinuousLinearMap (RingHom.id Real) E F), Ne (r A) 0
      t : Nat → Set E
      A : Nat → ContinuousLinearMap (RingHom.id Real) E F
      t_closed : ∀ (n : Nat), IsClosed (t n)
      st : HasSubset.Subset s (Set.iUnion fun n => t n)
      t_approx : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) (r  …
      ht : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (Eq (A …
      ⊢ ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (disjointed t n)) ( …
    -/
  · intro n; exact (t_approx n).mono_set (inter_subset_inter_right _ (disjointed_subset _ _))
             /-
               🎉 no goals
             -/


/-- Let `f` be a function which is sufficiently close (in the Lipschitz sense) to a given linear
map `A`. Then it expands the volume of any set by at most `m` for any `m > det A`. -/
theorem addHaar_image_le_mul_of_det_lt (A : E →L[ℝ] E) {m : ℝ≥0}
    (hm : ENNReal.ofReal |A.det| < m) :
    ∀ᶠ δ in 𝓝[>] (0 : ℝ≥0),
      ∀ (s : Set E) (f : E → E), ApproximatesLinearOn f A s δ → μ (f '' s) ≤ m * μ s := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    ⊢ Filter.Eventually (fun δ => ∀ (s : Set E) (f : E → E), ApproximatesLinearOn  …
  -/
  apply nhdsWithin_le_nhds
  /-
    case a
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
  -/
  let d := ENNReal.ofReal |A.det|
  -- construct a small neighborhood of `A '' (closedBall 0 1)` with measure comparable to
  -- the determinant of `A`.
  obtain ⟨ε, hε, εpos⟩ :
    ∃ ε : ℝ, μ (closedBall 0 ε + A '' closedBall 0 1) < m * μ (closedBall 0 1) ∧ 0 < ε := by
    have HC : IsCompact (A '' closedBall 0 1) :=
      (ProperSpace.isCompact_closedBall _ _).image A.continuous
    have L0 :
      Tendsto (fun ε => μ (cthickening ε (A '' closedBall 0 1))) (𝓝[>] 0)
        (𝓝 (μ (A '' closedBall 0 1))) := by
      apply Tendsto.mono_left _ nhdsWithin_le_nhds
      exact tendsto_measure_cthickening_of_isCompact HC
    have L1 :
      Tendsto (fun ε => μ (closedBall 0 ε + A '' closedBall 0 1)) (𝓝[>] 0)
        (𝓝 (μ (A '' closedBall 0 1))) := by
      apply L0.congr' _
      filter_upwards [self_mem_nhdsWithin] with r hr
      rw [← HC.add_closedBall_zero (le_of_lt hr), add_comm]
    have L2 :
      Tendsto (fun ε => μ (closedBall 0 ε + A '' closedBall 0 1)) (𝓝[>] 0)
        (𝓝 (d * μ (closedBall 0 1))) := by
      convert L1
      exact (addHaar_image_continuousLinearMap _ _ _).symm
    have I : d * μ (closedBall 0 1) < m * μ (closedBall 0 1) :=
      (ENNReal.mul_lt_mul_right (measure_closedBall_pos μ _ zero_lt_one).ne'
            measure_closedBall_lt_top.ne).2
        hm
    have H :
      ∀ᶠ b : ℝ in 𝓝[>] 0, μ (closedBall 0 b + A '' closedBall 0 1) < m * μ (closedBall 0 1) :=
      (tendsto_order.1 L2).2 _ I
    exact (H.and self_mem_nhdsWithin).exists
  /-
    case a.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    d : ENNReal := ENNReal.ofReal (abs A.det)
    ε : Real
    hε : LT.lt (μ (HAdd.hAdd (Metric.closedBall 0 ε) (Set.image (⇑A) (Metric.close …
    εpos : LT.lt 0 ε
    ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
  -/
  have : Iio (⟨ε, εpos.le⟩ : ℝ≥0) ∈ 𝓝 (0 : ℝ≥0) := by apply Iio_mem_nhds; exact εpos
  /-
    case a.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    d : ENNReal := ENNReal.ofReal (abs A.det)
    ε : Real
    hε : LT.lt (μ (HAdd.hAdd (Metric.closedBall 0 ε) (Set.image (⇑A) (Metric.close …
    εpos : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Set.Iio ⟨ε, ⋯⟩)
    ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
  -/
  filter_upwards [this]
  -- fix a function `f` which is close enough to `A`.
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    d : ENNReal := ENNReal.ofReal (abs A.det)
    ε : Real
    hε : LT.lt (μ (HAdd.hAdd (Metric.closedBall 0 ε) (Set.image (⇑A) (Metric.close …
    εpos : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Set.Iio ⟨ε, ⋯⟩)
    ⊢ ∀ (a : NNReal), Membership.mem (Set.Iio ⟨ε, ⋯⟩) a → ∀ (s : Set E) (f : E → E …
  -/
  intro δ hδ s f hf
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    d : ENNReal := ENNReal.ofReal (abs A.det)
    ε : Real
    hε : LT.lt (μ (HAdd.hAdd (Metric.closedBall 0 ε) (Set.image (⇑A) (Metric.close …
    εpos : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Set.Iio ⟨ε, ⋯⟩)
    δ : NNReal
    hδ : Membership.mem (Set.Iio ⟨ε, ⋯⟩) δ
    s : Set E
    f : E → E
    hf : ApproximatesLinearOn f A s δ
    ⊢ LE.le (μ (Set.image f s)) (HMul.hMul (↑m) (μ s))
  -/
  simp only [mem_Iio, ← NNReal.coe_lt_coe, NNReal.coe_mk] at hδ
  -- This function expands the volume of any ball by at most `m`
  have I : ∀ x r, x ∈ s → 0 ≤ r → μ (f '' (s ∩ closedBall x r)) ≤ m * μ (closedBall x r) := by
    intro x r xs r0
    have K : f '' (s ∩ closedBall x r) ⊆ A '' closedBall 0 r + closedBall (f x) (ε * r) := by
      rintro y ⟨z, ⟨zs, zr⟩, rfl⟩
      rw [mem_closedBall_iff_norm] at zr
      apply Set.mem_add.2 ⟨A (z - x), _, f z - f x - A (z - x) + f x, _, _⟩
      · apply mem_image_of_mem
        simpa only [dist_eq_norm, mem_closedBall, mem_closedBall_zero_iff, sub_zero] using zr
      · rw [mem_closedBall_iff_norm, add_sub_cancel_right]
        calc
          ‖f z - f x - A (z - x)‖ ≤ δ * ‖z - x‖ := hf _ zs _ xs
          _ ≤ ε * r := by gcongr
      · simp only [map_sub, Pi.sub_apply]
        abel
    have :
      A '' closedBall 0 r + closedBall (f x) (ε * r) =
        {f x} + r • (A '' closedBall 0 1 + closedBall 0 ε) := by
      rw [smul_add, ← add_assoc, add_comm {f x}, add_assoc, smul_closedBall _ _ εpos.le, smul_zero,
        singleton_add_closedBall_zero, ← image_smul_set ℝ E E A,
        _root_.smul_closedBall _ _ zero_le_one, smul_zero, Real.norm_eq_abs, abs_of_nonneg r0,
        mul_one, mul_comm]
    rw [this] at K
    calc
      μ (f '' (s ∩ closedBall x r)) ≤ μ ({f x} + r • (A '' closedBall 0 1 + closedBall 0 ε)) :=
        measure_mono K
      _ = ENNReal.ofReal (r ^ finrank ℝ E) * μ (A '' closedBall 0 1 + closedBall 0 ε) := by
        simp only [abs_of_nonneg r0, addHaar_smul, image_add_left, abs_pow, singleton_add,
          measure_preimage_add]
      _ ≤ ENNReal.ofReal (r ^ finrank ℝ E) * (m * μ (closedBall 0 1)) := by
        rw [add_comm]; gcongr
      _ = m * μ (closedBall x r) := by simp only [addHaar_closedBall' μ _ r0]; ring
  -- covering `s` by closed balls with total measure very close to `μ s`, one deduces that the
  -- measure of `f '' s` is at most `m * (μ s + a)` for any positive `a`.
  have J : ∀ᶠ a in 𝓝[>] (0 : ℝ≥0∞), μ (f '' s) ≤ m * (μ s + a) := by
    filter_upwards [self_mem_nhdsWithin] with a ha
    rw [mem_Ioi] at ha
    obtain ⟨t, r, t_count, ts, rpos, st, μt⟩ :
      ∃ (t : Set E) (r : E → ℝ),
        t.Countable ∧
          t ⊆ s ∧
            (∀ x : E, x ∈ t → 0 < r x) ∧
              (s ⊆ ⋃ x ∈ t, closedBall x (r x)) ∧
                (∑' x : ↥t, μ (closedBall (↑x) (r ↑x))) ≤ μ s + a :=
      Besicovitch.exists_closedBall_covering_tsum_measure_le μ ha.ne' (fun _ => Ioi 0) s
        fun x _ δ δpos => ⟨δ / 2, by simp [half_pos δpos, δpos]⟩
    haveI : Encodable t := t_count.toEncodable
    calc
      μ (f '' s) ≤ μ (⋃ x : t, f '' (s ∩ closedBall x (r x))) := by
        rw [biUnion_eq_iUnion] at st
        apply measure_mono
        rw [← image_iUnion, ← inter_iUnion]
        exact image_subset _ (subset_inter (Subset.refl _) st)
      _ ≤ ∑' x : t, μ (f '' (s ∩ closedBall x (r x))) := measure_iUnion_le _
      _ ≤ ∑' x : t, m * μ (closedBall x (r x)) :=
        (ENNReal.tsum_le_tsum fun x => I x (r x) (ts x.2) (rpos x x.2).le)
      _ ≤ m * (μ s + a) := by rw [ENNReal.tsum_mul_left]; gcongr
  -- taking the limit in `a`, one obtains the conclusion
  have L : Tendsto (fun a => (m : ℝ≥0∞) * (μ s + a)) (𝓝[>] 0) (𝓝 (m * (μ s + 0))) := by
    apply Tendsto.mono_left _ nhdsWithin_le_nhds
    apply ENNReal.Tendsto.const_mul (tendsto_const_nhds.add tendsto_id)
    simp only [ENNReal.coe_ne_top, Ne, or_true, not_false_iff]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    d : ENNReal := ENNReal.ofReal (abs A.det)
    ε : Real
    hε : LT.lt (μ (HAdd.hAdd (Metric.closedBall 0 ε) (Set.image (⇑A) (Metric.close …
    εpos : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Set.Iio ⟨ε, ⋯⟩)
    δ : NNReal
    s : Set E
    f : E → E
    hf : ApproximatesLinearOn f A s δ
    hδ : LT.lt (↑δ) ε
    I : ∀ (x : E) (r : Real), Membership.mem s x → LE.le 0 r → LE.le (μ (Set.image …
    J : Filter.Eventually (fun a => LE.le (μ (Set.image f s)) (HMul.hMul (↑m) (HAd …
    L : Filter.Tendsto (fun a => HMul.hMul (↑m) (HAdd.hAdd (μ s) a)) (nhdsWithin 0 …
    ⊢ LE.le (μ (Set.image f s)) (HMul.hMul (↑m) (μ s))
  -/
  rw [add_zero] at L
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (ENNReal.ofReal (abs A.det)) ↑m
    d : ENNReal := ENNReal.ofReal (abs A.det)
    ε : Real
    hε : LT.lt (μ (HAdd.hAdd (Metric.closedBall 0 ε) (Set.image (⇑A) (Metric.close …
    εpos : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Set.Iio ⟨ε, ⋯⟩)
    δ : NNReal
    s : Set E
    f : E → E
    hf : ApproximatesLinearOn f A s δ
    hδ : LT.lt (↑δ) ε
    I : ∀ (x : E) (r : Real), Membership.mem s x → LE.le 0 r → LE.le (μ (Set.image …
    J : Filter.Eventually (fun a => LE.le (μ (Set.image f s)) (HMul.hMul (↑m) (HAd …
    L : Filter.Tendsto (fun a => HMul.hMul (↑m) (HAdd.hAdd (μ s) a)) (nhdsWithin 0 …
    ⊢ LE.le (μ (Set.image f s)) (HMul.hMul (↑m) (μ s))
  -/
  exact ge_of_tendsto L J
  /-
    🎉 no goals
  -/


/-- Let `f` be a function which is sufficiently close (in the Lipschitz sense) to a given linear
map `A`. Then it expands the volume of any set by at least `m` for any `m < det A`. -/
theorem mul_le_addHaar_image_of_lt_det (A : E →L[ℝ] E) {m : ℝ≥0}
    (hm : (m : ℝ≥0∞) < ENNReal.ofReal |A.det|) :
    ∀ᶠ δ in 𝓝[>] (0 : ℝ≥0),
      ∀ (s : Set E) (f : E → E), ApproximatesLinearOn f A s δ → (m : ℝ≥0∞) * μ s ≤ μ (f '' s) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    ⊢ Filter.Eventually (fun δ => ∀ (s : Set E) (f : E → E), ApproximatesLinearOn  …
  -/
  apply nhdsWithin_le_nhds
  -- The assumption `hm` implies that `A` is invertible. If `f` is close enough to `A`, it is also
  -- invertible. One can then pass to the inverses, and deduce the estimate from
  -- `addHaar_image_le_mul_of_det_lt` applied to `f⁻¹` and `A⁻¹`.
  -- exclude first the trivial case where `m = 0`.
  /-
    case a
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
  -/
  rcases eq_or_lt_of_le (zero_le m) with (rfl | mpos)
    /-
      case a.inl
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      A : ContinuousLinearMap (RingHom.id Real) E E
      hm : LT.lt (↑0) (ENNReal.ofReal (abs A.det))
      ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
    -/
  · filter_upwards
    /-
      case a.inl.h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      A : ContinuousLinearMap (RingHom.id Real) E E
      hm : LT.lt (↑0) (ENNReal.ofReal (abs A.det))
      ⊢ ∀ (a : NNReal) (s : Set E) (f : E → E), ApproximatesLinearOn f A s a → LE.le …
    -/
    simp only [forall_const, zero_mul, imp_true_iff, zero_le, ENNReal.coe_zero]
    /-
      🎉 no goals
    -/
  have hA : A.det ≠ 0 := by
    intro h; simp only [h, ENNReal.not_lt_zero, ENNReal.ofReal_zero, abs_zero] at hm
  -- let `B` be the continuous linear equiv version of `A`.
  /-
    case a.inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    mpos : LT.lt 0 m
    hA : Ne A.det 0
    ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
  -/
  let B := A.toContinuousLinearEquivOfDetNeZero hA
  -- the determinant of `B.symm` is bounded by `m⁻¹`
  have I : ENNReal.ofReal |(B.symm : E →L[ℝ] E).det| < (m⁻¹ : ℝ≥0) := by
    simp only [ENNReal.ofReal, abs_inv, Real.toNNReal_inv, ContinuousLinearEquiv.det_coe_symm,
      ContinuousLinearMap.coe_toContinuousLinearEquivOfDetNeZero, ENNReal.coe_lt_coe] at hm ⊢
    exact NNReal.inv_lt_inv mpos.ne' hm
  -- therefore, we may apply `addHaar_image_le_mul_of_det_lt` to `B.symm` and `m⁻¹`.
  obtain ⟨δ₀, δ₀pos, hδ₀⟩ :
    ∃ δ : ℝ≥0,
      0 < δ ∧
        ∀ (t : Set E) (g : E → E),
          ApproximatesLinearOn g (B.symm : E →L[ℝ] E) t δ → μ (g '' t) ≤ ↑m⁻¹ * μ t := by
    have :
      ∀ᶠ δ : ℝ≥0 in 𝓝[>] 0,
        ∀ (t : Set E) (g : E → E),
          ApproximatesLinearOn g (B.symm : E →L[ℝ] E) t δ → μ (g '' t) ≤ ↑m⁻¹ * μ t :=
      addHaar_image_le_mul_of_det_lt μ B.symm I
    rcases (this.and self_mem_nhdsWithin).exists with ⟨δ₀, h, h'⟩
    exact ⟨δ₀, h', h⟩
  -- record smallness conditions for `δ` that will be needed to apply `hδ₀` below.
  have L1 : ∀ᶠ δ in 𝓝 (0 : ℝ≥0), Subsingleton E ∨ δ < ‖(B.symm : E →L[ℝ] E)‖₊⁻¹ := by
    by_cases h : Subsingleton E
    · simp only [h, true_or, eventually_const]
    simp only [h, false_or]
    apply Iio_mem_nhds
    simpa only [h, false_or, inv_pos] using B.subsingleton_or_nnnorm_symm_pos
  have L2 :
    ∀ᶠ δ in 𝓝 (0 : ℝ≥0), ‖(B.symm : E →L[ℝ] E)‖₊ * (‖(B.symm : E →L[ℝ] E)‖₊⁻¹ - δ)⁻¹ * δ < δ₀ := by
    have :
      Tendsto (fun δ => ‖(B.symm : E →L[ℝ] E)‖₊ * (‖(B.symm : E →L[ℝ] E)‖₊⁻¹ - δ)⁻¹ * δ) (𝓝 0)
        (𝓝 (‖(B.symm : E →L[ℝ] E)‖₊ * (‖(B.symm : E →L[ℝ] E)‖₊⁻¹ - 0)⁻¹ * 0)) := by
      rcases eq_or_ne ‖(B.symm : E →L[ℝ] E)‖₊ 0 with (H | H)
      · simpa only [H, zero_mul] using tendsto_const_nhds
      refine Tendsto.mul (tendsto_const_nhds.mul ?_) tendsto_id
      refine (Tendsto.sub tendsto_const_nhds tendsto_id).inv₀ ?_
      simpa only [tsub_zero, inv_eq_zero, Ne] using H
    simp only [mul_zero] at this
    exact (tendsto_order.1 this).2 δ₀ δ₀pos
  -- let `δ` be small enough, and `f` approximated by `B` up to `δ`.
  /-
    case a.inr.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    mpos : LT.lt 0 m
    hA : Ne A.det 0
    B : ContinuousLinearEquiv (RingHom.id Real) E E := A.toContinuousLinearEquivOf …
    I : LT.lt (ENNReal.ofReal (abs (↑B.symm).det)) ↑(Inv.inv m)
    δ₀ : NNReal
    δ₀pos : LT.lt 0 δ₀
    hδ₀ : ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g (↑B.symm) t δ₀ → LE.le …
    L1 : Filter.Eventually (fun δ => Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm …
    L2 : Filter.Eventually (fun δ => LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B …
    ⊢ Membership.mem (nhds 0) (setOf fun x => (fun δ => ∀ (s : Set E) (f : E → E), …
  -/
  filter_upwards [L1, L2]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    mpos : LT.lt 0 m
    hA : Ne A.det 0
    B : ContinuousLinearEquiv (RingHom.id Real) E E := A.toContinuousLinearEquivOf …
    I : LT.lt (ENNReal.ofReal (abs (↑B.symm).det)) ↑(Inv.inv m)
    δ₀ : NNReal
    δ₀pos : LT.lt 0 δ₀
    hδ₀ : ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g (↑B.symm) t δ₀ → LE.le …
    L1 : Filter.Eventually (fun δ => Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm …
    L2 : Filter.Eventually (fun δ => LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B …
    ⊢ ∀ (a : NNReal), Or (Subsingleton E) (LT.lt a (Inv.inv (NNNorm.nnnorm ↑B.symm …
  -/
  intro δ h1δ h2δ s f hf
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    mpos : LT.lt 0 m
    hA : Ne A.det 0
    B : ContinuousLinearEquiv (RingHom.id Real) E E := A.toContinuousLinearEquivOf …
    I : LT.lt (ENNReal.ofReal (abs (↑B.symm).det)) ↑(Inv.inv m)
    δ₀ : NNReal
    δ₀pos : LT.lt 0 δ₀
    hδ₀ : ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g (↑B.symm) t δ₀ → LE.le …
    L1 : Filter.Eventually (fun δ => Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm …
    L2 : Filter.Eventually (fun δ => LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B …
    δ : NNReal
    h1δ : Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm.nnnorm ↑B.symm)))
    h2δ : LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B.symm) (Inv.inv (HSub.hSub  …
    s : Set E
    f : E → E
    hf : ApproximatesLinearOn f A s δ
    ⊢ LE.le (HMul.hMul (↑m) (μ s)) (μ (Set.image f s))
  -/
  have hf' : ApproximatesLinearOn f (B : E →L[ℝ] E) s δ := by convert hf
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    mpos : LT.lt 0 m
    hA : Ne A.det 0
    B : ContinuousLinearEquiv (RingHom.id Real) E E := A.toContinuousLinearEquivOf …
    I : LT.lt (ENNReal.ofReal (abs (↑B.symm).det)) ↑(Inv.inv m)
    δ₀ : NNReal
    δ₀pos : LT.lt 0 δ₀
    hδ₀ : ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g (↑B.symm) t δ₀ → LE.le …
    L1 : Filter.Eventually (fun δ => Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm …
    L2 : Filter.Eventually (fun δ => LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B …
    δ : NNReal
    h1δ : Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm.nnnorm ↑B.symm)))
    h2δ : LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B.symm) (Inv.inv (HSub.hSub  …
    s : Set E
    f : E → E
    hf : ApproximatesLinearOn f A s δ
    hf' : ApproximatesLinearOn f (↑B) s δ
    ⊢ LE.le (HMul.hMul (↑m) (μ s)) (μ (Set.image f s))
  -/
  let F := hf'.toPartialEquiv h1δ
  -- the condition to be checked can be reformulated in terms of the inverse maps
  suffices H : μ (F.symm '' F.target) ≤ (m⁻¹ : ℝ≥0) * μ F.target by
    change (m : ℝ≥0∞) * μ F.source ≤ μ F.target
    rwa [← F.symm_image_target_eq_source, mul_comm, ← ENNReal.le_div_iff_mul_le, div_eq_mul_inv,
      mul_comm, ← ENNReal.coe_inv mpos.ne']
    · apply Or.inl
      simpa only [ENNReal.coe_eq_zero, Ne] using mpos.ne'
    · simp only [ENNReal.coe_ne_top, true_or, Ne, not_false_iff]
  -- as `f⁻¹` is well approximated by `B⁻¹`, the conclusion follows from `hδ₀`
  -- and our choice of `δ`.
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    m : NNReal
    hm : LT.lt (↑m) (ENNReal.ofReal (abs A.det))
    mpos : LT.lt 0 m
    hA : Ne A.det 0
    B : ContinuousLinearEquiv (RingHom.id Real) E E := A.toContinuousLinearEquivOf …
    I : LT.lt (ENNReal.ofReal (abs (↑B.symm).det)) ↑(Inv.inv m)
    δ₀ : NNReal
    δ₀pos : LT.lt 0 δ₀
    hδ₀ : ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g (↑B.symm) t δ₀ → LE.le …
    L1 : Filter.Eventually (fun δ => Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm …
    L2 : Filter.Eventually (fun δ => LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B …
    δ : NNReal
    h1δ : Or (Subsingleton E) (LT.lt δ (Inv.inv (NNNorm.nnnorm ↑B.symm)))
    h2δ : LT.lt (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑B.symm) (Inv.inv (HSub.hSub  …
    s : Set E
    f : E → E
    hf : ApproximatesLinearOn f A s δ
    hf' : ApproximatesLinearOn f (↑B) s δ
    F : PartialEquiv E E := hf'.toPartialEquiv h1δ
    ⊢ LE.le (μ (Set.image (↑F.symm) F.target)) (HMul.hMul (↑(Inv.inv m)) (μ F.targ …
  -/
  exact hδ₀ _ _ ((hf'.to_inv h1δ).mono_num h2δ.le)
  /-
    🎉 no goals
  -/


/-- If a differentiable function `f` is approximated by a linear map `A` on a set `s`, up to `δ`,
then at almost every `x` in `s` one has `‖f' x - A‖ ≤ δ`. -/
theorem _root_.ApproximatesLinearOn.norm_fderiv_sub_le {A : E →L[ℝ] E} {δ : ℝ≥0}
    (hf : ApproximatesLinearOn f A s δ) (hs : MeasurableSet s) (f' : E → E →L[ℝ] E)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) : ∀ᵐ x ∂μ.restrict s, ‖f' x - A‖₊ ≤ δ := by
  /- The conclusion will hold at the Lebesgue density points of `s` (which have full measure).
    At such a point `x`, for any `z` and any `ε > 0` one has for small `r`
    that `{x} + r • closedBall z ε` intersects `s`. At a point `y` in the intersection,
    `f y - f x` is close both to `f' x (r z)` (by differentiability) and to `A (r z)`
    (by linear approximation), so these two quantities are close, i.e., `(f' x - A) z` is small. -/
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    δ : NNReal
    hf : ApproximatesLinearOn f A s δ
    hs : MeasurableSet s
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (HSub.hSub (f' x) A)) δ) (M …
  -/
  filter_upwards [Besicovitch.ae_tendsto_measure_inter_div μ s, ae_restrict_mem hs]
  -- start from a Lebesgue density point `x`, belonging to `s`.
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    δ : NNReal
    hf : ApproximatesLinearOn f A s δ
    hs : MeasurableSet s
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ ∀ (a : E), Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.clos …
  -/
  intro x hx xs
  -- consider an arbitrary vector `z`.
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    δ : NNReal
    hf : ApproximatesLinearOn f A s δ
    hs : MeasurableSet s
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    x : E
    hx : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x …
    xs : Membership.mem s x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub (f' x) A)) δ
  -/
  apply ContinuousLinearMap.opNorm_le_bound _ δ.2 fun z => ?_
  -- to show that `‖(f' x - A) z‖ ≤ δ ‖z‖`, it suffices to do it up to some error that vanishes
  -- asymptotically in terms of `ε > 0`.
  suffices H : ∀ ε, 0 < ε → ‖(f' x - A) z‖ ≤ (δ + ε) * (‖z‖ + ε) + ‖f' x - A‖ * ε by
    have :
      Tendsto (fun ε : ℝ => ((δ : ℝ) + ε) * (‖z‖ + ε) + ‖f' x - A‖ * ε) (𝓝[>] 0)
        (𝓝 ((δ + 0) * (‖z‖ + 0) + ‖f' x - A‖ * 0)) :=
      Tendsto.mono_left (Continuous.tendsto (by fun_prop) 0) nhdsWithin_le_nhds
    simp only [add_zero, mul_zero] at this
    apply le_of_tendsto_of_tendsto tendsto_const_nhds this
    filter_upwards [self_mem_nhdsWithin]
    exact H
  -- fix a positive `ε`.
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    A : ContinuousLinearMap (RingHom.id Real) E E
    δ : NNReal
    hf : ApproximatesLinearOn f A s δ
    hs : MeasurableSet s
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    x : E
    hx : Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x …
    xs : Membership.mem s x
    z : E
    ⊢ ∀ (ε : Real), LT.lt 0 ε → LE.le (Norm.norm ((HSub.hSub (f' x) A) z)) (HAdd.h …
  -/
  intro ε εpos
  -- for small enough `r`, the rescaled ball `r • closedBall z ε` intersects `s`, as `x` is a
  -- density point
  have B₁ : ∀ᶠ r in 𝓝[>] (0 : ℝ), (s ∩ ({x} + r • closedBall z ε)).Nonempty :=
    eventually_nonempty_inter_smul_of_density_one μ s x hx _ measurableSet_closedBall
      (measure_closedBall_pos μ z εpos).ne'
  obtain ⟨ρ, ρpos, hρ⟩ :
    ∃ ρ > 0, ball x ρ ∩ s ⊆ {y : E | ‖f y - f x - (f' x) (y - x)‖ ≤ ε * ‖y - x‖} :=
    mem_nhdsWithin_iff.1 ((hf' x xs).isLittleO.def εpos)
  -- for small enough `r`, the rescaled ball `r • closedBall z ε` is included in the set where
  -- `f y - f x` is well approximated by `f' x (y - x)`.
  have B₂ : ∀ᶠ r in 𝓝[>] (0 : ℝ), {x} + r • closedBall z ε ⊆ ball x ρ := by
    apply nhdsWithin_le_nhds
    exact eventually_singleton_add_smul_subset isBounded_closedBall (ball_mem_nhds x ρpos)
  -- fix a small positive `r` satisfying the above properties, as well as a corresponding `y`.
  obtain ⟨r, ⟨y, ⟨ys, hy⟩⟩, rρ, rpos⟩ :
    ∃ r : ℝ,
      (s ∩ ({x} + r • closedBall z ε)).Nonempty ∧ {x} + r • closedBall z ε ⊆ ball x ρ ∧ 0 < r :=
    (B₁.and (B₂.and self_mem_nhdsWithin)).exists
  -- write `y = x + r a` with `a ∈ closedBall z ε`.
  obtain ⟨a, az, ya⟩ : ∃ a, a ∈ closedBall z ε ∧ y = x + r • a := by
    simp only [mem_smul_set, image_add_left, mem_preimage, singleton_add] at hy
    rcases hy with ⟨a, az, ha⟩
    exact ⟨a, az, by simp only [ha, add_neg_cancel_left]⟩
  have norm_a : ‖a‖ ≤ ‖z‖ + ε :=
    calc
      ‖a‖ = ‖z + (a - z)‖ := by simp only [_root_.add_sub_cancel]
      _ ≤ ‖z‖ + ‖a - z‖ := norm_add_le _ _
      _ ≤ ‖z‖ + ε := add_le_add_left (mem_closedBall_iff_norm.1 az) _
  -- use the approximation properties to control `(f' x - A) a`, and then `(f' x - A) z` as `z` is
  -- close to `a`.
  have I : r * ‖(f' x - A) a‖ ≤ r * (δ + ε) * (‖z‖ + ε) :=
    calc
      r * ‖(f' x - A) a‖ = ‖(f' x - A) (r • a)‖ := by
        simp only [ContinuousLinearMap.map_smul, norm_smul, Real.norm_eq_abs, abs_of_nonneg rpos.le]
      _ = ‖f y - f x - A (y - x) - (f y - f x - (f' x) (y - x))‖ := by
        congr 1
        simp only [ya, add_sub_cancel_left, sub_sub_sub_cancel_left, ContinuousLinearMap.coe_sub',
          eq_self_iff_true, sub_left_inj, Pi.sub_apply, ContinuousLinearMap.map_smul, smul_sub]
      _ ≤ ‖f y - f x - A (y - x)‖ + ‖f y - f x - (f' x) (y - x)‖ := norm_sub_le _ _
      _ ≤ δ * ‖y - x‖ + ε * ‖y - x‖ := (add_le_add (hf _ ys _ xs) (hρ ⟨rρ hy, ys⟩))
      _ = r * (δ + ε) * ‖a‖ := by
        simp only [ya, add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg rpos.le]
        ring
      _ ≤ r * (δ + ε) * (‖z‖ + ε) := by gcongr
  calc
    ‖(f' x - A) z‖ = ‖(f' x - A) a + (f' x - A) (z - a)‖ := by
      congr 1
      simp only [ContinuousLinearMap.coe_sub', map_sub, Pi.sub_apply]
      abel
    _ ≤ ‖(f' x - A) a‖ + ‖(f' x - A) (z - a)‖ := norm_add_le _ _
    _ ≤ (δ + ε) * (‖z‖ + ε) + ‖f' x - A‖ * ‖z - a‖ := by
      apply add_le_add
      · rw [mul_assoc] at I; exact (mul_le_mul_left rpos).1 I
      · apply ContinuousLinearMap.le_opNorm
    _ ≤ (δ + ε) * (‖z‖ + ε) + ‖f' x - A‖ * ε := by
      rw [mem_closedBall_iff_norm'] at az
      gcongr


/-- A differentiable function maps sets of measure zero to sets of measure zero. -/
theorem addHaar_image_eq_zero_of_differentiableOn_of_addHaar_eq_zero (hf : DifferentiableOn ℝ f s)
    (hs : μ s = 0) : μ (f '' s) = 0 := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf : DifferentiableOn Real f s
    hs : Eq (μ s) 0
    ⊢ Eq (μ (Set.image f s)) 0
  -/
  refine le_antisymm ?_ (zero_le _)
  have :
      ∀ A : E →L[ℝ] E, ∃ δ : ℝ≥0, 0 < δ ∧
        ∀ (t : Set E), ApproximatesLinearOn f A t δ →
          μ (f '' t) ≤ (Real.toNNReal |A.det| + 1 : ℝ≥0) * μ t := by
    intro A
    let m : ℝ≥0 := Real.toNNReal |A.det| + 1
    have I : ENNReal.ofReal |A.det| < m := by
      simp only [m, ENNReal.ofReal, lt_add_iff_pos_right, zero_lt_one, ENNReal.coe_lt_coe]
    rcases ((addHaar_image_le_mul_of_det_lt μ A I).and self_mem_nhdsWithin).exists with ⟨δ, h, h'⟩
    exact ⟨δ, h', fun t ht => h t f ht⟩
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf : DifferentiableOn Real f s
    hs : Eq (μ s) 0
    this : ∀ (A : ContinuousLinearMap (RingHom.id Real) E E), Exists fun δ => And  …
    ⊢ LE.le (μ (Set.image f s)) 0
  -/
  choose δ hδ using this
  obtain ⟨t, A, _, _, t_cover, ht, -⟩ :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] E),
      Pairwise (Disjoint on t) ∧
        (∀ n : ℕ, MeasurableSet (t n)) ∧
          (s ⊆ ⋃ n : ℕ, t n) ∧
            (∀ n : ℕ, ApproximatesLinearOn f (A n) (s ∩ t n) (δ (A n))) ∧
              (s.Nonempty → ∀ n, ∃ y ∈ s, A n = fderivWithin ℝ f s y) :=
    exists_partition_approximatesLinearOn_of_hasFDerivWithinAt f s (fderivWithin ℝ f s)
      (fun x xs => (hf x xs).hasFDerivWithinAt) δ fun A => (hδ A).1.ne'
  calc
    μ (f '' s) ≤ μ (⋃ n, f '' (s ∩ t n)) := by
      apply measure_mono
      rw [← image_iUnion, ← inter_iUnion]
      exact image_subset f (subset_inter Subset.rfl t_cover)
    _ ≤ ∑' n, μ (f '' (s ∩ t n)) := measure_iUnion_le _
    _ ≤ ∑' n, (Real.toNNReal |(A n).det| + 1 : ℝ≥0) * μ (s ∩ t n) := by
      apply ENNReal.tsum_le_tsum fun n => ?_
      apply (hδ (A n)).2
      exact ht n
    _ ≤ ∑' n, ((Real.toNNReal |(A n).det| + 1 : ℝ≥0) : ℝ≥0∞) * 0 := by
      refine ENNReal.tsum_le_tsum fun n => mul_le_mul_left' ?_ _
      exact le_trans (measure_mono inter_subset_left) (le_of_eq hs)
    _ = 0 := by simp only [tsum_zero, mul_zero]


/-- A version of **Sard's lemma** in fixed dimension: given a differentiable function from `E`
to `E` and a set where the differential is not invertible, then the image of this set has
zero measure. Here, we give an auxiliary statement towards this result. -/
theorem addHaar_image_eq_zero_of_det_fderivWithin_eq_zero_aux
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (R : ℝ) (hs : s ⊆ closedBall 0 R) (ε : ℝ≥0)
    (εpos : 0 < ε) (h'f' : ∀ x ∈ s, (f' x).det = 0) : μ (f '' s) ≤ ε * μ (closedBall 0 R) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    R : Real
    hs : HasSubset.Subset s (Metric.closedBall 0 R)
    ε : NNReal
    εpos : LT.lt 0 ε
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    ⊢ LE.le (μ (Set.image f s)) (HMul.hMul (↑ε) (μ (Metric.closedBall 0 R)))
  -/
  rcases eq_empty_or_nonempty s with (rfl | h's); · simp only [measure_empty, zero_le, image_empty]
                                                    /-
                                                      🎉 no goals
                                                    -/
  have :
      ∀ A : E →L[ℝ] E, ∃ δ : ℝ≥0, 0 < δ ∧
        ∀ (t : Set E), ApproximatesLinearOn f A t δ →
          μ (f '' t) ≤ (Real.toNNReal |A.det| + ε : ℝ≥0) * μ t := by
    intro A
    let m : ℝ≥0 := Real.toNNReal |A.det| + ε
    have I : ENNReal.ofReal |A.det| < m := by
      simp only [m, ENNReal.ofReal, lt_add_iff_pos_right, εpos, ENNReal.coe_lt_coe]
    rcases ((addHaar_image_le_mul_of_det_lt μ A I).and self_mem_nhdsWithin).exists with ⟨δ, h, h'⟩
    exact ⟨δ, h', fun t ht => h t f ht⟩
  /-
    case inr
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    R : Real
    hs : HasSubset.Subset s (Metric.closedBall 0 R)
    ε : NNReal
    εpos : LT.lt 0 ε
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    h's : s.Nonempty
    this : ∀ (A : ContinuousLinearMap (RingHom.id Real) E E), Exists fun δ => And  …
    ⊢ LE.le (μ (Set.image f s)) (HMul.hMul (↑ε) (μ (Metric.closedBall 0 R)))
  -/
  choose δ hδ using this
  obtain ⟨t, A, t_disj, t_meas, t_cover, ht, Af'⟩ :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] E),
      Pairwise (Disjoint on t) ∧
        (∀ n : ℕ, MeasurableSet (t n)) ∧
          (s ⊆ ⋃ n : ℕ, t n) ∧
            (∀ n : ℕ, ApproximatesLinearOn f (A n) (s ∩ t n) (δ (A n))) ∧
              (s.Nonempty → ∀ n, ∃ y ∈ s, A n = f' y) :=
    exists_partition_approximatesLinearOn_of_hasFDerivWithinAt f s f' hf' δ fun A => (hδ A).1.ne'
  calc
    μ (f '' s) ≤ μ (⋃ n, f '' (s ∩ t n)) := by
      rw [← image_iUnion, ← inter_iUnion]
      gcongr
      exact subset_inter Subset.rfl t_cover
    _ ≤ ∑' n, μ (f '' (s ∩ t n)) := measure_iUnion_le _
    _ ≤ ∑' n, (Real.toNNReal |(A n).det| + ε : ℝ≥0) * μ (s ∩ t n) := by
      gcongr
      exact (hδ (A _)).2 _ (ht _)
    _ = ∑' n, ε * μ (s ∩ t n) := by
      congr with n
      rcases Af' h's n with ⟨y, ys, hy⟩
      simp only [hy, h'f' y ys, Real.toNNReal_zero, abs_zero, zero_add]
    _ ≤ ε * ∑' n, μ (closedBall 0 R ∩ t n) := by
      rw [ENNReal.tsum_mul_left]
      gcongr
    _ = ε * μ (⋃ n, closedBall 0 R ∩ t n) := by
      rw [measure_iUnion]
      · exact pairwise_disjoint_mono t_disj fun n => inter_subset_right
      · intro n
        exact measurableSet_closedBall.inter (t_meas n)
    _ ≤ ε * μ (closedBall 0 R) := by
      rw [← inter_iUnion]
      exact mul_le_mul_left' (measure_mono inter_subset_left) _


/-- A version of Sard lemma in fixed dimension: given a differentiable function from `E` to `E` and
a set where the differential is not invertible, then the image of this set has zero measure. -/
theorem addHaar_image_eq_zero_of_det_fderivWithin_eq_zero
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (h'f' : ∀ x ∈ s, (f' x).det = 0) :
    μ (f '' s) = 0 := by
  suffices H : ∀ R, μ (f '' (s ∩ closedBall 0 R)) = 0 by
    apply le_antisymm _ (zero_le _)
    rw [← iUnion_inter_closedBall_nat s 0]
    calc
      μ (f '' ⋃ n : ℕ, s ∩ closedBall 0 n) ≤ ∑' n : ℕ, μ (f '' (s ∩ closedBall 0 n)) := by
        rw [image_iUnion]; exact measure_iUnion_le _
      _ ≤ 0 := by simp only [H, tsum_zero, nonpos_iff_eq_zero]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    ⊢ ∀ (R : Real), Eq (μ (Set.image f (Inter.inter s (Metric.closedBall 0 R)))) 0
  -/
  intro R
  have A : ∀ (ε : ℝ≥0), 0 < ε → μ (f '' (s ∩ closedBall 0 R)) ≤ ε * μ (closedBall 0 R) :=
    fun ε εpos =>
    addHaar_image_eq_zero_of_det_fderivWithin_eq_zero_aux μ
      (fun x hx => (hf' x hx.1).mono inter_subset_left) R inter_subset_right ε εpos
      fun x hx => h'f' x hx.1
  have B : Tendsto (fun ε : ℝ≥0 => (ε : ℝ≥0∞) * μ (closedBall 0 R)) (𝓝[>] 0) (𝓝 0) := by
    have :
      Tendsto (fun ε : ℝ≥0 => (ε : ℝ≥0∞) * μ (closedBall 0 R)) (𝓝 0)
        (𝓝 (((0 : ℝ≥0) : ℝ≥0∞) * μ (closedBall 0 R))) :=
      ENNReal.Tendsto.mul_const (ENNReal.tendsto_coe.2 tendsto_id)
        (Or.inr measure_closedBall_lt_top.ne)
    simp only [zero_mul, ENNReal.coe_zero] at this
    exact Tendsto.mono_left this nhdsWithin_le_nhds
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    R : Real
    A : ∀ (ε : NNReal), LT.lt 0 ε → LE.le (μ (Set.image f (Inter.inter s (Metric.c …
    B : Filter.Tendsto (fun ε => HMul.hMul (↑ε) (μ (Metric.closedBall 0 R))) (nhds …
    ⊢ Eq (μ (Set.image f (Inter.inter s (Metric.closedBall 0 R)))) 0
  -/
  apply le_antisymm _ (zero_le _)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    R : Real
    A : ∀ (ε : NNReal), LT.lt 0 ε → LE.le (μ (Set.image f (Inter.inter s (Metric.c …
    B : Filter.Tendsto (fun ε => HMul.hMul (↑ε) (μ (Metric.closedBall 0 R))) (nhds …
    ⊢ LE.le (μ (Set.image f (Inter.inter s (Metric.closedBall 0 R)))) 0
  -/
  apply ge_of_tendsto B
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    R : Real
    A : ∀ (ε : NNReal), LT.lt 0 ε → LE.le (μ (Set.image f (Inter.inter s (Metric.c …
    B : Filter.Tendsto (fun ε => HMul.hMul (↑ε) (μ (Metric.closedBall 0 R))) (nhds …
    ⊢ Filter.Eventually (fun c => LE.le (μ (Set.image f (Inter.inter s (Metric.clo …
  -/
  filter_upwards [self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    h'f' : ∀ (x : E), Membership.mem s x → Eq (f' x).det 0
    R : Real
    A : ∀ (ε : NNReal), LT.lt 0 ε → LE.le (μ (Set.image f (Inter.inter s (Metric.c …
    B : Filter.Tendsto (fun ε => HMul.hMul (↑ε) (μ (Metric.closedBall 0 R))) (nhds …
    ⊢ ∀ (a : NNReal), Membership.mem (Set.Ioi 0) a → LE.le (μ (Set.image f (Inter. …
  -/
  exact A
  /-
    🎉 no goals
  -/


/-- The derivative of a function on a measurable set is almost everywhere measurable on this set
with respect to Lebesgue measure. Note that, in general, it is not genuinely measurable there,
as `f'` is not unique (but only on a set of measure `0`, as the argument shows). -/
theorem aemeasurable_fderivWithin (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) : AEMeasurable f' (μ.restrict s) := by
  /- It suffices to show that `f'` can be uniformly approximated by a measurable function.
    Fix `ε > 0`. Thanks to `exists_partition_approximatesLinearOn_of_hasFDerivWithinAt`, one
    can find a countable measurable partition of `s` into sets `s ∩ t n` on which `f` is well
    approximated by linear maps `A n`. On almost all of `s ∩ t n`, it follows from
    `ApproximatesLinearOn.norm_fderiv_sub_le` that `f'` is uniformly approximated by `A n`, which
    gives the conclusion. -/
  -- fix a precision `ε`
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable f' (μ.restrict s)
  -/
  refine aemeasurable_of_unif_approx fun ε εpos => ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    ⊢ Exists fun f => And (AEMeasurable f (μ.restrict s)) (Filter.Eventually (fun  …
  -/
  let δ : ℝ≥0 := ⟨ε, le_of_lt εpos⟩
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    ⊢ Exists fun f => And (AEMeasurable f (μ.restrict s)) (Filter.Eventually (fun  …
  -/
  have δpos : 0 < δ := εpos
  -- partition `s` into sets `s ∩ t n` on which `f` is approximated by linear maps `A n`.
  obtain ⟨t, A, t_disj, t_meas, t_cover, ht, _⟩ :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] E),
      Pairwise (Disjoint on t) ∧
        (∀ n : ℕ, MeasurableSet (t n)) ∧
          (s ⊆ ⋃ n : ℕ, t n) ∧
            (∀ n : ℕ, ApproximatesLinearOn f (A n) (s ∩ t n) δ) ∧
              (s.Nonempty → ∀ n, ∃ y ∈ s, A n = f' y) :=
    exists_partition_approximatesLinearOn_of_hasFDerivWithinAt f s f' hf' (fun _ => δ) fun _ =>
      δpos.ne'
  -- define a measurable function `g` which coincides with `A n` on `t n`.
  obtain ⟨g, g_meas, hg⟩ :
      ∃ g : E → E →L[ℝ] E, Measurable g ∧ ∀ (n : ℕ) (x : E), x ∈ t n → g x = A n :=
    exists_measurable_piecewise t t_meas (fun n _ => A n) (fun n => measurable_const) <|
      t_disj.mono fun i j h => by simp only [h.inter_eq, eqOn_empty]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    δpos : LT.lt 0 δ
    t : Nat → Set E
    A : Nat → ContinuousLinearMap (RingHom.id Real) E E
    t_disj : Pairwise (Function.onFun Disjoint t)
    t_meas : ∀ (n : Nat), MeasurableSet (t n)
    t_cover : HasSubset.Subset s (Set.iUnion fun n => t n)
    ht : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) δ
    right✝ : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (E …
    g : E → ContinuousLinearMap (RingHom.id Real) E E
    g_meas : Measurable g
    hg : ∀ (n : Nat) (x : E), Membership.mem (t n) x → Eq (g x) (A n)
    ⊢ Exists fun f => And (AEMeasurable f (μ.restrict s)) (Filter.Eventually (fun  …
  -/
  refine ⟨g, g_meas.aemeasurable, ?_⟩
  -- reduce to checking that `f'` and `g` are close on almost all of `s ∩ t n`, for all `n`.
  suffices H : ∀ᵐ x : E ∂sum fun n ↦ μ.restrict (s ∩ t n), dist (g x) (f' x) ≤ ε by
    have : μ.restrict s ≤ sum fun n => μ.restrict (s ∩ t n) := by
      have : s = ⋃ n, s ∩ t n := by
        rw [← inter_iUnion]
        exact Subset.antisymm (subset_inter Subset.rfl t_cover) inter_subset_left
      conv_lhs => rw [this]
      exact restrict_iUnion_le
    exact ae_mono this H
  -- fix such an `n`.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    δpos : LT.lt 0 δ
    t : Nat → Set E
    A : Nat → ContinuousLinearMap (RingHom.id Real) E E
    t_disj : Pairwise (Function.onFun Disjoint t)
    t_meas : ∀ (n : Nat), MeasurableSet (t n)
    t_cover : HasSubset.Subset s (Set.iUnion fun n => t n)
    ht : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) δ
    right✝ : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (E …
    g : E → ContinuousLinearMap (RingHom.id Real) E E
    g_meas : Measurable g
    hg : ∀ (n : Nat) (x : E), Membership.mem (t n) x → Eq (g x) (A n)
    ⊢ Filter.Eventually (fun x => LE.le (Dist.dist (g x) (f' x)) ε) (MeasureTheory …
  -/
  refine ae_sum_iff.2 fun n => ?_
  -- on almost all `s ∩ t n`, `f' x` is close to `A n` thanks to
  -- `ApproximatesLinearOn.norm_fderiv_sub_le`.
  have E₁ : ∀ᵐ x : E ∂μ.restrict (s ∩ t n), ‖f' x - A n‖₊ ≤ δ :=
    (ht n).norm_fderiv_sub_le μ (hs.inter (t_meas n)) f' fun x hx =>
      (hf' x hx.1).mono inter_subset_left
  -- moreover, `g x` is equal to `A n` there.
  have E₂ : ∀ᵐ x : E ∂μ.restrict (s ∩ t n), g x = A n := by
    suffices H : ∀ᵐ x : E ∂μ.restrict (t n), g x = A n from
      ae_mono (restrict_mono inter_subset_right le_rfl) H
    filter_upwards [ae_restrict_mem (t_meas n)]
    exact hg n
  -- putting these two properties together gives the conclusion.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    δpos : LT.lt 0 δ
    t : Nat → Set E
    A : Nat → ContinuousLinearMap (RingHom.id Real) E E
    t_disj : Pairwise (Function.onFun Disjoint t)
    t_meas : ∀ (n : Nat), MeasurableSet (t n)
    t_cover : HasSubset.Subset s (Set.iUnion fun n => t n)
    ht : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) δ
    right✝ : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (E …
    g : E → ContinuousLinearMap (RingHom.id Real) E E
    g_meas : Measurable g
    hg : ∀ (n : Nat) (x : E), Membership.mem (t n) x → Eq (g x) (A n)
    n : Nat
    E₁ : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (HSub.hSub (f' x) (A n)) …
    E₂ : Filter.Eventually (fun x => Eq (g x) (A n)) (MeasureTheory.ae (μ.restrict …
    ⊢ Filter.Eventually (fun x => LE.le (Dist.dist (g x) (f' x)) ε) (MeasureTheory …
  -/
  filter_upwards [E₁, E₂] with x hx1 hx2
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    δpos : LT.lt 0 δ
    t : Nat → Set E
    A : Nat → ContinuousLinearMap (RingHom.id Real) E E
    t_disj : Pairwise (Function.onFun Disjoint t)
    t_meas : ∀ (n : Nat), MeasurableSet (t n)
    t_cover : HasSubset.Subset s (Set.iUnion fun n => t n)
    ht : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) δ
    right✝ : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (E …
    g : E → ContinuousLinearMap (RingHom.id Real) E E
    g_meas : Measurable g
    hg : ∀ (n : Nat) (x : E), Membership.mem (t n) x → Eq (g x) (A n)
    n : Nat
    E₁ : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (HSub.hSub (f' x) (A n)) …
    E₂ : Filter.Eventually (fun x => Eq (g x) (A n)) (MeasureTheory.ae (μ.restrict …
    x : E
    hx1 : LE.le (NNNorm.nnnorm (HSub.hSub (f' x) (A n))) δ
    hx2 : Eq (g x) (A n)
    ⊢ LE.le (Dist.dist (g x) (f' x)) ε
  -/
  rw [← nndist_eq_nnnorm] at hx1
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    δpos : LT.lt 0 δ
    t : Nat → Set E
    A : Nat → ContinuousLinearMap (RingHom.id Real) E E
    t_disj : Pairwise (Function.onFun Disjoint t)
    t_meas : ∀ (n : Nat), MeasurableSet (t n)
    t_cover : HasSubset.Subset s (Set.iUnion fun n => t n)
    ht : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) δ
    right✝ : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (E …
    g : E → ContinuousLinearMap (RingHom.id Real) E E
    g_meas : Measurable g
    hg : ∀ (n : Nat) (x : E), Membership.mem (t n) x → Eq (g x) (A n)
    n : Nat
    E₁ : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (HSub.hSub (f' x) (A n)) …
    E₂ : Filter.Eventually (fun x => Eq (g x) (A n)) (MeasureTheory.ae (μ.restrict …
    x : E
    hx1 : LE.le (NNDist.nndist (f' x) (A n)) δ
    hx2 : Eq (g x) (A n)
    ⊢ LE.le (Dist.dist (g x) (f' x)) ε
  -/
  rw [hx2, dist_comm]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : Real
    εpos : GT.gt ε 0
    δ : NNReal := ⟨ε, ⋯⟩
    δpos : LT.lt 0 δ
    t : Nat → Set E
    A : Nat → ContinuousLinearMap (RingHom.id Real) E E
    t_disj : Pairwise (Function.onFun Disjoint t)
    t_meas : ∀ (n : Nat), MeasurableSet (t n)
    t_cover : HasSubset.Subset s (Set.iUnion fun n => t n)
    ht : ∀ (n : Nat), ApproximatesLinearOn f (A n) (Inter.inter s (t n)) δ
    right✝ : s.Nonempty → ∀ (n : Nat), Exists fun y => And (Membership.mem s y) (E …
    g : E → ContinuousLinearMap (RingHom.id Real) E E
    g_meas : Measurable g
    hg : ∀ (n : Nat) (x : E), Membership.mem (t n) x → Eq (g x) (A n)
    n : Nat
    E₁ : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (HSub.hSub (f' x) (A n)) …
    E₂ : Filter.Eventually (fun x => Eq (g x) (A n)) (MeasureTheory.ae (μ.restrict …
    x : E
    hx1 : LE.le (NNDist.nndist (f' x) (A n)) δ
    hx2 : Eq (g x) (A n)
    ⊢ LE.le (Dist.dist (f' x) (A n)) ε
  -/
  exact hx1
  /-
    🎉 no goals
  -/


theorem aemeasurable_ofReal_abs_det_fderivWithin (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) :
    AEMeasurable (fun x => ENNReal.ofReal |(f' x).det|) (μ.restrict s) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable (fun x => ENNReal.ofReal (abs (f' x).det)) (μ.restrict s)
  -/
  apply ENNReal.measurable_ofReal.comp_aemeasurable
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable (fun x => abs (f' x).det) (μ.restrict s)
  -/
  refine continuous_abs.measurable.comp_aemeasurable ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable (fun x => (f' x).det) (μ.restrict s)
  -/
  refine ContinuousLinearMap.continuous_det.measurable.comp_aemeasurable ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable f' (μ.restrict s)
  -/
  exact aemeasurable_fderivWithin μ hs hf'
  /-
    🎉 no goals
  -/


theorem aemeasurable_toNNReal_abs_det_fderivWithin (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) :
    AEMeasurable (fun x => |(f' x).det|.toNNReal) (μ.restrict s) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable (fun x => (abs (f' x).det).toNNReal) (μ.restrict s)
  -/
  apply measurable_real_toNNReal.comp_aemeasurable
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable (fun x => abs (f' x).det) (μ.restrict s)
  -/
  refine continuous_abs.measurable.comp_aemeasurable ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable (fun x => (f' x).det) (μ.restrict s)
  -/
  refine ContinuousLinearMap.continuous_det.measurable.comp_aemeasurable ?_
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ AEMeasurable f' (μ.restrict s)
  -/
  exact aemeasurable_fderivWithin μ hs hf'
  /-
    🎉 no goals
  -/


/-- If a function is differentiable and injective on a measurable set,
then the image is measurable. -/
theorem measurable_image_of_fderivWithin (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) : MeasurableSet (f '' s) :=
  haveI : DifferentiableOn ℝ f s := fun x hx => (hf' x hx).differentiableWithinAt
  hs.image_of_continuousOn_injOn (DifferentiableOn.continuousOn this) hf


/-- If a function is differentiable and injective on a measurable set `s`, then its restriction
to `s` is a measurable embedding. -/
theorem measurableEmbedding_of_fderivWithin (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) :
    MeasurableEmbedding (s.restrict f) :=
  haveI : DifferentiableOn ℝ f s := fun x hx => (hf' x hx).differentiableWithinAt
  this.continuousOn.measurableEmbedding hs hf


theorem addHaar_image_le_lintegral_abs_det_fderiv_aux1 (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) {ε : ℝ≥0} (εpos : 0 < ε) :
    μ (f '' s) ≤ (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) + 2 * ε * μ s := by
  /- To bound `μ (f '' s)`, we cover `s` by sets where `f` is well-approximated by linear maps
    `A n` (and where `f'` is almost everywhere close to `A n`), and then use that `f` expands the
    measure of such a set by at most `(A n).det + ε`. -/
  have :
    ∀ A : E →L[ℝ] E,
      ∃ δ : ℝ≥0,
        0 < δ ∧
          (∀ B : E →L[ℝ] E, ‖B - A‖ ≤ δ → |B.det - A.det| ≤ ε) ∧
            ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g A t δ →
              μ (g '' t) ≤ (ENNReal.ofReal |A.det| + ε) * μ t := by
    intro A
    let m : ℝ≥0 := Real.toNNReal |A.det| + ε
    have I : ENNReal.ofReal |A.det| < m := by
      simp only [m, ENNReal.ofReal, lt_add_iff_pos_right, εpos, ENNReal.coe_lt_coe]
    rcases ((addHaar_image_le_mul_of_det_lt μ A I).and self_mem_nhdsWithin).exists with ⟨δ, h, δpos⟩
    obtain ⟨δ', δ'pos, hδ'⟩ : ∃ (δ' : ℝ), 0 < δ' ∧ ∀ B, dist B A < δ' → dist B.det A.det < ↑ε :=
      continuousAt_iff.1 (ContinuousLinearMap.continuous_det (E := E)).continuousAt ε εpos
    let δ'' : ℝ≥0 := ⟨δ' / 2, (half_pos δ'pos).le⟩
    refine ⟨min δ δ'', lt_min δpos (half_pos δ'pos), ?_, ?_⟩
    · intro B hB
      rw [← Real.dist_eq]
      apply (hδ' B _).le
      rw [dist_eq_norm]
      calc
        ‖B - A‖ ≤ (min δ δ'' : ℝ≥0) := hB
        _ ≤ δ'' := by simp only [le_refl, NNReal.coe_min, min_le_iff, or_true]
        _ < δ' := half_lt_self δ'pos
    · intro t g htg
      exact h t g (htg.mono_num (min_le_left _ _))
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ε : NNReal
    εpos : LT.lt 0 ε
    this : ∀ (A : ContinuousLinearMap (RingHom.id Real) E E), Exists fun δ => And  …
    ⊢ LE.le (μ (Set.image f s)) (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict s) …
  -/
  choose δ hδ using this
  obtain ⟨t, A, t_disj, t_meas, t_cover, ht, -⟩ :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] E),
      Pairwise (Disjoint on t) ∧
        (∀ n : ℕ, MeasurableSet (t n)) ∧
          (s ⊆ ⋃ n : ℕ, t n) ∧
            (∀ n : ℕ, ApproximatesLinearOn f (A n) (s ∩ t n) (δ (A n))) ∧
              (s.Nonempty → ∀ n, ∃ y ∈ s, A n = f' y) :=
    exists_partition_approximatesLinearOn_of_hasFDerivWithinAt f s f' hf' δ fun A => (hδ A).1.ne'
  calc
    μ (f '' s) ≤ μ (⋃ n, f '' (s ∩ t n)) := by
      apply measure_mono
      rw [← image_iUnion, ← inter_iUnion]
      exact image_subset f (subset_inter Subset.rfl t_cover)
    _ ≤ ∑' n, μ (f '' (s ∩ t n)) := measure_iUnion_le _
    _ ≤ ∑' n, (ENNReal.ofReal |(A n).det| + ε) * μ (s ∩ t n) := by
      apply ENNReal.tsum_le_tsum fun n => ?_
      apply (hδ (A n)).2.2
      exact ht n
    _ = ∑' n, ∫⁻ _ in s ∩ t n, ENNReal.ofReal |(A n).det| + ε ∂μ := by
      simp only [lintegral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter]
    _ ≤ ∑' n, ∫⁻ x in s ∩ t n, ENNReal.ofReal |(f' x).det| + 2 * ε ∂μ := by
      apply ENNReal.tsum_le_tsum fun n => ?_
      apply lintegral_mono_ae
      filter_upwards [(ht n).norm_fderiv_sub_le μ (hs.inter (t_meas n)) f' fun x hx =>
          (hf' x hx.1).mono inter_subset_left]
      intro x hx
      have I : |(A n).det| ≤ |(f' x).det| + ε :=
        calc
          |(A n).det| = |(f' x).det - ((f' x).det - (A n).det)| := by congr 1; abel
          _ ≤ |(f' x).det| + |(f' x).det - (A n).det| := abs_sub _ _
          _ ≤ |(f' x).det| + ε := add_le_add le_rfl ((hδ (A n)).2.1 _ hx)
      calc
        ENNReal.ofReal |(A n).det| + ε ≤ ENNReal.ofReal (|(f' x).det| + ε) + ε := by gcongr
        _ = ENNReal.ofReal |(f' x).det| + 2 * ε := by
          simp only [ENNReal.ofReal_add, abs_nonneg, two_mul, add_assoc, NNReal.zero_le_coe,
            ENNReal.ofReal_coe_nnreal]
    _ = ∫⁻ x in ⋃ n, s ∩ t n, ENNReal.ofReal |(f' x).det| + 2 * ε ∂μ := by
      have M : ∀ n : ℕ, MeasurableSet (s ∩ t n) := fun n => hs.inter (t_meas n)
      rw [lintegral_iUnion M]
      exact pairwise_disjoint_mono t_disj fun n => inter_subset_right
    _ = ∫⁻ x in s, ENNReal.ofReal |(f' x).det| + 2 * ε ∂μ := by
      rw [← inter_iUnion, inter_eq_self_of_subset_left t_cover]
    _ = (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) + 2 * ε * μ s := by
      simp only [lintegral_add_right' _ aemeasurable_const, setLIntegral_const]


theorem addHaar_image_le_lintegral_abs_det_fderiv_aux2 (hs : MeasurableSet s) (h's : μ s ≠ ∞)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) :
    μ (f '' s) ≤ ∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ := by
  -- We just need to let the error tend to `0` in the previous lemma.
  have :
    Tendsto (fun ε : ℝ≥0 => (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) + 2 * ε * μ s) (𝓝[>] 0)
      (𝓝 ((∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) + 2 * (0 : ℝ≥0) * μ s)) := by
    apply Tendsto.mono_left _ nhdsWithin_le_nhds
    refine tendsto_const_nhds.add ?_
    refine ENNReal.Tendsto.mul_const ?_ (Or.inr h's)
    exact ENNReal.Tendsto.const_mul (ENNReal.tendsto_coe.2 tendsto_id) (Or.inr ENNReal.coe_ne_top)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    this : Filter.Tendsto (fun ε => HAdd.hAdd (MeasureTheory.lintegral (μ.restrict …
    ⊢ LE.le (μ (Set.image f s)) (MeasureTheory.lintegral (μ.restrict s) fun x => E …
  -/
  simp only [add_zero, zero_mul, mul_zero, ENNReal.coe_zero] at this
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    this : Filter.Tendsto (fun ε => HAdd.hAdd (MeasureTheory.lintegral (μ.restrict …
    ⊢ LE.le (μ (Set.image f s)) (MeasureTheory.lintegral (μ.restrict s) fun x => E …
  -/
  apply ge_of_tendsto this
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    this : Filter.Tendsto (fun ε => HAdd.hAdd (MeasureTheory.lintegral (μ.restrict …
    ⊢ Filter.Eventually (fun c => LE.le (μ (Set.image f s)) (HAdd.hAdd (MeasureThe …
  -/
  filter_upwards [self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    this : Filter.Tendsto (fun ε => HAdd.hAdd (MeasureTheory.lintegral (μ.restrict …
    ⊢ ∀ (a : NNReal), Membership.mem (Set.Ioi 0) a → LE.le (μ (Set.image f s)) (HA …
  -/
  intro ε εpos
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    this : Filter.Tendsto (fun ε => HAdd.hAdd (MeasureTheory.lintegral (μ.restrict …
    ε : NNReal
    εpos : Membership.mem (Set.Ioi 0) ε
    ⊢ LE.le (μ (Set.image f s)) (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict s) …
  -/
  rw [mem_Ioi] at εpos
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    this : Filter.Tendsto (fun ε => HAdd.hAdd (MeasureTheory.lintegral (μ.restrict …
    ε : NNReal
    εpos : LT.lt 0 ε
    ⊢ LE.le (μ (Set.image f s)) (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict s) …
  -/
  exact addHaar_image_le_lintegral_abs_det_fderiv_aux1 μ hs hf' εpos
  /-
    🎉 no goals
  -/


theorem addHaar_image_le_lintegral_abs_det_fderiv (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) :
    μ (f '' s) ≤ ∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ := by
  /- We already know the result for finite-measure sets. We cover `s` by finite-measure sets using
    `spanningSets μ`, and apply the previous result to each of these parts. -/
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ LE.le (μ (Set.image f s)) (MeasureTheory.lintegral (μ.restrict s) fun x => E …
  -/
  let u n := disjointed (spanningSets μ) n
  have u_meas : ∀ n, MeasurableSet (u n) := by
    intro n
    apply MeasurableSet.disjointed fun i => ?_
    exact measurableSet_spanningSets μ i
  have A : s = ⋃ n, s ∩ u n := by
    rw [← inter_iUnion, iUnion_disjointed, iUnion_spanningSets, inter_univ]
  calc
    μ (f '' s) ≤ ∑' n, μ (f '' (s ∩ u n)) := by
      conv_lhs => rw [A, image_iUnion]
      exact measure_iUnion_le _
    _ ≤ ∑' n, ∫⁻ x in s ∩ u n, ENNReal.ofReal |(f' x).det| ∂μ := by
      apply ENNReal.tsum_le_tsum fun n => ?_
      apply
        addHaar_image_le_lintegral_abs_det_fderiv_aux2 μ (hs.inter (u_meas n)) _ fun x hx =>
          (hf' x hx.1).mono inter_subset_left
      have : μ (u n) < ∞ :=
        lt_of_le_of_lt (measure_mono (disjointed_subset _ _)) (measure_spanningSets_lt_top μ n)
      exact ne_of_lt (lt_of_le_of_lt (measure_mono inter_subset_right) this)
    _ = ∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ := by
      conv_rhs => rw [A]
      rw [lintegral_iUnion]
      · intro n; exact hs.inter (u_meas n)
      · exact pairwise_disjoint_mono (disjoint_disjointed _) fun n => inter_subset_right


theorem lintegral_abs_det_fderiv_le_addHaar_image_aux1 (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) {ε : ℝ≥0} (εpos : 0 < ε) :
    (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) ≤ μ (f '' s) + 2 * ε * μ s := by
  /- To bound `∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ`, we cover `s` by sets where `f` is
    well-approximated by linear maps `A n` (and where `f'` is almost everywhere close to `A n`),
    and then use that `f` expands the measure of such a set by at least `(A n).det - ε`. -/
  have :
    ∀ A : E →L[ℝ] E,
      ∃ δ : ℝ≥0,
        0 < δ ∧
          (∀ B : E →L[ℝ] E, ‖B - A‖ ≤ δ → |B.det - A.det| ≤ ε) ∧
            ∀ (t : Set E) (g : E → E), ApproximatesLinearOn g A t δ →
              ENNReal.ofReal |A.det| * μ t ≤ μ (g '' t) + ε * μ t := by
    intro A
    obtain ⟨δ', δ'pos, hδ'⟩ : ∃ (δ' : ℝ), 0 < δ' ∧ ∀ B, dist B A < δ' → dist B.det A.det < ↑ε :=
      continuousAt_iff.1 (ContinuousLinearMap.continuous_det (E := E)).continuousAt ε εpos
    let δ'' : ℝ≥0 := ⟨δ' / 2, (half_pos δ'pos).le⟩
    have I'' : ∀ B : E →L[ℝ] E, ‖B - A‖ ≤ ↑δ'' → |B.det - A.det| ≤ ↑ε := by
      intro B hB
      rw [← Real.dist_eq]
      apply (hδ' B _).le
      rw [dist_eq_norm]
      exact hB.trans_lt (half_lt_self δ'pos)
    rcases eq_or_ne A.det 0 with (hA | hA)
    · refine ⟨δ'', half_pos δ'pos, I'', ?_⟩
      simp only [hA, forall_const, zero_mul, ENNReal.ofReal_zero, imp_true_iff,
        zero_le, abs_zero]
    let m : ℝ≥0 := Real.toNNReal |A.det| - ε
    have I : (m : ℝ≥0∞) < ENNReal.ofReal |A.det| := by
      simp only [m, ENNReal.ofReal, ENNReal.coe_sub]
      apply ENNReal.sub_lt_self ENNReal.coe_ne_top
      · simpa only [abs_nonpos_iff, Real.toNNReal_eq_zero, ENNReal.coe_eq_zero, Ne] using hA
      · simp only [εpos.ne', ENNReal.coe_eq_zero, Ne, not_false_iff]
    rcases ((mul_le_addHaar_image_of_lt_det μ A I).and self_mem_nhdsWithin).exists with ⟨δ, h, δpos⟩
    refine ⟨min δ δ'', lt_min δpos (half_pos δ'pos), ?_, ?_⟩
    · intro B hB
      apply I'' _ (hB.trans _)
      simp only [le_refl, NNReal.coe_min, min_le_iff, or_true]
    · intro t g htg
      rcases eq_or_ne (μ t) ∞ with (ht | ht)
      · simp only [ht, εpos.ne', ENNReal.mul_top, ENNReal.coe_eq_zero, le_top, Ne,
          not_false_iff, _root_.add_top]
      have := h t g (htg.mono_num (min_le_left _ _))
      rwa [ENNReal.coe_sub, ENNReal.sub_mul, tsub_le_iff_right] at this
      simp only [ht, imp_true_iff, Ne, not_false_iff]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    ε : NNReal
    εpos : LT.lt 0 ε
    this : ∀ (A : ContinuousLinearMap (RingHom.id Real) E E), Exists fun δ => And  …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ENNReal.ofReal (abs ( …
  -/
  choose δ hδ using this
  obtain ⟨t, A, t_disj, t_meas, t_cover, ht, -⟩ :
    ∃ (t : ℕ → Set E) (A : ℕ → E →L[ℝ] E),
      Pairwise (Disjoint on t) ∧
        (∀ n : ℕ, MeasurableSet (t n)) ∧
          (s ⊆ ⋃ n : ℕ, t n) ∧
            (∀ n : ℕ, ApproximatesLinearOn f (A n) (s ∩ t n) (δ (A n))) ∧
              (s.Nonempty → ∀ n, ∃ y ∈ s, A n = f' y) :=
    exists_partition_approximatesLinearOn_of_hasFDerivWithinAt f s f' hf' δ fun A => (hδ A).1.ne'
  have s_eq : s = ⋃ n, s ∩ t n := by
    rw [← inter_iUnion]
    exact Subset.antisymm (subset_inter Subset.rfl t_cover) inter_subset_left
  calc
    (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) =
        ∑' n, ∫⁻ x in s ∩ t n, ENNReal.ofReal |(f' x).det| ∂μ := by
      conv_lhs => rw [s_eq]
      rw [lintegral_iUnion]
      · exact fun n => hs.inter (t_meas n)
      · exact pairwise_disjoint_mono t_disj fun n => inter_subset_right
    _ ≤ ∑' n, ∫⁻ _ in s ∩ t n, ENNReal.ofReal |(A n).det| + ε ∂μ := by
      apply ENNReal.tsum_le_tsum fun n => ?_
      apply lintegral_mono_ae
      filter_upwards [(ht n).norm_fderiv_sub_le μ (hs.inter (t_meas n)) f' fun x hx =>
          (hf' x hx.1).mono inter_subset_left]
      intro x hx
      have I : |(f' x).det| ≤ |(A n).det| + ε :=
        calc
          |(f' x).det| = |(A n).det + ((f' x).det - (A n).det)| := by congr 1; abel
          _ ≤ |(A n).det| + |(f' x).det - (A n).det| := abs_add _ _
          _ ≤ |(A n).det| + ε := add_le_add le_rfl ((hδ (A n)).2.1 _ hx)
      calc
        ENNReal.ofReal |(f' x).det| ≤ ENNReal.ofReal (|(A n).det| + ε) :=
          ENNReal.ofReal_le_ofReal I
        _ = ENNReal.ofReal |(A n).det| + ε := by
          simp only [ENNReal.ofReal_add, abs_nonneg, NNReal.zero_le_coe, ENNReal.ofReal_coe_nnreal]
    _ = ∑' n, (ENNReal.ofReal |(A n).det| * μ (s ∩ t n) + ε * μ (s ∩ t n)) := by
      simp only [setLIntegral_const, lintegral_add_right _ measurable_const]
    _ ≤ ∑' n, (μ (f '' (s ∩ t n)) + ε * μ (s ∩ t n) + ε * μ (s ∩ t n)) := by
      gcongr
      exact (hδ (A _)).2.2 _ _ (ht _)
    _ = μ (f '' s) + 2 * ε * μ s := by
      conv_rhs => rw [s_eq]
      rw [image_iUnion, measure_iUnion]; rotate_left
      · intro i j hij
        apply Disjoint.image _ hf inter_subset_left inter_subset_left
        exact Disjoint.mono inter_subset_right inter_subset_right (t_disj hij)
      · intro i
        exact
          measurable_image_of_fderivWithin (hs.inter (t_meas i))
            (fun x hx => (hf' x hx.1).mono inter_subset_left)
            (hf.mono inter_subset_left)
      rw [measure_iUnion]; rotate_left
      · exact pairwise_disjoint_mono t_disj fun i => inter_subset_right
      · exact fun i => hs.inter (t_meas i)
      rw [← ENNReal.tsum_mul_left, ← ENNReal.tsum_add]
      congr 1
      ext1 i
      rw [mul_assoc, two_mul, add_assoc]


theorem lintegral_abs_det_fderiv_le_addHaar_image_aux2 (hs : MeasurableSet s) (h's : μ s ≠ ∞)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) :
    (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) ≤ μ (f '' s) := by
  -- We just need to let the error tend to `0` in the previous lemma.
  have :
    Tendsto (fun ε : ℝ≥0 => μ (f '' s) + 2 * ε * μ s) (𝓝[>] 0)
      (𝓝 (μ (f '' s) + 2 * (0 : ℝ≥0) * μ s)) := by
    apply Tendsto.mono_left _ nhdsWithin_le_nhds
    refine tendsto_const_nhds.add ?_
    refine ENNReal.Tendsto.mul_const ?_ (Or.inr h's)
    exact ENNReal.Tendsto.const_mul (ENNReal.tendsto_coe.2 tendsto_id) (Or.inr ENNReal.coe_ne_top)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    this : Filter.Tendsto (fun ε => HAdd.hAdd (μ (Set.image f s)) (HMul.hMul (HMul …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ENNReal.ofReal (abs ( …
  -/
  simp only [add_zero, zero_mul, mul_zero, ENNReal.coe_zero] at this
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    this : Filter.Tendsto (fun ε => HAdd.hAdd (μ (Set.image f s)) (HMul.hMul (HMul …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ENNReal.ofReal (abs ( …
  -/
  apply ge_of_tendsto this
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    this : Filter.Tendsto (fun ε => HAdd.hAdd (μ (Set.image f s)) (HMul.hMul (HMul …
    ⊢ Filter.Eventually (fun c => LE.le (MeasureTheory.lintegral (μ.restrict s) fu …
  -/
  filter_upwards [self_mem_nhdsWithin]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    this : Filter.Tendsto (fun ε => HAdd.hAdd (μ (Set.image f s)) (HMul.hMul (HMul …
    ⊢ ∀ (a : NNReal), Membership.mem (Set.Ioi 0) a → LE.le (MeasureTheory.lintegra …
  -/
  intro ε εpos
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    this : Filter.Tendsto (fun ε => HAdd.hAdd (μ (Set.image f s)) (HMul.hMul (HMul …
    ε : NNReal
    εpos : Membership.mem (Set.Ioi 0) ε
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ENNReal.ofReal (abs ( …
  -/
  rw [mem_Ioi] at εpos
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    h's : Ne (μ s) Top.top
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    this : Filter.Tendsto (fun ε => HAdd.hAdd (μ (Set.image f s)) (HMul.hMul (HMul …
    ε : NNReal
    εpos : LT.lt 0 ε
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ENNReal.ofReal (abs ( …
  -/
  exact lintegral_abs_det_fderiv_le_addHaar_image_aux1 μ hs hf' hf εpos
  /-
    🎉 no goals
  -/


theorem lintegral_abs_det_fderiv_le_addHaar_image (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) :
    (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) ≤ μ (f '' s) := by
  /- We already know the result for finite-measure sets. We cover `s` by finite-measure sets using
    `spanningSets μ`, and apply the previous result to each of these parts. -/
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ENNReal.ofReal (abs ( …
  -/
  let u n := disjointed (spanningSets μ) n
  have u_meas : ∀ n, MeasurableSet (u n) := by
    intro n
    apply MeasurableSet.disjointed fun i => ?_
    exact measurableSet_spanningSets μ i
  have A : s = ⋃ n, s ∩ u n := by
    rw [← inter_iUnion, iUnion_disjointed, iUnion_spanningSets, inter_univ]
  calc
    (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) =
        ∑' n, ∫⁻ x in s ∩ u n, ENNReal.ofReal |(f' x).det| ∂μ := by
      conv_lhs => rw [A]
      rw [lintegral_iUnion]
      · intro n; exact hs.inter (u_meas n)
      · exact pairwise_disjoint_mono (disjoint_disjointed _) fun n => inter_subset_right
    _ ≤ ∑' n, μ (f '' (s ∩ u n)) := by
      apply ENNReal.tsum_le_tsum fun n => ?_
      apply
        lintegral_abs_det_fderiv_le_addHaar_image_aux2 μ (hs.inter (u_meas n)) _
          (fun x hx => (hf' x hx.1).mono inter_subset_left) (hf.mono inter_subset_left)
      have : μ (u n) < ∞ :=
        lt_of_le_of_lt (measure_mono (disjointed_subset _ _)) (measure_spanningSets_lt_top μ n)
      exact ne_of_lt (lt_of_le_of_lt (measure_mono inter_subset_right) this)
    _ = μ (f '' s) := by
      conv_rhs => rw [A, image_iUnion]
      rw [measure_iUnion]
      · intro i j hij
        apply Disjoint.image _ hf inter_subset_left inter_subset_left
        exact
          Disjoint.mono inter_subset_right inter_subset_right
            (disjoint_disjointed _ hij)
      · intro i
        exact
          measurable_image_of_fderivWithin (hs.inter (u_meas i))
            (fun x hx => (hf' x hx.1).mono inter_subset_left)
            (hf.mono inter_subset_left)


/-- Change of variable formula for differentiable functions, set version: if a function `f` is
injective and differentiable on a measurable set `s`, then the measure of `f '' s` is given by the
integral of `|(f' x).det|` on `s`.
Note that the measurability of `f '' s` is given by `measurable_image_of_fderivWithin`. -/
theorem lintegral_abs_det_fderiv_eq_addHaar_image (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) :
    (∫⁻ x in s, ENNReal.ofReal |(f' x).det| ∂μ) = μ (f '' s) :=
  le_antisymm (lintegral_abs_det_fderiv_le_addHaar_image μ hs hf' hf)
    (addHaar_image_le_lintegral_abs_det_fderiv μ hs hf')


/-- Change of variable formula for differentiable functions, set version: if a function `f` is
injective and differentiable on a measurable set `s`, then the pushforward of the measure with
density `|(f' x).det|` on `s` is the Lebesgue measure on the image set. This version requires
that `f` is measurable, as otherwise `Measure.map f` is zero per our definitions.
For a version without measurability assumption but dealing with the restricted
function `s.restrict f`, see `restrict_map_withDensity_abs_det_fderiv_eq_addHaar`.
-/
theorem map_withDensity_abs_det_fderiv_eq_addHaar (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) (h'f : Measurable f) :
    Measure.map f ((μ.restrict s).withDensity fun x => ENNReal.ofReal |(f' x).det|) =
      μ.restrict (f '' s) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    h'f : Measurable f
    ⊢ Eq (MeasureTheory.Measure.map f ((μ.restrict s).withDensity fun x => ENNReal …
  -/
  apply Measure.ext fun t ht => ?_
  rw [map_apply h'f ht, withDensity_apply _ (h'f ht), Measure.restrict_apply ht,
    restrict_restrict (h'f ht),
    lintegral_abs_det_fderiv_eq_addHaar_image μ ((h'f ht).inter hs)
      (fun x hx => (hf' x hx.2).mono inter_subset_right) (hf.mono inter_subset_right),
    image_preimage_inter]


/-- Change of variable formula for differentiable functions, set version: if a function `f` is
injective and differentiable on a measurable set `s`, then the pushforward of the measure with
density `|(f' x).det|` on `s` is the Lebesgue measure on the image set. This version is expressed
in terms of the restricted function `s.restrict f`.
For a version for the original function, but with a measurability assumption,
see `map_withDensity_abs_det_fderiv_eq_addHaar`.
-/
theorem restrict_map_withDensity_abs_det_fderiv_eq_addHaar (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) :
    Measure.map (s.restrict f) (comap (↑) (μ.withDensity fun x => ENNReal.ofReal |(f' x).det|)) =
      μ.restrict (f '' s) := by
  obtain ⟨u, u_meas, uf⟩ : ∃ u, Measurable u ∧ EqOn u f s := by
    classical
    refine ⟨piecewise s f 0, ?_, piecewise_eqOn _ _ _⟩
    refine ContinuousOn.measurable_piecewise ?_ continuous_zero.continuousOn hs
    have : DifferentiableOn ℝ f s := fun x hx => (hf' x hx).differentiableWithinAt
    exact this.continuousOn
  have u' : ∀ x ∈ s, HasFDerivWithinAt u (f' x) s x := fun x hx =>
    (hf' x hx).congr (fun y hy => uf hy) (uf hx)
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    u : E → E
    u_meas : Measurable u
    uf : Set.EqOn u f s
    u' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt u (f' x) s x
    ⊢ Eq (MeasureTheory.Measure.map (s.restrict f) (MeasureTheory.Measure.comap Su …
  -/
  set F : s → E := u ∘ (↑) with hF
  have A :
    Measure.map F (comap (↑) (μ.withDensity fun x => ENNReal.ofReal |(f' x).det|)) =
      μ.restrict (u '' s) := by
    rw [hF, ← Measure.map_map u_meas measurable_subtype_coe, map_comap_subtype_coe hs,
      restrict_withDensity hs]
    exact map_withDensity_abs_det_fderiv_eq_addHaar μ hs u' (hf.congr uf.symm) u_meas
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    u : E → E
    u_meas : Measurable u
    uf : Set.EqOn u f s
    u' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt u (f' x) s x
    F : ↑s → E := Function.comp u Subtype.val
    hF : Eq F (Function.comp u Subtype.val)
    A : Eq (MeasureTheory.Measure.map F (MeasureTheory.Measure.comap Subtype.val ( …
    ⊢ Eq (MeasureTheory.Measure.map (s.restrict f) (MeasureTheory.Measure.comap Su …
  -/
  rw [uf.image_eq] at A
  have : F = s.restrict f := by
    ext x
    exact uf x.2
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    u : E → E
    u_meas : Measurable u
    uf : Set.EqOn u f s
    u' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt u (f' x) s x
    F : ↑s → E := Function.comp u Subtype.val
    hF : Eq F (Function.comp u Subtype.val)
    A : Eq (MeasureTheory.Measure.map F (MeasureTheory.Measure.comap Subtype.val ( …
    this : Eq F (s.restrict f)
    ⊢ Eq (MeasureTheory.Measure.map (s.restrict f) (MeasureTheory.Measure.comap Su …
  -/
  rwa [this] at A
  /-
    🎉 no goals
  -/


theorem lintegral_image_eq_lintegral_abs_det_fderiv_mul (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) (g : E → ℝ≥0∞) :
    ∫⁻ x in f '' s, g x ∂μ = ∫⁻ x in s, ENNReal.ofReal |(f' x).det| * g (f x) ∂μ := by
  rw [← restrict_map_withDensity_abs_det_fderiv_eq_addHaar μ hs hf' hf,
    (measurableEmbedding_of_fderivWithin hs hf' hf).lintegral_map]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    g : E → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.comap Subtype.val (μ.with …
  -/
  simp only [Set.restrict_apply, ← Function.comp_apply (f := g)]
  rw [← (MeasurableEmbedding.subtype_coe hs).lintegral_map, map_comap_subtype_coe hs,
    setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀ _ _ _ hs]
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      s : Set E
      f : E → E
      f' : E → ContinuousLinearMap (RingHom.id Real) E E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hs : MeasurableSet s
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      hf : Set.InjOn f s
      g : E → ENNReal
      ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HMul.hMul (fun x => ENNR …
    -/
  · simp only [Pi.mul_apply]
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      s : Set E
      f : E → E
      f' : E → ContinuousLinearMap (RingHom.id Real) E E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hs : MeasurableSet s
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      hf : Set.InjOn f s
      g : E → ENNReal
      ⊢ Filter.Eventually (fun x => LT.lt (ENNReal.ofReal (abs (f' x).det)) Top.top) …
    -/
  · simp only [eventually_true, ENNReal.ofReal_lt_top]
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      s : Set E
      f : E → E
      f' : E → ContinuousLinearMap (RingHom.id Real) E E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hs : MeasurableSet s
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      hf : Set.InjOn f s
      g : E → ENNReal
      ⊢ AEMeasurable (fun x => ENNReal.ofReal (abs (f' x).det)) (μ.restrict s)
    -/
  · exact aemeasurable_ofReal_abs_det_fderivWithin μ hs hf'
    /-
      🎉 no goals
    -/


/-- Integrability in the change of variable formula for differentiable functions: if a
function `f` is injective and differentiable on a measurable set `s`, then a function
`g : E → F` is integrable on `f '' s` if and only if `|(f' x).det| • g ∘ f` is
integrable on `s`. -/
theorem integrableOn_image_iff_integrableOn_abs_det_fderiv_smul (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) (g : E → F) :
    IntegrableOn g (f '' s) μ ↔ IntegrableOn (fun x => |(f' x).det| • g (f x)) s μ := by
  rw [IntegrableOn, ← restrict_map_withDensity_abs_det_fderiv_eq_addHaar μ hs hf' hf,
    (measurableEmbedding_of_fderivWithin hs hf' hf).integrable_map_iff]
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    g : E → F
    ⊢ Iff (MeasureTheory.Integrable (Function.comp g (s.restrict f)) (MeasureTheor …
  -/
  simp only [Set.restrict_eq, ← Function.comp_assoc, ENNReal.ofReal]
  rw [← (MeasurableEmbedding.subtype_coe hs).integrable_map_iff, map_comap_subtype_coe hs,
    restrict_withDensity hs, integrable_withDensity_iff_integrable_coe_smul₀]
    /-
      E : Type u_1
      F : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      s : Set E
      f : E → E
      f' : E → ContinuousLinearMap (RingHom.id Real) E E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hs : MeasurableSet s
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      hf : Set.InjOn f s
      g : E → F
      ⊢ Iff (MeasureTheory.Integrable (fun x => HSMul.hSMul (↑(abs (f' x).det).toNNR …
    -/
  · simp_rw [IntegrableOn, Real.coe_toNNReal _ (abs_nonneg _), Function.comp_apply]
    /-
      🎉 no goals
    -/
    /-
      case hf
      E : Type u_1
      F : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      s : Set E
      f : E → E
      f' : E → ContinuousLinearMap (RingHom.id Real) E E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      hs : MeasurableSet s
      hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
      hf : Set.InjOn f s
      g : E → F
      ⊢ AEMeasurable (fun x => (abs (f' x).det).toNNReal) (μ.restrict s)
    -/
  · exact aemeasurable_toNNReal_abs_det_fderivWithin μ hs hf'
    /-
      🎉 no goals
    -/


/-- Change of variable formula for differentiable functions: if a function `f` is
injective and differentiable on a measurable set `s`, then the Bochner integral of a function
`g : E → F` on `f '' s` coincides with the integral of `|(f' x).det| • g ∘ f` on `s`. -/
theorem integral_image_eq_integral_abs_det_fderiv_smul (hs : MeasurableSet s)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hf : InjOn f s) (g : E → F) :
    ∫ x in f '' s, g x ∂μ = ∫ x in s, |(f' x).det| • g (f x) ∂μ := by
  rw [← restrict_map_withDensity_abs_det_fderiv_eq_addHaar μ hs hf' hf,
    (measurableEmbedding_of_fderivWithin hs hf' hf).integral_map]
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    g : E → F
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val (μ.withD …
  -/
  simp only [Set.restrict_apply, ← Function.comp_apply (f := g), ENNReal.ofReal]
  rw [← (MeasurableEmbedding.subtype_coe hs).integral_map, map_comap_subtype_coe hs,
    setIntegral_withDensity_eq_setIntegral_smul₀
      (aemeasurable_toNNReal_abs_det_fderivWithin μ hs hf') _ hs]
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    g : E → F
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (abs (f' x).d …
  -/
  congr with x
  /-
    case e_f.h
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hf : Set.InjOn f s
    g : E → F
    x : E
    ⊢ Eq (HSMul.hSMul (abs (f' x).det).toNNReal (Function.comp g f x)) (HSMul.hSMu …
  -/
  rw [NNReal.smul_def, Real.coe_toNNReal _ (abs_nonneg (f' x).det)]
  /-
    🎉 no goals
  -/

-- Porting note: move this to `Topology.Algebra.Module.Basic` when port is over

theorem det_one_smulRight {𝕜 : Type*} [CommRing 𝕜] [TopologicalSpace 𝕜] [ContinuousMul 𝕜] (v : 𝕜) :
    ((1 : 𝕜 →L[𝕜] 𝕜).smulRight v).det = v := by
  /-
    𝕜 : Type u_3
    inst✝² : CommRing 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : ContinuousMul 𝕜
    v : 𝕜
    ⊢ Eq (ContinuousLinearMap.smulRight 1 v).det v
  -/
  nontriviality 𝕜
  have : (1 : 𝕜 →L[𝕜] 𝕜).smulRight v = v • (1 : 𝕜 →L[𝕜] 𝕜) := by
    ext1
    simp only [ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply,
      Algebra.id.smul_eq_mul, one_mul, ContinuousLinearMap.coe_smul', Pi.smul_apply, mul_one]
  rw [this, ContinuousLinearMap.det, ContinuousLinearMap.coe_smul,
    ContinuousLinearMap.one_def, ContinuousLinearMap.coe_id, LinearMap.det_smul,
    Module.finrank_self, LinearMap.det_id, pow_one, mul_one]


/-- Integrability in the change of variable formula for differentiable functions (one-variable
version): if a function `f` is injective and differentiable on a measurable set `s ⊆ ℝ`, then a
function `g : ℝ → F` is integrable on `f '' s` if and only if `|(f' x)| • g ∘ f` is integrable on
`s`. -/
theorem integrableOn_image_iff_integrableOn_abs_deriv_smul {s : Set ℝ} {f : ℝ → ℝ} {f' : ℝ → ℝ}
    (hs : MeasurableSet s) (hf' : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) (hf : InjOn f s)
                  /-
                    E : Type u_1
                    F : Type u_2
                    inst✝⁷ : NormedAddCommGroup E
                    inst✝⁶ : NormedSpace Real E
                    inst✝⁵ : FiniteDimensional Real E
                    inst✝⁴ : NormedAddCommGroup F
                    inst✝³ : NormedSpace Real F
                    s✝ : Set E
                    f✝ : E → E
                    f'✝ : E → ContinuousLinearMap (RingHom.id Real) E E
                    inst✝² : MeasurableSpace E
                    inst✝¹ : BorelSpace E
                    μ : MeasureTheory.Measure E
                    inst✝ : μ.IsAddHaarMeasure
                    s : Set Real
                    f f' : Real → Real
                    hs : MeasurableSet s
                    hf' : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
                    hf : Set.InjOn f s
                    g : Real → F
                    ⊢ MeasureTheory.Measure Real
                  -/
                  /-
                    🎉 no goals
                  -/
    (g : ℝ → F) : IntegrableOn g (f '' s) ↔ IntegrableOn (fun x => |f' x| • g (f x)) s := by
                                            /-
                                              🎉 no goals
                                            -/
  simpa only [det_one_smulRight] using
    integrableOn_image_iff_integrableOn_abs_det_fderiv_smul volume hs
      (fun x hx => (hf' x hx).hasFDerivWithinAt) hf g


/-- Change of variable formula for differentiable functions (one-variable version): if a function
`f` is injective and differentiable on a measurable set `s ⊆ ℝ`, then the Bochner integral of a
function `g : ℝ → F` on `f '' s` coincides with the integral of `|(f' x)| • g ∘ f` on `s`. -/
theorem integral_image_eq_integral_abs_deriv_smul {s : Set ℝ} {f : ℝ → ℝ} {f' : ℝ → ℝ}
    (hs : MeasurableSet s) (hf' : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x)
    (hf : InjOn f s) (g : ℝ → F) : ∫ x in f '' s, g x = ∫ x in s, |f' x| • g (f x) := by
  simpa only [det_one_smulRight] using
    integral_image_eq_integral_abs_det_fderiv_smul volume hs
      (fun x hx => (hf' x hx).hasFDerivWithinAt) hf g


theorem integral_target_eq_integral_abs_det_fderiv_smul {f : PartialHomeomorph E E}
    (hf' : ∀ x ∈ f.source, HasFDerivAt f (f' x) x) (g : E → F) :
    ∫ x in f.target, g x ∂μ = ∫ x in f.source, |(f' x).det| • g (f x) ∂μ := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : PartialHomeomorph E E
    hf' : ∀ (x : E), Membership.mem f.source x → HasFDerivAt (↑f) (f' x) x
    g : E → F
    ⊢ Eq (MeasureTheory.integral (μ.restrict f.target) fun x => g x) (MeasureTheor …
  -/
  have : f '' f.source = f.target := PartialEquiv.image_source_eq_target f.toPartialEquiv
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : PartialHomeomorph E E
    hf' : ∀ (x : E), Membership.mem f.source x → HasFDerivAt (↑f) (f' x) x
    g : E → F
    this : Eq (Set.image (↑f) f.source) f.target
    ⊢ Eq (MeasureTheory.integral (μ.restrict f.target) fun x => g x) (MeasureTheor …
  -/
  rw [← this]
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : PartialHomeomorph E E
    hf' : ∀ (x : E), Membership.mem f.source x → HasFDerivAt (↑f) (f' x) x
    g : E → F
    this : Eq (Set.image (↑f) f.source) f.target
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.image (↑f) f.source)) fun x => g …
  -/
  apply integral_image_eq_integral_abs_det_fderiv_smul μ f.open_source.measurableSet _ f.injOn
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : PartialHomeomorph E E
    hf' : ∀ (x : E), Membership.mem f.source x → HasFDerivAt (↑f) (f' x) x
    g : E → F
    this : Eq (Set.image (↑f) f.source) f.target
    ⊢ ∀ (x : E), Membership.mem f.source x → HasFDerivWithinAt (↑f) (f' x) f.sourc …
  -/
  intro x hx
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : PartialHomeomorph E E
    hf' : ∀ (x : E), Membership.mem f.source x → HasFDerivAt (↑f) (f' x) x
    g : E → F
    this : Eq (Set.image (↑f) f.source) f.target
    x : E
    hx : Membership.mem f.source x
    ⊢ HasFDerivWithinAt (↑f) (f' x) f.source x
  -/
  exact (hf' x hx).hasFDerivWithinAt
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEmbedding.withDensity_ofReal_comap_apply_eq_integral_abs_det_fderiv_mul
    (hs : MeasurableSet s) (hf : MeasurableEmbedding f)
    {g : E → ℝ} (hg : ∀ᵐ x ∂μ, x ∈ f '' s → 0 ≤ g x) (hg_int : IntegrableOn g (f '' s) μ)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) :
    (μ.withDensity (fun x ↦ ENNReal.ofReal (g x))).comap f s
      = ENNReal.ofReal (∫ x in s, |(f' x).det| * g (f x) ∂μ) := by
  rw [Measure.comap_apply f hf.injective (fun t ht ↦ hf.measurableSet_image' ht) _ hs,
    withDensity_apply _ (hf.measurableSet_image' hs),
    ← ofReal_integral_eq_lintegral_ofReal hg_int
      ((ae_restrict_iff' (hf.measurableSet_image' hs)).mpr hg),
    integral_image_eq_integral_abs_det_fderiv_smul μ hs hf' hf.injective.injOn]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    s : Set E
    f : E → E
    f' : E → ContinuousLinearMap (RingHom.id Real) E E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    hs : MeasurableSet s
    hf : MeasurableEmbedding f
    g : E → Real
    hg : Filter.Eventually (fun x => Membership.mem (Set.image f s) x → LE.le 0 (g …
    hg_int : MeasureTheory.IntegrableOn g (Set.image f s) μ
    hf' : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSM …
  -/
  simp_rw [smul_eq_mul]
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEquiv.withDensity_ofReal_map_symm_apply_eq_integral_abs_det_fderiv_mul
    (hs : MeasurableSet s) (f : E ≃ᵐ E)
    {g : E → ℝ} (hg : ∀ᵐ x ∂μ, x ∈ f '' s → 0 ≤ g x) (hg_int : IntegrableOn g (f '' s) μ)
    (hf' : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) :
    (μ.withDensity (fun x ↦ ENNReal.ofReal (g x))).map f.symm s
      = ENNReal.ofReal (∫ x in s, |(f' x).det| * g (f x) ∂μ) := by
  rw [MeasurableEquiv.map_symm,
    MeasurableEmbedding.withDensity_ofReal_comap_apply_eq_integral_abs_det_fderiv_mul μ hs
      f.measurableEmbedding hg hg_int hf']


lemma _root_.MeasurableEmbedding.withDensity_ofReal_comap_apply_eq_integral_abs_deriv_mul
    {f : ℝ → ℝ} (hf : MeasurableEmbedding f) {s : Set ℝ} (hs : MeasurableSet s)
                                                            /-
                                                              E : Type u_1
                                                              F : Type u_2
                                                              inst✝⁷ : NormedAddCommGroup E
                                                              inst✝⁶ : NormedSpace Real E
                                                              inst✝⁵ : FiniteDimensional Real E
                                                              inst✝⁴ : NormedAddCommGroup F
                                                              inst✝³ : NormedSpace Real F
                                                              s✝ : Set E
                                                              f✝ : E → E
                                                              f' : E → ContinuousLinearMap (RingHom.id Real) E E
                                                              inst✝² : MeasurableSpace E
                                                              inst✝¹ : BorelSpace E
                                                              μ : MeasureTheory.Measure E
                                                              inst✝ : μ.IsAddHaarMeasure
                                                              f : Real → Real
                                                              hf : MeasurableEmbedding f
                                                              s : Set Real
                                                              hs : MeasurableSet s
                                                              g : Real → Real
                                                              hg : Filter.Eventually (fun x => Membership.mem (Set.image f s) x → LE.le 0 (g …
                                                              ⊢ MeasureTheory.Measure Real
                                                            -/
    {g : ℝ → ℝ} (hg : ∀ᵐ x, x ∈ f '' s → 0 ≤ g x) (hg_int : IntegrableOn g (f '' s))
                                                            /-
                                                              🎉 no goals
                                                            -/
    {f' : ℝ → ℝ} (hf' : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) :
    (volume.withDensity (fun x ↦ ENNReal.ofReal (g x))).comap f s
      = ENNReal.ofReal (∫ x in s, |f' x| * g (f x)) := by
  rw [hf.withDensity_ofReal_comap_apply_eq_integral_abs_det_fderiv_mul volume hs
    hg hg_int hf']
  /-
    f : Real → Real
    hf : MeasurableEmbedding f
    s : Set Real
    hs : MeasurableSet s
    g : Real → Real
    hg : Filter.Eventually (fun x => Membership.mem (Set.image f s) x → LE.le 0 (g …
    hg_int : MeasureTheory.IntegrableOn g (Set.image f s) MeasureTheory.MeasureSpa …
    f' : Real → Real
    hf' : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
  -/
  simp only [det_one_smulRight]
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEquiv.withDensity_ofReal_map_symm_apply_eq_integral_abs_deriv_mul
    (f : ℝ ≃ᵐ ℝ) {s : Set ℝ} (hs : MeasurableSet s)
                                                            /-
                                                              E : Type u_1
                                                              F : Type u_2
                                                              inst✝⁷ : NormedAddCommGroup E
                                                              inst✝⁶ : NormedSpace Real E
                                                              inst✝⁵ : FiniteDimensional Real E
                                                              inst✝⁴ : NormedAddCommGroup F
                                                              inst✝³ : NormedSpace Real F
                                                              s✝ : Set E
                                                              f✝ : E → E
                                                              f' : E → ContinuousLinearMap (RingHom.id Real) E E
                                                              inst✝² : MeasurableSpace E
                                                              inst✝¹ : BorelSpace E
                                                              μ : MeasureTheory.Measure E
                                                              inst✝ : μ.IsAddHaarMeasure
                                                              f : MeasurableEquiv Real Real
                                                              s : Set Real
                                                              hs : MeasurableSet s
                                                              g : Real → Real
                                                              hg : Filter.Eventually (fun x => Membership.mem (Set.image (⇑f) s) x → LE.le 0 …
                                                              ⊢ MeasureTheory.Measure Real
                                                            -/
    {g : ℝ → ℝ} (hg : ∀ᵐ x, x ∈ f '' s → 0 ≤ g x) (hg_int : IntegrableOn g (f '' s))
                                                            /-
                                                              🎉 no goals
                                                            -/
    {f' : ℝ → ℝ} (hf' : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) :
    (volume.withDensity (fun x ↦ ENNReal.ofReal (g x))).map f.symm s
      = ENNReal.ofReal (∫ x in s, |f' x| * g (f x)) := by
  rw [MeasurableEquiv.withDensity_ofReal_map_symm_apply_eq_integral_abs_det_fderiv_mul volume hs
      f hg hg_int hf']
  /-
    f : MeasurableEquiv Real Real
    s : Set Real
    hs : MeasurableSet s
    g : Real → Real
    hg : Filter.Eventually (fun x => Membership.mem (Set.image (⇑f) s) x → LE.le 0 …
    hg_int : MeasureTheory.IntegrableOn g (Set.image (⇑f) s) MeasureTheory.Measure …
    f' : Real → Real
    hf' : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt (⇑f) (f' x) s x
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
  -/
  simp only [det_one_smulRight]
  /-
    🎉 no goals
  -/


lemma _root_.MeasurableEmbedding.withDensity_ofReal_comap_apply_eq_integral_abs_deriv_mul'
    {f : ℝ → ℝ} (hf : MeasurableEmbedding f) {s : Set ℝ} (hs : MeasurableSet s)
    {f' : ℝ → ℝ} (hf' : ∀ x, HasDerivAt f (f' x) x)
                                                /-
                                                  E : Type u_1
                                                  F : Type u_2
                                                  inst✝⁷ : NormedAddCommGroup E
                                                  inst✝⁶ : NormedSpace Real E
                                                  inst✝⁵ : FiniteDimensional Real E
                                                  inst✝⁴ : NormedAddCommGroup F
                                                  inst✝³ : NormedSpace Real F
                                                  s✝ : Set E
                                                  f✝ : E → E
                                                  f'✝ : E → ContinuousLinearMap (RingHom.id Real) E E
                                                  inst✝² : MeasurableSpace E
                                                  inst✝¹ : BorelSpace E
                                                  μ : MeasureTheory.Measure E
                                                  inst✝ : μ.IsAddHaarMeasure
                                                  f : Real → Real
                                                  hf : MeasurableEmbedding f
                                                  s : Set Real
                                                  hs : MeasurableSet s
                                                  f' : Real → Real
                                                  hf' : ∀ (x : Real), HasDerivAt f (f' x) x
                                                  g : Real → Real
                                                  hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyLE 0 g
                                                  ⊢ MeasureTheory.Measure Real
                                                -/
    {g : ℝ → ℝ} (hg : 0 ≤ᵐ[volume] g) (hg_int : Integrable g) :
                                                /-
                                                  🎉 no goals
                                                -/
    (volume.withDensity (fun x ↦ ENNReal.ofReal (g x))).comap f s
      = ENNReal.ofReal (∫ x in s, |f' x| * g (f x)) :=
  hf.withDensity_ofReal_comap_apply_eq_integral_abs_deriv_mul hs
        /-
          f : Real → Real
          hf : MeasurableEmbedding f
          s : Set Real
          hs : MeasurableSet s
          f' : Real → Real
          hf' : ∀ (x : Real), HasDerivAt f (f' x) x
          g : Real → Real
          hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyLE 0 g
          hg_int : MeasureTheory.Integrable g MeasureTheory.MeasureSpace.volume
          ⊢ Filter.Eventually (fun x => Membership.mem (Set.image f s) x → LE.le 0 (g x) …
        -/
    (by filter_upwards [hg] with x hx using fun _ ↦ hx) hg_int.integrableOn
        /-
          🎉 no goals
        -/
    (fun x _ => (hf' x).hasDerivWithinAt)


lemma _root_.MeasurableEquiv.withDensity_ofReal_map_symm_apply_eq_integral_abs_deriv_mul'
    (f : ℝ ≃ᵐ ℝ) {s : Set ℝ} (hs : MeasurableSet s)
    {f' : ℝ → ℝ} (hf' : ∀ x, HasDerivAt f (f' x) x)
                                                /-
                                                  E : Type u_1
                                                  F : Type u_2
                                                  inst✝⁷ : NormedAddCommGroup E
                                                  inst✝⁶ : NormedSpace Real E
                                                  inst✝⁵ : FiniteDimensional Real E
                                                  inst✝⁴ : NormedAddCommGroup F
                                                  inst✝³ : NormedSpace Real F
                                                  s✝ : Set E
                                                  f✝ : E → E
                                                  f'✝ : E → ContinuousLinearMap (RingHom.id Real) E E
                                                  inst✝² : MeasurableSpace E
                                                  inst✝¹ : BorelSpace E
                                                  μ : MeasureTheory.Measure E
                                                  inst✝ : μ.IsAddHaarMeasure
                                                  f : MeasurableEquiv Real Real
                                                  s : Set Real
                                                  hs : MeasurableSet s
                                                  f' : Real → Real
                                                  hf' : ∀ (x : Real), HasDerivAt (⇑f) (f' x) x
                                                  g : Real → Real
                                                  hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyLE 0 g
                                                  ⊢ MeasureTheory.Measure Real
                                                -/
    {g : ℝ → ℝ} (hg : 0 ≤ᵐ[volume] g) (hg_int : Integrable g) :
                                                /-
                                                  🎉 no goals
                                                -/
    (volume.withDensity (fun x ↦ ENNReal.ofReal (g x))).map f.symm s
      = ENNReal.ofReal (∫ x in s, |f' x| * g (f x)) := by
  rw [MeasurableEquiv.withDensity_ofReal_map_symm_apply_eq_integral_abs_det_fderiv_mul volume hs
      f (by filter_upwards [hg] with x hx using fun _ ↦ hx) hg_int.integrableOn
      (fun x _ => (hf' x).hasDerivWithinAt)]
  /-
    f : MeasurableEquiv Real Real
    s : Set Real
    hs : MeasurableSet s
    f' : Real → Real
    hf' : ∀ (x : Real), HasDerivAt (⇑f) (f' x) x
    g : Real → Real
    hg : (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyLE 0 g
    hg_int : MeasureTheory.Integrable g MeasureTheory.MeasureSpace.volume
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral (MeasureTheory.MeasureSpace.volum …
  -/
  simp only [det_one_smulRight]
  /-
    🎉 no goals
  -/


