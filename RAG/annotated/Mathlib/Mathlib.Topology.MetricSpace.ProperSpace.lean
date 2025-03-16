/-- A pseudometric space is proper if all closed balls are compact. -/
class ProperSpace (α : Type u) [PseudoMetricSpace α] : Prop where
  isCompact_closedBall : ∀ x : α, ∀ r, IsCompact (closedBall x r)


/-- In a proper pseudometric space, all spheres are compact. -/
theorem isCompact_sphere {α : Type*} [PseudoMetricSpace α] [ProperSpace α] (x : α) (r : ℝ) :
    IsCompact (sphere x r) :=
  (isCompact_closedBall x r).of_isClosed_subset isClosed_sphere sphere_subset_closedBall


/-- In a proper pseudometric space, any sphere is a `CompactSpace` when considered as a subtype. -/
instance Metric.sphere.compactSpace {α : Type*} [PseudoMetricSpace α] [ProperSpace α]
    (x : α) (r : ℝ) : CompactSpace (sphere x r) :=
  isCompact_iff_compactSpace.mp (isCompact_sphere _ _)


/-- A proper pseudo metric space is sigma compact, and therefore second countable. -/
instance (priority := 100) secondCountable_of_proper [ProperSpace α] :
    SecondCountableTopology α := by
  -- We already have `sigmaCompactSpace_of_locallyCompact_secondCountable`, so we don't
  -- add an instance for `SigmaCompactSpace`.
  /-
    α : Type u
    β : Type v
    X : Type u_1
    ι : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    ⊢ SecondCountableTopology α
  -/
  suffices SigmaCompactSpace α from EMetric.secondCountable_of_sigmaCompact α
  /-
    α : Type u
    β : Type v
    X : Type u_1
    ι : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : ProperSpace α
    ⊢ SigmaCompactSpace α
  -/
  rcases em (Nonempty α) with (⟨⟨x⟩⟩ | hn)
    /-
      case inl.intro
      α : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      x : α
      ⊢ SigmaCompactSpace α
    -/
  · exact ⟨⟨fun n => closedBall x n, fun n => isCompact_closedBall _ _, iUnion_closedBall_nat _⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      hn : Not (Nonempty α)
      ⊢ SigmaCompactSpace α
    -/
  · exact ⟨⟨fun _ => ∅, fun _ => isCompact_empty, iUnion_eq_univ_iff.2 fun x => (hn ⟨x⟩).elim⟩⟩
    /-
      🎉 no goals
    -/


/-- If all closed balls of large enough radius are compact, then the space is proper. Especially
useful when the lower bound for the radius is 0. -/
theorem ProperSpace.of_isCompact_closedBall_of_le (R : ℝ)
    (h : ∀ x : α, ∀ r, R ≤ r → IsCompact (closedBall x r)) : ProperSpace α :=
  ⟨fun x r => IsCompact.of_isClosed_subset (h x (max r R) (le_max_right _ _)) isClosed_ball
    (closedBall_subset_closedBall <| le_max_left _ _)⟩


/-- If there exists a sequence of compact closed balls with the same center
such that the radii tend to infinity, then the space is proper. -/
theorem ProperSpace.of_seq_closedBall {β : Type*} {l : Filter β} [NeBot l] {x : α} {r : β → ℝ}
    (hr : Tendsto r l atTop) (hc : ∀ᶠ i in l, IsCompact (closedBall x (r i))) :
    ProperSpace α where
  isCompact_closedBall a r :=
    let ⟨_i, hci, hir⟩ := (hc.and <| hr.eventually_ge_atTop <| r + dist a x).exists
    hci.of_isClosed_subset isClosed_ball <| closedBall_subset_closedBall' hir

-- A compact pseudometric space is proper
-- see Note [lower instance priority]

instance (priority := 100) proper_of_compact [CompactSpace α] : ProperSpace α :=
  ⟨fun _ _ => isClosed_ball.isCompact⟩

-- see Note [lower instance priority]

/-- A proper space is locally compact -/
instance (priority := 100) locallyCompact_of_proper [ProperSpace α] : LocallyCompactSpace α :=
  .of_hasBasis (fun _ => nhds_basis_closedBall) fun _ _ _ =>
    isCompact_closedBall _ _

-- The `alias` command creates a definition, triggering the defLemma linter.

@[nolint defLemma, deprecated (since := "2024-11-13")]
alias locally_compact_of_proper := locallyCompact_of_proper

-- see Note [lower instance priority]

/-- A proper space is complete -/
instance (priority := 100) complete_of_proper [ProperSpace α] : CompleteSpace α :=
  ⟨fun {f} hf => by
    /- We want to show that the Cauchy filter `f` is converging. It suffices to find a closed
      ball (therefore compact by properness) where it is nontrivial. -/
    obtain ⟨t, t_fset, ht⟩ : ∃ t ∈ f, ∀ x ∈ t, ∀ y ∈ t, dist x y < 1 :=
      (Metric.cauchy_iff.1 hf).2 1 zero_lt_one
    /-
      case intro.intro
      α : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      f : Filter α
      hf : Cauchy f
      t : Set α
      t_fset : Membership.mem f t
      ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → LT.lt (Di …
      ⊢ Exists fun x => LE.le f (nhds x)
    -/
    rcases hf.1.nonempty_of_mem t_fset with ⟨x, xt⟩
    /-
      case intro.intro.intro
      α : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      f : Filter α
      hf : Cauchy f
      t : Set α
      t_fset : Membership.mem f t
      ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → LT.lt (Di …
      x : α
      xt : Membership.mem t x
      ⊢ Exists fun x => LE.le f (nhds x)
    -/
    have : closedBall x 1 ∈ f := mem_of_superset t_fset fun y yt => (ht y yt x xt).le
    rcases (isCompact_iff_totallyBounded_isComplete.1 (isCompact_closedBall x 1)).2 f hf
        (le_principal_iff.2 this) with
      ⟨y, -, hy⟩
    /-
      case intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : ProperSpace α
      f : Filter α
      hf : Cauchy f
      t : Set α
      t_fset : Membership.mem f t
      ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → LT.lt (Di …
      x : α
      xt : Membership.mem t x
      this : Membership.mem f (Metric.closedBall x 1)
      y : α
      hy : LE.le f (nhds y)
      ⊢ Exists fun x => LE.le f (nhds x)
    -/
    exact ⟨y, hy⟩⟩
    /-
      🎉 no goals
    -/


/-- A binary product of proper spaces is proper. -/
instance prod_properSpace {α : Type*} {β : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β]
    [ProperSpace α] [ProperSpace β] : ProperSpace (α × β) where
  isCompact_closedBall := by
    /-
      α✝ : Type u
      β✝ : Type v
      X : Type u_1
      ι : Type u_2
      inst✝⁴ : PseudoMetricSpace α✝
      α : Type u_3
      β : Type u_4
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : ProperSpace α
      inst✝ : ProperSpace β
      ⊢ ∀ (x : Prod α β) (r : Real), IsCompact (Metric.closedBall x r)
    -/
    rintro ⟨x, y⟩ r
    /-
      case mk
      α✝ : Type u
      β✝ : Type v
      X : Type u_1
      ι : Type u_2
      inst✝⁴ : PseudoMetricSpace α✝
      α : Type u_3
      β : Type u_4
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : ProperSpace α
      inst✝ : ProperSpace β
      x : α
      y : β
      r : Real
      ⊢ IsCompact (Metric.closedBall { fst := x, snd := y } r)
    -/
    rw [← closedBall_prod_same x y]
    /-
      case mk
      α✝ : Type u
      β✝ : Type v
      X : Type u_1
      ι : Type u_2
      inst✝⁴ : PseudoMetricSpace α✝
      α : Type u_3
      β : Type u_4
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : ProperSpace α
      inst✝ : ProperSpace β
      x : α
      y : β
      r : Real
      ⊢ IsCompact (SProd.sprod (Metric.closedBall x r) (Metric.closedBall y r))
    -/
    exact (isCompact_closedBall x r).prod (isCompact_closedBall y r)
    /-
      🎉 no goals
    -/


/-- A finite product of proper spaces is proper. -/
instance pi_properSpace {π : β → Type*} [Fintype β] [∀ b, PseudoMetricSpace (π b)]
    [h : ∀ b, ProperSpace (π b)] : ProperSpace (∀ b, π b) := by
  /-
    α : Type u
    β : Type v
    X : Type u_1
    ι : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    h : ∀ (b : β), ProperSpace (π b)
    ⊢ ProperSpace ((b : β) → π b)
  -/
  refine .of_isCompact_closedBall_of_le 0 fun x r hr => ?_
  /-
    α : Type u
    β : Type v
    X : Type u_1
    ι : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    h : ∀ (b : β), ProperSpace (π b)
    x : (b : β) → π b
    r : Real
    hr : LE.le 0 r
    ⊢ IsCompact (Metric.closedBall x r)
  -/
  rw [closedBall_pi _ hr]
  /-
    α : Type u
    β : Type v
    X : Type u_1
    ι : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    h : ∀ (b : β), ProperSpace (π b)
    x : (b : β) → π b
    r : Real
    hr : LE.le 0 r
    ⊢ IsCompact (Set.univ.pi fun b => Metric.closedBall (x b) r)
  -/
  exact isCompact_univ_pi fun _ => isCompact_closedBall _ _
  /-
    🎉 no goals
  -/


instance [PseudoMetricSpace X] [ProperSpace X] : ProperSpace (Additive X) := ‹ProperSpace X›

instance [PseudoMetricSpace X] [ProperSpace X] : ProperSpace (Multiplicative X) := ‹ProperSpace X›

instance [PseudoMetricSpace X] [ProperSpace X] : ProperSpace Xᵒᵈ := ‹ProperSpace X›

