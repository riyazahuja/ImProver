/-- A Polish space is a topological space with second countable topology, that can be endowed
with a metric for which it is complete.
We register an instance from complete second countable metric space to polish space, and not the
other way around as this is the most common use case.

To endow a Polish space with a complete metric space structure, do `letI := upgradePolishSpace α`.
-/
class PolishSpace (α : Type*) [h : TopologicalSpace α]
    extends SecondCountableTopology α : Prop where
  complete : ∃ m : MetricSpace α, m.toUniformSpace.toTopologicalSpace = h ∧
    @CompleteSpace α m.toUniformSpace


/-- A convenience class, for a Polish space endowed with a complete metric. No instance of this
class should be registered: It should be used as `letI := upgradePolishSpace α` to endow a Polish
space with a complete metric. -/
class UpgradedPolishSpace (α : Type*) extends MetricSpace α, SecondCountableTopology α,
  CompleteSpace α


instance (priority := 100) PolishSpace.of_separableSpace_completeSpace_metrizable [UniformSpace α]
    [SeparableSpace α] [CompleteSpace α] [(𝓤 α).IsCountablyGenerated] [T0Space α] :
    PolishSpace α where
  toSecondCountableTopology := UniformSpace.secondCountable_of_separable α
  complete := ⟨UniformSpace.metricSpace α, rfl, ‹_›⟩


/-- Construct on a Polish space a metric (compatible with the topology) which is complete. -/
def polishSpaceMetric (α : Type*) [TopologicalSpace α] [h : PolishSpace α] : MetricSpace α :=
  h.complete.choose.replaceTopology h.complete.choose_spec.1.symm


theorem complete_polishSpaceMetric (α : Type*) [ht : TopologicalSpace α] [h : PolishSpace α] :
    @CompleteSpace α (polishSpaceMetric α).toUniformSpace := by
  /-
    α : Type u_3
    ht : TopologicalSpace α
    h : PolishSpace α
    ⊢ CompleteSpace α
  -/
  convert h.complete.choose_spec.2
  /-
    case h.e'_2.h.e'_2.h.e'_2
    α : Type u_3
    ht : TopologicalSpace α
    h : PolishSpace α
    ⊢ Eq (polishSpaceMetric α) ⋯.choose
  -/
  exact MetricSpace.replaceTopology_eq _ _
  /-
    🎉 no goals
  -/


/-- This definition endows a Polish space with a complete metric. Use it as:
`letI := upgradePolishSpace α`. -/
def upgradePolishSpace (α : Type*) [TopologicalSpace α] [PolishSpace α] :
    UpgradedPolishSpace α :=
  letI := polishSpaceMetric α
  { complete_polishSpaceMetric α with }


instance (priority := 100) instMetrizableSpace (α : Type*) [TopologicalSpace α] [PolishSpace α] :
    MetrizableSpace α := by
  /-
    α✝ : Type u_1
    β : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    ⊢ TopologicalSpace.MetrizableSpace α
  -/
  letI := upgradePolishSpace α
  /-
    α✝ : Type u_1
    β : Type u_2
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    this : UpgradedPolishSpace α := upgradePolishSpace α
    ⊢ TopologicalSpace.MetrizableSpace α
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided." (since := "2024-02-23")]
theorem t2Space (α : Type*) [TopologicalSpace α] [PolishSpace α] : T2Space α := inferInstance


/-- A countable product of Polish spaces is Polish. -/
instance pi_countable {ι : Type*} [Countable ι] {E : ι → Type*} [∀ i, TopologicalSpace (E i)]
    [∀ i, PolishSpace (E i)] : PolishSpace (∀ i, E i) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝² : Countable ι
    E : ι → Type u_4
    inst✝¹ : (i : ι) → TopologicalSpace (E i)
    inst✝ : ∀ (i : ι), PolishSpace (E i)
    ⊢ PolishSpace ((i : ι) → E i)
  -/
  letI := fun i => upgradePolishSpace (E i)
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝² : Countable ι
    E : ι → Type u_4
    inst✝¹ : (i : ι) → TopologicalSpace (E i)
    inst✝ : ∀ (i : ι), PolishSpace (E i)
    this : (i : ι) → UpgradedPolishSpace (E i) := fun i => upgradePolishSpace (E i)
    ⊢ PolishSpace ((i : ι) → E i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A countable disjoint union of Polish spaces is Polish. -/
instance sigma {ι : Type*} [Countable ι] {E : ι → Type*} [∀ n, TopologicalSpace (E n)]
    [∀ n, PolishSpace (E n)] : PolishSpace (Σn, E n) :=
  letI := fun n => upgradePolishSpace (E n)
  letI : MetricSpace (Σn, E n) := Sigma.metricSpace
  haveI : CompleteSpace (Σn, E n) := Sigma.completeSpace
  inferInstance


/-- The product of two Polish spaces is Polish. -/
instance prod [TopologicalSpace α] [PolishSpace α] [TopologicalSpace β] [PolishSpace β] :
    PolishSpace (α × β) :=
  letI := upgradePolishSpace α
  letI := upgradePolishSpace β
  inferInstance


/-- The disjoint union of two Polish spaces is Polish. -/
instance sum [TopologicalSpace α] [PolishSpace α] [TopologicalSpace β] [PolishSpace β] :
    PolishSpace (α ⊕ β) :=
  letI := upgradePolishSpace α
  letI := upgradePolishSpace β
  inferInstance


/-- Any nonempty Polish space is the continuous image of the fundamental space `ℕ → ℕ`. -/
theorem exists_nat_nat_continuous_surjective (α : Type*) [TopologicalSpace α] [PolishSpace α]
    [Nonempty α] : ∃ f : (ℕ → ℕ) → α, Continuous f ∧ Surjective f :=
  letI := upgradePolishSpace α
  exists_nat_nat_continuous_surjective_of_completeSpace α


/-- Given a closed embedding into a Polish space, the source space is also Polish. -/
theorem _root_.Topology.IsClosedEmbedding.polishSpace [TopologicalSpace α] [TopologicalSpace β]
    [PolishSpace β] {f : α → β} (hf : IsClosedEmbedding f) : PolishSpace α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    f : α → β
    hf : Topology.IsClosedEmbedding f
    ⊢ PolishSpace α
  -/
  letI := upgradePolishSpace β
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    f : α → β
    hf : Topology.IsClosedEmbedding f
    this : UpgradedPolishSpace β := upgradePolishSpace β
    ⊢ PolishSpace α
  -/
  letI : MetricSpace α := hf.isEmbedding.comapMetricSpace f
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    f : α → β
    hf : Topology.IsClosedEmbedding f
    this✝ : UpgradedPolishSpace β := upgradePolishSpace β
    this : MetricSpace α := Topology.IsEmbedding.comapMetricSpace f ⋯
    ⊢ PolishSpace α
  -/
  haveI : SecondCountableTopology α := hf.isEmbedding.secondCountableTopology
  have : CompleteSpace α := by
    rw [completeSpace_iff_isComplete_range hf.isEmbedding.to_isometry.isUniformInducing]
    exact hf.isClosed_range.isComplete
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    f : α → β
    hf : Topology.IsClosedEmbedding f
    this✝² : UpgradedPolishSpace β := upgradePolishSpace β
    this✝¹ : MetricSpace α := Topology.IsEmbedding.comapMetricSpace f ⋯
    this✝ : SecondCountableTopology α
    this : CompleteSpace α
    ⊢ PolishSpace α
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias _root_.ClosedEmbedding.polishSpace := IsClosedEmbedding.polishSpace


/-- Any countable discrete space is Polish. -/
instance (priority := 50) polish_of_countable [TopologicalSpace α]
    [h : Countable α] [DiscreteTopology α] : PolishSpace α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    h : Countable α
    inst✝ : DiscreteTopology α
    ⊢ PolishSpace α
  -/
  obtain ⟨f, hf⟩ := h.exists_injective_nat
  have : IsClosedEmbedding f :=
    .of_continuous_injective_isClosedMap continuous_of_discreteTopology hf
      fun t _ ↦ isClosed_discrete _
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    h : Countable α
    inst✝ : DiscreteTopology α
    f : α → Nat
    hf : Function.Injective f
    this : Topology.IsClosedEmbedding f
    ⊢ PolishSpace α
  -/
  exact this.polishSpace
  /-
    🎉 no goals
  -/


/-- Pulling back a Polish topology under an equiv gives again a Polish topology. -/
theorem _root_.Equiv.polishSpace_induced [t : TopologicalSpace β] [PolishSpace β] (f : α ≃ β) :
    @PolishSpace α (t.induced f) :=
  letI : TopologicalSpace α := t.induced f
  (f.toHomeomorphOfIsInducing ⟨rfl⟩).isClosedEmbedding.polishSpace


/-- A closed subset of a Polish space is also Polish. -/
theorem _root_.IsClosed.polishSpace [TopologicalSpace α] [PolishSpace α] {s : Set α}
    (hs : IsClosed s) : PolishSpace s :=
  hs.isClosedEmbedding_subtypeVal.polishSpace


instance instPolishSpaceUniv [TopologicalSpace α] [PolishSpace α] :
    PolishSpace (univ : Set α) :=
  isClosed_univ.polishSpace


protected theorem _root_.CompletePseudometrizable.iInf {ι : Type*} [Countable ι]
    {t : ι → TopologicalSpace α} (ht₀ : ∃ t₀, @T2Space α t₀ ∧ ∀ i, t i ≤ t₀)
    (ht : ∀ i, ∃ u : UniformSpace α, CompleteSpace α ∧ 𝓤[u].IsCountablyGenerated ∧
      u.toTopologicalSpace = t i) :
    ∃ u : UniformSpace α, CompleteSpace α ∧
      𝓤[u].IsCountablyGenerated ∧ u.toTopologicalSpace = ⨅ i, t i := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    t : ι → TopologicalSpace α
    ht₀ : Exists fun t₀ => And (T2Space α) (∀ (i : ι), LE.le (t i) t₀)
    ht : ∀ (i : ι), Exists fun u => And (CompleteSpace α) (And (uniformity α).IsCo …
    ⊢ Exists fun u => And (CompleteSpace α) (And (uniformity α).IsCountablyGenerat …
  -/
  choose u hcomp hcount hut using ht
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    t : ι → TopologicalSpace α
    ht₀ : Exists fun t₀ => And (T2Space α) (∀ (i : ι), LE.le (t i) t₀)
    u : ι → UniformSpace α
    hcomp : ∀ (i : ι), CompleteSpace α
    hcount : ∀ (i : ι), (uniformity α).IsCountablyGenerated
    hut : ∀ (i : ι), Eq UniformSpace.toTopologicalSpace (t i)
    ⊢ Exists fun u => And (CompleteSpace α) (And (uniformity α).IsCountablyGenerat …
  -/
  obtain rfl : t = fun i ↦ (u i).toTopologicalSpace := (funext hut).symm
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    u : ι → UniformSpace α
    hcomp : ∀ (i : ι), CompleteSpace α
    hcount : ∀ (i : ι), (uniformity α).IsCountablyGenerated
    ht₀ : Exists fun t₀ => And (T2Space α) (∀ (i : ι), LE.le ((fun i => UniformSpa …
    hut : ∀ (i : ι), Eq UniformSpace.toTopologicalSpace ((fun i => UniformSpace.to …
    ⊢ Exists fun u_1 => And (CompleteSpace α) (And (uniformity α).IsCountablyGener …
  -/
  refine ⟨⨅ i, u i, .iInf hcomp ht₀, ?_, UniformSpace.toTopologicalSpace_iInf⟩
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    u : ι → UniformSpace α
    hcomp : ∀ (i : ι), CompleteSpace α
    hcount : ∀ (i : ι), (uniformity α).IsCountablyGenerated
    ht₀ : Exists fun t₀ => And (T2Space α) (∀ (i : ι), LE.le ((fun i => UniformSpa …
    hut : ∀ (i : ι), Eq UniformSpace.toTopologicalSpace ((fun i => UniformSpace.to …
    ⊢ (uniformity α).IsCountablyGenerated
  -/
  rw [iInf_uniformity]
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    u : ι → UniformSpace α
    hcomp : ∀ (i : ι), CompleteSpace α
    hcount : ∀ (i : ι), (uniformity α).IsCountablyGenerated
    ht₀ : Exists fun t₀ => And (T2Space α) (∀ (i : ι), LE.le ((fun i => UniformSpa …
    hut : ∀ (i : ι), Eq UniformSpace.toTopologicalSpace ((fun i => UniformSpace.to …
    ⊢ (iInf fun i => uniformity α).IsCountablyGenerated
  -/
  infer_instance
  /-
    🎉 no goals
  -/


protected theorem iInf {ι : Type*} [Countable ι] {t : ι → TopologicalSpace α}
    (ht₀ : ∃ i₀, ∀ i, t i ≤ t i₀) (ht : ∀ i, @PolishSpace α (t i)) : @PolishSpace α (⨅ i, t i) := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    t : ι → TopologicalSpace α
    ht₀ : Exists fun i₀ => ∀ (i : ι), LE.le (t i) (t i₀)
    ht : ∀ (i : ι), PolishSpace α
    ⊢ PolishSpace α
  -/
  rcases ht₀ with ⟨i₀, hi₀⟩
  rcases CompletePseudometrizable.iInf ⟨t i₀, letI := t i₀; haveI := ht i₀; inferInstance, hi₀⟩
    fun i ↦
      letI := t i; haveI := ht i; letI := upgradePolishSpace α
      ⟨inferInstance, inferInstance, inferInstance, rfl⟩
    with ⟨u, hcomp, hcount, htop⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    t : ι → TopologicalSpace α
    ht : ∀ (i : ι), PolishSpace α
    i₀ : ι
    hi₀ : ∀ (i : ι), LE.le (t i) (t i₀)
    u : UniformSpace α
    hcomp : CompleteSpace α
    hcount : (uniformity α).IsCountablyGenerated
    htop : Eq UniformSpace.toTopologicalSpace (iInf fun i => t i)
    ⊢ PolishSpace α
  -/
  rw [← htop]
  have : @SecondCountableTopology α u.toTopologicalSpace :=
    htop.symm ▸ secondCountableTopology_iInf fun i ↦ letI := t i; (ht i).toSecondCountableTopology
  have : @T1Space α u.toTopologicalSpace :=
    htop.symm ▸ t1Space_antitone (iInf_le _ i₀) (by letI := t i₀; haveI := ht i₀; infer_instance)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_3
    inst✝ : Countable ι
    t : ι → TopologicalSpace α
    ht : ∀ (i : ι), PolishSpace α
    i₀ : ι
    hi₀ : ∀ (i : ι), LE.le (t i) (t i₀)
    u : UniformSpace α
    hcomp : CompleteSpace α
    hcount : (uniformity α).IsCountablyGenerated
    htop : Eq UniformSpace.toTopologicalSpace (iInf fun i => t i)
    this✝ : SecondCountableTopology α
    this : T1Space α
    ⊢ PolishSpace α
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Given a Polish space, and countably many finer Polish topologies, there exists another Polish
topology which is finer than all of them. -/
theorem exists_polishSpace_forall_le {ι : Type*} [Countable ι] [t : TopologicalSpace α]
    [p : PolishSpace α] (m : ι → TopologicalSpace α) (hm : ∀ n, m n ≤ t)
    (h'm : ∀ n, @PolishSpace α (m n)) :
    ∃ t' : TopologicalSpace α, (∀ n, t' ≤ m n) ∧ t' ≤ t ∧ @PolishSpace α t' :=
  ⟨⨅ i : Option ι, i.elim t m, fun i ↦ iInf_le _ (some i), iInf_le _ none,
    .iInf ⟨none, Option.forall.2 ⟨le_rfl, hm⟩⟩ <| Option.forall.2 ⟨p, h'm⟩⟩


instance : PolishSpace ENNReal :=
  ENNReal.orderIsoUnitIntervalBirational.toHomeomorph.isClosedEmbedding.polishSpace


/-- A type synonym for a subset `s` of a metric space, on which we will construct another metric
for which it will be complete. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
def CompleteCopy {α : Type*} [MetricSpace α] (s : Opens α) : Type _ := s


/-- A distance on an open subset `s` of a metric space, designed to make it complete.  It is given
by `dist' x y = dist x y + |1 / dist x sᶜ - 1 / dist y sᶜ|`, where the second term blows up close to
the boundary to ensure that Cauchy sequences for `dist'` remain well inside `s`. -/
-- Porting note: in mathlib3 this was only a local instance.
instance instDist : Dist (CompleteCopy s) where
  dist x y := dist x.1 y.1 + abs (1 / infDist x.1 sᶜ - 1 / infDist y.1 sᶜ)


theorem dist_eq (x y : CompleteCopy s) :
    dist x y = dist x.1 y.1 + abs (1 / infDist x.1 sᶜ - 1 / infDist y.1 sᶜ) :=
  rfl


theorem dist_val_le_dist (x y : CompleteCopy s) : dist x.1 y.1 ≤ dist x y :=
  le_add_of_nonneg_right (abs_nonneg _)


instance : TopologicalSpace (CompleteCopy s) := inferInstanceAs (TopologicalSpace s)

instance [SecondCountableTopology α] : SecondCountableTopology (CompleteCopy s) :=
  inferInstanceAs (SecondCountableTopology s)

instance : T0Space (CompleteCopy s) := inferInstanceAs (T0Space s)


/-- A metric space structure on a subset `s` of a metric space, designed to make it complete
if `s` is open. It is given by `dist' x y = dist x y + |1 / dist x sᶜ - 1 / dist y sᶜ|`, where the
second term blows up close to the boundary to ensure that Cauchy sequences for `dist'` remain well
inside `s`.

Porting note: the definition changed to ensure that the `TopologicalSpace` structure on
`TopologicalSpace.Opens.CompleteCopy s` is definitionally equal to the original one. -/
-- Porting note: in mathlib3 this was only a local instance.
instance instMetricSpace : MetricSpace (CompleteCopy s) := by
  refine @MetricSpace.ofT0PseudoMetricSpace (CompleteCopy s)
    (.ofDistTopology dist (fun _ ↦ ?_) (fun _ _ ↦ ?_) (fun x y z ↦ ?_) fun t ↦ ?_) _
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : MetricSpace α
      s : TopologicalSpace.Opens α
      x✝ : s.CompleteCopy
      ⊢ Eq (Dist.dist x✝ x✝) 0
    -/
  · simp only [dist_eq, dist_self, one_div, sub_self, abs_zero, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : MetricSpace α
      s : TopologicalSpace.Opens α
      x✝¹ x✝ : s.CompleteCopy
      ⊢ Eq (Dist.dist x✝¹ x✝) (Dist.dist x✝ x✝¹)
    -/
  · simp only [dist_eq, dist_comm, abs_sub_comm]
    /-
      🎉 no goals
    -/
  · calc
      dist x z = dist x.1 z.1 + |1 / infDist x.1 sᶜ - 1 / infDist z.1 sᶜ| := rfl
      _ ≤ dist x.1 y.1 + dist y.1 z.1 + (|1 / infDist x.1 sᶜ - 1 / infDist y.1 sᶜ| +
            |1 / infDist y.1 sᶜ - 1 / infDist z.1 sᶜ|) :=
        add_le_add (dist_triangle _ _ _) (dist_triangle (1 / infDist _ _) _ _)
      _ = dist x y + dist y z := add_add_add_comm ..
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      inst✝ : MetricSpace α
      s : TopologicalSpace.Opens α
      t : Set s.CompleteCopy
      ⊢ Iff (IsOpen t) (∀ (x : s.CompleteCopy), Membership.mem t x → Exists fun ε => …
    -/
  · refine ⟨fun h x hx ↦ ?_, fun h ↦ isOpen_iff_mem_nhds.2 fun x hx ↦ ?_⟩
      /-
        case refine_4.refine_1
        α : Type u_1
        β : Type u_2
        inst✝ : MetricSpace α
        s : TopologicalSpace.Opens α
        t : Set s.CompleteCopy
        h : IsOpen t
        x : s.CompleteCopy
        hx : Membership.mem t x
        ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (y : s.CompleteCopy), LT.lt (Dist.dist x  …
      -/
    · rcases (Metric.isOpen_iff (α := s)).1 h x hx with ⟨ε, ε0, hε⟩
      /-
        case refine_4.refine_1.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝ : MetricSpace α
        s : TopologicalSpace.Opens α
        t : Set s.CompleteCopy
        h : IsOpen t
        x : s.CompleteCopy
        hx : Membership.mem t x
        ε : Real
        ε0 : GT.gt ε 0
        hε : HasSubset.Subset (Metric.ball x ε) t
        ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (y : s.CompleteCopy), LT.lt (Dist.dist x  …
      -/
      exact ⟨ε, ε0, fun y hy ↦ hε <| (dist_comm _ _).trans_lt <| (dist_val_le_dist _ _).trans_lt hy⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_4.refine_2
        α : Type u_1
        β : Type u_2
        inst✝ : MetricSpace α
        s : TopologicalSpace.Opens α
        t : Set s.CompleteCopy
        h : ∀ (x : s.CompleteCopy), Membership.mem t x → Exists fun ε => And (GT.gt ε  …
        x : s.CompleteCopy
        hx : Membership.mem t x
        ⊢ Membership.mem (nhds x) t
      -/
    · rcases h x hx with ⟨ε, ε0, hε⟩
      /-
        case refine_4.refine_2.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝ : MetricSpace α
        s : TopologicalSpace.Opens α
        t : Set s.CompleteCopy
        h : ∀ (x : s.CompleteCopy), Membership.mem t x → Exists fun ε => And (GT.gt ε  …
        x : s.CompleteCopy
        hx : Membership.mem t x
        ε : Real
        ε0 : GT.gt ε 0
        hε : ∀ (y : s.CompleteCopy), LT.lt (Dist.dist x y) ε → Membership.mem t y
        ⊢ Membership.mem (nhds x) t
      -/
      simp only [dist_eq, one_div] at hε
      have : Tendsto (fun y : s ↦ dist x.1 y.1 + |(infDist x.1 sᶜ)⁻¹ - (infDist y.1 sᶜ)⁻¹|)
          (𝓝 x) (𝓝 (dist x.1 x.1 + |(infDist x.1 sᶜ)⁻¹ - (infDist x.1 sᶜ)⁻¹|)) := by
        refine (tendsto_const_nhds.dist continuous_subtype_val.continuousAt).add
          (tendsto_const_nhds.sub <| ?_).abs
        refine (continuousAt_inv_infDist_pt ?_).comp continuous_subtype_val.continuousAt
        rw [s.isOpen.isClosed_compl.closure_eq, mem_compl_iff, not_not]
        exact x.2
      /-
        case refine_4.refine_2.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝ : MetricSpace α
        s : TopologicalSpace.Opens α
        t : Set s.CompleteCopy
        h : ∀ (x : s.CompleteCopy), Membership.mem t x → Exists fun ε => And (GT.gt ε  …
        x : s.CompleteCopy
        hx : Membership.mem t x
        ε : Real
        ε0 : GT.gt ε 0
        hε : ∀ (y : s.CompleteCopy), LT.lt (HAdd.hAdd (Dist.dist ↑x ↑y) (abs (HSub.hSu …
        this : Filter.Tendsto (fun y => HAdd.hAdd (Dist.dist ↑x ↑y) (abs (HSub.hSub (I …
        ⊢ Membership.mem (nhds x) t
      -/
      simp only [dist_self, sub_self, abs_zero, zero_add] at this
      /-
        case refine_4.refine_2.intro.intro
        α : Type u_1
        β : Type u_2
        inst✝ : MetricSpace α
        s : TopologicalSpace.Opens α
        t : Set s.CompleteCopy
        h : ∀ (x : s.CompleteCopy), Membership.mem t x → Exists fun ε => And (GT.gt ε  …
        x : s.CompleteCopy
        hx : Membership.mem t x
        ε : Real
        ε0 : GT.gt ε 0
        hε : ∀ (y : s.CompleteCopy), LT.lt (HAdd.hAdd (Dist.dist ↑x ↑y) (abs (HSub.hSu …
        this : Filter.Tendsto (fun y => HAdd.hAdd (Dist.dist ↑x ↑y) (abs (HSub.hSub (I …
        ⊢ Membership.mem (nhds x) t
      -/
      exact mem_of_superset (this <| gt_mem_nhds ε0) hε
      /-
        🎉 no goals
      -/

-- Porting note: no longer needed because the topologies are defeq


instance instCompleteSpace [CompleteSpace α] : CompleteSpace (CompleteCopy s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MetricSpace α
    s : TopologicalSpace.Opens α
    inst✝ : CompleteSpace α
    ⊢ CompleteSpace s.CompleteCopy
  -/
  refine Metric.complete_of_convergent_controlled_sequences ((1 / 2) ^ ·) (by simp) fun u hu ↦ ?_
  have A : CauchySeq fun n => (u n).1 := by
    refine cauchySeq_of_le_tendsto_0 (fun n : ℕ => (1 / 2) ^ n) (fun n m N hNn hNm => ?_) ?_
    · exact (dist_val_le_dist (u n) (u m)).trans (hu N n m hNn hNm).le
    · exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MetricSpace α
    s : TopologicalSpace.Opens α
    inst✝ : CompleteSpace α
    u : Nat → s.CompleteCopy
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    A : CauchySeq fun n => ↑(u n)
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  obtain ⟨x, xlim⟩ : ∃ x, Tendsto (fun n => (u n).1) atTop (𝓝 x) := cauchySeq_tendsto_of_complete A
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : MetricSpace α
    s : TopologicalSpace.Opens α
    inst✝ : CompleteSpace α
    u : Nat → s.CompleteCopy
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    A : CauchySeq fun n => ↑(u n)
    x : α
    xlim : Filter.Tendsto (fun n => ↑(u n)) Filter.atTop (nhds x)
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  by_cases xs : x ∈ s
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MetricSpace α
      s : TopologicalSpace.Opens α
      inst✝ : CompleteSpace α
      u : Nat → s.CompleteCopy
      hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
      A : CauchySeq fun n => ↑(u n)
      x : α
      xlim : Filter.Tendsto (fun n => ↑(u n)) Filter.atTop (nhds x)
      xs : Membership.mem s x
      ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
    -/
  · exact ⟨⟨x, xs⟩, tendsto_subtype_rng.2 xlim⟩
    /-
      🎉 no goals
    -/
  obtain ⟨C, hC⟩ : ∃ C, ∀ n, 1 / infDist (u n).1 sᶜ < C := by
    refine ⟨(1 / 2) ^ 0 + 1 / infDist (u 0).1 sᶜ, fun n ↦ ?_⟩
    rw [← sub_lt_iff_lt_add]
    calc
      _ ≤ |1 / infDist (u n).1 sᶜ - 1 / infDist (u 0).1 sᶜ| := le_abs_self _
      _ = |1 / infDist (u 0).1 sᶜ - 1 / infDist (u n).1 sᶜ| := abs_sub_comm _ _
      _ ≤ dist (u 0) (u n) := le_add_of_nonneg_left dist_nonneg
      _ < (1 / 2) ^ 0 := hu 0 0 n le_rfl n.zero_le
  /-
    case neg.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : MetricSpace α
    s : TopologicalSpace.Opens α
    inst✝ : CompleteSpace α
    u : Nat → s.CompleteCopy
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    A : CauchySeq fun n => ↑(u n)
    x : α
    xlim : Filter.Tendsto (fun n => ↑(u n)) Filter.atTop (nhds x)
    xs : Not (Membership.mem s x)
    C : Real
    hC : ∀ (n : Nat), LT.lt (HDiv.hDiv 1 (Metric.infDist (↑(u n)) (HasCompl.compl  …
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  have Cpos : 0 < C := lt_of_le_of_lt (div_nonneg zero_le_one infDist_nonneg) (hC 0)
  have Hmem : ∀ {y}, y ∈ s ↔ 0 < infDist y sᶜ := fun {y} ↦ by
    rw [← s.isOpen.isClosed_compl.not_mem_iff_infDist_pos ⟨x, xs⟩]; exact not_not.symm
  have I : ∀ n, 1 / C ≤ infDist (u n).1 sᶜ := fun n ↦ by
    have : 0 < infDist (u n).1 sᶜ := Hmem.1 (u n).2
    rw [div_le_iff₀' Cpos]
    exact (div_le_iff₀ this).1 (hC n).le
  have I' : 1 / C ≤ infDist x sᶜ :=
    have : Tendsto (fun n => infDist (u n).1 sᶜ) atTop (𝓝 (infDist x sᶜ)) :=
      ((continuous_infDist_pt (sᶜ : Set α)).tendsto x).comp xlim
    ge_of_tendsto' this I
  /-
    case neg.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : MetricSpace α
    s : TopologicalSpace.Opens α
    inst✝ : CompleteSpace α
    u : Nat → s.CompleteCopy
    hu : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (u n) (u m)) (( …
    A : CauchySeq fun n => ↑(u n)
    x : α
    xlim : Filter.Tendsto (fun n => ↑(u n)) Filter.atTop (nhds x)
    xs : Not (Membership.mem s x)
    C : Real
    hC : ∀ (n : Nat), LT.lt (HDiv.hDiv 1 (Metric.infDist (↑(u n)) (HasCompl.compl  …
    Cpos : LT.lt 0 C
    Hmem : ∀ {y : α}, Iff (Membership.mem s y) (LT.lt 0 (Metric.infDist y (HasComp …
    I : ∀ (n : Nat), LE.le (HDiv.hDiv 1 C) (Metric.infDist (↑(u n)) (HasCompl.comp …
    I' : LE.le (HDiv.hDiv 1 C) (Metric.infDist x (HasCompl.compl ↑s))
    ⊢ Exists fun x => Filter.Tendsto u Filter.atTop (nhds x)
  -/
  exact absurd (Hmem.2 <| lt_of_lt_of_le (div_pos one_pos Cpos) I') xs
  /-
    🎉 no goals
  -/


/-- An open subset of a Polish space is also Polish. -/
theorem _root_.IsOpen.polishSpace {α : Type*} [TopologicalSpace α] [PolishSpace α] {s : Set α}
    (hs : IsOpen s) : PolishSpace s := by
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Set α
    hs : IsOpen s
    ⊢ PolishSpace ↑s
  -/
  letI := upgradePolishSpace α
  /-
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Set α
    hs : IsOpen s
    this : UpgradedPolishSpace α := upgradePolishSpace α
    ⊢ PolishSpace ↑s
  -/
  lift s to Opens α using hs
  /-
    case intro
    α : Type u_3
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    this : UpgradedPolishSpace α := upgradePolishSpace α
    s : TopologicalSpace.Opens α
    ⊢ PolishSpace ↑↑s
  -/
  exact inferInstanceAs (PolishSpace s.CompleteCopy)
  /-
    🎉 no goals
  -/


/-- A set in a topological space is clopenable if there exists a finer Polish topology for which
this set is open and closed. It turns out that this notion is equivalent to being Borel-measurable,
but this is nontrivial (see `isClopenable_iff_measurableSet`). -/
def IsClopenable [t : TopologicalSpace α] (s : Set α) : Prop :=
  ∃ t' : TopologicalSpace α, t' ≤ t ∧ @PolishSpace α t' ∧ IsClosed[t'] s ∧ IsOpen[t'] s


/-- Given a closed set `s` in a Polish space, one can construct a finer Polish topology for
which `s` is both open and closed. -/
theorem _root_.IsClosed.isClopenable [TopologicalSpace α] [PolishSpace α] {s : Set α}
    (hs : IsClosed s) : IsClopenable s := by
  /- Both sets `s` and `sᶜ` admit a Polish topology. So does their disjoint union `s ⊕ sᶜ`.
    Pulling back this topology by the canonical bijection with `α` gives the desired Polish
    topology in which `s` is both open and closed. -/
  classical
  haveI : PolishSpace s := hs.polishSpace
  let t : Set α := sᶜ
  haveI : PolishSpace t := hs.isOpen_compl.polishSpace
  let f : s ⊕ t ≃ α := Equiv.Set.sumCompl s
  have hle : TopologicalSpace.coinduced f instTopologicalSpaceSum ≤ ‹_› := by
    simp only [instTopologicalSpaceSum, coinduced_sup, coinduced_compose, sup_le_iff,
      ← continuous_iff_coinduced_le]
    exact ⟨continuous_subtype_val, continuous_subtype_val⟩
  refine ⟨.coinduced f instTopologicalSpaceSum, hle, ?_, hs.mono hle, ?_⟩
  · rw [← f.induced_symm]
    exact f.symm.polishSpace_induced
  · rw [isOpen_coinduced, isOpen_sum_iff]
    simp only [preimage_preimage, f]
    have inl (x : s) : (Equiv.Set.sumCompl s) (Sum.inl x) = x := Equiv.Set.sumCompl_apply_inl ..
    have inr (x : ↑sᶜ) : (Equiv.Set.sumCompl s) (Sum.inr x) = x := Equiv.Set.sumCompl_apply_inr ..
    simp_rw [t, inl, inr, Subtype.coe_preimage_self]
    simp only [isOpen_univ, true_and]
    rw [Subtype.preimage_coe_compl']
    simp


theorem IsClopenable.compl [TopologicalSpace α] {s : Set α} (hs : IsClopenable s) :
    IsClopenable sᶜ := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : PolishSpace.IsClopenable s
    ⊢ PolishSpace.IsClopenable (HasCompl.compl s)
  -/
  rcases hs with ⟨t, t_le, t_polish, h, h'⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    t : TopologicalSpace α
    t_le : LE.le t inst✝
    t_polish : PolishSpace α
    h : IsClosed s
    h' : IsOpen s
    ⊢ PolishSpace.IsClopenable (HasCompl.compl s)
  -/
  exact ⟨t, t_le, t_polish, @IsOpen.isClosed_compl α t s h', @IsClosed.isOpen_compl α t s h⟩
  /-
    🎉 no goals
  -/


theorem _root_.IsOpen.isClopenable [TopologicalSpace α] [PolishSpace α] {s : Set α}
    (hs : IsOpen s) : IsClopenable s := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Set α
    hs : IsOpen s
    ⊢ PolishSpace.IsClopenable s
  -/
  simpa using hs.isClosed_compl.isClopenable.compl
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: generalize for free to `[Countable ι] {s : ι → Set α}`

theorem IsClopenable.iUnion [t : TopologicalSpace α] [PolishSpace α] {s : ℕ → Set α}
    (hs : ∀ n, IsClopenable (s n)) : IsClopenable (⋃ n, s n) := by
  /-
    α : Type u_1
    t : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Nat → Set α
    hs : ∀ (n : Nat), PolishSpace.IsClopenable (s n)
    ⊢ PolishSpace.IsClopenable (Set.iUnion fun n => s n)
  -/
  choose m mt m_polish _ m_open using hs
  obtain ⟨t', t'm, -, t'_polish⟩ :
      ∃ t' : TopologicalSpace α, (∀ n : ℕ, t' ≤ m n) ∧ t' ≤ t ∧ @PolishSpace α t' :=
    exists_polishSpace_forall_le m mt m_polish
  have A : IsOpen[t'] (⋃ n, s n) := by
    apply isOpen_iUnion
    intro n
    apply t'm n
    exact m_open n
  obtain ⟨t'', t''_le, t''_polish, h1, h2⟩ : ∃ t'' : TopologicalSpace α,
      t'' ≤ t' ∧ @PolishSpace α t'' ∧ IsClosed[t''] (⋃ n, s n) ∧ IsOpen[t''] (⋃ n, s n) :=
    @IsOpen.isClopenable α t' t'_polish _ A
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    t : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Nat → Set α
    m : Nat → TopologicalSpace α
    mt : ∀ (n : Nat), LE.le (m n) t
    m_polish : ∀ (n : Nat), PolishSpace α
    h✝ : ∀ (n : Nat), IsClosed (s n)
    m_open : ∀ (n : Nat), IsOpen (s n)
    t' : TopologicalSpace α
    t'm : ∀ (n : Nat), LE.le t' (m n)
    t'_polish : PolishSpace α
    A : IsOpen (Set.iUnion fun n => s n)
    t'' : TopologicalSpace α
    t''_le : LE.le t'' t'
    t''_polish : PolishSpace α
    h1 : IsClosed (Set.iUnion fun n => s n)
    h2 : IsOpen (Set.iUnion fun n => s n)
    ⊢ PolishSpace.IsClopenable (Set.iUnion fun n => s n)
  -/
  exact ⟨t'', t''_le.trans ((t'm 0).trans (mt 0)), t''_polish, h1, h2⟩
  /-
    🎉 no goals
  -/


