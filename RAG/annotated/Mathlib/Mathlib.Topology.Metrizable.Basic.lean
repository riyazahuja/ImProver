/-- A topological space is *pseudo metrizable* if there exists a pseudo metric space structure
compatible with the topology. To endow such a space with a compatible distance, use
`letI : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X`. -/
class PseudoMetrizableSpace (X : Type*) [t : TopologicalSpace X] : Prop where
  exists_pseudo_metric : ∃ m : PseudoMetricSpace X, m.toUniformSpace.toTopologicalSpace = t


instance (priority := 100) _root_.PseudoMetricSpace.toPseudoMetrizableSpace {X : Type*}
    [m : PseudoMetricSpace X] : PseudoMetrizableSpace X :=
  ⟨⟨m, rfl⟩⟩


/-- Construct on a metrizable space a metric compatible with the topology. -/
noncomputable def pseudoMetrizableSpacePseudoMetric (X : Type*) [TopologicalSpace X]
    [h : PseudoMetrizableSpace X] : PseudoMetricSpace X :=
  h.exists_pseudo_metric.choose.replaceTopology h.exists_pseudo_metric.choose_spec.symm


instance pseudoMetrizableSpace_prod [PseudoMetrizableSpace X] [PseudoMetrizableSpace Y] :
    PseudoMetrizableSpace (X × Y) :=
  letI : PseudoMetricSpace X := pseudoMetrizableSpacePseudoMetric X
  letI : PseudoMetricSpace Y := pseudoMetrizableSpacePseudoMetric Y
  inferInstance


/-- Given an inducing map of a topological space into a pseudo metrizable space, the source space
is also pseudo metrizable. -/
theorem _root_.Topology.IsInducing.pseudoMetrizableSpace [PseudoMetrizableSpace Y] {f : X → Y}
    (hf : IsInducing f) : PseudoMetrizableSpace X :=
  letI : PseudoMetricSpace Y := pseudoMetrizableSpacePseudoMetric Y
  ⟨⟨hf.comapPseudoMetricSpace, rfl⟩⟩


@[deprecated (since := "2024-10-28")]
alias _root_.Inducing.pseudoMetrizableSpace := IsInducing.pseudoMetrizableSpace


/-- Every pseudo-metrizable space is first countable. -/
instance (priority := 100) PseudoMetrizableSpace.firstCountableTopology
    [h : PseudoMetrizableSpace X] : FirstCountableTopology X := by
  /-
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h : TopologicalSpace.PseudoMetrizableSpace X
    ⊢ FirstCountableTopology X
  -/
  rcases h with ⟨_, hm⟩
  /-
    case mk.intro
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : Finite ι
    inst✝ : (i : ι) → TopologicalSpace (π i)
    w✝ : PseudoMetricSpace X
    hm : Eq UniformSpace.toTopologicalSpace inst✝³
    ⊢ FirstCountableTopology X
  -/
  rw [← hm]
  exact @UniformSpace.firstCountableTopology X PseudoMetricSpace.toUniformSpace
    EMetric.instIsCountablyGeneratedUniformity


instance PseudoMetrizableSpace.subtype [PseudoMetrizableSpace X] (s : Set X) :
    PseudoMetrizableSpace s :=
  IsInducing.subtypeVal.pseudoMetrizableSpace


instance pseudoMetrizableSpace_pi [∀ i, PseudoMetrizableSpace (π i)] :
    PseudoMetrizableSpace (∀ i, π i) := by
  /-
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TopologicalSpace.PseudoMetrizableSpace (π i)
    ⊢ TopologicalSpace.PseudoMetrizableSpace ((i : ι) → π i)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TopologicalSpace.PseudoMetrizableSpace (π i)
    val✝ : Fintype ι
    ⊢ TopologicalSpace.PseudoMetrizableSpace ((i : ι) → π i)
  -/
  letI := fun i => pseudoMetrizableSpacePseudoMetric (π i)
  /-
    case intro
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TopologicalSpace.PseudoMetrizableSpace (π i)
    val✝ : Fintype ι
    this : (i : ι) → PseudoMetricSpace (π i) := fun i => TopologicalSpace.pseudoMe …
    ⊢ TopologicalSpace.PseudoMetrizableSpace ((i : ι) → π i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A topological space is metrizable if there exists a metric space structure compatible with the
topology. To endow such a space with a compatible distance, use
`letI : MetricSpace X := TopologicalSpace.metrizableSpaceMetric X`. -/
class MetrizableSpace (X : Type*) [t : TopologicalSpace X] : Prop where
  exists_metric : ∃ m : MetricSpace X, m.toUniformSpace.toTopologicalSpace = t


instance (priority := 100) _root_.MetricSpace.toMetrizableSpace {X : Type*} [m : MetricSpace X] :
    MetrizableSpace X :=
  ⟨⟨m, rfl⟩⟩


instance (priority := 100) MetrizableSpace.toPseudoMetrizableSpace [h : MetrizableSpace X] :
    PseudoMetrizableSpace X :=
  let ⟨m, hm⟩ := h.1
  ⟨⟨m.toPseudoMetricSpace, hm⟩⟩


/-- Construct on a metrizable space a metric compatible with the topology. -/
noncomputable def metrizableSpaceMetric (X : Type*) [TopologicalSpace X] [h : MetrizableSpace X] :
    MetricSpace X :=
  h.exists_metric.choose.replaceTopology h.exists_metric.choose_spec.symm


instance (priority := 100) t2Space_of_metrizableSpace [MetrizableSpace X] : T2Space X :=
  letI : MetricSpace X := metrizableSpaceMetric X
  inferInstance


instance metrizableSpace_prod [MetrizableSpace X] [MetrizableSpace Y] : MetrizableSpace (X × Y) :=
  letI : MetricSpace X := metrizableSpaceMetric X
  letI : MetricSpace Y := metrizableSpaceMetric Y
  inferInstance


/-- Given an embedding of a topological space into a metrizable space, the source space is also
metrizable. -/
theorem _root_.Topology.IsEmbedding.metrizableSpace [MetrizableSpace Y] {f : X → Y}
    (hf : IsEmbedding f) : MetrizableSpace X :=
  letI : MetricSpace Y := metrizableSpaceMetric Y
  ⟨⟨hf.comapMetricSpace f, rfl⟩⟩


@[deprecated (since := "2024-10-26")]
alias _root_.Embedding.metrizableSpace := IsEmbedding.metrizableSpace


instance MetrizableSpace.subtype [MetrizableSpace X] (s : Set X) : MetrizableSpace s :=
  IsEmbedding.subtypeVal.metrizableSpace


instance metrizableSpace_pi [∀ i, MetrizableSpace (π i)] : MetrizableSpace (∀ i, π i) := by
  /-
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TopologicalSpace.MetrizableSpace (π i)
    ⊢ TopologicalSpace.MetrizableSpace ((i : ι) → π i)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TopologicalSpace.MetrizableSpace (π i)
    val✝ : Fintype ι
    ⊢ TopologicalSpace.MetrizableSpace ((i : ι) → π i)
  -/
  letI := fun i => metrizableSpaceMetric (π i)
  /-
    case intro
    ι : Type u_1
    X : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TopologicalSpace.MetrizableSpace (π i)
    val✝ : Fintype ι
    this : (i : ι) → MetricSpace (π i) := fun i => TopologicalSpace.metrizableSpac …
    ⊢ TopologicalSpace.MetrizableSpace ((i : ι) → π i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem IsSeparable.secondCountableTopology [PseudoMetrizableSpace X] {s : Set X}
    (hs : IsSeparable s) : SecondCountableTopology s := by
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    s : Set X
    hs : TopologicalSpace.IsSeparable s
    ⊢ SecondCountableTopology ↑s
  -/
  letI := pseudoMetrizableSpacePseudoMetric X
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    s : Set X
    hs : TopologicalSpace.IsSeparable s
    this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ SecondCountableTopology ↑s
  -/
  have := hs.separableSpace
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    s : Set X
    hs : TopologicalSpace.IsSeparable s
    this✝ : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMet …
    this : TopologicalSpace.SeparableSpace ↑s
    ⊢ SecondCountableTopology ↑s
  -/
  exact UniformSpace.secondCountable_of_separable s
  /-
    🎉 no goals
  -/


instance (X : Type*) [TopologicalSpace X] [c : CompactSpace X] [MetrizableSpace X] :
    SecondCountableTopology X := by
  /-
    ι : Type u_1
    X✝ : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁵ : TopologicalSpace X✝
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : Finite ι
    inst✝² : (i : ι) → TopologicalSpace (π i)
    X : Type u_5
    inst✝¹ : TopologicalSpace X
    c : CompactSpace X
    inst✝ : TopologicalSpace.MetrizableSpace X
    ⊢ SecondCountableTopology X
  -/
  obtain ⟨_, h⟩ := MetrizableSpace.exists_metric (X := X)
  /-
    case intro
    ι : Type u_1
    X✝ : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁵ : TopologicalSpace X✝
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : Finite ι
    inst✝² : (i : ι) → TopologicalSpace (π i)
    X : Type u_5
    inst✝¹ : TopologicalSpace X
    c : CompactSpace X
    inst✝ : TopologicalSpace.MetrizableSpace X
    w✝ : MetricSpace X
    h : Eq UniformSpace.toTopologicalSpace inst✝¹
    ⊢ SecondCountableTopology X
  -/
  rw [← h] at c ⊢
  /-
    case intro
    ι : Type u_1
    X✝ : Type u_2
    Y : Type u_3
    π : ι → Type u_4
    inst✝⁵ : TopologicalSpace X✝
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : Finite ι
    inst✝² : (i : ι) → TopologicalSpace (π i)
    X : Type u_5
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.MetrizableSpace X
    w✝ : MetricSpace X
    c : CompactSpace X
    h : Eq UniformSpace.toTopologicalSpace inst✝¹
    ⊢ SecondCountableTopology X
  -/
  infer_instance
  /-
    🎉 no goals
  -/


