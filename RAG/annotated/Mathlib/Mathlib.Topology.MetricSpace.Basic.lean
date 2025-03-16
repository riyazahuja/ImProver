instance (priority := 100) _root_.MetricSpace.instT0Space : T0Space γ where
  t0 _ _ h := eq_of_dist_eq_zero <| Metric.inseparable_iff.1 h


/-- A map between metric spaces is a uniform embedding if and only if the distance between `f x`
and `f y` is controlled in terms of the distance between `x` and `y` and conversely. -/
theorem isUniformEmbedding_iff' [MetricSpace β] {f : γ → β} :
    IsUniformEmbedding f ↔
      (∀ ε > 0, ∃ δ > 0, ∀ {a b : γ}, dist a b < δ → dist (f a) (f b) < ε) ∧
        ∀ δ > 0, ∃ ε > 0, ∀ {a b : γ}, dist (f a) (f b) < ε → dist a b < δ := by
  /-
    β : Type v
    γ : Type w
    inst✝¹ : MetricSpace γ
    inst✝ : MetricSpace β
    f : γ → β
    ⊢ Iff (IsUniformEmbedding f) (And (∀ (ε : Real), GT.gt ε 0 → Exists fun δ => A …
  -/
  rw [isUniformEmbedding_iff_isUniformInducing, isUniformInducing_iff, uniformContinuous_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_iff' := isUniformEmbedding_iff'


/-- If a `PseudoMetricSpace` is a T₀ space, then it is a `MetricSpace`. -/
abbrev _root_.MetricSpace.ofT0PseudoMetricSpace (α : Type*) [PseudoMetricSpace α] [T0Space α] :
    MetricSpace α where
  toPseudoMetricSpace := ‹_›
  eq_of_dist_eq_zero hdist := (Metric.inseparable_iff.2 hdist).eq

-- see Note [lower instance priority]

/-- A metric space induces an emetric space -/
instance (priority := 100) _root_.MetricSpace.toEMetricSpace : EMetricSpace γ :=
  .ofT0PseudoEMetricSpace γ


theorem isClosed_of_pairwise_le_dist {s : Set γ} {ε : ℝ} (hε : 0 < ε)
    (hs : s.Pairwise fun x y => ε ≤ dist x y) : IsClosed s :=
                                                        /-
                                                          γ : Type w
                                                          inst✝ : MetricSpace γ
                                                          s : Set γ
                                                          ε : Real
                                                          hε : LT.lt 0 ε
                                                          hs : s.Pairwise fun x y => LE.le ε (Dist.dist x y)
                                                          ⊢ s.Pairwise fun x y => Not (Membership.mem (setOf fun p => LT.lt (Dist.dist p …
                                                        -/
  isClosed_of_spaced_out (dist_mem_uniformity hε) <| by simpa using hs
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem isClosedEmbedding_of_pairwise_le_dist {α : Type*} [TopologicalSpace α] [DiscreteTopology α]
    {ε : ℝ} (hε : 0 < ε) {f : α → γ} (hf : Pairwise fun x y => ε ≤ dist (f x) (f y)) :
    IsClosedEmbedding f :=
                                                                 /-
                                                                   γ : Type w
                                                                   inst✝² : MetricSpace γ
                                                                   α : Type u_2
                                                                   inst✝¹ : TopologicalSpace α
                                                                   inst✝ : DiscreteTopology α
                                                                   ε : Real
                                                                   hε : LT.lt 0 ε
                                                                   f : α → γ
                                                                   hf : Pairwise fun x y => LE.le ε (Dist.dist (f x) (f y))
                                                                   ⊢ Pairwise fun x y => Not (Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 …
                                                                 -/
  isClosedEmbedding_of_spaced_out (dist_mem_uniformity hε) <| by simpa using hf
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_of_pairwise_le_dist := isClosedEmbedding_of_pairwise_le_dist


/-- If `f : β → α` sends any two distinct points to points at distance at least `ε > 0`, then
`f` is a uniform embedding with respect to the discrete uniformity on `β`. -/
theorem isUniformEmbedding_bot_of_pairwise_le_dist {β : Type*} {ε : ℝ} (hε : 0 < ε) {f : β → α}
    (hf : Pairwise fun x y => ε ≤ dist (f x) (f y)) :
                                  /-
                                    α : Type u
                                    β✝ : Type v
                                    X : Type u_1
                                    inst✝¹ : PseudoMetricSpace α
                                    γ : Type w
                                    inst✝ : MetricSpace γ
                                    x : γ
                                    s : Set γ
                                    β : Type u_2
                                    ε : Real
                                    hε : LT.lt 0 ε
                                    f : β → α
                                    hf : Pairwise fun x y => LE.le ε (Dist.dist (f x) (f y))
                                    ⊢ UniformSpace α
                                  -/
    @IsUniformEmbedding _ _ ⊥ (by infer_instance) f :=
                                  /-
                                    🎉 no goals
                                  -/
                                                                  /-
                                                                    α : Type u
                                                                    inst✝ : PseudoMetricSpace α
                                                                    β : Type u_2
                                                                    ε : Real
                                                                    hε : LT.lt 0 ε
                                                                    f : β → α
                                                                    hf : Pairwise fun x y => LE.le ε (Dist.dist (f x) (f y))
                                                                    ⊢ Pairwise fun x y => Not (Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 …
                                                                  -/
  isUniformEmbedding_of_spaced_out (dist_mem_uniformity hε) <| by simpa using hf
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_bot_of_pairwise_le_dist := isUniformEmbedding_bot_of_pairwise_le_dist


/-- One gets a metric space from an emetric space if the edistance
is everywhere finite, by pushing the edistance to reals. We set it up so that the edist and the
uniformity are defeq in the metric space and the emetric space. In this definition, the distance
is given separately, to be able to prescribe some expression which is not defeq to the push-forward
of the edistance to reals. -/
abbrev EMetricSpace.toMetricSpaceOfDist {α : Type u} [EMetricSpace α] (dist : α → α → ℝ)
    (edist_ne_top : ∀ x y : α, edist x y ≠ ⊤) (h : ∀ x y, dist x y = ENNReal.toReal (edist x y)) :
    MetricSpace α :=
  @MetricSpace.ofT0PseudoMetricSpace _
    (PseudoEMetricSpace.toPseudoMetricSpaceOfDist dist edist_ne_top h) _


/-- One gets a metric space from an emetric space if the edistance
is everywhere finite, by pushing the edistance to reals. We set it up so that the edist and the
uniformity are defeq in the metric space and the emetric space. -/
def EMetricSpace.toMetricSpace {α : Type u} [EMetricSpace α] (h : ∀ x y : α, edist x y ≠ ⊤) :
    MetricSpace α :=
  EMetricSpace.toMetricSpaceOfDist (fun x y => ENNReal.toReal (edist x y)) h fun _ _ => rfl


/-- Metric space structure pulled back by an injective function. Injectivity is necessary to
ensure that `dist x y = 0` only if `x = y`. -/
abbrev MetricSpace.induced {γ β} (f : γ → β) (hf : Function.Injective f) (m : MetricSpace β) :
    MetricSpace γ :=
  { PseudoMetricSpace.induced f m.toPseudoMetricSpace with
    eq_of_dist_eq_zero := fun h => hf (dist_eq_zero.1 h) }


/-- Pull back a metric space structure by a uniform embedding. This is a version of
`MetricSpace.induced` useful in case if the domain already has a `UniformSpace` structure. -/
abbrev IsUniformEmbedding.comapMetricSpace {α β} [UniformSpace α] [m : MetricSpace β] (f : α → β)
    (h : IsUniformEmbedding f) : MetricSpace α :=
  .replaceUniformity (.induced f h.injective m) h.comap_uniformity.symm


@[deprecated (since := "2024-10-03")]
alias UniformEmbedding.comapMetricSpace := IsUniformEmbedding.comapMetricSpace


/-- Pull back a metric space structure by an embedding. This is a version of
`MetricSpace.induced` useful in case if the domain already has a `TopologicalSpace` structure. -/
abbrev Topology.IsEmbedding.comapMetricSpace {α β} [TopologicalSpace α] [m : MetricSpace β]
    (f : α → β) (h : IsEmbedding f) : MetricSpace α :=
  .replaceTopology (.induced f h.injective m) h.eq_induced


@[deprecated (since := "2024-10-26")]
alias Embedding.comapMetricSpace := IsEmbedding.comapMetricSpace


instance Subtype.metricSpace {α : Type*} {p : α → Prop} [MetricSpace α] :
    MetricSpace (Subtype p) :=
  .induced Subtype.val Subtype.coe_injective ‹_›


@[to_additive]
instance {α : Type*} [MetricSpace α] : MetricSpace αᵐᵒᵖ :=
  MetricSpace.induced MulOpposite.unop MulOpposite.unop_injective ‹_›


/-- Instantiate the reals as a metric space. -/
instance Real.metricSpace : MetricSpace ℝ := .ofT0PseudoMetricSpace ℝ


instance : MetricSpace ℝ≥0 :=
  Subtype.metricSpace


instance [MetricSpace β] : MetricSpace (ULift β) :=
  MetricSpace.induced ULift.down ULift.down_injective ‹_›


instance Prod.metricSpaceMax [MetricSpace β] : MetricSpace (γ × β) :=
  .ofT0PseudoMetricSpace _


/-- A finite product of metric spaces is a metric space, with the sup distance. -/
instance metricSpacePi : MetricSpace (∀ b, π b) := .ofT0PseudoMetricSpace _


/-- A metric space is second countable if one can reconstruct up to any `ε>0` any element of the
space from countably many data. -/
theorem secondCountable_of_countable_discretization {α : Type u} [MetricSpace α]
    (H : ∀ ε > (0 : ℝ), ∃ (β : Type*) (_ : Encodable β) (F : α → β),
      ∀ x y, F x = F y → dist x y ≤ ε) :
    SecondCountableTopology α := by
  /-
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ⊢ SecondCountableTopology α
  -/
  refine secondCountable_of_almost_dense_set fun ε ε0 => ?_
  /-
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun s => And s.Countable (∀ (x : α), Exists fun y => And (Membership. …
  -/
  rcases H ε ε0 with ⟨β, fβ, F, hF⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u_2
    fβ : Encodable β
    F : α → β
    hF : ∀ (x y : α), Eq (F x) (F y) → LE.le (Dist.dist x y) ε
    ⊢ Exists fun s => And s.Countable (∀ (x : α), Exists fun y => And (Membership. …
  -/
  let Finv := rangeSplitting F
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u_2
    fβ : Encodable β
    F : α → β
    hF : ∀ (x y : α), Eq (F x) (F y) → LE.le (Dist.dist x y) ε
    Finv : ↑(Set.range F) → α := Set.rangeSplitting F
    ⊢ Exists fun s => And s.Countable (∀ (x : α), Exists fun y => And (Membership. …
  -/
  refine ⟨range Finv, ⟨countable_range _, fun x => ?_⟩⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u_2
    fβ : Encodable β
    F : α → β
    hF : ∀ (x y : α), Eq (F x) (F y) → LE.le (Dist.dist x y) ε
    Finv : ↑(Set.range F) → α := Set.rangeSplitting F
    x : α
    ⊢ Exists fun y => And (Membership.mem (Set.range Finv) y) (LE.le (Dist.dist x  …
  -/
  let x' := Finv ⟨F x, mem_range_self _⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u_2
    fβ : Encodable β
    F : α → β
    hF : ∀ (x y : α), Eq (F x) (F y) → LE.le (Dist.dist x y) ε
    Finv : ↑(Set.range F) → α := Set.rangeSplitting F
    x : α
    x' : α := Finv ⟨F x, ⋯⟩
    ⊢ Exists fun y => And (Membership.mem (Set.range Finv) y) (LE.le (Dist.dist x  …
  -/
  have : F x' = F x := apply_rangeSplitting F _
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : MetricSpace α
    H : ∀ (ε : Real), GT.gt ε 0 → Exists fun β => Exists fun x => Exists fun F =>  …
    ε : Real
    ε0 : GT.gt ε 0
    β : Type u_2
    fβ : Encodable β
    F : α → β
    hF : ∀ (x y : α), Eq (F x) (F y) → LE.le (Dist.dist x y) ε
    Finv : ↑(Set.range F) → α := Set.rangeSplitting F
    x : α
    x' : α := Finv ⟨F x, ⋯⟩
    this : Eq (F x') (F x)
    ⊢ Exists fun y => And (Membership.mem (Set.range Finv) y) (LE.le (Dist.dist x  …
  -/
  exact ⟨x', mem_range_self _, hF _ _ this.symm⟩
  /-
    🎉 no goals
  -/


instance SeparationQuotient.instDist {α : Type u} [PseudoMetricSpace α] :
    Dist (SeparationQuotient α) where
  dist := lift₂ dist fun x y x' y' hx hy ↦ by rw [dist_edist, dist_edist, ← edist_mk x,
    ← edist_mk x', mk_eq_mk.2 hx, mk_eq_mk.2 hy]


theorem SeparationQuotient.dist_mk {α : Type u} [PseudoMetricSpace α] (p q : α) :
    dist (mk p) (mk q) = dist p q :=
  rfl


instance SeparationQuotient.instMetricSpace {α : Type u} [PseudoMetricSpace α] :
    MetricSpace (SeparationQuotient α) :=
  EMetricSpace.toMetricSpaceOfDist dist (surjective_mk.forall₂.2 edist_ne_top) <|
    surjective_mk.forall₂.2 dist_edist


instance [MetricSpace X] : MetricSpace (Additive X) := ‹MetricSpace X›

instance [MetricSpace X] : MetricSpace (Multiplicative X) := ‹MetricSpace X›


instance MulOpposite.instMetricSpace [MetricSpace X] : MetricSpace Xᵐᵒᵖ :=
  MetricSpace.induced unop unop_injective ‹_›


instance [MetricSpace X] : MetricSpace Xᵒᵈ := ‹MetricSpace X›

