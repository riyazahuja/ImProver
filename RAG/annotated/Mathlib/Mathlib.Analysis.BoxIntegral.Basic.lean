local notation "ℝⁿ" => ι → ℝ


/-- The integral sum of `f : ℝⁿ → E` over a tagged prepartition `π` w.r.t. box-additive volume `vol`
with codomain `E →L[ℝ] F` is the sum of `vol J (f (π.tag J))` over all boxes of `π`. -/
def integralSum (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (π : TaggedPrepartition I) : F :=
  ∑ J ∈ π.boxes, vol J (f (π.tag J))


theorem integralSum_biUnionTagged (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (π : Prepartition I)
    (πi : ∀ J, TaggedPrepartition J) :
    integralSum f vol (π.biUnionTagged πi) = ∑ J ∈ π.boxes, integralSum f vol (πi J) := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π : BoxIntegral.Prepartition I
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
    ⊢ Eq (BoxIntegral.integralSum f vol (π.biUnionTagged πi)) (π.boxes.sum fun J = …
  -/
  refine (π.sum_biUnion_boxes _ _).trans <| sum_congr rfl fun J hJ => sum_congr rfl fun J' hJ' => ?_
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π : BoxIntegral.Prepartition I
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
    J : BoxIntegral.Box ι
    hJ : Membership.mem π.boxes J
    J' : BoxIntegral.Box ι
    hJ' : Membership.mem (πi J).boxes J'
    ⊢ Eq ((vol J') (f ((π.biUnionTagged πi).tag J'))) ((vol J') (f ((πi J).tag J')))
  -/
  rw [π.tag_biUnionTagged hJ hJ']
  /-
    🎉 no goals
  -/


theorem integralSum_biUnion_partition (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F)
    (π : TaggedPrepartition I) (πi : ∀ J, Prepartition J) (hπi : ∀ J ∈ π, (πi J).IsPartition) :
    integralSum f vol (π.biUnionPrepartition πi) = integralSum f vol π := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π : BoxIntegral.TaggedPrepartition I
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.Prepartition J
    hπi : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → (πi J).IsPartition
    ⊢ Eq (BoxIntegral.integralSum f vol (π.biUnionPrepartition πi)) (BoxIntegral.i …
  -/
  refine (π.sum_biUnion_boxes _ _).trans (sum_congr rfl fun J hJ => ?_)
  calc
    (∑ J' ∈ (πi J).boxes, vol J' (f (π.tag <| π.toPrepartition.biUnionIndex πi J'))) =
        ∑ J' ∈ (πi J).boxes, vol J' (f (π.tag J)) :=
      sum_congr rfl fun J' hJ' => by rw [Prepartition.biUnionIndex_of_mem _ hJ hJ']
    _ = vol J (f (π.tag J)) :=
      (vol.map ⟨⟨fun g : E →L[ℝ] F => g (f (π.tag J)), rfl⟩, fun _ _ => rfl⟩).sum_partition_boxes
        le_top (hπi J hJ)


theorem integralSum_inf_partition (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (π : TaggedPrepartition I)
    {π' : Prepartition I} (h : π'.IsPartition) :
    integralSum f vol (π.infPrepartition π') = integralSum f vol π :=
  integralSum_biUnion_partition f vol π _ fun _J hJ => h.restrict (Prepartition.le_of_mem _ hJ)


open Classical in
theorem integralSum_fiberwise {α} (g : Box ι → α) (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F)
    (π : TaggedPrepartition I) :
    (∑ y ∈ π.boxes.image g, integralSum f vol (π.filter (g · = y))) = integralSum f vol π :=
  π.sum_fiberwise g fun J => vol J (f <| π.tag J)


theorem integralSum_sub_partitions (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F)
    {π₁ π₂ : TaggedPrepartition I} (h₁ : π₁.IsPartition) (h₂ : π₂.IsPartition) :
    integralSum f vol π₁ - integralSum f vol π₂ =
      ∑ J ∈ (π₁.toPrepartition ⊓ π₂.toPrepartition).boxes,
        (vol J (f <| (π₁.infPrepartition π₂.toPrepartition).tag J) -
          vol J (f <| (π₂.infPrepartition π₁.toPrepartition).tag J)) := by
  rw [← integralSum_inf_partition f vol π₁ h₂, ← integralSum_inf_partition f vol π₂ h₁,
    integralSum, integralSum, Finset.sum_sub_distrib]
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : π₁.IsPartition
    h₂ : π₂.IsPartition
    ⊢ Eq (HSub.hSub ((π₁.infPrepartition π₂.toPrepartition).boxes.sum fun J => (vo …
  -/
  simp only [infPrepartition_toPrepartition, inf_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem integralSum_disjUnion (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) {π₁ π₂ : TaggedPrepartition I}
    (h : Disjoint π₁.iUnion π₂.iUnion) :
    integralSum f vol (π₁.disjUnion π₂ h) = integralSum f vol π₁ + integralSum f vol π₂ := by
  refine (Prepartition.sum_disj_union_boxes h _).trans
      (congr_arg₂ (· + ·) (sum_congr rfl fun J hJ => ?_) (sum_congr rfl fun J hJ => ?_))
    /-
      case refine_1
      ι : Type u
      E : Type v
      F : Type w
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      I : BoxIntegral.Box ι
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      π₁ π₂ : BoxIntegral.TaggedPrepartition I
      h : Disjoint π₁.iUnion π₂.iUnion
      J : BoxIntegral.Box ι
      hJ : Membership.mem π₁.boxes J
      ⊢ Eq ((vol J) (f ((π₁.disjUnion π₂ h).tag J))) ((vol J) (f (π₁.tag J)))
    -/
  · rw [disjUnion_tag_of_mem_left _ hJ]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u
      E : Type v
      F : Type w
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      I : BoxIntegral.Box ι
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      π₁ π₂ : BoxIntegral.TaggedPrepartition I
      h : Disjoint π₁.iUnion π₂.iUnion
      J : BoxIntegral.Box ι
      hJ : Membership.mem π₂.boxes J
      ⊢ Eq ((vol J) (f ((π₁.disjUnion π₂ h).tag J))) ((vol J) (f (π₂.tag J)))
    -/
  · rw [disjUnion_tag_of_mem_right _ hJ]
    /-
      🎉 no goals
    -/


@[simp]
theorem integralSum_add (f g : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (π : TaggedPrepartition I) :
    integralSum (f + g) vol π = integralSum f vol π + integralSum g vol π := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    f g : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π : BoxIntegral.TaggedPrepartition I
    ⊢ Eq (BoxIntegral.integralSum (HAdd.hAdd f g) vol π) (HAdd.hAdd (BoxIntegral.i …
  -/
  simp only [integralSum, Pi.add_apply, (vol _).map_add, Finset.sum_add_distrib]
  /-
    🎉 no goals
  -/


@[simp]
theorem integralSum_neg (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (π : TaggedPrepartition I) :
    integralSum (-f) vol π = -integralSum f vol π := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π : BoxIntegral.TaggedPrepartition I
    ⊢ Eq (BoxIntegral.integralSum (Neg.neg f) vol π) (Neg.neg (BoxIntegral.integra …
  -/
  simp only [integralSum, Pi.neg_apply, (vol _).map_neg, Finset.sum_neg_distrib]
  /-
    🎉 no goals
  -/


@[simp]
theorem integralSum_smul (c : ℝ) (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (π : TaggedPrepartition I) :
    integralSum (c • f) vol π = c • integralSum f vol π := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    I : BoxIntegral.Box ι
    c : Real
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    π : BoxIntegral.TaggedPrepartition I
    ⊢ Eq (BoxIntegral.integralSum (HSMul.hSMul c f) vol π) (HSMul.hSMul c (BoxInte …
  -/
  simp only [integralSum, Finset.smul_sum, Pi.smul_apply, ContinuousLinearMap.map_smul]
  /-
    🎉 no goals
  -/


/-- The predicate `HasIntegral I l f vol y` says that `y` is the integral of `f` over `I` along `l`
w.r.t. volume `vol`. This means that integral sums of `f` tend to `𝓝 y` along
`BoxIntegral.IntegrationParams.toFilteriUnion I ⊤`. -/
def HasIntegral (I : Box ι) (l : IntegrationParams) (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) (y : F) :
    Prop :=
  Tendsto (integralSum f vol) (l.toFilteriUnion I ⊤) (𝓝 y)


/-- A function is integrable if there exists a vector that satisfies the `HasIntegral`
predicate. -/
def Integrable (I : Box ι) (l : IntegrationParams) (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) :=
  ∃ y, HasIntegral I l f vol y


open Classical in
/-- The integral of a function `f` over a box `I` along a filter `l` w.r.t. a volume `vol`.
Returns zero on non-integrable functions. -/
def integral (I : Box ι) (l : IntegrationParams) (f : ℝⁿ → E) (vol : ι →ᵇᵃ E →L[ℝ] F) :=
  if h : Integrable I l f vol then h.choose else 0

-- Porting note: using the above notation ℝⁿ here causes the theorem below to be silently ignored
-- see https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Lean.204.20doesn't.20add.20lemma.20to.20the.20environment/near/363764522
-- and https://github.com/leanprover/lean4/issues/2257

/-- Reinterpret `BoxIntegral.HasIntegral` as `Filter.Tendsto`, e.g., dot-notation theorems
that are shadowed in the `BoxIntegral.HasIntegral` namespace. -/
theorem HasIntegral.tendsto (h : HasIntegral I l f vol y) :
    Tendsto (integralSum f vol) (l.toFilteriUnion I ⊤) (𝓝 y) :=
  h


/-- The `ε`-`δ` definition of `BoxIntegral.HasIntegral`. -/
theorem hasIntegral_iff : HasIntegral I l f vol y ↔
    ∀ ε > (0 : ℝ), ∃ r : ℝ≥0 → ℝⁿ → Ioi (0 : ℝ), (∀ c, l.RCond (r c)) ∧
      ∀ c π, l.MemBaseSet I c (r c) π → IsPartition π → dist (integralSum f vol π) y ≤ ε :=
  ((l.hasBasis_toFilteriUnion_top I).tendsto_iff nhds_basis_closedBall).trans <| by
    /-
      ι : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      I : BoxIntegral.Box ι
      inst✝ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      y : F
      ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And (∀ (c : NNReal), l.RCo …
    -/
    simp [@forall_swap ℝ≥0 (TaggedPrepartition I)]
    /-
      🎉 no goals
    -/


/-- Quite often it is more natural to prove an estimate of the form `a * ε`, not `ε` in the RHS of
`BoxIntegral.hasIntegral_iff`, so we provide this auxiliary lemma. -/
theorem HasIntegral.of_mul (a : ℝ)
    (h : ∀ ε : ℝ, 0 < ε → ∃ r : ℝ≥0 → ℝⁿ → Ioi (0 : ℝ), (∀ c, l.RCond (r c)) ∧ ∀ c π,
      l.MemBaseSet I c (r c) π → IsPartition π → dist (integralSum f vol π) y ≤ a * ε) :
    HasIntegral I l f vol y := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    y : F
    a : Real
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => And (∀ (c : NNReal), l.RCond (r  …
    ⊢ BoxIntegral.HasIntegral I l f vol y
  -/
  refine hasIntegral_iff.2 fun ε hε => ?_
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    y : F
    a : Real
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => And (∀ (c : NNReal), l.RCond (r  …
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  rcases exists_pos_mul_lt hε a with ⟨ε', hε', ha⟩
  /-
    case intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    y : F
    a : Real
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => And (∀ (c : NNReal), l.RCond (r  …
    ε : Real
    hε : GT.gt ε 0
    ε' : Real
    hε' : LT.lt 0 ε'
    ha : LT.lt (HMul.hMul a ε') ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  rcases h ε' hε' with ⟨r, hr, H⟩
  /-
    case intro.intro.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    y : F
    a : Real
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => And (∀ (c : NNReal), l.RCond (r  …
    ε : Real
    hε : GT.gt ε 0
    ε' : Real
    hε' : LT.lt 0 ε'
    ha : LT.lt (HMul.hMul a ε') ε
    r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
    hr : ∀ (c : NNReal), l.RCond (r c)
    H : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition I), l.MemBaseSet I c (r …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c : NNReal) (π : Box …
  -/
  exact ⟨r, hr, fun c π hπ hπp => (H c π hπ hπp).trans ha.le⟩
  /-
    🎉 no goals
  -/


theorem integrable_iff_cauchy [CompleteSpace F] :
    Integrable I l f vol ↔ Cauchy ((l.toFilteriUnion I ⊤).map (integralSum f vol)) :=
  cauchy_map_iff_exists_tendsto.symm


/-- In a complete space, a function is integrable if and only if its integral sums form a Cauchy
net. Here we restate this fact in terms of `∀ ε > 0, ∃ r, ...`. -/
theorem integrable_iff_cauchy_basis [CompleteSpace F] : Integrable I l f vol ↔
    ∀ ε > (0 : ℝ), ∃ r : ℝ≥0 → ℝⁿ → Ioi (0 : ℝ), (∀ c, l.RCond (r c)) ∧
      ∀ c₁ c₂ π₁ π₂, l.MemBaseSet I c₁ (r c₁) π₁ → π₁.IsPartition → l.MemBaseSet I c₂ (r c₂) π₂ →
        π₂.IsPartition → dist (integralSum f vol π₁) (integralSum f vol π₂) ≤ ε := by
  rw [integrable_iff_cauchy, cauchy_map_iff',
    (l.hasBasis_toFilteriUnion_top _).prod_self.tendsto_iff uniformity_basis_dist_le]
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And (∀ (c : NNReal), l.RCo …
  -/
  refine forall₂_congr fun ε _ => exists_congr fun r => ?_
  simp only [exists_prop, Prod.forall, Set.mem_iUnion, exists_imp, prod_mk_mem_set_prod_eq, and_imp,
    mem_inter_iff, mem_setOf_eq]
  exact
    and_congr Iff.rfl
      ⟨fun H c₁ c₂ π₁ π₂ h₁ hU₁ h₂ hU₂ => H π₁ π₂ c₁ h₁ hU₁ c₂ h₂ hU₂,
        fun H π₁ π₂ c₁ h₁ hU₁ c₂ h₂ hU₂ => H c₁ c₂ π₁ π₂ h₁ hU₁ h₂ hU₂⟩


theorem HasIntegral.mono {l₁ l₂ : IntegrationParams} (h : HasIntegral I l₁ f vol y) (hl : l₂ ≤ l₁) :
    HasIntegral I l₂ f vol y :=
  h.mono_left <| IntegrationParams.toFilteriUnion_mono _ hl _


protected theorem Integrable.hasIntegral (h : Integrable I l f vol) :
    HasIntegral I l f vol (integral I l f vol) := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ⊢ BoxIntegral.HasIntegral I l f vol (BoxIntegral.integral I l f vol)
  -/
  rw [integral, dif_pos h]
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ⊢ BoxIntegral.HasIntegral I l f vol (Exists.choose h)
  -/
  exact Classical.choose_spec h
  /-
    🎉 no goals
  -/


theorem Integrable.mono {l'} (h : Integrable I l f vol) (hle : l' ≤ l) : Integrable I l' f vol :=
  ⟨_, h.hasIntegral.mono hle⟩


theorem HasIntegral.unique (h : HasIntegral I l f vol y) (h' : HasIntegral I l f vol y') : y = y' :=
  tendsto_nhds_unique h h'


theorem HasIntegral.integrable (h : HasIntegral I l f vol y) : Integrable I l f vol :=
  ⟨_, h⟩


theorem HasIntegral.integral_eq (h : HasIntegral I l f vol y) : integral I l f vol = y :=
  h.integrable.hasIntegral.unique h


nonrec theorem HasIntegral.add (h : HasIntegral I l f vol y) (h' : HasIntegral I l g vol y') :
    HasIntegral I l (f + g) vol (y + y') := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f g : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    y y' : F
    h : BoxIntegral.HasIntegral I l f vol y
    h' : BoxIntegral.HasIntegral I l g vol y'
    ⊢ BoxIntegral.HasIntegral I l (HAdd.hAdd f g) vol (HAdd.hAdd y y')
  -/
  simpa only [HasIntegral, ← integralSum_add] using h.add h'
  /-
    🎉 no goals
  -/


theorem Integrable.add (hf : Integrable I l f vol) (hg : Integrable I l g vol) :
    Integrable I l (f + g) vol :=
  (hf.hasIntegral.add hg.hasIntegral).integrable


theorem integral_add (hf : Integrable I l f vol) (hg : Integrable I l g vol) :
    integral I l (f + g) vol = integral I l f vol + integral I l g vol :=
  (hf.hasIntegral.add hg.hasIntegral).integral_eq


nonrec theorem HasIntegral.neg (hf : HasIntegral I l f vol y) : HasIntegral I l (-f) vol (-y) := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    y : F
    hf : BoxIntegral.HasIntegral I l f vol y
    ⊢ BoxIntegral.HasIntegral I l (Neg.neg f) vol (Neg.neg y)
  -/
  simpa only [HasIntegral, ← integralSum_neg] using hf.neg
  /-
    🎉 no goals
  -/


theorem Integrable.neg (hf : Integrable I l f vol) : Integrable I l (-f) vol :=
  hf.hasIntegral.neg.integrable


theorem Integrable.of_neg (hf : Integrable I l (-f) vol) : Integrable I l f vol :=
  neg_neg f ▸ hf.neg


@[simp]
theorem integrable_neg : Integrable I l (-f) vol ↔ Integrable I l f vol :=
  ⟨fun h => h.of_neg, fun h => h.neg⟩


@[simp]
theorem integral_neg : integral I l (-f) vol = -integral I l f vol := by
  classical
  exact if h : Integrable I l f vol then h.hasIntegral.neg.integral_eq
  else by rw [integral, integral, dif_neg h, dif_neg (mt Integrable.of_neg h), neg_zero]


theorem HasIntegral.sub (h : HasIntegral I l f vol y) (h' : HasIntegral I l g vol y') :
                                               /-
                                                 ι : Type u
                                                 E : Type v
                                                 F : Type w
                                                 inst✝⁴ : NormedAddCommGroup E
                                                 inst✝³ : NormedSpace Real E
                                                 inst✝² : NormedAddCommGroup F
                                                 inst✝¹ : NormedSpace Real F
                                                 I : BoxIntegral.Box ι
                                                 inst✝ : Fintype ι
                                                 l : BoxIntegral.IntegrationParams
                                                 f g : (ι → Real) → E
                                                 vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
                                                 y y' : F
                                                 h : BoxIntegral.HasIntegral I l f vol y
                                                 h' : BoxIntegral.HasIntegral I l g vol y'
                                                 ⊢ BoxIntegral.HasIntegral I l (HSub.hSub f g) vol (HSub.hSub y y')
                                               -/
    HasIntegral I l (f - g) vol (y - y') := by simpa only [sub_eq_add_neg] using h.add h'.neg
                                               /-
                                                 🎉 no goals
                                               -/


theorem Integrable.sub (hf : Integrable I l f vol) (hg : Integrable I l g vol) :
    Integrable I l (f - g) vol :=
  (hf.hasIntegral.sub hg.hasIntegral).integrable


theorem integral_sub (hf : Integrable I l f vol) (hg : Integrable I l g vol) :
    integral I l (f - g) vol = integral I l f vol - integral I l g vol :=
  (hf.hasIntegral.sub hg.hasIntegral).integral_eq


theorem hasIntegral_const (c : E) : HasIntegral I l (fun _ => c) vol (vol I c) :=
  tendsto_const_nhds.congr' <| (l.eventually_isPartition I).mono fun _π hπ => Eq.symm <|
    (vol.map ⟨⟨fun g : E →L[ℝ] F ↦ g c, rfl⟩, fun _ _ ↦ rfl⟩).sum_partition_boxes le_top hπ


@[simp]
theorem integral_const (c : E) : integral I l (fun _ => c) vol = vol I c :=
  (hasIntegral_const c).integral_eq


theorem integrable_const (c : E) : Integrable I l (fun _ => c) vol :=
  ⟨_, hasIntegral_const c⟩


theorem hasIntegral_zero : HasIntegral I l (fun _ => (0 : E)) vol 0 := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    ⊢ BoxIntegral.HasIntegral I l (fun x => 0) vol 0
  -/
  simpa only [← (vol I).map_zero] using hasIntegral_const (0 : E)
  /-
    🎉 no goals
  -/


theorem integrable_zero : Integrable I l (fun _ => (0 : E)) vol :=
  ⟨0, hasIntegral_zero⟩


theorem integral_zero : integral I l (fun _ => (0 : E)) vol = 0 :=
  hasIntegral_zero.integral_eq


theorem HasIntegral.sum {α : Type*} {s : Finset α} {f : α → ℝⁿ → E} {g : α → F}
    (h : ∀ i ∈ s, HasIntegral I l (f i) vol (g i)) :
    HasIntegral I l (fun x => ∑ i ∈ s, f i x) vol (∑ i ∈ s, g i) := by
  classical
  induction' s using Finset.induction_on with a s ha ihs; · simp [hasIntegral_zero]
  simp only [Finset.sum_insert ha]; rw [Finset.forall_mem_insert] at h
  exact h.1.add (ihs h.2)


theorem HasIntegral.smul (hf : HasIntegral I l f vol y) (c : ℝ) :
    HasIntegral I l (c • f) vol (c • y) := by
  simpa only [HasIntegral, ← integralSum_smul] using
    (tendsto_const_nhds : Tendsto _ _ (𝓝 c)).smul hf


theorem Integrable.smul (hf : Integrable I l f vol) (c : ℝ) : Integrable I l (c • f) vol :=
  (hf.hasIntegral.smul c).integrable


theorem Integrable.of_smul {c : ℝ} (hf : Integrable I l (c • f) vol) (hc : c ≠ 0) :
    Integrable I l f vol := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : Real
    hf : BoxIntegral.Integrable I l (HSMul.hSMul c f) vol
    hc : Ne c 0
    ⊢ BoxIntegral.Integrable I l f vol
  -/
  simpa [inv_smul_smul₀ hc] using hf.smul c⁻¹
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_smul (c : ℝ) : integral I l (fun x => c • f x) vol = c • integral I l f vol := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : Real
    ⊢ Eq (BoxIntegral.integral I l (fun x => HSMul.hSMul c (f x)) vol) (HSMul.hSMu …
  -/
  rcases eq_or_ne c 0 with (rfl | hc); · simp only [zero_smul, integral_zero]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : Real
    hc : Ne c 0
    ⊢ Eq (BoxIntegral.integral I l (fun x => HSMul.hSMul c (f x)) vol) (HSMul.hSMu …
  -/
  by_cases hf : Integrable I l f vol
    /-
      case pos
      ι : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      I : BoxIntegral.Box ι
      inst✝ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      c : Real
      hc : Ne c 0
      hf : BoxIntegral.Integrable I l f vol
      ⊢ Eq (BoxIntegral.integral I l (fun x => HSMul.hSMul c (f x)) vol) (HSMul.hSMu …
    -/
  · exact (hf.hasIntegral.smul c).integral_eq
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      I : BoxIntegral.Box ι
      inst✝ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      c : Real
      hc : Ne c 0
      hf : Not (BoxIntegral.Integrable I l f vol)
      ⊢ Eq (BoxIntegral.integral I l (fun x => HSMul.hSMul c (f x)) vol) (HSMul.hSMu …
    -/
  · have : ¬Integrable I l (fun x => c • f x) vol := mt (fun h => h.of_smul hc) hf
    /-
      case neg
      ι : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      I : BoxIntegral.Box ι
      inst✝ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      c : Real
      hc : Ne c 0
      hf : Not (BoxIntegral.Integrable I l f vol)
      this : Not (BoxIntegral.Integrable I l (fun x => HSMul.hSMul c (f x)) vol)
      ⊢ Eq (BoxIntegral.integral I l (fun x => HSMul.hSMul c (f x)) vol) (HSMul.hSMu …
    -/
    rw [integral, integral, dif_neg hf, dif_neg this, smul_zero]
    /-
      🎉 no goals
    -/


/-- The integral of a nonnegative function w.r.t. a volume generated by a locally-finite measure is
nonnegative. -/
theorem integral_nonneg {g : ℝⁿ → ℝ} (hg : ∀ x ∈ Box.Icc I, 0 ≤ g x) (μ : Measure ℝⁿ)
    [IsLocallyFiniteMeasure μ] : 0 ≤ integral I l g μ.toBoxAdditive.toSMul := by
  /-
    ι : Type u
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    g : (ι → Real) → Real
    hg : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le 0 (g x)
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ LE.le 0 (BoxIntegral.integral I l g μ.toBoxAdditive.toSMul)
  -/
  by_cases hgi : Integrable I l g μ.toBoxAdditive.toSMul
    /-
      case pos
      ι : Type u
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      g : (ι → Real) → Real
      hg : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le 0 (g x)
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hgi : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      ⊢ LE.le 0 (BoxIntegral.integral I l g μ.toBoxAdditive.toSMul)
    -/
  · refine ge_of_tendsto' hgi.hasIntegral fun π => sum_nonneg fun J _ => ?_
    /-
      case pos
      ι : Type u
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      g : (ι → Real) → Real
      hg : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le 0 (g x)
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hgi : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      π : BoxIntegral.TaggedPrepartition I
      J : BoxIntegral.Box ι
      x✝ : Membership.mem π.boxes J
      ⊢ LE.le 0 ((μ.toBoxAdditive.toSMul J) (g (π.tag J)))
    -/
    exact mul_nonneg ENNReal.toReal_nonneg (hg _ <| π.tag_mem_Icc _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      g : (ι → Real) → Real
      hg : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le 0 (g x)
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hgi : Not (BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul)
      ⊢ LE.le 0 (BoxIntegral.integral I l g μ.toBoxAdditive.toSMul)
    -/
  · rw [integral, dif_neg hgi]
    /-
      🎉 no goals
    -/


/-- If `‖f x‖ ≤ g x` on `[l, u]` and `g` is integrable, then the norm of the integral of `f` is less
than or equal to the integral of `g`. -/
theorem norm_integral_le_of_norm_le {g : ℝⁿ → ℝ} (hle : ∀ x ∈ Box.Icc I, ‖f x‖ ≤ g x)
    (μ : Measure ℝⁿ) [IsLocallyFiniteMeasure μ] (hg : Integrable I l g μ.toBoxAdditive.toSMul) :
    ‖(integral I l f μ.toBoxAdditive.toSMul : E)‖ ≤ integral I l g μ.toBoxAdditive.toSMul := by
  /-
    ι : Type u
    E : Type v
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    g : (ι → Real) → Real
    hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm …
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hg : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
    ⊢ LE.le (Norm.norm (BoxIntegral.integral I l f μ.toBoxAdditive.toSMul)) (BoxIn …
  -/
  by_cases hfi : Integrable.{u, v, v} I l f μ.toBoxAdditive.toSMul
    /-
      case pos
      ι : Type u
      E : Type v
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      g : (ι → Real) → Real
      hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm …
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hg : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      hfi : BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul
      ⊢ LE.le (Norm.norm (BoxIntegral.integral I l f μ.toBoxAdditive.toSMul)) (BoxIn …
    -/
  · refine le_of_tendsto_of_tendsto' hfi.hasIntegral.norm hg.hasIntegral fun π => ?_
    /-
      case pos
      ι : Type u
      E : Type v
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      g : (ι → Real) → Real
      hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm …
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hg : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      hfi : BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul
      π : BoxIntegral.TaggedPrepartition I
      ⊢ LE.le (Norm.norm (BoxIntegral.integralSum f μ.toBoxAdditive.toSMul π)) (BoxI …
    -/
    refine norm_sum_le_of_le _ fun J _ => ?_
    simp only [BoxAdditiveMap.toSMul_apply, norm_smul, smul_eq_mul, Real.norm_eq_abs,
      μ.toBoxAdditive_apply, abs_of_nonneg ENNReal.toReal_nonneg]
    /-
      case pos
      ι : Type u
      E : Type v
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      g : (ι → Real) → Real
      hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm …
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hg : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      hfi : BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul
      π : BoxIntegral.TaggedPrepartition I
      J : BoxIntegral.Box ι
      x✝ : Membership.mem π.boxes J
      ⊢ LE.le (HMul.hMul (μ ↑J).toReal (Norm.norm (f (π.tag J)))) (HMul.hMul (μ ↑J). …
    -/
    exact mul_le_mul_of_nonneg_left (hle _ <| π.tag_mem_Icc _) ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      E : Type v
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      g : (ι → Real) → Real
      hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm …
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hg : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      hfi : Not (BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul)
      ⊢ LE.le (Norm.norm (BoxIntegral.integral I l f μ.toBoxAdditive.toSMul)) (BoxIn …
    -/
  · rw [integral, dif_neg hfi, norm_zero]
    /-
      case neg
      ι : Type u
      E : Type v
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      I : BoxIntegral.Box ι
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f : (ι → Real) → E
      g : (ι → Real) → Real
      hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm …
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      hg : BoxIntegral.Integrable I l g μ.toBoxAdditive.toSMul
      hfi : Not (BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul)
      ⊢ LE.le 0 (BoxIntegral.integral I l g μ.toBoxAdditive.toSMul)
    -/
    exact integral_nonneg (fun x hx => (norm_nonneg _).trans (hle x hx)) μ
    /-
      🎉 no goals
    -/


theorem norm_integral_le_of_le_const {c : ℝ}
    (hc : ∀ x ∈ Box.Icc I, ‖f x‖ ≤ c) (μ : Measure ℝⁿ) [IsLocallyFiniteMeasure μ] :
    ‖(integral I l f μ.toBoxAdditive.toSMul : E)‖ ≤ (μ I).toReal * c := by
  /-
    ι : Type u
    E : Type v
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    c : Real
    hc : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ LE.le (Norm.norm (BoxIntegral.integral I l f μ.toBoxAdditive.toSMul)) (HMul. …
  -/
  simpa only [integral_const] using norm_integral_le_of_norm_le hc μ (integrable_const c)
  /-
    🎉 no goals
  -/


/-- If `ε > 0`, then `BoxIntegral.Integrable.convergenceR` is a function `r : ℝ≥0 → ℝⁿ → (0, ∞)`
such that for every `c : ℝ≥0`, for every tagged partition `π` subordinate to `r` (and satisfying
additional distortion estimates if `BoxIntegral.IntegrationParams.bDistortion l = true`), the
corresponding integral sum is `ε`-close to the integral.

If `BoxIntegral.IntegrationParams.bRiemann = true`, then `r c x` does not depend on `x`. If
`ε ≤ 0`, then we use `r c x = 1`. -/
def convergenceR (h : Integrable I l f vol) (ε : ℝ) : ℝ≥0 → ℝⁿ → Ioi (0 : ℝ) :=
  if hε : 0 < ε then (hasIntegral_iff.1 h.hasIntegral ε hε).choose
  else fun _ _ => ⟨1, Set.mem_Ioi.2 zero_lt_one⟩


theorem convergenceR_cond (h : Integrable I l f vol) (ε : ℝ) (c : ℝ≥0) :
    l.RCond (h.convergenceR ε c) := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ε : Real
    c : NNReal
    ⊢ l.RCond (h.convergenceR ε c)
  -/
  rw [convergenceR]; split_ifs with h₀
  /-
    case pos
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ε : Real
    c : NNReal
    h₀ : LT.lt 0 ε
    ⊢ l.RCond (⋯.choose c)
  -/
  exacts [(hasIntegral_iff.1 h.hasIntegral ε h₀).choose_spec.1 _, fun _ x => rfl]
  /-
    🎉 no goals
  -/


theorem dist_integralSum_integral_le_of_memBaseSet (h : Integrable I l f vol) (h₀ : 0 < ε)
    (hπ : l.MemBaseSet I c (h.convergenceR ε c) π) (hπp : π.IsPartition) :
    dist (integralSum f vol π) (integral I l f vol) ≤ ε := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    h : BoxIntegral.Integrable I l f vol
    h₀ : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    hπp : π.IsPartition
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (BoxIntegral.integral I l …
  -/
  rw [convergenceR, dif_pos h₀] at hπ
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    h : BoxIntegral.Integrable I l f vol
    h₀ : LT.lt 0 ε
    hπ : l.MemBaseSet I c (⋯.choose c) π
    hπp : π.IsPartition
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (BoxIntegral.integral I l …
  -/
  exact (hasIntegral_iff.1 h.hasIntegral ε h₀).choose_spec.2 c _ hπ hπp
  /-
    🎉 no goals
  -/


/-- **Henstock-Sacks inequality**. Let `r₁ r₂ : ℝⁿ → (0, ∞)` be a function such that for any tagged
*partition* of `I` subordinate to `rₖ`, `k=1,2`, the integral sum of `f` over this partition differs
from the integral of `f` by at most `εₖ`. Then for any two tagged *prepartition* `π₁ π₂` subordinate
to `r₁` and `r₂` respectively and covering the same part of `I`, the integral sums of `f` over these
prepartitions differ from each other by at most `ε₁ + ε₂`.

The actual statement

- uses `BoxIntegral.Integrable.convergenceR` instead of a predicate assumption on `r`;
- uses `BoxIntegral.IntegrationParams.MemBaseSet` instead of “subordinate to `r`” to
  account for additional requirements like being a Henstock partition or having a bounded
  distortion.

See also `BoxIntegral.Integrable.dist_integralSum_sum_integral_le_of_memBaseSet_of_iUnion_eq` and
`BoxIntegral.Integrable.dist_integralSum_sum_integral_le_of_memBaseSet`.
-/
theorem dist_integralSum_le_of_memBaseSet (h : Integrable I l f vol) (hpos₁ : 0 < ε₁)
    (hpos₂ : 0 < ε₂) (h₁ : l.MemBaseSet I c₁ (h.convergenceR ε₁ c₁) π₁)
    (h₂ : l.MemBaseSet I c₂ (h.convergenceR ε₂ c₂) π₂) (HU : π₁.iUnion = π₂.iUnion) :
    dist (integralSum f vol π₁) (integralSum f vol π₂) ≤ ε₁ + ε₂ := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c₁ c₂ : NNReal
    ε₁ ε₂ : Real
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h : BoxIntegral.Integrable I l f vol
    hpos₁ : LT.lt 0 ε₁
    hpos₂ : LT.lt 0 ε₂
    h₁ : l.MemBaseSet I c₁ (h.convergenceR ε₁ c₁) π₁
    h₂ : l.MemBaseSet I c₂ (h.convergenceR ε₂ c₂) π₂
    HU : Eq π₁.iUnion π₂.iUnion
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π₁) (BoxIntegral.integralSum …
  -/
  rcases h₁.exists_common_compl h₂ HU with ⟨π, hπU, hπc₁, hπc₂⟩
  /-
    case intro.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c₁ c₂ : NNReal
    ε₁ ε₂ : Real
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h : BoxIntegral.Integrable I l f vol
    hpos₁ : LT.lt 0 ε₁
    hpos₂ : LT.lt 0 ε₂
    h₁ : l.MemBaseSet I c₁ (h.convergenceR ε₁ c₁) π₁
    h₂ : l.MemBaseSet I c₂ (h.convergenceR ε₂ c₂) π₂
    HU : Eq π₁.iUnion π₂.iUnion
    π : BoxIntegral.Prepartition I
    hπU : Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
    hπc₁ : Eq l.bDistortion Bool.true → LE.le π.distortion c₁
    hπc₂ : Eq l.bDistortion Bool.true → LE.le π.distortion c₂
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π₁) (BoxIntegral.integralSum …
  -/
  set r : ℝⁿ → Ioi (0 : ℝ) := fun x => min (h.convergenceR ε₁ c₁ x) (h.convergenceR ε₂ c₂ x)
  /-
    case intro.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c₁ c₂ : NNReal
    ε₁ ε₂ : Real
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h : BoxIntegral.Integrable I l f vol
    hpos₁ : LT.lt 0 ε₁
    hpos₂ : LT.lt 0 ε₂
    h₁ : l.MemBaseSet I c₁ (h.convergenceR ε₁ c₁) π₁
    h₂ : l.MemBaseSet I c₂ (h.convergenceR ε₂ c₂) π₂
    HU : Eq π₁.iUnion π₂.iUnion
    π : BoxIntegral.Prepartition I
    hπU : Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
    hπc₁ : Eq l.bDistortion Bool.true → LE.le π.distortion c₁
    hπc₂ : Eq l.bDistortion Bool.true → LE.le π.distortion c₂
    r : (ι → Real) → ↑(Set.Ioi 0) := fun x => Min.min (h.convergenceR ε₁ c₁ x) (h. …
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π₁) (BoxIntegral.integralSum …
  -/
  set πr := π.toSubordinate r
  have H₁ :
    dist (integralSum f vol (π₁.unionComplToSubordinate π hπU r)) (integral I l f vol) ≤ ε₁ :=
    h.dist_integralSum_integral_le_of_memBaseSet hpos₁
      (h₁.unionComplToSubordinate (fun _ _ => min_le_left _ _) hπU hπc₁)
      (isPartition_unionComplToSubordinate _ _ _ _)
  /-
    case intro.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c₁ c₂ : NNReal
    ε₁ ε₂ : Real
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h : BoxIntegral.Integrable I l f vol
    hpos₁ : LT.lt 0 ε₁
    hpos₂ : LT.lt 0 ε₂
    h₁ : l.MemBaseSet I c₁ (h.convergenceR ε₁ c₁) π₁
    h₂ : l.MemBaseSet I c₂ (h.convergenceR ε₂ c₂) π₂
    HU : Eq π₁.iUnion π₂.iUnion
    π : BoxIntegral.Prepartition I
    hπU : Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
    hπc₁ : Eq l.bDistortion Bool.true → LE.le π.distortion c₁
    hπc₂ : Eq l.bDistortion Bool.true → LE.le π.distortion c₂
    r : (ι → Real) → ↑(Set.Ioi 0) := fun x => Min.min (h.convergenceR ε₁ c₁ x) (h. …
    πr : BoxIntegral.TaggedPrepartition I := π.toSubordinate r
    H₁ : LE.le (Dist.dist (BoxIntegral.integralSum f vol (π₁.unionComplToSubordina …
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π₁) (BoxIntegral.integralSum …
  -/
  rw [HU] at hπU
  have H₂ :
    dist (integralSum f vol (π₂.unionComplToSubordinate π hπU r)) (integral I l f vol) ≤ ε₂ :=
    h.dist_integralSum_integral_le_of_memBaseSet hpos₂
      (h₂.unionComplToSubordinate (fun _ _ => min_le_right _ _) hπU hπc₂)
      (isPartition_unionComplToSubordinate _ _ _ _)
  /-
    case intro.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c₁ c₂ : NNReal
    ε₁ ε₂ : Real
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h : BoxIntegral.Integrable I l f vol
    hpos₁ : LT.lt 0 ε₁
    hpos₂ : LT.lt 0 ε₂
    h₁ : l.MemBaseSet I c₁ (h.convergenceR ε₁ c₁) π₁
    h₂ : l.MemBaseSet I c₂ (h.convergenceR ε₂ c₂) π₂
    HU : Eq π₁.iUnion π₂.iUnion
    π : BoxIntegral.Prepartition I
    hπU✝ : Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
    hπU : Eq π.iUnion (SDiff.sdiff (↑I) π₂.iUnion)
    hπc₁ : Eq l.bDistortion Bool.true → LE.le π.distortion c₁
    hπc₂ : Eq l.bDistortion Bool.true → LE.le π.distortion c₂
    r : (ι → Real) → ↑(Set.Ioi 0) := fun x => Min.min (h.convergenceR ε₁ c₁ x) (h. …
    πr : BoxIntegral.TaggedPrepartition I := π.toSubordinate r
    H₁ : LE.le (Dist.dist (BoxIntegral.integralSum f vol (π₁.unionComplToSubordina …
    H₂ : LE.le (Dist.dist (BoxIntegral.integralSum f vol (π₂.unionComplToSubordina …
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π₁) (BoxIntegral.integralSum …
  -/
  simpa [unionComplToSubordinate] using (dist_triangle_right _ _ _).trans (add_le_add H₁ H₂)
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on `I` along `l`, then for two sufficiently fine tagged prepartitions
(in the sense of the filter `BoxIntegral.IntegrationParams.toFilter l I`) such that they cover
the same part of `I`, the integral sums of `f` over `π₁` and `π₂` are very close to each other. -/
theorem tendsto_integralSum_toFilter_prod_self_inf_iUnion_eq_uniformity (h : Integrable I l f vol) :
    Tendsto (fun π : TaggedPrepartition I × TaggedPrepartition I =>
      (integralSum f vol π.1, integralSum f vol π.2))
        ((l.toFilter I ×ˢ l.toFilter I) ⊓ 𝓟 {π | π.1.iUnion = π.2.iUnion}) (𝓤 F) := by
  refine (((l.hasBasis_toFilter I).prod_self.inf_principal _).tendsto_iff
    uniformity_basis_dist_le).2 fun ε ε0 => ?_
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : Prod (BoxInteg …
  -/
  replace ε0 := half_pos ε0
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ε : Real
    ε0 : LT.lt 0 (HDiv.hDiv ε 2)
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : Prod (BoxInteg …
  -/
  use h.convergenceR (ε / 2), h.convergenceR_cond (ε / 2); rintro ⟨π₁, π₂⟩ ⟨⟨h₁, h₂⟩, hU⟩
  /-
    case right.mk.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ε : Real
    ε0 : LT.lt 0 (HDiv.hDiv ε 2)
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    hU : Membership.mem (setOf fun π => Eq π.1.iUnion π.2.iUnion) { fst := π₁, snd …
    h₁ : Membership.mem (setOf fun π => Exists fun c => l.MemBaseSet I c (h.conver …
    h₂ : Membership.mem (setOf fun π => Exists fun c => l.MemBaseSet I c (h.conver …
    ⊢ Membership.mem (setOf fun p => LE.le (Dist.dist p.1 p.2) ε) { fst := BoxInte …
  -/
  rw [← add_halves ε]
  /-
    case right.mk.intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    ε : Real
    ε0 : LT.lt 0 (HDiv.hDiv ε 2)
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    hU : Membership.mem (setOf fun π => Eq π.1.iUnion π.2.iUnion) { fst := π₁, snd …
    h₁ : Membership.mem (setOf fun π => Exists fun c => l.MemBaseSet I c (h.conver …
    h₂ : Membership.mem (setOf fun π => Exists fun c => l.MemBaseSet I c (h.conver …
    ⊢ Membership.mem (setOf fun p => LE.le (Dist.dist p.1 p.2) (HAdd.hAdd (HDiv.hD …
  -/
  exact h.dist_integralSum_le_of_memBaseSet ε0 ε0 h₁.choose_spec h₂.choose_spec hU
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on a box `I` along `l`, then for any fixed subset `s` of `I` that can be
represented as a finite union of boxes, the integral sums of `f` over tagged prepartitions that
cover exactly `s` form a Cauchy “sequence” along `l`. -/
theorem cauchy_map_integralSum_toFilteriUnion (h : Integrable I l f vol) (π₀ : Prepartition I) :
    Cauchy ((l.toFilteriUnion I π₀).map (integralSum f vol)) := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ⊢ Cauchy (Filter.map (BoxIntegral.integralSum f vol) (BoxIntegral.IntegrationP …
  -/
  refine ⟨inferInstance, ?_⟩
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ⊢ LE.le (SProd.sprod (Filter.map (BoxIntegral.integralSum f vol) (BoxIntegral. …
  -/
  rw [prod_map_map_eq, ← toFilter_inf_iUnion_eq, ← prod_inf_prod, prod_principal_principal]
  exact h.tendsto_integralSum_toFilter_prod_self_inf_iUnion_eq_uniformity.mono_left
    (inf_le_inf_left _ <| principal_mono.2 fun π h => h.1.trans h.2.symm)


theorem to_subbox_aux (h : Integrable I l f vol) (hJ : J ≤ I) :
    ∃ y : F, HasIntegral J l f vol y ∧
      Tendsto (integralSum f vol) (l.toFilteriUnion I (Prepartition.single I J hJ)) (𝓝 y) := by
  refine (cauchy_map_iff_exists_tendsto.1
    (h.cauchy_map_integralSum_toFilteriUnion (.single I J hJ))).imp fun y hy ↦ ⟨?_, hy⟩
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I J : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    hJ : LE.le J I
    y : F
    hy : Filter.Tendsto (BoxIntegral.integralSum f vol) (BoxIntegral.IntegrationPa …
    ⊢ BoxIntegral.HasIntegral J l f vol y
  -/
  convert hy.comp (l.tendsto_embedBox_toFilteriUnion_top hJ) -- faster than `exact` here
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on a box `I`, then it is integrable on any subbox of `I`. -/
theorem to_subbox (h : Integrable I l f vol) (hJ : J ≤ I) : Integrable J l f vol :=
  (h.to_subbox_aux hJ).imp fun _ => And.left


/-- If `f` is integrable on a box `I`, then integral sums of `f` over tagged prepartitions
that cover exactly a subbox `J ≤ I` tend to the integral of `f` over `J` along `l`. -/
theorem tendsto_integralSum_toFilteriUnion_single (h : Integrable I l f vol) (hJ : J ≤ I) :
    Tendsto (integralSum f vol) (l.toFilteriUnion I (Prepartition.single I J hJ))
      (𝓝 <| integral J l f vol) :=
  let ⟨_y, h₁, h₂⟩ := h.to_subbox_aux hJ
  h₁.integral_eq.symm ▸ h₂


/-- **Henstock-Sacks inequality**. Let `r : ℝⁿ → (0, ∞)` be a function such that for any tagged
*partition* of `I` subordinate to `r`, the integral sum of `f` over this partition differs from the
integral of `f` by at most `ε`. Then for any tagged *prepartition* `π` subordinate to `r`, the
integral sum of `f` over this prepartition differs from the integral of `f` over the part of `I`
covered by `π` by at most `ε`.

The actual statement

- uses `BoxIntegral.Integrable.convergenceR` instead of a predicate assumption on `r`;
- uses `BoxIntegral.IntegrationParams.MemBaseSet` instead of “subordinate to `r`” to
  account for additional requirements like being a Henstock partition or having a bounded
  distortion;
- takes an extra argument `π₀ : prepartition I` and an assumption `π.Union = π₀.Union` instead of
  using `π.to_prepartition`.
-/
theorem dist_integralSum_sum_integral_le_of_memBaseSet_of_iUnion_eq (h : Integrable I l f vol)
    (h0 : 0 < ε) (hπ : l.MemBaseSet I c (h.convergenceR ε c) π) {π₀ : Prepartition I}
    (hU : π.iUnion = π₀.iUnion) :
    dist (integralSum f vol π) (∑ J ∈ π₀.boxes, integral J l f vol) ≤ ε := by
  -- Let us prove that the distance is less than or equal to `ε + δ` for all positive `δ`.
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  refine le_of_forall_pos_le_add fun δ δ0 => ?_
  -- First we choose some constants.
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  set δ' : ℝ := δ / (#π₀.boxes + 1)
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    δ' : Real := HDiv.hDiv δ (HAdd.hAdd (↑π₀.boxes.card) 1)
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  have H0 : 0 < (#π₀.boxes + 1 : ℝ) := Nat.cast_add_one_pos _
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    δ' : Real := HDiv.hDiv δ (HAdd.hAdd (↑π₀.boxes.card) 1)
    H0 : LT.lt 0 (HAdd.hAdd (↑π₀.boxes.card) 1)
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  have δ'0 : 0 < δ' := div_pos δ0 H0
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    δ' : Real := HDiv.hDiv δ (HAdd.hAdd (↑π₀.boxes.card) 1)
    H0 : LT.lt 0 (HAdd.hAdd (↑π₀.boxes.card) 1)
    δ'0 : LT.lt 0 δ'
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  set C := max π₀.distortion π₀.compl.distortion
  /- Next we choose a tagged partition of each `J ∈ π₀` such that the integral sum of `f` over this
    partition is `δ'`-close to the integral of `f` over `J`. -/
  have : ∀ J ∈ π₀, ∃ πi : TaggedPrepartition J,
      πi.IsPartition ∧ dist (integralSum f vol πi) (integral J l f vol) ≤ δ' ∧
        l.MemBaseSet J C (h.convergenceR δ' C) πi := by
    intro J hJ
    have Hle : J ≤ I := π₀.le_of_mem hJ
    have HJi : Integrable J l f vol := h.to_subbox Hle
    set r := fun x => min (h.convergenceR δ' C x) (HJi.convergenceR δ' C x)
    have hJd : J.distortion ≤ C := le_trans (Finset.le_sup hJ) (le_max_left _ _)
    rcases l.exists_memBaseSet_isPartition J hJd r with ⟨πJ, hC, hp⟩
    have hC₁ : l.MemBaseSet J C (HJi.convergenceR δ' C) πJ := by
      refine hC.mono J le_rfl le_rfl fun x _ => ?_; exact min_le_right _ _
    have hC₂ : l.MemBaseSet J C (h.convergenceR δ' C) πJ := by
      refine hC.mono J le_rfl le_rfl fun x _ => ?_; exact min_le_left _ _
    exact ⟨πJ, hp, HJi.dist_integralSum_integral_le_of_memBaseSet δ'0 hC₁ hp, hC₂⟩
  /- Now we combine these tagged partitions into a tagged prepartition of `I` that covers the
    same part of `I` as `π₀` and apply `BoxIntegral.dist_integralSum_le_of_memBaseSet` to
    `π` and this prepartition. -/
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    δ' : Real := HDiv.hDiv δ (HAdd.hAdd (↑π₀.boxes.card) 1)
    H0 : LT.lt 0 (HAdd.hAdd (↑π₀.boxes.card) 1)
    δ'0 : LT.lt 0 δ'
    C : NNReal := Max.max π₀.distortion π₀.compl.distortion
    this : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → Exists fun πi => And π …
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  choose! πi hπip hπiδ' hπiC using this
  have : l.MemBaseSet I C (h.convergenceR δ' C) (π₀.biUnionTagged πi) :=
    biUnionTagged_memBaseSet hπiC hπip fun _ => le_max_right _ _
  have hU' : π.iUnion = (π₀.biUnionTagged πi).iUnion :=
    hU.trans (Prepartition.iUnion_biUnion_partition _ hπip).symm
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    δ' : Real := HDiv.hDiv δ (HAdd.hAdd (↑π₀.boxes.card) 1)
    H0 : LT.lt 0 (HAdd.hAdd (↑π₀.boxes.card) 1)
    δ'0 : LT.lt 0 δ'
    C : NNReal := Max.max π₀.distortion π₀.compl.distortion
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
    hπip : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → (πi J).IsPartition
    hπiδ' : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → LE.le (Dist.dist (Box …
    hπiC : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → l.MemBaseSet J C (h.co …
    this : l.MemBaseSet I C (h.convergenceR δ' C) (π₀.biUnionTagged πi)
    hU' : Eq π.iUnion (π₀.biUnionTagged πi).iUnion
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  have := h.dist_integralSum_le_of_memBaseSet h0 δ'0 hπ this hU'
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    π : BoxIntegral.TaggedPrepartition I
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    c : NNReal
    ε : Real
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    h0 : LT.lt 0 ε
    hπ : l.MemBaseSet I c (h.convergenceR ε c) π
    π₀ : BoxIntegral.Prepartition I
    hU : Eq π.iUnion π₀.iUnion
    δ : Real
    δ0 : LT.lt 0 δ
    δ' : Real := HDiv.hDiv δ (HAdd.hAdd (↑π₀.boxes.card) 1)
    H0 : LT.lt 0 (HAdd.hAdd (↑π₀.boxes.card) 1)
    δ'0 : LT.lt 0 δ'
    C : NNReal := Max.max π₀.distortion π₀.compl.distortion
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
    hπip : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → (πi J).IsPartition
    hπiδ' : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → LE.le (Dist.dist (Box …
    hπiC : ∀ (J : BoxIntegral.Box ι), Membership.mem π₀ J → l.MemBaseSet J C (h.co …
    this✝ : l.MemBaseSet I C (h.convergenceR δ' C) (π₀.biUnionTagged πi)
    hU' : Eq π.iUnion (π₀.biUnionTagged πi).iUnion
    this : LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (BoxIntegral.integra …
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f vol π) (π₀.boxes.sum fun J => Bo …
  -/
  rw [integralSum_biUnionTagged] at this
  calc
    dist (integralSum f vol π) (∑ J ∈ π₀.boxes, integral J l f vol) ≤
        dist (integralSum f vol π) (∑ J ∈ π₀.boxes, integralSum f vol (πi J)) +
          dist (∑ J ∈ π₀.boxes, integralSum f vol (πi J)) (∑ J ∈ π₀.boxes, integral J l f vol) :=
      dist_triangle _ _ _
    _ ≤ ε + δ' + ∑ _J ∈ π₀.boxes, δ' := add_le_add this (dist_sum_sum_le_of_le _ hπiδ')
    _ = ε + δ := by field_simp [δ']; ring


/-- **Henstock-Sacks inequality**. Let `r : ℝⁿ → (0, ∞)` be a function such that for any tagged
*partition* of `I` subordinate to `r`, the integral sum of `f` over this partition differs from the
integral of `f` by at most `ε`. Then for any tagged *prepartition* `π` subordinate to `r`, the
integral sum of `f` over this prepartition differs from the integral of `f` over the part of `I`
covered by `π` by at most `ε`.

The actual statement

- uses `BoxIntegral.Integrable.convergenceR` instead of a predicate assumption on `r`;
- uses `BoxIntegral.IntegrationParams.MemBaseSet` instead of “subordinate to `r`” to
  account for additional requirements like being a Henstock partition or having a bounded
  distortion;
-/
theorem dist_integralSum_sum_integral_le_of_memBaseSet (h : Integrable I l f vol) (h0 : 0 < ε)
    (hπ : l.MemBaseSet I c (h.convergenceR ε c) π) :
    dist (integralSum f vol π) (∑ J ∈ π.boxes, integral J l f vol) ≤ ε :=
  h.dist_integralSum_sum_integral_le_of_memBaseSet_of_iUnion_eq h0 hπ rfl


/-- Integral sum of `f` over a tagged prepartition `π` such that `π.Union = π₀.Union` tends to the
sum of integrals of `f` over the boxes of `π₀`. -/
theorem tendsto_integralSum_sum_integral (h : Integrable I l f vol) (π₀ : Prepartition I) :
    Tendsto (integralSum f vol) (l.toFilteriUnion I π₀)
      (𝓝 <| ∑ J ∈ π₀.boxes, integral J l f vol) := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ⊢ Filter.Tendsto (BoxIntegral.integralSum f vol) (BoxIntegral.IntegrationParam …
  -/
  refine ((l.hasBasis_toFilteriUnion I π₀).tendsto_iff nhds_basis_closedBall).2 fun ε ε0 => ?_
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  refine ⟨h.convergenceR ε, h.convergenceR_cond ε, ?_⟩
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ ∀ (x : BoxIntegral.TaggedPrepartition I), Membership.mem (setOf fun π => Exi …
  -/
  simp only [mem_inter_iff, Set.mem_iUnion, mem_setOf_eq]
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ ∀ (x : BoxIntegral.TaggedPrepartition I), (Exists fun c => And (l.MemBaseSet …
  -/
  rintro π ⟨c, hc, hU⟩
  /-
    case intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₀ : BoxIntegral.Prepartition I
    ε : Real
    ε0 : LT.lt 0 ε
    π : BoxIntegral.TaggedPrepartition I
    c : NNReal
    hc : l.MemBaseSet I c (h.convergenceR ε c) π
    hU : Eq π.iUnion π₀.iUnion
    ⊢ Membership.mem (Metric.closedBall (π₀.boxes.sum fun J => BoxIntegral.integra …
  -/
  exact h.dist_integralSum_sum_integral_le_of_memBaseSet_of_iUnion_eq ε0 hc hU
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on `I`, then `fun J ↦ integral J l f vol` is box-additive on subboxes of
`I`: if `π₁`, `π₂` are two prepartitions of `I` covering the same part of `I`, the sum of integrals
of `f` over the boxes of `π₁` is equal to the sum of integrals of `f` over the boxes of `π₂`.

See also `BoxIntegral.Integrable.toBoxAdditive` for a bundled version. -/
theorem sum_integral_congr (h : Integrable I l f vol) {π₁ π₂ : Prepartition I}
    (hU : π₁.iUnion = π₂.iUnion) :
    ∑ J ∈ π₁.boxes, integral J l f vol = ∑ J ∈ π₂.boxes, integral J l f vol := by
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₁ π₂ : BoxIntegral.Prepartition I
    hU : Eq π₁.iUnion π₂.iUnion
    ⊢ Eq (π₁.boxes.sum fun J => BoxIntegral.integral J l f vol) (π₂.boxes.sum fun  …
  -/
  refine tendsto_nhds_unique (h.tendsto_integralSum_sum_integral π₁) ?_
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₁ π₂ : BoxIntegral.Prepartition I
    hU : Eq π₁.iUnion π₂.iUnion
    ⊢ Filter.Tendsto (BoxIntegral.integralSum f vol) (BoxIntegral.IntegrationParam …
  -/
  rw [l.toFilteriUnion_congr _ hU]
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝¹ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    inst✝ : CompleteSpace F
    h : BoxIntegral.Integrable I l f vol
    π₁ π₂ : BoxIntegral.Prepartition I
    hU : Eq π₁.iUnion π₂.iUnion
    ⊢ Filter.Tendsto (BoxIntegral.integralSum f vol) (BoxIntegral.IntegrationParam …
  -/
  exact h.tendsto_integralSum_sum_integral π₂
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on `I`, then `fun J ↦ integral J l f vol` is box-additive on subboxes of
`I`: if `π₁`, `π₂` are two prepartitions of `I` covering the same part of `I`, the sum of integrals
of `f` over the boxes of `π₁` is equal to the sum of integrals of `f` over the boxes of `π₂`.

See also `BoxIntegral.Integrable.sum_integral_congr` for an unbundled version. -/
@[simps]
def toBoxAdditive (h : Integrable I l f vol) : ι →ᵇᵃ[I] F where
  toFun J := integral J l f vol
  sum_partition_boxes' J hJ π hπ := by
    /-
      ι : Type u
      E : Type v
      F : Type w
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      I J✝ : BoxIntegral.Box ι
      π✝ : BoxIntegral.TaggedPrepartition I
      inst✝¹ : Fintype ι
      l : BoxIntegral.IntegrationParams
      f g : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      y y' : F
      c c₁ c₂ : NNReal
      ε ε₁ ε₂ : Real
      π₁ π₂ : BoxIntegral.TaggedPrepartition I
      inst✝ : CompleteSpace F
      h : BoxIntegral.Integrable I l f vol
      J : BoxIntegral.Box ι
      hJ : LE.le ↑J ↑I
      π : BoxIntegral.Prepartition J
      hπ : π.IsPartition
      ⊢ Eq (π.boxes.sum fun Ji => (fun J => BoxIntegral.integral J l f vol) Ji) ((fu …
    -/
    replace hπ := hπ.iUnion_eq; rw [← Prepartition.iUnion_top] at hπ
    rw [(h.to_subbox (WithTop.coe_le_coe.1 hJ)).sum_integral_congr hπ, Prepartition.top_boxes,
      sum_singleton]


/-- A function that is bounded and a.e. continuous on a box `I` is integrable on `I`. -/
theorem integrable_of_bounded_and_ae_continuousWithinAt [CompleteSpace E] {I : Box ι} {f : ℝⁿ → E}
    (hb : ∃ C : ℝ, ∀ x ∈ Box.Icc I, ‖f x‖ ≤ C) (μ : Measure ℝⁿ) [IsLocallyFiniteMeasure μ]
    (hc : ∀ᵐ x ∂(μ.restrict (Box.Icc I)), ContinuousWithinAt f (Box.Icc I) x) :
    Integrable I l f μ.toBoxAdditive.toSMul := by
  /- We prove that f is integrable by proving that we can ensure that the integralSums over any
     two tagged prepartitions π₁ and π₂ can be made ε-close by making the partitions
     sufficiently fine.

     Start by defining some constants C, ε₁, ε₂ that will be useful later. -/
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ⊢ BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul
  -/
  refine integrable_iff_cauchy_basis.2 fun ε ε0 ↦ ?_
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  rcases exists_pos_mul_lt ε0 (2 * μ.toBoxAdditive I) with ⟨ε₁, ε₁0, hε₁⟩
  /-
    case intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    hb : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I)  …
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  rcases hb with ⟨C, hC⟩
  have C0 : 0 ≤ C := by
    obtain ⟨x, hx⟩ := BoxIntegral.Box.nonempty_coe I
    exact le_trans (norm_nonneg (f x)) <| hC x (I.coe_subset_Icc hx)
  /-
    case intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  rcases exists_pos_mul_lt ε0 (4 * C) with ⟨ε₂, ε₂0, hε₂⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  have ε₂0' : ENNReal.ofReal ε₂ ≠ 0 := ne_of_gt <| ofReal_pos.2 ε₂0

  -- The set of discontinuities of f is contained in an open set U with μ U < ε₂.
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  let D := { x ∈ Box.Icc I | ¬ ContinuousWithinAt f (Box.Icc I) x }
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  let μ' := μ.restrict (Box.Icc I)
  have μ'D : μ' D = 0 := by
    rcases eventually_iff_exists_mem.1 hc with ⟨V, ae, hV⟩
    exact eq_of_le_of_not_lt (mem_ae_iff.1 ae ▸ (μ'.mono <| fun x h xV ↦ h.2 (hV x xV))) not_lt_zero
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  obtain ⟨U, UD, Uopen, hU⟩ := Set.exists_isOpen_lt_add D (show μ' D ≠ ⊤ by simp [μ'D]) ε₂0'
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (HAdd.hAdd (μ' D) (ENNReal.ofReal ε₂))
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  rw [μ'D, zero_add] at hU

  /- Box.Icc I \ U is compact and avoids discontinuities of f, so there exists r > 0 such that for
     every x ∈ Box.Icc I \ U, the oscillation (within Box.Icc I) of f on the ball of radius r
     centered at x is ≤ ε₁ -/
  have comp : IsCompact (Box.Icc I \ U) :=
    I.isCompact_Icc.of_isClosed_subset (I.isCompact_Icc.isClosed.sdiff Uopen) Set.diff_subset
  have : ∀ x ∈ (Box.Icc I \ U), oscillationWithin f (Box.Icc I) x < (ENNReal.ofReal ε₁) := by
    intro x hx
    suffices oscillationWithin f (Box.Icc I) x = 0 by rw [this]; exact ofReal_pos.2 ε₁0
    simpa [OscillationWithin.eq_zero_iff_continuousWithinAt, D, hx.1] using hx.2 ∘ (fun a ↦ UD a)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  rcases comp.uniform_oscillationWithin this with ⟨r, r0, hr⟩

  /- We prove the claim for partitions π₁ and π₂ subordinate to r/2, by writing the difference as
     an integralSum over π₁ ⊓ π₂ and considering separately the boxes of π₁ ⊓ π₂ which are/aren't
     fully contained within U. -/
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    r : Real
    r0 : GT.gt r 0
    hr : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U)  …
    ⊢ Exists fun r => And (∀ (c : NNReal), l.RCond (r c)) (∀ (c₁ c₂ : NNReal) (π₁  …
  -/
  refine ⟨fun _ _ ↦ ⟨r / 2, half_pos r0⟩, fun _ _ _ ↦ rfl, fun c₁ c₂ π₁ π₂ h₁ h₁p h₂ h₂p ↦ ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    r : Real
    r0 : GT.gt r 0
    hr : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U)  …
    c₁ c₂ : NNReal
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₁) π₁
    h₁p : π₁.IsPartition
    h₂ : l.MemBaseSet I c₂ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₂) π₂
    h₂p : π₂.IsPartition
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum f μ.toBoxAdditive.toSMul π₁) (BoxI …
  -/
  simp only [dist_eq_norm, integralSum_sub_partitions _ _ h₁p h₂p, toSMul_apply, ← smul_sub]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    r : Real
    r0 : GT.gt r 0
    hr : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U)  …
    c₁ c₂ : NNReal
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₁) π₁
    h₁p : π₁.IsPartition
    h₂ : l.MemBaseSet I c₂ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₂) π₂
    h₂p : π₂.IsPartition
    ⊢ LE.le (Norm.norm ((Min.min π₁.toPrepartition π₂.toPrepartition).boxes.sum fu …
  -/
  have μI : μ I < ⊤ := lt_of_le_of_lt (μ.mono I.coe_subset_Icc) I.isCompact_Icc.measure_lt_top
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    r : Real
    r0 : GT.gt r 0
    hr : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U)  …
    c₁ c₂ : NNReal
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₁) π₁
    h₁p : π₁.IsPartition
    h₂ : l.MemBaseSet I c₂ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₂) π₂
    h₂p : π₂.IsPartition
    μI : LT.lt (μ ↑I) Top.top
    ⊢ LE.le (Norm.norm ((Min.min π₁.toPrepartition π₂.toPrepartition).boxes.sum fu …
  -/
  let t₁ (J : Box ι) : ℝⁿ := (π₁.infPrepartition π₂.toPrepartition).tag J
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    r : Real
    r0 : GT.gt r 0
    hr : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U)  …
    c₁ c₂ : NNReal
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₁) π₁
    h₁p : π₁.IsPartition
    h₂ : l.MemBaseSet I c₂ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₂) π₂
    h₂p : π₂.IsPartition
    μI : LT.lt (μ ↑I) Top.top
    t₁ : BoxIntegral.Box ι → ι → Real := fun J => (π₁.infPrepartition π₂.toPrepart …
    ⊢ LE.le (Norm.norm ((Min.min π₁.toPrepartition π₂.toPrepartition).boxes.sum fu …
  -/
  let t₂ (J : Box ι) : ℝⁿ := (π₂.infPrepartition π₁.toPrepartition).tag J
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hc : Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I)  …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    hε₁ : LT.lt (HMul.hMul (HMul.hMul 2 (μ.toBoxAdditive I)) ε₁) ε
    C : Real
    hC : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (Norm. …
    C0 : LE.le 0 C
    ε₂ : Real
    ε₂0 : LT.lt 0 ε₂
    hε₂ : LT.lt (HMul.hMul (HMul.hMul 4 C) ε₂) ε
    ε₂0' : Ne (ENNReal.ofReal ε₂) 0
    D : Set (ι → Real) := setOf fun x => And (Membership.mem (BoxIntegral.Box.Icc  …
    μ' : MeasureTheory.Measure (ι → Real) := μ.restrict (BoxIntegral.Box.Icc I)
    μ'D : Eq (μ' D) 0
    U : Set (ι → Real)
    UD : Superset U D
    Uopen : IsOpen U
    hU : LT.lt (μ' U) (ENNReal.ofReal ε₂)
    comp : IsCompact (SDiff.sdiff (BoxIntegral.Box.Icc I) U)
    this : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U …
    r : Real
    r0 : GT.gt r 0
    hr : ∀ (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) U)  …
    c₁ c₂ : NNReal
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₁) π₁
    h₁p : π₁.IsPartition
    h₂ : l.MemBaseSet I c₂ ((fun x x => ⟨HDiv.hDiv r 2, ⋯⟩) c₂) π₂
    h₂p : π₂.IsPartition
    μI : LT.lt (μ ↑I) Top.top
    t₁ : BoxIntegral.Box ι → ι → Real := fun J => (π₁.infPrepartition π₂.toPrepart …
    t₂ : BoxIntegral.Box ι → ι → Real := fun J => (π₂.infPrepartition π₁.toPrepart …
    ⊢ LE.le (Norm.norm ((Min.min π₁.toPrepartition π₂.toPrepartition).boxes.sum fu …
  -/
  let B := (π₁.toPrepartition ⊓ π₂.toPrepartition).boxes
  classical
  let B' := {J ∈ B | J.toSet ⊆ U}
  have hB' : B' ⊆ B := B.filter_subset (fun J ↦ J.toSet ⊆ U)
  have μJ_ne_top : ∀ J ∈ B, μ J ≠ ⊤ :=
    fun J hJ ↦ lt_top_iff_ne_top.1 <| lt_of_le_of_lt (μ.mono (Prepartition.le_of_mem' _ J hJ)) μI
  have un : ∀ S ⊆ B, ⋃ J ∈ S, J.toSet ⊆ I.toSet :=
    fun S hS ↦ iUnion_subset_iff.2 (fun J ↦ iUnion_subset_iff.2 fun hJ ↦ le_of_mem' _ J (hS hJ))
  rw [← sum_sdiff hB', ← add_halves ε]
  apply le_trans (norm_add_le _ _) (add_le_add ?_ ?_)

  /- If a box J is not contained within U, then the oscillation of f on J is small, which bounds
     the contribution of J to the overall sum. -/
  · have : ∀ J ∈ B \ B', ‖μ.toBoxAdditive J • (f (t₁ J) - f (t₂ J))‖ ≤ μ.toBoxAdditive J * ε₁ := by
      intro J hJ
      rw [mem_sdiff, B.mem_filter, not_and] at hJ
      rw [norm_smul, μ.toBoxAdditive_apply, Real.norm_of_nonneg toReal_nonneg]
      refine mul_le_mul_of_nonneg_left ?_ toReal_nonneg
      obtain ⟨x, xJ, xnU⟩ : ∃ x ∈ J, x ∉ U := Set.not_subset.1 (hJ.2 hJ.1)
      have hx : x ∈ Box.Icc I \ U := ⟨Box.coe_subset_Icc ((le_of_mem' _ J hJ.1) xJ), xnU⟩
      have ineq : edist (f (t₁ J)) (f (t₂ J)) ≤ EMetric.diam (f '' (ball x r ∩ (Box.Icc I))) := by
        apply edist_le_diam_of_mem <;>
          refine Set.mem_image_of_mem f ⟨?_, tag_mem_Icc _ J⟩ <;>
          refine closedBall_subset_ball (div_two_lt_of_pos r0) <| mem_closedBall_comm.1 ?_
        · exact h₁.isSubordinate.infPrepartition π₂.toPrepartition J hJ.1 (Box.coe_subset_Icc xJ)
        · exact h₂.isSubordinate.infPrepartition π₁.toPrepartition J
            ((π₁.mem_infPrepartition_comm).1 hJ.1) (Box.coe_subset_Icc xJ)
      rw [← emetric_ball] at ineq
      simpa only [edist_le_ofReal (le_of_lt ε₁0), dist_eq_norm, hJ.1] using ineq.trans (hr x hx)
    refine (norm_sum_le _ _).trans <| (sum_le_sum this).trans ?_
    rw [← sum_mul]
    trans μ.toBoxAdditive I * ε₁; swap
    · linarith
    simp_rw [mul_le_mul_right ε₁0, μ.toBoxAdditive_apply]
    refine le_trans ?_ <| toReal_mono (lt_top_iff_ne_top.1 μI) <| μ.mono <| un (B \ B') sdiff_subset
    rw [← toReal_sum (fun J hJ ↦ μJ_ne_top J (mem_sdiff.1 hJ).1), ← Finset.tsum_subtype]
    refine (toReal_mono <| ne_of_lt <| lt_of_le_of_lt (μ.mono <| un (B \ B') sdiff_subset) μI) ?_
    refine le_of_eq (measure_biUnion (countable_toSet _) ?_ (fun J _ ↦ J.measurableSet_coe)).symm
    exact fun J hJ J' hJ' hJJ' ↦ pairwiseDisjoint _ (mem_sdiff.1 hJ).1 (mem_sdiff.1 hJ').1 hJJ'

  -- The contribution of the boxes contained within U is bounded because f is bounded and μ U < ε₂.
  · have : ∀ J ∈ B', ‖μ.toBoxAdditive J • (f (t₁ J) - f (t₂ J))‖ ≤ μ.toBoxAdditive J * (2 * C) := by
      intro J _
      rw [norm_smul, μ.toBoxAdditive_apply, Real.norm_of_nonneg toReal_nonneg, two_mul]
      refine mul_le_mul_of_nonneg_left (le_trans (norm_sub_le _ _) (add_le_add ?_ ?_)) (by simp) <;>
        exact hC _ (TaggedPrepartition.tag_mem_Icc _ J)
    apply (norm_sum_le_of_le B' this).trans
    simp_rw [← sum_mul, μ.toBoxAdditive_apply, ← toReal_sum (fun J hJ ↦ μJ_ne_top J (hB' hJ))]
    suffices (∑ J in B', μ J).toReal ≤ ε₂ by
      linarith [mul_le_mul_of_nonneg_right this <| (mul_nonneg_iff_of_pos_left two_pos).2 C0]
    rw [← toReal_ofReal (le_of_lt ε₂0)]
    refine toReal_mono ofReal_ne_top (le_trans ?_ (le_of_lt hU))
    trans μ' (⋃ J ∈ B', J)
    · simp only [μ', μ.restrict_eq_self <| (un _ hB').trans I.coe_subset_Icc]
      exact le_of_eq <| Eq.symm <| measure_biUnion_finset
        (fun J hJ K hK hJK ↦ pairwiseDisjoint _ (hB' hJ) (hB' hK) hJK) fun J _ ↦ J.measurableSet_coe
    · apply μ'.mono
      simp_rw [iUnion_subset_iff]
      exact fun J hJ ↦ (mem_filter.1 hJ).2


/-- A function that is bounded on a box `I` and a.e. continuous is integrable on `I`.

This is a version of `integrable_of_bounded_and_ae_continuousWithinAt` with a stronger continuity
assumption so that the user does not need to specialize the continuity assumption to each box on
which the theorem is to be applied. -/
theorem integrable_of_bounded_and_ae_continuous [CompleteSpace E] {I : Box ι} {f : ℝⁿ → E}
    (hb : ∃ C : ℝ, ∀ x ∈ Box.Icc I, ‖f x‖ ≤ C) (μ : Measure ℝⁿ) [IsLocallyFiniteMeasure μ]
    (hc : ∀ᵐ x ∂μ, ContinuousAt f x) : Integrable I l f μ.toBoxAdditive.toSMul :=
  integrable_of_bounded_and_ae_continuousWithinAt l hb μ <|
    Eventually.filter_mono (ae_mono μ.restrict_le_self) (hc.mono fun _ h ↦ h.continuousWithinAt)



/-- A continuous function is box-integrable with respect to any locally finite measure.

This is true for any volume with bounded variation. -/
theorem integrable_of_continuousOn [CompleteSpace E] {I : Box ι} {f : ℝⁿ → E}
    (hc : ContinuousOn f (Box.Icc I)) (μ : Measure ℝⁿ) [IsLocallyFiniteMeasure μ] :
    Integrable.{u, v, v} I l f μ.toBoxAdditive.toSMul := by
  /-
    ι : Type u
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Fintype ι
    l : BoxIntegral.IntegrationParams
    inst✝¹ : CompleteSpace E
    I : BoxIntegral.Box ι
    f : (ι → Real) → E
    hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    μ : MeasureTheory.Measure (ι → Real)
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ BoxIntegral.Integrable I l f μ.toBoxAdditive.toSMul
  -/
  apply integrable_of_bounded_and_ae_continuousWithinAt
  · obtain ⟨C, hC⟩ := (NormedSpace.isBounded_iff_subset_smul_closedBall ℝ).1
                        (I.isCompact_Icc.image_of_continuousOn hc).isBounded
    use ‖C‖, fun x hx ↦ by
      simpa only [smul_unitClosedBall, mem_closedBall_zero_iff] using hC (Set.mem_image_of_mem f hx)
    /-
      case hc
      ι : Type u
      E : Type v
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : Fintype ι
      l : BoxIntegral.IntegrationParams
      inst✝¹ : CompleteSpace E
      I : BoxIntegral.Box ι
      f : (ι → Real) → E
      hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      ⊢ Filter.Eventually (fun x => ContinuousWithinAt f (BoxIntegral.Box.Icc I) x)  …
    -/
  · refine eventually_of_mem ?_ (fun x hx ↦ hc.continuousWithinAt hx)
    /-
      case hc
      ι : Type u
      E : Type v
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : Fintype ι
      l : BoxIntegral.IntegrationParams
      inst✝¹ : CompleteSpace E
      I : BoxIntegral.Box ι
      f : (ι → Real) → E
      hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      μ : MeasureTheory.Measure (ι → Real)
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      ⊢ Membership.mem (MeasureTheory.ae (μ.restrict (BoxIntegral.Box.Icc I))) (BoxI …
    -/
                                          /-
                                            🎉 no goals
                                          -/
    rw [mem_ae_iff, μ.restrict_apply] <;> simp [MeasurableSet.compl_iff.2 I.measurableSet_Icc]
                                          /-
                                            🎉 no goals
                                          -/


/-- This is an auxiliary lemma used to prove two statements at once. Use one of the next two
lemmas instead. -/
theorem HasIntegral.of_bRiemann_eq_false_of_forall_isLittleO (hl : l.bRiemann = false)
    (B : ι →ᵇᵃ[I] ℝ) (hB0 : ∀ J, 0 ≤ B J) (g : ι →ᵇᵃ[I] F) (s : Set ℝⁿ) (hs : s.Countable)
    (hlH : s.Nonempty → l.bHenstock = true)
    (H₁ : ∀ (c : ℝ≥0), ∀ x ∈ Box.Icc I ∩ s, ∀ ε > (0 : ℝ),
      ∃ δ > 0, ∀ J ≤ I, Box.Icc J ⊆ Metric.closedBall x δ → x ∈ Box.Icc J →
        (l.bDistortion → J.distortion ≤ c) → dist (vol J (f x)) (g J) ≤ ε)
    (H₂ : ∀ (c : ℝ≥0), ∀ x ∈ Box.Icc I \ s, ∀ ε > (0 : ℝ),
      ∃ δ > 0, ∀ J ≤ I, Box.Icc J ⊆ Metric.closedBall x δ → (l.bHenstock → x ∈ Box.Icc J) →
        (l.bDistortion → J.distortion ≤ c) → dist (vol J (f x)) (g J) ≤ ε * B J) :
    HasIntegral I l f vol (g I) := by
  /- We choose `r x` differently for `x ∈ s` and `x ∉ s`.

    For `x ∈ s`, we choose `εs` such that `∑' x : s, εs x < ε / 2 / 2 ^ #ι`, then choose `r x` so
    that `dist (vol J (f x)) (g J) ≤ εs x` for `J` in the `r x`-neighborhood of `x`. This guarantees
    that the sum of these distances over boxes `J` such that `π.tag J ∈ s` is less than `ε / 2`. We
    need an additional multiplier `2 ^ #ι` because different boxes can have the same tag.

    For `x ∉ s`, we choose `r x` so that `dist (vol (J (f x))) (g J) ≤ (ε / 2 / B I) * B J` for a
    box `J` in the `δ`-neighborhood of `x`. -/
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    H₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral.B …
    H₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.B …
    ⊢ BoxIntegral.HasIntegral I l f vol (g I)
  -/
  refine ((l.hasBasis_toFilteriUnion_top _).tendsto_iff Metric.nhds_basis_closedBall).2 ?_
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    H₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral.B …
    H₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.B …
    ⊢ ∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And (∀ (c : NNReal), l.RCond (i …
  -/
  intro ε ε0
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    H₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral.B …
    H₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.B …
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  simp only [← exists_prop, gt_iff_lt, Subtype.exists'] at H₁ H₂
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    ε : Real
    ε0 : LT.lt 0 ε
    H₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral.B …
    H₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.B …
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  choose! δ₁ Hδ₁ using H₁
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    ε : Real
    ε0 : LT.lt 0 ε
    H₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.B …
    δ₁ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral. …
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  choose! δ₂ Hδ₂ using H₂
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    ε : Real
    ε0 : LT.lt 0 ε
    δ₁ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral. …
    δ₂ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral. …
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  have ε0' := half_pos ε0; have H0 : 0 < (2 : ℝ) ^ Fintype.card ι := pow_pos zero_lt_two _
  /-
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    ε : Real
    ε0 : LT.lt 0 ε
    δ₁ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral. …
    δ₂ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral. …
    ε0' : LT.lt 0 (HDiv.hDiv ε 2)
    H0 : LT.lt 0 (HPow.hPow 2 (Fintype.card ι))
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  rcases hs.exists_pos_forall_sum_le (div_pos ε0' H0) with ⟨εs, hεs0, hεs⟩
  /-
    case intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    ε : Real
    ε0 : LT.lt 0 ε
    δ₁ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral. …
    δ₂ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral. …
    ε0' : LT.lt 0 (HDiv.hDiv ε 2)
    H0 : LT.lt 0 (HPow.hPow 2 (Fintype.card ι))
    εs : (ι → Real) → Real
    hεs0 : ∀ (i : ι → Real), LT.lt 0 (εs i)
    hεs : ∀ (t : Finset (ι → Real)), HasSubset.Subset (↑t) s → LE.le (t.sum fun i  …
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  simp only [le_div_iff₀' H0, mul_sum] at hεs
  /-
    case intro.intro
    ι : Type u
    E : Type v
    F : Type w
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    I : BoxIntegral.Box ι
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    f : (ι → Real) → E
    vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
    hl : Eq l.bRiemann Bool.false
    B : BoxIntegral.BoxAdditiveMap ι Real ↑I
    hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
    g : BoxIntegral.BoxAdditiveMap ι F ↑I
    s : Set (ι → Real)
    hs : s.Countable
    hlH : s.Nonempty → Eq l.bHenstock Bool.true
    ε : Real
    ε0 : LT.lt 0 ε
    δ₁ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral. …
    δ₂ : NNReal → (ι → Real) → Real → Subtype fun a => LT.lt 0 a
    Hδ₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral. …
    ε0' : LT.lt 0 (HDiv.hDiv ε 2)
    H0 : LT.lt 0 (HPow.hPow 2 (Fintype.card ι))
    εs : (ι → Real) → Real
    hεs0 : ∀ (i : ι → Real), LT.lt 0 (εs i)
    hεs : ∀ (t : Finset (ι → Real)), HasSubset.Subset (↑t) s → LE.le (t.sum fun i  …
    ⊢ Exists fun ia => And (∀ (c : NNReal), l.RCond (ia c)) (∀ (x : BoxIntegral.Ta …
  -/
  rcases exists_pos_mul_lt ε0' (B I) with ⟨ε', ε'0, hεI⟩
  classical
  set δ : ℝ≥0 → ℝⁿ → Ioi (0 : ℝ) := fun c x => if x ∈ s then δ₁ c x (εs x) else (δ₂ c) x ε'
  refine ⟨δ, fun c => l.rCond_of_bRiemann_eq_false hl, ?_⟩
  simp only [Set.mem_iUnion, mem_inter_iff, mem_setOf_eq]
  rintro π ⟨c, hπδ, hπp⟩
  -- Now we split the sum into two parts based on whether `π.tag J` belongs to `s` or not.
  rw [← g.sum_partition_boxes le_rfl hπp, Metric.mem_closedBall, integralSum,
    ← sum_filter_add_sum_filter_not π.boxes fun J => π.tag J ∈ s,
    ← sum_filter_add_sum_filter_not π.boxes fun J => π.tag J ∈ s, ← add_halves ε]
  refine dist_add_add_le_of_le ?_ ?_
  · rcases s.eq_empty_or_nonempty with (rfl | hsne); · simp [ε0'.le]
    /- For the boxes such that `π.tag J ∈ s`, we use the fact that at most `2 ^ #ι` boxes have the
        same tag. -/
    specialize hlH hsne
    have : ∀ J ∈ {J ∈ π.boxes | π.tag J ∈ s},
        dist (vol J (f <| π.tag J)) (g J) ≤ εs (π.tag J) := fun J hJ ↦ by
      rw [Finset.mem_filter] at hJ; cases' hJ with hJ hJs
      refine Hδ₁ c _ ⟨π.tag_mem_Icc _, hJs⟩ _ (hεs0 _) _ (π.le_of_mem' _ hJ) ?_
        (hπδ.2 hlH J hJ) fun hD => (Finset.le_sup hJ).trans (hπδ.3 hD)
      convert hπδ.1 J hJ using 3; exact (if_pos hJs).symm
    refine (dist_sum_sum_le_of_le _ this).trans ?_
    rw [sum_comp]
    refine (sum_le_sum ?_).trans (hεs _ ?_)
    · rintro b -
      rw [← Nat.cast_two, ← Nat.cast_pow, ← nsmul_eq_mul]
      refine nsmul_le_nsmul_left (hεs0 _).le ?_
      refine (Finset.card_le_card ?_).trans ((hπδ.isHenstock hlH).card_filter_tag_eq_le b)
      exact filter_subset_filter _ (filter_subset _ _)
    · rw [Finset.coe_image, Set.image_subset_iff]
      exact fun J hJ => (Finset.mem_filter.1 hJ).2
  /- Now we deal with boxes such that `π.tag J ∉ s`.
    In this case the estimate is straightforward. -/
  -- Porting note: avoided strange elaboration issues by rewriting using `calc`
  calc
    dist (∑ J ∈ π.boxes with ¬tag π J ∈ s, vol J (f (tag π J)))
      (∑ J ∈ π.boxes with ¬tag π J ∈ s, g J)
      ≤ ∑ J ∈ π.boxes with ¬tag π J ∈ s, ε' * B J := dist_sum_sum_le_of_le _ fun J hJ ↦ by
      rw [Finset.mem_filter] at hJ; cases' hJ with hJ hJs
      refine Hδ₂ c _ ⟨π.tag_mem_Icc _, hJs⟩ _ ε'0 _ (π.le_of_mem' _ hJ) ?_ (fun hH => hπδ.2 hH J hJ)
        fun hD => (Finset.le_sup hJ).trans (hπδ.3 hD)
      convert hπδ.1 J hJ using 3; exact (if_neg hJs).symm
    _ ≤ ∑ J ∈ π.boxes, ε' * B J := sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ ↦
      mul_nonneg ε'0.le (hB0 _)
    _ = B I * ε' := by rw [← mul_sum, B.sum_partition_boxes le_rfl hπp, mul_comm]
    _ ≤ ε / 2 := hεI.le


/-- A function `f` has Henstock (or `⊥`) integral over `I` is equal to the value of a box-additive
function `g` on `I` provided that `vol J (f x)` is sufficiently close to `g J` for sufficiently
small boxes `J ∋ x`. This lemma is useful to prove, e.g., to prove the Divergence theorem for
integral along `⊥`.

Let `l` be either `BoxIntegral.IntegrationParams.Henstock` or `⊥`. Let `g` a box-additive function
on subboxes of `I`. Suppose that there exists a nonnegative box-additive function `B` and a
countable set `s` with the following property.

For every `c : ℝ≥0`, a point `x ∈ I.Icc`, and a positive `ε` there exists `δ > 0` such that for any
box `J ≤ I` such that

- `x ∈ J.Icc ⊆ Metric.closedBall x δ`;
- if `l.bDistortion` (i.e., `l = ⊥`), then the distortion of `J` is less than or equal to `c`,

the distance between the term `vol J (f x)` of an integral sum corresponding to `J` and `g J` is
less than or equal to `ε` if `x ∈ s` and is less than or equal to `ε * B J` otherwise.

Then `f` is integrable on `I` along `l` with integral `g I`. -/
theorem HasIntegral.of_le_Henstock_of_forall_isLittleO (hl : l ≤ Henstock) (B : ι →ᵇᵃ[I] ℝ)
    (hB0 : ∀ J, 0 ≤ B J) (g : ι →ᵇᵃ[I] F) (s : Set ℝⁿ) (hs : s.Countable)
    (H₁ : ∀ (c : ℝ≥0), ∀ x ∈ Box.Icc I ∩ s, ∀ ε > (0 : ℝ),
      ∃ δ > 0, ∀ J ≤ I, Box.Icc J ⊆ Metric.closedBall x δ → x ∈ Box.Icc J →
        (l.bDistortion → J.distortion ≤ c) → dist (vol J (f x)) (g J) ≤ ε)
    (H₂ : ∀ (c : ℝ≥0), ∀ x ∈ Box.Icc I \ s, ∀ ε > (0 : ℝ),
      ∃ δ > 0, ∀ J ≤ I, Box.Icc J ⊆ Metric.closedBall x δ → x ∈ Box.Icc J →
        (l.bDistortion → J.distortion ≤ c) → dist (vol J (f x)) (g J) ≤ ε * B J) :
    HasIntegral I l f vol (g I) :=
  have A : l.bHenstock := Bool.eq_true_of_true_le hl.2.1
  HasIntegral.of_bRiemann_eq_false_of_forall_isLittleO (Bool.eq_false_of_le_false hl.1) B hB0 _ s hs
                          /-
                            ι : Type u
                            E : Type v
                            F : Type w
                            inst✝⁴ : NormedAddCommGroup E
                            inst✝³ : NormedSpace Real E
                            inst✝² : NormedAddCommGroup F
                            inst✝¹ : NormedSpace Real F
                            I : BoxIntegral.Box ι
                            inst✝ : Fintype ι
                            l : BoxIntegral.IntegrationParams
                            f : (ι → Real) → E
                            vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
                            hl : LE.le l BoxIntegral.IntegrationParams.Henstock
                            B : BoxIntegral.BoxAdditiveMap ι Real ↑I
                            hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
                            g : BoxIntegral.BoxAdditiveMap ι F ↑I
                            s : Set (ι → Real)
                            hs : s.Countable
                            H₁ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (Inter.inter (BoxIntegral.B …
                            H₂ : ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.B …
                            A : Eq l.bHenstock Bool.true
                            ⊢ ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box. …
                          -/
    (fun _ => A) H₁ <| by simpa only [A, true_imp_iff] using H₂
                          /-
                            🎉 no goals
                          -/


/-- Suppose that there exists a nonnegative box-additive function `B` with the following property.

For every `c : ℝ≥0`, a point `x ∈ I.Icc`, and a positive `ε` there exists `δ > 0` such that for any
box `J ≤ I` such that

- `J.Icc ⊆ Metric.closedBall x δ`;
- if `l.bDistortion` (i.e., `l = ⊥`), then the distortion of `J` is less than or equal to `c`,

the distance between the term `vol J (f x)` of an integral sum corresponding to `J` and `g J` is
less than or equal to `ε * B J`.

Then `f` is McShane integrable on `I` with integral `g I`. -/
theorem HasIntegral.mcShane_of_forall_isLittleO (B : ι →ᵇᵃ[I] ℝ) (hB0 : ∀ J, 0 ≤ B J)
    (g : ι →ᵇᵃ[I] F) (H : ∀ (_ : ℝ≥0), ∀ x ∈ Box.Icc I, ∀ ε > (0 : ℝ), ∃ δ > 0, ∀ J ≤ I,
      Box.Icc J ⊆ Metric.closedBall x δ → dist (vol J (f x)) (g J) ≤ ε * B J) :
    HasIntegral I McShane f vol (g I) :=
  (HasIntegral.of_bRiemann_eq_false_of_forall_isLittleO (l := McShane) rfl B hB0 g ∅ countable_empty
      (fun ⟨_x, hx⟩ => hx.elim) fun _ _ hx => hx.2.elim) <| by
    /-
      ι : Type u
      E : Type v
      F : Type w
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Real F
      I : BoxIntegral.Box ι
      inst✝ : Fintype ι
      f : (ι → Real) → E
      vol : BoxIntegral.BoxAdditiveMap ι (ContinuousLinearMap (RingHom.id Real) E F) …
      B : BoxIntegral.BoxAdditiveMap ι Real ↑I
      hB0 : ∀ (J : BoxIntegral.Box ι), LE.le 0 (B J)
      g : BoxIntegral.BoxAdditiveMap ι F ↑I
      H : NNReal → ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → ∀ (ε …
      ⊢ ∀ (c : NNReal) (x : ι → Real), Membership.mem (SDiff.sdiff (BoxIntegral.Box. …
    -/
    simpa only [McShane, Bool.coe_sort_false, false_imp_iff, true_imp_iff, diff_empty] using H
    /-
      🎉 no goals
    -/


