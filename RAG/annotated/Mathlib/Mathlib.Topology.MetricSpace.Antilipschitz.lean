/-- We say that `f : α → β` is `AntilipschitzWith K` if for any two points `x`, `y` we have
`edist x y ≤ K * edist (f x) (f y)`. -/
def AntilipschitzWith [PseudoEMetricSpace α] [PseudoEMetricSpace β] (K : ℝ≥0) (f : α → β) :=
  ∀ x y, edist x y ≤ K * edist (f x) (f y)


protected lemma AntilipschitzWith.edist_lt_top [PseudoEMetricSpace α] [PseudoMetricSpace β]
    {K : ℝ≥0} {f : α → β} (h : AntilipschitzWith K f) (x y : α) : edist x y < ⊤ :=
  (h x y).trans_lt <| ENNReal.mul_lt_top ENNReal.coe_lt_top (edist_lt_top _ _)


theorem AntilipschitzWith.edist_ne_top [PseudoEMetricSpace α] [PseudoMetricSpace β] {K : ℝ≥0}
    {f : α → β} (h : AntilipschitzWith K f) (x y : α) : edist x y ≠ ⊤ :=
  (h.edist_lt_top x y).ne


theorem antilipschitzWith_iff_le_mul_nndist :
    AntilipschitzWith K f ↔ ∀ x y, nndist x y ≤ K * nndist (f x) (f y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    ⊢ Iff (AntilipschitzWith K f) (∀ (x y : α), LE.le (NNDist.nndist x y) (HMul.hM …
  -/
  simp only [AntilipschitzWith, edist_nndist]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    ⊢ Iff (∀ (x y : α), LE.le (↑(NNDist.nndist x y)) (HMul.hMul ↑K ↑(NNDist.nndist …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


alias ⟨AntilipschitzWith.le_mul_nndist, AntilipschitzWith.of_le_mul_nndist⟩ :=
  antilipschitzWith_iff_le_mul_nndist


theorem antilipschitzWith_iff_le_mul_dist :
    AntilipschitzWith K f ↔ ∀ x y, dist x y ≤ K * dist (f x) (f y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    ⊢ Iff (AntilipschitzWith K f) (∀ (x y : α), LE.le (Dist.dist x y) (HMul.hMul ( …
  -/
  simp only [antilipschitzWith_iff_le_mul_nndist, dist_nndist]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    ⊢ Iff (∀ (x y : α), LE.le (NNDist.nndist x y) (HMul.hMul K (NNDist.nndist (f x …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


alias ⟨AntilipschitzWith.le_mul_dist, AntilipschitzWith.of_le_mul_dist⟩ :=
  antilipschitzWith_iff_le_mul_dist


theorem mul_le_nndist (hf : AntilipschitzWith K f) (x y : α) :
    K⁻¹ * nndist x y ≤ nndist (f x) (f y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    x y : α
    ⊢ LE.le (HMul.hMul (Inv.inv K) (NNDist.nndist x y)) (NNDist.nndist (f x) (f y))
  -/
  simpa only [div_eq_inv_mul] using NNReal.div_le_of_le_mul' (hf.le_mul_nndist x y)
  /-
    🎉 no goals
  -/


theorem mul_le_dist (hf : AntilipschitzWith K f) (x y : α) :
    (K⁻¹ * dist x y : ℝ) ≤ dist (f x) (f y) := mod_cast hf.mul_le_nndist x y


/-- Extract the constant from `hf : AntilipschitzWith K f`. This is useful, e.g.,
if `K` is given by a long formula, and we want to reuse this value. -/
@[nolint unusedArguments]
protected def k (_hf : AntilipschitzWith K f) : ℝ≥0 := K


protected theorem injective {α : Type*} {β : Type*} [EMetricSpace α] [PseudoEMetricSpace β]
    {K : ℝ≥0} {f : α → β} (hf : AntilipschitzWith K f) : Function.Injective f := fun x y h => by
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : EMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    x y : α
    h : Eq (f x) (f y)
    ⊢ Eq x y
  -/
  simpa only [h, edist_self, mul_zero, edist_le_zero] using hf x y
  /-
    🎉 no goals
  -/


theorem mul_le_edist (hf : AntilipschitzWith K f) (x y : α) :
    (K : ℝ≥0∞)⁻¹ * edist x y ≤ edist (f x) (f y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    x y : α
    ⊢ LE.le (HMul.hMul (Inv.inv ↑K) (EDist.edist x y)) (EDist.edist (f x) (f y))
  -/
  rw [mul_comm, ← div_eq_mul_inv]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    x y : α
    ⊢ LE.le (HDiv.hDiv (EDist.edist x y) ↑K) (EDist.edist (f x) (f y))
  -/
  exact ENNReal.div_le_of_le_mul' (hf x y)
  /-
    🎉 no goals
  -/


theorem ediam_preimage_le (hf : AntilipschitzWith K f) (s : Set β) : diam (f ⁻¹' s) ≤ K * diam s :=
  diam_le fun x hx y hy => (hf x y).trans <|
    mul_le_mul_left' (edist_le_diam_of_mem (mem_preimage.1 hx) hy) K


theorem le_mul_ediam_image (hf : AntilipschitzWith K f) (s : Set α) : diam s ≤ K * diam (f '' s) :=
  (diam_mono (subset_preimage_image _ _)).trans (hf.ediam_preimage_le (f '' s))


protected theorem id : AntilipschitzWith 1 (id : α → α) := fun x y => by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x y : α
    ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑1) (EDist.edist (id x) (id y)))
  -/
  simp only [ENNReal.coe_one, one_mul, id, le_refl]
  /-
    🎉 no goals
  -/


theorem comp {Kg : ℝ≥0} {g : β → γ} (hg : AntilipschitzWith Kg g) {Kf : ℝ≥0} {f : α → β}
    (hf : AntilipschitzWith Kf f) : AntilipschitzWith (Kf * Kg) (g ∘ f) := fun x y =>
  calc
    edist x y ≤ Kf * edist (f x) (f y) := hf x y
    _ ≤ Kf * (Kg * edist (g (f x)) (g (f y))) := mul_left_mono (hg _ _)
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  inst✝² : PseudoEMetricSpace α
                  inst✝¹ : PseudoEMetricSpace β
                  inst✝ : PseudoEMetricSpace γ
                  Kg : NNReal
                  g : β → γ
                  hg : AntilipschitzWith Kg g
                  Kf : NNReal
                  f : α → β
                  hf : AntilipschitzWith Kf f
                  x y : α
                  ⊢ Eq (HMul.hMul (↑Kf) (HMul.hMul (↑Kg) (EDist.edist (g (f x)) (g (f y))))) (HM …
                -/
    _ = _ := by rw [ENNReal.coe_mul, mul_assoc]; rfl
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem restrict (hf : AntilipschitzWith K f) (s : Set α) : AntilipschitzWith K (s.restrict f) :=
  fun x y => hf x y


theorem codRestrict (hf : AntilipschitzWith K f) {s : Set β} (hs : ∀ x, f x ∈ s) :
    AntilipschitzWith K (s.codRestrict f hs) := fun x y => hf x y


theorem to_rightInvOn' {s : Set α} (hf : AntilipschitzWith K (s.restrict f)) {g : β → α}
    {t : Set β} (g_maps : MapsTo g t s) (g_inv : RightInvOn g f t) :
    LipschitzWith K (t.restrict g) := fun x y => by
  simpa only [restrict_apply, g_inv x.mem, g_inv y.mem, Subtype.edist_mk_mk]
    using hf ⟨g x, g_maps x.mem⟩ ⟨g y, g_maps y.mem⟩


theorem to_rightInvOn (hf : AntilipschitzWith K f) {g : β → α} {t : Set β} (h : RightInvOn g f t) :
    LipschitzWith K (t.restrict g) :=
  (hf.restrict univ).to_rightInvOn' (mapsTo_univ g t) h


theorem to_rightInverse (hf : AntilipschitzWith K f) {g : β → α} (hg : Function.RightInverse g f) :
    LipschitzWith K g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    g : β → α
    hg : Function.RightInverse g f
    ⊢ LipschitzWith K g
  -/
  intro x y
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    g : β → α
    hg : Function.RightInverse g f
    x y : β
    ⊢ LE.le (EDist.edist (g x) (g y)) (HMul.hMul (↑K) (EDist.edist x y))
  -/
  have := hf (g x) (g y)
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    g : β → α
    hg : Function.RightInverse g f
    x y : β
    this : LE.le (EDist.edist (g x) (g y)) (HMul.hMul (↑K) (EDist.edist (f (g x))  …
    ⊢ LE.le (EDist.edist (g x) (g y)) (HMul.hMul (↑K) (EDist.edist x y))
  -/
  rwa [hg x, hg y] at this
  /-
    🎉 no goals
  -/


theorem comap_uniformity_le (hf : AntilipschitzWith K f) : (𝓤 β).comap (Prod.map f f) ≤ 𝓤 α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    ⊢ LE.le (Filter.comap (Prod.map f f) (uniformity β)) (uniformity α)
  -/
  refine ((uniformity_basis_edist.comap _).le_basis_iff uniformity_basis_edist).2 fun ε h₀ => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    ε : ENNReal
    h₀ : LT.lt 0 ε
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Set.preimage (Prod.map f  …
  -/
  refine ⟨(↑K)⁻¹ * ε, ENNReal.mul_pos (ENNReal.inv_ne_zero.2 ENNReal.coe_ne_top) h₀.ne', ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    ε : ENNReal
    h₀ : LT.lt 0 ε
    ⊢ HasSubset.Subset (Set.preimage (Prod.map f f) (setOf fun p => LT.lt (EDist.e …
  -/
  refine fun x hx => (hf x.1 x.2).trans_lt ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    ε : ENNReal
    h₀ : LT.lt 0 ε
    x : Prod α α
    hx : Membership.mem (Set.preimage (Prod.map f f) (setOf fun p => LT.lt (EDist. …
    ⊢ LT.lt (HMul.hMul (↑K) (EDist.edist (f x.1) (f x.2))) ε
  -/
  rw [mul_comm, ← div_eq_mul_inv] at hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    ε : ENNReal
    h₀ : LT.lt 0 ε
    x : Prod α α
    hx : Membership.mem (Set.preimage (Prod.map f f) (setOf fun p => LT.lt (EDist. …
    ⊢ LT.lt (HMul.hMul (↑K) (EDist.edist (f x.1) (f x.2))) ε
  -/
  rw [mul_comm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    K : NNReal
    f : α → β
    hf : AntilipschitzWith K f
    ε : ENNReal
    h₀ : LT.lt 0 ε
    x : Prod α α
    hx : Membership.mem (Set.preimage (Prod.map f f) (setOf fun p => LT.lt (EDist. …
    ⊢ LT.lt (HMul.hMul (EDist.edist (f x.1) (f x.2)) ↑K) ε
  -/
  exact ENNReal.mul_lt_of_lt_div hx
  /-
    🎉 no goals
  -/


theorem isUniformInducing (hf : AntilipschitzWith K f) (hfc : UniformContinuous f) :
    IsUniformInducing f :=
  ⟨le_antisymm hf.comap_uniformity_le hfc.le_comap⟩


@[deprecated (since := "2024-10-05")]
alias uniformInducing := isUniformInducing


lemma isUniformEmbedding {α β : Type*} [EMetricSpace α] [PseudoEMetricSpace β] {K : ℝ≥0} {f : α → β}
    (hf : AntilipschitzWith K f) (hfc : UniformContinuous f) : IsUniformEmbedding f :=
  ⟨hf.isUniformInducing hfc, hf.injective⟩


@[deprecated (since := "2024-10-01")] alias uniformEmbedding := isUniformEmbedding


theorem isComplete_range [CompleteSpace α] (hf : AntilipschitzWith K f)
    (hfc : UniformContinuous f) : IsComplete (range f) :=
  (hf.isUniformInducing hfc).isComplete_range


theorem isClosed_range {α β : Type*} [PseudoEMetricSpace α] [EMetricSpace β] [CompleteSpace α]
    {f : α → β} {K : ℝ≥0} (hf : AntilipschitzWith K f) (hfc : UniformContinuous f) :
    IsClosed (range f) :=
  (hf.isComplete_range hfc).isClosed


theorem isClosedEmbedding {α : Type*} {β : Type*} [EMetricSpace α] [EMetricSpace β] {K : ℝ≥0}
    {f : α → β} [CompleteSpace α] (hf : AntilipschitzWith K f) (hfc : UniformContinuous f) :
    IsClosedEmbedding f :=
  { (hf.isUniformEmbedding hfc).isEmbedding with isClosed_range := hf.isClosed_range hfc }


@[deprecated (since := "2024-10-20")]
alias closedEmbedding := isClosedEmbedding


theorem subtype_coe (s : Set α) : AntilipschitzWith 1 ((↑) : s → α) :=
  AntilipschitzWith.id.restrict s


@[nontriviality] -- Porting note: added `nontriviality`
theorem of_subsingleton [Subsingleton α] {K : ℝ≥0} : AntilipschitzWith K f := fun x y => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : PseudoEMetricSpace β
    f : α → β
    inst✝ : Subsingleton α
    K : NNReal
    x y : α
    ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑K) (EDist.edist (f x) (f y)))
  -/
  simp only [Subsingleton.elim x y, edist_self, zero_le]
  /-
    🎉 no goals
  -/


/-- If `f : α → β` is `0`-antilipschitz, then `α` is a `subsingleton`. -/
protected theorem subsingleton {α β} [EMetricSpace α] [PseudoEMetricSpace β] {f : α → β}
    (h : AntilipschitzWith 0 f) : Subsingleton α :=
  ⟨fun x y => edist_le_zero.1 <| (h x y).trans_eq <| zero_mul _⟩


theorem isBounded_preimage (hf : AntilipschitzWith K f) {s : Set β} (hs : IsBounded s) :
    IsBounded (f ⁻¹' s) :=
  isBounded_iff_ediam_ne_top.2 <| ne_top_of_le_ne_top
    (ENNReal.mul_ne_top ENNReal.coe_ne_top hs.ediam_ne_top) (hf.ediam_preimage_le _)


theorem tendsto_cobounded (hf : AntilipschitzWith K f) : Tendsto f (cobounded α) (cobounded β) :=
  compl_surjective.forall.2 fun _ ↦ hf.isBounded_preimage


/-- The image of a proper space under an expanding onto map is proper. -/
protected theorem properSpace {α : Type*} [MetricSpace α] {K : ℝ≥0} {f : α → β} [ProperSpace α]
    (hK : AntilipschitzWith K f) (f_cont : Continuous f) (hf : Function.Surjective f) :
    ProperSpace β := by
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K f
    f_cont : Continuous f
    hf : Function.Surjective f
    ⊢ ProperSpace β
  -/
  refine ⟨fun x₀ r => ?_⟩
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K f
    f_cont : Continuous f
    hf : Function.Surjective f
    x₀ : β
    r : Real
    ⊢ IsCompact (Metric.closedBall x₀ r)
  -/
  let K := f ⁻¹' closedBall x₀ r
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K✝ : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K✝ f
    f_cont : Continuous f
    hf : Function.Surjective f
    x₀ : β
    r : Real
    K : Set α := Set.preimage f (Metric.closedBall x₀ r)
    ⊢ IsCompact (Metric.closedBall x₀ r)
  -/
  have A : IsClosed K := isClosed_ball.preimage f_cont
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K✝ : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K✝ f
    f_cont : Continuous f
    hf : Function.Surjective f
    x₀ : β
    r : Real
    K : Set α := Set.preimage f (Metric.closedBall x₀ r)
    A : IsClosed K
    ⊢ IsCompact (Metric.closedBall x₀ r)
  -/
  have B : IsBounded K := hK.isBounded_preimage isBounded_closedBall
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K✝ : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K✝ f
    f_cont : Continuous f
    hf : Function.Surjective f
    x₀ : β
    r : Real
    K : Set α := Set.preimage f (Metric.closedBall x₀ r)
    A : IsClosed K
    B : Bornology.IsBounded K
    ⊢ IsCompact (Metric.closedBall x₀ r)
  -/
  have : IsCompact K := isCompact_iff_isClosed_bounded.2 ⟨A, B⟩
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K✝ : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K✝ f
    f_cont : Continuous f
    hf : Function.Surjective f
    x₀ : β
    r : Real
    K : Set α := Set.preimage f (Metric.closedBall x₀ r)
    A : IsClosed K
    B : Bornology.IsBounded K
    this : IsCompact K
    ⊢ IsCompact (Metric.closedBall x₀ r)
  -/
  convert this.image f_cont
  /-
    case h.e'_3
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    α : Type u_4
    inst✝¹ : MetricSpace α
    K✝ : NNReal
    f : α → β
    inst✝ : ProperSpace α
    hK : AntilipschitzWith K✝ f
    f_cont : Continuous f
    hf : Function.Surjective f
    x₀ : β
    r : Real
    K : Set α := Set.preimage f (Metric.closedBall x₀ r)
    A : IsClosed K
    B : Bornology.IsBounded K
    this : IsCompact K
    ⊢ Eq (Metric.closedBall x₀ r) (Set.image f K)
  -/
  exact (hf.image_preimage _).symm
  /-
    🎉 no goals
  -/


theorem isBounded_of_image2_left (f : α → β → γ) {K₁ : ℝ≥0}
    (hf : ∀ b, AntilipschitzWith K₁ fun a => f a b) {s : Set α} {t : Set β}
    (hst : IsBounded (Set.image2 f s t)) : IsBounded s ∨ IsBounded t := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : PseudoMetricSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : PseudoMetricSpace γ
    f : α → β → γ
    K₁ : NNReal
    hf : ∀ (b : β), AntilipschitzWith K₁ fun a => f a b
    s : Set α
    t : Set β
    hst : Bornology.IsBounded (Set.image2 f s t)
    ⊢ Or (Bornology.IsBounded s) (Bornology.IsBounded t)
  -/
  contrapose! hst
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : PseudoMetricSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : PseudoMetricSpace γ
    f : α → β → γ
    K₁ : NNReal
    hf : ∀ (b : β), AntilipschitzWith K₁ fun a => f a b
    s : Set α
    t : Set β
    hst : And (Not (Bornology.IsBounded s)) (Not (Bornology.IsBounded t))
    ⊢ Not (Bornology.IsBounded (Set.image2 f s t))
  -/
  obtain ⟨b, hb⟩ : t.Nonempty := nonempty_of_not_isBounded hst.2
  have : ¬IsBounded (Set.image2 f s {b}) := by
    intro h
    apply hst.1
    rw [Set.image2_singleton_right] at h
    replace h := (hf b).isBounded_preimage h
    exact h.subset (subset_preimage_image _ _)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : PseudoMetricSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : PseudoMetricSpace γ
    f : α → β → γ
    K₁ : NNReal
    hf : ∀ (b : β), AntilipschitzWith K₁ fun a => f a b
    s : Set α
    t : Set β
    hst : And (Not (Bornology.IsBounded s)) (Not (Bornology.IsBounded t))
    b : β
    hb : Membership.mem t b
    this : Not (Bornology.IsBounded (Set.image2 f s (Singleton.singleton b)))
    ⊢ Not (Bornology.IsBounded (Set.image2 f s t))
  -/
  exact mt (IsBounded.subset · (image2_subset subset_rfl (singleton_subset_iff.mpr hb))) this
  /-
    🎉 no goals
  -/


theorem isBounded_of_image2_right {f : α → β → γ} {K₂ : ℝ≥0} (hf : ∀ a, AntilipschitzWith K₂ (f a))
    {s : Set α} {t : Set β} (hst : IsBounded (Set.image2 f s t)) : IsBounded s ∨ IsBounded t :=
  Or.symm <| isBounded_of_image2_left (flip f) hf <| image2_swap f s t ▸ hst


theorem LipschitzWith.to_rightInverse [PseudoEMetricSpace α] [PseudoEMetricSpace β] {K : ℝ≥0}
    {f : α → β} (hf : LipschitzWith K f) {g : β → α} (hg : Function.RightInverse g f) :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             inst✝¹ : PseudoEMetricSpace α
                                             inst✝ : PseudoEMetricSpace β
                                             K : NNReal
                                             f : α → β
                                             hf : LipschitzWith K f
                                             g : β → α
                                             hg : Function.RightInverse g f
                                             x y : β
                                             ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑K) (EDist.edist (g x) (g y)))
                                           -/
    AntilipschitzWith K g := fun x y => by simpa only [hg _] using hf (g x) (g y)
                                           /-
                                             🎉 no goals
                                           -/


/-- The preimage of a proper space under a Lipschitz homeomorphism is proper. -/
protected theorem LipschitzWith.properSpace [PseudoMetricSpace α] [MetricSpace β] [ProperSpace β]
    {K : ℝ≥0} {f : α ≃ₜ β} (hK : LipschitzWith K f) : ProperSpace α :=
  (hK.to_rightInverse f.right_inv).properSpace f.symm.continuous f.symm.surjective

