/-- An isometry (also known as isometric embedding) is a map preserving the edistance
between pseudoemetric spaces, or equivalently the distance between pseudometric space. -/
def Isometry [PseudoEMetricSpace α] [PseudoEMetricSpace β] (f : α → β) : Prop :=
  ∀ x1 x2 : α, edist (f x1) (f x2) = edist x1 x2


/-- On pseudometric spaces, a map is an isometry if and only if it preserves nonnegative
distances. -/
theorem isometry_iff_nndist_eq [PseudoMetricSpace α] [PseudoMetricSpace β] {f : α → β} :
    Isometry f ↔ ∀ x y, nndist (f x) (f y) = nndist x y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    ⊢ Iff (Isometry f) (∀ (x y : α), Eq (NNDist.nndist (f x) (f y)) (NNDist.nndist …
  -/
  simp only [Isometry, edist_nndist, ENNReal.coe_inj]
  /-
    🎉 no goals
  -/


/-- On pseudometric spaces, a map is an isometry if and only if it preserves distances. -/
theorem isometry_iff_dist_eq [PseudoMetricSpace α] [PseudoMetricSpace β] {f : α → β} :
    Isometry f ↔ ∀ x y, dist (f x) (f y) = dist x y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    ⊢ Iff (Isometry f) (∀ (x y : α), Eq (Dist.dist (f x) (f y)) (Dist.dist x y))
  -/
  simp only [isometry_iff_nndist_eq, ← coe_nndist, NNReal.coe_inj]
  /-
    🎉 no goals
  -/


/-- An isometry preserves distances. -/
alias ⟨Isometry.dist_eq, _⟩ := isometry_iff_dist_eq


/-- A map that preserves distances is an isometry -/
alias ⟨_, Isometry.of_dist_eq⟩ := isometry_iff_dist_eq


/-- An isometry preserves non-negative distances. -/
alias ⟨Isometry.nndist_eq, _⟩ := isometry_iff_nndist_eq


/-- A map that preserves non-negative distances is an isometry. -/
alias ⟨_, Isometry.of_nndist_eq⟩ := isometry_iff_nndist_eq


/-- An isometry preserves edistances. -/
theorem edist_eq (hf : Isometry f) (x y : α) : edist (f x) (f y) = edist x y :=
  hf x y


theorem lipschitz (h : Isometry f) : LipschitzWith 1 f :=
  LipschitzWith.of_edist_le fun x y => (h x y).le


theorem antilipschitz (h : Isometry f) : AntilipschitzWith 1 f := fun x y => by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    h : Isometry f
    x y : α
    ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑1) (EDist.edist (f x) (f y)))
  -/
  simp only [h x y, ENNReal.coe_one, one_mul, le_refl]
  /-
    🎉 no goals
  -/


/-- Any map on a subsingleton is an isometry -/
@[nontriviality]
theorem _root_.isometry_subsingleton [Subsingleton α] : Isometry f := fun x y => by
  /-
    α : Type u
    β : Type v
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : PseudoEMetricSpace β
    f : α → β
    inst✝ : Subsingleton α
    x y : α
    ⊢ Eq (EDist.edist (f x) (f y)) (EDist.edist x y)
  -/
  rw [Subsingleton.elim x y]; simp
                              /-
                                🎉 no goals
                              -/


/-- The identity is an isometry -/
theorem _root_.isometry_id : Isometry (id : α → α) := fun _ _ => rfl


theorem prod_map {δ} [PseudoEMetricSpace δ] {f : α → β} {g : γ → δ} (hf : Isometry f)
    (hg : Isometry g) : Isometry (Prod.map f g) := fun x y => by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : PseudoEMetricSpace γ
    δ : Type u_2
    inst✝ : PseudoEMetricSpace δ
    f : α → β
    g : γ → δ
    hf : Isometry f
    hg : Isometry g
    x y : Prod α γ
    ⊢ Eq (EDist.edist (Prod.map f g x) (Prod.map f g y)) (EDist.edist x y)
  -/
  simp only [Prod.edist_eq, Prod.map_fst, hf.edist_eq, Prod.map_snd, hg.edist_eq]
  /-
    🎉 no goals
  -/


protected theorem piMap {ι} [Fintype ι] {α β : ι → Type*} [∀ i, PseudoEMetricSpace (α i)]
    [∀ i, PseudoEMetricSpace (β i)] (f : ∀ i, α i → β i) (hf : ∀ i, Isometry (f i)) :
    Isometry (Pi.map f) := fun x y => by
  /-
    ι : Type u_4
    inst✝² : Fintype ι
    α : ι → Type u_2
    β : ι → Type u_3
    inst✝¹ : (i : ι) → PseudoEMetricSpace (α i)
    inst✝ : (i : ι) → PseudoEMetricSpace (β i)
    f : (i : ι) → α i → β i
    hf : ∀ (i : ι), Isometry (f i)
    x y : (i : ι) → α i
    ⊢ Eq (EDist.edist (Pi.map f x) (Pi.map f y)) (EDist.edist x y)
  -/
  simp only [edist_pi_def, (hf _).edist_eq, Pi.map_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-06")] alias _root_.isometry_dcomp := Isometry.piMap


/-- The composition of isometries is an isometry. -/
theorem comp {g : β → γ} {f : α → β} (hg : Isometry g) (hf : Isometry f) : Isometry (g ∘ f) :=
  fun _ _ => (hg _ _).trans (hf _ _)


/-- An isometry from a metric space is a uniform continuous map -/
protected theorem uniformContinuous (hf : Isometry f) : UniformContinuous f :=
  hf.lipschitz.uniformContinuous


/-- An isometry from a metric space is a uniform inducing map -/
theorem isUniformInducing (hf : Isometry f) : IsUniformInducing f :=
  hf.antilipschitz.isUniformInducing hf.uniformContinuous


@[deprecated (since := "2024-10-05")]
alias uniformInducing := isUniformInducing


theorem tendsto_nhds_iff {ι : Type*} {f : α → β} {g : ι → α} {a : Filter ι} {b : α}
    (hf : Isometry f) : Filter.Tendsto g a (𝓝 b) ↔ Filter.Tendsto (f ∘ g) a (𝓝 (f b)) :=
  hf.isUniformInducing.isInducing.tendsto_nhds_iff


/-- An isometry is continuous. -/
protected theorem continuous (hf : Isometry f) : Continuous f :=
  hf.lipschitz.continuous


/-- The right inverse of an isometry is an isometry. -/
theorem right_inv {f : α → β} {g : β → α} (h : Isometry f) (hg : RightInverse g f) : Isometry g :=
                /-
                  α : Type u
                  β : Type v
                  inst✝¹ : PseudoEMetricSpace α
                  inst✝ : PseudoEMetricSpace β
                  f : α → β
                  g : β → α
                  h : Isometry f
                  hg : Function.RightInverse g f
                  x y : β
                  ⊢ Eq (EDist.edist (g x) (g y)) (EDist.edist x y)
                -/
  fun x y => by rw [← h, hg _, hg _]
                /-
                  🎉 no goals
                -/


theorem preimage_emetric_closedBall (h : Isometry f) (x : α) (r : ℝ≥0∞) :
    f ⁻¹' EMetric.closedBall (f x) r = EMetric.closedBall x r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    h : Isometry f
    x : α
    r : ENNReal
    ⊢ Eq (Set.preimage f (EMetric.closedBall (f x) r)) (EMetric.closedBall x r)
  -/
  ext y
  /-
    case h
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    h : Isometry f
    x : α
    r : ENNReal
    y : α
    ⊢ Iff (Membership.mem (Set.preimage f (EMetric.closedBall (f x) r)) y) (Member …
  -/
  simp [h.edist_eq]
  /-
    🎉 no goals
  -/


theorem preimage_emetric_ball (h : Isometry f) (x : α) (r : ℝ≥0∞) :
    f ⁻¹' EMetric.ball (f x) r = EMetric.ball x r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    h : Isometry f
    x : α
    r : ENNReal
    ⊢ Eq (Set.preimage f (EMetric.ball (f x) r)) (EMetric.ball x r)
  -/
  ext y
  /-
    case h
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    h : Isometry f
    x : α
    r : ENNReal
    y : α
    ⊢ Iff (Membership.mem (Set.preimage f (EMetric.ball (f x) r)) y) (Membership.m …
  -/
  simp [h.edist_eq]
  /-
    🎉 no goals
  -/


/-- Isometries preserve the diameter in pseudoemetric spaces. -/
theorem ediam_image (hf : Isometry f) (s : Set α) : EMetric.diam (f '' s) = EMetric.diam s :=
                                  /-
                                    α : Type u
                                    β : Type v
                                    inst✝¹ : PseudoEMetricSpace α
                                    inst✝ : PseudoEMetricSpace β
                                    f : α → β
                                    hf : Isometry f
                                    s : Set α
                                    d : ENNReal
                                    ⊢ Iff (LE.le (EMetric.diam (Set.image f s)) d) (LE.le (EMetric.diam s) d)
                                  -/
  eq_of_forall_ge_iff fun d => by simp only [EMetric.diam_le_iff, forall_mem_image, hf.edist_eq]
                                  /-
                                    🎉 no goals
                                  -/


theorem ediam_range (hf : Isometry f) : EMetric.diam (range f) = EMetric.diam (univ : Set α) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    hf : Isometry f
    ⊢ Eq (EMetric.diam (Set.range f)) (EMetric.diam Set.univ)
  -/
  rw [← image_univ]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    hf : Isometry f
    ⊢ Eq (EMetric.diam (Set.image f Set.univ)) (EMetric.diam Set.univ)
  -/
  exact hf.ediam_image univ
  /-
    🎉 no goals
  -/


theorem mapsTo_emetric_ball (hf : Isometry f) (x : α) (r : ℝ≥0∞) :
    MapsTo f (EMetric.ball x r) (EMetric.ball (f x) r) :=
  (hf.preimage_emetric_ball x r).ge


theorem mapsTo_emetric_closedBall (hf : Isometry f) (x : α) (r : ℝ≥0∞) :
    MapsTo f (EMetric.closedBall x r) (EMetric.closedBall (f x) r) :=
  (hf.preimage_emetric_closedBall x r).ge


/-- The injection from a subtype is an isometry -/
theorem _root_.isometry_subtype_coe {s : Set α} : Isometry ((↑) : s → α) := fun _ _ => rfl


theorem comp_continuousOn_iff {γ} [TopologicalSpace γ] (hf : Isometry f) {g : γ → α} {s : Set γ} :
    ContinuousOn (f ∘ g) s ↔ ContinuousOn g s :=
  hf.isUniformInducing.isInducing.continuousOn_iff.symm


theorem comp_continuous_iff {γ} [TopologicalSpace γ] (hf : Isometry f) {g : γ → α} :
    Continuous (f ∘ g) ↔ Continuous g :=
  hf.isUniformInducing.isInducing.continuous_iff.symm


/-- An isometry from an emetric space is injective -/
protected theorem injective (h : Isometry f) : Injective f :=
  h.antilipschitz.injective


/-- An isometry from an emetric space is a uniform embedding -/
lemma isUniformEmbedding (hf : Isometry f) : IsUniformEmbedding f :=
  hf.antilipschitz.isUniformEmbedding hf.lipschitz.uniformContinuous


@[deprecated (since := "2024-10-01")] alias uniformEmbedding := isUniformEmbedding


/-- An isometry from an emetric space is an embedding -/
theorem isEmbedding (hf : Isometry f) : IsEmbedding f := hf.isUniformEmbedding.isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding := isEmbedding


/-- An isometry from a complete emetric space is a closed embedding -/
theorem isClosedEmbedding [CompleteSpace α] [EMetricSpace γ] {f : α → γ} (hf : Isometry f) :
    IsClosedEmbedding f :=
  hf.antilipschitz.isClosedEmbedding hf.lipschitz.uniformContinuous


@[deprecated (since := "2024-10-20")]
alias closedEmbedding := isClosedEmbedding


/-- An isometry preserves the diameter in pseudometric spaces. -/
theorem diam_image (hf : Isometry f) (s : Set α) : Metric.diam (f '' s) = Metric.diam s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    hf : Isometry f
    s : Set α
    ⊢ Eq (Metric.diam (Set.image f s)) (Metric.diam s)
  -/
  rw [Metric.diam, Metric.diam, hf.ediam_image]
  /-
    🎉 no goals
  -/


theorem diam_range (hf : Isometry f) : Metric.diam (range f) = Metric.diam (univ : Set α) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    hf : Isometry f
    ⊢ Eq (Metric.diam (Set.range f)) (Metric.diam Set.univ)
  -/
  rw [← image_univ]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    hf : Isometry f
    ⊢ Eq (Metric.diam (Set.image f Set.univ)) (Metric.diam Set.univ)
  -/
  exact hf.diam_image univ
  /-
    🎉 no goals
  -/


theorem preimage_setOf_dist (hf : Isometry f) (x : α) (p : ℝ → Prop) :
    f ⁻¹' { y | p (dist y (f x)) } = { y | p (dist y x) } := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    hf : Isometry f
    x : α
    p : Real → Prop
    ⊢ Eq (Set.preimage f (setOf fun y => p (Dist.dist y (f x)))) (setOf fun y => p …
  -/
  ext y
  /-
    case h
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    hf : Isometry f
    x : α
    p : Real → Prop
    y : α
    ⊢ Iff (Membership.mem (Set.preimage f (setOf fun y => p (Dist.dist y (f x))))  …
  -/
  simp [hf.dist_eq]
  /-
    🎉 no goals
  -/


theorem preimage_closedBall (hf : Isometry f) (x : α) (r : ℝ) :
    f ⁻¹' Metric.closedBall (f x) r = Metric.closedBall x r :=
  hf.preimage_setOf_dist x (· ≤ r)


theorem preimage_ball (hf : Isometry f) (x : α) (r : ℝ) :
    f ⁻¹' Metric.ball (f x) r = Metric.ball x r :=
  hf.preimage_setOf_dist x (· < r)


theorem preimage_sphere (hf : Isometry f) (x : α) (r : ℝ) :
    f ⁻¹' Metric.sphere (f x) r = Metric.sphere x r :=
  hf.preimage_setOf_dist x (· = r)


theorem mapsTo_ball (hf : Isometry f) (x : α) (r : ℝ) :
    MapsTo f (Metric.ball x r) (Metric.ball (f x) r) :=
  (hf.preimage_ball x r).ge


theorem mapsTo_sphere (hf : Isometry f) (x : α) (r : ℝ) :
    MapsTo f (Metric.sphere x r) (Metric.sphere (f x) r) :=
  (hf.preimage_sphere x r).ge


theorem mapsTo_closedBall (hf : Isometry f) (x : α) (r : ℝ) :
    MapsTo f (Metric.closedBall x r) (Metric.closedBall (f x) r) :=
  (hf.preimage_closedBall x r).ge


/-- A uniform embedding from a uniform space to a metric space is an isometry with respect to the
induced metric space structure on the source space. -/
theorem IsUniformEmbedding.to_isometry {α β} [UniformSpace α] [MetricSpace β] {f : α → β}
    (h : IsUniformEmbedding f) : (letI := h.comapMetricSpace f; Isometry f) :=
  let _ := h.comapMetricSpace f
  Isometry.of_dist_eq fun _ _ => rfl


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.to_isometry := IsUniformEmbedding.to_isometry


/-- An embedding from a topological space to a metric space is an isometry with respect to the
induced metric space structure on the source space. -/
theorem Topology.IsEmbedding.to_isometry {α β} [TopologicalSpace α] [MetricSpace β] {f : α → β}
    (h : IsEmbedding f) : (letI := h.comapMetricSpace f; Isometry f) :=
  let _ := h.comapMetricSpace f
  Isometry.of_dist_eq fun _ _ => rfl


@[deprecated (since := "2024-10-26")]
alias Embedding.to_isometry := IsEmbedding.to_isometry

-- such a bijection need not exist

/-- `α` and `β` are isometric if there is an isometric bijection between them. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure IsometryEquiv (α : Type u) (β : Type v) [PseudoEMetricSpace α] [PseudoEMetricSpace β]
    extends α ≃ β where
  isometry_toFun : Isometry toFun


@[inherit_doc]
infixl:25 " ≃ᵢ " => IsometryEquiv


theorem toEquiv_injective : Injective (toEquiv : (α ≃ᵢ β) → (α ≃ β))
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


@[simp] theorem toEquiv_inj {e₁ e₂ : α ≃ᵢ β} : e₁.toEquiv = e₂.toEquiv ↔ e₁ = e₂ :=
  toEquiv_injective.eq_iff


instance : EquivLike (α ≃ᵢ β) α β where
  coe e := e.toEquiv
  inv e := e.toEquiv.symm
  left_inv e := e.left_inv
  right_inv e := e.right_inv
  coe_injective' _ _ h _ := toEquiv_injective <| DFunLike.ext' h


theorem coe_eq_toEquiv (h : α ≃ᵢ β) (a : α) : h a = h.toEquiv a := rfl


@[simp] theorem coe_toEquiv (h : α ≃ᵢ β) : ⇑h.toEquiv = h := rfl


@[simp] theorem coe_mk (e : α ≃ β) (h) : ⇑(mk e h) = e := rfl


protected theorem isometry (h : α ≃ᵢ β) : Isometry h :=
  h.isometry_toFun


protected theorem bijective (h : α ≃ᵢ β) : Bijective h :=
  h.toEquiv.bijective


protected theorem injective (h : α ≃ᵢ β) : Injective h :=
  h.toEquiv.injective


protected theorem surjective (h : α ≃ᵢ β) : Surjective h :=
  h.toEquiv.surjective


protected theorem edist_eq (h : α ≃ᵢ β) (x y : α) : edist (h x) (h y) = edist x y :=
  h.isometry.edist_eq x y


protected theorem dist_eq {α β : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β] (h : α ≃ᵢ β)
    (x y : α) : dist (h x) (h y) = dist x y :=
  h.isometry.dist_eq x y


protected theorem nndist_eq {α β : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β] (h : α ≃ᵢ β)
    (x y : α) : nndist (h x) (h y) = nndist x y :=
  h.isometry.nndist_eq x y


protected theorem continuous (h : α ≃ᵢ β) : Continuous h :=
  h.isometry.continuous


@[simp]
theorem ediam_image (h : α ≃ᵢ β) (s : Set α) : EMetric.diam (h '' s) = EMetric.diam s :=
  h.isometry.ediam_image s


@[ext]
theorem ext ⦃h₁ h₂ : α ≃ᵢ β⦄ (H : ∀ x, h₁ x = h₂ x) : h₁ = h₂ :=
  DFunLike.ext _ _ H


/-- Alternative constructor for isometric bijections,
taking as input an isometry, and a right inverse. -/
def mk' {α : Type u} [EMetricSpace α] (f : α → β) (g : β → α) (hfg : ∀ x, f (g x) = x)
    (hf : Isometry f) : α ≃ᵢ β where
  toFun := f
  invFun := g
  left_inv _ := hf.injective <| hfg _
  right_inv := hfg
  isometry_toFun := hf


/-- The identity isometry of a space. -/
protected def refl (α : Type*) [PseudoEMetricSpace α] : α ≃ᵢ α :=
  { Equiv.refl α with isometry_toFun := isometry_id }


/-- The composition of two isometric isomorphisms, as an isometric isomorphism. -/
protected def trans (h₁ : α ≃ᵢ β) (h₂ : β ≃ᵢ γ) : α ≃ᵢ γ :=
  { Equiv.trans h₁.toEquiv h₂.toEquiv with
    isometry_toFun := h₂.isometry_toFun.comp h₁.isometry_toFun }


@[simp]
theorem trans_apply (h₁ : α ≃ᵢ β) (h₂ : β ≃ᵢ γ) (x : α) : h₁.trans h₂ x = h₂ (h₁ x) :=
  rfl


/-- The inverse of an isometric isomorphism, as an isometric isomorphism. -/
protected def symm (h : α ≃ᵢ β) : β ≃ᵢ α where
  isometry_toFun := h.isometry.right_inv h.right_inv
  toEquiv := h.toEquiv.symm


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (h : α ≃ᵢ β) : α → β := h


/-- See Note [custom simps projection] -/
def Simps.symm_apply (h : α ≃ᵢ β) : β → α :=
  h.symm


@[simp]
theorem symm_symm (h : α ≃ᵢ β) : h.symm.symm = h := rfl


theorem symm_bijective : Bijective (IsometryEquiv.symm : (α ≃ᵢ β) → β ≃ᵢ α) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem apply_symm_apply (h : α ≃ᵢ β) (y : β) : h (h.symm y) = y :=
  h.toEquiv.apply_symm_apply y


@[simp]
theorem symm_apply_apply (h : α ≃ᵢ β) (x : α) : h.symm (h x) = x :=
  h.toEquiv.symm_apply_apply x


theorem symm_apply_eq (h : α ≃ᵢ β) {x : α} {y : β} : h.symm y = x ↔ y = h x :=
  h.toEquiv.symm_apply_eq


theorem eq_symm_apply (h : α ≃ᵢ β) {x : α} {y : β} : x = h.symm y ↔ h x = y :=
  h.toEquiv.eq_symm_apply


theorem symm_comp_self (h : α ≃ᵢ β) : (h.symm : β → α) ∘ h = id := funext h.left_inv


theorem self_comp_symm (h : α ≃ᵢ β) : (h : α → β) ∘ h.symm = id := funext h.right_inv


@[simp]
theorem range_eq_univ (h : α ≃ᵢ β) : range h = univ :=
  h.toEquiv.range_eq_univ


theorem image_symm (h : α ≃ᵢ β) : image h.symm = preimage h :=
  image_eq_preimage_of_inverse h.symm.toEquiv.left_inv h.symm.toEquiv.right_inv


theorem preimage_symm (h : α ≃ᵢ β) : preimage h.symm = image h :=
  (image_eq_preimage_of_inverse h.toEquiv.left_inv h.toEquiv.right_inv).symm


@[simp]
theorem symm_trans_apply (h₁ : α ≃ᵢ β) (h₂ : β ≃ᵢ γ) (x : γ) :
    (h₁.trans h₂).symm x = h₁.symm (h₂.symm x) :=
  rfl


theorem ediam_univ (h : α ≃ᵢ β) : EMetric.diam (univ : Set α) = EMetric.diam (univ : Set β) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    h : IsometryEquiv α β
    ⊢ Eq (EMetric.diam Set.univ) (EMetric.diam Set.univ)
  -/
  rw [← h.range_eq_univ, h.isometry.ediam_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem ediam_preimage (h : α ≃ᵢ β) (s : Set β) : EMetric.diam (h ⁻¹' s) = EMetric.diam s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    h : IsometryEquiv α β
    s : Set β
    ⊢ Eq (EMetric.diam (Set.preimage (⇑h) s)) (EMetric.diam s)
  -/
  rw [← image_symm, ediam_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_emetric_ball (h : α ≃ᵢ β) (x : β) (r : ℝ≥0∞) :
    h ⁻¹' EMetric.ball x r = EMetric.ball (h.symm x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    h : IsometryEquiv α β
    x : β
    r : ENNReal
    ⊢ Eq (Set.preimage (⇑h) (EMetric.ball x r)) (EMetric.ball (h.symm x) r)
  -/
  rw [← h.isometry.preimage_emetric_ball (h.symm x) r, h.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_emetric_closedBall (h : α ≃ᵢ β) (x : β) (r : ℝ≥0∞) :
    h ⁻¹' EMetric.closedBall x r = EMetric.closedBall (h.symm x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    h : IsometryEquiv α β
    x : β
    r : ENNReal
    ⊢ Eq (Set.preimage (⇑h) (EMetric.closedBall x r)) (EMetric.closedBall (h.symm  …
  -/
  rw [← h.isometry.preimage_emetric_closedBall (h.symm x) r, h.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_emetric_ball (h : α ≃ᵢ β) (x : α) (r : ℝ≥0∞) :
    h '' EMetric.ball x r = EMetric.ball (h x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    h : IsometryEquiv α β
    x : α
    r : ENNReal
    ⊢ Eq (Set.image (⇑h) (EMetric.ball x r)) (EMetric.ball (h x) r)
  -/
  rw [← h.preimage_symm, h.symm.preimage_emetric_ball, symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_emetric_closedBall (h : α ≃ᵢ β) (x : α) (r : ℝ≥0∞) :
    h '' EMetric.closedBall x r = EMetric.closedBall (h x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    h : IsometryEquiv α β
    x : α
    r : ENNReal
    ⊢ Eq (Set.image (⇑h) (EMetric.closedBall x r)) (EMetric.closedBall (h x) r)
  -/
  rw [← h.preimage_symm, h.symm.preimage_emetric_closedBall, symm_symm]
  /-
    🎉 no goals
  -/


/-- The (bundled) homeomorphism associated to an isometric isomorphism. -/
@[simps toEquiv]
protected def toHomeomorph (h : α ≃ᵢ β) : α ≃ₜ β where
  continuous_toFun := h.continuous
  continuous_invFun := h.symm.continuous
  toEquiv := h.toEquiv


@[simp]
theorem coe_toHomeomorph (h : α ≃ᵢ β) : ⇑h.toHomeomorph = h :=
  rfl


@[simp]
theorem coe_toHomeomorph_symm (h : α ≃ᵢ β) : ⇑h.toHomeomorph.symm = h.symm :=
  rfl


@[simp]
theorem comp_continuousOn_iff {γ} [TopologicalSpace γ] (h : α ≃ᵢ β) {f : γ → α} {s : Set γ} :
    ContinuousOn (h ∘ f) s ↔ ContinuousOn f s :=
  h.toHomeomorph.comp_continuousOn_iff _ _


@[simp]
theorem comp_continuous_iff {γ} [TopologicalSpace γ] (h : α ≃ᵢ β) {f : γ → α} :
    Continuous (h ∘ f) ↔ Continuous f :=
  h.toHomeomorph.comp_continuous_iff


@[simp]
theorem comp_continuous_iff' {γ} [TopologicalSpace γ] (h : α ≃ᵢ β) {f : β → γ} :
    Continuous (f ∘ h) ↔ Continuous f :=
  h.toHomeomorph.comp_continuous_iff'


/-- The group of isometries. -/
instance : Group (α ≃ᵢ α) where
  one := IsometryEquiv.refl _
  mul e₁ e₂ := e₂.trans e₁
  inv := IsometryEquiv.symm
  mul_assoc _ _ _ := rfl
  one_mul _ := ext fun _ => rfl
  mul_one _ := ext fun _ => rfl
  inv_mul_cancel e := ext e.symm_apply_apply


@[simp] theorem coe_one : ⇑(1 : α ≃ᵢ α) = id := rfl


@[simp] theorem coe_mul (e₁ e₂ : α ≃ᵢ α) : ⇑(e₁ * e₂) = e₁ ∘ e₂ := rfl


theorem mul_apply (e₁ e₂ : α ≃ᵢ α) (x : α) : (e₁ * e₂) x = e₁ (e₂ x) := rfl


@[simp] theorem inv_apply_self (e : α ≃ᵢ α) (x : α) : e⁻¹ (e x) = x := e.symm_apply_apply x


@[simp] theorem apply_inv_self (e : α ≃ᵢ α) (x : α) : e (e⁻¹ x) = x := e.apply_symm_apply x


theorem completeSpace_iff (e : α ≃ᵢ β) : CompleteSpace α ↔ CompleteSpace β := by
  simp only [completeSpace_iff_isComplete_univ, ← e.range_eq_univ, ← image_univ,
    isComplete_image_iff e.isometry.isUniformInducing]


protected theorem completeSpace [CompleteSpace β] (e : α ≃ᵢ β) : CompleteSpace α :=
  e.completeSpace_iff.2 ‹_›


/-- The natural isometry `∀ i, Y i ≃ᵢ ∀ j, Y (e.symm j)` obtained from a bijection `ι ≃ ι'` of
fintypes. `Equiv.piCongrLeft'` as an `IsometryEquiv`.-/
@[simps!]
def piCongrLeft' {ι' : Type*} [Fintype ι] [Fintype ι'] {Y : ι → Type*}
    [∀ j, PseudoEMetricSpace (Y j)] (e : ι ≃ ι') : (∀ i, Y i) ≃ᵢ ∀ j, Y (e.symm j) where
  toEquiv := Equiv.piCongrLeft' _ e
  isometry_toFun x1 x2 := by
    /-
      ι : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : PseudoEMetricSpace α
      inst✝⁴ : PseudoEMetricSpace β
      inst✝³ : PseudoEMetricSpace γ
      ι' : Type u_2
      inst✝² : Fintype ι
      inst✝¹ : Fintype ι'
      Y : ι → Type u_3
      inst✝ : (j : ι) → PseudoEMetricSpace (Y j)
      e : Equiv ι ι'
      x1 x2 : (i : ι) → Y i
      ⊢ Eq (EDist.edist ((Equiv.piCongrLeft' Y e).toFun x1) ((Equiv.piCongrLeft' Y e …
    -/
    simp_rw [edist_pi_def, Finset.sup_univ_eq_iSup]
    /-
      ι : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : PseudoEMetricSpace α
      inst✝⁴ : PseudoEMetricSpace β
      inst✝³ : PseudoEMetricSpace γ
      ι' : Type u_2
      inst✝² : Fintype ι
      inst✝¹ : Fintype ι'
      Y : ι → Type u_3
      inst✝ : (j : ι) → PseudoEMetricSpace (Y j)
      e : Equiv ι ι'
      x1 x2 : (i : ι) → Y i
      ⊢ Eq (iSup fun b => EDist.edist ((Equiv.piCongrLeft' Y e).toFun x1 b) ((Equiv. …
    -/
    exact (Equiv.iSup_comp (g := fun b ↦ edist (x1 b) (x2 b)) e.symm)
    /-
      🎉 no goals
    -/


/-- The natural isometry `∀ i, Y (e i) ≃ᵢ ∀ j, Y j` obtained from a bijection `ι ≃ ι'` of fintypes.
`Equiv.piCongrLeft` as an `IsometryEquiv`. -/
@[simps!]
def piCongrLeft {ι' : Type*} [Fintype ι] [Fintype ι'] {Y : ι' → Type*}
    [∀ j, PseudoEMetricSpace (Y j)] (e : ι ≃ ι') : (∀ i, Y (e i)) ≃ᵢ ∀ j, Y j :=
  (piCongrLeft' e.symm).symm


/-- The natural isometry `(α ⊕ β → γ) ≃ᵢ (α → γ) × (β → γ)` between the type of maps on a sum of
fintypes `α ⊕ β` and the pairs of functions on the types `α` and `β`.
`Equiv.sumArrowEquivProdArrow` as an `IsometryEquiv`.-/
@[simps!]
def sumArrowIsometryEquivProdArrow [Fintype α] [Fintype β] : (α ⊕ β → γ) ≃ᵢ (α → γ) × (β → γ) where
  toEquiv := Equiv.sumArrowEquivProdArrow _ _ _
                           /-
                             ι : Type u_1
                             α : Type u
                             β : Type v
                             γ : Type w
                             inst✝⁴ : PseudoEMetricSpace α
                             inst✝³ : PseudoEMetricSpace β
                             inst✝² : PseudoEMetricSpace γ
                             inst✝¹ : Fintype α
                             inst✝ : Fintype β
                             x✝¹ x✝ : Sum α β → γ
                             ⊢ Eq (EDist.edist ((Equiv.sumArrowEquivProdArrow α β γ).toFun x✝¹) ((Equiv.sum …
                           -/
  isometry_toFun _ _ := by simp [Prod.edist_eq, edist_pi_def, Finset.sup_univ_eq_iSup, iSup_sum]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem sumArrowIsometryEquivProdArrow_toHomeomorph {α β : Type*} [Fintype α] [Fintype β] :
    sumArrowIsometryEquivProdArrow.toHomeomorph
    = Homeomorph.sumArrowHomeomorphProdArrow (ι := α) (ι' := β) (X := γ) :=
  rfl


theorem _root_.Fin.edist_append_eq_max_edist (m n : ℕ) {x x2 : Fin m → α} {y y2 : Fin n → α} :
    edist (Fin.append x y) (Fin.append x2 y2) = max (edist x x2) (edist y y2) := by
  simp [edist_pi_def, Finset.sup_univ_eq_iSup, ← Equiv.iSup_comp (e := finSumFinEquiv),
    Prod.edist_eq, iSup_sum]


/-- The natural `IsometryEquiv` between `(Fin m → α) × (Fin n → α)` and `Fin (m + n) → α`.
`Fin.appendEquiv` as an `IsometryEquiv`.-/
@[simps!]
def _root_.Fin.appendIsometry (m n : ℕ) : (Fin m → α) × (Fin n → α) ≃ᵢ (Fin (m + n) → α) where
  toEquiv := Fin.appendEquiv _ _
                           /-
                             ι : Type u_1
                             α : Type u
                             β : Type v
                             γ : Type w
                             inst✝² : PseudoEMetricSpace α
                             inst✝¹ : PseudoEMetricSpace β
                             inst✝ : PseudoEMetricSpace γ
                             m n : Nat
                             x✝¹ x✝ : Prod (Fin m → α) (Fin n → α)
                             ⊢ Eq (EDist.edist ((Fin.appendEquiv m n).toFun x✝¹) ((Fin.appendEquiv m n).toF …
                           -/
  isometry_toFun _ _ := by simp_rw [Fin.appendEquiv, Fin.edist_append_eq_max_edist, Prod.edist_eq]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem _root_.Fin.appendIsometry_toHomeomorph (m n : ℕ) :
    (Fin.appendIsometry m n).toHomeomorph = Fin.appendHomeomorph (X := α) m n :=
  rfl


/-- `Equiv.funUnique` as an `IsometryEquiv`. -/
@[simps!]
def funUnique [Unique ι] [Fintype ι] : (ι → α) ≃ᵢ α where
  toEquiv := Equiv.funUnique ι α
                            /-
                              ι : Type u_1
                              α : Type u
                              β : Type v
                              γ : Type w
                              inst✝⁴ : PseudoEMetricSpace α
                              inst✝³ : PseudoEMetricSpace β
                              inst✝² : PseudoEMetricSpace γ
                              inst✝¹ : Unique ι
                              inst✝ : Fintype ι
                              x hx : ι → α
                              ⊢ Eq (EDist.edist ((Equiv.funUnique ι α).toFun x) ((Equiv.funUnique ι α).toFun …
                            -/
  isometry_toFun x hx := by simp [edist_pi_def, Finset.univ_unique, Finset.sup_singleton]
                            /-
                              🎉 no goals
                            -/


/-- `piFinTwoEquiv` as an `IsometryEquiv`. -/
@[simps!]
def piFinTwo (α : Fin 2 → Type*) [∀ i, PseudoEMetricSpace (α i)] : (∀ i, α i) ≃ᵢ α 0 × α 1 where
  toEquiv := piFinTwoEquiv α
                            /-
                              ι : Type u_1
                              α✝ : Type u
                              β : Type v
                              γ : Type w
                              inst✝³ : PseudoEMetricSpace α✝
                              inst✝² : PseudoEMetricSpace β
                              inst✝¹ : PseudoEMetricSpace γ
                              α : Fin 2 → Type u_2
                              inst✝ : (i : Fin 2) → PseudoEMetricSpace (α i)
                              x hx : (i : Fin 2) → α i
                              ⊢ Eq (EDist.edist ((piFinTwoEquiv α).toFun x) ((piFinTwoEquiv α).toFun hx)) (E …
                            -/
  isometry_toFun x hx := by simp [edist_pi_def, Fin.univ_succ, Prod.edist_eq]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem diam_image (s : Set α) : Metric.diam (h '' s) = Metric.diam s :=
  h.isometry.diam_image s


@[simp]
theorem diam_preimage (s : Set β) : Metric.diam (h ⁻¹' s) = Metric.diam s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    s : Set β
    ⊢ Eq (Metric.diam (Set.preimage (⇑h) s)) (Metric.diam s)
  -/
  rw [← image_symm, diam_image]
  /-
    🎉 no goals
  -/


include h in
theorem diam_univ : Metric.diam (univ : Set α) = Metric.diam (univ : Set β) :=
  congr_arg ENNReal.toReal h.ediam_univ


@[simp]
theorem preimage_ball (h : α ≃ᵢ β) (x : β) (r : ℝ) :
    h ⁻¹' Metric.ball x r = Metric.ball (h.symm x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    x : β
    r : Real
    ⊢ Eq (Set.preimage (⇑h) (Metric.ball x r)) (Metric.ball (h.symm x) r)
  -/
  rw [← h.isometry.preimage_ball (h.symm x) r, h.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_sphere (h : α ≃ᵢ β) (x : β) (r : ℝ) :
    h ⁻¹' Metric.sphere x r = Metric.sphere (h.symm x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    x : β
    r : Real
    ⊢ Eq (Set.preimage (⇑h) (Metric.sphere x r)) (Metric.sphere (h.symm x) r)
  -/
  rw [← h.isometry.preimage_sphere (h.symm x) r, h.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_closedBall (h : α ≃ᵢ β) (x : β) (r : ℝ) :
    h ⁻¹' Metric.closedBall x r = Metric.closedBall (h.symm x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    x : β
    r : Real
    ⊢ Eq (Set.preimage (⇑h) (Metric.closedBall x r)) (Metric.closedBall (h.symm x) …
  -/
  rw [← h.isometry.preimage_closedBall (h.symm x) r, h.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_ball (h : α ≃ᵢ β) (x : α) (r : ℝ) : h '' Metric.ball x r = Metric.ball (h x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    x : α
    r : Real
    ⊢ Eq (Set.image (⇑h) (Metric.ball x r)) (Metric.ball (h x) r)
  -/
  rw [← h.preimage_symm, h.symm.preimage_ball, symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_sphere (h : α ≃ᵢ β) (x : α) (r : ℝ) :
    h '' Metric.sphere x r = Metric.sphere (h x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    x : α
    r : Real
    ⊢ Eq (Set.image (⇑h) (Metric.sphere x r)) (Metric.sphere (h x) r)
  -/
  rw [← h.preimage_symm, h.symm.preimage_sphere, symm_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_closedBall (h : α ≃ᵢ β) (x : α) (r : ℝ) :
    h '' Metric.closedBall x r = Metric.closedBall (h x) r := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    h : IsometryEquiv α β
    x : α
    r : Real
    ⊢ Eq (Set.image (⇑h) (Metric.closedBall x r)) (Metric.closedBall (h x) r)
  -/
  rw [← h.preimage_symm, h.symm.preimage_closedBall, symm_symm]
  /-
    🎉 no goals
  -/


/-- An isometry induces an isometric isomorphism between the source space and the
range of the isometry. -/
@[simps! (config := { simpRhs := true }) toEquiv apply]
def Isometry.isometryEquivOnRange [EMetricSpace α] [PseudoEMetricSpace β] {f : α → β}
    (h : Isometry f) : α ≃ᵢ range f where
  isometry_toFun := h
  toEquiv := Equiv.ofInjective f h.injective


open NNReal in
/-- Post-composition by an isometry does not change the Lipschitz-property of a function. -/
lemma Isometry.lipschitzWith_iff {α β γ : Type*} [PseudoEMetricSpace α] [PseudoEMetricSpace β]
    [PseudoEMetricSpace γ] {f : α → β} {g : β → γ} (K : ℝ≥0) (h : Isometry g) :
    LipschitzWith K (g ∘ f) ↔ LipschitzWith K f := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : PseudoEMetricSpace β
    inst✝ : PseudoEMetricSpace γ
    f : α → β
    g : β → γ
    K : NNReal
    h : Isometry g
    ⊢ Iff (LipschitzWith K (Function.comp g f)) (LipschitzWith K f)
  -/
  simp [LipschitzWith, h.edist_eq]
  /-
    🎉 no goals
  -/

