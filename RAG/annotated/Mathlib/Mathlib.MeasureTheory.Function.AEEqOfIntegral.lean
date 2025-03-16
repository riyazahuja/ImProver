theorem ae_eq_zero_of_forall_inner [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
    [SecondCountableTopology E] {f : α → E} (hf : ∀ c : E, (fun x => (inner c (f x) : 𝕜)) =ᵐ[μ] 0) :
    f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : SecondCountableTopology E
    f : α → E
    hf : ∀ (c : E), (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner c (f x …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  let s := denseSeq E
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : SecondCountableTopology E
    f : α → E
    hf : ∀ (c : E), (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner c (f x …
    s : Nat → E := TopologicalSpace.denseSeq E
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hs : DenseRange s := denseRange_denseSeq E
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : SecondCountableTopology E
    f : α → E
    hf : ∀ (c : E), (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner c (f x …
    s : Nat → E := TopologicalSpace.denseSeq E
    hs : DenseRange s
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hf' : ∀ᵐ x ∂μ, ∀ n : ℕ, inner (s n) (f x) = (0 : 𝕜) := ae_all_iff.mpr fun n => hf (s n)
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : SecondCountableTopology E
    f : α → E
    hf : ∀ (c : E), (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner c (f x …
    s : Nat → E := TopologicalSpace.denseSeq E
    hs : DenseRange s
    hf' : Filter.Eventually (fun x => ∀ (n : Nat), Eq (Inner.inner (s n) (f x)) 0) …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  refine hf'.mono fun x hx => ?_
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : SecondCountableTopology E
    f : α → E
    hf : ∀ (c : E), (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner c (f x …
    s : Nat → E := TopologicalSpace.denseSeq E
    hs : DenseRange s
    hf' : Filter.Eventually (fun x => ∀ (n : Nat), Eq (Inner.inner (s n) (f x)) 0) …
    x : α
    hx : ∀ (n : Nat), Eq (Inner.inner (s n) (f x)) 0
    ⊢ Eq (f x) (0 x)
  -/
  rw [Pi.zero_apply, ← @inner_self_eq_zero 𝕜]
  have h_closed : IsClosed {c : E | inner c (f x) = (0 : 𝕜)} :=
    isClosed_eq (continuous_id.inner continuous_const) continuous_const
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : SecondCountableTopology E
    f : α → E
    hf : ∀ (c : E), (MeasureTheory.ae μ).EventuallyEq (fun x => Inner.inner c (f x …
    s : Nat → E := TopologicalSpace.denseSeq E
    hs : DenseRange s
    hf' : Filter.Eventually (fun x => ∀ (n : Nat), Eq (Inner.inner (s n) (f x)) 0) …
    x : α
    hx : ∀ (n : Nat), Eq (Inner.inner (s n) (f x)) 0
    h_closed : IsClosed (setOf fun c => Eq (Inner.inner c (f x)) 0)
    ⊢ Eq (Inner.inner (f x) (f x)) 0
  -/
  exact @isClosed_property ℕ E _ s (fun c => inner c (f x) = (0 : 𝕜)) hs h_closed hx _
  /-
    🎉 no goals
  -/


local notation "⟪" x ", " y "⟫" => y x


theorem ae_eq_zero_of_forall_dual_of_isSeparable [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    {t : Set E} (ht : TopologicalSpace.IsSeparable t) {f : α → E}
    (hf : ∀ c : Dual 𝕜 E, (fun x => ⟪f x, c⟫) =ᵐ[μ] 0) (h't : ∀ᵐ x ∂μ, f x ∈ t) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    ht : TopologicalSpace.IsSeparable t
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  rcases ht with ⟨d, d_count, hd⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    d : Set E
    d_count : d.Countable
    hd : HasSubset.Subset t (closure d)
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  haveI : Encodable d := d_count.toEncodable
  have : ∀ x : d, ∃ g : E →L[𝕜] 𝕜, ‖g‖ ≤ 1 ∧ g x = ‖(x : E)‖ :=
    fun x => exists_dual_vector'' 𝕜 (x : E)
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    d : Set E
    d_count : d.Countable
    hd : HasSubset.Subset t (closure d)
    this✝ : Encodable ↑d
    this : ∀ (x : ↑d), Exists fun g => And (LE.le (Norm.norm g) 1) (Eq (g ↑x) ↑(No …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  choose s hs using this
  have A : ∀ a : E, a ∈ t → (∀ x, ⟪a, s x⟫ = (0 : 𝕜)) → a = 0 := by
    intro a hat ha
    contrapose! ha
    have a_pos : 0 < ‖a‖ := by simp only [ha, norm_pos_iff, Ne, not_false_iff]
    have a_mem : a ∈ closure d := hd hat
    obtain ⟨x, hx⟩ : ∃ x : d, dist a x < ‖a‖ / 2 := by
      rcases Metric.mem_closure_iff.1 a_mem (‖a‖ / 2) (half_pos a_pos) with ⟨x, h'x, hx⟩
      exact ⟨⟨x, h'x⟩, hx⟩
    use x
    have I : ‖a‖ / 2 < ‖(x : E)‖ := by
      have : ‖a‖ ≤ ‖(x : E)‖ + ‖a - x‖ := norm_le_insert' _ _
      have : ‖a - x‖ < ‖a‖ / 2 := by rwa [dist_eq_norm] at hx
      linarith
    intro h
    apply lt_irrefl ‖s x x‖
    calc
      ‖s x x‖ = ‖s x (x - a)‖ := by simp only [h, sub_zero, ContinuousLinearMap.map_sub]
      _ ≤ 1 * ‖(x : E) - a‖ := ContinuousLinearMap.le_of_opNorm_le _ (hs x).1 _
      _ < ‖a‖ / 2 := by rw [one_mul]; rwa [dist_eq_norm'] at hx
      _ < ‖(x : E)‖ := I
      _ = ‖s x x‖ := by rw [(hs x).2, RCLike.norm_coe_norm]
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    d : Set E
    d_count : d.Countable
    hd : HasSubset.Subset t (closure d)
    this : Encodable ↑d
    s : ↑d → ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hs : ∀ (x : ↑d), And (LE.le (Norm.norm (s x)) 1) (Eq ((s x) ↑x) ↑(Norm.norm ↑x))
    A : ∀ (a : E), Membership.mem t a → (∀ (x : ↑d), Eq ((s x) a) 0) → Eq a 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hfs : ∀ y : d, ∀ᵐ x ∂μ, ⟪f x, s y⟫ = (0 : 𝕜) := fun y => hf (s y)
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    d : Set E
    d_count : d.Countable
    hd : HasSubset.Subset t (closure d)
    this : Encodable ↑d
    s : ↑d → ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hs : ∀ (x : ↑d), And (LE.le (Norm.norm (s x)) 1) (Eq ((s x) ↑x) ↑(Norm.norm ↑x))
    A : ∀ (a : E), Membership.mem t a → (∀ (x : ↑d), Eq ((s x) a) 0) → Eq a 0
    hfs : ∀ (y : ↑d), Filter.Eventually (fun x => Eq ((s y) (f x)) 0) (MeasureTheo …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hf' : ∀ᵐ x ∂μ, ∀ y : d, ⟪f x, s y⟫ = (0 : 𝕜) := by rwa [ae_all_iff]
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    d : Set E
    d_count : d.Countable
    hd : HasSubset.Subset t (closure d)
    this : Encodable ↑d
    s : ↑d → ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hs : ∀ (x : ↑d), And (LE.le (Norm.norm (s x)) 1) (Eq ((s x) ↑x) ↑(Norm.norm ↑x))
    A : ∀ (a : E), Membership.mem t a → (∀ (x : ↑d), Eq ((s x) a) 0) → Eq a 0
    hfs : ∀ (y : ↑d), Filter.Eventually (fun x => Eq ((s y) (f x)) 0) (MeasureTheo …
    hf' : Filter.Eventually (fun x => ∀ (y : ↑d), Eq ((s y) (f x)) 0) (MeasureTheo …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  filter_upwards [hf', h't] with x hx h'x
  /-
    case h
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    t : Set E
    f : α → E
    hf : ∀ (c : NormedSpace.Dual 𝕜 E), (MeasureTheory.ae μ).EventuallyEq (fun x => …
    h't : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    d : Set E
    d_count : d.Countable
    hd : HasSubset.Subset t (closure d)
    this : Encodable ↑d
    s : ↑d → ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hs : ∀ (x : ↑d), And (LE.le (Norm.norm (s x)) 1) (Eq ((s x) ↑x) ↑(Norm.norm ↑x))
    A : ∀ (a : E), Membership.mem t a → (∀ (x : ↑d), Eq ((s x) a) 0) → Eq a 0
    hfs : ∀ (y : ↑d), Filter.Eventually (fun x => Eq ((s y) (f x)) 0) (MeasureTheo …
    hf' : Filter.Eventually (fun x => ∀ (y : ↑d), Eq ((s y) (f x)) 0) (MeasureTheo …
    x : α
    hx : ∀ (y : ↑d), Eq ((s y) (f x)) 0
    h'x : Membership.mem t (f x)
    ⊢ Eq (f x) (0 x)
  -/
  exact A (f x) h'x hx
  /-
    🎉 no goals
  -/


theorem ae_eq_zero_of_forall_dual [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    [SecondCountableTopology E] {f : α → E} (hf : ∀ c : Dual 𝕜 E, (fun x => ⟪f x, c⟫) =ᵐ[μ] 0) :
    f =ᵐ[μ] 0 :=
  ae_eq_zero_of_forall_dual_of_isSeparable 𝕜 (.of_separableSpace Set.univ) hf
    (Eventually.of_forall fun _ => Set.mem_univ _)


theorem ae_nonneg_of_forall_setIntegral_nonneg (hf : Integrable f μ)
    (hf_zero : ∀ s, MeasurableSet s → μ s < ∞ → 0 ≤ ∫ x in s, f x ∂μ) : 0 ≤ᵐ[μ] f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 f
  -/
  simp_rw [EventuallyLE, Pi.zero_apply]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    ⊢ Filter.Eventually (fun x => LE.le 0 (f x)) (MeasureTheory.ae μ)
  -/
  rw [ae_const_le_iff_forall_lt_measure_zero]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    ⊢ ∀ (b : Real), LT.lt b 0 → Eq (μ (setOf fun x => LE.le (f x) b)) 0
  -/
  intro b hb_neg
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    b : Real
    hb_neg : LT.lt b 0
    ⊢ Eq (μ (setOf fun x => LE.le (f x) b)) 0
  -/
  let s := {x | f x ≤ b}
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    b : Real
    hb_neg : LT.lt b 0
    s : Set α := setOf fun x => LE.le (f x) b
    ⊢ Eq (μ (setOf fun x => LE.le (f x) b)) 0
  -/
  have hs : NullMeasurableSet s μ := nullMeasurableSet_le hf.1.aemeasurable aemeasurable_const
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    b : Real
    hb_neg : LT.lt b 0
    s : Set α := setOf fun x => LE.le (f x) b
    hs : MeasureTheory.NullMeasurableSet s μ
    ⊢ Eq (μ (setOf fun x => LE.le (f x) b)) 0
  -/
  have mus : μ s < ∞ := Integrable.measure_le_lt_top hf hb_neg
  have h_int_gt : (∫ x in s, f x ∂μ) ≤ b * (μ s).toReal := by
    have h_const_le : (∫ x in s, f x ∂μ) ≤ ∫ _ in s, b ∂μ := by
      refine setIntegral_mono_ae_restrict hf.integrableOn (integrableOn_const.mpr (Or.inr mus)) ?_
      rw [EventuallyLE, ae_restrict_iff₀ (hs.mono μ.restrict_le_self)]
      exact Eventually.of_forall fun x hxs => hxs
    rwa [setIntegral_const, smul_eq_mul, mul_comm] at h_const_le
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    b : Real
    hb_neg : LT.lt b 0
    s : Set α := setOf fun x => LE.le (f x) b
    hs : MeasureTheory.NullMeasurableSet s μ
    mus : LT.lt (μ s) Top.top
    h_int_gt : LE.le (MeasureTheory.integral (μ.restrict s) fun x => f x) (HMul.hM …
    ⊢ Eq (μ (setOf fun x => LE.le (f x) b)) 0
  -/
  contrapose! h_int_gt with H
  calc
    b * (μ s).toReal < 0 := mul_neg_of_neg_of_pos hb_neg <| ENNReal.toReal_pos H mus.ne
    _ ≤ ∫ x in s, f x ∂μ := by
      rw [← μ.restrict_toMeasurable mus.ne]
      exact hf_zero _ (measurableSet_toMeasurable ..) (by rwa [measure_toMeasurable])


@[deprecated (since := "2024-04-17")]
alias ae_nonneg_of_forall_set_integral_nonneg_of_stronglyMeasurable :=
  ae_nonneg_of_forall_setIntegral_nonneg


@[deprecated (since := "2024-07-12")]
alias ae_nonneg_of_forall_setIntegral_nonneg_of_stronglyMeasurable :=
  ae_nonneg_of_forall_setIntegral_nonneg


@[deprecated (since := "2024-04-17")]
alias ae_nonneg_of_forall_set_integral_nonneg :=
  ae_nonneg_of_forall_setIntegral_nonneg


theorem ae_le_of_forall_setIntegral_le {f g : α → ℝ} (hf : Integrable f μ) (hg : Integrable g μ)
    (hf_le : ∀ s, MeasurableSet s → μ s < ∞ → (∫ x in s, f x ∂μ) ≤ ∫ x in s, g x ∂μ) :
    f ≤ᵐ[μ] g := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hf_le : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureT …
    ⊢ (MeasureTheory.ae μ).EventuallyLE f g
  -/
  rw [← eventually_sub_nonneg]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hf_le : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureT …
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (HSub.hSub g f)
  -/
  refine ae_nonneg_of_forall_setIntegral_nonneg (hg.sub hf) fun s hs => ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hf_le : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureT …
    s : Set α
    hs : MeasurableSet s
    ⊢ LT.lt (μ s) Top.top → LE.le 0 (MeasureTheory.integral (μ.restrict s) fun x = …
  -/
  rw [integral_sub' hg.integrableOn hf.integrableOn, sub_nonneg]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hf_le : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le (MeasureT …
    s : Set α
    hs : MeasurableSet s
    ⊢ LT.lt (μ s) Top.top → LE.le (MeasureTheory.integral (μ.restrict s) fun a =>  …
  -/
  exact hf_le s hs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_le_of_forall_set_integral_le := ae_le_of_forall_setIntegral_le


theorem ae_nonneg_restrict_of_forall_setIntegral_nonneg_inter {f : α → ℝ} {t : Set α}
    (hf : IntegrableOn f t μ)
    (hf_zero : ∀ s, MeasurableSet s → μ (s ∩ t) < ∞ → 0 ≤ ∫ x in s ∩ t, f x ∂μ) :
    0 ≤ᵐ[μ.restrict t] f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    t : Set α
    hf : MeasureTheory.IntegrableOn f t μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ (Inter.inter s t)) Top.top …
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyLE 0 f
  -/
  refine ae_nonneg_of_forall_setIntegral_nonneg hf fun s hs h's => ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    t : Set α
    hf : MeasureTheory.IntegrableOn f t μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ (Inter.inter s t)) Top.top …
    s : Set α
    hs : MeasurableSet s
    h's : LT.lt ((μ.restrict t) s) Top.top
    ⊢ LE.le 0 (MeasureTheory.integral ((μ.restrict t).restrict s) fun x => f x)
  -/
  simp_rw [Measure.restrict_restrict hs]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    t : Set α
    hf : MeasureTheory.IntegrableOn f t μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ (Inter.inter s t)) Top.top …
    s : Set α
    hs : MeasurableSet s
    h's : LT.lt ((μ.restrict t) s) Top.top
    ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x)
  -/
  apply hf_zero s hs
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    t : Set α
    hf : MeasureTheory.IntegrableOn f t μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ (Inter.inter s t)) Top.top …
    s : Set α
    hs : MeasurableSet s
    h's : LT.lt ((μ.restrict t) s) Top.top
    ⊢ LT.lt (μ (Inter.inter s t)) Top.top
  -/
  rwa [Measure.restrict_apply hs] at h's
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_nonneg_restrict_of_forall_set_integral_nonneg_inter :=
  ae_nonneg_restrict_of_forall_setIntegral_nonneg_inter


theorem ae_nonneg_of_forall_setIntegral_nonneg_of_sigmaFinite [SigmaFinite μ] {f : α → ℝ}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s, MeasurableSet s → μ s < ∞ → 0 ≤ ∫ x in s, f x ∂μ) : 0 ≤ᵐ[μ] f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 f
  -/
  apply ae_of_forall_measure_lt_top_ae_restrict
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Filter.Eventually (fu …
  -/
  intro t t_meas t_lt_top
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    t : Set α
    t_meas : MeasurableSet t
    t_lt_top : LT.lt (μ t) Top.top
    ⊢ Filter.Eventually (fun x => LE.le (0 x) (f x)) (MeasureTheory.ae (μ.restrict …
  -/
  apply ae_nonneg_restrict_of_forall_setIntegral_nonneg_inter (hf_int_finite t t_meas t_lt_top)
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    t : Set α
    t_meas : MeasurableSet t
    t_lt_top : LT.lt (μ t) Top.top
    ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ (Inter.inter s t)) Top.top → LE.le …
  -/
  intro s s_meas _
  exact
    hf_zero _ (s_meas.inter t_meas)
      (lt_of_le_of_lt (measure_mono (Set.inter_subset_right)) t_lt_top)


@[deprecated (since := "2024-04-17")]
alias ae_nonneg_of_forall_set_integral_nonneg_of_sigmaFinite :=
  ae_nonneg_of_forall_setIntegral_nonneg_of_sigmaFinite


theorem AEFinStronglyMeasurable.ae_nonneg_of_forall_setIntegral_nonneg {f : α → ℝ}
    (hf : AEFinStronglyMeasurable f μ)
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s, MeasurableSet s → μ s < ∞ → 0 ≤ ∫ x in s, f x ∂μ) : 0 ≤ᵐ[μ] f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 f
  -/
  let t := hf.sigmaFiniteSet
  suffices 0 ≤ᵐ[μ.restrict t] f from
    ae_of_ae_restrict_of_ae_restrict_compl _ this hf.ae_eq_zero_compl.symm.le
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    t : Set α := hf.sigmaFiniteSet
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyLE 0 f
  -/
  haveI : SigmaFinite (μ.restrict t) := hf.sigmaFinite_restrict
  refine
    ae_nonneg_of_forall_setIntegral_nonneg_of_sigmaFinite (fun s hs hμts => ?_) fun s hs hμts => ?_
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμts : LT.lt ((μ.restrict t) s) Top.top
      ⊢ MeasureTheory.IntegrableOn f s (μ.restrict t)
    -/
  · rw [IntegrableOn, Measure.restrict_restrict hs]
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμts : LT.lt ((μ.restrict t) s) Top.top
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter s t))
    -/
    rw [Measure.restrict_apply hs] at hμts
    /-
      case refine_1
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμts : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter s t))
    -/
    exact hf_int_finite (s ∩ t) (hs.inter hf.measurableSet) hμts
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμts : LT.lt ((μ.restrict t) s) Top.top
      ⊢ LE.le 0 (MeasureTheory.integral ((μ.restrict t).restrict s) fun x => f x)
    -/
  · rw [Measure.restrict_restrict hs]
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμts : LT.lt ((μ.restrict t) s) Top.top
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x)
    -/
    rw [Measure.restrict_apply hs] at hμts
    /-
      case refine_2
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμts : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x)
    -/
    exact hf_zero (s ∩ t) (hs.inter hf.measurableSet) hμts
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias AEFinStronglyMeasurable.ae_nonneg_of_forall_set_integral_nonneg :=
  AEFinStronglyMeasurable.ae_nonneg_of_forall_setIntegral_nonneg


theorem ae_nonneg_restrict_of_forall_setIntegral_nonneg {f : α → ℝ}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s, MeasurableSet s → μ s < ∞ → 0 ≤ ∫ x in s, f x ∂μ) {t : Set α}
    (ht : MeasurableSet t) (hμt : μ t ≠ ∞) : 0 ≤ᵐ[μ.restrict t] f := by
  refine
    ae_nonneg_restrict_of_forall_setIntegral_nonneg_inter
      (hf_int_finite t ht (lt_top_iff_ne_top.mpr hμt)) fun s hs _ => ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (μ (Inter.inter s t)) Top.top
    ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x)
  -/
  refine hf_zero (s ∩ t) (hs.inter ht) ?_
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → LE.le 0 (Meas …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    s : Set α
    hs : MeasurableSet s
    x✝ : LT.lt (μ (Inter.inter s t)) Top.top
    ⊢ LT.lt (μ (Inter.inter s t)) Top.top
  -/
  exact (measure_mono Set.inter_subset_right).trans_lt (lt_top_iff_ne_top.mpr hμt)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_nonneg_restrict_of_forall_set_integral_nonneg :=
  ae_nonneg_restrict_of_forall_setIntegral_nonneg


theorem ae_eq_zero_restrict_of_forall_setIntegral_eq_zero_real {f : α → ℝ}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = 0) {t : Set α}
    (ht : MeasurableSet t) (hμt : μ t ≠ ∞) : f =ᵐ[μ.restrict t] 0 := by
  suffices h_and : f ≤ᵐ[μ.restrict t] 0 ∧ 0 ≤ᵐ[μ.restrict t] f from
    h_and.1.mp (h_and.2.mono fun x hx1 hx2 => le_antisymm hx2 hx1)
  refine
    ⟨?_,
      ae_nonneg_restrict_of_forall_setIntegral_nonneg hf_int_finite
        (fun s hs hμs => (hf_zero s hs hμs).symm.le) ht hμt⟩
  suffices h_neg : 0 ≤ᵐ[μ.restrict t] -f by
    refine h_neg.mono fun x hx => ?_
    rw [Pi.neg_apply] at hx
    simpa using hx
  refine
    ae_nonneg_restrict_of_forall_setIntegral_nonneg (fun s hs hμs => (hf_int_finite s hs hμs).neg)
      (fun s hs hμs => ?_) ht hμt
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict s) fun x => Neg.neg f x)
  -/
  simp_rw [Pi.neg_apply]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict s) fun x => Neg.neg (f x))
  -/
  rw [integral_neg, neg_nonneg]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    s : Set α
    hs : MeasurableSet s
    hμs : LT.lt (μ s) Top.top
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun a => f a) 0
  -/
  exact (hf_zero s hs hμs).le
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_zero_restrict_of_forall_set_integral_eq_zero_real :=
  ae_eq_zero_restrict_of_forall_setIntegral_eq_zero_real


theorem ae_eq_zero_restrict_of_forall_setIntegral_eq_zero {f : α → E}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = 0) {t : Set α}
    (ht : MeasurableSet t) (hμt : μ t ≠ ∞) : f =ᵐ[μ.restrict t] 0 := by
  rcases (hf_int_finite t ht hμt.lt_top).aestronglyMeasurable.isSeparable_ae_range with
    ⟨u, u_sep, hu⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    u : Set E
    u_sep : TopologicalSpace.IsSeparable u
    hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq f 0
  -/
  refine ae_eq_zero_of_forall_dual_of_isSeparable ℝ u_sep (fun c => ?_) hu
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    u : Set E
    u_sep : TopologicalSpace.IsSeparable u
    hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
    c : NormedSpace.Dual Real E
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq (fun x => c (f x)) 0
  -/
  refine ae_eq_zero_restrict_of_forall_setIntegral_eq_zero_real ?_ ?_ ht hμt
    /-
      case intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      u : Set E
      u_sep : TopologicalSpace.IsSeparable u
      hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
      c : NormedSpace.Dual Real E
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory.Integra …
    -/
  · intro s hs hμs
    /-
      case intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      u : Set E
      u_sep : TopologicalSpace.IsSeparable u
      hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
      c : NormedSpace.Dual Real E
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.IntegrableOn (fun x => c (f x)) s μ
    -/
    exact ContinuousLinearMap.integrable_comp c (hf_int_finite s hs hμs)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      u : Set E
      u_sep : TopologicalSpace.IsSeparable u
      hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
      c : NormedSpace.Dual Real E
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs hμs
    /-
      case intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      u : Set E
      u_sep : TopologicalSpace.IsSeparable u
      hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
      c : NormedSpace.Dual Real E
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => c (f x)) 0
    -/
    rw [ContinuousLinearMap.integral_comp_comm c (hf_int_finite s hs hμs), hf_zero s hs hμs]
    /-
      case intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      u : Set E
      u_sep : TopologicalSpace.IsSeparable u
      hu : Filter.Eventually (fun x => Membership.mem u (f x)) (MeasureTheory.ae (μ. …
      c : NormedSpace.Dual Real E
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ Eq (c 0) 0
    -/
    exact ContinuousLinearMap.map_zero _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_zero_restrict_of_forall_set_integral_eq_zero :=
  ae_eq_zero_restrict_of_forall_setIntegral_eq_zero


theorem ae_eq_restrict_of_forall_setIntegral_eq {f g : α → E}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hg_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn g s μ)
    (hfg_zero : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ)
    {t : Set α} (ht : MeasurableSet t) (hμt : μ t ≠ ∞) : f =ᵐ[μ.restrict t] g := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureT …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq f g
  -/
  rw [← sub_ae_eq_zero]
  have hfg' : ∀ s : Set α, MeasurableSet s → μ s < ∞ → (∫ x in s, (f - g) x ∂μ) = 0 := by
    intro s hs hμs
    rw [integral_sub' (hf_int_finite s hs hμs) (hg_int_finite s hs hμs)]
    exact sub_eq_zero.mpr (hfg_zero s hs hμs)
  have hfg_int : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn (f - g) s μ := fun s hs hμs =>
    (hf_int_finite s hs hμs).sub (hg_int_finite s hs hμs)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureT …
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    hfg' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheor …
    hfg_int : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory …
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq (HSub.hSub f g) 0
  -/
  exact ae_eq_zero_restrict_of_forall_setIntegral_eq_zero hfg_int hfg' ht hμt
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_restrict_of_forall_set_integral_eq :=
  ae_eq_restrict_of_forall_setIntegral_eq


theorem ae_eq_zero_of_forall_setIntegral_eq_of_sigmaFinite [SigmaFinite μ] {f : α → E}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = 0) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  let S := spanningSets μ
  rw [← @Measure.restrict_univ _ _ μ, ← iUnion_spanningSets μ, EventuallyEq, ae_iff,
    Measure.restrict_apply' (MeasurableSet.iUnion (measurableSet_spanningSets μ))]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    S : Nat → Set α := MeasureTheory.spanningSets μ
    ⊢ Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) (0 a))) (Set.iUnion fun b = …
  -/
  rw [Set.inter_iUnion, measure_iUnion_null_iff]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    S : Nat → Set α := MeasureTheory.spanningSets μ
    ⊢ ∀ (i : Nat), Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) (0 a))) (Measu …
  -/
  intro n
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    S : Nat → Set α := MeasureTheory.spanningSets μ
    n : Nat
    ⊢ Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) (0 a))) (MeasureTheory.span …
  -/
  have h_meas_n : MeasurableSet (S n) := measurableSet_spanningSets μ n
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    S : Nat → Set α := MeasureTheory.spanningSets μ
    n : Nat
    h_meas_n : MeasurableSet (S n)
    ⊢ Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) (0 a))) (MeasureTheory.span …
  -/
  have hμn : μ (S n) < ∞ := measure_spanningSets_lt_top μ n
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    S : Nat → Set α := MeasureTheory.spanningSets μ
    n : Nat
    h_meas_n : MeasurableSet (S n)
    hμn : LT.lt (μ (S n)) Top.top
    ⊢ Eq (μ (Inter.inter (setOf fun a => Not (Eq (f a) (0 a))) (MeasureTheory.span …
  -/
  rw [← Measure.restrict_apply' h_meas_n]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    S : Nat → Set α := MeasureTheory.spanningSets μ
    n : Nat
    h_meas_n : MeasurableSet (S n)
    hμn : LT.lt (μ (S n)) Top.top
    ⊢ Eq ((μ.restrict (S n)) (setOf fun a => Not (Eq (f a) (0 a)))) 0
  -/
  exact ae_eq_zero_restrict_of_forall_setIntegral_eq_zero hf_int_finite hf_zero h_meas_n hμn.ne
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_zero_of_forall_set_integral_eq_of_sigmaFinite :=
  ae_eq_zero_of_forall_setIntegral_eq_of_sigmaFinite


theorem ae_eq_of_forall_setIntegral_eq_of_sigmaFinite [SigmaFinite μ] {f g : α → E}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hg_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn g s μ)
    (hfg_eq : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ) :
    f =ᵐ[μ] g := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f g : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureThe …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  rw [← sub_ae_eq_zero]
  have hfg : ∀ s : Set α, MeasurableSet s → μ s < ∞ → (∫ x in s, (f - g) x ∂μ) = 0 := by
    intro s hs hμs
    rw [integral_sub' (hf_int_finite s hs hμs) (hg_int_finite s hs hμs),
      sub_eq_zero.mpr (hfg_eq s hs hμs)]
  have hfg_int : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn (f - g) s μ := fun s hs hμs =>
    (hf_int_finite s hs hμs).sub (hg_int_finite s hs hμs)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.SigmaFinite μ
    f g : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureThe …
    hfg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hfg_int : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSub.hSub f g) 0
  -/
  exact ae_eq_zero_of_forall_setIntegral_eq_of_sigmaFinite hfg_int hfg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_of_forall_set_integral_eq_of_sigmaFinite :=
  ae_eq_of_forall_setIntegral_eq_of_sigmaFinite


theorem AEFinStronglyMeasurable.ae_eq_zero_of_forall_setIntegral_eq_zero {f : α → E}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = 0)
    (hf : AEFinStronglyMeasurable f μ) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  let t := hf.sigmaFiniteSet
  suffices f =ᵐ[μ.restrict t] 0 from
    ae_of_ae_restrict_of_ae_restrict_compl _ this hf.ae_eq_zero_compl
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    t : Set α := hf.sigmaFiniteSet
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq f 0
  -/
  haveI : SigmaFinite (μ.restrict t) := hf.sigmaFinite_restrict
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    t : Set α := hf.sigmaFiniteSet
    this : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq f 0
  -/
  refine ae_eq_zero_of_forall_setIntegral_eq_of_sigmaFinite ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt ((μ.restrict t) s) Top.top → MeasureT …
    -/
  · intro s hs hμs
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt ((μ.restrict t) s) Top.top
      ⊢ MeasureTheory.IntegrableOn f s (μ.restrict t)
    -/
    rw [IntegrableOn, Measure.restrict_restrict hs]
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt ((μ.restrict t) s) Top.top
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter s t))
    -/
    rw [Measure.restrict_apply hs] at hμs
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter s t))
    -/
    exact hf_int_finite _ (hs.inter hf.measurableSet) hμs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt ((μ.restrict t) s) Top.top → Eq (Meas …
    -/
  · intro s hs hμs
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt ((μ.restrict t) s) Top.top
      ⊢ Eq (MeasureTheory.integral ((μ.restrict t).restrict s) fun x => f x) 0
    -/
    rw [Measure.restrict_restrict hs]
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt ((μ.restrict t) s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x) 0
    -/
    rw [Measure.restrict_apply hs] at hμs
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.AEFinStronglyMeasurable f μ
      t : Set α := hf.sigmaFiniteSet
      this : MeasureTheory.SigmaFinite (μ.restrict t)
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x) 0
    -/
    exact hf_zero _ (hs.inter hf.measurableSet) hμs
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias AEFinStronglyMeasurable.ae_eq_zero_of_forall_set_integral_eq_zero :=
  AEFinStronglyMeasurable.ae_eq_zero_of_forall_setIntegral_eq_zero


theorem AEFinStronglyMeasurable.ae_eq_of_forall_setIntegral_eq {f g : α → E}
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hg_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn g s μ)
    (hfg_eq : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ)
    (hf : AEFinStronglyMeasurable f μ) (hg : AEFinStronglyMeasurable g μ) : f =ᵐ[μ] g := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureThe …
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    hg : MeasureTheory.AEFinStronglyMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  rw [← sub_ae_eq_zero]
  have hfg : ∀ s : Set α, MeasurableSet s → μ s < ∞ → (∫ x in s, (f - g) x ∂μ) = 0 := by
    intro s hs hμs
    rw [integral_sub' (hf_int_finite s hs hμs) (hg_int_finite s hs hμs),
      sub_eq_zero.mpr (hfg_eq s hs hμs)]
  have hfg_int : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn (f - g) s μ := fun s hs hμs =>
    (hf_int_finite s hs hμs).sub (hg_int_finite s hs hμs)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hg_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hfg_eq : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureThe …
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    hg : MeasureTheory.AEFinStronglyMeasurable g μ
    hfg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hfg_int : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSub.hSub f g) 0
  -/
  exact (hf.sub hg).ae_eq_zero_of_forall_setIntegral_eq_zero hfg_int hfg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias AEFinStronglyMeasurable.ae_eq_of_forall_set_integral_eq :=
  AEFinStronglyMeasurable.ae_eq_of_forall_setIntegral_eq


theorem Lp.ae_eq_zero_of_forall_setIntegral_eq_zero (f : Lp E p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = 0) : f =ᵐ[μ] 0 :=
  AEFinStronglyMeasurable.ae_eq_zero_of_forall_setIntegral_eq_zero hf_int_finite hf_zero
    (Lp.finStronglyMeasurable _ hp_ne_zero hp_ne_top).aefinStronglyMeasurable


@[deprecated (since := "2024-04-17")]
alias Lp.ae_eq_zero_of_forall_set_integral_eq_zero :=
  Lp.ae_eq_zero_of_forall_setIntegral_eq_zero


theorem Lp.ae_eq_of_forall_setIntegral_eq (f g : Lp E p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (hf_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn f s μ)
    (hg_int_finite : ∀ s, MeasurableSet s → μ s < ∞ → IntegrableOn g s μ)
    (hfg : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ) :
    f =ᵐ[μ] g :=
  AEFinStronglyMeasurable.ae_eq_of_forall_setIntegral_eq hf_int_finite hg_int_finite hfg
    (Lp.finStronglyMeasurable _ hp_ne_zero hp_ne_top).aefinStronglyMeasurable
    (Lp.finStronglyMeasurable _ hp_ne_zero hp_ne_top).aefinStronglyMeasurable


@[deprecated (since := "2024-04-17")]
alias Lp.ae_eq_of_forall_set_integral_eq := Lp.ae_eq_of_forall_setIntegral_eq


theorem ae_eq_zero_of_forall_setIntegral_eq_of_finStronglyMeasurable_trim (hm : m ≤ m0) {f : α → E}
    (hf_int_finite : ∀ s, MeasurableSet[m] s → μ s < ∞ → IntegrableOn f s μ)
    (hf_zero : ∀ s : Set α, MeasurableSet[m] s → μ s < ∞ → ∫ x in s, f x ∂μ = 0)
    (hf : FinStronglyMeasurable f (μ.trim hm)) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hm : LE.le m m0
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  obtain ⟨t, ht_meas, htf_zero, htμ⟩ := hf.exists_set_sigmaFinite
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hm : LE.le m m0
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
    t : Set α
    ht_meas : MeasurableSet t
    htf_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  haveI : SigmaFinite ((μ.restrict t).trim hm) := by rwa [restrict_trim hm μ ht_meas] at htμ
  have htf_zero : f =ᵐ[μ.restrict tᶜ] 0 := by
    rw [EventuallyEq, ae_restrict_iff' (MeasurableSet.compl (hm _ ht_meas))]
    exact Eventually.of_forall htf_zero
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hm : LE.le m m0
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
    t : Set α
    ht_meas : MeasurableSet t
    htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
    this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
    htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hf_meas_m : StronglyMeasurable[m] f := hf.stronglyMeasurable
  suffices f =ᵐ[μ.restrict t] 0 from
    ae_of_ae_restrict_of_ae_restrict_compl _ this htf_zero
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hm : LE.le m m0
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
    t : Set α
    ht_meas : MeasurableSet t
    htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
    this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
    htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
    hf_meas_m : MeasureTheory.StronglyMeasurable f
    ⊢ (MeasureTheory.ae (μ.restrict t)).EventuallyEq f 0
  -/
  refine measure_eq_zero_of_trim_eq_zero hm ?_
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_2
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hm : LE.le m m0
    f : α → E
    hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
    t : Set α
    ht_meas : MeasurableSet t
    htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
    this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
    htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
    hf_meas_m : MeasureTheory.StronglyMeasurable f
    ⊢ Eq (((μ.restrict t).trim hm) (HasCompl.compl (setOf fun x => (fun x => Eq (f …
  -/
  refine ae_eq_zero_of_forall_setIntegral_eq_of_sigmaFinite ?_ ?_
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (((μ.restrict t).trim hm) s) Top.top  …
    -/
  · intro s hs hμs
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (((μ.restrict t).trim hm) s) Top.top
      ⊢ MeasureTheory.IntegrableOn f s ((μ.restrict t).trim hm)
    -/
    unfold IntegrableOn
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (((μ.restrict t).trim hm) s) Top.top
      ⊢ MeasureTheory.Integrable f (((μ.restrict t).trim hm).restrict s)
    -/
    rw [restrict_trim hm (μ.restrict t) hs, Measure.restrict_restrict (hm s hs)]
    rw [← restrict_trim hm μ ht_meas, Measure.restrict_apply hs,
      trim_measurableSet_eq hm (hs.inter ht_meas)] at hμs
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ MeasureTheory.Integrable f ((μ.restrict (Inter.inter s t)).trim hm)
    -/
    refine Integrable.trim hm ?_ hf_meas_m
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter s t))
    -/
    exact hf_int_finite _ (hs.inter ht_meas) hμs
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (((μ.restrict t).trim hm) s) Top.top  …
    -/
  · intro s hs hμs
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (((μ.restrict t).trim hm) s) Top.top
      ⊢ Eq (MeasureTheory.integral (((μ.restrict t).trim hm).restrict s) fun x => f  …
    -/
    rw [restrict_trim hm (μ.restrict t) hs, Measure.restrict_restrict (hm s hs)]
    rw [← restrict_trim hm μ ht_meas, Measure.restrict_apply hs,
      trim_measurableSet_eq hm (hs.inter ht_meas)] at hμs
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ Eq (MeasureTheory.integral ((μ.restrict (Inter.inter s t)).trim hm) fun x => …
    -/
    rw [← integral_trim hm hf_meas_m]
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      E : Type u_2
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hm : LE.le m m0
      f : α → E
      hf_int_finite : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Measure …
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf : MeasureTheory.FinStronglyMeasurable f (μ.trim hm)
      t : Set α
      ht_meas : MeasurableSet t
      htf_zero✝ : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ : MeasureTheory.SigmaFinite ((μ.trim hm).restrict t)
      this : MeasureTheory.SigmaFinite ((μ.restrict t).trim hm)
      htf_zero : (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
      hf_meas_m : MeasureTheory.StronglyMeasurable f
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ (Inter.inter s t)) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x => f x) 0
    -/
    exact hf_zero _ (hs.inter ht_meas) hμs
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias ae_eq_zero_of_forall_set_integral_eq_of_finStronglyMeasurable_trim :=
  ae_eq_zero_of_forall_setIntegral_eq_of_finStronglyMeasurable_trim


theorem Integrable.ae_eq_zero_of_forall_setIntegral_eq_zero {f : α → E} (hf : Integrable f μ)
    (hf_zero : ∀ s, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = 0) : f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hf_Lp : Memℒp f 1 μ := memℒp_one_iff_integrable.mpr hf
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_Lp : MeasureTheory.Memℒp f 1 μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  let f_Lp := hf_Lp.toLp f
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_Lp : MeasureTheory.Memℒp f 1 μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  have hf_f_Lp : f =ᵐ[μ] f_Lp := (Memℒp.coeFn_toLp hf_Lp).symm
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_Lp : MeasureTheory.Memℒp f 1 μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
    hf_f_Lp : (MeasureTheory.ae μ).EventuallyEq f ↑↑f_Lp
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  refine hf_f_Lp.trans ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
    hf_Lp : MeasureTheory.Memℒp f 1 μ
    f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
    hf_f_Lp : (MeasureTheory.ae μ).EventuallyEq f ↑↑f_Lp
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑f_Lp) 0
  -/
  refine Lp.ae_eq_zero_of_forall_setIntegral_eq_zero f_Lp one_ne_zero ENNReal.coe_ne_top ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.Integrable f μ
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_Lp : MeasureTheory.Memℒp f 1 μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
      hf_f_Lp : (MeasureTheory.ae μ).EventuallyEq f ↑↑f_Lp
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory.Integra …
    -/
  · exact fun s _ _ => Integrable.integrableOn (L1.integrable_coeFn _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.Integrable f μ
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_Lp : MeasureTheory.Memℒp f 1 μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
      hf_f_Lp : (MeasureTheory.ae μ).EventuallyEq f ↑↑f_Lp
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs hμs
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.Integrable f μ
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_Lp : MeasureTheory.Memℒp f 1 μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
      hf_f_Lp : (MeasureTheory.ae μ).EventuallyEq f ↑↑f_Lp
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑f_Lp x) 0
    -/
    rw [integral_congr_ae (ae_restrict_of_ae hf_f_Lp.symm)]
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f : α → E
      hf : MeasureTheory.Integrable f μ
      hf_zero : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTh …
      hf_Lp : MeasureTheory.Memℒp f 1 μ
      f_Lp : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 μ) x := MeasureTh …
      hf_f_Lp : (MeasureTheory.ae μ).EventuallyEq f ↑↑f_Lp
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun a => f a) 0
    -/
    exact hf_zero s hs hμs
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias Integrable.ae_eq_zero_of_forall_set_integral_eq_zero :=
  Integrable.ae_eq_zero_of_forall_setIntegral_eq_zero


theorem Integrable.ae_eq_of_forall_setIntegral_eq (f g : α → E) (hf : Integrable f μ)
    (hg : Integrable g μ)
    (hfg : ∀ s : Set α, MeasurableSet s → μ s < ∞ → ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ) :
    f =ᵐ[μ] g := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  rw [← sub_ae_eq_zero]
  have hfg' : ∀ s : Set α, MeasurableSet s → μ s < ∞ → (∫ x in s, (f - g) x ∂μ) = 0 := by
    intro s hs hμs
    rw [integral_sub' hf.integrableOn hg.integrableOn]
    exact sub_eq_zero.mpr (hfg s hs hμs)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory …
    hfg' : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheor …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSub.hSub f g) 0
  -/
  exact Integrable.ae_eq_zero_of_forall_setIntegral_eq_zero (hf.sub hg) hfg'
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias Integrable.ae_eq_of_forall_set_integral_eq :=
  Integrable.ae_eq_of_forall_setIntegral_eq


/-- If an integrable function has zero integral on all closed sets, then it is zero
almost everywhere. -/
lemma ae_eq_zero_of_forall_setIntegral_isClosed_eq_zero {μ : Measure β} {f : β → E}
    (hf : Integrable f μ) (h'f : ∀ (s : Set β), IsClosed s → ∫ x in s, f x ∂μ = 0) :
    f =ᵐ[μ] 0 := by
  suffices ∀ s, MeasurableSet s → ∫ x in s, f x ∂μ = 0 from
    hf.ae_eq_zero_of_forall_setIntegral_eq_zero (fun s hs _ ↦ this s hs)
  have A : ∀ (t : Set β), MeasurableSet t → ∫ (x : β) in t, f x ∂μ = 0
      → ∫ (x : β) in tᶜ, f x ∂μ = 0 := by
    intro t t_meas ht
    have I : ∫ x, f x ∂μ = 0 := by rw [← setIntegral_univ]; exact h'f _ isClosed_univ
    simpa [ht, I] using integral_add_compl t_meas hf
  /-
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    β : Type u_3
    inst✝² : TopologicalSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.Integrable f μ
    h'f : ∀ (s : Set β), IsClosed s → Eq (MeasureTheory.integral (μ.restrict s) fu …
    A : ∀ (t : Set β), MeasurableSet t → Eq (MeasureTheory.integral (μ.restrict t) …
    ⊢ ∀ (s : Set β), MeasurableSet s → Eq (MeasureTheory.integral (μ.restrict s) f …
  -/
  intro s hs
  induction s, hs using MeasurableSet.induction_on_open with
  | isOpen U hU => exact compl_compl U ▸ A _ hU.measurableSet.compl (h'f _ hU.isClosed_compl)
  | compl s hs ihs => exact A s hs ihs
  | iUnion g g_disj g_meas hg => simp [integral_iUnion g_meas g_disj hf.integrableOn, hg]


@[deprecated (since := "2024-04-17")]
alias ae_eq_zero_of_forall_set_integral_isClosed_eq_zero :=
  ae_eq_zero_of_forall_setIntegral_isClosed_eq_zero


/-- If an integrable function has zero integral on all compact sets in a sigma-compact space, then
it is zero almost everywhere. -/
lemma ae_eq_zero_of_forall_setIntegral_isCompact_eq_zero
    [SigmaCompactSpace β] [R1Space β] {μ : Measure β} {f : β → E} (hf : Integrable f μ)
    (h'f : ∀ (s : Set β), IsCompact s → ∫ x in s, f x ∂μ = 0) :
    f =ᵐ[μ] 0 := by
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.Integrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  apply ae_eq_zero_of_forall_setIntegral_isClosed_eq_zero hf (fun s hs ↦ ?_)
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.Integrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    s : Set β
    hs : IsClosed s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) 0
  -/
  let t : ℕ → Set β := fun n ↦ closure (compactCovering β n) ∩ s
  suffices H : Tendsto (fun n ↦ ∫ x in t n, f x ∂μ) atTop (𝓝 (∫ x in s, f x ∂μ)) by
    have A : ∀ n, ∫ x in t n, f x ∂μ = 0 :=
      fun n ↦ h'f _ ((isCompact_compactCovering β n).closure.inter_right hs)
    simp_rw [A, tendsto_const_nhds_iff] at H
    exact H.symm
  have B : s = ⋃ n, t n := by
    rw [← Set.iUnion_inter, iUnion_closure_compactCovering, Set.univ_inter]
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.Integrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    s : Set β
    hs : IsClosed s
    t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
    B : Eq s (Set.iUnion fun n => t n)
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral (μ.restrict (t n)) fun x =>  …
  -/
  rw [B]
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.Integrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    s : Set β
    hs : IsClosed s
    t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
    B : Eq s (Set.iUnion fun n => t n)
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral (μ.restrict (t n)) fun x =>  …
  -/
  apply tendsto_setIntegral_of_monotone
    /-
      case hsm
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.Integrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      s : Set β
      hs : IsClosed s
      t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
      B : Eq s (Set.iUnion fun n => t n)
      ⊢ ∀ (i : Nat), MeasurableSet (t i)
    -/
  · intros n
    /-
      case hsm
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.Integrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      s : Set β
      hs : IsClosed s
      t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
      B : Eq s (Set.iUnion fun n => t n)
      n : Nat
      ⊢ MeasurableSet (t n)
    -/
    exact (isClosed_closure.inter hs).measurableSet
    /-
      🎉 no goals
    -/
    /-
      case h_mono
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.Integrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      s : Set β
      hs : IsClosed s
      t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
      B : Eq s (Set.iUnion fun n => t n)
      ⊢ Monotone t
    -/
  · intros m n hmn
    /-
      case h_mono
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.Integrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      s : Set β
      hs : IsClosed s
      t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
      B : Eq s (Set.iUnion fun n => t n)
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le (t m) (t n)
    -/
    simp only [t, Set.le_iff_subset]
    /-
      case h_mono
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.Integrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      s : Set β
      hs : IsClosed s
      t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
      B : Eq s (Set.iUnion fun n => t n)
      m n : Nat
      hmn : LE.le m n
      ⊢ HasSubset.Subset (Inter.inter (closure (compactCovering β m)) s) (Inter.inte …
    -/
    gcongr
    /-
      🎉 no goals
    -/
    /-
      case hfi
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.Integrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      s : Set β
      hs : IsClosed s
      t : Nat → Set β := fun n => Inter.inter (closure (compactCovering β n)) s
      B : Eq s (Set.iUnion fun n => t n)
      ⊢ MeasureTheory.IntegrableOn f (Set.iUnion fun n => t n) μ
    -/
  · exact hf.integrableOn
    /-
      🎉 no goals
    -/


/-- If a locally integrable function has zero integral on all compact sets in a sigma-compact space,
then it is zero almost everywhere. -/
lemma ae_eq_zero_of_forall_setIntegral_isCompact_eq_zero'
    [SigmaCompactSpace β] [R1Space β] {μ : Measure β} {f : β → E} (hf : LocallyIntegrable f μ)
    (h'f : ∀ (s : Set β), IsCompact s → ∫ x in s, f x ∂μ = 0) :
    f =ᵐ[μ] 0 := by
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.LocallyIntegrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f 0
  -/
  rw [← μ.restrict_univ, ← iUnion_closure_compactCovering]
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.LocallyIntegrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    ⊢ (MeasureTheory.ae (μ.restrict (Set.iUnion fun n => closure (compactCovering  …
  -/
  apply (ae_restrict_iUnion_iff _ _).2 (fun n ↦ ?_)
  /-
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : CompleteSpace E
    β : Type u_3
    inst✝⁴ : TopologicalSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    inst✝¹ : SigmaCompactSpace β
    inst✝ : R1Space β
    μ : MeasureTheory.Measure β
    f : β → E
    hf : MeasureTheory.LocallyIntegrable f μ
    h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
    n : Nat
    ⊢ Filter.Eventually (fun x => Eq (f x) (0 x)) (MeasureTheory.ae (μ.restrict (c …
  -/
  apply ae_eq_zero_of_forall_setIntegral_isCompact_eq_zero
    /-
      case hf
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.LocallyIntegrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      n : Nat
      ⊢ MeasureTheory.Integrable f (μ.restrict (closure (compactCovering β n)))
    -/
  · exact hf.integrableOn_isCompact (isCompact_compactCovering β n).closure
    /-
      🎉 no goals
    -/
    /-
      case h'f
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.LocallyIntegrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      n : Nat
      ⊢ ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral ((μ.restrict (closur …
    -/
  · intro s hs
    /-
      case h'f
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.LocallyIntegrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      n : Nat
      s : Set β
      hs : IsCompact s
      ⊢ Eq (MeasureTheory.integral ((μ.restrict (closure (compactCovering β n))).res …
    -/
    rw [Measure.restrict_restrict' measurableSet_closure]
    /-
      case h'f
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      β : Type u_3
      inst✝⁴ : TopologicalSpace β
      inst✝³ : MeasurableSpace β
      inst✝² : BorelSpace β
      inst✝¹ : SigmaCompactSpace β
      inst✝ : R1Space β
      μ : MeasureTheory.Measure β
      f : β → E
      hf : MeasureTheory.LocallyIntegrable f μ
      h'f : ∀ (s : Set β), IsCompact s → Eq (MeasureTheory.integral (μ.restrict s) f …
      n : Nat
      s : Set β
      hs : IsCompact s
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Inter.inter s (closure (compactCover …
    -/
    exact h'f _ (hs.inter_right isClosed_closure)
    /-
      🎉 no goals
    -/


