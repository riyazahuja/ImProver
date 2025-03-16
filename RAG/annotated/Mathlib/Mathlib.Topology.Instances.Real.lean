instance : NoncompactSpace ℝ := Int.isClosedEmbedding_coe_real.noncompactSpace


theorem Real.uniformContinuous_add : UniformContinuous fun p : ℝ × ℝ => p.1 + p.2 :=
  Metric.uniformContinuous_iff.2 fun _ε ε0 =>
    let ⟨δ, δ0, Hδ⟩ := rat_add_continuous_lemma abs ε0
    ⟨δ, δ0, fun _ _ h =>
      let ⟨h₁, h₂⟩ := max_lt_iff.1 h
      Hδ h₁ h₂⟩


theorem Real.uniformContinuous_neg : UniformContinuous (@Neg.neg ℝ _) :=
  Metric.uniformContinuous_iff.2 fun ε ε0 =>
                            /-
                              ε : Real
                              ε0 : GT.gt ε 0
                              x✝¹ x✝ : Real
                              h : LT.lt (Dist.dist x✝¹ x✝) ε
                              ⊢ LT.lt (Dist.dist (Neg.neg x✝¹) (Neg.neg x✝)) ε
                            -/
    ⟨_, ε0, fun _ _ h => by simpa only [abs_sub_comm, Real.dist_eq, neg_sub_neg] using h⟩
                            /-
                              🎉 no goals
                            -/


instance : ContinuousStar ℝ := ⟨continuous_id⟩


instance : UniformAddGroup ℝ :=
  UniformAddGroup.mk' Real.uniformContinuous_add Real.uniformContinuous_neg

-- short-circuit type class inference

                                       /-
                                         α : Type u
                                         β : Type v
                                         γ : Type w
                                         ⊢ TopologicalAddGroup Real
                                       -/
instance : TopologicalAddGroup ℝ := by infer_instance
                                       /-
                                         🎉 no goals
                                       -/

instance : TopologicalRing ℝ := inferInstance

instance : TopologicalDivisionRing ℝ := inferInstance


instance : ProperSpace ℝ where
  isCompact_closedBall x r := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      x r : Real
      ⊢ IsCompact (Metric.closedBall x r)
    -/
    rw [Real.closedBall_eq_Icc]
    /-
      α : Type u
      β : Type v
      γ : Type w
      x r : Real
      ⊢ IsCompact (Set.Icc (HSub.hSub x r) (HAdd.hAdd x r))
    -/
    apply isCompact_Icc
    /-
      🎉 no goals
    -/


instance : SecondCountableTopology ℝ := secondCountable_of_proper


theorem Real.isTopologicalBasis_Ioo_rat :
    @IsTopologicalBasis ℝ _ (⋃ (a : ℚ) (b : ℚ) (_ : a < b), {Ioo (a : ℝ) b}) :=
                                           /-
                                             ⊢ ∀ (u : Set Real), Membership.mem (Set.iUnion fun a => Set.iUnion fun b => Se …
                                           -/
  isTopologicalBasis_of_isOpen_of_nhds (by simp +contextual [isOpen_Ioo])
                                           /-
                                             🎉 no goals
                                           -/
    fun a _ hav hv =>
    let ⟨_, _, ⟨hl, hu⟩, h⟩ := mem_nhds_iff_exists_Ioo_subset.mp (IsOpen.mem_nhds hv hav)
    let ⟨q, hlq, hqa⟩ := exists_rat_btwn hl
    let ⟨p, hap, hpu⟩ := exists_rat_btwn hu
    ⟨Ioo q p, by
      /-
        a : Real
        x✝ : Set Real
        hav : Membership.mem x✝ a
        hv : IsOpen x✝
        w✝¹ w✝ : Real
        hl : LT.lt w✝¹ a
        hu : LT.lt a w✝
        h : HasSubset.Subset (Set.Ioo w✝¹ w✝) x✝
        q : Rat
        hlq : LT.lt w✝¹ ↑q
        hqa : LT.lt (↑q) a
        p : Rat
        hap : LT.lt a ↑p
        hpu : LT.lt (↑p) w✝
        ⊢ Membership.mem (Set.iUnion fun a => Set.iUnion fun b => Set.iUnion fun x =>  …
      -/
      simp only [mem_iUnion]
      /-
        a : Real
        x✝ : Set Real
        hav : Membership.mem x✝ a
        hv : IsOpen x✝
        w✝¹ w✝ : Real
        hl : LT.lt w✝¹ a
        hu : LT.lt a w✝
        h : HasSubset.Subset (Set.Ioo w✝¹ w✝) x✝
        q : Rat
        hlq : LT.lt w✝¹ ↑q
        hqa : LT.lt (↑q) a
        p : Rat
        hap : LT.lt a ↑p
        hpu : LT.lt (↑p) w✝
        ⊢ Exists fun i => Exists fun i_1 => Exists fun i_2 => Membership.mem (Singleto …
      -/
      exact ⟨q, p, Rat.cast_lt.1 <| hqa.trans hap, rfl⟩, ⟨hqa, hap⟩, fun _ ⟨hqa', ha'p⟩ =>
      /-
        🎉 no goals
      -/
      h ⟨hlq.trans hqa', ha'p.trans hpu⟩⟩


@[simp]
theorem Real.cobounded_eq : cobounded ℝ = atBot ⊔ atTop := by
  /-
    ⊢ Eq (Bornology.cobounded Real) (Max.max Filter.atBot Filter.atTop)
  -/
  simp only [← comap_dist_right_atTop (0 : ℝ), Real.dist_eq, sub_zero, comap_abs_atTop]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-07")] alias Real.cocompact_eq := cocompact_eq_atBot_atTop


@[deprecated (since := "2024-02-07")] alias Real.atBot_le_cocompact := atBot_le_cocompact

@[deprecated (since := "2024-02-07")] alias Real.atTop_le_cocompact := atTop_le_cocompact

/- TODO(Mario): Prove that these are uniform isomorphisms instead of uniform embeddings
lemma uniform_embedding_add_rat {r : ℚ} : uniform_embedding (fun p : ℚ => p + r) :=
_

lemma uniform_embedding_mul_rat {q : ℚ} (hq : q ≠ 0) : uniform_embedding ((*) q) :=
_ -/

theorem Real.mem_closure_iff {s : Set ℝ} {x : ℝ} :
    x ∈ closure s ↔ ∀ ε > 0, ∃ y ∈ s, |y - x| < ε := by
  /-
    s : Set Real
    x : Real
    ⊢ Iff (Membership.mem (closure s) x) (∀ (ε : Real), GT.gt ε 0 → Exists fun y = …
  -/
  simp [mem_closure_iff_nhds_basis nhds_basis_ball, Real.dist_eq]
  /-
    🎉 no goals
  -/


theorem Real.uniformContinuous_inv (s : Set ℝ) {r : ℝ} (r0 : 0 < r) (H : ∀ x ∈ s, r ≤ |x|) :
    UniformContinuous fun p : s => p.1⁻¹ :=
  Metric.uniformContinuous_iff.2 fun _ε ε0 =>
    let ⟨δ, δ0, Hδ⟩ := rat_inv_continuous_lemma abs ε0 r0
    ⟨δ, δ0, fun {a b} h => Hδ (H _ a.2) (H _ b.2) h⟩


theorem Real.uniformContinuous_abs : UniformContinuous (abs : ℝ → ℝ) :=
  Metric.uniformContinuous_iff.2 fun ε ε0 =>
    ⟨ε, ε0, fun _ _ ↦ lt_of_le_of_lt (abs_abs_sub_abs_le_abs_sub _ _)⟩


theorem Real.continuous_inv : Continuous fun a : { r : ℝ // r ≠ 0 } => a.val⁻¹ :=
  continuousOn_inv₀.restrict


theorem Real.uniformContinuous_const_mul {x : ℝ} : UniformContinuous (x * ·) :=
  uniformContinuous_const_smul x


theorem Real.uniformContinuous_mul (s : Set (ℝ × ℝ)) {r₁ r₂ : ℝ}
    (H : ∀ x ∈ s, |(x : ℝ × ℝ).1| < r₁ ∧ |x.2| < r₂) :
    UniformContinuous fun p : s => p.1.1 * p.1.2 :=
  Metric.uniformContinuous_iff.2 fun _ε ε0 =>
    let ⟨δ, δ0, Hδ⟩ := rat_mul_continuous_lemma abs ε0
    ⟨δ, δ0, fun {a b} h =>
      let ⟨h₁, h₂⟩ := max_lt_iff.1 h
      Hδ (H _ a.2).1 (H _ b.2).2 h₁ h₂⟩

-- Porting note: moved `TopologicalRing` instance up


instance Real.instCompleteSpace : CompleteSpace ℝ := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    ⊢ CompleteSpace Real
  -/
  apply complete_of_cauchySeq_tendsto
  /-
    case a
    α : Type u
    β : Type v
    γ : Type w
    ⊢ ∀ (u : Nat → Real), CauchySeq u → Exists fun a => Filter.Tendsto u Filter.at …
  -/
  intro u hu
  /-
    case a
    α : Type u
    β : Type v
    γ : Type w
    u : Nat → Real
    hu : CauchySeq u
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  let c : CauSeq ℝ abs := ⟨u, Metric.cauchySeq_iff'.1 hu⟩
  /-
    case a
    α : Type u
    β : Type v
    γ : Type w
    u : Nat → Real
    hu : CauchySeq u
    c : CauSeq Real abs := ⟨u, ⋯⟩
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  refine ⟨c.lim, fun s h => ?_⟩
  /-
    case a
    α : Type u
    β : Type v
    γ : Type w
    u : Nat → Real
    hu : CauchySeq u
    c : CauSeq Real abs := ⟨u, ⋯⟩
    s : Set Real
    h : Membership.mem (nhds c.lim) s
    ⊢ Membership.mem (Filter.map u Filter.atTop) s
  -/
  rcases Metric.mem_nhds_iff.1 h with ⟨ε, ε0, hε⟩
  /-
    case a.intro.intro
    α : Type u
    β : Type v
    γ : Type w
    u : Nat → Real
    hu : CauchySeq u
    c : CauSeq Real abs := ⟨u, ⋯⟩
    s : Set Real
    h : Membership.mem (nhds c.lim) s
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball c.lim ε) s
    ⊢ Membership.mem (Filter.map u Filter.atTop) s
  -/
  have := c.equiv_lim ε ε0
  /-
    case a.intro.intro
    α : Type u
    β : Type v
    γ : Type w
    u : Nat → Real
    hu : CauchySeq u
    c : CauSeq Real abs := ⟨u, ⋯⟩
    s : Set Real
    h : Membership.mem (nhds c.lim) s
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball c.lim ε) s
    this : Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub c (Cau …
    ⊢ Membership.mem (Filter.map u Filter.atTop) s
  -/
  simp only [mem_map, mem_atTop_sets, mem_setOf_eq]
  /-
    case a.intro.intro
    α : Type u
    β : Type v
    γ : Type w
    u : Nat → Real
    hu : CauchySeq u
    c : CauSeq Real abs := ⟨u, ⋯⟩
    s : Set Real
    h : Membership.mem (nhds c.lim) s
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball c.lim ε) s
    this : Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abs (↑(HSub.hSub c (Cau …
    ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem (Set.preimage u s) b
  -/
  exact this.imp fun N hN n hn => hε (hN n hn)
  /-
    🎉 no goals
  -/


theorem Real.totallyBounded_ball (x ε : ℝ) : TotallyBounded (ball x ε) := by
  /-
    x ε : Real
    ⊢ TotallyBounded (Metric.ball x ε)
  -/
  rw [Real.ball_eq_Ioo]; apply totallyBounded_Ioo
                         /-
                           🎉 no goals
                         -/


theorem Real.subfield_eq_of_closed {K : Subfield ℝ} (hc : IsClosed (K : Set ℝ)) : K = ⊤ := by
  /-
    K : Subfield Real
    hc : IsClosed ↑K
    ⊢ Eq K Top.top
  -/
  rw [SetLike.ext'_iff, Subfield.coe_top, ← hc.closure_eq]
  /-
    K : Subfield Real
    hc : IsClosed ↑K
    ⊢ Eq (closure ↑K) Set.univ
  -/
  refine Rat.denseRange_cast.mono ?_ |>.closure_eq
  /-
    K : Subfield Real
    hc : IsClosed ↑K
    ⊢ HasSubset.Subset (Set.range Rat.cast) ↑K
  -/
  rintro - ⟨_, rfl⟩
  /-
    case intro
    K : Subfield Real
    hc : IsClosed ↑K
    w✝ : Rat
    ⊢ Membership.mem ↑K ↑w✝
  -/
  exact SubfieldClass.ratCast_mem K _
  /-
    🎉 no goals
  -/


theorem closure_of_rat_image_lt {q : ℚ} :
    closure (((↑) : ℚ → ℝ) '' { x | q < x }) = { r | ↑q ≤ r } :=
  Subset.antisymm
    (isClosed_Ici.closure_subset_iff.2
                                                  /-
                                                    q p : Rat
                                                    h : LT.lt q p
                                                    ⊢ Membership.mem (Set.preimage Rat.cast (Set.Ici ↑q)) p
                                                  -/
      (image_subset_iff.2 fun p (h : q < p) => by simpa using h.le))
                                                  /-
                                                    🎉 no goals
                                                  -/
    fun x hx => mem_closure_iff_nhds.2 fun _ ht =>
      let ⟨ε, ε0, hε⟩ := Metric.mem_nhds_iff.1 ht
      let ⟨p, h₁, h₂⟩ := exists_rat_btwn ((lt_add_iff_pos_right x).2 ε0)
                   /-
                     q : Rat
                     x : Real
                     hx : Membership.mem (setOf fun r => LE.le (↑q) r) x
                     x✝ : Set Real
                     ht : Membership.mem (nhds x) x✝
                     ε : Real
                     ε0 : GT.gt ε 0
                     hε : HasSubset.Subset (Metric.ball x ε) x✝
                     p : Rat
                     h₁ : LT.lt x ↑p
                     h₂ : LT.lt (↑p) (HAdd.hAdd x ε)
                     ⊢ Membership.mem (Metric.ball x ε) ↑p
                   -/
      ⟨p, hε <| by rwa [mem_ball, Real.dist_eq, abs_of_pos (sub_pos.2 h₁), sub_lt_iff_lt_add'],
                   /-
                     🎉 no goals
                   -/
        mem_image_of_mem _ <| Rat.cast_lt.1 <| lt_of_le_of_lt hx.out h₁⟩

/- TODO(Mario): Put these back only if needed later
lemma closure_of_rat_image_le_eq {q : ℚ} : closure ((coe : ℚ → ℝ) '' {x | q ≤ x}) = {r | ↑q ≤ r} :=
  _

lemma closure_of_rat_image_le_le_eq {a b : ℚ} (hab : a ≤ b) :
    closure (of_rat '' {q:ℚ | a ≤ q ∧ q ≤ b}) = {r:ℝ | of_rat a ≤ r ∧ r ≤ of_rat b} :=
  _
-/


instance instIsOrderBornology : IsOrderBornology ℝ where
  isBounded_iff_bddBelow_bddAbove s := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      s : Set Real
      ⊢ Iff (Bornology.IsBounded s) (And (BddBelow s) (BddAbove s))
    -/
    refine ⟨fun bdd ↦ ?_, fun h ↦ isBounded_of_bddAbove_of_bddBelow h.2 h.1⟩
    obtain ⟨r, hr⟩ : ∃ r : ℝ, s ⊆ Icc (-r) r := by
      simpa [Real.closedBall_eq_Icc] using bdd.subset_closedBall 0
    /-
      case intro
      α : Type u
      β : Type v
      γ : Type w
      s : Set Real
      bdd : Bornology.IsBounded s
      r : Real
      hr : HasSubset.Subset s (Set.Icc (Neg.neg r) r)
      ⊢ And (BddBelow s) (BddAbove s)
    -/
    exact ⟨bddBelow_Icc.mono hr, bddAbove_Icc.mono hr⟩
    /-
      🎉 no goals
    -/


/-- A continuous, periodic function has compact range. -/
theorem Periodic.compact_of_continuous [TopologicalSpace α] {f : ℝ → α} {c : ℝ} (hp : Periodic f c)
    (hc : c ≠ 0) (hf : Continuous f) : IsCompact (range f) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    f : Real → α
    c : Real
    hp : Function.Periodic f c
    hc : Ne c 0
    hf : Continuous f
    ⊢ IsCompact (Set.range f)
  -/
  rw [← hp.image_uIcc hc 0]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    f : Real → α
    c : Real
    hp : Function.Periodic f c
    hc : Ne c 0
    hf : Continuous f
    ⊢ IsCompact (Set.image f (Set.uIcc 0 (HAdd.hAdd 0 c)))
  -/
  exact isCompact_uIcc.image hf
  /-
    🎉 no goals
  -/


/-- A continuous, periodic function is bounded. -/
theorem Periodic.isBounded_of_continuous [PseudoMetricSpace α] {f : ℝ → α} {c : ℝ}
    (hp : Periodic f c) (hc : c ≠ 0) (hf : Continuous f) : IsBounded (range f) :=
  (hp.compact_of_continuous hc hf).isBounded


