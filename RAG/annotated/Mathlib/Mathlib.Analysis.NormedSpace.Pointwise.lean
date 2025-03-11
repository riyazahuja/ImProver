theorem ediam_smul_le (c : 𝕜) (s : Set E) : EMetric.diam (c • s) ≤ ‖c‖₊ • EMetric.diam s :=
  (lipschitzWith_smul c).ediam_image_le s


theorem ediam_smul₀ (c : 𝕜) (s : Set E) : EMetric.diam (c • s) = ‖c‖₊ • EMetric.diam s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    s : Set E
    ⊢ Eq (EMetric.diam (HSMul.hSMul c s)) (HSMul.hSMul (NNNorm.nnnorm c) (EMetric. …
  -/
  refine le_antisymm (ediam_smul_le c s) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    s : Set E
    ⊢ LE.le (HSMul.hSMul (NNNorm.nnnorm c) (EMetric.diam s)) (EMetric.diam (HSMul. …
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      s : Set E
      ⊢ LE.le (HSMul.hSMul (NNNorm.nnnorm 0) (EMetric.diam s)) (EMetric.diam (HSMul. …
    -/
  · obtain rfl | hs := s.eq_empty_or_nonempty
      /-
        case inl.inl
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NormedDivisionRing 𝕜
        inst✝² : SeminormedAddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : BoundedSMul 𝕜 E
        ⊢ LE.le (HSMul.hSMul (NNNorm.nnnorm 0) (EMetric.diam EmptyCollection.emptyColl …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      s : Set E
      hs : s.Nonempty
      ⊢ LE.le (HSMul.hSMul (NNNorm.nnnorm 0) (EMetric.diam s)) (EMetric.diam (HSMul. …
    -/
    simp [zero_smul_set hs, ← Set.singleton_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      s : Set E
      hc : Ne c 0
      ⊢ LE.le (HSMul.hSMul (NNNorm.nnnorm c) (EMetric.diam s)) (EMetric.diam (HSMul. …
    -/
  · have := (lipschitzWith_smul c⁻¹).ediam_image_le (c • s)
    rwa [← smul_eq_mul, ← ENNReal.smul_def, Set.image_smul, inv_smul_smul₀ hc s, nnnorm_inv,
      le_inv_smul_iff_of_pos (nnnorm_pos.2 hc)] at this


theorem diam_smul₀ (c : 𝕜) (x : Set E) : diam (c • x) = ‖c‖ * diam x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    x : Set E
    ⊢ Eq (Metric.diam (HSMul.hSMul c x)) (HMul.hMul (Norm.norm c) (Metric.diam x))
  -/
  simp_rw [diam, ediam_smul₀, ENNReal.toReal_smul, NNReal.smul_def, coe_nnnorm, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem infEdist_smul₀ {c : 𝕜} (hc : c ≠ 0) (s : Set E) (x : E) :
    EMetric.infEdist (c • x) (c • s) = ‖c‖₊ • EMetric.infEdist x s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    hc : Ne c 0
    s : Set E
    x : E
    ⊢ Eq (EMetric.infEdist (HSMul.hSMul c x) (HSMul.hSMul c s)) (HSMul.hSMul (NNNo …
  -/
  simp_rw [EMetric.infEdist]
  have : Function.Surjective ((c • ·) : E → E) :=
    Function.RightInverse.surjective (smul_inv_smul₀ hc)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    hc : Ne c 0
    s : Set E
    x : E
    this : Function.Surjective fun x => HSMul.hSMul c x
    ⊢ Eq (iInf fun y => iInf fun h => EDist.edist (HSMul.hSMul c x) y) (HSMul.hSMu …
  -/
  trans ⨅ (y) (_ : y ∈ s), ‖c‖₊ • edist x y
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      hc : Ne c 0
      s : Set E
      x : E
      this : Function.Surjective fun x => HSMul.hSMul c x
      ⊢ Eq (iInf fun y => iInf fun h => EDist.edist (HSMul.hSMul c x) y) (iInf fun y …
    -/
  · refine (this.iInf_congr _ fun y => ?_).symm
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      hc : Ne c 0
      s : Set E
      x : E
      this : Function.Surjective fun x => HSMul.hSMul c x
      y : E
      ⊢ Eq (iInf fun h => EDist.edist (HSMul.hSMul c x) (HSMul.hSMul c y)) (iInf fun …
    -/
    simp_rw [smul_mem_smul_set_iff₀ hc, edist_smul₀]
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      hc : Ne c 0
      s : Set E
      x : E
      this : Function.Surjective fun x => HSMul.hSMul c x
      ⊢ Eq (iInf fun y => iInf fun x_1 => HSMul.hSMul (NNNorm.nnnorm c) (EDist.edist …
    -/
  · have : (‖c‖₊ : ENNReal) ≠ 0 := by simp [hc]
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      hc : Ne c 0
      s : Set E
      x : E
      this✝ : Function.Surjective fun x => HSMul.hSMul c x
      this : Ne (↑(NNNorm.nnnorm c)) 0
      ⊢ Eq (iInf fun y => iInf fun x_1 => HSMul.hSMul (NNNorm.nnnorm c) (EDist.edist …
    -/
    simp_rw [ENNReal.smul_def, smul_eq_mul, ENNReal.mul_iInf_of_ne this ENNReal.coe_ne_top]
    /-
      🎉 no goals
    -/


theorem infDist_smul₀ {c : 𝕜} (hc : c ≠ 0) (s : Set E) (x : E) :
    Metric.infDist (c • x) (c • s) = ‖c‖ * Metric.infDist x s := by
  simp_rw [Metric.infDist, infEdist_smul₀ hc s, ENNReal.toReal_smul, NNReal.smul_def, coe_nnnorm,
    smul_eq_mul]


theorem smul_ball {c : 𝕜} (hc : c ≠ 0) (x : E) (r : ℝ) : c • ball x r = ball (c • x) (‖c‖ * r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    ⊢ Eq (HSMul.hSMul c (Metric.ball x r)) (Metric.ball (HSMul.hSMul c x) (HMul.hM …
  -/
  ext y
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem (HSMul.hSMul c (Metric.ball x r)) y) (Membership.mem (Me …
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ hc]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem (Metric.ball x r) (HSMul.hSMul (Inv.inv c) y)) (Membersh …
  -/
  conv_lhs => rw [← inv_smul_smul₀ hc x]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem (Metric.ball (HSMul.hSMul (Inv.inv c) (HSMul.hSMul c x)) …
  -/
  simp [← div_eq_inv_mul, div_lt_iff₀ (norm_pos_iff.2 hc), mul_comm _ r, dist_smul₀]
  /-
    🎉 no goals
  -/


theorem smul_unitBall {c : 𝕜} (hc : c ≠ 0) : c • ball (0 : E) (1 : ℝ) = ball (0 : E) ‖c‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    ⊢ Eq (HSMul.hSMul c (Metric.ball 0 1)) (Metric.ball 0 (Norm.norm c))
  -/
  rw [_root_.smul_ball hc, smul_zero, mul_one]
  /-
    🎉 no goals
  -/


theorem smul_sphere' {c : 𝕜} (hc : c ≠ 0) (x : E) (r : ℝ) :
    c • sphere x r = sphere (c • x) (‖c‖ * r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    ⊢ Eq (HSMul.hSMul c (Metric.sphere x r)) (Metric.sphere (HSMul.hSMul c x) (HMu …
  -/
  ext y
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem (HSMul.hSMul c (Metric.sphere x r)) y) (Membership.mem ( …
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ hc]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem (Metric.sphere x r) (HSMul.hSMul (Inv.inv c) y)) (Member …
  -/
  conv_lhs => rw [← inv_smul_smul₀ hc x]
  simp only [mem_sphere, dist_smul₀, norm_inv, ← div_eq_inv_mul, div_eq_iff (norm_pos_iff.2 hc).ne',
    mul_comm r]


theorem smul_closedBall' {c : 𝕜} (hc : c ≠ 0) (x : E) (r : ℝ) :
    c • closedBall x r = closedBall (c • x) (‖c‖ * r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : Ne c 0
    x : E
    r : Real
    ⊢ Eq (HSMul.hSMul c (Metric.closedBall x r)) (Metric.closedBall (HSMul.hSMul c …
  -/
  simp only [← ball_union_sphere, Set.smul_set_union, _root_.smul_ball hc, smul_sphere' hc]
  /-
    🎉 no goals
  -/


theorem set_smul_sphere_zero {s : Set 𝕜} (hs : 0 ∉ s) (r : ℝ) :
    s • sphere (0 : E) r = (‖·‖) ⁻¹' ((‖·‖ * r) '' s) :=
  calc
    s • sphere (0 : E) r = ⋃ c ∈ s, c • sphere (0 : E) r := iUnion_smul_left_image.symm
    _ = ⋃ c ∈ s, sphere (0 : E) (‖c‖ * r) := iUnion₂_congr fun c hc ↦ by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NormedField 𝕜
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        s : Set 𝕜
        hs : Not (Membership.mem s 0)
        r : Real
        c : 𝕜
        hc : Membership.mem s c
        ⊢ Eq (HSMul.hSMul c (Metric.sphere 0 r)) (Metric.sphere 0 (HMul.hMul (Norm.nor …
      -/
      rw [smul_sphere' (ne_of_mem_of_not_mem hc hs), smul_zero]
      /-
        🎉 no goals
      -/
                                         /-
                                           𝕜 : Type u_1
                                           E : Type u_2
                                           inst✝² : NormedField 𝕜
                                           inst✝¹ : SeminormedAddCommGroup E
                                           inst✝ : NormedSpace 𝕜 E
                                           s : Set 𝕜
                                           hs : Not (Membership.mem s 0)
                                           r : Real
                                           ⊢ Eq (Set.iUnion fun c => Set.iUnion fun h => Metric.sphere 0 (HMul.hMul (Norm …
                                         -/
    _ = (‖·‖) ⁻¹' ((‖·‖ * r) '' s) := by ext; simp [eq_comm]
                                              /-
                                                🎉 no goals
                                              -/


/-- Image of a bounded set in a normed space under scalar multiplication by a constant is
bounded. See also `Bornology.IsBounded.smul` for a similar lemma about an isometric action. -/
theorem Bornology.IsBounded.smul₀ {s : Set E} (hs : IsBounded s) (c : 𝕜) : IsBounded (c • s) :=
  (lipschitzWith_smul c).isBounded_image hs


/-- If `s` is a bounded set, then for small enough `r`, the set `{x} + r • s` is contained in any
fixed neighborhood of `x`. -/
theorem eventually_singleton_add_smul_subset {x : E} {s : Set E} (hs : Bornology.IsBounded s)
    {u : Set E} (hu : u ∈ 𝓝 x) : ∀ᶠ r in 𝓝 (0 : 𝕜), {x} + r • s ⊆ u := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (HAdd.hAdd (Singleton.singleton …
  -/
  obtain ⟨ε, εpos, hε⟩ : ∃ ε : ℝ, 0 < ε ∧ closedBall x ε ⊆ u := nhds_basis_closedBall.mem_iff.1 hu
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (HAdd.hAdd (Singleton.singleton …
  -/
  obtain ⟨R, Rpos, hR⟩ : ∃ R : ℝ, 0 < R ∧ s ⊆ closedBall 0 R := hs.subset_closedBall_lt 0 0
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (HAdd.hAdd (Singleton.singleton …
  -/
  have : Metric.closedBall (0 : 𝕜) (ε / R) ∈ 𝓝 (0 : 𝕜) := closedBall_mem_nhds _ (div_pos εpos Rpos)
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    ⊢ Filter.Eventually (fun r => HasSubset.Subset (HAdd.hAdd (Singleton.singleton …
  -/
  filter_upwards [this] with r hr
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    r : 𝕜
    hr : Membership.mem (Metric.closedBall 0 (HDiv.hDiv ε R)) r
    ⊢ HasSubset.Subset (HAdd.hAdd (Singleton.singleton x) (HSMul.hSMul r s)) u
  -/
  simp only [image_add_left, singleton_add]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    r : 𝕜
    hr : Membership.mem (Metric.closedBall 0 (HDiv.hDiv ε R)) r
    ⊢ HasSubset.Subset (Set.preimage (fun x_1 => HAdd.hAdd (Neg.neg x) x_1) (HSMul …
  -/
  intro y hy
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    r : 𝕜
    hr : Membership.mem (Metric.closedBall 0 (HDiv.hDiv ε R)) r
    y : E
    hy : Membership.mem (Set.preimage (fun x_1 => HAdd.hAdd (Neg.neg x) x_1) (HSMu …
    ⊢ Membership.mem u y
  -/
  obtain ⟨z, zs, hz⟩ : ∃ z : E, z ∈ s ∧ r • z = -x + y := by simpa [mem_smul_set] using hy
  have I : ‖r • z‖ ≤ ε :=
    calc
      ‖r • z‖ = ‖r‖ * ‖z‖ := norm_smul _ _
      _ ≤ ε / R * R :=
        (mul_le_mul (mem_closedBall_zero_iff.1 hr) (mem_closedBall_zero_iff.1 (hR zs))
          (norm_nonneg _) (div_pos εpos Rpos).le)
      _ = ε := by field_simp
  /-
    case h.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    r : 𝕜
    hr : Membership.mem (Metric.closedBall 0 (HDiv.hDiv ε R)) r
    y : E
    hy : Membership.mem (Set.preimage (fun x_1 => HAdd.hAdd (Neg.neg x) x_1) (HSMu …
    z : E
    zs : Membership.mem s z
    hz : Eq (HSMul.hSMul r z) (HAdd.hAdd (Neg.neg x) y)
    I : LE.le (Norm.norm (HSMul.hSMul r z)) ε
    ⊢ Membership.mem u y
  -/
  have : y = x + r • z := by simp only [hz, add_neg_cancel_left]
  /-
    case h.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this✝ : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    r : 𝕜
    hr : Membership.mem (Metric.closedBall 0 (HDiv.hDiv ε R)) r
    y : E
    hy : Membership.mem (Set.preimage (fun x_1 => HAdd.hAdd (Neg.neg x) x_1) (HSMu …
    z : E
    zs : Membership.mem s z
    hz : Eq (HSMul.hSMul r z) (HAdd.hAdd (Neg.neg x) y)
    I : LE.le (Norm.norm (HSMul.hSMul r z)) ε
    this : Eq y (HAdd.hAdd x (HSMul.hSMul r z))
    ⊢ Membership.mem u y
  -/
  apply hε
  /-
    case h.intro.intro.a
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    hs : Bornology.IsBounded s
    u : Set E
    hu : Membership.mem (nhds x) u
    ε : Real
    εpos : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) u
    R : Real
    Rpos : LT.lt 0 R
    hR : HasSubset.Subset s (Metric.closedBall 0 R)
    this✝ : Membership.mem (nhds 0) (Metric.closedBall 0 (HDiv.hDiv ε R))
    r : 𝕜
    hr : Membership.mem (Metric.closedBall 0 (HDiv.hDiv ε R)) r
    y : E
    hy : Membership.mem (Set.preimage (fun x_1 => HAdd.hAdd (Neg.neg x) x_1) (HSMu …
    z : E
    zs : Membership.mem s z
    hz : Eq (HSMul.hSMul r z) (HAdd.hAdd (Neg.neg x) y)
    I : LE.le (Norm.norm (HSMul.hSMul r z)) ε
    this : Eq y (HAdd.hAdd x (HSMul.hSMul r z))
    ⊢ Membership.mem (Metric.closedBall x ε) y
  -/
  simpa only [this, dist_eq_norm, add_sub_cancel_left, mem_closedBall] using I
  /-
    🎉 no goals
  -/


/-- In a real normed space, the image of the unit ball under scalar multiplication by a positive
constant `r` is the ball of radius `r`. -/
theorem smul_unitBall_of_pos {r : ℝ} (hr : 0 < r) : r • ball (0 : E) 1 = ball (0 : E) r := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (HSMul.hSMul r (Metric.ball 0 1)) (Metric.ball 0 r)
  -/
  rw [smul_unitBall hr.ne', Real.norm_of_nonneg hr.le]
  /-
    🎉 no goals
  -/


lemma Ioo_smul_sphere_zero {a b r : ℝ} (ha : 0 ≤ a) (hr : 0 < r) :
    Ioo a b • sphere (0 : E) r = ball 0 (b * r) \ closedBall 0 (a * r) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b r : Real
    ha : LE.le 0 a
    hr : LT.lt 0 r
    ⊢ Eq (HSMul.hSMul (Set.Ioo a b) (Metric.sphere 0 r)) (SDiff.sdiff (Metric.ball …
  -/
  have : EqOn (‖·‖) id (Ioo a b) := fun x hx ↦ abs_of_pos (ha.trans_lt hx.1)
  rw [set_smul_sphere_zero (by simp [ha.not_lt]), ← image_image (· * r), this.image_eq, image_id,
    image_mul_right_Ioo _ _ hr]
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b r : Real
    ha : LE.le 0 a
    hr : LT.lt 0 r
    this : Set.EqOn (fun x => Norm.norm x) id (Set.Ioo a b)
    ⊢ Eq (Set.preimage (fun x => Norm.norm x) (Set.Ioo (HMul.hMul a r) (HMul.hMul  …
  -/
  ext x; simp [and_comm]
         /-
           🎉 no goals
         -/

-- This is also true for `ℚ`-normed spaces

theorem exists_dist_eq (x z : E) {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    ∃ y, dist x y = b * dist x z ∧ dist y z = a * dist x z := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Exists fun y => And (Eq (Dist.dist x y) (HMul.hMul b (Dist.dist x z))) (Eq ( …
  -/
  use a • x + b • z
  /-
    case h
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ And (Eq (Dist.dist x (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b z))) (HMul. …
  -/
  nth_rw 1 [← one_smul ℝ x]
  /-
    case h
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ And (Eq (Dist.dist (HSMul.hSMul 1 x) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  nth_rw 4 [← one_smul ℝ z]
  /-
    case h
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ And (Eq (Dist.dist (HSMul.hSMul 1 x) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSM …
  -/
  simp [dist_eq_norm, ← hab, add_smul, ← smul_sub, norm_smul_of_nonneg, ha, hb]
  /-
    🎉 no goals
  -/


theorem exists_dist_le_le (hδ : 0 ≤ δ) (hε : 0 ≤ ε) (h : dist x z ≤ ε + δ) :
    ∃ y, dist x y ≤ δ ∧ dist y z ≤ ε := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : LE.le (Dist.dist x z) (HAdd.hAdd ε δ)
    ⊢ Exists fun y => And (LE.le (Dist.dist x y) δ) (LE.le (Dist.dist y z) ε)
  -/
  obtain rfl | hε' := hε.eq_or_lt
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x z : E
      δ : Real
      hδ : LE.le 0 δ
      hε : LE.le 0 0
      h : LE.le (Dist.dist x z) (HAdd.hAdd 0 δ)
      ⊢ Exists fun y => And (LE.le (Dist.dist x y) δ) (LE.le (Dist.dist y z) 0)
    -/
  · exact ⟨z, by rwa [zero_add] at h, (dist_self _).le⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : LE.le (Dist.dist x z) (HAdd.hAdd ε δ)
    hε' : LT.lt 0 ε
    ⊢ Exists fun y => And (LE.le (Dist.dist x y) δ) (LE.le (Dist.dist y z) ε)
  -/
  have hεδ := add_pos_of_pos_of_nonneg hε' hδ
  refine (exists_dist_eq x z (div_nonneg hε <| add_nonneg hε hδ)
    (div_nonneg hδ <| add_nonneg hε hδ) <| by
      rw [← add_div, div_self hεδ.ne']).imp
    fun y hy => ?_
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : LE.le (Dist.dist x z) (HAdd.hAdd ε δ)
    hε' : LT.lt 0 ε
    hεδ : LT.lt 0 (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LE.le (Dist.dist x y) δ) (LE.le (Dist.dist y z) ε)
  -/
  rw [hy.1, hy.2, div_mul_comm, div_mul_comm ε]
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : LE.le (Dist.dist x z) (HAdd.hAdd ε δ)
    hε' : LT.lt 0 ε
    hεδ : LT.lt 0 (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LE.le (HMul.hMul (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) δ) δ) (LE. …
  -/
  rw [← div_le_one hεδ] at h
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : LE.le (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) 1
    hε' : LT.lt 0 ε
    hεδ : LT.lt 0 (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LE.le (HMul.hMul (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) δ) δ) (LE. …
  -/
  exact ⟨mul_le_of_le_one_left hδ h, mul_le_of_le_one_left hε h⟩
  /-
    🎉 no goals
  -/

-- This is also true for `ℚ`-normed spaces

theorem exists_dist_le_lt (hδ : 0 ≤ δ) (hε : 0 < ε) (h : dist x z < ε + δ) :
    ∃ y, dist x y ≤ δ ∧ dist y z < ε := by
  refine (exists_dist_eq x z (div_nonneg hε.le <| add_nonneg hε.le hδ)
    (div_nonneg hδ <| add_nonneg hε.le hδ) <| by
      rw [← add_div, div_self (add_pos_of_pos_of_nonneg hε hδ).ne']).imp
    fun y hy => ?_
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LT.lt 0 ε
    h : LT.lt (Dist.dist x z) (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LE.le (Dist.dist x y) δ) (LT.lt (Dist.dist y z) ε)
  -/
  rw [hy.1, hy.2, div_mul_comm, div_mul_comm ε]
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LT.lt 0 ε
    h : LT.lt (Dist.dist x z) (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LE.le (HMul.hMul (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) δ) δ) (LT. …
  -/
  rw [← div_lt_one (add_pos_of_pos_of_nonneg hε hδ)] at h
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LT.lt 0 ε
    h : LT.lt (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) 1
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LE.le (HMul.hMul (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) δ) δ) (LT. …
  -/
  exact ⟨mul_le_of_le_one_left hδ h.le, mul_lt_of_lt_one_left hε h⟩
  /-
    🎉 no goals
  -/

-- This is also true for `ℚ`-normed spaces

theorem exists_dist_lt_le (hδ : 0 < δ) (hε : 0 ≤ ε) (h : dist x z < ε + δ) :
    ∃ y, dist x y < δ ∧ dist y z ≤ ε := by
  obtain ⟨y, yz, xy⟩ :=
    exists_dist_le_lt hε hδ (show dist z x < δ + ε by simpa only [dist_comm, add_comm] using h)
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LE.le 0 ε
    h : LT.lt (Dist.dist x z) (HAdd.hAdd ε δ)
    y : E
    yz : LE.le (Dist.dist z y) ε
    xy : LT.lt (Dist.dist y x) δ
    ⊢ Exists fun y => And (LT.lt (Dist.dist x y) δ) (LE.le (Dist.dist y z) ε)
  -/
  exact ⟨y, by simp [dist_comm x y, dist_comm y z, *]⟩
  /-
    🎉 no goals
  -/

-- This is also true for `ℚ`-normed spaces

theorem exists_dist_lt_lt (hδ : 0 < δ) (hε : 0 < ε) (h : dist x z < ε + δ) :
    ∃ y, dist x y < δ ∧ dist y z < ε := by
  refine (exists_dist_eq x z (div_nonneg hε.le <| add_nonneg hε.le hδ.le)
    (div_nonneg hδ.le <| add_nonneg hε.le hδ.le) <| by
      rw [← add_div, div_self (add_pos hε hδ).ne']).imp
    fun y hy => ?_
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : LT.lt (Dist.dist x z) (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LT.lt (Dist.dist x y) δ) (LT.lt (Dist.dist y z) ε)
  -/
  rw [hy.1, hy.2, div_mul_comm, div_mul_comm ε]
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : LT.lt (Dist.dist x z) (HAdd.hAdd ε δ)
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LT.lt (HMul.hMul (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) δ) δ) (LT. …
  -/
  rw [← div_lt_one (add_pos hε hδ)] at h
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x z : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : LT.lt (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) 1
    y : E
    hy : And (Eq (Dist.dist x y) (HMul.hMul (HDiv.hDiv δ (HAdd.hAdd ε δ)) (Dist.di …
    ⊢ And (LT.lt (HMul.hMul (HDiv.hDiv (Dist.dist x z) (HAdd.hAdd ε δ)) δ) δ) (LT. …
  -/
  exact ⟨mul_lt_of_lt_one_left hδ h, mul_lt_of_lt_one_left hε h⟩
  /-
    🎉 no goals
  -/

-- This is also true for `ℚ`-normed spaces

theorem disjoint_ball_ball_iff (hδ : 0 < δ) (hε : 0 < ε) :
    Disjoint (ball x δ) (ball y ε) ↔ δ + ε ≤ dist x y := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    ⊢ Iff (Disjoint (Metric.ball x δ) (Metric.ball y ε)) (LE.le (HAdd.hAdd δ ε) (D …
  -/
  refine ⟨fun h => le_of_not_lt fun hxy => ?_, ball_disjoint_ball⟩
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.ball y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd δ ε)
    ⊢ False
  -/
  rw [add_comm] at hxy
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.ball y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd ε δ)
    ⊢ False
  -/
  obtain ⟨z, hxz, hzy⟩ := exists_dist_lt_lt hδ hε hxy
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.ball y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd ε δ)
    z : E
    hxz : LT.lt (Dist.dist x z) δ
    hzy : LT.lt (Dist.dist z y) ε
    ⊢ False
  -/
  rw [dist_comm] at hxz
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LT.lt 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.ball y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd ε δ)
    z : E
    hxz : LT.lt (Dist.dist z x) δ
    hzy : LT.lt (Dist.dist z y) ε
    ⊢ False
  -/
  exact h.le_bot ⟨hxz, hzy⟩
  /-
    🎉 no goals
  -/

-- This is also true for `ℚ`-normed spaces

theorem disjoint_ball_closedBall_iff (hδ : 0 < δ) (hε : 0 ≤ ε) :
    Disjoint (ball x δ) (closedBall y ε) ↔ δ + ε ≤ dist x y := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LE.le 0 ε
    ⊢ Iff (Disjoint (Metric.ball x δ) (Metric.closedBall y ε)) (LE.le (HAdd.hAdd δ …
  -/
  refine ⟨fun h => le_of_not_lt fun hxy => ?_, ball_disjoint_closedBall⟩
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.closedBall y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd δ ε)
    ⊢ False
  -/
  rw [add_comm] at hxy
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.closedBall y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd ε δ)
    ⊢ False
  -/
  obtain ⟨z, hxz, hzy⟩ := exists_dist_lt_le hδ hε hxy
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.closedBall y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd ε δ)
    z : E
    hxz : LT.lt (Dist.dist x z) δ
    hzy : LE.le (Dist.dist z y) ε
    ⊢ False
  -/
  rw [dist_comm] at hxz
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LT.lt 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.ball x δ) (Metric.closedBall y ε)
    hxy : LT.lt (Dist.dist x y) (HAdd.hAdd ε δ)
    z : E
    hxz : LT.lt (Dist.dist z x) δ
    hzy : LE.le (Dist.dist z y) ε
    ⊢ False
  -/
  exact h.le_bot ⟨hxz, hzy⟩
  /-
    🎉 no goals
  -/

-- This is also true for `ℚ`-normed spaces

theorem disjoint_closedBall_ball_iff (hδ : 0 ≤ δ) (hε : 0 < ε) :
    Disjoint (closedBall x δ) (ball y ε) ↔ δ + ε ≤ dist x y := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LT.lt 0 ε
    ⊢ Iff (Disjoint (Metric.closedBall x δ) (Metric.ball y ε)) (LE.le (HAdd.hAdd δ …
  -/
  rw [disjoint_comm, disjoint_ball_closedBall_iff hε hδ, add_comm, dist_comm]
  /-
    🎉 no goals
  -/


theorem disjoint_closedBall_closedBall_iff (hδ : 0 ≤ δ) (hε : 0 ≤ ε) :
    Disjoint (closedBall x δ) (closedBall y ε) ↔ δ + ε < dist x y := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    ⊢ Iff (Disjoint (Metric.closedBall x δ) (Metric.closedBall y ε)) (LT.lt (HAdd. …
  -/
  refine ⟨fun h => lt_of_not_ge fun hxy => ?_, closedBall_disjoint_closedBall⟩
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.closedBall x δ) (Metric.closedBall y ε)
    hxy : GE.ge (HAdd.hAdd δ ε) (Dist.dist x y)
    ⊢ False
  -/
  rw [add_comm] at hxy
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.closedBall x δ) (Metric.closedBall y ε)
    hxy : GE.ge (HAdd.hAdd ε δ) (Dist.dist x y)
    ⊢ False
  -/
  obtain ⟨z, hxz, hzy⟩ := exists_dist_le_le hδ hε hxy
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.closedBall x δ) (Metric.closedBall y ε)
    hxy : GE.ge (HAdd.hAdd ε δ) (Dist.dist x y)
    z : E
    hxz : LE.le (Dist.dist x z) δ
    hzy : LE.le (Dist.dist z y) ε
    ⊢ False
  -/
  rw [dist_comm] at hxz
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    δ ε : Real
    hδ : LE.le 0 δ
    hε : LE.le 0 ε
    h : Disjoint (Metric.closedBall x δ) (Metric.closedBall y ε)
    hxy : GE.ge (HAdd.hAdd ε δ) (Dist.dist x y)
    z : E
    hxz : LE.le (Dist.dist z x) δ
    hzy : LE.le (Dist.dist z y) ε
    ⊢ False
  -/
  exact h.le_bot ⟨hxz, hzy⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem infEdist_thickening (hδ : 0 < δ) (s : Set E) (x : E) :
    infEdist x (thickening δ s) = infEdist x s - ENNReal.ofReal δ := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    ⊢ Eq (EMetric.infEdist x (Metric.thickening δ s)) (HSub.hSub (EMetric.infEdist …
  -/
  obtain hs | hs := lt_or_le (infEdist x s) (ENNReal.ofReal δ)
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ : Real
      hδ : LT.lt 0 δ
      s : Set E
      x : E
      hs : LT.lt (EMetric.infEdist x s) (ENNReal.ofReal δ)
      ⊢ Eq (EMetric.infEdist x (Metric.thickening δ s)) (HSub.hSub (EMetric.infEdist …
    -/
  · rw [infEdist_zero_of_mem, tsub_eq_zero_of_le hs.le]
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ : Real
      hδ : LT.lt 0 δ
      s : Set E
      x : E
      hs : LT.lt (EMetric.infEdist x s) (ENNReal.ofReal δ)
      ⊢ Membership.mem (Metric.thickening δ s) x
    -/
    exact hs
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    ⊢ Eq (EMetric.infEdist x (Metric.thickening δ s)) (HSub.hSub (EMetric.infEdist …
  -/
  refine (tsub_le_iff_right.2 infEdist_le_infEdist_thickening_add).antisymm' ?_
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    ⊢ LE.le (EMetric.infEdist x (Metric.thickening δ s)) (HSub.hSub (EMetric.infEd …
  -/
  refine le_sub_of_add_le_right ofReal_ne_top ?_
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    ⊢ LE.le (HAdd.hAdd (EMetric.infEdist x (Metric.thickening δ s)) (ENNReal.ofRea …
  -/
  refine le_infEdist.2 fun z hz => le_of_forall_lt' fun r h => ?_
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    z : E
    hz : Membership.mem s z
    r : ENNReal
    h : LT.lt (EDist.edist x z) r
    ⊢ LT.lt (HAdd.hAdd (EMetric.infEdist x (Metric.thickening δ s)) (ENNReal.ofRea …
  -/
  cases' r with r
  · exact add_lt_top.2 ⟨lt_top_iff_ne_top.2 <| infEdist_ne_top ⟨z, self_subset_thickening hδ _ hz⟩,
      ofReal_lt_top⟩
  have hr : 0 < ↑r - δ := by
    refine sub_pos_of_lt ?_
    have := hs.trans_lt ((infEdist_le_edist_of_mem hz).trans_lt h)
    rw [ofReal_eq_coe_nnreal hδ.le] at this
    exact mod_cast this
  /-
    case inr.coe
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    z : E
    hz : Membership.mem s z
    r : NNReal
    h : LT.lt (EDist.edist x z) ↑r
    hr : LT.lt 0 (HSub.hSub (↑r) δ)
    ⊢ LT.lt (HAdd.hAdd (EMetric.infEdist x (Metric.thickening δ s)) (ENNReal.ofRea …
  -/
  rw [edist_lt_coe, ← dist_lt_coe, ← add_sub_cancel δ ↑r] at h
  /-
    case inr.coe
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    z : E
    hz : Membership.mem s z
    r : NNReal
    h : LT.lt (Dist.dist x z) (HAdd.hAdd δ (HSub.hSub (↑r) δ))
    hr : LT.lt 0 (HSub.hSub (↑r) δ)
    ⊢ LT.lt (HAdd.hAdd (EMetric.infEdist x (Metric.thickening δ s)) (ENNReal.ofRea …
  -/
  obtain ⟨y, hxy, hyz⟩ := exists_dist_lt_lt hr hδ h
  refine (ENNReal.add_lt_add_right ofReal_ne_top <|
    infEdist_lt_iff.2 ⟨_, mem_thickening_iff.2 ⟨_, hz, hyz⟩, edist_lt_ofReal.2 hxy⟩).trans_le ?_
  /-
    case inr.coe.intro.intro
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    x : E
    hs : LE.le (ENNReal.ofReal δ) (EMetric.infEdist x s)
    z : E
    hz : Membership.mem s z
    r : NNReal
    h : LT.lt (Dist.dist x z) (HAdd.hAdd δ (HSub.hSub (↑r) δ))
    hr : LT.lt 0 (HSub.hSub (↑r) δ)
    y : E
    hxy : LT.lt (Dist.dist x y) (HSub.hSub (↑r) δ)
    hyz : LT.lt (Dist.dist y z) δ
    ⊢ LE.le (HAdd.hAdd (ENNReal.ofReal (HSub.hSub (↑r) δ)) (ENNReal.ofReal δ)) ↑r
  -/
  rw [← ofReal_add hr.le hδ.le, sub_add_cancel, ofReal_coe_nnreal]
  /-
    🎉 no goals
  -/


@[simp]
theorem thickening_thickening (hε : 0 < ε) (hδ : 0 < δ) (s : Set E) :
    thickening ε (thickening δ s) = thickening (ε + δ) s :=
  (thickening_thickening_subset _ _ _).antisymm fun x => by
    /-
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LT.lt 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x : E
      ⊢ Membership.mem (Metric.thickening (HAdd.hAdd ε δ) s) x → Membership.mem (Met …
    -/
    simp_rw [mem_thickening_iff]
    /-
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LT.lt 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x : E
      ⊢ (Exists fun z => And (Membership.mem s z) (LT.lt (Dist.dist x z) (HAdd.hAdd  …
    -/
    rintro ⟨z, hz, hxz⟩
    /-
      case intro.intro
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LT.lt 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x z : E
      hz : Membership.mem s z
      hxz : LT.lt (Dist.dist x z) (HAdd.hAdd ε δ)
      ⊢ Exists fun z => And (Exists fun z_1 => And (Membership.mem s z_1) (LT.lt (Di …
    -/
    rw [add_comm] at hxz
    /-
      case intro.intro
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LT.lt 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x z : E
      hz : Membership.mem s z
      hxz : LT.lt (Dist.dist x z) (HAdd.hAdd δ ε)
      ⊢ Exists fun z => And (Exists fun z_1 => And (Membership.mem s z_1) (LT.lt (Di …
    -/
    obtain ⟨y, hxy, hyz⟩ := exists_dist_lt_lt hε hδ hxz
    /-
      case intro.intro.intro.intro
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LT.lt 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x z : E
      hz : Membership.mem s z
      hxz : LT.lt (Dist.dist x z) (HAdd.hAdd δ ε)
      y : E
      hxy : LT.lt (Dist.dist x y) ε
      hyz : LT.lt (Dist.dist y z) δ
      ⊢ Exists fun z => And (Exists fun z_1 => And (Membership.mem s z_1) (LT.lt (Di …
    -/
    exact ⟨y, ⟨_, hz, hyz⟩, hxy⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem cthickening_thickening (hε : 0 ≤ ε) (hδ : 0 < δ) (s : Set E) :
    cthickening ε (thickening δ s) = cthickening (ε + δ) s :=
  (cthickening_thickening_subset hε _ _).antisymm fun x => by
    /-
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LE.le 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x : E
      ⊢ Membership.mem (Metric.cthickening (HAdd.hAdd ε δ) s) x → Membership.mem (Me …
    -/
    simp_rw [mem_cthickening_iff, ENNReal.ofReal_add hε hδ.le, infEdist_thickening hδ]
    /-
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LE.le 0 ε
      hδ : LT.lt 0 δ
      s : Set E
      x : E
      ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (ENNReal.ofReal ε) (ENNReal.ofReal δ …
    -/
    exact tsub_le_iff_right.2
    /-
      🎉 no goals
    -/

-- Note: `interior (cthickening δ s) ≠ thickening δ s` in general

@[simp]
theorem closure_thickening (hδ : 0 < δ) (s : Set E) :
    closure (thickening δ s) = cthickening δ s := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    hδ : LT.lt 0 δ
    s : Set E
    ⊢ Eq (closure (Metric.thickening δ s)) (Metric.cthickening δ s)
  -/
  rw [← cthickening_zero, cthickening_thickening le_rfl hδ, zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem infEdist_cthickening (δ : ℝ) (s : Set E) (x : E) :
    infEdist x (cthickening δ s) = infEdist x s - ENNReal.ofReal δ := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ : Real
    s : Set E
    x : E
    ⊢ Eq (EMetric.infEdist x (Metric.cthickening δ s)) (HSub.hSub (EMetric.infEdis …
  -/
  obtain hδ | hδ := le_or_lt δ 0
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ : Real
      s : Set E
      x : E
      hδ : LE.le δ 0
      ⊢ Eq (EMetric.infEdist x (Metric.cthickening δ s)) (HSub.hSub (EMetric.infEdis …
    -/
  · rw [cthickening_of_nonpos hδ, infEdist_closure, ofReal_of_nonpos hδ, tsub_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ : Real
      s : Set E
      x : E
      hδ : LT.lt 0 δ
      ⊢ Eq (EMetric.infEdist x (Metric.cthickening δ s)) (HSub.hSub (EMetric.infEdis …
    -/
  · rw [← closure_thickening hδ, infEdist_closure, infEdist_thickening hδ]
    /-
      🎉 no goals
    -/


@[simp]
theorem thickening_cthickening (hε : 0 < ε) (hδ : 0 ≤ δ) (s : Set E) :
    thickening ε (cthickening δ s) = thickening (ε + δ) s := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LE.le 0 δ
    s : Set E
    ⊢ Eq (Metric.thickening ε (Metric.cthickening δ s)) (Metric.thickening (HAdd.h …
  -/
  obtain rfl | hδ := hδ.eq_or_lt
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      ε : Real
      hε : LT.lt 0 ε
      s : Set E
      hδ : LE.le 0 0
      ⊢ Eq (Metric.thickening ε (Metric.cthickening 0 s)) (Metric.thickening (HAdd.h …
    -/
  · rw [cthickening_zero, thickening_closure, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LT.lt 0 ε
      hδ✝ : LE.le 0 δ
      s : Set E
      hδ : LT.lt 0 δ
      ⊢ Eq (Metric.thickening ε (Metric.cthickening δ s)) (Metric.thickening (HAdd.h …
    -/
  · rw [← closure_thickening hδ, thickening_closure, thickening_thickening hε hδ]
    /-
      🎉 no goals
    -/


@[simp]
theorem cthickening_cthickening (hε : 0 ≤ ε) (hδ : 0 ≤ δ) (s : Set E) :
    cthickening ε (cthickening δ s) = cthickening (ε + δ) s :=
  (cthickening_cthickening_subset hε hδ _).antisymm fun x => by
    /-
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LE.le 0 ε
      hδ : LE.le 0 δ
      s : Set E
      x : E
      ⊢ Membership.mem (Metric.cthickening (HAdd.hAdd ε δ) s) x → Membership.mem (Me …
    -/
    simp_rw [mem_cthickening_iff, ENNReal.ofReal_add hε hδ, infEdist_cthickening]
    /-
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      δ ε : Real
      hε : LE.le 0 ε
      hδ : LE.le 0 δ
      s : Set E
      x : E
      ⊢ LE.le (EMetric.infEdist x s) (HAdd.hAdd (ENNReal.ofReal ε) (ENNReal.ofReal δ …
    -/
    exact tsub_le_iff_right.2
    /-
      🎉 no goals
    -/


@[simp]
theorem thickening_ball (hε : 0 < ε) (hδ : 0 < δ) (x : E) :
    thickening ε (ball x δ) = ball x (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LT.lt 0 δ
    x : E
    ⊢ Eq (Metric.thickening ε (Metric.ball x δ)) (Metric.ball x (HAdd.hAdd ε δ))
  -/
  rw [← thickening_singleton, thickening_thickening hε hδ, thickening_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem thickening_closedBall (hε : 0 < ε) (hδ : 0 ≤ δ) (x : E) :
    thickening ε (closedBall x δ) = ball x (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LE.le 0 δ
    x : E
    ⊢ Eq (Metric.thickening ε (Metric.closedBall x δ)) (Metric.ball x (HAdd.hAdd ε …
  -/
  rw [← cthickening_singleton _ hδ, thickening_cthickening hε hδ, thickening_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem cthickening_ball (hε : 0 ≤ ε) (hδ : 0 < δ) (x : E) :
    cthickening ε (ball x δ) = closedBall x (ε + δ) := by
  rw [← thickening_singleton, cthickening_thickening hε hδ,
      cthickening_singleton _ (add_nonneg hε hδ.le)]


@[simp]
theorem cthickening_closedBall (hε : 0 ≤ ε) (hδ : 0 ≤ δ) (x : E) :
    cthickening ε (closedBall x δ) = closedBall x (ε + δ) := by
  rw [← cthickening_singleton _ hδ, cthickening_cthickening hε hδ,
      cthickening_singleton _ (add_nonneg hε hδ)]


theorem ball_add_ball (hε : 0 < ε) (hδ : 0 < δ) (a b : E) :
    ball a ε + ball b δ = ball (a + b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LT.lt 0 δ
    a b : E
    ⊢ Eq (HAdd.hAdd (Metric.ball a ε) (Metric.ball b δ)) (Metric.ball (HAdd.hAdd a …
  -/
  rw [ball_add, thickening_ball hε hδ b, Metric.vadd_ball, vadd_eq_add]
  /-
    🎉 no goals
  -/


theorem ball_sub_ball (hε : 0 < ε) (hδ : 0 < δ) (a b : E) :
    ball a ε - ball b δ = ball (a - b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LT.lt 0 δ
    a b : E
    ⊢ Eq (HSub.hSub (Metric.ball a ε) (Metric.ball b δ)) (Metric.ball (HSub.hSub a …
  -/
  simp_rw [sub_eq_add_neg, neg_ball, ball_add_ball hε hδ]
  /-
    🎉 no goals
  -/


theorem ball_add_closedBall (hε : 0 < ε) (hδ : 0 ≤ δ) (a b : E) :
    ball a ε + closedBall b δ = ball (a + b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LE.le 0 δ
    a b : E
    ⊢ Eq (HAdd.hAdd (Metric.ball a ε) (Metric.closedBall b δ)) (Metric.ball (HAdd. …
  -/
  rw [ball_add, thickening_closedBall hε hδ b, Metric.vadd_ball, vadd_eq_add]
  /-
    🎉 no goals
  -/


theorem ball_sub_closedBall (hε : 0 < ε) (hδ : 0 ≤ δ) (a b : E) :
    ball a ε - closedBall b δ = ball (a - b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LT.lt 0 ε
    hδ : LE.le 0 δ
    a b : E
    ⊢ Eq (HSub.hSub (Metric.ball a ε) (Metric.closedBall b δ)) (Metric.ball (HSub. …
  -/
  simp_rw [sub_eq_add_neg, neg_closedBall, ball_add_closedBall hε hδ]
  /-
    🎉 no goals
  -/


theorem closedBall_add_ball (hε : 0 ≤ ε) (hδ : 0 < δ) (a b : E) :
    closedBall a ε + ball b δ = ball (a + b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LE.le 0 ε
    hδ : LT.lt 0 δ
    a b : E
    ⊢ Eq (HAdd.hAdd (Metric.closedBall a ε) (Metric.ball b δ)) (Metric.ball (HAdd. …
  -/
  rw [add_comm, ball_add_closedBall hδ hε b, add_comm, add_comm δ]
  /-
    🎉 no goals
  -/


theorem closedBall_sub_ball (hε : 0 ≤ ε) (hδ : 0 < δ) (a b : E) :
    closedBall a ε - ball b δ = ball (a - b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    δ ε : Real
    hε : LE.le 0 ε
    hδ : LT.lt 0 δ
    a b : E
    ⊢ Eq (HSub.hSub (Metric.closedBall a ε) (Metric.ball b δ)) (Metric.ball (HSub. …
  -/
  simp_rw [sub_eq_add_neg, neg_ball, closedBall_add_ball hε hδ]
  /-
    🎉 no goals
  -/


theorem closedBall_add_closedBall [ProperSpace E] (hε : 0 ≤ ε) (hδ : 0 ≤ δ) (a b : E) :
    closedBall a ε + closedBall b δ = closedBall (a + b) (ε + δ) := by
  rw [(isCompact_closedBall _ _).add_closedBall hδ b, cthickening_closedBall hδ hε a,
    Metric.vadd_closedBall, vadd_eq_add, add_comm, add_comm δ]


theorem closedBall_sub_closedBall [ProperSpace E] (hε : 0 ≤ ε) (hδ : 0 ≤ δ) (a b : E) :
    closedBall a ε - closedBall b δ = closedBall (a - b) (ε + δ) := by
  /-
    E : Type u_2
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    δ ε : Real
    inst✝ : ProperSpace E
    hε : LE.le 0 ε
    hδ : LE.le 0 δ
    a b : E
    ⊢ Eq (HSub.hSub (Metric.closedBall a ε) (Metric.closedBall b δ)) (Metric.close …
  -/
  rw [sub_eq_add_neg, neg_closedBall, closedBall_add_closedBall hε hδ, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem smul_closedBall (c : 𝕜) (x : E) {r : ℝ} (hr : 0 ≤ r) :
    c • closedBall x r = closedBall (c • x) (‖c‖ * r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HSMul.hSMul c (Metric.closedBall x r)) (Metric.closedBall (HSMul.hSMul c …
  -/
  rcases eq_or_ne c 0 with (rfl | hc)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NormedField 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      r : Real
      hr : LE.le 0 r
      ⊢ Eq (HSMul.hSMul 0 (Metric.closedBall x r)) (Metric.closedBall (HSMul.hSMul 0 …
    -/
  · simp [hr, zero_smul_set, Set.singleton_zero, nonempty_closedBall]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NormedField 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      c : 𝕜
      x : E
      r : Real
      hr : LE.le 0 r
      hc : Ne c 0
      ⊢ Eq (HSMul.hSMul c (Metric.closedBall x r)) (Metric.closedBall (HSMul.hSMul c …
    -/
  · exact smul_closedBall' hc x r
    /-
      🎉 no goals
    -/


theorem smul_unitClosedBall (c : 𝕜) : c • closedBall (0 : E) (1 : ℝ) = closedBall (0 : E) ‖c‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    ⊢ Eq (HSMul.hSMul c (Metric.closedBall 0 1)) (Metric.closedBall 0 (Norm.norm c))
  -/
  rw [_root_.smul_closedBall _ _ zero_le_one, smul_zero, mul_one]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-01")] alias smul_closedUnitBall := smul_unitClosedBall


/-- In a real normed space, the image of the unit closed ball under multiplication by a nonnegative
number `r` is the closed ball of radius `r` with center at the origin. -/
theorem smul_unitClosedBall_of_nonneg {r : ℝ} (hr : 0 ≤ r) :
    r • closedBall (0 : E) 1 = closedBall (0 : E) r := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HSMul.hSMul r (Metric.closedBall 0 1)) (Metric.closedBall 0 r)
  -/
  rw [smul_unitClosedBall, Real.norm_of_nonneg hr]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-01")]
alias smul_closedUnitBall_of_nonneg := smul_unitClosedBall_of_nonneg


/-- In a nontrivial real normed space, a sphere is nonempty if and only if its radius is
nonnegative. -/
@[simp]
theorem NormedSpace.sphere_nonempty [Nontrivial E] {x : E} {r : ℝ} :
    (sphere x r).Nonempty ↔ 0 ≤ r := by
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Iff (Metric.sphere x r).Nonempty (LE.le 0 r)
  -/
  obtain ⟨y, hy⟩ := exists_ne x
  refine ⟨fun h => nonempty_closedBall.1 (h.mono sphere_subset_closedBall), fun hr =>
    ⟨r • ‖y - x‖⁻¹ • (y - x) + x, ?_⟩⟩
  /-
    case intro
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    y : E
    hy : Ne y x
    hr : LE.le 0 r
    ⊢ Membership.mem (Metric.sphere x r) (HAdd.hAdd (HSMul.hSMul r (HSMul.hSMul (I …
  -/
  have : ‖y - x‖ ≠ 0 := by simpa [sub_eq_zero]
  simp only [mem_sphere_iff_norm, add_sub_cancel_right, norm_smul, Real.norm_eq_abs, norm_inv,
    norm_norm, ne_eq, norm_eq_zero]
  /-
    case intro
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    y : E
    hy : Ne y x
    hr : LE.le 0 r
    this : Ne (Norm.norm (HSub.hSub y x)) 0
    ⊢ Eq (HMul.hMul (abs r) (HMul.hMul (Inv.inv (abs (Norm.norm (HSub.hSub y x)))) …
  -/
  simp only [abs_norm, ne_eq, norm_eq_zero]
  /-
    case intro
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    y : E
    hy : Ne y x
    hr : LE.le 0 r
    this : Ne (Norm.norm (HSub.hSub y x)) 0
    ⊢ Eq (HMul.hMul (abs r) (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub y x))) (Norm …
  -/
  rw [inv_mul_cancel₀ this, mul_one, abs_eq_self.mpr hr]
  /-
    🎉 no goals
  -/


theorem smul_sphere [Nontrivial E] (c : 𝕜) (x : E) {r : ℝ} (hr : 0 ≤ r) :
    c • sphere x r = sphere (c • x) (‖c‖ * r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    c : 𝕜
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HSMul.hSMul c (Metric.sphere x r)) (Metric.sphere (HSMul.hSMul c x) (HMu …
  -/
  rcases eq_or_ne c 0 with (rfl | hc)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      r : Real
      hr : LE.le 0 r
      ⊢ Eq (HSMul.hSMul 0 (Metric.sphere x r)) (Metric.sphere (HSMul.hSMul 0 x) (HMu …
    -/
  · simp [zero_smul_set, Set.singleton_zero, hr]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      c : 𝕜
      x : E
      r : Real
      hr : LE.le 0 r
      hc : Ne c 0
      ⊢ Eq (HSMul.hSMul c (Metric.sphere x r)) (Metric.sphere (HSMul.hSMul c x) (HMu …
    -/
  · exact smul_sphere' hc x r
    /-
      🎉 no goals
    -/


/-- Any ball `Metric.ball x r`, `0 < r` is the image of the unit ball under `fun y ↦ x + r • y`. -/
theorem affinity_unitBall {r : ℝ} (hr : 0 < r) (x : E) : x +ᵥ r • ball (0 : E) 1 = ball x r := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    r : Real
    hr : LT.lt 0 r
    x : E
    ⊢ Eq (HVAdd.hVAdd x (HSMul.hSMul r (Metric.ball 0 1))) (Metric.ball x r)
  -/
  rw [smul_unitBall_of_pos hr, vadd_ball_zero]
  /-
    🎉 no goals
  -/


/-- Any closed ball `Metric.closedBall x r`, `0 ≤ r` is the image of the unit closed ball under
`fun y ↦ x + r • y`. -/
theorem affinity_unitClosedBall {r : ℝ} (hr : 0 ≤ r) (x : E) :
    x +ᵥ r • closedBall (0 : E) 1 = closedBall x r := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    r : Real
    hr : LE.le 0 r
    x : E
    ⊢ Eq (HVAdd.hVAdd x (HSMul.hSMul r (Metric.closedBall 0 1))) (Metric.closedBal …
  -/
  rw [smul_unitClosedBall, Real.norm_of_nonneg hr, vadd_closedBall_zero]
  /-
    🎉 no goals
  -/


