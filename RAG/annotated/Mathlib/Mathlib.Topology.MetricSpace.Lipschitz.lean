theorem lipschitzWith_iff_dist_le_mul [PseudoMetricSpace α] [PseudoMetricSpace β] {K : ℝ≥0}
    {f : α → β} : LipschitzWith K f ↔ ∀ x y, dist (f x) (f y) ≤ K * dist x y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    ⊢ Iff (LipschitzWith K f) (∀ (x y : α), LE.le (Dist.dist (f x) (f y)) (HMul.hM …
  -/
  simp only [LipschitzWith, edist_nndist, dist_nndist]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    ⊢ Iff (∀ (x y : α), LE.le (↑(NNDist.nndist (f x) (f y))) (HMul.hMul ↑K ↑(NNDis …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


alias ⟨LipschitzWith.dist_le_mul, LipschitzWith.of_dist_le_mul⟩ := lipschitzWith_iff_dist_le_mul


theorem lipschitzOnWith_iff_dist_le_mul [PseudoMetricSpace α] [PseudoMetricSpace β] {K : ℝ≥0}
    {s : Set α} {f : α → β} :
    LipschitzOnWith K f s ↔ ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ K * dist x y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    s : Set α
    f : α → β
    ⊢ Iff (LipschitzOnWith K f s) (∀ (x : α), Membership.mem s x → ∀ (y : α), Memb …
  -/
  simp only [LipschitzOnWith, edist_nndist, dist_nndist]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    s : Set α
    f : α → β
    ⊢ Iff (∀ ⦃x : α⦄, Membership.mem s x → ∀ ⦃y : α⦄, Membership.mem s y → LE.le ( …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


alias ⟨LipschitzOnWith.dist_le_mul, LipschitzOnWith.of_dist_le_mul⟩ :=
  lipschitzOnWith_iff_dist_le_mul


protected theorem of_dist_le' {K : ℝ} (h : ∀ x y, dist (f x) (f y) ≤ K * dist x y) :
    LipschitzWith (Real.toNNReal K) f :=
  of_dist_le_mul fun x y =>
                           /-
                             α : Type u
                             β : Type v
                             inst✝¹ : PseudoMetricSpace α
                             inst✝ : PseudoMetricSpace β
                             f : α → β
                             K : Real
                             h : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) (HMul.hMul K (Dist.dist x y))
                             x y : α
                             ⊢ LE.le (HMul.hMul K (Dist.dist x y)) (HMul.hMul (↑K.toNNReal) (Dist.dist x y))
                           -/
    le_trans (h x y) <| by gcongr; apply Real.le_coe_toNNReal
                                   /-
                                     🎉 no goals
                                   -/


protected theorem mk_one (h : ∀ x y, dist (f x) (f y) ≤ dist x y) : LipschitzWith 1 f :=
                       /-
                         α : Type u
                         β : Type v
                         inst✝¹ : PseudoMetricSpace α
                         inst✝ : PseudoMetricSpace β
                         f : α → β
                         h : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) (Dist.dist x y)
                         ⊢ ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) (HMul.hMul (↑1) (Dist.dist x y))
                       -/
  of_dist_le_mul <| by simpa only [NNReal.coe_one, one_mul] using h
                       /-
                         🎉 no goals
                       -/


/-- For functions to `ℝ`, it suffices to prove `f x ≤ f y + K * dist x y`; this version
doesn't assume `0≤K`. -/
protected theorem of_le_add_mul' {f : α → ℝ} (K : ℝ) (h : ∀ x y, f x ≤ f y + K * dist x y) :
    LipschitzWith (Real.toNNReal K) f :=
  have I : ∀ x y, f x - f y ≤ K * dist x y := fun x y => sub_le_iff_le_add'.2 (h x y)
  LipschitzWith.of_dist_le' fun x y => abs_sub_le_iff.2 ⟨I x y, dist_comm y x ▸ I y x⟩


/-- For functions to `ℝ`, it suffices to prove `f x ≤ f y + K * dist x y`; this version
assumes `0≤K`. -/
protected theorem of_le_add_mul {f : α → ℝ} (K : ℝ≥0) (h : ∀ x y, f x ≤ f y + K * dist x y) :
                            /-
                              α : Type u
                              inst✝ : PseudoMetricSpace α
                              f : α → Real
                              K : NNReal
                              h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul (↑K) (Dist.dist x y)))
                              ⊢ LipschitzWith K f
                            -/
    LipschitzWith K f := by simpa only [Real.toNNReal_coe] using LipschitzWith.of_le_add_mul' K h
                            /-
                              🎉 no goals
                            -/


protected theorem of_le_add {f : α → ℝ} (h : ∀ x y, f x ≤ f y + dist x y) : LipschitzWith 1 f :=
                                      /-
                                        α : Type u
                                        inst✝ : PseudoMetricSpace α
                                        f : α → Real
                                        h : ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (Dist.dist x y))
                                        ⊢ ∀ (x y : α), LE.le (f x) (HAdd.hAdd (f y) (HMul.hMul (↑1) (Dist.dist x y)))
                                      -/
  LipschitzWith.of_le_add_mul 1 <| by simpa only [NNReal.coe_one, one_mul]
                                      /-
                                        🎉 no goals
                                      -/


protected theorem le_add_mul {f : α → ℝ} {K : ℝ≥0} (h : LipschitzWith K f) (x y) :
    f x ≤ f y + K * dist x y :=
  sub_le_iff_le_add'.1 <| le_trans (le_abs_self _) <| h.dist_le_mul x y


protected theorem iff_le_add_mul {f : α → ℝ} {K : ℝ≥0} :
    LipschitzWith K f ↔ ∀ x y, f x ≤ f y + K * dist x y :=
  ⟨LipschitzWith.le_add_mul, LipschitzWith.of_le_add_mul K⟩


theorem nndist_le (hf : LipschitzWith K f) (x y : α) : nndist (f x) (f y) ≤ K * nndist x y :=
  hf.dist_le_mul x y


theorem dist_le_mul_of_le (hf : LipschitzWith K f) (hr : dist x y ≤ r) : dist (f x) (f y) ≤ K * r :=
                                   /-
                                     α : Type u
                                     β : Type v
                                     inst✝¹ : PseudoMetricSpace α
                                     inst✝ : PseudoMetricSpace β
                                     K : NNReal
                                     f : α → β
                                     x y : α
                                     r : Real
                                     hf : LipschitzWith K f
                                     hr : LE.le (Dist.dist x y) r
                                     ⊢ LE.le (HMul.hMul (↑K) (Dist.dist x y)) (HMul.hMul (↑K) r)
                                   -/
  (hf.dist_le_mul x y).trans <| by gcongr
                                   /-
                                     🎉 no goals
                                   -/


theorem mapsTo_closedBall (hf : LipschitzWith K f) (x : α) (r : ℝ) :
    MapsTo f (Metric.closedBall x r) (Metric.closedBall (f x) (K * r)) := fun _y hy =>
  hf.dist_le_mul_of_le hy


theorem dist_lt_mul_of_lt (hf : LipschitzWith K f) (hK : K ≠ 0) (hr : dist x y < r) :
    dist (f x) (f y) < K * r :=
  (hf.dist_le_mul x y).trans_lt <| (mul_lt_mul_left <| NNReal.coe_pos.2 hK.bot_lt).2 hr


theorem mapsTo_ball (hf : LipschitzWith K f) (hK : K ≠ 0) (x : α) (r : ℝ) :
    MapsTo f (Metric.ball x r) (Metric.ball (f x) (K * r)) := fun _y hy =>
  hf.dist_lt_mul_of_lt hK hy


/-- A Lipschitz continuous map is a locally bounded map. -/
def toLocallyBoundedMap (f : α → β) (hf : LipschitzWith K f) : LocallyBoundedMap α β :=
  LocallyBoundedMap.ofMapBounded f fun _s hs =>
    let ⟨C, hC⟩ := Metric.isBounded_iff.1 hs
    Metric.isBounded_iff.2 ⟨K * C, forall_mem_image.2 fun _x hx => forall_mem_image.2 fun _y hy =>
      hf.dist_le_mul_of_le (hC hx hy)⟩


@[simp]
theorem coe_toLocallyBoundedMap (hf : LipschitzWith K f) : ⇑(hf.toLocallyBoundedMap f) = f :=
  rfl


theorem comap_cobounded_le (hf : LipschitzWith K f) :
    comap f (Bornology.cobounded β) ≤ Bornology.cobounded α :=
  (hf.toLocallyBoundedMap f).2


/-- The image of a bounded set under a Lipschitz map is bounded. -/
theorem isBounded_image (hf : LipschitzWith K f) {s : Set α} (hs : IsBounded s) :
    IsBounded (f '' s) :=
  hs.image (toLocallyBoundedMap f hf)


theorem diam_image_le (hf : LipschitzWith K f) (s : Set α) (hs : IsBounded s) :
    Metric.diam (f '' s) ≤ K * Metric.diam s :=
  Metric.diam_le_of_forall_dist_le (mul_nonneg K.coe_nonneg Metric.diam_nonneg) <|
    forall_mem_image.2 fun _x hx =>
      forall_mem_image.2 fun _y hy => hf.dist_le_mul_of_le <| Metric.dist_le_diam_of_mem hs hx hy


protected theorem dist_left (y : α) : LipschitzWith 1 (dist · y) :=
  LipschitzWith.mk_one fun _ _ => dist_dist_dist_le_left _ _ _


protected theorem dist_right (x : α) : LipschitzWith 1 (dist x) :=
  LipschitzWith.of_le_add fun _ _ => dist_triangle_right _ _ _


protected theorem dist : LipschitzWith 2 (Function.uncurry <| @dist α _) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ LipschitzWith 2 (Function.uncurry Dist.dist)
  -/
  rw [← one_add_one_eq_two]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ LipschitzWith (HAdd.hAdd 1 1) (Function.uncurry Dist.dist)
  -/
  exact LipschitzWith.uncurry LipschitzWith.dist_left LipschitzWith.dist_right
  /-
    🎉 no goals
  -/


theorem dist_iterate_succ_le_geometric {f : α → α} (hf : LipschitzWith K f) (x n) :
    dist (f^[n] x) (f^[n + 1] x) ≤ dist x (f x) * (K : ℝ) ^ n := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    K : NNReal
    f : α → α
    hf : LipschitzWith K f
    x : α
    n : Nat
    ⊢ LE.le (Dist.dist (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)) (HMu …
  -/
  rw [iterate_succ, mul_comm]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    K : NNReal
    f : α → α
    hf : LipschitzWith K f
    x : α
    n : Nat
    ⊢ LE.le (Dist.dist (Nat.iterate f n x) (Function.comp (Nat.iterate f n) f x))  …
  -/
  simpa only [NNReal.coe_pow] using (hf.iterate n).dist_le_mul x (f x)
  /-
    🎉 no goals
  -/


theorem _root_.lipschitzWith_max : LipschitzWith 1 fun p : ℝ × ℝ => max p.1 p.2 :=
  LipschitzWith.of_le_add fun _ _ => sub_le_iff_le_add'.1 <|
    (le_abs_self _).trans (abs_max_sub_max_le_max _ _ _ _)


theorem _root_.lipschitzWith_min : LipschitzWith 1 fun p : ℝ × ℝ => min p.1 p.2 :=
  LipschitzWith.of_le_add fun _ _ => sub_le_iff_le_add'.1 <|
    (le_abs_self _).trans (abs_min_sub_min_le_max _ _ _ _)


lemma _root_.Real.lipschitzWith_toNNReal : LipschitzWith 1 Real.toNNReal := by
  /-
    ⊢ LipschitzWith 1 Real.toNNReal
  -/
  refine lipschitzWith_iff_dist_le_mul.mpr (fun x y ↦ ?_)
  simpa only [NNReal.coe_one, dist_prod_same_right, one_mul, Real.dist_eq] using
    lipschitzWith_iff_dist_le_mul.mp lipschitzWith_max (x, 0) (y, 0)


lemma cauchySeq_comp (hf : LipschitzWith K f) {u : ℕ → α} (hu : CauchySeq u) :
    CauchySeq (f ∘ u) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    hf : LipschitzWith K f
    u : Nat → α
    hu : CauchySeq u
    ⊢ CauchySeq (Function.comp f u)
  -/
  rcases cauchySeq_iff_le_tendsto_0.1 hu with ⟨b, b_nonneg, hb, blim⟩
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    f : α → β
    hf : LipschitzWith K f
    u : Nat → α
    hu : CauchySeq u
    b : Nat → Real
    b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
    hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
    blim : Filter.Tendsto b Filter.atTop (nhds 0)
    ⊢ CauchySeq (Function.comp f u)
  -/
  refine cauchySeq_iff_le_tendsto_0.2 ⟨fun n ↦ K * b n, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      f : α → β
      hf : LipschitzWith K f
      u : Nat → α
      hu : CauchySeq u
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ ∀ (n : Nat), LE.le 0 ((fun n => HMul.hMul (↑K) (b n)) n)
    -/
  · exact fun n ↦ mul_nonneg (by positivity) (b_nonneg n)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      f : α → β
      hf : LipschitzWith K f
      u : Nat → α
      hu : CauchySeq u
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (Function.comp f u …
    -/
  · exact fun n m N hn hm ↦ hf.dist_le_mul_of_le (hb n m N hn hm)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      f : α → β
      hf : LipschitzWith K f
      u : Nat → α
      hu : CauchySeq u
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ Filter.Tendsto (fun n => HMul.hMul (↑K) (b n)) Filter.atTop (nhds 0)
    -/
  · rw [← mul_zero (K : ℝ)]
    /-
      case intro.intro.intro.refine_3
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      f : α → β
      hf : LipschitzWith K f
      u : Nat → α
      hu : CauchySeq u
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ Filter.Tendsto (fun n => HMul.hMul (↑K) (b n)) Filter.atTop (nhds (HMul.hMul …
    -/
    exact blim.const_mul _
    /-
      🎉 no goals
    -/


protected theorem max (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) :
    LipschitzWith (max Kf Kg) fun x => max (f x) (g x) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f g : α → Real
    Kf Kg : NNReal
    hf : LipschitzWith Kf f
    hg : LipschitzWith Kg g
    ⊢ LipschitzWith (Max.max Kf Kg) fun x => Max.max (f x) (g x)
  -/
  simpa only [(· ∘ ·), one_mul] using lipschitzWith_max.comp (hf.prod hg)
  /-
    🎉 no goals
  -/


protected theorem min (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) :
    LipschitzWith (max Kf Kg) fun x => min (f x) (g x) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f g : α → Real
    Kf Kg : NNReal
    hf : LipschitzWith Kf f
    hg : LipschitzWith Kg g
    ⊢ LipschitzWith (Max.max Kf Kg) fun x => Min.min (f x) (g x)
  -/
  simpa only [(· ∘ ·), one_mul] using lipschitzWith_min.comp (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem max_const (hf : LipschitzWith Kf f) (a : ℝ) : LipschitzWith Kf fun x => max (f x) a := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : α → Real
    Kf : NNReal
    hf : LipschitzWith Kf f
    a : Real
    ⊢ LipschitzWith Kf fun x => Max.max (f x) a
  -/
  simpa only [max_eq_left (zero_le Kf)] using hf.max (LipschitzWith.const a)
  /-
    🎉 no goals
  -/


theorem const_max (hf : LipschitzWith Kf f) (a : ℝ) : LipschitzWith Kf fun x => max a (f x) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : α → Real
    Kf : NNReal
    hf : LipschitzWith Kf f
    a : Real
    ⊢ LipschitzWith Kf fun x => Max.max a (f x)
  -/
  simpa only [max_comm] using hf.max_const a
  /-
    🎉 no goals
  -/


theorem min_const (hf : LipschitzWith Kf f) (a : ℝ) : LipschitzWith Kf fun x => min (f x) a := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : α → Real
    Kf : NNReal
    hf : LipschitzWith Kf f
    a : Real
    ⊢ LipschitzWith Kf fun x => Min.min (f x) a
  -/
  simpa only [max_eq_left (zero_le Kf)] using hf.min (LipschitzWith.const a)
  /-
    🎉 no goals
  -/


theorem const_min (hf : LipschitzWith Kf f) (a : ℝ) : LipschitzWith Kf fun x => min a (f x) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : α → Real
    Kf : NNReal
    hf : LipschitzWith Kf f
    a : Real
    ⊢ LipschitzWith Kf fun x => Min.min a (f x)
  -/
  simpa only [min_comm] using hf.min_const a
  /-
    🎉 no goals
  -/


protected theorem projIcc {a b : ℝ} (h : a ≤ b) : LipschitzWith 1 (projIcc a b h) :=
  ((LipschitzWith.id.const_min _).const_max _).subtype_mk _


protected theorem of_dist_le' {K : ℝ} (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ K * dist x y) :
    LipschitzOnWith (Real.toNNReal K) f s :=
  of_dist_le_mul fun x hx y hy =>
                                 /-
                                   α : Type u
                                   β : Type v
                                   inst✝¹ : PseudoMetricSpace α
                                   inst✝ : PseudoMetricSpace β
                                   s : Set α
                                   f : α → β
                                   K : Real
                                   h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (Dis …
                                   x : α
                                   hx : Membership.mem s x
                                   y : α
                                   hy : Membership.mem s y
                                   ⊢ LE.le (HMul.hMul K (Dist.dist x y)) (HMul.hMul (↑K.toNNReal) (Dist.dist x y))
                                 -/
    le_trans (h x hx y hy) <| by gcongr; apply Real.le_coe_toNNReal
                                         /-
                                           🎉 no goals
                                         -/


protected theorem mk_one (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ dist x y) :
    LipschitzOnWith 1 f s :=
                       /-
                         α : Type u
                         β : Type v
                         inst✝¹ : PseudoMetricSpace α
                         inst✝ : PseudoMetricSpace β
                         s : Set α
                         f : α → β
                         h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (Dis …
                         ⊢ ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (Dist. …
                       -/
  of_dist_le_mul <| by simpa only [NNReal.coe_one, one_mul] using h
                       /-
                         🎉 no goals
                       -/


/-- For functions to `ℝ`, it suffices to prove `f x ≤ f y + K * dist x y`; this version
doesn't assume `0≤K`. -/
protected theorem of_le_add_mul' {f : α → ℝ} (K : ℝ)
    (h : ∀ x ∈ s, ∀ y ∈ s, f x ≤ f y + K * dist x y) : LipschitzOnWith (Real.toNNReal K) f s :=
  have I : ∀ x ∈ s, ∀ y ∈ s, f x - f y ≤ K * dist x y := fun x hx y hy =>
    sub_le_iff_le_add'.2 (h x hx y hy)
  LipschitzOnWith.of_dist_le' fun x hx y hy =>
    abs_sub_le_iff.2 ⟨I x hx y hy, dist_comm y x ▸ I y hy x hx⟩


/-- For functions to `ℝ`, it suffices to prove `f x ≤ f y + K * dist x y`; this version
assumes `0≤K`. -/
protected theorem of_le_add_mul {f : α → ℝ} (K : ℝ≥0)
    (h : ∀ x ∈ s, ∀ y ∈ s, f x ≤ f y + K * dist x y) : LipschitzOnWith K f s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    f : α → Real
    K : NNReal
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (f x …
    ⊢ LipschitzOnWith K f s
  -/
  simpa only [Real.toNNReal_coe] using LipschitzOnWith.of_le_add_mul' K h
  /-
    🎉 no goals
  -/


protected theorem of_le_add {f : α → ℝ} (h : ∀ x ∈ s, ∀ y ∈ s, f x ≤ f y + dist x y) :
    LipschitzOnWith 1 f s :=
                                        /-
                                          α : Type u
                                          inst✝ : PseudoMetricSpace α
                                          s : Set α
                                          f : α → Real
                                          h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (f x …
                                          ⊢ ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le (f x)  …
                                        -/
  LipschitzOnWith.of_le_add_mul 1 <| by simpa only [NNReal.coe_one, one_mul]
                                        /-
                                          🎉 no goals
                                        -/


protected theorem le_add_mul {f : α → ℝ} {K : ℝ≥0} (h : LipschitzOnWith K f s) {x : α} (hx : x ∈ s)
    {y : α} (hy : y ∈ s) : f x ≤ f y + K * dist x y :=
  sub_le_iff_le_add'.1 <| le_trans (le_abs_self _) <| h.dist_le_mul x hx y hy


protected theorem iff_le_add_mul {f : α → ℝ} {K : ℝ≥0} :
    LipschitzOnWith K f s ↔ ∀ x ∈ s, ∀ y ∈ s, f x ≤ f y + K * dist x y :=
  ⟨LipschitzOnWith.le_add_mul, LipschitzOnWith.of_le_add_mul K⟩


theorem isBounded_image2 (f : α → β → γ) {K₁ K₂ : ℝ≥0} {s : Set α} {t : Set β}
    (hs : Bornology.IsBounded s) (ht : Bornology.IsBounded t)
    (hf₁ : ∀ b ∈ t, LipschitzOnWith K₁ (fun a => f a b) s)
    (hf₂ : ∀ a ∈ s, LipschitzOnWith K₂ (f a) t) : Bornology.IsBounded (Set.image2 f s t) :=
  Metric.isBounded_iff_ediam_ne_top.2 <|
    ne_top_of_le_ne_top
      (ENNReal.add_ne_top.mpr
        ⟨ENNReal.mul_ne_top ENNReal.coe_ne_top hs.ediam_ne_top,
          ENNReal.mul_ne_top ENNReal.coe_ne_top ht.ediam_ne_top⟩)
      (ediam_image2_le _ _ _ hf₁ hf₂)


lemma cauchySeq_comp (hf : LipschitzOnWith K f s)
    {u : ℕ → α} (hu : CauchySeq u) (h'u : range u ⊆ s) :
    CauchySeq (f ∘ u) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    s : Set α
    f : α → β
    hf : LipschitzOnWith K f s
    u : Nat → α
    hu : CauchySeq u
    h'u : HasSubset.Subset (Set.range u) s
    ⊢ CauchySeq (Function.comp f u)
  -/
  rcases cauchySeq_iff_le_tendsto_0.1 hu with ⟨b, b_nonneg, hb, blim⟩
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    K : NNReal
    s : Set α
    f : α → β
    hf : LipschitzOnWith K f s
    u : Nat → α
    hu : CauchySeq u
    h'u : HasSubset.Subset (Set.range u) s
    b : Nat → Real
    b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
    hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
    blim : Filter.Tendsto b Filter.atTop (nhds 0)
    ⊢ CauchySeq (Function.comp f u)
  -/
  refine cauchySeq_iff_le_tendsto_0.2 ⟨fun n ↦ K * b n, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ ∀ (n : Nat), LE.le 0 ((fun n => HMul.hMul (↑K) (b n)) n)
    -/
  · exact fun n ↦ mul_nonneg (by positivity) (b_nonneg n)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (Function.comp f u …
    -/
  · intro n m N hn hm
    /-
      case intro.intro.intro.refine_2
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      n m N : Nat
      hn : LE.le N n
      hm : LE.le N m
      ⊢ LE.le (Dist.dist (Function.comp f u n) (Function.comp f u m)) ((fun n => HMu …
    -/
    have A n : u n ∈ s := h'u (mem_range_self _)
    /-
      case intro.intro.intro.refine_2
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      n m N : Nat
      hn : LE.le N n
      hm : LE.le N m
      A : ∀ (n : Nat), Membership.mem s (u n)
      ⊢ LE.le (Dist.dist (Function.comp f u n) (Function.comp f u m)) ((fun n => HMu …
    -/
    apply (hf.dist_le_mul _ (A n) _ (A m)).trans
    /-
      case intro.intro.intro.refine_2
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      n m N : Nat
      hn : LE.le N n
      hm : LE.le N m
      A : ∀ (n : Nat), Membership.mem s (u n)
      ⊢ LE.le (HMul.hMul (↑K) (Dist.dist (u n) (u m))) ((fun n => HMul.hMul (↑K) (b  …
    -/
    exact mul_le_mul_of_nonneg_left (hb n m N hn hm) K.2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ Filter.Tendsto (fun n => HMul.hMul (↑K) (b n)) Filter.atTop (nhds 0)
    -/
  · rw [← mul_zero (K : ℝ)]
    /-
      case intro.intro.intro.refine_3
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      K : NNReal
      s : Set α
      f : α → β
      hf : LipschitzOnWith K f s
      u : Nat → α
      hu : CauchySeq u
      h'u : HasSubset.Subset (Set.range u) s
      b : Nat → Real
      b_nonneg : ∀ (n : Nat), LE.le 0 (b n)
      hb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (u n) (u m)) (b …
      blim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ Filter.Tendsto (fun n => HMul.hMul (↑K) (b n)) Filter.atTop (nhds (HMul.hMul …
    -/
    exact blim.const_mul _
    /-
      🎉 no goals
    -/


/-- The minimum of locally Lipschitz functions is locally Lipschitz. -/
protected lemma min (hf : LocallyLipschitz f) (hg : LocallyLipschitz g) :
    LocallyLipschitz (fun x => min (f x) (g x)) :=
  lipschitzWith_min.locallyLipschitz.comp (hf.prod hg)


/-- The maximum of locally Lipschitz functions is locally Lipschitz. -/
protected lemma max (hf : LocallyLipschitz f) (hg : LocallyLipschitz g) :
    LocallyLipschitz (fun x => max (f x) (g x)) :=
  lipschitzWith_max.locallyLipschitz.comp (hf.prod hg)


theorem max_const (hf : LocallyLipschitz f) (a : ℝ) : LocallyLipschitz fun x => max (f x) a :=
  hf.max (LocallyLipschitz.const a)


theorem const_max (hf : LocallyLipschitz f) (a : ℝ) : LocallyLipschitz fun x => max a (f x) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : α → Real
    hf : LocallyLipschitz f
    a : Real
    ⊢ LocallyLipschitz fun x => Max.max a (f x)
  -/
  simpa [max_comm] using (hf.max_const a)
  /-
    🎉 no goals
  -/


theorem min_const (hf : LocallyLipschitz f) (a : ℝ) : LocallyLipschitz fun x => min (f x) a :=
  hf.min (LocallyLipschitz.const a)


theorem const_min (hf : LocallyLipschitz f) (a : ℝ) : LocallyLipschitz fun x => min a (f x) := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    f : α → Real
    hf : LocallyLipschitz f
    a : Real
    ⊢ LocallyLipschitz fun x => Min.min a (f x)
  -/
  simpa [min_comm] using (hf.min_const a)
  /-
    🎉 no goals
  -/


/-- If a function is locally Lipschitz around a point, then it is continuous at this point. -/
theorem continuousAt_of_locally_lipschitz {x : α} {r : ℝ} (hr : 0 < r) (K : ℝ)
    (h : ∀ y, dist y x < r → dist (f y) (f x) ≤ K * dist y x) : ContinuousAt f x := by
  -- We use `h` to squeeze `dist (f y) (f x)` between `0` and `K * dist y x`
  refine tendsto_iff_dist_tendsto_zero.2 (squeeze_zero' (Eventually.of_forall fun _ => dist_nonneg)
    (mem_of_superset (ball_mem_nhds _ hr) h) ?_)
  -- Then show that `K * dist y x` tends to zero as `y → x`
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    x : α
    r : Real
    hr : LT.lt 0 r
    K : Real
    h : ∀ (y : α), LT.lt (Dist.dist y x) r → LE.le (Dist.dist (f y) (f x)) (HMul.h …
    ⊢ Filter.Tendsto (fun a => HMul.hMul K (Dist.dist a x)) (nhds x) (nhds 0)
  -/
  refine (continuous_const.mul (continuous_id.dist continuous_const)).tendsto' _ _ ?_
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    x : α
    r : Real
    hr : LT.lt 0 r
    K : Real
    h : ∀ (y : α), LT.lt (Dist.dist y x) r → LE.le (Dist.dist (f y) (f x)) (HMul.h …
    ⊢ Eq (HMul.hMul K (Dist.dist (id x) x)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A function `f : α → ℝ` which is `K`-Lipschitz on a subset `s` admits a `K`-Lipschitz extension
to the whole space. -/
theorem LipschitzOnWith.extend_real {f : α → ℝ} {s : Set α} {K : ℝ≥0} (hf : LipschitzOnWith K f s) :
    ∃ g : α → ℝ, LipschitzWith K g ∧ EqOn f g s := by
  /- An extension is given by `g y = Inf {f x + K * dist y x | x ∈ s}`. Taking `x = y`, one has
    `g y ≤ f y` for `y ∈ s`, and the other inequality holds because `f` is `K`-Lipschitz, so that it
    can not counterbalance the growth of `K * dist y x`. One readily checks from the formula that
    the extended function is also `K`-Lipschitz. -/
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  rcases eq_empty_or_nonempty s with (rfl | hs)
    /-
      case inl
      α : Type u
      inst✝ : PseudoMetricSpace α
      f : α → Real
      K : NNReal
      hf : LipschitzOnWith K f EmptyCollection.emptyCollection
      ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g EmptyCollection.emptyC …
    -/
  · exact ⟨fun _ => 0, (LipschitzWith.const _).weaken (zero_le _), eqOn_empty _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    hs : s.Nonempty
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  have : Nonempty s := by simp only [hs, nonempty_coe_sort]
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    hs : s.Nonempty
    this : Nonempty ↑s
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  let g := fun y : α => iInf fun x : s => f x + K * dist y x
  have B : ∀ y : α, BddBelow (range fun x : s => f x + K * dist y x) := fun y => by
    rcases hs with ⟨z, hz⟩
    refine ⟨f z - K * dist y z, ?_⟩
    rintro w ⟨t, rfl⟩
    dsimp
    rw [sub_le_iff_le_add, add_assoc, ← mul_add, add_comm (dist y t)]
    calc
      f z ≤ f t + K * dist z t := hf.le_add_mul hz t.2
      _ ≤ f t + K * (dist y z + dist y t) := by gcongr; apply dist_triangle_left
  have E : EqOn f g s := fun x hx => by
    refine le_antisymm (le_ciInf fun y => hf.le_add_mul hx y.2) ?_
    simpa only [add_zero, Subtype.coe_mk, mul_zero, dist_self] using ciInf_le (B x) ⟨x, hx⟩
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    hs : s.Nonempty
    this : Nonempty ↑s
    g : α → Real := fun y => iInf fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) (Dist. …
    B : ∀ (y : α), BddBelow (Set.range fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) ( …
    E : Set.EqOn f g s
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  refine ⟨g, LipschitzWith.of_le_add_mul K fun x y => ?_, E⟩
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    hs : s.Nonempty
    this : Nonempty ↑s
    g : α → Real := fun y => iInf fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) (Dist. …
    B : ∀ (y : α), BddBelow (Set.range fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) ( …
    E : Set.EqOn f g s
    x y : α
    ⊢ LE.le (g x) (HAdd.hAdd (g y) (HMul.hMul (↑K) (Dist.dist x y)))
  -/
  rw [← sub_le_iff_le_add]
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    hs : s.Nonempty
    this : Nonempty ↑s
    g : α → Real := fun y => iInf fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) (Dist. …
    B : ∀ (y : α), BddBelow (Set.range fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) ( …
    E : Set.EqOn f g s
    x y : α
    ⊢ LE.le (HSub.hSub (g x) (HMul.hMul (↑K) (Dist.dist x y))) (g y)
  -/
  refine le_ciInf fun z => ?_
  /-
    case inr
    α : Type u
    inst✝ : PseudoMetricSpace α
    f : α → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    hs : s.Nonempty
    this : Nonempty ↑s
    g : α → Real := fun y => iInf fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) (Dist. …
    B : ∀ (y : α), BddBelow (Set.range fun x => HAdd.hAdd (f ↑x) (HMul.hMul (↑K) ( …
    E : Set.EqOn f g s
    x y : α
    z : ↑s
    ⊢ LE.le (HSub.hSub (g x) (HMul.hMul (↑K) (Dist.dist x y))) (HAdd.hAdd (f ↑z) ( …
  -/
  rw [sub_le_iff_le_add]
  calc
    g x ≤ f z + K * dist x z := ciInf_le (B x) _
    _ ≤ f z + K * dist y z + K * dist x y := by
      rw [add_assoc, ← mul_add, add_comm (dist y z)]
      gcongr
      apply dist_triangle


/-- A function `f : α → (ι → ℝ)` which is `K`-Lipschitz on a subset `s` admits a `K`-Lipschitz
extension to the whole space. The same result for the space `ℓ^∞ (ι, ℝ)` over a possibly infinite
type `ι` is implemented in `LipschitzOnWith.extend_lp_infty`. -/
theorem LipschitzOnWith.extend_pi [Fintype ι] {f : α → ι → ℝ} {s : Set α}
    {K : ℝ≥0} (hf : LipschitzOnWith K f s) : ∃ g : α → ι → ℝ, LipschitzWith K g ∧ EqOn f g s := by
  have : ∀ i, ∃ g : α → ℝ, LipschitzWith K g ∧ EqOn (fun x => f x i) g s := fun i => by
    have : LipschitzOnWith K (fun x : α => f x i) s :=
      LipschitzOnWith.of_dist_le_mul fun x hx y hy =>
        (dist_le_pi_dist _ _ i).trans (hf.dist_le_mul x hx y hy)
    exact this.extend_real
  /-
    α : Type u
    ι : Type x
    inst✝¹ : PseudoMetricSpace α
    inst✝ : Fintype ι
    f : α → ι → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    this : ∀ (i : ι), Exists fun g => And (LipschitzWith K g) (Set.EqOn (fun x =>  …
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  choose g hg using this
  /-
    α : Type u
    ι : Type x
    inst✝¹ : PseudoMetricSpace α
    inst✝ : Fintype ι
    f : α → ι → Real
    s : Set α
    K : NNReal
    hf : LipschitzOnWith K f s
    g : ι → α → Real
    hg : ∀ (i : ι), And (LipschitzWith K (g i)) (Set.EqOn (fun x => f x i) (g i) s)
    ⊢ Exists fun g => And (LipschitzWith K g) (Set.EqOn f g s)
  -/
  refine ⟨fun x i => g i x, LipschitzWith.of_dist_le_mul fun x y => ?_, fun x hx ↦ ?_⟩
    /-
      case refine_1
      α : Type u
      ι : Type x
      inst✝¹ : PseudoMetricSpace α
      inst✝ : Fintype ι
      f : α → ι → Real
      s : Set α
      K : NNReal
      hf : LipschitzOnWith K f s
      g : ι → α → Real
      hg : ∀ (i : ι), And (LipschitzWith K (g i)) (Set.EqOn (fun x => f x i) (g i) s)
      x y : α
      ⊢ LE.le (Dist.dist (fun i => g i x) fun i => g i y) (HMul.hMul (↑K) (Dist.dist …
    -/
  · exact (dist_pi_le_iff (mul_nonneg K.2 dist_nonneg)).2 fun i => (hg i).1.dist_le_mul x y
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      ι : Type x
      inst✝¹ : PseudoMetricSpace α
      inst✝ : Fintype ι
      f : α → ι → Real
      s : Set α
      K : NNReal
      hf : LipschitzOnWith K f s
      g : ι → α → Real
      hg : ∀ (i : ι), And (LipschitzWith K (g i)) (Set.EqOn (fun x => f x i) (g i) s)
      x : α
      hx : Membership.mem s x
      ⊢ Eq (f x) ((fun x i => g i x) x)
    -/
  · ext1 i
    /-
      case refine_2.h
      α : Type u
      ι : Type x
      inst✝¹ : PseudoMetricSpace α
      inst✝ : Fintype ι
      f : α → ι → Real
      s : Set α
      K : NNReal
      hf : LipschitzOnWith K f s
      g : ι → α → Real
      hg : ∀ (i : ι), And (LipschitzWith K (g i)) (Set.EqOn (fun x => f x i) (g i) s)
      x : α
      hx : Membership.mem s x
      i : ι
      ⊢ Eq (f x i) ((fun x i => g i x) x i)
    -/
    exact (hg i).2 hx
    /-
      🎉 no goals
    -/

