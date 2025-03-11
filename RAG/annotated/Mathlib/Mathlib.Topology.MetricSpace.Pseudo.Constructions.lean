/-- Pseudometric space structure pulled back by a function. -/
abbrev PseudoMetricSpace.induced {α β} (f : α → β) (m : PseudoMetricSpace β) :
    PseudoMetricSpace α where
  dist x y := dist (f x) (f y)
  dist_self _ := dist_self _
  dist_comm _ _ := dist_comm _ _
  dist_triangle _ _ _ := dist_triangle _ _ _
  edist x y := edist (f x) (f y)
  edist_dist _ _ := edist_dist _ _
  toUniformSpace := UniformSpace.comap f m.toUniformSpace
  uniformity_dist := (uniformity_basis_dist.comap _).eq_biInf
  toBornology := Bornology.induced f
  cobounded_sets := Set.ext fun s => mem_comap_iff_compl.trans <| by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      inst✝ : PseudoMetricSpace α✝
      α : Type ?u.33
      β : Type ?u.30
      f : α → β
      m : PseudoMetricSpace β
      s : Set α
      ⊢ Iff (Membership.mem (Bornology.cobounded β) (HasCompl.compl (Set.image f (Ha …
    -/
    simp only [← isBounded_def, isBounded_iff, forall_mem_image, mem_setOf]
    /-
      🎉 no goals
    -/


/-- Pull back a pseudometric space structure by an inducing map. This is a version of
`PseudoMetricSpace.induced` useful in case if the domain already has a `TopologicalSpace`
structure. -/
def Topology.IsInducing.comapPseudoMetricSpace {α β : Type*} [TopologicalSpace α]
    [m : PseudoMetricSpace β] {f : α → β} (hf : IsInducing f) : PseudoMetricSpace α :=
  .replaceTopology (.induced f m) hf.eq_induced


@[deprecated (since := "2024-10-28")]
alias Inducing.comapPseudoMetricSpace := IsInducing.comapPseudoMetricSpace


/-- Pull back a pseudometric space structure by a uniform inducing map. This is a version of
`PseudoMetricSpace.induced` useful in case if the domain already has a `UniformSpace`
structure. -/
def IsUniformInducing.comapPseudoMetricSpace {α β} [UniformSpace α] [m : PseudoMetricSpace β]
    (f : α → β) (h : IsUniformInducing f) : PseudoMetricSpace α :=
  .replaceUniformity (.induced f m) h.comap_uniformity.symm


@[deprecated (since := "2024-10-08")] alias UniformInducing.comapPseudoMetricSpace :=
  IsUniformInducing.comapPseudoMetricSpace


instance Subtype.pseudoMetricSpace {p : α → Prop} : PseudoMetricSpace (Subtype p) :=
  PseudoMetricSpace.induced Subtype.val ‹_›


lemma Subtype.dist_eq {p : α → Prop} (x y : Subtype p) : dist x y = dist (x : α) y := rfl


lemma Subtype.nndist_eq {p : α → Prop} (x y : Subtype p) : nndist x y = nndist (x : α) y := rfl


@[to_additive]
instance instPseudoMetricSpace : PseudoMetricSpace αᵐᵒᵖ :=
  PseudoMetricSpace.induced MulOpposite.unop ‹_›


@[to_additive (attr := simp)]
lemma dist_unop (x y : αᵐᵒᵖ) : dist (unop x) (unop y) = dist x y := rfl


@[to_additive (attr := simp)]
lemma dist_op (x y : α) : dist (op x) (op y) = dist x y := rfl


@[to_additive (attr := simp)]
lemma nndist_unop (x y : αᵐᵒᵖ) : nndist (unop x) (unop y) = nndist x y := rfl


@[to_additive (attr := simp)]
lemma nndist_op (x y : α) : nndist (op x) (op y) = nndist x y := rfl


instance : PseudoMetricSpace ℝ≥0 := Subtype.pseudoMetricSpace


lemma NNReal.dist_eq (a b : ℝ≥0) : dist a b = |(a : ℝ) - b| := rfl


lemma NNReal.nndist_eq (a b : ℝ≥0) : nndist a b = max (a - b) (b - a) :=
  eq_of_forall_ge_iff fun _ => by
    /-
      a b x✝ : NNReal
      ⊢ Iff (LE.le (NNDist.nndist a b) x✝) (LE.le (Max.max (HSub.hSub a b) (HSub.hSu …
    -/
    simp only [max_le_iff, tsub_le_iff_right (α := ℝ≥0)]
    simp only [← NNReal.coe_le_coe, coe_nndist, dist_eq, abs_sub_le_iff,
      tsub_le_iff_right, NNReal.coe_add]


@[simp]
lemma NNReal.nndist_zero_eq_val (z : ℝ≥0) : nndist 0 z = z := by
  /-
    z : NNReal
    ⊢ Eq (NNDist.nndist 0 z) z
  -/
  simp only [NNReal.nndist_eq, max_eq_right, tsub_zero, zero_tsub, zero_le']
  /-
    🎉 no goals
  -/


@[simp]
lemma NNReal.nndist_zero_eq_val' (z : ℝ≥0) : nndist z 0 = z := by
  /-
    z : NNReal
    ⊢ Eq (NNDist.nndist z 0) z
  -/
  rw [nndist_comm]
  /-
    z : NNReal
    ⊢ Eq (NNDist.nndist 0 z) z
  -/
  exact NNReal.nndist_zero_eq_val z
  /-
    🎉 no goals
  -/


lemma NNReal.le_add_nndist (a b : ℝ≥0) : a ≤ b + nndist a b := by
  suffices (a : ℝ) ≤ (b : ℝ) + dist a b by
    rwa [← NNReal.coe_le_coe, NNReal.coe_add, coe_nndist]
  /-
    a b : NNReal
    ⊢ LE.le (↑a) (HAdd.hAdd (↑b) (Dist.dist a b))
  -/
  rw [← sub_le_iff_le_add']
  /-
    a b : NNReal
    ⊢ LE.le (HSub.hSub ↑a ↑b) (Dist.dist a b)
  -/
  exact le_of_abs_le (dist_eq a b).ge
  /-
    🎉 no goals
  -/


lemma NNReal.ball_zero_eq_Ico' (c : ℝ≥0) :
                                                       /-
                                                         c : NNReal
                                                         ⊢ Eq (Metric.ball 0 ↑c) (Set.Ico 0 c)
                                                       -/
    Metric.ball (0 : ℝ≥0) c.toReal = Set.Ico 0 c := by ext x; simp
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma NNReal.ball_zero_eq_Ico (c : ℝ) :
    Metric.ball (0 : ℝ≥0) c = Set.Ico 0 c.toNNReal := by
  /-
    c : Real
    ⊢ Eq (Metric.ball 0 c) (Set.Ico 0 c.toNNReal)
  -/
  by_cases c_pos : 0 < c
    /-
      case pos
      c : Real
      c_pos : LT.lt 0 c
      ⊢ Eq (Metric.ball 0 c) (Set.Ico 0 c.toNNReal)
    -/
  · convert NNReal.ball_zero_eq_Ico' ⟨c, c_pos.le⟩
    /-
      case h.e'_3.h.e'_4
      c : Real
      c_pos : LT.lt 0 c
      ⊢ Eq c.toNNReal ⟨c, ⋯⟩
    -/
    simp [Real.toNNReal, c_pos.le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    c : Real
    c_pos : Not (LT.lt 0 c)
    ⊢ Eq (Metric.ball 0 c) (Set.Ico 0 c.toNNReal)
  -/
  simp [not_lt.mp c_pos]
  /-
    🎉 no goals
  -/


lemma NNReal.closedBall_zero_eq_Icc' (c : ℝ≥0) :
                                                             /-
                                                               c : NNReal
                                                               ⊢ Eq (Metric.closedBall 0 ↑c) (Set.Icc 0 c)
                                                             -/
    Metric.closedBall (0 : ℝ≥0) c.toReal = Set.Icc 0 c := by ext x; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma NNReal.closedBall_zero_eq_Icc {c : ℝ} (c_nn : 0 ≤ c) :
    Metric.closedBall (0 : ℝ≥0) c = Set.Icc 0 c.toNNReal := by
  /-
    c : Real
    c_nn : LE.le 0 c
    ⊢ Eq (Metric.closedBall 0 c) (Set.Icc 0 c.toNNReal)
  -/
  convert NNReal.closedBall_zero_eq_Icc' ⟨c, c_nn⟩
  /-
    case h.e'_3.h.e'_4
    c : Real
    c_nn : LE.le 0 c
    ⊢ Eq c.toNNReal ⟨c, c_nn⟩
  -/
  simp [Real.toNNReal, c_nn]
  /-
    🎉 no goals
  -/


instance : PseudoMetricSpace (ULift β) := PseudoMetricSpace.induced ULift.down ‹_›


lemma dist_eq (x y : ULift β) : dist x y = dist x.down y.down := rfl


lemma nndist_eq (x y : ULift β) : nndist x y = nndist x.down y.down := rfl


@[simp] lemma dist_up_up (x y : β) : dist (ULift.up x) (ULift.up y) = dist x y := rfl


@[simp] lemma nndist_up_up (x y : β) : nndist (ULift.up x) (ULift.up y) = nndist x y := rfl


instance Prod.pseudoMetricSpaceMax : PseudoMetricSpace (α × β) :=
  let i := PseudoEMetricSpace.toPseudoMetricSpaceOfDist
    (fun x y : α × β => dist x.1 y.1 ⊔ dist x.2 y.2)
    (fun _ _ => (max_lt (edist_lt_top _ _) (edist_lt_top _ _)).ne) fun x y => by
      simp only [dist_edist, ← ENNReal.toReal_max (edist_ne_top _ _) (edist_ne_top _ _),
        Prod.edist_eq]
  i.replaceBornology fun s => by
    simp only [← isBounded_image_fst_and_snd, isBounded_iff_eventually, forall_mem_image, ←
      eventually_and, ← forall_and, ← max_le_iff]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      i : PseudoMetricSpace (Prod α β) := PseudoEMetricSpace.toPseudoMetricSpaceOfDi …
      s : Set (Prod α β)
      ⊢ Iff (Filter.Eventually (fun x => ∀ (x_1 : Prod α β), Membership.mem s x_1 →  …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma Prod.dist_eq {x y : α × β} : dist x y = max (dist x.1 y.1) (dist x.2 y.2) := rfl


@[simp]
lemma dist_prod_same_left {x : α} {y₁ y₂ : β} : dist (x, y₁) (x, y₂) = dist y₁ y₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    x : α
    y₁ y₂ : β
    ⊢ Eq (Dist.dist { fst := x, snd := y₁ } { fst := x, snd := y₂ }) (Dist.dist y₁ …
  -/
  simp [Prod.dist_eq, dist_nonneg]
  /-
    🎉 no goals
  -/


@[simp]
lemma dist_prod_same_right {x₁ x₂ : α} {y : β} : dist (x₁, y) (x₂, y) = dist x₁ x₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    x₁ x₂ : α
    y : β
    ⊢ Eq (Dist.dist { fst := x₁, snd := y } { fst := x₂, snd := y }) (Dist.dist x₁ …
  -/
  simp [Prod.dist_eq, dist_nonneg]
  /-
    🎉 no goals
  -/


lemma ball_prod_same (x : α) (y : β) (r : ℝ) : ball x r ×ˢ ball y r = ball (x, y) r :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝¹ : PseudoMetricSpace α
                    inst✝ : PseudoMetricSpace β
                    x : α
                    y : β
                    r : Real
                    z : Prod α β
                    ⊢ Iff (Membership.mem (SProd.sprod (Metric.ball x r) (Metric.ball y r)) z) (Me …
                  -/
  ext fun z => by simp [Prod.dist_eq]
                  /-
                    🎉 no goals
                  -/


lemma closedBall_prod_same (x : α) (y : β) (r : ℝ) :
                                                                              /-
                                                                                α : Type u_1
                                                                                β : Type u_2
                                                                                inst✝¹ : PseudoMetricSpace α
                                                                                inst✝ : PseudoMetricSpace β
                                                                                x : α
                                                                                y : β
                                                                                r : Real
                                                                                z : Prod α β
                                                                                ⊢ Iff (Membership.mem (SProd.sprod (Metric.closedBall x r) (Metric.closedBall  …
                                                                              -/
    closedBall x r ×ˢ closedBall y r = closedBall (x, y) r := ext fun z => by simp [Prod.dist_eq]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


lemma sphere_prod (x : α × β) (r : ℝ) :
    sphere x r = sphere x.1 r ×ˢ closedBall x.2 r ∪ closedBall x.1 r ×ˢ sphere x.2 r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    x : Prod α β
    r : Real
    ⊢ Eq (Metric.sphere x r) (Union.union (SProd.sprod (Metric.sphere x.1 r) (Metr …
  -/
  obtain hr | rfl | hr := lt_trichotomy r 0
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x : Prod α β
      r : Real
      hr : LT.lt r 0
      ⊢ Eq (Metric.sphere x r) (Union.union (SProd.sprod (Metric.sphere x.1 r) (Metr …
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x : Prod α β
      ⊢ Eq (Metric.sphere x 0) (Union.union (SProd.sprod (Metric.sphere x.1 0) (Metr …
    -/
  · cases x
    /-
      case inr.inl.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      fst✝ : α
      snd✝ : β
      ⊢ Eq (Metric.sphere { fst := fst✝, snd := snd✝ } 0) (Union.union (SProd.sprod  …
    -/
    simp_rw [← closedBall_eq_sphere_of_nonpos le_rfl, union_self, closedBall_prod_same]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x : Prod α β
      r : Real
      hr : LT.lt 0 r
      ⊢ Eq (Metric.sphere x r) (Union.union (SProd.sprod (Metric.sphere x.1 r) (Metr …
    -/
  · ext ⟨x', y'⟩
    simp_rw [Set.mem_union, Set.mem_prod, Metric.mem_closedBall, Metric.mem_sphere, Prod.dist_eq,
      max_eq_iff]
    /-
      case inr.inr.h.mk
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x : Prod α β
      r : Real
      hr : LT.lt 0 r
      x' : α
      y' : β
      ⊢ Iff (Or (And (Eq (Dist.dist x' x.1) r) (LE.le (Dist.dist y' x.2) (Dist.dist  …
    -/
    refine or_congr (and_congr_right ?_) (and_comm.trans (and_congr_left ?_))
    /-
      case inr.inr.h.mk.refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x : Prod α β
      r : Real
      hr : LT.lt 0 r
      x' : α
      y' : β
      ⊢ Eq (Dist.dist x' x.1) r → Iff (LE.le (Dist.dist y' x.2) (Dist.dist x' x.1))  …
    -/
    all_goals rintro rfl; rfl
    /-
      🎉 no goals
    -/


lemma uniformContinuous_dist : UniformContinuous fun p : α × α => dist p.1 p.2 :=
  Metric.uniformContinuous_iff.2 fun ε ε0 =>
    ⟨ε / 2, half_pos ε0, fun {a b} h =>
      calc dist (dist a.1 a.2) (dist b.1 b.2) ≤ dist a.1 b.1 + dist a.2 b.2 :=
        dist_dist_dist_le _ _ _ _
      _ ≤ dist a b + dist a b := add_le_add (le_max_left _ _) (le_max_right _ _)
      _ < ε / 2 + ε / 2 := add_lt_add h h
      _ = ε := add_halves ε⟩


protected lemma UniformContinuous.dist [UniformSpace β] {f g : β → α} (hf : UniformContinuous f)
    (hg : UniformContinuous g) : UniformContinuous fun b => dist (f b) (g b) :=
  uniformContinuous_dist.comp (hf.prod_mk hg)


@[continuity]
lemma continuous_dist : Continuous fun p : α × α ↦ dist p.1 p.2 := uniformContinuous_dist.continuous


@[continuity, fun_prop]
protected lemma Continuous.dist [TopologicalSpace β] {f g : β → α} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun b => dist (f b) (g b) :=
  continuous_dist.comp (hf.prod_mk hg : _)


protected lemma Filter.Tendsto.dist {f g : β → α} {x : Filter β} {a b : α}
    (hf : Tendsto f x (𝓝 a)) (hg : Tendsto g x (𝓝 b)) :
    Tendsto (fun x => dist (f x) (g x)) x (𝓝 (dist a b)) :=
  (continuous_dist.tendsto (a, b)).comp (hf.prod_mk_nhds hg)


lemma continuous_iff_continuous_dist [TopologicalSpace β] {f : β → α} :
    Continuous f ↔ Continuous fun x : β × β => dist (f x.1) (f x.2) :=
  ⟨fun h => h.fst'.dist h.snd', fun h =>
    continuous_iff_continuousAt.2 fun _ => tendsto_iff_dist_tendsto_zero.2 <|
      (h.comp (continuous_id.prod_mk continuous_const)).tendsto' _ _ <| dist_self _⟩


lemma uniformContinuous_nndist : UniformContinuous fun p : α × α => nndist p.1 p.2 :=
  uniformContinuous_dist.subtype_mk _


protected lemma UniformContinuous.nndist [UniformSpace β] {f g : β → α} (hf : UniformContinuous f)
    (hg : UniformContinuous g) : UniformContinuous fun b => nndist (f b) (g b) :=
  uniformContinuous_nndist.comp (hf.prod_mk hg)


lemma continuous_nndist : Continuous fun p : α × α => nndist p.1 p.2 :=
  uniformContinuous_nndist.continuous


@[fun_prop]
protected lemma Continuous.nndist [TopologicalSpace β] {f g : β → α} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun b => nndist (f b) (g b) :=
  continuous_nndist.comp (hf.prod_mk hg : _)


protected lemma Filter.Tendsto.nndist {f g : β → α} {x : Filter β} {a b : α}
    (hf : Tendsto f x (𝓝 a)) (hg : Tendsto g x (𝓝 b)) :
    Tendsto (fun x => nndist (f x) (g x)) x (𝓝 (nndist a b)) :=
  (continuous_nndist.tendsto (a, b)).comp (hf.prod_mk_nhds hg)

