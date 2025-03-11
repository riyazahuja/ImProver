variable (I) in
/-- Given a smooth manifold modelled on a finite dimensional space `E`,
`f : SmoothBumpFunction I M` is a smooth function on `M` such that in the extended chart `e` at
`f.c`:

* `f x = 1` in the closed ball of radius `f.rIn` centered at `f.c`;
* `f x = 0` outside of the ball of radius `f.rOut` centered at `f.c`;
* `0 ≤ f x ≤ 1` for all `x`.

The structure contains data required to construct a function with these properties. The function is
available as `⇑f` or `f x`. Formal statements of the properties listed above involve some
(pre)images under `extChartAt I f.c` and are given as lemmas in the `SmoothBumpFunction`
namespace. -/
structure SmoothBumpFunction (c : M) extends ContDiffBump (extChartAt I c c) where
  closedBall_subset : closedBall (extChartAt I c c) rOut ∩ range I ⊆ (extChartAt I c).target


/-- The function defined by `f : SmoothBumpFunction c`. Use automatic coercion to function
instead. -/
@[coe] def toFun : M → ℝ :=
  indicator (chartAt H c).source (f.toContDiffBump ∘ extChartAt I c)


instance : CoeFun (SmoothBumpFunction I c) fun _ => M → ℝ :=
  ⟨toFun⟩


theorem coe_def : ⇑f = indicator (chartAt H c).source (f.toContDiffBump ∘ extChartAt I c) :=
  rfl


theorem rOut_pos : 0 < f.rOut :=
  f.toContDiffBump.rOut_pos


theorem ball_subset : ball (extChartAt I c c) f.rOut ∩ range I ⊆ (extChartAt I c).target :=
  Subset.trans (inter_subset_inter_left _ ball_subset_closedBall) f.closedBall_subset


theorem ball_inter_range_eq_ball_inter_target :
    ball (extChartAt I c c) f.rOut ∩ range I =
      ball (extChartAt I c c) f.rOut ∩ (extChartAt I c).target :=
  (subset_inter inter_subset_left f.ball_subset).antisymm <| inter_subset_inter_right _ <|
    extChartAt_target_subset_range _


theorem eqOn_source : EqOn f (f.toContDiffBump ∘ extChartAt I c) (chartAt H c).source :=
  eqOn_indicator


theorem eventuallyEq_of_mem_source (hx : x ∈ (chartAt H c).source) :
    f =ᶠ[𝓝 x] f.toContDiffBump ∘ extChartAt I c :=
  f.eqOn_source.eventuallyEq_of_mem <| (chartAt H c).open_source.mem_nhds hx


theorem one_of_dist_le (hs : x ∈ (chartAt H c).source)
    (hd : dist (extChartAt I c x) (extChartAt I c c) ≤ f.rIn) : f x = 1 := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    hs : Membership.mem (chartAt H c).source x
    hd : LE.le (Dist.dist (↑(extChartAt I c) x) (↑(extChartAt I c) c)) f.rIn
    ⊢ Eq (↑f x) 1
  -/
  simp only [f.eqOn_source hs, (· ∘ ·), f.one_of_mem_closedBall hd]
  /-
    🎉 no goals
  -/


theorem support_eq_inter_preimage :
    support f = (chartAt H c).source ∩ extChartAt I c ⁻¹' ball (extChartAt I c c) f.rOut := by
  rw [coe_def, support_indicator, support_comp_eq_preimage, ← extChartAt_source I,
    ← (extChartAt I c).symm_image_target_inter_eq', ← (extChartAt I c).symm_image_target_inter_eq',
    f.support_eq]


theorem isOpen_support : IsOpen (support f) := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    ⊢ IsOpen (Function.support ↑f)
  -/
  rw [support_eq_inter_preimage]
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    ⊢ IsOpen (Inter.inter (chartAt H c).source (Set.preimage (↑(extChartAt I c)) ( …
  -/
  exact isOpen_extChartAt_preimage c isOpen_ball
  /-
    🎉 no goals
  -/


theorem support_eq_symm_image :
    support f = (extChartAt I c).symm '' (ball (extChartAt I c c) f.rOut ∩ range I) := by
  rw [f.support_eq_inter_preimage, ← extChartAt_source I,
    ← (extChartAt I c).symm_image_target_inter_eq', inter_comm,
    ball_inter_range_eq_ball_inter_target]


theorem support_subset_source : support f ⊆ (chartAt H c).source := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    ⊢ HasSubset.Subset (Function.support ↑f) (chartAt H c).source
  -/
  rw [f.support_eq_inter_preimage, ← extChartAt_source I]; exact inter_subset_left
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem image_eq_inter_preimage_of_subset_support {s : Set M} (hs : s ⊆ support f) :
    extChartAt I c '' s =
      closedBall (extChartAt I c c) f.rOut ∩ range I ∩ (extChartAt I c).symm ⁻¹' s := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hs : HasSubset.Subset s (Function.support ↑f)
    ⊢ Eq (Set.image (↑(extChartAt I c)) s) (Inter.inter (Inter.inter (Metric.close …
  -/
  rw [support_eq_inter_preimage, subset_inter_iff, ← extChartAt_source I, ← image_subset_iff] at hs
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hs : And (HasSubset.Subset s (extChartAt I c).source) (HasSubset.Subset (Set.i …
    ⊢ Eq (Set.image (↑(extChartAt I c)) s) (Inter.inter (Inter.inter (Metric.close …
  -/
  cases' hs with hse hsf
  /-
    case intro
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hse : HasSubset.Subset s (extChartAt I c).source
    hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
    ⊢ Eq (Set.image (↑(extChartAt I c)) s) (Inter.inter (Inter.inter (Metric.close …
  -/
  apply Subset.antisymm
    /-
      case intro.h₁
      E : Type uE
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      H : Type uH
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      c : M
      f : SmoothBumpFunction I c
      inst✝ : FiniteDimensional Real E
      s : Set M
      hse : HasSubset.Subset s (extChartAt I c).source
      hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
      ⊢ HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Inter.inter (Inter.inter …
    -/
  · refine subset_inter (subset_inter (hsf.trans ball_subset_closedBall) ?_) ?_
      /-
        case intro.h₁.refine_1
        E : Type uE
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        H : Type uH
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        c : M
        f : SmoothBumpFunction I c
        inst✝ : FiniteDimensional Real E
        s : Set M
        hse : HasSubset.Subset s (extChartAt I c).source
        hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
        ⊢ HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Set.range ↑I)
      -/
    · rintro _ ⟨x, -, rfl⟩; exact mem_range_self _
                            /-
                              🎉 no goals
                            -/
      /-
        case intro.h₁.refine_2
        E : Type uE
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        H : Type uH
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        c : M
        f : SmoothBumpFunction I c
        inst✝ : FiniteDimensional Real E
        s : Set M
        hse : HasSubset.Subset s (extChartAt I c).source
        hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
        ⊢ HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Set.preimage (↑(extChart …
      -/
    · rw [(extChartAt I c).image_eq_target_inter_inv_preimage hse]
      /-
        case intro.h₁.refine_2
        E : Type uE
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        H : Type uH
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝² : TopologicalSpace M
        inst✝¹ : ChartedSpace H M
        c : M
        f : SmoothBumpFunction I c
        inst✝ : FiniteDimensional Real E
        s : Set M
        hse : HasSubset.Subset s (extChartAt I c).source
        hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
        ⊢ HasSubset.Subset (Inter.inter (extChartAt I c).target (Set.preimage (↑(extCh …
      -/
      exact inter_subset_right
      /-
        🎉 no goals
      -/
    /-
      case intro.h₂
      E : Type uE
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      H : Type uH
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      c : M
      f : SmoothBumpFunction I c
      inst✝ : FiniteDimensional Real E
      s : Set M
      hse : HasSubset.Subset s (extChartAt I c).source
      hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
      ⊢ HasSubset.Subset (Inter.inter (Inter.inter (Metric.closedBall (↑(extChartAt  …
    -/
  · refine Subset.trans (inter_subset_inter_left _ f.closedBall_subset) ?_
    /-
      case intro.h₂
      E : Type uE
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      H : Type uH
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      c : M
      f : SmoothBumpFunction I c
      inst✝ : FiniteDimensional Real E
      s : Set M
      hse : HasSubset.Subset s (extChartAt I c).source
      hsf : HasSubset.Subset (Set.image (↑(extChartAt I c)) s) (Metric.ball (↑(extCh …
      ⊢ HasSubset.Subset (Inter.inter (extChartAt I c).target (Set.preimage (↑(extCh …
    -/
    rw [(extChartAt I c).image_eq_target_inter_inv_preimage hse]
    /-
      🎉 no goals
    -/


theorem mem_Icc : f x ∈ Icc (0 : ℝ) 1 := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    ⊢ Membership.mem (Set.Icc 0 1) (↑f x)
  -/
  have : f x = 0 ∨ f x = _ := indicator_eq_zero_or_self _ _ _
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    this : Or (Eq (↑f x) 0) (Eq (↑f x) (Function.comp (↑f.toContDiffBump) (↑(extCh …
    ⊢ Membership.mem (Set.Icc 0 1) (↑f x)
  -/
  cases' this with h h <;> rw [h]
  /-
    case inl
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    h : Eq (↑f x) 0
    ⊢ Membership.mem (Set.Icc 0 1) 0
  -/
  exacts [left_mem_Icc.2 zero_le_one, ⟨f.nonneg, f.le_one⟩]
  /-
    🎉 no goals
  -/


theorem nonneg : 0 ≤ f x :=
  f.mem_Icc.1


theorem le_one : f x ≤ 1 :=
  f.mem_Icc.2


theorem eventuallyEq_one_of_dist_lt (hs : x ∈ (chartAt H c).source)
    (hd : dist (extChartAt I c x) (extChartAt I c c) < f.rIn) : f =ᶠ[𝓝 x] 1 := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    hs : Membership.mem (chartAt H c).source x
    hd : LT.lt (Dist.dist (↑(extChartAt I c) x) (↑(extChartAt I c) c)) f.rIn
    ⊢ (nhds x).EventuallyEq (↑f) 1
  -/
  filter_upwards [IsOpen.mem_nhds (isOpen_extChartAt_preimage c isOpen_ball) ⟨hs, hd⟩]
  /-
    case h
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    hs : Membership.mem (chartAt H c).source x
    hd : LT.lt (Dist.dist (↑(extChartAt I c) x) (↑(extChartAt I c) c)) f.rIn
    ⊢ ∀ (a : M), Membership.mem (Inter.inter (chartAt H c).source (Set.preimage (↑ …
  -/
  rintro z ⟨hzs, hzd⟩
  /-
    case h.intro
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    x : M
    inst✝ : FiniteDimensional Real E
    hs : Membership.mem (chartAt H c).source x
    hd : LT.lt (Dist.dist (↑(extChartAt I c) x) (↑(extChartAt I c) c)) f.rIn
    z : M
    hzs : Membership.mem (chartAt H c).source z
    hzd : Membership.mem (Set.preimage (↑(extChartAt I c)) (Metric.ball (↑(extChar …
    ⊢ Eq (↑f z) (1 z)
  -/
  exact f.one_of_dist_le hzs <| le_of_lt hzd
  /-
    🎉 no goals
  -/


theorem eventuallyEq_one : f =ᶠ[𝓝 c] 1 :=
                                                             /-
                                                               E : Type uE
                                                               inst✝⁵ : NormedAddCommGroup E
                                                               inst✝⁴ : NormedSpace Real E
                                                               H : Type uH
                                                               inst✝³ : TopologicalSpace H
                                                               I : ModelWithCorners Real E H
                                                               M : Type uM
                                                               inst✝² : TopologicalSpace M
                                                               inst✝¹ : ChartedSpace H M
                                                               c : M
                                                               f : SmoothBumpFunction I c
                                                               inst✝ : FiniteDimensional Real E
                                                               ⊢ LT.lt (Dist.dist (↑(extChartAt I c) c) (↑(extChartAt I c) c)) f.rIn
                                                             -/
  f.eventuallyEq_one_of_dist_lt (mem_chart_source _ _) <| by rw [dist_self]; exact f.rIn_pos
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem eq_one : f c = 1 :=
  f.eventuallyEq_one.eq_of_nhds


theorem support_mem_nhds : support f ∈ 𝓝 c :=
                                         /-
                                           E : Type uE
                                           inst✝⁵ : NormedAddCommGroup E
                                           inst✝⁴ : NormedSpace Real E
                                           H : Type uH
                                           inst✝³ : TopologicalSpace H
                                           I : ModelWithCorners Real E H
                                           M : Type uM
                                           inst✝² : TopologicalSpace M
                                           inst✝¹ : ChartedSpace H M
                                           c : M
                                           f : SmoothBumpFunction I c
                                           inst✝ : FiniteDimensional Real E
                                           x : M
                                           hx : Eq (↑f x) (1 x)
                                           ⊢ Ne (↑f x) 0
                                         -/
  f.eventuallyEq_one.mono fun x hx => by rw [hx]; exact one_ne_zero
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem tsupport_mem_nhds : tsupport f ∈ 𝓝 c :=
  mem_of_superset f.support_mem_nhds subset_closure


theorem c_mem_support : c ∈ support f :=
  mem_of_mem_nhds f.support_mem_nhds


theorem nonempty_support : (support f).Nonempty :=
  ⟨c, f.c_mem_support⟩


theorem isCompact_symm_image_closedBall :
    IsCompact ((extChartAt I c).symm '' (closedBall (extChartAt I c c) f.rOut ∩ range I)) :=
  ((isCompact_closedBall _ _).inter_right I.isClosed_range).image_of_continuousOn <|
    (continuousOn_extChartAt_symm _).mono f.closedBall_subset


/-- Given a smooth bump function `f : SmoothBumpFunction I c`, the closed ball of radius `f.R` is
known to include the support of `f`. These closed balls (in the model normed space `E`) intersected
with `Set.range I` form a basis of `𝓝[range I] (extChartAt I c c)`. -/
theorem nhdsWithin_range_basis :
    (𝓝[range I] extChartAt I c c).HasBasis (fun _ : SmoothBumpFunction I c => True) fun f =>
      closedBall (extChartAt I c c) f.rOut ∩ range I := by
  refine ((nhdsWithin_hasBasis nhds_basis_closedBall _).restrict_subset
    (extChartAt_target_mem_nhdsWithin _)).to_hasBasis' ?_ ?_
    /-
      case refine_1
      E : Type uE
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type uH
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      c : M
      ⊢ ∀ (i : Real), And (LT.lt 0 i) (HasSubset.Subset (Inter.inter (Metric.closedB …
    -/
  · rintro R ⟨hR0, hsub⟩
    /-
      case refine_1.intro
      E : Type uE
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type uH
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      c : M
      R : Real
      hR0 : LT.lt 0 R
      hsub : HasSubset.Subset (Inter.inter (Metric.closedBall (↑(extChartAt I c) c)  …
      ⊢ Exists fun i' => And True (HasSubset.Subset (Inter.inter (Metric.closedBall  …
    -/
    exact ⟨⟨⟨R / 2, R, half_pos hR0, half_lt_self hR0⟩, hsub⟩, trivial, Subset.rfl⟩
    /-
      🎉 no goals
    -/
  · exact fun f _ => inter_mem (mem_nhdsWithin_of_mem_nhds <| closedBall_mem_nhds _ f.rOut_pos)
      self_mem_nhdsWithin


theorem isClosed_image_of_isClosed {s : Set M} (hsc : IsClosed s) (hs : s ⊆ support f) :
    IsClosed (extChartAt I c '' s) := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : HasSubset.Subset s (Function.support ↑f)
    ⊢ IsClosed (Set.image (↑(extChartAt I c)) s)
  -/
  rw [f.image_eq_inter_preimage_of_subset_support hs]
  refine ContinuousOn.preimage_isClosed_of_isClosed
    ((continuousOn_extChartAt_symm _).mono f.closedBall_subset) ?_ hsc
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : HasSubset.Subset s (Function.support ↑f)
    ⊢ IsClosed (Inter.inter (Metric.closedBall (↑(extChartAt I c) c) f.rOut) (Set. …
  -/
  exact IsClosed.inter isClosed_ball I.isClosed_range
  /-
    🎉 no goals
  -/


/-- If `f` is a smooth bump function and `s` closed subset of the support of `f` (i.e., of the open
ball of radius `f.rOut`), then there exists `0 < r < f.rOut` such that `s` is a subset of the open
ball of radius `r`. Formally, `s ⊆ e.source ∩ e ⁻¹' (ball (e c) r)`, where `e = extChartAt I c`. -/
theorem exists_r_pos_lt_subset_ball {s : Set M} (hsc : IsClosed s) (hs : s ⊆ support f) :
    ∃ r ∈ Ioo 0 f.rOut,
      s ⊆ (chartAt H c).source ∩ extChartAt I c ⁻¹' ball (extChartAt I c c) r := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : HasSubset.Subset s (Function.support ↑f)
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 f.rOut) r) (HasSubset.Subset  …
  -/
  set e := extChartAt I c
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : HasSubset.Subset s (Function.support ↑f)
    e : PartialEquiv M E := extChartAt I c
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 f.rOut) r) (HasSubset.Subset  …
  -/
  have : IsClosed (e '' s) := f.isClosed_image_of_isClosed hsc hs
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : HasSubset.Subset s (Function.support ↑f)
    e : PartialEquiv M E := extChartAt I c
    this : IsClosed (Set.image (↑e) s)
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 f.rOut) r) (HasSubset.Subset  …
  -/
  rw [support_eq_inter_preimage, subset_inter_iff, ← image_subset_iff] at hs
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : And (HasSubset.Subset s (chartAt H c).source) (HasSubset.Subset (Set.imag …
    e : PartialEquiv M E := extChartAt I c
    this : IsClosed (Set.image (↑e) s)
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 f.rOut) r) (HasSubset.Subset  …
  -/
  rcases exists_pos_lt_subset_ball f.rOut_pos this hs.2 with ⟨r, hrR, hr⟩
  /-
    case intro.intro
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    s : Set M
    hsc : IsClosed s
    hs : And (HasSubset.Subset s (chartAt H c).source) (HasSubset.Subset (Set.imag …
    e : PartialEquiv M E := extChartAt I c
    this : IsClosed (Set.image (↑e) s)
    r : Real
    hrR : Membership.mem (Set.Ioo 0 f.rOut) r
    hr : HasSubset.Subset (Set.image (↑e) s) (Metric.ball (↑(extChartAt I c) c) r)
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 f.rOut) r) (HasSubset.Subset  …
  -/
  exact ⟨r, hrR, subset_inter hs.1 (image_subset_iff.1 hr)⟩
  /-
    🎉 no goals
  -/


/-- Replace `rIn` with another value in the interval `(0, f.rOut)`. -/
@[simps rOut rIn]
def updateRIn (r : ℝ) (hr : r ∈ Ioo 0 f.rOut) : SmoothBumpFunction I c :=
  ⟨⟨r, f.rOut, hr.1, hr.2⟩, f.closedBall_subset⟩


@[simp]
theorem support_updateRIn {r : ℝ} (hr : r ∈ Ioo 0 f.rOut) :
    support (f.updateRIn r hr) = support f := by
  /-
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝ : FiniteDimensional Real E
    r : Real
    hr : Membership.mem (Set.Ioo 0 f.rOut) r
    ⊢ Eq (Function.support ↑(f.updateRIn r hr)) (Function.support ↑f)
  -/
  simp only [support_eq_inter_preimage, updateRIn_rOut]
  /-
    🎉 no goals
  -/

-- Porting note: was an `Inhabited` instance

instance : Nonempty (SmoothBumpFunction I c) := nhdsWithin_range_basis.nonempty


theorem isClosed_symm_image_closedBall :
    IsClosed ((extChartAt I c).symm '' (closedBall (extChartAt I c c) f.rOut ∩ range I)) :=
  f.isCompact_symm_image_closedBall.isClosed


theorem tsupport_subset_symm_image_closedBall :
    tsupport f ⊆ (extChartAt I c).symm '' (closedBall (extChartAt I c c) f.rOut ∩ range I) := by
  /-
    E : Type uE
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type uH
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝¹ : FiniteDimensional Real E
    inst✝ : T2Space M
    ⊢ HasSubset.Subset (tsupport ↑f) (Set.image (↑(extChartAt I c).symm) (Inter.in …
  -/
  rw [tsupport, support_eq_symm_image]
  exact closure_minimal (image_subset _ <| inter_subset_inter_left _ ball_subset_closedBall)
    f.isClosed_symm_image_closedBall


theorem tsupport_subset_extChartAt_source : tsupport f ⊆ (extChartAt I c).source :=
  calc
    tsupport f ⊆ (extChartAt I c).symm '' (closedBall (extChartAt I c c) f.rOut ∩ range I) :=
      f.tsupport_subset_symm_image_closedBall
    _ ⊆ (extChartAt I c).symm '' (extChartAt I c).target := image_subset _ f.closedBall_subset
    _ = (extChartAt I c).source := (extChartAt I c).symm_image_target_eq_source


theorem tsupport_subset_chartAt_source : tsupport f ⊆ (chartAt H c).source := by
  /-
    E : Type uE
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    H : Type uH
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝¹ : FiniteDimensional Real E
    inst✝ : T2Space M
    ⊢ HasSubset.Subset (tsupport ↑f) (chartAt H c).source
  -/
  simpa only [extChartAt_source] using f.tsupport_subset_extChartAt_source
  /-
    🎉 no goals
  -/


protected theorem hasCompactSupport : HasCompactSupport f :=
  f.isCompact_symm_image_closedBall.of_isClosed_subset isClosed_closure
    f.tsupport_subset_symm_image_closedBall


variable (c) in
/-- The closures of supports of smooth bump functions centered at `c` form a basis of `𝓝 c`.
In other words, each of these closures is a neighborhood of `c` and each neighborhood of `c`
includes `tsupport f` for some `f : SmoothBumpFunction I c`. -/
theorem nhds_basis_tsupport :
    (𝓝 c).HasBasis (fun _ : SmoothBumpFunction I c => True) fun f => tsupport f := by
  have :
    (𝓝 c).HasBasis (fun _ : SmoothBumpFunction I c => True) fun f =>
      (extChartAt I c).symm '' (closedBall (extChartAt I c c) f.rOut ∩ range I) := by
    rw [← map_extChartAt_symm_nhdsWithin_range (I := I) c]
    exact nhdsWithin_range_basis.map _
  exact this.to_hasBasis' (fun f _ => ⟨f, trivial, f.tsupport_subset_symm_image_closedBall⟩)
    fun f _ => f.tsupport_mem_nhds


/-- Given `s ∈ 𝓝 c`, the supports of smooth bump functions `f : SmoothBumpFunction I c` such that
`tsupport f ⊆ s` form a basis of `𝓝 c`.  In other words, each of these supports is a
neighborhood of `c` and each neighborhood of `c` includes `support f` for some
`f : SmoothBumpFunction I c` such that `tsupport f ⊆ s`. -/
theorem nhds_basis_support {s : Set M} (hs : s ∈ 𝓝 c) :
    (𝓝 c).HasBasis (fun f : SmoothBumpFunction I c => tsupport f ⊆ s) fun f => support f :=
  ((nhds_basis_tsupport c).restrict_subset hs).to_hasBasis'
    (fun f hf => ⟨f, hf.2, subset_closure⟩) fun f _ => f.support_mem_nhds


/-- A smooth bump function is infinitely smooth. -/
protected theorem contMDiff : ContMDiff I 𝓘(ℝ) ⊤ f := by
  /-
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝² : FiniteDimensional Real E
    inst✝¹ : T2Space M
    inst✝ : SmoothManifoldWithCorners I M
    ⊢ ContMDiff I (modelWithCornersSelf Real Real) Top.top ↑f
  -/
  refine contMDiff_of_tsupport fun x hx => ?_
  /-
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝² : FiniteDimensional Real E
    inst✝¹ : T2Space M
    inst✝ : SmoothManifoldWithCorners I M
    x : M
    hx : Membership.mem (tsupport ↑f) x
    ⊢ ContMDiffAt I (modelWithCornersSelf Real Real) Top.top (↑f) x
  -/
  have : x ∈ (chartAt H c).source := f.tsupport_subset_chartAt_source hx
  refine ContMDiffAt.congr_of_eventuallyEq ?_ <| f.eqOn_source.eventuallyEq_of_mem <|
    (chartAt H c).open_source.mem_nhds this
  /-
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝² : FiniteDimensional Real E
    inst✝¹ : T2Space M
    inst✝ : SmoothManifoldWithCorners I M
    x : M
    hx : Membership.mem (tsupport ↑f) x
    this : Membership.mem (chartAt H c).source x
    ⊢ ContMDiffAt I (modelWithCornersSelf Real Real) Top.top (Function.comp ↑f.toC …
  -/
  exact f.contDiffAt.contMDiffAt.comp _ (contMDiffAt_extChartAt' this)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smooth := SmoothBumpFunction.contMDiff


protected theorem contMDiffAt {x} : ContMDiffAt I 𝓘(ℝ) ⊤ f x :=
  f.contMDiff.contMDiffAt


@[deprecated (since := "2024-11-20")] alias smoothAt := SmoothBumpFunction.contMDiffAt


protected theorem continuous : Continuous f :=
  f.contMDiff.continuous


/-- If `f : SmoothBumpFunction I c` is a smooth bump function and `g : M → G` is a function smooth
on the source of the chart at `c`, then `f • g` is smooth on the whole manifold. -/
theorem contMDiff_smul {G} [NormedAddCommGroup G] [NormedSpace ℝ G] {g : M → G}
    (hg : ContMDiffOn I 𝓘(ℝ, G) ⊤ g (chartAt H c).source) :
    ContMDiff I 𝓘(ℝ, G) ⊤ fun x => f x • g x := by
  /-
    E : Type uE
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    H : Type uH
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : T2Space M
    inst✝² : SmoothManifoldWithCorners I M
    G : Type u_1
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    g : M → G
    hg : ContMDiffOn I (modelWithCornersSelf Real G) Top.top g (chartAt H c).source
    ⊢ ContMDiff I (modelWithCornersSelf Real G) Top.top fun x => HSMul.hSMul (↑f x …
  -/
  refine contMDiff_of_tsupport fun x hx => ?_
  have : x ∈ (chartAt H c).source :=
  -- Porting note: was a more readable `calc`
  -- calc
  --   x ∈ tsupport fun x => f x • g x := hx
  --   _ ⊆ tsupport f := tsupport_smul_subset_left _ _
  --   _ ⊆ (chart_at _ c).source := f.tsupport_subset_chartAt_source
    f.tsupport_subset_chartAt_source <| tsupport_smul_subset_left _ _ hx
  /-
    E : Type uE
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    H : Type uH
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    c : M
    f : SmoothBumpFunction I c
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : T2Space M
    inst✝² : SmoothManifoldWithCorners I M
    G : Type u_1
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    g : M → G
    hg : ContMDiffOn I (modelWithCornersSelf Real G) Top.top g (chartAt H c).source
    x : M
    hx : Membership.mem (tsupport fun x => HSMul.hSMul (↑f x) (g x)) x
    this : Membership.mem (chartAt H c).source x
    ⊢ ContMDiffAt I (modelWithCornersSelf Real G) Top.top (fun x => HSMul.hSMul (↑ …
  -/
  exact f.contMDiffAt.smul ((hg _ this).contMDiffAt <| (chartAt _ _).open_source.mem_nhds this)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smooth_smul := contMDiff_smul


