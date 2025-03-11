/-- If `E` is a finite dimensional space over `ℝ`, then `toEuclidean` is a continuous `ℝ`-linear
equivalence between `E` and the Euclidean space of the same dimension. -/
def toEuclidean : E ≃L[ℝ] EuclideanSpace ℝ (Fin <| finrank ℝ E) :=
  ContinuousLinearEquiv.ofFinrankEq finrank_euclideanSpace_fin.symm


/-- If `x` and `y` are two points in a finite dimensional space over `ℝ`, then `Euclidean.dist x y`
is the distance between these points in the metric defined by some inner product space structure on
`E`. -/
nonrec def dist (x y : E) : ℝ :=
  dist (toEuclidean x) (toEuclidean y)


/-- Closed ball w.r.t. the euclidean distance. -/
def closedBall (x : E) (r : ℝ) : Set E :=
  {y | dist y x ≤ r}


/-- Open ball w.r.t. the euclidean distance. -/
def ball (x : E) (r : ℝ) : Set E :=
  {y | dist y x < r}


theorem ball_eq_preimage (x : E) (r : ℝ) :
    ball x r = toEuclidean ⁻¹' Metric.ball (toEuclidean x) r :=
  rfl


theorem closedBall_eq_preimage (x : E) (r : ℝ) :
    closedBall x r = toEuclidean ⁻¹' Metric.closedBall (toEuclidean x) r :=
  rfl


theorem ball_subset_closedBall {x : E} {r : ℝ} : ball x r ⊆ closedBall x r := fun _ (hy : _ < r) =>
  le_of_lt hy


theorem isOpen_ball {x : E} {r : ℝ} : IsOpen (ball x r) :=
  Metric.isOpen_ball.preimage toEuclidean.continuous


theorem mem_ball_self {x : E} {r : ℝ} (hr : 0 < r) : x ∈ ball x r :=
  Metric.mem_ball_self hr


theorem closedBall_eq_image (x : E) (r : ℝ) :
    closedBall x r = toEuclidean.symm '' Metric.closedBall (toEuclidean x) r := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    r : Real
    ⊢ Eq (Euclidean.closedBall x r) (Set.image (⇑toEuclidean.symm) (Metric.closedB …
  -/
  rw [toEuclidean.image_symm_eq_preimage, closedBall_eq_preimage]
  /-
    🎉 no goals
  -/


nonrec theorem isCompact_closedBall {x : E} {r : ℝ} : IsCompact (closedBall x r) := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    r : Real
    ⊢ IsCompact (Euclidean.closedBall x r)
  -/
  rw [closedBall_eq_image]
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    r : Real
    ⊢ IsCompact (Set.image (⇑toEuclidean.symm) (Metric.closedBall (toEuclidean x)  …
  -/
  exact (isCompact_closedBall _ _).image toEuclidean.symm.continuous
  /-
    🎉 no goals
  -/


theorem isClosed_closedBall {x : E} {r : ℝ} : IsClosed (closedBall x r) :=
  isCompact_closedBall.isClosed


nonrec theorem closure_ball (x : E) {r : ℝ} (h : r ≠ 0) : closure (ball x r) = closedBall x r := by
  rw [ball_eq_preimage, ← toEuclidean.preimage_closure, closure_ball (toEuclidean x) h,
    closedBall_eq_preimage]


nonrec theorem exists_pos_lt_subset_ball {R : ℝ} {s : Set E} {x : E} (hR : 0 < R) (hs : IsClosed s)
    (h : s ⊆ ball x R) : ∃ r ∈ Ioo 0 R, s ⊆ ball x r := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    R : Real
    s : Set E
    x : E
    hR : LT.lt 0 R
    hs : IsClosed s
    h : HasSubset.Subset s (Euclidean.ball x R)
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 R) r) (HasSubset.Subset s (Eu …
  -/
  rw [ball_eq_preimage, ← image_subset_iff] at h
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    R : Real
    s : Set E
    x : E
    hR : LT.lt 0 R
    hs : IsClosed s
    h : HasSubset.Subset (Set.image (⇑toEuclidean) s) (Metric.ball (toEuclidean x) …
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 R) r) (HasSubset.Subset s (Eu …
  -/
  rcases exists_pos_lt_subset_ball hR (toEuclidean.isClosed_image.2 hs) h with ⟨r, hr, hsr⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    R : Real
    s : Set E
    x : E
    hR : LT.lt 0 R
    hs : IsClosed s
    h : HasSubset.Subset (Set.image (⇑toEuclidean) s) (Metric.ball (toEuclidean x) …
    r : Real
    hr : Membership.mem (Set.Ioo 0 R) r
    hsr : HasSubset.Subset (Set.image (⇑toEuclidean) s) (Metric.ball (toEuclidean  …
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo 0 R) r) (HasSubset.Subset s (Eu …
  -/
  exact ⟨r, hr, image_subset_iff.1 hsr⟩
  /-
    🎉 no goals
  -/


theorem nhds_basis_closedBall {x : E} : (𝓝 x).HasBasis (fun r : ℝ => 0 < r) (closedBall x) := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    ⊢ (nhds x).HasBasis (fun r => LT.lt 0 r) (Euclidean.closedBall x)
  -/
  rw [toEuclidean.toHomeomorph.nhds_eq_comap x]
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    ⊢ (Filter.comap (⇑toEuclidean.toHomeomorph) (nhds (toEuclidean.toHomeomorph x) …
  -/
  exact Metric.nhds_basis_closedBall.comap _
  /-
    🎉 no goals
  -/


theorem closedBall_mem_nhds {x : E} {r : ℝ} (hr : 0 < r) : closedBall x r ∈ 𝓝 x :=
  nhds_basis_closedBall.mem_of_mem hr


theorem nhds_basis_ball {x : E} : (𝓝 x).HasBasis (fun r : ℝ => 0 < r) (ball x) := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    ⊢ (nhds x).HasBasis (fun r => LT.lt 0 r) (Euclidean.ball x)
  -/
  rw [toEuclidean.toHomeomorph.nhds_eq_comap x]
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : T2Space E
    inst✝² : Module Real E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : FiniteDimensional Real E
    x : E
    ⊢ (Filter.comap (⇑toEuclidean.toHomeomorph) (nhds (toEuclidean.toHomeomorph x) …
  -/
  exact Metric.nhds_basis_ball.comap _
  /-
    🎉 no goals
  -/


theorem ball_mem_nhds {x : E} {r : ℝ} (hr : 0 < r) : ball x r ∈ 𝓝 x :=
  nhds_basis_ball.mem_of_mem hr


theorem ContDiff.euclidean_dist (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (h : ∀ x, f x ≠ g x) :
    ContDiff ℝ n fun x => Euclidean.dist (f x) (g x) := by
  /-
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    inst✝ : FiniteDimensional Real G
    f g : F → G
    n : ENat
    hf : ContDiff Real (↑n) f
    hg : ContDiff Real (↑n) g
    h : ∀ (x : F), Ne (f x) (g x)
    ⊢ ContDiff Real ↑n fun x => Euclidean.dist (f x) (g x)
  -/
  simp only [Euclidean.dist]
  /-
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    inst✝ : FiniteDimensional Real G
    f g : F → G
    n : ENat
    hf : ContDiff Real (↑n) f
    hg : ContDiff Real (↑n) g
    h : ∀ (x : F), Ne (f x) (g x)
    ⊢ ContDiff Real ↑n fun x => Dist.dist (toEuclidean (f x)) (toEuclidean (g x))
  -/
  apply ContDiff.dist ℝ
  exacts [(toEuclidean (E := G)).contDiff.comp hf,
    (toEuclidean (E := G)).contDiff.comp hg, fun x => toEuclidean.injective.ne (h x)]

