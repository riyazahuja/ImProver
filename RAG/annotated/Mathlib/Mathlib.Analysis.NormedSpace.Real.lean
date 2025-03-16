/-- If `E` is a nontrivial topological module over `ℝ`, then `E` has no isolated points.
This is a particular case of `Module.punctured_nhds_neBot`. -/
instance Real.punctured_nhds_module_neBot {E : Type*} [AddCommGroup E] [TopologicalSpace E]
    [ContinuousAdd E] [Nontrivial E] [Module ℝ E] [ContinuousSMul ℝ E] (x : E) : NeBot (𝓝[≠] x) :=
  Module.punctured_nhds_neBot ℝ E x


theorem inv_norm_smul_mem_unitClosedBall (x : E) :
    ‖x‖⁻¹ • x ∈ closedBall (0 : E) 1 := by
  simp only [mem_closedBall_zero_iff, norm_smul, norm_inv, norm_norm, ← div_eq_inv_mul,
    div_self_le_one]


@[deprecated (since := "2024-12-01")]
alias inv_norm_smul_mem_closed_unit_ball := inv_norm_smul_mem_unitClosedBall


theorem norm_smul_of_nonneg {t : ℝ} (ht : 0 ≤ t) (x : E) : ‖t • x‖ = t * ‖x‖ := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    t : Real
    ht : LE.le 0 t
    x : E
    ⊢ Eq (Norm.norm (HSMul.hSMul t x)) (HMul.hMul t (Norm.norm x))
  -/
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht]
  /-
    🎉 no goals
  -/


theorem dist_smul_add_one_sub_smul_le {r : ℝ} {x y : E} (h : r ∈ Icc 0 1) :
    dist (r • x + (1 - r) • y) x ≤ dist y x :=
  calc
    dist (r • x + (1 - r) • y) x = ‖1 - r‖ * ‖x - y‖ := by
      simp_rw [dist_eq_norm', ← norm_smul, sub_smul, one_smul, smul_sub, ← sub_sub, ← sub_add,
        sub_right_comm]
    _ = (1 - r) * dist y x := by
      /-
        E : Type u_1
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        x y : E
        h : Membership.mem (Set.Icc 0 1) r
        ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub 1 r)) (Norm.norm (HSub.hSub x y))) (HMul …
      -/
      rw [Real.norm_eq_abs, abs_eq_self.mpr (sub_nonneg.mpr h.2), dist_eq_norm']
      /-
        🎉 no goals
      -/
                                 /-
                                   E : Type u_1
                                   inst✝¹ : SeminormedAddCommGroup E
                                   inst✝ : NormedSpace Real E
                                   r : Real
                                   x y : E
                                   h : Membership.mem (Set.Icc 0 1) r
                                   ⊢ LE.le (HMul.hMul (HSub.hSub 1 r) (Dist.dist y x)) (HMul.hMul (HSub.hSub 1 0) …
                                 -/
    _ ≤ (1 - 0) * dist y x := by gcongr; exact h.1
                                         /-
                                           🎉 no goals
                                         -/
                       /-
                         E : Type u_1
                         inst✝¹ : SeminormedAddCommGroup E
                         inst✝ : NormedSpace Real E
                         r : Real
                         x y : E
                         h : Membership.mem (Set.Icc 0 1) r
                         ⊢ Eq (HMul.hMul (HSub.hSub 1 0) (Dist.dist y x)) (Dist.dist y x)
                       -/
    _ = dist y x := by rw [sub_zero, one_mul]
                       /-
                         🎉 no goals
                       -/


theorem closure_ball (x : E) {r : ℝ} (hr : r ≠ 0) : closure (ball x r) = closedBall x r := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (closure (Metric.ball x r)) (Metric.closedBall x r)
  -/
  refine Subset.antisymm closure_ball_subset_closedBall fun y hy => ?_
  have : ContinuousWithinAt (fun c : ℝ => c • (y - x) + x) (Ico 0 1) 1 :=
    ((continuous_id.smul continuous_const).add continuous_const).continuousWithinAt
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    y : E
    hy : Membership.mem (Metric.closedBall x r) y
    this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
    ⊢ Membership.mem (closure (Metric.ball x r)) y
  -/
  convert this.mem_closure _ _
    /-
      case h.e'_5
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      y : E
      hy : Membership.mem (Metric.closedBall x r) y
      this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
      ⊢ Eq y (HAdd.hAdd (HSMul.hSMul 1 (HSub.hSub y x)) x)
    -/
  · rw [one_smul, sub_add_cancel]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      y : E
      hy : Membership.mem (Metric.closedBall x r) y
      this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
      ⊢ Membership.mem (closure (Set.Ico 0 1)) 1
    -/
  · simp [closure_Ico zero_ne_one, zero_le_one]
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      y : E
      hy : Membership.mem (Metric.closedBall x r) y
      this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
      ⊢ Set.MapsTo (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x)) x) (Set.Ico 0 …
    -/
  · rintro c ⟨hc0, hc1⟩
    rw [mem_ball, dist_eq_norm, add_sub_cancel_right, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg hc0, mul_comm, ← mul_one r]
    /-
      case convert_3.intro
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      y : E
      hy : Membership.mem (Metric.closedBall x r) y
      this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
      c : Real
      hc0 : LE.le 0 c
      hc1 : LT.lt c 1
      ⊢ LT.lt (HMul.hMul (Norm.norm (HSub.hSub y x)) c) (HMul.hMul r 1)
    -/
    rw [mem_closedBall, dist_eq_norm] at hy
    /-
      case convert_3.intro
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr : Ne r 0
      y : E
      hy : LE.le (Norm.norm (HSub.hSub y x)) r
      this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
      c : Real
      hc0 : LE.le 0 c
      hc1 : LT.lt c 1
      ⊢ LT.lt (HMul.hMul (Norm.norm (HSub.hSub y x)) c) (HMul.hMul r 1)
    -/
    replace hr : 0 < r := ((norm_nonneg _).trans hy).lt_of_ne hr.symm
    /-
      case convert_3.intro
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      y : E
      hy : LE.le (Norm.norm (HSub.hSub y x)) r
      this : ContinuousWithinAt (fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x))  …
      c : Real
      hc0 : LE.le 0 c
      hc1 : LT.lt c 1
      hr : LT.lt 0 r
      ⊢ LT.lt (HMul.hMul (Norm.norm (HSub.hSub y x)) c) (HMul.hMul r 1)
    -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
    apply mul_lt_mul' <;> assumption
                          /-
                            🎉 no goals
                          -/


theorem frontier_ball (x : E) {r : ℝ} (hr : r ≠ 0) :
    frontier (ball x r) = sphere x r := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (frontier (Metric.ball x r)) (Metric.sphere x r)
  -/
  rw [frontier, closure_ball x hr, isOpen_ball.interior_eq, closedBall_diff_ball]
  /-
    🎉 no goals
  -/


theorem interior_closedBall (x : E) {r : ℝ} (hr : r ≠ 0) :
    interior (closedBall x r) = ball x r := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (interior (Metric.closedBall x r)) (Metric.ball x r)
  -/
  cases' hr.lt_or_lt with hr hr
    /-
      case inl
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr✝ : Ne r 0
      hr : LT.lt r 0
      ⊢ Eq (interior (Metric.closedBall x r)) (Metric.ball x r)
    -/
  · rw [closedBall_eq_empty.2 hr, ball_eq_empty.2 hr.le, interior_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr✝ : Ne r 0
    hr : LT.lt 0 r
    ⊢ Eq (interior (Metric.closedBall x r)) (Metric.ball x r)
  -/
  refine Subset.antisymm ?_ ball_subset_interior_closedBall
  /-
    case inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr✝ : Ne r 0
    hr : LT.lt 0 r
    ⊢ HasSubset.Subset (interior (Metric.closedBall x r)) (Metric.ball x r)
  -/
  intro y hy
  /-
    case inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr✝ : Ne r 0
    hr : LT.lt 0 r
    y : E
    hy : Membership.mem (interior (Metric.closedBall x r)) y
    ⊢ Membership.mem (Metric.ball x r) y
  -/
  rcases (mem_closedBall.1 <| interior_subset hy).lt_or_eq with (hr | rfl)
    /-
      case inr.inl
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      hr✝¹ : Ne r 0
      hr✝ : LT.lt 0 r
      y : E
      hy : Membership.mem (interior (Metric.closedBall x r)) y
      hr : LT.lt (Dist.dist y x) r
      ⊢ Membership.mem (Metric.ball x r) y
    -/
  · exact hr
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    hr✝ : Ne (Dist.dist y x) 0
    hr : LT.lt 0 (Dist.dist y x)
    hy : Membership.mem (interior (Metric.closedBall x (Dist.dist y x))) y
    ⊢ Membership.mem (Metric.ball x (Dist.dist y x)) y
  -/
  set f : ℝ → E := fun c : ℝ => c • (y - x) + x
  suffices f ⁻¹' closedBall x (dist y x) ⊆ Icc (-1) 1 by
    have hfc : Continuous f := (continuous_id.smul continuous_const).add continuous_const
    have hf1 : (1 : ℝ) ∈ f ⁻¹' interior (closedBall x <| dist y x) := by simpa [f]
    have h1 : (1 : ℝ) ∈ interior (Icc (-1 : ℝ) 1) :=
      interior_mono this (preimage_interior_subset_interior_preimage hfc hf1)
    simp at h1
  /-
    case inr.inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    hr✝ : Ne (Dist.dist y x) 0
    hr : LT.lt 0 (Dist.dist y x)
    hy : Membership.mem (interior (Metric.closedBall x (Dist.dist y x))) y
    f : Real → E := fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x)) x
    ⊢ HasSubset.Subset (Set.preimage f (Metric.closedBall x (Dist.dist y x))) (Set …
  -/
  intro c hc
  /-
    case inr.inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    hr✝ : Ne (Dist.dist y x) 0
    hr : LT.lt 0 (Dist.dist y x)
    hy : Membership.mem (interior (Metric.closedBall x (Dist.dist y x))) y
    f : Real → E := fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x)) x
    c : Real
    hc : Membership.mem (Set.preimage f (Metric.closedBall x (Dist.dist y x))) c
    ⊢ Membership.mem (Set.Icc (-1) 1) c
  -/
  rw [mem_Icc, ← abs_le, ← Real.norm_eq_abs, ← mul_le_mul_right hr]
  /-
    case inr.inr
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    hr✝ : Ne (Dist.dist y x) 0
    hr : LT.lt 0 (Dist.dist y x)
    hy : Membership.mem (interior (Metric.closedBall x (Dist.dist y x))) y
    f : Real → E := fun c => HAdd.hAdd (HSMul.hSMul c (HSub.hSub y x)) x
    c : Real
    hc : Membership.mem (Set.preimage f (Metric.closedBall x (Dist.dist y x))) c
    ⊢ LE.le (HMul.hMul (Norm.norm c) (Dist.dist y x)) (HMul.hMul 1 (Dist.dist y x))
  -/
  simpa [f, dist_eq_norm, norm_smul] using hc
  /-
    🎉 no goals
  -/


theorem frontier_closedBall (x : E) {r : ℝ} (hr : r ≠ 0) :
    frontier (closedBall x r) = sphere x r := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (frontier (Metric.closedBall x r)) (Metric.sphere x r)
  -/
  rw [frontier, closure_closedBall, interior_closedBall x hr, closedBall_diff_ball]
  /-
    🎉 no goals
  -/


theorem interior_sphere (x : E) {r : ℝ} (hr : r ≠ 0) : interior (sphere x r) = ∅ := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (interior (Metric.sphere x r)) EmptyCollection.emptyCollection
  -/
  rw [← frontier_closedBall x hr, interior_frontier isClosed_ball]
  /-
    🎉 no goals
  -/


theorem frontier_sphere (x : E) {r : ℝ} (hr : r ≠ 0) : frontier (sphere x r) = sphere x r := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    hr : Ne r 0
    ⊢ Eq (frontier (Metric.sphere x r)) (Metric.sphere x r)
  -/
  rw [isClosed_sphere.frontier_eq, interior_sphere x hr, diff_empty]
  /-
    🎉 no goals
  -/


theorem exists_norm_eq {c : ℝ} (hc : 0 ≤ c) : ∃ x : E, ‖x‖ = c := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    c : Real
    hc : LE.le 0 c
    ⊢ Exists fun x => Eq (Norm.norm x) c
  -/
  rcases exists_ne (0 : E) with ⟨x, hx⟩
  /-
    case intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    c : Real
    hc : LE.le 0 c
    x : E
    hx : Ne x 0
    ⊢ Exists fun x => Eq (Norm.norm x) c
  -/
  rw [← norm_ne_zero_iff] at hx
  /-
    case intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    c : Real
    hc : LE.le 0 c
    x : E
    hx : Ne (Norm.norm x) 0
    ⊢ Exists fun x => Eq (Norm.norm x) c
  -/
  use c • ‖x‖⁻¹ • x
  /-
    case h
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    c : Real
    hc : LE.le 0 c
    x : E
    hx : Ne (Norm.norm x) 0
    ⊢ Eq (Norm.norm (HSMul.hSMul c (HSMul.hSMul (Inv.inv (Norm.norm x)) x))) c
  -/
  simp [norm_smul, Real.norm_of_nonneg hc, abs_of_nonneg hc, inv_mul_cancel₀ hx]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_norm : range (norm : E → ℝ) = Ici 0 :=
  Subset.antisymm (range_subset_iff.2 norm_nonneg) fun _ => exists_norm_eq E


theorem nnnorm_surjective : Surjective (nnnorm : E → ℝ≥0) := fun c =>
  (exists_norm_eq E c.coe_nonneg).imp fun _ h => NNReal.eq h


@[simp]
theorem range_nnnorm : range (nnnorm : E → ℝ≥0) = univ :=
  (nnnorm_surjective E).range_eq


theorem interior_closedBall' (x : E) (r : ℝ) : interior (closedBall x r) = ball x r := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (interior (Metric.closedBall x r)) (Metric.ball x r)
  -/
  rcases eq_or_ne r 0 with (rfl | hr)
    /-
      case inl
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      ⊢ Eq (interior (Metric.closedBall x 0)) (Metric.ball x 0)
    -/
  · rw [closedBall_zero, ball_zero, interior_singleton]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      r : Real
      hr : Ne r 0
      ⊢ Eq (interior (Metric.closedBall x r)) (Metric.ball x r)
    -/
  · exact interior_closedBall x hr
    /-
      🎉 no goals
    -/


theorem frontier_closedBall' (x : E) (r : ℝ) : frontier (closedBall x r) = sphere x r := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (frontier (Metric.closedBall x r)) (Metric.sphere x r)
  -/
  rw [frontier, closure_closedBall, interior_closedBall' x r, closedBall_diff_ball]
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_sphere' (x : E) (r : ℝ) : interior (sphere x r) = ∅ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (interior (Metric.sphere x r)) EmptyCollection.emptyCollection
  -/
  rw [← frontier_closedBall' x, interior_frontier isClosed_ball]
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_sphere' (x : E) (r : ℝ) : frontier (sphere x r) = sphere x r := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    r : Real
    ⊢ Eq (frontier (Metric.sphere x r)) (Metric.sphere x r)
  -/
  rw [isClosed_sphere.frontier_eq, interior_sphere' x, diff_empty]
  /-
    🎉 no goals
  -/


