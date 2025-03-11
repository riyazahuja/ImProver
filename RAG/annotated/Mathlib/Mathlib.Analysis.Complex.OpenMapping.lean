/-- If the modulus of a holomorphic function `f` is bounded below by `ε` on a circle, then its range
contains a disk of radius `ε / 2`. -/
theorem DiffContOnCl.ball_subset_image_closedBall (h : DiffContOnCl ℂ f (ball z₀ r)) (hr : 0 < r)
    (hf : ∀ z ∈ sphere z₀ r, ε ≤ ‖f z - f z₀‖) (hz₀ : ∃ᶠ z in 𝓝 z₀, f z ≠ f z₀) :
    ball (f z₀) (ε / 2) ⊆ f '' closedBall z₀ r := by
  /- This is a direct application of the maximum principle. Pick `v` close to `f z₀`, and look at
    the function `fun z ↦ ‖f z - v‖`: it is bounded below on the circle, and takes a small value
    at `z₀` so it is not constant on the disk, which implies that its infimum is equal to `0` and
    hence that `v` is in the range of `f`. -/
  /-
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    ⊢ HasSubset.Subset (Metric.ball (f z₀) (HDiv.hDiv ε 2)) (Set.image f (Metric.c …
  -/
  rintro v hv
  /-
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    ⊢ Membership.mem (Set.image f (Metric.closedBall z₀ r)) v
  -/
  have h1 : DiffContOnCl ℂ (fun z => f z - v) (ball z₀ r) := h.sub_const v
  have h2 : ContinuousOn (fun z => ‖f z - v‖) (closedBall z₀ r) :=
    continuous_norm.comp_continuousOn (closure_ball z₀ hr.ne.symm ▸ h1.continuousOn)
  /-
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    ⊢ Membership.mem (Set.image f (Metric.closedBall z₀ r)) v
  -/
  have h3 : AnalyticOnNhd ℂ f (ball z₀ r) := h.differentiableOn.analyticOnNhd isOpen_ball
  have h4 : ∀ z ∈ sphere z₀ r, ε / 2 ≤ ‖f z - v‖ := fun z hz => by
    linarith [hf z hz, show ‖v - f z₀‖ < ε / 2 from mem_ball.mp hv,
      norm_sub_sub_norm_sub_le_norm_sub (f z) v (f z₀)]
  /-
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    ⊢ Membership.mem (Set.image f (Metric.closedBall z₀ r)) v
  -/
  have h5 : ‖f z₀ - v‖ < ε / 2 := by simpa [← dist_eq_norm, dist_comm] using mem_ball.mp hv
  obtain ⟨z, hz1, hz2⟩ : ∃ z ∈ ball z₀ r, IsLocalMin (fun z => ‖f z - v‖) z :=
    exists_isLocalMin_mem_ball h2 (mem_closedBall_self hr.le) fun z hz => h5.trans_le (h4 z hz)
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    ⊢ Membership.mem (Set.image f (Metric.closedBall z₀ r)) v
  -/
  refine ⟨z, ball_subset_closedBall hz1, sub_eq_zero.mp ?_⟩
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    ⊢ Eq (HSub.hSub (f z) v) 0
  -/
  have h6 := h1.differentiableOn.eventually_differentiableAt (isOpen_ball.mem_nhds hz1)
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    ⊢ Eq (HSub.hSub (f z) v) 0
  -/
  refine (eventually_eq_or_eq_zero_of_isLocalMin_norm h6 hz2).resolve_left fun key => ?_
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    key : Filter.Eventually (fun z_1 => Eq (HSub.hSub (f z_1) v) (HSub.hSub (f z)  …
    ⊢ False
  -/
  have h7 : ∀ᶠ w in 𝓝 z, f w = f z := by filter_upwards [key] with h; field_simp
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    key : Filter.Eventually (fun z_1 => Eq (HSub.hSub (f z_1) v) (HSub.hSub (f z)  …
    h7 : Filter.Eventually (fun w => Eq (f w) (f z)) (nhds z)
    ⊢ False
  -/
  replace h7 : ∃ᶠ w in 𝓝[≠] z, f w = f z := (h7.filter_mono nhdsWithin_le_nhds).frequently
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    key : Filter.Eventually (fun z_1 => Eq (HSub.hSub (f z_1) v) (HSub.hSub (f z)  …
    h7 : Filter.Frequently (fun w => Eq (f w) (f z)) (nhdsWithin z (HasCompl.compl …
    ⊢ False
  -/
  have h8 : IsPreconnected (ball z₀ r) := (convex_ball z₀ r).isPreconnected
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    key : Filter.Eventually (fun z_1 => Eq (HSub.hSub (f z_1) v) (HSub.hSub (f z)  …
    h7 : Filter.Frequently (fun w => Eq (f w) (f z)) (nhdsWithin z (HasCompl.compl …
    h8 : IsPreconnected (Metric.ball z₀ r)
    ⊢ False
  -/
  have h9 := h3.eqOn_of_preconnected_of_frequently_eq analyticOnNhd_const h8 hz1 h7
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    key : Filter.Eventually (fun z_1 => Eq (HSub.hSub (f z_1) v) (HSub.hSub (f z)  …
    h7 : Filter.Frequently (fun w => Eq (f w) (f z)) (nhdsWithin z (HasCompl.compl …
    h8 : IsPreconnected (Metric.ball z₀ r)
    h9 : Set.EqOn f (fun x => f z) (Metric.ball z₀ r)
    ⊢ False
  -/
  have h10 : f z = f z₀ := (h9 (mem_ball_self hr)).symm
  /-
    case intro.intro
    f : Complex → Complex
    z₀ : Complex
    ε r : Real
    h : DiffContOnCl Complex f (Metric.ball z₀ r)
    hr : LT.lt 0 r
    hf : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le ε (Norm.no …
    hz₀ : Filter.Frequently (fun z => Ne (f z) (f z₀)) (nhds z₀)
    v : Complex
    hv : Membership.mem (Metric.ball (f z₀) (HDiv.hDiv ε 2)) v
    h1 : DiffContOnCl Complex (fun z => HSub.hSub (f z) v) (Metric.ball z₀ r)
    h2 : ContinuousOn (fun z => Norm.norm (HSub.hSub (f z) v)) (Metric.closedBall  …
    h3 : AnalyticOnNhd Complex f (Metric.ball z₀ r)
    h4 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → LE.le (HDiv.hDiv …
    h5 : LT.lt (Norm.norm (HSub.hSub (f z₀) v)) (HDiv.hDiv ε 2)
    z : Complex
    hz1 : Membership.mem (Metric.ball z₀ r) z
    hz2 : IsLocalMin (fun z => Norm.norm (HSub.hSub (f z) v)) z
    h6 : Filter.Eventually (fun y => DifferentiableAt Complex (fun z => HSub.hSub  …
    key : Filter.Eventually (fun z_1 => Eq (HSub.hSub (f z_1) v) (HSub.hSub (f z)  …
    h7 : Filter.Frequently (fun w => Eq (f w) (f z)) (nhdsWithin z (HasCompl.compl …
    h8 : IsPreconnected (Metric.ball z₀ r)
    h9 : Set.EqOn f (fun x => f z) (Metric.ball z₀ r)
    h10 : Eq (f z) (f z₀)
    ⊢ False
  -/
  exact not_eventually.mpr hz₀ (mem_of_superset (ball_mem_nhds z₀ hr) (h10 ▸ h9))
  /-
    🎉 no goals
  -/


/-- A function `f : ℂ → ℂ` which is analytic at a point `z₀` is either constant in a neighborhood
of `z₀`, or behaves locally like an open function (in the sense that the image of every neighborhood
of `z₀` is a neighborhood of `f z₀`, as in `isOpenMap_iff_nhds_le`). For a function `f : E → ℂ`
the same result holds, see `AnalyticAt.eventually_constant_or_nhds_le_map_nhds`. -/
theorem AnalyticAt.eventually_constant_or_nhds_le_map_nhds_aux (hf : AnalyticAt ℂ f z₀) :
    (∀ᶠ z in 𝓝 z₀, f z = f z₀) ∨ 𝓝 (f z₀) ≤ map f (𝓝 z₀) := by
  /- The function `f` is analytic in a neighborhood of `z₀`; by the isolated zeros principle, if `f`
    is not constant in a neighborhood of `z₀`, then it is nonzero, and therefore bounded below, on
    every small enough circle around `z₀` and then `DiffContOnCl.ball_subset_image_closedBall`
    provides an explicit ball centered at `f z₀` contained in the range of `f`. -/
  /-
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    ⊢ Or (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀)) (LE.le (nhds (f  …
  -/
  refine or_iff_not_imp_left.mpr fun h => ?_
  /-
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    ⊢ LE.le (nhds (f z₀)) (Filter.map f (nhds z₀))
  -/
  refine (nhds_basis_ball.le_basis_iff (nhds_basis_closedBall.map f)).mpr fun R hR => ?_
  /-
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  have h1 := (hf.eventually_eq_or_eventually_ne analyticAt_const).resolve_left h
  /-
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  have h2 : ∀ᶠ z in 𝓝 z₀, AnalyticAt ℂ f z := (isOpen_analyticAt ℂ f).eventually_mem hf
  obtain ⟨ρ, hρ, h3, h4⟩ :
    ∃ ρ > 0, AnalyticOnNhd ℂ f (closedBall z₀ ρ) ∧ ∀ z ∈ closedBall z₀ ρ, z ≠ z₀ → f z ≠ f z₀ := by
    simpa only [setOf_and, subset_inter_iff] using
      nhds_basis_closedBall.mem_iff.mp (h2.and (eventually_nhdsWithin_iff.mp h1))
  replace h3 : DiffContOnCl ℂ f (ball z₀ ρ) :=
    ⟨h3.differentiableOn.mono ball_subset_closedBall,
      (closure_ball z₀ hρ.lt.ne.symm).symm ▸ h3.continuousOn⟩
  /-
    case intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  let r := ρ ⊓ R
  /-
    case intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    r : Real := Min.min ρ R
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  have hr : 0 < r := lt_inf_iff.mpr ⟨hρ, hR⟩
  /-
    case intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    r : Real := Min.min ρ R
    hr : LT.lt 0 r
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  have h5 : closedBall z₀ r ⊆ closedBall z₀ ρ := closedBall_subset_closedBall inf_le_left
  /-
    case intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    r : Real := Min.min ρ R
    hr : LT.lt 0 r
    h5 : HasSubset.Subset (Metric.closedBall z₀ r) (Metric.closedBall z₀ ρ)
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  have h6 : DiffContOnCl ℂ f (ball z₀ r) := h3.mono (ball_subset_ball inf_le_left)
  have h7 : ∀ z ∈ sphere z₀ r, f z ≠ f z₀ := fun z hz =>
    h4 z (h5 (sphere_subset_closedBall hz)) (ne_of_mem_sphere hz hr.ne.symm)
  /-
    case intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    r : Real := Min.min ρ R
    hr : LT.lt 0 r
    h5 : HasSubset.Subset (Metric.closedBall z₀ r) (Metric.closedBall z₀ ρ)
    h6 : DiffContOnCl Complex f (Metric.ball z₀ r)
    h7 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → Ne (f z) (f z₀)
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  have h8 : (sphere z₀ r).Nonempty := NormedSpace.sphere_nonempty.mpr hr.le
  have h9 : ContinuousOn (fun x => ‖f x - f z₀‖) (sphere z₀ r) := continuous_norm.comp_continuousOn
    ((h6.sub_const (f z₀)).continuousOn_ball.mono sphere_subset_closedBall)
  /-
    case intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    r : Real := Min.min ρ R
    hr : LT.lt 0 r
    h5 : HasSubset.Subset (Metric.closedBall z₀ r) (Metric.closedBall z₀ ρ)
    h6 : DiffContOnCl Complex f (Metric.ball z₀ r)
    h7 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → Ne (f z) (f z₀)
    h8 : (Metric.sphere z₀ r).Nonempty
    h9 : ContinuousOn (fun x => Norm.norm (HSub.hSub (f x) (f z₀))) (Metric.sphere …
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  obtain ⟨x, hx, hfx⟩ := (isCompact_sphere z₀ r).exists_isMinOn h8 h9
  /-
    case intro.intro.intro.intro.intro
    f : Complex → Complex
    z₀ : Complex
    hf : AnalyticAt Complex f z₀
    h : Not (Filter.Eventually (fun z => Eq (f z) (f z₀)) (nhds z₀))
    R : Real
    hR : LT.lt 0 R
    h1 : Filter.Eventually (fun z => Ne (f z) (f z₀)) (nhdsWithin z₀ (HasCompl.com …
    h2 : Filter.Eventually (fun z => AnalyticAt Complex f z) (nhds z₀)
    ρ : Real
    hρ : GT.gt ρ 0
    h4 : ∀ (z : Complex), Membership.mem (Metric.closedBall z₀ ρ) z → Ne z z₀ → Ne …
    h3 : DiffContOnCl Complex f (Metric.ball z₀ ρ)
    r : Real := Min.min ρ R
    hr : LT.lt 0 r
    h5 : HasSubset.Subset (Metric.closedBall z₀ r) (Metric.closedBall z₀ ρ)
    h6 : DiffContOnCl Complex f (Metric.ball z₀ r)
    h7 : ∀ (z : Complex), Membership.mem (Metric.sphere z₀ r) z → Ne (f z) (f z₀)
    h8 : (Metric.sphere z₀ r).Nonempty
    h9 : ContinuousOn (fun x => Norm.norm (HSub.hSub (f x) (f z₀))) (Metric.sphere …
    x : Complex
    hx : Membership.mem (Metric.sphere z₀ r) x
    hfx : IsMinOn (fun x => Norm.norm (HSub.hSub (f x) (f z₀))) (Metric.sphere z₀  …
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.ball (f z₀) i) (Se …
  -/
  refine ⟨‖f x - f z₀‖ / 2, half_pos (norm_sub_pos_iff.mpr (h7 x hx)), ?_⟩
  exact (h6.ball_subset_image_closedBall hr (fun z hz => hfx hz) (not_eventually.mp h)).trans
    (image_subset f (closedBall_subset_closedBall inf_le_right))


/-- The *open mapping theorem* for holomorphic functions, local version: is a function `g : E → ℂ`
is analytic at a point `z₀`, then either it is constant in a neighborhood of `z₀`, or it maps every
neighborhood of `z₀` to a neighborhood of `z₀`. For the particular case of a holomorphic function on
`ℂ`, see `AnalyticAt.eventually_constant_or_nhds_le_map_nhds_aux`. -/
theorem AnalyticAt.eventually_constant_or_nhds_le_map_nhds {z₀ : E} (hg : AnalyticAt ℂ g z₀) :
    (∀ᶠ z in 𝓝 z₀, g z = g z₀) ∨ 𝓝 (g z₀) ≤ map g (𝓝 z₀) := by
  /- The idea of the proof is to use the one-dimensional version applied to the restriction of `g`
    to lines going through `z₀` (indexed by `sphere (0 : E) 1`). If the restriction is eventually
    constant along each of these lines, then the identity theorem implies that `g` is constant on
    any ball centered at `z₀` on which it is analytic, and in particular `g` is eventually constant.
    If on the other hand there is one line along which `g` is not eventually constant, then the
    one-dimensional version of the open mapping theorem can be used to conclude. -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    g : E → Complex
    z₀ : E
    hg : AnalyticAt Complex g z₀
    ⊢ Or (Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)) (LE.le (nhds (g  …
  -/
  let ray : E → ℂ → E := fun z t => z₀ + t • z
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    g : E → Complex
    z₀ : E
    hg : AnalyticAt Complex g z₀
    ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
    ⊢ Or (Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)) (LE.le (nhds (g  …
  -/
  let gray : E → ℂ → ℂ := fun z => g ∘ ray z
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    g : E → Complex
    z₀ : E
    hg : AnalyticAt Complex g z₀
    ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
    gray : E → Complex → Complex := fun z => Function.comp g (ray z)
    ⊢ Or (Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)) (LE.le (nhds (g  …
  -/
  obtain ⟨r, hr, hgr⟩ := isOpen_iff.mp (isOpen_analyticAt ℂ g) z₀ hg
  have h1 : ∀ z ∈ sphere (0 : E) 1, AnalyticOnNhd ℂ (gray z) (ball 0 r) := by
    refine fun z hz t ht => AnalyticAt.comp ?_ ?_
    · exact hgr (by simpa [ray, norm_smul, mem_sphere_zero_iff_norm.mp hz] using ht)
    · exact analyticAt_const.add
        ((ContinuousLinearMap.smulRight (ContinuousLinearMap.id ℂ ℂ) z).analyticAt t)
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    g : E → Complex
    z₀ : E
    hg : AnalyticAt Complex g z₀
    ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
    gray : E → Complex → Complex := fun z => Function.comp g (ray z)
    r : Real
    hr : GT.gt r 0
    hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
    h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
    ⊢ Or (Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)) (LE.le (nhds (g  …
  -/
  by_cases h : ∀ z ∈ sphere (0 : E) 1, ∀ᶠ t in 𝓝 0, gray z t = gray z 0
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually (fun t …
      ⊢ Or (Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)) (LE.le (nhds (g  …
    -/
  · left
    -- If g is eventually constant along every direction, then it is eventually constant
    /-
      case pos.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually (fun t …
      ⊢ Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)
    -/
    refine eventually_of_mem (ball_mem_nhds z₀ hr) fun z hz => ?_
    /-
      case pos.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually (fun t …
      z : E
      hz : Membership.mem (Metric.ball z₀ r) z
      ⊢ Eq (g z) (g z₀)
    -/
    refine (eq_or_ne z z₀).casesOn (congr_arg g) fun h' => ?_
    /-
      case pos.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually (fun t …
      z : E
      hz : Membership.mem (Metric.ball z₀ r) z
      h' : Ne z z₀
      ⊢ Eq (g z) (g z₀)
    -/
    replace h' : ‖z - z₀‖ ≠ 0 := by simpa only [Ne, norm_eq_zero, sub_eq_zero]
    /-
      case pos.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually (fun t …
      z : E
      hz : Membership.mem (Metric.ball z₀ r) z
      h' : Ne (Norm.norm (HSub.hSub z z₀)) 0
      ⊢ Eq (g z) (g z₀)
    -/
    let w : E := ‖z - z₀‖⁻¹ • (z - z₀)
    have h3 : ∀ t ∈ ball (0 : ℂ) r, gray w t = g z₀ := by
      have e1 : IsPreconnected (ball (0 : ℂ) r) := (convex_ball 0 r).isPreconnected
      have e2 : w ∈ sphere (0 : E) 1 := by simp [w, norm_smul, inv_mul_cancel₀ h']
      specialize h1 w e2
      apply h1.eqOn_of_preconnected_of_eventuallyEq analyticOnNhd_const e1 (mem_ball_self hr)
      simpa [ray, gray] using h w e2
    /-
      case pos.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually (fun t …
      z : E
      hz : Membership.mem (Metric.ball z₀ r) z
      h' : Ne (Norm.norm (HSub.hSub z z₀)) 0
      w : E := HSMul.hSMul (Inv.inv (Norm.norm (HSub.hSub z z₀))) (HSub.hSub z z₀)
      h3 : ∀ (t : Complex), Membership.mem (Metric.ball 0 r) t → Eq (gray w t) (g z₀)
      ⊢ Eq (g z) (g z₀)
    -/
    have h4 : ‖z - z₀‖ < r := by simpa [dist_eq_norm] using mem_ball.mp hz
    replace h4 : ↑‖z - z₀‖ ∈ ball (0 : ℂ) r := by
      simpa only [mem_ball_zero_iff, norm_eq_abs, abs_ofReal, abs_norm]
    simpa only [ray, gray, w, smul_smul, mul_inv_cancel₀ h', one_smul, add_sub_cancel,
      Function.comp_apply, coe_smul] using h3 (↑‖z - z₀‖) h4
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : Not (∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually ( …
      ⊢ Or (Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)) (LE.le (nhds (g  …
    -/
  · right
    -- Otherwise, it is open along at least one direction and that implies the result
    /-
      case neg.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : Not (∀ (z : E), Membership.mem (Metric.sphere 0 1) z → Filter.Eventually ( …
      ⊢ LE.le (nhds (g z₀)) (Filter.map g (nhds z₀))
    -/
    push_neg at h
    /-
      case neg.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      h : Exists fun z => And (Membership.mem (Metric.sphere 0 1) z) (Not (Filter.Ev …
      ⊢ LE.le (nhds (g z₀)) (Filter.map g (nhds z₀))
    -/
    obtain ⟨z, hz, hrz⟩ := h
    /-
      case neg.h.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      h1 : ∀ (z : E), Membership.mem (Metric.sphere 0 1) z → AnalyticOnNhd Complex ( …
      z : E
      hz : Membership.mem (Metric.sphere 0 1) z
      hrz : Not (Filter.Eventually (fun t => Eq (gray z t) (gray z 0)) (nhds 0))
      ⊢ LE.le (nhds (g z₀)) (Filter.map g (nhds z₀))
    -/
    specialize h1 z hz 0 (mem_ball_self hr)
    /-
      case neg.h.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      z : E
      hz : Membership.mem (Metric.sphere 0 1) z
      hrz : Not (Filter.Eventually (fun t => Eq (gray z t) (gray z 0)) (nhds 0))
      h1 : AnalyticAt Complex (gray z) 0
      ⊢ LE.le (nhds (g z₀)) (Filter.map g (nhds z₀))
    -/
    have h7 := h1.eventually_constant_or_nhds_le_map_nhds_aux.resolve_left hrz
    /-
      case neg.h.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      z : E
      hz : Membership.mem (Metric.sphere 0 1) z
      hrz : Not (Filter.Eventually (fun t => Eq (gray z t) (gray z 0)) (nhds 0))
      h1 : AnalyticAt Complex (gray z) 0
      h7 : LE.le (nhds (gray z 0)) (Filter.map (gray z) (nhds 0))
      ⊢ LE.le (nhds (g z₀)) (Filter.map g (nhds z₀))
    -/
    rw [show gray z 0 = g z₀ by simp [gray, ray], ← map_compose] at h7
    /-
      case neg.h.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      z : E
      hz : Membership.mem (Metric.sphere 0 1) z
      hrz : Not (Filter.Eventually (fun t => Eq (gray z t) (gray z 0)) (nhds 0))
      h1 : AnalyticAt Complex (gray z) 0
      h7 : LE.le (nhds (g z₀)) (Function.comp (Filter.map g) (Filter.map (ray z)) (n …
      ⊢ LE.le (nhds (g z₀)) (Filter.map g (nhds z₀))
    -/
    refine h7.trans (map_mono ?_)
    have h10 : Continuous fun t : ℂ => z₀ + t • z :=
      continuous_const.add (continuous_id'.smul continuous_const)
    /-
      case neg.h.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      g : E → Complex
      z₀ : E
      hg : AnalyticAt Complex g z₀
      ray : E → Complex → E := fun z t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      gray : E → Complex → Complex := fun z => Function.comp g (ray z)
      r : Real
      hr : GT.gt r 0
      hgr : HasSubset.Subset (Metric.ball z₀ r) (setOf fun x => AnalyticAt Complex g …
      z : E
      hz : Membership.mem (Metric.sphere 0 1) z
      hrz : Not (Filter.Eventually (fun t => Eq (gray z t) (gray z 0)) (nhds 0))
      h1 : AnalyticAt Complex (gray z) 0
      h7 : LE.le (nhds (g z₀)) (Function.comp (Filter.map g) (Filter.map (ray z)) (n …
      h10 : Continuous fun t => HAdd.hAdd z₀ (HSMul.hSMul t z)
      ⊢ LE.le (Filter.map (ray z) (nhds 0)) (nhds z₀)
    -/
    simpa using h10.tendsto 0
    /-
      🎉 no goals
    -/


/-- The *open mapping theorem* for holomorphic functions, global version: if a function `g : E → ℂ`
is analytic on a connected set `U`, then either it is constant on `U`, or it is open on `U` (in the
sense that it maps any open set contained in `U` to an open set in `ℂ`). -/
theorem AnalyticOnNhd.is_constant_or_isOpen (hg : AnalyticOnNhd ℂ g U) (hU : IsPreconnected U) :
    (∃ w, ∀ z ∈ U, g z = w) ∨ ∀ s ⊆ U, IsOpen s → IsOpen (g '' s) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    U : Set E
    g : E → Complex
    hg : AnalyticOnNhd Complex g U
    hU : IsPreconnected U
    ⊢ Or (Exists fun w => ∀ (z : E), Membership.mem U z → Eq (g z) w) (∀ (s : Set  …
  -/
  by_cases h : ∃ z₀ ∈ U, ∀ᶠ z in 𝓝 z₀, g z = g z₀
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      U : Set E
      g : E → Complex
      hg : AnalyticOnNhd Complex g U
      hU : IsPreconnected U
      h : Exists fun z₀ => And (Membership.mem U z₀) (Filter.Eventually (fun z => Eq …
      ⊢ Or (Exists fun w => ∀ (z : E), Membership.mem U z → Eq (g z) w) (∀ (s : Set  …
    -/
  · obtain ⟨z₀, hz₀, h⟩ := h
    /-
      case pos.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      U : Set E
      g : E → Complex
      hg : AnalyticOnNhd Complex g U
      hU : IsPreconnected U
      z₀ : E
      hz₀ : Membership.mem U z₀
      h : Filter.Eventually (fun z => Eq (g z) (g z₀)) (nhds z₀)
      ⊢ Or (Exists fun w => ∀ (z : E), Membership.mem U z → Eq (g z) w) (∀ (s : Set  …
    -/
    exact Or.inl ⟨g z₀, hg.eqOn_of_preconnected_of_eventuallyEq analyticOnNhd_const hU hz₀ h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      U : Set E
      g : E → Complex
      hg : AnalyticOnNhd Complex g U
      hU : IsPreconnected U
      h : Not (Exists fun z₀ => And (Membership.mem U z₀) (Filter.Eventually (fun z  …
      ⊢ Or (Exists fun w => ∀ (z : E), Membership.mem U z → Eq (g z) w) (∀ (s : Set  …
    -/
  · push_neg at h
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      U : Set E
      g : E → Complex
      hg : AnalyticOnNhd Complex g U
      hU : IsPreconnected U
      h : ∀ (z₀ : E), Membership.mem U z₀ → Not (Filter.Eventually (fun z => Eq (g z …
      ⊢ Or (Exists fun w => ∀ (z : E), Membership.mem U z → Eq (g z) w) (∀ (s : Set  …
    -/
    refine Or.inr fun s hs1 hs2 => isOpen_iff_mem_nhds.mpr ?_
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      U : Set E
      g : E → Complex
      hg : AnalyticOnNhd Complex g U
      hU : IsPreconnected U
      h : ∀ (z₀ : E), Membership.mem U z₀ → Not (Filter.Eventually (fun z => Eq (g z …
      s : Set E
      hs1 : HasSubset.Subset s U
      hs2 : IsOpen s
      ⊢ ∀ (x : Complex), Membership.mem (Set.image g s) x → Membership.mem (nhds x)  …
    -/
    rintro z ⟨w, hw1, rfl⟩
    exact (hg w (hs1 hw1)).eventually_constant_or_nhds_le_map_nhds.resolve_left (h w (hs1 hw1))
        (image_mem_map (hs2.mem_nhds hw1))


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.is_constant_or_isOpen := AnalyticOnNhd.is_constant_or_isOpen

