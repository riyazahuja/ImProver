/-- **Removable singularity** theorem, weak version. If `f : ℂ → E` is differentiable in a punctured
neighborhood of a point and is continuous at this point, then it is analytic at this point. -/
theorem analyticAt_of_differentiable_on_punctured_nhds_of_continuousAt {f : ℂ → E} {c : ℂ}
    (hd : ∀ᶠ z in 𝓝[≠] c, DifferentiableAt ℂ f z) (hc : ContinuousAt f c) : AnalyticAt ℂ f c := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    hd : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhdsWithin c ( …
    hc : ContinuousAt f c
    ⊢ AnalyticAt Complex f c
  -/
  rcases (nhdsWithin_hasBasis nhds_basis_closedBall _).mem_iff.1 hd with ⟨R, hR0, hRs⟩
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    hd : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhdsWithin c ( …
    hc : ContinuousAt f c
    R : Real
    hR0 : LT.lt 0 R
    hRs : HasSubset.Subset (Inter.inter (Metric.closedBall c R) (HasCompl.compl (S …
    ⊢ AnalyticAt Complex f c
  -/
  lift R to ℝ≥0 using hR0.le
  replace hc : ContinuousOn f (closedBall c R) := by
    refine fun z hz => ContinuousAt.continuousWithinAt ?_
    rcases eq_or_ne z c with (rfl | hne)
    exacts [hc, (hRs ⟨hz, hne⟩).continuousAt]
  exact (hasFPowerSeriesOnBall_of_differentiable_off_countable (countable_singleton c) hc
    (fun z hz => hRs (diff_subset_diff_left ball_subset_closedBall hz)) hR0).analyticAt


theorem differentiableOn_compl_singleton_and_continuousAt_iff {f : ℂ → E} {s : Set ℂ} {c : ℂ}
    (hs : s ∈ 𝓝 c) :
    DifferentiableOn ℂ f (s \ {c}) ∧ ContinuousAt f c ↔ DifferentiableOn ℂ f s := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hs : Membership.mem (nhds c) s
    ⊢ Iff (And (DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c)) …
  -/
  refine ⟨?_, fun hd => ⟨hd.mono diff_subset, (hd.differentiableAt hs).continuousAt⟩⟩
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hs : Membership.mem (nhds c) s
    ⊢ And (DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))) (Co …
  -/
  rintro ⟨hd, hc⟩ x hx
  /-
    case intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hs : Membership.mem (nhds c) s
    hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))
    hc : ContinuousAt f c
    x : Complex
    hx : Membership.mem s x
    ⊢ DifferentiableWithinAt Complex f s x
  -/
  rcases eq_or_ne x c with (rfl | hne)
  · refine (analyticAt_of_differentiable_on_punctured_nhds_of_continuousAt
      ?_ hc).differentiableAt.differentiableWithinAt
    /-
      case intro.inl
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      s : Set Complex
      x : Complex
      hx : Membership.mem s x
      hs : Membership.mem (nhds x) s
      hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton x))
      hc : ContinuousAt f x
      ⊢ Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhdsWithin x (Has …
    -/
    refine eventually_nhdsWithin_iff.2 ((eventually_mem_nhds_iff.2 hs).mono fun z hz hzx => ?_)
    /-
      case intro.inl
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      s : Set Complex
      x : Complex
      hx : Membership.mem s x
      hs : Membership.mem (nhds x) s
      hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton x))
      hc : ContinuousAt f x
      z : Complex
      hz : Membership.mem (nhds z) s
      hzx : Membership.mem (HasCompl.compl (Singleton.singleton x)) z
      ⊢ DifferentiableAt Complex f z
    -/
    exact hd.differentiableAt (inter_mem hz (isOpen_ne.mem_nhds hzx))
    /-
      🎉 no goals
    -/
  · simpa only [DifferentiableWithinAt, HasFDerivWithinAt, hne.nhdsWithin_diff_singleton] using
      hd x ⟨hx, hne⟩


theorem differentiableOn_dslope {f : ℂ → E} {s : Set ℂ} {c : ℂ} (hc : s ∈ 𝓝 c) :
    DifferentiableOn ℂ (dslope f c) s ↔ DifferentiableOn ℂ f s :=
  ⟨fun h => h.of_dslope, fun h =>
    (differentiableOn_compl_singleton_and_continuousAt_iff hc).mp <|
      ⟨Iff.mpr (differentiableOn_dslope_of_nmem fun h => h.2 rfl) (h.mono diff_subset),
        continuousAt_dslope_same.2 <| h.differentiableAt hc⟩⟩


/-- **Removable singularity** theorem: if `s` is a neighborhood of `c : ℂ`, a function `f : ℂ → E`
is complex differentiable on `s \ {c}`, and $f(z) - f(c)=o((z-c)^{-1})$, then `f` redefined to be
equal to `limUnder (𝓝[≠] c) f` at `c` is complex differentiable on `s`. -/
theorem differentiableOn_update_limUnder_of_isLittleO {f : ℂ → E} {s : Set ℂ} {c : ℂ} (hc : s ∈ 𝓝 c)
    (hd : DifferentiableOn ℂ f (s \ {c}))
    (ho : (fun z => f z - f c) =o[𝓝[≠] c] fun z => (z - c)⁻¹) :
    DifferentiableOn ℂ (update f c (limUnder (𝓝[≠] c) f)) s := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hc : Membership.mem (nhds c) s
    hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    ⊢ DifferentiableOn Complex (Function.update f c (limUnder (nhdsWithin c (HasCo …
  -/
  set F : ℂ → E := fun z => (z - c) • f z
  suffices DifferentiableOn ℂ F (s \ {c}) ∧ ContinuousAt F c by
    rw [differentiableOn_compl_singleton_and_continuousAt_iff hc, ← differentiableOn_dslope hc,
      dslope_sub_smul] at this
    have hc : Tendsto f (𝓝[≠] c) (𝓝 (deriv F c)) :=
      continuousAt_update_same.mp (this.continuousOn.continuousAt hc)
    rwa [hc.limUnder_eq]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hc : Membership.mem (nhds c) s
    hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    F : Complex → E := fun z => HSMul.hSMul (HSub.hSub z c) (f z)
    ⊢ And (DifferentiableOn Complex F (SDiff.sdiff s (Singleton.singleton c))) (Co …
  -/
  refine ⟨(differentiableOn_id.sub_const _).smul hd, ?_⟩
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hc : Membership.mem (nhds c) s
    hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    F : Complex → E := fun z => HSMul.hSMul (HSub.hSub z c) (f z)
    ⊢ ContinuousAt F c
  -/
  rw [← continuousWithinAt_compl_self]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hc : Membership.mem (nhds c) s
    hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    F : Complex → E := fun z => HSMul.hSMul (HSub.hSub z c) (f z)
    ⊢ ContinuousWithinAt F (HasCompl.compl (Singleton.singleton c)) c
  -/
  have H := ho.tendsto_inv_smul_nhds_zero
  have H' : Tendsto (fun z => (z - c) • f c) (𝓝[≠] c) (𝓝 (F c)) :=
    (continuousWithinAt_id.tendsto.sub tendsto_const_nhds).smul tendsto_const_nhds
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    c : Complex
    hc : Membership.mem (nhds c) s
    hd : DifferentiableOn Complex f (SDiff.sdiff s (Singleton.singleton c))
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    F : Complex → E := fun z => HSMul.hSMul (HSub.hSub z c) (f z)
    H : Filter.Tendsto (fun x => HSMul.hSMul (Inv.inv (Inv.inv (HSub.hSub x c))) ( …
    H' : Filter.Tendsto (fun z => HSMul.hSMul (HSub.hSub z c) (f c)) (nhdsWithin c …
    ⊢ ContinuousWithinAt F (HasCompl.compl (Singleton.singleton c)) c
  -/
  simpa [← smul_add, ContinuousWithinAt] using H.add H'
  /-
    🎉 no goals
  -/


/-- **Removable singularity** theorem: if `s` is a punctured neighborhood of `c : ℂ`, a function
`f : ℂ → E` is complex differentiable on `s`, and $f(z) - f(c)=o((z-c)^{-1})$, then `f` redefined to
be equal to `limUnder (𝓝[≠] c) f` at `c` is complex differentiable on `{c} ∪ s`. -/
theorem differentiableOn_update_limUnder_insert_of_isLittleO {f : ℂ → E} {s : Set ℂ} {c : ℂ}
    (hc : s ∈ 𝓝[≠] c) (hd : DifferentiableOn ℂ f s)
    (ho : (fun z => f z - f c) =o[𝓝[≠] c] fun z => (z - c)⁻¹) :
    DifferentiableOn ℂ (update f c (limUnder (𝓝[≠] c) f)) (insert c s) :=
  differentiableOn_update_limUnder_of_isLittleO (insert_mem_nhds_iff.2 hc)
    (hd.mono fun _ hz => hz.1.resolve_left hz.2) ho


/-- **Removable singularity** theorem: if `s` is a neighborhood of `c : ℂ`, a function `f : ℂ → E`
is complex differentiable and is bounded on `s \ {c}`, then `f` redefined to be equal to
`limUnder (𝓝[≠] c) f` at `c` is complex differentiable on `s`. -/
theorem differentiableOn_update_limUnder_of_bddAbove {f : ℂ → E} {s : Set ℂ} {c : ℂ} (hc : s ∈ 𝓝 c)
    (hd : DifferentiableOn ℂ f (s \ {c})) (hb : BddAbove (norm ∘ f '' (s \ {c}))) :
    DifferentiableOn ℂ (update f c (limUnder (𝓝[≠] c) f)) s :=
  differentiableOn_update_limUnder_of_isLittleO hc hd <| IsBoundedUnder.isLittleO_sub_self_inv <|
    let ⟨C, hC⟩ := hb
    ⟨C + ‖f c‖, eventually_map.2 <| mem_nhdsWithin_iff_exists_mem_nhds_inter.2
      ⟨s, hc, fun _ hz => norm_sub_le_of_le (hC <| mem_image_of_mem _ hz) le_rfl⟩⟩


/-- **Removable singularity** theorem: if a function `f : ℂ → E` is complex differentiable on a
punctured neighborhood of `c` and $f(z) - f(c)=o((z-c)^{-1})$, then `f` has a limit at `c`. -/
theorem tendsto_limUnder_of_differentiable_on_punctured_nhds_of_isLittleO {f : ℂ → E} {c : ℂ}
    (hd : ∀ᶠ z in 𝓝[≠] c, DifferentiableAt ℂ f z)
    (ho : (fun z => f z - f c) =o[𝓝[≠] c] fun z => (z - c)⁻¹) :
    Tendsto f (𝓝[≠] c) (𝓝 <| limUnder (𝓝[≠] c) f) := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    hd : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhdsWithin c ( …
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    ⊢ Filter.Tendsto f (nhdsWithin c (HasCompl.compl (Singleton.singleton c))) (nh …
  -/
  rw [eventually_nhdsWithin_iff] at hd
  have : DifferentiableOn ℂ f ({z | z ≠ c → DifferentiableAt ℂ f z} \ {c}) := fun z hz =>
    (hz.1 hz.2).differentiableWithinAt
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    hd : Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.sin …
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    this : DifferentiableOn Complex f (SDiff.sdiff (setOf fun z => Ne z c → Differ …
    ⊢ Filter.Tendsto f (nhdsWithin c (HasCompl.compl (Singleton.singleton c))) (nh …
  -/
  have H := differentiableOn_update_limUnder_of_isLittleO hd this ho
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    hd : Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.sin …
    ho : Asymptotics.IsLittleO (nhdsWithin c (HasCompl.compl (Singleton.singleton  …
    this : DifferentiableOn Complex f (SDiff.sdiff (setOf fun z => Ne z c → Differ …
    H : DifferentiableOn Complex (Function.update f c (limUnder (nhdsWithin c (Has …
    ⊢ Filter.Tendsto f (nhdsWithin c (HasCompl.compl (Singleton.singleton c))) (nh …
  -/
  exact continuousAt_update_same.1 (H.differentiableAt hd).continuousAt
  /-
    🎉 no goals
  -/


/-- **Removable singularity** theorem: if a function `f : ℂ → E` is complex differentiable and
bounded on a punctured neighborhood of `c`, then `f` has a limit at `c`. -/
theorem tendsto_limUnder_of_differentiable_on_punctured_nhds_of_bounded_under {f : ℂ → E} {c : ℂ}
    (hd : ∀ᶠ z in 𝓝[≠] c, DifferentiableAt ℂ f z)
    (hb : IsBoundedUnder (· ≤ ·) (𝓝[≠] c) fun z => ‖f z - f c‖) :
    Tendsto f (𝓝[≠] c) (𝓝 <| limUnder (𝓝[≠] c) f) :=
  tendsto_limUnder_of_differentiable_on_punctured_nhds_of_isLittleO hd hb.isLittleO_sub_self_inv


/-- The Cauchy formula for the derivative of a holomorphic function. -/
theorem two_pi_I_inv_smul_circleIntegral_sub_sq_inv_smul_of_differentiable {U : Set ℂ}
    (hU : IsOpen U) {c w₀ : ℂ} {R : ℝ} {f : ℂ → E} (hc : closedBall c R ⊆ U)
    (hf : DifferentiableOn ℂ f U) (hw₀ : w₀ ∈ ball c R) :
    ((2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), ((z - w₀) ^ 2)⁻¹ • f z) = deriv f w₀ := by
  -- We apply the removable singularity theorem and the Cauchy formula to `dslope f w₀`
  have hf' : DifferentiableOn ℂ (dslope f w₀) U :=
    (differentiableOn_dslope (hU.mem_nhds ((ball_subset_closedBall.trans hc) hw₀))).mpr hf
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    U : Set Complex
    hU : IsOpen U
    c w₀ : Complex
    R : Real
    f : Complex → E
    hc : HasSubset.Subset (Metric.closedBall c R) U
    hf : DifferentiableOn Complex f U
    hw₀ : Membership.mem (Metric.ball c R) w₀
    hf' : DifferentiableOn Complex (dslope f w₀) U
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (circ …
  -/
  have h0 := (hf'.diffContOnCl_ball hc).two_pi_i_inv_smul_circleIntegral_sub_inv_smul hw₀
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    U : Set Complex
    hU : IsOpen U
    c w₀ : Complex
    R : Real
    f : Complex → E
    hc : HasSubset.Subset (Metric.closedBall c R) U
    hf : DifferentiableOn Complex f U
    hw₀ : Membership.mem (Metric.ball c R) w₀
    hf' : DifferentiableOn Complex (dslope f w₀) U
    h0 : Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (c …
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (circ …
  -/
  rw [← dslope_same, ← h0]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    U : Set Complex
    hU : IsOpen U
    c w₀ : Complex
    R : Real
    f : Complex → E
    hc : HasSubset.Subset (Metric.closedBall c R) U
    hf : DifferentiableOn Complex f U
    hw₀ : Membership.mem (Metric.ball c R) w₀
    hf' : DifferentiableOn Complex (dslope f w₀) U
    h0 : Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (c …
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (circ …
  -/
  congr 1
  /-
    case e_a
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    U : Set Complex
    hU : IsOpen U
    c w₀ : Complex
    R : Real
    f : Complex → E
    hc : HasSubset.Subset (Metric.closedBall c R) U
    hf : DifferentiableOn Complex f U
    hw₀ : Membership.mem (Metric.ball c R) w₀
    hf' : DifferentiableOn Complex (dslope f w₀) U
    h0 : Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (c …
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HPow.hPow (HSub.hSub z w₀ …
  -/
  trans ∮ z in C(c, R), ((z - w₀) ^ 2)⁻¹ • (f z - f w₀)
  · have h1 : ContinuousOn (fun z : ℂ => ((z - w₀) ^ 2)⁻¹) (sphere c R) := by
      refine ((continuous_id'.sub continuous_const).pow 2).continuousOn.inv₀ fun w hw h => ?_
      exact sphere_disjoint_ball.ne_of_mem hw hw₀ (sub_eq_zero.mp (sq_eq_zero_iff.mp h))
    have h2 : CircleIntegrable (fun z : ℂ => ((z - w₀) ^ 2)⁻¹ • f z) c R := by
      refine ContinuousOn.circleIntegrable (pos_of_mem_ball hw₀).le ?_
      exact h1.smul (hf.continuousOn.mono (sphere_subset_closedBall.trans hc))
    have h3 : CircleIntegrable (fun z : ℂ => ((z - w₀) ^ 2)⁻¹ • f w₀) c R :=
      ContinuousOn.circleIntegrable (pos_of_mem_ball hw₀).le (h1.smul continuousOn_const)
    have h4 : (∮ z : ℂ in C(c, R), ((z - w₀) ^ 2)⁻¹) = 0 := by
      simpa using circleIntegral.integral_sub_zpow_of_ne (by decide : (-2 : ℤ) ≠ -1) c w₀ R
    simp only [smul_sub, circleIntegral.integral_sub h2 h3, h4, circleIntegral.integral_smul_const,
      zero_smul, sub_zero]
    /-
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      U : Set Complex
      hU : IsOpen U
      c w₀ : Complex
      R : Real
      f : Complex → E
      hc : HasSubset.Subset (Metric.closedBall c R) U
      hf : DifferentiableOn Complex f U
      hw₀ : Membership.mem (Metric.ball c R) w₀
      hf' : DifferentiableOn Complex (dslope f w₀) U
      h0 : Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (c …
      ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HPow.hPow (HSub.hSub z w₀ …
    -/
  · refine circleIntegral.integral_congr (pos_of_mem_ball hw₀).le fun z hz => ?_
    simp only [dslope_of_ne, Metric.sphere_disjoint_ball.ne_of_mem hz hw₀, slope, ← smul_assoc, sq,
      mul_inv, Ne, not_false_iff, vsub_eq_sub, Algebra.id.smul_eq_mul]


