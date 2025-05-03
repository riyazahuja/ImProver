theorem Asymptotics.IsBigO.continuousMultilinearMap_apply_eq_zero {n : ℕ} {p : E[×n]→L[𝕜] F}
    (h : (fun y => p fun _ => y) =O[𝓝 0] fun y => ‖y‖ ^ (n + 1)) (y : E) : (p fun _ => y) = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    p : ContinuousMultilinearMap 𝕜 (fun i => E) F
    h : Asymptotics.IsBigO (nhds 0) (fun y => p fun x => y) fun y => HPow.hPow (No …
    y : E
    ⊢ Eq (p fun x => y) 0
  -/
  obtain ⟨c, c_pos, hc⟩ := h.exists_pos
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    p : ContinuousMultilinearMap 𝕜 (fun i => E) F
    h : Asymptotics.IsBigO (nhds 0) (fun y => p fun x => y) fun y => HPow.hPow (No …
    y : E
    c : Real
    c_pos : GT.gt c 0
    hc : Asymptotics.IsBigOWith c (nhds 0) (fun y => p fun x => y) fun y => HPow.h …
    ⊢ Eq (p fun x => y) 0
  -/
  obtain ⟨t, ht, t_open, z_mem⟩ := eventually_nhds_iff.mp (isBigOWith_iff.mp hc)
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    p : ContinuousMultilinearMap 𝕜 (fun i => E) F
    h : Asymptotics.IsBigO (nhds 0) (fun y => p fun x => y) fun y => HPow.hPow (No …
    y : E
    c : Real
    c_pos : GT.gt c 0
    hc : Asymptotics.IsBigOWith c (nhds 0) (fun y => p fun x => y) fun y => HPow.h …
    t : Set E
    ht : ∀ (y : E), Membership.mem t y → LE.le (Norm.norm (p fun x => y)) (HMul.hM …
    t_open : IsOpen t
    z_mem : Membership.mem t 0
    ⊢ Eq (p fun x => y) 0
  -/
  obtain ⟨δ, δ_pos, δε⟩ := (Metric.isOpen_iff.mp t_open) 0 z_mem
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    p : ContinuousMultilinearMap 𝕜 (fun i => E) F
    h : Asymptotics.IsBigO (nhds 0) (fun y => p fun x => y) fun y => HPow.hPow (No …
    y : E
    c : Real
    c_pos : GT.gt c 0
    hc : Asymptotics.IsBigOWith c (nhds 0) (fun y => p fun x => y) fun y => HPow.h …
    t : Set E
    ht : ∀ (y : E), Membership.mem t y → LE.le (Norm.norm (p fun x => y)) (HMul.hM …
    t_open : IsOpen t
    z_mem : Membership.mem t 0
    δ : Real
    δ_pos : GT.gt δ 0
    δε : HasSubset.Subset (Metric.ball 0 δ) t
    ⊢ Eq (p fun x => y) 0
  -/
  clear h hc z_mem
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    p : ContinuousMultilinearMap 𝕜 (fun i => E) F
    y : E
    c : Real
    c_pos : GT.gt c 0
    t : Set E
    ht : ∀ (y : E), Membership.mem t y → LE.le (Norm.norm (p fun x => y)) (HMul.hM …
    t_open : IsOpen t
    δ : Real
    δ_pos : GT.gt δ 0
    δε : HasSubset.Subset (Metric.ball 0 δ) t
    ⊢ Eq (p fun x => y) 0
  -/
  cases' n with n
  · exact norm_eq_zero.mp (by
      -- Porting note: the symmetric difference of the `simpa only` sets:
      -- added `zero_add, pow_one`
      -- removed `zero_pow, Ne.def, Nat.one_ne_zero, not_false_iff`
      simpa only [fin0_apply_norm, norm_eq_zero, norm_zero, zero_add, pow_one,
        mul_zero, norm_le_zero_iff] using ht 0 (δε (Metric.mem_ball_self δ_pos)))
  · refine Or.elim (Classical.em (y = 0))
      (fun hy => by simpa only [hy] using p.map_zero) fun hy => ?_
    /-
      case intro.intro.intro.intro.intro.intro.intro.succ
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      y : E
      c : Real
      c_pos : GT.gt c 0
      t : Set E
      t_open : IsOpen t
      δ : Real
      δ_pos : GT.gt δ 0
      δε : HasSubset.Subset (Metric.ball 0 δ) t
      n : Nat
      p : ContinuousMultilinearMap 𝕜 (fun i => E) F
      ht : ∀ (y : E), Membership.mem t y → LE.le (Norm.norm (p fun x => y)) (HMul.hM …
      hy : Not (Eq y 0)
      ⊢ Eq (p fun x => y) 0
    -/
    replace hy := norm_pos_iff.mpr hy
    /-
      case intro.intro.intro.intro.intro.intro.intro.succ
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      y : E
      c : Real
      c_pos : GT.gt c 0
      t : Set E
      t_open : IsOpen t
      δ : Real
      δ_pos : GT.gt δ 0
      δε : HasSubset.Subset (Metric.ball 0 δ) t
      n : Nat
      p : ContinuousMultilinearMap 𝕜 (fun i => E) F
      ht : ∀ (y : E), Membership.mem t y → LE.le (Norm.norm (p fun x => y)) (HMul.hM …
      hy : LT.lt 0 (Norm.norm y)
      ⊢ Eq (p fun x => y) 0
    -/
    refine norm_eq_zero.mp (le_antisymm (le_of_forall_pos_le_add fun ε ε_pos => ?_) (norm_nonneg _))
    /-
      case intro.intro.intro.intro.intro.intro.intro.succ
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      y : E
      c : Real
      c_pos : GT.gt c 0
      t : Set E
      t_open : IsOpen t
      δ : Real
      δ_pos : GT.gt δ 0
      δε : HasSubset.Subset (Metric.ball 0 δ) t
      n : Nat
      p : ContinuousMultilinearMap 𝕜 (fun i => E) F
      ht : ∀ (y : E), Membership.mem t y → LE.le (Norm.norm (p fun x => y)) (HMul.hM …
      hy : LT.lt 0 (Norm.norm y)
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ LE.le (Norm.norm (p fun x => y)) (HAdd.hAdd 0 ε)
    -/
    have h₀ := _root_.mul_pos c_pos (pow_pos hy (n.succ + 1))
    obtain ⟨k, k_pos, k_norm⟩ := NormedField.exists_norm_lt 𝕜
      (lt_min (mul_pos δ_pos (inv_pos.mpr hy)) (mul_pos ε_pos (inv_pos.mpr h₀)))
    have h₁ : ‖k • y‖ < δ := by
      rw [norm_smul]
      exact inv_mul_cancel_right₀ hy.ne.symm δ ▸
        mul_lt_mul_of_pos_right (lt_of_lt_of_le k_norm (min_le_left _ _)) hy
    have h₂ :=
      calc
        ‖p fun _ => k • y‖ ≤ c * ‖k • y‖ ^ (n.succ + 1) := by
          -- Porting note: now Lean wants `_root_.`
          simpa only [norm_pow, _root_.norm_norm] using ht (k • y) (δε (mem_ball_zero_iff.mpr h₁))
          --simpa only [norm_pow, norm_norm] using ht (k • y) (δε (mem_ball_zero_iff.mpr h₁))
        _ = ‖k‖ ^ n.succ * (‖k‖ * (c * ‖y‖ ^ (n.succ + 1))) := by
          -- Porting note: added `Nat.succ_eq_add_one` since otherwise `ring` does not conclude.
          simp only [norm_smul, mul_pow, Nat.succ_eq_add_one]
          -- Porting note: removed `rw [pow_succ]`, since it now becomes superfluous.
          ring
    have h₃ : ‖k‖ * (c * ‖y‖ ^ (n.succ + 1)) < ε :=
      inv_mul_cancel_right₀ h₀.ne.symm ε ▸
        mul_lt_mul_of_pos_right (lt_of_lt_of_le k_norm (min_le_right _ _)) h₀
    calc
      ‖p fun _ => y‖ = ‖k⁻¹ ^ n.succ‖ * ‖p fun _ => k • y‖ := by
        simpa only [inv_smul_smul₀ (norm_pos_iff.mp k_pos), norm_smul, Finset.prod_const,
          Finset.card_fin] using
          congr_arg norm (p.map_smul_univ (fun _ : Fin n.succ => k⁻¹) fun _ : Fin n.succ => k • y)
      _ ≤ ‖k⁻¹ ^ n.succ‖ * (‖k‖ ^ n.succ * (‖k‖ * (c * ‖y‖ ^ (n.succ + 1)))) := by gcongr
      _ = ‖(k⁻¹ * k) ^ n.succ‖ * (‖k‖ * (c * ‖y‖ ^ (n.succ + 1))) := by
        rw [← mul_assoc]
        simp [norm_mul, mul_pow]
      _ ≤ 0 + ε := by
        rw [inv_mul_cancel₀ (norm_pos_iff.mp k_pos)]
        simpa using h₃.le


/-- If a formal multilinear series `p` represents the zero function at `x : E`, then the
terms `p n (fun i ↦ y)` appearing in the sum are zero for any `n : ℕ`, `y : E`. -/
theorem HasFPowerSeriesAt.apply_eq_zero {p : FormalMultilinearSeries 𝕜 E F} {x : E}
    (h : HasFPowerSeriesAt 0 p x) (n : ℕ) : ∀ y : E, (p n fun _ => y) = 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    h : HasFPowerSeriesAt 0 p x
    n : Nat
    ⊢ ∀ (y : E), Eq ((p n) fun x => y) 0
  -/
  refine Nat.strong_induction_on n fun k hk => ?_
  have psum_eq : p.partialSum (k + 1) = fun y => p k fun _ => y := by
    funext z
    refine Finset.sum_eq_single _ (fun b hb hnb => ?_) fun hn => ?_
    · have := Finset.mem_range_succ_iff.mp hb
      simp only [hk b (this.lt_of_ne hnb), Pi.zero_apply]
    · exact False.elim (hn (Finset.mem_range.mpr (lt_add_one k)))
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    h : HasFPowerSeriesAt 0 p x
    n k : Nat
    hk : ∀ (m : Nat), LT.lt m k → ∀ (y : E), Eq ((p m) fun x => y) 0
    psum_eq : Eq (p.partialSum (HAdd.hAdd k 1)) fun y => (p k) fun x => y
    ⊢ ∀ (y : E), Eq ((p k) fun x => y) 0
  -/
  replace h := h.isBigO_sub_partialSum_pow k.succ
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    n k : Nat
    hk : ∀ (m : Nat), LT.lt m k → ∀ (y : E), Eq ((p m) fun x => y) 0
    psum_eq : Eq (p.partialSum (HAdd.hAdd k 1)) fun y => (p k) fun x => y
    h : Asymptotics.IsBigO (nhds 0) (fun y => HSub.hSub (0 (HAdd.hAdd x y)) (p.par …
    ⊢ ∀ (y : E), Eq ((p k) fun x => y) 0
  -/
  simp only [psum_eq, zero_sub, Pi.zero_apply, Asymptotics.isBigO_neg_left] at h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    n k : Nat
    hk : ∀ (m : Nat), LT.lt m k → ∀ (y : E), Eq ((p m) fun x => y) 0
    psum_eq : Eq (p.partialSum (HAdd.hAdd k 1)) fun y => (p k) fun x => y
    h : Asymptotics.IsBigO (nhds 0) (fun x => (p k) fun x_1 => x) fun y => HPow.hP …
    ⊢ ∀ (y : E), Eq ((p k) fun x => y) 0
  -/
  exact h.continuousMultilinearMap_apply_eq_zero
  /-
    🎉 no goals
  -/


/-- A one-dimensional formal multilinear series representing the zero function is zero. -/
theorem HasFPowerSeriesAt.eq_zero {p : FormalMultilinearSeries 𝕜 𝕜 E} {x : 𝕜}
    (h : HasFPowerSeriesAt 0 p x) : p = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    x : 𝕜
    h : HasFPowerSeriesAt 0 p x
    ⊢ Eq p 0
  -/
  ext n x
  /-
    case h.H
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    x✝ : 𝕜
    h : HasFPowerSeriesAt 0 p x✝
    n : Nat
    x : Fin n → 𝕜
    ⊢ Eq ((p n) x) ((0 n) x)
  -/
  rw [← mkPiRing_apply_one_eq_self (p n)]
  /-
    case h.H
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : FormalMultilinearSeries 𝕜 𝕜 E
    x✝ : 𝕜
    h : HasFPowerSeriesAt 0 p x✝
    n : Nat
    x : Fin n → 𝕜
    ⊢ Eq ((ContinuousMultilinearMap.mkPiRing 𝕜 (Fin n) ((p n) fun x => 1)) x) ((0  …
  -/
  simp [h.apply_eq_zero n 1]
  /-
    🎉 no goals
  -/


/-- One-dimensional formal multilinear series representing the same function are equal. -/
theorem HasFPowerSeriesAt.eq_formalMultilinearSeries {p₁ p₂ : FormalMultilinearSeries 𝕜 𝕜 E}
    {f : 𝕜 → E} {x : 𝕜} (h₁ : HasFPowerSeriesAt f p₁ x) (h₂ : HasFPowerSeriesAt f p₂ x) : p₁ = p₂ :=
                                                         /-
                                                           𝕜 : Type u_1
                                                           inst✝² : NontriviallyNormedField 𝕜
                                                           E : Type u_2
                                                           inst✝¹ : NormedAddCommGroup E
                                                           inst✝ : NormedSpace 𝕜 E
                                                           p₁ p₂ : FormalMultilinearSeries 𝕜 𝕜 E
                                                           f : 𝕜 → E
                                                           x : 𝕜
                                                           h₁ : HasFPowerSeriesAt f p₁ x
                                                           h₂ : HasFPowerSeriesAt f p₂ x
                                                           ⊢ HasFPowerSeriesAt 0 (HSub.hSub p₁ p₂) x
                                                         -/
  sub_eq_zero.mp (HasFPowerSeriesAt.eq_zero (x := x) (by simpa only [sub_self] using h₁.sub h₂))
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem HasFPowerSeriesAt.eq_formalMultilinearSeries_of_eventually
    {p q : FormalMultilinearSeries 𝕜 𝕜 E} {f g : 𝕜 → E} {x : 𝕜} (hp : HasFPowerSeriesAt f p x)
    (hq : HasFPowerSeriesAt g q x) (heq : ∀ᶠ z in 𝓝 x, f z = g z) : p = q :=
  (hp.congr heq).eq_formalMultilinearSeries hq


/-- A one-dimensional formal multilinear series representing a locally zero function is zero. -/
theorem HasFPowerSeriesAt.eq_zero_of_eventually {p : FormalMultilinearSeries 𝕜 𝕜 E} {f : 𝕜 → E}
    {x : 𝕜} (hp : HasFPowerSeriesAt f p x) (hf : f =ᶠ[𝓝 x] 0) : p = 0 :=
  (hp.congr hf).eq_zero


/-- If a function `f : 𝕜 → E` has two power series representations at `x`, then the given radii in
which convergence is guaranteed may be interchanged. This can be useful when the formal multilinear
series in one representation has a particularly nice form, but the other has a larger radius. -/
theorem HasFPowerSeriesOnBall.exchange_radius {p₁ p₂ : FormalMultilinearSeries 𝕜 𝕜 E} {f : 𝕜 → E}
    {r₁ r₂ : ℝ≥0∞} {x : 𝕜} (h₁ : HasFPowerSeriesOnBall f p₁ x r₁)
    (h₂ : HasFPowerSeriesOnBall f p₂ x r₂) : HasFPowerSeriesOnBall f p₁ x r₂ :=
  h₂.hasFPowerSeriesAt.eq_formalMultilinearSeries h₁.hasFPowerSeriesAt ▸ h₂


/-- If a function `f : 𝕜 → E` has power series representation `p` on a ball of some radius and for
each positive radius it has some power series representation, then `p` converges to `f` on the whole
`𝕜`. -/
theorem HasFPowerSeriesOnBall.r_eq_top_of_exists {f : 𝕜 → E} {r : ℝ≥0∞} {x : 𝕜}
    {p : FormalMultilinearSeries 𝕜 𝕜 E} (h : HasFPowerSeriesOnBall f p x r)
    (h' : ∀ (r' : ℝ≥0) (_ : 0 < r'), ∃ p' : FormalMultilinearSeries 𝕜 𝕜 E,
      HasFPowerSeriesOnBall f p' x r') :
    HasFPowerSeriesOnBall f p x ∞ :=
  { r_le := ENNReal.le_of_forall_pos_nnreal_lt fun r hr _ =>
      let ⟨_, hp'⟩ := h' r hr
      (h.exchange_radius hp').r_le
    r_pos := ENNReal.coe_lt_top
    hasSum := fun {y} _ =>
      let ⟨r', hr'⟩ := exists_gt ‖y‖₊
      let ⟨_, hp'⟩ := h' r' hr'.ne_bot.bot_lt
      (h.exchange_radius hp').hasSum <| mem_emetric_ball_zero_iff.mpr (ENNReal.coe_lt_coe.2 hr') }


/-- If an analytic function vanishes around a point, then it is uniformly zero along
a connected set. Superseded by `eqOn_zero_of_preconnected_of_locally_zero` which does not assume
completeness of the target space. -/
theorem eqOn_zero_of_preconnected_of_eventuallyEq_zero_aux [CompleteSpace F] {f : E → F} {U : Set E}
    (hf : AnalyticOnNhd 𝕜 f U) (hU : IsPreconnected U)
    {z₀ : E} (h₀ : z₀ ∈ U) (hfz₀ : f =ᶠ[𝓝 z₀] 0) :
    EqOn f 0 U := by
  /- Let `u` be the set of points around which `f` vanishes. It is clearly open. We have to show
    that its limit points in `U` still belong to it, from which the inclusion `U ⊆ u` will follow
    by connectedness. -/
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    ⊢ Set.EqOn f 0 U
  -/
  let u := {x | f =ᶠ[𝓝 x] 0}
  suffices main : closure u ∩ U ⊆ u by
    have Uu : U ⊆ u :=
      hU.subset_of_closure_inter_subset isOpen_setOf_eventually_nhds ⟨z₀, h₀, hfz₀⟩ main
    intro z hz
    simpa using mem_of_mem_nhds (Uu hz)
  /- Take a limit point `x`, then a ball `B (x, r)` on which it has a power series expansion, and
    then `y ∈ B (x, r/2) ∩ u`. Then `f` has a power series expansion on `B (y, r/2)` as it is
    contained in `B (x, r)`. All the coefficients in this series expansion vanish, as `f` is zero
    on a neighborhood of `y`. Therefore, `f` is zero on `B (y, r/2)`. As this ball contains `x`,
    it follows that `f` vanishes on a neighborhood of `x`, proving the claim. -/
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    ⊢ HasSubset.Subset (Inter.inter (closure u) U) u
  -/
  rintro x ⟨xu, xU⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    x : E
    xu : Membership.mem (closure u) x
    xU : Membership.mem U x
    ⊢ Membership.mem u x
  -/
  rcases hf x xU with ⟨p, r, hp⟩
  obtain ⟨y, yu, hxy⟩ : ∃ y ∈ u, edist x y < r / 2 :=
    EMetric.mem_closure_iff.1 xu (r / 2) (ENNReal.half_pos hp.r_pos.ne')
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    x : E
    xu : Membership.mem (closure u) x
    xU : Membership.mem U x
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesOnBall f p x r
    y : E
    yu : Membership.mem u y
    hxy : LT.lt (EDist.edist x y) (HDiv.hDiv r 2)
    ⊢ Membership.mem u x
  -/
  let q := p.changeOrigin (y - x)
  have has_series : HasFPowerSeriesOnBall f q y (r / 2) := by
    have A : (‖y - x‖₊ : ℝ≥0∞) < r / 2 := by rwa [edist_comm, edist_eq_coe_nnnorm_sub] at hxy
    have := hp.changeOrigin (A.trans_le ENNReal.half_le_self)
    simp only [add_sub_cancel] at this
    apply this.mono (ENNReal.half_pos hp.r_pos.ne')
    apply ENNReal.le_sub_of_add_le_left ENNReal.coe_ne_top
    apply (add_le_add A.le (le_refl (r / 2))).trans (le_of_eq _)
    exact ENNReal.add_halves _
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    x : E
    xu : Membership.mem (closure u) x
    xU : Membership.mem U x
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesOnBall f p x r
    y : E
    yu : Membership.mem u y
    hxy : LT.lt (EDist.edist x y) (HDiv.hDiv r 2)
    q : FormalMultilinearSeries 𝕜 E F := p.changeOrigin (HSub.hSub y x)
    has_series : HasFPowerSeriesOnBall f q y (HDiv.hDiv r 2)
    ⊢ Membership.mem u x
  -/
  have M : EMetric.ball y (r / 2) ∈ 𝓝 x := EMetric.isOpen_ball.mem_nhds hxy
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    x : E
    xu : Membership.mem (closure u) x
    xU : Membership.mem U x
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesOnBall f p x r
    y : E
    yu : Membership.mem u y
    hxy : LT.lt (EDist.edist x y) (HDiv.hDiv r 2)
    q : FormalMultilinearSeries 𝕜 E F := p.changeOrigin (HSub.hSub y x)
    has_series : HasFPowerSeriesOnBall f q y (HDiv.hDiv r 2)
    M : Membership.mem (nhds x) (EMetric.ball y (HDiv.hDiv r 2))
    ⊢ Membership.mem u x
  -/
  filter_upwards [M] with z hz
  /-
    case h
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    x : E
    xu : Membership.mem (closure u) x
    xU : Membership.mem U x
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesOnBall f p x r
    y : E
    yu : Membership.mem u y
    hxy : LT.lt (EDist.edist x y) (HDiv.hDiv r 2)
    q : FormalMultilinearSeries 𝕜 E F := p.changeOrigin (HSub.hSub y x)
    has_series : HasFPowerSeriesOnBall f q y (HDiv.hDiv r 2)
    M : Membership.mem (nhds x) (EMetric.ball y (HDiv.hDiv r 2))
    z : E
    hz : Membership.mem (EMetric.ball y (HDiv.hDiv r 2)) z
    ⊢ Eq (f z) (0 z)
  -/
  have A : HasSum (fun n : ℕ => q n fun _ : Fin n => z - y) (f z) := has_series.hasSum_sub hz
  have B : HasSum (fun n : ℕ => q n fun _ : Fin n => z - y) 0 := by
    have : HasFPowerSeriesAt 0 q y := has_series.hasFPowerSeriesAt.congr yu
    convert hasSum_zero (α := F) using 2
    ext n
    exact this.apply_eq_zero n _
  /-
    case h
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    u : Set E := setOf fun x => (nhds x).EventuallyEq f 0
    x : E
    xu : Membership.mem (closure u) x
    xU : Membership.mem U x
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesOnBall f p x r
    y : E
    yu : Membership.mem u y
    hxy : LT.lt (EDist.edist x y) (HDiv.hDiv r 2)
    q : FormalMultilinearSeries 𝕜 E F := p.changeOrigin (HSub.hSub y x)
    has_series : HasFPowerSeriesOnBall f q y (HDiv.hDiv r 2)
    M : Membership.mem (nhds x) (EMetric.ball y (HDiv.hDiv r 2))
    z : E
    hz : Membership.mem (EMetric.ball y (HDiv.hDiv r 2)) z
    A : HasSum (fun n => (q n) fun x => HSub.hSub z y) (f z)
    B : HasSum (fun n => (q n) fun x => HSub.hSub z y) 0
    ⊢ Eq (f z) (0 z)
  -/
  exact HasSum.unique A B
  /-
    🎉 no goals
  -/


/-- The *identity principle* for analytic functions: If an analytic function vanishes in a whole
neighborhood of a point `z₀`, then it is uniformly zero along a connected set. For a one-dimensional
version assuming only that the function vanishes at some points arbitrarily close to `z₀`, see
`eqOn_zero_of_preconnected_of_frequently_eq_zero`. -/
theorem eqOn_zero_of_preconnected_of_eventuallyEq_zero {f : E → F} {U : Set E}
    (hf : AnalyticOnNhd 𝕜 f U) (hU : IsPreconnected U)
    {z₀ : E} (h₀ : z₀ ∈ U) (hfz₀ : f =ᶠ[𝓝 z₀] 0) :
    EqOn f 0 U := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    ⊢ Set.EqOn f 0 U
  -/
  let F' := UniformSpace.Completion F
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    F' : Type u_3 := UniformSpace.Completion F
    ⊢ Set.EqOn f 0 U
  -/
  set e : F →L[𝕜] F' := UniformSpace.Completion.toComplL
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    F' : Type u_3 := UniformSpace.Completion F
    e : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    ⊢ Set.EqOn f 0 U
  -/
  have : AnalyticOnNhd 𝕜 (e ∘ f) U := fun x hx => (e.analyticAt _).comp (hf x hx)
  have A : EqOn (e ∘ f) 0 U := by
    apply eqOn_zero_of_preconnected_of_eventuallyEq_zero_aux this hU h₀
    filter_upwards [hfz₀] with x hx
    simp only [hx, Function.comp_apply, Pi.zero_apply, map_zero]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    F' : Type u_3 := UniformSpace.Completion F
    e : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    this : AnalyticOnNhd 𝕜 (Function.comp (⇑e) f) U
    A : Set.EqOn (Function.comp (⇑e) f) 0 U
    ⊢ Set.EqOn f 0 U
  -/
  intro z hz
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    F' : Type u_3 := UniformSpace.Completion F
    e : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    this : AnalyticOnNhd 𝕜 (Function.comp (⇑e) f) U
    A : Set.EqOn (Function.comp (⇑e) f) 0 U
    z : E
    hz : Membership.mem U z
    ⊢ Eq (f z) (0 z)
  -/
  have : e (f z) = e 0 := by simpa only using A hz
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfz₀ : (nhds z₀).EventuallyEq f 0
    F' : Type u_3 := UniformSpace.Completion F
    e : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    this✝ : AnalyticOnNhd 𝕜 (Function.comp (⇑e) f) U
    A : Set.EqOn (Function.comp (⇑e) f) 0 U
    z : E
    hz : Membership.mem U z
    this : Eq (e (f z)) (e 0)
    ⊢ Eq (f z) (0 z)
  -/
  exact UniformSpace.Completion.coe_injective F this
  /-
    🎉 no goals
  -/


/-- The *identity principle* for analytic functions: If two analytic functions coincide in a whole
neighborhood of a point `z₀`, then they coincide globally along a connected set.
For a one-dimensional version assuming only that the functions coincide at some points
arbitrarily close to `z₀`, see `eqOn_of_preconnected_of_frequently_eq`. -/
theorem eqOn_of_preconnected_of_eventuallyEq {f g : E → F} {U : Set E} (hf : AnalyticOnNhd 𝕜 f U)
    (hg : AnalyticOnNhd 𝕜 g U) (hU : IsPreconnected U) {z₀ : E} (h₀ : z₀ ∈ U) (hfg : f =ᶠ[𝓝 z₀] g) :
    EqOn f g U := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    U : Set E
    hf : AnalyticOnNhd 𝕜 f U
    hg : AnalyticOnNhd 𝕜 g U
    hU : IsPreconnected U
    z₀ : E
    h₀ : Membership.mem U z₀
    hfg : (nhds z₀).EventuallyEq f g
    ⊢ Set.EqOn f g U
  -/
  have hfg' : f - g =ᶠ[𝓝 z₀] 0 := hfg.mono fun z h => by simp [h]
  simpa [sub_eq_zero] using fun z hz =>
    (hf.sub hg).eqOn_zero_of_preconnected_of_eventuallyEq_zero hU h₀ hfg' hz


/-- The *identity principle* for analytic functions: If two analytic functions on a normed space
coincide in a neighborhood of a point `z₀`, then they coincide everywhere.
For a one-dimensional version assuming only that the functions coincide at some points
arbitrarily close to `z₀`, see `eq_of_frequently_eq`. -/
theorem eq_of_eventuallyEq {f g : E → F} [PreconnectedSpace E] (hf : AnalyticOnNhd 𝕜 f univ)
    (hg : AnalyticOnNhd 𝕜 g univ) {z₀ : E} (hfg : f =ᶠ[𝓝 z₀] g) : f = g :=
  funext fun x =>
    eqOn_of_preconnected_of_eventuallyEq hf hg isPreconnected_univ (mem_univ z₀) hfg (mem_univ x)


