local postfix:100 "̂" => UniformSpace.Completion


theorem norm_max_aux₁ [CompleteSpace F] {f : ℂ → F} {z w : ℂ}
    (hd : DiffContOnCl ℂ f (ball z (dist w z)))
    (hz : IsMaxOn (norm ∘ f) (closedBall z (dist w z)) z) : ‖f w‖ = ‖f z‖ := by
  -- Consider a circle of radius `r = dist w z`.
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    hd : DiffContOnCl Complex f (Metric.ball z (Dist.dist w z))
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z (Dist.dist w z)) z
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  set r : ℝ := dist w z
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  have hw : w ∈ closedBall z r := mem_closedBall.2 le_rfl
  -- Assume the converse. Since `‖f w‖ ≤ ‖f z‖`, we have `‖f w‖ < ‖f z‖`.
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  refine (isMaxOn_iff.1 hz _ hw).antisymm (not_lt.1 ?_)
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    ⊢ Not (LT.lt (Function.comp Norm.norm f w) (Function.comp Norm.norm f z))
  -/
  rintro hw_lt : ‖f w‖ < ‖f z‖
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
    ⊢ False
  -/
  have hr : 0 < r := dist_pos.2 (ne_of_apply_ne (norm ∘ f) hw_lt.ne)
  -- Due to Cauchy integral formula, it suffices to prove the following inequality.
  suffices ‖∮ ζ in C(z, r), (ζ - z)⁻¹ • f ζ‖ < 2 * π * ‖f z‖ by
    refine this.ne ?_
    have A : (∮ ζ in C(z, r), (ζ - z)⁻¹ • f ζ) = (2 * π * I : ℂ) • f z :=
      hd.circleIntegral_sub_inv_smul (mem_ball_self hr)
    simp [A, norm_smul, Real.pi_pos.le]
  suffices ‖∮ ζ in C(z, r), (ζ - z)⁻¹ • f ζ‖ < 2 * π * r * (‖f z‖ / r) by
    rwa [mul_assoc, mul_div_cancel₀ _ hr.ne'] at this
  /- This inequality is true because `‖(ζ - z)⁻¹ • f ζ‖ ≤ ‖f z‖ / r` for all `ζ` on the circle and
    this inequality is strict at `ζ = w`. -/
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
    hr : LT.lt 0 r
    ⊢ LT.lt (Norm.norm (circleIntegral (fun ζ => HSMul.hSMul (Inv.inv (HSub.hSub ζ …
  -/
  have hsub : sphere z r ⊆ closedBall z r := sphere_subset_closedBall
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
    hr : LT.lt 0 r
    hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
    ⊢ LT.lt (Norm.norm (circleIntegral (fun ζ => HSMul.hSMul (Inv.inv (HSub.hSub ζ …
  -/
  refine circleIntegral.norm_integral_lt_of_norm_le_const_of_lt hr ?_ ?_ ⟨w, rfl, ?_⟩
    /-
      case refine_1
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ⊢ ContinuousOn (fun ζ => HSMul.hSMul (Inv.inv (HSub.hSub ζ z)) (f ζ)) (Metric. …
    -/
  · show ContinuousOn (fun ζ : ℂ => (ζ - z)⁻¹ • f ζ) (sphere z r)
    /-
      case refine_1
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ⊢ ContinuousOn (fun ζ => HSMul.hSMul (Inv.inv (HSub.hSub ζ z)) (f ζ)) (Metric. …
    -/
    refine ((continuousOn_id.sub continuousOn_const).inv₀ ?_).smul (hd.continuousOn_ball.mono hsub)
    /-
      case refine_1
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ⊢ ∀ (x : Complex), Membership.mem (Metric.sphere z r) x → Ne (HSub.hSub (id x) …
    -/
    exact fun ζ hζ => sub_ne_zero.2 (ne_of_mem_sphere hζ hr.ne')
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ⊢ ∀ (z_1 : Complex), Membership.mem (Metric.sphere z r) z_1 → LE.le (Norm.norm …
    -/
  · show ∀ ζ ∈ sphere z r, ‖(ζ - z)⁻¹ • f ζ‖ ≤ ‖f z‖ / r
    /-
      case refine_2
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ⊢ ∀ (ζ : Complex), Membership.mem (Metric.sphere z r) ζ → LE.le (Norm.norm (HS …
    -/
    rintro ζ (hζ : abs (ζ - z) = r)
    /-
      case refine_2
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ζ : Complex
      hζ : Eq (Complex.abs (HSub.hSub ζ z)) r
      ⊢ LE.le (Norm.norm (HSMul.hSMul (Inv.inv (HSub.hSub ζ z)) (f ζ))) (HDiv.hDiv ( …
    -/
    rw [le_div_iff₀ hr, norm_smul, norm_inv, norm_eq_abs, hζ, mul_comm, mul_inv_cancel_left₀ hr.ne']
    /-
      case refine_2
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace Complex F
      inst✝ : CompleteSpace F
      f : Complex → F
      z w : Complex
      r : Real := Dist.dist w z
      hd : DiffContOnCl Complex f (Metric.ball z r)
      hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
      hw : Membership.mem (Metric.closedBall z r) w
      hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
      hr : LT.lt 0 r
      hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
      ζ : Complex
      hζ : Eq (Complex.abs (HSub.hSub ζ z)) r
      ⊢ LE.le (Norm.norm (f ζ)) (Norm.norm (f z))
    -/
    exact hz (hsub hζ)
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
    hr : LT.lt 0 r
    hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
    ⊢ LT.lt (Norm.norm (HSMul.hSMul (Inv.inv (HSub.hSub w z)) (f w))) (HDiv.hDiv ( …
  -/
  show ‖(w - z)⁻¹ • f w‖ < ‖f z‖ / r
  /-
    case refine_3
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
    hr : LT.lt 0 r
    hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
    ⊢ LT.lt (Norm.norm (HSMul.hSMul (Inv.inv (HSub.hSub w z)) (f w))) (HDiv.hDiv ( …
  -/
  rw [norm_smul, norm_inv, norm_eq_abs, ← div_eq_inv_mul]
  /-
    case refine_3
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    f : Complex → F
    z w : Complex
    r : Real := Dist.dist w z
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z r) z
    hw : Membership.mem (Metric.closedBall z r) w
    hw_lt : LT.lt (Norm.norm (f w)) (Norm.norm (f z))
    hr : LT.lt 0 r
    hsub : HasSubset.Subset (Metric.sphere z r) (Metric.closedBall z r)
    ⊢ LT.lt (HDiv.hDiv (Norm.norm (f w)) (Complex.abs (HSub.hSub w z))) (HDiv.hDiv …
  -/
  exact (div_lt_div_iff_of_pos_right hr).2 hw_lt
  /-
    🎉 no goals
  -/


theorem norm_max_aux₂ {f : ℂ → F} {z w : ℂ} (hd : DiffContOnCl ℂ f (ball z (dist w z)))
    (hz : IsMaxOn (norm ∘ f) (closedBall z (dist w z)) z) : ‖f w‖ = ‖f z‖ := by
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    z w : Complex
    hd : DiffContOnCl Complex f (Metric.ball z (Dist.dist w z))
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z (Dist.dist w z)) z
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  set e : F →L[ℂ] F̂ := UniformSpace.Completion.toComplL
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    z w : Complex
    hd : DiffContOnCl Complex f (Metric.ball z (Dist.dist w z))
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.closedBall z (Dist.dist w z)) z
    e : ContinuousLinearMap (RingHom.id Complex) F (UniformSpace.Completion F) :=  …
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  have he : ∀ x, ‖e x‖ = ‖x‖ := UniformSpace.Completion.norm_coe
  replace hz : IsMaxOn (norm ∘ e ∘ f) (closedBall z (dist w z)) z := by
    simpa only [IsMaxOn, Function.comp_def, he] using hz
  simpa only [he, Function.comp_def]
    using norm_max_aux₁ (e.differentiable.comp_diffContOnCl hd) hz


theorem norm_max_aux₃ {f : ℂ → F} {z w : ℂ} {r : ℝ} (hr : dist w z = r)
    (hd : DiffContOnCl ℂ f (ball z r)) (hz : IsMaxOn (norm ∘ f) (ball z r) z) : ‖f w‖ = ‖f z‖ := by
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    z w : Complex
    r : Real
    hr : Eq (Dist.dist w z) r
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  subst r
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    z w : Complex
    hd : DiffContOnCl Complex f (Metric.ball z (Dist.dist w z))
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z (Dist.dist w z)) z
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  rcases eq_or_ne w z with (rfl | hne); · rfl
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    z w : Complex
    hd : DiffContOnCl Complex f (Metric.ball z (Dist.dist w z))
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z (Dist.dist w z)) z
    hne : Ne w z
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  rw [← dist_ne_zero] at hne
  /-
    case inr
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    z w : Complex
    hd : DiffContOnCl Complex f (Metric.ball z (Dist.dist w z))
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z (Dist.dist w z)) z
    hne : Ne (Dist.dist w z) 0
    ⊢ Eq (Norm.norm (f w)) (Norm.norm (f z))
  -/
  exact norm_max_aux₂ hd (closure_ball z hne ▸ hz.closure hd.continuousOn.norm)
  /-
    🎉 no goals
  -/


/-- **Maximum modulus principle** on a closed ball: if `f : E → F` is continuous on a closed ball,
is complex differentiable on the corresponding open ball, and the norm `‖f w‖` takes its maximum
value on the open ball at its center, then the norm `‖f w‖` is constant on the closed ball. -/
theorem norm_eqOn_closedBall_of_isMaxOn {f : E → F} {z : E} {r : ℝ}
    (hd : DiffContOnCl ℂ f (ball z r)) (hz : IsMaxOn (norm ∘ f) (ball z r) z) :
    EqOn (norm ∘ f) (const E ‖f z‖) (closedBall z r) := by
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    ⊢ Set.EqOn (Function.comp Norm.norm f) (Function.const E (Norm.norm (f z))) (M …
  -/
  intro w hw
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    w : E
    hw : Membership.mem (Metric.closedBall z r) w
    ⊢ Eq (Function.comp Norm.norm f w) (Function.const E (Norm.norm (f z)) w)
  -/
  rw [mem_closedBall, dist_comm] at hw
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    w : E
    hw : LE.le (Dist.dist z w) r
    ⊢ Eq (Function.comp Norm.norm f w) (Function.const E (Norm.norm (f z)) w)
  -/
  rcases eq_or_ne z w with (rfl | hne); · rfl
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    w : E
    hw : LE.le (Dist.dist z w) r
    hne : Ne z w
    ⊢ Eq (Function.comp Norm.norm f w) (Function.const E (Norm.norm (f z)) w)
  -/
  set e := (lineMap z w : ℂ → E)
  /-
    case inr
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    w : E
    hw : LE.le (Dist.dist z w) r
    hne : Ne z w
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    ⊢ Eq (Function.comp Norm.norm f w) (Function.const E (Norm.norm (f z)) w)
  -/
  have hde : Differentiable ℂ e := (differentiable_id.smul_const (w - z)).add_const z
  /-
    case inr
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    w : E
    hw : LE.le (Dist.dist z w) r
    hne : Ne z w
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    hde : Differentiable Complex e
    ⊢ Eq (Function.comp Norm.norm f w) (Function.const E (Norm.norm (f z)) w)
  -/
  suffices ‖(f ∘ e) (1 : ℂ)‖ = ‖(f ∘ e) (0 : ℂ)‖ by simpa [e]
  /-
    case inr
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    z : E
    r : Real
    hd : DiffContOnCl Complex f (Metric.ball z r)
    hz : IsMaxOn (Function.comp Norm.norm f) (Metric.ball z r) z
    w : E
    hw : LE.le (Dist.dist z w) r
    hne : Ne z w
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    hde : Differentiable Complex e
    ⊢ Eq (Norm.norm (Function.comp f e 1)) (Norm.norm (Function.comp f e 0))
  -/
  have hr : dist (1 : ℂ) 0 = 1 := by simp
  have hball : MapsTo e (ball 0 1) (ball z r) := by
    refine ((lipschitzWith_lineMap z w).mapsTo_ball (mt nndist_eq_zero.1 hne) 0 1).mono
      Subset.rfl ?_
    simpa only [lineMap_apply_zero, mul_one, coe_nndist] using ball_subset_ball hw
  exact norm_max_aux₃ hr (hd.comp hde.diffContOnCl hball)
      (hz.comp_mapsTo hball (lineMap_apply_zero z w))


/-- **Maximum modulus principle**: if `f : E → F` is complex differentiable on a set `s`, the norm
of `f` takes it maximum on `s` at `z`, and `w` is a point such that the closed ball with center `z`
and radius `dist w z` is included in `s`, then `‖f w‖ = ‖f z‖`. -/
theorem norm_eq_norm_of_isMaxOn_of_ball_subset {f : E → F} {s : Set E} {z w : E}
    (hd : DiffContOnCl ℂ f s) (hz : IsMaxOn (norm ∘ f) s z) (hsub : ball z (dist w z) ⊆ s) :
    ‖f w‖ = ‖f z‖ :=
  norm_eqOn_closedBall_of_isMaxOn (hd.mono hsub) (hz.on_subset hsub) (mem_closedBall.2 le_rfl)


/-- **Maximum modulus principle**: if `f : E → F` is complex differentiable in a neighborhood of `c`
and the norm `‖f z‖` has a local maximum at `c`, then `‖f z‖` is locally constant in a neighborhood
of `c`. -/
theorem norm_eventually_eq_of_isLocalMax {f : E → F} {c : E}
    (hd : ∀ᶠ z in 𝓝 c, DifferentiableAt ℂ f z) (hc : IsLocalMax (norm ∘ f) c) :
    ∀ᶠ y in 𝓝 c, ‖f y‖ = ‖f c‖ := by
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    c : E
    hd : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMax (Function.comp Norm.norm f) c
    ⊢ Filter.Eventually (fun y => Eq (Norm.norm (f y)) (Norm.norm (f c))) (nhds c)
  -/
  rcases nhds_basis_closedBall.eventually_iff.1 (hd.and hc) with ⟨r, hr₀, hr⟩
  exact nhds_basis_closedBall.eventually_iff.2
    ⟨r, hr₀, norm_eqOn_closedBall_of_isMaxOn (DifferentiableOn.diffContOnCl fun x hx =>
        (hr <| closure_ball_subset_closedBall hx).1.differentiableWithinAt) fun x hx =>
      (hr <| ball_subset_closedBall hx).2⟩


theorem isOpen_setOf_mem_nhds_and_isMaxOn_norm {f : E → F} {s : Set E}
    (hd : DifferentiableOn ℂ f s) : IsOpen {z | s ∈ 𝓝 z ∧ IsMaxOn (norm ∘ f) s z} := by
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    s : Set E
    hd : DifferentiableOn Complex f s
    ⊢ IsOpen (setOf fun z => And (Membership.mem (nhds z) s) (IsMaxOn (Function.co …
  -/
  refine isOpen_iff_mem_nhds.2 fun z hz => (eventually_eventually_nhds.2 hz.1).and ?_
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    s : Set E
    hd : DifferentiableOn Complex f s
    z : E
    hz : Membership.mem (setOf fun z => And (Membership.mem (nhds z) s) (IsMaxOn ( …
    ⊢ Filter.Eventually (IsMaxOn (Function.comp Norm.norm f) s) (nhds z)
  -/
  replace hd : ∀ᶠ w in 𝓝 z, DifferentiableAt ℂ f w := hd.eventually_differentiableAt hz.1
  exact (norm_eventually_eq_of_isLocalMax hd <| hz.2.isLocalMax hz.1).mono fun x hx y hy =>
    le_trans (hz.2 hy).out hx.ge


/-- **Maximum modulus principle** on a connected set. Let `U` be a (pre)connected open set in a
complex normed space. Let `f : E → F` be a function that is complex differentiable on `U`. Suppose
that `‖f x‖` takes its maximum value on `U` at `c ∈ U`. Then `‖f x‖ = ‖f c‖` for all `x ∈ U`. -/
theorem norm_eqOn_of_isPreconnected_of_isMaxOn {f : E → F} {U : Set E} {c : E}
    (hc : IsPreconnected U) (ho : IsOpen U) (hd : DifferentiableOn ℂ f U) (hcU : c ∈ U)
    (hm : IsMaxOn (norm ∘ f) U c) : EqOn (norm ∘ f) (const E ‖f c‖) U := by
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    ⊢ Set.EqOn (Function.comp Norm.norm f) (Function.const E (Norm.norm (f c))) U
  -/
  set V := U ∩ {z | IsMaxOn (norm ∘ f) U z}
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    ⊢ Set.EqOn (Function.comp Norm.norm f) (Function.const E (Norm.norm (f c))) U
  -/
  have hV : ∀ x ∈ V, ‖f x‖ = ‖f c‖ := fun x hx => le_antisymm (hm hx.1) (hx.2 hcU)
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    hV : ∀ (x : E), Membership.mem V x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    ⊢ Set.EqOn (Function.comp Norm.norm f) (Function.const E (Norm.norm (f c))) U
  -/
  suffices U ⊆ V from fun x hx => hV x (this hx)
  have hVo : IsOpen V := by
    simpa only [ho.mem_nhds_iff, setOf_and, setOf_mem_eq]
      using isOpen_setOf_mem_nhds_and_isMaxOn_norm hd
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    hV : ∀ (x : E), Membership.mem V x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    hVo : IsOpen V
    ⊢ HasSubset.Subset U V
  -/
  have hVne : (U ∩ V).Nonempty := ⟨c, hcU, hcU, hm⟩
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    hV : ∀ (x : E), Membership.mem V x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    ⊢ HasSubset.Subset U V
  -/
  set W := U ∩ {z | ‖f z‖ ≠ ‖f c‖}
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    hV : ∀ (x : E), Membership.mem V x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set E := Inter.inter U (setOf fun z => Ne (Norm.norm (f z)) (Norm.norm (f  …
    ⊢ HasSubset.Subset U V
  -/
  have hWo : IsOpen W := hd.continuousOn.norm.isOpen_inter_preimage ho isOpen_ne
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    hV : ∀ (x : E), Membership.mem V x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set E := Inter.inter U (setOf fun z => Ne (Norm.norm (f z)) (Norm.norm (f  …
    hWo : IsOpen W
    ⊢ HasSubset.Subset U V
  -/
  have hdVW : Disjoint V W := disjoint_left.mpr fun x hxV hxW => hxW.2 (hV x hxV)
  have hUVW : U ⊆ V ∪ W := fun x hx =>
    (eq_or_ne ‖f x‖ ‖f c‖).imp (fun h => ⟨hx, fun y hy => (hm hy).out.trans_eq h.symm⟩)
      (And.intro hx)
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    U : Set E
    c : E
    hc : IsPreconnected U
    ho : IsOpen U
    hd : DifferentiableOn Complex f U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set E := Inter.inter U (setOf fun z => IsMaxOn (Function.comp Norm.norm f) …
    hV : ∀ (x : E), Membership.mem V x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set E := Inter.inter U (setOf fun z => Ne (Norm.norm (f z)) (Norm.norm (f  …
    hWo : IsOpen W
    hdVW : Disjoint V W
    hUVW : HasSubset.Subset U (Union.union V W)
    ⊢ HasSubset.Subset U V
  -/
  exact hc.subset_left_of_subset_union hVo hWo hdVW hUVW hVne
  /-
    🎉 no goals
  -/


/-- **Maximum modulus principle** on a connected set. Let `U` be a (pre)connected open set in a
complex normed space.  Let `f : E → F` be a function that is complex differentiable on `U` and is
continuous on its closure. Suppose that `‖f x‖` takes its maximum value on `U` at `c ∈ U`. Then
`‖f x‖ = ‖f c‖` for all `x ∈ closure U`. -/
theorem norm_eqOn_closure_of_isPreconnected_of_isMaxOn {f : E → F} {U : Set E} {c : E}
    (hc : IsPreconnected U) (ho : IsOpen U) (hd : DiffContOnCl ℂ f U) (hcU : c ∈ U)
    (hm : IsMaxOn (norm ∘ f) U c) : EqOn (norm ∘ f) (const E ‖f c‖) (closure U) :=
  (norm_eqOn_of_isPreconnected_of_isMaxOn hc ho hd.differentiableOn hcU hm).of_subset_closure
    hd.continuousOn.norm continuousOn_const subset_closure Subset.rfl


/-- **Maximum modulus principle** on a connected set. Let `U` be a (pre)connected open set in a
complex normed space.  Let `f : E → F` be a function that is complex differentiable on `U`. Suppose
that `‖f x‖` takes its maximum value on `U` at `c ∈ U`. Then `f x = f c` for all `x ∈ U`.

TODO: change assumption from `IsMaxOn` to `IsLocalMax`. -/
theorem eqOn_of_isPreconnected_of_isMaxOn_norm {f : E → F} {U : Set E} {c : E}
    (hc : IsPreconnected U) (ho : IsOpen U) (hd : DifferentiableOn ℂ f U) (hcU : c ∈ U)
    (hm : IsMaxOn (norm ∘ f) U c) : EqOn f (const E (f c)) U := fun x hx =>
  have H₁ : ‖f x‖ = ‖f c‖ := norm_eqOn_of_isPreconnected_of_isMaxOn hc ho hd hcU hm hx
  have H₂ : ‖f x + f c‖ = ‖f c + f c‖ :=
    norm_eqOn_of_isPreconnected_of_isMaxOn hc ho (hd.add_const _) hcU hm.norm_add_self hx
                                        /-
                                          E : Type u
                                          inst✝⁴ : NormedAddCommGroup E
                                          inst✝³ : NormedSpace Complex E
                                          F : Type v
                                          inst✝² : NormedAddCommGroup F
                                          inst✝¹ : NormedSpace Complex F
                                          inst✝ : StrictConvexSpace Real F
                                          f : E → F
                                          U : Set E
                                          c : E
                                          hc : IsPreconnected U
                                          ho : IsOpen U
                                          hd : DifferentiableOn Complex f U
                                          hcU : Membership.mem U c
                                          hm : IsMaxOn (Function.comp Norm.norm f) U c
                                          x : E
                                          hx : Membership.mem U x
                                          H₁ : Eq (Norm.norm (f x)) (Norm.norm (f c))
                                          H₂ : Eq (Norm.norm (HAdd.hAdd (f x) (f c))) (Norm.norm (HAdd.hAdd (f c) (f c)))
                                          ⊢ Eq (Norm.norm (HAdd.hAdd (f x) (Function.const E (f c) x))) (HAdd.hAdd (Norm …
                                        -/
  eq_of_norm_eq_of_norm_add_eq H₁ <| by simp only [H₂, SameRay.rfl.norm_add, H₁, Function.const]
                                        /-
                                          🎉 no goals
                                        -/


/-- **Maximum modulus principle** on a connected set. Let `U` be a (pre)connected open set in a
complex normed space.  Let `f : E → F` be a function that is complex differentiable on `U` and is
continuous on its closure. Suppose that `‖f x‖` takes its maximum value on `U` at `c ∈ U`. Then
`f x = f c` for all `x ∈ closure U`. -/
theorem eqOn_closure_of_isPreconnected_of_isMaxOn_norm {f : E → F} {U : Set E} {c : E}
    (hc : IsPreconnected U) (ho : IsOpen U) (hd : DiffContOnCl ℂ f U) (hcU : c ∈ U)
    (hm : IsMaxOn (norm ∘ f) U c) : EqOn f (const E (f c)) (closure U) :=
  (eqOn_of_isPreconnected_of_isMaxOn_norm hc ho hd.differentiableOn hcU hm).of_subset_closure
    hd.continuousOn continuousOn_const subset_closure Subset.rfl


/-- **Maximum modulus principle**. Let `f : E → F` be a function between complex normed spaces.
Suppose that the codomain `F` is a strictly convex space, `f` is complex differentiable on a set
`s`, `f` is continuous on the closure of `s`, the norm of `f` takes it maximum on `s` at `z`, and
`w` is a point such that the closed ball with center `z` and radius `dist w z` is included in `s`,
then `f w = f z`. -/
theorem eq_of_isMaxOn_of_ball_subset {f : E → F} {s : Set E} {z w : E} (hd : DiffContOnCl ℂ f s)
    (hz : IsMaxOn (norm ∘ f) s z) (hsub : ball z (dist w z) ⊆ s) : f w = f z :=
  have H₁ : ‖f w‖ = ‖f z‖ := norm_eq_norm_of_isMaxOn_of_ball_subset hd hz hsub
  have H₂ : ‖f w + f z‖ = ‖f z + f z‖ :=
    norm_eq_norm_of_isMaxOn_of_ball_subset (hd.add_const _) hz.norm_add_self hsub
                                        /-
                                          E : Type u
                                          inst✝⁴ : NormedAddCommGroup E
                                          inst✝³ : NormedSpace Complex E
                                          F : Type v
                                          inst✝² : NormedAddCommGroup F
                                          inst✝¹ : NormedSpace Complex F
                                          inst✝ : StrictConvexSpace Real F
                                          f : E → F
                                          s : Set E
                                          z w : E
                                          hd : DiffContOnCl Complex f s
                                          hz : IsMaxOn (Function.comp Norm.norm f) s z
                                          hsub : HasSubset.Subset (Metric.ball z (Dist.dist w z)) s
                                          H₁ : Eq (Norm.norm (f w)) (Norm.norm (f z))
                                          H₂ : Eq (Norm.norm (HAdd.hAdd (f w) (f z))) (Norm.norm (HAdd.hAdd (f z) (f z)))
                                          ⊢ Eq (Norm.norm (HAdd.hAdd (f w) (f z))) (HAdd.hAdd (Norm.norm (f w)) (Norm.no …
                                        -/
  eq_of_norm_eq_of_norm_add_eq H₁ <| by simp only [H₂, SameRay.rfl.norm_add, H₁]
                                        /-
                                          🎉 no goals
                                        -/


/-- **Maximum modulus principle** on a closed ball. Suppose that a function `f : E → F` from a
normed complex space to a strictly convex normed complex space has the following properties:

- it is continuous on a closed ball `Metric.closedBall z r`,
- it is complex differentiable on the corresponding open ball;
- the norm `‖f w‖` takes its maximum value on the open ball at its center.

Then `f` is a constant on the closed ball. -/
theorem eqOn_closedBall_of_isMaxOn_norm {f : E → F} {z : E} {r : ℝ}
    (hd : DiffContOnCl ℂ f (ball z r)) (hz : IsMaxOn (norm ∘ f) (ball z r) z) :
    EqOn f (const E (f z)) (closedBall z r) := fun _x hx =>
  eq_of_isMaxOn_of_ball_subset hd hz <| ball_subset_ball hx


/-- If `f` is differentiable on the open unit ball `{z : ℂ | ‖z‖ < 1}`, and `‖f‖` attains a maximum
in this open ball, then `f` is constant.-/
lemma eq_const_of_exists_max {f : E → F} {b : ℝ} (h_an : DifferentiableOn ℂ f (ball 0 b))
    {v : E} (hv : v ∈ ball 0 b) (hv_max : IsMaxOn (norm ∘ f) (ball 0 b) v) :
    Set.EqOn f (Function.const E (f v)) (ball 0 b) :=
  Complex.eqOn_of_isPreconnected_of_isMaxOn_norm (convex_ball 0 b).isPreconnected
    isOpen_ball h_an hv hv_max


/-- If `f` is a function differentiable on the open unit ball, and there exists an `r < 1` such that
any value of `‖f‖` on the open ball is bounded above by some value on the closed ball of radius `r`,
then `f` is constant. -/
lemma eq_const_of_exists_le [ProperSpace E] {f : E → F} {r b : ℝ}
    (h_an : DifferentiableOn ℂ f (ball 0 b)) (hr_nn : 0 ≤ r) (hr_lt : r < b)
    (hr : ∀ z, z ∈ (ball 0 b) → ∃ w, w ∈ closedBall 0 r ∧ ‖f z‖ ≤ ‖f w‖) :
    Set.EqOn f (Function.const E (f 0)) (ball 0 b) := by
  obtain ⟨x, hx_mem, hx_max⟩ := isCompact_closedBall (0 : E) r |>.exists_isMaxOn
    (nonempty_closedBall.mpr hr_nn)
    (h_an.continuousOn.mono <| closedBall_subset_ball hr_lt).norm
  suffices Set.EqOn f (Function.const E (f x)) (ball 0 b) by
    rwa [this (mem_ball_self (hr_nn.trans_lt hr_lt))]
  /-
    case intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : StrictConvexSpace Real F
    inst✝ : ProperSpace E
    f : E → F
    r b : Real
    h_an : DifferentiableOn Complex f (Metric.ball 0 b)
    hr_nn : LE.le 0 r
    hr_lt : LT.lt r b
    hr : ∀ (z : E), Membership.mem (Metric.ball 0 b) z → Exists fun w => And (Memb …
    x : E
    hx_mem : Membership.mem (Metric.closedBall 0 r) x
    hx_max : IsMaxOn (fun x => Norm.norm (f x)) (Metric.closedBall 0 r) x
    ⊢ Set.EqOn f (Function.const E (f x)) (Metric.ball 0 b)
  -/
  apply eq_const_of_exists_max h_an (closedBall_subset_ball hr_lt hx_mem) (fun z hz ↦ ?_)
  /-
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : StrictConvexSpace Real F
    inst✝ : ProperSpace E
    f : E → F
    r b : Real
    h_an : DifferentiableOn Complex f (Metric.ball 0 b)
    hr_nn : LE.le 0 r
    hr_lt : LT.lt r b
    hr : ∀ (z : E), Membership.mem (Metric.ball 0 b) z → Exists fun w => And (Memb …
    x : E
    hx_mem : Membership.mem (Metric.closedBall 0 r) x
    hx_max : IsMaxOn (fun x => Norm.norm (f x)) (Metric.closedBall 0 r) x
    z : E
    hz : Membership.mem (Metric.ball 0 b) z
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Function.comp Norm.norm  …
  -/
  obtain ⟨w, hw, hw'⟩ := hr z hz
  /-
    case intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : StrictConvexSpace Real F
    inst✝ : ProperSpace E
    f : E → F
    r b : Real
    h_an : DifferentiableOn Complex f (Metric.ball 0 b)
    hr_nn : LE.le 0 r
    hr_lt : LT.lt r b
    hr : ∀ (z : E), Membership.mem (Metric.ball 0 b) z → Exists fun w => And (Memb …
    x : E
    hx_mem : Membership.mem (Metric.closedBall 0 r) x
    hx_max : IsMaxOn (fun x => Norm.norm (f x)) (Metric.closedBall 0 r) x
    z : E
    hz : Membership.mem (Metric.ball 0 b) z
    w : E
    hw : Membership.mem (Metric.closedBall 0 r) w
    hw' : LE.le (Norm.norm (f z)) (Norm.norm (f w))
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Function.comp Norm.norm  …
  -/
  exact hw'.trans (hx_max hw)
  /-
    🎉 no goals
  -/


/-- **Maximum modulus principle**: if `f : E → F` is complex differentiable in a neighborhood of `c`
and the norm `‖f z‖` has a local maximum at `c`, then `f` is locally constant in a neighborhood
of `c`. -/
theorem eventually_eq_of_isLocalMax_norm {f : E → F} {c : E}
    (hd : ∀ᶠ z in 𝓝 c, DifferentiableAt ℂ f z) (hc : IsLocalMax (norm ∘ f) c) :
    ∀ᶠ y in 𝓝 c, f y = f c := by
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : StrictConvexSpace Real F
    f : E → F
    c : E
    hd : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMax (Function.comp Norm.norm f) c
    ⊢ Filter.Eventually (fun y => Eq (f y) (f c)) (nhds c)
  -/
  rcases nhds_basis_closedBall.eventually_iff.1 (hd.and hc) with ⟨r, hr₀, hr⟩
  exact nhds_basis_closedBall.eventually_iff.2
    ⟨r, hr₀, eqOn_closedBall_of_isMaxOn_norm (DifferentiableOn.diffContOnCl fun x hx =>
        (hr <| closure_ball_subset_closedBall hx).1.differentiableWithinAt) fun x hx =>
      (hr <| ball_subset_closedBall hx).2⟩


theorem eventually_eq_or_eq_zero_of_isLocalMin_norm {f : E → ℂ} {c : E}
    (hf : ∀ᶠ z in 𝓝 c, DifferentiableAt ℂ f z) (hc : IsLocalMin (norm ∘ f) c) :
    (∀ᶠ z in 𝓝 c, f z = f c) ∨ f c = 0 := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : E → Complex
    c : E
    hf : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMin (Function.comp Norm.norm f) c
    ⊢ Or (Filter.Eventually (fun z => Eq (f z) (f c)) (nhds c)) (Eq (f c) 0)
  -/
  refine or_iff_not_imp_right.mpr fun h => ?_
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : E → Complex
    c : E
    hf : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMin (Function.comp Norm.norm f) c
    h : Not (Eq (f c) 0)
    ⊢ Filter.Eventually (fun z => Eq (f z) (f c)) (nhds c)
  -/
  have h1 : ∀ᶠ z in 𝓝 c, f z ≠ 0 := hf.self_of_nhds.continuousAt.eventually_ne h
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : E → Complex
    c : E
    hf : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMin (Function.comp Norm.norm f) c
    h : Not (Eq (f c) 0)
    h1 : Filter.Eventually (fun z => Ne (f z) 0) (nhds c)
    ⊢ Filter.Eventually (fun z => Eq (f z) (f c)) (nhds c)
  -/
  have h2 : IsLocalMax (norm ∘ f)⁻¹ c := hc.inv (h1.mono fun z => norm_pos_iff.mpr)
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : E → Complex
    c : E
    hf : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMin (Function.comp Norm.norm f) c
    h : Not (Eq (f c) 0)
    h1 : Filter.Eventually (fun z => Ne (f z) 0) (nhds c)
    h2 : IsLocalMax (Inv.inv (Function.comp Norm.norm f)) c
    ⊢ Filter.Eventually (fun z => Eq (f z) (f c)) (nhds c)
  -/
  have h3 : IsLocalMax (norm ∘ f⁻¹) c := by refine h2.congr (Eventually.of_forall ?_); simp
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : E → Complex
    c : E
    hf : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMin (Function.comp Norm.norm f) c
    h : Not (Eq (f c) 0)
    h1 : Filter.Eventually (fun z => Ne (f z) 0) (nhds c)
    h2 : IsLocalMax (Inv.inv (Function.comp Norm.norm f)) c
    h3 : IsLocalMax (Function.comp Norm.norm (Inv.inv f)) c
    ⊢ Filter.Eventually (fun z => Eq (f z) (f c)) (nhds c)
  -/
  have h4 : ∀ᶠ z in 𝓝 c, DifferentiableAt ℂ f⁻¹ z := by filter_upwards [hf, h1] with z h using h.inv
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : E → Complex
    c : E
    hf : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    hc : IsLocalMin (Function.comp Norm.norm f) c
    h : Not (Eq (f c) 0)
    h1 : Filter.Eventually (fun z => Ne (f z) 0) (nhds c)
    h2 : IsLocalMax (Inv.inv (Function.comp Norm.norm f)) c
    h3 : IsLocalMax (Function.comp Norm.norm (Inv.inv f)) c
    h4 : Filter.Eventually (fun z => DifferentiableAt Complex (Inv.inv f) z) (nhds …
    ⊢ Filter.Eventually (fun z => Eq (f z) (f c)) (nhds c)
  -/
  filter_upwards [eventually_eq_of_isLocalMax_norm h4 h3] with z using inv_inj.mp
  /-
    🎉 no goals
  -/


/-- **Maximum modulus principle**: if `f : E → F` is complex differentiable on a nonempty bounded
set `U` and is continuous on its closure, then there exists a point `z ∈ frontier U` such that
`(‖f ·‖)` takes it maximum value on `closure U` at `z`. -/
theorem exists_mem_frontier_isMaxOn_norm [FiniteDimensional ℂ E] {f : E → F} {U : Set E}
    (hb : IsBounded U) (hne : U.Nonempty) (hd : DiffContOnCl ℂ f U) :
    ∃ z ∈ frontier U, IsMaxOn (norm ∘ f) (closure U) z := by
  /-
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    ⊢ Exists fun z => And (Membership.mem (frontier U) z) (IsMaxOn (Function.comp  …
  -/
  have hc : IsCompact (closure U) := hb.isCompact_closure
  obtain ⟨w, hwU, hle⟩ : ∃ w ∈ closure U, IsMaxOn (norm ∘ f) (closure U) w :=
    hc.exists_isMaxOn hne.closure hd.continuousOn.norm
  /-
    case intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hwU : Membership.mem (closure U) w
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    ⊢ Exists fun z => And (Membership.mem (frontier U) z) (IsMaxOn (Function.comp  …
  -/
  rw [closure_eq_interior_union_frontier, mem_union] at hwU
  /-
    case intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hwU : Or (Membership.mem (interior U) w) (Membership.mem (frontier U) w)
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    ⊢ Exists fun z => And (Membership.mem (frontier U) z) (IsMaxOn (Function.comp  …
  -/
  cases' hwU with hwU hwU; rotate_left; · exact ⟨w, hwU, hle⟩
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case intro.intro.inl
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    hwU : Membership.mem (interior U) w
    ⊢ Exists fun z => And (Membership.mem (frontier U) z) (IsMaxOn (Function.comp  …
  -/
  have : interior U ≠ univ := ne_top_of_le_ne_top hc.ne_univ interior_subset_closure
  /-
    case intro.intro.inl
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    hwU : Membership.mem (interior U) w
    this : Ne (interior U) Set.univ
    ⊢ Exists fun z => And (Membership.mem (frontier U) z) (IsMaxOn (Function.comp  …
  -/
  rcases exists_mem_frontier_infDist_compl_eq_dist hwU this with ⟨z, hzU, hzw⟩
  /-
    case intro.intro.inl.intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    hwU : Membership.mem (interior U) w
    this : Ne (interior U) Set.univ
    z : E
    hzU : Membership.mem (frontier (interior U)) z
    hzw : Eq (Metric.infDist w (HasCompl.compl (interior U))) (Dist.dist w z)
    ⊢ Exists fun z => And (Membership.mem (frontier U) z) (IsMaxOn (Function.comp  …
  -/
  refine ⟨z, frontier_interior_subset hzU, fun x hx => (hle hx).out.trans_eq ?_⟩
  /-
    case intro.intro.inl.intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    hwU : Membership.mem (interior U) w
    this : Ne (interior U) Set.univ
    z : E
    hzU : Membership.mem (frontier (interior U)) z
    hzw : Eq (Metric.infDist w (HasCompl.compl (interior U))) (Dist.dist w z)
    x : E
    hx : Membership.mem (closure U) x
    ⊢ Eq (Function.comp Norm.norm f w) (Function.comp Norm.norm f z)
  -/
  refine (norm_eq_norm_of_isMaxOn_of_ball_subset hd (hle.on_subset subset_closure) ?_).symm
  /-
    case intro.intro.inl.intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    hwU : Membership.mem (interior U) w
    this : Ne (interior U) Set.univ
    z : E
    hzU : Membership.mem (frontier (interior U)) z
    hzw : Eq (Metric.infDist w (HasCompl.compl (interior U))) (Dist.dist w z)
    x : E
    hx : Membership.mem (closure U) x
    ⊢ HasSubset.Subset (Metric.ball w (Dist.dist z w)) U
  -/
  rw [dist_comm, ← hzw]
  /-
    case intro.intro.inl.intro.intro
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Complex F
    inst✝¹ : Nontrivial E
    inst✝ : FiniteDimensional Complex E
    f : E → F
    U : Set E
    hb : Bornology.IsBounded U
    hne : U.Nonempty
    hd : DiffContOnCl Complex f U
    hc : IsCompact (closure U)
    w : E
    hle : IsMaxOn (Function.comp Norm.norm f) (closure U) w
    hwU : Membership.mem (interior U) w
    this : Ne (interior U) Set.univ
    z : E
    hzU : Membership.mem (frontier (interior U)) z
    hzw : Eq (Metric.infDist w (HasCompl.compl (interior U))) (Dist.dist w z)
    x : E
    hx : Membership.mem (closure U) x
    ⊢ HasSubset.Subset (Metric.ball w (Metric.infDist w (HasCompl.compl (interior  …
  -/
  exact ball_infDist_compl_subset.trans interior_subset
  /-
    🎉 no goals
  -/


/-- **Maximum modulus principle**: if `f : E → F` is complex differentiable on a bounded set `U` and
`‖f z‖ ≤ C` for any `z ∈ frontier U`, then the same is true for any `z ∈ closure U`. -/
theorem norm_le_of_forall_mem_frontier_norm_le {f : E → F} {U : Set E} (hU : IsBounded U)
    (hd : DiffContOnCl ℂ f U) {C : ℝ} (hC : ∀ z ∈ frontier U, ‖f z‖ ≤ C) {z : E}
    (hz : z ∈ closure U) : ‖f z‖ ≤ C := by
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hd : DiffContOnCl Complex f U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem (closure U) z
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rw [closure_eq_self_union_frontier, union_comm, mem_union] at hz
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hd : DiffContOnCl Complex f U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Or (Membership.mem (frontier U) z) (Membership.mem U z)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  cases' hz with hz hz; · exact hC z hz
                          /-
                            🎉 no goals
                          -/
  /- In case of a finite dimensional domain, one can just apply
    `Complex.exists_mem_frontier_isMaxOn_norm`. To make it work in any Banach space, we restrict
    the function to a line first. -/
  /-
    case inr
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hd : DiffContOnCl Complex f U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem U z
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rcases exists_ne z with ⟨w, hne⟩
  /-
    case inr.intro
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hd : DiffContOnCl Complex f U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem U z
    w : E
    hne : Ne w z
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  set e := (lineMap z w : ℂ → E)
  /-
    case inr.intro
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hd : DiffContOnCl Complex f U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem U z
    w : E
    hne : Ne w z
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have hde : Differentiable ℂ e := (differentiable_id.smul_const (w - z)).add_const z
  /-
    case inr.intro
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hd : DiffContOnCl Complex f U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem U z
    w : E
    hne : Ne w z
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    hde : Differentiable Complex e
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have hL : AntilipschitzWith (nndist z w)⁻¹ e := antilipschitzWith_lineMap hne.symm
  replace hd : DiffContOnCl ℂ (f ∘ e) (e ⁻¹' U) :=
    hd.comp hde.diffContOnCl (mapsTo_preimage _ _)
  /-
    case inr.intro
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem U z
    w : E
    hne : Ne w z
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    hde : Differentiable Complex e
    hL : AntilipschitzWith (Inv.inv (NNDist.nndist z w)) e
    hd : DiffContOnCl Complex (Function.comp f e) (Set.preimage e U)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have h₀ : (0 : ℂ) ∈ e ⁻¹' U := by simpa only [e, mem_preimage, lineMap_apply_zero]
  /-
    case inr.intro
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    U : Set E
    hU : Bornology.IsBounded U
    C : Real
    hC : ∀ (z : E), Membership.mem (frontier U) z → LE.le (Norm.norm (f z)) C
    z : E
    hz : Membership.mem U z
    w : E
    hne : Ne w z
    e : Complex → E := ⇑(AffineMap.lineMap z w)
    hde : Differentiable Complex e
    hL : AntilipschitzWith (Inv.inv (NNDist.nndist z w)) e
    hd : DiffContOnCl Complex (Function.comp f e) (Set.preimage e U)
    h₀ : Membership.mem (Set.preimage e U) 0
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rcases exists_mem_frontier_isMaxOn_norm (hL.isBounded_preimage hU) ⟨0, h₀⟩ hd with ⟨ζ, hζU, hζ⟩
  calc
    ‖f z‖ = ‖f (e 0)‖ := by simp only [e, lineMap_apply_zero]
    _ ≤ ‖f (e ζ)‖ := hζ (subset_closure h₀)
    _ ≤ C := hC _ (hde.continuous.frontier_preimage_subset _ hζU)


/-- If two complex differentiable functions `f g : E → F` are equal on the boundary of a bounded set
`U`, then they are equal on `closure U`. -/
theorem eqOn_closure_of_eqOn_frontier {f g : E → F} {U : Set E} (hU : IsBounded U)
    (hf : DiffContOnCl ℂ f U) (hg : DiffContOnCl ℂ g U) (hfg : EqOn f g (frontier U)) :
    EqOn f g (closure U) := by
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f g : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hf : DiffContOnCl Complex f U
    hg : DiffContOnCl Complex g U
    hfg : Set.EqOn f g (frontier U)
    ⊢ Set.EqOn f g (closure U)
  -/
  suffices H : ∀ z ∈ closure U, ‖(f - g) z‖ ≤ 0 by simpa [sub_eq_zero] using H
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f g : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hf : DiffContOnCl Complex f U
    hg : DiffContOnCl Complex g U
    hfg : Set.EqOn f g (frontier U)
    ⊢ ∀ (z : E), Membership.mem (closure U) z → LE.le (Norm.norm (HSub.hSub f g z) …
  -/
  refine fun z hz => norm_le_of_forall_mem_frontier_norm_le hU (hf.sub hg) (fun w hw => ?_) hz
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f g : E → F
    U : Set E
    hU : Bornology.IsBounded U
    hf : DiffContOnCl Complex f U
    hg : DiffContOnCl Complex g U
    hfg : Set.EqOn f g (frontier U)
    z : E
    hz : Membership.mem (closure U) z
    w : E
    hw : Membership.mem (frontier U) w
    ⊢ LE.le (Norm.norm (HSub.hSub f g w)) 0
  -/
  simp [hfg hw]
  /-
    🎉 no goals
  -/


/-- If two complex differentiable functions `f g : E → F` are equal on the boundary of a bounded set
`U`, then they are equal on `U`. -/
theorem eqOn_of_eqOn_frontier {f g : E → F} {U : Set E} (hU : IsBounded U) (hf : DiffContOnCl ℂ f U)
    (hg : DiffContOnCl ℂ g U) (hfg : EqOn f g (frontier U)) : EqOn f g U :=
  (eqOn_closure_of_eqOn_frontier hU hf hg hfg).mono subset_closure


