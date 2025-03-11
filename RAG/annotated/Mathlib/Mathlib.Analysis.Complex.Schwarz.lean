/-- An auxiliary lemma for `Complex.norm_dslope_le_div_of_mapsTo_ball`. -/
theorem schwarz_aux {f : ℂ → ℂ} (hd : DifferentiableOn ℂ f (ball c R₁))
    (h_maps : MapsTo f (ball c R₁) (ball (f c) R₂)) (hz : z ∈ ball c R₁) :
    ‖dslope f c z‖ ≤ R₂ / R₁ := by
  /-
    R₁ R₂ : Real
    c z : Complex
    f : Complex → Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  have hR₁ : 0 < R₁ := nonempty_ball.1 ⟨z, hz⟩
  suffices ∀ᶠ r in 𝓝[<] R₁, ‖dslope f c z‖ ≤ R₂ / r by
    refine ge_of_tendsto ?_ this
    exact (tendsto_const_nhds.div tendsto_id hR₁.ne').mono_left nhdsWithin_le_nhds
  /-
    R₁ R₂ : Real
    c z : Complex
    f : Complex → Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    hR₁ : LT.lt 0 R₁
    ⊢ Filter.Eventually (fun r => LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ r …
  -/
  rw [mem_ball] at hz
  /-
    R₁ R₂ : Real
    c z : Complex
    f : Complex → Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : LT.lt (Dist.dist z c) R₁
    hR₁ : LT.lt 0 R₁
    ⊢ Filter.Eventually (fun r => LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ r …
  -/
  filter_upwards [Ioo_mem_nhdsLT hz] with r hr
  /-
    case h
    R₁ R₂ : Real
    c z : Complex
    f : Complex → Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : LT.lt (Dist.dist z c) R₁
    hR₁ : LT.lt 0 R₁
    r : Real
    hr : Membership.mem (Set.Ioo (Dist.dist z c) R₁) r
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ r)
  -/
  have hr₀ : 0 < r := dist_nonneg.trans_lt hr.1
  replace hd : DiffContOnCl ℂ (dslope f c) (ball c r) := by
    refine DifferentiableOn.diffContOnCl ?_
    rw [closure_ball c hr₀.ne']
    exact ((differentiableOn_dslope <| ball_mem_nhds _ hR₁).mpr hd).mono
      (closedBall_subset_ball hr.2)
  /-
    case h
    R₁ R₂ : Real
    c z : Complex
    f : Complex → Complex
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : LT.lt (Dist.dist z c) R₁
    hR₁ : LT.lt 0 R₁
    r : Real
    hr : Membership.mem (Set.Ioo (Dist.dist z c) R₁) r
    hr₀ : LT.lt 0 r
    hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ r)
  -/
  refine norm_le_of_forall_mem_frontier_norm_le isBounded_ball hd ?_ ?_
    /-
      case h.refine_1
      R₁ R₂ : Real
      c z : Complex
      f : Complex → Complex
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz : LT.lt (Dist.dist z c) R₁
      hR₁ : LT.lt 0 R₁
      r : Real
      hr : Membership.mem (Set.Ioo (Dist.dist z c) R₁) r
      hr₀ : LT.lt 0 r
      hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
      ⊢ ∀ (z : Complex), Membership.mem (frontier (Metric.ball c r)) z → LE.le (Norm …
    -/
  · rw [frontier_ball c hr₀.ne']
    /-
      case h.refine_1
      R₁ R₂ : Real
      c z : Complex
      f : Complex → Complex
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz : LT.lt (Dist.dist z c) R₁
      hR₁ : LT.lt 0 R₁
      r : Real
      hr : Membership.mem (Set.Ioo (Dist.dist z c) R₁) r
      hr₀ : LT.lt 0 r
      hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
      ⊢ ∀ (z : Complex), Membership.mem (Metric.sphere c r) z → LE.le (Norm.norm (ds …
    -/
    intro z hz
    /-
      case h.refine_1
      R₁ R₂ : Real
      c z✝ : Complex
      f : Complex → Complex
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz✝ : LT.lt (Dist.dist z✝ c) R₁
      hR₁ : LT.lt 0 R₁
      r : Real
      hr : Membership.mem (Set.Ioo (Dist.dist z✝ c) R₁) r
      hr₀ : LT.lt 0 r
      hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
      z : Complex
      hz : Membership.mem (Metric.sphere c r) z
      ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ r)
    -/
    have hz' : z ≠ c := ne_of_mem_sphere hz hr₀.ne'
    rw [dslope_of_ne _ hz', slope_def_module, norm_smul, norm_inv, mem_sphere_iff_norm.1 hz, ←
      div_eq_inv_mul, div_le_div_iff_of_pos_right hr₀, ← dist_eq_norm]
    /-
      case h.refine_1
      R₁ R₂ : Real
      c z✝ : Complex
      f : Complex → Complex
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz✝ : LT.lt (Dist.dist z✝ c) R₁
      hR₁ : LT.lt 0 R₁
      r : Real
      hr : Membership.mem (Set.Ioo (Dist.dist z✝ c) R₁) r
      hr₀ : LT.lt 0 r
      hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
      z : Complex
      hz : Membership.mem (Metric.sphere c r) z
      hz' : Ne z c
      ⊢ LE.le (Dist.dist (f z) (f c)) R₂
    -/
    exact le_of_lt (h_maps (mem_ball.2 (by rw [mem_sphere.1 hz]; exact hr.2)))
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R₁ R₂ : Real
      c z : Complex
      f : Complex → Complex
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz : LT.lt (Dist.dist z c) R₁
      hR₁ : LT.lt 0 R₁
      r : Real
      hr : Membership.mem (Set.Ioo (Dist.dist z c) R₁) r
      hr₀ : LT.lt 0 r
      hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
      ⊢ Membership.mem (closure (Metric.ball c r)) z
    -/
  · rw [closure_ball c hr₀.ne', mem_closedBall]
    /-
      case h.refine_2
      R₁ R₂ : Real
      c z : Complex
      f : Complex → Complex
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz : LT.lt (Dist.dist z c) R₁
      hR₁ : LT.lt 0 R₁
      r : Real
      hr : Membership.mem (Set.Ioo (Dist.dist z c) R₁) r
      hr₀ : LT.lt 0 r
      hd : DiffContOnCl Complex (dslope f c) (Metric.ball c r)
      ⊢ LE.le (Dist.dist z c) r
    -/
    exact hr.1.le
    /-
      🎉 no goals
    -/


/-- Two cases of the **Schwarz Lemma** (derivative and distance), merged together. -/
theorem norm_dslope_le_div_of_mapsTo_ball (hd : DifferentiableOn ℂ f (ball c R₁))
    (h_maps : MapsTo f (ball c R₁) (ball (f c) R₂)) (hz : z ∈ ball c R₁) :
    ‖dslope f c z‖ ≤ R₂ / R₁ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  have hR₁ : 0 < R₁ := nonempty_ball.1 ⟨z, hz⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    hR₁ : LT.lt 0 R₁
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  have hR₂ : 0 < R₂ := nonempty_ball.1 ⟨f z, h_maps hz⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    hR₁ : LT.lt 0 R₁
    hR₂ : LT.lt 0 R₂
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  rcases eq_or_ne (dslope f c z) 0 with hc | hc
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R₁ R₂ : Real
      f : Complex → E
      c z : Complex
      hd : DifferentiableOn Complex f (Metric.ball c R₁)
      h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
      hz : Membership.mem (Metric.ball c R₁) z
      hR₁ : LT.lt 0 R₁
      hR₂ : LT.lt 0 R₂
      hc : Eq (dslope f c z) 0
      ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
    -/
  · rw [hc, norm_zero]; exact div_nonneg hR₂.le hR₁.le
                        /-
                          🎉 no goals
                        -/
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    hR₁ : LT.lt 0 R₁
    hR₂ : LT.lt 0 R₂
    hc : Ne (dslope f c z) 0
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  rcases exists_dual_vector ℂ _ hc with ⟨g, hg, hgf⟩
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    hR₁ : LT.lt 0 R₁
    hR₂ : LT.lt 0 R₂
    hc : Ne (dslope f c z) 0
    g : ContinuousLinearMap (RingHom.id Complex) E Complex
    hg : Eq (Norm.norm g) 1
    hgf : Eq (g (dslope f c z)) ↑(Norm.norm (dslope f c z))
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  have hg' : ‖g‖₊ = 1 := NNReal.eq hg
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    hR₁ : LT.lt 0 R₁
    hR₂ : LT.lt 0 R₂
    hc : Ne (dslope f c z) 0
    g : ContinuousLinearMap (RingHom.id Complex) E Complex
    hg : Eq (Norm.norm g) 1
    hgf : Eq (g (dslope f c z)) ↑(Norm.norm (dslope f c z))
    hg' : Eq (NNNorm.nnnorm g) 1
    ⊢ LE.le (Norm.norm (dslope f c z)) (HDiv.hDiv R₂ R₁)
  -/
  have hg₀ : ‖g‖₊ ≠ 0 := by simpa only [hg'] using one_ne_zero
  calc
    ‖dslope f c z‖ = ‖dslope (g ∘ f) c z‖ := by
      rw [g.dslope_comp, hgf, RCLike.norm_ofReal, abs_norm]
      exact fun _ => hd.differentiableAt (ball_mem_nhds _ hR₁)
    _ ≤ R₂ / R₁ := by
      refine schwarz_aux (g.differentiable.comp_differentiableOn hd) (MapsTo.comp ?_ h_maps) hz
      simpa only [hg', NNReal.coe_one, one_mul] using g.lipschitz.mapsTo_ball hg₀ (f c) R₂


/-- Equality case in the **Schwarz Lemma**: in the setup of `norm_dslope_le_div_of_mapsTo_ball`, if
`‖dslope f c z₀‖ = R₂ / R₁` holds at a point in the ball then the map `f` is affine. -/
theorem affine_of_mapsTo_ball_of_exists_norm_dslope_eq_div [CompleteSpace E] [StrictConvexSpace ℝ E]
    (hd : DifferentiableOn ℂ f (ball c R₁)) (h_maps : Set.MapsTo f (ball c R₁) (ball (f c) R₂))
    (h_z₀ : z₀ ∈ ball c R₁) (h_eq : ‖dslope f c z₀‖ = R₂ / R₁) :
    Set.EqOn f (fun z => f c + (z - c) • dslope f c z₀) (ball c R₁) := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z₀ : Complex
    inst✝¹ : CompleteSpace E
    inst✝ : StrictConvexSpace Real E
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h_z₀ : Membership.mem (Metric.ball c R₁) z₀
    h_eq : Eq (Norm.norm (dslope f c z₀)) (HDiv.hDiv R₂ R₁)
    ⊢ Set.EqOn f (fun z => HAdd.hAdd (f c) (HSMul.hSMul (HSub.hSub z c) (dslope f  …
  -/
  set g := dslope f c
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z₀ : Complex
    inst✝¹ : CompleteSpace E
    inst✝ : StrictConvexSpace Real E
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h_z₀ : Membership.mem (Metric.ball c R₁) z₀
    g : Complex → E := dslope f c
    h_eq : Eq (Norm.norm (g z₀)) (HDiv.hDiv R₂ R₁)
    ⊢ Set.EqOn f (fun z => HAdd.hAdd (f c) (HSMul.hSMul (HSub.hSub z c) (g z₀))) ( …
  -/
  rintro z hz
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z₀ : Complex
    inst✝¹ : CompleteSpace E
    inst✝ : StrictConvexSpace Real E
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h_z₀ : Membership.mem (Metric.ball c R₁) z₀
    g : Complex → E := dslope f c
    h_eq : Eq (Norm.norm (g z₀)) (HDiv.hDiv R₂ R₁)
    z : Complex
    hz : Membership.mem (Metric.ball c R₁) z
    ⊢ Eq (f z) ((fun z => HAdd.hAdd (f c) (HSMul.hSMul (HSub.hSub z c) (g z₀))) z)
  -/
  by_cases h : z = c; · simp [h]
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z₀ : Complex
    inst✝¹ : CompleteSpace E
    inst✝ : StrictConvexSpace Real E
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h_z₀ : Membership.mem (Metric.ball c R₁) z₀
    g : Complex → E := dslope f c
    h_eq : Eq (Norm.norm (g z₀)) (HDiv.hDiv R₂ R₁)
    z : Complex
    hz : Membership.mem (Metric.ball c R₁) z
    h : Not (Eq z c)
    ⊢ Eq (f z) ((fun z => HAdd.hAdd (f c) (HSMul.hSMul (HSub.hSub z c) (g z₀))) z)
  -/
  have h_R₁ : 0 < R₁ := nonempty_ball.mp ⟨_, h_z₀⟩
  have g_le_div : ∀ z ∈ ball c R₁, ‖g z‖ ≤ R₂ / R₁ := fun z hz =>
    norm_dslope_le_div_of_mapsTo_ball hd h_maps hz
  have g_max : IsMaxOn (norm ∘ g) (ball c R₁) z₀ :=
    isMaxOn_iff.mpr fun z hz => by simpa [h_eq] using g_le_div z hz
  have g_diff : DifferentiableOn ℂ g (ball c R₁) :=
    (differentiableOn_dslope (isOpen_ball.mem_nhds (mem_ball_self h_R₁))).mpr hd
  have : g z = g z₀ := eqOn_of_isPreconnected_of_isMaxOn_norm (convex_ball c R₁).isPreconnected
    isOpen_ball g_diff h_z₀ g_max hz
  /-
    case neg
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z₀ : Complex
    inst✝¹ : CompleteSpace E
    inst✝ : StrictConvexSpace Real E
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h_z₀ : Membership.mem (Metric.ball c R₁) z₀
    g : Complex → E := dslope f c
    h_eq : Eq (Norm.norm (g z₀)) (HDiv.hDiv R₂ R₁)
    z : Complex
    hz : Membership.mem (Metric.ball c R₁) z
    h : Not (Eq z c)
    h_R₁ : LT.lt 0 R₁
    g_le_div : ∀ (z : Complex), Membership.mem (Metric.ball c R₁) z → LE.le (Norm. …
    g_max : IsMaxOn (Function.comp Norm.norm g) (Metric.ball c R₁) z₀
    g_diff : DifferentiableOn Complex g (Metric.ball c R₁)
    this : Eq (g z) (g z₀)
    ⊢ Eq (f z) ((fun z => HAdd.hAdd (f c) (HSMul.hSMul (HSub.hSub z c) (g z₀))) z)
  -/
  simp only [g] at this
  /-
    case neg
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z₀ : Complex
    inst✝¹ : CompleteSpace E
    inst✝ : StrictConvexSpace Real E
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h_z₀ : Membership.mem (Metric.ball c R₁) z₀
    g : Complex → E := dslope f c
    h_eq : Eq (Norm.norm (g z₀)) (HDiv.hDiv R₂ R₁)
    z : Complex
    hz : Membership.mem (Metric.ball c R₁) z
    h : Not (Eq z c)
    h_R₁ : LT.lt 0 R₁
    g_le_div : ∀ (z : Complex), Membership.mem (Metric.ball c R₁) z → LE.le (Norm. …
    g_max : IsMaxOn (Function.comp Norm.norm g) (Metric.ball c R₁) z₀
    g_diff : DifferentiableOn Complex g (Metric.ball c R₁)
    this : Eq (dslope f c z) (dslope f c z₀)
    ⊢ Eq (f z) ((fun z => HAdd.hAdd (f c) (HSMul.hSMul (HSub.hSub z c) (g z₀))) z)
  -/
  simp [g, ← this]
  /-
    🎉 no goals
  -/


theorem affine_of_mapsTo_ball_of_exists_norm_dslope_eq_div' [CompleteSpace E]
    [StrictConvexSpace ℝ E] (hd : DifferentiableOn ℂ f (ball c R₁))
    (h_maps : Set.MapsTo f (ball c R₁) (ball (f c) R₂))
    (h_z₀ : ∃ z₀ ∈ ball c R₁, ‖dslope f c z₀‖ = R₂ / R₁) :
    ∃ C : E, ‖C‖ = R₂ / R₁ ∧ Set.EqOn f (fun z => f c + (z - c) • C) (ball c R₁) :=
  let ⟨z₀, h_z₀, h_eq⟩ := h_z₀
  ⟨dslope f c z₀, h_eq, affine_of_mapsTo_ball_of_exists_norm_dslope_eq_div hd h_maps h_z₀ h_eq⟩


/-- The **Schwarz Lemma**: if `f : ℂ → E` sends an open disk with center `c` and a positive radius
`R₁` to an open ball with center `f c` and radius `R₂`, then the absolute value of the derivative of
`f` at `c` is at most the ratio `R₂ / R₁`. -/
theorem norm_deriv_le_div_of_mapsTo_ball (hd : DifferentiableOn ℂ f (ball c R₁))
    (h_maps : MapsTo f (ball c R₁) (ball (f c) R₂)) (h₀ : 0 < R₁) : ‖deriv f c‖ ≤ R₂ / R₁ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    h₀ : LT.lt 0 R₁
    ⊢ LE.le (Norm.norm (deriv f c)) (HDiv.hDiv R₂ R₁)
  -/
  simpa only [dslope_same] using norm_dslope_le_div_of_mapsTo_ball hd h_maps (mem_ball_self h₀)
  /-
    🎉 no goals
  -/


/-- The **Schwarz Lemma**: if `f : ℂ → E` sends an open disk with center `c` and radius `R₁` to an
open ball with center `f c` and radius `R₂`, then for any `z` in the former disk we have
`dist (f z) (f c) ≤ (R₂ / R₁) * dist z c`. -/
theorem dist_le_div_mul_dist_of_mapsTo_ball (hd : DifferentiableOn ℂ f (ball c R₁))
    (h_maps : MapsTo f (ball c R₁) (ball (f c) R₂)) (hz : z ∈ ball c R₁) :
    dist (f z) (f c) ≤ R₂ / R₁ * dist z c := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    R₁ R₂ : Real
    f : Complex → E
    c z : Complex
    hd : DifferentiableOn Complex f (Metric.ball c R₁)
    h_maps : Set.MapsTo f (Metric.ball c R₁) (Metric.ball (f c) R₂)
    hz : Membership.mem (Metric.ball c R₁) z
    ⊢ LE.le (Dist.dist (f z) (f c)) (HMul.hMul (HDiv.hDiv R₂ R₁) (Dist.dist z c))
  -/
  rcases eq_or_ne z c with (rfl | hne)
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      R₁ R₂ : Real
      f : Complex → E
      z : Complex
      hd : DifferentiableOn Complex f (Metric.ball z R₁)
      h_maps : Set.MapsTo f (Metric.ball z R₁) (Metric.ball (f z) R₂)
      hz : Membership.mem (Metric.ball z R₁) z
      ⊢ LE.le (Dist.dist (f z) (f z)) (HMul.hMul (HDiv.hDiv R₂ R₁) (Dist.dist z z))
    -/
  · simp only [dist_self, mul_zero, le_rfl]
    /-
      🎉 no goals
    -/
  simpa only [dslope_of_ne _ hne, slope_def_module, norm_smul, norm_inv, ← div_eq_inv_mul, ←
    dist_eq_norm, div_le_iff₀ (dist_pos.2 hne)] using norm_dslope_le_div_of_mapsTo_ball hd h_maps hz


/-- The **Schwarz Lemma**: if `f : ℂ → ℂ` sends an open disk with center `c` and a positive radius
`R₁` to an open disk with center `f c` and radius `R₂`, then the absolute value of the derivative of
`f` at `c` is at most the ratio `R₂ / R₁`. -/
theorem abs_deriv_le_div_of_mapsTo_ball (hd : DifferentiableOn ℂ f (ball c R₁))
    (h_maps : MapsTo f (ball c R₁) (ball (f c) R₂)) (h₀ : 0 < R₁) : abs (deriv f c) ≤ R₂ / R₁ :=
  norm_deriv_le_div_of_mapsTo_ball hd h_maps h₀


/-- The **Schwarz Lemma**: if `f : ℂ → ℂ` sends an open disk of positive radius to itself and the
center of this disk to itself, then the absolute value of the derivative of `f` at the center of
this disk is at most `1`. -/
theorem abs_deriv_le_one_of_mapsTo_ball (hd : DifferentiableOn ℂ f (ball c R))
    (h_maps : MapsTo f (ball c R) (ball c R)) (hc : f c = c) (h₀ : 0 < R) : abs (deriv f c) ≤ 1 :=
                                           /-
                                             f : Complex → Complex
                                             c : Complex
                                             R : Real
                                             hd : DifferentiableOn Complex f (Metric.ball c R)
                                             h_maps : Set.MapsTo f (Metric.ball c R) (Metric.ball c R)
                                             hc : Eq (f c) c
                                             h₀ : LT.lt 0 R
                                             ⊢ Set.MapsTo f (Metric.ball c R) (Metric.ball (f c) R)
                                           -/
  (norm_deriv_le_div_of_mapsTo_ball hd (by rwa [hc]) h₀).trans_eq (div_self h₀.ne')
                                           /-
                                             🎉 no goals
                                           -/


/-- The **Schwarz Lemma**: if `f : ℂ → ℂ` sends an open disk to itself and the center `c` of this
disk to itself, then for any point `z` of this disk we have `dist (f z) c ≤ dist z c`. -/
theorem dist_le_dist_of_mapsTo_ball_self (hd : DifferentiableOn ℂ f (ball c R))
    (h_maps : MapsTo f (ball c R) (ball c R)) (hc : f c = c) (hz : z ∈ ball c R) :
    dist (f z) c ≤ dist z c := by
  -- Porting note: `simp` was failing to use `div_self`
  /-
    f : Complex → Complex
    c z : Complex
    R : Real
    hd : DifferentiableOn Complex f (Metric.ball c R)
    h_maps : Set.MapsTo f (Metric.ball c R) (Metric.ball c R)
    hc : Eq (f c) c
    hz : Membership.mem (Metric.ball c R) z
    ⊢ LE.le (Dist.dist (f z) c) (Dist.dist z c)
  -/
  have := dist_le_div_mul_dist_of_mapsTo_ball hd (by rwa [hc]) hz
  /-
    f : Complex → Complex
    c z : Complex
    R : Real
    hd : DifferentiableOn Complex f (Metric.ball c R)
    h_maps : Set.MapsTo f (Metric.ball c R) (Metric.ball c R)
    hc : Eq (f c) c
    hz : Membership.mem (Metric.ball c R) z
    this : LE.le (Dist.dist (f z) (f c)) (HMul.hMul (HDiv.hDiv R R) (Dist.dist z c))
    ⊢ LE.le (Dist.dist (f z) c) (Dist.dist z c)
  -/
  rwa [hc, div_self, one_mul] at this
  /-
    f : Complex → Complex
    c z : Complex
    R : Real
    hd : DifferentiableOn Complex f (Metric.ball c R)
    h_maps : Set.MapsTo f (Metric.ball c R) (Metric.ball c R)
    hc : Eq (f c) c
    hz : Membership.mem (Metric.ball c R) z
    this : LE.le (Dist.dist (f z) c) (HMul.hMul (HDiv.hDiv R R) (Dist.dist z c))
    ⊢ Ne R 0
  -/
  exact (nonempty_ball.1 ⟨z, hz⟩).ne'
  /-
    🎉 no goals
  -/


/-- The **Schwarz Lemma**: if `f : ℂ → ℂ` sends an open disk with center `0` to itself, then for any
point `z` of this disk we have `abs (f z) ≤ abs z`. -/
theorem abs_le_abs_of_mapsTo_ball_self (hd : DifferentiableOn ℂ f (ball 0 R))
    (h_maps : MapsTo f (ball 0 R) (ball 0 R)) (h₀ : f 0 = 0) (hz : abs z < R) :
    abs (f z) ≤ abs z := by
  /-
    f : Complex → Complex
    z : Complex
    R : Real
    hd : DifferentiableOn Complex f (Metric.ball 0 R)
    h_maps : Set.MapsTo f (Metric.ball 0 R) (Metric.ball 0 R)
    h₀ : Eq (f 0) 0
    hz : LT.lt (Complex.abs z) R
    ⊢ LE.le (Complex.abs (f z)) (Complex.abs z)
  -/
  replace hz : z ∈ ball (0 : ℂ) R := mem_ball_zero_iff.2 hz
  /-
    f : Complex → Complex
    z : Complex
    R : Real
    hd : DifferentiableOn Complex f (Metric.ball 0 R)
    h_maps : Set.MapsTo f (Metric.ball 0 R) (Metric.ball 0 R)
    h₀ : Eq (f 0) 0
    hz : Membership.mem (Metric.ball 0 R) z
    ⊢ LE.le (Complex.abs (f z)) (Complex.abs z)
  -/
  simpa only [dist_zero_right] using dist_le_dist_of_mapsTo_ball_self hd h_maps h₀ hz
  /-
    🎉 no goals
  -/


