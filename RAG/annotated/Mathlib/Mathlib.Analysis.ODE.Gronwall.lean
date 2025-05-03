/-- Upper bound used in several Grönwall-like inequalities. -/
noncomputable def gronwallBound (δ K ε x : ℝ) : ℝ :=
  if K = 0 then δ + ε * x else δ * exp (K * x) + ε / K * (exp (K * x) - 1)


theorem gronwallBound_K0 (δ ε : ℝ) : gronwallBound δ 0 ε = fun x => δ + ε * x :=
  funext fun _ => if_pos rfl


theorem gronwallBound_of_K_ne_0 {δ K ε : ℝ} (hK : K ≠ 0) :
    gronwallBound δ K ε = fun x => δ * exp (K * x) + ε / K * (exp (K * x) - 1) :=
  funext fun _ => if_neg hK


theorem hasDerivAt_gronwallBound (δ K ε x : ℝ) :
    HasDerivAt (gronwallBound δ K ε) (K * gronwallBound δ K ε x + ε) x := by
  /-
    δ K ε x : Real
    ⊢ HasDerivAt (gronwallBound δ K ε) (HAdd.hAdd (HMul.hMul K (gronwallBound δ K  …
  -/
  by_cases hK : K = 0
    /-
      case pos
      δ K ε x : Real
      hK : Eq K 0
      ⊢ HasDerivAt (gronwallBound δ K ε) (HAdd.hAdd (HMul.hMul K (gronwallBound δ K  …
    -/
  · subst K
    /-
      case pos
      δ ε x : Real
      ⊢ HasDerivAt (gronwallBound δ 0 ε) (HAdd.hAdd (HMul.hMul 0 (gronwallBound δ 0  …
    -/
    simp only [gronwallBound_K0, zero_mul, zero_add]
    /-
      case pos
      δ ε x : Real
      ⊢ HasDerivAt (fun x => HAdd.hAdd δ (HMul.hMul ε x)) ε x
    -/
    convert ((hasDerivAt_id x).const_mul ε).const_add δ
    /-
      case h.e'_9
      δ ε x : Real
      ⊢ Eq ε (HMul.hMul ε 1)
    -/
    rw [mul_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      δ K ε x : Real
      hK : Not (Eq K 0)
      ⊢ HasDerivAt (gronwallBound δ K ε) (HAdd.hAdd (HMul.hMul K (gronwallBound δ K  …
    -/
  · simp only [gronwallBound_of_K_ne_0 hK]
    convert (((hasDerivAt_id x).const_mul K).exp.const_mul δ).add
      ((((hasDerivAt_id x).const_mul K).exp.sub_const 1).const_mul (ε / K)) using 1
    /-
      case h.e'_9
      δ K ε x : Real
      hK : Not (Eq K 0)
      ⊢ Eq (HAdd.hAdd (HMul.hMul K (HAdd.hAdd (HMul.hMul δ (Real.exp (HMul.hMul K x) …
    -/
    simp only [id, mul_add, (mul_assoc _ _ _).symm, mul_comm _ K, mul_div_cancel₀ _ hK]
    /-
      case h.e'_9
      δ K ε x : Real
      hK : Not (Eq K 0)
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul K δ) (Real.exp (HMul.hMul K x …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem hasDerivAt_gronwallBound_shift (δ K ε x a : ℝ) :
    HasDerivAt (fun y => gronwallBound δ K ε (y - a)) (K * gronwallBound δ K ε (x - a) + ε) x := by
  /-
    δ K ε x a : Real
    ⊢ HasDerivAt (fun y => gronwallBound δ K ε (HSub.hSub y a)) (HAdd.hAdd (HMul.h …
  -/
  convert (hasDerivAt_gronwallBound δ K ε _).comp x ((hasDerivAt_id x).sub_const a) using 1
  /-
    case h.e'_9
    δ K ε x a : Real
    ⊢ Eq (HAdd.hAdd (HMul.hMul K (gronwallBound δ K ε (HSub.hSub x a))) ε) (HMul.h …
  -/
  rw [id, mul_one]
  /-
    🎉 no goals
  -/


theorem gronwallBound_x0 (δ K ε : ℝ) : gronwallBound δ K ε 0 = δ := by
  /-
    δ K ε : Real
    ⊢ Eq (gronwallBound δ K ε 0) δ
  -/
  by_cases hK : K = 0
    /-
      case pos
      δ K ε : Real
      hK : Eq K 0
      ⊢ Eq (gronwallBound δ K ε 0) δ
    -/
  · simp only [gronwallBound, if_pos hK, mul_zero, add_zero]
    /-
      🎉 no goals
    -/
  · simp only [gronwallBound, if_neg hK, mul_zero, exp_zero, sub_self, mul_one,
      add_zero]


theorem gronwallBound_ε0 (δ K x : ℝ) : gronwallBound δ K 0 x = δ * exp (K * x) := by
  /-
    δ K x : Real
    ⊢ Eq (gronwallBound δ K 0 x) (HMul.hMul δ (Real.exp (HMul.hMul K x)))
  -/
  by_cases hK : K = 0
    /-
      case pos
      δ K x : Real
      hK : Eq K 0
      ⊢ Eq (gronwallBound δ K 0 x) (HMul.hMul δ (Real.exp (HMul.hMul K x)))
    -/
  · simp only [gronwallBound_K0, hK, zero_mul, exp_zero, add_zero, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      δ K x : Real
      hK : Not (Eq K 0)
      ⊢ Eq (gronwallBound δ K 0 x) (HMul.hMul δ (Real.exp (HMul.hMul K x)))
    -/
  · simp only [gronwallBound_of_K_ne_0 hK, zero_div, zero_mul, add_zero]
    /-
      🎉 no goals
    -/


theorem gronwallBound_ε0_δ0 (K x : ℝ) : gronwallBound 0 K 0 x = 0 := by
  /-
    K x : Real
    ⊢ Eq (gronwallBound 0 K 0 x) 0
  -/
  simp only [gronwallBound_ε0, zero_mul]
  /-
    🎉 no goals
  -/


theorem gronwallBound_continuous_ε (δ K x : ℝ) : Continuous fun ε => gronwallBound δ K ε x := by
  /-
    δ K x : Real
    ⊢ Continuous fun ε => gronwallBound δ K ε x
  -/
  by_cases hK : K = 0
    /-
      case pos
      δ K x : Real
      hK : Eq K 0
      ⊢ Continuous fun ε => gronwallBound δ K ε x
    -/
  · simp only [gronwallBound_K0, hK]
    /-
      case pos
      δ K x : Real
      hK : Eq K 0
      ⊢ Continuous fun ε => HAdd.hAdd δ (HMul.hMul ε x)
    -/
    exact continuous_const.add (continuous_id.mul continuous_const)
    /-
      🎉 no goals
    -/
    /-
      case neg
      δ K x : Real
      hK : Not (Eq K 0)
      ⊢ Continuous fun ε => gronwallBound δ K ε x
    -/
  · simp only [gronwallBound_of_K_ne_0 hK]
    /-
      case neg
      δ K x : Real
      hK : Not (Eq K 0)
      ⊢ Continuous fun ε => HAdd.hAdd (HMul.hMul δ (Real.exp (HMul.hMul K x))) (HMul …
    -/
    exact continuous_const.add ((continuous_id.mul continuous_const).mul continuous_const)
    /-
      🎉 no goals
    -/


/-- A Grönwall-like inequality: if `f : ℝ → ℝ` is continuous on `[a, b]` and satisfies
the inequalities `f a ≤ δ` and
`∀ x ∈ [a, b), liminf_{z→x+0} (f z - f x)/(z - x) ≤ K * (f x) + ε`, then `f x`
is bounded by `gronwallBound δ K ε (x - a)` on `[a, b]`.

See also `norm_le_gronwallBound_of_norm_deriv_right_le` for a version bounding `‖f x‖`,
`f : ℝ → E`. -/
theorem le_gronwallBound_of_liminf_deriv_right_le {f f' : ℝ → ℝ} {δ K ε : ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ x ∈ Ico a b, ∀ r, f' x < r → ∃ᶠ z in 𝓝[>] x, (z - x)⁻¹ * (f z - f x) < r)
    (ha : f a ≤ δ) (bound : ∀ x ∈ Ico a b, f' x ≤ K * f x + ε) :
    ∀ x ∈ Icc a b, f x ≤ gronwallBound δ K ε (x - a) := by
  have H : ∀ x ∈ Icc a b, ∀ ε' ∈ Ioi ε, f x ≤ gronwallBound δ K ε' (x - a) := by
    intro x hx ε' hε'
    apply image_le_of_liminf_slope_right_lt_deriv_boundary hf hf'
    · rwa [sub_self, gronwallBound_x0]
    · exact fun x => hasDerivAt_gronwallBound_shift δ K ε' x a
    · intro x hx hfB
      rw [← hfB]
      apply lt_of_le_of_lt (bound x hx)
      exact add_lt_add_left (mem_Ioi.1 hε') _
    · exact hx
  /-
    f f' : Real → Real
    δ K ε a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    ha : LE.le (f a) δ
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (f' x) (HAdd.hAdd …
    H : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (ε' : Real), Membership.m …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (f x) (gronwallBound δ  …
  -/
  intro x hx
  /-
    f f' : Real → Real
    δ K ε a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    ha : LE.le (f a) δ
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (f' x) (HAdd.hAdd …
    H : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (ε' : Real), Membership.m …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    ⊢ LE.le (f x) (gronwallBound δ K ε (HSub.hSub x a))
  -/
  change f x ≤ (fun ε' => gronwallBound δ K ε' (x - a)) ε
  /-
    f f' : Real → Real
    δ K ε a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    ha : LE.le (f a) δ
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (f' x) (HAdd.hAdd …
    H : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (ε' : Real), Membership.m …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    ⊢ LE.le (f x) ((fun ε' => gronwallBound δ K ε' (HSub.hSub x a)) ε)
  -/
  convert continuousWithinAt_const.closure_le _ _ (H x hx)
    /-
      case convert_2
      f f' : Real → Real
      δ K ε a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
      ha : LE.le (f a) δ
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (f' x) (HAdd.hAdd …
      H : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (ε' : Real), Membership.m …
      x : Real
      hx : Membership.mem (Set.Icc a b) x
      ⊢ Membership.mem (closure (Set.Ioi ε)) ε
    -/
  · simp only [closure_Ioi, left_mem_Ici]
    /-
      🎉 no goals
    -/
  /-
    case convert_3
    f f' : Real → Real
    δ K ε a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    ha : LE.le (f a) δ
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (f' x) (HAdd.hAdd …
    H : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (ε' : Real), Membership.m …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    ⊢ ContinuousWithinAt (fun y => gronwallBound δ K y (HSub.hSub x a)) (Set.Ioi ε …
  -/
  exact (gronwallBound_continuous_ε δ K (x - a)).continuousWithinAt
  /-
    🎉 no goals
  -/


/-- A Grönwall-like inequality: if `f : ℝ → E` is continuous on `[a, b]`, has right derivative
`f' x` at every point `x ∈ [a, b)`, and satisfies the inequalities `‖f a‖ ≤ δ`,
`∀ x ∈ [a, b), ‖f' x‖ ≤ K * ‖f x‖ + ε`, then `‖f x‖` is bounded by `gronwallBound δ K ε (x - a)`
on `[a, b]`. -/
theorem norm_le_gronwallBound_of_norm_deriv_right_le {f f' : ℝ → E} {δ K ε : ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    (ha : ‖f a‖ ≤ δ) (bound : ∀ x ∈ Ico a b, ‖f' x‖ ≤ K * ‖f x‖ + ε) :
    ∀ x ∈ Icc a b, ‖f x‖ ≤ gronwallBound δ K ε (x - a) :=
  le_gronwallBound_of_liminf_deriv_right_le (continuous_norm.comp_continuousOn hf)
    (fun x hx _r hr => (hf' x hx).liminf_right_slope_norm_le hr) ha bound


/-- If `f` and `g` are two approximate solutions of the same ODE, then the distance between them
can't grow faster than exponentially. This is a simple corollary of Grönwall's inequality, and some
people call this Grönwall's inequality too.

This version assumes all inequalities to be true in some time-dependent set `s t`,
and assumes that the solutions never leave this set. -/
theorem dist_le_of_approx_trajectories_ODE_of_mem
    (hv : ∀ t ∈ Ico a b, LipschitzOnWith K (v t) (s t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ico a b, HasDerivWithinAt f (f' t) (Ici t) t)
    (f_bound : ∀ t ∈ Ico a b, dist (f' t) (v t (f t)) ≤ εf)
    (hfs : ∀ t ∈ Ico a b, f t ∈ s t)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ico a b, HasDerivWithinAt g (g' t) (Ici t) t)
    (g_bound : ∀ t ∈ Ico a b, dist (g' t) (v t (g t)) ≤ εg)
    (hgs : ∀ t ∈ Ico a b, g t ∈ s t)
    (ha : dist (f a) (g a) ≤ δ) :
    ∀ t ∈ Icc a b, dist (f t) (g t) ≤ gronwallBound δ K (εf + εg) (t - a) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Dist.dist (f a) (g a)) δ
    ⊢ ∀ (t : Real), Membership.mem (Set.Icc a b) t → LE.le (Dist.dist (f t) (g t)) …
  -/
  simp only [dist_eq_norm] at ha ⊢
  have h_deriv : ∀ t ∈ Ico a b, HasDerivWithinAt (fun t => f t - g t) (f' t - g' t) (Ici t) t :=
    fun t ht => (hf' t ht).sub (hg' t ht)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    ⊢ ∀ (t : Real), Membership.mem (Set.Icc a b) t → LE.le (Norm.norm (HSub.hSub ( …
  -/
  apply norm_le_gronwallBound_of_norm_deriv_right_le (hf.sub hg) h_deriv ha
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    ⊢ ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  intro t ht
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    t : Real
    ht : Membership.mem (Set.Ico a b) t
    ⊢ LE.le (Norm.norm (HSub.hSub (f' t) (g' t))) (HAdd.hAdd (HMul.hMul (↑K) (Norm …
  -/
  have := dist_triangle4_right (f' t) (g' t) (v t (f t)) (v t (g t))
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    t : Real
    ht : Membership.mem (Set.Ico a b) t
    this : LE.le (Dist.dist (f' t) (g' t)) (HAdd.hAdd (HAdd.hAdd (Dist.dist (f' t) …
    ⊢ LE.le (Norm.norm (HSub.hSub (f' t) (g' t))) (HAdd.hAdd (HMul.hMul (↑K) (Norm …
  -/
  have hv := (hv t ht).dist_le_mul _ (hfs t ht) _ (hgs t ht)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv✝ : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) ( …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    t : Real
    ht : Membership.mem (Set.Ico a b) t
    this : LE.le (Dist.dist (f' t) (g' t)) (HAdd.hAdd (HAdd.hAdd (Dist.dist (f' t) …
    hv : LE.le (Dist.dist (v t (f t)) (v t (g t))) (HMul.hMul (↑K) (Dist.dist (f t …
    ⊢ LE.le (Norm.norm (HSub.hSub (f' t) (g' t))) (HAdd.hAdd (HMul.hMul (↑K) (Norm …
  -/
  rw [← dist_eq_norm, ← dist_eq_norm]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv✝ : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) ( …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    t : Real
    ht : Membership.mem (Set.Ico a b) t
    this : LE.le (Dist.dist (f' t) (g' t)) (HAdd.hAdd (HAdd.hAdd (Dist.dist (f' t) …
    hv : LE.le (Dist.dist (v t (f t)) (v t (g t))) (HMul.hMul (↑K) (Dist.dist (f t …
    ⊢ LE.le (Dist.dist (f' t) (g' t)) (HAdd.hAdd (HMul.hMul (↑K) (Dist.dist (f t)  …
  -/
  refine this.trans ((add_le_add (add_le_add (f_bound t ht) (g_bound t ht)) hv).trans ?_)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g f' g' : Real → E
    a b εf εg δ : Real
    hv✝ : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) ( …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (f' t) …
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (f'  …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (g' t) …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (g'  …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Norm.norm (HSub.hSub (f a) (g a))) δ
    h_deriv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt (fun …
    t : Real
    ht : Membership.mem (Set.Ico a b) t
    this : LE.le (Dist.dist (f' t) (g' t)) (HAdd.hAdd (HAdd.hAdd (Dist.dist (f' t) …
    hv : LE.le (Dist.dist (v t (f t)) (v t (g t))) (HMul.hMul (↑K) (Dist.dist (f t …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd εf εg) (HMul.hMul (↑K) (Dist.dist (f t) (g t)))) …
  -/
  rw [add_comm]
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are two approximate solutions of the same ODE, then the distance between them
can't grow faster than exponentially. This is a simple corollary of Grönwall's inequality, and some
people call this Grönwall's inequality too.

This version assumes all inequalities to be true in the whole space. -/
theorem dist_le_of_approx_trajectories_ODE
    (hv : ∀ t, LipschitzWith K (v t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ico a b, HasDerivWithinAt f (f' t) (Ici t) t)
    (f_bound : ∀ t ∈ Ico a b, dist (f' t) (v t (f t)) ≤ εf)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ico a b, HasDerivWithinAt g (g' t) (Ici t) t)
    (g_bound : ∀ t ∈ Ico a b, dist (g' t) (v t (g t)) ≤ εg)
    (ha : dist (f a) (g a) ≤ δ) :
    ∀ t ∈ Icc a b, dist (f t) (g t) ≤ gronwallBound δ K (εf + εg) (t - a) :=
  have hfs : ∀ t ∈ Ico a b, f t ∈ @univ E := fun _ _ => trivial
  dist_le_of_approx_trajectories_ODE_of_mem (fun t _ => (hv t).lipschitzOnWith) hf hf'
    f_bound hfs hg hg' g_bound (fun _ _ => trivial) ha


/-- If `f` and `g` are two exact solutions of the same ODE, then the distance between them
can't grow faster than exponentially. This is a simple corollary of Grönwall's inequality, and some
people call this Grönwall's inequality too.

This version assumes all inequalities to be true in some time-dependent set `s t`,
and assumes that the solutions never leave this set. -/
theorem dist_le_of_trajectories_ODE_of_mem
    (hv : ∀ t ∈ Ico a b, LipschitzOnWith K (v t) (s t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ico a b, HasDerivWithinAt f (v t (f t)) (Ici t) t)
    (hfs : ∀ t ∈ Ico a b, f t ∈ s t)
    (hg : ContinuousOn g (Icc a b)) (hg' : ∀ t ∈ Ico a b, HasDerivWithinAt g (v t (g t)) (Ici t) t)
    (hgs : ∀ t ∈ Ico a b, g t ∈ s t) (ha : dist (f a) (g a) ≤ δ) :
    ∀ t ∈ Icc a b, dist (f t) (g t) ≤ δ * exp (K * (t - a)) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (v t ( …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (v t ( …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Dist.dist (f a) (g a)) δ
    ⊢ ∀ (t : Real), Membership.mem (Set.Icc a b) t → LE.le (Dist.dist (f t) (g t)) …
  -/
  have f_bound : ∀ t ∈ Ico a b, dist (v t (f t)) (v t (f t)) ≤ 0 := by intros; rw [dist_self]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (v t ( …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (v t ( …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Dist.dist (f a) (g a)) δ
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (v t …
    ⊢ ∀ (t : Real), Membership.mem (Set.Icc a b) t → LE.le (Dist.dist (f t) (g t)) …
  -/
  have g_bound : ∀ t ∈ Ico a b, dist (v t (g t)) (v t (g t)) ≤ 0 := by intros; rw [dist_self]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (v t ( …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (v t ( …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Dist.dist (f a) (g a)) δ
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (v t …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (v t …
    ⊢ ∀ (t : Real), Membership.mem (Set.Icc a b) t → LE.le (Dist.dist (f t) (g t)) …
  -/
  intro t ht
  have :=
    dist_le_of_approx_trajectories_ODE_of_mem hv hf hf' f_bound hfs hg hg' g_bound hgs ha t ht
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b δ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (v t ( …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (v t ( …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : LE.le (Dist.dist (f a) (g a)) δ
    f_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (v t …
    g_bound : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LE.le (Dist.dist (v t …
    t : Real
    ht : Membership.mem (Set.Icc a b) t
    this : LE.le (Dist.dist (f t) (g t)) (gronwallBound δ (↑K) (HAdd.hAdd 0 0) (HS …
    ⊢ LE.le (Dist.dist (f t) (g t)) (HMul.hMul δ (Real.exp (HMul.hMul (↑K) (HSub.h …
  -/
  rwa [zero_add, gronwallBound_ε0] at this
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are two exact solutions of the same ODE, then the distance between them
can't grow faster than exponentially. This is a simple corollary of Grönwall's inequality, and some
people call this Grönwall's inequality too.

This version assumes all inequalities to be true in the whole space. -/
theorem dist_le_of_trajectories_ODE
    (hv : ∀ t, LipschitzWith K (v t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ico a b, HasDerivWithinAt f (v t (f t)) (Ici t) t)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ico a b, HasDerivWithinAt g (v t (g t)) (Ici t) t)
    (ha : dist (f a) (g a) ≤ δ) :
    ∀ t ∈ Icc a b, dist (f t) (g t) ≤ δ * exp (K * (t - a)) :=
  have hfs : ∀ t ∈ Ico a b, f t ∈ @univ E := fun _ _ => trivial
  dist_le_of_trajectories_ODE_of_mem (fun t _ => (hv t).lipschitzOnWith) hf hf' hfs hg
    hg' (fun _ _ => trivial) ha


/-- There exists only one solution of an ODE \(\dot x=v(t, x)\) in a set `s ⊆ ℝ × E` with
a given initial value provided that the RHS is Lipschitz continuous in `x` within `s`,
and we consider only solutions included in `s`.

This version shows uniqueness in a closed interval `Icc a b`, where `a` is the initial time. -/
theorem ODE_solution_unique_of_mem_Icc_right
    (hv : ∀ t ∈ Ico a b, LipschitzOnWith K (v t) (s t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ico a b, HasDerivWithinAt f (v t (f t)) (Ici t) t)
    (hfs : ∀ t ∈ Ico a b, f t ∈ s t)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ico a b, HasDerivWithinAt g (v t (g t)) (Ici t) t)
    (hgs : ∀ t ∈ Ico a b, g t ∈ s t)
    (ha : f a = g a) :
    EqOn f g (Icc a b) := fun t ht ↦ by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (v t ( …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (v t ( …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : Eq (f a) (g a)
    t : Real
    ht : Membership.mem (Set.Icc a b) t
    ⊢ Eq (f t) (g t)
  -/
  have := dist_le_of_trajectories_ODE_of_mem hv hf hf' hfs hg hg' hgs (dist_le_zero.2 ha) t ht
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ico a b) t → LipschitzOnWith K (v t) (s …
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt f (v t ( …
    hfs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ico a b) t → HasDerivWithinAt g (v t ( …
    hgs : ∀ (t : Real), Membership.mem (Set.Ico a b) t → Membership.mem (s t) (g t)
    ha : Eq (f a) (g a)
    t : Real
    ht : Membership.mem (Set.Icc a b) t
    this : LE.le (Dist.dist (f t) (g t)) (HMul.hMul 0 (Real.exp (HMul.hMul (↑K) (H …
    ⊢ Eq (f t) (g t)
  -/
  rwa [zero_mul, dist_le_zero] at this
  /-
    🎉 no goals
  -/


/-- A time-reversed version of `ODE_solution_unique_of_mem_Icc_right`. Uniqueness is shown in a
closed interval `Icc a b`, where `b` is the "initial" time. -/
theorem ODE_solution_unique_of_mem_Icc_left
    (hv : ∀ t ∈ Ioc a b, LipschitzOnWith K (v t) (s t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ioc a b, HasDerivWithinAt f (v t (f t)) (Iic t) t)
    (hfs : ∀ t ∈ Ioc a b, f t ∈ s t)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ioc a b, HasDerivWithinAt g (v t (g t)) (Iic t) t)
    (hgs : ∀ t ∈ Ioc a b, g t ∈ s t)
    (hb : f b = g b) :
    EqOn f g (Icc a b) := by
  have hv' : ∀ t ∈ Ico (-b) (-a), LipschitzOnWith K (Neg.neg ∘ (v (-t))) (s (-t)) := by
    intro t ht
    replace ht : -t ∈ Ioc a b := by
      simp at ht ⊢
      constructor <;> linarith
    rw [← one_mul K]
    exact LipschitzWith.id.neg.comp_lipschitzOnWith (hv _ ht)
  have hmt1 : MapsTo Neg.neg (Icc (-b) (-a)) (Icc a b) :=
    fun _ ht ↦ ⟨le_neg.mp ht.2, neg_le.mp ht.1⟩
  have hmt2 : MapsTo Neg.neg (Ico (-b) (-a)) (Ioc a b) :=
    fun _ ht ↦ ⟨lt_neg.mp ht.2, neg_le.mp ht.1⟩
  have hmt3 (t : ℝ) : MapsTo Neg.neg (Ici t) (Iic (-t)) :=
    fun _ ht' ↦ mem_Iic.mpr <| neg_le_neg ht'
  suffices EqOn (f ∘ Neg.neg) (g ∘ Neg.neg) (Icc (-b) (-a)) by
    rw [eqOn_comp_right_iff] at this
    convert this
    simp
  apply ODE_solution_unique_of_mem_Icc_right hv'
    (hf.comp continuousOn_neg hmt1) _ (fun _ ht ↦ hfs _ (hmt2 ht))
    (hg.comp continuousOn_neg hmt1) _ (fun _ ht ↦ hgs _ (hmt2 ht)) (by simp [hb])
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      v : Real → E → E
      s : Real → Set E
      K : NNReal
      f g : Real → E
      a b : Real
      hv : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → LipschitzOnWith K (v t) (s …
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt f (v t ( …
      hfs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (f t)
      hg : ContinuousOn g (Set.Icc a b)
      hg' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt g (v t ( …
      hgs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (g t)
      hb : Eq (f b) (g b)
      hv' : ∀ (t : Real), Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t → Lipsc …
      hmt1 : Set.MapsTo Neg.neg (Set.Icc (Neg.neg b) (Neg.neg a)) (Set.Icc a b)
      hmt2 : Set.MapsTo Neg.neg (Set.Ico (Neg.neg b) (Neg.neg a)) (Set.Ioc a b)
      hmt3 : ∀ (t : Real), Set.MapsTo Neg.neg (Set.Ici t) (Set.Iic (Neg.neg t))
      ⊢ ∀ (t : Real), Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t → HasDerivW …
    -/
  · intros t ht
    convert HasFDerivWithinAt.comp_hasDerivWithinAt t (hf' (-t) (hmt2 ht))
      (hasDerivAt_neg t).hasDerivWithinAt (hmt3 t)
    /-
      case h.e'_9
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      v : Real → E → E
      s : Real → Set E
      K : NNReal
      f g : Real → E
      a b : Real
      hv : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → LipschitzOnWith K (v t) (s …
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt f (v t ( …
      hfs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (f t)
      hg : ContinuousOn g (Set.Icc a b)
      hg' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt g (v t ( …
      hgs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (g t)
      hb : Eq (f b) (g b)
      hv' : ∀ (t : Real), Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t → Lipsc …
      hmt1 : Set.MapsTo Neg.neg (Set.Icc (Neg.neg b) (Neg.neg a)) (Set.Icc a b)
      hmt2 : Set.MapsTo Neg.neg (Set.Ico (Neg.neg b) (Neg.neg a)) (Set.Ioc a b)
      hmt3 : ∀ (t : Real), Set.MapsTo Neg.neg (Set.Ici t) (Set.Iic (Neg.neg t))
      t : Real
      ht : Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t
      ⊢ Eq (Function.comp Neg.neg (v (Neg.neg t)) (Function.comp f Neg.neg t)) ((Con …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      v : Real → E → E
      s : Real → Set E
      K : NNReal
      f g : Real → E
      a b : Real
      hv : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → LipschitzOnWith K (v t) (s …
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt f (v t ( …
      hfs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (f t)
      hg : ContinuousOn g (Set.Icc a b)
      hg' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt g (v t ( …
      hgs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (g t)
      hb : Eq (f b) (g b)
      hv' : ∀ (t : Real), Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t → Lipsc …
      hmt1 : Set.MapsTo Neg.neg (Set.Icc (Neg.neg b) (Neg.neg a)) (Set.Icc a b)
      hmt2 : Set.MapsTo Neg.neg (Set.Ico (Neg.neg b) (Neg.neg a)) (Set.Ioc a b)
      hmt3 : ∀ (t : Real), Set.MapsTo Neg.neg (Set.Ici t) (Set.Iic (Neg.neg t))
      ⊢ ∀ (t : Real), Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t → HasDerivW …
    -/
  · intros t ht
    convert HasFDerivWithinAt.comp_hasDerivWithinAt t (hg' (-t) (hmt2 ht))
      (hasDerivAt_neg t).hasDerivWithinAt (hmt3 t)
    /-
      case h.e'_9
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      v : Real → E → E
      s : Real → Set E
      K : NNReal
      f g : Real → E
      a b : Real
      hv : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → LipschitzOnWith K (v t) (s …
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt f (v t ( …
      hfs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (f t)
      hg : ContinuousOn g (Set.Icc a b)
      hg' : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → HasDerivWithinAt g (v t ( …
      hgs : ∀ (t : Real), Membership.mem (Set.Ioc a b) t → Membership.mem (s t) (g t)
      hb : Eq (f b) (g b)
      hv' : ∀ (t : Real), Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t → Lipsc …
      hmt1 : Set.MapsTo Neg.neg (Set.Icc (Neg.neg b) (Neg.neg a)) (Set.Icc a b)
      hmt2 : Set.MapsTo Neg.neg (Set.Ico (Neg.neg b) (Neg.neg a)) (Set.Ioc a b)
      hmt3 : ∀ (t : Real), Set.MapsTo Neg.neg (Set.Ici t) (Set.Iic (Neg.neg t))
      t : Real
      ht : Membership.mem (Set.Ico (Neg.neg b) (Neg.neg a)) t
      ⊢ Eq (Function.comp Neg.neg (v (Neg.neg t)) (Function.comp g Neg.neg t)) ((Con …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- A version of `ODE_solution_unique_of_mem_Icc_right` for uniqueness in a closed interval whose
interior contains the initial time. -/
theorem ODE_solution_unique_of_mem_Icc
    (hv : ∀ t ∈ Ioo a b, LipschitzOnWith K (v t) (s t))
    (ht : t₀ ∈ Ioo a b)
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ioo a b, HasDerivAt f (v t (f t)) t)
    (hfs : ∀ t ∈ Ioo a b, f t ∈ s t)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ioo a b, HasDerivAt g (v t (g t)) t)
    (hgs : ∀ t ∈ Ioo a b, g t ∈ s t)
    (heq : f t₀ = g t₀) :
    EqOn f g (Icc a b) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b t₀ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → LipschitzOnWith K (v t) (s …
    ht : Membership.mem (Set.Ioo a b) t₀
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt f (v t (f t)) t
    hfs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt g (v t (g t)) t
    hgs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (g t)
    heq : Eq (f t₀) (g t₀)
    ⊢ Set.EqOn f g (Set.Icc a b)
  -/
  rw [← Icc_union_Icc_eq_Icc (le_of_lt ht.1) (le_of_lt ht.2)]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b t₀ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → LipschitzOnWith K (v t) (s …
    ht : Membership.mem (Set.Ioo a b) t₀
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt f (v t (f t)) t
    hfs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (f t)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt g (v t (g t)) t
    hgs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (g t)
    heq : Eq (f t₀) (g t₀)
    ⊢ Set.EqOn f g (Union.union (Set.Icc a t₀) (Set.Icc t₀ b))
  -/
  apply EqOn.union
    /-
      case h₁
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      v : Real → E → E
      s : Real → Set E
      K : NNReal
      f g : Real → E
      a b t₀ : Real
      hv : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → LipschitzOnWith K (v t) (s …
      ht : Membership.mem (Set.Ioo a b) t₀
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt f (v t (f t)) t
      hfs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (f t)
      hg : ContinuousOn g (Set.Icc a b)
      hg' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt g (v t (g t)) t
      hgs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (g t)
      heq : Eq (f t₀) (g t₀)
      ⊢ Set.EqOn f g (Set.Icc a t₀)
    -/
  · have hss : Ioc a t₀ ⊆ Ioo a b := Ioc_subset_Ioo_right ht.2
    exact ODE_solution_unique_of_mem_Icc_left (fun t ht ↦ hv t (hss ht))
      (hf.mono <| Icc_subset_Icc_right <| le_of_lt ht.2)
      (fun _ ht' ↦ (hf' _ (hss ht')).hasDerivWithinAt) (fun _ ht' ↦ (hfs _ (hss ht')))
      (hg.mono <| Icc_subset_Icc_right <| le_of_lt ht.2)
      (fun _ ht' ↦ (hg' _ (hss ht')).hasDerivWithinAt) (fun _ ht' ↦ (hgs _ (hss ht'))) heq
    /-
      case h₂
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      v : Real → E → E
      s : Real → Set E
      K : NNReal
      f g : Real → E
      a b t₀ : Real
      hv : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → LipschitzOnWith K (v t) (s …
      ht : Membership.mem (Set.Ioo a b) t₀
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt f (v t (f t)) t
      hfs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (f t)
      hg : ContinuousOn g (Set.Icc a b)
      hg' : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → HasDerivAt g (v t (g t)) t
      hgs : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → Membership.mem (s t) (g t)
      heq : Eq (f t₀) (g t₀)
      ⊢ Set.EqOn f g (Set.Icc t₀ b)
    -/
  · have hss : Ico t₀ b ⊆ Ioo a b := Ico_subset_Ioo_left ht.1
    exact ODE_solution_unique_of_mem_Icc_right (fun t ht ↦ hv t (hss ht))
      (hf.mono <| Icc_subset_Icc_left <| le_of_lt ht.1)
      (fun _ ht' ↦ (hf' _ (hss ht')).hasDerivWithinAt) (fun _ ht' ↦ (hfs _ (hss ht')))
      (hg.mono <| Icc_subset_Icc_left <| le_of_lt ht.1)
      (fun _ ht' ↦ (hg' _ (hss ht')).hasDerivWithinAt) (fun _ ht' ↦ (hgs _ (hss ht'))) heq


/-- A version of `ODE_solution_unique_of_mem_Icc` for uniqueness in an open interval. -/
theorem ODE_solution_unique_of_mem_Ioo
    (hv : ∀ t ∈ Ioo a b, LipschitzOnWith K (v t) (s t))
    (ht : t₀ ∈ Ioo a b)
    (hf : ∀ t ∈ Ioo a b, HasDerivAt f (v t (f t)) t ∧ f t ∈ s t)
    (hg : ∀ t ∈ Ioo a b, HasDerivAt g (v t (g t)) t ∧ g t ∈ s t)
    (heq : f t₀ = g t₀) :
    EqOn f g (Ioo a b) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b t₀ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → LipschitzOnWith K (v t) (s …
    ht : Membership.mem (Set.Ioo a b) t₀
    hf : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → And (HasDerivAt f (v t (f  …
    hg : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → And (HasDerivAt g (v t (g  …
    heq : Eq (f t₀) (g t₀)
    ⊢ Set.EqOn f g (Set.Ioo a b)
  -/
  intros t' ht'
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    a b t₀ : Real
    hv : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → LipschitzOnWith K (v t) (s …
    ht : Membership.mem (Set.Ioo a b) t₀
    hf : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → And (HasDerivAt f (v t (f  …
    hg : ∀ (t : Real), Membership.mem (Set.Ioo a b) t → And (HasDerivAt g (v t (g  …
    heq : Eq (f t₀) (g t₀)
    t' : Real
    ht' : Membership.mem (Set.Ioo a b) t'
    ⊢ Eq (f t') (g t')
  -/
  rcases lt_or_le t' t₀ with (h | h)
  · have hss : Icc t' t₀ ⊆ Ioo a b :=
      fun _ ht'' ↦ ⟨lt_of_lt_of_le ht'.1 ht''.1, lt_of_le_of_lt ht''.2 ht.2⟩
    exact ODE_solution_unique_of_mem_Icc_left
      (fun t'' ht'' ↦ hv t'' ((Ioc_subset_Icc_self.trans hss) ht''))
      (continuousOn_of_forall_continuousAt fun _ ht'' ↦ (hf _ <| hss ht'').1.continuousAt)
      (fun _ ht'' ↦ (hf _ <| hss <| Ioc_subset_Icc_self ht'').1.hasDerivWithinAt)
      (fun _ ht'' ↦ (hf _ <| hss <| Ioc_subset_Icc_self ht'').2)
      (continuousOn_of_forall_continuousAt fun _ ht'' ↦ (hg _ <| hss ht'').1.continuousAt)
      (fun _ ht'' ↦ (hg _ <| hss <| Ioc_subset_Icc_self ht'').1.hasDerivWithinAt)
      (fun _ ht'' ↦ (hg _ <| hss <| Ioc_subset_Icc_self ht'').2) heq
      ⟨le_rfl, le_of_lt h⟩
  · have hss : Icc t₀ t' ⊆ Ioo a b :=
      fun _ ht'' ↦ ⟨lt_of_lt_of_le ht.1 ht''.1, lt_of_le_of_lt ht''.2 ht'.2⟩
    exact ODE_solution_unique_of_mem_Icc_right
      (fun t'' ht'' ↦ hv t'' ((Ico_subset_Icc_self.trans hss) ht''))
      (continuousOn_of_forall_continuousAt fun _ ht'' ↦ (hf _ <| hss ht'').1.continuousAt)
      (fun _ ht'' ↦ (hf _ <| hss <| Ico_subset_Icc_self ht'').1.hasDerivWithinAt)
      (fun _ ht'' ↦ (hf _ <| hss <| Ico_subset_Icc_self ht'').2)
      (continuousOn_of_forall_continuousAt fun _ ht'' ↦ (hg _ <| hss ht'').1.continuousAt)
      (fun _ ht'' ↦ (hg _ <| hss <| Ico_subset_Icc_self ht'').1.hasDerivWithinAt)
      (fun _ ht'' ↦ (hg _ <| hss <| Ico_subset_Icc_self ht'').2) heq
      ⟨h, le_rfl⟩


/-- Local unqueness of ODE solutions. -/
theorem ODE_solution_unique_of_eventually
    (hv : ∀ᶠ t in 𝓝 t₀, LipschitzOnWith K (v t) (s t))
    (hf : ∀ᶠ t in 𝓝 t₀, HasDerivAt f (v t (f t)) t ∧ f t ∈ s t)
    (hg : ∀ᶠ t in 𝓝 t₀, HasDerivAt g (v t (g t)) t ∧ g t ∈ s t)
    (heq : f t₀ = g t₀) : f =ᶠ[𝓝 t₀] g := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    t₀ : Real
    hv : Filter.Eventually (fun t => LipschitzOnWith K (v t) (s t)) (nhds t₀)
    hf : Filter.Eventually (fun t => And (HasDerivAt f (v t (f t)) t) (Membership. …
    hg : Filter.Eventually (fun t => And (HasDerivAt g (v t (g t)) t) (Membership. …
    heq : Eq (f t₀) (g t₀)
    ⊢ (nhds t₀).EventuallyEq f g
  -/
  obtain ⟨ε, hε, h⟩ := eventually_nhds_iff_ball.mp (hv.and (hf.and hg))
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    t₀ : Real
    hv : Filter.Eventually (fun t => LipschitzOnWith K (v t) (s t)) (nhds t₀)
    hf : Filter.Eventually (fun t => And (HasDerivAt f (v t (f t)) t) (Membership. …
    hg : Filter.Eventually (fun t => And (HasDerivAt g (v t (g t)) t) (Membership. …
    heq : Eq (f t₀) (g t₀)
    ε : Real
    hε : GT.gt ε 0
    h : ∀ (y : Real), Membership.mem (Metric.ball t₀ ε) y → And (LipschitzOnWith K …
    ⊢ (nhds t₀).EventuallyEq f g
  -/
  rw [Filter.eventuallyEq_iff_exists_mem]
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    t₀ : Real
    hv : Filter.Eventually (fun t => LipschitzOnWith K (v t) (s t)) (nhds t₀)
    hf : Filter.Eventually (fun t => And (HasDerivAt f (v t (f t)) t) (Membership. …
    hg : Filter.Eventually (fun t => And (HasDerivAt g (v t (g t)) t) (Membership. …
    heq : Eq (f t₀) (g t₀)
    ε : Real
    hε : GT.gt ε 0
    h : ∀ (y : Real), Membership.mem (Metric.ball t₀ ε) y → And (LipschitzOnWith K …
    ⊢ Exists fun s => And (Membership.mem (nhds t₀) s) (Set.EqOn f g s)
  -/
  refine ⟨ball t₀ ε, ball_mem_nhds _ hε, ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    v : Real → E → E
    s : Real → Set E
    K : NNReal
    f g : Real → E
    t₀ : Real
    hv : Filter.Eventually (fun t => LipschitzOnWith K (v t) (s t)) (nhds t₀)
    hf : Filter.Eventually (fun t => And (HasDerivAt f (v t (f t)) t) (Membership. …
    hg : Filter.Eventually (fun t => And (HasDerivAt g (v t (g t)) t) (Membership. …
    heq : Eq (f t₀) (g t₀)
    ε : Real
    hε : GT.gt ε 0
    h : ∀ (y : Real), Membership.mem (Metric.ball t₀ ε) y → And (LipschitzOnWith K …
    ⊢ Set.EqOn f g (Metric.ball t₀ ε)
  -/
  simp_rw [Real.ball_eq_Ioo] at *
  apply ODE_solution_unique_of_mem_Ioo (fun _ ht ↦ (h _ ht).1)
    (Real.ball_eq_Ioo t₀ ε ▸ mem_ball_self hε)
    (fun _ ht ↦ (h _ ht).2.1) (fun _ ht ↦ (h _ ht).2.2) heq


/-- There exists only one solution of an ODE \(\dot x=v(t, x)\) with
a given initial value provided that the RHS is Lipschitz continuous in `x`. -/
theorem ODE_solution_unique
    (hv : ∀ t, LipschitzWith K (v t))
    (hf : ContinuousOn f (Icc a b))
    (hf' : ∀ t ∈ Ico a b, HasDerivWithinAt f (v t (f t)) (Ici t) t)
    (hg : ContinuousOn g (Icc a b))
    (hg' : ∀ t ∈ Ico a b, HasDerivWithinAt g (v t (g t)) (Ici t) t)
    (ha : f a = g a) :
    EqOn f g (Icc a b) :=
  have hfs : ∀ t ∈ Ico a b, f t ∈ @univ E := fun _ _ => trivial
  ODE_solution_unique_of_mem_Icc_right (fun t _ => (hv t).lipschitzOnWith) hf hf' hfs hg hg'
    (fun _ _ => trivial) ha

