/-- A circle integral which coincides with `deriv f z` whenever one can apply the Cauchy formula for
the derivative. It is useful in the proof that locally uniform limits of holomorphic functions are
holomorphic, because it depends continuously on `f` for the uniform topology. -/
noncomputable def cderiv (r : ℝ) (f : ℂ → E) (z : ℂ) : E :=
  (2 * π * I : ℂ)⁻¹ • ∮ w in C(z, r), ((w - z) ^ 2)⁻¹ • f w


theorem cderiv_eq_deriv [CompleteSpace E] (hU : IsOpen U) (hf : DifferentiableOn ℂ f U) (hr : 0 < r)
    (hzr : closedBall z r ⊆ U) : cderiv r f z = deriv f z :=
  two_pi_I_inv_smul_circleIntegral_sub_sq_inv_smul_of_differentiable hU hzr hf (mem_ball_self hr)


theorem norm_cderiv_le (hr : 0 < r) (hf : ∀ w ∈ sphere z r, ‖f w‖ ≤ M) :
    ‖cderiv r f z‖ ≤ M / r := by
  have hM : 0 ≤ M := by
    obtain ⟨w, hw⟩ : (sphere z r).Nonempty := NormedSpace.sphere_nonempty.mpr hr.le
    exact (norm_nonneg _).trans (hf w hw)
  have h1 : ∀ w ∈ sphere z r, ‖((w - z) ^ 2)⁻¹ • f w‖ ≤ M / r ^ 2 := by
    intro w hw
    simp only [mem_sphere_iff_norm, norm_eq_abs] at hw
    simp only [norm_smul, inv_mul_eq_div, hw, norm_eq_abs, map_inv₀, Complex.abs_pow]
    exact div_le_div₀ hM (hf w hw) (sq_pos_of_pos hr) le_rfl
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    M r : Real
    f : Complex → E
    hr : LT.lt 0 r
    hf : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    hM : LE.le 0 M
    h1 : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    ⊢ LE.le (Norm.norm (Complex.cderiv r f z)) (HDiv.hDiv M r)
  -/
  have h2 := circleIntegral.norm_integral_le_of_norm_le_const hr.le h1
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    M r : Real
    f : Complex → E
    hr : LT.lt 0 r
    hf : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    hM : LE.le 0 M
    h1 : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    h2 : LE.le (Norm.norm (circleIntegral (fun z_1 => HSMul.hSMul (Inv.inv (HPow.h …
    ⊢ LE.le (Norm.norm (Complex.cderiv r f z)) (HDiv.hDiv M r)
  -/
  simp only [cderiv, norm_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    M r : Real
    f : Complex → E
    hr : LT.lt 0 r
    hf : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    hM : LE.le 0 M
    h1 : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    h2 : LE.le (Norm.norm (circleIntegral (fun z_1 => HSMul.hSMul (Inv.inv (HPow.h …
    ⊢ LE.le (HMul.hMul (Norm.norm (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Compl …
  -/
  refine (mul_le_mul le_rfl h2 (norm_nonneg _) (norm_nonneg _)).trans (le_of_eq ?_)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    M r : Real
    f : Complex → E
    hr : LT.lt 0 r
    hf : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    hM : LE.le 0 M
    h1 : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    h2 : LE.le (Norm.norm (circleIntegral (fun z_1 => HSMul.hSMul (Inv.inv (HPow.h …
    ⊢ Eq (HMul.hMul (Norm.norm (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex. …
  -/
  field_simp [_root_.abs_of_nonneg Real.pi_pos.le]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    M r : Real
    f : Complex → E
    hr : LT.lt 0 r
    hf : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    hM : LE.le 0 M
    h1 : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm  …
    h2 : LE.le (Norm.norm (circleIntegral (fun z_1 => HSMul.hSMul (Inv.inv (HPow.h …
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) r) M) r) (HMul.hMu …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem cderiv_sub (hr : 0 < r) (hf : ContinuousOn f (sphere z r))
    (hg : ContinuousOn g (sphere z r)) : cderiv r (f - g) z = cderiv r f z - cderiv r g z := by
  have h1 : ContinuousOn (fun w : ℂ => ((w - z) ^ 2)⁻¹) (sphere z r) := by
    refine ((continuous_id'.sub continuous_const).pow 2).continuousOn.inv₀ fun w hw h => hr.ne ?_
    rwa [mem_sphere_iff_norm, sq_eq_zero_iff.mp h, norm_zero] at hw
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    r : Real
    f g : Complex → E
    hr : LT.lt 0 r
    hf : ContinuousOn f (Metric.sphere z r)
    hg : ContinuousOn g (Metric.sphere z r)
    h1 : ContinuousOn (fun w => Inv.inv (HPow.hPow (HSub.hSub w z) 2)) (Metric.sph …
    ⊢ Eq (Complex.cderiv r (HSub.hSub f g) z) (HSub.hSub (Complex.cderiv r f z) (C …
  -/
  simp_rw [cderiv, ← smul_sub]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    r : Real
    f g : Complex → E
    hr : LT.lt 0 r
    hf : ContinuousOn f (Metric.sphere z r)
    hg : ContinuousOn g (Metric.sphere z r)
    h1 : ContinuousOn (fun w => Inv.inv (HPow.hPow (HSub.hSub w z) 2)) (Metric.sph …
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (circ …
  -/
  congr 1
  simpa only [Pi.sub_apply, smul_sub] using
    circleIntegral.integral_sub ((h1.smul hf).circleIntegrable hr.le)
      ((h1.smul hg).circleIntegrable hr.le)


theorem norm_cderiv_lt (hr : 0 < r) (hfM : ∀ w ∈ sphere z r, ‖f w‖ < M)
    (hf : ContinuousOn f (sphere z r)) : ‖cderiv r f z‖ < M / r := by
  obtain ⟨L, hL1, hL2⟩ : ∃ L < M, ∀ w ∈ sphere z r, ‖f w‖ ≤ L := by
    have e1 : (sphere z r).Nonempty := NormedSpace.sphere_nonempty.mpr hr.le
    have e2 : ContinuousOn (fun w => ‖f w‖) (sphere z r) := continuous_norm.comp_continuousOn hf
    obtain ⟨x, hx, hx'⟩ := (isCompact_sphere z r).exists_isMaxOn e1 e2
    exact ⟨‖f x‖, hfM x hx, hx'⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    M r : Real
    f : Complex → E
    hr : LT.lt 0 r
    hfM : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LT.lt (Norm.norm …
    hf : ContinuousOn f (Metric.sphere z r)
    L : Real
    hL1 : LT.lt L M
    hL2 : ∀ (w : Complex), Membership.mem (Metric.sphere z r) w → LE.le (Norm.norm …
    ⊢ LT.lt (Norm.norm (Complex.cderiv r f z)) (HDiv.hDiv M r)
  -/
  exact (norm_cderiv_le hr hL2).trans_lt ((div_lt_div_iff_of_pos_right hr).mpr hL1)
  /-
    🎉 no goals
  -/


theorem norm_cderiv_sub_lt (hr : 0 < r) (hfg : ∀ w ∈ sphere z r, ‖f w - g w‖ < M)
    (hf : ContinuousOn f (sphere z r)) (hg : ContinuousOn g (sphere z r)) :
    ‖cderiv r f z - cderiv r g z‖ < M / r :=
  cderiv_sub hr hf hg ▸ norm_cderiv_lt hr hfg (hf.sub hg)


theorem _root_.TendstoUniformlyOn.cderiv (hF : TendstoUniformlyOn F f φ (cthickening δ K))
    (hδ : 0 < δ) (hFn : ∀ᶠ n in φ, ContinuousOn (F n) (cthickening δ K)) :
    TendstoUniformlyOn (cderiv δ ∘ F) (cderiv δ f) φ K := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : TendstoUniformlyOn F f φ (Metric.cthickening δ K)
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    ⊢ TendstoUniformlyOn (Function.comp (Complex.cderiv δ) F) (Complex.cderiv δ f) …
  -/
  rcases φ.eq_or_neBot with rfl | hne
    /-
      case inl
      E : Type u_1
      ι : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      K : Set Complex
      δ : Real
      F : ι → Complex → E
      f : Complex → E
      hδ : LT.lt 0 δ
      hF : TendstoUniformlyOn F f Bot.bot (Metric.cthickening δ K)
      hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) …
      ⊢ TendstoUniformlyOn (Function.comp (Complex.cderiv δ) F) (Complex.cderiv δ f) …
    -/
  · simp only [TendstoUniformlyOn, eventually_bot, imp_true_iff]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : TendstoUniformlyOn F f φ (Metric.cthickening δ K)
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    ⊢ TendstoUniformlyOn (Function.comp (Complex.cderiv δ) F) (Complex.cderiv δ f) …
  -/
  have e1 : ContinuousOn f (cthickening δ K) := TendstoUniformlyOn.continuousOn hF hFn
  /-
    case inr
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : TendstoUniformlyOn F f φ (Metric.cthickening δ K)
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    e1 : ContinuousOn f (Metric.cthickening δ K)
    ⊢ TendstoUniformlyOn (Function.comp (Complex.cderiv δ) F) (Complex.cderiv δ f) …
  -/
  rw [tendstoUniformlyOn_iff] at hF ⊢
  /-
    case inr
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : Complex), Me …
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    e1 : ContinuousOn f (Metric.cthickening δ K)
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : Complex), Membe …
  -/
  rintro ε hε
  /-
    case inr
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : Complex), Me …
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    e1 : ContinuousOn f (Metric.cthickening δ K)
    ε : Real
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun n => ∀ (x : Complex), Membership.mem K x → LT.lt (Dis …
  -/
  filter_upwards [hF (ε * δ) (mul_pos hε hδ), hFn] with n h h' z hz
  /-
    case h
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : Complex), Me …
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    e1 : ContinuousOn f (Metric.cthickening δ K)
    ε : Real
    hε : GT.gt ε 0
    n : ι
    h : ∀ (x : Complex), Membership.mem (Metric.cthickening δ K) x → LT.lt (Dist.d …
    h' : ContinuousOn (F n) (Metric.cthickening δ K)
    z : Complex
    hz : Membership.mem K z
    ⊢ LT.lt (Dist.dist (Complex.cderiv δ f z) (Function.comp (Complex.cderiv δ) F  …
  -/
  simp_rw [dist_eq_norm] at h ⊢
  have e2 : ∀ w ∈ sphere z δ, ‖f w - F n w‖ < ε * δ := fun w hw1 =>
    h w (closedBall_subset_cthickening hz δ (sphere_subset_closedBall hw1))
  /-
    case h
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : Complex), Me …
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    e1 : ContinuousOn f (Metric.cthickening δ K)
    ε : Real
    hε : GT.gt ε 0
    n : ι
    h' : ContinuousOn (F n) (Metric.cthickening δ K)
    z : Complex
    hz : Membership.mem K z
    h : ∀ (x : Complex), Membership.mem (Metric.cthickening δ K) x → LT.lt (Norm.n …
    e2 : ∀ (w : Complex), Membership.mem (Metric.sphere z δ) w → LT.lt (Norm.norm  …
    ⊢ LT.lt (Norm.norm (HSub.hSub (Complex.cderiv δ f z) (Function.comp (Complex.c …
  -/
  have e3 := sphere_subset_closedBall.trans (closedBall_subset_cthickening hz δ)
  have hf : ContinuousOn f (sphere z δ) :=
    e1.mono (sphere_subset_closedBall.trans (closedBall_subset_cthickening hz δ))
  /-
    case h
    E : Type u_1
    ι : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    K : Set Complex
    δ : Real
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    hF : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun n => ∀ (x : Complex), Me …
    hδ : LT.lt 0 δ
    hFn : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    hne : φ.NeBot
    e1 : ContinuousOn f (Metric.cthickening δ K)
    ε : Real
    hε : GT.gt ε 0
    n : ι
    h' : ContinuousOn (F n) (Metric.cthickening δ K)
    z : Complex
    hz : Membership.mem K z
    h : ∀ (x : Complex), Membership.mem (Metric.cthickening δ K) x → LT.lt (Norm.n …
    e2 : ∀ (w : Complex), Membership.mem (Metric.sphere z δ) w → LT.lt (Norm.norm  …
    e3 : HasSubset.Subset (Metric.sphere z δ) (Metric.cthickening δ K)
    hf : ContinuousOn f (Metric.sphere z δ)
    ⊢ LT.lt (Norm.norm (HSub.hSub (Complex.cderiv δ f z) (Function.comp (Complex.c …
  -/
  simpa only [mul_div_cancel_right₀ _ hδ.ne.symm] using norm_cderiv_sub_lt hδ e2 hf (h'.mono e3)
  /-
    🎉 no goals
  -/


theorem tendstoUniformlyOn_deriv_of_cthickening_subset (hf : TendstoLocallyUniformlyOn F f φ U)
    (hF : ∀ᶠ n in φ, DifferentiableOn ℂ (F n) U) {δ : ℝ} (hδ : 0 < δ) (hK : IsCompact K)
    (hU : IsOpen U) (hKU : cthickening δ K ⊆ U) :
    TendstoUniformlyOn (deriv ∘ F) (cderiv δ f) φ K := by
  have h1 : ∀ᶠ n in φ, ContinuousOn (F n) (cthickening δ K) := by
    filter_upwards [hF] with n h using h.continuousOn.mono hKU
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U K : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    δ : Real
    hδ : LT.lt 0 δ
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    ⊢ TendstoUniformlyOn (Function.comp deriv F) (Complex.cderiv δ f) φ K
  -/
  have h2 : IsCompact (cthickening δ K) := hK.cthickening
  have h3 : TendstoUniformlyOn F f φ (cthickening δ K) :=
    (tendstoLocallyUniformlyOn_iff_forall_isCompact hU).mp hf (cthickening δ K) hKU h2
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U K : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    δ : Real
    hδ : LT.lt 0 δ
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    h2 : IsCompact (Metric.cthickening δ K)
    h3 : TendstoUniformlyOn F f φ (Metric.cthickening δ K)
    ⊢ TendstoUniformlyOn (Function.comp deriv F) (Complex.cderiv δ f) φ K
  -/
  apply (h3.cderiv hδ h1).congr
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U K : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    δ : Real
    hδ : LT.lt 0 δ
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    h2 : IsCompact (Metric.cthickening δ K)
    h3 : TendstoUniformlyOn F f φ (Metric.cthickening δ K)
    ⊢ Filter.Eventually (fun n => Set.EqOn (Function.comp (Complex.cderiv δ) F n)  …
  -/
  filter_upwards [hF] with n h z hz
  /-
    case h
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U K : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    δ : Real
    hδ : LT.lt 0 δ
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : Filter.Eventually (fun n => ContinuousOn (F n) (Metric.cthickening δ K)) φ
    h2 : IsCompact (Metric.cthickening δ K)
    h3 : TendstoUniformlyOn F f φ (Metric.cthickening δ K)
    n : ι
    h : DifferentiableOn Complex (F n) U
    z : Complex
    hz : Membership.mem K z
    ⊢ Eq (Function.comp (Complex.cderiv δ) F n z) (Function.comp deriv F n z)
  -/
  exact cderiv_eq_deriv hU h hδ ((closedBall_subset_cthickening hz δ).trans hKU)
  /-
    🎉 no goals
  -/


theorem exists_cthickening_tendstoUniformlyOn (hf : TendstoLocallyUniformlyOn F f φ U)
    (hF : ∀ᶠ n in φ, DifferentiableOn ℂ (F n) U) (hK : IsCompact K) (hU : IsOpen U) (hKU : K ⊆ U) :
    ∃ δ > 0, cthickening δ K ⊆ U ∧ TendstoUniformlyOn (deriv ∘ F) (cderiv δ f) φ K := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U K : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    ⊢ Exists fun δ => And (GT.gt δ 0) (And (HasSubset.Subset (Metric.cthickening δ …
  -/
  obtain ⟨δ, hδ, hKδ⟩ := hK.exists_cthickening_subset_open hU hKU
  /-
    case intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U K : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    δ : Real
    hδ : LT.lt 0 δ
    hKδ : HasSubset.Subset (Metric.cthickening δ K) U
    ⊢ Exists fun δ => And (GT.gt δ 0) (And (HasSubset.Subset (Metric.cthickening δ …
  -/
  exact ⟨δ, hδ, hKδ, tendstoUniformlyOn_deriv_of_cthickening_subset hf hF hδ hK hU hKδ⟩
  /-
    🎉 no goals
  -/


/-- A locally uniform limit of holomorphic functions on an open domain of the complex plane is
holomorphic (the derivatives converge locally uniformly to that of the limit, which is proved
as `TendstoLocallyUniformlyOn.deriv`). -/
theorem _root_.TendstoLocallyUniformlyOn.differentiableOn [φ.NeBot]
    (hf : TendstoLocallyUniformlyOn F f φ U) (hF : ∀ᶠ n in φ, DifferentiableOn ℂ (F n) U)
    (hU : IsOpen U) : DifferentiableOn ℂ f U := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝¹ : CompleteSpace E
    inst✝ : φ.NeBot
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    ⊢ DifferentiableOn Complex f U
  -/
  rintro x hx
  /-
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝¹ : CompleteSpace E
    inst✝ : φ.NeBot
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    x : Complex
    hx : Membership.mem U x
    ⊢ DifferentiableWithinAt Complex f U x
  -/
  obtain ⟨K, ⟨hKx, hK⟩, hKU⟩ := (compact_basis_nhds x).mem_iff.mp (hU.mem_nhds hx)
  /-
    case intro.intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝¹ : CompleteSpace E
    inst✝ : φ.NeBot
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    x : Complex
    hx : Membership.mem U x
    K : Set Complex
    hKU : HasSubset.Subset K U
    hKx : Membership.mem (nhds x) K
    hK : IsCompact K
    ⊢ DifferentiableWithinAt Complex f U x
  -/
  obtain ⟨δ, _, _, h1⟩ := exists_cthickening_tendstoUniformlyOn hf hF hK hU hKU
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝¹ : CompleteSpace E
    inst✝ : φ.NeBot
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    x : Complex
    hx : Membership.mem U x
    K : Set Complex
    hKU : HasSubset.Subset K U
    hKx : Membership.mem (nhds x) K
    hK : IsCompact K
    δ : Real
    left✝¹ : GT.gt δ 0
    left✝ : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : TendstoUniformlyOn (Function.comp deriv F) (Complex.cderiv δ f) φ K
    ⊢ DifferentiableWithinAt Complex f U x
  -/
  have h2 : interior K ⊆ U := interior_subset.trans hKU
  have h3 : ∀ᶠ n in φ, DifferentiableOn ℂ (F n) (interior K) := by
    filter_upwards [hF] with n h using h.mono h2
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝¹ : CompleteSpace E
    inst✝ : φ.NeBot
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    x : Complex
    hx : Membership.mem U x
    K : Set Complex
    hKU : HasSubset.Subset K U
    hKx : Membership.mem (nhds x) K
    hK : IsCompact K
    δ : Real
    left✝¹ : GT.gt δ 0
    left✝ : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : TendstoUniformlyOn (Function.comp deriv F) (Complex.cderiv δ f) φ K
    h2 : HasSubset.Subset (interior K) U
    h3 : Filter.Eventually (fun n => DifferentiableOn Complex (F n) (interior K)) φ
    ⊢ DifferentiableWithinAt Complex f U x
  -/
  have h4 : TendstoLocallyUniformlyOn F f φ (interior K) := hf.mono h2
  have h5 : TendstoLocallyUniformlyOn (deriv ∘ F) (cderiv δ f) φ (interior K) :=
    h1.tendstoLocallyUniformlyOn.mono interior_subset
  have h6 : ∀ x ∈ interior K, HasDerivAt f (cderiv δ f x) x := fun x h =>
    hasDerivAt_of_tendsto_locally_uniformly_on' isOpen_interior h5 h3 (fun _ => h4.tendsto_at) h
  have h7 : DifferentiableOn ℂ f (interior K) := fun x hx =>
    (h6 x hx).differentiableAt.differentiableWithinAt
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝¹ : CompleteSpace E
    inst✝ : φ.NeBot
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    x : Complex
    hx : Membership.mem U x
    K : Set Complex
    hKU : HasSubset.Subset K U
    hKx : Membership.mem (nhds x) K
    hK : IsCompact K
    δ : Real
    left✝¹ : GT.gt δ 0
    left✝ : HasSubset.Subset (Metric.cthickening δ K) U
    h1 : TendstoUniformlyOn (Function.comp deriv F) (Complex.cderiv δ f) φ K
    h2 : HasSubset.Subset (interior K) U
    h3 : Filter.Eventually (fun n => DifferentiableOn Complex (F n) (interior K)) φ
    h4 : TendstoLocallyUniformlyOn F f φ (interior K)
    h5 : TendstoLocallyUniformlyOn (Function.comp deriv F) (Complex.cderiv δ f) φ  …
    h6 : ∀ (x : Complex), Membership.mem (interior K) x → HasDerivAt f (Complex.cd …
    h7 : DifferentiableOn Complex f (interior K)
    ⊢ DifferentiableWithinAt Complex f U x
  -/
  exact (h7.differentiableAt (interior_mem_nhds.mpr hKx)).differentiableWithinAt
  /-
    🎉 no goals
  -/


theorem _root_.TendstoLocallyUniformlyOn.deriv (hf : TendstoLocallyUniformlyOn F f φ U)
    (hF : ∀ᶠ n in φ, DifferentiableOn ℂ (F n) U) (hU : IsOpen U) :
    TendstoLocallyUniformlyOn (deriv ∘ F) (deriv f) φ U := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    ⊢ TendstoLocallyUniformlyOn (Function.comp _root_.deriv F) (_root_.deriv f) φ U
  -/
  rw [tendstoLocallyUniformlyOn_iff_forall_isCompact hU]
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    ⊢ ∀ (K : Set Complex), HasSubset.Subset K U → IsCompact K → TendstoUniformlyOn …
  -/
  rcases φ.eq_or_neBot with rfl | hne
    /-
      case inl
      E : Type u_1
      ι : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      U : Set Complex
      F : ι → Complex → E
      f : Complex → E
      inst✝ : CompleteSpace E
      hU : IsOpen U
      hf : TendstoLocallyUniformlyOn F f Bot.bot U
      hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) Bot.bot
      ⊢ ∀ (K : Set Complex), HasSubset.Subset K U → IsCompact K → TendstoUniformlyOn …
    -/
  · simp only [TendstoUniformlyOn, eventually_bot, imp_true_iff]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    hne : φ.NeBot
    ⊢ ∀ (K : Set Complex), HasSubset.Subset K U → IsCompact K → TendstoUniformlyOn …
  -/
  rintro K hKU hK
  /-
    case inr
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    hne : φ.NeBot
    K : Set Complex
    hKU : HasSubset.Subset K U
    hK : IsCompact K
    ⊢ TendstoUniformlyOn (Function.comp _root_.deriv F) (_root_.deriv f) φ K
  -/
  obtain ⟨δ, hδ, hK4, h⟩ := exists_cthickening_tendstoUniformlyOn hf hF hK hU hKU
  /-
    case inr.intro.intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    hne : φ.NeBot
    K : Set Complex
    hKU : HasSubset.Subset K U
    hK : IsCompact K
    δ : Real
    hδ : GT.gt δ 0
    hK4 : HasSubset.Subset (Metric.cthickening δ K) U
    h : TendstoUniformlyOn (Function.comp _root_.deriv F) (Complex.cderiv δ f) φ K
    ⊢ TendstoUniformlyOn (Function.comp _root_.deriv F) (_root_.deriv f) φ K
  -/
  refine h.congr_right fun z hz => cderiv_eq_deriv hU (hf.differentiableOn hF hU) hδ ?_
  /-
    case inr.intro.intro.intro
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    φ : Filter ι
    F : ι → Complex → E
    f : Complex → E
    inst✝ : CompleteSpace E
    hf : TendstoLocallyUniformlyOn F f φ U
    hF : Filter.Eventually (fun n => DifferentiableOn Complex (F n) U) φ
    hU : IsOpen U
    hne : φ.NeBot
    K : Set Complex
    hKU : HasSubset.Subset K U
    hK : IsCompact K
    δ : Real
    hδ : GT.gt δ 0
    hK4 : HasSubset.Subset (Metric.cthickening δ K) U
    h : TendstoUniformlyOn (Function.comp _root_.deriv F) (Complex.cderiv δ f) φ K
    z : Complex
    hz : Membership.mem K z
    ⊢ HasSubset.Subset (Metric.closedBall z δ) U
  -/
  exact (closedBall_subset_cthickening hz δ).trans hK4
  /-
    🎉 no goals
  -/


/-- If the terms in the sum `∑' (i : ι), F i` are uniformly bounded on `U` by a
summable function, and each term in the sum is differentiable on `U`, then so is the sum. -/
theorem differentiableOn_tsum_of_summable_norm {u : ι → ℝ} (hu : Summable u)
    (hf : ∀ i : ι, DifferentiableOn ℂ (F i) U) (hU : IsOpen U)
    (hF_le : ∀ (i : ι) (w : ℂ), w ∈ U → ‖F i w‖ ≤ u i) :
    DifferentiableOn ℂ (fun w : ℂ => ∑' i : ι, F i w) U := by
  classical
  have hc := (tendstoUniformlyOn_tsum hu hF_le).tendstoLocallyUniformlyOn
  refine hc.differentiableOn (Eventually.of_forall fun s => ?_) hU
  exact DifferentiableOn.sum fun i _ => hf i


/-- If the terms in the sum `∑' (i : ι), F i` are uniformly bounded on `U` by a
summable function, then the sum of `deriv F i` at a point in `U` is the derivative of the
sum. -/
theorem hasSum_deriv_of_summable_norm {u : ι → ℝ} (hu : Summable u)
    (hf : ∀ i : ι, DifferentiableOn ℂ (F i) U) (hU : IsOpen U)
    (hF_le : ∀ (i : ι) (w : ℂ), w ∈ U → ‖F i w‖ ≤ u i) (hz : z ∈ U) :
    HasSum (fun i : ι => deriv (F i) z) (deriv (fun w : ℂ => ∑' i : ι, F i w) z) := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    z : Complex
    F : ι → Complex → E
    inst✝ : CompleteSpace E
    u : ι → Real
    hu : Summable u
    hf : ∀ (i : ι), DifferentiableOn Complex (F i) U
    hU : IsOpen U
    hF_le : ∀ (i : ι) (w : Complex), Membership.mem U w → LE.le (Norm.norm (F i w) …
    hz : Membership.mem U z
    ⊢ HasSum (fun i => deriv (F i) z) (deriv (fun w => tsum fun i => F i w) z)
  -/
  rw [HasSum]
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    z : Complex
    F : ι → Complex → E
    inst✝ : CompleteSpace E
    u : ι → Real
    hu : Summable u
    hf : ∀ (i : ι), DifferentiableOn Complex (F i) U
    hU : IsOpen U
    hF_le : ∀ (i : ι) (w : Complex), Membership.mem U w → LE.le (Norm.norm (F i w) …
    hz : Membership.mem U z
    ⊢ Filter.Tendsto (fun s => s.sum fun b => deriv (F b) z) Filter.atTop (nhds (d …
  -/
  have hc := (tendstoUniformlyOn_tsum hu hF_le).tendstoLocallyUniformlyOn
  convert (hc.deriv (Eventually.of_forall fun s =>
    DifferentiableOn.sum fun i _ => hf i) hU).tendsto_at hz using 1
  /-
    case h.e'_3
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    z : Complex
    F : ι → Complex → E
    inst✝ : CompleteSpace E
    u : ι → Real
    hu : Summable u
    hf : ∀ (i : ι), DifferentiableOn Complex (F i) U
    hU : IsOpen U
    hF_le : ∀ (i : ι) (w : Complex), Membership.mem U w → LE.le (Norm.norm (F i w) …
    hz : Membership.mem U z
    hc : TendstoLocallyUniformlyOn (fun t x => t.sum fun n => F n x) (fun x => tsu …
    ⊢ Eq (fun s => s.sum fun b => deriv (F b) z) fun i => Function.comp deriv (fun …
  -/
  ext1 s
  /-
    case h.e'_3.h
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    U : Set Complex
    z : Complex
    F : ι → Complex → E
    inst✝ : CompleteSpace E
    u : ι → Real
    hu : Summable u
    hf : ∀ (i : ι), DifferentiableOn Complex (F i) U
    hU : IsOpen U
    hF_le : ∀ (i : ι) (w : Complex), Membership.mem U w → LE.le (Norm.norm (F i w) …
    hz : Membership.mem U z
    hc : TendstoLocallyUniformlyOn (fun t x => t.sum fun n => F n x) (fun x => tsu …
    s : Finset ι
    ⊢ Eq (s.sum fun b => deriv (F b) z) (Function.comp deriv (fun t x => t.sum fun …
  -/
  exact (deriv_sum fun i _ => (hf i).differentiableAt (hU.mem_nhds hz)).symm
  /-
    🎉 no goals
  -/


/-- The logarithmic derivative of a sequence of functions converging locally uniformly to a
function is the logarithmic derivative of the limit function. -/
theorem logDeriv_tendsto {ι : Type*} {p : Filter ι} (f : ι → ℂ → ℂ) (g : ℂ → ℂ)
    {s : Set ℂ} (hs : IsOpen s) (x : s) (hF : TendstoLocallyUniformlyOn f g p s)
    (hf : ∀ᶠ n : ι in p, DifferentiableOn ℂ (f n) s) (hg : g x ≠ 0) :
    Tendsto (fun n : ι => logDeriv (f n) x) p (𝓝 ((logDeriv g) x)) := by
  /-
    ι : Type u_3
    p : Filter ι
    f : ι → Complex → Complex
    g : Complex → Complex
    s : Set Complex
    hs : IsOpen s
    x : ↑s
    hF : TendstoLocallyUniformlyOn f g p s
    hf : Filter.Eventually (fun n => DifferentiableOn Complex (f n) s) p
    hg : Ne (g ↑x) 0
    ⊢ Filter.Tendsto (fun n => logDeriv (f n) ↑x) p (nhds (logDeriv g ↑x))
  -/
  simp_rw [logDeriv]
  /-
    ι : Type u_3
    p : Filter ι
    f : ι → Complex → Complex
    g : Complex → Complex
    s : Set Complex
    hs : IsOpen s
    x : ↑s
    hF : TendstoLocallyUniformlyOn f g p s
    hf : Filter.Eventually (fun n => DifferentiableOn Complex (f n) s) p
    hg : Ne (g ↑x) 0
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (deriv (f n)) (f n) ↑x) p (nhds (HDiv.hDi …
  -/
  apply Tendsto.div ((hF.deriv hf hs).tendsto_at x.2) (hF.tendsto_at x.2) hg
  /-
    🎉 no goals
  -/


