variable (E) in
theorem not_differentiableAt_norm_zero [Nontrivial E] :
    ¬DifferentiableAt ℝ (‖·‖) (0 : E) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    ⊢ Not (DifferentiableAt Real (fun x => Norm.norm x) 0)
  -/
  obtain ⟨x, hx⟩ := NormedSpace.exists_lt_norm ℝ E 0
  /-
    case intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    hx : LT.lt 0 (Norm.norm x)
    ⊢ Not (DifferentiableAt Real (fun x => Norm.norm x) 0)
  -/
  intro h
  /-
    case intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    hx : LT.lt 0 (Norm.norm x)
    h : DifferentiableAt Real (fun x => Norm.norm x) 0
    ⊢ False
  -/
  have : DifferentiableAt ℝ (fun t : ℝ ↦ ‖t • x‖) 0 := DifferentiableAt.comp _ (by simpa) (by simp)
  have : DifferentiableAt ℝ (|·|) (0 : ℝ) := by
    simp_rw [norm_smul, norm_eq_abs] at this
    have aux : abs = fun t ↦ (1 / ‖x‖) * (|t| * ‖x‖) := by field_simp
    rw [aux]
    exact this.const_mul _
  /-
    case intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : Nontrivial E
    x : E
    hx : LT.lt 0 (Norm.norm x)
    h : DifferentiableAt Real (fun x => Norm.norm x) 0
    this✝ : DifferentiableAt Real (fun t => Norm.norm (HSMul.hSMul t x)) 0
    this : DifferentiableAt Real (fun x => abs x) 0
    ⊢ False
  -/
  exact not_differentiableAt_abs_zero this
  /-
    🎉 no goals
  -/


theorem ContDiffAt.contDiffAt_norm_smul (ht : t ≠ 0) (h : ContDiffAt ℝ n (‖·‖) x) :
    ContDiffAt ℝ n (‖·‖) (t • x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
  -/
  have h1 : ContDiffAt ℝ n (fun y ↦ t⁻¹ • y) (t • x) := (contDiff_const_smul t⁻¹).contDiffAt
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    h1 : ContDiffAt Real n (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul t x)
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
  -/
  have h2 : ContDiffAt ℝ n (fun y ↦ |t| * ‖y‖) x := h.const_smul |t|
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    h1 : ContDiffAt Real n (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul t x)
    h2 : ContDiffAt Real n (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) x
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
  -/
  conv at h2 => enter [4]; rw [← one_smul ℝ x, ← inv_mul_cancel₀ ht, mul_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    h1 : ContDiffAt Real n (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul t x)
    h2 : ContDiffAt Real n (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
  -/
  convert h2.comp (t • x) h1 using 1
  /-
    case h.e'_10
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    h1 : ContDiffAt Real n (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul t x)
    h2 : ContDiffAt Real n (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    ⊢ Eq (fun x => Norm.norm x) (Function.comp (fun y => HMul.hMul (_root_.abs t)  …
  -/
  ext y
  /-
    case h.e'_10.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    h1 : ContDiffAt Real n (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul t x)
    h2 : ContDiffAt Real n (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    y : E
    ⊢ Eq (Norm.norm y) (Function.comp (fun y => HMul.hMul (_root_.abs t) (Norm.nor …
  -/
  simp only [Function.comp_apply]
  /-
    case h.e'_10.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    ht : Ne t 0
    h : ContDiffAt Real n (fun x => Norm.norm x) x
    h1 : ContDiffAt Real n (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul t x)
    h2 : ContDiffAt Real n (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    y : E
    ⊢ Eq (Norm.norm y) (HMul.hMul (_root_.abs t) (Norm.norm (HSMul.hSMul (Inv.inv  …
  -/
  rw [norm_smul, ← mul_assoc, norm_eq_abs, ← abs_mul, mul_inv_cancel₀ ht, abs_one, one_mul]
  /-
    🎉 no goals
  -/


theorem contDiffAt_norm_smul_iff (ht : t ≠ 0) :
    ContDiffAt ℝ n (‖·‖) x ↔ ContDiffAt ℝ n (‖·‖) (t • x) where
  mp h := h.contDiffAt_norm_smul ht
  mpr hd := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      n : WithTop ENat
      x : E
      t : Real
      ht : Ne t 0
      hd : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
      ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
    -/
    convert hd.contDiffAt_norm_smul (inv_ne_zero ht)
    /-
      case h.e'_11
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      n : WithTop ENat
      x : E
      t : Real
      ht : Ne t 0
      hd : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
      ⊢ Eq x (HSMul.hSMul (Inv.inv t) (HSMul.hSMul t x))
    -/
    rw [smul_smul, inv_mul_cancel₀ ht, one_smul]
    /-
      🎉 no goals
    -/


theorem ContDiffAt.contDiffAt_norm_of_smul (h : ContDiffAt ℝ n (‖·‖) (t • x)) :
    ContDiffAt ℝ n (‖·‖) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
  -/
  rcases eq_bot_or_bot_lt n with rfl | hn
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      h : ContDiffAt Real Bot.bot (fun x => Norm.norm x) (HSMul.hSMul t x)
      ⊢ ContDiffAt Real Bot.bot (fun x => Norm.norm x) x
    -/
  · apply contDiffAt_zero.2
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      h : ContDiffAt Real Bot.bot (fun x => Norm.norm x) (HSMul.hSMul t x)
      ⊢ Exists fun u => And (Membership.mem (nhds x) u) (ContinuousOn (fun x => Norm …
    -/
    exact ⟨univ, univ_mem, continuous_norm.continuousOn⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
    hn : LT.lt Bot.bot n
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
  -/
  replace hn : 1 ≤ n := ENat.add_one_natCast_le_withTop_of_lt hn
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    n : WithTop ENat
    x : E
    t : Real
    h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
    hn : LE.le 1 n
    ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
  -/
  obtain rfl | ht := eq_or_ne t 0
    /-
      case inr.inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      n : WithTop ENat
      x : E
      hn : LE.le 1 n
      h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul 0 x)
      ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
    -/
  · by_cases hE : Nontrivial E
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        n : WithTop ENat
        x : E
        hn : LE.le 1 n
        h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Nontrivial E
        ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
      -/
    · rw [zero_smul] at h
      exact (mt (ContDiffAt.differentiableAt · (mod_cast hn)))
        (not_differentiableAt_norm_zero E) h |>.elim
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        n : WithTop ENat
        x : E
        hn : LE.le 1 n
        h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Not (Nontrivial E)
        ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
      -/
    · rw [not_nontrivial_iff_subsingleton] at hE
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        n : WithTop ENat
        x : E
        hn : LE.le 1 n
        h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Subsingleton E
        ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
      -/
      rw [eq_const_of_subsingleton (‖·‖) 0]
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        n : WithTop ENat
        x : E
        hn : LE.le 1 n
        h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Subsingleton E
        ⊢ ContDiffAt Real n (Function.const E (Norm.norm 0)) x
      -/
      exact contDiffAt_const
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      n : WithTop ENat
      x : E
      t : Real
      h : ContDiffAt Real n (fun x => Norm.norm x) (HSMul.hSMul t x)
      hn : LE.le 1 n
      ht : Ne t 0
      ⊢ ContDiffAt Real n (fun x => Norm.norm x) x
    -/
  · exact contDiffAt_norm_smul_iff ht |>.2 h
    /-
      🎉 no goals
    -/


theorem HasStrictFDerivAt.hasStrictFDerivAt_norm_smul
    (ht : t ≠ 0) (h : HasStrictFDerivAt (‖·‖) f x) :
    HasStrictFDerivAt (‖·‖) ((SignType.sign t : ℝ) • f) (t • x) := by
  have h1 : HasStrictFDerivAt (fun y ↦ t⁻¹ • y) (t⁻¹ • ContinuousLinearMap.id ℝ E) (t • x) :=
    hasStrictFDerivAt_id (t • x) |>.const_smul t⁻¹
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    h1 : HasStrictFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv. …
    ⊢ HasStrictFDerivAt (fun x => Norm.norm x) (HSMul.hSMul (↑(SignType.sign t)) f …
  -/
  have h2 : HasStrictFDerivAt (fun y ↦ |t| * ‖y‖) (|t| • f) x := h.const_smul |t|
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    h1 : HasStrictFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv. …
    h2 : HasStrictFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    ⊢ HasStrictFDerivAt (fun x => Norm.norm x) (HSMul.hSMul (↑(SignType.sign t)) f …
  -/
  conv at h2 => enter [3]; rw [← one_smul ℝ x, ← inv_mul_cancel₀ ht, mul_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    h1 : HasStrictFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv. …
    h2 : HasStrictFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    ⊢ HasStrictFDerivAt (fun x => Norm.norm x) (HSMul.hSMul (↑(SignType.sign t)) f …
  -/
  convert h2.comp (t • x) h1 with y
    /-
      case h.e'_11.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : ContinuousLinearMap (RingHom.id Real) E Real
      x : E
      t : Real
      ht : Ne t 0
      h : HasStrictFDerivAt (fun x => Norm.norm x) f x
      h1 : HasStrictFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv. …
      h2 : HasStrictFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
      y : E
      ⊢ Eq (Norm.norm y) (HMul.hMul (_root_.abs t) (Norm.norm (HSMul.hSMul (Inv.inv  …
    -/
  · rw [norm_smul, ← mul_assoc, norm_eq_abs, ← abs_mul, mul_inv_cancel₀ ht, abs_one, one_mul]
    /-
      🎉 no goals
    -/
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    h1 : HasStrictFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv. …
    h2 : HasStrictFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    ⊢ Eq (HSMul.hSMul (↑(SignType.sign t)) f) ((HSMul.hSMul (_root_.abs t) f).comp …
  -/
  ext y
  simp only [coe_smul', Pi.smul_apply, smul_eq_mul, comp_smulₛₗ, map_inv₀, RingHom.id_apply,
    comp_id]
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    h1 : HasStrictFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv. …
    h2 : HasStrictFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMu …
    y : E
    ⊢ Eq (HMul.hMul (↑(SignType.sign t)) (f y)) (HMul.hMul (Inv.inv t) (HMul.hMul  …
  -/
  rw [eq_inv_mul_iff_mul_eq₀ ht, ← mul_assoc, self_mul_sign]
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.hasStrictDerivAt_norm_smul_neg
    (ht : t < 0) (h : HasStrictFDerivAt (‖·‖) f x) :
    HasStrictFDerivAt (‖·‖) (-f) (t • x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : LT.lt t 0
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    ⊢ HasStrictFDerivAt (fun x => Norm.norm x) (Neg.neg f) (HSMul.hSMul t x)
  -/
  simpa [ht] using h.hasStrictFDerivAt_norm_smul ht.ne
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.hasStrictDerivAt_norm_smul_pos
    (ht : 0 < t) (h : HasStrictFDerivAt (‖·‖) f x) :
    HasStrictFDerivAt (‖·‖) f (t • x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : LT.lt 0 t
    h : HasStrictFDerivAt (fun x => Norm.norm x) f x
    ⊢ HasStrictFDerivAt (fun x => Norm.norm x) f (HSMul.hSMul t x)
  -/
  simpa [ht] using h.hasStrictFDerivAt_norm_smul ht.ne'
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.hasFDerivAt_norm_smul
    (ht : t ≠ 0) (h : HasFDerivAt (‖·‖) f x) :
    HasFDerivAt (‖·‖) ((SignType.sign t : ℝ) • f) (t • x) := by
  have h1 : HasFDerivAt (fun y ↦ t⁻¹ • y) (t⁻¹ • ContinuousLinearMap.id ℝ E) (t • x) :=
    hasFDerivAt_id (t • x) |>.const_smul t⁻¹
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasFDerivAt (fun x => Norm.norm x) f x
    h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
    ⊢ HasFDerivAt (fun x => Norm.norm x) (HSMul.hSMul (↑(SignType.sign t)) f) (HSM …
  -/
  have h2 : HasFDerivAt (fun y ↦ |t| * ‖y‖) (|t| • f) x := h.const_smul |t|
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasFDerivAt (fun x => Norm.norm x) f x
    h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
    h2 : HasFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMul.hSMu …
    ⊢ HasFDerivAt (fun x => Norm.norm x) (HSMul.hSMul (↑(SignType.sign t)) f) (HSM …
  -/
  conv at h2 => enter [3]; rw [← one_smul ℝ x, ← inv_mul_cancel₀ ht, mul_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : Ne t 0
    h : HasFDerivAt (fun x => Norm.norm x) f x
    h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
    h2 : HasFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMul.hSMu …
    ⊢ HasFDerivAt (fun x => Norm.norm x) (HSMul.hSMul (↑(SignType.sign t)) f) (HSM …
  -/
  convert h2.comp (t • x) h1 using 2 with y
    /-
      case h.e'_11.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : ContinuousLinearMap (RingHom.id Real) E Real
      x : E
      t : Real
      ht : Ne t 0
      h : HasFDerivAt (fun x => Norm.norm x) f x
      h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
      h2 : HasFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMul.hSMu …
      y : E
      ⊢ Eq (Norm.norm y) (Function.comp (fun y => HMul.hMul (_root_.abs t) (Norm.nor …
    -/
  · simp only [Function.comp_apply]
    /-
      case h.e'_11.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : ContinuousLinearMap (RingHom.id Real) E Real
      x : E
      t : Real
      ht : Ne t 0
      h : HasFDerivAt (fun x => Norm.norm x) f x
      h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
      h2 : HasFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMul.hSMu …
      y : E
      ⊢ Eq (Norm.norm y) (HMul.hMul (_root_.abs t) (Norm.norm (HSMul.hSMul (Inv.inv  …
    -/
    rw [norm_smul, ← mul_assoc, norm_eq_abs, ← abs_mul, mul_inv_cancel₀ ht, abs_one, one_mul]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_12
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : ContinuousLinearMap (RingHom.id Real) E Real
      x : E
      t : Real
      ht : Ne t 0
      h : HasFDerivAt (fun x => Norm.norm x) f x
      h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
      h2 : HasFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMul.hSMu …
      ⊢ Eq (HSMul.hSMul (↑(SignType.sign t)) f) ((HSMul.hSMul (_root_.abs t) f).comp …
    -/
  · ext y
    simp only [coe_smul', Pi.smul_apply, smul_eq_mul, comp_smulₛₗ, map_inv₀, RingHom.id_apply,
      comp_id]
    /-
      case h.e'_12.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : ContinuousLinearMap (RingHom.id Real) E Real
      x : E
      t : Real
      ht : Ne t 0
      h : HasFDerivAt (fun x => Norm.norm x) f x
      h1 : HasFDerivAt (fun y => HSMul.hSMul (Inv.inv t) y) (HSMul.hSMul (Inv.inv t) …
      h2 : HasFDerivAt (fun y => HMul.hMul (_root_.abs t) (Norm.norm y)) (HSMul.hSMu …
      y : E
      ⊢ Eq (HMul.hMul (↑(SignType.sign t)) (f y)) (HMul.hMul (Inv.inv t) (HMul.hMul  …
    -/
    rw [eq_inv_mul_iff_mul_eq₀ ht, ← mul_assoc, self_mul_sign]
    /-
      🎉 no goals
    -/


theorem HasFDerivAt.hasFDerivAt_norm_smul_neg
    (ht : t < 0) (h : HasFDerivAt (‖·‖) f x) :
    HasFDerivAt (‖·‖) (-f) (t • x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : LT.lt t 0
    h : HasFDerivAt (fun x => Norm.norm x) f x
    ⊢ HasFDerivAt (fun x => Norm.norm x) (Neg.neg f) (HSMul.hSMul t x)
  -/
  simpa [ht] using h.hasFDerivAt_norm_smul ht.ne
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.hasFDerivAt_norm_smul_pos
    (ht : 0 < t) (h : HasFDerivAt (‖·‖) f x) :
    HasFDerivAt (‖·‖) f (t • x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    t : Real
    ht : LT.lt 0 t
    h : HasFDerivAt (fun x => Norm.norm x) f x
    ⊢ HasFDerivAt (fun x => Norm.norm x) f (HSMul.hSMul t x)
  -/
  simpa [ht] using h.hasFDerivAt_norm_smul ht.ne'
  /-
    🎉 no goals
  -/


theorem differentiableAt_norm_smul (ht : t ≠ 0) :
    DifferentiableAt ℝ (‖·‖) x ↔ DifferentiableAt ℝ (‖·‖) (t • x) where
  mp hd := (hd.hasFDerivAt.hasFDerivAt_norm_smul ht).differentiableAt
  mpr hd := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      ht : Ne t 0
      hd : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul t x)
      ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
    -/
    convert (hd.hasFDerivAt.hasFDerivAt_norm_smul (inv_ne_zero ht)).differentiableAt
    /-
      case h.e'_12
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      ht : Ne t 0
      hd : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul t x)
      ⊢ Eq x (HSMul.hSMul (Inv.inv t) (HSMul.hSMul t x))
    -/
    rw [smul_smul, inv_mul_cancel₀ ht, one_smul]
    /-
      🎉 no goals
    -/


theorem DifferentiableAt.differentiableAt_norm_of_smul (h : DifferentiableAt ℝ (‖·‖) (t • x)) :
    DifferentiableAt ℝ (‖·‖) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    t : Real
    h : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul t x)
    ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
  -/
  obtain rfl | ht := eq_or_ne t 0
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      h : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul 0 x)
      ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
    -/
  · by_cases hE : Nontrivial E
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Nontrivial E
        ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
      -/
    · rw [zero_smul] at h
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) 0
        hE : Nontrivial E
        ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
      -/
      exact not_differentiableAt_norm_zero E h |>.elim
      /-
        🎉 no goals
      -/
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Not (Nontrivial E)
        ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
      -/
    · rw [not_nontrivial_iff_subsingleton] at hE
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul 0 x)
        hE : Subsingleton E
        ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
      -/
      exact (hasFDerivAt_of_subsingleton _ _).differentiableAt
      /-
        🎉 no goals
      -/
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      h : DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul t x)
      ht : Ne t 0
      ⊢ DifferentiableAt Real (fun x => Norm.norm x) x
    -/
  · exact differentiableAt_norm_smul ht |>.2 h
    /-
      🎉 no goals
    -/


theorem DifferentiableAt.fderiv_norm_self {x : E} (h : DifferentiableAt ℝ (‖·‖) x) :
    fderiv ℝ (‖·‖) x x = ‖x‖ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    h : DifferentiableAt Real (fun x => Norm.norm x) x
    ⊢ Eq ((fderiv Real (fun x => Norm.norm x) x) x) (Norm.norm x)
  -/
  rw [← h.lineDeriv_eq_fderiv, lineDeriv]
  have this (t : ℝ) : ‖x + t • x‖ = |1 + t| * ‖x‖ := by
    rw [← norm_eq_abs, ← norm_smul, add_smul, one_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    h : DifferentiableAt Real (fun x => Norm.norm x) x
    this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
    ⊢ Eq (deriv (fun t => Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) 0) (Norm.norm …
  -/
  simp_rw [this]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    h : DifferentiableAt Real (fun x => Norm.norm x) x
    this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
    ⊢ Eq (deriv (fun t => HMul.hMul (_root_.abs (HAdd.hAdd 1 t)) (Norm.norm x)) 0) …
  -/
  rw [deriv_mul_const]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      h : DifferentiableAt Real (fun x => Norm.norm x) x
      this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
      ⊢ Eq (HMul.hMul (deriv (fun t => _root_.abs (HAdd.hAdd 1 t)) 0) (Norm.norm x)) …
    -/
  · conv_lhs => enter [1, 1]; change _root_.abs ∘ (fun t ↦ 1 + t)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      h : DifferentiableAt Real (fun x => Norm.norm x) x
      this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
      ⊢ Eq (HMul.hMul (deriv (Function.comp _root_.abs fun t => HAdd.hAdd 1 t) 0) (N …
    -/
    rw [deriv_comp, deriv_abs, deriv_const_add]
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) x
        this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
        ⊢ Eq (HMul.hMul (HMul.hMul (↑(SignType.sign (HAdd.hAdd 1 0))) (deriv (fun t => …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case hh₂
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) x
        this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
        ⊢ DifferentiableAt Real _root_.abs (HAdd.hAdd 1 0)
      -/
    · exact differentiableAt_abs (by norm_num)
      /-
        🎉 no goals
      -/
      /-
        case hh
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        h : DifferentiableAt Real (fun x => Norm.norm x) x
        this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
        ⊢ DifferentiableAt Real (fun t => HAdd.hAdd 1 t) 0
      -/
    · exact differentiableAt_id.const_add _
      /-
        🎉 no goals
      -/
    /-
      case hc
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      h : DifferentiableAt Real (fun x => Norm.norm x) x
      this : ∀ (t : Real), Eq (Norm.norm (HAdd.hAdd x (HSMul.hSMul t x))) (HMul.hMul …
      ⊢ DifferentiableAt Real (fun t => _root_.abs (HAdd.hAdd 1 t)) 0
    -/
  · exact (differentiableAt_abs (by norm_num)).comp _ (differentiableAt_id.const_add _)
    /-
      🎉 no goals
    -/


variable (x t) in
theorem fderiv_norm_smul :
    fderiv ℝ (‖·‖) (t • x) = (SignType.sign t : ℝ) • (fderiv ℝ (‖·‖) x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    t : Real
    ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
  -/
  by_cases hE : Nontrivial E
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      hE : Nontrivial E
      ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
    -/
  · by_cases hd : DifferentiableAt ℝ (‖·‖) x
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        t : Real
        hE : Nontrivial E
        hd : DifferentiableAt Real (fun x => Norm.norm x) x
        ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
      -/
    · obtain rfl | ht := eq_or_ne t 0
        /-
          case pos.inl
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          x : E
          hE : Nontrivial E
          hd : DifferentiableAt Real (fun x => Norm.norm x) x
          ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul 0 x)) (HSMul.hSMul (↑(Si …
        -/
      · simp only [zero_smul, _root_.sign_zero, SignType.coe_zero]
        /-
          case pos.inl
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          x : E
          hE : Nontrivial E
          hd : DifferentiableAt Real (fun x => Norm.norm x) x
          ⊢ Eq (fderiv Real (fun x => Norm.norm x) 0) 0
        -/
        exact fderiv_zero_of_not_differentiableAt <| not_differentiableAt_norm_zero E
        /-
          🎉 no goals
        -/
        /-
          case pos.inr
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          x : E
          t : Real
          hE : Nontrivial E
          hd : DifferentiableAt Real (fun x => Norm.norm x) x
          ht : Ne t 0
          ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
        -/
      · rw [(hd.hasFDerivAt.hasFDerivAt_norm_smul ht).fderiv]
        /-
          🎉 no goals
        -/
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        x : E
        t : Real
        hE : Nontrivial E
        hd : Not (DifferentiableAt Real (fun x => Norm.norm x) x)
        ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
      -/
    · rw [fderiv_zero_of_not_differentiableAt hd, fderiv_zero_of_not_differentiableAt]
        /-
          case neg
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          x : E
          t : Real
          hE : Nontrivial E
          hd : Not (DifferentiableAt Real (fun x => Norm.norm x) x)
          ⊢ Eq 0 (HSMul.hSMul (↑(SignType.sign t)) 0)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case neg
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          x : E
          t : Real
          hE : Nontrivial E
          hd : Not (DifferentiableAt Real (fun x => Norm.norm x) x)
          ⊢ Not (DifferentiableAt Real (fun x => Norm.norm x) (HSMul.hSMul t x))
        -/
      · exact mt DifferentiableAt.differentiableAt_norm_of_smul hd
        /-
          🎉 no goals
        -/
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      hE : Not (Nontrivial E)
      ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
    -/
  · rw [not_nontrivial_iff_subsingleton] at hE
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      t : Real
      hE : Subsingleton E
      ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (HSMul.hSMul (↑(Si …
    -/
    simp_rw [(hasFDerivAt_of_subsingleton _ _).fderiv, smul_zero]
    /-
      🎉 no goals
    -/


theorem fderiv_norm_smul_pos (ht : 0 < t) :
    fderiv ℝ (‖·‖) (t • x) = fderiv ℝ (‖·‖) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    t : Real
    ht : LT.lt 0 t
    ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (fderiv Real (fun  …
  -/
  simp [fderiv_norm_smul, ht]
  /-
    🎉 no goals
  -/


theorem fderiv_norm_smul_neg (ht : t < 0) :
    fderiv ℝ (‖·‖) (t • x) = -fderiv ℝ (‖·‖) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    t : Real
    ht : LT.lt t 0
    ⊢ Eq (fderiv Real (fun x => Norm.norm x) (HSMul.hSMul t x)) (Neg.neg (fderiv R …
  -/
  simp [fderiv_norm_smul, ht]
  /-
    🎉 no goals
  -/


theorem norm_fderiv_norm [Nontrivial E] (h : DifferentiableAt ℝ (‖·‖) x) :
    ‖fderiv ℝ (‖·‖) x‖ = 1 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    x : E
    inst✝ : Nontrivial E
    h : DifferentiableAt Real (fun x => Norm.norm x) x
    ⊢ Eq (Norm.norm (fderiv Real (fun x => Norm.norm x) x)) 1
  -/
  have : x ≠ 0 := fun hx ↦ not_differentiableAt_norm_zero E (hx ▸ h)
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    x : E
    inst✝ : Nontrivial E
    h : DifferentiableAt Real (fun x => Norm.norm x) x
    this : Ne x 0
    ⊢ Eq (Norm.norm (fderiv Real (fun x => Norm.norm x) x)) 1
  -/
  refine le_antisymm (NNReal.coe_one ▸ norm_fderiv_le_of_lipschitz ℝ lipschitzWith_one_norm) ?_
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    x : E
    inst✝ : Nontrivial E
    h : DifferentiableAt Real (fun x => Norm.norm x) x
    this : Ne x 0
    ⊢ LE.le 1 (Norm.norm (fderiv Real (fun x => Norm.norm x) x))
  -/
  apply le_of_mul_le_mul_right _ (norm_pos_iff.2 this)
  calc
    1 * ‖x‖ = fderiv ℝ (‖·‖) x x := by rw [one_mul, h.fderiv_norm_self]
    _ ≤ ‖fderiv ℝ (‖·‖) x x‖ := le_norm_self _
    _ ≤ ‖fderiv ℝ (‖·‖) x‖ * ‖x‖ := le_opNorm _ _

