theorem hasFDerivAt_norm_rpow (x : E) {p : ℝ} (hp : 1 < p) :
    HasFDerivAt (fun x : E ↦ ‖x‖ ^ p) ((p * ‖x‖ ^ (p - 2)) • innerSL ℝ x) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    x : E
    p : Real
    hp : LT.lt 1 p
    ⊢ HasFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HMul.hMul p ( …
  -/
  by_cases hx : x = 0
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Eq x 0
      ⊢ HasFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HMul.hMul p ( …
    -/
  · simp only [hx, norm_zero, map_zero, smul_zero]
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Eq x 0
      ⊢ HasFDerivAt (fun x => HPow.hPow (Norm.norm x) p) 0 0
    -/
    have h2p : 0 < p - 1 := sub_pos.mpr hp
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Eq x 0
      h2p : LT.lt 0 (HSub.hSub p 1)
      ⊢ HasFDerivAt (fun x => HPow.hPow (Norm.norm x) p) 0 0
    -/
    rw [HasFDerivAt, hasFDerivAtFilter_iff_isLittleO]
    calc (fun x : E ↦ ‖x‖ ^ p - ‖(0 : E)‖ ^ p - 0)
        = (fun x : E ↦ ‖x‖ ^ p) := by simp [zero_lt_one.trans hp |>.ne']
      _ = (fun x : E ↦ ‖x‖ * ‖x‖ ^ (p - 1)) := by
          ext x
          rw [← rpow_one_add' (norm_nonneg x) (by positivity)]
          ring_nf
      _ =o[𝓝 0] (fun x : E ↦ ‖x‖ * 1) := by
        refine (isBigO_refl _ _).mul_isLittleO <| (isLittleO_const_iff <| by norm_num).mpr ?_
        convert continuousAt_id.norm.rpow_const (.inr h2p.le) |>.tendsto
        simp [h2p.ne']
      _ =O[𝓝 0] (fun (x : E) ↦ x - 0) := by
        simp_rw [mul_one, isBigO_norm_left (f' := fun x ↦ x), sub_zero, isBigO_refl]
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Not (Eq x 0)
      ⊢ HasFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HMul.hMul p ( …
    -/
  · apply HasStrictFDerivAt.hasFDerivAt
    /-
      case neg.hf
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Not (Eq x 0)
      ⊢ HasStrictFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HMul.hM …
    -/
    convert (hasStrictFDerivAt_norm_sq x).rpow_const (p := p / 2) (by simp [hx]) using 0
    /-
      case a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Not (Eq x 0)
      ⊢ Iff (HasStrictFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HM …
    -/
    simp_rw [← Real.rpow_natCast_mul (norm_nonneg _), ← Nat.cast_smul_eq_nsmul ℝ, smul_smul]
    /-
      case a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Not (Eq x 0)
      ⊢ Iff (HasStrictFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HM …
    -/
    ring_nf -- doesn't close the goal?
    /-
      case a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Not (Eq x 0)
      ⊢ Iff (HasStrictFDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HSMul.hSMul (HM …
    -/
    congr! 2
    /-
      case a.a.h.e'_12.h.e'_5
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      x : E
      p : Real
      hp : LT.lt 1 p
      hx : Not (Eq x 0)
      ⊢ Eq (HMul.hMul p (HPow.hPow (Norm.norm x) (HAdd.hAdd (-2) p))) (HMul.hMul (HM …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem differentiable_norm_rpow {p : ℝ} (hp : 1 < p) :
    Differentiable ℝ (fun x : E ↦ ‖x‖ ^ p) :=
  fun x ↦ hasFDerivAt_norm_rpow x hp |>.differentiableAt


theorem hasDerivAt_norm_rpow (x : ℝ) {p : ℝ} (hp : 1 < p) :
    HasDerivAt (fun x : ℝ ↦ ‖x‖ ^ p) (p * ‖x‖ ^ (p - 2) * x) x := by
  /-
    x p : Real
    hp : LT.lt 1 p
    ⊢ HasDerivAt (fun x => HPow.hPow (Norm.norm x) p) (HMul.hMul (HMul.hMul p (HPo …
  -/
  convert hasFDerivAt_norm_rpow x hp |>.hasDerivAt using 1; simp
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem hasDerivAt_abs_rpow (x : ℝ) {p : ℝ} (hp : 1 < p) :
    HasDerivAt (fun x : ℝ ↦ |x| ^ p) (p * |x| ^ (p - 2) * x) x := by
  /-
    x p : Real
    hp : LT.lt 1 p
    ⊢ HasDerivAt (fun x => HPow.hPow (abs x) p) (HMul.hMul (HMul.hMul p (HPow.hPow …
  -/
  simpa using hasDerivAt_norm_rpow x hp
  /-
    🎉 no goals
  -/


theorem fderiv_norm_rpow (x : E) {p : ℝ} (hp : 1 < p) :
    fderiv ℝ (fun x ↦ ‖x‖ ^ p) x = (p * ‖x‖ ^ (p - 2)) • innerSL ℝ x :=
  hasFDerivAt_norm_rpow x hp |>.fderiv


theorem Differentiable.fderiv_norm_rpow {f : F → E} (hf : Differentiable ℝ f)
    {x : F} {p : ℝ} (hp : 1 < p) :
    fderiv ℝ (fun x ↦ ‖f x‖ ^ p) x =
    (p * ‖f x‖ ^ (p - 2)) • (innerSL ℝ (f x)).comp (fderiv ℝ f x) :=
  hasFDerivAt_norm_rpow (f x) hp |>.comp x (hf x).hasFDerivAt |>.fderiv


theorem norm_fderiv_norm_rpow_le {f : F → E} (hf : Differentiable ℝ f) {x : F}
    {p : ℝ} (hp : 1 < p) :
    ‖fderiv ℝ (fun x ↦ ‖f x‖ ^ p) x‖ ≤ p * ‖f x‖ ^ (p - 1) * ‖fderiv ℝ f x‖ := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : F → E
    hf : Differentiable Real f
    x : F
    p : Real
    hp : LT.lt 1 p
    ⊢ LE.le (Norm.norm (fderiv Real (fun x => HPow.hPow (Norm.norm (f x)) p) x)) ( …
  -/
  rw [hf.fderiv_norm_rpow hp, norm_smul, norm_mul]
  simp_rw [norm_rpow_of_nonneg (norm_nonneg _), norm_norm, norm_eq_abs,
    abs_eq_self.mpr <| zero_le_one.trans hp.le, mul_assoc]
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : F → E
    hf : Differentiable Real f
    x : F
    p : Real
    hp : LT.lt 1 p
    ⊢ LE.le (HMul.hMul p (HMul.hMul (HPow.hPow (Norm.norm (f x)) (HSub.hSub p 2))  …
  -/
  gcongr _ * ?_
  refine mul_le_mul_of_nonneg_left (ContinuousLinearMap.opNorm_comp_le ..) (by positivity)
    |>.trans_eq ?_
  /-
    case h
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : F → E
    hf : Differentiable Real f
    x : F
    p : Real
    hp : LT.lt 1 p
    ⊢ Eq (HMul.hMul (HPow.hPow (Norm.norm (f x)) (HSub.hSub p 2)) (HMul.hMul (Norm …
  -/
  rw [innerSL_apply_norm, ← mul_assoc, ← Real.rpow_add_one' (by positivity) (by linarith)]
  /-
    case h
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : F → E
    hf : Differentiable Real f
    x : F
    p : Real
    hp : LT.lt 1 p
    ⊢ Eq (HMul.hMul (HPow.hPow (Norm.norm (f x)) (HAdd.hAdd (HSub.hSub p 2) 1)) (N …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem norm_fderiv_norm_id_rpow (x : E) {p : ℝ} (hp : 1 < p) :
    ‖fderiv ℝ (fun x ↦ ‖x‖ ^ p) x‖ = p * ‖x‖ ^ (p - 1) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    x : E
    p : Real
    hp : LT.lt 1 p
    ⊢ Eq (Norm.norm (fderiv Real (fun x => HPow.hPow (Norm.norm x) p) x)) (HMul.hM …
  -/
  rw [fderiv_norm_rpow x hp, norm_smul, norm_mul]
  simp_rw [norm_rpow_of_nonneg (norm_nonneg _), norm_norm, norm_eq_abs,
    abs_eq_self.mpr <| zero_le_one.trans hp.le, mul_assoc, innerSL_apply_norm]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    x : E
    p : Real
    hp : LT.lt 1 p
    ⊢ Eq (HMul.hMul p (HMul.hMul (HPow.hPow (Norm.norm x) (HSub.hSub p 2)) (Norm.n …
  -/
  rw [← Real.rpow_add_one' (by positivity) (by linarith)]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    x : E
    p : Real
    hp : LT.lt 1 p
    ⊢ Eq (HMul.hMul p (HPow.hPow (Norm.norm x) (HAdd.hAdd (HSub.hSub p 2) 1))) (HM …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem nnnorm_fderiv_norm_rpow_le {f : F → E} (hf : Differentiable ℝ f)
    {x : F} {p : ℝ≥0} (hp : 1 < p) :
    ‖fderiv ℝ (fun x ↦ ‖f x‖ ^ (p : ℝ)) x‖₊ ≤ p * ‖f x‖₊ ^ ((p : ℝ) - 1) * ‖fderiv ℝ f x‖₊ :=
  norm_fderiv_norm_rpow_le hf hp


theorem contDiff_norm_rpow {p : ℝ} (hp : 1 < p) : ContDiff ℝ 1 (fun x : E ↦ ‖x‖ ^ p) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    p : Real
    hp : LT.lt 1 p
    ⊢ ContDiff Real 1 fun x => HPow.hPow (Norm.norm x) p
  -/
  rw [contDiff_one_iff_fderiv]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    p : Real
    hp : LT.lt 1 p
    ⊢ And (Differentiable Real fun x => HPow.hPow (Norm.norm x) p) (Continuous (fd …
  -/
  refine ⟨fun x ↦ hasFDerivAt_norm_rpow x hp |>.differentiableAt, ?_⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    p : Real
    hp : LT.lt 1 p
    ⊢ Continuous (fderiv Real fun x => HPow.hPow (Norm.norm x) p)
  -/
  simp_rw [continuous_iff_continuousAt]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    p : Real
    hp : LT.lt 1 p
    ⊢ ∀ (x : E), ContinuousAt (fderiv Real fun x => HPow.hPow (Norm.norm x) p) x
  -/
  intro x
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    p : Real
    hp : LT.lt 1 p
    x : E
    ⊢ ContinuousAt (fderiv Real fun x => HPow.hPow (Norm.norm x) p) x
  -/
  by_cases hx : x = 0
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      p : Real
      hp : LT.lt 1 p
      x : E
      hx : Eq x 0
      ⊢ ContinuousAt (fderiv Real fun x => HPow.hPow (Norm.norm x) p) x
    -/
  · simp_rw [hx, ContinuousAt, fderiv_norm_rpow (0 : E) hp, norm_zero, map_zero, smul_zero]
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      p : Real
      hp : LT.lt 1 p
      x : E
      hx : Eq x 0
      ⊢ Filter.Tendsto (fderiv Real fun x => HPow.hPow (Norm.norm x) p) (nhds 0) (nh …
    -/
    rw [tendsto_zero_iff_norm_tendsto_zero]
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le (tendsto_const_nhds) ?_
      (fun _ ↦ norm_nonneg _) (fun _ ↦ norm_fderiv_norm_id_rpow _ hp |>.le)
    suffices ContinuousAt (fun x : E ↦ p * ‖x‖ ^ (p - 1)) 0  by
      simpa [ContinuousAt, sub_ne_zero_of_ne hp.ne'] using this
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      p : Real
      hp : LT.lt 1 p
      x : E
      hx : Eq x 0
      ⊢ ContinuousAt (fun x => HMul.hMul p (HPow.hPow (Norm.norm x) (HSub.hSub p 1)) …
    -/
    fun_prop (discharger := simp [hp.le])
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      p : Real
      hp : LT.lt 1 p
      x : E
      hx : Not (Eq x 0)
      ⊢ ContinuousAt (fderiv Real fun x => HPow.hPow (Norm.norm x) p) x
    -/
  · simp_rw [funext fun x ↦ fderiv_norm_rpow (E := E) (x := x) hp]
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      p : Real
      hp : LT.lt 1 p
      x : E
      hx : Not (Eq x 0)
      ⊢ ContinuousAt (fun x => HSMul.hSMul (HMul.hMul p (HPow.hPow (Norm.norm x) (HS …
    -/
    fun_prop (discharger := simp [hx])
    /-
      🎉 no goals
    -/


theorem ContDiff.norm_rpow {f : F → E} (hf : ContDiff ℝ 1 f) {p : ℝ} (hp : 1 < p) :
    ContDiff ℝ 1 (fun x ↦ ‖f x‖ ^ p) :=
  contDiff_norm_rpow hp |>.comp hf


theorem Differentiable.norm_rpow {f : F → E} (hf : Differentiable ℝ f) {p : ℝ} (hp : 1 < p) :
    Differentiable ℝ (fun x ↦ ‖f x‖ ^ p) :=
  contDiff_norm_rpow hp |>.differentiable le_rfl |>.comp hf


