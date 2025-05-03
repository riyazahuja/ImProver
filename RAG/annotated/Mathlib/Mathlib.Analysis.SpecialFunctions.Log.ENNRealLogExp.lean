@[simp] lemma EReal.log_exp (x : EReal) : log (exp x) = x := by
  /-
    x : EReal
    ⊢ Eq x.exp.log x
  -/
  induction x
    /-
      case h_bot
      ⊢ Eq Bot.bot.exp.log Bot.bot
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h_real
      a✝ : Real
      ⊢ Eq (↑a✝).exp.log ↑a✝
    -/
  · rw [exp_coe, log_ofReal, if_neg (not_le.mpr (Real.exp_pos _)), Real.log_exp]
    /-
      🎉 no goals
    -/
    /-
      case h_top
      ⊢ Eq Top.top.exp.log Top.top
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp] lemma ENNReal.exp_log (x : ℝ≥0∞) : exp (log x) = x := by
  /-
    x : ENNReal
    ⊢ Eq x.log.exp x
  -/
  by_cases hx_top : x = ∞
    /-
      case pos
      x : ENNReal
      hx_top : Eq x Top.top
      ⊢ Eq x.log.exp x
    -/
  · simp [hx_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : ENNReal
    hx_top : Not (Eq x Top.top)
    ⊢ Eq x.log.exp x
  -/
  by_cases hx_zero : x = 0
    /-
      case pos
      x : ENNReal
      hx_top : Not (Eq x Top.top)
      hx_zero : Eq x 0
      ⊢ Eq x.log.exp x
    -/
  · simp [hx_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : ENNReal
    hx_top : Not (Eq x Top.top)
    hx_zero : Not (Eq x 0)
    ⊢ Eq x.log.exp x
  -/
  have hx_pos : 0 < x.toReal := ENNReal.toReal_pos hx_zero hx_top
  /-
    case neg
    x : ENNReal
    hx_top : Not (Eq x Top.top)
    hx_zero : Not (Eq x 0)
    hx_pos : LT.lt 0 x.toReal
    ⊢ Eq x.log.exp x
  -/
  rw [← ENNReal.ofReal_toReal hx_top, log_ofReal_of_pos hx_pos, exp_coe, Real.exp_log hx_pos]
  /-
    🎉 no goals
  -/


lemma exp_nmul (x : EReal) (n : ℕ) : exp (n * x) = (exp x) ^ n := by
  /-
    x : EReal
    n : Nat
    ⊢ Eq (HMul.hMul (↑n) x).exp (HPow.hPow x.exp n)
  -/
  simp_rw [← log_eq_iff, log_pow, log_exp]
  /-
    🎉 no goals
  -/


lemma exp_mul (x : EReal) (y : ℝ) : exp (x * y) = (exp x) ^ y := by
  /-
    x : EReal
    y : Real
    ⊢ Eq (HMul.hMul x ↑y).exp (HPow.hPow x.exp y)
  -/
  rw [← log_eq_iff, log_rpow, log_exp, log_exp, mul_comm]
  /-
    🎉 no goals
  -/


/-- `ENNReal.log` and its inverse `EReal.exp` are an order isomorphism between `ℝ≥0∞` and
`EReal`. -/
noncomputable
def logOrderIso : ℝ≥0∞ ≃o EReal where
  toFun := log
  invFun := exp
  left_inv x := exp_log x
  right_inv x := log_exp x
                     /-
                       ⊢ ∀ {a b : ENNReal}, Iff (LE.le ({ toFun := ENNReal.log, invFun := EReal.exp,  …
                     -/
  map_rel_iff' := by simp only [Equiv.coe_fn_mk, log_le_log_iff, forall_const]
                     /-
                       🎉 no goals
                     -/


@[simp] lemma logOrderIso_apply (x : ℝ≥0∞) : logOrderIso x = log x := rfl


/-- `EReal.exp` and its inverse `ENNReal.log` are an order isomorphism between `EReal` and
`ℝ≥0∞`. -/
noncomputable
def _root_.EReal.expOrderIso := logOrderIso.symm


@[simp] lemma _root_.EReal.expOrderIso_apply (x : EReal) : expOrderIso x = exp x := rfl


@[simp] lemma logOrderIso_symm : logOrderIso.symm = expOrderIso := rfl

@[simp] lemma _root_.EReal.expOrderIso_symm : expOrderIso.symm = logOrderIso := rfl


/-- `log` as a homeomorphism. -/
noncomputable def logHomeomorph : ℝ≥0∞ ≃ₜ EReal := logOrderIso.toHomeomorph


@[simp] lemma logHomeomorph_apply (x : ℝ≥0∞) : logHomeomorph x = log x := rfl


/-- `exp` as a homeomorphism. -/
noncomputable def _root_.EReal.expHomeomorph : EReal ≃ₜ ℝ≥0∞ := expOrderIso.toHomeomorph


@[simp] lemma _root_.EReal.expHomeomorph_apply (x : EReal) : expHomeomorph x = exp x := rfl


@[simp] lemma logHomeomorph_symm : logHomeomorph.symm = expHomeomorph := rfl


@[simp] lemma _root_.EReal.expHomeomorph_symm : expHomeomorph.symm = logHomeomorph := rfl


@[continuity, fun_prop]
lemma continuous_log : Continuous log := logOrderIso.continuous


@[continuity, fun_prop]
lemma continuous_exp : Continuous exp := expOrderIso.continuous


@[measurability, fun_prop]
lemma measurable_log : Measurable log := continuous_log.measurable


@[measurability, fun_prop]
lemma _root_.EReal.measurable_exp : Measurable exp := continuous_exp.measurable


@[measurability, fun_prop]
lemma _root_.Measurable.ennreal_log {α : Type*} {_ : MeasurableSpace α}
    {f : α → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun x ↦ log (f x) := measurable_log.comp hf


@[measurability, fun_prop]
lemma _root_.Measurable.ereal_exp {α : Type*} {_ : MeasurableSpace α}
    {f : α → EReal} (hf : Measurable f) :
    Measurable fun x ↦ exp (f x) := measurable_exp.comp hf


instance : PolishSpace EReal := ENNReal.logOrderIso.symm.toHomeomorph.isClosedEmbedding.polishSpace

