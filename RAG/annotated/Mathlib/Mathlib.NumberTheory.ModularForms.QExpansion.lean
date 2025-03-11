local notation "I∞" => comap Complex.im atTop

local notation "𝕢" => Periodic.qParam


theorem Function.Periodic.im_invQParam_pos_of_abs_lt_one
    {h : ℝ} (hh : 0 < h) {q : ℂ} (hq : q.abs < 1) (hq_ne : q ≠ 0) :
    0 < im (Periodic.invQParam h q) :=
  im_invQParam .. ▸ mul_pos_of_neg_of_neg
    (div_neg_of_neg_of_pos (neg_lt_zero.mpr hh) Real.two_pi_pos)
    ((Real.log_neg_iff (abs.pos hq_ne)).mpr hq)


lemma Function.Periodic.abs_qParam_le_of_one_half_le_im {ξ : ℂ} (hξ : 1 / 2 ≤ ξ.im) :
    ‖𝕢 1 ξ‖ ≤ rexp (-π) := by
  rwa [Periodic.qParam, ofReal_one, div_one, norm_eq_abs, abs_exp, Real.exp_le_exp,
    mul_right_comm, mul_I_re, neg_le_neg_iff, ← ofReal_ofNat, ← ofReal_mul, im_ofReal_mul,
    mul_comm _ π, mul_assoc, le_mul_iff_one_le_right Real.pi_pos, ← div_le_iff₀' two_pos]


theorem periodic_comp_ofComplex [SlashInvariantFormClass F Γ(n) k] :
    Periodic (f ∘ ofComplex) n := by
  /-
    k : Int
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    n : Nat
    f : F
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma n) k
    ⊢ Function.Periodic (Function.comp ⇑f ↑UpperHalfPlane.ofComplex) ↑n
  -/
  intro w
  /-
    k : Int
    F : Type u_1
    inst✝¹ : FunLike F UpperHalfPlane Complex
    n : Nat
    f : F
    inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma n) k
    w : Complex
    ⊢ Eq (Function.comp (⇑f) (↑UpperHalfPlane.ofComplex) (HAdd.hAdd w ↑n)) (Functi …
  -/
  by_cases hw : 0 < im w
    /-
      case pos
      k : Int
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      n : Nat
      f : F
      inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma n) k
      w : Complex
      hw : LT.lt 0 w.im
      ⊢ Eq (Function.comp (⇑f) (↑UpperHalfPlane.ofComplex) (HAdd.hAdd w ↑n)) (Functi …
    -/
  · have : 0 < im (w + n) := by simp only [add_im, natCast_im, add_zero, hw]
    /-
      case pos
      k : Int
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      n : Nat
      f : F
      inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma n) k
      w : Complex
      hw : LT.lt 0 w.im
      this : LT.lt 0 (HAdd.hAdd w ↑n).im
      ⊢ Eq (Function.comp (⇑f) (↑UpperHalfPlane.ofComplex) (HAdd.hAdd w ↑n)) (Functi …
    -/
    simp only [comp_apply, ofComplex_apply_of_im_pos this, ofComplex_apply_of_im_pos hw]
    /-
      case pos
      k : Int
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      n : Nat
      f : F
      inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma n) k
      w : Complex
      hw : LT.lt 0 w.im
      this : LT.lt 0 (HAdd.hAdd w ↑n).im
      ⊢ Eq (f ⟨HAdd.hAdd w ↑n, this⟩) (f ⟨w, hw⟩)
    -/
    convert SlashInvariantForm.vAdd_width_periodic n k 1 f ⟨w, hw⟩ using 2
    simp only [Int.cast_one, mul_one, UpperHalfPlane.ext_iff, coe_mk_subtype, coe_vadd,
      ofReal_natCast, add_comm]
    /-
      case neg
      k : Int
      F : Type u_1
      inst✝¹ : FunLike F UpperHalfPlane Complex
      n : Nat
      f : F
      inst✝ : SlashInvariantFormClass F (CongruenceSubgroup.Gamma n) k
      w : Complex
      hw : Not (LT.lt 0 w.im)
      ⊢ Eq (Function.comp (⇑f) (↑UpperHalfPlane.ofComplex) (HAdd.hAdd w ↑n)) (Functi …
    -/
  · have : im (w + n) ≤ 0 := by simpa only [add_im, natCast_im, add_zero, not_lt] using hw
    simp only [comp_apply, ofComplex_apply_of_im_nonpos this,
      ofComplex_apply_of_im_nonpos (not_lt.mp hw)]


/--
The analytic function `F` such that `f τ = F (exp (2 * π * I * τ / n))`, extended by a choice of
limit at `0`.
-/
def cuspFunction : ℂ → ℂ := Function.Periodic.cuspFunction n (f ∘ ofComplex)


theorem eq_cuspFunction [NeZero n] [SlashInvariantFormClass F Γ(n) k] (τ : ℍ) :
    cuspFunction n f (𝕢 n τ) = f τ := by
  simpa only [comp_apply, ofComplex_apply]
    using (periodic_comp_ofComplex n f).eq_cuspFunction (NeZero.ne _) τ


theorem differentiableAt_comp_ofComplex [ModularFormClass F Γ k] {z : ℂ} (hz : 0 < im z) :
    DifferentiableAt ℂ (f ∘ ofComplex) z :=
  mdifferentiableAt_iff_differentiableAt.mp ((holo f _).comp z (mdifferentiableAt_ofComplex hz))


theorem bounded_at_infty_comp_ofComplex [ModularFormClass F Γ k] :
    BoundedAtFilter I∞ (f ∘ ofComplex) := by
  simpa only [SlashAction.slash_one, ModularForm.toSlashInvariantForm_coe]
    using (ModularFormClass.bdd_at_infty f 1).comp_tendsto tendsto_comap_im_ofComplex


theorem differentiableAt_cuspFunction [NeZero n] [ModularFormClass F Γ(n) k]
    {q : ℂ} (hq : q.abs < 1) :
    DifferentiableAt ℂ (cuspFunction n f) q := by
  /-
    k : Int
    F : Type u_1
    inst✝² : FunLike F UpperHalfPlane Complex
    n : Nat
    f : F
    inst✝¹ : NeZero n
    inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma n) k
    q : Complex
    hq : LT.lt (Complex.abs q) 1
    ⊢ DifferentiableAt Complex (SlashInvariantFormClass.cuspFunction n f) q
  -/
  have npos : 0 < (n : ℝ) := mod_cast (Nat.pos_iff_ne_zero.mpr (NeZero.ne _))
  /-
    k : Int
    F : Type u_1
    inst✝² : FunLike F UpperHalfPlane Complex
    n : Nat
    f : F
    inst✝¹ : NeZero n
    inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma n) k
    q : Complex
    hq : LT.lt (Complex.abs q) 1
    npos : LT.lt 0 ↑n
    ⊢ DifferentiableAt Complex (SlashInvariantFormClass.cuspFunction n f) q
  -/
  rcases eq_or_ne q 0 with rfl | hq'
  · exact (periodic_comp_ofComplex n f).differentiableAt_cuspFunction_zero npos
      (eventually_of_mem (preimage_mem_comap (Ioi_mem_atTop 0))
        (fun _ ↦ differentiableAt_comp_ofComplex f))
      (bounded_at_infty_comp_ofComplex f)
  · exact Periodic.qParam_right_inv npos.ne' hq' ▸
      (periodic_comp_ofComplex n f).differentiableAt_cuspFunction npos.ne'
        <| differentiableAt_comp_ofComplex _ <| Periodic.im_invQParam_pos_of_abs_lt_one npos hq hq'


lemma analyticAt_cuspFunction_zero [NeZero n] [ModularFormClass F Γ(n) k] :
    AnalyticAt ℂ (cuspFunction n f) 0 :=
  DifferentiableOn.analyticAt
    (fun q hq ↦ (differentiableAt_cuspFunction _ _ hq).differentiableWithinAt)
        /-
          k : Int
          F : Type u_1
          inst✝² : FunLike F UpperHalfPlane Complex
          n : Nat
          f : F
          inst✝¹ : NeZero n
          inst✝ : ModularFormClass F (CongruenceSubgroup.Gamma n) k
          ⊢ Membership.mem (nhds 0) fun q => Real.lt✝ (Complex.abs q) 1
        -/
    (by simpa only [ball_zero_eq] using Metric.ball_mem_nhds (0 : ℂ) zero_lt_one)
        /-
          🎉 no goals
        -/


theorem zero_at_infty_comp_ofComplex [CuspFormClass F Γ k] : ZeroAtFilter I∞ (f ∘ ofComplex) := by
  simpa only [SlashAction.slash_one, toSlashInvariantForm_coe]
    using (zero_at_infty f 1).comp tendsto_comap_im_ofComplex


theorem cuspFunction_apply_zero [NeZero n] [CuspFormClass F Γ(n) k] :
    cuspFunction n f 0 = 0 :=
  Periodic.cuspFunction_zero_of_zero_at_inf (mod_cast (Nat.pos_iff_ne_zero.mpr (NeZero.ne _)))
    (zero_at_infty_comp_ofComplex f)


theorem exp_decay_atImInfty [NeZero n] [CuspFormClass F Γ(n) k] :
    f =O[atImInfty] fun τ ↦ Real.exp (-2 * π * τ.im / n) := by
  simpa only [neg_mul, comp_def, ofComplex_apply, coe_im] using
    ((periodic_comp_ofComplex n f).exp_decay_of_zero_at_inf
      (mod_cast (Nat.pos_iff_ne_zero.mpr (NeZero.ne _)))
      (eventually_of_mem (preimage_mem_comap (Ioi_mem_atTop 0))
        fun _ ↦ differentiableAt_comp_ofComplex f)
      (zero_at_infty_comp_ofComplex f)).comp_tendsto tendsto_coe_atImInfty


