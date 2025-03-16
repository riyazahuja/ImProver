/-- A structure designed to hold the hypotheses for the Mellin-functional-equation argument
(most general version: rapid decay at `∞` up to constant terms) -/
structure WeakFEPair where
  /-- The functions whose Mellin transform we study -/
  (f g : ℝ → E)
  /-- Weight (exponent in the functional equation) -/
  (k : ℝ)
  /-- Root number -/
  (ε : ℂ)
  /-- Constant terms at `∞` -/
  (f₀ g₀ : E)
            /-
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              inst✝ : NormedSpace Complex E
              f g : Real → E
              k : Real
              ε : Complex
              f₀ g₀ : E
              ⊢ MeasureTheory.Measure Real
            -/
  (hf_int : LocallyIntegrableOn f (Ioi 0))
            /-
              🎉 no goals
            -/
            /-
              E : Type u_1
              inst✝¹ : NormedAddCommGroup E
              inst✝ : NormedSpace Complex E
              f g : Real → E
              k : Real
              ε : Complex
              f₀ g₀ : E
              hf_int : MeasureTheory.LocallyIntegrableOn f (Set.Ioi 0) MeasureTheory.Measure …
              ⊢ MeasureTheory.Measure Real
            -/
  (hg_int : LocallyIntegrableOn g (Ioi 0))
            /-
              🎉 no goals
            -/
  (hk : 0 < k)
  (hε : ε ≠ 0)
  (h_feq : ∀ x ∈ Ioi 0, f (1 / x) = (ε * ↑(x ^ k)) • g x)
  (hf_top (r : ℝ) : (f · - f₀) =O[atTop] (· ^ r))
  (hg_top (r : ℝ) : (g · - g₀) =O[atTop] (· ^ r))


/-- A structure designed to hold the hypotheses for the Mellin-functional-equation argument
(version without constant terms) -/
structure StrongFEPair extends WeakFEPair E where (hf₀ : f₀ = 0) (hg₀ : g₀ = 0)


/-- Reformulated functional equation with `f` and `g` interchanged. -/
lemma WeakFEPair.h_feq' (P : WeakFEPair E) (x : ℝ) (hx : 0 < x) :
    P.g (1 / x) = (P.ε⁻¹ * ↑(x ^ P.k)) • P.f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (P.g (HDiv.hDiv 1 x)) (HSMul.hSMul (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x …
  -/
  rw [(div_div_cancel₀ (one_ne_zero' ℝ) ▸ P.h_feq (1 / x) (one_div_pos.mpr hx):), ← mul_smul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (P.g (HDiv.hDiv 1 x)) (HSMul.hSMul (HMul.hMul (HMul.hMul (Inv.inv P.ε) ↑( …
  -/
  convert (one_smul ℂ (P.g (1 / x))).symm using 2
  /-
    case h.e'_3.h.e'_5
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) (HMul.hMul P.ε ↑( …
  -/
  rw [one_div, inv_rpow hx.le, ofReal_inv]
  /-
    case h.e'_3.h.e'_5
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) (HMul.hMul P.ε (I …
  -/
  field_simp [P.hε, (rpow_pos_of_pos hx _).ne']
  /-
    🎉 no goals
  -/


/-- The hypotheses are symmetric in `f` and `g`, with the constant `ε` replaced by `ε⁻¹`. -/
def WeakFEPair.symm (P : WeakFEPair E) : WeakFEPair E where
  hf_int := P.hg_int
  hg_int := P.hf_int
  hf_top := P.hg_top
  hg_top := P.hf_top
  hε     := inv_ne_zero P.hε
  hk     := P.hk
  h_feq  := P.h_feq'


/-- The hypotheses are symmetric in `f` and `g`, with the constant `ε` replaced by `ε⁻¹`. -/
def StrongFEPair.symm (P : StrongFEPair E) : StrongFEPair E where
  toWeakFEPair := P.toWeakFEPair.symm
  hf₀ := P.hg₀
  hg₀ := P.hf₀


/-- As `x → 0`, we have `f x = x ^ (-P.k) • constant` up to a rapidly decaying error. -/
lemma hf_zero (P : WeakFEPair E) (r : ℝ) :
    (fun x ↦ P.f x - (P.ε * ↑(x ^ (-P.k))) • P.g₀) =O[𝓝[>] 0] (· ^ r) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r : Real
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HSub.hSub (P.f x) (H …
  -/
  have := (P.hg_top (-(r + P.k))).comp_tendsto tendsto_inv_nhdsGT_zero
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r : Real
    this : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (Function.comp (fun x =>  …
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HSub.hSub (P.f x) (H …
  -/
  simp_rw [IsBigO, IsBigOWith, eventually_nhdsWithin_iff] at this ⊢
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r : Real
    this : Exists fun c => Filter.Eventually (fun x => Membership.mem (Set.Ioi 0)  …
    ⊢ Exists fun c => Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → L …
  -/
  obtain ⟨C, hC⟩ := this
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    ⊢ Exists fun c => Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → L …
  -/
  use ‖P.ε‖ * C
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.norm  …
  -/
  filter_upwards [hC] with x hC' (hx : 0 < x)
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    x : Real
    hC' : Membership.mem (Set.Ioi 0) x → LE.le (Norm.norm (Function.comp (fun x => …
    hx : LT.lt 0 x
    ⊢ LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow  …
  -/
  have h_nv2 : ↑(x ^ P.k) ≠ (0 : ℂ) := ofReal_ne_zero.mpr (rpow_pos_of_pos hx _).ne'
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    x : Real
    hC' : Membership.mem (Set.Ioi 0) x → LE.le (Norm.norm (Function.comp (fun x => …
    hx : LT.lt 0 x
    h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
    ⊢ LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow  …
  -/
  have h_nv : P.ε⁻¹ * ↑(x ^ P.k) ≠ 0 := mul_ne_zero P.symm.hε h_nv2
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    x : Real
    hC' : Membership.mem (Set.Ioi 0) x → LE.le (Norm.norm (Function.comp (fun x => …
    hx : LT.lt 0 x
    h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
    h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
    ⊢ LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow  …
  -/
  specialize hC' hx
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    x : Real
    hx : LT.lt 0 x
    h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
    h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
    hC' : LE.le (Norm.norm (Function.comp (fun x => HSub.hSub (P.g x) P.g₀) (fun x …
    ⊢ LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow  …
  -/
  simp_rw [Function.comp_apply, ← one_div, P.h_feq' _ hx] at hC'
  rw [← ((mul_inv_cancel₀ h_nv).symm ▸ one_smul ℂ P.g₀ :), mul_smul _ _ P.g₀, ← smul_sub, norm_smul,
    ← le_div_iff₀' (lt_of_le_of_ne (norm_nonneg _) (norm_ne_zero_iff.mpr h_nv).symm)] at hC'
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    r C : Real
    hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
    x : Real
    hx : LT.lt 0 x
    h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
    h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
    hC' : LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (Inv.inv (HMul.hMul (In …
    ⊢ LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow  …
  -/
  convert hC' using 1
    /-
      case h.e'_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r C : Real
      hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
      x : Real
      hx : LT.lt 0 x
      h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
      h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
      hC' : LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (Inv.inv (HMul.hMul (In …
      ⊢ Eq (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x ( …
    -/
  · congr 3
    /-
      case h.e'_3.e_a.e_a.e_a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r C : Real
      hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
      x : Real
      hx : LT.lt 0 x
      h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
      h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
      hC' : LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (Inv.inv (HMul.hMul (In …
      ⊢ Eq (HMul.hMul P.ε ↑(HPow.hPow x (Neg.neg P.k))) (Inv.inv (HMul.hMul (Inv.inv …
    -/
    rw [rpow_neg hx.le]
    /-
      case h.e'_3.e_a.e_a.e_a
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r C : Real
      hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
      x : Real
      hx : LT.lt 0 x
      h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
      h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
      hC' : LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (Inv.inv (HMul.hMul (In …
      ⊢ Eq (HMul.hMul P.ε ↑(Inv.inv (HPow.hPow x P.k))) (Inv.inv (HMul.hMul (Inv.inv …
    -/
    field_simp
    /-
      🎉 no goals
    -/
  · simp_rw [norm_mul, norm_real, one_div, inv_rpow hx.le, rpow_neg hx.le, inv_inv, norm_inv,
      norm_of_nonneg (rpow_pos_of_pos hx _).le, rpow_add hx]
    /-
      case h.e'_4
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r C : Real
      hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
      x : Real
      hx : LT.lt 0 x
      h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
      h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
      hC' : LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (Inv.inv (HMul.hMul (In …
      ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm P.ε) C) (HPow.hPow x r)) (HDiv.hDiv (HMu …
    -/
    field_simp
    /-
      case h.e'_4
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r C : Real
      hC : Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.no …
      x : Real
      hx : LT.lt 0 x
      h_nv2 : Ne (↑(HPow.hPow x P.k)) 0
      h_nv : Ne (HMul.hMul (Inv.inv P.ε) ↑(HPow.hPow x P.k)) 0
      hC' : LE.le (Norm.norm (HSub.hSub (P.f x) (HSMul.hSMul (Inv.inv (HMul.hMul (In …
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Complex.abs P.ε) C) (HPow.hPow x r)) (H …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Power asymptotic for `f - f₀` as `x → 0`. -/
lemma hf_zero' (P : WeakFEPair E) :
    (fun x : ℝ ↦ P.f x - P.f₀) =O[𝓝[>] 0] (· ^ (-P.k)) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HSub.hSub (P.f x) P. …
  -/
  simp_rw [← fun x ↦ sub_add_sub_cancel (P.f x) ((P.ε * ↑(x ^ (-P.k))) • P.g₀) P.f₀]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HAdd.hAdd (HSub.hSub …
  -/
  refine (P.hf_zero _).add (IsBigO.sub ?_ ?_)
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HSMul.hSMul (HMul.hM …
    -/
  · rw [← isBigO_norm_norm]
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => Norm.norm (HSMul.hSM …
    -/
    simp_rw [mul_smul, norm_smul, mul_comm _ ‖P.g₀‖, ← mul_assoc, norm_real]
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HMul.hMul (HMul.hMul …
    -/
    apply (isBigO_refl _ _).const_mul_left
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => P.f₀) fun x => HPow. …
    -/
  · refine IsBigO.of_bound ‖P.f₀‖ (eventually_nhdsWithin_iff.mpr ?_)
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.norm  …
    -/
    filter_upwards [eventually_le_nhds zero_lt_one] with x hx' (hx : 0 < x)
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx' : LE.le x 1
      hx : LT.lt 0 x
      ⊢ LE.le (Norm.norm P.f₀) (HMul.hMul (Norm.norm P.f₀) (Norm.norm (HPow.hPow x ( …
    -/
    apply le_mul_of_one_le_right (norm_nonneg _)
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx' : LE.le x 1
      hx : LT.lt 0 x
      ⊢ LE.le 1 (Norm.norm (HPow.hPow x (Neg.neg P.k)))
    -/
    rw [norm_of_nonneg (rpow_pos_of_pos hx _).le, rpow_neg hx.le]
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx' : LE.le x 1
      hx : LT.lt 0 x
      ⊢ LE.le 1 (Inv.inv (HPow.hPow x P.k))
    -/
    exact (one_le_inv₀ (rpow_pos_of_pos hx _)).2 (rpow_le_one hx.le hx' P.hk.le)
    /-
      🎉 no goals
    -/


/-- As `x → ∞`, `f x` decays faster than any power of `x`. -/
lemma hf_top' (r : ℝ) : P.f =O[atTop] (· ^ r) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    r : Real
    ⊢ Asymptotics.IsBigO Filter.atTop P.f fun x => HPow.hPow x r
  -/
  simpa only [P.hf₀, sub_zero] using P.hf_top r
  /-
    🎉 no goals
  -/


/-- As `x → 0`, `f x` decays faster than any power of `x`. -/
lemma hf_zero' (r : ℝ) : P.f =O[𝓝[>] 0] (· ^ r) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    r : Real
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) P.f fun x => HPow.hPow x r
  -/
  have := P.hg₀ ▸ P.hf_zero r
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    r : Real
    this : Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HSub.hSub (P.f  …
    ⊢ Asymptotics.IsBigO (nhdsWithin 0 (Set.Ioi 0)) P.f fun x => HPow.hPow x r
  -/
  simpa only [smul_zero, sub_zero]
  /-
    🎉 no goals
  -/


/-- The completed L-function. -/
def Λ : ℂ → E := mellin P.f


/-- The Mellin transform of `f` is well-defined and equal to `P.Λ s`, for all `s`. -/
theorem hasMellin (s : ℂ) : HasMellin P.f s (P.Λ s) :=
  let ⟨_, ht⟩ := exists_gt s.re
  let ⟨_, hu⟩ := exists_lt s.re
  ⟨mellinConvergent_of_isBigO_rpow P.hf_int (P.hf_top' _) ht (P.hf_zero' _) hu, rfl⟩


lemma Λ_eq : P.Λ = mellin P.f := rfl


lemma symm_Λ_eq : P.symm.Λ = mellin P.g := rfl


/-- If `(f, g)` are a strong FE pair, then the Mellin transform of `f` is entire. -/
theorem differentiable_Λ : Differentiable ℂ P.Λ := fun s ↦
  let ⟨_, ht⟩ := exists_gt s.re
  let ⟨_, hu⟩ := exists_lt s.re
  mellin_differentiableAt_of_isBigO_rpow P.hf_int (P.hf_top' _) ht (P.hf_zero' _) hu


/-- Main theorem about strong FE pairs: if `(f, g)` are a strong FE pair, then the Mellin
transforms of `f` and `g` are related by `s ↦ k - s`.

This is proved by making a substitution `t ↦ t⁻¹` in the Mellin transform integral. -/
theorem functional_equation (s : ℂ) :
    P.Λ (P.k - s) = P.ε • P.symm.Λ s := by
  -- unfold definition:
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    ⊢ Eq (P.Λ (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (P.symm.Λ s))
  -/
  rw [P.Λ_eq, P.symm_Λ_eq]
  -- substitute `t ↦ t⁻¹` in `mellin P.g s`
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (mellin P.g s))
  -/
  have step1 := mellin_comp_rpow P.g (-s) (-1)
  simp_rw [abs_neg, abs_one, inv_one, one_smul, ofReal_neg, ofReal_one, div_neg, div_one, neg_neg,
    rpow_neg_one, ← one_div] at step1
  -- introduce a power of `t` to match the hypothesis `P.h_feq`
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (mellin P.g s))
  -/
  have step2 := mellin_cpow_smul (fun t ↦ P.g (1 / t)) (P.k - s) (-P.k)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (mellin P.g s))
  -/
  rw [← sub_eq_add_neg, sub_right_comm, sub_self, zero_sub, step1] at step2
  -- put in the constant `P.ε`
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (mellin P.g s))
  -/
  have step3 := mellin_const_smul (fun t ↦ (t : ℂ) ^ (-P.k : ℂ) • P.g (1 / t)) (P.k - s) P.ε
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    step3 : Eq (mellin (fun t => HSMul.hSMul P.ε (HSMul.hSMul (HPow.hPow (↑t) (Neg …
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (mellin P.g s))
  -/
  rw [step2] at step3
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    step3 : Eq (mellin (fun t => HSMul.hSMul P.ε (HSMul.hSMul (HPow.hPow (↑t) (Neg …
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (HSMul.hSMul P.ε (mellin P.g s))
  -/
  rw [← step3]
  -- now the integrand matches `P.h_feq'` on `Ioi 0`, so we can apply `setIntegral_congr_fun`
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    step3 : Eq (mellin (fun t => HSMul.hSMul P.ε (HSMul.hSMul (HPow.hPow (↑t) (Neg …
    ⊢ Eq (mellin P.f (HSub.hSub (↑P.k) s)) (mellin (fun t => HSMul.hSMul P.ε (HSMu …
  -/
  refine setIntegral_congr_fun measurableSet_Ioi (fun t ht ↦ ?_)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    step3 : Eq (mellin (fun t => HSMul.hSMul P.ε (HSMul.hSMul (HPow.hPow (↑t) (Neg …
    t : Real
    ht : Membership.mem (Set.Ioi 0) t
    ⊢ Eq (HSMul.hSMul (HPow.hPow (↑t) (HSub.hSub (HSub.hSub (↑P.k) s) 1)) (P.f t)) …
  -/
  simp_rw [P.h_feq' t ht, ← mul_smul]
  -- some simple `cpow` arithmetic to finish
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    step3 : Eq (mellin (fun t => HSMul.hSMul P.ε (HSMul.hSMul (HPow.hPow (↑t) (Neg …
    t : Real
    ht : Membership.mem (Set.Ioi 0) t
    ⊢ Eq (HSMul.hSMul (HPow.hPow (↑t) (HSub.hSub (HSub.hSub (↑P.k) s) 1)) (P.f t)) …
  -/
  rw [cpow_neg, ofReal_cpow (le_of_lt ht)]
  have : (t : ℂ) ^ (P.k : ℂ) ≠ 0 := by
    simpa only [← ofReal_cpow (le_of_lt ht), ofReal_ne_zero] using (rpow_pos_of_pos ht _).ne'
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : StrongFEPair E
    s : Complex
    step1 : Eq (mellin (fun t => P.g (HDiv.hDiv 1 t)) (Neg.neg s)) (mellin P.g s)
    step2 : Eq (mellin (fun t => HSMul.hSMul (HPow.hPow (↑t) (Neg.neg ↑P.k)) (P.g  …
    step3 : Eq (mellin (fun t => HSMul.hSMul P.ε (HSMul.hSMul (HPow.hPow (↑t) (Neg …
    t : Real
    ht : Membership.mem (Set.Ioi 0) t
    this : Ne (HPow.hPow ↑t ↑P.k) 0
    ⊢ Eq (HSMul.hSMul (HPow.hPow (↑t) (HSub.hSub (HSub.hSub (↑P.k) s) 1)) (P.f t)) …
  -/
  field_simp [P.hε]
  /-
    🎉 no goals
  -/


/-- Piecewise modified version of `f` with optimal asymptotics. We deliberately choose intervals
which don't quite join up, so the function is `0` at `x = 1`, in order to maintain symmetry;
there is no "good" choice of value at `1`. -/
def f_modif : ℝ → E :=
  (Ioi 1).indicator (fun x ↦ P.f x - P.f₀) +
  (Ioo 0 1).indicator (fun x ↦ P.f x - (P.ε * ↑(x ^ (-P.k))) • P.g₀)


/-- Piecewise modified version of `g` with optimal asymptotics. -/
def g_modif : ℝ → E :=
  (Ioi 1).indicator (fun x ↦ P.g x - P.g₀) +
  (Ioo 0 1).indicator (fun x ↦ P.g x - (P.ε⁻¹ * ↑(x ^ (-P.k))) • P.f₀)


lemma hf_modif_int :
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ MeasureTheory.Measure Real
    -/
    LocallyIntegrableOn P.f_modif (Ioi 0) := by
    /-
      🎉 no goals
    -/
  have : LocallyIntegrableOn (fun x : ℝ ↦ (P.ε * ↑(x ^ (-P.k))) • P.g₀) (Ioi 0) := by
    refine ContinuousOn.locallyIntegrableOn ?_ measurableSet_Ioi
    refine continuousOn_of_forall_continuousAt (fun x (hx : 0 < x) ↦ ?_)
    refine (continuousAt_const.mul ?_).smul continuousAt_const
    exact continuous_ofReal.continuousAt.comp (continuousAt_rpow_const _ _ (Or.inl hx.ne'))
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
    ⊢ MeasureTheory.LocallyIntegrableOn P.f_modif (Set.Ioi 0) MeasureTheory.Measur …
  -/
  refine LocallyIntegrableOn.add (fun x hx ↦ ?_) (fun x hx ↦ ?_)
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ MeasureTheory.IntegrableAtFilter ((Set.Ioi 1).indicator fun x => HSub.hSub ( …
    -/
  · obtain ⟨s, hs, hs'⟩ := P.hf_int.sub (locallyIntegrableOn_const _) x hx
    /-
      case refine_1.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      s : Set Real
      hs : Membership.mem (nhdsWithin x (Set.Ioi 0)) s
      hs' : MeasureTheory.IntegrableOn (HSub.hSub P.f fun x => ?m.79635) s MeasureTh …
      ⊢ MeasureTheory.IntegrableAtFilter ((Set.Ioi 1).indicator fun x => HSub.hSub ( …
    -/
    refine ⟨s, hs, ?_⟩
    rw [IntegrableOn, integrable_indicator_iff measurableSet_Ioi, IntegrableOn,
      Measure.restrict_restrict measurableSet_Ioi, ← IntegrableOn]
    /-
      case refine_1.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      s : Set Real
      hs : Membership.mem (nhdsWithin x (Set.Ioi 0)) s
      hs' : MeasureTheory.IntegrableOn (HSub.hSub P.f fun x => ?m.79635) s MeasureTh …
      ⊢ MeasureTheory.IntegrableOn (fun x => HSub.hSub (P.f x) P.f₀) (Inter.inter (S …
    -/
    exact hs'.mono_set Set.inter_subset_right
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ MeasureTheory.IntegrableAtFilter ((Set.Ioo 0 1).indicator fun x => HSub.hSub …
    -/
  · obtain ⟨s, hs, hs'⟩ := P.hf_int.sub this x hx
    /-
      case refine_2.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      s : Set Real
      hs : Membership.mem (nhdsWithin x (Set.Ioi 0)) s
      hs' : MeasureTheory.IntegrableOn (HSub.hSub P.f fun x => HSMul.hSMul (HMul.hMu …
      ⊢ MeasureTheory.IntegrableAtFilter ((Set.Ioo 0 1).indicator fun x => HSub.hSub …
    -/
    refine ⟨s, hs, ?_⟩
    rw [IntegrableOn, integrable_indicator_iff measurableSet_Ioo, IntegrableOn,
      Measure.restrict_restrict measurableSet_Ioo, ← IntegrableOn]
    /-
      case refine_2.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      this : MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (HMul.hMul P.ε  …
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      s : Set Real
      hs : Membership.mem (nhdsWithin x (Set.Ioi 0)) s
      hs' : MeasureTheory.IntegrableOn (HSub.hSub P.f fun x => HSMul.hSMul (HMul.hMu …
      ⊢ MeasureTheory.IntegrableOn (fun x => HSub.hSub (P.f x) (HSMul.hSMul (HMul.hM …
    -/
    exact hs'.mono_set Set.inter_subset_right
    /-
      🎉 no goals
    -/


lemma hf_modif_FE (x : ℝ) (hx : 0 < x) :
    P.f_modif (1 / x) = (P.ε * ↑(x ^ P.k)) • P.g_modif x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (P.f_modif (HDiv.hDiv 1 x)) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x P.k …
  -/
  rcases lt_trichotomy 1 x with hx' | rfl | hx'
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt 1 x
      ⊢ Eq (P.f_modif (HDiv.hDiv 1 x)) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x P.k …
    -/
  · have : 1 / x < 1 := by rwa [one_div_lt hx one_pos, div_one]
    rw [f_modif, Pi.add_apply, indicator_of_not_mem (not_mem_Ioi.mpr this.le),
      zero_add, indicator_of_mem (mem_Ioo.mpr ⟨div_pos one_pos hx, this⟩), g_modif, Pi.add_apply,
      indicator_of_mem (mem_Ioi.mpr hx'), indicator_of_not_mem
      (not_mem_Ioo_of_ge hx'.le), add_zero, P.h_feq _ hx, smul_sub]
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt 1 x
      this : LT.lt (HDiv.hDiv 1 x) 1
      ⊢ Eq (HSub.hSub (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x P.k)) (P.g x)) (HSMu …
    -/
    simp_rw [rpow_neg (one_div_pos.mpr hx).le, one_div, inv_rpow hx.le, inv_inv]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      hx : LT.lt 0 1
      ⊢ Eq (P.f_modif (1 / 1)) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow 1 P.k)) (P.g_ …
    -/
  · simp [f_modif, g_modif]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt x 1
      ⊢ Eq (P.f_modif (HDiv.hDiv 1 x)) (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x P.k …
    -/
  · have : 1 < 1 / x := by rwa [lt_one_div one_pos hx, div_one]
    rw [f_modif, Pi.add_apply, indicator_of_mem (mem_Ioi.mpr this),
      indicator_of_not_mem (not_mem_Ioo_of_ge this.le), add_zero, g_modif, Pi.add_apply,
      indicator_of_not_mem (not_mem_Ioi.mpr hx'.le),
      indicator_of_mem (mem_Ioo.mpr ⟨hx, hx'⟩), zero_add, P.h_feq _ hx, smul_sub]
    /-
      case inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt x 1
      this : LT.lt 1 (HDiv.hDiv 1 x)
      ⊢ Eq (HSub.hSub (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x P.k)) (P.g x)) P.f₀) …
    -/
    simp_rw [rpow_neg hx.le, ← mul_smul]
    /-
      case inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt x 1
      this : LT.lt 1 (HDiv.hDiv 1 x)
      ⊢ Eq (HSub.hSub (HSMul.hSMul (HMul.hMul P.ε ↑(HPow.hPow x P.k)) (P.g x)) P.f₀) …
    -/
    field_simp [(rpow_pos_of_pos hx P.k).ne', P.hε]
    /-
      🎉 no goals
    -/


/-- Given a weak FE-pair `(f, g)`, modify it into a strong FE-pair by subtracting suitable
correction terms from `f` and `g`. -/
def toStrongFEPair : StrongFEPair E where
  hf_int   := P.hf_modif_int
  hg_int   := P.symm.hf_modif_int
  h_feq    := P.hf_modif_FE
  hε       := P.hε
  hk       := P.hk
  hf₀      := rfl
  hg₀      := rfl
  hf_top r := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r : Real
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (P.f_modif x) 0) fun x = …
    -/
    refine (P.hf_top r).congr' ?_ (by rfl)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r : Real
      ⊢ Filter.atTop.EventuallyEq (fun x => HSub.hSub (P.f x) P.f₀) fun x => HSub.hS …
    -/
    filter_upwards [eventually_gt_atTop 1] with x hx
    rw [f_modif, Pi.add_apply, indicator_of_mem (mem_Ioi.mpr hx),
      indicator_of_not_mem (not_mem_Ioo_of_ge hx.le), add_zero, sub_zero]
  hg_top r := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r : Real
      ⊢ Asymptotics.IsBigO Filter.atTop (fun x => HSub.hSub (P.symm.f_modif x) 0) fu …
    -/
    refine (P.hg_top r).congr' ?_ (by rfl)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r : Real
      ⊢ Filter.atTop.EventuallyEq (fun x => HSub.hSub (P.g x) P.g₀) fun x => HSub.hS …
    -/
    filter_upwards [eventually_gt_atTop 1] with x hx
    rw [f_modif, Pi.add_apply, indicator_of_mem (mem_Ioi.mpr hx),
      indicator_of_not_mem (not_mem_Ioo_of_ge hx.le), add_zero, sub_zero]
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      r x : Real
      hx : LT.lt 1 x
      ⊢ Eq (HSub.hSub (P.g x) P.g₀) (HSub.hSub (P.symm.f x) P.symm.f₀)
    -/
    rfl
    /-
      🎉 no goals
    -/

/- Alternative form for the difference between `f - f₀` and its modified term. -/

lemma f_modif_aux1 : EqOn (fun x ↦ P.f_modif x - P.f x + P.f₀)
    ((Ioo 0 1).indicator (fun x : ℝ ↦ P.f₀ - (P.ε * ↑(x ^ (-P.k))) • P.g₀)
    + ({1} : Set ℝ).indicator (fun _ ↦ P.f₀ - P.f 1)) (Ioi 0) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Set.EqOn (fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) (HAdd.h …
  -/
  intro x (hx : 0 < x)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq ((fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) x) (HAdd.hAd …
  -/
  simp_rw [f_modif, Pi.add_apply]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    x : Real
    hx : LT.lt 0 x
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd ((Set.Ioi 1).indicator (fun x => HSub.hS …
  -/
  rcases lt_trichotomy x 1 with hx' | rfl | hx'
  · simp_rw [indicator_of_not_mem (not_mem_Ioi.mpr hx'.le),
      indicator_of_mem (mem_Ioo.mpr ⟨hx, hx'⟩),
      indicator_of_not_mem (mem_singleton_iff.not.mpr hx'.ne)]
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt x 1
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd 0 (HSub.hSub (P.f x) (HSMul.hSMul (HMul. …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      hx : LT.lt 0 1
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd ((Set.Ioi 1).indicator (fun x => HSub.hS …
    -/
  · simp [add_comm, sub_eq_add_neg]
    /-
      🎉 no goals
    -/
  · simp_rw [indicator_of_mem (mem_Ioi.mpr hx'),
      indicator_of_not_mem (not_mem_Ioo_of_ge hx'.le),
      indicator_of_not_mem (mem_singleton_iff.not.mpr hx'.ne')]
    /-
      case inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      x : Real
      hx : LT.lt 0 x
      hx' : LT.lt 1 x
      ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub (P.f x) P.f₀) 0) (P.f x)) P.f …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/-- Compute the Mellin transform of the modifying term used to kill off the constants at
`0` and `∞`. -/
lemma f_modif_aux2 [CompleteSpace E] {s : ℂ} (hs : P.k < re s) :
    mellin (fun x ↦ P.f_modif x - P.f x + P.f₀) s = (1 / s) • P.f₀ + (P.ε  / (P.k - s)) • P.g₀ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    ⊢ Eq (mellin (fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) s) (H …
  -/
  have h_re1 : -1 < re (s - 1) := by simpa using P.hk.trans hs
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    h_re1 : LT.lt (-1) (HSub.hSub s 1).re
    ⊢ Eq (mellin (fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) s) (H …
  -/
  have h_re2 : -1 < re (s - P.k - 1) := by simpa using hs
  calc
  _ = ∫ (x : ℝ) in Ioi 0, (x : ℂ) ^ (s - 1) •
      ((Ioo 0 1).indicator (fun t : ℝ ↦ P.f₀ - (P.ε * ↑(t ^ (-P.k))) • P.g₀) x
      + ({1} : Set ℝ).indicator (fun _ ↦ P.f₀ - P.f 1) x) :=
    setIntegral_congr_fun measurableSet_Ioi (fun x hx ↦ by simp [f_modif_aux1 P hx])
  _ = ∫ (x : ℝ) in Ioi 0, (x : ℂ) ^ (s - 1) • ((Ioo 0 1).indicator
      (fun t : ℝ ↦ P.f₀ - (P.ε * ↑(t ^ (-P.k))) • P.g₀) x) := by
    refine setIntegral_congr_ae measurableSet_Ioi (eventually_of_mem (U := {1}ᶜ)
        (compl_mem_ae_iff.mpr (subsingleton_singleton.measure_zero _)) (fun x hx _ ↦ ?_))
    rw [indicator_of_not_mem hx, add_zero]
  _ = ∫ (x : ℝ) in Ioc 0 1, (x : ℂ) ^ (s - 1) • (P.f₀ - (P.ε * ↑(x ^ (-P.k))) • P.g₀) := by
    simp_rw [← indicator_smul, setIntegral_indicator measurableSet_Ioo,
      inter_eq_right.mpr Ioo_subset_Ioi_self, integral_Ioc_eq_integral_Ioo]
  _ = ∫ x : ℝ in Ioc 0 1, ((x : ℂ) ^ (s - 1) • P.f₀ - P.ε • (x : ℂ) ^ (s - P.k - 1) • P.g₀) := by
    refine setIntegral_congr_fun measurableSet_Ioc (fun x ⟨hx, _⟩ ↦ ?_)
    rw [ofReal_cpow hx.le, ofReal_neg, smul_sub, ← mul_smul, mul_comm, mul_assoc, mul_smul,
      mul_comm, ← cpow_add _ _ (ofReal_ne_zero.mpr hx.ne'), ← sub_eq_add_neg, sub_right_comm]
  _ = (∫ (x : ℝ) in Ioc 0 1, (x : ℂ) ^ (s - 1)) • P.f₀
        - P.ε • (∫ (x : ℝ) in Ioc 0 1, (x : ℂ) ^ (s - P.k - 1)) • P.g₀ := by
    rw [integral_sub, integral_smul, integral_smul_const, integral_smul_const]
    · apply Integrable.smul_const
      rw [← IntegrableOn, ← intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one]
      exact intervalIntegral.intervalIntegrable_cpow' h_re1
    · refine (Integrable.smul_const ?_ _).smul _
      rw [← IntegrableOn, ← intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one]
      exact intervalIntegral.intervalIntegrable_cpow' h_re2
  _ = _ := by simp_rw [← intervalIntegral.integral_of_le zero_le_one,
      integral_cpow (Or.inl h_re1), integral_cpow (Or.inl h_re2), ofReal_zero, ofReal_one,
      one_cpow, sub_add_cancel, zero_cpow fun h ↦ lt_irrefl _ (P.hk.le.trans_lt (zero_re ▸ h ▸ hs)),
      zero_cpow (sub_ne_zero.mpr (fun h ↦ lt_irrefl _ ((ofReal_re _) ▸ h ▸ hs)) : s - P.k ≠ 0),
      sub_zero, sub_eq_add_neg (_ •  _), ← mul_smul, ← neg_smul, mul_one_div, ← div_neg, neg_sub]


/-- An entire function which differs from the Mellin transform of `f - f₀`, where defined, by a
correction term of the form `A / s + B / (k - s)`. -/
def Λ₀ : ℂ → E := mellin P.f_modif


/-- A meromorphic function which agrees with the Mellin transform of `f - f₀` where defined -/
def Λ (s : ℂ) : E := P.Λ₀ s - (1 / s) • P.f₀ - (P.ε / (P.k - s)) • P.g₀


lemma Λ₀_eq (s : ℂ) : P.Λ₀ s = P.Λ s + (1 / s) • P.f₀ + (P.ε / (P.k - s)) • P.g₀ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    s : Complex
    ⊢ Eq (P.Λ₀ s) (HAdd.hAdd (HAdd.hAdd (P.Λ s) (HSMul.hSMul (HDiv.hDiv 1 s) P.f₀) …
  -/
  unfold Λ Λ₀
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    s : Complex
    ⊢ Eq (mellin P.f_modif s) (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (mellin  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma symm_Λ₀_eq (s : ℂ) :
    P.symm.Λ₀ s = P.symm.Λ s + (1 / s) • P.g₀ + (P.ε⁻¹ / (P.k - s)) • P.f₀ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    s : Complex
    ⊢ Eq (P.symm.Λ₀ s) (HAdd.hAdd (HAdd.hAdd (P.symm.Λ s) (HSMul.hSMul (HDiv.hDiv  …
  -/
  rw [P.symm.Λ₀_eq]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    s : Complex
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (P.symm.Λ s) (HSMul.hSMul (HDiv.hDiv 1 s) P.symm.f₀ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem differentiable_Λ₀ : Differentiable ℂ P.Λ₀ := P.toStrongFEPair.differentiable_Λ


theorem differentiableAt_Λ {s : ℂ} (hs : s ≠ 0 ∨ P.f₀ = 0) (hs' : s ≠ P.k ∨ P.g₀ = 0) :
    DifferentiableAt ℂ P.Λ s := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    s : Complex
    hs : Or (Ne s 0) (Eq P.f₀ 0)
    hs' : Or (Ne s ↑P.k) (Eq P.g₀ 0)
    ⊢ DifferentiableAt Complex P.Λ s
  -/
  refine ((P.differentiable_Λ₀ s).sub ?_).sub ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Or (Ne s 0) (Eq P.f₀ 0)
      hs' : Or (Ne s ↑P.k) (Eq P.g₀ 0)
      ⊢ DifferentiableAt Complex (fun y => HSMul.hSMul (HDiv.hDiv 1 y) P.f₀) s
    -/
  · rcases hs with hs | hs
      /-
        case refine_1.inl
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        P : WeakFEPair E
        s : Complex
        hs' : Or (Ne s ↑P.k) (Eq P.g₀ 0)
        hs : Ne s 0
        ⊢ DifferentiableAt Complex (fun y => HSMul.hSMul (HDiv.hDiv 1 y) P.f₀) s
      -/
    · simpa only [one_div] using (differentiableAt_inv hs).smul_const P.f₀
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        P : WeakFEPair E
        s : Complex
        hs' : Or (Ne s ↑P.k) (Eq P.g₀ 0)
        hs : Eq P.f₀ 0
        ⊢ DifferentiableAt Complex (fun y => HSMul.hSMul (HDiv.hDiv 1 y) P.f₀) s
      -/
    · simpa only [hs, smul_zero] using differentiableAt_const (0 : E)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Or (Ne s 0) (Eq P.f₀ 0)
      hs' : Or (Ne s ↑P.k) (Eq P.g₀ 0)
      ⊢ DifferentiableAt Complex (fun y => HSMul.hSMul (HDiv.hDiv P.ε (HSub.hSub (↑P …
    -/
  · rcases hs' with hs' | hs'
      /-
        case refine_2.inl
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        P : WeakFEPair E
        s : Complex
        hs : Or (Ne s 0) (Eq P.f₀ 0)
        hs' : Ne s ↑P.k
        ⊢ DifferentiableAt Complex (fun y => HSMul.hSMul (HDiv.hDiv P.ε (HSub.hSub (↑P …
      -/
    · apply DifferentiableAt.smul_const
      /-
        case refine_2.inl.hc
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        P : WeakFEPair E
        s : Complex
        hs : Or (Ne s 0) (Eq P.f₀ 0)
        hs' : Ne s ↑P.k
        ⊢ DifferentiableAt Complex (fun y => HDiv.hDiv P.ε (HSub.hSub (↑P.k) y)) s
      -/
      apply (differentiableAt_const _).div ((differentiableAt_const _).sub (differentiable_id _))
      /-
        case refine_2.inl.hc
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        P : WeakFEPair E
        s : Complex
        hs : Or (Ne s 0) (Eq P.f₀ 0)
        hs' : Ne s ↑P.k
        ⊢ Ne (HSub.hSub (↑P.k) s) 0
      -/
      rwa [sub_ne_zero, ne_comm]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        P : WeakFEPair E
        s : Complex
        hs : Or (Ne s 0) (Eq P.f₀ 0)
        hs' : Eq P.g₀ 0
        ⊢ DifferentiableAt Complex (fun y => HSMul.hSMul (HDiv.hDiv P.ε (HSub.hSub (↑P …
      -/
    · simpa only [hs', smul_zero] using differentiableAt_const (0 : E)
      /-
        🎉 no goals
      -/


/-- Relation between `Λ s` and the Mellin transform of `f - f₀`, where the latter is defined. -/
theorem hasMellin [CompleteSpace E]
    {s : ℂ} (hs : P.k < s.re) : HasMellin (P.f · - P.f₀) s (P.Λ s) := by
  have hc1 : MellinConvergent (P.f · - P.f₀) s :=
    let ⟨_, ht⟩ := exists_gt s.re
    mellinConvergent_of_isBigO_rpow (P.hf_int.sub (locallyIntegrableOn_const _)) (P.hf_top _) ht
      P.hf_zero' hs
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    hc1 : MellinConvergent (fun x => HSub.hSub (P.f x) P.f₀) s
    ⊢ HasMellin (fun x => HSub.hSub (P.f x) P.f₀) s (P.Λ s)
  -/
  refine ⟨hc1, ?_⟩
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    hc1 : MellinConvergent (fun x => HSub.hSub (P.f x) P.f₀) s
    ⊢ Eq (mellin (fun x => HSub.hSub (P.f x) P.f₀) s) (P.Λ s)
  -/
  have hc2 : HasMellin P.f_modif s (P.Λ₀ s) := P.toStrongFEPair.hasMellin s
  have hc3 : mellin (fun x ↦ f_modif P x - f P x + P.f₀) s =
    (1 / s) • P.f₀ + (P.ε / (↑P.k - s)) • P.g₀ := P.f_modif_aux2 hs
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    hc1 : MellinConvergent (fun x => HSub.hSub (P.f x) P.f₀) s
    hc2 : HasMellin P.f_modif s (P.Λ₀ s)
    hc3 : Eq (mellin (fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) s …
    ⊢ Eq (mellin (fun x => HSub.hSub (P.f x) P.f₀) s) (P.Λ s)
  -/
  have := (hasMellin_sub hc2.1 hc1).2
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    hc1 : MellinConvergent (fun x => HSub.hSub (P.f x) P.f₀) s
    hc2 : HasMellin P.f_modif s (P.Λ₀ s)
    hc3 : Eq (mellin (fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) s …
    this : Eq (mellin (fun t => HSub.hSub (P.f_modif t) (HSub.hSub (P.f t) P.f₀))  …
    ⊢ Eq (mellin (fun x => HSub.hSub (P.f x) P.f₀) s) (P.Λ s)
  -/
  simp_rw [← sub_add, hc3, eq_sub_iff_add_eq, ← eq_sub_iff_add_eq', ← sub_sub] at this
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    P : WeakFEPair E
    inst✝ : CompleteSpace E
    s : Complex
    hs : LT.lt P.k s.re
    hc1 : MellinConvergent (fun x => HSub.hSub (P.f x) P.f₀) s
    hc2 : HasMellin P.f_modif s (P.Λ₀ s)
    hc3 : Eq (mellin (fun x => HAdd.hAdd (HSub.hSub (P.f_modif x) (P.f x)) P.f₀) s …
    this : Eq (mellin (fun x => HSub.hSub (P.f x) P.f₀) s) (HSub.hSub (HSub.hSub ( …
    ⊢ Eq (mellin (fun x => HSub.hSub (P.f x) P.f₀) s) (P.Λ s)
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- Functional equation formulated for `Λ₀`. -/
theorem functional_equation₀ (s : ℂ) : P.Λ₀ (P.k - s) = P.ε • P.symm.Λ₀ s :=
  P.toStrongFEPair.functional_equation s


/-- Functional equation formulated for `Λ`. -/
theorem functional_equation (s : ℂ) :
    P.Λ (P.k - s) = P.ε • P.symm.Λ s := by
  linear_combination (norm := module) P.functional_equation₀ s - P.Λ₀_eq (P.k - s)
    + congr(P.ε • $(P.symm_Λ₀_eq s)) + congr(($(mul_inv_cancel₀ P.hε) / ((P.k:ℂ) - s)) • P.f₀)


/-- The residue of `Λ` at `s = k` is equal to `ε • g₀`. -/
theorem Λ_residue_k :
    Tendsto (fun s : ℂ ↦ (s - P.k) • P.Λ s) (𝓝[≠] P.k) (𝓝 (P.ε • P.g₀)) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Filter.Tendsto (fun s => HSMul.hSMul (HSub.hSub s ↑P.k) (P.Λ s)) (nhdsWithin …
  -/
  simp_rw [Λ, smul_sub, (by simp : 𝓝 (P.ε • P.g₀) = 𝓝 (0 - 0 - -P.ε • P.g₀))]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HSub.hSub (HSMul.hSMul (HSub.hSub s ↑P.k …
  -/
  refine ((Tendsto.sub ?_ ?_).mono_left nhdsWithin_le_nhds).sub ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul (HSub.hSub s ↑P.k) (P.Λ₀ s)) (nhds ↑P.k …
    -/
  · rw [(by rw [sub_self, zero_smul] : 𝓝 0 = 𝓝 ((P.k - P.k : ℂ) • P.Λ₀ P.k))]
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul (HSub.hSub s ↑P.k) (P.Λ₀ s)) (nhds ↑P.k …
    -/
    apply ((continuous_sub_right _).smul P.differentiable_Λ₀.continuous).tendsto
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul (HSub.hSub s ↑P.k) (HSMul.hSMul (HDiv.h …
    -/
  · rw [(by rw [sub_self, zero_smul] : 𝓝 0 = 𝓝 ((P.k - P.k : ℂ) • (1 / P.k : ℂ) • P.f₀))]
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul (HSub.hSub s ↑P.k) (HSMul.hSMul (HDiv.h …
    -/
    refine (continuous_sub_right _).continuousAt.smul (ContinuousAt.smul ?_ continuousAt_const)
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ ContinuousAt (HDiv.hDiv 1) ↑P.k
    -/
    exact continuousAt_const.div continuousAt_id (ofReal_ne_zero.mpr P.hk.ne')
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul (HSub.hSub s ↑P.k) (HSMul.hSMul (HDiv.h …
    -/
  · refine (tendsto_const_nhds.mono_left nhdsWithin_le_nhds).congr' ?_
    /-
      case refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ (nhdsWithin (↑P.k) (HasCompl.compl (Singleton.singleton ↑P.k))).EventuallyEq …
    -/
    refine eventually_nhdsWithin_of_forall (fun s (hs : s ≠ P.k) ↦ ?_)
    /-
      case refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Ne s ↑P.k
      ⊢ Eq ((fun x => HSMul.hSMul (Neg.neg P.ε) P.g₀) s) ((fun s => HSMul.hSMul (HSu …
    -/
    match_scalars
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Ne s ↑P.k
      ⊢ Eq (HMul.hMul (Neg.neg P.ε) 1) (HMul.hMul (HSub.hSub s ↑P.k) (HMul.hMul (HDi …
    -/
    field_simp [sub_ne_zero.mpr hs.symm]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Ne s ↑P.k
      ⊢ Eq (Neg.neg (HMul.hMul P.ε (HSub.hSub (↑P.k) s))) (HMul.hMul (HSub.hSub s ↑P …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The residue of `Λ` at `s = 0` is equal to `-f₀`. -/
theorem Λ_residue_zero :
    Tendsto (fun s : ℂ ↦ s • P.Λ s) (𝓝[≠] 0) (𝓝 (-P.f₀)) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Filter.Tendsto (fun s => HSMul.hSMul s (P.Λ s)) (nhdsWithin 0 (HasCompl.comp …
  -/
  simp_rw [Λ, smul_sub, (by simp : 𝓝 (-P.f₀) = 𝓝 (((0 : ℂ) • P.Λ₀ 0) - P.f₀ - 0))]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    P : WeakFEPair E
    ⊢ Filter.Tendsto (fun s => HSub.hSub (HSub.hSub (HSMul.hSMul s (P.Λ₀ s)) (HSMu …
  -/
  refine ((Tendsto.mono_left ?_ nhdsWithin_le_nhds).sub ?_).sub ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul s (P.Λ₀ s)) (nhds 0) (nhds (HSMul.hSMul …
    -/
  · exact (continuous_id.smul P.differentiable_Λ₀.continuous).tendsto _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul s (HSMul.hSMul (HDiv.hDiv 1 s) P.f₀)) ( …
    -/
  · refine (tendsto_const_nhds.mono_left nhdsWithin_le_nhds).congr' ?_
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).EventuallyEq (fun x  …
    -/
    refine eventually_nhdsWithin_of_forall (fun s (hs : s ≠ 0) ↦ ?_)
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Ne s 0
      ⊢ Eq ((fun x => P.f₀) s) ((fun s => HSMul.hSMul s (HSMul.hSMul (HDiv.hDiv 1 s) …
    -/
    match_scalars
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      s : Complex
      hs : Ne s 0
      ⊢ Eq 1 (HMul.hMul s (HMul.hMul (HDiv.hDiv 1 s) 1))
    -/
    field_simp [sub_ne_zero.mpr hs.symm]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      P : WeakFEPair E
      ⊢ Filter.Tendsto (fun s => HSMul.hSMul s (HSMul.hSMul (HDiv.hDiv P.ε (HSub.hSu …
    -/
  · rw [show 𝓝 0 = 𝓝 ((0 : ℂ) • (P.ε / (P.k - 0 : ℂ)) • P.g₀) by rw [zero_smul]]
    exact (continuousAt_id.smul ((continuousAt_const.div ((continuous_sub_left _).continuousAt)
      (by simpa using P.hk.ne')).smul continuousAt_const)).mono_left nhdsWithin_le_nhds


