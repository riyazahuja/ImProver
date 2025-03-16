/-- The radius of a formal multilinear series is equal to
$\liminf_{n\to\infty} \frac{1}{\sqrt[n]{‖p n‖}}$. The actual statement uses `ℝ≥0` and some
coercions. -/
theorem radius_eq_liminf :
    p.radius = liminf (fun n => (1 / (‖p n‖₊ ^ (1 / (n : ℝ)) : ℝ≥0) : ℝ≥0∞)) atTop := by
  -- Porting note: added type ascription to make elaborated statement match Lean 3 version
  have :
    ∀ (r : ℝ≥0) {n : ℕ},
      0 < n → ((r : ℝ≥0∞) ≤ 1 / ↑(‖p n‖₊ ^ (1 / (n : ℝ))) ↔ ‖p n‖₊ * r ^ n ≤ 1) := by
    intro r n hn
    have : 0 < (n : ℝ) := Nat.cast_pos.2 hn
    conv_lhs =>
      rw [one_div, ENNReal.le_inv_iff_mul_le, ← ENNReal.coe_mul, ENNReal.coe_le_one_iff, one_div, ←
        NNReal.rpow_one r, ← mul_inv_cancel₀ this.ne', NNReal.rpow_mul, ← NNReal.mul_rpow, ←
        NNReal.one_rpow n⁻¹, NNReal.rpow_le_rpow_iff (inv_pos.2 this), mul_comm,
        NNReal.rpow_natCast]
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
    this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
    ⊢ Eq p.radius (Filter.liminf (fun n => HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnorm  …
  -/
  apply le_antisymm <;> refine ENNReal.le_of_forall_nnreal_lt fun r hr => ?_
  · have := ((TFAE_exists_lt_isLittleO_pow (fun n => ‖p n‖ * r ^ n) 1).out 1 7).1
      (p.isLittleO_of_lt_radius hr)
    /-
      case a
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      this✝ : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(H …
      r : NNReal
      hr : LT.lt (↑r) p.radius
      this : Exists fun a => And (Membership.mem (Set.Ioo 0 1) a) (Filter.Eventually …
      ⊢ LE.le (↑r) (Filter.liminf (fun n => HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnorm ( …
    -/
    obtain ⟨a, ha, H⟩ := this
    /-
      case a.intro.intro
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
      r : NNReal
      hr : LT.lt (↑r) p.radius
      a : Real
      ha : Membership.mem (Set.Ioo 0 1) a
      H : Filter.Eventually (fun n => LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow. …
      ⊢ LE.le (↑r) (Filter.liminf (fun n => HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnorm ( …
    -/
    apply le_liminf_of_le
      /-
        case a.intro.intro.hf
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
        r : NNReal
        hr : LT.lt (↑r) p.radius
        a : Real
        ha : Membership.mem (Set.Ioo 0 1) a
        H : Filter.Eventually (fun n => LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow. …
        ⊢ autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) Filter.atTop f …
      -/
    · infer_param
      /-
        🎉 no goals
      -/
      /-
        case a.intro.intro.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
        r : NNReal
        hr : LT.lt (↑r) p.radius
        a : Real
        ha : Membership.mem (Set.Ioo 0 1) a
        H : Filter.Eventually (fun n => LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow. …
        ⊢ Filter.Eventually (fun n => LE.le (↑r) (HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnno …
      -/
    · rw [← eventually_map]
      refine
        H.mp ((eventually_gt_atTop 0).mono fun n hn₀ hn => (this _ hn₀).2 (NNReal.coe_le_coe.1 ?_))
      /-
        case a.intro.intro.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
        r : NNReal
        hr : LT.lt (↑r) p.radius
        a : Real
        ha : Membership.mem (Set.Ioo 0 1) a
        H : Filter.Eventually (fun n => LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow. …
        n : Nat
        hn₀ : LT.lt 0 n
        hn : LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow.hPow (↑r) n))) (HPow.hPow a …
        ⊢ LE.le ↑(HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow r n)) ↑1
      -/
      push_cast
      /-
        case a.intro.intro.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
        r : NNReal
        hr : LT.lt (↑r) p.radius
        a : Real
        ha : Membership.mem (Set.Ioo 0 1) a
        H : Filter.Eventually (fun n => LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow. …
        n : Nat
        hn₀ : LT.lt 0 n
        hn : LE.le (abs (HMul.hMul (Norm.norm (p n)) (HPow.hPow (↑r) n))) (HPow.hPow a …
        ⊢ LE.le (HMul.hMul (Norm.norm (p n)) (HPow.hPow (↑r) n)) 1
      -/
      exact (le_abs_self _).trans (hn.trans (pow_le_one₀ ha.1.le ha.2.le))
      /-
        🎉 no goals
      -/
    /-
      case a
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
      r : NNReal
      hr : LT.lt (↑r) (Filter.liminf (fun n => HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnor …
      ⊢ LE.le (↑r) p.radius
    -/
  · refine p.le_radius_of_isBigO (IsBigO.of_bound 1 ?_)
    /-
      case a
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
      r : NNReal
      hr : LT.lt (↑r) (Filter.liminf (fun n => HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnor …
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HMul.hMul (Norm.norm (p x)) (H …
    -/
    refine (eventually_lt_of_lt_liminf hr).mp ((eventually_gt_atTop 0).mono fun n hn₀ hn => ?_)
    /-
      case a
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      this : ∀ (r : NNReal) {n : Nat}, LT.lt 0 n → Iff (LE.le (↑r) (HDiv.hDiv 1 ↑(HP …
      r : NNReal
      hr : LT.lt (↑r) (Filter.liminf (fun n => HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnor …
      n : Nat
      hn₀ : LT.lt 0 n
      hn : LT.lt (↑r) (HDiv.hDiv 1 ↑(HPow.hPow (NNNorm.nnnorm (p n)) (HDiv.hDiv 1 ↑n …
      ⊢ LE.le (Norm.norm (HMul.hMul (Norm.norm (p n)) (HPow.hPow (↑r) n))) (HMul.hMu …
    -/
    simpa using NNReal.coe_le_coe.2 ((this _ hn₀).1 hn.le)
    /-
      🎉 no goals
    -/


