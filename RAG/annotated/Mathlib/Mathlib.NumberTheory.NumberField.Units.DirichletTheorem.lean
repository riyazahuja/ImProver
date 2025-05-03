/-- The distinguished infinite place. -/
def w₀ : InfinitePlace K := (inferInstance : Nonempty (InfinitePlace K)).some


/-- The logarithmic embedding of the units (seen as an `Additive` group). -/
def _root_.NumberField.Units.logEmbedding :
    Additive ((𝓞 K)ˣ) →+ ({w : InfinitePlace K // w ≠ w₀} → ℝ) :=
{ toFun := fun x w => mult w.val * Real.log (w.val ↑x.toMul)
                  /-
                    K : Type u_1
                    inst✝¹ : Field K
                    inst✝ : NumberField K
                    ⊢ Eq ((fun x w => HMul.hMul (↑(↑w).mult) (Real.log (↑w ((algebraMap (NumberFie …
                  -/
  map_zero' := by simp; rfl
                        /-
                          🎉 no goals
                        -/
                            /-
                              K : Type u_1
                              inst✝¹ : Field K
                              inst✝ : NumberField K
                              x✝¹ x✝ : Additive (Units (NumberField.RingOfIntegers K))
                              ⊢ Eq ({ toFun := fun x w => HMul.hMul (↑(↑w).mult) (Real.log (↑w ((algebraMap  …
                            -/
  map_add' := fun _ _ => by simp [Real.log_mul, mul_add]; rfl }
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem logEmbedding_component (x : (𝓞 K)ˣ) (w : {w : InfinitePlace K // w ≠ w₀}) :
    (logEmbedding K (Additive.ofMul x)) w = mult w.val * Real.log (w.val x) := rfl


open scoped Classical in
theorem sum_logEmbedding_component (x : (𝓞 K)ˣ) :
    ∑ w, logEmbedding K (Additive.ofMul x) w =
      - mult (w₀ : InfinitePlace K) * Real.log (w₀ (x : K)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    ⊢ Eq (Finset.univ.sum fun w => (NumberField.Units.logEmbedding K) (Additive.of …
  -/
  have h := congr_arg Real.log (prod_eq_abs_norm (x : K))
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    h : Eq (Real.log (Finset.univ.prod fun w => HPow.hPow (w ((algebraMap (NumberF …
    ⊢ Eq (Finset.univ.sum fun w => (NumberField.Units.logEmbedding K) (Additive.of …
  -/
  rw [Units.norm, Rat.cast_one, Real.log_one, Real.log_prod] at h
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      h : Eq (Finset.univ.sum fun i => Real.log (HPow.hPow (i ((algebraMap (NumberFi …
      ⊢ Eq (Finset.univ.sum fun w => (NumberField.Units.logEmbedding K) (Additive.of …
    -/
  · simp_rw [Real.log_pow] at h
    rw [← insert_erase (mem_univ w₀), sum_insert (not_mem_erase w₀ univ), add_comm,
      add_eq_zero_iff_eq_neg] at h
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      h : Eq ((Finset.univ.erase NumberField.Units.dirichletUnitTheorem.w₀).sum fun  …
      ⊢ Eq (Finset.univ.sum fun w => (NumberField.Units.logEmbedding K) (Additive.of …
    -/
    convert h using 1
      /-
        case h.e'_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : Units (NumberField.RingOfIntegers K)
        h : Eq ((Finset.univ.erase NumberField.Units.dirichletUnitTheorem.w₀).sum fun  …
        ⊢ Eq (Finset.univ.sum fun w => (NumberField.Units.logEmbedding K) (Additive.of …
      -/
    · refine (sum_subtype _ (fun w => ?_) (fun w => (mult w) * (Real.log (w (x : K))))).symm
      /-
        case h.e'_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : Units (NumberField.RingOfIntegers K)
        h : Eq ((Finset.univ.erase NumberField.Units.dirichletUnitTheorem.w₀).sum fun  …
        w : NumberField.InfinitePlace K
        ⊢ Iff (Membership.mem (Finset.univ.erase NumberField.Units.dirichletUnitTheore …
      -/
      exact ⟨ne_of_mem_erase, fun h => mem_erase_of_ne_of_mem h (mem_univ w)⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : Units (NumberField.RingOfIntegers K)
        h : Eq ((Finset.univ.erase NumberField.Units.dirichletUnitTheorem.w₀).sum fun  …
        ⊢ Eq (HMul.hMul (Neg.neg ↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (Rea …
      -/
    · norm_num
      /-
        🎉 no goals
      -/
    /-
      case hf
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      h : Eq (Real.log (Finset.univ.prod fun w => HPow.hPow (w ((algebraMap (NumberF …
      ⊢ ∀ (x_1 : NumberField.InfinitePlace K), Membership.mem Finset.univ x_1 → Ne ( …
    -/
  · exact fun w _ => pow_ne_zero _ (AbsoluteValue.ne_zero _ (coe_ne_zero x))
    /-
      🎉 no goals
    -/


theorem mult_log_place_eq_zero {x : (𝓞 K)ˣ} {w : InfinitePlace K} :
    mult w * Real.log (w x) = 0 ↔ w x = 1 := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : Units (NumberField.RingOfIntegers K)
    w : NumberField.InfinitePlace K
    ⊢ Iff (Eq (HMul.hMul (↑w.mult) (Real.log (w ((algebraMap (NumberField.RingOfIn …
  -/
  rw [mul_eq_zero, or_iff_right, Real.log_eq_zero, or_iff_right, or_iff_left]
    /-
      K : Type u_1
      inst✝ : Field K
      x : Units (NumberField.RingOfIntegers K)
      w : NumberField.InfinitePlace K
      ⊢ Not (Eq (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x)) (-1))
    -/
  · linarith [(apply_nonneg _ _ : 0 ≤ w x)]
    /-
      🎉 no goals
    -/
    /-
      K : Type u_1
      inst✝ : Field K
      x : Units (NumberField.RingOfIntegers K)
      w : NumberField.InfinitePlace K
      ⊢ Not (Eq (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x)) 0)
    -/
  · simp only [ne_eq, map_eq_zero, coe_ne_zero x, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      K : Type u_1
      inst✝ : Field K
      x : Units (NumberField.RingOfIntegers K)
      w : NumberField.InfinitePlace K
      ⊢ Not (Eq (↑w.mult) 0)
    -/
  · refine (ne_of_gt ?_)
    /-
      K : Type u_1
      inst✝ : Field K
      x : Units (NumberField.RingOfIntegers K)
      w : NumberField.InfinitePlace K
      ⊢ LT.lt 0 ↑w.mult
    -/
                             /-
                               🎉 no goals
                             -/
    rw [mult]; split_ifs <;> norm_num
                             /-
                               🎉 no goals
                             -/


theorem logEmbedding_eq_zero_iff {x : (𝓞 K)ˣ} :
    logEmbedding K (Additive.ofMul x) = 0 ↔ x ∈ torsion K := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    ⊢ Iff (Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0) (Membersh …
  -/
  rw [mem_torsion]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    ⊢ Iff (Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0) (∀ (w : N …
  -/
  refine ⟨fun h w => ?_, fun h => ?_⟩
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      h : Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0
      w : NumberField.InfinitePlace K
      ⊢ Eq (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x)) 1
    -/
  · by_cases hw : w = w₀
    · suffices -mult w₀ * Real.log (w₀ (x : K)) = 0 by
        rw [neg_mul, neg_eq_zero, ← hw] at this
        exact mult_log_place_eq_zero.mp this
      /-
        case pos
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : Units (NumberField.RingOfIntegers K)
        h : Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0
        w : NumberField.InfinitePlace K
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        ⊢ Eq (HMul.hMul (Neg.neg ↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (Rea …
      -/
      rw [← sum_logEmbedding_component, sum_eq_zero]
      /-
        case pos
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : Units (NumberField.RingOfIntegers K)
        h : Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0
        w : NumberField.InfinitePlace K
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        ⊢ ∀ (x_1 : Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀), M …
      -/
      exact fun w _ => congrFun h w
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : Units (NumberField.RingOfIntegers K)
        h : Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0
        w : NumberField.InfinitePlace K
        hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
        ⊢ Eq (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x)) 1
      -/
    · exact mult_log_place_eq_zero.mp (congrFun h ⟨w, hw⟩)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      h : ∀ (w : NumberField.InfinitePlace K), Eq (w ((algebraMap (NumberField.RingO …
      ⊢ Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x)) 0
    -/
  · ext w
    /-
      case refine_2.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      h : ∀ (w : NumberField.InfinitePlace K), Eq (w ((algebraMap (NumberField.RingO …
      w : Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀
      ⊢ Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul x) w) (0 w)
    -/
    rw [logEmbedding_component, h w.val, Real.log_one, mul_zero, Pi.zero_apply]
    /-
      🎉 no goals
    -/


open scoped Classical in
theorem logEmbedding_component_le {r : ℝ} {x : (𝓞 K)ˣ} (hr : 0 ≤ r) (h : ‖logEmbedding K x‖ ≤ r)
    (w : {w : InfinitePlace K // w ≠ w₀}) : |logEmbedding K (Additive.ofMul x) w| ≤ r := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    r : Real
    x : Units (NumberField.RingOfIntegers K)
    hr : LE.le 0 r
    h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) x)) r
    w : Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀
    ⊢ LE.le (abs ((NumberField.Units.logEmbedding K) (Additive.ofMul x) w)) r
  -/
  lift r to NNReal using hr
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    w : Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀
    r : NNReal
    h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) x)) ↑r
    ⊢ LE.le (abs ((NumberField.Units.logEmbedding K) (Additive.ofMul x) w)) ↑r
  -/
  simp_rw [Pi.norm_def, NNReal.coe_le_coe, Finset.sup_le_iff, ← NNReal.coe_le_coe] at h
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    w : Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀
    r : NNReal
    h : ∀ (b : Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀), M …
    ⊢ LE.le (abs ((NumberField.Units.logEmbedding K) (Additive.ofMul x) w)) ↑r
  -/
  exact h w (mem_univ _)
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem log_le_of_logEmbedding_le {r : ℝ} {x : (𝓞 K)ˣ} (hr : 0 ≤ r)
    (h : ‖logEmbedding K (Additive.ofMul x)‖ ≤ r) (w : InfinitePlace K) :
    |Real.log (w x)| ≤ (Fintype.card (InfinitePlace K)) * r := by
  have tool : ∀ x : ℝ, 0 ≤ x → x ≤ mult w * x := fun x hx => by
    nth_rw 1 [← one_mul x]
    refine mul_le_mul ?_ le_rfl hx ?_
    all_goals { rw [mult]; split_ifs <;> norm_num }
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    r : Real
    x : Units (NumberField.RingOfIntegers K)
    hr : LE.le 0 r
    h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
    w : NumberField.InfinitePlace K
    tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
    ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
  -/
  by_cases hw : w = w₀
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
  · have hyp := congr_arg (‖·‖) (sum_logEmbedding_component x).symm
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
      hyp : Eq ((fun x => Norm.norm x) (HMul.hMul (Neg.neg ↑NumberField.Units.dirich …
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
    replace hyp := (le_of_eq hyp).trans (norm_sum_le _ _)
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
      hyp : LE.le ((fun x => Norm.norm x) (HMul.hMul (Neg.neg ↑NumberField.Units.dir …
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
    simp_rw [norm_mul, norm_neg, Real.norm_eq_abs, Nat.abs_cast] at hyp
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
      hyp : LE.le (HMul.hMul (↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (abs  …
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
    refine (le_trans ?_ hyp).trans ?_
      /-
        case pos.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        hyp : LE.le (HMul.hMul (↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (abs  …
        ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
      -/
    · rw [← hw]
      /-
        case pos.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        hyp : LE.le (HMul.hMul (↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (abs  …
        ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
      -/
      exact tool _ (abs_nonneg _)
      /-
        🎉 no goals
      -/
    · refine (sum_le_card_nsmul univ _ _
        (fun w _ => logEmbedding_component_le hr h w)).trans ?_
      /-
        case pos.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        hyp : LE.le (HMul.hMul (↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (abs  …
        ⊢ LE.le (HSMul.hSMul Finset.univ.card r) (HMul.hMul (↑(Fintype.card (NumberFie …
      -/
      rw [nsmul_eq_mul]
      /-
        case pos.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        hyp : LE.le (HMul.hMul (↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (abs  …
        ⊢ LE.le (HMul.hMul (↑Finset.univ.card) r) (HMul.hMul (↑(Fintype.card (NumberFi …
      -/
      refine mul_le_mul ?_ le_rfl hr (Fintype.card (InfinitePlace K)).cast_nonneg
      /-
        case pos.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Eq w NumberField.Units.dirichletUnitTheorem.w₀
        hyp : LE.le (HMul.hMul (↑NumberField.Units.dirichletUnitTheorem.w₀.mult) (abs  …
        ⊢ LE.le ↑Finset.univ.card ↑(Fintype.card (NumberField.InfinitePlace K))
      -/
      simp [card_univ]
      /-
        🎉 no goals
      -/
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
  · have hyp := logEmbedding_component_le hr h ⟨w, hw⟩
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
      hyp : LE.le (abs ((NumberField.Units.logEmbedding K) (Additive.ofMul (Additive …
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
    rw [logEmbedding_component, abs_mul, Nat.abs_cast] at hyp
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      x : Units (NumberField.RingOfIntegers K)
      hr : LE.le 0 r
      h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
      w : NumberField.InfinitePlace K
      tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
      hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
      hyp : LE.le (HMul.hMul (↑(↑⟨w, hw⟩).mult) (abs (Real.log (↑⟨w, hw⟩ ((algebraMa …
      ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
    -/
    refine (le_trans ?_ hyp).trans ?_
      /-
        case neg.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
        hyp : LE.le (HMul.hMul (↑(↑⟨w, hw⟩).mult) (abs (Real.log (↑⟨w, hw⟩ ((algebraMa …
        ⊢ LE.le (abs (Real.log (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑x))) …
      -/
    · exact tool _ (abs_nonneg _)
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
        hyp : LE.le (HMul.hMul (↑(↑⟨w, hw⟩).mult) (abs (Real.log (↑⟨w, hw⟩ ((algebraMa …
        ⊢ LE.le r (HMul.hMul (↑(Fintype.card (NumberField.InfinitePlace K))) r)
      -/
    · nth_rw 1 [← one_mul r]
      /-
        case neg.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        x : Units (NumberField.RingOfIntegers K)
        hr : LE.le 0 r
        h : LE.le (Norm.norm ((NumberField.Units.logEmbedding K) (Additive.ofMul x))) r
        w : NumberField.InfinitePlace K
        tool : ∀ (x : Real), LE.le 0 x → LE.le x (HMul.hMul (↑w.mult) x)
        hw : Not (Eq w NumberField.Units.dirichletUnitTheorem.w₀)
        hyp : LE.le (HMul.hMul (↑(↑⟨w, hw⟩).mult) (abs (Real.log (↑⟨w, hw⟩ ((algebraMa …
        ⊢ LE.le (HMul.hMul 1 r) (HMul.hMul (↑(Fintype.card (NumberField.InfinitePlace  …
      -/
      exact mul_le_mul (Nat.one_le_cast.mpr Fintype.card_pos) (le_of_eq rfl) hr (Nat.cast_nonneg _)
      /-
        🎉 no goals
      -/


/-- The lattice formed by the image of the logarithmic embedding. -/
noncomputable def _root_.NumberField.Units.unitLattice :
    Submodule ℤ ({w : InfinitePlace K // w ≠ w₀} → ℝ) :=
  Submodule.map (logEmbedding K).toIntLinearMap ⊤


open scoped Classical in
theorem unitLattice_inter_ball_finite (r : ℝ) :
    ((unitLattice K : Set ({ w : InfinitePlace K // w ≠ w₀} → ℝ)) ∩
      Metric.closedBall 0 r).Finite := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    r : Real
    ⊢ (Inter.inter (↑(NumberField.Units.unitLattice K)) (Metric.closedBall 0 r)).F …
  -/
  obtain hr | hr := lt_or_le r 0
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LT.lt r 0
      ⊢ (Inter.inter (↑(NumberField.Units.unitLattice K)) (Metric.closedBall 0 r)).F …
    -/
  · convert Set.finite_empty
    /-
      case h.e'_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LT.lt r 0
      ⊢ Eq (Inter.inter (↑(NumberField.Units.unitLattice K)) (Metric.closedBall 0 r) …
    -/
    rw [Metric.closedBall_eq_empty.mpr hr]
    /-
      case h.e'_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LT.lt r 0
      ⊢ Eq (Inter.inter (↑(NumberField.Units.unitLattice K)) EmptyCollection.emptyCo …
    -/
    exact Set.inter_empty _
    /-
      🎉 no goals
    -/
  · suffices {x : (𝓞 K)ˣ | IsIntegral ℤ (x : K) ∧
        ∀ (φ : K →+* ℂ), ‖φ x‖ ≤ Real.exp ((Fintype.card (InfinitePlace K)) * r)}.Finite by
      refine (Set.Finite.image (logEmbedding K) this).subset ?_
      rintro _ ⟨⟨x, ⟨_, rfl⟩⟩, hx⟩
      refine ⟨x, ⟨x.val.prop, (le_iff_le _ _).mp (fun w => (Real.log_le_iff_le_exp ?_).mp ?_)⟩, rfl⟩
      · exact pos_iff.mpr (coe_ne_zero x)
      · rw [mem_closedBall_zero_iff] at hx
        exact (le_abs_self _).trans (log_le_of_logEmbedding_le hr hx w)
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LE.le 0 r
      ⊢ (setOf fun x => And (IsIntegral Int ((algebraMap (NumberField.RingOfIntegers …
    -/
    refine Set.Finite.of_finite_image ?_ (coe_injective K).injOn
    refine (Embeddings.finite_of_norm_le K ℂ
        (Real.exp ((Fintype.card (InfinitePlace K)) * r))).subset ?_
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LE.le 0 r
      ⊢ HasSubset.Subset (Set.image (fun x => (algebraMap (NumberField.RingOfInteger …
    -/
    rintro _ ⟨x, ⟨⟨h_int, h_le⟩, rfl⟩⟩
    /-
      case inr.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LE.le 0 r
      x : Units (NumberField.RingOfIntegers K)
      h_int : IsIntegral Int ((algebraMap (NumberField.RingOfIntegers K) K) ↑x)
      h_le : ∀ (φ : RingHom K Complex), LE.le (Norm.norm (φ ((algebraMap (NumberFiel …
      ⊢ Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K Comp …
    -/
    exact ⟨h_int, h_le⟩
    /-
      🎉 no goals
    -/


include hB in
/-- This result shows that there always exists a next term in the sequence. -/
theorem seq_next {x : 𝓞 K} (hx : x ≠ 0) :
    ∃ y : 𝓞 K, y ≠ 0 ∧
      (∀ w, w ≠ w₁ → w y < w x) ∧
      |Algebra.norm ℚ (y : K)| ≤ B := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : Nat
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    x : NumberField.RingOfIntegers K
    hx : Ne x 0
    ⊢ Exists fun y => And (Ne y 0) (And (∀ (w : NumberField.InfinitePlace K), Ne w …
  -/
  have hx' := RingOfIntegers.coe_ne_zero_iff.mpr hx
  let f : InfinitePlace K → ℝ≥0 :=
    fun w => ⟨(w x) / 2, div_nonneg (AbsoluteValue.nonneg _ _) (by norm_num)⟩
  suffices ∀ w, w ≠ w₁ → f w ≠ 0 by
    obtain ⟨g, h_geqf, h_gprod⟩ := adjust_f K B this
    obtain ⟨y, h_ynz, h_yle⟩ := exists_ne_zero_mem_ringOfIntegers_lt K (f := g)
      (by rw [convexBodyLT_volume]; convert hB; exact congr_arg ((↑) : NNReal → ENNReal) h_gprod)
    refine ⟨y, h_ynz, fun w hw => (h_geqf w hw ▸ h_yle w).trans ?_, ?_⟩
    · rw [← Rat.cast_le (K := ℝ), Rat.cast_natCast]
      calc
        _ = ∏ w : InfinitePlace K, w (algebraMap _ K y) ^ mult w :=
          (prod_eq_abs_norm (algebraMap _ K y)).symm
        _ ≤ ∏ w : InfinitePlace K, (g w : ℝ) ^ mult w := by gcongr with w; exact (h_yle w).le
        _ ≤ (B : ℝ) := by
          simp_rw [← NNReal.coe_pow, ← NNReal.coe_prod]
          exact le_of_eq (congr_arg toReal h_gprod)
    · refine div_lt_self ?_ (by norm_num)
      exact pos_iff.mpr hx'
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : Nat
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    x : NumberField.RingOfIntegers K
    hx : Ne x 0
    hx' : Ne ((algebraMap (NumberField.RingOfIntegers K) K) x) 0
    f : NumberField.InfinitePlace K → NNReal := fun w => ⟨HDiv.hDiv (w ↑x) 2, ⋯⟩
    ⊢ ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
  -/
  intro _ _
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : Nat
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    x : NumberField.RingOfIntegers K
    hx : Ne x 0
    hx' : Ne ((algebraMap (NumberField.RingOfIntegers K) K) x) 0
    f : NumberField.InfinitePlace K → NNReal := fun w => ⟨HDiv.hDiv (w ↑x) 2, ⋯⟩
    w✝ : NumberField.InfinitePlace K
    a✝ : Ne w✝ w₁
    ⊢ Ne (f w✝) 0
  -/
  rw [ne_eq, Nonneg.mk_eq_zero, div_eq_zero_iff, map_eq_zero, not_or]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : Nat
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    x : NumberField.RingOfIntegers K
    hx : Ne x 0
    hx' : Ne ((algebraMap (NumberField.RingOfIntegers K) K) x) 0
    f : NumberField.InfinitePlace K → NNReal := fun w => ⟨HDiv.hDiv (w ↑x) 2, ⋯⟩
    w✝ : NumberField.InfinitePlace K
    a✝ : Ne w✝ w₁
    ⊢ And (Not (Eq (↑x) 0)) (Not (Eq 2 0))
  -/
  exact ⟨hx', by norm_num⟩
  /-
    🎉 no goals
  -/


/-- An infinite sequence of nonzero algebraic integers of `K` satisfying the following properties:
• `seq n` is nonzero;
• for `w : InfinitePlace K`, `w ≠ w₁ → w (seq n+1) < w (seq n)`;
• `∣norm (seq n)∣ ≤ B`. -/
def seq : ℕ → { x : 𝓞 K // x ≠ 0 }
                /-
                  K : Type u_1
                  inst✝¹ : Field K
                  inst✝ : NumberField K
                  w₁ : NumberField.InfinitePlace K
                  B : Nat
                  hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
                  ⊢ Ne 1 0
                -/
  | 0 => ⟨1, by norm_num⟩
                /-
                  🎉 no goals
                -/
  | n + 1 =>
    ⟨(seq_next K w₁ hB (seq n).prop).choose, (seq_next K w₁ hB (seq n).prop).choose_spec.1⟩


/-- The terms of the sequence are nonzero. -/
theorem seq_ne_zero (n : ℕ) : algebraMap (𝓞 K) K (seq K w₁ hB n) ≠ 0 :=
  RingOfIntegers.coe_ne_zero_iff.mpr (seq K w₁ hB n).prop


/-- The sequence is strictly decreasing at infinite places distinct from `w₁`. -/
theorem seq_decreasing {n m : ℕ} (h : n < m) (w : InfinitePlace K) (hw : w ≠ w₁) :
    w (algebraMap (𝓞 K) K (seq K w₁ hB m)) < w (algebraMap (𝓞 K) K (seq K w₁ hB n)) := by
  induction m with
  | zero =>
      exfalso
      exact Nat.not_succ_le_zero n h
  | succ m m_ih =>
      cases eq_or_lt_of_le (Nat.le_of_lt_succ h) with
      | inl hr =>
          rw [hr]
          exact (seq_next K w₁ hB (seq K w₁ hB m).prop).choose_spec.2.1 w hw
      | inr hr =>
          refine lt_trans ?_ (m_ih hr)
          exact (seq_next K w₁ hB (seq K w₁ hB m).prop).choose_spec.2.1 w hw


/-- The terms of the sequence have norm bounded by `B`. -/
theorem seq_norm_le (n : ℕ) :
    Int.natAbs (Algebra.norm ℤ (seq K w₁ hB n : 𝓞 K)) ≤ B := by
  cases n with
  | zero =>
      have : 1 ≤ B := by
        contrapose! hB
        simp only [Nat.lt_one_iff.mp hB, CharP.cast_eq_zero, mul_zero, zero_le]
      simp only [ne_eq, seq, map_one, Int.natAbs_one, this]
  | succ n =>
      rw [← Nat.cast_le (α := ℚ), Int.cast_natAbs, Int.cast_abs, Algebra.coe_norm_int]
      exact (seq_next K w₁ hB (seq K w₁ hB n).prop).choose_spec.2.2


/-- Construct a unit associated to the place `w₁`. The family, for `w₁ ≠ w₀`, formed by the
image by the `logEmbedding` of these units is `ℝ`-linearly independent, see
`unitLattice_span_eq_top`. -/
theorem exists_unit (w₁ : InfinitePlace K) :
    ∃ u : (𝓞 K)ˣ, ∀ w : InfinitePlace K, w ≠ w₁ → Real.log (w u) < 0 := by
  obtain ⟨B, hB⟩ : ∃ B : ℕ, minkowskiBound K 1 < (convexBodyLTFactor K) * B := by
    conv => congr; ext; rw [mul_comm]
    exact ENNReal.exists_nat_mul_gt (ENNReal.coe_ne_zero.mpr (convexBodyLTFactor_ne_zero K))
      (ne_of_lt (minkowskiBound_lt_top K 1))
  rsuffices ⟨n, m, hnm, h⟩ : ∃ n m, n < m ∧
      (Ideal.span ({ (seq K w₁ hB n : 𝓞 K) }) = Ideal.span ({ (seq K w₁ hB m : 𝓞 K) }))
    /-
      case intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w₁ : NumberField.InfinitePlace K
      B : Nat
      hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
      n m : Nat
      hnm : LT.lt n m
      h : Eq (Ideal.span (Singleton.singleton ↑(NumberField.Units.dirichletUnitTheor …
      ⊢ Exists fun u => ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → LT.lt (Real.l …
    -/
  · have hu := Ideal.span_singleton_eq_span_singleton.mp h
    /-
      case intro.intro.intro.intro
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w₁ : NumberField.InfinitePlace K
      B : Nat
      hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
      n m : Nat
      hnm : LT.lt n m
      h : Eq (Ideal.span (Singleton.singleton ↑(NumberField.Units.dirichletUnitTheor …
      hu : Associated ↑(NumberField.Units.dirichletUnitTheorem.seq K w₁ hB n) ↑(Numb …
      ⊢ Exists fun u => ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → LT.lt (Real.l …
    -/
    refine ⟨hu.choose, fun w hw => Real.log_neg ?_ ?_⟩
      /-
        case intro.intro.intro.intro.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₁ : NumberField.InfinitePlace K
        B : Nat
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        n m : Nat
        hnm : LT.lt n m
        h : Eq (Ideal.span (Singleton.singleton ↑(NumberField.Units.dirichletUnitTheor …
        hu : Associated ↑(NumberField.Units.dirichletUnitTheorem.seq K w₁ hB n) ↑(Numb …
        w : NumberField.InfinitePlace K
        hw : Ne w w₁
        ⊢ LT.lt 0 (w ((algebraMap (NumberField.RingOfIntegers K) K) ↑(Exists.choose hu …
      -/
    · exact pos_iff.mpr (coe_ne_zero _)
      /-
        🎉 no goals
      -/
    · calc
        _ = w (algebraMap (𝓞 K) K (seq K w₁ hB m) * (algebraMap (𝓞 K) K (seq K w₁ hB n))⁻¹) := by
          rw [← congr_arg (algebraMap (𝓞 K) K) hu.choose_spec, mul_comm, map_mul (algebraMap _ _),
          ← mul_assoc, inv_mul_cancel₀ (seq_ne_zero K w₁ hB n), one_mul]
      _ = w (algebraMap (𝓞 K) K (seq K w₁ hB m)) * w (algebraMap (𝓞 K) K (seq K w₁ hB n))⁻¹ :=
        _root_.map_mul _ _ _
      _ < 1 := by
        rw [map_inv₀, mul_inv_lt_iff₀' (pos_iff.mpr (seq_ne_zero K w₁ hB n)), mul_one]
        exact seq_decreasing K w₁ hB hnm w hw
  refine Set.Finite.exists_lt_map_eq_of_forall_mem (t := {I : Ideal (𝓞 K) | Ideal.absNorm I ≤ B})
    (fun n ↦ ?_) (Ideal.finite_setOf_absNorm_le B)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : Nat
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    n : Nat
    ⊢ Membership.mem (setOf fun I => LE.le (Ideal.absNorm I) B) (Ideal.span (Singl …
  -/
  rw [Set.mem_setOf_eq, Ideal.absNorm_span_singleton]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : Nat
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    n : Nat
    ⊢ LE.le ((Algebra.norm Int) ↑(NumberField.Units.dirichletUnitTheorem.seq K w₁  …
  -/
  exact seq_norm_le K w₁ hB n
  /-
    🎉 no goals
  -/


theorem unitLattice_span_eq_top :
    Submodule.span ℝ (unitLattice K : Set ({w : InfinitePlace K // w ≠ w₀} → ℝ)) = ⊤ := by
  classical
  refine le_antisymm le_top ?_
  -- The standard basis
  let B := Pi.basisFun ℝ {w : InfinitePlace K // w ≠ w₀}
  -- The image by log_embedding of the family of units constructed above
  let v := fun w : { w : InfinitePlace K // w ≠ w₀ } =>
    logEmbedding K (Additive.ofMul (exists_unit K w).choose)
  -- To prove the result, it is enough to prove that the family `v` is linearly independent
  suffices B.det v ≠ 0 by
    rw [← isUnit_iff_ne_zero, ← is_basis_iff_det] at this
    rw [← this.2]
    refine  Submodule.span_monotone fun _ ⟨w, hw⟩ ↦ ⟨(exists_unit K w).choose, trivial, hw⟩
  rw [Basis.det_apply]
  -- We use a specific lemma to prove that this determinant is nonzero
  refine det_ne_zero_of_sum_col_lt_diag (fun w => ?_)
  simp_rw [Real.norm_eq_abs, B, Basis.coePiBasisFun.toMatrix_eq_transpose, Matrix.transpose_apply]
  rw [← sub_pos, sum_congr rfl (fun x hx => abs_of_neg ?_), sum_neg_distrib, sub_neg_eq_add,
    sum_erase_eq_sub (mem_univ _), ← add_comm_sub]
  · refine add_pos_of_nonneg_of_pos ?_ ?_
    · rw [sub_nonneg]
      exact le_abs_self _
    · rw [sum_logEmbedding_component (exists_unit K w).choose]
      refine mul_pos_of_neg_of_neg ?_ ((exists_unit K w).choose_spec _ w.prop.symm)
      rw [mult]; split_ifs <;> norm_num
  · refine mul_neg_of_pos_of_neg ?_ ((exists_unit K w).choose_spec x ?_)
    · rw [mult]; split_ifs <;> norm_num
    · exact Subtype.ext_iff_val.not.mp (ne_of_mem_erase hx)


/-- The unit rank of the number field `K`, it is equal to `card (InfinitePlace K) - 1`. -/
def rank : ℕ := Fintype.card (InfinitePlace K) - 1


instance instDiscrete_unitLattice : DiscreteTopology (unitLattice K) := by
  classical
  refine discreteTopology_of_isOpen_singleton_zero ?_
  refine isOpen_singleton_of_finite_mem_nhds 0 (s := Metric.closedBall 0 1) ?_ ?_
  · exact Metric.closedBall_mem_nhds _ (by norm_num)
  · refine Set.Finite.of_finite_image ?_ (Set.injOn_of_injective Subtype.val_injective)
    convert unitLattice_inter_ball_finite K 1
    ext x
    refine ⟨?_, fun ⟨hx1, hx2⟩ => ⟨⟨x, hx1⟩, hx2, rfl⟩⟩
    rintro ⟨x, hx, rfl⟩
    exact ⟨Subtype.mem x, hx⟩


open scoped Classical in
instance instZLattice_unitLattice : IsZLattice ℝ (unitLattice K) where
  span_top := unitLattice_span_eq_top K


protected theorem finrank_eq_rank :
    finrank ℝ ({w : InfinitePlace K // w ≠ w₀} → ℝ) = Units.rank K := by
  classical
  simp only [finrank_fintype_fun_eq_card, Fintype.card_subtype_compl,
    Fintype.card_ofSubsingleton, rank]


@[simp]
theorem unitLattice_rank :
    finrank ℤ (unitLattice K) = Units.rank K := by
  classical
  rw [← Units.finrank_eq_rank, ZLattice.rank ℝ]


/-- The map obtained by quotienting by the kernel of `logEmbedding`. -/
def logEmbeddingQuot :
    Additive ((𝓞 K)ˣ ⧸ (torsion K)) →+ ({w : InfinitePlace K // w ≠ w₀} → ℝ) :=
  MonoidHom.toAdditive' <|
    (QuotientGroup.kerLift (AddMonoidHom.toMultiplicative' (logEmbedding K))).comp
      (QuotientGroup.quotientMulEquivOfEq (by
        /-
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          ⊢ Eq (NumberField.Units.torsion K) (AddMonoidHom.toMultiplicative' (NumberFiel …
        -/
        ext
        rw [MonoidHom.mem_ker, AddMonoidHom.toMultiplicative'_apply_apply, ofAdd_eq_one,
          ← logEmbedding_eq_zero_iff])).toMonoidHom


@[simp]
theorem logEmbeddingQuot_apply (x : (𝓞 K)ˣ) :
    logEmbeddingQuot K (Additive.ofMul (QuotientGroup.mk x)) =
      logEmbedding K (Additive.ofMul x) := rfl


theorem logEmbeddingQuot_injective :
    Function.Injective (logEmbeddingQuot K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Function.Injective ⇑(NumberField.Units.logEmbeddingQuot K)
  -/
  unfold logEmbeddingQuot
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Function.Injective ⇑(MonoidHom.toAdditive' ((QuotientGroup.kerLift (AddMonoi …
  -/
  intro _ _ h
  simp_rw [MonoidHom.toAdditive'_apply_apply, MonoidHom.coe_comp, MulEquiv.coe_toMonoidHom,
    Function.comp_apply, EmbeddingLike.apply_eq_iff_eq] at h
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a₁✝ a₂✝ : Additive (HasQuotient.Quotient (Units (NumberField.RingOfIntegers K) …
    h : Eq ((QuotientGroup.kerLift (AddMonoidHom.toMultiplicative' (NumberField.Un …
    ⊢ Eq a₁✝ a₂✝
  -/
  exact (EmbeddingLike.apply_eq_iff_eq _).mp <| (QuotientGroup.kerLift_injective _).eq_iff.mp h
  /-
    🎉 no goals
  -/


/-- The linear equivalence between `(𝓞 K)ˣ ⧸ (torsion K)` as an additive `ℤ`-module and
`unitLattice` . -/
def logEmbeddingEquiv :
    Additive ((𝓞 K)ˣ ⧸ (torsion K)) ≃ₗ[ℤ] (unitLattice K) :=
  LinearEquiv.ofBijective ((logEmbeddingQuot K).codRestrict (unitLattice K)
    (Quotient.ind fun _ ↦ logEmbeddingQuot_apply K _ ▸
      Submodule.mem_map_of_mem trivial)).toIntLinearMap
    ⟨fun _ _ ↦ by
      rw [AddMonoidHom.coe_toIntLinearMap, AddMonoidHom.codRestrict_apply,
        AddMonoidHom.codRestrict_apply, Subtype.mk.injEq]
      /-
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x✝¹ x✝ : Additive (HasQuotient.Quotient (Units (NumberField.RingOfIntegers K)) …
        ⊢ Eq ((NumberField.Units.logEmbeddingQuot K) x✝¹) ((NumberField.Units.logEmbed …
      -/
      /-
        🎉 no goals
      -/
      apply logEmbeddingQuot_injective K, fun ⟨a, ⟨b, _, ha⟩⟩ ↦ ⟨⟦b⟧, by simpa using ha⟩⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem logEmbeddingEquiv_apply (x : (𝓞 K)ˣ) :
    logEmbeddingEquiv K (Additive.ofMul (QuotientGroup.mk x)) =
      logEmbedding K (Additive.ofMul x) := rfl


instance : Module.Free ℤ (Additive ((𝓞 K)ˣ ⧸ (torsion K))) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Module.Free Int (Additive (HasQuotient.Quotient (Units (NumberField.RingOfIn …
  -/
  classical exact Module.Free.of_equiv (logEmbeddingEquiv K).symm
  /-
    🎉 no goals
  -/


instance : Module.Finite ℤ (Additive ((𝓞 K)ˣ ⧸ (torsion K))) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Module.Finite Int (Additive (HasQuotient.Quotient (Units (NumberField.RingOf …
  -/
  classical exact Module.Finite.equiv (logEmbeddingEquiv K).symm
  /-
    🎉 no goals
  -/

-- Note that we prove this instance first and then deduce from it the instance
-- `Monoid.FG (𝓞 K)ˣ`, and not the other way around, due to no `Subgroup` version
-- of `Submodule.fg_of_fg_map_of_fg_inf_ker` existing.

instance : Module.Finite ℤ (Additive (𝓞 K)ˣ) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Module.Finite Int (Additive (Units (NumberField.RingOfIntegers K)))
  -/
  rw [Module.finite_def]
  refine Submodule.fg_of_fg_map_of_fg_inf_ker
    (MonoidHom.toAdditive (QuotientGroup.mk' (torsion K))).toIntLinearMap ?_ ?_
  · rw [Submodule.map_top, LinearMap.range_eq_top.mpr
      (by exact QuotientGroup.mk'_surjective (torsion K)), ← Module.finite_def]
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ Module.Finite Int (Additive (HasQuotient.Quotient (Units (NumberField.RingOf …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  · rw [inf_of_le_right le_top, AddMonoidHom.coe_toIntLinearMap_ker, MonoidHom.coe_toAdditive_ker,
      QuotientGroup.ker_mk', Submodule.fg_iff_add_subgroup_fg,
      AddSubgroup.toIntSubmodule_toAddSubgroup, ← AddGroup.fg_iff_addSubgroup_fg]
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ AddGroup.FG (Subtype fun x => Membership.mem (Subgroup.toAddSubgroup (Number …
    -/
    have : Finite (Subgroup.toAddSubgroup (torsion K)) := (inferInstance : Finite (torsion K))
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      this : Finite (Subtype fun x => Membership.mem (Subgroup.toAddSubgroup (Number …
      ⊢ AddGroup.FG (Subtype fun x => Membership.mem (Subgroup.toAddSubgroup (Number …
    -/
    exact AddGroup.fg_of_finite
    /-
      🎉 no goals
    -/


instance : Monoid.FG (𝓞 K)ˣ := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Monoid.FG (Units (NumberField.RingOfIntegers K))
  -/
  rw [Monoid.fg_iff_add_fg, ← AddGroup.fg_iff_addMonoid_fg, ← Module.Finite.iff_addGroup_fg]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Module.Finite Int (Additive (Units (NumberField.RingOfIntegers K)))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem rank_modTorsion :
    Module.finrank ℤ (Additive ((𝓞 K)ˣ ⧸ (torsion K))) = rank K := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Eq (Module.finrank Int (Additive (HasQuotient.Quotient (Units (NumberField.R …
  -/
  rw [← LinearEquiv.finrank_eq (logEmbeddingEquiv K).symm, unitLattice_rank]
  /-
    🎉 no goals
  -/


/-- A basis of the quotient `(𝓞 K)ˣ ⧸ (torsion K)` seen as an additive ℤ-module. -/
def basisModTorsion : Basis (Fin (rank K)) ℤ (Additive ((𝓞 K)ˣ ⧸ (torsion K))) :=
  Basis.reindex (Module.Free.chooseBasis ℤ _) (Fintype.equivOfCardEq <| by
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ Eq (Fintype.card (Module.Free.ChooseBasisIndex Int (Additive (HasQuotient.Qu …
    -/
    rw [← Module.finrank_eq_card_chooseBasisIndex, rank_modTorsion, Fintype.card_fin])
    /-
      🎉 no goals
    -/


/-- The basis of the `unitLattice` obtained by mapping `basisModTorsion` via `logEmbedding`. -/
def basisUnitLattice : Basis (Fin (rank K)) ℤ (unitLattice K) :=
  (basisModTorsion K).map (logEmbeddingEquiv K)


/-- A fundamental system of units of `K`. The units of `fundSystem` are arbitrary lifts of the
units in `basisModTorsion`. -/
def fundSystem : Fin (rank K) → (𝓞 K)ˣ :=
  -- `:)` prevents the `⧸` decaying to a quotient by `leftRel` when we unfold this later
  fun i => Quotient.out ((basisModTorsion K i).toMul:)


theorem fundSystem_mk (i : Fin (rank K)) :
    Additive.ofMul (QuotientGroup.mk (fundSystem K i)) = (basisModTorsion K i) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    i : Fin (NumberField.Units.rank K)
    ⊢ Eq (Additive.ofMul ↑(NumberField.Units.fundSystem K i)) ((NumberField.Units. …
  -/
  simp_rw [fundSystem, Equiv.apply_eq_iff_eq_symm_apply, Additive.ofMul_symm_eq, Quotient.out_eq']
  /-
    🎉 no goals
  -/


theorem logEmbedding_fundSystem (i : Fin (rank K)) :
    logEmbedding K (Additive.ofMul (fundSystem K i)) = basisUnitLattice K i := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    i : Fin (NumberField.Units.rank K)
    ⊢ Eq ((NumberField.Units.logEmbedding K) (Additive.ofMul (NumberField.Units.fu …
  -/
  rw [basisUnitLattice, Basis.map_apply, ← fundSystem_mk, logEmbeddingEquiv_apply]
  /-
    🎉 no goals
  -/


/-- The exponents that appear in the unique decomposition of a unit as the product of
a root of unity and powers of the units of the fundamental system `fundSystem` (see
`exist_unique_eq_mul_prod`) are given by the representation of the unit on `basisModTorsion`. -/
theorem fun_eq_repr {x ζ : (𝓞 K)ˣ} {f : Fin (rank K) → ℤ} (hζ : ζ ∈ torsion K)
    (h : x = ζ * ∏ i, (fundSystem K i) ^ (f i)) :
    f = (basisModTorsion K).repr (Additive.ofMul ↑x) := by
  suffices Additive.ofMul ↑x = ∑ i, (f i) • (basisModTorsion K i) by
    rw [← (basisModTorsion K).repr_sum_self f, ← this]
  calc
    Additive.ofMul ↑x
    _ = ∑ i, (f i) • Additive.ofMul ↑(fundSystem K i) := by
          rw [h, QuotientGroup.mk_mul, (QuotientGroup.eq_one_iff _).mpr hζ, one_mul,
            QuotientGroup.mk_prod, ofMul_prod]; rfl
    _ = ∑ i, (f i) • (basisModTorsion K i) := by
          simp_rw [fundSystem, QuotientGroup.out_eq', ofMul_toMul]


/-- **Dirichlet Unit Theorem**. Any unit `x` of `𝓞 K` can be written uniquely as the product of
a root of unity and powers of the units of the fundamental system `fundSystem`. -/
theorem exist_unique_eq_mul_prod (x : (𝓞 K)ˣ) : ∃! ζe : torsion K × (Fin (rank K) → ℤ),
    x = ζe.1 * ∏ i, (fundSystem K i) ^ (ζe.2 i) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    ⊢ ExistsUnique fun ζe => Eq x (HMul.hMul (↑ζe.1) (Finset.univ.prod fun i => HP …
  -/
  let ζ := x * (∏ i, (fundSystem K i) ^ ((basisModTorsion K).repr (Additive.ofMul ↑x) i))⁻¹
  have h_tors : ζ ∈ torsion K := by
    rw [← QuotientGroup.eq_one_iff, QuotientGroup.mk_mul, QuotientGroup.mk_inv, ← ofMul_eq_zero,
      ofMul_mul, ofMul_inv, QuotientGroup.mk_prod, ofMul_prod]
    simp_rw [QuotientGroup.mk_zpow, ofMul_zpow, fundSystem, QuotientGroup.out_eq']
    rw [add_eq_zero_iff_eq_neg, neg_neg]
    exact ((basisModTorsion K).sum_repr (Additive.ofMul ↑x)).symm
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    ζ : Units (NumberField.RingOfIntegers K) := HMul.hMul x (Inv.inv (Finset.univ. …
    h_tors : Membership.mem (NumberField.Units.torsion K) ζ
    ⊢ ExistsUnique fun ζe => Eq x (HMul.hMul (↑ζe.1) (Finset.univ.prod fun i => HP …
  -/
  refine ⟨⟨⟨ζ, h_tors⟩, ((basisModTorsion K).repr (Additive.ofMul ↑x) : Fin (rank K) → ℤ)⟩, ?_, ?_⟩
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      ζ : Units (NumberField.RingOfIntegers K) := HMul.hMul x (Inv.inv (Finset.univ. …
      h_tors : Membership.mem (NumberField.Units.torsion K) ζ
      ⊢ (fun ζe => Eq x (HMul.hMul (↑ζe.1) (Finset.univ.prod fun i => HPow.hPow (Num …
    -/
  · simp only [ζ, _root_.inv_mul_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      ζ : Units (NumberField.RingOfIntegers K) := HMul.hMul x (Inv.inv (Finset.univ. …
      h_tors : Membership.mem (NumberField.Units.torsion K) ζ
      ⊢ ∀ (y : Prod (Subtype fun x => Membership.mem (NumberField.Units.torsion K) x …
    -/
  · rintro ⟨⟨ζ', h_tors'⟩, η⟩ hf
    /-
      case refine_2.mk.mk
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      ζ : Units (NumberField.RingOfIntegers K) := HMul.hMul x (Inv.inv (Finset.univ. …
      h_tors : Membership.mem (NumberField.Units.torsion K) ζ
      η : Fin (NumberField.Units.rank K) → Int
      ζ' : Units (NumberField.RingOfIntegers K)
      h_tors' : Membership.mem (NumberField.Units.torsion K) ζ'
      hf : Eq x (HMul.hMul (↑{ fst := ⟨ζ', h_tors'⟩, snd := η }.1) (Finset.univ.prod …
      ⊢ Eq { fst := ⟨ζ', h_tors'⟩, snd := η } { fst := ⟨ζ, h_tors⟩, snd := ⇑((Number …
    -/
    simp only [ζ, ← fun_eq_repr K h_tors' hf, Prod.mk.injEq, Subtype.mk.injEq, and_true]
    /-
      case refine_2.mk.mk
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      ζ : Units (NumberField.RingOfIntegers K) := HMul.hMul x (Inv.inv (Finset.univ. …
      h_tors : Membership.mem (NumberField.Units.torsion K) ζ
      η : Fin (NumberField.Units.rank K) → Int
      ζ' : Units (NumberField.RingOfIntegers K)
      h_tors' : Membership.mem (NumberField.Units.torsion K) ζ'
      hf : Eq x (HMul.hMul (↑{ fst := ⟨ζ', h_tors'⟩, snd := η }.1) (Finset.univ.prod …
      ⊢ Eq ζ' (HMul.hMul x (Inv.inv (Finset.univ.prod fun x => HPow.hPow (NumberFiel …
    -/
    nth_rewrite 1 [hf]
    /-
      case refine_2.mk.mk
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : Units (NumberField.RingOfIntegers K)
      ζ : Units (NumberField.RingOfIntegers K) := HMul.hMul x (Inv.inv (Finset.univ. …
      h_tors : Membership.mem (NumberField.Units.torsion K) ζ
      η : Fin (NumberField.Units.rank K) → Int
      ζ' : Units (NumberField.RingOfIntegers K)
      h_tors' : Membership.mem (NumberField.Units.torsion K) ζ'
      hf : Eq x (HMul.hMul (↑{ fst := ⟨ζ', h_tors'⟩, snd := η }.1) (Finset.univ.prod …
      ⊢ Eq ζ' (HMul.hMul (HMul.hMul (↑{ fst := ⟨ζ', h_tors'⟩, snd := η }.1) (Finset. …
    -/
    rw [_root_.mul_inv_cancel_right]
    /-
      🎉 no goals
    -/


