/-- Summing the translates of `f` by `ℤ • p` gives a map which is periodic with period `p`.
(This is true without any convergence conditions, since if the sum doesn't converge it is taken to
be the zero map, which is periodic.) -/
theorem periodic_tsum_comp_add_zsmul [AddCommGroup X] [TopologicalAddGroup X] [AddCommMonoid Y]
    [ContinuousAdd Y] [T2Space Y] (f : C(X, Y)) (p : X) :
    Function.Periodic (⇑(∑' n : ℤ, f.comp (ContinuousMap.addRight (n • p)))) p := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : AddCommGroup X
    inst✝³ : TopologicalAddGroup X
    inst✝² : AddCommMonoid Y
    inst✝¹ : ContinuousAdd Y
    inst✝ : T2Space Y
    f : ContinuousMap X Y
    p : X
    ⊢ Function.Periodic (⇑(tsum fun n => f.comp (ContinuousMap.addRight (HSMul.hSM …
  -/
  intro x
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : AddCommGroup X
    inst✝³ : TopologicalAddGroup X
    inst✝² : AddCommMonoid Y
    inst✝¹ : ContinuousAdd Y
    inst✝ : T2Space Y
    f : ContinuousMap X Y
    p x : X
    ⊢ Eq ((tsum fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))) (HAdd. …
  -/
  by_cases h : Summable fun n : ℤ => f.comp (ContinuousMap.addRight (n • p))
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : AddCommGroup X
      inst✝³ : TopologicalAddGroup X
      inst✝² : AddCommMonoid Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : T2Space Y
      f : ContinuousMap X Y
      p x : X
      h : Summable fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))
      ⊢ Eq ((tsum fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))) (HAdd. …
    -/
  · convert congr_arg (fun f : C(X, Y) => f x) ((Equiv.addRight (1 : ℤ)).tsum_eq _) using 1
    -- Porting note: in mathlib3 the proof from here was:
    -- simp_rw [← tsum_apply h, ← tsum_apply ((equiv.add_right (1 : ℤ)).summable_iff.mpr h),
    --   equiv.coe_add_right, comp_apply, coe_add_right, add_one_zsmul, add_comm (_ • p) p,
    --   ← add_assoc]
    -- However now the second `← tsum_apply` doesn't fire unless we use `erw`.
    /-
      case h.e'_2
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : AddCommGroup X
      inst✝³ : TopologicalAddGroup X
      inst✝² : AddCommMonoid Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : T2Space Y
      f : ContinuousMap X Y
      p x : X
      h : Summable fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))
      ⊢ Eq ((tsum fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))) (HAdd. …
    -/
    simp_rw [← tsum_apply h]
    /-
      case h.e'_2
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : AddCommGroup X
      inst✝³ : TopologicalAddGroup X
      inst✝² : AddCommMonoid Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : T2Space Y
      f : ContinuousMap X Y
      p x : X
      h : Summable fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))
      ⊢ Eq (tsum fun i => (f.comp (ContinuousMap.addRight (HSMul.hSMul i p))) (HAdd. …
    -/
    erw [← tsum_apply ((Equiv.addRight (1 : ℤ)).summable_iff.mpr h)]
    /-
      case h.e'_2
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : AddCommGroup X
      inst✝³ : TopologicalAddGroup X
      inst✝² : AddCommMonoid Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : T2Space Y
      f : ContinuousMap X Y
      p x : X
      h : Summable fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))
      ⊢ Eq (tsum fun i => (f.comp (ContinuousMap.addRight (HSMul.hSMul i p))) (HAdd. …
    -/
    simp [coe_addRight, add_one_zsmul, add_comm (_ • p) p, ← add_assoc]
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : AddCommGroup X
      inst✝³ : TopologicalAddGroup X
      inst✝² : AddCommMonoid Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : T2Space Y
      f : ContinuousMap X Y
      p x : X
      h : Not (Summable fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p)))
      ⊢ Eq ((tsum fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p))) (HAdd. …
    -/
  · rw [tsum_eq_zero_of_not_summable h]
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : AddCommGroup X
      inst✝³ : TopologicalAddGroup X
      inst✝² : AddCommMonoid Y
      inst✝¹ : ContinuousAdd Y
      inst✝ : T2Space Y
      f : ContinuousMap X Y
      p x : X
      h : Not (Summable fun n => f.comp (ContinuousMap.addRight (HSMul.hSMul n p)))
      ⊢ Eq (0 (HAdd.hAdd x p)) (0 x)
    -/
    simp only [coe_zero, Pi.zero_apply]
    /-
      🎉 no goals
    -/


