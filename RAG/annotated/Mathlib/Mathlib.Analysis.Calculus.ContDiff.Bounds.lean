/-- Bounding the norm of the iterated derivative of `B (f x) (g x)` within a set in terms of the
iterated derivatives of `f` and `g` when `B` is bilinear. This lemma is an auxiliary version
assuming all spaces live in the same universe, to enable an induction. Use instead
`ContinuousLinearMap.norm_iteratedFDerivWithin_le_of_bilinear` that removes this assumption. -/
theorem ContinuousLinearMap.norm_iteratedFDerivWithin_le_of_bilinear_aux {Du Eu Fu Gu : Type u}
    [NormedAddCommGroup Du] [NormedSpace 𝕜 Du] [NormedAddCommGroup Eu] [NormedSpace 𝕜 Eu]
    [NormedAddCommGroup Fu] [NormedSpace 𝕜 Fu] [NormedAddCommGroup Gu] [NormedSpace 𝕜 Gu]
    (B : Eu →L[𝕜] Fu →L[𝕜] Gu) {f : Du → Eu} {g : Du → Fu} {n : ℕ} {s : Set Du} {x : Du}
    (hf : ContDiffOn 𝕜 n f s) (hg : ContDiffOn 𝕜 n g s) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    ‖iteratedFDerivWithin 𝕜 n (fun y => B (f y) (g y)) s x‖ ≤
      ‖B‖ * ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ := by
  /- We argue by induction on `n`. The bound is trivial for `n = 0`. For `n + 1`, we write
    the `(n+1)`-th derivative as the `n`-th derivative of the derivative `B f g' + B f' g`,
    and apply the inductive assumption to each of those two terms. For this induction to make sense,
    the spaces of linear maps that appear in the induction should be in the same universe as the
    original spaces, which explains why we assume in the lemma that all spaces live in the same
    universe. -/
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    Du Eu Fu Gu : Type u
    inst✝⁷ : NormedAddCommGroup Du
    inst✝⁶ : NormedSpace 𝕜 Du
    inst✝⁵ : NormedAddCommGroup Eu
    inst✝⁴ : NormedSpace 𝕜 Eu
    inst✝³ : NormedAddCommGroup Fu
    inst✝² : NormedSpace 𝕜 Fu
    inst✝¹ : NormedAddCommGroup Gu
    inst✝ : NormedSpace 𝕜 Gu
    B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
    f : Du → Eu
    g : Du → Fu
    n : Nat
    s : Set Du
    x : Du
    hf : ContDiffOn 𝕜 (↑n) f s
    hg : ContDiffOn 𝕜 (↑n) g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  induction' n with n IH generalizing Eu Fu Gu
  · simp only [norm_iteratedFDerivWithin_zero, zero_add, Finset.range_one,
      Finset.sum_singleton, Nat.choose_self, Nat.cast_one, one_mul, Nat.sub_zero, ← mul_assoc]
    /-
      case zero
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑0) f s
      hg : ContDiffOn 𝕜 (↑0) g s
      ⊢ LE.le (Norm.norm ((B (f x)) (g x))) (HMul.hMul (HMul.hMul (Norm.norm B) (Nor …
    -/
    apply B.le_opNorm₂
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      IH : ∀ {Eu Fu Gu : Type u} [inst : NormedAddCommGroup Eu] [inst_1 : NormedSpac …
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
      hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g s
      ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) (fun y => (B (f y)) …
    -/
  · have In : (n : WithTop ℕ∞) + 1 ≤ n.succ := by simp only [Nat.cast_succ, le_refl]
    -- Porting note: the next line is a hack allowing Lean to find the operator norm instance.
    let norm := @ContinuousLinearMap.hasOpNorm _ _ Eu ((Du →L[𝕜] Fu) →L[𝕜] Du →L[𝕜] Gu) _ _ _ _ _ _
      (RingHom.id 𝕜)
    have I1 :
        ‖iteratedFDerivWithin 𝕜 n (fun y : Du => B.precompR Du (f y) (fderivWithin 𝕜 g s y)) s x‖ ≤
          ‖B‖ * ∑ i ∈ Finset.range (n + 1), n.choose i * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
            ‖iteratedFDerivWithin 𝕜 (n + 1 - i) g s x‖ := by
      calc
        ‖iteratedFDerivWithin 𝕜 n (fun y : Du => B.precompR Du (f y) (fderivWithin 𝕜 g s y)) s x‖ ≤
            ‖B.precompR Du‖ * ∑ i ∈ Finset.range (n + 1),
              n.choose i * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
                ‖iteratedFDerivWithin 𝕜 (n - i) (fderivWithin 𝕜 g s) s x‖ :=
          IH _ (hf.of_le (Nat.cast_le.2 (Nat.le_succ n))) (hg.fderivWithin hs In)
        _ ≤ ‖B‖ * ∑ i ∈ Finset.range (n + 1), n.choose i * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
              ‖iteratedFDerivWithin 𝕜 (n - i) (fderivWithin 𝕜 g s) s x‖ :=
            mul_le_mul_of_nonneg_right (B.norm_precompR_le Du) (by positivity)
        _ = _ := by
          congr 1
          apply Finset.sum_congr rfl fun i hi => ?_
          rw [Nat.succ_sub (Nat.lt_succ_iff.1 (Finset.mem_range.1 hi)),
            ← norm_iteratedFDerivWithin_fderivWithin hs hx]
    -- Porting note: the next line is a hack allowing Lean to find the operator norm instance.
    let norm := @ContinuousLinearMap.hasOpNorm _ _ (Du →L[𝕜] Eu) (Fu →L[𝕜] Du →L[𝕜] Gu) _ _ _ _ _ _
      (RingHom.id 𝕜)
    have I2 :
        ‖iteratedFDerivWithin 𝕜 n (fun y : Du => B.precompL Du (fderivWithin 𝕜 f s y) (g y)) s x‖ ≤
        ‖B‖ * ∑ i ∈ Finset.range (n + 1), n.choose i * ‖iteratedFDerivWithin 𝕜 (i + 1) f s x‖ *
          ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ :=
      calc
        ‖iteratedFDerivWithin 𝕜 n (fun y : Du => B.precompL Du (fderivWithin 𝕜 f s y) (g y)) s x‖ ≤
            ‖B.precompL Du‖ * ∑ i ∈ Finset.range (n + 1),
              n.choose i * ‖iteratedFDerivWithin 𝕜 i (fderivWithin 𝕜 f s) s x‖ *
                ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ :=
          IH _ (hf.fderivWithin hs In) (hg.of_le (Nat.cast_le.2 (Nat.le_succ n)))
        _ ≤ ‖B‖ * ∑ i ∈ Finset.range (n + 1),
            n.choose i * ‖iteratedFDerivWithin 𝕜 i (fderivWithin 𝕜 f s) s x‖ *
              ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ :=
            mul_le_mul_of_nonneg_right (B.norm_precompL_le Du) (by positivity)
        _ = _ := by
          congr 1
          apply Finset.sum_congr rfl fun i _ => ?_
          rw [← norm_iteratedFDerivWithin_fderivWithin hs hx]
    have J : iteratedFDerivWithin 𝕜 n
        (fun y : Du => fderivWithin 𝕜 (fun y : Du => B (f y) (g y)) s y) s x =
          iteratedFDerivWithin 𝕜 n (fun y => B.precompR Du (f y)
            (fderivWithin 𝕜 g s y) + B.precompL Du (fderivWithin 𝕜 f s y) (g y)) s x := by
      apply iteratedFDerivWithin_congr (fun y hy => ?_) hx
      have L : (1 : WithTop ℕ∞) ≤ n.succ := by
        simpa only [ENat.coe_one, Nat.one_le_cast] using Nat.succ_pos n
      exact B.fderivWithin_of_bilinear (hf.differentiableOn L y hy) (hg.differentiableOn L y hy)
        (hs y hy)
    /-
      case succ
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      IH : ∀ {Eu Fu Gu : Type u} [inst : NormedAddCommGroup Eu] [inst_1 : NormedSpac …
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
      hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g s
      In : LE.le (HAdd.hAdd (↑n) 1) ↑n.succ
      norm✝ : Norm (ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (Ring …
      I1 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      norm : Norm (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom. …
      I2 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      J : Eq (iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 (fun y => (B (f y))  …
      ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) (fun y => (B (f y)) …
    -/
    rw [← norm_iteratedFDerivWithin_fderivWithin hs hx, J]
    have A : ContDiffOn 𝕜 n (fun y => B.precompR Du (f y) (fderivWithin 𝕜 g s y)) s :=
      (B.precompR Du).isBoundedBilinearMap.contDiff.comp₂_contDiffOn
        (hf.of_le (Nat.cast_le.2 (Nat.le_succ n))) (hg.fderivWithin hs In)
    have A' : ContDiffOn 𝕜 n (fun y => B.precompL Du (fderivWithin 𝕜 f s y) (g y)) s :=
      (B.precompL Du).isBoundedBilinearMap.contDiff.comp₂_contDiffOn (hf.fderivWithin hs In)
        (hg.of_le (Nat.cast_le.2 (Nat.le_succ n)))
    /-
      case succ
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      IH : ∀ {Eu Fu Gu : Type u} [inst : NormedAddCommGroup Eu] [inst_1 : NormedSpac …
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
      hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g s
      In : LE.le (HAdd.hAdd (↑n) 1) ↑n.succ
      norm✝ : Norm (ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (Ring …
      I1 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      norm : Norm (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom. …
      I2 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      J : Eq (iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 (fun y => (B (f y))  …
      A : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompR Du B) (f y)) (f …
      A' : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompL Du B) (fderivW …
      ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => HAdd.hAdd (((Continuous …
    -/
    rw [iteratedFDerivWithin_add_apply' A A' hs hx]
    /-
      case succ
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      IH : ∀ {Eu Fu Gu : Type u} [inst : NormedAddCommGroup Eu] [inst_1 : NormedSpac …
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
      hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g s
      In : LE.le (HAdd.hAdd (↑n) 1) ↑n.succ
      norm✝ : Norm (ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (Ring …
      I1 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      norm : Norm (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom. …
      I2 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      J : Eq (iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 (fun y => (B (f y))  …
      A : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompR Du B) (f y)) (f …
      A' : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompL Du B) (fderivW …
      ⊢ LE.le (Norm.norm (HAdd.hAdd (iteratedFDerivWithin 𝕜 n (fun y => ((Continuous …
    -/
    apply (norm_add_le _ _).trans ((add_le_add I1 I2).trans (le_of_eq ?_))
    /-
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      IH : ∀ {Eu Fu Gu : Type u} [inst : NormedAddCommGroup Eu] [inst_1 : NormedSpac …
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
      hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g s
      In : LE.le (HAdd.hAdd (↑n) 1) ↑n.succ
      norm✝ : Norm (ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (Ring …
      I1 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      norm : Norm (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom. …
      I2 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      J : Eq (iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 (fun y => (B (f y))  …
      A : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompR Du B) (f y)) (f …
      A' : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompL Du B) (fderivW …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm B) ((Finset.range (HAdd.hAdd n 1)).sum f …
    -/
    simp_rw [← mul_add, mul_assoc]
    /-
      𝕜 : Type u_1
      inst✝⁸ : NontriviallyNormedField 𝕜
      Du : Type u
      inst✝⁷ : NormedAddCommGroup Du
      inst✝⁶ : NormedSpace 𝕜 Du
      s : Set Du
      x : Du
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      IH : ∀ {Eu Fu Gu : Type u} [inst : NormedAddCommGroup Eu] [inst_1 : NormedSpac …
      Eu Fu Gu : Type u
      inst✝⁵ : NormedAddCommGroup Eu
      inst✝⁴ : NormedSpace 𝕜 Eu
      inst✝³ : NormedAddCommGroup Fu
      inst✝² : NormedSpace 𝕜 Fu
      inst✝¹ : NormedAddCommGroup Gu
      inst✝ : NormedSpace 𝕜 Gu
      B : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜)  …
      f : Du → Eu
      g : Du → Fu
      hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
      hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g s
      In : LE.le (HAdd.hAdd (↑n) 1) ↑n.succ
      norm✝ : Norm (ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (Ring …
      I1 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      norm : Norm (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom. …
      I2 : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => ((ContinuousLinearMa …
      J : Eq (iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 (fun y => (B (f y))  …
      A : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompR Du B) (f y)) (f …
      A' : ContDiffOn 𝕜 (↑n) (fun y => ((ContinuousLinearMap.precompL Du B) (fderivW …
      ⊢ Eq (HMul.hMul (Norm.norm B) (HAdd.hAdd ((Finset.range (HAdd.hAdd n 1)).sum f …
    -/
    congr 1
    exact (Finset.sum_choose_succ_mul
      (fun i j => ‖iteratedFDerivWithin 𝕜 i f s x‖ * ‖iteratedFDerivWithin 𝕜 j g s x‖) n).symm


/-- Bounding the norm of the iterated derivative of `B (f x) (g x)` within a set in terms of the
iterated derivatives of `f` and `g` when `B` is bilinear:
`‖D^n (x ↦ B (f x) (g x))‖ ≤ ‖B‖ ∑_{k ≤ n} n.choose k ‖D^k f‖ ‖D^{n-k} g‖` -/
theorem ContinuousLinearMap.norm_iteratedFDerivWithin_le_of_bilinear (B : E →L[𝕜] F →L[𝕜] G)
    {f : D → E} {g : D → F} {N : WithTop ℕ∞} {s : Set D} {x : D} (hf : ContDiffOn 𝕜 N f s)
    (hg : ContDiffOn 𝕜 N g s) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ} (hn : n ≤ N) :
    ‖iteratedFDerivWithin 𝕜 n (fun y => B (f y) (g y)) s x‖ ≤
      ‖B‖ * ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ := by
  /- We reduce the bound to the case where all spaces live in the same universe (in which we
    already have proved the result), by using linear isometries between the spaces and their `ULift`
    to a common universe. These linear isometries preserve the norm of the iterated derivative. -/
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  let Du : Type max uD uE uF uG := ULift.{max uE uF uG, uD} D
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  let Eu : Type max uD uE uF uG := ULift.{max uD uF uG, uE} E
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  let Fu : Type max uD uE uF uG := ULift.{max uD uE uG, uF} F
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  let Gu : Type max uD uE uF uG := ULift.{max uD uE uF, uG} G
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  have isoD : Du ≃ₗᵢ[𝕜] D := LinearIsometryEquiv.ulift 𝕜 D
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  have isoE : Eu ≃ₗᵢ[𝕜] E := LinearIsometryEquiv.ulift 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  have isoF : Fu ≃ₗᵢ[𝕜] F := LinearIsometryEquiv.ulift 𝕜 F
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  have isoG : Gu ≃ₗᵢ[𝕜] G := LinearIsometryEquiv.ulift 𝕜 G
  -- lift `f` and `g` to versions `fu` and `gu` on the lifted spaces.
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  set fu : Du → Eu := isoE.symm ∘ f ∘ isoD with hfu
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  set gu : Du → Fu := isoF.symm ∘ g ∘ isoD with hgu
  -- lift the bilinear map `B` to a bilinear map `Bu` on the lifted spaces.
  set Bu₀ : Eu →L[𝕜] Fu →L[𝕜] G := ((B.comp (isoE : Eu →L[𝕜] E)).flip.comp (isoF : Fu →L[𝕜] F)).flip
    with hBu₀
  let Bu : Eu →L[𝕜] Fu →L[𝕜] Gu :=
   ContinuousLinearMap.compL 𝕜 Eu (Fu →L[𝕜] G) (Fu →L[𝕜] Gu)
    (ContinuousLinearMap.compL 𝕜 Fu G Gu (isoG.symm : G →L[𝕜] Gu)) Bu₀
  have hBu : Bu = ContinuousLinearMap.compL 𝕜 Eu (Fu →L[𝕜] G) (Fu →L[𝕜] Gu)
      (ContinuousLinearMap.compL 𝕜 Fu G Gu (isoG.symm : G →L[𝕜] Gu)) Bu₀ := rfl
  have Bu_eq : (fun y => Bu (fu y) (gu y)) = isoG.symm ∘ (fun y => B (f y) (g y)) ∘ isoD := by
    ext1 y
    simp [Du, Eu, Fu, Gu, hBu, hBu₀, hfu, hgu]
  -- All norms are preserved by the lifting process.
  have Bu_le : ‖Bu‖ ≤ ‖B‖ := by
    refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg B) fun y => ?_
    refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun x => ?_
    simp only [Du, Eu, Fu, Gu, hBu, hBu₀, compL_apply, coe_comp', Function.comp_apply,
      ContinuousLinearEquiv.coe_coe, LinearIsometryEquiv.coe_coe, flip_apply,
      LinearIsometryEquiv.norm_map]
    calc
      ‖B (isoE y) (isoF x)‖ ≤ ‖B (isoE y)‖ * ‖isoF x‖ := ContinuousLinearMap.le_opNorm _ _
      _ ≤ ‖B‖ * ‖isoE y‖ * ‖isoF x‖ := by gcongr; apply ContinuousLinearMap.le_opNorm
      _ = ‖B‖ * ‖y‖ * ‖x‖ := by simp only [LinearIsometryEquiv.norm_map]
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    gu : Du → Fu := Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD)
    hgu : Eq gu (Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD))
    Bu₀ : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜 …
    hBu₀ : Eq Bu₀ ((B.comp ↑{ toLinearEquiv := isoE.toLinearEquiv, continuous_toFu …
    Bu : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜) …
    hBu : Eq Bu (((ContinuousLinearMap.compL 𝕜 Eu (ContinuousLinearMap (RingHom.id …
    Bu_eq : Eq (fun y => (Bu (fu y)) (gu y)) (Function.comp (⇑isoG.symm) (Function …
    Bu_le : LE.le (Norm.norm Bu) (Norm.norm B)
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  let su := isoD ⁻¹' s
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    gu : Du → Fu := Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD)
    hgu : Eq gu (Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD))
    Bu₀ : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜 …
    hBu₀ : Eq Bu₀ ((B.comp ↑{ toLinearEquiv := isoE.toLinearEquiv, continuous_toFu …
    Bu : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜) …
    hBu : Eq Bu (((ContinuousLinearMap.compL 𝕜 Eu (ContinuousLinearMap (RingHom.id …
    Bu_eq : Eq (fun y => (Bu (fu y)) (gu y)) (Function.comp (⇑isoG.symm) (Function …
    Bu_le : LE.le (Norm.norm Bu) (Norm.norm B)
    su : Set Du := Set.preimage (⇑isoD) s
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  have hsu : UniqueDiffOn 𝕜 su := isoD.toContinuousLinearEquiv.uniqueDiffOn_preimage_iff.2 hs
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    gu : Du → Fu := Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD)
    hgu : Eq gu (Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD))
    Bu₀ : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜 …
    hBu₀ : Eq Bu₀ ((B.comp ↑{ toLinearEquiv := isoE.toLinearEquiv, continuous_toFu …
    Bu : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜) …
    hBu : Eq Bu (((ContinuousLinearMap.compL 𝕜 Eu (ContinuousLinearMap (RingHom.id …
    Bu_eq : Eq (fun y => (Bu (fu y)) (gu y)) (Function.comp (⇑isoG.symm) (Function …
    Bu_le : LE.le (Norm.norm Bu) (Norm.norm B)
    su : Set Du := Set.preimage (⇑isoD) s
    hsu : UniqueDiffOn 𝕜 su
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  let xu := isoD.symm x
  have hxu : xu ∈ su := by
    simpa only [xu, su, Set.mem_preimage, LinearIsometryEquiv.apply_symm_apply] using hx
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    gu : Du → Fu := Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD)
    hgu : Eq gu (Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD))
    Bu₀ : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜 …
    hBu₀ : Eq Bu₀ ((B.comp ↑{ toLinearEquiv := isoE.toLinearEquiv, continuous_toFu …
    Bu : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜) …
    hBu : Eq Bu (((ContinuousLinearMap.compL 𝕜 Eu (ContinuousLinearMap (RingHom.id …
    Bu_eq : Eq (fun y => (Bu (fu y)) (gu y)) (Function.comp (⇑isoG.symm) (Function …
    Bu_le : LE.le (Norm.norm Bu) (Norm.norm B)
    su : Set Du := Set.preimage (⇑isoD) s
    hsu : UniqueDiffOn 𝕜 su
    xu : Du := isoD.symm x
    hxu : Membership.mem su xu
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  have xu_x : isoD xu = x := by simp only [xu, LinearIsometryEquiv.apply_symm_apply]
  have hfu : ContDiffOn 𝕜 n fu su :=
    isoE.symm.contDiff.comp_contDiffOn
      ((hf.of_le hn).comp_continuousLinearMap (isoD : Du →L[𝕜] D))
  have hgu : ContDiffOn 𝕜 n gu su :=
    isoF.symm.contDiff.comp_contDiffOn
      ((hg.of_le hn).comp_continuousLinearMap (isoD : Du →L[𝕜] D))
  have Nfu : ∀ i, ‖iteratedFDerivWithin 𝕜 i fu su xu‖ = ‖iteratedFDerivWithin 𝕜 i f s x‖ := by
    intro i
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_left _ _ hsu hxu]
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_right _ _ hs, xu_x]
    rwa [← xu_x] at hx
  have Ngu : ∀ i, ‖iteratedFDerivWithin 𝕜 i gu su xu‖ = ‖iteratedFDerivWithin 𝕜 i g s x‖ := by
    intro i
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_left _ _ hsu hxu]
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_right _ _ hs, xu_x]
    rwa [← xu_x] at hx
  have NBu :
    ‖iteratedFDerivWithin 𝕜 n (fun y => Bu (fu y) (gu y)) su xu‖ =
      ‖iteratedFDerivWithin 𝕜 n (fun y => B (f y) (g y)) s x‖ := by
    rw [Bu_eq]
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_left _ _ hsu hxu]
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_right _ _ hs, xu_x]
    rwa [← xu_x] at hx
  -- state the bound for the lifted objects, and deduce the original bound from it.
  have : ‖iteratedFDerivWithin 𝕜 n (fun y => Bu (fu y) (gu y)) su xu‖ ≤
      ‖Bu‖ * ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i fu su xu‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) gu su xu‖ :=
    Bu.norm_iteratedFDerivWithin_le_of_bilinear_aux hfu hgu hsu hxu
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu✝ : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    gu : Du → Fu := Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD)
    hgu✝ : Eq gu (Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD))
    Bu₀ : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜 …
    hBu₀ : Eq Bu₀ ((B.comp ↑{ toLinearEquiv := isoE.toLinearEquiv, continuous_toFu …
    Bu : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜) …
    hBu : Eq Bu (((ContinuousLinearMap.compL 𝕜 Eu (ContinuousLinearMap (RingHom.id …
    Bu_eq : Eq (fun y => (Bu (fu y)) (gu y)) (Function.comp (⇑isoG.symm) (Function …
    Bu_le : LE.le (Norm.norm Bu) (Norm.norm B)
    su : Set Du := Set.preimage (⇑isoD) s
    hsu : UniqueDiffOn 𝕜 su
    xu : Du := isoD.symm x
    hxu : Membership.mem su xu
    xu_x : Eq (isoD xu) x
    hfu : ContDiffOn 𝕜 (↑n) fu su
    hgu : ContDiffOn 𝕜 (↑n) gu su
    Nfu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i fu su xu)) (Norm.no …
    Ngu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i gu su xu)) (Norm.no …
    NBu : Eq (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (Bu (fu y)) (gu y)) su …
    this : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (Bu (fu y)) (gu y) …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  simp only [Nfu, Ngu, NBu] at this
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    Du : Type (max uD uE uF uG) := ULift.{max uE uF uG, uD} D
    Eu : Type (max uD uE uF uG) := ULift.{max uD uF uG, uE} E
    Fu : Type (max uD uE uF uG) := ULift.{max uD uE uG, uF} F
    Gu : Type (max uD uE uF uG) := ULift.{max uD uE uF, uG} G
    isoD : LinearIsometryEquiv (RingHom.id 𝕜) Du D
    isoE : LinearIsometryEquiv (RingHom.id 𝕜) Eu E
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : Du → Eu := Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD)
    hfu✝ : Eq fu (Function.comp (⇑isoE.symm) (Function.comp f ⇑isoD))
    gu : Du → Fu := Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD)
    hgu✝ : Eq gu (Function.comp (⇑isoF.symm) (Function.comp g ⇑isoD))
    Bu₀ : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜 …
    hBu₀ : Eq Bu₀ ((B.comp ↑{ toLinearEquiv := isoE.toLinearEquiv, continuous_toFu …
    Bu : ContinuousLinearMap (RingHom.id 𝕜) Eu (ContinuousLinearMap (RingHom.id 𝕜) …
    hBu : Eq Bu (((ContinuousLinearMap.compL 𝕜 Eu (ContinuousLinearMap (RingHom.id …
    Bu_eq : Eq (fun y => (Bu (fu y)) (gu y)) (Function.comp (⇑isoG.symm) (Function …
    Bu_le : LE.le (Norm.norm Bu) (Norm.norm B)
    su : Set Du := Set.preimage (⇑isoD) s
    hsu : UniqueDiffOn 𝕜 su
    xu : Du := isoD.symm x
    hxu : Membership.mem su xu
    xu_x : Eq (isoD xu) x
    hfu : ContDiffOn 𝕜 (↑n) fu su
    hgu : ContDiffOn 𝕜 (↑n) gu su
    Nfu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i fu su xu)) (Norm.no …
    Ngu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i gu su xu)) (Norm.no …
    NBu : Eq (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (Bu (fu y)) (gu y)) su …
    this : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  exact this.trans (mul_le_mul_of_nonneg_right Bu_le (by positivity))
  /-
    🎉 no goals
  -/


/-- Bounding the norm of the iterated derivative of `B (f x) (g x)` in terms of the
iterated derivatives of `f` and `g` when `B` is bilinear:
`‖D^n (x ↦ B (f x) (g x))‖ ≤ ‖B‖ ∑_{k ≤ n} n.choose k ‖D^k f‖ ‖D^{n-k} g‖` -/
theorem ContinuousLinearMap.norm_iteratedFDeriv_le_of_bilinear (B : E →L[𝕜] F →L[𝕜] G) {f : D → E}
    {g : D → F} {N : WithTop ℕ∞} (hf : ContDiff 𝕜 N f) (hg : ContDiff 𝕜 N g) (x : D) {n : ℕ}
    (hn : n ≤ N) :
    ‖iteratedFDeriv 𝕜 n (fun y => B (f y) (g y)) x‖ ≤ ‖B‖ * ∑ i ∈ Finset.range (n + 1),
      (n.choose i : ℝ) * ‖iteratedFDeriv 𝕜 i f x‖ * ‖iteratedFDeriv 𝕜 (n - i) g x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    hf : ContDiff 𝕜 N f
    hg : ContDiff 𝕜 N g
    x : D
    n : Nat
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (fun y => (B (f y)) (g y)) x)) (HMul.hM …
  -/
  simp_rw [← iteratedFDerivWithin_univ]
  exact B.norm_iteratedFDerivWithin_le_of_bilinear hf.contDiffOn hg.contDiffOn uniqueDiffOn_univ
    (mem_univ x) hn


/-- Bounding the norm of the iterated derivative of `B (f x) (g x)` within a set in terms of the
iterated derivatives of `f` and `g` when `B` is bilinear of norm at most `1`:
`‖D^n (x ↦ B (f x) (g x))‖ ≤ ∑_{k ≤ n} n.choose k ‖D^k f‖ ‖D^{n-k} g‖` -/
theorem ContinuousLinearMap.norm_iteratedFDerivWithin_le_of_bilinear_of_le_one
    (B : E →L[𝕜] F →L[𝕜] G) {f : D → E} {g : D → F} {N : WithTop ℕ∞} {s : Set D} {x : D}
    (hf : ContDiffOn 𝕜 N f s) (hg : ContDiffOn 𝕜 N g s) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ}
    (hn : n ≤ N) (hB : ‖B‖ ≤ 1) : ‖iteratedFDerivWithin 𝕜 n (fun y => B (f y) (g y)) s x‖ ≤
      ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    hB : LE.le (Norm.norm B) 1
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (B (f y)) (g y)) s x))  …
  -/
  apply (B.norm_iteratedFDerivWithin_le_of_bilinear hf hg hs hx hn).trans
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    s : Set D
    x : D
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    hn : LE.le (↑n) N
    hB : LE.le (Norm.norm B) 1
    ⊢ LE.le (HMul.hMul (Norm.norm B) ((Finset.range (HAdd.hAdd n 1)).sum fun i =>  …
  -/
  exact mul_le_of_le_one_left (by positivity) hB
  /-
    🎉 no goals
  -/


/-- Bounding the norm of the iterated derivative of `B (f x) (g x)` in terms of the
iterated derivatives of `f` and `g` when `B` is bilinear of norm at most `1`:
`‖D^n (x ↦ B (f x) (g x))‖ ≤ ∑_{k ≤ n} n.choose k ‖D^k f‖ ‖D^{n-k} g‖` -/
theorem ContinuousLinearMap.norm_iteratedFDeriv_le_of_bilinear_of_le_one (B : E →L[𝕜] F →L[𝕜] G)
    {f : D → E} {g : D → F} {N : WithTop ℕ∞} (hf : ContDiff 𝕜 N f) (hg : ContDiff 𝕜 N g)
    (x : D) {n : ℕ} (hn : n ≤ N) (hB : ‖B‖ ≤ 1) :
    ‖iteratedFDeriv 𝕜 n (fun y => B (f y) (g y)) x‖ ≤
      ∑ i ∈ Finset.range (n + 1),
        (n.choose i : ℝ) * ‖iteratedFDeriv 𝕜 i f x‖ * ‖iteratedFDeriv 𝕜 (n - i) g x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : D → E
    g : D → F
    N : WithTop ENat
    hf : ContDiff 𝕜 N f
    hg : ContDiff 𝕜 N g
    x : D
    n : Nat
    hn : LE.le (↑n) N
    hB : LE.le (Norm.norm B) 1
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (fun y => (B (f y)) (g y)) x)) ((Finset …
  -/
  simp_rw [← iteratedFDerivWithin_univ]
  exact B.norm_iteratedFDerivWithin_le_of_bilinear_of_le_one hf.contDiffOn hg.contDiffOn
    uniqueDiffOn_univ (mem_univ x) hn hB


theorem norm_iteratedFDerivWithin_smul_le {f : E → 𝕜'} {g : E → F} {N : WithTop ℕ∞}
    (hf : ContDiffOn 𝕜 N f s) (hg : ContDiffOn 𝕜 N g s) (hs : UniqueDiffOn 𝕜 s) {x : E} (hx : x ∈ s)
    {n : ℕ} (hn : n ≤ N) : ‖iteratedFDerivWithin 𝕜 n (fun y => f y • g y) s x‖ ≤
      ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ :=
  (ContinuousLinearMap.lsmul 𝕜 𝕜' :
    𝕜' →L[𝕜] F →L[𝕜] F).norm_iteratedFDerivWithin_le_of_bilinear_of_le_one
      hf hg hs hx hn ContinuousLinearMap.opNorm_lsmul_le


theorem norm_iteratedFDeriv_smul_le {f : E → 𝕜'} {g : E → F} {N : WithTop ℕ∞} (hf : ContDiff 𝕜 N f)
    (hg : ContDiff 𝕜 N g) (x : E) {n : ℕ} (hn : n ≤ N) :
    ‖iteratedFDeriv 𝕜 n (fun y => f y • g y) x‖ ≤ ∑ i ∈ Finset.range (n + 1),
      (n.choose i : ℝ) * ‖iteratedFDeriv 𝕜 i f x‖ * ‖iteratedFDeriv 𝕜 (n - i) g x‖ :=
  (ContinuousLinearMap.lsmul 𝕜 𝕜' : 𝕜' →L[𝕜] F →L[𝕜] F).norm_iteratedFDeriv_le_of_bilinear_of_le_one
    hf hg x hn ContinuousLinearMap.opNorm_lsmul_le


theorem norm_iteratedFDerivWithin_mul_le {f : E → A} {g : E → A} {N : WithTop ℕ∞}
    (hf : ContDiffOn 𝕜 N f s) (hg : ContDiffOn 𝕜 N g s) (hs : UniqueDiffOn 𝕜 s)
    {x : E} (hx : x ∈ s) {n : ℕ} (hn : n ≤ N) :
    ‖iteratedFDerivWithin 𝕜 n (fun y => f y * g y) s x‖ ≤
      ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ :=
  (ContinuousLinearMap.mul 𝕜 A :
    A →L[𝕜] A →L[𝕜] A).norm_iteratedFDerivWithin_le_of_bilinear_of_le_one
      hf hg hs hx hn (ContinuousLinearMap.opNorm_mul_le _ _)


theorem norm_iteratedFDeriv_mul_le {f : E → A} {g : E → A} {N : WithTop ℕ∞} (hf : ContDiff 𝕜 N f)
    (hg : ContDiff 𝕜 N g) (x : E) {n : ℕ} (hn : n ≤ N) :
    ‖iteratedFDeriv 𝕜 n (fun y => f y * g y) x‖ ≤ ∑ i ∈ Finset.range (n + 1),
      (n.choose i : ℝ) * ‖iteratedFDeriv 𝕜 i f x‖ * ‖iteratedFDeriv 𝕜 (n - i) g x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    A : Type u_3
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    f g : E → A
    N : WithTop ENat
    hf : ContDiff 𝕜 N f
    hg : ContDiff 𝕜 N g
    x : E
    n : Nat
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (fun y => HMul.hMul (f y) (g y)) x)) (( …
  -/
  simp_rw [← iteratedFDerivWithin_univ]
  exact norm_iteratedFDerivWithin_mul_le
    hf.contDiffOn hg.contDiffOn uniqueDiffOn_univ (mem_univ x) hn

-- TODO: Add `norm_iteratedFDeriv[Within]_list_prod_le` for non-commutative `NormedRing A`.


theorem norm_iteratedFDerivWithin_prod_le [DecidableEq ι] [NormOneClass A'] {u : Finset ι}
    {f : ι → E → A'} {N : WithTop ℕ∞} (hf : ∀ i ∈ u, ContDiffOn 𝕜 N (f i) s)
    (hs : UniqueDiffOn 𝕜 s) {x : E} (hx : x ∈ s) {n : ℕ} (hn : n ≤ N) :
    ‖iteratedFDerivWithin 𝕜 n (∏ j ∈ u, f j ·) s x‖ ≤
      ∑ p ∈ u.sym n, (p : Multiset ι).multinomial *
        ∏ j ∈ u, ‖iteratedFDerivWithin 𝕜 (Multiset.count j p) (f j) s x‖ := by
  induction u using Finset.induction generalizing n with
  | empty =>
    cases n with
    | zero => simp [Sym.eq_nil_of_card_zero]
    | succ n => simp [iteratedFDerivWithin_succ_const]
  | @insert i u hi IH =>
    conv => lhs; simp only [Finset.prod_insert hi]
    simp only [Finset.mem_insert, forall_eq_or_imp] at hf
    refine le_trans (norm_iteratedFDerivWithin_mul_le hf.1 (contDiffOn_prod hf.2) hs hx hn) ?_
    rw [← Finset.sum_coe_sort (Finset.sym _ _)]
    rw [Finset.sum_equiv (Finset.symInsertEquiv hi) (t := Finset.univ)
      (g := (fun v ↦ v.multinomial *
          ∏ j ∈ insert i u, ‖iteratedFDerivWithin 𝕜 (v.count j) (f j) s x‖) ∘
        Sym.toMultiset ∘ Subtype.val ∘ (Finset.symInsertEquiv hi).symm)
      (by simp) (by simp only [← comp_apply (g := Finset.symInsertEquiv hi), comp_assoc]; simp)]
    rw [← Finset.univ_sigma_univ, Finset.sum_sigma, Finset.sum_range]
    simp only [comp_apply, Finset.symInsertEquiv_symm_apply_coe]
    refine Finset.sum_le_sum ?_
    intro m _
    specialize IH hf.2 (n := n - m) (le_trans (by exact_mod_cast n.sub_le m) hn)
    refine le_trans (mul_le_mul_of_nonneg_left IH (by simp [mul_nonneg])) ?_
    rw [Finset.mul_sum, ← Finset.sum_coe_sort]
    refine Finset.sum_le_sum ?_
    simp only [Finset.mem_univ, forall_true_left, Subtype.forall, Finset.mem_sym_iff]
    intro p hp
    refine le_of_eq ?_
    rw [Finset.prod_insert hi]
    have hip : i ∉ p := mt (hp i) hi
    rw [Sym.count_coe_fill_self_of_not_mem hip, Sym.multinomial_coe_fill_of_not_mem hip]
    suffices ∏ j ∈ u, ‖iteratedFDerivWithin 𝕜 (Multiset.count j p) (f j) s x‖ =
        ∏ j ∈ u, ‖iteratedFDerivWithin 𝕜 (Multiset.count j (Sym.fill i m p)) (f j) s x‖ by
      rw [this, Nat.cast_mul]
      ring
    refine Finset.prod_congr rfl ?_
    intro j hj
    have hji : j ≠ i := mt (· ▸ hj) hi
    rw [Sym.count_coe_fill_of_ne hji]


theorem norm_iteratedFDeriv_prod_le [DecidableEq ι] [NormOneClass A'] {u : Finset ι}
    {f : ι → E → A'} {N : WithTop ℕ∞} (hf : ∀ i ∈ u, ContDiff 𝕜 N (f i)) {x : E} {n : ℕ}
    (hn : n ≤ N) :
    ‖iteratedFDeriv 𝕜 n (∏ j ∈ u, f j ·) x‖ ≤
      ∑ p ∈ u.sym n, (p : Multiset ι).multinomial *
        ∏ j ∈ u, ‖iteratedFDeriv 𝕜 ((p : Multiset ι).count j) (f j) x‖ := by
  simpa [iteratedFDerivWithin_univ] using
    norm_iteratedFDerivWithin_prod_le (fun i hi ↦ (hf i hi).contDiffOn) uniqueDiffOn_univ
      (mem_univ x) hn


/-- If the derivatives within a set of `g` at `f x` are bounded by `C`, and the `i`-th derivative
within a set of `f` at `x` is bounded by `D^i` for all `1 ≤ i ≤ n`, then the `n`-th derivative
of `g ∘ f` is bounded by `n! * C * D^n`.
This lemma proves this estimate assuming additionally that two of the spaces live in the same
universe, to make an induction possible. Use instead `norm_iteratedFDerivWithin_comp_le` that
removes this assumption. -/
theorem norm_iteratedFDerivWithin_comp_le_aux {Fu Gu : Type u} [NormedAddCommGroup Fu]
    [NormedSpace 𝕜 Fu] [NormedAddCommGroup Gu] [NormedSpace 𝕜 Gu] {g : Fu → Gu} {f : E → Fu} {n : ℕ}
    {s : Set E} {t : Set Fu} {x : E} (hg : ContDiffOn 𝕜 n g t) (hf : ContDiffOn 𝕜 n f s)
    (ht : UniqueDiffOn 𝕜 t) (hs : UniqueDiffOn 𝕜 s) (hst : MapsTo f s t) (hx : x ∈ s) {C : ℝ}
    {D : ℝ} (hC : ∀ i, i ≤ n → ‖iteratedFDerivWithin 𝕜 i g t (f x)‖ ≤ C)
    (hD : ∀ i, 1 ≤ i → i ≤ n → ‖iteratedFDerivWithin 𝕜 i f s x‖ ≤ D ^ i) :
    ‖iteratedFDerivWithin 𝕜 n (g ∘ f) s x‖ ≤ n ! * C * D ^ n := by
  /- We argue by induction on `n`, using that `D^(n+1) (g ∘ f) = D^n (g ' ∘ f ⬝ f')`. The successive
    derivatives of `g' ∘ f` are controlled thanks to the inductive assumption, and those of `f'` are
    controlled by assumption.
    As composition of linear maps is a bilinear map, one may use
    `ContinuousLinearMap.norm_iteratedFDeriv_le_of_bilinear_of_le_one` to get from these a bound
    on `D^n (g ' ∘ f ⬝ f')`. -/
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    Fu Gu : Type u
    inst✝³ : NormedAddCommGroup Fu
    inst✝² : NormedSpace 𝕜 Fu
    inst✝¹ : NormedAddCommGroup Gu
    inst✝ : NormedSpace 𝕜 Gu
    g : Fu → Gu
    f : E → Fu
    n : Nat
    s : Set E
    t : Set Fu
    x : E
    hg : ContDiffOn 𝕜 (↑n) g t
    hf : ContDiffOn 𝕜 (↑n) f s
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  induction' n using Nat.case_strong_induction_on with n IH generalizing Gu
  · simpa [norm_iteratedFDerivWithin_zero, Nat.factorial_zero, algebraMap.coe_one, one_mul,
      pow_zero, mul_one, comp_apply] using hC 0 le_rfl
  /-
    case hi
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    Fu : Type u
    inst✝³ : NormedAddCommGroup Fu
    inst✝² : NormedSpace 𝕜 Fu
    f : E → Fu
    s : Set E
    t : Set Fu
    x : E
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    n : Nat
    IH : ∀ (m : Nat), LE.le m n → ∀ {Gu : Type u} [inst : NormedAddCommGroup Gu] [ …
    Gu : Type u
    inst✝¹ : NormedAddCommGroup Gu
    inst✝ : NormedSpace 𝕜 Gu
    g : Fu → Gu
    hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g t
    hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
    hC : ∀ (i : Nat), LE.le i (HAdd.hAdd n 1) → LE.le (Norm.norm (iteratedFDerivWi …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i (HAdd.hAdd n 1) → LE.le (Norm.norm (iter …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) (Function.comp g f) …
  -/
  have M : (n : WithTop ℕ∞) < n.succ := Nat.cast_lt.2 n.lt_succ_self
  /-
    case hi
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    Fu : Type u
    inst✝³ : NormedAddCommGroup Fu
    inst✝² : NormedSpace 𝕜 Fu
    f : E → Fu
    s : Set E
    t : Set Fu
    x : E
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    n : Nat
    IH : ∀ (m : Nat), LE.le m n → ∀ {Gu : Type u} [inst : NormedAddCommGroup Gu] [ …
    Gu : Type u
    inst✝¹ : NormedAddCommGroup Gu
    inst✝ : NormedSpace 𝕜 Gu
    g : Fu → Gu
    hg : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) g t
    hf : ContDiffOn 𝕜 (↑(HAdd.hAdd n 1)) f s
    hC : ∀ (i : Nat), LE.le i (HAdd.hAdd n 1) → LE.le (Norm.norm (iteratedFDerivWi …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i (HAdd.hAdd n 1) → LE.le (Norm.norm (iter …
    M : LT.lt ↑n ↑n.succ
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) (Function.comp g f) …
  -/
  have Cnonneg : 0 ≤ C := (norm_nonneg _).trans (hC 0 bot_le)
  have Dnonneg : 0 ≤ D := by
    have : 1 ≤ n + 1 := by simp only [le_add_iff_nonneg_left, zero_le']
    simpa only [pow_one] using (norm_nonneg _).trans (hD 1 le_rfl this)
  -- use the inductive assumption to bound the derivatives of `g' ∘ f`.
  have I : ∀ i ∈ Finset.range (n + 1),
      ‖iteratedFDerivWithin 𝕜 i (fderivWithin 𝕜 g t ∘ f) s x‖ ≤ i ! * C * D ^ i := by
    intro i hi
    simp only [Finset.mem_range_succ_iff] at hi
    apply IH i hi
    · apply hg.fderivWithin ht
      simp only [Nat.cast_succ]
      exact add_le_add_right (Nat.cast_le.2 hi) _
    · apply hf.of_le (Nat.cast_le.2 (hi.trans n.le_succ))
    · intro j hj
      have : ‖iteratedFDerivWithin 𝕜 j (fderivWithin 𝕜 g t) t (f x)‖ =
          ‖iteratedFDerivWithin 𝕜 (j + 1) g t (f x)‖ := by
        rw [iteratedFDerivWithin_succ_eq_comp_right ht (hst hx), comp_apply,
          LinearIsometryEquiv.norm_map]
      rw [this]
      exact hC (j + 1) (add_le_add (hj.trans hi) le_rfl)
    · intro j hj h'j
      exact hD j hj (h'j.trans (hi.trans n.le_succ))
  -- reformulate `hD` as a bound for the derivatives of `f'`.
  have J : ∀ i, ‖iteratedFDerivWithin 𝕜 (n - i) (fderivWithin 𝕜 f s) s x‖ ≤ D ^ (n - i + 1) := by
    intro i
    have : ‖iteratedFDerivWithin 𝕜 (n - i) (fderivWithin 𝕜 f s) s x‖ =
        ‖iteratedFDerivWithin 𝕜 (n - i + 1) f s x‖ := by
      rw [iteratedFDerivWithin_succ_eq_comp_right hs hx, comp_apply, LinearIsometryEquiv.norm_map]
    rw [this]
    apply hD
    · simp only [le_add_iff_nonneg_left, zero_le']
    · apply Nat.succ_le_succ tsub_le_self
  -- Now put these together: first, notice that we have to bound `D^n (g' ∘ f ⬝ f')`.
  calc
    ‖iteratedFDerivWithin 𝕜 (n + 1) (g ∘ f) s x‖ =
        ‖iteratedFDerivWithin 𝕜 n (fun y : E => fderivWithin 𝕜 (g ∘ f) s y) s x‖ := by
      rw [iteratedFDerivWithin_succ_eq_comp_right hs hx, comp_apply,
        LinearIsometryEquiv.norm_map]
    _ = ‖iteratedFDerivWithin 𝕜 n (fun y : E => ContinuousLinearMap.compL 𝕜 E Fu Gu
        (fderivWithin 𝕜 g t (f y)) (fderivWithin 𝕜 f s y)) s x‖ := by
      have L : (1 : WithTop ℕ∞) ≤ n.succ := by
        simpa only [ENat.coe_one, Nat.one_le_cast] using n.succ_pos
      congr 1
      refine iteratedFDerivWithin_congr (fun y hy => ?_) hx _
      apply fderivWithin_comp _ _ _ hst (hs y hy)
      · exact hg.differentiableOn L _ (hst hy)
      · exact hf.differentiableOn L _ hy
    -- bound it using the fact that the composition of linear maps is a bilinear operation,
    -- for which we have bounds for the`n`-th derivative.
    _ ≤ ∑ i ∈ Finset.range (n + 1),
        (n.choose i : ℝ) * ‖iteratedFDerivWithin 𝕜 i (fderivWithin 𝕜 g t ∘ f) s x‖ *
          ‖iteratedFDerivWithin 𝕜 (n - i) (fderivWithin 𝕜 f s) s x‖ := by
      have A : ContDiffOn 𝕜 n (fderivWithin 𝕜 g t ∘ f) s := by
        apply ContDiffOn.comp _ (hf.of_le M.le) hst
        apply hg.fderivWithin ht
        simp only [Nat.cast_succ, le_refl]
      have B : ContDiffOn 𝕜 n (fderivWithin 𝕜 f s) s := by
        apply hf.fderivWithin hs
        simp only [Nat.cast_succ, le_refl]
      exact (ContinuousLinearMap.compL 𝕜 E Fu Gu).norm_iteratedFDerivWithin_le_of_bilinear_of_le_one
        A B hs hx le_rfl (ContinuousLinearMap.norm_compL_le 𝕜 E Fu Gu)
    -- bound each of the terms using the estimates on previous derivatives (that use the inductive
    -- assumption for `g' ∘ f`).
    _ ≤ ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * (i ! * C * D ^ i) * D ^ (n - i + 1) := by
      gcongr with i hi
      · exact I i hi
      · exact J i
    -- We are left with trivial algebraic manipulations to see that this is smaller than
    -- the claimed bound.
    _ = ∑ i ∈ Finset.range (n + 1),
      -- Porting note: had to insert a few more explicit type ascriptions in this and similar
      -- expressions.
        (n ! : ℝ) * ((i ! : ℝ)⁻¹ * i !) * C * (D ^ i * D ^ (n - i + 1)) * ((n - i)! : ℝ)⁻¹ := by
      congr! 1 with i hi
      simp only [Nat.cast_choose ℝ (Finset.mem_range_succ_iff.1 hi), div_eq_inv_mul, mul_inv]
      ring
    _ = ∑ i ∈ Finset.range (n + 1), (n ! : ℝ) * 1 * C * D ^ (n + 1) * ((n - i)! : ℝ)⁻¹ := by
      congr! with i hi
      · apply inv_mul_cancel₀
        simpa only [Ne, Nat.cast_eq_zero] using i.factorial_ne_zero
      · rw [← pow_add]
        congr 1
        rw [Nat.add_succ, Nat.succ_inj']
        exact Nat.add_sub_of_le (Finset.mem_range_succ_iff.1 hi)
    _ ≤ ∑ i ∈ Finset.range (n + 1), (n ! : ℝ) * 1 * C * D ^ (n + 1) * 1 := by
      gcongr with i
      apply inv_le_one_of_one_le₀
      simpa only [Nat.one_le_cast] using (n - i).factorial_pos
    _ = (n + 1)! * C * D ^ (n + 1) := by
      simp only [mul_assoc, mul_one, Finset.sum_const, Finset.card_range, nsmul_eq_mul,
        Nat.factorial_succ, Nat.cast_mul]


/-- If the derivatives within a set of `g` at `f x` are bounded by `C`, and the `i`-th derivative
within a set of `f` at `x` is bounded by `D^i` for all `1 ≤ i ≤ n`, then the `n`-th derivative
of `g ∘ f` is bounded by `n! * C * D^n`. -/
theorem norm_iteratedFDerivWithin_comp_le {g : F → G} {f : E → F} {n : ℕ} {s : Set E} {t : Set F}
    {x : E} {N : WithTop ℕ∞} (hg : ContDiffOn 𝕜 N g t) (hf : ContDiffOn 𝕜 N f s) (hn : n ≤ N)
    (ht : UniqueDiffOn 𝕜 t) (hs : UniqueDiffOn 𝕜 s) (hst : MapsTo f s t) (hx : x ∈ s) {C : ℝ}
    {D : ℝ} (hC : ∀ i, i ≤ n → ‖iteratedFDerivWithin 𝕜 i g t (f x)‖ ≤ C)
    (hD : ∀ i, 1 ≤ i → i ≤ n → ‖iteratedFDerivWithin 𝕜 i f s x‖ ≤ D ^ i) :
    ‖iteratedFDerivWithin 𝕜 n (g ∘ f) s x‖ ≤ n ! * C * D ^ n := by
  /- We reduce the bound to the case where all spaces live in the same universe (in which we
    already have proved the result), by using linear isometries between the spaces and their `ULift`
    to a common universe. These linear isometries preserve the norm of the iterated derivative. -/
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  let Fu : Type max uF uG := ULift.{uG, uF} F
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  let Gu : Type max uF uG := ULift.{uF, uG} G
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  have isoF : Fu ≃ₗᵢ[𝕜] F := LinearIsometryEquiv.ulift 𝕜 F
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  have isoG : Gu ≃ₗᵢ[𝕜] G := LinearIsometryEquiv.ulift 𝕜 G
  -- lift `f` and `g` to versions `fu` and `gu` on the lifted spaces.
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  let fu : E → Fu := isoF.symm ∘ f
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  let gu : Fu → Gu := isoG.symm ∘ g ∘ isoF
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  let tu := isoF ⁻¹' t
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    tu : Set Fu := Set.preimage (⇑isoF) t
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  have htu : UniqueDiffOn 𝕜 tu := isoF.toContinuousLinearEquiv.uniqueDiffOn_preimage_iff.2 ht
  have hstu : MapsTo fu s tu := fun y hy ↦ by
    simpa only [fu, tu, mem_preimage, comp_apply, LinearIsometryEquiv.apply_symm_apply] using hst hy
  have Ffu : isoF (fu x) = f x := by
    simp only [fu, comp_apply, LinearIsometryEquiv.apply_symm_apply]
  -- All norms are preserved by the lifting process.
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    tu : Set Fu := Set.preimage (⇑isoF) t
    htu : UniqueDiffOn 𝕜 tu
    hstu : Set.MapsTo fu s tu
    Ffu : Eq (isoF (fu x)) (f x)
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  have hfu : ContDiffOn 𝕜 n fu s := isoF.symm.contDiff.comp_contDiffOn (hf.of_le hn)
  have hgu : ContDiffOn 𝕜 n gu tu :=
    isoG.symm.contDiff.comp_contDiffOn
      ((hg.of_le hn).comp_continuousLinearMap (isoF : Fu →L[𝕜] F))
  have Nfu : ∀ i, ‖iteratedFDerivWithin 𝕜 i fu s x‖ = ‖iteratedFDerivWithin 𝕜 i f s x‖ := fun i ↦ by
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_left _ _ hs hx]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    tu : Set Fu := Set.preimage (⇑isoF) t
    htu : UniqueDiffOn 𝕜 tu
    hstu : Set.MapsTo fu s tu
    Ffu : Eq (isoF (fu x)) (f x)
    hfu : ContDiffOn 𝕜 (↑n) fu s
    hgu : ContDiffOn 𝕜 (↑n) gu tu
    Nfu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i fu s x)) (Norm.norm …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  simp_rw [← Nfu] at hD
  have Ngu : ∀ i,
      ‖iteratedFDerivWithin 𝕜 i gu tu (fu x)‖ = ‖iteratedFDerivWithin 𝕜 i g t (f x)‖ := fun i ↦ by
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_left _ _ htu (hstu hx)]
    rw [LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_right _ _ ht, Ffu]
    rw [Ffu]
    exact hst hx
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i g t ( …
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    tu : Set Fu := Set.preimage (⇑isoF) t
    htu : UniqueDiffOn 𝕜 tu
    hstu : Set.MapsTo fu s tu
    Ffu : Eq (isoF (fu x)) (f x)
    hfu : ContDiffOn 𝕜 (↑n) fu s
    hgu : ContDiffOn 𝕜 (↑n) gu tu
    Nfu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i fu s x)) (Norm.norm …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Ngu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i gu tu (fu x))) (Nor …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  simp_rw [← Ngu] at hC
  have Nfgu :
      ‖iteratedFDerivWithin 𝕜 n (g ∘ f) s x‖ = ‖iteratedFDerivWithin 𝕜 n (gu ∘ fu) s x‖ := by
    have : gu ∘ fu = isoG.symm ∘ g ∘ f := by
      ext x
      simp only [fu, gu, comp_apply, LinearIsometryEquiv.map_eq_iff,
        LinearIsometryEquiv.apply_symm_apply]
    rw [this, LinearIsometryEquiv.norm_iteratedFDerivWithin_comp_left _ _ hs hx]
  -- deduce the required bound from the one for `gu ∘ fu`.
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    tu : Set Fu := Set.preimage (⇑isoF) t
    htu : UniqueDiffOn 𝕜 tu
    hstu : Set.MapsTo fu s tu
    Ffu : Eq (isoF (fu x)) (f x)
    hfu : ContDiffOn 𝕜 (↑n) fu s
    hgu : ContDiffOn 𝕜 (↑n) gu tu
    Nfu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i fu s x)) (Norm.norm …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Ngu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i gu tu (fu x))) (Nor …
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i gu tu …
    Nfgu : Eq (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (Norm …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (HMul.h …
  -/
  rw [Nfgu]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    s : Set E
    t : Set F
    x : E
    N : WithTop ENat
    hg : ContDiffOn 𝕜 N g t
    hf : ContDiffOn 𝕜 N f s
    hn : LE.le (↑n) N
    ht : UniqueDiffOn 𝕜 t
    hs : UniqueDiffOn 𝕜 s
    hst : Set.MapsTo f s t
    hx : Membership.mem s x
    C D : Real
    Fu : Type (max uF uG) := ULift.{uG, uF} F
    Gu : Type (max uF uG) := ULift.{uF, uG} G
    isoF : LinearIsometryEquiv (RingHom.id 𝕜) Fu F
    isoG : LinearIsometryEquiv (RingHom.id 𝕜) Gu G
    fu : E → Fu := Function.comp (⇑isoF.symm) f
    gu : Fu → Gu := Function.comp (⇑isoG.symm) (Function.comp g ⇑isoF)
    tu : Set Fu := Set.preimage (⇑isoF) t
    htu : UniqueDiffOn 𝕜 tu
    hstu : Set.MapsTo fu s tu
    Ffu : Eq (isoF (fu x)) (f x)
    hfu : ContDiffOn 𝕜 (↑n) fu s
    hgu : ContDiffOn 𝕜 (↑n) gu tu
    Nfu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i fu s x)) (Norm.norm …
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDerivWith …
    Ngu : ∀ (i : Nat), Eq (Norm.norm (iteratedFDerivWithin 𝕜 i gu tu (fu x))) (Nor …
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDerivWithin 𝕜 i gu tu …
    Nfgu : Eq (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp g f) s x)) (Norm …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp gu fu) s x)) (HMul …
  -/
  exact norm_iteratedFDerivWithin_comp_le_aux hgu hfu htu hs hstu hx hC hD
  /-
    🎉 no goals
  -/


/-- If the derivatives of `g` at `f x` are bounded by `C`, and the `i`-th derivative
of `f` at `x` is bounded by `D^i` for all `1 ≤ i ≤ n`, then the `n`-th derivative
of `g ∘ f` is bounded by `n! * C * D^n`. -/
theorem norm_iteratedFDeriv_comp_le {g : F → G} {f : E → F} {n : ℕ} {N : WithTop ℕ∞}
    (hg : ContDiff 𝕜 N g) (hf : ContDiff 𝕜 N f) (hn : n ≤ N) (x : E) {C : ℝ} {D : ℝ}
    (hC : ∀ i, i ≤ n → ‖iteratedFDeriv 𝕜 i g (f x)‖ ≤ C)
    (hD : ∀ i, 1 ≤ i → i ≤ n → ‖iteratedFDeriv 𝕜 i f x‖ ≤ D ^ i) :
    ‖iteratedFDeriv 𝕜 n (g ∘ f) x‖ ≤ n ! * C * D ^ n := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    n : Nat
    N : WithTop ENat
    hg : ContDiff 𝕜 N g
    hf : ContDiff 𝕜 N f
    hn : LE.le (↑n) N
    x : E
    C D : Real
    hC : ∀ (i : Nat), LE.le i n → LE.le (Norm.norm (iteratedFDeriv 𝕜 i g (f x))) C
    hD : ∀ (i : Nat), LE.le 1 i → LE.le i n → LE.le (Norm.norm (iteratedFDeriv 𝕜 i …
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (Function.comp g f) x)) (HMul.hMul (HMu …
  -/
  simp_rw [← iteratedFDerivWithin_univ] at hC hD ⊢
  exact norm_iteratedFDerivWithin_comp_le hg.contDiffOn hf.contDiffOn hn uniqueDiffOn_univ
    uniqueDiffOn_univ (mapsTo_univ _ _) (mem_univ x) hC hD


theorem norm_iteratedFDerivWithin_clm_apply {f : E → F →L[𝕜] G} {g : E → F} {s : Set E} {x : E}
    {N : WithTop ℕ∞} {n : ℕ} (hf : ContDiffOn 𝕜 N f s) (hg : ContDiffOn 𝕜 N g s)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) (hn : n ≤ N) :
    ‖iteratedFDerivWithin 𝕜 n (fun y => (f y) (g y)) s x‖ ≤
      ∑ i ∈ Finset.range (n + 1), ↑(n.choose i) * ‖iteratedFDerivWithin 𝕜 i f s x‖ *
        ‖iteratedFDerivWithin 𝕜 (n - i) g s x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    g : E → F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (f y) (g y)) s x)) ((Fi …
  -/
  let B : (F →L[𝕜] G) →L[𝕜] F →L[𝕜] G := ContinuousLinearMap.flip (ContinuousLinearMap.apply 𝕜 G)
  have hB : ‖B‖ ≤ 1 := by
    simp only [B, ContinuousLinearMap.opNorm_flip, ContinuousLinearMap.apply]
    refine ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun f => ?_
    simp only [ContinuousLinearMap.coe_id', id, one_mul]
    rfl
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    g : E → F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hg : ContDiffOn 𝕜 N g s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    B : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    hB : LE.le (Norm.norm B) 1
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (f y) (g y)) s x)) ((Fi …
  -/
  exact B.norm_iteratedFDerivWithin_le_of_bilinear_of_le_one hf hg hs hx hn hB
  /-
    🎉 no goals
  -/


theorem norm_iteratedFDeriv_clm_apply {f : E → F →L[𝕜] G} {g : E → F} {N : WithTop ℕ∞} {n : ℕ}
    (hf : ContDiff 𝕜 N f) (hg : ContDiff 𝕜 N g) (x : E) (hn : n ≤ N) :
    ‖iteratedFDeriv 𝕜 n (fun y : E => (f y) (g y)) x‖ ≤ ∑ i ∈ Finset.range (n + 1),
      ↑(n.choose i) * ‖iteratedFDeriv 𝕜 i f x‖ * ‖iteratedFDeriv 𝕜 (n - i) g x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    g : E → F
    N : WithTop ENat
    n : Nat
    hf : ContDiff 𝕜 N f
    hg : ContDiff 𝕜 N g
    x : E
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (fun y => (f y) (g y)) x)) ((Finset.ran …
  -/
  simp only [← iteratedFDerivWithin_univ]
  exact norm_iteratedFDerivWithin_clm_apply hf.contDiffOn hg.contDiffOn uniqueDiffOn_univ
    (Set.mem_univ x) hn


theorem norm_iteratedFDerivWithin_clm_apply_const {f : E → F →L[𝕜] G} {c : F} {s : Set E} {x : E}
    {N : WithTop ℕ∞} {n : ℕ} (hf : ContDiffOn 𝕜 N f s) (hs : UniqueDiffOn 𝕜 s)
    (hx : x ∈ s) (hn : n ≤ N) :
    ‖iteratedFDerivWithin 𝕜 n (fun y : E => (f y) c) s x‖ ≤
      ‖c‖ * ‖iteratedFDerivWithin 𝕜 n f s x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (f y) c) s x)) (HMul.hM …
  -/
  let g : (F →L[𝕜] G) →L[𝕜] G := ContinuousLinearMap.apply 𝕜 G c
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    g : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (f y) c) s x)) (HMul.hM …
  -/
  have h := g.norm_compContinuousMultilinearMap_le (iteratedFDerivWithin 𝕜 n f s x)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    g : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    h : LE.le (Norm.norm (g.compContinuousMultilinearMap (iteratedFDerivWithin 𝕜 n …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (f y) c) s x)) (HMul.hM …
  -/
  rw [← g.iteratedFDerivWithin_comp_left hf hs hx hn] at h
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    g : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    h : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp (⇑g) f) s x)) (H …
    ⊢ LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (fun y => (f y) c) s x)) (HMul.hM …
  -/
  refine h.trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    g : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    h : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp (⇑g) f) s x)) (H …
    ⊢ LE.le (Norm.norm g) (Norm.norm c)
  -/
  refine g.opNorm_le_bound (norm_nonneg _) fun f => ?_
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f✝ : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f✝ s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    g : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    h : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp (⇑g) f✝) s x)) ( …
    f : ContinuousLinearMap (RingHom.id 𝕜) F G
    ⊢ LE.le (Norm.norm (g f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
  -/
  rw [ContinuousLinearMap.apply_apply, mul_comm]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f✝ : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    s : Set E
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiffOn 𝕜 N f✝ s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    hn : LE.le (↑n) N
    g : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) F G …
    h : LE.le (Norm.norm (iteratedFDerivWithin 𝕜 n (Function.comp (⇑g) f✝) s x)) ( …
    f : ContinuousLinearMap (RingHom.id 𝕜) F G
    ⊢ LE.le (Norm.norm (f c)) (HMul.hMul (Norm.norm f) (Norm.norm c))
  -/
  exact f.le_opNorm c
  /-
    🎉 no goals
  -/


theorem norm_iteratedFDeriv_clm_apply_const {f : E → F →L[𝕜] G} {c : F} {x : E}
    {N : WithTop ℕ∞} {n : ℕ} (hf : ContDiff 𝕜 N f) (hn : n ≤ N) :
    ‖iteratedFDeriv 𝕜 n (fun y : E => (f y) c) x‖ ≤ ‖c‖ * ‖iteratedFDeriv 𝕜 n f x‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type uG
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → ContinuousLinearMap (RingHom.id 𝕜) F G
    c : F
    x : E
    N : WithTop ENat
    n : Nat
    hf : ContDiff 𝕜 N f
    hn : LE.le (↑n) N
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (fun y => (f y) c) x)) (HMul.hMul (Norm …
  -/
  simp only [← iteratedFDerivWithin_univ]
  exact norm_iteratedFDerivWithin_clm_apply_const hf.contDiffOn uniqueDiffOn_univ
    (Set.mem_univ x) hn


