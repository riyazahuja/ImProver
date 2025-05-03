/-- Suppose that a function `f : ℂ → E` is continuous on a closed rectangle with opposite corners at
`z w : ℂ`, is *real* differentiable at all but countably many points of the corresponding open
rectangle, and $\frac{\partial f}{\partial \bar z}$ is integrable on this rectangle. Then the
integral of `f` over the boundary of the rectangle is equal to the integral of
$2i\frac{\partial f}{\partial \bar z}=i\frac{\partial f}{\partial x}-\frac{\partial f}{\partial y}$
over the rectangle. -/
theorem integral_boundary_rect_of_hasFDerivAt_real_off_countable (f : ℂ → E) (f' : ℂ → ℂ →L[ℝ] E)
    (z w : ℂ) (s : Set ℂ) (hs : s.Countable)
    (Hc : ContinuousOn f ([[z.re, w.re]] ×ℂ [[z.im, w.im]]))
    (Hd : ∀ x ∈ Ioo (min z.re w.re) (max z.re w.re) ×ℂ Ioo (min z.im w.im) (max z.im w.im) \ s,
      HasFDerivAt f (f' x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            inst✝ : CompleteSpace E
            f : Complex → E
            f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
            z w : Complex
            s : Set Complex
            hs : s.Countable
            Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
            Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
            ⊢ MeasureTheory.Measure Complex
          -/
    (Hi : IntegrableOn (fun z => I • f' z 1 - f' z I) ([[z.re, w.re]] ×ℂ [[z.im, w.im]])) :
          /-
            🎉 no goals
          -/
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (re w + y * I)) -
      I • ∫ y : ℝ in z.im..w.im, f (re z + y * I) =
      ∫ x : ℝ in z.re..w.re, ∫ y : ℝ in z.im..w.im, I • f' (x + y * I) 1 - f' (x + y * I) I := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (HAdd.hAdd …
  -/
  set e : (ℝ × ℝ) ≃L[ℝ] ℂ := equivRealProdCLM.symm
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (HAdd.hAdd …
  -/
  have he : ∀ x y : ℝ, ↑x + ↑y * I = e (x, y) := fun x y => (mk_eq_add_mul_I x y).symm
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he : ∀ (x y : Real), Eq (HAdd.hAdd (↑x) (HMul.hMul (↑y) Complex.I)) (e { fst : …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (HAdd.hAdd …
  -/
  have he₁ : e (1, 0) = 1 := rfl; have he₂ : e (0, 1) = I := rfl
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he : ∀ (x y : Real), Eq (HAdd.hAdd (↑x) (HMul.hMul (↑y) Complex.I)) (e { fst : …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (HAdd.hAdd …
  -/
  simp only [he] at *
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  set F : ℝ × ℝ → E := f ∘ e
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  set F' : ℝ × ℝ → ℝ × ℝ →L[ℝ] E := fun p => (f' (e p)).comp (e : ℝ × ℝ →L[ℝ] ℂ)
  have hF' : ∀ p : ℝ × ℝ, (-(I • F' p)) (1, 0) + F' p (0, 1) = -(I • f' (e p) 1 - f' (e p) I) := by
    rintro ⟨x, y⟩
    simp only [F', ContinuousLinearMap.neg_apply, ContinuousLinearMap.smul_apply,
      ContinuousLinearMap.comp_apply, ContinuousLinearEquiv.coe_coe, he₁, he₂, neg_add_eq_sub,
      neg_sub]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    F' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real) E …
    hF' : ∀ (p : Prod Real Real), Eq (HAdd.hAdd ((Neg.neg (HSMul.hSMul Complex.I ( …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  set R : Set (ℝ × ℝ) := [[z.re, w.re]] ×ˢ [[w.im, z.im]]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    F' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real) E …
    hF' : ∀ (p : Prod Real Real), Eq (HAdd.hAdd ((Neg.neg (HSMul.hSMul Complex.I ( …
    R : Set (Prod Real Real) := SProd.sprod (Set.uIcc z.re w.re) (Set.uIcc w.im z. …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  set t : Set (ℝ × ℝ) := e ⁻¹' s
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    F' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real) E …
    hF' : ∀ (p : Prod Real Real), Eq (HAdd.hAdd ((Neg.neg (HSMul.hSMul Complex.I ( …
    R : Set (Prod Real Real) := SProd.sprod (Set.uIcc z.re w.re) (Set.uIcc w.im z. …
    t : Set (Prod Real Real) := Set.preimage (⇑e) s
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  rw [uIcc_comm z.im] at Hc Hi; rw [min_comm z.im, max_comm z.im] at Hd
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc w.im z.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    F' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real) E …
    hF' : ∀ (p : Prod Real Real), Eq (HAdd.hAdd ((Neg.neg (HSMul.hSMul Complex.I ( …
    R : Set (Prod Real Real) := SProd.sprod (Set.uIcc z.re w.re) (Set.uIcc w.im z. …
    t : Set (Prod Real Real) := Set.preimage (⇑e) s
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  have hR : e ⁻¹' ([[z.re, w.re]] ×ℂ [[w.im, z.im]]) = R := rfl
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc w.im z.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    F' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real) E …
    hF' : ∀ (p : Prod Real Real), Eq (HAdd.hAdd ((Neg.neg (HSMul.hSMul Complex.I ( …
    R : Set (Prod Real Real) := SProd.sprod (Set.uIcc z.re w.re) (Set.uIcc w.im z. …
    t : Set (Prod Real Real) := Set.preimage (⇑e) s
    hR : Eq (Set.preimage (⇑e) (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc w. …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (intervalIntegral (fun x => f (e { fst : …
  -/
  have htc : ContinuousOn F R := Hc.comp e.continuousOn hR.ge
  have htd :
    ∀ p ∈ Ioo (min z.re w.re) (max z.re w.re) ×ˢ Ioo (min w.im z.im) (max w.im z.im) \ t,
      HasFDerivAt F (F' p) p :=
    fun p hp => (Hd (e p) hp).comp p e.hasFDerivAt
  simp_rw [← intervalIntegral.integral_smul, intervalIntegral.integral_symm w.im z.im, ←
    intervalIntegral.integral_neg, ← hF']
  refine (integral2_divergence_prod_of_hasFDerivWithinAt_off_countable (fun p => -(I • F p)) F
    (fun p => -(I • F' p)) F' z.re w.im w.re z.im t (hs.preimage e.injective)
    (htc.const_smul _).neg htc (fun p hp => ((htd p hp).const_smul I).neg) htd ?_).symm
  rw [← (volume_preserving_equiv_real_prod.symm _).integrableOn_comp_preimage
    (MeasurableEquiv.measurableEmbedding _)] at Hi
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
    z w : Complex
    s : Set Complex
    hs : s.Countable
    Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc w.im z.im))
    Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
    Hi : MeasureTheory.IntegrableOn (Function.comp (fun z => HSub.hSub (HSMul.hSMu …
    e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) Complex := Comple …
    he₁ : Eq (e { fst := 1, snd := 0 }) 1
    he₂ : Eq (e { fst := 0, snd := 1 }) Complex.I
    he : Real → Real → True
    F : Prod Real Real → E := Function.comp f ⇑e
    F' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real) E …
    hF' : ∀ (p : Prod Real Real), Eq (HAdd.hAdd ((Neg.neg (HSMul.hSMul Complex.I ( …
    R : Set (Prod Real Real) := SProd.sprod (Set.uIcc z.re w.re) (Set.uIcc w.im z. …
    t : Set (Prod Real Real) := Set.preimage (⇑e) s
    hR : Eq (Set.preimage (⇑e) (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc w. …
    htc : ContinuousOn F R
    htd : ∀ (p : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    ⊢ MeasureTheory.IntegrableOn (fun x => HAdd.hAdd (((fun p => Neg.neg (HSMul.hS …
  -/
  simpa only [hF'] using Hi.neg
  /-
    🎉 no goals
  -/


/-- Suppose that a function `f : ℂ → E` is continuous on a closed rectangle with opposite corners at
`z w : ℂ`, is *real* differentiable on the corresponding open rectangle, and
$\frac{\partial f}{\partial \bar z}$ is integrable on this rectangle. Then the integral of `f` over
the boundary of the rectangle is equal to the integral of
$2i\frac{\partial f}{\partial \bar z}=i\frac{\partial f}{\partial x}-\frac{\partial f}{\partial y}$
over the rectangle. -/
theorem integral_boundary_rect_of_continuousOn_of_hasFDerivAt_real (f : ℂ → E) (f' : ℂ → ℂ →L[ℝ] E)
    (z w : ℂ) (Hc : ContinuousOn f ([[z.re, w.re]] ×ℂ [[z.im, w.im]]))
    (Hd : ∀ x ∈ Ioo (min z.re w.re) (max z.re w.re) ×ℂ Ioo (min z.im w.im) (max z.im w.im),
      HasFDerivAt f (f' x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            inst✝ : CompleteSpace E
            f : Complex → E
            f' : Complex → ContinuousLinearMap (RingHom.id Real) Complex E
            z w : Complex
            Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
            Hd : ∀ (x : Complex), Membership.mem (Complex.reProdIm (Set.Ioo (Min.min z.re  …
            ⊢ MeasureTheory.Measure Complex
          -/
    (Hi : IntegrableOn (fun z => I • f' z 1 - f' z I) ([[z.re, w.re]] ×ℂ [[z.im, w.im]])) :
          /-
            🎉 no goals
          -/
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (re w + y * I)) -
      I • (∫ y : ℝ in z.im..w.im, f (re z + y * I)) =
      ∫ x : ℝ in z.re..w.re, ∫ y : ℝ in z.im..w.im, I • f' (x + y * I) 1 - f' (x + y * I) I :=
  integral_boundary_rect_of_hasFDerivAt_real_off_countable f f' z w ∅ countable_empty Hc
    (fun x hx => Hd x hx.1) Hi


/-- Suppose that a function `f : ℂ → E` is *real* differentiable on a closed rectangle with opposite
corners at `z w : ℂ` and $\frac{\partial f}{\partial \bar z}$ is integrable on this rectangle. Then
the integral of `f` over the boundary of the rectangle is equal to the integral of
$2i\frac{\partial f}{\partial \bar z}=i\frac{\partial f}{\partial x}-\frac{\partial f}{\partial y}$
over the rectangle. -/
theorem integral_boundary_rect_of_differentiableOn_real (f : ℂ → E) (z w : ℂ)
    (Hd : DifferentiableOn ℝ f ([[z.re, w.re]] ×ℂ [[z.im, w.im]]))
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            inst✝ : CompleteSpace E
            f : Complex → E
            z w : Complex
            Hd : DifferentiableOn Real f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc  …
            ⊢ MeasureTheory.Measure Complex
          -/
    (Hi : IntegrableOn (fun z => I • fderiv ℝ f z 1 - fderiv ℝ f z I)
          /-
            🎉 no goals
          -/
      ([[z.re, w.re]] ×ℂ [[z.im, w.im]])) :
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (re w + y * I)) -
      I • (∫ y : ℝ in z.im..w.im, f (re z + y * I)) =
      ∫ x : ℝ in z.re..w.re, ∫ y : ℝ in z.im..w.im,
        I • fderiv ℝ f (x + y * I) 1 - fderiv ℝ f (x + y * I) I :=
  integral_boundary_rect_of_hasFDerivAt_real_off_countable f (fderiv ℝ f) z w ∅ countable_empty
    Hd.continuousOn
    (fun x hx => Hd.hasFDerivAt <| by
      /-
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Complex E
        inst✝ : CompleteSpace E
        f : Complex → E
        z w : Complex
        Hd : DifferentiableOn Real f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc  …
        Hi : MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I ((f …
        x : Complex
        hx : Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo (Min.min z.re w.re …
        ⊢ Membership.mem (nhds x) (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.i …
      -/
      simpa only [← mem_interior_iff_mem_nhds, interior_reProdIm, uIcc, interior_Icc] using hx.1)
      /-
        🎉 no goals
      -/
    Hi


/-- **Cauchy-Goursat theorem** for a rectangle: the integral of a complex differentiable function
over the boundary of a rectangle equals zero. More precisely, if `f` is continuous on a closed
rectangle and is complex differentiable at all but countably many points of the corresponding open
rectangle, then its integral over the boundary of the rectangle equals zero. -/
theorem integral_boundary_rect_eq_zero_of_differentiable_on_off_countable (f : ℂ → E) (z w : ℂ)
    (s : Set ℂ) (hs : s.Countable) (Hc : ContinuousOn f ([[z.re, w.re]] ×ℂ [[z.im, w.im]]))
    (Hd : ∀ x ∈ Ioo (min z.re w.re) (max z.re w.re) ×ℂ Ioo (min z.im w.im) (max z.im w.im) \ s,
      DifferentiableAt ℂ f x) :
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (re w + y * I)) -
      I • (∫ y : ℝ in z.im..w.im, f (re z + y * I)) = 0 := by
  refine (integral_boundary_rect_of_hasFDerivAt_real_off_countable f
    (fun z => (fderiv ℂ f z).restrictScalars ℝ) z w s hs Hc
    (fun x hx => (Hd x hx).hasFDerivAt.restrictScalars ℝ) ?_).trans ?_ <;>
      /-
        case refine_1
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Complex E
        inst✝ : CompleteSpace E
        f : Complex → E
        z w : Complex
        s : Set Complex
        hs : s.Countable
        Hc : ContinuousOn f (Complex.reProdIm (Set.uIcc z.re w.re) (Set.uIcc z.im w.im))
        Hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Complex.reProdIm (Set.Ioo ( …
        ⊢ MeasureTheory.IntegrableOn (fun z => HSub.hSub (HSMul.hSMul Complex.I (((fun …
      -/
      /-
        🎉 no goals
      -/
      simp [← ContinuousLinearMap.map_smul]
      /-
        🎉 no goals
      -/


/-- **Cauchy-Goursat theorem for a rectangle**: the integral of a complex differentiable function
over the boundary of a rectangle equals zero. More precisely, if `f` is continuous on a closed
rectangle and is complex differentiable on the corresponding open rectangle, then its integral over
the boundary of the rectangle equals zero. -/
theorem integral_boundary_rect_eq_zero_of_continuousOn_of_differentiableOn (f : ℂ → E) (z w : ℂ)
    (Hc : ContinuousOn f ([[z.re, w.re]] ×ℂ [[z.im, w.im]]))
    (Hd : DifferentiableOn ℂ f
      (Ioo (min z.re w.re) (max z.re w.re) ×ℂ Ioo (min z.im w.im) (max z.im w.im))) :
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (re w + y * I)) -
      I • (∫ y : ℝ in z.im..w.im, f (re z + y * I)) = 0 :=
  integral_boundary_rect_eq_zero_of_differentiable_on_off_countable f z w ∅ countable_empty Hc
    fun _x hx => Hd.differentiableAt <| (isOpen_Ioo.reProdIm isOpen_Ioo).mem_nhds hx.1


/-- **Cauchy-Goursat theorem** for a rectangle: the integral of a complex differentiable function
over the boundary of a rectangle equals zero. More precisely, if `f` is complex differentiable on a
closed rectangle, then its integral over the boundary of the rectangle equals zero. -/
theorem integral_boundary_rect_eq_zero_of_differentiableOn (f : ℂ → E) (z w : ℂ)
    (H : DifferentiableOn ℂ f ([[z.re, w.re]] ×ℂ [[z.im, w.im]])) :
    (∫ x : ℝ in z.re..w.re, f (x + z.im * I)) - (∫ x : ℝ in z.re..w.re, f (x + w.im * I)) +
      I • (∫ y : ℝ in z.im..w.im, f (re w + y * I)) -
      I • (∫ y : ℝ in z.im..w.im, f (re z + y * I)) = 0 :=
  integral_boundary_rect_eq_zero_of_continuousOn_of_differentiableOn f z w H.continuousOn <|
    H.mono <|
      inter_subset_inter (preimage_mono Ioo_subset_Icc_self) (preimage_mono Ioo_subset_Icc_self)


/-- If `f : ℂ → E` is continuous on the closed annulus `r ≤ ‖z - c‖ ≤ R`, `0 < r ≤ R`,
and is complex differentiable at all but countably many points of its interior,
then the integrals of `f z / (z - c)` (formally, `(z - c)⁻¹ • f z`)
over the circles `‖z - c‖ = r` and `‖z - c‖ = R` are equal to each other. -/
theorem circleIntegral_sub_center_inv_smul_eq_of_differentiable_on_annulus_off_countable {c : ℂ}
    {r R : ℝ} (h0 : 0 < r) (hle : r ≤ R) {f : ℂ → E} {s : Set ℂ} (hs : s.Countable)
    (hc : ContinuousOn f (closedBall c R \ ball c r))
    (hd : ∀ z ∈ (ball c R \ closedBall c r) \ s, DifferentiableAt ℂ f z) :
    (∮ z in C(c, R), (z - c)⁻¹ • f z) = ∮ z in C(c, r), (z - c)⁻¹ • f z := by
  /- We apply the previous lemma to `fun z ↦ f (c + exp z)` on the rectangle
    `[log r, log R] × [0, 2 * π]`. -/
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    r R : Real
    h0 : LT.lt 0 r
    hle : LE.le r R
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hc : ContinuousOn f (SDiff.sdiff (Metric.closedBall c R) (Metric.ball c r))
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z c)) (f z)) c  …
  -/
  set A := closedBall c R \ ball c r
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    r R : Real
    h0 : LT.lt 0 r
    hle : LE.le r R
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c R) (Metric.ball c r)
    hc : ContinuousOn f A
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z c)) (f z)) c  …
  -/
  obtain ⟨a, rfl⟩ : ∃ a, Real.exp a = r := ⟨Real.log r, Real.exp_log h0⟩
  /-
    case intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    R : Real
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    hle : LE.le (Real.exp a) R
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c R) (Metric.ball c (Real.ex …
    hc : ContinuousOn f A
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z c)) (f z)) c  …
  -/
  obtain ⟨b, rfl⟩ : ∃ b, Real.exp b = R := ⟨Real.log R, Real.exp_log (h0.trans_le hle)⟩
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le (Real.exp a) (Real.exp b)
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z c)) (f z)) c  …
  -/
  rw [Real.exp_le_exp] at hle
  -- Unfold definition of `circleIntegral` and cancel some terms.
  suffices
    (∫ θ in (0)..2 * π, I • f (circleMap c (Real.exp b) θ)) =
      ∫ θ in (0)..2 * π, I • f (circleMap c (Real.exp a) θ) by
    simpa only [circleIntegral, add_sub_cancel_left, ofReal_exp, ← exp_add, smul_smul, ←
      div_eq_mul_inv, mul_div_cancel_left₀ _ (circleMap_ne_center (Real.exp_pos _).ne'),
      circleMap_sub_center, deriv_circleMap]
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le a b
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul Complex.I (f (circleMap c (Real.e …
  -/
  set R := [[a, b]] ×ℂ [[0, 2 * π]]
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le a b
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    R : Set Complex := Complex.reProdIm (Set.uIcc a b) (Set.uIcc 0 (HMul.hMul 2 Re …
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul Complex.I (f (circleMap c (Real.e …
  -/
  set g : ℂ → ℂ := (c + exp ·)
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le a b
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    R : Set Complex := Complex.reProdIm (Set.uIcc a b) (Set.uIcc 0 (HMul.hMul 2 Re …
    g : Complex → Complex := fun x => HAdd.hAdd c (Complex.exp x)
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul Complex.I (f (circleMap c (Real.e …
  -/
  have hdg : Differentiable ℂ g := differentiable_exp.const_add _
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le a b
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    R : Set Complex := Complex.reProdIm (Set.uIcc a b) (Set.uIcc 0 (HMul.hMul 2 Re …
    g : Complex → Complex := fun x => HAdd.hAdd c (Complex.exp x)
    hdg : Differentiable Complex g
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul Complex.I (f (circleMap c (Real.e …
  -/
  replace hs : (g ⁻¹' s).Countable := (hs.preimage (add_right_injective c)).preimage_cexp
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le a b
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    R : Set Complex := Complex.reProdIm (Set.uIcc a b) (Set.uIcc 0 (HMul.hMul 2 Re …
    g : Complex → Complex := fun x => HAdd.hAdd c (Complex.exp x)
    hdg : Differentiable Complex g
    hs : (Set.preimage g s).Countable
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul Complex.I (f (circleMap c (Real.e …
  -/
  have h_maps : MapsTo g R A := by rintro z ⟨h, -⟩; simpa [g, A, dist_eq, abs_exp, hle] using h.symm
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    f : Complex → E
    s : Set Complex
    a : Real
    h0 : LT.lt 0 (Real.exp a)
    b : Real
    hle : LE.le a b
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    A : Set Complex := SDiff.sdiff (Metric.closedBall c (Real.exp b)) (Metric.ball …
    hc : ContinuousOn f A
    R : Set Complex := Complex.reProdIm (Set.uIcc a b) (Set.uIcc 0 (HMul.hMul 2 Re …
    g : Complex → Complex := fun x => HAdd.hAdd c (Complex.exp x)
    hdg : Differentiable Complex g
    hs : (Set.preimage g s).Countable
    h_maps : Set.MapsTo g R A
    ⊢ Eq (intervalIntegral (fun θ => HSMul.hSMul Complex.I (f (circleMap c (Real.e …
  -/
  replace hc : ContinuousOn (f ∘ g) R := hc.comp hdg.continuous.continuousOn h_maps
  replace hd : ∀ z ∈ Ioo (min a b) (max a b) ×ℂ Ioo (min 0 (2 * π)) (max 0 (2 * π)) \ g ⁻¹' s,
      DifferentiableAt ℂ (f ∘ g) z := by
    refine fun z hz => (hd (g z) ⟨?_, hz.2⟩).comp z (hdg _)
    simpa [g, dist_eq, abs_exp, hle, and_comm] using hz.1.1
  simpa [g, circleMap, exp_periodic _, sub_eq_zero, ← exp_add] using
    integral_boundary_rect_eq_zero_of_differentiable_on_off_countable _ ⟨a, 0⟩ ⟨b, 2 * π⟩ _ hs hc hd


/-- **Cauchy-Goursat theorem** for an annulus. If `f : ℂ → E` is continuous on the closed annulus
`r ≤ ‖z - c‖ ≤ R`, `0 < r ≤ R`, and is complex differentiable at all but countably many points of
its interior, then the integrals of `f` over the circles `‖z - c‖ = r` and `‖z - c‖ = R` are equal
to each other. -/
theorem circleIntegral_eq_of_differentiable_on_annulus_off_countable {c : ℂ} {r R : ℝ} (h0 : 0 < r)
    (hle : r ≤ R) {f : ℂ → E} {s : Set ℂ} (hs : s.Countable)
    (hc : ContinuousOn f (closedBall c R \ ball c r))
    (hd : ∀ z ∈ (ball c R \ closedBall c r) \ s, DifferentiableAt ℂ f z) :
    (∮ z in C(c, R), f z) = ∮ z in C(c, r), f z :=
  calc
    (∮ z in C(c, R), f z) = ∮ z in C(c, R), (z - c)⁻¹ • (z - c) • f z :=
      (circleIntegral.integral_sub_inv_smul_sub_smul _ _ _ _).symm
    _ = ∮ z in C(c, r), (z - c)⁻¹ • (z - c) • f z :=
      (circleIntegral_sub_center_inv_smul_eq_of_differentiable_on_annulus_off_countable h0 hle hs
        ((continuousOn_id.sub continuousOn_const).smul hc) fun z hz =>
        (differentiableAt_id.sub_const _).smul (hd z hz))
    _ = ∮ z in C(c, r), f z := circleIntegral.integral_sub_inv_smul_sub_smul _ _ _ _


/-- **Cauchy integral formula** for the value at the center of a disc. If `f` is continuous on a
punctured closed disc of radius `R`, is differentiable at all but countably many points of the
interior of this disc, and has a limit `y` at the center of the disc, then the integral
$\oint_{‖z-c‖=R} \frac{f(z)}{z-c}\,dz$ is equal to `2πiy`. -/
theorem circleIntegral_sub_center_inv_smul_of_differentiable_on_off_countable_of_tendsto {c : ℂ}
    {R : ℝ} (h0 : 0 < R) {f : ℂ → E} {y : E} {s : Set ℂ} (hs : s.Countable)
    (hc : ContinuousOn f (closedBall c R \ {c}))
    (hd : ∀ z ∈ (ball c R \ {c}) \ s, DifferentiableAt ℂ f z) (hy : Tendsto f (𝓝[{c}ᶜ] c) (𝓝 y)) :
    (∮ z in C(c, R), (z - c)⁻¹ • f z) = (2 * π * I : ℂ) • y := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    R : Real
    h0 : LT.lt 0 R
    f : Complex → E
    y : E
    s : Set Complex
    hs : s.Countable
    hc : ContinuousOn f (SDiff.sdiff (Metric.closedBall c R) (Singleton.singleton  …
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    hy : Filter.Tendsto f (nhdsWithin c (HasCompl.compl (Singleton.singleton c)))  …
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z c)) (f z)) c  …
  -/
  rw [← sub_eq_zero, ← norm_le_zero_iff]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    c : Complex
    R : Real
    h0 : LT.lt 0 R
    f : Complex → E
    y : E
    s : Set Complex
    hs : s.Countable
    hc : ContinuousOn f (SDiff.sdiff (Metric.closedBall c R) (Singleton.singleton  …
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (SDiff.sdiff (Metric.ball c  …
    hy : Filter.Tendsto f (nhdsWithin c (HasCompl.compl (Singleton.singleton c)))  …
    ⊢ LE.le (Norm.norm (HSub.hSub (circleIntegral (fun z => HSMul.hSMul (Inv.inv ( …
  -/
  refine le_of_forall_le_of_dense fun ε ε0 => ?_
  obtain ⟨δ, δ0, hδ⟩ : ∃ δ > (0 : ℝ), ∀ z ∈ closedBall c δ \ {c}, dist (f z) y < ε / (2 * π) :=
    ((nhdsWithin_hasBasis nhds_basis_closedBall _).tendsto_iff nhds_basis_ball).1 hy _
      (div_pos ε0 Real.two_pi_pos)
  obtain ⟨r, hr0, hrδ, hrR⟩ : ∃ r, 0 < r ∧ r ≤ δ ∧ r ≤ R :=
    ⟨min δ R, lt_min δ0 h0, min_le_left _ _, min_le_right _ _⟩
  have hsub : closedBall c R \ ball c r ⊆ closedBall c R \ {c} :=
    diff_subset_diff_right (singleton_subset_iff.2 <| mem_ball_self hr0)
  have hsub' : ball c R \ closedBall c r ⊆ ball c R \ {c} :=
    diff_subset_diff_right (singleton_subset_iff.2 <| mem_closedBall_self hr0.le)
  have hzne : ∀ z ∈ sphere c r, z ≠ c := fun z hz =>
    ne_of_mem_of_not_mem hz fun h => hr0.ne' <| dist_self c ▸ Eq.symm h
  /- The integral `∮ z in C(c, r), f z / (z - c)` does not depend on `0 < r ≤ R` and tends to
    `2πIy` as `r → 0`. -/
  calc
    ‖(∮ z in C(c, R), (z - c)⁻¹ • f z) - (2 * ↑π * I) • y‖ =
        ‖(∮ z in C(c, r), (z - c)⁻¹ • f z) - ∮ z in C(c, r), (z - c)⁻¹ • y‖ := by
      congr 2
      · exact circleIntegral_sub_center_inv_smul_eq_of_differentiable_on_annulus_off_countable hr0
          hrR hs (hc.mono hsub) fun z hz => hd z ⟨hsub' hz.1, hz.2⟩
      · simp [hr0.ne']
    _ = ‖∮ z in C(c, r), (z - c)⁻¹ • (f z - y)‖ := by
      simp only [smul_sub]
      have hc' : ContinuousOn (fun z => (z - c)⁻¹) (sphere c r) :=
        (continuousOn_id.sub continuousOn_const).inv₀ fun z hz => sub_ne_zero.2 <| hzne _ hz
      rw [circleIntegral.integral_sub] <;> refine (hc'.smul ?_).circleIntegrable hr0.le
      · exact hc.mono <| subset_inter
          (sphere_subset_closedBall.trans <| closedBall_subset_closedBall hrR) hzne
      · exact continuousOn_const
    _ ≤ 2 * π * r * (r⁻¹ * (ε / (2 * π))) := by
      refine circleIntegral.norm_integral_le_of_norm_le_const hr0.le fun z hz => ?_
      specialize hzne z hz
      rw [mem_sphere, dist_eq_norm] at hz
      rw [norm_smul, norm_inv, hz, ← dist_eq_norm]
      refine mul_le_mul_of_nonneg_left (hδ _ ⟨?_, hzne⟩).le (inv_nonneg.2 hr0.le)
      rwa [mem_closedBall_iff_norm, hz]
    _ = ε := by field_simp [hr0.ne', Real.two_pi_pos.ne']; ac_rfl


/-- **Cauchy integral formula** for the value at the center of a disc. If `f : ℂ → E` is continuous
on a closed disc of radius `R` and is complex differentiable at all but countably many points of its
interior, then the integral $\oint_{|z-c|=R} \frac{f(z)}{z-c}\,dz$ is equal to `2πiy`. -/
theorem circleIntegral_sub_center_inv_smul_of_differentiable_on_off_countable {R : ℝ} (h0 : 0 < R)
    {f : ℂ → E} {c : ℂ} {s : Set ℂ} (hs : s.Countable) (hc : ContinuousOn f (closedBall c R))
    (hd : ∀ z ∈ ball c R \ s, DifferentiableAt ℂ f z) :
    (∮ z in C(c, R), (z - c)⁻¹ • f z) = (2 * π * I : ℂ) • f c :=
  circleIntegral_sub_center_inv_smul_of_differentiable_on_off_countable_of_tendsto h0 hs
    (hc.mono diff_subset) (fun z hz => hd z ⟨hz.1.1, hz.2⟩)
    (hc.continuousAt <| closedBall_mem_nhds _ h0).continuousWithinAt


/-- **Cauchy-Goursat theorem** for a disk: if `f : ℂ → E` is continuous on a closed disk
`{z | ‖z - c‖ ≤ R}` and is complex differentiable at all but countably many points of its interior,
then the integral $\oint_{|z-c|=R}f(z)\,dz$ equals zero. -/
theorem circleIntegral_eq_zero_of_differentiable_on_off_countable {R : ℝ} (h0 : 0 ≤ R) {f : ℂ → E}
    {c : ℂ} {s : Set ℂ} (hs : s.Countable) (hc : ContinuousOn f (closedBall c R))
    (hd : ∀ z ∈ ball c R \ s, DifferentiableAt ℂ f z) : (∮ z in C(c, R), f z) = 0 := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    h0 : LE.le 0 R
    f : Complex → E
    c : Complex
    s : Set Complex
    hs : s.Countable
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (z : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) z → Dif …
    ⊢ Eq (circleIntegral (fun z => f z) c R) 0
  -/
  rcases h0.eq_or_lt with (rfl | h0); · apply circleIntegral.integral_radius_zero
                                        /-
                                          🎉 no goals
                                        -/
  calc
    (∮ z in C(c, R), f z) = ∮ z in C(c, R), (z - c)⁻¹ • (z - c) • f z :=
      (circleIntegral.integral_sub_inv_smul_sub_smul _ _ _ _).symm
    _ = (2 * ↑π * I : ℂ) • (c - c) • f c :=
      (circleIntegral_sub_center_inv_smul_of_differentiable_on_off_countable h0 hs
        ((continuousOn_id.sub continuousOn_const).smul hc) fun z hz =>
        (differentiableAt_id.sub_const _).smul (hd z hz))
    _ = 0 := by rw [sub_self, zero_smul, smul_zero]


/-- An auxiliary lemma for
`Complex.circleIntegral_sub_inv_smul_of_differentiable_on_off_countable`. This lemma assumes
`w ∉ s` while the main lemma drops this assumption. -/
theorem circleIntegral_sub_inv_smul_of_differentiable_on_off_countable_aux {R : ℝ} {c w : ℂ}
    {f : ℂ → E} {s : Set ℂ} (hs : s.Countable) (hw : w ∈ ball c R \ s)
    (hc : ContinuousOn f (closedBall c R)) (hd : ∀ x ∈ ball c R \ s, DifferentiableAt ℂ f x) :
    (∮ z in C(c, R), (z - w)⁻¹ • f z) = (2 * π * I : ℂ) • f w := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (SDiff.sdiff (Metric.ball c R) s) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (f z)) c  …
  -/
  have hR : 0 < R := dist_nonneg.trans_lt hw.1
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (SDiff.sdiff (Metric.ball c R) s) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (f z)) c  …
  -/
  set F : ℂ → E := dslope f w
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (SDiff.sdiff (Metric.ball c R) s) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    F : Complex → E := dslope f w
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (f z)) c  …
  -/
  have hws : (insert w s).Countable := hs.insert w
  have hcF : ContinuousOn F (closedBall c R) :=
    (continuousOn_dslope <| closedBall_mem_nhds_of_mem hw.1).2 ⟨hc, hd _ hw⟩
  have hdF : ∀ z ∈ ball (c : ℂ) R \ insert w s, DifferentiableAt ℂ F z := fun z hz =>
    (differentiableAt_dslope_of_ne (ne_of_mem_of_not_mem (mem_insert _ _) hz.2).symm).2
      (hd _ (diff_subset_diff_right (subset_insert _ _) hz))
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (SDiff.sdiff (Metric.ball c R) s) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    F : Complex → E := dslope f w
    hws : (Insert.insert w s).Countable
    hcF : ContinuousOn F (Metric.closedBall c R)
    hdF : ∀ (z : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) (Insert.i …
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (f z)) c  …
  -/
  have HI := circleIntegral_eq_zero_of_differentiable_on_off_countable hR.le hws hcF hdF
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (SDiff.sdiff (Metric.ball c R) s) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    F : Complex → E := dslope f w
    hws : (Insert.insert w s).Countable
    hcF : ContinuousOn F (Metric.closedBall c R)
    hdF : ∀ (z : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) (Insert.i …
    HI : Eq (circleIntegral (fun z => F z) c R) 0
    ⊢ Eq (circleIntegral (fun z => HSMul.hSMul (Inv.inv (HSub.hSub z w)) (f z)) c  …
  -/
  have hne : ∀ z ∈ sphere c R, z ≠ w := fun z hz => ne_of_mem_of_not_mem hz (ne_of_lt hw.1)
  have hFeq : EqOn F (fun z => (z - w)⁻¹ • f z - (z - w)⁻¹ • f w) (sphere c R) := fun z hz ↦
    calc
      F z = (z - w)⁻¹ • (f z - f w) := update_of_ne (hne z hz) ..
      _ = (z - w)⁻¹ • f z - (z - w)⁻¹ • f w := smul_sub _ _ _
  have hc' : ContinuousOn (fun z => (z - w)⁻¹) (sphere c R) :=
    (continuousOn_id.sub continuousOn_const).inv₀ fun z hz => sub_ne_zero.2 <| hne z hz
  rw [← circleIntegral.integral_sub_inv_of_mem_ball hw.1, ← circleIntegral.integral_smul_const, ←
    sub_eq_zero, ← circleIntegral.integral_sub, ← circleIntegral.integral_congr hR.le hFeq, HI]
  exacts [(hc'.smul (hc.mono sphere_subset_closedBall)).circleIntegrable hR.le,
    (hc'.smul continuousOn_const).circleIntegrable hR.le]


/-- **Cauchy integral formula**: if `f : ℂ → E` is continuous on a closed disc of radius `R` and is
complex differentiable at all but countably many points of its interior, then for any `w` in this
interior we have $\frac{1}{2πi}\oint_{|z-c|=R}(z-w)^{-1}f(z)\,dz=f(w)$.
-/
theorem two_pi_I_inv_smul_circleIntegral_sub_inv_smul_of_differentiable_on_off_countable {R : ℝ}
    {c w : ℂ} {f : ℂ → E} {s : Set ℂ} (hs : s.Countable) (hw : w ∈ ball c R)
    (hc : ContinuousOn f (closedBall c R)) (hd : ∀ x ∈ ball c R \ s, DifferentiableAt ℂ f x) :
    ((2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z) = f w := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (Metric.ball c R) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (circ …
  -/
  have hR : 0 < R := dist_nonneg.trans_lt hw
  suffices w ∈ closure (ball c R \ s) by
    lift R to ℝ≥0 using hR.le
    have A : ContinuousAt (fun w => (2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z) w := by
      have := hasFPowerSeriesOn_cauchy_integral
        ((hc.mono sphere_subset_closedBall).circleIntegrable R.coe_nonneg) hR
      refine this.continuousOn.continuousAt (EMetric.isOpen_ball.mem_nhds ?_)
      rwa [Metric.emetric_ball_nnreal]
    have B : ContinuousAt f w := hc.continuousAt (closedBall_mem_nhds_of_mem hw)
    refine tendsto_nhds_unique_of_frequently_eq A B ((mem_closure_iff_frequently.1 this).mono ?_)
    intro z hz
    rw [circleIntegral_sub_inv_smul_of_differentiable_on_off_countable_aux hs hz hc hd,
      inv_smul_smul₀]
    simp [Real.pi_ne_zero, I_ne_zero]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (Metric.ball c R) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    ⊢ Membership.mem (closure (SDiff.sdiff (Metric.ball c R) s)) w
  -/
  refine mem_closure_iff_nhds.2 fun t ht => ?_
  -- TODO: generalize to any vector space over `ℝ`
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (Metric.ball c R) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    t : Set Complex
    ht : Membership.mem (nhds w) t
    ⊢ (Inter.inter t (SDiff.sdiff (Metric.ball c R) s)).Nonempty
  -/
  set g : ℝ → ℂ := fun x => w + ofReal x
  have : Tendsto g (𝓝 0) (𝓝 w) :=
    (continuous_const.add continuous_ofReal).tendsto' 0 w (add_zero _)
  rcases mem_nhds_iff_exists_Ioo_subset.1 (this <| inter_mem ht <| isOpen_ball.mem_nhds hw) with
    ⟨l, u, hlu₀, hlu_sub⟩
  obtain ⟨x, hx⟩ : (Ioo l u \ g ⁻¹' s).Nonempty := by
    refine diff_nonempty.2 fun hsub => ?_
    have : (Ioo l u).Countable :=
      (hs.preimage ((add_right_injective w).comp ofReal_injective)).mono hsub
    rw [← Cardinal.le_aleph0_iff_set_countable, Cardinal.mk_Ioo_real (hlu₀.1.trans hlu₀.2)] at this
    exact this.not_lt Cardinal.aleph0_lt_continuum
  /-
    case intro.intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (Metric.ball c R) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    hR : LT.lt 0 R
    t : Set Complex
    ht : Membership.mem (nhds w) t
    g : Real → Complex := fun x => HAdd.hAdd w ↑x
    this : Filter.Tendsto g (nhds 0) (nhds w)
    l u : Real
    hlu₀ : Membership.mem (Set.Ioo l u) 0
    hlu_sub : HasSubset.Subset (Set.Ioo l u) (Set.preimage g (Inter.inter t (Metri …
    x : Real
    hx : Membership.mem (SDiff.sdiff (Set.Ioo l u) (Set.preimage g s)) x
    ⊢ (Inter.inter t (SDiff.sdiff (Metric.ball c R) s)).Nonempty
  -/
  exact ⟨g x, (hlu_sub hx.1).1, (hlu_sub hx.1).2, hx.2⟩
  /-
    🎉 no goals
  -/


/-- **Cauchy integral formula**: if `f : ℂ → E` is continuous on a closed disc of radius `R` and is
complex differentiable at all but countably many points of its interior, then for any `w` in this
interior we have $\oint_{|z-c|=R}(z-w)^{-1}f(z)\,dz=2πif(w)$.
-/
theorem circleIntegral_sub_inv_smul_of_differentiable_on_off_countable {R : ℝ} {c w : ℂ} {f : ℂ → E}
    {s : Set ℂ} (hs : s.Countable) (hw : w ∈ ball c R) (hc : ContinuousOn f (closedBall c R))
    (hd : ∀ x ∈ ball c R \ s, DifferentiableAt ℂ f x) :
    (∮ z in C(c, R), (z - w)⁻¹ • f z) = (2 * π * I : ℂ) • f w := by
  rw [← two_pi_I_inv_smul_circleIntegral_sub_inv_smul_of_differentiable_on_off_countable
    hs hw hc hd, smul_inv_smul₀]
  /-
    case ha
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    s : Set Complex
    hs : s.Countable
    hw : Membership.mem (Metric.ball c R) w
    hc : ContinuousOn f (Metric.closedBall c R)
    hd : ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) s) x → Dif …
    ⊢ Ne (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0
  -/
  simp [Real.pi_ne_zero, I_ne_zero]
  /-
    🎉 no goals
  -/


/-- **Cauchy integral formula**: if `f : ℂ → E` is complex differentiable on an open disc and is
continuous on its closure, then for any `w` in this open ball we have
$\oint_{|z-c|=R}(z-w)^{-1}f(z)\,dz=2πif(w)$. -/
theorem _root_.DiffContOnCl.circleIntegral_sub_inv_smul {R : ℝ} {c w : ℂ} {f : ℂ → E}
    (h : DiffContOnCl ℂ f (ball c R)) (hw : w ∈ ball c R) :
    (∮ z in C(c, R), (z - w)⁻¹ • f z) = (2 * π * I : ℂ) • f w :=
  circleIntegral_sub_inv_smul_of_differentiable_on_off_countable countable_empty hw
    h.continuousOn_ball fun _x hx => h.differentiableAt isOpen_ball hx.1


/-- **Cauchy integral formula**: if `f : ℂ → E` is complex differentiable on an open disc and is
continuous on its closure, then for any `w` in this open ball we have
$\frac{1}{2πi}\oint_{|z-c|=R}(z-w)^{-1}f(z)\,dz=f(w)$. -/
theorem _root_.DiffContOnCl.two_pi_i_inv_smul_circleIntegral_sub_inv_smul {R : ℝ} {c w : ℂ}
    {f : ℂ → E} (hf : DiffContOnCl ℂ f (ball c R)) (hw : w ∈ ball c R) :
    ((2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - w)⁻¹ • f z) = f w := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    R : Real
    c w : Complex
    f : Complex → E
    hf : DiffContOnCl Complex f (Metric.ball c R)
    hw : Membership.mem (Metric.ball c R) w
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I)) (circ …
  -/
  have hR : 0 < R := not_le.mp (ball_eq_empty.not.mp (Set.nonempty_of_mem hw).ne_empty)
  refine two_pi_I_inv_smul_circleIntegral_sub_inv_smul_of_differentiable_on_off_countable
    countable_empty hw ?_ ?_
    /-
      case refine_1
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      R : Real
      c w : Complex
      f : Complex → E
      hf : DiffContOnCl Complex f (Metric.ball c R)
      hw : Membership.mem (Metric.ball c R) w
      hR : LT.lt 0 R
      ⊢ ContinuousOn f (Metric.closedBall c R)
    -/
  · simpa only [closure_ball c hR.ne.symm] using hf.continuousOn
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      R : Real
      c w : Complex
      f : Complex → E
      hf : DiffContOnCl Complex f (Metric.ball c R)
      hw : Membership.mem (Metric.ball c R) w
      hR : LT.lt 0 R
      ⊢ ∀ (x : Complex), Membership.mem (SDiff.sdiff (Metric.ball c R) EmptyCollecti …
    -/
  · simpa only [diff_empty] using fun z hz => hf.differentiableAt isOpen_ball hz
    /-
      🎉 no goals
    -/


/-- **Cauchy integral formula**: if `f : ℂ → E` is complex differentiable on a closed disc of radius
`R`, then for any `w` in its interior we have $\oint_{|z-c|=R}(z-w)^{-1}f(z)\,dz=2πif(w)$. -/
theorem _root_.DifferentiableOn.circleIntegral_sub_inv_smul {R : ℝ} {c w : ℂ} {f : ℂ → E}
    (hd : DifferentiableOn ℂ f (closedBall c R)) (hw : w ∈ ball c R) :
    (∮ z in C(c, R), (z - w)⁻¹ • f z) = (2 * π * I : ℂ) • f w :=
  (hd.mono closure_ball_subset_closedBall).diffContOnCl.circleIntegral_sub_inv_smul hw


/-- **Cauchy integral formula**: if `f : ℂ → ℂ` is continuous on a closed disc of radius `R` and is
complex differentiable at all but countably many points of its interior, then for any `w` in this
interior we have $\oint_{|z-c|=R}\frac{f(z)}{z-w}dz=2\pi i\,f(w)$.
-/
theorem circleIntegral_div_sub_of_differentiable_on_off_countable {R : ℝ} {c w : ℂ} {s : Set ℂ}
    (hs : s.Countable) (hw : w ∈ ball c R) {f : ℂ → ℂ} (hc : ContinuousOn f (closedBall c R))
    (hd : ∀ z ∈ ball c R \ s, DifferentiableAt ℂ f z) :
    (∮ z in C(c, R), f z / (z - w)) = 2 * π * I * f w := by
  simpa only [smul_eq_mul, div_eq_inv_mul] using
    circleIntegral_sub_inv_smul_of_differentiable_on_off_countable hs hw hc hd


/-- If `f : ℂ → E` is continuous on a closed ball of positive radius and is differentiable at all
but countably many points of the corresponding open ball, then it is analytic on the open ball with
coefficients of the power series given by Cauchy integral formulas. -/
theorem hasFPowerSeriesOnBall_of_differentiable_off_countable {R : ℝ≥0} {c : ℂ} {f : ℂ → E}
    {s : Set ℂ} (hs : s.Countable) (hc : ContinuousOn f (closedBall c R))
    (hd : ∀ z ∈ ball c R \ s, DifferentiableAt ℂ f z) (hR : 0 < R) :
    HasFPowerSeriesOnBall f (cauchyPowerSeries f c R) c R where
  r_le := le_radius_cauchyPowerSeries _ _ _
  r_pos := ENNReal.coe_pos.2 hR
  hasSum := fun {w} hw => by
    have hw' : c + w ∈ ball c R := by
      simpa only [add_mem_ball_iff_norm, ← coe_nnnorm, mem_emetric_ball_zero_iff,
        NNReal.coe_lt_coe, ENNReal.coe_lt_coe] using hw
    rw [← two_pi_I_inv_smul_circleIntegral_sub_inv_smul_of_differentiable_on_off_countable
      hs hw' hc hd]
    exact (hasFPowerSeriesOn_cauchy_integral
      ((hc.mono sphere_subset_closedBall).circleIntegrable R.2) hR).hasSum hw


/-- If `f : ℂ → E` is complex differentiable on an open disc of positive radius and is continuous
on its closure, then it is analytic on the open disc with coefficients of the power series given by
Cauchy integral formulas. -/
theorem _root_.DiffContOnCl.hasFPowerSeriesOnBall {R : ℝ≥0} {c : ℂ} {f : ℂ → E}
    (hf : DiffContOnCl ℂ f (ball c R)) (hR : 0 < R) :
    HasFPowerSeriesOnBall f (cauchyPowerSeries f c R) c R :=
  hasFPowerSeriesOnBall_of_differentiable_off_countable countable_empty hf.continuousOn_ball
    (fun _z hz => hf.differentiableAt isOpen_ball hz.1) hR


/-- If `f : ℂ → E` is complex differentiable on a closed disc of positive radius, then it is
analytic on the corresponding open disc, and the coefficients of the power series are given by
Cauchy integral formulas. See also
`Complex.hasFPowerSeriesOnBall_of_differentiable_off_countable` for a version of this lemma with
weaker assumptions. -/
protected theorem _root_.DifferentiableOn.hasFPowerSeriesOnBall {R : ℝ≥0} {c : ℂ} {f : ℂ → E}
    (hd : DifferentiableOn ℂ f (closedBall c R)) (hR : 0 < R) :
    HasFPowerSeriesOnBall f (cauchyPowerSeries f c R) c R :=
  (hd.mono closure_ball_subset_closedBall).diffContOnCl.hasFPowerSeriesOnBall hR


/-- If `f : ℂ → E` is complex differentiable on some set `s`, then it is analytic at any point `z`
such that `s ∈ 𝓝 z` (equivalently, `z ∈ interior s`). -/
protected theorem _root_.DifferentiableOn.analyticAt {s : Set ℂ} {f : ℂ → E} {z : ℂ}
    (hd : DifferentiableOn ℂ f s) (hz : s ∈ 𝓝 z) : AnalyticAt ℂ f z := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    s : Set Complex
    f : Complex → E
    z : Complex
    hd : DifferentiableOn Complex f s
    hz : Membership.mem (nhds z) s
    ⊢ AnalyticAt Complex f z
  -/
  rcases nhds_basis_closedBall.mem_iff.1 hz with ⟨R, hR0, hRs⟩
  /-
    case intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    s : Set Complex
    f : Complex → E
    z : Complex
    hd : DifferentiableOn Complex f s
    hz : Membership.mem (nhds z) s
    R : Real
    hR0 : LT.lt 0 R
    hRs : HasSubset.Subset (Metric.closedBall z R) s
    ⊢ AnalyticAt Complex f z
  -/
  lift R to ℝ≥0 using hR0.le
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    s : Set Complex
    f : Complex → E
    z : Complex
    hd : DifferentiableOn Complex f s
    hz : Membership.mem (nhds z) s
    R : NNReal
    hR0 : LT.lt 0 ↑R
    hRs : HasSubset.Subset (Metric.closedBall z ↑R) s
    ⊢ AnalyticAt Complex f z
  -/
  exact ((hd.mono hRs).hasFPowerSeriesOnBall hR0).analyticAt
  /-
    🎉 no goals
  -/


theorem _root_.DifferentiableOn.analyticOnNhd {s : Set ℂ} {f : ℂ → E} (hd : DifferentiableOn ℂ f s)
    (hs : IsOpen s) : AnalyticOnNhd ℂ f s := fun _z hz => hd.analyticAt (hs.mem_nhds hz)


theorem _root_.DifferentiableOn.analyticOn {s : Set ℂ} {f : ℂ → E} (hd : DifferentiableOn ℂ f s)
    (hs : IsOpen s) : AnalyticOn ℂ f s :=
  (hd.analyticOnNhd hs).analyticOn


/-- If `f : ℂ → E` is complex differentiable on some open set `s`, then it is continuously
differentiable on `s`. -/
protected theorem _root_.DifferentiableOn.contDiffOn {s : Set ℂ} {f : ℂ → E} {n : WithTop ℕ∞}
    (hd : DifferentiableOn ℂ f s) (hs : IsOpen s) : ContDiffOn ℂ n f s :=
  (hd.analyticOnNhd hs).contDiffOn_of_completeSpace


/-- A complex differentiable function `f : ℂ → E` is analytic at every point. -/
protected theorem _root_.Differentiable.analyticAt {f : ℂ → E} (hf : Differentiable ℂ f) (z : ℂ) :
    AnalyticAt ℂ f z :=
  hf.differentiableOn.analyticAt univ_mem


/-- A complex differentiable function `f : ℂ → E` is continuously differentiable at every point. -/
protected theorem _root_.Differentiable.contDiff
    {f : ℂ → E} (hf : Differentiable ℂ f) {n : WithTop ℕ∞} :
    ContDiff ℂ n f :=
  contDiff_iff_contDiffAt.mpr fun z ↦ (hf.analyticAt z).contDiffAt


/-- When `f : ℂ → E` is differentiable, the `cauchyPowerSeries f z R` represents `f` as a power
series centered at `z` in the entirety of `ℂ`, regardless of `R : ℝ≥0`, with `0 < R`. -/
protected theorem _root_.Differentiable.hasFPowerSeriesOnBall {f : ℂ → E} (h : Differentiable ℂ f)
    (z : ℂ) {R : ℝ≥0} (hR : 0 < R) : HasFPowerSeriesOnBall f (cauchyPowerSeries f z R) z ∞ :=
  (h.differentiableOn.hasFPowerSeriesOnBall hR).r_eq_top_of_exists fun _r hr =>
    ⟨_, h.differentiableOn.hasFPowerSeriesOnBall hr⟩


/-- On an open set, `f : ℂ → E` is analytic iff it is differentiable -/
theorem analyticOnNhd_iff_differentiableOn {f : ℂ → E} {s : Set ℂ} (o : IsOpen s) :
    AnalyticOnNhd ℂ f s ↔ DifferentiableOn ℂ f s :=
  ⟨AnalyticOnNhd.differentiableOn, fun d _ zs ↦ d.analyticAt (o.mem_nhds zs)⟩


/-- On an open set, `f : ℂ → E` is analytic iff it is differentiable -/
theorem analyticOn_iff_differentiableOn {f : ℂ → E} {s : Set ℂ} (o : IsOpen s) :
    AnalyticOn ℂ f s ↔ DifferentiableOn ℂ f s := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    o : IsOpen s
    ⊢ Iff (AnalyticOn Complex f s) (DifferentiableOn Complex f s)
  -/
  rw [o.analyticOn_iff_analyticOnNhd]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    s : Set Complex
    o : IsOpen s
    ⊢ Iff (AnalyticOnNhd Complex f s) (DifferentiableOn Complex f s)
  -/
  exact analyticOnNhd_iff_differentiableOn o
  /-
    🎉 no goals
  -/


/-- `f : ℂ → E` is entire iff it's differentiable -/
theorem analyticOnNhd_univ_iff_differentiable {f : ℂ → E} :
    AnalyticOnNhd ℂ f univ ↔ Differentiable ℂ f := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    ⊢ Iff (AnalyticOnNhd Complex f Set.univ) (Differentiable Complex f)
  -/
  simp only [← differentiableOn_univ]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    ⊢ Iff (AnalyticOnNhd Complex f Set.univ) (DifferentiableOn Complex f Set.univ)
  -/
  exact analyticOnNhd_iff_differentiableOn isOpen_univ
  /-
    🎉 no goals
  -/


theorem analyticOn_univ_iff_differentiable {f : ℂ → E} :
    AnalyticOn ℂ f univ ↔ Differentiable ℂ f := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    ⊢ Iff (AnalyticOn Complex f Set.univ) (Differentiable Complex f)
  -/
  rw [analyticOn_univ]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    ⊢ Iff (AnalyticOnNhd Complex f Set.univ) (Differentiable Complex f)
  -/
  exact analyticOnNhd_univ_iff_differentiable
  /-
    🎉 no goals
  -/


/-- `f : ℂ → E` is analytic at `z` iff it's differentiable near `z` -/
theorem analyticAt_iff_eventually_differentiableAt {f : ℂ → E} {c : ℂ} :
    AnalyticAt ℂ f c ↔ ∀ᶠ z in 𝓝 c, DifferentiableAt ℂ f z := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    ⊢ Iff (AnalyticAt Complex f c) (Filter.Eventually (fun z => DifferentiableAt C …
  -/
  constructor
    /-
      case mp
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      c : Complex
      ⊢ AnalyticAt Complex f c → Filter.Eventually (fun z => DifferentiableAt Comple …
    -/
  · intro fa
    /-
      case mp
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      c : Complex
      fa : AnalyticAt Complex f c
      ⊢ Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
    -/
    filter_upwards [fa.eventually_analyticAt]
    /-
      case h
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      c : Complex
      fa : AnalyticAt Complex f c
      ⊢ ∀ (a : Complex), AnalyticAt Complex f a → DifferentiableAt Complex f a
    -/
    apply AnalyticAt.differentiableAt
    /-
      🎉 no goals
    -/
    /-
      case mpr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      c : Complex
      ⊢ Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c) → Analyti …
    -/
  · intro d
    /-
      case mpr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      c : Complex
      d : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
      ⊢ AnalyticAt Complex f c
    -/
    rcases _root_.eventually_nhds_iff.mp d with ⟨s, d, o, m⟩
    have h : AnalyticOnNhd ℂ f s := by
      refine DifferentiableOn.analyticOnNhd ?_ o
      intro z m
      exact (d z m).differentiableWithinAt
    /-
      case mpr.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      inst✝ : CompleteSpace E
      f : Complex → E
      c : Complex
      d✝ : Filter.Eventually (fun z => DifferentiableAt Complex f z) (nhds c)
      s : Set Complex
      d : ∀ (y : Complex), Membership.mem s y → DifferentiableAt Complex f y
      o : IsOpen s
      m : Membership.mem s c
      h : AnalyticOnNhd Complex f s
      ⊢ AnalyticAt Complex f c
    -/
    exact h _ m
    /-
      🎉 no goals
    -/


