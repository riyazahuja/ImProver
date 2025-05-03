lemma NormedSpace.exp_continuousMap_eq (f : C(α, 𝕜)) :
    exp 𝕜 f = (⟨exp 𝕜 ∘ f, exp_continuous.comp f.continuous⟩ : C(α, 𝕜)) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    f : ContinuousMap α 𝕜
    ⊢ Eq (NormedSpace.exp 𝕜 f) { toFun := Function.comp (NormedSpace.exp 𝕜) ⇑f, co …
  -/
  ext a
  /-
    case h
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    f : ContinuousMap α 𝕜
    a : α
    ⊢ Eq ((NormedSpace.exp 𝕜 f) a) ({ toFun := Function.comp (NormedSpace.exp 𝕜) ⇑ …
  -/
  simp only [Function.comp_apply, NormedSpace.exp, FormalMultilinearSeries.sum]
  /-
    case h
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    f : ContinuousMap α 𝕜
    a : α
    ⊢ Eq ((tsum fun n => (NormedSpace.expSeries 𝕜 (ContinuousMap α 𝕜) n) fun x =>  …
  -/
  have h_sum := NormedSpace.expSeries_summable (𝕂 := 𝕜) f
  /-
    case h
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    f : ContinuousMap α 𝕜
    a : α
    h_sum : Summable fun n => (NormedSpace.expSeries 𝕜 (ContinuousMap α 𝕜) n) fun  …
    ⊢ Eq ((tsum fun n => (NormedSpace.expSeries 𝕜 (ContinuousMap α 𝕜) n) fun x =>  …
  -/
  simp_rw [← ContinuousMap.tsum_apply h_sum a, NormedSpace.expSeries_apply_eq]
  /-
    case h
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    f : ContinuousMap α 𝕜
    a : α
    h_sum : Summable fun n => (NormedSpace.expSeries 𝕜 (ContinuousMap α 𝕜) n) fun  …
    ⊢ Eq (tsum fun i => (HSMul.hSMul (Inv.inv ↑i.factorial) (HPow.hPow f i)) a) ({ …
  -/
  simp [NormedSpace.exp_eq_tsum]
  /-
    🎉 no goals
  -/


lemma exp_eq_normedSpace_exp {a : A} (ha : p a := by cfc_tac) :
    cfc (exp 𝕜 : 𝕜 → 𝕜) a = exp 𝕜 a := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (NormedSpace.exp 𝕜) a) (NormedSpace.exp 𝕜 a)
  -/
  conv_rhs => rw [← cfc_id 𝕜 a ha, cfc_apply id a ha]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    ⊢ Eq (cfc (NormedSpace.exp 𝕜) a) (NormedSpace.exp 𝕜 ((cfcHom ha) { toFun := (s …
  -/
  have h := (cfcHom_isClosedEmbedding (R := 𝕜) (show p a from ha)).continuous
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    h : Continuous ⇑(cfcHom ⋯)
    ⊢ Eq (cfc (NormedSpace.exp 𝕜) a) (NormedSpace.exp 𝕜 ((cfcHom ha) { toFun := (s …
  -/
  have _ : ContinuousOn (exp 𝕜) (spectrum 𝕜 a) := exp_continuous.continuousOn
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    h : Continuous ⇑(cfcHom ⋯)
    x✝ : ContinuousOn (NormedSpace.exp 𝕜) (spectrum 𝕜 a)
    ⊢ Eq (cfc (NormedSpace.exp 𝕜) a) (NormedSpace.exp 𝕜 ((cfcHom ha) { toFun := (s …
  -/
  simp_rw [← map_exp 𝕜 _ h, cfc_apply (exp 𝕜) a ha]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    h : Continuous ⇑(cfcHom ⋯)
    x✝ : ContinuousOn (NormedSpace.exp 𝕜) (spectrum 𝕜 a)
    ⊢ Eq ((cfcHom ha) { toFun := (spectrum 𝕜 a).restrict (NormedSpace.exp 𝕜), cont …
  -/
  congr 1
  /-
    case h.e_6.h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    h : Continuous ⇑(cfcHom ⋯)
    x✝ : ContinuousOn (NormedSpace.exp 𝕜) (spectrum 𝕜 a)
    ⊢ Eq { toFun := (spectrum 𝕜 a).restrict (NormedSpace.exp 𝕜), continuous_toFun  …
  -/
  ext
  /-
    case h.e_6.h.h
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁶ : RCLike 𝕜
    p : A → Prop
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ContinuousFunctionalCalculus 𝕜 p
    a : A
    ha : autoParam (p a) _auto✝
    h : Continuous ⇑(cfcHom ⋯)
    x✝ : ContinuousOn (NormedSpace.exp 𝕜) (spectrum 𝕜 a)
    a✝ : ↑(spectrum 𝕜 a)
    ⊢ Eq ({ toFun := (spectrum 𝕜 a).restrict (NormedSpace.exp 𝕜), continuous_toFun …
  -/
  simp [exp_continuousMap_eq]
  /-
    🎉 no goals
  -/


lemma real_exp_eq_normedSpace_exp {a : A} (ha : IsSelfAdjoint a := by cfc_tac) :
    cfc Real.exp a = exp ℝ a :=
  Real.exp_eq_exp_ℝ ▸ exp_eq_normedSpace_exp ha


@[aesop safe apply (rule_sets := [CStarAlgebra])]
lemma _root_.IsSelfAdjoint.exp_nonneg {𝕜 : Type*} [Field 𝕜] [Algebra 𝕜 A]
    [PartialOrder A] [StarOrderedRing A] {a : A} (ha : IsSelfAdjoint a) :
    0 ≤ exp 𝕜 a := by
  /-
    A : Type u_1
    inst✝⁹ : NormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    𝕜 : Type u_2
    inst✝³ : Field 𝕜
    inst✝² : Algebra 𝕜 A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a : A
    ha : IsSelfAdjoint a
    ⊢ LE.le 0 (NormedSpace.exp 𝕜 a)
  -/
  rw [exp_eq_exp 𝕜 ℝ, ← real_exp_eq_normedSpace_exp]
  /-
    A : Type u_1
    inst✝⁹ : NormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : TopologicalRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    𝕜 : Type u_2
    inst✝³ : Field 𝕜
    inst✝² : Algebra 𝕜 A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a : A
    ha : IsSelfAdjoint a
    ⊢ LE.le 0 (cfc Real.exp a)
  -/
  exact cfc_nonneg fun x _ => Real.exp_nonneg x
  /-
    🎉 no goals
  -/


lemma complex_exp_eq_normedSpace_exp {a : A} (ha : p a := by cfc_tac) :
    cfc Complex.exp a = exp ℂ a :=
  Complex.exp_eq_exp_ℂ ▸ exp_eq_normedSpace_exp ha


/-- The real logarithm, defined via the continuous functional calculus. This can be used on
matrices, operators on a Hilbert space, elements of a C⋆-algebra, etc. -/
noncomputable def log (a : A) : A := cfc Real.log a


@[simp]
protected lemma _root_.IsSelfAdjoint.log {a : A} : IsSelfAdjoint (log a) := cfc_predicate _ a



                                               /-
                                                 A : Type u_1
                                                 inst✝³ : NormedRing A
                                                 inst✝² : StarRing A
                                                 inst✝¹ : NormedAlgebra Real A
                                                 inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
                                                 ⊢ Eq (CFC.log 0) 0
                                               -/
@[simp] lemma log_zero : log (0 : A) = 0 := by simp [log]
                                               /-
                                                 🎉 no goals
                                               -/


                                              /-
                                                A : Type u_1
                                                inst✝³ : NormedRing A
                                                inst✝² : StarRing A
                                                inst✝¹ : NormedAlgebra Real A
                                                inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
                                                ⊢ Eq (CFC.log 1) 0
                                              -/
@[simp] lemma log_one : log (1 : A) = 0 := by simp [log]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma log_algebraMap {r : ℝ} : log (algebraMap ℝ A r) = algebraMap ℝ A (Real.log r) := by
  /-
    A : Type u_1
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    r : Real
    ⊢ Eq (CFC.log ((algebraMap Real A) r)) ((algebraMap Real A) (Real.log r))
  -/
  simp [log]
  /-
    🎉 no goals
  -/


lemma log_smul {r : ℝ} (a : A) (ha₂ : ∀ x ∈ spectrum ℝ a, 0 < x) (hr : 0 < r)
    (ha₁ : IsSelfAdjoint a := by cfc_tac) :
    log (r • a) = algebraMap ℝ A (Real.log r) + log a := by
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    r : Real
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    hr : LT.lt 0 r
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (CFC.log (HSMul.hSMul r a)) (HAdd.hAdd ((algebraMap Real A) (Real.log r)) …
  -/
  have : ∀ x ∈ spectrum ℝ a, x ≠ 0 := by peel ha₂ with x hx h; exact h.ne'
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    r : Real
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    hr : LT.lt 0 r
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    this : ∀ (x : Real), Membership.mem (spectrum Real a) x → Ne x 0
    ⊢ Eq (CFC.log (HSMul.hSMul r a)) (HAdd.hAdd ((algebraMap Real A) (Real.log r)) …
  -/
  rw [log, ← cfc_smul_id (R := ℝ) r a, ← cfc_comp Real.log (r • ·) a, log]
  calc
    _ = cfc (fun z => Real.log r + Real.log z) a :=
      cfc_congr (Real.log_mul hr.ne' <| ne_of_gt <| ha₂ · ·)
    _ = _ := by rw [cfc_const_add _ _ _]

-- TODO: Relate the hypothesis to a notion of strict positivity

lemma log_pow (n : ℕ) (a : A) (ha₂ : ∀ x ∈ spectrum ℝ a, 0 < x)
    (ha₁ : IsSelfAdjoint a := by cfc_tac) : log (a ^ n) = n • log a := by
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    n : Nat
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (CFC.log (HPow.hPow a n)) (HSMul.hSMul n (CFC.log a))
  -/
  have : ∀ x ∈ spectrum ℝ a, x ≠ 0 := by peel ha₂ with x hx h; exact h.ne'
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    n : Nat
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    this : ∀ (x : Real), Membership.mem (spectrum Real a) x → Ne x 0
    ⊢ Eq (CFC.log (HPow.hPow a n)) (HSMul.hSMul n (CFC.log a))
  -/
  have ha₂' : ContinuousOn Real.log (spectrum ℝ a) := by fun_prop (disch := assumption)
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    n : Nat
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    this : ∀ (x : Real), Membership.mem (spectrum Real a) x → Ne x 0
    ha₂' : ContinuousOn Real.log (spectrum Real a)
    ⊢ Eq (CFC.log (HPow.hPow a n)) (HSMul.hSMul n (CFC.log a))
  -/
  have ha₂'' : ContinuousOn Real.log ((· ^ n) '' spectrum ℝ a)  := by fun_prop (disch := aesop)
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    n : Nat
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    this : ∀ (x : Real), Membership.mem (spectrum Real a) x → Ne x 0
    ha₂' : ContinuousOn Real.log (spectrum Real a)
    ha₂'' : ContinuousOn Real.log (Set.image (fun x => HPow.hPow x n) (spectrum Re …
    ⊢ Eq (CFC.log (HPow.hPow a n)) (HSMul.hSMul n (CFC.log a))
  -/
  rw [log, ← cfc_pow_id (R := ℝ) a n ha₁, ← cfc_comp' Real.log (· ^ n) a ha₂'', log]
  /-
    A : Type u_1
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra Real A
    inst✝¹ : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    n : Nat
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    this : ∀ (x : Real), Membership.mem (spectrum Real a) x → Ne x 0
    ha₂' : ContinuousOn Real.log (spectrum Real a)
    ha₂'' : ContinuousOn Real.log (Set.image (fun x => HPow.hPow x n) (spectrum Re …
    ⊢ Eq (cfc (fun x => Real.log (HPow.hPow x n)) a) (HSMul.hSMul n (cfc Real.log  …
  -/
  simp_rw [Real.log_pow, ← Nat.cast_smul_eq_nsmul ℝ n, cfc_const_mul (n : ℝ) Real.log a ha₂']
  /-
    🎉 no goals
  -/


lemma log_exp (a : A) (ha : IsSelfAdjoint a := by cfc_tac) : log (NormedSpace.exp ℝ a) = a := by
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra Real A
    inst✝² : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : CompleteSpace A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    ⊢ Eq (CFC.log (NormedSpace.exp Real a)) a
  -/
  have hcont : ContinuousOn Real.log (Real.exp '' spectrum ℝ a) := by fun_prop (disch := aesop)
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra Real A
    inst✝² : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : CompleteSpace A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    hcont : ContinuousOn Real.log (Set.image Real.exp (spectrum Real a))
    ⊢ Eq (CFC.log (NormedSpace.exp Real a)) a
  -/
  rw [log, ← real_exp_eq_normedSpace_exp, ← cfc_comp' Real.log Real.exp a hcont]
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra Real A
    inst✝² : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : CompleteSpace A
    a : A
    ha : autoParam (IsSelfAdjoint a) _auto✝
    hcont : ContinuousOn Real.log (Set.image Real.exp (spectrum Real a))
    ⊢ Eq (cfc (fun x => Real.log (Real.exp x)) a) a
  -/
  simp [cfc_id' (R := ℝ) a]
  /-
    🎉 no goals
  -/

-- TODO: Relate the hypothesis to a notion of strict positivity

lemma exp_log (a : A) (ha₂ : ∀ x ∈ spectrum ℝ a, 0 < x) (ha₁ : IsSelfAdjoint a := by cfc_tac) :
    NormedSpace.exp ℝ (log a) = a := by
  have ha₃ : ContinuousOn Real.log (spectrum ℝ a) := by
    have : ∀ x ∈ spectrum ℝ a, x ≠ 0 := by peel ha₂ with x hx h; exact h.ne'
    fun_prop (disch := assumption)
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra Real A
    inst✝² : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : CompleteSpace A
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    ha₃ : ContinuousOn Real.log (spectrum Real a)
    ⊢ Eq (NormedSpace.exp Real (CFC.log a)) a
  -/
  rw [← real_exp_eq_normedSpace_exp .log, log, ← cfc_comp' Real.exp Real.log a (by fun_prop) ha₃]
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra Real A
    inst✝² : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : CompleteSpace A
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    ha₃ : ContinuousOn Real.log (spectrum Real a)
    ⊢ Eq (cfc (fun x => Real.exp (Real.log x)) a) a
  -/
  conv_rhs => rw [← cfc_id (R := ℝ) a ha₁]
  /-
    A : Type u_1
    inst✝⁵ : NormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedAlgebra Real A
    inst✝² : ContinuousFunctionalCalculus Real IsSelfAdjoint
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : CompleteSpace A
    a : A
    ha₂ : ∀ (x : Real), Membership.mem (spectrum Real a) x → LT.lt 0 x
    ha₁ : autoParam (IsSelfAdjoint a) _auto✝
    ha₃ : ContinuousOn Real.log (spectrum Real a)
    ⊢ Eq (cfc (fun x => Real.exp (Real.log x)) a) (cfc id a)
  -/
  exact cfc_congr (Real.exp_log <| ha₂ · ·)
  /-
    🎉 no goals
  -/


