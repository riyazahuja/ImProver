theorem AnalyticWithinAt.aeval_polynomial (hf : AnalyticWithinAt 𝕜 f s z) (p : A[X]) :
    AnalyticWithinAt 𝕜 (fun x ↦ aeval (f x) p) s z := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CommSemiring A
    z : E
    s : Set E
    inst✝² : NormedRing B
    inst✝¹ : NormedAlgebra 𝕜 B
    inst✝ : Algebra A B
    f : E → B
    hf : AnalyticWithinAt 𝕜 f s z
    p : Polynomial A
    ⊢ AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) p) s z
  -/
  refine p.induction_on (fun k ↦ ?_) (fun p q hp hq ↦ ?_) fun p i hp ↦ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      s : Set E
      inst✝² : NormedRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      f : E → B
      hf : AnalyticWithinAt 𝕜 f s z
      p : Polynomial A
      k : A
      ⊢ AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) (Polynomial.C k)) s z
    -/
  · simp_rw [aeval_C]; apply analyticWithinAt_const
                       /-
                         🎉 no goals
                       -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      s : Set E
      inst✝² : NormedRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      f : E → B
      hf : AnalyticWithinAt 𝕜 f s z
      p✝ p q : Polynomial A
      hp : AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) p) s z
      hq : AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) q) s z
      ⊢ AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) (HAdd.hAdd p q)) s z
    -/
  · simp_rw [aeval_add]; exact hp.add hq
                         /-
                           🎉 no goals
                         -/
    /-
      case refine_3
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      s : Set E
      inst✝² : NormedRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      f : E → B
      hf : AnalyticWithinAt 𝕜 f s z
      p✝ : Polynomial A
      p : Nat
      i : A
      hp : AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) (HMul.hMul (Polynom …
      ⊢ AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) (HMul.hMul (Polynomial …
    -/
  · convert hp.mul hf
    /-
      case h.e'_9.h
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      s : Set E
      inst✝² : NormedRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      f : E → B
      hf : AnalyticWithinAt 𝕜 f s z
      p✝ : Polynomial A
      p : Nat
      i : A
      hp : AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) (HMul.hMul (Polynom …
      x✝ : E
      ⊢ Eq ((Polynomial.aeval (f x✝)) (HMul.hMul (Polynomial.C i) (HPow.hPow Polynom …
    -/
    simp_rw [pow_succ, aeval_mul, ← mul_assoc, aeval_X]
    /-
      🎉 no goals
    -/


theorem AnalyticAt.aeval_polynomial (hf : AnalyticAt 𝕜 f z) (p : A[X]) :
    AnalyticAt 𝕜 (fun x ↦ aeval (f x) p) z := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CommSemiring A
    z : E
    inst✝² : NormedRing B
    inst✝¹ : NormedAlgebra 𝕜 B
    inst✝ : Algebra A B
    f : E → B
    hf : AnalyticAt 𝕜 f z
    p : Polynomial A
    ⊢ AnalyticAt 𝕜 (fun x => (Polynomial.aeval (f x)) p) z
  -/
  rw [← analyticWithinAt_univ] at hf ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CommSemiring A
    z : E
    inst✝² : NormedRing B
    inst✝¹ : NormedAlgebra 𝕜 B
    inst✝ : Algebra A B
    f : E → B
    hf : AnalyticWithinAt 𝕜 f Set.univ z
    p : Polynomial A
    ⊢ AnalyticWithinAt 𝕜 (fun x => (Polynomial.aeval (f x)) p) Set.univ z
  -/
  exact hf.aeval_polynomial p
  /-
    🎉 no goals
  -/


theorem AnalyticOnNhd.aeval_polynomial (hf : AnalyticOnNhd 𝕜 f s) (p : A[X]) :
    AnalyticOnNhd 𝕜 (fun x ↦ aeval (f x) p) s := fun x hx ↦ (hf x hx).aeval_polynomial p


theorem AnalyticOn.aeval_polynomial (hf : AnalyticOn 𝕜 f s) (p : A[X]) :
    AnalyticOn 𝕜 (fun x ↦ aeval (f x) p) s := fun x hx ↦ (hf x hx).aeval_polynomial p


theorem AnalyticOnNhd.eval_polynomial {A} [NormedCommRing A] [NormedAlgebra 𝕜 A] (p : A[X]) :
    AnalyticOnNhd 𝕜 (eval · p) Set.univ := analyticOnNhd_id.aeval_polynomial p


theorem AnalyticOn.eval_polynomial {A} [NormedCommRing A] [NormedAlgebra 𝕜 A] (p : A[X]) :
    AnalyticOn 𝕜 (eval · p) Set.univ := analyticOn_id.aeval_polynomial p


theorem AnalyticAt.aeval_mvPolynomial (hf : ∀ i, AnalyticAt 𝕜 (f · i) z) (p : MvPolynomial σ A) :
    AnalyticAt 𝕜 (fun x ↦ aeval (f x) p) z := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CommSemiring A
    z : E
    inst✝² : NormedCommRing B
    inst✝¹ : NormedAlgebra 𝕜 B
    inst✝ : Algebra A B
    σ : Type u_5
    f : E → σ → B
    hf : ∀ (i : σ), AnalyticAt 𝕜 (fun x => f x i) z
    p : MvPolynomial σ A
    ⊢ AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) p) z
  -/
  apply p.induction_on (fun k ↦ ?_) (fun p q hp hq ↦ ?_) fun p i hp ↦ ?_ -- `refine` doesn't work
    /-
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      inst✝² : NormedCommRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      σ : Type u_5
      f : E → σ → B
      hf : ∀ (i : σ), AnalyticAt 𝕜 (fun x => f x i) z
      p : MvPolynomial σ A
      k : A
      ⊢ AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) (MvPolynomial.C k)) z
    -/
  · simp_rw [aeval_C]; apply analyticAt_const
                       /-
                         🎉 no goals
                       -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      inst✝² : NormedCommRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      σ : Type u_5
      f : E → σ → B
      hf : ∀ (i : σ), AnalyticAt 𝕜 (fun x => f x i) z
      p✝ p q : MvPolynomial σ A
      hp : AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) p) z
      hq : AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) q) z
      ⊢ AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) (HAdd.hAdd p q)) z
    -/
  · simp_rw [map_add]; exact hp.add hq
                       /-
                         🎉 no goals
                       -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CommSemiring A
      z : E
      inst✝² : NormedCommRing B
      inst✝¹ : NormedAlgebra 𝕜 B
      inst✝ : Algebra A B
      σ : Type u_5
      f : E → σ → B
      hf : ∀ (i : σ), AnalyticAt 𝕜 (fun x => f x i) z
      p✝ p : MvPolynomial σ A
      i : σ
      hp : AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) p) z
      ⊢ AnalyticAt 𝕜 (fun x => (MvPolynomial.aeval (f x)) (HMul.hMul p (MvPolynomial …
    -/
  · simp_rw [map_mul, aeval_X]; exact hp.mul (hf i)
                                /-
                                  🎉 no goals
                                -/


theorem AnalyticOnNhd.aeval_mvPolynomial
    (hf : ∀ i, AnalyticOnNhd 𝕜 (f · i) s) (p : MvPolynomial σ A) :
    AnalyticOnNhd 𝕜 (fun x ↦ aeval (f x) p) s := fun x hx ↦ .aeval_mvPolynomial (hf · x hx) p


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.aeval_mvPolynomial := AnalyticOnNhd.aeval_mvPolynomial


theorem AnalyticOnNhd.eval_continuousLinearMap (f : E →L[𝕜] σ → B) (p : MvPolynomial σ B) :
    AnalyticOnNhd 𝕜 (fun x ↦ eval (f x) p) Set.univ :=
  fun x _ ↦ .aeval_mvPolynomial (fun i ↦ ((ContinuousLinearMap.proj i).comp f).analyticAt x) p


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.eval_continuousLinearMap := AnalyticOnNhd.eval_continuousLinearMap


theorem AnalyticOnNhd.eval_continuousLinearMap' (f : σ → E →L[𝕜] B) (p : MvPolynomial σ B) :
    AnalyticOnNhd 𝕜 (fun x ↦ eval (f · x) p) Set.univ :=
  fun x _ ↦ .aeval_mvPolynomial (fun i ↦ (f i).analyticAt x) p


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.eval_continuousLinearMap' := AnalyticOnNhd.eval_continuousLinearMap'


theorem AnalyticOnNhd.eval_linearMap (f : E →ₗ[𝕜] σ → B) (p : MvPolynomial σ B) :
    AnalyticOnNhd 𝕜 (fun x ↦ eval (f x) p) Set.univ :=
  AnalyticOnNhd.eval_continuousLinearMap { f with cont := f.continuous_of_finiteDimensional } p


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.eval_linearMap := AnalyticOnNhd.eval_linearMap


theorem AnalyticOnNhd.eval_linearMap' (f : σ → E →ₗ[𝕜] B) (p : MvPolynomial σ B) :
    AnalyticOnNhd 𝕜 (fun x ↦ eval (f · x) p) Set.univ := AnalyticOnNhd.eval_linearMap (.pi f) p


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.eval_linearMap' := AnalyticOnNhd.eval_linearMap'


theorem AnalyticOnNhd.eval_mvPolynomial [Fintype σ] (p : MvPolynomial σ 𝕜) :
    AnalyticOnNhd 𝕜 (eval · p) Set.univ :=
  AnalyticOnNhd.eval_linearMap (.id (R := 𝕜) (M := σ → 𝕜)) p


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.eval_mvPolynomial := AnalyticOnNhd.eval_mvPolynomial


