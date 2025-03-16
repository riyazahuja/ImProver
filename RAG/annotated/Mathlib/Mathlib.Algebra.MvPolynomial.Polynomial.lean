theorem polynomial_eval_eval₂ [CommSemiring R] [CommSemiring S]
    {x : S} (f : R →+* Polynomial S) (g : σ → Polynomial S) (p : MvPolynomial σ R) :
    Polynomial.eval x (eval₂ f g p) =
      eval₂ ((Polynomial.evalRingHom x).comp f) (fun s => Polynomial.eval x (g s)) p := by
  /-
    R : Type u_1
    S : Type u_2
    σ : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    x : S
    f : RingHom R (Polynomial S)
    g : σ → Polynomial S
    p : MvPolynomial σ R
    ⊢ Eq (Polynomial.eval x (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ ((Poly …
  -/
  apply induction_on p
    /-
      case h_C
      R : Type u_1
      S : Type u_2
      σ : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      x : S
      f : RingHom R (Polynomial S)
      g : σ → Polynomial S
      p : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq (Polynomial.eval x (MvPolynomial.eval₂ f g (MvPolynomial.C a)) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u_1
      S : Type u_2
      σ : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      x : S
      f : RingHom R (Polynomial S)
      g : σ → Polynomial S
      p : MvPolynomial σ R
      ⊢ ∀ (p q : MvPolynomial σ R), Eq (Polynomial.eval x (MvPolynomial.eval₂ f g p) …
    -/
  · intro p q hp hq
    /-
      case h_add
      R : Type u_1
      S : Type u_2
      σ : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      x : S
      f : RingHom R (Polynomial S)
      g : σ → Polynomial S
      p✝ p q : MvPolynomial σ R
      hp : Eq (Polynomial.eval x (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ ((P …
      hq : Eq (Polynomial.eval x (MvPolynomial.eval₂ f g q)) (MvPolynomial.eval₂ ((P …
      ⊢ Eq (Polynomial.eval x (MvPolynomial.eval₂ f g (HAdd.hAdd p q))) (MvPolynomia …
    -/
    simp [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u_1
      S : Type u_2
      σ : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      x : S
      f : RingHom R (Polynomial S)
      g : σ → Polynomial S
      p : MvPolynomial σ R
      ⊢ ∀ (p : MvPolynomial σ R) (n : σ), Eq (Polynomial.eval x (MvPolynomial.eval₂  …
    -/
  · intro p n hp
    /-
      case h_X
      R : Type u_1
      S : Type u_2
      σ : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      x : S
      f : RingHom R (Polynomial S)
      g : σ → Polynomial S
      p✝ p : MvPolynomial σ R
      n : σ
      hp : Eq (Polynomial.eval x (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ ((P …
      ⊢ Eq (Polynomial.eval x (MvPolynomial.eval₂ f g (HMul.hMul p (MvPolynomial.X n …
    -/
    simp [hp]
    /-
      🎉 no goals
    -/


theorem eval_polynomial_eval_finSuccEquiv {n : ℕ} {x : Fin n → R}
    [CommSemiring R] (f : MvPolynomial (Fin (n + 1)) R) (q : MvPolynomial (Fin n) R) :
    (eval x) (Polynomial.eval q (finSuccEquiv R n f)) = eval (Fin.cases (eval x q) x) f := by
  /-
    R : Type u_1
    n : Nat
    x : Fin n → R
    inst✝ : CommSemiring R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    q : MvPolynomial (Fin n) R
    ⊢ Eq ((MvPolynomial.eval x) (Polynomial.eval q ((MvPolynomial.finSuccEquiv R n …
  -/
  simp only [finSuccEquiv_apply, coe_eval₂Hom, polynomial_eval_eval₂, eval_eval₂]
  conv in RingHom.comp _ _ =>
    refine @RingHom.ext _ _ _ _ _ (RingHom.id _) fun r => ?_
    simp
  /-
    R : Type u_1
    n : Nat
    x : Fin n → R
    inst✝ : CommSemiring R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    q : MvPolynomial (Fin n) R
    ⊢ Eq (MvPolynomial.eval₂ (RingHom.id R) (fun s => (MvPolynomial.eval x) (Polyn …
  -/
  simp only [eval₂_id]
  /-
    R : Type u_1
    n : Nat
    x : Fin n → R
    inst✝ : CommSemiring R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    q : MvPolynomial (Fin n) R
    ⊢ Eq ((MvPolynomial.eval fun s => (MvPolynomial.eval x) (Polynomial.eval q (Fi …
  -/
  congr
  /-
    case e_a.e_f
    R : Type u_1
    n : Nat
    x : Fin n → R
    inst✝ : CommSemiring R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    q : MvPolynomial (Fin n) R
    ⊢ Eq (fun s => (MvPolynomial.eval x) (Polynomial.eval q (Fin.cases Polynomial. …
  -/
  funext i
  /-
    case e_a.e_f.h
    R : Type u_1
    n : Nat
    x : Fin n → R
    inst✝ : CommSemiring R
    f : MvPolynomial (Fin (HAdd.hAdd n 1)) R
    q : MvPolynomial (Fin n) R
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq ((MvPolynomial.eval x) (Polynomial.eval q (Fin.cases Polynomial.X (fun k  …
  -/
  refine Fin.cases (by simp) (by simp) i
  /-
    🎉 no goals
  -/


