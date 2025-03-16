/-- The notation `Y` for `X` in the `Polynomial` scope. -/
scoped[Polynomial.Bivariate] notation3:max "Y" => Polynomial.X (R := Polynomial _)


/-- The notation `R[X][Y]` for `R[X][X]` in the `Polynomial` scope. -/
scoped[Polynomial.Bivariate] notation3:max R "[X][Y]" => Polynomial (Polynomial R)


/-- `evalEval x y p` is the evaluation `p(x,y)` of a two-variable polynomial `p : R[X][Y]`. -/
abbrev evalEval (x y : R) (p : R[X][Y]) : R := eval x (eval (C y) p)


/-- A constant viewed as a polynomial in two variables. -/
abbrev CC (r : R) : R[X][Y] := C (C r)


lemma evalEval_C (x y : R) (p : R[X]) : (C p).evalEval x y = p.eval x := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    p : Polynomial R
    ⊢ Eq (Polynomial.evalEval x y (Polynomial.C p)) (Polynomial.eval x p)
  -/
  rw [evalEval, eval_C]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_CC (x y : R) (p : R) : (CC p).evalEval x y = p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y p : R
    ⊢ Eq (Polynomial.evalEval x y (Polynomial.CC p)) p
  -/
  rw [evalEval_C, eval_C]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_zero (x y : R) : (0 : R[X][Y]).evalEval x y = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y 0) 0
  -/
  simp only [evalEval, eval_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_one (x y : R) : (1 : R[X][Y]).evalEval x y = 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y 1) 1
  -/
  simp only [evalEval, eval_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_natCast (x y : R) (n : ℕ) : (n : R[X][Y]).evalEval x y = n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    n : Nat
    ⊢ Eq (Polynomial.evalEval x y ↑n) ↑n
  -/
  simp only [evalEval, eval_natCast]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_X (x y : R) : X.evalEval x y = y := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    ⊢ Eq (Polynomial.evalEval x y Polynomial.X) y
  -/
  rw [evalEval, eval_X, eval_C]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_add (x y : R) (p q : R[X][Y]) :
    (p + q).evalEval x y = p.evalEval x y + q.evalEval x y := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    p q : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (HAdd.hAdd p q)) (HAdd.hAdd (Polynomial.evalEval …
  -/
  simp only [evalEval, eval_add]
  /-
    🎉 no goals
  -/


lemma evalEval_sum (x y : R) (p : R[X]) (f : ℕ → R → R[X][Y]) :
    (p.sum f).evalEval x y = p.sum fun n a => (f n a).evalEval x y := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x y : R
    p : Polynomial R
    f : Nat → R → Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (p.sum f)) (p.sum fun n a => Polynomial.evalEval …
  -/
  simp only [evalEval, eval, eval₂_sum]
  /-
    🎉 no goals
  -/


lemma evalEval_finset_sum {ι : Type*} (s : Finset ι) (x y : R) (f : ι → R[X][Y]) :
    (∑ i ∈ s, f i).evalEval x y = ∑ i ∈ s, (f i).evalEval x y := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ι : Type u_3
    s : Finset ι
    x y : R
    f : ι → Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (s.sum fun i => f i)) (s.sum fun i => Polynomial …
  -/
  simp only [evalEval, eval_finset_sum]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_smul [Monoid S] [DistribMulAction S R] [IsScalarTower S R R] (x y : R) (s : S)
    (p : R[X][Y]) : (s • p).evalEval x y = s • p.evalEval x y := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : Semiring R
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S R
    inst✝ : IsScalarTower S R R
    x y : R
    s : S
    p : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (HSMul.hSMul s p)) (HSMul.hSMul s (Polynomial.ev …
  -/
  simp only [evalEval, eval_smul]
  /-
    🎉 no goals
  -/


lemma evalEval_surjective (x y : R) : Function.Surjective <| evalEval x y :=
  fun y => ⟨CC y, evalEval_CC ..⟩


@[simp]
lemma evalEval_neg (x y : R) (p : R[X][Y]) : (-p).evalEval x y = -p.evalEval x y := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    p : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (Neg.neg p)) (Neg.neg (Polynomial.evalEval x y p))
  -/
  simp only [evalEval, eval_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_sub (x y : R) (p q : R[X][Y]) :
    (p - q).evalEval x y = p.evalEval x y - q.evalEval x y := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    p q : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (HSub.hSub p q)) (HSub.hSub (Polynomial.evalEval …
  -/
  simp only [evalEval, eval_sub]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_intCast (x y : R) (n : ℤ) : (n : R[X][Y]).evalEval x y = n := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    n : Int
    ⊢ Eq (Polynomial.evalEval x y ↑n) ↑n
  -/
  simp only [evalEval, eval_intCast]
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_mul (x y : R) (p q : R[X][Y]) :
    (p * q).evalEval x y = p.evalEval x y * q.evalEval x y := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x y : R
    p q : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (HMul.hMul p q)) (HMul.hMul (Polynomial.evalEval …
  -/
  simp only [evalEval, eval_mul]
  /-
    🎉 no goals
  -/


lemma evalEval_prod {ι : Type*} (s : Finset ι) (x y : R) (p : ι → R[X][Y]) :
    (∏ j ∈ s, p j).evalEval x y = ∏ j ∈ s, (p j).evalEval x y := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ι : Type u_3
    s : Finset ι
    x y : R
    p : ι → Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.evalEval x y (s.prod fun j => p j)) (s.prod fun j => Polynomi …
  -/
  simp only [evalEval, eval_prod]
  /-
    🎉 no goals
  -/


lemma evalEval_list_prod (x y : R) (l : List R[X][Y]) :
    l.prod.evalEval x y = (l.map <| evalEval x y).prod := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x y : R
    l : List (Polynomial (Polynomial R))
    ⊢ Eq (Polynomial.evalEval x y l.prod) (List.map (Polynomial.evalEval x y) l).p …
  -/
  simpa only [evalEval, eval_list_prod, List.map_map] using by rfl
  /-
    🎉 no goals
  -/


lemma evalEval_multiset_prod (x y : R) (l : Multiset R[X][Y]) :
    l.prod.evalEval x y = (l.map <| evalEval x y).prod := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x y : R
    l : Multiset (Polynomial (Polynomial R))
    ⊢ Eq (Polynomial.evalEval x y l.prod) (Multiset.map (Polynomial.evalEval x y)  …
  -/
  simpa only [evalEval, eval_multiset_prod, Multiset.map_map] using by rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma evalEval_pow (x y : R) (p : R[X][Y]) (n : ℕ) : (p ^ n).evalEval x y = p.evalEval x y ^ n := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x y : R
    p : Polynomial (Polynomial R)
    n : Nat
    ⊢ Eq (Polynomial.evalEval x y (HPow.hPow p n)) (HPow.hPow (Polynomial.evalEval …
  -/
  simp only [evalEval, eval_pow]
  /-
    🎉 no goals
  -/


lemma evalEval_dvd (x y : R) {p q : R[X][Y]} : p ∣ q → p.evalEval x y ∣ q.evalEval x y :=
  eval_dvd ∘ eval_dvd


lemma coe_algebraMap_eq_CC : algebraMap R R[X][Y] = CC (R := R) := rfl


/-- `evalEval x y` as a ring homomorphism. -/
@[simps!] abbrev evalEvalRingHom (x y : R) : R[X][Y] →+* R :=
  (evalRingHom x).comp (evalRingHom <| C y)


lemma coe_evalEvalRingHom (x y : R) : evalEvalRingHom x y = evalEval x y := rfl


lemma evalEvalRingHom_eq (x : R) : evalEvalRingHom x = eval₂RingHom (evalRingHom x) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x : R
    ⊢ Eq (Polynomial.evalEvalRingHom x) (Polynomial.eval₂RingHom (Polynomial.evalR …
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


lemma eval₂_evalRingHom (x : R) : eval₂ (evalRingHom x) = evalEval x := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x : R
    ⊢ Eq (Polynomial.eval₂ (Polynomial.evalRingHom x)) (Polynomial.evalEval x)
  -/
  ext1; rw [← coe_evalEvalRingHom, evalEvalRingHom_eq, coe_eval₂RingHom]
        /-
          🎉 no goals
        -/


lemma map_evalRingHom_eval (x y : R) (p : R[X][Y]) :
    (p.map <| evalRingHom x).eval y = p.evalEval x y := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x y : R
    p : Polynomial (Polynomial R)
    ⊢ Eq (Polynomial.eval y (Polynomial.map (Polynomial.evalRingHom x) p)) (Polyno …
  -/
  rw [eval_map, eval₂_evalRingHom]
  /-
    🎉 no goals
  -/


lemma map_mapRingHom_eval_map : (p.map <| mapRingHom f).eval (q.map f) = (p.eval q).map f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial (Polynomial R)
    q : Polynomial R
    ⊢ Eq (Polynomial.eval (Polynomial.map f q) (Polynomial.map (Polynomial.mapRing …
  -/
  rw [eval_map, ← coe_mapRingHom, eval₂_hom]
  /-
    🎉 no goals
  -/


lemma map_mapRingHom_eval_map_eval (r : R) :
    ((p.map <| mapRingHom f).eval <| q.map f).eval (f r) = f ((p.eval q).eval r) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial (Polynomial R)
    q : Polynomial R
    r : R
    ⊢ Eq (Polynomial.eval (f r) (Polynomial.eval (Polynomial.map f q) (Polynomial. …
  -/
  rw [map_mapRingHom_eval_map, eval_map, eval₂_hom]
  /-
    🎉 no goals
  -/


lemma map_mapRingHom_evalEval (x y : R) :
    (p.map <| mapRingHom f).evalEval (f x) (f y) = f (p.evalEval x y) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial (Polynomial R)
    x y : R
    ⊢ Eq (Polynomial.evalEval (f x) (f y) (Polynomial.map (Polynomial.mapRingHom f …
  -/
  rw [evalEval, ← map_mapRingHom_eval_map_eval, map_C]
  /-
    🎉 no goals
  -/


/-- Two equivalent ways to express the evaluation of a bivariate polynomial over `R`
at a point in the affine plane over an `R`-algebra `S`. -/
lemma eval₂RingHom_eval₂RingHom (f : R →+* S) (x y : S) :
    eval₂RingHom (eval₂RingHom f x) y =
      (evalEvalRingHom x y).comp (mapRingHom <| mapRingHom f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    x y : S
    ⊢ Eq (Polynomial.eval₂RingHom (Polynomial.eval₂RingHom f x) y) ((Polynomial.ev …
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


lemma eval₂_eval₂RingHom_apply (f : R →+* S) (x y : S) (p : R[X][Y]) :
    eval₂ (eval₂RingHom f x) y p = (p.map <| mapRingHom f).evalEval x y :=
  congr($(eval₂RingHom_eval₂RingHom f x y) p)


lemma eval_C_X_comp_eval₂_map_C_X :
    (evalRingHom (C X : R[X][Y])).comp (eval₂RingHom (mapRingHom <| algebraMap R R[X][Y]) (C Y)) =
      .id _ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq ((Polynomial.evalRingHom (Polynomial.C Polynomial.X)).comp (Polynomial.ev …
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- Viewing `R[X,Y,X']` as an `R[X']`-algebra, a polynomial `p : R[X',Y']` can be evaluated at
`Y : R[X,Y,X']` (substitution of `Y'` by `Y`), obtaining another polynomial in `R[X,Y,X']`.
When this polynomial is then evaluated at `X' = X`, the original polynomial `p` is recovered. -/
lemma eval_C_X_eval₂_map_C_X {p : R[X][Y]} :
    eval (C X) (eval₂ (mapRingHom <| algebraMap R R[X][Y]) (C Y) p) = p :=
  congr($eval_C_X_comp_eval₂_map_C_X p)


/-- If the evaluation (`evalEval`) of a bivariate polynomial `p : R[X][Y]` at a point (x,y)
is zero, then `Polynomial.evalEval x y` factors through `AdjoinRoot.evalEval`, a ring homomorphism
from `AdjoinRoot p` to `R`. -/
@[simps!] def evalEval : AdjoinRoot p →+* R :=
  lift (evalRingHom x) y <| eval₂_evalRingHom x ▸ h


lemma evalEval_mk (g : R[X][Y]) : evalEval h (mk p g) = g.evalEval x y := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    x y : R
    p : Polynomial (Polynomial R)
    h : Eq (Polynomial.evalEval x y p) 0
    g : Polynomial (Polynomial R)
    ⊢ Eq ((AdjoinRoot.evalEval h) ((AdjoinRoot.mk p) g)) (Polynomial.evalEval x y g)
  -/
  rw [evalEval, lift_mk, eval₂_evalRingHom]
  /-
    🎉 no goals
  -/


