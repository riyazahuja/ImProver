instance instCommRingMvPolynomial : CommRing (MvPolynomial σ R) :=
  AddMonoidAlgebra.commRing


@[simp]
theorem C_sub : (C (a - a') : MvPolynomial σ R) = C a - C a' :=
  RingHom.map_sub _ _ _


@[simp]
theorem C_neg : (C (-a) : MvPolynomial σ R) = -C a :=
  RingHom.map_neg _ _


@[simp]
theorem coeff_neg (m : σ →₀ ℕ) (p : MvPolynomial σ R) : coeff m (-p) = -coeff m p :=
  Finsupp.neg_apply _ _


@[simp]
theorem coeff_sub (m : σ →₀ ℕ) (p q : MvPolynomial σ R) : coeff m (p - q) = coeff m p - coeff m q :=
  Finsupp.sub_apply _ _ _


@[simp]
theorem support_neg : (-p).support = p.support :=
  Finsupp.support_neg p


theorem support_sub [DecidableEq σ] (p q : MvPolynomial σ R) :
    (p - q).support ⊆ p.support ∪ q.support :=
  Finsupp.support_sub


@[simp]
theorem degrees_neg (p : MvPolynomial σ R) : (-p).degrees = p.degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommRing R
    p : MvPolynomial σ R
    ⊢ Eq (Neg.neg p).degrees p.degrees
  -/
  rw [degrees, support_neg]; rfl
                             /-
                               🎉 no goals
                             -/


theorem degrees_sub [DecidableEq σ] (p q : MvPolynomial σ R) :
    (p - q).degrees ≤ p.degrees ⊔ q.degrees := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommRing R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    ⊢ LE.le (HSub.hSub p q).degrees (Max.max p.degrees q.degrees)
  -/
  simpa only [sub_eq_add_neg] using le_trans (degrees_add p (-q)) (by rw [degrees_neg])
  /-
    🎉 no goals
  -/


@[simp]
theorem degreeOf_neg (i : σ) (p : MvPolynomial σ R) : degreeOf i (-p) = degreeOf i p := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommRing R
    i : σ
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.degreeOf i (Neg.neg p)) (MvPolynomial.degreeOf i p)
  -/
  rw [degreeOf, degreeOf, degrees_neg]
  /-
    🎉 no goals
  -/


theorem degreeOf_sub_le (i : σ) (p q : MvPolynomial σ R) :
    degreeOf i (p - q) ≤ max (degreeOf i p) (degreeOf i q) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommRing R
    i : σ
    p q : MvPolynomial σ R
    ⊢ LE.le (MvPolynomial.degreeOf i (HSub.hSub p q)) (Max.max (MvPolynomial.degre …
  -/
  simpa only [sub_eq_add_neg, degreeOf_neg] using degreeOf_add_le i p (-q)
  /-
    🎉 no goals
  -/


@[simp]
                                            /-
                                              R : Type u
                                              σ : Type u_1
                                              inst✝ : CommRing R
                                              p : MvPolynomial σ R
                                              ⊢ Eq (Neg.neg p).vars p.vars
                                            -/
theorem vars_neg : (-p).vars = p.vars := by simp [vars, degrees_neg]
                                            /-
                                              🎉 no goals
                                            -/


theorem vars_sub_subset [DecidableEq σ] : (p - q).vars ⊆ p.vars ∪ q.vars := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommRing R
    p q : MvPolynomial σ R
    inst✝ : DecidableEq σ
    ⊢ HasSubset.Subset (HSub.hSub p q).vars (Union.union p.vars q.vars)
  -/
                                             /-
                                               🎉 no goals
                                             -/
  convert vars_add_subset p (-q) using 2 <;> simp [sub_eq_add_neg]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem vars_sub_of_disjoint [DecidableEq σ] (hpq : Disjoint p.vars q.vars) :
    (p - q).vars = p.vars ∪ q.vars := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommRing R
    p q : MvPolynomial σ R
    inst✝ : DecidableEq σ
    hpq : Disjoint p.vars q.vars
    ⊢ Eq (HSub.hSub p q).vars (Union.union p.vars q.vars)
  -/
  rw [← vars_neg q] at hpq
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommRing R
    p q : MvPolynomial σ R
    inst✝ : DecidableEq σ
    hpq : Disjoint p.vars (Neg.neg q).vars
    ⊢ Eq (HSub.hSub p q).vars (Union.union p.vars q.vars)
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  convert vars_add_of_disjoint hpq using 2 <;> simp [sub_eq_add_neg]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem eval₂_sub : (p - q).eval₂ f g = p.eval₂ f g - q.eval₂ f g :=
  (eval₂Hom f g).map_sub _ _


theorem eval_sub (f : σ → R) : eval f (p - q) = eval f p - eval f q :=
  eval₂_sub _ _ _


@[simp]
theorem eval₂_neg : (-p).eval₂ f g = -p.eval₂ f g :=
  (eval₂Hom f g).map_neg _


theorem eval_neg (f : σ → R) : eval f (-p) = -eval f p :=
  eval₂_neg _ _ _


theorem hom_C (f : MvPolynomial σ ℤ →+* S) (n : ℤ) : f (C n) = (n : S) :=
  eq_intCast (f.comp C) n


/-- A ring homomorphism f : Z[X_1, X_2, ...] → R
is determined by the evaluations f(X_1), f(X_2), ... -/
@[simp]
theorem eval₂Hom_X {R : Type u} (c : ℤ →+* S) (f : MvPolynomial R ℤ →+* S) (x : MvPolynomial R ℤ) :
    eval₂ c (f ∘ X) x = f x := by
  apply MvPolynomial.induction_on x
    (fun n => by
      rw [hom_C f, eval₂_C]
      exact eq_intCast c n)
    (fun p q hp hq => by
      rw [eval₂_add, hp, hq]
      exact (f.map_add _ _).symm)
    (fun p n hp => by
      rw [eval₂_mul, eval₂_X, hp]
      exact (f.map_mul _ _).symm)


/-- Ring homomorphisms out of integer polynomials on a type `σ` are the same as
functions out of the type `σ`, -/
def homEquiv : (MvPolynomial σ ℤ →+* S) ≃ (σ → S) where
  toFun f := f ∘ X
  invFun f := eval₂Hom (Int.castRingHom S) f
  left_inv _ := RingHom.ext <| eval₂Hom_X _ _
                                    /-
                                      R : Type u
                                      S : Type v
                                      σ : Type u_1
                                      a a' a₁ a₂ : R
                                      e : Nat
                                      n m : σ
                                      s : Finsupp σ Nat
                                      inst✝¹ : CommRing R
                                      p q : MvPolynomial σ R
                                      inst✝ : CommRing S
                                      f✝ : RingHom R S
                                      g f : σ → S
                                      x : σ
                                      ⊢ Eq ((fun f => Function.comp (⇑f) MvPolynomial.X) ((fun f => MvPolynomial.eva …
                                    -/
  right_inv f := funext fun x => by simp only [coe_eval₂Hom, Function.comp_apply, eval₂_X]
                                    /-
                                      🎉 no goals
                                    -/


theorem degreeOf_sub_lt {x : σ} {f g : MvPolynomial σ R} {k : ℕ} (h : 0 < k)
    (hf : ∀ m : σ →₀ ℕ, m ∈ f.support → k ≤ m x → coeff m f = coeff m g)
    (hg : ∀ m : σ →₀ ℕ, m ∈ g.support → k ≤ m x → coeff m f = coeff m g) :
    degreeOf x (f - g) < k := by
  classical
  rw [degreeOf_lt_iff h]
  intro m hm
  by_contra! hc
  have h := support_sub σ f g hm
  simp only [mem_support_iff, Ne, coeff_sub, sub_eq_zero] at hm
  cases' Finset.mem_union.1 h with cf cg
  · exact hm (hf m cf hc)
  · exact hm (hg m cg hc)


@[simp]
theorem totalDegree_neg (a : MvPolynomial σ R) : (-a).totalDegree = a.totalDegree := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommRing R
    a : MvPolynomial σ R
    ⊢ Eq (Neg.neg a).totalDegree a.totalDegree
  -/
  simp only [totalDegree, support_neg]
  /-
    🎉 no goals
  -/


theorem totalDegree_sub (a b : MvPolynomial σ R) :
    (a - b).totalDegree ≤ max a.totalDegree b.totalDegree :=
  calc
                                                     /-
                                                       R : Type u
                                                       σ : Type u_1
                                                       inst✝ : CommRing R
                                                       a b : MvPolynomial σ R
                                                       ⊢ Eq (HSub.hSub a b).totalDegree (HAdd.hAdd a (Neg.neg b)).totalDegree
                                                     -/
    (a - b).totalDegree = (a + -b).totalDegree := by rw [sub_eq_add_neg]
                                                     /-
                                                       🎉 no goals
                                                     -/
    _ ≤ max a.totalDegree (-b).totalDegree := totalDegree_add a (-b)
                                              /-
                                                R : Type u
                                                σ : Type u_1
                                                inst✝ : CommRing R
                                                a b : MvPolynomial σ R
                                                ⊢ Eq (Max.max a.totalDegree (Neg.neg b).totalDegree) (Max.max a.totalDegree b. …
                                              -/
    _ = max a.totalDegree b.totalDegree := by rw [totalDegree_neg]
                                              /-
                                                🎉 no goals
                                              -/


theorem totalDegree_sub_C_le (p : MvPolynomial σ R) (r : R) :
    totalDegree (p - C r) ≤ totalDegree p :=
                                       /-
                                         R : Type u
                                         σ : Type u_1
                                         inst✝ : CommRing R
                                         p : MvPolynomial σ R
                                         r : R
                                         ⊢ Eq (Max.max p.totalDegree (MvPolynomial.C r).totalDegree) p.totalDegree
                                       -/
  (totalDegree_sub _ _).trans_eq <| by rw [totalDegree_C, Nat.max_zero]
                                       /-
                                         🎉 no goals
                                       -/


