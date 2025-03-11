noncomputable def quotientSpanXSubCAlgEquivAux2 (x : R) :
    (R[X] ⧸ (RingHom.ker (aeval x).toRingHom : Ideal R[X])) ≃ₐ[R] R :=
  let e := RingHom.quotientKerEquivOfRightInverse (fun x => by
    /-
      R : Type u_1
      inst✝ : CommRing R
      x✝ x : R
      ⊢ Eq ((Polynomial.aeval x✝) ((fun a => Polynomial.C a) x)) x
    -/
    exact eval_C : Function.RightInverse (fun a : R => (C a : R[X])) (@aeval R R _ _ _ x))
    /-
      🎉 no goals
    -/
  { e with commutes' := fun r => e.apply_symm_apply r }


noncomputable def quotientSpanXSubCAlgEquivAux1 (x : R) :
    (R[X] ⧸ Ideal.span {X - C x}) ≃ₐ[R] (R[X] ⧸ (RingHom.ker (aeval x).toRingHom : Ideal R[X])) :=
  @Ideal.quotientEquivAlgOfEq R R[X] _ _ _ _ _ (ker_evalRingHom x).symm

-- Porting note: need to split this definition into two sub-definitions to prevent time out

/-- For a commutative ring $R$, evaluating a polynomial at an element $x \in R$ induces an
isomorphism of $R$-algebras $R[X] / \langle X - x \rangle \cong R$. -/
noncomputable def quotientSpanXSubCAlgEquiv (x : R) :
    (R[X] ⧸ Ideal.span ({X - C x} : Set R[X])) ≃ₐ[R] R :=
  (quotientSpanXSubCAlgEquivAux1 x).trans (quotientSpanXSubCAlgEquivAux2 x)


@[simp]
theorem quotientSpanXSubCAlgEquiv_mk (x : R) (p : R[X]) :
    quotientSpanXSubCAlgEquiv x (Ideal.Quotient.mk _ p) = p.eval x :=
  rfl


@[simp]
theorem quotientSpanXSubCAlgEquiv_symm_apply (x : R) (y : R) :
    (quotientSpanXSubCAlgEquiv x).symm y = algebraMap R _ y :=
  rfl


/-- For a commutative ring $R$, evaluating a polynomial at an element $y \in R$ induces an
isomorphism of $R$-algebras $R[X] / \langle x, X - y \rangle \cong R / \langle x \rangle$. -/
noncomputable def quotientSpanCXSubCAlgEquiv (x y : R) :
    (R[X] ⧸ (Ideal.span {C x, X - C y} : Ideal R[X])) ≃ₐ[R] R ⧸ (Ideal.span {x} : Ideal R) :=
                                      /-
                                        R : Type u_1
                                        inst✝ : CommRing R
                                        x y : R
                                        ⊢ Eq (Ideal.span (Insert.insert (Polynomial.C x) (Singleton.singleton (HSub.hS …
                                      -/
  (Ideal.quotientEquivAlgOfEq R <| by rw [Ideal.span_insert, sup_comm]).trans <|
                                      /-
                                        🎉 no goals
                                      -/
    (DoubleQuot.quotQuotEquivQuotSupₐ R _ _).symm.trans <|
      (Ideal.quotientEquivAlg _ _ (quotientSpanXSubCAlgEquiv y) rfl).trans <|
        Ideal.quotientEquivAlgOfEq R <| by
          /-
            R : Type u_1
            inst✝ : CommRing R
            x y : R
            ⊢ Eq (Ideal.map (↑(Polynomial.quotientSpanXSubCAlgEquiv y)) (Ideal.map (Ideal. …
          -/
          simp only [Ideal.map_span, Set.image_singleton]; congr 2; exact eval_C
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- For a commutative ring $R$, evaluating a polynomial at elements $y(X) \in R[X]$ and $x \in R$
induces an isomorphism of $R$-algebras $R[X, Y] / \langle X - x, Y - y(X) \rangle \cong R$. -/
noncomputable def quotientSpanCXSubCXSubCAlgEquiv {x : R} {y : R[X]} :
    @AlgEquiv R (R[X][X] ⧸ (Ideal.span {C (X - C x), X - C y} : Ideal <| R[X][X])) R _ _ _
      (Ideal.Quotient.algebra R) _ :=
((quotientSpanCXSubCAlgEquiv (X - C x) y).restrictScalars R).trans <| quotientSpanXSubCAlgEquiv x


lemma modByMonic_eq_zero_iff_quotient_eq_zero (p q : R[X]) (hq : q.Monic) :
    p %ₘ q = 0 ↔ (p : R[X] ⧸ Ideal.span {q}) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hq : q.Monic
    ⊢ Iff (Eq (p.modByMonic q) 0) (Eq ((Ideal.Quotient.mk (Ideal.span (Singleton.s …
  -/
  rw [modByMonic_eq_zero_iff_dvd hq, Ideal.Quotient.eq_zero_iff_dvd]
  /-
    🎉 no goals
  -/


theorem quotient_map_C_eq_zero {I : Ideal R} :
    ∀ a ∈ I, ((Quotient.mk (map (C : R →+* R[X]) I : Ideal R[X])).comp C) a = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ ∀ (a : R), Membership.mem I a → Eq (((Ideal.Quotient.mk (Ideal.map Polynomia …
  -/
  intro a ha
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : R
    ha : Membership.mem I a
    ⊢ Eq (((Ideal.Quotient.mk (Ideal.map Polynomial.C I)).comp Polynomial.C) a) 0
  -/
  rw [RingHom.comp_apply, Quotient.eq_zero_iff_mem]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : R
    ha : Membership.mem I a
    ⊢ Membership.mem (Ideal.map Polynomial.C I) (Polynomial.C a)
  -/
  exact mem_map_of_mem _ ha
  /-
    🎉 no goals
  -/


theorem eval₂_C_mk_eq_zero {I : Ideal R} :
    ∀ f ∈ (map (C : R →+* R[X]) I : Ideal R[X]), eval₂RingHom (C.comp (Quotient.mk I)) X f = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ ∀ (f : Polynomial R), Membership.mem (Ideal.map Polynomial.C I) f → Eq ((Pol …
  -/
  intro a ha
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    ⊢ Eq ((Polynomial.eval₂RingHom (Polynomial.C.comp (Ideal.Quotient.mk I)) Polyn …
  -/
  rw [← sum_monomial_eq a]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    ⊢ Eq ((Polynomial.eval₂RingHom (Polynomial.C.comp (Ideal.Quotient.mk I)) Polyn …
  -/
  dsimp
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    ⊢ Eq (Polynomial.eval₂ (Polynomial.C.comp (Ideal.Quotient.mk I)) Polynomial.X  …
  -/
  rw [eval₂_sum]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    ⊢ Eq (a.sum fun n a => Polynomial.eval₂ (Polynomial.C.comp (Ideal.Quotient.mk  …
  -/
  refine Finset.sum_eq_zero fun n _ => ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    n : Nat
    x✝ : Membership.mem a.support n
    ⊢ Eq ((fun n a => Polynomial.eval₂ (Polynomial.C.comp (Ideal.Quotient.mk I)) P …
  -/
  dsimp
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    n : Nat
    x✝ : Membership.mem a.support n
    ⊢ Eq (Polynomial.eval₂ (Polynomial.C.comp (Ideal.Quotient.mk I)) Polynomial.X  …
  -/
  rw [eval₂_monomial (C.comp (Quotient.mk I)) X]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    n : Nat
    x✝ : Membership.mem a.support n
    ⊢ Eq (HMul.hMul ((Polynomial.C.comp (Ideal.Quotient.mk I)) (a.coeff n)) (HPow. …
  -/
  refine mul_eq_zero_of_left (Polynomial.ext fun m => ?_) (X ^ n)
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    n : Nat
    x✝ : Membership.mem a.support n
    m : Nat
    ⊢ Eq (((Polynomial.C.comp (Ideal.Quotient.mk I)) (a.coeff n)).coeff m) (Polyno …
  -/
  erw [coeff_C]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    a : Polynomial R
    ha : Membership.mem (Ideal.map Polynomial.C I) a
    n : Nat
    x✝ : Membership.mem a.support n
    m : Nat
    ⊢ Eq (ite (Eq m 0) ((Ideal.Quotient.mk I) (a.coeff n)) 0) (Polynomial.coeff 0 m)
  -/
  by_cases h : m = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      a : Polynomial R
      ha : Membership.mem (Ideal.map Polynomial.C I) a
      n : Nat
      x✝ : Membership.mem a.support n
      m : Nat
      h : Eq m 0
      ⊢ Eq (ite (Eq m 0) ((Ideal.Quotient.mk I) (a.coeff n)) 0) (Polynomial.coeff 0 m)
    -/
  · simpa [h] using Quotient.eq_zero_iff_mem.2 ((mem_map_C_iff.1 ha) n)
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      a : Polynomial R
      ha : Membership.mem (Ideal.map Polynomial.C I) a
      n : Nat
      x✝ : Membership.mem a.support n
      m : Nat
      h : Not (Eq m 0)
      ⊢ Eq (ite (Eq m 0) ((Ideal.Quotient.mk I) (a.coeff n)) 0) (Polynomial.coeff 0 m)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


/-- If `I` is an ideal of `R`, then the ring polynomials over the quotient ring `I.quotient` is
isomorphic to the quotient of `R[X]` by the ideal `map C I`,
where `map C I` contains exactly the polynomials whose coefficients all lie in `I`. -/
def polynomialQuotientEquivQuotientPolynomial (I : Ideal R) :
    (R ⧸ I)[X] ≃+* R[X] ⧸ (map C I : Ideal R[X]) where
  toFun :=
    eval₂RingHom
      (Quotient.lift I ((Quotient.mk (map C I : Ideal R[X])).comp C) quotient_map_C_eq_zero)
      (Quotient.mk (map C I : Ideal R[X]) X)
  invFun :=
    Quotient.lift (map C I : Ideal R[X]) (eval₂RingHom (C.comp (Quotient.mk I)) X)
      eval₂_C_mk_eq_zero
                     /-
                       R : Type u_1
                       inst✝ : CommRing R
                       I : Ideal R
                       f g : Polynomial (HasQuotient.Quotient R I)
                       ⊢ Eq ({ toFun := ⇑(Polynomial.eval₂RingHom (Ideal.Quotient.lift I ((Ideal.Quot …
                     -/
  map_mul' f g := by simp only [coe_eval₂RingHom, eval₂_mul]
                     /-
                       🎉 no goals
                     -/
    /-
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      ⊢ Function.LeftInverse ⇑(Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polyn …
    -/
                     /-
                       R : Type u_1
                       inst✝ : CommRing R
                       I : Ideal R
                       f g : Polynomial (HasQuotient.Quotient R I)
                       ⊢ Eq ({ toFun := ⇑(Polynomial.eval₂RingHom (Ideal.Quotient.lift I ((Ideal.Quot …
                     -/
    /-
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      f : Polynomial (HasQuotient.Quotient R I)
      ⊢ Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂RingHom …
    -/
  map_add' f g := by simp only [eval₂_add, coe_eval₂RingHom]
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        f : Polynomial (HasQuotient.Quotient R I)
        ⊢ ∀ (p q : Polynomial (HasQuotient.Quotient R I)), Eq ((Ideal.Quotient.lift (I …
      -/
                     /-
                       🎉 no goals
                     -/
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        f p q : Polynomial (HasQuotient.Quotient R I)
        hp : Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂Ring …
        hq : Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂Ring …
        ⊢ Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂RingHom …
      -/
  left_inv := by
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        f p q : Polynomial (HasQuotient.Quotient R I)
        hp : Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂Ring …
        hq : Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂Ring …
        ⊢ Eq ((Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Polynomial.eval₂RingHom …
      -/
    intro f
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        f : Polynomial (HasQuotient.Quotient R I)
        ⊢ ∀ (n : Nat) (a : HasQuotient.Quotient R I), Eq ((Ideal.Quotient.lift (Ideal. …
      -/
    refine Polynomial.induction_on' f ?_ ?_
    · intro p q hp hq
      simp only [coe_eval₂RingHom] at hp hq
      simp only [coe_eval₂RingHom, hp, hq, RingHom.map_add]
    · rintro n ⟨x⟩
    /-
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      ⊢ Function.RightInverse ⇑(Ideal.Quotient.lift (Ideal.map Polynomial.C I) (Poly …
    -/
      simp only [← smul_X_eq_monomial, C_mul', Quotient.lift_mk, Submodule.Quotient.quot_mk_eq_mk,
    /-
      case mk
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      x✝ : HasQuotient.Quotient (Polynomial R) (Ideal.map Polynomial.C I)
      f : Polynomial R
      ⊢ Eq ((Polynomial.eval₂RingHom (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ide …
    -/
        Quotient.mk_eq_mk, eval₂_X_pow, eval₂_smul, coe_eval₂RingHom, RingHom.map_pow, eval₂_C,
        RingHom.coe_comp, RingHom.map_mul, eval₂_X, Function.comp_apply]
      /-
        case mk.refine_1
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        x✝ : HasQuotient.Quotient (Polynomial R) (Ideal.map Polynomial.C I)
        f : Polynomial R
        ⊢ ∀ (p q : Polynomial R), Eq ((Polynomial.eval₂RingHom (Ideal.Quotient.lift I  …
      -/
  right_inv := by
    rintro ⟨f⟩
    refine Polynomial.induction_on' f ?_ ?_
      /-
        case mk.refine_1
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        x✝ : HasQuotient.Quotient (Polynomial R) (Ideal.map Polynomial.C I)
        f p q : Polynomial R
        hp : Eq (Polynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal.ma …
        hq : Eq (Polynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal.ma …
        ⊢ Eq (HAdd.hAdd (Polynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk ( …
      -/
    · -- Porting note: was `simp_intro p q hp hq`
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_2
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        x✝ : HasQuotient.Quotient (Polynomial R) (Ideal.map Polynomial.C I)
        f : Polynomial R
        ⊢ ∀ (n : Nat) (a : R), Eq ((Polynomial.eval₂RingHom (Ideal.Quotient.lift I ((I …
      -/
      intros p q hp hq
      simp only [Submodule.Quotient.quot_mk_eq_mk, Quotient.mk_eq_mk, map_add, Quotient.lift_mk,
        coe_eval₂RingHom] at hp hq ⊢
      rw [hp, hq]
    · intro n a
      simp only [← smul_X_eq_monomial, ← C_mul' a (X ^ n), Quotient.lift_mk,
        Submodule.Quotient.quot_mk_eq_mk, Quotient.mk_eq_mk, eval₂_X_pow, eval₂_smul,
        coe_eval₂RingHom, RingHom.map_pow, eval₂_C, RingHom.coe_comp, RingHom.map_mul, eval₂_X,
        Function.comp_apply]


@[simp]
theorem polynomialQuotientEquivQuotientPolynomial_symm_mk (I : Ideal R) (f : R[X]) :
    I.polynomialQuotientEquivQuotientPolynomial.symm (Quotient.mk _ f) = f.map (Quotient.mk I) := by
  rw [polynomialQuotientEquivQuotientPolynomial, RingEquiv.symm_mk, RingEquiv.coe_mk,
    Equiv.coe_fn_mk, Quotient.lift_mk, coe_eval₂RingHom, eval₂_eq_eval_map, ← Polynomial.map_map,
    ← eval₂_eq_eval_map, Polynomial.eval₂_C_X]


@[simp]
theorem polynomialQuotientEquivQuotientPolynomial_map_mk (I : Ideal R) (f : R[X]) :
    I.polynomialQuotientEquivQuotientPolynomial (f.map <| Quotient.mk I) =
    Quotient.mk (map C I : Ideal R[X]) f := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    f : Polynomial R
    ⊢ Eq (I.polynomialQuotientEquivQuotientPolynomial (Polynomial.map (Ideal.Quoti …
  -/
  apply (polynomialQuotientEquivQuotientPolynomial I).symm.injective
  /-
    case a
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    f : Polynomial R
    ⊢ Eq (I.polynomialQuotientEquivQuotientPolynomial.symm (I.polynomialQuotientEq …
  -/
  rw [RingEquiv.symm_apply_apply, polynomialQuotientEquivQuotientPolynomial_symm_mk]
  /-
    🎉 no goals
  -/


/-- If `P` is a prime ideal of `R`, then `R[x]/(P)` is an integral domain. -/
theorem isDomain_map_C_quotient {P : Ideal R} (_ : IsPrime P) :
    IsDomain (R[X] ⧸ (map (C : R →+* R[X]) P : Ideal R[X])) :=
  MulEquiv.isDomain (Polynomial (R ⧸ P)) (polynomialQuotientEquivQuotientPolynomial P).symm


/-- Given any ring `R` and an ideal `I` of `R[X]`, we get a map `R → R[x] → R[x]/I`.
  If we let `R` be the image of `R` in `R[x]/I` then we also have a map `R[x] → R'[x]`.
  In particular we can map `I` across this map, to get `I'` and a new map `R' → R'[x] → R'[x]/I`.
  This theorem shows `I'` will not contain any non-zero constant polynomials. -/
theorem eq_zero_of_polynomial_mem_map_range (I : Ideal R[X]) (x : ((Quotient.mk I).comp C).range)
    (hx : C x ∈ I.map (Polynomial.mapRingHom ((Quotient.mk I).comp C).rangeRestrict)) : x = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal (Polynomial R)
    x : Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Polynomial.C). …
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom ((Ideal.Quotient.mk I).c …
    ⊢ Eq x 0
  -/
  let i := ((Quotient.mk I).comp C).rangeRestrict
  have hi' : RingHom.ker (Polynomial.mapRingHom i) ≤ I := by
    refine fun f hf => polynomial_mem_ideal_of_coeff_mem_ideal I f fun n => ?_
    rw [mem_comap, ← Quotient.eq_zero_iff_mem, ← RingHom.comp_apply]
    rw [RingHom.mem_ker, coe_mapRingHom] at hf
    replace hf := congr_arg (fun f : Polynomial _ => f.coeff n) hf
    simp only [coeff_map, coeff_zero] at hf
    rwa [Subtype.ext_iff, RingHom.coe_rangeRestrict] at hf
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal (Polynomial R)
    x : Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Polynomial.C). …
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom ((Ideal.Quotient.mk I).c …
    i : RingHom R (Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Pol …
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    ⊢ Eq x 0
  -/
  obtain ⟨x, hx'⟩ := x
  /-
    case mk
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal (Polynomial R)
    i : RingHom R (Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Pol …
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    x : HasQuotient.Quotient (Polynomial R) I
    hx' : Membership.mem ((Ideal.Quotient.mk I).comp Polynomial.C).range x
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom ((Ideal.Quotient.mk I).c …
    ⊢ Eq ⟨x, hx'⟩ 0
  -/
  obtain ⟨y, rfl⟩ := RingHom.mem_range.1 hx'
  /-
    case mk.intro
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal (Polynomial R)
    i : RingHom R (Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Pol …
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    y : R
    hx' : Membership.mem ((Ideal.Quotient.mk I).comp Polynomial.C).range (((Ideal. …
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom ((Ideal.Quotient.mk I).c …
    ⊢ Eq ⟨((Ideal.Quotient.mk I).comp Polynomial.C) y, hx'⟩ 0
  -/
  refine Subtype.eq ?_
  /-
    case mk.intro
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal (Polynomial R)
    i : RingHom R (Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Pol …
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    y : R
    hx' : Membership.mem ((Ideal.Quotient.mk I).comp Polynomial.C).range (((Ideal. …
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom ((Ideal.Quotient.mk I).c …
    ⊢ Eq ↑⟨((Ideal.Quotient.mk I).comp Polynomial.C) y, hx'⟩ ↑0
  -/
  simp only [RingHom.comp_apply, Quotient.eq_zero_iff_mem, ZeroMemClass.coe_zero]
  suffices C (i y) ∈ I.map (Polynomial.mapRingHom i) by
    obtain ⟨f, hf⟩ := mem_image_of_mem_map_of_surjective (Polynomial.mapRingHom i)
      (Polynomial.map_surjective _ (RingHom.rangeRestrict_surjective ((Quotient.mk I).comp C))) this
    refine sub_add_cancel (C y) f ▸ I.add_mem (hi' ?_ : C y - f ∈ I) hf.1
    rw [RingHom.mem_ker, RingHom.map_sub, hf.2, sub_eq_zero, coe_mapRingHom, map_C]
  /-
    case mk.intro
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal (Polynomial R)
    i : RingHom R (Subtype fun x => Membership.mem ((Ideal.Quotient.mk I).comp Pol …
    hi' : LE.le (RingHom.ker (Polynomial.mapRingHom i)) I
    y : R
    hx' : Membership.mem ((Ideal.Quotient.mk I).comp Polynomial.C).range (((Ideal. …
    hx : Membership.mem (Ideal.map (Polynomial.mapRingHom ((Ideal.Quotient.mk I).c …
    ⊢ Membership.mem (Ideal.map (Polynomial.mapRingHom i) I) (Polynomial.C (i y))
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem quotient_map_C_eq_zero {I : Ideal R} {i : R} (hi : i ∈ I) :
    (Ideal.Quotient.mk (Ideal.map (C : R →+* MvPolynomial σ R) I :
      Ideal (MvPolynomial σ R))).comp C i = 0 := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    i : R
    hi : Membership.mem I i
    ⊢ Eq (((Ideal.Quotient.mk (Ideal.map MvPolynomial.C I)).comp MvPolynomial.C) i …
  -/
  simp only [Function.comp_apply, RingHom.coe_comp, Ideal.Quotient.eq_zero_iff_mem]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    i : R
    hi : Membership.mem I i
    ⊢ Membership.mem (Ideal.map MvPolynomial.C I) (MvPolynomial.C i)
  -/
  exact Ideal.mem_map_of_mem _ hi
  /-
    🎉 no goals
  -/


theorem eval₂_C_mk_eq_zero {I : Ideal R} {a : MvPolynomial σ R}
    (ha : a ∈ (Ideal.map (C : R →+* MvPolynomial σ R) I : Ideal (MvPolynomial σ R))) :
    eval₂Hom (C.comp (Ideal.Quotient.mk I)) X a = 0 := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    a : MvPolynomial σ R
    ha : Membership.mem (Ideal.map MvPolynomial.C I) a
    ⊢ Eq ((MvPolynomial.eval₂Hom (MvPolynomial.C.comp (Ideal.Quotient.mk I)) MvPol …
  -/
  rw [as_sum a]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    a : MvPolynomial σ R
    ha : Membership.mem (Ideal.map MvPolynomial.C I) a
    ⊢ Eq ((MvPolynomial.eval₂Hom (MvPolynomial.C.comp (Ideal.Quotient.mk I)) MvPol …
  -/
  rw [coe_eval₂Hom, eval₂_sum]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    a : MvPolynomial σ R
    ha : Membership.mem (Ideal.map MvPolynomial.C I) a
    ⊢ Eq (a.support.sum fun x => MvPolynomial.eval₂ (MvPolynomial.C.comp (Ideal.Qu …
  -/
  refine Finset.sum_eq_zero fun n _ => ?_
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    a : MvPolynomial σ R
    ha : Membership.mem (Ideal.map MvPolynomial.C I) a
    n : Finsupp σ Nat
    x✝ : Membership.mem a.support n
    ⊢ Eq (MvPolynomial.eval₂ (MvPolynomial.C.comp (Ideal.Quotient.mk I)) MvPolynom …
  -/
  simp only [eval₂_monomial, Function.comp_apply, RingHom.coe_comp]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    a : MvPolynomial σ R
    ha : Membership.mem (Ideal.map MvPolynomial.C I) a
    n : Finsupp σ Nat
    x✝ : Membership.mem a.support n
    ⊢ Eq (HMul.hMul (MvPolynomial.C ((Ideal.Quotient.mk I) (MvPolynomial.coeff n a …
  -/
  refine mul_eq_zero_of_left ?_ _
  suffices coeff n a ∈ I by
    rw [← @Ideal.mk_ker R _ I, RingHom.mem_ker] at this
    simp only [this, C_0]
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    a : MvPolynomial σ R
    ha : Membership.mem (Ideal.map MvPolynomial.C I) a
    n : Finsupp σ Nat
    x✝ : Membership.mem a.support n
    ⊢ Membership.mem I (MvPolynomial.coeff n a)
  -/
  exact mem_map_C_iff.1 ha n
  /-
    🎉 no goals
  -/


lemma quotientEquivQuotientMvPolynomial_rightInverse (I : Ideal R) :
    Function.RightInverse
      (eval₂ (Ideal.Quotient.lift I
        ((Ideal.Quotient.mk (Ideal.map C I : Ideal (MvPolynomial σ R))).comp C)
          fun _ hi => quotient_map_C_eq_zero hi)
          fun i => Ideal.Quotient.mk (Ideal.map C I : Ideal (MvPolynomial σ R)) (X i))
      (Ideal.Quotient.lift (Ideal.map C I : Ideal (MvPolynomial σ R))
        (eval₂Hom (C.comp (Ideal.Quotient.mk I)) X) fun _ ha => eval₂_C_mk_eq_zero ha) := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Function.RightInverse (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quo …
  -/
  intro f
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    f : MvPolynomial σ (HasQuotient.Quotient R I)
    ⊢ Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂Hom …
  -/
  apply induction_on f
    /-
      case h_C
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ (HasQuotient.Quotient R I)
      ⊢ ∀ (a : HasQuotient.Quotient R I), Eq ((Ideal.Quotient.lift (Ideal.map MvPoly …
    -/
  · intro r
    /-
      case h_C
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ (HasQuotient.Quotient R I)
      r : HasQuotient.Quotient R I
      ⊢ Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂Hom …
    -/
    obtain ⟨r, rfl⟩ := Ideal.Quotient.mk_surjective r
    rw [eval₂_C, Ideal.Quotient.lift_mk, RingHom.comp_apply, Ideal.Quotient.lift_mk, eval₂Hom_C,
      RingHom.comp_apply]
    /-
      case h_add
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ (HasQuotient.Quotient R I)
      ⊢ ∀ (p q : MvPolynomial σ (HasQuotient.Quotient R I)), Eq ((Ideal.Quotient.lif …
    -/
  · intros p q hp hq
    simp only [RingHom.map_add, MvPolynomial.coe_eval₂Hom, coe_eval₂Hom, MvPolynomial.eval₂_add]
      at hp hq ⊢
    /-
      case h_add
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f p q : MvPolynomial σ (HasQuotient.Quotient R I)
      hp : Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂ …
      hq : Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂ …
      ⊢ Eq (HAdd.hAdd ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomi …
    -/
    rw [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ (HasQuotient.Quotient R I)
      ⊢ ∀ (p : MvPolynomial σ (HasQuotient.Quotient R I)) (n : σ), Eq ((Ideal.Quotie …
    -/
  · intros p i hp
    /-
      case h_X
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f p : MvPolynomial σ (HasQuotient.Quotient R I)
      i : σ
      hp : Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂ …
      ⊢ Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂Hom …
    -/
    simp only [coe_eval₂Hom] at hp
    /-
      case h_X
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f p : MvPolynomial σ (HasQuotient.Quotient R I)
      i : σ
      hp : Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂ …
      ⊢ Eq ((Ideal.Quotient.lift (Ideal.map MvPolynomial.C I) (MvPolynomial.eval₂Hom …
    -/
    simp only [hp, coe_eval₂Hom, Ideal.Quotient.lift_mk, eval₂_mul, RingHom.map_mul, eval₂_X]
    /-
      🎉 no goals
    -/


lemma quotientEquivQuotientMvPolynomial_leftInverse (I : Ideal R) :
    Function.LeftInverse
      (eval₂ (Ideal.Quotient.lift I
        ((Ideal.Quotient.mk (Ideal.map C I : Ideal (MvPolynomial σ R))).comp C)
          fun _ hi => quotient_map_C_eq_zero hi)
          fun i => Ideal.Quotient.mk (Ideal.map C I : Ideal (MvPolynomial σ R)) (X i))
      (Ideal.Quotient.lift (Ideal.map C I : Ideal (MvPolynomial σ R))
        (eval₂Hom (C.comp (Ideal.Quotient.mk I)) X) fun _ ha => eval₂_C_mk_eq_zero ha) := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Function.LeftInverse (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quot …
  -/
  intro f
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    f : HasQuotient.Quotient (MvPolynomial σ R) (Ideal.map MvPolynomial.C I)
    ⊢ Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal.map …
  -/
  obtain ⟨f, rfl⟩ := Ideal.Quotient.mk_surjective f
  /-
    case intro
    R : Type u_1
    σ : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    f : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal.map …
  -/
  apply induction_on f
    /-
      case intro.h_C
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk …
    -/
  · intro r
    rw [Ideal.Quotient.lift_mk, eval₂Hom_C, RingHom.comp_apply, eval₂_C, Ideal.Quotient.lift_mk,
      RingHom.comp_apply]
    /-
      case intro.h_add
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ R
      ⊢ ∀ (p q : MvPolynomial σ R), Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I (( …
    -/
  · intros p q hp hq
    /-
      case intro.h_add
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f p q : MvPolynomial σ R
      hp : Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal. …
      hq : Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal. …
      ⊢ Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal.map …
    -/
    rw [Ideal.Quotient.lift_mk] at hp hq ⊢
    simp only [Submodule.Quotient.quot_mk_eq_mk, eval₂_add, RingHom.map_add, coe_eval₂Hom,
      Ideal.Quotient.lift_mk, Ideal.Quotient.mk_eq_mk] at hp hq ⊢
    /-
      case intro.h_add
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f p q : MvPolynomial σ R
      hp : Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal. …
      hq : Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal. …
      ⊢ Eq (HAdd.hAdd (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk …
    -/
    rw [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case intro.h_X
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f : MvPolynomial σ R
      ⊢ ∀ (p : MvPolynomial σ R) (n : σ), Eq (MvPolynomial.eval₂ (Ideal.Quotient.lif …
    -/
  · intros p i hp
    simp only [Submodule.Quotient.quot_mk_eq_mk, coe_eval₂Hom, Ideal.Quotient.lift_mk,
      Ideal.Quotient.mk_eq_mk, eval₂_mul, RingHom.map_mul, eval₂_X] at hp ⊢
    /-
      case intro.h_X
      R : Type u_1
      σ : Type u_2
      inst✝ : CommRing R
      I : Ideal R
      f p : MvPolynomial σ R
      i : σ
      hp : Eq (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal. …
      ⊢ Eq (HMul.hMul (MvPolynomial.eval₂ (Ideal.Quotient.lift I ((Ideal.Quotient.mk …
    -/
    simp only [hp]
    /-
      🎉 no goals
    -/

-- Porting note: this definition was split to avoid timeouts.

/-- If `I` is an ideal of `R`, then the ring `MvPolynomial σ I.quotient` is isomorphic as an
`R`-algebra to the quotient of `MvPolynomial σ R` by the ideal generated by `I`. -/
noncomputable def quotientEquivQuotientMvPolynomial (I : Ideal R) :
    MvPolynomial σ (R ⧸ I) ≃ₐ[R] MvPolynomial σ R ⧸ (Ideal.map C I : Ideal (MvPolynomial σ R)) :=
  let e : MvPolynomial σ (R ⧸ I) →ₐ[R]
      MvPolynomial σ R ⧸ (Ideal.map C I : Ideal (MvPolynomial σ R)) :=
    { eval₂Hom
      (Ideal.Quotient.lift I ((Ideal.Quotient.mk (Ideal.map C I : Ideal (MvPolynomial σ R))).comp C)
        fun _ hi => quotient_map_C_eq_zero hi)
      fun i => Ideal.Quotient.mk (Ideal.map C I : Ideal (MvPolynomial σ R)) (X i) with
      commutes' := fun r => eval₂Hom_C _ _ (Ideal.Quotient.mk I r) }
  { e with
    invFun := Ideal.Quotient.lift (Ideal.map C I : Ideal (MvPolynomial σ R))
      (eval₂Hom (C.comp (Ideal.Quotient.mk I)) X) fun _ ha => eval₂_C_mk_eq_zero ha
    left_inv := quotientEquivQuotientMvPolynomial_rightInverse I
    right_inv := quotientEquivQuotientMvPolynomial_leftInverse I }


