theorem smul_eq_map [MulSemiringAction M R] (m : M) :
    HSMul.hSMul m = map (MulSemiringAction.toRingHom M R m) := by
  suffices DistribMulAction.toAddMonoidHom R[X] m =
      (mapRingHom (MulSemiringAction.toRingHom M R m)).toAddMonoidHom by
    ext1 r
    exact DFunLike.congr_fun this r
  /-
    M : Type u_1
    inst✝² : Monoid M
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    m : M
    ⊢ Eq (DistribMulAction.toAddMonoidHom (Polynomial R) m) (Polynomial.mapRingHom …
  -/
  ext n r : 2
  /-
    case h.h
    M : Type u_1
    inst✝² : Monoid M
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    m : M
    n : Nat
    r : R
    ⊢ Eq (((DistribMulAction.toAddMonoidHom (Polynomial R) m).comp (Polynomial.mon …
  -/
  change m • monomial n r = map (MulSemiringAction.toRingHom M R m) (monomial n r)
  /-
    case h.h
    M : Type u_1
    inst✝² : Monoid M
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    m : M
    n : Nat
    r : R
    ⊢ Eq (HSMul.hSMul m ((Polynomial.monomial n) r)) (Polynomial.map (MulSemiringA …
  -/
  rw [Polynomial.map_monomial, Polynomial.smul_monomial, MulSemiringAction.toRingHom_apply]
  /-
    🎉 no goals
  -/


noncomputable instance [MulSemiringAction M R] : MulSemiringAction M R[X] :=
  { Polynomial.distribMulAction with
    smul_one := fun m ↦
      smul_eq_map R m ▸ Polynomial.map_one (MulSemiringAction.toRingHom M R m)
    smul_mul := fun m _ _ ↦
      smul_eq_map R m ▸ Polynomial.map_mul (MulSemiringAction.toRingHom M R m) }


@[simp]
theorem smul_X (m : M) : (m • X : R[X]) = X :=
  (smul_eq_map R m).symm ▸ map_X _


theorem smul_eval_smul (m : M) (f : S[X]) (x : S) : (m • f).eval (m • x) = m • f.eval x :=
                                        /-
                                          M : Type u_1
                                          inst✝² : Monoid M
                                          S : Type u_3
                                          inst✝¹ : CommSemiring S
                                          inst✝ : MulSemiringAction M S
                                          m : M
                                          f : Polynomial S
                                          x r : S
                                          ⊢ Eq (Polynomial.eval (HSMul.hSMul m x) (HSMul.hSMul m (Polynomial.C r))) (HSM …
                                        -/
  Polynomial.induction_on f (fun r ↦ by rw [smul_C, eval_C, eval_C])
                                        /-
                                          🎉 no goals
                                        -/
                          /-
                            M : Type u_1
                            inst✝² : Monoid M
                            S : Type u_3
                            inst✝¹ : CommSemiring S
                            inst✝ : MulSemiringAction M S
                            m : M
                            f✝ : Polynomial S
                            x : S
                            f g : Polynomial S
                            ihf : Eq (Polynomial.eval (HSMul.hSMul m x) (HSMul.hSMul m f)) (HSMul.hSMul m  …
                            ihg : Eq (Polynomial.eval (HSMul.hSMul m x) (HSMul.hSMul m g)) (HSMul.hSMul m  …
                            ⊢ Eq (Polynomial.eval (HSMul.hSMul m x) (HSMul.hSMul m (HAdd.hAdd f g))) (HSMu …
                          -/
    (fun f g ihf ihg ↦ by rw [smul_add, eval_add, ihf, ihg, eval_add, smul_add]) fun n r _ ↦ by
                          /-
                            🎉 no goals
                          -/
    rw [smul_mul', smul_pow', smul_C, smul_X, eval_mul, eval_C, eval_pow, eval_X, eval_mul, eval_C,
      eval_pow, eval_X, smul_mul', smul_pow']


theorem eval_smul' [MulSemiringAction G S] (g : G) (f : S[X]) (x : S) :
    f.eval (g • x) = g • (g⁻¹ • f).eval x := by
  /-
    S : Type u_3
    inst✝² : CommSemiring S
    G : Type u_4
    inst✝¹ : Group G
    inst✝ : MulSemiringAction G S
    g : G
    f : Polynomial S
    x : S
    ⊢ Eq (Polynomial.eval (HSMul.hSMul g x) f) (HSMul.hSMul g (Polynomial.eval x ( …
  -/
  rw [← smul_eval_smul, smul_inv_smul]
  /-
    🎉 no goals
  -/


theorem smul_eval [MulSemiringAction G S] (g : G) (f : S[X]) (x : S) :
    (g • f).eval x = g • f.eval (g⁻¹ • x) := by
  /-
    S : Type u_3
    inst✝² : CommSemiring S
    G : Type u_4
    inst✝¹ : Group G
    inst✝ : MulSemiringAction G S
    g : G
    f : Polynomial S
    x : S
    ⊢ Eq (Polynomial.eval x (HSMul.hSMul g f)) (HSMul.hSMul g (Polynomial.eval (HS …
  -/
  rw [← smul_eval_smul, smul_inv_smul]
  /-
    🎉 no goals
  -/


/-- the product of `(X - g • x)` over distinct `g • x`. -/
noncomputable def prodXSubSMul (x : R) : R[X] :=
  letI := Classical.decEq R
  (Finset.univ : Finset (G ⧸ MulAction.stabilizer G x)).prod fun g ↦
    Polynomial.X - Polynomial.C (ofQuotientStabilizer G x g)


theorem prodXSubSMul.monic (x : R) : (prodXSubSMul G R x).Monic :=
  Polynomial.monic_prod_of_monic _ _ fun _ _ ↦ Polynomial.monic_X_sub_C _


theorem prodXSubSMul.eval (x : R) : (prodXSubSMul G R x).eval x = 0 :=
  letI := Classical.decEq R
  (map_prod ((Polynomial.aeval x).toRingHom.toMonoidHom : R[X] →* R) _ _).trans <|
                                                                      /-
                                                                        G : Type u_2
                                                                        inst✝³ : Group G
                                                                        inst✝² : Fintype G
                                                                        R : Type u_3
                                                                        inst✝¹ : CommRing R
                                                                        inst✝ : MulSemiringAction G R
                                                                        x : R
                                                                        this : DecidableEq R := Classical.decEq R
                                                                        ⊢ Eq (↑(Polynomial.aeval x).toRingHom (HSub.hSub Polynomial.X (Polynomial.C (M …
                                                                      -/
    Finset.prod_eq_zero (Finset.mem_univ <| QuotientGroup.mk 1) <| by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem prodXSubSMul.smul (x : R) (g : G) : g • prodXSubSMul G R x = prodXSubSMul G R x :=
  letI := Classical.decEq R
  Finset.smul_prod'.trans <|
    Fintype.prod_bijective _ (MulAction.bijective g) _ _ fun g' ↦ by
      /-
        G : Type u_2
        inst✝³ : Group G
        inst✝² : Fintype G
        R : Type u_3
        inst✝¹ : CommRing R
        inst✝ : MulSemiringAction G R
        x : R
        g : G
        this : DecidableEq R := Classical.decEq R
        g' : HasQuotient.Quotient G (MulAction.stabilizer G x)
        ⊢ Eq (HSMul.hSMul g (HSub.hSub Polynomial.X (Polynomial.C (MulAction.ofQuotien …
      -/
      rw [ofQuotientStabilizer_smul, smul_sub, Polynomial.smul_X, Polynomial.smul_C]
      /-
        🎉 no goals
      -/


theorem prodXSubSMul.coeff (x : R) (g : G) (n : ℕ) :
    g • (prodXSubSMul G R x).coeff n = (prodXSubSMul G R x).coeff n := by
  /-
    G : Type u_2
    inst✝³ : Group G
    inst✝² : Fintype G
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : MulSemiringAction G R
    x : R
    g : G
    n : Nat
    ⊢ Eq (HSMul.hSMul g ((prodXSubSMul G R x).coeff n)) ((prodXSubSMul G R x).coef …
  -/
  rw [← Polynomial.coeff_smul, prodXSubSMul.smul]
  /-
    🎉 no goals
  -/


/-- An equivariant map induces an equivariant map on polynomials. -/
protected noncomputable def polynomial (g : P →+*[M] Q) : P[X] →+*[M] Q[X] where
  toFun := map g
  map_smul' m p :=
    Polynomial.induction_on p
      (fun b ↦ by rw [MonoidHom.id_apply, smul_C, map_C, coe_fn_coe, g.map_smul, map_C,
          coe_fn_coe, smul_C])
      (fun p q ihp ihq ↦ by
        /-
          M : Type u_1
          inst✝⁴ : Monoid M
          P : Type u_2
          inst✝³ : CommSemiring P
          inst✝² : MulSemiringAction M P
          Q : Type u_3
          inst✝¹ : CommSemiring Q
          inst✝ : MulSemiringAction M Q
          g : MulSemiringActionHom (MonoidHom.id M) P Q
          m : M
          p✝ p q : Polynomial P
          ihp : Eq (Polynomial.map (↑g) (HSMul.hSMul m p)) (HSMul.hSMul ((MonoidHom.id M …
          ihq : Eq (Polynomial.map (↑g) (HSMul.hSMul m q)) (HSMul.hSMul ((MonoidHom.id M …
          ⊢ Eq (Polynomial.map (↑g) (HSMul.hSMul m (HAdd.hAdd p q))) (HSMul.hSMul ((Mono …
        -/
        rw [smul_add, Polynomial.map_add, ihp, ihq, Polynomial.map_add, smul_add])
        /-
          🎉 no goals
        -/
      fun n b _ ↦ by rw [MonoidHom.id_apply, smul_mul', smul_C, smul_pow', smul_X,
        Polynomial.map_mul, map_C, Polynomial.map_pow,
        map_X, coe_fn_coe, g.map_smul, Polynomial.map_mul, map_C, Polynomial.map_pow, map_X,
        smul_mul', smul_C, smul_pow', smul_X, coe_fn_coe]
  -- Porting note: added `.toRingHom`
  map_zero' := Polynomial.map_zero g.toRingHom
  map_add' _ _ := Polynomial.map_add g.toRingHom
  map_one' := Polynomial.map_one g.toRingHom
  map_mul' _ _ := Polynomial.map_mul g.toRingHom


@[simp]
theorem coe_polynomial (g : P →+*[M] Q) : (g.polynomial : P[X] → Q[X]) = map g := rfl


