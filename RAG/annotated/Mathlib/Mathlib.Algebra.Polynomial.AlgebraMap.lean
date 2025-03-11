/-- Note that this instance also provides `Algebra R R[X]`. -/
instance algebraOfAlgebra : Algebra R A[X] where
  smul_def' r p :=
    toFinsupp_injective <| by
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q r✝ : Polynomial R
        r : R
        p : Polynomial A
        ⊢ Eq (HSMul.hSMul r p).toFinsupp (HMul.hMul ((Polynomial.C.comp (algebraMap R  …
      -/
      dsimp only [RingHom.toFun_eq_coe, RingHom.comp_apply]
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q r✝ : Polynomial R
        r : R
        p : Polynomial A
        ⊢ Eq (HSMul.hSMul r p).toFinsupp (HMul.hMul (Polynomial.C ((algebraMap R A) r) …
      -/
      rw [toFinsupp_smul, toFinsupp_mul, toFinsupp_C]
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q r✝ : Polynomial R
        r : R
        p : Polynomial A
        ⊢ Eq (HSMul.hSMul r p.toFinsupp) (HMul.hMul (AddMonoidAlgebra.single 0 ((algeb …
      -/
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q r✝ : Polynomial R
        r : R
        p : Polynomial A
        ⊢ Eq (HMul.hMul ((Polynomial.C.comp (algebraMap R A)) r) p).toFinsupp (HMul.hM …
      -/
      exact Algebra.smul_def' _ _
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q r✝ : Polynomial R
        r : R
        p : Polynomial A
        ⊢ Eq (HMul.hMul (Polynomial.C ((algebraMap R A) r)) p).toFinsupp (HMul.hMul p  …
      -/
      /-
        🎉 no goals
      -/
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q r✝ : Polynomial R
        r : R
        p : Polynomial A
        ⊢ Eq (HMul.hMul (AddMonoidAlgebra.single 0 ((algebraMap R A) r)) p.toFinsupp)  …
      -/
  commutes' r p :=
      /-
        🎉 no goals
      -/
    toFinsupp_injective <| by
      dsimp only [RingHom.toFun_eq_coe, RingHom.comp_apply]
      simp_rw [toFinsupp_mul, toFinsupp_C]
      convert Algebra.commutes' r p.toFinsupp
  toRingHom := C.comp (algebraMap R A)


@[simp]
theorem algebraMap_apply (r : R) : algebraMap R A[X] r = C (algebraMap R A r) :=
  rfl


@[simp]
theorem toFinsupp_algebraMap (r : R) : (algebraMap R A[X] r).toFinsupp = algebraMap R _ r :=
  show toFinsupp (C (algebraMap _ _ r)) = _ by
    /-
      R : Type u
      A : Type z
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      r : R
      ⊢ Eq (Polynomial.C ((algebraMap R A) r)).toFinsupp ((algebraMap R (AddMonoidAl …
    -/
    rw [toFinsupp_C]
    /-
      R : Type u
      A : Type z
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      r : R
      ⊢ Eq (AddMonoidAlgebra.single 0 ((algebraMap R A) r)) ((algebraMap R (AddMonoi …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem ofFinsupp_algebraMap (r : R) : (⟨algebraMap R _ r⟩ : A[X]) = algebraMap R A[X] r :=
  toFinsupp_injective (toFinsupp_algebraMap _).symm


/-- When we have `[CommSemiring R]`, the function `C` is the same as `algebraMap R R[X]`.

(But note that `C` is defined when `R` is not necessarily commutative, in which case
`algebraMap` is not available.)
-/
theorem C_eq_algebraMap (r : R) : C r = algebraMap R R[X] r :=
  rfl


@[simp]
theorem algebraMap_eq : algebraMap R R[X] = C :=
  rfl


/-- `Polynomial.C` as an `AlgHom`. -/
@[simps! apply]
def CAlgHom : A →ₐ[R] A[X] where
  toRingHom := C
  commutes' _ := rfl


/-- Extensionality lemma for algebra maps out of `A'[X]` over a smaller base ring than `A'`
-/
@[ext 1100]
theorem algHom_ext' {f g : A[X] →ₐ[R] B}
    (hC : f.comp CAlgHom = g.comp CAlgHom)
    (hX : f X = g X) : f = g :=
  AlgHom.coe_ringHom_injective (ringHom_ext' (congr_arg AlgHom.toRingHom hC) hX)


open AddMonoidAlgebra in
/-- Algebra isomorphism between `R[X]` and `R[ℕ]`. This is just an
implementation detail, but it can be useful to transfer results from `Finsupp` to polynomials. -/
@[simps!]
def toFinsuppIsoAlg : R[X] ≃ₐ[R] R[ℕ] :=
  { toFinsuppIso R with
    commutes' := fun r => by
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p q r✝ : Polynomial R
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap R (Polynomial R)) r)) ((algebraMap R (AddMonoi …
      -/
      dsimp }
      /-
        🎉 no goals
      -/


instance subalgebraNontrivial [Nontrivial A] : Nontrivial (Subalgebra R A[X]) :=
  ⟨⟨⊥, ⊤, by
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Semiring B
        inst✝² : Algebra R A
        inst✝¹ : Algebra R B
        p q r : Polynomial R
        inst✝ : Nontrivial A
        ⊢ Ne Bot.bot Top.top
      -/
      rw [Ne, SetLike.ext_iff, not_forall]
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Semiring B
        inst✝² : Algebra R A
        inst✝¹ : Algebra R B
        p q r : Polynomial R
        inst✝ : Nontrivial A
        ⊢ Exists fun x => Not (Iff (Membership.mem Bot.bot x) (Membership.mem Top.top  …
      -/
      refine ⟨X, ?_⟩
      simp only [Algebra.mem_bot, not_exists, Set.mem_range, iff_true, Algebra.mem_top,
        algebraMap_apply, not_forall]
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Semiring B
        inst✝² : Algebra R A
        inst✝¹ : Algebra R B
        p q r : Polynomial R
        inst✝ : Nontrivial A
        ⊢ ∀ (x : R), Not (Eq (Polynomial.C ((algebraMap R A) x)) Polynomial.X)
      -/
      intro x
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Semiring B
        inst✝² : Algebra R A
        inst✝¹ : Algebra R B
        p q r : Polynomial R
        inst✝ : Nontrivial A
        x : R
        ⊢ Not (Eq (Polynomial.C ((algebraMap R A) x)) Polynomial.X)
      -/
      rw [ext_iff, not_forall]
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Semiring B
        inst✝² : Algebra R A
        inst✝¹ : Algebra R B
        p q r : Polynomial R
        inst✝ : Nontrivial A
        x : R
        ⊢ Exists fun x_1 => Not (Eq ((Polynomial.C ((algebraMap R A) x)).coeff x_1) (P …
      -/
      refine ⟨1, ?_⟩
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Semiring B
        inst✝² : Algebra R A
        inst✝¹ : Algebra R B
        p q r : Polynomial R
        inst✝ : Nontrivial A
        x : R
        ⊢ Not (Eq ((Polynomial.C ((algebraMap R A) x)).coeff 1) (Polynomial.X.coeff 1))
      -/
      simp [coeff_C]⟩⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem algHom_eval₂_algebraMap {R A B : Type*} [CommSemiring R] [Semiring A] [Semiring B]
    [Algebra R A] [Algebra R B] (p : R[X]) (f : A →ₐ[R] B) (a : A) :
    f (eval₂ (algebraMap R A) a p) = eval₂ (algebraMap R B) (f a) p := by
  /-
    R : Type u_3
    A : Type u_4
    B : Type u_5
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    p : Polynomial R
    f : AlgHom R A B
    a : A
    ⊢ Eq (f (Polynomial.eval₂ (algebraMap R A) a p)) (Polynomial.eval₂ (algebraMap …
  -/
  simp only [eval₂_eq_sum, sum_def]
  /-
    R : Type u_3
    A : Type u_4
    B : Type u_5
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    p : Polynomial R
    f : AlgHom R A B
    a : A
    ⊢ Eq (f (p.support.sum fun n => HMul.hMul ((algebraMap R A) (p.coeff n)) (HPow …
  -/
  simp only [map_sum, map_mul, map_pow, eq_intCast, map_intCast, AlgHom.commutes]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_algebraMap_X {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A] (p : R[X])
    (f : R[X] →ₐ[R] A) : eval₂ (algebraMap R A) (f X) p = f p := by
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    f : AlgHom R (Polynomial R) A
    ⊢ Eq (Polynomial.eval₂ (algebraMap R A) (f Polynomial.X) p) (f p)
  -/
  conv_rhs => rw [← Polynomial.sum_C_mul_X_pow_eq p]
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    f : AlgHom R (Polynomial R) A
    ⊢ Eq (Polynomial.eval₂ (algebraMap R A) (f Polynomial.X) p) (f (p.sum fun n a  …
  -/
  simp only [eval₂_eq_sum, sum_def]
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    f : AlgHom R (Polynomial R) A
    ⊢ Eq (p.support.sum fun n => HMul.hMul ((algebraMap R A) (p.coeff n)) (HPow.hP …
  -/
  simp only [map_sum, map_mul, map_pow, eq_intCast, map_intCast]
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    f : AlgHom R (Polynomial R) A
    ⊢ Eq (p.support.sum fun n => HMul.hMul ((algebraMap R A) (p.coeff n)) (HPow.hP …
  -/
  simp [Polynomial.C_eq_algebraMap]
  /-
    🎉 no goals
  -/

-- these used to be about `algebraMap ℤ R`, but now the simp-normal form is `Int.castRingHom R`.

@[simp]
theorem ringHom_eval₂_intCastRingHom {R S : Type*} [Ring R] [Ring S] (p : ℤ[X]) (f : R →+* S)
    (r : R) : f (eval₂ (Int.castRingHom R) r p) = eval₂ (Int.castRingHom S) (f r) p :=
  algHom_eval₂_algebraMap p f.toIntAlgHom r


@[deprecated (since := "2024-05-27")]
alias ringHom_eval₂_cast_int_ringHom := ringHom_eval₂_intCastRingHom


@[simp]
theorem eval₂_intCastRingHom_X {R : Type*} [Ring R] (p : ℤ[X]) (f : ℤ[X] →+* R) :
    eval₂ (Int.castRingHom R) (f X) p = f p :=
  eval₂_algebraMap_X p f.toIntAlgHom


@[deprecated (since := "2024-04-17")]
alias eval₂_int_castRingHom_X := eval₂_intCastRingHom_X


/-- `Polynomial.eval₂` as an `AlgHom` for noncommutative algebras.

This is `Polynomial.eval₂RingHom'` for `AlgHom`s. -/
@[simps!]
def eval₂AlgHom' (f : A →ₐ[R] B) (b : B) (hf : ∀ a, Commute (f a) b) : A[X] →ₐ[R] B where
  toRingHom := eval₂RingHom' f b hf
  commutes' _ := (eval₂_C _ _).trans (f.commutes _)


/-- `Polynomial.map` as an `AlgHom` for noncommutative algebras.

  This is the algebra version of `Polynomial.mapRingHom`. -/
def mapAlgHom (f : A →ₐ[R] B) : Polynomial A →ₐ[R] Polynomial B where
  toRingHom := mapRingHom f.toRingHom
                  /-
                    R : Type u
                    S : Type v
                    T : Type w
                    A : Type z
                    A' : Type u_1
                    B : Type u_2
                    a b : R
                    n : Nat
                    inst✝⁴ : CommSemiring R
                    inst✝³ : Semiring A
                    inst✝² : Semiring B
                    inst✝¹ : Algebra R A
                    inst✝ : Algebra R B
                    p q r : Polynomial R
                    f : AlgHom R A B
                    ⊢ ∀ (r : R), Eq ((↑↑(Polynomial.mapRingHom f.toRingHom)).toFun ((algebraMap R  …
                  -/
  commutes' := by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem coe_mapAlgHom (f : A →ₐ[R] B) : ⇑(mapAlgHom f) = map f :=
  rfl


@[simp]
theorem mapAlgHom_id : mapAlgHom (AlgHom.id R A) = AlgHom.id R (Polynomial A) :=
  AlgHom.ext fun _x => map_id


@[simp]
theorem mapAlgHom_coe_ringHom (f : A →ₐ[R] B) :
    ↑(mapAlgHom f : _ →ₐ[R] Polynomial B) = (mapRingHom ↑f : Polynomial A →+* Polynomial B) :=
  rfl


@[simp]
theorem mapAlgHom_comp (C : Type z) [Semiring C] [Algebra R C] (f : B →ₐ[R] C) (g : A →ₐ[R] B) :
    (mapAlgHom f).comp (mapAlgHom g) = mapAlgHom (f.comp g) := by
  /-
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgHom R B C
    g : AlgHom R A B
    ⊢ Eq ((Polynomial.mapAlgHom f).comp (Polynomial.mapAlgHom g)) (Polynomial.mapA …
  -/
  apply AlgHom.ext
  /-
    case H
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgHom R B C
    g : AlgHom R A B
    ⊢ ∀ (x : Polynomial A), Eq (((Polynomial.mapAlgHom f).comp (Polynomial.mapAlgH …
  -/
  intro x
  /-
    case H
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgHom R B C
    g : AlgHom R A B
    x : Polynomial A
    ⊢ Eq (((Polynomial.mapAlgHom f).comp (Polynomial.mapAlgHom g)) x) ((Polynomial …
  -/
  simp [AlgHom.comp_algebraMap, map_map]
  /-
    case H
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgHom R B C
    g : AlgHom R A B
    x : Polynomial A
    ⊢ Eq (Polynomial.map ((↑f).comp ↑g) x) (Polynomial.map (↑(f.comp g)) x)
  -/
  congr
  /-
    🎉 no goals
  -/


theorem mapAlgHom_eq_eval₂AlgHom'_CAlgHom (f : A →ₐ[R] B) : mapAlgHom f = eval₂AlgHom'
    (CAlgHom.comp f) X (fun a => (commute_X (C (f a))).symm) := by
  /-
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ Eq (Polynomial.mapAlgHom f) (Polynomial.eval₂AlgHom' (Polynomial.CAlgHom.com …
  -/
  apply AlgHom.ext
  /-
    case H
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ ∀ (x : Polynomial A), Eq ((Polynomial.mapAlgHom f) x) ((Polynomial.eval₂AlgH …
  -/
  intro x
  /-
    case H
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    f : AlgHom R A B
    x : Polynomial A
    ⊢ Eq ((Polynomial.mapAlgHom f) x) ((Polynomial.eval₂AlgHom' (Polynomial.CAlgHo …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- If `A` and `B` are isomorphic as `R`-algebras, then so are their polynomial rings -/
def mapAlgEquiv (f : A ≃ₐ[R] B) : Polynomial A ≃ₐ[R] Polynomial B :=
                                                                           /-
                                                                             R : Type u
                                                                             S : Type v
                                                                             T : Type w
                                                                             A : Type z
                                                                             A' : Type u_1
                                                                             B : Type u_2
                                                                             a b : R
                                                                             n : Nat
                                                                             inst✝⁴ : CommSemiring R
                                                                             inst✝³ : Semiring A
                                                                             inst✝² : Semiring B
                                                                             inst✝¹ : Algebra R A
                                                                             inst✝ : Algebra R B
                                                                             p q r : Polynomial R
                                                                             f : AlgEquiv R A B
                                                                             ⊢ Eq ((Polynomial.mapAlgHom ↑f).comp (Polynomial.mapAlgHom ↑f.symm)) (AlgHom.i …
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  AlgEquiv.ofAlgHom (mapAlgHom f.toAlgHom) (mapAlgHom f.symm.toAlgHom) (by simp) (by simp)
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem coe_mapAlgEquiv (f : A ≃ₐ[R] B) : ⇑(mapAlgEquiv f) = map f :=
  rfl


@[simp]
theorem mapAlgEquiv_id : mapAlgEquiv (@AlgEquiv.refl R A _ _ _) = AlgEquiv.refl :=
  AlgEquiv.ext fun _x => map_id


@[simp]
theorem mapAlgEquiv_coe_ringHom (f : A ≃ₐ[R] B) :
    ↑(mapAlgEquiv f : _ ≃ₐ[R] Polynomial B) = (mapRingHom ↑f : Polynomial A →+* Polynomial B) :=
  rfl


@[simp]
theorem mapAlgEquiv_comp (C : Type z) [Semiring C] [Algebra R C] (f : A ≃ₐ[R] B) (g : B ≃ₐ[R] C) :
    (mapAlgEquiv f).trans (mapAlgEquiv g) = mapAlgEquiv (f.trans g) := by
  /-
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgEquiv R A B
    g : AlgEquiv R B C
    ⊢ Eq ((Polynomial.mapAlgEquiv f).trans (Polynomial.mapAlgEquiv g)) (Polynomial …
  -/
  apply AlgEquiv.ext
  /-
    case h
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgEquiv R A B
    g : AlgEquiv R B C
    ⊢ ∀ (a : Polynomial A), Eq (((Polynomial.mapAlgEquiv f).trans (Polynomial.mapA …
  -/
  intro x
  /-
    case h
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgEquiv R A B
    g : AlgEquiv R B C
    x : Polynomial A
    ⊢ Eq (((Polynomial.mapAlgEquiv f).trans (Polynomial.mapAlgEquiv g)) x) ((Polyn …
  -/
  simp [AlgEquiv.trans_apply, map_map]
  /-
    case h
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    C : Type z
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : AlgEquiv R A B
    g : AlgEquiv R B C
    x : Polynomial A
    ⊢ Eq (Polynomial.map ((↑g).comp ↑f) x) (Polynomial.map (↑(f.trans g)) x)
  -/
  congr
  /-
    🎉 no goals
  -/


/-- Given a valuation `x` of the variable in an `R`-algebra `A`, `aeval R A x` is
the unique `R`-algebra homomorphism from `R[X]` to `A` sending `X` to `x`.

This is a stronger variant of the linear map `Polynomial.leval`. -/
def aeval : R[X] →ₐ[R] A :=
  eval₂AlgHom' (Algebra.ofId _ _) x (Algebra.commutes · _)


@[ext 1200]
theorem algHom_ext {f g : R[X] →ₐ[R] B} (hX : f X = g X) :
    f = g :=
  algHom_ext' (Subsingleton.elim _ _) hX


theorem aeval_def (p : R[X]) : aeval x p = eval₂ (algebraMap R A) x p :=
  rfl

-- Porting note: removed `@[simp]` because `simp` can prove this

theorem aeval_zero : aeval x (0 : R[X]) = 0 :=
  map_zero (aeval x)


@[simp]
theorem aeval_X : aeval x (X : R[X]) = x :=
  eval₂_X _ x


@[simp]
theorem aeval_C (r : R) : aeval x (C r) = algebraMap R A r :=
  eval₂_C _ x


@[simp]
theorem aeval_monomial {n : ℕ} {r : R} : aeval x (monomial n r) = algebraMap _ _ r * x ^ n :=
  eval₂_monomial _ _

-- Porting note: removed `@[simp]` because `simp` can prove this

theorem aeval_X_pow {n : ℕ} : aeval x ((X : R[X]) ^ n) = x ^ n :=
  eval₂_X_pow _ _

-- Porting note: removed `@[simp]` because `simp` can prove this

theorem aeval_add : aeval x (p + q) = aeval x p + aeval x q :=
  map_add _ _ _

-- Porting note: removed `@[simp]` because `simp` can prove this

theorem aeval_one : aeval x (1 : R[X]) = 1 :=
  map_one _

-- Porting note: removed `@[simp]` because `simp` can prove this

theorem aeval_natCast (n : ℕ) : aeval x (n : R[X]) = n :=
  map_natCast _ _


@[deprecated (since := "2024-04-17")]
alias aeval_nat_cast := aeval_natCast


theorem aeval_mul : aeval x (p * q) = aeval x p * aeval x q :=
  map_mul _ _ _


theorem comp_eq_aeval : p.comp q = aeval q p := rfl


theorem aeval_comp {A : Type*} [Semiring A] [Algebra R A] (x : A) :
    aeval x (p.comp q) = aeval (aeval x q) p :=
  eval₂_comp' x p q


/-- Two polynomials `p` and `q` such that `p(q(X))=X` and `q(p(X))=X`
  induces an automorphism of the polynomial algebra. -/
@[simps!]
def algEquivOfCompEqX (p q : R[X]) (hpq : p.comp q = X) (hqp : q.comp p = X) : R[X] ≃ₐ[R] R[X] := by
  /-
    R : Type u
    S : Type v
    T : Type w
    A : Type z
    A' : Type u_1
    B : Type u_2
    a b : R
    n : Nat
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : CommSemiring A'
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    p✝ q✝ : Polynomial R
    x : A
    p q : Polynomial R
    hpq : Eq (p.comp q) Polynomial.X
    hqp : Eq (q.comp p) Polynomial.X
    ⊢ AlgEquiv R (Polynomial R) (Polynomial R)
  -/
  refine AlgEquiv.ofAlgHom (aeval p) (aeval q) ?_ ?_ <;>
    /-
      case refine_1
      R : Type u
      S : Type v
      T : Type w
      A : Type z
      A' : Type u_1
      B : Type u_2
      a b : R
      n : Nat
      inst✝⁵ : CommSemiring R
      inst✝⁴ : Semiring A
      inst✝³ : CommSemiring A'
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      p✝ q✝ : Polynomial R
      x : A
      p q : Polynomial R
      hpq : Eq (p.comp q) Polynomial.X
      hqp : Eq (q.comp p) Polynomial.X
      ⊢ Eq ((Polynomial.aeval p).comp (Polynomial.aeval q)) (AlgHom.id R (Polynomial …
    -/
    /-
      🎉 no goals
    -/
    exact AlgHom.ext fun _ ↦ by simp [← comp_eq_aeval, comp_assoc, hpq, hqp]
    /-
      🎉 no goals
    -/


@[simp]
theorem algEquivOfCompEqX_eq_iff (p q p' q' : R[X])
    (hpq : p.comp q = X) (hqp : q.comp p = X) (hpq' : p'.comp q' = X) (hqp' : q'.comp p' = X) :
    algEquivOfCompEqX p q hpq hqp = algEquivOfCompEqX p' q' hpq' hqp' ↔ p = p' :=
              /-
                R : Type u
                inst✝ : CommSemiring R
                p q p' q' : Polynomial R
                hpq : Eq (p.comp q) Polynomial.X
                hqp : Eq (q.comp p) Polynomial.X
                hpq' : Eq (p'.comp q') Polynomial.X
                hqp' : Eq (q'.comp p') Polynomial.X
                h : Eq (p.algEquivOfCompEqX q hpq hqp) (p'.algEquivOfCompEqX q' hpq' hqp')
                ⊢ Eq p p'
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by simpa using congr($h X), fun h ↦ by ext1; simp [h]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem algEquivOfCompEqX_symm (p q : R[X]) (hpq : p.comp q = X) (hqp : q.comp p = X) :
    (algEquivOfCompEqX p q hpq hqp).symm = algEquivOfCompEqX q p hqp hpq := rfl


/-- The automorphism of the polynomial algebra given by `p(X) ↦ p(a * X + b)`,
  with inverse `p(X) ↦ p(a⁻¹ * (X - b))`. -/
@[simps!]
def algEquivCMulXAddC {R : Type*} [CommRing R] (a b : R) [Invertible a] : R[X] ≃ₐ[R] R[X] :=
  algEquivOfCompEqX (C a * X + C b) (C ⅟ a * (X - C b))
        /-
          R✝ : Type u
          S : Type v
          T : Type w
          A : Type z
          A' : Type u_1
          B : Type u_2
          a✝ b✝ : R✝
          n : Nat
          inst✝⁷ : CommSemiring R✝
          inst✝⁶ : Semiring A
          inst✝⁵ : CommSemiring A'
          inst✝⁴ : Semiring B
          inst✝³ : Algebra R✝ A
          inst✝² : Algebra R✝ B
          p q : Polynomial R✝
          x : A
          R : Type u_3
          inst✝¹ : CommRing R
          a b : R
          inst✝ : Invertible a
          ⊢ Eq ((HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).c …
        -/
        /-
          🎉 no goals
        -/
    (by simp [← C_mul, ← mul_assoc]) (by simp [← C_mul, ← mul_assoc])
                                         /-
                                           🎉 no goals
                                         -/


theorem algEquivCMulXAddC_symm_eq {R : Type*} [CommRing R] (a b : R) [Invertible a] :
    (algEquivCMulXAddC a b).symm =  algEquivCMulXAddC (⅟ a) (- ⅟ a * b) := by
  /-
    R : Type u_3
    inst✝¹ : CommRing R
    a b : R
    inst✝ : Invertible a
    ⊢ Eq (Polynomial.algEquivCMulXAddC a b).symm (Polynomial.algEquivCMulXAddC (In …
  -/
  ext p : 1
  /-
    case h
    R : Type u_3
    inst✝¹ : CommRing R
    a b : R
    inst✝ : Invertible a
    p : Polynomial R
    ⊢ Eq ((Polynomial.algEquivCMulXAddC a b).symm p) ((Polynomial.algEquivCMulXAdd …
  -/
  simp only [algEquivCMulXAddC_symm_apply, neg_mul, algEquivCMulXAddC_apply, map_neg, map_mul]
  /-
    case h
    R : Type u_3
    inst✝¹ : CommRing R
    a b : R
    inst✝ : Invertible a
    p : Polynomial R
    ⊢ Eq ((Polynomial.aeval (HMul.hMul (Polynomial.C (Invertible.invOf a)) (HSub.h …
  -/
  congr
  /-
    case h.e_a.e_x
    R : Type u_3
    inst✝¹ : CommRing R
    a b : R
    inst✝ : Invertible a
    p : Polynomial R
    ⊢ Eq (HMul.hMul (Polynomial.C (Invertible.invOf a)) (HSub.hSub Polynomial.X (P …
  -/
  simp [mul_add, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- The automorphism of the polynomial algebra given by `p(X) ↦ p(X+t)`,
  with inverse `p(X) ↦ p(X-t)`. -/
@[simps!]
def algEquivAevalXAddC {R : Type*} [CommRing R] (t : R) : R[X] ≃ₐ[R] R[X] :=
                                            /-
                                              R✝ : Type u
                                              S : Type v
                                              T : Type w
                                              A : Type z
                                              A' : Type u_1
                                              B : Type u_2
                                              a b : R✝
                                              n : Nat
                                              inst✝⁶ : CommSemiring R✝
                                              inst✝⁵ : Semiring A
                                              inst✝⁴ : CommSemiring A'
                                              inst✝³ : Semiring B
                                              inst✝² : Algebra R✝ A
                                              inst✝¹ : Algebra R✝ B
                                              p q : Polynomial R✝
                                              x : A
                                              R : Type u_3
                                              inst✝ : CommRing R
                                              t : R
                                              ⊢ Eq ((HAdd.hAdd Polynomial.X (Polynomial.C t)).comp (HSub.hSub Polynomial.X ( …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  algEquivOfCompEqX (X + C t) (X - C t) (by simp) (by simp)
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem algEquivAevalXAddC_eq_iff {R : Type*} [CommRing R] (t t' : R) :
    algEquivAevalXAddC t = algEquivAevalXAddC t' ↔ t = t' := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    t t' : R
    ⊢ Iff (Eq (Polynomial.algEquivAevalXAddC t) (Polynomial.algEquivAevalXAddC t') …
  -/
  simp [algEquivAevalXAddC]
  /-
    🎉 no goals
  -/


@[simp]
theorem algEquivAevalXAddC_symm {R : Type*} [CommRing R] (t : R) :
    (algEquivAevalXAddC t).symm = algEquivAevalXAddC (-t) := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    t : R
    ⊢ Eq (Polynomial.algEquivAevalXAddC t).symm (Polynomial.algEquivAevalXAddC (Ne …
  -/
  simp [algEquivAevalXAddC, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- The involutive automorphism of the polynomial algebra given by `p(X) ↦ p(-X)`. -/
@[simps!]
def algEquivAevalNegX {R : Type*} [CommRing R] : R[X] ≃ₐ[R] R[X] :=
                                  /-
                                    R✝ : Type u
                                    S : Type v
                                    T : Type w
                                    A : Type z
                                    A' : Type u_1
                                    B : Type u_2
                                    a b : R✝
                                    n : Nat
                                    inst✝⁶ : CommSemiring R✝
                                    inst✝⁵ : Semiring A
                                    inst✝⁴ : CommSemiring A'
                                    inst✝³ : Semiring B
                                    inst✝² : Algebra R✝ A
                                    inst✝¹ : Algebra R✝ B
                                    p q : Polynomial R✝
                                    x : A
                                    R : Type u_3
                                    inst✝ : CommRing R
                                    ⊢ Eq ((Neg.neg Polynomial.X).comp (Neg.neg Polynomial.X)) Polynomial.X
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  algEquivOfCompEqX (-X) (-X) (by simp) (by simp)
                                            /-
                                              🎉 no goals
                                            -/


theorem comp_neg_X_comp_neg_X {R : Type*} [CommRing R] (p : R[X]) :
    (p.comp (-X)).comp (-X) = p := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    p : Polynomial R
    ⊢ Eq ((p.comp (Neg.neg Polynomial.X)).comp (Neg.neg Polynomial.X)) p
  -/
  rw [comp_assoc]
  /-
    R : Type u_3
    inst✝ : CommRing R
    p : Polynomial R
    ⊢ Eq (p.comp ((Neg.neg Polynomial.X).comp (Neg.neg Polynomial.X))) p
  -/
  simp only [neg_comp, X_comp, neg_neg, comp_X]
  /-
    🎉 no goals
  -/


theorem aeval_algHom (f : A →ₐ[R] B) (x : A) : aeval (f x) = f.comp (aeval x) :=
                   /-
                     R : Type u
                     A : Type z
                     B : Type u_2
                     inst✝⁴ : CommSemiring R
                     inst✝³ : Semiring A
                     inst✝² : Semiring B
                     inst✝¹ : Algebra R A
                     inst✝ : Algebra R B
                     f : AlgHom R A B
                     x : A
                     ⊢ Eq ((Polynomial.aeval (f x)) Polynomial.X) ((f.comp (Polynomial.aeval x)) Po …
                   -/
  algHom_ext <| by simp only [aeval_X, AlgHom.comp_apply]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem aeval_X_left : aeval (X : R[X]) = AlgHom.id R R[X] :=
  algHom_ext <| aeval_X X


theorem aeval_X_left_apply (p : R[X]) : aeval X p = p :=
  AlgHom.congr_fun (@aeval_X_left R _) p


theorem eval_unique (φ : R[X] →ₐ[R] A) (p) : φ p = eval₂ (algebraMap R A) (φ X) p := by
  /-
    R : Type u
    A : Type z
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    φ : AlgHom R (Polynomial R) A
    p : Polynomial R
    ⊢ Eq (φ p) (Polynomial.eval₂ (algebraMap R A) (φ Polynomial.X) p)
  -/
  rw [← aeval_def, aeval_algHom, aeval_X_left, AlgHom.comp_id]
  /-
    🎉 no goals
  -/


theorem aeval_algHom_apply {F : Type*} [FunLike F A B] [AlgHomClass F R A B]
    (f : F) (x : A) (p : R[X]) :
    aeval (f x) p = f (aeval x p) := by
  refine Polynomial.induction_on p (by simp [AlgHomClass.commutes]) (fun p q hp hq => ?_)
    (by simp [AlgHomClass.commutes])
  /-
    R : Type u
    A : Type z
    B : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    F : Type u_3
    inst✝¹ : FunLike F A B
    inst✝ : AlgHomClass F R A B
    f : F
    x : A
    p✝ p q : Polynomial R
    hp : Eq ((Polynomial.aeval (f x)) p) (f ((Polynomial.aeval x) p))
    hq : Eq ((Polynomial.aeval (f x)) q) (f ((Polynomial.aeval x) q))
    ⊢ Eq ((Polynomial.aeval (f x)) (HAdd.hAdd p q)) (f ((Polynomial.aeval x) (HAdd …
  -/
  rw [map_add, hp, hq, ← map_add, ← map_add]
  /-
    🎉 no goals
  -/


@[simp]
lemma coe_aeval_mk_apply {S : Subalgebra R A} (h : x ∈ S) :
    (aeval (⟨x, h⟩ : S) p : A) = aeval x p :=
  (aeval_algHom_apply S.val (⟨x, h⟩ : S) p).symm


theorem aeval_algEquiv (f : A ≃ₐ[R] B) (x : A) : aeval (f x) = (f : A →ₐ[R] B).comp (aeval x) :=
  aeval_algHom (f : A →ₐ[R] B) x


theorem aeval_algebraMap_apply_eq_algebraMap_eval (x : R) (p : R[X]) :
    aeval (algebraMap R A x) p = algebraMap R A (p.eval x) :=
  aeval_algHom_apply (Algebra.ofId R A) x p


@[simp]
theorem coe_aeval_eq_eval (r : R) : (aeval r : R[X] → R) = eval r :=
  rfl


@[simp]
theorem coe_aeval_eq_evalRingHom (x : R) :
    ((aeval x : R[X] →ₐ[R] R) : R[X] →+* R) = evalRingHom x :=
  rfl


@[simp]
theorem aeval_fn_apply {X : Type*} (g : R[X]) (f : X → R) (x : X) :
    ((aeval f) g) x = aeval (f x) g :=
  (aeval_algHom_apply (Pi.evalAlgHom R (fun _ => R) x) f g).symm


@[norm_cast]
theorem aeval_subalgebra_coe (g : R[X]) {A : Type*} [Semiring A] [Algebra R A] (s : Subalgebra R A)
    (f : s) : (aeval f g : A) = aeval (f : A) g :=
  (aeval_algHom_apply s.val f g).symm


theorem coeff_zero_eq_aeval_zero (p : R[X]) : p.coeff 0 = aeval 0 p := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Polynomial R
    ⊢ Eq (p.coeff 0) ((Polynomial.aeval 0) p)
  -/
  simp [coeff_zero_eq_eval_zero]
  /-
    🎉 no goals
  -/


theorem coeff_zero_eq_aeval_zero' (p : R[X]) : algebraMap R A (p.coeff 0) = aeval (0 : A) p := by
  /-
    R : Type u
    A : Type z
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    ⊢ Eq ((algebraMap R A) (p.coeff 0)) ((Polynomial.aeval 0) p)
  -/
  simp [aeval_def]
  /-
    🎉 no goals
  -/


theorem map_aeval_eq_aeval_map {S T U : Type*} [Semiring S] [CommSemiring T] [Semiring U]
    [Algebra R S] [Algebra T U] {φ : R →+* T} {ψ : S →+* U}
    (h : (algebraMap T U).comp φ = ψ.comp (algebraMap R S)) (p : R[X]) (a : S) :
    ψ (aeval a p) = aeval (ψ a) (p.map φ) := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Type u_3
    T : Type u_4
    U : Type u_5
    inst✝⁴ : Semiring S
    inst✝³ : CommSemiring T
    inst✝² : Semiring U
    inst✝¹ : Algebra R S
    inst✝ : Algebra T U
    φ : RingHom R T
    ψ : RingHom S U
    h : Eq ((algebraMap T U).comp φ) (ψ.comp (algebraMap R S))
    p : Polynomial R
    a : S
    ⊢ Eq (ψ ((Polynomial.aeval a) p)) ((Polynomial.aeval (ψ a)) (Polynomial.map φ  …
  -/
  conv_rhs => rw [aeval_def, ← eval_map]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    S : Type u_3
    T : Type u_4
    U : Type u_5
    inst✝⁴ : Semiring S
    inst✝³ : CommSemiring T
    inst✝² : Semiring U
    inst✝¹ : Algebra R S
    inst✝ : Algebra T U
    φ : RingHom R T
    ψ : RingHom S U
    h : Eq ((algebraMap T U).comp φ) (ψ.comp (algebraMap R S))
    p : Polynomial R
    a : S
    ⊢ Eq (ψ ((Polynomial.aeval a) p)) (Polynomial.eval (ψ a) (Polynomial.map (alge …
  -/
  rw [map_map, h, ← map_map, eval_map, eval₂_at_apply, aeval_def, eval_map]
  /-
    🎉 no goals
  -/


theorem aeval_eq_zero_of_dvd_aeval_eq_zero [CommSemiring S] [CommSemiring T] [Algebra S T]
    {p q : S[X]} (h₁ : p ∣ q) {a : T} (h₂ : aeval a p = 0) : aeval a q = 0 := by
  /-
    S : Type v
    T : Type w
    inst✝² : CommSemiring S
    inst✝¹ : CommSemiring T
    inst✝ : Algebra S T
    p q : Polynomial S
    h₁ : Dvd.dvd p q
    a : T
    h₂ : Eq ((Polynomial.aeval a) p) 0
    ⊢ Eq ((Polynomial.aeval a) q) 0
  -/
  rw [aeval_def, ← eval_map] at h₂ ⊢
  /-
    S : Type v
    T : Type w
    inst✝² : CommSemiring S
    inst✝¹ : CommSemiring T
    inst✝ : Algebra S T
    p q : Polynomial S
    h₁ : Dvd.dvd p q
    a : T
    h₂ : Eq (Polynomial.eval a (Polynomial.map (algebraMap S T) p)) 0
    ⊢ Eq (Polynomial.eval a (Polynomial.map (algebraMap S T) q)) 0
  -/
  exact eval_eq_zero_of_dvd_of_eval_eq_zero (Polynomial.map_dvd (algebraMap S T) h₁) h₂
  /-
    🎉 no goals
  -/


theorem aeval_eq_sum_range [Algebra R S] {p : R[X]} (x : S) :
    aeval x p = ∑ i ∈ Finset.range (p.natDegree + 1), p.coeff i • x ^ i := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial R
    x : S
    ⊢ Eq ((Polynomial.aeval x) p) ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fu …
  -/
  simp_rw [Algebra.smul_def]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial R
    x : S
    ⊢ Eq ((Polynomial.aeval x) p) ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fu …
  -/
  exact eval₂_eq_sum_range (algebraMap R S) x
  /-
    🎉 no goals
  -/


theorem aeval_eq_sum_range' [Algebra R S] {p : R[X]} {n : ℕ} (hn : p.natDegree < n) (x : S) :
    aeval x p = ∑ i ∈ Finset.range n, p.coeff i • x ^ i := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    x : S
    ⊢ Eq ((Polynomial.aeval x) p) ((Finset.range n).sum fun i => HSMul.hSMul (p.co …
  -/
  simp_rw [Algebra.smul_def]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    x : S
    ⊢ Eq ((Polynomial.aeval x) p) ((Finset.range n).sum fun x_1 => HMul.hMul ((alg …
  -/
  exact eval₂_eq_sum_range' (algebraMap R S) hn x
  /-
    🎉 no goals
  -/


theorem isRoot_of_eval₂_map_eq_zero (hf : Function.Injective f) {r : R} :
    eval₂ f (f r) p = 0 → p.IsRoot r := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    r : R
    ⊢ Eq (Polynomial.eval₂ f (f r) p) 0 → p.IsRoot r
  -/
  intro h
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    r : R
    h : Eq (Polynomial.eval₂ f (f r) p) 0
    ⊢ p.IsRoot r
  -/
  apply hf
  /-
    case a
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    r : R
    h : Eq (Polynomial.eval₂ f (f r) p) 0
    ⊢ Eq (f (Polynomial.eval r p)) (f 0)
  -/
  rw [← eval₂_hom, h, f.map_zero]
  /-
    🎉 no goals
  -/


theorem isRoot_of_aeval_algebraMap_eq_zero [Algebra R S] {p : R[X]}
    (inj : Function.Injective (algebraMap R S)) {r : R} (hr : aeval (algebraMap R S r) p = 0) :
    p.IsRoot r :=
  isRoot_of_eval₂_map_eq_zero inj hr


/-- Version of `aeval` for defining algebra homs out of `R[X]` over a smaller base ring
  than `R`. -/
def aevalTower (f : R →ₐ[S] A') (x : A') : R[X] →ₐ[S] A' :=
  eval₂AlgHom' f x fun _ => Commute.all _ _


@[simp]
theorem aevalTower_X : aevalTower g y X = y :=
  eval₂_X _ _


@[simp]
theorem aevalTower_C (x : R) : aevalTower g y (C x) = g x :=
  eval₂_C _ _


@[simp]
theorem aevalTower_comp_C : (aevalTower g y : R[X] →+* A').comp C = g :=
  RingHom.ext <| aevalTower_C _ _


theorem aevalTower_algebraMap (x : R) : aevalTower g y (algebraMap R R[X] x) = g x :=
  eval₂_C _ _


theorem aevalTower_comp_algebraMap : (aevalTower g y : R[X] →+* A').comp (algebraMap R R[X]) = g :=
  aevalTower_comp_C _ _


theorem aevalTower_toAlgHom (x : R) : aevalTower g y (IsScalarTower.toAlgHom S R R[X] x) = g x :=
  aevalTower_algebraMap _ _ _


@[simp]
theorem aevalTower_comp_toAlgHom : (aevalTower g y).comp (IsScalarTower.toAlgHom S R R[X]) = g :=
  AlgHom.coe_ringHom_injective <| aevalTower_comp_algebraMap _ _


@[simp]
theorem aevalTower_id : aevalTower (AlgHom.id S S) = aeval := by
  /-
    S : Type v
    inst✝ : CommSemiring S
    ⊢ Eq (Polynomial.aevalTower (AlgHom.id S S)) Polynomial.aeval
  -/
  ext s
  /-
    case h.hX
    S : Type v
    inst✝ : CommSemiring S
    s : S
    ⊢ Eq ((Polynomial.aevalTower (AlgHom.id S S) s) Polynomial.X) ((Polynomial.aev …
  -/
  simp only [eval_X, aevalTower_X, coe_aeval_eq_eval]
  /-
    🎉 no goals
  -/


@[simp]
theorem aevalTower_ofId : aevalTower (Algebra.ofId S A') = aeval := by
  /-
    S : Type v
    A' : Type u_1
    inst✝² : CommSemiring A'
    inst✝¹ : CommSemiring S
    inst✝ : Algebra S A'
    ⊢ Eq (Polynomial.aevalTower (Algebra.ofId S A')) Polynomial.aeval
  -/
  ext
  /-
    case h.hX
    S : Type v
    A' : Type u_1
    inst✝² : CommSemiring A'
    inst✝¹ : CommSemiring S
    inst✝ : Algebra S A'
    x✝ : A'
    ⊢ Eq ((Polynomial.aevalTower (Algebra.ofId S A') x✝) Polynomial.X) ((Polynomia …
  -/
  simp only [aeval_X, aevalTower_X]
  /-
    🎉 no goals
  -/


theorem dvd_term_of_dvd_eval_of_dvd_terms {z p : S} {f : S[X]} (i : ℕ) (dvd_eval : p ∣ f.eval z)
    (dvd_terms : ∀ j ≠ i, p ∣ f.coeff j * z ^ j) : p ∣ f.coeff i * z ^ i := by
  /-
    S : Type v
    inst✝ : CommRing S
    z p : S
    f : Polynomial S
    i : Nat
    dvd_eval : Dvd.dvd p (Polynomial.eval z f)
    dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
    ⊢ Dvd.dvd p (HMul.hMul (f.coeff i) (HPow.hPow z i))
  -/
  by_cases hi : i ∈ f.support
    /-
      case pos
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (Polynomial.eval z f)
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Membership.mem f.support i
      ⊢ Dvd.dvd p (HMul.hMul (f.coeff i) (HPow.hPow z i))
    -/
  · rw [eval, eval₂_eq_sum, sum_def] at dvd_eval
    /-
      case pos
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (f.support.sum fun n => HMul.hMul ((RingHom.id S) (f.coef …
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Membership.mem f.support i
      ⊢ Dvd.dvd p (HMul.hMul (f.coeff i) (HPow.hPow z i))
    -/
    rw [← Finset.insert_erase hi, Finset.sum_insert (Finset.not_mem_erase _ _)] at dvd_eval
    /-
      case pos
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (HAdd.hAdd (HMul.hMul ((RingHom.id S) (f.coeff i)) (HPow. …
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Membership.mem f.support i
      ⊢ Dvd.dvd p (HMul.hMul (f.coeff i) (HPow.hPow z i))
    -/
    refine (dvd_add_left ?_).mp dvd_eval
    /-
      case pos
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (HAdd.hAdd (HMul.hMul ((RingHom.id S) (f.coeff i)) (HPow. …
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Membership.mem f.support i
      ⊢ Dvd.dvd p ((f.support.erase i).sum fun x => HMul.hMul ((RingHom.id S) (f.coe …
    -/
    apply Finset.dvd_sum
    /-
      case pos.h
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (HAdd.hAdd (HMul.hMul ((RingHom.id S) (f.coeff i)) (HPow. …
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Membership.mem f.support i
      ⊢ ∀ (i_1 : Nat), Membership.mem (f.support.erase i) i_1 → Dvd.dvd p (HMul.hMul …
    -/
    intro j hj
    /-
      case pos.h
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (HAdd.hAdd (HMul.hMul ((RingHom.id S) (f.coeff i)) (HPow. …
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Membership.mem f.support i
      j : Nat
      hj : Membership.mem (f.support.erase i) j
      ⊢ Dvd.dvd p (HMul.hMul ((RingHom.id S) (f.coeff j)) (HPow.hPow z j))
    -/
    exact dvd_terms j (Finset.ne_of_mem_erase hj)
    /-
      🎉 no goals
    -/
    /-
      case neg
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (Polynomial.eval z f)
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Not (Membership.mem f.support i)
      ⊢ Dvd.dvd p (HMul.hMul (f.coeff i) (HPow.hPow z i))
    -/
  · convert dvd_zero p
    /-
      case h.e'_4
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (Polynomial.eval z f)
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Not (Membership.mem f.support i)
      ⊢ Eq (HMul.hMul (f.coeff i) (HPow.hPow z i)) 0
    -/
    rw [not_mem_support_iff] at hi
    /-
      case h.e'_4
      S : Type v
      inst✝ : CommRing S
      z p : S
      f : Polynomial S
      i : Nat
      dvd_eval : Dvd.dvd p (Polynomial.eval z f)
      dvd_terms : ∀ (j : Nat), Ne j i → Dvd.dvd p (HMul.hMul (f.coeff j) (HPow.hPow  …
      hi : Eq (f.coeff i) 0
      ⊢ Eq (HMul.hMul (f.coeff i) (HPow.hPow z i)) 0
    -/
    simp [hi]
    /-
      🎉 no goals
    -/


theorem dvd_term_of_isRoot_of_dvd_terms {r p : S} {f : S[X]} (i : ℕ) (hr : f.IsRoot r)
    (h : ∀ j ≠ i, p ∣ f.coeff j * r ^ j) : p ∣ f.coeff i * r ^ i :=
  dvd_term_of_dvd_eval_of_dvd_terms i (Eq.symm hr ▸ dvd_zero p) h


/-- The evaluation map is not generally multiplicative when the coefficient ring is noncommutative,
but nevertheless any polynomial of the form `p * (X - monomial 0 r)` is sent to zero
when evaluated at `r`.

This is the key step in our proof of the Cayley-Hamilton theorem.
-/
theorem eval_mul_X_sub_C {p : R[X]} (r : R) : (p * (X - C r)).eval r = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    r : R
    ⊢ Eq (Polynomial.eval r (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r)) …
  -/
  simp only [eval, eval₂_eq_sum, RingHom.id_apply]
  have bound :=
    calc
      (p * (X - C r)).natDegree ≤ p.natDegree + (X - C r).natDegree := natDegree_mul_le
      _ ≤ p.natDegree + 1 := add_le_add_left (natDegree_X_sub_C_le _) _
      _ < p.natDegree + 2 := lt_add_one _
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    r : R
    bound : LT.lt (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).natDegre …
    ⊢ Eq ((HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).sum fun e a => H …
  -/
  rw [sum_over_range' _ _ (p.natDegree + 2) bound]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    r : R
    bound : LT.lt (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).natDegre …
    ⊢ Eq ((Finset.range (HAdd.hAdd p.natDegree 2)).sum fun a => HMul.hMul ((HMul.h …
  -/
  swap
    /-
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      r : R
      bound : LT.lt (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).natDegre …
      ⊢ ∀ (n : Nat), Eq (HMul.hMul 0 (HPow.hPow r n)) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    r : R
    bound : LT.lt (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).natDegre …
    ⊢ Eq ((Finset.range (HAdd.hAdd p.natDegree 2)).sum fun a => HMul.hMul ((HMul.h …
  -/
  rw [sum_range_succ']
  conv_lhs =>
    congr
    arg 2
    simp [coeff_mul_X_sub_C, sub_mul, mul_assoc, ← pow_succ']
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    r : R
    bound : LT.lt (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).natDegre …
    ⊢ Eq (HAdd.hAdd ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fun k => HSub.hS …
  -/
  rw [sum_range_sub']
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    r : R
    bound : LT.lt (HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).natDegre …
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (p.coeff 0) (HPow.hPow r (HAdd.hAdd 0 1) …
  -/
  simp [coeff_monomial]
  /-
    🎉 no goals
  -/


theorem not_isUnit_X_sub_C [Nontrivial R] (r : R) : ¬IsUnit (X - C r) :=
                                                       /-
                                                         R : Type u
                                                         inst✝¹ : Ring R
                                                         inst✝ : Nontrivial R
                                                         r : R
                                                         x✝ : IsUnit (HSub.hSub Polynomial.X (Polynomial.C r))
                                                         g : Polynomial R
                                                         _hfg : Eq (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C r)) g) 1
                                                         hgf : Eq (HMul.hMul g (HSub.hSub Polynomial.X (Polynomial.C r))) 1
                                                         ⊢ Eq 0 1
                                                       -/
  fun ⟨⟨_, g, _hfg, hgf⟩, rfl⟩ => zero_ne_one' R <| by rw [← eval_mul_X_sub_C, hgf, eval_one]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem aeval_endomorphism {M : Type*} [AddCommGroup M] [Module R M] (f : M →ₗ[R] M)
    (v : M) (p : R[X]) : aeval f p v = p.sum fun n b => b • (f ^ n) v := by
  /-
    R : Type u
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    v : M
    p : Polynomial R
    ⊢ Eq (((Polynomial.aeval f) p) v) (p.sum fun n b => HSMul.hSMul b ((HPow.hPow  …
  -/
  rw [aeval_def, eval₂_eq_sum]
  /-
    R : Type u
    inst✝² : CommRing R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    v : M
    p : Polynomial R
    ⊢ Eq ((p.sum fun e a => HMul.hMul ((algebraMap R (LinearMap (RingHom.id R) M M …
  -/
  exact map_sum (LinearMap.applyₗ v) _ _
  /-
    🎉 no goals
  -/


lemma X_sub_C_pow_dvd_iff {n : ℕ} : (X - C t) ^ n ∣ p ↔ X ^ n ∣ p.comp (X + C t) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    ⊢ Iff (Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) n) p) (Dvd …
  -/
  convert (map_dvd_iff <| algEquivAevalXAddC t).symm using 2
  /-
    case h.e'_2.h.e'_3
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n) ((Polynomial.algEquivAevalXAddC t) (HPow.hPow  …
  -/
  simp [C_eq_algebraMap]
  /-
    🎉 no goals
  -/


lemma comp_X_add_C_eq_zero_iff : p.comp (X + C t) = 0 ↔ p = 0 :=
  EmbeddingLike.map_eq_zero_iff (f := algEquivAevalXAddC t)


lemma comp_X_add_C_ne_zero_iff : p.comp (X + C t) ≠ 0 ↔ p ≠ 0 := comp_X_add_C_eq_zero_iff.not


lemma dvd_comp_C_mul_X_add_C_iff (p q : R[X]) (a b : R) [Invertible a] :
    p ∣ q.comp (C a * X + C b) ↔ p.comp (C ⅟ a * (X - C b)) ∣ q := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    p q : Polynomial R
    a b : R
    inst✝ : Invertible a
    ⊢ Iff (Dvd.dvd p (q.comp (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X)  …
  -/
  convert map_dvd_iff <| algEquivCMulXAddC a b using 2
  /-
    case h.e'_1.h.e'_3
    R : Type u
    inst✝¹ : CommRing R
    p q : Polynomial R
    a b : R
    inst✝ : Invertible a
    ⊢ Eq p ((Polynomial.algEquivCMulXAddC a b) (p.comp (HMul.hMul (Polynomial.C (I …
  -/
  simp [← comp_eq_aeval, comp_assoc, ← mul_assoc, ← C_mul]
  /-
    🎉 no goals
  -/


lemma dvd_comp_X_sub_C_iff (p q : R[X]) (a : R) :
    p ∣ q.comp (X - C a) ↔ p.comp (X + C a) ∣ q := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    a : R
    ⊢ Iff (Dvd.dvd p (q.comp (HSub.hSub Polynomial.X (Polynomial.C a)))) (Dvd.dvd  …
  -/
  let _ := invertibleOne (α := R)
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    a : R
    x✝ : Invertible 1 := invertibleOne
    ⊢ Iff (Dvd.dvd p (q.comp (HSub.hSub Polynomial.X (Polynomial.C a)))) (Dvd.dvd  …
  -/
  simpa using dvd_comp_C_mul_X_add_C_iff p q 1 (-a)
  /-
    🎉 no goals
  -/


lemma dvd_comp_X_add_C_iff (p q : R[X]) (a : R) :
    p ∣ q.comp (X + C a) ↔ p.comp (X - C a) ∣ q := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    a : R
    ⊢ Iff (Dvd.dvd p (q.comp (HAdd.hAdd Polynomial.X (Polynomial.C a)))) (Dvd.dvd  …
  -/
  simpa using dvd_comp_X_sub_C_iff p q (-a)
  /-
    🎉 no goals
  -/


lemma dvd_comp_neg_X_iff (p q : R[X]) : p ∣ q.comp (-X) ↔ p.comp (-X) ∣ q := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    ⊢ Iff (Dvd.dvd p (q.comp (Neg.neg Polynomial.X))) (Dvd.dvd (p.comp (Neg.neg Po …
  -/
  let _ := invertibleOne (α := R)
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    x✝ : Invertible 1 := invertibleOne
    ⊢ Iff (Dvd.dvd p (q.comp (Neg.neg Polynomial.X))) (Dvd.dvd (p.comp (Neg.neg Po …
  -/
  let _ := invertibleNeg (α := R) 1
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    x✝¹ : Invertible 1 := invertibleOne
    x✝ : Invertible (-1) := invertibleNeg 1
    ⊢ Iff (Dvd.dvd p (q.comp (Neg.neg Polynomial.X))) (Dvd.dvd (p.comp (Neg.neg Po …
  -/
  simpa using dvd_comp_C_mul_X_add_C_iff p q (-1) 0
  /-
    🎉 no goals
  -/


lemma units_coeff_zero_smul (c : R[X]ˣ) (p : R[X]) : (c : R[X]).coeff 0 • p = c * p := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    c : Units (Polynomial R)
    p : Polynomial R
    ⊢ Eq (HSMul.hSMul ((↑c).coeff 0) p) (HMul.hMul (↑c) p)
  -/
  rw [← Polynomial.C_mul', ← Polynomial.eq_C_of_degree_eq_zero (degree_coe_units c)]
  /-
    🎉 no goals
  -/


lemma aeval_apply_smul_mem_of_le_comap'
    [Semiring A] [Algebra R A] [Module A M] [IsScalarTower R A M] (hm : m ∈ q) (p : R[X]) (a : A)
    (hq : q ≤ q.comap (Algebra.lsmul R R M a)) :
    aeval a p • m ∈ q := by
  refine p.induction_on (M := fun f ↦ aeval a f • m ∈ q) (by simpa) (fun f₁ f₂ h₁ h₂ ↦ ?_)
    (fun n t hmq ↦ ?_)
    /-
      case refine_1
      R : Type u
      A : Type z
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      q : Submodule R M
      m : M
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      hm : Membership.mem q m
      p : Polynomial R
      a : A
      hq : LE.le q (Submodule.comap ((Algebra.lsmul R R M) a) q)
      f₁ f₂ : Polynomial R
      h₁ : (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) f₁
      h₂ : (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) f₂
      ⊢ (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) (HAdd.h …
    -/
  · simp_rw [map_add, add_smul]
    /-
      case refine_1
      R : Type u
      A : Type z
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      q : Submodule R M
      m : M
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      hm : Membership.mem q m
      p : Polynomial R
      a : A
      hq : LE.le q (Submodule.comap ((Algebra.lsmul R R M) a) q)
      f₁ f₂ : Polynomial R
      h₁ : (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) f₁
      h₂ : (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) f₂
      ⊢ Membership.mem q (HAdd.hAdd (HSMul.hSMul ((Polynomial.aeval a) f₁) m) (HSMul …
    -/
    exact Submodule.add_mem q h₁ h₂
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      A : Type z
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      q : Submodule R M
      m : M
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      hm : Membership.mem q m
      p : Polynomial R
      a : A
      hq : LE.le q (Submodule.comap ((Algebra.lsmul R R M) a) q)
      n : Nat
      t : R
      hmq : (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) (HM …
      ⊢ (fun f => Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) f) m)) (HMul.h …
    -/
  · dsimp only at hmq ⊢
    /-
      case refine_2
      R : Type u
      A : Type z
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      q : Submodule R M
      m : M
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      hm : Membership.mem q m
      p : Polynomial R
      a : A
      hq : LE.le q (Submodule.comap ((Algebra.lsmul R R M) a) q)
      n : Nat
      t : R
      hmq : Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) (HMul.hMul (Polynomi …
      ⊢ Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) (HMul.hMul (Polynomial.C …
    -/
    rw [pow_succ', mul_left_comm, map_mul, aeval_X, mul_smul]
    /-
      case refine_2
      R : Type u
      A : Type z
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      q : Submodule R M
      m : M
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      hm : Membership.mem q m
      p : Polynomial R
      a : A
      hq : LE.le q (Submodule.comap ((Algebra.lsmul R R M) a) q)
      n : Nat
      t : R
      hmq : Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) (HMul.hMul (Polynomi …
      ⊢ Membership.mem q (HSMul.hSMul a (HSMul.hSMul ((Polynomial.aeval a) (HMul.hMu …
    -/
    rw [← q.map_le_iff_le_comap] at hq
    /-
      case refine_2
      R : Type u
      A : Type z
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      q : Submodule R M
      m : M
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      hm : Membership.mem q m
      p : Polynomial R
      a : A
      hq : LE.le (Submodule.map ((Algebra.lsmul R R M) a) q) q
      n : Nat
      t : R
      hmq : Membership.mem q (HSMul.hSMul ((Polynomial.aeval a) (HMul.hMul (Polynomi …
      ⊢ Membership.mem q (HSMul.hSMul a (HSMul.hSMul ((Polynomial.aeval a) (HMul.hMu …
    -/
    exact hq ⟨_, hmq, rfl⟩
    /-
      🎉 no goals
    -/


lemma aeval_apply_smul_mem_of_le_comap
    (hm : m ∈ q) (p : R[X]) (f : Module.End R M) (hq : q ≤ q.comap f) :
    aeval f p m ∈ q :=
  aeval_apply_smul_mem_of_le_comap' hm p f hq


theorem eq_zero_of_mul_eq_zero_of_smul (P : R[X]) (h : ∀ r : R, r • P = 0 → r = 0) :
    ∀ (Q : R[X]), P * Q = 0 → Q = 0 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    ⊢ ∀ (Q : Polynomial R), Eq (HMul.hMul P Q) 0 → Eq Q 0
  -/
  intro Q hQ
  suffices ∀ i, P.coeff i • Q = 0 by
    rw [← leadingCoeff_eq_zero]
    apply h
    simpa [ext_iff, mul_comm Q.leadingCoeff] using fun i ↦ congr_arg (·.coeff Q.natDegree) (this i)
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    ⊢ ∀ (i : Nat), Eq (HSMul.hSMul (P.coeff i) Q) 0
  -/
  apply Nat.strong_decreasing_induction
    /-
      case base
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      ⊢ Exists fun n => ∀ (m : Nat), GT.gt m n → Eq (HSMul.hSMul (P.coeff m) Q) 0
    -/
  · use P.natDegree
    /-
      case h
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      ⊢ ∀ (m : Nat), GT.gt m P.natDegree → Eq (HSMul.hSMul (P.coeff m) Q) 0
    -/
    intro i hi
    /-
      case h
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      i : Nat
      hi : GT.gt i P.natDegree
      ⊢ Eq (HSMul.hSMul (P.coeff i) Q) 0
    -/
    rw [coeff_eq_zero_of_natDegree_lt hi, zero_smul]
    /-
      🎉 no goals
    -/
  /-
    case step
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    ⊢ ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → Eq (HSMul.hSMul (P.coeff m) Q) 0) → E …
  -/
  intro l IH
  /-
    case step
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    ⊢ Eq (HSMul.hSMul (P.coeff l) Q) 0
  -/
  obtain _|hl := (natDegree_smul_le (P.coeff l) Q).lt_or_eq
    /-
      case step.inl
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      l : Nat
      IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
      h✝ : LT.lt (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
      ⊢ Eq (HSMul.hSMul (P.coeff l) Q) 0
    -/
  · apply eq_zero_of_mul_eq_zero_of_smul _ h (P.coeff l • Q)
    /-
      case step.inl
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      l : Nat
      IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
      h✝ : LT.lt (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
      ⊢ Eq (HMul.hMul P (HSMul.hSMul (P.coeff l) Q)) 0
    -/
    rw [smul_eq_C_mul, mul_left_comm, hQ, mul_zero]
    /-
      🎉 no goals
    -/
  suffices P.coeff l * Q.leadingCoeff = 0 by
    rwa [← leadingCoeff_eq_zero, ← coeff_natDegree, coeff_smul, hl, coeff_natDegree, smul_eq_mul]
  /-
    case step.inr
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    ⊢ Eq (HMul.hMul (P.coeff l) Q.leadingCoeff) 0
  -/
  let m := Q.natDegree
  /-
    case step.inr
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    m : Nat := Q.natDegree
    ⊢ Eq (HMul.hMul (P.coeff l) Q.leadingCoeff) 0
  -/
  suffices (P * Q).coeff (l + m) = P.coeff l * Q.leadingCoeff by rw [← this, hQ, coeff_zero]
  /-
    case step.inr
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    m : Nat := Q.natDegree
    ⊢ Eq ((HMul.hMul P Q).coeff (HAdd.hAdd l m)) (HMul.hMul (P.coeff l) Q.leadingC …
  -/
  rw [coeff_mul]
  /-
    case step.inr
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    m : Nat := Q.natDegree
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd l m)).sum fun x => HMul. …
  -/
  apply Finset.sum_eq_single (l, m) _ (by simp)
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    m : Nat := Q.natDegree
    ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
  -/
  simp only [Finset.mem_antidiagonal, ne_eq, Prod.forall, Prod.mk.injEq, not_and]
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    m : Nat := Q.natDegree
    ⊢ ∀ (a b : Nat), Eq (HAdd.hAdd a b) (HAdd.hAdd l m) → (Eq a l → Not (Eq b m))  …
  -/
  intro i j hij H
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
    Q : Polynomial R
    hQ : Eq (HMul.hMul P Q) 0
    l : Nat
    IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
    hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
    m : Nat := Q.natDegree
    i j : Nat
    hij : Eq (HAdd.hAdd i j) (HAdd.hAdd l m)
    H : Eq i l → Not (Eq j m)
    ⊢ Eq (HMul.hMul (P.coeff i) (Q.coeff j)) 0
  -/
  obtain hi|rfl|hi := lt_trichotomy i l
    /-
      case inl
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      l : Nat
      IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
      hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
      m : Nat := Q.natDegree
      i j : Nat
      hij : Eq (HAdd.hAdd i j) (HAdd.hAdd l m)
      H : Eq i l → Not (Eq j m)
      hi : LT.lt i l
      ⊢ Eq (HMul.hMul (P.coeff i) (Q.coeff j)) 0
    -/
  · have hj : m < j := by omega
    /-
      case inl
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      l : Nat
      IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
      hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
      m : Nat := Q.natDegree
      i j : Nat
      hij : Eq (HAdd.hAdd i j) (HAdd.hAdd l m)
      H : Eq i l → Not (Eq j m)
      hi : LT.lt i l
      hj : LT.lt m j
      ⊢ Eq (HMul.hMul (P.coeff i) (Q.coeff j)) 0
    -/
    rw [coeff_eq_zero_of_natDegree_lt hj, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      m : Nat := Q.natDegree
      i j : Nat
      IH : ∀ (m : Nat), GT.gt m i → Eq (HSMul.hSMul (P.coeff m) Q) 0
      hl : Eq (HSMul.hSMul (P.coeff i) Q).natDegree Q.natDegree
      hij : Eq (HAdd.hAdd i j) (HAdd.hAdd i m)
      H : Eq i i → Not (Eq j m)
      ⊢ Eq (HMul.hMul (P.coeff i) (Q.coeff j)) 0
    -/
  · omega
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u
      inst✝ : CommSemiring R
      P : Polynomial R
      h : ∀ (r : R), Eq (HSMul.hSMul r P) 0 → Eq r 0
      Q : Polynomial R
      hQ : Eq (HMul.hMul P Q) 0
      l : Nat
      IH : ∀ (m : Nat), GT.gt m l → Eq (HSMul.hSMul (P.coeff m) Q) 0
      hl : Eq (HSMul.hSMul (P.coeff l) Q).natDegree Q.natDegree
      m : Nat := Q.natDegree
      i j : Nat
      hij : Eq (HAdd.hAdd i j) (HAdd.hAdd l m)
      H : Eq i l → Not (Eq j m)
      hi : LT.lt l i
      ⊢ Eq (HMul.hMul (P.coeff i) (Q.coeff j)) 0
    -/
  · rw [← coeff_C_mul, ← smul_eq_C_mul, IH _ hi, coeff_zero]
    /-
      🎉 no goals
    -/
termination_by Q => Q.natDegree


open nonZeroDivisors in
/-- *McCoy theorem*: a polynomial `P : R[X]` is a zerodivisor if and only if there is `a : R`
such that `a ≠ 0` and `a • P = 0`. -/
theorem nmem_nonZeroDivisors_iff {P : R[X]} : P ∉ R[X]⁰ ↔ ∃ a : R, a ≠ 0 ∧ a • P = 0 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    ⊢ Iff (Not (Membership.mem (nonZeroDivisors (Polynomial R)) P)) (Exists fun a  …
  -/
  refine ⟨fun hP ↦ ?_, fun ⟨a, ha, h⟩ h1 ↦ ha <| C_eq_zero.1 <| (h1 _) <| smul_eq_C_mul a ▸ h⟩
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    hP : Not (Membership.mem (nonZeroDivisors (Polynomial R)) P)
    ⊢ Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a P) 0)
  -/
  by_contra! h
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    hP : Not (Membership.mem (nonZeroDivisors (Polynomial R)) P)
    h : ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a P) 0
    ⊢ False
  -/
  obtain ⟨Q, hQ⟩ := _root_.nmem_nonZeroDivisors_iff.1 hP
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    hP : Not (Membership.mem (nonZeroDivisors (Polynomial R)) P)
    h : ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a P) 0
    Q : Polynomial R
    hQ : Membership.mem (setOf fun s => And (Eq (HMul.hMul s P) 0) (Ne s 0)) Q
    ⊢ False
  -/
  refine hQ.2 (eq_zero_of_mul_eq_zero_of_smul P (fun a ha ↦ ?_) Q (mul_comm P _ ▸ hQ.1))
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    hP : Not (Membership.mem (nonZeroDivisors (Polynomial R)) P)
    h : ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a P) 0
    Q : Polynomial R
    hQ : Membership.mem (setOf fun s => And (Eq (HMul.hMul s P) 0) (Ne s 0)) Q
    a : R
    ha : Eq (HSMul.hSMul a P) 0
    ⊢ Eq a 0
  -/
  contrapose! ha
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    hP : Not (Membership.mem (nonZeroDivisors (Polynomial R)) P)
    h : ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a P) 0
    Q : Polynomial R
    hQ : Membership.mem (setOf fun s => And (Eq (HMul.hMul s P) 0) (Ne s 0)) Q
    a : R
    ha : Ne a 0
    ⊢ Ne (HSMul.hSMul a P) 0
  -/
  exact h a ha
  /-
    🎉 no goals
  -/


open nonZeroDivisors in
protected lemma mem_nonZeroDivisors_iff {P : R[X]} : P ∈ R[X]⁰ ↔ ∀ a : R, a • P = 0 → a = 0 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    P : Polynomial R
    ⊢ Iff (Membership.mem (nonZeroDivisors (Polynomial R)) P) (∀ (a : R), Eq (HSMu …
  -/
  simpa [not_imp_not] using (nmem_nonZeroDivisors_iff (P := P)).not
  /-
    🎉 no goals
  -/


