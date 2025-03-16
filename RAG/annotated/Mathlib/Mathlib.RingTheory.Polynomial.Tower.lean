@[simp]
theorem aeval_map_algebraMap (x : B) (p : R[X]) : aeval x (map (algebraMap R A) p) = aeval x p := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : B
    p : Polynomial R
    ⊢ Eq ((Polynomial.aeval x) (Polynomial.map (algebraMap R A) p)) ((Polynomial.a …
  -/
  rw [aeval_def, aeval_def, eval₂_map, IsScalarTower.algebraMap_eq R A B]
  /-
    🎉 no goals
  -/


@[simp]
lemma eval_map_algebraMap (P : R[X]) (b : B) :
    (map (algebraMap R B) P).eval b = aeval b P := by
  /-
    R : Type u_1
    B : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    P : Polynomial R
    b : B
    ⊢ Eq (Polynomial.eval b (Polynomial.map (algebraMap R B) P)) ((Polynomial.aeva …
  -/
  rw [aeval_def, eval_map]
  /-
    🎉 no goals
  -/


theorem aeval_algebraMap_apply (x : A) (p : R[X]) :
    aeval (algebraMap A B x) p = algebraMap A B (aeval x p) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : A
    p : Polynomial R
    ⊢ Eq ((Polynomial.aeval ((algebraMap A B) x)) p) ((algebraMap A B) ((Polynomia …
  -/
  rw [aeval_def, aeval_def, hom_eval₂, ← IsScalarTower.algebraMap_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem aeval_algebraMap_eq_zero_iff [NoZeroSMulDivisors A B] [Nontrivial B] (x : A) (p : R[X]) :
    aeval (algebraMap A B x) p = 0 ↔ aeval x p = 0 := by
  rw [aeval_algebraMap_apply, Algebra.algebraMap_eq_smul_one, smul_eq_zero,
    iff_false_intro (one_ne_zero' B), or_false]


theorem aeval_algebraMap_eq_zero_iff_of_injective {x : A} {p : R[X]}
    (h : Function.Injective (algebraMap A B)) : aeval (algebraMap A B x) p = 0 ↔ aeval x p = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : A
    p : Polynomial R
    h : Function.Injective ⇑(algebraMap A B)
    ⊢ Iff (Eq ((Polynomial.aeval ((algebraMap A B) x)) p) 0) (Eq ((Polynomial.aeva …
  -/
  rw [aeval_algebraMap_apply, ← (algebraMap A B).map_zero, h.eq_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem aeval_coe (S : Subalgebra R A) (x : S) (p : R[X]) : aeval (x : A) p = aeval x p :=
  aeval_algebraMap_apply A x p


