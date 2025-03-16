theorem aeval_map_algebraMap (x : σ → B) (p : MvPolynomial σ R) :
    aeval x (map (algebraMap R A) p) = aeval x p := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    σ : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : CommSemiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : σ → B
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.aeval x) ((MvPolynomial.map (algebraMap R A)) p)) ((MvPoly …
  -/
  rw [aeval_def, aeval_def, eval₂_map, IsScalarTower.algebraMap_eq R A B]
  /-
    🎉 no goals
  -/


theorem aeval_algebraMap_apply (x : σ → A) (p : MvPolynomial σ R) :
    aeval (algebraMap A B ∘ x) p = algebraMap A B (MvPolynomial.aeval x p) := by
  rw [aeval_def, aeval_def, ← coe_eval₂Hom, ← coe_eval₂Hom, map_eval₂Hom, ←
    IsScalarTower.algebraMap_eq]
  -- Porting note: added
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    σ : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : CommSemiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : σ → A
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval₂Hom (algebraMap R B) (Function.comp (⇑(algebraMap A B …
  -/
  simp only [Function.comp_def]
  /-
    🎉 no goals
  -/


theorem aeval_algebraMap_eq_zero_iff [NoZeroSMulDivisors A B] [Nontrivial B] (x : σ → A)
    (p : MvPolynomial σ R) : aeval (algebraMap A B ∘ x) p = 0 ↔ aeval x p = 0 := by
  rw [aeval_algebraMap_apply, Algebra.algebraMap_eq_smul_one, smul_eq_zero,
    iff_false_intro (one_ne_zero' B), or_false]


theorem aeval_algebraMap_eq_zero_iff_of_injective {x : σ → A} {p : MvPolynomial σ R}
    (h : Function.Injective (algebraMap A B)) :
    aeval (algebraMap A B ∘ x) p = 0 ↔ aeval x p = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    σ : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : CommSemiring B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : σ → A
    p : MvPolynomial σ R
    h : Function.Injective ⇑(algebraMap A B)
    ⊢ Iff (Eq ((MvPolynomial.aeval (Function.comp (⇑(algebraMap A B)) x)) p) 0) (E …
  -/
  rw [aeval_algebraMap_apply, ← (algebraMap A B).map_zero, h.eq_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem mvPolynomial_aeval_coe (S : Subalgebra R A) (x : σ → S) (p : MvPolynomial σ R) :
                                                   /-
                                                     R : Type u_1
                                                     A : Type u_2
                                                     σ : Type u_4
                                                     inst✝² : CommSemiring R
                                                     inst✝¹ : CommSemiring A
                                                     inst✝ : Algebra R A
                                                     S : Subalgebra R A
                                                     x : σ → Subtype fun x => Membership.mem S x
                                                     p : MvPolynomial σ R
                                                     ⊢ Eq ((MvPolynomial.aeval fun i => ↑(x i)) p) ↑((MvPolynomial.aeval x) p)
                                                   -/
    aeval (fun i => (x i : A)) p = aeval x p := by convert aeval_algebraMap_apply A x p
                                                   /-
                                                     🎉 no goals
                                                   -/


