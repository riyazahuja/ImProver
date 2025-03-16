protected lemma IsWeightedHomogeneous.pderiv [AddCancelCommMonoid M] {w : σ → M} {n n' : M} {i : σ}
    (h : φ.IsWeightedHomogeneous w n) (h' : n' + w i = n) :
    (pderiv i φ).IsWeightedHomogeneous w n' := by
  rw [← mem_weightedHomogeneousSubmodule, weightedHomogeneousSubmodule_eq_finsupp_supported,
    Finsupp.supported_eq_span_single] at h
  /-
    R : Type u_1
    σ : Type u_2
    M : Type u_3
    inst✝¹ : CommSemiring R
    φ : MvPolynomial σ R
    inst✝ : AddCancelCommMonoid M
    w : σ → M
    n n' : M
    i : σ
    h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
    h' : Eq (HAdd.hAdd n' (w i)) n
    ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) φ) n'
  -/
  refine Submodule.span_induction ?_ ?_ (fun p q _ _ hp hq ↦ ?_) (fun r p _ h ↦ ?_) h
    /-
      case refine_1
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      ⊢ ∀ (x : MvPolynomial σ R), Membership.mem (Set.image (fun i => Finsupp.single …
    -/
  · rintro _ ⟨m, hm, rfl⟩
    /-
      case refine_1.intro.intro
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) ((fun i => Fin …
    -/
    simp_rw [single_eq_monomial, pderiv_monomial, one_mul]
    /-
      case refine_1.intro.intro
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.monomial (HSub.hSub m (F …
    -/
    by_cases hi : m i = 0
      /-
        case pos
        R : Type u_1
        σ : Type u_2
        M : Type u_3
        inst✝¹ : CommSemiring R
        φ : MvPolynomial σ R
        inst✝ : AddCancelCommMonoid M
        w : σ → M
        n n' : M
        i : σ
        h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
        h' : Eq (HAdd.hAdd n' (w i)) n
        m : Finsupp σ Nat
        hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
        hi : Eq (m i) 0
        ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.monomial (HSub.hSub m (F …
      -/
    · rw [hi, Nat.cast_zero, monomial_zero]; apply isWeightedHomogeneous_zero
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case neg
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      hi : Not (Eq (m i) 0)
      ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.monomial (HSub.hSub m (F …
    -/
    convert isWeightedHomogeneous_monomial ..
    /-
      case neg.convert_10
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      hi : Not (Eq (m i) 0)
      ⊢ Eq ((Finsupp.weight w) (HSub.hSub m (Finsupp.single i 1))) n'
    -/
    rw [← add_right_cancel_iff (a := w i), h', ← hm, weight_sub_single_add hi]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) 0) n'
    -/
  · rw [map_zero]; apply isWeightedHomogeneous_zero
                   /-
                     🎉 no goals
                   -/
    /-
      case refine_3
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      h' : Eq (HAdd.hAdd n' (w i)) n
      p q : MvPolynomial σ R
      x✝¹ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1 …
      x✝ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) …
      hp : MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) p) n'
      hq : MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) q) n'
      ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) (HAdd.hAdd p q …
    -/
  · rw [map_add]; exact hp.add hq
                  /-
                    🎉 no goals
                  -/
    /-
      case refine_4
      R : Type u_1
      σ : Type u_2
      M : Type u_3
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : AddCancelCommMonoid M
      w : σ → M
      n n' : M
      i : σ
      h✝ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) …
      h' : Eq (HAdd.hAdd n' (w i)) n
      r : R
      p : MvPolynomial σ R
      x✝ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) …
      h : MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) p) n'
      ⊢ MvPolynomial.IsWeightedHomogeneous w ((MvPolynomial.pderiv i) (HSMul.hSMul r …
    -/
  · rw [(pderiv i).map_smul]; exact (weightedHomogeneousSubmodule ..).smul_mem _ h
                              /-
                                🎉 no goals
                              -/


protected lemma IsHomogeneous.pderiv {n : ℕ} {i : σ} (h : φ.IsHomogeneous n) :
    (pderiv i φ).IsHomogeneous (n - 1) := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝ : CommSemiring R
    φ : MvPolynomial σ R
    n : Nat
    i : σ
    h : φ.IsHomogeneous n
    ⊢ ((MvPolynomial.pderiv i) φ).IsHomogeneous (HSub.hSub n 1)
  -/
  obtain _ | n := n
    /-
      case zero
      R : Type u_1
      σ : Type u_2
      inst✝ : CommSemiring R
      φ : MvPolynomial σ R
      i : σ
      h : φ.IsHomogeneous 0
      ⊢ ((MvPolynomial.pderiv i) φ).IsHomogeneous (HSub.hSub 0 1)
    -/
  · rw [← totalDegree_zero_iff_isHomogeneous, totalDegree_eq_zero_iff_eq_C] at h
    /-
      case zero
      R : Type u_1
      σ : Type u_2
      inst✝ : CommSemiring R
      φ : MvPolynomial σ R
      i : σ
      h : Eq φ (MvPolynomial.C (MvPolynomial.coeff 0 φ))
      ⊢ ((MvPolynomial.pderiv i) φ).IsHomogeneous (HSub.hSub 0 1)
    -/
    rw [h, pderiv_C]; apply isHomogeneous_zero
                      /-
                        🎉 no goals
                      -/
    /-
      case succ
      R : Type u_1
      σ : Type u_2
      inst✝ : CommSemiring R
      φ : MvPolynomial σ R
      i : σ
      n : Nat
      h : φ.IsHomogeneous (HAdd.hAdd n 1)
      ⊢ ((MvPolynomial.pderiv i) φ).IsHomogeneous (HSub.hSub (HAdd.hAdd n 1) 1)
    -/
  · exact IsWeightedHomogeneous.pderiv h rfl
    /-
      🎉 no goals
    -/


open Finset in
/-- Euler's identity for weighted homogeneous polynomials. -/
theorem IsWeightedHomogeneous.sum_weight_X_mul_pderiv {w : σ → ℕ}
    (h : φ.IsWeightedHomogeneous w n) : ∑ i : σ, w i • (X i * pderiv i φ) = n • φ := by
  rw [← mem_weightedHomogeneousSubmodule, weightedHomogeneousSubmodule_eq_finsupp_supported,
    supported_eq_span_single] at h
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    φ : MvPolynomial σ R
    inst✝ : Fintype σ
    n : Nat
    w : σ → Nat
    h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X i) …
  -/
  refine Submodule.span_induction ?_ ?_ (fun p q _ _ hp hq ↦ ?_) (fun r p _ h ↦ ?_) h
    /-
      case refine_1
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      ⊢ ∀ (x : MvPolynomial σ R), Membership.mem (Set.image (fun i => Finsupp.single …
    -/
  · rintro _ ⟨m, hm, rfl⟩
    /-
      case refine_1.intro.intro
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X i) …
    -/
    simp_rw [single_eq_monomial, X_mul_pderiv_monomial, smul_smul, ← sum_smul, mul_comm (w _)]
    /-
      case refine_1.intro.intro
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      ⊢ Eq (HSMul.hSMul (Finset.univ.sum fun x => HMul.hMul (m x) (w x)) ((MvPolynom …
    -/
    congr
    /-
      case refine_1.intro.intro.e_a
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      m : Finsupp σ Nat
      hm : Membership.mem (setOf fun d => Eq ((Finsupp.weight w) d) n) m
      ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (m x) (w x)) n
    -/
    rwa [Set.mem_setOf, weight_apply, sum_fintype] at hm
    /-
      case refine_1.intro.intro.e_a.h
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      m : Finsupp σ Nat
      hm : Eq (m.sum fun i c => HSMul.hSMul c (w i)) n
      ⊢ ∀ (i : σ), Eq (HSMul.hSMul 0 (w i)) 0
    -/
    intro; apply zero_smul
           /-
             🎉 no goals
           -/
    /-
      case refine_2
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X i) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1)  …
      p q : MvPolynomial σ R
      x✝¹ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1 …
      x✝ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) …
      hp : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X …
      hq : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X …
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X i) …
    -/
  · simp_rw [map_add, left_distrib, smul_add, sum_add_distrib, hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u_1
      σ : Type u_2
      inst✝¹ : CommSemiring R
      φ : MvPolynomial σ R
      inst✝ : Fintype σ
      n : Nat
      w : σ → Nat
      h✝ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) …
      r : R
      p : MvPolynomial σ R
      x✝ : Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) …
      h : Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X  …
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (w i) (HMul.hMul (MvPolynomial.X i) …
    -/
  · simp_rw [(pderiv _).map_smul, nsmul_eq_mul, mul_smul_comm, ← Finset.smul_sum, ← nsmul_eq_mul, h]
    /-
      🎉 no goals
    -/


/-- Euler's identity for homogeneous polynomials. -/
theorem IsHomogeneous.sum_X_mul_pderiv (h : φ.IsHomogeneous n) :
    ∑ i : σ, X i * pderiv i φ = n • φ := by
  /-
    R : Type u_1
    σ : Type u_2
    inst✝¹ : CommSemiring R
    φ : MvPolynomial σ R
    inst✝ : Fintype σ
    n : Nat
    h : φ.IsHomogeneous n
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (MvPolynomial.X i) ((MvPolynomial.pde …
  -/
  simp_rw [← h.sum_weight_X_mul_pderiv, Pi.one_apply, one_smul]
  /-
    🎉 no goals
  -/


