theorem mvPolynomial_zeroLocus_definable {ι K : Type*} [Field K]
    [CompatibleRing K] (S : Finset (MvPolynomial ι K)) :
    Set.Definable (⋃ p ∈ S, p.coeff '' p.support : Set K) Language.ring
      (zeroLocus (Ideal.span (S : Set (MvPolynomial ι K)))) := by
  /-
    ι : Type u_1
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    S : Finset (MvPolynomial ι K)
    ⊢ (Set.iUnion fun p => Set.iUnion fun h => Set.image (fun m => MvPolynomial.co …
  -/
  rw [Set.definable_iff_exists_formula_sum]
  /-
    ι : Type u_1
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    S : Finset (MvPolynomial ι K)
    ⊢ Exists fun φ => Eq (MvPolynomial.zeroLocus (Ideal.span ↑S)) (setOf fun v =>  …
  -/
  let p' := genericPolyMap (fun p : S => p.1.support)
  /-
    ι : Type u_1
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    S : Finset (MvPolynomial ι K)
    p' : (Subtype fun x => Membership.mem S x) → FreeCommRing (Sum (Sigma fun i => …
    ⊢ Exists fun φ => Eq (MvPolynomial.zeroLocus (Ideal.span ↑S)) (setOf fun v =>  …
  -/
  letI := Classical.decEq ι
  /-
    ι : Type u_1
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    S : Finset (MvPolynomial ι K)
    p' : (Subtype fun x => Membership.mem S x) → FreeCommRing (Sum (Sigma fun i => …
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Exists fun φ => Eq (MvPolynomial.zeroLocus (Ideal.span ↑S)) (setOf fun v =>  …
  -/
  letI := Classical.decEq K
  /-
    ι : Type u_1
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    S : Finset (MvPolynomial ι K)
    p' : (Subtype fun x => Membership.mem S x) → FreeCommRing (Sum (Sigma fun i => …
    this✝ : DecidableEq ι := Classical.decEq ι
    this : DecidableEq K := Classical.decEq K
    ⊢ Exists fun φ => Eq (MvPolynomial.zeroLocus (Ideal.span ↑S)) (setOf fun v =>  …
  -/
  rw [MvPolynomial.zeroLocus_span]
  refine ⟨BoundedFormula.iInf
      (fun i : S => Term.equal
        ((termOfFreeCommRing (p' i)).relabel
          (Sum.map (fun p => ⟨p.1.1.coeff p.2.1, by
            simp only [Set.mem_iUnion]
            exact ⟨p.1.1, p.1.2, Set.mem_image_of_mem _ p.2.2⟩⟩) id)) 0), ?_⟩
  /-
    ι : Type u_1
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : FirstOrder.Ring.CompatibleRing K
    S : Finset (MvPolynomial ι K)
    p' : (Subtype fun x => Membership.mem S x) → FreeCommRing (Sum (Sigma fun i => …
    this✝ : DecidableEq ι := Classical.decEq ι
    this : DecidableEq K := Classical.decEq K
    ⊢ Eq (setOf fun x => ∀ (p : MvPolynomial ι K), Membership.mem (↑S) p → Eq ((Mv …
  -/
  simp [Formula.Realize, Term.equal, Function.comp_def, p']
  /-
    🎉 no goals
  -/



