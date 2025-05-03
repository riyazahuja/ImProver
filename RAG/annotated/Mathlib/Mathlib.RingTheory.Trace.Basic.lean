theorem Algebra.traceForm_toMatrix_powerBasis (h : PowerBasis R S) :
    BilinForm.toMatrix h.basis (traceForm R S) = of fun i j => trace R S (h.gen ^ (i.1 + j.1)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    h : PowerBasis R S
    ⊢ Eq ((BilinForm.toMatrix h.basis) (Algebra.traceForm R S)) (Matrix.of fun i j …
  -/
  ext; rw [traceForm_toMatrix, of_apply, pow_add, h.basis_eq_pow, h.basis_eq_pow]
       /-
         🎉 no goals
       -/


/-- Given `pb : PowerBasis K S`, the trace of `pb.gen` is `-(minpoly K pb.gen).nextCoeff`. -/
theorem PowerBasis.trace_gen_eq_nextCoeff_minpoly [Nontrivial S] (pb : PowerBasis K S) :
    Algebra.trace K S pb.gen = -(minpoly K pb.gen).nextCoeff := by
  /-
    S : Type u_2
    inst✝³ : CommRing S
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra K S
    inst✝ : Nontrivial S
    pb : PowerBasis K S
    ⊢ Eq ((Algebra.trace K S) pb.gen) (Neg.neg (minpoly K pb.gen).nextCoeff)
  -/
  have d_pos : 0 < pb.dim := PowerBasis.dim_pos pb
  /-
    S : Type u_2
    inst✝³ : CommRing S
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra K S
    inst✝ : Nontrivial S
    pb : PowerBasis K S
    d_pos : LT.lt 0 pb.dim
    ⊢ Eq ((Algebra.trace K S) pb.gen) (Neg.neg (minpoly K pb.gen).nextCoeff)
  -/
  have d_pos' : 0 < (minpoly K pb.gen).natDegree := by simpa
  /-
    S : Type u_2
    inst✝³ : CommRing S
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Algebra K S
    inst✝ : Nontrivial S
    pb : PowerBasis K S
    d_pos : LT.lt 0 pb.dim
    d_pos' : LT.lt 0 (minpoly K pb.gen).natDegree
    ⊢ Eq ((Algebra.trace K S) pb.gen) (Neg.neg (minpoly K pb.gen).nextCoeff)
  -/
  haveI : Nonempty (Fin pb.dim) := ⟨⟨0, d_pos⟩⟩
  rw [trace_eq_matrix_trace pb.basis, trace_eq_neg_charpoly_coeff, charpoly_leftMulMatrix, ←
    pb.natDegree_minpoly, Fintype.card_fin, ← nextCoeff_of_natDegree_pos d_pos']


/-- Given `pb : PowerBasis K S`, then the trace of `pb.gen` is
`((minpoly K pb.gen).aroots F).sum`. -/
theorem PowerBasis.trace_gen_eq_sum_roots [Nontrivial S] (pb : PowerBasis K S)
    (hf : (minpoly K pb.gen).Splits (algebraMap K F)) :
    algebraMap K F (trace K S pb.gen) = ((minpoly K pb.gen).aroots F).sum := by
  rw [PowerBasis.trace_gen_eq_nextCoeff_minpoly, RingHom.map_neg, ←
    nextCoeff_map (algebraMap K F).injective,
    sum_roots_eq_nextCoeff_of_monic_of_split ((minpoly.monic (PowerBasis.isIntegral_gen _)).map _)
      ((splits_id_iff_splits _).2 hf),
    neg_neg]


theorem trace_gen_eq_zero {x : L} (hx : ¬IsIntegral K x) :
    Algebra.trace K K⟮x⟯ (AdjoinSimple.gen K x) = 0 := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : Not (IsIntegral K x)
    ⊢ Eq ((Algebra.trace K (Subtype fun x_1 => Membership.mem (IntermediateField.a …
  -/
  rw [trace_eq_zero_of_not_exists_basis, LinearMap.zero_apply]
  /-
    case h
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : Not (IsIntegral K x)
    ⊢ Not (Exists fun s => Nonempty (Basis (Subtype fun x_1 => Membership.mem s x_ …
  -/
  contrapose! hx
  /-
    case h
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : Exists fun s => Nonempty (Basis (Subtype fun x_1 => Membership.mem s x_1) …
    ⊢ IsIntegral K x
  -/
  obtain ⟨s, ⟨b⟩⟩ := hx
  /-
    case h.intro.intro
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    s : Finset (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin K (Sin …
    b : Basis (Subtype fun x_1 => Membership.mem s x_1) K (Subtype fun x_1 => Memb …
    ⊢ IsIntegral K x
  -/
  refine .of_mem_of_fg K⟮x⟯.toSubalgebra ?_ x ?_
    /-
      case h.intro.intro.refine_1
      K : Type u_4
      L : Type u_5
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      x : L
      s : Finset (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin K (Sin …
      b : Basis (Subtype fun x_1 => Membership.mem s x_1) K (Subtype fun x_1 => Memb …
      ⊢ (Subalgebra.toSubmodule (IntermediateField.adjoin K (Singleton.singleton x)) …
    -/
  · exact (Submodule.fg_iff_finiteDimensional _).mpr (FiniteDimensional.of_fintype_basis b)
    /-
      🎉 no goals
    -/
    /-
      case h.intro.intro.refine_2
      K : Type u_4
      L : Type u_5
      inst✝² : Field K
      inst✝¹ : Field L
      inst✝ : Algebra K L
      x : L
      s : Finset (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin K (Sin …
      b : Basis (Subtype fun x_1 => Membership.mem s x_1) K (Subtype fun x_1 => Memb …
      ⊢ Membership.mem (IntermediateField.adjoin K (Singleton.singleton x)).toSubalg …
    -/
  · exact subset_adjoin K _ (Set.mem_singleton x)
    /-
      🎉 no goals
    -/


theorem trace_gen_eq_sum_roots (x : L) (hf : (minpoly K x).Splits (algebraMap K F)) :
    algebraMap K F (trace K K⟮x⟯ (AdjoinSimple.gen K x)) =
      ((minpoly K x).aroots F).sum := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    ⊢ Eq ((algebraMap K F) ((Algebra.trace K (Subtype fun x_1 => Membership.mem (I …
  -/
  have injKxL := (algebraMap K⟮x⟯ L).injective
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
    ⊢ Eq ((algebraMap K F) ((Algebra.trace K (Subtype fun x_1 => Membership.mem (I …
  -/
  by_cases hx : IsIntegral K x; swap
    /-
      case neg
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      F : Type u_6
      inst✝¹ : Field F
      inst✝ : Algebra K F
      x : L
      hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
      injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
      hx : Not (IsIntegral K x)
      ⊢ Eq ((algebraMap K F) ((Algebra.trace K (Subtype fun x_1 => Membership.mem (I …
    -/
  · simp [minpoly.eq_zero hx, trace_gen_eq_zero hx, aroots_def]
    /-
      🎉 no goals
    -/
  /-
    case pos
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
    hx : IsIntegral K x
    ⊢ Eq ((algebraMap K F) ((Algebra.trace K (Subtype fun x_1 => Membership.mem (I …
  -/
  rw [← adjoin.powerBasis_gen hx, (adjoin.powerBasis hx).trace_gen_eq_sum_roots] <;>
    /-
      case pos
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      F : Type u_6
      inst✝¹ : Field F
      inst✝ : Algebra K F
      x : L
      hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
      injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
      hx : IsIntegral K x
      ⊢ Eq ((minpoly K (IntermediateField.adjoin.powerBasis hx).gen).aroots F).sum ( …
    -/
    rw [adjoin.powerBasis_gen hx, ← minpoly.algebraMap_eq injKxL] <;>
    /-
      case pos
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      F : Type u_6
      inst✝¹ : Field F
      inst✝ : Algebra K F
      x : L
      hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
      injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
      hx : IsIntegral K x
      ⊢ Eq ((minpoly K ((algebraMap (Subtype fun x_1 => Membership.mem (Intermediate …
    -/
    /-
      🎉 no goals
    -/
    try simp only [AdjoinSimple.algebraMap_gen _ _]
  /-
    case pos
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
    hx : IsIntegral K x
    ⊢ Polynomial.Splits (algebraMap K F) (minpoly K x)
  -/
  exact hf
  /-
    🎉 no goals
  -/


theorem trace_eq_trace_adjoin [FiniteDimensional K L] (x : L) :
    Algebra.trace K L x = finrank K⟮x⟯ L • trace K K⟮x⟯ (AdjoinSimple.gen K x) := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    ⊢ Eq ((Algebra.trace K L) x) (HSMul.hSMul (Module.finrank (Subtype fun x_1 =>  …
  -/
  rw [← trace_trace (S := K⟮x⟯)]
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    ⊢ Eq ((Algebra.trace K (Subtype fun x_1 => Membership.mem (IntermediateField.a …
  -/
  conv in x => rw [← IntermediateField.AdjoinSimple.algebraMap_gen K x]
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    x : L
    ⊢ Eq ((Algebra.trace K (Subtype fun x_1 => Membership.mem (IntermediateField.a …
  -/
  rw [trace_algebraMap, LinearMap.map_smul_of_tower]
  /-
    🎉 no goals
  -/


theorem trace_eq_sum_roots [FiniteDimensional K L] {x : L}
    (hF : (minpoly K x).Splits (algebraMap K F)) :
    algebraMap K F (Algebra.trace K L x) =
      finrank K⟮x⟯ L • ((minpoly K x).aroots F).sum := by
  rw [trace_eq_trace_adjoin K x, Algebra.smul_def, RingHom.map_mul, ← Algebra.smul_def,
    IntermediateField.AdjoinSimple.trace_gen_eq_sum_roots _ hF, IsScalarTower.algebraMap_smul]


theorem Algebra.isIntegral_trace [FiniteDimensional L F] {x : F} (hx : IsIntegral R x) :
    IsIntegral R (Algebra.trace L F x) := by
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    L : Type u_5
    inst✝⁶ : Field L
    F : Type u_6
    inst✝⁵ : Field F
    inst✝⁴ : Algebra R L
    inst✝³ : Algebra L F
    inst✝² : Algebra R F
    inst✝¹ : IsScalarTower R L F
    inst✝ : FiniteDimensional L F
    x : F
    hx : IsIntegral R x
    ⊢ IsIntegral R ((Algebra.trace L F) x)
  -/
  have hx' : IsIntegral L x := hx.tower_top
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    L : Type u_5
    inst✝⁶ : Field L
    F : Type u_6
    inst✝⁵ : Field F
    inst✝⁴ : Algebra R L
    inst✝³ : Algebra L F
    inst✝² : Algebra R F
    inst✝¹ : IsScalarTower R L F
    inst✝ : FiniteDimensional L F
    x : F
    hx : IsIntegral R x
    hx' : IsIntegral L x
    ⊢ IsIntegral R ((Algebra.trace L F) x)
  -/
  rw [← isIntegral_algebraMap_iff (algebraMap L (AlgebraicClosure F)).injective, trace_eq_sum_roots]
    /-
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      ⊢ IsIntegral R (HSMul.hSMul (Module.finrank (Subtype fun x_1 => Membership.mem …
    -/
  · refine (IsIntegral.multiset_sum ?_).nsmul _
    /-
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      ⊢ ∀ (x_1 : AlgebraicClosure F), Membership.mem ((minpoly L x).aroots (Algebrai …
    -/
    intro y hy
    /-
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      y : AlgebraicClosure F
      hy : Membership.mem ((minpoly L x).aroots (AlgebraicClosure F)) y
      ⊢ IsIntegral R y
    -/
    rw [mem_roots_map (minpoly.ne_zero hx')] at hy
    /-
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      y : AlgebraicClosure F
      hy : Eq (Polynomial.eval₂ (algebraMap L (AlgebraicClosure F)) y (minpoly L x)) 0
      ⊢ IsIntegral R y
    -/
    use minpoly R x, minpoly.monic hx
    /-
      case right
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      y : AlgebraicClosure F
      hy : Eq (Polynomial.eval₂ (algebraMap L (AlgebraicClosure F)) y (minpoly L x)) 0
      ⊢ Eq (Polynomial.eval₂ (algebraMap R (AlgebraicClosure F)) y (minpoly R x)) 0
    -/
    rw [← aeval_def] at hy ⊢
    /-
      case right
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      y : AlgebraicClosure F
      hy : Eq ((Polynomial.aeval y) (minpoly L x)) 0
      ⊢ Eq ((Polynomial.aeval y) (minpoly R x)) 0
    -/
    exact minpoly.aeval_of_isScalarTower R x y hy
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝⁷ : CommRing R
      L : Type u_5
      inst✝⁶ : Field L
      F : Type u_6
      inst✝⁵ : Field F
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra L F
      inst✝² : Algebra R F
      inst✝¹ : IsScalarTower R L F
      inst✝ : FiniteDimensional L F
      x : F
      hx : IsIntegral R x
      hx' : IsIntegral L x
      ⊢ Polynomial.Splits (algebraMap L (AlgebraicClosure F)) (minpoly L x)
    -/
  · apply IsAlgClosed.splits_codomain
    /-
      🎉 no goals
    -/


lemma Algebra.trace_eq_of_algEquiv {A B C : Type*} [CommRing A] [CommRing B] [CommRing C]
    [Algebra A B] [Algebra A C] (e : B ≃ₐ[A] C) (x) :
    Algebra.trace A C (e x) = Algebra.trace A B x := by
  /-
    A : Type u_7
    B : Type u_8
    C : Type u_9
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : CommRing C
    inst✝¹ : Algebra A B
    inst✝ : Algebra A C
    e : AlgEquiv A B C
    x : B
    ⊢ Eq ((Algebra.trace A C) (e x)) ((Algebra.trace A B) x)
  -/
  simp_rw [Algebra.trace_apply, ← LinearMap.trace_conj' _ e.toLinearEquiv]
  /-
    A : Type u_7
    B : Type u_8
    C : Type u_9
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : CommRing C
    inst✝¹ : Algebra A B
    inst✝ : Algebra A C
    e : AlgEquiv A B C
    x : B
    ⊢ Eq ((LinearMap.trace A C) ((Algebra.lmul A C) (e x))) ((LinearMap.trace A C) …
  -/
  congr; ext; simp [LinearEquiv.conj_apply]
              /-
                🎉 no goals
              -/


lemma Algebra.trace_eq_of_ringEquiv {A B C : Type*} [CommRing A] [CommRing B] [CommRing C]
    [Algebra A C] [Algebra B C] (e : A ≃+* B) (he : (algebraMap B C).comp e = algebraMap A C) (x) :
    e (Algebra.trace A C x) = Algebra.trace B C x := by
  classical
  by_cases h : ∃ s : Finset C, Nonempty (Basis s B C)
  · obtain ⟨s, ⟨b⟩⟩ := h
    letI : Algebra A B := RingHom.toAlgebra e
    letI : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' he.symm
    rw [Algebra.trace_eq_matrix_trace b,
      Algebra.trace_eq_matrix_trace (b.mapCoeffs e.symm (by simp [Algebra.smul_def, ← he]))]
    show e.toAddMonoidHom _ = _
    rw [AddMonoidHom.map_trace]
    congr
    ext i j
    simp [leftMulMatrix_apply, LinearMap.toMatrix_apply]
  rw [trace_eq_zero_of_not_exists_basis _ h, trace_eq_zero_of_not_exists_basis,
    LinearMap.zero_apply, LinearMap.zero_apply, map_zero]
  intro ⟨s, ⟨b⟩⟩
  exact h ⟨s, ⟨b.mapCoeffs e (by simp [Algebra.smul_def, ← he])⟩⟩


lemma Algebra.trace_eq_of_equiv_equiv {A₁ B₁ A₂ B₂ : Type*} [CommRing A₁] [CommRing B₁]
    [CommRing A₂] [CommRing B₂] [Algebra A₁ B₁] [Algebra A₂ B₂] (e₁ : A₁ ≃+* A₂) (e₂ : B₁ ≃+* B₂)
    (he : RingHom.comp (algebraMap A₂ B₂) ↑e₁ = RingHom.comp ↑e₂ (algebraMap A₁ B₁)) (x) :
    Algebra.trace A₁ B₁ x = e₁.symm (Algebra.trace A₂ B₂ (e₂ x)) := by
  /-
    A₁ : Type u_7
    B₁ : Type u_8
    A₂ : Type u_9
    B₂ : Type u_10
    inst✝⁵ : CommRing A₁
    inst✝⁴ : CommRing B₁
    inst✝³ : CommRing A₂
    inst✝² : CommRing B₂
    inst✝¹ : Algebra A₁ B₁
    inst✝ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    x : B₁
    ⊢ Eq ((Algebra.trace A₁ B₁) x) (e₁.symm ((Algebra.trace A₂ B₂) (e₂ x)))
  -/
  letI := (RingHom.comp (e₂ : B₁ →+* B₂) (algebraMap A₁ B₁)).toAlgebra
  /-
    A₁ : Type u_7
    B₁ : Type u_8
    A₂ : Type u_9
    B₂ : Type u_10
    inst✝⁵ : CommRing A₁
    inst✝⁴ : CommRing B₁
    inst✝³ : CommRing A₂
    inst✝² : CommRing B₂
    inst✝¹ : Algebra A₁ B₁
    inst✝ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    x : B₁
    this : Algebra A₁ B₂ := ((↑e₂).comp (algebraMap A₁ B₁)).toAlgebra
    ⊢ Eq ((Algebra.trace A₁ B₁) x) (e₁.symm ((Algebra.trace A₂ B₂) (e₂ x)))
  -/
  let e' : B₁ ≃ₐ[A₁] B₂ := { e₂ with commutes' := fun _ ↦ rfl }
  rw [← Algebra.trace_eq_of_ringEquiv e₁ he, ← Algebra.trace_eq_of_algEquiv e',
    RingEquiv.symm_apply_apply]
  /-
    A₁ : Type u_7
    B₁ : Type u_8
    A₂ : Type u_9
    B₂ : Type u_10
    inst✝⁵ : CommRing A₁
    inst✝⁴ : CommRing B₁
    inst✝³ : CommRing A₂
    inst✝² : CommRing B₂
    inst✝¹ : Algebra A₁ B₁
    inst✝ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    x : B₁
    this : Algebra A₁ B₂ := ((↑e₂).comp (algebraMap A₁ B₁)).toAlgebra
    e' : AlgEquiv A₁ B₁ B₂ := { toEquiv := e₂.toEquiv, map_mul' := ⋯, map_add' :=  …
    ⊢ Eq ((Algebra.trace A₁ B₂) (e' x)) ((Algebra.trace A₁ B₂) (e₂ x))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem trace_eq_sum_embeddings_gen (pb : PowerBasis K L)
    (hE : (minpoly K pb.gen).Splits (algebraMap K E)) (hfx : IsSeparable K pb.gen) :
    algebraMap K E (Algebra.trace K L pb.gen) =
      (@Finset.univ _ (PowerBasis.AlgHom.fintype pb)).sum fun σ => σ pb.gen := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    E : Type u_7
    inst✝¹ : Field E
    inst✝ : Algebra K E
    pb : PowerBasis K L
    hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
    hfx : IsSeparable K pb.gen
    ⊢ Eq ((algebraMap K E) ((Algebra.trace K L) pb.gen)) (Finset.univ.sum fun σ => …
  -/
  letI := Classical.decEq E
  -- Porting note: the following `letI` was not needed.
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    E : Type u_7
    inst✝¹ : Field E
    inst✝ : Algebra K E
    pb : PowerBasis K L
    hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
    hfx : IsSeparable K pb.gen
    this : DecidableEq E := Classical.decEq E
    ⊢ Eq ((algebraMap K E) ((Algebra.trace K L) pb.gen)) (Finset.univ.sum fun σ => …
  -/
  letI : Fintype (L →ₐ[K] E) := PowerBasis.AlgHom.fintype pb
  rw [pb.trace_gen_eq_sum_roots hE, Fintype.sum_equiv pb.liftEquiv', Finset.sum_mem_multiset,
    Finset.sum_eq_multiset_sum, Multiset.toFinset_val, Multiset.dedup_eq_self.mpr _,
    Multiset.map_id]
    /-
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      E : Type u_7
      inst✝¹ : Field E
      inst✝ : Algebra K E
      pb : PowerBasis K L
      hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
      hfx : IsSeparable K pb.gen
      this✝ : DecidableEq E := Classical.decEq E
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      ⊢ ((minpoly K pb.gen).aroots E).Nodup
    -/
  · exact nodup_roots ((separable_map _).mpr hfx)
    /-
      🎉 no goals
    -/
  -- Porting note: the following goal does not exist in mathlib3.
    /-
      case f
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      E : Type u_7
      inst✝¹ : Field E
      inst✝ : Algebra K E
      pb : PowerBasis K L
      hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
      hfx : IsSeparable K pb.gen
      this✝ : DecidableEq E := Classical.decEq E
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      ⊢ (Subtype fun x => Membership.mem ((minpoly K pb.gen).aroots E) x) → E
    -/
  · exact (fun x => x.1)
    /-
      🎉 no goals
    -/
    /-
      case hfg
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      E : Type u_7
      inst✝¹ : Field E
      inst✝ : Algebra K E
      pb : PowerBasis K L
      hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
      hfx : IsSeparable K pb.gen
      this✝ : DecidableEq E := Classical.decEq E
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      ⊢ ∀ (x : Subtype fun x => Membership.mem ((minpoly K pb.gen).aroots E) x), Eq  …
    -/
  · intro x; rfl
             /-
               🎉 no goals
             -/
    /-
      case h
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      E : Type u_7
      inst✝¹ : Field E
      inst✝ : Algebra K E
      pb : PowerBasis K L
      hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
      hfx : IsSeparable K pb.gen
      this✝ : DecidableEq E := Classical.decEq E
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      ⊢ ∀ (x : AlgHom K L E), Eq (x pb.gen) ↑(pb.liftEquiv' x)
    -/
  · intro σ
    /-
      case h
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      E : Type u_7
      inst✝¹ : Field E
      inst✝ : Algebra K E
      pb : PowerBasis K L
      hE : Polynomial.Splits (algebraMap K E) (minpoly K pb.gen)
      hfx : IsSeparable K pb.gen
      this✝ : DecidableEq E := Classical.decEq E
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      σ : AlgHom K L E
      ⊢ Eq (σ pb.gen) ↑(pb.liftEquiv' σ)
    -/
    rw [PowerBasis.liftEquiv'_apply_coe]
    /-
      🎉 no goals
    -/


theorem sum_embeddings_eq_finrank_mul [FiniteDimensional K F] [Algebra.IsSeparable K F]
    (pb : PowerBasis K L) :
    ∑ σ : F →ₐ[K] E, σ (algebraMap L F pb.gen) =
      finrank L F •
        (@Finset.univ _ (PowerBasis.AlgHom.fintype pb)).sum fun σ : L →ₐ[K] E => σ pb.gen := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra K L
    F : Type u_6
    inst✝⁸ : Field F
    inst✝⁷ : Algebra L F
    inst✝⁶ : Algebra K F
    inst✝⁵ : IsScalarTower K L F
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : IsAlgClosed E
    inst✝¹ : FiniteDimensional K F
    inst✝ : Algebra.IsSeparable K F
    pb : PowerBasis K L
    ⊢ Eq (Finset.univ.sum fun σ => σ ((algebraMap L F) pb.gen)) (HSMul.hSMul (Modu …
  -/
  haveI : FiniteDimensional L F := FiniteDimensional.right K L F
  /-
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra K L
    F : Type u_6
    inst✝⁸ : Field F
    inst✝⁷ : Algebra L F
    inst✝⁶ : Algebra K F
    inst✝⁵ : IsScalarTower K L F
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : IsAlgClosed E
    inst✝¹ : FiniteDimensional K F
    inst✝ : Algebra.IsSeparable K F
    pb : PowerBasis K L
    this : FiniteDimensional L F
    ⊢ Eq (Finset.univ.sum fun σ => σ ((algebraMap L F) pb.gen)) (HSMul.hSMul (Modu …
  -/
  haveI : Algebra.IsSeparable L F := Algebra.isSeparable_tower_top_of_isSeparable K L F
  /-
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra K L
    F : Type u_6
    inst✝⁸ : Field F
    inst✝⁷ : Algebra L F
    inst✝⁶ : Algebra K F
    inst✝⁵ : IsScalarTower K L F
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : IsAlgClosed E
    inst✝¹ : FiniteDimensional K F
    inst✝ : Algebra.IsSeparable K F
    pb : PowerBasis K L
    this✝ : FiniteDimensional L F
    this : Algebra.IsSeparable L F
    ⊢ Eq (Finset.univ.sum fun σ => σ ((algebraMap L F) pb.gen)) (HSMul.hSMul (Modu …
  -/
  letI : Fintype (L →ₐ[K] E) := PowerBasis.AlgHom.fintype pb
  /-
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Algebra K L
    F : Type u_6
    inst✝⁸ : Field F
    inst✝⁷ : Algebra L F
    inst✝⁶ : Algebra K F
    inst✝⁵ : IsScalarTower K L F
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : IsAlgClosed E
    inst✝¹ : FiniteDimensional K F
    inst✝ : Algebra.IsSeparable K F
    pb : PowerBasis K L
    this✝¹ : FiniteDimensional L F
    this✝ : Algebra.IsSeparable L F
    this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
    ⊢ Eq (Finset.univ.sum fun σ => σ ((algebraMap L F) pb.gen)) (HSMul.hSMul (Modu …
  -/
  letI : ∀ f : L →ₐ[K] E, Fintype (haveI := f.toRingHom.toAlgebra; AlgHom L F E) := ?_
  · rw [Fintype.sum_equiv algHomEquivSigma (fun σ : F →ₐ[K] E => _) fun σ => σ.1 pb.gen, ←
      Finset.univ_sigma_univ, Finset.sum_sigma, ← Finset.sum_nsmul]
      /-
        case refine_2
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝² : FiniteDimensional L F
        this✝¹ : Algebra.IsSeparable L F
        this✝ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this : (f : AlgHom K L E) → Fintype (AlgHom L F E) := ?refine_1
        ⊢ Eq (Finset.univ.sum fun a => Finset.univ.sum fun s => ⟨a, s⟩.fst pb.gen) (Fi …
      -/
    · refine Finset.sum_congr rfl fun σ _ => ?_
      /-
        case refine_2
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝² : FiniteDimensional L F
        this✝¹ : Algebra.IsSeparable L F
        this✝ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this : (f : AlgHom K L E) → Fintype (AlgHom L F E) := ?refine_1
        σ : AlgHom K L E
        x✝ : Membership.mem Finset.univ σ
        ⊢ Eq (Finset.univ.sum fun s => ⟨σ, s⟩.fst pb.gen) (HSMul.hSMul (Module.finrank …
      -/
      letI : Algebra L E := σ.toRingHom.toAlgebra
      -- Porting note: `Finset.card_univ` was inside `simp only`.
      /-
        case refine_2
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝³ : FiniteDimensional L F
        this✝² : Algebra.IsSeparable L F
        this✝¹ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this✝ : (f : AlgHom K L E) → Fintype (AlgHom L F E) := ?refine_1
        σ : AlgHom K L E
        x✝ : Membership.mem Finset.univ σ
        this : Algebra L E := σ.toAlgebra
        ⊢ Eq (Finset.univ.sum fun s => ⟨σ, s⟩.fst pb.gen) (HSMul.hSMul (Module.finrank …
      -/
      simp only [Finset.sum_const]
      /-
        case refine_2
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝³ : FiniteDimensional L F
        this✝² : Algebra.IsSeparable L F
        this✝¹ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this✝ : (f : AlgHom K L E) → Fintype (AlgHom L F E) := ?refine_1
        σ : AlgHom K L E
        x✝ : Membership.mem Finset.univ σ
        this : Algebra L E := σ.toAlgebra
        ⊢ Eq (HSMul.hSMul Finset.univ.card (σ pb.gen)) (HSMul.hSMul (Module.finrank L  …
      -/
      congr
      /-
        case refine_2.e_a
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝³ : FiniteDimensional L F
        this✝² : Algebra.IsSeparable L F
        this✝¹ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this✝ : (f : AlgHom K L E) → Fintype (AlgHom L F E) := ?refine_1
        σ : AlgHom K L E
        x✝ : Membership.mem Finset.univ σ
        this : Algebra L E := σ.toAlgebra
        ⊢ Eq Finset.univ.card (Module.finrank L F)
      -/
      rw [← AlgHom.card L F E]
      /-
        case refine_2.e_a
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝³ : FiniteDimensional L F
        this✝² : Algebra.IsSeparable L F
        this✝¹ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this✝ : (f : AlgHom K L E) → Fintype (AlgHom L F E) := ?refine_1
        σ : AlgHom K L E
        x✝ : Membership.mem Finset.univ σ
        this : Algebra L E := σ.toAlgebra
        ⊢ Eq Finset.univ.card (Fintype.card (AlgHom L F E))
      -/
      exact Finset.card_univ (α := F →ₐ[L] E)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        K : Type u_4
        L : Type u_5
        inst✝¹¹ : Field K
        inst✝¹⁰ : Field L
        inst✝⁹ : Algebra K L
        F : Type u_6
        inst✝⁸ : Field F
        inst✝⁷ : Algebra L F
        inst✝⁶ : Algebra K F
        inst✝⁵ : IsScalarTower K L F
        E : Type u_7
        inst✝⁴ : Field E
        inst✝³ : Algebra K E
        inst✝² : IsAlgClosed E
        inst✝¹ : FiniteDimensional K F
        inst✝ : Algebra.IsSeparable K F
        pb : PowerBasis K L
        this✝² : FiniteDimensional L F
        this✝¹ : Algebra.IsSeparable L F
        this✝ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
        this : (f : AlgHom K L E) → Fintype (AlgHom L F E) := fun σ => minpoly.AlgHom. …
        ⊢ ∀ (x : AlgHom K F E), Eq (x ((algebraMap L F) pb.gen)) ((algHomEquivSigma x) …
      -/
    · intro σ
      simp only [algHomEquivSigma, Equiv.coe_fn_mk, AlgHom.restrictDomain, AlgHom.comp_apply,
        IsScalarTower.coe_toAlgHom']


theorem trace_eq_sum_embeddings [FiniteDimensional K L] [Algebra.IsSeparable K L] {x : L} :
    algebraMap K E (Algebra.trace K L x) = ∑ σ : L →ₐ[K] E, σ x := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Algebra K L
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : IsAlgClosed E
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    ⊢ Eq ((algebraMap K E) ((Algebra.trace K L) x)) (Finset.univ.sum fun σ => σ x)
  -/
  have hx := Algebra.IsSeparable.isIntegral K x
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Algebra K L
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : IsAlgClosed E
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    hx : IsIntegral K x
    ⊢ Eq ((algebraMap K E) ((Algebra.trace K L) x)) (Finset.univ.sum fun σ => σ x)
  -/
  let pb := adjoin.powerBasis hx
  rw [trace_eq_trace_adjoin K x, Algebra.smul_def, RingHom.map_mul, ← adjoin.powerBasis_gen hx,
    trace_eq_sum_embeddings_gen E pb (IsAlgClosed.splits_codomain _), ← Algebra.smul_def,
    algebraMap_smul]
    /-
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      E : Type u_7
      inst✝⁴ : Field E
      inst✝³ : Algebra K E
      inst✝² : IsAlgClosed E
      inst✝¹ : FiniteDimensional K L
      inst✝ : Algebra.IsSeparable K L
      x : L
      hx : IsIntegral K x
      pb : PowerBasis K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin …
      ⊢ Eq (HSMul.hSMul (Module.finrank (Subtype fun x_1 => Membership.mem (Intermed …
    -/
  · exact (sum_embeddings_eq_finrank_mul L E pb).symm
    /-
      🎉 no goals
    -/
    /-
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      E : Type u_7
      inst✝⁴ : Field E
      inst✝³ : Algebra K E
      inst✝² : IsAlgClosed E
      inst✝¹ : FiniteDimensional K L
      inst✝ : Algebra.IsSeparable K L
      x : L
      hx : IsIntegral K x
      pb : PowerBasis K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin …
      ⊢ IsSeparable K pb.gen
    -/
  · haveI := Algebra.isSeparable_tower_bot_of_isSeparable K K⟮x⟯ L
    /-
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      E : Type u_7
      inst✝⁴ : Field E
      inst✝³ : Algebra K E
      inst✝² : IsAlgClosed E
      inst✝¹ : FiniteDimensional K L
      inst✝ : Algebra.IsSeparable K L
      x : L
      hx : IsIntegral K x
      pb : PowerBasis K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin …
      this : Algebra.IsSeparable K (Subtype fun x_1 => Membership.mem (IntermediateF …
      ⊢ IsSeparable K pb.gen
    -/
    exact Algebra.IsSeparable.isSeparable K _
    /-
      🎉 no goals
    -/


theorem trace_eq_sum_automorphisms (x : L) [FiniteDimensional K L] [IsGalois K L] :
    algebraMap K L (Algebra.trace K L x) = ∑ σ : L ≃ₐ[K] L, σ x := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    x : L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsGalois K L
    ⊢ Eq ((algebraMap K L) ((Algebra.trace K L) x)) (Finset.univ.sum fun σ => σ x)
  -/
  apply NoZeroSMulDivisors.algebraMap_injective L (AlgebraicClosure L)
  /-
    case a
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    x : L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsGalois K L
    ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.trace K  …
  -/
  rw [_root_.map_sum (algebraMap L (AlgebraicClosure L))]
  /-
    case a
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    x : L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsGalois K L
    ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.trace K  …
  -/
  rw [← Fintype.sum_equiv (Normal.algHomEquivAut K (AlgebraicClosure L) L)]
    /-
      case a
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      x : L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsGalois K L
      ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.trace K  …
    -/
  · rw [← trace_eq_sum_embeddings (AlgebraicClosure L) (x := x)]
    /-
      case a
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      x : L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsGalois K L
      ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.trace K  …
    -/
    simp only [algebraMap_eq_smul_one, smul_one_smul]
    /-
      🎉 no goals
    -/
    /-
      case a.h
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      x : L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsGalois K L
      ⊢ ∀ (x_1 : AlgHom K L (AlgebraicClosure L)), Eq (x_1 x) ((algebraMap L (Algebr …
    -/
  · intro σ
    simp only [Normal.algHomEquivAut, AlgHom.restrictNormal', Equiv.coe_fn_mk,
      AlgEquiv.coe_ofBijective, AlgHom.restrictNormal_commutes, id.map_eq_id, RingHom.id_apply]


/-- Given an `A`-algebra `B` and `b`, a `κ`-indexed family of elements of `B`, we define
`traceMatrix A b` as the matrix whose `(i j)`-th element is the trace of `b i * b j`. -/
noncomputable def traceMatrix (b : κ → B) : Matrix κ κ A :=
  of fun i j => traceForm A B (b i) (b j)

-- TODO: set as an equation lemma for `traceMatrix`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem traceMatrix_apply (b : κ → B) (i j) : traceMatrix A b i j = traceForm A B (b i) (b j) :=
  rfl


theorem traceMatrix_reindex {κ' : Type*} (b : Basis κ A B) (f : κ ≃ κ') :
                                                                      /-
                                                                        κ : Type w
                                                                        A : Type u
                                                                        B : Type v
                                                                        inst✝² : CommRing A
                                                                        inst✝¹ : CommRing B
                                                                        inst✝ : Algebra A B
                                                                        κ' : Type u_7
                                                                        b : Basis κ A B
                                                                        f : Equiv κ κ'
                                                                        ⊢ Eq (Algebra.traceMatrix A ⇑(b.reindex f)) ((Matrix.reindex f f) (Algebra.tra …
                                                                      -/
    traceMatrix A (b.reindex f) = reindex f f (traceMatrix A b) := by ext (x y); simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem traceMatrix_of_matrix_vecMul [Fintype κ] (b : κ → B) (P : Matrix κ κ A) :
    traceMatrix A (b ᵥ* P.map (algebraMap A B)) = Pᵀ * traceMatrix A b * P := by
  /-
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    ⊢ Eq (Algebra.traceMatrix A (Matrix.vecMul b (P.map ⇑(algebraMap A B)))) (HMul …
  -/
  ext (α β)
  rw [traceMatrix_apply, vecMul, dotProduct, vecMul, dotProduct, Matrix.mul_apply,
    BilinForm.sum_left,
    Fintype.sum_congr _ _ fun i : κ =>
      BilinForm.sum_right _ _ (b i * P.map (algebraMap A B) i α) fun y : κ =>
        b y * P.map (algebraMap A B) y β,
    sum_comm]
  /-
    case a
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β : κ
    ⊢ Eq (Finset.univ.sum fun y => Finset.univ.sum fun x => ((Algebra.traceForm A  …
  -/
  congr; ext x
  /-
    case a.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x : κ
    ⊢ Eq (Finset.univ.sum fun x_1 => ((Algebra.traceForm A B) (HMul.hMul (b x_1) ( …
  -/
  rw [Matrix.mul_apply, sum_mul]
  /-
    case a.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x : κ
    ⊢ Eq (Finset.univ.sum fun x_1 => ((Algebra.traceForm A B) (HMul.hMul (b x_1) ( …
  -/
  congr; ext y
  /-
    case a.e_f.h.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x y : κ
    ⊢ Eq (((Algebra.traceForm A B) (HMul.hMul (b y) (P.map (⇑(algebraMap A B)) y α …
  -/
  rw [map_apply, traceForm_apply, mul_comm (b y), ← smul_def]
  simp only [id.smul_eq_mul, RingHom.id_apply, map_apply, transpose_apply, LinearMap.map_smulₛₗ,
    traceForm_apply, Algebra.smul_mul_assoc]
  /-
    case a.e_f.h.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x y : κ
    ⊢ Eq (HMul.hMul (P y α) ((Algebra.trace A B) (HMul.hMul (b y) (HMul.hMul (b x) …
  -/
  rw [mul_comm (b x), ← smul_def]
  /-
    case a.e_f.h.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x y : κ
    ⊢ Eq (HMul.hMul (P y α) ((Algebra.trace A B) (HMul.hMul (b y) (HSMul.hSMul (P  …
  -/
  ring_nf
  /-
    case a.e_f.h.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x y : κ
    ⊢ Eq (HMul.hMul (P y α) ((Algebra.trace A B) (HMul.hMul (b y) (HSMul.hSMul (P  …
  -/
  rw [mul_assoc]
  /-
    case a.e_f.h.e_f.h
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    α β x y : κ
    ⊢ Eq (HMul.hMul (P y α) ((Algebra.trace A B) (HMul.hMul (b y) (HSMul.hSMul (P  …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem traceMatrix_of_matrix_mulVec [Fintype κ] (b : κ → B) (P : Matrix κ κ A) :
    traceMatrix A (P.map (algebraMap A B) *ᵥ b) = P * traceMatrix A b * Pᵀ := by
  /-
    κ : Type w
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Fintype κ
    b : κ → B
    P : Matrix κ κ A
    ⊢ Eq (Algebra.traceMatrix A ((P.map ⇑(algebraMap A B)).mulVec b)) (HMul.hMul ( …
  -/
  refine AddEquiv.injective (transposeAddEquiv κ κ A) ?_
  rw [transposeAddEquiv_apply, transposeAddEquiv_apply, ← vecMul_transpose, ← transpose_map,
    traceMatrix_of_matrix_vecMul, transpose_transpose]


theorem traceMatrix_of_basis [Fintype κ] [DecidableEq κ] (b : Basis κ A B) :
    traceMatrix A b = BilinForm.toMatrix b (traceForm A B) := by
  /-
    κ : Type w
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : Fintype κ
    inst✝ : DecidableEq κ
    b : Basis κ A B
    ⊢ Eq (Algebra.traceMatrix A ⇑b) ((BilinForm.toMatrix b) (Algebra.traceForm A B))
  -/
  ext (i j)
  /-
    case a
    κ : Type w
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : Fintype κ
    inst✝ : DecidableEq κ
    b : Basis κ A B
    i j : κ
    ⊢ Eq (Algebra.traceMatrix A (⇑b) i j) ((BilinForm.toMatrix b) (Algebra.traceFo …
  -/
  rw [traceMatrix_apply, traceForm_apply, traceForm_toMatrix]
  /-
    🎉 no goals
  -/


theorem traceMatrix_of_basis_mulVec (b : Basis ι A B) (z : B) :
    traceMatrix A b *ᵥ b.equivFun z = fun i => trace A B (z * b i) := by
  /-
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    ⊢ Eq ((Algebra.traceMatrix A ⇑b).mulVec (b.equivFun z)) fun i => (Algebra.trac …
  -/
  ext i
  rw [← col_apply (ι := Fin 1) (traceMatrix A b *ᵥ b.equivFun z) i 0, col_mulVec,
    Matrix.mul_apply, traceMatrix]
  /-
    case h
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    i : ι
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul (Matrix.of (fun i j => ((Algebra.trac …
  -/
  simp only [col_apply, traceForm_apply]
  conv_lhs =>
    congr
    rfl
    ext
    rw [mul_comm _ (b.equivFun z _), ← smul_eq_mul, of_apply, ← LinearMap.map_smul]
  /-
    case h
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    i : ι
    ⊢ Eq (Finset.univ.sum fun x => (Algebra.trace A B) (HSMul.hSMul (b.equivFun z  …
  -/
  rw [← _root_.map_sum]
  /-
    case h
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    i : ι
    ⊢ Eq ((Algebra.trace A B) (Finset.univ.sum fun x => HSMul.hSMul (b.equivFun z  …
  -/
  congr
  conv_lhs =>
    congr
    rfl
    ext
    rw [← mul_smul_comm]
  /-
    case h.h.e_6.h
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    i : ι
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (b i) (HSMul.hSMul (b.equivFun z x) ( …
  -/
  rw [← Finset.mul_sum, mul_comm z]
  /-
    case h.h.e_6.h
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    i : ι
    ⊢ Eq (HMul.hMul (b i) (Finset.univ.sum fun i => HSMul.hSMul (b.equivFun z i) ( …
  -/
  congr
  /-
    case h.h.e_6.h.e_a
    ι : Type w
    inst✝³ : Fintype ι
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    b : Basis ι A B
    z : B
    i : ι
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (b.equivFun z i) (b i)) z
  -/
  rw [b.sum_equivFun]
  /-
    🎉 no goals
  -/


/-- `embeddingsMatrix A C b : Matrix κ (B →ₐ[A] C) C` is the matrix whose `(i, σ)` coefficient is
  `σ (b i)`. It is mostly useful for fields when `Fintype.card κ = finrank A B` and `C` is
  algebraically closed. -/
def embeddingsMatrix (b : κ → B) : Matrix κ (B →ₐ[A] C) C :=
  of fun i (σ : B →ₐ[A] C) => σ (b i)

-- TODO: set as an equation lemma for `embeddingsMatrix`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem embeddingsMatrix_apply (b : κ → B) (i) (σ : B →ₐ[A] C) :
    embeddingsMatrix A C b i σ = σ (b i) :=
  rfl


/-- `embeddingsMatrixReindex A C b e : Matrix κ κ C` is the matrix whose `(i, j)` coefficient
  is `σⱼ (b i)`, where `σⱼ : B →ₐ[A] C` is the embedding corresponding to `j : κ` given by a
  bijection `e : κ ≃ (B →ₐ[A] C)`. It is mostly useful for fields and `C` is algebraically closed.
  In this case, in presence of `h : Fintype.card κ = finrank A B`, one can take
  `e := equivOfCardEq ((AlgHom.card A B C).trans h.symm)`. -/
def embeddingsMatrixReindex (b : κ → B) (e : κ ≃ (B →ₐ[A] C)) :=
  reindex (Equiv.refl κ) e.symm (embeddingsMatrix A C b)


theorem embeddingsMatrixReindex_eq_vandermonde (pb : PowerBasis A B)
    (e : Fin pb.dim ≃ (B →ₐ[A] C)) :
    embeddingsMatrixReindex A C pb.basis e = (vandermonde fun i => e i pb.gen)ᵀ := by
  /-
    A : Type u
    B : Type v
    C : Type z
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : CommRing C
    inst✝ : Algebra A C
    pb : PowerBasis A B
    e : Equiv (Fin pb.dim) (AlgHom A B C)
    ⊢ Eq (Algebra.embeddingsMatrixReindex A C (⇑pb.basis) e) (Matrix.vandermonde f …
  -/
  ext i j
  /-
    case a
    A : Type u
    B : Type v
    C : Type z
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : CommRing C
    inst✝ : Algebra A C
    pb : PowerBasis A B
    e : Equiv (Fin pb.dim) (AlgHom A B C)
    i j : Fin pb.dim
    ⊢ Eq (Algebra.embeddingsMatrixReindex A C (⇑pb.basis) e i j) ((Matrix.vandermo …
  -/
  simp [embeddingsMatrixReindex, embeddingsMatrix]
  /-
    🎉 no goals
  -/


theorem traceMatrix_eq_embeddingsMatrix_mul_trans : (traceMatrix K b).map (algebraMap K E) =
    embeddingsMatrix K E b * (embeddingsMatrix K E b)ᵀ := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Algebra K L
    κ : Type w
    E : Type z
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : Module.Finite K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : IsAlgClosed E
    b : κ → L
    ⊢ Eq ((Algebra.traceMatrix K b).map ⇑(algebraMap K E)) (HMul.hMul (Algebra.emb …
  -/
  ext (i j); simp [trace_eq_sum_embeddings, embeddingsMatrix, Matrix.mul_apply]
             /-
               🎉 no goals
             -/


theorem traceMatrix_eq_embeddingsMatrixReindex_mul_trans [Fintype κ] (e : κ ≃ (L →ₐ[K] E)) :
    (traceMatrix K b).map (algebraMap K E) =
      embeddingsMatrixReindex K E b e * (embeddingsMatrixReindex K E b e)ᵀ := by
  rw [traceMatrix_eq_embeddingsMatrix_mul_trans, embeddingsMatrixReindex, reindex_apply,
    transpose_submatrix, ← submatrix_mul_transpose_submatrix, ← Equiv.coe_refl, Equiv.refl_symm]


theorem det_traceMatrix_ne_zero' [Algebra.IsSeparable K L] : det (traceMatrix K pb.basis) ≠ 0 := by
  suffices algebraMap K (AlgebraicClosure L) (det (traceMatrix K pb.basis)) ≠ 0 by
    refine mt (fun ht => ?_) this
    rw [ht, RingHom.map_zero]
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Ne ((algebraMap K (AlgebraicClosure L)) (Algebra.traceMatrix K ⇑pb.basis).de …
  -/
  haveI : FiniteDimensional K L := pb.finite
  /-
    K : Type u_4
    L : Type u_5
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    pb : PowerBasis K L
    inst✝ : Algebra.IsSeparable K L
    this : FiniteDimensional K L
    ⊢ Ne ((algebraMap K (AlgebraicClosure L)) (Algebra.traceMatrix K ⇑pb.basis).de …
  -/
  let e : Fin pb.dim ≃ (L →ₐ[K] AlgebraicClosure L) := (Fintype.equivFinOfCardEq ?_).symm
  · rw [RingHom.map_det, RingHom.mapMatrix_apply,
      traceMatrix_eq_embeddingsMatrixReindex_mul_trans K _ _ e,
      embeddingsMatrixReindex_eq_vandermonde, det_mul, det_transpose]
    /-
      case refine_2
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      this : FiniteDimensional K L
      e : Equiv (Fin pb.dim) (AlgHom K L (AlgebraicClosure L)) := (Fintype.equivFinO …
      ⊢ Ne (HMul.hMul (Matrix.vandermonde fun i => (e i) pb.gen).det (Matrix.vanderm …
    -/
    refine mt mul_self_eq_zero.mp ?_
    /-
      case refine_2
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      this : FiniteDimensional K L
      e : Equiv (Fin pb.dim) (AlgHom K L (AlgebraicClosure L)) := (Fintype.equivFinO …
      ⊢ Not (Eq (Matrix.vandermonde fun i => (e i) pb.gen).det 0)
    -/
    simp only [det_vandermonde, Finset.prod_eq_zero_iff, not_exists, sub_eq_zero]
    /-
      case refine_2
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      this : FiniteDimensional K L
      e : Equiv (Fin pb.dim) (AlgHom K L (AlgebraicClosure L)) := (Fintype.equivFinO …
      ⊢ ∀ (x : Fin pb.dim), Not (And (Membership.mem Finset.univ x) (Exists fun a => …
    -/
    rintro i ⟨_, j, hij, h⟩
    /-
      case refine_2.intro.intro.intro
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      this : FiniteDimensional K L
      e : Equiv (Fin pb.dim) (AlgHom K L (AlgebraicClosure L)) := (Fintype.equivFinO …
      i : Fin pb.dim
      left✝ : Membership.mem Finset.univ i
      j : Fin pb.dim
      hij : Membership.mem (Finset.Ioi i) j
      h : Eq ((e j) pb.gen) ((e i) pb.gen)
      ⊢ False
    -/
    exact (Finset.mem_Ioi.mp hij).ne' (e.injective <| pb.algHom_ext h)
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      K : Type u_4
      L : Type u_5
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      pb : PowerBasis K L
      inst✝ : Algebra.IsSeparable K L
      this : FiniteDimensional K L
      ⊢ Eq (Fintype.card (AlgHom K L (AlgebraicClosure L))) pb.dim
    -/
  · rw [AlgHom.card, pb.finrank]
    /-
      🎉 no goals
    -/


theorem det_traceForm_ne_zero [Algebra.IsSeparable K L] [DecidableEq ι] (b : Basis ι K L) :
    det (BilinForm.toMatrix b (traceForm K L)) ≠ 0 := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    ι : Type w
    inst✝² : Fintype ι
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : DecidableEq ι
    b : Basis ι K L
    ⊢ Ne ((BilinForm.toMatrix b) (Algebra.traceForm K L)).det 0
  -/
  haveI : FiniteDimensional K L := FiniteDimensional.of_fintype_basis b
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    ι : Type w
    inst✝² : Fintype ι
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : DecidableEq ι
    b : Basis ι K L
    this : FiniteDimensional K L
    ⊢ Ne ((BilinForm.toMatrix b) (Algebra.traceForm K L)).det 0
  -/
  let pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable _ _
  rw [← BilinForm.toMatrix_mul_basis_toMatrix pb.basis b, ←
    det_comm' (pb.basis.toMatrix_mul_toMatrix_flip b) _, ← Matrix.mul_assoc, det_mul]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    ι : Type w
    inst✝² : Fintype ι
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : DecidableEq ι
    b : Basis ι K L
    this : FiniteDimensional K L
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ Ne (HMul.hMul (HMul.hMul (pb.basis.toMatrix ⇑b) (pb.basis.toMatrix ⇑b).trans …
  -/
  swap; · apply Basis.toMatrix_mul_toMatrix_flip
          /-
            🎉 no goals
          -/
  refine
    mul_ne_zero
      (isUnit_of_mul_eq_one _ ((b.toMatrix pb.basis)ᵀ * b.toMatrix pb.basis).det ?_).ne_zero ?_
  · calc
      (pb.basis.toMatrix b * (pb.basis.toMatrix b)ᵀ).det *
            ((b.toMatrix pb.basis)ᵀ * b.toMatrix pb.basis).det =
          (pb.basis.toMatrix b * (b.toMatrix pb.basis * pb.basis.toMatrix b)ᵀ *
              b.toMatrix pb.basis).det := by
        simp only [← det_mul, Matrix.mul_assoc, Matrix.transpose_mul]
      _ = 1 := by
        simp only [Basis.toMatrix_mul_toMatrix_flip, Matrix.transpose_one, Matrix.mul_one,
          Matrix.det_one]
  /-
    case refine_2
    K : Type u_4
    L : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : Algebra K L
    ι : Type w
    inst✝² : Fintype ι
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : DecidableEq ι
    b : Basis ι K L
    this : FiniteDimensional K L
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ Ne ((BilinForm.toMatrix pb.basis) (Algebra.traceForm K L)).det 0
  -/
  simpa only [traceMatrix_of_basis] using det_traceMatrix_ne_zero' pb
  /-
    🎉 no goals
  -/


/-- Let $L/K$ be a finite extension of fields. If $L/K$ is separable,
then `traceForm` is nondegenerate. -/
@[stacks 0BIL "(1) => (3)"]
theorem traceForm_nondegenerate [FiniteDimensional K L] [Algebra.IsSeparable K L] :
    (traceForm K L).Nondegenerate :=
  BilinForm.nondegenerate_of_det_ne_zero (traceForm K L) _
    (det_traceForm_ne_zero (Module.finBasis K L))


theorem Algebra.trace_ne_zero [FiniteDimensional K L] [Algebra.IsSeparable K L] :
    Algebra.trace K L ≠ 0 := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Ne (Algebra.trace K L) 0
  -/
  intro e
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    e : Eq (Algebra.trace K L) 0
    ⊢ False
  -/
  let pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable _ _
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    e : Eq (Algebra.trace K L) 0
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ False
  -/
  apply det_traceMatrix_ne_zero' pb
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    e : Eq (Algebra.trace K L) 0
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ Eq (Algebra.traceMatrix K ⇑pb.basis).det 0
  -/
  rw [show traceMatrix K pb.basis = 0 by ext; simp [e], Matrix.det_zero]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    e : Eq (Algebra.trace K L) 0
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ Nonempty (Fin pb.dim)
  -/
  rw [← pb.finrank, ← Fin.pos_iff_nonempty]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    e : Eq (Algebra.trace K L) 0
    pb : PowerBasis K L := Field.powerBasisOfFiniteOfSeparable K L
    ⊢ LT.lt 0 (Module.finrank K L)
  -/
  exact finrank_pos
  /-
    🎉 no goals
  -/


theorem Algebra.trace_surjective [FiniteDimensional K L] [Algebra.IsSeparable K L] :
    Function.Surjective (Algebra.trace K L) := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Function.Surjective ⇑(Algebra.trace K L)
  -/
  rw [← LinearMap.range_eq_top]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Eq (LinearMap.range (Algebra.trace K L)) Top.top
  -/
  apply (IsSimpleOrder.eq_bot_or_eq_top (α := Ideal K) _).resolve_left
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Not (Eq (LinearMap.range (Algebra.trace K L)) Bot.bot)
  -/
  rw [LinearMap.range_eq_bot]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    ⊢ Not (Eq (Algebra.trace K L) 0)
  -/
  exact Algebra.trace_ne_zero K L
  /-
    🎉 no goals
  -/


/--
The dual basis of a powerbasis `{1, x, x²...}` under the trace form is `aᵢ / f'(x)`,
with `f` being the minimal polynomial of `x` and `f / (X - x) = ∑ aᵢxⁱ`.
-/
lemma traceForm_dualBasis_powerBasis_eq [FiniteDimensional K L] [Algebra.IsSeparable K L]
    (pb : PowerBasis K L) (i) :
    (Algebra.traceForm K L).dualBasis (traceForm_nondegenerate K L) pb.basis i =
      (minpolyDiv K pb.gen).coeff i / aeval pb.gen (derivative <| minpoly K pb.gen) := by
  classical
  apply ((Algebra.traceForm K L).toDual (traceForm_nondegenerate K L)).injective
  apply pb.basis.ext
  intro j
  simp only [BilinForm.toDual_def, BilinForm.apply_dualBasis_left]
  apply (algebraMap K (AlgebraicClosure K)).injective
  have := congr_arg (coeff · i) (sum_smul_minpolyDiv_eq_X_pow (AlgebraicClosure K)
    pb.adjoin_gen_eq_top (r := j) (pb.finrank.symm ▸ j.prop))
  simp only [AlgEquiv.toAlgHom_eq_coe, Polynomial.map_smul, map_div₀,
    map_pow, RingHom.coe_coe, AlgHom.coe_coe, finset_sum_coeff, coeff_smul, coeff_map, smul_eq_mul,
    coeff_X_pow, ← Fin.ext_iff, @eq_comm _ i] at this
  rw [PowerBasis.coe_basis]
  simp only [RingHom.map_ite_one_zero, traceForm_apply]
  rw [← this, trace_eq_sum_embeddings (E := AlgebraicClosure K)]
  apply Finset.sum_congr rfl
  intro σ _
  simp only [_root_.map_mul, map_div₀, map_pow]
  ring


/-- The trace of a nilpotent element is nilpotent. -/
lemma trace_isNilpotent_of_isNilpotent {R S : Type*} [CommRing R] [CommRing S] [Algebra R S] {x : S}
    (hx : IsNilpotent x) : IsNilpotent (trace R S x) :=
  LinearMap.isNilpotent_trace_of_isNilpotent (hx.map (lmul R S))


