/-- Given `pb : PowerBasis K S`, then the norm of `pb.gen` is
`(-1) ^ pb.dim * coeff (minpoly K pb.gen) 0`. -/
theorem PowerBasis.norm_gen_eq_coeff_zero_minpoly (pb : PowerBasis R S) :
    norm R pb.gen = (-1) ^ pb.dim * coeff (minpoly R pb.gen) 0 := by
  rw [norm_eq_matrix_det pb.basis, det_eq_sign_charpoly_coeff, charpoly_leftMulMatrix,
    Fintype.card_fin]


/-- Given `pb : PowerBasis R S`, then the norm of `pb.gen` is
`((minpoly R pb.gen).aroots F).prod`. -/
theorem PowerBasis.norm_gen_eq_prod_roots [Algebra R F] (pb : PowerBasis R S)
    (hf : (minpoly R pb.gen).Splits (algebraMap R F)) :
    algebraMap R F (norm R pb.gen) = ((minpoly R pb.gen).aroots F).prod := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra R F
    pb : PowerBasis R S
    hf : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
    ⊢ Eq ((algebraMap R F) ((Algebra.norm R) pb.gen)) ((minpoly R pb.gen).aroots F …
  -/
  haveI := Module.nontrivial R F
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra R F
    pb : PowerBasis R S
    hf : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
    this : Nontrivial R
    ⊢ Eq ((algebraMap R F) ((Algebra.norm R) pb.gen)) ((minpoly R pb.gen).aroots F …
  -/
  have := minpoly.monic pb.isIntegral_gen
  rw [PowerBasis.norm_gen_eq_coeff_zero_minpoly, ← pb.natDegree_minpoly, RingHom.map_mul,
    ← coeff_map,
    prod_roots_eq_coeff_zero_of_monic_of_splits (this.map _) ((splits_id_iff_splits _).2 hf),
    this.natDegree_map, map_pow, ← mul_assoc, ← mul_pow]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra R F
    pb : PowerBasis R S
    hf : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
    this✝ : Nontrivial R
    this : (minpoly R pb.gen).Monic
    ⊢ Eq (HMul.hMul (HPow.hPow (HMul.hMul ((algebraMap R F) (-1)) (-1)) (minpoly R …
  -/
  simp only [map_neg, _root_.map_one, neg_mul, neg_neg, one_pow, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_zero [Nontrivial S] [Module.Free R S] [Module.Finite R S] : norm R (0 : S) = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    inst✝² : Nontrivial S
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    ⊢ Eq ((Algebra.norm R) 0) 0
  -/
  nontriviality
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    inst✝² : Nontrivial S
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    a✝ : Nontrivial R
    ⊢ Eq ((Algebra.norm R) 0) 0
  -/
  rw [norm_apply, coe_lmul_eq_mul, map_zero, LinearMap.det_zero' (Module.Free.chooseBasis R S)]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_eq_zero_iff [IsDomain R] [IsDomain S] [Module.Free R S] [Module.Finite R S] {x : S} :
    norm R x = 0 ↔ x = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : Ring S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsDomain S
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    x : S
    ⊢ Iff (Eq ((Algebra.norm R) x) 0) (Eq x 0)
  -/
  constructor
  /-
    case mp
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : Ring S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsDomain S
    inst✝¹ : Module.Free R S
    inst✝ : Module.Finite R S
    x : S
    ⊢ Eq ((Algebra.norm R) x) 0 → Eq x 0
  -/
  swap
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsDomain R
      inst✝² : IsDomain S
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      x : S
      ⊢ Eq x 0 → Eq ((Algebra.norm R) x) 0
    -/
  · rintro rfl; exact norm_zero
                /-
                  🎉 no goals
                -/
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsDomain R
      inst✝² : IsDomain S
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      x : S
      ⊢ Eq ((Algebra.norm R) x) 0 → Eq x 0
    -/
  · let b := Module.Free.chooseBasis R S
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsDomain R
      inst✝² : IsDomain S
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      x : S
      b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
      ⊢ Eq ((Algebra.norm R) x) 0 → Eq x 0
    -/
    let decEq := Classical.decEq (Module.Free.ChooseBasisIndex R S)
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsDomain R
      inst✝² : IsDomain S
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      x : S
      b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
      decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
      ⊢ Eq ((Algebra.norm R) x) 0 → Eq x 0
    -/
    rw [norm_eq_matrix_det b, ← Matrix.exists_mulVec_eq_zero_iff]
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsDomain R
      inst✝² : IsDomain S
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      x : S
      b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
      decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
      ⊢ (Exists fun v => And (Ne v 0) (Eq (((Algebra.leftMulMatrix b) x).mulVec v) 0 …
    -/
    rintro ⟨v, v_ne, hv⟩
    rw [← b.equivFun.apply_symm_apply v, b.equivFun_symm_apply, b.equivFun_apply,
      leftMulMatrix_mulVec_repr] at hv
    /-
      case mp.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsDomain R
      inst✝² : IsDomain S
      inst✝¹ : Module.Free R S
      inst✝ : Module.Finite R S
      x : S
      b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
      decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
      v : Module.Free.ChooseBasisIndex R S → R
      v_ne : Ne v 0
      hv : Eq (⇑(b.repr (HMul.hMul x (Finset.univ.sum fun i => HSMul.hSMul (v i) (b  …
      ⊢ Eq x 0
    -/
    refine (mul_eq_zero.mp (b.ext_elem fun i => ?_)).resolve_right (show ∑ i, v i • b i ≠ 0 from ?_)
      /-
        case mp.intro.intro.refine_1
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : Ring S
        inst✝⁴ : Algebra R S
        inst✝³ : IsDomain R
        inst✝² : IsDomain S
        inst✝¹ : Module.Free R S
        inst✝ : Module.Finite R S
        x : S
        b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
        decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
        v : Module.Free.ChooseBasisIndex R S → R
        v_ne : Ne v 0
        hv : Eq (⇑(b.repr (HMul.hMul x (Finset.univ.sum fun i => HSMul.hSMul (v i) (b  …
        i : Module.Free.ChooseBasisIndex R S
        ⊢ Eq ((b.repr (HMul.hMul x (Finset.univ.sum fun i => HSMul.hSMul (v i) (b i))) …
      -/
    · simpa only [LinearEquiv.map_zero, Pi.zero_apply] using congr_fun hv i
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.refine_2
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : Ring S
        inst✝⁴ : Algebra R S
        inst✝³ : IsDomain R
        inst✝² : IsDomain S
        inst✝¹ : Module.Free R S
        inst✝ : Module.Finite R S
        x : S
        b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
        decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
        v : Module.Free.ChooseBasisIndex R S → R
        v_ne : Ne v 0
        hv : Eq (⇑(b.repr (HMul.hMul x (Finset.univ.sum fun i => HSMul.hSMul (v i) (b  …
        ⊢ Ne (Finset.univ.sum fun i => HSMul.hSMul (v i) (b i)) 0
      -/
    · contrapose! v_ne with sum_eq
      /-
        case mp.intro.intro.refine_2
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : Ring S
        inst✝⁴ : Algebra R S
        inst✝³ : IsDomain R
        inst✝² : IsDomain S
        inst✝¹ : Module.Free R S
        inst✝ : Module.Finite R S
        x : S
        b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
        decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
        v : Module.Free.ChooseBasisIndex R S → R
        hv : Eq (⇑(b.repr (HMul.hMul x (Finset.univ.sum fun i => HSMul.hSMul (v i) (b  …
        sum_eq : Eq (Finset.univ.sum fun i => HSMul.hSMul (v i) (b i)) 0
        ⊢ Eq v 0
      -/
      apply b.equivFun.symm.injective
      /-
        case mp.intro.intro.refine_2.a
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : Ring S
        inst✝⁴ : Algebra R S
        inst✝³ : IsDomain R
        inst✝² : IsDomain S
        inst✝¹ : Module.Free R S
        inst✝ : Module.Finite R S
        x : S
        b : Basis (Module.Free.ChooseBasisIndex R S) R S := Module.Free.chooseBasis R S
        decEq : DecidableEq (Module.Free.ChooseBasisIndex R S) := Classical.decEq (Mod …
        v : Module.Free.ChooseBasisIndex R S → R
        hv : Eq (⇑(b.repr (HMul.hMul x (Finset.univ.sum fun i => HSMul.hSMul (v i) (b  …
        sum_eq : Eq (Finset.univ.sum fun i => HSMul.hSMul (v i) (b i)) 0
        ⊢ Eq (b.equivFun.symm v) (b.equivFun.symm 0)
      -/
      rw [b.equivFun_symm_apply, sum_eq, LinearEquiv.map_zero]
      /-
        🎉 no goals
      -/


theorem norm_ne_zero_iff [IsDomain R] [IsDomain S] [Module.Free R S] [Module.Finite R S] {x : S} :
    norm R x ≠ 0 ↔ x ≠ 0 := not_iff_not.mpr norm_eq_zero_iff


/-- This is `Algebra.norm_eq_zero_iff` composed with `Algebra.norm_apply`. -/
@[simp]
theorem norm_eq_zero_iff' [IsDomain R] [IsDomain S] [Module.Free R S] [Module.Finite R S] {x : S} :
    LinearMap.det (LinearMap.mul R S x) = 0 ↔ x = 0 := norm_eq_zero_iff


theorem norm_eq_zero_iff_of_basis [IsDomain R] [IsDomain S] (b : Basis ι R S) {x : S} :
    Algebra.norm R x = 0 ↔ x = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    ι : Type w
    inst✝² : Finite ι
    inst✝¹ : IsDomain R
    inst✝ : IsDomain S
    b : Basis ι R S
    x : S
    ⊢ Iff (Eq ((Algebra.norm R) x) 0) (Eq x 0)
  -/
  haveI : Module.Free R S := Module.Free.of_basis b
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    ι : Type w
    inst✝² : Finite ι
    inst✝¹ : IsDomain R
    inst✝ : IsDomain S
    b : Basis ι R S
    x : S
    this : Module.Free R S
    ⊢ Iff (Eq ((Algebra.norm R) x) 0) (Eq x 0)
  -/
  haveI : Module.Finite R S := Module.Finite.of_basis b
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring S
    inst✝³ : Algebra R S
    ι : Type w
    inst✝² : Finite ι
    inst✝¹ : IsDomain R
    inst✝ : IsDomain S
    b : Basis ι R S
    x : S
    this✝ : Module.Free R S
    this : Module.Finite R S
    ⊢ Iff (Eq ((Algebra.norm R) x) 0) (Eq x 0)
  -/
  exact norm_eq_zero_iff
  /-
    🎉 no goals
  -/


theorem norm_ne_zero_iff_of_basis [IsDomain R] [IsDomain S] (b : Basis ι R S) {x : S} :
    Algebra.norm R x ≠ 0 ↔ x ≠ 0 :=
  not_iff_not.mpr (norm_eq_zero_iff_of_basis b)


theorem norm_eq_norm_adjoin [FiniteDimensional K L] [Algebra.IsSeparable K L] (x : L) :
    norm K x = norm K (AdjoinSimple.gen K x) ^ finrank K⟮x⟯ L := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    ⊢ Eq ((Algebra.norm K) x) (HPow.hPow ((Algebra.norm K) (IntermediateField.Adjo …
  -/
  letI := Algebra.isSeparable_tower_top_of_isSeparable K K⟮x⟯ L
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    this : Algebra.IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateFie …
    ⊢ Eq ((Algebra.norm K) x) (HPow.hPow ((Algebra.norm K) (IntermediateField.Adjo …
  -/
  let pbL := Field.powerBasisOfFiniteOfSeparable K⟮x⟯ L
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    this : Algebra.IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateFie …
    pbL : PowerBasis (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin  …
    ⊢ Eq ((Algebra.norm K) x) (HPow.hPow ((Algebra.norm K) (IntermediateField.Adjo …
  -/
  let pbx := IntermediateField.adjoin.powerBasis (Algebra.IsSeparable.isIntegral K x)
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [← AdjoinSimple.algebraMap_gen K x, norm_eq_matrix_det (pbx.basis.smulTower pbL.basis) _,
    smulTower_leftMulMatrix_algebraMap, det_blockDiagonal, norm_eq_matrix_det pbx.basis]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    this : Algebra.IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateFie …
    pbL : PowerBasis (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin  …
    pbx : PowerBasis K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoi …
    ⊢ Eq (Finset.univ.prod fun k => ((Algebra.leftMulMatrix pbx.basis) (Intermedia …
  -/
  simp only [Finset.card_fin, Finset.prod_const]
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    this : Algebra.IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateFie …
    pbL : PowerBasis (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin  …
    pbx : PowerBasis K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoi …
    ⊢ Eq (HPow.hPow ((Algebra.leftMulMatrix pbx.basis) (IntermediateField.AdjoinSi …
  -/
  congr
  /-
    case e_a
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : Algebra.IsSeparable K L
    x : L
    this : Algebra.IsSeparable (Subtype fun x_1 => Membership.mem (IntermediateFie …
    pbL : PowerBasis (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin  …
    pbx : PowerBasis K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoi …
    ⊢ Eq pbL.dim (Module.finrank (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rw [← PowerBasis.finrank, AdjoinSimple.algebraMap_gen K x]
  /-
    🎉 no goals
  -/


theorem _root_.IntermediateField.AdjoinSimple.norm_gen_eq_one {x : L} (hx : ¬IsIntegral K x) :
    norm K (AdjoinSimple.gen K x) = 1 := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    x : L
    hx : Not (IsIntegral K x)
    ⊢ Eq ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K x)) 1
  -/
  rw [norm_eq_one_of_not_exists_basis]
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
  · exact (Submodule.fg_iff_finiteDimensional _).mpr (.of_fintype_basis b)
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
  · exact IntermediateField.subset_adjoin K _ (Set.mem_singleton x)
    /-
      🎉 no goals
    -/


theorem _root_.IntermediateField.AdjoinSimple.norm_gen_eq_prod_roots (x : L)
    (hf : (minpoly K x).Splits (algebraMap K F)) :
    (algebraMap K F) (norm K (AdjoinSimple.gen K x)) =
      ((minpoly K x).aroots F).prod := by
  /-
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Field F
    inst✝¹ : Algebra K L
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    ⊢ Eq ((algebraMap K F) ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K …
  -/
  have injKxL := (algebraMap K⟮x⟯ L).injective
  /-
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Field F
    inst✝¹ : Algebra K L
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
    ⊢ Eq ((algebraMap K F) ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K …
  -/
  by_cases hx : IsIntegral K x; swap
    /-
      case neg
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field F
      inst✝¹ : Algebra K L
      inst✝ : Algebra K F
      x : L
      hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
      injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
      hx : Not (IsIntegral K x)
      ⊢ Eq ((algebraMap K F) ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K …
    -/
  · simp [minpoly.eq_zero hx, IntermediateField.AdjoinSimple.norm_gen_eq_one hx, aroots_def]
    /-
      🎉 no goals
    -/
  /-
    case pos
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Field F
    inst✝¹ : Algebra K L
    inst✝ : Algebra K F
    x : L
    hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
    injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
    hx : IsIntegral K x
    ⊢ Eq ((algebraMap K F) ((Algebra.norm K) (IntermediateField.AdjoinSimple.gen K …
  -/
  rw [← adjoin.powerBasis_gen hx, PowerBasis.norm_gen_eq_prod_roots] <;>
    /-
      case pos
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field F
      inst✝¹ : Algebra K L
      inst✝ : Algebra K F
      x : L
      hf : Polynomial.Splits (algebraMap K F) (minpoly K x)
      injKxL : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (I …
      hx : IsIntegral K x
      ⊢ Eq ((minpoly K (IntermediateField.adjoin.powerBasis hx).gen).aroots F).prod  …
    -/
    rw [adjoin.powerBasis_gen hx, ← minpoly.algebraMap_eq injKxL] <;>
    /-
      case pos
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Field F
      inst✝¹ : Algebra K L
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
    simp only [AdjoinSimple.algebraMap_gen _ _, hf]
    /-
      🎉 no goals
    -/


theorem norm_eq_prod_embeddings_gen [Algebra R F] (pb : PowerBasis R S)
    (hE : (minpoly R pb.gen).Splits (algebraMap R F)) (hfx : IsSeparable R pb.gen) :
    algebraMap R F (norm R pb.gen) =
      (@Finset.univ _ (PowerBasis.AlgHom.fintype pb)).prod fun σ => σ pb.gen := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra R F
    pb : PowerBasis R S
    hE : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
    hfx : IsSeparable R pb.gen
    ⊢ Eq ((algebraMap R F) ((Algebra.norm R) pb.gen)) (Finset.univ.prod fun σ => σ …
  -/
  letI := Classical.decEq F
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    F : Type u_6
    inst✝¹ : Field F
    inst✝ : Algebra R F
    pb : PowerBasis R S
    hE : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
    hfx : IsSeparable R pb.gen
    this : DecidableEq F := Classical.decEq F
    ⊢ Eq ((algebraMap R F) ((Algebra.norm R) pb.gen)) (Finset.univ.prod fun σ => σ …
  -/
  rw [PowerBasis.norm_gen_eq_prod_roots pb hE]
  rw [@Fintype.prod_equiv (S →ₐ[R] F) _ _ (PowerBasis.AlgHom.fintype pb) _ _ pb.liftEquiv'
    (fun σ => σ pb.gen) (fun x => x) ?_]
  · rw [Finset.prod_mem_multiset, Finset.prod_eq_multiset_prod, Multiset.toFinset_val,
      Multiset.dedup_eq_self.mpr, Multiset.map_id]
      /-
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : Ring S
        inst✝² : Algebra R S
        F : Type u_6
        inst✝¹ : Field F
        inst✝ : Algebra R F
        pb : PowerBasis R S
        hE : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
        hfx : IsSeparable R pb.gen
        this : DecidableEq F := Classical.decEq F
        ⊢ ((minpoly R pb.gen).aroots F).Nodup
      -/
    · exact nodup_roots hfx.map
      /-
        🎉 no goals
      -/
      /-
        case hfg
        R : Type u_1
        S : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : Ring S
        inst✝² : Algebra R S
        F : Type u_6
        inst✝¹ : Field F
        inst✝ : Algebra R F
        pb : PowerBasis R S
        hE : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
        hfx : IsSeparable R pb.gen
        this : DecidableEq F := Classical.decEq F
        ⊢ ∀ (x : Subtype fun x => Membership.mem ((minpoly R pb.gen).aroots F) x), Eq  …
      -/
    · intro x; rfl
               /-
                 🎉 no goals
               -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : Ring S
      inst✝² : Algebra R S
      F : Type u_6
      inst✝¹ : Field F
      inst✝ : Algebra R F
      pb : PowerBasis R S
      hE : Polynomial.Splits (algebraMap R F) (minpoly R pb.gen)
      hfx : IsSeparable R pb.gen
      this : DecidableEq F := Classical.decEq F
      ⊢ ∀ (x : AlgHom R S F), Eq ((fun σ => σ pb.gen) x) ((fun x => ↑x) (pb.liftEqui …
    -/
  · intro σ; simp only [PowerBasis.liftEquiv'_apply_coe]
             /-
               🎉 no goals
             -/


theorem norm_eq_prod_roots [Algebra.IsSeparable K L] [FiniteDimensional K L] {x : L}
    (hF : (minpoly K x).Splits (algebraMap K F)) :
    algebraMap K F (norm K x) =
      ((minpoly K x).aroots F).prod ^ finrank K⟮x⟯ L := by
  /-
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Field F
    inst✝³ : Algebra K L
    inst✝² : Algebra K F
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : FiniteDimensional K L
    x : L
    hF : Polynomial.Splits (algebraMap K F) (minpoly K x)
    ⊢ Eq ((algebraMap K F) ((Algebra.norm K) x)) (HPow.hPow ((minpoly K x).aroots  …
  -/
  rw [norm_eq_norm_adjoin K x, map_pow, IntermediateField.AdjoinSimple.norm_gen_eq_prod_roots _ hF]
  /-
    🎉 no goals
  -/


theorem prod_embeddings_eq_finrank_pow [Algebra L F] [IsScalarTower K L F] [IsAlgClosed E]
    [Algebra.IsSeparable K F] [FiniteDimensional K F] (pb : PowerBasis K L) :
    ∏ σ : F →ₐ[K] E, σ (algebraMap L F pb.gen) =
      ((@Finset.univ _ (PowerBasis.AlgHom.fintype pb)).prod
        fun σ : L →ₐ[K] E => σ pb.gen) ^ finrank L F := by
  /-
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Field F
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra K F
    E : Type u_7
    inst✝⁶ : Field E
    inst✝⁵ : Algebra K E
    inst✝⁴ : Algebra L F
    inst✝³ : IsScalarTower K L F
    inst✝² : IsAlgClosed E
    inst✝¹ : Algebra.IsSeparable K F
    inst✝ : FiniteDimensional K F
    pb : PowerBasis K L
    ⊢ Eq (Finset.univ.prod fun σ => σ ((algebraMap L F) pb.gen)) (HPow.hPow (Finse …
  -/
  haveI : FiniteDimensional L F := FiniteDimensional.right K L F
  /-
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Field F
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra K F
    E : Type u_7
    inst✝⁶ : Field E
    inst✝⁵ : Algebra K E
    inst✝⁴ : Algebra L F
    inst✝³ : IsScalarTower K L F
    inst✝² : IsAlgClosed E
    inst✝¹ : Algebra.IsSeparable K F
    inst✝ : FiniteDimensional K F
    pb : PowerBasis K L
    this : FiniteDimensional L F
    ⊢ Eq (Finset.univ.prod fun σ => σ ((algebraMap L F) pb.gen)) (HPow.hPow (Finse …
  -/
  haveI : Algebra.IsSeparable L F := Algebra.isSeparable_tower_top_of_isSeparable K L F
  /-
    K : Type u_4
    L : Type u_5
    F : Type u_6
    inst✝¹¹ : Field K
    inst✝¹⁰ : Field L
    inst✝⁹ : Field F
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra K F
    E : Type u_7
    inst✝⁶ : Field E
    inst✝⁵ : Algebra K E
    inst✝⁴ : Algebra L F
    inst✝³ : IsScalarTower K L F
    inst✝² : IsAlgClosed E
    inst✝¹ : Algebra.IsSeparable K F
    inst✝ : FiniteDimensional K F
    pb : PowerBasis K L
    this✝ : FiniteDimensional L F
    this : Algebra.IsSeparable L F
    ⊢ Eq (Finset.univ.prod fun σ => σ ((algebraMap L F) pb.gen)) (HPow.hPow (Finse …
  -/
  letI : Fintype (L →ₐ[K] E) := PowerBasis.AlgHom.fintype pb
  rw [Fintype.prod_equiv algHomEquivSigma (fun σ : F →ₐ[K] E => _) fun σ => σ.1 pb.gen,
    ← Finset.univ_sigma_univ, Finset.prod_sigma, ← Finset.prod_pow]
    /-
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝¹¹ : Field K
      inst✝¹⁰ : Field L
      inst✝⁹ : Field F
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra K F
      E : Type u_7
      inst✝⁶ : Field E
      inst✝⁵ : Algebra K E
      inst✝⁴ : Algebra L F
      inst✝³ : IsScalarTower K L F
      inst✝² : IsAlgClosed E
      inst✝¹ : Algebra.IsSeparable K F
      inst✝ : FiniteDimensional K F
      pb : PowerBasis K L
      this✝¹ : FiniteDimensional L F
      this✝ : Algebra.IsSeparable L F
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      ⊢ Eq (Finset.univ.prod fun a => Finset.univ.prod fun s => ⟨a, s⟩.fst pb.gen) ( …
    -/
  · refine Finset.prod_congr rfl fun σ _ => ?_
    /-
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝¹¹ : Field K
      inst✝¹⁰ : Field L
      inst✝⁹ : Field F
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra K F
      E : Type u_7
      inst✝⁶ : Field E
      inst✝⁵ : Algebra K E
      inst✝⁴ : Algebra L F
      inst✝³ : IsScalarTower K L F
      inst✝² : IsAlgClosed E
      inst✝¹ : Algebra.IsSeparable K F
      inst✝ : FiniteDimensional K F
      pb : PowerBasis K L
      this✝¹ : FiniteDimensional L F
      this✝ : Algebra.IsSeparable L F
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      σ : AlgHom K L E
      x✝ : Membership.mem Finset.univ σ
      ⊢ Eq (Finset.univ.prod fun s => ⟨σ, s⟩.fst pb.gen) (HPow.hPow (σ pb.gen) (Modu …
    -/
    letI : Algebra L E := σ.toRingHom.toAlgebra
    /-
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝¹¹ : Field K
      inst✝¹⁰ : Field L
      inst✝⁹ : Field F
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra K F
      E : Type u_7
      inst✝⁶ : Field E
      inst✝⁵ : Algebra K E
      inst✝⁴ : Algebra L F
      inst✝³ : IsScalarTower K L F
      inst✝² : IsAlgClosed E
      inst✝¹ : Algebra.IsSeparable K F
      inst✝ : FiniteDimensional K F
      pb : PowerBasis K L
      this✝² : FiniteDimensional L F
      this✝¹ : Algebra.IsSeparable L F
      this✝ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      σ : AlgHom K L E
      x✝ : Membership.mem Finset.univ σ
      this : Algebra L E := σ.toAlgebra
      ⊢ Eq (Finset.univ.prod fun s => ⟨σ, s⟩.fst pb.gen) (HPow.hPow (σ pb.gen) (Modu …
    -/
    simp_rw [Finset.prod_const]
    /-
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝¹¹ : Field K
      inst✝¹⁰ : Field L
      inst✝⁹ : Field F
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra K F
      E : Type u_7
      inst✝⁶ : Field E
      inst✝⁵ : Algebra K E
      inst✝⁴ : Algebra L F
      inst✝³ : IsScalarTower K L F
      inst✝² : IsAlgClosed E
      inst✝¹ : Algebra.IsSeparable K F
      inst✝ : FiniteDimensional K F
      pb : PowerBasis K L
      this✝² : FiniteDimensional L F
      this✝¹ : Algebra.IsSeparable L F
      this✝ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      σ : AlgHom K L E
      x✝ : Membership.mem Finset.univ σ
      this : Algebra L E := σ.toAlgebra
      ⊢ Eq (HPow.hPow (σ pb.gen) Finset.univ.card) (HPow.hPow (σ pb.gen) (Module.fin …
    -/
    congr
    /-
      case e_a
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝¹¹ : Field K
      inst✝¹⁰ : Field L
      inst✝⁹ : Field F
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra K F
      E : Type u_7
      inst✝⁶ : Field E
      inst✝⁵ : Algebra K E
      inst✝⁴ : Algebra L F
      inst✝³ : IsScalarTower K L F
      inst✝² : IsAlgClosed E
      inst✝¹ : Algebra.IsSeparable K F
      inst✝ : FiniteDimensional K F
      pb : PowerBasis K L
      this✝² : FiniteDimensional L F
      this✝¹ : Algebra.IsSeparable L F
      this✝ : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      σ : AlgHom K L E
      x✝ : Membership.mem Finset.univ σ
      this : Algebra L E := σ.toAlgebra
      ⊢ Eq Finset.univ.card (Module.finrank L F)
    -/
    exact AlgHom.card L F E
    /-
      🎉 no goals
    -/
    /-
      K : Type u_4
      L : Type u_5
      F : Type u_6
      inst✝¹¹ : Field K
      inst✝¹⁰ : Field L
      inst✝⁹ : Field F
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra K F
      E : Type u_7
      inst✝⁶ : Field E
      inst✝⁵ : Algebra K E
      inst✝⁴ : Algebra L F
      inst✝³ : IsScalarTower K L F
      inst✝² : IsAlgClosed E
      inst✝¹ : Algebra.IsSeparable K F
      inst✝ : FiniteDimensional K F
      pb : PowerBasis K L
      this✝¹ : FiniteDimensional L F
      this✝ : Algebra.IsSeparable L F
      this : Fintype (AlgHom K L E) := PowerBasis.AlgHom.fintype pb
      ⊢ ∀ (x : AlgHom K F E), Eq (x ((algebraMap L F) pb.gen)) ((algHomEquivSigma x) …
    -/
  · intro σ
    simp only [algHomEquivSigma, Equiv.coe_fn_mk, AlgHom.restrictDomain, AlgHom.comp_apply,
      IsScalarTower.coe_toAlgHom']


/-- For `L/K` a finite separable extension of fields and `E` an algebraically closed extension
of `K`, the norm (down to `K`) of an element `x` of `L` is equal to the product of the images
of `x` over all the `K`-embeddings `σ` of `L` into `E`. -/
theorem norm_eq_prod_embeddings [FiniteDimensional K L] [Algebra.IsSeparable K L] [IsAlgClosed E]
    (x : L) : algebraMap K E (norm K x) = ∏ σ : L →ₐ[K] E, σ x := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Algebra K L
    E : Type u_7
    inst✝⁴ : Field E
    inst✝³ : Algebra K E
    inst✝² : FiniteDimensional K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : IsAlgClosed E
    x : L
    ⊢ Eq ((algebraMap K E) ((Algebra.norm K) x)) (Finset.univ.prod fun σ => σ x)
  -/
  have hx := Algebra.IsSeparable.isIntegral K x
  rw [norm_eq_norm_adjoin K x, RingHom.map_pow, ← adjoin.powerBasis_gen hx,
    norm_eq_prod_embeddings_gen E (adjoin.powerBasis hx) (IsAlgClosed.splits_codomain _)]
    /-
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      E : Type u_7
      inst✝⁴ : Field E
      inst✝³ : Algebra K E
      inst✝² : FiniteDimensional K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : IsAlgClosed E
      x : L
      hx : IsIntegral K x
      ⊢ Eq (HPow.hPow (Finset.univ.prod fun σ => σ (IntermediateField.adjoin.powerBa …
    -/
  · exact (prod_embeddings_eq_finrank_pow L (L := K⟮x⟯) E (adjoin.powerBasis hx)).symm
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
      inst✝² : FiniteDimensional K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : IsAlgClosed E
      x : L
      hx : IsIntegral K x
      ⊢ IsSeparable K (IntermediateField.adjoin.powerBasis hx).gen
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
      inst✝² : FiniteDimensional K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : IsAlgClosed E
      x : L
      hx : IsIntegral K x
      this : Algebra.IsSeparable K (Subtype fun x_1 => Membership.mem (IntermediateF …
      ⊢ IsSeparable K (IntermediateField.adjoin.powerBasis hx).gen
    -/
    exact Algebra.IsSeparable.isSeparable K _
    /-
      🎉 no goals
    -/


theorem norm_eq_prod_automorphisms [FiniteDimensional K L] [IsGalois K L] (x : L) :
    algebraMap K L (norm K x) = ∏ σ : L ≃ₐ[K] L, σ x := by
  /-
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsGalois K L
    x : L
    ⊢ Eq ((algebraMap K L) ((Algebra.norm K) x)) (Finset.univ.prod fun σ => σ x)
  -/
  apply NoZeroSMulDivisors.algebraMap_injective L (AlgebraicClosure L)
  /-
    case a
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsGalois K L
    x : L
    ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.norm K)  …
  -/
  rw [map_prod (algebraMap L (AlgebraicClosure L))]
  /-
    case a
    K : Type u_4
    L : Type u_5
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsGalois K L
    x : L
    ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.norm K)  …
  -/
  rw [← Fintype.prod_equiv (Normal.algHomEquivAut K (AlgebraicClosure L) L)]
    /-
      case a
      K : Type u_4
      L : Type u_5
      inst✝⁴ : Field K
      inst✝³ : Field L
      inst✝² : Algebra K L
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsGalois K L
      x : L
      ⊢ Eq ((algebraMap L (AlgebraicClosure L)) ((algebraMap K L) ((Algebra.norm K)  …
    -/
  · rw [← norm_eq_prod_embeddings _ _ x, ← IsScalarTower.algebraMap_apply]
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
      inst✝¹ : FiniteDimensional K L
      inst✝ : IsGalois K L
      x : L
      ⊢ ∀ (x_1 : AlgHom K L (AlgebraicClosure L)), Eq (x_1 x) ((algebraMap L (Algebr …
    -/
  · intro σ
    simp only [Normal.algHomEquivAut, AlgHom.restrictNormal', Equiv.coe_fn_mk,
      AlgEquiv.coe_ofBijective, AlgHom.restrictNormal_commutes, id.map_eq_id, RingHom.id_apply]


theorem isIntegral_norm [Algebra R L] [Algebra R K] [IsScalarTower R K L] [Algebra.IsSeparable K L]
    [FiniteDimensional K L] {x : L} (hx : IsIntegral R x) : IsIntegral R (norm K x) := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    K : Type u_4
    L : Type u_5
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra R L
    inst✝³ : Algebra R K
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsIntegral R x
    ⊢ IsIntegral R ((Algebra.norm K) x)
  -/
  have hx' : IsIntegral K x := hx.tower_top
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    K : Type u_4
    L : Type u_5
    inst✝⁷ : Field K
    inst✝⁶ : Field L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra R L
    inst✝³ : Algebra R K
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : FiniteDimensional K L
    x : L
    hx : IsIntegral R x
    hx' : IsIntegral K x
    ⊢ IsIntegral R ((Algebra.norm K) x)
  -/
  rw [← isIntegral_algebraMap_iff (algebraMap K (AlgebraicClosure L)).injective, norm_eq_prod_roots]
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra R K
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : FiniteDimensional K L
      x : L
      hx : IsIntegral R x
      hx' : IsIntegral K x
      ⊢ IsIntegral R (HPow.hPow ((minpoly K x).aroots (AlgebraicClosure L)).prod (Mo …
    -/
  · refine (IsIntegral.multiset_prod fun y hy => ?_).pow _
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra R K
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : FiniteDimensional K L
      x : L
      hx : IsIntegral R x
      hx' : IsIntegral K x
      y : AlgebraicClosure L
      hy : Membership.mem ((minpoly K x).aroots (AlgebraicClosure L)) y
      ⊢ IsIntegral R y
    -/
    rw [mem_roots_map (minpoly.ne_zero hx')] at hy
    /-
      R : Type u_1
      inst✝⁸ : CommRing R
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra R K
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : FiniteDimensional K L
      x : L
      hx : IsIntegral R x
      hx' : IsIntegral K x
      y : AlgebraicClosure L
      hy : Eq (Polynomial.eval₂ (algebraMap K (AlgebraicClosure L)) y (minpoly K x)) 0
      ⊢ IsIntegral R y
    -/
    use minpoly R x, minpoly.monic hx
    /-
      case right
      R : Type u_1
      inst✝⁸ : CommRing R
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra R K
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : FiniteDimensional K L
      x : L
      hx : IsIntegral R x
      hx' : IsIntegral K x
      y : AlgebraicClosure L
      hy : Eq (Polynomial.eval₂ (algebraMap K (AlgebraicClosure L)) y (minpoly K x)) 0
      ⊢ Eq (Polynomial.eval₂ (algebraMap R (AlgebraicClosure L)) y (minpoly R x)) 0
    -/
    rw [← aeval_def] at hy ⊢
    /-
      case right
      R : Type u_1
      inst✝⁸ : CommRing R
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra R K
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : FiniteDimensional K L
      x : L
      hx : IsIntegral R x
      hx' : IsIntegral K x
      y : AlgebraicClosure L
      hy : Eq ((Polynomial.aeval y) (minpoly K x)) 0
      ⊢ Eq ((Polynomial.aeval y) (minpoly R x)) 0
    -/
    exact minpoly.aeval_of_isScalarTower R x y hy
    /-
      🎉 no goals
    -/
    /-
      case hF
      R : Type u_1
      inst✝⁸ : CommRing R
      K : Type u_4
      L : Type u_5
      inst✝⁷ : Field K
      inst✝⁶ : Field L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : Algebra R K
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsSeparable K L
      inst✝ : FiniteDimensional K L
      x : L
      hx : IsIntegral R x
      hx' : IsIntegral K x
      ⊢ Polynomial.Splits (algebraMap K (AlgebraicClosure L)) (minpoly K x)
    -/
  · apply IsAlgClosed.splits_codomain
    /-
      🎉 no goals
    -/


lemma norm_eq_of_algEquiv [Ring T] [Algebra R T] (e : S ≃ₐ[R] T) (x) :
    Algebra.norm R (e x) = Algebra.norm R x := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    inst✝¹ : Ring T
    inst✝ : Algebra R T
    e : AlgEquiv R S T
    x : S
    ⊢ Eq ((Algebra.norm R) (e x)) ((Algebra.norm R) x)
  -/
  simp_rw [Algebra.norm_apply, ← LinearMap.det_conj _ e.toLinearEquiv]; congr; ext; simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


lemma norm_eq_of_ringEquiv {A B C : Type*} [CommRing A] [CommRing B] [Ring C]
    [Algebra A C] [Algebra B C] (e : A ≃+* B) (he : (algebraMap B C).comp e = algebraMap A C)
    (x : C) :
    e (Algebra.norm A x) = Algebra.norm B x := by
  classical
  by_cases h : ∃ s : Finset C, Nonempty (Basis s B C)
  · obtain ⟨s, ⟨b⟩⟩ := h
    letI : Algebra A B := RingHom.toAlgebra e
    letI : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' he.symm
    rw [Algebra.norm_eq_matrix_det b,
      Algebra.norm_eq_matrix_det (b.mapCoeffs e.symm (by simp [Algebra.smul_def, ← he])),
      e.map_det]
    congr
    ext i j
    simp [leftMulMatrix_apply, LinearMap.toMatrix_apply]
  rw [norm_eq_one_of_not_exists_basis _ h, norm_eq_one_of_not_exists_basis, _root_.map_one]
  intro ⟨s, ⟨b⟩⟩
  exact h ⟨s, ⟨b.mapCoeffs e (by simp [Algebra.smul_def, ← he])⟩⟩


lemma norm_eq_of_equiv_equiv {A₁ B₁ A₂ B₂ : Type*} [CommRing A₁] [Ring B₁]
    [CommRing A₂] [Ring B₂] [Algebra A₁ B₁] [Algebra A₂ B₂] (e₁ : A₁ ≃+* A₂) (e₂ : B₁ ≃+* B₂)
    (he : RingHom.comp (algebraMap A₂ B₂) ↑e₁ = RingHom.comp ↑e₂ (algebraMap A₁ B₁)) (x) :
    Algebra.norm A₁ x = e₁.symm (Algebra.norm A₂ (e₂ x)) := by
  /-
    A₁ : Type u_8
    B₁ : Type u_9
    A₂ : Type u_10
    B₂ : Type u_11
    inst✝⁵ : CommRing A₁
    inst✝⁴ : Ring B₁
    inst✝³ : CommRing A₂
    inst✝² : Ring B₂
    inst✝¹ : Algebra A₁ B₁
    inst✝ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    x : B₁
    ⊢ Eq ((Algebra.norm A₁) x) (e₁.symm ((Algebra.norm A₂) (e₂ x)))
  -/
  letI := (RingHom.comp (e₂ : B₁ →+* B₂) (algebraMap A₁ B₁)).toAlgebra' ?_
    /-
      case refine_2
      A₁ : Type u_8
      B₁ : Type u_9
      A₂ : Type u_10
      B₂ : Type u_11
      inst✝⁵ : CommRing A₁
      inst✝⁴ : Ring B₁
      inst✝³ : CommRing A₂
      inst✝² : Ring B₂
      inst✝¹ : Algebra A₁ B₁
      inst✝ : Algebra A₂ B₂
      e₁ : RingEquiv A₁ A₂
      e₂ : RingEquiv B₁ B₂
      he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
      x : B₁
      this : Algebra A₁ B₂ := ((↑e₂).comp (algebraMap A₁ B₁)).toAlgebra' ?refine_1
      ⊢ Eq ((Algebra.norm A₁) x) (e₁.symm ((Algebra.norm A₂) (e₂ x)))
    -/
  · let e' : B₁ ≃ₐ[A₁] B₂ := { e₂ with commutes' := fun _ ↦ rfl }
    /-
      case refine_2
      A₁ : Type u_8
      B₁ : Type u_9
      A₂ : Type u_10
      B₂ : Type u_11
      inst✝⁵ : CommRing A₁
      inst✝⁴ : Ring B₁
      inst✝³ : CommRing A₂
      inst✝² : Ring B₂
      inst✝¹ : Algebra A₁ B₁
      inst✝ : Algebra A₂ B₂
      e₁ : RingEquiv A₁ A₂
      e₂ : RingEquiv B₁ B₂
      he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
      x : B₁
      this : Algebra A₁ B₂ := ((↑e₂).comp (algebraMap A₁ B₁)).toAlgebra' ?refine_1
      e' : AlgEquiv A₁ B₁ B₂ := { toEquiv := e₂.toEquiv, map_mul' := ⋯, map_add' :=  …
      ⊢ Eq ((Algebra.norm A₁) x) (e₁.symm ((Algebra.norm A₂) (e₂ x)))
    -/
    rw [← Algebra.norm_eq_of_ringEquiv e₁ he, ← Algebra.norm_eq_of_algEquiv e']
    /-
      case refine_2
      A₁ : Type u_8
      B₁ : Type u_9
      A₂ : Type u_10
      B₂ : Type u_11
      inst✝⁵ : CommRing A₁
      inst✝⁴ : Ring B₁
      inst✝³ : CommRing A₂
      inst✝² : Ring B₂
      inst✝¹ : Algebra A₁ B₁
      inst✝ : Algebra A₂ B₂
      e₁ : RingEquiv A₁ A₂
      e₂ : RingEquiv B₁ B₂
      he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
      x : B₁
      this : Algebra A₁ B₂ := ((↑e₂).comp (algebraMap A₁ B₁)).toAlgebra' ?refine_1
      e' : AlgEquiv A₁ B₁ B₂ := { toEquiv := e₂.toEquiv, map_mul' := ⋯, map_add' :=  …
      ⊢ Eq ((Algebra.norm A₁) (e' x)) (e₁.symm (e₁ ((Algebra.norm A₁) (e₂ x))))
    -/
    simp [e']
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    A₁ : Type u_8
    B₁ : Type u_9
    A₂ : Type u_10
    B₂ : Type u_11
    inst✝⁵ : CommRing A₁
    inst✝⁴ : Ring B₁
    inst✝³ : CommRing A₂
    inst✝² : Ring B₂
    inst✝¹ : Algebra A₁ B₁
    inst✝ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    x : B₁
    ⊢ ∀ (c : A₁) (x : B₂), Eq (HMul.hMul (((↑e₂).comp (algebraMap A₁ B₁)) c) x) (H …
  -/
  intros c x
  /-
    case refine_1
    A₁ : Type u_8
    B₁ : Type u_9
    A₂ : Type u_10
    B₂ : Type u_11
    inst✝⁵ : CommRing A₁
    inst✝⁴ : Ring B₁
    inst✝³ : CommRing A₂
    inst✝² : Ring B₂
    inst✝¹ : Algebra A₁ B₁
    inst✝ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    x✝ : B₁
    c : A₁
    x : B₂
    ⊢ Eq (HMul.hMul (((↑e₂).comp (algebraMap A₁ B₁)) c) x) (HMul.hMul x (((↑e₂).co …
  -/
  apply e₂.symm.injective
  simp only [RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply, _root_.map_mul,
    RingEquiv.symm_apply_apply, commutes]


