/-- An invertible matrix yields a linear equivalence from the free module to itself.

See `Matrix.toLinearEquiv` for the same map on arbitrary modules.
-/
def toLinearEquiv' (P : Matrix n n R) (_ : Invertible P) : (n → R) ≃ₗ[R] n → R :=
  GeneralLinearGroup.generalLinearEquiv _ _ <|
    Matrix.GeneralLinearGroup.toLin <| unitOfInvertible P


@[simp]
theorem toLinearEquiv'_apply (P : Matrix n n R) (h : Invertible P) :
    (P.toLinearEquiv' h : Module.End R (n → R)) = Matrix.toLin' P :=
  rfl


@[simp]
theorem toLinearEquiv'_symm_apply (P : Matrix n n R) (h : Invertible P) :
    (↑(P.toLinearEquiv' h).symm : Module.End R (n → R)) = Matrix.toLin' (⅟ P) :=
  rfl


/-- Given `hA : IsUnit A.det` and `b : Basis R b`, `A.toLinearEquiv b hA` is
the `LinearEquiv` arising from `toLin b b A`.

See `Matrix.toLinearEquiv'` for this result on `n → R`.
-/
@[simps apply]
noncomputable def toLinearEquiv [DecidableEq n] (A : Matrix n n R) (hA : IsUnit A.det) :
    M ≃ₗ[R] M where
  __ := toLin b b A
  toFun := toLin b b A
  invFun := toLin b b A⁻¹
  left_inv x := by
    simp_rw [← LinearMap.comp_apply, ← Matrix.toLin_mul b b b, Matrix.nonsing_inv_mul _ hA,
      toLin_one, LinearMap.id_apply]
  right_inv x := by
    simp_rw [← LinearMap.comp_apply, ← Matrix.toLin_mul b b b, Matrix.mul_nonsing_inv _ hA,
      toLin_one, LinearMap.id_apply]


theorem ker_toLin_eq_bot [DecidableEq n] (A : Matrix n n R) (hA : IsUnit A.det) :
    LinearMap.ker (toLin b b A) = ⊥ :=
  ker_eq_bot.mpr (toLinearEquiv b A hA).injective


theorem range_toLin_eq_top [DecidableEq n] (A : Matrix n n R) (hA : IsUnit A.det) :
    LinearMap.range (toLin b b A) = ⊤ :=
  range_eq_top.mpr (toLinearEquiv b A hA).surjective


/-- This holds for all integral domains (see `Matrix.exists_mulVec_eq_zero_iff`),
not just fields, but it's easier to prove it for the field of fractions first. -/
theorem exists_mulVec_eq_zero_iff_aux {K : Type*} [DecidableEq n] [Field K] {M : Matrix n n K} :
    (∃ v ≠ 0, M *ᵥ v = 0) ↔ M.det = 0 := by
  /-
    n : Type u_1
    inst✝² : Fintype n
    K : Type u_4
    inst✝¹ : DecidableEq n
    inst✝ : Field K
    M : Matrix n n K
    ⊢ Iff (Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)) (Eq M.det 0)
  -/
  constructor
    /-
      case mp
      n : Type u_1
      inst✝² : Fintype n
      K : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Field K
      M : Matrix n n K
      ⊢ (Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)) → Eq M.det 0
    -/
  · rintro ⟨v, hv, mul_eq⟩
    /-
      case mp.intro.intro
      n : Type u_1
      inst✝² : Fintype n
      K : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Field K
      M : Matrix n n K
      v : n → K
      hv : Ne v 0
      mul_eq : Eq (M.mulVec v) 0
      ⊢ Eq M.det 0
    -/
    contrapose! hv
    /-
      case mp.intro.intro
      n : Type u_1
      inst✝² : Fintype n
      K : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Field K
      M : Matrix n n K
      v : n → K
      mul_eq : Eq (M.mulVec v) 0
      hv : Ne M.det 0
      ⊢ Eq v 0
    -/
    exact eq_zero_of_mulVec_eq_zero hv mul_eq
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Type u_1
      inst✝² : Fintype n
      K : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Field K
      M : Matrix n n K
      ⊢ Eq M.det 0 → Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)
    -/
  · contrapose!
    /-
      case mpr
      n : Type u_1
      inst✝² : Fintype n
      K : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Field K
      M : Matrix n n K
      ⊢ (∀ (v : n → K), Ne v 0 → Ne (M.mulVec v) 0) → Ne M.det 0
    -/
    intro h
    have : Function.Injective (Matrix.toLin' M) := by
      simpa only [← LinearMap.ker_eq_bot, ker_toLin'_eq_bot_iff, not_imp_not] using h
    have :
      M *
          LinearMap.toMatrix'
            ((LinearEquiv.ofInjectiveEndo (Matrix.toLin' M) this).symm : (n → K) →ₗ[K] n → K) =
        1 := by
      refine Matrix.toLin'.injective (LinearMap.ext fun v => ?_)
      rw [Matrix.toLin'_mul, Matrix.toLin'_one, Matrix.toLin'_toMatrix', LinearMap.comp_apply]
      exact (LinearEquiv.ofInjectiveEndo (Matrix.toLin' M) this).apply_symm_apply v
    /-
      case mpr
      n : Type u_1
      inst✝² : Fintype n
      K : Type u_4
      inst✝¹ : DecidableEq n
      inst✝ : Field K
      M : Matrix n n K
      h : ∀ (v : n → K), Ne v 0 → Ne (M.mulVec v) 0
      this✝ : Function.Injective ⇑(Matrix.toLin' M)
      this : Eq (HMul.hMul M (LinearMap.toMatrix' ↑(LinearEquiv.ofInjectiveEndo (Mat …
      ⊢ Ne M.det 0
    -/
    exact Matrix.det_ne_zero_of_right_inverse this
    /-
      🎉 no goals
    -/


theorem exists_mulVec_eq_zero_iff' {A : Type*} (K : Type*) [DecidableEq n] [CommRing A]
    [Nontrivial A] [Field K] [Algebra A K] [IsFractionRing A K] {M : Matrix n n A} :
    (∃ v ≠ 0, M *ᵥ v = 0) ↔ M.det = 0 := by
  have : (∃ v ≠ 0, (algebraMap A K).mapMatrix M *ᵥ v = 0) ↔ _ :=
    exists_mulVec_eq_zero_iff_aux
  /-
    n : Type u_1
    inst✝⁶ : Fintype n
    A : Type u_4
    K : Type u_5
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommRing A
    inst✝³ : Nontrivial A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    M : Matrix n n A
    this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
    ⊢ Iff (Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)) (Eq M.det 0)
  -/
  rw [← RingHom.map_det, IsFractionRing.to_map_eq_zero_iff] at this
  /-
    n : Type u_1
    inst✝⁶ : Fintype n
    A : Type u_4
    K : Type u_5
    inst✝⁵ : DecidableEq n
    inst✝⁴ : CommRing A
    inst✝³ : Nontrivial A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    M : Matrix n n A
    this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
    ⊢ Iff (Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)) (Eq M.det 0)
  -/
  refine Iff.trans ?_ this; constructor <;> rintro ⟨v, hv, mul_eq⟩
    /-
      case mp.intro.intro
      n : Type u_1
      inst✝⁶ : Fintype n
      A : Type u_4
      K : Type u_5
      inst✝⁵ : DecidableEq n
      inst✝⁴ : CommRing A
      inst✝³ : Nontrivial A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      M : Matrix n n A
      this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
      v : n → A
      hv : Ne v 0
      mul_eq : Eq (M.mulVec v) 0
      ⊢ Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).mulVec v) 0)
    -/
  · refine ⟨fun i => algebraMap _ _ (v i), mt (fun h => funext fun i => ?_) hv, ?_⟩
      /-
        case mp.intro.intro.refine_1
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
        v : n → A
        hv : Ne v 0
        mul_eq : Eq (M.mulVec v) 0
        h : Eq (fun i => (algebraMap A K) (v i)) 0
        i : n
        ⊢ Eq (v i) (0 i)
      -/
    · exact IsFractionRing.to_map_eq_zero_iff.mp (congr_fun h i)
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.refine_2
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
        v : n → A
        hv : Ne v 0
        mul_eq : Eq (M.mulVec v) 0
        ⊢ Eq (((algebraMap A K).mapMatrix M).mulVec fun i => (algebraMap A K) (v i)) 0
      -/
    · ext i
      /-
        case mp.intro.intro.refine_2.h
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
        v : n → A
        hv : Ne v 0
        mul_eq : Eq (M.mulVec v) 0
        i : n
        ⊢ Eq (((algebraMap A K).mapMatrix M).mulVec (fun i => (algebraMap A K) (v i))  …
      -/
      refine (RingHom.map_mulVec _ _ _ i).symm.trans ?_
      /-
        case mp.intro.intro.refine_2.h
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
        v : n → A
        hv : Ne v 0
        mul_eq : Eq (M.mulVec v) 0
        i : n
        ⊢ Eq ((algebraMap A K) (M.mulVec v i)) (0 i)
      -/
      rw [mul_eq, Pi.zero_apply, RingHom.map_zero, Pi.zero_apply]
      /-
        🎉 no goals
      -/
    /-
      case mpr.intro.intro
      n : Type u_1
      inst✝⁶ : Fintype n
      A : Type u_4
      K : Type u_5
      inst✝⁵ : DecidableEq n
      inst✝⁴ : CommRing A
      inst✝³ : Nontrivial A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      M : Matrix n n A
      this : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M).m …
      v : n → K
      hv : Ne v 0
      mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
      ⊢ Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)
    -/
  · letI := Classical.decEq K
    obtain ⟨⟨b, hb⟩, ba_eq⟩ :=
      IsLocalization.exist_integer_multiples_of_finset (nonZeroDivisors A) (Finset.univ.image v)
    /-
      case mpr.intro.intro.intro.mk
      n : Type u_1
      inst✝⁶ : Fintype n
      A : Type u_4
      K : Type u_5
      inst✝⁵ : DecidableEq n
      inst✝⁴ : CommRing A
      inst✝³ : Nontrivial A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      M : Matrix n n A
      this✝ : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M). …
      v : n → K
      hv : Ne v 0
      mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
      this : DecidableEq K := Classical.decEq K
      b : A
      hb : Membership.mem (nonZeroDivisors A) b
      ba_eq : ∀ (a : K), Membership.mem (Finset.image v Finset.univ) a → IsLocalizat …
      ⊢ Exists fun v => And (Ne v 0) (Eq (M.mulVec v) 0)
    -/
    choose f hf using ba_eq
    refine
      ⟨fun i => f _ (Finset.mem_image.mpr ⟨i, Finset.mem_univ i, rfl⟩),
        mt (fun h => funext fun i => ?_) hv, ?_⟩
      /-
        case mpr.intro.intro.intro.mk.refine_1
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this✝ : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M). …
        v : n → K
        hv : Ne v 0
        mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
        this : DecidableEq K := Classical.decEq K
        b : A
        hb : Membership.mem (nonZeroDivisors A) b
        f : (a : K) → Membership.mem (Finset.image v Finset.univ) a → A
        hf : ∀ (a : K) (a_1 : Membership.mem (Finset.image v Finset.univ) a), Eq ((alg …
        h : Eq (fun i => f (v i) ⋯) 0
        i : n
        ⊢ Eq (v i) (0 i)
      -/
    · have := congr_arg (algebraMap A K) (congr_fun h i)
      rw [hf, Subtype.coe_mk, Pi.zero_apply, RingHom.map_zero, Algebra.smul_def, mul_eq_zero,
        IsFractionRing.to_map_eq_zero_iff] at this
      /-
        case mpr.intro.intro.intro.mk.refine_1
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this✝¹ : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M) …
        v : n → K
        hv : Ne v 0
        mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
        this✝ : DecidableEq K := Classical.decEq K
        b : A
        hb : Membership.mem (nonZeroDivisors A) b
        f : (a : K) → Membership.mem (Finset.image v Finset.univ) a → A
        hf : ∀ (a : K) (a_1 : Membership.mem (Finset.image v Finset.univ) a), Eq ((alg …
        h : Eq (fun i => f (v i) ⋯) 0
        i : n
        this : Or (Eq b 0) (Eq (v i) 0)
        ⊢ Eq (v i) (0 i)
      -/
      exact this.resolve_left (nonZeroDivisors.ne_zero hb)
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.intro.mk.refine_2
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this✝ : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M). …
        v : n → K
        hv : Ne v 0
        mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
        this : DecidableEq K := Classical.decEq K
        b : A
        hb : Membership.mem (nonZeroDivisors A) b
        f : (a : K) → Membership.mem (Finset.image v Finset.univ) a → A
        hf : ∀ (a : K) (a_1 : Membership.mem (Finset.image v Finset.univ) a), Eq ((alg …
        ⊢ Eq (M.mulVec fun i => f (v i) ⋯) 0
      -/
    · ext i
      /-
        case mpr.intro.intro.intro.mk.refine_2.h
        n : Type u_1
        inst✝⁶ : Fintype n
        A : Type u_4
        K : Type u_5
        inst✝⁵ : DecidableEq n
        inst✝⁴ : CommRing A
        inst✝³ : Nontrivial A
        inst✝² : Field K
        inst✝¹ : Algebra A K
        inst✝ : IsFractionRing A K
        M : Matrix n n A
        this✝ : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M). …
        v : n → K
        hv : Ne v 0
        mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
        this : DecidableEq K := Classical.decEq K
        b : A
        hb : Membership.mem (nonZeroDivisors A) b
        f : (a : K) → Membership.mem (Finset.image v Finset.univ) a → A
        hf : ∀ (a : K) (a_1 : Membership.mem (Finset.image v Finset.univ) a), Eq ((alg …
        i : n
        ⊢ Eq (M.mulVec (fun i => f (v i) ⋯) i) (0 i)
      -/
      refine IsFractionRing.injective A K ?_
      calc
        algebraMap A K ((M *ᵥ (fun i : n => f (v i) _)) i) =
            ((algebraMap A K).mapMatrix M *ᵥ algebraMap _ K b • v) i := ?_
        _ = 0 := ?_
        _ = algebraMap A K 0 := (RingHom.map_zero _).symm
      · simp_rw [RingHom.map_mulVec, mulVec, dotProduct, Function.comp_apply, hf,
          RingHom.mapMatrix_apply, Pi.smul_apply, smul_eq_mul, Algebra.smul_def]
        /-
          case mpr.intro.intro.intro.mk.refine_2.h.calc_2
          n : Type u_1
          inst✝⁶ : Fintype n
          A : Type u_4
          K : Type u_5
          inst✝⁵ : DecidableEq n
          inst✝⁴ : CommRing A
          inst✝³ : Nontrivial A
          inst✝² : Field K
          inst✝¹ : Algebra A K
          inst✝ : IsFractionRing A K
          M : Matrix n n A
          this✝ : Iff (Exists fun v => And (Ne v 0) (Eq (((algebraMap A K).mapMatrix M). …
          v : n → K
          hv : Ne v 0
          mul_eq : Eq (((algebraMap A K).mapMatrix M).mulVec v) 0
          this : DecidableEq K := Classical.decEq K
          b : A
          hb : Membership.mem (nonZeroDivisors A) b
          f : (a : K) → Membership.mem (Finset.image v Finset.univ) a → A
          hf : ∀ (a : K) (a_1 : Membership.mem (Finset.image v Finset.univ) a), Eq ((alg …
          i : n
          ⊢ Eq (((algebraMap A K).mapMatrix M).mulVec (HSMul.hSMul ((algebraMap A K) b)  …
        -/
      · rw [mulVec_smul, mul_eq, Pi.smul_apply, Pi.zero_apply, smul_zero]
        /-
          🎉 no goals
        -/


theorem exists_mulVec_eq_zero_iff {A : Type*} [DecidableEq n] [CommRing A] [IsDomain A]
    {M : Matrix n n A} : (∃ v ≠ 0, M *ᵥ v = 0) ↔ M.det = 0 :=
  exists_mulVec_eq_zero_iff' (FractionRing A)


theorem exists_vecMul_eq_zero_iff {A : Type*} [DecidableEq n] [CommRing A] [IsDomain A]
    {M : Matrix n n A} : (∃ v ≠ 0, v ᵥ* M = 0) ↔ M.det = 0 := by
  /-
    n : Type u_1
    inst✝³ : Fintype n
    A : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    M : Matrix n n A
    ⊢ Iff (Exists fun v => And (Ne v 0) (Eq (Matrix.vecMul v M) 0)) (Eq M.det 0)
  -/
  simpa only [← M.det_transpose, ← mulVec_transpose] using exists_mulVec_eq_zero_iff
  /-
    🎉 no goals
  -/


theorem nondegenerate_iff_det_ne_zero {A : Type*} [DecidableEq n] [CommRing A] [IsDomain A]
    {M : Matrix n n A} : Nondegenerate M ↔ M.det ≠ 0 := by
  /-
    n : Type u_1
    inst✝³ : Fintype n
    A : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    M : Matrix n n A
    ⊢ Iff M.Nondegenerate (Ne M.det 0)
  -/
  rw [ne_eq, ← exists_vecMul_eq_zero_iff]
  /-
    n : Type u_1
    inst✝³ : Fintype n
    A : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    M : Matrix n n A
    ⊢ Iff M.Nondegenerate (Not (Exists fun v => And (Ne v 0) (Eq (Matrix.vecMul v  …
  -/
  push_neg
  /-
    n : Type u_1
    inst✝³ : Fintype n
    A : Type u_4
    inst✝² : DecidableEq n
    inst✝¹ : CommRing A
    inst✝ : IsDomain A
    M : Matrix n n A
    ⊢ Iff M.Nondegenerate (∀ (v : n → A), Ne v 0 → Ne (Matrix.vecMul v M) 0)
  -/
  constructor
    /-
      case mp
      n : Type u_1
      inst✝³ : Fintype n
      A : Type u_4
      inst✝² : DecidableEq n
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      M : Matrix n n A
      ⊢ M.Nondegenerate → ∀ (v : n → A), Ne v 0 → Ne (Matrix.vecMul v M) 0
    -/
  · intro hM v hv hMv
    /-
      case mp
      n : Type u_1
      inst✝³ : Fintype n
      A : Type u_4
      inst✝² : DecidableEq n
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      M : Matrix n n A
      hM : M.Nondegenerate
      v : n → A
      hv : Ne v 0
      hMv : Eq (Matrix.vecMul v M) 0
      ⊢ False
    -/
    obtain ⟨w, hwMv⟩ := hM.exists_not_ortho_of_ne_zero hv
    /-
      case mp.intro
      n : Type u_1
      inst✝³ : Fintype n
      A : Type u_4
      inst✝² : DecidableEq n
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      M : Matrix n n A
      hM : M.Nondegenerate
      v : n → A
      hv : Ne v 0
      hMv : Eq (Matrix.vecMul v M) 0
      w : n → A
      hwMv : Ne (dotProduct v (M.mulVec w)) 0
      ⊢ False
    -/
    simp [dotProduct_mulVec, hMv, zero_dotProduct, ne_eq, not_true] at hwMv
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Type u_1
      inst✝³ : Fintype n
      A : Type u_4
      inst✝² : DecidableEq n
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      M : Matrix n n A
      ⊢ (∀ (v : n → A), Ne v 0 → Ne (Matrix.vecMul v M) 0) → M.Nondegenerate
    -/
  · intro h v hv
    /-
      case mpr
      n : Type u_1
      inst✝³ : Fintype n
      A : Type u_4
      inst✝² : DecidableEq n
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      M : Matrix n n A
      h : ∀ (v : n → A), Ne v 0 → Ne (Matrix.vecMul v M) 0
      v : n → A
      hv : ∀ (w : n → A), Eq (dotProduct v (M.mulVec w)) 0
      ⊢ Eq v 0
    -/
    refine not_imp_not.mp (h v) (funext fun i => ?_)
    /-
      case mpr
      n : Type u_1
      inst✝³ : Fintype n
      A : Type u_4
      inst✝² : DecidableEq n
      inst✝¹ : CommRing A
      inst✝ : IsDomain A
      M : Matrix n n A
      h : ∀ (v : n → A), Ne v 0 → Ne (Matrix.vecMul v M) 0
      v : n → A
      hv : ∀ (w : n → A), Eq (dotProduct v (M.mulVec w)) 0
      i : n
      ⊢ Eq (Matrix.vecMul v M i) (0 i)
    -/
    simpa only [dotProduct_mulVec, dotProduct_single, mul_one] using hv (Pi.single i 1)
    /-
      🎉 no goals
    -/


alias ⟨Nondegenerate.det_ne_zero, Nondegenerate.of_det_ne_zero⟩ := nondegenerate_iff_det_ne_zero


/-- A matrix whose nondiagonal entries are negative with the sum of the entries of each
column positive has nonzero determinant. -/
lemma det_ne_zero_of_sum_col_pos [DecidableEq n] {S : Type*} [LinearOrderedCommRing S]
    {A : Matrix n n S} (h1 : Pairwise fun i j => A i j < 0) (h2 : ∀ j, 0 < ∑ i, A i j) :
    A.det ≠ 0 := by
  /-
    n : Type u_1
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    S : Type u_2
    inst✝ : LinearOrderedCommRing S
    A : Matrix n n S
    h1 : Pairwise fun i j => LT.lt (A i j) 0
    h2 : ∀ (j : n), LT.lt 0 (Finset.univ.sum fun i => A i j)
    ⊢ Ne A.det 0
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h2 : ∀ (j : n), LT.lt 0 (Finset.univ.sum fun i => A i j)
      h✝ : IsEmpty n
      ⊢ Ne A.det 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h2 : ∀ (j : n), LT.lt 0 (Finset.univ.sum fun i => A i j)
      h✝ : Nonempty n
      ⊢ Ne A.det 0
    -/
  · contrapose! h2
    /-
      case inr
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h✝ : Nonempty n
      h2 : Eq A.det 0
      ⊢ Exists fun j => LE.le (Finset.univ.sum fun i => A i j) 0
    -/
    obtain ⟨v, ⟨h_vnz, h_vA⟩⟩ := Matrix.exists_vecMul_eq_zero_iff.mpr h2
    /-
      case inr.intro.intro
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h✝ : Nonempty n
      h2 : Eq A.det 0
      v : n → S
      h_vnz : Ne v 0
      h_vA : Eq (Matrix.vecMul v A) 0
      ⊢ Exists fun j => LE.le (Finset.univ.sum fun i => A i j) 0
    -/
    wlog h_sup : 0 < Finset.sup' Finset.univ Finset.univ_nonempty v
      /-
        case inr.intro.intro.inr
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
        h_sup : Not (LT.lt 0 (Finset.univ.sup' ⋯ v))
        ⊢ Exists fun j => LE.le (Finset.univ.sum fun i => A i j) 0
      -/
    · refine this h1 inferInstance h2 (-1 • v) ?_ ?_ ?_
        /-
          case inr.intro.intro.inr.refine_1
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          h_sup : Not (LT.lt 0 (Finset.univ.sup' ⋯ v))
          ⊢ Ne (HSMul.hSMul (-1) v) 0
        -/
      · exact smul_ne_zero (by norm_num) h_vnz
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.intro.inr.refine_2
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          h_sup : Not (LT.lt 0 (Finset.univ.sup' ⋯ v))
          ⊢ Eq (Matrix.vecMul (HSMul.hSMul (-1) v) A) 0
        -/
      · rw [Matrix.vecMul_smul, h_vA, smul_zero]
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.intro.inr.refine_3
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          h_sup : Not (LT.lt 0 (Finset.univ.sup' ⋯ v))
          ⊢ LT.lt 0 (Finset.univ.sup' ⋯ (HSMul.hSMul (-1) v))
        -/
      · obtain ⟨i, hi⟩ := Function.ne_iff.mp h_vnz
        /-
          case inr.intro.intro.inr.refine_3.intro
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          h_sup : Not (LT.lt 0 (Finset.univ.sup' ⋯ v))
          i : n
          hi : Ne (v i) (0 i)
          ⊢ LT.lt 0 (Finset.univ.sup' ⋯ (HSMul.hSMul (-1) v))
        -/
        simp_rw [Finset.lt_sup'_iff, Finset.mem_univ, true_and] at h_sup ⊢
        /-
          case inr.intro.intro.inr.refine_3.intro
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          i : n
          hi : Ne (v i) (0 i)
          h_sup : Not (Exists fun b => LT.lt 0 (v b))
          ⊢ Exists fun b => LT.lt 0 (HSMul.hSMul (-1) v b)
        -/
        simp_rw [not_exists, not_lt] at h_sup
        /-
          case inr.intro.intro.inr.refine_3.intro
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          i : n
          hi : Ne (v i) (0 i)
          h_sup : ∀ (x : n), LE.le (v x) 0
          ⊢ Exists fun b => LT.lt 0 (HSMul.hSMul (-1) v b)
        -/
        refine ⟨i, ?_⟩
        /-
          case inr.intro.intro.inr.refine_3.intro
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          i : n
          hi : Ne (v i) (0 i)
          h_sup : ∀ (x : n), LE.le (v x) 0
          ⊢ LT.lt 0 (HSMul.hSMul (-1) v i)
        -/
        rw [Pi.smul_apply, neg_smul, one_smul, Left.neg_pos_iff]
        /-
          case inr.intro.intro.inr.refine_3.intro
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          this : ∀ {n : Type u_1} [inst : Fintype n] [inst_1 : DecidableEq n] {S : Type  …
          i : n
          hi : Ne (v i) (0 i)
          h_sup : ∀ (x : n), LE.le (v x) 0
          ⊢ LT.lt (v i) 0
        -/
        exact Ne.lt_of_le hi (h_sup i)
        /-
          🎉 no goals
        -/
      /-
        n✝ : Type u_1
        inst✝³ : Fintype n✝
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
        ⊢ Exists fun j => LE.le (Finset.univ.sum fun i => A i j) 0
      -/
    · obtain ⟨j₀, -, h_j₀⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty v
      /-
        case intro.intro
        n✝ : Type u_1
        inst✝³ : Fintype n✝
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
        j₀ : n
        h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
        ⊢ Exists fun j => LE.le (Finset.univ.sum fun i => A i j) 0
      -/
      refine ⟨j₀, ?_⟩
      /-
        case intro.intro
        n✝ : Type u_1
        inst✝³ : Fintype n✝
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
        j₀ : n
        h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
        ⊢ LE.le (Finset.univ.sum fun i => A i j₀) 0
      -/
      rw [← mul_le_mul_left (h_j₀ ▸ h_sup), Finset.mul_sum, mul_zero]
      /-
        case intro.intro
        n✝ : Type u_1
        inst✝³ : Fintype n✝
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
        j₀ : n
        h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
        ⊢ LE.le (Finset.univ.sum fun i => HMul.hMul (v j₀) (A i j₀)) 0
      -/
      rw [show 0 = ∑ i, v i * A i j₀ from (congrFun h_vA j₀).symm]
      /-
        case intro.intro
        n✝ : Type u_1
        inst✝³ : Fintype n✝
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
        j₀ : n
        h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
        ⊢ LE.le (Finset.univ.sum fun i => HMul.hMul (v j₀) (A i j₀)) (Finset.univ.sum  …
      -/
      refine Finset.sum_le_sum (fun i hi => ?_)
      /-
        case intro.intro
        n✝ : Type u_1
        inst✝³ : Fintype n✝
        n : Type u_1
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        S : Type u_2
        inst✝ : LinearOrderedCommRing S
        A : Matrix n n S
        h1 : Pairwise fun i j => LT.lt (A i j) 0
        h✝ : Nonempty n
        h2 : Eq A.det 0
        v : n → S
        h_vnz : Ne v 0
        h_vA : Eq (Matrix.vecMul v A) 0
        h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
        j₀ : n
        h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
        i : n
        hi : Membership.mem Finset.univ i
        ⊢ LE.le (HMul.hMul (v j₀) (A i j₀)) (HMul.hMul (v i) (A i j₀))
      -/
      by_cases h : i = j₀
        /-
          case pos
          n✝ : Type u_1
          inst✝³ : Fintype n✝
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
          j₀ : n
          h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
          i : n
          hi : Membership.mem Finset.univ i
          h : Eq i j₀
          ⊢ LE.le (HMul.hMul (v j₀) (A i j₀)) (HMul.hMul (v i) (A i j₀))
        -/
      · rw [h]
        /-
          🎉 no goals
        -/
        /-
          case neg
          n✝ : Type u_1
          inst✝³ : Fintype n✝
          n : Type u_1
          inst✝² : Fintype n
          inst✝¹ : DecidableEq n
          S : Type u_2
          inst✝ : LinearOrderedCommRing S
          A : Matrix n n S
          h1 : Pairwise fun i j => LT.lt (A i j) 0
          h✝ : Nonempty n
          h2 : Eq A.det 0
          v : n → S
          h_vnz : Ne v 0
          h_vA : Eq (Matrix.vecMul v A) 0
          h_sup : LT.lt 0 (Finset.univ.sup' ⋯ v)
          j₀ : n
          h_j₀ : Eq (Finset.univ.sup' ⋯ v) (v j₀)
          i : n
          hi : Membership.mem Finset.univ i
          h : Not (Eq i j₀)
          ⊢ LE.le (HMul.hMul (v j₀) (A i j₀)) (HMul.hMul (v i) (A i j₀))
        -/
      · exact (mul_le_mul_right_of_neg (h1 h)).mpr (h_j₀ ▸ Finset.le_sup' v hi)
        /-
          🎉 no goals
        -/


/-- A matrix whose nondiagonal entries are negative with the sum of the entries of each
row positive has nonzero determinant. -/
lemma det_ne_zero_of_sum_row_pos [DecidableEq n] {S : Type*} [LinearOrderedCommRing S]
    {A : Matrix n n S} (h1 : Pairwise fun i j => A i j < 0) (h2 : ∀ i, 0 < ∑ j, A i j) :
    A.det ≠ 0 := by
  /-
    n : Type u_1
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    S : Type u_2
    inst✝ : LinearOrderedCommRing S
    A : Matrix n n S
    h1 : Pairwise fun i j => LT.lt (A i j) 0
    h2 : ∀ (i : n), LT.lt 0 (Finset.univ.sum fun j => A i j)
    ⊢ Ne A.det 0
  -/
  rw [← Matrix.det_transpose]
  /-
    n : Type u_1
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    S : Type u_2
    inst✝ : LinearOrderedCommRing S
    A : Matrix n n S
    h1 : Pairwise fun i j => LT.lt (A i j) 0
    h2 : ∀ (i : n), LT.lt 0 (Finset.univ.sum fun j => A i j)
    ⊢ Ne A.transpose.det 0
  -/
  refine det_ne_zero_of_sum_col_pos ?_ ?_
    /-
      case refine_1
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h2 : ∀ (i : n), LT.lt 0 (Finset.univ.sum fun j => A i j)
      ⊢ Pairwise fun i j => LT.lt (A.transpose i j) 0
    -/
  · simp_rw [Matrix.transpose_apply]
    /-
      case refine_1
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h2 : ∀ (i : n), LT.lt 0 (Finset.univ.sum fun j => A i j)
      ⊢ Pairwise fun i j => LT.lt (A j i) 0
    -/
    exact fun i j h => h1 h.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h2 : ∀ (i : n), LT.lt 0 (Finset.univ.sum fun j => A i j)
      ⊢ ∀ (j : n), LT.lt 0 (Finset.univ.sum fun i => A.transpose i j)
    -/
  · simp_rw [Matrix.transpose_apply]
    /-
      case refine_2
      n : Type u_1
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      S : Type u_2
      inst✝ : LinearOrderedCommRing S
      A : Matrix n n S
      h1 : Pairwise fun i j => LT.lt (A i j) 0
      h2 : ∀ (i : n), LT.lt 0 (Finset.univ.sum fun j => A i j)
      ⊢ ∀ (j : n), LT.lt 0 (Finset.univ.sum fun x => A j x)
    -/
    exact h2
    /-
      🎉 no goals
    -/


