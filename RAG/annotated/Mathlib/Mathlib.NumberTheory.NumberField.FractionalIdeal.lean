instance (I : FractionalIdeal (𝓞 K)⁰ K) : Module.Free ℤ I := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K
    ⊢ Module.Free Int (Subtype fun x => Membership.mem (↑I) x)
  -/
  refine Free.of_equiv (LinearEquiv.restrictScalars ℤ (I.equivNum ?_)).symm
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K
    ⊢ Ne (↑I.den) 0
  -/
  exact nonZeroDivisors.coe_ne_zero I.den
  /-
    🎉 no goals
  -/


instance (I : FractionalIdeal (𝓞 K)⁰ K) : Module.Finite ℤ I := by
  refine Module.Finite.of_surjective
    (LinearEquiv.restrictScalars ℤ (I.equivNum ?_)).symm.toLinearMap (LinearEquiv.surjective _)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K
    ⊢ Ne (↑I.den) 0
  -/
  exact nonZeroDivisors.coe_ne_zero I.den
  /-
    🎉 no goals
  -/


instance (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ) :
    IsLocalizedModule ℤ⁰ ((Submodule.subtype (I : Submodule (𝓞 K) K)).restrictScalars ℤ) where
  map_units x := by
    rw [← (Algebra.lmul _ _).commutes, Algebra.lmul_isUnit_iff, isUnit_iff_ne_zero, eq_intCast,
      Int.cast_ne_zero]
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      x : Subtype fun x => Membership.mem (nonZeroDivisors Int) x
      ⊢ Ne (↑x) 0
    -/
    exact nonZeroDivisors.coe_ne_zero x
    /-
      🎉 no goals
    -/
  surj' x := by
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      x : K
      ⊢ Exists fun x_1 => Eq (HSMul.hSMul x_1.2 x) ((↑Int (↑↑I).subtype) x_1.1)
    -/
    obtain ⟨⟨a, _, d, hd, rfl⟩, h⟩ := IsLocalization.surj (Algebra.algebraMapSubmonoid (𝓞 K) ℤ⁰) x
    refine ⟨⟨⟨Ideal.absNorm I.1.num * (algebraMap _ K a), I.1.num_le ?_⟩, d * Ideal.absNorm I.1.num,
      ?_⟩ , ?_⟩
      /-
        case intro.mk.mk.intro.intro.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
        ⊢ Membership.mem ((fun a => ↑a) ↑(↑I).num) (HMul.hMul (↑(Ideal.absNorm (↑I).nu …
      -/
    · simp_rw [FractionalIdeal.val_eq_coe, FractionalIdeal.coe_coeIdeal]
      /-
        case intro.mk.mk.intro.intro.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
        ⊢ Membership.mem (IsLocalization.coeSubmodule K (↑I).num) (HMul.hMul (↑(Ideal. …
      -/
      refine (IsLocalization.mem_coeSubmodule _ _).mpr ⟨Ideal.absNorm I.1.num * a, ?_, ?_⟩
        /-
          case intro.mk.mk.intro.intro.refine_1.refine_1
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
          x : K
          a : NumberField.RingOfIntegers K
          d : Int
          hd : Membership.mem (↑(nonZeroDivisors Int)) d
          h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
          ⊢ Membership.mem (↑I).num (HMul.hMul (↑(Ideal.absNorm (↑I).num)) a)
        -/
      · exact Ideal.mul_mem_right _ _ I.1.num.absNorm_mem
        /-
          🎉 no goals
        -/
        /-
          case intro.mk.mk.intro.intro.refine_1.refine_2
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
          x : K
          a : NumberField.RingOfIntegers K
          d : Int
          hd : Membership.mem (↑(nonZeroDivisors Int)) d
          h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
          ⊢ Eq ((algebraMap (NumberField.RingOfIntegers K) K) (HMul.hMul (↑(Ideal.absNor …
        -/
      · rw [map_mul, map_natCast]
        /-
          🎉 no goals
        -/
      /-
        case intro.mk.mk.intro.intro.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
        ⊢ Membership.mem (nonZeroDivisors Int) (HMul.hMul d ↑(Ideal.absNorm (↑I).num))
      -/
    · refine Submonoid.mul_mem _ hd (mem_nonZeroDivisors_of_ne_zero ?_)
      /-
        case intro.mk.mk.intro.intro.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
        ⊢ Ne (↑(Ideal.absNorm (↑I).num)) 0
      -/
      rw [Nat.cast_ne_zero, ne_eq, Ideal.absNorm_eq_zero_iff]
      /-
        case intro.mk.mk.intro.intro.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
        ⊢ Not (Eq (↑I).num Bot.bot)
      -/
      exact FractionalIdeal.num_eq_zero_iff.not.mpr <| Units.ne_zero I
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.mk.intro.intro.refine_3
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ↑{ fst := a …
        ⊢ Eq (HSMul.hSMul { fst := ⟨HMul.hMul (↑(Ideal.absNorm (↑I).num)) ((algebraMap …
      -/
    · simp_rw [LinearMap.coe_restrictScalars, Submodule.coe_subtype] at h ⊢
      /-
        case intro.mk.mk.intro.intro.refine_3
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ((algebraMa …
        ⊢ Eq (HSMul.hSMul ⟨HMul.hMul d ↑(Ideal.absNorm (↑I).num), ⋯⟩ x) (HMul.hMul (↑( …
      -/
      rw [← h]
      simp only [Submonoid.mk_smul, zsmul_eq_mul, Int.cast_mul, Int.cast_natCast, algebraMap_int_eq,
        eq_intCast, map_intCast]
      /-
        case intro.mk.mk.intro.intro.refine_3
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
        x : K
        a : NumberField.RingOfIntegers K
        d : Int
        hd : Membership.mem (↑(nonZeroDivisors Int)) d
        h : Eq (HMul.hMul x ((algebraMap (NumberField.RingOfIntegers K) K) ((algebraMa …
        ⊢ Eq (HMul.hMul (HMul.hMul ↑d ↑(Ideal.absNorm (↑I).num)) x) (HMul.hMul (↑(Idea …
      -/
      ring
      /-
        🎉 no goals
      -/
  exists_of_eq h :=
           /-
             K : Type u_1
             inst✝¹ : Field K
             inst✝ : NumberField K
             I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
             x₁✝ x₂✝ : Subtype fun x => Membership.mem (↑↑I) x
             h : Eq ((↑Int (↑↑I).subtype) x₁✝) ((↑Int (↑↑I).subtype) x₂✝)
             ⊢ Eq (HSMul.hSMul 1 x₁✝) (HSMul.hSMul 1 x₂✝)
           -/
    ⟨1, by rwa [one_smul, one_smul, ← (Submodule.injective_subtype I.1.coeToSubmodule).eq_iff]⟩
           /-
             🎉 no goals
           -/


/-- A `ℤ`-basis of a fractional ideal. -/
noncomputable def fractionalIdealBasis (I : FractionalIdeal (𝓞 K)⁰ K) :
    Basis (Free.ChooseBasisIndex ℤ I) ℤ I := Free.chooseBasis ℤ I


/-- A `ℚ`-basis of `K` that spans `I` over `ℤ`, see `mem_span_basisOfFractionalIdeal` below. -/
noncomputable def basisOfFractionalIdeal (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ) :
    Basis (Free.ChooseBasisIndex ℤ I) ℚ K :=
  (fractionalIdealBasis K I.1).ofIsLocalizedModule ℚ ℤ⁰
    ((Submodule.subtype (I : Submodule (𝓞 K) K)).restrictScalars ℤ)


theorem basisOfFractionalIdeal_apply (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ)
    (i : Free.ChooseBasisIndex ℤ I) :
    basisOfFractionalIdeal K I i = fractionalIdealBasis K I.1 i :=
  (fractionalIdealBasis K I.1).ofIsLocalizedModule_apply ℚ ℤ⁰ _ i


theorem mem_span_basisOfFractionalIdeal {I : (FractionalIdeal (𝓞 K)⁰ K)ˣ} {x : K} :
    x ∈ Submodule.span ℤ (Set.range (basisOfFractionalIdeal K I)) ↔ x ∈ (I : Set K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    x : K
    ⊢ Iff (Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.basisOfFrac …
  -/
  rw [basisOfFractionalIdeal, (fractionalIdealBasis K I.1).ofIsLocalizedModule_span ℚ ℤ⁰ _]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    x : K
    ⊢ Iff (Membership.mem (LinearMap.range (↑Int (↑↑I).subtype)) x) (Membership.me …
  -/
  simp
  /-
    🎉 no goals
  -/


open Module in
theorem fractionalIdeal_rank (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ) :
    finrank ℤ I = finrank ℤ (𝓞 K) := by
  rw [finrank_eq_card_chooseBasisIndex, RingOfIntegers.rank,
    finrank_eq_card_basis (basisOfFractionalIdeal K I)]


/-- The absolute value of the determinant of the base change from `integralBasis` to
`basisOfFractionalIdeal I` is equal to the norm of `I`. -/
theorem det_basisOfFractionalIdeal_eq_absNorm (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ)
    (e : (Free.ChooseBasisIndex ℤ (𝓞 K)) ≃ (Free.ChooseBasisIndex ℤ I)) :
    |(integralBasis K).det ((basisOfFractionalIdeal K I).reindex e.symm)| =
      FractionalIdeal.absNorm I.1 := by
  rw [← FractionalIdeal.abs_det_basis_change (RingOfIntegers.basis K) I.1
    ((fractionalIdealBasis K I.1).reindex e.symm)]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Eq (abs ((NumberField.integralBasis K).det ⇑((NumberField.basisOfFractionalI …
  -/
  congr
  /-
    case e_a.h.e_6.h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Eq (⇑((NumberField.basisOfFractionalIdeal K I).reindex e.symm)) (Function.co …
  -/
  ext
  /-
    case e_a.h.e_6.h.h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    x✝ : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    ⊢ Eq (((NumberField.basisOfFractionalIdeal K I).reindex e.symm) x✝) (Function. …
  -/
  simpa using basisOfFractionalIdeal_apply K I _
  /-
    🎉 no goals
  -/


