noncomputable instance instFintypeClassGroup : Fintype (ClassGroup (𝓞 K)) :=
  ClassGroup.fintypeOfAdmissibleOfFinite ℚ K AbsoluteValue.absIsAdmissible


/-- The class number of a number field is the (finite) cardinality of the class group. -/
noncomputable def classNumber : ℕ :=
  Fintype.card (ClassGroup (𝓞 K))


/-- The class number of a number field is `1` iff the ring of integers is a PID. -/
theorem classNumber_eq_one_iff : classNumber K = 1 ↔ IsPrincipalIdealRing (𝓞 K) :=
  card_classGroup_eq_one_iff


theorem exists_ideal_in_class_of_norm_le (C : ClassGroup (𝓞 K)) :
    ∃ I : (Ideal (𝓞 K))⁰, ClassGroup.mk0 I = C ∧
      Ideal.absNorm (I : Ideal (𝓞 K)) ≤ (4 / π) ^ nrComplexPlaces K *
        ((finrank ℚ K).factorial / (finrank ℚ K) ^ (finrank ℚ K) * Real.sqrt |discr K|) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    C : ClassGroup (NumberField.RingOfIntegers K)
    ⊢ Exists fun I => And (Eq (ClassGroup.mk0 I) C) (LE.le (↑(Ideal.absNorm ↑I)) ( …
  -/
  obtain ⟨J, hJ⟩ := ClassGroup.mk0_surjective C⁻¹
  obtain ⟨_, ⟨a, ha, rfl⟩, h_nz, h_nm⟩ :=
    exists_ne_zero_mem_ideal_of_norm_le_mul_sqrt_discr K (FractionalIdeal.mk0 K J)
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    C : ClassGroup (NumberField.RingOfIntegers K)
    J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
    hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
    a : NumberField.RingOfIntegers K
    ha : Membership.mem (↑↑J) a
    h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((Algebra.linearMap (NumberField.RingO …
    ⊢ Exists fun I => And (Eq (ClassGroup.mk0 I) C) (LE.le (↑(Ideal.absNorm ↑I)) ( …
  -/
  obtain ⟨I₀, hI⟩ := Ideal.dvd_iff_le.mpr ((Ideal.span_singleton_le_iff_mem J).mpr (by convert ha))
  have : I₀ ≠ 0 := by
    contrapose! h_nz
    rw [h_nz, mul_zero, show 0 = (⊥ : Ideal (𝓞 K)) by rfl, Ideal.span_singleton_eq_bot] at hI
    rw [Algebra.linearMap_apply, hI, map_zero]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    C : ClassGroup (NumberField.RingOfIntegers K)
    J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
    hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
    a : NumberField.RingOfIntegers K
    ha : Membership.mem (↑↑J) a
    h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((Algebra.linearMap (NumberField.RingO …
    I₀ : Ideal (NumberField.RingOfIntegers K)
    hI : Eq (Ideal.span (Singleton.singleton a)) (HMul.hMul (↑J) I₀)
    this : Ne I₀ 0
    ⊢ Exists fun I => And (Eq (ClassGroup.mk0 I) C) (LE.le (↑(Ideal.absNorm ↑I)) ( …
  -/
  let I := (⟨I₀, mem_nonZeroDivisors_iff_ne_zero.mpr this⟩ : (Ideal (𝓞 K))⁰)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    C : ClassGroup (NumberField.RingOfIntegers K)
    J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
    hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
    a : NumberField.RingOfIntegers K
    ha : Membership.mem (↑↑J) a
    h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((Algebra.linearMap (NumberField.RingO …
    I₀ : Ideal (NumberField.RingOfIntegers K)
    hI : Eq (Ideal.span (Singleton.singleton a)) (HMul.hMul (↑J) I₀)
    this : Ne I₀ 0
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
    ⊢ Exists fun I => And (Eq (ClassGroup.mk0 I) C) (LE.le (↑(Ideal.absNorm ↑I)) ( …
  -/
  refine ⟨I, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      C : ClassGroup (NumberField.RingOfIntegers K)
      J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
      a : NumberField.RingOfIntegers K
      ha : Membership.mem (↑↑J) a
      h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
      h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((Algebra.linearMap (NumberField.RingO …
      I₀ : Ideal (NumberField.RingOfIntegers K)
      hI : Eq (Ideal.span (Singleton.singleton a)) (HMul.hMul (↑J) I₀)
      this : Ne I₀ 0
      I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      ⊢ Eq (ClassGroup.mk0 I) C
    -/
  · suffices ClassGroup.mk0 I = (ClassGroup.mk0 J)⁻¹ by rw [this, hJ, inv_inv]
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      C : ClassGroup (NumberField.RingOfIntegers K)
      J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
      a : NumberField.RingOfIntegers K
      ha : Membership.mem (↑↑J) a
      h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
      h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((Algebra.linearMap (NumberField.RingO …
      I₀ : Ideal (NumberField.RingOfIntegers K)
      hI : Eq (Ideal.span (Singleton.singleton a)) (HMul.hMul (↑J) I₀)
      this : Ne I₀ 0
      I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      ⊢ Eq (ClassGroup.mk0 I) (Inv.inv (ClassGroup.mk0 J))
    -/
    exact ClassGroup.mk0_eq_mk0_inv_iff.mpr ⟨a, Subtype.coe_ne_coe.1 h_nz, by rw [mul_comm, hI]⟩
    /-
      🎉 no goals
    -/
  · rw [← FractionalIdeal.absNorm_span_singleton (𝓞 K), Algebra.linearMap_apply,
      ← FractionalIdeal.coeIdeal_span_singleton, FractionalIdeal.coeIdeal_absNorm, hI, map_mul,
      Nat.cast_mul, Rat.cast_mul, show Ideal.absNorm I₀ = Ideal.absNorm (I : Ideal (𝓞 K)) by rfl,
      Rat.cast_natCast, Rat.cast_natCast, FractionalIdeal.coe_mk0,
      FractionalIdeal.coeIdeal_absNorm, Rat.cast_natCast, mul_div_assoc, mul_assoc, mul_assoc]
      at h_nm
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      C : ClassGroup (NumberField.RingOfIntegers K)
      J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
      a : NumberField.RingOfIntegers K
      ha : Membership.mem (↑↑J) a
      h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
      I₀ : Ideal (NumberField.RingOfIntegers K)
      hI : Eq (Ideal.span (Singleton.singleton a)) (HMul.hMul (↑J) I₀)
      this : Ne I₀ 0
      I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      h_nm : LE.le (HMul.hMul ↑(Ideal.absNorm ↑J) ↑(Ideal.absNorm ↑I)) (HMul.hMul (↑ …
      ⊢ LE.le (↑(Ideal.absNorm ↑I)) (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) (Num …
    -/
    refine le_of_mul_le_mul_of_pos_left h_nm ?_
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      C : ClassGroup (NumberField.RingOfIntegers K)
      J : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      hJ : Eq (ClassGroup.mk0 J) (Inv.inv C)
      a : NumberField.RingOfIntegers K
      ha : Membership.mem (↑↑J) a
      h_nz : Ne ((Algebra.linearMap (NumberField.RingOfIntegers K) K) a) 0
      I₀ : Ideal (NumberField.RingOfIntegers K)
      hI : Eq (Ideal.span (Singleton.singleton a)) (HMul.hMul (↑J) I₀)
      this : Ne I₀ 0
      I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
      h_nm : LE.le (HMul.hMul ↑(Ideal.absNorm ↑J) ↑(Ideal.absNorm ↑I)) (HMul.hMul (↑ …
      ⊢ LT.lt 0 ↑(Ideal.absNorm ↑J)
    -/
    exact Nat.cast_pos.mpr <| Nat.pos_of_ne_zero <| Ideal.absNorm_ne_zero_of_nonZeroDivisors J
    /-
      🎉 no goals
    -/


theorem _root_.RingOfIntegers.isPrincipalIdealRing_of_abs_discr_lt
    (h : |discr K| < (2 * (π / 4) ^ nrComplexPlaces K *
      ((finrank ℚ K) ^ (finrank ℚ K) / (finrank ℚ K).factorial)) ^ 2) :
    IsPrincipalIdealRing (𝓞 K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt (↑(abs (NumberField.discr K))) (HPow.hPow (HMul.hMul (HMul.hMul 2 (H …
    ⊢ IsPrincipalIdealRing (NumberField.RingOfIntegers K)
  -/
  have : 0 < finrank ℚ K := finrank_pos -- Lean needs to know that for positivity to succeed
  rw [← Real.sqrt_lt (by positivity) (by positivity), mul_assoc, ← inv_mul_lt_iff₀' (by positivity),
    mul_inv, ← inv_pow, inv_div, inv_div, mul_assoc, Int.cast_abs] at h
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) (NumberField.InfinitePla …
    this : LT.lt 0 (Module.finrank Rat K)
    ⊢ IsPrincipalIdealRing (NumberField.RingOfIntegers K)
  -/
  rw [← classNumber_eq_one_iff, classNumber, Fintype.card_eq_one_iff]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) (NumberField.InfinitePla …
    this : LT.lt 0 (Module.finrank Rat K)
    ⊢ Exists fun x => ∀ (y : ClassGroup (NumberField.RingOfIntegers K)), Eq y x
  -/
  refine ⟨1, fun C ↦ ?_⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) (NumberField.InfinitePla …
    this : LT.lt 0 (Module.finrank Rat K)
    C : ClassGroup (NumberField.RingOfIntegers K)
    ⊢ Eq C 1
  -/
  obtain ⟨I, rfl, hI⟩ := exists_ideal_in_class_of_norm_le C
  have : Ideal.absNorm I.1 = 1 := by
    refine le_antisymm (Nat.lt_succ.mp ?_) (Nat.one_le_iff_ne_zero.mpr
      (Ideal.absNorm_ne_zero_of_nonZeroDivisors I))
    exact Nat.cast_lt.mp <| lt_of_le_of_lt hI h
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) (NumberField.InfinitePla …
    this✝ : LT.lt 0 (Module.finrank Rat K)
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
    hI : LE.le (↑(Ideal.absNorm ↑I)) (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) ( …
    this : Eq (Ideal.absNorm ↑I) 1
    ⊢ Eq (ClassGroup.mk0 I) 1
  -/
  rw [ClassGroup.mk0_eq_one_iff, Ideal.absNorm_eq_one_iff.mp this]
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) (NumberField.InfinitePla …
    this✝ : LT.lt 0 (Module.finrank Rat K)
    I : Subtype fun x => Membership.mem (nonZeroDivisors (Ideal (NumberField.RingO …
    hI : LE.le (↑(Ideal.absNorm ↑I)) (HMul.hMul (HPow.hPow (HDiv.hDiv 4 Real.pi) ( …
    this : Eq (Ideal.absNorm ↑I) 1
    ⊢ Submodule.IsPrincipal Top.top
  -/
  exact top_isPrincipal
  /-
    🎉 no goals
  -/


theorem classNumber_eq : NumberField.classNumber ℚ = 1 :=
  classNumber_eq_one_iff.mpr <| by
    convert IsPrincipalIdealRing.of_surjective
      (Rat.ringOfIntegersEquiv.symm : ℤ →+* 𝓞 ℚ) Rat.ringOfIntegersEquiv.symm.surjective


