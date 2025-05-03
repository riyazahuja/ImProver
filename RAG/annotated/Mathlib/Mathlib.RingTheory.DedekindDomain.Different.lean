/-- Under the AKLB setting, `Iᵛ := traceDual A K (I : Submodule B L)` is the
`Submodule B L` such that `x ∈ Iᵛ ↔ ∀ y ∈ I, Tr(x, y) ∈ A` -/
noncomputable
def Submodule.traceDual (I : Submodule B L) : Submodule B L where
  __ := (traceForm K L).dualSubmodule (I.restrictScalars A)
  smul_mem' c x hx a ha := by
    /-
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type ?u.2603
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Field K
      inst✝⁸ : CommRing B
      inst✝⁷ : Field L
      inst✝⁶ : Algebra A K
      inst✝⁵ : Algebra B L
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra K L
      inst✝² : Algebra A L
      inst✝¹ : IsScalarTower A K L
      inst✝ : IsScalarTower A B L
      I : Submodule B L
      c : B
      x : L
      hx : Membership.mem __spread✝⁻⁰.carrier x
      a : L
      ha : Membership.mem (Submodule.restrictScalars A I) a
      ⊢ Membership.mem 1 (((Algebra.traceForm K L) (HSMul.hSMul c x)) a)
    -/
    rw [traceForm_apply, smul_mul_assoc, mul_comm, ← smul_mul_assoc, mul_comm]
    /-
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type ?u.2603
      inst✝¹⁰ : CommRing A
      inst✝⁹ : Field K
      inst✝⁸ : CommRing B
      inst✝⁷ : Field L
      inst✝⁶ : Algebra A K
      inst✝⁵ : Algebra B L
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra K L
      inst✝² : Algebra A L
      inst✝¹ : IsScalarTower A K L
      inst✝ : IsScalarTower A B L
      I : Submodule B L
      c : B
      x : L
      hx : Membership.mem __spread✝⁻⁰.carrier x
      a : L
      ha : Membership.mem (Submodule.restrictScalars A I) a
      ⊢ Membership.mem 1 ((Algebra.trace K L) (HMul.hMul x (HSMul.hSMul c a)))
    -/
    exact hx _ (Submodule.smul_mem _ c ha)
    /-
      🎉 no goals
    -/


local notation:max I:max "ᵛ" => Submodule.traceDual A K I


lemma mem_traceDual {I : Submodule B L} {x} :
    x ∈ Iᵛ ↔ ∀ a ∈ I, traceForm K L x a ∈ (algebraMap A K).range :=
  forall₂_congr fun _ _ ↦ mem_one


lemma le_traceDual_iff_map_le_one {I J : Submodule B L} :
    I ≤ Jᵛ ↔ ((I * J : Submodule B L).restrictScalars A).map
      ((trace K L).restrictScalars A) ≤ 1 := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁰ : CommRing A
    inst✝⁹ : Field K
    inst✝⁸ : CommRing B
    inst✝⁷ : Field L
    inst✝⁶ : Algebra A K
    inst✝⁵ : Algebra B L
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra K L
    inst✝² : Algebra A L
    inst✝¹ : IsScalarTower A K L
    inst✝ : IsScalarTower A B L
    I J : Submodule B L
    ⊢ Iff (LE.le I (Submodule.traceDual A K J)) (LE.le (Submodule.map (↑A (Algebra …
  -/
  rw [Submodule.map_le_iff_le_comap, Submodule.restrictScalars_mul, Submodule.mul_le]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁰ : CommRing A
    inst✝⁹ : Field K
    inst✝⁸ : CommRing B
    inst✝⁷ : Field L
    inst✝⁶ : Algebra A K
    inst✝⁵ : Algebra B L
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra K L
    inst✝² : Algebra A L
    inst✝¹ : IsScalarTower A K L
    inst✝ : IsScalarTower A B L
    I J : Submodule B L
    ⊢ Iff (LE.le I (Submodule.traceDual A K J)) (∀ (m : L), Membership.mem (Submod …
  -/
  simp [SetLike.le_def, mem_traceDual]
  /-
    🎉 no goals
  -/


lemma le_traceDual_mul_iff {I J J' : Submodule B L} :
    I ≤ (J * J')ᵛ ↔ I * J ≤ J'ᵛ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁰ : CommRing A
    inst✝⁹ : Field K
    inst✝⁸ : CommRing B
    inst✝⁷ : Field L
    inst✝⁶ : Algebra A K
    inst✝⁵ : Algebra B L
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra K L
    inst✝² : Algebra A L
    inst✝¹ : IsScalarTower A K L
    inst✝ : IsScalarTower A B L
    I J J' : Submodule B L
    ⊢ Iff (LE.le I (Submodule.traceDual A K (HMul.hMul J J'))) (LE.le (HMul.hMul I …
  -/
  simp_rw [le_traceDual_iff_map_le_one, mul_assoc]
  /-
    🎉 no goals
  -/


lemma le_traceDual {I J : Submodule B L} :
    I ≤ Jᵛ ↔ I * J ≤ 1ᵛ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁰ : CommRing A
    inst✝⁹ : Field K
    inst✝⁸ : CommRing B
    inst✝⁷ : Field L
    inst✝⁶ : Algebra A K
    inst✝⁵ : Algebra B L
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra K L
    inst✝² : Algebra A L
    inst✝¹ : IsScalarTower A K L
    inst✝ : IsScalarTower A B L
    I J : Submodule B L
    ⊢ Iff (LE.le I (Submodule.traceDual A K J)) (LE.le (HMul.hMul I J) (Submodule. …
  -/
  rw [← le_traceDual_mul_iff, mul_one]
  /-
    🎉 no goals
  -/


lemma le_traceDual_comm {I J : Submodule B L} :
                          /-
                            A : Type u_1
                            K : Type u_2
                            L : Type u
                            B : Type u_3
                            inst✝¹⁰ : CommRing A
                            inst✝⁹ : Field K
                            inst✝⁸ : CommRing B
                            inst✝⁷ : Field L
                            inst✝⁶ : Algebra A K
                            inst✝⁵ : Algebra B L
                            inst✝⁴ : Algebra A B
                            inst✝³ : Algebra K L
                            inst✝² : Algebra A L
                            inst✝¹ : IsScalarTower A K L
                            inst✝ : IsScalarTower A B L
                            I J : Submodule B L
                            ⊢ Iff (LE.le I (Submodule.traceDual A K J)) (LE.le J (Submodule.traceDual A K  …
                          -/
    I ≤ Jᵛ ↔ J ≤ Iᵛ := by rw [le_traceDual, mul_comm, ← le_traceDual]
                          /-
                            🎉 no goals
                          -/


lemma le_traceDual_traceDual {I : Submodule B L} :
    I ≤ Iᵛᵛ := le_traceDual_comm.mpr le_rfl


@[simp]
lemma traceDual_bot :
                                   /-
                                     A : Type u_1
                                     K : Type u_2
                                     L : Type u
                                     B : Type u_3
                                     inst✝¹⁰ : CommRing A
                                     inst✝⁹ : Field K
                                     inst✝⁸ : CommRing B
                                     inst✝⁷ : Field L
                                     inst✝⁶ : Algebra A K
                                     inst✝⁵ : Algebra B L
                                     inst✝⁴ : Algebra A B
                                     inst✝³ : Algebra K L
                                     inst✝² : Algebra A L
                                     inst✝¹ : IsScalarTower A K L
                                     inst✝ : IsScalarTower A B L
                                     ⊢ Eq (Submodule.traceDual A K Bot.bot) Top.top
                                   -/
    (⊥ : Submodule B L)ᵛ = ⊤ := by ext; simpa [mem_traceDual, -RingHom.mem_range] using zero_mem _
                                        /-
                                          🎉 no goals
                                        -/


open scoped Classical in
lemma traceDual_top' :
    (⊤ : Submodule B L)ᵛ =
      if ((LinearMap.range (Algebra.trace K L)).restrictScalars A ≤ 1) then ⊤ else ⊥ := by
  classical
  split_ifs with h
  · rw [_root_.eq_top_iff]
    exact fun _ _ _ _ ↦ h ⟨_, rfl⟩
  · simp only [SetLike.le_def, restrictScalars_mem, LinearMap.mem_range, mem_one,
      forall_exists_index, forall_apply_eq_imp_iff, not_forall, not_exists] at h
    obtain ⟨b, hb⟩ := h
    simp_rw [eq_bot_iff, SetLike.le_def, mem_bot, mem_traceDual, mem_top, true_implies,
      traceForm_apply, RingHom.mem_range]
    contrapose! hb with hx'
    obtain ⟨c, hc, hc0⟩ := hx'
    simpa [hc0] using hc (c⁻¹ * b)


lemma traceDual_top [Decidable (IsField A)] :
    (⊤ : Submodule B L)ᵛ = if IsField A then ⊤ else ⊥ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsDomain A
    inst✝³ : IsFractionRing A K
    inst✝² : FiniteDimensional K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : Decidable (IsField A)
    ⊢ Eq (Submodule.traceDual A K Top.top) (ite (IsField A) Top.top Bot.bot)
  -/
  convert traceDual_top'
  rw [← IsFractionRing.surjective_iff_isField (R := A) (K := K),
    LinearMap.range_eq_top.mpr (Algebra.trace_surjective K L),
    ← RingHom.range_eq_top, _root_.eq_top_iff]
  /-
    case h.e'_3.h₁.a
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsDomain A
    inst✝³ : IsFractionRing A K
    inst✝² : FiniteDimensional K L
    inst✝¹ : Algebra.IsSeparable K L
    inst✝ : Decidable (IsField A)
    ⊢ Iff (LE.le Top.top (algebraMap A K).range) (LE.le (Submodule.restrictScalars …
  -/
  simp [SetLike.le_def]
  /-
    🎉 no goals
  -/


variable (A K) in
lemma map_equiv_traceDual [IsDomain A] [IsFractionRing B L] [IsDomain B]
    [NoZeroSMulDivisors A B] (I : Submodule B (FractionRing B)) :
    (traceDual A (FractionRing A) I).map (FractionRing.algEquiv B L) =
      traceDual A K (I.map (FractionRing.algEquiv B L)) := by
  show Submodule.map (FractionRing.algEquiv B L).toLinearEquiv.toLinearMap _ =
    traceDual A K (I.map (FractionRing.algEquiv B L).toLinearEquiv.toLinearMap)
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsFractionRing A K
    inst✝³ : IsDomain A
    inst✝² : IsFractionRing B L
    inst✝¹ : IsDomain B
    inst✝ : NoZeroSMulDivisors A B
    I : Submodule B (FractionRing B)
    ⊢ Eq (Submodule.map (↑(FractionRing.algEquiv B L).toLinearEquiv) (Submodule.tr …
  -/
  rw [Submodule.map_equiv_eq_comap_symm, Submodule.map_equiv_eq_comap_symm]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsFractionRing A K
    inst✝³ : IsDomain A
    inst✝² : IsFractionRing B L
    inst✝¹ : IsDomain B
    inst✝ : NoZeroSMulDivisors A B
    I : Submodule B (FractionRing B)
    ⊢ Eq (Submodule.comap (↑(FractionRing.algEquiv B L).toLinearEquiv.symm) (Submo …
  -/
  ext x
  simp only [AlgEquiv.toLinearEquiv_symm, AlgEquiv.toLinearEquiv_toLinearMap,
    traceDual, traceForm_apply, Submodule.mem_comap, AlgEquiv.toLinearMap_apply,
    Submodule.mem_mk, AddSubmonoid.mem_mk, AddSubsemigroup.mem_mk, Set.mem_setOf_eq]
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsFractionRing A K
    inst✝³ : IsDomain A
    inst✝² : IsFractionRing B L
    inst✝¹ : IsDomain B
    inst✝ : NoZeroSMulDivisors A B
    I : Submodule B (FractionRing B)
    x : L
    ⊢ Iff (Membership.mem ((Algebra.traceForm (FractionRing A) (FractionRing B)).d …
  -/
  apply (FractionRing.algEquiv B L).forall_congr
  simp only [restrictScalars_mem, traceForm_apply, AlgEquiv.toEquiv_eq_coe,
    EquivLike.coe_coe, mem_comap, AlgEquiv.toLinearMap_apply, AlgEquiv.symm_apply_apply]
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsFractionRing A K
    inst✝³ : IsDomain A
    inst✝² : IsFractionRing B L
    inst✝¹ : IsDomain B
    inst✝ : NoZeroSMulDivisors A B
    I : Submodule B (FractionRing B)
    x : L
    ⊢ ∀ (a : FractionRing B), Iff (Membership.mem I a → Membership.mem 1 ((Algebra …
  -/
  refine fun {y} ↦ (forall_congr' fun hy ↦ ?_)
  rw [Algebra.trace_eq_of_equiv_equiv (FractionRing.algEquiv A K).toRingEquiv
    (FractionRing.algEquiv B L).toRingEquiv]
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁵ : CommRing A
    inst✝¹⁴ : Field K
    inst✝¹³ : CommRing B
    inst✝¹² : Field L
    inst✝¹¹ : Algebra A K
    inst✝¹⁰ : Algebra B L
    inst✝⁹ : Algebra A B
    inst✝⁸ : Algebra K L
    inst✝⁷ : Algebra A L
    inst✝⁶ : IsScalarTower A K L
    inst✝⁵ : IsScalarTower A B L
    inst✝⁴ : IsFractionRing A K
    inst✝³ : IsDomain A
    inst✝² : IsFractionRing B L
    inst✝¹ : IsDomain B
    inst✝ : NoZeroSMulDivisors A B
    I : Submodule B (FractionRing B)
    x : L
    y : FractionRing B
    hy : Membership.mem I y
    ⊢ Iff (Membership.mem 1 ((FractionRing.algEquiv A K).toRingEquiv.symm ((Algebr …
  -/
  swap
    /-
      case h.he
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁵ : CommRing A
      inst✝¹⁴ : Field K
      inst✝¹³ : CommRing B
      inst✝¹² : Field L
      inst✝¹¹ : Algebra A K
      inst✝¹⁰ : Algebra B L
      inst✝⁹ : Algebra A B
      inst✝⁸ : Algebra K L
      inst✝⁷ : Algebra A L
      inst✝⁶ : IsScalarTower A K L
      inst✝⁵ : IsScalarTower A B L
      inst✝⁴ : IsFractionRing A K
      inst✝³ : IsDomain A
      inst✝² : IsFractionRing B L
      inst✝¹ : IsDomain B
      inst✝ : NoZeroSMulDivisors A B
      I : Submodule B (FractionRing B)
      x : L
      y : FractionRing B
      hy : Membership.mem I y
      ⊢ Eq ((algebraMap K L).comp ↑(FractionRing.algEquiv A K).toRingEquiv) ((↑(Frac …
    -/
  · apply IsLocalization.ringHom_ext (M := A⁰); ext
    simp only [AlgEquiv.toRingEquiv_eq_coe, AlgEquiv.toRingEquiv_toRingHom, RingHom.coe_comp,
      RingHom.coe_coe, Function.comp_apply, AlgEquiv.commutes, ← IsScalarTower.algebraMap_apply]
    rw [IsScalarTower.algebraMap_apply A B (FractionRing B), AlgEquiv.commutes,
      ← IsScalarTower.algebraMap_apply]
  simp only [AlgEquiv.toRingEquiv_eq_coe, _root_.map_mul, AlgEquiv.coe_ringEquiv,
    AlgEquiv.apply_symm_apply, ← AlgEquiv.symm_toRingEquiv, mem_one, AlgEquiv.algebraMap_eq_apply]


lemma Submodule.mem_traceDual_iff_isIntegral {I : Submodule B L} {x} :
    x ∈ Iᵛ ↔ ∀ a ∈ I, IsIntegral A (traceForm K L x a) :=
  forall₂_congr fun _ _ ↦ mem_one.trans IsIntegrallyClosed.isIntegral_iff.symm


lemma Submodule.one_le_traceDual_one :
    (1 : Submodule B L) ≤ 1ᵛ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    ⊢ LE.le 1 (Submodule.traceDual A K 1)
  -/
  rw [le_traceDual_iff_map_le_one, mul_one, one_eq_range]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    ⊢ LE.le (Submodule.map (↑A (Algebra.trace K L)) (Submodule.restrictScalars A ( …
  -/
  rintro _ ⟨x, ⟨x, rfl⟩, rfl⟩
  /-
    case intro.intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    x : B
    ⊢ Membership.mem 1 ((↑A (Algebra.trace K L)) ((Algebra.linearMap B L) x))
  -/
  rw [mem_one]
  /-
    case intro.intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    x : B
    ⊢ Exists fun y => Eq ((algebraMap A K) y) ((↑A (Algebra.trace K L)) ((Algebra. …
  -/
  apply IsIntegrallyClosed.isIntegral_iff.mp
  /-
    case intro.intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    x : B
    ⊢ IsIntegral A ((↑A (Algebra.trace K L)) ((Algebra.linearMap B L) x))
  -/
  apply isIntegral_trace
  /-
    case intro.intro.intro.hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    x : B
    ⊢ IsIntegral A ((Algebra.linearMap B L) x)
  -/
  rw [IsIntegralClosure.isIntegral_iff (A := B)]
  /-
    case intro.intro.intro.hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁴ : CommRing A
    inst✝¹³ : Field K
    inst✝¹² : CommRing B
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra A K
    inst✝⁹ : Algebra B L
    inst✝⁸ : Algebra A B
    inst✝⁷ : Algebra K L
    inst✝⁶ : Algebra A L
    inst✝⁵ : IsScalarTower A K L
    inst✝⁴ : IsScalarTower A B L
    inst✝³ : IsFractionRing A K
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : FiniteDimensional K L
    inst✝ : IsIntegralClosure B A L
    x : B
    ⊢ Exists fun y => Eq ((algebraMap B L) y) ((Algebra.linearMap B L) x)
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


/-- If `b` is an `A`-integral basis of `L` with discriminant `b`, then `d • a * x` is integral over
  `A` for all `a ∈ I` and `x ∈ Iᵛ`. -/
lemma isIntegral_discr_mul_of_mem_traceDual
    (I : Submodule B L) {ι} [DecidableEq ι] [Fintype ι]
    {b : Basis ι K L} (hb : ∀ i, IsIntegral A (b i))
    {a x : L} (ha : a ∈ I) (hx : x ∈ Iᵛ) :
    IsIntegral A ((discr K b) • a * x) := by
  have hinv : IsUnit (traceMatrix K b).det := by
    simpa [← discr_def] using discr_isUnit_of_basis _ b
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    ⊢ IsIntegral A (HMul.hMul (HSMul.hSMul (Algebra.discr K ⇑b) a) x)
  -/
  have H := mulVec_cramer (traceMatrix K b) fun i => trace K L (x * a * b i)
  have : Function.Injective (traceMatrix K b).mulVec := by
    rwa [mulVec_injective_iff_isUnit, isUnit_iff_isUnit_det]
  rw [← traceMatrix_of_basis_mulVec, ← mulVec_smul, this.eq_iff,
    traceMatrix_of_basis_mulVec] at H
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    ⊢ IsIntegral A (HMul.hMul (HSMul.hSMul (Algebra.discr K ⇑b) a) x)
  -/
  rw [← b.equivFun.symm_apply_apply (_ * _), b.equivFun_symm_apply]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    ⊢ IsIntegral A (Finset.univ.sum fun i => HSMul.hSMul (b.equivFun (HMul.hMul (H …
  -/
  apply IsIntegral.sum
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    ⊢ ∀ (x_1 : ι), Membership.mem Finset.univ x_1 → IsIntegral A (HSMul.hSMul (b.e …
  -/
  intro i _
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    ⊢ IsIntegral A (HSMul.hSMul (b.equivFun (HMul.hMul (HSMul.hSMul (Algebra.discr …
  -/
  rw [smul_mul_assoc, b.equivFun.map_smul, discr_def, mul_comm, ← H, Algebra.smul_def]
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    ⊢ IsIntegral A (HMul.hMul ((algebraMap K L) ((Algebra.traceMatrix K ⇑b).cramer …
  -/
  refine RingHom.IsIntegralElem.mul _ ?_ (hb _)
  /-
    case h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    ⊢ (algebraMap A L).IsIntegralElem ((algebraMap K L) ((Algebra.traceMatrix K ⇑b …
  -/
  apply IsIntegral.algebraMap
  /-
    case h.h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    ⊢ IsIntegral A ((Algebra.traceMatrix K ⇑b).cramer (fun i => (Algebra.trace K L …
  -/
  rw [cramer_apply]
  /-
    case h.h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    ⊢ IsIntegral A ((Algebra.traceMatrix K ⇑b).updateCol i fun i => (Algebra.trace …
  -/
  apply IsIntegral.det
  /-
    case h.h.h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    ⊢ ∀ (i_1 j : ι), IsIntegral A ((Algebra.traceMatrix K ⇑b).updateCol i (fun i = …
  -/
  intros j k
  /-
    case h.h.h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    j k : ι
    ⊢ IsIntegral A ((Algebra.traceMatrix K ⇑b).updateCol i (fun i => (Algebra.trac …
  -/
  rw [updateCol_apply]
  /-
    case h.h.h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁷ : CommRing A
    inst✝¹⁶ : Field K
    inst✝¹⁵ : CommRing B
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra A K
    inst✝¹² : Algebra B L
    inst✝¹¹ : Algebra A B
    inst✝¹⁰ : Algebra K L
    inst✝⁹ : Algebra A L
    inst✝⁸ : IsScalarTower A K L
    inst✝⁷ : IsScalarTower A B L
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : Algebra.IsSeparable K L
    I : Submodule B L
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι K L
    hb : ∀ (i : ι), IsIntegral A (b i)
    a x : L
    ha : Membership.mem I a
    hx : Membership.mem (Submodule.traceDual A K I) x
    hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
    H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
    this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
    i : ι
    a✝ : Membership.mem Finset.univ i
    j k : ι
    ⊢ IsIntegral A (ite (Eq k i) ((Algebra.trace K L) (HMul.hMul (HMul.hMul x a) ( …
  -/
  split
    /-
      case h.h.h.isTrue
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : Membership.mem (Submodule.traceDual A K I) x
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Eq k i
      ⊢ IsIntegral A ((Algebra.trace K L) (HMul.hMul (HMul.hMul x a) (b j)))
    -/
  · rw [mul_assoc]
    /-
      case h.h.h.isTrue
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : Membership.mem (Submodule.traceDual A K I) x
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Eq k i
      ⊢ IsIntegral A ((Algebra.trace K L) (HMul.hMul x (HMul.hMul a (b j))))
    -/
    rw [mem_traceDual_iff_isIntegral] at hx
    /-
      case h.h.h.isTrue
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : ∀ (a : L), Membership.mem I a → IsIntegral A (((Algebra.traceForm K L) x) …
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Eq k i
      ⊢ IsIntegral A ((Algebra.trace K L) (HMul.hMul x (HMul.hMul a (b j))))
    -/
    apply hx
    /-
      case h.h.h.isTrue.a
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : ∀ (a : L), Membership.mem I a → IsIntegral A (((Algebra.traceForm K L) x) …
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Eq k i
      ⊢ Membership.mem I (HMul.hMul a (b j))
    -/
    have ⟨y, hy⟩ := (IsIntegralClosure.isIntegral_iff (A := B)).mp (hb j)
    /-
      case h.h.h.isTrue.a
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : ∀ (a : L), Membership.mem I a → IsIntegral A (((Algebra.traceForm K L) x) …
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Eq k i
      y : B
      hy : Eq ((algebraMap B L) y) (b j)
      ⊢ Membership.mem I (HMul.hMul a (b j))
    -/
    rw [mul_comm, ← hy, ← Algebra.smul_def]
    /-
      case h.h.h.isTrue.a
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : ∀ (a : L), Membership.mem I a → IsIntegral A (((Algebra.traceForm K L) x) …
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Eq k i
      y : B
      hy : Eq ((algebraMap B L) y) (b j)
      ⊢ Membership.mem I (HSMul.hSMul y a)
    -/
    exact I.smul_mem _ (ha)
    /-
      🎉 no goals
    -/
    /-
      case h.h.h.isFalse
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁷ : CommRing A
      inst✝¹⁶ : Field K
      inst✝¹⁵ : CommRing B
      inst✝¹⁴ : Field L
      inst✝¹³ : Algebra A K
      inst✝¹² : Algebra B L
      inst✝¹¹ : Algebra A B
      inst✝¹⁰ : Algebra K L
      inst✝⁹ : Algebra A L
      inst✝⁸ : IsScalarTower A K L
      inst✝⁷ : IsScalarTower A B L
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : FiniteDimensional K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : Algebra.IsSeparable K L
      I : Submodule B L
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      b : Basis ι K L
      hb : ∀ (i : ι), IsIntegral A (b i)
      a x : L
      ha : Membership.mem I a
      hx : Membership.mem (Submodule.traceDual A K I) x
      hinv : IsUnit (Algebra.traceMatrix K ⇑b).det
      H : Eq ((Algebra.traceMatrix K ⇑b).cramer fun i => (Algebra.trace K L) (HMul.h …
      this : Function.Injective (Algebra.traceMatrix K ⇑b).mulVec
      i : ι
      a✝ : Membership.mem Finset.univ i
      j k : ι
      h✝ : Not (Eq k i)
      ⊢ IsIntegral A (Algebra.traceMatrix K (⇑b) j k)
    -/
  · exact isIntegral_trace (RingHom.IsIntegralElem.mul _ (hb j) (hb k))
    /-
      🎉 no goals
    -/


open scoped Classical in
/-- The dual of a non-zero fractional ideal is the dual of the submodule under the traceform. -/
noncomputable
def dual (I : FractionalIdeal B⁰ L) :
    FractionalIdeal B⁰ L :=
  if hI : I = 0 then 0 else
  ⟨Iᵛ, by
    classical
    have ⟨s, b, hb⟩ := FiniteDimensional.exists_is_basis_integral A K L
    obtain ⟨x, hx, hx'⟩ := exists_ne_zero_mem_isInteger hI
    have ⟨y, hy⟩ := (IsIntegralClosure.isIntegral_iff (A := B)).mp
      (IsIntegral.algebraMap (B := L) (discr_isIntegral K hb))
    refine ⟨y * x, mem_nonZeroDivisors_iff_ne_zero.mpr (mul_ne_zero ?_ hx), fun z hz ↦ ?_⟩
    · rw [← (IsIntegralClosure.algebraMap_injective B A L).ne_iff, hy, RingHom.map_zero,
        ← (algebraMap K L).map_zero, (algebraMap K L).injective.ne_iff]
      exact discr_not_zero_of_basis K b
    · convert isIntegral_discr_mul_of_mem_traceDual I hb hx' hz using 1
      · ext w; exact (IsIntegralClosure.isIntegral_iff (A := B)).symm
      · rw [Algebra.smul_def, RingHom.map_mul, hy, ← Algebra.smul_def]⟩


lemma coe_dual (hI : I ≠ 0) :
                                            /-
                                              A : Type u_1
                                              K : Type u_2
                                              L : Type u
                                              B : Type u_3
                                              inst✝¹⁸ : CommRing A
                                              inst✝¹⁷ : Field K
                                              inst✝¹⁶ : CommRing B
                                              inst✝¹⁵ : Field L
                                              inst✝¹⁴ : Algebra A K
                                              inst✝¹³ : Algebra B L
                                              inst✝¹² : Algebra A B
                                              inst✝¹¹ : Algebra K L
                                              inst✝¹⁰ : Algebra A L
                                              inst✝⁹ : IsScalarTower A K L
                                              inst✝⁸ : IsScalarTower A B L
                                              inst✝⁷ : IsDomain A
                                              inst✝⁶ : IsFractionRing A K
                                              inst✝⁵ : FiniteDimensional K L
                                              inst✝⁴ : Algebra.IsSeparable K L
                                              inst✝³ : IsIntegralClosure B A L
                                              inst✝² : IsFractionRing B L
                                              inst✝¹ : IsIntegrallyClosed A
                                              inst✝ : IsDedekindDomain B
                                              I : FractionalIdeal (nonZeroDivisors B) L
                                              hI : Ne I 0
                                              ⊢ Eq (↑(FractionalIdeal.dual A K I)) (Submodule.traceDual A K ↑I)
                                            -/
    (dual A K I : Submodule B L) = Iᵛ := by rw [dual, dif_neg hI, coe_mk]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
lemma coe_dual_one :
    (dual A K (1 : FractionalIdeal B⁰ L) : Submodule B L) = 1ᵛ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    ⊢ Eq (↑(FractionalIdeal.dual A K 1)) (Submodule.traceDual A K 1)
  -/
  rw [← coe_one, coe_dual]
  /-
    case hI
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    ⊢ Ne 1 0
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma dual_zero :
                                                  /-
                                                    A : Type u_1
                                                    K : Type u_2
                                                    L : Type u
                                                    B : Type u_3
                                                    inst✝¹⁸ : CommRing A
                                                    inst✝¹⁷ : Field K
                                                    inst✝¹⁶ : CommRing B
                                                    inst✝¹⁵ : Field L
                                                    inst✝¹⁴ : Algebra A K
                                                    inst✝¹³ : Algebra B L
                                                    inst✝¹² : Algebra A B
                                                    inst✝¹¹ : Algebra K L
                                                    inst✝¹⁰ : Algebra A L
                                                    inst✝⁹ : IsScalarTower A K L
                                                    inst✝⁸ : IsScalarTower A B L
                                                    inst✝⁷ : IsDomain A
                                                    inst✝⁶ : IsFractionRing A K
                                                    inst✝⁵ : FiniteDimensional K L
                                                    inst✝⁴ : Algebra.IsSeparable K L
                                                    inst✝³ : IsIntegralClosure B A L
                                                    inst✝² : IsFractionRing B L
                                                    inst✝¹ : IsIntegrallyClosed A
                                                    inst✝ : IsDedekindDomain B
                                                    ⊢ Eq (FractionalIdeal.dual A K 0) 0
                                                  -/
    dual A K (0 : FractionalIdeal B⁰ L) = 0 := by rw [dual, dif_pos rfl]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma mem_dual (hI : I ≠ 0) {x} :
    x ∈ dual A K I ↔ ∀ a ∈ I, traceForm K L x a ∈ (algebraMap A K).range := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    x : L
    ⊢ Iff (Membership.mem (FractionalIdeal.dual A K I) x) (∀ (a : L), Membership.m …
  -/
  rw [dual, dif_neg hI]; exact forall₂_congr fun _ _ ↦ mem_one
                         /-
                           🎉 no goals
                         -/


lemma dual_ne_zero (hI : I ≠ 0) :
    dual A K I ≠ 0 := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    ⊢ Ne (FractionalIdeal.dual A K I) 0
  -/
  obtain ⟨b, hb, hb'⟩ := I.prop
  suffices algebraMap B L b ∈ dual A K I by
    intro e
    rw [e, mem_zero_iff, ← (algebraMap B L).map_zero,
      (IsIntegralClosure.algebraMap_injective B A L).eq_iff] at this
    exact mem_nonZeroDivisors_iff_ne_zero.mp hb this
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    b : B
    hb : Membership.mem (nonZeroDivisors B) b
    hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
    ⊢ Membership.mem (FractionalIdeal.dual A K I) ((algebraMap B L) b)
  -/
  rw [mem_dual hI]
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    b : B
    hb : Membership.mem (nonZeroDivisors B) b
    hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
    ⊢ ∀ (a : L), Membership.mem I a → Membership.mem (algebraMap A K).range (((Alg …
  -/
  intro a ha
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    b : B
    hb : Membership.mem (nonZeroDivisors B) b
    hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
    a : L
    ha : Membership.mem I a
    ⊢ Membership.mem (algebraMap A K).range (((Algebra.traceForm K L) ((algebraMap …
  -/
  apply IsIntegrallyClosed.isIntegral_iff.mp
  /-
    case intro.intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    b : B
    hb : Membership.mem (nonZeroDivisors B) b
    hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
    a : L
    ha : Membership.mem I a
    ⊢ IsIntegral A (((Algebra.traceForm K L) ((algebraMap B L) b)) a)
  -/
  apply isIntegral_trace
  /-
    case intro.intro.hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    b : B
    hb : Membership.mem (nonZeroDivisors B) b
    hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
    a : L
    ha : Membership.mem I a
    ⊢ IsIntegral A (((Algebra.lmul K L).toLinearMap ((algebraMap B L) b)) a)
  -/
  dsimp
  /-
    case intro.intro.hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    b : B
    hb : Membership.mem (nonZeroDivisors B) b
    hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
    a : L
    ha : Membership.mem I a
    ⊢ IsIntegral A (HMul.hMul ((algebraMap B L) b) a)
  -/
  convert hb' a ha using 1
    /-
      case h.e
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I : FractionalIdeal (nonZeroDivisors B) L
      hI : Ne I 0
      b : B
      hb : Membership.mem (nonZeroDivisors B) b
      hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
      a : L
      ha : Membership.mem I a
      ⊢ Eq (IsIntegral A) (IsLocalization.IsInteger B)
    -/
  · ext w
    /-
      case h.e.h.a
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I : FractionalIdeal (nonZeroDivisors B) L
      hI : Ne I 0
      b : B
      hb : Membership.mem (nonZeroDivisors B) b
      hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
      a : L
      ha : Membership.mem I a
      w : L
      ⊢ Iff (IsIntegral A w) (IsLocalization.IsInteger B w)
    -/
    exact IsIntegralClosure.isIntegral_iff (A := B)
    /-
      🎉 no goals
    -/
    /-
      case h.e'_1
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I : FractionalIdeal (nonZeroDivisors B) L
      hI : Ne I 0
      b : B
      hb : Membership.mem (nonZeroDivisors B) b
      hb' : ∀ (b_1 : L), Membership.mem (↑I) b_1 → IsLocalization.IsInteger B (HSMul …
      a : L
      ha : Membership.mem I a
      ⊢ Eq (HMul.hMul ((algebraMap B L) b) a) (HSMul.hSMul b a)
    -/
  · exact (Algebra.smul_def _ _).symm
    /-
      🎉 no goals
    -/


@[simp]
lemma dual_eq_zero_iff :
    dual A K I = 0 ↔ I = 0 :=
  ⟨not_imp_not.mp (dual_ne_zero A K), fun e ↦ e.symm ▸ dual_zero A K L B⟩


lemma dual_ne_zero_iff :
    dual A K I ≠ 0 ↔ I ≠ 0 := dual_eq_zero_iff.not


lemma le_dual_inv_aux (hI : I ≠ 0) (hIJ : I * J ≤ 1) :
    J ≤ dual A K I := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    ⊢ LE.le J (FractionalIdeal.dual A K I)
  -/
  rw [dual, dif_neg hI]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    ⊢ LE.le J ⟨Submodule.traceDual A K ↑I, ⋯⟩
  -/
  intro x hx y hy
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    ⊢ Membership.mem 1 (((Algebra.traceForm K L) x) y)
  -/
  rw [mem_one]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    ⊢ Exists fun y_1 => Eq ((algebraMap A K) y_1) (((Algebra.traceForm K L) x) y)
  -/
  apply IsIntegrallyClosed.isIntegral_iff.mp
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    ⊢ IsIntegral A (((Algebra.traceForm K L) x) y)
  -/
  apply isIntegral_trace
  /-
    case hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    ⊢ IsIntegral A (((Algebra.lmul K L).toLinearMap x) y)
  -/
  rw [IsIntegralClosure.isIntegral_iff (A := B)]
  /-
    case hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    ⊢ Exists fun y_1 => Eq ((algebraMap B L) y_1) (((Algebra.lmul K L).toLinearMap …
  -/
  have ⟨z, _, hz⟩ := hIJ (FractionalIdeal.mul_mem_mul hy hx)
  /-
    case hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    z : B
    left✝ : Membership.mem (↑Top.top) z
    hz : Eq ((Algebra.linearMap B L) z) (HMul.hMul y x)
    ⊢ Exists fun y_1 => Eq ((algebraMap B L) y_1) (((Algebra.lmul K L).toLinearMap …
  -/
  rw [mul_comm] at hz
  /-
    case hx
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hIJ : LE.le (HMul.hMul I J) 1
    x : L
    hx : Membership.mem ((fun a => ↑a) J) x
    y : L
    hy : Membership.mem (Submodule.restrictScalars A ↑I) y
    z : B
    left✝ : Membership.mem (↑Top.top) z
    hz : Eq ((Algebra.linearMap B L) z) (HMul.hMul x y)
    ⊢ Exists fun y_1 => Eq ((algebraMap B L) y_1) (((Algebra.lmul K L).toLinearMap …
  -/
  exact ⟨z, hz⟩
  /-
    🎉 no goals
  -/


lemma one_le_dual_one :
    1 ≤ dual A K (1 : FractionalIdeal B⁰ L) :=
                                      /-
                                        A : Type u_1
                                        K : Type u_2
                                        L : Type u
                                        B : Type u_3
                                        inst✝¹⁸ : CommRing A
                                        inst✝¹⁷ : Field K
                                        inst✝¹⁶ : CommRing B
                                        inst✝¹⁵ : Field L
                                        inst✝¹⁴ : Algebra A K
                                        inst✝¹³ : Algebra B L
                                        inst✝¹² : Algebra A B
                                        inst✝¹¹ : Algebra K L
                                        inst✝¹⁰ : Algebra A L
                                        inst✝⁹ : IsScalarTower A K L
                                        inst✝⁸ : IsScalarTower A B L
                                        inst✝⁷ : IsDomain A
                                        inst✝⁶ : IsFractionRing A K
                                        inst✝⁵ : FiniteDimensional K L
                                        inst✝⁴ : Algebra.IsSeparable K L
                                        inst✝³ : IsIntegralClosure B A L
                                        inst✝² : IsFractionRing B L
                                        inst✝¹ : IsIntegrallyClosed A
                                        inst✝ : IsDedekindDomain B
                                        ⊢ LE.le (HMul.hMul 1 1) 1
                                      -/
  le_dual_inv_aux A K one_ne_zero (by rw [one_mul])
                                      /-
                                        🎉 no goals
                                      -/


lemma le_dual_iff (hJ : J ≠ 0) :
    I ≤ dual A K J ↔ I * J ≤ dual A K 1 := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hJ : Ne J 0
    ⊢ Iff (LE.le I (FractionalIdeal.dual A K J)) (LE.le (HMul.hMul I J) (Fractiona …
  -/
  by_cases hI : I = 0
    /-
      case pos
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I J : FractionalIdeal (nonZeroDivisors B) L
      hJ : Ne J 0
      hI : Eq I 0
      ⊢ Iff (LE.le I (FractionalIdeal.dual A K J)) (LE.le (HMul.hMul I J) (Fractiona …
    -/
  · simp [hI, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hJ : Ne J 0
    hI : Not (Eq I 0)
    ⊢ Iff (LE.le I (FractionalIdeal.dual A K J)) (LE.le (HMul.hMul I J) (Fractiona …
  -/
  rw [← coe_le_coe, ← coe_le_coe, coe_mul, coe_dual A K hJ, coe_dual_one, le_traceDual]
  /-
    🎉 no goals
  -/


lemma inv_le_dual :
    I⁻¹ ≤ dual A K I := by
  classical
  exact if hI : I = 0 then by simp [hI] else le_dual_inv_aux A K hI (le_of_eq (mul_inv_cancel₀ hI))


lemma dual_inv_le :
    (dual A K I)⁻¹ ≤ I := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    ⊢ LE.le (Inv.inv (FractionalIdeal.dual A K I)) I
  -/
  by_cases hI : I = 0; · simp [hI]
                         /-
                           🎉 no goals
                         -/
  convert mul_right_mono ((dual A K I)⁻¹)
    (mul_left_mono I (inv_le_dual A K I)) using 1
    /-
      case h.e'_3
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I : FractionalIdeal (nonZeroDivisors B) L
      hI : Not (Eq I 0)
      ⊢ Eq (Inv.inv (FractionalIdeal.dual A K I)) ((fun J => HMul.hMul J (Inv.inv (F …
    -/
  · simp only [mul_inv_cancel₀ hI, one_mul]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I : FractionalIdeal (nonZeroDivisors B) L
      hI : Not (Eq I 0)
      ⊢ Eq I ((fun J => HMul.hMul J (Inv.inv (FractionalIdeal.dual A K I))) ((fun x  …
    -/
  · simp only [mul_inv_cancel₀ (dual_ne_zero A K (hI := hI)), mul_assoc, mul_one]
    /-
      🎉 no goals
    -/


lemma dual_eq_mul_inv :
    dual A K I = dual A K 1 * I⁻¹ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    ⊢ Eq (FractionalIdeal.dual A K I) (HMul.hMul (FractionalIdeal.dual A K 1) (Inv …
  -/
  by_cases hI : I = 0; · simp [hI]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Not (Eq I 0)
    ⊢ Eq (FractionalIdeal.dual A K I) (HMul.hMul (FractionalIdeal.dual A K 1) (Inv …
  -/
  apply le_antisymm
  · suffices dual A K I * I ≤ dual A K 1 by
      convert mul_right_mono I⁻¹ this using 1; simp only [mul_inv_cancel₀ hI, mul_one, mul_assoc]
    /-
      case neg.a
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I : FractionalIdeal (nonZeroDivisors B) L
      hI : Not (Eq I 0)
      ⊢ LE.le (HMul.hMul (FractionalIdeal.dual A K I) I) (FractionalIdeal.dual A K 1)
    -/
    rw [← le_dual_iff A K hI]
    /-
      🎉 no goals
    -/
  /-
    case neg.a
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Not (Eq I 0)
    ⊢ LE.le (HMul.hMul (FractionalIdeal.dual A K 1) (Inv.inv I)) (FractionalIdeal. …
  -/
  rw [le_dual_iff A K hI, mul_assoc, inv_mul_cancel₀ hI, mul_one]
  /-
    🎉 no goals
  -/


lemma dual_div_dual :
    dual A K J / dual A K I = I / J := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    ⊢ Eq (HDiv.hDiv (FractionalIdeal.dual A K J) (FractionalIdeal.dual A K I)) (HD …
  -/
  rw [dual_eq_mul_inv A K J, dual_eq_mul_inv A K I, mul_div_mul_comm, div_self, one_mul]
    /-
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I J : FractionalIdeal (nonZeroDivisors B) L
      ⊢ Eq (HDiv.hDiv (Inv.inv J) (Inv.inv I)) (HDiv.hDiv I J)
    -/
  · exact inv_div_inv J I
    /-
      🎉 no goals
    -/
    /-
      A : Type u_1
      K : Type u_2
      L : Type u
      B : Type u_3
      inst✝¹⁸ : CommRing A
      inst✝¹⁷ : Field K
      inst✝¹⁶ : CommRing B
      inst✝¹⁵ : Field L
      inst✝¹⁴ : Algebra A K
      inst✝¹³ : Algebra B L
      inst✝¹² : Algebra A B
      inst✝¹¹ : Algebra K L
      inst✝¹⁰ : Algebra A L
      inst✝⁹ : IsScalarTower A K L
      inst✝⁸ : IsScalarTower A B L
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsFractionRing A K
      inst✝⁵ : FiniteDimensional K L
      inst✝⁴ : Algebra.IsSeparable K L
      inst✝³ : IsIntegralClosure B A L
      inst✝² : IsFractionRing B L
      inst✝¹ : IsIntegrallyClosed A
      inst✝ : IsDedekindDomain B
      I J : FractionalIdeal (nonZeroDivisors B) L
      ⊢ Ne (FractionalIdeal.dual A K 1) 0
    -/
  · simp only [ne_eq, dual_eq_zero_iff, one_ne_zero, not_false_eq_true]
    /-
      🎉 no goals
    -/


lemma dual_mul_self (hI : I ≠ 0) :
    dual A K I * I = dual A K 1 := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    ⊢ Eq (HMul.hMul (FractionalIdeal.dual A K I) I) (FractionalIdeal.dual A K 1)
  -/
  rw [dual_eq_mul_inv, mul_assoc, inv_mul_cancel₀ hI, mul_one]
  /-
    🎉 no goals
  -/


lemma self_mul_dual (hI : I ≠ 0) :
    I * dual A K I = dual A K 1 := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    ⊢ Eq (HMul.hMul I (FractionalIdeal.dual A K I)) (FractionalIdeal.dual A K 1)
  -/
  rw [mul_comm, dual_mul_self A K hI]
  /-
    🎉 no goals
  -/


lemma dual_inv :
                                        /-
                                          A : Type u_1
                                          K : Type u_2
                                          L : Type u
                                          B : Type u_3
                                          inst✝¹⁸ : CommRing A
                                          inst✝¹⁷ : Field K
                                          inst✝¹⁶ : CommRing B
                                          inst✝¹⁵ : Field L
                                          inst✝¹⁴ : Algebra A K
                                          inst✝¹³ : Algebra B L
                                          inst✝¹² : Algebra A B
                                          inst✝¹¹ : Algebra K L
                                          inst✝¹⁰ : Algebra A L
                                          inst✝⁹ : IsScalarTower A K L
                                          inst✝⁸ : IsScalarTower A B L
                                          inst✝⁷ : IsDomain A
                                          inst✝⁶ : IsFractionRing A K
                                          inst✝⁵ : FiniteDimensional K L
                                          inst✝⁴ : Algebra.IsSeparable K L
                                          inst✝³ : IsIntegralClosure B A L
                                          inst✝² : IsFractionRing B L
                                          inst✝¹ : IsIntegrallyClosed A
                                          inst✝ : IsDedekindDomain B
                                          I : FractionalIdeal (nonZeroDivisors B) L
                                          ⊢ Eq (FractionalIdeal.dual A K (Inv.inv I)) (HMul.hMul (FractionalIdeal.dual A …
                                        -/
    dual A K I⁻¹ = dual A K 1 * I := by rw [dual_eq_mul_inv, inv_inv]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
lemma dual_dual :
    dual A K (dual A K I) = I := by
  rw [dual_eq_mul_inv, dual_eq_mul_inv A K (I := I), mul_inv, inv_inv, ← mul_assoc, mul_inv_cancel₀,
    one_mul]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    ⊢ Ne (FractionalIdeal.dual A K 1) 0
  -/
  rw [dual_ne_zero_iff]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I : FractionalIdeal (nonZeroDivisors B) L
    ⊢ Ne 1 0
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma dual_le_dual (hI : I ≠ 0) (hJ : J ≠ 0) :
    dual A K I ≤ dual A K J ↔ J ≤ I := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hJ : Ne J 0
    ⊢ Iff (LE.le (FractionalIdeal.dual A K I) (FractionalIdeal.dual A K J)) (LE.le …
  -/
  nth_rewrite 2 [← dual_dual A K I]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsFractionRing B L
    inst✝¹ : IsIntegrallyClosed A
    inst✝ : IsDedekindDomain B
    I J : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    hJ : Ne J 0
    ⊢ Iff (LE.le (FractionalIdeal.dual A K I) (FractionalIdeal.dual A K J)) (LE.le …
  -/
  rw [le_dual_iff A K hJ, le_dual_iff A K (I := J) (by rwa [dual_ne_zero_iff]), mul_comm]
  /-
    🎉 no goals
  -/


lemma dual_involutive :
    Function.Involutive (dual A K : FractionalIdeal B⁰ L → FractionalIdeal B⁰ L) := dual_dual A K


lemma dual_injective :
    Function.Injective (dual A K : FractionalIdeal B⁰ L → FractionalIdeal B⁰ L) :=
  dual_involutive.injective


/-- The different ideal of an extension of integral domains `B/A` is the inverse of the dual of `A`
as an ideal of `B`. See `coeIdeal_differentIdeal` and `coeSubmodule_differentIdeal`. -/
def differentIdeal [NoZeroSMulDivisors A B] : Ideal B :=
  (1 / Submodule.traceDual A (FractionRing A) 1 : Submodule B (FractionRing B)).comap
    (Algebra.linearMap B (FractionRing B))


lemma coeSubmodule_differentIdeal_fractionRing
    [NoZeroSMulDivisors A B] [Algebra.IsIntegral A B]
    [Algebra.IsSeparable (FractionRing A) (FractionRing B)]
    [FiniteDimensional (FractionRing A) (FractionRing B)] :
    coeSubmodule (FractionRing B) (differentIdeal A B) =
      1 / Submodule.traceDual A (FractionRing A) 1 := by
  have : IsIntegralClosure B A (FractionRing B) :=
    IsIntegralClosure.of_isIntegrallyClosed _ _ _
  /-
    A : Type u_1
    B : Type u_3
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDedekindDomain B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : IsIntegralClosure B A (FractionRing B)
    ⊢ Eq (IsLocalization.coeSubmodule (FractionRing B) (differentIdeal A B)) (HDiv …
  -/
  rw [coeSubmodule, differentIdeal, Submodule.map_comap_eq, inf_eq_right]
  have := FractionalIdeal.dual_inv_le (A := A) (K := FractionRing A)
    (1 : FractionalIdeal B⁰ (FractionRing B))
  /-
    A : Type u_1
    B : Type u_3
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDedekindDomain B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this✝ : IsIntegralClosure B A (FractionRing B)
    this : LE.le (Inv.inv (FractionalIdeal.dual A (FractionRing A) 1)) 1
    ⊢ LE.le (HDiv.hDiv 1 (Submodule.traceDual A (FractionRing A) 1)) (LinearMap.ra …
  -/
  have : _ ≤ ((1 : FractionalIdeal B⁰ (FractionRing B)) : Submodule B (FractionRing B)) := this
  /-
    A : Type u_1
    B : Type u_3
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    inst✝⁶ : IsDomain A
    inst✝⁵ : IsIntegrallyClosed A
    inst✝⁴ : IsDedekindDomain B
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Algebra.IsIntegral A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    inst✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this✝¹ : IsIntegralClosure B A (FractionRing B)
    this✝ : LE.le (Inv.inv (FractionalIdeal.dual A (FractionRing A) 1)) 1
    this : LE.le ((fun a => ↑a) (Inv.inv (FractionalIdeal.dual A (FractionRing A)  …
    ⊢ LE.le (HDiv.hDiv 1 (Submodule.traceDual A (FractionRing A) 1)) (LinearMap.ra …
  -/
  simp only [← one_div, FractionalIdeal.val_eq_coe] at this
  rw [FractionalIdeal.coe_div (FractionalIdeal.dual_ne_zero _ _ _),
    FractionalIdeal.coe_dual] at this
    /-
      A : Type u_1
      B : Type u_3
      inst✝⁹ : CommRing A
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra A B
      inst✝⁶ : IsDomain A
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : IsDedekindDomain B
      inst✝³ : NoZeroSMulDivisors A B
      inst✝² : Algebra.IsIntegral A B
      inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
      inst✝ : FiniteDimensional (FractionRing A) (FractionRing B)
      this✝¹ : IsIntegralClosure B A (FractionRing B)
      this✝ : LE.le (Inv.inv (FractionalIdeal.dual A (FractionRing A) 1)) 1
      this : LE.le (HDiv.hDiv (↑1) (Submodule.traceDual A (FractionRing A) ↑1)) ↑1
      ⊢ LE.le (HDiv.hDiv 1 (Submodule.traceDual A (FractionRing A) 1)) (LinearMap.ra …
    -/
  · simpa only [FractionalIdeal.coe_one, Submodule.one_eq_range] using this
    /-
      🎉 no goals
    -/
    /-
      case hI
      A : Type u_1
      B : Type u_3
      inst✝⁹ : CommRing A
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra A B
      inst✝⁶ : IsDomain A
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : IsDedekindDomain B
      inst✝³ : NoZeroSMulDivisors A B
      inst✝² : Algebra.IsIntegral A B
      inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
      inst✝ : FiniteDimensional (FractionRing A) (FractionRing B)
      this✝¹ : IsIntegralClosure B A (FractionRing B)
      this✝ : LE.le (Inv.inv (FractionalIdeal.dual A (FractionRing A) 1)) 1
      this : LE.le (HDiv.hDiv ↑1 ↑(FractionalIdeal.dual A (FractionRing A) 1)) ↑1
      ⊢ Ne 1 0
    -/
  · exact one_ne_zero
    /-
      🎉 no goals
    -/
    /-
      A : Type u_1
      B : Type u_3
      inst✝⁹ : CommRing A
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra A B
      inst✝⁶ : IsDomain A
      inst✝⁵ : IsIntegrallyClosed A
      inst✝⁴ : IsDedekindDomain B
      inst✝³ : NoZeroSMulDivisors A B
      inst✝² : Algebra.IsIntegral A B
      inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
      inst✝ : FiniteDimensional (FractionRing A) (FractionRing B)
      this✝¹ : IsIntegralClosure B A (FractionRing B)
      this✝ : LE.le (Inv.inv (FractionalIdeal.dual A (FractionRing A) 1)) 1
      this : LE.le ↑(HDiv.hDiv 1 (FractionalIdeal.dual A (FractionRing A) 1)) ↑1
      ⊢ Ne 1 0
    -/
  · exact one_ne_zero
    /-
      🎉 no goals
    -/


lemma coeSubmodule_differentIdeal [NoZeroSMulDivisors A B] :
    coeSubmodule L (differentIdeal A B) = 1 / Submodule.traceDual A K 1 := by
  have : (FractionRing.algEquiv B L).toLinearEquiv.comp (Algebra.linearMap B (FractionRing B)) =
    Algebra.linearMap B L := by ext; simp
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linearMa …
    ⊢ Eq (IsLocalization.coeSubmodule L (differentIdeal A B)) (HDiv.hDiv 1 (Submod …
  -/
  rw [coeSubmodule, ← this]
  have H : RingHom.comp (algebraMap (FractionRing A) (FractionRing B))
      ↑(FractionRing.algEquiv A K).symm.toRingEquiv =
        RingHom.comp ↑(FractionRing.algEquiv B L).symm.toRingEquiv (algebraMap K L) := by
    apply IsLocalization.ringHom_ext A⁰
    ext
    simp only [AlgEquiv.toRingEquiv_eq_coe, RingHom.coe_comp, RingHom.coe_coe,
      AlgEquiv.coe_ringEquiv, Function.comp_apply, AlgEquiv.commutes,
      ← IsScalarTower.algebraMap_apply]
    rw [IsScalarTower.algebraMap_apply A B L, AlgEquiv.commutes, ← IsScalarTower.algebraMap_apply]
  have : Algebra.IsSeparable (FractionRing A) (FractionRing B) :=
    Algebra.IsSeparable.of_equiv_equiv _ _ H
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝ : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linearM …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    ⊢ Eq (Submodule.map ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebr …
  -/
  have : FiniteDimensional (FractionRing A) (FractionRing B) := Module.Finite.of_equiv_equiv _ _ H
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝¹ : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Eq (Submodule.map ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebr …
  -/
  have : Algebra.IsIntegral A B := IsIntegralClosure.isIntegral_algebra _ L
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝² : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : Algebra.IsIntegral A B
    ⊢ Eq (Submodule.map ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebr …
  -/
  simp only [AlgEquiv.toLinearEquiv_toLinearMap, Submodule.map_comp]
  rw [← coeSubmodule, coeSubmodule_differentIdeal_fractionRing _ _,
    Submodule.map_div, ← AlgEquiv.toAlgHom_toLinearMap, Submodule.map_one]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝² : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : Algebra.IsIntegral A B
    ⊢ Eq (HDiv.hDiv 1 (Submodule.map (↑(FractionRing.algEquiv B L)).toLinearMap (S …
  -/
  congr 1
  /-
    case e_a
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝² : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : Algebra.IsIntegral A B
    ⊢ Eq (Submodule.map (↑(FractionRing.algEquiv B L)).toLinearMap (Submodule.trac …
  -/
  refine (map_equiv_traceDual A K _).trans ?_
  /-
    case e_a
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝² : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : Algebra.IsIntegral A B
    ⊢ Eq (Submodule.traceDual A K (Submodule.map (FractionRing.algEquiv B L) 1)) ( …
  -/
  congr 1
  /-
    case e_a.e_I
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝² : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : Algebra.IsIntegral A B
    ⊢ Eq (Submodule.map (FractionRing.algEquiv B L) 1) 1
  -/
  ext
  /-
    case e_a.e_I.h
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    this✝² : Eq ((↑(FractionRing.algEquiv B L).toLinearEquiv).comp (Algebra.linear …
    H : Eq ((algebraMap (FractionRing A) (FractionRing B)).comp ↑(FractionRing.alg …
    this✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    this✝ : FiniteDimensional (FractionRing A) (FractionRing B)
    this : Algebra.IsIntegral A B
    x✝ : L
    ⊢ Iff (Membership.mem (Submodule.map (FractionRing.algEquiv B L) 1) x✝) (Membe …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma coeIdeal_differentIdeal [NoZeroSMulDivisors A B] :
    ↑(differentIdeal A B) = (FractionalIdeal.dual A K (1 : FractionalIdeal B⁰ L))⁻¹ := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    inst✝ : NoZeroSMulDivisors A B
    ⊢ Eq (↑(differentIdeal A B)) (Inv.inv (FractionalIdeal.dual A K 1))
  -/
  apply FractionalIdeal.coeToSubmodule_injective
  simp only [FractionalIdeal.coe_div
    (FractionalIdeal.dual_ne_zero _ _ (@one_ne_zero (FractionalIdeal B⁰ L) _ _ _)),
    FractionalIdeal.coe_coeIdeal, coeSubmodule_differentIdeal A K, inv_eq_one_div,
    FractionalIdeal.coe_dual_one, FractionalIdeal.coe_one]


lemma differentialIdeal_le_fractionalIdeal_iff
    {I : FractionalIdeal B⁰ L} (hI : I ≠ 0) [NoZeroSMulDivisors A B] :
    differentIdeal A B ≤ I ↔ (((I⁻¹ : _) : Submodule B L).restrictScalars A).map
      ((Algebra.trace K L).restrictScalars A) ≤ 1 := by
  rw [coeIdeal_differentIdeal A K L B, FractionalIdeal.inv_le_comm (by simp) hI,
    ← FractionalIdeal.coe_le_coe, FractionalIdeal.coe_dual_one]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    inst✝ : NoZeroSMulDivisors A B
    ⊢ Iff (LE.le (↑(Inv.inv I)) (Submodule.traceDual A K 1)) (LE.le (Submodule.map …
  -/
  refine le_traceDual_iff_map_le_one.trans ?_
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : Field K
    inst✝¹⁷ : CommRing B
    inst✝¹⁶ : Field L
    inst✝¹⁵ : Algebra A K
    inst✝¹⁴ : Algebra B L
    inst✝¹³ : Algebra A B
    inst✝¹² : Algebra K L
    inst✝¹¹ : Algebra A L
    inst✝¹⁰ : IsScalarTower A K L
    inst✝⁹ : IsScalarTower A B L
    inst✝⁸ : IsDomain A
    inst✝⁷ : IsFractionRing A K
    inst✝⁶ : FiniteDimensional K L
    inst✝⁵ : Algebra.IsSeparable K L
    inst✝⁴ : IsIntegralClosure B A L
    inst✝³ : IsIntegrallyClosed A
    inst✝² : IsDedekindDomain B
    inst✝¹ : IsFractionRing B L
    I : FractionalIdeal (nonZeroDivisors B) L
    hI : Ne I 0
    inst✝ : NoZeroSMulDivisors A B
    ⊢ Iff (LE.le (Submodule.map (↑A (Algebra.trace K L)) (Submodule.restrictScalar …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma differentialIdeal_le_iff {I : Ideal B} (hI : I ≠ ⊥) [NoZeroSMulDivisors A B] :
    differentIdeal A B ≤ I ↔ (((I⁻¹ : FractionalIdeal B⁰ L) : Submodule B L).restrictScalars A).map
      ((Algebra.trace K L).restrictScalars A) ≤ 1 :=
  (FractionalIdeal.coeIdeal_le_coeIdeal _).symm.trans
                                                                                    /-
                                                                                      A : Type u_1
                                                                                      K : Type u_2
                                                                                      L : Type u
                                                                                      B : Type u_3
                                                                                      inst✝¹⁹ : CommRing A
                                                                                      inst✝¹⁸ : Field K
                                                                                      inst✝¹⁷ : CommRing B
                                                                                      inst✝¹⁶ : Field L
                                                                                      inst✝¹⁵ : Algebra A K
                                                                                      inst✝¹⁴ : Algebra B L
                                                                                      inst✝¹³ : Algebra A B
                                                                                      inst✝¹² : Algebra K L
                                                                                      inst✝¹¹ : Algebra A L
                                                                                      inst✝¹⁰ : IsScalarTower A K L
                                                                                      inst✝⁹ : IsScalarTower A B L
                                                                                      inst✝⁸ : IsDomain A
                                                                                      inst✝⁷ : IsFractionRing A K
                                                                                      inst✝⁶ : FiniteDimensional K L
                                                                                      inst✝⁵ : Algebra.IsSeparable K L
                                                                                      inst✝⁴ : IsIntegralClosure B A L
                                                                                      inst✝³ : IsIntegrallyClosed A
                                                                                      inst✝² : IsDedekindDomain B
                                                                                      inst✝¹ : IsFractionRing B L
                                                                                      I : Ideal B
                                                                                      hI : Ne I Bot.bot
                                                                                      inst✝ : NoZeroSMulDivisors A B
                                                                                      ⊢ Ne (↑I) 0
                                                                                    -/
    (differentialIdeal_le_fractionalIdeal_iff (I := (I : FractionalIdeal B⁰ L)) (by simpa))
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


open Pointwise Polynomial in
lemma traceForm_dualSubmodule_adjoin
    {x : L} (hx : Algebra.adjoin K {x} = ⊤) (hAx : IsIntegral A x) :
    (traceForm K L).dualSubmodule (Subalgebra.toSubmodule (Algebra.adjoin A {x})) =
      (aeval x (derivative <| minpoly K x) : L)⁻¹ •
        (Subalgebra.toSubmodule (Algebra.adjoin A {x})) := by
  classical
  have hKx : IsIntegral K x := Algebra.IsIntegral.isIntegral x
  let pb := (Algebra.adjoin.powerBasis' hKx).map
    ((Subalgebra.equivOfEq _ _ hx).trans (Subalgebra.topEquiv))
  have pbgen : pb.gen = x := by simp [pb]
  have hpb : ⇑(LinearMap.BilinForm.dualBasis (traceForm K L) _ pb.basis) = _ :=
    _root_.funext (traceForm_dualBasis_powerBasis_eq pb)
  have : (Subalgebra.toSubmodule (Algebra.adjoin A {x})) =
      Submodule.span A (Set.range pb.basis) := by
    rw [← span_range_natDegree_eq_adjoin (minpoly.monic hAx) (minpoly.aeval _ _)]
    congr; ext y
    have : natDegree (minpoly A x) = natDegree (minpoly K x) := by
      rw [minpoly.isIntegrallyClosed_eq_field_fractions' K hAx, (minpoly.monic hAx).natDegree_map]
    simp only [Finset.coe_image, Finset.coe_range, Set.mem_image, Set.mem_Iio, Set.mem_range,
      pb.basis_eq_pow, pbgen]
    simp only [PowerBasis.map_dim, adjoin.powerBasis'_dim, this]
    exact ⟨fun ⟨a, b, c⟩ ↦ ⟨⟨a, b⟩, c⟩, fun ⟨⟨a, b⟩, c⟩ ↦ ⟨a, b, c⟩⟩
  clear_value pb
  conv_lhs => rw [this]
  rw [← span_coeff_minpolyDiv hAx, LinearMap.BilinForm.dualSubmodule_span_of_basis,
    Submodule.smul_span, hpb]
  show _ = Submodule.span A (_ '' _)
  simp only [← Set.range_comp, smul_eq_mul, div_eq_inv_mul, pbgen,
    minpolyDiv_eq_of_isIntegrallyClosed K hAx]
  apply le_antisymm <;> rw [Submodule.span_le]
  · rintro _ ⟨i, rfl⟩; exact Submodule.subset_span ⟨i, rfl⟩
  · rintro _ ⟨i, rfl⟩
    by_cases hi : i < pb.dim
    · exact Submodule.subset_span ⟨⟨i, hi⟩, rfl⟩
    · rw [Function.comp_apply, coeff_eq_zero_of_natDegree_lt, mul_zero]
      · exact zero_mem _
      rw [← pb.natDegree_minpoly, pbgen, ← natDegree_minpolyDiv_succ hKx,
        ← Nat.succ_eq_add_one] at hi
      exact le_of_not_lt hi


open Polynomial Pointwise in
lemma conductor_mul_differentIdeal [NoZeroSMulDivisors A B]
    (x : B) (hx : Algebra.adjoin K {algebraMap B L x} = ⊤) :
    (conductor A x) * differentIdeal A B = Ideal.span {aeval x (derivative (minpoly A x))} := by
  classical
  have hAx : IsIntegral A x := IsIntegralClosure.isIntegral A L x
  haveI := IsIntegralClosure.isFractionRing_of_finite_extension A K L B
  apply FractionalIdeal.coeIdeal_injective (K := L)
  simp only [FractionalIdeal.coeIdeal_mul, FractionalIdeal.coeIdeal_span_singleton]
  rw [coeIdeal_differentIdeal A K L B,
    mul_inv_eq_iff_eq_mul₀]
  swap
  · exact FractionalIdeal.dual_ne_zero A K one_ne_zero
  apply FractionalIdeal.coeToSubmodule_injective
  simp only [FractionalIdeal.coe_coeIdeal, FractionalIdeal.coe_mul,
    FractionalIdeal.coe_spanSingleton, Submodule.span_singleton_mul]
  ext y
  have hne₁ : aeval (algebraMap B L x) (derivative (minpoly K (algebraMap B L x))) ≠ 0 :=
    (Algebra.IsSeparable.isSeparable _ _).aeval_derivative_ne_zero (minpoly.aeval _ _)
  have : algebraMap B L (aeval x (derivative (minpoly A x))) ≠ 0 := by
    rwa [minpoly.isIntegrallyClosed_eq_field_fractions K L hAx, derivative_map,
      aeval_map_algebraMap, aeval_algebraMap_apply] at hne₁
  rw [Submodule.mem_smul_iff_inv_mul_mem this, FractionalIdeal.mem_coe, FractionalIdeal.mem_dual,
    mem_coeSubmodule_conductor]
  swap
  · exact one_ne_zero
  have hne₂ : (aeval (algebraMap B L x) (derivative (minpoly K (algebraMap B L x))))⁻¹ ≠ 0 := by
    rwa [ne_eq, inv_eq_zero]
  have : IsIntegral A (algebraMap B L x) := IsIntegral.map (IsScalarTower.toAlgHom A B L) hAx
  simp_rw [← Subalgebra.mem_toSubmodule, ← Submodule.mul_mem_smul_iff (y := y * _)
    (mem_nonZeroDivisors_of_ne_zero hne₂)]
  rw [← traceForm_dualSubmodule_adjoin A K hx this]
  simp only [LinearMap.BilinForm.mem_dualSubmodule, traceForm_apply, Subalgebra.mem_toSubmodule,
    minpoly.isIntegrallyClosed_eq_field_fractions K L hAx,
    derivative_map, aeval_map_algebraMap, aeval_algebraMap_apply, mul_assoc,
    FractionalIdeal.mem_one_iff, forall_exists_index, forall_apply_eq_imp_iff]
  simp_rw [← IsScalarTower.toAlgHom_apply A B L x, ← AlgHom.map_adjoin_singleton]
  simp only [Subalgebra.mem_map, IsScalarTower.coe_toAlgHom', Submodule.one_eq_range,
    forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, ← _root_.map_mul]
  exact ⟨fun H b ↦ (mul_one b) ▸ H b 1 (one_mem _), fun H _ _ _ ↦ H _⟩


open Polynomial Pointwise in
lemma aeval_derivative_mem_differentIdeal [NoZeroSMulDivisors A B]
    (x : B) (hx : Algebra.adjoin K {algebraMap B L x} = ⊤) :
    aeval x (derivative (minpoly A x)) ∈ differentIdeal A B := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : IsDedekindDomain B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    hx : Eq (Algebra.adjoin K (Singleton.singleton ((algebraMap B L) x))) Top.top
    ⊢ Membership.mem (differentIdeal A B) ((Polynomial.aeval x) (Polynomial.deriva …
  -/
  refine SetLike.le_def.mp ?_ (Ideal.mem_span_singleton_self _)
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : IsDedekindDomain B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    hx : Eq (Algebra.adjoin K (Singleton.singleton ((algebraMap B L) x))) Top.top
    ⊢ LE.le (Ideal.span (Singleton.singleton ((Polynomial.aeval x) (Polynomial.der …
  -/
  rw [← conductor_mul_differentIdeal A K L x hx]
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝¹⁸ : CommRing A
    inst✝¹⁷ : Field K
    inst✝¹⁶ : CommRing B
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra A K
    inst✝¹³ : Algebra B L
    inst✝¹² : Algebra A B
    inst✝¹¹ : Algebra K L
    inst✝¹⁰ : Algebra A L
    inst✝⁹ : IsScalarTower A K L
    inst✝⁸ : IsScalarTower A B L
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsFractionRing A K
    inst✝⁵ : FiniteDimensional K L
    inst✝⁴ : Algebra.IsSeparable K L
    inst✝³ : IsIntegralClosure B A L
    inst✝² : IsIntegrallyClosed A
    inst✝¹ : IsDedekindDomain B
    inst✝ : NoZeroSMulDivisors A B
    x : B
    hx : Eq (Algebra.adjoin K (Singleton.singleton ((algebraMap B L) x))) Top.top
    ⊢ LE.le (HMul.hMul (conductor A x) (differentIdeal A B)) (differentIdeal A B)
  -/
  exact Ideal.mul_le_left
  /-
    🎉 no goals
  -/


include K L in
lemma pow_sub_one_dvd_differentIdeal_aux [IsFractionRing B L] [IsDedekindDomain A]
    [NoZeroSMulDivisors A B] [Module.Finite A B]
    {p : Ideal A} [p.IsMaximal] (P : Ideal B) {e : ℕ} (he : e ≠ 0) (hp : p ≠ ⊥)
    (hP : P ^ e ∣ p.map (algebraMap A B)) : P ^ (e - 1) ∣ differentIdeal A B := by
  /-
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝²² : CommRing A
    inst✝²¹ : Field K
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : Field L
    inst✝¹⁸ : Algebra A K
    inst✝¹⁷ : Algebra B L
    inst✝¹⁶ : Algebra A B
    inst✝¹⁵ : Algebra K L
    inst✝¹⁴ : Algebra A L
    inst✝¹³ : IsScalarTower A K L
    inst✝¹² : IsScalarTower A B L
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsFractionRing A K
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsFractionRing B L
    inst✝³ : IsDedekindDomain A
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Module.Finite A B
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    he : Ne e 0
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    ⊢ Dvd.dvd (HPow.hPow P (HSub.hSub e 1)) (differentIdeal A B)
  -/
  obtain ⟨a, ha⟩ := (pow_dvd_pow _ (Nat.sub_le e 1)).trans hP
  have hp' := (Ideal.map_eq_bot_iff_of_injective
    (NoZeroSMulDivisors.algebraMap_injective A B)).not.mpr hp
  /-
    case intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝²² : CommRing A
    inst✝²¹ : Field K
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : Field L
    inst✝¹⁸ : Algebra A K
    inst✝¹⁷ : Algebra B L
    inst✝¹⁶ : Algebra A B
    inst✝¹⁵ : Algebra K L
    inst✝¹⁴ : Algebra A L
    inst✝¹³ : IsScalarTower A K L
    inst✝¹² : IsScalarTower A B L
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsFractionRing A K
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsFractionRing B L
    inst✝³ : IsDedekindDomain A
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Module.Finite A B
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    he : Ne e 0
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    a : Ideal B
    ha : Eq (Ideal.map (algebraMap A B) p) (HMul.hMul (HPow.hPow P (HSub.hSub e 1) …
    hp' : Not (Eq (Ideal.map (algebraMap A B) p) Bot.bot)
    ⊢ Dvd.dvd (HPow.hPow P (HSub.hSub e 1)) (differentIdeal A B)
  -/
  have habot : a ≠ ⊥ := fun ha' ↦ hp' (by simpa [ha'] using ha)
  have hPbot : P ≠ ⊥ := by
    rintro rfl; apply hp'
    rwa [← Ideal.zero_eq_bot, zero_pow he, zero_dvd_iff, Ideal.zero_eq_bot] at hP
  have : p.map (algebraMap A B) ∣ a ^ e := by
    obtain ⟨b, hb⟩ := hP
    apply_fun (· ^ e : Ideal B → _) at ha
    apply_fun (· ^ (e - 1) : Ideal B → _) at hb
    simp only [mul_pow, ← pow_mul, mul_comm e] at ha hb
    conv_lhs at ha => rw [← Nat.sub_add_cancel (Nat.one_le_iff_ne_zero.mpr he)]
    rw [pow_add, hb, mul_assoc, mul_right_inj' (pow_ne_zero _ hPbot), pow_one, mul_comm] at ha
    exact ⟨_, ha.symm⟩
  suffices ∀ x ∈ a, intTrace A B x ∈ p by
    have hP : ((P ^ (e - 1) : _)⁻¹ : FractionalIdeal B⁰ L) = a / p.map (algebraMap A B) := by
      apply inv_involutive.injective
      simp only [inv_inv, ha, FractionalIdeal.coeIdeal_mul, inv_div, ne_eq,
          FractionalIdeal.coeIdeal_eq_zero, mul_div_assoc]
      rw [div_self (by simpa), mul_one]
    rw [Ideal.dvd_iff_le, differentialIdeal_le_iff (K := K) (L := L) (pow_ne_zero _ hPbot), hP,
      Submodule.map_le_iff_le_comap]
    intro x hx
    rw [Submodule.restrictScalars_mem, FractionalIdeal.mem_coe,
      FractionalIdeal.mem_div_iff_of_nonzero (by simpa using hp')] at hx
    rw [Submodule.mem_comap, LinearMap.coe_restrictScalars, ← FractionalIdeal.coe_one,
      ← div_self (G₀ := FractionalIdeal A⁰ K) (a := p) (by simpa using hp),
      FractionalIdeal.mem_coe, FractionalIdeal.mem_div_iff_of_nonzero (by simpa using hp)]
    simp only [FractionalIdeal.mem_coeIdeal, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂] at hx
    intro y hy'
    obtain ⟨y, hy, rfl : algebraMap A K _ = _⟩ := (FractionalIdeal.mem_coeIdeal _).mp hy'
    obtain ⟨z, hz, hz'⟩ := hx _ (Ideal.mem_map_of_mem _ hy)
    have : trace K L (algebraMap B L z) ∈ (p : FractionalIdeal A⁰ K) := by
      rw [← algebraMap_intTrace (A := A)]
      exact ⟨intTrace A B z, this z hz, rfl⟩
    rwa [mul_comm, ← smul_eq_mul, ← LinearMap.map_smul, smul_def, mul_comm,
      ← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply A B L, ← hz']
  /-
    case intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝²² : CommRing A
    inst✝²¹ : Field K
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : Field L
    inst✝¹⁸ : Algebra A K
    inst✝¹⁷ : Algebra B L
    inst✝¹⁶ : Algebra A B
    inst✝¹⁵ : Algebra K L
    inst✝¹⁴ : Algebra A L
    inst✝¹³ : IsScalarTower A K L
    inst✝¹² : IsScalarTower A B L
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsFractionRing A K
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsFractionRing B L
    inst✝³ : IsDedekindDomain A
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Module.Finite A B
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    he : Ne e 0
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    a : Ideal B
    ha : Eq (Ideal.map (algebraMap A B) p) (HMul.hMul (HPow.hPow P (HSub.hSub e 1) …
    hp' : Not (Eq (Ideal.map (algebraMap A B) p) Bot.bot)
    habot : Ne a Bot.bot
    hPbot : Ne P Bot.bot
    this : Dvd.dvd (Ideal.map (algebraMap A B) p) (HPow.hPow a e)
    ⊢ ∀ (x : B), Membership.mem a x → Membership.mem p ((Algebra.intTrace A B) x)
  -/
  intros x hx
  rw [← Ideal.Quotient.eq_zero_iff_mem, ← trace_quotient_eq_of_isDedekindDomain,
    ← isNilpotent_iff_eq_zero]
  /-
    case intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝²² : CommRing A
    inst✝²¹ : Field K
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : Field L
    inst✝¹⁸ : Algebra A K
    inst✝¹⁷ : Algebra B L
    inst✝¹⁶ : Algebra A B
    inst✝¹⁵ : Algebra K L
    inst✝¹⁴ : Algebra A L
    inst✝¹³ : IsScalarTower A K L
    inst✝¹² : IsScalarTower A B L
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsFractionRing A K
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsFractionRing B L
    inst✝³ : IsDedekindDomain A
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Module.Finite A B
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    he : Ne e 0
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    a : Ideal B
    ha : Eq (Ideal.map (algebraMap A B) p) (HMul.hMul (HPow.hPow P (HSub.hSub e 1) …
    hp' : Not (Eq (Ideal.map (algebraMap A B) p) Bot.bot)
    habot : Ne a Bot.bot
    hPbot : Ne P Bot.bot
    this : Dvd.dvd (Ideal.map (algebraMap A B) p) (HPow.hPow a e)
    x : B
    hx : Membership.mem a x
    ⊢ IsNilpotent ((Algebra.trace (HasQuotient.Quotient A p) (HasQuotient.Quotient …
  -/
  refine trace_isNilpotent_of_isNilpotent ⟨e, ?_⟩
  /-
    case intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝²² : CommRing A
    inst✝²¹ : Field K
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : Field L
    inst✝¹⁸ : Algebra A K
    inst✝¹⁷ : Algebra B L
    inst✝¹⁶ : Algebra A B
    inst✝¹⁵ : Algebra K L
    inst✝¹⁴ : Algebra A L
    inst✝¹³ : IsScalarTower A K L
    inst✝¹² : IsScalarTower A B L
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsFractionRing A K
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsFractionRing B L
    inst✝³ : IsDedekindDomain A
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Module.Finite A B
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    he : Ne e 0
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    a : Ideal B
    ha : Eq (Ideal.map (algebraMap A B) p) (HMul.hMul (HPow.hPow P (HSub.hSub e 1) …
    hp' : Not (Eq (Ideal.map (algebraMap A B) p) Bot.bot)
    habot : Ne a Bot.bot
    hPbot : Ne P Bot.bot
    this : Dvd.dvd (Ideal.map (algebraMap A B) p) (HPow.hPow a e)
    x : B
    hx : Membership.mem a x
    ⊢ Eq (HPow.hPow ((Ideal.Quotient.mk (Ideal.map (algebraMap A B) p)) x) e) 0
  -/
  rw [← map_pow, Ideal.Quotient.eq_zero_iff_mem]
  /-
    case intro
    A : Type u_1
    K : Type u_2
    L : Type u
    B : Type u_3
    inst✝²² : CommRing A
    inst✝²¹ : Field K
    inst✝²⁰ : CommRing B
    inst✝¹⁹ : Field L
    inst✝¹⁸ : Algebra A K
    inst✝¹⁷ : Algebra B L
    inst✝¹⁶ : Algebra A B
    inst✝¹⁵ : Algebra K L
    inst✝¹⁴ : Algebra A L
    inst✝¹³ : IsScalarTower A K L
    inst✝¹² : IsScalarTower A B L
    inst✝¹¹ : IsDomain A
    inst✝¹⁰ : IsFractionRing A K
    inst✝⁹ : FiniteDimensional K L
    inst✝⁸ : Algebra.IsSeparable K L
    inst✝⁷ : IsIntegralClosure B A L
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsFractionRing B L
    inst✝³ : IsDedekindDomain A
    inst✝² : NoZeroSMulDivisors A B
    inst✝¹ : Module.Finite A B
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    he : Ne e 0
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    a : Ideal B
    ha : Eq (Ideal.map (algebraMap A B) p) (HMul.hMul (HPow.hPow P (HSub.hSub e 1) …
    hp' : Not (Eq (Ideal.map (algebraMap A B) p) Bot.bot)
    habot : Ne a Bot.bot
    hPbot : Ne P Bot.bot
    this : Dvd.dvd (Ideal.map (algebraMap A B) p) (HPow.hPow a e)
    x : B
    hx : Membership.mem a x
    ⊢ Membership.mem (Ideal.map (algebraMap A B) p) (HPow.hPow x e)
  -/
  exact (Ideal.dvd_iff_le.mp this) <| Ideal.pow_mem_pow hx _
  /-
    🎉 no goals
  -/


lemma pow_sub_one_dvd_differentIdeal [IsDedekindDomain A] [NoZeroSMulDivisors A B]
    [Module.Finite A B] [Algebra.IsSeparable (FractionRing A) (FractionRing B)]
    {p : Ideal A} [p.IsMaximal] (P : Ideal B) (e : ℕ) (hp : p ≠ ⊥)
    (hP : P ^ e ∣ p.map (algebraMap A B)) : P ^ (e - 1) ∣ differentIdeal A B := by
  have : IsLocalization (algebraMapSubmonoid B A⁰) (FractionRing B) :=
    IsIntegralClosure.isLocalization _ (FractionRing A) _ _
  have : FiniteDimensional (FractionRing A) (FractionRing B) :=
    Module.Finite_of_isLocalization A B _ _ A⁰
  /-
    A : Type u_1
    B : Type u_3
    inst✝¹⁰ : CommRing A
    inst✝⁹ : CommRing B
    inst✝⁸ : Algebra A B
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsDedekindDomain A
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Module.Finite A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    ⊢ Dvd.dvd (HPow.hPow P (HSub.hSub e 1)) (differentIdeal A B)
  -/
  by_cases he : e = 0
    /-
      case pos
      A : Type u_1
      B : Type u_3
      inst✝¹⁰ : CommRing A
      inst✝⁹ : CommRing B
      inst✝⁸ : Algebra A B
      inst✝⁷ : IsDomain A
      inst✝⁶ : IsIntegrallyClosed A
      inst✝⁵ : IsDedekindDomain B
      inst✝⁴ : IsDedekindDomain A
      inst✝³ : NoZeroSMulDivisors A B
      inst✝² : Module.Finite A B
      inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
      p : Ideal A
      inst✝ : p.IsMaximal
      P : Ideal B
      e : Nat
      hp : Ne p Bot.bot
      hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
      this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
      this : FiniteDimensional (FractionRing A) (FractionRing B)
      he : Eq e 0
      ⊢ Dvd.dvd (HPow.hPow P (HSub.hSub e 1)) (differentIdeal A B)
    -/
  · rw [he, pow_zero]; exact one_dvd _
                       /-
                         🎉 no goals
                       -/
  /-
    case neg
    A : Type u_1
    B : Type u_3
    inst✝¹⁰ : CommRing A
    inst✝⁹ : CommRing B
    inst✝⁸ : Algebra A B
    inst✝⁷ : IsDomain A
    inst✝⁶ : IsIntegrallyClosed A
    inst✝⁵ : IsDedekindDomain B
    inst✝⁴ : IsDedekindDomain A
    inst✝³ : NoZeroSMulDivisors A B
    inst✝² : Module.Finite A B
    inst✝¹ : Algebra.IsSeparable (FractionRing A) (FractionRing B)
    p : Ideal A
    inst✝ : p.IsMaximal
    P : Ideal B
    e : Nat
    hp : Ne p Bot.bot
    hP : Dvd.dvd (HPow.hPow P e) (Ideal.map (algebraMap A B) p)
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid B (nonZeroDivisors A)) (Fr …
    this : FiniteDimensional (FractionRing A) (FractionRing B)
    he : Not (Eq e 0)
    ⊢ Dvd.dvd (HPow.hPow P (HSub.hSub e 1)) (differentIdeal A B)
  -/
  exact pow_sub_one_dvd_differentIdeal_aux A (FractionRing A) (FractionRing B) _ he hp hP
  /-
    🎉 no goals
  -/


