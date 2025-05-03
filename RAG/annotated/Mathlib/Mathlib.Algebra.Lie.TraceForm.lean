local notation "φ" => LieModule.toEnd R L M


/-- A finite, free representation of a Lie algebra `L` induces a bilinear form on `L` called
the trace Form. See also `killingForm`. -/
noncomputable def traceForm : LinearMap.BilinForm R L :=
  ((LinearMap.mul _ _).compl₁₂ (φ).toLinearMap (φ).toLinearMap).compr₂ (trace R M)


lemma traceForm_apply_apply (x y : L) :
    traceForm R L M x y = trace R _ (φ x ∘ₗ φ y) :=
  rfl


lemma traceForm_comm (x y : L) : traceForm R L M x y = traceForm R L M y x :=
  LinearMap.trace_mul_comm R (φ x) (φ y)


lemma traceForm_isSymm : LinearMap.IsSymm (traceForm R L M) := LieModule.traceForm_comm R L M


@[simp] lemma traceForm_flip : LinearMap.flip (traceForm R L M) = traceForm R L M :=
  Eq.symm <| LinearMap.ext₂ <| traceForm_comm R L M


/-- The trace form of a Lie module is compatible with the action of the Lie algebra.

See also `LieModule.traceForm_apply_lie_apply'`. -/
lemma traceForm_apply_lie_apply (x y z : L) :
    traceForm R L M ⁅x, y⁆ z = traceForm R L M x ⁅y, z⁆ := by
  calc traceForm R L M ⁅x, y⁆ z
      = trace R _ (φ ⁅x, y⁆ ∘ₗ φ z) := by simp only [traceForm_apply_apply]
    _ = trace R _ ((φ x * φ y - φ y * φ x) * φ z) := ?_
    _ = trace R _ (φ x * (φ y * φ z)) - trace R _ (φ y * (φ x * φ z)) := ?_
    _ = trace R _ (φ x * (φ y * φ z)) - trace R _ (φ x * (φ z * φ y)) := ?_
    _ = traceForm R L M x ⁅y, z⁆ := ?_
    /-
      case calc_1
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y z : L
      ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp ((LieModule.toEnd R L M) (Bracket. …
    -/
  · simp only [LieHom.map_lie, Ring.lie_def, ← LinearMap.mul_eq_comp]
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y z : L
      ⊢ Eq ((LinearMap.trace R M) (HMul.hMul (HSub.hSub (HMul.hMul ((LieModule.toEnd …
    -/
  · simp only [sub_mul, mul_sub, map_sub, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y z : L
      ⊢ Eq (HSub.hSub ((LinearMap.trace R M) (HMul.hMul ((LieModule.toEnd R L M) x)  …
    -/
  · simp only [LinearMap.trace_mul_cycle' R (φ x) (φ z) (φ y)]
    /-
      🎉 no goals
    -/
  · simp only [traceForm_apply_apply, LieHom.map_lie, Ring.lie_def, mul_sub, map_sub,
      ← LinearMap.mul_eq_comp]


/-- Given a representation `M` of a Lie algebra `L`, the action of any `x : L` is skew-adjoint wrt
the trace form. -/
lemma traceForm_apply_lie_apply' (x y z : L) :
    traceForm R L M ⁅x, y⁆ z = - traceForm R L M y ⁅x, z⁆ :=
  calc traceForm R L M ⁅x, y⁆ z
                                         /-
                                           R : Type u_1
                                           L : Type u_3
                                           M : Type u_4
                                           inst✝⁶ : CommRing R
                                           inst✝⁵ : LieRing L
                                           inst✝⁴ : LieAlgebra R L
                                           inst✝³ : AddCommGroup M
                                           inst✝² : Module R M
                                           inst✝¹ : LieRingModule L M
                                           inst✝ : LieModule R L M
                                           x y z : L
                                           ⊢ Eq (((LieModule.traceForm R L M) (Bracket.bracket x y)) z) (Neg.neg (((LieMo …
                                         -/
      = - traceForm R L M ⁅y, x⁆ z := by rw [← lie_skew x y, map_neg, LinearMap.neg_apply]
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           R : Type u_1
                                           L : Type u_3
                                           M : Type u_4
                                           inst✝⁶ : CommRing R
                                           inst✝⁵ : LieRing L
                                           inst✝⁴ : LieAlgebra R L
                                           inst✝³ : AddCommGroup M
                                           inst✝² : Module R M
                                           inst✝¹ : LieRingModule L M
                                           inst✝ : LieModule R L M
                                           x y z : L
                                           ⊢ Eq (Neg.neg (((LieModule.traceForm R L M) (Bracket.bracket y x)) z)) (Neg.ne …
                                         -/
    _ = - traceForm R L M y ⁅x, z⁆ := by rw [traceForm_apply_lie_apply]
                                         /-
                                           🎉 no goals
                                         -/


lemma traceForm_lieInvariant : (traceForm R L M).lieInvariant L := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ LinearMap.BilinForm.lieInvariant L (LieModule.traceForm R L M)
  -/
  intro x y z
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x y z : L
    ⊢ Eq (((LieModule.traceForm R L M) (Bracket.bracket x y)) z) (Neg.neg (((LieMo …
  -/
  rw [← lie_skew, map_neg, LinearMap.neg_apply, LieModule.traceForm_apply_lie_apply R L M]
  /-
    🎉 no goals
  -/


/-- This lemma justifies the terminology "invariant" for trace forms. -/
@[simp] lemma lie_traceForm_eq_zero (x : L) : ⁅x, traceForm R L M⁆ = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x : L
    ⊢ Eq (Bracket.bracket x (LieModule.traceForm R L M)) 0
  -/
  ext y z
  rw [LieHom.lie_apply, LinearMap.sub_apply, Module.Dual.lie_apply, LinearMap.zero_apply,
    LinearMap.zero_apply, traceForm_apply_lie_apply', sub_self]


@[simp] lemma traceForm_eq_zero_of_isNilpotent [IsReduced R] [IsNilpotent R L M] :
    traceForm R L M = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : IsReduced R
    inst✝ : LieModule.IsNilpotent R L M
    ⊢ Eq (LieModule.traceForm R L M) 0
  -/
  ext x y
  /-
    case H
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : IsReduced R
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    ⊢ Eq (((LieModule.traceForm R L M) x) y) ((0 x) y)
  -/
  simp only [traceForm_apply_apply, LinearMap.zero_apply, ← isNilpotent_iff_eq_zero]
  /-
    case H
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : IsReduced R
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    ⊢ _root_.IsNilpotent ((LinearMap.trace R M) (LinearMap.comp ((LieModule.toEnd  …
  -/
  apply LinearMap.isNilpotent_trace_of_isNilpotent
  /-
    case H.hf
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    inst✝¹ : IsReduced R
    inst✝ : LieModule.IsNilpotent R L M
    x y : L
    ⊢ _root_.IsNilpotent (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.t …
  -/
  exact isNilpotent_toEnd_of_isNilpotent₂ R L M x y
  /-
    🎉 no goals
  -/


@[simp]
lemma traceForm_genWeightSpace_eq [Module.Free R M]
    [IsDomain R] [IsPrincipalIdealRing R]
    [LieAlgebra.IsNilpotent R L] [IsNoetherian R M] [LinearWeights R L M] (χ : L → R) (x y : L) :
    traceForm R L (genWeightSpace M χ) x y = finrank R (genWeightSpace M χ) • (χ x * χ y) := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieAlgebra R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : LieModule R L M
    inst✝⁵ : Module.Free R M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : LieModule.LinearWeights R L M
    χ : L → R
    x y : L
    ⊢ Eq (((LieModule.traceForm R L (Subtype fun x => Membership.mem (LieModule.ge …
  -/
  set d := finrank R (genWeightSpace M χ)
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieAlgebra R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : LieModule R L M
    inst✝⁵ : Module.Free R M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : LieModule.LinearWeights R L M
    χ : L → R
    x y : L
    d : Nat := Module.finrank R (Subtype fun x => Membership.mem (LieModule.genWei …
    ⊢ Eq (((LieModule.traceForm R L (Subtype fun x => Membership.mem (LieModule.ge …
  -/
  have h₁ : χ y • d • χ x - χ y • χ x • (d : R) = 0 := by simp [mul_comm (χ x)]
  have h₂ : χ x • d • χ y = d • (χ x * χ y) := by
    simpa [nsmul_eq_mul, smul_eq_mul] using mul_left_comm (χ x) d (χ y)
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieAlgebra R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : LieModule R L M
    inst✝⁵ : Module.Free R M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : LieModule.LinearWeights R L M
    χ : L → R
    x y : L
    d : Nat := Module.finrank R (Subtype fun x => Membership.mem (LieModule.genWei …
    h₁ : Eq (HSub.hSub (HSMul.hSMul (χ y) (HSMul.hSMul d (χ x))) (HSMul.hSMul (χ y …
    h₂ : Eq (HSMul.hSMul (χ x) (HSMul.hSMul d (χ y))) (HSMul.hSMul d (HMul.hMul (χ …
    ⊢ Eq (((LieModule.traceForm R L (Subtype fun x => Membership.mem (LieModule.ge …
  -/
  have := traceForm_eq_zero_of_isNilpotent R L (shiftedGenWeightSpace R L M χ)
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieAlgebra R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : LieModule R L M
    inst✝⁵ : Module.Free R M
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : LieAlgebra.IsNilpotent R L
    inst✝¹ : IsNoetherian R M
    inst✝ : LieModule.LinearWeights R L M
    χ : L → R
    x y : L
    d : Nat := Module.finrank R (Subtype fun x => Membership.mem (LieModule.genWei …
    h₁ : Eq (HSub.hSub (HSMul.hSMul (χ y) (HSMul.hSMul d (χ x))) (HSMul.hSMul (χ y …
    h₂ : Eq (HSMul.hSMul (χ x) (HSMul.hSMul d (χ y))) (HSMul.hSMul d (HMul.hMul (χ …
    this : Eq (LieModule.traceForm R L (Subtype fun x => Membership.mem (LieModule …
    ⊢ Eq (((LieModule.traceForm R L (Subtype fun x => Membership.mem (LieModule.ge …
  -/
  replace this := LinearMap.congr_fun (LinearMap.congr_fun this x) y
  rwa [LinearMap.zero_apply, LinearMap.zero_apply, traceForm_apply_apply,
    shiftedGenWeightSpace.toEnd_eq, shiftedGenWeightSpace.toEnd_eq,
    ← LinearEquiv.conj_comp, LinearMap.trace_conj', LinearMap.comp_sub, LinearMap.sub_comp,
    LinearMap.sub_comp, map_sub, map_sub, map_sub, LinearMap.comp_smul, LinearMap.smul_comp,
    LinearMap.comp_id, LinearMap.id_comp, LinearMap.map_smul, LinearMap.map_smul,
    trace_toEnd_genWeightSpace, trace_toEnd_genWeightSpace,
    LinearMap.comp_smul, LinearMap.smul_comp, LinearMap.id_comp, map_smul, map_smul,
    LinearMap.trace_id, ← traceForm_apply_apply, h₁, h₂, sub_zero, sub_eq_zero] at this


/-- The upper and lower central series of `L` are orthogonal wrt the trace form of any Lie module
`M`. -/
lemma traceForm_eq_zero_if_mem_lcs_of_mem_ucs {x y : L} (k : ℕ)
    (hx : x ∈ (⊤ : LieIdeal R L).lcs L k) (hy : y ∈ (⊥ : LieIdeal R L).ucs k) :
    traceForm R L M x y = 0 := by
  induction k generalizing x y with
  | zero =>
    replace hy : y = 0 := by simpa using hy
    simp [hy]
  | succ k ih =>
    rw [LieSubmodule.ucs_succ, LieSubmodule.mem_normalizer] at hy
    simp_rw [LieIdeal.lcs_succ, ← LieSubmodule.mem_toSubmodule,
      LieSubmodule.lieIdeal_oper_eq_linear_span', LieSubmodule.mem_top, true_and] at hx
    refine Submodule.span_induction ?_ ?_ (fun z w _ _ hz hw ↦ ?_) (fun t z _ hz ↦ ?_) hx
    · rintro - ⟨z, w, hw, rfl⟩
      rw [← lie_skew, map_neg, LinearMap.neg_apply, neg_eq_zero, traceForm_apply_lie_apply]
      exact ih hw (hy _)
    · simp
    · simp [hz, hw]
    · simp [hz]


lemma traceForm_apply_eq_zero_of_mem_lcs_of_mem_center {x y : L}
    (hx : x ∈ lowerCentralSeries R L L 1) (hy : y ∈ LieAlgebra.center R L) :
    traceForm R L M x y = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x y : L
    hx : Membership.mem (LieModule.lowerCentralSeries R L L 1) x
    hy : Membership.mem (LieAlgebra.center R L) y
    ⊢ Eq (((LieModule.traceForm R L M) x) y) 0
  -/
  apply traceForm_eq_zero_if_mem_lcs_of_mem_ucs R L M 1
    /-
      case hx
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y : L
      hx : Membership.mem (LieModule.lowerCentralSeries R L L 1) x
      hy : Membership.mem (LieAlgebra.center R L) y
      ⊢ Membership.mem (Top.top.lcs L 1) x
    -/
  · simpa using hx
    /-
      🎉 no goals
    -/
    /-
      case hy
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      x y : L
      hx : Membership.mem (LieModule.lowerCentralSeries R L L 1) x
      hy : Membership.mem (LieAlgebra.center R L) y
      ⊢ Membership.mem (LieSubmodule.ucs 1 Bot.bot) y
    -/
  · simpa using hy
    /-
      🎉 no goals
    -/

-- This is barely worth having: it usually follows from `LieModule.traceForm_eq_zero_of_isNilpotent`

@[simp] lemma traceForm_eq_zero_of_isTrivial [IsTrivial L M] :
    traceForm R L M = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsTrivial L M
    ⊢ Eq (LieModule.traceForm R L M) 0
  -/
  ext x y
  /-
    case H
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsTrivial L M
    x y : L
    ⊢ Eq (((LieModule.traceForm R L M) x) y) ((0 x) y)
  -/
  suffices φ x ∘ₗ φ y = 0 by simp [traceForm_apply_apply, this]
  /-
    case H
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsTrivial L M
    x y : L
    ⊢ Eq (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.toEnd R L M) y)) 0
  -/
  ext m
  /-
    case H.h
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieModule.IsTrivial L M
    x y : L
    m : M
    ⊢ Eq ((LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.toEnd R L M) y)) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a bilinear form `B` on a representation `M` of a nilpotent Lie algebra `L`, if `B` is
invariant (in the sense that the action of `L` is skew-adjoint wrt `B`) then components of the
Fitting decomposition of `M` are orthogonal wrt `B`. -/
lemma eq_zero_of_mem_genWeightSpace_mem_posFitting [LieAlgebra.IsNilpotent R L]
    {B : LinearMap.BilinForm R M} (hB : ∀ (x : L) (m n : M), B ⁅x, m⁆ n = - B m ⁅x, n⁆)
    {m₀ m₁ : M} (hm₀ : m₀ ∈ genWeightSpace M (0 : L → R)) (hm₁ : m₁ ∈ posFittingComp R L M) :
    B m₀ m₁ = 0 := by
  replace hB : ∀ x (k : ℕ) m n, B m ((φ x ^ k) n) = (- 1 : R) ^ k • B ((φ x ^ k) m) n := by
    intro x k
    induction k with
    | zero => simp
    | succ k ih =>
    intro m n
    replace hB : ∀ m, B m (φ x n) = (- 1 : R) • B (φ x m) n := by simp [hB]
    have : (-1 : R) ^ k • (-1 : R) = (-1 : R) ^ (k + 1) := by rw [pow_succ (-1 : R), smul_eq_mul]
    conv_lhs => rw [pow_succ, LinearMap.mul_eq_comp, LinearMap.comp_apply, ih, hB,
      ← (φ x).comp_apply, ← LinearMap.mul_eq_comp, ← pow_succ', ← smul_assoc, this]
  suffices ∀ (x : L) m, m ∈ posFittingCompOf R M x → B m₀ m = 0 by
    apply LieSubmodule.iSup_induction _ hm₁ this (map_zero _)
    aesop
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    B : LinearMap.BilinForm R M
    m₀ m₁ : M
    hm₀ : Membership.mem (LieModule.genWeightSpace M 0) m₀
    hm₁ : Membership.mem (LieModule.posFittingComp R L M) m₁
    hB : ∀ (x : L) (k : Nat) (m n : M), Eq ((B m) ((HPow.hPow ((LieModule.toEnd R  …
    ⊢ ∀ (x : L) (m : M), Membership.mem (LieModule.posFittingCompOf R M x) m → Eq  …
  -/
  clear hm₁ m₁; intro x m₁ hm₁
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    B : LinearMap.BilinForm R M
    m₀ : M
    hm₀ : Membership.mem (LieModule.genWeightSpace M 0) m₀
    hB : ∀ (x : L) (k : Nat) (m n : M), Eq ((B m) ((HPow.hPow ((LieModule.toEnd R  …
    x : L
    m₁ : M
    hm₁ : Membership.mem (LieModule.posFittingCompOf R M x) m₁
    ⊢ Eq ((B m₀) m₁) 0
  -/
  simp only [mem_genWeightSpace, Pi.zero_apply, zero_smul, sub_zero] at hm₀
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    B : LinearMap.BilinForm R M
    m₀ : M
    hB : ∀ (x : L) (k : Nat) (m n : M), Eq ((B m) ((HPow.hPow ((LieModule.toEnd R  …
    x : L
    m₁ : M
    hm₁ : Membership.mem (LieModule.posFittingCompOf R M x) m₁
    hm₀ : ∀ (x : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) …
    ⊢ Eq ((B m₀) m₁) 0
  -/
  obtain ⟨k, hk⟩ := hm₀ x
  /-
    case intro
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    B : LinearMap.BilinForm R M
    m₀ : M
    hB : ∀ (x : L) (k : Nat) (m n : M), Eq ((B m) ((HPow.hPow ((LieModule.toEnd R  …
    x : L
    m₁ : M
    hm₁ : Membership.mem (LieModule.posFittingCompOf R M x) m₁
    hm₀ : ∀ (x : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) …
    k : Nat
    hk : Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m₀) 0
    ⊢ Eq ((B m₀) m₁) 0
  -/
  obtain ⟨m, rfl⟩ := (mem_posFittingCompOf R x m₁).mp hm₁ k
  /-
    case intro.intro
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    B : LinearMap.BilinForm R M
    m₀ : M
    hB : ∀ (x : L) (k : Nat) (m n : M), Eq ((B m) ((HPow.hPow ((LieModule.toEnd R  …
    x : L
    hm₀ : ∀ (x : L), Exists fun k => Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) …
    k : Nat
    hk : Eq ((HPow.hPow ((LieModule.toEnd R L M) x) k) m₀) 0
    m : M
    hm₁ : Membership.mem (LieModule.posFittingCompOf R M x) ((HPow.hPow ((LieModul …
    ⊢ Eq ((B m₀) ((HPow.hPow ((LieModule.toEnd R L M) x) k) m)) 0
  -/
  simp [hB, hk]
  /-
    🎉 no goals
  -/


lemma trace_toEnd_eq_zero_of_mem_lcs
    {k : ℕ} {x : L} (hk : 1 ≤ k) (hx : x ∈ lowerCentralSeries R L L k) :
    trace R _ (toEnd R L M x) = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    k : Nat
    x : L
    hk : LE.le 1 k
    hx : Membership.mem (LieModule.lowerCentralSeries R L L k) x
    ⊢ Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0
  -/
  replace hx : x ∈ lowerCentralSeries R L L 1 := antitone_lowerCentralSeries _ _ _ hk hx
  replace hx : x ∈ Submodule.span R {m | ∃ u v : L, ⁅u, v⁆ = m} := by
    rw [lowerCentralSeries_succ, ← LieSubmodule.mem_toSubmodule,
      LieSubmodule.lieIdeal_oper_eq_linear_span'] at hx
    simpa using hx
  refine Submodule.span_induction (p := fun x _ ↦ trace R _ (toEnd R L M x) = 0)
    ?_ ?_ (fun u v _ _ hu hv ↦ ?_) (fun t u _ hu ↦ ?_) hx
    /-
      case refine_1
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      x : L
      hk : LE.le 1 k
      hx : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      ⊢ ∀ (x : L) (h : Membership.mem (setOf fun m => Exists fun u => Exists fun v = …
    -/
  · intro y ⟨u, v, huv⟩
    /-
      case refine_1
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      x : L
      hk : LE.le 1 k
      hx : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      y u v : L
      huv : Eq (Bracket.bracket u v) y
      ⊢ Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) y)) 0
    -/
    simp [← huv]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      x : L
      hk : LE.le 1 k
      hx : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      ⊢ (fun x x_1 => Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0) 0 ⋯
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      x : L
      hk : LE.le 1 k
      hx : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      u v : L
      x✝¹ : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists  …
      x✝ : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      hu : (fun x x_1 => Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0) u …
      hv : (fun x x_1 => Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0) v …
      ⊢ (fun x x_1 => Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0) (HAd …
    -/
  · simp [hu, hv]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      k : Nat
      x : L
      hk : LE.le 1 k
      hx : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      t : R
      u : L
      x✝ : Membership.mem (Submodule.span R (setOf fun m => Exists fun u => Exists f …
      hu : (fun x x_1 => Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0) u …
      ⊢ (fun x x_1 => Eq ((LinearMap.trace R M) ((LieModule.toEnd R L M) x)) 0) (HSM …
    -/
  · simp [hu]
    /-
      🎉 no goals
    -/


@[simp]
lemma traceForm_lieSubalgebra_mk_left (L' : LieSubalgebra R L) {x : L} (hx : x ∈ L') (y : L') :
    traceForm R L' M ⟨x, hx⟩ y = traceForm R L M x y :=
  rfl


@[simp]
lemma traceForm_lieSubalgebra_mk_right (L' : LieSubalgebra R L) {x : L'} {y : L} (hy : y ∈ L') :
    traceForm R L' M x ⟨y, hy⟩ = traceForm R L M x y :=
  rfl


lemma traceForm_eq_sum_genWeightSpaceOf
    [NoZeroSMulDivisors R M] [IsNoetherian R M] [IsTriangularizable R L M] (z : L) :
    traceForm R L M =
    ∑ χ ∈ (finite_genWeightSpaceOf_ne_bot R L M z).toFinset,
      traceForm R L (genWeightSpaceOf M χ z) := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieAlgebra R L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : LieRingModule L M
    inst✝⁶ : LieModule R L M
    inst✝⁵ : LieAlgebra.IsNilpotent R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : IsNoetherian R M
    inst✝ : LieModule.IsTriangularizable R L M
    z : L
    ⊢ Eq (LieModule.traceForm R L M) (⋯.toFinset.sum fun χ => LieModule.traceForm  …
  -/
  ext x y
  have hxy : ∀ χ : R, MapsTo ((toEnd R L M x).comp (toEnd R L M y))
      (genWeightSpaceOf M χ z) (genWeightSpaceOf M χ z) :=
    fun χ m hm ↦ LieSubmodule.lie_mem _ <| LieSubmodule.lie_mem _ hm
  have hfin : {χ : R | (genWeightSpaceOf M χ z : Submodule R M) ≠ ⊥}.Finite := by
    convert finite_genWeightSpaceOf_ne_bot R L M z
    exact LieSubmodule.toSubmodule_eq_bot (genWeightSpaceOf M _ _)
  classical
  have h := LieSubmodule.iSupIndep_iff_toSubmodule.mp <| iSupIndep_genWeightSpaceOf R L M z
  have hds := DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top h <| by
    simp [← LieSubmodule.iSup_toSubmodule]
  simp only [LinearMap.coeFn_sum, Finset.sum_apply, traceForm_apply_apply,
    LinearMap.trace_eq_sum_trace_restrict' hds hfin hxy]
  exact Finset.sum_congr (by simp) (fun χ _ ↦ rfl)

-- In characteristic zero (or even just `LinearWeights R L M`) a stronger result holds (no
-- `⊓ LieAlgebra.center R L`) TODO prove this using `LieModule.traceForm_eq_sum_finrank_nsmul_mul`.

lemma lowerCentralSeries_one_inf_center_le_ker_traceForm [Module.Free R M] [Module.Finite R M] :
    lowerCentralSeries R L L 1 ⊓ LieAlgebra.center R L ≤ LinearMap.ker (traceForm R L M) := by
  /- Sketch of proof (due to Zassenhaus):

  Let `z ∈ lowerCentralSeries R L L 1 ⊓ LieAlgebra.center R L` and `x : L`. We must show that
  `trace (φ x ∘ φ z) = 0` where `φ z : End R M` indicates the action of `z` on `M` (and likewise
  for `φ x`).

  Because `z` belongs to the indicated intersection, it has two key properties:
  (a) the trace of the action of `z` vanishes on any Lie module of `L`
      (see `LieModule.trace_toEnd_eq_zero_of_mem_lcs`),
  (b) `z` commutes with all elements of `L`.

  If `φ x` were triangularizable, we could write `M` as a direct sum of generalized eigenspaces of
  `φ x`. Because `L` is nilpotent these are all Lie submodules, thus Lie modules in their own right,
  and thus by (a) above we learn that `trace (φ z) = 0` restricted to each generalized eigenspace.
  Because `z` commutes with `x`, this forces `trace (φ x ∘ φ z) = 0` on each generalized eigenspace,
  and so by summing the traces on each generalized eigenspace we learn the total trace is zero, as
  required (see `LinearMap.trace_comp_eq_zero_of_commute_of_trace_restrict_eq_zero`).

  To cater for the fact that `φ x` may not be triangularizable, we first extend the scalars from `R`
  to `AlgebraicClosure (FractionRing R)` and argue using the action of `A ⊗ L` on `A ⊗ M`. -/
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    ⊢ LE.le (LieIdeal.toLieSubalgebra R L (Min.min (LieModule.lowerCentralSeries R …
  -/
  rintro z ⟨hz : z ∈ lowerCentralSeries R L L 1, hzc : z ∈ LieAlgebra.center R L⟩
  /-
    case intro
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    z : L
    hz : Membership.mem (LieModule.lowerCentralSeries R L L 1) z
    hzc : Membership.mem (LieAlgebra.center R L) z
    ⊢ Membership.mem (LinearMap.ker (LieModule.traceForm R L M)) z
  -/
  ext x
  /-
    case intro.h
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    z : L
    hz : Membership.mem (LieModule.lowerCentralSeries R L L 1) z
    hzc : Membership.mem (LieAlgebra.center R L) z
    x : L
    ⊢ Eq (((LieModule.traceForm R L M) z) x) (0 x)
  -/
  rw [traceForm_apply_apply, LinearMap.zero_apply]
  /-
    case intro.h
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    z : L
    hz : Membership.mem (LieModule.lowerCentralSeries R L L 1) z
    hzc : Membership.mem (LieAlgebra.center R L) z
    x : L
    ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp ((LieModule.toEnd R L M) z) ((LieM …
  -/
  let A := AlgebraicClosure (FractionRing R)
  suffices algebraMap R A (trace R _ ((φ z).comp (φ x))) = 0 by
    have _i : NoZeroSMulDivisors R A := NoZeroSMulDivisors.trans R (FractionRing R) A
    rw [← map_zero (algebraMap R A)] at this
    exact NoZeroSMulDivisors.algebraMap_injective R A this
  rw [← LinearMap.trace_baseChange, LinearMap.baseChange_comp, ← toEnd_baseChange,
    ← toEnd_baseChange]
  replace hz : 1 ⊗ₜ z ∈ lowerCentralSeries A (A ⊗[R] L) (A ⊗[R] L) 1 := by
    simp only [lowerCentralSeries_succ, lowerCentralSeries_zero] at hz ⊢
    rw [← LieSubmodule.baseChange_top, ← LieSubmodule.lie_baseChange]
    exact Submodule.tmul_mem_baseChange_of_mem 1 hz
  replace hzc : 1 ⊗ₜ[R] z ∈ LieAlgebra.center A (A ⊗[R] L) := by
    simp only [mem_maxTrivSubmodule] at hzc ⊢
    intro y
    exact y.induction_on rfl (fun a u ↦ by simp [hzc u])
      (fun u v hu hv ↦ by simp [A, hu, hv])
  /-
    case intro.h
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : LieAlgebra.IsNilpotent R L
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    z x : L
    A : Type u_1 := AlgebraicClosure (FractionRing R)
    hz : Membership.mem (LieModule.lowerCentralSeries A (TensorProduct R A L) (Ten …
    hzc : Membership.mem (LieAlgebra.center A (TensorProduct R A L)) (TensorProduc …
    ⊢ Eq ((LinearMap.trace A (TensorProduct R A M)) (LinearMap.comp ((LieModule.to …
  -/
  apply LinearMap.trace_comp_eq_zero_of_commute_of_trace_restrict_eq_zero
    /-
      case intro.h.hf
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      inst✝³ : IsDomain R
      inst✝² : IsPrincipalIdealRing R
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      z x : L
      A : Type u_1 := AlgebraicClosure (FractionRing R)
      hz : Membership.mem (LieModule.lowerCentralSeries A (TensorProduct R A L) (Ten …
      hzc : Membership.mem (LieAlgebra.center A (TensorProduct R A L)) (TensorProduc …
      ⊢ Eq (iSup fun μ => ((LieModule.toEnd A (TensorProduct R A L) (TensorProduct R …
    -/
  · exact IsTriangularizable.maxGenEigenspace_eq_top (1 ⊗ₜ[R] x)
    /-
      🎉 no goals
    -/
  · exact fun μ ↦ trace_toEnd_eq_zero_of_mem_lcs A (A ⊗[R] L)
      (genWeightSpaceOf (A ⊗[R] M) μ ((1:A) ⊗ₜ[R] x)) (le_refl 1) hz
    /-
      case intro.h.h_comm
      R : Type u_1
      L : Type u_3
      M : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra R L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : LieModule R L M
      inst✝⁴ : LieAlgebra.IsNilpotent R L
      inst✝³ : IsDomain R
      inst✝² : IsPrincipalIdealRing R
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      z x : L
      A : Type u_1 := AlgebraicClosure (FractionRing R)
      hz : Membership.mem (LieModule.lowerCentralSeries A (TensorProduct R A L) (Ten …
      hzc : Membership.mem (LieAlgebra.center A (TensorProduct R A L)) (TensorProduc …
      ⊢ Commute ((LieModule.toEnd A (TensorProduct R A L) (TensorProduct R A M)) (Te …
    -/
  · exact commute_toEnd_of_mem_center_right (A ⊗[R] M) hzc (1 ⊗ₜ x)
    /-
      🎉 no goals
    -/


/-- A nilpotent Lie algebra with a representation whose trace form is non-singular is Abelian. -/
lemma isLieAbelian_of_ker_traceForm_eq_bot [Module.Free R M] [Module.Finite R M]
    (h : LinearMap.ker (traceForm R L M) = ⊥) : IsLieAbelian L := by
  simpa only [← disjoint_lowerCentralSeries_maxTrivSubmodule_iff R L L, disjoint_iff_inf_le,
    LieIdeal.toLieSubalgebra_toSubmodule, LieSubmodule.toSubmodule_eq_bot, h]
    using lowerCentralSeries_one_inf_center_le_ker_traceForm R L M


lemma trace_eq_trace_restrict_of_le_idealizer
    (hy' : ∀ m ∈ N, (φ x ∘ₗ φ y) m ∈ N := fun m _ ↦ N.lie_mem (N.mem_idealizer.mp (h hy) m)) :
    trace R M (φ x ∘ₗ φ y) = trace R N ((φ x ∘ₗ φ y).restrict hy') := by
  suffices ∀ m, ⁅x, ⁅y, m⁆⁆ ∈ N by
    have : (trace R { x // x ∈ N }) ((φ x ∘ₗ φ y).restrict _) = (trace R M) (φ x ∘ₗ φ y) :=
      (φ x ∘ₗ φ y).trace_restrict_eq_of_forall_mem _ this
    simp [this]
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x y : L
    hy : Membership.mem I y
    hy' : optParam (∀ (m : M), Membership.mem N m → Membership.mem N ((LinearMap.c …
    ⊢ ∀ (m : M), Membership.mem N (Bracket.bracket x (Bracket.bracket y m))
  -/
  exact fun m ↦ N.lie_mem (h hy m)
  /-
    🎉 no goals
  -/


include h in
lemma traceForm_eq_of_le_idealizer :
    traceForm R I N = (traceForm R L M).restrict I := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    ⊢ Eq (LieModule.traceForm R (Subtype fun x => Membership.mem I x) (Subtype fun …
  -/
  ext ⟨x, hx⟩ ⟨y, hy⟩
  /-
    case H.mk.mk
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x : L
    hx : Membership.mem I x
    y : L
    hy : Membership.mem I y
    ⊢ Eq (((LieModule.traceForm R (Subtype fun x => Membership.mem I x) (Subtype f …
  -/
  change _ = trace R M (φ x ∘ₗ φ y)
  /-
    case H.mk.mk
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x : L
    hx : Membership.mem I x
    y : L
    hy : Membership.mem I y
    ⊢ Eq (((LieModule.traceForm R (Subtype fun x => Membership.mem I x) (Subtype f …
  -/
  rw [N.trace_eq_trace_restrict_of_le_idealizer I h x hy]
  /-
    case H.mk.mk
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x : L
    hx : Membership.mem I x
    y : L
    hy : Membership.mem I y
    ⊢ Eq (((LieModule.traceForm R (Subtype fun x => Membership.mem I x) (Subtype f …
  -/
  rfl
  /-
    🎉 no goals
  -/


include h hy in
/-- Note that this result is slightly stronger than it might look at first glance: we only assume
that `N` is trivial over `I` rather than all of `L`. This means that it applies in the important
case of an Abelian ideal (which has `M = L` and `N = I`). -/
lemma traceForm_eq_zero_of_isTrivial [LieModule.IsTrivial I N] :
    trace R M (φ x ∘ₗ φ y) = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x y : L
    hy : Membership.mem I y
    inst✝ : LieModule.IsTrivial (Subtype fun x => Membership.mem I x) (Subtype fun …
    ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp ((LieModule.toEnd R L M) x) ((LieM …
  -/
  let hy' : ∀ m ∈ N, (φ x ∘ₗ φ y) m ∈ N := fun m _ ↦ N.lie_mem (N.mem_idealizer.mp (h hy) m)
  suffices (φ x ∘ₗ φ y).restrict hy' = 0 by
    simp [this, N.trace_eq_trace_restrict_of_le_idealizer I h x hy]
  /-
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x y : L
    hy : Membership.mem I y
    inst✝ : LieModule.IsTrivial (Subtype fun x => Membership.mem I x) (Subtype fun …
    hy' : ∀ (m : M), Membership.mem N m → Membership.mem N ((LinearMap.comp ((LieM …
    ⊢ Eq ((LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.toEnd R L M) y)) …
  -/
  ext (n : N)
  /-
    case h.a
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x y : L
    hy : Membership.mem I y
    inst✝ : LieModule.IsTrivial (Subtype fun x => Membership.mem I x) (Subtype fun …
    hy' : ∀ (m : M), Membership.mem N m → Membership.mem N ((LinearMap.comp ((LieM …
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq ↑(((LinearMap.comp ((LieModule.toEnd R L M) x) ((LieModule.toEnd R L M) y …
  -/
  suffices ⁅y, (n : M)⁆ = 0 by simp [this]
  /-
    case h.a
    R : Type u_1
    L : Type u_3
    M : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra R L
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : LieModule R L M
    inst✝⁴ : Module.Free R M
    inst✝³ : Module.Finite R M
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    N : LieSubmodule R L M
    I : LieIdeal R L
    h : LE.le I N.idealizer
    x y : L
    hy : Membership.mem I y
    inst✝ : LieModule.IsTrivial (Subtype fun x => Membership.mem I x) (Subtype fun …
    hy' : ∀ (m : M), Membership.mem N m → Membership.mem N ((LinearMap.comp ((LieM …
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq (Bracket.bracket y ↑n) 0
  -/
  exact Submodule.coe_eq_zero.mpr (LieModule.IsTrivial.trivial (⟨y, hy⟩ : I) n)
  /-
    🎉 no goals
  -/


/-- A finite, free (as an `R`-module) Lie algebra `L` carries a bilinear form on `L`.

This is a specialisation of `LieModule.traceForm` to the adjoint representation of `L`. -/
noncomputable abbrev killingForm : LinearMap.BilinForm R L := LieModule.traceForm R L L


open LieAlgebra in
lemma killingForm_apply_apply (x y : L) : killingForm R L x y = trace R L (ad R L x ∘ₗ ad R L y) :=
  LieModule.traceForm_apply_apply R L L x y


lemma killingForm_eq_zero_of_mem_zeroRoot_mem_posFitting
    (H : LieSubalgebra R L) [LieAlgebra.IsNilpotent R H]
    {x₀ x₁ : L}
    (hx₀ : x₀ ∈ LieAlgebra.zeroRootSubalgebra R L H)
    (hx₁ : x₁ ∈ LieModule.posFittingComp R H L) :
    killingForm R L x₀ x₁ = 0 :=
  LieModule.eq_zero_of_mem_genWeightSpace_mem_posFitting R H L
    (fun x y z ↦ LieModule.traceForm_apply_lie_apply' R L L x y z) hx₀ hx₁


/-- The orthogonal complement of an ideal with respect to the killing form is an ideal. -/
noncomputable def killingCompl : LieIdeal R L :=
  LieAlgebra.InvariantForm.orthogonal (killingForm R L) (LieModule.traceForm_lieInvariant R L L) I


@[simp] lemma toSubmodule_killingCompl :
    LieSubmodule.toSubmodule I.killingCompl = (killingForm R L).orthogonal I.toSubmodule :=
  rfl


@[simp] lemma mem_killingCompl {x : L} :
    x ∈ I.killingCompl ↔ ∀ y ∈ I, killingForm R L y x = 0 := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x : L
    ⊢ Iff (Membership.mem (LieIdeal.killingCompl R L I) x) (∀ (y : L), Membership. …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma coe_killingCompl_top :
    killingCompl R L ⊤ = LinearMap.ker (killingForm R L) := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Eq (LieIdeal.toLieSubalgebra R L (LieIdeal.killingCompl R L Top.top)).toSubm …
  -/
  ext x
  /-
    case h
    R : Type u_1
    L : Type u_3
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    ⊢ Iff (Membership.mem (LieIdeal.toLieSubalgebra R L (LieIdeal.killingCompl R L …
  -/
  simp [LinearMap.ext_iff, LinearMap.BilinForm.IsOrtho, LieModule.traceForm_comm R L L x]
  /-
    🎉 no goals
  -/


lemma restrict_killingForm :
    (killingForm R L).restrict I = LieModule.traceForm R I L :=
  rfl


lemma killingForm_eq :
    killingForm R I = (killingForm R L).restrict I :=
                                                      /-
                                                        R : Type u_1
                                                        L : Type u_3
                                                        inst✝⁶ : CommRing R
                                                        inst✝⁵ : LieRing L
                                                        inst✝⁴ : LieAlgebra R L
                                                        I : LieIdeal R L
                                                        inst✝³ : Module.Free R L
                                                        inst✝² : Module.Finite R L
                                                        inst✝¹ : IsDomain R
                                                        inst✝ : IsPrincipalIdealRing R
                                                        ⊢ LE.le I (LieSubmodule.idealizer I)
                                                      -/
  LieSubmodule.traceForm_eq_of_le_idealizer I I <| by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] lemma le_killingCompl_top_of_isLieAbelian [IsLieAbelian I] :
    I ≤ LieIdeal.killingCompl R L ⊤ := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    I : LieIdeal R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ LE.le I (LieIdeal.killingCompl R L Top.top)
  -/
  intro x (hx : x ∈ I)
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    I : LieIdeal R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    x : L
    hx : Membership.mem I x
    ⊢ Membership.mem (LieIdeal.killingCompl R L Top.top) x
  -/
  simp only [mem_killingCompl, LieSubmodule.mem_top, forall_true_left]
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    I : LieIdeal R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    x : L
    hx : Membership.mem I x
    ⊢ ∀ (y : L), Eq (((killingForm R L) y) x) 0
  -/
  intro y
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    I : LieIdeal R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    x : L
    hx : Membership.mem I x
    y : L
    ⊢ Eq (((killingForm R L) y) x) 0
  -/
  rw [LieModule.traceForm_apply_apply]
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    I : LieIdeal R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    x : L
    hx : Membership.mem I x
    y : L
    ⊢ Eq ((LinearMap.trace R L) (LinearMap.comp ((LieModule.toEnd R L L) y) ((LieM …
  -/
  exact LieSubmodule.traceForm_eq_zero_of_isTrivial I I (by simp) _ hx
  /-
    🎉 no goals
  -/


lemma traceForm_eq_sum_finrank_nsmul_mul (x y : L) :
    traceForm K L M x y = ∑ χ : Weight K L M, finrank K (genWeightSpace M χ) • (χ x * χ y) := by
  have hxy : ∀ χ : Weight K L M, MapsTo (toEnd K L M x ∘ₗ toEnd K L M y)
      (genWeightSpace M χ) (genWeightSpace M χ) :=
    fun χ m hm ↦ LieSubmodule.lie_mem _ <| LieSubmodule.lie_mem _ hm
  classical
  have hds := DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top
    (LieSubmodule.iSupIndep_iff_toSubmodule.mp <| iSupIndep_genWeightSpace' K L M)
    (LieSubmodule.iSup_eq_top_iff_toSubmodule.mp <| iSup_genWeightSpace_eq_top' K L M)
  simp_rw [traceForm_apply_apply, LinearMap.trace_eq_sum_trace_restrict hds hxy,
    ← traceForm_genWeightSpace_eq K L M _ x y]
  rfl


/-- See also `LieModule.traceForm_eq_sum_finrank_nsmul'` for an expression omitting the zero
weights. -/
lemma traceForm_eq_sum_finrank_nsmul :
    traceForm K L M = ∑ χ : Weight K L M, finrank K (genWeightSpace M χ) •
      (χ : L →ₗ[K] K).smulRight (χ : L →ₗ[K] K) := by
  /-
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    ⊢ Eq (LieModule.traceForm K L M) (Finset.univ.sum fun χ => HSMul.hSMul (Module …
  -/
  ext
  /-
    case H
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    x✝ y✝ : L
    ⊢ Eq (((LieModule.traceForm K L M) x✝) y✝) (((Finset.univ.sum fun χ => HSMul.h …
  -/
  rw [traceForm_eq_sum_finrank_nsmul_mul, ← Finset.sum_attach]
  /-
    case H
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    x✝ y✝ : L
    ⊢ Eq (Finset.univ.attach.sum fun x => HSMul.hSMul (Module.finrank K (Subtype f …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A variant of `LieModule.traceForm_eq_sum_finrank_nsmul` in which the sum is taken only over the
non-zero weights. -/
lemma traceForm_eq_sum_finrank_nsmul' :
    traceForm K L M = ∑ χ in {χ : Weight K L M | χ.IsNonZero}, finrank K (genWeightSpace M χ) •
      (χ : L →ₗ[K] K).smulRight (χ : L →ₗ[K] K) := by
  classical
  suffices ∑ χ in {χ : Weight K L M | χ.IsZero}, finrank K (genWeightSpace M χ) •
      (χ : L →ₗ[K] K).smulRight (χ : L →ₗ[K] K) = 0 by
    rw [traceForm_eq_sum_finrank_nsmul,
      ← Finset.sum_filter_add_sum_filter_not (p := fun χ : Weight K L M ↦ χ.IsNonZero)]
    simp [this]
  refine Finset.sum_eq_zero fun χ hχ ↦ ?_
  replace hχ : (χ : L →ₗ[K] K) = 0 := by simpa [← Weight.coe_toLinear_eq_zero_iff] using hχ
  simp [hχ]

-- The reverse inclusion should also hold: TODO prove this!

lemma range_traceForm_le_span_weight :
    LinearMap.range (traceForm K L M) ≤ span K (range (Weight.toLinear K L M)) := by
  /-
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    ⊢ LE.le (LinearMap.range (LieModule.traceForm K L M)) (Submodule.span K (Set.r …
  -/
  rintro - ⟨x, rfl⟩
  /-
    case intro
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    x : L
    ⊢ Membership.mem (Submodule.span K (Set.range (LieModule.Weight.toLinear K L M …
  -/
  rw [LieModule.traceForm_eq_sum_finrank_nsmul, LinearMap.coeFn_sum, Finset.sum_apply]
  /-
    case intro
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    x : L
    ⊢ Membership.mem (Submodule.span K (Set.range (LieModule.Weight.toLinear K L M …
  -/
  refine Submodule.sum_mem _ fun χ _ ↦ ?_
  simp_rw [LinearMap.smul_apply, LinearMap.coe_smulRight, Weight.toLinear_apply,
    ← Nat.cast_smul_eq_nsmul K]
  /-
    case intro
    K : Type u_2
    L : Type u_3
    M : Type u_4
    inst✝¹⁰ : LieRing L
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : Field K
    inst✝⁶ : LieAlgebra K L
    inst✝⁵ : Module K M
    inst✝⁴ : LieModule K L M
    inst✝³ : FiniteDimensional K M
    inst✝² : LieAlgebra.IsNilpotent K L
    inst✝¹ : LieModule.LinearWeights K L M
    inst✝ : LieModule.IsTriangularizable K L M
    x : L
    χ : LieModule.Weight K L M
    x✝ : Membership.mem Finset.univ χ
    ⊢ Membership.mem (Submodule.span K (Set.range (LieModule.Weight.toLinear K L M …
  -/
  exact Submodule.smul_mem _ _ <| Submodule.smul_mem _ _ <| subset_span <| mem_range_self χ
  /-
    🎉 no goals
  -/


