/-- We say a Lie algebra is Killing if its Killing form is non-singular.

NB: This is not standard terminology (the literature does not seem to name Lie algebras with this
property). -/
class IsKilling : Prop where
  /-- We say a Lie algebra is Killing if its Killing form is non-singular. -/
  killingCompl_top_eq_bot : LieIdeal.killingCompl R L ⊤ = ⊥


@[simp] lemma ker_killingForm_eq_bot :
    LinearMap.ker (killingForm R L) = ⊥ := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsKilling R L
    ⊢ Eq (LinearMap.ker (killingForm R L)) Bot.bot
  -/
  simp [← LieIdeal.coe_killingCompl_top, killingCompl_top_eq_bot]
  /-
    🎉 no goals
  -/


lemma killingForm_nondegenerate :
    (killingForm R L).Nondegenerate := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝³ : CommRing R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : LieAlgebra.IsKilling R L
    ⊢ (killingForm R L).Nondegenerate
  -/
  simp [LinearMap.BilinForm.nondegenerate_iff_ker_eq_bot]
  /-
    🎉 no goals
  -/


variable {R L} in
lemma ideal_eq_bot_of_isLieAbelian
    [Module.Free R L] [Module.Finite R L] [IsDomain R] [IsPrincipalIdealRing R]
    (I : LieIdeal R L) [IsLieAbelian I] : I = ⊥ := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : LieAlgebra.IsKilling R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    I : LieIdeal R L
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ Eq I Bot.bot
  -/
  rw [eq_bot_iff, ← killingCompl_top_eq_bot]
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    inst✝⁵ : LieAlgebra.IsKilling R L
    inst✝⁴ : Module.Free R L
    inst✝³ : Module.Finite R L
    inst✝² : IsDomain R
    inst✝¹ : IsPrincipalIdealRing R
    I : LieIdeal R L
    inst✝ : IsLieAbelian (Subtype fun x => Membership.mem I x)
    ⊢ LE.le I (LieIdeal.killingCompl R L Top.top)
  -/
  exact I.le_killingCompl_top_of_isLieAbelian
  /-
    🎉 no goals
  -/


instance instSemisimple [IsKilling K L] [Module.Finite K L] : IsSemisimple K L := by
  /-
    R : Type u_1
    K : Type u_2
    L : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : Field K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : LieAlgebra K L
    inst✝² : LieAlgebra.IsKilling R L
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : Module.Finite K L
    ⊢ LieAlgebra.IsSemisimple K L
  -/
  apply InvariantForm.isSemisimple_of_nondegenerate (Φ := killingForm K L)
    /-
      case hΦ_nondeg
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : Field K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : LieAlgebra K L
      inst✝² : LieAlgebra.IsKilling R L
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : Module.Finite K L
      ⊢ (killingForm K L).Nondegenerate
    -/
  · exact IsKilling.killingForm_nondegenerate _ _
    /-
      🎉 no goals
    -/
    /-
      case hΦ_inv
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : Field K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : LieAlgebra K L
      inst✝² : LieAlgebra.IsKilling R L
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : Module.Finite K L
      ⊢ LinearMap.BilinForm.lieInvariant L (killingForm K L)
    -/
  · exact LieModule.traceForm_lieInvariant _ _ _
    /-
      🎉 no goals
    -/
    /-
      case hΦ_refl
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : Field K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : LieAlgebra K L
      inst✝² : LieAlgebra.IsKilling R L
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : Module.Finite K L
      ⊢ (killingForm K L).IsRefl
    -/
  · exact (LieModule.traceForm_isSymm K L L).isRefl
    /-
      🎉 no goals
    -/
    /-
      case hL
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : Field K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : LieAlgebra K L
      inst✝² : LieAlgebra.IsKilling R L
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : Module.Finite K L
      ⊢ ∀ (I : LieIdeal K L), IsAtom I → Not (IsLieAbelian (Subtype fun x => Members …
    -/
  · intro I h₁ h₂
    /-
      case hL
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : Field K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : LieAlgebra K L
      inst✝² : LieAlgebra.IsKilling R L
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : Module.Finite K L
      I : LieIdeal K L
      h₁ : IsAtom I
      h₂ : IsLieAbelian (Subtype fun x => Membership.mem I x)
      ⊢ False
    -/
    exact h₁.1 <| IsKilling.ideal_eq_bot_of_isLieAbelian I
    /-
      🎉 no goals
    -/


/-- The converse of this is true over a field of characteristic zero. There are counterexamples
over fields with positive characteristic.

Note that when the coefficients are a field this instance is redundant since we have
`LieAlgebra.IsKilling.instSemisimple` and `LieAlgebra.IsSemisimple.instHasTrivialRadical`. -/
instance instHasTrivialRadical
    [Module.Free R L] [Module.Finite R L] [IsDomain R] [IsPrincipalIdealRing R] :
    HasTrivialRadical R L :=
  (hasTrivialRadical_iff_no_abelian_ideals R L).mpr IsKilling.ideal_eq_bot_of_isLieAbelian


/-- Given an equivalence `e` of Lie algebras from `L` to `L'`, and elements `x y : L`, the
respective Killing forms of `L` and `L'` satisfy `κ'(e x, e y) = κ(x, y)`. -/
@[simp] lemma killingForm_of_equiv_apply (e : L ≃ₗ⁅R⁆ L') (x y : L) :
    killingForm R L' (e x) (e y) = killingForm R L x y := by
  simp_rw [killingForm_apply_apply, ← LieAlgebra.conj_ad_apply, ← LinearEquiv.conj_comp,
    LinearMap.trace_conj']


/-- Given a Killing Lie algebra `L`, if `L'` is isomorphic to `L`, then `L'` is Killing too. -/
lemma isKilling_of_equiv [IsKilling R L] (e : L ≃ₗ⁅R⁆ L') : IsKilling R L' := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    ⊢ LieAlgebra.IsKilling R L'
  -/
  constructor
  /-
    case killingCompl_top_eq_bot
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    ⊢ Eq (LieIdeal.killingCompl R L' Top.top) Bot.bot
  -/
  ext x'
  /-
    case killingCompl_top_eq_bot.h
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    x' : L'
    ⊢ Iff (Membership.mem (LieIdeal.killingCompl R L' Top.top) x') (Membership.mem …
  -/
  simp_rw [LieIdeal.mem_killingCompl, LieModule.traceForm_comm]
  /-
    case killingCompl_top_eq_bot.h
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    x' : L'
    ⊢ Iff (∀ (y : L'), Membership.mem Top.top y → Eq (((LieModule.traceForm R L' L …
  -/
  refine ⟨fun hx' ↦ ?_, fun hx y _ ↦ hx ▸ LinearMap.map_zero₂ (killingForm R L') y⟩
  suffices e.symm x' ∈ LinearMap.ker (killingForm R L) by
    rw [IsKilling.ker_killingForm_eq_bot] at this
    simpa [map_zero] using (e : L ≃ₗ[R] L').congr_arg this
  /-
    case killingCompl_top_eq_bot.h
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    x' : L'
    hx' : ∀ (y : L'), Membership.mem Top.top y → Eq (((LieModule.traceForm R L' L' …
    ⊢ Membership.mem (LinearMap.ker (killingForm R L)) (e.symm x')
  -/
  ext y
  /-
    case killingCompl_top_eq_bot.h.h
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    x' : L'
    hx' : ∀ (y : L'), Membership.mem Top.top y → Eq (((LieModule.traceForm R L' L' …
    y : L
    ⊢ Eq (((killingForm R L) (e.symm x')) y) (0 y)
  -/
  replace hx' : ∀ y', killingForm R L' x' y' = 0 := by simpa using hx'
  /-
    case killingCompl_top_eq_bot.h.h
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    x' : L'
    y : L
    hx' : ∀ (y' : L'), Eq (((killingForm R L') x') y') 0
    ⊢ Eq (((killingForm R L) (e.symm x')) y) (0 y)
  -/
  specialize hx' (e y)
  /-
    case killingCompl_top_eq_bot.h.h
    R : Type u_1
    L : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : LieRing L
    inst✝³ : LieAlgebra R L
    L' : Type u_4
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra.IsKilling R L
    e : LieEquiv R L L'
    x' : L'
    y : L
    hx' : Eq (((killingForm R L') x') (e y)) 0
    ⊢ Eq (((killingForm R L) (e.symm x')) y) (0 y)
  -/
  rwa [← e.apply_symm_apply x', killingForm_of_equiv_apply] at hx'
  /-
    🎉 no goals
  -/


alias _root_.LieEquiv.isKilling := LieAlgebra.isKilling_of_equiv


