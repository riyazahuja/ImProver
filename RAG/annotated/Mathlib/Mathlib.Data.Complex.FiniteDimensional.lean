instance : FiniteDimensional ℝ ℂ := .of_fintype_basis basisOneI


/-- `ℂ` is a finite extension of `ℝ` of degree 2, i.e `[ℂ : ℝ] = 2` -/
@[simp, stacks 09G4]
theorem finrank_real_complex : finrank ℝ ℂ = 2 := by
  /-
    ⊢ Eq (Module.finrank Real Complex) 2
  -/
  rw [finrank_eq_card_basis basisOneI, Fintype.card_fin]
  /-
    🎉 no goals
  -/


@[simp]
                                                      /-
                                                        ⊢ Eq (Module.rank Real Complex) 2
                                                      -/
theorem rank_real_complex : Module.rank ℝ ℂ = 2 := by simp [← finrank_eq_rank, finrank_real_complex]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem rank_real_complex'.{u} : Cardinal.lift.{u} (Module.rank ℝ ℂ) = 2 := by
  /-
    ⊢ Eq (Cardinal.lift.{u, 0} (Module.rank Real Complex)) 2
  -/
  rw [← finrank_eq_rank, finrank_real_complex, Cardinal.lift_natCast, Nat.cast_ofNat]
  /-
    🎉 no goals
  -/


/-- `Fact` version of the dimension of `ℂ` over `ℝ`, locally useful in the definition of the
circle. -/
theorem finrank_real_complex_fact : Fact (finrank ℝ ℂ = 2) :=
  ⟨finrank_real_complex⟩


instance (priority := 100) FiniteDimensional.complexToReal (E : Type*) [AddCommGroup E]
    [Module ℂ E] [FiniteDimensional ℂ E] : FiniteDimensional ℝ E :=
  FiniteDimensional.trans ℝ ℂ E


theorem rank_real_of_complex (E : Type*) [AddCommGroup E] [Module ℂ E] :
    Module.rank ℝ E = 2 * Module.rank ℂ E :=
  Cardinal.lift_inj.{_,0}.1 <| by
    /-
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Complex E
      ⊢ Eq (Cardinal.lift.{0, u_1} (Module.rank Real E)) (Cardinal.lift.{0, u_1} (HM …
    -/
    rw [← lift_rank_mul_lift_rank ℝ ℂ E, Complex.rank_real_complex']
    /-
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Complex E
      ⊢ Eq (HMul.hMul 2 (Cardinal.lift.{0, u_1} (Module.rank Complex E))) (Cardinal. …
    -/
    simp only [Cardinal.lift_id']
    /-
      🎉 no goals
    -/


theorem finrank_real_of_complex (E : Type*) [AddCommGroup E] [Module ℂ E] :
    Module.finrank ℝ E = 2 * Module.finrank ℂ E := by
  /-
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Complex E
    ⊢ Eq (Module.finrank Real E) (HMul.hMul 2 (Module.finrank Complex E))
  -/
  rw [← Module.finrank_mul_finrank ℝ ℂ E, Complex.finrank_real_complex]
  /-
    🎉 no goals
  -/


@[simp]
lemma Real.rank_rat_real : Module.rank ℚ ℝ = continuum := by
  /-
    ⊢ Eq (Module.rank Rat Real) Cardinal.continuum
  -/
  refine (Free.rank_eq_mk_of_infinite_lt ℚ ℝ ?_).trans mk_real
  /-
    ⊢ LT.lt (Cardinal.lift.{0, 0} (Cardinal.mk Rat)) (Cardinal.lift.{0, 0} (Cardin …
  -/
  simpa [mk_real] using aleph0_lt_continuum
  /-
    🎉 no goals
  -/


/-- `C` has an uncountable basis over `ℚ`. -/
@[simp, stacks 09G0]
lemma Complex.rank_rat_complex : Module.rank ℚ ℂ = continuum := by
  /-
    ⊢ Eq (Module.rank Rat Complex) Cardinal.continuum
  -/
  refine (Free.rank_eq_mk_of_infinite_lt ℚ ℂ ?_).trans mk_complex
  /-
    ⊢ LT.lt (Cardinal.lift.{0, 0} (Cardinal.mk Rat)) (Cardinal.lift.{0, 0} (Cardin …
  -/
  simpa using aleph0_lt_continuum
  /-
    🎉 no goals
  -/


/-- `ℂ` and `ℝ` are isomorphic as vector spaces over `ℚ`, or equivalently,
as additive groups. -/
theorem Complex.nonempty_linearEquiv_real : Nonempty (ℂ ≃ₗ[ℚ] ℝ) :=
                                                   /-
                                                     ⊢ Eq (Module.rank Rat Complex) (Module.rank Rat Real)
                                                   -/
  LinearEquiv.nonempty_equiv_iff_rank_eq.mpr <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


