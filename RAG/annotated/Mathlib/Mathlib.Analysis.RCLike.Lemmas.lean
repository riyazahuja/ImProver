theorem ofReal_eval (p : ℝ[X]) (x : ℝ) : (↑(p.eval x) : K) = aeval (↑x) p :=
  (@aeval_algebraMap_apply_eq_algebraMap_eval ℝ K _ _ _ x p).symm


variable (K) in
lemma RCLike.span_one_I : Submodule.span ℝ (M := K) {1, I} = ⊤ := by
  suffices ∀ x : K, ∃ a b : ℝ, a • 1 + b • I = x by
    simpa [Submodule.eq_top_iff', Submodule.mem_span_pair]
  /-
    K : Type u_1
    inst✝ : RCLike K
    ⊢ ∀ (x : K), Exists fun a => Exists fun b => Eq (HAdd.hAdd (HSMul.hSMul a 1) ( …
  -/
  exact fun x ↦ ⟨re x, im x, by simp [real_smul_eq_coe_mul]⟩
  /-
    🎉 no goals
  -/


variable (K) in
lemma RCLike.rank_le_two : Module.rank ℝ K ≤ 2 :=
  calc
                                                                 /-
                                                                   K : Type u_1
                                                                   inst✝ : RCLike K
                                                                   ⊢ Eq (Module.rank Real K) (Module.rank Real (Subtype fun x => Membership.mem ( …
                                                                 -/
    _ = Module.rank ℝ ↥(Submodule.span ℝ ({1, I} : Set K)) := by rw [span_one_I]; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
    _ ≤ #({1, I} : Finset K) := by
      -- TODO: `simp` doesn't rewrite inside the type argument to `Module.rank`, but `rw` does.
      -- We should introduce `Submodule.rank` to fix this.
      /-
        K : Type u_1
        inst✝ : RCLike K
        ⊢ LE.le (Module.rank Real (Subtype fun x => Membership.mem (Submodule.span Rea …
      -/
      have := rank_span_finset_le (R := ℝ) (M := K) {1, I}
      /-
        K : Type u_1
        inst✝ : RCLike K
        this : LE.le (Module.rank Real (Subtype fun x => Membership.mem (Submodule.spa …
        ⊢ LE.le (Module.rank Real (Subtype fun x => Membership.mem (Submodule.span Rea …
      -/
      rw [Finset.coe_pair] at this
      /-
        K : Type u_1
        inst✝ : RCLike K
        this : LE.le (Module.rank Real (Subtype fun x => Membership.mem (Submodule.spa …
        ⊢ LE.le (Module.rank Real (Subtype fun x => Membership.mem (Submodule.span Rea …
      -/
      simpa [span_one_I] using this
      /-
        🎉 no goals
      -/
    _ ≤ 2 := mod_cast Finset.card_le_two


variable (K) in
lemma RCLike.finrank_le_two : Module.finrank ℝ K ≤ 2 :=
  Module.finrank_le_of_rank_le <| rank_le_two _


/-- An `RCLike` field is finite-dimensional over `ℝ`, since it is spanned by `{1, I}`. -/
                                                               /-
                                                                 K : Type u_1
                                                                 E : Type u_2
                                                                 inst✝ : RCLike K
                                                                 ⊢ Eq (Submodule.span Real ↑(Insert.insert 1 (Singleton.singleton RCLike.I))) T …
                                                               -/
instance rclike_to_real : FiniteDimensional ℝ K := ⟨{1, I}, by simp [span_one_I]⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A finite dimensional vector space over an `RCLike` is a proper metric space.

This is not an instance because it would cause a search for `FiniteDimensional ?x E` before
`RCLike ?x`. -/
theorem proper_rclike [FiniteDimensional K E] : ProperSpace E := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝³ : RCLike K
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace K E
    inst✝ : FiniteDimensional K E
    ⊢ ProperSpace E
  -/
  letI : NormedSpace ℝ E := RestrictScalars.normedSpace ℝ K E
  /-
    K : Type u_1
    E : Type u_2
    inst✝³ : RCLike K
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace K E
    inst✝ : FiniteDimensional K E
    this : NormedSpace Real E := RestrictScalars.normedSpace Real K E
    ⊢ ProperSpace E
  -/
  letI : FiniteDimensional ℝ E := FiniteDimensional.trans ℝ K E
  /-
    K : Type u_1
    E : Type u_2
    inst✝³ : RCLike K
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace K E
    inst✝ : FiniteDimensional K E
    this✝ : NormedSpace Real E := RestrictScalars.normedSpace Real K E
    this : FiniteDimensional Real E := FiniteDimensional.trans Real K E
    ⊢ ProperSpace E
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance RCLike.properSpace_submodule (S : Submodule K E) [FiniteDimensional K S] :
    ProperSpace S :=
  proper_rclike K S


@[simp, rclike_simps]
theorem reCLM_norm : ‖(reCLM : K →L[ℝ] ℝ)‖ = 1 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    ⊢ Eq (Norm.norm RCLike.reCLM) 1
  -/
  apply le_antisymm (LinearMap.mkContinuous_norm_le _ zero_le_one _)
  /-
    K : Type u_1
    inst✝ : RCLike K
    ⊢ LE.le 1 (Norm.norm (RCLike.reLm.mkContinuous 1 ⋯))
  -/
  convert ContinuousLinearMap.ratio_le_opNorm (reCLM : K →L[ℝ] ℝ) (1 : K)
  /-
    case h.e'_3
    K : Type u_1
    inst✝ : RCLike K
    ⊢ Eq 1 (HDiv.hDiv (Norm.norm (RCLike.reCLM 1)) (Norm.norm 1))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, rclike_simps]
theorem conjCLE_norm : ‖(@conjCLE K _ : K →L[ℝ] K)‖ = 1 :=
  (@conjLIE K _).toLinearIsometry.norm_toContinuousLinearMap


@[simp, rclike_simps]
theorem ofRealCLM_norm : ‖(ofRealCLM : ℝ →L[ℝ] K)‖ = 1 :=
  -- Porting note: the following timed out
  -- LinearIsometry.norm_toContinuousLinearMap ofRealLI
  LinearIsometry.norm_toContinuousLinearMap _


open ComplexConjugate in
lemma aeval_conj (p : ℝ[X]) (z : K) : aeval (conj z) p = conj (aeval z p) :=
  aeval_algHom_apply (RCLike.conjAe (K := K)) z p


lemma aeval_ofReal (p : ℝ[X]) (x : ℝ) : aeval (RCLike.ofReal x : K) p = eval x p :=
  aeval_algHom_apply RCLike.ofRealAm x p


