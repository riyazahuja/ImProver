/-- Lagrange multipliers theorem: if `φ : E → ℝ` has a local extremum on the set `{x | f x = f x₀}`
at `x₀`, both `f : E → F` and `φ` are strictly differentiable at `x₀`, and the codomain of `f` is
a complete space, then the linear map `x ↦ (f' x, φ' x)` is not surjective. -/
theorem IsLocalExtrOn.range_ne_top_of_hasStrictFDerivAt
    (hextr : IsLocalExtrOn φ {x | f x = f x₀} x₀) (hf' : HasStrictFDerivAt f f' x₀)
    (hφ' : HasStrictFDerivAt φ φ' x₀) : LinearMap.range (f'.prod φ') ≠ ⊤ := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : E → F
    φ : E → Real
    x₀ : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    ⊢ Ne (LinearMap.range (f'.prod φ')) Top.top
  -/
  intro htop
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : E → F
    φ : E → Real
    x₀ : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    htop : Eq (LinearMap.range (f'.prod φ')) Top.top
    ⊢ False
  -/
  set fφ := fun x => (f x, φ x)
  have A : map φ (𝓝[f ⁻¹' {f x₀}] x₀) = 𝓝 (φ x₀) := by
    change map (Prod.snd ∘ fφ) (𝓝[fφ ⁻¹' {p | p.1 = f x₀}] x₀) = 𝓝 (φ x₀)
    rw [← map_map, nhdsWithin, map_inf_principal_preimage, (hf'.prod hφ').map_nhds_eq_of_surj htop]
    exact map_snd_nhdsWithin _
  /-
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : E → F
    φ : E → Real
    x₀ : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    htop : Eq (LinearMap.range (f'.prod φ')) Top.top
    fφ : E → Prod F Real := fun x => { fst := f x, snd := φ x }
    A : Eq (Filter.map φ (nhdsWithin x₀ (Set.preimage f (Singleton.singleton (f x₀ …
    ⊢ False
  -/
  exact hextr.not_nhds_le_map A.ge
  /-
    🎉 no goals
  -/


/-- Lagrange multipliers theorem: if `φ : E → ℝ` has a local extremum on the set `{x | f x = f x₀}`
at `x₀`, both `f : E → F` and `φ` are strictly differentiable at `x₀`, and the codomain of `f` is
a complete space, then there exist `Λ : dual ℝ F` and `Λ₀ : ℝ` such that `(Λ, Λ₀) ≠ 0` and
`Λ (f' x) + Λ₀ • φ' x = 0` for all `x`. -/
theorem IsLocalExtrOn.exists_linear_map_of_hasStrictFDerivAt
    (hextr : IsLocalExtrOn φ {x | f x = f x₀} x₀) (hf' : HasStrictFDerivAt f f' x₀)
    (hφ' : HasStrictFDerivAt φ φ' x₀) :
    ∃ (Λ : Module.Dual ℝ F) (Λ₀ : ℝ), (Λ, Λ₀) ≠ 0 ∧ ∀ x, Λ (f' x) + Λ₀ • φ' x = 0 := by
  rcases Submodule.exists_le_ker_of_lt_top _
      (lt_top_iff_ne_top.2 <| hextr.range_ne_top_of_hasStrictFDerivAt hf' hφ') with
    ⟨Λ', h0, hΛ'⟩
  set e : ((F →ₗ[ℝ] ℝ) × ℝ) ≃ₗ[ℝ] F × ℝ →ₗ[ℝ] ℝ :=
    ((LinearEquiv.refl ℝ (F →ₗ[ℝ] ℝ)).prod (LinearMap.ringLmapEquivSelf ℝ ℝ ℝ).symm).trans
      (LinearMap.coprodEquiv ℝ)
  /-
    case intro.intro
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : E → F
    φ : E → Real
    x₀ : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    Λ' : LinearMap (RingHom.id Real) (Prod F Real) Real
    h0 : Ne Λ' 0
    hΛ' : LE.le (LinearMap.range (f'.prod φ')) (LinearMap.ker Λ')
    e : LinearEquiv (RingHom.id Real) (Prod (LinearMap (RingHom.id Real) F Real) R …
    ⊢ Exists fun Λ => Exists fun Λ₀ => And (Ne { fst := Λ, snd := Λ₀ } 0) (∀ (x :  …
  -/
  rcases e.surjective Λ' with ⟨⟨Λ, Λ₀⟩, rfl⟩
  /-
    case intro.intro.intro.mk
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : E → F
    φ : E → Real
    x₀ : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    e : LinearEquiv (RingHom.id Real) (Prod (LinearMap (RingHom.id Real) F Real) R …
    Λ : LinearMap (RingHom.id Real) F Real
    Λ₀ : Real
    h0 : Ne (e { fst := Λ, snd := Λ₀ }) 0
    hΛ' : LE.le (LinearMap.range (f'.prod φ')) (LinearMap.ker (e { fst := Λ, snd : …
    ⊢ Exists fun Λ => Exists fun Λ₀ => And (Ne { fst := Λ, snd := Λ₀ } 0) (∀ (x :  …
  -/
  refine ⟨Λ, Λ₀, e.map_ne_zero_iff.1 h0, fun x => ?_⟩
  /-
    case intro.intro.intro.mk
    E : Type u_1
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : CompleteSpace E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : E → F
    φ : E → Real
    x₀ : E
    f' : ContinuousLinearMap (RingHom.id Real) E F
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    e : LinearEquiv (RingHom.id Real) (Prod (LinearMap (RingHom.id Real) F Real) R …
    Λ : LinearMap (RingHom.id Real) F Real
    Λ₀ : Real
    h0 : Ne (e { fst := Λ, snd := Λ₀ }) 0
    hΛ' : LE.le (LinearMap.range (f'.prod φ')) (LinearMap.ker (e { fst := Λ, snd : …
    x : E
    ⊢ Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
  -/
  convert LinearMap.congr_fun (LinearMap.range_le_ker_iff.1 hΛ') x using 1
  -- squeezed `simp [mul_comm]` to speed up elaboration
  simp only [e, smul_eq_mul, LinearEquiv.trans_apply, LinearEquiv.prod_apply,
    LinearEquiv.refl_apply, LinearMap.ringLmapEquivSelf_symm_apply, LinearMap.coprodEquiv_apply,
    ContinuousLinearMap.coe_prod, LinearMap.coprod_comp_prod, LinearMap.add_apply,
    LinearMap.coe_comp, ContinuousLinearMap.coe_coe, Function.comp_apply, LinearMap.coe_smulRight,
    LinearMap.one_apply, mul_comm]


/-- Lagrange multipliers theorem: if `φ : E → ℝ` has a local extremum on the set `{x | f x = f x₀}`
at `x₀`, and both `f : E → ℝ` and `φ` are strictly differentiable at `x₀`, then there exist
`a b : ℝ` such that `(a, b) ≠ 0` and `a • f' + b • φ' = 0`. -/
theorem IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt_1d {f : E → ℝ} {f' : E →L[ℝ] ℝ}
    (hextr : IsLocalExtrOn φ {x | f x = f x₀} x₀) (hf' : HasStrictFDerivAt f f' x₀)
    (hφ' : HasStrictFDerivAt φ φ' x₀) : ∃ a b : ℝ, (a, b) ≠ 0 ∧ a • f' + b • φ' = 0 := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    ⊢ Exists fun a => Exists fun b => And (Ne { fst := a, snd := b } 0) (Eq (HAdd. …
  -/
  obtain ⟨Λ, Λ₀, hΛ, hfΛ⟩ := hextr.exists_linear_map_of_hasStrictFDerivAt hf' hφ'
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
    hf' : HasStrictFDerivAt f f' x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    Λ : Module.Dual Real Real
    Λ₀ : Real
    hΛ : Ne { fst := Λ, snd := Λ₀ } 0
    hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
    ⊢ Exists fun a => Exists fun b => And (Ne { fst := a, snd := b } 0) (Eq (HAdd. …
  -/
  refine ⟨Λ 1, Λ₀, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hΛ : Ne { fst := Λ, snd := Λ₀ } 0
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      ⊢ Ne { fst := Λ 1, snd := Λ₀ } 0
    -/
  · contrapose! hΛ
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      hΛ : Eq { fst := Λ 1, snd := Λ₀ } 0
      ⊢ Eq { fst := Λ, snd := Λ₀ } 0
    -/
    simp only [Prod.mk_eq_zero] at hΛ ⊢
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      hΛ : And (Eq (Λ 1) 0) (Eq Λ₀ 0)
      ⊢ And (Eq Λ 0) (Eq Λ₀ 0)
    -/
    refine ⟨LinearMap.ext fun x => ?_, hΛ.2⟩
    /-
      case intro.intro.intro.refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      hΛ : And (Eq (Λ 1) 0) (Eq Λ₀ 0)
      x : Real
      ⊢ Eq (Λ x) (0 x)
    -/
    simpa [hΛ.1] using Λ.map_smul x 1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hΛ : Ne { fst := Λ, snd := Λ₀ } 0
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Λ 1) f') (HSMul.hSMul Λ₀ φ')) 0
    -/
  · ext x
    have H₁ : Λ (f' x) = f' x * Λ 1 := by
      simpa only [mul_one, Algebra.id.smul_eq_mul] using Λ.map_smul (f' x) 1
    /-
      case intro.intro.intro.refine_2.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hΛ : Ne { fst := Λ, snd := Λ₀ } 0
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      x : E
      H₁ : Eq (Λ (f' x)) (HMul.hMul (f' x) (Λ 1))
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul (Λ 1) f') (HSMul.hSMul Λ₀ φ')) x) (0 x)
    -/
    have H₂ : f' x * Λ 1 + Λ₀ * φ' x = 0 := by simpa only [Algebra.id.smul_eq_mul, H₁] using hfΛ x
    /-
      case intro.intro.intro.refine_2.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (f x) (f x₀)) x₀
      hf' : HasStrictFDerivAt f f' x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      Λ : Module.Dual Real Real
      Λ₀ : Real
      hΛ : Ne { fst := Λ, snd := Λ₀ } 0
      hfΛ : ∀ (x : E), Eq (HAdd.hAdd (Λ (f' x)) (HSMul.hSMul Λ₀ (φ' x))) 0
      x : E
      H₁ : Eq (Λ (f' x)) (HMul.hMul (f' x) (Λ 1))
      H₂ : Eq (HAdd.hAdd (HMul.hMul (f' x) (Λ 1)) (HMul.hMul Λ₀ (φ' x))) 0
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul (Λ 1) f') (HSMul.hSMul Λ₀ φ')) x) (0 x)
    -/
    simpa [mul_comm] using H₂
    /-
      🎉 no goals
    -/


/-- Lagrange multipliers theorem, 1d version. Let `f : ι → E → ℝ` be a finite family of functions.
Suppose that `φ : E → ℝ` has a local extremum on the set `{x | ∀ i, f i x = f i x₀}` at `x₀`.
Suppose that all functions `f i` as well as `φ` are strictly differentiable at `x₀`.
Then the derivatives `f' i : E → L[ℝ] ℝ` and `φ' : E →L[ℝ] ℝ` are linearly dependent:
there exist `Λ : ι → ℝ` and `Λ₀ : ℝ`, `(Λ, Λ₀) ≠ 0`, such that `∑ i, Λ i • f' i + Λ₀ • φ' = 0`.

See also `IsLocalExtrOn.linear_dependent_of_hasStrictFDerivAt` for a version that
states `¬LinearIndependent ℝ _` instead of existence of `Λ` and `Λ₀`. -/
theorem IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt {ι : Type*} [Fintype ι]
    {f : ι → E → ℝ} {f' : ι → E →L[ℝ] ℝ} (hextr : IsLocalExtrOn φ {x | ∀ i, f i x = f i x₀} x₀)
    (hf' : ∀ i, HasStrictFDerivAt (f i) (f' i) x₀) (hφ' : HasStrictFDerivAt φ φ' x₀) :
    ∃ (Λ : ι → ℝ) (Λ₀ : ℝ), (Λ, Λ₀) ≠ 0 ∧ (∑ i, Λ i • f' i) + Λ₀ • φ' = 0 := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Fintype ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => ∀ (i : ι), Eq (f i x) (f i x₀)) x₀
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    ⊢ Exists fun Λ => Exists fun Λ₀ => And (Ne { fst := Λ, snd := Λ₀ } 0) (Eq (HAd …
  -/
  letI := Classical.decEq ι
  replace hextr : IsLocalExtrOn φ {x | (fun i => f i x) = fun i => f i x₀} x₀ := by
    simpa only [funext_iff] using hextr
  rcases hextr.exists_linear_map_of_hasStrictFDerivAt (hasStrictFDerivAt_pi.2 fun i => hf' i)
      hφ' with
    ⟨Λ, Λ₀, h0, hsum⟩
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Fintype ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    this : DecidableEq ι := Classical.decEq ι
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (fun i => f i x) fun i => f i x₀) x₀
    Λ : Module.Dual Real (ι → Real)
    Λ₀ : Real
    h0 : Ne { fst := Λ, snd := Λ₀ } 0
    hsum : ∀ (x : E), Eq (HAdd.hAdd (Λ ((ContinuousLinearMap.pi f') x)) (HSMul.hSM …
    ⊢ Exists fun Λ => Exists fun Λ₀ => And (Ne { fst := Λ, snd := Λ₀ } 0) (Eq (HAd …
  -/
  rcases (LinearEquiv.piRing ℝ ℝ ι ℝ).symm.surjective Λ with ⟨Λ, rfl⟩
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Fintype ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    this : DecidableEq ι := Classical.decEq ι
    hextr : IsLocalExtrOn φ (setOf fun x => Eq (fun i => f i x) fun i => f i x₀) x₀
    Λ₀ : Real
    Λ : ι → Real
    h0 : Ne { fst := (LinearEquiv.piRing Real Real ι Real).symm Λ, snd := Λ₀ } 0
    hsum : ∀ (x : E), Eq (HAdd.hAdd (((LinearEquiv.piRing Real Real ι Real).symm Λ …
    ⊢ Exists fun Λ => Exists fun Λ₀ => And (Ne { fst := Λ, snd := Λ₀ } 0) (Eq (HAd …
  -/
  refine ⟨Λ, Λ₀, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      ι : Type u_3
      inst✝ : Fintype ι
      f : ι → E → Real
      f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
      hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      this : DecidableEq ι := Classical.decEq ι
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (fun i => f i x) fun i => f i x₀) x₀
      Λ₀ : Real
      Λ : ι → Real
      h0 : Ne { fst := (LinearEquiv.piRing Real Real ι Real).symm Λ, snd := Λ₀ } 0
      hsum : ∀ (x : E), Eq (HAdd.hAdd (((LinearEquiv.piRing Real Real ι Real).symm Λ …
      ⊢ Ne { fst := Λ, snd := Λ₀ } 0
    -/
  · simpa only [Ne, Prod.ext_iff, LinearEquiv.map_eq_zero_iff, Prod.fst_zero] using h0
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      ι : Type u_3
      inst✝ : Fintype ι
      f : ι → E → Real
      f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
      hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      this : DecidableEq ι := Classical.decEq ι
      hextr : IsLocalExtrOn φ (setOf fun x => Eq (fun i => f i x) fun i => f i x₀) x₀
      Λ₀ : Real
      Λ : ι → Real
      h0 : Ne { fst := (LinearEquiv.piRing Real Real ι Real).symm Λ, snd := Λ₀ } 0
      hsum : ∀ (x : E), Eq (HAdd.hAdd (((LinearEquiv.piRing Real Real ι Real).symm Λ …
      ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (Λ i) (f' i)) (HSMul.hSM …
    -/
  · ext x; simpa [mul_comm] using hsum x
           /-
             🎉 no goals
           -/


/-- Lagrange multipliers theorem. Let `f : ι → E → ℝ` be a finite family of functions.
Suppose that `φ : E → ℝ` has a local extremum on the set `{x | ∀ i, f i x = f i x₀}` at `x₀`.
Suppose that all functions `f i` as well as `φ` are strictly differentiable at `x₀`.
Then the derivatives `f' i : E → L[ℝ] ℝ` and `φ' : E →L[ℝ] ℝ` are linearly dependent.

See also `IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt` for a version that
that states existence of Lagrange multipliers `Λ` and `Λ₀` instead of using
`¬LinearIndependent ℝ _` -/
theorem IsLocalExtrOn.linear_dependent_of_hasStrictFDerivAt {ι : Type*} [Finite ι] {f : ι → E → ℝ}
    {f' : ι → E →L[ℝ] ℝ} (hextr : IsLocalExtrOn φ {x | ∀ i, f i x = f i x₀} x₀)
    (hf' : ∀ i, HasStrictFDerivAt (f i) (f' i) x₀) (hφ' : HasStrictFDerivAt φ φ' x₀) :
    ¬LinearIndependent ℝ (Option.elim' φ' f' : Option ι → E →L[ℝ] ℝ) := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Finite ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => ∀ (i : ι), Eq (f i x) (f i x₀)) x₀
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    ⊢ Not (LinearIndependent Real (Option.elim' φ' f'))
  -/
  cases nonempty_fintype ι
  /-
    case intro
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Finite ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => ∀ (i : ι), Eq (f i x) (f i x₀)) x₀
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    val✝ : Fintype ι
    ⊢ Not (LinearIndependent Real (Option.elim' φ' f'))
  -/
  rw [Fintype.linearIndependent_iff]; push_neg
  /-
    case intro
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Finite ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => ∀ (i : ι), Eq (f i x) (f i x₀)) x₀
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    val✝ : Fintype ι
    ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Option. …
  -/
  rcases hextr.exists_multipliers_of_hasStrictFDerivAt hf' hφ' with ⟨Λ, Λ₀, hΛ, hΛf⟩
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    φ : E → Real
    x₀ : E
    φ' : ContinuousLinearMap (RingHom.id Real) E Real
    ι : Type u_3
    inst✝ : Finite ι
    f : ι → E → Real
    f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
    hextr : IsLocalExtrOn φ (setOf fun x => ∀ (i : ι), Eq (f i x) (f i x₀)) x₀
    hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
    hφ' : HasStrictFDerivAt φ φ' x₀
    val✝ : Fintype ι
    Λ : ι → Real
    Λ₀ : Real
    hΛ : Ne { fst := Λ, snd := Λ₀ } 0
    hΛf : Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (Λ i) (f' i)) (HSMul …
    ⊢ Exists fun g => And (Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Option. …
  -/
  refine ⟨Option.elim' Λ₀ Λ, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : CompleteSpace E
      φ : E → Real
      x₀ : E
      φ' : ContinuousLinearMap (RingHom.id Real) E Real
      ι : Type u_3
      inst✝ : Finite ι
      f : ι → E → Real
      f' : ι → ContinuousLinearMap (RingHom.id Real) E Real
      hextr : IsLocalExtrOn φ (setOf fun x => ∀ (i : ι), Eq (f i x) (f i x₀)) x₀
      hf' : ∀ (i : ι), HasStrictFDerivAt (f i) (f' i) x₀
      hφ' : HasStrictFDerivAt φ φ' x₀
      val✝ : Fintype ι
      Λ : ι → Real
      Λ₀ : Real
      hΛ : Ne { fst := Λ, snd := Λ₀ } 0
      hΛf : Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (Λ i) (f' i)) (HSMul …
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (Option.elim' Λ₀ Λ i) (Option.elim' …
    -/
  · simpa [add_comm] using hΛf
    /-
      🎉 no goals
    -/
  · simpa only [funext_iff, not_and_or, or_comm, Option.exists, Prod.mk_eq_zero, Ne,
      not_forall] using hΛ

