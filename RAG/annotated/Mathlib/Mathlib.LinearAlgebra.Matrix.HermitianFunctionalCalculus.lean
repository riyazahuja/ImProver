lemma finite_real_spectrum : (spectrum ℝ A).Finite := by
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    ⊢ (spectrum Real A).Finite
  -/
  rw [← spectrum.preimage_algebraMap 𝕜]
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    ⊢ (Set.preimage (⇑(algebraMap Real 𝕜)) (spectrum 𝕜 A)).Finite
  -/
  exact A.finite_spectrum.preimage (NoZeroSMulDivisors.algebraMap_injective ℝ 𝕜).injOn
  /-
    🎉 no goals
  -/


instance : Finite (spectrum ℝ A) := A.finite_real_spectrum


/-- The `ℝ`-spectrum of a Hermitian matrix over `RCLike` field is the range of the eigenvalue
function -/
theorem eigenvalues_eq_spectrum_real {a : Matrix n n 𝕜} (ha : IsHermitian a) :
    spectrum ℝ a = Set.range (ha.eigenvalues) := by
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    a : Matrix n n 𝕜
    ha : a.IsHermitian
    ⊢ Eq (spectrum Real a) (Set.range ha.eigenvalues)
  -/
  ext x
  conv_lhs => rw [ha.spectral_theorem, unitary.spectrum.unitary_conjugate,
  ← spectrum.algebraMap_mem_iff 𝕜, spectrum_diagonal, RCLike.algebraMap_eq_ofReal]
  /-
    case h
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    a : Matrix n n 𝕜
    ha : a.IsHermitian
    x : Real
    ⊢ Iff (Membership.mem (Set.range (Function.comp RCLike.ofReal ha.eigenvalues)) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The star algebra homomorphism underlying the instance of the continuous functional
calculus of a Hermitian matrix. This is an auxiliary definition and is not intended
for use outside of this file. -/
@[simps]
noncomputable def cfcAux : C(spectrum ℝ A, ℝ) →⋆ₐ[ℝ] (Matrix n n 𝕜) where
  toFun := fun g => (eigenvectorUnitary hA : Matrix n n 𝕜) *
    diagonal (RCLike.ofReal ∘ g ∘ (fun i ↦ ⟨hA.eigenvalues i, hA.eigenvalues_mem_spectrum_real i⟩))
    * star (eigenvectorUnitary hA : Matrix n n 𝕜)
                 /-
                   n : Type u_1
                   𝕜 : Type u_2
                   inst✝² : RCLike 𝕜
                   inst✝¹ : Fintype n
                   inst✝ : DecidableEq n
                   A : Matrix n n 𝕜
                   hA : A.IsHermitian
                   ⊢ Eq ((fun g => HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Matrix.diagonal …
                 -/
  map_one' := by simp [Pi.one_def (f := fun _ : n ↦ 𝕜)]
                 /-
                   🎉 no goals
                 -/
  map_mul' f g := by
    have {a b c d e f : Matrix n n 𝕜} : (a * b * c) * (d * e * f) = a * (b * (c * d) * e) * f := by
      simp only [mul_assoc]
    simp only [this, ContinuousMap.coe_mul, SetLike.coe_mem, unitary.star_mul_self_of_mem, mul_one,
      diagonal_mul_diagonal, Function.comp_apply]
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f g : ContinuousMap (↑(spectrum Real A)) Real
      this : ∀ {a b c d e f : Matrix n n 𝕜}, Eq (HMul.hMul (HMul.hMul (HMul.hMul a b …
      ⊢ Eq (HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Matrix.diagonal (Function …
    -/
    congr! with i
    /-
      case h.e'_5.h.e'_6.h.e'_5.h
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f g : ContinuousMap (↑(spectrum Real A)) Real
      this : ∀ {a b c d e f : Matrix n n 𝕜}, Eq (HMul.hMul (HMul.hMul (HMul.hMul a b …
      i : n
      ⊢ Eq (Function.comp RCLike.ofReal (Function.comp (HMul.hMul ⇑f ⇑g) fun i => ⟨h …
    -/
    simp
    /-
      🎉 no goals
    -/
                  /-
                    n : Type u_1
                    𝕜 : Type u_2
                    inst✝² : RCLike 𝕜
                    inst✝¹ : Fintype n
                    inst✝ : DecidableEq n
                    A : Matrix n n 𝕜
                    hA : A.IsHermitian
                    ⊢ Eq ((↑{ toFun := fun g => HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Mat …
                  -/
  map_zero' := by simp [Pi.zero_def (f := fun _ : n ↦ 𝕜)]
                  /-
                    🎉 no goals
                  -/
  map_add' f g := by
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f g : ContinuousMap (↑(spectrum Real A)) Real
      ⊢ Eq ((↑{ toFun := fun g => HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Mat …
    -/
    simp only [ContinuousMap.coe_add, ← add_mul, ← mul_add, diagonal_add, Function.comp_apply]
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f g : ContinuousMap (↑(spectrum Real A)) Real
      ⊢ Eq (HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Matrix.diagonal (Function …
    -/
    congr! with i
    /-
      case h.e'_5.h.e'_6.h.e'_5.h
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f g : ContinuousMap (↑(spectrum Real A)) Real
      i : n
      ⊢ Eq (Function.comp RCLike.ofReal (Function.comp (HAdd.hAdd ⇑f ⇑g) fun i => ⟨h …
    -/
    simp
    /-
      🎉 no goals
    -/
  commutes' r := by
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      r : Real
      ⊢ Eq ((↑↑{ toFun := fun g => HMul.hMul (HMul.hMul (↑hA.eigenvectorUnitary) (Ma …
    -/
    simp only [Function.comp_def, algebraMap_apply, smul_eq_mul, mul_one]
    rw [← mul_one (algebraMap _ _ _), ← unitary.coe_mul_star_self hA.eigenvectorUnitary,
      ← Algebra.left_comm, unitary.coe_star, mul_assoc]
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      r : Real
      ⊢ Eq (HMul.hMul (↑hA.eigenvectorUnitary) (HMul.hMul (Matrix.diagonal fun x =>  …
    -/
    congr!
    /-
      🎉 no goals
    -/
  map_star' f := by
    simp only [star_trivial, StarMul.star_mul, star_star, star_eq_conjTranspose (diagonal _),
      diagonal_conjTranspose, mul_assoc]
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f : ContinuousMap (↑(spectrum Real A)) Real
      ⊢ Eq (HMul.hMul (↑hA.eigenvectorUnitary) (HMul.hMul (Matrix.diagonal (Function …
    -/
    congr!
    /-
      case h.e'_6.h.e'_5.h.e'_5
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f : ContinuousMap (↑(spectrum Real A)) Real
      ⊢ Eq (Function.comp RCLike.ofReal (Function.comp ⇑f fun i => ⟨hA.eigenvalues i …
    -/
    ext
    /-
      case h.e'_6.h.e'_5.h.e'_5.h
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      f : ContinuousMap (↑(spectrum Real A)) Real
      x✝ : n
      ⊢ Eq (Function.comp RCLike.ofReal (Function.comp ⇑f fun i => ⟨hA.eigenvalues i …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma isClosedEmbedding_cfcAux : IsClosedEmbedding hA.cfcAux := by
  have h0 : FiniteDimensional ℝ C(spectrum ℝ A, ℝ) :=
    FiniteDimensional.of_injective (ContinuousMap.coeFnLinearMap ℝ (M := ℝ)) DFunLike.coe_injective
  refine LinearMap.isClosedEmbedding_of_injective (𝕜 := ℝ) (E := C(spectrum ℝ A, ℝ))
    (F := Matrix n n 𝕜) (f := hA.cfcAux) <| LinearMap.ker_eq_bot'.mpr fun f hf ↦ ?_
  have h2 :
      diagonal (RCLike.ofReal ∘ f ∘ fun i ↦ ⟨hA.eigenvalues i, hA.eigenvalues_mem_spectrum_real i⟩)
        = (0 : Matrix n n 𝕜) := by
    simp only [LinearMap.coe_coe, cfcAux_apply] at hf
    replace hf := congr($(hf) * (eigenvectorUnitary hA : Matrix n n 𝕜))
    simp only [mul_assoc, SetLike.coe_mem, unitary.star_mul_self_of_mem, mul_one, zero_mul] at hf
    simpa [← mul_assoc] using congr((star hA.eigenvectorUnitary : Matrix n n 𝕜) * $(hf))
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    ⊢ Eq f 0
  -/
  ext x
  /-
    case h
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    x : ↑(spectrum Real A)
    ⊢ Eq (f x) (0 x)
  -/
  simp only [ContinuousMap.zero_apply]
  /-
    case h
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    x : ↑(spectrum Real A)
    ⊢ Eq (f x) 0
  -/
  obtain ⟨x, hx⟩ := x
  /-
    case h.mk
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    x : Real
    hx : Membership.mem (spectrum Real A) x
    ⊢ Eq (f ⟨x, hx⟩) 0
  -/
  obtain ⟨i, rfl⟩ := hA.eigenvalues_eq_spectrum_real ▸ hx
  /-
    case h.mk.intro
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    i : n
    hx : Membership.mem (spectrum Real A) (hA.eigenvalues i)
    ⊢ Eq (f ⟨hA.eigenvalues i, hx⟩) 0
  -/
  rw [← diagonal_zero] at h2
  /-
    case h.mk.intro
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    i : n
    hx : Membership.mem (spectrum Real A) (hA.eigenvalues i)
    ⊢ Eq (f ⟨hA.eigenvalues i, hx⟩) 0
  -/
  have := (diagonal_eq_diagonal_iff).mp h2
  /-
    case h.mk.intro
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    h0 : FiniteDimensional Real (ContinuousMap (↑(spectrum Real A)) Real)
    f : ContinuousMap (↑(spectrum Real A)) Real
    hf : Eq (↑hA.cfcAux f) 0
    h2 : Eq (Matrix.diagonal (Function.comp RCLike.ofReal (Function.comp ⇑f fun i  …
    i : n
    hx : Membership.mem (spectrum Real A) (hA.eigenvalues i)
    this : ∀ (i : n), Eq (Function.comp RCLike.ofReal (Function.comp ⇑f fun i => ⟨ …
    ⊢ Eq (f ⟨hA.eigenvalues i, hx⟩) 0
  -/
  refine RCLike.ofReal_eq_zero.mp (this i)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_cfcAux := isClosedEmbedding_cfcAux


lemma cfcAux_id : hA.cfcAux (.restrict (spectrum ℝ A) (.id ℝ)) = A := by
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    ⊢ Eq (hA.cfcAux (ContinuousMap.restrict (spectrum Real A) (ContinuousMap.id Re …
  -/
  conv_rhs => rw [hA.spectral_theorem]
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    ⊢ Eq (hA.cfcAux (ContinuousMap.restrict (spectrum Real A) (ContinuousMap.id Re …
  -/
  congr!
  /-
    🎉 no goals
  -/


/-- Instance of the continuous functional calculus for a Hermitian matrix over `𝕜` with
`RCLike 𝕜`. -/
instance instContinuousFunctionalCalculus :
    ContinuousFunctionalCalculus ℝ (IsSelfAdjoint : Matrix n n 𝕜 → Prop) where
  exists_cfc_of_predicate a ha := by
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      a : Matrix n n 𝕜
      ha : IsSelfAdjoint a
      ⊢ Exists fun φ => And (Topology.IsClosedEmbedding ⇑φ) (And (Eq (φ (ContinuousM …
    -/
    replace ha : IsHermitian a := ha
    refine ⟨ha.cfcAux, ha.isClosedEmbedding_cfcAux, ha.cfcAux_id, fun f ↦ ?map_spec,
      fun f ↦ ?hermitian⟩
    case map_spec =>
      apply Set.eq_of_subset_of_subset
      · rw [← ContinuousMap.spectrum_eq_range f]
        apply AlgHom.spectrum_apply_subset
      · rw [cfcAux_apply, unitary.spectrum.unitary_conjugate]
        rintro - ⟨x , rfl⟩
        apply spectrum.of_algebraMap_mem 𝕜
        simp only [Function.comp_apply, Set.mem_range, spectrum_diagonal]
        obtain ⟨x, hx⟩ := x
        obtain ⟨i, rfl⟩ := ha.eigenvalues_eq_spectrum_real ▸ hx
        exact ⟨i, rfl⟩
    case hermitian =>
      simp only [isSelfAdjoint_iff, cfcAux_apply, mul_assoc, star_mul, star_star]
      rw [star_eq_conjTranspose, diagonal_conjTranspose]
      congr!
      simp [Pi.star_def, Function.comp_def]
    /-
      n : Type u_1
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      A : Matrix n n 𝕜
      hA : A.IsHermitian
      inst✝ : Nontrivial (Matrix n n 𝕜)
      a : Matrix n n 𝕜
      ha : IsSelfAdjoint a
      ⊢ (spectrum Real a).Nonempty
    -/
  spectrum_nonempty a ha := by
      /-
        case inl
        n : Type u_1
        𝕜 : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        A : Matrix n n 𝕜
        hA : A.IsHermitian
        inst✝ : Nontrivial (Matrix n n 𝕜)
        a : Matrix n n 𝕜
        ha : IsSelfAdjoint a
        h : IsEmpty n
        ⊢ (spectrum Real a).Nonempty
      -/
    obtain (h | h) := isEmpty_or_nonempty n
      /-
        case inl.intro.intro
        n : Type u_1
        𝕜 : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        A : Matrix n n 𝕜
        hA : A.IsHermitian
        inst✝ : Nontrivial (Matrix n n 𝕜)
        a : Matrix n n 𝕜
        ha : IsSelfAdjoint a
        h : IsEmpty n
        x y : Matrix n n 𝕜
        hxy : Ne x y
        ⊢ (spectrum Real a).Nonempty
      -/
    · obtain ⟨x, y, hxy⟩ := exists_pair_ne (Matrix n n 𝕜)
      /-
        🎉 no goals
      -/
      /-
        case inr
        n : Type u_1
        𝕜 : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : Fintype n
        inst✝¹ : DecidableEq n
        A : Matrix n n 𝕜
        hA : A.IsHermitian
        inst✝ : Nontrivial (Matrix n n 𝕜)
        a : Matrix n n 𝕜
        ha : IsSelfAdjoint a
        h : Nonempty n
        ⊢ (spectrum Real a).Nonempty
      -/
      exact False.elim <| Matrix.of.symm.injective.ne hxy <| Subsingleton.elim _ _
      /-
        🎉 no goals
      -/
    · exact eigenvalues_eq_spectrum_real ha ▸ Set.range_nonempty _
  predicate_zero := .zero _


instance instUniqueContinuousFunctionalCalculus :
    UniqueContinuousFunctionalCalculus ℝ (Matrix n n 𝕜) :=
  let _ : NormedRing (Matrix n n 𝕜) := Matrix.linftyOpNormedRing
  let _ : NormedAlgebra ℝ (Matrix n n 𝕜) := Matrix.linftyOpNormedAlgebra
  inferInstance


/-- The continuous functional calculus of a Hermitian matrix as a triple product using the
spectral theorem. Note that this actually operates on bare functions since every function is
continuous on the spectrum of a matrix, since the spectrum is finite. This is shown to be equal to
the generic continuous functional calculus API in `Matrix.IsHermitian.cfc_eq`. In general, users
should prefer the generic API, especially because it will make rewriting easier. -/
protected noncomputable def cfc (f : ℝ → ℝ) : Matrix n n 𝕜 :=
  (eigenvectorUnitary hA : Matrix n n 𝕜) * diagonal (RCLike.ofReal ∘ f ∘ hA.eigenvalues)
    * star (eigenvectorUnitary hA : Matrix n n 𝕜)


lemma cfc_eq (f : ℝ → ℝ) : cfc f A = hA.cfc f := by
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    f : Real → Real
    ⊢ Eq (cfc f A) (hA.cfc f)
  -/
  have hA' : IsSelfAdjoint A := hA
  have := cfcHom_eq_of_continuous_of_map_id hA' hA.cfcAux hA.isClosedEmbedding_cfcAux.continuous
    hA.cfcAux_id
  /-
    n : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n 𝕜
    hA : A.IsHermitian
    f : Real → Real
    hA' : IsSelfAdjoint A
    this : Eq (cfcHom hA') hA.cfcAux
    ⊢ Eq (cfc f A) (hA.cfc f)
  -/
  rw [cfc_apply f A hA' (by rw [continuousOn_iff_continuous_restrict]; fun_prop), this]
  simp only [cfcAux_apply, ContinuousMap.coe_mk, Function.comp_def, Set.restrict_apply,
    IsHermitian.cfc]


