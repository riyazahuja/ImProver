/-- Basis vectors are eigenvectors of associated diagonal linear operator. -/
lemma hasEigenvector_toLin_diagonal (d : n → R) (i : n) (b : Basis n R M) :
    HasEigenvector (toLin b b (diagonal d)) (d i) (b i) :=
                                /-
                                  R : Type u_1
                                  n : Type u_2
                                  M : Type u_3
                                  inst✝⁵ : DecidableEq n
                                  inst✝⁴ : Fintype n
                                  inst✝³ : CommRing R
                                  inst✝² : Nontrivial R
                                  inst✝¹ : AddCommGroup M
                                  inst✝ : Module R M
                                  d : n → R
                                  i : n
                                  b : Basis n R M
                                  ⊢ Eq (((Matrix.toLin b b) (Matrix.diagonal d)) (b i)) (HSMul.hSMul (d i) (b i))
                                -/
  ⟨mem_eigenspace_iff.mpr <| by simp [diagonal], Basis.ne_zero b i⟩
                                /-
                                  🎉 no goals
                                -/


/--  Standard basis vectors are eigenvectors of any associated diagonal linear operator. -/
lemma hasEigenvector_toLin'_diagonal (d : n → R) (i : n) :
    HasEigenvector (toLin' (diagonal d)) (d i) (Pi.basisFun R n i)  :=
  hasEigenvector_toLin_diagonal _ _ (Pi.basisFun R n)


/-- Eigenvalues of a diagonal linear operator are the diagonal entries. -/
lemma hasEigenvalue_toLin_diagonal_iff (d : n → R) {μ : R} [NoZeroSMulDivisors R M]
    (b : Basis n R M) : HasEigenvalue (toLin b b (diagonal d)) μ ↔ ∃ i, d i = μ := by
  have (i : n) : HasEigenvalue (toLin b b (diagonal d)) (d i) :=
    hasEigenvalue_of_hasEigenvector <| hasEigenvector_toLin_diagonal d i b
  /-
    R : Type u_1
    n : Type u_2
    M : Type u_3
    inst✝⁶ : DecidableEq n
    inst✝⁵ : Fintype n
    inst✝⁴ : CommRing R
    inst✝³ : Nontrivial R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    d : n → R
    μ : R
    inst✝ : NoZeroSMulDivisors R M
    b : Basis n R M
    this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
    ⊢ Iff (Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) μ) (E …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
      ⊢ Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) μ → Exists …
    -/
  · contrapose!
    /-
      case mp
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
      ⊢ (∀ (i : n), Ne (d i) μ) → Not (Module.End.HasEigenvalue ((Matrix.toLin b b)  …
    -/
    intro hμ h_eig
    have h_iSup : ⨆ μ ∈ Set.range d, eigenspace (toLin b b (diagonal d)) μ = ⊤ := by
      rw [eq_top_iff, ← b.span_eq, Submodule.span_le]
      rintro - ⟨i, rfl⟩
      simp only [SetLike.mem_coe]
      apply Submodule.mem_iSup_of_mem (d i)
      apply Submodule.mem_iSup_of_mem ⟨i, rfl⟩
      rw [mem_eigenspace_iff]
      exact (hasEigenvector_toLin_diagonal d i b).apply_eq_smul
    /-
      case mp
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
      hμ : ∀ (i : n), Ne (d i) μ
      h_eig : Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) μ
      h_iSup : Eq (iSup fun μ => iSup fun h => Module.End.eigenspace ((Matrix.toLin  …
      ⊢ False
    -/
    have hμ_not_mem : μ ∉ Set.range d := by simpa using fun i ↦ (hμ i)
    /-
      case mp
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
      hμ : ∀ (i : n), Ne (d i) μ
      h_eig : Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) μ
      h_iSup : Eq (iSup fun μ => iSup fun h => Module.End.eigenspace ((Matrix.toLin  …
      hμ_not_mem : Not (Membership.mem (Set.range d) μ)
      ⊢ False
    -/
    have := eigenspaces_iSupIndep (toLin b b (diagonal d)) |>.disjoint_biSup hμ_not_mem
    /-
      case mp
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this✝ : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagon …
      hμ : ∀ (i : n), Ne (d i) μ
      h_eig : Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) μ
      h_iSup : Eq (iSup fun μ => iSup fun h => Module.End.eigenspace ((Matrix.toLin  …
      hμ_not_mem : Not (Membership.mem (Set.range d) μ)
      this : Disjoint (Module.End.eigenspace ((Matrix.toLin b b) (Matrix.diagonal d) …
      ⊢ False
    -/
    rw [h_iSup, disjoint_top] at this
    /-
      case mp
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this✝ : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagon …
      hμ : ∀ (i : n), Ne (d i) μ
      h_eig : Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) μ
      h_iSup : Eq (iSup fun μ => iSup fun h => Module.End.eigenspace ((Matrix.toLin  …
      hμ_not_mem : Not (Membership.mem (Set.range d) μ)
      this : Eq (Module.End.eigenspace ((Matrix.toLin b b) (Matrix.diagonal d)) μ) B …
      ⊢ False
    -/
    exact h_eig this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      μ : R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
      ⊢ (Exists fun i => Eq (d i) μ) → Module.End.HasEigenvalue ((Matrix.toLin b b)  …
    -/
  · rintro ⟨i, rfl⟩
    /-
      case mpr.intro
      R : Type u_1
      n : Type u_2
      M : Type u_3
      inst✝⁶ : DecidableEq n
      inst✝⁵ : Fintype n
      inst✝⁴ : CommRing R
      inst✝³ : Nontrivial R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      d : n → R
      inst✝ : NoZeroSMulDivisors R M
      b : Basis n R M
      this : ∀ (i : n), Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagona …
      i : n
      ⊢ Module.End.HasEigenvalue ((Matrix.toLin b b) (Matrix.diagonal d)) (d i)
    -/
    exact this i
    /-
      🎉 no goals
    -/


/-- Eigenvalues of a diagonal linear operator with respect to standard basis
    are the diagonal entries. -/
lemma hasEigenvalue_toLin'_diagonal_iff [NoZeroDivisors R] (d : n → R) {μ : R} :
    HasEigenvalue (toLin' (diagonal d)) μ ↔ (∃ i, d i = μ) :=
  hasEigenvalue_toLin_diagonal_iff _ <| Pi.basisFun R n


/-- The spectrum of the diagonal operator is the range of the diagonal viewed as a function. -/
lemma spectrum_diagonal [Field R] (d : n → R) :
    spectrum R (diagonal d) = Set.range d := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Field R
    d : n → R
    ⊢ Eq (spectrum R (Matrix.diagonal d)) (Set.range d)
  -/
  ext μ
  /-
    case h
    R : Type u_1
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Field R
    d : n → R
    μ : R
    ⊢ Iff (Membership.mem (spectrum R (Matrix.diagonal d)) μ) (Membership.mem (Set …
  -/
  rw [← AlgEquiv.spectrum_eq (toLinAlgEquiv <| Pi.basisFun R n), ← hasEigenvalue_iff_mem_spectrum]
  /-
    case h
    R : Type u_1
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Field R
    d : n → R
    μ : R
    ⊢ Iff (Module.End.HasEigenvalue ((Matrix.toLinAlgEquiv (Pi.basisFun R n)) (Mat …
  -/
  exact hasEigenvalue_toLin'_diagonal_iff d
  /-
    🎉 no goals
  -/


