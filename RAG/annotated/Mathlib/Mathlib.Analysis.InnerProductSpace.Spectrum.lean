local notation "⟪" x ", " y "⟫" => @inner 𝕜 E _ x y


/-- A self-adjoint operator preserves orthogonal complements of its eigenspaces. -/
theorem invariant_orthogonalComplement_eigenspace (hT : T.IsSymmetric) (μ : 𝕜)
    (v : E) (hv : v ∈ (eigenspace T μ)ᗮ) : T v ∈ (eigenspace T μ)ᗮ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    v : E
    hv : Membership.mem (Module.End.eigenspace T μ).orthogonal v
    ⊢ Membership.mem (Module.End.eigenspace T μ).orthogonal (T v)
  -/
  intro w hw
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    v : E
    hv : Membership.mem (Module.End.eigenspace T μ).orthogonal v
    w : E
    hw : Membership.mem (Module.End.eigenspace T μ) w
    ⊢ Eq (Inner.inner w (T v)) 0
  -/
  have : T w = (μ : 𝕜) • w := by rwa [mem_eigenspace_iff] at hw
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    v : E
    hv : Membership.mem (Module.End.eigenspace T μ).orthogonal v
    w : E
    hw : Membership.mem (Module.End.eigenspace T μ) w
    this : Eq (T w) (HSMul.hSMul μ w)
    ⊢ Eq (Inner.inner w (T v)) 0
  -/
  simp [← hT w, this, inner_smul_left, hv w hw]
  /-
    🎉 no goals
  -/


/-- The eigenvalues of a self-adjoint operator are real. -/
theorem conj_eigenvalue_eq_self (hT : T.IsSymmetric) {μ : 𝕜} (hμ : HasEigenvalue T μ) :
    conj μ = μ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    hμ : Module.End.HasEigenvalue T μ
    ⊢ Eq ((starRingEnd 𝕜) μ) μ
  -/
  obtain ⟨v, hv₁, hv₂⟩ := hμ.exists_hasEigenvector
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    hμ : Module.End.HasEigenvalue T μ
    v : E
    hv₁ : Membership.mem ((Module.End.genEigenspace T μ) 1) v
    hv₂ : Ne v 0
    ⊢ Eq ((starRingEnd 𝕜) μ) μ
  -/
  rw [mem_eigenspace_iff] at hv₁
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    hμ : Module.End.HasEigenvalue T μ
    v : E
    hv₁ : Eq (T v) (HSMul.hSMul μ v)
    hv₂ : Ne v 0
    ⊢ Eq ((starRingEnd 𝕜) μ) μ
  -/
  simpa [hv₂, inner_smul_left, inner_smul_right, hv₁] using hT v v
  /-
    🎉 no goals
  -/


/-- The eigenspaces of a self-adjoint operator are mutually orthogonal. -/
theorem orthogonalFamily_eigenspaces (hT : T.IsSymmetric) :
    OrthogonalFamily 𝕜 (fun μ => eigenspace T μ) fun μ => (eigenspace T μ).subtypeₗᵢ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    ⊢ OrthogonalFamily 𝕜 (fun μ => Subtype fun x => Membership.mem (Module.End.eig …
  -/
  rintro μ ν hμν ⟨v, hv⟩ ⟨w, hw⟩
  /-
    case mk.mk
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ ν : 𝕜
    hμν : Ne μ ν
    v : E
    hv : Membership.mem (Module.End.eigenspace T μ) v
    w : E
    hw : Membership.mem (Module.End.eigenspace T ν) w
    ⊢ Eq (Inner.inner (((fun μ => (Module.End.eigenspace T μ).subtypeₗᵢ) μ) ⟨v, hv …
  -/
  by_cases hv' : v = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : LinearMap (RingHom.id 𝕜) E E
      hT : T.IsSymmetric
      μ ν : 𝕜
      hμν : Ne μ ν
      v : E
      hv : Membership.mem (Module.End.eigenspace T μ) v
      w : E
      hw : Membership.mem (Module.End.eigenspace T ν) w
      hv' : Eq v 0
      ⊢ Eq (Inner.inner (((fun μ => (Module.End.eigenspace T μ).subtypeₗᵢ) μ) ⟨v, hv …
    -/
  · simp [hv']
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ ν : 𝕜
    hμν : Ne μ ν
    v : E
    hv : Membership.mem (Module.End.eigenspace T μ) v
    w : E
    hw : Membership.mem (Module.End.eigenspace T ν) w
    hv' : Not (Eq v 0)
    ⊢ Eq (Inner.inner (((fun μ => (Module.End.eigenspace T μ).subtypeₗᵢ) μ) ⟨v, hv …
  -/
  have H := hT.conj_eigenvalue_eq_self (hasEigenvalue_of_hasEigenvector ⟨hv, hv'⟩)
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ ν : 𝕜
    hμν : Ne μ ν
    v : E
    hv : Membership.mem (Module.End.eigenspace T μ) v
    w : E
    hw : Membership.mem (Module.End.eigenspace T ν) w
    hv' : Not (Eq v 0)
    H : Eq ((starRingEnd 𝕜) μ) μ
    ⊢ Eq (Inner.inner (((fun μ => (Module.End.eigenspace T μ).subtypeₗᵢ) μ) ⟨v, hv …
  -/
  rw [mem_eigenspace_iff] at hv hw
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ ν : 𝕜
    hμν : Ne μ ν
    v : E
    hv✝ : Membership.mem (Module.End.eigenspace T μ) v
    hv : Eq (T v) (HSMul.hSMul μ v)
    w : E
    hw✝ : Membership.mem (Module.End.eigenspace T ν) w
    hw : Eq (T w) (HSMul.hSMul ν w)
    hv' : Not (Eq v 0)
    H : Eq ((starRingEnd 𝕜) μ) μ
    ⊢ Eq (Inner.inner (((fun μ => (Module.End.eigenspace T μ).subtypeₗᵢ) μ) ⟨v, hv …
  -/
  refine Or.resolve_left ?_ hμν.symm
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ ν : 𝕜
    hμν : Ne μ ν
    v : E
    hv✝ : Membership.mem (Module.End.eigenspace T μ) v
    hv : Eq (T v) (HSMul.hSMul μ v)
    w : E
    hw✝ : Membership.mem (Module.End.eigenspace T ν) w
    hw : Eq (T w) (HSMul.hSMul ν w)
    hv' : Not (Eq v 0)
    H : Eq ((starRingEnd 𝕜) μ) μ
    ⊢ Or (Eq ν μ) (Eq (Inner.inner (((fun μ => (Module.End.eigenspace T μ).subtype …
  -/
  simpa [inner_smul_left, inner_smul_right, hv, hw, H] using (hT v w).symm
  /-
    🎉 no goals
  -/


theorem orthogonalFamily_eigenspaces' (hT : T.IsSymmetric) :
    OrthogonalFamily 𝕜 (fun μ : Eigenvalues T => eigenspace T μ) fun μ =>
      (eigenspace T μ).subtypeₗᵢ :=
  hT.orthogonalFamily_eigenspaces.comp Subtype.coe_injective


/-- The mutual orthogonal complement of the eigenspaces of a self-adjoint operator on an inner
product space is an invariant subspace of the operator. -/
theorem orthogonalComplement_iSup_eigenspaces_invariant (hT : T.IsSymmetric)
    ⦃v : E⦄ (hv : v ∈ (⨆ μ, eigenspace T μ)ᗮ) : T v ∈ (⨆ μ, eigenspace T μ)ᗮ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    v : E
    hv : Membership.mem (iSup fun μ => Module.End.eigenspace T μ).orthogonal v
    ⊢ Membership.mem (iSup fun μ => Module.End.eigenspace T μ).orthogonal (T v)
  -/
  rw [← Submodule.iInf_orthogonal] at hv ⊢
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    v : E
    hv : Membership.mem (iInf fun i => (Module.End.eigenspace T i).orthogonal) v
    ⊢ Membership.mem (iInf fun i => (Module.End.eigenspace T i).orthogonal) (T v)
  -/
  exact T.iInf_invariant hT.invariant_orthogonalComplement_eigenspace v hv
  /-
    🎉 no goals
  -/


/-- The mutual orthogonal complement of the eigenspaces of a self-adjoint operator on an inner
product space has no eigenvalues. -/
theorem orthogonalComplement_iSup_eigenspaces (hT : T.IsSymmetric) (μ : 𝕜) :
    eigenspace (T.restrict hT.orthogonalComplement_iSup_eigenspaces_invariant) μ = ⊥ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    ⊢ Eq (Module.End.eigenspace (T.restrict ⋯) μ) Bot.bot
  -/
  set p : Submodule 𝕜 E := (⨆ μ, eigenspace T μ)ᗮ
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    p : Submodule 𝕜 E := (iSup fun μ => Module.End.eigenspace T μ).orthogonal
    ⊢ Eq (Module.End.eigenspace (T.restrict ⋯) μ) Bot.bot
  -/
  refine eigenspace_restrict_eq_bot hT.orthogonalComplement_iSup_eigenspaces_invariant ?_
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    p : Submodule 𝕜 E := (iSup fun μ => Module.End.eigenspace T μ).orthogonal
    ⊢ Disjoint (Module.End.eigenspace T μ) (iSup fun μ => Module.End.eigenspace T  …
  -/
  have H₂ : eigenspace T μ ⟂ p := (Submodule.isOrtho_orthogonal_right _).mono_left (le_iSup _ _)
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    μ : 𝕜
    p : Submodule 𝕜 E := (iSup fun μ => Module.End.eigenspace T μ).orthogonal
    H₂ : (Module.End.eigenspace T μ).IsOrtho p
    ⊢ Disjoint (Module.End.eigenspace T μ) (iSup fun μ => Module.End.eigenspace T  …
  -/
  exact H₂.disjoint
  /-
    🎉 no goals
  -/


/-- The mutual orthogonal complement of the eigenspaces of a self-adjoint operator on a
finite-dimensional inner product space is trivial. -/
theorem orthogonalComplement_iSup_eigenspaces_eq_bot (hT : T.IsSymmetric) :
    (⨆ μ, eigenspace T μ)ᗮ = ⊥ := by
  have hT' : IsSymmetric _ :=
    hT.restrict_invariant hT.orthogonalComplement_iSup_eigenspaces_invariant
  -- a self-adjoint operator on a nontrivial inner product space has an eigenvalue
  haveI :=
    hT'.subsingleton_of_no_eigenvalue_finiteDimensional hT.orthogonalComplement_iSup_eigenspaces
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    hT' : (T.restrict ⋯).IsSymmetric
    this : Subsingleton (Subtype fun x => Membership.mem (iSup fun μ => Module.End …
    ⊢ Eq (iSup fun μ => Module.End.eigenspace T μ).orthogonal Bot.bot
  -/
  exact Submodule.eq_bot_of_subsingleton
  /-
    🎉 no goals
  -/


theorem orthogonalComplement_iSup_eigenspaces_eq_bot' (hT : T.IsSymmetric) :
    (⨆ μ : Eigenvalues T, eigenspace T μ)ᗮ = ⊥ :=
  show (⨆ μ : { μ // eigenspace T μ ≠ ⊥ }, eigenspace T μ)ᗮ = ⊥ by
    /-
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      T : LinearMap (RingHom.id 𝕜) E E
      inst✝ : FiniteDimensional 𝕜 E
      hT : T.IsSymmetric
      ⊢ Eq (iSup fun μ => Module.End.eigenspace T ↑μ).orthogonal Bot.bot
    -/
    rw [iSup_ne_bot_subtype, hT.orthogonalComplement_iSup_eigenspaces_eq_bot]
    /-
      🎉 no goals
    -/


/-- The eigenspaces of a self-adjoint operator on a finite-dimensional inner product space `E` gives
an internal direct sum decomposition of `E`.

Note this takes `hT` as a `Fact` to allow it to be an instance. -/
noncomputable instance directSumDecomposition [hT : Fact T.IsSymmetric] :
    DirectSum.Decomposition fun μ : Eigenvalues T => eigenspace T μ :=
                                                                               /-
                                                                                 𝕜 : Type u_1
                                                                                 inst✝³ : RCLike 𝕜
                                                                                 E : Type u_2
                                                                                 inst✝² : NormedAddCommGroup E
                                                                                 inst✝¹ : InnerProductSpace 𝕜 E
                                                                                 T : LinearMap (RingHom.id 𝕜) E E
                                                                                 inst✝ : FiniteDimensional 𝕜 E
                                                                                 hT : Fact T.IsSymmetric
                                                                                 μ : Module.End.Eigenvalues T
                                                                                 ⊢ CompleteSpace (Subtype fun x => Membership.mem (Module.End.eigenspace T (↑T  …
                                                                               -/
  haveI h : ∀ μ : Eigenvalues T, CompleteSpace (eigenspace T μ) := fun μ => by infer_instance
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  hT.out.orthogonalFamily_eigenspaces'.decomposition
    (Submodule.orthogonal_eq_bot_iff.mp hT.out.orthogonalComplement_iSup_eigenspaces_eq_bot')


theorem directSum_decompose_apply [_hT : Fact T.IsSymmetric] (x : E) (μ : Eigenvalues T) :
    DirectSum.decompose (fun μ : Eigenvalues T => eigenspace T μ) x μ =
      orthogonalProjection (eigenspace T μ) x :=
  rfl


/-- The eigenspaces of a self-adjoint operator on a finite-dimensional inner product space `E` gives
an internal direct sum decomposition of `E`. -/
theorem direct_sum_isInternal (hT : T.IsSymmetric) :
    DirectSum.IsInternal fun μ : Eigenvalues T => eigenspace T μ :=
  hT.orthogonalFamily_eigenspaces'.isInternal_iff.mpr
    hT.orthogonalComplement_iSup_eigenspaces_eq_bot'


/-- Isometry from an inner product space `E` to the direct sum of the eigenspaces of some
self-adjoint operator `T` on `E`. -/
noncomputable def diagonalization : E ≃ₗᵢ[𝕜] PiLp 2 fun μ : Eigenvalues T => eigenspace T μ :=
  hT.direct_sum_isInternal.isometryL2OfOrthogonalFamily hT.orthogonalFamily_eigenspaces'


@[simp]
theorem diagonalization_symm_apply (w : PiLp 2 fun μ : Eigenvalues T => eigenspace T μ) :
    hT.diagonalization.symm w = ∑ μ, w μ :=
  hT.direct_sum_isInternal.isometryL2OfOrthogonalFamily_symm_apply
    hT.orthogonalFamily_eigenspaces' w


/-- *Diagonalization theorem*, *spectral theorem*; version 1: A self-adjoint operator `T` on a
finite-dimensional inner product space `E` acts diagonally on the decomposition of `E` into the
direct sum of the eigenspaces of `T`. -/
theorem diagonalization_apply_self_apply (v : E) (μ : Eigenvalues T) :
    hT.diagonalization (T v) μ = (μ : 𝕜) • hT.diagonalization v μ := by
  suffices
    ∀ w : PiLp 2 fun μ : Eigenvalues T => eigenspace T μ,
      T (hT.diagonalization.symm w) = hT.diagonalization.symm fun μ => (μ : 𝕜) • w μ by
    simpa only [LinearIsometryEquiv.symm_apply_apply, LinearIsometryEquiv.apply_symm_apply] using
      congr_arg (fun w => hT.diagonalization w μ) (this (hT.diagonalization v))
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    v : E
    μ : Module.End.Eigenvalues T
    ⊢ ∀ (w : PiLp 2 fun μ => Subtype fun x => Membership.mem (Module.End.eigenspac …
  -/
  intro w
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    v : E
    μ : Module.End.Eigenvalues T
    w : PiLp 2 fun μ => Subtype fun x => Membership.mem (Module.End.eigenspace T ( …
    ⊢ Eq (T (hT.diagonalization.symm w)) (hT.diagonalization.symm fun μ => HSMul.h …
  -/
  have hwT : ∀ μ, T (w μ) = (μ : 𝕜) • w μ := fun μ => mem_eigenspace_iff.1 (w μ).2
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    v : E
    μ : Module.End.Eigenvalues T
    w : PiLp 2 fun μ => Subtype fun x => Membership.mem (Module.End.eigenspace T ( …
    hwT : ∀ (μ : Module.End.Eigenvalues T), Eq (T ↑(w μ)) (HSMul.hSMul (↑T 1 μ) ↑( …
    ⊢ Eq (T (hT.diagonalization.symm w)) (hT.diagonalization.symm fun μ => HSMul.h …
  -/
  simp only [hwT, diagonalization_symm_apply, map_sum, Submodule.coe_smul_of_tower]
  /-
    🎉 no goals
  -/


/-- A choice of orthonormal basis of eigenvectors for self-adjoint operator `T` on a
finite-dimensional inner product space `E`.

TODO Postcompose with a permutation so that these eigenvectors are listed in increasing order of
eigenvalue. -/
noncomputable irreducible_def eigenvectorBasis : OrthonormalBasis (Fin n) 𝕜 E :=
  hT.direct_sum_isInternal.subordinateOrthonormalBasis hn hT.orthogonalFamily_eigenspaces'


/-- The sequence of real eigenvalues associated to the standard orthonormal basis of eigenvectors
for a self-adjoint operator `T` on `E`.

TODO Postcompose with a permutation so that these eigenvalues are listed in increasing order. -/
noncomputable irreducible_def eigenvalues (i : Fin n) : ℝ :=
  @RCLike.re 𝕜 _ <| (hT.direct_sum_isInternal.subordinateOrthonormalBasisIndex hn i
    hT.orthogonalFamily_eigenspaces').val


theorem hasEigenvector_eigenvectorBasis (i : Fin n) :
    HasEigenvector T (hT.eigenvalues hn i) (hT.eigenvectorBasis hn i) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    i : Fin n
    ⊢ Module.End.HasEigenvector T (↑(hT.eigenvalues hn i)) ((hT.eigenvectorBasis h …
  -/
  let v : E := hT.eigenvectorBasis hn i
  let μ : 𝕜 :=
    (hT.direct_sum_isInternal.subordinateOrthonormalBasisIndex hn i
      hT.orthogonalFamily_eigenspaces').val
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    i : Fin n
    v : E := (hT.eigenvectorBasis hn) i
    μ : 𝕜 := ↑T (DirectSum.IsInternal.subordinateOrthonormalBasisIndex hn ⋯ i ⋯)
    ⊢ Module.End.HasEigenvector T (↑(hT.eigenvalues hn i)) ((hT.eigenvectorBasis h …
  -/
  simp_rw [eigenvalues]
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    i : Fin n
    v : E := (hT.eigenvectorBasis hn) i
    μ : 𝕜 := ↑T (DirectSum.IsInternal.subordinateOrthonormalBasisIndex hn ⋯ i ⋯)
    ⊢ Module.End.HasEigenvector T (↑(RCLike.re (↑T (DirectSum.IsInternal.subordina …
  -/
  change HasEigenvector T (RCLike.re μ) v
  have key : HasEigenvector T μ v := by
    have H₁ : v ∈ eigenspace T μ := by
      simp_rw [v, eigenvectorBasis]
      exact
        hT.direct_sum_isInternal.subordinateOrthonormalBasis_subordinate hn i
          hT.orthogonalFamily_eigenspaces'
    have H₂ : v ≠ 0 := by simpa using (hT.eigenvectorBasis hn).toBasis.ne_zero i
    exact ⟨H₁, H₂⟩
  have re_μ : ↑(RCLike.re μ) = μ := by
    rw [← RCLike.conj_eq_iff_re]
    exact hT.conj_eigenvalue_eq_self (hasEigenvalue_of_hasEigenvector key)
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    i : Fin n
    v : E := (hT.eigenvectorBasis hn) i
    μ : 𝕜 := ↑T (DirectSum.IsInternal.subordinateOrthonormalBasisIndex hn ⋯ i ⋯)
    key : Module.End.HasEigenvector T μ v
    re_μ : Eq (↑(RCLike.re μ)) μ
    ⊢ Module.End.HasEigenvector T (↑(RCLike.re μ)) v
  -/
  simpa [re_μ] using key
  /-
    🎉 no goals
  -/


theorem hasEigenvalue_eigenvalues (i : Fin n) : HasEigenvalue T (hT.eigenvalues hn i) :=
  Module.End.hasEigenvalue_of_hasEigenvector (hT.hasEigenvector_eigenvectorBasis hn i)


@[simp]
theorem apply_eigenvectorBasis (i : Fin n) :
    T (hT.eigenvectorBasis hn i) = (hT.eigenvalues hn i : 𝕜) • hT.eigenvectorBasis hn i :=
  mem_eigenspace_iff.mp (hT.hasEigenvector_eigenvectorBasis hn i).1


/-- *Diagonalization theorem*, *spectral theorem*; version 2: A self-adjoint operator `T` on a
finite-dimensional inner product space `E` acts diagonally on the identification of `E` with
Euclidean space induced by an orthonormal basis of eigenvectors of `T`. -/
theorem eigenvectorBasis_apply_self_apply (v : E) (i : Fin n) :
    (hT.eigenvectorBasis hn).repr (T v) i =
      hT.eigenvalues hn i * (hT.eigenvectorBasis hn).repr v i := by
  suffices
    ∀ w : EuclideanSpace 𝕜 (Fin n),
      T ((hT.eigenvectorBasis hn).repr.symm w) =
        (hT.eigenvectorBasis hn).repr.symm fun i => hT.eigenvalues hn i * w i by
    simpa [OrthonormalBasis.sum_repr_symm] using
      congr_arg (fun v => (hT.eigenvectorBasis hn).repr v i)
        (this ((hT.eigenvectorBasis hn).repr v))
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    v : E
    i : Fin n
    ⊢ ∀ (w : EuclideanSpace 𝕜 (Fin n)), Eq (T ((hT.eigenvectorBasis hn).repr.symm  …
  -/
  intro w
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    v : E
    i : Fin n
    w : EuclideanSpace 𝕜 (Fin n)
    ⊢ Eq (T ((hT.eigenvectorBasis hn).repr.symm w)) ((hT.eigenvectorBasis hn).repr …
  -/
  simp_rw [← OrthonormalBasis.sum_repr_symm, map_sum, map_smul, apply_eigenvectorBasis]
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    v : E
    i : Fin n
    w : EuclideanSpace 𝕜 (Fin n)
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (w x) (HSMul.hSMul (↑(hT.eigenvalue …
  -/
  apply Fintype.sum_congr
  /-
    case h
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    v : E
    i : Fin n
    w : EuclideanSpace 𝕜 (Fin n)
    ⊢ ∀ (a : Fin n), Eq (HSMul.hSMul (w a) (HSMul.hSMul (↑(hT.eigenvalues hn a)) ( …
  -/
  intro a
  /-
    case h
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    inst✝ : FiniteDimensional 𝕜 E
    hT : T.IsSymmetric
    n : Nat
    hn : Eq (Module.finrank 𝕜 E) n
    v : E
    i : Fin n
    w : EuclideanSpace 𝕜 (Fin n)
    a : Fin n
    ⊢ Eq (HSMul.hSMul (w a) (HSMul.hSMul (↑(hT.eigenvalues hn a)) ((hT.eigenvector …
  -/
  rw [smul_smul, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_product_apply_eigenvector {μ : 𝕜} {v : E} {T : E →ₗ[𝕜] E}
    (h : v ∈ Module.End.eigenspace T μ) : ⟪v, T v⟫ = μ * (‖v‖ : 𝕜) ^ 2 := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : 𝕜
    v : E
    T : LinearMap (RingHom.id 𝕜) E E
    h : Membership.mem (Module.End.eigenspace T μ) v
    ⊢ Eq (Inner.inner v (T v)) (HMul.hMul μ (HPow.hPow (↑(Norm.norm v)) 2))
  -/
  simp only [mem_eigenspace_iff.mp h, inner_smul_right, inner_self_eq_norm_sq_to_K]
  /-
    🎉 no goals
  -/


theorem eigenvalue_nonneg_of_nonneg {μ : ℝ} {T : E →ₗ[𝕜] E} (hμ : HasEigenvalue T μ)
    (hnn : ∀ x : E, 0 ≤ RCLike.re ⟪x, T x⟫) : 0 ≤ μ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : Real
    T : LinearMap (RingHom.id 𝕜) E E
    hμ : Module.End.HasEigenvalue T ↑μ
    hnn : ∀ (x : E), LE.le 0 (RCLike.re (Inner.inner x (T x)))
    ⊢ LE.le 0 μ
  -/
  obtain ⟨v, hv⟩ := hμ.exists_hasEigenvector
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : Real
    T : LinearMap (RingHom.id 𝕜) E E
    hμ : Module.End.HasEigenvalue T ↑μ
    hnn : ∀ (x : E), LE.le 0 (RCLike.re (Inner.inner x (T x)))
    v : E
    hv : Module.End.HasEigenvector T (↑μ) v
    ⊢ LE.le 0 μ
  -/
  have hpos : (0 : ℝ) < ‖v‖ ^ 2 := by simpa only [sq_pos_iff, norm_ne_zero_iff] using hv.2
  have : RCLike.re ⟪v, T v⟫ = μ * ‖v‖ ^ 2 := by
    have := congr_arg RCLike.re (inner_product_apply_eigenvector hv.1)
    -- Porting note: why can't `exact_mod_cast` do this? These lemmas are marked `norm_cast`
    rw [← RCLike.ofReal_pow, ← RCLike.ofReal_mul] at this
    exact mod_cast this
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : Real
    T : LinearMap (RingHom.id 𝕜) E E
    hμ : Module.End.HasEigenvalue T ↑μ
    hnn : ∀ (x : E), LE.le 0 (RCLike.re (Inner.inner x (T x)))
    v : E
    hv : Module.End.HasEigenvector T (↑μ) v
    hpos : LT.lt 0 (HPow.hPow (Norm.norm v) 2)
    this : Eq (RCLike.re (Inner.inner v (T v))) (HMul.hMul μ (HPow.hPow (Norm.norm …
    ⊢ LE.le 0 μ
  -/
  exact (mul_nonneg_iff_of_pos_right hpos).mp (this ▸ hnn v)
  /-
    🎉 no goals
  -/


theorem eigenvalue_pos_of_pos {μ : ℝ} {T : E →ₗ[𝕜] E} (hμ : HasEigenvalue T μ)
    (hnn : ∀ x : E, 0 < RCLike.re ⟪x, T x⟫) : 0 < μ := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : Real
    T : LinearMap (RingHom.id 𝕜) E E
    hμ : Module.End.HasEigenvalue T ↑μ
    hnn : ∀ (x : E), LT.lt 0 (RCLike.re (Inner.inner x (T x)))
    ⊢ LT.lt 0 μ
  -/
  obtain ⟨v, hv⟩ := hμ.exists_hasEigenvector
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : Real
    T : LinearMap (RingHom.id 𝕜) E E
    hμ : Module.End.HasEigenvalue T ↑μ
    hnn : ∀ (x : E), LT.lt 0 (RCLike.re (Inner.inner x (T x)))
    v : E
    hv : Module.End.HasEigenvector T (↑μ) v
    ⊢ LT.lt 0 μ
  -/
  have hpos : (0 : ℝ) < ‖v‖ ^ 2 := by simpa only [sq_pos_iff, norm_ne_zero_iff] using hv.2
  have : RCLike.re ⟪v, T v⟫ = μ * ‖v‖ ^ 2 := by
    have := congr_arg RCLike.re (inner_product_apply_eigenvector hv.1)
    -- Porting note: why can't `exact_mod_cast` do this? These lemmas are marked `norm_cast`
    rw [← RCLike.ofReal_pow, ← RCLike.ofReal_mul] at this
    exact mod_cast this
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    μ : Real
    T : LinearMap (RingHom.id 𝕜) E E
    hμ : Module.End.HasEigenvalue T ↑μ
    hnn : ∀ (x : E), LT.lt 0 (RCLike.re (Inner.inner x (T x)))
    v : E
    hv : Module.End.HasEigenvector T (↑μ) v
    hpos : LT.lt 0 (HPow.hPow (Norm.norm v) 2)
    this : Eq (RCLike.re (Inner.inner v (T v))) (HMul.hMul μ (HPow.hPow (Norm.norm …
    ⊢ LT.lt 0 μ
  -/
  exact (mul_pos_iff_of_pos_right hpos).mp (this ▸ hnn v)
  /-
    🎉 no goals
  -/


