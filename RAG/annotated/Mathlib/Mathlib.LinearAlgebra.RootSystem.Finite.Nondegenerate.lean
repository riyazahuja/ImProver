/-- We say a finite root pairing is anisotropic if there are no roots / coroots which have length
zero wrt the root / coroot forms.

Examples include crystallographic pairings in characteristic zero
`RootPairing.instIsAnisotropicOfIsCrystallographic` and pairings over ordered scalars.
`RootPairing.instIsAnisotropicOfLinearOrderedCommRing`. -/
class IsAnisotropic : Prop where
  rootForm_root_ne_zero (i : ι) : P.RootForm (P.root i) (P.root i) ≠ 0
  corootForm_coroot_ne_zero (i : ι) : P.CorootForm (P.coroot i) (P.coroot i) ≠ 0


instance [P.IsAnisotropic] : P.flip.IsAnisotropic where
  rootForm_root_ne_zero := IsAnisotropic.corootForm_coroot_ne_zero
  corootForm_coroot_ne_zero := IsAnisotropic.rootForm_root_ne_zero


/-- An auxiliary lemma en route to `RootPairing.instIsAnisotropicOfIsCrystallographic`. -/
private lemma rootForm_root_ne_zero_aux [CharZero R] [P.IsCrystallographic] (i : ι) :
    P.RootForm (P.root i) (P.root i) ≠ 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : Module R M
    inst✝² : Module R N
    P : RootPairing ι R M N
    inst✝¹ : CharZero R
    inst✝ : P.IsCrystallographic
    i : ι
    ⊢ Ne ((P.RootForm (P.root i)) (P.root i)) 0
  -/
  choose z hz using P.exists_int i
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : Module R M
    inst✝² : Module R N
    P : RootPairing ι R M N
    inst✝¹ : CharZero R
    inst✝ : P.IsCrystallographic
    i : ι
    z : ι → Int
    hz : ∀ (j : ι), Eq (↑(z j)) (P.pairing i j)
    ⊢ Ne ((P.RootForm (P.root i)) (P.root i)) 0
  -/
  simp only [rootForm_apply_apply, PerfectPairing.flip_apply_apply, root_coroot_eq_pairing, ← hz]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : Module R M
    inst✝² : Module R N
    P : RootPairing ι R M N
    inst✝¹ : CharZero R
    inst✝ : P.IsCrystallographic
    i : ι
    z : ι → Int
    hz : ∀ (j : ι), Eq (↑(z j)) (P.pairing i j)
    ⊢ Ne (Finset.univ.sum fun x => HMul.hMul ↑(z x) ↑(z x)) 0
  -/
  suffices 0 < ∑ i, z i * z i by norm_cast; exact this.ne'
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : Module R M
    inst✝² : Module R N
    P : RootPairing ι R M N
    inst✝¹ : CharZero R
    inst✝ : P.IsCrystallographic
    i : ι
    z : ι → Int
    hz : ∀ (j : ι), Eq (↑(z j)) (P.pairing i j)
    ⊢ LT.lt 0 (Finset.univ.sum fun i => HMul.hMul (z i) (z i))
  -/
  refine Finset.sum_pos' (fun i _ ↦ mul_self_nonneg (z i)) ⟨i, Finset.mem_univ i, ?_⟩
  have hzi : z i = 2 := by
    specialize hz i
    rw [pairing_same] at hz
    norm_cast at hz
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : Module R M
    inst✝² : Module R N
    P : RootPairing ι R M N
    inst✝¹ : CharZero R
    inst✝ : P.IsCrystallographic
    i : ι
    z : ι → Int
    hz : ∀ (j : ι), Eq (↑(z j)) (P.pairing i j)
    hzi : Eq (z i) 2
    ⊢ LT.lt 0 (HMul.hMul (z i) (z i))
  -/
  simp [hzi]
  /-
    🎉 no goals
  -/


instance instIsAnisotropicOfIsCrystallographic [CharZero R] [P.IsCrystallographic] :
    IsAnisotropic P where
  rootForm_root_ne_zero := P.rootForm_root_ne_zero_aux
  corootForm_coroot_ne_zero := P.flip.rootForm_root_ne_zero_aux


@[simp]
lemma finrank_rootSpan_map_polarization_eq_finrank_corootSpan :
    finrank R (P.rootSpan.map P.Polarization) = finrank R P.corootSpan := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (Submodule.map P.Polar …
  -/
  rw [← LinearMap.range_domRestrict]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.range (P.Po …
  -/
  apply (Submodule.finrank_mono P.range_polarization_domRestrict_le_span_coroot).antisymm
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem P.corootSpan x)) (M …
  -/
  have : IsReflexive R N := PerfectPairing.reflexive_right P.toPerfectPairing
  have h_ne : ∏ i, P.RootForm (P.root i) (P.root i) ≠ 0 :=
    Finset.prod_ne_zero_iff.mpr fun i _ ↦ IsAnisotropic.rootForm_root_ne_zero i
  refine LinearMap.finrank_le_of_isSMulRegular P.corootSpan
    (LinearMap.range (P.Polarization.domRestrict P.rootSpan))
    (smul_right_injective N h_ne)
    fun _ hx => ?_
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    this : Module.IsReflexive R N
    h_ne : Ne (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) 0
    x✝ : N
    hx : Membership.mem P.corootSpan x✝
    ⊢ Membership.mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) (HS …
  -/
  obtain ⟨c, hc⟩ := (mem_span_range_iff_exists_fun R).mp hx
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    this : Module.IsReflexive R N
    h_ne : Ne (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) 0
    x✝ : N
    hx : Membership.mem P.corootSpan x✝
    c : ι → R
    hc : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (P.coroot i)) x✝
    ⊢ Membership.mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) (HS …
  -/
  rw [← hc, Finset.smul_sum]
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    this : Module.IsReflexive R N
    h_ne : Ne (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) 0
    x✝ : N
    hx : Membership.mem P.corootSpan x✝
    c : ι → R
    hc : Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (P.coroot i)) x✝
    ⊢ Membership.mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) (Fi …
  -/
  simp_rw [smul_smul, mul_comm, ← smul_smul]
  exact Submodule.sum_smul_mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) c
    (fun c _ ↦ prod_rootForm_smul_coroot_mem_range_domRestrict P c)


/-- An auxiliary lemma en route to `RootPairing.finrank_corootSpan_eq`. -/
private lemma finrank_corootSpan_le :
    finrank R P.corootSpan ≤ finrank R P.rootSpan := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem P.corootSpan x)) (M …
  -/
  rw [← finrank_rootSpan_map_polarization_eq_finrank_corootSpan]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem (Submodule.map P.Po …
  -/
  exact Submodule.finrank_map_le P.Polarization P.rootSpan
  /-
    🎉 no goals
  -/


lemma finrank_corootSpan_eq :
    finrank R P.corootSpan = finrank R P.rootSpan :=
  le_antisymm P.finrank_corootSpan_le P.flip.finrank_corootSpan_le


lemma disjoint_rootSpan_ker_rootForm :
    Disjoint P.rootSpan (LinearMap.ker P.RootForm) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ Disjoint P.rootSpan (LinearMap.ker P.RootForm)
  -/
  have : IsReflexive R M := PerfectPairing.reflexive_left P.toPerfectPairing
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    this : Module.IsReflexive R M
    ⊢ Disjoint P.rootSpan (LinearMap.ker P.RootForm)
  -/
  rw [← P.ker_polarization_eq_ker_rootForm]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    this : Module.IsReflexive R M
    ⊢ Disjoint P.rootSpan (LinearMap.ker P.Polarization)
  -/
  refine Submodule.disjoint_ker_of_finrank_le (L := P.rootSpan) P.Polarization ?_
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Fintype ι
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    this : Module.IsReflexive R M
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem P.rootSpan x)) (Mod …
  -/
  rw [P.finrank_rootSpan_map_polarization_eq_finrank_corootSpan, P.finrank_corootSpan_eq]
  /-
    🎉 no goals
  -/


lemma disjoint_corootSpan_ker_corootForm :
    Disjoint P.corootSpan (LinearMap.ker P.CorootForm) :=
  P.flip.disjoint_rootSpan_ker_rootForm


lemma isCompl_rootSpan_ker_rootForm :
    IsCompl P.rootSpan (LinearMap.ker P.RootForm) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ IsCompl P.rootSpan (LinearMap.ker P.RootForm)
  -/
  have _iM : IsReflexive R M := PerfectPairing.reflexive_left P.toPerfectPairing
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    _iM : Module.IsReflexive R M
    ⊢ IsCompl P.rootSpan (LinearMap.ker P.RootForm)
  -/
  have _iN : IsReflexive R N := PerfectPairing.reflexive_right P.toPerfectPairing
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    _iM : Module.IsReflexive R M
    _iN : Module.IsReflexive R N
    ⊢ IsCompl P.rootSpan (LinearMap.ker P.RootForm)
  -/
  refine (Submodule.isCompl_iff_disjoint _ _ ?_).mpr P.disjoint_rootSpan_ker_rootForm
  have aux : finrank R M = finrank R P.rootSpan + finrank R P.corootSpan.dualAnnihilator := by
    rw [P.toPerfectPairing.finrank_eq, ← P.finrank_corootSpan_eq,
      Subspace.finrank_add_finrank_dualAnnihilator_eq P.corootSpan]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    _iM : Module.IsReflexive R M
    _iN : Module.IsReflexive R N
    aux : Eq (Module.finrank R M) (HAdd.hAdd (Module.finrank R (Subtype fun x => M …
    ⊢ LE.le (Module.finrank R M) (HAdd.hAdd (Module.finrank R (Subtype fun x => Me …
  -/
  rw [aux, add_le_add_iff_left]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    _iM : Module.IsReflexive R M
    _iN : Module.IsReflexive R N
    aux : Eq (Module.finrank R M) (HAdd.hAdd (Module.finrank R (Subtype fun x => M …
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem P.corootSpan.dualAn …
  -/
  convert Submodule.finrank_mono P.corootSpan_dualAnnihilator_le_ker_rootForm
  /-
    case h.e'_3
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    _iM : Module.IsReflexive R M
    _iN : Module.IsReflexive R N
    aux : Eq (Module.finrank R M) (HAdd.hAdd (Module.finrank R (Subtype fun x => M …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem P.corootSpan.dualAnnih …
  -/
  exact (LinearEquiv.finrank_map_eq _ _).symm
  /-
    🎉 no goals
  -/


lemma isCompl_corootSpan_ker_corootForm :
    IsCompl P.corootSpan (LinearMap.ker P.CorootForm) :=
  P.flip.isCompl_rootSpan_ker_rootForm


/-- See also `RootPairing.rootForm_restrict_nondegenerate_of_ordered`.

Note that this applies to crystallographic root systems in characteristic zero via
`RootPairing.instIsAnisotropicOfIsCrystallographic`. -/
lemma rootForm_restrict_nondegenerate_of_isAnisotropic :
    LinearMap.Nondegenerate (P.RootForm.restrict P.rootSpan) :=
  P.rootForm_symmetric.nondegenerate_restrict_of_isCompl_ker P.isCompl_rootSpan_ker_rootForm


@[simp]
lemma orthogonal_rootSpan_eq :
    P.RootForm.orthogonal P.rootSpan = LinearMap.ker P.RootForm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ Eq (P.RootForm.orthogonal P.rootSpan) (LinearMap.ker P.RootForm)
  -/
  rw [← LinearMap.BilinForm.orthogonal_top_eq_ker P.rootForm_symmetric.isRefl]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ Eq (P.RootForm.orthogonal P.rootSpan) (P.RootForm.orthogonal Top.top)
  -/
  refine le_antisymm ?_ (by intro; aesop)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    ⊢ LE.le (P.RootForm.orthogonal P.rootSpan) (P.RootForm.orthogonal Top.top)
  -/
  rintro x hx y -
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    x : M
    hx : Membership.mem (P.RootForm.orthogonal P.rootSpan) x
    y : M
    ⊢ P.RootForm.IsOrtho y x
  -/
  simp only [LinearMap.BilinForm.mem_orthogonal_iff, LinearMap.BilinForm.IsOrtho] at hx ⊢
  obtain ⟨u, hu, v, hv, rfl⟩ : ∃ᵉ (u ∈ P.rootSpan) (v ∈ LinearMap.ker P.RootForm), u + v = y := by
    rw [← Submodule.mem_sup, P.isCompl_rootSpan_ker_rootForm.sup_eq_top]; exact Submodule.mem_top
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    x : M
    hx : ∀ (n : M), Membership.mem P.rootSpan n → Eq ((P.RootForm n) x) 0
    u : M
    hu : Membership.mem P.rootSpan u
    v : M
    hv : Membership.mem (LinearMap.ker P.RootForm) v
    ⊢ Eq ((P.RootForm (HAdd.hAdd u v)) x) 0
  -/
  simp only [LinearMap.mem_ker] at hv
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Field R
    inst✝² : Module R M
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    inst✝ : P.IsAnisotropic
    x : M
    hx : ∀ (n : M), Membership.mem P.rootSpan n → Eq ((P.RootForm n) x) 0
    u : M
    hu : Membership.mem P.rootSpan u
    v : M
    hv : Eq (P.RootForm v) 0
    ⊢ Eq ((P.RootForm (HAdd.hAdd u v)) x) 0
  -/
  simp [hx _ hu, hv]
  /-
    🎉 no goals
  -/


@[simp]
lemma orthogonal_corootSpan_eq :
    P.CorootForm.orthogonal P.corootSpan = LinearMap.ker P.CorootForm :=
  P.flip.orthogonal_rootSpan_eq


instance instIsAnisotropicOfLinearOrderedCommRing : IsAnisotropic P where
  rootForm_root_ne_zero i := (P.rootForm_root_self_pos i).ne'
  corootForm_coroot_ne_zero i := (P.flip.rootForm_root_self_pos i).ne'


/-- See also `RootPairing.rootForm_restrict_nondegenerate_of_isAnisotropic`. -/
lemma rootForm_restrict_nondegenerate_of_ordered :
    LinearMap.Nondegenerate (P.RootForm.restrict P.rootSpan) :=
  (P.RootForm.nondegenerate_restrict_iff_disjoint_ker (rootForm_self_non_neg P)
    P.rootForm_symmetric).mpr P.disjoint_rootSpan_ker_rootForm


lemma eq_zero_of_mem_rootSpan_of_rootForm_self_eq_zero {x : M}
    (hx : x ∈ P.rootSpan) (hx' : P.RootForm x x = 0) :
    x = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : Module R M
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : Membership.mem P.rootSpan x
    hx' : Eq ((P.RootForm x) x) 0
    ⊢ Eq x 0
  -/
  have : x ∈ P.rootSpan ⊓ LinearMap.ker P.RootForm := ⟨hx, P.rootForm_self_eq_zero_iff.mp hx'⟩
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : Module R M
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : Membership.mem P.rootSpan x
    hx' : Eq ((P.RootForm x) x) 0
    this : Membership.mem (Min.min P.rootSpan (LinearMap.ker P.RootForm)) x
    ⊢ Eq x 0
  -/
  simpa [P.disjoint_rootSpan_ker_rootForm.eq_bot] using this
  /-
    🎉 no goals
  -/


lemma rootForm_pos_of_ne_zero {x : M} (hx : x ∈ P.rootSpan) (h : x ≠ 0) :
    0 < P.RootForm x x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : Module R M
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : Membership.mem P.rootSpan x
    h : Ne x 0
    ⊢ LT.lt 0 ((P.RootForm x) x)
  -/
  apply (P.rootForm_self_non_neg x).lt_of_ne
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : Module R M
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : Membership.mem P.rootSpan x
    h : Ne x 0
    ⊢ Ne 0 ((P.RootForm x) x)
  -/
  contrapose! h
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : LinearOrderedCommRing R
    inst✝¹ : Module R M
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : Membership.mem P.rootSpan x
    h : Eq 0 ((P.RootForm x) x)
    ⊢ Eq x 0
  -/
  exact P.eq_zero_of_mem_rootSpan_of_rootForm_self_eq_zero hx h.symm
  /-
    🎉 no goals
  -/


lemma _root_.RootSystem.rootForm_anisotropic (P : RootSystem ι R M N) :
    P.RootForm.toQuadraticMap.Anisotropic :=
  fun x ↦ P.eq_zero_of_mem_rootSpan_of_rootForm_self_eq_zero <| by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : Fintype ι
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup N
      inst✝² : LinearOrderedCommRing R
      inst✝¹ : Module R M
      inst✝ : Module R N
      P : RootSystem ι R M N
      x : M
      ⊢ Membership.mem P.rootSpan x
    -/
    simpa only [rootSpan, P.span_eq_top] using Submodule.mem_top
    /-
      🎉 no goals
    -/


