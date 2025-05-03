instance : Module.Finite R P.rootSpan := Finite.span_of_finite R <| finite_range P.root


instance : Module.Finite R P.corootSpan := Finite.span_of_finite R <| finite_range P.coroot


/-- An invariant linear map from weight space to coweight space. -/
def Polarization : M →ₗ[R] N :=
  ∑ i, LinearMap.toSpanSingleton R N (P.coroot i) ∘ₗ P.coroot' i


@[simp]
lemma Polarization_apply (x : M) :
    P.Polarization x = ∑ i, P.coroot' i x • P.coroot i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    ⊢ Eq (P.Polarization x) (Finset.univ.sum fun i => HSMul.hSMul ((P.coroot' i) x …
  -/
  simp [Polarization]
  /-
    🎉 no goals
  -/


/-- An invariant linear map from coweight space to weight space. -/
def CoPolarization : N →ₗ[R] M :=
  ∑ i, LinearMap.toSpanSingleton R M (P.root i) ∘ₗ P.root' i


@[simp]
lemma CoPolarization_apply (x : N) :
    P.CoPolarization x = ∑ i, P.root' i x • P.root i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : N
    ⊢ Eq (P.CoPolarization x) (Finset.univ.sum fun i => HSMul.hSMul ((P.root' i) x …
  -/
  simp [CoPolarization]
  /-
    🎉 no goals
  -/


lemma CoPolarization_eq : P.CoPolarization = P.flip.Polarization :=
  rfl


/-- An invariant inner product on the weight space. -/
def RootForm : LinearMap.BilinForm R M :=
  ∑ i, (P.coroot' i).smulRight (P.coroot' i)


/-- An invariant inner product on the coweight space. -/
def CorootForm : LinearMap.BilinForm R N :=
  ∑ i, (P.root' i).smulRight (P.root' i)


lemma rootForm_apply_apply (x y : M) : P.RootForm x y =
    ∑ i, P.coroot' i x * P.coroot' i y := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : M
    ⊢ Eq ((P.RootForm x) y) (Finset.univ.sum fun i => HMul.hMul ((P.coroot' i) x)  …
  -/
  simp [RootForm]
  /-
    🎉 no goals
  -/


lemma corootForm_apply_apply (x y : N) : P.CorootForm x y =
    ∑ i, P.root' i x * P.root' i y := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : N
    ⊢ Eq ((P.CorootForm x) y) (Finset.univ.sum fun i => HMul.hMul ((P.root' i) x)  …
  -/
  simp [CorootForm]
  /-
    🎉 no goals
  -/


lemma toPerfectPairing_apply_apply_Polarization (x y : M) :
    P.toPerfectPairing y (P.Polarization x) = P.RootForm x y := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : M
    ⊢ Eq ((P.toPerfectPairing y) (P.Polarization x)) ((P.RootForm x) y)
  -/
  simp [RootForm]
  /-
    🎉 no goals
  -/


lemma toPerfectPairing_apply_CoPolarization (x : N) :
    P.toPerfectPairing (P.CoPolarization x) = P.CorootForm x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : N
    ⊢ Eq (P.toPerfectPairing (P.CoPolarization x)) (P.CorootForm x)
  -/
  ext y
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : N
    ⊢ Eq ((P.toPerfectPairing (P.CoPolarization x)) y) ((P.CorootForm x) y)
  -/
  exact P.flip.toPerfectPairing_apply_apply_Polarization x y
  /-
    🎉 no goals
  -/


lemma flip_comp_polarization_eq_rootForm :
    P.flip.toLin ∘ₗ P.Polarization = P.RootForm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq (P.flip.toLin.comp P.Polarization) P.RootForm
  -/
  ext; simp [rootForm_apply_apply, RootPairing.flip]
       /-
         🎉 no goals
       -/


lemma self_comp_coPolarization_eq_corootForm :
    P.toLin ∘ₗ P.CoPolarization = P.CorootForm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq (P.toLin.comp P.CoPolarization) P.CorootForm
  -/
  ext; simp [corootForm_apply_apply]
       /-
         🎉 no goals
       -/


lemma polarization_apply_eq_zero_iff (m : M) :
    P.Polarization m = 0 ↔ P.RootForm m = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    m : M
    ⊢ Iff (Eq (P.Polarization m) 0) (Eq (P.RootForm m) 0)
  -/
  rw [← flip_comp_polarization_eq_rootForm]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    m : M
    ⊢ Iff (Eq (P.Polarization m) 0) (Eq ((P.flip.toLin.comp P.Polarization) m) 0)
  -/
  refine ⟨fun h ↦ by simp [h], fun h ↦ ?_⟩
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    m : M
    h : Eq ((P.flip.toLin.comp P.Polarization) m) 0
    ⊢ Eq (P.Polarization m) 0
  -/
  change P.toDualRight (P.Polarization m) = 0 at h
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    m : M
    h : Eq (P.toDualRight (P.Polarization m)) 0
    ⊢ Eq (P.Polarization m) 0
  -/
  simp only [EmbeddingLike.map_eq_zero_iff] at h
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    m : M
    h : Eq (P.Polarization m) 0
    ⊢ Eq (P.Polarization m) 0
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma coPolarization_apply_eq_zero_iff (n : N) :
    P.CoPolarization n = 0 ↔ P.CorootForm n = 0 :=
  P.flip.polarization_apply_eq_zero_iff n


lemma ker_polarization_eq_ker_rootForm :
    LinearMap.ker P.Polarization = LinearMap.ker P.RootForm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Eq (LinearMap.ker P.Polarization) (LinearMap.ker P.RootForm)
  -/
  ext; simp only [LinearMap.mem_ker, P.polarization_apply_eq_zero_iff]
       /-
         🎉 no goals
       -/


lemma ker_copolarization_eq_ker_corootForm :
    LinearMap.ker P.CoPolarization = LinearMap.ker P.CorootForm :=
  P.flip.ker_polarization_eq_ker_rootForm


lemma rootForm_symmetric :
    LinearMap.IsSymm P.RootForm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ LinearMap.IsSymm P.RootForm
  -/
  simp [LinearMap.IsSymm, mul_comm, rootForm_apply_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma rootForm_reflection_reflection_apply (i : ι) (x y : M) :
    P.RootForm (P.reflection i x) (P.reflection i y) = P.RootForm x y := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    x y : M
    ⊢ Eq ((P.RootForm ((P.reflection i) x)) ((P.reflection i) y)) ((P.RootForm x) y)
  -/
  simp only [rootForm_apply_apply, coroot'_reflection]
  exact Fintype.sum_equiv (P.reflection_perm i)
    (fun j ↦ (P.coroot' (P.reflection_perm i j) x) * (P.coroot' (P.reflection_perm i j) y))
    (fun j ↦ P.coroot' j x * P.coroot' j y) (congrFun rfl)


/-- This is SGA3 XXI Lemma 1.2.1 (10), key for proving nondegeneracy and positivity. -/
lemma rootForm_self_smul_coroot (i : ι) :
    (P.RootForm (P.root i) (P.root i)) • P.coroot i = 2 • P.Polarization (P.root i) := by
  have hP : P.Polarization (P.root i) =
      ∑ j : ι, P.pairing i (P.reflection_perm i j) • P.coroot (P.reflection_perm i j) := by
    simp_rw [Polarization_apply, root_coroot'_eq_pairing]
    exact (Fintype.sum_equiv (P.reflection_perm i)
          (fun j ↦ P.pairing i (P.reflection_perm i j) • P.coroot (P.reflection_perm i j))
          (fun j ↦ P.pairing i j • P.coroot j) (congrFun rfl)).symm
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    hP : Eq (P.Polarization (P.root i)) (Finset.univ.sum fun j => HSMul.hSMul (P.p …
    ⊢ Eq (HSMul.hSMul ((P.RootForm (P.root i)) (P.root i)) (P.coroot i)) (HSMul.hS …
  -/
  rw [two_nsmul]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    hP : Eq (P.Polarization (P.root i)) (Finset.univ.sum fun j => HSMul.hSMul (P.p …
    ⊢ Eq (HSMul.hSMul ((P.RootForm (P.root i)) (P.root i)) (P.coroot i)) (HAdd.hAd …
  -/
  nth_rw 2 [hP]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    hP : Eq (P.Polarization (P.root i)) (Finset.univ.sum fun j => HSMul.hSMul (P.p …
    ⊢ Eq (HSMul.hSMul ((P.RootForm (P.root i)) (P.root i)) (P.coroot i)) (HAdd.hAd …
  -/
  rw [Polarization_apply]
  simp only [root_coroot'_eq_pairing, pairing_reflection_perm, pairing_reflection_perm_self_left,
    ← reflection_perm_coroot, smul_sub, neg_smul, sub_neg_eq_add]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    hP : Eq (P.Polarization (P.root i)) (Finset.univ.sum fun j => HSMul.hSMul (P.p …
    ⊢ Eq (HSMul.hSMul ((P.RootForm (P.root i)) (P.root i)) (P.coroot i)) (HAdd.hAd …
  -/
  rw [Finset.sum_add_distrib, ← add_assoc, ← sub_eq_iff_eq_add]
  simp only [rootForm_apply_apply, LinearMap.coe_comp, comp_apply, Polarization_apply,
    root_coroot_eq_pairing, map_sum, LinearMapClass.map_smul, Finset.sum_neg_distrib, ← smul_assoc]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    hP : Eq (P.Polarization (P.root i)) (Finset.univ.sum fun j => HSMul.hSMul (P.p …
    ⊢ Eq (HSub.hSub (HSMul.hSMul (Finset.univ.sum fun i_1 => HMul.hMul ((P.coroot' …
  -/
  rw [Finset.sum_smul, add_neg_eq_zero.mpr rfl]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    hP : Eq (P.Polarization (P.root i)) (Finset.univ.sum fun j => HSMul.hSMul (P.p …
    ⊢ Eq (HSub.hSub (Finset.univ.sum fun i_1 => HSMul.hSMul (HMul.hMul ((P.coroot' …
  -/
  exact sub_eq_zero_of_eq rfl
  /-
    🎉 no goals
  -/


lemma four_smul_rootForm_sq_eq_coxeterWeight_smul (i j : ι) :
    4 • (P.RootForm (P.root i) (P.root j)) ^ 2 = P.coxeterWeight i j •
      (P.RootForm (P.root i) (P.root i) * P.RootForm (P.root j) (P.root j)) := by
  have hij : 4 • (P.RootForm (P.root i)) (P.root j) =
      2 • P.toPerfectPairing (P.root j) (2 • P.Polarization (P.root i)) := by
    rw [← toPerfectPairing_apply_apply_Polarization, LinearMap.map_smul_of_tower, ← smul_assoc,
      Nat.nsmul_eq_mul]
  have hji : 2 • (P.RootForm (P.root i)) (P.root j) =
      P.toPerfectPairing (P.root i) (2 • P.Polarization (P.root j)) := by
    rw [show (P.RootForm (P.root i)) (P.root j) = (P.RootForm (P.root j)) (P.root i) by
      apply rootForm_symmetric, ← toPerfectPairing_apply_apply_Polarization,
      LinearMap.map_smul_of_tower]
  rw [sq, nsmul_eq_mul, ← mul_assoc, ← nsmul_eq_mul, hij, ← rootForm_self_smul_coroot,
    smul_mul_assoc 2, ← mul_smul_comm, hji, ← rootForm_self_smul_coroot, map_smul, ← pairing,
    map_smul, ← pairing, smul_eq_mul, smul_eq_mul, smul_eq_mul, coxeterWeight]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    hij : Eq (HSMul.hSMul 4 ((P.RootForm (P.root i)) (P.root j))) (HSMul.hSMul 2 ( …
    hji : Eq (HSMul.hSMul 2 ((P.RootForm (P.root i)) (P.root j))) ((P.toPerfectPai …
    ⊢ Eq (HMul.hMul (HMul.hMul ((P.RootForm (P.root i)) (P.root i)) (P.pairing j i …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma corootForm_self_smul_root (i : ι) :
    (P.CorootForm (P.coroot i) (P.coroot i)) • P.root i = 2 • P.CoPolarization (P.coroot i) :=
  rootForm_self_smul_coroot (P.flip) i


lemma rootForm_self_sum_of_squares (x : M) :
    IsSumSq (P.RootForm x x) :=
  P.rootForm_apply_apply x x ▸ IsSumSq.sum_mul_self Finset.univ _


lemma rootForm_root_self (j : ι) :
    P.RootForm (P.root j) (P.root j) = ∑ (i : ι), (P.pairing j i) * (P.pairing j i) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    j : ι
    ⊢ Eq ((P.RootForm (P.root j)) (P.root j)) (Finset.univ.sum fun i => HMul.hMul  …
  -/
  simp [rootForm_apply_apply]
  /-
    🎉 no goals
  -/


theorem range_polarization_domRestrict_le_span_coroot :
    LinearMap.range (P.Polarization.domRestrict P.rootSpan) ≤ P.corootSpan := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ LE.le (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) P.corootSpan
  -/
  intro y hy
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    y : N
    hy : Membership.mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) y
    ⊢ Membership.mem P.corootSpan y
  -/
  obtain ⟨x, hx⟩ := hy
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    y : N
    x : Subtype fun x => Membership.mem P.rootSpan x
    hx : Eq ((P.Polarization.domRestrict P.rootSpan) x) y
    ⊢ Membership.mem P.corootSpan y
  -/
  rw [← hx, LinearMap.domRestrict_apply, Polarization_apply]
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    y : N
    x : Subtype fun x => Membership.mem P.rootSpan x
    hx : Eq ((P.Polarization.domRestrict P.rootSpan) x) y
    ⊢ Membership.mem P.corootSpan (Finset.univ.sum fun i => HSMul.hSMul ((P.coroot …
  -/
  refine (mem_span_range_iff_exists_fun R).mpr ?_
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    y : N
    x : Subtype fun x => Membership.mem P.rootSpan x
    hx : Eq ((P.Polarization.domRestrict P.rootSpan) x) y
    ⊢ Exists fun c => Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (P.coroot i)) …
  -/
  use fun i => (P.toPerfectPairing x) (P.coroot i)
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    y : N
    x : Subtype fun x => Membership.mem P.rootSpan x
    hx : Eq ((P.Polarization.domRestrict P.rootSpan) x) y
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((fun i => (P.toPerfectPairing ↑x)  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem corootSpan_dualAnnihilator_le_ker_rootForm :
    P.corootSpan.dualAnnihilator.map P.toDualLeft.symm ≤ LinearMap.ker P.RootForm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ LE.le (Submodule.map P.toDualLeft.symm P.corootSpan.dualAnnihilator) (Linear …
  -/
  rw [← SetLike.coe_subset_coe, coe_corootSpan_dualAnnihilator_map]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ HasSubset.Subset (setOf fun x => ∀ (i : ι), Eq ((P.coroot' i) x) 0) ↑(Linear …
  -/
  intro x hx
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : Membership.mem (setOf fun x => ∀ (i : ι), Eq ((P.coroot' i) x) 0) x
    ⊢ Membership.mem (↑(LinearMap.ker P.RootForm)) x
  -/
  simp only [coroot', PerfectPairing.flip_apply_apply, mem_setOf_eq] at hx
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : ∀ (i : ι), Eq ((P.toPerfectPairing x) (P.coroot i)) 0
    ⊢ Membership.mem (↑(LinearMap.ker P.RootForm)) x
  -/
  ext y
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x : M
    hx : ∀ (i : ι), Eq ((P.toPerfectPairing x) (P.coroot i)) 0
    y : M
    ⊢ Eq ((P.RootForm x) y) (0 y)
  -/
  simp [rootForm_apply_apply, hx]
  /-
    🎉 no goals
  -/


theorem rootSpan_dualAnnihilator_le_ker_rootForm :
    P.rootSpan.dualAnnihilator.map P.toDualRight.symm ≤ LinearMap.ker P.CorootForm :=
  P.flip.corootSpan_dualAnnihilator_le_ker_rootForm


lemma prod_rootForm_smul_coroot_mem_range_domRestrict (i : ι) :
    (∏ a : ι, P.RootForm (P.root a) (P.root a)) • P.coroot i ∈
      LinearMap.range (P.Polarization.domRestrict (P.rootSpan)) := by
  obtain ⟨c, hc⟩ := Finset.dvd_prod_of_mem (fun a ↦ P.RootForm (P.root a) (P.root a))
    (Finset.mem_univ i)
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    c : R
    hc : Eq (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) (HMul.h …
    ⊢ Membership.mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) (HS …
  -/
  rw [hc, mul_comm, mul_smul, rootForm_self_smul_coroot]
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    c : R
    hc : Eq (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) (HMul.h …
    ⊢ Membership.mem (LinearMap.range (P.Polarization.domRestrict P.rootSpan)) (HS …
  -/
  refine LinearMap.mem_range.mpr ?_
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    c : R
    hc : Eq (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) (HMul.h …
    ⊢ Exists fun y => Eq ((P.Polarization.domRestrict P.rootSpan) y) (HSMul.hSMul  …
  -/
  use ⟨(c • 2 • P.root i), by aesop⟩
  /-
    case h
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    c : R
    hc : Eq (Finset.univ.prod fun i => (P.RootForm (P.root i)) (P.root i)) (HMul.h …
    ⊢ Eq ((P.Polarization.domRestrict P.rootSpan) ⟨HSMul.hSMul c (HSMul.hSMul 2 (P …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem rootForm_self_non_neg (x : M) : 0 ≤ P.RootForm x x :=
  IsSumSq.nonneg (P.rootForm_self_sum_of_squares x)


lemma rootForm_self_eq_zero_iff {x : M} :
    P.RootForm x x = 0 ↔ x ∈ LinearMap.ker P.RootForm :=
  P.RootForm.apply_apply_same_eq_zero_iff P.rootForm_self_non_neg P.rootForm_symmetric


lemma rootForm_root_self_pos (i : ι) :
    0 < P.RootForm (P.root i) (P.root i) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ LT.lt 0 ((P.RootForm (P.root i)) (P.root i))
  -/
  simp only [rootForm_apply_apply]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ LT.lt 0 (Finset.univ.sum fun i_1 => HMul.hMul ((P.coroot' i_1) (P.root i)) ( …
  -/
  exact Finset.sum_pos' (fun j _ ↦ mul_self_nonneg _) ⟨i, by simp⟩
  /-
    🎉 no goals
  -/


/-- SGA3 XXI Prop. 2.3.1 -/
lemma coxeterWeight_le_four (i j : ι) : P.coxeterWeight i j ≤ 4 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  set li := P.RootForm (P.root i) (P.root i)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  set lj := P.RootForm (P.root j) (P.root j)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    lj : R := (P.RootForm (P.root j)) (P.root j)
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  set lij := P.RootForm (P.root i) (P.root j)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    lj : R := (P.RootForm (P.root j)) (P.root j)
    lij : R := (P.RootForm (P.root i)) (P.root j)
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  have hi := P.rootForm_root_self_pos i
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    lj : R := (P.RootForm (P.root j)) (P.root j)
    lij : R := (P.RootForm (P.root i)) (P.root j)
    hi : LT.lt 0 ((P.RootForm (P.root i)) (P.root i))
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  have hj := P.rootForm_root_self_pos j
  have cs : 4 * lij ^ 2 ≤ 4 * (li * lj) := by
    rw [mul_le_mul_left four_pos]
    exact LinearMap.BilinForm.apply_sq_le_of_symm P.RootForm P.rootForm_self_non_neg
      P.rootForm_symmetric (P.root i) (P.root j)
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    lj : R := (P.RootForm (P.root j)) (P.root j)
    lij : R := (P.RootForm (P.root i)) (P.root j)
    hi : LT.lt 0 ((P.RootForm (P.root i)) (P.root i))
    hj : LT.lt 0 ((P.RootForm (P.root j)) (P.root j))
    cs : LE.le (HMul.hMul 4 (HPow.hPow lij 2)) (HMul.hMul 4 (HMul.hMul li lj))
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  have key : 4 • lij ^ 2 = _ • (li * lj) := P.four_smul_rootForm_sq_eq_coxeterWeight_smul i j
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    lj : R := (P.RootForm (P.root j)) (P.root j)
    lij : R := (P.RootForm (P.root i)) (P.root j)
    hi : LT.lt 0 ((P.RootForm (P.root i)) (P.root i))
    hj : LT.lt 0 ((P.RootForm (P.root j)) (P.root j))
    cs : LE.le (HMul.hMul 4 (HPow.hPow lij 2)) (HMul.hMul 4 (HMul.hMul li lj))
    key : Eq (HSMul.hSMul 4 (HPow.hPow lij 2)) (HSMul.hSMul (P.coxeterWeight i j)  …
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  simp only [nsmul_eq_mul, smul_eq_mul, Nat.cast_ofNat] at key
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι
    inst✝⁴ : LinearOrderedCommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i j : ι
    li : R := (P.RootForm (P.root i)) (P.root i)
    lj : R := (P.RootForm (P.root j)) (P.root j)
    lij : R := (P.RootForm (P.root i)) (P.root j)
    hi : LT.lt 0 ((P.RootForm (P.root i)) (P.root i))
    hj : LT.lt 0 ((P.RootForm (P.root j)) (P.root j))
    cs : LE.le (HMul.hMul 4 (HPow.hPow lij 2)) (HMul.hMul 4 (HMul.hMul li lj))
    key : Eq (HMul.hMul 4 (HPow.hPow lij 2)) (HMul.hMul (P.coxeterWeight i j) (HMu …
    ⊢ LE.le (P.coxeterWeight i j) 4
  -/
  rwa [key, mul_le_mul_right (by positivity)] at cs
  /-
    🎉 no goals
  -/


instance instIsRootPositiveRootForm : IsRootPositive P P.RootForm where
  zero_lt_apply_root i := P.rootForm_root_self_pos i
  symm := P.rootForm_symmetric
  apply_reflection_eq := P.rootForm_reflection_reflection_apply


lemma coxeterWeight_mem_set_of_isCrystallographic (i j : ι) [P.IsCrystallographic] :
    P.coxeterWeight i j ∈ ({0, 1, 2, 3, 4} : Set R) := by
  obtain ⟨n, hcn⟩ : ∃ n : ℕ, P.coxeterWeight i j = n := by
    obtain ⟨z, hz⟩ := P.exists_int_eq_coxeterWeight i j
    have hz₀ : 0 ≤ z := by simpa [hz] using P.coxeterWeight_non_neg P.RootForm i j
    obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hz₀
    exact ⟨n, by simp [hz]⟩
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    i j : ι
    inst✝ : P.IsCrystallographic
    n : Nat
    hcn : Eq (P.coxeterWeight i j) ↑n
    ⊢ Membership.mem (Insert.insert 0 (Insert.insert 1 (Insert.insert 2 (Insert.in …
  -/
  have : P.coxeterWeight i j ≤ 4 := P.coxeterWeight_le_four i j
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    i j : ι
    inst✝ : P.IsCrystallographic
    n : Nat
    hcn : Eq (P.coxeterWeight i j) ↑n
    this : LE.le (P.coxeterWeight i j) 4
    ⊢ Membership.mem (Insert.insert 0 (Insert.insert 1 (Insert.insert 2 (Insert.in …
  -/
  simp only [hcn, mem_insert_iff, mem_singleton_iff] at this ⊢
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    i j : ι
    inst✝ : P.IsCrystallographic
    n : Nat
    hcn : Eq (P.coxeterWeight i j) ↑n
    this : LE.le (↑n) 4
    ⊢ Or (Eq (↑n) 0) (Or (Eq (↑n) 1) (Or (Eq (↑n) 2) (Or (Eq (↑n) 3) (Eq (↑n) 4))))
  -/
  norm_cast at this ⊢
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    i j : ι
    inst✝ : P.IsCrystallographic
    n : Nat
    hcn : Eq (P.coxeterWeight i j) ↑n
    this : LE.le n 4
    ⊢ Or (Eq n 0) (Or (Eq n 1) (Or (Eq n 2) (Or (Eq n 3) (Eq n 4))))
  -/
  omega
  /-
    🎉 no goals
  -/


