/-- The `IndepMatroid` whose independent sets are the independent subsets of `R`. -/
@[simps] def restrictIndepMatroid (M : Matroid α) (R : Set α) : IndepMatroid α where
  E := R
  Indep I := M.Indep I ∧ I ⊆ R
  indep_empty := ⟨M.empty_indep, empty_subset _⟩
  indep_subset := fun _ _ h hIJ ↦ ⟨h.1.subset hIJ, hIJ.trans h.2⟩
  indep_aug := by
    /-
      α : Type u_1
      M✝ : Matroid α
      R✝ I X Y : Set α
      M : Matroid α
      R : Set α
      ⊢ ∀ ⦃I B : Set α⦄, (fun I => And (M.Indep I) (HasSubset.Subset I R)) I → Not ( …
    -/
    rintro I I' ⟨hI, hIY⟩ (hIn : ¬ M.Basis' I R) (hI' : M.Basis' I' R)
    /-
      case intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I I' : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis' I R)
      hI' : M.Basis' I' R
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff I' I) x) ((fun I => And (M. …
    -/
    rw [basis'_iff_basis_inter_ground] at hIn hI'
    /-
      case intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I I' : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      hI' : M.Basis I' (Inter.inter R M.E)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff I' I) x) ((fun I => And (M. …
    -/
    obtain ⟨B', hB', rfl⟩ := hI'.exists_base
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      B' : Set α
      hB' : M.Base B'
      hI' : M.Basis (Inter.inter B' (Inter.inter R M.E)) (Inter.inter R M.E)
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff (Inter.inter B' (Inter.inte …
    -/
    obtain ⟨B, hB, hIB, hBIB'⟩ := hI.exists_base_subset_union_base hB'
    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      B' : Set α
      hB' : M.Base B'
      hI' : M.Basis (Inter.inter B' (Inter.inter R M.E)) (Inter.inter R M.E)
      B : Set α
      hB : M.Base B
      hIB : HasSubset.Subset I B
      hBIB' : HasSubset.Subset B (Union.union I B')
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff (Inter.inter B' (Inter.inte …
    -/
    rw [hB'.inter_basis_iff_compl_inter_basis_dual, diff_inter_diff] at hI'

    have hss : M.E \ (B' ∪ (R ∩ M.E)) ⊆ M.E \ (B ∪ (R ∩ M.E)) := by
      apply diff_subset_diff_right
      rw [union_subset_iff, and_iff_left subset_union_right, union_comm]
      exact hBIB'.trans (union_subset_union_left _ (subset_inter hIY hI.subset_ground))

    have hi : M✶.Indep (M.E \ (B ∪ (R ∩ M.E))) := by
      rw [dual_indep_iff_exists]
      exact ⟨B, hB, disjoint_of_subset_right subset_union_left disjoint_sdiff_left⟩

    have h_eq := hI'.eq_of_subset_indep hi hss
      (diff_subset_diff_right subset_union_right)
    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      B' : Set α
      hB' : M.Base B'
      hI' : M.dual.Basis (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E))) (SDi …
      B : Set α
      hB : M.Base B
      hIB : HasSubset.Subset I B
      hBIB' : HasSubset.Subset B (Union.union I B')
      hss : HasSubset.Subset (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E)))  …
      hi : M.dual.Indep (SDiff.sdiff M.E (Union.union B (Inter.inter R M.E)))
      h_eq : Eq (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E))) (SDiff.sdiff  …
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff (Inter.inter B' (Inter.inte …
    -/
    rw [h_eq, ← diff_inter_diff, ← hB.inter_basis_iff_compl_inter_basis_dual] at hI'

    obtain ⟨J, hJ, hIJ⟩ := hI.subset_basis_of_subset
      (subset_inter hIB (subset_inter hIY hI.subset_ground))
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      B' : Set α
      hB' : M.Base B'
      B : Set α
      hI' : M.Basis (Inter.inter B (Inter.inter R M.E)) (Inter.inter R M.E)
      hB : M.Base B
      hIB : HasSubset.Subset I B
      hBIB' : HasSubset.Subset B (Union.union I B')
      hss : HasSubset.Subset (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E)))  …
      hi : M.dual.Indep (SDiff.sdiff M.E (Union.union B (Inter.inter R M.E)))
      h_eq : Eq (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E))) (SDiff.sdiff  …
      J : Set α
      hJ : M.Basis J (Inter.inter B (Inter.inter R M.E))
      hIJ : HasSubset.Subset I J
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff (Inter.inter B' (Inter.inte …
    -/
    obtain rfl := hI'.indep.eq_of_basis hJ

    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      B' : Set α
      hB' : M.Base B'
      B : Set α
      hI' : M.Basis (Inter.inter B (Inter.inter R M.E)) (Inter.inter R M.E)
      hB : M.Base B
      hIB : HasSubset.Subset I B
      hBIB' : HasSubset.Subset B (Union.union I B')
      hss : HasSubset.Subset (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E)))  …
      hi : M.dual.Indep (SDiff.sdiff M.E (Union.union B (Inter.inter R M.E)))
      h_eq : Eq (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E))) (SDiff.sdiff  …
      hJ : M.Basis (Inter.inter B (Inter.inter R M.E)) (Inter.inter B (Inter.inter R …
      hIJ : HasSubset.Subset I (Inter.inter B (Inter.inter R M.E))
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff (Inter.inter B' (Inter.inte …
    -/
    have hIJ' : I ⊂ B ∩ (R ∩ M.E) := hIJ.ssubset_of_ne (fun he ↦ hIn (by rwa [he]))
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R I : Set α
      hI : M.Indep I
      hIY : HasSubset.Subset I R
      hIn : Not (M.Basis I (Inter.inter R M.E))
      B' : Set α
      hB' : M.Base B'
      B : Set α
      hI' : M.Basis (Inter.inter B (Inter.inter R M.E)) (Inter.inter R M.E)
      hB : M.Base B
      hIB : HasSubset.Subset I B
      hBIB' : HasSubset.Subset B (Union.union I B')
      hss : HasSubset.Subset (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E)))  …
      hi : M.dual.Indep (SDiff.sdiff M.E (Union.union B (Inter.inter R M.E)))
      h_eq : Eq (SDiff.sdiff M.E (Union.union B' (Inter.inter R M.E))) (SDiff.sdiff  …
      hJ : M.Basis (Inter.inter B (Inter.inter R M.E)) (Inter.inter B (Inter.inter R …
      hIJ : HasSubset.Subset I (Inter.inter B (Inter.inter R M.E))
      hIJ' : HasSSubset.SSubset I (Inter.inter B (Inter.inter R M.E))
      ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff (Inter.inter B' (Inter.inte …
    -/
    obtain ⟨e, he⟩ := exists_of_ssubset hIJ'
    exact ⟨e, ⟨⟨(hBIB' he.1.1).elim (fun h ↦ (he.2 h).elim) id,he.1.2⟩, he.2⟩,
      hI'.indep.subset (insert_subset he.1 hIJ), insert_subset he.1.2.1 hIY⟩
  indep_maximal := by
    /-
      α : Type u_1
      M✝ : Matroid α
      R✝ I X Y : Set α
      M : Matroid α
      R : Set α
      ⊢ ∀ (X : Set α), HasSubset.Subset X R → Matroid.ExistsMaximalSubsetProperty (f …
    -/
    rintro A hAR I ⟨hI, _⟩ hIA
    /-
      case intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R A : Set α
      hAR : HasSubset.Subset A R
      I : Set α
      hI : M.Indep I
      right✝ : HasSubset.Subset I R
      hIA : HasSubset.Subset I A
      ⊢ Exists fun J => And (HasSubset.Subset I J) (Maximal (fun K => And ((fun I => …
    -/
    obtain ⟨J, hJ, hIJ⟩ := hI.subset_basis'_of_subset hIA
    /-
      case intro.intro.intro
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R A : Set α
      hAR : HasSubset.Subset A R
      I : Set α
      hI : M.Indep I
      right✝ : HasSubset.Subset I R
      hIA : HasSubset.Subset I A
      J : Set α
      hJ : M.Basis' J A
      hIJ : HasSubset.Subset I J
      ⊢ Exists fun J => And (HasSubset.Subset I J) (Maximal (fun K => And ((fun I => …
    -/
    use J
    simp only [hIJ, and_assoc, maximal_subset_iff, hJ.indep, hJ.subset, and_imp, true_and,
      hJ.subset.trans hAR]
    /-
      case h
      α : Type u_1
      M✝ : Matroid α
      R✝ I✝ X Y : Set α
      M : Matroid α
      R A : Set α
      hAR : HasSubset.Subset A R
      I : Set α
      hI : M.Indep I
      right✝ : HasSubset.Subset I R
      hIA : HasSubset.Subset I A
      J : Set α
      hJ : M.Basis' J A
      hIJ : HasSubset.Subset I J
      ⊢ ∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset t R → HasSubset.Subset t A → Has …
    -/
    exact fun K hK _ hKA hJK ↦ hJ.eq_of_subset_indep hK hJK hKA
    /-
      🎉 no goals
    -/
  subset_ground _ := And.right


/-- Change the ground set of a matroid to some `R : Set α`. The independent sets of the restriction
  are the independent subsets of the new ground set. Most commonly used when `R ⊆ M.E`,
  but it is convenient not to require this. The elements of `R \ M.E` become 'loops'. -/
def restrict (M : Matroid α) (R : Set α) : Matroid α := (M.restrictIndepMatroid R).matroid


/-- `M ↾ R` means `M.restrict R`. -/
scoped infixl:65  " ↾ " => Matroid.restrict


@[simp] theorem restrict_indep_iff : (M ↾ R).Indep I ↔ M.Indep I ∧ I ⊆ R := Iff.rfl


theorem Indep.indep_restrict_of_subset (h : M.Indep I) (hIR : I ⊆ R) : (M ↾ R).Indep I :=
  restrict_indep_iff.mpr ⟨h,hIR⟩


theorem Indep.of_restrict (hI : (M ↾ R).Indep I) : M.Indep I :=
  (restrict_indep_iff.1 hI).1


@[simp] theorem restrict_ground_eq : (M ↾ R).E = R := rfl


theorem restrict_finite {R : Set α} (hR : R.Finite) : (M ↾ R).Finite :=
  ⟨hR⟩


@[simp] theorem restrict_dep_iff : (M ↾ R).Dep X ↔ ¬ M.Indep X ∧ X ⊆ R := by
  /-
    α : Type u_1
    M : Matroid α
    R X : Set α
    ⊢ Iff ((M.restrict R).Dep X) (And (Not (M.Indep X)) (HasSubset.Subset X R))
  -/
  rw [Dep, restrict_indep_iff, restrict_ground_eq]; tauto
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp] theorem restrict_ground_eq_self (M : Matroid α) : (M ↾ M.E) = M := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq (M.restrict M.E) M
  -/
  refine ext_indep rfl ?_; aesop
                           /-
                             🎉 no goals
                           -/


theorem restrict_restrict_eq {R₁ R₂ : Set α} (M : Matroid α) (hR : R₂ ⊆ R₁) :
    (M ↾ R₁) ↾ R₂ = M ↾ R₂ := by
  /-
    α : Type u_1
    R₁ R₂ : Set α
    M : Matroid α
    hR : HasSubset.Subset R₂ R₁
    ⊢ Eq ((M.restrict R₁).restrict R₂) (M.restrict R₂)
  -/
  refine ext_indep rfl ?_
  /-
    α : Type u_1
    R₁ R₂ : Set α
    M : Matroid α
    hR : HasSubset.Subset R₂ R₁
    ⊢ ∀ ⦃I : Set α⦄, HasSubset.Subset I ((M.restrict R₁).restrict R₂).E → Iff (((M …
  -/
  simp only [restrict_ground_eq, restrict_indep_iff, and_congr_left_iff, and_iff_left_iff_imp]
  /-
    α : Type u_1
    R₁ R₂ : Set α
    M : Matroid α
    hR : HasSubset.Subset R₂ R₁
    ⊢ ∀ ⦃I : Set α⦄, HasSubset.Subset I R₂ → HasSubset.Subset I R₂ → M.Indep I → H …
  -/
  exact fun _ h _ _ ↦ h.trans hR
  /-
    🎉 no goals
  -/


@[simp] theorem restrict_idem (M : Matroid α) (R : Set α) : M ↾ R ↾ R = M ↾ R := by
  /-
    α : Type u_1
    M : Matroid α
    R : Set α
    ⊢ Eq ((M.restrict R).restrict R) (M.restrict R)
  -/
  rw [M.restrict_restrict_eq Subset.rfl]
  /-
    🎉 no goals
  -/


@[simp] theorem base_restrict_iff (hX : X ⊆ M.E := by aesop_mat) :
    (M ↾ X).Base I ↔ M.Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff ((M.restrict X).Base I) (M.Basis I X)
  -/
  simp_rw [base_iff_maximal_indep, Basis, and_iff_left hX, maximal_iff, restrict_indep_iff]
  /-
    🎉 no goals
  -/


theorem base_restrict_iff' : (M ↾ X).Base I ↔ M.Basis' I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    ⊢ Iff ((M.restrict X).Base I) (M.Basis' I X)
  -/
  simp_rw [base_iff_maximal_indep, Basis', maximal_iff, restrict_indep_iff]
  /-
    🎉 no goals
  -/


theorem Basis.restrict_base (h : M.Basis I X) : (M ↾ X).Base I :=
  (base_restrict_iff h.subset_ground).2 h


instance restrict_finiteRk [M.FiniteRk] (R : Set α) : (M ↾ R).FiniteRk :=
  let ⟨_, hB⟩ := (M ↾ R).exists_base
  hB.finiteRk_of_finite (hB.indep.of_restrict.finite)


instance restrict_finitary [Finitary M] (R : Set α) : Finitary (M ↾ R) := by
  /-
    α : Type u_1
    M : Matroid α
    R✝ I X Y : Set α
    inst✝ : M.Finitary
    R : Set α
    ⊢ (M.restrict R).Finitary
  -/
  refine ⟨fun I hI ↦ ?_⟩
  /-
    α : Type u_1
    M : Matroid α
    R✝ I✝ X Y : Set α
    inst✝ : M.Finitary
    R I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → (M.restrict R).Indep J
    ⊢ (M.restrict R).Indep I
  -/
  simp only [restrict_indep_iff] at *
  /-
    α : Type u_1
    M : Matroid α
    R✝ I✝ X Y : Set α
    inst✝ : M.Finitary
    R I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → And (M.Indep J) (HasSubs …
    ⊢ And (M.Indep I) (HasSubset.Subset I R)
  -/
  rw [indep_iff_forall_finite_subset_indep]
  exact ⟨fun J hJ hJfin ↦ (hI J hJ hJfin).1,
    fun e heI ↦ singleton_subset_iff.1 (hI _ (by simpa) (toFinite _)).2⟩


@[simp] theorem Basis.base_restrict (h : M.Basis I X) : (M ↾ X).Base I :=
  (base_restrict_iff h.subset_ground).mpr h


theorem Basis.basis_restrict_of_subset (hI : M.Basis I X) (hXY : X ⊆ Y) : (M ↾ Y).Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hXY : HasSubset.Subset X Y
    ⊢ (M.restrict Y).Basis I X
  -/
  rwa [← base_restrict_iff, M.restrict_restrict_eq hXY, base_restrict_iff]
  /-
    🎉 no goals
  -/


theorem basis'_restrict_iff : (M ↾ R).Basis' I X ↔ M.Basis' I (X ∩ R) ∧ I ⊆ R := by
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    ⊢ Iff ((M.restrict R).Basis' I X) (And (M.Basis' I (Inter.inter X R)) (HasSubs …
  -/
  simp_rw [Basis', maximal_iff, restrict_indep_iff, subset_inter_iff, and_imp]
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    ⊢ Iff (And (And (And (M.Indep I) (HasSubset.Subset I R)) (HasSubset.Subset I X …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem basis_restrict_iff' : (M ↾ R).Basis I X ↔ M.Basis I (X ∩ M.E) ∧ X ⊆ R := by
  rw [basis_iff_basis'_subset_ground, basis'_restrict_iff, restrict_ground_eq, and_congr_left_iff,
    ← basis'_iff_basis_inter_ground]
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    ⊢ HasSubset.Subset X R → Iff (And (M.Basis' I (Inter.inter X R)) (HasSubset.Su …
  -/
  intro hXR
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    hXR : HasSubset.Subset X R
    ⊢ Iff (And (M.Basis' I (Inter.inter X R)) (HasSubset.Subset I R)) (M.Basis' I X)
  -/
  rw [inter_eq_self_of_subset_left hXR, and_iff_left_iff_imp]
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    hXR : HasSubset.Subset X R
    ⊢ M.Basis' I X → HasSubset.Subset I R
  -/
  exact fun h ↦ h.subset.trans hXR
  /-
    🎉 no goals
  -/


theorem basis_restrict_iff (hR : R ⊆ M.E := by aesop_mat) :
    (M ↾ R).Basis I X ↔ M.Basis I X ∧ X ⊆ R := by
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    hR : autoParam (HasSubset.Subset R M.E) _auto✝
    ⊢ Iff ((M.restrict R).Basis I X) (And (M.Basis I X) (HasSubset.Subset X R))
  -/
  rw [basis_restrict_iff', and_congr_left_iff]
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    hR : autoParam (HasSubset.Subset R M.E) _auto✝
    ⊢ HasSubset.Subset X R → Iff (M.Basis I (Inter.inter X M.E)) (M.Basis I X)
  -/
  intro hXR
  /-
    α : Type u_1
    M : Matroid α
    R I X : Set α
    hR : autoParam (HasSubset.Subset R M.E) _auto✝
    hXR : HasSubset.Subset X R
    ⊢ Iff (M.Basis I (Inter.inter X M.E)) (M.Basis I X)
  -/
  rw [← basis'_iff_basis_inter_ground, basis'_iff_basis]
  /-
    🎉 no goals
  -/


theorem restrict_eq_restrict_iff (M M' : Matroid α) (X : Set α) :
    M ↾ X = M' ↾ X ↔ ∀ I, I ⊆ X → (M.Indep I ↔ M'.Indep I) := by
  /-
    α : Type u_1
    M M' : Matroid α
    X : Set α
    ⊢ Iff (Eq (M.restrict X) (M'.restrict X)) (∀ (I : Set α), HasSubset.Subset I X …
  -/
  refine ⟨fun h I hIX ↦ ?_, fun h ↦ ext_indep rfl fun I (hI : I ⊆ X) ↦ ?_⟩
  · rw [← and_iff_left (a := (M.Indep I)) hIX, ← and_iff_left (a := (M'.Indep I)) hIX,
      ← restrict_indep_iff, h, restrict_indep_iff]
  /-
    case refine_2
    α : Type u_1
    M M' : Matroid α
    X : Set α
    h : ∀ (I : Set α), HasSubset.Subset I X → Iff (M.Indep I) (M'.Indep I)
    I : Set α
    hI : HasSubset.Subset I X
    ⊢ Iff ((M.restrict X).Indep I) ((M'.restrict X).Indep I)
  -/
  rw [restrict_indep_iff, and_iff_left hI, restrict_indep_iff, and_iff_left hI, h _ hI]
  /-
    🎉 no goals
  -/


@[simp] theorem restrict_eq_self_iff : M ↾ R = M ↔ R = M.E :=
              /-
                α : Type u_1
                M : Matroid α
                R : Set α
                h : Eq (M.restrict R) M
                ⊢ Eq R M.E
              -/
                        /-
                          🎉 no goals
                        -/
  ⟨fun h ↦ by rw [← h]; rfl, fun h ↦ by simp [h]⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- `Restriction N M` means that `N = M ↾ R` for some subset `R` of `M.E` -/
def Restriction (N M : Matroid α) : Prop := ∃ R ⊆ M.E, N = M ↾ R


/-- `StrictRestriction N M` means that `N = M ↾ R` for some strict subset `R` of `M.E` -/
def StrictRestriction (N M : Matroid α) : Prop := Restriction N M ∧ ¬ Restriction M N


/-- `N ≤r M` means that `N` is a `Restriction` of `M`. -/
scoped infix:50  " ≤r " => Restriction


/-- `N <r M` means that `N` is a `StrictRestriction` of `M`. -/
scoped infix:50  " <r " => StrictRestriction


/-- A type synonym for matroids with the restriction order.
  (The `PartialOrder` on `Matroid α` is reserved for the minor order)  -/
@[ext] structure Matroidᵣ (α : Type*) where ofMatroid ::
  /-- The underlying `Matroid`.-/
  toMatroid : Matroid α


instance {α : Type*} : CoeOut (Matroidᵣ α) (Matroid α) where
  coe := Matroidᵣ.toMatroid


@[simp] theorem Matroidᵣ.coe_inj {M₁ M₂ : Matroidᵣ α} :
    (M₁ : Matroid α) = (M₂ : Matroid α) ↔ M₁ = M₂ := by
  /-
    α : Type u_1
    M₁ M₂ : Matroid.Matroidᵣ α
    ⊢ Iff (Eq M₁.toMatroid M₂.toMatroid) (Eq M₁ M₂)
  -/
  cases M₁; cases M₂; simp
                      /-
                        🎉 no goals
                      -/


instance {α : Type*} : PartialOrder (Matroidᵣ α) where
  le := (· ≤r ·)
  le_refl M := ⟨(M : Matroid α).E, Subset.rfl, (M : Matroid α).restrict_ground_eq_self.symm⟩
  le_trans M₁ M₂ M₃ := by
    /-
      α✝ : Type u_1
      M : Matroid α✝
      R I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ M₃ : Matroid.Matroidᵣ α
      ⊢ LE.le M₁ M₂ → LE.le M₂ M₃ → LE.le M₁ M₃
    -/
    rintro ⟨R, hR, h₁⟩ ⟨R', hR', h₂⟩
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ M₃ : Matroid.Matroidᵣ α
      R : Set α
      hR : HasSubset.Subset R M₂.toMatroid.E
      h₁ : Eq M₁.toMatroid (M₂.toMatroid.restrict R)
      R' : Set α
      hR' : HasSubset.Subset R' M₃.toMatroid.E
      h₂ : Eq M₂.toMatroid (M₃.toMatroid.restrict R')
      ⊢ LE.le M₁ M₃
    -/
    change _ ≤r _
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ M₃ : Matroid.Matroidᵣ α
      R : Set α
      hR : HasSubset.Subset R M₂.toMatroid.E
      h₁ : Eq M₁.toMatroid (M₂.toMatroid.restrict R)
      R' : Set α
      hR' : HasSubset.Subset R' M₃.toMatroid.E
      h₂ : Eq M₂.toMatroid (M₃.toMatroid.restrict R')
      ⊢ M₁.toMatroid.Restriction M₃.toMatroid
    -/
    rw [h₂] at h₁ hR
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ M₃ : Matroid.Matroidᵣ α
      R R' : Set α
      hR : HasSubset.Subset R (M₃.toMatroid.restrict R').E
      h₁ : Eq M₁.toMatroid ((M₃.toMatroid.restrict R').restrict R)
      hR' : HasSubset.Subset R' M₃.toMatroid.E
      h₂ : Eq M₂.toMatroid (M₃.toMatroid.restrict R')
      ⊢ M₁.toMatroid.Restriction M₃.toMatroid
    -/
    rw [h₁, restrict_restrict_eq _ (show R ⊆ R' from hR)]
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ M₃ : Matroid.Matroidᵣ α
      R R' : Set α
      hR : HasSubset.Subset R (M₃.toMatroid.restrict R').E
      h₁ : Eq M₁.toMatroid ((M₃.toMatroid.restrict R').restrict R)
      hR' : HasSubset.Subset R' M₃.toMatroid.E
      h₂ : Eq M₂.toMatroid (M₃.toMatroid.restrict R')
      ⊢ (M₃.toMatroid.restrict R).Restriction M₃.toMatroid
    -/
    exact ⟨R, hR.trans hR', rfl⟩
    /-
      🎉 no goals
    -/
  le_antisymm M₁ M₂ := by
    /-
      α✝ : Type u_1
      M : Matroid α✝
      R I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ : Matroid.Matroidᵣ α
      ⊢ LE.le M₁ M₂ → LE.le M₂ M₁ → Eq M₁ M₂
    -/
    rintro ⟨R, hR, h⟩ ⟨R', hR', h'⟩
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ : Matroid.Matroidᵣ α
      R : Set α
      hR : HasSubset.Subset R M₂.toMatroid.E
      h : Eq M₁.toMatroid (M₂.toMatroid.restrict R)
      R' : Set α
      hR' : HasSubset.Subset R' M₁.toMatroid.E
      h' : Eq M₂.toMatroid (M₁.toMatroid.restrict R')
      ⊢ Eq M₁ M₂
    -/
    rw [h', restrict_ground_eq] at hR
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ : Matroid.Matroidᵣ α
      R : Set α
      h : Eq M₁.toMatroid (M₂.toMatroid.restrict R)
      R' : Set α
      hR : HasSubset.Subset R R'
      hR' : HasSubset.Subset R' M₁.toMatroid.E
      h' : Eq M₂.toMatroid (M₁.toMatroid.restrict R')
      ⊢ Eq M₁ M₂
    -/
    rw [h, restrict_ground_eq] at hR'
    /-
      case intro.intro.intro.intro
      α✝ : Type u_1
      M : Matroid α✝
      R✝ I X Y : Set α✝
      N : Matroid α✝
      α : Type u_2
      M₁ M₂ : Matroid.Matroidᵣ α
      R : Set α
      h : Eq M₁.toMatroid (M₂.toMatroid.restrict R)
      R' : Set α
      hR : HasSubset.Subset R R'
      hR' : HasSubset.Subset R' R
      h' : Eq M₂.toMatroid (M₁.toMatroid.restrict R')
      ⊢ Eq M₁ M₂
    -/
    rw [← Matroidᵣ.coe_inj, h, h', hR.antisymm hR', restrict_idem]
    /-
      🎉 no goals
    -/


@[simp] protected theorem Matroidᵣ.le_iff {M M' : Matroidᵣ α} :
    M ≤ M' ↔ (M : Matroid α) ≤r (M' : Matroid α) := Iff.rfl


@[simp] protected theorem Matroidᵣ.lt_iff {M M' : Matroidᵣ α} :
    M < M' ↔ (M : Matroid α) <r (M' : Matroid α) := Iff.rfl


theorem ofMatroid_le_iff {M M' : Matroid α} :
    Matroidᵣ.ofMatroid M ≤ Matroidᵣ.ofMatroid M' ↔ M ≤r M' := by
  /-
    α : Type u_1
    M M' : Matroid α
    ⊢ Iff (LE.le { toMatroid := M } { toMatroid := M' }) (M.Restriction M')
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ofMatroid_lt_iff {M M' : Matroid α} :
    Matroidᵣ.ofMatroid M < Matroidᵣ.ofMatroid M' ↔ M <r M' := by
  /-
    α : Type u_1
    M M' : Matroid α
    ⊢ Iff (LT.lt { toMatroid := M } { toMatroid := M' }) (M.StrictRestriction M')
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Restriction.refl : M ≤r M :=
  le_refl (Matroidᵣ.ofMatroid M)


theorem Restriction.antisymm {M' : Matroid α} (h : M ≤r M') (h' : M' ≤r M) : M = M' := by
  /-
    α : Type u_1
    M M' : Matroid α
    h : M.Restriction M'
    h' : M'.Restriction M
    ⊢ Eq M M'
  -/
  simpa using (ofMatroid_le_iff.2 h).antisymm (ofMatroid_le_iff.2 h')
  /-
    🎉 no goals
  -/


theorem Restriction.trans {M₁ M₂ M₃ : Matroid α} (h : M₁ ≤r M₂) (h' : M₂ ≤r M₃) : M₁ ≤r M₃ :=
  le_trans (α := Matroidᵣ α) h h'


theorem restrict_restriction (M : Matroid α) (R : Set α) (hR : R ⊆ M.E := by aesop_mat) :
    M ↾ R ≤r M :=
  ⟨R, hR, rfl⟩


theorem Restriction.eq_restrict (h : N ≤r M) : M ↾ N.E = N := by
  /-
    α : Type u_1
    M N : Matroid α
    h : N.Restriction M
    ⊢ Eq (M.restrict N.E) N
  -/
  obtain ⟨R, -, rfl⟩ := h; rw [restrict_ground_eq]
                           /-
                             🎉 no goals
                           -/


theorem Restriction.subset (h : N ≤r M) : N.E ⊆ M.E := by
  /-
    α : Type u_1
    M N : Matroid α
    h : N.Restriction M
    ⊢ HasSubset.Subset N.E M.E
  -/
  obtain ⟨R, hR, rfl⟩ := h; exact hR
                            /-
                              🎉 no goals
                            -/


theorem Restriction.exists_eq_restrict (h : N ≤r M) : ∃ R ⊆ M.E, N = M ↾ R :=
  h


theorem Restriction.of_subset {R' : Set α} (M : Matroid α) (h : R ⊆ R') : (M ↾ R) ≤r (M ↾ R') := by
  /-
    α : Type u_1
    R R' : Set α
    M : Matroid α
    h : HasSubset.Subset R R'
    ⊢ (M.restrict R).Restriction (M.restrict R')
  -/
  rw [← restrict_restrict_eq M h]; exact restrict_restriction _ _ h
                                   /-
                                     🎉 no goals
                                   -/


theorem restriction_iff_exists : (N ≤r M) ↔ ∃ R, R ⊆ M.E ∧ N = M ↾ R := by
  /-
    α : Type u_1
    M N : Matroid α
    ⊢ Iff (N.Restriction M) (Exists fun R => And (HasSubset.Subset R M.E) (Eq N (M …
  -/
  use Restriction.exists_eq_restrict; rintro ⟨R, hR, rfl⟩; exact restrict_restriction M R hR
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem StrictRestriction.restriction (h : N <r M) : N ≤r M :=
  h.1


theorem StrictRestriction.ne (h : N <r M) : N ≠ M := by
  /-
    α : Type u_1
    M N : Matroid α
    h : N.StrictRestriction M
    ⊢ Ne N M
  -/
  rintro rfl; rw [← ofMatroid_lt_iff] at h; simp at h
                                            /-
                                              🎉 no goals
                                            -/


theorem StrictRestriction.irrefl (M : Matroid α) : ¬ (M <r M) :=
  fun h ↦ h.ne rfl


theorem StrictRestriction.ssubset (h : N <r M) : N.E ⊂ M.E := by
  /-
    α : Type u_1
    M N : Matroid α
    h : N.StrictRestriction M
    ⊢ HasSSubset.SSubset N.E M.E
  -/
  obtain ⟨R, -, rfl⟩ := h.1
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    R : Set α
    h : (M.restrict R).StrictRestriction M
    ⊢ HasSSubset.SSubset (M.restrict R).E M.E
  -/
  refine h.restriction.subset.ssubset_of_ne (fun h' ↦ h.2 ⟨R, Subset.rfl, ?_⟩)
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    R : Set α
    h : (M.restrict R).StrictRestriction M
    h' : Eq (M.restrict R).E M.E
    ⊢ Eq M ((M.restrict R).restrict R)
  -/
  rw [show R = M.E from h', restrict_idem, restrict_ground_eq_self]
  /-
    🎉 no goals
  -/


theorem StrictRestriction.eq_restrict (h : N <r M) : M ↾ N.E = N :=
  h.restriction.eq_restrict


theorem StrictRestriction.exists_eq_restrict (h : N <r M) : ∃ R, R ⊂ M.E ∧ N = M ↾ R :=
                      /-
                        α : Type u_1
                        M N : Matroid α
                        h : N.StrictRestriction M
                        ⊢ Eq N (M.restrict N.E)
                      -/
  ⟨N.E, h.ssubset, by rw [h.eq_restrict]⟩
                      /-
                        🎉 no goals
                      -/


theorem Restriction.strictRestriction_of_ne (h : N ≤r M) (hne : N ≠ M) : N <r M :=
  ⟨h, fun h' ↦ hne <| h.antisymm h'⟩


theorem Restriction.eq_or_strictRestriction (h : N ≤r M) : N = M ∨ N <r M := by
  /-
    α : Type u_1
    M N : Matroid α
    h : N.Restriction M
    ⊢ Or (Eq N M) (N.StrictRestriction M)
  -/
  simpa using eq_or_lt_of_le (ofMatroid_le_iff.2 h)
  /-
    🎉 no goals
  -/


theorem restrict_strictRestriction {M : Matroid α} (hR : R ⊂ M.E) : M ↾ R <r M := by
  /-
    α : Type u_1
    R : Set α
    M : Matroid α
    hR : HasSSubset.SSubset R M.E
    ⊢ (M.restrict R).StrictRestriction M
  -/
  refine (M.restrict_restriction R hR.subset).strictRestriction_of_ne (fun h ↦ ?_)
  /-
    α : Type u_1
    R : Set α
    M : Matroid α
    hR : HasSSubset.SSubset R M.E
    h : Eq (M.restrict R) M
    ⊢ False
  -/
  rw [← h, restrict_ground_eq] at hR
  /-
    α : Type u_1
    R : Set α
    M : Matroid α
    hR : HasSSubset.SSubset R R
    h : Eq (M.restrict R) M
    ⊢ False
  -/
  exact hR.ne rfl
  /-
    🎉 no goals
  -/


theorem Restriction.strictRestriction_of_ground_ne (h : N ≤r M) (hne : N.E ≠ M.E) : N <r M := by
  /-
    α : Type u_1
    M N : Matroid α
    h : N.Restriction M
    hne : Ne N.E M.E
    ⊢ N.StrictRestriction M
  -/
  rw [← h.eq_restrict]
  /-
    α : Type u_1
    M N : Matroid α
    h : N.Restriction M
    hne : Ne N.E M.E
    ⊢ (M.restrict N.E).StrictRestriction M
  -/
  exact restrict_strictRestriction (h.subset.ssubset_of_ne hne)
  /-
    🎉 no goals
  -/


theorem StrictRestriction.of_ssubset {R' : Set α} (M : Matroid α) (h : R ⊂ R') :
    (M ↾ R) <r (M ↾ R') :=
  (Restriction.of_subset M h.subset).strictRestriction_of_ground_ne h.ne


theorem Restriction.finite {M : Matroid α} [M.Finite] (h : N ≤r M) : N.Finite := by
  /-
    α : Type u_1
    N M : Matroid α
    inst✝ : M.Finite
    h : N.Restriction M
    ⊢ N.Finite
  -/
  obtain ⟨R, hR, rfl⟩ := h
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    inst✝ : M.Finite
    R : Set α
    hR : HasSubset.Subset R M.E
    ⊢ (M.restrict R).Finite
  -/
  exact restrict_finite <| M.ground_finite.subset hR
  /-
    🎉 no goals
  -/


theorem Restriction.finiteRk {M : Matroid α} [FiniteRk M] (h : N ≤r M) : N.FiniteRk := by
  /-
    α : Type u_1
    N M : Matroid α
    inst✝ : M.FiniteRk
    h : N.Restriction M
    ⊢ N.FiniteRk
  -/
  obtain ⟨R, -, rfl⟩ := h
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    inst✝ : M.FiniteRk
    R : Set α
    ⊢ (M.restrict R).FiniteRk
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem Restriction.finitary {M : Matroid α} [Finitary M] (h : N ≤r M) : N.Finitary := by
  /-
    α : Type u_1
    N M : Matroid α
    inst✝ : M.Finitary
    h : N.Restriction M
    ⊢ N.Finitary
  -/
  obtain ⟨R, -, rfl⟩ := h
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    inst✝ : M.Finitary
    R : Set α
    ⊢ (M.restrict R).Finitary
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem finite_setOf_restriction (M : Matroid α) [M.Finite] : {N | N ≤r M}.Finite :=
  (M.ground_finite.finite_subsets.image (fun R ↦ M ↾ R)).subset <|
       /-
         α : Type u_1
         M : Matroid α
         inst✝ : M.Finite
         ⊢ HasSubset.Subset (setOf fun N => N.Restriction M) (Set.image (fun R => M.res …
       -/
    by rintro _ ⟨R, hR, rfl⟩; exact ⟨_, hR, rfl⟩
                              /-
                                🎉 no goals
                              -/


theorem Indep.of_restriction (hI : N.Indep I) (hNM : N ≤r M) : M.Indep I := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    N : Matroid α
    hI : N.Indep I
    hNM : N.Restriction M
    ⊢ M.Indep I
  -/
  obtain ⟨R, -, rfl⟩ := hNM; exact hI.of_restrict
                             /-
                               🎉 no goals
                             -/


theorem Indep.indep_restriction (hI : M.Indep I) (hNM : N ≤r M) (hIN : I ⊆ N.E) : N.Indep I := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    N : Matroid α
    hI : M.Indep I
    hNM : N.Restriction M
    hIN : HasSubset.Subset I N.E
    ⊢ N.Indep I
  -/
  obtain ⟨R, -, rfl⟩ := hNM; simpa [hI]
                             /-
                               🎉 no goals
                             -/


theorem Basis.basis_restriction (hI : M.Basis I X) (hNM : N ≤r M) (hX : X ⊆ N.E) : N.Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    N : Matroid α
    hI : M.Basis I X
    hNM : N.Restriction M
    hX : HasSubset.Subset X N.E
    ⊢ N.Basis I X
  -/
  obtain ⟨R, hR, rfl⟩ := hNM; rwa [basis_restrict_iff, and_iff_left (show X ⊆ R from hX)]
                              /-
                                🎉 no goals
                              -/


theorem Basis.of_restriction (hI : N.Basis I X) (hNM : N ≤r M) : M.Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    N : Matroid α
    hI : N.Basis I X
    hNM : N.Restriction M
    ⊢ M.Basis I X
  -/
  obtain ⟨R, hR, rfl⟩ := hNM; exact ((basis_restrict_iff hR).1 hI).1
                              /-
                                🎉 no goals
                              -/


theorem Base.basis_of_restriction (hI : N.Base I) (hNM : N ≤r M) : M.Basis I N.E := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    N : Matroid α
    hI : N.Base I
    hNM : N.Restriction M
    ⊢ M.Basis I N.E
  -/
  obtain ⟨R, hR, rfl⟩ := hNM; rwa [base_restrict_iff] at hI
                              /-
                                🎉 no goals
                              -/


theorem Dep.of_restriction (hX : N.Dep X) (hNM : N ≤r M) : M.Dep X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    N : Matroid α
    hX : N.Dep X
    hNM : N.Restriction M
    ⊢ M.Dep X
  -/
  obtain ⟨R, hR, rfl⟩ := hNM
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    X R : Set α
    hR : HasSubset.Subset R M.E
    hX : (M.restrict R).Dep X
    ⊢ M.Dep X
  -/
  rw [restrict_dep_iff] at hX
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    X R : Set α
    hR : HasSubset.Subset R M.E
    hX : And (Not (M.Indep X)) (HasSubset.Subset X R)
    ⊢ M.Dep X
  -/
  exact ⟨hX.1, hX.2.trans hR⟩
  /-
    🎉 no goals
  -/


theorem Dep.dep_restriction (hX : M.Dep X) (hNM : N ≤r M) (hXE : X ⊆ N.E := by aesop_mat) :
    N.Dep X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    N : Matroid α
    hX : M.Dep X
    hNM : N.Restriction M
    hXE : autoParam (HasSubset.Subset X N.E) _auto✝
    ⊢ N.Dep X
  -/
  obtain ⟨R, -, rfl⟩ := hNM; simpa [hX.not_indep]
                             /-
                               🎉 no goals
                             -/


theorem Basis.transfer (hIX : M.Basis I X) (hJX : M.Basis J X) (hXY : X ⊆ Y) (hJY : M.Basis J Y) :
    M.Basis I Y := by
  /-
    α : Type u_1
    M : Matroid α
    I X Y J : Set α
    hIX : M.Basis I X
    hJX : M.Basis J X
    hXY : HasSubset.Subset X Y
    hJY : M.Basis J Y
    ⊢ M.Basis I Y
  -/
  rw [← base_restrict_iff]; rw [← base_restrict_iff] at hJY
  /-
    α : Type u_1
    M : Matroid α
    I X Y J : Set α
    hIX : M.Basis I X
    hJX : M.Basis J X
    hXY : HasSubset.Subset X Y
    hJY : (M.restrict Y).Base J
    ⊢ (M.restrict Y).Base I
  -/
  exact hJY.base_of_basis_superset hJX.subset (hIX.basis_restrict_of_subset hXY)
  /-
    🎉 no goals
  -/


theorem Basis.basis_of_basis_of_subset_of_subset (hI : M.Basis I X) (hJ : M.Basis J Y) (hJX : J ⊆ X)
    (hIY : I ⊆ Y) : M.Basis I Y := by
  /-
    α : Type u_1
    M : Matroid α
    I X Y J : Set α
    hI : M.Basis I X
    hJ : M.Basis J Y
    hJX : HasSubset.Subset J X
    hIY : HasSubset.Subset I Y
    ⊢ M.Basis I Y
  -/
  have hI' := hI.basis_subset (subset_inter hI.subset hIY) inter_subset_left
  /-
    α : Type u_1
    M : Matroid α
    I X Y J : Set α
    hI : M.Basis I X
    hJ : M.Basis J Y
    hJX : HasSubset.Subset J X
    hIY : HasSubset.Subset I Y
    hI' : M.Basis I (Inter.inter X Y)
    ⊢ M.Basis I Y
  -/
  have hJ' := hJ.basis_subset (subset_inter hJX hJ.subset) inter_subset_right
  /-
    α : Type u_1
    M : Matroid α
    I X Y J : Set α
    hI : M.Basis I X
    hJ : M.Basis J Y
    hJX : HasSubset.Subset J X
    hIY : HasSubset.Subset I Y
    hI' : M.Basis I (Inter.inter X Y)
    hJ' : M.Basis J (Inter.inter X Y)
    ⊢ M.Basis I Y
  -/
  exact hI'.transfer hJ' inter_subset_right hJ
  /-
    🎉 no goals
  -/


theorem Indep.exists_basis_subset_union_basis (hI : M.Indep I) (hIX : I ⊆ X) (hJ : M.Basis J X) :
    ∃ I', M.Basis I' X ∧ I ⊆ I' ∧ I' ⊆ I ∪ J := by
  obtain ⟨I', hI', hII', hI'IJ⟩ :=
    (hI.indep_restrict_of_subset hIX).exists_base_subset_union_base (Basis.base_restrict hJ)
  /-
    case intro.intro.intro
    α : Type u_1
    M : Matroid α
    I X J : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hJ : M.Basis J X
    I' : Set α
    hI' : (M.restrict X).Base I'
    hII' : HasSubset.Subset I I'
    hI'IJ : HasSubset.Subset I' (Union.union I J)
    ⊢ Exists fun I' => And (M.Basis I' X) (And (HasSubset.Subset I I') (HasSubset. …
  -/
  rw [base_restrict_iff] at hI'
  /-
    case intro.intro.intro
    α : Type u_1
    M : Matroid α
    I X J : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hJ : M.Basis J X
    I' : Set α
    hI' : M.Basis I' X
    hII' : HasSubset.Subset I I'
    hI'IJ : HasSubset.Subset I' (Union.union I J)
    ⊢ Exists fun I' => And (M.Basis I' X) (And (HasSubset.Subset I I') (HasSubset. …
  -/
  exact ⟨I', hI', hII', hI'IJ⟩
  /-
    🎉 no goals
  -/


theorem Indep.exists_insert_of_not_basis (hI : M.Indep I) (hIX : I ⊆ X) (hI' : ¬M.Basis I X)
    (hJ : M.Basis J X) : ∃ e ∈ J \ I, M.Indep (insert e I) := by
  /-
    α : Type u_1
    M : Matroid α
    I X J : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hI' : Not (M.Basis I X)
    hJ : M.Basis J X
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff J I) e) (M.Indep (Insert.in …
  -/
  rw [← base_restrict_iff] at hI'; rw [← base_restrict_iff] at hJ
  /-
    α : Type u_1
    M : Matroid α
    I X J : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hI' : Not ((M.restrict X).Base I)
    hJ : (M.restrict X).Base J
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff J I) e) (M.Indep (Insert.in …
  -/
  obtain ⟨e, he, hi⟩ := (hI.indep_restrict_of_subset hIX).exists_insert_of_not_base hI' hJ
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I X J : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hI' : Not ((M.restrict X).Base I)
    hJ : (M.restrict X).Base J
    e : α
    he : Membership.mem (SDiff.sdiff J I) e
    hi : (M.restrict X).Indep (Insert.insert e I)
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff J I) e) (M.Indep (Insert.in …
  -/
  exact ⟨e, he, (restrict_indep_iff.mp hi).1⟩
  /-
    🎉 no goals
  -/


theorem Basis.base_of_base_subset (hIX : M.Basis I X) (hB : M.Base B) (hBX : B ⊆ X) : M.Base I :=
  hB.base_of_basis_superset hBX hIX


theorem Basis.exchange (hIX : M.Basis I X) (hJX : M.Basis J X) (he : e ∈ I \ J) :
    ∃ f ∈ J \ I, M.Basis (insert f (I \ {e})) X := by
  /-
    α : Type u_1
    M : Matroid α
    I X J : Set α
    e : α
    hIX : M.Basis I X
    hJX : M.Basis J X
    he : Membership.mem (SDiff.sdiff I J) e
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff J I) f) (M.Basis (Insert.in …
  -/
  obtain ⟨y,hy, h⟩ := hIX.restrict_base.exchange hJX.restrict_base he
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I X J : Set α
    e : α
    hIX : M.Basis I X
    hJX : M.Basis J X
    he : Membership.mem (SDiff.sdiff I J) e
    y : α
    hy : Membership.mem (SDiff.sdiff J I) y
    h : (M.restrict X).Base (Insert.insert y (SDiff.sdiff I (Singleton.singleton e …
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff J I) f) (M.Basis (Insert.in …
  -/
  exact ⟨y, hy, by rwa [base_restrict_iff] at h⟩
  /-
    🎉 no goals
  -/


theorem Basis.eq_exchange_of_diff_eq_singleton (hI : M.Basis I X) (hJ : M.Basis J X)
    (hIJ : I \ J = {e}) : ∃ f ∈ J \ I, J = insert f I \ {e} := by
  /-
    α : Type u_1
    M : Matroid α
    I X J : Set α
    e : α
    hI : M.Basis I X
    hJ : M.Basis J X
    hIJ : Eq (SDiff.sdiff I J) (Singleton.singleton e)
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff J I) f) (Eq J (SDiff.sdiff  …
  -/
  rw [← base_restrict_iff] at hI hJ; exact hI.eq_exchange_of_diff_eq_singleton hJ hIJ
                                     /-
                                       🎉 no goals
                                     -/


theorem Basis'.encard_eq_encard (hI : M.Basis' I X) (hJ : M.Basis' J X) : I.encard = J.encard := by
  /-
    α : Type u_1
    M : Matroid α
    I X J : Set α
    hI : M.Basis' I X
    hJ : M.Basis' J X
    ⊢ Eq I.encard J.encard
  -/
  rw [← base_restrict_iff'] at hI hJ; exact hI.card_eq_card_of_base hJ
                                      /-
                                        🎉 no goals
                                      -/


theorem Basis.encard_eq_encard (hI : M.Basis I X) (hJ : M.Basis J X) : I.encard = J.encard :=
  hI.basis'.encard_eq_encard hJ.basis'


/-- Any independent set can be extended into a larger independent set. -/
theorem Indep.augment (hI : M.Indep I) (hJ : M.Indep J) (hIJ : I.encard < J.encard) :
    ∃ e ∈ J \ I, M.Indep (insert e I) := by
  /-
    α : Type u_1
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJ : M.Indep J
    hIJ : LT.lt I.encard J.encard
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff J I) e) (M.Indep (Insert.in …
  -/
  by_contra! he
  have hb : M.Basis I (I ∪ J) := by
    simp_rw [hI.basis_iff_forall_insert_dep subset_union_left, union_diff_left, mem_diff,
      and_imp, dep_iff, insert_subset_iff, and_iff_left hI.subset_ground]
    exact fun e heJ heI ↦ ⟨he e ⟨heJ, heI⟩, hJ.subset_ground heJ⟩
  /-
    α : Type u_1
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJ : M.Indep J
    hIJ : LT.lt I.encard J.encard
    he : ∀ (e : α), Membership.mem (SDiff.sdiff J I) e → Not (M.Indep (Insert.inse …
    hb : M.Basis I (Union.union I J)
    ⊢ False
  -/
  obtain ⟨J', hJ', hJJ'⟩ := hJ.subset_basis_of_subset I.subset_union_right
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJ : M.Indep J
    hIJ : LT.lt I.encard J.encard
    he : ∀ (e : α), Membership.mem (SDiff.sdiff J I) e → Not (M.Indep (Insert.inse …
    hb : M.Basis I (Union.union I J)
    J' : Set α
    hJ' : M.Basis J' (Union.union I J)
    hJJ' : HasSubset.Subset J J'
    ⊢ False
  -/
  rw [← hJ'.encard_eq_encard hb] at hIJ
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I J : Set α
    hI : M.Indep I
    hJ : M.Indep J
    he : ∀ (e : α), Membership.mem (SDiff.sdiff J I) e → Not (M.Indep (Insert.inse …
    hb : M.Basis I (Union.union I J)
    J' : Set α
    hIJ : LT.lt J'.encard J.encard
    hJ' : M.Basis J' (Union.union I J)
    hJJ' : HasSubset.Subset J J'
    ⊢ False
  -/
  exact hIJ.not_le (encard_mono hJJ')
  /-
    🎉 no goals
  -/


