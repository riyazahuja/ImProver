/-- The `Matroid α` with empty ground set. -/
def emptyOn (α : Type*) : Matroid α where
  E := ∅
  Base := (· = ∅)
  Indep := (· = ∅)
                   /-
                     α✝ : Type u_1
                     M : Matroid α✝
                     E B I X R J : Set α✝
                     α : Type u_2
                     ⊢ ∀ ⦃I : Set α⦄, Iff ((fun x => Eq x EmptyCollection.emptyCollection) I) (Exis …
                   -/
  indep_iff' := by simp [subset_empty_iff]
                   /-
                     🎉 no goals
                   -/
  exists_base := ⟨∅, rfl⟩
                      /-
                        α✝ : Type u_1
                        M : Matroid α✝
                        E B I X R J : Set α✝
                        α : Type u_2
                        ⊢ Matroid.ExchangeProperty fun x => Eq x EmptyCollection.emptyCollection
                      -/
  base_exchange := by rintro _ _ rfl; simp
                                      /-
                                        🎉 no goals
                                      -/
                   /-
                     α✝ : Type u_1
                     M : Matroid α✝
                     E B I X R J : Set α✝
                     α : Type u_2
                     ⊢ ∀ (X : Set α), HasSubset.Subset X EmptyCollection.emptyCollection → Matroid. …
                   -/
  maximality := by rintro _ _ _ rfl -; exact ⟨∅, by simp [Maximal]⟩
                                       /-
                                         🎉 no goals
                                       -/
                      /-
                        α✝ : Type u_1
                        M : Matroid α✝
                        E B I X R J : Set α✝
                        α : Type u_2
                        ⊢ ∀ (B : Set α), (fun x => Eq x EmptyCollection.emptyCollection) B → HasSubset …
                      -/
  subset_ground := by simp
                      /-
                        🎉 no goals
                      -/


@[simp] theorem emptyOn_ground : (emptyOn α).E = ∅ := rfl


@[simp] theorem emptyOn_base_iff : (emptyOn α).Base B ↔ B = ∅ := Iff.rfl


@[simp] theorem emptyOn_indep_iff : (emptyOn α).Indep I ↔ I = ∅ := Iff.rfl


theorem ground_eq_empty_iff : (M.E = ∅) ↔ M = emptyOn α := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Iff (Eq M.E EmptyCollection.emptyCollection) (Eq M (Matroid.emptyOn α))
  -/
  simp only [emptyOn, ext_iff_indep, iff_self_and]
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq M.E EmptyCollection.emptyCollection → ∀ ⦃I : Set α⦄, HasSubset.Subset I M …
  -/
  exact fun h ↦ by simp [h, subset_empty_iff]
  /-
    🎉 no goals
  -/


@[simp] theorem emptyOn_dual_eq : (emptyOn α)✶ = emptyOn α := by
  /-
    α : Type u_1
    ⊢ Eq (Matroid.emptyOn α).dual (Matroid.emptyOn α)
  -/
  rw [← ground_eq_empty_iff]; rfl
                              /-
                                🎉 no goals
                              -/


@[simp] theorem restrict_empty (M : Matroid α) : M ↾ (∅ : Set α) = emptyOn α := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq (M.restrict EmptyCollection.emptyCollection) (Matroid.emptyOn α)
  -/
  simp [← ground_eq_empty_iff]
  /-
    🎉 no goals
  -/


theorem eq_emptyOn_or_nonempty (M : Matroid α) : M = emptyOn α ∨ Matroid.Nonempty M := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Or (Eq M (Matroid.emptyOn α)) M.Nonempty
  -/
  rw [← ground_eq_empty_iff]
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Or (Eq M.E EmptyCollection.emptyCollection) M.Nonempty
  -/
  exact M.E.eq_empty_or_nonempty.elim Or.inl (fun h ↦ Or.inr ⟨h⟩)
  /-
    🎉 no goals
  -/


theorem eq_emptyOn [IsEmpty α] (M : Matroid α) : M = emptyOn α := by
  /-
    α : Type u_1
    inst✝ : IsEmpty α
    M : Matroid α
    ⊢ Eq M (Matroid.emptyOn α)
  -/
  rw [← ground_eq_empty_iff]
  /-
    α : Type u_1
    inst✝ : IsEmpty α
    M : Matroid α
    ⊢ Eq M.E EmptyCollection.emptyCollection
  -/
  exact M.E.eq_empty_of_isEmpty
  /-
    🎉 no goals
  -/


instance finite_emptyOn (α : Type*) : (emptyOn α).Finite :=
  ⟨finite_empty⟩


/-- The `Matroid α` with ground set `E` whose only base is `∅` -/
def loopyOn (E : Set α) : Matroid α := emptyOn α ↾ E


@[simp] theorem loopyOn_ground (E : Set α) : (loopyOn E).E = E := rfl


@[simp] theorem loopyOn_empty (α : Type*) : loopyOn (∅ : Set α) = emptyOn α := by
  /-
    α : Type u_2
    ⊢ Eq (Matroid.loopyOn EmptyCollection.emptyCollection) (Matroid.emptyOn α)
  -/
  rw [← ground_eq_empty_iff, loopyOn_ground]
  /-
    🎉 no goals
  -/


@[simp] theorem loopyOn_indep_iff : (loopyOn E).Indep I ↔ I = ∅ := by
  /-
    α : Type u_1
    E I : Set α
    ⊢ Iff ((Matroid.loopyOn E).Indep I) (Eq I EmptyCollection.emptyCollection)
  -/
  simp only [loopyOn, restrict_indep_iff, emptyOn_indep_iff, and_iff_left_iff_imp]
  /-
    α : Type u_1
    E I : Set α
    ⊢ Eq I EmptyCollection.emptyCollection → HasSubset.Subset I E
  -/
  rintro rfl; apply empty_subset
              /-
                🎉 no goals
              -/


theorem eq_loopyOn_iff : M = loopyOn E ↔ M.E = E ∧ ∀ X ⊆ M.E, M.Indep X → X = ∅ := by
  /-
    α : Type u_1
    M : Matroid α
    E : Set α
    ⊢ Iff (Eq M (Matroid.loopyOn E)) (And (Eq M.E E) (∀ (X : Set α), HasSubset.Sub …
  -/
  simp only [ext_iff_indep, loopyOn_ground, loopyOn_indep_iff, and_congr_right_iff]
  /-
    α : Type u_1
    M : Matroid α
    E : Set α
    ⊢ Eq M.E E → Iff (∀ ⦃I : Set α⦄, HasSubset.Subset I M.E → Iff (M.Indep I) (Eq  …
  -/
  rintro rfl
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Iff (∀ ⦃I : Set α⦄, HasSubset.Subset I M.E → Iff (M.Indep I) (Eq I EmptyColl …
  -/
  refine ⟨fun h I hI ↦ (h hI).1, fun h I hIE ↦ ⟨h I hIE, by rintro rfl; simp⟩⟩
  /-
    🎉 no goals
  -/


@[simp] theorem loopyOn_base_iff : (loopyOn E).Base B ↔ B = ∅ := by
  /-
    α : Type u_1
    E B : Set α
    ⊢ Iff ((Matroid.loopyOn E).Base B) (Eq B EmptyCollection.emptyCollection)
  -/
  simp [Maximal, base_iff_maximal_indep]
  /-
    🎉 no goals
  -/


@[simp] theorem loopyOn_basis_iff : (loopyOn E).Basis I X ↔ I = ∅ ∧ X ⊆ E :=
  ⟨fun h ↦ ⟨loopyOn_indep_iff.mp h.indep, h.subset_ground⟩,
       /-
         α : Type u_1
         E I X : Set α
         ⊢ And (Eq I EmptyCollection.emptyCollection) (HasSubset.Subset X E) → (Matroid …
       -/
    by rintro ⟨rfl, hX⟩; rw [basis_iff]; simp⟩
                                         /-
                                           🎉 no goals
                                         -/


instance : FiniteRk (loopyOn E) :=
  ⟨⟨∅, loopyOn_base_iff.2 rfl, finite_empty⟩⟩


theorem Finite.loopyOn_finite (hE : E.Finite) : Matroid.Finite (loopyOn E) :=
  ⟨hE⟩


@[simp] theorem loopyOn_restrict (E R : Set α) : (loopyOn E) ↾ R = loopyOn R := by
  /-
    α : Type u_1
    E R : Set α
    ⊢ Eq ((Matroid.loopyOn E).restrict R) (Matroid.loopyOn R)
  -/
  refine ext_indep rfl ?_
  /-
    α : Type u_1
    E R : Set α
    ⊢ ∀ ⦃I : Set α⦄, HasSubset.Subset I ((Matroid.loopyOn E).restrict R).E → Iff ( …
  -/
  simp only [restrict_ground_eq, restrict_indep_iff, loopyOn_indep_iff, and_iff_left_iff_imp]
  /-
    α : Type u_1
    E R : Set α
    ⊢ ∀ ⦃I : Set α⦄, HasSubset.Subset I R → Eq I EmptyCollection.emptyCollection → …
  -/
  exact fun _ h _ ↦ h
  /-
    🎉 no goals
  -/


theorem empty_base_iff : M.Base ∅ ↔ M = loopyOn M.E := by
  simp only [base_iff_maximal_indep, Maximal, empty_indep, le_eq_subset, empty_subset,
    subset_empty_iff, true_implies, true_and, ext_iff_indep, loopyOn_ground,
    loopyOn_indep_iff]
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Iff (∀ ⦃y : Set α⦄, M.Indep y → Eq y EmptyCollection.emptyCollection) (∀ ⦃I  …
  -/
  exact ⟨fun h I _ ↦ ⟨@h _, fun hI ↦ by simp [hI]⟩, fun h I hI ↦ (h hI.subset_ground).1 hI⟩
  /-
    🎉 no goals
  -/


theorem eq_loopyOn_or_rkPos (M : Matroid α) : M = loopyOn M.E ∨ RkPos M := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Or (Eq M (Matroid.loopyOn M.E)) M.RkPos
  -/
  rw [← empty_base_iff, rkPos_iff_empty_not_base]; apply em
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem not_rkPos_iff : ¬RkPos M ↔ M = loopyOn M.E := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Iff (Not M.RkPos) (Eq M (Matroid.loopyOn M.E))
  -/
  rw [rkPos_iff_empty_not_base, not_iff_comm, empty_base_iff]
  /-
    🎉 no goals
  -/


/-- The `Matroid α` with ground set `E` whose only base is `E`. -/
def freeOn (E : Set α) : Matroid α := (loopyOn E)✶


@[simp] theorem freeOn_ground : (freeOn E).E = E := rfl


@[simp] theorem freeOn_dual_eq : (freeOn E)✶ = loopyOn E := by
  /-
    α : Type u_1
    E : Set α
    ⊢ Eq (Matroid.freeOn E).dual (Matroid.loopyOn E)
  -/
  rw [freeOn, dual_dual]
  /-
    🎉 no goals
  -/


@[simp] theorem loopyOn_dual_eq : (loopyOn E)✶ = freeOn E := rfl


@[simp] theorem freeOn_empty (α : Type*) : freeOn (∅ : Set α) = emptyOn α := by
  /-
    α : Type u_2
    ⊢ Eq (Matroid.freeOn EmptyCollection.emptyCollection) (Matroid.emptyOn α)
  -/
  simp [freeOn]
  /-
    🎉 no goals
  -/


@[simp] theorem freeOn_base_iff : (freeOn E).Base B ↔ B = E := by
  simp only [freeOn, loopyOn_ground, dual_base_iff', loopyOn_base_iff, diff_eq_empty,
    ← subset_antisymm_iff, eq_comm (a := E)]


@[simp] theorem freeOn_indep_iff : (freeOn E).Indep I ↔ I ⊆ E := by
  /-
    α : Type u_1
    E I : Set α
    ⊢ Iff ((Matroid.freeOn E).Indep I) (HasSubset.Subset I E)
  -/
  simp [indep_iff]
  /-
    🎉 no goals
  -/


theorem freeOn_indep (hIE : I ⊆ E) : (freeOn E).Indep I :=
  freeOn_indep_iff.2 hIE


@[simp] theorem freeOn_basis_iff : (freeOn E).Basis I X ↔ I = X ∧ X ⊆ E := by
  /-
    α : Type u_1
    E I X : Set α
    ⊢ Iff ((Matroid.freeOn E).Basis I X) (And (Eq I X) (HasSubset.Subset X E))
  -/
  use fun h ↦ ⟨(freeOn_indep h.subset_ground).eq_of_basis h ,h.subset_ground⟩
  /-
    case mpr
    α : Type u_1
    E I X : Set α
    ⊢ And (Eq I X) (HasSubset.Subset X E) → (Matroid.freeOn E).Basis I X
  -/
  rintro ⟨rfl, hIE⟩
  /-
    case mpr.intro
    α : Type u_1
    E I : Set α
    hIE : HasSubset.Subset I E
    ⊢ (Matroid.freeOn E).Basis I I
  -/
  exact (freeOn_indep hIE).basis_self
  /-
    🎉 no goals
  -/


@[simp] theorem freeOn_basis'_iff : (freeOn E).Basis' I X ↔ I = X ∩ E := by
  rw [basis'_iff_basis_inter_ground, freeOn_basis_iff, freeOn_ground,
    and_iff_left inter_subset_right]


theorem eq_freeOn_iff : M = freeOn E ↔ M.E = E ∧ M.Indep E := by
  /-
    α : Type u_1
    M : Matroid α
    E : Set α
    ⊢ Iff (Eq M (Matroid.freeOn E)) (And (Eq M.E E) (M.Indep E))
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      E : Set α
      ⊢ Eq M (Matroid.freeOn E) → And (Eq M.E E) (M.Indep E)
    -/
  · rintro rfl; simp [Subset.rfl]
                /-
                  🎉 no goals
                -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    E : Set α
    h : And (Eq M.E E) (M.Indep E)
    ⊢ Eq M (Matroid.freeOn E)
  -/
  simp only [ext_iff_indep, freeOn_ground, freeOn_indep_iff, h.1, true_and]
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    E : Set α
    h : And (Eq M.E E) (M.Indep E)
    ⊢ ∀ ⦃I : Set α⦄, HasSubset.Subset I E → Iff (M.Indep I) (HasSubset.Subset I E)
  -/
  exact fun I hIX ↦ iff_of_true (h.2.subset hIX) hIX
  /-
    🎉 no goals
  -/


theorem ground_indep_iff_eq_freeOn : M.Indep M.E ↔ M = freeOn M.E := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Iff (M.Indep M.E) (Eq M (Matroid.freeOn M.E))
  -/
  simp [eq_freeOn_iff]
  /-
    🎉 no goals
  -/


theorem freeOn_restrict (h : R ⊆ E) : (freeOn E) ↾ R = freeOn R := by
  /-
    α : Type u_1
    E R : Set α
    h : HasSubset.Subset R E
    ⊢ Eq ((Matroid.freeOn E).restrict R) (Matroid.freeOn R)
  -/
  simp [h, eq_freeOn_iff, Subset.rfl]
  /-
    🎉 no goals
  -/


theorem restrict_eq_freeOn_iff : M ↾ I = freeOn I ↔ M.Indep I := by
  rw [eq_freeOn_iff, and_iff_right M.restrict_ground_eq, restrict_indep_iff,
    and_iff_left Subset.rfl]


theorem Indep.restrict_eq_freeOn (hI : M.Indep I) : M ↾ I = freeOn I := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    ⊢ Eq (M.restrict I) (Matroid.freeOn I)
  -/
  rwa [restrict_eq_freeOn_iff]
  /-
    🎉 no goals
  -/


/-- The matroid on `E` whose unique base is the subset `I` of `E`.
Intended for use when `I ⊆ E`; if this not not the case, then the base is `I ∩ E`. -/
def uniqueBaseOn (I E : Set α) : Matroid α := freeOn I ↾ E


@[simp] theorem uniqueBaseOn_ground : (uniqueBaseOn I E).E = E :=
  rfl


theorem uniqueBaseOn_base_iff (hIE : I ⊆ E) : (uniqueBaseOn I E).Base B ↔ B = I := by
  /-
    α : Type u_1
    E B I : Set α
    hIE : HasSubset.Subset I E
    ⊢ Iff ((Matroid.uniqueBaseOn I E).Base B) (Eq B I)
  -/
  rw [uniqueBaseOn, base_restrict_iff', freeOn_basis'_iff, inter_eq_self_of_subset_right hIE]
  /-
    🎉 no goals
  -/


theorem uniqueBaseOn_inter_ground_eq (I E : Set α) :
    uniqueBaseOn (I ∩ E) E = uniqueBaseOn I E := by
  simp only [uniqueBaseOn, restrict_eq_restrict_iff, freeOn_indep_iff, subset_inter_iff,
    iff_self_and]
  /-
    α : Type u_1
    I E : Set α
    ⊢ ∀ (I_1 : Set α), HasSubset.Subset I_1 E → Iff (And (HasSubset.Subset I_1 I)  …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp] theorem uniqueBaseOn_indep_iff' : (uniqueBaseOn I E).Indep J ↔ J ⊆ I ∩ E := by
  /-
    α : Type u_1
    E I J : Set α
    ⊢ Iff ((Matroid.uniqueBaseOn I E).Indep J) (HasSubset.Subset J (Inter.inter I  …
  -/
  rw [uniqueBaseOn, restrict_indep_iff, freeOn_indep_iff, subset_inter_iff]
  /-
    🎉 no goals
  -/


theorem uniqueBaseOn_indep_iff (hIE : I ⊆ E) : (uniqueBaseOn I E).Indep J ↔ J ⊆ I := by
  /-
    α : Type u_1
    E I J : Set α
    hIE : HasSubset.Subset I E
    ⊢ Iff ((Matroid.uniqueBaseOn I E).Indep J) (HasSubset.Subset J I)
  -/
  rw [uniqueBaseOn, restrict_indep_iff, freeOn_indep_iff, and_iff_left_iff_imp]
  /-
    α : Type u_1
    E I J : Set α
    hIE : HasSubset.Subset I E
    ⊢ HasSubset.Subset J I → HasSubset.Subset J E
  -/
  exact fun h ↦ h.trans hIE
  /-
    🎉 no goals
  -/


theorem uniqueBaseOn_basis_iff (hX : X ⊆ E) : (uniqueBaseOn I E).Basis J X ↔ J = X ∩ I := by
  /-
    α : Type u_1
    E I X J : Set α
    hX : HasSubset.Subset X E
    ⊢ Iff ((Matroid.uniqueBaseOn I E).Basis J X) (Eq J (Inter.inter X I))
  -/
  rw [basis_iff_maximal]
  exact maximal_iff_eq (by simp [inter_subset_left.trans hX])
    (by simp (config := {contextual := true}))


theorem uniqueBaseOn_inter_basis (hX : X ⊆ E) : (uniqueBaseOn I E).Basis (X ∩ I) X := by
  /-
    α : Type u_1
    E I X : Set α
    hX : HasSubset.Subset X E
    ⊢ (Matroid.uniqueBaseOn I E).Basis (Inter.inter X I) X
  -/
  rw [uniqueBaseOn_basis_iff hX]
  /-
    🎉 no goals
  -/


@[simp] theorem uniqueBaseOn_dual_eq (I E : Set α) :
    (uniqueBaseOn I E)✶ = uniqueBaseOn (E \ I) E := by
  /-
    α : Type u_1
    I E : Set α
    ⊢ Eq (Matroid.uniqueBaseOn I E).dual (Matroid.uniqueBaseOn (SDiff.sdiff E I) E)
  -/
  rw [← uniqueBaseOn_inter_ground_eq]
  /-
    α : Type u_1
    I E : Set α
    ⊢ Eq (Matroid.uniqueBaseOn (Inter.inter I E) E).dual (Matroid.uniqueBaseOn (SD …
  -/
  refine ext_base rfl (fun B (hB : B ⊆ E) ↦ ?_)
  rw [dual_base_iff, uniqueBaseOn_base_iff inter_subset_right, uniqueBaseOn_base_iff diff_subset,
    uniqueBaseOn_ground]
  exact ⟨fun h ↦ by rw [← diff_diff_cancel_left hB, h, diff_inter_self_eq_diff],
    fun h ↦ by rw [h, inter_comm I]; simp⟩


@[simp] theorem uniqueBaseOn_self (I : Set α) : uniqueBaseOn I I = freeOn I := by
  /-
    α : Type u_1
    I : Set α
    ⊢ Eq (Matroid.uniqueBaseOn I I) (Matroid.freeOn I)
  -/
  rw [uniqueBaseOn, freeOn_restrict rfl.subset]
  /-
    🎉 no goals
  -/


@[simp] theorem uniqueBaseOn_empty (I : Set α) : uniqueBaseOn ∅ I = loopyOn I := by
  /-
    α : Type u_1
    I : Set α
    ⊢ Eq (Matroid.uniqueBaseOn EmptyCollection.emptyCollection I) (Matroid.loopyOn …
  -/
  rw [← dual_inj, uniqueBaseOn_dual_eq, diff_empty, uniqueBaseOn_self, loopyOn_dual_eq]
  /-
    🎉 no goals
  -/


theorem uniqueBaseOn_restrict' (I E R : Set α) :
    (uniqueBaseOn I E) ↾ R = uniqueBaseOn (I ∩ R ∩ E) R := by
  simp_rw [ext_iff_indep, restrict_ground_eq, uniqueBaseOn_ground, true_and,
    restrict_indep_iff, uniqueBaseOn_indep_iff', subset_inter_iff]
  /-
    α : Type u_1
    I E R : Set α
    ⊢ ∀ ⦃I_1 : Set α⦄, HasSubset.Subset I_1 R → Iff (And (And (HasSubset.Subset I_ …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem uniqueBaseOn_restrict (h : I ⊆ E) (R : Set α) :
    (uniqueBaseOn I E) ↾ R = uniqueBaseOn (I ∩ R) R := by
  /-
    α : Type u_1
    E I : Set α
    h : HasSubset.Subset I E
    R : Set α
    ⊢ Eq ((Matroid.uniqueBaseOn I E).restrict R) (Matroid.uniqueBaseOn (Inter.inte …
  -/
  rw [uniqueBaseOn_restrict', inter_right_comm, inter_eq_self_of_subset_left h]
  /-
    🎉 no goals
  -/


