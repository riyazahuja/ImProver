lemma preimage_val_eq_univ_of_subset (h : A ⊆ B) : A ↓∩ B = univ := by
  /-
    α : Type u_2
    A B : Set α
    h : HasSubset.Subset A B
    ⊢ Eq (Set.preimage Subtype.val B) Set.univ
  -/
  rw [eq_univ_iff_forall, Subtype.forall]
  /-
    α : Type u_2
    A B : Set α
    h : HasSubset.Subset A B
    ⊢ ∀ (a : α) (b : Membership.mem A a), Membership.mem (Set.preimage Subtype.val …
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma preimage_val_sUnion : A ↓∩ (⋃₀ S) = ⋃₀ { (A ↓∩ B) | B ∈ S } := by
  /-
    α : Type u_2
    A : Set α
    S : Set (Set α)
    ⊢ Eq (Set.preimage Subtype.val S.sUnion) (setOf fun x => Exists fun B => And ( …
  -/
  erw [sUnion_image]
  /-
    α : Type u_2
    A : Set α
    S : Set (Set α)
    ⊢ Eq (Set.preimage Subtype.val S.sUnion) (Set.iUnion fun a => Set.iUnion fun h …
  -/
  simp_rw [sUnion_eq_biUnion, preimage_iUnion]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_val_iInter : A ↓∩ (⋂ i, s i) = ⋂ i, A ↓∩ s i := preimage_iInter


lemma preimage_val_sInter : A ↓∩ (⋂₀ S) = ⋂₀ { (A ↓∩ B) | B ∈ S } := by
  /-
    α : Type u_2
    A : Set α
    S : Set (Set α)
    ⊢ Eq (Set.preimage Subtype.val S.sInter) (setOf fun x => Exists fun B => And ( …
  -/
  erw [sInter_image]
  /-
    α : Type u_2
    A : Set α
    S : Set (Set α)
    ⊢ Eq (Set.preimage Subtype.val S.sInter) (Set.iInter fun a => Set.iInter fun h …
  -/
  simp_rw [sInter_eq_biInter, preimage_iInter]
  /-
    🎉 no goals
  -/


lemma preimage_val_sInter_eq_sInter : A ↓∩ (⋂₀ S) = ⋂₀ ((A ↓∩ ·) '' S) := by
  /-
    α : Type u_2
    A : Set α
    S : Set (Set α)
    ⊢ Eq (Set.preimage Subtype.val S.sInter) (Set.image (fun x => Set.preimage Sub …
  -/
  simp only [preimage_sInter, sInter_image]
  /-
    🎉 no goals
  -/


lemma eq_of_preimage_val_eq_of_subset (hB : B ⊆ A) (hC : C ⊆ A) (h : A ↓∩ B = A ↓∩ C) : B = C := by
  /-
    α : Type u_2
    A B C : Set α
    hB : HasSubset.Subset B A
    hC : HasSubset.Subset C A
    h : Eq (Set.preimage Subtype.val B) (Set.preimage Subtype.val C)
    ⊢ Eq B C
  -/
  simp only [← inter_eq_right] at hB hC
  /-
    α : Type u_2
    A B C : Set α
    h : Eq (Set.preimage Subtype.val B) (Set.preimage Subtype.val C)
    hB : Eq (Inter.inter A B) B
    hC : Eq (Inter.inter A C) C
    ⊢ Eq B C
  -/
  simp only [Subtype.preimage_val_eq_preimage_val_iff, hB, hC] at h
  /-
    α : Type u_2
    A B C : Set α
    hB : Eq (Inter.inter A B) B
    hC : Eq (Inter.inter A C) C
    h : Eq B C
    ⊢ Eq B C
  -/
  exact h
  /-
    🎉 no goals
  -/


@[simp]
lemma image_val_union : (↑(D ∪ E) : Set α) = ↑D ∪ ↑E := image_union _ _ _


@[simp]
lemma image_val_inter : (↑(D ∩ E) : Set α) = ↑D ∩ ↑E := image_inter Subtype.val_injective


@[simp]
lemma image_val_diff : (↑(D \ E) : Set α) = ↑D \ ↑E := image_diff Subtype.val_injective _ _


@[simp]
lemma image_val_compl : ↑(Dᶜ) = A \ ↑D := by
  /-
    α : Type u_2
    A : Set α
    D : Set ↑A
    ⊢ Eq (Set.image Subtype.val (HasCompl.compl D)) (SDiff.sdiff A (Set.image Subt …
  -/
  rw [compl_eq_univ_diff, image_val_diff, image_univ, Subtype.range_coe_subtype, setOf_mem_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma image_val_sUnion : ↑(⋃₀ T) = ⋃₀ { (B : Set α) | B ∈ T} := by
  /-
    α : Type u_2
    A : Set α
    T : Set (Set ↑A)
    ⊢ Eq (Set.image Subtype.val T.sUnion) (setOf fun x => Exists fun B => And (Mem …
  -/
  rw [image_sUnion, image]
  /-
    🎉 no goals
  -/


@[simp]
lemma image_val_iUnion : ↑(⋃ i, t i) = ⋃ i, (t i : Set α) := image_iUnion


@[simp]
lemma image_val_sInter (hT : T.Nonempty) : (↑(⋂₀ T) : Set α) = ⋂₀ { (↑B : Set α) | B ∈ T } := by
  /-
    α : Type u_2
    A : Set α
    T : Set (Set ↑A)
    hT : T.Nonempty
    ⊢ Eq (Set.image Subtype.val T.sInter) (setOf fun x => Exists fun B => And (Mem …
  -/
  erw [sInter_image]
  /-
    α : Type u_2
    A : Set α
    T : Set (Set ↑A)
    hT : T.Nonempty
    ⊢ Eq (Set.image Subtype.val T.sInter) (Set.iInter fun a => Set.iInter fun h => …
  -/
  rw [sInter_eq_biInter, Subtype.val_injective.injOn.image_biInter_eq hT]
  /-
    🎉 no goals
  -/


@[simp]
lemma image_val_iInter [Nonempty ι] : (↑(⋂ i, t i) : Set α) = ⋂ i, (↑(t i) : Set α) :=
  Subtype.val_injective.injOn.image_iInter_eq


@[simp]
lemma image_val_union_self_right_eq : A ∪ ↑D = A :=
  union_eq_left.2 image_val_subset


@[simp]
lemma image_val_union_self_left_eq : ↑D ∪ A = A :=
  union_eq_right.2 image_val_subset


@[simp]
lemma image_val_inter_self_right_eq_coe : A ∩ ↑D = ↑D :=
  inter_eq_right.2 image_val_subset

@[deprecated (since := "2024-10-25")]
alias cou_inter_self_right_eq_coe := image_val_inter_self_right_eq_coe


@[simp]
lemma image_val_inter_self_left_eq_coe : ↑D ∩ A = ↑D :=
  inter_eq_left.2 image_val_subset


lemma subset_preimage_val_image_val_iff : D ⊆ A ↓∩ ↑E ↔ D ⊆ E := by
  /-
    α : Type u_2
    A : Set α
    D E : Set ↑A
    ⊢ Iff (HasSubset.Subset D (Set.preimage Subtype.val (Set.image Subtype.val E)) …
  -/
  rw [preimage_image_eq _ Subtype.val_injective]
  /-
    🎉 no goals
  -/


@[simp]
lemma image_val_inj : (D : Set α) = ↑E ↔ D = E := Subtype.val_injective.image_injective.eq_iff


lemma image_val_injective : Function.Injective ((↑) : Set A → Set α) :=
  Subtype.val_injective.image_injective


lemma subset_of_image_val_subset_image_val (h : (↑D : Set α) ⊆ ↑E) : D ⊆ E :=
  (image_subset_image_iff Subtype.val_injective).1 h


@[mono]
lemma image_val_mono (h : D ⊆ E) : (↑D : Set α) ⊆ ↑E :=
  (image_subset_image_iff Subtype.val_injective).2 h


lemma image_val_preimage_val_subset_self : ↑(A ↓∩ B) ⊆ B :=
  image_preimage_subset _ _


lemma preimage_val_image_val_eq_self : A ↓∩ ↑D = D :=
  Function.Injective.preimage_image Subtype.val_injective _


