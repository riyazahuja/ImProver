@[simp]
theorem Iic_disjoint_Ioi (h : a ≤ b) : Disjoint (Iic a) (Ioi b) :=
  disjoint_left.mpr fun _ ha hb => (h.trans_lt hb).not_le ha


@[simp]
theorem Iio_disjoint_Ici (h : a ≤ b) : Disjoint (Iio a) (Ici b) :=
  disjoint_left.mpr fun _ ha hb => (h.trans_lt' ha).not_le hb


@[simp]
theorem Iic_disjoint_Ioc (h : a ≤ b) : Disjoint (Iic a) (Ioc b c) :=
  (Iic_disjoint_Ioi h).mono le_rfl Ioc_subset_Ioi_self


@[simp]
theorem Ioc_disjoint_Ioc_same : Disjoint (Ioc a b) (Ioc b c) :=
  (Iic_disjoint_Ioc le_rfl).mono Ioc_subset_Iic_self le_rfl


@[simp]
theorem Ico_disjoint_Ico_same : Disjoint (Ico a b) (Ico b c) :=
  disjoint_left.mpr fun _ hab hbc => hab.2.not_le hbc.1


@[simp]
theorem Ici_disjoint_Iic : Disjoint (Ici a) (Iic b) ↔ ¬a ≤ b := by
  /-
    α : Type v
    inst✝ : Preorder α
    a b : α
    ⊢ Iff (Disjoint (Set.Ici a) (Set.Iic b)) (Not (LE.le a b))
  -/
  rw [Set.disjoint_iff_inter_eq_empty, Ici_inter_Iic, Icc_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem Iic_disjoint_Ici : Disjoint (Iic a) (Ici b) ↔ ¬b ≤ a :=
  disjoint_comm.trans Ici_disjoint_Iic


@[simp]
theorem Ioc_disjoint_Ioi (h : b ≤ c) : Disjoint (Ioc a b) (Ioi c) :=
  disjoint_left.mpr (fun _ hx hy ↦ (hx.2.trans h).not_lt hy)


theorem Ioc_disjoint_Ioi_same : Disjoint (Ioc a b) (Ioi b) :=
  Ioc_disjoint_Ioi le_rfl


@[simp]
theorem iUnion_Iic : ⋃ a : α, Iic a = univ :=
  iUnion_eq_univ_iff.2 fun x => ⟨x, right_mem_Iic⟩


@[simp]
theorem iUnion_Ici : ⋃ a : α, Ici a = univ :=
  iUnion_eq_univ_iff.2 fun x => ⟨x, left_mem_Ici⟩


@[simp]
theorem iUnion_Icc_right (a : α) : ⋃ b, Icc a b = Ici a := by
  /-
    α : Type v
    inst✝ : Preorder α
    a : α
    ⊢ Eq (Set.iUnion fun b => Set.Icc a b) (Set.Ici a)
  -/
  simp only [← Ici_inter_Iic, ← inter_iUnion, iUnion_Iic, inter_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ioc_right (a : α) : ⋃ b, Ioc a b = Ioi a := by
  /-
    α : Type v
    inst✝ : Preorder α
    a : α
    ⊢ Eq (Set.iUnion fun b => Set.Ioc a b) (Set.Ioi a)
  -/
  simp only [← Ioi_inter_Iic, ← inter_iUnion, iUnion_Iic, inter_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Icc_left (b : α) : ⋃ a, Icc a b = Iic b := by
  /-
    α : Type v
    inst✝ : Preorder α
    b : α
    ⊢ Eq (Set.iUnion fun a => Set.Icc a b) (Set.Iic b)
  -/
  simp only [← Ici_inter_Iic, ← iUnion_inter, iUnion_Ici, univ_inter]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ico_left (b : α) : ⋃ a, Ico a b = Iio b := by
  /-
    α : Type v
    inst✝ : Preorder α
    b : α
    ⊢ Eq (Set.iUnion fun a => Set.Ico a b) (Set.Iio b)
  -/
  simp only [← Ici_inter_Iio, ← iUnion_inter, iUnion_Ici, univ_inter]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Iio [NoMaxOrder α] : ⋃ a : α, Iio a = univ :=
  iUnion_eq_univ_iff.2 exists_gt


@[simp]
theorem iUnion_Ioi [NoMinOrder α] : ⋃ a : α, Ioi a = univ :=
  iUnion_eq_univ_iff.2 exists_lt


@[simp]
theorem iUnion_Ico_right [NoMaxOrder α] (a : α) : ⋃ b, Ico a b = Ici a := by
  /-
    α : Type v
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Eq (Set.iUnion fun b => Set.Ico a b) (Set.Ici a)
  -/
  simp only [← Ici_inter_Iio, ← inter_iUnion, iUnion_Iio, inter_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ioo_right [NoMaxOrder α] (a : α) : ⋃ b, Ioo a b = Ioi a := by
  /-
    α : Type v
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Eq (Set.iUnion fun b => Set.Ioo a b) (Set.Ioi a)
  -/
  simp only [← Ioi_inter_Iio, ← inter_iUnion, iUnion_Iio, inter_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ioc_left [NoMinOrder α] (b : α) : ⋃ a, Ioc a b = Iic b := by
  /-
    α : Type v
    inst✝¹ : Preorder α
    inst✝ : NoMinOrder α
    b : α
    ⊢ Eq (Set.iUnion fun a => Set.Ioc a b) (Set.Iic b)
  -/
  simp only [← Ioi_inter_Iic, ← iUnion_inter, iUnion_Ioi, univ_inter]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ioo_left [NoMinOrder α] (b : α) : ⋃ a, Ioo a b = Iio b := by
  /-
    α : Type v
    inst✝¹ : Preorder α
    inst✝ : NoMinOrder α
    b : α
    ⊢ Eq (Set.iUnion fun a => Set.Ioo a b) (Set.Iio b)
  -/
  simp only [← Ioi_inter_Iio, ← iUnion_inter, iUnion_Ioi, univ_inter]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_disjoint_Ico : Disjoint (Ico a₁ a₂) (Ico b₁ b₂) ↔ min a₂ b₂ ≤ max a₁ b₁ := by
  /-
    α : Type v
    inst✝ : LinearOrder α
    a₁ a₂ b₁ b₂ : α
    ⊢ Iff (Disjoint (Set.Ico a₁ a₂) (Set.Ico b₁ b₂)) (LE.le (Min.min a₂ b₂) (Max.m …
  -/
  simp_rw [Set.disjoint_iff_inter_eq_empty, Ico_inter_Ico, Ico_eq_empty_iff, not_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioc_disjoint_Ioc : Disjoint (Ioc a₁ a₂) (Ioc b₁ b₂) ↔ min a₂ b₂ ≤ max a₁ b₁ := by
  /-
    α : Type v
    inst✝ : LinearOrder α
    a₁ a₂ b₁ b₂ : α
    ⊢ Iff (Disjoint (Set.Ioc a₁ a₂) (Set.Ioc b₁ b₂)) (LE.le (Min.min a₂ b₂) (Max.m …
  -/
  have h : _ ↔ min (toDual a₁) (toDual b₁) ≤ max (toDual a₂) (toDual b₂) := Ico_disjoint_Ico
  /-
    α : Type v
    inst✝ : LinearOrder α
    a₁ a₂ b₁ b₂ : α
    h : Iff (Disjoint (Set.Ico (OrderDual.toDual a₂) (OrderDual.toDual a₁)) (Set.I …
    ⊢ Iff (Disjoint (Set.Ioc a₁ a₂) (Set.Ioc b₁ b₂)) (LE.le (Min.min a₂ b₂) (Max.m …
  -/
  simpa only [dual_Ico] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioo_disjoint_Ioo [DenselyOrdered α] :
    Disjoint (Set.Ioo a₁ a₂) (Set.Ioo b₁ b₂) ↔ min a₂ b₂ ≤ max a₁ b₁ := by
  /-
    α : Type v
    inst✝¹ : LinearOrder α
    a₁ a₂ b₁ b₂ : α
    inst✝ : DenselyOrdered α
    ⊢ Iff (Disjoint (Set.Ioo a₁ a₂) (Set.Ioo b₁ b₂)) (LE.le (Min.min a₂ b₂) (Max.m …
  -/
  simp_rw [Set.disjoint_iff_inter_eq_empty, Ioo_inter_Ioo, Ioo_eq_empty_iff, not_lt]
  /-
    🎉 no goals
  -/


/-- If two half-open intervals are disjoint and the endpoint of one lies in the other,
  then it must be equal to the endpoint of the other. -/
theorem eq_of_Ico_disjoint {x₁ x₂ y₁ y₂ : α} (h : Disjoint (Ico x₁ x₂) (Ico y₁ y₂)) (hx : x₁ < x₂)
    (h2 : x₂ ∈ Ico y₁ y₂) : y₁ = x₂ := by
  /-
    α : Type v
    inst✝ : LinearOrder α
    x₁ x₂ y₁ y₂ : α
    h : Disjoint (Set.Ico x₁ x₂) (Set.Ico y₁ y₂)
    hx : LT.lt x₁ x₂
    h2 : Membership.mem (Set.Ico y₁ y₂) x₂
    ⊢ Eq y₁ x₂
  -/
  rw [Ico_disjoint_Ico, min_eq_left (le_of_lt h2.2), le_max_iff] at h
  /-
    α : Type v
    inst✝ : LinearOrder α
    x₁ x₂ y₁ y₂ : α
    h : Or (LE.le x₂ x₁) (LE.le x₂ y₁)
    hx : LT.lt x₁ x₂
    h2 : Membership.mem (Set.Ico y₁ y₂) x₂
    ⊢ Eq y₁ x₂
  -/
  apply le_antisymm h2.1
  /-
    α : Type v
    inst✝ : LinearOrder α
    x₁ x₂ y₁ y₂ : α
    h : Or (LE.le x₂ x₁) (LE.le x₂ y₁)
    hx : LT.lt x₁ x₂
    h2 : Membership.mem (Set.Ico y₁ y₂) x₂
    ⊢ LE.le x₂ y₁
  -/
  exact h.elim (fun h => absurd hx (not_lt_of_le h)) id
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ico_eq_Iio_self_iff {f : ι → α} {a : α} :
    ⋃ i, Ico (f i) a = Iio a ↔ ∀ x < a, ∃ i, f i ≤ x := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    a : α
    ⊢ Iff (Eq (Set.iUnion fun i => Set.Ico (f i) a) (Set.Iio a)) (∀ (x : α), LT.lt …
  -/
  simp [← Ici_inter_Iio, ← iUnion_inter, subset_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_Ioc_eq_Ioi_self_iff {f : ι → α} {a : α} :
    ⋃ i, Ioc a (f i) = Ioi a ↔ ∀ x, a < x → ∃ i, x ≤ f i := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    a : α
    ⊢ Iff (Eq (Set.iUnion fun i => Set.Ioc a (f i)) (Set.Ioi a)) (∀ (x : α), LT.lt …
  -/
  simp [← Ioi_inter_Iic, ← inter_iUnion, subset_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem biUnion_Ico_eq_Iio_self_iff {p : ι → Prop} {f : ∀ i, p i → α} {a : α} :
    ⋃ (i) (hi : p i), Ico (f i hi) a = Iio a ↔ ∀ x < a, ∃ i hi, f i hi ≤ x := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    p : ι → Prop
    f : (i : ι) → p i → α
    a : α
    ⊢ Iff (Eq (Set.iUnion fun i => Set.iUnion fun hi => Set.Ico (f i hi) a) (Set.I …
  -/
  simp [← Ici_inter_Iio, ← iUnion_inter, subset_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem biUnion_Ioc_eq_Ioi_self_iff {p : ι → Prop} {f : ∀ i, p i → α} {a : α} :
    ⋃ (i) (hi : p i), Ioc a (f i hi) = Ioi a ↔ ∀ x, a < x → ∃ i hi, x ≤ f i hi := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    p : ι → Prop
    f : (i : ι) → p i → α
    a : α
    ⊢ Iff (Eq (Set.iUnion fun i => Set.iUnion fun hi => Set.Ioc a (f i hi)) (Set.I …
  -/
  simp [← Ioi_inter_Iic, ← inter_iUnion, subset_def]
  /-
    🎉 no goals
  -/


theorem IsGLB.biUnion_Ioi_eq (h : IsGLB s a) : ⋃ x ∈ s, Ioi x = Ioi a := by
  /-
    α : Type v
    inst✝ : LinearOrder α
    s : Set α
    a : α
    h : IsGLB s a
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => Set.Ioi x) (Set.Ioi a)
  -/
  refine (iUnion₂_subset fun x hx => ?_).antisymm fun x hx => ?_
    /-
      case refine_1
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      h : IsGLB s a
      x : α
      hx : Membership.mem s x
      ⊢ HasSubset.Subset (Set.Ioi x) (Set.Ioi a)
    -/
  · exact Ioi_subset_Ioi (h.1 hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      h : IsGLB s a
      x : α
      hx : Membership.mem (Set.Ioi a) x
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.Ioi i) x
    -/
  · rcases h.exists_between hx with ⟨y, hys, _, hyx⟩
    /-
      case refine_2.intro.intro.intro
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      h : IsGLB s a
      x : α
      hx : Membership.mem (Set.Ioi a) x
      y : α
      hys : Membership.mem s y
      left✝ : LE.le a y
      hyx : LT.lt y x
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.Ioi i) x
    -/
    exact mem_biUnion hys hyx
    /-
      🎉 no goals
    -/


theorem IsGLB.iUnion_Ioi_eq (h : IsGLB (range f) a) : ⋃ x, Ioi (f x) = Ioi a :=
  biUnion_range.symm.trans h.biUnion_Ioi_eq


theorem IsLUB.biUnion_Iio_eq (h : IsLUB s a) : ⋃ x ∈ s, Iio x = Iio a :=
  h.dual.biUnion_Ioi_eq


theorem IsLUB.iUnion_Iio_eq (h : IsLUB (range f) a) : ⋃ x, Iio (f x) = Iio a :=
  h.dual.iUnion_Ioi_eq


theorem IsGLB.biUnion_Ici_eq_Ioi (a_glb : IsGLB s a) (a_not_mem : a ∉ s) :
    ⋃ x ∈ s, Ici x = Ioi a := by
  /-
    α : Type v
    inst✝ : LinearOrder α
    s : Set α
    a : α
    a_glb : IsGLB s a
    a_not_mem : Not (Membership.mem s a)
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => Set.Ici x) (Set.Ioi a)
  -/
  refine (iUnion₂_subset fun x hx => ?_).antisymm fun x hx => ?_
    /-
      case refine_1
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      a_glb : IsGLB s a
      a_not_mem : Not (Membership.mem s a)
      x : α
      hx : Membership.mem s x
      ⊢ HasSubset.Subset (Set.Ici x) (Set.Ioi a)
    -/
  · exact Ici_subset_Ioi.mpr (lt_of_le_of_ne (a_glb.1 hx) fun h => (h ▸ a_not_mem) hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      a_glb : IsGLB s a
      a_not_mem : Not (Membership.mem s a)
      x : α
      hx : Membership.mem (Set.Ioi a) x
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.Ici i) x
    -/
  · rcases a_glb.exists_between hx with ⟨y, hys, _, hyx⟩
    /-
      case refine_2.intro.intro.intro
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      a_glb : IsGLB s a
      a_not_mem : Not (Membership.mem s a)
      x : α
      hx : Membership.mem (Set.Ioi a) x
      y : α
      hys : Membership.mem s y
      left✝ : LE.le a y
      hyx : LT.lt y x
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.Ici i) x
    -/
    rw [mem_iUnion₂]
    /-
      case refine_2.intro.intro.intro
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      a_glb : IsGLB s a
      a_not_mem : Not (Membership.mem s a)
      x : α
      hx : Membership.mem (Set.Ioi a) x
      y : α
      hys : Membership.mem s y
      left✝ : LE.le a y
      hyx : LT.lt y x
      ⊢ Exists fun i => Exists fun j => Membership.mem (Set.Ici i) x
    -/
    exact ⟨y, hys, hyx.le⟩
    /-
      🎉 no goals
    -/


theorem IsGLB.biUnion_Ici_eq_Ici (a_glb : IsGLB s a) (a_mem : a ∈ s) :
    ⋃ x ∈ s, Ici x = Ici a := by
  /-
    α : Type v
    inst✝ : LinearOrder α
    s : Set α
    a : α
    a_glb : IsGLB s a
    a_mem : Membership.mem s a
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => Set.Ici x) (Set.Ici a)
  -/
  refine (iUnion₂_subset fun x hx => ?_).antisymm fun x hx => ?_
    /-
      case refine_1
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      a_glb : IsGLB s a
      a_mem : Membership.mem s a
      x : α
      hx : Membership.mem s x
      ⊢ HasSubset.Subset (Set.Ici x) (Set.Ici a)
    -/
  · exact Ici_subset_Ici.mpr (mem_lowerBounds.mp a_glb.1 x hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type v
      inst✝ : LinearOrder α
      s : Set α
      a : α
      a_glb : IsGLB s a
      a_mem : Membership.mem s a
      x : α
      hx : Membership.mem (Set.Ici a) x
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun j => Set.Ici i) x
    -/
  · exact mem_iUnion₂.mpr ⟨a, a_mem, hx⟩
    /-
      🎉 no goals
    -/


theorem IsLUB.biUnion_Iic_eq_Iio (a_lub : IsLUB s a) (a_not_mem : a ∉ s) :
    ⋃ x ∈ s, Iic x = Iio a :=
  a_lub.dual.biUnion_Ici_eq_Ioi a_not_mem


theorem IsLUB.biUnion_Iic_eq_Iic (a_lub : IsLUB s a) (a_mem : a ∈ s) : ⋃ x ∈ s, Iic x = Iic a :=
  a_lub.dual.biUnion_Ici_eq_Ici a_mem


theorem iUnion_Ici_eq_Ioi_iInf {R : Type*} [CompleteLinearOrder R] {f : ι → R}
    (no_least_elem : ⨅ i, f i ∉ range f) : ⋃ i : ι, Ici (f i) = Ioi (⨅ i, f i) := by
  simp only [← IsGLB.biUnion_Ici_eq_Ioi (@isGLB_iInf _ _ _ f) no_least_elem, mem_range,
    iUnion_exists, iUnion_iUnion_eq']


theorem iUnion_Iic_eq_Iio_iSup {R : Type*} [CompleteLinearOrder R] {f : ι → R}
    (no_greatest_elem : (⨆ i, f i) ∉ range f) : ⋃ i : ι, Iic (f i) = Iio (⨆ i, f i) :=
  @iUnion_Ici_eq_Ioi_iInf ι (OrderDual R) _ f no_greatest_elem


theorem iUnion_Ici_eq_Ici_iInf {R : Type*} [CompleteLinearOrder R] {f : ι → R}
    (has_least_elem : (⨅ i, f i) ∈ range f) : ⋃ i : ι, Ici (f i) = Ici (⨅ i, f i) := by
  simp only [← IsGLB.biUnion_Ici_eq_Ici (@isGLB_iInf _ _ _ f) has_least_elem, mem_range,
    iUnion_exists, iUnion_iUnion_eq']


theorem iUnion_Iic_eq_Iic_iSup {R : Type*} [CompleteLinearOrder R] {f : ι → R}
    (has_greatest_elem : (⨆ i, f i) ∈ range f) : ⋃ i : ι, Iic (f i) = Iic (⨆ i, f i) :=
  @iUnion_Ici_eq_Ici_iInf ι (OrderDual R) _ f has_greatest_elem


theorem iUnion_Iio_eq_univ_iff : ⋃ i, Iio (f i) = univ ↔ (¬ BddAbove (range f)) := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    ⊢ Iff (Eq (Set.iUnion fun i => Set.Iio (f i)) Set.univ) (Not (BddAbove (Set.ra …
  -/
  simp [not_bddAbove_iff, Set.eq_univ_iff_forall]
  /-
    🎉 no goals
  -/


theorem iUnion_Iic_of_not_bddAbove_range (hf : ¬ BddAbove (range f)) : ⋃ i, Iic (f i) = univ := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddAbove (Set.range f))
    ⊢ Eq (Set.iUnion fun i => Set.Iic (f i)) Set.univ
  -/
  refine  Set.eq_univ_of_subset ?_ (iUnion_Iio_eq_univ_iff.mpr hf)
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddAbove (Set.range f))
    ⊢ HasSubset.Subset (Set.iUnion fun i => Set.Iio (f i)) (Set.iUnion fun i => Se …
  -/
  gcongr
  /-
    case h
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddAbove (Set.range f))
    i✝ : ι
    ⊢ HasSubset.Subset (Set.Iio (f i✝)) (Set.Iic (f i✝))
  -/
  exact Iio_subset_Iic_self
  /-
    🎉 no goals
  -/


theorem iInter_Iic_eq_empty_iff : ⋂ i, Iic (f i) = ∅ ↔ ¬ BddBelow (range f) := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    ⊢ Iff (Eq (Set.iInter fun i => Set.Iic (f i)) EmptyCollection.emptyCollection) …
  -/
  simp [not_bddBelow_iff, Set.eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


theorem iInter_Iio_of_not_bddBelow_range (hf : ¬ BddBelow (range f)) : ⋂ i, Iio (f i) = ∅ := by
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddBelow (Set.range f))
    ⊢ Eq (Set.iInter fun i => Set.Iio (f i)) EmptyCollection.emptyCollection
  -/
  refine eq_empty_of_subset_empty ?_
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddBelow (Set.range f))
    ⊢ HasSubset.Subset (Set.iInter fun i => Set.Iio (f i)) EmptyCollection.emptyCo …
  -/
  rw [← iInter_Iic_eq_empty_iff.mpr hf]
  /-
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddBelow (Set.range f))
    ⊢ HasSubset.Subset (Set.iInter fun i => Set.Iio (f i)) (Set.iInter fun i => Se …
  -/
  gcongr
  /-
    case h
    ι : Sort u
    α : Type v
    inst✝ : LinearOrder α
    f : ι → α
    hf : Not (BddBelow (Set.range f))
    i✝ : ι
    ⊢ HasSubset.Subset (Set.Iio (f i✝)) (Set.Iic (f i✝))
  -/
  exact Iio_subset_Iic_self
  /-
    🎉 no goals
  -/


