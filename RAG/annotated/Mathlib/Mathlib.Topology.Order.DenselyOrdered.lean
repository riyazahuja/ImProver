/-- The closure of the interval `(a, +∞)` is the closed interval `[a, +∞)`, unless `a` is a top
element. -/
theorem closure_Ioi' {a : α} (h : (Ioi a).Nonempty) : closure (Ioi a) = Ici a := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    h : (Set.Ioi a).Nonempty
    ⊢ Eq (closure (Set.Ioi a)) (Set.Ici a)
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a : α
      h : (Set.Ioi a).Nonempty
      ⊢ HasSubset.Subset (closure (Set.Ioi a)) (Set.Ici a)
    -/
  · exact closure_minimal Ioi_subset_Ici_self isClosed_Ici
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a : α
      h : (Set.Ioi a).Nonempty
      ⊢ HasSubset.Subset (Set.Ici a) (closure (Set.Ioi a))
    -/
  · rw [← diff_subset_closure_iff, Ici_diff_Ioi_same, singleton_subset_iff]
    /-
      case h₂
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a : α
      h : (Set.Ioi a).Nonempty
      ⊢ Membership.mem (closure (Set.Ioi a)) a
    -/
    exact isGLB_Ioi.mem_closure h
    /-
      🎉 no goals
    -/


/-- The closure of the interval `(a, +∞)` is the closed interval `[a, +∞)`. -/
@[simp]
theorem closure_Ioi (a : α) [NoMaxOrder α] : closure (Ioi a) = Ici a :=
  closure_Ioi' nonempty_Ioi


/-- The closure of the interval `(-∞, a)` is the closed interval `(-∞, a]`, unless `a` is a bottom
element. -/
theorem closure_Iio' (h : (Iio a).Nonempty) : closure (Iio a) = Iic a :=
  closure_Ioi' (α := αᵒᵈ) h


/-- The closure of the interval `(-∞, a)` is the interval `(-∞, a]`. -/
@[simp]
theorem closure_Iio (a : α) [NoMinOrder α] : closure (Iio a) = Iic a :=
  closure_Iio' nonempty_Iio


/-- The closure of the open interval `(a, b)` is the closed interval `[a, b]`. -/
@[simp]
theorem closure_Ioo {a b : α} (hab : a ≠ b) : closure (Ioo a b) = Icc a b := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    hab : Ne a b
    ⊢ Eq (closure (Set.Ioo a b)) (Set.Icc a b)
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (closure (Set.Ioo a b)) (Set.Icc a b)
    -/
  · exact closure_minimal Ioo_subset_Icc_self isClosed_Icc
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioo a b))
    -/
  · cases' hab.lt_or_lt with hab hab
      /-
        case h₂.inl
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        hab✝ : Ne a b
        hab : LT.lt a b
        ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioo a b))
      -/
    · rw [← diff_subset_closure_iff, Icc_diff_Ioo_same hab.le]
      /-
        case h₂.inl
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        hab✝ : Ne a b
        hab : LT.lt a b
        ⊢ HasSubset.Subset (Insert.insert a (Singleton.singleton b)) (closure (Set.Ioo …
      -/
      have hab' : (Ioo a b).Nonempty := nonempty_Ioo.2 hab
      /-
        case h₂.inl
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        hab✝ : Ne a b
        hab : LT.lt a b
        hab' : (Set.Ioo a b).Nonempty
        ⊢ HasSubset.Subset (Insert.insert a (Singleton.singleton b)) (closure (Set.Ioo …
      -/
      simp only [insert_subset_iff, singleton_subset_iff]
      /-
        case h₂.inl
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        hab✝ : Ne a b
        hab : LT.lt a b
        hab' : (Set.Ioo a b).Nonempty
        ⊢ And (Membership.mem (closure (Set.Ioo a b)) a) (Membership.mem (closure (Set …
      -/
      exact ⟨(isGLB_Ioo hab).mem_closure hab', (isLUB_Ioo hab).mem_closure hab'⟩
      /-
        🎉 no goals
      -/
      /-
        case h₂.inr
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        hab✝ : Ne a b
        hab : LT.lt b a
        ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioo a b))
      -/
    · rw [Icc_eq_empty_of_lt hab]
      /-
        case h₂.inr
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        hab✝ : Ne a b
        hab : LT.lt b a
        ⊢ HasSubset.Subset EmptyCollection.emptyCollection (closure (Set.Ioo a b))
      -/
      exact empty_subset _
      /-
        🎉 no goals
      -/


/-- The closure of the interval `(a, b]` is the closed interval `[a, b]`. -/
@[simp]
theorem closure_Ioc {a b : α} (hab : a ≠ b) : closure (Ioc a b) = Icc a b := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    hab : Ne a b
    ⊢ Eq (closure (Set.Ioc a b)) (Set.Icc a b)
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (closure (Set.Ioc a b)) (Set.Icc a b)
    -/
  · exact closure_minimal Ioc_subset_Icc_self isClosed_Icc
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioc a b))
    -/
  · apply Subset.trans _ (closure_mono Ioo_subset_Ioc_self)
    /-
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioo a b))
    -/
    rw [closure_Ioo hab]
    /-
      🎉 no goals
    -/


/-- The closure of the interval `[a, b)` is the closed interval `[a, b]`. -/
@[simp]
theorem closure_Ico {a b : α} (hab : a ≠ b) : closure (Ico a b) = Icc a b := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    hab : Ne a b
    ⊢ Eq (closure (Set.Ico a b)) (Set.Icc a b)
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (closure (Set.Ico a b)) (Set.Icc a b)
    -/
  · exact closure_minimal Ico_subset_Icc_self isClosed_Icc
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ico a b))
    -/
  · apply Subset.trans _ (closure_mono Ioo_subset_Ico_self)
    /-
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a b : α
      hab : Ne a b
      ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioo a b))
    -/
    rw [closure_Ioo hab]
    /-
      🎉 no goals
    -/


@[simp]
theorem interior_Ici' {a : α} (ha : (Iio a).Nonempty) : interior (Ici a) = Ioi a := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    ha : (Set.Iio a).Nonempty
    ⊢ Eq (interior (Set.Ici a)) (Set.Ioi a)
  -/
  rw [← compl_Iio, interior_compl, closure_Iio' ha, compl_Iic]
  /-
    🎉 no goals
  -/


theorem interior_Ici [NoMinOrder α] {a : α} : interior (Ici a) = Ioi a :=
  interior_Ici' nonempty_Iio


@[simp]
theorem interior_Iic' {a : α} (ha : (Ioi a).Nonempty) : interior (Iic a) = Iio a :=
  interior_Ici' (α := αᵒᵈ) ha


theorem interior_Iic [NoMaxOrder α] {a : α} : interior (Iic a) = Iio a :=
  interior_Iic' nonempty_Ioi


@[simp]
theorem interior_Icc [NoMinOrder α] [NoMaxOrder α] {a b : α} : interior (Icc a b) = Ioo a b := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    a b : α
    ⊢ Eq (interior (Set.Icc a b)) (Set.Ioo a b)
  -/
  rw [← Ici_inter_Iic, interior_inter, interior_Ici, interior_Iic, Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem Icc_mem_nhds_iff [NoMinOrder α] [NoMaxOrder α] {a b x : α} :
    Icc a b ∈ 𝓝 x ↔ x ∈ Ioo a b := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    a b x : α
    ⊢ Iff (Membership.mem (nhds x) (Set.Icc a b)) (Membership.mem (Set.Ioo a b) x)
  -/
  rw [← interior_Icc, mem_interior_iff_mem_nhds]
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_Ico [NoMinOrder α] {a b : α} : interior (Ico a b) = Ioo a b := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : NoMinOrder α
    a b : α
    ⊢ Eq (interior (Set.Ico a b)) (Set.Ioo a b)
  -/
  rw [← Ici_inter_Iio, interior_inter, interior_Ici, interior_Iio, Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ico_mem_nhds_iff [NoMinOrder α] {a b x : α} : Ico a b ∈ 𝓝 x ↔ x ∈ Ioo a b := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : NoMinOrder α
    a b x : α
    ⊢ Iff (Membership.mem (nhds x) (Set.Ico a b)) (Membership.mem (Set.Ioo a b) x)
  -/
  rw [← interior_Ico, mem_interior_iff_mem_nhds]
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_Ioc [NoMaxOrder α] {a b : α} : interior (Ioc a b) = Ioo a b := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : NoMaxOrder α
    a b : α
    ⊢ Eq (interior (Set.Ioc a b)) (Set.Ioo a b)
  -/
  rw [← Ioi_inter_Iic, interior_inter, interior_Ioi, interior_Iic, Ioi_inter_Iio]
  /-
    🎉 no goals
  -/


@[simp]
theorem Ioc_mem_nhds_iff [NoMaxOrder α] {a b x : α} : Ioc a b ∈ 𝓝 x ↔ x ∈ Ioo a b := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : NoMaxOrder α
    a b x : α
    ⊢ Iff (Membership.mem (nhds x) (Set.Ioc a b)) (Membership.mem (Set.Ioo a b) x)
  -/
  rw [← interior_Ioc, mem_interior_iff_mem_nhds]
  /-
    🎉 no goals
  -/


theorem closure_interior_Icc {a b : α} (h : a ≠ b) : closure (interior (Icc a b)) = Icc a b :=
  (closure_minimal interior_subset isClosed_Icc).antisymm <|
    calc
      Icc a b = closure (Ioo a b) := (closure_Ioo h).symm
      _ ⊆ closure (interior (Icc a b)) :=
        closure_mono (interior_maximal Ioo_subset_Icc_self isOpen_Ioo)


theorem Ioc_subset_closure_interior (a b : α) : Ioc a b ⊆ closure (interior (Ioc a b)) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    ⊢ HasSubset.Subset (Set.Ioc a b) (closure (interior (Set.Ioc a b)))
  -/
  rcases eq_or_ne a b with (rfl | h)
    /-
      case inl
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a : α
      ⊢ HasSubset.Subset (Set.Ioc a a) (closure (interior (Set.Ioc a a)))
    -/
  · simp
    /-
      🎉 no goals
    -/
  · calc
      Ioc a b ⊆ Icc a b := Ioc_subset_Icc_self
      _ = closure (Ioo a b) := (closure_Ioo h).symm
      _ ⊆ closure (interior (Ioc a b)) :=
        closure_mono (interior_maximal Ioo_subset_Ioc_self isOpen_Ioo)


theorem Ico_subset_closure_interior (a b : α) : Ico a b ⊆ closure (interior (Ico a b)) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    ⊢ HasSubset.Subset (Set.Ico a b) (closure (interior (Set.Ico a b)))
  -/
  simpa only [dual_Ioc] using Ioc_subset_closure_interior (OrderDual.toDual b) (OrderDual.toDual a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_Ici' {a : α} (ha : (Iio a).Nonempty) : frontier (Ici a) = {a} := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    ha : (Set.Iio a).Nonempty
    ⊢ Eq (frontier (Set.Ici a)) (Singleton.singleton a)
  -/
  simp [frontier, ha]
  /-
    🎉 no goals
  -/


theorem frontier_Ici [NoMinOrder α] {a : α} : frontier (Ici a) = {a} :=
  frontier_Ici' nonempty_Iio


@[simp]
theorem frontier_Iic' {a : α} (ha : (Ioi a).Nonempty) : frontier (Iic a) = {a} := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    ha : (Set.Ioi a).Nonempty
    ⊢ Eq (frontier (Set.Iic a)) (Singleton.singleton a)
  -/
  simp [frontier, ha]
  /-
    🎉 no goals
  -/


theorem frontier_Iic [NoMaxOrder α] {a : α} : frontier (Iic a) = {a} :=
  frontier_Iic' nonempty_Ioi


@[simp]
theorem frontier_Ioi' {a : α} (ha : (Ioi a).Nonempty) : frontier (Ioi a) = {a} := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    ha : (Set.Ioi a).Nonempty
    ⊢ Eq (frontier (Set.Ioi a)) (Singleton.singleton a)
  -/
  simp [frontier, closure_Ioi' ha, Iic_diff_Iio, Icc_self]
  /-
    🎉 no goals
  -/


theorem frontier_Ioi [NoMaxOrder α] {a : α} : frontier (Ioi a) = {a} :=
  frontier_Ioi' nonempty_Ioi


@[simp]
theorem frontier_Iio' {a : α} (ha : (Iio a).Nonempty) : frontier (Iio a) = {a} := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    ha : (Set.Iio a).Nonempty
    ⊢ Eq (frontier (Set.Iio a)) (Singleton.singleton a)
  -/
  simp [frontier, closure_Iio' ha, Iic_diff_Iio, Icc_self]
  /-
    🎉 no goals
  -/


theorem frontier_Iio [NoMinOrder α] {a : α} : frontier (Iio a) = {a} :=
  frontier_Iio' nonempty_Iio


@[simp]
theorem frontier_Icc [NoMinOrder α] [NoMaxOrder α] {a b : α} (h : a ≤ b) :
                                      /-
                                        α : Type u_1
                                        inst✝⁵ : TopologicalSpace α
                                        inst✝⁴ : LinearOrder α
                                        inst✝³ : OrderTopology α
                                        inst✝² : DenselyOrdered α
                                        inst✝¹ : NoMinOrder α
                                        inst✝ : NoMaxOrder α
                                        a b : α
                                        h : LE.le a b
                                        ⊢ Eq (frontier (Set.Icc a b)) (Insert.insert a (Singleton.singleton b))
                                      -/
    frontier (Icc a b) = {a, b} := by simp [frontier, h, Icc_diff_Ioo_same]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem frontier_Ioo {a b : α} (h : a < b) : frontier (Ioo a b) = {a, b} := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    h : LT.lt a b
    ⊢ Eq (frontier (Set.Ioo a b)) (Insert.insert a (Singleton.singleton b))
  -/
  rw [frontier, closure_Ioo h.ne, interior_Ioo, Icc_diff_Ioo_same h.le]
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_Ico [NoMinOrder α] {a b : α} (h : a < b) : frontier (Ico a b) = {a, b} := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : NoMinOrder α
    a b : α
    h : LT.lt a b
    ⊢ Eq (frontier (Set.Ico a b)) (Insert.insert a (Singleton.singleton b))
  -/
  rw [frontier, closure_Ico h.ne, interior_Ico, Icc_diff_Ioo_same h.le]
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_Ioc [NoMaxOrder α] {a b : α} (h : a < b) : frontier (Ioc a b) = {a, b} := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : NoMaxOrder α
    a b : α
    h : LT.lt a b
    ⊢ Eq (frontier (Set.Ioc a b)) (Insert.insert a (Singleton.singleton b))
  -/
  rw [frontier, closure_Ioc h.ne, interior_Ioc, Icc_diff_Ioo_same h.le]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_Ioi_neBot' {a b : α} (H₁ : (Ioi a).Nonempty) (H₂ : a ≤ b) :
    NeBot (𝓝[Ioi a] b) :=
                                           /-
                                             α : Type u_1
                                             inst✝³ : TopologicalSpace α
                                             inst✝² : LinearOrder α
                                             inst✝¹ : OrderTopology α
                                             inst✝ : DenselyOrdered α
                                             a b : α
                                             H₁ : (Set.Ioi a).Nonempty
                                             H₂ : LE.le a b
                                             ⊢ Membership.mem (closure (Set.Ioi a)) b
                                           -/
  mem_closure_iff_nhdsWithin_neBot.1 <| by rwa [closure_Ioi' H₁]
                                           /-
                                             🎉 no goals
                                           -/


theorem nhdsWithin_Ioi_neBot [NoMaxOrder α] {a b : α} (H : a ≤ b) : NeBot (𝓝[Ioi a] b) :=
  nhdsWithin_Ioi_neBot' nonempty_Ioi H


theorem nhdsGT_neBot_of_exists_gt {a : α} (H : ∃ b, a < b) : NeBot (𝓝[>] a) :=
  nhdsWithin_Ioi_neBot' H (le_refl a)


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioi_self_neBot' := nhdsGT_neBot_of_exists_gt


instance nhdsGT_neBot [NoMaxOrder α] (a : α) : NeBot (𝓝[>] a) := nhdsWithin_Ioi_neBot le_rfl


@[deprecated nhdsGT_neBot (since := "2024-12-22")]
theorem nhdsWithin_Ioi_self_neBot [NoMaxOrder α] (a : α) : NeBot (𝓝[>] a) := nhdsGT_neBot a


theorem nhdsWithin_Iio_neBot' {b c : α} (H₁ : (Iio c).Nonempty) (H₂ : b ≤ c) :
    NeBot (𝓝[Iio c] b) :=
                                           /-
                                             α : Type u_1
                                             inst✝³ : TopologicalSpace α
                                             inst✝² : LinearOrder α
                                             inst✝¹ : OrderTopology α
                                             inst✝ : DenselyOrdered α
                                             b c : α
                                             H₁ : (Set.Iio c).Nonempty
                                             H₂ : LE.le b c
                                             ⊢ Membership.mem (closure (Set.Iio c)) b
                                           -/
  mem_closure_iff_nhdsWithin_neBot.1 <| by rwa [closure_Iio' H₁]
                                           /-
                                             🎉 no goals
                                           -/


theorem nhdsWithin_Iio_neBot [NoMinOrder α] {a b : α} (H : a ≤ b) : NeBot (𝓝[Iio b] a) :=
  nhdsWithin_Iio_neBot' nonempty_Iio H


theorem nhdsWithin_Iio_self_neBot' {b : α} (H : (Iio b).Nonempty) : NeBot (𝓝[<] b) :=
  nhdsWithin_Iio_neBot' H (le_refl b)


instance nhdsLT_neBot [NoMinOrder α] (a : α) : NeBot (𝓝[<] a) := nhdsWithin_Iio_neBot (le_refl a)


@[deprecated nhdsLT_neBot (since := "2024-12-22")]
theorem nhdsWithin_Iio_self_neBot [NoMinOrder α] (a : α) : NeBot (𝓝[<] a) := nhdsLT_neBot a


theorem right_nhdsWithin_Ico_neBot {a b : α} (H : a < b) : NeBot (𝓝[Ico a b] b) :=
  (isLUB_Ico H).nhdsWithin_neBot (nonempty_Ico.2 H)


theorem left_nhdsWithin_Ioc_neBot {a b : α} (H : a < b) : NeBot (𝓝[Ioc a b] a) :=
  (isGLB_Ioc H).nhdsWithin_neBot (nonempty_Ioc.2 H)


theorem left_nhdsWithin_Ioo_neBot {a b : α} (H : a < b) : NeBot (𝓝[Ioo a b] a) :=
  (isGLB_Ioo H).nhdsWithin_neBot (nonempty_Ioo.2 H)


theorem right_nhdsWithin_Ioo_neBot {a b : α} (H : a < b) : NeBot (𝓝[Ioo a b] b) :=
  (isLUB_Ioo H).nhdsWithin_neBot (nonempty_Ioo.2 H)


theorem comap_coe_nhdsLT_of_Ioo_subset (hb : s ⊆ Iio b) (hs : s.Nonempty → ∃ a < b, Ioo a b ⊆ s) :
    comap ((↑) : s → α) (𝓝[<] b) = atTop := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    b : α
    s : Set α
    hb : HasSubset.Subset s (Set.Iio b)
    hs : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo a …
    ⊢ Eq (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) Filter.atTop
  -/
  nontriviality
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    b : α
    s : Set α
    hb : HasSubset.Subset s (Set.Iio b)
    hs : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo a …
    a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
    ⊢ Eq (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) Filter.atTop
  -/
  haveI : Nonempty s := nontrivial_iff_nonempty.1 ‹_›
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    b : α
    s : Set α
    hb : HasSubset.Subset s (Set.Iio b)
    hs : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo a …
    a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
    this : Nonempty ↑s
    ⊢ Eq (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) Filter.atTop
  -/
  rcases hs (nonempty_subtype.1 ‹_›) with ⟨a, h, hs⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    b : α
    s : Set α
    hb : HasSubset.Subset s (Set.Iio b)
    hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
    a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
    this : Nonempty ↑s
    a : α
    h : LT.lt a b
    hs : HasSubset.Subset (Set.Ioo a b) s
    ⊢ Eq (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) Filter.atTop
  -/
  ext u; constructor
    /-
      case intro.intro.h.mp
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      ⊢ Membership.mem (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) u → Mem …
    -/
  · rintro ⟨t, ht, hts⟩
    obtain ⟨x, ⟨hxa : a ≤ x, hxb : x < b⟩, hxt : Ioo x b ⊆ t⟩ :=
      (mem_nhdsLT_iff_exists_mem_Ico_Ioo_subset h).mp ht
    /-
      case intro.intro.h.mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      t : Set α
      ht : Membership.mem (nhdsWithin b (Set.Iio b)) t
      hts : HasSubset.Subset (Set.preimage Subtype.val t) u
      x : α
      hxt : HasSubset.Subset (Set.Ioo x b) t
      hxa : LE.le a x
      hxb : LT.lt x b
      ⊢ Membership.mem Filter.atTop u
    -/
    obtain ⟨y, hxy, hyb⟩ := exists_between hxb
    /-
      case intro.intro.h.mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      t : Set α
      ht : Membership.mem (nhdsWithin b (Set.Iio b)) t
      hts : HasSubset.Subset (Set.preimage Subtype.val t) u
      x : α
      hxt : HasSubset.Subset (Set.Ioo x b) t
      hxa : LE.le a x
      hxb : LT.lt x b
      y : α
      hxy : LT.lt x y
      hyb : LT.lt y b
      ⊢ Membership.mem Filter.atTop u
    -/
    refine mem_of_superset (mem_atTop ⟨y, hs ⟨hxa.trans_lt hxy, hyb⟩⟩) ?_
    /-
      case intro.intro.h.mp.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      t : Set α
      ht : Membership.mem (nhdsWithin b (Set.Iio b)) t
      hts : HasSubset.Subset (Set.preimage Subtype.val t) u
      x : α
      hxt : HasSubset.Subset (Set.Ioo x b) t
      hxa : LE.le a x
      hxb : LT.lt x b
      y : α
      hxy : LT.lt x y
      hyb : LT.lt y b
      ⊢ HasSubset.Subset (setOf fun b_1 => LE.le ⟨y, ⋯⟩ b_1) u
    -/
    rintro ⟨z, hzs⟩ (hyz : y ≤ z)
    /-
      case intro.intro.h.mp.intro.intro.intro.intro.intro.intro.intro.mk
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      t : Set α
      ht : Membership.mem (nhdsWithin b (Set.Iio b)) t
      hts : HasSubset.Subset (Set.preimage Subtype.val t) u
      x : α
      hxt : HasSubset.Subset (Set.Ioo x b) t
      hxa : LE.le a x
      hxb : LT.lt x b
      y : α
      hxy : LT.lt x y
      hyb : LT.lt y b
      z : α
      hzs : Membership.mem s z
      hyz : LE.le y z
      ⊢ Membership.mem u ⟨z, hzs⟩
    -/
    exact hts (hxt ⟨hxy.trans_le hyz, hb hzs⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h.mpr
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      ⊢ Membership.mem Filter.atTop u → Membership.mem (Filter.comap Subtype.val (nh …
    -/
  · intro hu
    /-
      case intro.intro.h.mpr
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      hu : Membership.mem Filter.atTop u
      ⊢ Membership.mem (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) u
    -/
    obtain ⟨x : s, hx : ∀ z, x ≤ z → z ∈ u⟩ := mem_atTop_sets.1 hu
    /-
      case intro.intro.h.mpr.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs✝ : s.Nonempty → Exists fun a => And (LT.lt a b) (HasSubset.Subset (Set.Ioo  …
      a✝ : Nontrivial (Filter (Subtype fun x => Membership.mem s x))
      this : Nonempty ↑s
      a : α
      h : LT.lt a b
      hs : HasSubset.Subset (Set.Ioo a b) s
      u : Set (Subtype fun x => Membership.mem s x)
      hu : Membership.mem Filter.atTop u
      x : ↑s
      hx : ∀ (z : ↑s), LE.le x z → Membership.mem u z
      ⊢ Membership.mem (Filter.comap Subtype.val (nhdsWithin b (Set.Iio b))) u
    -/
    exact ⟨Ioo x b, Ioo_mem_nhdsLT (hb x.2), fun z hz => hx _ hz.1.le⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias comap_coe_nhdsWithin_Iio_of_Ioo_subset := comap_coe_nhdsLT_of_Ioo_subset


set_option backward.isDefEq.lazyWhnfCore false in -- See https://github.com/leanprover-community/mathlib4/issues/12534
theorem comap_coe_nhdsGT_of_Ioo_subset (ha : s ⊆ Ioi a) (hs : s.Nonempty → ∃ b > a, Ioo a b ⊆ s) :
    comap ((↑) : s → α) (𝓝[>] a) = atBot :=
  comap_coe_nhdsLT_of_Ioo_subset (show ofDual ⁻¹' s ⊆ Iio (toDual a) from ha) fun h => by
    /-
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      a : α
      s : Set α
      ha : HasSubset.Subset s (Set.Ioi a)
      hs : s.Nonempty → Exists fun b => And (GT.gt b a) (HasSubset.Subset (Set.Ioo a …
      h : (Set.preimage (⇑OrderDual.ofDual) s).Nonempty
      ⊢ Exists fun a_1 => And (LT.lt a_1 (OrderDual.toDual a)) (HasSubset.Subset (Se …
    -/
    simpa only [OrderDual.exists, dual_Ioo] using hs h
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias comap_coe_nhdsWithin_Ioi_of_Ioo_subset := comap_coe_nhdsGT_of_Ioo_subset


theorem map_coe_atTop_of_Ioo_subset (hb : s ⊆ Iio b) (hs : ∀ a' < b, ∃ a < b, Ioo a b ⊆ s) :
    map ((↑) : s → α) atTop = 𝓝[<] b := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    b : α
    s : Set α
    hb : HasSubset.Subset s (Set.Iio b)
    hs : ∀ (a' : α), LT.lt a' b → Exists fun a => And (LT.lt a b) (HasSubset.Subse …
    ⊢ Eq (Filter.map Subtype.val Filter.atTop) (nhdsWithin b (Set.Iio b))
  -/
  rcases eq_empty_or_nonempty (Iio b) with (hb' | ⟨a, ha⟩)
    /-
      case inl
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs : ∀ (a' : α), LT.lt a' b → Exists fun a => And (LT.lt a b) (HasSubset.Subse …
      hb' : Eq (Set.Iio b) EmptyCollection.emptyCollection
      ⊢ Eq (Filter.map Subtype.val Filter.atTop) (nhdsWithin b (Set.Iio b))
    -/
  · have : IsEmpty s := ⟨fun x => hb'.subset (hb x.2)⟩
    /-
      case inl
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs : ∀ (a' : α), LT.lt a' b → Exists fun a => And (LT.lt a b) (HasSubset.Subse …
      hb' : Eq (Set.Iio b) EmptyCollection.emptyCollection
      this : IsEmpty ↑s
      ⊢ Eq (Filter.map Subtype.val Filter.atTop) (nhdsWithin b (Set.Iio b))
    -/
    rw [filter_eq_bot_of_isEmpty atTop, Filter.map_bot, hb', nhdsWithin_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs : ∀ (a' : α), LT.lt a' b → Exists fun a => And (LT.lt a b) (HasSubset.Subse …
      a : α
      ha : Membership.mem (Set.Iio b) a
      ⊢ Eq (Filter.map Subtype.val Filter.atTop) (nhdsWithin b (Set.Iio b))
    -/
  · rw [← comap_coe_nhdsLT_of_Ioo_subset hb fun _ => hs a ha, map_comap_of_mem]
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs : ∀ (a' : α), LT.lt a' b → Exists fun a => And (LT.lt a b) (HasSubset.Subse …
      a : α
      ha : Membership.mem (Set.Iio b) a
      ⊢ Membership.mem (nhdsWithin b (Set.Iio b)) (Set.range Subtype.val)
    -/
    rw [Subtype.range_val]
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      b : α
      s : Set α
      hb : HasSubset.Subset s (Set.Iio b)
      hs : ∀ (a' : α), LT.lt a' b → Exists fun a => And (LT.lt a b) (HasSubset.Subse …
      a : α
      ha : Membership.mem (Set.Iio b) a
      ⊢ Membership.mem (nhdsWithin b (Set.Iio b)) s
    -/
    exact (mem_nhdsLT_iff_exists_Ioo_subset' ha).2 (hs a ha)
    /-
      🎉 no goals
    -/


theorem map_coe_atBot_of_Ioo_subset (ha : s ⊆ Ioi a) (hs : ∀ b' > a, ∃ b > a, Ioo a b ⊆ s) :
    map ((↑) : s → α) atBot = 𝓝[>] a := by
  -- the elaborator gets stuck without `(... : _)`
  refine (map_coe_atTop_of_Ioo_subset (show ofDual ⁻¹' s ⊆ Iio (toDual a) from ha)
    fun b' hb' => ?_ : _)
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    s : Set α
    ha : HasSubset.Subset s (Set.Ioi a)
    hs : ∀ (b' : α), GT.gt b' a → Exists fun b => And (GT.gt b a) (HasSubset.Subse …
    b' : OrderDual α
    hb' : LT.lt b' (OrderDual.toDual a)
    ⊢ Exists fun a_1 => And (LT.lt a_1 (OrderDual.toDual a)) (HasSubset.Subset (Se …
  -/
  simpa only [OrderDual.exists, dual_Ioo] using hs b' hb'
  /-
    🎉 no goals
  -/


/-- The `atTop` filter for an open interval `Ioo a b` comes from the left-neighbourhoods filter at
the right endpoint in the ambient order. -/
theorem comap_coe_Ioo_nhdsLT (a b : α) : comap ((↑) : Ioo a b → α) (𝓝[<] b) = atTop :=
  comap_coe_nhdsLT_of_Ioo_subset Ioo_subset_Iio_self fun h => ⟨a, nonempty_Ioo.1 h, Subset.refl _⟩


@[deprecated (since := "2024-12-22")]
alias comap_coe_Ioo_nhdsWithin_Iio := comap_coe_Ioo_nhdsLT


/-- The `atBot` filter for an open interval `Ioo a b` comes from the right-neighbourhoods filter at
the left endpoint in the ambient order. -/
theorem comap_coe_Ioo_nhdsGT (a b : α) : comap ((↑) : Ioo a b → α) (𝓝[>] a) = atBot :=
  comap_coe_nhdsGT_of_Ioo_subset Ioo_subset_Ioi_self fun h => ⟨b, nonempty_Ioo.1 h, Subset.refl _⟩


@[deprecated (since := "2024-12-22")]
alias comap_coe_Ioo_nhdsWithin_Ioi := comap_coe_Ioo_nhdsGT


theorem comap_coe_Ioi_nhdsGT (a : α) : comap ((↑) : Ioi a → α) (𝓝[>] a) = atBot :=
  comap_coe_nhdsGT_of_Ioo_subset (Subset.refl _) fun ⟨x, hx⟩ => ⟨x, hx, Ioo_subset_Ioi_self⟩


@[deprecated (since := "2024-12-22")]
alias comap_coe_Ioi_nhdsWithin_Ioi := comap_coe_Ioi_nhdsGT


theorem comap_coe_Iio_nhdsLT (a : α) : comap ((↑) : Iio a → α) (𝓝[<] a) = atTop :=
  comap_coe_Ioi_nhdsGT (α := αᵒᵈ) a


@[deprecated (since := "2024-12-22")]
alias comap_coe_Iio_nhdsWithin_Iio := comap_coe_Iio_nhdsLT


@[simp]
theorem map_coe_Ioo_atTop {a b : α} (h : a < b) : map ((↑) : Ioo a b → α) atTop = 𝓝[<] b :=
  map_coe_atTop_of_Ioo_subset Ioo_subset_Iio_self fun _ _ => ⟨_, h, Subset.refl _⟩


@[simp]
theorem map_coe_Ioo_atBot {a b : α} (h : a < b) : map ((↑) : Ioo a b → α) atBot = 𝓝[>] a :=
  map_coe_atBot_of_Ioo_subset Ioo_subset_Ioi_self fun _ _ => ⟨_, h, Subset.refl _⟩


@[simp]
theorem map_coe_Ioi_atBot (a : α) : map ((↑) : Ioi a → α) atBot = 𝓝[>] a :=
  map_coe_atBot_of_Ioo_subset (Subset.refl _) fun b hb => ⟨b, hb, Ioo_subset_Ioi_self⟩


@[simp]
theorem map_coe_Iio_atTop (a : α) : map ((↑) : Iio a → α) atTop = 𝓝[<] a :=
  map_coe_Ioi_atBot (α := αᵒᵈ) _


@[simp]
theorem tendsto_comp_coe_Ioo_atTop (h : a < b) :
    Tendsto (fun x : Ioo a b => f x) atTop l ↔ Tendsto f (𝓝[<] b) l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    l : Filter β
    f : α → β
    h : LT.lt a b
    ⊢ Iff (Filter.Tendsto (fun x => f ↑x) Filter.atTop l) (Filter.Tendsto f (nhdsW …
  -/
  rw [← map_coe_Ioo_atTop h, tendsto_map'_iff]; rfl
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem tendsto_comp_coe_Ioo_atBot (h : a < b) :
    Tendsto (fun x : Ioo a b => f x) atBot l ↔ Tendsto f (𝓝[>] a) l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    l : Filter β
    f : α → β
    h : LT.lt a b
    ⊢ Iff (Filter.Tendsto (fun x => f ↑x) Filter.atBot l) (Filter.Tendsto f (nhdsW …
  -/
  rw [← map_coe_Ioo_atBot h, tendsto_map'_iff]; rfl
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem tendsto_comp_coe_Ioi_atBot :
    Tendsto (fun x : Ioi a => f x) atBot l ↔ Tendsto f (𝓝[>] a) l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    l : Filter β
    f : α → β
    ⊢ Iff (Filter.Tendsto (fun x => f ↑x) Filter.atBot l) (Filter.Tendsto f (nhdsW …
  -/
  rw [← map_coe_Ioi_atBot, tendsto_map'_iff]; rfl
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem tendsto_comp_coe_Iio_atTop :
    Tendsto (fun x : Iio a => f x) atTop l ↔ Tendsto f (𝓝[<] a) l := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    l : Filter β
    f : α → β
    ⊢ Iff (Filter.Tendsto (fun x => f ↑x) Filter.atTop l) (Filter.Tendsto f (nhdsW …
  -/
  rw [← map_coe_Iio_atTop, tendsto_map'_iff]; rfl
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem tendsto_Ioo_atTop {f : β → Ioo a b} :
    Tendsto f l atTop ↔ Tendsto (fun x => (f x : α)) l (𝓝[<] b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    l : Filter β
    f : β → ↑(Set.Ioo a b)
    ⊢ Iff (Filter.Tendsto f l Filter.atTop) (Filter.Tendsto (fun x => ↑(f x)) l (n …
  -/
  rw [← comap_coe_Ioo_nhdsLT, tendsto_comap_iff]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem tendsto_Ioo_atBot {f : β → Ioo a b} :
    Tendsto f l atBot ↔ Tendsto (fun x => (f x : α)) l (𝓝[>] a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    l : Filter β
    f : β → ↑(Set.Ioo a b)
    ⊢ Iff (Filter.Tendsto f l Filter.atBot) (Filter.Tendsto (fun x => ↑(f x)) l (n …
  -/
  rw [← comap_coe_Ioo_nhdsGT, tendsto_comap_iff]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem tendsto_Ioi_atBot {f : β → Ioi a} :
    Tendsto f l atBot ↔ Tendsto (fun x => (f x : α)) l (𝓝[>] a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    l : Filter β
    f : β → ↑(Set.Ioi a)
    ⊢ Iff (Filter.Tendsto f l Filter.atBot) (Filter.Tendsto (fun x => ↑(f x)) l (n …
  -/
  rw [← comap_coe_Ioi_nhdsGT, tendsto_comap_iff]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem tendsto_Iio_atTop {f : β → Iio a} :
    Tendsto f l atTop ↔ Tendsto (fun x => (f x : α)) l (𝓝[<] a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a : α
    l : Filter β
    f : β → ↑(Set.Iio a)
    ⊢ Iff (Filter.Tendsto f l Filter.atTop) (Filter.Tendsto (fun x => ↑(f x)) l (n …
  -/
  rw [← comap_coe_Iio_nhdsLT, tendsto_comap_iff]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


instance (x : α) [Nontrivial α] : NeBot (𝓝[≠] x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a b : α
    s : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    ⊢ (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
  -/
  refine forall_mem_nonempty_iff_neBot.1 fun s hs => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a b : α
    s✝ : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    s : Set α
    hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
    ⊢ s.Nonempty
  -/
  obtain ⟨u, u_open, xu, us⟩ : ∃ u : Set α, IsOpen u ∧ x ∈ u ∧ u ∩ {x}ᶜ ⊆ s := mem_nhdsWithin.1 hs
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a b : α
    s✝ : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    s : Set α
    hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    us : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) s
    ⊢ s.Nonempty
  -/
  obtain ⟨a, b, a_lt_b, hab⟩ : ∃ a b : α, a < b ∧ Ioo a b ⊆ u := u_open.exists_Ioo_subset ⟨x, xu⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a✝ b✝ : α
    s✝ : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    s : Set α
    hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    us : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) s
    a b : α
    a_lt_b : LT.lt a b
    hab : HasSubset.Subset (Set.Ioo a b) u
    ⊢ s.Nonempty
  -/
  obtain ⟨y, hy⟩ : ∃ y, a < y ∧ y < b := exists_between a_lt_b
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a✝ b✝ : α
    s✝ : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    s : Set α
    hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    us : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) s
    a b : α
    a_lt_b : LT.lt a b
    hab : HasSubset.Subset (Set.Ioo a b) u
    y : α
    hy : And (LT.lt a y) (LT.lt y b)
    ⊢ s.Nonempty
  -/
  rcases ne_or_eq x y with (xy | rfl)
    /-
      case intro.intro.intro.intro.intro.intro.intro.inl
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : DenselyOrdered α
      a✝ b✝ : α
      s✝ : Set α
      l : Filter β
      f : α → β
      x : α
      inst✝ : Nontrivial α
      s : Set α
      hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
      u : Set α
      u_open : IsOpen u
      xu : Membership.mem u x
      us : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) s
      a b : α
      a_lt_b : LT.lt a b
      hab : HasSubset.Subset (Set.Ioo a b) u
      y : α
      hy : And (LT.lt a y) (LT.lt y b)
      xy : Ne x y
      ⊢ s.Nonempty
    -/
  · exact ⟨y, us ⟨hab hy, xy.symm⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.intro.intro.intro.inr
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a✝ b✝ : α
    s✝ : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    s : Set α
    hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    us : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) s
    a b : α
    a_lt_b : LT.lt a b
    hab : HasSubset.Subset (Set.Ioo a b) u
    hy : And (LT.lt a x) (LT.lt x b)
    ⊢ s.Nonempty
  -/
  obtain ⟨z, hz⟩ : ∃ z, a < z ∧ z < x := exists_between hy.1
  /-
    case intro.intro.intro.intro.intro.intro.intro.inr.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    a✝ b✝ : α
    s✝ : Set α
    l : Filter β
    f : α → β
    x : α
    inst✝ : Nontrivial α
    s : Set α
    hs : Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) s
    u : Set α
    u_open : IsOpen u
    xu : Membership.mem u x
    us : HasSubset.Subset (Inter.inter u (HasCompl.compl (Singleton.singleton x))) s
    a b : α
    a_lt_b : LT.lt a b
    hab : HasSubset.Subset (Set.Ioo a b) u
    hy : And (LT.lt a x) (LT.lt x b)
    z : α
    hz : And (LT.lt a z) (LT.lt z x)
    ⊢ s.Nonempty
  -/
  exact ⟨z, us ⟨hab ⟨hz.1, hz.2.trans hy.2⟩, hz.2.ne⟩⟩
  /-
    🎉 no goals
  -/


/-- Let `s` be a dense set in a nontrivial dense linear order `α`. If `s` is a
separable space (e.g., if `α` has a second countable topology), then there exists a countable
dense subset `t ⊆ s` such that `t` does not contain bottom/top elements of `α`. -/
theorem Dense.exists_countable_dense_subset_no_bot_top [Nontrivial α] {s : Set α} [SeparableSpace s]
    (hs : Dense s) :
    ∃ t, t ⊆ s ∧ t.Countable ∧ Dense t ∧ (∀ x, IsBot x → x ∉ t) ∧ ∀ x, IsTop x → x ∉ t := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : Nontrivial α
    s : Set α
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hs : Dense s
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (And (Dense t) ( …
  -/
  rcases hs.exists_countable_dense_subset with ⟨t, hts, htc, htd⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : Nontrivial α
    s : Set α
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    hs : Dense s
    t : Set α
    hts : HasSubset.Subset t s
    htc : t.Countable
    htd : Dense t
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (And (Dense t) ( …
  -/
  refine ⟨t \ ({ x | IsBot x } ∪ { x | IsTop x }), ?_, ?_, ?_, fun x hx => ?_, fun x hx => ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderTopology α
      inst✝² : DenselyOrdered α
      inst✝¹ : Nontrivial α
      s : Set α
      inst✝ : TopologicalSpace.SeparableSpace ↑s
      hs : Dense s
      t : Set α
      hts : HasSubset.Subset t s
      htc : t.Countable
      htd : Dense t
      ⊢ HasSubset.Subset (SDiff.sdiff t (Union.union (setOf fun x => IsBot x) (setOf …
    -/
  · exact diff_subset.trans hts
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderTopology α
      inst✝² : DenselyOrdered α
      inst✝¹ : Nontrivial α
      s : Set α
      inst✝ : TopologicalSpace.SeparableSpace ↑s
      hs : Dense s
      t : Set α
      hts : HasSubset.Subset t s
      htc : t.Countable
      htd : Dense t
      ⊢ (SDiff.sdiff t (Union.union (setOf fun x => IsBot x) (setOf fun x => IsTop x …
    -/
  · exact htc.mono diff_subset
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderTopology α
      inst✝² : DenselyOrdered α
      inst✝¹ : Nontrivial α
      s : Set α
      inst✝ : TopologicalSpace.SeparableSpace ↑s
      hs : Dense s
      t : Set α
      hts : HasSubset.Subset t s
      htc : t.Countable
      htd : Dense t
      ⊢ Dense (SDiff.sdiff t (Union.union (setOf fun x => IsBot x) (setOf fun x => I …
    -/
  · exact htd.diff_finite ((subsingleton_isBot α).finite.union (subsingleton_isTop α).finite)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_4
      α : Type u_1
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderTopology α
      inst✝² : DenselyOrdered α
      inst✝¹ : Nontrivial α
      s : Set α
      inst✝ : TopologicalSpace.SeparableSpace ↑s
      hs : Dense s
      t : Set α
      hts : HasSubset.Subset t s
      htc : t.Countable
      htd : Dense t
      x : α
      hx : IsBot x
      ⊢ Not (Membership.mem (SDiff.sdiff t (Union.union (setOf fun x => IsBot x) (se …
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_5
      α : Type u_1
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderTopology α
      inst✝² : DenselyOrdered α
      inst✝¹ : Nontrivial α
      s : Set α
      inst✝ : TopologicalSpace.SeparableSpace ↑s
      hs : Dense s
      t : Set α
      hts : HasSubset.Subset t s
      htc : t.Countable
      htd : Dense t
      x : α
      hx : IsTop x
      ⊢ Not (Membership.mem (SDiff.sdiff t (Union.union (setOf fun x => IsBot x) (se …
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/


/-- If `α` is a nontrivial separable dense linear order, then there exists a
countable dense set `s : Set α` that contains neither top nor bottom elements of `α`.
For a dense set containing both bot and top elements, see
`exists_countable_dense_bot_top`. -/
theorem exists_countable_dense_no_bot_top [SeparableSpace α] [Nontrivial α] :
    ∃ s : Set α, s.Countable ∧ Dense s ∧ (∀ x, IsBot x → x ∉ s) ∧ ∀ x, IsTop x → x ∉ s := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : TopologicalSpace.SeparableSpace α
    inst✝ : Nontrivial α
    ⊢ Exists fun s => And s.Countable (And (Dense s) (And (∀ (x : α), IsBot x → No …
  -/
  simpa using dense_univ.exists_countable_dense_subset_no_bot_top
  /-
    🎉 no goals
  -/


