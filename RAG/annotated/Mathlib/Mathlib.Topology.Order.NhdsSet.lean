@[simp] theorem nhdsSet_Ioi : 𝓝ˢ (Ioi a) = 𝓟 (Ioi a) := isOpen_Ioi.nhdsSet_eq

@[simp] theorem nhdsSet_Iio : 𝓝ˢ (Iio a) = 𝓟 (Iio a) := isOpen_Iio.nhdsSet_eq

@[simp] theorem nhdsSet_Ioo : 𝓝ˢ (Ioo a b) = 𝓟 (Ioo a b) := isOpen_Ioo.nhdsSet_eq


theorem nhdsSet_Ici : 𝓝ˢ (Ici a) = 𝓝 a ⊔ 𝓟 (Ioi a) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    a : α
    ⊢ Eq (nhdsSet (Set.Ici a)) (Max.max (nhds a) (Filter.principal (Set.Ioi a)))
  -/
  rw [← Ioi_insert, nhdsSet_insert, nhdsSet_Ioi]
  /-
    🎉 no goals
  -/


theorem nhdsSet_Iic : 𝓝ˢ (Iic a) = 𝓝 a ⊔ 𝓟 (Iio a) := nhdsSet_Ici (α := αᵒᵈ)


theorem nhdsSet_Ico (h : a < b) : 𝓝ˢ (Ico a b) = 𝓝 a ⊔ 𝓟 (Ioo a b) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    a b : α
    h : LT.lt a b
    ⊢ Eq (nhdsSet (Set.Ico a b)) (Max.max (nhds a) (Filter.principal (Set.Ioo a b)))
  -/
  rw [← Ioo_insert_left h, nhdsSet_insert, nhdsSet_Ioo]
  /-
    🎉 no goals
  -/


theorem nhdsSet_Ioc (h : a < b) : 𝓝ˢ (Ioc a b) = 𝓝 b ⊔ 𝓟 (Ioo a b) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    a b : α
    h : LT.lt a b
    ⊢ Eq (nhdsSet (Set.Ioc a b)) (Max.max (nhds b) (Filter.principal (Set.Ioo a b)))
  -/
  rw [← Ioo_insert_right h, nhdsSet_insert, nhdsSet_Ioo]
  /-
    🎉 no goals
  -/


theorem nhdsSet_Icc (h : a ≤ b) : 𝓝ˢ (Icc a b) = 𝓝 a ⊔ 𝓝 b ⊔ 𝓟 (Ioo a b) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    a b : α
    h : LE.le a b
    ⊢ Eq (nhdsSet (Set.Icc a b)) (Max.max (Max.max (nhds a) (nhds b)) (Filter.prin …
  -/
  rcases h.eq_or_lt with rfl | hlt
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      a : α
      h : LE.le a a
      ⊢ Eq (nhdsSet (Set.Icc a a)) (Max.max (Max.max (nhds a) (nhds a)) (Filter.prin …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : LinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderClosedTopology α
      a b : α
      h : LE.le a b
      hlt : LT.lt a b
      ⊢ Eq (nhdsSet (Set.Icc a b)) (Max.max (Max.max (nhds a) (nhds b)) (Filter.prin …
    -/
  · rw [← Ioc_insert_left h, nhdsSet_insert, nhdsSet_Ioc hlt, sup_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem Ioi_mem_nhdsSet_Ici_iff : Ioi a ∈ 𝓝ˢ (Ici b) ↔ a < b := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    a b : α
    ⊢ Iff (Membership.mem (nhdsSet (Set.Ici b)) (Set.Ioi a)) (LT.lt a b)
  -/
  rw [isOpen_Ioi.mem_nhdsSet, Ici_subset_Ioi]
  /-
    🎉 no goals
  -/


alias ⟨_, Ioi_mem_nhdsSet_Ici⟩ := Ioi_mem_nhdsSet_Ici_iff


theorem Ici_mem_nhdsSet_Ici (h : a < b) : Ici a ∈ 𝓝ˢ (Ici b) :=
  mem_of_superset (Ioi_mem_nhdsSet_Ici h) Ioi_subset_Ici_self


theorem Iio_mem_nhdsSet_Iic_iff : Iio b ∈ 𝓝ˢ (Iic a) ↔ a < b :=
  Ioi_mem_nhdsSet_Ici_iff (α := αᵒᵈ)


alias ⟨_, Iio_mem_nhdsSet_Iic⟩ := Iio_mem_nhdsSet_Iic_iff


theorem Iic_mem_nhdsSet_Iic (h : a < b) : Iic b ∈ 𝓝ˢ (Iic a) :=
  Ici_mem_nhdsSet_Ici (α := αᵒᵈ) h


theorem Ioi_mem_nhdsSet_Icc (h : a < b) : Ioi a ∈ 𝓝ˢ (Icc b c) :=
  nhdsSet_mono Icc_subset_Ici_self <| Ioi_mem_nhdsSet_Ici h


theorem Ici_mem_nhdsSet_Icc (h : a < b) : Ici a ∈ 𝓝ˢ (Icc b c) :=
  mem_of_superset (Ioi_mem_nhdsSet_Icc h) Ioi_subset_Ici_self


theorem Iio_mem_nhdsSet_Icc (h : b < c) : Iio c ∈ 𝓝ˢ (Icc a b) :=
  nhdsSet_mono Icc_subset_Iic_self <| Iio_mem_nhdsSet_Iic h


theorem Iic_mem_nhdsSet_Icc (h : b < c) : Iic c ∈ 𝓝ˢ (Icc a b) :=
  mem_of_superset (Iio_mem_nhdsSet_Icc h) Iio_subset_Iic_self


theorem Ioo_mem_nhdsSet_Icc (h : a < b) (h' : c < d) : Ioo a d ∈ 𝓝ˢ (Icc b c) :=
  inter_mem (Ioi_mem_nhdsSet_Icc h) (Iio_mem_nhdsSet_Icc h')


theorem Ico_mem_nhdsSet_Icc (h : a < b) (h' : c < d) : Ico a d ∈ 𝓝ˢ (Icc b c) :=
  inter_mem (Ici_mem_nhdsSet_Icc h) (Iio_mem_nhdsSet_Icc h')


theorem Ioc_mem_nhdsSet_Icc (h : a < b) (h' : c < d) : Ioc a d ∈ 𝓝ˢ (Icc b c) :=
  inter_mem (Ioi_mem_nhdsSet_Icc h) (Iic_mem_nhdsSet_Icc h')


theorem Icc_mem_nhdsSet_Icc (h : a < b) (h' : c < d) : Icc a d ∈ 𝓝ˢ (Icc b c) :=
  inter_mem (Ici_mem_nhdsSet_Icc h) (Iic_mem_nhdsSet_Icc h')


theorem Ici_mem_nhdsSet_Ico (h : a < b) : Ici a ∈ 𝓝ˢ (Ico b c) :=
  nhdsSet_mono Ico_subset_Icc_self <| Ici_mem_nhdsSet_Icc h


theorem Ioi_mem_nhdsSet_Ico (h : a < b) : Ioi a ∈ 𝓝ˢ (Ico b c) :=
  nhdsSet_mono Ico_subset_Icc_self <| Ioi_mem_nhdsSet_Icc h


theorem Iio_mem_nhdsSet_Ico (h : b ≤ c) : Iio c ∈ 𝓝ˢ (Ico a b) :=
                                         /-
                                           α : Type u_1
                                           inst✝² : LinearOrder α
                                           inst✝¹ : TopologicalSpace α
                                           inst✝ : OrderClosedTopology α
                                           a b c : α
                                           h : LE.le b c
                                           ⊢ Membership.mem (nhdsSet (Set.Iio b)) (Set.Iio c)
                                         -/
  nhdsSet_mono Ico_subset_Iio_self <| by simpa
                                         /-
                                           🎉 no goals
                                         -/


theorem Iic_mem_nhdsSet_Ico (h : b ≤ c) : Iic c ∈ 𝓝ˢ (Ico a b) :=
  mem_of_superset (Iio_mem_nhdsSet_Ico h) Iio_subset_Iic_self


theorem Ioo_mem_nhdsSet_Ico (h : a < b) (h' : c ≤ d) : Ioo a d ∈ 𝓝ˢ (Ico b c) :=
  inter_mem (Ioi_mem_nhdsSet_Ico h) (Iio_mem_nhdsSet_Ico h')


theorem Icc_mem_nhdsSet_Ico (h : a < b) (h' : c ≤ d) : Icc a d ∈ 𝓝ˢ (Ico b c) :=
  inter_mem (Ici_mem_nhdsSet_Ico h) (Iic_mem_nhdsSet_Ico h')


theorem Ioc_mem_nhdsSet_Ico (h : a < b) (h' : c ≤ d) : Ioc a d ∈ 𝓝ˢ (Ico b c) :=
  inter_mem (Ioi_mem_nhdsSet_Ico h) (Iic_mem_nhdsSet_Ico h')


theorem Ico_mem_nhdsSet_Ico (h : a < b) (h' : c ≤ d) : Ico a d ∈ 𝓝ˢ (Ico b c) :=
  inter_mem (Ici_mem_nhdsSet_Ico h) (Iio_mem_nhdsSet_Ico h')


theorem Ioi_mem_nhdsSet_Ioc (h : a ≤ b) : Ioi a ∈ 𝓝ˢ (Ioc b c) :=
                                         /-
                                           α : Type u_1
                                           inst✝² : LinearOrder α
                                           inst✝¹ : TopologicalSpace α
                                           inst✝ : OrderClosedTopology α
                                           a b c : α
                                           h : LE.le a b
                                           ⊢ Membership.mem (nhdsSet (Set.Ioi b)) (Set.Ioi a)
                                         -/
  nhdsSet_mono Ioc_subset_Ioi_self <| by simpa
                                         /-
                                           🎉 no goals
                                         -/


theorem Iio_mem_nhdsSet_Ioc (h : b < c) : Iio c ∈ 𝓝ˢ (Ioc a b) :=
  nhdsSet_mono Ioc_subset_Icc_self <| Iio_mem_nhdsSet_Icc h


theorem Ici_mem_nhdsSet_Ioc (h : a ≤ b) : Ici a ∈ 𝓝ˢ (Ioc b c) :=
  mem_of_superset (Ioi_mem_nhdsSet_Ioc h) Ioi_subset_Ici_self


theorem Iic_mem_nhdsSet_Ioc (h : b < c) : Iic c ∈ 𝓝ˢ (Ioc a b) :=
  nhdsSet_mono Ioc_subset_Icc_self <| Iic_mem_nhdsSet_Icc h


theorem Ioo_mem_nhdsSet_Ioc (h : a ≤ b) (h' : c < d) : Ioo a d ∈ 𝓝ˢ (Ioc b c) :=
  inter_mem (Ioi_mem_nhdsSet_Ioc h) (Iio_mem_nhdsSet_Ioc h')


theorem Icc_mem_nhdsSet_Ioc (h : a ≤ b) (h' : c < d) : Icc a d ∈ 𝓝ˢ (Ioc b c) :=
  inter_mem (Ici_mem_nhdsSet_Ioc h) (Iic_mem_nhdsSet_Ioc h')


theorem Ioc_mem_nhdsSet_Ioc (h : a ≤ b) (h' : c < d) : Ioc a d ∈ 𝓝ˢ (Ioc b c) :=
  inter_mem (Ioi_mem_nhdsSet_Ioc h) (Iic_mem_nhdsSet_Ioc h')


theorem Ico_mem_nhdsSet_Ioc (h : a ≤ b) (h' : c < d) : Ico a d ∈ 𝓝ˢ (Ioc b c) :=
  inter_mem (Ici_mem_nhdsSet_Ioc h) (Iio_mem_nhdsSet_Ioc h')


theorem hasBasis_nhdsSet_Iic_Iio (a : α) [h : Nonempty (Ioi a)] :
    HasBasis (𝓝ˢ (Iic a)) (a < ·) Iio := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a : α
    h : Nonempty ↑(Set.Ioi a)
    ⊢ (nhdsSet (Set.Iic a)).HasBasis (fun x => LT.lt a x) Set.Iio
  -/
  refine ⟨fun s ↦ ⟨fun hs ↦ ?_, fun ⟨b, hab, hb⟩ ↦ mem_of_superset (Iio_mem_nhdsSet_Iic hab) hb⟩⟩
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a : α
    h : Nonempty ↑(Set.Ioi a)
    s : Set α
    hs : Membership.mem (nhdsSet (Set.Iic a)) s
    ⊢ Exists fun i => And (LT.lt a i) (HasSubset.Subset (Set.Iio i) s)
  -/
  rw [nhdsSet_Iic, mem_sup, mem_principal] at hs
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a : α
    h : Nonempty ↑(Set.Ioi a)
    s : Set α
    hs : And (Membership.mem (nhds a) s) (HasSubset.Subset (Set.Iio a) s)
    ⊢ Exists fun i => And (LT.lt a i) (HasSubset.Subset (Set.Iio i) s)
  -/
  rcases exists_Ico_subset_of_mem_nhds hs.1 (Set.nonempty_coe_sort.1 h) with ⟨b, hab, hbs⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a : α
    h : Nonempty ↑(Set.Ioi a)
    s : Set α
    hs : And (Membership.mem (nhds a) s) (HasSubset.Subset (Set.Iio a) s)
    b : α
    hab : LT.lt a b
    hbs : HasSubset.Subset (Set.Ico a b) s
    ⊢ Exists fun i => And (LT.lt a i) (HasSubset.Subset (Set.Iio i) s)
  -/
  exact ⟨b, hab, Iio_subset_Iio_union_Ico.trans (union_subset hs.2 hbs)⟩
  /-
    🎉 no goals
  -/


theorem hasBasis_nhdsSet_Iic_Iic (a : α) [NeBot (𝓝[>] a)] :
    HasBasis (𝓝ˢ (Iic a)) (a < ·) Iic := by
  have : Nonempty (Ioi a) :=
    (Filter.nonempty_of_mem (self_mem_nhdsWithin : Ioi a ∈ 𝓝[>] a)).to_subtype
  refine (hasBasis_nhdsSet_Iic_Iio _).to_hasBasis
    (fun c hc ↦ ?_) (fun _ h ↦ ⟨_, h, Iio_subset_Iic_self⟩)
  /-
    α : Type u_1
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    a : α
    inst✝ : (nhdsWithin a (Set.Ioi a)).NeBot
    this : Nonempty ↑(Set.Ioi a)
    c : α
    hc : LT.lt a c
    ⊢ Exists fun i' => And (LT.lt a i') (HasSubset.Subset (Set.Iic i') (Set.Iio c))
  -/
  simpa only [Iic_subset_Iio] using Filter.nonempty_of_mem (Ioo_mem_nhdsGT hc)
  /-
    🎉 no goals
  -/


@[simp]
theorem Iic_mem_nhdsSet_Iic_iff {a b : α} [NeBot (𝓝[>] b)] : Iic a ∈ 𝓝ˢ (Iic b) ↔ b < a :=
  (hasBasis_nhdsSet_Iic_Iic b).mem_iff.trans
    ⟨fun ⟨_c, hbc, hca⟩ ↦ hbc.trans_le (Iic_subset_Iic.1 hca), fun h ↦ ⟨_, h, Subset.rfl⟩⟩


theorem hasBasis_nhdsSet_Ici_Ioi (a : α) [Nonempty (Iio a)] :
    HasBasis (𝓝ˢ (Ici a)) (· < a) Ioi :=
  have : Nonempty (Ioi (toDual a)) := ‹_›; hasBasis_nhdsSet_Iic_Iio (toDual a)


theorem hasBasis_nhdsSet_Ici_Ici (a : α) [NeBot (𝓝[<] a)] :
    HasBasis (𝓝ˢ (Ici a)) (· < a) Ici :=
  have : NeBot (𝓝[>] (toDual a)) := ‹_›; hasBasis_nhdsSet_Iic_Iic (toDual a)


@[simp]
theorem Ici_mem_nhdsSet_Ici_iff {a b : α} [NeBot (𝓝[<] b)] : Ici a ∈ 𝓝ˢ (Ici b) ↔ a < b :=
  have : NeBot (𝓝[>] (toDual b)) := ‹_›; Iic_mem_nhdsSet_Iic_iff (a := toDual a) (b := toDual b)

