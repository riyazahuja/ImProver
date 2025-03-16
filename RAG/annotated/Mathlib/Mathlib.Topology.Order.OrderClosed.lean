/-- If `α` is a topological space and a preorder, `ClosedIicTopology α` means that `Iic a` is
closed for all `a : α`. -/
class ClosedIicTopology (α : Type*) [TopologicalSpace α] [Preorder α] : Prop where
  /-- For any `a`, the set `(-∞, a]` is closed. -/
  isClosed_Iic (a : α) : IsClosed (Iic a)


/-- If `α` is a topological space and a preorder, `ClosedIciTopology α` means that `Ici a` is
closed for all `a : α`. -/
class ClosedIciTopology (α : Type*) [TopologicalSpace α] [Preorder α] : Prop where
  /-- For any `a`, the set `[a, +∞)` is closed. -/
  isClosed_Ici (a : α) : IsClosed (Ici a)


/-- A topology on a set which is both a topological space and a preorder is _order-closed_ if the
set of points `(x, y)` with `x ≤ y` is closed in the product space. We introduce this as a mixin.
This property is satisfied for the order topology on a linear order, but it can be satisfied more
generally, and suffices to derive many interesting properties relating order and topology. -/
class OrderClosedTopology (α : Type*) [TopologicalSpace α] [Preorder α] : Prop where
  /-- The set `{ (x, y) | x ≤ y }` is a closed set. -/
  isClosed_le' : IsClosed { p : α × α | p.1 ≤ p.2 }


instance [TopologicalSpace α] [h : FirstCountableTopology α] : FirstCountableTopology αᵒᵈ := h

instance [TopologicalSpace α] [h : SecondCountableTopology α] : SecondCountableTopology αᵒᵈ := h


theorem Dense.orderDual [TopologicalSpace α] {s : Set α} (hs : Dense s) :
    Dense (OrderDual.ofDual ⁻¹' s) :=
  hs


protected lemma BddAbove.of_closure : BddAbove (closure s) → BddAbove s :=
  BddAbove.mono subset_closure


protected lemma BddBelow.of_closure : BddBelow (closure s) → BddBelow s :=
  BddBelow.mono subset_closure


theorem isClosed_Iic : IsClosed (Iic a) :=
  ClosedIicTopology.isClosed_Iic a


@[deprecated isClosed_Iic (since := "2024-02-15")]
lemma ClosedIicTopology.isClosed_le' (a : α) : IsClosed {x | x ≤ a} := isClosed_Iic a

instance : ClosedIciTopology αᵒᵈ where
  isClosed_Ici _ := isClosed_Iic (α := α)


@[simp]
theorem closure_Iic (a : α) : closure (Iic a) = Iic a :=
  isClosed_Iic.closure_eq


theorem le_of_tendsto_of_frequently {x : Filter β} (lim : Tendsto f x (𝓝 a))
    (h : ∃ᶠ c in x, f c ≤ b) : a ≤ b :=
  isClosed_Iic.mem_of_frequently_of_tendsto h lim


theorem le_of_tendsto {x : Filter β} [NeBot x] (lim : Tendsto f x (𝓝 a))
    (h : ∀ᶠ c in x, f c ≤ b) : a ≤ b :=
  isClosed_Iic.mem_of_tendsto lim h


theorem le_of_tendsto' {x : Filter β} [NeBot x] (lim : Tendsto f x (𝓝 a))
    (h : ∀ c, f c ≤ b) : a ≤ b :=
  le_of_tendsto lim (Eventually.of_forall h)


@[simp] lemma upperBounds_closure (s : Set α) : upperBounds (closure s : Set α) = upperBounds s :=
                 /-
                   α : Type u
                   inst✝² : TopologicalSpace α
                   inst✝¹ : Preorder α
                   inst✝ : ClosedIicTopology α
                   s : Set α
                   a : α
                   ⊢ Iff (Membership.mem (upperBounds (closure s)) a) (Membership.mem (upperBound …
                 -/
  ext fun a ↦ by simp_rw [mem_upperBounds_iff_subset_Iic, isClosed_Iic.closure_subset_iff]
                 /-
                   🎉 no goals
                 -/


@[simp] lemma bddAbove_closure : BddAbove (closure s) ↔ BddAbove s := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ClosedIicTopology α
    s : Set α
    ⊢ Iff (BddAbove (closure s)) (BddAbove s)
  -/
  simp_rw [BddAbove, upperBounds_closure]
  /-
    🎉 no goals
  -/


protected alias ⟨_, BddAbove.closure⟩ := bddAbove_closure


@[simp]
theorem disjoint_nhds_atBot_iff : Disjoint (𝓝 a) atBot ↔ ¬IsBot a := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ClosedIicTopology α
    a : α
    ⊢ Iff (Disjoint (nhds a) Filter.atBot) (Not (IsBot a))
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a : α
      ⊢ Disjoint (nhds a) Filter.atBot → Not (IsBot a)
    -/
  · intro hd hbot
    /-
      case mp
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a : α
      hd : Disjoint (nhds a) Filter.atBot
      hbot : IsBot a
      ⊢ False
    -/
    rw [hbot.atBot_eq, disjoint_principal_right] at hd
    /-
      case mp
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a : α
      hd : Membership.mem (nhds a) (HasCompl.compl (Set.Iic a))
      hbot : IsBot a
      ⊢ False
    -/
    exact mem_of_mem_nhds hd le_rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a : α
      ⊢ Not (IsBot a) → Disjoint (nhds a) Filter.atBot
    -/
  · simp only [IsBot, not_forall]
    /-
      case mpr
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a : α
      ⊢ (Exists fun x => Not (LE.le a x)) → Disjoint (nhds a) Filter.atBot
    -/
    rintro ⟨b, hb⟩
    /-
      case mpr.intro
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a b : α
      hb : Not (LE.le a b)
      ⊢ Disjoint (nhds a) Filter.atBot
    -/
    refine disjoint_of_disjoint_of_mem disjoint_compl_left ?_ (Iic_mem_atBot b)
    /-
      case mpr.intro
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : Preorder α
      inst✝ : ClosedIicTopology α
      a b : α
      hb : Not (LE.le a b)
      ⊢ Membership.mem (nhds a) (HasCompl.compl (Set.Iic b))
    -/
    exact isClosed_Iic.isOpen_compl.mem_nhds hb
    /-
      🎉 no goals
    -/


theorem IsLUB.range_of_tendsto {F : Filter β} [F.NeBot] (hle : ∀ i, f i ≤ a)
    (hlim : Tendsto f F (𝓝 a)) : IsLUB (range f) a :=
  ⟨forall_mem_range.mpr hle, fun _c hc ↦ le_of_tendsto' hlim fun i ↦ hc <| mem_range_self i⟩


                                                                 /-
                                                                   α : Type u
                                                                   inst✝³ : Preorder α
                                                                   inst✝² : NoBotOrder α
                                                                   inst✝¹ : TopologicalSpace α
                                                                   inst✝ : ClosedIicTopology α
                                                                   a : α
                                                                   ⊢ Disjoint (nhds a) Filter.atBot
                                                                 -/
theorem disjoint_nhds_atBot (a : α) : Disjoint (𝓝 a) atBot := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem inf_nhds_atBot (a : α) : 𝓝 a ⊓ atBot = ⊥ := (disjoint_nhds_atBot a).eq_bot


theorem not_tendsto_nhds_of_tendsto_atBot (hf : Tendsto f l atBot) (a : α) : ¬Tendsto f l (𝓝 a) :=
  hf.not_tendsto (disjoint_nhds_atBot a).symm


theorem not_tendsto_atBot_of_tendsto_nhds (hf : Tendsto f l (𝓝 a)) : ¬Tendsto f l atBot :=
  hf.not_tendsto (disjoint_nhds_atBot a)


theorem iSup_eq_of_forall_le_of_tendsto {ι : Type*} {F : Filter ι} [Filter.NeBot F]
    [ConditionallyCompleteLattice α] [TopologicalSpace α] [ClosedIicTopology α]
    {a : α} {f : ι → α} (hle : ∀ i, f i ≤ a) (hlim : Filter.Tendsto f F (𝓝 a)) :
    ⨆ i, f i = a :=
  have := F.nonempty_of_neBot
  (IsLUB.range_of_tendsto hle hlim).ciSup_eq


theorem iUnion_Iic_eq_Iio_of_lt_of_tendsto {ι : Type*} {F : Filter ι} [F.NeBot]
    [ConditionallyCompleteLinearOrder α] [TopologicalSpace α] [ClosedIicTopology α]
    {a : α} {f : ι → α} (hlt : ∀ i, f i < a) (hlim : Tendsto f F (𝓝 a)) :
    ⋃ i : ι, Iic (f i) = Iio a := by
  have obs : a ∉ range f := by
    rw [mem_range]
    rintro ⟨i, rfl⟩
    exact (hlt i).false
  /-
    α : Type u
    ι : Type u_1
    F : Filter ι
    inst✝³ : F.NeBot
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : ClosedIicTopology α
    a : α
    f : ι → α
    hlt : ∀ (i : ι), LT.lt (f i) a
    hlim : Filter.Tendsto f F (nhds a)
    obs : Not (Membership.mem (Set.range f) a)
    ⊢ Eq (Set.iUnion fun i => Set.Iic (f i)) (Set.Iio a)
  -/
  rw [← biUnion_range, (IsLUB.range_of_tendsto (le_of_lt <| hlt ·) hlim).biUnion_Iic_eq_Iio obs]
  /-
    🎉 no goals
  -/


theorem isOpen_Ioi : IsOpen (Ioi a) := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : ClosedIicTopology α
    a : α
    ⊢ IsOpen (Set.Ioi a)
  -/
  rw [← compl_Iic]
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : ClosedIicTopology α
    a : α
    ⊢ IsOpen (HasCompl.compl (Set.Iic a))
  -/
  exact isClosed_Iic.isOpen_compl
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_Ioi : interior (Ioi a) = Ioi a :=
  isOpen_Ioi.interior_eq


theorem Ioi_mem_nhds (h : a < b) : Ioi a ∈ 𝓝 b := IsOpen.mem_nhds isOpen_Ioi h


theorem eventually_gt_nhds (hab : b < a) : ∀ᶠ x in 𝓝 a, b < x := Ioi_mem_nhds hab


theorem Ici_mem_nhds (h : a < b) : Ici a ∈ 𝓝 b :=
  mem_of_superset (Ioi_mem_nhds h) Ioi_subset_Ici_self


theorem eventually_ge_nhds (hab : b < a) : ∀ᶠ x in 𝓝 a, b ≤ x := Ici_mem_nhds hab


theorem Filter.Tendsto.eventually_const_lt {l : Filter γ} {f : γ → α} {u v : α} (hv : u < v)
    (h : Filter.Tendsto f l (𝓝 v)) : ∀ᶠ a in l, u < f a :=
  h.eventually <| eventually_gt_nhds hv


@[deprecated (since := "2024-11-17")]
alias eventually_gt_of_tendsto_gt := Filter.Tendsto.eventually_const_lt


theorem Filter.Tendsto.eventually_const_le {l : Filter γ} {f : γ → α} {u v : α} (hv : u < v)
    (h : Tendsto f l (𝓝 v)) : ∀ᶠ a in l, u ≤ f a :=
  h.eventually <| eventually_ge_nhds hv


@[deprecated (since := "2024-11-17")]
alias eventually_ge_of_tendsto_gt := Filter.Tendsto.eventually_const_le


protected theorem Dense.exists_gt [NoMaxOrder α] {s : Set α} (hs : Dense s) (x : α) :
    ∃ y ∈ s, x < y :=
  hs.exists_mem_open isOpen_Ioi (exists_gt x)


protected theorem Dense.exists_ge [NoMaxOrder α] {s : Set α} (hs : Dense s) (x : α) :
    ∃ y ∈ s, x ≤ y :=
  (hs.exists_gt x).imp fun _ h ↦ ⟨h.1, h.2.le⟩


theorem Dense.exists_ge' {s : Set α} (hs : Dense s) (htop : ∀ x, IsTop x → x ∈ s) (x : α) :
    ∃ y ∈ s, x ≤ y := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : ClosedIicTopology α
    s : Set α
    hs : Dense s
    htop : ∀ (x : α), IsTop x → Membership.mem s x
    x : α
    ⊢ Exists fun y => And (Membership.mem s y) (LE.le x y)
  -/
  by_cases hx : IsTop x
    /-
      case pos
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Dense s
      htop : ∀ (x : α), IsTop x → Membership.mem s x
      x : α
      hx : IsTop x
      ⊢ Exists fun y => And (Membership.mem s y) (LE.le x y)
    -/
  · exact ⟨x, htop x hx, le_rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Dense s
      htop : ∀ (x : α), IsTop x → Membership.mem s x
      x : α
      hx : Not (IsTop x)
      ⊢ Exists fun y => And (Membership.mem s y) (LE.le x y)
    -/
  · simp only [IsTop, not_forall, not_le] at hx
    /-
      case neg
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Dense s
      htop : ∀ (x : α), IsTop x → Membership.mem s x
      x : α
      hx : Exists fun x_1 => LT.lt x x_1
      ⊢ Exists fun y => And (Membership.mem s y) (LE.le x y)
    -/
    rcases hs.exists_mem_open isOpen_Ioi hx with ⟨y, hys, hy : x < y⟩
    /-
      case neg.intro.intro
      α : Type u
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : ClosedIicTopology α
      s : Set α
      hs : Dense s
      htop : ∀ (x : α), IsTop x → Membership.mem s x
      x : α
      hx : Exists fun x_1 => LT.lt x x_1
      y : α
      hys : Membership.mem s y
      hy : LT.lt x y
      ⊢ Exists fun y => And (Membership.mem s y) (LE.le x y)
    -/
    exact ⟨y, hys, hy.le⟩
    /-
      🎉 no goals
    -/


theorem Ioo_mem_nhdsLT (H : a < b) : Ioo a b ∈ 𝓝[<] b := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : ClosedIicTopology α
    a b : α
    H : LT.lt a b
    ⊢ Membership.mem (nhdsWithin b (Set.Iio b)) (Set.Ioo a b)
  -/
  simpa only [← Iio_inter_Ioi] using inter_mem_nhdsWithin _ (Ioi_mem_nhds H)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")] alias Ioo_mem_nhdsWithin_Iio' := Ioo_mem_nhdsLT


theorem Ioo_mem_nhdsLT_of_mem (H : b ∈ Ioc a c) : Ioo a c ∈ 𝓝[<] b :=
  mem_of_superset (Ioo_mem_nhdsLT H.1) <| Ioo_subset_Ioo_right H.2


@[deprecated (since := "2024-12-21")] alias Ioo_mem_nhdsWithin_Iio := Ioo_mem_nhdsLT_of_mem


protected theorem CovBy.nhdsLT (h : a ⋖ b) : 𝓝[<] b = ⊥ :=
  empty_mem_iff_bot.mp <| h.Ioo_eq ▸ Ioo_mem_nhdsLT h.1


@[deprecated (since := "2024-12-21")] protected alias CovBy.nhdsWithin_Iio := CovBy.nhdsLT


protected theorem PredOrder.nhdsLT [PredOrder α] : 𝓝[<] a = ⊥ := by
  if h : IsMin a then simp [h.Iio_eq]
  else exact (Order.pred_covBy_of_not_isMin h).nhdsLT


@[deprecated (since := "2024-12-21")] protected alias PredOrder.nhdsWithin_Iio := PredOrder.nhdsLT


theorem Ico_mem_nhdsLT_of_mem (H : b ∈ Ioc a c) : Ico a c ∈ 𝓝[<] b :=
  mem_of_superset (Ioo_mem_nhdsLT_of_mem H) Ioo_subset_Ico_self


@[deprecated (since := "2024-12-21")] alias Ico_mem_nhdsWithin_Iio := Ico_mem_nhdsLT_of_mem


theorem Ico_mem_nhdsLT (H : a < b) : Ico a b ∈ 𝓝[<] b := Ico_mem_nhdsLT_of_mem ⟨H, le_rfl⟩


@[deprecated (since := "2024-12-21")] alias Ico_mem_nhdsWithin_Iio' := Ico_mem_nhdsLT


theorem Ioc_mem_nhdsLT_of_mem (H : b ∈ Ioc a c) : Ioc a c ∈ 𝓝[<] b :=
  mem_of_superset (Ioo_mem_nhdsLT_of_mem H) Ioo_subset_Ioc_self


@[deprecated (since := "2024-12-21")] alias Ioc_mem_nhdsWithin_Iio := Ioc_mem_nhdsLT_of_mem


theorem Ioc_mem_nhdsLT (H : a < b) : Ioc a b ∈ 𝓝[<] b := Ioc_mem_nhdsLT_of_mem ⟨H, le_rfl⟩


@[deprecated (since := "2024-12-21")] alias Ioc_mem_nhdsWithin_Iio' := Ioc_mem_nhdsLT


theorem Icc_mem_nhdsLT_of_mem (H : b ∈ Ioc a c) : Icc a c ∈ 𝓝[<] b :=
  mem_of_superset (Ioo_mem_nhdsLT_of_mem H) Ioo_subset_Icc_self


@[deprecated (since := "2024-12-21")] alias Icc_mem_nhdsWithin_Iio := Icc_mem_nhdsLT_of_mem


theorem Icc_mem_nhdsLT (H : a < b) : Icc a b ∈ 𝓝[<] b := Icc_mem_nhdsLT_of_mem ⟨H, le_rfl⟩


@[deprecated (since := "2024-12-21")] alias Icc_mem_nhdsWithin_Iio' := Icc_mem_nhdsLT


@[simp]
theorem nhdsWithin_Ico_eq_nhdsLT (h : a < b) : 𝓝[Ico a b] b = 𝓝[<] b :=
  nhdsWithin_inter_of_mem <| nhdsWithin_le_nhds <| Ici_mem_nhds h


@[deprecated (since := "2024-12-21")]
alias nhdsWithin_Ico_eq_nhdsWithin_Iio := nhdsWithin_Ico_eq_nhdsLT


@[simp]
theorem nhdsWithin_Ioo_eq_nhdsLT (h : a < b) : 𝓝[Ioo a b] b = 𝓝[<] b :=
  nhdsWithin_inter_of_mem <| nhdsWithin_le_nhds <| Ioi_mem_nhds h


@[deprecated (since := "2024-12-21")]
alias nhdsWithin_Ioo_eq_nhdsWithin_Iio := nhdsWithin_Ioo_eq_nhdsLT


@[simp]
theorem continuousWithinAt_Ico_iff_Iio (h : a < b) :
    ContinuousWithinAt f (Ico a b) b ↔ ContinuousWithinAt f (Iio b) b := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIicTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Ico a b) b) (ContinuousWithinAt f (Set.Iio b) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Ico_eq_nhdsLT h]
  /-
    🎉 no goals
  -/


@[simp]
theorem continuousWithinAt_Ioo_iff_Iio (h : a < b) :
    ContinuousWithinAt f (Ioo a b) b ↔ ContinuousWithinAt f (Iio b) b := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIicTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Ioo a b) b) (ContinuousWithinAt f (Set.Iio b) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Ioo_eq_nhdsLT h]
  /-
    🎉 no goals
  -/


protected theorem CovBy.nhdsLE (H : a ⋖ b) : 𝓝[≤] b = pure b := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : ClosedIicTopology α
    a b : α
    H : CovBy a b
    ⊢ Eq (nhdsWithin b (Set.Iic b)) (Pure.pure b)
  -/
  rw [← Iio_insert, nhdsWithin_insert, H.nhdsLT, sup_bot_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")]
protected alias CovBy.nhdsWithin_Iic := CovBy.nhdsLE


protected theorem PredOrder.nhdsLE [PredOrder α] : 𝓝[≤] b = pure b := by
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIicTopology α
    b : α
    inst✝ : PredOrder α
    ⊢ Eq (nhdsWithin b (Set.Iic b)) (Pure.pure b)
  -/
  rw [← Iio_insert, nhdsWithin_insert, PredOrder.nhdsLT, sup_bot_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-21")]
protected alias PredOrder.nhdsWithin_Iic := PredOrder.nhdsLE


theorem Ioc_mem_nhdsLE (H : a < b) : Ioc a b ∈ 𝓝[≤] b :=
  inter_mem (nhdsWithin_le_nhds <| Ioi_mem_nhds H) self_mem_nhdsWithin


@[deprecated (since := "2024-12-21")] alias Ioc_mem_nhdsWithin_Iic' := Ioc_mem_nhdsLE


theorem Ioo_mem_nhdsLE_of_mem (H : b ∈ Ioo a c) : Ioo a c ∈ 𝓝[≤] b :=
  mem_of_superset (Ioc_mem_nhdsLE H.1) <| Ioc_subset_Ioo_right H.2


@[deprecated (since := "2024-12-21")] alias Ioo_mem_nhdsWithin_Iic := Ioo_mem_nhdsLE_of_mem


theorem Ico_mem_nhdsLE_of_mem (H : b ∈ Ioo a c) : Ico a c ∈ 𝓝[≤] b :=
  mem_of_superset (Ioo_mem_nhdsLE_of_mem H) Ioo_subset_Ico_self


@[deprecated (since := "2024-12-22")]
alias Ico_mem_nhdsWithin_Iic := Ico_mem_nhdsLE_of_mem


theorem Ioc_mem_nhdsLE_of_mem (H : b ∈ Ioc a c) : Ioc a c ∈ 𝓝[≤] b :=
  mem_of_superset (Ioc_mem_nhdsLE H.1) <| Ioc_subset_Ioc_right H.2


@[deprecated (since := "2024-12-22")]
alias Ioc_mem_nhdsWithin_Iic := Ioc_mem_nhdsLE_of_mem


theorem Icc_mem_nhdsLE_of_mem (H : b ∈ Ioc a c) : Icc a c ∈ 𝓝[≤] b :=
  mem_of_superset (Ioc_mem_nhdsLE_of_mem H) Ioc_subset_Icc_self


@[deprecated (since := "2024-12-22")]
alias Icc_mem_nhdsWithin_Iic := Icc_mem_nhdsLE_of_mem


theorem Icc_mem_nhdsLE (H : a < b) : Icc a b ∈ 𝓝[≤] b := Icc_mem_nhdsLE_of_mem ⟨H, le_rfl⟩


@[deprecated (since := "2024-12-22")]
alias Icc_mem_nhdsWithin_Iic' := Icc_mem_nhdsLE


@[simp]
theorem nhdsWithin_Icc_eq_nhdsLE (h : a < b) : 𝓝[Icc a b] b = 𝓝[≤] b :=
  nhdsWithin_inter_of_mem <| nhdsWithin_le_nhds <| Ici_mem_nhds h


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Icc_eq_nhdsWithin_Iic := nhdsWithin_Icc_eq_nhdsLE


@[simp]
theorem nhdsWithin_Ioc_eq_nhdsLE (h : a < b) : 𝓝[Ioc a b] b = 𝓝[≤] b :=
  nhdsWithin_inter_of_mem <| nhdsWithin_le_nhds <| Ioi_mem_nhds h


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioc_eq_nhdsWithin_Iic := nhdsWithin_Ioc_eq_nhdsLE


@[simp]
theorem continuousWithinAt_Icc_iff_Iic (h : a < b) :
    ContinuousWithinAt f (Icc a b) b ↔ ContinuousWithinAt f (Iic b) b := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIicTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Icc a b) b) (ContinuousWithinAt f (Set.Iic b) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Icc_eq_nhdsLE h]
  /-
    🎉 no goals
  -/


@[simp]
theorem continuousWithinAt_Ioc_iff_Iic (h : a < b) :
    ContinuousWithinAt f (Ioc a b) b ↔ ContinuousWithinAt f (Iic b) b := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIicTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Ioc a b) b) (ContinuousWithinAt f (Set.Iic b) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Ioc_eq_nhdsLE h]
  /-
    🎉 no goals
  -/


theorem isClosed_Ici {a : α} : IsClosed (Ici a) :=
  ClosedIciTopology.isClosed_Ici a


@[deprecated isClosed_Ici (since := "2024-02-15")]
lemma ClosedIciTopology.isClosed_ge' (a : α) : IsClosed {x | a ≤ x} := isClosed_Ici a

instance : ClosedIicTopology αᵒᵈ where
  isClosed_Iic _ := isClosed_Ici (α := α)


@[simp]
theorem closure_Ici (a : α) : closure (Ici a) = Ici a :=
  isClosed_Ici.closure_eq


lemma ge_of_tendsto_of_frequently {x : Filter β} (lim : Tendsto f x (𝓝 a))
    (h : ∃ᶠ c in x, b ≤ f c) : b ≤ a :=
  isClosed_Ici.mem_of_frequently_of_tendsto h lim


theorem ge_of_tendsto {x : Filter β} [NeBot x] (lim : Tendsto f x (𝓝 a))
    (h : ∀ᶠ c in x, b ≤ f c) : b ≤ a :=
  isClosed_Ici.mem_of_tendsto lim h


theorem ge_of_tendsto' {x : Filter β} [NeBot x] (lim : Tendsto f x (𝓝 a))
    (h : ∀ c, b ≤ f c) : b ≤ a :=
  ge_of_tendsto lim (Eventually.of_forall h)


@[simp] lemma lowerBounds_closure (s : Set α) : lowerBounds (closure s : Set α) = lowerBounds s :=
                 /-
                   α : Type u
                   inst✝² : TopologicalSpace α
                   inst✝¹ : Preorder α
                   inst✝ : ClosedIciTopology α
                   s : Set α
                   a : α
                   ⊢ Iff (Membership.mem (lowerBounds (closure s)) a) (Membership.mem (lowerBound …
                 -/
  ext fun a ↦ by simp_rw [mem_lowerBounds_iff_subset_Ici, isClosed_Ici.closure_subset_iff]
                 /-
                   🎉 no goals
                 -/


@[simp] lemma bddBelow_closure : BddBelow (closure s) ↔ BddBelow s := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Preorder α
    inst✝ : ClosedIciTopology α
    s : Set α
    ⊢ Iff (BddBelow (closure s)) (BddBelow s)
  -/
  simp_rw [BddBelow, lowerBounds_closure]
  /-
    🎉 no goals
  -/


protected alias ⟨_, BddBelow.closure⟩ := bddBelow_closure


@[simp]
theorem disjoint_nhds_atTop_iff : Disjoint (𝓝 a) atTop ↔ ¬IsTop a :=
  disjoint_nhds_atBot_iff (α := αᵒᵈ)


theorem IsGLB.range_of_tendsto {F : Filter β} [F.NeBot] (hle : ∀ i, a ≤ f i)
    (hlim : Tendsto f F (𝓝 a)) : IsGLB (range f) a :=
  IsLUB.range_of_tendsto (α := αᵒᵈ) hle hlim


theorem disjoint_nhds_atTop (a : α) : Disjoint (𝓝 a) atTop := disjoint_nhds_atBot (toDual a)


@[simp]
theorem inf_nhds_atTop (a : α) : 𝓝 a ⊓ atTop = ⊥ := (disjoint_nhds_atTop a).eq_bot


theorem not_tendsto_nhds_of_tendsto_atTop (hf : Tendsto f l atTop) (a : α) : ¬Tendsto f l (𝓝 a) :=
  hf.not_tendsto (disjoint_nhds_atTop a).symm


theorem not_tendsto_atTop_of_tendsto_nhds (hf : Tendsto f l (𝓝 a)) : ¬Tendsto f l atTop :=
  hf.not_tendsto (disjoint_nhds_atTop a)


theorem iInf_eq_of_forall_le_of_tendsto {ι : Type*} {F : Filter ι} [F.NeBot]
    [ConditionallyCompleteLattice α] [TopologicalSpace α] [ClosedIciTopology α]
    {a : α} {f : ι → α} (hle : ∀ i, a ≤ f i) (hlim : Tendsto f F (𝓝 a)) :
    ⨅ i, f i = a :=
  iSup_eq_of_forall_le_of_tendsto (α := αᵒᵈ) hle hlim


theorem iUnion_Ici_eq_Ioi_of_lt_of_tendsto {ι : Type*} {F : Filter ι} [F.NeBot]
    [ConditionallyCompleteLinearOrder α] [TopologicalSpace α] [ClosedIciTopology α]
    {a : α} {f : ι → α} (hlt : ∀ i, a < f i) (hlim : Tendsto f F (𝓝 a)) :
    ⋃ i : ι, Ici (f i) = Ioi a :=
  iUnion_Iic_eq_Iio_of_lt_of_tendsto (α := αᵒᵈ) hlt hlim


theorem isOpen_Iio : IsOpen (Iio a) := isOpen_Ioi (α := αᵒᵈ)


@[simp] theorem interior_Iio : interior (Iio a) = Iio a := isOpen_Iio.interior_eq


theorem Iio_mem_nhds (h : a < b) : Iio b ∈ 𝓝 a := isOpen_Iio.mem_nhds h


theorem eventually_lt_nhds (hab : a < b) : ∀ᶠ x in 𝓝 a, x < b := Iio_mem_nhds hab


theorem Iic_mem_nhds (h : a < b) : Iic b ∈ 𝓝 a :=
  mem_of_superset (Iio_mem_nhds h) Iio_subset_Iic_self


theorem eventually_le_nhds (hab : a < b) : ∀ᶠ x in 𝓝 a, x ≤ b := Iic_mem_nhds hab


theorem Filter.Tendsto.eventually_lt_const {l : Filter γ} {f : γ → α} {u v : α} (hv : v < u)
    (h : Filter.Tendsto f l (𝓝 v)) : ∀ᶠ a in l, f a < u :=
  h.eventually <| eventually_lt_nhds hv


@[deprecated (since := "2024-11-17")]
alias eventually_lt_of_tendsto_lt := Filter.Tendsto.eventually_lt_const


theorem Filter.Tendsto.eventually_le_const {l : Filter γ} {f : γ → α} {u v : α} (hv : v < u)
    (h : Tendsto f l (𝓝 v)) : ∀ᶠ a in l, f a ≤ u :=
  h.eventually <| eventually_le_nhds hv


@[deprecated (since := "2024-11-17")]
alias eventually_le_of_tendsto_lt := Filter.Tendsto.eventually_le_const


protected theorem Dense.exists_lt [NoMinOrder α] {s : Set α} (hs : Dense s) (x : α) :
    ∃ y ∈ s, y < x :=
  hs.orderDual.exists_gt x


protected theorem Dense.exists_le [NoMinOrder α] {s : Set α} (hs : Dense s) (x : α) :
    ∃ y ∈ s, y ≤ x :=
  hs.orderDual.exists_ge x


theorem Dense.exists_le' {s : Set α} (hs : Dense s) (hbot : ∀ x, IsBot x → x ∈ s) (x : α) :
    ∃ y ∈ s, y ≤ x :=
  hs.orderDual.exists_ge' hbot x


theorem Ioo_mem_nhdsGT_of_mem (H : b ∈ Ico a c) : Ioo a c ∈ 𝓝[>] b :=
  mem_nhdsWithin.2
                                /-
                                  α : Type u
                                  inst✝² : TopologicalSpace α
                                  inst✝¹ : LinearOrder α
                                  inst✝ : ClosedIciTopology α
                                  a b c : α
                                  H : Membership.mem (Set.Ico a c) b
                                  ⊢ HasSubset.Subset (Inter.inter (Set.Iio c) (Set.Ioi b)) (Set.Ioo a c)
                                -/
    ⟨Iio c, isOpen_Iio, H.2, by rw [inter_comm, Ioi_inter_Iio]; exact Ioo_subset_Ioo_left H.1⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: swap `'`?


@[deprecated (since := "2024-12-22")] alias Ioo_mem_nhdsWithin_Ioi := Ioo_mem_nhdsGT_of_mem


theorem Ioo_mem_nhdsGT (H : a < b) : Ioo a b ∈ 𝓝[>] a := Ioo_mem_nhdsGT_of_mem ⟨le_rfl, H⟩


@[deprecated (since := "2024-12-22")] alias Ioo_mem_nhdsWithin_Ioi' := Ioo_mem_nhdsGT


protected theorem CovBy.nhdsGT (h : a ⋖ b) : 𝓝[>] a = ⊥ := h.toDual.nhdsLT


@[deprecated (since := "2024-12-22")] alias CovBy.nhdsWithin_Ioi := CovBy.nhdsGT


protected theorem SuccOrder.nhdsGT [SuccOrder α] : 𝓝[>] a = ⊥ := PredOrder.nhdsLT (α := αᵒᵈ)


@[deprecated (since := "2024-12-22")] alias SuccOrder.nhdsWithin_Ioi := SuccOrder.nhdsGT


theorem Ioc_mem_nhdsGT_of_mem (H : b ∈ Ico a c) : Ioc a c ∈ 𝓝[>] b :=
  mem_of_superset (Ioo_mem_nhdsGT_of_mem H) Ioo_subset_Ioc_self


@[deprecated (since := "2024-12-22")]
alias Ioc_mem_nhdsWithin_Ioi := Ioc_mem_nhdsGT_of_mem


theorem Ioc_mem_nhdsGT (H : a < b) : Ioc a b ∈ 𝓝[>] a := Ioc_mem_nhdsGT_of_mem ⟨le_rfl, H⟩


@[deprecated (since := "2024-12-22")] alias Ioc_mem_nhdsWithin_Ioi' := Ioc_mem_nhdsGT


theorem Ico_mem_nhdsGT_of_mem (H : b ∈ Ico a c) : Ico a c ∈ 𝓝[>] b :=
  mem_of_superset (Ioo_mem_nhdsGT_of_mem H) Ioo_subset_Ico_self


@[deprecated (since := "2024-12-22")] alias Ico_mem_nhdsWithin_Ioi := Ico_mem_nhdsGT_of_mem


theorem Ico_mem_nhdsGT (H : a < b) : Ico a b ∈ 𝓝[>] a := Ico_mem_nhdsGT_of_mem ⟨le_rfl, H⟩


@[deprecated (since := "2024-12-22")] alias Ico_mem_nhdsWithin_Ioi' := Ico_mem_nhdsGT


theorem Icc_mem_nhdsGT_of_mem (H : b ∈ Ico a c) : Icc a c ∈ 𝓝[>] b :=
  mem_of_superset (Ioo_mem_nhdsGT_of_mem H) Ioo_subset_Icc_self


@[deprecated (since := "2024-12-22")] alias Icc_mem_nhdsWithin_Ioi := Icc_mem_nhdsGT_of_mem


theorem Icc_mem_nhdsGT (H : a < b) : Icc a b ∈ 𝓝[>] a := Icc_mem_nhdsGT_of_mem ⟨le_rfl, H⟩


@[deprecated (since := "2024-12-22")] alias Icc_mem_nhdsWithin_Ioi' := Icc_mem_nhdsGT


@[simp]
theorem nhdsWithin_Ioc_eq_nhdsGT (h : a < b) : 𝓝[Ioc a b] a = 𝓝[>] a :=
  nhdsWithin_inter_of_mem' <| nhdsWithin_le_nhds <| Iic_mem_nhds h


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioc_eq_nhdsWithin_Ioi := nhdsWithin_Ioc_eq_nhdsGT


@[simp]
theorem nhdsWithin_Ioo_eq_nhdsGT (h : a < b) : 𝓝[Ioo a b] a = 𝓝[>] a :=
  nhdsWithin_inter_of_mem' <| nhdsWithin_le_nhds <| Iio_mem_nhds h


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ioo_eq_nhdsWithin_Ioi := nhdsWithin_Ioo_eq_nhdsGT


@[simp]
theorem continuousWithinAt_Ioc_iff_Ioi (h : a < b) :
    ContinuousWithinAt f (Ioc a b) a ↔ ContinuousWithinAt f (Ioi a) a := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIciTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Ioc a b) a) (ContinuousWithinAt f (Set.Ioi a) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Ioc_eq_nhdsGT h]
  /-
    🎉 no goals
  -/


@[simp]
theorem continuousWithinAt_Ioo_iff_Ioi (h : a < b) :
    ContinuousWithinAt f (Ioo a b) a ↔ ContinuousWithinAt f (Ioi a) a := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIciTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Ioo a b) a) (ContinuousWithinAt f (Set.Ioi a) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Ioo_eq_nhdsGT h]
  /-
    🎉 no goals
  -/


protected theorem CovBy.nhdsGE (H : a ⋖ b) : 𝓝[≥] a = pure a := H.toDual.nhdsLE


@[deprecated (since := "2024-12-22")] alias CovBy.nhdsWithin_Ici := CovBy.nhdsGE


protected theorem SuccOrder.nhdsGE [SuccOrder α] : 𝓝[≥] a = pure a :=
  PredOrder.nhdsLE (α := αᵒᵈ)


@[deprecated (since := "2024-12-22")]
alias SuccOrder.nhdsWithin_Ici := SuccOrder.nhdsGE


theorem Ico_mem_nhdsGE (H : a < b) : Ico a b ∈ 𝓝[≥] a :=
  inter_mem_nhdsWithin _ <| Iio_mem_nhds H


@[deprecated (since := "2024-12-22")] alias Ico_mem_nhdsWithin_Ici' := Ico_mem_nhdsGE


theorem Ico_mem_nhdsGE_of_mem (H : b ∈ Ico a c) : Ico a c ∈ 𝓝[≥] b :=
  mem_of_superset (Ico_mem_nhdsGE H.2) <| Ico_subset_Ico_left H.1


@[deprecated (since := "2024-12-22")]
alias Ico_mem_nhdsWithin_Ici := Ico_mem_nhdsGE_of_mem


theorem Ioo_mem_nhdsGE_of_mem (H : b ∈ Ioo a c) : Ioo a c ∈ 𝓝[≥] b :=
  mem_of_superset (Ico_mem_nhdsGE H.2) <| Ico_subset_Ioo_left H.1


@[deprecated (since := "2024-12-22")]
alias Ioo_mem_nhdsWithin_Ici := Ioo_mem_nhdsGE_of_mem


theorem Ioc_mem_nhdsGE_of_mem (H : b ∈ Ioo a c) : Ioc a c ∈ 𝓝[≥] b :=
  mem_of_superset (Ioo_mem_nhdsGE_of_mem H) Ioo_subset_Ioc_self


@[deprecated (since := "2024-12-22")] alias Ioc_mem_nhdsWithin_Ici := Ioc_mem_nhdsGE_of_mem


theorem Icc_mem_nhdsGE_of_mem (H : b ∈ Ico a c) : Icc a c ∈ 𝓝[≥] b :=
  mem_of_superset (Ico_mem_nhdsGE_of_mem H) Ico_subset_Icc_self


@[deprecated (since := "2024-12-22")]
alias Icc_mem_nhdsWithin_Ici := Icc_mem_nhdsGE_of_mem


theorem Icc_mem_nhdsGE (H : a < b) : Icc a b ∈ 𝓝[≥] a := Icc_mem_nhdsGE_of_mem ⟨le_rfl, H⟩


@[deprecated (since := "2024-12-22")] alias Icc_mem_nhdsWithin_Ici' := Icc_mem_nhdsGE


@[simp]
theorem nhdsWithin_Icc_eq_nhdsGE (h : a < b) : 𝓝[Icc a b] a = 𝓝[≥] a :=
  nhdsWithin_inter_of_mem' <| nhdsWithin_le_nhds <| Iic_mem_nhds h


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Icc_eq_nhdsWithin_Ici := nhdsWithin_Icc_eq_nhdsGE


@[simp]
theorem nhdsWithin_Ico_eq_nhdsGE (h : a < b) : 𝓝[Ico a b] a = 𝓝[≥] a :=
  nhdsWithin_inter_of_mem' <| nhdsWithin_le_nhds <| Iio_mem_nhds h


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_Ico_eq_nhdsWithin_Ici := nhdsWithin_Ico_eq_nhdsGE


@[simp]
theorem continuousWithinAt_Icc_iff_Ici (h : a < b) :
    ContinuousWithinAt f (Icc a b) a ↔ ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIciTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Icc a b) a) (ContinuousWithinAt f (Set.Ici a) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Icc_eq_nhdsGE h]
  /-
    🎉 no goals
  -/


@[simp]
theorem continuousWithinAt_Ico_iff_Ici (h : a < b) :
    ContinuousWithinAt f (Ico a b) a ↔ ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : ClosedIciTopology α
    inst✝ : TopologicalSpace β
    a b : α
    f : α → β
    h : LT.lt a b
    ⊢ Iff (ContinuousWithinAt f (Set.Ico a b) a) (ContinuousWithinAt f (Set.Ici a) …
  -/
  simp only [ContinuousWithinAt, nhdsWithin_Ico_eq_nhdsGE h]
  /-
    🎉 no goals
  -/


instance {p : α → Prop} : OrderClosedTopology (Subtype p) :=
  have this : Continuous fun p : Subtype p × Subtype p => ((p.fst : α), (p.snd : α)) :=
    continuous_subtype_val.prodMap continuous_subtype_val
  OrderClosedTopology.mk (t.isClosed_le'.preimage this)


theorem isClosed_le_prod : IsClosed { p : α × α | p.1 ≤ p.2 } :=
  t.isClosed_le'


theorem isClosed_le [TopologicalSpace β] {f g : β → α} (hf : Continuous f) (hg : Continuous g) :
    IsClosed { b | f b ≤ g b } :=
  continuous_iff_isClosed.mp (hf.prod_mk hg) _ isClosed_le_prod


instance : ClosedIicTopology α where
  isClosed_Iic _ := isClosed_le continuous_id continuous_const


instance : ClosedIciTopology α where
  isClosed_Ici _ := isClosed_le continuous_const continuous_id


instance : OrderClosedTopology αᵒᵈ :=
  ⟨(OrderClosedTopology.isClosed_le' (α := α)).preimage continuous_swap⟩


theorem isClosed_Icc {a b : α} : IsClosed (Icc a b) :=
  IsClosed.inter isClosed_Ici isClosed_Iic


@[simp]
theorem closure_Icc (a b : α) : closure (Icc a b) = Icc a b :=
  isClosed_Icc.closure_eq


theorem le_of_tendsto_of_tendsto {f g : β → α} {b : Filter β} {a₁ a₂ : α} [NeBot b]
    (hf : Tendsto f b (𝓝 a₁)) (hg : Tendsto g b (𝓝 a₂)) (h : f ≤ᶠ[b] g) : a₁ ≤ a₂ :=
  have : Tendsto (fun b => (f b, g b)) b (𝓝 (a₁, a₂)) := hf.prod_mk_nhds hg
  show (a₁, a₂) ∈ { p : α × α | p.1 ≤ p.2 } from t.isClosed_le'.mem_of_tendsto this h


alias tendsto_le_of_eventuallyLE := le_of_tendsto_of_tendsto


theorem le_of_tendsto_of_tendsto' {f g : β → α} {b : Filter β} {a₁ a₂ : α} [NeBot b]
    (hf : Tendsto f b (𝓝 a₁)) (hg : Tendsto g b (𝓝 a₂)) (h : ∀ x, f x ≤ g x) : a₁ ≤ a₂ :=
  le_of_tendsto_of_tendsto hf hg (Eventually.of_forall h)


@[simp]
theorem closure_le_eq [TopologicalSpace β] {f g : β → α} (hf : Continuous f) (hg : Continuous g) :
    closure { b | f b ≤ g b } = { b | f b ≤ g b } :=
  (isClosed_le hf hg).closure_eq


theorem closure_lt_subset_le [TopologicalSpace β] {f g : β → α} (hf : Continuous f)
    (hg : Continuous g) : closure { b | f b < g b } ⊆ { b | f b ≤ g b } :=
  (closure_minimal fun _ => le_of_lt) <| isClosed_le hf hg


theorem ContinuousWithinAt.closure_le [TopologicalSpace β] {f g : β → α} {s : Set β} {x : β}
    (hx : x ∈ closure s) (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x)
    (h : ∀ y ∈ s, f y ≤ g y) : f x ≤ g x :=
  show (f x, g x) ∈ { p : α × α | p.1 ≤ p.2 } from
    OrderClosedTopology.isClosed_le'.closure_subset ((hf.prod hg).mem_closure hx h)


/-- If `s` is a closed set and two functions `f` and `g` are continuous on `s`,
then the set `{x ∈ s | f x ≤ g x}` is a closed set. -/
theorem IsClosed.isClosed_le [TopologicalSpace β] {f g : β → α} {s : Set β} (hs : IsClosed s)
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) : IsClosed ({ x ∈ s | f x ≤ g x }) :=
  (hf.prod hg).preimage_isClosed_of_isClosed hs OrderClosedTopology.isClosed_le'


theorem le_on_closure [TopologicalSpace β] {f g : β → α} {s : Set β} (h : ∀ x ∈ s, f x ≤ g x)
    (hf : ContinuousOn f (closure s)) (hg : ContinuousOn g (closure s)) ⦃x⦄ (hx : x ∈ closure s) :
    f x ≤ g x :=
  have : s ⊆ { y ∈ closure s | f y ≤ g y } := fun y hy => ⟨subset_closure hy, h y hy⟩
  (closure_minimal this (isClosed_closure.isClosed_le hf hg) hx).2


theorem IsClosed.epigraph [TopologicalSpace β] {f : β → α} {s : Set β} (hs : IsClosed s)
    (hf : ContinuousOn f s) : IsClosed { p : β × α | p.1 ∈ s ∧ f p.1 ≤ p.2 } :=
  (hs.preimage continuous_fst).isClosed_le (hf.comp continuousOn_fst Subset.rfl) continuousOn_snd


theorem IsClosed.hypograph [TopologicalSpace β] {f : β → α} {s : Set β} (hs : IsClosed s)
    (hf : ContinuousOn f s) : IsClosed { p : β × α | p.1 ∈ s ∧ p.2 ≤ f p.1 } :=
  (hs.preimage continuous_fst).isClosed_le continuousOn_snd (hf.comp continuousOn_fst Subset.rfl)


instance (priority := 90) OrderClosedTopology.to_t2Space : T2Space α :=
  t2_iff_isClosed_diagonal.2 <| by
    simpa only [diagonal, le_antisymm_iff] using
      t.isClosed_le'.inter (isClosed_le continuous_snd continuous_fst)


theorem isOpen_lt [TopologicalSpace β] {f g : β → α} (hf : Continuous f) (hg : Continuous g) :
    IsOpen { b | f b < g b } := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    inst✝ : TopologicalSpace β
    f g : β → α
    hf : Continuous f
    hg : Continuous g
    ⊢ IsOpen (setOf fun b => LT.lt (f b) (g b))
  -/
  simpa only [lt_iff_not_le] using (isClosed_le hg hf).isOpen_compl
  /-
    🎉 no goals
  -/


theorem isOpen_lt_prod : IsOpen { p : α × α | p.1 < p.2 } :=
  isOpen_lt continuous_fst continuous_snd


theorem isOpen_Ioo : IsOpen (Ioo a b) :=
  IsOpen.inter isOpen_Ioi isOpen_Iio


@[simp]
theorem interior_Ioo : interior (Ioo a b) = Ioo a b :=
  isOpen_Ioo.interior_eq


theorem Ioo_subset_closure_interior : Ioo a b ⊆ closure (interior (Ioo a b)) := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    a b : α
    ⊢ HasSubset.Subset (Set.Ioo a b) (closure (interior (Set.Ioo a b)))
  -/
  simp only [interior_Ioo, subset_closure]
  /-
    🎉 no goals
  -/


theorem Ioo_mem_nhds {a b x : α} (ha : a < x) (hb : x < b) : Ioo a b ∈ 𝓝 x :=
  IsOpen.mem_nhds isOpen_Ioo ⟨ha, hb⟩


theorem Ioc_mem_nhds {a b x : α} (ha : a < x) (hb : x < b) : Ioc a b ∈ 𝓝 x :=
  mem_of_superset (Ioo_mem_nhds ha hb) Ioo_subset_Ioc_self


theorem Ico_mem_nhds {a b x : α} (ha : a < x) (hb : x < b) : Ico a b ∈ 𝓝 x :=
  mem_of_superset (Ioo_mem_nhds ha hb) Ioo_subset_Ico_self


theorem Icc_mem_nhds {a b x : α} (ha : a < x) (hb : x < b) : Icc a b ∈ 𝓝 x :=
  mem_of_superset (Ioo_mem_nhds ha hb) Ioo_subset_Icc_self


/-- The only order closed topology on a linear order which is a `PredOrder` and a `SuccOrder`
is the discrete topology.

This theorem is not an instance,
because it causes searches for `PredOrder` and `SuccOrder` with their `Preorder` arguments
and very rarely matches. -/
theorem DiscreteTopology.of_predOrder_succOrder [PredOrder α] [SuccOrder α] :
    DiscreteTopology α := by
  /-
    α : Type u
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderClosedTopology α
    inst✝¹ : PredOrder α
    inst✝ : SuccOrder α
    ⊢ DiscreteTopology α
  -/
  refine discreteTopology_iff_nhds.mpr fun a ↦ ?_
  rw [← nhdsWithin_univ, ← Iic_union_Ioi, nhdsWithin_union, PredOrder.nhdsLE, SuccOrder.nhdsGT,
    sup_bot_eq]


theorem lt_subset_interior_le (hf : Continuous f) (hg : Continuous g) :
    { b | f b < g b } ⊆ interior { b | f b ≤ g b } :=
  (interior_maximal fun _ => le_of_lt) <| isOpen_lt hf hg


theorem frontier_le_subset_eq (hf : Continuous f) (hg : Continuous g) :
    frontier { b | f b ≤ g b } ⊆ { b | f b = g b } := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    ⊢ HasSubset.Subset (frontier (setOf fun b => LE.le (f b) (g b))) (setOf fun b  …
  -/
  rw [frontier_eq_closure_inter_closure, closure_le_eq hf hg]
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    ⊢ HasSubset.Subset (Inter.inter (setOf fun b => LE.le (f b) (g b)) (closure (H …
  -/
  rintro b ⟨hb₁, hb₂⟩
  /-
    case intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    b : β
    hb₁ : Membership.mem (setOf fun b => LE.le (f b) (g b)) b
    hb₂ : Membership.mem (closure (HasCompl.compl (setOf fun b => LE.le (f b) (g b …
    ⊢ Membership.mem (setOf fun b => Eq (f b) (g b)) b
  -/
  refine le_antisymm hb₁ (closure_lt_subset_le hg hf ?_)
  /-
    case intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    b : β
    hb₁ : Membership.mem (setOf fun b => LE.le (f b) (g b)) b
    hb₂ : Membership.mem (closure (HasCompl.compl (setOf fun b => LE.le (f b) (g b …
    ⊢ Membership.mem (closure (setOf fun b => LT.lt (g b) (f b))) b
  -/
  convert hb₂ using 2; simp only [not_le.symm]; rfl
                                                /-
                                                  🎉 no goals
                                                -/


theorem frontier_Iic_subset (a : α) : frontier (Iic a) ⊆ {a} :=
  frontier_le_subset_eq (@continuous_id α _) continuous_const


theorem frontier_Ici_subset (a : α) : frontier (Ici a) ⊆ {a} :=
  frontier_Iic_subset (α := αᵒᵈ) _


theorem frontier_lt_subset_eq (hf : Continuous f) (hg : Continuous g) :
    frontier { b | f b < g b } ⊆ { b | f b = g b } := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    ⊢ HasSubset.Subset (frontier (setOf fun b => LT.lt (f b) (g b))) (setOf fun b  …
  -/
  simpa only [← not_lt, ← compl_setOf, frontier_compl, eq_comm] using frontier_le_subset_eq hg hf
  /-
    🎉 no goals
  -/


theorem continuous_if_le [TopologicalSpace γ] [∀ x, Decidable (f x ≤ g x)] {f' g' : β → γ}
    (hf : Continuous f) (hg : Continuous g) (hf' : ContinuousOn f' { x | f x ≤ g x })
    (hg' : ContinuousOn g' { x | g x ≤ f x }) (hfg : ∀ x, f x = g x → f' x = g' x) :
    Continuous fun x => if f x ≤ g x then f' x else g' x := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderClosedTopology α
    f g : β → α
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : (x : β) → Decidable (LE.le (f x) (g x))
    f' g' : β → γ
    hf : Continuous f
    hg : Continuous g
    hf' : ContinuousOn f' (setOf fun x => LE.le (f x) (g x))
    hg' : ContinuousOn g' (setOf fun x => LE.le (g x) (f x))
    hfg : ∀ (x : β), Eq (f x) (g x) → Eq (f' x) (g' x)
    ⊢ Continuous fun x => ite (LE.le (f x) (g x)) (f' x) (g' x)
  -/
  refine continuous_if (fun a ha => hfg _ (frontier_le_subset_eq hf hg ha)) ?_ (hg'.mono ?_)
    /-
      case refine_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderClosedTopology α
      f g : β → α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : (x : β) → Decidable (LE.le (f x) (g x))
      f' g' : β → γ
      hf : Continuous f
      hg : Continuous g
      hf' : ContinuousOn f' (setOf fun x => LE.le (f x) (g x))
      hg' : ContinuousOn g' (setOf fun x => LE.le (g x) (f x))
      hfg : ∀ (x : β), Eq (f x) (g x) → Eq (f' x) (g' x)
      ⊢ ContinuousOn f' (closure (setOf fun x => LE.le (f x) (g x)))
    -/
  · rwa [(isClosed_le hf hg).closure_eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderClosedTopology α
      f g : β → α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : (x : β) → Decidable (LE.le (f x) (g x))
      f' g' : β → γ
      hf : Continuous f
      hg : Continuous g
      hf' : ContinuousOn f' (setOf fun x => LE.le (f x) (g x))
      hg' : ContinuousOn g' (setOf fun x => LE.le (g x) (f x))
      hfg : ∀ (x : β), Eq (f x) (g x) → Eq (f' x) (g' x)
      ⊢ HasSubset.Subset (closure (setOf fun x => Not (LE.le (f x) (g x)))) (setOf f …
    -/
  · simp only [not_le]
    /-
      case refine_2
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : OrderClosedTopology α
      f g : β → α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : (x : β) → Decidable (LE.le (f x) (g x))
      f' g' : β → γ
      hf : Continuous f
      hg : Continuous g
      hf' : ContinuousOn f' (setOf fun x => LE.le (f x) (g x))
      hg' : ContinuousOn g' (setOf fun x => LE.le (g x) (f x))
      hfg : ∀ (x : β), Eq (f x) (g x) → Eq (f' x) (g' x)
      ⊢ HasSubset.Subset (closure (setOf fun x => LT.lt (g x) (f x))) (setOf fun x = …
    -/
    exact closure_lt_subset_le hg hf
    /-
      🎉 no goals
    -/


theorem Continuous.if_le [TopologicalSpace γ] [∀ x, Decidable (f x ≤ g x)] {f' g' : β → γ}
    (hf' : Continuous f') (hg' : Continuous g') (hf : Continuous f) (hg : Continuous g)
    (hfg : ∀ x, f x = g x → f' x = g' x) : Continuous fun x => if f x ≤ g x then f' x else g' x :=
  continuous_if_le hf hg hf'.continuousOn hg'.continuousOn hfg


theorem Filter.Tendsto.eventually_lt {l : Filter γ} {f g : γ → α} {y z : α} (hf : Tendsto f l (𝓝 y))
    (hg : Tendsto g l (𝓝 z)) (hyz : y < z) : ∀ᶠ x in l, f x < g x :=
  let ⟨_a, ha, _b, hb, h⟩ := hyz.exists_disjoint_Iio_Ioi
  (hg.eventually (Ioi_mem_nhds hb)).mp <| (hf.eventually (Iio_mem_nhds ha)).mono fun _ h₁ h₂ =>
    h _ h₁ _ h₂


nonrec theorem ContinuousAt.eventually_lt {x₀ : β} (hf : ContinuousAt f x₀) (hg : ContinuousAt g x₀)
    (hfg : f x₀ < g x₀) : ∀ᶠ x in 𝓝 x₀, f x < g x :=
  hf.eventually_lt hg hfg


@[continuity, fun_prop]
protected theorem Continuous.min (hf : Continuous f) (hg : Continuous g) :
    Continuous fun b => min (f b) (g b) := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous fun b => Min.min (f b) (g b)
  -/
  simp only [min_def]
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    f g : β → α
    inst✝ : TopologicalSpace β
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous fun b => ite (LE.le (f b) (g b)) (f b) (g b)
  -/
  exact hf.if_le hg hf hg fun x => id
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
protected theorem Continuous.max (hf : Continuous f) (hg : Continuous g) :
    Continuous fun b => max (f b) (g b) :=
  Continuous.min (α := αᵒᵈ) hf hg


theorem continuous_min : Continuous fun p : α × α => min p.1 p.2 :=
  continuous_fst.min continuous_snd


theorem continuous_max : Continuous fun p : α × α => max p.1 p.2 :=
  continuous_fst.max continuous_snd


protected theorem Filter.Tendsto.max {b : Filter β} {a₁ a₂ : α} (hf : Tendsto f b (𝓝 a₁))
    (hg : Tendsto g b (𝓝 a₂)) : Tendsto (fun b => max (f b) (g b)) b (𝓝 (max a₁ a₂)) :=
  (continuous_max.tendsto (a₁, a₂)).comp (hf.prod_mk_nhds hg)


protected theorem Filter.Tendsto.min {b : Filter β} {a₁ a₂ : α} (hf : Tendsto f b (𝓝 a₁))
    (hg : Tendsto g b (𝓝 a₂)) : Tendsto (fun b => min (f b) (g b)) b (𝓝 (min a₁ a₂)) :=
  (continuous_min.tendsto (a₁, a₂)).comp (hf.prod_mk_nhds hg)


protected theorem Filter.Tendsto.max_right {l : Filter β} {a : α} (h : Tendsto f l (𝓝 a)) :
    Tendsto (fun i => max a (f i)) l (𝓝 a) := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhds a)
    ⊢ Filter.Tendsto (fun i => Max.max a (f i)) l (nhds a)
  -/
  convert ((continuous_max.comp (@Continuous.Prod.mk α α _ _ a)).tendsto a).comp h
  /-
    case h.e'_5.h.e'_3
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhds a)
    ⊢ Eq a (Function.comp (fun p => Max.max p.1 p.2) (fun y => { fst := a, snd :=  …
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem Filter.Tendsto.max_left {l : Filter β} {a : α} (h : Tendsto f l (𝓝 a)) :
    Tendsto (fun i => max (f i) a) l (𝓝 a) := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhds a)
    ⊢ Filter.Tendsto (fun i => Max.max (f i) a) l (nhds a)
  -/
  simp_rw [max_comm _ a]
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhds a)
    ⊢ Filter.Tendsto (fun i => Max.max a (f i)) l (nhds a)
  -/
  exact h.max_right
  /-
    🎉 no goals
  -/


theorem Filter.tendsto_nhds_max_right {l : Filter β} {a : α} (h : Tendsto f l (𝓝[>] a)) :
    Tendsto (fun i => max a (f i)) l (𝓝[>] a) := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhdsWithin a (Set.Ioi a))
    ⊢ Filter.Tendsto (fun i => Max.max a (f i)) l (nhdsWithin a (Set.Ioi a))
  -/
  obtain ⟨h₁ : Tendsto f l (𝓝 a), h₂ : ∀ᶠ i in l, f i ∈ Ioi a⟩ := tendsto_nhdsWithin_iff.mp h
  /-
    case intro
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhdsWithin a (Set.Ioi a))
    h₁ : Filter.Tendsto f l (nhds a)
    h₂ : Filter.Eventually (fun i => Membership.mem (Set.Ioi a) (f i)) l
    ⊢ Filter.Tendsto (fun i => Max.max a (f i)) l (nhdsWithin a (Set.Ioi a))
  -/
  exact tendsto_nhdsWithin_iff.mpr ⟨h₁.max_right, h₂.mono fun i hi => lt_max_of_lt_right hi⟩
  /-
    🎉 no goals
  -/


theorem Filter.tendsto_nhds_max_left {l : Filter β} {a : α} (h : Tendsto f l (𝓝[>] a)) :
    Tendsto (fun i => max (f i) a) l (𝓝[>] a) := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhdsWithin a (Set.Ioi a))
    ⊢ Filter.Tendsto (fun i => Max.max (f i) a) l (nhdsWithin a (Set.Ioi a))
  -/
  simp_rw [max_comm _ a]
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    f : β → α
    l : Filter β
    a : α
    h : Filter.Tendsto f l (nhdsWithin a (Set.Ioi a))
    ⊢ Filter.Tendsto (fun i => Max.max a (f i)) l (nhdsWithin a (Set.Ioi a))
  -/
  exact Filter.tendsto_nhds_max_right h
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.min_right {l : Filter β} {a : α} (h : Tendsto f l (𝓝 a)) :
    Tendsto (fun i => min a (f i)) l (𝓝 a) :=
  Filter.Tendsto.max_right (α := αᵒᵈ) h


theorem Filter.Tendsto.min_left {l : Filter β} {a : α} (h : Tendsto f l (𝓝 a)) :
    Tendsto (fun i => min (f i) a) l (𝓝 a) :=
  Filter.Tendsto.max_left (α := αᵒᵈ) h


theorem Filter.tendsto_nhds_min_right {l : Filter β} {a : α} (h : Tendsto f l (𝓝[<] a)) :
    Tendsto (fun i => min a (f i)) l (𝓝[<] a) :=
  Filter.tendsto_nhds_max_right (α := αᵒᵈ) h


theorem Filter.tendsto_nhds_min_left {l : Filter β} {a : α} (h : Tendsto f l (𝓝[<] a)) :
    Tendsto (fun i => min (f i) a) l (𝓝[<] a) :=
  Filter.tendsto_nhds_max_left (α := αᵒᵈ) h


theorem Dense.exists_between [DenselyOrdered α] {s : Set α} (hs : Dense s) {x y : α} (h : x < y) :
    ∃ z ∈ s, z ∈ Ioo x y :=
  hs.exists_mem_open isOpen_Ioo (nonempty_Ioo.2 h)


theorem Dense.Ioi_eq_biUnion [DenselyOrdered α] {s : Set α} (hs : Dense s) (x : α) :
    Ioi x = ⋃ y ∈ s ∩ Ioi x, Ioi y := by
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    inst✝ : DenselyOrdered α
    s : Set α
    hs : Dense s
    x : α
    ⊢ Eq (Set.Ioi x) (Set.iUnion fun y => Set.iUnion fun h => Set.Ioi y)
  -/
  refine Subset.antisymm (fun z hz ↦ ?_) (iUnion₂_subset fun y hy ↦ Ioi_subset_Ioi (le_of_lt hy.2))
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    inst✝ : DenselyOrdered α
    s : Set α
    hs : Dense s
    x z : α
    hz : Membership.mem (Set.Ioi x) z
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Set.Ioi y) z
  -/
  rcases hs.exists_between hz with ⟨y, hys, hxy, hyz⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    inst✝ : DenselyOrdered α
    s : Set α
    hs : Dense s
    x z : α
    hz : Membership.mem (Set.Ioi x) z
    y : α
    hys : Membership.mem s y
    hxy : LT.lt x y
    hyz : LT.lt y z
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Set.Ioi y) z
  -/
  exact mem_iUnion₂.2 ⟨y, ⟨hys, hxy⟩, hyz⟩
  /-
    🎉 no goals
  -/


theorem Dense.Iio_eq_biUnion [DenselyOrdered α] {s : Set α} (hs : Dense s) (x : α) :
    Iio x = ⋃ y ∈ s ∩ Iio x, Iio y :=
  Dense.Ioi_eq_biUnion (α := αᵒᵈ) hs x


instance [Preorder α] [TopologicalSpace α] [OrderClosedTopology α] [Preorder β] [TopologicalSpace β]
    [OrderClosedTopology β] : OrderClosedTopology (α × β) :=
  ⟨(isClosed_le continuous_fst.fst continuous_snd.fst).inter
    (isClosed_le continuous_fst.snd continuous_snd.snd)⟩


instance {ι : Type*} {α : ι → Type*} [∀ i, Preorder (α i)] [∀ i, TopologicalSpace (α i)]
    [∀ i, OrderClosedTopology (α i)] : OrderClosedTopology (∀ i, α i) := by
  /-
    α✝ : Type u
    β : Type v
    γ : Type w
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Preorder (α i)
    inst✝¹ : (i : ι) → TopologicalSpace (α i)
    inst✝ : ∀ (i : ι), OrderClosedTopology (α i)
    ⊢ OrderClosedTopology ((i : ι) → α i)
  -/
  constructor
  /-
    case isClosed_le'
    α✝ : Type u
    β : Type v
    γ : Type w
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Preorder (α i)
    inst✝¹ : (i : ι) → TopologicalSpace (α i)
    inst✝ : ∀ (i : ι), OrderClosedTopology (α i)
    ⊢ IsClosed (setOf fun p => LE.le p.1 p.2)
  -/
  simp only [Pi.le_def, setOf_forall]
  /-
    case isClosed_le'
    α✝ : Type u
    β : Type v
    γ : Type w
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Preorder (α i)
    inst✝¹ : (i : ι) → TopologicalSpace (α i)
    inst✝ : ∀ (i : ι), OrderClosedTopology (α i)
    ⊢ IsClosed (Set.iInter fun i => setOf fun x => LE.le (x.1 i) (x.2 i))
  -/
  exact isClosed_iInter fun i => isClosed_le (continuous_apply i).fst' (continuous_apply i).snd'
  /-
    🎉 no goals
  -/


instance Pi.orderClosedTopology' [Preorder β] [TopologicalSpace β] [OrderClosedTopology β] :
    OrderClosedTopology (α → β) :=
  inferInstance

