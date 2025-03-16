theorem IsLUB.frequently_mem {a : α} {s : Set α} (ha : IsLUB s a) (hs : s.Nonempty) :
    ∃ᶠ x in 𝓝[≤] a, x ∈ s := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a : α
    s : Set α
    ha : IsLUB s a
    hs : s.Nonempty
    ⊢ Filter.Frequently (fun x => Membership.mem s x) (nhdsWithin a (Set.Iic a))
  -/
  rcases hs with ⟨a', ha'⟩
  /-
    case intro
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a : α
    s : Set α
    ha : IsLUB s a
    a' : α
    ha' : Membership.mem s a'
    ⊢ Filter.Frequently (fun x => Membership.mem s x) (nhdsWithin a (Set.Iic a))
  -/
  intro h
  /-
    case intro
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a : α
    s : Set α
    ha : IsLUB s a
    a' : α
    ha' : Membership.mem s a'
    h : Filter.Eventually (fun x => Not ((fun x => Membership.mem s x) x)) (nhdsWi …
    ⊢ False
  -/
  rcases (ha.1 ha').eq_or_lt with (rfl | ha'a)
    /-
      case intro.inl
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      s : Set α
      a' : α
      ha' : Membership.mem s a'
      ha : IsLUB s a'
      h : Filter.Eventually (fun x => Not ((fun x => Membership.mem s x) x)) (nhdsWi …
      ⊢ False
    -/
  · exact h.self_of_nhdsWithin le_rfl ha'
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      a' : α
      ha' : Membership.mem s a'
      h : Filter.Eventually (fun x => Not ((fun x => Membership.mem s x) x)) (nhdsWi …
      ha'a : LT.lt a' a
      ⊢ False
    -/
  · rcases (mem_nhdsLE_iff_exists_Ioc_subset' ha'a).1 h with ⟨b, hba, hb⟩
    /-
      case intro.inr.intro.intro
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      a' : α
      ha' : Membership.mem s a'
      h : Filter.Eventually (fun x => Not ((fun x => Membership.mem s x) x)) (nhdsWi …
      ha'a : LT.lt a' a
      b : α
      hba : Membership.mem (Set.Iio a) b
      hb : HasSubset.Subset (Set.Ioc b a) (setOf fun x => (fun x => Not ((fun x => M …
      ⊢ False
    -/
    rcases ha.exists_between hba with ⟨b', hb's, hb'⟩
    /-
      case intro.inr.intro.intro.intro.intro
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      s : Set α
      ha : IsLUB s a
      a' : α
      ha' : Membership.mem s a'
      h : Filter.Eventually (fun x => Not ((fun x => Membership.mem s x) x)) (nhdsWi …
      ha'a : LT.lt a' a
      b : α
      hba : Membership.mem (Set.Iio a) b
      hb : HasSubset.Subset (Set.Ioc b a) (setOf fun x => (fun x => Not ((fun x => M …
      b' : α
      hb's : Membership.mem s b'
      hb' : And (LT.lt b b') (LE.le b' a)
      ⊢ False
    -/
    exact hb hb' hb's
    /-
      🎉 no goals
    -/


theorem IsLUB.frequently_nhds_mem {a : α} {s : Set α} (ha : IsLUB s a) (hs : s.Nonempty) :
    ∃ᶠ x in 𝓝 a, x ∈ s :=
  (ha.frequently_mem hs).filter_mono inf_le_left


theorem IsGLB.frequently_mem {a : α} {s : Set α} (ha : IsGLB s a) (hs : s.Nonempty) :
    ∃ᶠ x in 𝓝[≥] a, x ∈ s :=
  IsLUB.frequently_mem (α := αᵒᵈ) ha hs


theorem IsGLB.frequently_nhds_mem {a : α} {s : Set α} (ha : IsGLB s a) (hs : s.Nonempty) :
    ∃ᶠ x in 𝓝 a, x ∈ s :=
  (ha.frequently_mem hs).filter_mono inf_le_left


theorem IsLUB.mem_closure {a : α} {s : Set α} (ha : IsLUB s a) (hs : s.Nonempty) : a ∈ closure s :=
  (ha.frequently_nhds_mem hs).mem_closure


theorem IsGLB.mem_closure {a : α} {s : Set α} (ha : IsGLB s a) (hs : s.Nonempty) : a ∈ closure s :=
  (ha.frequently_nhds_mem hs).mem_closure


theorem IsLUB.nhdsWithin_neBot {a : α} {s : Set α} (ha : IsLUB s a) (hs : s.Nonempty) :
    NeBot (𝓝[s] a) :=
  mem_closure_iff_nhdsWithin_neBot.1 (ha.mem_closure hs)


theorem IsGLB.nhdsWithin_neBot {a : α} {s : Set α} (ha : IsGLB s a) (hs : s.Nonempty) :
    NeBot (𝓝[s] a) :=
  IsLUB.nhdsWithin_neBot (α := αᵒᵈ) ha hs


theorem isLUB_of_mem_nhds {s : Set α} {a : α} {f : Filter α} (hsa : a ∈ upperBounds s) (hsf : s ∈ f)
    [NeBot (f ⊓ 𝓝 a)] : IsLUB s a :=
  ⟨hsa, fun b hb =>
    not_lt.1 fun hba =>
      have : s ∩ { a | b < a } ∈ f ⊓ 𝓝 a := inter_mem_inf hsf (IsOpen.mem_nhds (isOpen_lt' _) hba)
      let ⟨_x, ⟨hxs, hxb⟩⟩ := Filter.nonempty_of_mem this
      have : b < b := lt_of_lt_of_le hxb <| hb hxs
      lt_irrefl b this⟩


theorem isLUB_of_mem_closure {s : Set α} {a : α} (hsa : a ∈ upperBounds s) (hsf : a ∈ closure s) :
    IsLUB s a := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    s : Set α
    a : α
    hsa : Membership.mem (upperBounds s) a
    hsf : Membership.mem (closure s) a
    ⊢ IsLUB s a
  -/
  rw [mem_closure_iff_clusterPt, ClusterPt, inf_comm] at hsf
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    s : Set α
    a : α
    hsa : Membership.mem (upperBounds s) a
    hsf : (Min.min (Filter.principal s) (nhds a)).NeBot
    ⊢ IsLUB s a
  -/
  exact isLUB_of_mem_nhds hsa (mem_principal_self s)
  /-
    🎉 no goals
  -/


theorem isGLB_of_mem_nhds {s : Set α} {a : α} {f : Filter α} (hsa : a ∈ lowerBounds s) (hsf : s ∈ f)
    [NeBot (f ⊓ 𝓝 a)] :
    IsGLB s a :=
  isLUB_of_mem_nhds (α := αᵒᵈ) hsa hsf


theorem isGLB_of_mem_closure {s : Set α} {a : α} (hsa : a ∈ lowerBounds s) (hsf : a ∈ closure s) :
    IsGLB s a :=
  isLUB_of_mem_closure (α := αᵒᵈ) hsa hsf


theorem IsLUB.mem_upperBounds_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ]
    {f : α → γ} {s : Set α} {a : α} {b : γ} (hf : MonotoneOn f s) (ha : IsLUB s a)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : b ∈ upperBounds (f '' s) := by
  /-
    α : Type u_1
    γ : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : Preorder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderClosedTopology γ
    f : α → γ
    s : Set α
    a : α
    b : γ
    hf : MonotoneOn f s
    ha : IsLUB s a
    hb : Filter.Tendsto f (nhdsWithin a s) (nhds b)
    ⊢ Membership.mem (upperBounds (Set.image f s)) b
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    γ : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : Preorder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderClosedTopology γ
    f : α → γ
    s : Set α
    a : α
    b : γ
    hf : MonotoneOn f s
    ha : IsLUB s a
    hb : Filter.Tendsto f (nhdsWithin a s) (nhds b)
    x : α
    hx : Membership.mem s x
    ⊢ LE.le (f x) b
  -/
  replace ha := ha.inter_Ici_of_mem hx
  /-
    case intro.intro
    α : Type u_1
    γ : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : Preorder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderClosedTopology γ
    f : α → γ
    s : Set α
    a : α
    b : γ
    hf : MonotoneOn f s
    hb : Filter.Tendsto f (nhdsWithin a s) (nhds b)
    x : α
    hx : Membership.mem s x
    ha : IsLUB (Inter.inter s (Set.Ici x)) a
    ⊢ LE.le (f x) b
  -/
  haveI := ha.nhdsWithin_neBot ⟨x, hx, le_rfl⟩
  /-
    case intro.intro
    α : Type u_1
    γ : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : Preorder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderClosedTopology γ
    f : α → γ
    s : Set α
    a : α
    b : γ
    hf : MonotoneOn f s
    hb : Filter.Tendsto f (nhdsWithin a s) (nhds b)
    x : α
    hx : Membership.mem s x
    ha : IsLUB (Inter.inter s (Set.Ici x)) a
    this : (nhdsWithin a (Inter.inter s (Set.Ici x))).NeBot
    ⊢ LE.le (f x) b
  -/
  refine ge_of_tendsto (hb.mono_left (nhdsWithin_mono a (inter_subset_left (t := Ici x)))) ?_
  /-
    case intro.intro
    α : Type u_1
    γ : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : Preorder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderClosedTopology γ
    f : α → γ
    s : Set α
    a : α
    b : γ
    hf : MonotoneOn f s
    hb : Filter.Tendsto f (nhdsWithin a s) (nhds b)
    x : α
    hx : Membership.mem s x
    ha : IsLUB (Inter.inter s (Set.Ici x)) a
    this : (nhdsWithin a (Inter.inter s (Set.Ici x))).NeBot
    ⊢ Filter.Eventually (fun c => LE.le (f x) (f c)) (nhdsWithin a (Inter.inter s  …
  -/
  exact mem_of_superset self_mem_nhdsWithin fun y hy => hf hx hy.1 hy.2
  /-
    🎉 no goals
  -/

-- For a version of this theorem in which the convergence considered on the domain `α` is as `x : α`
-- tends to infinity, rather than tending to a point `x` in `α`, see `isLUB_of_tendsto_atTop`

theorem IsLUB.isLUB_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ] {f : α → γ}
    {s : Set α} {a : α} {b : γ} (hf : MonotoneOn f s) (ha : IsLUB s a) (hs : s.Nonempty)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : IsLUB (f '' s) b :=
  haveI := ha.nhdsWithin_neBot hs
  ⟨ha.mem_upperBounds_of_tendsto hf hb, fun _b' hb' =>
    le_of_tendsto hb (mem_of_superset self_mem_nhdsWithin fun _ hx => hb' <| mem_image_of_mem _ hx)⟩


theorem IsGLB.mem_lowerBounds_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ]
    {f : α → γ} {s : Set α} {a : α} {b : γ} (hf : MonotoneOn f s) (ha : IsGLB s a)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : b ∈ lowerBounds (f '' s) :=
  IsLUB.mem_upperBounds_of_tendsto (α := αᵒᵈ) (γ := γᵒᵈ) hf.dual ha hb

-- For a version of this theorem in which the convergence considered on the domain `α` is as
-- `x : α` tends to negative infinity, rather than tending to a point `x` in `α`, see
-- `isGLB_of_tendsto_atBot`

theorem IsGLB.isGLB_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ] {f : α → γ}
    {s : Set α} {a : α} {b : γ} (hf : MonotoneOn f s) :
    IsGLB s a → s.Nonempty → Tendsto f (𝓝[s] a) (𝓝 b) → IsGLB (f '' s) b :=
  IsLUB.isLUB_of_tendsto (α := αᵒᵈ) (γ := γᵒᵈ) hf.dual


theorem IsLUB.mem_lowerBounds_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ]
    {f : α → γ} {s : Set α} {a : α} {b : γ} (hf : AntitoneOn f s) (ha : IsLUB s a)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : b ∈ lowerBounds (f '' s) :=
  IsLUB.mem_upperBounds_of_tendsto (γ := γᵒᵈ) hf ha hb


theorem IsLUB.isGLB_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ] {f : α → γ}
    {s : Set α} {a : α} {b : γ} (hf : AntitoneOn f s) (ha : IsLUB s a) (hs : s.Nonempty)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : IsGLB (f '' s) b :=
  IsLUB.isLUB_of_tendsto (γ := γᵒᵈ) hf ha hs hb


theorem IsGLB.mem_upperBounds_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ]
    {f : α → γ} {s : Set α} {a : α} {b : γ} (hf : AntitoneOn f s) (ha : IsGLB s a)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : b ∈ upperBounds (f '' s) :=
  IsGLB.mem_lowerBounds_of_tendsto (γ := γᵒᵈ) hf ha hb


theorem IsGLB.isLUB_of_tendsto [Preorder γ] [TopologicalSpace γ] [OrderClosedTopology γ] {f : α → γ}
    {s : Set α} {a : α} {b : γ} (hf : AntitoneOn f s) (ha : IsGLB s a) (hs : s.Nonempty)
    (hb : Tendsto f (𝓝[s] a) (𝓝 b)) : IsLUB (f '' s) b :=
  IsGLB.isGLB_of_tendsto (γ := γᵒᵈ) hf ha hs hb


theorem IsLUB.mem_of_isClosed {a : α} {s : Set α} (ha : IsLUB s a) (hs : s.Nonempty)
    (sc : IsClosed s) : a ∈ s :=
  sc.closure_subset <| ha.mem_closure hs


alias IsClosed.isLUB_mem := IsLUB.mem_of_isClosed


theorem IsGLB.mem_of_isClosed {a : α} {s : Set α} (ha : IsGLB s a) (hs : s.Nonempty)
    (sc : IsClosed s) : a ∈ s :=
  sc.closure_subset <| ha.mem_closure hs


alias IsClosed.isGLB_mem := IsGLB.mem_of_isClosed


theorem IsLUB.exists_seq_strictMono_tendsto_of_not_mem {t : Set α} {x : α}
    [IsCountablyGenerated (𝓝 x)] (htx : IsLUB t x) (not_mem : x ∉ t) (ht : t.Nonempty) :
    ∃ u : ℕ → α, StrictMono u ∧ (∀ n, u n < x) ∧ Tendsto u atTop (𝓝 x) ∧ ∀ n, u n ∈ t := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    t : Set α
    x : α
    inst✝ : (nhds x).IsCountablyGenerated
    htx : IsLUB t x
    not_mem : Not (Membership.mem t x)
    ht : t.Nonempty
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (F …
  -/
  obtain ⟨v, hvx, hvt⟩ := exists_seq_forall_of_frequently (htx.frequently_mem ht)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    t : Set α
    x : α
    inst✝ : (nhds x).IsCountablyGenerated
    htx : IsLUB t x
    not_mem : Not (Membership.mem t x)
    ht : t.Nonempty
    v : Nat → α
    hvx : Filter.Tendsto v Filter.atTop (nhdsWithin x (Set.Iic x))
    hvt : ∀ (n : Nat), Membership.mem t (v n)
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (F …
  -/
  replace hvx := hvx.mono_right nhdsWithin_le_nhds
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    t : Set α
    x : α
    inst✝ : (nhds x).IsCountablyGenerated
    htx : IsLUB t x
    not_mem : Not (Membership.mem t x)
    ht : t.Nonempty
    v : Nat → α
    hvt : ∀ (n : Nat), Membership.mem t (v n)
    hvx : Filter.Tendsto v Filter.atTop (nhds x)
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (F …
  -/
  have hvx' : ∀ {n}, v n < x := (htx.1 (hvt _)).lt_of_ne (ne_of_mem_of_not_mem (hvt _) not_mem)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    t : Set α
    x : α
    inst✝ : (nhds x).IsCountablyGenerated
    htx : IsLUB t x
    not_mem : Not (Membership.mem t x)
    ht : t.Nonempty
    v : Nat → α
    hvt : ∀ (n : Nat), Membership.mem t (v n)
    hvx : Filter.Tendsto v Filter.atTop (nhds x)
    hvx' : ∀ {n : Nat}, LT.lt (v n) x
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (F …
  -/
  have : ∀ k, ∀ᶠ l in atTop, v k < v l := fun k => hvx.eventually (lt_mem_nhds hvx')
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    t : Set α
    x : α
    inst✝ : (nhds x).IsCountablyGenerated
    htx : IsLUB t x
    not_mem : Not (Membership.mem t x)
    ht : t.Nonempty
    v : Nat → α
    hvt : ∀ (n : Nat), Membership.mem t (v n)
    hvx : Filter.Tendsto v Filter.atTop (nhds x)
    hvx' : ∀ {n : Nat}, LT.lt (v n) x
    this : ∀ (k : Nat), Filter.Eventually (fun l => LT.lt (v k) (v l)) Filter.atTop
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (F …
  -/
  choose N hN hvN using fun k => ((eventually_gt_atTop k).and (this k)).exists
  refine ⟨fun k => v (N^[k] 0), strictMono_nat_of_lt_succ fun _ => ?_, fun _ => hvx',
    hvx.comp (strictMono_nat_of_lt_succ fun _ => ?_).tendsto_atTop, fun _ => hvt _⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      t : Set α
      x : α
      inst✝ : (nhds x).IsCountablyGenerated
      htx : IsLUB t x
      not_mem : Not (Membership.mem t x)
      ht : t.Nonempty
      v : Nat → α
      hvt : ∀ (n : Nat), Membership.mem t (v n)
      hvx : Filter.Tendsto v Filter.atTop (nhds x)
      hvx' : ∀ {n : Nat}, LT.lt (v n) x
      this : ∀ (k : Nat), Filter.Eventually (fun l => LT.lt (v k) (v l)) Filter.atTop
      N : Nat → Nat
      hN : ∀ (k : Nat), LT.lt k (N k)
      hvN : ∀ (k : Nat), LT.lt (v k) (v (N k))
      x✝ : Nat
      ⊢ LT.lt (v (Nat.iterate N x✝ 0)) (v (Nat.iterate N (HAdd.hAdd x✝ 1) 0))
    -/
  · rw [iterate_succ_apply']; exact hvN _
                              /-
                                🎉 no goals
                              -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      t : Set α
      x : α
      inst✝ : (nhds x).IsCountablyGenerated
      htx : IsLUB t x
      not_mem : Not (Membership.mem t x)
      ht : t.Nonempty
      v : Nat → α
      hvt : ∀ (n : Nat), Membership.mem t (v n)
      hvx : Filter.Tendsto v Filter.atTop (nhds x)
      hvx' : ∀ {n : Nat}, LT.lt (v n) x
      this : ∀ (k : Nat), Filter.Eventually (fun l => LT.lt (v k) (v l)) Filter.atTop
      N : Nat → Nat
      hN : ∀ (k : Nat), LT.lt k (N k)
      hvN : ∀ (k : Nat), LT.lt (v k) (v (N k))
      x✝ : Nat
      ⊢ LT.lt (Nat.iterate N x✝ 0) (Nat.iterate N (HAdd.hAdd x✝ 1) 0)
    -/
  · rw [iterate_succ_apply']; exact hN _
                              /-
                                🎉 no goals
                              -/


theorem IsLUB.exists_seq_monotone_tendsto {t : Set α} {x : α} [IsCountablyGenerated (𝓝 x)]
    (htx : IsLUB t x) (ht : t.Nonempty) :
    ∃ u : ℕ → α, Monotone u ∧ (∀ n, u n ≤ x) ∧ Tendsto u atTop (𝓝 x) ∧ ∀ n, u n ∈ t := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    t : Set α
    x : α
    inst✝ : (nhds x).IsCountablyGenerated
    htx : IsLUB t x
    ht : t.Nonempty
    ⊢ Exists fun u => And (Monotone u) (And (∀ (n : Nat), LE.le (u n) x) (And (Fil …
  -/
  by_cases h : x ∈ t
    /-
      case pos
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      t : Set α
      x : α
      inst✝ : (nhds x).IsCountablyGenerated
      htx : IsLUB t x
      ht : t.Nonempty
      h : Membership.mem t x
      ⊢ Exists fun u => And (Monotone u) (And (∀ (n : Nat), LE.le (u n) x) (And (Fil …
    -/
  · exact ⟨fun _ => x, monotone_const, fun n => le_rfl, tendsto_const_nhds, fun _ => h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      t : Set α
      x : α
      inst✝ : (nhds x).IsCountablyGenerated
      htx : IsLUB t x
      ht : t.Nonempty
      h : Not (Membership.mem t x)
      ⊢ Exists fun u => And (Monotone u) (And (∀ (n : Nat), LE.le (u n) x) (And (Fil …
    -/
  · rcases htx.exists_seq_strictMono_tendsto_of_not_mem h ht with ⟨u, hu⟩
    /-
      case neg.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      t : Set α
      x : α
      inst✝ : (nhds x).IsCountablyGenerated
      htx : IsLUB t x
      ht : t.Nonempty
      h : Not (Membership.mem t x)
      u : Nat → α
      hu : And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (Filter.Tendsto …
      ⊢ Exists fun u => And (Monotone u) (And (∀ (n : Nat), LE.le (u n) x) (And (Fil …
    -/
    exact ⟨u, hu.1.monotone, fun n => (hu.2.1 n).le, hu.2.2⟩
    /-
      🎉 no goals
    -/


theorem exists_seq_strictMono_tendsto' {α : Type*} [LinearOrder α] [TopologicalSpace α]
    [DenselyOrdered α] [OrderTopology α] [FirstCountableTopology α] {x y : α} (hy : y < x) :
    ∃ u : ℕ → α, StrictMono u ∧ (∀ n, u n ∈ Ioo y x) ∧ Tendsto u atTop (𝓝 x) := by
  /-
    α : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : DenselyOrdered α
    inst✝¹ : OrderTopology α
    inst✝ : FirstCountableTopology α
    x y : α
    hy : LT.lt y x
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), Membership.mem (Set.Io …
  -/
  have hx : x ∉ Ioo y x := fun h => (lt_irrefl x h.2).elim
  /-
    α : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : DenselyOrdered α
    inst✝¹ : OrderTopology α
    inst✝ : FirstCountableTopology α
    x y : α
    hy : LT.lt y x
    hx : Not (Membership.mem (Set.Ioo y x) x)
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), Membership.mem (Set.Io …
  -/
  have ht : Set.Nonempty (Ioo y x) := nonempty_Ioo.2 hy
  /-
    α : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : DenselyOrdered α
    inst✝¹ : OrderTopology α
    inst✝ : FirstCountableTopology α
    x y : α
    hy : LT.lt y x
    hx : Not (Membership.mem (Set.Ioo y x) x)
    ht : (Set.Ioo y x).Nonempty
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), Membership.mem (Set.Io …
  -/
  rcases (isLUB_Ioo hy).exists_seq_strictMono_tendsto_of_not_mem hx ht with ⟨u, hu⟩
  /-
    case intro
    α : Type u_3
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : DenselyOrdered α
    inst✝¹ : OrderTopology α
    inst✝ : FirstCountableTopology α
    x y : α
    hy : LT.lt y x
    hx : Not (Membership.mem (Set.Ioo y x) x)
    ht : (Set.Ioo y x).Nonempty
    u : Nat → α
    hu : And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (And (Filter.Tendsto …
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), Membership.mem (Set.Io …
  -/
  exact ⟨u, hu.1, hu.2.2.symm⟩
  /-
    🎉 no goals
  -/


theorem exists_seq_strictMono_tendsto [DenselyOrdered α] [NoMinOrder α] [FirstCountableTopology α]
    (x : α) : ∃ u : ℕ → α, StrictMono u ∧ (∀ n, u n < x) ∧ Tendsto u atTop (𝓝 x) := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : NoMinOrder α
    inst✝ : FirstCountableTopology α
    x : α
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (Filter …
  -/
  obtain ⟨y, hy⟩ : ∃ y, y < x := exists_lt x
  /-
    case intro
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : NoMinOrder α
    inst✝ : FirstCountableTopology α
    x y : α
    hy : LT.lt y x
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (Filter …
  -/
  rcases exists_seq_strictMono_tendsto' hy with ⟨u, hu_mono, hu_mem, hux⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : OrderTopology α
    inst✝² : DenselyOrdered α
    inst✝¹ : NoMinOrder α
    inst✝ : FirstCountableTopology α
    x y : α
    hy : LT.lt y x
    u : Nat → α
    hu_mono : StrictMono u
    hu_mem : ∀ (n : Nat), Membership.mem (Set.Ioo y x) (u n)
    hux : Filter.Tendsto u Filter.atTop (nhds x)
    ⊢ Exists fun u => And (StrictMono u) (And (∀ (n : Nat), LT.lt (u n) x) (Filter …
  -/
  exact ⟨u, hu_mono, fun n => (hu_mem n).2, hux⟩
  /-
    🎉 no goals
  -/


theorem exists_seq_strictMono_tendsto_nhdsWithin [DenselyOrdered α] [NoMinOrder α]
    [FirstCountableTopology α] (x : α) :
    ∃ u : ℕ → α, StrictMono u ∧ (∀ n, u n < x) ∧ Tendsto u atTop (𝓝[<] x) :=
  let ⟨u, hu, hx, h⟩ := exists_seq_strictMono_tendsto x
  ⟨u, hu, hx, tendsto_nhdsWithin_mono_right (range_subset_iff.2 hx) <| tendsto_nhdsWithin_range.2 h⟩


theorem exists_seq_tendsto_sSup {α : Type*} [ConditionallyCompleteLinearOrder α]
    [TopologicalSpace α] [OrderTopology α] [FirstCountableTopology α] {S : Set α} (hS : S.Nonempty)
    (hS' : BddAbove S) : ∃ u : ℕ → α, Monotone u ∧ Tendsto u atTop (𝓝 (sSup S)) ∧ ∀ n, u n ∈ S := by
  /-
    α : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : FirstCountableTopology α
    S : Set α
    hS : S.Nonempty
    hS' : BddAbove S
    ⊢ Exists fun u => And (Monotone u) (And (Filter.Tendsto u Filter.atTop (nhds ( …
  -/
  rcases (isLUB_csSup hS hS').exists_seq_monotone_tendsto hS with ⟨u, hu⟩
  /-
    case intro
    α : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : FirstCountableTopology α
    S : Set α
    hS : S.Nonempty
    hS' : BddAbove S
    u : Nat → α
    hu : And (Monotone u) (And (∀ (n : Nat), LE.le (u n) (SupSet.sSup S)) (And (Fi …
    ⊢ Exists fun u => And (Monotone u) (And (Filter.Tendsto u Filter.atTop (nhds ( …
  -/
  exact ⟨u, hu.1, hu.2.2⟩
  /-
    🎉 no goals
  -/


theorem IsGLB.exists_seq_strictAnti_tendsto_of_not_mem {t : Set α} {x : α}
    [IsCountablyGenerated (𝓝 x)] (htx : IsGLB t x) (not_mem : x ∉ t) (ht : t.Nonempty) :
    ∃ u : ℕ → α, StrictAnti u ∧ (∀ n, x < u n) ∧ Tendsto u atTop (𝓝 x) ∧ ∀ n, u n ∈ t :=
  IsLUB.exists_seq_strictMono_tendsto_of_not_mem (α := αᵒᵈ) htx not_mem ht


theorem IsGLB.exists_seq_antitone_tendsto {t : Set α} {x : α} [IsCountablyGenerated (𝓝 x)]
    (htx : IsGLB t x) (ht : t.Nonempty) :
    ∃ u : ℕ → α, Antitone u ∧ (∀ n, x ≤ u n) ∧ Tendsto u atTop (𝓝 x) ∧ ∀ n, u n ∈ t :=
  IsLUB.exists_seq_monotone_tendsto (α := αᵒᵈ) htx ht


theorem exists_seq_strictAnti_tendsto' [DenselyOrdered α] [FirstCountableTopology α] {x y : α}
    (hy : x < y) : ∃ u : ℕ → α, StrictAnti u ∧ (∀ n, u n ∈ Ioo x y) ∧ Tendsto u atTop (𝓝 x) := by
  simpa only [dual_Ioo]
    using exists_seq_strictMono_tendsto' (α := αᵒᵈ) (OrderDual.toDual_lt_toDual.2 hy)


theorem exists_seq_strictAnti_tendsto [DenselyOrdered α] [NoMaxOrder α] [FirstCountableTopology α]
    (x : α) : ∃ u : ℕ → α, StrictAnti u ∧ (∀ n, x < u n) ∧ Tendsto u atTop (𝓝 x) :=
  exists_seq_strictMono_tendsto (α := αᵒᵈ) x


theorem exists_seq_strictAnti_tendsto_nhdsWithin [DenselyOrdered α] [NoMaxOrder α]
    [FirstCountableTopology α] (x : α) :
    ∃ u : ℕ → α, StrictAnti u ∧ (∀ n, x < u n) ∧ Tendsto u atTop (𝓝[>] x) :=
  exists_seq_strictMono_tendsto_nhdsWithin (α := αᵒᵈ) _


theorem exists_seq_strictAnti_strictMono_tendsto [DenselyOrdered α] [FirstCountableTopology α]
    {x y : α} (h : x < y) :
    ∃ u v : ℕ → α, StrictAnti u ∧ StrictMono v ∧ (∀ k, u k ∈ Ioo x y) ∧ (∀ l, v l ∈ Ioo x y) ∧
      (∀ k l, u k < v l) ∧ Tendsto u atTop (𝓝 x) ∧ Tendsto v atTop (𝓝 y) := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : FirstCountableTopology α
    x y : α
    h : LT.lt x y
    ⊢ Exists fun u => Exists fun v => And (StrictAnti u) (And (StrictMono v) (And  …
  -/
  rcases exists_seq_strictAnti_tendsto' h with ⟨u, hu_anti, hu_mem, hux⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : DenselyOrdered α
    inst✝ : FirstCountableTopology α
    x y : α
    h : LT.lt x y
    u : Nat → α
    hu_anti : StrictAnti u
    hu_mem : ∀ (n : Nat), Membership.mem (Set.Ioo x y) (u n)
    hux : Filter.Tendsto u Filter.atTop (nhds x)
    ⊢ Exists fun u => Exists fun v => And (StrictAnti u) (And (StrictMono v) (And  …
  -/
  rcases exists_seq_strictMono_tendsto' (hu_mem 0).2 with ⟨v, hv_mono, hv_mem, hvy⟩
  exact
    ⟨u, v, hu_anti, hv_mono, hu_mem, fun l => ⟨(hu_mem 0).1.trans (hv_mem l).1, (hv_mem l).2⟩,
      fun k l => (hu_anti.antitone (zero_le k)).trans_lt (hv_mem l).1, hux, hvy⟩


theorem exists_seq_tendsto_sInf {α : Type*} [ConditionallyCompleteLinearOrder α]
    [TopologicalSpace α] [OrderTopology α] [FirstCountableTopology α] {S : Set α} (hS : S.Nonempty)
    (hS' : BddBelow S) : ∃ u : ℕ → α, Antitone u ∧ Tendsto u atTop (𝓝 (sInf S)) ∧ ∀ n, u n ∈ S :=
  exists_seq_tendsto_sSup (α := αᵒᵈ) hS hS'


