/-- Ad hoc typeclass stating that neighborhoods are eventually bounded above. -/
class BoundedLENhdsClass (α : Type*) [Preorder α] [TopologicalSpace α] : Prop where
  isBounded_le_nhds (a : α) : (𝓝 a).IsBounded (· ≤ ·)


/-- Ad hoc typeclass stating that neighborhoods are eventually bounded below. -/
class BoundedGENhdsClass (α : Type*) [Preorder α] [TopologicalSpace α] : Prop where
  isBounded_ge_nhds (a : α) : (𝓝 a).IsBounded (· ≥ ·)


theorem isBounded_le_nhds (a : α) : (𝓝 a).IsBounded (· ≤ ·) :=
  BoundedLENhdsClass.isBounded_le_nhds _


theorem Filter.Tendsto.isBoundedUnder_le (h : Tendsto u f (𝓝 a)) : f.IsBoundedUnder (· ≤ ·) u :=
  (isBounded_le_nhds a).mono h


theorem Filter.Tendsto.bddAbove_range_of_cofinite [IsDirected α (· ≤ ·)]
    (h : Tendsto u cofinite (𝓝 a)) : BddAbove (Set.range u) :=
  h.isBoundedUnder_le.bddAbove_range_of_cofinite


theorem Filter.Tendsto.bddAbove_range [IsDirected α (· ≤ ·)] {u : ℕ → α}
    (h : Tendsto u atTop (𝓝 a)) : BddAbove (Set.range u) :=
  h.isBoundedUnder_le.bddAbove_range


theorem isCobounded_ge_nhds (a : α) : (𝓝 a).IsCobounded (· ≥ ·) :=
  (isBounded_le_nhds a).isCobounded_flip


theorem Filter.Tendsto.isCoboundedUnder_ge [NeBot f] (h : Tendsto u f (𝓝 a)) :
    f.IsCoboundedUnder (· ≥ ·) u :=
  h.isBoundedUnder_le.isCobounded_flip


instance : BoundedGENhdsClass αᵒᵈ := ⟨@isBounded_le_nhds α _ _ _⟩


instance Prod.instBoundedLENhdsClass : BoundedLENhdsClass (α × β) := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : BoundedLENhdsClass α
    inst✝ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a : α
    ⊢ BoundedLENhdsClass (Prod α β)
  -/
  refine ⟨fun x ↦ ?_⟩
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : BoundedLENhdsClass α
    inst✝ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a : α
    x : Prod α β
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (nhds x)
  -/
  obtain ⟨a, ha⟩ := isBounded_le_nhds x.1
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : BoundedLENhdsClass α
    inst✝ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a✝ : α
    x : Prod α β
    a : α
    ha : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (nhds x.1)
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (nhds x)
  -/
  obtain ⟨b, hb⟩ := isBounded_le_nhds x.2
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : BoundedLENhdsClass α
    inst✝ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a✝ : α
    x : Prod α β
    a : α
    ha : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (nhds x.1)
    b : β
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (nhds x.2)
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (nhds x)
  -/
  rw [← @Prod.mk.eta _ _ x, nhds_prod_eq]
  /-
    case intro.intro
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : BoundedLENhdsClass α
    inst✝ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a✝ : α
    x : Prod α β
    a : α
    ha : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x a) (nhds x.1)
    b : β
    hb : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x b) (nhds x.2)
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (SProd.sprod (nhds x.1) (nhds x. …
  -/
  exact ⟨(a, b), ha.prod_mk hb⟩
  /-
    🎉 no goals
  -/


instance Pi.instBoundedLENhdsClass [Finite ι] [∀ i, Preorder (π i)] [∀ i, TopologicalSpace (π i)]
    [∀ i, BoundedLENhdsClass (π i)] : BoundedLENhdsClass (∀ i, π i) := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁹ : Preorder α
    inst✝⁸ : Preorder β
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : TopologicalSpace β
    inst✝⁵ : BoundedLENhdsClass α
    inst✝⁴ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a : α
    inst✝³ : Finite ι
    inst✝² : (i : ι) → Preorder (π i)
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), BoundedLENhdsClass (π i)
    ⊢ BoundedLENhdsClass ((i : ι) → π i)
  -/
  refine ⟨fun x ↦ ?_⟩
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁹ : Preorder α
    inst✝⁸ : Preorder β
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : TopologicalSpace β
    inst✝⁵ : BoundedLENhdsClass α
    inst✝⁴ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a : α
    inst✝³ : Finite ι
    inst✝² : (i : ι) → Preorder (π i)
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), BoundedLENhdsClass (π i)
    x : (i : ι) → π i
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (nhds x)
  -/
  rw [nhds_pi]
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁹ : Preorder α
    inst✝⁸ : Preorder β
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : TopologicalSpace β
    inst✝⁵ : BoundedLENhdsClass α
    inst✝⁴ : BoundedLENhdsClass β
    f : Filter ι
    u : ι → α
    a : α
    inst✝³ : Finite ι
    inst✝² : (i : ι) → Preorder (π i)
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), BoundedLENhdsClass (π i)
    x : (i : ι) → π i
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (Filter.pi fun i => nhds (x i))
  -/
  choose f hf using fun i ↦ isBounded_le_nhds (x i)
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    R : Type u_4
    S : Type u_5
    π : ι → Type u_6
    inst✝⁹ : Preorder α
    inst✝⁸ : Preorder β
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : TopologicalSpace β
    inst✝⁵ : BoundedLENhdsClass α
    inst✝⁴ : BoundedLENhdsClass β
    f✝ : Filter ι
    u : ι → α
    a : α
    inst✝³ : Finite ι
    inst✝² : (i : ι) → Preorder (π i)
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), BoundedLENhdsClass (π i)
    x f : (i : ι) → π i
    hf : ∀ (i : ι), Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x (f i) …
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (Filter.pi fun i => nhds (x i))
  -/
  exact ⟨f, eventually_pi hf⟩
  /-
    🎉 no goals
  -/


theorem isBounded_ge_nhds (a : α) : (𝓝 a).IsBounded (· ≥ ·) :=
  BoundedGENhdsClass.isBounded_ge_nhds _


theorem Filter.Tendsto.isBoundedUnder_ge (h : Tendsto u f (𝓝 a)) : f.IsBoundedUnder (· ≥ ·) u :=
  (isBounded_ge_nhds a).mono h


theorem Filter.Tendsto.bddBelow_range_of_cofinite [IsDirected α (· ≥ ·)]
    (h : Tendsto u cofinite (𝓝 a)) : BddBelow (Set.range u) :=
  h.isBoundedUnder_ge.bddBelow_range_of_cofinite


theorem Filter.Tendsto.bddBelow_range [IsDirected α (· ≥ ·)] {u : ℕ → α}
    (h : Tendsto u atTop (𝓝 a)) : BddBelow (Set.range u) :=
  h.isBoundedUnder_ge.bddBelow_range


theorem isCobounded_le_nhds (a : α) : (𝓝 a).IsCobounded (· ≤ ·) :=
  (isBounded_ge_nhds a).isCobounded_flip


theorem Filter.Tendsto.isCoboundedUnder_le [NeBot f] (h : Tendsto u f (𝓝 a)) :
    f.IsCoboundedUnder (· ≤ ·) u :=
  h.isBoundedUnder_ge.isCobounded_flip


instance : BoundedLENhdsClass αᵒᵈ := ⟨@isBounded_ge_nhds α _ _ _⟩


instance Prod.instBoundedGENhdsClass : BoundedGENhdsClass (α × β) :=
  ⟨(Prod.instBoundedLENhdsClass (α := αᵒᵈ) (β := βᵒᵈ)).isBounded_le_nhds⟩


instance Pi.instBoundedGENhdsClass [Finite ι] [∀ i, Preorder (π i)] [∀ i, TopologicalSpace (π i)]
    [∀ i, BoundedGENhdsClass (π i)] : BoundedGENhdsClass (∀ i, π i) :=
  ⟨(Pi.instBoundedLENhdsClass (π := fun i ↦ (π i)ᵒᵈ)).isBounded_le_nhds⟩


instance (priority := 100) OrderTop.to_BoundedLENhdsClass [OrderTop α] : BoundedLENhdsClass α :=
  ⟨fun _a ↦ isBounded_le_of_top⟩

-- See note [lower instance priority]

instance (priority := 100) OrderBot.to_BoundedGENhdsClass [OrderBot α] : BoundedGENhdsClass α :=
  ⟨fun _a ↦ isBounded_ge_of_bot⟩


instance (priority := 100) BoundedLENhdsClass.of_closedIciTopology [LinearOrder α]
    [TopologicalSpace α] [ClosedIciTopology α] : BoundedLENhdsClass α :=
  ⟨fun a ↦ ((isTop_or_exists_gt a).elim fun h ↦ ⟨a, Eventually.of_forall h⟩) <|
    Exists.imp fun _b ↦ eventually_le_nhds⟩

-- See note [lower instance priority]

instance (priority := 100) BoundedGENhdsClass.of_closedIicTopology [LinearOrder α]
    [TopologicalSpace α] [ClosedIicTopology α] : BoundedGENhdsClass α :=
  inferInstanceAs <| BoundedGENhdsClass αᵒᵈᵒᵈ


/-- If the liminf and the limsup of a filter coincide, then this filter converges to
their common value, at least if the filter is eventually bounded above and below. -/
theorem le_nhds_of_limsSup_eq_limsInf {f : Filter α} {a : α} (hl : f.IsBounded (· ≤ ·))
    (hg : f.IsBounded (· ≥ ·)) (hs : f.limsSup = a) (hi : f.limsInf = a) : f ≤ 𝓝 a :=
  tendsto_order.2 ⟨fun _ hb ↦ gt_mem_sets_of_limsInf_gt hg <| hi.symm ▸ hb,
    fun _ hb ↦ lt_mem_sets_of_limsSup_lt hl <| hs.symm ▸ hb⟩


theorem limsSup_nhds (a : α) : limsSup (𝓝 a) = a :=
  csInf_eq_of_forall_ge_of_forall_gt_exists_lt (isBounded_le_nhds a)
    (fun a' (h : { n : α | n ≤ a' } ∈ 𝓝 a) ↦ show a ≤ a' from @mem_of_mem_nhds α a _ _ h)
    fun b (hba : a < b) ↦
    show ∃ c, { n : α | n ≤ c } ∈ 𝓝 a ∧ c < b from
      match dense_or_discrete a b with
      | Or.inl ⟨c, hac, hcb⟩ => ⟨c, ge_mem_nhds hac, hcb⟩
      | Or.inr ⟨_, h⟩ => ⟨a, (𝓝 a).sets_of_superset (gt_mem_nhds hba) h, hba⟩


theorem limsInf_nhds (a : α) : limsInf (𝓝 a) = a :=
  limsSup_nhds (α := αᵒᵈ) a


/-- If a filter is converging, its limsup coincides with its limit. -/
theorem limsInf_eq_of_le_nhds {f : Filter α} {a : α} [NeBot f] (h : f ≤ 𝓝 a) : f.limsInf = a :=
  have hb_ge : IsBounded (· ≥ ·) f := (isBounded_ge_nhds a).mono h
  have hb_le : IsBounded (· ≤ ·) f := (isBounded_le_nhds a).mono h
  le_antisymm
    (calc
      f.limsInf ≤ f.limsSup := limsInf_le_limsSup hb_le hb_ge
      _ ≤ (𝓝 a).limsSup := limsSup_le_limsSup_of_le h hb_ge.isCobounded_flip (isBounded_le_nhds a)
      _ = a := limsSup_nhds a)
    (calc
      a = (𝓝 a).limsInf := (limsInf_nhds a).symm
      _ ≤ f.limsInf := limsInf_le_limsInf_of_le h (isBounded_ge_nhds a) hb_le.isCobounded_flip)


/-- If a filter is converging, its liminf coincides with its limit. -/
theorem limsSup_eq_of_le_nhds {f : Filter α} {a : α} [NeBot f] (h : f ≤ 𝓝 a) : f.limsSup = a :=
  limsInf_eq_of_le_nhds (α := αᵒᵈ) h


/-- If a function has a limit, then its limsup coincides with its limit. -/
theorem Filter.Tendsto.limsup_eq {f : Filter β} {u : β → α} {a : α} [NeBot f]
    (h : Tendsto u f (𝓝 a)) : limsup u f = a :=
  limsSup_eq_of_le_nhds h


/-- If a function has a limit, then its liminf coincides with its limit. -/
theorem Filter.Tendsto.liminf_eq {f : Filter β} {u : β → α} {a : α} [NeBot f]
    (h : Tendsto u f (𝓝 a)) : liminf u f = a :=
  limsInf_eq_of_le_nhds h


/-- If the liminf and the limsup of a function coincide, then the limit of the function
exists and has the same value. -/
theorem tendsto_of_liminf_eq_limsup {f : Filter β} {u : β → α} {a : α} (hinf : liminf u f = a)
    (hsup : limsup u f = a) (h : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h' : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) : Tendsto u f (𝓝 a) :=
  le_nhds_of_limsSup_eq_limsInf h h' hsup hinf


/-- If a number `a` is less than or equal to the `liminf` of a function `f` at some filter
and is greater than or equal to the `limsup` of `f`, then `f` tends to `a` along this filter. -/
theorem tendsto_of_le_liminf_of_limsup_le {f : Filter β} {u : β → α} {a : α} (hinf : a ≤ liminf u f)
    (hsup : limsup u f ≤ a) (h : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h' : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) : Tendsto u f (𝓝 a) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    f : Filter β
    u : β → α
    a : α
    hinf : LE.le a (Filter.liminf u f)
    hsup : LE.le (Filter.limsup u f) a
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    ⊢ Filter.Tendsto u f (nhds a)
  -/
  rcases f.eq_or_neBot with rfl | _
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      u : β → α
      a : α
      hinf : LE.le a (Filter.liminf u Bot.bot)
      hsup : LE.le (Filter.limsup u Bot.bot) a
      h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Bot.bot u) _au …
      h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot u) _a …
      ⊢ Filter.Tendsto u Bot.bot (nhds a)
    -/
  · exact tendsto_bot
    /-
      🎉 no goals
    -/
  · exact tendsto_of_liminf_eq_limsup (le_antisymm (le_trans (liminf_le_limsup h h') hsup) hinf)
      (le_antisymm hsup (le_trans hinf (liminf_le_limsup h h'))) h h'


/-- Assume that, for any `a < b`, a sequence can not be infinitely many times below `a` and
above `b`. If it is also ultimately bounded above and below, then it has to converge. This even
works if `a` and `b` are restricted to a dense subset.
-/
theorem tendsto_of_no_upcrossings [DenselyOrdered α] {f : Filter β} {u : β → α} {s : Set α}
    (hs : Dense s) (H : ∀ a ∈ s, ∀ b ∈ s, a < b → ¬((∃ᶠ n in f, u n < a) ∧ ∃ᶠ n in f, b < u n))
    (h : f.IsBoundedUnder (· ≤ ·) u := by isBoundedDefault)
    (h' : f.IsBoundedUnder (· ≥ ·) u := by isBoundedDefault) :
    ∃ c : α, Tendsto u f (𝓝 c) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    ⊢ Exists fun c => Filter.Tendsto u f (nhds c)
  -/
  rcases f.eq_or_neBot with rfl | hbot
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      u : β → α
      s : Set α
      hs : Dense s
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
      h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Bot.bot u) _au …
      h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot u) _a …
      ⊢ Exists fun c => Filter.Tendsto u Bot.bot (nhds c)
    -/
  · exact ⟨sInf ∅, tendsto_bot⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    hbot : f.NeBot
    ⊢ Exists fun c => Filter.Tendsto u f (nhds c)
  -/
  refine ⟨limsup u f, ?_⟩
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    hbot : f.NeBot
    ⊢ Filter.Tendsto u f (nhds (Filter.limsup u f))
  -/
  apply tendsto_of_le_liminf_of_limsup_le _ le_rfl h h'
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    hbot : f.NeBot
    ⊢ LE.le (Filter.limsup u f) (Filter.liminf u f)
  -/
  by_contra! hlt
  obtain ⟨a, ⟨⟨la, au⟩, as⟩⟩ : ∃ a, (f.liminf u < a ∧ a < f.limsup u) ∧ a ∈ s :=
    dense_iff_inter_open.1 hs (Set.Ioo (f.liminf u) (f.limsup u)) isOpen_Ioo
      (Set.nonempty_Ioo.2 hlt)
  obtain ⟨b, ⟨⟨ab, bu⟩, bs⟩⟩ : ∃ b, (a < b ∧ b < f.limsup u) ∧ b ∈ s :=
    dense_iff_inter_open.1 hs (Set.Ioo a (f.limsup u)) isOpen_Ioo (Set.nonempty_Ioo.2 au)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    hbot : f.NeBot
    hlt : LT.lt (Filter.liminf u f) (Filter.limsup u f)
    a : α
    as : Membership.mem s a
    la : LT.lt (Filter.liminf u f) a
    au : LT.lt a (Filter.limsup u f)
    b : α
    bs : Membership.mem s b
    ab : LT.lt a b
    bu : LT.lt b (Filter.limsup u f)
    ⊢ False
  -/
  have A : ∃ᶠ n in f, u n < a := frequently_lt_of_liminf_lt (IsBounded.isCobounded_ge h) la
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    hbot : f.NeBot
    hlt : LT.lt (Filter.liminf u f) (Filter.limsup u f)
    a : α
    as : Membership.mem s a
    la : LT.lt (Filter.liminf u f) a
    au : LT.lt a (Filter.limsup u f)
    b : α
    bs : Membership.mem s b
    ab : LT.lt a b
    bu : LT.lt b (Filter.limsup u f)
    A : Filter.Frequently (fun n => LT.lt (u n) a) f
    ⊢ False
  -/
  have B : ∃ᶠ n in f, b < u n := frequently_lt_of_lt_limsup (IsBounded.isCobounded_le h') bu
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    f : Filter β
    u : β → α
    s : Set α
    hs : Dense s
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → LT.lt a b  …
    h : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h' : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    hbot : f.NeBot
    hlt : LT.lt (Filter.liminf u f) (Filter.limsup u f)
    a : α
    as : Membership.mem s a
    la : LT.lt (Filter.liminf u f) a
    au : LT.lt a (Filter.limsup u f)
    b : α
    bs : Membership.mem s b
    ab : LT.lt a b
    bu : LT.lt b (Filter.limsup u f)
    A : Filter.Frequently (fun n => LT.lt (u n) a) f
    B : Filter.Frequently (fun n => LT.lt b (u n)) f
    ⊢ False
  -/
  exact H a as b bs ab ⟨A, B⟩
  /-
    🎉 no goals
  -/


theorem eventually_le_limsup (hf : IsBoundedUnder (· ≤ ·) f u := by isBoundedDefault) :
    ∀ᶠ b in f, u b ≤ f.limsup u := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : FirstCountableTopology α
    f : Filter β
    inst✝ : CountableInterFilter f
    u : β → α
    hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ⊢ Filter.Eventually (fun b => LE.le (u b) (Filter.limsup u f)) f
  -/
  obtain ha | ha := isTop_or_exists_gt (f.limsup u)
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : FirstCountableTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      ha : IsTop (Filter.limsup u f)
      ⊢ Filter.Eventually (fun b => LE.le (u b) (Filter.limsup u f)) f
    -/
  · exact Eventually.of_forall fun _ => ha _
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝⁴ : ConditionallyCompleteLinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    inst✝¹ : FirstCountableTopology α
    f : Filter β
    inst✝ : CountableInterFilter f
    u : β → α
    hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    ha : Exists fun b => LT.lt (Filter.limsup u f) b
    ⊢ Filter.Eventually (fun b => LE.le (u b) (Filter.limsup u f)) f
  -/
  by_cases H : IsGLB (Set.Ioi (f.limsup u)) (f.limsup u)
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : FirstCountableTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      ha : Exists fun b => LT.lt (Filter.limsup u f) b
      H : IsGLB (Set.Ioi (Filter.limsup u f)) (Filter.limsup u f)
      ⊢ Filter.Eventually (fun b => LE.le (u b) (Filter.limsup u f)) f
    -/
  · obtain ⟨u, -, -, hua, hu⟩ := H.exists_seq_antitone_tendsto ha
    /-
      case pos.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : FirstCountableTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u✝ : β → α
      hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u✝) _auto✝
      ha : Exists fun b => LT.lt (Filter.limsup u✝ f) b
      H : IsGLB (Set.Ioi (Filter.limsup u✝ f)) (Filter.limsup u✝ f)
      u : Nat → α
      hua : Filter.Tendsto u Filter.atTop (nhds (Filter.limsup u✝ f))
      hu : ∀ (n : Nat), Membership.mem (Set.Ioi (Filter.limsup u✝ f)) (u n)
      ⊢ Filter.Eventually (fun b => LE.le (u✝ b) (Filter.limsup u✝ f)) f
    -/
    have := fun n => eventually_lt_of_limsup_lt (hu n) hf
    exact
      (eventually_countable_forall.2 this).mono fun b hb =>
        ge_of_tendsto hua <| Eventually.of_forall fun n => (hb _).le
  · obtain ⟨x, hx, xa⟩ : ∃ x, (∀ ⦃b⦄, f.limsup u < b → x ≤ b) ∧ f.limsup u < x := by
      simp only [IsGLB, IsGreatest, lowerBounds, upperBounds, Set.mem_Ioi, Set.mem_setOf_eq,
        not_and, not_forall, not_le, exists_prop] at H
      exact H fun x => le_of_lt
    /-
      case neg.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : FirstCountableTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      ha : Exists fun b => LT.lt (Filter.limsup u f) b
      H : Not (IsGLB (Set.Ioi (Filter.limsup u f)) (Filter.limsup u f))
      x : α
      hx : ∀ ⦃b : α⦄, LT.lt (Filter.limsup u f) b → LE.le x b
      xa : LT.lt (Filter.limsup u f) x
      ⊢ Filter.Eventually (fun b => LE.le (u b) (Filter.limsup u f)) f
    -/
    filter_upwards [eventually_lt_of_limsup_lt xa hf] with y hy
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : FirstCountableTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      ha : Exists fun b => LT.lt (Filter.limsup u f) b
      H : Not (IsGLB (Set.Ioi (Filter.limsup u f)) (Filter.limsup u f))
      x : α
      hx : ∀ ⦃b : α⦄, LT.lt (Filter.limsup u f) b → LE.le x b
      xa : LT.lt (Filter.limsup u f) x
      y : β
      hy : LT.lt (u y) x
      ⊢ LE.le (u y) (Filter.limsup u f)
    -/
    contrapose! hy
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝⁴ : ConditionallyCompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : OrderTopology α
      inst✝¹ : FirstCountableTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      hf : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
      ha : Exists fun b => LT.lt (Filter.limsup u f) b
      H : Not (IsGLB (Set.Ioi (Filter.limsup u f)) (Filter.limsup u f))
      x : α
      hx : ∀ ⦃b : α⦄, LT.lt (Filter.limsup u f) b → LE.le x b
      xa : LT.lt (Filter.limsup u f) x
      y : β
      hy : LT.lt (Filter.limsup u f) (u y)
      ⊢ LE.le x (u y)
    -/
    exact hx hy
    /-
      🎉 no goals
    -/


theorem eventually_liminf_le (hf : IsBoundedUnder (· ≥ ·) f u := by isBoundedDefault) :
    ∀ᶠ b in f, f.liminf u ≤ u b :=
  eventually_le_limsup (α := αᵒᵈ) hf


@[simp]
theorem limsup_eq_bot : f.limsup u = ⊥ ↔ u =ᶠ[f] ⊥ :=
  ⟨fun h =>
                        /-
                          α : Type u_2
                          β : Type u_3
                          inst✝⁴ : CompleteLinearOrder α
                          inst✝³ : TopologicalSpace α
                          inst✝² : FirstCountableTopology α
                          inst✝¹ : OrderTopology α
                          f : Filter β
                          inst✝ : CountableInterFilter f
                          u : β → α
                          h : Eq (Filter.limsup u f) Bot.bot
                          ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
                        -/
    (EventuallyLE.trans eventually_le_limsup <| Eventually.of_forall fun _ => h.le).mono fun _ hx =>
                        /-
                          🎉 no goals
                        -/
      le_antisymm hx bot_le,
    fun h => by
    /-
      α : Type u_2
      β : Type u_3
      inst✝⁴ : CompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : FirstCountableTopology α
      inst✝¹ : OrderTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      h : f.EventuallyEq u Bot.bot
      ⊢ Eq (Filter.limsup u f) Bot.bot
    -/
    rw [limsup_congr h]
    /-
      α : Type u_2
      β : Type u_3
      inst✝⁴ : CompleteLinearOrder α
      inst✝³ : TopologicalSpace α
      inst✝² : FirstCountableTopology α
      inst✝¹ : OrderTopology α
      f : Filter β
      inst✝ : CountableInterFilter f
      u : β → α
      h : f.EventuallyEq u Bot.bot
      ⊢ Eq (Filter.limsup Bot.bot f) Bot.bot
    -/
    exact limsup_const_bot⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem liminf_eq_top : f.liminf u = ⊤ ↔ u =ᶠ[f] ⊤ :=
  limsup_eq_bot (α := αᵒᵈ)


/-- An antitone function between (conditionally) complete linear ordered spaces sends a
`Filter.limsSup` to the `Filter.liminf` of the image if the function is continuous at the `limsSup`
(and the filter is bounded from above and frequently bounded from below). -/
theorem Antitone.map_limsSup_of_continuousAt {F : Filter R} [NeBot F] {f : R → S}
    (f_decr : Antitone f) (f_cont : ContinuousAt f F.limsSup)
    (bdd_above : F.IsBounded (· ≤ ·) := by isBoundedDefault)
    (cobdd : F.IsCobounded (· ≤ ·) := by isBoundedDefault) :
    f F.limsSup = F.liminf f := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝⁶ : ConditionallyCompleteLinearOrder R
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : OrderTopology R
    inst✝³ : ConditionallyCompleteLinearOrder S
    inst✝² : TopologicalSpace S
    inst✝¹ : OrderTopology S
    F : Filter R
    inst✝ : F.NeBot
    f : R → S
    f_decr : Antitone f
    f_cont : ContinuousAt f F.limsSup
    bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
    cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
    ⊢ Eq (f F.limsSup) (Filter.liminf f F)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      ⊢ LE.le (f F.limsSup) (Filter.liminf f F)
    -/
  · rw [limsSup, f_decr.map_csInf_of_continuousAt f_cont bdd_above cobdd]
    /-
      case a
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      ⊢ LE.le (SupSet.sSup (Set.image f (setOf fun a => Filter.Eventually (fun n =>  …
    -/
    apply le_of_forall_lt
    /-
      case a.H
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      ⊢ ∀ (c : S), LT.lt c (SupSet.sSup (Set.image f (setOf fun a => Filter.Eventual …
    -/
    intro c hc
    /-
      case a.H
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      c : S
      hc : LT.lt c (SupSet.sSup (Set.image f (setOf fun a => Filter.Eventually (fun  …
      ⊢ LT.lt c (Filter.liminf f F)
    -/
    simp only [liminf, limsInf, eventually_map] at hc ⊢
    obtain ⟨d, hd, h'd⟩ :=
      exists_lt_of_lt_csSup (bdd_above.recOn fun x hx ↦ ⟨f x, Set.mem_image_of_mem f hx⟩) hc
    /-
      case a.H.intro.intro
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      c : S
      hc : LT.lt c (SupSet.sSup (Set.image f (setOf fun a => Filter.Eventually (fun  …
      d : S
      hd : Membership.mem (Set.image f (setOf fun a => Filter.Eventually (fun n => L …
      h'd : LT.lt c d
      ⊢ LT.lt c (SupSet.sSup (setOf fun a => Filter.Eventually (fun a_1 => LE.le a ( …
    -/
    apply lt_csSup_of_lt ?_ ?_ h'd
    · simpa only [BddAbove, upperBounds]
        using Antitone.isCoboundedUnder_ge_of_isCobounded f_decr cobdd
      /-
        R : Type u_4
        S : Type u_5
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        inst✝³ : ConditionallyCompleteLinearOrder S
        inst✝² : TopologicalSpace S
        inst✝¹ : OrderTopology S
        F : Filter R
        inst✝ : F.NeBot
        f : R → S
        f_decr : Antitone f
        f_cont : ContinuousAt f F.limsSup
        bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        c : S
        hc : LT.lt c (SupSet.sSup (Set.image f (setOf fun a => Filter.Eventually (fun  …
        d : S
        hd : Membership.mem (Set.image f (setOf fun a => Filter.Eventually (fun n => L …
        h'd : LT.lt c d
        ⊢ Membership.mem (setOf fun a => Filter.Eventually (fun a_1 => LE.le a (f a_1) …
      -/
    · rcases hd with ⟨e, ⟨he, fe_eq_d⟩⟩
      /-
        case intro.intro
        R : Type u_4
        S : Type u_5
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        inst✝³ : ConditionallyCompleteLinearOrder S
        inst✝² : TopologicalSpace S
        inst✝¹ : OrderTopology S
        F : Filter R
        inst✝ : F.NeBot
        f : R → S
        f_decr : Antitone f
        f_cont : ContinuousAt f F.limsSup
        bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        c : S
        hc : LT.lt c (SupSet.sSup (Set.image f (setOf fun a => Filter.Eventually (fun  …
        d : S
        h'd : LT.lt c d
        e : R
        he : Membership.mem (setOf fun a => Filter.Eventually (fun n => LE.le n a) F) e
        fe_eq_d : Eq (f e) d
        ⊢ Membership.mem (setOf fun a => Filter.Eventually (fun a_1 => LE.le a (f a_1) …
      -/
      filter_upwards [he] with x hx using (fe_eq_d.symm ▸ f_decr hx)
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      ⊢ LE.le (Filter.liminf f F) (f F.limsSup)
    -/
  · by_cases h' : ∃ c, c < F.limsSup ∧ Set.Ioo c F.limsSup = ∅
      /-
        case pos
        R : Type u_4
        S : Type u_5
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        inst✝³ : ConditionallyCompleteLinearOrder S
        inst✝² : TopologicalSpace S
        inst✝¹ : OrderTopology S
        F : Filter R
        inst✝ : F.NeBot
        f : R → S
        f_decr : Antitone f
        f_cont : ContinuousAt f F.limsSup
        bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        h' : Exists fun c => And (LT.lt c F.limsSup) (Eq (Set.Ioo c F.limsSup) EmptyCo …
        ⊢ LE.le (Filter.liminf f F) (f F.limsSup)
      -/
    · rcases h' with ⟨c, c_lt, hc⟩
      have B : ∃ᶠ n in F, F.limsSup ≤ n := by
        apply (frequently_lt_of_lt_limsSup cobdd c_lt).mono
        intro x hx
        by_contra!
        have : (Set.Ioo c F.limsSup).Nonempty := ⟨x, ⟨hx, this⟩⟩
        simp only [hc, Set.not_nonempty_empty] at this
      /-
        case pos.intro.intro
        R : Type u_4
        S : Type u_5
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        inst✝³ : ConditionallyCompleteLinearOrder S
        inst✝² : TopologicalSpace S
        inst✝¹ : OrderTopology S
        F : Filter R
        inst✝ : F.NeBot
        f : R → S
        f_decr : Antitone f
        f_cont : ContinuousAt f F.limsSup
        bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        c : R
        c_lt : LT.lt c F.limsSup
        hc : Eq (Set.Ioo c F.limsSup) EmptyCollection.emptyCollection
        B : Filter.Frequently (fun n => LE.le F.limsSup n) F
        ⊢ LE.le (Filter.liminf f F) (f F.limsSup)
      -/
      apply liminf_le_of_frequently_le _ (bdd_above.isBoundedUnder f_decr)
      /-
        R : Type u_4
        S : Type u_5
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        inst✝³ : ConditionallyCompleteLinearOrder S
        inst✝² : TopologicalSpace S
        inst✝¹ : OrderTopology S
        F : Filter R
        inst✝ : F.NeBot
        f : R → S
        f_decr : Antitone f
        f_cont : ContinuousAt f F.limsSup
        bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
        c : R
        c_lt : LT.lt c F.limsSup
        hc : Eq (Set.Ioo c F.limsSup) EmptyCollection.emptyCollection
        B : Filter.Frequently (fun n => LE.le F.limsSup n) F
        ⊢ Filter.Frequently (fun x => LE.le (f x) (f F.limsSup)) F
      -/
      exact B.mono fun x hx ↦ f_decr hx
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      h' : Not (Exists fun c => And (LT.lt c F.limsSup) (Eq (Set.Ioo c F.limsSup) Em …
      ⊢ LE.le (Filter.liminf f F) (f F.limsSup)
    -/
    push_neg at h'
    /-
      case neg
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      h' : ∀ (c : R), LT.lt c F.limsSup → (Set.Ioo c F.limsSup).Nonempty
      ⊢ LE.le (Filter.liminf f F) (f F.limsSup)
    -/
    by_contra! H
    have not_bot : ¬ IsBot F.limsSup := fun maybe_bot ↦
      lt_irrefl (F.liminf f) <| lt_of_le_of_lt
        (liminf_le_of_frequently_le (Frequently.of_forall (fun r ↦ f_decr (maybe_bot r)))
          (bdd_above.isBoundedUnder f_decr)) H
    obtain ⟨l, l_lt, h'l⟩ :
        ∃ l < F.limsSup, Set.Ioc l F.limsSup ⊆ { x : R | f x < F.liminf f } := by
      apply exists_Ioc_subset_of_mem_nhds ((tendsto_order.1 f_cont.tendsto).2 _ H)
      simpa [IsBot] using not_bot
    obtain ⟨m, l_m, m_lt⟩ : (Set.Ioo l F.limsSup).Nonempty := by
      contrapose! h'
      exact ⟨l, l_lt, h'⟩
    have B : F.liminf f ≤ f m := by
      apply liminf_le_of_frequently_le _ _
      · apply (frequently_lt_of_lt_limsSup cobdd m_lt).mono
        exact fun x hx ↦ f_decr hx.le
      · exact IsBounded.isBoundedUnder f_decr bdd_above
    /-
      case neg.intro.intro.intro.intro
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      h' : ∀ (c : R), LT.lt c F.limsSup → (Set.Ioo c F.limsSup).Nonempty
      H : LT.lt (f F.limsSup) (Filter.liminf f F)
      not_bot : Not (IsBot F.limsSup)
      l : R
      l_lt : LT.lt l F.limsSup
      h'l : HasSubset.Subset (Set.Ioc l F.limsSup) (setOf fun x => LT.lt (f x) (Filt …
      m : R
      l_m : LT.lt l m
      m_lt : LT.lt m F.limsSup
      B : LE.le (Filter.liminf f F) (f m)
      ⊢ False
    -/
    have I : f m < F.liminf f := h'l ⟨l_m, m_lt.le⟩
    /-
      case neg.intro.intro.intro.intro
      R : Type u_4
      S : Type u_5
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : ConditionallyCompleteLinearOrder S
      inst✝² : TopologicalSpace S
      inst✝¹ : OrderTopology S
      F : Filter R
      inst✝ : F.NeBot
      f : R → S
      f_decr : Antitone f
      f_cont : ContinuousAt f F.limsSup
      bdd_above : autoParam (Filter.IsBounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      cobdd : autoParam (Filter.IsCobounded (fun x1 x2 => LE.le x1 x2) F) _auto✝
      h' : ∀ (c : R), LT.lt c F.limsSup → (Set.Ioo c F.limsSup).Nonempty
      H : LT.lt (f F.limsSup) (Filter.liminf f F)
      not_bot : Not (IsBot F.limsSup)
      l : R
      l_lt : LT.lt l F.limsSup
      h'l : HasSubset.Subset (Set.Ioc l F.limsSup) (setOf fun x => LT.lt (f x) (Filt …
      m : R
      l_m : LT.lt l m
      m_lt : LT.lt m F.limsSup
      B : LE.le (Filter.liminf f F) (f m)
      I : LT.lt (f m) (Filter.liminf f F)
      ⊢ False
    -/
    exact lt_irrefl _ (B.trans_lt I)
    /-
      🎉 no goals
    -/


/-- A continuous antitone function between (conditionally) complete linear ordered spaces sends a
`Filter.limsup` to the `Filter.liminf` of the images (if the filter is bounded from above and
frequently bounded from below). -/
theorem Antitone.map_limsup_of_continuousAt {f : R → S} (f_decr : Antitone f) (a : ι → R)
    (f_cont : ContinuousAt f (F.limsup a))
    (bdd_above : F.IsBoundedUnder (· ≤ ·) a := by isBoundedDefault)
    (cobdd : F.IsCoboundedUnder (· ≤ ·) a := by isBoundedDefault) :
    f (F.limsup a) = F.liminf (f ∘ a) :=
  f_decr.map_limsSup_of_continuousAt f_cont bdd_above cobdd


/-- An antitone function between (conditionally) complete linear ordered spaces sends a
`Filter.limsInf` to the `Filter.limsup` of the image if the function is continuous at the `limsInf`
(and the filter is bounded from below and frequently bounded from above). -/
theorem Antitone.map_limsInf_of_continuousAt {F : Filter R} [NeBot F] {f : R → S}
    (f_decr : Antitone f) (f_cont : ContinuousAt f F.limsInf)
    (cobdd : F.IsCobounded (· ≥ ·) := by isBoundedDefault)
    (bdd_below : F.IsBounded (· ≥ ·) := by isBoundedDefault) : f F.limsInf = F.limsup f :=
  Antitone.map_limsSup_of_continuousAt (R := Rᵒᵈ) (S := Sᵒᵈ) f_decr.dual f_cont bdd_below cobdd


/-- A continuous antitone function between (conditionally) complete linear ordered spaces sends a
`Filter.liminf` to the `Filter.limsup` of the images (if the filter is bounded from below and
frequently bounded from above). -/
theorem Antitone.map_liminf_of_continuousAt {f : R → S} (f_decr : Antitone f) (a : ι → R)
    (f_cont : ContinuousAt f (F.liminf a))
    (cobdd : F.IsCoboundedUnder (· ≥ ·) a := by isBoundedDefault)
    (bdd_below : F.IsBoundedUnder (· ≥ ·) a := by isBoundedDefault) :
    f (F.liminf a) = F.limsup (f ∘ a) :=
  f_decr.map_limsInf_of_continuousAt f_cont cobdd bdd_below


/-- A monotone function between (conditionally) complete linear ordered spaces sends a
`Filter.limsSup` to the `Filter.limsup` of the image if the function is continuous at the `limsSup`
(and the filter is bounded from above and frequently bounded from below). -/
theorem Monotone.map_limsSup_of_continuousAt {F : Filter R} [NeBot F] {f : R → S}
    (f_incr : Monotone f) (f_cont : ContinuousAt f F.limsSup)
    (bdd_above : F.IsBounded (· ≤ ·) := by isBoundedDefault)
    (cobdd : F.IsCobounded (· ≤ ·) := by isBoundedDefault) : f F.limsSup = F.limsup f :=
  Antitone.map_limsSup_of_continuousAt (S := Sᵒᵈ) f_incr f_cont bdd_above cobdd


/-- A continuous monotone function between (conditionally) complete linear ordered spaces sends a
`Filter.limsup` to the `Filter.limsup` of the images (if the filter is bounded from above and
frequently bounded from below). -/
theorem Monotone.map_limsup_of_continuousAt {f : R → S} (f_incr : Monotone f) (a : ι → R)
    (f_cont : ContinuousAt f (F.limsup a))
    (bdd_above : F.IsBoundedUnder (· ≤ ·) a := by isBoundedDefault)
    (cobdd : F.IsCoboundedUnder (· ≤ ·) a := by isBoundedDefault) :
    f (F.limsup a) = F.limsup (f ∘ a) :=
  f_incr.map_limsSup_of_continuousAt f_cont bdd_above cobdd


/-- A monotone function between (conditionally) complete linear ordered spaces sends a
`Filter.limsInf` to the `Filter.liminf` of the image if the function is continuous at the `limsInf`
(and the filter is bounded from below and frequently bounded from above). -/
theorem Monotone.map_limsInf_of_continuousAt {F : Filter R} [NeBot F] {f : R → S}
    (f_incr : Monotone f) (f_cont : ContinuousAt f F.limsInf)
    (cobdd : F.IsCobounded (· ≥ ·) := by isBoundedDefault)
    (bdd_below : F.IsBounded (· ≥ ·) := by isBoundedDefault) : f F.limsInf = F.liminf f :=
  Antitone.map_limsSup_of_continuousAt (R := Rᵒᵈ) f_incr.dual f_cont bdd_below cobdd


/-- A continuous monotone function between (conditionally) complete linear ordered spaces sends a
`Filter.liminf` to the `Filter.liminf` of the images (if the filter is bounded from below and
frequently bounded from above). -/
theorem Monotone.map_liminf_of_continuousAt {f : R → S} (f_incr : Monotone f) (a : ι → R)
    (f_cont : ContinuousAt f (F.liminf a))
    (cobdd : F.IsCoboundedUnder (· ≥ ·) a := by isBoundedDefault)
    (bdd_below : F.IsBoundedUnder (· ≥ ·) a := by isBoundedDefault) :
    f (F.liminf a) = F.liminf (f ∘ a) :=
  f_incr.map_limsInf_of_continuousAt f_cont cobdd bdd_below


theorem limsup_eq_tendsto_sum_indicator_nat_atTop (s : ℕ → Set α) :
    limsup s atTop = { ω | Tendsto
      (fun n ↦ ∑ k ∈ Finset.range n, (s (k + 1)).indicator (1 : α → ℕ) ω) atTop atTop } := by
  /-
    α : Type u_2
    s : Nat → Set α
    ⊢ Eq (Filter.limsup s Filter.atTop) (setOf fun ω => Filter.Tendsto (fun n => ( …
  -/
  ext ω
  simp only [limsup_eq_iInf_iSup_of_nat, Set.iSup_eq_iUnion, Set.iInf_eq_iInter,
    Set.mem_iInter, Set.mem_iUnion, exists_prop]
  /-
    case h
    α : Type u_2
    s : Nat → Set α
    ω : α
    ⊢ Iff (∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1 …
  -/
  constructor
    /-
      case h.mp
      α : Type u_2
      s : Nat → Set α
      ω : α
      ⊢ (∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω) …
    -/
  · intro hω
    refine tendsto_atTop_atTop_of_monotone' (fun n m hnm ↦ Finset.sum_mono_set_of_nonneg
      (fun i ↦ Set.indicator_nonneg (fun _ _ ↦ zero_le_one) _) (Finset.range_mono hnm)) ?_
    /-
      case h.mp
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
      ⊢ Not (BddAbove (Set.range fun n => (Finset.range n).sum fun k => (s (HAdd.hAd …
    -/
    rintro ⟨i, h⟩
    /-
      case h.mp.intro
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
      i : Nat
      h : Membership.mem (upperBounds (Set.range fun n => (Finset.range n).sum fun k …
      ⊢ False
    -/
    simp only [mem_upperBounds, Set.mem_range, forall_exists_index, forall_apply_eq_imp_iff] at h
    /-
      case h.mp.intro
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
      i : Nat
      h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
      ⊢ False
    -/
    induction' i with k hk
      /-
        case h.mp.intro.zero
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        ⊢ False
      -/
    · obtain ⟨j, hj₁, hj₂⟩ := hω 1
      refine not_lt.2 (h <| j + 1)
        (lt_of_le_of_lt (Finset.sum_const_zero.symm : 0 = ∑ k ∈ Finset.range (j + 1), 0).le ?_)
      refine Finset.sum_lt_sum (fun m _ ↦ Set.indicator_nonneg (fun _ _ ↦ zero_le_one) _)
        ⟨j - 1, Finset.mem_range.2 (lt_of_le_of_lt (Nat.sub_le _ _) j.lt_succ_self), ?_⟩
      /-
        case h.mp.intro.zero.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        j : Nat
        hj₁ : GE.ge j 1
        hj₂ : Membership.mem (s j) ω
        ⊢ LT.lt 0 ((s (HAdd.hAdd (HSub.hSub j 1) 1)).indicator 1 ω)
      -/
      rw [Nat.sub_add_cancel hj₁, Set.indicator_of_mem hj₂]
      /-
        case h.mp.intro.zero.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        j : Nat
        hj₁ : GE.ge j 1
        hj₂ : Membership.mem (s j) ω
        ⊢ LT.lt 0 (1 ω)
      -/
      exact zero_lt_one
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.succ
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        hk : (∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).in …
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        ⊢ False
      -/
    · rw [imp_false] at hk
      /-
        case h.mp.intro.succ
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        hk : Not (∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1) …
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        ⊢ False
      -/
      push_neg at hk
      /-
        case h.mp.intro.succ
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        hk : Exists fun a => LT.lt k ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1) …
        ⊢ False
      -/
      obtain ⟨i, hi⟩ := hk
      /-
        case h.mp.intro.succ.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        i : Nat
        hi : LT.lt k ((Finset.range i).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω)
        ⊢ False
      -/
      obtain ⟨j, hj₁, hj₂⟩ := hω (i + 1)
      replace hi : (∑ k ∈ Finset.range i, (s (k + 1)).indicator 1 ω) = k + 1 :=
        le_antisymm (h i) hi
      /-
        case h.mp.intro.succ.intro.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        i j : Nat
        hj₁ : GE.ge j (HAdd.hAdd i 1)
        hj₂ : Membership.mem (s j) ω
        hi : Eq ((Finset.range i).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) (HAd …
        ⊢ False
      -/
      refine not_lt.2 (h <| j + 1) ?_
      /-
        case h.mp.intro.succ.intro.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        i j : Nat
        hj₁ : GE.ge j (HAdd.hAdd i 1)
        hj₂ : Membership.mem (s j) ω
        hi : Eq ((Finset.range i).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) (HAd …
        ⊢ LT.lt (HAdd.hAdd k 1) ((Finset.range (HAdd.hAdd j 1)).sum fun k => (s (HAdd. …
      -/
      rw [← Finset.sum_range_add_sum_Ico _ (i.le_succ.trans (hj₁.trans j.le_succ)), hi]
      /-
        case h.mp.intro.succ.intro.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        i j : Nat
        hj₁ : GE.ge j (HAdd.hAdd i 1)
        hj₂ : Membership.mem (s j) ω
        hi : Eq ((Finset.range i).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) (HAd …
        ⊢ LT.lt (HAdd.hAdd k 1) (HAdd.hAdd (HAdd.hAdd k 1) ((Finset.Ico i j.succ).sum  …
      -/
      refine lt_add_of_pos_right _ ?_
      /-
        case h.mp.intro.succ.intro.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        i j : Nat
        hj₁ : GE.ge j (HAdd.hAdd i 1)
        hj₂ : Membership.mem (s j) ω
        hi : Eq ((Finset.range i).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) (HAd …
        ⊢ LT.lt 0 ((Finset.Ico i j.succ).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω)
      -/
      rw [(Finset.sum_const_zero.symm : 0 = ∑ k ∈ Finset.Ico i (j + 1), 0)]
      refine Finset.sum_lt_sum (fun m _ ↦ Set.indicator_nonneg (fun _ _ ↦ zero_le_one) _)
        ⟨j - 1, Finset.mem_Ico.2 ⟨(Nat.le_sub_iff_add_le (le_trans ((le_add_iff_nonneg_left _).2
          zero_le') hj₁)).2 hj₁, lt_of_le_of_lt (Nat.sub_le _ _) j.lt_succ_self⟩, ?_⟩
      rw [Nat.sub_add_cancel (le_trans ((le_add_iff_nonneg_left _).2 zero_le') hj₁),
        Set.indicator_of_mem hj₂]
      /-
        case h.mp.intro.succ.intro.intro.intro
        α : Type u_2
        s : Nat → Set α
        ω : α
        hω : ∀ (i : Nat), Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
        k : Nat
        h : ∀ (a : Nat), LE.le ((Finset.range a).sum fun k => (s (HAdd.hAdd k 1)).indi …
        i j : Nat
        hj₁ : GE.ge j (HAdd.hAdd i 1)
        hj₂ : Membership.mem (s j) ω
        hi : Eq ((Finset.range i).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) (HAd …
        ⊢ LT.lt 0 (1 ω)
      -/
      exact zero_lt_one
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      α : Type u_2
      s : Nat → Set α
      ω : α
      ⊢ Membership.mem (setOf fun ω => Filter.Tendsto (fun n => (Finset.range n).sum …
    -/
  · rintro hω i
    /-
      case h.mpr
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : Membership.mem (setOf fun ω => Filter.Tendsto (fun n => (Finset.range n). …
      i : Nat
      ⊢ Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
    -/
    rw [Set.mem_setOf_eq, tendsto_atTop_atTop] at hω
    /-
      case h.mpr
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : ∀ (b : Nat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b ((Finset.ra …
      i : Nat
      ⊢ Exists fun i_1 => And (GE.ge i_1 i) (Membership.mem (s i_1) ω)
    -/
    by_contra! hcon
    /-
      case h.mpr
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : ∀ (b : Nat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b ((Finset.ra …
      i : Nat
      hcon : ∀ (i_1 : Nat), GE.ge i_1 i → Not (Membership.mem (s i_1) ω)
      ⊢ False
    -/
    obtain ⟨j, h⟩ := hω (i + 1)
    have : (∑ k ∈ Finset.range j, (s (k + 1)).indicator 1 ω) ≤ i := by
      have hle : ∀ j ≤ i, (∑ k ∈ Finset.range j, (s (k + 1)).indicator 1 ω) ≤ i := by
        refine fun j hij ↦
          (Finset.sum_le_card_nsmul _ _ _ ?_ : _ ≤ (Finset.range j).card • 1).trans ?_
        · exact fun m _ ↦ Set.indicator_apply_le' (fun _ ↦ le_rfl) fun _ ↦ zero_le_one
        · simpa only [Finset.card_range, smul_eq_mul, mul_one]
      by_cases hij : j < i
      · exact hle _ hij.le
      · rw [← Finset.sum_range_add_sum_Ico _ (not_lt.1 hij)]
        suffices (∑ k ∈ Finset.Ico i j, (s (k + 1)).indicator 1 ω) = 0 by
          rw [this, add_zero]
          exact hle _ le_rfl
        refine Finset.sum_eq_zero fun m hm ↦ ?_
        exact Set.indicator_of_not_mem (hcon _ <| (Finset.mem_Ico.1 hm).1.trans m.le_succ) _
    /-
      case h.mpr.intro
      α : Type u_2
      s : Nat → Set α
      ω : α
      hω : ∀ (b : Nat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b ((Finset.ra …
      i : Nat
      hcon : ∀ (i_1 : Nat), GE.ge i_1 i → Not (Membership.mem (s i_1) ω)
      j : Nat
      h : ∀ (a : Nat), LE.le j a → LE.le (HAdd.hAdd i 1) ((Finset.range a).sum fun k …
      this : LE.le ((Finset.range j).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) i
      ⊢ False
    -/
    exact not_le.2 (lt_of_lt_of_le i.lt_succ_self <| h _ le_rfl) this
    /-
      🎉 no goals
    -/


theorem limsup_eq_tendsto_sum_indicator_atTop (R : Type*) [StrictOrderedSemiring R] [Archimedean R]
    (s : ℕ → Set α) : limsup s atTop = { ω | Tendsto
      (fun n ↦ ∑ k ∈ Finset.range n, (s (k + 1)).indicator (1 : α → R) ω) atTop atTop } := by
  /-
    α : Type u_2
    R : Type u_7
    inst✝¹ : StrictOrderedSemiring R
    inst✝ : Archimedean R
    s : Nat → Set α
    ⊢ Eq (Filter.limsup s Filter.atTop) (setOf fun ω => Filter.Tendsto (fun n => ( …
  -/
  rw [limsup_eq_tendsto_sum_indicator_nat_atTop s]
  /-
    α : Type u_2
    R : Type u_7
    inst✝¹ : StrictOrderedSemiring R
    inst✝ : Archimedean R
    s : Nat → Set α
    ⊢ Eq (setOf fun ω => Filter.Tendsto (fun n => (Finset.range n).sum fun k => (s …
  -/
  ext ω
  /-
    case h
    α : Type u_2
    R : Type u_7
    inst✝¹ : StrictOrderedSemiring R
    inst✝ : Archimedean R
    s : Nat → Set α
    ω : α
    ⊢ Iff (Membership.mem (setOf fun ω => Filter.Tendsto (fun n => (Finset.range n …
  -/
  simp only [Set.mem_setOf_eq]
  rw [(_ : (fun n ↦ ∑ k ∈ Finset.range n, (s (k + 1)).indicator (1 : α → R) ω) = fun n ↦
    ↑(∑ k ∈ Finset.range n, (s (k + 1)).indicator (1 : α → ℕ) ω))]
    /-
      case h
      α : Type u_2
      R : Type u_7
      inst✝¹ : StrictOrderedSemiring R
      inst✝ : Archimedean R
      s : Nat → Set α
      ω : α
      ⊢ Iff (Filter.Tendsto (fun n => (Finset.range n).sum fun k => (s (HAdd.hAdd k  …
    -/
  · exact tendsto_natCast_atTop_iff.symm
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      R : Type u_7
      inst✝¹ : StrictOrderedSemiring R
      inst✝ : Archimedean R
      s : Nat → Set α
      ω : α
      ⊢ Eq (fun n => (Finset.range n).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω …
    -/
  · ext n
    /-
      case h
      α : Type u_2
      R : Type u_7
      inst✝¹ : StrictOrderedSemiring R
      inst✝ : Archimedean R
      s : Nat → Set α
      ω : α
      n : Nat
      ⊢ Eq ((Finset.range n).sum fun k => (s (HAdd.hAdd k 1)).indicator 1 ω) ↑((Fins …
    -/
    simp only [Set.indicator, Pi.one_apply, Finset.sum_boole, Nat.cast_id]
    /-
      🎉 no goals
    -/


lemma le_limsup_add (h₁ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u := by isBoundedDefault)
    (h₂ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u := by isBoundedDefault)
    (h₃ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f v := by isBoundedDefault)
    (h₄ : IsBoundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v := by isBoundedDefault) :
    (limsup u f) + liminf v f ≤ limsup (u + v) f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    ⊢ LE.le (HAdd.hAdd (Filter.limsup u f) (Filter.liminf v f)) (Filter.limsup (HA …
  -/
  have h := isCoboundedUnder_le_add h₄ h₂ -- These `have` tactic improve performance.
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd v u)
    ⊢ LE.le (HAdd.hAdd (Filter.limsup u f) (Filter.liminf v f)) (Filter.limsup (HA …
  -/
  have h' := isBoundedUnder_le_add h₃ h₁
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd v u)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd v u)
    ⊢ LE.le (HAdd.hAdd (Filter.limsup u f) (Filter.liminf v f)) (Filter.limsup (HA …
  -/
  rw [add_comm] at h h'
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (HAdd.hAdd (Filter.limsup u f) (Filter.liminf v f)) (Filter.limsup (HA …
  -/
  refine add_le_of_forall_lt fun a a_u b b_v ↦ (le_limsup_iff h h').2 fun c c_ab ↦ ?_
  refine ((frequently_lt_of_lt_limsup h₂ a_u).and_eventually
    (eventually_lt_of_lt_liminf b_v h₄)).mono fun _ ab_x ↦ ?_
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : LT.lt a (Filter.limsup u f)
    b : α
    b_v : LT.lt b (Filter.liminf v f)
    c : α
    c_ab : LT.lt c (HAdd.hAdd a b)
    x✝ : ι
    ab_x : And (LT.lt a (u x✝)) (LT.lt b (v x✝))
    ⊢ LT.lt c (HAdd.hAdd u v x✝)
  -/
  exact c_ab.trans (add_lt_add ab_x.1 ab_x.2)
  /-
    🎉 no goals
  -/


lemma limsup_add_le (h₁ : IsBoundedUnder (fun x1 x2 ↦ x1 ≥ x2) f u := by isBoundedDefault)
    (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u := by isBoundedDefault)
    (h₃ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≤ x2) f v := by isBoundedDefault)
    (h₄ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f v := by isBoundedDefault) :
    limsup (u + v) f ≤ (limsup u f) + limsup v f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  have h := isCoboundedUnder_le_add h₁ h₃
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  have h' := isBoundedUnder_le_add h₂ h₄
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  refine le_add_of_forall_lt fun a a_u b b_v ↦ ?_
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : GT.gt a (Filter.limsup u f)
    b : α
    b_v : GT.gt b (Filter.limsup v f)
    ⊢ LE.le (Filter.limsup (HAdd.hAdd u v) f) (HAdd.hAdd a b)
  -/
  rw [limsup_le_iff h h']
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : GT.gt a (Filter.limsup u f)
    b : α
    b_v : GT.gt b (Filter.limsup v f)
    ⊢ ∀ (y : α), GT.gt y (HAdd.hAdd a b) → Filter.Eventually (fun a => LT.lt (HAdd …
  -/
  intro c c_ab
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : GT.gt a (Filter.limsup u f)
    b : α
    b_v : GT.gt b (Filter.limsup v f)
    c : α
    c_ab : GT.gt c (HAdd.hAdd a b)
    ⊢ Filter.Eventually (fun a => LT.lt (HAdd.hAdd u v a) c) f
  -/
  filter_upwards [eventually_lt_of_limsup_lt a_u, eventually_lt_of_limsup_lt b_v] with x a_x b_x
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : GT.gt a (Filter.limsup u f)
    b : α
    b_v : GT.gt b (Filter.limsup v f)
    c : α
    c_ab : GT.gt c (HAdd.hAdd a b)
    x : ι
    a_x : LT.lt (u x) a
    b_x : LT.lt (v x) b
    ⊢ LT.lt (HAdd.hAdd u v x) c
  -/
  exact (add_lt_add a_x b_x).trans c_ab
  /-
    🎉 no goals
  -/


lemma le_liminf_add (h₁ : IsBoundedUnder (fun x1 x2 ↦ x1 ≥ x2) f u := by isBoundedDefault)
    (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u := by isBoundedDefault)
    (h₃ : IsBoundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v := by isBoundedDefault)
    (h₄ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v := by isBoundedDefault) :
    (liminf u f) + liminf v f ≤ liminf (u + v) f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    ⊢ LE.le (HAdd.hAdd (Filter.liminf u f) (Filter.liminf v f)) (Filter.liminf (HA …
  -/
  have h := isCoboundedUnder_ge_add h₂ h₄
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (HAdd.hAdd (Filter.liminf u f) (Filter.liminf v f)) (Filter.liminf (HA …
  -/
  have h' := isBoundedUnder_ge_add h₁ h₃
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (HAdd.hAdd (Filter.liminf u f) (Filter.liminf v f)) (Filter.liminf (HA …
  -/
  refine add_le_of_forall_lt fun a a_u b b_v ↦ ?_
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : LT.lt a (Filter.liminf u f)
    b : α
    b_v : LT.lt b (Filter.liminf v f)
    ⊢ LE.le (HAdd.hAdd a b) (Filter.liminf (HAdd.hAdd u v) f)
  -/
  rw [le_liminf_iff h h']
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : LT.lt a (Filter.liminf u f)
    b : α
    b_v : LT.lt b (Filter.liminf v f)
    ⊢ ∀ (y : α), LT.lt y (HAdd.hAdd a b) → Filter.Eventually (fun a => LT.lt y (HA …
  -/
  intro c c_ab
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : LT.lt a (Filter.liminf u f)
    b : α
    b_v : LT.lt b (Filter.liminf v f)
    c : α
    c_ab : LT.lt c (HAdd.hAdd a b)
    ⊢ Filter.Eventually (fun a => LT.lt c (HAdd.hAdd u v a)) f
  -/
  filter_upwards [eventually_lt_of_lt_liminf a_u, eventually_lt_of_lt_liminf b_v] with x a_x b_x
  /-
    case h
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : LT.lt a (Filter.liminf u f)
    b : α
    b_v : LT.lt b (Filter.liminf v f)
    c : α
    c_ab : LT.lt c (HAdd.hAdd a b)
    x : ι
    a_x : LT.lt a (u x)
    b_x : LT.lt b (v x)
    ⊢ LT.lt c (HAdd.hAdd u v x)
  -/
  exact c_ab.trans (add_lt_add a_x b_x)
  /-
    🎉 no goals
  -/


lemma liminf_add_le (h₁ : IsBoundedUnder (fun x1 x2 ↦ x1 ≥ x2) f u := by isBoundedDefault)
    (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u := by isBoundedDefault)
    (h₃ : IsBoundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v := by isBoundedDefault)
    (h₄ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v := by isBoundedDefault) :
    liminf (u + v) f ≤ (limsup u f) + liminf v f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    ⊢ LE.le (Filter.liminf (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  have h := isCoboundedUnder_ge_add h₂ h₄
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (Filter.liminf (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  have h' := isBoundedUnder_ge_add h₁ h₃
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    ⊢ LE.le (Filter.liminf (HAdd.hAdd u v) f) (HAdd.hAdd (Filter.limsup u f) (Filt …
  -/
  refine le_add_of_forall_lt fun a a_u b b_v ↦ (liminf_le_iff h h').2 fun _ c_ab ↦ ?_
  refine ((frequently_lt_of_liminf_lt h₄ b_v).and_eventually
    (eventually_lt_of_limsup_lt a_u h₂)).mono fun _ ab_x ↦ ?_
  /-
    ι : Type u_1
    α : Type u_2
    inst✝⁴ : AddCommGroup α
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : DenselyOrdered α
    inst✝¹ : CovariantClass α α (fun a b => HAdd.hAdd a b) fun x1 x2 => LE.le x1 x2
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → α
    h₁ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f u) _auto✝
    h₂ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u) _auto✝
    h₃ : autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h₄ : autoParam (Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v) _auto✝
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f (HAdd.hAdd u v)
    a : α
    a_u : GT.gt a (Filter.limsup u f)
    b : α
    b_v : GT.gt b (Filter.liminf v f)
    x✝¹ : α
    c_ab : GT.gt x✝¹ (HAdd.hAdd a b)
    x✝ : ι
    ab_x : And (LT.lt (v x✝) b) (LT.lt (u x✝) a)
    ⊢ LT.lt (HAdd.hAdd u v x✝) x✝¹
  -/
  exact (add_lt_add ab_x.2 ab_x.1).trans c_ab
  /-
    🎉 no goals
  -/


lemma le_limsup_mul (h₁ : 0 ≤ᶠ[f] u) (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u)
    (h₃ : 0 ≤ᶠ[f] v) (h₄ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f v) :
    (limsup u f) * liminf v f ≤ limsup (u * v) f := by
  have h := (isBoundedUnder_of_eventually_ge (a := 0)
    <| (h₁.and h₃).mono fun x ⟨u_0, v_0⟩ ↦ mul_nonneg u_0 v_0).isCoboundedUnder_le
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => HMul.hMul (u …
    ⊢ LE.le (HMul.hMul (Filter.limsup u f) (Filter.liminf v f)) (Filter.limsup (HM …
  -/
  have h' := isBoundedUnder_le_mul_of_nonneg h₁ h₂ h₃ h₄
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => HMul.hMul (u …
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
    ⊢ LE.le (HMul.hMul (Filter.limsup u f) (Filter.liminf v f)) (Filter.limsup (HM …
  -/
  have u0 : 0 ≤ limsup u f := le_limsup_of_frequently_le h₁.frequently h₂
  have uv : 0 ≤ limsup (u * v) f :=
    le_limsup_of_frequently_le ((h₁.and h₃).mono fun _ ⟨hu, hv⟩ ↦ mul_nonneg hu hv).frequently h'
  refine mul_le_of_forall_lt_of_nonneg u0 uv fun a _ au b b0 bv ↦ (le_limsup_iff h h').2
    fun c c_ab ↦ ?_
  refine ((frequently_lt_of_lt_limsup
    (isBoundedUnder_of_eventually_ge h₁).isCoboundedUnder_le au).and_eventually
    ((eventually_lt_of_lt_liminf bv (isBoundedUnder_of_eventually_ge h₃)).and
    (h₁.and h₃))).mono fun x ⟨xa, ⟨xb, u0, _⟩⟩ ↦ ?_
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => HMul.hMul (u …
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
    u0✝ : LE.le 0 (Filter.limsup u f)
    uv : LE.le 0 (Filter.limsup (HMul.hMul u v) f)
    a : Real
    x✝¹ : GE.ge a 0
    au : LT.lt a (Filter.limsup u f)
    b : Real
    b0 : GE.ge b 0
    bv : LT.lt b (Filter.liminf v f)
    c : Real
    c_ab : LT.lt c (HMul.hMul a b)
    x : ι
    x✝ : And (LT.lt a (u x)) (And (LT.lt b (v x)) (And (LE.le (0 x) (u x)) (LE.le  …
    xa : LT.lt a (u x)
    xb : LT.lt b (v x)
    u0 : LE.le (0 x) (u x)
    right✝ : LE.le (0 x) (v x)
    ⊢ LT.lt c (HMul.hMul (u x) (v x))
  -/
  exact c_ab.trans_le (mul_le_mul xa.le xb.le b0 u0)
  /-
    🎉 no goals
  -/


lemma limsup_mul_le (h₁ : 0 ≤ᶠ[f] u) (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u)
    (h₃ : 0 ≤ᶠ[f] v) (h₄ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f v) :
    limsup (u * v) f ≤ (limsup u f) * limsup v f := by
  have h := (isBoundedUnder_of_eventually_ge (a := 0)
    <| (h₁.and h₃).mono fun x ⟨u_0, v_0⟩ ↦ mul_nonneg u_0 v_0).isCoboundedUnder_le
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => HMul.hMul (u …
    ⊢ LE.le (Filter.limsup (HMul.hMul u v) f) (HMul.hMul (Filter.limsup u f) (Filt …
  -/
  have h' := isBoundedUnder_le_mul_of_nonneg h₁ h₂ h₃ h₄
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => HMul.hMul (u …
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
    ⊢ LE.le (Filter.limsup (HMul.hMul u v) f) (HMul.hMul (Filter.limsup u f) (Filt …
  -/
  refine le_mul_of_forall_lt₀ fun a a_u b b_v ↦ (limsup_le_iff h h').2 fun c c_ab ↦ ?_
  filter_upwards [eventually_lt_of_limsup_lt a_u, eventually_lt_of_limsup_lt b_v, h₁, h₃]
    with x x_a x_b u_0 v_0
  /-
    case h
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) f fun x => HMul.hMul (u …
    h' : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f (HMul.hMul u v)
    a : Real
    a_u : GT.gt a (Filter.limsup u f)
    b : Real
    b_v : GT.gt b (Filter.limsup v f)
    c : Real
    c_ab : GT.gt c (HMul.hMul a b)
    x : ι
    x_a : LT.lt (u x) a
    x_b : LT.lt (v x) b
    u_0 : LE.le (0 x) (u x)
    v_0 : LE.le (0 x) (v x)
    ⊢ LT.lt (HMul.hMul (u x) (v x)) c
  -/
  exact (mul_le_mul x_a.le x_b.le v_0 (u_0.trans x_a.le)).trans_lt c_ab
  /-
    🎉 no goals
  -/


lemma le_liminf_mul (h₁ : 0 ≤ᶠ[f] u) (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u)
    (h₃ : 0 ≤ᶠ[f] v) (h₄ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v) :
    (liminf u f) * liminf v f ≤ liminf (u * v) f := by
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    ⊢ LE.le (HMul.hMul (Filter.liminf u f) (Filter.liminf v f)) (Filter.liminf (HM …
  -/
  have h := isCoboundedUnder_ge_mul_of_nonneg h₁ h₂ h₃ h₄
  have h' := isBoundedUnder_of_eventually_ge (a := 0)
    <| (h₁.and h₃).mono fun x ⟨u0, v0⟩ ↦ mul_nonneg u0 v0
  apply mul_le_of_forall_lt_of_nonneg (le_liminf_of_le h₂.isCoboundedUnder_ge h₁)
    (le_liminf_of_le h ((h₁.and h₃).mono fun x ⟨u0, v0⟩ ↦ mul_nonneg u0 v0))
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f fun x => HMul.hMul (u  …
    ⊢ ∀ (a' : Real), GE.ge a' 0 → LT.lt a' (Filter.liminf u f) → ∀ (b' : Real), GE …
  -/
  intro a a0 au b b0 bv
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f fun x => HMul.hMul (u  …
    a : Real
    a0 : GE.ge a 0
    au : LT.lt a (Filter.liminf u f)
    b : Real
    b0 : GE.ge b 0
    bv : LT.lt b (Filter.liminf v f)
    ⊢ LE.le (HMul.hMul a b) (Filter.liminf (HMul.hMul u v) f)
  -/
  refine (le_liminf_iff h h').2 fun c c_ab ↦ ?_
  filter_upwards [eventually_lt_of_lt_liminf au (isBoundedUnder_of_eventually_ge h₁),
    eventually_lt_of_lt_liminf bv (isBoundedUnder_of_eventually_ge h₃)] with x xa xb
  /-
    case h
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f fun x => HMul.hMul (u  …
    a : Real
    a0 : GE.ge a 0
    au : LT.lt a (Filter.liminf u f)
    b : Real
    b0 : GE.ge b 0
    bv : LT.lt b (Filter.liminf v f)
    c : Real
    c_ab : LT.lt c (HMul.hMul a b)
    x : ι
    xa : LT.lt a (u x)
    xb : LT.lt b (v x)
    ⊢ LT.lt c (HMul.hMul u v x)
  -/
  exact c_ab.trans_le (mul_le_mul xa.le xb.le b0 (a0.trans xa.le))
  /-
    🎉 no goals
  -/


lemma liminf_mul_le (h₁ : 0 ≤ᶠ[f] u) (h₂ : IsBoundedUnder (fun x1 x2 ↦ x1 ≤ x2) f u)
    (h₃ : 0 ≤ᶠ[f] v) (h₄ : IsCoboundedUnder (fun x1 x2 ↦ x1 ≥ x2) f v) :
    liminf (u * v) f ≤ (limsup u f) * liminf v f := by
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    ⊢ LE.le (Filter.liminf (HMul.hMul u v) f) (HMul.hMul (Filter.limsup u f) (Filt …
  -/
  have h := isCoboundedUnder_ge_mul_of_nonneg h₁ h₂ h₃ h₄
  have h' := isBoundedUnder_of_eventually_ge (a := 0)
    <| (h₁.and h₃).mono fun x ⟨u_0, v_0⟩ ↦ mul_nonneg u_0 v_0
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f fun x => HMul.hMul (u  …
    ⊢ LE.le (Filter.liminf (HMul.hMul u v) f) (HMul.hMul (Filter.limsup u f) (Filt …
  -/
  refine le_mul_of_forall_lt₀ fun a a_u b b_v ↦ (liminf_le_iff h h').2 fun c c_ab ↦ ?_
  refine ((frequently_lt_of_liminf_lt h₄ b_v).and_eventually ((eventually_lt_of_limsup_lt a_u).and
    (h₁.and h₃))).mono fun x ⟨x_v, x_u, u_0, v_0⟩ ↦ ?_
  /-
    ι : Type u_1
    f : Filter ι
    inst✝ : f.NeBot
    u v : ι → Real
    h₁ : f.EventuallyLE 0 u
    h₂ : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
    h₃ : f.EventuallyLE 0 v
    h₄ : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f v
    h : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) f (HMul.hMul u v)
    h' : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) f fun x => HMul.hMul (u  …
    a : Real
    a_u : GT.gt a (Filter.limsup u f)
    b : Real
    b_v : GT.gt b (Filter.liminf v f)
    c : Real
    c_ab : GT.gt c (HMul.hMul a b)
    x : ι
    x✝ : And (LT.lt (v x) b) (And (LT.lt (u x) a) (And (LE.le (0 x) (u x)) (LE.le  …
    x_v : LT.lt (v x) b
    x_u : LT.lt (u x) a
    u_0 : LE.le (0 x) (u x)
    v_0 : LE.le (0 x) (v x)
    ⊢ LT.lt (HMul.hMul u v x) c
  -/
  exact (mul_le_mul x_u.le x_v.le v_0 (u_0.trans x_u.le)).trans_lt c_ab
  /-
    🎉 no goals
  -/


/-- `liminf (c + xᵢ) = c + liminf xᵢ`. -/
lemma limsup_const_add (F : Filter ι) [NeBot F] [Add R] [ContinuousAdd R]
    [AddLeftMono R] (f : ι → R) (c : R)
    (bdd_above : F.IsBoundedUnder (· ≤ ·) f) (cobdd : F.IsCoboundedUnder (· ≤ ·) f) :
    Filter.limsup (fun i ↦ c + f i) F = c + Filter.limsup f F :=
  (Monotone.map_limsSup_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ c + x)
    (fun _ _ h ↦ add_le_add_left h c) (continuous_add_left c).continuousAt bdd_above cobdd).symm


/-- `limsup (xᵢ + c) = (limsup xᵢ) + c`. -/
lemma limsup_add_const (F : Filter ι) [NeBot F] [Add R] [ContinuousAdd R]
    [AddRightMono R] (f : ι → R) (c : R)
    (bdd_above : F.IsBoundedUnder (· ≤ ·) f) (cobdd : F.IsCoboundedUnder (· ≤ ·) f) :
    Filter.limsup (fun i ↦ f i + c) F = Filter.limsup f F + c :=
  (Monotone.map_limsSup_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ x + c)
    (fun _ _ h ↦ add_le_add_right h c) (continuous_add_right c).continuousAt bdd_above cobdd).symm


/-- `liminf (c + xᵢ) = c + limsup xᵢ`. -/
lemma liminf_const_add (F : Filter ι) [NeBot F] [Add R] [ContinuousAdd R]
    [AddLeftMono R] (f : ι → R) (c : R)
    (cobdd : F.IsCoboundedUnder (· ≥ ·) f) (bdd_below : F.IsBoundedUnder (· ≥ ·) f) :
    Filter.liminf (fun i ↦ c + f i) F = c + Filter.liminf f F :=
  (Monotone.map_limsInf_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ c + x)
    (fun _ _ h ↦ add_le_add_left h c) (continuous_add_left c).continuousAt cobdd bdd_below).symm


/-- `liminf (xᵢ + c) = (liminf xᵢ) + c`. -/
lemma liminf_add_const (F : Filter ι) [NeBot F] [Add R] [ContinuousAdd R]
    [AddRightMono R] (f : ι → R) (c : R)
    (cobdd : F.IsCoboundedUnder (· ≥ ·) f) (bdd_below : F.IsBoundedUnder (· ≥ ·) f) :
    Filter.liminf (fun i ↦ f i + c) F = Filter.liminf f F + c :=
  (Monotone.map_limsInf_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ x + c)
    (fun _ _ h ↦ add_le_add_right h c) (continuous_add_right c).continuousAt cobdd bdd_below).symm


/-- `limsup (c - xᵢ) = c - liminf xᵢ`. -/
lemma limsup_const_sub (F : Filter ι) [AddCommSemigroup R] [Sub R] [ContinuousSub R] [OrderedSub R]
    [AddLeftMono R] (f : ι → R) (c : R)
    (cobdd : F.IsCoboundedUnder (· ≥ ·) f) (bdd_below : F.IsBoundedUnder (· ≥ ·) f) :
    Filter.limsup (fun i ↦ c - f i) F = c - Filter.liminf f F := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁷ : ConditionallyCompleteLinearOrder R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : OrderTopology R
    F : Filter ι
    inst✝⁴ : AddCommSemigroup R
    inst✝³ : Sub R
    inst✝² : ContinuousSub R
    inst✝¹ : OrderedSub R
    inst✝ : AddLeftMono R
    f : ι → R
    c : R
    cobdd : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) F f
    bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) F f
    ⊢ Eq (Filter.limsup (fun i => HSub.hSub c (f i)) F) (HSub.hSub c (Filter.limin …
  -/
  rcases F.eq_or_neBot with rfl | _
    /-
      case inl
      ι : Type u_1
      R : Type u_4
      inst✝⁷ : ConditionallyCompleteLinearOrder R
      inst✝⁶ : TopologicalSpace R
      inst✝⁵ : OrderTopology R
      inst✝⁴ : AddCommSemigroup R
      inst✝³ : Sub R
      inst✝² : ContinuousSub R
      inst✝¹ : OrderedSub R
      inst✝ : AddLeftMono R
      f : ι → R
      c : R
      cobdd : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
      ⊢ Eq (Filter.limsup (fun i => HSub.hSub c (f i)) Bot.bot) (HSub.hSub c (Filter …
    -/
  · simp only [liminf, limsInf, limsup, limsSup, map_bot, eventually_bot, Set.setOf_true]
    /-
      case inl
      ι : Type u_1
      R : Type u_4
      inst✝⁷ : ConditionallyCompleteLinearOrder R
      inst✝⁶ : TopologicalSpace R
      inst✝⁵ : OrderTopology R
      inst✝⁴ : AddCommSemigroup R
      inst✝³ : Sub R
      inst✝² : ContinuousSub R
      inst✝¹ : OrderedSub R
      inst✝ : AddLeftMono R
      f : ι → R
      c : R
      cobdd : Filter.IsCoboundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
      ⊢ Eq (InfSet.sInf Set.univ) (HSub.hSub c (SupSet.sSup Set.univ))
    -/
    simp only [IsCoboundedUnder, IsCobounded, map_bot, eventually_bot, true_implies] at cobdd
    /-
      case inl
      ι : Type u_1
      R : Type u_4
      inst✝⁷ : ConditionallyCompleteLinearOrder R
      inst✝⁶ : TopologicalSpace R
      inst✝⁵ : OrderTopology R
      inst✝⁴ : AddCommSemigroup R
      inst✝³ : Sub R
      inst✝² : ContinuousSub R
      inst✝¹ : OrderedSub R
      inst✝ : AddLeftMono R
      f : ι → R
      c : R
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
      cobdd : Exists fun b => ∀ (a : R), GE.ge b a
      ⊢ Eq (InfSet.sInf Set.univ) (HSub.hSub c (SupSet.sSup Set.univ))
    -/
    rcases cobdd with ⟨x, hx⟩
    refine (csInf_le ?_ (Set.mem_univ _)).antisymm
      (tsub_le_iff_tsub_le.1 (le_csSup ?_ (Set.mem_univ _)))
      /-
        case inl.intro.refine_1
        ι : Type u_1
        R : Type u_4
        inst✝⁷ : ConditionallyCompleteLinearOrder R
        inst✝⁶ : TopologicalSpace R
        inst✝⁵ : OrderTopology R
        inst✝⁴ : AddCommSemigroup R
        inst✝³ : Sub R
        inst✝² : ContinuousSub R
        inst✝¹ : OrderedSub R
        inst✝ : AddLeftMono R
        f : ι → R
        c : R
        bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
        x : R
        hx : ∀ (a : R), GE.ge x a
        ⊢ BddBelow Set.univ
      -/
    · refine ⟨x - x, mem_lowerBounds.2 fun y ↦ ?_⟩
      /-
        case inl.intro.refine_1
        ι : Type u_1
        R : Type u_4
        inst✝⁷ : ConditionallyCompleteLinearOrder R
        inst✝⁶ : TopologicalSpace R
        inst✝⁵ : OrderTopology R
        inst✝⁴ : AddCommSemigroup R
        inst✝³ : Sub R
        inst✝² : ContinuousSub R
        inst✝¹ : OrderedSub R
        inst✝ : AddLeftMono R
        f : ι → R
        c : R
        bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
        x : R
        hx : ∀ (a : R), GE.ge x a
        y : R
        ⊢ Membership.mem Set.univ y → LE.le (HSub.hSub x x) y
      -/
      simp only [Set.mem_univ, true_implies]
      /-
        case inl.intro.refine_1
        ι : Type u_1
        R : Type u_4
        inst✝⁷ : ConditionallyCompleteLinearOrder R
        inst✝⁶ : TopologicalSpace R
        inst✝⁵ : OrderTopology R
        inst✝⁴ : AddCommSemigroup R
        inst✝³ : Sub R
        inst✝² : ContinuousSub R
        inst✝¹ : OrderedSub R
        inst✝ : AddLeftMono R
        f : ι → R
        c : R
        bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
        x : R
        hx : ∀ (a : R), GE.ge x a
        y : R
        ⊢ LE.le (HSub.hSub x x) y
      -/
      exact tsub_le_iff_tsub_le.1 (hx (x - y))
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.refine_2
        ι : Type u_1
        R : Type u_4
        inst✝⁷ : ConditionallyCompleteLinearOrder R
        inst✝⁶ : TopologicalSpace R
        inst✝⁵ : OrderTopology R
        inst✝⁴ : AddCommSemigroup R
        inst✝³ : Sub R
        inst✝² : ContinuousSub R
        inst✝¹ : OrderedSub R
        inst✝ : AddLeftMono R
        f : ι → R
        c : R
        bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
        x : R
        hx : ∀ (a : R), GE.ge x a
        ⊢ BddAbove Set.univ
      -/
    · refine ⟨x, mem_upperBounds.2 fun y ↦ ?_⟩
      /-
        case inl.intro.refine_2
        ι : Type u_1
        R : Type u_4
        inst✝⁷ : ConditionallyCompleteLinearOrder R
        inst✝⁶ : TopologicalSpace R
        inst✝⁵ : OrderTopology R
        inst✝⁴ : AddCommSemigroup R
        inst✝³ : Sub R
        inst✝² : ContinuousSub R
        inst✝¹ : OrderedSub R
        inst✝ : AddLeftMono R
        f : ι → R
        c : R
        bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Bot.bot f
        x : R
        hx : ∀ (a : R), GE.ge x a
        y : R
        ⊢ Membership.mem Set.univ y → LE.le y x
      -/
      simp only [Set.mem_univ, hx y, implies_true]
      /-
        🎉 no goals
      -/
  · exact (Antitone.map_limsInf_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ c - x)
    (fun _ _ h ↦ tsub_le_tsub_left h c) (continuous_sub_left c).continuousAt cobdd bdd_below).symm


/-- `limsup (xᵢ - c) = (limsup xᵢ) - c`. -/
lemma limsup_sub_const (F : Filter ι) [AddCommSemigroup R] [Sub R] [ContinuousSub R] [OrderedSub R]
    (f : ι → R) (c : R)
    (bdd_above : F.IsBoundedUnder (· ≤ ·) f) (cobdd : F.IsCoboundedUnder (· ≤ ·) f) :
    Filter.limsup (fun i ↦ f i - c) F = Filter.limsup f F - c := by
  /-
    ι : Type u_1
    R : Type u_4
    inst✝⁶ : ConditionallyCompleteLinearOrder R
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : OrderTopology R
    F : Filter ι
    inst✝³ : AddCommSemigroup R
    inst✝² : Sub R
    inst✝¹ : ContinuousSub R
    inst✝ : OrderedSub R
    f : ι → R
    c : R
    bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) F f
    cobdd : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) F f
    ⊢ Eq (Filter.limsup (fun i => HSub.hSub (f i) c) F) (HSub.hSub (Filter.limsup  …
  -/
  rcases F.eq_or_neBot with rfl | _
  · have {a : R} : sInf Set.univ ≤ a := by
      apply csInf_le _ (Set.mem_univ a)
      simp only [IsCoboundedUnder, IsCobounded, map_bot, eventually_bot, true_implies] at cobdd
      rcases cobdd with ⟨x, hx⟩
      refine ⟨x, mem_lowerBounds.2 fun y ↦ ?_⟩
      simp only [Set.mem_univ, hx y, implies_true]
    /-
      case inl
      ι : Type u_1
      R : Type u_4
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : AddCommSemigroup R
      inst✝² : Sub R
      inst✝¹ : ContinuousSub R
      inst✝ : OrderedSub R
      f : ι → R
      c : R
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Bot.bot f
      cobdd : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Bot.bot f
      this : ∀ {a : R}, LE.le (InfSet.sInf Set.univ) a
      ⊢ Eq (Filter.limsup (fun i => HSub.hSub (f i) c) Bot.bot) (HSub.hSub (Filter.l …
    -/
    simp only [limsup, limsSup, map_bot, eventually_bot, Set.setOf_true]
    /-
      case inl
      ι : Type u_1
      R : Type u_4
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      inst✝³ : AddCommSemigroup R
      inst✝² : Sub R
      inst✝¹ : ContinuousSub R
      inst✝ : OrderedSub R
      f : ι → R
      c : R
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Bot.bot f
      cobdd : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Bot.bot f
      this : ∀ {a : R}, LE.le (InfSet.sInf Set.univ) a
      ⊢ Eq (InfSet.sInf Set.univ) (HSub.hSub (InfSet.sInf Set.univ) c)
    -/
    exact this.antisymm (tsub_le_iff_right.2 this)
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      R : Type u_4
      inst✝⁶ : ConditionallyCompleteLinearOrder R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : OrderTopology R
      F : Filter ι
      inst✝³ : AddCommSemigroup R
      inst✝² : Sub R
      inst✝¹ : ContinuousSub R
      inst✝ : OrderedSub R
      f : ι → R
      c : R
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) F f
      cobdd : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) F f
      h✝ : F.NeBot
      ⊢ Eq (Filter.limsup (fun i => HSub.hSub (f i) c) F) (HSub.hSub (Filter.limsup  …
    -/
  · apply (Monotone.map_limsSup_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ x - c) _ _).symm
      /-
        ι : Type u_1
        R : Type u_4
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        F : Filter ι
        inst✝³ : AddCommSemigroup R
        inst✝² : Sub R
        inst✝¹ : ContinuousSub R
        inst✝ : OrderedSub R
        f : ι → R
        c : R
        bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) F f
        cobdd : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) F f
        h✝ : F.NeBot
        ⊢ Monotone fun x => HSub.hSub x c
      -/
    · exact fun _ _ h ↦ tsub_le_tsub_right h c
      /-
        🎉 no goals
      -/
      /-
        ι : Type u_1
        R : Type u_4
        inst✝⁶ : ConditionallyCompleteLinearOrder R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : OrderTopology R
        F : Filter ι
        inst✝³ : AddCommSemigroup R
        inst✝² : Sub R
        inst✝¹ : ContinuousSub R
        inst✝ : OrderedSub R
        f : ι → R
        c : R
        bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) F f
        cobdd : Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) F f
        h✝ : F.NeBot
        ⊢ ContinuousAt (fun x => HSub.hSub x c) (Filter.map f F).limsSup
      -/
    · exact (continuous_sub_right c).continuousAt
      /-
        🎉 no goals
      -/


/-- `liminf (c - xᵢ) = c - limsup xᵢ`. -/
lemma liminf_const_sub (F : Filter ι) [NeBot F] [AddCommSemigroup R] [Sub R] [ContinuousSub R]
    [OrderedSub R] [AddLeftMono R] (f : ι → R) (c : R)
    (bdd_above : F.IsBoundedUnder (· ≤ ·) f) (cobdd : F.IsCoboundedUnder (· ≤ ·) f) :
    Filter.liminf (fun i ↦ c - f i) F = c - Filter.limsup f F :=
  (Antitone.map_limsSup_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ c - x)
    (fun _ _ h ↦ tsub_le_tsub_left h c) (continuous_sub_left c).continuousAt bdd_above cobdd).symm


/-- `liminf (xᵢ - c) = (liminf xᵢ) - c`. -/
lemma liminf_sub_const (F : Filter ι) [NeBot F] [AddCommSemigroup R] [Sub R] [ContinuousSub R]
    [OrderedSub R] (f : ι → R) (c : R)
    (cobdd : F.IsCoboundedUnder (· ≥ ·) f) (bdd_below : F.IsBoundedUnder (· ≥ ·) f) :
    Filter.liminf (fun i ↦ f i - c) F = Filter.liminf f F - c :=
  (Monotone.map_limsInf_of_continuousAt (F := F.map f) (f := fun (x : R) ↦ x - c)
    (fun _ _ h ↦ tsub_le_tsub_right h c) (continuous_sub_right c).continuousAt cobdd bdd_below).symm


