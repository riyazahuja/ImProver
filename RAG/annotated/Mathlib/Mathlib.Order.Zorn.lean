/-- Local notation for the relation being considered. -/
local infixl:50 " ≺ " => r


/-- **Zorn's lemma**

If every chain has an upper bound, then there exists a maximal element. -/
theorem exists_maximal_of_chains_bounded (h : ∀ c, IsChain r c → ∃ ub, ∀ a ∈ c, a ≺ ub)
    (trans : ∀ {a b c}, a ≺ b → b ≺ c → a ≺ c) : ∃ m, ∀ a, m ≺ a → a ≺ m :=
  have : ∃ ub, ∀ a ∈ maxChain r, a ≺ ub := h _ <| maxChain_spec.left
  let ⟨ub, (hub : ∀ a ∈ maxChain r, a ≺ ub)⟩ := this
  ⟨ub, fun a ha =>
    have : IsChain r (insert a <| maxChain r) :=
      maxChain_spec.1.insert fun b hb _ => Or.inr <| trans (hub b hb) ha
    hub a <| by
      /-
        α : Type u_1
        r : α → α → Prop
        h : ∀ (c : Set α), IsChain r c → Exists fun ub => ∀ (a : α), Membership.mem c  …
        trans : ∀ {a b c : α}, r a b → r b c → r a c
        this✝ : Exists fun ub => ∀ (a : α), Membership.mem (maxChain r) a → r a ub
        ub : α
        hub : ∀ (a : α), Membership.mem (maxChain r) a → r a ub
        a : α
        ha : r ub a
        this : IsChain r (Insert.insert a (maxChain r))
        ⊢ Membership.mem (maxChain r) a
      -/
      rw [maxChain_spec.right this (subset_insert _ _)]
      /-
        α : Type u_1
        r : α → α → Prop
        h : ∀ (c : Set α), IsChain r c → Exists fun ub => ∀ (a : α), Membership.mem c  …
        trans : ∀ {a b c : α}, r a b → r b c → r a c
        this✝ : Exists fun ub => ∀ (a : α), Membership.mem (maxChain r) a → r a ub
        ub : α
        hub : ∀ (a : α), Membership.mem (maxChain r) a → r a ub
        a : α
        ha : r ub a
        this : IsChain r (Insert.insert a (maxChain r))
        ⊢ Membership.mem (Insert.insert a (maxChain r)) a
      -/
      exact mem_insert _ _⟩
      /-
        🎉 no goals
      -/


/-- A variant of Zorn's lemma. If every nonempty chain of a nonempty type has an upper bound, then
there is a maximal element.
-/
theorem exists_maximal_of_nonempty_chains_bounded [Nonempty α]
    (h : ∀ c, IsChain r c → c.Nonempty → ∃ ub, ∀ a ∈ c, a ≺ ub)
    (trans : ∀ {a b c}, a ≺ b → b ≺ c → a ≺ c) : ∃ m, ∀ a, m ≺ a → a ≺ m :=
  exists_maximal_of_chains_bounded
    (fun c hc =>
      (eq_empty_or_nonempty c).elim
        (fun h => ⟨Classical.arbitrary α, fun x hx => (h ▸ hx : x ∈ (∅ : Set α)).elim⟩) (h c hc))
    trans


theorem zorn_le (h : ∀ c : Set α, IsChain (· ≤ ·) c → BddAbove c) : ∃ m : α, IsMax m :=
  exists_maximal_of_chains_bounded h le_trans


theorem zorn_le_nonempty [Nonempty α]
    (h : ∀ c : Set α, IsChain (· ≤ ·) c → c.Nonempty → BddAbove c) : ∃ m : α, IsMax m :=
  exists_maximal_of_nonempty_chains_bounded h le_trans


theorem zorn_le₀ (s : Set α) (ih : ∀ c ⊆ s, IsChain (· ≤ ·) c → ∃ ub ∈ s, ∀ z ∈ c, z ≤ ub) :
    ∃ m, Maximal (· ∈ s) m :=
  let ⟨⟨m, hms⟩, h⟩ :=
    @zorn_le s _ fun c hc =>
      let ⟨ub, hubs, hub⟩ :=
        ih (Subtype.val '' c) (fun _ ⟨⟨_, hx⟩, _, h⟩ => h ▸ hx)
          (by
            /-
              α : Type u_1
              inst✝ : Preorder α
              s : Set α
              ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
              c : Set ↑s
              hc : IsChain (fun x1 x2 => LE.le x1 x2) c
              ⊢ IsChain (fun x1 x2 => LE.le x1 x2) (Set.image Subtype.val c)
            -/
            rintro _ ⟨p, hpc, rfl⟩ _ ⟨q, hqc, rfl⟩ hpq
            /-
              case intro.intro.intro.intro
              α : Type u_1
              inst✝ : Preorder α
              s : Set α
              ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
              c : Set ↑s
              hc : IsChain (fun x1 x2 => LE.le x1 x2) c
              p : Subtype fun x => Membership.mem s x
              hpc : Membership.mem c p
              q : Subtype fun x => Membership.mem s x
              hqc : Membership.mem c q
              hpq : Ne ↑p ↑q
              ⊢ Or ((fun x1 x2 => LE.le x1 x2) ↑p ↑q) ((fun x1 x2 => LE.le x1 x2) ↑q ↑p)
            -/
            exact hc hpc hqc fun t => hpq (Subtype.ext_iff.1 t))
            /-
              🎉 no goals
            -/
      ⟨⟨ub, hubs⟩, fun ⟨_, _⟩ hc => hub _ ⟨_, hc, rfl⟩⟩
  ⟨m, hms, fun z hzs hmz => @h ⟨z, hzs⟩ hmz⟩


theorem zorn_le_nonempty₀ (s : Set α)
    (ih : ∀ c ⊆ s, IsChain (· ≤ ·) c → ∀ y ∈ c, ∃ ub ∈ s, ∀ z ∈ c, z ≤ ub) (x : α) (hxs : x ∈ s) :
    ∃ m, x ≤ m ∧ Maximal (· ∈ s) m := by
  -- Porting note: the first three lines replace the following two lines in mathlib3.
  -- The mathlib3 `rcases` supports holes for proof obligations, this is not yet implemented in 4.
  -- rcases zorn_preorder₀ ({ y ∈ s | x ≤ y }) fun c hcs hc => ?_ with ⟨m, ⟨hms, hxm⟩, hm⟩
  -- · exact ⟨m, hms, hxm, fun z hzs hmz => hm _ ⟨hzs, hxm.trans hmz⟩ hmz⟩
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
    x : α
    hxs : Membership.mem s x
    ⊢ Exists fun m => And (LE.le x m) (Maximal (fun x => Membership.mem s x) m)
  -/
  have H := zorn_le₀ ({ y ∈ s | x ≤ y }) fun c hcs hc => ?_
    /-
      case refine_2
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
      x : α
      hxs : Membership.mem s x
      H : Exists fun m => Maximal (fun x_1 => Membership.mem (setOf fun y => And (Me …
      ⊢ Exists fun m => And (LE.le x m) (Maximal (fun x => Membership.mem s x) m)
    -/
  · rcases H with ⟨m, ⟨hms, hxm⟩, hm⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
      x : α
      hxs : Membership.mem s x
      m : α
      hm : ∀ ⦃y : α⦄, (fun x_1 => Membership.mem (setOf fun y => And (Membership.mem …
      hms : Membership.mem s m
      hxm : LE.le x m
      ⊢ Exists fun m => And (LE.le x m) (Maximal (fun x => Membership.mem s x) m)
    -/
    exact ⟨m, hxm, hms, fun z hzs hmz => @hm _ ⟨hzs, hxm.trans hmz⟩ hmz⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      α : Type u_1
      inst✝ : Preorder α
      s : Set α
      ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
      x : α
      hxs : Membership.mem s x
      c : Set α
      hcs : HasSubset.Subset c (setOf fun y => And (Membership.mem s y) (LE.le x y))
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ⊢ Exists fun ub => And (Membership.mem (setOf fun y => And (Membership.mem s y …
    -/
  · rcases c.eq_empty_or_nonempty with (rfl | ⟨y, hy⟩)
      /-
        case refine_1.inl
        α : Type u_1
        inst✝ : Preorder α
        s : Set α
        ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
        x : α
        hxs : Membership.mem s x
        hcs : HasSubset.Subset EmptyCollection.emptyCollection (setOf fun y => And (Me …
        hc : IsChain (fun x1 x2 => LE.le x1 x2) EmptyCollection.emptyCollection
        ⊢ Exists fun ub => And (Membership.mem (setOf fun y => And (Membership.mem s y …
      -/
    · exact ⟨x, ⟨hxs, le_rfl⟩, fun z => False.elim⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.intro
        α : Type u_1
        inst✝ : Preorder α
        s : Set α
        ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
        x : α
        hxs : Membership.mem s x
        c : Set α
        hcs : HasSubset.Subset c (setOf fun y => And (Membership.mem s y) (LE.le x y))
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        y : α
        hy : Membership.mem c y
        ⊢ Exists fun ub => And (Membership.mem (setOf fun y => And (Membership.mem s y …
      -/
    · rcases ih c (fun z hz => (hcs hz).1) hc y hy with ⟨z, hzs, hz⟩
      /-
        case refine_1.inr.intro.intro.intro
        α : Type u_1
        inst✝ : Preorder α
        s : Set α
        ih : ∀ (c : Set α), HasSubset.Subset c s → IsChain (fun x1 x2 => LE.le x1 x2)  …
        x : α
        hxs : Membership.mem s x
        c : Set α
        hcs : HasSubset.Subset c (setOf fun y => And (Membership.mem s y) (LE.le x y))
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        y : α
        hy : Membership.mem c y
        z : α
        hzs : Membership.mem s z
        hz : ∀ (z_1 : α), Membership.mem c z_1 → LE.le z_1 z
        ⊢ Exists fun ub => And (Membership.mem (setOf fun y => And (Membership.mem s y …
      -/
      exact ⟨z, ⟨hzs, (hcs hy).2.trans <| hz _ hy⟩, hz⟩
      /-
        🎉 no goals
      -/


theorem zorn_le_nonempty_Ici₀ (a : α)
    (ih : ∀ c ⊆ Ici a, IsChain (· ≤ ·) c → ∀ y ∈ c, ∃ ub, ∀ z ∈ c, z ≤ ub) (x : α) (hax : a ≤ x) :
    ∃ m, x ≤ m ∧ IsMax m := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a : α
    ih : ∀ (c : Set α), HasSubset.Subset c (Set.Ici a) → IsChain (fun x1 x2 => LE. …
    x : α
    hax : LE.le a x
    ⊢ Exists fun m => And (LE.le x m) (IsMax m)
  -/
  let ⟨m, hxm, ham, hm⟩ := zorn_le_nonempty₀ (Ici a) (fun c hca hc y hy ↦ ?_) x hax
    /-
      case refine_2
      α : Type u_1
      inst✝ : Preorder α
      a : α
      ih : ∀ (c : Set α), HasSubset.Subset c (Set.Ici a) → IsChain (fun x1 x2 => LE. …
      x : α
      hax : LE.le a x
      m : α
      hxm : LE.le x m
      ham : (fun x => Membership.mem (Set.Ici a) x) m
      hm : ∀ ⦃y : α⦄, (fun x => Membership.mem (Set.Ici a) x) y → LE.le m y → LE.le  …
      ⊢ Exists fun m => And (LE.le x m) (IsMax m)
    -/
  · exact ⟨m, hxm, fun z hmz => hm (ham.trans hmz) hmz⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      α : Type u_1
      inst✝ : Preorder α
      a : α
      ih : ∀ (c : Set α), HasSubset.Subset c (Set.Ici a) → IsChain (fun x1 x2 => LE. …
      x : α
      hax : LE.le a x
      c : Set α
      hca : HasSubset.Subset c (Set.Ici a)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      y : α
      hy : Membership.mem c y
      ⊢ Exists fun ub => And (Membership.mem (Set.Ici a) ub) (∀ (z : α), Membership. …
    -/
  · have ⟨ub, hub⟩ := ih c hca hc y hy
    /-
      case refine_1
      α : Type u_1
      inst✝ : Preorder α
      a : α
      ih : ∀ (c : Set α), HasSubset.Subset c (Set.Ici a) → IsChain (fun x1 x2 => LE. …
      x : α
      hax : LE.le a x
      c : Set α
      hca : HasSubset.Subset c (Set.Ici a)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      y : α
      hy : Membership.mem c y
      ub : α
      hub : ∀ (z : α), Membership.mem c z → LE.le z ub
      ⊢ Exists fun ub => And (Membership.mem (Set.Ici a) ub) (∀ (z : α), Membership. …
    -/
    exact ⟨ub, (hca hy).trans (hub y hy), hub⟩
    /-
      🎉 no goals
    -/


theorem zorn_subset (S : Set (Set α))
    (h : ∀ c ⊆ S, IsChain (· ⊆ ·) c → ∃ ub ∈ S, ∀ s ∈ c, s ⊆ ub) : ∃ m, Maximal (· ∈ S) m :=
  zorn_le₀ S h


theorem zorn_subset_nonempty (S : Set (Set α))
    (H : ∀ c ⊆ S, IsChain (· ⊆ ·) c → c.Nonempty → ∃ ub ∈ S, ∀ s ∈ c, s ⊆ ub) (x) (hx : x ∈ S) :
    ∃ m, x ⊆ m ∧ Maximal (· ∈ S) m :=
  zorn_le_nonempty₀ _ (fun _ cS hc y yc => H _ cS hc ⟨y, yc⟩) _ hx


theorem zorn_superset (S : Set (Set α))
    (h : ∀ c ⊆ S, IsChain (· ⊆ ·) c → ∃ lb ∈ S, ∀ s ∈ c, lb ⊆ s) : ∃ m, Minimal (· ∈ S) m :=
  (@zorn_le₀ (Set α)ᵒᵈ _ S) fun c cS hc => h c cS hc.symm


theorem zorn_superset_nonempty (S : Set (Set α))
    (H : ∀ c ⊆ S, IsChain (· ⊆ ·) c → c.Nonempty → ∃ lb ∈ S, ∀ s ∈ c, lb ⊆ s) (x) (hx : x ∈ S) :
    ∃ m, m ⊆ x ∧ Minimal (· ∈ S) m :=
  @zorn_le_nonempty₀ (Set α)ᵒᵈ _ S (fun _ cS hc y yc => H _ cS hc.symm ⟨y, yc⟩) _ hx


/-- Every chain is contained in a maximal chain. This generalizes Hausdorff's maximality principle.
-/
theorem IsChain.exists_maxChain (hc : IsChain r c) : ∃ M, @IsMaxChain _ r M ∧ c ⊆ M := by
  -- Porting note: the first three lines replace the following two lines in mathlib3.
  -- The mathlib3 `obtain` supports holes for proof obligations, this is not yet implemented in 4.
  -- obtain ⟨M, ⟨_, hM₀⟩, hM₁, hM₂⟩ :=
  --   zorn_subset_nonempty { s | c ⊆ s ∧ IsChain r s } _ c ⟨Subset.rfl, hc⟩
  /-
    α : Type u_1
    r : α → α → Prop
    c : Set α
    hc : IsChain r c
    ⊢ Exists fun M => And (IsMaxChain r M) (HasSubset.Subset c M)
  -/
  have H := zorn_subset_nonempty { s | c ⊆ s ∧ IsChain r s } ?_ c ⟨Subset.rfl, hc⟩
    /-
      case refine_2
      α : Type u_1
      r : α → α → Prop
      c : Set α
      hc : IsChain r c
      H : Exists fun m => And (HasSubset.Subset c m) (Maximal (fun x => Membership.m …
      ⊢ Exists fun M => And (IsMaxChain r M) (HasSubset.Subset c M)
    -/
  · obtain ⟨M, hcM, hM⟩ := H
    /-
      case refine_2.intro.intro
      α : Type u_1
      r : α → α → Prop
      c : Set α
      hc : IsChain r c
      M : Set α
      hcM : HasSubset.Subset c M
      hM : Maximal (fun x => Membership.mem (setOf fun s => And (HasSubset.Subset c  …
      ⊢ Exists fun M => And (IsMaxChain r M) (HasSubset.Subset c M)
    -/
    exact ⟨M, ⟨hM.prop.2, fun d hd hMd ↦ hM.eq_of_subset ⟨hcM.trans hMd, hd⟩ hMd⟩, hcM⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    α : Type u_1
    r : α → α → Prop
    c : Set α
    hc : IsChain r c
    ⊢ ∀ (c_1 : Set (Set α)), HasSubset.Subset c_1 (setOf fun s => And (HasSubset.S …
  -/
  rintro cs hcs₀ hcs₁ ⟨s, hs⟩
  refine
    ⟨⋃₀cs, ⟨fun _ ha => Set.mem_sUnion_of_mem ((hcs₀ hs).left ha) hs, ?_⟩, fun _ =>
      Set.subset_sUnion_of_mem⟩
  /-
    case refine_1.intro
    α : Type u_1
    r : α → α → Prop
    c : Set α
    hc : IsChain r c
    cs : Set (Set α)
    hcs₀ : HasSubset.Subset cs (setOf fun s => And (HasSubset.Subset c s) (IsChain …
    hcs₁ : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) cs
    s : Set α
    hs : Membership.mem cs s
    ⊢ IsChain r cs.sUnion
  -/
  rintro y ⟨sy, hsy, hysy⟩ z ⟨sz, hsz, hzsz⟩ hyz
  /-
    case refine_1.intro.intro.intro.intro.intro
    α : Type u_1
    r : α → α → Prop
    c : Set α
    hc : IsChain r c
    cs : Set (Set α)
    hcs₀ : HasSubset.Subset cs (setOf fun s => And (HasSubset.Subset c s) (IsChain …
    hcs₁ : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) cs
    s : Set α
    hs : Membership.mem cs s
    y : α
    sy : Set α
    hsy : Membership.mem cs sy
    hysy : Membership.mem sy y
    z : α
    sz : Set α
    hsz : Membership.mem cs sz
    hzsz : Membership.mem sz z
    hyz : Ne y z
    ⊢ Or (r y z) (r z y)
  -/
  obtain rfl | hsseq := eq_or_ne sy sz
    /-
      case refine_1.intro.intro.intro.intro.intro.inl
      α : Type u_1
      r : α → α → Prop
      c : Set α
      hc : IsChain r c
      cs : Set (Set α)
      hcs₀ : HasSubset.Subset cs (setOf fun s => And (HasSubset.Subset c s) (IsChain …
      hcs₁ : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) cs
      s : Set α
      hs : Membership.mem cs s
      y : α
      sy : Set α
      hsy : Membership.mem cs sy
      hysy : Membership.mem sy y
      z : α
      hyz : Ne y z
      hsz : Membership.mem cs sy
      hzsz : Membership.mem sy z
      ⊢ Or (r y z) (r z y)
    -/
  · exact (hcs₀ hsy).right hysy hzsz hyz
    /-
      🎉 no goals
    -/
  /-
    case refine_1.intro.intro.intro.intro.intro.inr
    α : Type u_1
    r : α → α → Prop
    c : Set α
    hc : IsChain r c
    cs : Set (Set α)
    hcs₀ : HasSubset.Subset cs (setOf fun s => And (HasSubset.Subset c s) (IsChain …
    hcs₁ : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) cs
    s : Set α
    hs : Membership.mem cs s
    y : α
    sy : Set α
    hsy : Membership.mem cs sy
    hysy : Membership.mem sy y
    z : α
    sz : Set α
    hsz : Membership.mem cs sz
    hzsz : Membership.mem sz z
    hyz : Ne y z
    hsseq : Ne sy sz
    ⊢ Or (r y z) (r z y)
  -/
  cases' hcs₁ hsy hsz hsseq with h h
    /-
      case refine_1.intro.intro.intro.intro.intro.inr.inl
      α : Type u_1
      r : α → α → Prop
      c : Set α
      hc : IsChain r c
      cs : Set (Set α)
      hcs₀ : HasSubset.Subset cs (setOf fun s => And (HasSubset.Subset c s) (IsChain …
      hcs₁ : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) cs
      s : Set α
      hs : Membership.mem cs s
      y : α
      sy : Set α
      hsy : Membership.mem cs sy
      hysy : Membership.mem sy y
      z : α
      sz : Set α
      hsz : Membership.mem cs sz
      hzsz : Membership.mem sz z
      hyz : Ne y z
      hsseq : Ne sy sz
      h : (fun x1 x2 => HasSubset.Subset x1 x2) sy sz
      ⊢ Or (r y z) (r z y)
    -/
  · exact (hcs₀ hsz).right (h hysy) hzsz hyz
    /-
      🎉 no goals
    -/
    /-
      case refine_1.intro.intro.intro.intro.intro.inr.inr
      α : Type u_1
      r : α → α → Prop
      c : Set α
      hc : IsChain r c
      cs : Set (Set α)
      hcs₀ : HasSubset.Subset cs (setOf fun s => And (HasSubset.Subset c s) (IsChain …
      hcs₁ : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) cs
      s : Set α
      hs : Membership.mem cs s
      y : α
      sy : Set α
      hsy : Membership.mem cs sy
      hysy : Membership.mem sy y
      z : α
      sz : Set α
      hsz : Membership.mem cs sz
      hzsz : Membership.mem sz z
      hyz : Ne y z
      hsseq : Ne sy sz
      h : (fun x1 x2 => HasSubset.Subset x1 x2) sz sy
      ⊢ Or (r y z) (r z y)
    -/
  · exact (hcs₀ hsy).right hysy (h hzsz) hyz
    /-
      🎉 no goals
    -/


lemma _root_.IsChain.exists_subset_flag (hc : IsChain (· ≤ ·) c) : ∃ s : Flag α, c ⊆ s :=
  let ⟨s, hs, hcs⟩ := hc.exists_maxChain; ⟨ofIsMaxChain s hs, hcs⟩


lemma exists_mem (a : α) : ∃ s : Flag α, a ∈ s :=
  let ⟨s, hs⟩ := Set.subsingleton_singleton (a := a).isChain.exists_subset_flag
  ⟨s, hs rfl⟩


lemma exists_mem_mem (hab : a ≤ b) : ∃ s : Flag α, a ∈ s ∧ b ∈ s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    a b : α
    hab : LE.le a b
    ⊢ Exists fun s => And (Membership.mem s a) (Membership.mem s b)
  -/
  simpa [Set.insert_subset_iff] using (IsChain.pair hab).exists_subset_flag
  /-
    🎉 no goals
  -/


instance : Nonempty (Flag α) := ⟨.ofIsMaxChain _ maxChain_spec⟩


