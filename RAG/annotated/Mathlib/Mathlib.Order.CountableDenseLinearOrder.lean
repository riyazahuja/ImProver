/-- Suppose `α` is a nonempty dense linear order without endpoints, and
    suppose `lo`, `hi`, are finite subsets with all of `lo` strictly
    before `hi`. Then there is an element of `α` strictly between `lo`
    and `hi`. -/
theorem exists_between_finsets [DenselyOrdered α] [NoMinOrder α]
    [NoMaxOrder α] [nonem : Nonempty α] (lo hi : Finset α) (lo_lt_hi : ∀ x ∈ lo, ∀ y ∈ hi, x < y) :
    ∃ m : α, (∀ x ∈ lo, x < m) ∧ ∀ y ∈ hi, m < y :=
  if nlo : lo.Nonempty then
    if nhi : hi.Nonempty then
      -- both sets are nonempty, use `DenselyOrdered`
        Exists.elim
        (exists_between (lo_lt_hi _ (Finset.max'_mem _ nlo) _ (Finset.min'_mem _ nhi))) fun m hm ↦
        ⟨m, fun x hx ↦ lt_of_le_of_lt (Finset.le_max' lo x hx) hm.1, fun y hy ↦
          lt_of_lt_of_le hm.2 (Finset.min'_le hi y hy)⟩
    else-- upper set is empty, use `NoMaxOrder`
        Exists.elim
        (exists_gt (Finset.max' lo nlo)) fun m hm ↦
        ⟨m, fun x hx ↦ lt_of_le_of_lt (Finset.le_max' lo x hx) hm, fun y hy ↦ (nhi ⟨y, hy⟩).elim⟩
  else
    if nhi : hi.Nonempty then
      -- lower set is empty, use `NoMinOrder`
        Exists.elim
        (exists_lt (Finset.min' hi nhi)) fun m hm ↦
        ⟨m, fun x hx ↦ (nlo ⟨x, hx⟩).elim, fun y hy ↦ lt_of_lt_of_le hm (Finset.min'_le hi y hy)⟩
    else -- both sets are empty, use `Nonempty`
          nonem.elim
        fun m ↦ ⟨m, fun x hx ↦ (nlo ⟨x, hx⟩).elim, fun y hy ↦ (nhi ⟨y, hy⟩).elim⟩


lemma exists_orderEmbedding_insert [DenselyOrdered β] [NoMinOrder β] [NoMaxOrder β]
    [nonem : Nonempty β]  (S : Finset α) (f : S ↪o β) (a : α) :
    ∃ (g : (insert a S : Finset α) ↪o β),
      g ∘ (Set.inclusion ((S.subset_insert a) : ↑S ⊆ ↑(insert a S))) = f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : DenselyOrdered β
    inst✝¹ : NoMinOrder β
    inst✝ : NoMaxOrder β
    nonem : Nonempty β
    S : Finset α
    f : OrderEmbedding (Subtype fun x => Membership.mem S x) β
    a : α
    ⊢ Exists fun g => Eq (Function.comp (⇑g) (Set.inclusion ⋯)) ⇑f
  -/
  let Slt := (S.attach.filter (fun (x : S) => x < a)).image f
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : DenselyOrdered β
    inst✝¹ : NoMinOrder β
    inst✝ : NoMaxOrder β
    nonem : Nonempty β
    S : Finset α
    f : OrderEmbedding (Subtype fun x => Membership.mem S x) β
    a : α
    Slt : Finset β := Finset.image (⇑f) (Finset.filter (fun x => LT.lt (↑x) a) S.a …
    ⊢ Exists fun g => Eq (Function.comp (⇑g) (Set.inclusion ⋯)) ⇑f
  -/
  let Sgt := (S.attach.filter (fun (x : S) => a < x)).image f
  obtain ⟨b, hb, hb'⟩ := Order.exists_between_finsets Slt Sgt (fun x hx y hy => by
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_attach, true_and, Subtype.exists,
      exists_and_left, Slt, Sgt] at hx hy
    obtain ⟨_, hx, _, rfl⟩ := hx
    obtain ⟨_, hy, _, rfl⟩ := hy
    exact f.strictMono (hx.trans hy))
  refine ⟨OrderEmbedding.ofStrictMono
    (fun (x : (insert a S : Finset α)) => if hx : x.1 ∈ S then f ⟨x.1, hx⟩ else b) ?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrder α
      inst✝³ : LinearOrder β
      inst✝² : DenselyOrdered β
      inst✝¹ : NoMinOrder β
      inst✝ : NoMaxOrder β
      nonem : Nonempty β
      S : Finset α
      f : OrderEmbedding (Subtype fun x => Membership.mem S x) β
      a : α
      Slt : Finset β := Finset.image (⇑f) (Finset.filter (fun x => LT.lt (↑x) a) S.a …
      Sgt : Finset β := Finset.image (⇑f) (Finset.filter (fun x => LT.lt a ↑x) S.att …
      b : β
      hb : ∀ (x : β), Membership.mem Slt x → LT.lt x b
      hb' : ∀ (y : β), Membership.mem Sgt y → LT.lt b y
      ⊢ StrictMono fun x => dite (Membership.mem S ↑x) (fun hx => f ⟨↑x, hx⟩) fun hx …
    -/
  · rintro ⟨x, hx⟩ ⟨y, hy⟩ hxy
    if hxS : x ∈ S
    then if hyS : y ∈ S
      then simpa only [hxS, hyS, ↓reduceDIte, OrderEmbedding.lt_iff_lt, Subtype.mk_lt_mk]
      else
        obtain rfl := Finset.eq_of_mem_insert_of_not_mem hy hyS
        simp only [hxS, hyS, ↓reduceDIte]
        exact hb _ (Finset.mem_image_of_mem _ (Finset.mem_filter.2 ⟨Finset.mem_attach _ _, hxy⟩))
    else
      obtain rfl := Finset.eq_of_mem_insert_of_not_mem hx hxS
      if hyS : y ∈ S
      then
        simp only [hxS, hyS, ↓reduceDIte]
        exact hb' _ (Finset.mem_image_of_mem _ (Finset.mem_filter.2 ⟨Finset.mem_attach _ _, hxy⟩))
      else simp only [Finset.eq_of_mem_insert_of_not_mem hy hyS, lt_self_iff_false] at hxy
    /-
      case intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrder α
      inst✝³ : LinearOrder β
      inst✝² : DenselyOrdered β
      inst✝¹ : NoMinOrder β
      inst✝ : NoMaxOrder β
      nonem : Nonempty β
      S : Finset α
      f : OrderEmbedding (Subtype fun x => Membership.mem S x) β
      a : α
      Slt : Finset β := Finset.image (⇑f) (Finset.filter (fun x => LT.lt (↑x) a) S.a …
      Sgt : Finset β := Finset.image (⇑f) (Finset.filter (fun x => LT.lt a ↑x) S.att …
      b : β
      hb : ∀ (x : β), Membership.mem Slt x → LT.lt x b
      hb' : ∀ (y : β), Membership.mem Sgt y → LT.lt b y
      ⊢ Eq (Function.comp (⇑(OrderEmbedding.ofStrictMono (fun x => dite (Membership. …
    -/
  · ext x
    simp only [Finset.coe_sort_coe, OrderEmbedding.coe_ofStrictMono, Finset.insert_val,
      Function.comp_apply, Finset.coe_mem, ↓reduceDIte, Subtype.coe_eta]


/-- The type of partial order isomorphisms between `α` and `β` defined on finite subsets.
    A partial order isomorphism is encoded as a finite subset of `α × β`, consisting
    of pairs which should be identified. -/
def PartialIso : Type _ :=
  { f : Finset (α × β) //
    ∀ p ∈ f, ∀ q ∈ f,
      cmp (Prod.fst p) (Prod.fst q) = cmp (Prod.snd p) (Prod.snd q) }


instance : Inhabited (PartialIso α β) := ⟨⟨∅, fun _p h _q ↦ (Finset.not_mem_empty _ h).elim⟩⟩


instance : Preorder (PartialIso α β) := Subtype.preorder _


/-- For each `a`, we can find a `b` in the codomain, such that `a`'s relation to
the domain of `f` is `b`'s relation to the image of `f`.

Thus, if `a` is not already in `f`, then we can extend `f` by sending `a` to `b`.
-/
theorem exists_across [DenselyOrdered β] [NoMinOrder β] [NoMaxOrder β] [Nonempty β]
    (f : PartialIso α β) (a : α) :
    ∃ b : β, ∀ p ∈ f.val, cmp (Prod.fst p) a = cmp (Prod.snd p) b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : LinearOrder β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    f : Order.PartialIso α β
    a : α
    ⊢ Exists fun b => ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cm …
  -/
  by_cases h : ∃ b, (a, b) ∈ f.val
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      f : Order.PartialIso α β
      a : α
      h : Exists fun b => Membership.mem ↑f { fst := a, snd := b }
      ⊢ Exists fun b => ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cm …
    -/
  · cases' h with b hb
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      f : Order.PartialIso α β
      a : α
      b : β
      hb : Membership.mem ↑f { fst := a, snd := b }
      ⊢ Exists fun b => ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cm …
    -/
    exact ⟨b, fun p hp ↦ f.prop _ hp _ hb⟩
    /-
      🎉 no goals
    -/
  have :
    ∀ x ∈ (f.val.filter fun p : α × β ↦ p.fst < a).image Prod.snd,
      ∀ y ∈ (f.val.filter fun p : α × β ↦ a < p.fst).image Prod.snd, x < y := by
    intro x hx y hy
    rw [Finset.mem_image] at hx hy
    rcases hx with ⟨p, hp1, rfl⟩
    rcases hy with ⟨q, hq1, rfl⟩
    rw [Finset.mem_filter] at hp1 hq1
    rw [← lt_iff_lt_of_cmp_eq_cmp (f.prop _ hp1.1 _ hq1.1)]
    exact lt_trans hp1.right hq1.right
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : LinearOrder β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    f : Order.PartialIso α β
    a : α
    h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
    this : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun p  …
    ⊢ Exists fun b => ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cm …
  -/
  cases' exists_between_finsets _ _ this with b hb
  /-
    case neg.intro
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : LinearOrder β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    f : Order.PartialIso α β
    a : α
    h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
    this : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun p  …
    b : β
    hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
    ⊢ Exists fun b => ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cm …
  -/
  use b
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : LinearOrder β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    f : Order.PartialIso α β
    a : α
    h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
    this : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun p  …
    b : β
    hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
    ⊢ ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
  -/
  rintro ⟨p1, p2⟩ hp
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : LinearOrder β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    f : Order.PartialIso α β
    a : α
    h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
    this : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun p  …
    b : β
    hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
    p1 : α
    p2 : β
    hp : Membership.mem ↑f { fst := p1, snd := p2 }
    ⊢ Eq (cmp { fst := p1, snd := p2 }.1 a) (cmp { fst := p1, snd := p2 }.2 b)
  -/
  have : p1 ≠ a := fun he ↦ h ⟨p2, he ▸ hp⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : LinearOrder β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    f : Order.PartialIso α β
    a : α
    h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
    this✝ : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun p …
    b : β
    hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
    p1 : α
    p2 : β
    hp : Membership.mem ↑f { fst := p1, snd := p2 }
    this : Ne p1 a
    ⊢ Eq (cmp { fst := p1, snd := p2 }.1 a) (cmp { fst := p1, snd := p2 }.2 b)
  -/
  cases' lt_or_gt_of_ne this with hl hr
  · have : p1 < a ∧ p2 < b :=
      ⟨hl, hb.1 _ (Finset.mem_image.mpr ⟨(p1, p2), Finset.mem_filter.mpr ⟨hp, hl⟩, rfl⟩)⟩
    /-
      case h.mk.inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      f : Order.PartialIso α β
      a : α
      h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
      this✝¹ : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun  …
      b : β
      hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
      p1 : α
      p2 : β
      hp : Membership.mem ↑f { fst := p1, snd := p2 }
      this✝ : Ne p1 a
      hl : LT.lt p1 a
      this : And (LT.lt p1 a) (LT.lt p2 b)
      ⊢ Eq (cmp { fst := p1, snd := p2 }.1 a) (cmp { fst := p1, snd := p2 }.2 b)
    -/
    rw [← cmp_eq_lt_iff, ← cmp_eq_lt_iff] at this
    /-
      case h.mk.inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      f : Order.PartialIso α β
      a : α
      h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
      this✝¹ : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun  …
      b : β
      hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
      p1 : α
      p2 : β
      hp : Membership.mem ↑f { fst := p1, snd := p2 }
      this✝ : Ne p1 a
      hl : LT.lt p1 a
      this : And (Eq (cmp p1 a) Ordering.lt) (Eq (cmp p2 b) Ordering.lt)
      ⊢ Eq (cmp { fst := p1, snd := p2 }.1 a) (cmp { fst := p1, snd := p2 }.2 b)
    -/
    exact this.1.trans this.2.symm
    /-
      🎉 no goals
    -/
  · have : a < p1 ∧ b < p2 :=
      ⟨hr, hb.2 _ (Finset.mem_image.mpr ⟨(p1, p2), Finset.mem_filter.mpr ⟨hp, hr⟩, rfl⟩)⟩
    /-
      case h.mk.inr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      f : Order.PartialIso α β
      a : α
      h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
      this✝¹ : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun  …
      b : β
      hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
      p1 : α
      p2 : β
      hp : Membership.mem ↑f { fst := p1, snd := p2 }
      this✝ : Ne p1 a
      hr : GT.gt p1 a
      this : And (LT.lt a p1) (LT.lt b p2)
      ⊢ Eq (cmp { fst := p1, snd := p2 }.1 a) (cmp { fst := p1, snd := p2 }.2 b)
    -/
    rw [← cmp_eq_gt_iff, ← cmp_eq_gt_iff] at this
    /-
      case h.mk.inr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      f : Order.PartialIso α β
      a : α
      h : Not (Exists fun b => Membership.mem ↑f { fst := a, snd := b })
      this✝¹ : ∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun  …
      b : β
      hb : And (∀ (x : β), Membership.mem (Finset.image Prod.snd (Finset.filter (fun …
      p1 : α
      p2 : β
      hp : Membership.mem ↑f { fst := p1, snd := p2 }
      this✝ : Ne p1 a
      hr : GT.gt p1 a
      this : And (Eq (cmp p1 a) Ordering.gt) (Eq (cmp p2 b) Ordering.gt)
      ⊢ Eq (cmp { fst := p1, snd := p2 }.1 a) (cmp { fst := p1, snd := p2 }.2 b)
    -/
    exact this.1.trans this.2.symm
    /-
      🎉 no goals
    -/


/-- A partial isomorphism between `α` and `β` is also a partial isomorphism between `β` and `α`. -/
protected def comm : PartialIso α β → PartialIso β α :=
  Subtype.map (Finset.image (Equiv.prodComm _ _)) fun f hf p hp q hq ↦
    Eq.symm <|
      hf ((Equiv.prodComm α β).symm p)
        (by
          /-
            α : Type u_1
            β : Type u_2
            inst✝¹ : LinearOrder α
            inst✝ : LinearOrder β
            f : Finset (Prod α β)
            hf : ∀ (p : Prod α β), Membership.mem f p → ∀ (q : Prod α β), Membership.mem f …
            p : Prod β α
            hp : Membership.mem (Finset.image (⇑(Equiv.prodComm α β)) f) p
            q : Prod β α
            hq : Membership.mem (Finset.image (⇑(Equiv.prodComm α β)) f) q
            ⊢ Membership.mem f ((Equiv.prodComm α β).symm p)
          -/
          rw [← Finset.mem_coe, Finset.coe_image, Equiv.image_eq_preimage] at hp
          /-
            α : Type u_1
            β : Type u_2
            inst✝¹ : LinearOrder α
            inst✝ : LinearOrder β
            f : Finset (Prod α β)
            hf : ∀ (p : Prod α β), Membership.mem f p → ∀ (q : Prod α β), Membership.mem f …
            p : Prod β α
            hp : Membership.mem (Set.preimage ⇑(Equiv.prodComm α β).symm ↑f) p
            q : Prod β α
            hq : Membership.mem (Finset.image (⇑(Equiv.prodComm α β)) f) q
            ⊢ Membership.mem f ((Equiv.prodComm α β).symm p)
          -/
          rwa [← Finset.mem_coe])
          /-
            🎉 no goals
          -/
        ((Equiv.prodComm α β).symm q)
        (by
          /-
            α : Type u_1
            β : Type u_2
            inst✝¹ : LinearOrder α
            inst✝ : LinearOrder β
            f : Finset (Prod α β)
            hf : ∀ (p : Prod α β), Membership.mem f p → ∀ (q : Prod α β), Membership.mem f …
            p : Prod β α
            hp : Membership.mem (Finset.image (⇑(Equiv.prodComm α β)) f) p
            q : Prod β α
            hq : Membership.mem (Finset.image (⇑(Equiv.prodComm α β)) f) q
            ⊢ Membership.mem f ((Equiv.prodComm α β).symm q)
          -/
          rw [← Finset.mem_coe, Finset.coe_image, Equiv.image_eq_preimage] at hq
          /-
            α : Type u_1
            β : Type u_2
            inst✝¹ : LinearOrder α
            inst✝ : LinearOrder β
            f : Finset (Prod α β)
            hf : ∀ (p : Prod α β), Membership.mem f p → ∀ (q : Prod α β), Membership.mem f …
            p : Prod β α
            hp : Membership.mem (Finset.image (⇑(Equiv.prodComm α β)) f) p
            q : Prod β α
            hq : Membership.mem (Set.preimage ⇑(Equiv.prodComm α β).symm ↑f) q
            ⊢ Membership.mem f ((Equiv.prodComm α β).symm q)
          -/
          rwa [← Finset.mem_coe])
          /-
            🎉 no goals
          -/


/-- The set of partial isomorphisms defined at `a : α`, together with a proof that any
    partial isomorphism can be extended to one defined at `a`. -/
def definedAtLeft [DenselyOrdered β] [NoMinOrder β] [NoMaxOrder β] [Nonempty β] (a : α) :
    Cofinal (PartialIso α β) where
  carrier := {f | ∃ b : β, (a, b) ∈ f.val}
  isCofinal f := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      a : α
      f : Order.PartialIso α β
      ⊢ Exists fun y => And (Membership.mem (setOf fun f => Exists fun b => Membersh …
    -/
    cases' exists_across f a with b a_b
    refine
      ⟨⟨insert (a, b) f.val, fun p hp q hq ↦ ?_⟩, ⟨b, Finset.mem_insert_self _ _⟩,
        Finset.subset_insert _ _⟩
    /-
      case intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      a : α
      f : Order.PartialIso α β
      b : β
      a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
      p : Prod α β
      hp : Membership.mem (Insert.insert { fst := a, snd := b } ↑f) p
      q : Prod α β
      hq : Membership.mem (Insert.insert { fst := a, snd := b } ↑f) q
      ⊢ Eq (cmp p.1 q.1) (cmp p.2 q.2)
    -/
    rw [Finset.mem_insert] at hp hq
    /-
      case intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered β
      inst✝² : NoMinOrder β
      inst✝¹ : NoMaxOrder β
      inst✝ : Nonempty β
      a : α
      f : Order.PartialIso α β
      b : β
      a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
      p : Prod α β
      hp : Or (Eq p { fst := a, snd := b }) (Membership.mem (↑f) p)
      q : Prod α β
      hq : Or (Eq q { fst := a, snd := b }) (Membership.mem (↑f) q)
      ⊢ Eq (cmp p.1 q.1) (cmp p.2 q.2)
    -/
    rcases hp with (rfl | pf) <;> rcases hq with (rfl | qf)
      /-
        case intro.inl.inl
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered β
        inst✝² : NoMinOrder β
        inst✝¹ : NoMaxOrder β
        inst✝ : Nonempty β
        a : α
        f : Order.PartialIso α β
        b : β
        a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
        ⊢ Eq (cmp { fst := a, snd := b }.1 { fst := a, snd := b }.1) (cmp { fst := a,  …
      -/
    · simp only [cmp_self_eq_eq]
      /-
        🎉 no goals
      -/
      /-
        case intro.inl.inr
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered β
        inst✝² : NoMinOrder β
        inst✝¹ : NoMaxOrder β
        inst✝ : Nonempty β
        a : α
        f : Order.PartialIso α β
        b : β
        a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
        q : Prod α β
        qf : Membership.mem (↑f) q
        ⊢ Eq (cmp { fst := a, snd := b }.1 q.1) (cmp { fst := a, snd := b }.2 q.2)
      -/
    · rw [cmp_eq_cmp_symm]
      /-
        case intro.inl.inr
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered β
        inst✝² : NoMinOrder β
        inst✝¹ : NoMaxOrder β
        inst✝ : Nonempty β
        a : α
        f : Order.PartialIso α β
        b : β
        a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
        q : Prod α β
        qf : Membership.mem (↑f) q
        ⊢ Eq (cmp q.1 { fst := a, snd := b }.1) (cmp q.2 { fst := a, snd := b }.2)
      -/
      exact a_b _ qf
      /-
        🎉 no goals
      -/
      /-
        case intro.inr.inl
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered β
        inst✝² : NoMinOrder β
        inst✝¹ : NoMaxOrder β
        inst✝ : Nonempty β
        a : α
        f : Order.PartialIso α β
        b : β
        a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
        p : Prod α β
        pf : Membership.mem (↑f) p
        ⊢ Eq (cmp p.1 { fst := a, snd := b }.1) (cmp p.2 { fst := a, snd := b }.2)
      -/
    · exact a_b _ pf
      /-
        🎉 no goals
      -/
      /-
        case intro.inr.inr
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered β
        inst✝² : NoMinOrder β
        inst✝¹ : NoMaxOrder β
        inst✝ : Nonempty β
        a : α
        f : Order.PartialIso α β
        b : β
        a_b : ∀ (p : Prod α β), Membership.mem (↑f) p → Eq (cmp p.1 a) (cmp p.2 b)
        p q : Prod α β
        pf : Membership.mem (↑f) p
        qf : Membership.mem (↑f) q
        ⊢ Eq (cmp p.1 q.1) (cmp p.2 q.2)
      -/
    · exact f.prop _ pf _ qf
      /-
        🎉 no goals
      -/


/-- The set of partial isomorphisms defined at `b : β`, together with a proof that any
    partial isomorphism can be extended to include `b`. We prove this by symmetry. -/
def definedAtRight [DenselyOrdered α] [NoMinOrder α] [NoMaxOrder α] [Nonempty α] (b : β) :
    Cofinal (PartialIso α β) where
  carrier := {f | ∃ a, (a, b) ∈ f.val}
  isCofinal f := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered α
      inst✝² : NoMinOrder α
      inst✝¹ : NoMaxOrder α
      inst✝ : Nonempty α
      b : β
      f : Order.PartialIso α β
      ⊢ Exists fun y => And (Membership.mem (setOf fun f => Exists fun a => Membersh …
    -/
    rcases (definedAtLeft α b).isCofinal f.comm with ⟨f', ⟨a, ha⟩, hl⟩
    /-
      case intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : LinearOrder β
      inst✝³ : DenselyOrdered α
      inst✝² : NoMinOrder α
      inst✝¹ : NoMaxOrder α
      inst✝ : Nonempty α
      b : β
      f : Order.PartialIso α β
      f' : Order.PartialIso β α
      hl : LE.le f.comm f'
      a : α
      ha : Membership.mem ↑f' { fst := b, snd := a }
      ⊢ Exists fun y => And (Membership.mem (setOf fun f => Exists fun a => Membersh …
    -/
    refine ⟨f'.comm, ⟨a, ?_⟩, ?_⟩
      /-
        case intro.intro.intro.refine_1
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered α
        inst✝² : NoMinOrder α
        inst✝¹ : NoMaxOrder α
        inst✝ : Nonempty α
        b : β
        f : Order.PartialIso α β
        f' : Order.PartialIso β α
        hl : LE.le f.comm f'
        a : α
        ha : Membership.mem ↑f' { fst := b, snd := a }
        ⊢ Membership.mem ↑f'.comm { fst := a, snd := b }
      -/
    · change (a, b) ∈ f'.val.image _
      /-
        case intro.intro.intro.refine_1
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered α
        inst✝² : NoMinOrder α
        inst✝¹ : NoMaxOrder α
        inst✝ : Nonempty α
        b : β
        f : Order.PartialIso α β
        f' : Order.PartialIso β α
        hl : LE.le f.comm f'
        a : α
        ha : Membership.mem ↑f' { fst := b, snd := a }
        ⊢ Membership.mem (Finset.image ⇑(Equiv.prodComm β α) ↑f') { fst := a, snd := b }
      -/
      rwa [← Finset.mem_coe, Finset.coe_image, Equiv.image_eq_preimage]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_2
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : LinearOrder β
        inst✝³ : DenselyOrdered α
        inst✝² : NoMinOrder α
        inst✝¹ : NoMaxOrder α
        inst✝ : Nonempty α
        b : β
        f : Order.PartialIso α β
        f' : Order.PartialIso β α
        hl : LE.le f.comm f'
        a : α
        ha : Membership.mem ↑f' { fst := b, snd := a }
        ⊢ LE.le f f'.comm
      -/
    · change _ ⊆ f'.val.image _
      rwa [← Finset.coe_subset, Finset.coe_image, ← Equiv.symm_image_subset, ← Finset.coe_image,
        Finset.coe_subset]


/-- Given an ideal which intersects `definedAtLeft β a`, pick `b : β` such that
    some partial function in the ideal maps `a` to `b`. -/
def funOfIdeal [DenselyOrdered β] [NoMinOrder β] [NoMaxOrder β] [Nonempty β] (a : α)
    (I : Ideal (PartialIso α β)) :
    (∃ f, f ∈ definedAtLeft β a ∧ f ∈ I) → { b // ∃ f ∈ I, (a, b) ∈ Subtype.val f } :=
  Classical.indefiniteDescription _ ∘ fun ⟨f, ⟨b, hb⟩, hf⟩ ↦ ⟨b, f, hf, hb⟩


/-- Given an ideal which intersects `definedAtRight α b`, pick `a : α` such that
    some partial function in the ideal maps `a` to `b`. -/
def invOfIdeal [DenselyOrdered α] [NoMinOrder α] [NoMaxOrder α] [Nonempty α] (b : β)
    (I : Ideal (PartialIso α β)) :
    (∃ f, f ∈ definedAtRight α b ∧ f ∈ I) → { a // ∃ f ∈ I, (a, b) ∈ Subtype.val f } :=
  Classical.indefiniteDescription _ ∘ fun ⟨f, ⟨a, ha⟩, hf⟩ ↦ ⟨a, f, hf, ha⟩


/-- Any countable linear order embeds in any nontrivial dense linear order. -/
theorem embedding_from_countable_to_dense [Countable α] [DenselyOrdered β] [Nontrivial β] :
    Nonempty (α ↪o β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  cases nonempty_encodable α
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  rcases exists_pair_lt β with ⟨x, y, hxy⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  cases' exists_between hxy with a ha
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    a : β
    ha : And (LT.lt x a) (LT.lt a y)
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  haveI : Nonempty (Set.Ioo x y) := ⟨⟨a, ha⟩⟩
  let our_ideal : Ideal (PartialIso α _) :=
    idealOfCofinals default (definedAtLeft (Set.Ioo x y))
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    a : β
    ha : And (LT.lt x a) (LT.lt a y)
    this : Nonempty ↑(Set.Ioo x y)
    our_ideal : Order.Ideal (Order.PartialIso α ↑(Set.Ioo x y)) := Order.idealOfCo …
    ⊢ Nonempty (OrderEmbedding α β)
  -/
  let F a := funOfIdeal a our_ideal (cofinal_meets_idealOfCofinals _ _ a)
  refine
    ⟨RelEmbedding.trans (OrderEmbedding.ofStrictMono (fun a ↦ (F a).val) fun a₁ a₂ ↦ ?_)
        (OrderEmbedding.subtype _)⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    a : β
    ha : And (LT.lt x a) (LT.lt a y)
    this : Nonempty ↑(Set.Ioo x y)
    our_ideal : Order.Ideal (Order.PartialIso α ↑(Set.Ioo x y)) := Order.idealOfCo …
    F : (a : α) → Subtype fun b => Exists fun f => And (Membership.mem our_ideal f …
    a₁ a₂ : α
    ⊢ LT.lt a₁ a₂ → LT.lt ((fun a => ↑(F a)) a₁) ((fun a => ↑(F a)) a₂)
  -/
  rcases (F a₁).prop with ⟨f, hf, ha₁⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    a : β
    ha : And (LT.lt x a) (LT.lt a y)
    this : Nonempty ↑(Set.Ioo x y)
    our_ideal : Order.Ideal (Order.PartialIso α ↑(Set.Ioo x y)) := Order.idealOfCo …
    F : (a : α) → Subtype fun b => Exists fun f => And (Membership.mem our_ideal f …
    a₁ a₂ : α
    f : Order.PartialIso α ↑(Set.Ioo x y)
    hf : Membership.mem our_ideal f
    ha₁ : Membership.mem ↑f { fst := a₁, snd := ↑(F a₁) }
    ⊢ LT.lt a₁ a₂ → LT.lt ((fun a => ↑(F a)) a₁) ((fun a => ↑(F a)) a₂)
  -/
  rcases (F a₂).prop with ⟨g, hg, ha₂⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    a : β
    ha : And (LT.lt x a) (LT.lt a y)
    this : Nonempty ↑(Set.Ioo x y)
    our_ideal : Order.Ideal (Order.PartialIso α ↑(Set.Ioo x y)) := Order.idealOfCo …
    F : (a : α) → Subtype fun b => Exists fun f => And (Membership.mem our_ideal f …
    a₁ a₂ : α
    f : Order.PartialIso α ↑(Set.Ioo x y)
    hf : Membership.mem our_ideal f
    ha₁ : Membership.mem ↑f { fst := a₁, snd := ↑(F a₁) }
    g : Order.PartialIso α ↑(Set.Ioo x y)
    hg : Membership.mem our_ideal g
    ha₂ : Membership.mem ↑g { fst := a₂, snd := ↑(F a₂) }
    ⊢ LT.lt a₁ a₂ → LT.lt ((fun a => ↑(F a)) a₁) ((fun a => ↑(F a)) a₂)
  -/
  rcases our_ideal.directed _ hf _ hg with ⟨m, _hm, fm, gm⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrder α
    inst✝³ : LinearOrder β
    inst✝² : Countable α
    inst✝¹ : DenselyOrdered β
    inst✝ : Nontrivial β
    val✝ : Encodable α
    x y : β
    hxy : LT.lt x y
    a : β
    ha : And (LT.lt x a) (LT.lt a y)
    this : Nonempty ↑(Set.Ioo x y)
    our_ideal : Order.Ideal (Order.PartialIso α ↑(Set.Ioo x y)) := Order.idealOfCo …
    F : (a : α) → Subtype fun b => Exists fun f => And (Membership.mem our_ideal f …
    a₁ a₂ : α
    f : Order.PartialIso α ↑(Set.Ioo x y)
    hf : Membership.mem our_ideal f
    ha₁ : Membership.mem ↑f { fst := a₁, snd := ↑(F a₁) }
    g : Order.PartialIso α ↑(Set.Ioo x y)
    hg : Membership.mem our_ideal g
    ha₂ : Membership.mem ↑g { fst := a₂, snd := ↑(F a₂) }
    m : Order.PartialIso α ↑(Set.Ioo x y)
    _hm : Membership.mem (↑our_ideal) m
    fm : LE.le f m
    gm : LE.le g m
    ⊢ LT.lt a₁ a₂ → LT.lt ((fun a => ↑(F a)) a₁) ((fun a => ↑(F a)) a₂)
  -/
  exact (lt_iff_lt_of_cmp_eq_cmp <| m.prop (a₁, _) (fm ha₁) (a₂, _) (gm ha₂)).mp
  /-
    🎉 no goals
  -/


/-- Any two countable dense, nonempty linear orders without endpoints are order isomorphic. -/
theorem iso_of_countable_dense [Countable α] [DenselyOrdered α] [NoMinOrder α] [NoMaxOrder α]
    [Nonempty α] [Countable β] [DenselyOrdered β] [NoMinOrder β] [NoMaxOrder β] [Nonempty β] :
    Nonempty (α ≃o β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹¹ : LinearOrder α
    inst✝¹⁰ : LinearOrder β
    inst✝⁹ : Countable α
    inst✝⁸ : DenselyOrdered α
    inst✝⁷ : NoMinOrder α
    inst✝⁶ : NoMaxOrder α
    inst✝⁵ : Nonempty α
    inst✝⁴ : Countable β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    ⊢ Nonempty (OrderIso α β)
  -/
  cases nonempty_encodable α
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹¹ : LinearOrder α
    inst✝¹⁰ : LinearOrder β
    inst✝⁹ : Countable α
    inst✝⁸ : DenselyOrdered α
    inst✝⁷ : NoMinOrder α
    inst✝⁶ : NoMaxOrder α
    inst✝⁵ : Nonempty α
    inst✝⁴ : Countable β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    val✝ : Encodable α
    ⊢ Nonempty (OrderIso α β)
  -/
  cases nonempty_encodable β
  let to_cofinal : α ⊕ β → Cofinal (PartialIso α β) := fun p ↦
    Sum.recOn p (definedAtLeft β) (definedAtRight α)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹¹ : LinearOrder α
    inst✝¹⁰ : LinearOrder β
    inst✝⁹ : Countable α
    inst✝⁸ : DenselyOrdered α
    inst✝⁷ : NoMinOrder α
    inst✝⁶ : NoMaxOrder α
    inst✝⁵ : Nonempty α
    inst✝⁴ : Countable β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    val✝¹ : Encodable α
    val✝ : Encodable β
    to_cofinal : Sum α β → Order.Cofinal (Order.PartialIso α β) := fun p => Sum.re …
    ⊢ Nonempty (OrderIso α β)
  -/
  let our_ideal : Ideal (PartialIso α β) := idealOfCofinals default to_cofinal
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹¹ : LinearOrder α
    inst✝¹⁰ : LinearOrder β
    inst✝⁹ : Countable α
    inst✝⁸ : DenselyOrdered α
    inst✝⁷ : NoMinOrder α
    inst✝⁶ : NoMaxOrder α
    inst✝⁵ : Nonempty α
    inst✝⁴ : Countable β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    val✝¹ : Encodable α
    val✝ : Encodable β
    to_cofinal : Sum α β → Order.Cofinal (Order.PartialIso α β) := fun p => Sum.re …
    our_ideal : Order.Ideal (Order.PartialIso α β) := Order.idealOfCofinals Inhabi …
    ⊢ Nonempty (OrderIso α β)
  -/
  let F a := funOfIdeal a our_ideal (cofinal_meets_idealOfCofinals _ to_cofinal (Sum.inl a))
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹¹ : LinearOrder α
    inst✝¹⁰ : LinearOrder β
    inst✝⁹ : Countable α
    inst✝⁸ : DenselyOrdered α
    inst✝⁷ : NoMinOrder α
    inst✝⁶ : NoMaxOrder α
    inst✝⁵ : Nonempty α
    inst✝⁴ : Countable β
    inst✝³ : DenselyOrdered β
    inst✝² : NoMinOrder β
    inst✝¹ : NoMaxOrder β
    inst✝ : Nonempty β
    val✝¹ : Encodable α
    val✝ : Encodable β
    to_cofinal : Sum α β → Order.Cofinal (Order.PartialIso α β) := fun p => Sum.re …
    our_ideal : Order.Ideal (Order.PartialIso α β) := Order.idealOfCofinals Inhabi …
    F : (a : α) → Subtype fun b => Exists fun f => And (Membership.mem our_ideal f …
    ⊢ Nonempty (OrderIso α β)
  -/
  let G b := invOfIdeal b our_ideal (cofinal_meets_idealOfCofinals _ to_cofinal (Sum.inr b))
  exact ⟨OrderIso.ofCmpEqCmp (fun a ↦ (F a).val) (fun b ↦ (G b).val) fun a b ↦ by
      rcases (F a).prop with ⟨f, hf, ha⟩
      rcases (G b).prop with ⟨g, hg, hb⟩
      rcases our_ideal.directed _ hf _ hg with ⟨m, _, fm, gm⟩
      exact m.prop (a, _) (fm ha) (_, b) (gm hb)⟩


