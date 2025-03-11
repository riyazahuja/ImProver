protected theorem isAsymm (h : WellFounded r) : IsAsymm α r := ⟨h.asymmetric⟩


protected theorem isIrrefl (h : WellFounded r) : IsIrrefl α r := @IsAsymm.isIrrefl α r h.isAsymm


instance [WellFoundedRelation α] : IsAsymm α WellFoundedRelation.rel :=
  WellFoundedRelation.wf.isAsymm


instance : IsIrrefl α WellFoundedRelation.rel := IsAsymm.isIrrefl


theorem mono (hr : WellFounded r) (h : ∀ a b, r' a b → r a b) : WellFounded r' :=
  Subrelation.wf (h _ _) hr


theorem onFun {α β : Sort*} {r : β → β → Prop} {f : α → β} :
    WellFounded r → WellFounded (r on f) :=
  InvImage.wf _


/-- If `r` is a well-founded relation, then any nonempty set has a minimal element
with respect to `r`. -/
theorem has_min {α} {r : α → α → Prop} (H : WellFounded r) (s : Set α) :
    s.Nonempty → ∃ a ∈ s, ∀ x ∈ s, ¬r x a
  | ⟨a, ha⟩ => show ∃ b ∈ s, ∀ x ∈ s, ¬r x b from
    Acc.recOn (H.apply a) (fun x _ IH =>
        not_imp_not.1 fun hne hx => hne <| ⟨x, hx, fun y hy hyx => hne <| IH y hyx hy⟩)
      ha


/-- A minimal element of a nonempty set in a well-founded order.

If you're working with a nonempty linear order, consider defining a
`ConditionallyCompleteLinearOrderBot` instance via
`WellFounded.conditionallyCompleteLinearOrderWithBot` and using `Inf` instead. -/
noncomputable def min {r : α → α → Prop} (H : WellFounded r) (s : Set α) (h : s.Nonempty) : α :=
  Classical.choose (H.has_min s h)


theorem min_mem {r : α → α → Prop} (H : WellFounded r) (s : Set α) (h : s.Nonempty) :
    H.min s h ∈ s :=
  let ⟨h, _⟩ := Classical.choose_spec (H.has_min s h)
  h


theorem not_lt_min {r : α → α → Prop} (H : WellFounded r) (s : Set α) (h : s.Nonempty) {x}
    (hx : x ∈ s) : ¬r x (H.min s h) :=
  let ⟨_, h'⟩ := Classical.choose_spec (H.has_min s h)
  h' _ hx


theorem wellFounded_iff_has_min {r : α → α → Prop} :
    WellFounded r ↔ ∀ s : Set α, s.Nonempty → ∃ m ∈ s, ∀ x ∈ s, ¬r x m := by
  /-
    α : Type u_1
    r : α → α → Prop
    ⊢ Iff (WellFounded r) (∀ (s : Set α), s.Nonempty → Exists fun m => And (Member …
  -/
  refine ⟨fun h => h.has_min, fun h => ⟨fun x => ?_⟩⟩
  /-
    α : Type u_1
    r : α → α → Prop
    h : ∀ (s : Set α), s.Nonempty → Exists fun m => And (Membership.mem s m) (∀ (x …
    x : α
    ⊢ Acc r x
  -/
  by_contra hx
  /-
    α : Type u_1
    r : α → α → Prop
    h : ∀ (s : Set α), s.Nonempty → Exists fun m => And (Membership.mem s m) (∀ (x …
    x : α
    hx : Not (Acc r x)
    ⊢ False
  -/
  obtain ⟨m, hm, hm'⟩ := h {x | ¬Acc r x} ⟨x, hx⟩
  /-
    case intro.intro
    α : Type u_1
    r : α → α → Prop
    h : ∀ (s : Set α), s.Nonempty → Exists fun m => And (Membership.mem s m) (∀ (x …
    x : α
    hx : Not (Acc r x)
    m : α
    hm : Membership.mem (setOf fun x => Not (Acc r x)) m
    hm' : ∀ (x : α), Membership.mem (setOf fun x => Not (Acc r x)) x → Not (r x m)
    ⊢ False
  -/
  refine hm ⟨_, fun y hy => ?_⟩
  /-
    case intro.intro
    α : Type u_1
    r : α → α → Prop
    h : ∀ (s : Set α), s.Nonempty → Exists fun m => And (Membership.mem s m) (∀ (x …
    x : α
    hx : Not (Acc r x)
    m : α
    hm : Membership.mem (setOf fun x => Not (Acc r x)) m
    hm' : ∀ (x : α), Membership.mem (setOf fun x => Not (Acc r x)) x → Not (r x m)
    y : α
    hy : r y m
    ⊢ Acc r y
  -/
  by_contra hy'
  /-
    case intro.intro
    α : Type u_1
    r : α → α → Prop
    h : ∀ (s : Set α), s.Nonempty → Exists fun m => And (Membership.mem s m) (∀ (x …
    x : α
    hx : Not (Acc r x)
    m : α
    hm : Membership.mem (setOf fun x => Not (Acc r x)) m
    hm' : ∀ (x : α), Membership.mem (setOf fun x => Not (Acc r x)) x → Not (r x m)
    y : α
    hy : r y m
    hy' : Not (Acc r y)
    ⊢ False
  -/
  exact hm' y hy' hy
  /-
    🎉 no goals
  -/


/-- The supremum of a bounded, well-founded order -/
protected noncomputable def sup {r : α → α → Prop} (wf : WellFounded r) (s : Set α)
    (h : Bounded r s) : α :=
  wf.min { x | ∀ a ∈ s, r a x } h


protected theorem lt_sup {r : α → α → Prop} (wf : WellFounded r) {s : Set α} (h : Bounded r s) {x}
    (hx : x ∈ s) : r x (wf.sup s h) :=
  min_mem wf { x | ∀ a ∈ s, r a x } h x hx


open Classical in
set_option linter.deprecated false in
/-- A successor of an element `x` in a well-founded order is a minimal element `y` such that
`x < y` if one exists. Otherwise it is `x` itself. -/
@[deprecated "If you have a linear order, consider defining a `SuccOrder` instance through
`ConditionallyCompleteLinearOrder.toSuccOrder`." (since := "2024-10-25")]
protected noncomputable def succ {r : α → α → Prop} (wf : WellFounded r) (x : α) : α :=
  if h : ∃ y, r x y then wf.min { y | r x y } h else x


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-10-25")]
protected theorem lt_succ {r : α → α → Prop} (wf : WellFounded r) {x : α} (h : ∃ y, r x y) :
    r x (wf.succ x) := by
  /-
    α : Type u_1
    r : α → α → Prop
    wf : WellFounded r
    x : α
    h : Exists fun y => r x y
    ⊢ r x (wf.succ x)
  -/
  rw [WellFounded.succ, dif_pos h]
  /-
    α : Type u_1
    r : α → α → Prop
    wf : WellFounded r
    x : α
    h : Exists fun y => r x y
    ⊢ r x (wf.min (setOf fun y => r x y) h)
  -/
  apply min_mem
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-10-25")]
protected theorem lt_succ_iff {r : α → α → Prop} [wo : IsWellOrder α r] {x : α} (h : ∃ y, r x y)
    (y : α) : r y (wo.wf.succ x) ↔ r y x ∨ y = x := by
  /-
    α : Type u_1
    r : α → α → Prop
    wo : IsWellOrder α r
    x : α
    h : Exists fun y => r x y
    y : α
    ⊢ Iff (r y (⋯.succ x)) (Or (r y x) (Eq y x))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      r : α → α → Prop
      wo : IsWellOrder α r
      x : α
      h : Exists fun y => r x y
      y : α
      ⊢ r y (⋯.succ x) → Or (r y x) (Eq y x)
    -/
  · intro h'
    have : ¬r x y := by
      intro hy
      rw [WellFounded.succ, dif_pos] at h'
      exact wo.wf.not_lt_min _ h hy h'
    /-
      case mp
      α : Type u_1
      r : α → α → Prop
      wo : IsWellOrder α r
      x : α
      h : Exists fun y => r x y
      y : α
      h' : r y (⋯.succ x)
      this : Not (r x y)
      ⊢ Or (r y x) (Eq y x)
    -/
    rcases trichotomous_of r x y with (hy | hy | hy)
      /-
        case mp.inl
        α : Type u_1
        r : α → α → Prop
        wo : IsWellOrder α r
        x : α
        h : Exists fun y => r x y
        y : α
        h' : r y (⋯.succ x)
        this : Not (r x y)
        hy : r x y
        ⊢ Or (r y x) (Eq y x)
      -/
    · exfalso
      /-
        case mp.inl
        α : Type u_1
        r : α → α → Prop
        wo : IsWellOrder α r
        x : α
        h : Exists fun y => r x y
        y : α
        h' : r y (⋯.succ x)
        this : Not (r x y)
        hy : r x y
        ⊢ False
      -/
      exact this hy
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inl
        α : Type u_1
        r : α → α → Prop
        wo : IsWellOrder α r
        x : α
        h : Exists fun y => r x y
        y : α
        h' : r y (⋯.succ x)
        this : Not (r x y)
        hy : Eq x y
        ⊢ Or (r y x) (Eq y x)
      -/
    · right
      /-
        case mp.inr.inl.h
        α : Type u_1
        r : α → α → Prop
        wo : IsWellOrder α r
        x : α
        h : Exists fun y => r x y
        y : α
        h' : r y (⋯.succ x)
        this : Not (r x y)
        hy : Eq x y
        ⊢ Eq y x
      -/
      exact hy.symm
      /-
        🎉 no goals
      -/
    /-
      case mp.inr.inr
      α : Type u_1
      r : α → α → Prop
      wo : IsWellOrder α r
      x : α
      h : Exists fun y => r x y
      y : α
      h' : r y (⋯.succ x)
      this : Not (r x y)
      hy : r y x
      ⊢ Or (r y x) (Eq y x)
    -/
    left
    /-
      case mp.inr.inr.h
      α : Type u_1
      r : α → α → Prop
      wo : IsWellOrder α r
      x : α
      h : Exists fun y => r x y
      y : α
      h' : r y (⋯.succ x)
      this : Not (r x y)
      hy : r y x
      ⊢ r y x
    -/
    exact hy
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    r : α → α → Prop
    wo : IsWellOrder α r
    x : α
    h : Exists fun y => r x y
    y : α
    ⊢ Or (r y x) (Eq y x) → r y (⋯.succ x)
  -/
                        /-
                          🎉 no goals
                        -/
  rintro (hy | rfl); (· exact _root_.trans hy (wo.wf.lt_succ h)); exact wo.wf.lt_succ h
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem WellFounded.min_le (h : WellFounded ((· < ·) : β → β → Prop))
    {x : β} {s : Set β} (hx : x ∈ s) (hne : s.Nonempty := ⟨x, hx⟩) : h.min s hne ≤ x :=
  not_lt.1 <| h.not_lt_min _ _ hx


theorem Set.range_injOn_strictMono [WellFoundedLT β] :
    Set.InjOn Set.range { f : β → γ | StrictMono f } := by
  /-
    β : Type u_2
    γ : Type u_3
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : WellFoundedLT β
    ⊢ Set.InjOn Set.range (setOf fun f => StrictMono f)
  -/
  intro f hf g hg hfg
  /-
    β : Type u_2
    γ : Type u_3
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : WellFoundedLT β
    f : β → γ
    hf : Membership.mem (setOf fun f => StrictMono f) f
    g : β → γ
    hg : Membership.mem (setOf fun f => StrictMono f) g
    hfg : Eq (Set.range f) (Set.range g)
    ⊢ Eq f g
  -/
  ext a
  /-
    case h
    β : Type u_2
    γ : Type u_3
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : WellFoundedLT β
    f : β → γ
    hf : Membership.mem (setOf fun f => StrictMono f) f
    g : β → γ
    hg : Membership.mem (setOf fun f => StrictMono f) g
    hfg : Eq (Set.range f) (Set.range g)
    a : β
    ⊢ Eq (f a) (g a)
  -/
  apply WellFoundedLT.induction a
  /-
    case h
    β : Type u_2
    γ : Type u_3
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : WellFoundedLT β
    f : β → γ
    hf : Membership.mem (setOf fun f => StrictMono f) f
    g : β → γ
    hg : Membership.mem (setOf fun f => StrictMono f) g
    hfg : Eq (Set.range f) (Set.range g)
    a : β
    ⊢ ∀ (x : β), (∀ (y : β), LT.lt y x → Eq (f y) (g y)) → Eq (f x) (g x)
  -/
  intro a IH
  /-
    case h
    β : Type u_2
    γ : Type u_3
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : WellFoundedLT β
    f : β → γ
    hf : Membership.mem (setOf fun f => StrictMono f) f
    g : β → γ
    hg : Membership.mem (setOf fun f => StrictMono f) g
    hfg : Eq (Set.range f) (Set.range g)
    a✝ a : β
    IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
    ⊢ Eq (f a) (g a)
  -/
  obtain ⟨b, hb⟩ := hfg ▸ mem_range_self a
  /-
    case h.intro
    β : Type u_2
    γ : Type u_3
    inst✝² : LinearOrder β
    inst✝¹ : Preorder γ
    inst✝ : WellFoundedLT β
    f : β → γ
    hf : Membership.mem (setOf fun f => StrictMono f) f
    g : β → γ
    hg : Membership.mem (setOf fun f => StrictMono f) g
    hfg : Eq (Set.range f) (Set.range g)
    a✝ a : β
    IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
    b : β
    hb : Eq (g b) (f a)
    ⊢ Eq (f a) (g a)
  -/
  obtain h | rfl | h := lt_trichotomy b a
    /-
      case h.intro.inl
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (g b) (f a)
      h : LT.lt b a
      ⊢ Eq (f a) (g a)
    -/
  · rw [← IH b h] at hb
    /-
      case h.intro.inl
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (f b) (f a)
      h : LT.lt b a
      ⊢ Eq (f a) (g a)
    -/
    cases (hf.injective hb).not_lt h
    /-
      🎉 no goals
    -/
    /-
      case h.intro.inr.inl
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a b : β
      IH : ∀ (y : β), LT.lt y b → Eq (f y) (g y)
      hb : Eq (g b) (f b)
      ⊢ Eq (f b) (g b)
    -/
  · rw [hb]
    /-
      🎉 no goals
    -/
    /-
      case h.intro.inr.inr
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (g b) (f a)
      h : LT.lt a b
      ⊢ Eq (f a) (g a)
    -/
  · obtain ⟨c, hc⟩ := hfg.symm ▸ mem_range_self a
    /-
      case h.intro.inr.inr.intro
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (g b) (f a)
      h : LT.lt a b
      c : β
      hc : Eq (f c) (g a)
      ⊢ Eq (f a) (g a)
    -/
    have := hg h
    /-
      case h.intro.inr.inr.intro
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (g b) (f a)
      h : LT.lt a b
      c : β
      hc : Eq (f c) (g a)
      this : LT.lt (g a) (g b)
      ⊢ Eq (f a) (g a)
    -/
    rw [hb, ← hc, hf.lt_iff_lt] at this
    /-
      case h.intro.inr.inr.intro
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (g b) (f a)
      h : LT.lt a b
      c : β
      hc : Eq (f c) (g a)
      this : LT.lt c a
      ⊢ Eq (f a) (g a)
    -/
    rw [IH c this] at hc
    /-
      case h.intro.inr.inr.intro
      β : Type u_2
      γ : Type u_3
      inst✝² : LinearOrder β
      inst✝¹ : Preorder γ
      inst✝ : WellFoundedLT β
      f : β → γ
      hf : Membership.mem (setOf fun f => StrictMono f) f
      g : β → γ
      hg : Membership.mem (setOf fun f => StrictMono f) g
      hfg : Eq (Set.range f) (Set.range g)
      a✝ a : β
      IH : ∀ (y : β), LT.lt y a → Eq (f y) (g y)
      b : β
      hb : Eq (g b) (f a)
      h : LT.lt a b
      c : β
      hc : Eq (g c) (g a)
      this : LT.lt c a
      ⊢ Eq (f a) (g a)
    -/
    cases (hg.injective hc).not_lt this
    /-
      🎉 no goals
    -/


theorem Set.range_injOn_strictAnti [WellFoundedGT β] :
    Set.InjOn Set.range { f : β → γ | StrictAnti f } :=
  fun _ hf _ hg ↦ Set.range_injOn_strictMono (β := βᵒᵈ) hf.dual hg.dual


theorem StrictMono.range_inj [WellFoundedLT β] {f g : β → γ}
    (hf : StrictMono f) (hg : StrictMono g) : Set.range f = Set.range g ↔ f = g :=
  Set.range_injOn_strictMono.eq_iff hf hg


theorem StrictAnti.range_inj [WellFoundedGT β] {f g : β → γ}
    (hf : StrictAnti f) (hg : StrictAnti g) : Set.range f = Set.range g ↔ f = g :=
  Set.range_injOn_strictAnti.eq_iff hf hg


@[deprecated StrictMono.range_inj (since := "2024-09-11")]
theorem WellFounded.eq_strictMono_iff_eq_range (h : WellFounded ((· < ·) : β → β → Prop))
    {f g : β → γ} (hf : StrictMono f) (hg : StrictMono g) :
    Set.range f = Set.range g ↔ f = g :=
  @StrictMono.range_inj β γ _ _ ⟨h⟩ f g hf hg


/-- A strictly monotone function `f` on a well-order satisfies `x ≤ f x` for all `x`. -/
theorem StrictMono.id_le [WellFoundedLT β] {f : β → β} (hf : StrictMono f) : id ≤ f := by
  /-
    β : Type u_2
    inst✝¹ : LinearOrder β
    inst✝ : WellFoundedLT β
    f : β → β
    hf : StrictMono f
    ⊢ LE.le id f
  -/
  rw [Pi.le_def]
  /-
    β : Type u_2
    inst✝¹ : LinearOrder β
    inst✝ : WellFoundedLT β
    f : β → β
    hf : StrictMono f
    ⊢ ∀ (i : β), LE.le (id i) (f i)
  -/
  by_contra! H
  /-
    β : Type u_2
    inst✝¹ : LinearOrder β
    inst✝ : WellFoundedLT β
    f : β → β
    hf : StrictMono f
    H : Exists fun i => LT.lt (f i) (id i)
    ⊢ False
  -/
  obtain ⟨m, hm, hm'⟩ := wellFounded_lt.has_min _ H
  /-
    case intro.intro
    β : Type u_2
    inst✝¹ : LinearOrder β
    inst✝ : WellFoundedLT β
    f : β → β
    hf : StrictMono f
    H : Exists fun i => LT.lt (f i) (id i)
    m : β
    hm : Membership.mem (fun x => Preorder.toLT.1 (f x) (id x)) m
    hm' : ∀ (x : β), Membership.mem (fun x => Preorder.toLT.1 (f x) (id x)) x → No …
    ⊢ False
  -/
  exact hm' _ (hf hm) hm
  /-
    🎉 no goals
  -/


theorem StrictMono.le_apply [WellFoundedLT β] {f : β → β} (hf : StrictMono f) {x} : x ≤ f x :=
  hf.id_le x


/-- A strictly monotone function `f` on a cowell-order satisfies `f x ≤ x` for all `x`. -/
theorem StrictMono.le_id [WellFoundedGT β] {f : β → β} (hf : StrictMono f) : f ≤ id :=
  StrictMono.id_le (β := βᵒᵈ) hf.dual


theorem StrictMono.apply_le [WellFoundedGT β] {f : β → β} (hf : StrictMono f) {x} : f x ≤ x :=
  StrictMono.le_apply (β := βᵒᵈ) hf.dual


@[deprecated StrictMono.le_apply (since := "2024-09-11")]
theorem WellFounded.self_le_of_strictMono (h : WellFounded ((· < ·) : β → β → Prop))
    {f : β → β} (hf : StrictMono f) : ∀ n, n ≤ f n := by
  /-
    β : Type u_2
    inst✝ : LinearOrder β
    h : WellFounded fun x1 x2 => LT.lt x1 x2
    f : β → β
    hf : StrictMono f
    ⊢ ∀ (n : β), LE.le n (f n)
  -/
  by_contra! h₁
  /-
    β : Type u_2
    inst✝ : LinearOrder β
    h : WellFounded fun x1 x2 => LT.lt x1 x2
    f : β → β
    hf : StrictMono f
    h₁ : Exists fun n => LT.lt (f n) n
    ⊢ False
  -/
  have h₂ := h.min_mem _ h₁
  /-
    β : Type u_2
    inst✝ : LinearOrder β
    h : WellFounded fun x1 x2 => LT.lt x1 x2
    f : β → β
    hf : StrictMono f
    h₁ : Exists fun n => LT.lt (f n) n
    h₂ : Membership.mem (fun x => Preorder.toLT.1 (f x) x) (h.min (fun x => Preord …
    ⊢ False
  -/
  exact h.not_lt_min _ h₁ (hf h₂) h₂
  /-
    🎉 no goals
  -/


theorem StrictMono.not_bddAbove_range_of_wellFoundedLT {f : β → β} [WellFoundedLT β] [NoMaxOrder β]
    (hf : StrictMono f) : ¬ BddAbove (Set.range f) := by
  /-
    β : Type u_2
    inst✝² : LinearOrder β
    f : β → β
    inst✝¹ : WellFoundedLT β
    inst✝ : NoMaxOrder β
    hf : StrictMono f
    ⊢ Not (BddAbove (Set.range f))
  -/
  rintro ⟨a, ha⟩
  /-
    case intro
    β : Type u_2
    inst✝² : LinearOrder β
    f : β → β
    inst✝¹ : WellFoundedLT β
    inst✝ : NoMaxOrder β
    hf : StrictMono f
    a : β
    ha : Membership.mem (upperBounds (Set.range f)) a
    ⊢ False
  -/
  obtain ⟨b, hb⟩ := exists_gt a
  /-
    case intro.intro
    β : Type u_2
    inst✝² : LinearOrder β
    f : β → β
    inst✝¹ : WellFoundedLT β
    inst✝ : NoMaxOrder β
    hf : StrictMono f
    a : β
    ha : Membership.mem (upperBounds (Set.range f)) a
    b : β
    hb : LT.lt a b
    ⊢ False
  -/
  exact ((hf.le_apply.trans_lt (hf hb)).trans_le <| ha (Set.mem_range_self _)).false
  /-
    🎉 no goals
  -/


theorem StrictMono.not_bddBelow_range_of_wellFoundedGT {f : β → β} [WellFoundedGT β] [NoMinOrder β]
    (hf : StrictMono f) : ¬ BddBelow (Set.range f) :=
  hf.dual.not_bddAbove_range_of_wellFoundedLT


/-- Given a function `f : α → β` where `β` carries a well-founded `<`, this is an element of `α`
whose image under `f` is minimal in the sense of `Function.not_lt_argmin`. -/
noncomputable def argmin [Nonempty α] : α :=
  WellFounded.min (InvImage.wf f h) Set.univ Set.univ_nonempty


theorem not_lt_argmin [Nonempty α] (a : α) : ¬f a < f (argmin f h) :=
  WellFounded.not_lt_min (InvImage.wf f h) _ _ (Set.mem_univ a)


/-- Given a function `f : α → β` where `β` carries a well-founded `<`, and a non-empty subset `s`
of `α`, this is an element of `s` whose image under `f` is minimal in the sense of
`Function.not_lt_argminOn`. -/
noncomputable def argminOn (s : Set α) (hs : s.Nonempty) : α :=
  WellFounded.min (InvImage.wf f h) s hs


@[simp]
theorem argminOn_mem (s : Set α) (hs : s.Nonempty) : argminOn f h s hs ∈ s :=
  WellFounded.min_mem _ _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] removed as it will never apply

theorem not_lt_argminOn (s : Set α) {a : α} (ha : a ∈ s)
    (hs : s.Nonempty := Set.nonempty_of_mem ha) : ¬f a < f (argminOn f h s hs) :=
  WellFounded.not_lt_min (InvImage.wf f h) s hs ha


theorem argmin_le (a : α) [Nonempty α] : f (argmin f h) ≤ f a :=
  not_lt.mp <| not_lt_argmin f h a

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] removed as it will never apply

theorem argminOn_le (s : Set α) {a : α} (ha : a ∈ s) (hs : s.Nonempty := Set.nonempty_of_mem ha) :
    f (argminOn f h s hs) ≤ f a :=
  not_lt.mp <| not_lt_argminOn f h s ha hs


/-- Let `r` be a relation on `α`, let `f : α → β` be a function, let `C : β → Prop`, and
let `bot : α`. This induction principle shows that `C (f bot)` holds, given that
* some `a` that is accessible by `r` satisfies `C (f a)`, and
* for each `b` such that `f b ≠ f bot` and `C (f b)` holds, there is `c`
  satisfying `r c b` and `C (f c)`. -/
theorem Acc.induction_bot' {α β} {r : α → α → Prop} {a bot : α} (ha : Acc r a) {C : β → Prop}
    {f : α → β} (ih : ∀ b, f b ≠ f bot → C (f b) → ∃ c, r c b ∧ C (f c)) : C (f a) → C (f bot) :=
  (@Acc.recOn _ _ (fun x _ => C (f x) → C (f bot)) _ ha) fun x _ ih' hC =>
    (eq_or_ne (f x) (f bot)).elim (fun h => h ▸ hC) (fun h =>
      let ⟨y, hy₁, hy₂⟩ := ih x h hC
      ih' y hy₁ hy₂)


/-- Let `r` be a relation on `α`, let `C : α → Prop` and let `bot : α`.
This induction principle shows that `C bot` holds, given that
* some `a` that is accessible by `r` satisfies `C a`, and
* for each `b ≠ bot` such that `C b` holds, there is `c` satisfying `r c b` and `C c`. -/
theorem Acc.induction_bot {α} {r : α → α → Prop} {a bot : α} (ha : Acc r a) {C : α → Prop}
    (ih : ∀ b, b ≠ bot → C b → ∃ c, r c b ∧ C c) : C a → C bot :=
  ha.induction_bot' ih


/-- Let `r` be a well-founded relation on `α`, let `f : α → β` be a function,
let `C : β → Prop`, and let `bot : α`.
This induction principle shows that `C (f bot)` holds, given that
* some `a` satisfies `C (f a)`, and
* for each `b` such that `f b ≠ f bot` and `C (f b)` holds, there is `c`
  satisfying `r c b` and `C (f c)`. -/
theorem WellFounded.induction_bot' {α β} {r : α → α → Prop} (hwf : WellFounded r) {a bot : α}
    {C : β → Prop} {f : α → β} (ih : ∀ b, f b ≠ f bot → C (f b) → ∃ c, r c b ∧ C (f c)) :
    C (f a) → C (f bot) :=
  (hwf.apply a).induction_bot' ih


/-- Let `r` be a well-founded relation on `α`, let `C : α → Prop`, and let `bot : α`.
This induction principle shows that `C bot` holds, given that
* some `a` satisfies `C a`, and
* for each `b` that satisfies `C b`, there is `c` satisfying `r c b` and `C c`.

The naming is inspired by the fact that when `r` is transitive, it follows that `bot` is
the smallest element w.r.t. `r` that satisfies `C`. -/
theorem WellFounded.induction_bot {α} {r : α → α → Prop} (hwf : WellFounded r) {a bot : α}
    {C : α → Prop} (ih : ∀ b, b ≠ bot → C b → ∃ c, r c b ∧ C c) : C a → C bot :=
  hwf.induction_bot' ih


/-- A nonempty linear order with well-founded `<` has a bottom element. -/
noncomputable def WellFoundedLT.toOrderBot {α} [LinearOrder α] [Nonempty α] [h : WellFoundedLT α] :
    OrderBot α where
  bot := h.wf.min _ Set.univ_nonempty
  bot_le a := h.wf.min_le (Set.mem_univ a)


/-- A nonempty linear order with well-founded `>` has a top element. -/
noncomputable def WellFoundedGT.toOrderTop {α} [LinearOrder α] [Nonempty α] [WellFoundedGT α] :
    OrderTop α :=
  have := WellFoundedLT.toOrderBot (α := αᵒᵈ)
  inferInstanceAs (OrderTop αᵒᵈᵒᵈ)

