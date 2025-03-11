/-- **Zorn's lemma**: A partial order is coatomic if every nonempty chain `c`, `⊤ ∉ c`, has an upper
bound not equal to `⊤`. -/
theorem IsCoatomic.of_isChain_bounded {α : Type*} [PartialOrder α] [OrderTop α]
    (h : ∀ c : Set α, IsChain (· ≤ ·) c → c.Nonempty → ⊤ ∉ c → ∃ x ≠ ⊤, x ∈ upperBounds c) :
    IsCoatomic α := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : OrderTop α
    h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
    ⊢ IsCoatomic α
  -/
  refine ⟨fun x => le_top.eq_or_lt.imp_right fun hx => ?_⟩
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : OrderTop α
    h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
    x : α
    hx : LT.lt x Top.top
    ⊢ Exists fun a => And (IsCoatom a) (LE.le x a)
  -/
  have := zorn_le_nonempty₀ (Ico x ⊤) (fun c hxc hc y hy => ?_) x (left_mem_Ico.2 hx)
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : OrderTop α
      h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
      x : α
      hx : LT.lt x Top.top
      this : Exists fun m => And (LE.le x m) (Maximal (fun x_1 => Membership.mem (Se …
      ⊢ Exists fun a => And (IsCoatom a) (LE.le x a)
    -/
  · obtain ⟨y, hxy, hmax⟩ := this
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : OrderTop α
      h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
      x : α
      hx : LT.lt x Top.top
      y : α
      hxy : LE.le x y
      hmax : Maximal (fun x_1 => Membership.mem (Set.Ico x Top.top) x_1) y
      ⊢ Exists fun a => And (IsCoatom a) (LE.le x a)
    -/
    refine ⟨y, ⟨hmax.prop.2.ne, fun z hyz ↦ le_top.eq_or_lt.resolve_right fun hz => ?_⟩, hxy⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝¹ : PartialOrder α
      inst✝ : OrderTop α
      h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
      x : α
      hx : LT.lt x Top.top
      y : α
      hxy : LE.le x y
      hmax : Maximal (fun x_1 => Membership.mem (Set.Ico x Top.top) x_1) y
      z : α
      hyz : LT.lt y z
      hz : LT.lt z Top.top
      ⊢ False
    -/
    exact hyz.ne <| hmax.eq_of_le ⟨hxy.trans hyz.le, hz⟩ hyz.le
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : OrderTop α
    h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
    x : α
    hx : LT.lt x Top.top
    c : Set α
    hxc : HasSubset.Subset c (Set.Ico x Top.top)
    hc : IsChain (fun x1 x2 => LE.le x1 x2) c
    y : α
    hy : Membership.mem c y
    ⊢ Exists fun ub => And (Membership.mem (Set.Ico x Top.top) ub) (∀ (z : α), Mem …
  -/
  rcases h c hc ⟨y, hy⟩ fun h => (hxc h).2.ne rfl with ⟨z, hz, hcz⟩
  /-
    case refine_1.intro.intro
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : OrderTop α
    h : ∀ (c : Set α), IsChain (fun x1 x2 => LE.le x1 x2) c → c.Nonempty → Not (Me …
    x : α
    hx : LT.lt x Top.top
    c : Set α
    hxc : HasSubset.Subset c (Set.Ico x Top.top)
    hc : IsChain (fun x1 x2 => LE.le x1 x2) c
    y : α
    hy : Membership.mem c y
    z : α
    hz : Ne z Top.top
    hcz : Membership.mem (upperBounds c) z
    ⊢ Exists fun ub => And (Membership.mem (Set.Ico x Top.top) ub) (∀ (z : α), Mem …
  -/
  exact ⟨z, ⟨le_trans (hxc hy).1 (hcz hy), hz.lt_top⟩, hcz⟩
  /-
    🎉 no goals
  -/


/-- **Zorn's lemma**: A partial order is atomic if every nonempty chain `c`, `⊥ ∉ c`, has a lower
bound not equal to `⊥`. -/
theorem IsAtomic.of_isChain_bounded {α : Type*} [PartialOrder α] [OrderBot α]
    (h :
      ∀ c : Set α,
        IsChain (· ≤ ·) c → c.Nonempty → ⊥ ∉ c → ∃ x ≠ ⊥, x ∈ lowerBounds c) :
    IsAtomic α :=
  isCoatomic_dual_iff_isAtomic.mp <| IsCoatomic.of_isChain_bounded fun c hc => h c hc.symm

