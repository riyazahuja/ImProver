/-- Let `f : α → β` be a function from a linear order `α` to a topological space `β`, and
let `a : α`. The limit strictly to the left of `f` at `a`, denoted with `leftLim f a`, is defined
by using the order topology on `α`. If `a` is isolated to its left or the function has no left
limit, we use `f a` instead to guarantee a good behavior in most cases. -/
noncomputable def Function.leftLim (f : α → β) (a : α) : β := by
  classical
  haveI : Nonempty β := ⟨f a⟩
  letI : TopologicalSpace α := Preorder.topology α
  exact if 𝓝[<] a = ⊥ ∨ ¬∃ y, Tendsto f (𝓝[<] a) (𝓝 y) then f a else limUnder (𝓝[<] a) f


/-- Let `f : α → β` be a function from a linear order `α` to a topological space `β`, and
let `a : α`. The limit strictly to the right of `f` at `a`, denoted with `rightLim f a`, is defined
by using the order topology on `α`. If `a` is isolated to its right or the function has no right
limit, , we use `f a` instead to guarantee a good behavior in most cases. -/
noncomputable def Function.rightLim (f : α → β) (a : α) : β :=
  @Function.leftLim αᵒᵈ β _ _ f a


theorem leftLim_eq_of_tendsto [hα : TopologicalSpace α] [h'α : OrderTopology α] [T2Space β]
    {f : α → β} {a : α} {y : β} (h : 𝓝[<] a ≠ ⊥) (h' : Tendsto f (𝓝[<] a) (𝓝 y)) :
    leftLim f a = y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    inst✝ : T2Space β
    f : α → β
    a : α
    y : β
    h : Ne (nhdsWithin a (Set.Iio a)) Bot.bot
    h' : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    ⊢ Eq (Function.leftLim f a) y
  -/
  have h'' : ∃ y, Tendsto f (𝓝[<] a) (𝓝 y) := ⟨y, h'⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    inst✝ : T2Space β
    f : α → β
    a : α
    y : β
    h : Ne (nhdsWithin a (Set.Iio a)) Bot.bot
    h' : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    h'' : Exists fun y => Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    ⊢ Eq (Function.leftLim f a) y
  -/
  rw [h'α.topology_eq_generate_intervals] at h h' h''
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    inst✝ : T2Space β
    f : α → β
    a : α
    y : β
    h : Ne (nhdsWithin a (Set.Iio a)) Bot.bot
    h' : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    h'' : Exists fun y => Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    ⊢ Eq (Function.leftLim f a) y
  -/
  simp only [leftLim, h, h'', not_true, or_self_iff, if_false]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    inst✝ : T2Space β
    f : α → β
    a : α
    y : β
    h : Ne (nhdsWithin a (Set.Iio a)) Bot.bot
    h' : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    h'' : Exists fun y => Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    ⊢ Eq (limUnder (nhdsWithin a (Set.Iio a)) f) y
  -/
  haveI := neBot_iff.2 h
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    inst✝ : T2Space β
    f : α → β
    a : α
    y : β
    h : Ne (nhdsWithin a (Set.Iio a)) Bot.bot
    h' : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    h'' : Exists fun y => Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds y)
    this : (nhdsWithin a (Set.Iio a)).NeBot
    ⊢ Eq (limUnder (nhdsWithin a (Set.Iio a)) f) y
  -/
  exact lim_eq h'
  /-
    🎉 no goals
  -/


theorem leftLim_eq_of_eq_bot [hα : TopologicalSpace α] [h'α : OrderTopology α] (f : α → β) {a : α}
    (h : 𝓝[<] a = ⊥) : leftLim f a = f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    f : α → β
    a : α
    h : Eq (nhdsWithin a (Set.Iio a)) Bot.bot
    ⊢ Eq (Function.leftLim f a) (f a)
  -/
  rw [h'α.topology_eq_generate_intervals] at h
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : TopologicalSpace β
    hα : TopologicalSpace α
    h'α : OrderTopology α
    f : α → β
    a : α
    h : Eq (nhdsWithin a (Set.Iio a)) Bot.bot
    ⊢ Eq (Function.leftLim f a) (f a)
  -/
  simp [leftLim, ite_eq_left_iff, h]
  /-
    🎉 no goals
  -/


theorem rightLim_eq_of_tendsto [TopologicalSpace α] [OrderTopology α] [T2Space β]
    {f : α → β} {a : α} {y : β} (h : 𝓝[>] a ≠ ⊥) (h' : Tendsto f (𝓝[>] a) (𝓝 y)) :
    Function.rightLim f a = y :=
  @leftLim_eq_of_tendsto αᵒᵈ _ _ _ _ _ _ f a y h h'


theorem rightLim_eq_of_eq_bot [TopologicalSpace α] [OrderTopology α] (f : α → β) {a : α}
    (h : 𝓝[>] a = ⊥) : rightLim f a = f a :=
  @leftLim_eq_of_eq_bot αᵒᵈ _ _ _ _ _  f a h


theorem leftLim_eq_sSup [TopologicalSpace α] [OrderTopology α] (h : 𝓝[<] x ≠ ⊥) :
    leftLim f x = sSup (f '' Iio x) :=
  leftLim_eq_of_tendsto h (hf.tendsto_nhdsLT x)


theorem rightLim_eq_sInf [TopologicalSpace α] [OrderTopology α] (h : 𝓝[>] x ≠ ⊥) :
    rightLim f x = sInf (f '' Ioi x) :=
  rightLim_eq_of_tendsto h (hf.tendsto_nhdsGT x)


theorem leftLim_le (h : x ≤ y) : leftLim f x ≤ f y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    ⊢ LE.le (Function.leftLim f x) (f y)
  -/
  letI : TopologicalSpace α := Preorder.topology α
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    this : TopologicalSpace α := Preorder.topology α
    ⊢ LE.le (Function.leftLim f x) (f y)
  -/
  haveI : OrderTopology α := ⟨rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    ⊢ LE.le (Function.leftLim f x) (f y)
  -/
  rcases eq_or_ne (𝓝[<] x) ⊥ with (h' | h')
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Eq (nhdsWithin x (Set.Iio x)) Bot.bot
      ⊢ LE.le (Function.leftLim f x) (f y)
    -/
  · simpa [leftLim, h'] using hf h
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    ⊢ LE.le (Function.leftLim f x) (f y)
  -/
  haveI A : NeBot (𝓝[<] x) := neBot_iff.2 h'
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    A : (nhdsWithin x (Set.Iio x)).NeBot
    ⊢ LE.le (Function.leftLim f x) (f y)
  -/
  rw [leftLim_eq_sSup hf h']
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    A : (nhdsWithin x (Set.Iio x)).NeBot
    ⊢ LE.le (SupSet.sSup (Set.image f (Set.Iio x))) (f y)
  -/
  refine csSup_le ?_ ?_
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
      A : (nhdsWithin x (Set.Iio x)).NeBot
      ⊢ (Set.image f (Set.Iio x)).Nonempty
    -/
  · simp only [image_nonempty]
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
      A : (nhdsWithin x (Set.Iio x)).NeBot
      ⊢ (Set.Iio x).Nonempty
    -/
    exact (forall_mem_nonempty_iff_neBot.2 A) _ self_mem_nhdsWithin
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
      A : (nhdsWithin x (Set.Iio x)).NeBot
      ⊢ ∀ (b : β), Membership.mem (Set.image f (Set.Iio x)) b → LE.le b (f y)
    -/
  · simp only [mem_image, mem_Iio, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    /-
      case inr.refine_2
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
      A : (nhdsWithin x (Set.Iio x)).NeBot
      ⊢ ∀ (a : α), LT.lt a x → LE.le (f a) (f y)
    -/
    intro z hz
    /-
      case inr.refine_2
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
      A : (nhdsWithin x (Set.Iio x)).NeBot
      z : α
      hz : LT.lt z x
      ⊢ LE.le (f z) (f y)
    -/
    exact hf (hz.le.trans h)
    /-
      🎉 no goals
    -/


theorem le_leftLim (h : x < y) : f x ≤ leftLim f y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    ⊢ LE.le (f x) (Function.leftLim f y)
  -/
  letI : TopologicalSpace α := Preorder.topology α
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this : TopologicalSpace α := Preorder.topology α
    ⊢ LE.le (f x) (Function.leftLim f y)
  -/
  haveI : OrderTopology α := ⟨rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    ⊢ LE.le (f x) (Function.leftLim f y)
  -/
  rcases eq_or_ne (𝓝[<] y) ⊥ with (h' | h')
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LT.lt x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Eq (nhdsWithin y (Set.Iio y)) Bot.bot
      ⊢ LE.le (f x) (Function.leftLim f y)
    -/
  · rw [leftLim_eq_of_eq_bot _ h']
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LT.lt x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Eq (nhdsWithin y (Set.Iio y)) Bot.bot
      ⊢ LE.le (f x) (f y)
    -/
    exact hf h.le
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin y (Set.Iio y)) Bot.bot
    ⊢ LE.le (f x) (Function.leftLim f y)
  -/
  rw [leftLim_eq_sSup hf h']
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin y (Set.Iio y)) Bot.bot
    ⊢ LE.le (f x) (SupSet.sSup (Set.image f (Set.Iio y)))
  -/
  refine le_csSup ⟨f y, ?_⟩ (mem_image_of_mem _ h)
  simp only [upperBounds, mem_image, mem_Iio, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, mem_setOf_eq]
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin y (Set.Iio y)) Bot.bot
    ⊢ ∀ (a : α), LT.lt a y → LE.le (f a) (f y)
  -/
  intro z hz
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : Ne (nhdsWithin y (Set.Iio y)) Bot.bot
    z : α
    hz : LT.lt z y
    ⊢ LE.le (f z) (f y)
  -/
  exact hf hz.le
  /-
    🎉 no goals
  -/


@[mono]
protected theorem leftLim : Monotone (leftLim f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    ⊢ Monotone (Function.leftLim f)
  -/
  intro x y h
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LE.le x y
    ⊢ LE.le (Function.leftLim f x) (Function.leftLim f y)
  -/
  rcases eq_or_lt_of_le h with (rfl | hxy)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x : α
      h : LE.le x x
      ⊢ LE.le (Function.leftLim f x) (Function.leftLim f x)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LE.le x y
      hxy : LT.lt x y
      ⊢ LE.le (Function.leftLim f x) (Function.leftLim f y)
    -/
  · exact (hf.leftLim_le le_rfl).trans (hf.le_leftLim hxy)
    /-
      🎉 no goals
    -/


theorem le_rightLim (h : x ≤ y) : f x ≤ rightLim f y :=
  hf.dual.leftLim_le h


theorem rightLim_le (h : x < y) : rightLim f x ≤ f y :=
  hf.dual.le_leftLim h


@[mono]
protected theorem rightLim : Monotone (rightLim f) := fun _ _ h => hf.dual.leftLim h


theorem leftLim_le_rightLim (h : x ≤ y) : leftLim f x ≤ rightLim f y :=
  (hf.leftLim_le le_rfl).trans (hf.le_rightLim h)


theorem rightLim_le_leftLim (h : x < y) : rightLim f x ≤ leftLim f y := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    ⊢ LE.le (Function.rightLim f x) (Function.leftLim f y)
  -/
  letI : TopologicalSpace α := Preorder.topology α
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this : TopologicalSpace α := Preorder.topology α
    ⊢ LE.le (Function.rightLim f x) (Function.leftLim f y)
  -/
  haveI : OrderTopology α := ⟨rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    ⊢ LE.le (Function.rightLim f x) (Function.leftLim f y)
  -/
  rcases eq_or_neBot (𝓝[<] y) with (h' | h')
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      hf : Monotone f
      x y : α
      h : LT.lt x y
      this✝ : TopologicalSpace α := Preorder.topology α
      this : OrderTopology α
      h' : Eq (nhdsWithin y (Set.Iio y)) Bot.bot
      ⊢ LE.le (Function.rightLim f x) (Function.leftLim f y)
    -/
  · simpa [leftLim, h'] using rightLim_le hf h
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    hf : Monotone f
    x y : α
    h : LT.lt x y
    this✝ : TopologicalSpace α := Preorder.topology α
    this : OrderTopology α
    h' : (nhdsWithin y (Set.Iio y)).NeBot
    ⊢ LE.le (Function.rightLim f x) (Function.leftLim f y)
  -/
  obtain ⟨a, ⟨xa, ay⟩⟩ : (Ioo x y).Nonempty := nonempty_of_mem (Ioo_mem_nhdsLT h)
  calc
    rightLim f x ≤ f a := hf.rightLim_le xa
    _ ≤ leftLim f y := hf.le_leftLim ay


theorem tendsto_leftLim (x : α) : Tendsto f (𝓝[<] x) (𝓝 (leftLim f x)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    x : α
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (Function.leftLim f x))
  -/
  rcases eq_or_ne (𝓝[<] x) ⊥ with (h' | h')
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : ConditionallyCompleteLinearOrder β
      inst✝³ : TopologicalSpace β
      inst✝² : OrderTopology β
      f : α → β
      hf : Monotone f
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      x : α
      h' : Eq (nhdsWithin x (Set.Iio x)) Bot.bot
      ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (Function.leftLim f x))
    -/
  · simp [h']
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    x : α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (Function.leftLim f x))
  -/
  rw [leftLim_eq_sSup hf h']
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    x : α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (SupSet.sSup (Set.image f  …
  -/
  exact hf.tendsto_nhdsLT x
  /-
    🎉 no goals
  -/


theorem tendsto_leftLim_within (x : α) : Tendsto f (𝓝[<] x) (𝓝[≤] leftLim f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    x : α
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhdsWithin (Function.leftLim f  …
  -/
  apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within f (hf.tendsto_leftLim x)
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    x : α
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Iic (Function.leftLim f x) …
  -/
  filter_upwards [@self_mem_nhdsWithin _ _ x (Iio x)] with y hy using hf.le_leftLim hy
  /-
    🎉 no goals
  -/


theorem tendsto_rightLim (x : α) : Tendsto f (𝓝[>] x) (𝓝 (rightLim f x)) :=
  hf.dual.tendsto_leftLim x


theorem tendsto_rightLim_within (x : α) : Tendsto f (𝓝[>] x) (𝓝[≥] rightLim f x) :=
  hf.dual.tendsto_leftLim_within x


/-- A monotone function is continuous to the left at a point if and only if its left limit
coincides with the value of the function. -/
theorem continuousWithinAt_Iio_iff_leftLim_eq :
    ContinuousWithinAt f (Iio x) x ↔ leftLim f x = f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    x : α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    ⊢ Iff (ContinuousWithinAt f (Set.Iio x) x) (Eq (Function.leftLim f x) (f x))
  -/
  rcases eq_or_ne (𝓝[<] x) ⊥ with (h' | h')
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : ConditionallyCompleteLinearOrder β
      inst✝³ : TopologicalSpace β
      inst✝² : OrderTopology β
      f : α → β
      hf : Monotone f
      x : α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      h' : Eq (nhdsWithin x (Set.Iio x)) Bot.bot
      ⊢ Iff (ContinuousWithinAt f (Set.Iio x) x) (Eq (Function.leftLim f x) (f x))
    -/
  · simp [leftLim_eq_of_eq_bot f h', ContinuousWithinAt, h']
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    x : α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    ⊢ Iff (ContinuousWithinAt f (Set.Iio x) x) (Eq (Function.leftLim f x) (f x))
  -/
  haveI : (𝓝[Iio x] x).NeBot := neBot_iff.2 h'
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    x : α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    this : (nhdsWithin x (Set.Iio x)).NeBot
    ⊢ Iff (ContinuousWithinAt f (Set.Iio x) x) (Eq (Function.leftLim f x) (f x))
  -/
  refine ⟨fun h => tendsto_nhds_unique (hf.tendsto_leftLim x) h.tendsto, fun h => ?_⟩
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    x : α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    this : (nhdsWithin x (Set.Iio x)).NeBot
    h : Eq (Function.leftLim f x) (f x)
    ⊢ ContinuousWithinAt f (Set.Iio x) x
  -/
  have := hf.tendsto_leftLim x
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    x : α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    h' : Ne (nhdsWithin x (Set.Iio x)) Bot.bot
    this✝ : (nhdsWithin x (Set.Iio x)).NeBot
    h : Eq (Function.leftLim f x) (f x)
    this : Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (Function.leftLim f x))
    ⊢ ContinuousWithinAt f (Set.Iio x) x
  -/
  rwa [h] at this
  /-
    🎉 no goals
  -/


/-- A monotone function is continuous to the right at a point if and only if its right limit
coincides with the value of the function. -/
theorem continuousWithinAt_Ioi_iff_rightLim_eq :
    ContinuousWithinAt f (Ioi x) x ↔ rightLim f x = f x :=
  hf.dual.continuousWithinAt_Iio_iff_leftLim_eq


/-- A monotone function is continuous at a point if and only if its left and right limits
coincide. -/
theorem continuousAt_iff_leftLim_eq_rightLim : ContinuousAt f x ↔ leftLim f x = rightLim f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : ConditionallyCompleteLinearOrder β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderTopology β
    f : α → β
    hf : Monotone f
    x : α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    ⊢ Iff (ContinuousAt f x) (Eq (Function.leftLim f x) (Function.rightLim f x))
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
  · have A : leftLim f x = f x :=
      hf.continuousWithinAt_Iio_iff_leftLim_eq.1 h.continuousWithinAt
    have B : rightLim f x = f x :=
      hf.continuousWithinAt_Ioi_iff_rightLim_eq.1 h.continuousWithinAt
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : ConditionallyCompleteLinearOrder β
      inst✝³ : TopologicalSpace β
      inst✝² : OrderTopology β
      f : α → β
      hf : Monotone f
      x : α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      h : ContinuousAt f x
      A : Eq (Function.leftLim f x) (f x)
      B : Eq (Function.rightLim f x) (f x)
      ⊢ Eq (Function.leftLim f x) (Function.rightLim f x)
    -/
    exact A.trans B.symm
    /-
      🎉 no goals
    -/
  · have h' : leftLim f x = f x := by
      apply le_antisymm (leftLim_le hf (le_refl _))
      rw [h]
      exact le_rightLim hf (le_refl _)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : ConditionallyCompleteLinearOrder β
      inst✝³ : TopologicalSpace β
      inst✝² : OrderTopology β
      f : α → β
      hf : Monotone f
      x : α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      h : Eq (Function.leftLim f x) (Function.rightLim f x)
      h' : Eq (Function.leftLim f x) (f x)
      ⊢ ContinuousAt f x
    -/
    refine continuousAt_iff_continuous_left'_right'.2 ⟨?_, ?_⟩
      /-
        case refine_2.refine_1
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : ConditionallyCompleteLinearOrder β
        inst✝³ : TopologicalSpace β
        inst✝² : OrderTopology β
        f : α → β
        hf : Monotone f
        x : α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        h : Eq (Function.leftLim f x) (Function.rightLim f x)
        h' : Eq (Function.leftLim f x) (f x)
        ⊢ ContinuousWithinAt f (Set.Iio x) x
      -/
    · exact hf.continuousWithinAt_Iio_iff_leftLim_eq.2 h'
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : ConditionallyCompleteLinearOrder β
        inst✝³ : TopologicalSpace β
        inst✝² : OrderTopology β
        f : α → β
        hf : Monotone f
        x : α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        h : Eq (Function.leftLim f x) (Function.rightLim f x)
        h' : Eq (Function.leftLim f x) (f x)
        ⊢ ContinuousWithinAt f (Set.Ioi x) x
      -/
    · rw [h] at h'
      /-
        case refine_2.refine_2
        α : Type u_1
        β : Type u_2
        inst✝⁵ : LinearOrder α
        inst✝⁴ : ConditionallyCompleteLinearOrder β
        inst✝³ : TopologicalSpace β
        inst✝² : OrderTopology β
        f : α → β
        hf : Monotone f
        x : α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        h : Eq (Function.leftLim f x) (Function.rightLim f x)
        h' : Eq (Function.rightLim f x) (f x)
        ⊢ ContinuousWithinAt f (Set.Ioi x) x
      -/
      exact hf.continuousWithinAt_Ioi_iff_rightLim_eq.2 h'
      /-
        🎉 no goals
      -/


/-- In a second countable space, the set of points where a monotone function is not right-continuous
is at most countable. Superseded by `countable_not_continuousAt` which gives the two-sided
version. -/
theorem countable_not_continuousWithinAt_Ioi [SecondCountableTopology β] :
    Set.Countable { x | ¬ContinuousWithinAt f (Ioi x) x } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    ⊢ (setOf fun x => Not (ContinuousWithinAt f (Set.Ioi x) x)).Countable
  -/
  apply (countable_image_lt_image_Ioi f).mono
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    ⊢ HasSubset.Subset (setOf fun x => Not (ContinuousWithinAt f (Set.Ioi x) x)) ( …
  -/
  rintro x (hx : ¬ContinuousWithinAt f (Ioi x) x)
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : Not (ContinuousWithinAt f (Set.Ioi x) x)
    ⊢ Membership.mem (setOf fun x => Exists fun z => And (LT.lt (f x) z) (∀ (y : α …
  -/
  dsimp
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : Not (ContinuousWithinAt f (Set.Ioi x) x)
    ⊢ Exists fun z => And (LT.lt (f x) z) (∀ (y : α), LT.lt x y → LE.le z (f y))
  -/
  contrapose! hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : ∀ (z : β), LT.lt (f x) z → Exists fun y => And (LT.lt x y) (LT.lt (f y) z)
    ⊢ ContinuousWithinAt f (Set.Ioi x) x
  -/
  refine tendsto_order.2 ⟨fun m hm => ?_, fun u hu => ?_⟩
  · filter_upwards [@self_mem_nhdsWithin _ _ x (Ioi x)] with y hy using hm.trans_le
      (hf (le_of_lt hy))
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : ∀ (z : β), LT.lt (f x) z → Exists fun y => And (LT.lt x y) (LT.lt (f y) z)
    u : β
    hu : GT.gt u (f x)
    ⊢ Filter.Eventually (fun b => LT.lt (f b) u) (nhdsWithin x (Set.Ioi x))
  -/
  rcases hx u hu with ⟨v, xv, fvu⟩
  /-
    case refine_2.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : ∀ (z : β), LT.lt (f x) z → Exists fun y => And (LT.lt x y) (LT.lt (f y) z)
    u : β
    hu : GT.gt u (f x)
    v : α
    xv : LT.lt x v
    fvu : LT.lt (f v) u
    ⊢ Filter.Eventually (fun b => LT.lt (f b) u) (nhdsWithin x (Set.Ioi x))
  -/
  have : Ioo x v ∈ 𝓝[>] x := Ioo_mem_nhdsGT xv
  /-
    case refine_2.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : ∀ (z : β), LT.lt (f x) z → Exists fun y => And (LT.lt x y) (LT.lt (f y) z)
    u : β
    hu : GT.gt u (f x)
    v : α
    xv : LT.lt x v
    fvu : LT.lt (f v) u
    this : Membership.mem (nhdsWithin x (Set.Ioi x)) (Set.Ioo x v)
    ⊢ Filter.Eventually (fun b => LT.lt (f b) u) (nhdsWithin x (Set.Ioi x))
  -/
  filter_upwards [this] with y hy
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : ∀ (z : β), LT.lt (f x) z → Exists fun y => And (LT.lt x y) (LT.lt (f y) z)
    u : β
    hu : GT.gt u (f x)
    v : α
    xv : LT.lt x v
    fvu : LT.lt (f v) u
    this : Membership.mem (nhdsWithin x (Set.Ioi x)) (Set.Ioo x v)
    y : α
    hy : Membership.mem (Set.Ioo x v) y
    ⊢ LT.lt (f y) u
  -/
  apply (hf hy.2.le).trans_lt fvu
  /-
    🎉 no goals
  -/


/-- In a second countable space, the set of points where a monotone function is not left-continuous
is at most countable. Superseded by `countable_not_continuousAt` which gives the two-sided
version. -/
theorem countable_not_continuousWithinAt_Iio [SecondCountableTopology β] :
    Set.Countable { x | ¬ContinuousWithinAt f (Iio x) x } :=
  hf.dual.countable_not_continuousWithinAt_Ioi


/-- In a second countable space, the set of points where a monotone function is not continuous
is at most countable. -/
theorem countable_not_continuousAt [SecondCountableTopology β] :
    Set.Countable { x | ¬ContinuousAt f x } := by
  apply
    (hf.countable_not_continuousWithinAt_Ioi.union hf.countable_not_continuousWithinAt_Iio).mono
      _
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    ⊢ HasSubset.Subset (setOf fun x => Not (ContinuousAt f x)) (Union.union (setOf …
  -/
  refine compl_subset_compl.1 ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    ⊢ HasSubset.Subset (HasCompl.compl (Union.union (setOf fun x => Not (Continuou …
  -/
  simp only [compl_union]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    ⊢ HasSubset.Subset (Inter.inter (HasCompl.compl (setOf fun x => Not (Continuou …
  -/
  rintro x ⟨hx, h'x⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : Membership.mem (HasCompl.compl (setOf fun x => Not (ContinuousWithinAt f  …
    h'x : Membership.mem (HasCompl.compl (setOf fun x => Not (ContinuousWithinAt f …
    ⊢ Membership.mem (HasCompl.compl (setOf fun x => Not (ContinuousAt f x))) x
  -/
  simp only [mem_setOf_eq, Classical.not_not, mem_compl_iff] at hx h'x ⊢
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : ConditionallyCompleteLinearOrder β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderTopology β
    f : α → β
    hf : Monotone f
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology β
    x : α
    hx : ContinuousWithinAt f (Set.Ioi x) x
    h'x : ContinuousWithinAt f (Set.Iio x) x
    ⊢ ContinuousAt f x
  -/
  exact continuousAt_iff_continuous_left'_right'.2 ⟨h'x, hx⟩
  /-
    🎉 no goals
  -/


theorem le_leftLim (h : x ≤ y) : f y ≤ leftLim f x :=
  hf.dual_right.leftLim_le h


theorem leftLim_le (h : x < y) : leftLim f y ≤ f x :=
  hf.dual_right.le_leftLim h


@[mono]
protected theorem leftLim : Antitone (leftLim f) :=
  hf.dual_right.leftLim


theorem rightLim_le (h : x ≤ y) : rightLim f y ≤ f x :=
  hf.dual_right.le_rightLim h


theorem le_rightLim (h : x < y) : f y ≤ rightLim f x :=
  hf.dual_right.rightLim_le h


@[mono]
protected theorem rightLim : Antitone (rightLim f) :=
  hf.dual_right.rightLim


theorem rightLim_le_leftLim (h : x ≤ y) : rightLim f y ≤ leftLim f x :=
  hf.dual_right.leftLim_le_rightLim h


theorem leftLim_le_rightLim (h : x < y) : leftLim f y ≤ rightLim f x :=
  hf.dual_right.rightLim_le_leftLim h


theorem tendsto_leftLim (x : α) : Tendsto f (𝓝[<] x) (𝓝 (leftLim f x)) :=
  hf.dual_right.tendsto_leftLim x


theorem tendsto_leftLim_within (x : α) : Tendsto f (𝓝[<] x) (𝓝[≥] leftLim f x) :=
  hf.dual_right.tendsto_leftLim_within x


theorem tendsto_rightLim (x : α) : Tendsto f (𝓝[>] x) (𝓝 (rightLim f x)) :=
  hf.dual_right.tendsto_rightLim x


theorem tendsto_rightLim_within (x : α) : Tendsto f (𝓝[>] x) (𝓝[≤] rightLim f x) :=
  hf.dual_right.tendsto_rightLim_within x


/-- An antitone function is continuous to the left at a point if and only if its left limit
coincides with the value of the function. -/
theorem continuousWithinAt_Iio_iff_leftLim_eq :
    ContinuousWithinAt f (Iio x) x ↔ leftLim f x = f x :=
  hf.dual_right.continuousWithinAt_Iio_iff_leftLim_eq


/-- An antitone function is continuous to the right at a point if and only if its right limit
coincides with the value of the function. -/
theorem continuousWithinAt_Ioi_iff_rightLim_eq :
    ContinuousWithinAt f (Ioi x) x ↔ rightLim f x = f x :=
  hf.dual_right.continuousWithinAt_Ioi_iff_rightLim_eq


/-- An antitone function is continuous at a point if and only if its left and right limits
coincide. -/
theorem continuousAt_iff_leftLim_eq_rightLim : ContinuousAt f x ↔ leftLim f x = rightLim f x :=
  hf.dual_right.continuousAt_iff_leftLim_eq_rightLim


/-- In a second countable space, the set of points where an antitone function is not continuous
is at most countable. -/
theorem countable_not_continuousAt [SecondCountableTopology β] :
    Set.Countable { x | ¬ContinuousAt f x } :=
  hf.dual_right.countable_not_continuousAt


