/-- Any partial order can be extended to a linear order.
-/
theorem extend_partialOrder {α : Type u} (r : α → α → Prop) [IsPartialOrder α r] :
    ∃ s : α → α → Prop, IsLinearOrder α s ∧ r ≤ s := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    ⊢ Exists fun s => And (IsLinearOrder α s) (LE.le r s)
  -/
  let S := { s | IsPartialOrder α s }
  have hS : ∀ c, c ⊆ S → IsChain (· ≤ ·) c → ∀ y ∈ c, ∃ ub ∈ S, ∀ z ∈ c, z ≤ ub := by
    rintro c hc₁ hc₂ s hs
    haveI := (hc₁ hs).1
    refine ⟨sSup c, ?_, fun z hz => le_sSup hz⟩
    refine
        { refl := ?_
          trans := ?_
          antisymm := ?_ } <;>
      simp_rw [binary_relation_sSup_iff]
    · intro x
      exact ⟨s, hs, refl x⟩
    · rintro x y z ⟨s₁, h₁s₁, h₂s₁⟩ ⟨s₂, h₁s₂, h₂s₂⟩
      haveI : IsPartialOrder _ _ := hc₁ h₁s₁
      haveI : IsPartialOrder _ _ := hc₁ h₁s₂
      cases' hc₂.total h₁s₁ h₁s₂ with h h
      · exact ⟨s₂, h₁s₂, _root_.trans (h _ _ h₂s₁) h₂s₂⟩
      · exact ⟨s₁, h₁s₁, _root_.trans h₂s₁ (h _ _ h₂s₂)⟩
    · rintro x y ⟨s₁, h₁s₁, h₂s₁⟩ ⟨s₂, h₁s₂, h₂s₂⟩
      haveI : IsPartialOrder _ _ := hc₁ h₁s₁
      haveI : IsPartialOrder _ _ := hc₁ h₁s₂
      cases' hc₂.total h₁s₁ h₁s₂ with h h
      · exact antisymm (h _ _ h₂s₁) h₂s₂
      · apply antisymm h₂s₁ (h _ _ h₂s₂)
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
    hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
    ⊢ Exists fun s => And (IsLinearOrder α s) (LE.le r s)
  -/
  obtain ⟨s, hrs, hs⟩ := zorn_le_nonempty₀ S hS r ‹_›
  /-
    case intro.intro
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
    hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
    s : α → α → Prop
    hrs : LE.le r s
    hs : Maximal (fun x => Membership.mem S x) s
    ⊢ Exists fun s => And (IsLinearOrder α s) (LE.le r s)
  -/
  haveI : IsPartialOrder α s := hs.prop
  refine ⟨s,
    { total := ?_, refl := hs.1.refl, trans := hs.1.trans, antisymm := hs.1.antisymm }, hrs⟩
  /-
    case intro.intro
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
    hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
    s : α → α → Prop
    hrs : LE.le r s
    hs : Maximal (fun x => Membership.mem S x) s
    this : IsPartialOrder α s
    ⊢ ∀ (a b : α), Or (s a b) (s b a)
  -/
  intro x y
  /-
    case intro.intro
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
    hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
    s : α → α → Prop
    hrs : LE.le r s
    hs : Maximal (fun x => Membership.mem S x) s
    this : IsPartialOrder α s
    x y : α
    ⊢ Or (s x y) (s y x)
  -/
  by_contra! h
  /-
    case intro.intro
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
    hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
    s : α → α → Prop
    hrs : LE.le r s
    hs : Maximal (fun x => Membership.mem S x) s
    this : IsPartialOrder α s
    x y : α
    h : And (Not (s x y)) (Not (s y x))
    ⊢ False
  -/
  let s' x' y' := s x' y' ∨ s x' x ∧ s y y'
  /-
    case intro.intro
    α : Type u
    r : α → α → Prop
    inst✝ : IsPartialOrder α r
    S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
    hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
    s : α → α → Prop
    hrs : LE.le r s
    hs : Maximal (fun x => Membership.mem S x) s
    this : IsPartialOrder α s
    x y : α
    h : And (Not (s x y)) (Not (s y x))
    s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
    ⊢ False
  -/
  rw [hs.eq_of_le (y := s') ?_ fun _ _ ↦ Or.inl] at h
    /-
      case intro.intro
      α : Type u
      r : α → α → Prop
      inst✝ : IsPartialOrder α r
      S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
      hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
      s : α → α → Prop
      hrs : LE.le r s
      hs : Maximal (fun x => Membership.mem S x) s
      this : IsPartialOrder α s
      x y : α
      s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
      h : And (Not (s' x y)) (Not (s' y x))
      ⊢ False
    -/
  · apply h.1 (Or.inr ⟨refl _, refl _⟩)
    /-
      🎉 no goals
    -/
  · refine
    { refl := fun x ↦ Or.inl (refl _)
      trans := ?_
      antisymm := ?_ }
      /-
        case refine_1
        α : Type u
        r : α → α → Prop
        inst✝ : IsPartialOrder α r
        S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
        hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
        s : α → α → Prop
        hrs : LE.le r s
        hs : Maximal (fun x => Membership.mem S x) s
        this : IsPartialOrder α s
        x y : α
        h : And (Not (s x y)) (Not (s y x))
        s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
        ⊢ ∀ (a b c : α), s' a b → s' b c → s' a c
      -/
    · rintro a b c (ab | ⟨ax : s a x, yb : s y b⟩) (bc | ⟨bx : s b x, yc : s y c⟩)
        /-
          case refine_1.inl.inl
          α : Type u
          r : α → α → Prop
          inst✝ : IsPartialOrder α r
          S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
          hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
          s : α → α → Prop
          hrs : LE.le r s
          hs : Maximal (fun x => Membership.mem S x) s
          this : IsPartialOrder α s
          x y : α
          h : And (Not (s x y)) (Not (s y x))
          s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
          a b c : α
          ab : s a b
          bc : s b c
          ⊢ s' a c
        -/
      · exact Or.inl (_root_.trans ab bc)
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inl.inr.intro
          α : Type u
          r : α → α → Prop
          inst✝ : IsPartialOrder α r
          S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
          hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
          s : α → α → Prop
          hrs : LE.le r s
          hs : Maximal (fun x => Membership.mem S x) s
          this : IsPartialOrder α s
          x y : α
          h : And (Not (s x y)) (Not (s y x))
          s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
          a b c : α
          ab : s a b
          bx : s b x
          yc : s y c
          ⊢ s' a c
        -/
      · exact Or.inr ⟨_root_.trans ab bx, yc⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr.intro.inl
          α : Type u
          r : α → α → Prop
          inst✝ : IsPartialOrder α r
          S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
          hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
          s : α → α → Prop
          hrs : LE.le r s
          hs : Maximal (fun x => Membership.mem S x) s
          this : IsPartialOrder α s
          x y : α
          h : And (Not (s x y)) (Not (s y x))
          s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
          a b c : α
          ax : s a x
          yb : s y b
          bc : s b c
          ⊢ s' a c
        -/
      · exact Or.inr ⟨ax, _root_.trans yb bc⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr.intro.inr.intro
          α : Type u
          r : α → α → Prop
          inst✝ : IsPartialOrder α r
          S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
          hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
          s : α → α → Prop
          hrs : LE.le r s
          hs : Maximal (fun x => Membership.mem S x) s
          this : IsPartialOrder α s
          x y : α
          h : And (Not (s x y)) (Not (s y x))
          s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
          a b c : α
          ax : s a x
          yb : s y b
          bx : s b x
          yc : s y c
          ⊢ s' a c
        -/
      · exact Or.inr ⟨ax, yc⟩
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      α : Type u
      r : α → α → Prop
      inst✝ : IsPartialOrder α r
      S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
      hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
      s : α → α → Prop
      hrs : LE.le r s
      hs : Maximal (fun x => Membership.mem S x) s
      this : IsPartialOrder α s
      x y : α
      h : And (Not (s x y)) (Not (s y x))
      s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
      ⊢ ∀ (a b : α), s' a b → s' b a → Eq a b
    -/
    rintro a b (ab | ⟨ax : s a x, yb : s y b⟩) (ba | ⟨bx : s b x, ya : s y a⟩)
      /-
        case refine_2.inl.inl
        α : Type u
        r : α → α → Prop
        inst✝ : IsPartialOrder α r
        S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
        hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
        s : α → α → Prop
        hrs : LE.le r s
        hs : Maximal (fun x => Membership.mem S x) s
        this : IsPartialOrder α s
        x y : α
        h : And (Not (s x y)) (Not (s y x))
        s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
        a b : α
        ab : s a b
        ba : s b a
        ⊢ Eq a b
      -/
    · exact antisymm ab ba
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inl.inr.intro
        α : Type u
        r : α → α → Prop
        inst✝ : IsPartialOrder α r
        S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
        hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
        s : α → α → Prop
        hrs : LE.le r s
        hs : Maximal (fun x => Membership.mem S x) s
        this : IsPartialOrder α s
        x y : α
        h : And (Not (s x y)) (Not (s y x))
        s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
        a b : α
        ab : s a b
        bx : s b x
        ya : s y a
        ⊢ Eq a b
      -/
    · exact (h.2 (_root_.trans ya (_root_.trans ab bx))).elim
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.inl
        α : Type u
        r : α → α → Prop
        inst✝ : IsPartialOrder α r
        S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
        hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
        s : α → α → Prop
        hrs : LE.le r s
        hs : Maximal (fun x => Membership.mem S x) s
        this : IsPartialOrder α s
        x y : α
        h : And (Not (s x y)) (Not (s y x))
        s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
        a b : α
        ax : s a x
        yb : s y b
        ba : s b a
        ⊢ Eq a b
      -/
    · exact (h.2 (_root_.trans yb (_root_.trans ba ax))).elim
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.inr.intro
        α : Type u
        r : α → α → Prop
        inst✝ : IsPartialOrder α r
        S : Set (α → α → Prop) := setOf fun s => IsPartialOrder α s
        hS : ∀ (c : Set (α → α → Prop)), HasSubset.Subset c S → IsChain (fun x1 x2 =>  …
        s : α → α → Prop
        hrs : LE.le r s
        hs : Maximal (fun x => Membership.mem S x) s
        this : IsPartialOrder α s
        x y : α
        h : And (Not (s x y)) (Not (s y x))
        s' : α → α → Prop := fun x' y' => Or (s x' y') (And (s x' x) (s y y'))
        a b : α
        ax : s a x
        yb : s y b
        bx : s b x
        ya : s y a
        ⊢ Eq a b
      -/
    · exact (h.2 (_root_.trans yb bx)).elim
      /-
        🎉 no goals
      -/


/-- A type alias for `α`, intended to extend a partial order on `α` to a linear order. -/
def LinearExtension (α : Type u) : Type u :=
  α


noncomputable instance {α : Type u} [PartialOrder α] : LinearOrder (LinearExtension α) where
  le := (extend_partialOrder ((· ≤ ·) : α → α → Prop)).choose
  le_refl := (extend_partialOrder ((· ≤ ·) : α → α → Prop)).choose_spec.1.1.1.1.1
  le_trans := (extend_partialOrder ((· ≤ ·) : α → α → Prop)).choose_spec.1.1.1.2.1
  le_antisymm := (extend_partialOrder ((· ≤ ·) : α → α → Prop)).choose_spec.1.1.2.1
  le_total := (extend_partialOrder ((· ≤ ·) : α → α → Prop)).choose_spec.1.2.1
  decidableLE := Classical.decRel _


/-- The embedding of `α` into `LinearExtension α` as an order homomorphism. -/
def toLinearExtension {α : Type u} [PartialOrder α] : α →o LinearExtension α where
  toFun x := x
  monotone' := (extend_partialOrder ((· ≤ ·) : α → α → Prop)).choose_spec.2


instance {α : Type u} [Inhabited α] : Inhabited (LinearExtension α) :=
  ⟨(default : α)⟩

