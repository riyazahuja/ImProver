/-- Quotients of polynomial functors.

Roughly speaking, saying that `F` is a quotient of a polynomial functor means that for each `α`,
elements of `F α` are represented by pairs `⟨a, f⟩`, where `a` is the shape of the object and
`f` indexes the relevant elements of `α`, in a suitably natural manner.
-/
class QPF (F : Type u → Type u) extends Functor F where
  P : PFunctor.{u}
  abs : ∀ {α}, P α → F α
  repr : ∀ {α}, F α → P α
  abs_repr : ∀ {α} (x : F α), abs (repr x) = x
  abs_map : ∀ {α β} (f : α → β) (p : P α), abs (P.map f p) = f <$> abs p


theorem id_map {α : Type _} (x : F α) : id <$> x = x := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    ⊢ Eq (Functor.map id x) x
  -/
  rw [← abs_repr x]
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    ⊢ Eq (Functor.map id (QPF.abs (QPF.repr x))) (QPF.abs (QPF.repr x))
  -/
  cases' repr x with a f
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ Eq (Functor.map id (QPF.abs ⟨a, f⟩)) (QPF.abs ⟨a, f⟩)
  -/
  rw [← abs_map]
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ Eq (QPF.abs ((QPF.P F).map id ⟨a, f⟩)) (QPF.abs ⟨a, f⟩)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comp_map {α β γ : Type _} (f : α → β) (g : β → γ) (x : F α) :
    (g ∘ f) <$> x = g <$> f <$> x := by
  /-
    F : Type u → Type u
    q : QPF F
    α β γ : Type u
    f : α → β
    g : β → γ
    x : F α
    ⊢ Eq (Functor.map (Function.comp g f) x) (Functor.map g (Functor.map f x))
  -/
  rw [← abs_repr x]
  /-
    F : Type u → Type u
    q : QPF F
    α β γ : Type u
    f : α → β
    g : β → γ
    x : F α
    ⊢ Eq (Functor.map (Function.comp g f) (QPF.abs (QPF.repr x))) (Functor.map g ( …
  -/
  cases' repr x with a f
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    α β γ : Type u
    f✝ : α → β
    g : β → γ
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ Eq (Functor.map (Function.comp g f✝) (QPF.abs ⟨a, f⟩)) (Functor.map g (Funct …
  -/
  rw [← abs_map, ← abs_map, ← abs_map]
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    α β γ : Type u
    f✝ : α → β
    g : β → γ
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ Eq (QPF.abs ((QPF.P F).map (Function.comp g f✝) ⟨a, f⟩)) (QPF.abs ((QPF.P F) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem lawfulFunctor
    (h : ∀ α β : Type u, @Functor.mapConst F _ α _ = Functor.map ∘ Function.const β) :
    LawfulFunctor F :=
  { map_const := @h
    id_map := @id_map F _
    comp_map := @comp_map F _ }

/-
Lifting predicates and relations
-/

theorem liftp_iff {α : Type u} (p : α → Prop) (x : F α) :
    Liftp p x ↔ ∃ a f, x = abs ⟨a, f⟩ ∧ ∀ i, p (f i) := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    ⊢ Iff (Functor.Liftp p x) (Exists fun a => Exists fun f => And (Eq x (QPF.abs  …
  -/
  constructor
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      ⊢ Functor.Liftp p x → Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f …
    -/
  · rintro ⟨y, hy⟩
    /-
      case mp.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      ⊢ Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QPF.P F …
    -/
    cases' h : repr y with a f
    /-
      case mp.intro.mk
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QPF.P F …
    -/
    use a, fun i => (f i).val
    /-
      case h
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ And (Eq x (QPF.abs ⟨a, fun i => ↑(f i)⟩)) (∀ (i : (QPF.P F).B a), p ((fun i  …
    -/
    constructor
      /-
        case h.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        p : α → Prop
        x : F α
        y : F (Subtype p)
        hy : Eq (Functor.map Subtype.val y) x
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype p
        h : Eq (QPF.repr y) ⟨a, f⟩
        ⊢ Eq x (QPF.abs ⟨a, fun i => ↑(f i)⟩)
      -/
    · rw [← hy, ← abs_repr y, h, ← abs_map]
      /-
        case h.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        p : α → Prop
        x : F α
        y : F (Subtype p)
        hy : Eq (Functor.map Subtype.val y) x
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype p
        h : Eq (QPF.repr y) ⟨a, f⟩
        ⊢ Eq (QPF.abs ((QPF.P F).map Subtype.val ⟨a, f⟩)) (QPF.abs ⟨a, fun i => ↑(f i)⟩)
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ ∀ (i : (QPF.P F).B a), p ↑(f i)
    -/
    intro i
    /-
      case h.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      i : (QPF.P F).B a
      ⊢ p ↑(f i)
    -/
    apply (f i).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    ⊢ (Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QPF.P  …
  -/
  rintro ⟨a, f, h₀, h₁⟩
  /-
    case mpr.intro.intro.intro
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h₀ : Eq x (QPF.abs ⟨a, f⟩)
    h₁ : ∀ (i : (QPF.P F).B a), p (f i)
    ⊢ Functor.Liftp p x
  -/
  use abs ⟨a, fun i => ⟨f i, h₁ i⟩⟩
  /-
    case h
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h₀ : Eq x (QPF.abs ⟨a, f⟩)
    h₁ : ∀ (i : (QPF.P F).B a), p (f i)
    ⊢ Eq (Functor.map Subtype.val (QPF.abs ⟨a, fun i => ⟨f i, ⋯⟩⟩)) x
  -/
  rw [← abs_map, h₀]; rfl
                      /-
                        🎉 no goals
                      -/


theorem liftp_iff' {α : Type u} (p : α → Prop) (x : F α) :
    Liftp p x ↔ ∃ u : q.P α, abs u = x ∧ ∀ i, p (u.snd i) := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    ⊢ Iff (Functor.Liftp p x) (Exists fun u => And (Eq (QPF.abs u) x) (∀ (i : (QPF …
  -/
  constructor
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      ⊢ Functor.Liftp p x → Exists fun u => And (Eq (QPF.abs u) x) (∀ (i : (QPF.P F) …
    -/
  · rintro ⟨y, hy⟩
    /-
      case mp.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      ⊢ Exists fun u => And (Eq (QPF.abs u) x) (∀ (i : (QPF.P F).B u.fst), p (u.snd  …
    -/
    cases' h : repr y with a f
    /-
      case mp.intro.mk
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ Exists fun u => And (Eq (QPF.abs u) x) (∀ (i : (QPF.P F).B u.fst), p (u.snd  …
    -/
    use ⟨a, fun i => (f i).val⟩
    /-
      case h
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ And (Eq (QPF.abs ⟨a, fun i => ↑(f i)⟩) x) (∀ (i : (QPF.P F).B ⟨a, fun i => ↑ …
    -/
    dsimp
    /-
      case h
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ And (Eq (QPF.abs ⟨a, fun i => ↑(f i)⟩) x) (∀ (i : (QPF.P F).B a), p ↑(f i))
    -/
    constructor
      /-
        case h.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        p : α → Prop
        x : F α
        y : F (Subtype p)
        hy : Eq (Functor.map Subtype.val y) x
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype p
        h : Eq (QPF.repr y) ⟨a, f⟩
        ⊢ Eq (QPF.abs ⟨a, fun i => ↑(f i)⟩) x
      -/
    · rw [← hy, ← abs_repr y, h, ← abs_map]
      /-
        case h.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        p : α → Prop
        x : F α
        y : F (Subtype p)
        hy : Eq (Functor.map Subtype.val y) x
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype p
        h : Eq (QPF.repr y) ⟨a, f⟩
        ⊢ Eq (QPF.abs ⟨a, fun i => ↑(f i)⟩) (QPF.abs ((QPF.P F).map Subtype.val ⟨a, f⟩))
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      ⊢ ∀ (i : (QPF.P F).B a), p ↑(f i)
    -/
    intro i
    /-
      case h.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      p : α → Prop
      x : F α
      y : F (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype p
      h : Eq (QPF.repr y) ⟨a, f⟩
      i : (QPF.P F).B a
      ⊢ p ↑(f i)
    -/
    apply (f i).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    ⊢ (Exists fun u => And (Eq (QPF.abs u) x) (∀ (i : (QPF.P F).B u.fst), p (u.snd …
  -/
  rintro ⟨⟨a, f⟩, h₀, h₁⟩; dsimp at *
  /-
    case mpr.intro.mk.intro
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h₀ : Eq (QPF.abs ⟨a, f⟩) x
    h₁ : ∀ (i : (QPF.P F).B a), p (f i)
    ⊢ Functor.Liftp p x
  -/
  use abs ⟨a, fun i => ⟨f i, h₁ i⟩⟩
  /-
    case h
    F : Type u → Type u
    q : QPF F
    α : Type u
    p : α → Prop
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h₀ : Eq (QPF.abs ⟨a, f⟩) x
    h₁ : ∀ (i : (QPF.P F).B a), p (f i)
    ⊢ Eq (Functor.map Subtype.val (QPF.abs ⟨a, fun i => ⟨f i, ⋯⟩⟩)) x
  -/
  rw [← abs_map, ← h₀]; rfl
                        /-
                          🎉 no goals
                        -/


theorem liftr_iff {α : Type u} (r : α → α → Prop) (x y : F α) :
    Liftr r x y ↔ ∃ a f₀ f₁, x = abs ⟨a, f₀⟩ ∧ y = abs ⟨a, f₁⟩ ∧ ∀ i, r (f₀ i) (f₁ i) := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    r : α → α → Prop
    x y : F α
    ⊢ Iff (Functor.Liftr r x y) (Exists fun a => Exists fun f₀ => Exists fun f₁ => …
  -/
  constructor
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      ⊢ Functor.Liftr r x y → Exists fun a => Exists fun f₀ => Exists fun f₁ => And  …
    -/
  · rintro ⟨u, xeq, yeq⟩
    /-
      case mp.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      u : F (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x (QPF.abs ⟨a, f₀⟩ …
    -/
    cases' h : repr u with a f
    /-
      case mp.intro.intro.mk
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      u : F (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
      h : Eq (QPF.repr u) ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x (QPF.abs ⟨a, f₀⟩ …
    -/
    use a, fun i => (f i).val.fst, fun i => (f i).val.snd
    /-
      case h
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      u : F (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
      h : Eq (QPF.repr u) ⟨a, f⟩
      ⊢ And (Eq x (QPF.abs ⟨a, fun i => (↑(f i)).1⟩)) (And (Eq y (QPF.abs ⟨a, fun i  …
    -/
    constructor
      /-
        case h.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        r : α → α → Prop
        x y : F α
        u : F (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
        h : Eq (QPF.repr u) ⟨a, f⟩
        ⊢ Eq x (QPF.abs ⟨a, fun i => (↑(f i)).1⟩)
      -/
    · rw [← xeq, ← abs_repr u, h, ← abs_map]
      /-
        case h.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        r : α → α → Prop
        x y : F α
        u : F (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
        h : Eq (QPF.repr u) ⟨a, f⟩
        ⊢ Eq (QPF.abs ((QPF.P F).map (fun t => (↑t).1) ⟨a, f⟩)) (QPF.abs ⟨a, fun i =>  …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      u : F (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
      h : Eq (QPF.repr u) ⟨a, f⟩
      ⊢ And (Eq y (QPF.abs ⟨a, fun i => (↑(f i)).2⟩)) (∀ (i : (QPF.P F).B a), r (↑(f …
    -/
    constructor
      /-
        case h.right.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        r : α → α → Prop
        x y : F α
        u : F (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
        h : Eq (QPF.repr u) ⟨a, f⟩
        ⊢ Eq y (QPF.abs ⟨a, fun i => (↑(f i)).2⟩)
      -/
    · rw [← yeq, ← abs_repr u, h, ← abs_map]
      /-
        case h.right.left
        F : Type u → Type u
        q : QPF F
        α : Type u
        r : α → α → Prop
        x y : F α
        u : F (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : (QPF.P F).A
        f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
        h : Eq (QPF.repr u) ⟨a, f⟩
        ⊢ Eq (QPF.abs ((QPF.P F).map (fun t => (↑t).2) ⟨a, f⟩)) (QPF.abs ⟨a, fun i =>  …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      u : F (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
      h : Eq (QPF.repr u) ⟨a, f⟩
      ⊢ ∀ (i : (QPF.P F).B a), r (↑(f i)).1 (↑(f i)).2
    -/
    intro i
    /-
      case h.right.right
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      u : F (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : (QPF.P F).A
      f : (QPF.P F).B a → Subtype fun p => r p.1 p.2
      h : Eq (QPF.repr u) ⟨a, f⟩
      i : (QPF.P F).B a
      ⊢ r (↑(f i)).1 (↑(f i)).2
    -/
    exact (f i).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    r : α → α → Prop
    x y : F α
    ⊢ (Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x (QPF.abs ⟨a, f₀ …
  -/
  rintro ⟨a, f₀, f₁, xeq, yeq, h⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    F : Type u → Type u
    q : QPF F
    α : Type u
    r : α → α → Prop
    x y : F α
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → α
    xeq : Eq x (QPF.abs ⟨a, f₀⟩)
    yeq : Eq y (QPF.abs ⟨a, f₁⟩)
    h : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    ⊢ Functor.Liftr r x y
  -/
  use abs ⟨a, fun i => ⟨(f₀ i, f₁ i), h i⟩⟩
  /-
    case h
    F : Type u → Type u
    q : QPF F
    α : Type u
    r : α → α → Prop
    x y : F α
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → α
    xeq : Eq x (QPF.abs ⟨a, f₀⟩)
    yeq : Eq y (QPF.abs ⟨a, f₁⟩)
    h : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    ⊢ And (Eq (Functor.map (fun t => (↑t).1) (QPF.abs ⟨a, fun i => ⟨{ fst := f₀ i, …
  -/
  constructor
    /-
      case h.left
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      a : (QPF.P F).A
      f₀ f₁ : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f₀⟩)
      yeq : Eq y (QPF.abs ⟨a, f₁⟩)
      h : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
      ⊢ Eq (Functor.map (fun t => (↑t).1) (QPF.abs ⟨a, fun i => ⟨{ fst := f₀ i, snd  …
    -/
  · rw [xeq, ← abs_map]
    /-
      case h.left
      F : Type u → Type u
      q : QPF F
      α : Type u
      r : α → α → Prop
      x y : F α
      a : (QPF.P F).A
      f₀ f₁ : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f₀⟩)
      yeq : Eq y (QPF.abs ⟨a, f₁⟩)
      h : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
      ⊢ Eq (QPF.abs ((QPF.P F).map (fun t => (↑t).1) ⟨a, fun i => ⟨{ fst := f₀ i, sn …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case h.right
    F : Type u → Type u
    q : QPF F
    α : Type u
    r : α → α → Prop
    x y : F α
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → α
    xeq : Eq x (QPF.abs ⟨a, f₀⟩)
    yeq : Eq y (QPF.abs ⟨a, f₁⟩)
    h : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    ⊢ Eq (Functor.map (fun t => (↑t).2) (QPF.abs ⟨a, fun i => ⟨{ fst := f₀ i, snd  …
  -/
  rw [yeq, ← abs_map]; rfl
                       /-
                         🎉 no goals
                       -/


/-- does recursion on `q.P.W` using `g : F α → α` rather than `g : P α → α` -/
def recF {α : Type _} (g : F α → α) : q.P.W → α
  | ⟨a, f⟩ => g (abs ⟨a, fun x => recF g (f x)⟩)


theorem recF_eq {α : Type _} (g : F α → α) (x : q.P.W) :
    recF g x = g (abs (q.P.map (recF g) x.dest)) := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    x : (QPF.P F).W
    ⊢ Eq (QPF.recF g x) (g (QPF.abs ((QPF.P F).map (QPF.recF g) x.dest)))
  -/
  cases x
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    a✝ : (QPF.P F).A
    f✝ : (QPF.P F).B a✝ → WType (QPF.P F).B
    ⊢ Eq (QPF.recF g (WType.mk a✝ f✝)) (g (QPF.abs ((QPF.P F).map (QPF.recF g) (PF …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem recF_eq' {α : Type _} (g : F α → α) (a : q.P.A) (f : q.P.B a → q.P.W) :
    recF g ⟨a, f⟩ = g (abs (q.P.map (recF g) ⟨a, f⟩)) :=
  rfl


/-- two trees are equivalent if their F-abstractions are -/
inductive Wequiv : q.P.W → q.P.W → Prop
  | ind (a : q.P.A) (f f' : q.P.B a → q.P.W) : (∀ x, Wequiv (f x) (f' x)) → Wequiv ⟨a, f⟩ ⟨a, f'⟩
  | abs (a : q.P.A) (f : q.P.B a → q.P.W) (a' : q.P.A) (f' : q.P.B a' → q.P.W) :
      abs ⟨a, f⟩ = abs ⟨a', f'⟩ → Wequiv ⟨a, f⟩ ⟨a', f'⟩
  | trans (u v w : q.P.W) : Wequiv u v → Wequiv v w → Wequiv u w


/-- `recF` is insensitive to the representation -/
theorem recF_eq_of_Wequiv {α : Type u} (u : F α → α) (x y : q.P.W) :
    Wequiv x y → recF u x = recF u y := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    u : F α → α
    x y : (QPF.P F).W
    ⊢ QPF.Wequiv x y → Eq (QPF.recF u x) (QPF.recF u y)
  -/
  intro h
  induction h with
  | ind a f f' _ ih => simp only [recF_eq', PFunctor.map_eq, Function.comp_def, ih]
  | abs a f a' f' h => simp only [recF_eq', abs_map, h]
  | trans x y z _ _ ih₁ ih₂ => exact Eq.trans ih₁ ih₂


theorem Wequiv.abs' (x y : q.P.W) (h : QPF.abs x.dest = QPF.abs y.dest) : Wequiv x y := by
  /-
    F : Type u → Type u
    q : QPF F
    x y : (QPF.P F).W
    h : Eq (QPF.abs x.dest) (QPF.abs y.dest)
    ⊢ QPF.Wequiv x y
  -/
  cases x
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    y : (QPF.P F).W
    a✝ : (QPF.P F).A
    f✝ : (QPF.P F).B a✝ → WType (QPF.P F).B
    h : Eq (QPF.abs (PFunctor.W.dest (WType.mk a✝ f✝))) (QPF.abs y.dest)
    ⊢ QPF.Wequiv (WType.mk a✝ f✝) y
  -/
  cases y
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    a✝¹ : (QPF.P F).A
    f✝¹ : (QPF.P F).B a✝¹ → WType (QPF.P F).B
    a✝ : (QPF.P F).A
    f✝ : (QPF.P F).B a✝ → WType (QPF.P F).B
    h : Eq (QPF.abs (PFunctor.W.dest (WType.mk a✝¹ f✝¹))) (QPF.abs (PFunctor.W.des …
    ⊢ QPF.Wequiv (WType.mk a✝¹ f✝¹) (WType.mk a✝ f✝)
  -/
  apply Wequiv.abs
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    a✝¹ : (QPF.P F).A
    f✝¹ : (QPF.P F).B a✝¹ → WType (QPF.P F).B
    a✝ : (QPF.P F).A
    f✝ : (QPF.P F).B a✝ → WType (QPF.P F).B
    h : Eq (QPF.abs (PFunctor.W.dest (WType.mk a✝¹ f✝¹))) (QPF.abs (PFunctor.W.des …
    ⊢ Eq (QPF.abs ⟨a✝¹, f✝¹⟩) (QPF.abs ⟨a✝, f✝⟩)
  -/
  apply h
  /-
    🎉 no goals
  -/


theorem Wequiv.refl (x : q.P.W) : Wequiv x x := by
  /-
    F : Type u → Type u
    q : QPF F
    x : (QPF.P F).W
    ⊢ QPF.Wequiv x x
  -/
  cases' x with a f
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ⊢ QPF.Wequiv (WType.mk a f) (WType.mk a f)
  -/
  exact Wequiv.abs a f a f rfl
  /-
    🎉 no goals
  -/


theorem Wequiv.symm (x y : q.P.W) : Wequiv x y → Wequiv y x := by
  /-
    F : Type u → Type u
    q : QPF F
    x y : (QPF.P F).W
    ⊢ QPF.Wequiv x y → QPF.Wequiv y x
  -/
  intro h
  induction h with
  | ind a f f' _ ih => exact Wequiv.ind _ _ _ ih
  | abs a f a' f' h => exact Wequiv.abs _ _ _ _ h.symm
  | trans x y z _ _ ih₁ ih₂ => exact QPF.Wequiv.trans _ _ _ ih₂ ih₁


/-- maps every element of the W type to a canonical representative -/
def Wrepr : q.P.W → q.P.W :=
  recF (PFunctor.W.mk ∘ repr)


theorem Wrepr_equiv (x : q.P.W) : Wequiv (Wrepr x) x := by
  /-
    F : Type u → Type u
    q : QPF F
    x : (QPF.P F).W
    ⊢ QPF.Wequiv (QPF.Wrepr x) x
  -/
  induction' x with a f ih
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
    ⊢ QPF.Wequiv (QPF.Wrepr (WType.mk a f)) (WType.mk a f)
  -/
  apply Wequiv.trans
    /-
      case mk.a
      F : Type u → Type u
      q : QPF F
      a : (QPF.P F).A
      f : (QPF.P F).B a → WType (QPF.P F).B
      ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
      ⊢ QPF.Wequiv (QPF.Wrepr (WType.mk a f)) ?mk.v
    -/
  · change Wequiv (Wrepr ⟨a, f⟩) (PFunctor.W.mk (q.P.map Wrepr ⟨a, f⟩))
    /-
      case mk.a
      F : Type u → Type u
      q : QPF F
      a : (QPF.P F).A
      f : (QPF.P F).B a → WType (QPF.P F).B
      ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
      ⊢ QPF.Wequiv (QPF.Wrepr (WType.mk a f)) (PFunctor.W.mk ((QPF.P F).map QPF.Wrep …
    -/
    apply Wequiv.abs'
    /-
      case mk.a.h
      F : Type u → Type u
      q : QPF F
      a : (QPF.P F).A
      f : (QPF.P F).B a → WType (QPF.P F).B
      ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
      ⊢ Eq (QPF.abs (QPF.Wrepr (WType.mk a f)).dest) (QPF.abs (PFunctor.W.mk ((QPF.P …
    -/
    have : Wrepr ⟨a, f⟩ = PFunctor.W.mk (repr (abs (q.P.map Wrepr ⟨a, f⟩))) := rfl
    /-
      case mk.a.h
      F : Type u → Type u
      q : QPF F
      a : (QPF.P F).A
      f : (QPF.P F).B a → WType (QPF.P F).B
      ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
      this : Eq (QPF.Wrepr (WType.mk a f)) (PFunctor.W.mk (QPF.repr (QPF.abs ((QPF.P …
      ⊢ Eq (QPF.abs (QPF.Wrepr (WType.mk a f)).dest) (QPF.abs (PFunctor.W.mk ((QPF.P …
    -/
    rw [this, PFunctor.W.dest_mk, abs_repr]
    /-
      case mk.a.h
      F : Type u → Type u
      q : QPF F
      a : (QPF.P F).A
      f : (QPF.P F).B a → WType (QPF.P F).B
      ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
      this : Eq (QPF.Wrepr (WType.mk a f)) (PFunctor.W.mk (QPF.repr (QPF.abs ((QPF.P …
      ⊢ Eq (QPF.abs ((QPF.P F).map QPF.Wrepr ⟨a, f⟩)) (QPF.abs (PFunctor.W.mk ((QPF. …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case mk.a
    F : Type u → Type u
    q : QPF F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), QPF.Wequiv (QPF.Wrepr (f a)) (f a)
    ⊢ QPF.Wequiv (PFunctor.W.mk ((QPF.P F).map QPF.Wrepr ⟨a, f⟩)) (WType.mk a f)
  -/
  apply Wequiv.ind; exact ih
                    /-
                      🎉 no goals
                    -/


/-- Define the fixed point as the quotient of trees under the equivalence relation `Wequiv`. -/
def Wsetoid : Setoid q.P.W :=
  ⟨Wequiv, @Wequiv.refl _ _, @Wequiv.symm _ _, @Wequiv.trans _ _⟩


/-- inductive type defined as initial algebra of a Quotient of Polynomial Functor -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
def Fix (F : Type u → Type u) [q : QPF F] :=
  Quotient (Wsetoid : Setoid q.P.W)


/-- recursor of a type defined by a qpf -/
def Fix.rec {α : Type _} (g : F α → α) : Fix F → α :=
  Quot.lift (recF g) (recF_eq_of_Wequiv g)


/-- access the underlying W-type of a fixpoint data type -/
def fixToW : Fix F → q.P.W :=
  Quotient.lift Wrepr (recF_eq_of_Wequiv fun x => @PFunctor.W.mk q.P (repr x))


/-- constructor of a type defined by a qpf -/
def Fix.mk (x : F (Fix F)) : Fix F :=
  Quot.mk _ (PFunctor.W.mk (q.P.map fixToW (repr x)))


/-- destructor of a type defined by a qpf -/
def Fix.dest : Fix F → F (Fix F) :=
  Fix.rec (Functor.map Fix.mk)


theorem Fix.rec_eq {α : Type _} (g : F α → α) (x : F (Fix F)) :
    Fix.rec g (Fix.mk x) = g (Fix.rec g <$> x) := by
  have : recF g ∘ fixToW = Fix.rec g := by
    ext ⟨x⟩
    apply recF_eq_of_Wequiv
    rw [fixToW]
    apply Wrepr_equiv
  conv =>
    lhs
    rw [Fix.rec, Fix.mk]
    dsimp
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    x : F (QPF.Fix F)
    this : Eq (Function.comp (QPF.recF g) QPF.fixToW) (QPF.Fix.rec g)
    ⊢ Eq (QPF.recF g (PFunctor.W.mk ((QPF.P F).map QPF.fixToW (QPF.repr x)))) (g ( …
  -/
  cases' h : repr x with a f
  rw [PFunctor.map_eq, recF_eq, ← PFunctor.map_eq, PFunctor.W.dest_mk, PFunctor.map_map, abs_map,
    ← h, abs_repr, this]


theorem Fix.ind_aux (a : q.P.A) (f : q.P.B a → q.P.W) :
    Fix.mk (abs ⟨a, fun x => ⟦f x⟧⟩) = ⟦⟨a, f⟩⟧ := by
  have : Fix.mk (abs ⟨a, fun x => ⟦f x⟧⟩) = ⟦Wrepr ⟨a, f⟩⟧ := by
    apply Quot.sound; apply Wequiv.abs'
    rw [PFunctor.W.dest_mk, abs_map, abs_repr, ← abs_map, PFunctor.map_eq]
    simp only [Wrepr, recF_eq, PFunctor.W.dest_mk, abs_repr, Function.comp]
    rfl
  /-
    F : Type u → Type u
    q : QPF F
    a : (QPF.P F).A
    f : (QPF.P F).B a → (QPF.P F).W
    this : Eq (QPF.Fix.mk (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩)) ( …
    ⊢ Eq (QPF.Fix.mk (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩)) (Quoti …
  -/
  rw [this]
  /-
    F : Type u → Type u
    q : QPF F
    a : (QPF.P F).A
    f : (QPF.P F).B a → (QPF.P F).W
    this : Eq (QPF.Fix.mk (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩)) ( …
    ⊢ Eq (Quotient.mk QPF.Wsetoid (QPF.Wrepr (WType.mk a f))) (Quotient.mk QPF.Wse …
  -/
  apply Quot.sound
  /-
    case a
    F : Type u → Type u
    q : QPF F
    a : (QPF.P F).A
    f : (QPF.P F).B a → (QPF.P F).W
    this : Eq (QPF.Fix.mk (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩)) ( …
    ⊢ QPF.Wsetoid (QPF.Wrepr (WType.mk a f)) (WType.mk a f)
  -/
  apply Wrepr_equiv
  /-
    🎉 no goals
  -/


theorem Fix.ind_rec {α : Type u} (g₁ g₂ : Fix F → α)
    (h : ∀ x : F (Fix F), g₁ <$> x = g₂ <$> x → g₁ (Fix.mk x) = g₂ (Fix.mk x)) :
    ∀ x, g₁ x = g₂ x := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    ⊢ ∀ (x : QPF.Fix F), Eq (g₁ x) (g₂ x)
  -/
  rintro ⟨x⟩
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    x✝ : QPF.Fix F
    x : (QPF.P F).W
    ⊢ Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) x)) (g₂ (Quot.mk (⇑QPF.Wsetoid) x))
  -/
  induction' x with a f ih
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) (f a))) (g₂ (Quot.m …
    ⊢ Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) (WType.mk a f))) (g₂ (Quot.mk (⇑QPF.Wsetoid)  …
  -/
  change g₁ ⟦⟨a, f⟩⟧ = g₂ ⟦⟨a, f⟩⟧
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) (f a))) (g₂ (Quot.m …
    ⊢ Eq (g₁ (Quotient.mk QPF.Wsetoid (WType.mk a f))) (g₂ (Quotient.mk QPF.Wsetoi …
  -/
  rw [← Fix.ind_aux a f]; apply h
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) (f a))) (g₂ (Quot.m …
    ⊢ Eq (Functor.map g₁ (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩)) (F …
  -/
  rw [← abs_map, ← abs_map, PFunctor.map_eq, PFunctor.map_eq]
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) (f a))) (g₂ (Quot.m …
    ⊢ Eq (QPF.abs ⟨a, Function.comp g₁ fun x => Quotient.mk QPF.Wsetoid (f x)⟩) (Q …
  -/
  congr with x
  /-
    case mk.mk.a.e_a.e_snd.h
    F : Type u → Type u
    q : QPF F
    α : Type u
    g₁ g₂ : QPF.Fix F → α
    h : ∀ (x : F (QPF.Fix F)), Eq (Functor.map g₁ x) (Functor.map g₂ x) → Eq (g₁ ( …
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), Eq (g₁ (Quot.mk (⇑QPF.Wsetoid) (f a))) (g₂ (Quot.m …
    x : (QPF.P F).B a
    ⊢ Eq (Function.comp g₁ (fun x => Quotient.mk QPF.Wsetoid (f x)) x) (Function.c …
  -/
  apply ih
  /-
    🎉 no goals
  -/


theorem Fix.rec_unique {α : Type u} (g : F α → α) (h : Fix F → α)
    (hyp : ∀ x, h (Fix.mk x) = g (h <$> x)) : Fix.rec g = h := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    h : QPF.Fix F → α
    hyp : ∀ (x : F (QPF.Fix F)), Eq (h (QPF.Fix.mk x)) (g (Functor.map h x))
    ⊢ Eq (QPF.Fix.rec g) h
  -/
  ext x
  /-
    case h
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    h : QPF.Fix F → α
    hyp : ∀ (x : F (QPF.Fix F)), Eq (h (QPF.Fix.mk x)) (g (Functor.map h x))
    x : QPF.Fix F
    ⊢ Eq (QPF.Fix.rec g x) (h x)
  -/
  apply Fix.ind_rec
  /-
    case h.h
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    h : QPF.Fix F → α
    hyp : ∀ (x : F (QPF.Fix F)), Eq (h (QPF.Fix.mk x)) (g (Functor.map h x))
    x : QPF.Fix F
    ⊢ ∀ (x : F (QPF.Fix F)), Eq (Functor.map (QPF.Fix.rec g) x) (Functor.map h x)  …
  -/
  intro x hyp'
  /-
    case h.h
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : F α → α
    h : QPF.Fix F → α
    hyp : ∀ (x : F (QPF.Fix F)), Eq (h (QPF.Fix.mk x)) (g (Functor.map h x))
    x✝ : QPF.Fix F
    x : F (QPF.Fix F)
    hyp' : Eq (Functor.map (QPF.Fix.rec g) x) (Functor.map h x)
    ⊢ Eq (QPF.Fix.rec g (QPF.Fix.mk x)) (h (QPF.Fix.mk x))
  -/
  rw [hyp, ← hyp', Fix.rec_eq]
  /-
    🎉 no goals
  -/


theorem Fix.mk_dest (x : Fix F) : Fix.mk (Fix.dest x) = x := by
  /-
    F : Type u → Type u
    q : QPF F
    x : QPF.Fix F
    ⊢ Eq (QPF.Fix.mk x.dest) x
  -/
  change (Fix.mk ∘ Fix.dest) x = id x
  /-
    F : Type u → Type u
    q : QPF F
    x : QPF.Fix F
    ⊢ Eq (Function.comp QPF.Fix.mk QPF.Fix.dest x) (id x)
  -/
  apply Fix.ind_rec (mk ∘ dest) id
  /-
    case h
    F : Type u → Type u
    q : QPF F
    x : QPF.Fix F
    ⊢ ∀ (x : F (QPF.Fix F)), Eq (Functor.map (Function.comp QPF.Fix.mk QPF.Fix.des …
  -/
  intro x
  /-
    case h
    F : Type u → Type u
    q : QPF F
    x✝ : QPF.Fix F
    x : F (QPF.Fix F)
    ⊢ Eq (Functor.map (Function.comp QPF.Fix.mk QPF.Fix.dest) x) (Functor.map id x …
  -/
  rw [Function.comp_apply, id_eq, Fix.dest, Fix.rec_eq, id_map, comp_map]
  /-
    case h
    F : Type u → Type u
    q : QPF F
    x✝ : QPF.Fix F
    x : F (QPF.Fix F)
    ⊢ Eq (Functor.map QPF.Fix.mk (Functor.map (QPF.Fix.rec (Functor.map QPF.Fix.mk …
  -/
  intro h
  /-
    case h
    F : Type u → Type u
    q : QPF F
    x✝ : QPF.Fix F
    x : F (QPF.Fix F)
    h : Eq (Functor.map QPF.Fix.mk (Functor.map (QPF.Fix.rec (Functor.map QPF.Fix. …
    ⊢ Eq (QPF.Fix.mk (Functor.map QPF.Fix.mk (Functor.map (QPF.Fix.rec (Functor.ma …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem Fix.dest_mk (x : F (Fix F)) : Fix.dest (Fix.mk x) = x := by
  /-
    F : Type u → Type u
    q : QPF F
    x : F (QPF.Fix F)
    ⊢ Eq (QPF.Fix.mk x).dest x
  -/
  unfold Fix.dest; rw [Fix.rec_eq, ← Fix.dest, ← comp_map]
  conv =>
    rhs
    rw [← id_map x]
  /-
    F : Type u → Type u
    q : QPF F
    x : F (QPF.Fix F)
    ⊢ Eq (Functor.map (Function.comp QPF.Fix.mk QPF.Fix.dest) x) (Functor.map id x)
  -/
  congr with x
  /-
    case e_a.h
    F : Type u → Type u
    q : QPF F
    x✝ : F (QPF.Fix F)
    x : QPF.Fix F
    ⊢ Eq (Function.comp QPF.Fix.mk QPF.Fix.dest x) (id x)
  -/
  apply Fix.mk_dest
  /-
    🎉 no goals
  -/


theorem Fix.ind (p : Fix F → Prop) (h : ∀ x : F (Fix F), Liftp p x → p (Fix.mk x)) : ∀ x, p x := by
  /-
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    ⊢ ∀ (x : QPF.Fix F), p x
  -/
  rintro ⟨x⟩
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    x : (QPF.P F).W
    ⊢ p (Quot.mk (⇑QPF.Wsetoid) x)
  -/
  induction' x with a f ih
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), p (Quot.mk (⇑QPF.Wsetoid) (f a))
    ⊢ p (Quot.mk (⇑QPF.Wsetoid) (WType.mk a f))
  -/
  change p ⟦⟨a, f⟩⟧
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), p (Quot.mk (⇑QPF.Wsetoid) (f a))
    ⊢ p (Quotient.mk QPF.Wsetoid (WType.mk a f))
  -/
  rw [← Fix.ind_aux a f]
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), p (Quot.mk (⇑QPF.Wsetoid) (f a))
    ⊢ p (QPF.Fix.mk (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩))
  -/
  apply h
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), p (Quot.mk (⇑QPF.Wsetoid) (f a))
    ⊢ Functor.Liftp p (QPF.abs ⟨a, fun x => Quotient.mk QPF.Wsetoid (f x)⟩)
  -/
  rw [liftp_iff]
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), p (Quot.mk (⇑QPF.Wsetoid) (f a))
    ⊢ Exists fun a_1 => Exists fun f_1 => And (Eq (QPF.abs ⟨a, fun x => Quotient.m …
  -/
  refine ⟨_, _, rfl, ?_⟩
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    p : QPF.Fix F → Prop
    h : ∀ (x : F (QPF.Fix F)), Functor.Liftp p x → p (QPF.Fix.mk x)
    x✝ : QPF.Fix F
    a : (QPF.P F).A
    f : (QPF.P F).B a → WType (QPF.P F).B
    ih : ∀ (a : (QPF.P F).B a), p (Quot.mk (⇑QPF.Wsetoid) (f a))
    ⊢ ∀ (i : (QPF.P F).B a), p (Quotient.mk QPF.Wsetoid (f i))
  -/
  convert ih
  /-
    🎉 no goals
  -/


/-- does recursion on `q.P.M` using `g : α → F α` rather than `g : α → P α` -/
def corecF {α : Type _} (g : α → F α) : α → q.P.M :=
  PFunctor.M.corec fun x => repr (g x)


theorem corecF_eq {α : Type _} (g : α → F α) (x : α) :
    PFunctor.M.dest (corecF g x) = q.P.map (corecF g) (repr (g x)) := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : α → F α
    x : α
    ⊢ Eq (QPF.corecF g x).dest ((QPF.P F).map (QPF.corecF g) (QPF.repr (g x)))
  -/
  rw [corecF, PFunctor.M.dest_corec]
  /-
    🎉 no goals
  -/

-- Equivalence

/-- A pre-congruence on `q.P.M` *viewed as an F-coalgebra*. Not necessarily symmetric. -/
def IsPrecongr (r : q.P.M → q.P.M → Prop) : Prop :=
  ∀ ⦃x y⦄, r x y →
    abs (q.P.map (Quot.mk r) (PFunctor.M.dest x)) = abs (q.P.map (Quot.mk r) (PFunctor.M.dest y))


/-- The maximal congruence on `q.P.M`. -/
def Mcongr : q.P.M → q.P.M → Prop := fun x y => ∃ r, IsPrecongr r ∧ r x y


/-- coinductive type defined as the final coalgebra of a qpf -/
def Cofix (F : Type u → Type u) [q : QPF F] :=
  Quot (@Mcongr F q)


instance [Inhabited q.P.A] : Inhabited (Cofix F) :=
  ⟨Quot.mk _ default⟩


/-- corecursor for type defined by `Cofix` -/
def Cofix.corec {α : Type _} (g : α → F α) (x : α) : Cofix F :=
  Quot.mk _ (corecF g x)


/-- destructor for type defined by `Cofix` -/
def Cofix.dest : Cofix F → F (Cofix F) :=
  Quot.lift (fun x => Quot.mk Mcongr <$> abs (PFunctor.M.dest x))
    (by
      /-
        F : Type u → Type u
        q : QPF F
        ⊢ ∀ (a b : (QPF.P F).M), QPF.Mcongr a b → Eq ((fun x => Functor.map (Quot.mk Q …
      -/
      rintro x y ⟨r, pr, rxy⟩
      /-
        case intro.intro
        F : Type u → Type u
        q : QPF F
        x y : (QPF.P F).M
        r : (QPF.P F).M → (QPF.P F).M → Prop
        pr : QPF.IsPrecongr r
        rxy : r x y
        ⊢ Eq ((fun x => Functor.map (Quot.mk QPF.Mcongr) (QPF.abs x.dest)) x) ((fun x  …
      -/
      dsimp
      have : ∀ x y, r x y → Mcongr x y := by
        intro x y h
        exact ⟨r, pr, h⟩
      /-
        case intro.intro
        F : Type u → Type u
        q : QPF F
        x y : (QPF.P F).M
        r : (QPF.P F).M → (QPF.P F).M → Prop
        pr : QPF.IsPrecongr r
        rxy : r x y
        this : ∀ (x y : (QPF.P F).M), r x y → QPF.Mcongr x y
        ⊢ Eq (Functor.map (Quot.mk QPF.Mcongr) (QPF.abs x.dest)) (Functor.map (Quot.mk …
      -/
      rw [← Quot.factor_mk_eq _ _ this]
      conv =>
        lhs
        rw [comp_map, ← abs_map, pr rxy, abs_map, ← comp_map])


theorem Cofix.dest_corec {α : Type u} (g : α → F α) (x : α) :
    Cofix.dest (Cofix.corec g x) = Cofix.corec g <$> g x := by
  conv =>
    lhs
    rw [Cofix.dest, Cofix.corec]
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : α → F α
    x : α
    ⊢ Eq (Quot.lift (fun x => Functor.map (Quot.mk QPF.Mcongr) (QPF.abs x.dest)) ⋯ …
  -/
  dsimp
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    g : α → F α
    x : α
    ⊢ Eq (Functor.map (Quot.mk QPF.Mcongr) (QPF.abs (QPF.corecF g x).dest)) (Funct …
  -/
  rw [corecF_eq, abs_map, abs_repr, ← comp_map]; rfl
                                                 /-
                                                   🎉 no goals
                                                 -/


private theorem Cofix.bisim_aux (r : Cofix F → Cofix F → Prop) (h' : ∀ x, r x x)
    (h : ∀ x y, r x y → Quot.mk r <$> Cofix.dest x = Quot.mk r <$> Cofix.dest y) :
    ∀ x y, r x y → x = y := by
  /-
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h' : ∀ (x : QPF.Cofix F), r x x
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    ⊢ ∀ (x y : QPF.Cofix F), r x y → Eq x y
  -/
  rintro ⟨x⟩ ⟨y⟩ rxy
  /-
    case mk.mk
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h' : ∀ (x : QPF.Cofix F), r x x
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    x✝ : QPF.Cofix F
    x : (QPF.P F).M
    y✝ : QPF.Cofix F
    y : (QPF.P F).M
    rxy : r (Quot.mk QPF.Mcongr x) (Quot.mk QPF.Mcongr y)
    ⊢ Eq (Quot.mk QPF.Mcongr x) (Quot.mk QPF.Mcongr y)
  -/
  apply Quot.sound
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h' : ∀ (x : QPF.Cofix F), r x x
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    x✝ : QPF.Cofix F
    x : (QPF.P F).M
    y✝ : QPF.Cofix F
    y : (QPF.P F).M
    rxy : r (Quot.mk QPF.Mcongr x) (Quot.mk QPF.Mcongr y)
    ⊢ QPF.Mcongr x y
  -/
  let r' x y := r (Quot.mk _ x) (Quot.mk _ y)
  have : IsPrecongr r' := by
    intro a b r'ab
    have h₀ :
      Quot.mk r <$> Quot.mk Mcongr <$> abs (PFunctor.M.dest a) =
        Quot.mk r <$> Quot.mk Mcongr <$> abs (PFunctor.M.dest b) :=
      h _ _ r'ab
    have h₁ : ∀ u v : q.P.M, Mcongr u v → Quot.mk r' u = Quot.mk r' v := by
      intro u v cuv
      apply Quot.sound
      simp only [r']
      rw [Quot.sound cuv]
      apply h'
    let f : Quot r → Quot r' :=
      Quot.lift (Quot.lift (Quot.mk r') h₁) <| by
        rintro ⟨c⟩ ⟨d⟩ rcd
        exact Quot.sound rcd
    have : f ∘ Quot.mk r ∘ Quot.mk Mcongr = Quot.mk r' := rfl
    rw [← this, ← PFunctor.map_map _ _ f, ← PFunctor.map_map _ _ (Quot.mk r), abs_map, abs_map,
      abs_map, h₀]
    rw [← PFunctor.map_map _ _ f, ← PFunctor.map_map _ _ (Quot.mk r), abs_map, abs_map, abs_map]
  /-
    case mk.mk.a
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h' : ∀ (x : QPF.Cofix F), r x x
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    x✝ : QPF.Cofix F
    x : (QPF.P F).M
    y✝ : QPF.Cofix F
    y : (QPF.P F).M
    rxy : r (Quot.mk QPF.Mcongr x) (Quot.mk QPF.Mcongr y)
    r' : (QPF.P F).M → (QPF.P F).M → Prop := fun x y => r (Quot.mk QPF.Mcongr x) ( …
    this : QPF.IsPrecongr r'
    ⊢ QPF.Mcongr x y
  -/
  exact ⟨r', this, rxy⟩
  /-
    🎉 no goals
  -/


theorem Cofix.bisim_rel (r : Cofix F → Cofix F → Prop)
    (h : ∀ x y, r x y → Quot.mk r <$> Cofix.dest x = Quot.mk r <$> Cofix.dest y) :
    ∀ x y, r x y → x = y := by
  /-
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    ⊢ ∀ (x y : QPF.Cofix F), r x y → Eq x y
  -/
  let r' (x y) := x = y ∨ r x y
  /-
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
    ⊢ ∀ (x y : QPF.Cofix F), r x y → Eq x y
  -/
  intro x y rxy
  /-
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
    x y : QPF.Cofix F
    rxy : r x y
    ⊢ Eq x y
  -/
  apply Cofix.bisim_aux r'
    /-
      case h'
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x y : QPF.Cofix F
      rxy : r x y
      ⊢ ∀ (x : QPF.Cofix F), r' x x
    -/
  · intro x
    /-
      case h'
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y : QPF.Cofix F
      rxy : r x✝ y
      x : QPF.Cofix F
      ⊢ r' x x
    -/
    left
    /-
      case h'.h
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y : QPF.Cofix F
      rxy : r x✝ y
      x : QPF.Cofix F
      ⊢ Eq x x
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x y : QPF.Cofix F
      rxy : r x y
      ⊢ ∀ (x y : QPF.Cofix F), r' x y → Eq (Functor.map (Quot.mk r') x.dest) (Functo …
    -/
  · intro x y r'xy
    /-
      case h
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y✝ : QPF.Cofix F
      rxy : r x✝ y✝
      x y : QPF.Cofix F
      r'xy : r' x y
      ⊢ Eq (Functor.map (Quot.mk r') x.dest) (Functor.map (Quot.mk r') y.dest)
    -/
    cases' r'xy with r'xy r'xy
      /-
        case h.inl
        F : Type u → Type u
        q : QPF F
        r : QPF.Cofix F → QPF.Cofix F → Prop
        h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
        r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
        x✝ y✝ : QPF.Cofix F
        rxy : r x✝ y✝
        x y : QPF.Cofix F
        r'xy : Eq x y
        ⊢ Eq (Functor.map (Quot.mk r') x.dest) (Functor.map (Quot.mk r') y.dest)
      -/
    · rw [r'xy]
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y✝ : QPF.Cofix F
      rxy : r x✝ y✝
      x y : QPF.Cofix F
      r'xy : r x y
      ⊢ Eq (Functor.map (Quot.mk r') x.dest) (Functor.map (Quot.mk r') y.dest)
    -/
    have : ∀ x y, r x y → r' x y := fun x y h => Or.inr h
    /-
      case h.inr
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y✝ : QPF.Cofix F
      rxy : r x✝ y✝
      x y : QPF.Cofix F
      r'xy : r x y
      this : ∀ (x y : QPF.Cofix F), r x y → r' x y
      ⊢ Eq (Functor.map (Quot.mk r') x.dest) (Functor.map (Quot.mk r') y.dest)
    -/
    rw [← Quot.factor_mk_eq _ _ this]
    /-
      case h.inr
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y✝ : QPF.Cofix F
      rxy : r x✝ y✝
      x y : QPF.Cofix F
      r'xy : r x y
      this : ∀ (x y : QPF.Cofix F), r x y → r' x y
      ⊢ Eq (Functor.map (Function.comp (Quot.factor r r' this) (Quot.mk r)) x.dest)  …
    -/
    dsimp [r']
    /-
      case h.inr
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y✝ : QPF.Cofix F
      rxy : r x✝ y✝
      x y : QPF.Cofix F
      r'xy : r x y
      this : ∀ (x y : QPF.Cofix F), r x y → r' x y
      ⊢ Eq (Functor.map (Function.comp (Quot.factor r (fun x y => Or (Eq x y) (r x y …
    -/
    rw [@comp_map _ q _ _ _ (Quot.mk r), @comp_map _ q _ _ _ (Quot.mk r)]
    /-
      case h.inr
      F : Type u → Type u
      q : QPF F
      r : QPF.Cofix F → QPF.Cofix F → Prop
      h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
      r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
      x✝ y✝ : QPF.Cofix F
      rxy : r x✝ y✝
      x y : QPF.Cofix F
      r'xy : r x y
      this : ∀ (x y : QPF.Cofix F), r x y → r' x y
      ⊢ Eq (Functor.map (Quot.factor r (fun x y => Or (Eq x y) (r x y)) this) (Funct …
    -/
    rw [h _ _ r'xy]
    /-
      🎉 no goals
    -/
  /-
    case a
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functo …
    r' : QPF.Cofix F → QPF.Cofix F → Prop := fun x y => Or (Eq x y) (r x y)
    x y : QPF.Cofix F
    rxy : r x y
    ⊢ r' x y
  -/
  right; exact rxy
         /-
           🎉 no goals
         -/


theorem Cofix.bisim (r : Cofix F → Cofix F → Prop)
    (h : ∀ x y, r x y → Liftr r (Cofix.dest x) (Cofix.dest y)) : ∀ x y, r x y → x = y := by
  /-
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    ⊢ ∀ (x y : QPF.Cofix F), r x y → Eq x y
  -/
  apply Cofix.bisim_rel
  /-
    case h
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    ⊢ ∀ (x y : QPF.Cofix F), r x y → Eq (Functor.map (Quot.mk r) x.dest) (Functor. …
  -/
  intro x y rxy
  /-
    case h
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    x y : QPF.Cofix F
    rxy : r x y
    ⊢ Eq (Functor.map (Quot.mk r) x.dest) (Functor.map (Quot.mk r) y.dest)
  -/
  rcases (liftr_iff r _ _).mp (h x y rxy) with ⟨a, f₀, f₁, dxeq, dyeq, h'⟩
  /-
    case h.intro.intro.intro.intro.intro
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    x y : QPF.Cofix F
    rxy : r x y
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → QPF.Cofix F
    dxeq : Eq x.dest (QPF.abs ⟨a, f₀⟩)
    dyeq : Eq y.dest (QPF.abs ⟨a, f₁⟩)
    h' : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    ⊢ Eq (Functor.map (Quot.mk r) x.dest) (Functor.map (Quot.mk r) y.dest)
  -/
  rw [dxeq, dyeq, ← abs_map, ← abs_map, PFunctor.map_eq, PFunctor.map_eq]
  /-
    case h.intro.intro.intro.intro.intro
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    x y : QPF.Cofix F
    rxy : r x y
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → QPF.Cofix F
    dxeq : Eq x.dest (QPF.abs ⟨a, f₀⟩)
    dyeq : Eq y.dest (QPF.abs ⟨a, f₁⟩)
    h' : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    ⊢ Eq (QPF.abs ⟨a, Function.comp (Quot.mk r) f₀⟩) (QPF.abs ⟨a, Function.comp (Q …
  -/
  congr 2 with i
  /-
    case h.intro.intro.intro.intro.intro.e_a.e_snd.h
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    x y : QPF.Cofix F
    rxy : r x y
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → QPF.Cofix F
    dxeq : Eq x.dest (QPF.abs ⟨a, f₀⟩)
    dyeq : Eq y.dest (QPF.abs ⟨a, f₁⟩)
    h' : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    i : (QPF.P F).B a
    ⊢ Eq (Function.comp (Quot.mk r) f₀ i) (Function.comp (Quot.mk r) f₁ i)
  -/
  apply Quot.sound
  /-
    case h.intro.intro.intro.intro.intro.e_a.e_snd.h.a
    F : Type u → Type u
    q : QPF F
    r : QPF.Cofix F → QPF.Cofix F → Prop
    h : ∀ (x y : QPF.Cofix F), r x y → Functor.Liftr r x.dest y.dest
    x y : QPF.Cofix F
    rxy : r x y
    a : (QPF.P F).A
    f₀ f₁ : (QPF.P F).B a → QPF.Cofix F
    dxeq : Eq x.dest (QPF.abs ⟨a, f₀⟩)
    dyeq : Eq y.dest (QPF.abs ⟨a, f₁⟩)
    h' : ∀ (i : (QPF.P F).B a), r (f₀ i) (f₁ i)
    i : (QPF.P F).B a
    ⊢ r (f₀ i) (f₁ i)
  -/
  apply h'
  /-
    🎉 no goals
  -/


theorem Cofix.bisim' {α : Type*} (Q : α → Prop) (u v : α → Cofix F)
    (h : ∀ x, Q x → ∃ a f f', Cofix.dest (u x) = abs ⟨a, f⟩ ∧ Cofix.dest (v x) = abs ⟨a, f'⟩ ∧
      ∀ i, ∃ x', Q x' ∧ f i = u x' ∧ f' i = v x') :
    ∀ x, Q x → u x = v x := fun x Qx =>
  let R := fun w z : Cofix F => ∃ x', Q x' ∧ w = u x' ∧ z = v x'
  Cofix.bisim R
    (fun x y ⟨x', Qx', xeq, yeq⟩ => by
      /-
        F : Type u → Type u
        q : QPF F
        α : Type u_1
        Q : α → Prop
        u v : α → QPF.Cofix F
        h : ∀ (x : α), Q x → Exists fun a => Exists fun f => Exists fun f' => And (Eq  …
        x✝¹ : α
        Qx : Q x✝¹
        R : QPF.Cofix F → QPF.Cofix F → Prop := fun w z => Exists fun x' => And (Q x') …
        x y : QPF.Cofix F
        x✝ : R x y
        x' : α
        Qx' : Q x'
        xeq : Eq x (u x')
        yeq : Eq y (v x')
        ⊢ Functor.Liftr R x.dest y.dest
      -/
      rcases h x' Qx' with ⟨a, f, f', ux'eq, vx'eq, h'⟩
      /-
        case intro.intro.intro.intro.intro
        F : Type u → Type u
        q : QPF F
        α : Type u_1
        Q : α → Prop
        u v : α → QPF.Cofix F
        h : ∀ (x : α), Q x → Exists fun a => Exists fun f => Exists fun f' => And (Eq  …
        x✝¹ : α
        Qx : Q x✝¹
        R : QPF.Cofix F → QPF.Cofix F → Prop := fun w z => Exists fun x' => And (Q x') …
        x y : QPF.Cofix F
        x✝ : R x y
        x' : α
        Qx' : Q x'
        xeq : Eq x (u x')
        yeq : Eq y (v x')
        a : (QPF.P F).A
        f f' : (QPF.P F).B a → QPF.Cofix F
        ux'eq : Eq (u x').dest (QPF.abs ⟨a, f⟩)
        vx'eq : Eq (v x').dest (QPF.abs ⟨a, f'⟩)
        h' : ∀ (i : (QPF.P F).B a), Exists fun x' => And (Q x') (And (Eq (f i) (u x')) …
        ⊢ Functor.Liftr R x.dest y.dest
      -/
      rw [liftr_iff]
      /-
        case intro.intro.intro.intro.intro
        F : Type u → Type u
        q : QPF F
        α : Type u_1
        Q : α → Prop
        u v : α → QPF.Cofix F
        h : ∀ (x : α), Q x → Exists fun a => Exists fun f => Exists fun f' => And (Eq  …
        x✝¹ : α
        Qx : Q x✝¹
        R : QPF.Cofix F → QPF.Cofix F → Prop := fun w z => Exists fun x' => And (Q x') …
        x y : QPF.Cofix F
        x✝ : R x y
        x' : α
        Qx' : Q x'
        xeq : Eq x (u x')
        yeq : Eq y (v x')
        a : (QPF.P F).A
        f f' : (QPF.P F).B a → QPF.Cofix F
        ux'eq : Eq (u x').dest (QPF.abs ⟨a, f⟩)
        vx'eq : Eq (v x').dest (QPF.abs ⟨a, f'⟩)
        h' : ∀ (i : (QPF.P F).B a), Exists fun x' => And (Q x') (And (Eq (f i) (u x')) …
        ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x.dest (QPF.abs ⟨a …
      -/
      exact ⟨a, f, f', xeq.symm ▸ ux'eq, yeq.symm ▸ vx'eq, h'⟩)
      /-
        🎉 no goals
      -/
    _ _ ⟨x, Qx, rfl, rfl⟩


/-- composition of qpfs gives another qpf -/
def comp : QPF (Functor.Comp F₂ F₁) where
  P := PFunctor.comp q₂.P q₁.P
  abs {α} := by
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      ⊢ ↑((QPF.P F₂).comp (QPF.P F₁)) α → Functor.Comp F₂ F₁ α
    -/
    dsimp [Functor.Comp]
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      ⊢ ↑((QPF.P F₂).comp (QPF.P F₁)) α → F₂ (F₁ α)
    -/
    intro p
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      p : ↑((QPF.P F₂).comp (QPF.P F₁)) α
      ⊢ F₂ (F₁ α)
    -/
    exact abs ⟨p.1.1, fun x => abs ⟨p.1.2 x, fun y => p.2 ⟨x, y⟩⟩⟩
    /-
      🎉 no goals
    -/
  repr {α} := by
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      ⊢ Functor.Comp F₂ F₁ α → ↑((QPF.P F₂).comp (QPF.P F₁)) α
    -/
    dsimp [Functor.Comp]
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      ⊢ F₂ (F₁ α) → ↑((QPF.P F₂).comp (QPF.P F₁)) α
    -/
    intro y
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      y : F₂ (F₁ α)
      ⊢ ↑((QPF.P F₂).comp (QPF.P F₁)) α
    -/
    refine ⟨⟨(repr y).1, fun u => (repr ((repr y).2 u)).1⟩, ?_⟩
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      y : F₂ (F₁ α)
      ⊢ ((QPF.P F₂).comp (QPF.P F₁)).B ⟨(QPF.repr y).fst, fun u => (QPF.repr ((QPF.r …
    -/
    dsimp [PFunctor.comp]
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      y : F₂ (F₁ α)
      ⊢ (Sigma fun u => (QPF.P F₁).B (QPF.repr ((QPF.repr y).snd u)).fst) → α
    -/
    intro x
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      y : F₂ (F₁ α)
      x : Sigma fun u => (QPF.P F₁).B (QPF.repr ((QPF.repr y).snd u)).fst
      ⊢ α
    -/
    exact (repr ((repr y).2 x.1)).snd x.2
    /-
      🎉 no goals
    -/
  abs_repr {α} := by
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      ⊢ ∀ (x : Functor.Comp F₂ F₁ α), Eq ((fun {α} => id fun p => QPF.abs ⟨p.fst.fst …
    -/
    dsimp [Functor.Comp]
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      ⊢ ∀ (x : F₂ (F₁ α)), Eq (QPF.abs ⟨(QPF.repr x).fst, fun x_1 => QPF.abs ⟨(QPF.r …
    -/
    intro x
    conv =>
      rhs
      rw [← abs_repr x]
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      x : F₂ (F₁ α)
      ⊢ Eq (QPF.abs ⟨(QPF.repr x).fst, fun x_1 => QPF.abs ⟨(QPF.repr ((QPF.repr x).s …
    -/
    cases' repr x with a f
    /-
      case mk
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      x : F₂ (F₁ α)
      a : (QPF.P F₂).A
      f : (QPF.P F₂).B a → F₁ α
      ⊢ Eq (QPF.abs ⟨⟨a, f⟩.fst, fun x => QPF.abs ⟨(QPF.repr (⟨a, f⟩.snd x)).fst, fu …
    -/
    dsimp
    /-
      case mk
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      x : F₂ (F₁ α)
      a : (QPF.P F₂).A
      f : (QPF.P F₂).B a → F₁ α
      ⊢ Eq (QPF.abs ⟨a, fun x => QPF.abs ⟨(QPF.repr (f x)).fst, fun y => (QPF.repr ( …
    -/
    congr with x
    /-
      case mk.e_a.e_snd.h
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      x✝ : F₂ (F₁ α)
      a : (QPF.P F₂).A
      f : (QPF.P F₂).B a → F₁ α
      x : (QPF.P F₂).B a
      ⊢ Eq (QPF.abs ⟨(QPF.repr (f x)).fst, fun y => (QPF.repr (f x)).snd y⟩) (f x)
    -/
    cases' h' : repr (f x) with b g
    /-
      case mk.e_a.e_snd.h.mk
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α : Type u
      x✝ : F₂ (F₁ α)
      a : (QPF.P F₂).A
      f : (QPF.P F₂).B a → F₁ α
      x : (QPF.P F₂).B a
      b : (QPF.P F₁).A
      g : (QPF.P F₁).B b → α
      h' : Eq (QPF.repr (f x)) ⟨b, g⟩
      ⊢ Eq (QPF.abs ⟨⟨b, g⟩.fst, fun y => ⟨b, g⟩.snd y⟩) (f x)
    -/
    dsimp; rw [← h', abs_repr]
           /-
             🎉 no goals
           -/
  abs_map {α β} f := by
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      ⊢ ∀ (p : ↑((QPF.P F₂).comp (QPF.P F₁)) α), Eq ((fun {α} => id fun p => QPF.abs …
    -/
    dsimp (config := { unfoldPartialApp := true }) [Functor.Comp, PFunctor.comp]
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      ⊢ ∀ (p : ↑{ A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ …
    -/
    intro p
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      p : ↑{ A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => S …
      ⊢ Eq (QPF.abs ⟨({ A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fu …
    -/
    cases' p with a g; dsimp
    /-
      case mk
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      a : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq (QPF.abs ⟨({ A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fu …
    -/
    cases' a with b h; dsimp
    /-
      case mk.mk
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq (QPF.abs ⟨({ A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fu …
    -/
    symm
    /-
      case mk.mk
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq (Functor.map f (QPF.abs ⟨b, fun x => QPF.abs ⟨h x, fun y => g ⟨x, y⟩⟩⟩))  …
    -/
    trans
      /-
        F₂ : Type u → Type u
        q₂ : QPF F₂
        F₁ : Type u → Type u
        q₁ : QPF F₁
        α β : Type u
        f : α → β
        b : (QPF.P F₂).A
        h : (QPF.P F₂).B b → (QPF.P F₁).A
        g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
        ⊢ Eq (Functor.map f (QPF.abs ⟨b, fun x => QPF.abs ⟨h x, fun y => g ⟨x, y⟩⟩⟩))  …
      -/
    · symm
      /-
        F₂ : Type u → Type u
        q₂ : QPF F₂
        F₁ : Type u → Type u
        q₁ : QPF F₁
        α β : Type u
        f : α → β
        b : (QPF.P F₂).A
        h : (QPF.P F₂).B b → (QPF.P F₁).A
        g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
        ⊢ Eq ?a (Functor.map f (QPF.abs ⟨b, fun x => QPF.abs ⟨h x, fun y => g ⟨x, y⟩⟩⟩))
      -/
      apply abs_map
      /-
        🎉 no goals
      -/
    /-
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq (QPF.abs ((QPF.P F₂).map (fun x => Functor.map f x) ⟨b, fun x => QPF.abs  …
    -/
    congr
    /-
      case e_a
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq ((QPF.P F₂).map (fun x => Functor.map f x) ⟨b, fun x => QPF.abs ⟨h x, fun …
    -/
    rw [PFunctor.map_eq]
    /-
      case e_a
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq ⟨b, Function.comp (fun x => Functor.map f x) fun x => QPF.abs ⟨h x, fun y …
    -/
    dsimp [Function.comp_def]
    /-
      case e_a
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq ⟨b, fun x => Functor.map f (QPF.abs ⟨h x, fun y => g ⟨x, y⟩⟩)⟩ ⟨({ A := S …
    -/
    congr
    /-
      case e_a.e_snd
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      ⊢ Eq (fun x => Functor.map f (QPF.abs ⟨h x, fun y => g ⟨x, y⟩⟩)) fun x => QPF. …
    -/
    ext x
    /-
      case e_a.e_snd.h
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      x : (QPF.P F₂).B b
      ⊢ Eq (Functor.map f (QPF.abs ⟨h x, fun y => g ⟨x, y⟩⟩)) (QPF.abs ⟨({ A := Sigm …
    -/
    rw [← abs_map]
    /-
      case e_a.e_snd.h
      F₂ : Type u → Type u
      q₂ : QPF F₂
      F₁ : Type u → Type u
      q₁ : QPF F₁
      α β : Type u
      f : α → β
      b : (QPF.P F₂).A
      h : (QPF.P F₂).B b → (QPF.P F₁).A
      g : { A := Sigma fun a₂ => (QPF.P F₂).B a₂ → (QPF.P F₁).A, B := fun a₂a₁ => Si …
      x : (QPF.P F₂).B b
      ⊢ Eq (QPF.abs ((QPF.P F₁).map f ⟨h x, fun y => g ⟨x, y⟩⟩)) (QPF.abs ⟨({ A := S …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Given a qpf `F` and a well-behaved surjection `FG_abs` from `F α` to
functor `G α`, `G` is a qpf. We can consider `G` a quotient on `F` where
elements `x y : F α` are in the same equivalence class if
`FG_abs x = FG_abs y`. -/
def quotientQPF (FG_abs_repr : ∀ {α} (x : G α), FG_abs (FG_repr x) = x)
    (FG_abs_map : ∀ {α β} (f : α → β) (x : F α), FG_abs (f <$> x) = f <$> FG_abs x) : QPF G where
  P := q.P
  abs {_} p := FG_abs (abs p)
  repr {_} x := repr (FG_repr x)
                       /-
                         F : Type u → Type u
                         q : QPF F
                         G : Type u → Type u
                         inst✝ : Functor G
                         FG_abs : {α : Type u} → F α → G α
                         FG_repr : {α : Type u} → G α → F α
                         FG_abs_repr : ∀ {α : Type u} (x : G α), Eq (FG_abs (FG_repr x)) x
                         FG_abs_map : ∀ {α β : Type u} (f : α → β) (x : F α), Eq (FG_abs (Functor.map f …
                         α : Type u
                         x : G α
                         ⊢ Eq ((fun {x} p => FG_abs (QPF.abs p)) ((fun {x} x_1 => QPF.repr (FG_repr x_1 …
                       -/
  abs_repr {α} x := by simp only; rw [abs_repr, FG_abs_repr]
                                  /-
                                    🎉 no goals
                                  -/
                          /-
                            F : Type u → Type u
                            q : QPF F
                            G : Type u → Type u
                            inst✝ : Functor G
                            FG_abs : {α : Type u} → F α → G α
                            FG_repr : {α : Type u} → G α → F α
                            FG_abs_repr : ∀ {α : Type u} (x : G α), Eq (FG_abs (FG_repr x)) x
                            FG_abs_map : ∀ {α β : Type u} (f : α → β) (x : F α), Eq (FG_abs (Functor.map f …
                            α β : Type u
                            f : α → β
                            x : ↑(QPF.P F) α
                            ⊢ Eq ((fun {x} p => FG_abs (QPF.abs p)) ((QPF.P F).map f x)) (Functor.map f (( …
                          -/
  abs_map {α β} f x := by simp only; rw [abs_map, FG_abs_map]
                                     /-
                                       🎉 no goals
                                     -/


theorem mem_supp {α : Type u} (x : F α) (u : α) :
    u ∈ supp x ↔ ∀ a f, abs ⟨a, f⟩ = x → u ∈ f '' univ := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    u : α
    ⊢ Iff (Membership.mem (Functor.supp x) u) (∀ (a : (QPF.P F).A) (f : (QPF.P F). …
  -/
  rw [supp]; dsimp; constructor
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      u : α
      ⊢ (∀ ⦃p : α → Prop⦄, Functor.Liftp p x → p u) → ∀ (a : (QPF.P F).A) (f : (QPF. …
    -/
  · intro h a f haf
    have : Liftp (fun u => u ∈ f '' univ) x := by
      rw [liftp_iff]
      exact ⟨a, f, haf.symm, fun i => mem_image_of_mem _ (mem_univ _)⟩
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      u : α
      h : ∀ ⦃p : α → Prop⦄, Functor.Liftp p x → p u
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      haf : Eq (QPF.abs ⟨a, f⟩) x
      this : Functor.Liftp (fun u => Membership.mem (Set.image f Set.univ) u) x
      ⊢ Membership.mem (Set.image f Set.univ) u
    -/
    exact h this
    /-
      🎉 no goals
    -/
  /-
    case mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    u : α
    ⊢ (∀ (a : (QPF.P F).A) (f : (QPF.P F).B a → α), Eq (QPF.abs ⟨a, f⟩) x → Member …
  -/
  intro h p; rw [liftp_iff]
  /-
    case mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    u : α
    h : ∀ (a : (QPF.P F).A) (f : (QPF.P F).B a → α), Eq (QPF.abs ⟨a, f⟩) x → Membe …
    p : α → Prop
    ⊢ (Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QPF.P  …
  -/
  rintro ⟨a, f, xeq, h'⟩
  /-
    case mpr.intro.intro.intro
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    u : α
    h : ∀ (a : (QPF.P F).A) (f : (QPF.P F).B a → α), Eq (QPF.abs ⟨a, f⟩) x → Membe …
    p : α → Prop
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq x (QPF.abs ⟨a, f⟩)
    h' : ∀ (i : (QPF.P F).B a), p (f i)
    ⊢ p u
  -/
  rcases h a f xeq.symm with ⟨i, _, hi⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    u : α
    h : ∀ (a : (QPF.P F).A) (f : (QPF.P F).B a → α), Eq (QPF.abs ⟨a, f⟩) x → Membe …
    p : α → Prop
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq x (QPF.abs ⟨a, f⟩)
    h' : ∀ (i : (QPF.P F).B a), p (f i)
    i : (QPF.P F).B a
    left✝ : Membership.mem Set.univ i
    hi : Eq (f i) u
    ⊢ p u
  -/
  rw [← hi]; apply h'
             /-
               🎉 no goals
             -/


theorem supp_eq {α : Type u} (x : F α) :
    supp x = { u | ∀ a f, abs ⟨a, f⟩ = x → u ∈ f '' univ } := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    ⊢ Eq (Functor.supp x) (setOf fun u => ∀ (a : (QPF.P F).A) (f : (QPF.P F).B a → …
  -/
  ext
  /-
    case h
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    x✝ : α
    ⊢ Iff (Membership.mem (Functor.supp x) x✝) (Membership.mem (setOf fun u => ∀ ( …
  -/
  apply mem_supp
  /-
    🎉 no goals
  -/


theorem has_good_supp_iff {α : Type u} (x : F α) :
    (∀ p, Liftp p x ↔ ∀ u ∈ supp x, p u) ↔
      ∃ a f, abs ⟨a, f⟩ = x ∧ ∀ a' f', abs ⟨a', f'⟩ = x → f '' univ ⊆ f' '' univ := by
  /-
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    ⊢ Iff (∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (F …
  -/
  constructor
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      ⊢ (∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Funct …
    -/
  · intro h
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      ⊢ Exists fun a => Exists fun f => And (Eq (QPF.abs ⟨a, f⟩) x) (∀ (a' : (QPF.P  …
    -/
    have : Liftp (supp x) x := by rw [h]; intro u; exact id
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      this : Functor.Liftp (Functor.supp x) x
      ⊢ Exists fun a => Exists fun f => And (Eq (QPF.abs ⟨a, f⟩) x) (∀ (a' : (QPF.P  …
    -/
    rw [liftp_iff] at this
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      this : Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QP …
      ⊢ Exists fun a => Exists fun f => And (Eq (QPF.abs ⟨a, f⟩) x) (∀ (a' : (QPF.P  …
    -/
    rcases this with ⟨a, f, xeq, h'⟩
    /-
      case mp.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f⟩)
      h' : ∀ (i : (QPF.P F).B a), Functor.supp x (f i)
      ⊢ Exists fun a => Exists fun f => And (Eq (QPF.abs ⟨a, f⟩) x) (∀ (a' : (QPF.P  …
    -/
    refine ⟨a, f, xeq.symm, ?_⟩
    /-
      case mp.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f⟩)
      h' : ∀ (i : (QPF.P F).B a), Functor.supp x (f i)
      ⊢ ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x → Ha …
    -/
    intro a' f' h''
    /-
      case mp.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f⟩)
      h' : ∀ (i : (QPF.P F).B a), Functor.supp x (f i)
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      h'' : Eq (QPF.abs ⟨a', f'⟩) x
      ⊢ HasSubset.Subset (Set.image f Set.univ) (Set.image f' Set.univ)
    -/
    rintro u ⟨i, _, hfi⟩
    /-
      case mp.intro.intro.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f⟩)
      h' : ∀ (i : (QPF.P F).B a), Functor.supp x (f i)
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      h'' : Eq (QPF.abs ⟨a', f'⟩) x
      u : α
      i : (QPF.P F).B a
      left✝ : Membership.mem Set.univ i
      hfi : Eq (f i) u
      ⊢ Membership.mem (Set.image f' Set.univ) u
    -/
    have : u ∈ supp x := by rw [← hfi]; apply h'
    /-
      case mp.intro.intro.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      h : ∀ (p : α → Prop), Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Func …
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq x (QPF.abs ⟨a, f⟩)
      h' : ∀ (i : (QPF.P F).B a), Functor.supp x (f i)
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      h'' : Eq (QPF.abs ⟨a', f'⟩) x
      u : α
      i : (QPF.P F).B a
      left✝ : Membership.mem Set.univ i
      hfi : Eq (f i) u
      this : Membership.mem (Functor.supp x) u
      ⊢ Membership.mem (Set.image f' Set.univ) u
    -/
    exact (mem_supp x u).mp this _ _ h''
    /-
      🎉 no goals
    -/
  /-
    case mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    ⊢ (Exists fun a => Exists fun f => And (Eq (QPF.abs ⟨a, f⟩) x) (∀ (a' : (QPF.P …
  -/
  rintro ⟨a, f, xeq, h⟩ p; rw [liftp_iff]; constructor
    /-
      case mpr.intro.intro.intro.mp
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq (QPF.abs ⟨a, f⟩) x
      h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
      p : α → Prop
      ⊢ (Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QPF.P  …
    -/
  · rintro ⟨a', f', xeq', h'⟩ u usuppx
    /-
      case mpr.intro.intro.intro.mp.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq (QPF.abs ⟨a, f⟩) x
      h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
      p : α → Prop
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      xeq' : Eq x (QPF.abs ⟨a', f'⟩)
      h' : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      usuppx : Membership.mem (Functor.supp x) u
      ⊢ p u
    -/
    rcases (mem_supp x u).mp usuppx a' f' xeq'.symm with ⟨i, _, f'ieq⟩
    /-
      case mpr.intro.intro.intro.mp.intro.intro.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq (QPF.abs ⟨a, f⟩) x
      h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
      p : α → Prop
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      xeq' : Eq x (QPF.abs ⟨a', f'⟩)
      h' : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      usuppx : Membership.mem (Functor.supp x) u
      i : (QPF.P F).B a'
      left✝ : Membership.mem Set.univ i
      f'ieq : Eq (f' i) u
      ⊢ p u
    -/
    rw [← f'ieq]
    /-
      case mpr.intro.intro.intro.mp.intro.intro.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      α : Type u
      x : F α
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      xeq : Eq (QPF.abs ⟨a, f⟩) x
      h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
      p : α → Prop
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      xeq' : Eq x (QPF.abs ⟨a', f'⟩)
      h' : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      usuppx : Membership.mem (Functor.supp x) u
      i : (QPF.P F).B a'
      left✝ : Membership.mem Set.univ i
      f'ieq : Eq (f' i) u
      ⊢ p (f' i)
    -/
    apply h'
    /-
      🎉 no goals
    -/
  /-
    case mpr.intro.intro.intro.mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq (QPF.abs ⟨a, f⟩) x
    h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
    p : α → Prop
    ⊢ (∀ (u : α), Membership.mem (Functor.supp x) u → p u) → Exists fun a => Exist …
  -/
  intro h'
  /-
    case mpr.intro.intro.intro.mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq (QPF.abs ⟨a, f⟩) x
    h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
    p : α → Prop
    h' : ∀ (u : α), Membership.mem (Functor.supp x) u → p u
    ⊢ Exists fun a => Exists fun f => And (Eq x (QPF.abs ⟨a, f⟩)) (∀ (i : (QPF.P F …
  -/
  refine ⟨a, f, xeq.symm, ?_⟩; intro i
  /-
    case mpr.intro.intro.intro.mpr
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq (QPF.abs ⟨a, f⟩) x
    h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
    p : α → Prop
    h' : ∀ (u : α), Membership.mem (Functor.supp x) u → p u
    i : (QPF.P F).B a
    ⊢ p (f i)
  -/
  apply h'; rw [mem_supp]
  /-
    case mpr.intro.intro.intro.mpr.a
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq (QPF.abs ⟨a, f⟩) x
    h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
    p : α → Prop
    h' : ∀ (u : α), Membership.mem (Functor.supp x) u → p u
    i : (QPF.P F).B a
    ⊢ ∀ (a : (QPF.P F).A) (f_1 : (QPF.P F).B a → α), Eq (QPF.abs ⟨a, f_1⟩) x → Mem …
  -/
  intro a' f' xeq'
  /-
    case mpr.intro.intro.intro.mpr.a
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq (QPF.abs ⟨a, f⟩) x
    h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
    p : α → Prop
    h' : ∀ (u : α), Membership.mem (Functor.supp x) u → p u
    i : (QPF.P F).B a
    a' : (QPF.P F).A
    f' : (QPF.P F).B a' → α
    xeq' : Eq (QPF.abs ⟨a', f'⟩) x
    ⊢ Membership.mem (Set.image f' Set.univ) (f i)
  -/
  apply h a' f' xeq'
  /-
    case mpr.intro.intro.intro.mpr.a.a
    F : Type u → Type u
    q : QPF F
    α : Type u
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    xeq : Eq (QPF.abs ⟨a, f⟩) x
    h : ∀ (a' : (QPF.P F).A) (f' : (QPF.P F).B a' → α), Eq (QPF.abs ⟨a', f'⟩) x →  …
    p : α → Prop
    h' : ∀ (u : α), Membership.mem (Functor.supp x) u → p u
    i : (QPF.P F).B a
    a' : (QPF.P F).A
    f' : (QPF.P F).B a' → α
    xeq' : Eq (QPF.abs ⟨a', f'⟩) x
    ⊢ Membership.mem (Set.image f Set.univ) (f i)
  -/
  apply mem_image_of_mem _ (mem_univ _)
  /-
    🎉 no goals
  -/


/-- A qpf is said to be uniform if every polynomial functor
representing a single value all have the same range. -/
def IsUniform : Prop :=
  ∀ ⦃α : Type u⦄ (a a' : q.P.A) (f : q.P.B a → α) (f' : q.P.B a' → α),
    abs ⟨a, f⟩ = abs ⟨a', f'⟩ → f '' univ = f' '' univ


/-- does `abs` preserve `Liftp`? -/
def LiftpPreservation : Prop :=
  ∀ ⦃α⦄ (p : α → Prop) (x : q.P α), Liftp p (abs x) ↔ Liftp p x


/-- does `abs` preserve `supp`? -/
def SuppPreservation : Prop :=
  ∀ ⦃α⦄ (x : q.P α), supp (abs x) = supp x


theorem supp_eq_of_isUniform (h : q.IsUniform) {α : Type u} (a : q.P.A) (f : q.P.B a → α) :
    supp (abs ⟨a, f⟩) = f '' univ := by
  /-
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ Eq (Functor.supp (QPF.abs ⟨a, f⟩)) (Set.image f Set.univ)
  -/
  ext u; rw [mem_supp]; constructor
    /-
      case h.mp
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      u : α
      ⊢ (∀ (a_1 : (QPF.P F).A) (f_1 : (QPF.P F).B a_1 → α), Eq (QPF.abs ⟨a_1, f_1⟩)  …
    -/
  · intro h'
    /-
      case h.mp
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      u : α
      h' : ∀ (a_1 : (QPF.P F).A) (f_1 : (QPF.P F).B a_1 → α), Eq (QPF.abs ⟨a_1, f_1⟩ …
      ⊢ Membership.mem (Set.image f Set.univ) u
    -/
    apply h' _ _ rfl
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    u : α
    ⊢ Membership.mem (Set.image f Set.univ) u → ∀ (a_2 : (QPF.P F).A) (f_1 : (QPF. …
  -/
  intro h' a' f' e
  /-
    case h.mpr
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    u : α
    h' : Membership.mem (Set.image f Set.univ) u
    a' : (QPF.P F).A
    f' : (QPF.P F).B a' → α
    e : Eq (QPF.abs ⟨a', f'⟩) (QPF.abs ⟨a, f⟩)
    ⊢ Membership.mem (Set.image f' Set.univ) u
  -/
  rw [← h _ _ _ _ e.symm]; apply h'
                           /-
                             🎉 no goals
                           -/


theorem liftp_iff_of_isUniform (h : q.IsUniform) {α : Type u} (x : F α) (p : α → Prop) :
    Liftp p x ↔ ∀ u ∈ supp x, p u := by
  /-
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    x : F α
    p : α → Prop
    ⊢ Iff (Functor.Liftp p x) (∀ (u : α), Membership.mem (Functor.supp x) u → p u)
  -/
  rw [liftp_iff, ← abs_repr x]
  /-
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    x : F α
    p : α → Prop
    ⊢ Iff (Exists fun a => Exists fun f => And (Eq (QPF.abs (QPF.repr x)) (QPF.abs …
  -/
  cases' repr x with a f; constructor
    /-
      case mk.mp
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      x : F α
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      ⊢ (Exists fun a_1 => Exists fun f_1 => And (Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a_1, …
    -/
  · rintro ⟨a', f', abseq, hf⟩ u
    /-
      case mk.mp.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      x : F α
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      abseq : Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a', f'⟩)
      hf : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      ⊢ Membership.mem (Functor.supp (QPF.abs ⟨a, f⟩)) u → p u
    -/
    rw [supp_eq_of_isUniform h, h _ _ _ _ abseq]
    /-
      case mk.mp.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      x : F α
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      abseq : Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a', f'⟩)
      hf : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      ⊢ Membership.mem (Set.image f' Set.univ) u → p u
    -/
    rintro ⟨i, _, hi⟩
    /-
      case mk.mp.intro.intro.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      x : F α
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      abseq : Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a', f'⟩)
      hf : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      i : (QPF.P F).B a'
      left✝ : Membership.mem Set.univ i
      hi : Eq (f' i) u
      ⊢ p u
    -/
    rw [← hi]
    /-
      case mk.mp.intro.intro.intro.intro.intro
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      x : F α
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      a' : (QPF.P F).A
      f' : (QPF.P F).B a' → α
      abseq : Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a', f'⟩)
      hf : ∀ (i : (QPF.P F).B a'), p (f' i)
      u : α
      i : (QPF.P F).B a'
      left✝ : Membership.mem Set.univ i
      hi : Eq (f' i) u
      ⊢ p (f' i)
    -/
    apply hf
    /-
      🎉 no goals
    -/
  /-
    case mk.mpr
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    x : F α
    p : α → Prop
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ (∀ (u : α), Membership.mem (Functor.supp (QPF.abs ⟨a, f⟩)) u → p u) → Exists …
  -/
  intro h'
  /-
    case mk.mpr
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    x : F α
    p : α → Prop
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h' : ∀ (u : α), Membership.mem (Functor.supp (QPF.abs ⟨a, f⟩)) u → p u
    ⊢ Exists fun a_1 => Exists fun f_1 => And (Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a_1,  …
  -/
  refine ⟨a, f, rfl, fun i => h' _ ?_⟩
  /-
    case mk.mpr
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    x : F α
    p : α → Prop
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h' : ∀ (u : α), Membership.mem (Functor.supp (QPF.abs ⟨a, f⟩)) u → p u
    i : (QPF.P F).B a
    ⊢ Membership.mem (Functor.supp (QPF.abs ⟨a, f⟩)) (f i)
  -/
  rw [supp_eq_of_isUniform h]
  /-
    case mk.mpr
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α : Type u
    x : F α
    p : α → Prop
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    h' : ∀ (u : α), Membership.mem (Functor.supp (QPF.abs ⟨a, f⟩)) u → p u
    i : (QPF.P F).B a
    ⊢ Membership.mem (Set.image f Set.univ) (f i)
  -/
  exact ⟨i, mem_univ i, rfl⟩
  /-
    🎉 no goals
  -/


theorem supp_map (h : q.IsUniform) {α β : Type u} (g : α → β) (x : F α) :
    supp (g <$> x) = g '' supp x := by
  /-
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α β : Type u
    g : α → β
    x : F α
    ⊢ Eq (Functor.supp (Functor.map g x)) (Set.image g (Functor.supp x))
  -/
  rw [← abs_repr x]; cases' repr x with a f; rw [← abs_map, PFunctor.map_eq]
  /-
    case mk
    F : Type u → Type u
    q : QPF F
    h : QPF.IsUniform
    α β : Type u
    g : α → β
    x : F α
    a : (QPF.P F).A
    f : (QPF.P F).B a → α
    ⊢ Eq (Functor.supp (QPF.abs ⟨a, Function.comp g f⟩)) (Set.image g (Functor.sup …
  -/
  rw [supp_eq_of_isUniform h, supp_eq_of_isUniform h, image_comp]
  /-
    🎉 no goals
  -/


theorem suppPreservation_iff_uniform : q.SuppPreservation ↔ q.IsUniform := by
  /-
    F : Type u → Type u
    q : QPF F
    ⊢ Iff QPF.SuppPreservation QPF.IsUniform
  -/
  constructor
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      ⊢ QPF.SuppPreservation → QPF.IsUniform
    -/
  · intro h α a a' f f' h'
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      h : QPF.SuppPreservation
      α : Type u
      a a' : (QPF.P F).A
      f : (QPF.P F).B a → α
      f' : (QPF.P F).B a' → α
      h' : Eq (QPF.abs ⟨a, f⟩) (QPF.abs ⟨a', f'⟩)
      ⊢ Eq (Set.image f Set.univ) (Set.image f' Set.univ)
    -/
    rw [← PFunctor.supp_eq, ← PFunctor.supp_eq, ← h, h', h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      F : Type u → Type u
      q : QPF F
      ⊢ QPF.IsUniform → QPF.SuppPreservation
    -/
  · rintro h α ⟨a, f⟩
    /-
      case mpr.mk
      F : Type u → Type u
      q : QPF F
      h : QPF.IsUniform
      α : Type u
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      ⊢ Eq (Functor.supp (QPF.abs ⟨a, f⟩)) (Functor.supp ⟨a, f⟩)
    -/
    rwa [supp_eq_of_isUniform, PFunctor.supp_eq]
    /-
      🎉 no goals
    -/


theorem suppPreservation_iff_liftpPreservation : q.SuppPreservation ↔ q.LiftpPreservation := by
  /-
    F : Type u → Type u
    q : QPF F
    ⊢ Iff QPF.SuppPreservation QPF.LiftpPreservation
  -/
  constructor <;> intro h
    /-
      case mp
      F : Type u → Type u
      q : QPF F
      h : QPF.SuppPreservation
      ⊢ QPF.LiftpPreservation
    -/
  · rintro α p ⟨a, f⟩
    /-
      case mp.mk
      F : Type u → Type u
      q : QPF F
      h : QPF.SuppPreservation
      α : Type u
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      ⊢ Iff (Functor.Liftp p (QPF.abs ⟨a, f⟩)) (Functor.Liftp p ⟨a, f⟩)
    -/
    have h' := h
    /-
      case mp.mk
      F : Type u → Type u
      q : QPF F
      h : QPF.SuppPreservation
      α : Type u
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      h' : QPF.SuppPreservation
      ⊢ Iff (Functor.Liftp p (QPF.abs ⟨a, f⟩)) (Functor.Liftp p ⟨a, f⟩)
    -/
    rw [suppPreservation_iff_uniform] at h'
    /-
      case mp.mk
      F : Type u → Type u
      q : QPF F
      h : QPF.SuppPreservation
      α : Type u
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      h' : QPF.IsUniform
      ⊢ Iff (Functor.Liftp p (QPF.abs ⟨a, f⟩)) (Functor.Liftp p ⟨a, f⟩)
    -/
    dsimp only [SuppPreservation, supp] at h
    /-
      case mp.mk
      F : Type u → Type u
      q : QPF F
      h : ∀ ⦃α : Type u⦄ (x : ↑(QPF.P F) α), Eq (setOf fun y => ∀ ⦃p : α → Prop⦄, Fu …
      α : Type u
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      h' : QPF.IsUniform
      ⊢ Iff (Functor.Liftp p (QPF.abs ⟨a, f⟩)) (Functor.Liftp p ⟨a, f⟩)
    -/
    rw [liftp_iff_of_isUniform h', supp_eq_of_isUniform h', PFunctor.liftp_iff']
    /-
      case mp.mk
      F : Type u → Type u
      q : QPF F
      h : ∀ ⦃α : Type u⦄ (x : ↑(QPF.P F) α), Eq (setOf fun y => ∀ ⦃p : α → Prop⦄, Fu …
      α : Type u
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      h' : QPF.IsUniform
      ⊢ Iff (∀ (u : α), Membership.mem (Set.image f Set.univ) u → p u) (∀ (i : (QPF. …
    -/
    simp only [image_univ, mem_range, exists_imp]
    /-
      case mp.mk
      F : Type u → Type u
      q : QPF F
      h : ∀ ⦃α : Type u⦄ (x : ↑(QPF.P F) α), Eq (setOf fun y => ∀ ⦃p : α → Prop⦄, Fu …
      α : Type u
      p : α → Prop
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      h' : QPF.IsUniform
      ⊢ Iff (∀ (u : α) (x : (QPF.P F).B a), Eq (f x) u → p u) (∀ (i : (QPF.P F).B a) …
    -/
                                              /-
                                                🎉 no goals
                                              -/
    constructor <;> intros <;> subst_vars <;> solve_by_elim
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case mpr
      F : Type u → Type u
      q : QPF F
      h : QPF.LiftpPreservation
      ⊢ QPF.SuppPreservation
    -/
  · rintro α ⟨a, f⟩
    /-
      case mpr.mk
      F : Type u → Type u
      q : QPF F
      h : QPF.LiftpPreservation
      α : Type u
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      ⊢ Eq (Functor.supp (QPF.abs ⟨a, f⟩)) (Functor.supp ⟨a, f⟩)
    -/
    simp only [LiftpPreservation] at h
    /-
      case mpr.mk
      F : Type u → Type u
      q : QPF F
      h : ∀ ⦃α : Type u⦄ (p : α → Prop) (x : ↑(QPF.P F) α), Iff (Functor.Liftp p (QP …
      α : Type u
      a : (QPF.P F).A
      f : (QPF.P F).B a → α
      ⊢ Eq (Functor.supp (QPF.abs ⟨a, f⟩)) (Functor.supp ⟨a, f⟩)
    -/
    simp only [supp, h]
    /-
      🎉 no goals
    -/


theorem liftpPreservation_iff_uniform : q.LiftpPreservation ↔ q.IsUniform := by
  /-
    F : Type u → Type u
    q : QPF F
    ⊢ Iff QPF.LiftpPreservation QPF.IsUniform
  -/
  rw [← suppPreservation_iff_liftpPreservation, suppPreservation_iff_uniform]
  /-
    🎉 no goals
  -/


