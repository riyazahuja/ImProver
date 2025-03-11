/-- A polynomial functor `P` is given by a type `A` and a family `B` of types over `A`. `P` maps
any type `α` to a new type `P α`, which is defined as the sigma type `Σ x, P.B x → α`.

An element of `P α` is a pair `⟨a, f⟩`, where `a` is an element of a type `A` and
`f : B a → α`. Think of `a` as the shape of the object and `f` as an index to the relevant
elements of `α`.
-/
@[pp_with_univ]
structure PFunctor where
  /-- The head type -/
  A : Type u
  /-- The child family of types -/
  B : A → Type u


instance : Inhabited PFunctor :=
  ⟨⟨default, default⟩⟩


/-- Applying `P` to an object of `Type` -/
@[coe]
def Obj (α : Type v) :=
  Σ x : P.A, P.B x → α


instance : CoeFun PFunctor.{u} (fun _ => Type v → Type (max u v)) where
  coe := Obj


/-- Applying `P` to a morphism of `Type` -/
def map (f : α → β) : P α → P β :=
  fun ⟨a, g⟩ => ⟨a, f ∘ g⟩


instance Obj.inhabited [Inhabited P.A] [Inhabited α] : Inhabited (P α) :=
  ⟨⟨default, default⟩⟩


instance : Functor.{v, max u v} P.Obj where map := @map P


/-- We prefer `PFunctor.map` to `Functor.map` because it is universe-polymorphic. -/
@[simp]
theorem map_eq_map {α β : Type v} (f : α → β) (x : P α) : f <$> x = P.map f x :=
  rfl


@[simp]
protected theorem map_eq (f : α → β) (a : P.A) (g : P.B a → α) :
    P.map f ⟨a, g⟩ = ⟨a, f ∘ g⟩ :=
  rfl


@[simp]
protected theorem id_map : ∀ x : P α, P.map id x = x := fun ⟨_, _⟩ => rfl


@[simp]
protected theorem map_map (f : α → β) (g : β → γ) :
    ∀ x : P α, P.map g (P.map f x) = P.map (g ∘ f) x := fun ⟨_, _⟩ => rfl


instance : LawfulFunctor.{v, max u v} P.Obj where
  map_const := rfl
  id_map x := P.id_map x
  comp_map f g x := P.map_map f g x |>.symm


/-- re-export existing definition of W-types and
adapt it to a packaged definition of polynomial functor -/
def W :=
  WType P.B

/- inhabitants of W types is awkward to encode as an instance
assumption because there needs to be a value `a : P.A`
such that `P.B a` is empty to yield a finite tree -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- attribute [nolint has_nonempty_instance] W


/-- root element of a W tree -/
def W.head : W P → P.A
  | ⟨a, _f⟩ => a


/-- children of the root of a W tree -/
def W.children : ∀ x : W P, P.B (W.head x) → W P
  | ⟨_a, f⟩ => f


/-- destructor for W-types -/
def W.dest : W P → P (W P)
  | ⟨a, f⟩ => ⟨a, f⟩


/-- constructor for W-types -/
def W.mk : P (W P) → W P
  | ⟨a, f⟩ => ⟨a, f⟩


@[simp]
                                                            /-
                                                              P : PFunctor.{u}
                                                              p : ↑P P.W
                                                              ⊢ Eq (PFunctor.W.mk p).dest p
                                                            -/
theorem W.dest_mk (p : P (W P)) : W.dest (W.mk p) = p := by cases p; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
                                                        /-
                                                          P : PFunctor.{u}
                                                          p : P.W
                                                          ⊢ Eq (PFunctor.W.mk p.dest) p
                                                        -/
theorem W.mk_dest (p : W P) : W.mk (W.dest p) = p := by cases p; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- `Idx` identifies a location inside the application of a pfunctor.
For `F : PFunctor`, `x : F α` and `i : F.Idx`, `i` can designate
one part of `x` or is invalid, if `i.1 ≠ x.1` -/
def Idx :=
  Σ x : P.A, P.B x


instance Idx.inhabited [Inhabited P.A] [Inhabited (P.B default)] : Inhabited P.Idx :=
  ⟨⟨default, default⟩⟩


/-- `x.iget i` takes the component of `x` designated by `i` if any is or returns
a default value -/
def Obj.iget [DecidableEq P.A] {α} [Inhabited α] (x : P α) (i : P.Idx) : α :=
  if h : i.1 = x.1 then x.2 (cast (congr_arg _ h) i.2) else default


@[simp]
                                                                  /-
                                                                    P : PFunctor.{u}
                                                                    α : Type v₁
                                                                    β : Type v₂
                                                                    x : ↑P α
                                                                    f : α → β
                                                                    ⊢ Eq (P.map f x).fst x.fst
                                                                  -/
theorem fst_map (x : P α) (f : α → β) : (P.map f x).1 = x.1 := by cases x; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem iget_map [DecidableEq P.A] [Inhabited α] [Inhabited β] (x : P α)
    (f : α → β) (i : P.Idx) (h : i.1 = x.1) : (P.map f x).iget i = f (x.iget i) := by
  /-
    P : PFunctor.{u}
    α : Type v₁
    β : Type v₂
    inst✝² : DecidableEq P.A
    inst✝¹ : Inhabited α
    inst✝ : Inhabited β
    x : ↑P α
    f : α → β
    i : P.Idx
    h : Eq i.fst x.fst
    ⊢ Eq ((P.map f x).iget i) (f (x.iget i))
  -/
  simp only [Obj.iget, fst_map, *, dif_pos, eq_self_iff_true]
  /-
    P : PFunctor.{u}
    α : Type v₁
    β : Type v₂
    inst✝² : DecidableEq P.A
    inst✝¹ : Inhabited α
    inst✝ : Inhabited β
    x : ↑P α
    f : α → β
    i : P.Idx
    h : Eq i.fst x.fst
    ⊢ Eq ((P.map f x).snd (cast ⋯ i.snd)) (f (x.snd (cast ⋯ i.snd)))
  -/
  cases x
  /-
    case mk
    P : PFunctor.{u}
    α : Type v₁
    β : Type v₂
    inst✝² : DecidableEq P.A
    inst✝¹ : Inhabited α
    inst✝ : Inhabited β
    f : α → β
    i : P.Idx
    fst✝ : P.A
    snd✝ : P.B fst✝ → α
    h : Eq i.fst ⟨fst✝, snd✝⟩.fst
    ⊢ Eq ((P.map f ⟨fst✝, snd✝⟩).snd (cast ⋯ i.snd)) (f (⟨fst✝, snd✝⟩.snd (cast ⋯  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- functor composition for polynomial functors -/
def comp (P₂ P₁ : PFunctor.{u}) : PFunctor.{u} :=
  ⟨Σ a₂ : P₂.1, P₂.2 a₂ → P₁.1, fun a₂a₁ => Σ u : P₂.2 a₂a₁.1, P₁.2 (a₂a₁.2 u)⟩


/-- constructor for composition -/
def comp.mk (P₂ P₁ : PFunctor.{u}) {α : Type} (x : P₂ (P₁ α)) : comp P₂ P₁ α :=
  ⟨⟨x.1, Sigma.fst ∘ x.2⟩, fun a₂a₁ => (x.2 a₂a₁.1).2 a₂a₁.2⟩


/-- destructor for composition -/
def comp.get (P₂ P₁ : PFunctor.{u}) {α : Type} (x : comp P₂ P₁ α) : P₂ (P₁ α) :=
  ⟨x.1.1, fun a₂ => ⟨x.1.2 a₂, fun a₁ => x.2 ⟨a₂, a₁⟩⟩⟩


theorem liftp_iff {α : Type u} (p : α → Prop) (x : P α) :
    Liftp p x ↔ ∃ a f, x = ⟨a, f⟩ ∧ ∀ i, p (f i) := by
  /-
    P : PFunctor.{u}
    α : Type u
    p : α → Prop
    x : ↑P α
    ⊢ Iff (Functor.Liftp p x) (Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) ( …
  -/
  constructor
    /-
      case mp
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      x : ↑P α
      ⊢ Functor.Liftp p x → Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i  …
    -/
  · rintro ⟨y, hy⟩
    /-
      case mp.intro
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      x : ↑P α
      y : ↑P (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      ⊢ Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i : P.B a), p (f i))
    -/
    cases' h : y with a f
    /-
      case mp.intro.mk
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      x : ↑P α
      y : ↑P (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : P.A
      f : P.B a → Subtype p
      h : Eq y ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i : P.B a), p (f i))
    -/
    refine ⟨a, fun i => (f i).val, ?_, fun i => (f i).property⟩
    /-
      case mp.intro.mk
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      x : ↑P α
      y : ↑P (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : P.A
      f : P.B a → Subtype p
      h : Eq y ⟨a, f⟩
      ⊢ Eq x ⟨a, fun i => ↑(f i)⟩
    -/
    rw [← hy, h, map_eq_map, PFunctor.map_eq]
    /-
      case mp.intro.mk
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      x : ↑P α
      y : ↑P (Subtype p)
      hy : Eq (Functor.map Subtype.val y) x
      a : P.A
      f : P.B a → Subtype p
      h : Eq y ⟨a, f⟩
      ⊢ Eq ⟨a, Function.comp Subtype.val f⟩ ⟨a, fun i => ↑(f i)⟩
    -/
    congr
    /-
      🎉 no goals
    -/
  /-
    case mpr
    P : PFunctor.{u}
    α : Type u
    p : α → Prop
    x : ↑P α
    ⊢ (Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i : P.B a), p (f i))) …
  -/
  rintro ⟨a, f, xeq, pf⟩
  /-
    case mpr.intro.intro.intro
    P : PFunctor.{u}
    α : Type u
    p : α → Prop
    x : ↑P α
    a : P.A
    f : P.B a → α
    xeq : Eq x ⟨a, f⟩
    pf : ∀ (i : P.B a), p (f i)
    ⊢ Functor.Liftp p x
  -/
  use ⟨a, fun i => ⟨f i, pf i⟩⟩
  /-
    case h
    P : PFunctor.{u}
    α : Type u
    p : α → Prop
    x : ↑P α
    a : P.A
    f : P.B a → α
    xeq : Eq x ⟨a, f⟩
    pf : ∀ (i : P.B a), p (f i)
    ⊢ Eq (Functor.map Subtype.val ⟨a, fun i => ⟨f i, ⋯⟩⟩) x
  -/
  rw [xeq]; rfl
            /-
              🎉 no goals
            -/


theorem liftp_iff' {α : Type u} (p : α → Prop) (a : P.A) (f : P.B a → α) :
    @Liftp.{u} P.Obj _ α p ⟨a, f⟩ ↔ ∀ i, p (f i) := by
  /-
    P : PFunctor.{u}
    α : Type u
    p : α → Prop
    a : P.A
    f : P.B a → α
    ⊢ Iff (Functor.Liftp p ⟨a, f⟩) (∀ (i : P.B a), p (f i))
  -/
  simp only [liftp_iff, Sigma.mk.inj_iff]; constructor <;> intro h
    /-
      case mp
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      a : P.A
      f : P.B a → α
      h : Exists fun a_1 => Exists fun f_1 => And (Eq ⟨a, f⟩ ⟨a_1, f_1⟩) (∀ (i : P.B …
      ⊢ ∀ (i : P.B a), p (f i)
    -/
  · rcases h with ⟨a', f', heq, h'⟩
    /-
      case mp.intro.intro.intro
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      a : P.A
      f : P.B a → α
      a' : P.A
      f' : P.B a' → α
      heq : Eq ⟨a, f⟩ ⟨a', f'⟩
      h' : ∀ (i : P.B a'), p (f' i)
      ⊢ ∀ (i : P.B a), p (f i)
    -/
    cases heq
    /-
      case mp.intro.intro.intro.refl
      P : PFunctor.{u}
      α : Type u
      p : α → Prop
      a : P.A
      f : P.B a → α
      h' : ∀ (i : P.B a), p (f i)
      ⊢ ∀ (i : P.B a), p (f i)
    -/
    assumption
    /-
      🎉 no goals
    -/
  /-
    case mpr
    P : PFunctor.{u}
    α : Type u
    p : α → Prop
    a : P.A
    f : P.B a → α
    h : ∀ (i : P.B a), p (f i)
    ⊢ Exists fun a_1 => Exists fun f_1 => And (Eq ⟨a, f⟩ ⟨a_1, f_1⟩) (∀ (i : P.B a …
  -/
  repeat' first |constructor|assumption
  /-
    🎉 no goals
  -/


theorem liftr_iff {α : Type u} (r : α → α → Prop) (x y : P α) :
    Liftr r x y ↔ ∃ a f₀ f₁, x = ⟨a, f₀⟩ ∧ y = ⟨a, f₁⟩ ∧ ∀ i, r (f₀ i) (f₁ i) := by
  /-
    P : PFunctor.{u}
    α : Type u
    r : α → α → Prop
    x y : ↑P α
    ⊢ Iff (Functor.Liftr r x y) (Exists fun a => Exists fun f₀ => Exists fun f₁ => …
  -/
  constructor
    /-
      case mp
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      ⊢ Functor.Liftr r x y → Exists fun a => Exists fun f₀ => Exists fun f₁ => And  …
    -/
  · rintro ⟨u, xeq, yeq⟩
    /-
      case mp.intro.intro
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      u : ↑P (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x ⟨a, f₀⟩) (And (E …
    -/
    cases' h : u with a f
    /-
      case mp.intro.intro.mk
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      u : ↑P (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : P.A
      f : P.B a → Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x ⟨a, f₀⟩) (And (E …
    -/
    use a, fun i => (f i).val.fst, fun i => (f i).val.snd
    /-
      case h
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      u : ↑P (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : P.A
      f : P.B a → Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ And (Eq x ⟨a, fun i => (↑(f i)).1⟩) (And (Eq y ⟨a, fun i => (↑(f i)).2⟩) (∀  …
    -/
    constructor
      /-
        case h.left
        P : PFunctor.{u}
        α : Type u
        r : α → α → Prop
        x y : ↑P α
        u : ↑P (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : P.A
        f : P.B a → Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq x ⟨a, fun i => (↑(f i)).1⟩
      -/
    · rw [← xeq, h]
      /-
        case h.left
        P : PFunctor.{u}
        α : Type u
        r : α → α → Prop
        x y : ↑P α
        u : ↑P (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : P.A
        f : P.B a → Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq (Functor.map (fun t => (↑t).1) ⟨a, f⟩) ⟨a, fun i => (↑(f i)).1⟩
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      u : ↑P (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : P.A
      f : P.B a → Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ And (Eq y ⟨a, fun i => (↑(f i)).2⟩) (∀ (i : P.B a), r (↑(f i)).1 (↑(f i)).2)
    -/
    constructor
      /-
        case h.right.left
        P : PFunctor.{u}
        α : Type u
        r : α → α → Prop
        x y : ↑P α
        u : ↑P (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : P.A
        f : P.B a → Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq y ⟨a, fun i => (↑(f i)).2⟩
      -/
    · rw [← yeq, h]
      /-
        case h.right.left
        P : PFunctor.{u}
        α : Type u
        r : α → α → Prop
        x y : ↑P α
        u : ↑P (Subtype fun p => r p.1 p.2)
        xeq : Eq (Functor.map (fun t => (↑t).1) u) x
        yeq : Eq (Functor.map (fun t => (↑t).2) u) y
        a : P.A
        f : P.B a → Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq (Functor.map (fun t => (↑t).2) ⟨a, f⟩) ⟨a, fun i => (↑(f i)).2⟩
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right.right
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      u : ↑P (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : P.A
      f : P.B a → Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ ∀ (i : P.B a), r (↑(f i)).1 (↑(f i)).2
    -/
    intro i
    /-
      case h.right.right
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      u : ↑P (Subtype fun p => r p.1 p.2)
      xeq : Eq (Functor.map (fun t => (↑t).1) u) x
      yeq : Eq (Functor.map (fun t => (↑t).2) u) y
      a : P.A
      f : P.B a → Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      i : P.B a
      ⊢ r (↑(f i)).1 (↑(f i)).2
    -/
    exact (f i).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    P : PFunctor.{u}
    α : Type u
    r : α → α → Prop
    x y : ↑P α
    ⊢ (Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x ⟨a, f₀⟩) (And ( …
  -/
  rintro ⟨a, f₀, f₁, xeq, yeq, h⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    P : PFunctor.{u}
    α : Type u
    r : α → α → Prop
    x y : ↑P α
    a : P.A
    f₀ f₁ : P.B a → α
    xeq : Eq x ⟨a, f₀⟩
    yeq : Eq y ⟨a, f₁⟩
    h : ∀ (i : P.B a), r (f₀ i) (f₁ i)
    ⊢ Functor.Liftr r x y
  -/
  use ⟨a, fun i => ⟨(f₀ i, f₁ i), h i⟩⟩
  /-
    case h
    P : PFunctor.{u}
    α : Type u
    r : α → α → Prop
    x y : ↑P α
    a : P.A
    f₀ f₁ : P.B a → α
    xeq : Eq x ⟨a, f₀⟩
    yeq : Eq y ⟨a, f₁⟩
    h : ∀ (i : P.B a), r (f₀ i) (f₁ i)
    ⊢ And (Eq (Functor.map (fun t => (↑t).1) ⟨a, fun i => ⟨{ fst := f₀ i, snd := f …
  -/
  constructor
    /-
      case h.left
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      a : P.A
      f₀ f₁ : P.B a → α
      xeq : Eq x ⟨a, f₀⟩
      yeq : Eq y ⟨a, f₁⟩
      h : ∀ (i : P.B a), r (f₀ i) (f₁ i)
      ⊢ Eq (Functor.map (fun t => (↑t).1) ⟨a, fun i => ⟨{ fst := f₀ i, snd := f₁ i } …
    -/
  · rw [xeq]
    /-
      case h.left
      P : PFunctor.{u}
      α : Type u
      r : α → α → Prop
      x y : ↑P α
      a : P.A
      f₀ f₁ : P.B a → α
      xeq : Eq x ⟨a, f₀⟩
      yeq : Eq y ⟨a, f₁⟩
      h : ∀ (i : P.B a), r (f₀ i) (f₁ i)
      ⊢ Eq (Functor.map (fun t => (↑t).1) ⟨a, fun i => ⟨{ fst := f₀ i, snd := f₁ i } …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case h.right
    P : PFunctor.{u}
    α : Type u
    r : α → α → Prop
    x y : ↑P α
    a : P.A
    f₀ f₁ : P.B a → α
    xeq : Eq x ⟨a, f₀⟩
    yeq : Eq y ⟨a, f₁⟩
    h : ∀ (i : P.B a), r (f₀ i) (f₁ i)
    ⊢ Eq (Functor.map (fun t => (↑t).2) ⟨a, fun i => ⟨{ fst := f₀ i, snd := f₁ i } …
  -/
  rw [yeq]; rfl
            /-
              🎉 no goals
            -/


theorem supp_eq {α : Type u} (a : P.A) (f : P.B a → α) :
    @supp.{u} P.Obj _ α (⟨a, f⟩ : P α) = f '' univ := by
  /-
    P : PFunctor.{u}
    α : Type u
    a : P.A
    f : P.B a → α
    ⊢ Eq (Functor.supp ⟨a, f⟩) (Set.image f Set.univ)
  -/
  ext x; simp only [supp, image_univ, mem_range, mem_setOf_eq]
  /-
    case h
    P : PFunctor.{u}
    α : Type u
    a : P.A
    f : P.B a → α
    x : α
    ⊢ Iff (∀ ⦃p : α → Prop⦄, Functor.Liftp p ⟨a, f⟩ → p x) (Exists fun y => Eq (f  …
  -/
  constructor <;> intro h
    /-
      case h.mp
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      h : ∀ ⦃p : α → Prop⦄, Functor.Liftp p ⟨a, f⟩ → p x
      ⊢ Exists fun y => Eq (f y) x
    -/
  · apply @h fun x => ∃ y : P.B a, f y = x
    /-
      case h.mp
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      h : ∀ ⦃p : α → Prop⦄, Functor.Liftp p ⟨a, f⟩ → p x
      ⊢ Functor.Liftp (fun x => Exists fun y => Eq (f y) x) ⟨a, f⟩
    -/
    rw [liftp_iff']
    /-
      case h.mp
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      h : ∀ ⦃p : α → Prop⦄, Functor.Liftp p ⟨a, f⟩ → p x
      ⊢ ∀ (i : P.B a), Exists fun y => Eq (f y) (f i)
    -/
    intro
    /-
      case h.mp
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      h : ∀ ⦃p : α → Prop⦄, Functor.Liftp p ⟨a, f⟩ → p x
      i✝ : P.B a
      ⊢ Exists fun y => Eq (f y) (f i✝)
    -/
    exact ⟨_, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      h : Exists fun y => Eq (f y) x
      ⊢ ∀ ⦃p : α → Prop⦄, Functor.Liftp p ⟨a, f⟩ → p x
    -/
  · simp only [liftp_iff']
    /-
      case h.mpr
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      h : Exists fun y => Eq (f y) x
      ⊢ ∀ ⦃p : α → Prop⦄, (∀ (i : P.B a), p (f i)) → p x
    -/
    cases h
    /-
      case h.mpr.intro
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      x : α
      w✝ : P.B a
      h✝ : Eq (f w✝) x
      ⊢ ∀ ⦃p : α → Prop⦄, (∀ (i : P.B a), p (f i)) → p x
    -/
    subst x
    /-
      case h.mpr.intro
      P : PFunctor.{u}
      α : Type u
      a : P.A
      f : P.B a → α
      w✝ : P.B a
      ⊢ ∀ ⦃p : α → Prop⦄, (∀ (i : P.B a), p (f i)) → p (f w✝)
    -/
    tauto
    /-
      🎉 no goals
    -/


