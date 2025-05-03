/-- Multivariate quotients of polynomial functors.
-/
class MvQPF {n : ℕ} (F : TypeVec.{u} n → Type*) extends MvFunctor F where
  P : MvPFunctor.{u} n
  abs : ∀ {α}, P α → F α
  repr : ∀ {α}, F α → P α
  abs_repr : ∀ {α} (x : F α), abs (repr x) = x
  abs_map : ∀ {α β} (f : α ⟹ β) (p : P α), abs (f <$$> p) = f <$$> abs p


protected theorem id_map {α : TypeVec n} (x : F α) : TypeVec.id <$$> x = x := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    ⊢ Eq (MvFunctor.map TypeVec.id x) x
  -/
  rw [← abs_repr x, ← abs_map]
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    ⊢ Eq (MvQPF.abs (MvFunctor.map TypeVec.id (MvQPF.repr x))) (MvQPF.abs (MvQPF.r …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_map {α β γ : TypeVec n} (f : α ⟹ β) (g : β ⟹ γ) (x : F α) :
    (g ⊚ f) <$$> x = g <$$> f <$$> x := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α β γ : TypeVec.{u} n
    f : α.Arrow β
    g : β.Arrow γ
    x : F α
    ⊢ Eq (MvFunctor.map (TypeVec.comp g f) x) (MvFunctor.map g (MvFunctor.map f x))
  -/
  rw [← abs_repr x, ← abs_map, ← abs_map, ← abs_map]
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α β γ : TypeVec.{u} n
    f : α.Arrow β
    g : β.Arrow γ
    x : F α
    ⊢ Eq (MvQPF.abs (MvFunctor.map (TypeVec.comp g f) (MvQPF.repr x))) (MvQPF.abs  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance (priority := 100) lawfulMvFunctor : LawfulMvFunctor F where
  id_map := @MvQPF.id_map n F _
  comp_map := @comp_map n F _

-- Lifting predicates and relations

theorem liftP_iff {α : TypeVec n} (p : ∀ ⦃i⦄, α i → Prop) (x : F α) :
    LiftP p x ↔ ∃ a f, x = abs ⟨a, f⟩ ∧ ∀ i j, p (f i j) := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : F α
    ⊢ Iff (MvFunctor.LiftP p x) (Exists fun a => Exists fun f => And (Eq x (MvQPF. …
  -/
  constructor
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : F α
      ⊢ MvFunctor.LiftP p x → Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨ …
    -/
  · rintro ⟨y, hy⟩
    /-
      case mp.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : F α
      y : F fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      ⊢ Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : Fin2 n …
    -/
    cases' h : repr y with a f
    /-
      case mp.intro.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : F α
      y : F fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype p
      h : Eq (MvQPF.repr y) ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : Fin2 n …
    -/
    use a, fun i j => (f i j).val
    /-
      case h
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : F α
      y : F fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype p
      h : Eq (MvQPF.repr y) ⟨a, f⟩
      ⊢ And (Eq x (MvQPF.abs ⟨a, fun i j => ↑(f i j)⟩)) (∀ (i : Fin2 n) (j : (MvQPF. …
    -/
    constructor
      /-
        case h.left
        n : Nat
        F : TypeVec.{u} n → Type u_1
        q : MvQPF F
        α : TypeVec.{u} n
        p : ⦃i : Fin2 n⦄ → α i → Prop
        x : F α
        y : F fun i => Subtype p
        hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
        a : (MvQPF.P F).A
        f : ((MvQPF.P F).B a).Arrow fun i => Subtype p
        h : Eq (MvQPF.repr y) ⟨a, f⟩
        ⊢ Eq x (MvQPF.abs ⟨a, fun i j => ↑(f i j)⟩)
      -/
    · rw [← hy, ← abs_repr y, h, ← abs_map]; rfl
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case h.right
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : F α
      y : F fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype p
      h : Eq (MvQPF.repr y) ⟨a, f⟩
      ⊢ ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), p ↑(f i j)
    -/
    intro i j
    /-
      case h.right
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : F α
      y : F fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype p
      h : Eq (MvQPF.repr y) ⟨a, f⟩
      i : Fin2 n
      j : (MvQPF.P F).B a i
      ⊢ p ↑(f i j)
    -/
    apply (f i j).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : F α
    ⊢ (Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : Fin2  …
  -/
  rintro ⟨a, f, h₀, h₁⟩
  /-
    case mpr.intro.intro.intro
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    h₀ : Eq x (MvQPF.abs ⟨a, f⟩)
    h₁ : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), p (f i j)
    ⊢ MvFunctor.LiftP p x
  -/
  use abs ⟨a, fun i j => ⟨f i j, h₁ i j⟩⟩
  /-
    case h
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    h₀ : Eq x (MvQPF.abs ⟨a, f⟩)
    h₁ : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), p (f i j)
    ⊢ Eq (MvFunctor.map (fun i => Subtype.val) (MvQPF.abs ⟨a, fun i j => ⟨f i j, ⋯ …
  -/
  rw [← abs_map, h₀]; rfl
                      /-
                        🎉 no goals
                      -/


theorem liftR_iff {α : TypeVec n} (r : ∀ /- ⦃i⦄ -/ {i}, α i → α i → Prop) (x y : F α) :
    LiftR r x y ↔ ∃ a f₀ f₁, x = abs ⟨a, f₀⟩ ∧ y = abs ⟨a, f₁⟩ ∧ ∀ i j, r (f₀ i j) (f₁ i j) := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    r : {i : Fin2 n} → α i → α i → Prop
    x y : F α
    ⊢ Iff (MvFunctor.LiftR (fun {i} => r) x y) (Exists fun a => Exists fun f₀ => E …
  -/
  constructor
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      ⊢ MvFunctor.LiftR (fun {i} => r) x y → Exists fun a => Exists fun f₀ => Exists …
    -/
  · rintro ⟨u, xeq, yeq⟩
    /-
      case mp.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x (MvQPF.abs ⟨a, f …
    -/
    cases' h : repr u with a f
    /-
      case mp.intro.intro.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      h : Eq (MvQPF.repr u) ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x (MvQPF.abs ⟨a, f …
    -/
    use a, fun i j => (f i j).val.fst, fun i j => (f i j).val.snd
    /-
      case h
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      h : Eq (MvQPF.repr u) ⟨a, f⟩
      ⊢ And (Eq x (MvQPF.abs ⟨a, fun i j => (↑(f i j)).1⟩)) (And (Eq y (MvQPF.abs ⟨a …
    -/
    constructor
      /-
        case h.left
        n : Nat
        F : TypeVec.{u} n → Type u_1
        q : MvQPF F
        α : TypeVec.{u} n
        r : {i : Fin2 n} → α i → α i → Prop
        x y : F α
        u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
        xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
        yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
        a : (MvQPF.P F).A
        f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
        h : Eq (MvQPF.repr u) ⟨a, f⟩
        ⊢ Eq x (MvQPF.abs ⟨a, fun i j => (↑(f i j)).1⟩)
      -/
    · rw [← xeq, ← abs_repr u, h, ← abs_map]; rfl
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case h.right
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      h : Eq (MvQPF.repr u) ⟨a, f⟩
      ⊢ And (Eq y (MvQPF.abs ⟨a, fun i j => (↑(f i j)).2⟩)) (∀ (i : Fin2 n) (j : (Mv …
    -/
    constructor
      /-
        case h.right.left
        n : Nat
        F : TypeVec.{u} n → Type u_1
        q : MvQPF F
        α : TypeVec.{u} n
        r : {i : Fin2 n} → α i → α i → Prop
        x y : F α
        u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
        xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
        yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
        a : (MvQPF.P F).A
        f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
        h : Eq (MvQPF.repr u) ⟨a, f⟩
        ⊢ Eq y (MvQPF.abs ⟨a, fun i j => (↑(f i j)).2⟩)
      -/
    · rw [← yeq, ← abs_repr u, h, ← abs_map]; rfl
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case h.right.right
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      h : Eq (MvQPF.repr u) ⟨a, f⟩
      ⊢ ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), r (↑(f i j)).1 (↑(f i j)).2
    -/
    intro i j
    /-
      case h.right.right
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      u : F fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow fun i => Subtype fun p => (fun {i} => r) p.1 p.2
      h : Eq (MvQPF.repr u) ⟨a, f⟩
      i : Fin2 n
      j : (MvQPF.P F).B a i
      ⊢ r (↑(f i j)).1 (↑(f i j)).2
    -/
    exact (f i j).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    r : {i : Fin2 n} → α i → α i → Prop
    x y : F α
    ⊢ (Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x (MvQPF.abs ⟨a,  …
  -/
  rintro ⟨a, f₀, f₁, xeq, yeq, h⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    r : {i : Fin2 n} → α i → α i → Prop
    x y : F α
    a : (MvQPF.P F).A
    f₀ f₁ : ((MvQPF.P F).B a).Arrow α
    xeq : Eq x (MvQPF.abs ⟨a, f₀⟩)
    yeq : Eq y (MvQPF.abs ⟨a, f₁⟩)
    h : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), r (f₀ i j) (f₁ i j)
    ⊢ MvFunctor.LiftR (fun {i} => r) x y
  -/
  use abs ⟨a, fun i j => ⟨(f₀ i j, f₁ i j), h i j⟩⟩
  /-
    case h
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    r : {i : Fin2 n} → α i → α i → Prop
    x y : F α
    a : (MvQPF.P F).A
    f₀ f₁ : ((MvQPF.P F).B a).Arrow α
    xeq : Eq x (MvQPF.abs ⟨a, f₀⟩)
    yeq : Eq y (MvQPF.abs ⟨a, f₁⟩)
    h : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), r (f₀ i j) (f₁ i j)
    ⊢ And (Eq (MvFunctor.map (fun i t => (↑t).1) (MvQPF.abs ⟨a, fun i j => ⟨{ fst  …
  -/
  dsimp; constructor
    /-
      case h.left
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      r : {i : Fin2 n} → α i → α i → Prop
      x y : F α
      a : (MvQPF.P F).A
      f₀ f₁ : ((MvQPF.P F).B a).Arrow α
      xeq : Eq x (MvQPF.abs ⟨a, f₀⟩)
      yeq : Eq y (MvQPF.abs ⟨a, f₁⟩)
      h : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), r (f₀ i j) (f₁ i j)
      ⊢ Eq (MvFunctor.map (fun i t => (↑t).1) (MvQPF.abs ⟨a, fun i j => ⟨{ fst := f₀ …
    -/
  · rw [xeq, ← abs_map]; rfl
                         /-
                           🎉 no goals
                         -/
  /-
    case h.right
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    r : {i : Fin2 n} → α i → α i → Prop
    x y : F α
    a : (MvQPF.P F).A
    f₀ f₁ : ((MvQPF.P F).B a).Arrow α
    xeq : Eq x (MvQPF.abs ⟨a, f₀⟩)
    yeq : Eq y (MvQPF.abs ⟨a, f₁⟩)
    h : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), r (f₀ i j) (f₁ i j)
    ⊢ Eq (MvFunctor.map (fun i t => (↑t).2) (MvQPF.abs ⟨a, fun i j => ⟨{ fst := f₀ …
  -/
  rw [yeq, ← abs_map]; rfl
                       /-
                         🎉 no goals
                       -/


theorem mem_supp {α : TypeVec n} (x : F α) (i) (u : α i) :
    u ∈ supp x i ↔ ∀ a f, abs ⟨a, f⟩ = x → u ∈ f i '' univ := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    i : Fin2 n
    u : α i
    ⊢ Iff (Membership.mem (MvFunctor.supp x i) u) (∀ (a : (MvQPF.P F).A) (f : ((Mv …
  -/
  rw [supp]; dsimp; constructor
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      i : Fin2 n
      u : α i
      ⊢ (∀ ⦃P : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P x → P i u) → ∀ (a : (M …
    -/
  · intro h a f haf
    have : LiftP (fun i u => u ∈ f i '' univ) x := by
      rw [liftP_iff]
      refine ⟨a, f, haf.symm, ?_⟩
      intro i u
      exact mem_image_of_mem _ (mem_univ _)
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      i : Fin2 n
      u : α i
      h : ∀ ⦃P : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P x → P i u
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      haf : Eq (MvQPF.abs ⟨a, f⟩) x
      this : MvFunctor.LiftP (fun i u => Membership.mem (Set.image (f i) Set.univ) u …
      ⊢ Membership.mem (Set.image (f i) Set.univ) u
    -/
    exact h this
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    i : Fin2 n
    u : α i
    ⊢ (∀ (a : (MvQPF.P F).A) (f : ((MvQPF.P F).B a).Arrow α), Eq (MvQPF.abs ⟨a, f⟩ …
  -/
  intro h p; rw [liftP_iff]
  /-
    case mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    i : Fin2 n
    u : α i
    h : ∀ (a : (MvQPF.P F).A) (f : ((MvQPF.P F).B a).Arrow α), Eq (MvQPF.abs ⟨a, f …
    p : (i : Fin2 n) → α i → Prop
    ⊢ (Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : Fin2  …
  -/
  rintro ⟨a, f, xeq, h'⟩
  /-
    case mpr.intro.intro.intro
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    i : Fin2 n
    u : α i
    h : ∀ (a : (MvQPF.P F).A) (f : ((MvQPF.P F).B a).Arrow α), Eq (MvQPF.abs ⟨a, f …
    p : (i : Fin2 n) → α i → Prop
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq x (MvQPF.abs ⟨a, f⟩)
    h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), p i (f i j)
    ⊢ p i u
  -/
  rcases h a f xeq.symm with ⟨i, _, hi⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    i✝ : Fin2 n
    u : α i✝
    h : ∀ (a : (MvQPF.P F).A) (f : ((MvQPF.P F).B a).Arrow α), Eq (MvQPF.abs ⟨a, f …
    p : (i : Fin2 n) → α i → Prop
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq x (MvQPF.abs ⟨a, f⟩)
    h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), p i (f i j)
    i : (MvQPF.P F).B a i✝
    left✝ : Membership.mem Set.univ i
    hi : Eq (f i✝ i) u
    ⊢ p i✝ u
  -/
  rw [← hi]; apply h'
             /-
               🎉 no goals
             -/


theorem supp_eq {α : TypeVec n} {i} (x : F α) :
                                                                     /-
                                                                       n : Nat
                                                                       F : TypeVec.{u} n → Type u_1
                                                                       q : MvQPF F
                                                                       α : TypeVec.{u} n
                                                                       i : Fin2 n
                                                                       x : F α
                                                                       ⊢ Eq (MvFunctor.supp x i) (setOf fun u => ∀ (a : (MvQPF.P F).A) (f : ((MvQPF.P …
                                                                     -/
    supp x i = { u | ∀ a f, abs ⟨a, f⟩ = x → u ∈ f i '' univ } := by ext; apply mem_supp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem has_good_supp_iff {α : TypeVec n} (x : F α) :
    (∀ p, LiftP p x ↔ ∀ (i), ∀ u ∈ supp x i, p i u) ↔
      ∃ a f, abs ⟨a, f⟩ = x ∧ ∀ i a' f', abs ⟨a', f'⟩ = x → f i '' univ ⊆ f' i '' univ := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    ⊢ Iff (∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fi …
  -/
  constructor
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      ⊢ (∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2 n …
    -/
  · intro h
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      ⊢ Exists fun a => Exists fun f => And (Eq (MvQPF.abs ⟨a, f⟩) x) (∀ (i : Fin2 n …
    -/
    have : LiftP (supp x) x := by rw [h]; introv; exact id
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      this : MvFunctor.LiftP (MvFunctor.supp x) x
      ⊢ Exists fun a => Exists fun f => And (Eq (MvQPF.abs ⟨a, f⟩) x) (∀ (i : Fin2 n …
    -/
    rw [liftP_iff] at this
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      this : Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : F …
      ⊢ Exists fun a => Exists fun f => And (Eq (MvQPF.abs ⟨a, f⟩) x) (∀ (i : Fin2 n …
    -/
    rcases this with ⟨a, f, xeq, h'⟩
    /-
      case mp.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq x (MvQPF.abs ⟨a, f⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), MvFunctor.supp x i (f i j)
      ⊢ Exists fun a => Exists fun f => And (Eq (MvQPF.abs ⟨a, f⟩) x) (∀ (i : Fin2 n …
    -/
    refine ⟨a, f, xeq.symm, ?_⟩
    /-
      case mp.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq x (MvQPF.abs ⟨a, f⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), MvFunctor.supp x i (f i j)
      ⊢ ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq (M …
    -/
    intro a' f' h''
    /-
      case mp.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq x (MvQPF.abs ⟨a, f⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), MvFunctor.supp x i (f i j)
      a' : Fin2 n
      f' : (MvQPF.P F).A
      h'' : ((MvQPF.P F).B f').Arrow α
      ⊢ Eq (MvQPF.abs ⟨f', h''⟩) x → HasSubset.Subset (Set.image (f a') Set.univ) (S …
    -/
    rintro hu u ⟨j, _h₂, hfi⟩
    /-
      case mp.intro.intro.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq x (MvQPF.abs ⟨a, f⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), MvFunctor.supp x i (f i j)
      a' : Fin2 n
      f' : (MvQPF.P F).A
      h'' : ((MvQPF.P F).B f').Arrow α
      hu : Eq (MvQPF.abs ⟨f', h''⟩) x
      u : α a'
      j : (MvQPF.P F).B a a'
      _h₂ : Membership.mem Set.univ j
      hfi : Eq (f a' j) u
      ⊢ Membership.mem (Set.image (h'' a') Set.univ) u
    -/
    have hh : u ∈ supp x a' := by rw [← hfi]; apply h'
    /-
      case mp.intro.intro.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      h : ∀ (p : (i : Fin2 n) → α i → Prop), Iff (MvFunctor.LiftP p x) (∀ (i : Fin2  …
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq x (MvQPF.abs ⟨a, f⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a i), MvFunctor.supp x i (f i j)
      a' : Fin2 n
      f' : (MvQPF.P F).A
      h'' : ((MvQPF.P F).B f').Arrow α
      hu : Eq (MvQPF.abs ⟨f', h''⟩) x
      u : α a'
      j : (MvQPF.P F).B a a'
      _h₂ : Membership.mem Set.univ j
      hfi : Eq (f a' j) u
      hh : Membership.mem (MvFunctor.supp x a') u
      ⊢ Membership.mem (Set.image (h'' a') Set.univ) u
    -/
    exact (mem_supp x _ u).mp hh _ _ hu
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    ⊢ (Exists fun a => Exists fun f => And (Eq (MvQPF.abs ⟨a, f⟩) x) (∀ (i : Fin2  …
  -/
  rintro ⟨a, f, xeq, h⟩ p; rw [liftP_iff]; constructor
    /-
      case mpr.intro.intro.intro.mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq (MvQPF.abs ⟨a, f⟩) x
      h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
      p : (i : Fin2 n) → α i → Prop
      ⊢ (Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : Fin2  …
    -/
  · rintro ⟨a', f', xeq', h'⟩ i u usuppx
    /-
      case mpr.intro.intro.intro.mp.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq (MvQPF.abs ⟨a, f⟩) x
      h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
      p : (i : Fin2 n) → α i → Prop
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      xeq' : Eq x (MvQPF.abs ⟨a', f'⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      i : Fin2 n
      u : α i
      usuppx : Membership.mem (MvFunctor.supp x i) u
      ⊢ p i u
    -/
    rcases (mem_supp x _ u).mp (@usuppx) a' f' xeq'.symm with ⟨i, _, f'ieq⟩
    /-
      case mpr.intro.intro.intro.mp.intro.intro.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq (MvQPF.abs ⟨a, f⟩) x
      h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
      p : (i : Fin2 n) → α i → Prop
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      xeq' : Eq x (MvQPF.abs ⟨a', f'⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      i✝ : Fin2 n
      u : α i✝
      usuppx : Membership.mem (MvFunctor.supp x i✝) u
      i : (MvQPF.P F).B a' i✝
      left✝ : Membership.mem Set.univ i
      f'ieq : Eq (f' i✝ i) u
      ⊢ p i✝ u
    -/
    rw [← f'ieq]
    /-
      case mpr.intro.intro.intro.mp.intro.intro.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      α : TypeVec.{u} n
      x : F α
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      xeq : Eq (MvQPF.abs ⟨a, f⟩) x
      h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
      p : (i : Fin2 n) → α i → Prop
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      xeq' : Eq x (MvQPF.abs ⟨a', f'⟩)
      h' : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      i✝ : Fin2 n
      u : α i✝
      usuppx : Membership.mem (MvFunctor.supp x i✝) u
      i : (MvQPF.P F).B a' i✝
      left✝ : Membership.mem Set.univ i
      f'ieq : Eq (f' i✝ i) u
      ⊢ p i✝ (f' i✝ i)
    -/
    apply h'
    /-
      🎉 no goals
    -/
  /-
    case mpr.intro.intro.intro.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq (MvQPF.abs ⟨a, f⟩) x
    h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
    p : (i : Fin2 n) → α i → Prop
    ⊢ (∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp x i) u → p i u) →  …
  -/
  intro h'
  /-
    case mpr.intro.intro.intro.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq (MvQPF.abs ⟨a, f⟩) x
    h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
    p : (i : Fin2 n) → α i → Prop
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp x i) u → p i u
    ⊢ Exists fun a => Exists fun f => And (Eq x (MvQPF.abs ⟨a, f⟩)) (∀ (i : Fin2 n …
  -/
  refine ⟨a, f, xeq.symm, ?_⟩; intro j y
  /-
    case mpr.intro.intro.intro.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq (MvQPF.abs ⟨a, f⟩) x
    h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
    p : (i : Fin2 n) → α i → Prop
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp x i) u → p i u
    j : Fin2 n
    y : (MvQPF.P F).B a j
    ⊢ p j (f j y)
  -/
  apply h'; rw [mem_supp]
  /-
    case mpr.intro.intro.intro.mpr.a
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq (MvQPF.abs ⟨a, f⟩) x
    h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
    p : (i : Fin2 n) → α i → Prop
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp x i) u → p i u
    j : Fin2 n
    y : (MvQPF.P F).B a j
    ⊢ ∀ (a : (MvQPF.P F).A) (f_1 : ((MvQPF.P F).B a).Arrow α), Eq (MvQPF.abs ⟨a, f …
  -/
  intro a' f' xeq'
  /-
    case mpr.intro.intro.intro.mpr.a
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq (MvQPF.abs ⟨a, f⟩) x
    h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
    p : (i : Fin2 n) → α i → Prop
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp x i) u → p i u
    j : Fin2 n
    y : (MvQPF.P F).B a j
    a' : (MvQPF.P F).A
    f' : ((MvQPF.P F).B a').Arrow α
    xeq' : Eq (MvQPF.abs ⟨a', f'⟩) x
    ⊢ Membership.mem (Set.image (f' j) Set.univ) (f j y)
  -/
  apply h _ a' f' xeq'
  /-
    case mpr.intro.intro.intro.mpr.a.a
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    α : TypeVec.{u} n
    x : F α
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    xeq : Eq (MvQPF.abs ⟨a, f⟩) x
    h : ∀ (i : Fin2 n) (a' : (MvQPF.P F).A) (f' : ((MvQPF.P F).B a').Arrow α), Eq  …
    p : (i : Fin2 n) → α i → Prop
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp x i) u → p i u
    j : Fin2 n
    y : (MvQPF.P F).B a j
    a' : (MvQPF.P F).A
    f' : ((MvQPF.P F).B a').Arrow α
    xeq' : Eq (MvQPF.abs ⟨a', f'⟩) x
    ⊢ Membership.mem (Set.image (f j) Set.univ) (f j y)
  -/
  apply mem_image_of_mem _ (mem_univ _)
  /-
    🎉 no goals
  -/


/-- A qpf is said to be uniform if every polynomial functor
representing a single value all have the same range. -/
def IsUniform : Prop :=
  ∀ ⦃α : TypeVec n⦄ (a a' : q.P.A) (f : q.P.B a ⟹ α) (f' : q.P.B a' ⟹ α),
    abs ⟨a, f⟩ = abs ⟨a', f'⟩ → ∀ i, f i '' univ = f' i '' univ


/-- does `abs` preserve `liftp`? -/
def LiftPPreservation : Prop :=
  ∀ ⦃α : TypeVec n⦄ (p : ∀ ⦃i⦄, α i → Prop) (x : q.P α), LiftP p (abs x) ↔ LiftP p x


/-- does `abs` preserve `supp`? -/
def SuppPreservation : Prop :=
  ∀ ⦃α⦄ (x : q.P α), supp (abs x) = supp x


theorem supp_eq_of_isUniform (h : q.IsUniform) {α : TypeVec n} (a : q.P.A) (f : q.P.B a ⟹ α) :
    ∀ i, supp (abs ⟨a, f⟩) i = f i '' univ := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    ⊢ ∀ (i : Fin2 n), Eq (MvFunctor.supp (MvQPF.abs ⟨a, f⟩) i) (Set.image (f i) Se …
  -/
  intro; ext u; rw [mem_supp]; constructor
    /-
      case h.mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      i✝ : Fin2 n
      u : α i✝
      ⊢ (∀ (a_1 : (MvQPF.P F).A) (f_1 : ((MvQPF.P F).B a_1).Arrow α), Eq (MvQPF.abs  …
    -/
  · intro h'
    /-
      case h.mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      i✝ : Fin2 n
      u : α i✝
      h' : ∀ (a_1 : (MvQPF.P F).A) (f_1 : ((MvQPF.P F).B a_1).Arrow α), Eq (MvQPF.ab …
      ⊢ Membership.mem (Set.image (f i✝) Set.univ) u
    -/
    apply h' _ _ rfl
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    i✝ : Fin2 n
    u : α i✝
    ⊢ Membership.mem (Set.image (f i✝) Set.univ) u → ∀ (a_2 : (MvQPF.P F).A) (f_1  …
  -/
  intro h' a' f' e
  /-
    case h.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    i✝ : Fin2 n
    u : α i✝
    h' : Membership.mem (Set.image (f i✝) Set.univ) u
    a' : (MvQPF.P F).A
    f' : ((MvQPF.P F).B a').Arrow α
    e : Eq (MvQPF.abs ⟨a', f'⟩) (MvQPF.abs ⟨a, f⟩)
    ⊢ Membership.mem (Set.image (f' i✝) Set.univ) u
  -/
  rw [← h _ _ _ _ e.symm]; apply h'
                           /-
                             🎉 no goals
                           -/


theorem liftP_iff_of_isUniform (h : q.IsUniform) {α : TypeVec n} (x : F α) (p : ∀ i, α i → Prop) :
    LiftP p x ↔ ∀ (i), ∀ u ∈ supp x i, p i u := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    x : F α
    p : (i : Fin2 n) → α i → Prop
    ⊢ Iff (MvFunctor.LiftP p x) (∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunct …
  -/
  rw [liftP_iff, ← abs_repr x]
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    x : F α
    p : (i : Fin2 n) → α i → Prop
    ⊢ Iff (Exists fun a => Exists fun f => And (Eq (MvQPF.abs (MvQPF.repr x)) (MvQ …
  -/
  cases' repr x with a f; constructor
    /-
      case mk.mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      x : F α
      p : (i : Fin2 n) → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      ⊢ (Exists fun a_1 => Exists fun f_1 => And (Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨ …
    -/
  · rintro ⟨a', f', abseq, hf⟩ u
    /-
      case mk.mp.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      x : F α
      p : (i : Fin2 n) → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      abseq : Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨a', f'⟩)
      hf : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      u : Fin2 n
      ⊢ ∀ (u_1 : α u), Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f⟩) u) u_1 → p  …
    -/
    rw [supp_eq_of_isUniform h, h _ _ _ _ abseq]
    /-
      case mk.mp.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      x : F α
      p : (i : Fin2 n) → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      abseq : Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨a', f'⟩)
      hf : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      u : Fin2 n
      ⊢ ∀ (u_1 : α u), Membership.mem (Set.image (f' u) Set.univ) u_1 → p u u_1
    -/
    rintro b ⟨i, _, hi⟩
    /-
      case mk.mp.intro.intro.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      x : F α
      p : (i : Fin2 n) → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      abseq : Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨a', f'⟩)
      hf : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      u : Fin2 n
      b : α u
      i : (MvQPF.P F).B a' u
      left✝ : Membership.mem Set.univ i
      hi : Eq (f' u i) b
      ⊢ p u b
    -/
    rw [← hi]
    /-
      case mk.mp.intro.intro.intro.intro.intro
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      x : F α
      p : (i : Fin2 n) → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      a' : (MvQPF.P F).A
      f' : ((MvQPF.P F).B a').Arrow α
      abseq : Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨a', f'⟩)
      hf : ∀ (i : Fin2 n) (j : (MvQPF.P F).B a' i), p i (f' i j)
      u : Fin2 n
      b : α u
      i : (MvQPF.P F).B a' u
      left✝ : Membership.mem Set.univ i
      hi : Eq (f' u i) b
      ⊢ p u (f' u i)
    -/
    apply hf
    /-
      🎉 no goals
    -/
  /-
    case mk.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    x : F α
    p : (i : Fin2 n) → α i → Prop
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    ⊢ (∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f⟩) …
  -/
  intro h'
  /-
    case mk.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    x : F α
    p : (i : Fin2 n) → α i → Prop
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f …
    ⊢ Exists fun a_1 => Exists fun f_1 => And (Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨a …
  -/
  refine ⟨a, f, rfl, fun _ i => h' _ _ ?_⟩
  /-
    case mk.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    x : F α
    p : (i : Fin2 n) → α i → Prop
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f …
    x✝ : Fin2 n
    i : (MvQPF.P F).B a x✝
    ⊢ Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f⟩) x✝) (f x✝ i)
  -/
  rw [supp_eq_of_isUniform h]
  /-
    case mk.mpr
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α : TypeVec.{u} n
    x : F α
    p : (i : Fin2 n) → α i → Prop
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    h' : ∀ (i : Fin2 n) (u : α i), Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f …
    x✝ : Fin2 n
    i : (MvQPF.P F).B a x✝
    ⊢ Membership.mem (Set.image (f x✝) Set.univ) (f x✝ i)
  -/
  exact ⟨i, mem_univ i, rfl⟩
  /-
    🎉 no goals
  -/


theorem supp_map (h : q.IsUniform) {α β : TypeVec n} (g : α ⟹ β) (x : F α) (i) :
    supp (g <$$> x) i = g i '' supp x i := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α β : TypeVec.{u} n
    g : α.Arrow β
    x : F α
    i : Fin2 n
    ⊢ Eq (MvFunctor.supp (MvFunctor.map g x) i) (Set.image (g i) (MvFunctor.supp x …
  -/
  rw [← abs_repr x]; cases' repr x with a f; rw [← abs_map, MvPFunctor.map_eq]
  /-
    case mk
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α β : TypeVec.{u} n
    g : α.Arrow β
    x : F α
    i : Fin2 n
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    ⊢ Eq (MvFunctor.supp (MvQPF.abs ⟨a, TypeVec.comp g f⟩) i) (Set.image (g i) (Mv …
  -/
  rw [supp_eq_of_isUniform h, supp_eq_of_isUniform h, ← image_comp]
  /-
    case mk
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    h : MvQPF.IsUniform
    α β : TypeVec.{u} n
    g : α.Arrow β
    x : F α
    i : Fin2 n
    a : (MvQPF.P F).A
    f : ((MvQPF.P F).B a).Arrow α
    ⊢ Eq (Set.image (TypeVec.comp g f i) Set.univ) (Set.image (Function.comp (g i) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem suppPreservation_iff_isUniform : q.SuppPreservation ↔ q.IsUniform := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    ⊢ Iff MvQPF.SuppPreservation MvQPF.IsUniform
  -/
  constructor
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      ⊢ MvQPF.SuppPreservation → MvQPF.IsUniform
    -/
  · intro h α a a' f f' h' i
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.SuppPreservation
      α : TypeVec.{u} n
      a a' : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      f' : ((MvQPF.P F).B a').Arrow α
      h' : Eq (MvQPF.abs ⟨a, f⟩) (MvQPF.abs ⟨a', f'⟩)
      i : Fin2 n
      ⊢ Eq (Set.image (f i) Set.univ) (Set.image (f' i) Set.univ)
    -/
    rw [← MvPFunctor.supp_eq, ← MvPFunctor.supp_eq, ← h, h', h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      ⊢ MvQPF.IsUniform → MvQPF.SuppPreservation
    -/
  · rintro h α ⟨a, f⟩
    /-
      case mpr.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      ⊢ Eq (MvFunctor.supp (MvQPF.abs ⟨a, f⟩)) (MvFunctor.supp ⟨a, f⟩)
    -/
    ext
    /-
      case mpr.mk.h.h
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.IsUniform
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      x✝¹ : Fin2 n
      x✝ : α x✝¹
      ⊢ Iff (Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f⟩) x✝¹) x✝) (Membership. …
    -/
    rwa [supp_eq_of_isUniform, MvPFunctor.supp_eq]
    /-
      🎉 no goals
    -/


theorem suppPreservation_iff_liftpPreservation : q.SuppPreservation ↔ q.LiftPPreservation := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    ⊢ Iff MvQPF.SuppPreservation MvQPF.LiftPPreservation
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.SuppPreservation
      ⊢ MvQPF.LiftPPreservation
    -/
  · rintro α p ⟨a, f⟩
    /-
      case mp.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.SuppPreservation
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      ⊢ Iff (MvFunctor.LiftP p (MvQPF.abs ⟨a, f⟩)) (MvFunctor.LiftP p ⟨a, f⟩)
    -/
    have h' := h
    /-
      case mp.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.SuppPreservation
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      h' : MvQPF.SuppPreservation
      ⊢ Iff (MvFunctor.LiftP p (MvQPF.abs ⟨a, f⟩)) (MvFunctor.LiftP p ⟨a, f⟩)
    -/
    rw [suppPreservation_iff_isUniform] at h'
    /-
      case mp.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.SuppPreservation
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      h' : MvQPF.IsUniform
      ⊢ Iff (MvFunctor.LiftP p (MvQPF.abs ⟨a, f⟩)) (MvFunctor.LiftP p ⟨a, f⟩)
    -/
    dsimp only [SuppPreservation, supp] at h
    simp only [liftP_iff_of_isUniform, supp_eq_of_isUniform, MvPFunctor.liftP_iff', h',
      image_univ, mem_range, exists_imp]
    /-
      case mp.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : ∀ ⦃α : TypeVec.{u} n⦄ (x : ↑(MvQPF.P F) α), Eq (MvFunctor.supp (MvQPF.abs  …
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      h' : MvQPF.IsUniform
      ⊢ Iff (∀ (i : Fin2 n) (u : α i) (x : (MvQPF.P F).B a i), Eq (f i x) u → p u) ( …
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
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.LiftPPreservation
      ⊢ MvQPF.SuppPreservation
    -/
  · rintro α ⟨a, f⟩
    /-
      case mpr.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : MvQPF.LiftPPreservation
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      ⊢ Eq (MvFunctor.supp (MvQPF.abs ⟨a, f⟩)) (MvFunctor.supp ⟨a, f⟩)
    -/
    simp only [LiftPPreservation] at h
    /-
      case mpr.mk
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : ∀ ⦃α : TypeVec.{u} n⦄ (p : ⦃i : Fin2 n⦄ → α i → Prop) (x : ↑(MvQPF.P F) α) …
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      ⊢ Eq (MvFunctor.supp (MvQPF.abs ⟨a, f⟩)) (MvFunctor.supp ⟨a, f⟩)
    -/
    ext
    /-
      case mpr.mk.h.h
      n : Nat
      F : TypeVec.{u} n → Type u_1
      q : MvQPF F
      h : ∀ ⦃α : TypeVec.{u} n⦄ (p : ⦃i : Fin2 n⦄ → α i → Prop) (x : ↑(MvQPF.P F) α) …
      α : TypeVec.{u} n
      a : (MvQPF.P F).A
      f : ((MvQPF.P F).B a).Arrow α
      x✝¹ : Fin2 n
      x✝ : α x✝¹
      ⊢ Iff (Membership.mem (MvFunctor.supp (MvQPF.abs ⟨a, f⟩) x✝¹) x✝) (Membership. …
    -/
    simp only [supp, h, mem_setOf_eq]
    /-
      🎉 no goals
    -/


theorem liftpPreservation_iff_uniform : q.LiftPPreservation ↔ q.IsUniform := by
  /-
    n : Nat
    F : TypeVec.{u} n → Type u_1
    q : MvQPF F
    ⊢ Iff MvQPF.LiftPPreservation MvQPF.IsUniform
  -/
  rw [← suppPreservation_iff_liftpPreservation, suppPreservation_iff_isUniform]
  /-
    🎉 no goals
  -/


/-- Any type function `F` that is (extensionally) equivalent to a QPF, is itself a QPF,
assuming that the functorial map of `F` behaves similar to `MvFunctor.ofEquiv eqv` -/
def ofEquiv {F F' : TypeVec.{u} n → Type*} [q : MvQPF F'] [MvFunctor F]
    (eqv : ∀ α, F α ≃ F' α)
    (map_eq : ∀ (α β : TypeVec n) (f : α ⟹ β) (a : F α),
      f <$$> a = ((eqv _).symm <| f <$$> eqv _ a) := by intros; rfl) :
    MvQPF F where
  P         := q.P
  abs α     := (eqv _).symm <| q.abs α
  repr α    := q.repr <| eqv _ α
                  /-
                    n : Nat
                    F✝ : TypeVec.{u} n → Type u_1
                    q✝ : MvQPF F✝
                    F : TypeVec.{u} n → Type u_2
                    F' : TypeVec.{u} n → Type u_3
                    q : MvQPF F'
                    inst✝ : MvFunctor F
                    eqv : (α : TypeVec.{u} n) → Equiv (F α) (F' α)
                    map_eq : autoParam (∀ (α β : TypeVec.{u} n) (f : α.Arrow β) (a : F α), Eq (MvF …
                    ⊢ ∀ {α : TypeVec.{u} n} (x : F α), Eq ((fun {α} α_1 => (eqv α).symm (MvQPF.abs …
                  -/
  abs_repr  := by simp [q.abs_repr]
                  /-
                    🎉 no goals
                  -/
                  /-
                    n : Nat
                    F✝ : TypeVec.{u} n → Type u_1
                    q✝ : MvQPF F✝
                    F : TypeVec.{u} n → Type u_2
                    F' : TypeVec.{u} n → Type u_3
                    q : MvQPF F'
                    inst✝ : MvFunctor F
                    eqv : (α : TypeVec.{u} n) → Equiv (F α) (F' α)
                    map_eq : autoParam (∀ (α β : TypeVec.{u} n) (f : α.Arrow β) (a : F α), Eq (MvF …
                    ⊢ ∀ {α β : TypeVec.{u} n} (f : α.Arrow β) (p : ↑(MvQPF.P F') α), Eq ((fun {α}  …
                  -/
  abs_map   := by simp [q.abs_map, map_eq]
                  /-
                    🎉 no goals
                  -/


/-- Every polynomial functor is a (trivial) QPF -/
instance MvPFunctor.instMvQPFObj {n} (P : MvPFunctor n) : MvQPF P where
  P := P
  abs := id
  repr := id
                 /-
                   n : Nat
                   P : MvPFunctor.{?u.10299} n
                   ⊢ ∀ {α : TypeVec.{?u.10299} n} (x : ↑P α), Eq ((fun {α} => id) ((fun {α} => id …
                 -/
  abs_repr := by intros; rfl
                         /-
                           🎉 no goals
                         -/
                /-
                  n : Nat
                  P : MvPFunctor.{?u.10299} n
                  ⊢ ∀ {α β : TypeVec.{?u.10299} n} (f : α.Arrow β) (p : ↑P α), Eq ((fun {α} => i …
                -/
  abs_map := by intros; rfl
                        /-
                          🎉 no goals
                        -/

