/-- multivariate polynomial functors
-/
@[pp_with_univ]
structure MvPFunctor (n : ℕ) where
  /-- The head type -/
  A : Type u
  /-- The child family of types -/
  B : A → TypeVec.{u} n


/-- Applying `P` to an object of `Type` -/
@[coe]
def Obj (α : TypeVec.{u} n) : Type u :=
  Σ a : P.A, P.B a ⟹ α


instance : CoeFun (MvPFunctor.{u} n) (fun _ => TypeVec.{u} n → Type u) where
  coe := Obj


/-- Applying `P` to a morphism of `Type` -/
def map {α β : TypeVec n} (f : α ⟹ β) : P α → P β := fun ⟨a, g⟩ => ⟨a, TypeVec.comp f g⟩


instance : Inhabited (MvPFunctor n) :=
  ⟨⟨default, default⟩⟩


instance Obj.inhabited {α : TypeVec n} [Inhabited P.A] [∀ i, Inhabited (α i)] :
    Inhabited (P α) :=
  ⟨⟨default, fun _ _ => default⟩⟩


instance : MvFunctor.{u} P.Obj :=
  ⟨@MvPFunctor.map n P⟩


theorem map_eq {α β : TypeVec n} (g : α ⟹ β) (a : P.A) (f : P.B a ⟹ α) :
    @MvFunctor.map _ P.Obj _ _ _ g ⟨a, f⟩ = ⟨a, g ⊚ f⟩ :=
  rfl


theorem id_map {α : TypeVec n} : ∀ x : P α, TypeVec.id <$$> x = x
  | ⟨_, _⟩ => rfl


theorem comp_map {α β γ : TypeVec n} (f : α ⟹ β) (g : β ⟹ γ) :
    ∀ x : P α, (g ⊚ f) <$$> x = g <$$> f <$$> x
  | ⟨_, _⟩ => rfl


instance : LawfulMvFunctor.{u} P.Obj where
  id_map := @id_map _ P
  comp_map := @comp_map _ P


/-- Constant functor where the input object does not affect the output -/
def const (n : ℕ) (A : Type u) : MvPFunctor n :=
  { A
    B := fun _ _ => PEmpty }


/-- Constructor for the constant functor -/
def const.mk (x : A) {α} : const n A α :=
  ⟨x, fun _ a => PEmpty.elim a⟩


/-- Destructor for the constant functor -/
def const.get (x : const n A α) : A :=
  x.1


@[simp]
theorem const.get_map (f : α ⟹ β) (x : const n A α) : const.get (f <$$> x) = const.get x := by
  /-
    n : Nat
    A : Type u
    α β : TypeVec.{u} n
    f : α.Arrow β
    x : ↑(MvPFunctor.const n A) α
    ⊢ Eq (MvPFunctor.const.get (MvFunctor.map f x)) (MvPFunctor.const.get x)
  -/
  cases x
  /-
    case mk
    n : Nat
    A : Type u
    α β : TypeVec.{u} n
    f : α.Arrow β
    fst✝ : (MvPFunctor.const n A).A
    snd✝ : ((MvPFunctor.const n A).B fst✝).Arrow α
    ⊢ Eq (MvPFunctor.const.get (MvFunctor.map f ⟨fst✝, snd✝⟩)) (MvPFunctor.const.g …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem const.get_mk (x : A) : const.get (const.mk n x : const n A α) = x := rfl


@[simp]
theorem const.mk_get (x : const n A α) : const.mk n (const.get x) = x := by
  /-
    n : Nat
    A : Type u
    α : TypeVec.{u} n
    x : ↑(MvPFunctor.const n A) α
    ⊢ Eq (MvPFunctor.const.mk n (MvPFunctor.const.get x)) x
  -/
  cases x
  /-
    case mk
    n : Nat
    A : Type u
    α : TypeVec.{u} n
    fst✝ : (MvPFunctor.const n A).A
    snd✝ : ((MvPFunctor.const n A).B fst✝).Arrow α
    ⊢ Eq (MvPFunctor.const.mk n (MvPFunctor.const.get ⟨fst✝, snd✝⟩)) ⟨fst✝, snd✝⟩
  -/
  dsimp [const.get, const.mk]
  /-
    case mk
    n : Nat
    A : Type u
    α : TypeVec.{u} n
    fst✝ : (MvPFunctor.const n A).A
    snd✝ : ((MvPFunctor.const n A).B fst✝).Arrow α
    ⊢ Eq ⟨fst✝, fun x a => PEmpty.elim a⟩ ⟨fst✝, snd✝⟩
  -/
  congr with (_⟨⟩)
  /-
    🎉 no goals
  -/


/-- Functor composition on polynomial functors -/
def comp (P : MvPFunctor.{u} n) (Q : Fin2 n → MvPFunctor.{u} m) : MvPFunctor m where
  A := Σ a₂ : P.1, ∀ i, P.2 a₂ i → (Q i).1
  B a i := Σ(j : _) (b : P.2 a.1 j), (Q j).2 (a.snd j b) i


/-- Constructor for functor composition -/
def comp.mk (x : P (fun i => Q i α)) : comp P Q α :=
  ⟨⟨x.1, fun _ a => (x.2 _ a).1⟩, fun i a => (x.snd a.fst a.snd.fst).snd i a.snd.snd⟩


/-- Destructor for functor composition -/
def comp.get (x : comp P Q α) : P (fun i => Q i α) :=
  ⟨x.1.1, fun i a => ⟨x.fst.snd i a, fun (j : Fin2 m) (b : (Q i).B _ j) => x.snd j ⟨i, ⟨a, b⟩⟩⟩⟩


theorem comp.get_map (f : α ⟹ β) (x : comp P Q α) :
    comp.get (f <$$> x) = (fun i (x : Q i α) => f <$$> x) <$$> comp.get x := by
  /-
    n m : Nat
    P : MvPFunctor.{u} n
    Q : Fin2 n → MvPFunctor.{u} m
    α β : TypeVec.{u} m
    f : α.Arrow β
    x : ↑(P.comp Q) α
    ⊢ Eq (MvPFunctor.comp.get (MvFunctor.map f x)) (MvFunctor.map (fun i x => MvFu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp.get_mk (x : P (fun i => Q i α)) : comp.get (comp.mk x) = x := by
  /-
    n m : Nat
    P : MvPFunctor.{u} n
    Q : Fin2 n → MvPFunctor.{u} m
    α : TypeVec.{u} m
    x : ↑P fun i => ↑(Q i) α
    ⊢ Eq (MvPFunctor.comp.get (MvPFunctor.comp.mk x)) x
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp.mk_get (x : comp P Q α) : comp.mk (comp.get x) = x := by
  /-
    n m : Nat
    P : MvPFunctor.{u} n
    Q : Fin2 n → MvPFunctor.{u} m
    α : TypeVec.{u} m
    x : ↑(P.comp Q) α
    ⊢ Eq (MvPFunctor.comp.mk (MvPFunctor.comp.get x)) x
  -/
  rfl
  /-
    🎉 no goals
  -/

/-
lifting predicates and relations
-/

theorem liftP_iff {α : TypeVec n} (p : ∀ ⦃i⦄, α i → Prop) (x : P α) :
    LiftP p x ↔ ∃ a f, x = ⟨a, f⟩ ∧ ∀ i j, p (f i j) := by
  /-
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : ↑P α
    ⊢ Iff (MvFunctor.LiftP p x) (Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) …
  -/
  constructor
    /-
      case mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : ↑P α
      ⊢ MvFunctor.LiftP p x → Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ ( …
    -/
  · rintro ⟨y, hy⟩
    /-
      case mp.intro
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : ↑P α
      y : ↑P fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      ⊢ Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i : Fin2 n) (j : P.B a …
    -/
    cases' h : y with a f
    /-
      case mp.intro.mk
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : ↑P α
      y : ↑P fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : P.A
      f : (P.B a).Arrow fun i => Subtype p
      h : Eq y ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i : Fin2 n) (j : P.B a …
    -/
    refine ⟨a, fun i j => (f i j).val, ?_, fun i j => (f i j).property⟩
    /-
      case mp.intro.mk
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : ↑P α
      y : ↑P fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : P.A
      f : (P.B a).Arrow fun i => Subtype p
      h : Eq y ⟨a, f⟩
      ⊢ Eq x ⟨a, fun i j => ↑(f i j)⟩
    -/
    rw [← hy, h, map_eq]
    /-
      case mp.intro.mk
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      x : ↑P α
      y : ↑P fun i => Subtype p
      hy : Eq (MvFunctor.map (fun i => Subtype.val) y) x
      a : P.A
      f : (P.B a).Arrow fun i => Subtype p
      h : Eq y ⟨a, f⟩
      ⊢ Eq ⟨a, TypeVec.comp (fun i => Subtype.val) f⟩ ⟨a, fun i j => ↑(f i j)⟩
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : ↑P α
    ⊢ (Exists fun a => Exists fun f => And (Eq x ⟨a, f⟩) (∀ (i : Fin2 n) (j : P.B  …
  -/
  rintro ⟨a, f, xeq, pf⟩
  /-
    case mpr.intro.intro.intro
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : ↑P α
    a : P.A
    f : (P.B a).Arrow α
    xeq : Eq x ⟨a, f⟩
    pf : ∀ (i : Fin2 n) (j : P.B a i), p (f i j)
    ⊢ MvFunctor.LiftP p x
  -/
  use ⟨a, fun i j => ⟨f i j, pf i j⟩⟩
  /-
    case h
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    x : ↑P α
    a : P.A
    f : (P.B a).Arrow α
    xeq : Eq x ⟨a, f⟩
    pf : ∀ (i : Fin2 n) (j : P.B a i), p (f i j)
    ⊢ Eq (MvFunctor.map (fun i => Subtype.val) ⟨a, fun i j => ⟨f i j, ⋯⟩⟩) x
  -/
  rw [xeq]; rfl
            /-
              🎉 no goals
            -/


theorem liftP_iff' {α : TypeVec n} (p : ∀ ⦃i⦄, α i → Prop) (a : P.A) (f : P.B a ⟹ α) :
    @LiftP.{u} _ P.Obj _ α p ⟨a, f⟩ ↔ ∀ i x, p (f i x) := by
  /-
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    p : ⦃i : Fin2 n⦄ → α i → Prop
    a : P.A
    f : (P.B a).Arrow α
    ⊢ Iff (MvFunctor.LiftP p ⟨a, f⟩) (∀ (i : Fin2 n) (x : P.B a i), p (f i x))
  -/
  simp only [liftP_iff, Sigma.mk.inj_iff]; constructor
    /-
      case mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : P.A
      f : (P.B a).Arrow α
      ⊢ (Exists fun a_1 => Exists fun f_1 => And (Eq ⟨a, f⟩ ⟨a_1, f_1⟩) (∀ (i : Fin2 …
    -/
  · rintro ⟨_, _, ⟨⟩, _⟩
    /-
      case mp.intro.intro.intro.refl
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : P.A
      f : (P.B a).Arrow α
      right✝ : ∀ (i : Fin2 n) (j : P.B a i), p (f i j)
      ⊢ ∀ (i : Fin2 n) (x : P.B a i), p (f i x)
    -/
    assumption
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : P.A
      f : (P.B a).Arrow α
      ⊢ (∀ (i : Fin2 n) (x : P.B a i), p (f i x)) → Exists fun a_2 => Exists fun f_1 …
    -/
  · intro
    /-
      case mpr
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      p : ⦃i : Fin2 n⦄ → α i → Prop
      a : P.A
      f : (P.B a).Arrow α
      a✝ : ∀ (i : Fin2 n) (x : P.B a i), p (f i x)
      ⊢ Exists fun a_1 => Exists fun f_1 => And (Eq ⟨a, f⟩ ⟨a_1, f_1⟩) (∀ (i : Fin2  …
    -/
    repeat' first |constructor|assumption
    /-
      🎉 no goals
    -/


theorem liftR_iff {α : TypeVec n} (r : ∀ ⦃i⦄, α i → α i → Prop) (x y : P α) :
    LiftR @r x y ↔ ∃ a f₀ f₁, x = ⟨a, f₀⟩ ∧ y = ⟨a, f₁⟩ ∧ ∀ i j, r (f₀ i j) (f₁ i j) := by
  /-
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    r : ⦃i : Fin2 n⦄ → α i → α i → Prop
    x y : ↑P α
    ⊢ Iff (MvFunctor.LiftR r x y) (Exists fun a => Exists fun f₀ => Exists fun f₁  …
  -/
  constructor
    /-
      case mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      ⊢ MvFunctor.LiftR r x y → Exists fun a => Exists fun f₀ => Exists fun f₁ => An …
    -/
  · rintro ⟨u, xeq, yeq⟩
    /-
      case mp.intro.intro
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      u : ↑P fun i => Subtype fun p => r p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x ⟨a, f₀⟩) (And (E …
    -/
    cases' h : u with a f
    /-
      case mp.intro.intro.mk
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      u : ↑P fun i => Subtype fun p => r p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : P.A
      f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x ⟨a, f₀⟩) (And (E …
    -/
    use a, fun i j => (f i j).val.fst, fun i j => (f i j).val.snd
    /-
      case h
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      u : ↑P fun i => Subtype fun p => r p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : P.A
      f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ And (Eq x ⟨a, fun i j => (↑(f i j)).1⟩) (And (Eq y ⟨a, fun i j => (↑(f i j)) …
    -/
    constructor
      /-
        case h.left
        n : Nat
        P : MvPFunctor.{u} n
        α : TypeVec.{u} n
        r : ⦃i : Fin2 n⦄ → α i → α i → Prop
        x y : ↑P α
        u : ↑P fun i => Subtype fun p => r p.1 p.2
        xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
        yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
        a : P.A
        f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq x ⟨a, fun i j => (↑(f i j)).1⟩
      -/
    · rw [← xeq, h]
      /-
        case h.left
        n : Nat
        P : MvPFunctor.{u} n
        α : TypeVec.{u} n
        r : ⦃i : Fin2 n⦄ → α i → α i → Prop
        x y : ↑P α
        u : ↑P fun i => Subtype fun p => r p.1 p.2
        xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
        yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
        a : P.A
        f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq (MvFunctor.map (fun i t => (↑t).1) ⟨a, f⟩) ⟨a, fun i j => (↑(f i j)).1⟩
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      u : ↑P fun i => Subtype fun p => r p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : P.A
      f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ And (Eq y ⟨a, fun i j => (↑(f i j)).2⟩) (∀ (i : Fin2 n) (j : P.B a i), r (↑( …
    -/
    constructor
      /-
        case h.right.left
        n : Nat
        P : MvPFunctor.{u} n
        α : TypeVec.{u} n
        r : ⦃i : Fin2 n⦄ → α i → α i → Prop
        x y : ↑P α
        u : ↑P fun i => Subtype fun p => r p.1 p.2
        xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
        yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
        a : P.A
        f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq y ⟨a, fun i j => (↑(f i j)).2⟩
      -/
    · rw [← yeq, h]
      /-
        case h.right.left
        n : Nat
        P : MvPFunctor.{u} n
        α : TypeVec.{u} n
        r : ⦃i : Fin2 n⦄ → α i → α i → Prop
        x y : ↑P α
        u : ↑P fun i => Subtype fun p => r p.1 p.2
        xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
        yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
        a : P.A
        f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
        h : Eq u ⟨a, f⟩
        ⊢ Eq (MvFunctor.map (fun i t => (↑t).2) ⟨a, f⟩) ⟨a, fun i j => (↑(f i j)).2⟩
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.right.right
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      u : ↑P fun i => Subtype fun p => r p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : P.A
      f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      ⊢ ∀ (i : Fin2 n) (j : P.B a i), r (↑(f i j)).1 (↑(f i j)).2
    -/
    intro i j
    /-
      case h.right.right
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      u : ↑P fun i => Subtype fun p => r p.1 p.2
      xeq : Eq (MvFunctor.map (fun i t => (↑t).1) u) x
      yeq : Eq (MvFunctor.map (fun i t => (↑t).2) u) y
      a : P.A
      f : (P.B a).Arrow fun i => Subtype fun p => r p.1 p.2
      h : Eq u ⟨a, f⟩
      i : Fin2 n
      j : P.B a i
      ⊢ r (↑(f i j)).1 (↑(f i j)).2
    -/
    exact (f i j).property
    /-
      🎉 no goals
    -/
  /-
    case mpr
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    r : ⦃i : Fin2 n⦄ → α i → α i → Prop
    x y : ↑P α
    ⊢ (Exists fun a => Exists fun f₀ => Exists fun f₁ => And (Eq x ⟨a, f₀⟩) (And ( …
  -/
  rintro ⟨a, f₀, f₁, xeq, yeq, h⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    r : ⦃i : Fin2 n⦄ → α i → α i → Prop
    x y : ↑P α
    a : P.A
    f₀ f₁ : (P.B a).Arrow α
    xeq : Eq x ⟨a, f₀⟩
    yeq : Eq y ⟨a, f₁⟩
    h : ∀ (i : Fin2 n) (j : P.B a i), r (f₀ i j) (f₁ i j)
    ⊢ MvFunctor.LiftR r x y
  -/
  use ⟨a, fun i j => ⟨(f₀ i j, f₁ i j), h i j⟩⟩
  /-
    case h
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    r : ⦃i : Fin2 n⦄ → α i → α i → Prop
    x y : ↑P α
    a : P.A
    f₀ f₁ : (P.B a).Arrow α
    xeq : Eq x ⟨a, f₀⟩
    yeq : Eq y ⟨a, f₁⟩
    h : ∀ (i : Fin2 n) (j : P.B a i), r (f₀ i j) (f₁ i j)
    ⊢ And (Eq (MvFunctor.map (fun i t => (↑t).1) ⟨a, fun i j => ⟨{ fst := f₀ i j,  …
  -/
  dsimp; constructor
    /-
      case h.left
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      a : P.A
      f₀ f₁ : (P.B a).Arrow α
      xeq : Eq x ⟨a, f₀⟩
      yeq : Eq y ⟨a, f₁⟩
      h : ∀ (i : Fin2 n) (j : P.B a i), r (f₀ i j) (f₁ i j)
      ⊢ Eq (MvFunctor.map (fun i t => (↑t).1) ⟨a, fun i j => ⟨{ fst := f₀ i j, snd : …
    -/
  · rw [xeq]
    /-
      case h.left
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      r : ⦃i : Fin2 n⦄ → α i → α i → Prop
      x y : ↑P α
      a : P.A
      f₀ f₁ : (P.B a).Arrow α
      xeq : Eq x ⟨a, f₀⟩
      yeq : Eq y ⟨a, f₁⟩
      h : ∀ (i : Fin2 n) (j : P.B a i), r (f₀ i j) (f₁ i j)
      ⊢ Eq (MvFunctor.map (fun i t => (↑t).1) ⟨a, fun i j => ⟨{ fst := f₀ i j, snd : …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case h.right
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    r : ⦃i : Fin2 n⦄ → α i → α i → Prop
    x y : ↑P α
    a : P.A
    f₀ f₁ : (P.B a).Arrow α
    xeq : Eq x ⟨a, f₀⟩
    yeq : Eq y ⟨a, f₁⟩
    h : ∀ (i : Fin2 n) (j : P.B a i), r (f₀ i j) (f₁ i j)
    ⊢ Eq (MvFunctor.map (fun i t => (↑t).2) ⟨a, fun i j => ⟨{ fst := f₀ i j, snd : …
  -/
  rw [yeq]; rfl
            /-
              🎉 no goals
            -/


theorem supp_eq {α : TypeVec n} (a : P.A) (f : P.B a ⟹ α) (i) :
    @supp.{u} _ P.Obj _ α (⟨a, f⟩ : P α) i = f i '' univ := by
  /-
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    a : P.A
    f : (P.B a).Arrow α
    i : Fin2 n
    ⊢ Eq (MvFunctor.supp ⟨a, f⟩ i) (Set.image (f i) Set.univ)
  -/
  ext x; simp only [supp, image_univ, mem_range, mem_setOf_eq]
  /-
    case h
    n : Nat
    P : MvPFunctor.{u} n
    α : TypeVec.{u} n
    a : P.A
    f : (P.B a).Arrow α
    i : Fin2 n
    x : α i
    ⊢ Iff (∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P_1 ⟨a, f⟩ → P_1 i …
  -/
  constructor <;> intro h
    /-
      case h.mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      h : ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P_1 ⟨a, f⟩ → P_1 i x
      ⊢ Exists fun y => Eq (f i y) x
    -/
  · apply @h fun i x => ∃ y : P.B a i, f i y = x
    /-
      case h.mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      h : ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P_1 ⟨a, f⟩ → P_1 i x
      ⊢ MvFunctor.LiftP (fun i x => Exists fun y => Eq (f i y) x) ⟨a, f⟩
    -/
    rw [liftP_iff']
    /-
      case h.mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      h : ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P_1 ⟨a, f⟩ → P_1 i x
      ⊢ ∀ (i : Fin2 n) (x : P.B a i), Exists fun y => Eq (f i y) (f i x)
    -/
    intros
    /-
      case h.mp
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      h : ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P_1 ⟨a, f⟩ → P_1 i x
      i✝ : Fin2 n
      x✝ : P.B a i✝
      ⊢ Exists fun y => Eq (f i✝ y) (f i✝ x✝)
    -/
    exact ⟨_, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      h : Exists fun y => Eq (f i y) x
      ⊢ ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, MvFunctor.LiftP P_1 ⟨a, f⟩ → P_1 i x
    -/
  · simp only [liftP_iff']
    /-
      case h.mpr
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      h : Exists fun y => Eq (f i y) x
      ⊢ ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, (∀ (i : Fin2 n) (x : P.B a i), P_1 i (f …
    -/
    cases h
    /-
      case h.mpr.intro
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      x : α i
      w✝ : P.B a i
      h✝ : Eq (f i w✝) x
      ⊢ ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, (∀ (i : Fin2 n) (x : P.B a i), P_1 i (f …
    -/
    subst x
    /-
      case h.mpr.intro
      n : Nat
      P : MvPFunctor.{u} n
      α : TypeVec.{u} n
      a : P.A
      f : (P.B a).Arrow α
      i : Fin2 n
      w✝ : P.B a i
      ⊢ ∀ ⦃P_1 : (i : Fin2 n) → α i → Prop⦄, (∀ (i : Fin2 n) (x : P.B a i), P_1 i (f …
    -/
    tauto
    /-
      🎉 no goals
    -/


/-- Split polynomial functor, get an n-ary functor
from an `n+1`-ary functor -/
def drop : MvPFunctor n where
  A := P.A
  B a := (P.B a).drop


/-- Split polynomial functor, get a univariate functor
from an `n+1`-ary functor -/
def last : PFunctor where
  A := P.A
  B a := (P.B a).last


/-- append arrows of a polynomial functor application -/
abbrev appendContents {α : TypeVec n} {β : Type*} {a : P.A} (f' : P.drop.B a ⟹ α)
    (f : P.last.B a → β) : P.B a ⟹ (α ::: β) :=
  splitFun f' f


