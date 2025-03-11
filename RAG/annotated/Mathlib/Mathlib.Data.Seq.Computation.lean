/-- `Computation α` is the type of unbounded computations returning `α`.
  An element of `Computation α` is an infinite sequence of `Option α` such
  that if `f n = some a` for some `n` then it is constantly `some a` after that. -/
def Computation (α : Type u) : Type u :=
  { f : Stream' (Option α) // ∀ ⦃n a⦄, f n = some a → f (n + 1) = some a }


/-- `pure a` is the computation that immediately terminates with result `a`. -/
-- Porting note: `return` is reserved, so changed to `pure`
def pure (a : α) : Computation α :=
  ⟨Stream'.const (some a), fun _ _ => id⟩


instance : CoeTC α (Computation α) :=
  ⟨pure⟩

-- note [use has_coe_t]

/-- `think c` is the computation that delays for one "tick" and then performs
  computation `c`. -/
def think (c : Computation α) : Computation α :=
  ⟨Stream'.cons none c.1, fun n a h => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      c : Computation α
      n : Nat
      a : α
      h : Eq (Stream'.cons Option.none (↑c) n) (Option.some a)
      ⊢ Eq (Stream'.cons Option.none (↑c) (HAdd.hAdd n 1)) (Option.some a)
    -/
    cases' n with n
      /-
        case zero
        α : Type u
        β : Type v
        γ : Type w
        c : Computation α
        a : α
        h : Eq (Stream'.cons Option.none (↑c) 0) (Option.some a)
        ⊢ Eq (Stream'.cons Option.none (↑c) (HAdd.hAdd 0 1)) (Option.some a)
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case succ
        α : Type u
        β : Type v
        γ : Type w
        c : Computation α
        a : α
        n : Nat
        h : Eq (Stream'.cons Option.none (↑c) (HAdd.hAdd n 1)) (Option.some a)
        ⊢ Eq (Stream'.cons Option.none (↑c) (HAdd.hAdd (HAdd.hAdd n 1) 1)) (Option.som …
      -/
    · exact c.2 h⟩
      /-
        🎉 no goals
      -/


/-- `thinkN c n` is the computation that delays for `n` ticks and then performs
  computation `c`. -/
def thinkN (c : Computation α) : ℕ → Computation α
  | 0 => c
  | n + 1 => think (thinkN c n)

-- check for immediate result

/-- `head c` is the first step of computation, either `some a` if `c = pure a`
  or `none` if `c = think c'`. -/
def head (c : Computation α) : Option α :=
  c.1.head

-- one step of computation

/-- `tail c` is the remainder of computation, either `c` if `c = pure a`
  or `c'` if `c = think c'`. -/
def tail (c : Computation α) : Computation α :=
  ⟨c.1.tail, fun _ _ h => c.2 h⟩


/-- `empty α` is the computation that never returns, an infinite sequence of
  `think`s. -/
def empty (α) : Computation α :=
  ⟨Stream'.const none, fun _ _ => id⟩


instance : Inhabited (Computation α) :=
  ⟨empty _⟩


/-- `runFor c n` evaluates `c` for `n` steps and returns the result, or `none`
  if it did not terminate after `n` steps. -/
def runFor : Computation α → ℕ → Option α :=
  Subtype.val


/-- `destruct c` is the destructor for `Computation α` as a coinductive type.
  It returns `inl a` if `c = pure a` and `inr c'` if `c = think c'`. -/
def destruct (c : Computation α) : α ⊕ (Computation α) :=
  match c.1 0 with
  | none => Sum.inr (tail c)
  | some a => Sum.inl a


/-- `run c` is an unsound meta function that runs `c` to completion, possibly
  resulting in an infinite loop in the VM. -/
unsafe def run : Computation α → α
  | c =>
    match destruct c with
    | Sum.inl a => a
    | Sum.inr ca => run ca


theorem destruct_eq_pure {s : Computation α} {a : α} : destruct s = Sum.inl a → s = pure a := by
  /-
    α : Type u
    s : Computation α
    a : α
    ⊢ Eq s.destruct (Sum.inl a) → Eq s (Computation.pure a)
  -/
  dsimp [destruct]
  /-
    α : Type u
    s : Computation α
    a : α
    ⊢ Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (↑s 0) (fu …
  -/
  induction' f0 : s.1 0 with _ <;> intro h
    /-
      case none
      α : Type u
      s : Computation α
      a : α
      f0 : Eq (↑s 0) Option.none
      h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) Option.n …
      ⊢ Eq s (Computation.pure a)
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u
      s : Computation α
      a val✝ : α
      f0 : Eq (↑s 0) (Option.some val✝)
      h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Option. …
      ⊢ Eq s (Computation.pure a)
    -/
  · apply Subtype.eq
    /-
      case some.a
      α : Type u
      s : Computation α
      a val✝ : α
      f0 : Eq (↑s 0) (Option.some val✝)
      h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Option. …
      ⊢ Eq ↑s ↑(Computation.pure a)
    -/
    funext n
    /-
      case some.a.h
      α : Type u
      s : Computation α
      a val✝ : α
      f0 : Eq (↑s 0) (Option.some val✝)
      h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Option. …
      n : Nat
      ⊢ Eq (↑s n) (↑(Computation.pure a) n)
    -/
    induction' n with n IH
      /-
        case some.a.h.zero
        α : Type u
        s : Computation α
        a val✝ : α
        f0 : Eq (↑s 0) (Option.some val✝)
        h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Option. …
        ⊢ Eq (↑s 0) (↑(Computation.pure a) 0)
      -/
    · injection h with h'
      /-
        case some.a.h.zero
        α : Type u
        s : Computation α
        a val✝ : α
        f0 : Eq (↑s 0) (Option.some val✝)
        h' : Eq val✝ a
        ⊢ Eq (↑s 0) (↑(Computation.pure a) 0)
      -/
      rwa [h'] at f0
      /-
        🎉 no goals
      -/
      /-
        case some.a.h.succ
        α : Type u
        s : Computation α
        a val✝ : α
        f0 : Eq (↑s 0) (Option.some val✝)
        h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Option. …
        n : Nat
        IH : Eq (↑s n) (↑(Computation.pure a) n)
        ⊢ Eq (↑s (HAdd.hAdd n 1)) (↑(Computation.pure a) (HAdd.hAdd n 1))
      -/
    · exact s.2 IH
      /-
        🎉 no goals
      -/


theorem destruct_eq_think {s : Computation α} {s'} : destruct s = Sum.inr s' → s = think s' := by
  /-
    α : Type u
    s s' : Computation α
    ⊢ Eq s.destruct (Sum.inr s') → Eq s s'.think
  -/
  dsimp [destruct]
  /-
    α : Type u
    s s' : Computation α
    ⊢ Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (↑s 0) (fu …
  -/
  induction' f0 : s.1 0 with a' <;> intro h
    /-
      case none
      α : Type u
      s s' : Computation α
      f0 : Eq (↑s 0) Option.none
      h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) Option.n …
      ⊢ Eq s s'.think
    -/
  · injection h with h'
    /-
      case none
      α : Type u
      s s' : Computation α
      f0 : Eq (↑s 0) Option.none
      h' : Eq s.tail s'
      ⊢ Eq s s'.think
    -/
    rw [← h']
    /-
      case none
      α : Type u
      s s' : Computation α
      f0 : Eq (↑s 0) Option.none
      h' : Eq s.tail s'
      ⊢ Eq s s.tail.think
    -/
    cases' s with f al
    /-
      case none.mk
      α : Type u
      s' : Computation α
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      f0 : Eq (↑⟨f, al⟩ 0) Option.none
      h' : Eq (Computation.tail ⟨f, al⟩) s'
      ⊢ Eq ⟨f, al⟩ (Computation.tail ⟨f, al⟩).think
    -/
    apply Subtype.eq
    /-
      case none.mk.a
      α : Type u
      s' : Computation α
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      f0 : Eq (↑⟨f, al⟩ 0) Option.none
      h' : Eq (Computation.tail ⟨f, al⟩) s'
      ⊢ Eq ↑⟨f, al⟩ ↑(Computation.tail ⟨f, al⟩).think
    -/
    dsimp [think, tail]
    /-
      case none.mk.a
      α : Type u
      s' : Computation α
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      f0 : Eq (↑⟨f, al⟩ 0) Option.none
      h' : Eq (Computation.tail ⟨f, al⟩) s'
      ⊢ Eq f (Stream'.cons Option.none f.tail)
    -/
    rw [← f0]
    /-
      case none.mk.a
      α : Type u
      s' : Computation α
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      f0 : Eq (↑⟨f, al⟩ 0) Option.none
      h' : Eq (Computation.tail ⟨f, al⟩) s'
      ⊢ Eq f (Stream'.cons (↑⟨f, al⟩ 0) f.tail)
    -/
    exact (Stream'.eta f).symm
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u
      s s' : Computation α
      a' : α
      f0 : Eq (↑s 0) (Option.some a')
      h : Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Option. …
      ⊢ Eq s s'.think
    -/
  · contradiction
    /-
      🎉 no goals
    -/


@[simp]
theorem destruct_pure (a : α) : destruct (pure a) = Sum.inl a :=
  rfl


@[simp]
theorem destruct_think : ∀ s : Computation α, destruct (think s) = Sum.inr s
  | ⟨_, _⟩ => rfl


@[simp]
theorem destruct_empty : destruct (empty α) = Sum.inr (empty α) :=
  rfl


@[simp]
theorem head_pure (a : α) : head (pure a) = some a :=
  rfl


@[simp]
theorem head_think (s : Computation α) : head (think s) = none :=
  rfl


@[simp]
theorem head_empty : head (empty α) = none :=
  rfl


@[simp]
theorem tail_pure (a : α) : tail (pure a) = pure a :=
  rfl


@[simp]
theorem tail_think (s : Computation α) : tail (think s) = s := by
  /-
    α : Type u
    s : Computation α
    ⊢ Eq s.think.tail s
  -/
  cases' s with f al; apply Subtype.eq; dsimp [tail, think]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem tail_empty : tail (empty α) = empty α :=
  rfl


theorem think_empty : empty α = think (empty α) :=
  destruct_eq_think destruct_empty


/-- Recursion principle for computations, compare with `List.recOn`. -/
def recOn {C : Computation α → Sort v} (s : Computation α) (h1 : ∀ a, C (pure a))
    (h2 : ∀ s, C (think s)) : C s :=
  match H : destruct s with
  | Sum.inl v => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      C : Computation α → Sort v
      s : Computation α
      h1 : (a : α) → C (Computation.pure a)
      h2 : (s : Computation α) → C s.think
      v : α
      H : Eq s.destruct (Sum.inl v)
      ⊢ C s
    -/
    rw [destruct_eq_pure H]
    /-
      α : Type u
      β : Type v
      γ : Type w
      C : Computation α → Sort v
      s : Computation α
      h1 : (a : α) → C (Computation.pure a)
      h2 : (s : Computation α) → C s.think
      v : α
      H : Eq s.destruct (Sum.inl v)
      ⊢ C (Computation.pure v)
    -/
    apply h1
    /-
      🎉 no goals
    -/
  | Sum.inr v => match v with
    | ⟨a, s'⟩ => by
      /-
        α : Type u
        β : Type v
        γ : Type w
        C : Computation α → Sort v
        s : Computation α
        h1 : (a : α) → C (Computation.pure a)
        h2 : (s : Computation α) → C s.think
        v : Computation α
        a : Stream' (Option α)
        s' : ∀ ⦃n : Nat⦄ ⦃a_1 : α⦄, Eq (a n) (Option.some a_1) → Eq (a (HAdd.hAdd n 1) …
        H : Eq s.destruct (Sum.inr ⟨a, s'⟩)
        ⊢ C s
      -/
      rw [destruct_eq_think H]
      /-
        α : Type u
        β : Type v
        γ : Type w
        C : Computation α → Sort v
        s : Computation α
        h1 : (a : α) → C (Computation.pure a)
        h2 : (s : Computation α) → C s.think
        v : Computation α
        a : Stream' (Option α)
        s' : ∀ ⦃n : Nat⦄ ⦃a_1 : α⦄, Eq (a n) (Option.some a_1) → Eq (a (HAdd.hAdd n 1) …
        H : Eq s.destruct (Sum.inr ⟨a, s'⟩)
        ⊢ C (Computation.think ⟨a, s'⟩)
      -/
      apply h2
      /-
        🎉 no goals
      -/


/-- Corecursor constructor for `corec`-/
def Corec.f (f : β → α ⊕ β) : α ⊕ β → Option α × (α ⊕ β)
  | Sum.inl a => (some a, Sum.inl a)
  | Sum.inr b =>
    (match f b with
      | Sum.inl a => some a
      | Sum.inr _ => none,
      f b)


/-- `corec f b` is the corecursor for `Computation α` as a coinductive type.
  If `f b = inl a` then `corec f b = pure a`, and if `f b = inl b'` then
  `corec f b = think (corec f b')`. -/
def corec (f : β → α ⊕ β) (b : β) : Computation α := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Sum α β
    b : β
    ⊢ Computation α
  -/
  refine ⟨Stream'.corec' (Corec.f f) (Sum.inr b), fun n a' h => ?_⟩
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Sum α β
    b : β
    n : Nat
    a' : α
    h : Eq (Stream'.corec' (Computation.Corec.f f) (Sum.inr b) n) (Option.some a')
    ⊢ Eq (Stream'.corec' (Computation.Corec.f f) (Sum.inr b) (HAdd.hAdd n 1)) (Opt …
  -/
  rw [Stream'.corec'_eq]
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Sum α β
    b : β
    n : Nat
    a' : α
    h : Eq (Stream'.corec' (Computation.Corec.f f) (Sum.inr b) n) (Option.some a')
    ⊢ Eq (Stream'.cons (Computation.Corec.f f (Sum.inr b)).1 (Stream'.corec' (Comp …
  -/
  change Stream'.corec' (Corec.f f) (Corec.f f (Sum.inr b)).2 n = some a'
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Sum α β
    b : β
    n : Nat
    a' : α
    h : Eq (Stream'.corec' (Computation.Corec.f f) (Sum.inr b) n) (Option.some a')
    ⊢ Eq (Stream'.corec' (Computation.Corec.f f) (Computation.Corec.f f (Sum.inr b …
  -/
  revert h; generalize Sum.inr b = o; revert o
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Sum α β
    b : β
    n : Nat
    a' : α
    ⊢ ∀ (o : Sum α β), Eq (Stream'.corec' (Computation.Corec.f f) o n) (Option.som …
  -/
  induction' n with n IH <;> intro o
    /-
      case zero
      α : Type u
      β : Type v
      γ : Type w
      f : β → Sum α β
      b : β
      a' : α
      o : Sum α β
      ⊢ Eq (Stream'.corec' (Computation.Corec.f f) o 0) (Option.some a') → Eq (Strea …
    -/
  · change (Corec.f f o).1 = some a' → (Corec.f f (Corec.f f o).2).1 = some a'
    /-
      case zero
      α : Type u
      β : Type v
      γ : Type w
      f : β → Sum α β
      b : β
      a' : α
      o : Sum α β
      ⊢ Eq (Computation.Corec.f f o).1 (Option.some a') → Eq (Computation.Corec.f f  …
    -/
    cases' o with _ b <;> intro h
      /-
        case zero.inl
        α : Type u
        β : Type v
        γ : Type w
        f : β → Sum α β
        b : β
        a' val✝ : α
        h : Eq (Computation.Corec.f f (Sum.inl val✝)).1 (Option.some a')
        ⊢ Eq (Computation.Corec.f f (Computation.Corec.f f (Sum.inl val✝)).2).1 (Optio …
      -/
    · exact h
      /-
        🎉 no goals
      -/
    /-
      case zero.inr
      α : Type u
      β : Type v
      γ : Type w
      f : β → Sum α β
      b✝ : β
      a' : α
      b : β
      h : Eq (Computation.Corec.f f (Sum.inr b)).1 (Option.some a')
      ⊢ Eq (Computation.Corec.f f (Computation.Corec.f f (Sum.inr b)).2).1 (Option.s …
    -/
                                   /-
                                     🎉 no goals
                                   -/
    unfold Corec.f at *; split <;> simp_all
                                   /-
                                     🎉 no goals
                                   -/
    /-
      case succ
      α : Type u
      β : Type v
      γ : Type w
      f : β → Sum α β
      b : β
      a' : α
      n : Nat
      IH : ∀ (o : Sum α β), Eq (Stream'.corec' (Computation.Corec.f f) o n) (Option. …
      o : Sum α β
      ⊢ Eq (Stream'.corec' (Computation.Corec.f f) o (HAdd.hAdd n 1)) (Option.some a …
    -/
  · rw [Stream'.corec'_eq (Corec.f f) (Corec.f f o).2, Stream'.corec'_eq (Corec.f f) o]
    /-
      case succ
      α : Type u
      β : Type v
      γ : Type w
      f : β → Sum α β
      b : β
      a' : α
      n : Nat
      IH : ∀ (o : Sum α β), Eq (Stream'.corec' (Computation.Corec.f f) o n) (Option. …
      o : Sum α β
      ⊢ Eq (Stream'.cons (Computation.Corec.f f o).1 (Stream'.corec' (Computation.Co …
    -/
    exact IH (Corec.f f o).2
    /-
      🎉 no goals
    -/


/-- left map of `⊕` -/
def lmap (f : α → β) : α ⊕ γ → β ⊕ γ
  | Sum.inl a => Sum.inl (f a)
  | Sum.inr b => Sum.inr b


/-- right map of `⊕` -/
def rmap (f : β → γ) : α ⊕ β → α ⊕ γ
  | Sum.inl a => Sum.inl a
  | Sum.inr b => Sum.inr (f b)


attribute [simp] lmap rmap

-- Porting note: this was far less painful in mathlib3. There seem to be two issues;
-- firstly, in mathlib3 we have `corec.F._match_1` and it's the obvious map α ⊕ β → option α.
-- In mathlib4 we have `Corec.f.match_1` and it's something completely different.
-- Secondly, the proof that `Stream'.corec' (Corec.f f) (Sum.inr b) 0` is this function
-- evaluated at `f b`, used to be `rfl` and now is `cases, rfl`.

@[simp]
theorem corec_eq (f : β → α ⊕ β) (b : β) : destruct (corec f b) = rmap (corec f) (f b) := by
  /-
    α : Type u
    β : Type v
    f : β → Sum α β
    b : β
    ⊢ Eq (Computation.corec f b).destruct (Computation.rmap (Computation.corec f)  …
  -/
  dsimp [corec, destruct]
  rw [show Stream'.corec' (Corec.f f) (Sum.inr b) 0 =
    Sum.rec Option.some (fun _ ↦ none) (f b) by
    dsimp [Corec.f, Stream'.corec', Stream'.corec, Stream'.map, Stream'.get, Stream'.iterate]
    match (f b) with
    | Sum.inl x => rfl
    | Sum.inr x => rfl
    ]
  /-
    α : Type u
    β : Type v
    f : β → Sum α β
    b : β
    ⊢ Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Sum.rec O …
  -/
  induction' h : f b with a b'; · rfl
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case inr
    α : Type u
    β : Type v
    f : β → Sum α β
    b b' : β
    h : Eq (f b) (Sum.inr b')
    ⊢ Eq (Computation.destruct.match_1 (fun x => Sum α (Computation α)) (Sum.rec O …
  -/
  dsimp [Corec.f, destruct]
  /-
    case inr
    α : Type u
    β : Type v
    f : β → Sum α β
    b b' : β
    h : Eq (f b) (Sum.inr b')
    ⊢ Eq (Sum.inr (Computation.tail ⟨Stream'.corec' (Computation.Corec.f f) (Sum.i …
  -/
  apply congr_arg; apply Subtype.eq
  /-
    case inr.h.a
    α : Type u
    β : Type v
    f : β → Sum α β
    b b' : β
    h : Eq (f b) (Sum.inr b')
    ⊢ Eq ↑(Computation.tail ⟨Stream'.corec' (Computation.Corec.f f) (Sum.inr b), ⋯ …
  -/
  dsimp [corec, tail]
  /-
    case inr.h.a
    α : Type u
    β : Type v
    f : β → Sum α β
    b b' : β
    h : Eq (f b) (Sum.inr b')
    ⊢ Eq (Stream'.corec' (Computation.Corec.f f) (Sum.inr b)).tail (Stream'.corec' …
  -/
  rw [Stream'.corec'_eq, Stream'.tail_cons]
  /-
    case inr.h.a
    α : Type u
    β : Type v
    f : β → Sum α β
    b b' : β
    h : Eq (f b) (Sum.inr b')
    ⊢ Eq (Stream'.corec' (Computation.Corec.f f) (Computation.Corec.f f (Sum.inr b …
  -/
  dsimp [Corec.f]; rw [h]
                   /-
                     🎉 no goals
                   -/


/-- bisimilarity relation -/
local infixl:50 " ~ " => R


/-- Bisimilarity over a sum of `Computation`s -/
def BisimO : α ⊕ (Computation α) → α ⊕ (Computation α) → Prop
  | Sum.inl a, Sum.inl a' => a = a'
  | Sum.inr s, Sum.inr s' => R s s'
  | _, _ => False


attribute [simp] BisimO

/-- Attribute expressing bisimilarity over two `Computation`s -/
def IsBisimulation :=
  ∀ ⦃s₁ s₂⦄, s₁ ~ s₂ → BisimO R (destruct s₁) (destruct s₂)

-- If two computations are bisimilar, then they are equal

theorem eq_of_bisim (bisim : IsBisimulation R) {s₁ s₂} (r : s₁ ~ s₂) : s₁ = s₂ := by
  /-
    α : Type u
    R : Computation α → Computation α → Prop
    bisim : Computation.IsBisimulation R
    s₁ s₂ : Computation α
    r : R s₁ s₂
    ⊢ Eq s₁ s₂
  -/
  apply Subtype.eq
  /-
    case a
    α : Type u
    R : Computation α → Computation α → Prop
    bisim : Computation.IsBisimulation R
    s₁ s₂ : Computation α
    r : R s₁ s₂
    ⊢ Eq ↑s₁ ↑s₂
  -/
  apply Stream'.eq_of_bisim fun x y => ∃ s s' : Computation α, s.1 = x ∧ s'.1 = y ∧ R s s'
    /-
      case a.bisim
      α : Type u
      R : Computation α → Computation α → Prop
      bisim : Computation.IsBisimulation R
      s₁ s₂ : Computation α
      r : R s₁ s₂
      ⊢ Stream'.IsBisimulation fun x y => Exists fun s => Exists fun s' => And (Eq ( …
    -/
  · dsimp [Stream'.IsBisimulation]
    /-
      case a.bisim
      α : Type u
      R : Computation α → Computation α → Prop
      bisim : Computation.IsBisimulation R
      s₁ s₂ : Computation α
      r : R s₁ s₂
      ⊢ ∀ ⦃s₁ s₂ : Stream' (Option α)⦄, (Exists fun s => Exists fun s' => And (Eq (↑ …
    -/
    intro t₁ t₂ e
    match t₁, t₂, e with
    | _, _, ⟨s, s', rfl, rfl, r⟩ =>
      suffices head s = head s' ∧ R (tail s) (tail s') from
        And.imp id (fun r => ⟨tail s, tail s', by cases s; rfl, by cases s'; rfl, r⟩) this
      have h := bisim r; revert r h
      apply recOn s _ _ <;> intro r' <;> apply recOn s' _ _ <;> intro a' r h
      · constructor <;> dsimp at h
        · rw [h]
        · rw [h] at r
          rw [tail_pure, tail_pure,h]
          assumption
      · rw [destruct_pure, destruct_think] at h
        exact False.elim h
      · rw [destruct_pure, destruct_think] at h
        exact False.elim h
      · simp_all
    /-
      case a.a
      α : Type u
      R : Computation α → Computation α → Prop
      bisim : Computation.IsBisimulation R
      s₁ s₂ : Computation α
      r : R s₁ s₂
      ⊢ Exists fun s => Exists fun s' => And (Eq ↑s ↑s₁) (And (Eq ↑s' ↑s₂) (R s s'))
    -/
  · exact ⟨s₁, s₂, rfl, rfl, r⟩
    /-
      🎉 no goals
    -/


/-- Assertion that a `Computation` limits to a given value-/
protected def Mem (s : Computation α) (a : α) :=
  some a ∈ s.1


instance : Membership α (Computation α) :=
  ⟨Computation.Mem⟩


theorem le_stable (s : Computation α) {a m n} (h : m ≤ n) : s.1 m = some a → s.1 n = some a := by
  /-
    α : Type u
    s : Computation α
    a : α
    m n : Nat
    h : LE.le m n
    ⊢ Eq (↑s m) (Option.some a) → Eq (↑s n) (Option.some a)
  -/
  cases' s with f al
  /-
    case mk
    α : Type u
    a : α
    m n : Nat
    h : LE.le m n
    f : Stream' (Option α)
    al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
    ⊢ Eq (↑⟨f, al⟩ m) (Option.some a) → Eq (↑⟨f, al⟩ n) (Option.some a)
  -/
  induction' h with n _ IH
  /-
    case mk.refl
    α : Type u
    a : α
    m n : Nat
    f : Stream' (Option α)
    al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
    ⊢ Eq (↑⟨f, al⟩ m) (Option.some a) → Eq (↑⟨f, al⟩ m) (Option.some a)
  -/
  exacts [id, fun h2 => al (IH h2)]
  /-
    🎉 no goals
  -/


theorem mem_unique {s : Computation α} {a b : α} : a ∈ s → b ∈ s → a = b
  | ⟨m, ha⟩, ⟨n, hb⟩ => by
    injection
      (le_stable s (le_max_left m n) ha.symm).symm.trans (le_stable s (le_max_right m n) hb.symm)


theorem Mem.left_unique : Relator.LeftUnique ((· ∈ ·) : α → Computation α → Prop) := fun _ _ _ =>
  mem_unique


/-- `Terminates s` asserts that the computation `s` eventually terminates with some value. -/
class Terminates (s : Computation α) : Prop where
  /-- assertion that there is some term `a` such that the `Computation` terminates -/
  term : ∃ a, a ∈ s


theorem terminates_iff (s : Computation α) : Terminates s ↔ ∃ a, a ∈ s :=
  ⟨fun h => h.1, Terminates.mk⟩


theorem terminates_of_mem {s : Computation α} {a : α} (h : a ∈ s) : Terminates s :=
  ⟨⟨a, h⟩⟩


theorem terminates_def (s : Computation α) : Terminates s ↔ ∃ n, (s.1 n).isSome :=
  ⟨fun ⟨⟨a, n, h⟩⟩ =>
    ⟨n, by
      /-
        α : Type u
        s : Computation α
        x✝ : s.Terminates
        a : α
        n : Nat
        h : (fun b => Eq (Option.some a) b) ((↑s).get n)
        ⊢ Eq (↑s n).isSome Bool.true
      -/
      dsimp [Stream'.get] at h
      /-
        α : Type u
        s : Computation α
        x✝ : s.Terminates
        a : α
        n : Nat
        h : Eq (Option.some a) (↑s n)
        ⊢ Eq (↑s n).isSome Bool.true
      -/
      rw [← h]
      /-
        α : Type u
        s : Computation α
        x✝ : s.Terminates
        a : α
        n : Nat
        h : Eq (Option.some a) (↑s n)
        ⊢ Eq (Option.some a).isSome Bool.true
      -/
      exact rfl⟩,
      /-
        🎉 no goals
      -/
    fun ⟨n, h⟩ => ⟨⟨Option.get _ h, n, (Option.eq_some_of_isSome h).symm⟩⟩⟩


theorem ret_mem (a : α) : a ∈ pure a :=
  Exists.intro 0 rfl


theorem eq_of_pure_mem {a a' : α} (h : a' ∈ pure a) : a' = a :=
  mem_unique h (ret_mem _)


instance ret_terminates (a : α) : Terminates (pure a) :=
  terminates_of_mem (ret_mem _)


theorem think_mem {s : Computation α} {a} : a ∈ s → a ∈ think s
  | ⟨n, h⟩ => ⟨n + 1, h⟩


instance think_terminates (s : Computation α) : ∀ [Terminates s], Terminates (think s)
  | ⟨⟨a, n, h⟩⟩ => ⟨⟨a, n + 1, h⟩⟩


theorem of_think_mem {s : Computation α} {a} : a ∈ think s → a ∈ s
  | ⟨n, h⟩ => by
    /-
      α : Type u
      s : Computation α
      a : α
      n : Nat
      h : (fun b => Eq (Option.some a) b) ((↑s.think).get n)
      ⊢ Membership.mem s a
    -/
    cases' n with n'
      /-
        case zero
        α : Type u
        s : Computation α
        a : α
        h : Eq (Option.some a) ((↑s.think).get 0)
        ⊢ Membership.mem s a
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case succ
        α : Type u
        s : Computation α
        a : α
        n' : Nat
        h : Eq (Option.some a) ((↑s.think).get (HAdd.hAdd n' 1))
        ⊢ Membership.mem s a
      -/
    · exact ⟨n', h⟩
      /-
        🎉 no goals
      -/


theorem of_think_terminates {s : Computation α} : Terminates (think s) → Terminates s
  | ⟨⟨a, h⟩⟩ => ⟨⟨a, of_think_mem h⟩⟩


                                                                /-
                                                                  α : Type u
                                                                  a : α
                                                                  x✝ : Membership.mem (Computation.empty α) a
                                                                  n : Nat
                                                                  h : (fun b => Eq (Option.some a) b) ((↑(Computation.empty α)).get n)
                                                                  ⊢ False
                                                                -/
theorem not_mem_empty (a : α) : a ∉ empty α := fun ⟨n, h⟩ => by contradiction
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem not_terminates_empty : ¬Terminates (empty α) := fun ⟨⟨a, h⟩⟩ => not_mem_empty a h


theorem eq_empty_of_not_terminates {s} (H : ¬Terminates s) : s = empty α := by
  /-
    α : Type u
    s : Computation α
    H : Not s.Terminates
    ⊢ Eq s (Computation.empty α)
  -/
  apply Subtype.eq; funext n
  /-
    case a.h
    α : Type u
    s : Computation α
    H : Not s.Terminates
    n : Nat
    ⊢ Eq (↑s n) (↑(Computation.empty α) n)
  -/
  induction' h : s.val n with _; · rfl
                                   /-
                                     🎉 no goals
                                   -/
  /-
    case a.h.some
    α : Type u
    s : Computation α
    H : Not s.Terminates
    n : Nat
    val✝ : α
    h : Eq (↑s n) (Option.some val✝)
    ⊢ Eq (Option.some val✝) (↑(Computation.empty α) n)
  -/
  refine absurd ?_ H; exact ⟨⟨_, _, h.symm⟩⟩
                      /-
                        🎉 no goals
                      -/


theorem thinkN_mem {s : Computation α} {a} : ∀ n, a ∈ thinkN s n ↔ a ∈ s
  | 0 => Iff.rfl
  | n + 1 => Iff.trans ⟨of_think_mem, think_mem⟩ (thinkN_mem n)


instance thinkN_terminates (s : Computation α) : ∀ [Terminates s] (n), Terminates (thinkN s n)
  | ⟨⟨a, h⟩⟩, n => ⟨⟨a, (thinkN_mem n).2 h⟩⟩


theorem of_thinkN_terminates (s : Computation α) (n) : Terminates (thinkN s n) → Terminates s
  | ⟨⟨a, h⟩⟩ => ⟨⟨a, (thinkN_mem _).1 h⟩⟩


/-- `Promises s a`, or `s ~> a`, asserts that although the computation `s`
  may not terminate, if it does, then the result is `a`. -/
def Promises (s : Computation α) (a : α) : Prop :=
  ∀ ⦃a'⦄, a' ∈ s → a = a'


/-- `Promises s a`, or `s ~> a`, asserts that although the computation `s`
  may not terminate, if it does, then the result is `a`. -/
scoped infixl:50 " ~> " => Promises


theorem mem_promises {s : Computation α} {a : α} : a ∈ s → s ~> a := fun h _ => mem_unique h


theorem empty_promises (a : α) : empty α ~> a := fun _ h => absurd h (not_mem_empty _)


/-- `length s` gets the number of steps of a terminating computation -/
def length : ℕ :=
  Nat.find ((terminates_def _).1 h)


/-- `get s` returns the result of a terminating computation -/
def get : α :=
  Option.get _ (Nat.find_spec <| (terminates_def _).1 h)


theorem get_mem : get s ∈ s :=
  Exists.intro (length s) (Option.eq_some_of_isSome _).symm


theorem get_eq_of_mem {a} : a ∈ s → get s = a :=
  mem_unique (get_mem _)


                                                    /-
                                                      α : Type u
                                                      s : Computation α
                                                      h : s.Terminates
                                                      a : α
                                                      ⊢ Eq s.get a → Membership.mem s a
                                                    -/
theorem mem_of_get_eq {a} : get s = a → a ∈ s := by intro h; rw [← h]; apply get_mem
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem get_think : get (think s) = get s :=
  get_eq_of_mem _ <|
    let ⟨n, h⟩ := get_mem s
    ⟨n + 1, h⟩


@[simp]
theorem get_thinkN (n) : get (thinkN s n) = get s :=
  get_eq_of_mem _ <| (thinkN_mem _).2 (get_mem _)


theorem get_promises : s ~> get s := fun _ => get_eq_of_mem _


theorem mem_of_promises {a} (p : s ~> a) : a ∈ s := by
  /-
    α : Type u
    s : Computation α
    h : s.Terminates
    a : α
    p : s.Promises a
    ⊢ Membership.mem s a
  -/
  cases' h with h
  /-
    case mk
    α : Type u
    s : Computation α
    a : α
    p : s.Promises a
    h : Exists fun a => Membership.mem s a
    ⊢ Membership.mem s a
  -/
  cases' h with a' h
  /-
    case mk.intro
    α : Type u
    s : Computation α
    a : α
    p : s.Promises a
    a' : α
    h : Membership.mem s a'
    ⊢ Membership.mem s a
  -/
  rw [p h]
  /-
    case mk.intro
    α : Type u
    s : Computation α
    a : α
    p : s.Promises a
    a' : α
    h : Membership.mem s a'
    ⊢ Membership.mem s a'
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem get_eq_of_promises {a} : s ~> a → get s = a :=
  get_eq_of_mem _ ∘ mem_of_promises _


/-- `Results s a n` completely characterizes a terminating computation:
  it asserts that `s` terminates after exactly `n` steps, with result `a`. -/
def Results (s : Computation α) (a : α) (n : ℕ) :=
  ∃ h : a ∈ s, @length _ s (terminates_of_mem h) = n


theorem results_of_terminates (s : Computation α) [_T : Terminates s] :
    Results s (get s) (length s) :=
  ⟨get_mem _, rfl⟩


theorem results_of_terminates' (s : Computation α) [T : Terminates s] {a} (h : a ∈ s) :
                                 /-
                                   α : Type u
                                   s : Computation α
                                   T : s.Terminates
                                   a : α
                                   h : Membership.mem s a
                                   ⊢ s.Results a s.length
                                 -/
    Results s a (length s) := by rw [← get_eq_of_mem _ h]; apply results_of_terminates
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem Results.mem {s : Computation α} {a n} : Results s a n → a ∈ s
  | ⟨m, _⟩ => m


theorem Results.terminates {s : Computation α} {a n} (h : Results s a n) : Terminates s :=
  terminates_of_mem h.mem


theorem Results.length {s : Computation α} {a n} [_T : Terminates s] : Results s a n → length s = n
  | ⟨_, h⟩ => h


theorem Results.val_unique {s : Computation α} {a b m n} (h1 : Results s a m) (h2 : Results s b n) :
    a = b :=
  mem_unique h1.mem h2.mem


theorem Results.len_unique {s : Computation α} {a b m n} (h1 : Results s a m) (h2 : Results s b n) :
                /-
                  α : Type u
                  s : Computation α
                  a b : α
                  m n : Nat
                  h1 : s.Results a m
                  h2 : s.Results b n
                  ⊢ Eq m n
                -/
    m = n := by haveI := h1.terminates; haveI := h2.terminates; rw [← h1.length, h2.length]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem exists_results_of_mem {s : Computation α} {a} (h : a ∈ s) : ∃ n, Results s a n :=
  haveI := terminates_of_mem h
  ⟨_, results_of_terminates' s h⟩


@[simp]
theorem get_pure (a : α) : get (pure a) = a :=
  get_eq_of_mem _ ⟨0, rfl⟩


@[simp]
theorem length_pure (a : α) : length (pure a) = 0 :=
  let h := Computation.ret_terminates a
  Nat.eq_zero_of_le_zero <| Nat.find_min' ((terminates_def (pure a)).1 h) rfl


theorem results_pure (a : α) : Results (pure a) a 0 :=
  ⟨ret_mem a, length_pure _⟩


@[simp]
theorem length_think (s : Computation α) [h : Terminates s] : length (think s) = length s + 1 := by
  /-
    α : Type u
    s : Computation α
    h : s.Terminates
    ⊢ Eq s.think.length (HAdd.hAdd s.length 1)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      s : Computation α
      h : s.Terminates
      ⊢ LE.le s.think.length (HAdd.hAdd s.length 1)
    -/
  · exact Nat.find_min' _ (Nat.find_spec ((terminates_def _).1 h))
    /-
      🎉 no goals
    -/
  · have : (Option.isSome ((think s).val (length (think s))) : Prop) :=
      Nat.find_spec ((terminates_def _).1 s.think_terminates)
    /-
      case a
      α : Type u
      s : Computation α
      h : s.Terminates
      this : Eq (↑s.think s.think.length).isSome Bool.true
      ⊢ LE.le (HAdd.hAdd s.length 1) s.think.length
    -/
    revert this; cases' length (think s) with n <;> intro this
      /-
        case a.zero
        α : Type u
        s : Computation α
        h : s.Terminates
        this : Eq (↑s.think 0).isSome Bool.true
        ⊢ LE.le (HAdd.hAdd s.length 1) 0
      -/
    · simp [think, Stream'.cons] at this
      /-
        🎉 no goals
      -/
      /-
        case a.succ
        α : Type u
        s : Computation α
        h : s.Terminates
        n : Nat
        this : Eq (↑s.think (HAdd.hAdd n 1)).isSome Bool.true
        ⊢ LE.le (HAdd.hAdd s.length 1) (HAdd.hAdd n 1)
      -/
    · apply Nat.succ_le_succ
      /-
        case a.succ.a
        α : Type u
        s : Computation α
        h : s.Terminates
        n : Nat
        this : Eq (↑s.think (HAdd.hAdd n 1)).isSome Bool.true
        ⊢ LE.le s.length n
      -/
      apply Nat.find_min'
      /-
        case a.succ.a.h
        α : Type u
        s : Computation α
        h : s.Terminates
        n : Nat
        this : Eq (↑s.think (HAdd.hAdd n 1)).isSome Bool.true
        ⊢ Eq (↑s n).isSome Bool.true
      -/
      apply this
      /-
        🎉 no goals
      -/


theorem results_think {s : Computation α} {a n} (h : Results s a n) : Results (think s) a (n + 1) :=
  haveI := h.terminates
                       /-
                         α : Type u
                         s : Computation α
                         a : α
                         n : Nat
                         h : s.Results a n
                         this : s.Terminates
                         ⊢ Eq s.think.length (HAdd.hAdd n 1)
                       -/
  ⟨think_mem h.mem, by rw [length_think, h.length]⟩
                       /-
                         🎉 no goals
                       -/


theorem of_results_think {s : Computation α} {a n} (h : Results (think s) a n) :
    ∃ m, Results s a m ∧ n = m + 1 := by
  /-
    α : Type u
    s : Computation α
    a : α
    n : Nat
    h : s.think.Results a n
    ⊢ Exists fun m => And (s.Results a m) (Eq n (HAdd.hAdd m 1))
  -/
  haveI := of_think_terminates h.terminates
  /-
    α : Type u
    s : Computation α
    a : α
    n : Nat
    h : s.think.Results a n
    this : s.Terminates
    ⊢ Exists fun m => And (s.Results a m) (Eq n (HAdd.hAdd m 1))
  -/
  have := results_of_terminates' _ (of_think_mem h.mem)
  /-
    α : Type u
    s : Computation α
    a : α
    n : Nat
    h : s.think.Results a n
    this✝ : s.Terminates
    this : s.Results a s.length
    ⊢ Exists fun m => And (s.Results a m) (Eq n (HAdd.hAdd m 1))
  -/
  exact ⟨_, this, Results.len_unique h (results_think this)⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem results_think_iff {s : Computation α} {a n} : Results (think s) a (n + 1) ↔ Results s a n :=
  ⟨fun h => by
    /-
      α : Type u
      s : Computation α
      a : α
      n : Nat
      h : s.think.Results a (HAdd.hAdd n 1)
      ⊢ s.Results a n
    -/
    let ⟨n', r, e⟩ := of_results_think h
    /-
      α : Type u
      s : Computation α
      a : α
      n : Nat
      h : s.think.Results a (HAdd.hAdd n 1)
      n' : Nat
      r : s.Results a n'
      e : Eq (HAdd.hAdd n 1) (HAdd.hAdd n' 1)
      ⊢ s.Results a n
    -/
    injection e with h'; rwa [h'], results_think⟩
                         /-
                           🎉 no goals
                         -/


theorem results_thinkN {s : Computation α} {a m} :
    ∀ n, Results s a m → Results (thinkN s n) a (m + n)
  | 0, h => h
  | n + 1, h => results_think (results_thinkN n h)


theorem results_thinkN_pure (a : α) (n) : Results (thinkN (pure a) n) a n := by
  /-
    α : Type u
    a : α
    n : Nat
    ⊢ ((Computation.pure a).thinkN n).Results a n
  -/
  have := results_thinkN n (results_pure a); rwa [Nat.zero_add] at this
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem length_thinkN (s : Computation α) [_h : Terminates s] (n) :
    length (thinkN s n) = length s + n :=
  (results_thinkN n (results_of_terminates _)).length


theorem eq_thinkN {s : Computation α} {a n} (h : Results s a n) : s = thinkN (pure a) n := by
  /-
    α : Type u
    s : Computation α
    a : α
    n : Nat
    h : s.Results a n
    ⊢ Eq s ((Computation.pure a).thinkN n)
  -/
  revert s
  /-
    α : Type u
    a : α
    n : Nat
    ⊢ ∀ {s : Computation α}, s.Results a n → Eq s ((Computation.pure a).thinkN n)
  -/
  induction n with | zero => _ | succ n IH => _ <;>
   /-
     case zero
     α : Type u
     a : α
     ⊢ ∀ {s : Computation α}, s.Results a 0 → Eq s ((Computation.pure a).thinkN 0)
   -/
  (intro s; apply recOn s (fun a' => _) fun s => _) <;> intro a h
    /-
      α : Type u
      a✝ : α
      s : Computation α
      a : α
      h : (Computation.pure a).Results a✝ 0
      ⊢ Eq (Computation.pure a) ((Computation.pure a✝).thinkN 0)
    -/
  · rw [← eq_of_pure_mem h.mem]
    /-
      α : Type u
      a✝ : α
      s : Computation α
      a : α
      h : (Computation.pure a).Results a✝ 0
      ⊢ Eq (Computation.pure a✝) ((Computation.pure a✝).thinkN 0)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      a✝ : α
      s a : Computation α
      h : a.think.Results a✝ 0
      ⊢ Eq a.think ((Computation.pure a✝).thinkN 0)
    -/
  · cases' of_results_think h with n h
    /-
      case intro
      α : Type u
      a✝ : α
      s a : Computation α
      h✝ : a.think.Results a✝ 0
      n : Nat
      h : And (a.Results a✝ n) (Eq 0 (HAdd.hAdd n 1))
      ⊢ Eq a.think ((Computation.pure a✝).thinkN 0)
    -/
    cases h
    /-
      case intro.intro
      α : Type u
      a✝ : α
      s a : Computation α
      h : a.think.Results a✝ 0
      n : Nat
      left✝ : a.Results a✝ n
      right✝ : Eq 0 (HAdd.hAdd n 1)
      ⊢ Eq a.think ((Computation.pure a✝).thinkN 0)
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      a✝ : α
      n : Nat
      IH : ∀ {s : Computation α}, s.Results a✝ n → Eq s ((Computation.pure a✝).think …
      s : Computation α
      a : α
      h : (Computation.pure a).Results a✝ (HAdd.hAdd n 1)
      ⊢ Eq (Computation.pure a) ((Computation.pure a✝).thinkN (HAdd.hAdd n 1))
    -/
  · have := h.len_unique (results_pure _)
    /-
      α : Type u
      a✝ : α
      n : Nat
      IH : ∀ {s : Computation α}, s.Results a✝ n → Eq s ((Computation.pure a✝).think …
      s : Computation α
      a : α
      h : (Computation.pure a).Results a✝ (HAdd.hAdd n 1)
      this : Eq (HAdd.hAdd n 1) 0
      ⊢ Eq (Computation.pure a) ((Computation.pure a✝).thinkN (HAdd.hAdd n 1))
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      a✝ : α
      n : Nat
      IH : ∀ {s : Computation α}, s.Results a✝ n → Eq s ((Computation.pure a✝).think …
      s a : Computation α
      h : a.think.Results a✝ (HAdd.hAdd n 1)
      ⊢ Eq a.think ((Computation.pure a✝).thinkN (HAdd.hAdd n 1))
    -/
  · rw [IH (results_think_iff.1 h)]
    /-
      α : Type u
      a✝ : α
      n : Nat
      IH : ∀ {s : Computation α}, s.Results a✝ n → Eq s ((Computation.pure a✝).think …
      s a : Computation α
      h : a.think.Results a✝ (HAdd.hAdd n 1)
      ⊢ Eq ((Computation.pure a✝).thinkN n).think ((Computation.pure a✝).thinkN (HAd …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem eq_thinkN' (s : Computation α) [_h : Terminates s] :
    s = thinkN (pure (get s)) (length s) :=
  eq_thinkN (results_of_terminates _)


/-- Recursor based on membership -/
def memRecOn {C : Computation α → Sort v} {a s} (M : a ∈ s) (h1 : C (pure a))
    (h2 : ∀ s, C s → C (think s)) : C s := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    C : Computation α → Sort v
    a : α
    s : Computation α
    M : Membership.mem s a
    h1 : C (Computation.pure a)
    h2 : (s : Computation α) → C s → C s.think
    ⊢ C s
  -/
  haveI T := terminates_of_mem M
  /-
    α : Type u
    β : Type v
    γ : Type w
    C : Computation α → Sort v
    a : α
    s : Computation α
    M : Membership.mem s a
    h1 : C (Computation.pure a)
    h2 : (s : Computation α) → C s → C s.think
    T : s.Terminates
    ⊢ C s
  -/
  rw [eq_thinkN' s, get_eq_of_mem s M]
  /-
    α : Type u
    β : Type v
    γ : Type w
    C : Computation α → Sort v
    a : α
    s : Computation α
    M : Membership.mem s a
    h1 : C (Computation.pure a)
    h2 : (s : Computation α) → C s → C s.think
    T : s.Terminates
    ⊢ C ((Computation.pure a).thinkN s.length)
  -/
  generalize length s = n
  /-
    α : Type u
    β : Type v
    γ : Type w
    C : Computation α → Sort v
    a : α
    s : Computation α
    M : Membership.mem s a
    h1 : C (Computation.pure a)
    h2 : (s : Computation α) → C s → C s.think
    T : s.Terminates
    n : Nat
    ⊢ C ((Computation.pure a).thinkN n)
  -/
  induction' n with n IH; exacts [h1, h2 _ IH]
                          /-
                            🎉 no goals
                          -/


/-- Recursor based on assertion of `Terminates`-/
def terminatesRecOn
    {C : Computation α → Sort v}
    (s) [Terminates s]
    (h1 : ∀ a, C (pure a))
    (h2 : ∀ s, C s → C (think s)) : C s :=
  memRecOn (get_mem s) (h1 _) h2


/-- Map a function on the result of a computation. -/
def map (f : α → β) : Computation α → Computation β
  | ⟨s, al⟩ =>
    ⟨s.map fun o => Option.casesOn o none (some ∘ f), fun n b => by
      /-
        α : Type u
        β : Type v
        γ : Type w
        f : α → β
        s : Stream' (Option α)
        al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
        n : Nat
        b : β
        ⊢ Eq (Stream'.map (fun o => Option.casesOn o Option.none (Function.comp Option …
      -/
      dsimp [Stream'.map, Stream'.get]
      /-
        α : Type u
        β : Type v
        γ : Type w
        f : α → β
        s : Stream' (Option α)
        al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
        n : Nat
        b : β
        ⊢ Eq (Option.rec Option.none (fun val => Option.some (f val)) (s n)) (Option.s …
      -/
      induction' e : s n with a <;> intro h
        /-
          case none
          α : Type u
          β : Type v
          γ : Type w
          f : α → β
          s : Stream' (Option α)
          al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
          n : Nat
          b : β
          e : Eq (s n) Option.none
          h : Eq (Option.rec Option.none (fun val => Option.some (f val)) Option.none) ( …
          ⊢ Eq (Option.rec Option.none (fun val => Option.some (f val)) (s (HAdd.hAdd n  …
        -/
      · contradiction
        /-
          🎉 no goals
        -/
        /-
          case some
          α : Type u
          β : Type v
          γ : Type w
          f : α → β
          s : Stream' (Option α)
          al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
          n : Nat
          b : β
          a : α
          e : Eq (s n) (Option.some a)
          h : Eq (Option.rec Option.none (fun val => Option.some (f val)) (Option.some a …
          ⊢ Eq (Option.rec Option.none (fun val => Option.some (f val)) (s (HAdd.hAdd n  …
        -/
      · rw [al e]; exact h⟩
                   /-
                     🎉 no goals
                   -/


/-- bind over a `Sum` of `Computation`-/
def Bind.g : β ⊕ Computation β → β ⊕ (Computation α ⊕ Computation β)
  | Sum.inl b => Sum.inl b
  | Sum.inr cb' => Sum.inr <| Sum.inr cb'


/-- bind over a function mapping `α` to a `Computation`-/
def Bind.f (f : α → Computation β) :
    Computation α ⊕ Computation β → β ⊕ (Computation α ⊕ Computation β)
  | Sum.inl ca =>
    match destruct ca with
    | Sum.inl a => Bind.g <| destruct (f a)
    | Sum.inr ca' => Sum.inr <| Sum.inl ca'
  | Sum.inr cb => Bind.g <| destruct cb


/-- Compose two computations into a monadic `bind` operation. -/
def bind (c : Computation α) (f : α → Computation β) : Computation β :=
  corec (Bind.f f) (Sum.inl c)


instance : Bind Computation :=
  ⟨@bind⟩


theorem has_bind_eq_bind {β} (c : Computation α) (f : α → Computation β) : c >>= f = bind c f :=
  rfl


/-- Flatten a computation of computations into a single computation. -/
def join (c : Computation (Computation α)) : Computation α :=
  c >>= id


@[simp]
theorem map_pure (f : α → β) (a) : map f (pure a) = pure (f a) :=
  rfl


@[simp]
theorem map_think (f : α → β) : ∀ s, map f (think s) = think (map f s)
                  /-
                    α : Type u
                    β : Type v
                    f : α → β
                    s : Stream' (Option α)
                    al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
                    ⊢ Eq (Computation.map f (Computation.think ⟨s, al⟩)) (Computation.map f ⟨s, al …
                  -/
  | ⟨s, al⟩ => by apply Subtype.eq; dsimp [think, map]; rw [Stream'.map_cons]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem destruct_map (f : α → β) (s) : destruct (map f s) = lmap f (rmap (map f) (destruct s)) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Computation α
    ⊢ Eq (Computation.map f s).destruct (Computation.lmap f (Computation.rmap (Com …
  -/
                              /-
                                🎉 no goals
                              -/
  apply s.recOn <;> intro <;> simp
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem map_id : ∀ s : Computation α, map id s = s
  | ⟨f, al⟩ => by
    /-
      α : Type u
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      ⊢ Eq (Computation.map id ⟨f, al⟩) ⟨f, al⟩
    -/
    apply Subtype.eq; simp only [map, comp_apply, id_eq]
    /-
      case a
      α : Type u
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      ⊢ Eq (Stream'.map (fun o => Option.rec Option.none (fun val => Option.some val …
    -/
    have e : @Option.rec α (fun _ => Option α) none some = id := by ext ⟨⟩ <;> rfl
    /-
      case a
      α : Type u
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      e : Eq (Option.rec Option.none Option.some) id
      ⊢ Eq (Stream'.map (fun o => Option.rec Option.none (fun val => Option.some val …
    -/
    have h : ((fun x : Option α => x) = id) := rfl
    /-
      case a
      α : Type u
      f : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (f n) (Option.some a) → Eq (f (HAdd.hAdd n 1)) (O …
      e : Eq (Option.rec Option.none Option.some) id
      h : Eq (fun x => x) id
      ⊢ Eq (Stream'.map (fun o => Option.rec Option.none (fun val => Option.some val …
    -/
    simp [e, h, Stream'.map_id]
    /-
      🎉 no goals
    -/


theorem map_comp (f : α → β) (g : β → γ) : ∀ s : Computation α, map (g ∘ f) s = map g (map f s)
  | ⟨s, al⟩ => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → β
      g : β → γ
      s : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
      ⊢ Eq (Computation.map (Function.comp g f) ⟨s, al⟩) (Computation.map g (Computa …
    -/
    apply Subtype.eq; dsimp [map]
    /-
      case a
      α : Type u
      β : Type v
      γ : Type w
      f : α → β
      g : β → γ
      s : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
      ⊢ Eq (Stream'.map (fun o => Option.rec Option.none (fun val => Option.some (g  …
    -/
    apply congr_arg fun f : _ → Option γ => Stream'.map f s
    /-
      case a
      α : Type u
      β : Type v
      γ : Type w
      f : α → β
      g : β → γ
      s : Stream' (Option α)
      al : ∀ ⦃n : Nat⦄ ⦃a : α⦄, Eq (s n) (Option.some a) → Eq (s (HAdd.hAdd n 1)) (O …
      ⊢ Eq (fun o => Option.rec Option.none (fun val => Option.some (g (f val))) o)  …
    -/
               /-
                 🎉 no goals
               -/
    ext ⟨⟩ <;> rfl
               /-
                 🎉 no goals
               -/


@[simp]
theorem ret_bind (a) (f : α → Computation β) : bind (pure a) f = f a := by
  apply
    eq_of_bisim fun c₁ c₂ => c₁ = bind (pure a) f ∧ c₂ = f a ∨ c₁ = corec (Bind.f f) (Sum.inr c₂)
    /-
      case bisim
      α : Type u
      β : Type v
      a : α
      f : α → Computation β
      ⊢ Computation.IsBisimulation fun c₁ c₂ => Or (And (Eq c₁ ((Computation.pure a) …
    -/
  · intro c₁ c₂ h
    match c₁, c₂, h with
    | _, _, Or.inl ⟨rfl, rfl⟩ =>
      simp only [BisimO, bind, Bind.f, corec_eq, rmap, destruct_pure]
      cases' destruct (f a) with b cb <;> simp [Bind.g]
    | _, c, Or.inr rfl =>
      simp only [BisimO, Bind.f, corec_eq, rmap]
      cases' destruct c with b cb <;> simp [Bind.g]
    /-
      case r
      α : Type u
      β : Type v
      a : α
      f : α → Computation β
      ⊢ Or (And (Eq ((Computation.pure a).bind f) ((Computation.pure a).bind f)) (Eq …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem think_bind (c) (f : α → Computation β) : bind (think c) f = think (bind c f) :=
                          /-
                            α : Type u
                            β : Type v
                            c : Computation α
                            f : α → Computation β
                            ⊢ Eq (c.think.bind f).destruct (Sum.inr (c.bind f))
                          -/
  destruct_eq_think <| by simp [bind, Bind.f]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem bind_pure (f : α → β) (s) : bind s (pure ∘ f) = map f s := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Computation α
    ⊢ Eq (s.bind (Function.comp Computation.pure f)) (Computation.map f s)
  -/
  apply eq_of_bisim fun c₁ c₂ => c₁ = c₂ ∨ ∃ s, c₁ = bind s (pure ∘ f) ∧ c₂ = map f s
    /-
      case bisim
      α : Type u
      β : Type v
      f : α → β
      s : Computation α
      ⊢ Computation.IsBisimulation fun c₁ c₂ => Or (Eq c₁ c₂) (Exists fun s => And ( …
    -/
  · intro c₁ c₂ h
    match c₁, c₂, h with
    | _, c₂, Or.inl (Eq.refl _) => cases' destruct c₂ with b cb <;> simp
    | _, _, Or.inr ⟨s, rfl, rfl⟩ =>
      apply recOn s <;> intro s
      · simp
      · simpa using Or.inr ⟨s, rfl, rfl⟩
    /-
      case r
      α : Type u
      β : Type v
      f : α → β
      s : Computation α
      ⊢ Or (Eq (s.bind (Function.comp Computation.pure f)) (Computation.map f s)) (E …
    -/
  · exact Or.inr ⟨s, rfl, rfl⟩
    /-
      🎉 no goals
    -/

-- Porting note: used to use `rw [bind_pure]`

@[simp]
theorem bind_pure' (s : Computation α) : bind s pure = s := by
  /-
    α : Type u
    s : Computation α
    ⊢ Eq (s.bind Computation.pure) s
  -/
  apply eq_of_bisim fun c₁ c₂ => c₁ = c₂ ∨ ∃ s, c₁ = bind s pure ∧ c₂ = s
    /-
      case bisim
      α : Type u
      s : Computation α
      ⊢ Computation.IsBisimulation fun c₁ c₂ => Or (Eq c₁ c₂) (Exists fun s => And ( …
    -/
  · intro c₁ c₂ h
    match c₁, c₂, h with
    | _, c₂, Or.inl (Eq.refl _) => cases' destruct c₂ with b cb <;> simp
    | _, _, Or.inr ⟨s, rfl, rfl⟩ =>
      apply recOn s <;> intro s <;> simp
    /-
      case r
      α : Type u
      s : Computation α
      ⊢ Or (Eq (s.bind Computation.pure) s) (Exists fun s_1 => And (Eq (s.bind Compu …
    -/
  · exact Or.inr ⟨s, rfl, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem bind_assoc (s : Computation α) (f : α → Computation β) (g : β → Computation γ) :
    bind (bind s f) g = bind s fun x : α => bind (f x) g := by
  apply
    eq_of_bisim fun c₁ c₂ =>
      c₁ = c₂ ∨ ∃ s, c₁ = bind (bind s f) g ∧ c₂ = bind s fun x : α => bind (f x) g
    /-
      case bisim
      α : Type u
      β : Type v
      γ : Type w
      s : Computation α
      f : α → Computation β
      g : β → Computation γ
      ⊢ Computation.IsBisimulation fun c₁ c₂ => Or (Eq c₁ c₂) (Exists fun s => And ( …
    -/
  · intro c₁ c₂ h
    match c₁, c₂, h with
    | _, c₂, Or.inl (Eq.refl _) => cases' destruct c₂ with b cb <;> simp
    | _, _, Or.inr ⟨s, rfl, rfl⟩ =>
      apply recOn s <;> intro s
      · simp only [BisimO, ret_bind]; generalize f s = fs
        apply recOn fs <;> intro t <;> simp
        · cases' destruct (g t) with b cb <;> simp
      · simpa  [BisimO] using Or.inr ⟨s, rfl, rfl⟩
    /-
      case r
      α : Type u
      β : Type v
      γ : Type w
      s : Computation α
      f : α → Computation β
      g : β → Computation γ
      ⊢ Or (Eq ((s.bind f).bind g) (s.bind fun x => (f x).bind g)) (Exists fun s_1 = …
    -/
  · exact Or.inr ⟨s, rfl, rfl⟩
    /-
      🎉 no goals
    -/


theorem results_bind {s : Computation α} {f : α → Computation β} {a b m n} (h1 : Results s a m)
    (h2 : Results (f a) b n) : Results (bind s f) b (n + m) := by
  /-
    α : Type u
    β : Type v
    s : Computation α
    f : α → Computation β
    a : α
    b : β
    m n : Nat
    h1 : s.Results a m
    h2 : (f a).Results b n
    ⊢ (s.bind f).Results b (HAdd.hAdd n m)
  -/
  have := h1.mem; revert m
  /-
    α : Type u
    β : Type v
    s : Computation α
    f : α → Computation β
    a : α
    b : β
    n : Nat
    h2 : (f a).Results b n
    this : Membership.mem s a
    ⊢ ∀ {m : Nat}, s.Results a m → (s.bind f).Results b (HAdd.hAdd n m)
  -/
  apply memRecOn this _ fun s IH => _
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      ⊢ ∀ {m : Nat}, (Computation.pure a).Results a m → ((Computation.pure a).bind f …
    -/
  · intro _ h1
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      m✝ : Nat
      h1 : (Computation.pure a).Results a m✝
      ⊢ ((Computation.pure a).bind f).Results b (HAdd.hAdd n m✝)
    -/
    rw [ret_bind]
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      m✝ : Nat
      h1 : (Computation.pure a).Results a m✝
      ⊢ (f a).Results b (HAdd.hAdd n m✝)
    -/
    rw [h1.len_unique (results_pure _)]
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      m✝ : Nat
      h1 : (Computation.pure a).Results a m✝
      ⊢ (f a).Results b (HAdd.hAdd n 0)
    -/
    exact h2
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      ⊢ ∀ (s : Computation α), (∀ {m : Nat}, s.Results a m → (s.bind f).Results b (H …
    -/
  · intro _ h3 _ h1
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      s✝ : Computation α
      h3 : ∀ {m : Nat}, s✝.Results a m → (s✝.bind f).Results b (HAdd.hAdd n m)
      m✝ : Nat
      h1 : s✝.think.Results a m✝
      ⊢ (s✝.think.bind f).Results b (HAdd.hAdd n m✝)
    -/
    rw [think_bind]
    /-
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      s✝ : Computation α
      h3 : ∀ {m : Nat}, s✝.Results a m → (s✝.bind f).Results b (HAdd.hAdd n m)
      m✝ : Nat
      h1 : s✝.think.Results a m✝
      ⊢ (s✝.bind f).think.Results b (HAdd.hAdd n m✝)
    -/
    cases' of_results_think h1 with m' h
    /-
      case intro
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      s✝ : Computation α
      h3 : ∀ {m : Nat}, s✝.Results a m → (s✝.bind f).Results b (HAdd.hAdd n m)
      m✝ : Nat
      h1 : s✝.think.Results a m✝
      m' : Nat
      h : And (s✝.Results a m') (Eq m✝ (HAdd.hAdd m' 1))
      ⊢ (s✝.bind f).think.Results b (HAdd.hAdd n m✝)
    -/
    cases' h with h1 e
    /-
      case intro.intro
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      s✝ : Computation α
      h3 : ∀ {m : Nat}, s✝.Results a m → (s✝.bind f).Results b (HAdd.hAdd n m)
      m✝ : Nat
      h1✝ : s✝.think.Results a m✝
      m' : Nat
      h1 : s✝.Results a m'
      e : Eq m✝ (HAdd.hAdd m' 1)
      ⊢ (s✝.bind f).think.Results b (HAdd.hAdd n m✝)
    -/
    rw [e]
    /-
      case intro.intro
      α : Type u
      β : Type v
      s : Computation α
      f : α → Computation β
      a : α
      b : β
      n : Nat
      h2 : (f a).Results b n
      this : Membership.mem s a
      s✝ : Computation α
      h3 : ∀ {m : Nat}, s✝.Results a m → (s✝.bind f).Results b (HAdd.hAdd n m)
      m✝ : Nat
      h1✝ : s✝.think.Results a m✝
      m' : Nat
      h1 : s✝.Results a m'
      e : Eq m✝ (HAdd.hAdd m' 1)
      ⊢ (s✝.bind f).think.Results b (HAdd.hAdd n (HAdd.hAdd m' 1))
    -/
    exact results_think (h3 h1)
    /-
      🎉 no goals
    -/


theorem mem_bind {s : Computation α} {f : α → Computation β} {a b} (h1 : a ∈ s) (h2 : b ∈ f a) :
    b ∈ bind s f :=
  let ⟨_, h1⟩ := exists_results_of_mem h1
  let ⟨_, h2⟩ := exists_results_of_mem h2
  (results_bind h1 h2).mem


instance terminates_bind (s : Computation α) (f : α → Computation β) [Terminates s]
    [Terminates (f (get s))] : Terminates (bind s f) :=
  terminates_of_mem (mem_bind (get_mem s) (get_mem (f (get s))))


@[simp]
theorem get_bind (s : Computation α) (f : α → Computation β) [Terminates s]
    [Terminates (f (get s))] : get (bind s f) = get (f (get s)) :=
  get_eq_of_mem _ (mem_bind (get_mem s) (get_mem (f (get s))))


@[simp]
theorem length_bind (s : Computation α) (f : α → Computation β) [_T1 : Terminates s]
    [_T2 : Terminates (f (get s))] : length (bind s f) = length (f (get s)) + length s :=
  (results_of_terminates _).len_unique <|
    results_bind (results_of_terminates _) (results_of_terminates _)


theorem of_results_bind {s : Computation α} {f : α → Computation β} {b k} :
    Results (bind s f) b k → ∃ a m n, Results s a m ∧ Results (f a) b n ∧ k = n + m := by
  /-
    α : Type u
    β : Type v
    s : Computation α
    f : α → Computation β
    b : β
    k : Nat
    ⊢ (s.bind f).Results b k → Exists fun a => Exists fun m => Exists fun n => And …
  -/
  induction k generalizing s with | zero => _ | succ n IH => _
        /-
          case zero
          α : Type u
          β : Type v
          f : α → Computation β
          b : β
          s : Computation α
          ⊢ (s.bind f).Results b 0 → Exists fun a => Exists fun m => Exists fun n => And …
        -/
    <;> apply recOn s (fun a => _) fun s' => _ <;> intro e h
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      s : Computation α
      e : α
      h : ((Computation.pure e).bind f).Results b 0
      ⊢ Exists fun a => Exists fun m => Exists fun n => And ((Computation.pure e).Re …
    -/
  · simp only [ret_bind] at h
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      s : Computation α
      e : α
      h : (f e).Results b 0
      ⊢ Exists fun a => Exists fun m => Exists fun n => And ((Computation.pure e).Re …
    -/
    exact ⟨e, _, _, results_pure _, h, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      s e : Computation α
      h : (e.think.bind f).Results b 0
      ⊢ Exists fun a => Exists fun m => Exists fun n => And (e.think.Results a m) (A …
    -/
  · have := congr_arg head (eq_thinkN h)
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      s e : Computation α
      h : (e.think.bind f).Results b 0
      this : Eq (e.think.bind f).head ((Computation.pure b).thinkN 0).head
      ⊢ Exists fun a => Exists fun m => Exists fun n => And (e.think.Results a m) (A …
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      n : Nat
      IH : ∀ {s : Computation α}, (s.bind f).Results b n → Exists fun a => Exists fu …
      s : Computation α
      e : α
      h : ((Computation.pure e).bind f).Results b (HAdd.hAdd n 1)
      ⊢ Exists fun a => Exists fun m => Exists fun n_1 => And ((Computation.pure e). …
    -/
  · simp only [ret_bind] at h
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      n : Nat
      IH : ∀ {s : Computation α}, (s.bind f).Results b n → Exists fun a => Exists fu …
      s : Computation α
      e : α
      h : (f e).Results b (HAdd.hAdd n 1)
      ⊢ Exists fun a => Exists fun m => Exists fun n_1 => And ((Computation.pure e). …
    -/
    exact ⟨e, _, n + 1, results_pure _, h, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      n : Nat
      IH : ∀ {s : Computation α}, (s.bind f).Results b n → Exists fun a => Exists fu …
      s e : Computation α
      h : (e.think.bind f).Results b (HAdd.hAdd n 1)
      ⊢ Exists fun a => Exists fun m => Exists fun n_1 => And (e.think.Results a m)  …
    -/
  · simp only [think_bind, results_think_iff] at h
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      n : Nat
      IH : ∀ {s : Computation α}, (s.bind f).Results b n → Exists fun a => Exists fu …
      s e : Computation α
      h : (e.bind f).Results b n
      ⊢ Exists fun a => Exists fun m => Exists fun n_1 => And (e.think.Results a m)  …
    -/
    let ⟨a, m, n', h1, h2, e'⟩ := IH h
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      n : Nat
      IH : ∀ {s : Computation α}, (s.bind f).Results b n → Exists fun a => Exists fu …
      s e : Computation α
      h : (e.bind f).Results b n
      a : α
      m n' : Nat
      h1 : e.Results a m
      h2 : (f a).Results b n'
      e' : Eq n (HAdd.hAdd n' m)
      ⊢ Exists fun a => Exists fun m => Exists fun n_1 => And (e.think.Results a m)  …
    -/
    rw [e']
    /-
      α : Type u
      β : Type v
      f : α → Computation β
      b : β
      n : Nat
      IH : ∀ {s : Computation α}, (s.bind f).Results b n → Exists fun a => Exists fu …
      s e : Computation α
      h : (e.bind f).Results b n
      a : α
      m n' : Nat
      h1 : e.Results a m
      h2 : (f a).Results b n'
      e' : Eq n (HAdd.hAdd n' m)
      ⊢ Exists fun a => Exists fun m_1 => Exists fun n => And (e.think.Results a m_1 …
    -/
    exact ⟨a, m.succ, n', results_think h1, h2, rfl⟩
    /-
      🎉 no goals
    -/


theorem exists_of_mem_bind {s : Computation α} {f : α → Computation β} {b} (h : b ∈ bind s f) :
    ∃ a ∈ s, b ∈ f a :=
  let ⟨_, h⟩ := exists_results_of_mem h
  let ⟨a, _, _, h1, h2, _⟩ := of_results_bind h
  ⟨a, h1.mem, h2.mem⟩


theorem bind_promises {s : Computation α} {f : α → Computation β} {a b} (h1 : s ~> a)
    (h2 : f a ~> b) : bind s f ~> b := fun b' bB => by
  /-
    α : Type u
    β : Type v
    s : Computation α
    f : α → Computation β
    a : α
    b : β
    h1 : s.Promises a
    h2 : (f a).Promises b
    b' : β
    bB : Membership.mem (s.bind f) b'
    ⊢ Eq b b'
  -/
  rcases exists_of_mem_bind bB with ⟨a', a's, ba'⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    s : Computation α
    f : α → Computation β
    a : α
    b : β
    h1 : s.Promises a
    h2 : (f a).Promises b
    b' : β
    bB : Membership.mem (s.bind f) b'
    a' : α
    a's : Membership.mem s a'
    ba' : Membership.mem (f a') b'
    ⊢ Eq b b'
  -/
  rw [← h1 a's] at ba'; exact h2 ba'
                        /-
                          🎉 no goals
                        -/


instance monad : Monad Computation where
  map := @map
  pure := @pure
  bind := @bind


                                      /-
                                        α : Type u
                                        β : Type v
                                        γ : Type w
                                        ⊢ ∀ {α β : Type u_1} (x : α) (y : Computation β), Eq (Functor.mapConst x y) (F …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
instance : LawfulMonad Computation := LawfulMonad.mk'
                                      /-
                                        🎉 no goals
                                      -/
  (id_map := @map_id)
  (bind_pure_comp := @bind_pure)
  (pure_bind := @ret_bind)
  (bind_assoc := @bind_assoc)


theorem has_map_eq_map {β} (f : α → β) (c : Computation α) : f <$> c = map f c :=
  rfl


@[simp]
theorem pure_def (a) : (return a : Computation α) = pure a :=
  rfl


@[simp]
theorem map_pure' {α β} : ∀ (f : α → β) (a), f <$> pure a = pure (f a) :=
  map_pure


@[simp]
theorem map_think' {α β} : ∀ (f : α → β) (s), f <$> think s = think (f <$> s) :=
  map_think


theorem mem_map (f : α → β) {a} {s : Computation α} (m : a ∈ s) : f a ∈ map f s := by
  /-
    α : Type u
    β : Type v
    f : α → β
    a : α
    s : Computation α
    m : Membership.mem s a
    ⊢ Membership.mem (Computation.map f s) (f a)
  -/
  rw [← bind_pure]; apply mem_bind m; apply ret_mem
                                      /-
                                        🎉 no goals
                                      -/


theorem exists_of_mem_map {f : α → β} {b : β} {s : Computation α} (h : b ∈ map f s) :
    ∃ a, a ∈ s ∧ f a = b := by
  /-
    α : Type u
    β : Type v
    f : α → β
    b : β
    s : Computation α
    h : Membership.mem (Computation.map f s) b
    ⊢ Exists fun a => And (Membership.mem s a) (Eq (f a) b)
  -/
  rw [← bind_pure] at h
  /-
    α : Type u
    β : Type v
    f : α → β
    b : β
    s : Computation α
    h : Membership.mem (s.bind (Function.comp Computation.pure f)) b
    ⊢ Exists fun a => And (Membership.mem s a) (Eq (f a) b)
  -/
  let ⟨a, as, fb⟩ := exists_of_mem_bind h
  /-
    α : Type u
    β : Type v
    f : α → β
    b : β
    s : Computation α
    h : Membership.mem (s.bind (Function.comp Computation.pure f)) b
    a : α
    as : Membership.mem s a
    fb : Membership.mem (Function.comp Computation.pure f a) b
    ⊢ Exists fun a => And (Membership.mem s a) (Eq (f a) b)
  -/
  exact ⟨a, as, mem_unique (ret_mem _) fb⟩
  /-
    🎉 no goals
  -/


instance terminates_map (f : α → β) (s : Computation α) [Terminates s] : Terminates (map f s) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    s : Computation α
    inst✝ : s.Terminates
    ⊢ (Computation.map f s).Terminates
  -/
  rw [← bind_pure]; exact terminates_of_mem (mem_bind (get_mem s) (get_mem (α := β) (f (get s))))
                    /-
                      🎉 no goals
                    -/


theorem terminates_map_iff (f : α → β) (s : Computation α) : Terminates (map f s) ↔ Terminates s :=
  ⟨fun ⟨⟨_, h⟩⟩ =>
    let ⟨_, h1, _⟩ := exists_of_mem_map h
    ⟨⟨_, h1⟩⟩,
    @Computation.terminates_map _ _ _ _⟩

-- Parallel computation

/-- `c₁ <|> c₂` calculates `c₁` and `c₂` simultaneously, returning
  the first one that gives a result. -/
def orElse (c₁ : Computation α) (c₂ : Unit → Computation α) : Computation α :=
  @Computation.corec α (Computation α × Computation α)
    (fun ⟨c₁, c₂⟩ =>
      match destruct c₁ with
      | Sum.inl a => Sum.inl a
      | Sum.inr c₁' =>
        match destruct c₂ with
        | Sum.inl a => Sum.inl a
        | Sum.inr c₂' => Sum.inr (c₁', c₂'))
    (c₁, c₂ ())


instance instAlternativeComputation : Alternative Computation :=
  { Computation.monad with
    orElse := @orElse
    failure := @empty }

-- Porting note: Added unfolds as the code does not work without it

@[simp]
theorem ret_orElse (a : α) (c₂ : Computation α) : (pure a <|> c₂) = pure a :=
  destruct_eq_pure <| by
    /-
      α : Type u
      a : α
      c₂ : Computation α
      ⊢ Eq (HOrElse.hOrElse (Computation.pure a) fun x => c₂).destruct (Sum.inl a)
    -/
    unfold_projs
    /-
      α : Type u
      a : α
      c₂ : Computation α
      ⊢ Eq ((Computation.pure a).orElse fun x => c₂).destruct (Sum.inl a)
    -/
    simp [orElse]
    /-
      🎉 no goals
    -/

-- Porting note: Added unfolds as the code does not work without it

@[simp]
theorem orElse_pure (c₁ : Computation α) (a : α) : (think c₁ <|> pure a) = pure a :=
  destruct_eq_pure <| by
    /-
      α : Type u
      c₁ : Computation α
      a : α
      ⊢ Eq (HOrElse.hOrElse c₁.think fun x => Computation.pure a).destruct (Sum.inl a)
    -/
    unfold_projs
    /-
      α : Type u
      c₁ : Computation α
      a : α
      ⊢ Eq (c₁.think.orElse fun x => Computation.pure a).destruct (Sum.inl a)
    -/
    simp [orElse]
    /-
      🎉 no goals
    -/

-- Porting note: Added unfolds as the code does not work without it

@[simp]
theorem orElse_think (c₁ c₂ : Computation α) : (think c₁ <|> think c₂) = think (c₁ <|> c₂) :=
  destruct_eq_think <| by
    /-
      α : Type u
      c₁ c₂ : Computation α
      ⊢ Eq (HOrElse.hOrElse c₁.think fun x => c₂.think).destruct (Sum.inr (HOrElse.h …
    -/
    unfold_projs
    /-
      α : Type u
      c₁ c₂ : Computation α
      ⊢ Eq (c₁.think.orElse fun x => c₂.think).destruct (Sum.inr (c₁.orElse fun x => …
    -/
    simp [orElse]
    /-
      🎉 no goals
    -/


@[simp]
theorem empty_orElse (c) : (empty α <|> c) = c := by
  /-
    α : Type u
    c : Computation α
    ⊢ Eq (HOrElse.hOrElse (Computation.empty α) fun x => c) c
  -/
  apply eq_of_bisim (fun c₁ c₂ => (empty α <|> c₂) = c₁) _ rfl
  /-
    α : Type u
    c : Computation α
    ⊢ Computation.IsBisimulation fun c₁ c₂ => Eq (HOrElse.hOrElse (Computation.emp …
  -/
  intro s' s h; rw [← h]
  /-
    α : Type u
    c s' s : Computation α
    h : Eq (HOrElse.hOrElse (Computation.empty α) fun x => s) s'
    ⊢ Computation.BisimO (fun c₁ c₂ => Eq (HOrElse.hOrElse (Computation.empty α) f …
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  apply recOn s <;> intro s <;> rw [think_empty] <;> simp
  /-
    case h2
    α : Type u
    c s' s✝ : Computation α
    h : Eq (HOrElse.hOrElse (Computation.empty α) fun x => s✝) s'
    s : Computation α
    ⊢ Eq (HOrElse.hOrElse (Computation.empty α).think fun x => s) (HOrElse.hOrElse …
  -/
  rw [← think_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem orElse_empty (c : Computation α) : (c <|> empty α) = c := by
  /-
    α : Type u
    c : Computation α
    ⊢ Eq (HOrElse.hOrElse c fun x => Computation.empty α) c
  -/
  apply eq_of_bisim (fun c₁ c₂ => (c₂ <|> empty α) = c₁) _ rfl
  /-
    α : Type u
    c : Computation α
    ⊢ Computation.IsBisimulation fun c₁ c₂ => Eq (HOrElse.hOrElse c₂ fun x => Comp …
  -/
  intro s' s h; rw [← h]
  /-
    α : Type u
    c s' s : Computation α
    h : Eq (HOrElse.hOrElse s fun x => Computation.empty α) s'
    ⊢ Computation.BisimO (fun c₁ c₂ => Eq (HOrElse.hOrElse c₂ fun x => Computation …
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  apply recOn s <;> intro s <;> rw [think_empty] <;> simp
  /-
    case h2
    α : Type u
    c s' s✝ : Computation α
    h : Eq (HOrElse.hOrElse s✝ fun x => Computation.empty α) s'
    s : Computation α
    ⊢ Eq (HOrElse.hOrElse s fun x => (Computation.empty α).think) (HOrElse.hOrElse …
  -/
  rw [← think_empty]
  /-
    🎉 no goals
  -/


/-- `c₁ ~ c₂` asserts that `c₁` and `c₂` either both terminate with the same result,
  or both loop forever. -/
def Equiv (c₁ c₂ : Computation α) : Prop :=
  ∀ a, a ∈ c₁ ↔ a ∈ c₂


/-- equivalence relation for computations -/
scoped infixl:50 " ~ " => Equiv


@[refl]
theorem Equiv.refl (s : Computation α) : s ~ s := fun _ => Iff.rfl


@[symm]
theorem Equiv.symm {s t : Computation α} : s ~ t → t ~ s := fun h a => (h a).symm


@[trans]
theorem Equiv.trans {s t u : Computation α} : s ~ t → t ~ u → s ~ u := fun h1 h2 a =>
  (h1 a).trans (h2 a)


theorem Equiv.equivalence : Equivalence (@Equiv α) :=
  ⟨@Equiv.refl _, @Equiv.symm _, @Equiv.trans _⟩


theorem equiv_of_mem {s t : Computation α} {a} (h1 : a ∈ s) (h2 : a ∈ t) : s ~ t := fun a' =>
                /-
                  α : Type u
                  s t : Computation α
                  a : α
                  h1 : Membership.mem s a
                  h2 : Membership.mem t a
                  a' : α
                  ma : Membership.mem s a'
                  ⊢ Membership.mem t a'
                -/
                                       /-
                                         🎉 no goals
                                       -/
  ⟨fun ma => by rw [mem_unique ma h1]; exact h2, fun ma => by rw [mem_unique ma h2]; exact h1⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem terminates_congr {c₁ c₂ : Computation α} (h : c₁ ~ c₂) : Terminates c₁ ↔ Terminates c₂ := by
  /-
    α : Type u
    c₁ c₂ : Computation α
    h : c₁.Equiv c₂
    ⊢ Iff c₁.Terminates c₂.Terminates
  -/
  simp only [terminates_iff, exists_congr h]
  /-
    🎉 no goals
  -/


theorem promises_congr {c₁ c₂ : Computation α} (h : c₁ ~ c₂) (a) : c₁ ~> a ↔ c₂ ~> a :=
  forall_congr' fun a' => imp_congr (h a') Iff.rfl


theorem get_equiv {c₁ c₂ : Computation α} (h : c₁ ~ c₂) [Terminates c₁] [Terminates c₂] :
    get c₁ = get c₂ :=
  get_eq_of_mem _ <| (h _).2 <| get_mem _


theorem think_equiv (s : Computation α) : think s ~ s := fun _ => ⟨of_think_mem, think_mem⟩


theorem thinkN_equiv (s : Computation α) (n) : thinkN s n ~ s := fun _ => thinkN_mem n


theorem bind_congr {s1 s2 : Computation α} {f1 f2 : α → Computation β} (h1 : s1 ~ s2)
    (h2 : ∀ a, f1 a ~ f2 a) : bind s1 f1 ~ bind s2 f2 := fun b =>
  ⟨fun h =>
    let ⟨a, ha, hb⟩ := exists_of_mem_bind h
    mem_bind ((h1 a).1 ha) ((h2 a b).1 hb),
    fun h =>
    let ⟨a, ha, hb⟩ := exists_of_mem_bind h
    mem_bind ((h1 a).2 ha) ((h2 a b).2 hb)⟩


theorem equiv_pure_of_mem {s : Computation α} {a} (h : a ∈ s) : s ~ pure a :=
  equiv_of_mem h (ret_mem _)


/-- `LiftRel R ca cb` is a generalization of `Equiv` to relations other than
  equality. It asserts that if `ca` terminates with `a`, then `cb` terminates with
  some `b` such that `R a b`, and if `cb` terminates with `b` then `ca` terminates
  with some `a` such that `R a b`. -/
def LiftRel (R : α → β → Prop) (ca : Computation α) (cb : Computation β) : Prop :=
  (∀ {a}, a ∈ ca → ∃ b, b ∈ cb ∧ R a b) ∧ ∀ {b}, b ∈ cb → ∃ a, a ∈ ca ∧ R a b


theorem LiftRel.swap (R : α → β → Prop) (ca : Computation α) (cb : Computation β) :
    LiftRel (swap R) cb ca ↔ LiftRel R ca cb :=
  @and_comm _ _


theorem lift_eq_iff_equiv (c₁ c₂ : Computation α) : LiftRel (· = ·) c₁ c₂ ↔ c₁ ~ c₂ :=
  ⟨fun ⟨h1, h2⟩ a =>
                  /-
                    α : Type u
                    c₁ c₂ : Computation α
                    x✝ : Computation.LiftRel (fun x1 x2 => Eq x1 x2) c₁ c₂
                    a : α
                    h1 : ∀ {a : α}, Membership.mem c₁ a → Exists fun b => And (Membership.mem c₂ b …
                    h2 : ∀ {b : α}, Membership.mem c₂ b → Exists fun a => And (Membership.mem c₁ a …
                    a1 : Membership.mem c₁ a
                    ⊢ Membership.mem c₂ a
                  -/
    ⟨fun a1 => by let ⟨b, b2, ab⟩ := h1 a1; rwa [ab],
                                            /-
                                              🎉 no goals
                                            -/
                  /-
                    α : Type u
                    c₁ c₂ : Computation α
                    x✝ : Computation.LiftRel (fun x1 x2 => Eq x1 x2) c₁ c₂
                    a : α
                    h1 : ∀ {a : α}, Membership.mem c₁ a → Exists fun b => And (Membership.mem c₂ b …
                    h2 : ∀ {b : α}, Membership.mem c₂ b → Exists fun a => And (Membership.mem c₁ a …
                    a2 : Membership.mem c₂ a
                    ⊢ Membership.mem c₁ a
                  -/
     fun a2 => by let ⟨b, b1, ab⟩ := h2 a2; rwa [← ab]⟩,
                                            /-
                                              🎉 no goals
                                            -/
    fun e => ⟨fun {a} a1 => ⟨a, (e _).1 a1, rfl⟩, fun {a} a2 => ⟨a, (e _).2 a2, rfl⟩⟩⟩


theorem LiftRel.refl (R : α → α → Prop) (H : Reflexive R) : Reflexive (LiftRel R) := fun _ =>
  ⟨fun {a} as => ⟨a, as, H a⟩, fun {b} bs => ⟨b, bs, H b⟩⟩


theorem LiftRel.symm (R : α → α → Prop) (H : Symmetric R) : Symmetric (LiftRel R) :=
  fun _ _ ⟨l, r⟩ =>
  ⟨fun {_} a2 =>
    let ⟨b, b1, ab⟩ := r a2
    ⟨b, b1, H ab⟩,
    fun {_} a1 =>
    let ⟨b, b2, ab⟩ := l a1
    ⟨b, b2, H ab⟩⟩


theorem LiftRel.trans (R : α → α → Prop) (H : Transitive R) : Transitive (LiftRel R) :=
  fun _ _ _ ⟨l1, r1⟩ ⟨l2, r2⟩ =>
  ⟨fun {_} a1 =>
    let ⟨_, b2, ab⟩ := l1 a1
    let ⟨c, c3, bc⟩ := l2 b2
    ⟨c, c3, H ab bc⟩,
    fun {_} c3 =>
    let ⟨_, b2, bc⟩ := r2 c3
    let ⟨a, a1, ab⟩ := r1 b2
    ⟨a, a1, H ab bc⟩⟩


theorem LiftRel.equiv (R : α → α → Prop) : Equivalence R → Equivalence (LiftRel R)
                                                    /-
                                                      α : Type u
                                                      R : α → α → Prop
                                                      refl : ∀ (x : α), R x x
                                                      symm : ∀ {x y : α}, R x y → R y x
                                                      trans : ∀ {x y z : α}, R x y → R y z → R x z
                                                      ⊢ ∀ {x y : Computation α}, Computation.LiftRel R x y → Computation.LiftRel R y x
                                                    -/
  | ⟨refl, symm, trans⟩ => ⟨LiftRel.refl R refl, by apply LiftRel.symm; apply symm,
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
       /-
         α : Type u
         R : α → α → Prop
         refl : ∀ (x : α), R x x
         symm : ∀ {x y : α}, R x y → R y x
         trans : ∀ {x y z : α}, R x y → R y z → R x z
         ⊢ ∀ {x y z : Computation α}, Computation.LiftRel R x y → Computation.LiftRel R …
       -/
    by apply LiftRel.trans; apply trans⟩
                            /-
                              🎉 no goals
                            -/
  -- Porting note: The code above was:
  -- | ⟨refl, symm, trans⟩ => ⟨LiftRel.refl R refl, LiftRel.symm R symm, LiftRel.trans R trans⟩
  --
  -- The code fails to identify `symm` as being symmetric.


theorem LiftRel.imp {R S : α → β → Prop} (H : ∀ {a b}, R a b → S a b) (s t) :
    LiftRel R s t → LiftRel S s t
  | ⟨l, r⟩ =>
    ⟨fun {_} as =>
      let ⟨b, bt, ab⟩ := l as
      ⟨b, bt, H ab⟩,
      fun {_} bt =>
      let ⟨a, as, ab⟩ := r bt
      ⟨a, as, H ab⟩⟩


theorem terminates_of_liftRel {R : α → β → Prop} {s t} :
    LiftRel R s t → (Terminates s ↔ Terminates t)
  | ⟨l, r⟩ =>
    ⟨fun ⟨⟨_, as⟩⟩ =>
      let ⟨b, bt, _⟩ := l as
      ⟨⟨b, bt⟩⟩,
      fun ⟨⟨_, bt⟩⟩ =>
      let ⟨a, as, _⟩ := r bt
      ⟨⟨a, as⟩⟩⟩


theorem rel_of_liftRel {R : α → β → Prop} {ca cb} :
    LiftRel R ca cb → ∀ {a b}, a ∈ ca → b ∈ cb → R a b
  | ⟨l, _⟩, a, b, ma, mb => by
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      ca : Computation α
      cb : Computation β
      l : ∀ {a : α}, Membership.mem ca a → Exists fun b => And (Membership.mem cb b) …
      right✝ : ∀ {b : β}, Membership.mem cb b → Exists fun a => And (Membership.mem  …
      a : α
      b : β
      ma : Membership.mem ca a
      mb : Membership.mem cb b
      ⊢ R a b
    -/
    let ⟨b', mb', ab'⟩ := l ma
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      ca : Computation α
      cb : Computation β
      l : ∀ {a : α}, Membership.mem ca a → Exists fun b => And (Membership.mem cb b) …
      right✝ : ∀ {b : β}, Membership.mem cb b → Exists fun a => And (Membership.mem  …
      a : α
      b : β
      ma : Membership.mem ca a
      mb : Membership.mem cb b
      b' : β
      mb' : Membership.mem cb b'
      ab' : R a b'
      ⊢ R a b
    -/
    rw [mem_unique mb mb']; exact ab'
                            /-
                              🎉 no goals
                            -/


theorem liftRel_of_mem {R : α → β → Prop} {a b ca cb} (ma : a ∈ ca) (mb : b ∈ cb) (ab : R a b) :
    LiftRel R ca cb :=
                      /-
                        α : Type u
                        β : Type v
                        R : α → β → Prop
                        a : α
                        b : β
                        ca : Computation α
                        cb : Computation β
                        ma : Membership.mem ca a
                        mb : Membership.mem cb b
                        ab : R a b
                        a' : α
                        ma' : Membership.mem ca a'
                        ⊢ Exists fun b => And (Membership.mem cb b) (R a' b)
                      -/
  ⟨fun {a'} ma' => by rw [mem_unique ma' ma]; exact ⟨b, mb, ab⟩, fun {b'} mb' => by
                                              /-
                                                🎉 no goals
                                              -/
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      a : α
      b : β
      ca : Computation α
      cb : Computation β
      ma : Membership.mem ca a
      mb : Membership.mem cb b
      ab : R a b
      b' : β
      mb' : Membership.mem cb b'
      ⊢ Exists fun a => And (Membership.mem ca a) (R a b')
    -/
    rw [mem_unique mb' mb]; exact ⟨a, ma, ab⟩⟩
                            /-
                              🎉 no goals
                            -/


theorem exists_of_liftRel_left {R : α → β → Prop} {ca cb} (H : LiftRel R ca cb) {a} (h : a ∈ ca) :
    ∃ b, b ∈ cb ∧ R a b :=
  H.left h


theorem exists_of_liftRel_right {R : α → β → Prop} {ca cb} (H : LiftRel R ca cb) {b} (h : b ∈ cb) :
    ∃ a, a ∈ ca ∧ R a b :=
  H.right h


theorem liftRel_def {R : α → β → Prop} {ca cb} :
    LiftRel R ca cb ↔ (Terminates ca ↔ Terminates cb) ∧ ∀ {a b}, a ∈ ca → b ∈ cb → R a b :=
  ⟨fun h =>
    ⟨terminates_of_liftRel h, fun {a b} ma mb => by
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        ca : Computation α
        cb : Computation β
        h : Computation.LiftRel R ca cb
        a : α
        b : β
        ma : Membership.mem ca a
        mb : Membership.mem cb b
        ⊢ R a b
      -/
      let ⟨b', mb', ab⟩ := h.left ma
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        ca : Computation α
        cb : Computation β
        h : Computation.LiftRel R ca cb
        a : α
        b : β
        ma : Membership.mem ca a
        mb : Membership.mem cb b
        b' : β
        mb' : Membership.mem cb b'
        ab : R a b'
        ⊢ R a b
      -/
      rwa [mem_unique mb mb']⟩,
      /-
        🎉 no goals
      -/
    fun ⟨l, r⟩ =>
    ⟨fun {_} ma =>
      let ⟨⟨b, mb⟩⟩ := l.1 ⟨⟨_, ma⟩⟩
      ⟨b, mb, r ma mb⟩,
      fun {_} mb =>
      let ⟨⟨a, ma⟩⟩ := l.2 ⟨⟨_, mb⟩⟩
      ⟨a, ma, r ma mb⟩⟩⟩


theorem liftRel_bind {δ} (R : α → β → Prop) (S : γ → δ → Prop) {s1 : Computation α}
    {s2 : Computation β} {f1 : α → Computation γ} {f2 : β → Computation δ} (h1 : LiftRel R s1 s2)
    (h2 : ∀ {a b}, R a b → LiftRel S (f1 a) (f2 b)) : LiftRel S (bind s1 f1) (bind s2 f2) :=
  let ⟨l1, r1⟩ := h1
  ⟨fun {_} cB =>
    let ⟨_, a1, c₁⟩ := exists_of_mem_bind cB
    let ⟨_, b2, ab⟩ := l1 a1
    let ⟨l2, _⟩ := h2 ab
    let ⟨_, d2, cd⟩ := l2 c₁
    ⟨_, mem_bind b2 d2, cd⟩,
    fun {_} dB =>
    let ⟨_, b1, d1⟩ := exists_of_mem_bind dB
    let ⟨_, a2, ab⟩ := r1 b1
    let ⟨_, r2⟩ := h2 ab
    let ⟨_, c₂, cd⟩ := r2 d1
    ⟨_, mem_bind a2 c₂, cd⟩⟩


@[simp]
theorem liftRel_pure_left (R : α → β → Prop) (a : α) (cb : Computation β) :
    LiftRel R (pure a) cb ↔ ∃ b, b ∈ cb ∧ R a b :=
  ⟨fun ⟨l, _⟩ => l (ret_mem _), fun ⟨b, mb, ab⟩ =>
                        /-
                          α : Type u
                          β : Type v
                          R : α → β → Prop
                          a : α
                          cb : Computation β
                          x✝ : Exists fun b => And (Membership.mem cb b) (R a b)
                          b : β
                          mb : Membership.mem cb b
                          ab : R a b
                          a' : α
                          ma' : Membership.mem (Computation.pure a) a'
                          ⊢ Exists fun b => And (Membership.mem cb b) (R a' b)
                        -/
    ⟨fun {a'} ma' => by rw [eq_of_pure_mem ma']; exact ⟨b, mb, ab⟩, fun {b'} mb' =>
                                                 /-
                                                   🎉 no goals
                                                 -/
                        /-
                          α : Type u
                          β : Type v
                          R : α → β → Prop
                          a : α
                          cb : Computation β
                          x✝ : Exists fun b => And (Membership.mem cb b) (R a b)
                          b : β
                          mb : Membership.mem cb b
                          ab : R a b
                          b' : β
                          mb' : Membership.mem cb b'
                          ⊢ R a b'
                        -/
      ⟨_, ret_mem _, by rw [mem_unique mb' mb]; exact ab⟩⟩⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem liftRel_pure_right (R : α → β → Prop) (ca : Computation α) (b : β) :
                                                      /-
                                                        α : Type u
                                                        β : Type v
                                                        R : α → β → Prop
                                                        ca : Computation α
                                                        b : β
                                                        ⊢ Iff (Computation.LiftRel R ca (Computation.pure b)) (Exists fun a => And (Me …
                                                      -/
    LiftRel R ca (pure b) ↔ ∃ a, a ∈ ca ∧ R a b := by rw [LiftRel.swap, liftRel_pure_left]
                                                      /-
                                                        🎉 no goals
                                                      -/

-- Porting note: `simpNF` wants to simplify based on `liftRel_pure_right` but point is to prove
-- a general invariant on `LiftRel`

@[simp, nolint simpNF]
theorem liftRel_pure (R : α → β → Prop) (a : α) (b : β) :
    LiftRel R (pure a) (pure b) ↔ R a b := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    a : α
    b : β
    ⊢ Iff (Computation.LiftRel R (Computation.pure a) (Computation.pure b)) (R a b)
  -/
  rw [liftRel_pure_left]
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    a : α
    b : β
    ⊢ Iff (Exists fun b_1 => And (Membership.mem (Computation.pure b) b_1) (R a b_ …
  -/
  exact ⟨fun ⟨b', mb', ab'⟩ => by rwa [eq_of_pure_mem mb'] at ab', fun ab => ⟨_, ret_mem _, ab⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem liftRel_think_left (R : α → β → Prop) (ca : Computation α) (cb : Computation β) :
    LiftRel R (think ca) cb ↔ LiftRel R ca cb :=
  and_congr (forall_congr' fun _ => imp_congr ⟨of_think_mem, think_mem⟩ Iff.rfl)
    (forall_congr' fun _ =>
      imp_congr Iff.rfl <| exists_congr fun _ => and_congr ⟨of_think_mem, think_mem⟩ Iff.rfl)


@[simp]
theorem liftRel_think_right (R : α → β → Prop) (ca : Computation α) (cb : Computation β) :
    LiftRel R ca (think cb) ↔ LiftRel R ca cb := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    ca : Computation α
    cb : Computation β
    ⊢ Iff (Computation.LiftRel R ca cb.think) (Computation.LiftRel R ca cb)
  -/
  rw [← LiftRel.swap R, ← LiftRel.swap R]; apply liftRel_think_left
                                           /-
                                             🎉 no goals
                                           -/


theorem liftRel_mem_cases {R : α → β → Prop} {ca cb} (Ha : ∀ a ∈ ca, LiftRel R ca cb)
    (Hb : ∀ b ∈ cb, LiftRel R ca cb) : LiftRel R ca cb :=
  ⟨fun {_} ma => (Ha _ ma).left ma, fun {_} mb => (Hb _ mb).right mb⟩


theorem liftRel_congr {R : α → β → Prop} {ca ca' : Computation α} {cb cb' : Computation β}
    (ha : ca ~ ca') (hb : cb ~ cb') : LiftRel R ca cb ↔ LiftRel R ca' cb' :=
  and_congr
    (forall_congr' fun _ => imp_congr (ha _) <| exists_congr fun _ => and_congr (hb _) Iff.rfl)
    (forall_congr' fun _ => imp_congr (hb _) <| exists_congr fun _ => and_congr (ha _) Iff.rfl)


theorem liftRel_map {δ} (R : α → β → Prop) (S : γ → δ → Prop) {s1 : Computation α}
    {s2 : Computation β} {f1 : α → γ} {f2 : β → δ} (h1 : LiftRel R s1 s2)
    (h2 : ∀ {a b}, R a b → S (f1 a) (f2 b)) : LiftRel S (map f1 s1) (map f2 s2) := by
  -- Porting note: The line below was:
  -- rw [← bind_pure, ← bind_pure]; apply lift_rel_bind _ _ h1; simp; exact @h2
  --
  -- The code fails to work on the last exact.
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_1
    R : α → β → Prop
    S : γ → δ → Prop
    s1 : Computation α
    s2 : Computation β
    f1 : α → γ
    f2 : β → δ
    h1 : Computation.LiftRel R s1 s2
    h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
    ⊢ Computation.LiftRel S (Computation.map f1 s1) (Computation.map f2 s2)
  -/
  rw [← bind_pure, ← bind_pure]; apply liftRel_bind _ _ h1
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_1
    R : α → β → Prop
    S : γ → δ → Prop
    s1 : Computation α
    s2 : Computation β
    f1 : α → γ
    f2 : β → δ
    h1 : Computation.LiftRel R s1 s2
    h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
    ⊢ ∀ {a : α} {b : β}, R a b → Computation.LiftRel S (Function.comp Computation. …
  -/
  simp only [comp_apply, liftRel_pure_right]
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_1
    R : α → β → Prop
    S : γ → δ → Prop
    s1 : Computation α
    s2 : Computation β
    f1 : α → γ
    f2 : β → δ
    h1 : Computation.LiftRel R s1 s2
    h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
    ⊢ ∀ {a : α} {b : β}, R a b → Exists fun a_2 => And (Membership.mem (Computatio …
  -/
  intros a b h; exact ⟨f1 a, ⟨ret_mem _, @h2 a b h⟩⟩
                /-
                  🎉 no goals
                -/

-- Porting note: deleted initial arguments `(_R : α → α → Prop) (_S : β → β → Prop)`: unused

theorem map_congr {s1 s2 : Computation α} {f : α → β}
    (h1 : s1 ~ s2) : map f s1 ~ map f s2 := by
  /-
    α : Type u
    β : Type v
    s1 s2 : Computation α
    f : α → β
    h1 : s1.Equiv s2
    ⊢ (Computation.map f s1).Equiv (Computation.map f s2)
  -/
  rw [← lift_eq_iff_equiv]
  /-
    α : Type u
    β : Type v
    s1 s2 : Computation α
    f : α → β
    h1 : s1.Equiv s2
    ⊢ Computation.LiftRel (fun x1 x2 => Eq x1 x2) (Computation.map f s1) (Computat …
  -/
  exact liftRel_map Eq _ ((lift_eq_iff_equiv _ _).2 h1) fun {a} b => congr_arg _
  /-
    🎉 no goals
  -/


/-- Alternate definition of `LiftRel` over relations between `Computation`s -/
def LiftRelAux (R : α → β → Prop) (C : Computation α → Computation β → Prop) :
    α ⊕ (Computation α) → β ⊕ (Computation β) → Prop
  | Sum.inl a, Sum.inl b => R a b
  | Sum.inl a, Sum.inr cb => ∃ b, b ∈ cb ∧ R a b
  | Sum.inr ca, Sum.inl b => ∃ a, a ∈ ca ∧ R a b
  | Sum.inr ca, Sum.inr cb => C ca cb


@[simp] lemma liftRelAux_inl_inl {a : α} {b : β} :
  LiftRelAux R C (Sum.inl a) (Sum.inl b) = R a b := rfl

@[simp] lemma liftRelAux_inl_inr {a : α} {cb} :
    LiftRelAux R C (Sum.inl a) (Sum.inr cb) = ∃ b, b ∈ cb ∧ R a b :=
  rfl

@[simp] lemma liftRelAux_inr_inl {b : β} {ca} :
    LiftRelAux R C (Sum.inr ca) (Sum.inl b) = ∃ a, a ∈ ca ∧ R a b :=
  rfl

@[simp] lemma liftRelAux_inr_inr {ca cb} :
    LiftRelAux R C (Sum.inr ca) (Sum.inr cb) = C ca cb :=
  rfl


@[simp]
theorem LiftRelAux.ret_left (R : α → β → Prop) (C : Computation α → Computation β → Prop) (a cb) :
    LiftRelAux R C (Sum.inl a) (destruct cb) ↔ ∃ b, b ∈ cb ∧ R a b := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    C : Computation α → Computation β → Prop
    a : α
    cb : Computation β
    ⊢ Iff (Computation.LiftRelAux R C (Sum.inl a) cb.destruct) (Exists fun b => An …
  -/
  apply cb.recOn (fun b => _) fun cb => _
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      a : α
      cb : Computation β
      ⊢ ∀ (b : β), Iff (Computation.LiftRelAux R C (Sum.inl a) (Computation.pure b). …
    -/
  · intro b
    exact
      ⟨fun h => ⟨_, ret_mem _, h⟩, fun ⟨b', mb, h⟩ => by rw [mem_unique (ret_mem _) mb]; exact h⟩
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      a : α
      cb : Computation β
      ⊢ ∀ (cb : Computation β), Iff (Computation.LiftRelAux R C (Sum.inl a) cb.think …
    -/
  · intro
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      a : α
      cb cb✝ : Computation β
      ⊢ Iff (Computation.LiftRelAux R C (Sum.inl a) cb✝.think.destruct) (Exists fun  …
    -/
    rw [destruct_think]
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      a : α
      cb cb✝ : Computation β
      ⊢ Iff (Computation.LiftRelAux R C (Sum.inl a) (Sum.inr cb✝)) (Exists fun b =>  …
    -/
    exact ⟨fun ⟨b, h, r⟩ => ⟨b, think_mem h, r⟩, fun ⟨b, h, r⟩ => ⟨b, of_think_mem h, r⟩⟩
    /-
      🎉 no goals
    -/


theorem LiftRelAux.swap (R : α → β → Prop) (C) (a b) :
    LiftRelAux (swap R) (swap C) b a = LiftRelAux R C a b := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    C : Computation α → Computation β → Prop
    a : Sum α (Computation α)
    b : Sum β (Computation β)
    ⊢ Eq (Computation.LiftRelAux (Function.swap R) (Function.swap C) b a) (Computa …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  cases' a with a ca <;> cases' b with b cb <;> simp only [LiftRelAux]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem LiftRelAux.ret_right (R : α → β → Prop) (C : Computation α → Computation β → Prop) (b ca) :
    LiftRelAux R C (destruct ca) (Sum.inl b) ↔ ∃ a, a ∈ ca ∧ R a b := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    C : Computation α → Computation β → Prop
    b : β
    ca : Computation α
    ⊢ Iff (Computation.LiftRelAux R C ca.destruct (Sum.inl b)) (Exists fun a => An …
  -/
  rw [← LiftRelAux.swap, LiftRelAux.ret_left]
  /-
    🎉 no goals
  -/


theorem LiftRelRec.lem {R : α → β → Prop} (C : Computation α → Computation β → Prop)
    (H : ∀ {ca cb}, C ca cb → LiftRelAux R C (destruct ca) (destruct cb)) (ca cb) (Hc : C ca cb) (a)
    (ha : a ∈ ca) : LiftRel R ca cb := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    C : Computation α → Computation β → Prop
    H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
    ca : Computation α
    cb : Computation β
    Hc : C ca cb
    a : α
    ha : Membership.mem ca a
    ⊢ Computation.LiftRel R ca cb
  -/
  revert cb
  refine memRecOn (C := (fun ca ↦ ∀ (cb : Computation β), C ca cb → LiftRel R ca cb))
                                 /-
                                   case refine_1
                                   α : Type u
                                   β : Type v
                                   R : α → β → Prop
                                   C : Computation α → Computation β → Prop
                                   H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
                                   ca : Computation α
                                   a : α
                                   ha : Membership.mem ca a
                                   ⊢ (fun ca => ∀ (cb : Computation β), C ca cb → Computation.LiftRel R ca cb) (C …
                                 -/
    ha ?_ (fun ca' IH => ?_) <;> intro cb Hc <;> have h := H Hc
    /-
      case refine_1
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
      ca : Computation α
      a : α
      ha : Membership.mem ca a
      cb : Computation β
      Hc : C (Computation.pure a) cb
      h : Computation.LiftRelAux R C (Computation.pure a).destruct cb.destruct
      ⊢ Computation.LiftRel R (Computation.pure a) cb
    -/
  · simp only [destruct_pure, LiftRelAux.ret_left] at h
    /-
      case refine_1
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
      ca : Computation α
      a : α
      ha : Membership.mem ca a
      cb : Computation β
      Hc : C (Computation.pure a) cb
      h : Exists fun b => And (Membership.mem cb b) (R a b)
      ⊢ Computation.LiftRel R (Computation.pure a) cb
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
      ca : Computation α
      a : α
      ha : Membership.mem ca a
      ca' : Computation α
      IH : (fun ca => ∀ (cb : Computation β), C ca cb → Computation.LiftRel R ca cb) …
      cb : Computation β
      Hc : C ca'.think cb
      h : Computation.LiftRelAux R C ca'.think.destruct cb.destruct
      ⊢ Computation.LiftRel R ca'.think cb
    -/
  · simp only [liftRel_think_left]
    /-
      case refine_2
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
      ca : Computation α
      a : α
      ha : Membership.mem ca a
      ca' : Computation α
      IH : (fun ca => ∀ (cb : Computation β), C ca cb → Computation.LiftRel R ca cb) …
      cb : Computation β
      Hc : C ca'.think cb
      h : Computation.LiftRelAux R C ca'.think.destruct cb.destruct
      ⊢ Computation.LiftRel R ca' cb
    -/
    revert h
    /-
      case refine_2
      α : Type u
      β : Type v
      R : α → β → Prop
      C : Computation α → Computation β → Prop
      H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
      ca : Computation α
      a : α
      ha : Membership.mem ca a
      ca' : Computation α
      IH : (fun ca => ∀ (cb : Computation β), C ca cb → Computation.LiftRel R ca cb) …
      cb : Computation β
      Hc : C ca'.think cb
      ⊢ Computation.LiftRelAux R C ca'.think.destruct cb.destruct → Computation.Lift …
    -/
    apply cb.recOn (fun b => _) fun cb' => _ <;> intros _ h
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        C : Computation α → Computation β → Prop
        H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
        ca : Computation α
        a : α
        ha : Membership.mem ca a
        ca' : Computation α
        IH : (fun ca => ∀ (cb : Computation β), C ca cb → Computation.LiftRel R ca cb) …
        cb : Computation β
        Hc : C ca'.think cb
        b✝ : β
        h : Computation.LiftRelAux R C ca'.think.destruct (Computation.pure b✝).destruct
        ⊢ Computation.LiftRel R ca' (Computation.pure b✝)
      -/
    · simpa using h
      /-
        🎉 no goals
      -/
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        C : Computation α → Computation β → Prop
        H : ∀ {ca : Computation α} {cb : Computation β}, C ca cb → Computation.LiftRel …
        ca : Computation α
        a : α
        ha : Membership.mem ca a
        ca' : Computation α
        IH : (fun ca => ∀ (cb : Computation β), C ca cb → Computation.LiftRel R ca cb) …
        cb : Computation β
        Hc : C ca'.think cb
        cb'✝ : Computation β
        h : Computation.LiftRelAux R C ca'.think.destruct cb'✝.think.destruct
        ⊢ Computation.LiftRel R ca' cb'✝.think
      -/
    · simpa [h] using IH _ h
      /-
        🎉 no goals
      -/


theorem liftRel_rec {R : α → β → Prop} (C : Computation α → Computation β → Prop)
    (H : ∀ {ca cb}, C ca cb → LiftRelAux R C (destruct ca) (destruct cb)) (ca cb) (Hc : C ca cb) :
    LiftRel R ca cb :=
  liftRel_mem_cases (LiftRelRec.lem C (@H) ca cb Hc) fun b hb =>
    (LiftRel.swap _ _ _).2 <|
      LiftRelRec.lem (swap C) (fun {_ _} h => cast (LiftRelAux.swap _ _ _ _).symm <| H h) cb ca Hc b
        hb


