/-- Weak sequences.

  While the `Seq` structure allows for lists which may not be finite,
  a weak sequence also allows the computation of each element to
  involve an indeterminate amount of computation, including possibly
  an infinite loop. This is represented as a regular `Seq` interspersed
  with `none` elements to indicate that computation is ongoing.

  This model is appropriate for Haskell style lazy lists, and is closed
  under most interesting computation patterns on infinite lists,
  but conversely it is difficult to extract elements from it. -/
def WSeq (α) :=
  Seq (Option α)

/-
coinductive WSeq (α : Type u) : Type u
| nil : WSeq α
| cons : α → WSeq α → WSeq α
| think : WSeq α → WSeq α
-/


/-- Turn a sequence into a weak sequence -/
@[coe]
def ofSeq : Seq α → WSeq α :=
  (· <$> ·) some


/-- Turn a list into a weak sequence -/
@[coe]
def ofList (l : List α) : WSeq α :=
  ofSeq l


/-- Turn a stream into a weak sequence -/
@[coe]
def ofStream (l : Stream' α) : WSeq α :=
  ofSeq l


instance coeSeq : Coe (Seq α) (WSeq α) :=
  ⟨ofSeq⟩


instance coeList : Coe (List α) (WSeq α) :=
  ⟨ofList⟩


instance coeStream : Coe (Stream' α) (WSeq α) :=
  ⟨ofStream⟩


/-- The empty weak sequence -/
def nil : WSeq α :=
  Seq.nil


instance inhabited : Inhabited (WSeq α) :=
  ⟨nil⟩


/-- Prepend an element to a weak sequence -/
def cons (a : α) : WSeq α → WSeq α :=
  Seq.cons (some a)


/-- Compute for one tick, without producing any elements -/
def think : WSeq α → WSeq α :=
  Seq.cons none


/-- Destruct a weak sequence, to (eventually possibly) produce either
  `none` for `nil` or `some (a, s)` if an element is produced. -/
def destruct : WSeq α → Computation (Option (α × WSeq α)) :=
  Computation.corec fun s =>
    match Seq.destruct s with
    | none => Sum.inl none
    | some (none, s') => Sum.inr s'
    | some (some a, s') => Sum.inl (some (a, s'))


/-- Recursion principle for weak sequences, compare with `List.recOn`. -/
def recOn {C : WSeq α → Sort v} (s : WSeq α) (h1 : C nil) (h2 : ∀ x s, C (cons x s))
    (h3 : ∀ s, C (think s)) : C s :=
  Seq.recOn s h1 fun o => Option.recOn o h3 h2


/-- membership for weak sequences-/
protected def Mem (s : WSeq α) (a : α) :=
  Seq.Mem s (some a)


instance membership : Membership α (WSeq α) :=
  ⟨WSeq.Mem⟩


theorem not_mem_nil (a : α) : a ∉ @nil α :=
  Seq.not_mem_nil (some a)


/-- Get the head of a weak sequence. This involves a possibly
  infinite computation. -/
def head (s : WSeq α) : Computation (Option α) :=
  Computation.map (Prod.fst <$> ·) (destruct s)


/-- Encode a computation yielding a weak sequence into additional
  `think` constructors in a weak sequence -/
def flatten : Computation (WSeq α) → WSeq α :=
  Seq.corec fun c =>
    match Computation.destruct c with
    | Sum.inl s => Seq.omap (return ·) (Seq.destruct s)
    | Sum.inr c' => some (none, c')


/-- Get the tail of a weak sequence. This doesn't need a `Computation`
  wrapper, unlike `head`, because `flatten` allows us to hide this
  in the construction of the weak sequence itself. -/
def tail (s : WSeq α) : WSeq α :=
  flatten <| (fun o => Option.recOn o nil Prod.snd) <$> destruct s


/-- drop the first `n` elements from `s`. -/
def drop (s : WSeq α) : ℕ → WSeq α
  | 0 => s
  | n + 1 => tail (drop s n)


/-- Get the nth element of `s`. -/
def get? (s : WSeq α) (n : ℕ) : Computation (Option α) :=
  head (drop s n)


/-- Convert `s` to a list (if it is finite and completes in finite time). -/
def toList (s : WSeq α) : Computation (List α) :=
  @Computation.corec (List α) (List α × WSeq α)
    (fun ⟨l, s⟩ =>
      match Seq.destruct s with
      | none => Sum.inl l.reverse
      | some (none, s') => Sum.inr (l, s')
      | some (some a, s') => Sum.inr (a::l, s'))
    ([], s)


/-- Get the length of `s` (if it is finite and completes in finite time). -/
def length (s : WSeq α) : Computation ℕ :=
  @Computation.corec ℕ (ℕ × WSeq α)
    (fun ⟨n, s⟩ =>
      match Seq.destruct s with
      | none => Sum.inl n
      | some (none, s') => Sum.inr (n, s')
      | some (some _, s') => Sum.inr (n + 1, s'))
    (0, s)


/-- A weak sequence is finite if `toList s` terminates. Equivalently,
  it is a finite number of `think` and `cons` applied to `nil`. -/
class IsFinite (s : WSeq α) : Prop where
  out : (toList s).Terminates


instance toList_terminates (s : WSeq α) [h : IsFinite s] : (toList s).Terminates :=
  h.out


/-- Get the list corresponding to a finite weak sequence. -/
def get (s : WSeq α) [IsFinite s] : List α :=
  (toList s).get


/-- A weak sequence is *productive* if it never stalls forever - there are
 always a finite number of `think`s between `cons` constructors.
 The sequence itself is allowed to be infinite though. -/
class Productive (s : WSeq α) : Prop where
  get?_terminates : ∀ n, (get? s n).Terminates


theorem productive_iff (s : WSeq α) : Productive s ↔ ∀ n, (get? s n).Terminates :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


instance get?_terminates (s : WSeq α) [h : Productive s] : ∀ n, (get? s n).Terminates :=
  h.get?_terminates


instance head_terminates (s : WSeq α) [Productive s] : (head s).Terminates :=
  s.get?_terminates 0


/-- Replace the `n`th element of `s` with `a`. -/
def updateNth (s : WSeq α) (n : ℕ) (a : α) : WSeq α :=
  @Seq.corec (Option α) (ℕ × WSeq α)
    (fun ⟨n, s⟩ =>
      match Seq.destruct s, n with
      | none, _ => none
      | some (none, s'), n => some (none, n, s')
      | some (some a', s'), 0 => some (some a', 0, s')
      | some (some _, s'), 1 => some (some a, 0, s')
      | some (some a', s'), n + 2 => some (some a', n + 1, s'))
    (n + 1, s)


/-- Remove the `n`th element of `s`. -/
def removeNth (s : WSeq α) (n : ℕ) : WSeq α :=
  @Seq.corec (Option α) (ℕ × WSeq α)
    (fun ⟨n, s⟩ =>
      match Seq.destruct s, n with
      | none, _ => none
      | some (none, s'), n => some (none, n, s')
      | some (some a', s'), 0 => some (some a', 0, s')
      | some (some _, s'), 1 => some (none, 0, s')
      | some (some a', s'), n + 2 => some (some a', n + 1, s'))
    (n + 1, s)


/-- Map the elements of `s` over `f`, removing any values that yield `none`. -/
def filterMap (f : α → Option β) : WSeq α → WSeq β :=
  Seq.corec fun s =>
    match Seq.destruct s with
    | none => none
    | some (none, s') => some (none, s')
    | some (some a, s') => some (f a, s')


/-- Select the elements of `s` that satisfy `p`. -/
def filter (p : α → Prop) [DecidablePred p] : WSeq α → WSeq α :=
  filterMap fun a => if p a then some a else none

-- example of infinite list manipulations

/-- Get the first element of `s` satisfying `p`. -/
def find (p : α → Prop) [DecidablePred p] (s : WSeq α) : Computation (Option α) :=
  head <| filter p s


/-- Zip a function over two weak sequences -/
def zipWith (f : α → β → γ) (s1 : WSeq α) (s2 : WSeq β) : WSeq γ :=
  @Seq.corec (Option γ) (WSeq α × WSeq β)
    (fun ⟨s1, s2⟩ =>
      match Seq.destruct s1, Seq.destruct s2 with
      | some (none, s1'), some (none, s2') => some (none, s1', s2')
      | some (some _, _), some (none, s2') => some (none, s1, s2')
      | some (none, s1'), some (some _, _) => some (none, s1', s2)
      | some (some a1, s1'), some (some a2, s2') => some (some (f a1 a2), s1', s2')
      | _, _ => none)
    (s1, s2)


/-- Zip two weak sequences into a single sequence of pairs -/
def zip : WSeq α → WSeq β → WSeq (α × β) :=
  zipWith Prod.mk


/-- Get the list of indexes of elements of `s` satisfying `p` -/
def findIndexes (p : α → Prop) [DecidablePred p] (s : WSeq α) : WSeq ℕ :=
  (zip s (Stream'.nats : WSeq ℕ)).filterMap fun ⟨a, n⟩ => if p a then some n else none


/-- Get the index of the first element of `s` satisfying `p` -/
def findIndex (p : α → Prop) [DecidablePred p] (s : WSeq α) : Computation ℕ :=
  (fun o => Option.getD o 0) <$> head (findIndexes p s)


/-- Get the index of the first occurrence of `a` in `s` -/
def indexOf [DecidableEq α] (a : α) : WSeq α → Computation ℕ :=
  findIndex (Eq a)


/-- Get the indexes of occurrences of `a` in `s` -/
def indexesOf [DecidableEq α] (a : α) : WSeq α → WSeq ℕ :=
  findIndexes (Eq a)


/-- `union s1 s2` is a weak sequence which interleaves `s1` and `s2` in
  some order (nondeterministically). -/
def union (s1 s2 : WSeq α) : WSeq α :=
  @Seq.corec (Option α) (WSeq α × WSeq α)
    (fun ⟨s1, s2⟩ =>
      match Seq.destruct s1, Seq.destruct s2 with
      | none, none => none
      | some (a1, s1'), none => some (a1, s1', nil)
      | none, some (a2, s2') => some (a2, nil, s2')
      | some (none, s1'), some (none, s2') => some (none, s1', s2')
      | some (some a1, s1'), some (none, s2') => some (some a1, s1', s2')
      | some (none, s1'), some (some a2, s2') => some (some a2, s1', s2')
      | some (some a1, s1'), some (some a2, s2') => some (some a1, cons a2 s1', s2'))
    (s1, s2)


/-- Returns `true` if `s` is `nil` and `false` if `s` has an element -/
def isEmpty (s : WSeq α) : Computation Bool :=
  Computation.map Option.isNone <| head s


/-- Calculate one step of computation -/
def compute (s : WSeq α) : WSeq α :=
  match Seq.destruct s with
  | some (none, s') => s'
  | _ => s


/-- Get the first `n` elements of a weak sequence -/
def take (s : WSeq α) (n : ℕ) : WSeq α :=
  @Seq.corec (Option α) (ℕ × WSeq α)
    (fun ⟨n, s⟩ =>
      match n, Seq.destruct s with
      | 0, _ => none
      | _ + 1, none => none
      | m + 1, some (none, s') => some (none, m + 1, s')
      | m + 1, some (some a, s') => some (some a, m, s'))
    (n, s)


/-- Split the sequence at position `n` into a finite initial segment
  and the weak sequence tail -/
def splitAt (s : WSeq α) (n : ℕ) : Computation (List α × WSeq α) :=
  @Computation.corec (List α × WSeq α) (ℕ × List α × WSeq α)
    (fun ⟨n, l, s⟩ =>
      match n, Seq.destruct s with
      | 0, _ => Sum.inl (l.reverse, s)
      | _ + 1, none => Sum.inl (l.reverse, s)
      | _ + 1, some (none, s') => Sum.inr (n, l, s')
      | m + 1, some (some a, s') => Sum.inr (m, a::l, s'))
    (n, [], s)


/-- Returns `true` if any element of `s` satisfies `p` -/
def any (s : WSeq α) (p : α → Bool) : Computation Bool :=
  Computation.corec
    (fun s : WSeq α =>
      match Seq.destruct s with
      | none => Sum.inl false
      | some (none, s') => Sum.inr s'
      | some (some a, s') => if p a then Sum.inl true else Sum.inr s')
    s


/-- Returns `true` if every element of `s` satisfies `p` -/
def all (s : WSeq α) (p : α → Bool) : Computation Bool :=
  Computation.corec
    (fun s : WSeq α =>
      match Seq.destruct s with
      | none => Sum.inl true
      | some (none, s') => Sum.inr s'
      | some (some a, s') => if p a then Sum.inr s' else Sum.inl false)
    s


/-- Apply a function to the elements of the sequence to produce a sequence
  of partial results. (There is no `scanr` because this would require
  working from the end of the sequence, which may not exist.) -/
def scanl (f : α → β → α) (a : α) (s : WSeq β) : WSeq α :=
  cons a <|
    @Seq.corec (Option α) (α × WSeq β)
      (fun ⟨a, s⟩ =>
        match Seq.destruct s with
        | none => none
        | some (none, s') => some (none, a, s')
        | some (some b, s') =>
          let a' := f a b
          some (some a', a', s'))
      (a, s)


/-- Get the weak sequence of initial segments of the input sequence -/
def inits (s : WSeq α) : WSeq (List α) :=
  cons [] <|
    @Seq.corec (Option (List α)) (Batteries.DList α × WSeq α)
      (fun ⟨l, s⟩ =>
        match Seq.destruct s with
        | none => none
        | some (none, s') => some (none, l, s')
        | some (some a, s') =>
          let l' := l.push a
          some (some l'.toList, l', s'))
      (Batteries.DList.empty, s)


/-- Like take, but does not wait for a result. Calculates `n` steps of
  computation and returns the sequence computed so far -/
def collect (s : WSeq α) (n : ℕ) : List α :=
  (Seq.take n s).filterMap id


/-- Append two weak sequences. As with `Seq.append`, this may not use
  the second sequence if the first one takes forever to compute -/
def append : WSeq α → WSeq α → WSeq α :=
  Seq.append


/-- Map a function over a weak sequence -/
def map (f : α → β) : WSeq α → WSeq β :=
  Seq.map (Option.map f)


/-- Flatten a sequence of weak sequences. (Note that this allows
  empty sequences, unlike `Seq.join`.) -/
def join (S : WSeq (WSeq α)) : WSeq α :=
  Seq.join
    ((fun o : Option (WSeq α) =>
        match o with
        | none => Seq1.ret none
        | some s => (none, s)) <$>
      S)


/-- Monadic bind operator for weak sequences -/
def bind (s : WSeq α) (f : α → WSeq β) : WSeq β :=
  join (map f s)


/-- lift a relation to a relation over weak sequences -/
@[simp]
def LiftRelO (R : α → β → Prop) (C : WSeq α → WSeq β → Prop) :
    Option (α × WSeq α) → Option (β × WSeq β) → Prop
  | none, none => True
  | some (a, s), some (b, t) => R a b ∧ C s t
  | _, _ => False

theorem LiftRelO.imp {R S : α → β → Prop} {C D : WSeq α → WSeq β → Prop} (H1 : ∀ a b, R a b → S a b)
    (H2 : ∀ s t, C s t → D s t) : ∀ {o p}, LiftRelO R C o p → LiftRelO S D o p
  | none, none, _ => trivial
  | some (_, _), some (_, _), h => And.imp (H1 _ _) (H2 _ _) h
  | none, some _, h => False.elim h
  | some (_, _), none, h => False.elim h


theorem LiftRelO.imp_right (R : α → β → Prop) {C D : WSeq α → WSeq β → Prop}
    (H : ∀ s t, C s t → D s t) {o p} : LiftRelO R C o p → LiftRelO R D o p :=
  LiftRelO.imp (fun _ _ => id) H


/-- Definition of bisimilarity for weak sequences -/
@[simp]
def BisimO (R : WSeq α → WSeq α → Prop) : Option (α × WSeq α) → Option (α × WSeq α) → Prop :=
  LiftRelO (· = ·) R


theorem BisimO.imp {R S : WSeq α → WSeq α → Prop} (H : ∀ s t, R s t → S s t) {o p} :
    BisimO R o p → BisimO S o p :=
  LiftRelO.imp_right _ H


/-- Two weak sequences are `LiftRel R` related if they are either both empty,
  or they are both nonempty and the heads are `R` related and the tails are
  `LiftRel R` related. (This is a coinductive definition.) -/
def LiftRel (R : α → β → Prop) (s : WSeq α) (t : WSeq β) : Prop :=
  ∃ C : WSeq α → WSeq β → Prop,
    C s t ∧ ∀ {s t}, C s t → Computation.LiftRel (LiftRelO R C) (destruct s) (destruct t)


/-- If two sequences are equivalent, then they have the same values and
  the same computational behavior (i.e. if one loops forever then so does
  the other), although they may differ in the number of `think`s needed to
  arrive at the answer. -/
def Equiv : WSeq α → WSeq α → Prop :=
  LiftRel (· = ·)


theorem liftRel_destruct {R : α → β → Prop} {s : WSeq α} {t : WSeq β} :
    LiftRel R s t → Computation.LiftRel (LiftRelO R (LiftRel R)) (destruct s) (destruct t)
  | ⟨R, h1, h2⟩ => by
    /-
      α : Type u
      β : Type v
      R✝ : α → β → Prop
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      R : Stream'.WSeq α → Stream'.WSeq β → Prop
      h1 : R s t
      h2 : ∀ {s : Stream'.WSeq α} {t : Stream'.WSeq β}, R s t → Computation.LiftRel  …
      ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R✝ (Stream'.WSeq.LiftRel R✝)) s.d …
    -/
    refine Computation.LiftRel.imp ?_ _ _ (h2 h1)
    /-
      α : Type u
      β : Type v
      R✝ : α → β → Prop
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      R : Stream'.WSeq α → Stream'.WSeq β → Prop
      h1 : R s t
      h2 : ∀ {s : Stream'.WSeq α} {t : Stream'.WSeq β}, R s t → Computation.LiftRel  …
      ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
    -/
    apply LiftRelO.imp_right
    /-
      case H
      α : Type u
      β : Type v
      R✝ : α → β → Prop
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      R : Stream'.WSeq α → Stream'.WSeq β → Prop
      h1 : R s t
      h2 : ∀ {s : Stream'.WSeq α} {t : Stream'.WSeq β}, R s t → Computation.LiftRel  …
      ⊢ ∀ (s : Stream'.WSeq α) (t : Stream'.WSeq β), R s t → Stream'.WSeq.LiftRel R✝ …
    -/
    exact fun s' t' h' => ⟨R, h', @h2⟩
    /-
      🎉 no goals
    -/


theorem liftRel_destruct_iff {R : α → β → Prop} {s : WSeq α} {t : WSeq β} :
    LiftRel R s t ↔ Computation.LiftRel (LiftRelO R (LiftRel R)) (destruct s) (destruct t) :=
  ⟨liftRel_destruct, fun h =>
    ⟨fun s t =>
      LiftRel R s t ∨ Computation.LiftRel (LiftRelO R (LiftRel R)) (destruct s) (destruct t),
      Or.inr h, fun {s t} h => by
      have h : Computation.LiftRel (LiftRelO R (LiftRel R)) (destruct s) (destruct t) := by
        cases' h with h h
        · exact liftRel_destruct h
        · assumption
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Computation.LiftRel (Stream' …
        h : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s.d …
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
      -/
      apply Computation.LiftRel.imp _ _ _ h
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Computation.LiftRel (Stream' …
        h : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s.d …
        ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
      -/
      intro a b
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Computation.LiftRel (Stream' …
        h : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s.d …
        a : Option (Prod α (Stream'.WSeq α))
        b : Option (Prod β (Stream'.WSeq β))
        ⊢ Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) a b → Stream'.WSeq.LiftRelO …
      -/
      apply LiftRelO.imp_right
      /-
        case H
        α : Type u
        β : Type v
        R : α → β → Prop
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Computation.LiftRel (Stream' …
        h : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s.d …
        a : Option (Prod α (Stream'.WSeq α))
        b : Option (Prod β (Stream'.WSeq β))
        ⊢ ∀ (s : Stream'.WSeq α) (t : Stream'.WSeq β), Stream'.WSeq.LiftRel R s t → Or …
      -/
      intro s t
      /-
        case H
        α : Type u
        β : Type v
        R : α → β → Prop
        s✝¹ : Stream'.WSeq α
        t✝¹ : Stream'.WSeq β
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s …
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Computation.LiftRel (Stream' …
        h : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝. …
        a : Option (Prod α (Stream'.WSeq α))
        b : Option (Prod β (Stream'.WSeq β))
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        ⊢ Stream'.WSeq.LiftRel R s t → Or (Stream'.WSeq.LiftRel R s t) (Computation.Li …
      -/
      apply Or.inl⟩⟩
      /-
        🎉 no goals
      -/

-- Porting note: To avoid ambiguous notation, `~` became `~ʷ`.

@[inherit_doc] infixl:50 " ~ʷ " => Equiv


theorem destruct_congr {s t : WSeq α} :
    s ~ʷ t → Computation.LiftRel (BisimO (· ~ʷ ·)) (destruct s) (destruct t) :=
  liftRel_destruct


theorem destruct_congr_iff {s t : WSeq α} :
    s ~ʷ t ↔ Computation.LiftRel (BisimO (· ~ʷ ·)) (destruct s) (destruct t) :=
  liftRel_destruct_iff


theorem LiftRel.refl (R : α → α → Prop) (H : Reflexive R) : Reflexive (LiftRel R) := fun s => by
  /-
    α : Type u
    R : α → α → Prop
    H : Reflexive R
    s : Stream'.WSeq α
    ⊢ Stream'.WSeq.LiftRel R s s
  -/
  refine ⟨(· = ·), rfl, fun {s t} (h : s = t) => ?_⟩
  /-
    α : Type u
    R : α → α → Prop
    H : Reflexive R
    s✝ s t : Stream'.WSeq α
    h : Eq s t
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun x1 x2 => Eq x1 x2) s.destru …
  -/
  rw [← h]
  /-
    α : Type u
    R : α → α → Prop
    H : Reflexive R
    s✝ s t : Stream'.WSeq α
    h : Eq s t
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun x1 x2 => Eq x1 x2) s.destru …
  -/
  apply Computation.LiftRel.refl
  /-
    case H
    α : Type u
    R : α → α → Prop
    H : Reflexive R
    s✝ s t : Stream'.WSeq α
    h : Eq s t
    ⊢ Reflexive (Stream'.WSeq.LiftRelO R fun x1 x2 => Eq x1 x2)
  -/
  intro a
  /-
    case H
    α : Type u
    R : α → α → Prop
    H : Reflexive R
    s✝ s t : Stream'.WSeq α
    h : Eq s t
    a : Option (Prod α (Stream'.WSeq α))
    ⊢ Stream'.WSeq.LiftRelO R (fun x1 x2 => Eq x1 x2) a a
  -/
  cases' a with a
    /-
      case H.none
      α : Type u
      R : α → α → Prop
      H : Reflexive R
      s✝ s t : Stream'.WSeq α
      h : Eq s t
      ⊢ Stream'.WSeq.LiftRelO R (fun x1 x2 => Eq x1 x2) Option.none Option.none
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case H.some
      α : Type u
      R : α → α → Prop
      H : Reflexive R
      s✝ s t : Stream'.WSeq α
      h : Eq s t
      a : Prod α (Stream'.WSeq α)
      ⊢ Stream'.WSeq.LiftRelO R (fun x1 x2 => Eq x1 x2) (Option.some a) (Option.some …
    -/
  · cases a
    /-
      case H.some.mk
      α : Type u
      R : α → α → Prop
      H : Reflexive R
      s✝ s t : Stream'.WSeq α
      h : Eq s t
      fst✝ : α
      snd✝ : Stream'.WSeq α
      ⊢ Stream'.WSeq.LiftRelO R (fun x1 x2 => Eq x1 x2) (Option.some { fst := fst✝,  …
    -/
    simp only [LiftRelO, and_true]
    /-
      case H.some.mk
      α : Type u
      R : α → α → Prop
      H : Reflexive R
      s✝ s t : Stream'.WSeq α
      h : Eq s t
      fst✝ : α
      snd✝ : Stream'.WSeq α
      ⊢ R fst✝ fst✝
    -/
    apply H
    /-
      🎉 no goals
    -/


theorem LiftRelO.swap (R : α → β → Prop) (C) :
    swap (LiftRelO R C) = LiftRelO (swap R) (swap C) := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    C : Stream'.WSeq α → Stream'.WSeq β → Prop
    ⊢ Eq (Function.swap (Stream'.WSeq.LiftRelO R C)) (Stream'.WSeq.LiftRelO (Funct …
  -/
  funext x y
  /-
    case h.h
    α : Type u
    β : Type v
    R : α → β → Prop
    C : Stream'.WSeq α → Stream'.WSeq β → Prop
    x : Option (Prod β (Stream'.WSeq β))
    y : Option (Prod α (Stream'.WSeq α))
    ⊢ Eq (Function.swap (Stream'.WSeq.LiftRelO R C) x y) (Stream'.WSeq.LiftRelO (F …
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
  rcases x with ⟨⟩ | ⟨hx, jx⟩ <;> rcases y with ⟨⟩ | ⟨hy, jy⟩ <;> rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem LiftRel.swap_lem {R : α → β → Prop} {s1 s2} (h : LiftRel R s1 s2) :
    LiftRel (swap R) s2 s1 := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s1 : Stream'.WSeq α
    s2 : Stream'.WSeq β
    h : Stream'.WSeq.LiftRel R s1 s2
    ⊢ Stream'.WSeq.LiftRel (Function.swap R) s2 s1
  -/
  refine ⟨swap (LiftRel R), h, fun {s t} (h : LiftRel R t s) => ?_⟩
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s1 : Stream'.WSeq α
    s2 : Stream'.WSeq β
    h✝ : Stream'.WSeq.LiftRel R s1 s2
    s : Stream'.WSeq β
    t : Stream'.WSeq α
    h : Stream'.WSeq.LiftRel R t s
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO (Function.swap R) (Function.swap  …
  -/
  rw [← LiftRelO.swap, Computation.LiftRel.swap]
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s1 : Stream'.WSeq α
    s2 : Stream'.WSeq β
    h✝ : Stream'.WSeq.LiftRel R s1 s2
    s : Stream'.WSeq β
    t : Stream'.WSeq α
    h : Stream'.WSeq.LiftRel R t s
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t.des …
  -/
  apply liftRel_destruct h
  /-
    🎉 no goals
  -/


theorem LiftRel.swap (R : α → β → Prop) : swap (LiftRel R) = LiftRel (swap R) :=
  funext fun _ => funext fun _ => propext ⟨LiftRel.swap_lem, LiftRel.swap_lem⟩


theorem LiftRel.symm (R : α → α → Prop) (H : Symmetric R) : Symmetric (LiftRel R) :=
                                                        /-
                                                          α : Type u
                                                          R : α → α → Prop
                                                          H : Symmetric R
                                                          s1 s2 : Stream'.WSeq α
                                                          h : Function.swap (Stream'.WSeq.LiftRel R) s2 s1
                                                          ⊢ Stream'.WSeq.LiftRel R s2 s1
                                                        -/
  fun s1 s2 (h : Function.swap (LiftRel R) s2 s1) => by rwa [LiftRel.swap, H.swap_eq] at h
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem LiftRel.trans (R : α → α → Prop) (H : Transitive R) : Transitive (LiftRel R) :=
  fun s t u h1 h2 => by
  /-
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s t u : Stream'.WSeq α
    h1 : Stream'.WSeq.LiftRel R s t
    h2 : Stream'.WSeq.LiftRel R t u
    ⊢ Stream'.WSeq.LiftRel R s u
  -/
  refine ⟨fun s u => ∃ t, LiftRel R s t ∧ LiftRel R t u, ⟨t, h1, h2⟩, fun {s u} h => ?_⟩
  /-
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s✝ t u✝ : Stream'.WSeq α
    h1 : Stream'.WSeq.LiftRel R s✝ t
    h2 : Stream'.WSeq.LiftRel R t u✝
    s u : Stream'.WSeq α
    h : (fun s u => Exists fun t => And (Stream'.WSeq.LiftRel R s t) (Stream'.WSeq …
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s u => Exists fun t => And  …
  -/
  rcases h with ⟨t, h1, h2⟩
  /-
    case intro.intro
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s✝ t✝ u✝ : Stream'.WSeq α
    h1✝ : Stream'.WSeq.LiftRel R s✝ t✝
    h2✝ : Stream'.WSeq.LiftRel R t✝ u✝
    s u t : Stream'.WSeq α
    h1 : Stream'.WSeq.LiftRel R s t
    h2 : Stream'.WSeq.LiftRel R t u
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s u => Exists fun t => And  …
  -/
  have h1 := liftRel_destruct h1
  /-
    case intro.intro
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s✝ t✝ u✝ : Stream'.WSeq α
    h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
    h2✝ : Stream'.WSeq.LiftRel R t✝ u✝
    s u t : Stream'.WSeq α
    h1✝ : Stream'.WSeq.LiftRel R s t
    h2 : Stream'.WSeq.LiftRel R t u
    h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s u => Exists fun t => And  …
  -/
  have h2 := liftRel_destruct h2
  refine
    Computation.liftRel_def.2
      ⟨(Computation.terminates_of_liftRel h1).trans (Computation.terminates_of_liftRel h2),
        fun {a c} ha hc => ?_⟩
  /-
    case intro.intro
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s✝ t✝ u✝ : Stream'.WSeq α
    h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
    h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
    s u t : Stream'.WSeq α
    h1✝ : Stream'.WSeq.LiftRel R s t
    h2✝ : Stream'.WSeq.LiftRel R t u
    h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
    h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
    a c : Option (Prod α (Stream'.WSeq α))
    ha : Membership.mem s.destruct a
    hc : Membership.mem u.destruct c
    ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
  -/
  rcases h1.left ha with ⟨b, hb, t1⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s✝ t✝ u✝ : Stream'.WSeq α
    h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
    h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
    s u t : Stream'.WSeq α
    h1✝ : Stream'.WSeq.LiftRel R s t
    h2✝ : Stream'.WSeq.LiftRel R t u
    h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
    h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
    a c : Option (Prod α (Stream'.WSeq α))
    ha : Membership.mem s.destruct a
    hc : Membership.mem u.destruct c
    b : Option (Prod α (Stream'.WSeq α))
    hb : Membership.mem t.destruct b
    t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) a b
    ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
  -/
  have t2 := Computation.rel_of_liftRel h2 hb hc
  /-
    case intro.intro.intro.intro
    α : Type u
    R : α → α → Prop
    H : Transitive R
    s✝ t✝ u✝ : Stream'.WSeq α
    h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
    h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
    s u t : Stream'.WSeq α
    h1✝ : Stream'.WSeq.LiftRel R s t
    h2✝ : Stream'.WSeq.LiftRel R t u
    h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
    h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
    a c : Option (Prod α (Stream'.WSeq α))
    ha : Membership.mem s.destruct a
    hc : Membership.mem u.destruct c
    b : Option (Prod α (Stream'.WSeq α))
    hb : Membership.mem t.destruct b
    t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) a b
    t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b c
    ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
  -/
  cases' a with a <;> cases' c with c
    /-
      case intro.intro.intro.intro.none.none
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      b : Option (Prod α (Stream'.WSeq α))
      hb : Membership.mem t.destruct b
      ha : Membership.mem s.destruct Option.none
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none b
      hc : Membership.mem u.destruct Option.none
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b Option.none
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
  · trivial
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.none.some
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      b : Option (Prod α (Stream'.WSeq α))
      hb : Membership.mem t.destruct b
      ha : Membership.mem s.destruct Option.none
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none b
      c : Prod α (Stream'.WSeq α)
      hc : Membership.mem u.destruct (Option.some c)
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b (Option.some c)
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
  · cases b
      /-
        case intro.intro.intro.intro.none.some.none
        α : Type u
        R : α → α → Prop
        H : Transitive R
        s✝ t✝ u✝ : Stream'.WSeq α
        h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
        h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
        s u t : Stream'.WSeq α
        h1✝ : Stream'.WSeq.LiftRel R s t
        h2✝ : Stream'.WSeq.LiftRel R t u
        h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
        h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
        ha : Membership.mem s.destruct Option.none
        c : Prod α (Stream'.WSeq α)
        hc : Membership.mem u.destruct (Option.some c)
        hb : Membership.mem t.destruct Option.none
        t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
        t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none (Option.some …
        ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
      -/
    · cases t2
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.none.some.some
        α : Type u
        R : α → α → Prop
        H : Transitive R
        s✝ t✝ u✝ : Stream'.WSeq α
        h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
        h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
        s u t : Stream'.WSeq α
        h1✝ : Stream'.WSeq.LiftRel R s t
        h2✝ : Stream'.WSeq.LiftRel R t u
        h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
        h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
        ha : Membership.mem s.destruct Option.none
        c : Prod α (Stream'.WSeq α)
        hc : Membership.mem u.destruct (Option.some c)
        val✝ : Prod α (Stream'.WSeq α)
        hb : Membership.mem t.destruct (Option.some val✝)
        t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none (Option.some …
        t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some val✝) (Opti …
        ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
      -/
    · cases t1
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.some.none
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      b : Option (Prod α (Stream'.WSeq α))
      hb : Membership.mem t.destruct b
      a : Prod α (Stream'.WSeq α)
      ha : Membership.mem s.destruct (Option.some a)
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some a) b
      hc : Membership.mem u.destruct Option.none
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b Option.none
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
  · cases a
    /-
      case intro.intro.intro.intro.some.none.mk
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      b : Option (Prod α (Stream'.WSeq α))
      hb : Membership.mem t.destruct b
      hc : Membership.mem u.destruct Option.none
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b Option.none
      fst✝ : α
      snd✝ : Stream'.WSeq α
      ha : Membership.mem s.destruct (Option.some { fst := fst✝, snd := snd✝ })
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := fs …
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    cases' b with b
      /-
        case intro.intro.intro.intro.some.none.mk.none
        α : Type u
        R : α → α → Prop
        H : Transitive R
        s✝ t✝ u✝ : Stream'.WSeq α
        h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
        h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
        s u t : Stream'.WSeq α
        h1✝ : Stream'.WSeq.LiftRel R s t
        h2✝ : Stream'.WSeq.LiftRel R t u
        h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
        h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
        hc : Membership.mem u.destruct Option.none
        fst✝ : α
        snd✝ : Stream'.WSeq α
        ha : Membership.mem s.destruct (Option.some { fst := fst✝, snd := snd✝ })
        hb : Membership.mem t.destruct Option.none
        t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
        t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := fs …
        ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
      -/
    · cases t1
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.some.none.mk.some
        α : Type u
        R : α → α → Prop
        H : Transitive R
        s✝ t✝ u✝ : Stream'.WSeq α
        h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
        h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
        s u t : Stream'.WSeq α
        h1✝ : Stream'.WSeq.LiftRel R s t
        h2✝ : Stream'.WSeq.LiftRel R t u
        h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
        h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
        hc : Membership.mem u.destruct Option.none
        fst✝ : α
        snd✝ : Stream'.WSeq α
        ha : Membership.mem s.destruct (Option.some { fst := fst✝, snd := snd✝ })
        b : Prod α (Stream'.WSeq α)
        hb : Membership.mem t.destruct (Option.some b)
        t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some b) Option.n …
        t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := fs …
        ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
      -/
    · cases b
      /-
        case intro.intro.intro.intro.some.none.mk.some.mk
        α : Type u
        R : α → α → Prop
        H : Transitive R
        s✝ t✝ u✝ : Stream'.WSeq α
        h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
        h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
        s u t : Stream'.WSeq α
        h1✝ : Stream'.WSeq.LiftRel R s t
        h2✝ : Stream'.WSeq.LiftRel R t u
        h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
        h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
        hc : Membership.mem u.destruct Option.none
        fst✝¹ : α
        snd✝¹ : Stream'.WSeq α
        ha : Membership.mem s.destruct (Option.some { fst := fst✝¹, snd := snd✝¹ })
        fst✝ : α
        snd✝ : Stream'.WSeq α
        hb : Membership.mem t.destruct (Option.some { fst := fst✝, snd := snd✝ })
        t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := fs …
        t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := fs …
        ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
      -/
      cases t2
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.some.some
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s. …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      b : Option (Prod α (Stream'.WSeq α))
      hb : Membership.mem t.destruct b
      a : Prod α (Stream'.WSeq α)
      ha : Membership.mem s.destruct (Option.some a)
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some a) b
      c : Prod α (Stream'.WSeq α)
      hc : Membership.mem u.destruct (Option.some c)
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b (Option.some c)
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
  · cases' a with a s
    /-
      case intro.intro.intro.intro.some.some.mk
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝¹ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s✝ u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s✝ t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      b : Option (Prod α (Stream'.WSeq α))
      hb : Membership.mem t.destruct b
      c : Prod α (Stream'.WSeq α)
      hc : Membership.mem u.destruct (Option.some c)
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) b (Option.some c)
      a : α
      s : Stream'.WSeq α
      ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a, …
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    cases' b with b
      /-
        case intro.intro.intro.intro.some.some.mk.none
        α : Type u
        R : α → α → Prop
        H : Transitive R
        s✝¹ t✝ u✝ : Stream'.WSeq α
        h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝
        h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
        s✝ u t : Stream'.WSeq α
        h1✝ : Stream'.WSeq.LiftRel R s✝ t
        h2✝ : Stream'.WSeq.LiftRel R t u
        h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
        h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
        c : Prod α (Stream'.WSeq α)
        hc : Membership.mem u.destruct (Option.some c)
        a : α
        s : Stream'.WSeq α
        ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
        hb : Membership.mem t.destruct Option.none
        t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none (Option.some …
        t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a, …
        ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
      -/
    · cases t1
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.some.some.mk.some
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝¹ t✝ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝
      h2✝¹ : Stream'.WSeq.LiftRel R t✝ u✝
      s✝ u t : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s✝ t
      h2✝ : Stream'.WSeq.LiftRel R t u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t. …
      c : Prod α (Stream'.WSeq α)
      hc : Membership.mem u.destruct (Option.some c)
      a : α
      s : Stream'.WSeq α
      ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
      b : Prod α (Stream'.WSeq α)
      hb : Membership.mem t.destruct (Option.some b)
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some b) (Option. …
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a, …
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    cases' b with b t
    /-
      case intro.intro.intro.intro.some.some.mk.some.mk
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝¹ t✝¹ u✝ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝¹
      h2✝¹ : Stream'.WSeq.LiftRel R t✝¹ u✝
      s✝ u t✝ : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝ : Stream'.WSeq.LiftRel R t✝ u
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t✝ …
      c : Prod α (Stream'.WSeq α)
      hc : Membership.mem u.destruct (Option.some c)
      a : α
      s : Stream'.WSeq α
      ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
      b : α
      t : Stream'.WSeq α
      hb : Membership.mem t✝.destruct (Option.some { fst := b, snd := t })
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := b, …
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a, …
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    cases' c with c u
    /-
      case intro.intro.intro.intro.some.some.mk.some.mk.mk
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝¹ t✝¹ u✝¹ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝¹
      h2✝¹ : Stream'.WSeq.LiftRel R t✝¹ u✝¹
      s✝ u✝ t✝ : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝ : Stream'.WSeq.LiftRel R t✝ u✝
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t✝ …
      a : α
      s : Stream'.WSeq α
      ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
      b : α
      t : Stream'.WSeq α
      hb : Membership.mem t✝.destruct (Option.some { fst := b, snd := t })
      t1 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a, …
      c : α
      u : Stream'.WSeq α
      hc : Membership.mem u✝.destruct (Option.some { fst := c, snd := u })
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := b, …
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    cases' t1 with ab st
    /-
      case intro.intro.intro.intro.some.some.mk.some.mk.mk.intro
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝¹ t✝¹ u✝¹ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝¹
      h2✝¹ : Stream'.WSeq.LiftRel R t✝¹ u✝¹
      s✝ u✝ t✝ : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝ : Stream'.WSeq.LiftRel R t✝ u✝
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t✝ …
      a : α
      s : Stream'.WSeq α
      ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
      b : α
      t : Stream'.WSeq α
      hb : Membership.mem t✝.destruct (Option.some { fst := b, snd := t })
      c : α
      u : Stream'.WSeq α
      hc : Membership.mem u✝.destruct (Option.some { fst := c, snd := u })
      t2 : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := b, …
      ab : R a b
      st : Stream'.WSeq.LiftRel R s t
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    cases' t2 with bc tu
    /-
      case intro.intro.intro.intro.some.some.mk.some.mk.mk.intro.intro
      α : Type u
      R : α → α → Prop
      H : Transitive R
      s✝¹ t✝¹ u✝¹ : Stream'.WSeq α
      h1✝¹ : Stream'.WSeq.LiftRel R s✝¹ t✝¹
      h2✝¹ : Stream'.WSeq.LiftRel R t✝¹ u✝¹
      s✝ u✝ t✝ : Stream'.WSeq α
      h1✝ : Stream'.WSeq.LiftRel R s✝ t✝
      h2✝ : Stream'.WSeq.LiftRel R t✝ u✝
      h1 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) s✝ …
      h2 : Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) t✝ …
      a : α
      s : Stream'.WSeq α
      ha : Membership.mem s✝.destruct (Option.some { fst := a, snd := s })
      b : α
      t : Stream'.WSeq α
      hb : Membership.mem t✝.destruct (Option.some { fst := b, snd := t })
      c : α
      u : Stream'.WSeq α
      hc : Membership.mem u✝.destruct (Option.some { fst := c, snd := u })
      ab : R a b
      st : Stream'.WSeq.LiftRel R s t
      bc : R b c
      tu : Stream'.WSeq.LiftRel R t u
      ⊢ Stream'.WSeq.LiftRelO R (fun s u => Exists fun t => And (Stream'.WSeq.LiftRe …
    -/
    exact ⟨H ab bc, t, st, tu⟩
    /-
      🎉 no goals
    -/


theorem LiftRel.equiv (R : α → α → Prop) : Equivalence R → Equivalence (LiftRel R)
  | ⟨refl, symm, trans⟩ => ⟨LiftRel.refl R refl, @(LiftRel.symm R @symm), @(LiftRel.trans R @trans)⟩


@[refl]
theorem Equiv.refl : ∀ s : WSeq α, s ~ʷ s :=
  LiftRel.refl (· = ·) Eq.refl


@[symm]
theorem Equiv.symm : ∀ {s t : WSeq α}, s ~ʷ t → t ~ʷ s :=
  @(LiftRel.symm (· = ·) (@Eq.symm _))


@[trans]
theorem Equiv.trans : ∀ {s t u : WSeq α}, s ~ʷ t → t ~ʷ u → s ~ʷ u :=
  @(LiftRel.trans (· = ·) (@Eq.trans _))


theorem Equiv.equivalence : Equivalence (@Equiv α) :=
  ⟨@Equiv.refl _, @Equiv.symm _, @Equiv.trans _⟩


@[simp]
theorem destruct_nil : destruct (nil : WSeq α) = Computation.pure none :=
  Computation.destruct_eq_pure rfl


@[simp]
theorem destruct_cons (a : α) (s) : destruct (cons a s) = Computation.pure (some (a, s)) :=
                                     /-
                                       α : Type u
                                       a : α
                                       s : Stream'.WSeq α
                                       ⊢ Eq (Stream'.WSeq.cons a s).destruct.destruct (Sum.inl (Option.some { fst :=  …
                                     -/
  Computation.destruct_eq_pure <| by simp [destruct, cons, Computation.rmap]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem destruct_think (s : WSeq α) : destruct (think s) = (destruct s).think :=
                                      /-
                                        α : Type u
                                        s : Stream'.WSeq α
                                        ⊢ Eq s.think.destruct.destruct (Sum.inr s.destruct)
                                      -/
  Computation.destruct_eq_think <| by simp [destruct, think, Computation.rmap]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem seq_destruct_nil : Seq.destruct (nil : WSeq α) = none :=
  Seq.destruct_nil


@[simp]
theorem seq_destruct_cons (a : α) (s) : Seq.destruct (cons a s) = some (some a, s) :=
  Seq.destruct_cons _ _


@[simp]
theorem seq_destruct_think (s : WSeq α) : Seq.destruct (think s) = some (none, s) :=
  Seq.destruct_cons _ _


@[simp]
                                                                     /-
                                                                       α : Type u
                                                                       ⊢ Eq Stream'.WSeq.nil.head (Computation.pure Option.none)
                                                                     -/
theorem head_nil : head (nil : WSeq α) = Computation.pure none := by simp [head]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
                                                                                  /-
                                                                                    α : Type u
                                                                                    a : α
                                                                                    s : Stream'.WSeq α
                                                                                    ⊢ Eq (Stream'.WSeq.cons a s).head (Computation.pure (Option.some a))
                                                                                  -/
theorem head_cons (a : α) (s) : head (cons a s) = Computation.pure (some a) := by simp [head]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
                                                                        /-
                                                                          α : Type u
                                                                          s : Stream'.WSeq α
                                                                          ⊢ Eq s.think.head s.head.think
                                                                        -/
theorem head_think (s : WSeq α) : head (think s) = (head s).think := by simp [head]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem flatten_pure (s : WSeq α) : flatten (Computation.pure s) = s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ Eq (Stream'.WSeq.flatten (Computation.pure s)) s
  -/
  refine Seq.eq_of_bisim (fun s1 s2 => flatten (Computation.pure s2) = s1) ?_ rfl
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Eq (Stream'.WSeq.flatten (Computatio …
  -/
  intro s' s h
  /-
    α : Type u
    s✝ : Stream'.WSeq α
    s' s : Stream'.Seq (Option α)
    h : Eq (Stream'.WSeq.flatten (Computation.pure s)) s'
    ⊢ Stream'.Seq.BisimO (fun s1 s2 => Eq (Stream'.WSeq.flatten (Computation.pure  …
  -/
  rw [← h]
  /-
    α : Type u
    s✝ : Stream'.WSeq α
    s' s : Stream'.Seq (Option α)
    h : Eq (Stream'.WSeq.flatten (Computation.pure s)) s'
    ⊢ Stream'.Seq.BisimO (fun s1 s2 => Eq (Stream'.WSeq.flatten (Computation.pure  …
  -/
  simp only [Seq.BisimO, flatten, Seq.omap, pure_def, Seq.corec_eq, destruct_pure]
  cases Seq.destruct s with
  | none => simp
  | some val =>
    cases' val with o s'
    simp


@[simp]
theorem flatten_think (c : Computation (WSeq α)) : flatten c.think = think (flatten c) :=
                             /-
                               α : Type u
                               c : Computation (Stream'.WSeq α)
                               ⊢ Eq (Stream'.Seq.destruct (Stream'.WSeq.flatten c.think)) (Option.some { fst  …
                             -/
  Seq.destruct_eq_cons <| by simp [flatten, think]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem destruct_flatten (c : Computation (WSeq α)) : destruct (flatten c) = c >>= destruct := by
  refine
    Computation.eq_of_bisim
      (fun c1 c2 => c1 = c2 ∨ ∃ c, c1 = destruct (flatten c) ∧ c2 = Computation.bind c destruct) ?_
      (Or.inr ⟨c, rfl, rfl⟩)
  /-
    α : Type u
    c : Computation (Stream'.WSeq α)
    ⊢ Computation.IsBisimulation fun c1 c2 => Or (Eq c1 c2) (Exists fun c => And ( …
  -/
  intro c1 c2 h
  exact
    match c1, c2, h with
    | c, _, Or.inl rfl => by cases c.destruct <;> simp
    | _, _, Or.inr ⟨c, rfl, rfl⟩ => by
      induction' c using Computation.recOn with a c'
      · simp; cases (destruct a).destruct <;> simp
      · simpa using Or.inr ⟨c', rfl, rfl⟩


theorem head_terminates_iff (s : WSeq α) : Terminates (head s) ↔ Terminates (destruct s) :=
  terminates_map_iff _ (destruct s)


@[simp]
                                                   /-
                                                     α : Type u
                                                     ⊢ Eq Stream'.WSeq.nil.tail Stream'.WSeq.nil
                                                   -/
theorem tail_nil : tail (nil : WSeq α) = nil := by simp [tail]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
                                                          /-
                                                            α : Type u
                                                            a : α
                                                            s : Stream'.WSeq α
                                                            ⊢ Eq (Stream'.WSeq.cons a s).tail s
                                                          -/
theorem tail_cons (a : α) (s) : tail (cons a s) = s := by simp [tail]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                                        /-
                                                                          α : Type u
                                                                          s : Stream'.WSeq α
                                                                          ⊢ Eq s.think.tail s.tail.think
                                                                        -/
theorem tail_think (s : WSeq α) : tail (think s) = (tail s).think := by simp [tail]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
                                                          /-
                                                            α : Type u
                                                            n : Nat
                                                            ⊢ Eq (Stream'.WSeq.nil.drop n) Stream'.WSeq.nil
                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
theorem dropn_nil (n) : drop (nil : WSeq α) n = nil := by induction n <;> simp [*, drop]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem dropn_cons (a : α) (s) (n) : drop (cons a s) (n + 1) = drop s n := by
  induction n with
  | zero => simp [drop]
  | succ n n_ih =>
    simp [drop, ← n_ih]


@[simp]
theorem dropn_think (s : WSeq α) (n) : drop (think s) n = (drop s n).think := by
  /-
    α : Type u
    s : Stream'.WSeq α
    n : Nat
    ⊢ Eq (s.think.drop n) (s.drop n).think
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, drop]
                  /-
                    🎉 no goals
                  -/


theorem dropn_add (s : WSeq α) (m) : ∀ n, drop s (m + n) = drop (drop s m) n
  | 0 => rfl
  | n + 1 => congr_arg tail (dropn_add s m n)


theorem dropn_tail (s : WSeq α) (n) : drop (tail s) n = drop s (n + 1) := by
  /-
    α : Type u
    s : Stream'.WSeq α
    n : Nat
    ⊢ Eq (s.tail.drop n) (s.drop (HAdd.hAdd n 1))
  -/
  rw [Nat.add_comm]
  /-
    α : Type u
    s : Stream'.WSeq α
    n : Nat
    ⊢ Eq (s.tail.drop n) (s.drop (HAdd.hAdd 1 n))
  -/
  symm
  /-
    α : Type u
    s : Stream'.WSeq α
    n : Nat
    ⊢ Eq (s.drop (HAdd.hAdd 1 n)) (s.tail.drop n)
  -/
  apply dropn_add
  /-
    🎉 no goals
  -/


theorem get?_add (s : WSeq α) (m n) : get? s (m + n) = get? (drop s m) n :=
  congr_arg head (dropn_add _ _ _)


theorem get?_tail (s : WSeq α) (n) : get? (tail s) n = get? s (n + 1) :=
  congr_arg head (dropn_tail _ _)


@[simp]
theorem join_nil : join nil = (nil : WSeq α) :=
  Seq.join_nil


@[simp]
theorem join_think (S : WSeq (WSeq α)) : join (think S) = think (join S) := by
  /-
    α : Type u
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Eq S.think.join S.join.think
  -/
  simp only [join, think]
  /-
    α : Type u
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Eq (Functor.map (fun o => Stream'.WSeq.join.match_1 (fun o => Stream'.Seq1 ( …
  -/
  dsimp only [(· <$> ·)]
  /-
    α : Type u
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Eq (Stream'.Seq.map (fun o => Stream'.WSeq.join.match_1 (fun o => Stream'.Se …
  -/
  simp [join, Seq1.ret]
  /-
    🎉 no goals
  -/


@[simp]
theorem join_cons (s : WSeq α) (S) : join (cons s S) = think (append s (join S)) := by
  /-
    α : Type u
    s : Stream'.WSeq α
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Eq (Stream'.WSeq.cons s S).join (s.append S.join).think
  -/
  simp only [join, think]
  /-
    α : Type u
    s : Stream'.WSeq α
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Eq (Functor.map (fun o => Stream'.WSeq.join.match_1 (fun o => Stream'.Seq1 ( …
  -/
  dsimp only [(· <$> ·)]
  /-
    α : Type u
    s : Stream'.WSeq α
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Eq (Stream'.Seq.map (fun o => Stream'.WSeq.join.match_1 (fun o => Stream'.Se …
  -/
  simp [join, cons, append]
  /-
    🎉 no goals
  -/


@[simp]
theorem nil_append (s : WSeq α) : append nil s = s :=
  Seq.nil_append _


@[simp]
theorem cons_append (a : α) (s t) : append (cons a s) t = cons a (append s t) :=
  Seq.cons_append _ _ _


@[simp]
theorem think_append (s t : WSeq α) : append (think s) t = think (append s t) :=
  Seq.cons_append _ _ _


@[simp]
theorem append_nil (s : WSeq α) : append s nil = s :=
  Seq.append_nil _


@[simp]
theorem append_assoc (s t u : WSeq α) : append (append s t) u = append s (append t u) :=
  Seq.append_assoc _ _ _


/-- auxiliary definition of tail over weak sequences -/
@[simp]
def tail.aux : Option (α × WSeq α) → Computation (Option (α × WSeq α))
  | none => Computation.pure none
  | some (_, s) => destruct s


theorem destruct_tail (s : WSeq α) : destruct (tail s) = destruct s >>= tail.aux := by
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ Eq s.tail.destruct (Bind.bind s.destruct Stream'.WSeq.tail.aux)
  -/
  simp only [tail, destruct_flatten, tail.aux]; rw [← bind_pure_comp, LawfulMonad.bind_assoc]
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ Eq (Bind.bind s.destruct fun x => Bind.bind (Pure.pure (Option.rec Stream'.W …
  -/
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
  apply congr_arg; ext1 (_ | ⟨a, s⟩) <;> apply (@pure_bind Computation _ _ _ _ _ _).trans _ <;> simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


/-- auxiliary definition of drop over weak sequences -/
@[simp]
def drop.aux : ℕ → Option (α × WSeq α) → Computation (Option (α × WSeq α))
  | 0 => Computation.pure
  | n + 1 => fun a => tail.aux a >>= drop.aux n


theorem drop.aux_none : ∀ n, @drop.aux α n none = Computation.pure none
  | 0 => rfl
  | n + 1 =>
    show Computation.bind (Computation.pure none) (drop.aux n) = Computation.pure none by
      /-
        α : Type u
        n : Nat
        ⊢ Eq ((Computation.pure Option.none).bind (Stream'.WSeq.drop.aux n)) (Computat …
      -/
      rw [ret_bind, drop.aux_none n]
      /-
        🎉 no goals
      -/


theorem destruct_dropn : ∀ (s : WSeq α) (n), destruct (drop s n) = destruct s >>= drop.aux n
  | _, 0 => (bind_pure' _).symm
  | s, n + 1 => by
    /-
      α : Type u
      s : Stream'.WSeq α
      n : Nat
      ⊢ Eq (s.drop (HAdd.hAdd n 1)).destruct (Bind.bind s.destruct (Stream'.WSeq.dro …
    -/
    rw [← dropn_tail, destruct_dropn _ n, destruct_tail, LawfulMonad.bind_assoc]
    /-
      α : Type u
      s : Stream'.WSeq α
      n : Nat
      ⊢ Eq (Bind.bind s.destruct fun x => Bind.bind (Stream'.WSeq.tail.aux x) (Strea …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem head_terminates_of_head_tail_terminates (s : WSeq α) [T : Terminates (head (tail s))] :
    Terminates (head s) :=
  (head_terminates_iff _).2 <| by
    /-
      α : Type u
      s : Stream'.WSeq α
      T : s.tail.head.Terminates
      ⊢ s.destruct.Terminates
    -/
    rcases (head_terminates_iff _).1 T with ⟨⟨a, h⟩⟩
    /-
      case mk.intro
      α : Type u
      s : Stream'.WSeq α
      T : s.tail.head.Terminates
      a : Option (Prod α (Stream'.WSeq α))
      h : Membership.mem s.tail.destruct a
      ⊢ s.destruct.Terminates
    -/
    simp? [tail] at h says simp only [tail, destruct_flatten, bind_map_left] at h
    /-
      case mk.intro
      α : Type u
      s : Stream'.WSeq α
      T : s.tail.head.Terminates
      a : Option (Prod α (Stream'.WSeq α))
      h : Membership.mem (Bind.bind s.destruct fun a => (Option.rec Stream'.WSeq.nil …
      ⊢ s.destruct.Terminates
    -/
    rcases exists_of_mem_bind h with ⟨s', h1, _⟩
    /-
      case mk.intro.intro.intro
      α : Type u
      s : Stream'.WSeq α
      T : s.tail.head.Terminates
      a : Option (Prod α (Stream'.WSeq α))
      h : Membership.mem (Bind.bind s.destruct fun a => (Option.rec Stream'.WSeq.nil …
      s' : Option (Prod α (Stream'.WSeq α))
      h1 : Membership.mem s.destruct s'
      right✝ : Membership.mem (Option.rec Stream'.WSeq.nil Prod.snd s').destruct a
      ⊢ s.destruct.Terminates
    -/
    exact terminates_of_mem h1
    /-
      🎉 no goals
    -/


theorem destruct_some_of_destruct_tail_some {s : WSeq α} {a} (h : some a ∈ destruct (tail s)) :
    ∃ a', some a' ∈ destruct s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a : Prod α (Stream'.WSeq α)
    h : Membership.mem s.tail.destruct (Option.some a)
    ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
  -/
  unfold tail Functor.map at h; simp only [destruct_flatten] at h
  /-
    α : Type u
    s : Stream'.WSeq α
    a : Prod α (Stream'.WSeq α)
    h : Membership.mem (Bind.bind (Applicative.toFunctor.1 (fun o => Option.rec St …
    ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
  -/
  rcases exists_of_mem_bind h with ⟨t, tm, td⟩; clear h
  /-
    case intro.intro
    α : Type u
    s : Stream'.WSeq α
    a : Prod α (Stream'.WSeq α)
    t : Stream'.WSeq α
    tm : Membership.mem (Applicative.toFunctor.1 (fun o => Option.rec Stream'.WSeq …
    td : Membership.mem t.destruct (Option.some a)
    ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
  -/
  rcases Computation.exists_of_mem_map tm with ⟨t', ht', ht2⟩; clear tm
  /-
    case intro.intro.intro.intro
    α : Type u
    s : Stream'.WSeq α
    a : Prod α (Stream'.WSeq α)
    t : Stream'.WSeq α
    td : Membership.mem t.destruct (Option.some a)
    t' : Option (Prod α (Stream'.WSeq α))
    ht' : Membership.mem s.destruct t'
    ht2 : Eq (Option.rec Stream'.WSeq.nil Prod.snd t') t
    ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
  -/
  cases' t' with t' <;> rw [← ht2] at td <;> simp only [destruct_nil] at td
    /-
      case intro.intro.intro.intro.none
      α : Type u
      s : Stream'.WSeq α
      a : Prod α (Stream'.WSeq α)
      t : Stream'.WSeq α
      ht' : Membership.mem s.destruct Option.none
      ht2 : Eq (Option.rec Stream'.WSeq.nil Prod.snd Option.none) t
      td : Membership.mem (Computation.pure Option.none) (Option.some a)
      ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
    -/
  · have := mem_unique td (ret_mem _)
    /-
      case intro.intro.intro.intro.none
      α : Type u
      s : Stream'.WSeq α
      a : Prod α (Stream'.WSeq α)
      t : Stream'.WSeq α
      ht' : Membership.mem s.destruct Option.none
      ht2 : Eq (Option.rec Stream'.WSeq.nil Prod.snd Option.none) t
      td : Membership.mem (Computation.pure Option.none) (Option.some a)
      this : Eq (Option.some a) Option.none
      ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.some
      α : Type u
      s : Stream'.WSeq α
      a : Prod α (Stream'.WSeq α)
      t : Stream'.WSeq α
      t' : Prod α (Stream'.WSeq α)
      td : Membership.mem t'.2.destruct (Option.some a)
      ht' : Membership.mem s.destruct (Option.some t')
      ht2 : Eq (Option.rec Stream'.WSeq.nil Prod.snd (Option.some t')) t
      ⊢ Exists fun a' => Membership.mem s.destruct (Option.some a')
    -/
  · exact ⟨_, ht'⟩
    /-
      🎉 no goals
    -/


theorem head_some_of_head_tail_some {s : WSeq α} {a} (h : some a ∈ head (tail s)) :
    ∃ a', some a' ∈ head s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a : α
    h : Membership.mem s.tail.head (Option.some a)
    ⊢ Exists fun a' => Membership.mem s.head (Option.some a')
  -/
  unfold head at h
  /-
    α : Type u
    s : Stream'.WSeq α
    a : α
    h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) s.tail.d …
    ⊢ Exists fun a' => Membership.mem s.head (Option.some a')
  -/
  rcases Computation.exists_of_mem_map h with ⟨o, md, e⟩; clear h
  /-
    case intro.intro
    α : Type u
    s : Stream'.WSeq α
    a : α
    o : Option (Prod α (Stream'.WSeq α))
    md : Membership.mem s.tail.destruct o
    e : Eq (Functor.map Prod.fst o) (Option.some a)
    ⊢ Exists fun a' => Membership.mem s.head (Option.some a')
  -/
  cases' o with o <;> [injection e; injection e with h']; clear h'
  /-
    case intro.intro.some
    α : Type u
    s : Stream'.WSeq α
    a : α
    o : Prod α (Stream'.WSeq α)
    md : Membership.mem s.tail.destruct (Option.some o)
    ⊢ Exists fun a' => Membership.mem s.head (Option.some a')
  -/
  cases' destruct_some_of_destruct_tail_some md with a am
  /-
    case intro.intro.some.intro
    α : Type u
    s : Stream'.WSeq α
    a✝ : α
    o : Prod α (Stream'.WSeq α)
    md : Membership.mem s.tail.destruct (Option.some o)
    a : Prod α (Stream'.WSeq α)
    am : Membership.mem s.destruct (Option.some a)
    ⊢ Exists fun a' => Membership.mem s.head (Option.some a')
  -/
  exact ⟨_, Computation.mem_map (@Prod.fst α (WSeq α) <$> ·) am⟩
  /-
    🎉 no goals
  -/


theorem head_some_of_get?_some {s : WSeq α} {a n} (h : some a ∈ get? s n) :
    ∃ a', some a' ∈ head s := by
  induction n generalizing a with
  | zero => exact ⟨_, h⟩
  | succ n IH =>
      let ⟨a', h'⟩ := head_some_of_head_tail_some h
      exact IH h'


instance productive_tail (s : WSeq α) [Productive s] : Productive (tail s) :=
               /-
                 α : Type u
                 β : Type v
                 γ : Type w
                 s : Stream'.WSeq α
                 inst✝ : s.Productive
                 n : Nat
                 ⊢ (s.tail.get? n).Terminates
               -/
  ⟨fun n => by rw [get?_tail]; infer_instance⟩
                               /-
                                 🎉 no goals
                               -/


instance productive_dropn (s : WSeq α) [Productive s] (n) : Productive (drop s n) :=
               /-
                 α : Type u
                 β : Type v
                 γ : Type w
                 s : Stream'.WSeq α
                 inst✝ : s.Productive
                 n m : Nat
                 ⊢ ((s.drop n).get? m).Terminates
               -/
  ⟨fun m => by rw [← get?_add]; infer_instance⟩
                                /-
                                  🎉 no goals
                                -/


/-- Given a productive weak sequence, we can collapse all the `think`s to
  produce a sequence. -/
def toSeq (s : WSeq α) [Productive s] : Seq α :=
  ⟨fun n => (get? s n).get,
   fun {n} h => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.WSeq α
      inst✝ : s.Productive
      n : Nat
      h : Eq ((fun n => (s.get? n).get) n) Option.none
      ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
    -/
    cases e : Computation.get (get? s (n + 1))
      /-
        case none
        α : Type u
        β : Type v
        γ : Type w
        s : Stream'.WSeq α
        inst✝ : s.Productive
        n : Nat
        h : Eq ((fun n => (s.get? n).get) n) Option.none
        e : Eq (s.get? (HAdd.hAdd n 1)).get Option.none
        ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
      -/
    · assumption
      /-
        🎉 no goals
      -/
    /-
      case some
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.WSeq α
      inst✝ : s.Productive
      n : Nat
      h : Eq ((fun n => (s.get? n).get) n) Option.none
      val✝ : α
      e : Eq (s.get? (HAdd.hAdd n 1)).get (Option.some val✝)
      ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
    -/
    have := Computation.mem_of_get_eq _ e
    /-
      case some
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.WSeq α
      inst✝ : s.Productive
      n : Nat
      h : Eq ((fun n => (s.get? n).get) n) Option.none
      val✝ : α
      e : Eq (s.get? (HAdd.hAdd n 1)).get (Option.some val✝)
      this : Membership.mem (s.get? (HAdd.hAdd n 1)) (Option.some val✝)
      ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
    -/
    simp? [get?] at this h says simp only [get?] at this h
    /-
      case some
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.WSeq α
      inst✝ : s.Productive
      n : Nat
      h : Eq (s.drop n).head.get Option.none
      val✝ : α
      e : Eq (s.get? (HAdd.hAdd n 1)).get (Option.some val✝)
      this : Membership.mem (s.drop (HAdd.hAdd n 1)).head (Option.some val✝)
      ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
    -/
    cases' head_some_of_head_tail_some this with a' h'
    /-
      case some.intro
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.WSeq α
      inst✝ : s.Productive
      n : Nat
      h : Eq (s.drop n).head.get Option.none
      val✝ : α
      e : Eq (s.get? (HAdd.hAdd n 1)).get (Option.some val✝)
      this : Membership.mem (s.drop (HAdd.hAdd n 1)).head (Option.some val✝)
      a' : α
      h' : Membership.mem (s.drop n).head (Option.some a')
      ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
    -/
    have := mem_unique h' (@Computation.mem_of_get_eq _ _ _ _ h)
    /-
      case some.intro
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.WSeq α
      inst✝ : s.Productive
      n : Nat
      h : Eq (s.drop n).head.get Option.none
      val✝ : α
      e : Eq (s.get? (HAdd.hAdd n 1)).get (Option.some val✝)
      this✝ : Membership.mem (s.drop (HAdd.hAdd n 1)).head (Option.some val✝)
      a' : α
      h' : Membership.mem (s.drop n).head (Option.some a')
      this : Eq (Option.some a') Option.none
      ⊢ Eq ((fun n => (s.get? n).get) (HAdd.hAdd n 1)) Option.none
    -/
    contradiction⟩
    /-
      🎉 no goals
    -/


theorem get?_terminates_le {s : WSeq α} {m n} (h : m ≤ n) :
    Terminates (get? s n) → Terminates (get? s m) := by
  /-
    α : Type u
    s : Stream'.WSeq α
    m n : Nat
    h : LE.le m n
    ⊢ (s.get? n).Terminates → (s.get? m).Terminates
  -/
  induction' h with m' _ IH
  /-
    case refl
    α : Type u
    s : Stream'.WSeq α
    m n : Nat
    ⊢ (s.get? m).Terminates → (s.get? m).Terminates
  -/
  exacts [id, fun T => IH (@head_terminates_of_head_tail_terminates _ _ T)]
  /-
    🎉 no goals
  -/


theorem head_terminates_of_get?_terminates {s : WSeq α} {n} :
    Terminates (get? s n) → Terminates (head s) :=
  get?_terminates_le (Nat.zero_le n)


theorem destruct_terminates_of_get?_terminates {s : WSeq α} {n} (T : Terminates (get? s n)) :
    Terminates (destruct s) :=
  (head_terminates_iff _).1 <| head_terminates_of_get?_terminates T


theorem mem_rec_on {C : WSeq α → Prop} {a s} (M : a ∈ s) (h1 : ∀ b s', a = b ∨ C s' → C (cons b s'))
    (h2 : ∀ s, C s → C (think s)) : C s := by
  /-
    α : Type u
    C : Stream'.WSeq α → Prop
    a : α
    s : Stream'.WSeq α
    M : Membership.mem s a
    h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
    h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
    ⊢ C s
  -/
  apply Seq.mem_rec_on M
  /-
    α : Type u
    C : Stream'.WSeq α → Prop
    a : α
    s : Stream'.WSeq α
    M : Membership.mem s a
    h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
    h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
    ⊢ ∀ (b : Option α) (s' : Stream'.Seq (Option α)), Or (Eq (Option.some a) b) (C …
  -/
  intro o s' h; cases' o with b
    /-
      case none
      α : Type u
      C : Stream'.WSeq α → Prop
      a : α
      s : Stream'.WSeq α
      M : Membership.mem s a
      h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
      h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
      s' : Stream'.Seq (Option α)
      h : Or (Eq (Option.some a) Option.none) (C s')
      ⊢ C (Stream'.Seq.cons Option.none s')
    -/
  · apply h2
    /-
      case none.a
      α : Type u
      C : Stream'.WSeq α → Prop
      a : α
      s : Stream'.WSeq α
      M : Membership.mem s a
      h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
      h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
      s' : Stream'.Seq (Option α)
      h : Or (Eq (Option.some a) Option.none) (C s')
      ⊢ C s'
    -/
    cases h
      /-
        case none.a.inl
        α : Type u
        C : Stream'.WSeq α → Prop
        a : α
        s : Stream'.WSeq α
        M : Membership.mem s a
        h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
        h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
        s' : Stream'.Seq (Option α)
        h✝ : Eq (Option.some a) Option.none
        ⊢ C s'
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case none.a.inr
        α : Type u
        C : Stream'.WSeq α → Prop
        a : α
        s : Stream'.WSeq α
        M : Membership.mem s a
        h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
        h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
        s' : Stream'.Seq (Option α)
        h✝ : C s'
        ⊢ C s'
      -/
    · assumption
      /-
        🎉 no goals
      -/
    /-
      case some
      α : Type u
      C : Stream'.WSeq α → Prop
      a : α
      s : Stream'.WSeq α
      M : Membership.mem s a
      h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
      h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
      s' : Stream'.Seq (Option α)
      b : α
      h : Or (Eq (Option.some a) (Option.some b)) (C s')
      ⊢ C (Stream'.Seq.cons (Option.some b) s')
    -/
  · apply h1
    /-
      case some.a
      α : Type u
      C : Stream'.WSeq α → Prop
      a : α
      s : Stream'.WSeq α
      M : Membership.mem s a
      h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
      h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
      s' : Stream'.Seq (Option α)
      b : α
      h : Or (Eq (Option.some a) (Option.some b)) (C s')
      ⊢ Or (Eq a b) (C s')
    -/
    apply Or.imp_left _ h
    /-
      α : Type u
      C : Stream'.WSeq α → Prop
      a : α
      s : Stream'.WSeq α
      M : Membership.mem s a
      h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
      h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
      s' : Stream'.Seq (Option α)
      b : α
      h : Or (Eq (Option.some a) (Option.some b)) (C s')
      ⊢ Eq (Option.some a) (Option.some b) → Eq a b
    -/
    intro h
    /-
      α : Type u
      C : Stream'.WSeq α → Prop
      a : α
      s : Stream'.WSeq α
      M : Membership.mem s a
      h1 : ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (C s') → C (Stream'.WSeq.con …
      h2 : ∀ (s : Stream'.WSeq α), C s → C s.think
      s' : Stream'.Seq (Option α)
      b : α
      h✝ : Or (Eq (Option.some a) (Option.some b)) (C s')
      h : Eq (Option.some a) (Option.some b)
      ⊢ Eq a b
    -/
    injection h
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_think (s : WSeq α) (a) : a ∈ think s ↔ a ∈ s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a : α
    ⊢ Iff (Membership.mem s.think a) (Membership.mem s a)
  -/
  cases' s with f al
  /-
    case mk
    α : Type u
    a : α
    f : Stream' (Option (Option α))
    al : f.IsSeq
    ⊢ Iff (Membership.mem (Stream'.WSeq.think ⟨f, al⟩) a) (Membership.mem ⟨f, al⟩ a)
  -/
  change (some (some a) ∈ some none::f) ↔ some (some a) ∈ f
  /-
    case mk
    α : Type u
    a : α
    f : Stream' (Option (Option α))
    al : f.IsSeq
    ⊢ Iff (Membership.mem (Stream'.cons (Option.some Option.none) f) (Option.some  …
  -/
  constructor <;> intro h
    /-
      case mk.mp
      α : Type u
      a : α
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem (Stream'.cons (Option.some Option.none) f) (Option.some (Op …
      ⊢ Membership.mem f (Option.some (Option.some a))
    -/
  · apply (Stream'.eq_or_mem_of_mem_cons h).resolve_left
    /-
      case mk.mp
      α : Type u
      a : α
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem (Stream'.cons (Option.some Option.none) f) (Option.some (Op …
      ⊢ Not (Eq (Option.some (Option.some a)) (Option.some Option.none))
    -/
    intro
    /-
      case mk.mp
      α : Type u
      a : α
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem (Stream'.cons (Option.some Option.none) f) (Option.some (Op …
      a✝ : Eq (Option.some (Option.some a)) (Option.some Option.none)
      ⊢ False
    -/
    injections
    /-
      🎉 no goals
    -/
    /-
      case mk.mpr
      α : Type u
      a : α
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem f (Option.some (Option.some a))
      ⊢ Membership.mem (Stream'.cons (Option.some Option.none) f) (Option.some (Opti …
    -/
  · apply Stream'.mem_cons_of_mem _ h
    /-
      🎉 no goals
    -/


theorem eq_or_mem_iff_mem {s : WSeq α} {a a' s'} :
    some (a', s') ∈ destruct s → (a ∈ s ↔ a = a' ∨ a ∈ s') := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a a' : α
    s' : Stream'.WSeq α
    ⊢ Membership.mem s.destruct (Option.some { fst := a', snd := s' }) → Iff (Memb …
  -/
  generalize e : destruct s = c; intro h
  /-
    α : Type u
    s : Stream'.WSeq α
    a a' : α
    s' : Stream'.WSeq α
    c : Computation (Option (Prod α (Stream'.WSeq α)))
    e : Eq s.destruct c
    h : Membership.mem c (Option.some { fst := a', snd := s' })
    ⊢ Iff (Membership.mem s a) (Or (Eq a a') (Membership.mem s' a))
  -/
  revert s
  /-
    α : Type u
    a a' : α
    s' : Stream'.WSeq α
    c : Computation (Option (Prod α (Stream'.WSeq α)))
    h : Membership.mem c (Option.some { fst := a', snd := s' })
    ⊢ ∀ {s : Stream'.WSeq α}, Eq s.destruct c → Iff (Membership.mem s a) (Or (Eq a …
  -/
  apply Computation.memRecOn h <;> [skip; intro c IH] <;> intro s <;>
    /-
      case h1
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      s : Stream'.WSeq α
      ⊢ Eq s.destruct (Computation.pure (Option.some { fst := a', snd := s' })) → If …
    -/
    induction' s using WSeq.recOn with x s s <;>
    /-
      case h1.h1
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      ⊢ Eq Stream'.WSeq.nil.destruct (Computation.pure (Option.some { fst := a', snd …
    -/
    intro m <;>
    /-
      case h1.h1
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      m : Eq Stream'.WSeq.nil.destruct (Computation.pure (Option.some { fst := a', s …
      ⊢ Iff (Membership.mem Stream'.WSeq.nil a) (Or (Eq a a') (Membership.mem s' a))
    -/
    have := congr_arg Computation.destruct m <;>
    /-
      case h1.h1
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      m : Eq Stream'.WSeq.nil.destruct (Computation.pure (Option.some { fst := a', s …
      this : Eq Stream'.WSeq.nil.destruct.destruct (Computation.pure (Option.some {  …
      ⊢ Iff (Membership.mem Stream'.WSeq.nil a) (Or (Eq a a') (Membership.mem s' a))
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
    /-
      🎉 no goals
    -/
    simp at this
    /-
      case h1.h2
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      x : α
      s : Stream'.WSeq α
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      this : And (Eq x a') (Eq s s')
      ⊢ Iff (Membership.mem (Stream'.WSeq.cons x s) a) (Or (Eq a a') (Membership.mem …
    -/
  · cases' this with i1 i2
    /-
      case h1.h2.intro
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      x : α
      s : Stream'.WSeq α
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      i1 : Eq x a'
      i2 : Eq s s'
      ⊢ Iff (Membership.mem (Stream'.WSeq.cons x s) a) (Or (Eq a a') (Membership.mem …
    -/
    rw [i1, i2]
    /-
      case h1.h2.intro
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c (Option.some { fst := a', snd := s' })
      x : α
      s : Stream'.WSeq α
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      i1 : Eq x a'
      i2 : Eq s s'
      ⊢ Iff (Membership.mem (Stream'.WSeq.cons a' s') a) (Or (Eq a a') (Membership.m …
    -/
    cases' s' with f al
    /-
      case h1.h2.intro.mk
      α : Type u
      a a' : α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      x : α
      s : Stream'.WSeq α
      i1 : Eq x a'
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      i2 : Eq s ⟨f, al⟩
      ⊢ Iff (Membership.mem (Stream'.WSeq.cons a' ⟨f, al⟩) a) (Or (Eq a a') (Members …
    -/
    dsimp only [cons, Membership.mem, WSeq.Mem, Seq.Mem, Seq.cons]
    /-
      case h1.h2.intro.mk
      α : Type u
      a a' : α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      x : α
      s : Stream'.WSeq α
      i1 : Eq x a'
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      i2 : Eq s ⟨f, al⟩
      ⊢ Iff (Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) (Stream'.cons …
    -/
    have h_a_eq_a' : a = a' ↔ some (some a) = some (some a') := by simp
    /-
      case h1.h2.intro.mk
      α : Type u
      a a' : α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      x : α
      s : Stream'.WSeq α
      i1 : Eq x a'
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      i2 : Eq s ⟨f, al⟩
      h_a_eq_a' : Iff (Eq a a') (Eq (Option.some (Option.some a)) (Option.some (Opti …
      ⊢ Iff (Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) (Stream'.cons …
    -/
    rw [h_a_eq_a']
    /-
      case h1.h2.intro.mk
      α : Type u
      a a' : α
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      x : α
      s : Stream'.WSeq α
      i1 : Eq x a'
      f : Stream' (Option (Option α))
      al : f.IsSeq
      h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
      m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
      i2 : Eq s ⟨f, al⟩
      h_a_eq_a' : Iff (Eq a a') (Eq (Option.some (Option.some a)) (Option.some (Opti …
      ⊢ Iff (Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) (Stream'.cons …
    -/
    refine ⟨Stream'.eq_or_mem_of_mem_cons, fun o => ?_⟩
      /-
        case h1.h2.intro.mk
        α : Type u
        a a' : α
        c : Computation (Option (Prod α (Stream'.WSeq α)))
        x : α
        s : Stream'.WSeq α
        i1 : Eq x a'
        f : Stream' (Option (Option α))
        al : f.IsSeq
        h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
        m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
        i2 : Eq s ⟨f, al⟩
        h_a_eq_a' : Iff (Eq a a') (Eq (Option.some (Option.some a)) (Option.some (Opti …
        o : Or (Eq (Option.some (Option.some a)) (Option.some (Option.some a'))) (Stre …
        ⊢ Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) (Stream'.cons (Opt …
      -/
    · cases' o with e m
        /-
          case h1.h2.intro.mk.inl
          α : Type u
          a a' : α
          c : Computation (Option (Prod α (Stream'.WSeq α)))
          x : α
          s : Stream'.WSeq α
          i1 : Eq x a'
          f : Stream' (Option (Option α))
          al : f.IsSeq
          h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
          m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
          i2 : Eq s ⟨f, al⟩
          h_a_eq_a' : Iff (Eq a a') (Eq (Option.some (Option.some a)) (Option.some (Opti …
          e : Eq (Option.some (Option.some a)) (Option.some (Option.some a'))
          ⊢ Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) (Stream'.cons (Opt …
        -/
      · rw [e]
        /-
          case h1.h2.intro.mk.inl
          α : Type u
          a a' : α
          c : Computation (Option (Prod α (Stream'.WSeq α)))
          x : α
          s : Stream'.WSeq α
          i1 : Eq x a'
          f : Stream' (Option (Option α))
          al : f.IsSeq
          h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
          m : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst : …
          i2 : Eq s ⟨f, al⟩
          h_a_eq_a' : Iff (Eq a a') (Eq (Option.some (Option.some a)) (Option.some (Opti …
          e : Eq (Option.some (Option.some a)) (Option.some (Option.some a'))
          ⊢ Stream'.Any (fun b => Eq (Option.some (Option.some a')) b) (Stream'.cons (Op …
        -/
        apply Stream'.mem_cons
        /-
          🎉 no goals
        -/
        /-
          case h1.h2.intro.mk.inr
          α : Type u
          a a' : α
          c : Computation (Option (Prod α (Stream'.WSeq α)))
          x : α
          s : Stream'.WSeq α
          i1 : Eq x a'
          f : Stream' (Option (Option α))
          al : f.IsSeq
          h : Membership.mem c (Option.some { fst := a', snd := ⟨f, al⟩ })
          m✝ : Eq (Stream'.WSeq.cons x s).destruct (Computation.pure (Option.some { fst  …
          i2 : Eq s ⟨f, al⟩
          h_a_eq_a' : Iff (Eq a a') (Eq (Option.some (Option.some a)) (Option.some (Opti …
          m : Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) f
          ⊢ Stream'.Any (fun b => Eq (Option.some (Option.some a)) b) (Stream'.cons (Opt …
        -/
      · exact Stream'.mem_cons_of_mem _ m
        /-
          🎉 no goals
        -/
    /-
      case h2.h3
      α : Type u
      a a' : α
      s' : Stream'.WSeq α
      c✝ : Computation (Option (Prod α (Stream'.WSeq α)))
      h : Membership.mem c✝ (Option.some { fst := a', snd := s' })
      c : Computation (Option (Prod α (Stream'.WSeq α)))
      IH : ∀ {s : Stream'.WSeq α}, Eq s.destruct c → Iff (Membership.mem s a) (Or (E …
      s : Stream'.WSeq α
      m : Eq s.think.destruct c.think
      this : Eq s.destruct c
      ⊢ Iff (Membership.mem s.think a) (Or (Eq a a') (Membership.mem s' a))
    -/
  · simp [IH this]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_cons_iff (s : WSeq α) (b) {a} : a ∈ cons b s ↔ a = b ∨ a ∈ s :=
                          /-
                            α : Type u
                            s : Stream'.WSeq α
                            b a : α
                            ⊢ Membership.mem (Stream'.WSeq.cons b s).destruct (Option.some { fst := b, snd …
                          -/
  eq_or_mem_iff_mem <| by simp [ret_mem]
                          /-
                            🎉 no goals
                          -/


theorem mem_cons_of_mem {s : WSeq α} (b) {a} (h : a ∈ s) : a ∈ cons b s :=
  (mem_cons_iff _ _).2 (Or.inr h)


theorem mem_cons (s : WSeq α) (a) : a ∈ cons a s :=
  (mem_cons_iff _ _).2 (Or.inl rfl)


theorem mem_of_mem_tail {s : WSeq α} {a} : a ∈ tail s → a ∈ s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a : α
    ⊢ Membership.mem s.tail a → Membership.mem s a
  -/
  intro h; have := h; cases' h with n e; revert s; simp only [Stream'.get]
  /-
    case intro
    α : Type u
    a : α
    n : Nat
    ⊢ ∀ {s : Stream'.WSeq α}, Membership.mem s.tail a → Eq (Option.some (Option.so …
  -/
  induction' n with n IH <;> intro s <;> induction' s using WSeq.recOn with x s s <;>
    /-
      case intro.zero.h1
      α : Type u
      a : α
      ⊢ Membership.mem Stream'.WSeq.nil.tail a → Eq (Option.some (Option.some a)) (↑ …
    -/
    simp <;> intro m e <;>
    /-
      case intro.zero.h1
      α : Type u
      a : α
      m : Membership.mem Stream'.WSeq.nil a
      e : Eq (Option.some (Option.some a)) (↑Stream'.WSeq.nil 0)
      ⊢ Membership.mem Stream'.WSeq.nil a
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
    injections
    /-
      case intro.zero.h2
      α : Type u
      a x : α
      s : Stream'.WSeq α
      m : Membership.mem s a
      e : Eq (Option.some (Option.some a)) (↑s 0)
      ⊢ Or (Eq a x) (Membership.mem s a)
    -/
  · exact Or.inr m
    /-
      🎉 no goals
    -/
    /-
      case intro.succ.h2
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem s.tail a → Eq (Option.some (Option …
      x : α
      s : Stream'.WSeq α
      m : Membership.mem s a
      e : Eq (Option.some (Option.some a)) (↑s (HAdd.hAdd n 1))
      ⊢ Or (Eq a x) (Membership.mem s a)
    -/
  · exact Or.inr m
    /-
      🎉 no goals
    -/
    /-
      case intro.succ.h3
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem s.tail a → Eq (Option.some (Option …
      s : Stream'.WSeq α
      m : Membership.mem s.tail a
      e : Eq (Option.some (Option.some a)) (↑s.tail.think (HAdd.hAdd n 1))
      ⊢ Membership.mem s a
    -/
  · apply IH m
    /-
      case intro.succ.h3
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem s.tail a → Eq (Option.some (Option …
      s : Stream'.WSeq α
      m : Membership.mem s.tail a
      e : Eq (Option.some (Option.some a)) (↑s.tail.think (HAdd.hAdd n 1))
      ⊢ Eq (Option.some (Option.some a)) (↑s.tail n)
    -/
    rw [e]
    /-
      case intro.succ.h3
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem s.tail a → Eq (Option.some (Option …
      s : Stream'.WSeq α
      m : Membership.mem s.tail a
      e : Eq (Option.some (Option.some a)) (↑s.tail.think (HAdd.hAdd n 1))
      ⊢ Eq (↑s.tail.think (HAdd.hAdd n 1)) (↑s.tail n)
    -/
    cases tail s
    /-
      case intro.succ.h3.mk
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem s.tail a → Eq (Option.some (Option …
      s : Stream'.WSeq α
      m : Membership.mem s.tail a
      e : Eq (Option.some (Option.some a)) (↑s.tail.think (HAdd.hAdd n 1))
      val✝ : Stream' (Option (Option α))
      property✝ : val✝.IsSeq
      ⊢ Eq (↑(Stream'.WSeq.think ⟨val✝, property✝⟩) (HAdd.hAdd n 1)) (↑⟨val✝, proper …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem mem_of_mem_dropn {s : WSeq α} {a} : ∀ {n}, a ∈ drop s n → a ∈ s
  | 0, h => h
  | n + 1, h => @mem_of_mem_dropn s a n (mem_of_mem_tail h)


theorem get?_mem {s : WSeq α} {a n} : some a ∈ get? s n → a ∈ s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a : α
    n : Nat
    ⊢ Membership.mem (s.get? n) (Option.some a) → Membership.mem s a
  -/
  revert s; induction' n with n IH <;> intro s h
  · -- Porting note: This line is required to infer metavariables in
    --               `Computation.exists_of_mem_map`.
    /-
      case zero
      α : Type u
      a : α
      s : Stream'.WSeq α
      h : Membership.mem (s.get? 0) (Option.some a)
      ⊢ Membership.mem s a
    -/
    dsimp only [get?, head] at h
    /-
      case zero
      α : Type u
      a : α
      s : Stream'.WSeq α
      h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
      ⊢ Membership.mem s a
    -/
    rcases Computation.exists_of_mem_map h with ⟨o, h1, h2⟩
    /-
      case zero.intro.intro
      α : Type u
      a : α
      s : Stream'.WSeq α
      h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
      o : Option (Prod α (Stream'.WSeq α))
      h1 : Membership.mem (s.drop 0).destruct o
      h2 : Eq (Functor.map Prod.fst o) (Option.some a)
      ⊢ Membership.mem s a
    -/
    cases' o with o
      /-
        case zero.intro.intro.none
        α : Type u
        a : α
        s : Stream'.WSeq α
        h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
        h1 : Membership.mem (s.drop 0).destruct Option.none
        h2 : Eq (Functor.map Prod.fst Option.none) (Option.some a)
        ⊢ Membership.mem s a
      -/
    · injection h2
      /-
        🎉 no goals
      -/
    /-
      case zero.intro.intro.some
      α : Type u
      a : α
      s : Stream'.WSeq α
      h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
      o : Prod α (Stream'.WSeq α)
      h1 : Membership.mem (s.drop 0).destruct (Option.some o)
      h2 : Eq (Functor.map Prod.fst (Option.some o)) (Option.some a)
      ⊢ Membership.mem s a
    -/
    injection h2 with h'
    /-
      case zero.intro.intro.some
      α : Type u
      a : α
      s : Stream'.WSeq α
      h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
      o : Prod α (Stream'.WSeq α)
      h1 : Membership.mem (s.drop 0).destruct (Option.some o)
      h' : Eq o.1 a
      ⊢ Membership.mem s a
    -/
    cases' o with a' s'
    /-
      case zero.intro.intro.some.mk
      α : Type u
      a : α
      s : Stream'.WSeq α
      h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
      a' : α
      s' : Stream'.WSeq α
      h1 : Membership.mem (s.drop 0).destruct (Option.some { fst := a', snd := s' })
      h' : Eq { fst := a', snd := s' }.1 a
      ⊢ Membership.mem s a
    -/
    exact (eq_or_mem_iff_mem h1).2 (Or.inl h'.symm)
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem (s.get? n) (Option.some a) → Membe …
      s : Stream'.WSeq α
      h : Membership.mem (s.get? (HAdd.hAdd n 1)) (Option.some a)
      ⊢ Membership.mem s a
    -/
  · have := @IH (tail s)
    /-
      case succ
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem (s.get? n) (Option.some a) → Membe …
      s : Stream'.WSeq α
      h : Membership.mem (s.get? (HAdd.hAdd n 1)) (Option.some a)
      this : Membership.mem (s.tail.get? n) (Option.some a) → Membership.mem s.tail a
      ⊢ Membership.mem s a
    -/
    rw [get?_tail] at this
    /-
      case succ
      α : Type u
      a : α
      n : Nat
      IH : ∀ {s : Stream'.WSeq α}, Membership.mem (s.get? n) (Option.some a) → Membe …
      s : Stream'.WSeq α
      h : Membership.mem (s.get? (HAdd.hAdd n 1)) (Option.some a)
      this : Membership.mem (s.get? (HAdd.hAdd n 1)) (Option.some a) → Membership.me …
      ⊢ Membership.mem s a
    -/
    exact mem_of_mem_tail (this h)
    /-
      🎉 no goals
    -/


theorem exists_get?_of_mem {s : WSeq α} {a} (h : a ∈ s) : ∃ n, some a ∈ get? s n := by
  /-
    α : Type u
    s : Stream'.WSeq α
    a : α
    h : Membership.mem s a
    ⊢ Exists fun n => Membership.mem (s.get? n) (Option.some a)
  -/
  apply mem_rec_on h
    /-
      case h1
      α : Type u
      s : Stream'.WSeq α
      a : α
      h : Membership.mem s a
      ⊢ ∀ (b : α) (s' : Stream'.WSeq α), Or (Eq a b) (Exists fun n => Membership.mem …
    -/
  · intro a' s' h
    /-
      case h1
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      a' : α
      s' : Stream'.WSeq α
      h : Or (Eq a a') (Exists fun n => Membership.mem (s'.get? n) (Option.some a))
      ⊢ Exists fun n => Membership.mem ((Stream'.WSeq.cons a' s').get? n) (Option.so …
    -/
    cases' h with h h
      /-
        case h1.inl
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        h : Eq a a'
        ⊢ Exists fun n => Membership.mem ((Stream'.WSeq.cons a' s').get? n) (Option.so …
      -/
    · exists 0
      /-
        case h1.inl
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        h : Eq a a'
        ⊢ Membership.mem ((Stream'.WSeq.cons a' s').get? 0) (Option.some a)
      -/
      simp only [get?, drop, head_cons]
      /-
        case h1.inl
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        h : Eq a a'
        ⊢ Membership.mem (Computation.pure (Option.some a')) (Option.some a)
      -/
      rw [h]
      /-
        case h1.inl
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        h : Eq a a'
        ⊢ Membership.mem (Computation.pure (Option.some a')) (Option.some a')
      -/
      apply ret_mem
      /-
        🎉 no goals
      -/
      /-
        case h1.inr
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        h : Exists fun n => Membership.mem (s'.get? n) (Option.some a)
        ⊢ Exists fun n => Membership.mem ((Stream'.WSeq.cons a' s').get? n) (Option.so …
      -/
    · cases' h with n h
      /-
        case h1.inr.intro
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        n : Nat
        h : Membership.mem (s'.get? n) (Option.some a)
        ⊢ Exists fun n => Membership.mem ((Stream'.WSeq.cons a' s').get? n) (Option.so …
      -/
      exists n + 1
      /-
        case h1.inr.intro
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        a' : α
        s' : Stream'.WSeq α
        n : Nat
        h : Membership.mem (s'.get? n) (Option.some a)
        ⊢ Membership.mem ((Stream'.WSeq.cons a' s').get? (HAdd.hAdd n 1)) (Option.some …
      -/
      simpa [get?]
      /-
        🎉 no goals
      -/
    /-
      case h2
      α : Type u
      s : Stream'.WSeq α
      a : α
      h : Membership.mem s a
      ⊢ ∀ (s : Stream'.WSeq α), (Exists fun n => Membership.mem (s.get? n) (Option.s …
    -/
  · intro s' h
    /-
      case h2
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      s' : Stream'.WSeq α
      h : Exists fun n => Membership.mem (s'.get? n) (Option.some a)
      ⊢ Exists fun n => Membership.mem (s'.think.get? n) (Option.some a)
    -/
    cases' h with n h
    /-
      case h2.intro
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      s' : Stream'.WSeq α
      n : Nat
      h : Membership.mem (s'.get? n) (Option.some a)
      ⊢ Exists fun n => Membership.mem (s'.think.get? n) (Option.some a)
    -/
    exists n
    /-
      case h2.intro
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      s' : Stream'.WSeq α
      n : Nat
      h : Membership.mem (s'.get? n) (Option.some a)
      ⊢ Membership.mem (s'.think.get? n) (Option.some a)
    -/
    simp only [get?, dropn_think, head_think]
    /-
      case h2.intro
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      s' : Stream'.WSeq α
      n : Nat
      h : Membership.mem (s'.get? n) (Option.some a)
      ⊢ Membership.mem (s'.drop n).head.think (Option.some a)
    -/
    apply think_mem h
    /-
      🎉 no goals
    -/


theorem exists_dropn_of_mem {s : WSeq α} {a} (h : a ∈ s) :
    ∃ n s', some (a, s') ∈ destruct (drop s n) :=
  let ⟨n, h⟩ := exists_get?_of_mem h
  ⟨n, by
    /-
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    rcases (head_terminates_iff _).1 ⟨⟨_, h⟩⟩ with ⟨⟨o, om⟩⟩
    /-
      case mk.intro
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      o : Option (Prod α (Stream'.WSeq α))
      om : Membership.mem (s.drop n).destruct o
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    have := Computation.mem_unique (Computation.mem_map _ om) h
    /-
      case mk.intro
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      o : Option (Prod α (Stream'.WSeq α))
      om : Membership.mem (s.drop n).destruct o
      this : Eq (Functor.map Prod.fst o) (Option.some a)
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    cases' o with o
      /-
        case mk.intro.none
        α : Type u
        s : Stream'.WSeq α
        a : α
        h✝ : Membership.mem s a
        n : Nat
        h : Membership.mem (s.get? n) (Option.some a)
        om : Membership.mem (s.drop n).destruct Option.none
        this : Eq (Functor.map Prod.fst Option.none) (Option.some a)
        ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
      -/
    · injection this
      /-
        🎉 no goals
      -/
    /-
      case mk.intro.some
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      o : Prod α (Stream'.WSeq α)
      om : Membership.mem (s.drop n).destruct (Option.some o)
      this : Eq (Functor.map Prod.fst (Option.some o)) (Option.some a)
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    injection this with i
    /-
      case mk.intro.some
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      o : Prod α (Stream'.WSeq α)
      om : Membership.mem (s.drop n).destruct (Option.some o)
      i : Eq o.1 a
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    cases' o with a' s'
    /-
      case mk.intro.some.mk
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      a' : α
      s' : Stream'.WSeq α
      om : Membership.mem (s.drop n).destruct (Option.some { fst := a', snd := s' })
      i : Eq { fst := a', snd := s' }.1 a
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    dsimp at i
    /-
      case mk.intro.some.mk
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      a' : α
      s' : Stream'.WSeq α
      om : Membership.mem (s.drop n).destruct (Option.some { fst := a', snd := s' })
      i : Eq a' a
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    rw [i] at om
    /-
      case mk.intro.some.mk
      α : Type u
      s : Stream'.WSeq α
      a : α
      h✝ : Membership.mem s a
      n : Nat
      h : Membership.mem (s.get? n) (Option.some a)
      a' : α
      s' : Stream'.WSeq α
      om : Membership.mem (s.drop n).destruct (Option.some { fst := a, snd := s' })
      i : Eq a' a
      ⊢ Exists fun s' => Membership.mem (s.drop n).destruct (Option.some { fst := a, …
    -/
    exact ⟨_, om⟩⟩
    /-
      🎉 no goals
    -/


theorem liftRel_dropn_destruct {R : α → β → Prop} {s t} (H : LiftRel R s t) :
    ∀ n, Computation.LiftRel (LiftRelO R (LiftRel R)) (destruct (drop s n)) (destruct (drop t n))
  | 0 => liftRel_destruct H
  | n + 1 => by
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      H : Stream'.WSeq.LiftRel R s t
      n : Nat
      ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) (s.dr …
    -/
    simp only [LiftRelO, drop, Nat.add_eq, Nat.add_zero, destruct_tail, tail.aux]
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      H : Stream'.WSeq.LiftRel R s t
      n : Nat
      ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) (Bind …
    -/
    apply liftRel_bind
      /-
        case h1
        α : Type u
        β : Type v
        R : α → β → Prop
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        H : Stream'.WSeq.LiftRel R s t
        n : Nat
        ⊢ Computation.LiftRel ?R (s.drop n).destruct (t.drop n).destruct
      -/
    · apply liftRel_dropn_destruct H n
      /-
        🎉 no goals
      -/
    exact fun {a b} o =>
      match a, b, o with
      | none, none, _ => by
        -- Porting note: These 2 theorems should be excluded.
        simp [-liftRel_pure_left, -liftRel_pure_right]
      | some (a, s), some (b, t), ⟨_, h2⟩ => by simpa [tail.aux] using liftRel_destruct h2


theorem exists_of_liftRel_left {R : α → β → Prop} {s t} (H : LiftRel R s t) {a} (h : a ∈ s) :
    ∃ b, b ∈ t ∧ R a b := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    H : Stream'.WSeq.LiftRel R s t
    a : α
    h : Membership.mem s a
    ⊢ Exists fun b => And (Membership.mem t b) (R a b)
  -/
  let ⟨n, h⟩ := exists_get?_of_mem h
  -- Porting note: This line is required to infer metavariables in
  --               `Computation.exists_of_mem_map`.
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    H : Stream'.WSeq.LiftRel R s t
    a : α
    h✝ : Membership.mem s a
    n : Nat
    h : Membership.mem (s.get? n) (Option.some a)
    ⊢ Exists fun b => And (Membership.mem t b) (R a b)
  -/
  dsimp only [get?, head] at h
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    H : Stream'.WSeq.LiftRel R s t
    a : α
    h✝ : Membership.mem s a
    n : Nat
    h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
    ⊢ Exists fun b => And (Membership.mem t b) (R a b)
  -/
  let ⟨some (_, s'), sd, rfl⟩ := Computation.exists_of_mem_map h
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    H : Stream'.WSeq.LiftRel R s t
    a : α
    h✝ : Membership.mem s a
    n : Nat
    h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
    s' : Stream'.WSeq α
    sd : Membership.mem (s.drop n).destruct (Option.some { fst := a, snd := s' })
    ⊢ Exists fun b => And (Membership.mem t b) (R a b)
  -/
  let ⟨some (b, t'), td, ⟨ab, _⟩⟩ := (liftRel_dropn_destruct H n).left sd
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    H : Stream'.WSeq.LiftRel R s t
    a : α
    h✝ : Membership.mem s a
    n : Nat
    h : Membership.mem (Computation.map (fun x => Functor.map Prod.fst x) (s.drop  …
    s' : Stream'.WSeq α
    sd : Membership.mem (s.drop n).destruct (Option.some { fst := a, snd := s' })
    b : β
    t' : Stream'.WSeq β
    td : Membership.mem (t.drop n).destruct (Option.some { fst := b, snd := t' })
    ab : R a b
    right✝ : Stream'.WSeq.LiftRel R s' t'
    ⊢ Exists fun b => And (Membership.mem t b) (R a b)
  -/
  exact ⟨b, get?_mem (Computation.mem_map (Prod.fst.{v, v} <$> ·) td), ab⟩
  /-
    🎉 no goals
  -/


theorem exists_of_liftRel_right {R : α → β → Prop} {s t} (H : LiftRel R s t) {b} (h : b ∈ t) :
                             /-
                               α : Type u
                               β : Type v
                               R : α → β → Prop
                               s : Stream'.WSeq α
                               t : Stream'.WSeq β
                               H : Stream'.WSeq.LiftRel R s t
                               b : β
                               h : Membership.mem t b
                               ⊢ Exists fun a => And (Membership.mem s a) (R a b)
                             -/
    ∃ a, a ∈ s ∧ R a b := by rw [← LiftRel.swap] at H; exact exists_of_liftRel_left H h
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem head_terminates_of_mem {s : WSeq α} {a} (h : a ∈ s) : Terminates (head s) :=
  let ⟨_, h⟩ := exists_get?_of_mem h
  head_terminates_of_get?_terminates ⟨⟨_, h⟩⟩


theorem of_mem_append {s₁ s₂ : WSeq α} {a : α} : a ∈ append s₁ s₂ → a ∈ s₁ ∨ a ∈ s₂ :=
  Seq.of_mem_append


theorem mem_append_left {s₁ s₂ : WSeq α} {a : α} : a ∈ s₁ → a ∈ append s₁ s₂ :=
  Seq.mem_append_left


theorem exists_of_mem_map {f} {b : β} : ∀ {s : WSeq α}, b ∈ map f s → ∃ a, a ∈ s ∧ f a = b
  | ⟨g, al⟩, h => by
    /-
      α : Type u
      β : Type v
      f : α → β
      b : β
      g : Stream' (Option (Option α))
      al : g.IsSeq
      h : Membership.mem (Stream'.WSeq.map f ⟨g, al⟩) b
      ⊢ Exists fun a => And (Membership.mem ⟨g, al⟩ a) (Eq (f a) b)
    -/
    let ⟨o, om, oe⟩ := Seq.exists_of_mem_map h
    /-
      α : Type u
      β : Type v
      f : α → β
      b : β
      g : Stream' (Option (Option α))
      al : g.IsSeq
      h : Membership.mem (Stream'.WSeq.map f ⟨g, al⟩) b
      o : Option α
      om : Membership.mem ⟨g, al⟩ o
      oe : Eq (Option.map f o) (Option.some b)
      ⊢ Exists fun a => And (Membership.mem ⟨g, al⟩ a) (Eq (f a) b)
    -/
    cases' o with a
      /-
        case none
        α : Type u
        β : Type v
        f : α → β
        b : β
        g : Stream' (Option (Option α))
        al : g.IsSeq
        h : Membership.mem (Stream'.WSeq.map f ⟨g, al⟩) b
        om : Membership.mem ⟨g, al⟩ Option.none
        oe : Eq (Option.map f Option.none) (Option.some b)
        ⊢ Exists fun a => And (Membership.mem ⟨g, al⟩ a) (Eq (f a) b)
      -/
    · injection oe
      /-
        🎉 no goals
      -/
    /-
      case some
      α : Type u
      β : Type v
      f : α → β
      b : β
      g : Stream' (Option (Option α))
      al : g.IsSeq
      h : Membership.mem (Stream'.WSeq.map f ⟨g, al⟩) b
      a : α
      om : Membership.mem ⟨g, al⟩ (Option.some a)
      oe : Eq (Option.map f (Option.some a)) (Option.some b)
      ⊢ Exists fun a => And (Membership.mem ⟨g, al⟩ a) (Eq (f a) b)
    -/
    injection oe with h'
    /-
      case some
      α : Type u
      β : Type v
      f : α → β
      b : β
      g : Stream' (Option (Option α))
      al : g.IsSeq
      h : Membership.mem (Stream'.WSeq.map f ⟨g, al⟩) b
      a : α
      om : Membership.mem ⟨g, al⟩ (Option.some a)
      h' : Eq (f a) b
      ⊢ Exists fun a => And (Membership.mem ⟨g, al⟩ a) (Eq (f a) b)
    -/
    exact ⟨a, om, h'⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem liftRel_nil (R : α → β → Prop) : LiftRel R nil nil := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    ⊢ Stream'.WSeq.LiftRel R Stream'.WSeq.nil Stream'.WSeq.nil
  -/
  rw [liftRel_destruct_iff]
  -- Porting note: These 2 theorems should be excluded.
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R)) Strea …
  -/
  simp [-liftRel_pure_left, -liftRel_pure_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftRel_cons (R : α → β → Prop) (a b s t) :
    LiftRel R (cons a s) (cons b t) ↔ R a b ∧ LiftRel R s t := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    a : α
    b : β
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    ⊢ Iff (Stream'.WSeq.LiftRel R (Stream'.WSeq.cons a s) (Stream'.WSeq.cons b t)) …
  -/
  rw [liftRel_destruct_iff]
  -- Porting note: These 2 theorems should be excluded.
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    a : α
    b : β
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    ⊢ Iff (Computation.LiftRel (Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R))  …
  -/
  simp [-liftRel_pure_left, -liftRel_pure_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftRel_think_left (R : α → β → Prop) (s t) : LiftRel R (think s) t ↔ LiftRel R s t := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    ⊢ Iff (Stream'.WSeq.LiftRel R s.think t) (Stream'.WSeq.LiftRel R s t)
  -/
  rw [liftRel_destruct_iff, liftRel_destruct_iff]; simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem liftRel_think_right (R : α → β → Prop) (s t) : LiftRel R s (think t) ↔ LiftRel R s t := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    s : Stream'.WSeq α
    t : Stream'.WSeq β
    ⊢ Iff (Stream'.WSeq.LiftRel R s t.think) (Stream'.WSeq.LiftRel R s t)
  -/
  rw [liftRel_destruct_iff, liftRel_destruct_iff]; simp
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem cons_congr {s t : WSeq α} (a : α) (h : s ~ʷ t) : cons a s ~ʷ cons a t := by
  /-
    α : Type u
    s t : Stream'.WSeq α
    a : α
    h : s.Equiv t
    ⊢ (Stream'.WSeq.cons a s).Equiv (Stream'.WSeq.cons a t)
  -/
  unfold Equiv; simpa using h
                /-
                  🎉 no goals
                -/


                                                      /-
                                                        α : Type u
                                                        s : Stream'.WSeq α
                                                        ⊢ s.think.Equiv s
                                                      -/
theorem think_equiv (s : WSeq α) : think s ~ʷ s := by unfold Equiv; simpa using Equiv.refl _
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem think_congr {s t : WSeq α} (h : s ~ʷ t) : think s ~ʷ think t := by
  /-
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    ⊢ s.think.Equiv t.think
  -/
  unfold Equiv; simpa using h
                /-
                  🎉 no goals
                -/


theorem head_congr : ∀ {s t : WSeq α}, s ~ʷ t → head s ~ head t := by
  suffices ∀ {s t : WSeq α}, s ~ʷ t → ∀ {o}, o ∈ head s → o ∈ head t from fun s t h o =>
    ⟨this h, this h.symm⟩
  /-
    α : Type u
    ⊢ ∀ {s t : Stream'.WSeq α}, s.Equiv t → ∀ {o : Option α}, Membership.mem s.hea …
  -/
  intro s t h o ho
  /-
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    o : Option α
    ho : Membership.mem s.head o
    ⊢ Membership.mem t.head o
  -/
  rcases @Computation.exists_of_mem_map _ _ _ _ (destruct s) ho with ⟨ds, dsm, dse⟩
  /-
    case intro.intro
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    o : Option α
    ho : Membership.mem s.head o
    ds : Option (Prod α (Stream'.WSeq α))
    dsm : Membership.mem s.destruct ds
    dse : Eq (Functor.map Prod.fst ds) o
    ⊢ Membership.mem t.head o
  -/
  rw [← dse]
  /-
    case intro.intro
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    o : Option α
    ho : Membership.mem s.head o
    ds : Option (Prod α (Stream'.WSeq α))
    dsm : Membership.mem s.destruct ds
    dse : Eq (Functor.map Prod.fst ds) o
    ⊢ Membership.mem t.head (Functor.map Prod.fst ds)
  -/
  cases' destruct_congr h with l r
  /-
    case intro.intro.intro
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    o : Option α
    ho : Membership.mem s.head o
    ds : Option (Prod α (Stream'.WSeq α))
    dsm : Membership.mem s.destruct ds
    dse : Eq (Functor.map Prod.fst ds) o
    l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
    r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
    ⊢ Membership.mem t.head (Functor.map Prod.fst ds)
  -/
  rcases l dsm with ⟨dt, dtm, dst⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    o : Option α
    ho : Membership.mem s.head o
    ds : Option (Prod α (Stream'.WSeq α))
    dsm : Membership.mem s.destruct ds
    dse : Eq (Functor.map Prod.fst ds) o
    l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
    r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
    dt : Option (Prod α (Stream'.WSeq α))
    dtm : Membership.mem t.destruct dt
    dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) ds dt
    ⊢ Membership.mem t.head (Functor.map Prod.fst ds)
  -/
  cases' ds with a <;> cases' dt with b
    /-
      case intro.intro.intro.intro.intro.none.none
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      dsm : Membership.mem s.destruct Option.none
      dse : Eq (Functor.map Prod.fst Option.none) o
      dtm : Membership.mem t.destruct Option.none
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) Option.none Option.none
      ⊢ Membership.mem t.head (Functor.map Prod.fst Option.none)
    -/
  · apply Computation.mem_map _ dtm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.none.some
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      dsm : Membership.mem s.destruct Option.none
      dse : Eq (Functor.map Prod.fst Option.none) o
      b : Prod α (Stream'.WSeq α)
      dtm : Membership.mem t.destruct (Option.some b)
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) Option.none (Option.some b)
      ⊢ Membership.mem t.head (Functor.map Prod.fst Option.none)
    -/
  · cases b
    /-
      case intro.intro.intro.intro.intro.none.some.mk
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      dsm : Membership.mem s.destruct Option.none
      dse : Eq (Functor.map Prod.fst Option.none) o
      fst✝ : α
      snd✝ : Stream'.WSeq α
      dtm : Membership.mem t.destruct (Option.some { fst := fst✝, snd := snd✝ })
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) Option.none (Option.some  …
      ⊢ Membership.mem t.head (Functor.map Prod.fst Option.none)
    -/
    cases dst
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.some.none
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      a : Prod α (Stream'.WSeq α)
      dsm : Membership.mem s.destruct (Option.some a)
      dse : Eq (Functor.map Prod.fst (Option.some a)) o
      dtm : Membership.mem t.destruct Option.none
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some a) Option.none
      ⊢ Membership.mem t.head (Functor.map Prod.fst (Option.some a))
    -/
  · cases a
    /-
      case intro.intro.intro.intro.intro.some.none.mk
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      dtm : Membership.mem t.destruct Option.none
      fst✝ : α
      snd✝ : Stream'.WSeq α
      dsm : Membership.mem s.destruct (Option.some { fst := fst✝, snd := snd✝ })
      dse : Eq (Functor.map Prod.fst (Option.some { fst := fst✝, snd := snd✝ })) o
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some { fst := fst …
      ⊢ Membership.mem t.head (Functor.map Prod.fst (Option.some { fst := fst✝, snd  …
    -/
    cases dst
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.some.some
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      a : Prod α (Stream'.WSeq α)
      dsm : Membership.mem s.destruct (Option.some a)
      dse : Eq (Functor.map Prod.fst (Option.some a)) o
      b : Prod α (Stream'.WSeq α)
      dtm : Membership.mem t.destruct (Option.some b)
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some a) (Option.s …
      ⊢ Membership.mem t.head (Functor.map Prod.fst (Option.some a))
    -/
  · cases' a with a s'
    /-
      case intro.intro.intro.intro.intro.some.some.mk
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      b : Prod α (Stream'.WSeq α)
      dtm : Membership.mem t.destruct (Option.some b)
      a : α
      s' : Stream'.WSeq α
      dsm : Membership.mem s.destruct (Option.some { fst := a, snd := s' })
      dse : Eq (Functor.map Prod.fst (Option.some { fst := a, snd := s' })) o
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some { fst := a,  …
      ⊢ Membership.mem t.head (Functor.map Prod.fst (Option.some { fst := a, snd :=  …
    -/
    cases' b with b t'
    /-
      case intro.intro.intro.intro.intro.some.some.mk.mk
      α : Type u
      s t : Stream'.WSeq α
      h : s.Equiv t
      o : Option α
      ho : Membership.mem s.head o
      l : ∀ {a : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Ex …
      r : ∀ {b : Option (Prod α (Stream'.WSeq α))}, Membership.mem t.destruct b → Ex …
      a : α
      s' : Stream'.WSeq α
      dsm : Membership.mem s.destruct (Option.some { fst := a, snd := s' })
      dse : Eq (Functor.map Prod.fst (Option.some { fst := a, snd := s' })) o
      b : α
      t' : Stream'.WSeq α
      dtm : Membership.mem t.destruct (Option.some { fst := b, snd := t' })
      dst : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some { fst := a,  …
      ⊢ Membership.mem t.head (Functor.map Prod.fst (Option.some { fst := a, snd :=  …
    -/
    rw [dst.left]
    exact @Computation.mem_map _ _ (@Functor.map _ _ (α × WSeq α) _ Prod.fst)
      (some (b, t')) (destruct t) dtm


theorem flatten_equiv {c : Computation (WSeq α)} {s} (h : s ∈ c) : flatten c ~ʷ s := by
  /-
    α : Type u
    c : Computation (Stream'.WSeq α)
    s : Stream'.WSeq α
    h : Membership.mem c s
    ⊢ (Stream'.WSeq.flatten c).Equiv s
  -/
  apply Computation.memRecOn h
    /-
      case h1
      α : Type u
      c : Computation (Stream'.WSeq α)
      s : Stream'.WSeq α
      h : Membership.mem c s
      ⊢ (Stream'.WSeq.flatten (Computation.pure s)).Equiv s
    -/
  · simp [Equiv.refl]
    /-
      🎉 no goals
    -/
    /-
      case h2
      α : Type u
      c : Computation (Stream'.WSeq α)
      s : Stream'.WSeq α
      h : Membership.mem c s
      ⊢ ∀ (s_1 : Computation (Stream'.WSeq α)), (Stream'.WSeq.flatten s_1).Equiv s → …
    -/
  · intro s'
    /-
      case h2
      α : Type u
      c : Computation (Stream'.WSeq α)
      s : Stream'.WSeq α
      h : Membership.mem c s
      s' : Computation (Stream'.WSeq α)
      ⊢ (Stream'.WSeq.flatten s').Equiv s → (Stream'.WSeq.flatten s'.think).Equiv s
    -/
    apply Equiv.trans
    /-
      case h2.a
      α : Type u
      c : Computation (Stream'.WSeq α)
      s : Stream'.WSeq α
      h : Membership.mem c s
      s' : Computation (Stream'.WSeq α)
      ⊢ (Stream'.WSeq.flatten s'.think).Equiv (Stream'.WSeq.flatten s')
    -/
    simp [think_equiv]
    /-
      🎉 no goals
    -/


theorem liftRel_flatten {R : α → β → Prop} {c1 : Computation (WSeq α)} {c2 : Computation (WSeq β)}
    (h : c1.LiftRel (LiftRel R) c2) : LiftRel R (flatten c1) (flatten c2) :=
  let S s t := ∃ c1 c2, s = flatten c1 ∧ t = flatten c2 ∧ Computation.LiftRel (LiftRel R) c1 c2
  ⟨S, ⟨c1, c2, rfl, rfl, h⟩, fun {s t} h =>
    match s, t, h with
    | _, _, ⟨c1, c2, rfl, rfl, h⟩ => by
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        c1✝ : Computation (Stream'.WSeq α)
        c2✝ : Computation (Stream'.WSeq β)
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1✝ c2✝
        S : Stream'.WSeq α → Stream'.WSeq β → Prop := fun s t => Exists fun c1 => Exis …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : S s t
        c1 : Computation (Stream'.WSeq α)
        c2 : Computation (Stream'.WSeq β)
        h : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1 c2
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R S) (Stream'.WSeq.flatten c1).de …
      -/
      simp only [destruct_flatten]; apply liftRel_bind _ _ h
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        c1✝ : Computation (Stream'.WSeq α)
        c2✝ : Computation (Stream'.WSeq β)
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1✝ c2✝
        S : Stream'.WSeq α → Stream'.WSeq β → Prop := fun s t => Exists fun c1 => Exis …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : S s t
        c1 : Computation (Stream'.WSeq α)
        c2 : Computation (Stream'.WSeq β)
        h : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1 c2
        ⊢ ∀ {a : Stream'.WSeq α} {b : Stream'.WSeq β}, Stream'.WSeq.LiftRel R a b → Co …
      -/
      intro a b ab; apply Computation.LiftRel.imp _ _ _ (liftRel_destruct ab)
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        c1✝ : Computation (Stream'.WSeq α)
        c2✝ : Computation (Stream'.WSeq β)
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1✝ c2✝
        S : Stream'.WSeq α → Stream'.WSeq β → Prop := fun s t => Exists fun c1 => Exis …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : S s t
        c1 : Computation (Stream'.WSeq α)
        c2 : Computation (Stream'.WSeq β)
        h : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1 c2
        a : Stream'.WSeq α
        b : Stream'.WSeq β
        ab : Stream'.WSeq.LiftRel R a b
        ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
      -/
      intro a b; apply LiftRelO.imp_right
      /-
        case H
        α : Type u
        β : Type v
        R : α → β → Prop
        c1✝ : Computation (Stream'.WSeq α)
        c2✝ : Computation (Stream'.WSeq β)
        h✝¹ : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1✝ c2✝
        S : Stream'.WSeq α → Stream'.WSeq β → Prop := fun s t => Exists fun c1 => Exis …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : S s t
        c1 : Computation (Stream'.WSeq α)
        c2 : Computation (Stream'.WSeq β)
        h : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1 c2
        a✝ : Stream'.WSeq α
        b✝ : Stream'.WSeq β
        ab : Stream'.WSeq.LiftRel R a✝ b✝
        a : Option (Prod α (Stream'.WSeq α))
        b : Option (Prod β (Stream'.WSeq β))
        ⊢ ∀ (s : Stream'.WSeq α) (t : Stream'.WSeq β), Stream'.WSeq.LiftRel R s t → S  …
      -/
      intro s t h; refine ⟨Computation.pure s, Computation.pure t, ?_, ?_, ?_⟩ <;>
        -- Porting note: These 2 theorems should be excluded.
        /-
          case H.refine_1
          α : Type u
          β : Type v
          R : α → β → Prop
          c1✝ : Computation (Stream'.WSeq α)
          c2✝ : Computation (Stream'.WSeq β)
          h✝² : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1✝ c2✝
          S : Stream'.WSeq α → Stream'.WSeq β → Prop := fun s t => Exists fun c1 => Exis …
          s✝ : Stream'.WSeq α
          t✝ : Stream'.WSeq β
          h✝¹ : S s✝ t✝
          c1 : Computation (Stream'.WSeq α)
          c2 : Computation (Stream'.WSeq β)
          h✝ : Computation.LiftRel (Stream'.WSeq.LiftRel R) c1 c2
          a✝ : Stream'.WSeq α
          b✝ : Stream'.WSeq β
          ab : Stream'.WSeq.LiftRel R a✝ b✝
          a : Option (Prod α (Stream'.WSeq α))
          b : Option (Prod β (Stream'.WSeq β))
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h : Stream'.WSeq.LiftRel R s t
          ⊢ Eq s (Stream'.WSeq.flatten (Computation.pure s))
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        simp [h, -liftRel_pure_left, -liftRel_pure_right]⟩
        /-
          🎉 no goals
        -/


theorem flatten_congr {c1 c2 : Computation (WSeq α)} :
    Computation.LiftRel Equiv c1 c2 → flatten c1 ~ʷ flatten c2 :=
  liftRel_flatten


theorem tail_congr {s t : WSeq α} (h : s ~ʷ t) : tail s ~ʷ tail t := by
  /-
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    ⊢ s.tail.Equiv t.tail
  -/
  apply flatten_congr
  /-
    case a
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    ⊢ Computation.LiftRel Stream'.WSeq.Equiv (Functor.map (fun o => Option.recOn o …
  -/
  dsimp only [(· <$> ·)]; rw [← Computation.bind_pure, ← Computation.bind_pure]
  /-
    case a
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    ⊢ Computation.LiftRel Stream'.WSeq.Equiv (s.destruct.bind (Function.comp Compu …
  -/
  apply liftRel_bind _ _ (destruct_congr h)
  /-
    case a
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    ⊢ ∀ {a b : Option (Prod α (Stream'.WSeq α))}, Stream'.WSeq.BisimO (fun x1 x2 = …
  -/
  intro a b h; simp only [comp_apply, liftRel_pure]
  /-
    case a
    α : Type u
    s t : Stream'.WSeq α
    h✝ : s.Equiv t
    a b : Option (Prod α (Stream'.WSeq α))
    h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) a b
    ⊢ (Option.rec Stream'.WSeq.nil Prod.snd a).Equiv (Option.rec Stream'.WSeq.nil  …
  -/
  cases' a with a <;> cases' b with b
    /-
      case a.none.none
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) Option.none Option.none
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd Option.none).Equiv (Option.rec Stream' …
    -/
  · trivial
    /-
      🎉 no goals
    -/
    /-
      case a.none.some
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      b : Prod α (Stream'.WSeq α)
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) Option.none (Option.some b)
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd Option.none).Equiv (Option.rec Stream' …
    -/
  · cases h
    /-
      🎉 no goals
    -/
    /-
      case a.some.none
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      a : Prod α (Stream'.WSeq α)
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some a) Option.none
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd (Option.some a)).Equiv (Option.rec Str …
    -/
  · cases a
    /-
      case a.some.none.mk
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      fst✝ : α
      snd✝ : Stream'.WSeq α
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some { fst := fst✝, …
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd (Option.some { fst := fst✝, snd := snd …
    -/
    cases h
    /-
      🎉 no goals
    -/
    /-
      case a.some.some
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      a b : Prod α (Stream'.WSeq α)
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some a) (Option.som …
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd (Option.some a)).Equiv (Option.rec Str …
    -/
  · cases' a with a s'
    /-
      case a.some.some.mk
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      b : Prod α (Stream'.WSeq α)
      a : α
      s' : Stream'.WSeq α
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some { fst := a, sn …
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd (Option.some { fst := a, snd := s' })) …
    -/
    cases' b with b t'
    /-
      case a.some.some.mk.mk
      α : Type u
      s t : Stream'.WSeq α
      h✝ : s.Equiv t
      a : α
      s' : Stream'.WSeq α
      b : α
      t' : Stream'.WSeq α
      h : Stream'.WSeq.BisimO (fun x1 x2 => x1.Equiv x2) (Option.some { fst := a, sn …
      ⊢ (Option.rec Stream'.WSeq.nil Prod.snd (Option.some { fst := a, snd := s' })) …
    -/
    exact h.right
    /-
      🎉 no goals
    -/


theorem dropn_congr {s t : WSeq α} (h : s ~ʷ t) (n) : drop s n ~ʷ drop t n := by
  /-
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    n : Nat
    ⊢ (s.drop n).Equiv (t.drop n)
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, tail_congr, drop]
                  /-
                    🎉 no goals
                  -/


theorem get?_congr {s t : WSeq α} (h : s ~ʷ t) (n) : get? s n ~ get? t n :=
  head_congr (dropn_congr h _)


theorem mem_congr {s t : WSeq α} (h : s ~ʷ t) (a) : a ∈ s ↔ a ∈ t :=
  suffices ∀ {s t : WSeq α}, s ~ʷ t → a ∈ s → a ∈ t from ⟨this h, this h.symm⟩
  fun {_ _} h as =>
  let ⟨_, hn⟩ := exists_get?_of_mem as
  get?_mem ((get?_congr h _ _).1 hn)


theorem productive_congr {s t : WSeq α} (h : s ~ʷ t) : Productive s ↔ Productive t := by
  /-
    α : Type u
    s t : Stream'.WSeq α
    h : s.Equiv t
    ⊢ Iff s.Productive t.Productive
  -/
  simp only [productive_iff]; exact forall_congr' fun n => terminates_congr <| get?_congr h _
                              /-
                                🎉 no goals
                              -/


theorem Equiv.ext {s t : WSeq α} (h : ∀ n, get? s n ~ get? t n) : s ~ʷ t :=
  ⟨fun s t => ∀ n, get? s n ~ get? t n, h, fun {s t} h => by
    /-
      α : Type u
      s✝ t✝ : Stream'.WSeq α
      h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
      s t : Stream'.WSeq α
      h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
      ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) fun s t = …
    -/
    refine liftRel_def.2 ⟨?_, ?_⟩
      /-
        case refine_1
        α : Type u
        s✝ t✝ : Stream'.WSeq α
        h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
        s t : Stream'.WSeq α
        h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
        ⊢ Iff s.destruct.Terminates t.destruct.Terminates
      -/
    · rw [← head_terminates_iff, ← head_terminates_iff]
      /-
        case refine_1
        α : Type u
        s✝ t✝ : Stream'.WSeq α
        h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
        s t : Stream'.WSeq α
        h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
        ⊢ Iff s.head.Terminates t.head.Terminates
      -/
      exact terminates_congr (h 0)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u
        s✝ t✝ : Stream'.WSeq α
        h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
        s t : Stream'.WSeq α
        h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
        ⊢ ∀ {a b : Option (Prod α (Stream'.WSeq α))}, Membership.mem s.destruct a → Me …
      -/
    · intro a b ma mb
      /-
        case refine_2
        α : Type u
        s✝ t✝ : Stream'.WSeq α
        h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
        s t : Stream'.WSeq α
        h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
        a b : Option (Prod α (Stream'.WSeq α))
        ma : Membership.mem s.destruct a
        mb : Membership.mem t.destruct b
        ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
      -/
      cases' a with a <;> cases' b with b
        /-
          case refine_2.none.none
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          ma : Membership.mem s.destruct Option.none
          mb : Membership.mem t.destruct Option.none
          ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
        -/
      · trivial
        /-
          🎉 no goals
        -/
        /-
          case refine_2.none.some
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          ma : Membership.mem s.destruct Option.none
          b : Prod α (Stream'.WSeq α)
          mb : Membership.mem t.destruct (Option.some b)
          ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
        -/
      · injection mem_unique (Computation.mem_map _ ma) ((h 0 _).2 (Computation.mem_map _ mb))
        /-
          🎉 no goals
        -/
        /-
          case refine_2.some.none
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          a : Prod α (Stream'.WSeq α)
          ma : Membership.mem s.destruct (Option.some a)
          mb : Membership.mem t.destruct Option.none
          ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
        -/
      · injection mem_unique (Computation.mem_map _ ma) ((h 0 _).2 (Computation.mem_map _ mb))
        /-
          🎉 no goals
        -/
        /-
          case refine_2.some.some
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          a : Prod α (Stream'.WSeq α)
          ma : Membership.mem s.destruct (Option.some a)
          b : Prod α (Stream'.WSeq α)
          mb : Membership.mem t.destruct (Option.some b)
          ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
        -/
      · cases' a with a s'
        /-
          case refine_2.some.some.mk
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          b : Prod α (Stream'.WSeq α)
          mb : Membership.mem t.destruct (Option.some b)
          a : α
          s' : Stream'.WSeq α
          ma : Membership.mem s.destruct (Option.some { fst := a, snd := s' })
          ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
        -/
        cases' b with b t'
        injection mem_unique (Computation.mem_map _ ma) ((h 0 _).2 (Computation.mem_map _ mb)) with
          ab
        /-
          case refine_2.some.some.mk.mk
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          a : α
          s' : Stream'.WSeq α
          ma : Membership.mem s.destruct (Option.some { fst := a, snd := s' })
          b : α
          t' : Stream'.WSeq α
          mb : Membership.mem t.destruct (Option.some { fst := b, snd := t' })
          ab : Eq { fst := a, snd := s' }.1 { fst := b, snd := t' }.1
          ⊢ Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) (fun s t => ∀ (n : Nat), (s.ge …
        -/
        refine ⟨ab, fun n => ?_⟩
        refine
          (get?_congr (flatten_equiv (Computation.mem_map _ ma)) n).symm.trans
            ((?_ : get? (tail s) n ~ get? (tail t) n).trans
              (get?_congr (flatten_equiv (Computation.mem_map _ mb)) n))
        /-
          case refine_2.some.some.mk.mk
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          a : α
          s' : Stream'.WSeq α
          ma : Membership.mem s.destruct (Option.some { fst := a, snd := s' })
          b : α
          t' : Stream'.WSeq α
          mb : Membership.mem t.destruct (Option.some { fst := b, snd := t' })
          ab : Eq { fst := a, snd := s' }.1 { fst := b, snd := t' }.1
          n : Nat
          ⊢ (s.tail.get? n).Equiv (t.tail.get? n)
        -/
        rw [get?_tail, get?_tail]
        /-
          case refine_2.some.some.mk.mk
          α : Type u
          s✝ t✝ : Stream'.WSeq α
          h✝ : ∀ (n : Nat), (s✝.get? n).Equiv (t✝.get? n)
          s t : Stream'.WSeq α
          h : (fun s t => ∀ (n : Nat), (s.get? n).Equiv (t.get? n)) s t
          a : α
          s' : Stream'.WSeq α
          ma : Membership.mem s.destruct (Option.some { fst := a, snd := s' })
          b : α
          t' : Stream'.WSeq α
          mb : Membership.mem t.destruct (Option.some { fst := b, snd := t' })
          ab : Eq { fst := a, snd := s' }.1 { fst := b, snd := t' }.1
          n : Nat
          ⊢ (s.get? (HAdd.hAdd n 1)).Equiv (t.get? (HAdd.hAdd n 1))
        -/
        apply h⟩
        /-
          🎉 no goals
        -/


theorem length_eq_map (s : WSeq α) : length s = Computation.map List.length (toList s) := by
  refine
    Computation.eq_of_bisim
      (fun c1 c2 =>
        ∃ (l : List α) (s : WSeq α),
          c1 = Computation.corec (fun ⟨n, s⟩ =>
            match Seq.destruct s with
            | none => Sum.inl n
            | some (none, s') => Sum.inr (n, s')
            | some (some _, s') => Sum.inr (n + 1, s')) (l.length, s) ∧
            c2 = Computation.map List.length (Computation.corec (fun ⟨l, s⟩ =>
              match Seq.destruct s with
              | none => Sum.inl l.reverse
              | some (none, s') => Sum.inr (l, s')
              | some (some a, s') => Sum.inr (a::l, s')) (l, s)))
      ?_ ⟨[], s, rfl, rfl⟩
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ Computation.IsBisimulation fun c1 c2 => Exists fun l => Exists fun s => And  …
  -/
  intro s1 s2 h; rcases h with ⟨l, s, h⟩; rw [h.left, h.right]
  /-
    case intro.intro
    α : Type u
    s✝ : Stream'.WSeq α
    s1 s2 : Computation Nat
    l : List α
    s : Stream'.WSeq α
    h : And (Eq s1 (Computation.corec (fun x => Stream'.WSeq.length.match_1 (fun x …
    ⊢ Computation.BisimO (fun c1 c2 => Exists fun l => Exists fun s => And (Eq c1  …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  induction' s using WSeq.recOn with a s s <;> simp [toList, nil, cons, think, length]
    /-
      case intro.intro.h2
      α : Type u
      s✝ : Stream'.WSeq α
      s1 s2 : Computation Nat
      l : List α
      a : α
      s : Stream'.WSeq α
      h : And (Eq s1 (Computation.corec (fun x => Stream'.WSeq.length.match_1 (fun x …
      ⊢ Exists fun l_1 => Exists fun s_1 => And (Eq (Computation.corec (fun x => Str …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · refine ⟨a::l, s, ?_, ?_⟩ <;> simp
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case intro.intro.h3
      α : Type u
      s✝ : Stream'.WSeq α
      s1 s2 : Computation Nat
      l : List α
      s : Stream'.WSeq α
      h : And (Eq s1 (Computation.corec (fun x => Stream'.WSeq.length.match_1 (fun x …
      ⊢ Exists fun l_1 => Exists fun s_1 => And (Eq (Computation.corec (fun x => Str …
    -/
                              /-
                                🎉 no goals
                              -/
  · refine ⟨l, s, ?_, ?_⟩ <;> simp
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem ofList_nil : ofList [] = (nil : WSeq α) :=
  rfl


@[simp]
theorem ofList_cons (a : α) (l) : ofList (a::l) = cons a (ofList l) :=
                                                                                             /-
                                                                                               α : Type u
                                                                                               a : α
                                                                                               l : List α
                                                                                               ⊢ Eq (Stream'.Seq.map Option.some ↑(List.cons a l)) (Stream'.Seq.cons (Option. …
                                                                                             -/
  show Seq.map some (Seq.ofList (a::l)) = Seq.cons (some a) (Seq.map some (Seq.ofList l)) by simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[simp]
theorem toList'_nil (l : List α) :
    Computation.corec (fun ⟨l, s⟩ =>
      match Seq.destruct s with
      | none => Sum.inl l.reverse
      | some (none, s') => Sum.inr (l, s')
      | some (some a, s') => Sum.inr (a::l, s')) (l, nil) = Computation.pure l.reverse :=
  destruct_eq_pure rfl


@[simp]
theorem toList'_cons (l : List α) (s : WSeq α) (a : α) :
    Computation.corec (fun ⟨l, s⟩ =>
      match Seq.destruct s with
      | none => Sum.inl l.reverse
      | some (none, s') => Sum.inr (l, s')
      | some (some a, s') => Sum.inr (a::l, s')) (l, cons a s) =
      (Computation.corec (fun ⟨l, s⟩ =>
        match Seq.destruct s with
        | none => Sum.inl l.reverse
        | some (none, s') => Sum.inr (l, s')
        | some (some a, s') => Sum.inr (a::l, s')) (a::l, s)).think :=
                          /-
                            α : Type u
                            l : List α
                            s : Stream'.WSeq α
                            a : α
                            ⊢ Eq (Computation.corec (fun x => Stream'.WSeq.toList.match_1 (fun x => Sum (L …
                          -/
  destruct_eq_think <| by simp [toList, cons]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem toList'_think (l : List α) (s : WSeq α) :
    Computation.corec (fun ⟨l, s⟩ =>
      match Seq.destruct s with
      | none => Sum.inl l.reverse
      | some (none, s') => Sum.inr (l, s')
      | some (some a, s') => Sum.inr (a::l, s')) (l, think s) =
      (Computation.corec (fun ⟨l, s⟩ =>
        match Seq.destruct s with
        | none => Sum.inl l.reverse
        | some (none, s') => Sum.inr (l, s')
        | some (some a, s') => Sum.inr (a::l, s')) (l, s)).think :=
                          /-
                            α : Type u
                            l : List α
                            s : Stream'.WSeq α
                            ⊢ Eq (Computation.corec (fun x => Stream'.WSeq.toList.match_1 (fun x => Sum (L …
                          -/
  destruct_eq_think <| by simp [toList, think]
                          /-
                            🎉 no goals
                          -/


theorem toList'_map (l : List α) (s : WSeq α) :
    Computation.corec (fun ⟨l, s⟩ =>
      match Seq.destruct s with
      | none => Sum.inl l.reverse
      | some (none, s') => Sum.inr (l, s')
      | some (some a, s') => Sum.inr (a :: l, s')) (l, s) = (l.reverse ++ ·) <$> toList s := by
  refine
    Computation.eq_of_bisim
      (fun c1 c2 =>
        ∃ (l' : List α) (s : WSeq α),
          c1 = Computation.corec (fun ⟨l, s⟩ =>
            match Seq.destruct s with
            | none => Sum.inl l.reverse
            | some (none, s') => Sum.inr (l, s')
            | some (some a, s') => Sum.inr (a::l, s')) (l' ++ l, s) ∧
            c2 = Computation.map (l.reverse ++ ·) (Computation.corec (fun ⟨l, s⟩ =>
              match Seq.destruct s with
              | none => Sum.inl l.reverse
              | some (none, s') => Sum.inr (l, s')
              | some (some a, s') => Sum.inr (a::l, s')) (l', s)))
      ?_ ⟨[], s, rfl, rfl⟩
  /-
    α : Type u
    l : List α
    s : Stream'.WSeq α
    ⊢ Computation.IsBisimulation fun c1 c2 => Exists fun l' => Exists fun s => And …
  -/
  intro s1 s2 h; rcases h with ⟨l', s, h⟩; rw [h.left, h.right]
  /-
    case intro.intro
    α : Type u
    l : List α
    s✝ : Stream'.WSeq α
    s1 s2 : Computation (List α)
    l' : List α
    s : Stream'.WSeq α
    h : And (Eq s1 (Computation.corec (fun x => Stream'.WSeq.toList.match_1 (fun x …
    ⊢ Computation.BisimO (fun c1 c2 => Exists fun l' => Exists fun s => And (Eq c1 …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  induction' s using WSeq.recOn with a s s <;> simp [toList, nil, cons, think, length]
    /-
      case intro.intro.h2
      α : Type u
      l : List α
      s✝ : Stream'.WSeq α
      s1 s2 : Computation (List α)
      l' : List α
      a : α
      s : Stream'.WSeq α
      h : And (Eq s1 (Computation.corec (fun x => Stream'.WSeq.toList.match_1 (fun x …
      ⊢ Exists fun l'_1 => Exists fun s_1 => And (Eq (Computation.corec (fun x => St …
    -/
                                  /-
                                    🎉 no goals
                                  -/
  · refine ⟨a::l', s, ?_, ?_⟩ <;> simp
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case intro.intro.h3
      α : Type u
      l : List α
      s✝ : Stream'.WSeq α
      s1 s2 : Computation (List α)
      l' : List α
      s : Stream'.WSeq α
      h : And (Eq s1 (Computation.corec (fun x => Stream'.WSeq.toList.match_1 (fun x …
      ⊢ Exists fun l'_1 => Exists fun s_1 => And (Eq (Computation.corec (fun x => St …
    -/
                               /-
                                 🎉 no goals
                               -/
  · refine ⟨l', s, ?_, ?_⟩ <;> simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem toList_cons (a : α) (s) : toList (cons a s) = (List.cons a <$> toList s).think :=
  destruct_eq_think <| by
    /-
      α : Type u
      a : α
      s : Stream'.WSeq α
      ⊢ Eq (Stream'.WSeq.cons a s).toList.destruct (Sum.inr (Functor.map (List.cons  …
    -/
    unfold toList
    /-
      α : Type u
      a : α
      s : Stream'.WSeq α
      ⊢ Eq (Computation.corec (fun x => Stream'.WSeq.toList.match_1 (fun x => Sum (L …
    -/
    simp only [toList'_cons, Computation.destruct_think, Sum.inr.injEq]
    /-
      α : Type u
      a : α
      s : Stream'.WSeq α
      ⊢ Eq (Computation.corec (fun x => Stream'.WSeq.destruct.match_1 (fun x => Sum  …
    -/
    rw [toList'_map]
    /-
      α : Type u
      a : α
      s : Stream'.WSeq α
      ⊢ Eq (Functor.map (fun x => HAppend.hAppend (List.cons a List.nil).reverse x)  …
    -/
    simp only [List.reverse_cons, List.reverse_nil, List.nil_append, List.singleton_append]
    /-
      α : Type u
      a : α
      s : Stream'.WSeq α
      ⊢ Eq (Functor.map (fun x => List.cons a x) s.toList) (Functor.map (List.cons a …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem toList_nil : toList (nil : WSeq α) = Computation.pure [] :=
  destruct_eq_pure rfl


theorem toList_ofList (l : List α) : l ∈ toList (ofList l) := by
  /-
    α : Type u
    l : List α
    ⊢ Membership.mem (↑l).toList l
  -/
  induction' l with a l IH
    /-
      case nil
      α : Type u
      ⊢ Membership.mem (↑List.nil).toList List.nil
    -/
  · simp [ret_mem]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      a : α
      l : List α
      IH : Membership.mem (↑l).toList l
      ⊢ Membership.mem (↑(List.cons a l)).toList (List.cons a l)
    -/
  · simpa [ret_mem] using think_mem (Computation.mem_map _ IH)
    /-
      🎉 no goals
    -/


@[simp]
theorem destruct_ofSeq (s : Seq α) :
    destruct (ofSeq s) = Computation.pure (s.head.map fun a => (a, ofSeq s.tail)) :=
  destruct_eq_pure <| by
    simp only [destruct, Seq.destruct, Option.map_eq_map, ofSeq, Computation.corec_eq, rmap,
      Seq.head]
    /-
      α : Type u
      s : Stream'.Seq α
      ⊢ Eq (Computation.Corec.f.match_1 (fun x => Sum (Option (Prod α (Stream'.WSeq  …
    -/
    rw [show Seq.get? (some <$> s) 0 = some <$> Seq.get? s 0 by apply Seq.map_get?]
    /-
      α : Type u
      s : Stream'.Seq α
      ⊢ Eq (Computation.Corec.f.match_1 (fun x => Sum (Option (Prod α (Stream'.WSeq  …
    -/
    cases' Seq.get? s 0 with a
      /-
        case none
        α : Type u
        s : Stream'.Seq α
        ⊢ Eq (Computation.Corec.f.match_1 (fun x => Sum (Option (Prod α (Stream'.WSeq  …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case some
      α : Type u
      s : Stream'.Seq α
      a : α
      ⊢ Eq (Computation.Corec.f.match_1 (fun x => Sum (Option (Prod α (Stream'.WSeq  …
    -/
    dsimp only [(· <$> ·)]
    /-
      case some
      α : Type u
      s : Stream'.Seq α
      a : α
      ⊢ Eq (Computation.Corec.f.match_1 (fun x => Sum (Option (Prod α (Stream'.WSeq  …
    -/
    simp [destruct]
    /-
      🎉 no goals
    -/


@[simp]
theorem head_ofSeq (s : Seq α) : head (ofSeq s) = Computation.pure s.head := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (↑s).head (Computation.pure s.head)
  -/
  simp only [head, Option.map_eq_map, destruct_ofSeq, Computation.map_pure, Option.map_map]
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (Computation.pure (Option.map (Function.comp Prod.fst fun a => { fst := a …
  -/
                       /-
                         🎉 no goals
                       -/
  cases Seq.head s <;> rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem tail_ofSeq (s : Seq α) : tail (ofSeq s) = ofSeq s.tail := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (↑s).tail ↑s.tail
  -/
  simp only [tail, destruct_ofSeq, map_pure', flatten_pure]
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (Option.rec Stream'.WSeq.nil Prod.snd (Option.map (fun a => { fst := a, s …
  -/
  induction' s using Seq.recOn with x s <;> simp only [ofSeq, Seq.tail_nil, Seq.head_nil,
    Option.map_none', Seq.tail_cons, Seq.head_cons, Option.map_some']
    /-
      case h1
      α : Type u
      ⊢ Eq Stream'.WSeq.nil (Functor.map Option.some Stream'.Seq.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem dropn_ofSeq (s : Seq α) : ∀ n, drop (ofSeq s) n = ofSeq (s.drop n)
  | 0 => rfl
  | n + 1 => by
    /-
      α : Type u
      s : Stream'.Seq α
      n : Nat
      ⊢ Eq ((↑s).drop (HAdd.hAdd n 1)) ↑(s.drop (HAdd.hAdd n 1))
    -/
    simp only [drop, Nat.add_eq, Nat.add_zero, Seq.drop]
    /-
      α : Type u
      s : Stream'.Seq α
      n : Nat
      ⊢ Eq ((↑s).drop n).tail ↑(s.drop n).tail
    -/
    rw [dropn_ofSeq s n, tail_ofSeq]
    /-
      🎉 no goals
    -/


theorem get?_ofSeq (s : Seq α) (n) : get? (ofSeq s) n = Computation.pure (Seq.get? s n) := by
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Eq ((↑s).get? n) (Computation.pure (s.get? n))
  -/
  dsimp [get?]; rw [dropn_ofSeq, head_ofSeq, Seq.head_dropn]
                /-
                  🎉 no goals
                -/


instance productive_ofSeq (s : Seq α) : Productive (ofSeq s) :=
               /-
                 α : Type u
                 β : Type v
                 γ : Type w
                 s : Stream'.Seq α
                 n : Nat
                 ⊢ ((↑s).get? n).Terminates
               -/
  ⟨fun n => by rw [get?_ofSeq]; infer_instance⟩
                                /-
                                  🎉 no goals
                                -/


theorem toSeq_ofSeq (s : Seq α) : toSeq (ofSeq s) = s := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (↑s).toSeq s
  -/
  apply Subtype.eq; funext n
  /-
    case a.h
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Eq (↑(↑s).toSeq n) (↑s n)
  -/
  dsimp [toSeq]; apply get_eq_of_mem
  /-
    case a.h.a
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Membership.mem ((↑s).get? n) (↑s n)
  -/
  rw [get?_ofSeq]; apply ret_mem
                   /-
                     🎉 no goals
                   -/


/-- The monadic `return a` is a singleton list containing `a`. -/
def ret (a : α) : WSeq α :=
  ofList [a]


@[simp]
theorem map_nil (f : α → β) : map f nil = nil :=
  rfl


@[simp]
theorem map_cons (f : α → β) (a s) : map f (cons a s) = cons (f a) (map f s) :=
  Seq.map_cons _ _ _


@[simp]
theorem map_think (f : α → β) (s) : map f (think s) = think (map f s) :=
  Seq.map_cons _ _ _


@[simp]
                                                 /-
                                                   α : Type u
                                                   s : Stream'.WSeq α
                                                   ⊢ Eq (Stream'.WSeq.map id s) s
                                                 -/
theorem map_id (s : WSeq α) : map id s = s := by simp [map]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                                  /-
                                                                    α : Type u
                                                                    β : Type v
                                                                    f : α → β
                                                                    a : α
                                                                    ⊢ Eq (Stream'.WSeq.map f (Stream'.WSeq.ret a)) (Stream'.WSeq.ret (f a))
                                                                  -/
theorem map_ret (f : α → β) (a) : map f (ret a) = ret (f a) := by simp [ret]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem map_append (f : α → β) (s t) : map f (append s t) = append (map f s) (map f t) :=
  Seq.map_append _ _ _


theorem map_comp (f : α → β) (g : β → γ) (s : WSeq α) : map (g ∘ f) s = map g (map f s) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    g : β → γ
    s : Stream'.WSeq α
    ⊢ Eq (Stream'.WSeq.map (Function.comp g f) s) (Stream'.WSeq.map g (Stream'.WSe …
  -/
  dsimp [map]; rw [← Seq.map_comp]
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    g : β → γ
    s : Stream'.WSeq α
    ⊢ Eq (Stream'.Seq.map (Option.map (Function.comp g f)) s) (Stream'.Seq.map (Fu …
  -/
  apply congr_fun; apply congr_arg
  /-
    case h.h
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    g : β → γ
    s : Stream'.WSeq α
    ⊢ Eq (Option.map (Function.comp g f)) (Function.comp (Option.map g) (Option.ma …
  -/
             /-
               🎉 no goals
             -/
  ext ⟨⟩ <;> rfl
             /-
               🎉 no goals
             -/


theorem mem_map (f : α → β) {a : α} {s : WSeq α} : a ∈ s → f a ∈ map f s :=
  Seq.mem_map (Option.map f)

-- The converse is not true without additional assumptions

theorem exists_of_mem_join {a : α} : ∀ {S : WSeq (WSeq α)}, a ∈ join S → ∃ s, s ∈ S ∧ a ∈ s := by
  suffices
    ∀ ss : WSeq α,
      a ∈ ss → ∀ s S, append s (join S) = ss → a ∈ append s (join S) → a ∈ s ∨ ∃ s, s ∈ S ∧ a ∈ s
    from fun S h => (this _ h nil S (by simp) (by simp [h])).resolve_left (not_mem_nil _)
  /-
    α : Type u
    a : α
    ⊢ ∀ (ss : Stream'.WSeq α), Membership.mem ss a → ∀ (s : Stream'.WSeq α) (S : S …
  -/
  intro ss h; apply mem_rec_on h <;> [intro b ss o; intro ss IH] <;> intro s S
  · induction' s using WSeq.recOn with b' s s <;>
      [induction' S using WSeq.recOn with s S S; skip; skip] <;>
      /-
        case h1.h1.h1
        α : Type u
        a : α
        ss✝ : Stream'.WSeq α
        h : Membership.mem ss✝ a
        b : α
        ss : Stream'.WSeq α
        o : Or (Eq a b) (∀ (s : Stream'.WSeq α) (S : Stream'.WSeq (Stream'.WSeq α)), E …
        ⊢ Eq (Stream'.WSeq.nil.append Stream'.WSeq.nil.join) (Stream'.WSeq.cons b ss)  …
      -/
      intro ej m <;> simp at ej <;> have := congr_arg Seq.destruct ej <;>
      /-
        case h1.h1.h1
        α : Type u
        a : α
        ss✝ : Stream'.WSeq α
        h : Membership.mem ss✝ a
        b : α
        ss : Stream'.WSeq α
        o : Or (Eq a b) (∀ (s : Stream'.WSeq α) (S : Stream'.WSeq (Stream'.WSeq α)), E …
        m : Membership.mem (Stream'.WSeq.nil.append Stream'.WSeq.nil.join) a
        ej : Eq Stream'.WSeq.nil (Stream'.WSeq.cons b ss)
        this : Eq (Stream'.Seq.destruct Stream'.WSeq.nil) (Stream'.Seq.destruct (Strea …
        ⊢ Or (Membership.mem Stream'.WSeq.nil a) (Exists fun s => And (Membership.mem  …
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
      /-
        🎉 no goals
      -/
      simp at this; cases this
    /-
      case h1.h2.intro
      α : Type u
      a : α
      ss✝ : Stream'.WSeq α
      h : Membership.mem ss✝ a
      b : α
      ss : Stream'.WSeq α
      o : Or (Eq a b) (∀ (s : Stream'.WSeq α) (S : Stream'.WSeq (Stream'.WSeq α)), E …
      S : Stream'.WSeq (Stream'.WSeq α)
      b' : α
      s : Stream'.WSeq α
      m : Membership.mem ((Stream'.WSeq.cons b' s).append S.join) a
      ej : Eq (Stream'.WSeq.cons b' (s.append S.join)) (Stream'.WSeq.cons b ss)
      left✝ : Eq b' b
      right✝ : Eq (s.append S.join) ss
      ⊢ Or (Membership.mem (Stream'.WSeq.cons b' s) a) (Exists fun s => And (Members …
    -/
    substs b' ss
    /-
      case h1.h2.intro
      α : Type u
      a : α
      ss : Stream'.WSeq α
      h : Membership.mem ss a
      b : α
      S : Stream'.WSeq (Stream'.WSeq α)
      s : Stream'.WSeq α
      m : Membership.mem ((Stream'.WSeq.cons b s).append S.join) a
      o : Or (Eq a b) (∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α) …
      ej : Eq (Stream'.WSeq.cons b (s.append S.join)) (Stream'.WSeq.cons b (s.append …
      ⊢ Or (Membership.mem (Stream'.WSeq.cons b s) a) (Exists fun s => And (Membersh …
    -/
    simp? at m ⊢ says simp only [cons_append, mem_cons_iff] at m ⊢
    /-
      case h1.h2.intro
      α : Type u
      a : α
      ss : Stream'.WSeq α
      h : Membership.mem ss a
      b : α
      S : Stream'.WSeq (Stream'.WSeq α)
      s : Stream'.WSeq α
      o : Or (Eq a b) (∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α) …
      ej : Eq (Stream'.WSeq.cons b (s.append S.join)) (Stream'.WSeq.cons b (s.append …
      m : Or (Eq a b) (Membership.mem (s.append S.join) a)
      ⊢ Or (Or (Eq a b) (Membership.mem s a)) (Exists fun s => And (Membership.mem S …
    -/
    cases' o with e IH
      /-
        case h1.h2.intro.inl
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        b : α
        S : Stream'.WSeq (Stream'.WSeq α)
        s : Stream'.WSeq α
        ej : Eq (Stream'.WSeq.cons b (s.append S.join)) (Stream'.WSeq.cons b (s.append …
        m : Or (Eq a b) (Membership.mem (s.append S.join) a)
        e : Eq a b
        ⊢ Or (Or (Eq a b) (Membership.mem s a)) (Exists fun s => And (Membership.mem S …
      -/
    · simp [e]
      /-
        🎉 no goals
      -/
    /-
      case h1.h2.intro.inr
      α : Type u
      a : α
      ss : Stream'.WSeq α
      h : Membership.mem ss a
      b : α
      S : Stream'.WSeq (Stream'.WSeq α)
      s : Stream'.WSeq α
      ej : Eq (Stream'.WSeq.cons b (s.append S.join)) (Stream'.WSeq.cons b (s.append …
      m : Or (Eq a b) (Membership.mem (s.append S.join) a)
      IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
      ⊢ Or (Or (Eq a b) (Membership.mem s a)) (Exists fun s => And (Membership.mem S …
    -/
    cases' m with e m
      /-
        case h1.h2.intro.inr.inl
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        b : α
        S : Stream'.WSeq (Stream'.WSeq α)
        s : Stream'.WSeq α
        ej : Eq (Stream'.WSeq.cons b (s.append S.join)) (Stream'.WSeq.cons b (s.append …
        IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
        e : Eq a b
        ⊢ Or (Or (Eq a b) (Membership.mem s a)) (Exists fun s => And (Membership.mem S …
      -/
    · simp [e]
      /-
        🎉 no goals
      -/
    /-
      case h1.h2.intro.inr.inr
      α : Type u
      a : α
      ss : Stream'.WSeq α
      h : Membership.mem ss a
      b : α
      S : Stream'.WSeq (Stream'.WSeq α)
      s : Stream'.WSeq α
      ej : Eq (Stream'.WSeq.cons b (s.append S.join)) (Stream'.WSeq.cons b (s.append …
      IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
      m : Membership.mem (s.append S.join) a
      ⊢ Or (Or (Eq a b) (Membership.mem s a)) (Exists fun s => And (Membership.mem S …
    -/
    exact Or.imp_left Or.inr (IH _ _ rfl m)
    /-
      🎉 no goals
    -/
  · induction' s using WSeq.recOn with b' s s <;>
      [induction' S using WSeq.recOn with s S S; skip; skip] <;>
      /-
        case h2.h1.h1
        α : Type u
        a : α
        ss✝ : Stream'.WSeq α
        h : Membership.mem ss✝ a
        ss : Stream'.WSeq α
        IH : ∀ (s : Stream'.WSeq α) (S : Stream'.WSeq (Stream'.WSeq α)), Eq (s.append  …
        ⊢ Eq (Stream'.WSeq.nil.append Stream'.WSeq.nil.join) ss.think → Membership.mem …
      -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      intro ej m <;> simp at ej <;> have := congr_arg Seq.destruct ej <;> simp at this <;>
      /-
        case h2.h1.h2
        α : Type u
        a : α
        ss✝ : Stream'.WSeq α
        h : Membership.mem ss✝ a
        ss : Stream'.WSeq α
        IH : ∀ (s : Stream'.WSeq α) (S : Stream'.WSeq (Stream'.WSeq α)), Eq (s.append  …
        s : Stream'.WSeq α
        S : Stream'.WSeq (Stream'.WSeq α)
        m : Membership.mem (Stream'.WSeq.nil.append (Stream'.WSeq.cons s S).join) a
        ej : Eq (s.append S.join).think ss.think
        this : Eq (s.append S.join) ss
        ⊢ Or (Membership.mem Stream'.WSeq.nil a) (Exists fun s_1 => And (Membership.me …
      -/
      subst ss
      /-
        case h2.h1.h2
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        s : Stream'.WSeq α
        S : Stream'.WSeq (Stream'.WSeq α)
        m : Membership.mem (Stream'.WSeq.nil.append (Stream'.WSeq.cons s S).join) a
        IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
        ej : Eq (s.append S.join).think (s.append S.join).think
        ⊢ Or (Membership.mem Stream'.WSeq.nil a) (Exists fun s_1 => And (Membership.me …
      -/
    · apply Or.inr
      -- Porting note: `exists_eq_or_imp` should be excluded.
      /-
        case h2.h1.h2.h
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        s : Stream'.WSeq α
        S : Stream'.WSeq (Stream'.WSeq α)
        m : Membership.mem (Stream'.WSeq.nil.append (Stream'.WSeq.cons s S).join) a
        IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
        ej : Eq (s.append S.join).think (s.append S.join).think
        ⊢ Exists fun s_1 => And (Membership.mem (Stream'.WSeq.cons s S) s_1) (Membersh …
      -/
      simp [-exists_eq_or_imp] at m ⊢
      /-
        case h2.h1.h2.h
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        s : Stream'.WSeq α
        S : Stream'.WSeq (Stream'.WSeq α)
        IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
        ej : Eq (s.append S.join).think (s.append S.join).think
        m : Membership.mem (s.append S.join) a
        ⊢ Exists fun s_1 => And (Or (Eq s_1 s) (Membership.mem S s_1)) (Membership.mem …
      -/
      cases' IH s S rfl m with as ex
        /-
          case h2.h1.h2.h.inl
          α : Type u
          a : α
          ss : Stream'.WSeq α
          h : Membership.mem ss a
          s : Stream'.WSeq α
          S : Stream'.WSeq (Stream'.WSeq α)
          IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
          ej : Eq (s.append S.join).think (s.append S.join).think
          m : Membership.mem (s.append S.join) a
          as : Membership.mem s a
          ⊢ Exists fun s_1 => And (Or (Eq s_1 s) (Membership.mem S s_1)) (Membership.mem …
        -/
      · exact ⟨s, Or.inl rfl, as⟩
        /-
          🎉 no goals
        -/
        /-
          case h2.h1.h2.h.inr
          α : Type u
          a : α
          ss : Stream'.WSeq α
          h : Membership.mem ss a
          s : Stream'.WSeq α
          S : Stream'.WSeq (Stream'.WSeq α)
          IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
          ej : Eq (s.append S.join).think (s.append S.join).think
          m : Membership.mem (s.append S.join) a
          ex : Exists fun s => And (Membership.mem S s) (Membership.mem s a)
          ⊢ Exists fun s_1 => And (Or (Eq s_1 s) (Membership.mem S s_1)) (Membership.mem …
        -/
      · rcases ex with ⟨s', sS, as⟩
        /-
          case h2.h1.h2.h.inr.intro.intro
          α : Type u
          a : α
          ss : Stream'.WSeq α
          h : Membership.mem ss a
          s : Stream'.WSeq α
          S : Stream'.WSeq (Stream'.WSeq α)
          IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
          ej : Eq (s.append S.join).think (s.append S.join).think
          m : Membership.mem (s.append S.join) a
          s' : Stream'.WSeq α
          sS : Membership.mem S s'
          as : Membership.mem s' a
          ⊢ Exists fun s_1 => And (Or (Eq s_1 s) (Membership.mem S s_1)) (Membership.mem …
        -/
        exact ⟨s', Or.inr sS, as⟩
        /-
          🎉 no goals
        -/
      /-
        case h2.h1.h3
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        S : Stream'.WSeq (Stream'.WSeq α)
        m : Membership.mem (Stream'.WSeq.nil.append S.think.join) a
        IH : ∀ (s : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s.appen …
        ej : Eq S.join.think S.join.think
        ⊢ Or (Membership.mem Stream'.WSeq.nil a) (Exists fun s => And (Membership.mem  …
      -/
    · apply Or.inr
      /-
        case h2.h1.h3.h
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        S : Stream'.WSeq (Stream'.WSeq α)
        m : Membership.mem (Stream'.WSeq.nil.append S.think.join) a
        IH : ∀ (s : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s.appen …
        ej : Eq S.join.think S.join.think
        ⊢ Exists fun s => And (Membership.mem S.think s) (Membership.mem s a)
      -/
      simp? at m says simp only [join_think, nil_append, mem_think] at m
      /-
        case h2.h1.h3.h
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        S : Stream'.WSeq (Stream'.WSeq α)
        IH : ∀ (s : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s.appen …
        ej : Eq S.join.think S.join.think
        m : Membership.mem S.join a
        ⊢ Exists fun s => And (Membership.mem S.think s) (Membership.mem s a)
      -/
      rcases (IH nil S (by simp) (by simp [m])).resolve_left (not_mem_nil _) with ⟨s, sS, as⟩
      /-
        case h2.h1.h3.h.intro.intro
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        S : Stream'.WSeq (Stream'.WSeq α)
        IH : ∀ (s : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s.appen …
        ej : Eq S.join.think S.join.think
        m : Membership.mem S.join a
        s : Stream'.WSeq α
        sS : Membership.mem S s
        as : Membership.mem s a
        ⊢ Exists fun s => And (Membership.mem S.think s) (Membership.mem s a)
      -/
      exact ⟨s, by simp [sS], as⟩
      /-
        🎉 no goals
      -/
      /-
        case h2.h3
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        S : Stream'.WSeq (Stream'.WSeq α)
        s : Stream'.WSeq α
        m : Membership.mem (s.think.append S.join) a
        IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
        ej : Eq (s.append S.join).think (s.append S.join).think
        ⊢ Or (Membership.mem s.think a) (Exists fun s => And (Membership.mem S s) (Mem …
      -/
    · simp only [think_append, mem_think] at m IH ⊢
      /-
        case h2.h3
        α : Type u
        a : α
        ss : Stream'.WSeq α
        h : Membership.mem ss a
        S : Stream'.WSeq (Stream'.WSeq α)
        s : Stream'.WSeq α
        IH : ∀ (s_1 : Stream'.WSeq α) (S_1 : Stream'.WSeq (Stream'.WSeq α)), Eq (s_1.a …
        ej : Eq (s.append S.join).think (s.append S.join).think
        m : Membership.mem (s.append S.join) a
        ⊢ Or (Membership.mem s a) (Exists fun s => And (Membership.mem S s) (Membershi …
      -/
      apply IH _ _ rfl m
      /-
        🎉 no goals
      -/


theorem exists_of_mem_bind {s : WSeq α} {f : α → WSeq β} {b} (h : b ∈ bind s f) :
    ∃ a ∈ s, b ∈ f a :=
  let ⟨t, tm, bt⟩ := exists_of_mem_join h
  let ⟨a, as, e⟩ := exists_of_mem_map tm
             /-
               α : Type u
               β : Type v
               s : Stream'.WSeq α
               f : α → Stream'.WSeq β
               b : β
               h : Membership.mem (s.bind f) b
               t : Stream'.WSeq β
               tm : Membership.mem (Stream'.WSeq.map f s) t
               bt : Membership.mem t b
               a : α
               as : Membership.mem s a
               e : Eq (f a) t
               ⊢ Membership.mem (f a) b
             -/
  ⟨a, as, by rwa [e]⟩
             /-
               🎉 no goals
             -/


theorem destruct_map (f : α → β) (s : WSeq α) :
    destruct (map f s) = Computation.map (Option.map (Prod.map f (map f))) (destruct s) := by
  apply
    Computation.eq_of_bisim fun c1 c2 =>
      ∃ s,
        c1 = destruct (map f s) ∧
          c2 = Computation.map (Option.map (Prod.map f (map f))) (destruct s)
    /-
      case bisim
      α : Type u
      β : Type v
      f : α → β
      s : Stream'.WSeq α
      ⊢ Computation.IsBisimulation fun c1 c2 => Exists fun s => And (Eq c1 (Stream'. …
    -/
  · intro c1 c2 h
    /-
      case bisim
      α : Type u
      β : Type v
      f : α → β
      s : Stream'.WSeq α
      c1 c2 : Computation (Option (Prod β (Stream'.WSeq β)))
      h : Exists fun s => And (Eq c1 (Stream'.WSeq.map f s).destruct) (Eq c2 (Comput …
      ⊢ Computation.BisimO (fun c1 c2 => Exists fun s => And (Eq c1 (Stream'.WSeq.ma …
    -/
    cases' h with s h
    /-
      case bisim.intro
      α : Type u
      β : Type v
      f : α → β
      s✝ : Stream'.WSeq α
      c1 c2 : Computation (Option (Prod β (Stream'.WSeq β)))
      s : Stream'.WSeq α
      h : And (Eq c1 (Stream'.WSeq.map f s).destruct) (Eq c2 (Computation.map (Optio …
      ⊢ Computation.BisimO (fun c1 c2 => Exists fun s => And (Eq c1 (Stream'.WSeq.ma …
    -/
    rw [h.left, h.right]
    /-
      case bisim.intro
      α : Type u
      β : Type v
      f : α → β
      s✝ : Stream'.WSeq α
      c1 c2 : Computation (Option (Prod β (Stream'.WSeq β)))
      s : Stream'.WSeq α
      h : And (Eq c1 (Stream'.WSeq.map f s).destruct) (Eq c2 (Computation.map (Optio …
      ⊢ Computation.BisimO (fun c1 c2 => Exists fun s => And (Eq c1 (Stream'.WSeq.ma …
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    induction' s using WSeq.recOn with a s s <;> simp
    /-
      case bisim.intro.h3
      α : Type u
      β : Type v
      f : α → β
      s✝ : Stream'.WSeq α
      c1 c2 : Computation (Option (Prod β (Stream'.WSeq β)))
      s : Stream'.WSeq α
      h : And (Eq c1 (Stream'.WSeq.map f s.think).destruct) (Eq c2 (Computation.map  …
      ⊢ Exists fun s_1 => And (Eq (Stream'.WSeq.map f s).destruct (Stream'.WSeq.map  …
    -/
    exact ⟨s, rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case r
      α : Type u
      β : Type v
      f : α → β
      s : Stream'.WSeq α
      ⊢ Exists fun s_1 => And (Eq (Stream'.WSeq.map f s).destruct (Stream'.WSeq.map  …
    -/
  · exact ⟨s, rfl, rfl⟩
    /-
      🎉 no goals
    -/


theorem liftRel_map {δ} (R : α → β → Prop) (S : γ → δ → Prop) {s1 : WSeq α} {s2 : WSeq β}
    {f1 : α → γ} {f2 : β → δ} (h1 : LiftRel R s1 s2) (h2 : ∀ {a b}, R a b → S (f1 a) (f2 b)) :
    LiftRel S (map f1 s1) (map f2 s2) :=
  ⟨fun s1 s2 => ∃ s t, s1 = map f1 s ∧ s2 = map f2 t ∧ LiftRel R s t, ⟨s1, s2, rfl, rfl, h1⟩,
    fun {s1 s2} h =>
    match s1, s2, h with
    | _, _, ⟨s, t, rfl, rfl, h⟩ => by
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        R : α → β → Prop
        S : γ → δ → Prop
        s1✝ : Stream'.WSeq α
        s2✝ : Stream'.WSeq β
        f1 : α → γ
        f2 : β → δ
        h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
        h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
        s1 : Stream'.WSeq γ
        s2 : Stream'.WSeq δ
        h✝ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.ma …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s t
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO S fun s1 s2 => Exists fun s => Ex …
      -/
      simp only [exists_and_left, destruct_map]
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        R : α → β → Prop
        S : γ → δ → Prop
        s1✝ : Stream'.WSeq α
        s2✝ : Stream'.WSeq β
        f1 : α → γ
        f2 : β → δ
        h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
        h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
        s1 : Stream'.WSeq γ
        s2 : Stream'.WSeq δ
        h✝ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.ma …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s t
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO S fun s1 s2 => Exists fun s => An …
      -/
      apply Computation.liftRel_map _ _ (liftRel_destruct h)
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        R : α → β → Prop
        S : γ → δ → Prop
        s1✝ : Stream'.WSeq α
        s2✝ : Stream'.WSeq β
        f1 : α → γ
        f2 : β → δ
        h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
        h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
        s1 : Stream'.WSeq γ
        s2 : Stream'.WSeq δ
        h✝ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.ma …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s t
        ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
      -/
      intro o p h
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        R : α → β → Prop
        S : γ → δ → Prop
        s1✝ : Stream'.WSeq α
        s2✝ : Stream'.WSeq β
        f1 : α → γ
        f2 : β → δ
        h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
        h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
        s1 : Stream'.WSeq γ
        s2 : Stream'.WSeq δ
        h✝¹ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.m …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : Stream'.WSeq.LiftRel R s t
        o : Option (Prod α (Stream'.WSeq α))
        p : Option (Prod β (Stream'.WSeq β))
        h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) o p
        ⊢ Stream'.WSeq.LiftRelO S (fun s1 s2 => Exists fun s => And (Eq s1 (Stream'.WS …
      -/
                                              /-
                                                🎉 no goals
                                              -/
      cases' o with a <;> cases' p with b <;> simp
        /-
          case none.some
          α : Type u
          β : Type v
          γ : Type w
          δ : Type u_1
          R : α → β → Prop
          S : γ → δ → Prop
          s1✝ : Stream'.WSeq α
          s2✝ : Stream'.WSeq β
          f1 : α → γ
          f2 : β → δ
          h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
          h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
          s1 : Stream'.WSeq γ
          s2 : Stream'.WSeq δ
          h✝¹ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.m …
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s t
          b : Prod β (Stream'.WSeq β)
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none (Option.some b)
          ⊢ False
        -/
      · cases b; cases h
                 /-
                   🎉 no goals
                 -/
        /-
          case some.none
          α : Type u
          β : Type v
          γ : Type w
          δ : Type u_1
          R : α → β → Prop
          S : γ → δ → Prop
          s1✝ : Stream'.WSeq α
          s2✝ : Stream'.WSeq β
          f1 : α → γ
          f2 : β → δ
          h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
          h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
          s1 : Stream'.WSeq γ
          s2 : Stream'.WSeq δ
          h✝¹ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.m …
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s t
          a : Prod α (Stream'.WSeq α)
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some a) Option.none
          ⊢ False
        -/
      · cases a; cases h
                 /-
                   🎉 no goals
                 -/
        /-
          case some.some
          α : Type u
          β : Type v
          γ : Type w
          δ : Type u_1
          R : α → β → Prop
          S : γ → δ → Prop
          s1✝ : Stream'.WSeq α
          s2✝ : Stream'.WSeq β
          f1 : α → γ
          f2 : β → δ
          h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
          h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
          s1 : Stream'.WSeq γ
          s2 : Stream'.WSeq δ
          h✝¹ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.m …
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s t
          a : Prod α (Stream'.WSeq α)
          b : Prod β (Stream'.WSeq β)
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some a) (Option.s …
          ⊢ And (S (f1 a.1) (f2 b.1)) (Exists fun s => And (Eq (Stream'.WSeq.map f1 a.2) …
        -/
      · cases' a with a s; cases' b with b t
        /-
          case some.some.mk.mk
          α : Type u
          β : Type v
          γ : Type w
          δ : Type u_1
          R : α → β → Prop
          S : γ → δ → Prop
          s1✝ : Stream'.WSeq α
          s2✝ : Stream'.WSeq β
          f1 : α → γ
          f2 : β → δ
          h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
          h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
          s1 : Stream'.WSeq γ
          s2 : Stream'.WSeq δ
          h✝¹ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.m …
          s✝ : Stream'.WSeq α
          t✝ : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s✝ t✝
          a : α
          s : Stream'.WSeq α
          b : β
          t : Stream'.WSeq β
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a,  …
          ⊢ And (S (f1 { fst := a, snd := s }.1) (f2 { fst := b, snd := t }.1)) (Exists  …
        -/
        cases' h with r h
        /-
          case some.some.mk.mk.intro
          α : Type u
          β : Type v
          γ : Type w
          δ : Type u_1
          R : α → β → Prop
          S : γ → δ → Prop
          s1✝ : Stream'.WSeq α
          s2✝ : Stream'.WSeq β
          f1 : α → γ
          f2 : β → δ
          h1 : Stream'.WSeq.LiftRel R s1✝ s2✝
          h2 : ∀ {a : α} {b : β}, R a b → S (f1 a) (f2 b)
          s1 : Stream'.WSeq γ
          s2 : Stream'.WSeq δ
          h✝¹ : (fun s1 s2 => Exists fun s => Exists fun t => And (Eq s1 (Stream'.WSeq.m …
          s✝ : Stream'.WSeq α
          t✝ : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s✝ t✝
          a : α
          s : Stream'.WSeq α
          b : β
          t : Stream'.WSeq β
          r : R a b
          h : Stream'.WSeq.LiftRel R s t
          ⊢ And (S (f1 { fst := a, snd := s }.1) (f2 { fst := b, snd := t }.1)) (Exists  …
        -/
        exact ⟨h2 r, s, rfl, t, rfl, h⟩⟩
        /-
          🎉 no goals
        -/


theorem map_congr (f : α → β) {s t : WSeq α} (h : s ~ʷ t) : map f s ~ʷ map f t :=
  liftRel_map _ _ h fun {_ _} => congr_arg _


/-- auxiliary definition of `destruct_append` over weak sequences -/
@[simp]
def destruct_append.aux (t : WSeq α) : Option (α × WSeq α) → Computation (Option (α × WSeq α))
  | none => destruct t
  | some (a, s) => Computation.pure (some (a, append s t))


theorem destruct_append (s t : WSeq α) :
    destruct (append s t) = (destruct s).bind (destruct_append.aux t) := by
  apply
    Computation.eq_of_bisim
      (fun c1 c2 =>
        ∃ s t, c1 = destruct (append s t) ∧ c2 = (destruct s).bind (destruct_append.aux t))
      _ ⟨s, t, rfl, rfl⟩
  /-
    α : Type u
    s t : Stream'.WSeq α
    ⊢ Computation.IsBisimulation fun c1 c2 => Exists fun s => Exists fun t => And  …
  -/
  intro c1 c2 h; rcases h with ⟨s, t, h⟩; rw [h.left, h.right]
  /-
    case intro.intro
    α : Type u
    s✝ t✝ : Stream'.WSeq α
    c1 c2 : Computation (Option (Prod α (Stream'.WSeq α)))
    s t : Stream'.WSeq α
    h : And (Eq c1 (s.append t).destruct) (Eq c2 (s.destruct.bind (Stream'.WSeq.de …
    ⊢ Computation.BisimO (fun c1 c2 => Exists fun s => Exists fun t => And (Eq c1  …
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  induction' s using WSeq.recOn with a s s <;> simp
    /-
      case intro.intro.h1
      α : Type u
      s t✝ : Stream'.WSeq α
      c1 c2 : Computation (Option (Prod α (Stream'.WSeq α)))
      t : Stream'.WSeq α
      h : And (Eq c1 (Stream'.WSeq.nil.append t).destruct) (Eq c2 (Stream'.WSeq.nil. …
      ⊢ Computation.BisimO.match_1 (fun x x => Prop) t.destruct.destruct t.destruct. …
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  · induction' t using WSeq.recOn with b t t <;> simp
      /-
        case intro.intro.h1.h3
        α : Type u
        s t✝ : Stream'.WSeq α
        c1 c2 : Computation (Option (Prod α (Stream'.WSeq α)))
        t : Stream'.WSeq α
        h : And (Eq c1 (Stream'.WSeq.nil.append t.think).destruct) (Eq c2 (Stream'.WSe …
        ⊢ Exists fun s => Exists fun t_1 => And (Eq t.destruct (s.append t_1).destruct …
      -/
                                  /-
                                    🎉 no goals
                                  -/
    · refine ⟨nil, t, ?_, ?_⟩ <;> simp
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case intro.intro.h3
      α : Type u
      s✝ t✝ : Stream'.WSeq α
      c1 c2 : Computation (Option (Prod α (Stream'.WSeq α)))
      t s : Stream'.WSeq α
      h : And (Eq c1 (s.think.append t).destruct) (Eq c2 (s.think.destruct.bind (Str …
      ⊢ Exists fun s_1 => Exists fun t_1 => And (Eq (s.append t).destruct (s_1.appen …
    -/
  · exact ⟨s, t, rfl, rfl⟩
    /-
      🎉 no goals
    -/


/-- auxiliary definition of `destruct_join` over weak sequences -/
@[simp]
def destruct_join.aux : Option (WSeq α × WSeq (WSeq α)) → Computation (Option (α × WSeq α))
  | none => Computation.pure none
  | some (s, S) => (destruct (append s (join S))).think


theorem destruct_join (S : WSeq (WSeq α)) :
    destruct (join S) = (destruct S).bind destruct_join.aux := by
  apply
    Computation.eq_of_bisim
      (fun c1 c2 =>
        c1 = c2 ∨ ∃ S, c1 = destruct (join S) ∧ c2 = (destruct S).bind destruct_join.aux)
      _ (Or.inr ⟨S, rfl, rfl⟩)
  /-
    α : Type u
    S : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Computation.IsBisimulation fun c1 c2 => Or (Eq c1 c2) (Exists fun S => And ( …
  -/
  intro c1 c2 h
  exact
    match c1, c2, h with
    | c, _, Or.inl <| rfl => by cases c.destruct <;> simp
    | _, _, Or.inr ⟨S, rfl, rfl⟩ => by
      induction' S using WSeq.recOn with s S S <;> simp
      · refine Or.inr ⟨S, rfl, rfl⟩


theorem liftRel_append (R : α → β → Prop) {s1 s2 : WSeq α} {t1 t2 : WSeq β} (h1 : LiftRel R s1 t1)
    (h2 : LiftRel R s2 t2) : LiftRel R (append s1 s2) (append t1 t2) :=
  ⟨fun s t => LiftRel R s t ∨ ∃ s1 t1, s = append s1 s2 ∧ t = append t1 t2 ∧ LiftRel R s1 t1,
    Or.inr ⟨s1, t1, rfl, rfl, h1⟩, fun {s t} h =>
    match s, t, h with
    | s, t, Or.inl h => by
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s1 s2 : Stream'.WSeq α
        t1 t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1 t1
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun  …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s t
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
      -/
      apply Computation.LiftRel.imp _ _ _ (liftRel_destruct h)
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s1 s2 : Stream'.WSeq α
        t1 t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1 t1
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun  …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s t
        ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
      -/
      intro a b; apply LiftRelO.imp_right
      /-
        case H
        α : Type u
        β : Type v
        R : α → β → Prop
        s1 s2 : Stream'.WSeq α
        t1 t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1 t1
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s✝ : Stream'.WSeq α
        t✝ : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun  …
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s t
        a : Option (Prod α (Stream'.WSeq α))
        b : Option (Prod β (Stream'.WSeq β))
        ⊢ ∀ (s : Stream'.WSeq α) (t : Stream'.WSeq β), Stream'.WSeq.LiftRel R s t → Or …
      -/
      intro s t; apply Or.inl
                 /-
                   🎉 no goals
                 -/
    | _, _, Or.inr ⟨s1, t1, rfl, rfl, h⟩ => by
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s1✝ s2 : Stream'.WSeq α
        t1✝ t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun  …
        s1 : Stream'.WSeq α
        t1 : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s1 t1
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
      -/
      simp only [LiftRelO, exists_and_left, destruct_append, destruct_append.aux]
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s1✝ s2 : Stream'.WSeq α
        t1✝ t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun  …
        s1 : Stream'.WSeq α
        t1 : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s1 t1
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
      -/
      apply Computation.liftRel_bind _ _ (liftRel_destruct h)
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s1✝ s2 : Stream'.WSeq α
        t1✝ t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun  …
        s1 : Stream'.WSeq α
        t1 : Stream'.WSeq β
        h : Stream'.WSeq.LiftRel R s1 t1
        ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
      -/
      intro o p h
      /-
        α : Type u
        β : Type v
        R : α → β → Prop
        s1✝ s2 : Stream'.WSeq α
        t1✝ t2 : Stream'.WSeq β
        h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
        h2 : Stream'.WSeq.LiftRel R s2 t2
        s : Stream'.WSeq α
        t : Stream'.WSeq β
        h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
        s1 : Stream'.WSeq α
        t1 : Stream'.WSeq β
        h✝ : Stream'.WSeq.LiftRel R s1 t1
        o : Option (Prod α (Stream'.WSeq α))
        p : Option (Prod β (Stream'.WSeq β))
        h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) o p
        ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
      -/
      cases' o with a <;> cases' p with b
        /-
          case none.none
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
      · simp only [destruct_append.aux]
        /-
          case none.none
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
        apply Computation.LiftRel.imp _ _ _ (liftRel_destruct h2)
        /-
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
          ⊢ ∀ {a : Option (Prod α (Stream'.WSeq α))} {b : Option (Prod β (Stream'.WSeq β …
        -/
        intro a b
        /-
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
          a : Option (Prod α (Stream'.WSeq α))
          b : Option (Prod β (Stream'.WSeq β))
          ⊢ Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) a b → Stream'.WSeq.LiftRelO …
        -/
        apply LiftRelO.imp_right
        /-
          case H
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
          a : Option (Prod α (Stream'.WSeq α))
          b : Option (Prod β (Stream'.WSeq β))
          ⊢ ∀ (s : Stream'.WSeq α) (t : Stream'.WSeq β), Stream'.WSeq.LiftRel R s t → Or …
        -/
        intro s t
        /-
          case H
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s✝ : Stream'.WSeq α
          t✝ : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none Option.none
          a : Option (Prod α (Stream'.WSeq α))
          b : Option (Prod β (Stream'.WSeq β))
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          ⊢ Stream'.WSeq.LiftRel R s t → Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1  …
        -/
        apply Or.inl
        /-
          🎉 no goals
        -/
        /-
          case none.some
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          b : Prod β (Stream'.WSeq β)
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) Option.none (Option.some b)
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
      · cases b; cases h
                 /-
                   🎉 no goals
                 -/
        /-
          case some.none
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          a : Prod α (Stream'.WSeq α)
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some a) Option.none
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
      · cases a; cases h
                 /-
                   🎉 no goals
                 -/
        /-
          case some.some
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s : Stream'.WSeq α
          t : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          a : Prod α (Stream'.WSeq α)
          b : Prod β (Stream'.WSeq β)
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some a) (Option.s …
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
      · cases' a with a s; cases' b with b t
        /-
          case some.some.mk.mk
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s✝ : Stream'.WSeq α
          t✝ : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          a : α
          s : Stream'.WSeq α
          b : β
          t : Stream'.WSeq β
          h : Stream'.WSeq.LiftRelO R (Stream'.WSeq.LiftRel R) (Option.some { fst := a,  …
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
        cases' h with r h
        -- Porting note: These 2 theorems should be excluded.
        /-
          case some.some.mk.mk.intro
          α : Type u
          β : Type v
          R : α → β → Prop
          s1✝ s2 : Stream'.WSeq α
          t1✝ t2 : Stream'.WSeq β
          h1 : Stream'.WSeq.LiftRel R s1✝ t1✝
          h2 : Stream'.WSeq.LiftRel R s2 t2
          s✝ : Stream'.WSeq α
          t✝ : Stream'.WSeq β
          h✝¹ : (fun s t => Or (Stream'.WSeq.LiftRel R s t) (Exists fun s1 => Exists fun …
          s1 : Stream'.WSeq α
          t1 : Stream'.WSeq β
          h✝ : Stream'.WSeq.LiftRel R s1 t1
          a : α
          s : Stream'.WSeq α
          b : β
          t : Stream'.WSeq β
          r : R a b
          h : Stream'.WSeq.LiftRel R s t
          ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s t => Or (Stream'.WSeq.Lif …
        -/
        simpa [-liftRel_pure_left, -liftRel_pure_right] using ⟨r, Or.inr ⟨s, rfl, t, rfl, h⟩⟩⟩
        /-
          🎉 no goals
        -/


theorem liftRel_join.lem (R : α → β → Prop) {S T} {U : WSeq α → WSeq β → Prop}
    (ST : LiftRel (LiftRel R) S T)
    (HU :
      ∀ s1 s2,
        (∃ s t S T,
            s1 = append s (join S) ∧
              s2 = append t (join T) ∧ LiftRel R s t ∧ LiftRel (LiftRel R) S T) →
          U s1 s2)
    {a} (ma : a ∈ destruct (join S)) : ∃ b, b ∈ destruct (join T) ∧ LiftRelO R U a b := by
  /-
    α : Type u
    β : Type v
    R : α → β → Prop
    S : Stream'.WSeq (Stream'.WSeq α)
    T : Stream'.WSeq (Stream'.WSeq β)
    U : Stream'.WSeq α → Stream'.WSeq β → Prop
    ST : Stream'.WSeq.LiftRel (Stream'.WSeq.LiftRel R) S T
    HU : ∀ (s1 : Stream'.WSeq α) (s2 : Stream'.WSeq β), (Exists fun s => Exists fu …
    a : Option (Prod α (Stream'.WSeq α))
    ma : Membership.mem S.join.destruct a
    ⊢ Exists fun b => And (Membership.mem T.join.destruct b) (Stream'.WSeq.LiftRel …
  -/
  cases' exists_results_of_mem ma with n h; clear ma; revert S T ST a
  /-
    case intro
    α : Type u
    β : Type v
    R : α → β → Prop
    U : Stream'.WSeq α → Stream'.WSeq β → Prop
    HU : ∀ (s1 : Stream'.WSeq α) (s2 : Stream'.WSeq β), (Exists fun s => Exists fu …
    n : Nat
    ⊢ ∀ {S : Stream'.WSeq (Stream'.WSeq α)} {T : Stream'.WSeq (Stream'.WSeq β)}, S …
  -/
  induction' n using Nat.strongRecOn with n IH
  /-
    case intro.ind
    α : Type u
    β : Type v
    R : α → β → Prop
    U : Stream'.WSeq α → Stream'.WSeq β → Prop
    HU : ∀ (s1 : Stream'.WSeq α) (s2 : Stream'.WSeq β), (Exists fun s => Exists fu …
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ {S : Stream'.WSeq (Stream'.WSeq α)} {T : Strea …
    ⊢ ∀ {S : Stream'.WSeq (Stream'.WSeq α)} {T : Stream'.WSeq (Stream'.WSeq β)}, S …
  -/
  intro S T ST a ra; simp only [destruct_join] at ra
  exact
    let ⟨o, m, k, rs1, rs2, en⟩ := of_results_bind ra
    let ⟨p, mT, rop⟩ := Computation.exists_of_liftRel_left (liftRel_destruct ST) rs1.mem
    match o, p, rop, rs1, rs2, mT with
    | none, none, _, _, rs2, mT => by
      simp only [destruct_join]
      exact ⟨none, mem_bind mT (ret_mem _), by rw [eq_of_pure_mem rs2.mem]; trivial⟩
    | some (s, S'), some (t, T'), ⟨st, ST'⟩, _, rs2, mT => by
      simp? [destruct_append]  at rs2  says simp only [destruct_join.aux, destruct_append] at rs2
      exact
        let ⟨k1, rs3, ek⟩ := of_results_think rs2
        let ⟨o', m1, n1, rs4, rs5, ek1⟩ := of_results_bind rs3
        let ⟨p', mt, rop'⟩ := Computation.exists_of_liftRel_left (liftRel_destruct st) rs4.mem
        match o', p', rop', rs4, rs5, mt with
        | none, none, _, _, rs5', mt => by
          have : n1 < n := by
            rw [en, ek, ek1]
            apply lt_of_lt_of_le _ (Nat.le_add_right _ _)
            apply Nat.lt_succ_of_le (Nat.le_add_right _ _)
          let ⟨ob, mb, rob⟩ := IH _ this ST' rs5'
          refine ⟨ob, ?_, rob⟩
          · simp (config := { unfoldPartialApp := true }) only [destruct_join, destruct_join.aux]
            apply mem_bind mT
            simp only [destruct_append, destruct_append.aux]
            apply think_mem
            apply mem_bind mt
            exact mb
        | some (a, s'), some (b, t'), ⟨ab, st'⟩, _, rs5, mt => by
          simp?  at rs5  says simp only [destruct_append.aux] at rs5
          refine ⟨some (b, append t' (join T')), ?_, ?_⟩
          · simp (config := { unfoldPartialApp := true }) only [destruct_join, destruct_join.aux]
            apply mem_bind mT
            simp only [destruct_append, destruct_append.aux]
            apply think_mem
            apply mem_bind mt
            apply ret_mem
          rw [eq_of_pure_mem rs5.mem]
          exact ⟨ab, HU _ _ ⟨s', t', S', T', rfl, rfl, st', ST'⟩⟩


theorem liftRel_join (R : α → β → Prop) {S : WSeq (WSeq α)} {T : WSeq (WSeq β)}
    (h : LiftRel (LiftRel R) S T) : LiftRel R (join S) (join T) :=
  ⟨fun s1 s2 =>
    ∃ s t S T,
      s1 = append s (join S) ∧ s2 = append t (join T) ∧ LiftRel R s t ∧ LiftRel (LiftRel R) S T,
                        /-
                          α : Type u
                          β : Type v
                          R : α → β → Prop
                          S : Stream'.WSeq (Stream'.WSeq α)
                          T : Stream'.WSeq (Stream'.WSeq β)
                          h : Stream'.WSeq.LiftRel (Stream'.WSeq.LiftRel R) S T
                          ⊢ Eq S.join (Stream'.WSeq.nil.append S.join)
                        -/
                        /-
                          🎉 no goals
                        -/
                                 /-
                                   🎉 no goals
                                 -/
    ⟨nil, nil, S, T, by simp, by simp, by simp, h⟩, fun {s1 s2} ⟨s, t, S, T, h1, h2, st, ST⟩ => by
                                          /-
                                            🎉 no goals
                                          -/
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      S✝ : Stream'.WSeq (Stream'.WSeq α)
      T✝ : Stream'.WSeq (Stream'.WSeq β)
      h : Stream'.WSeq.LiftRel (Stream'.WSeq.LiftRel R) S✝ T✝
      s1 : Stream'.WSeq α
      s2 : Stream'.WSeq β
      x✝ : (fun s1 s2 => Exists fun s => Exists fun t => Exists fun S => Exists fun  …
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      S : Stream'.WSeq (Stream'.WSeq α)
      T : Stream'.WSeq (Stream'.WSeq β)
      h1 : Eq s1 (s.append S.join)
      h2 : Eq s2 (t.append T.join)
      st : Stream'.WSeq.LiftRel R s t
      ST : Stream'.WSeq.LiftRel (Stream'.WSeq.LiftRel R) S T
      ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s1 s2 => Exists fun s => Ex …
    -/
    rw [h1, h2]; rw [destruct_append, destruct_append]
    /-
      α : Type u
      β : Type v
      R : α → β → Prop
      S✝ : Stream'.WSeq (Stream'.WSeq α)
      T✝ : Stream'.WSeq (Stream'.WSeq β)
      h : Stream'.WSeq.LiftRel (Stream'.WSeq.LiftRel R) S✝ T✝
      s1 : Stream'.WSeq α
      s2 : Stream'.WSeq β
      x✝ : (fun s1 s2 => Exists fun s => Exists fun t => Exists fun S => Exists fun  …
      s : Stream'.WSeq α
      t : Stream'.WSeq β
      S : Stream'.WSeq (Stream'.WSeq α)
      T : Stream'.WSeq (Stream'.WSeq β)
      h1 : Eq s1 (s.append S.join)
      h2 : Eq s2 (t.append T.join)
      st : Stream'.WSeq.LiftRel R s t
      ST : Stream'.WSeq.LiftRel (Stream'.WSeq.LiftRel R) S T
      ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO R fun s1 s2 => Exists fun s => Ex …
    -/
    apply Computation.liftRel_bind _ _ (liftRel_destruct st)
    exact fun {o p} h =>
      match o, p, h with
      | some (a, s), some (b, t), ⟨h1, h2⟩ => by
        -- Porting note: These 2 theorems should be excluded.
        simpa [-liftRel_pure_left, -liftRel_pure_right] using ⟨h1, s, t, S, rfl, T, rfl, h2, ST⟩
      | none, none, _ => by
        -- Porting note: `LiftRelO` should be excluded.
        dsimp [destruct_append.aux, Computation.LiftRel, -LiftRelO]; constructor
        · intro
          apply liftRel_join.lem _ ST fun _ _ => id
        · intro b mb
          rw [← LiftRelO.swap]
          apply liftRel_join.lem (swap R)
          · rw [← LiftRel.swap R, ← LiftRel.swap]
            apply ST
          · rw [← LiftRel.swap R, ← LiftRel.swap (LiftRel R)]
            exact fun s1 s2 ⟨s, t, S, T, h1, h2, st, ST⟩ => ⟨t, s, T, S, h2, h1, st, ST⟩
          · exact mb⟩


theorem join_congr {S T : WSeq (WSeq α)} (h : LiftRel Equiv S T) : join S ~ʷ join T :=
  liftRel_join _ h


theorem liftRel_bind {δ} (R : α → β → Prop) (S : γ → δ → Prop) {s1 : WSeq α} {s2 : WSeq β}
    {f1 : α → WSeq γ} {f2 : β → WSeq δ} (h1 : LiftRel R s1 s2)
    (h2 : ∀ {a b}, R a b → LiftRel S (f1 a) (f2 b)) : LiftRel S (bind s1 f1) (bind s2 f2) :=
  liftRel_join _ (liftRel_map _ _ h1 @h2)


theorem bind_congr {s1 s2 : WSeq α} {f1 f2 : α → WSeq β} (h1 : s1 ~ʷ s2) (h2 : ∀ a, f1 a ~ʷ f2 a) :
    bind s1 f1 ~ʷ bind s2 f2 :=
                                        /-
                                          α : Type u
                                          β : Type v
                                          s1 s2 : Stream'.WSeq α
                                          f1 f2 : α → Stream'.WSeq β
                                          h1 : s1.Equiv s2
                                          h2 : ∀ (a : α), (f1 a).Equiv (f2 a)
                                          a b : α
                                          h : Eq a b
                                          ⊢ Stream'.WSeq.LiftRel (fun x1 x2 => Eq x1 x2) (f1 a) (f2 b)
                                        -/
  liftRel_bind _ _ h1 fun {a b} h => by rw [h]; apply h2
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                                        /-
                                                          α : Type u
                                                          s : Stream'.WSeq α
                                                          ⊢ (Stream'.WSeq.ret s).join.Equiv s
                                                        -/
theorem join_ret (s : WSeq α) : join (ret s) ~ʷ s := by simpa [ret] using think_equiv _
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem join_map_ret (s : WSeq α) : join (map ret s) ~ʷ s := by
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ (Stream'.WSeq.map Stream'.WSeq.ret s).join.Equiv s
  -/
  refine ⟨fun s1 s2 => join (map ret s2) = s1, rfl, ?_⟩
  /-
    α : Type u
    s : Stream'.WSeq α
    ⊢ ∀ {s t : Stream'.WSeq α}, (fun s1 s2 => Eq (Stream'.WSeq.map Stream'.WSeq.re …
  -/
  intro s' s h; rw [← h]
  /-
    α : Type u
    s✝ s' s : Stream'.WSeq α
    h : Eq (Stream'.WSeq.map Stream'.WSeq.ret s).join s'
    ⊢ Computation.LiftRel (Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) fun s1 s2 …
  -/
  apply liftRel_rec fun c1 c2 => ∃ s, c1 = destruct (join (map ret s)) ∧ c2 = destruct s
  · exact fun {c1 c2} h =>
      match c1, c2, h with
      | _, _, ⟨s, rfl, rfl⟩ => by
        clear h
        -- Porting note: `ret` is simplified in `simp` so `ret`s become `fun a => cons a nil` here.
        have : ∀ s, ∃ s' : WSeq α,
            (map (fun a => cons a nil) s).join.destruct =
              (map (fun a => cons a nil) s').join.destruct ∧ destruct s = s'.destruct :=
          fun s => ⟨s, rfl, rfl⟩
        induction' s using WSeq.recOn with a s s <;>
          simp (config := { unfoldPartialApp := true }) [ret, ret_mem, this, Option.exists]
    /-
      case Hc
      α : Type u
      s✝ s' s : Stream'.WSeq α
      h : Eq (Stream'.WSeq.map Stream'.WSeq.ret s).join s'
      ⊢ Exists fun s_1 => And (Eq (Stream'.WSeq.map Stream'.WSeq.ret s).join.destruc …
    -/
  · exact ⟨s, rfl, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem join_append (S T : WSeq (WSeq α)) : join (append S T) ~ʷ append (join S) (join T) := by
  refine
    ⟨fun s1 s2 =>
      ∃ s S T, s1 = append s (join (append S T)) ∧ s2 = append s (append (join S) (join T)),
      ⟨nil, S, T, by simp, by simp⟩, ?_⟩
  /-
    α : Type u
    S T : Stream'.WSeq (Stream'.WSeq α)
    ⊢ ∀ {s t : Stream'.WSeq α}, (fun s1 s2 => Exists fun s => Exists fun S => Exis …
  -/
  intro s1 s2 h
  apply
    liftRel_rec
      (fun c1 c2 =>
        ∃ (s : WSeq α) (S T : _),
          c1 = destruct (append s (join (append S T))) ∧
            c2 = destruct (append s (append (join S) (join T))))
      _ _ _
      (let ⟨s, S, T, h1, h2⟩ := h
      ⟨s, S, T, congr_arg destruct h1, congr_arg destruct h2⟩)
  /-
    α : Type u
    S T : Stream'.WSeq (Stream'.WSeq α)
    s1 s2 : Stream'.WSeq α
    h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
    ⊢ ∀ {ca cb : Computation (Option (Prod α (Stream'.WSeq α)))}, (fun c1 c2 => Ex …
  -/
  rintro c1 c2 ⟨s, S, T, rfl, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    S✝ T✝ : Stream'.WSeq (Stream'.WSeq α)
    s1 s2 : Stream'.WSeq α
    h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
    s : Stream'.WSeq α
    S T : Stream'.WSeq (Stream'.WSeq α)
    ⊢ Computation.LiftRelAux (Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) fun s1 …
  -/
  induction' s using WSeq.recOn with a s s <;> simp
    /-
      case intro.intro.intro.intro.h1
      α : Type u
      S✝ T✝ : Stream'.WSeq (Stream'.WSeq α)
      s1 s2 : Stream'.WSeq α
      h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
      S T : Stream'.WSeq (Stream'.WSeq α)
      ⊢ Computation.LiftRelAux (Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) fun s1 …
    -/
  · induction' S using WSeq.recOn with s S S <;> simp
      /-
        case intro.intro.intro.intro.h1.h1
        α : Type u
        S T✝ : Stream'.WSeq (Stream'.WSeq α)
        s1 s2 : Stream'.WSeq α
        h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
        T : Stream'.WSeq (Stream'.WSeq α)
        ⊢ Computation.LiftRelAux (Stream'.WSeq.LiftRelO (fun x1 x2 => Eq x1 x2) fun s1 …
      -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    · induction' T using WSeq.recOn with s T T <;> simp
        /-
          case intro.intro.intro.intro.h1.h1.h2
          α : Type u
          S T✝ : Stream'.WSeq (Stream'.WSeq α)
          s1 s2 : Stream'.WSeq α
          h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
          s : Stream'.WSeq α
          T : Stream'.WSeq (Stream'.WSeq α)
          ⊢ Exists fun s_1 => Exists fun S => Exists fun T_1 => And (Eq (s.append T.join …
        -/
                                       /-
                                         🎉 no goals
                                       -/
      · refine ⟨s, nil, T, ?_, ?_⟩ <;> simp
                                       /-
                                         🎉 no goals
                                       -/
        /-
          case intro.intro.intro.intro.h1.h1.h3
          α : Type u
          S T✝ : Stream'.WSeq (Stream'.WSeq α)
          s1 s2 : Stream'.WSeq α
          h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
          T : Stream'.WSeq (Stream'.WSeq α)
          ⊢ Exists fun s => Exists fun S => Exists fun T_1 => And (Eq T.join.destruct (s …
        -/
                                         /-
                                           🎉 no goals
                                         -/
      · refine ⟨nil, nil, T, ?_, ?_⟩ <;> simp
                                         /-
                                           🎉 no goals
                                         -/
      /-
        case intro.intro.intro.intro.h1.h2
        α : Type u
        S✝ T✝ : Stream'.WSeq (Stream'.WSeq α)
        s1 s2 : Stream'.WSeq α
        h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
        T : Stream'.WSeq (Stream'.WSeq α)
        s : Stream'.WSeq α
        S : Stream'.WSeq (Stream'.WSeq α)
        ⊢ Exists fun s_1 => Exists fun S_1 => Exists fun T_1 => And (Eq (s.append (S.a …
      -/
    · exact ⟨s, S, T, rfl, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.h1.h3
        α : Type u
        S✝ T✝ : Stream'.WSeq (Stream'.WSeq α)
        s1 s2 : Stream'.WSeq α
        h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
        T S : Stream'.WSeq (Stream'.WSeq α)
        ⊢ Exists fun s => Exists fun S_1 => Exists fun T_1 => And (Eq (S.append T).joi …
      -/
                                     /-
                                       🎉 no goals
                                     -/
    · refine ⟨nil, S, T, ?_, ?_⟩ <;> simp
                                     /-
                                       🎉 no goals
                                     -/
    /-
      case intro.intro.intro.intro.h2
      α : Type u
      S✝ T✝ : Stream'.WSeq (Stream'.WSeq α)
      s1 s2 : Stream'.WSeq α
      h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
      S T : Stream'.WSeq (Stream'.WSeq α)
      a : α
      s : Stream'.WSeq α
      ⊢ Exists fun s_1 => Exists fun S_1 => Exists fun T_1 => And (Eq (s.append (S.a …
    -/
  · exact ⟨s, S, T, rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.h3
      α : Type u
      S✝ T✝ : Stream'.WSeq (Stream'.WSeq α)
      s1 s2 : Stream'.WSeq α
      h : Exists fun s => Exists fun S => Exists fun T => And (Eq s1 (s.append (S.ap …
      S T : Stream'.WSeq (Stream'.WSeq α)
      s : Stream'.WSeq α
      ⊢ Exists fun s_1 => Exists fun S_1 => Exists fun T_1 => And (Eq (s.append (S.a …
    -/
  · exact ⟨s, S, T, rfl, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem bind_ret (f : α → β) (s) : bind s (ret ∘ f) ~ʷ map f s := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Stream'.WSeq α
    ⊢ (s.bind (Function.comp Stream'.WSeq.ret f)).Equiv (Stream'.WSeq.map f s)
  -/
  dsimp [bind]
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Stream'.WSeq α
    ⊢ (Stream'.WSeq.map (Function.comp Stream'.WSeq.ret f) s).join.Equiv (Stream'. …
  -/
  rw [map_comp]
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Stream'.WSeq α
    ⊢ (Stream'.WSeq.map Stream'.WSeq.ret (Stream'.WSeq.map f s)).join.Equiv (Strea …
  -/
  apply join_map_ret
  /-
    🎉 no goals
  -/


@[simp]
                                                                        /-
                                                                          α : Type u
                                                                          β : Type v
                                                                          a : α
                                                                          f : α → Stream'.WSeq β
                                                                          ⊢ ((Stream'.WSeq.ret a).bind f).Equiv (f a)
                                                                        -/
theorem ret_bind (a : α) (f : α → WSeq β) : bind (ret a) f ~ʷ f a := by simp [bind]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem map_join (f : α → β) (S) : map f (join S) = join (map (map f) S) := by
  apply
    Seq.eq_of_bisim fun s1 s2 =>
      ∃ s S, s1 = append s (map f (join S)) ∧ s2 = append s (join (map (map f) S))
    /-
      case bisim
      α : Type u
      β : Type v
      f : α → β
      S : Stream'.WSeq (Stream'.WSeq α)
      ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => Exists fun S => And  …
    -/
  · intro s1 s2 h
    exact
      match s1, s2, h with
      | _, _, ⟨s, S, rfl, rfl⟩ => by
        induction' s using WSeq.recOn with a s s <;> simp
        · induction' S using WSeq.recOn with s S S <;> simp
          · exact ⟨map f s, S, rfl, rfl⟩
          · refine ⟨nil, S, ?_, ?_⟩ <;> simp
        · exact ⟨_, _, rfl, rfl⟩
        · exact ⟨_, _, rfl, rfl⟩
    /-
      case r
      α : Type u
      β : Type v
      f : α → β
      S : Stream'.WSeq (Stream'.WSeq α)
      ⊢ Exists fun s => Exists fun S_1 => And (Eq (Stream'.WSeq.map f S.join) (s.app …
    -/
                                /-
                                  🎉 no goals
                                -/
  · refine ⟨nil, S, ?_, ?_⟩ <;> simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem join_join (SS : WSeq (WSeq (WSeq α))) : join (join SS) ~ʷ join (map join SS) := by
  refine
    ⟨fun s1 s2 =>
      ∃ s S SS,
        s1 = append s (join (append S (join SS))) ∧
          s2 = append s (append (join S) (join (map join SS))),
      ⟨nil, nil, SS, by simp, by simp⟩, ?_⟩
  /-
    α : Type u
    SS : Stream'.WSeq (Stream'.WSeq (Stream'.WSeq α))
    ⊢ ∀ {s t : Stream'.WSeq α}, (fun s1 s2 => Exists fun s => Exists fun S => Exis …
  -/
  intro s1 s2 h
  apply
    liftRel_rec
      (fun c1 c2 =>
        ∃ s S SS,
          c1 = destruct (append s (join (append S (join SS)))) ∧
            c2 = destruct (append s (append (join S) (join (map join SS)))))
      _ (destruct s1) (destruct s2)
      (let ⟨s, S, SS, h1, h2⟩ := h
      ⟨s, S, SS, by simp [h1], by simp [h2]⟩)
  /-
    α : Type u
    SS : Stream'.WSeq (Stream'.WSeq (Stream'.WSeq α))
    s1 s2 : Stream'.WSeq α
    h : Exists fun s => Exists fun S => Exists fun SS => And (Eq s1 (s.append (S.a …
    ⊢ ∀ {ca cb : Computation (Option (Prod α (Stream'.WSeq α)))}, (fun c1 c2 => Ex …
  -/
  intro c1 c2 h
  exact
    match c1, c2, h with
    | _, _, ⟨s, S, SS, rfl, rfl⟩ => by
      clear h
      induction' s using WSeq.recOn with a s s <;> simp
      · induction' S using WSeq.recOn with s S S <;> simp
        · induction' SS using WSeq.recOn with S SS SS <;> simp
          · refine ⟨nil, S, SS, ?_, ?_⟩ <;> simp
          · refine ⟨nil, nil, SS, ?_, ?_⟩ <;> simp
        · exact ⟨s, S, SS, rfl, rfl⟩
        · refine ⟨nil, S, SS, ?_, ?_⟩ <;> simp
      · exact ⟨s, S, SS, rfl, rfl⟩
      · exact ⟨s, S, SS, rfl, rfl⟩


@[simp]
theorem bind_assoc (s : WSeq α) (f : α → WSeq β) (g : β → WSeq γ) :
    bind (bind s f) g ~ʷ bind s fun x : α => bind (f x) g := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    s : Stream'.WSeq α
    f : α → Stream'.WSeq β
    g : β → Stream'.WSeq γ
    ⊢ ((s.bind f).bind g).Equiv (s.bind fun x => (f x).bind g)
  -/
  simp only [bind, map_join]; erw [← map_comp f (map g), map_comp (map g ∘ f) join]
  /-
    α : Type u
    β : Type v
    γ : Type w
    s : Stream'.WSeq α
    f : α → Stream'.WSeq β
    g : β → Stream'.WSeq γ
    ⊢ (Stream'.WSeq.map (Function.comp (Stream'.WSeq.map g) f) s).join.join.Equiv  …
  -/
  apply join_join
  /-
    🎉 no goals
  -/


instance monad : Monad WSeq where
  map := @map
  pure := @ret
  bind := @bind

/-
  Unfortunately, WSeq is not a lawful monad, because it does not satisfy
  the monad laws exactly, only up to sequence equivalence.
  Furthermore, even quotienting by the equivalence is not sufficient,
  because the join operation involves lists of quotient elements,
  with a lifted equivalence relation, and pure quotients cannot handle
  this type of construction.

instance lawfulMonad : LawfulMonad WSeq :=
  { id_map := @map_id,
    bind_pure_comp := @bind_ret,
    pure_bind := @ret_bind,
    bind_assoc := @bind_assoc }
-/

