/-- A stream `s : Option α` is a sequence if `s.get n = none` implies `s.get (n + 1) = none`.
-/
def IsSeq {α : Type u} (s : Stream' (Option α)) : Prop :=
  ∀ {n : ℕ}, s n = none → s (n + 1) = none


/-- `Seq α` is the type of possibly infinite lists (referred here as sequences).
  It is encoded as an infinite stream of options such that if `f n = none`, then
  `f m = none` for all `m ≥ n`. -/
def Seq (α : Type u) : Type u :=
  { f : Stream' (Option α) // f.IsSeq }


/-- `Seq1 α` is the type of nonempty sequences. -/
def Seq1 (α) :=
  α × Seq α


/-- The empty sequence -/
def nil : Seq α :=
  ⟨Stream'.const none, fun {_} _ => rfl⟩


instance : Inhabited (Seq α) :=
  ⟨nil⟩


/-- Prepend an element to a sequence -/
def cons (a : α) (s : Seq α) : Seq α :=
  ⟨some a::s.1, by
    /-
      α : Type u
      β : Type v
      γ : Type w
      a : α
      s : Stream'.Seq α
      ⊢ (Stream'.cons (Option.some a) ↑s).IsSeq
    -/
    rintro (n | _) h
      /-
        case zero
        α : Type u
        β : Type v
        γ : Type w
        a : α
        s : Stream'.Seq α
        h : Eq (Stream'.cons (Option.some a) (↑s) 0) Option.none
        ⊢ Eq (Stream'.cons (Option.some a) (↑s) (HAdd.hAdd 0 1)) Option.none
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
        a : α
        s : Stream'.Seq α
        n✝ : Nat
        h : Eq (Stream'.cons (Option.some a) (↑s) (HAdd.hAdd n✝ 1)) Option.none
        ⊢ Eq (Stream'.cons (Option.some a) (↑s) (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)) Option …
      -/
    · exact s.2 h⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem val_cons (s : Seq α) (x : α) : (cons x s).val = some x::s.val :=
  rfl


/-- Get the nth element of a sequence (if it exists) -/
def get? : Seq α → ℕ → Option α :=
  Subtype.val


@[simp]
theorem get?_mk (f hf) : @get? α ⟨f, hf⟩ = f :=
  rfl


@[simp]
theorem get?_nil (n : ℕ) : (@nil α).get? n = none :=
  rfl


@[simp]
theorem get?_cons_zero (a : α) (s : Seq α) : (cons a s).get? 0 = some a :=
  rfl


@[simp]
theorem get?_cons_succ (a : α) (s : Seq α) (n : ℕ) : (cons a s).get? (n + 1) = s.get? n :=
  rfl


@[ext]
protected theorem ext {s t : Seq α} (h : ∀ n : ℕ, s.get? n = t.get? n) : s = t :=
  Subtype.eq <| funext h


theorem cons_injective2 : Function.Injective2 (cons : α → Seq α → Seq α) := fun x y s t h =>
      /-
        α : Type u
        x y : α
        s t : Stream'.Seq α
        h : Eq (Stream'.Seq.cons x s) (Stream'.Seq.cons y t)
        ⊢ Eq x y
      -/
  ⟨by rw [← Option.some_inj, ← get?_cons_zero, h, get?_cons_zero],
      /-
        🎉 no goals
      -/
                        /-
                          α : Type u
                          x y : α
                          s t : Stream'.Seq α
                          h : Eq (Stream'.Seq.cons x s) (Stream'.Seq.cons y t)
                          n : Nat
                          ⊢ Eq (s.get? n) (t.get? n)
                        -/
    Seq.ext fun n => by simp_rw [← get?_cons_succ x s n, h, get?_cons_succ]⟩
                        /-
                          🎉 no goals
                        -/


theorem cons_left_injective (s : Seq α) : Function.Injective fun x => cons x s :=
  cons_injective2.left _


theorem cons_right_injective (x : α) : Function.Injective (cons x) :=
  cons_injective2.right _


/-- A sequence has terminated at position `n` if the value at position `n` equals `none`. -/
def TerminatedAt (s : Seq α) (n : ℕ) : Prop :=
  s.get? n = none


/-- It is decidable whether a sequence terminates at a given position. -/
instance terminatedAtDecidable (s : Seq α) (n : ℕ) : Decidable (s.TerminatedAt n) :=
                                            /-
                                              α : Type u
                                              β : Type v
                                              γ : Type w
                                              s : Stream'.Seq α
                                              n : Nat
                                              ⊢ Iff (s.TerminatedAt n) (Eq (s.get? n).isNone Bool.true)
                                            -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  decidable_of_iff' (s.get? n).isNone <| by unfold TerminatedAt; cases s.get? n <;> simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- A sequence terminates if there is some position `n` at which it has terminated. -/
def Terminates (s : Seq α) : Prop :=
  ∃ n : ℕ, s.TerminatedAt n


theorem not_terminates_iff {s : Seq α} : ¬s.Terminates ↔ ∀ n, (s.get? n).isSome := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Iff (Not s.Terminates) (∀ (n : Nat), Eq (s.get? n).isSome Bool.true)
  -/
  simp only [Terminates, TerminatedAt, ← Ne.eq_def, Option.ne_none_iff_isSome, not_exists, iff_self]
  /-
    🎉 no goals
  -/


/-- Functorial action of the functor `Option (α × _)` -/
@[simp]
def omap (f : β → γ) : Option (α × β) → Option (α × γ)
  | none => none
  | some (a, b) => some (a, f b)


/-- Get the first element of a sequence -/
def head (s : Seq α) : Option α :=
  get? s 0


/-- Get the tail of a sequence (or `nil` if the sequence is `nil`) -/
def tail (s : Seq α) : Seq α :=
  ⟨s.1.tail, fun n' => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      s : Stream'.Seq α
      n✝ : Nat
      n' : Eq ((↑s).tail n✝) Option.none
      ⊢ Eq ((↑s).tail (HAdd.hAdd n✝ 1)) Option.none
    -/
    cases' s with f al
    /-
      case mk
      α : Type u
      β : Type v
      γ : Type w
      n✝ : Nat
      f : Stream' (Option α)
      al : f.IsSeq
      n' : Eq ((↑⟨f, al⟩).tail n✝) Option.none
      ⊢ Eq ((↑⟨f, al⟩).tail (HAdd.hAdd n✝ 1)) Option.none
    -/
    exact al n'⟩
    /-
      🎉 no goals
    -/


/-- member definition for `Seq`-/
protected def Mem (s : Seq α) (a : α) :=
  some a ∈ s.1


instance : Membership α (Seq α) :=
  ⟨Seq.Mem⟩


theorem le_stable (s : Seq α) {m n} (h : m ≤ n) : s.get? m = none → s.get? n = none := by
  /-
    α : Type u
    s : Stream'.Seq α
    m n : Nat
    h : LE.le m n
    ⊢ Eq (s.get? m) Option.none → Eq (s.get? n) Option.none
  -/
  cases' s with f al
  /-
    case mk
    α : Type u
    m n : Nat
    h : LE.le m n
    f : Stream' (Option α)
    al : f.IsSeq
    ⊢ Eq (Stream'.Seq.get? ⟨f, al⟩ m) Option.none → Eq (Stream'.Seq.get? ⟨f, al⟩ n …
  -/
  induction' h with n _ IH
  /-
    case mk.refl
    α : Type u
    m n : Nat
    f : Stream' (Option α)
    al : f.IsSeq
    ⊢ Eq (Stream'.Seq.get? ⟨f, al⟩ m) Option.none → Eq (Stream'.Seq.get? ⟨f, al⟩ m …
  -/
  exacts [id, fun h2 => al (IH h2)]
  /-
    🎉 no goals
  -/


/-- If a sequence terminated at position `n`, it also terminated at `m ≥ n`. -/
theorem terminated_stable : ∀ (s : Seq α) {m n : ℕ}, m ≤ n → s.TerminatedAt m → s.TerminatedAt n :=
  le_stable


/-- If `s.get? n = some aₙ` for some value `aₙ`, then there is also some value `aₘ` such
that `s.get? = some aₘ` for `m ≤ n`.
-/
theorem ge_stable (s : Seq α) {aₙ : α} {n m : ℕ} (m_le_n : m ≤ n)
    (s_nth_eq_some : s.get? n = some aₙ) : ∃ aₘ : α, s.get? m = some aₘ :=
                               /-
                                 α : Type u
                                 s : Stream'.Seq α
                                 aₙ : α
                                 n m : Nat
                                 m_le_n : LE.le m n
                                 s_nth_eq_some : Eq (s.get? n) (Option.some aₙ)
                                 ⊢ Ne (s.get? n) Option.none
                               -/
  have : s.get? n ≠ none := by simp [s_nth_eq_some]
                               /-
                                 🎉 no goals
                               -/
  have : s.get? m ≠ none := mt (s.le_stable m_le_n) this
  Option.ne_none_iff_exists'.mp this


                                                                               /-
                                                                                 α : Type u
                                                                                 a : α
                                                                                 x✝ : Membership.mem Stream'.Seq.nil a
                                                                                 w✝ : Nat
                                                                                 h : Eq (Option.some a) Option.none
                                                                                 ⊢ False
                                                                               -/
theorem not_mem_nil (a : α) : a ∉ @nil α := fun ⟨_, (h : some a = none)⟩ => by injection h
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem mem_cons (a : α) : ∀ s : Seq α, a ∈ cons a s
  | ⟨_, _⟩ => Stream'.mem_cons (some a) _


theorem mem_cons_of_mem (y : α) {a : α} : ∀ {s : Seq α}, a ∈ s → a ∈ cons y s
  | ⟨_, _⟩ => Stream'.mem_cons_of_mem (some y)


theorem eq_or_mem_of_mem_cons {a b : α} : ∀ {s : Seq α}, a ∈ cons b s → a = b ∨ a ∈ s
                                                                        /-
                                                                          α : Type u
                                                                          a b : α
                                                                          val✝ : Stream' (Option α)
                                                                          property✝ : val✝.IsSeq
                                                                          h✝ : Membership.mem (Stream'.Seq.cons b ⟨val✝, property✝⟩) a
                                                                          h : Eq (Option.some a) (Option.some b)
                                                                          ⊢ Eq a b
                                                                        -/
  | ⟨_, _⟩, h => (Stream'.eq_or_mem_of_mem_cons h).imp_left fun h => by injection h
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem mem_cons_iff {a b : α} {s : Seq α} : a ∈ cons b s ↔ a = b ∨ a ∈ s :=
                             /-
                               α : Type u
                               a b : α
                               s : Stream'.Seq α
                               ⊢ Or (Eq a b) (Membership.mem s a) → Membership.mem (Stream'.Seq.cons b s) a
                             -/
  ⟨eq_or_mem_of_mem_cons, by rintro (rfl | m) <;> [apply mem_cons; exact mem_cons_of_mem _ m]⟩
                             /-
                               🎉 no goals
                             -/


/-- Destructor for a sequence, resulting in either `none` (for `nil`) or
  `some (a, s)` (for `cons a s`). -/
def destruct (s : Seq α) : Option (Seq1 α) :=
  (fun a' => (a', s.tail)) <$> get? s 0


theorem destruct_eq_nil {s : Seq α} : destruct s = none → s = nil := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq s.destruct Option.none → Eq s Stream'.Seq.nil
  -/
  dsimp [destruct]
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := s.tail }) (s.get? 0)) Option.n …
  -/
  induction' f0 : get? s 0 <;> intro h
    /-
      case none
      α : Type u
      s : Stream'.Seq α
      f0 : Eq (s.get? 0) Option.none
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) Option.none) Optio …
      ⊢ Eq s Stream'.Seq.nil
    -/
  · apply Subtype.eq
    /-
      case none.a
      α : Type u
      s : Stream'.Seq α
      f0 : Eq (s.get? 0) Option.none
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) Option.none) Optio …
      ⊢ Eq ↑s ↑Stream'.Seq.nil
    -/
    funext n
    /-
      case none.a.h
      α : Type u
      s : Stream'.Seq α
      f0 : Eq (s.get? 0) Option.none
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) Option.none) Optio …
      n : Nat
      ⊢ Eq (↑s n) (↑Stream'.Seq.nil n)
    -/
    induction' n with n IH
    /-
      case none.a.h.zero
      α : Type u
      s : Stream'.Seq α
      f0 : Eq (s.get? 0) Option.none
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) Option.none) Optio …
      ⊢ Eq (↑s 0) (↑Stream'.Seq.nil 0)
    -/
    exacts [f0, s.2 IH]
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u
      s : Stream'.Seq α
      val✝ : α
      f0 : Eq (s.get? 0) (Option.some val✝)
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) (Option.some val✝) …
      ⊢ Eq s Stream'.Seq.nil
    -/
  · contradiction
    /-
      🎉 no goals
    -/


theorem destruct_eq_cons {s : Seq α} {a s'} : destruct s = some (a, s') → s = cons a s' := by
  /-
    α : Type u
    s : Stream'.Seq α
    a : α
    s' : Stream'.Seq α
    ⊢ Eq s.destruct (Option.some { fst := a, snd := s' }) → Eq s (Stream'.Seq.cons …
  -/
  dsimp [destruct]
  /-
    α : Type u
    s : Stream'.Seq α
    a : α
    s' : Stream'.Seq α
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := s.tail }) (s.get? 0)) (Option. …
  -/
  induction' f0 : get? s 0 with a' <;> intro h
    /-
      case none
      α : Type u
      s : Stream'.Seq α
      a : α
      s' : Stream'.Seq α
      f0 : Eq (s.get? 0) Option.none
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) Option.none) (Opti …
      ⊢ Eq s (Stream'.Seq.cons a s')
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u
      s : Stream'.Seq α
      a : α
      s' : Stream'.Seq α
      a' : α
      f0 : Eq (s.get? 0) (Option.some a')
      h : Eq (Option.map (fun a' => { fst := a', snd := s.tail }) (Option.some a'))  …
      ⊢ Eq s (Stream'.Seq.cons a s')
    -/
  · cases' s with f al
    /-
      case some.mk
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a')
      h : Eq (Option.map (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨f, al⟩ })  …
      ⊢ Eq ⟨f, al⟩ (Stream'.Seq.cons a s')
    -/
    injections _ h1 h2
    /-
      case some.mk
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a')
      h1 : Eq a' a
      h2 : Eq (Stream'.Seq.tail ⟨f, al⟩) s'
      ⊢ Eq ⟨f, al⟩ (Stream'.Seq.cons a s')
    -/
    rw [← h2]
    /-
      case some.mk
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a')
      h1 : Eq a' a
      h2 : Eq (Stream'.Seq.tail ⟨f, al⟩) s'
      ⊢ Eq ⟨f, al⟩ (Stream'.Seq.cons a (Stream'.Seq.tail ⟨f, al⟩))
    -/
    apply Subtype.eq
    /-
      case some.mk.a
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a')
      h1 : Eq a' a
      h2 : Eq (Stream'.Seq.tail ⟨f, al⟩) s'
      ⊢ Eq ↑⟨f, al⟩ ↑(Stream'.Seq.cons a (Stream'.Seq.tail ⟨f, al⟩))
    -/
    dsimp [tail, cons]
    /-
      case some.mk.a
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a')
      h1 : Eq a' a
      h2 : Eq (Stream'.Seq.tail ⟨f, al⟩) s'
      ⊢ Eq f (Stream'.cons (Option.some a) f.tail)
    -/
    rw [h1] at f0
    /-
      case some.mk.a
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a)
      h1 : Eq a' a
      h2 : Eq (Stream'.Seq.tail ⟨f, al⟩) s'
      ⊢ Eq f (Stream'.cons (Option.some a) f.tail)
    -/
    rw [← f0]
    /-
      case some.mk.a
      α : Type u
      a : α
      s' : Stream'.Seq α
      a' : α
      f : Stream' (Option α)
      al : f.IsSeq
      f0 : Eq (Stream'.Seq.get? ⟨f, al⟩ 0) (Option.some a)
      h1 : Eq a' a
      h2 : Eq (Stream'.Seq.tail ⟨f, al⟩) s'
      ⊢ Eq f (Stream'.cons (Stream'.Seq.get? ⟨f, al⟩ 0) f.tail)
    -/
    exact (Stream'.eta f).symm
    /-
      🎉 no goals
    -/


@[simp]
theorem destruct_nil : destruct (nil : Seq α) = none :=
  rfl


@[simp]
theorem destruct_cons (a : α) : ∀ s, destruct (cons a s) = some (a, s)
  | ⟨f, al⟩ => by
    /-
      α : Type u
      a : α
      f : Stream' (Option α)
      al : f.IsSeq
      ⊢ Eq (Stream'.Seq.cons a ⟨f, al⟩).destruct (Option.some { fst := a, snd := ⟨f, …
    -/
    unfold cons destruct Functor.map
    /-
      α : Type u
      a : α
      f : Stream' (Option α)
      al : f.IsSeq
      ⊢ Eq (instFunctorOption.1 (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨Str …
    -/
    apply congr_arg fun s => some (a, s)
    /-
      α : Type u
      a : α
      f : Stream' (Option α)
      al : f.IsSeq
      ⊢ Eq (Stream'.Seq.tail ⟨Stream'.cons (Option.some a) ↑⟨f, al⟩, ⋯⟩) ⟨f, al⟩
    -/
    apply Subtype.eq; dsimp [tail]
                      /-
                        🎉 no goals
                      -/

-- Porting note: needed universe annotation to avoid universe issues

theorem head_eq_destruct (s : Seq α) : head.{u} s = Prod.fst.{u} <$> destruct.{u} s := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq s.head (Functor.map Prod.fst s.destruct)
  -/
                                           /-
                                             🎉 no goals
                                           -/
  unfold destruct head; cases get? s 0 <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem head_nil : head (nil : Seq α) = none :=
  rfl


@[simp]
theorem head_cons (a : α) (s) : head (cons a s) = some a := by
  /-
    α : Type u
    a : α
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq.cons a s).head (Option.some a)
  -/
  rw [head_eq_destruct, destruct_cons, Option.map_eq_map, Option.map_some']
  /-
    🎉 no goals
  -/


@[simp]
theorem tail_nil : tail (nil : Seq α) = nil :=
  rfl


@[simp]
theorem tail_cons (a : α) (s) : tail (cons a s) = s := by
  /-
    α : Type u
    a : α
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq.cons a s).tail s
  -/
  cases' s with f al
  /-
    case mk
    α : Type u
    a : α
    f : Stream' (Option α)
    al : f.IsSeq
    ⊢ Eq (Stream'.Seq.cons a ⟨f, al⟩).tail ⟨f, al⟩
  -/
  apply Subtype.eq
  /-
    case mk.a
    α : Type u
    a : α
    f : Stream' (Option α)
    al : f.IsSeq
    ⊢ Eq ↑(Stream'.Seq.cons a ⟨f, al⟩).tail ↑⟨f, al⟩
  -/
  dsimp [tail, cons]
  /-
    🎉 no goals
  -/


@[simp]
theorem get?_tail (s : Seq α) (n) : get? (tail s) n = get? s (n + 1) :=
  rfl


/-- Recursion principle for sequences, compare with `List.recOn`. -/
def recOn {C : Seq α → Sort v} (s : Seq α) (h1 : C nil) (h2 : ∀ x s, C (cons x s)) :
    C s := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    C : Stream'.Seq α → Sort v
    s : Stream'.Seq α
    h1 : C Stream'.Seq.nil
    h2 : (x : α) → (s : Stream'.Seq α) → C (Stream'.Seq.cons x s)
    ⊢ C s
  -/
  cases' H : destruct s with v
    /-
      case none
      α : Type u
      β : Type v
      γ : Type w
      C : Stream'.Seq α → Sort v
      s : Stream'.Seq α
      h1 : C Stream'.Seq.nil
      h2 : (x : α) → (s : Stream'.Seq α) → C (Stream'.Seq.cons x s)
      H : Eq s.destruct Option.none
      ⊢ C s
    -/
  · rw [destruct_eq_nil H]
    /-
      case none
      α : Type u
      β : Type v
      γ : Type w
      C : Stream'.Seq α → Sort v
      s : Stream'.Seq α
      h1 : C Stream'.Seq.nil
      h2 : (x : α) → (s : Stream'.Seq α) → C (Stream'.Seq.cons x s)
      H : Eq s.destruct Option.none
      ⊢ C Stream'.Seq.nil
    -/
    apply h1
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u
      β : Type v
      γ : Type w
      C : Stream'.Seq α → Sort v
      s : Stream'.Seq α
      h1 : C Stream'.Seq.nil
      h2 : (x : α) → (s : Stream'.Seq α) → C (Stream'.Seq.cons x s)
      v : Stream'.Seq1 α
      H : Eq s.destruct (Option.some v)
      ⊢ C s
    -/
  · cases' v with a s'
    /-
      case some.mk
      α : Type u
      β : Type v
      γ : Type w
      C : Stream'.Seq α → Sort v
      s : Stream'.Seq α
      h1 : C Stream'.Seq.nil
      h2 : (x : α) → (s : Stream'.Seq α) → C (Stream'.Seq.cons x s)
      a : α
      s' : Stream'.Seq α
      H : Eq s.destruct (Option.some { fst := a, snd := s' })
      ⊢ C s
    -/
    rw [destruct_eq_cons H]
    /-
      case some.mk
      α : Type u
      β : Type v
      γ : Type w
      C : Stream'.Seq α → Sort v
      s : Stream'.Seq α
      h1 : C Stream'.Seq.nil
      h2 : (x : α) → (s : Stream'.Seq α) → C (Stream'.Seq.cons x s)
      a : α
      s' : Stream'.Seq α
      H : Eq s.destruct (Option.some { fst := a, snd := s' })
      ⊢ C (Stream'.Seq.cons a s')
    -/
    apply h2
    /-
      🎉 no goals
    -/


theorem mem_rec_on {C : Seq α → Prop} {a s} (M : a ∈ s)
    (h1 : ∀ b s', a = b ∨ C s' → C (cons b s')) : C s := by
  /-
    α : Type u
    C : Stream'.Seq α → Prop
    a : α
    s : Stream'.Seq α
    M : Membership.mem s a
    h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
    ⊢ C s
  -/
  cases' M with k e; unfold Stream'.get at e
  /-
    case intro
    α : Type u
    C : Stream'.Seq α → Prop
    a : α
    s : Stream'.Seq α
    h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
    k : Nat
    e : (fun b => Eq (Option.some a) b) (↑s k)
    ⊢ C s
  -/
  induction' k with k IH generalizing s
  · have TH : s = cons a (tail s) := by
      apply destruct_eq_cons
      unfold destruct get? Functor.map
      rw [← e]
      rfl
    /-
      case intro.zero
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      s : Stream'.Seq α
      e : Eq (Option.some a) (↑s 0)
      TH : Eq s (Stream'.Seq.cons a s.tail)
      ⊢ C s
    -/
    rw [TH]
    /-
      case intro.zero
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      s : Stream'.Seq α
      e : Eq (Option.some a) (↑s 0)
      TH : Eq s (Stream'.Seq.cons a s.tail)
      ⊢ C (Stream'.Seq.cons a s.tail)
    -/
    apply h1 _ _ (Or.inl rfl)
    /-
      🎉 no goals
    -/
  -- Porting note: had to reshuffle `intro`
  /-
    case intro.succ
    α : Type u
    C : Stream'.Seq α → Prop
    a : α
    h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
    k : Nat
    IH : ∀ {s : Stream'.Seq α}, Eq (Option.some a) (↑s k) → C s
    s : Stream'.Seq α
    e : Eq (Option.some a) (↑s (HAdd.hAdd k 1))
    ⊢ C s
  -/
  revert e; apply s.recOn _ fun b s' => _
    /-
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      k : Nat
      IH : ∀ {s : Stream'.Seq α}, Eq (Option.some a) (↑s k) → C s
      s : Stream'.Seq α
      ⊢ Eq (Option.some a) (↑Stream'.Seq.nil (HAdd.hAdd k 1)) → C Stream'.Seq.nil
    -/
  · intro e; injection e
             /-
               🎉 no goals
             -/
    /-
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      k : Nat
      IH : ∀ {s : Stream'.Seq α}, Eq (Option.some a) (↑s k) → C s
      s : Stream'.Seq α
      ⊢ ∀ (b : α) (s' : Stream'.Seq α), Eq (Option.some a) (↑(Stream'.Seq.cons b s') …
    -/
  · intro b s' e
    /-
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      k : Nat
      IH : ∀ {s : Stream'.Seq α}, Eq (Option.some a) (↑s k) → C s
      s : Stream'.Seq α
      b : α
      s' : Stream'.Seq α
      e : Eq (Option.some a) (↑(Stream'.Seq.cons b s') (HAdd.hAdd k 1))
      ⊢ C (Stream'.Seq.cons b s')
    -/
    have h_eq : (cons b s').val (Nat.succ k) = s'.val k := by cases s'; rfl
    /-
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      k : Nat
      IH : ∀ {s : Stream'.Seq α}, Eq (Option.some a) (↑s k) → C s
      s : Stream'.Seq α
      b : α
      s' : Stream'.Seq α
      e : Eq (Option.some a) (↑(Stream'.Seq.cons b s') (HAdd.hAdd k 1))
      h_eq : Eq (↑(Stream'.Seq.cons b s') k.succ) (↑s' k)
      ⊢ C (Stream'.Seq.cons b s')
    -/
    rw [h_eq] at e
    /-
      α : Type u
      C : Stream'.Seq α → Prop
      a : α
      h1 : ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (C s') → C (Stream'.Seq.cons  …
      k : Nat
      IH : ∀ {s : Stream'.Seq α}, Eq (Option.some a) (↑s k) → C s
      s : Stream'.Seq α
      b : α
      s' : Stream'.Seq α
      e : Eq (Option.some a) (↑s' k)
      h_eq : Eq (↑(Stream'.Seq.cons b s') k.succ) (↑s' k)
      ⊢ C (Stream'.Seq.cons b s')
    -/
    apply h1 _ _ (Or.inr (IH e))
    /-
      🎉 no goals
    -/


/-- Corecursor over pairs of `Option` values -/
def Corec.f (f : β → Option (α × β)) : Option β → Option α × Option β
  | none => (none, none)
  | some b =>
    match f b with
    | none => (none, none)
    | some (a, b') => (some a, some b')


/-- Corecursor for `Seq α` as a coinductive type. Iterates `f` to produce new elements
  of the sequence until `none` is obtained. -/
def corec (f : β → Option (α × β)) (b : β) : Seq α := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Option (Prod α β)
    b : β
    ⊢ Stream'.Seq α
  -/
  refine ⟨Stream'.corec' (Corec.f f) (some b), fun {n} h => ?_⟩
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Option (Prod α β)
    b : β
    n : Nat
    h : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) n) Option.none
    ⊢ Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) (HAdd.hAdd n 1))  …
  -/
  rw [Stream'.corec'_eq]
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Option (Prod α β)
    b : β
    n : Nat
    h : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) n) Option.none
    ⊢ Eq (Stream'.cons (Stream'.Seq.Corec.f f (Option.some b)).1 (Stream'.corec' ( …
  -/
  change Stream'.corec' (Corec.f f) (Corec.f f (some b)).2 n = none
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Option (Prod α β)
    b : β
    n : Nat
    h : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) n) Option.none
    ⊢ Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Stream'.Seq.Corec.f f (Option.so …
  -/
  revert h; generalize some b = o; revert o
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : β → Option (Prod α β)
    b : β
    n : Nat
    ⊢ ∀ (o : Option β), Eq (Stream'.corec' (Stream'.Seq.Corec.f f) o n) Option.non …
  -/
  induction' n with n IH <;> intro o
    /-
      case zero
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b : β
      o : Option β
      ⊢ Eq (Stream'.corec' (Stream'.Seq.Corec.f f) o 0) Option.none → Eq (Stream'.co …
    -/
  · change (Corec.f f o).1 = none → (Corec.f f (Corec.f f o).2).1 = none
    /-
      case zero
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b : β
      o : Option β
      ⊢ Eq (Stream'.Seq.Corec.f f o).1 Option.none → Eq (Stream'.Seq.Corec.f f (Stre …
    -/
    cases' o with b <;> intro h
      /-
        case zero.none
        α : Type u
        β : Type v
        γ : Type w
        f : β → Option (Prod α β)
        b : β
        h : Eq (Stream'.Seq.Corec.f f Option.none).1 Option.none
        ⊢ Eq (Stream'.Seq.Corec.f f (Stream'.Seq.Corec.f f Option.none).2).1 Option.none
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case zero.some
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b✝ b : β
      h : Eq (Stream'.Seq.Corec.f f (Option.some b)).1 Option.none
      ⊢ Eq (Stream'.Seq.Corec.f f (Stream'.Seq.Corec.f f (Option.some b)).2).1 Optio …
    -/
    dsimp [Corec.f] at h
    /-
      case zero.some
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b✝ b : β
      h : Eq (Stream'.Seq.omap.match_1 (fun x => Prod (Option α) (Option β)) (f b) ( …
      ⊢ Eq (Stream'.Seq.Corec.f f (Stream'.Seq.Corec.f f (Option.some b)).2).1 Optio …
    -/
    dsimp [Corec.f]
    /-
      case zero.some
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b✝ b : β
      h : Eq (Stream'.Seq.omap.match_1 (fun x => Prod (Option α) (Option β)) (f b) ( …
      ⊢ Eq (Stream'.Seq.Corec.f.match_1 (fun x => Prod (Option α) (Option β)) (Strea …
    -/
    revert h; cases' h₁ : f b with s <;> intro h
      /-
        case zero.some.none
        α : Type u
        β : Type v
        γ : Type w
        f : β → Option (Prod α β)
        b✝ b : β
        h₁ : Eq (f b) Option.none
        h : Eq (Stream'.Seq.omap.match_1 (fun x => Prod (Option α) (Option β)) Option. …
        ⊢ Eq (Stream'.Seq.Corec.f.match_1 (fun x => Prod (Option α) (Option β)) (Strea …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case zero.some.some
        α : Type u
        β : Type v
        γ : Type w
        f : β → Option (Prod α β)
        b✝ b : β
        s : Prod α β
        h₁ : Eq (f b) (Option.some s)
        h : Eq (Stream'.Seq.omap.match_1 (fun x => Prod (Option α) (Option β)) (Option …
        ⊢ Eq (Stream'.Seq.Corec.f.match_1 (fun x => Prod (Option α) (Option β)) (Strea …
      -/
    · cases' s with a b'
      /-
        case zero.some.some.mk
        α : Type u
        β : Type v
        γ : Type w
        f : β → Option (Prod α β)
        b✝ b : β
        a : α
        b' : β
        h₁ : Eq (f b) (Option.some { fst := a, snd := b' })
        h : Eq (Stream'.Seq.omap.match_1 (fun x => Prod (Option α) (Option β)) (Option …
        ⊢ Eq (Stream'.Seq.Corec.f.match_1 (fun x => Prod (Option α) (Option β)) (Strea …
      -/
      contradiction
      /-
        🎉 no goals
      -/
    /-
      case succ
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b : β
      n : Nat
      IH : ∀ (o : Option β), Eq (Stream'.corec' (Stream'.Seq.Corec.f f) o n) Option. …
      o : Option β
      ⊢ Eq (Stream'.corec' (Stream'.Seq.Corec.f f) o (HAdd.hAdd n 1)) Option.none →  …
    -/
  · rw [Stream'.corec'_eq (Corec.f f) (Corec.f f o).2, Stream'.corec'_eq (Corec.f f) o]
    /-
      case succ
      α : Type u
      β : Type v
      γ : Type w
      f : β → Option (Prod α β)
      b : β
      n : Nat
      IH : ∀ (o : Option β), Eq (Stream'.corec' (Stream'.Seq.Corec.f f) o n) Option. …
      o : Option β
      ⊢ Eq (Stream'.cons (Stream'.Seq.Corec.f f o).1 (Stream'.corec' (Stream'.Seq.Co …
    -/
    exact IH (Corec.f f o).2
    /-
      🎉 no goals
    -/


@[simp]
theorem corec_eq (f : β → Option (α × β)) (b : β) :
    destruct (corec f b) = omap (corec f) (f b) := by
  /-
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    ⊢ Eq (Stream'.Seq.corec f b).destruct (Stream'.Seq.omap (Stream'.Seq.corec f)  …
  -/
  dsimp [corec, destruct, get]
  -- Porting note: next two lines were `change`...`with`...
  /-
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨Stream'.core …
  -/
  have h : Stream'.corec' (Corec.f f) (some b) 0 = (Corec.f f (some b)).1 := rfl
  /-
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Seq …
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨Stream'.core …
  -/
  rw [h]
  /-
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Seq …
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨Stream'.core …
  -/
  dsimp [Corec.f]
  /-
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Seq …
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨Stream'.core …
  -/
  induction' h : f b with s; · rfl
                               /-
                                 🎉 no goals
                               -/
  /-
    case some
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h✝ : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Se …
    s : Prod α β
    h : Eq (f b) (Option.some s)
    ⊢ Eq (Option.map (fun a' => { fst := a', snd := Stream'.Seq.tail ⟨Stream'.core …
  -/
  cases' s with a b'; dsimp [Corec.f]
  /-
    case some.mk
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h✝ : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Se …
    a : α
    b' : β
    h : Eq (f b) (Option.some { fst := a, snd := b' })
    ⊢ Eq (Option.some { fst := a, snd := Stream'.Seq.tail ⟨Stream'.corec' (Stream' …
  -/
  apply congr_arg fun b' => some (a, b')
  /-
    case some.mk
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h✝ : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Se …
    a : α
    b' : β
    h : Eq (f b) (Option.some { fst := a, snd := b' })
    ⊢ Eq (Stream'.Seq.tail ⟨Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) …
  -/
  apply Subtype.eq
  /-
    case some.mk.a
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h✝ : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Se …
    a : α
    b' : β
    h : Eq (f b) (Option.some { fst := a, snd := b' })
    ⊢ Eq ↑(Stream'.Seq.tail ⟨Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b …
  -/
  dsimp [corec, tail]
  /-
    case some.mk.a
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h✝ : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Se …
    a : α
    b' : β
    h : Eq (f b) (Option.some { fst := a, snd := b' })
    ⊢ Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b)).tail (Stream'.co …
  -/
  rw [Stream'.corec'_eq, Stream'.tail_cons]
  /-
    case some.mk.a
    α : Type u
    β : Type v
    f : β → Option (Prod α β)
    b : β
    h✝ : Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Option.some b) 0) (Stream'.Se …
    a : α
    b' : β
    h : Eq (f b) (Option.some { fst := a, snd := b' })
    ⊢ Eq (Stream'.corec' (Stream'.Seq.Corec.f f) (Stream'.Seq.Corec.f f (Option.so …
  -/
  dsimp [Corec.f]; rw [h]
                   /-
                     🎉 no goals
                   -/


local infixl:50 " ~ " => R


/-- Bisimilarity relation over `Option` of `Seq1 α`-/
def BisimO : Option (Seq1 α) → Option (Seq1 α) → Prop
  | none, none => True
  | some (a, s), some (a', s') => a = a' ∧ R s s'
  | _, _ => False


attribute [simp] BisimO

/-- a relation is bisimilar if it meets the `BisimO` test -/
def IsBisimulation :=
  ∀ ⦃s₁ s₂⦄, s₁ ~ s₂ → BisimO R (destruct s₁) (destruct s₂)

-- If two streams are bisimilar, then they are equal

theorem eq_of_bisim (bisim : IsBisimulation R) {s₁ s₂} (r : s₁ ~ s₂) : s₁ = s₂ := by
  /-
    α : Type u
    R : Stream'.Seq α → Stream'.Seq α → Prop
    bisim : Stream'.Seq.IsBisimulation R
    s₁ s₂ : Stream'.Seq α
    r : R s₁ s₂
    ⊢ Eq s₁ s₂
  -/
  apply Subtype.eq
  /-
    case a
    α : Type u
    R : Stream'.Seq α → Stream'.Seq α → Prop
    bisim : Stream'.Seq.IsBisimulation R
    s₁ s₂ : Stream'.Seq α
    r : R s₁ s₂
    ⊢ Eq ↑s₁ ↑s₂
  -/
  apply Stream'.eq_of_bisim fun x y => ∃ s s' : Seq α, s.1 = x ∧ s'.1 = y ∧ R s s'
    /-
      case a.bisim
      α : Type u
      R : Stream'.Seq α → Stream'.Seq α → Prop
      bisim : Stream'.Seq.IsBisimulation R
      s₁ s₂ : Stream'.Seq α
      r : R s₁ s₂
      ⊢ Stream'.IsBisimulation fun x y => Exists fun s => Exists fun s' => And (Eq ( …
    -/
  · dsimp [Stream'.IsBisimulation]
    /-
      case a.bisim
      α : Type u
      R : Stream'.Seq α → Stream'.Seq α → Prop
      bisim : Stream'.Seq.IsBisimulation R
      s₁ s₂ : Stream'.Seq α
      r : R s₁ s₂
      ⊢ ∀ ⦃s₁ s₂ : Stream' (Option α)⦄, (Exists fun s => Exists fun s' => And (Eq (↑ …
    -/
    intro t₁ t₂ e
    exact
    match t₁, t₂, e with
    | _, _, ⟨s, s', rfl, rfl, r⟩ => by
      suffices head s = head s' ∧ R (tail s) (tail s') from
        And.imp id (fun r => ⟨tail s, tail s', by cases s; rfl, by cases s'; rfl, r⟩) this
      have := bisim r; revert r this
      apply recOn s _ _ <;> apply recOn s' _ _
      · intro r _
        constructor
        · rfl
        · assumption
      · intro x s _ this
        rw [destruct_nil, destruct_cons] at this
        exact False.elim this
      · intro x s _ this
        rw [destruct_nil, destruct_cons] at this
        exact False.elim this
      · intro x s x' s' _ this
        rw [destruct_cons, destruct_cons] at this
        rw [head_cons, head_cons, tail_cons, tail_cons]
        cases' this with h1 h2
        constructor
        · rw [h1]
        · exact h2
    /-
      case a.a
      α : Type u
      R : Stream'.Seq α → Stream'.Seq α → Prop
      bisim : Stream'.Seq.IsBisimulation R
      s₁ s₂ : Stream'.Seq α
      r : R s₁ s₂
      ⊢ Exists fun s => Exists fun s' => And (Eq ↑s ↑s₁) (And (Eq ↑s' ↑s₂) (R s s'))
    -/
  · exact ⟨s₁, s₂, rfl, rfl, r⟩
    /-
      🎉 no goals
    -/


theorem coinduction :
    ∀ {s₁ s₂ : Seq α},
      head s₁ = head s₂ →
        (∀ (β : Type u) (fr : Seq α → β), fr s₁ = fr s₂ → fr (tail s₁) = fr (tail s₂)) → s₁ = s₂
  | _, _, hh, ht =>
    Subtype.eq (Stream'.coinduction hh fun β fr => ht β fun s => fr s.1)


theorem coinduction2 (s) (f g : Seq α → Seq β)
    (H :
      ∀ s,
        BisimO (fun s1 s2 : Seq β => ∃ s : Seq α, s1 = f s ∧ s2 = g s) (destruct (f s))
          (destruct (g s))) :
    f s = g s := by
  /-
    α : Type u
    β : Type v
    s : Stream'.Seq α
    f g : Stream'.Seq α → Stream'.Seq β
    H : ∀ (s : Stream'.Seq α), Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => An …
    ⊢ Eq (f s) (g s)
  -/
  refine eq_of_bisim (fun s1 s2 => ∃ s, s1 = f s ∧ s2 = g s) ?_ ⟨s, rfl, rfl⟩
  /-
    α : Type u
    β : Type v
    s : Stream'.Seq α
    f g : Stream'.Seq α → Stream'.Seq β
    H : ∀ (s : Stream'.Seq α), Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => An …
    ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => And (Eq s1 (f s)) (E …
  -/
  intro s1 s2 h; rcases h with ⟨s, h1, h2⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    s✝ : Stream'.Seq α
    f g : Stream'.Seq α → Stream'.Seq β
    H : ∀ (s : Stream'.Seq α), Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => An …
    s1 s2 : Stream'.Seq β
    s : Stream'.Seq α
    h1 : Eq s1 (f s)
    h2 : Eq s2 (g s)
    ⊢ Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => And (Eq s1 (f s)) (Eq s2 (g …
  -/
  rw [h1, h2]; apply H
               /-
                 🎉 no goals
               -/


/-- Embed a list as a sequence -/
@[coe]
def ofList (l : List α) : Seq α :=
  ⟨List.get? l, fun {n} h => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      l : List α
      n : Nat
      h : Eq (l.get? n) Option.none
      ⊢ Eq (l.get? (HAdd.hAdd n 1)) Option.none
    -/
    rw [List.get?_eq_none_iff] at h ⊢
    /-
      α : Type u
      β : Type v
      γ : Type w
      l : List α
      n : Nat
      h : LE.le l.length n
      ⊢ LE.le l.length (HAdd.hAdd n 1)
    -/
    exact h.trans (Nat.le_succ n)⟩
    /-
      🎉 no goals
    -/


instance coeList : Coe (List α) (Seq α) :=
  ⟨ofList⟩


@[simp]
theorem ofList_nil : ofList [] = (nil : Seq α) :=
  rfl


@[simp]
theorem ofList_get (l : List α) (n : ℕ) : (ofList l).get? n = l.get? n :=
  rfl


@[simp]
theorem ofList_cons (a : α) (l : List α) : ofList (a::l) = cons a (ofList l) := by
  /-
    α : Type u
    a : α
    l : List α
    ⊢ Eq (↑(List.cons a l)) (Stream'.Seq.cons a ↑l)
  -/
                   /-
                     🎉 no goals
                   -/
  ext1 (_ | n) <;> rfl
                   /-
                     🎉 no goals
                   -/


theorem ofList_injective : Function.Injective (ofList : List α → _) :=
  fun _ _ h => List.ext_get? fun _ => congr_fun (Subtype.ext_iff.1 h) _


/-- Embed an infinite stream as a sequence -/
@[coe]
def ofStream (s : Stream' α) : Seq α :=
                               /-
                                 α : Type u
                                 β : Type v
                                 γ : Type w
                                 s : Stream' α
                                 n : Nat
                                 h : Eq (Stream'.map Option.some s n) Option.none
                                 ⊢ Eq (Stream'.map Option.some s (HAdd.hAdd n 1)) Option.none
                               -/
  ⟨s.map some, fun {n} h => by contradiction⟩
                               /-
                                 🎉 no goals
                               -/


instance coeStream : Coe (Stream' α) (Seq α) :=
  ⟨ofStream⟩


/-- Embed a `MLList α` as a sequence. Note that even though this
  is non-meta, it will produce infinite sequences if used with
  cyclic `MLList`s created by meta constructions. -/
def ofMLList : MLList Id α → Seq α :=
  corec fun l =>
    match l.uncons with
    | .none => none
    | .some (a, l') => some (a, l')


@[deprecated (since := "2024-07-26")] alias ofLazyList := ofMLList


instance coeMLList : Coe (MLList Id α) (Seq α) :=
  ⟨ofMLList⟩


@[deprecated (since := "2024-07-26")] alias coeLazyList := coeMLList


/-- Translate a sequence into a `MLList`. -/
unsafe def toMLList : Seq α → MLList Id α
  | s =>
    match destruct s with
    | none => .nil
    | some (a, s') => .cons a (toMLList s')


@[deprecated (since := "2024-07-26")] alias toLazyList := toMLList


/-- Translate a sequence to a list. This function will run forever if
  run on an infinite sequence. -/
unsafe def forceToList (s : Seq α) : List α :=
  (toMLList s).force


/-- The sequence of natural numbers some 0, some 1, ... -/
def nats : Seq ℕ :=
  Stream'.nats


@[simp]
theorem nats_get? (n : ℕ) : nats.get? n = some n :=
  rfl


/-- Append two sequences. If `s₁` is infinite, then `s₁ ++ s₂ = s₁`,
  otherwise it puts `s₂` at the location of the `nil` in `s₁`. -/
def append (s₁ s₂ : Seq α) : Seq α :=
  @corec α (Seq α × Seq α)
    (fun ⟨s₁, s₂⟩ =>
      match destruct s₁ with
      | none => omap (fun s₂ => (nil, s₂)) (destruct s₂)
      | some (a, s₁') => some (a, s₁', s₂))
    (s₁, s₂)


/-- Map a function over a sequence. -/
def map (f : α → β) : Seq α → Seq β
  | ⟨s, al⟩ =>
    ⟨s.map (Option.map f), fun {n} => by
      /-
        α : Type u
        β : Type v
        γ : Type w
        f : α → β
        s : Stream' (Option α)
        al : s.IsSeq
        n : Nat
        ⊢ Eq (Stream'.map (Option.map f) s n) Option.none → Eq (Stream'.map (Option.ma …
      -/
      dsimp [Stream'.map, Stream'.get]
      /-
        α : Type u
        β : Type v
        γ : Type w
        f : α → β
        s : Stream' (Option α)
        al : s.IsSeq
        n : Nat
        ⊢ Eq (Option.map f (s n)) Option.none → Eq (Option.map f (s (HAdd.hAdd n 1)))  …
      -/
      induction' e : s n with e <;> intro
        /-
          case none
          α : Type u
          β : Type v
          γ : Type w
          f : α → β
          s : Stream' (Option α)
          al : s.IsSeq
          n : Nat
          e : Eq (s n) Option.none
          a✝ : Eq (Option.map f Option.none) Option.none
          ⊢ Eq (Option.map f (s (HAdd.hAdd n 1))) Option.none
        -/
      · rw [al e]
        /-
          case none
          α : Type u
          β : Type v
          γ : Type w
          f : α → β
          s : Stream' (Option α)
          al : s.IsSeq
          n : Nat
          e : Eq (s n) Option.none
          a✝ : Eq (Option.map f Option.none) Option.none
          ⊢ Eq (Option.map f Option.none) Option.none
        -/
        assumption
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
          al : s.IsSeq
          n : Nat
          e✝ : α
          e : Eq (s n) (Option.some e✝)
          a✝ : Eq (Option.map f (Option.some e✝)) Option.none
          ⊢ Eq (Option.map f (s (HAdd.hAdd n 1))) Option.none
        -/
      · contradiction⟩
        /-
          🎉 no goals
        -/


/-- Flatten a sequence of sequences. (It is required that the
  sequences be nonempty to ensure productivity; in the case
  of an infinite sequence of `nil`, the first element is never
  generated.) -/
def join : Seq (Seq1 α) → Seq α :=
  corec fun S =>
    match destruct S with
    | none => none
    | some ((a, s), S') =>
      some
        (a,
          match destruct s with
          | none => S'
          | some s' => cons s' S')


/-- Remove the first `n` elements from the sequence. -/
def drop (s : Seq α) : ℕ → Seq α
  | 0 => s
  | n + 1 => tail (drop s n)


attribute [simp] drop


/-- Take the first `n` elements of the sequence (producing a list) -/
def take : ℕ → Seq α → List α
  | 0, _ => []
  | n + 1, s =>
    match destruct s with
    | none => []
    | some (x, r) => List.cons x (take n r)


/-- Split a sequence at `n`, producing a finite initial segment
  and an infinite tail. -/
def splitAt : ℕ → Seq α → List α × Seq α
  | 0, s => ([], s)
  | n + 1, s =>
    match destruct s with
    | none => ([], nil)
    | some (x, s') =>
      let (l, r) := splitAt n s'
      (List.cons x l, r)


/-- Combine two sequences with a function -/
def zipWith (f : α → β → γ) (s₁ : Seq α) (s₂ : Seq β) : Seq γ :=
  ⟨fun n => Option.map₂ f (s₁.get? n) (s₂.get? n), fun {_} hn =>
    Option.map₂_eq_none_iff.2 <| (Option.map₂_eq_none_iff.1 hn).imp s₁.2 s₂.2⟩


@[simp]
theorem get?_zipWith (f : α → β → γ) (s s' n) :
    (zipWith f s s').get? n = Option.map₂ f (s.get? n) (s'.get? n) :=
  rfl


/-- Pair two sequences into a sequence of pairs -/
def zip : Seq α → Seq β → Seq (α × β) :=
  zipWith Prod.mk


theorem get?_zip (s : Seq α) (t : Seq β) (n : ℕ) :
    get? (zip s t) n = Option.map₂ Prod.mk (get? s n) (get? t n) :=
  get?_zipWith _ _ _ _


/-- Separate a sequence of pairs into two sequences -/
def unzip (s : Seq (α × β)) : Seq α × Seq β :=
  (map Prod.fst s, map Prod.snd s)


/-- Enumerate a sequence by tagging each element with its index. -/
def enum (s : Seq α) : Seq (ℕ × α) :=
  Seq.zip nats s


@[simp]
theorem get?_enum (s : Seq α) (n : ℕ) : get? (enum s) n = Option.map (Prod.mk n) (get? s n) :=
  get?_zip _ _ _


@[simp]
theorem enum_nil : enum (nil : Seq α) = nil :=
  rfl


/-- The length of a terminating sequence. -/
def length (s : Seq α) (h : s.Terminates) : ℕ :=
  Nat.find h


/-- Convert a sequence which is known to terminate into a list -/
def toList (s : Seq α) (h : s.Terminates) : List α :=
  take (length s h) s


/-- Convert a sequence which is known not to terminate into a stream -/
def toStream (s : Seq α) (h : ¬s.Terminates) : Stream' α := fun n =>
  Option.get _ <| not_terminates_iff.1 h n


/-- Convert a sequence into either a list or a stream depending on whether
  it is finite or infinite. (Without decidability of the infiniteness predicate,
  this is not constructively possible.) -/
def toListOrStream (s : Seq α) [Decidable s.Terminates] : List α ⊕ Stream' α :=
  if h : s.Terminates then Sum.inl (toList s h) else Sum.inr (toStream s h)


@[simp]
theorem nil_append (s : Seq α) : append nil s = s := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq.nil.append s) s
  -/
  apply coinduction2; intro s
  /-
    case H
    α : Type u
    s✝ s : Stream'.Seq α
    ⊢ Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => And (Eq s1 (Stream'.Seq.nil …
  -/
  dsimp [append]; rw [corec_eq]
  /-
    case H
    α : Type u
    s✝ s : Stream'.Seq α
    ⊢ Stream'.Seq.BisimO.match_1 (fun x x => Prop) (Stream'.Seq.omap (Stream'.Seq. …
  -/
  dsimp [append]; apply recOn s _ _
    /-
      α : Type u
      s✝ s : Stream'.Seq α
      ⊢ Stream'.Seq.BisimO.match_1 (fun x x => Prop) (Stream'.Seq.omap.match_1 (fun  …
    -/
  · trivial
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      s✝ s : Stream'.Seq α
      ⊢ ∀ (x : α) (s : Stream'.Seq α), Stream'.Seq.BisimO.match_1 (fun x x => Prop)  …
    -/
  · intro x s
    /-
      α : Type u
      s✝¹ s✝ : Stream'.Seq α
      x : α
      s : Stream'.Seq α
      ⊢ Stream'.Seq.BisimO.match_1 (fun x x => Prop) (Stream'.Seq.omap.match_1 (fun  …
    -/
    rw [destruct_cons]
    /-
      α : Type u
      s✝¹ s✝ : Stream'.Seq α
      x : α
      s : Stream'.Seq α
      ⊢ Stream'.Seq.BisimO.match_1 (fun x x => Prop) (Stream'.Seq.omap.match_1 (fun  …
    -/
    dsimp
    /-
      α : Type u
      s✝¹ s✝ : Stream'.Seq α
      x : α
      s : Stream'.Seq α
      ⊢ And (Eq x x) (Exists fun s_1 => And (Eq (Stream'.Seq.corec (fun x => Stream' …
    -/
    exact ⟨rfl, s, rfl, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem getElem?_take : ∀ (n k : ℕ) (s : Seq α),
    (s.take k)[n]? = if n < k then s.get? n else none
                  /-
                    α : Type u
                    n : Nat
                    s : Stream'.Seq α
                    ⊢ Eq (GetElem?.getElem? (Stream'.Seq.take 0 s) n) (ite (LT.lt n 0) (s.get? n)  …
                  -/
  | n, 0, s => by simp [take]
                  /-
                    🎉 no goals
                  -/
  | n, k+1, s => by
    /-
      α : Type u
      n k : Nat
      s : Stream'.Seq α
      ⊢ Eq (GetElem?.getElem? (Stream'.Seq.take (HAdd.hAdd k 1) s) n) (ite (LT.lt n  …
    -/
    rw [take]
    cases h : destruct s with
    | none =>
      simp [destruct_eq_nil h]
    | some a =>
      match a with
      | (x, r) =>
        rw [destruct_eq_cons h]
        match n with
        | 0 => simp
        | n+1 => simp [List.get?_cons_succ, Nat.add_lt_add_iff_right, get?_cons_succ, getElem?_take]


theorem terminatedAt_ofList (l : List α) :
    (ofList l).TerminatedAt l.length := by
  /-
    α : Type u
    l : List α
    ⊢ (↑l).TerminatedAt l.length
  -/
  simp [ofList, TerminatedAt]
  /-
    🎉 no goals
  -/


theorem terminates_ofList (l : List α) : (ofList l).Terminates :=
  ⟨_, terminatedAt_ofList l⟩


theorem terminatedAt_nil : TerminatedAt (nil : Seq α) 0 := rfl


@[simp]
theorem terminates_nil : Terminates (nil : Seq α) := ⟨0, rfl⟩


@[simp]
theorem length_nil : length (nil : Seq α) terminates_nil = 0 := rfl


@[simp]
theorem get?_zero_eq_none {s : Seq α} : s.get? 0 = none ↔ s = nil := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Iff (Eq (s.get? 0) Option.none) (Eq s Stream'.Seq.nil)
  -/
  refine ⟨fun h => ?_, fun h => h ▸ rfl⟩
  /-
    α : Type u
    s : Stream'.Seq α
    h : Eq (s.get? 0) Option.none
    ⊢ Eq s Stream'.Seq.nil
  -/
  ext1 n
  /-
    case h
    α : Type u
    s : Stream'.Seq α
    h : Eq (s.get? 0) Option.none
    n : Nat
    ⊢ Eq (s.get? n) (Stream'.Seq.nil.get? n)
  -/
  exact le_stable s (Nat.zero_le _) h
  /-
    🎉 no goals
  -/


@[simp] theorem length_eq_zero {s : Seq α} {h : s.Terminates} :
    s.length h = 0 ↔ s = nil := by
  /-
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    ⊢ Iff (Eq (s.length h) 0) (Eq s Stream'.Seq.nil)
  -/
  simp [length, TerminatedAt]
  /-
    🎉 no goals
  -/


theorem terminatedAt_zero_iff {s : Seq α} : s.TerminatedAt 0 ↔ s = nil := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Iff (s.TerminatedAt 0) (Eq s Stream'.Seq.nil)
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u
      s : Stream'.Seq α
      ⊢ s.TerminatedAt 0 → Eq s Stream'.Seq.nil
    -/
  · intro h
    /-
      case refine_1
      α : Type u
      s : Stream'.Seq α
      h : s.TerminatedAt 0
      ⊢ Eq s Stream'.Seq.nil
    -/
    ext n
    /-
      case refine_1.h.a
      α : Type u
      s : Stream'.Seq α
      h : s.TerminatedAt 0
      n : Nat
      a✝ : α
      ⊢ Iff (Membership.mem (s.get? n) a✝) (Membership.mem (Stream'.Seq.nil.get? n)  …
    -/
    rw [le_stable _ (Nat.zero_le _) h]
    /-
      case refine_1.h.a
      α : Type u
      s : Stream'.Seq α
      h : s.TerminatedAt 0
      n : Nat
      a✝ : α
      ⊢ Iff (Membership.mem Option.none a✝) (Membership.mem (Stream'.Seq.nil.get? n) …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      s : Stream'.Seq α
      ⊢ Eq s Stream'.Seq.nil → s.TerminatedAt 0
    -/
  · rintro rfl
    /-
      case refine_2
      α : Type u
      ⊢ Stream'.Seq.nil.TerminatedAt 0
    -/
    simp [TerminatedAt]
    /-
      🎉 no goals
    -/


/-- The statement of `length_le_iff'` does not assume that the sequence terminates. For a
simpler statement of the theorem where the sequence is known to terminate see `length_le_iff` -/
theorem length_le_iff' {s : Seq α} {n : ℕ} :
    (∃ h, s.length h ≤ n) ↔ s.TerminatedAt n := by
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Iff (Exists fun h => LE.le (s.length h) n) (s.TerminatedAt n)
  -/
  simp only [length, Nat.find_le_iff, TerminatedAt, Terminates, exists_prop]
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Iff (And (Exists fun n => Eq (s.get? n) Option.none) (Exists fun m => And (L …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u
      s : Stream'.Seq α
      n : Nat
      ⊢ And (Exists fun n => Eq (s.get? n) Option.none) (Exists fun m => And (LE.le  …
    -/
  · rintro ⟨_, k, hkn, hk⟩
    /-
      case refine_1.intro.intro.intro
      α : Type u
      s : Stream'.Seq α
      n : Nat
      left✝ : Exists fun n => Eq (s.get? n) Option.none
      k : Nat
      hkn : LE.le k n
      hk : Eq (s.get? k) Option.none
      ⊢ Eq (s.get? n) Option.none
    -/
    exact le_stable s hkn hk
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      s : Stream'.Seq α
      n : Nat
      ⊢ Eq (s.get? n) Option.none → And (Exists fun n => Eq (s.get? n) Option.none)  …
    -/
  · intro hn
    /-
      case refine_2
      α : Type u
      s : Stream'.Seq α
      n : Nat
      hn : Eq (s.get? n) Option.none
      ⊢ And (Exists fun n => Eq (s.get? n) Option.none) (Exists fun m => And (LE.le  …
    -/
    exact ⟨⟨n, hn⟩, ⟨n, le_rfl, hn⟩⟩
    /-
      🎉 no goals
    -/


/-- The statement of `length_le_iff` assumes that the sequence terminates. For a
statement of the where the sequence is not known to terminate see `length_le_iff'` -/
theorem length_le_iff {s : Seq α} {n : ℕ} {h : s.Terminates} :
    s.length h ≤ n ↔ s.TerminatedAt n := by
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    h : s.Terminates
    ⊢ Iff (LE.le (s.length h) n) (s.TerminatedAt n)
  -/
  rw [← length_le_iff']; simp [h]
                         /-
                           🎉 no goals
                         -/


/-- The statement of `lt_length_iff'` does not assume that the sequence terminates. For a
simpler statement of the theorem where the sequence is known to terminate see `lt_length_iff` -/
theorem lt_length_iff' {s : Seq α} {n : ℕ} :
    (∀ h : s.Terminates, n < s.length h) ↔ ∃ a, a ∈ s.get? n := by
  simp only [Terminates, TerminatedAt, length, Nat.lt_find_iff, forall_exists_index, Option.mem_def,
    ← Option.ne_none_iff_exists', ne_eq]
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Iff (∀ (x : Nat), Eq (s.get? x) Option.none → ∀ (m : Nat), LE.le m n → Not ( …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u
      s : Stream'.Seq α
      n : Nat
      ⊢ (∀ (x : Nat), Eq (s.get? x) Option.none → ∀ (m : Nat), LE.le m n → Not (Eq ( …
    -/
  · intro h hn
    /-
      case refine_1
      α : Type u
      s : Stream'.Seq α
      n : Nat
      h : ∀ (x : Nat), Eq (s.get? x) Option.none → ∀ (m : Nat), LE.le m n → Not (Eq  …
      hn : Eq (s.get? n) Option.none
      ⊢ False
    -/
    exact h n hn n le_rfl hn
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      s : Stream'.Seq α
      n : Nat
      ⊢ Not (Eq (s.get? n) Option.none) → ∀ (x : Nat), Eq (s.get? x) Option.none → ∀ …
    -/
  · intro hn _ _ k hkn hk
    /-
      case refine_2
      α : Type u
      s : Stream'.Seq α
      n : Nat
      hn : Not (Eq (s.get? n) Option.none)
      x✝ : Nat
      h✝ : Eq (s.get? x✝) Option.none
      k : Nat
      hkn : LE.le k n
      hk : Eq (s.get? k) Option.none
      ⊢ False
    -/
    exact hn <| le_stable s hkn hk
    /-
      🎉 no goals
    -/


/-- The statement of `length_le_iff` assumes that the sequence terminates. For a
statement of the where the sequence is not known to terminate see `length_le_iff'` -/
theorem lt_length_iff {s : Seq α} {n : ℕ} {h : s.Terminates} :
    n < s.length h ↔ ∃ a, a ∈ s.get? n := by
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    h : s.Terminates
    ⊢ Iff (LT.lt n (s.length h)) (Exists fun a => Membership.mem (s.get? n) a)
  -/
  rw [← lt_length_iff']; simp [h]
                         /-
                           🎉 no goals
                         -/


theorem length_take_of_le_length {s : Seq α} {n : ℕ}
    (hle : ∀ h : s.Terminates, n ≤ s.length h) : (s.take n).length = n := by
  induction n generalizing s with
  | zero => simp [take]
  | succ n ih =>
      rw [take, destruct]
      let ⟨a, ha⟩ := lt_length_iff'.1 (fun ht => lt_of_lt_of_le (Nat.succ_pos _) (hle ht))
      simp [Option.mem_def.1 ha]
      rw [ih]
      intro h
      simp only [length, tail, Nat.le_find_iff, TerminatedAt, get?_mk, Stream'.tail]
      intro m hmn hs
      have := lt_length_iff'.1 (fun ht => (Nat.lt_of_succ_le (hle ht)))
      rw [le_stable s (Nat.succ_le_of_lt hmn) hs] at this
      simp at this


@[simp]
theorem length_toList (s : Seq α) (h : s.Terminates) : (toList s h).length = length s h := by
  /-
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    ⊢ Eq (s.toList h).length (s.length h)
  -/
  rw [toList, length_take_of_le_length]
  /-
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    ⊢ ∀ (h_1 : s.Terminates), LE.le (s.length h) (s.length h_1)
  -/
  intro _
  /-
    α : Type u
    s : Stream'.Seq α
    h h✝ : s.Terminates
    ⊢ LE.le (s.length h) (s.length h✝)
  -/
  exact le_rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem?_toList (s : Seq α) (h : s.Terminates) (n : ℕ) : (toList s h)[n]? = s.get? n := by
  /-
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    n : Nat
    ⊢ Eq (GetElem?.getElem? (s.toList h) n) (s.get? n)
  -/
  ext k
  simp only [ofList, toList, get?_mk, Option.mem_def, getElem?_take, Nat.lt_find_iff, length,
    Option.ite_none_right_eq_some, and_iff_right_iff_imp, TerminatedAt, List.get?_eq_getElem?]
  /-
    case a
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    n : Nat
    k : α
    ⊢ Eq (s.get? n) (Option.some k) → ∀ (m : Nat), LE.le m n → Not (Eq (s.get? m)  …
  -/
  intro h m hmn
  /-
    case a
    α : Type u
    s : Stream'.Seq α
    h✝ : s.Terminates
    n : Nat
    k : α
    h : Eq (s.get? n) (Option.some k)
    m : Nat
    hmn : LE.le m n
    ⊢ Not (Eq (s.get? m) Option.none)
  -/
  let ⟨a, ha⟩ := ge_stable s hmn h
  /-
    case a
    α : Type u
    s : Stream'.Seq α
    h✝ : s.Terminates
    n : Nat
    k : α
    h : Eq (s.get? n) (Option.some k)
    m : Nat
    hmn : LE.le m n
    a : α
    ha : Eq (s.get? m) (Option.some a)
    ⊢ Not (Eq (s.get? m) Option.none)
  -/
  simp [ha]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofList_toList (s : Seq α) (h : s.Terminates) :
    ofList (toList s h) = s := by
  /-
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    ⊢ Eq (↑(s.toList h)) s
  -/
  ext n; simp [ofList, List.get?_eq_getElem?]
         /-
           🎉 no goals
         -/


@[simp]
theorem toList_ofList (l : List α) : toList (ofList l) (terminates_ofList l) = l :=
                       /-
                         α : Type u
                         l : List α
                         ⊢ Eq ↑((↑l).toList ⋯) ↑l
                       -/
  ofList_injective (by simp)
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem toList_nil : toList (nil : Seq α) ⟨0, terminatedAt_zero_iff.2 rfl⟩ = [] := by
  /-
    α : Type u
    ⊢ Eq (Stream'.Seq.nil.toList ⋯) List.nil
  -/
  ext; simp [nil, toList, const]
       /-
         🎉 no goals
       -/


theorem getLast?_toList (s : Seq α) (h : s.Terminates) :
    (toList s h).getLast? = s.get? (s.length h - 1) := by
  /-
    α : Type u
    s : Stream'.Seq α
    h : s.Terminates
    ⊢ Eq (s.toList h).getLast? (s.get? (HSub.hSub (s.length h) 1))
  -/
  rw [List.getLast?_eq_getElem?, getElem?_toList, length_toList]
  /-
    🎉 no goals
  -/


@[simp]
theorem cons_append (a : α) (s t) : append (cons a s) t = cons a (append s t) :=
  destruct_eq_cons <| by
    /-
      α : Type u
      a : α
      s t : Stream'.Seq α
      ⊢ Eq ((Stream'.Seq.cons a s).append t).destruct (Option.some { fst := a, snd : …
    -/
    dsimp [append]; rw [corec_eq]
    /-
      α : Type u
      a : α
      s t : Stream'.Seq α
      ⊢ Eq (Stream'.Seq.omap (Stream'.Seq.corec fun x => Stream'.Seq.toMLList.match_ …
    -/
    dsimp [append]; rw [destruct_cons]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem append_nil (s : Seq α) : append s nil = s := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (s.append Stream'.Seq.nil) s
  -/
  apply coinduction2 s; intro s
  /-
    case H
    α : Type u
    s✝ s : Stream'.Seq α
    ⊢ Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => And (Eq s1 (s.append Stream …
  -/
  apply recOn s _ _
    /-
      α : Type u
      s✝ s : Stream'.Seq α
      ⊢ Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => And (Eq s1 (s.append Stream …
    -/
  · trivial
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      s✝ s : Stream'.Seq α
      ⊢ ∀ (x : α) (s : Stream'.Seq α), Stream'.Seq.BisimO (fun s1 s2 => Exists fun s …
    -/
  · intro x s
    /-
      α : Type u
      s✝¹ s✝ : Stream'.Seq α
      x : α
      s : Stream'.Seq α
      ⊢ Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => And (Eq s1 (s.append Stream …
    -/
    rw [cons_append, destruct_cons, destruct_cons]
    /-
      α : Type u
      s✝¹ s✝ : Stream'.Seq α
      x : α
      s : Stream'.Seq α
      ⊢ Stream'.Seq.BisimO (fun s1 s2 => Exists fun s => And (Eq s1 (s.append Stream …
    -/
    dsimp
    /-
      α : Type u
      s✝¹ s✝ : Stream'.Seq α
      x : α
      s : Stream'.Seq α
      ⊢ And (Eq x x) (Exists fun s_1 => And (Eq (s.append Stream'.Seq.nil) (s_1.appe …
    -/
    exact ⟨rfl, s, rfl, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem append_assoc (s t u : Seq α) : append (append s t) u = append s (append t u) := by
  /-
    α : Type u
    s t u : Stream'.Seq α
    ⊢ Eq ((s.append t).append u) (s.append (t.append u))
  -/
  apply eq_of_bisim fun s1 s2 => ∃ s t u, s1 = append (append s t) u ∧ s2 = append s (append t u)
    /-
      case bisim
      α : Type u
      s t u : Stream'.Seq α
      ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => Exists fun t => Exis …
    -/
  · intro s1 s2 h
    exact
      match s1, s2, h with
      | _, _, ⟨s, t, u, rfl, rfl⟩ => by
        apply recOn s <;> simp
        · apply recOn t <;> simp
          · apply recOn u <;> simp
            · intro _ u
              refine ⟨nil, nil, u, ?_, ?_⟩ <;> simp
          · intro _ t
            refine ⟨nil, t, u, ?_, ?_⟩ <;> simp
        · intro _ s
          exact ⟨s, t, u, rfl, rfl⟩
    /-
      case r
      α : Type u
      s t u : Stream'.Seq α
      ⊢ Exists fun s_1 => Exists fun t_1 => Exists fun u_1 => And (Eq ((s.append t). …
    -/
  · exact ⟨s, t, u, rfl, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_nil (f : α → β) : map f nil = nil :=
  rfl


@[simp]
theorem map_cons (f : α → β) (a) : ∀ s, map f (cons a s) = cons (f a) (map f s)
                  /-
                    α : Type u
                    β : Type v
                    f : α → β
                    a : α
                    s : Stream' (Option α)
                    al : s.IsSeq
                    ⊢ Eq (Stream'.Seq.map f (Stream'.Seq.cons a ⟨s, al⟩)) (Stream'.Seq.cons (f a)  …
                  -/
  | ⟨s, al⟩ => by apply Subtype.eq; dsimp [cons, map]; rw [Stream'.map_cons]; rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem map_id : ∀ s : Seq α, map id s = s
  | ⟨s, al⟩ => by
    /-
      α : Type u
      s : Stream' (Option α)
      al : s.IsSeq
      ⊢ Eq (Stream'.Seq.map id ⟨s, al⟩) ⟨s, al⟩
    -/
    apply Subtype.eq; dsimp [map]
    /-
      case a
      α : Type u
      s : Stream' (Option α)
      al : s.IsSeq
      ⊢ Eq (Stream'.map (Option.map id) s) s
    -/
    rw [Option.map_id, Stream'.map_id]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_tail (f : α → β) : ∀ s, map f (tail s) = tail (map f s)
                  /-
                    α : Type u
                    β : Type v
                    f : α → β
                    s : Stream' (Option α)
                    al : s.IsSeq
                    ⊢ Eq (Stream'.Seq.map f (Stream'.Seq.tail ⟨s, al⟩)) (Stream'.Seq.map f ⟨s, al⟩ …
                  -/
  | ⟨s, al⟩ => by apply Subtype.eq; dsimp [tail, map]
                                    /-
                                      🎉 no goals
                                    -/


theorem map_comp (f : α → β) (g : β → γ) : ∀ s : Seq α, map (g ∘ f) s = map g (map f s)
  | ⟨s, al⟩ => by
    /-
      α : Type u
      β : Type v
      γ : Type w
      f : α → β
      g : β → γ
      s : Stream' (Option α)
      al : s.IsSeq
      ⊢ Eq (Stream'.Seq.map (Function.comp g f) ⟨s, al⟩) (Stream'.Seq.map g (Stream' …
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
      al : s.IsSeq
      ⊢ Eq (Stream'.map (Option.map (Function.comp g f)) s) (Stream'.map (Function.c …
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
      al : s.IsSeq
      ⊢ Eq (Option.map (Function.comp g f)) (Function.comp (Option.map g) (Option.ma …
    -/
               /-
                 🎉 no goals
               -/
    ext ⟨⟩ <;> rfl
               /-
                 🎉 no goals
               -/


@[simp]
theorem map_append (f : α → β) (s t) : map f (append s t) = append (map f s) (map f t) := by
  apply
    eq_of_bisim (fun s1 s2 => ∃ s t, s1 = map f (append s t) ∧ s2 = append (map f s) (map f t)) _
      ⟨s, t, rfl, rfl⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    s t : Stream'.Seq α
    ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => Exists fun t => And  …
  -/
  intro s1 s2 h
  exact
    match s1, s2, h with
    | _, _, ⟨s, t, rfl, rfl⟩ => by
      apply recOn s <;> simp
      · apply recOn t <;> simp
        · intro _ t
          refine ⟨nil, t, ?_, ?_⟩ <;> simp
      · intro _ s
        exact ⟨s, t, rfl, rfl⟩


@[simp]
theorem map_get? (f : α → β) : ∀ s n, get? (map f s) n = (get? s n).map f
  | ⟨_, _⟩, _ => rfl


@[simp]
theorem terminatedAt_map_iff {f : α → β} {s : Seq α} {n : ℕ} :
    (map f s).TerminatedAt n ↔ s.TerminatedAt n := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Stream'.Seq α
    n : Nat
    ⊢ Iff ((Stream'.Seq.map f s).TerminatedAt n) (s.TerminatedAt n)
  -/
  simp [TerminatedAt]
  /-
    🎉 no goals
  -/


@[simp]
theorem terminates_map_iff {f : α → β} {s : Seq α}  :
    (map f s).Terminates ↔ s.Terminates := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Stream'.Seq α
    ⊢ Iff (Stream'.Seq.map f s).Terminates s.Terminates
  -/
  simp [Terminates]
  /-
    🎉 no goals
  -/


@[simp]
theorem length_map {s : Seq α} {f : α → β} (h : (s.map f).Terminates) :
    (s.map f).length h = s.length (terminates_map_iff.1 h) := by
  /-
    α : Type u
    β : Type v
    s : Stream'.Seq α
    f : α → β
    h : (Stream'.Seq.map f s).Terminates
    ⊢ Eq ((Stream'.Seq.map f s).length h) (s.length ⋯)
  -/
  rw [length]
  /-
    α : Type u
    β : Type v
    s : Stream'.Seq α
    f : α → β
    h : (Stream'.Seq.map f s).Terminates
    ⊢ Eq (Nat.find h) (s.length ⋯)
  -/
  congr
  /-
    case e_p
    α : Type u
    β : Type v
    s : Stream'.Seq α
    f : α → β
    h : (Stream'.Seq.map f s).Terminates
    ⊢ Eq (Stream'.Seq.map f s).TerminatedAt s.TerminatedAt
  -/
  ext
  /-
    case e_p.h.a
    α : Type u
    β : Type v
    s : Stream'.Seq α
    f : α → β
    h : (Stream'.Seq.map f s).Terminates
    x✝ : Nat
    ⊢ Iff ((Stream'.Seq.map f s).TerminatedAt x✝) (s.TerminatedAt x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Functor Seq where map := @map


instance : LawfulFunctor Seq where
  id_map := @map_id
  comp_map := @map_comp
  map_const := rfl


@[simp]
theorem join_nil : join nil = (nil : Seq α) :=
  destruct_eq_nil rfl

--@[simp] -- Porting note: simp can prove: `join_cons` is more general

theorem join_cons_nil (a : α) (S) : join (cons (a, nil) S) = cons a (join S) :=
                         /-
                           α : Type u
                           a : α
                           S : Stream'.Seq (Stream'.Seq1 α)
                           ⊢ Eq (Stream'.Seq.cons { fst := a, snd := Stream'.Seq.nil } S).join.destruct ( …
                         -/
  destruct_eq_cons <| by simp [join]
                         /-
                           🎉 no goals
                         -/

--@[simp] -- Porting note: simp can prove: `join_cons` is more general

theorem join_cons_cons (a b : α) (s S) :
    join (cons (a, cons b s) S) = cons a (join (cons (b, s) S)) :=
                         /-
                           α : Type u
                           a b : α
                           s : Stream'.Seq α
                           S : Stream'.Seq (Stream'.Seq1 α)
                           ⊢ Eq (Stream'.Seq.cons { fst := a, snd := Stream'.Seq.cons b s } S).join.destr …
                         -/
  destruct_eq_cons <| by simp [join]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem join_cons (a : α) (s S) : join (cons (a, s) S) = cons a (append s (join S)) := by
  apply
    eq_of_bisim
      (fun s1 s2 => s1 = s2 ∨ ∃ a s S, s1 = join (cons (a, s) S) ∧ s2 = cons a (append s (join S)))
      _ (Or.inr ⟨a, s, S, rfl, rfl⟩)
  /-
    α : Type u
    a : α
    s : Stream'.Seq α
    S : Stream'.Seq (Stream'.Seq1 α)
    ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Or (Eq s1 s2) (Exists fun a => Exist …
  -/
  intro s1 s2 h
  exact
    match s1, s2, h with
    | s, _, Or.inl <| Eq.refl s => by
      apply recOn s; · trivial
      · intro x s
        rw [destruct_cons]
        exact ⟨rfl, Or.inl rfl⟩
    | _, _, Or.inr ⟨a, s, S, rfl, rfl⟩ => by
      apply recOn s
      · simp [join_cons_cons, join_cons_nil]
      · intro x s
        simpa [join_cons_cons, join_cons_nil] using Or.inr ⟨x, s, S, rfl, rfl⟩


@[simp]
theorem join_append (S T : Seq (Seq1 α)) : join (append S T) = append (join S) (join T) := by
  apply
    eq_of_bisim fun s1 s2 =>
      ∃ s S T, s1 = append s (join (append S T)) ∧ s2 = append s (append (join S) (join T))
    /-
      case bisim
      α : Type u
      S T : Stream'.Seq (Stream'.Seq1 α)
      ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => Exists fun S => Exis …
    -/
  · intro s1 s2 h
    exact
      match s1, s2, h with
      | _, _, ⟨s, S, T, rfl, rfl⟩ => by
        apply recOn s <;> simp
        · apply recOn S <;> simp
          · apply recOn T
            · simp
            · intro s T
              cases' s with a s; simp only [join_cons, destruct_cons, true_and]
              refine ⟨s, nil, T, ?_, ?_⟩ <;> simp
          · intro s S
            cases' s with a s
            simpa using ⟨s, S, T, rfl, rfl⟩
        · intro _ s
          exact ⟨s, S, T, rfl, rfl⟩
    /-
      case r
      α : Type u
      S T : Stream'.Seq (Stream'.Seq1 α)
      ⊢ Exists fun s => Exists fun S_1 => Exists fun T_1 => And (Eq (S.append T).joi …
    -/
                                   /-
                                     🎉 no goals
                                   -/
  · refine ⟨nil, S, T, ?_, ?_⟩ <;> simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem ofStream_cons (a : α) (s) : ofStream (a::s) = cons a (ofStream s) := by
  /-
    α : Type u
    a : α
    s : Stream' α
    ⊢ Eq (↑(Stream'.cons a s)) (Stream'.Seq.cons a ↑s)
  -/
  apply Subtype.eq; simp only [ofStream, cons]; rw [Stream'.map_cons]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem ofList_append (l l' : List α) : ofList (l ++ l') = append (ofList l) (ofList l') := by
  /-
    α : Type u
    l l' : List α
    ⊢ Eq (↑(HAppend.hAppend l l')) ((↑l).append ↑l')
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem ofStream_append (l : List α) (s : Stream' α) :
    ofStream (l ++ₛ s) = append (ofList l) (ofStream s) := by
  /-
    α : Type u
    l : List α
    s : Stream' α
    ⊢ Eq (↑(Stream'.appendStream' l s)) ((↑l).append ↑s)
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [*, Stream'.nil_append_stream, Stream'.cons_append_stream]
                  /-
                    🎉 no goals
                  -/


/-- Convert a sequence into a list, embedded in a computation to allow for
  the possibility of infinite sequences (in which case the computation
  never returns anything). -/
def toList' {α} (s : Seq α) : Computation (List α) :=
  @Computation.corec (List α) (List α × Seq α)
    (fun ⟨l, s⟩ =>
      match destruct s with
      | none => Sum.inl l.reverse
      | some (a, s') => Sum.inr (a::l, s'))
    ([], s)


theorem dropn_add (s : Seq α) (m) : ∀ n, drop s (m + n) = drop (drop s m) n
  | 0 => rfl
  | n + 1 => congr_arg tail (dropn_add s _ n)


theorem dropn_tail (s : Seq α) (n) : drop (tail s) n = drop s (n + 1) := by
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Eq (s.tail.drop n) (s.drop (HAdd.hAdd n 1))
  -/
  rw [Nat.add_comm]; symm; apply dropn_add
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem head_dropn (s : Seq α) (n) : head (drop s n) = get? s n := by
  /-
    α : Type u
    s : Stream'.Seq α
    n : Nat
    ⊢ Eq (s.drop n).head (s.get? n)
  -/
  induction' n with n IH generalizing s; · rfl
                                           /-
                                             🎉 no goals
                                           -/
  /-
    case succ
    α : Type u
    n : Nat
    IH : ∀ (s : Stream'.Seq α), Eq (s.drop n).head (s.get? n)
    s : Stream'.Seq α
    ⊢ Eq (s.drop (HAdd.hAdd n 1)).head (s.get? (HAdd.hAdd n 1))
  -/
  rw [← get?_tail, ← dropn_tail]; apply IH
                                  /-
                                    🎉 no goals
                                  -/


theorem mem_map (f : α → β) {a : α} : ∀ {s : Seq α}, a ∈ s → f a ∈ map f s
  | ⟨_, _⟩ => Stream'.mem_map (Option.map f)


theorem exists_of_mem_map {f} {b : β} : ∀ {s : Seq α}, b ∈ map f s → ∃ a, a ∈ s ∧ f a = b :=
  fun {s} h => by match s with
  | ⟨g, al⟩ =>
    let ⟨o, om, oe⟩ := @Stream'.exists_of_mem_map _ _ (Option.map f) (some b) g h
    cases' o with a
    · injection oe
    · injection oe with h'; exact ⟨a, om, h'⟩


theorem of_mem_append {s₁ s₂ : Seq α} {a : α} (h : a ∈ append s₁ s₂) : a ∈ s₁ ∨ a ∈ s₂ := by
  /-
    α : Type u
    s₁ s₂ : Stream'.Seq α
    a : α
    h : Membership.mem (s₁.append s₂) a
    ⊢ Or (Membership.mem s₁ a) (Membership.mem s₂ a)
  -/
  have := h; revert this
  /-
    α : Type u
    s₁ s₂ : Stream'.Seq α
    a : α
    h : Membership.mem (s₁.append s₂) a
    ⊢ Membership.mem (s₁.append s₂) a → Or (Membership.mem s₁ a) (Membership.mem s …
  -/
  generalize e : append s₁ s₂ = ss; intro h; revert s₁
  /-
    α : Type u
    s₂ : Stream'.Seq α
    a : α
    ss : Stream'.Seq α
    h : Membership.mem ss a
    ⊢ ∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq (s₁.append s₂)  …
  -/
  apply mem_rec_on h _
  /-
    α : Type u
    s₂ : Stream'.Seq α
    a : α
    ss : Stream'.Seq α
    h : Membership.mem ss a
    ⊢ ∀ (b : α) (s' : Stream'.Seq α), Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Members …
  -/
  intro b s' o s₁
  /-
    α : Type u
    s₂ : Stream'.Seq α
    a : α
    ss : Stream'.Seq α
    h : Membership.mem ss a
    b : α
    s' : Stream'.Seq α
    o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
    s₁ : Stream'.Seq α
    ⊢ Membership.mem (s₁.append s₂) a → Eq (s₁.append s₂) (Stream'.Seq.cons b s')  …
  -/
  apply s₁.recOn _ fun c t₁ => _
    /-
      α : Type u
      s₂ : Stream'.Seq α
      a : α
      ss : Stream'.Seq α
      h : Membership.mem ss a
      b : α
      s' : Stream'.Seq α
      o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
      s₁ : Stream'.Seq α
      ⊢ Membership.mem (Stream'.Seq.nil.append s₂) a → Eq (Stream'.Seq.nil.append s₂ …
    -/
  · intro m _
    /-
      α : Type u
      s₂ : Stream'.Seq α
      a : α
      ss : Stream'.Seq α
      h : Membership.mem ss a
      b : α
      s' : Stream'.Seq α
      o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
      s₁ : Stream'.Seq α
      m : Membership.mem (Stream'.Seq.nil.append s₂) a
      e✝ : Eq (Stream'.Seq.nil.append s₂) (Stream'.Seq.cons b s')
      ⊢ Or (Membership.mem Stream'.Seq.nil a) (Membership.mem s₂ a)
    -/
    apply Or.inr
    /-
      case h
      α : Type u
      s₂ : Stream'.Seq α
      a : α
      ss : Stream'.Seq α
      h : Membership.mem ss a
      b : α
      s' : Stream'.Seq α
      o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
      s₁ : Stream'.Seq α
      m : Membership.mem (Stream'.Seq.nil.append s₂) a
      e✝ : Eq (Stream'.Seq.nil.append s₂) (Stream'.Seq.cons b s')
      ⊢ Membership.mem s₂ a
    -/
    simpa using m
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      s₂ : Stream'.Seq α
      a : α
      ss : Stream'.Seq α
      h : Membership.mem ss a
      b : α
      s' : Stream'.Seq α
      o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
      s₁ : Stream'.Seq α
      ⊢ ∀ (c : α) (t₁ : Stream'.Seq α), Membership.mem ((Stream'.Seq.cons c t₁).appe …
    -/
  · intro c t₁ m e
    /-
      α : Type u
      s₂ : Stream'.Seq α
      a : α
      ss : Stream'.Seq α
      h : Membership.mem ss a
      b : α
      s' : Stream'.Seq α
      o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
      s₁ : Stream'.Seq α
      c : α
      t₁ : Stream'.Seq α
      m : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
      e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
      ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
    -/
    have this := congr_arg destruct e
    /-
      α : Type u
      s₂ : Stream'.Seq α
      a : α
      ss : Stream'.Seq α
      h : Membership.mem ss a
      b : α
      s' : Stream'.Seq α
      o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
      s₁ : Stream'.Seq α
      c : α
      t₁ : Stream'.Seq α
      m : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
      e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
      this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
      ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
    -/
    cases' show a = c ∨ a ∈ append t₁ s₂ by simpa using m with e' m
      /-
        case inl
        α : Type u
        s₂ : Stream'.Seq α
        a : α
        ss : Stream'.Seq α
        h : Membership.mem ss a
        b : α
        s' : Stream'.Seq α
        o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
        s₁ : Stream'.Seq α
        c : α
        t₁ : Stream'.Seq α
        m : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
        e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
        this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
        e' : Eq a c
        ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
      -/
    · rw [e']
      /-
        case inl
        α : Type u
        s₂ : Stream'.Seq α
        a : α
        ss : Stream'.Seq α
        h : Membership.mem ss a
        b : α
        s' : Stream'.Seq α
        o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
        s₁ : Stream'.Seq α
        c : α
        t₁ : Stream'.Seq α
        m : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
        e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
        this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
        e' : Eq a c
        ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) c) (Membership.mem s₂ c)
      -/
      exact Or.inl (mem_cons _ _)
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        s₂ : Stream'.Seq α
        a : α
        ss : Stream'.Seq α
        h : Membership.mem ss a
        b : α
        s' : Stream'.Seq α
        o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
        s₁ : Stream'.Seq α
        c : α
        t₁ : Stream'.Seq α
        m✝ : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
        e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
        this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
        m : Membership.mem (t₁.append s₂) a
        ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
      -/
    · cases' show c = b ∧ append t₁ s₂ = s' by simpa with i1 i2
      /-
        case inr.intro
        α : Type u
        s₂ : Stream'.Seq α
        a : α
        ss : Stream'.Seq α
        h : Membership.mem ss a
        b : α
        s' : Stream'.Seq α
        o : Or (Eq a b) (∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq  …
        s₁ : Stream'.Seq α
        c : α
        t₁ : Stream'.Seq α
        m✝ : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
        e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
        this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
        m : Membership.mem (t₁.append s₂) a
        i1 : Eq c b
        i2 : Eq (t₁.append s₂) s'
        ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
      -/
      cases' o with e' IH
        /-
          case inr.intro.inl
          α : Type u
          s₂ : Stream'.Seq α
          a : α
          ss : Stream'.Seq α
          h : Membership.mem ss a
          b : α
          s' s₁ : Stream'.Seq α
          c : α
          t₁ : Stream'.Seq α
          m✝ : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
          e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
          this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
          m : Membership.mem (t₁.append s₂) a
          i1 : Eq c b
          i2 : Eq (t₁.append s₂) s'
          e' : Eq a b
          ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
        -/
      · simp [i1, e']
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.inr
          α : Type u
          s₂ : Stream'.Seq α
          a : α
          ss : Stream'.Seq α
          h : Membership.mem ss a
          b : α
          s' s₁ : Stream'.Seq α
          c : α
          t₁ : Stream'.Seq α
          m✝ : Membership.mem ((Stream'.Seq.cons c t₁).append s₂) a
          e : Eq ((Stream'.Seq.cons c t₁).append s₂) (Stream'.Seq.cons b s')
          this : Eq ((Stream'.Seq.cons c t₁).append s₂).destruct (Stream'.Seq.cons b s') …
          m : Membership.mem (t₁.append s₂) a
          i1 : Eq c b
          i2 : Eq (t₁.append s₂) s'
          IH : ∀ {s₁ : Stream'.Seq α}, Membership.mem (s₁.append s₂) a → Eq (s₁.append s …
          ⊢ Or (Membership.mem (Stream'.Seq.cons c t₁) a) (Membership.mem s₂ a)
        -/
      · exact Or.imp_left (mem_cons_of_mem _) (IH m i2)
        /-
          🎉 no goals
        -/


theorem mem_append_left {s₁ s₂ : Seq α} {a : α} (h : a ∈ s₁) : a ∈ append s₁ s₂ := by
  /-
    α : Type u
    s₁ s₂ : Stream'.Seq α
    a : α
    h : Membership.mem s₁ a
    ⊢ Membership.mem (s₁.append s₂) a
  -/
  apply mem_rec_on h; intros; simp [*]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem enum_cons (s : Seq α) (x : α) :
    enum (cons x s) = cons (0, x) (map (Prod.map Nat.succ id) (enum s)) := by
  /-
    α : Type u
    s : Stream'.Seq α
    x : α
    ⊢ Eq (Stream'.Seq.cons x s).enum (Stream'.Seq.cons { fst := 0, snd := x } (Str …
  -/
  ext ⟨n⟩ : 1
    /-
      case h.zero
      α : Type u
      s : Stream'.Seq α
      x : α
      ⊢ Eq ((Stream'.Seq.cons x s).enum.get? 0) ((Stream'.Seq.cons { fst := 0, snd : …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      α : Type u
      s : Stream'.Seq α
      x : α
      n✝ : Nat
      ⊢ Eq ((Stream'.Seq.cons x s).enum.get? (HAdd.hAdd n✝ 1)) ((Stream'.Seq.cons {  …
    -/
  · simp only [get?_enum, get?_cons_succ, map_get?, Option.map_map]
    /-
      case h.succ
      α : Type u
      s : Stream'.Seq α
      x : α
      n✝ : Nat
      ⊢ Eq (Option.map (Prod.mk (HAdd.hAdd n✝ 1)) (s.get? n✝)) (Option.map (Function …
    -/
    congr
    /-
      🎉 no goals
    -/


/-- Convert a `Seq1` to a sequence. -/
def toSeq : Seq1 α → Seq α
  | (a, s) => Seq.cons a s


instance coeSeq : Coe (Seq1 α) (Seq α) :=
  ⟨toSeq⟩


/-- Map a function on a `Seq1` -/
def map (f : α → β) : Seq1 α → Seq1 β
  | (a, s) => (f a, Seq.map f s)


theorem map_pair {f : α → β} {a s} : map f (a, s) = (f a, Seq.map f s) := rfl


theorem map_id : ∀ s : Seq1 α, map id s = s
                 /-
                   α : Type u
                   a : α
                   s : Stream'.Seq α
                   ⊢ Eq (Stream'.Seq1.map id { fst := a, snd := s }) { fst := a, snd := s }
                 -/
  | ⟨a, s⟩ => by simp [map]
                 /-
                   🎉 no goals
                 -/


/-- Flatten a nonempty sequence of nonempty sequences -/
def join : Seq1 (Seq1 α) → Seq1 α
  | ((a, s), S) =>
    match destruct s with
    | none => (a, Seq.join S)
    | some s' => (a, Seq.join (Seq.cons s' S))


@[simp]
theorem join_nil (a : α) (S) : join ((a, nil), S) = (a, Seq.join S) :=
  rfl


@[simp]
theorem join_cons (a b : α) (s S) :
    join ((a, Seq.cons b s), S) = (a, Seq.join (Seq.cons (b, s) S)) := by
  /-
    α : Type u
    a b : α
    s : Stream'.Seq α
    S : Stream'.Seq (Stream'.Seq1 α)
    ⊢ Eq (Stream'.Seq1.join { fst := { fst := a, snd := Stream'.Seq.cons b s }, sn …
  -/
  dsimp [join]; rw [destruct_cons]
                /-
                  🎉 no goals
                -/


/-- The `return` operator for the `Seq1` monad,
  which produces a singleton sequence. -/
def ret (a : α) : Seq1 α :=
  (a, nil)


instance [Inhabited α] : Inhabited (Seq1 α) :=
  ⟨ret default⟩


/-- The `bind` operator for the `Seq1` monad,
  which maps `f` on each element of `s` and appends the results together.
  (Not all of `s` may be evaluated, because the first few elements of `s`
  may already produce an infinite result.) -/
def bind (s : Seq1 α) (f : α → Seq1 β) : Seq1 β :=
  join (map f s)


@[simp]
theorem join_map_ret (s : Seq α) : Seq.join (Seq.map ret s) = s := by
  /-
    α : Type u
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq.map Stream'.Seq1.ret s).join s
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  apply coinduction2 s; intro s; apply recOn s <;> simp [ret]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem bind_ret (f : α → β) : ∀ s, bind s (ret ∘ f) = map f s
  | ⟨a, s⟩ => by
    /-
      α : Type u
      β : Type v
      f : α → β
      a : α
      s : Stream'.Seq α
      ⊢ Eq (Stream'.Seq1.bind { fst := a, snd := s } (Function.comp Stream'.Seq1.ret …
    -/
    dsimp [bind, map]
    -- Porting note: Was `rw [map_comp]; simp [Function.comp, ret]`
    /-
      α : Type u
      β : Type v
      f : α → β
      a : α
      s : Stream'.Seq α
      ⊢ Eq (Stream'.Seq1.join { fst := Stream'.Seq1.ret (f a), snd := Stream'.Seq.ma …
    -/
    rw [map_comp, ret]
    /-
      α : Type u
      β : Type v
      f : α → β
      a : α
      s : Stream'.Seq α
      ⊢ Eq (Stream'.Seq1.join { fst := { fst := f a, snd := Stream'.Seq.nil }, snd : …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem ret_bind (a : α) (f : α → Seq1 β) : bind (ret a) f = f a := by
  /-
    α : Type u
    β : Type v
    a : α
    f : α → Stream'.Seq1 β
    ⊢ Eq ((Stream'.Seq1.ret a).bind f) (f a)
  -/
  simp only [bind, map, ret.eq_1, map_nil]
  /-
    α : Type u
    β : Type v
    a : α
    f : α → Stream'.Seq1 β
    ⊢ Eq (Stream'.Seq1.join { fst := f a, snd := Stream'.Seq.nil }) (f a)
  -/
  cases' f a with a s
  /-
    case mk
    α : Type u
    β : Type v
    a✝ : α
    f : α → Stream'.Seq1 β
    a : β
    s : Stream'.Seq β
    ⊢ Eq (Stream'.Seq1.join { fst := { fst := a, snd := s }, snd := Stream'.Seq.ni …
  -/
                               /-
                                 🎉 no goals
                               -/
  apply recOn s <;> intros <;> simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem map_join' (f : α → β) (S) : Seq.map f (Seq.join S) = Seq.join (Seq.map (map f) S) := by
  apply
    Seq.eq_of_bisim fun s1 s2 =>
      ∃ s S,
        s1 = Seq.append s (Seq.map f (Seq.join S)) ∧ s2 = append s (Seq.join (Seq.map (map f) S))
    /-
      case bisim
      α : Type u
      β : Type v
      f : α → β
      S : Stream'.Seq (Stream'.Seq1 α)
      ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => Exists fun S => And  …
    -/
  · intro s1 s2 h
    exact
      match s1, s2, h with
      | _, _, ⟨s, S, rfl, rfl⟩ => by
        apply recOn s <;> simp
        · apply recOn S <;> simp
          · intro x S
            cases' x with a s
            simpa [map] using ⟨_, _, rfl, rfl⟩
        · intro _ s
          exact ⟨s, S, rfl, rfl⟩
    /-
      case r
      α : Type u
      β : Type v
      f : α → β
      S : Stream'.Seq (Stream'.Seq1 α)
      ⊢ Exists fun s => Exists fun S_1 => And (Eq (Stream'.Seq.map f S.join) (s.appe …
    -/
                                /-
                                  🎉 no goals
                                -/
  · refine ⟨nil, S, ?_, ?_⟩ <;> simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem map_join (f : α → β) : ∀ S, map f (join S) = join (map (map f) S)
                      /-
                        α : Type u
                        β : Type v
                        f : α → β
                        a : α
                        s : Stream'.Seq α
                        S : Stream'.Seq (Stream'.Seq1 α)
                        ⊢ Eq (Stream'.Seq1.map f (Stream'.Seq1.join { fst := { fst := a, snd := s }, s …
                      -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  | ((a, s), S) => by apply recOn s <;> intros <;> simp [map]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem join_join (SS : Seq (Seq1 (Seq1 α))) :
    Seq.join (Seq.join SS) = Seq.join (Seq.map join SS) := by
  apply
    Seq.eq_of_bisim fun s1 s2 =>
      ∃ s SS,
        s1 = Seq.append s (Seq.join (Seq.join SS)) ∧ s2 = Seq.append s (Seq.join (Seq.map join SS))
    /-
      case bisim
      α : Type u
      SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 α))
      ⊢ Stream'.Seq.IsBisimulation fun s1 s2 => Exists fun s => Exists fun SS => And …
    -/
  · intro s1 s2 h
    exact
      match s1, s2, h with
      | _, _, ⟨s, SS, rfl, rfl⟩ => by
        apply recOn s <;> simp
        · apply recOn SS <;> simp
          · intro S SS
            cases' S with s S; cases' s with x s
            simp only [Seq.join_cons, join_append, destruct_cons]
            apply recOn s <;> simp
            · exact ⟨_, _, rfl, rfl⟩
            · intro x s
              refine ⟨Seq.cons x (append s (Seq.join S)), SS, ?_, ?_⟩ <;> simp
        · intro _ s
          exact ⟨s, SS, rfl, rfl⟩
    /-
      case r
      α : Type u
      SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 α))
      ⊢ Exists fun s => Exists fun SS_1 => And (Eq SS.join.join (s.append SS_1.join. …
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · refine ⟨nil, SS, ?_, ?_⟩ <;> simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem bind_assoc (s : Seq1 α) (f : α → Seq1 β) (g : β → Seq1 γ) :
    bind (bind s f) g = bind s fun x : α => bind (f x) g := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    s : Stream'.Seq1 α
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    ⊢ Eq ((s.bind f).bind g) (s.bind fun x => (f x).bind g)
  -/
  cases' s with a s
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [bind, map]`.
  /-
    case mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a : α
    s : Stream'.Seq α
    ⊢ Eq ((Stream'.Seq1.bind { fst := a, snd := s } f).bind g) (Stream'.Seq1.bind  …
  -/
  simp only [bind, map_pair, map_join]
  /-
    case mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a : α
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq1.join { fst := Stream'.Seq1.map g (f a), snd := Stream'.Seq. …
  -/
  rw [← map_comp]
  /-
    case mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a : α
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq1.join { fst := Stream'.Seq1.map g (f a), snd := Stream'.Seq. …
  -/
  simp only [show (fun x => join (map g (f x))) = join ∘ (map g ∘ f) from rfl]
  /-
    case mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a : α
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq1.join { fst := Stream'.Seq1.map g (f a), snd := Stream'.Seq. …
  -/
  rw [map_comp _ join]
  /-
    case mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a : α
    s : Stream'.Seq α
    ⊢ Eq (Stream'.Seq1.join { fst := Stream'.Seq1.map g (f a), snd := Stream'.Seq. …
  -/
  generalize Seq.map (map g ∘ f) s = SS
  /-
    case mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a : α
    s : Stream'.Seq α
    SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 γ))
    ⊢ Eq (Stream'.Seq1.join { fst := Stream'.Seq1.map g (f a), snd := SS }).join ( …
  -/
  rcases map g (f a) with ⟨⟨a, s⟩, S⟩
  -- Porting note: Instead of `apply recOn s <;> intros`, `induction'` are used to
  --   give names to variables.
  /-
    case mk.mk.mk
    α : Type u
    β : Type v
    γ : Type w
    f : α → Stream'.Seq1 β
    g : β → Stream'.Seq1 γ
    a✝ : α
    s✝ : Stream'.Seq α
    SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 γ))
    S : Stream'.Seq (Stream'.Seq1 γ)
    a : γ
    s : Stream'.Seq γ
    ⊢ Eq (Stream'.Seq1.join { fst := { fst := { fst := a, snd := s }, snd := S },  …
  -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  induction' s using recOn with x s_1 <;> induction' S using recOn with x_1 s_2 <;> simp
    /-
      case mk.mk.mk.h1.h2
      α : Type u
      β : Type v
      γ : Type w
      f : α → Stream'.Seq1 β
      g : β → Stream'.Seq1 γ
      a✝ : α
      s : Stream'.Seq α
      SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 γ))
      a : γ
      x_1 : Stream'.Seq1 γ
      s_2 : Stream'.Seq (Stream'.Seq1 γ)
      ⊢ Eq { fst := a, snd := (Stream'.Seq.cons x_1 (s_2.append SS.join)).join } (St …
    -/
  · cases' x_1 with x t
    /-
      case mk.mk.mk.h1.h2.mk
      α : Type u
      β : Type v
      γ : Type w
      f : α → Stream'.Seq1 β
      g : β → Stream'.Seq1 γ
      a✝ : α
      s : Stream'.Seq α
      SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 γ))
      a : γ
      s_2 : Stream'.Seq (Stream'.Seq1 γ)
      x : γ
      t : Stream'.Seq γ
      ⊢ Eq { fst := a, snd := (Stream'.Seq.cons { fst := x, snd := t } (s_2.append S …
    -/
                                 /-
                                   🎉 no goals
                                 -/
    apply recOn t <;> intros <;> simp
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case mk.mk.mk.h2.h2
      α : Type u
      β : Type v
      γ : Type w
      f : α → Stream'.Seq1 β
      g : β → Stream'.Seq1 γ
      a✝ : α
      s : Stream'.Seq α
      SS : Stream'.Seq (Stream'.Seq1 (Stream'.Seq1 γ))
      a x : γ
      s_1 : Stream'.Seq γ
      x_1 : Stream'.Seq1 γ
      s_2 : Stream'.Seq (Stream'.Seq1 γ)
      ⊢ Eq { fst := a, snd := Stream'.Seq.cons x (s_1.append (Stream'.Seq.cons x_1 ( …
    -/
  · cases' x_1 with y t; simp
                         /-
                           🎉 no goals
                         -/


instance monad : Monad Seq1 where
  map := @map
  pure := @ret
  bind := @bind


                                           /-
                                             α : Type u
                                             β : Type v
                                             γ : Type w
                                             ⊢ ∀ {α β : Type u_1} (x : α) (y : Stream'.Seq1 β), Eq (Functor.mapConst x y) ( …
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
instance lawfulMonad : LawfulMonad Seq1 := LawfulMonad.mk'
                                           /-
                                             🎉 no goals
                                           -/
  (id_map := @map_id)
  (bind_pure_comp := @bind_ret)
  (pure_bind := @ret_bind)
  (bind_assoc := @bind_assoc)


