instance [Inhabited α] : Inhabited (Stream' α) :=
  ⟨Stream'.const default⟩


protected theorem eta (s : Stream' α) : (head s::tail s) = s :=
                     /-
                       α : Type u
                       s : Stream' α
                       i : Nat
                       ⊢ Eq (Stream'.cons s.head s.tail i) (s i)
                     -/
                                 /-
                                   🎉 no goals
                                 -/
  funext fun i => by cases i <;> rfl
                                 /-
                                   🎉 no goals
                                 -/


@[ext]
protected theorem ext {s₁ s₂ : Stream' α} : (∀ n, get s₁ n = get s₂ n) → s₁ = s₂ :=
  fun h => funext h


@[simp]
theorem get_zero_cons (a : α) (s : Stream' α) : get (a::s) 0 = a :=
  rfl


@[simp]
theorem head_cons (a : α) (s : Stream' α) : head (a::s) = a :=
  rfl


@[simp]
theorem tail_cons (a : α) (s : Stream' α) : tail (a::s) = s :=
  rfl


@[simp]
theorem get_drop (n m : ℕ) (s : Stream' α) : get (drop m s) n = get s (n + m) :=
  rfl


theorem tail_eq_drop (s : Stream' α) : tail s = drop 1 s :=
  rfl


@[simp]
theorem drop_drop (n m : ℕ) (s : Stream' α) : drop n (drop m s) = drop (n + m) s := by
  /-
    α : Type u
    n m : Nat
    s : Stream' α
    ⊢ Eq (Stream'.drop n (Stream'.drop m s)) (Stream'.drop (HAdd.hAdd n m) s)
  -/
  ext; simp [Nat.add_assoc]
       /-
         🎉 no goals
       -/


@[simp] theorem get_tail {n : ℕ} {s : Stream' α} : s.tail.get n = s.get (n + 1) := rfl


@[simp] theorem tail_drop' {i : ℕ} {s : Stream' α} : tail (drop i s) = s.drop (i+1) := by
  /-
    α : Type u
    i : Nat
    s : Stream' α
    ⊢ Eq (Stream'.drop i s).tail (Stream'.drop (HAdd.hAdd i 1) s)
  -/
  ext; simp [Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]
       /-
         🎉 no goals
       -/


@[simp] theorem drop_tail' {i : ℕ} {s : Stream' α} : drop i (tail s) = s.drop (i+1) := rfl


                                                                                    /-
                                                                                      α : Type u
                                                                                      n : Nat
                                                                                      s : Stream' α
                                                                                      ⊢ Eq (Stream'.drop n s).tail (Stream'.drop n s.tail)
                                                                                    -/
theorem tail_drop (n : ℕ) (s : Stream' α) : tail (drop n s) = drop n (tail s) := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem get_succ (n : ℕ) (s : Stream' α) : get s (succ n) = get (tail s) n :=
  rfl


@[simp]
theorem get_succ_cons (n : ℕ) (s : Stream' α) (x : α) : get (x::s) n.succ = get s n :=
  rfl


@[simp] theorem drop_zero {s : Stream' α} : s.drop 0 = s := rfl


theorem drop_succ (n : ℕ) (s : Stream' α) : drop (succ n) s = drop n (tail s) :=
  rfl


                                                                            /-
                                                                              α : Type u
                                                                              a : Stream' α
                                                                              n : Nat
                                                                              ⊢ Eq (Stream'.drop n a).head (a.get n)
                                                                            -/
theorem head_drop (a : Stream' α) (n : ℕ) : (a.drop n).head = a.get n := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem cons_injective2 : Function.Injective2 (cons : α → Stream' α → Stream' α) := fun x y s t h =>
      /-
        α : Type u
        x y : α
        s t : Stream' α
        h : Eq (Stream'.cons x s) (Stream'.cons y t)
        ⊢ Eq x y
      -/
  ⟨by rw [← get_zero_cons x s, h, get_zero_cons],
      /-
        🎉 no goals
      -/
                            /-
                              α : Type u
                              x y : α
                              s t : Stream' α
                              h : Eq (Stream'.cons x s) (Stream'.cons y t)
                              n : Nat
                              ⊢ Eq (s.get n) (t.get n)
                            -/
    Stream'.ext fun n => by rw [← get_succ_cons n _ x, h, get_succ_cons]⟩
                            /-
                              🎉 no goals
                            -/


theorem cons_injective_left (s : Stream' α) : Function.Injective fun x => cons x s :=
  cons_injective2.left _


theorem cons_injective_right (x : α) : Function.Injective (cons x) :=
  cons_injective2.right _


theorem all_def (p : α → Prop) (s : Stream' α) : All p s = ∀ n, p (get s n) :=
  rfl


theorem any_def (p : α → Prop) (s : Stream' α) : Any p s = ∃ n, p (get s n) :=
  rfl


@[simp]
theorem mem_cons (a : α) (s : Stream' α) : a ∈ a::s :=
  Exists.intro 0 rfl


theorem mem_cons_of_mem {a : α} {s : Stream' α} (b : α) : a ∈ s → a ∈ b::s := fun ⟨n, h⟩ =>
                            /-
                              α : Type u
                              a : α
                              s : Stream' α
                              b : α
                              x✝ : Membership.mem s a
                              n : Nat
                              h : (fun b => Eq a b) (s.get n)
                              ⊢ (fun b => Eq a b) ((Stream'.cons b s).get n.succ)
                            -/
  Exists.intro (succ n) (by rw [get_succ, tail_cons, h])
                            /-
                              🎉 no goals
                            -/


theorem eq_or_mem_of_mem_cons {a b : α} {s : Stream' α} : (a ∈ b::s) → a = b ∨ a ∈ s :=
    fun ⟨n, h⟩ => by
  /-
    α : Type u
    a b : α
    s : Stream' α
    x✝ : Membership.mem (Stream'.cons b s) a
    n : Nat
    h : (fun b => Eq a b) ((Stream'.cons b s).get n)
    ⊢ Or (Eq a b) (Membership.mem s a)
  -/
  cases' n with n'
    /-
      case zero
      α : Type u
      a b : α
      s : Stream' α
      x✝ : Membership.mem (Stream'.cons b s) a
      h : Eq a ((Stream'.cons b s).get 0)
      ⊢ Or (Eq a b) (Membership.mem s a)
    -/
  · left
    /-
      case zero.h
      α : Type u
      a b : α
      s : Stream' α
      x✝ : Membership.mem (Stream'.cons b s) a
      h : Eq a ((Stream'.cons b s).get 0)
      ⊢ Eq a b
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      a b : α
      s : Stream' α
      x✝ : Membership.mem (Stream'.cons b s) a
      n' : Nat
      h : Eq a ((Stream'.cons b s).get (HAdd.hAdd n' 1))
      ⊢ Or (Eq a b) (Membership.mem s a)
    -/
  · right
    /-
      case succ.h
      α : Type u
      a b : α
      s : Stream' α
      x✝ : Membership.mem (Stream'.cons b s) a
      n' : Nat
      h : Eq a ((Stream'.cons b s).get (HAdd.hAdd n' 1))
      ⊢ Membership.mem s a
    -/
    rw [get_succ, tail_cons] at h
    /-
      case succ.h
      α : Type u
      a b : α
      s : Stream' α
      x✝ : Membership.mem (Stream'.cons b s) a
      n' : Nat
      h : Eq a (s.get n')
      ⊢ Membership.mem s a
    -/
    exact ⟨n', h⟩
    /-
      🎉 no goals
    -/


theorem mem_of_get_eq {n : ℕ} {s : Stream' α} {a : α} : a = get s n → a ∈ s := fun h =>
  Exists.intro n h


theorem drop_map (n : ℕ) (s : Stream' α) : drop n (map f s) = map f (drop n s) :=
  Stream'.ext fun _ => rfl


@[simp]
theorem get_map (n : ℕ) (s : Stream' α) : get (map f s) n = f (get s n) :=
  rfl


theorem tail_map (s : Stream' α) : tail (map f s) = map f (tail s) := rfl


@[simp]
theorem head_map (s : Stream' α) : head (map f s) = f (head s) :=
  rfl


theorem map_eq (s : Stream' α) : map f s = f (head s)::map f (tail s) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Stream' α
    ⊢ Eq (Stream'.map f s) (Stream'.cons (f s.head) (Stream'.map f s.tail))
  -/
  rw [← Stream'.eta (map f s), tail_map, head_map]
  /-
    🎉 no goals
  -/


theorem map_cons (a : α) (s : Stream' α) : map f (a::s) = f a::map f s := by
  /-
    α : Type u
    β : Type v
    f : α → β
    a : α
    s : Stream' α
    ⊢ Eq (Stream'.map f (Stream'.cons a s)) (Stream'.cons (f a) (Stream'.map f s))
  -/
  rw [← Stream'.eta (map f (a::s)), map_eq]; rfl
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem map_id (s : Stream' α) : map id s = s :=
  rfl


@[simp]
theorem map_map (g : β → δ) (f : α → β) (s : Stream' α) : map g (map f s) = map (g ∘ f) s :=
  rfl


@[simp]
theorem map_tail (s : Stream' α) : map f (tail s) = tail (map f s) :=
  rfl


theorem mem_map {a : α} {s : Stream' α} : a ∈ s → f a ∈ map f s := fun ⟨n, h⟩ =>
                     /-
                       α : Type u
                       β : Type v
                       f : α → β
                       a : α
                       s : Stream' α
                       x✝ : Membership.mem s a
                       n : Nat
                       h : (fun b => Eq a b) (s.get n)
                       ⊢ (fun b => Eq (f a) b) ((Stream'.map f s).get n)
                     -/
  Exists.intro n (by rw [get_map, h])
                     /-
                       🎉 no goals
                     -/


theorem exists_of_mem_map {f} {b : β} {s : Stream' α} : b ∈ map f s → ∃ a, a ∈ s ∧ f a = b :=
  fun ⟨n, h⟩ => ⟨get s n, ⟨n, rfl⟩, h.symm⟩


theorem drop_zip (n : ℕ) (s₁ : Stream' α) (s₂ : Stream' β) :
    drop n (zip f s₁ s₂) = zip f (drop n s₁) (drop n s₂) :=
  Stream'.ext fun _ => rfl


@[simp]
theorem get_zip (n : ℕ) (s₁ : Stream' α) (s₂ : Stream' β) :
    get (zip f s₁ s₂) n = f (get s₁ n) (get s₂ n) :=
  rfl


theorem head_zip (s₁ : Stream' α) (s₂ : Stream' β) : head (zip f s₁ s₂) = f (head s₁) (head s₂) :=
  rfl


theorem tail_zip (s₁ : Stream' α) (s₂ : Stream' β) :
    tail (zip f s₁ s₂) = zip f (tail s₁) (tail s₂) :=
  rfl


theorem zip_eq (s₁ : Stream' α) (s₂ : Stream' β) :
    zip f s₁ s₂ = f (head s₁) (head s₂)::zip f (tail s₁) (tail s₂) := by
  /-
    α : Type u
    β : Type v
    δ : Type w
    f : α → β → δ
    s₁ : Stream' α
    s₂ : Stream' β
    ⊢ Eq (Stream'.zip f s₁ s₂) (Stream'.cons (f s₁.head s₂.head) (Stream'.zip f s₁ …
  -/
  rw [← Stream'.eta (zip f s₁ s₂)]; rfl
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem get_enum (s : Stream' α) (n : ℕ) : get (enum s) n = (n, s.get n) :=
  rfl


theorem enum_eq_zip (s : Stream' α) : enum s = zip Prod.mk nats s :=
  rfl


@[simp]
theorem mem_const (a : α) : a ∈ const a :=
  Exists.intro 0 rfl


theorem const_eq (a : α) : const a = a::const a := by
  /-
    α : Type u
    a : α
    ⊢ Eq (Stream'.const a) (Stream'.cons a (Stream'.const a))
  -/
  apply Stream'.ext; intro n
  /-
    case a
    α : Type u
    a : α
    n : Nat
    ⊢ Eq ((Stream'.const a).get n) ((Stream'.cons a (Stream'.const a)).get n)
  -/
              /-
                🎉 no goals
              -/
  cases n <;> rfl
              /-
                🎉 no goals
              -/


@[simp]
theorem tail_const (a : α) : tail (const a) = const a :=
                                          /-
                                            α : Type u
                                            a : α
                                            this : Eq (Stream'.cons a (Stream'.const a)).tail (Stream'.const a)
                                            ⊢ Eq (Stream'.const a).tail (Stream'.const a)
                                          -/
  suffices tail (a::const a) = const a by rwa [← const_eq] at this
                                          /-
                                            🎉 no goals
                                          -/
  rfl


@[simp]
theorem map_const (f : α → β) (a : α) : map f (const a) = const (f a) :=
  rfl


@[simp]
theorem get_const (n : ℕ) (a : α) : get (const a) n = a :=
  rfl


@[simp]
theorem drop_const (n : ℕ) (a : α) : drop n (const a) = const a :=
  Stream'.ext fun _ => rfl


@[simp]
theorem head_iterate (f : α → α) (a : α) : head (iterate f a) = a :=
  rfl


theorem get_succ_iterate' (n : ℕ) (f : α → α) (a : α) :
    get (iterate f a) (succ n) = f (get (iterate f a) n) := rfl


theorem tail_iterate (f : α → α) (a : α) : tail (iterate f a) = iterate f (f a) := by
  /-
    α : Type u
    f : α → α
    a : α
    ⊢ Eq (Stream'.iterate f a).tail (Stream'.iterate f (f a))
  -/
  ext n
  /-
    case a
    α : Type u
    f : α → α
    a : α
    n : Nat
    ⊢ Eq ((Stream'.iterate f a).tail.get n) ((Stream'.iterate f (f a)).get n)
  -/
  rw [get_tail]
  /-
    case a
    α : Type u
    f : α → α
    a : α
    n : Nat
    ⊢ Eq ((Stream'.iterate f a).get (HAdd.hAdd n 1)) ((Stream'.iterate f (f a)).ge …
  -/
  induction' n with n' ih
    /-
      case a.zero
      α : Type u
      f : α → α
      a : α
      ⊢ Eq ((Stream'.iterate f a).get (HAdd.hAdd 0 1)) ((Stream'.iterate f (f a)).ge …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case a.succ
      α : Type u
      f : α → α
      a : α
      n' : Nat
      ih : Eq ((Stream'.iterate f a).get (HAdd.hAdd n' 1)) ((Stream'.iterate f (f a) …
      ⊢ Eq ((Stream'.iterate f a).get (HAdd.hAdd (HAdd.hAdd n' 1) 1)) ((Stream'.iter …
    -/
  · rw [get_succ_iterate', ih, get_succ_iterate']
    /-
      🎉 no goals
    -/


theorem iterate_eq (f : α → α) (a : α) : iterate f a = a::iterate f (f a) := by
  /-
    α : Type u
    f : α → α
    a : α
    ⊢ Eq (Stream'.iterate f a) (Stream'.cons a (Stream'.iterate f (f a)))
  -/
  rw [← Stream'.eta (iterate f a)]
  /-
    α : Type u
    f : α → α
    a : α
    ⊢ Eq (Stream'.cons (Stream'.iterate f a).head (Stream'.iterate f a).tail) (Str …
  -/
  rw [tail_iterate]; rfl
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem get_zero_iterate (f : α → α) (a : α) : get (iterate f a) 0 = a :=
  rfl


theorem get_succ_iterate (n : ℕ) (f : α → α) (a : α) :
                                                               /-
                                                                 α : Type u
                                                                 n : Nat
                                                                 f : α → α
                                                                 a : α
                                                                 ⊢ Eq ((Stream'.iterate f a).get n.succ) ((Stream'.iterate f (f a)).get n)
                                                               -/
    get (iterate f a) (succ n) = get (iterate f (f a)) n := by rw [get_succ, tail_iterate]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- equivalence relation -/
local infixl:50 " ~ " => R


/-- Streams `s₁` and `s₂` are defined to be bisimulations if
their heads are equal and tails are bisimulations. -/
def IsBisimulation :=
  ∀ ⦃s₁ s₂⦄, s₁ ~ s₂ →
      head s₁ = head s₂ ∧ tail s₁ ~ tail s₂


theorem get_of_bisim (bisim : IsBisimulation R) :
    ∀ {s₁ s₂} (n), s₁ ~ s₂ → get s₁ n = get s₂ n ∧ drop (n + 1) s₁ ~ drop (n + 1) s₂
  | _, _, 0, h => bisim h
  | _, _, n + 1, h =>
    match bisim h with
    | ⟨_, trel⟩ => get_of_bisim bisim n trel

-- If two streams are bisimilar, then they are equal

theorem eq_of_bisim (bisim : IsBisimulation R) : ∀ {s₁ s₂}, s₁ ~ s₂ → s₁ = s₂ := fun r =>
  Stream'.ext fun n => And.left (get_of_bisim R bisim n r)


theorem bisim_simple (s₁ s₂ : Stream' α) :
    head s₁ = head s₂ → s₁ = tail s₁ → s₂ = tail s₂ → s₁ = s₂ := fun hh ht₁ ht₂ =>
  eq_of_bisim (fun s₁ s₂ => head s₁ = head s₂ ∧ s₁ = tail s₁ ∧ s₂ = tail s₂)
    (fun s₁ s₂ ⟨h₁, h₂, h₃⟩ => by
      /-
        α : Type u
        s₁✝ s₂✝ : Stream' α
        hh : Eq s₁✝.head s₂✝.head
        ht₁ : Eq s₁✝ s₁✝.tail
        ht₂ : Eq s₂✝ s₂✝.tail
        s₁ s₂ : Stream' α
        x✝ : (fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ s₁.tail) (Eq s₂ s₂.tai …
        h₁ : Eq s₁.head s₂.head
        h₂ : Eq s₁ s₁.tail
        h₃ : Eq s₂ s₂.tail
        ⊢ And (Eq s₁.head s₂.head) ((fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ …
      -/
      constructor
        /-
          case left
          α : Type u
          s₁✝ s₂✝ : Stream' α
          hh : Eq s₁✝.head s₂✝.head
          ht₁ : Eq s₁✝ s₁✝.tail
          ht₂ : Eq s₂✝ s₂✝.tail
          s₁ s₂ : Stream' α
          x✝ : (fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ s₁.tail) (Eq s₂ s₂.tai …
          h₁ : Eq s₁.head s₂.head
          h₂ : Eq s₁ s₁.tail
          h₃ : Eq s₂ s₂.tail
          ⊢ Eq s₁.head s₂.head
        -/
      · exact h₁
        /-
          🎉 no goals
        -/
      /-
        case right
        α : Type u
        s₁✝ s₂✝ : Stream' α
        hh : Eq s₁✝.head s₂✝.head
        ht₁ : Eq s₁✝ s₁✝.tail
        ht₂ : Eq s₂✝ s₂✝.tail
        s₁ s₂ : Stream' α
        x✝ : (fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ s₁.tail) (Eq s₂ s₂.tai …
        h₁ : Eq s₁.head s₂.head
        h₂ : Eq s₁ s₁.tail
        h₃ : Eq s₂ s₂.tail
        ⊢ (fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ s₁.tail) (Eq s₂ s₂.tail)) …
      -/
      rw [← h₂, ← h₃]
       /-
         case right
         α : Type u
         s₁✝ s₂✝ : Stream' α
         hh : Eq s₁✝.head s₂✝.head
         ht₁ : Eq s₁✝ s₁✝.tail
         ht₂ : Eq s₂✝ s₂✝.tail
         s₁ s₂ : Stream' α
         x✝ : (fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ s₁.tail) (Eq s₂ s₂.tai …
         h₁ : Eq s₁.head s₂.head
         h₂ : Eq s₁ s₁.tail
         h₃ : Eq s₂ s₂.tail
         ⊢ (fun s₁ s₂ => And (Eq s₁.head s₂.head) (And (Eq s₁ s₁.tail) (Eq s₂ s₂.tail)) …
       -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
      (repeat' constructor) <;> assumption)
                                /-
                                  🎉 no goals
                                -/
    (And.intro hh (And.intro ht₁ ht₂))


theorem coinduction {s₁ s₂ : Stream' α} :
    head s₁ = head s₂ →
      (∀ (β : Type u) (fr : Stream' α → β),
      fr s₁ = fr s₂ → fr (tail s₁) = fr (tail s₂)) → s₁ = s₂ :=
  fun hh ht =>
  eq_of_bisim
    (fun s₁ s₂ =>
      head s₁ = head s₂ ∧
        ∀ (β : Type u) (fr : Stream' α → β), fr s₁ = fr s₂ → fr (tail s₁) = fr (tail s₂))
    (fun s₁ s₂ h =>
      have h₁ : head s₁ = head s₂ := And.left h
      have h₂ : head (tail s₁) = head (tail s₂) := And.right h α (@head α) h₁
      have h₃ :
        ∀ (β : Type u) (fr : Stream' α → β),
          fr (tail s₁) = fr (tail s₂) → fr (tail (tail s₁)) = fr (tail (tail s₂)) :=
        fun β fr => And.right h β fun s => fr (tail s)
      And.intro h₁ (And.intro h₂ h₃))
    (And.intro hh ht)


@[simp]
theorem iterate_id (a : α) : iterate id a = const a :=
                                    /-
                                      α : Type u
                                      a : α
                                      β : Type u
                                      fr : Stream' α → β
                                      ch : Eq (fr (Stream'.iterate id a)) (fr (Stream'.const a))
                                      ⊢ Eq (fr (Stream'.iterate id a).tail) (fr (Stream'.const a).tail)
                                    -/
  coinduction rfl fun β fr ch => by rw [tail_iterate, tail_const]; exact ch
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem map_iterate (f : α → α) (a : α) : iterate f (f a) = map f (iterate f a) := by
  /-
    α : Type u
    f : α → α
    a : α
    ⊢ Eq (Stream'.iterate f (f a)) (Stream'.map f (Stream'.iterate f a))
  -/
  funext n
  /-
    case h
    α : Type u
    f : α → α
    a : α
    n : Nat
    ⊢ Eq (Stream'.iterate f (f a) n) (Stream'.map f (Stream'.iterate f a) n)
  -/
  induction' n with n' ih
    /-
      case h.zero
      α : Type u
      f : α → α
      a : α
      ⊢ Eq (Stream'.iterate f (f a) 0) (Stream'.map f (Stream'.iterate f a) 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      α : Type u
      f : α → α
      a : α
      n' : Nat
      ih : Eq (Stream'.iterate f (f a) n') (Stream'.map f (Stream'.iterate f a) n')
      ⊢ Eq (Stream'.iterate f (f a) (HAdd.hAdd n' 1)) (Stream'.map f (Stream'.iterat …
    -/
  · unfold map iterate get
    /-
      case h.succ
      α : Type u
      f : α → α
      a : α
      n' : Nat
      ih : Eq (Stream'.iterate f (f a) n') (Stream'.map f (Stream'.iterate f a) n')
      ⊢ Eq (f (Stream'.iterate f (f a) n')) (f (Stream'.iterate f a (HAdd.hAdd n' 1)))
    -/
    rw [map, get] at ih
    /-
      case h.succ
      α : Type u
      f : α → α
      a : α
      n' : Nat
      ih : Eq (Stream'.iterate f (f a) n') (f (Stream'.iterate f a n'))
      ⊢ Eq (f (Stream'.iterate f (f a) n')) (f (Stream'.iterate f a (HAdd.hAdd n' 1)))
    -/
    rw [iterate]
    /-
      case h.succ
      α : Type u
      f : α → α
      a : α
      n' : Nat
      ih : Eq (Stream'.iterate f (f a) n') (f (Stream'.iterate f a n'))
      ⊢ Eq (f (Stream'.iterate f (f a) n')) (f (f (Stream'.iterate f a n')))
    -/
    exact congrArg f ih
    /-
      🎉 no goals
    -/


theorem corec_def (f : α → β) (g : α → α) (a : α) : corec f g a = map f (iterate g a) :=
  rfl


theorem corec_eq (f : α → β) (g : α → α) (a : α) : corec f g a = f a::corec f g (g a) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    g : α → α
    a : α
    ⊢ Eq (Stream'.corec f g a) (Stream'.cons (f a) (Stream'.corec f g (g a)))
  -/
  rw [corec_def, map_eq, head_iterate, tail_iterate]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem corec_id_id_eq_const (a : α) : corec id id a = const a := by
  /-
    α : Type u
    a : α
    ⊢ Eq (Stream'.corec id id a) (Stream'.const a)
  -/
  rw [corec_def, map_id, iterate_id]
  /-
    🎉 no goals
  -/


theorem corec_id_f_eq_iterate (f : α → α) (a : α) : corec id f a = iterate f a :=
  rfl


theorem corec'_eq (f : α → β × α) (a : α) : corec' f a = (f a).1::corec' f (f a).2 :=
  corec_eq _ _ _


theorem unfolds_eq (g : α → β) (f : α → α) (a : α) : unfolds g f a = g a::unfolds g f (f a) := by
  /-
    α : Type u
    β : Type v
    g : α → β
    f : α → α
    a : α
    ⊢ Eq (Stream'.unfolds g f a) (Stream'.cons (g a) (Stream'.unfolds g f (f a)))
  -/
  unfold unfolds; rw [corec_eq]
                  /-
                    🎉 no goals
                  -/


theorem get_unfolds_head_tail : ∀ (n : ℕ) (s : Stream' α),
    get (unfolds head tail s) n = get s n := by
  /-
    α : Type u
    ⊢ ∀ (n : Nat) (s : Stream' α), Eq ((Stream'.unfolds Stream'.head Stream'.tail  …
  -/
  intro n; induction' n with n' ih
    /-
      case zero
      α : Type u
      ⊢ ∀ (s : Stream' α), Eq ((Stream'.unfolds Stream'.head Stream'.tail s).get 0)  …
    -/
  · intro s
    /-
      case zero
      α : Type u
      s : Stream' α
      ⊢ Eq ((Stream'.unfolds Stream'.head Stream'.tail s).get 0) (s.get 0)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq ((Stream'.unfolds Stream'.head Stream'.tail s).get  …
      ⊢ ∀ (s : Stream' α), Eq ((Stream'.unfolds Stream'.head Stream'.tail s).get (HA …
    -/
  · intro s
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq ((Stream'.unfolds Stream'.head Stream'.tail s).get  …
      s : Stream' α
      ⊢ Eq ((Stream'.unfolds Stream'.head Stream'.tail s).get (HAdd.hAdd n' 1)) (s.g …
    -/
    rw [get_succ, get_succ, unfolds_eq, tail_cons, ih]
    /-
      🎉 no goals
    -/


theorem unfolds_head_eq : ∀ s : Stream' α, unfolds head tail s = s := fun s =>
  Stream'.ext fun n => get_unfolds_head_tail n s


theorem interleave_eq (s₁ s₂ : Stream' α) : s₁ ⋈ s₂ = head s₁::head s₂::(tail s₁ ⋈ tail s₂) := by
  /-
    α : Type u
    s₁ s₂ : Stream' α
    ⊢ Eq (s₁.interleave s₂) (Stream'.cons s₁.head (Stream'.cons s₂.head (s₁.tail.i …
  -/
  let t := tail s₁ ⋈ tail s₂
  /-
    α : Type u
    s₁ s₂ : Stream' α
    t : Stream' α := s₁.tail.interleave s₂.tail
    ⊢ Eq (s₁.interleave s₂) (Stream'.cons s₁.head (Stream'.cons s₂.head (s₁.tail.i …
  -/
  show s₁ ⋈ s₂ = head s₁::head s₂::t
  /-
    α : Type u
    s₁ s₂ : Stream' α
    t : Stream' α := s₁.tail.interleave s₂.tail
    ⊢ Eq (s₁.interleave s₂) (Stream'.cons s₁.head (Stream'.cons s₂.head t))
  -/
  unfold interleave; unfold corecOn; rw [corec_eq]; dsimp; rw [corec_eq]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem tail_interleave (s₁ s₂ : Stream' α) : tail (s₁ ⋈ s₂) = s₂ ⋈ tail s₁ := by
  /-
    α : Type u
    s₁ s₂ : Stream' α
    ⊢ Eq (s₁.interleave s₂).tail (s₂.interleave s₁.tail)
  -/
  unfold interleave corecOn; rw [corec_eq]; rfl
                                            /-
                                              🎉 no goals
                                            -/


theorem interleave_tail_tail (s₁ s₂ : Stream' α) : tail s₁ ⋈ tail s₂ = tail (tail (s₁ ⋈ s₂)) := by
  /-
    α : Type u
    s₁ s₂ : Stream' α
    ⊢ Eq (s₁.tail.interleave s₂.tail) (s₁.interleave s₂).tail.tail
  -/
  rw [interleave_eq s₁ s₂]; rfl
                            /-
                              🎉 no goals
                            -/


theorem get_interleave_left : ∀ (n : ℕ) (s₁ s₂ : Stream' α),
    get (s₁ ⋈ s₂) (2 * n) = get s₁ n
  | 0, _, _ => rfl
  | n + 1, s₁, s₂ => by
    /-
      α : Type u
      n : Nat
      s₁ s₂ : Stream' α
      ⊢ Eq ((s₁.interleave s₂).get (HMul.hMul 2 (HAdd.hAdd n 1))) (s₁.get (HAdd.hAdd …
    -/
    change get (s₁ ⋈ s₂) (succ (succ (2 * n))) = get s₁ (succ n)
    /-
      α : Type u
      n : Nat
      s₁ s₂ : Stream' α
      ⊢ Eq ((s₁.interleave s₂).get (HMul.hMul 2 n).succ.succ) (s₁.get n.succ)
    -/
    rw [get_succ, get_succ, interleave_eq, tail_cons, tail_cons]
    /-
      α : Type u
      n : Nat
      s₁ s₂ : Stream' α
      ⊢ Eq ((s₁.tail.interleave s₂.tail).get (HMul.hMul 2 n)) (s₁.get n.succ)
    -/
    rw [get_interleave_left n (tail s₁) (tail s₂)]
    /-
      α : Type u
      n : Nat
      s₁ s₂ : Stream' α
      ⊢ Eq (s₁.tail.get n) (s₁.get n.succ)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem get_interleave_right : ∀ (n : ℕ) (s₁ s₂ : Stream' α),
    get (s₁ ⋈ s₂) (2 * n + 1) = get s₂ n
  | 0, _, _ => rfl
  | n + 1, s₁, s₂ => by
    /-
      α : Type u
      n : Nat
      s₁ s₂ : Stream' α
      ⊢ Eq ((s₁.interleave s₂).get (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd n 1)) 1)) (s₂. …
    -/
    change get (s₁ ⋈ s₂) (succ (succ (2 * n + 1))) = get s₂ (succ n)
    rw [get_succ, get_succ, interleave_eq, tail_cons, tail_cons,
      get_interleave_right n (tail s₁) (tail s₂)]
    /-
      α : Type u
      n : Nat
      s₁ s₂ : Stream' α
      ⊢ Eq (s₂.tail.get n) (s₂.get n.succ)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem mem_interleave_left {a : α} {s₁ : Stream' α} (s₂ : Stream' α) : a ∈ s₁ → a ∈ s₁ ⋈ s₂ :=
                                         /-
                                           α : Type u
                                           a : α
                                           s₁ s₂ : Stream' α
                                           x✝ : Membership.mem s₁ a
                                           n : Nat
                                           h : (fun b => Eq a b) (s₁.get n)
                                           ⊢ (fun b => Eq a b) ((s₁.interleave s₂).get (HMul.hMul 2 n))
                                         -/
  fun ⟨n, h⟩ => Exists.intro (2 * n) (by rw [h, get_interleave_left])
                                         /-
                                           🎉 no goals
                                         -/


theorem mem_interleave_right {a : α} {s₁ : Stream' α} (s₂ : Stream' α) : a ∈ s₂ → a ∈ s₁ ⋈ s₂ :=
                                             /-
                                               α : Type u
                                               a : α
                                               s₁ s₂ : Stream' α
                                               x✝ : Membership.mem s₂ a
                                               n : Nat
                                               h : (fun b => Eq a b) (s₂.get n)
                                               ⊢ (fun b => Eq a b) ((s₁.interleave s₂).get (HAdd.hAdd (HMul.hMul 2 n) 1))
                                             -/
  fun ⟨n, h⟩ => Exists.intro (2 * n + 1) (by rw [h, get_interleave_right])
                                             /-
                                               🎉 no goals
                                             -/


theorem odd_eq (s : Stream' α) : odd s = even (tail s) :=
  rfl


@[simp]
theorem head_even (s : Stream' α) : head (even s) = head s :=
  rfl


theorem tail_even (s : Stream' α) : tail (even s) = even (tail (tail s)) := by
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq s.even.tail s.tail.tail.even
  -/
  unfold even
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq (Stream'.corec Stream'.head (fun s => s.tail.tail) s).tail (Stream'.corec …
  -/
  rw [corec_eq]
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq (Stream'.cons s.head (Stream'.corec Stream'.head (fun s => s.tail.tail) s …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem even_cons_cons (a₁ a₂ : α) (s : Stream' α) : even (a₁::a₂::s) = a₁::even s := by
  /-
    α : Type u
    a₁ a₂ : α
    s : Stream' α
    ⊢ Eq (Stream'.cons a₁ (Stream'.cons a₂ s)).even (Stream'.cons a₁ s.even)
  -/
  unfold even
  /-
    α : Type u
    a₁ a₂ : α
    s : Stream' α
    ⊢ Eq (Stream'.corec Stream'.head (fun s => s.tail.tail) (Stream'.cons a₁ (Stre …
  -/
  rw [corec_eq]; rfl
                 /-
                   🎉 no goals
                 -/


theorem even_tail (s : Stream' α) : even (tail s) = odd s :=
  rfl


theorem even_interleave (s₁ s₂ : Stream' α) : even (s₁ ⋈ s₂) = s₁ :=
  eq_of_bisim (fun s₁' s₁ => ∃ s₂, s₁' = even (s₁ ⋈ s₂))
    (fun s₁' s₁ ⟨s₂, h₁⟩ => by
      /-
        α : Type u
        s₁✝ s₂✝ s₁' s₁ : Stream' α
        x✝ : (fun s₁' s₁ => Exists fun s₂ => Eq s₁' (s₁.interleave s₂).even) s₁' s₁
        s₂ : Stream' α
        h₁ : Eq s₁' (s₁.interleave s₂).even
        ⊢ And (Eq s₁'.head s₁.head) ((fun s₁' s₁ => Exists fun s₂ => Eq s₁' (s₁.interl …
      -/
      rw [h₁]
      /-
        α : Type u
        s₁✝ s₂✝ s₁' s₁ : Stream' α
        x✝ : (fun s₁' s₁ => Exists fun s₂ => Eq s₁' (s₁.interleave s₂).even) s₁' s₁
        s₂ : Stream' α
        h₁ : Eq s₁' (s₁.interleave s₂).even
        ⊢ And (Eq (s₁.interleave s₂).even.head s₁.head) ((fun s₁' s₁ => Exists fun s₂  …
      -/
      constructor
        /-
          case left
          α : Type u
          s₁✝ s₂✝ s₁' s₁ : Stream' α
          x✝ : (fun s₁' s₁ => Exists fun s₂ => Eq s₁' (s₁.interleave s₂).even) s₁' s₁
          s₂ : Stream' α
          h₁ : Eq s₁' (s₁.interleave s₂).even
          ⊢ Eq (s₁.interleave s₂).even.head s₁.head
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case right
          α : Type u
          s₁✝ s₂✝ s₁' s₁ : Stream' α
          x✝ : (fun s₁' s₁ => Exists fun s₂ => Eq s₁' (s₁.interleave s₂).even) s₁' s₁
          s₂ : Stream' α
          h₁ : Eq s₁' (s₁.interleave s₂).even
          ⊢ (fun s₁' s₁ => Exists fun s₂ => Eq s₁' (s₁.interleave s₂).even) (s₁.interlea …
        -/
      · exact ⟨tail s₂, by rw [interleave_eq, even_cons_cons, tail_cons]⟩)
        /-
          🎉 no goals
        -/
    (Exists.intro s₂ rfl)


theorem interleave_even_odd (s₁ : Stream' α) : even s₁ ⋈ odd s₁ = s₁ :=
  eq_of_bisim (fun s' s => s' = even s ⋈ odd s)
    (fun s' s (h : s' = even s ⋈ odd s) => by
      /-
        α : Type u
        s₁ s' s : Stream' α
        h : Eq s' (s.even.interleave s.odd)
        ⊢ And (Eq s'.head s.head) ((fun s' s => Eq s' (s.even.interleave s.odd)) s'.ta …
      -/
      rw [h]; constructor
        /-
          case left
          α : Type u
          s₁ s' s : Stream' α
          h : Eq s' (s.even.interleave s.odd)
          ⊢ Eq (s.even.interleave s.odd).head s.head
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case right
          α : Type u
          s₁ s' s : Stream' α
          h : Eq s' (s.even.interleave s.odd)
          ⊢ (fun s' s => Eq s' (s.even.interleave s.odd)) (s.even.interleave s.odd).tail …
        -/
      · simp [odd_eq, odd_eq, tail_interleave, tail_even])
        /-
          🎉 no goals
        -/
    rfl


theorem get_even : ∀ (n : ℕ) (s : Stream' α), get (even s) n = get s (2 * n)
  | 0, _ => rfl
  | succ n, s => by
    /-
      α : Type u
      n : Nat
      s : Stream' α
      ⊢ Eq (s.even.get n.succ) (s.get (HMul.hMul 2 n.succ))
    -/
    change get (even s) (succ n) = get s (succ (succ (2 * n)))
    /-
      α : Type u
      n : Nat
      s : Stream' α
      ⊢ Eq (s.even.get n.succ) (s.get (HMul.hMul 2 n).succ.succ)
    -/
    rw [get_succ, get_succ, tail_even, get_even n]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem get_odd : ∀ (n : ℕ) (s : Stream' α), get (odd s) n = get s (2 * n + 1) := fun n s => by
  /-
    α : Type u
    n : Nat
    s : Stream' α
    ⊢ Eq (s.odd.get n) (s.get (HAdd.hAdd (HMul.hMul 2 n) 1))
  -/
  rw [odd_eq, get_even]; rfl
                         /-
                           🎉 no goals
                         -/


theorem mem_of_mem_even (a : α) (s : Stream' α) : a ∈ even s → a ∈ s := fun ⟨n, h⟩ =>
                           /-
                             α : Type u
                             a : α
                             s : Stream' α
                             x✝ : Membership.mem s.even a
                             n : Nat
                             h : (fun b => Eq a b) (s.even.get n)
                             ⊢ (fun b => Eq a b) (s.get (HMul.hMul 2 n))
                           -/
  Exists.intro (2 * n) (by rw [h, get_even])
                           /-
                             🎉 no goals
                           -/


theorem mem_of_mem_odd (a : α) (s : Stream' α) : a ∈ odd s → a ∈ s := fun ⟨n, h⟩ =>
                               /-
                                 α : Type u
                                 a : α
                                 s : Stream' α
                                 x✝ : Membership.mem s.odd a
                                 n : Nat
                                 h : (fun b => Eq a b) (s.odd.get n)
                                 ⊢ (fun b => Eq a b) (s.get (HAdd.hAdd (HMul.hMul 2 n) 1))
                               -/
  Exists.intro (2 * n + 1) (by rw [h, get_odd])
                               /-
                                 🎉 no goals
                               -/


theorem nil_append_stream (s : Stream' α) : appendStream' [] s = s :=
  rfl


theorem cons_append_stream (a : α) (l : List α) (s : Stream' α) :
    appendStream' (a::l) s = a::appendStream' l s :=
  rfl


theorem append_append_stream : ∀ (l₁ l₂ : List α) (s : Stream' α),
    l₁ ++ l₂ ++ₛ s = l₁ ++ₛ (l₂ ++ₛ s)
  | [], _, _ => rfl
  | List.cons a l₁, l₂, s => by
    /-
      α : Type u
      a : α
      l₁ l₂ : List α
      s : Stream' α
      ⊢ Eq (Stream'.appendStream' (HAppend.hAppend (List.cons a l₁) l₂) s) (Stream'. …
    -/
    rw [List.cons_append, cons_append_stream, cons_append_stream, append_append_stream l₁]
    /-
      🎉 no goals
    -/


theorem map_append_stream (f : α → β) :
    ∀ (l : List α) (s : Stream' α), map f (l ++ₛ s) = List.map f l ++ₛ map f s
  | [], _ => rfl
  | List.cons a l, s => by
    /-
      α : Type u
      β : Type v
      f : α → β
      a : α
      l : List α
      s : Stream' α
      ⊢ Eq (Stream'.map f (Stream'.appendStream' (List.cons a l) s)) (Stream'.append …
    -/
    rw [cons_append_stream, List.map_cons, map_cons, cons_append_stream, map_append_stream f l]
    /-
      🎉 no goals
    -/


theorem drop_append_stream : ∀ (l : List α) (s : Stream' α), drop l.length (l ++ₛ s) = s
                /-
                  α : Type u
                  s : Stream' α
                  ⊢ Eq (Stream'.drop List.nil.length (Stream'.appendStream' List.nil s)) s
                -/
  | [], s => by rfl
                /-
                  🎉 no goals
                -/
  | List.cons a l, s => by
    /-
      α : Type u
      a : α
      l : List α
      s : Stream' α
      ⊢ Eq (Stream'.drop (List.cons a l).length (Stream'.appendStream' (List.cons a  …
    -/
    rw [List.length_cons, drop_succ, cons_append_stream, tail_cons, drop_append_stream l s]
    /-
      🎉 no goals
    -/


theorem append_stream_head_tail (s : Stream' α) : [head s] ++ₛ tail s = s := by
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq (Stream'.appendStream' (List.cons s.head List.nil) s.tail) s
  -/
  rw [cons_append_stream, nil_append_stream, Stream'.eta]
  /-
    🎉 no goals
  -/


theorem mem_append_stream_right : ∀ {a : α} (l : List α) {s : Stream' α}, a ∈ s → a ∈ l ++ₛ s
  | _, [], _, h => h
  | a, List.cons _ l, s, h =>
    have ih : a ∈ l ++ₛ s := mem_append_stream_right l h
    mem_cons_of_mem _ ih


theorem mem_append_stream_left : ∀ {a : α} {l : List α} (s : Stream' α), a ∈ l → a ∈ l ++ₛ s
  | _, [], _, h => absurd h (List.not_mem_nil _)
  | a, List.cons b l, s, h =>
    Or.elim (List.eq_or_mem_of_mem_cons h) (fun aeqb : a = b => Exists.intro 0 aeqb)
      fun ainl : a ∈ l => mem_cons_of_mem b (mem_append_stream_left s ainl)


@[simp]
theorem take_zero (s : Stream' α) : take 0 s = [] :=
  rfl

-- This lemma used to be simp, but we removed it from the simp set because:
-- 1) It duplicates the (often large) `s` term, resulting in large tactic states.
-- 2) It conflicts with the very useful `dropLast_take` lemma below (causing nonconfluence).

theorem take_succ (n : ℕ) (s : Stream' α) : take (succ n) s = head s::take n (tail s) :=
  rfl


@[simp] theorem take_succ_cons {a : α} (n : ℕ) (s : Stream' α) :
    take (n+1) (a::s) = a :: take n s := rfl


theorem take_succ' {s : Stream' α} : ∀ n, s.take (n+1) = s.take n ++ [s.get n]
  | 0 => rfl
              /-
                α : Type u
                s : Stream' α
                n : Nat
                ⊢ Eq (Stream'.take (HAdd.hAdd (HAdd.hAdd n 1) 1) s) (HAppend.hAppend (Stream'. …
              -/
  | n+1 => by rw [take_succ, take_succ' n, ← List.cons_append, ← take_succ, get_tail]
              /-
                🎉 no goals
              -/


@[simp]
theorem length_take (n : ℕ) (s : Stream' α) : (take n s).length = n := by
  /-
    α : Type u
    n : Nat
    s : Stream' α
    ⊢ Eq (Stream'.take n s).length n
  -/
                                 /-
                                   🎉 no goals
                                 -/
  induction n generalizing s <;> simp [*, take_succ]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem take_take {s : Stream' α} : ∀ {m n}, (s.take n).take m = s.take (min n m)
               /-
                 α : Type u
                 s : Stream' α
                 n : Nat
                 ⊢ Eq (List.take 0 (Stream'.take n s)) (Stream'.take (Min.min n 0) s)
               -/
  | 0, n => by rw [Nat.min_zero, List.take_zero, take_zero]
               /-
                 🎉 no goals
               -/
               /-
                 α : Type u
                 s : Stream' α
                 m : Nat
                 ⊢ Eq (List.take m (Stream'.take 0 s)) (Stream'.take (Min.min 0 m) s)
               -/
  | m, 0 => by rw [Nat.zero_min, take_zero, List.take_nil]
               /-
                 🎉 no goals
               -/
                   /-
                     α : Type u
                     s : Stream' α
                     m n : Nat
                     ⊢ Eq (List.take (HAdd.hAdd m 1) (Stream'.take (HAdd.hAdd n 1) s)) (Stream'.tak …
                   -/
  | m+1, n+1 => by rw [take_succ, List.take_succ_cons, Nat.succ_min_succ, take_succ, take_take]
                   /-
                     🎉 no goals
                   -/


@[simp] theorem concat_take_get {n : ℕ} {s : Stream' α} : s.take n ++ [s.get n] = s.take (n+1) :=
  (take_succ' n).symm


theorem get?_take {s : Stream' α} : ∀ {k n}, k < n → (s.take n).get? k = s.get k
  | 0, _+1, _ => rfl
                      /-
                        α : Type u
                        s : Stream' α
                        k n : Nat
                        h : LT.lt (HAdd.hAdd k 1) (HAdd.hAdd n 1)
                        ⊢ Eq ((Stream'.take (HAdd.hAdd n 1) s).get? (HAdd.hAdd k 1)) (Option.some (s.g …
                      -/
  | k+1, n+1, h => by rw [take_succ, List.get?, get?_take (Nat.lt_of_succ_lt_succ h), get_succ]
                      /-
                        🎉 no goals
                      -/


theorem get?_take_succ (n : ℕ) (s : Stream' α) :
    List.get? (take (succ n) s) n = some (get s n) :=
  get?_take (Nat.lt_succ_self n)


@[simp] theorem dropLast_take {n : ℕ} {xs : Stream' α} :
    (Stream'.take n xs).dropLast = Stream'.take (n-1) xs := by
  cases n with
  | zero => simp
  | succ n => rw [take_succ', List.dropLast_concat, Nat.add_one_sub_one]


@[simp]
theorem append_take_drop : ∀ (n : ℕ) (s : Stream' α),
    appendStream' (take n s) (drop n s) = s := by
  /-
    α : Type u
    ⊢ ∀ (n : Nat) (s : Stream' α), Eq (Stream'.appendStream' (Stream'.take n s) (S …
  -/
  intro n
  /-
    α : Type u
    n : Nat
    ⊢ ∀ (s : Stream' α), Eq (Stream'.appendStream' (Stream'.take n s) (Stream'.dro …
  -/
  induction' n with n' ih
    /-
      case zero
      α : Type u
      ⊢ ∀ (s : Stream' α), Eq (Stream'.appendStream' (Stream'.take 0 s) (Stream'.dro …
    -/
  · intro s
    /-
      case zero
      α : Type u
      s : Stream' α
      ⊢ Eq (Stream'.appendStream' (Stream'.take 0 s) (Stream'.drop 0 s)) s
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq (Stream'.appendStream' (Stream'.take n' s) (Stream' …
      ⊢ ∀ (s : Stream' α), Eq (Stream'.appendStream' (Stream'.take (HAdd.hAdd n' 1)  …
    -/
  · intro s
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq (Stream'.appendStream' (Stream'.take n' s) (Stream' …
      s : Stream' α
      ⊢ Eq (Stream'.appendStream' (Stream'.take (HAdd.hAdd n' 1) s) (Stream'.drop (H …
    -/
    rw [take_succ, drop_succ, cons_append_stream, ih (tail s), Stream'.eta]
    /-
      🎉 no goals
    -/

-- Take theorem reduces a proof of equality of infinite streams to an
-- induction over all their finite approximations.

theorem take_theorem (s₁ s₂ : Stream' α) : (∀ n : ℕ, take n s₁ = take n s₂) → s₁ = s₂ := by
  /-
    α : Type u
    s₁ s₂ : Stream' α
    ⊢ (∀ (n : Nat), Eq (Stream'.take n s₁) (Stream'.take n s₂)) → Eq s₁ s₂
  -/
  intro h; apply Stream'.ext; intro n
  /-
    case a
    α : Type u
    s₁ s₂ : Stream' α
    h : ∀ (n : Nat), Eq (Stream'.take n s₁) (Stream'.take n s₂)
    n : Nat
    ⊢ Eq (s₁.get n) (s₂.get n)
  -/
  induction' n with n _
    /-
      case a.zero
      α : Type u
      s₁ s₂ : Stream' α
      h : ∀ (n : Nat), Eq (Stream'.take n s₁) (Stream'.take n s₂)
      ⊢ Eq (s₁.get 0) (s₂.get 0)
    -/
  · have aux := h 1
    simp? [take] at aux says
      simp only [take, List.cons.injEq, and_true] at aux
    /-
      case a.zero
      α : Type u
      s₁ s₂ : Stream' α
      h : ∀ (n : Nat), Eq (Stream'.take n s₁) (Stream'.take n s₂)
      aux : Eq s₁.head s₂.head
      ⊢ Eq (s₁.get 0) (s₂.get 0)
    -/
    exact aux
    /-
      🎉 no goals
    -/
  · have h₁ : some (get s₁ (succ n)) = some (get s₂ (succ n)) := by
      rw [← get?_take_succ, ← get?_take_succ, h (succ (succ n))]
    /-
      case a.succ
      α : Type u
      s₁ s₂ : Stream' α
      h : ∀ (n : Nat), Eq (Stream'.take n s₁) (Stream'.take n s₂)
      n : Nat
      a✝ : Eq (s₁.get n) (s₂.get n)
      h₁ : Eq (Option.some (s₁.get n.succ)) (Option.some (s₂.get n.succ))
      ⊢ Eq (s₁.get (HAdd.hAdd n 1)) (s₂.get (HAdd.hAdd n 1))
    -/
    injection h₁
    /-
      🎉 no goals
    -/


protected theorem cycle_g_cons (a : α) (a₁ : α) (l₁ : List α) (a₀ : α) (l₀ : List α) :
    Stream'.cycleG (a, a₁::l₁, a₀, l₀) = (a₁, l₁, a₀, l₀) :=
  rfl


theorem cycle_eq : ∀ (l : List α) (h : l ≠ []), cycle l h = l ++ₛ cycle l h
  | [], h => absurd rfl h
  | List.cons a l, _ =>
    have gen : ∀ l' a', corec Stream'.cycleF Stream'.cycleG (a', l', a, l) =
        (a'::l') ++ₛ corec Stream'.cycleF Stream'.cycleG (a, l, a, l) := by
      /-
        α : Type u
        a : α
        l : List α
        x✝ : Ne (List.cons a l) List.nil
        ⊢ ∀ (l' : List α) (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG {  …
      -/
      intro l'
      /-
        α : Type u
        a : α
        l : List α
        x✝ : Ne (List.cons a l) List.nil
        l' : List α
        ⊢ ∀ (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a', snd …
      -/
      induction' l' with a₁ l₁ ih
        /-
          case nil
          α : Type u
          a : α
          l : List α
          x✝ : Ne (List.cons a l) List.nil
          ⊢ ∀ (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a', snd …
        -/
      · intros
        /-
          case nil
          α : Type u
          a : α
          l : List α
          x✝ : Ne (List.cons a l) List.nil
          a'✝ : α
          ⊢ Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a'✝, snd := { fst : …
        -/
        rw [corec_eq]
        /-
          case nil
          α : Type u
          a : α
          l : List α
          x✝ : Ne (List.cons a l) List.nil
          a'✝ : α
          ⊢ Eq (Stream'.cons (Stream'.cycleF { fst := a'✝, snd := { fst := List.nil, snd …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case cons
          α : Type u
          a : α
          l : List α
          x✝ : Ne (List.cons a l) List.nil
          a₁ : α
          l₁ : List α
          ih : ∀ (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a',  …
          ⊢ ∀ (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a', snd …
        -/
      · intros
        /-
          case cons
          α : Type u
          a : α
          l : List α
          x✝ : Ne (List.cons a l) List.nil
          a₁ : α
          l₁ : List α
          ih : ∀ (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a',  …
          a'✝ : α
          ⊢ Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a'✝, snd := { fst : …
        -/
        rw [corec_eq, Stream'.cycle_g_cons, ih a₁]
        /-
          case cons
          α : Type u
          a : α
          l : List α
          x✝ : Ne (List.cons a l) List.nil
          a₁ : α
          l₁ : List α
          ih : ∀ (a' : α), Eq (Stream'.corec Stream'.cycleF Stream'.cycleG { fst := a',  …
          a'✝ : α
          ⊢ Eq (Stream'.cons (Stream'.cycleF { fst := a'✝, snd := { fst := List.cons a₁  …
        -/
        rfl
        /-
          🎉 no goals
        -/
    gen l a


theorem mem_cycle {a : α} {l : List α} : ∀ h : l ≠ [], a ∈ l → a ∈ cycle l h := fun h ainl => by
  /-
    α : Type u
    a : α
    l : List α
    h : Ne l List.nil
    ainl : Membership.mem l a
    ⊢ Membership.mem (Stream'.cycle l h) a
  -/
  rw [cycle_eq]; exact mem_append_stream_left _ ainl
                 /-
                   🎉 no goals
                 -/


@[simp]
                                                /-
                                                  α : Type u
                                                  β : Type v
                                                  δ : Type w
                                                  a : α
                                                  ⊢ Ne (List.cons a List.nil) List.nil
                                                -/
theorem cycle_singleton (a : α) : cycle [a] (by simp) = const a :=
                                                /-
                                                  🎉 no goals
                                                -/
                                    /-
                                      α : Type u
                                      a : α
                                      β : Type u
                                      fr : Stream' α → β
                                      ch : Eq (fr (Stream'.cycle (List.cons a List.nil) ⋯)) (fr (Stream'.const a))
                                      ⊢ Eq (fr (Stream'.cycle (List.cons a List.nil) ⋯).tail) (fr (Stream'.const a). …
                                    -/
  coinduction rfl fun β fr ch => by rwa [cycle_eq, const_eq]
                                    /-
                                      🎉 no goals
                                    -/


theorem tails_eq (s : Stream' α) : tails s = tail s::tails (tail s) := by
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq s.tails (Stream'.cons s.tail s.tail.tails)
  -/
  unfold tails; rw [corec_eq]; rfl
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem get_tails : ∀ (n : ℕ) (s : Stream' α), get (tails s) n = drop n (tail s) := by
  /-
    α : Type u
    ⊢ ∀ (n : Nat) (s : Stream' α), Eq (s.tails.get n) (Stream'.drop n s.tail)
  -/
  intro n; induction' n with n' ih
    /-
      case zero
      α : Type u
      ⊢ ∀ (s : Stream' α), Eq (s.tails.get 0) (Stream'.drop 0 s.tail)
    -/
  · intros
    /-
      case zero
      α : Type u
      s✝ : Stream' α
      ⊢ Eq (s✝.tails.get 0) (Stream'.drop 0 s✝.tail)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq (s.tails.get n') (Stream'.drop n' s.tail)
      ⊢ ∀ (s : Stream' α), Eq (s.tails.get (HAdd.hAdd n' 1)) (Stream'.drop (HAdd.hAd …
    -/
  · intro s
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq (s.tails.get n') (Stream'.drop n' s.tail)
      s : Stream' α
      ⊢ Eq (s.tails.get (HAdd.hAdd n' 1)) (Stream'.drop (HAdd.hAdd n' 1) s.tail)
    -/
    rw [get_succ, drop_succ, tails_eq, tail_cons, ih]
    /-
      🎉 no goals
    -/


theorem tails_eq_iterate (s : Stream' α) : tails s = iterate tail (tail s) :=
  rfl


theorem inits_core_eq (l : List α) (s : Stream' α) :
    initsCore l s = l::initsCore (l ++ [head s]) (tail s) := by
    /-
      α : Type u
      l : List α
      s : Stream' α
      ⊢ Eq (Stream'.initsCore l s) (Stream'.cons l (Stream'.initsCore (HAppend.hAppe …
    -/
    unfold initsCore corecOn
    /-
      α : Type u
      l : List α
      s : Stream' α
      ⊢ Eq (Stream'.corec (fun x => Stream'.initsCore.match_1 (fun x => List α) x fu …
    -/
    rw [corec_eq]
    /-
      🎉 no goals
    -/


theorem tail_inits (s : Stream' α) :
    tail (inits s) = initsCore [head s, head (tail s)] (tail (tail s)) := by
    /-
      α : Type u
      s : Stream' α
      ⊢ Eq s.inits.tail (Stream'.initsCore (List.cons s.head (List.cons s.tail.head  …
    -/
    unfold inits
    /-
      α : Type u
      s : Stream' α
      ⊢ Eq (Stream'.initsCore (List.cons s.head List.nil) s.tail).tail (Stream'.init …
    -/
    rw [inits_core_eq]; rfl
                        /-
                          🎉 no goals
                        -/


theorem inits_tail (s : Stream' α) : inits (tail s) = initsCore [head (tail s)] (tail (tail s)) :=
  rfl


theorem cons_get_inits_core :
    ∀ (a : α) (n : ℕ) (l : List α) (s : Stream' α),
      (a::get (initsCore l s) n) = get (initsCore (a::l) s) n := by
  /-
    α : Type u
    ⊢ ∀ (a : α) (n : Nat) (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'. …
  -/
  intro a n
  /-
    α : Type u
    a : α
    n : Nat
    ⊢ ∀ (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'.initsCore l s).get …
  -/
  induction' n with n' ih
    /-
      case zero
      α : Type u
      a : α
      ⊢ ∀ (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'.initsCore l s).get …
    -/
  · intros
    /-
      case zero
      α : Type u
      a : α
      l✝ : List α
      s✝ : Stream' α
      ⊢ Eq (List.cons a ((Stream'.initsCore l✝ s✝).get 0)) ((Stream'.initsCore (List …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      a : α
      n' : Nat
      ih : ∀ (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'.initsCore l s). …
      ⊢ ∀ (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'.initsCore l s).get …
    -/
  · intro l s
    /-
      case succ
      α : Type u
      a : α
      n' : Nat
      ih : ∀ (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'.initsCore l s). …
      l : List α
      s : Stream' α
      ⊢ Eq (List.cons a ((Stream'.initsCore l s).get (HAdd.hAdd n' 1))) ((Stream'.in …
    -/
    rw [get_succ, inits_core_eq, tail_cons, ih, inits_core_eq (a::l) s]
    /-
      case succ
      α : Type u
      a : α
      n' : Nat
      ih : ∀ (l : List α) (s : Stream' α), Eq (List.cons a ((Stream'.initsCore l s). …
      l : List α
      s : Stream' α
      ⊢ Eq ((Stream'.initsCore (List.cons a (HAppend.hAppend l (List.cons s.head Lis …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem get_inits : ∀ (n : ℕ) (s : Stream' α), get (inits s) n = take (succ n) s := by
  /-
    α : Type u
    ⊢ ∀ (n : Nat) (s : Stream' α), Eq (s.inits.get n) (Stream'.take n.succ s)
  -/
  intro n; induction' n with n' ih
    /-
      case zero
      α : Type u
      ⊢ ∀ (s : Stream' α), Eq (s.inits.get 0) (Stream'.take (Nat.succ 0) s)
    -/
  · intros
    /-
      case zero
      α : Type u
      s✝ : Stream' α
      ⊢ Eq (s✝.inits.get 0) (Stream'.take (Nat.succ 0) s✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq (s.inits.get n') (Stream'.take n'.succ s)
      ⊢ ∀ (s : Stream' α), Eq (s.inits.get (HAdd.hAdd n' 1)) (Stream'.take (HAdd.hAd …
    -/
  · intros
    /-
      case succ
      α : Type u
      n' : Nat
      ih : ∀ (s : Stream' α), Eq (s.inits.get n') (Stream'.take n'.succ s)
      s✝ : Stream' α
      ⊢ Eq (s✝.inits.get (HAdd.hAdd n' 1)) (Stream'.take (HAdd.hAdd n' 1).succ s✝)
    -/
    rw [get_succ, take_succ, ← ih, tail_inits, inits_tail, cons_get_inits_core]
    /-
      🎉 no goals
    -/


theorem inits_eq (s : Stream' α) :
    inits s = [head s]::map (List.cons (head s)) (inits (tail s)) := by
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq s.inits (Stream'.cons (List.cons s.head List.nil) (Stream'.map (List.cons …
  -/
  apply Stream'.ext; intro n
  /-
    case a
    α : Type u
    s : Stream' α
    n : Nat
    ⊢ Eq (s.inits.get n) ((Stream'.cons (List.cons s.head List.nil) (Stream'.map ( …
  -/
  cases n
    /-
      case a.zero
      α : Type u
      s : Stream' α
      ⊢ Eq (s.inits.get 0) ((Stream'.cons (List.cons s.head List.nil) (Stream'.map ( …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case a.succ
      α : Type u
      s : Stream' α
      n✝ : Nat
      ⊢ Eq (s.inits.get (HAdd.hAdd n✝ 1)) ((Stream'.cons (List.cons s.head List.nil) …
    -/
  · rw [get_inits, get_succ, tail_cons, get_map, get_inits]
    /-
      case a.succ
      α : Type u
      s : Stream' α
      n✝ : Nat
      ⊢ Eq (Stream'.take (HAdd.hAdd n✝ 1).succ s) (List.cons s.head (Stream'.take n✝ …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem zip_inits_tails (s : Stream' α) : zip appendStream' (inits s) (tails s) = const s := by
  /-
    α : Type u
    s : Stream' α
    ⊢ Eq (Stream'.zip Stream'.appendStream' s.inits s.tails) (Stream'.const s)
  -/
  apply Stream'.ext; intro n
  rw [get_zip, get_inits, get_tails, get_const, take_succ, cons_append_stream, append_take_drop,
    Stream'.eta]


theorem identity (s : Stream' α) : pure id ⊛ s = s :=
  rfl


theorem composition (g : Stream' (β → δ)) (f : Stream' (α → β)) (s : Stream' α) :
    pure comp ⊛ g ⊛ f ⊛ s = g ⊛ (f ⊛ s) :=
  rfl


theorem homomorphism (f : α → β) (a : α) : pure f ⊛ pure a = pure (f a) :=
  rfl


theorem interchange (fs : Stream' (α → β)) (a : α) :
    fs ⊛ pure a = (pure fun f : α → β => f a) ⊛ fs :=
  rfl


theorem map_eq_apply (f : α → β) (s : Stream' α) : map f s = pure f ⊛ s :=
  rfl


theorem get_nats (n : ℕ) : get nats n = n :=
  rfl


theorem nats_eq : nats = cons 0 (map succ nats) := by
  /-
    ⊢ Eq Stream'.nats (Stream'.cons 0 (Stream'.map Nat.succ Stream'.nats))
  -/
  apply Stream'.ext; intro n
  /-
    case a
    n : Nat
    ⊢ Eq (Stream'.nats.get n) ((Stream'.cons 0 (Stream'.map Nat.succ Stream'.nats) …
  -/
  cases n
    /-
      case a.zero
      ⊢ Eq (Stream'.nats.get 0) ((Stream'.cons 0 (Stream'.map Nat.succ Stream'.nats) …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case a.succ
    n✝ : Nat
    ⊢ Eq (Stream'.nats.get (HAdd.hAdd n✝ 1)) ((Stream'.cons 0 (Stream'.map Nat.suc …
  -/
  rw [get_succ]; rfl
                 /-
                   🎉 no goals
                 -/


