/-- Auxiliary definition for `argmax` and `argmin`. -/
def argAux (a : Option α) (b : α) : Option α :=
  Option.casesOn a (some b) fun c => if r b c then some b else some c


@[simp]
theorem foldl_argAux_eq_none : l.foldl (argAux r) o = none ↔ l = [] ∧ o = none :=
                          /-
                            α : Type u_1
                            r : α → α → Prop
                            inst✝ : DecidableRel r
                            l : List α
                            o : Option α
                            ⊢ Iff (Eq (List.foldl (List.argAux r) o List.nil) Option.none) (And (Eq List.n …
                          -/
  List.reverseRecOn l (by simp) fun tl hd => by
                          /-
                            🎉 no goals
                          -/
    simp only [foldl_append, foldl_cons, argAux, foldl_nil, append_eq_nil, and_false, false_and,
      iff_false]
    /-
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      o : Option α
      tl : List α
      hd : α
      ⊢ Iff (Eq (List.foldl (List.argAux r) o tl) Option.none) (And (Eq tl List.nil) …
    -/
    cases foldl (argAux r) o tl
      /-
        case none
        α : Type u_1
        r : α → α → Prop
        inst✝ : DecidableRel r
        l : List α
        o : Option α
        tl : List α
        hd : α
        ⊢ Iff (Eq Option.none Option.none) (And (Eq tl List.nil) (Eq o Option.none)) → …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case some
        α : Type u_1
        r : α → α → Prop
        inst✝ : DecidableRel r
        l : List α
        o : Option α
        tl : List α
        hd val✝ : α
        ⊢ Iff (Eq (Option.some val✝) Option.none) (And (Eq tl List.nil) (Eq o Option.n …
      -/
    · simp only [false_iff, not_and]
      /-
        case some
        α : Type u_1
        r : α → α → Prop
        inst✝ : DecidableRel r
        l : List α
        o : Option α
        tl : List α
        hd val✝ : α
        ⊢ Iff (Eq (Option.some val✝) Option.none) (And (Eq tl List.nil) (Eq o Option.n …
      -/
                    /-
                      🎉 no goals
                    -/
      split_ifs <;> simp
                    /-
                      🎉 no goals
                    -/


private theorem foldl_argAux_mem (l) : ∀ a m : α, m ∈ foldl (argAux r) (some a) l → m ∈ a :: l :=
                          /-
                            α : Type u_1
                            r : α → α → Prop
                            inst✝ : DecidableRel r
                            l : List α
                            ⊢ ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) List …
                          -/
  List.reverseRecOn l (by simp [eq_comm])
                          /-
                            🎉 no goals
                          -/
    (by
      /-
        α : Type u_1
        r : α → α → Prop
        inst✝ : DecidableRel r
        l : List α
        ⊢ ∀ (l : List α) (a : α), (∀ (a m : α), Membership.mem (List.foldl (List.argAu …
      -/
      intro tl hd ih a m
      /-
        α : Type u_1
        r : α → α → Prop
        inst✝ : DecidableRel r
        l tl : List α
        hd : α
        ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
        a m : α
        ⊢ Membership.mem (List.foldl (List.argAux r) (Option.some a) (HAppend.hAppend  …
      -/
      simp only [foldl_append, foldl_cons, foldl_nil, argAux]
      /-
        α : Type u_1
        r : α → α → Prop
        inst✝ : DecidableRel r
        l tl : List α
        hd : α
        ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
        a m : α
        ⊢ Membership.mem (Option.rec (Option.some hd) (fun val => ite (r hd val) (Opti …
      -/
      cases hf : foldl (argAux r) (some a) tl
        /-
          case none
          α : Type u_1
          r : α → α → Prop
          inst✝ : DecidableRel r
          l tl : List α
          hd : α
          ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
          a m : α
          hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) Option.none
          ⊢ Membership.mem (Option.rec (Option.some hd) (fun val => ite (r hd val) (Opti …
        -/
      · simp +contextual
        /-
          🎉 no goals
        -/
        /-
          case some
          α : Type u_1
          r : α → α → Prop
          inst✝ : DecidableRel r
          l tl : List α
          hd : α
          ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
          a m val✝ : α
          hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) (Option.some val✝)
          ⊢ Membership.mem (Option.rec (Option.some hd) (fun val => ite (r hd val) (Opti …
        -/
      · dsimp only
        /-
          case some
          α : Type u_1
          r : α → α → Prop
          inst✝ : DecidableRel r
          l tl : List α
          hd : α
          ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
          a m val✝ : α
          hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) (Option.some val✝)
          ⊢ Membership.mem (ite (r hd val✝) (Option.some hd) (Option.some val✝)) m → Mem …
        -/
        split_ifs
          /-
            case pos
            α : Type u_1
            r : α → α → Prop
            inst✝ : DecidableRel r
            l tl : List α
            hd : α
            ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
            a m val✝ : α
            hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) (Option.some val✝)
            h✝ : r hd val✝
            ⊢ Membership.mem (Option.some hd) m → Membership.mem (List.cons a (HAppend.hAp …
          -/
        · simp +contextual
          /-
            🎉 no goals
          -/
        · -- `finish [ih _ _ hf]` closes this goal
          /-
            case neg
            α : Type u_1
            r : α → α → Prop
            inst✝ : DecidableRel r
            l tl : List α
            hd : α
            ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
            a m val✝ : α
            hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) (Option.some val✝)
            h✝ : Not (r hd val✝)
            ⊢ Membership.mem (Option.some val✝) m → Membership.mem (List.cons a (HAppend.h …
          -/
          simp only [List.mem_cons] at ih
          /-
            case neg
            α : Type u_1
            r : α → α → Prop
            inst✝ : DecidableRel r
            l tl : List α
            hd a m val✝ : α
            hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) (Option.some val✝)
            h✝ : Not (r hd val✝)
            ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
            ⊢ Membership.mem (Option.some val✝) m → Membership.mem (List.cons a (HAppend.h …
          -/
          rcases ih _ _ hf with rfl | H
          · simp +contextual only [Option.mem_def, Option.some.injEq,
              find?, eq_comm, mem_cons, mem_append, mem_singleton, true_or, implies_true]
            /-
              case neg.inr
              α : Type u_1
              r : α → α → Prop
              inst✝ : DecidableRel r
              l tl : List α
              hd a m val✝ : α
              hf : Eq (List.foldl (List.argAux r) (Option.some a) tl) (Option.some val✝)
              h✝ : Not (r hd val✝)
              ih : ∀ (a m : α), Membership.mem (List.foldl (List.argAux r) (Option.some a) t …
              H : Membership.mem tl val✝
              ⊢ Membership.mem (Option.some val✝) m → Membership.mem (List.cons a (HAppend.h …
            -/
          · simp +contextual [@eq_comm _ _ m, H])
            /-
              🎉 no goals
            -/


@[simp]
theorem argAux_self (hr₀ : Irreflexive r) (a : α) : argAux r (some a) a = a :=
  if_neg <| hr₀ _


theorem not_of_mem_foldl_argAux (hr₀ : Irreflexive r) (hr₁ : Transitive r) :
    ∀ {a m : α} {o : Option α}, a ∈ l → m ∈ foldl (argAux r) o l → ¬r a m := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    ⊢ ∀ {a m : α} {o : Option α}, Membership.mem l a → Membership.mem (List.foldl  …
  -/
  induction' l using List.reverseRecOn with tl a ih
    /-
      case nil
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      ⊢ ∀ {a m : α} {o : Option α}, Membership.mem List.nil a → Membership.mem (List …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case append_singleton
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    tl : List α
    a : α
    ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
    ⊢ ∀ {a_1 m : α} {o : Option α}, Membership.mem (HAppend.hAppend tl (List.cons  …
  -/
  intro b m o hb ho
  /-
    case append_singleton
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    tl : List α
    a : α
    ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
    b m : α
    o : Option α
    hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
    ho : Membership.mem (List.foldl (List.argAux r) o (HAppend.hAppend tl (List.co …
    ⊢ Not (r b m)
  -/
  rw [foldl_append, foldl_cons, foldl_nil, argAux] at ho
  /-
    case append_singleton
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    tl : List α
    a : α
    ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
    b m : α
    o : Option α
    hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
    ho : Membership.mem (Option.casesOn (List.foldl (List.argAux r) o tl) (Option. …
    ⊢ Not (r b m)
  -/
  cases' hf : foldl (argAux r) o tl with c
    /-
      case append_singleton.none
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b m : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      ho : Membership.mem (Option.casesOn (List.foldl (List.argAux r) o tl) (Option. …
      hf : Eq (List.foldl (List.argAux r) o tl) Option.none
      ⊢ Not (r b m)
    -/
  · rw [hf] at ho
    /-
      case append_singleton.none
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b m : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      ho : Membership.mem (Option.casesOn Option.none (Option.some a) fun c => ite ( …
      hf : Eq (List.foldl (List.argAux r) o tl) Option.none
      ⊢ Not (r b m)
    -/
    rw [foldl_argAux_eq_none] at hf
    /-
      case append_singleton.none
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b m : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      ho : Membership.mem (Option.casesOn Option.none (Option.some a) fun c => ite ( …
      hf : And (Eq tl List.nil) (Eq o Option.none)
      ⊢ Not (r b m)
    -/
    simp_all [hf.1, hf.2, hr₀ _]
    /-
      🎉 no goals
    -/
  /-
    case append_singleton.some
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    tl : List α
    a : α
    ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
    b m : α
    o : Option α
    hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
    ho : Membership.mem (Option.casesOn (List.foldl (List.argAux r) o tl) (Option. …
    c : α
    hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
    ⊢ Not (r b m)
  -/
  rw [hf, Option.mem_def] at ho
  /-
    case append_singleton.some
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    tl : List α
    a : α
    ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
    b m : α
    o : Option α
    hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
    c : α
    ho : Eq (Option.casesOn (Option.some c) (Option.some a) fun c => ite (r a c) ( …
    hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
    ⊢ Not (r b m)
  -/
  dsimp only at ho
  /-
    case append_singleton.some
    α : Type u_1
    r : α → α → Prop
    inst✝ : DecidableRel r
    l : List α
    hr₀ : Irreflexive r
    hr₁ : Transitive r
    tl : List α
    a : α
    ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
    b m : α
    o : Option α
    hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
    c : α
    ho : Eq (ite (r a c) (Option.some a) (Option.some c)) (Option.some m)
    hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
    ⊢ Not (r b m)
  -/
  split_ifs at ho with hac <;> cases' mem_append.1 hb with h h <;>
    /-
      case pos.inl
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b m : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      c : α
      hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
      hac : r a c
      ho : Eq (Option.some a) (Option.some m)
      h : Membership.mem tl b
      ⊢ Not (r b m)
    -/
    injection ho with ho <;> subst ho
    /-
      case pos.inl
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      c : α
      hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
      hac : r a c
      h : Membership.mem tl b
      ⊢ Not (r b a)
    -/
  · exact fun hba => ih h hf (hr₁ hba hac)
    /-
      🎉 no goals
    -/
    /-
      case pos.inr
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      c : α
      hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
      hac : r a c
      h : Membership.mem (List.cons a List.nil) b
      ⊢ Not (r b a)
    -/
  · simp_all [hr₀ _]
    /-
      🎉 no goals
    -/
    /-
      case neg.inl
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      c : α
      hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
      hac : Not (r a c)
      h : Membership.mem tl b
      ⊢ Not (r b c)
    -/
  · exact ih h hf
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      α : Type u_1
      r : α → α → Prop
      inst✝ : DecidableRel r
      l : List α
      hr₀ : Irreflexive r
      hr₁ : Transitive r
      tl : List α
      a : α
      ih : ∀ {a m : α} {o : Option α}, Membership.mem tl a → Membership.mem (List.fo …
      b : α
      o : Option α
      hb : Membership.mem (HAppend.hAppend tl (List.cons a List.nil)) b
      c : α
      hf : Eq (List.foldl (List.argAux r) o tl) (Option.some c)
      hac : Not (r a c)
      h : Membership.mem (List.cons a List.nil) b
      ⊢ Not (r b c)
    -/
  · simp_all
    /-
      🎉 no goals
    -/


/-- `argmax f l` returns `some a`, where `f a` is maximal among the elements of `l`, in the sense
that there is no `b ∈ l` with `f a < f b`. If `a`, `b` are such that `f a = f b`, it returns
whichever of `a` or `b` comes first in the list. `argmax f [] = none`. -/
def argmax (f : α → β) (l : List α) : Option α :=
  l.foldl (argAux fun b c => f c < f b) none


/-- `argmin f l` returns `some a`, where `f a` is minimal among the elements of `l`, in the sense
that there is no `b ∈ l` with `f b < f a`. If `a`, `b` are such that `f a = f b`, it returns
whichever of `a` or `b` comes first in the list. `argmin f [] = none`. -/
def argmin (f : α → β) (l : List α) :=
  l.foldl (argAux fun b c => f b < f c) none


@[simp]
theorem argmax_nil (f : α → β) : argmax f [] = none :=
  rfl


@[simp]
theorem argmin_nil (f : α → β) : argmin f [] = none :=
  rfl


@[simp]
theorem argmax_singleton {f : α → β} {a : α} : argmax f [a] = a :=
  rfl


@[simp]
theorem argmin_singleton {f : α → β} {a : α} : argmin f [a] = a :=
  rfl


theorem not_lt_of_mem_argmax : a ∈ l → m ∈ argmax f l → ¬f m < f a :=
  not_of_mem_foldl_argAux _ (fun x h => lt_irrefl (f x) h)
    (fun _ _ z hxy hyz => lt_trans (a := f z) hyz hxy)


theorem not_lt_of_mem_argmin : a ∈ l → m ∈ argmin f l → ¬f a < f m :=
  not_of_mem_foldl_argAux _ (fun x h => lt_irrefl (f x) h)
    (fun x _ _ hxy hyz => lt_trans (a := f x) hxy hyz)


theorem argmax_concat (f : α → β) (a : α) (l : List α) :
    argmax f (l ++ [a]) =
      Option.casesOn (argmax f l) (some a) fun c => if f c < f a then some a else some c := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder β
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    f : α → β
    a : α
    l : List α
    ⊢ Eq (List.argmax f (HAppend.hAppend l (List.cons a List.nil))) (Option.casesO …
  -/
  rw [argmax, argmax]; simp [argAux]
                       /-
                         🎉 no goals
                       -/


theorem argmin_concat (f : α → β) (a : α) (l : List α) :
    argmin f (l ++ [a]) =
      Option.casesOn (argmin f l) (some a) fun c => if f a < f c then some a else some c :=
  @argmax_concat _ βᵒᵈ _ _ _ _ _


theorem argmax_mem : ∀ {l : List α} {m : α}, m ∈ argmax f l → m ∈ l
                /-
                  α : Type u_1
                  β : Type u_2
                  inst✝¹ : Preorder β
                  inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
                  f : α → β
                  m : α
                  ⊢ Membership.mem (List.argmax f List.nil) m → Membership.mem List.nil m
                -/
  | [], m => by simp
                /-
                  🎉 no goals
                -/
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝¹ : Preorder β
                        inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
                        f : α → β
                        hd : α
                        tl : List α
                        m : α
                        ⊢ Membership.mem (List.argmax f (List.cons hd tl)) m → Membership.mem (List.co …
                      -/
  | hd :: tl, m => by simpa [argmax, argAux] using foldl_argAux_mem _ tl hd m
                      /-
                        🎉 no goals
                      -/


theorem argmin_mem : ∀ {l : List α} {m : α}, m ∈ argmin f l → m ∈ l :=
  @argmax_mem _ βᵒᵈ _ _ _


@[simp]
                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            inst✝¹ : Preorder β
                                                            inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
                                                            f : α → β
                                                            l : List α
                                                            ⊢ Iff (Eq (List.argmax f l) Option.none) (Eq l List.nil)
                                                          -/
theorem argmax_eq_none : l.argmax f = none ↔ l = [] := by simp [argmax]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem argmin_eq_none : l.argmin f = none ↔ l = [] :=
  @argmax_eq_none _ βᵒᵈ _ _ _ _


theorem le_of_mem_argmax : a ∈ l → m ∈ argmax f l → f a ≤ f m := fun ha hm =>
  le_of_not_lt <| not_lt_of_mem_argmax ha hm


theorem le_of_mem_argmin : a ∈ l → m ∈ argmin f l → f m ≤ f a :=
  @le_of_mem_argmax _ βᵒᵈ _ _ _ _ _


theorem argmax_cons (f : α → β) (a : α) (l : List α) :
    argmax f (a :: l) =
      Option.casesOn (argmax f l) (some a) fun c => if f a < f c then some c else some a :=
  List.reverseRecOn l rfl fun hd tl ih => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : α → β
      a : α
      l hd : List α
      tl : α
      ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
      ⊢ Eq (List.argmax f (List.cons a (HAppend.hAppend hd (List.cons tl List.nil))) …
    -/
    rw [← cons_append, argmax_concat, ih, argmax_concat]
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : α → β
      a : α
      l hd : List α
      tl : α
      ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
      ⊢ Eq (Option.casesOn (Option.casesOn (List.argmax f hd) (Option.some a) fun c  …
    -/
    cases' h : argmax f hd with m
      /-
        case none
        α : Type u_1
        β : Type u_2
        inst✝ : LinearOrder β
        f : α → β
        a : α
        l hd : List α
        tl : α
        ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
        h : Eq (List.argmax f hd) Option.none
        ⊢ Eq (Option.casesOn (Option.casesOn Option.none (Option.some a) fun c => ite  …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : α → β
      a : α
      l hd : List α
      tl : α
      ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
      m : α
      h : Eq (List.argmax f hd) (Option.some m)
      ⊢ Eq (Option.casesOn (Option.casesOn (Option.some m) (Option.some a) fun c =>  …
    -/
    dsimp
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : α → β
      a : α
      l hd : List α
      tl : α
      ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
      m : α
      h : Eq (List.argmax f hd) (Option.some m)
      ⊢ Eq (Option.rec (Option.some tl) (fun val => ite (LT.lt (f val) (f tl)) (Opti …
    -/
    rw [← apply_ite, ← apply_ite]
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : α → β
      a : α
      l hd : List α
      tl : α
      ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
      m : α
      h : Eq (List.argmax f hd) (Option.some m)
      ⊢ Eq (Option.rec (Option.some tl) (fun val => ite (LT.lt (f val) (f tl)) (Opti …
    -/
    dsimp
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder β
      f : α → β
      a : α
      l hd : List α
      tl : α
      ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
      m : α
      h : Eq (List.argmax f hd) (Option.some m)
      ⊢ Eq (ite (LT.lt (f (ite (LT.lt (f a) (f m)) m a)) (f tl)) (Option.some tl) (O …
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
    split_ifs <;> try rfl
                  /-
                    🎉 no goals
                  -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝ : LinearOrder β
        f : α → β
        a : α
        l hd : List α
        tl : α
        ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
        m : α
        h : Eq (List.argmax f hd) (Option.some m)
        h✝² : LT.lt (f a) (f m)
        h✝¹ : LT.lt (f m) (f tl)
        h✝ : Not (LT.lt (f a) (f tl))
        ⊢ Eq (Option.some tl) (Option.some a)
      -/
    · exact absurd (lt_trans ‹f a < f m› ‹_›) ‹_›
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝ : LinearOrder β
        f : α → β
        a : α
        l hd : List α
        tl : α
        ih : Eq (List.argmax f (List.cons a hd)) (Option.casesOn (List.argmax f hd) (O …
        m : α
        h : Eq (List.argmax f hd) (Option.some m)
        h✝² : Not (LT.lt (f a) (f m))
        h✝¹ : LT.lt (f a) (f tl)
        h✝ : Not (LT.lt (f m) (f tl))
        ⊢ Eq (Option.some tl) (Option.some a)
      -/
    · cases (‹f a < f tl›.lt_or_lt _).elim ‹_› ‹_›
      /-
        🎉 no goals
      -/


theorem argmin_cons (f : α → β) (a : α) (l : List α) :
    argmin f (a :: l) =
      Option.casesOn (argmin f l) (some a) fun c => if f c < f a then some c else some a :=
  @argmax_cons α βᵒᵈ _ _ _ _


theorem index_of_argmax :
    ∀ {l : List α} {m : α}, m ∈ argmax f l → ∀ {a}, a ∈ l → f m ≤ f a → l.indexOf m ≤ l.indexOf a
                            /-
                              α : Type u_1
                              β : Type u_2
                              inst✝¹ : LinearOrder β
                              f : α → β
                              inst✝ : DecidableEq α
                              m : α
                              x✝³ : Membership.mem (List.argmax f List.nil) m
                              x✝² : α
                              x✝¹ : Membership.mem List.nil x✝²
                              x✝ : LE.le (f m) (f x✝²)
                              ⊢ LE.le (List.indexOf m List.nil) (List.indexOf x✝² List.nil)
                            -/
  | [], m, _, _, _, _ => by simp
                            /-
                              🎉 no goals
                            -/
  | hd :: tl, m, hm, a, ha, ham => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder β
      f : α → β
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      m : α
      hm : Membership.mem (List.argmax f (List.cons hd tl)) m
      a : α
      ha : Membership.mem (List.cons hd tl) a
      ham : LE.le (f m) (f a)
      ⊢ LE.le (List.indexOf m (List.cons hd tl)) (List.indexOf a (List.cons hd tl))
    -/
    simp only [indexOf_cons, argmax_cons, Option.mem_def] at hm ⊢
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder β
      f : α → β
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      m a : α
      ha : Membership.mem (List.cons hd tl) a
      ham : LE.le (f m) (f a)
      hm : Eq (Option.rec (Option.some hd) (fun val => ite (LT.lt (f hd) (f val)) (O …
      ⊢ LE.le (cond (BEq.beq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (cond (BEq.b …
    -/
    cases h : argmax f tl
      /-
        case none
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        m a : α
        ha : Membership.mem (List.cons hd tl) a
        ham : LE.le (f m) (f a)
        hm : Eq (Option.rec (Option.some hd) (fun val => ite (LT.lt (f hd) (f val)) (O …
        h : Eq (List.argmax f tl) Option.none
        ⊢ LE.le (cond (BEq.beq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (cond (BEq.b …
      -/
    · rw [h] at hm
      /-
        case none
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        m a : α
        ha : Membership.mem (List.cons hd tl) a
        ham : LE.le (f m) (f a)
        hm : Eq (Option.rec (Option.some hd) (fun val => ite (LT.lt (f hd) (f val)) (O …
        h : Eq (List.argmax f tl) Option.none
        ⊢ LE.le (cond (BEq.beq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (cond (BEq.b …
      -/
      simp_all
      /-
        🎉 no goals
      -/
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder β
      f : α → β
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      m a : α
      ha : Membership.mem (List.cons hd tl) a
      ham : LE.le (f m) (f a)
      hm : Eq (Option.rec (Option.some hd) (fun val => ite (LT.lt (f hd) (f val)) (O …
      val✝ : α
      h : Eq (List.argmax f tl) (Option.some val✝)
      ⊢ LE.le (cond (BEq.beq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (cond (BEq.b …
    -/
    rw [h] at hm
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder β
      f : α → β
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      m a : α
      ha : Membership.mem (List.cons hd tl) a
      ham : LE.le (f m) (f a)
      val✝ : α
      hm : Eq (Option.rec (Option.some hd) (fun val => ite (LT.lt (f hd) (f val)) (O …
      h : Eq (List.argmax f tl) (Option.some val✝)
      ⊢ LE.le (cond (BEq.beq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (cond (BEq.b …
    -/
    dsimp only at hm
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder β
      f : α → β
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      m a : α
      ha : Membership.mem (List.cons hd tl) a
      ham : LE.le (f m) (f a)
      val✝ : α
      hm : Eq (ite (LT.lt (f hd) (f val✝)) (Option.some val✝) (Option.some hd)) (Opt …
      h : Eq (List.argmax f tl) (Option.some val✝)
      ⊢ LE.le (cond (BEq.beq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (cond (BEq.b …
    -/
    simp only [cond_eq_if, beq_iff_eq]
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder β
      f : α → β
      inst✝ : DecidableEq α
      hd : α
      tl : List α
      m a : α
      ha : Membership.mem (List.cons hd tl) a
      ham : LE.le (f m) (f a)
      val✝ : α
      hm : Eq (ite (LT.lt (f hd) (f val✝)) (Option.some val✝) (Option.some hd)) (Opt …
      h : Eq (List.argmax f tl) (Option.some val✝)
      ⊢ LE.le (ite (Eq hd m) 0 (HAdd.hAdd (List.indexOf m tl) 1)) (ite (Eq hd a) 0 ( …
    -/
    obtain ha | ha := ha <;> split_ifs at hm <;> injection hm with hm <;> subst hm
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        val✝ : α
        h : Eq (List.argmax f tl) (Option.some val✝)
        h✝ : LT.lt (f hd) (f val✝)
        ham : LE.le (f val✝) (f hd)
        ⊢ LE.le (ite (Eq hd val✝) 0 (HAdd.hAdd (List.indexOf val✝ tl) 1)) (ite (Eq hd  …
      -/
    · cases not_le_of_lt ‹_› ‹_›
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        val✝ : α
        h : Eq (List.argmax f tl) (Option.some val✝)
        h✝ : Not (LT.lt (f hd) (f val✝))
        ham : LE.le (f hd) (f hd)
        ⊢ LE.le (ite (Eq hd hd) 0 (HAdd.hAdd (List.indexOf hd tl) 1)) (ite (Eq hd hd)  …
      -/
    · rw [if_pos rfl]
      /-
        🎉 no goals
      -/
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        a val✝ : α
        h : Eq (List.argmax f tl) (Option.some val✝)
        a✝ : List.Mem a tl
        h✝ : LT.lt (f hd) (f val✝)
        ham : LE.le (f val✝) (f a)
        ⊢ LE.le (ite (Eq hd val✝) 0 (HAdd.hAdd (List.indexOf val✝ tl) 1)) (ite (Eq hd  …
      -/
    · rw [if_neg, if_neg]
        /-
          case pos
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrder β
          f : α → β
          inst✝ : DecidableEq α
          hd : α
          tl : List α
          a val✝ : α
          h : Eq (List.argmax f tl) (Option.some val✝)
          a✝ : List.Mem a tl
          h✝ : LT.lt (f hd) (f val✝)
          ham : LE.le (f val✝) (f a)
          ⊢ LE.le (HAdd.hAdd (List.indexOf val✝ tl) 1) (HAdd.hAdd (List.indexOf a tl) 1)
        -/
      · exact Nat.succ_le_succ (index_of_argmax h (by assumption) ham)
        /-
          🎉 no goals
        -/
        /-
          case pos.hnc
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrder β
          f : α → β
          inst✝ : DecidableEq α
          hd : α
          tl : List α
          a val✝ : α
          h : Eq (List.argmax f tl) (Option.some val✝)
          a✝ : List.Mem a tl
          h✝ : LT.lt (f hd) (f val✝)
          ham : LE.le (f val✝) (f a)
          ⊢ Not (Eq hd a)
        -/
      · exact ne_of_apply_ne f (lt_of_lt_of_le ‹_› ‹_›).ne
        /-
          🎉 no goals
        -/
        /-
          case pos.hnc
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrder β
          f : α → β
          inst✝ : DecidableEq α
          hd : α
          tl : List α
          a val✝ : α
          h : Eq (List.argmax f tl) (Option.some val✝)
          a✝ : List.Mem a tl
          h✝ : LT.lt (f hd) (f val✝)
          ham : LE.le (f val✝) (f a)
          ⊢ Not (Eq hd val✝)
        -/
      · exact ne_of_apply_ne _ ‹f hd < f _›.ne
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        a val✝ : α
        h : Eq (List.argmax f tl) (Option.some val✝)
        a✝ : List.Mem a tl
        h✝ : Not (LT.lt (f hd) (f val✝))
        ham : LE.le (f hd) (f a)
        ⊢ LE.le (ite (Eq hd hd) 0 (HAdd.hAdd (List.indexOf hd tl) 1)) (ite (Eq hd a) 0 …
      -/
    · rw [if_pos rfl]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        inst✝ : DecidableEq α
        hd : α
        tl : List α
        a val✝ : α
        h : Eq (List.argmax f tl) (Option.some val✝)
        a✝ : List.Mem a tl
        h✝ : Not (LT.lt (f hd) (f val✝))
        ham : LE.le (f hd) (f a)
        ⊢ LE.le 0 (ite (Eq hd a) 0 (HAdd.hAdd (List.indexOf a tl) 1))
      -/
      exact Nat.zero_le _
      /-
        🎉 no goals
      -/


theorem index_of_argmin :
    ∀ {l : List α} {m : α}, m ∈ argmin f l → ∀ {a}, a ∈ l → f a ≤ f m → l.indexOf m ≤ l.indexOf a :=
  @index_of_argmax _ βᵒᵈ _ _ _


theorem mem_argmax_iff :
    m ∈ argmax f l ↔
      m ∈ l ∧ (∀ a ∈ l, f a ≤ f m) ∧ ∀ a ∈ l, f m ≤ f a → l.indexOf m ≤ l.indexOf a :=
  ⟨fun hm => ⟨argmax_mem hm, fun _ ha => le_of_mem_argmax ha hm, fun _ => index_of_argmax hm⟩,
    by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        l : List α
        m : α
        inst✝ : DecidableEq α
        ⊢ And (Membership.mem l m) (And (∀ (a : α), Membership.mem l a → LE.le (f a) ( …
      -/
      rintro ⟨hml, ham, hma⟩
      /-
        case intro.intro
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder β
        f : α → β
        l : List α
        m : α
        inst✝ : DecidableEq α
        hml : Membership.mem l m
        ham : ∀ (a : α), Membership.mem l a → LE.le (f a) (f m)
        hma : ∀ (a : α), Membership.mem l a → LE.le (f m) (f a) → LE.le (List.indexOf  …
        ⊢ Membership.mem (List.argmax f l) m
      -/
      cases' harg : argmax f l with n
        /-
          case intro.intro.none
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrder β
          f : α → β
          l : List α
          m : α
          inst✝ : DecidableEq α
          hml : Membership.mem l m
          ham : ∀ (a : α), Membership.mem l a → LE.le (f a) (f m)
          hma : ∀ (a : α), Membership.mem l a → LE.le (f m) (f a) → LE.le (List.indexOf  …
          harg : Eq (List.argmax f l) Option.none
          ⊢ Membership.mem Option.none m
        -/
      · simp_all
        /-
          🎉 no goals
        -/
      · have :=
          _root_.le_antisymm (hma n (argmax_mem harg) (le_of_mem_argmax hml harg))
            (index_of_argmax harg hml (ham _ (argmax_mem harg)))
        /-
          case intro.intro.some
          α : Type u_1
          β : Type u_2
          inst✝¹ : LinearOrder β
          f : α → β
          l : List α
          m : α
          inst✝ : DecidableEq α
          hml : Membership.mem l m
          ham : ∀ (a : α), Membership.mem l a → LE.le (f a) (f m)
          hma : ∀ (a : α), Membership.mem l a → LE.le (f m) (f a) → LE.le (List.indexOf  …
          n : α
          harg : Eq (List.argmax f l) (Option.some n)
          this : Eq (List.indexOf m l) (List.indexOf n l)
          ⊢ Membership.mem (Option.some n) m
        -/
        rw [(indexOf_inj hml (argmax_mem harg)).1 this, Option.mem_def]⟩
        /-
          🎉 no goals
        -/


theorem argmax_eq_some_iff :
    argmax f l = some m ↔
      m ∈ l ∧ (∀ a ∈ l, f a ≤ f m) ∧ ∀ a ∈ l, f m ≤ f a → l.indexOf m ≤ l.indexOf a :=
  mem_argmax_iff


theorem mem_argmin_iff :
    m ∈ argmin f l ↔
      m ∈ l ∧ (∀ a ∈ l, f m ≤ f a) ∧ ∀ a ∈ l, f a ≤ f m → l.indexOf m ≤ l.indexOf a :=
  @mem_argmax_iff _ βᵒᵈ _ _ _ _ _


theorem argmin_eq_some_iff :
    argmin f l = some m ↔
      m ∈ l ∧ (∀ a ∈ l, f m ≤ f a) ∧ ∀ a ∈ l, f a ≤ f m → l.indexOf m ≤ l.indexOf a :=
  mem_argmin_iff


/-- `maximum l` returns a `WithBot α`, the largest element of `l` for nonempty lists, and `⊥` for
`[]`  -/
def maximum (l : List α) : WithBot α :=
  argmax id l


/-- `minimum l` returns a `WithTop α`, the smallest element of `l` for nonempty lists, and `⊤` for
`[]`  -/
def minimum (l : List α) : WithTop α :=
  argmin id l


@[simp]
theorem maximum_nil : maximum ([] : List α) = ⊥ :=
  rfl


@[simp]
theorem minimum_nil : minimum ([] : List α) = ⊤ :=
  rfl


@[simp]
theorem maximum_singleton (a : α) : maximum [a] = a :=
  rfl


@[simp]
theorem minimum_singleton (a : α) : minimum [a] = a :=
  rfl


theorem maximum_mem {l : List α} {m : α} : (maximum l : WithTop α) = m → m ∈ l :=
  argmax_mem


theorem minimum_mem {l : List α} {m : α} : (minimum l : WithBot α) = m → m ∈ l :=
  argmin_mem


@[simp]
theorem maximum_eq_bot {l : List α} : l.maximum = ⊥ ↔ l = [] :=
  argmax_eq_none


@[simp, deprecated maximum_eq_bot "Don't mix Option and WithBot" (since := "2024-05-27")]
theorem maximum_eq_none {l : List α} : l.maximum = none ↔ l = [] := maximum_eq_bot


@[simp]
theorem minimum_eq_top {l : List α} : l.minimum = ⊤ ↔ l = [] :=
  argmin_eq_none


@[simp, deprecated minimum_eq_top "Don't mix Option and WithTop" (since := "2024-05-27")]
theorem minimum_eq_none {l : List α} : l.minimum = none ↔ l = [] := minimum_eq_top


theorem not_maximum_lt_of_mem : a ∈ l → (maximum l : WithBot α) = m → ¬m < a :=
  not_lt_of_mem_argmax


@[deprecated (since := "2024-12-29")] alias not_lt_maximum_of_mem := not_maximum_lt_of_mem


theorem not_lt_minimum_of_mem : a ∈ l → (minimum l : WithTop α) = m → ¬a < m :=
  not_lt_of_mem_argmin


@[deprecated (since := "2024-12-29")] alias minimum_not_lt_of_mem := not_lt_minimum_of_mem


theorem not_maximum_lt_of_mem' (ha : a ∈ l) : ¬maximum l < (a : WithBot α) := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    l : List α
    a : α
    ha : Membership.mem l a
    ⊢ Not (LT.lt l.maximum ↑a)
  -/
                          /-
                            🎉 no goals
                          -/
  cases h : l.maximum <;> simp_all [not_maximum_lt_of_mem ha]
                          /-
                            🎉 no goals
                          -/


@[deprecated (since := "2024-12-29")] alias not_lt_maximum_of_mem' := not_maximum_lt_of_mem'


theorem not_lt_minimum_of_mem' (ha : a ∈ l) : ¬(a : WithTop α) < minimum l := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : DecidableRel fun x1 x2 => LT.lt x1 x2
    l : List α
    a : α
    ha : Membership.mem l a
    ⊢ Not (LT.lt (↑a) l.minimum)
  -/
                          /-
                            🎉 no goals
                          -/
  cases h : l.minimum <;> simp_all [not_lt_minimum_of_mem ha]
                          /-
                            🎉 no goals
                          -/


theorem maximum_concat (a : α) (l : List α) : maximum (l ++ [a]) = max (maximum l) a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a : α
    l : List α
    ⊢ Eq (HAppend.hAppend l (List.cons a List.nil)).maximum (Max.max l.maximum ↑a)
  -/
  simp only [maximum, argmax_concat, id]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a : α
    l : List α
    ⊢ Eq (Option.rec (Option.some a) (fun val => ite (LT.lt val a) (Option.some a) …
  -/
  cases argmax id l
    /-
      case none
      α : Type u_1
      inst✝ : LinearOrder α
      a : α
      l : List α
      ⊢ Eq (Option.rec (Option.some a) (fun val => ite (LT.lt val a) (Option.some a) …
    -/
  · exact (max_eq_right bot_le).symm
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u_1
      inst✝ : LinearOrder α
      a : α
      l : List α
      val✝ : α
      ⊢ Eq (Option.rec (Option.some a) (fun val => ite (LT.lt val a) (Option.some a) …
    -/
  · simp [WithBot.some_eq_coe, max_def_lt, WithBot.coe_lt_coe]
    /-
      🎉 no goals
    -/


theorem le_maximum_of_mem : a ∈ l → (maximum l : WithBot α) = m → a ≤ m :=
  le_of_mem_argmax


theorem minimum_le_of_mem : a ∈ l → (minimum l : WithTop α) = m → m ≤ a :=
  le_of_mem_argmin


theorem le_maximum_of_mem' (ha : a ∈ l) : (a : WithBot α) ≤ maximum l :=
  le_of_not_lt <| not_maximum_lt_of_mem' ha


theorem minimum_le_of_mem' (ha : a ∈ l) : minimum l ≤ (a : WithTop α) :=
  le_of_not_lt <| not_lt_minimum_of_mem' ha


theorem minimum_concat (a : α) (l : List α) : minimum (l ++ [a]) = min (minimum l) a :=
  @maximum_concat αᵒᵈ _ _ _


theorem maximum_cons (a : α) (l : List α) : maximum (a :: l) = max ↑a (maximum l) :=
                          /-
                            α : Type u_1
                            inst✝ : LinearOrder α
                            a : α
                            l : List α
                            ⊢ Eq (List.cons a List.nil).maximum (Max.max (↑a) List.nil.maximum)
                          -/
  List.reverseRecOn l (by simp [@max_eq_left (WithBot α) _ _ _ bot_le]) fun tl hd ih => by
                          /-
                            🎉 no goals
                          -/
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a : α
      l tl : List α
      hd : α
      ih : Eq (List.cons a tl).maximum (Max.max (↑a) tl.maximum)
      ⊢ Eq (List.cons a (HAppend.hAppend tl (List.cons hd List.nil))).maximum (Max.m …
    -/
    rw [← cons_append, maximum_concat, ih, maximum_concat, max_assoc]
    /-
      🎉 no goals
    -/


theorem minimum_cons (a : α) (l : List α) : minimum (a :: l) = min ↑a (minimum l) :=
  @maximum_cons αᵒᵈ _ _ _


theorem maximum_le_of_forall_le {b : WithBot α} (h : ∀ a ∈ l, a ≤ b) : l.maximum ≤ b := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [maximum_cons, max_le_iff]
    exact ⟨h a (by simp), ih fun a w => h a (mem_cons.mpr (Or.inr w))⟩


theorem le_minimum_of_forall_le {b : WithTop α} (h : ∀ a ∈ l, b ≤ a) : b ≤ l.minimum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [minimum_cons, le_min_iff]
    exact ⟨h a (by simp), ih fun a w => h a (mem_cons.mpr (Or.inr w))⟩


theorem maximum_eq_coe_iff : maximum l = m ↔ m ∈ l ∧ ∀ a ∈ l, a ≤ m := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    m : α
    ⊢ Iff (Eq l.maximum ↑m) (And (Membership.mem l m) (∀ (a : α), Membership.mem l …
  -/
  rw [maximum, ← WithBot.some_eq_coe, argmax_eq_some_iff]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    m : α
    ⊢ Iff (And (Membership.mem l m) (And (∀ (a : α), Membership.mem l a → LE.le (i …
  -/
  simp only [id_eq, and_congr_right_iff, and_iff_left_iff_imp]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    m : α
    ⊢ Membership.mem l m → (∀ (a : α), Membership.mem l a → LE.le a m) → ∀ (a : α) …
  -/
  intro _ h a hal hma
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    m : α
    a✝ : Membership.mem l m
    h : ∀ (a : α), Membership.mem l a → LE.le a m
    a : α
    hal : Membership.mem l a
    hma : LE.le m a
    ⊢ LE.le (List.indexOf m l) (List.indexOf a l)
  -/
  rw [_root_.le_antisymm hma (h a hal)]
  /-
    🎉 no goals
  -/


theorem minimum_eq_coe_iff : minimum l = m ↔ m ∈ l ∧ ∀ a ∈ l, m ≤ a :=
  @maximum_eq_coe_iff αᵒᵈ _ _ _


theorem coe_le_maximum_iff : a ≤ l.maximum ↔ ∃ b, b ∈ l ∧ a ≤ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    a : α
    ⊢ Iff (LE.le (↑a) l.maximum) (Exists fun b => And (Membership.mem l b) (LE.le  …
  -/
                   /-
                     🎉 no goals
                   -/
  induction' l <;> simp [maximum_cons, *]
                   /-
                     🎉 no goals
                   -/


theorem minimum_le_coe_iff : l.minimum ≤ a ↔ ∃ b, b ∈ l ∧ b ≤ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    a : α
    ⊢ Iff (LE.le l.minimum ↑a) (Exists fun b => And (Membership.mem l b) (LE.le b  …
  -/
                   /-
                     🎉 no goals
                   -/
  induction' l <;> simp [minimum_cons, *]
                   /-
                     🎉 no goals
                   -/


theorem maximum_ne_bot_of_ne_nil (h : l ≠ []) : l.maximum ≠ ⊥ :=
                                    /-
                                      α : Type u_1
                                      inst✝ : LinearOrder α
                                      l : List α
                                      h : Ne l List.nil
                                      head✝ : α
                                      tail✝ : List α
                                      x✝ : Ne (List.cons head✝ tail✝) List.nil
                                      ⊢ Ne (List.cons head✝ tail✝).maximum Bot.bot
                                    -/
  match l, h with | _ :: _, _ => by simp [maximum_cons]
                                    /-
                                      🎉 no goals
                                    -/


theorem minimum_ne_top_of_ne_nil (h : l ≠ []) : l.minimum ≠ ⊤ :=
  @maximum_ne_bot_of_ne_nil αᵒᵈ _ _ h


theorem maximum_ne_bot_of_length_pos (h : 0 < l.length) : l.maximum ≠ ⊥ :=
                                    /-
                                      α : Type u_1
                                      inst✝ : LinearOrder α
                                      l : List α
                                      h : LT.lt 0 l.length
                                      head✝ : α
                                      tail✝ : List α
                                      x✝ : LT.lt 0 (List.cons head✝ tail✝).length
                                      ⊢ Ne (List.cons head✝ tail✝).maximum Bot.bot
                                    -/
  match l, h with | _ :: _, _ => by simp [maximum_cons]
                                    /-
                                      🎉 no goals
                                    -/


theorem minimum_ne_top_of_length_pos (h : 0 < l.length) : l.minimum ≠ ⊤ :=
  maximum_ne_bot_of_length_pos (α := αᵒᵈ) h


/-- The maximum value in a non-empty `List`. -/
def maximum_of_length_pos (h : 0 < l.length) : α :=
  WithBot.unbot l.maximum (maximum_ne_bot_of_length_pos h)


/-- The minimum value in a non-empty `List`. -/
def minimum_of_length_pos (h : 0 < l.length) : α :=
  maximum_of_length_pos (α := αᵒᵈ) h


@[simp]
lemma coe_maximum_of_length_pos (h : 0 < l.length) :
    (l.maximum_of_length_pos h : α) = l.maximum :=
  WithBot.coe_unbot _ _


@[simp]
lemma coe_minimum_of_length_pos (h : 0 < l.length) :
    (l.minimum_of_length_pos h : α) = l.minimum :=
  WithTop.coe_untop _ _


@[simp]
theorem le_maximum_of_length_pos_iff {b : α} (h : 0 < l.length) :
    b ≤ maximum_of_length_pos h ↔ b ≤ l.maximum :=
  WithBot.le_unbot_iff _


@[simp]
theorem minimum_of_length_pos_le_iff {b : α} (h : 0 < l.length) :
    minimum_of_length_pos h ≤ b ↔ l.minimum ≤ b :=
  WithTop.untop_le_iff _


theorem maximum_of_length_pos_mem (h : 0 < l.length) :
    maximum_of_length_pos h ∈ l := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    h : LT.lt 0 l.length
    ⊢ Membership.mem l (List.maximum_of_length_pos h)
  -/
  apply maximum_mem
  /-
    case a
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    h : LT.lt 0 l.length
    ⊢ Eq l.maximum ↑(List.maximum_of_length_pos h)
  -/
  simp only [coe_maximum_of_length_pos]
  /-
    🎉 no goals
  -/


theorem minimum_of_length_pos_mem (h : 0 < l.length) :
    minimum_of_length_pos h ∈ l :=
  maximum_of_length_pos_mem (α := αᵒᵈ) h


theorem le_maximum_of_length_pos_of_mem (h : a ∈ l) (w : 0 < l.length) :
    a ≤ l.maximum_of_length_pos w := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    a : α
    h : Membership.mem l a
    w : LT.lt 0 l.length
    ⊢ LE.le a (List.maximum_of_length_pos w)
  -/
  simp only [le_maximum_of_length_pos_iff]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    a : α
    h : Membership.mem l a
    w : LT.lt 0 l.length
    ⊢ LE.le (↑a) l.maximum
  -/
  exact le_maximum_of_mem' h
  /-
    🎉 no goals
  -/


theorem minimum_of_length_pos_le_of_mem (h : a ∈ l) (w : 0 < l.length) :
    l.minimum_of_length_pos w ≤ a :=
  le_maximum_of_length_pos_of_mem (α := αᵒᵈ) h w


theorem getElem_le_maximum_of_length_pos {i : ℕ} (w : i < l.length) (h := (Nat.zero_lt_of_lt w)) :
    l[i] ≤ l.maximum_of_length_pos h := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    i : Nat
    w : LT.lt i l.length
    h : optParam (LT.lt 0 l.length) ⋯
    ⊢ LE.le (GetElem.getElem l i w) (List.maximum_of_length_pos h)
  -/
  apply le_maximum_of_length_pos_of_mem
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrder α
    l : List α
    i : Nat
    w : LT.lt i l.length
    h : optParam (LT.lt 0 l.length) ⋯
    ⊢ Membership.mem l (GetElem.getElem l i w)
  -/
  exact getElem_mem _
  /-
    🎉 no goals
  -/


theorem minimum_of_length_pos_le_getElem {i : ℕ} (w : i < l.length) (h := (Nat.zero_lt_of_lt w)) :
    l.minimum_of_length_pos h ≤ l[i] :=
  getElem_le_maximum_of_length_pos (α := αᵒᵈ) w


lemma getD_max?_eq_unbot'_maximum (l : List α) (d : α) :
    l.max?.getD d = l.maximum.unbot' d := by
  cases hy : l.maximum with
  | bot => simp [List.maximum_eq_bot.mp hy]
  | coe y =>
    rw [List.maximum_eq_coe_iff] at hy
    simp only [WithBot.unbot'_coe]
    cases hz : l.max? with
    | none => simp [List.max?_eq_none_iff.mp hz] at hy
    | some z =>
      have : Std.Antisymm (α := α) (· ≤ ·) := ⟨_root_.le_antisymm⟩
      rw [List.max?_eq_some_iff] at hz
      · rw [Option.getD_some]
        exact _root_.le_antisymm (hy.right _ hz.left) (hz.right _ hy.left)
      all_goals simp [le_total]


@[deprecated (since := "2024-09-29")]
alias getD_maximum?_eq_unbot'_maximum := getD_max?_eq_unbot'_maximum


lemma getD_min?_eq_untop'_minimum (l : List α) (d : α) :
    l.min?.getD d = l.minimum.untop' d :=
  getD_max?_eq_unbot'_maximum (α := αᵒᵈ) _ _


@[deprecated (since := "2024-09-29")]
alias getD_minimum?_eq_untop'_minimum := getD_min?_eq_untop'_minimum


@[simp]
theorem foldr_max_of_ne_nil (h : l ≠ []) : ↑(l.foldr max ⊥) = l.maximum := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    l : List α
    h : Ne l List.nil
    ⊢ Eq (↑(List.foldr Max.max Bot.bot l)) l.maximum
  -/
  induction' l with hd tl IH
    /-
      case nil
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      l : List α
      h : Ne List.nil List.nil
      ⊢ Eq (↑(List.foldr Max.max Bot.bot List.nil)) List.nil.maximum
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      l : List α
      hd : α
      tl : List α
      IH : Ne tl List.nil → Eq (↑(List.foldr Max.max Bot.bot tl)) tl.maximum
      h : Ne (List.cons hd tl) List.nil
      ⊢ Eq (↑(List.foldr Max.max Bot.bot (List.cons hd tl))) (List.cons hd tl).maximum
    -/
  · rw [maximum_cons, foldr, WithBot.coe_max]
    /-
      case cons
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      l : List α
      hd : α
      tl : List α
      IH : Ne tl List.nil → Eq (↑(List.foldr Max.max Bot.bot tl)) tl.maximum
      h : Ne (List.cons hd tl) List.nil
      ⊢ Eq (Max.max ↑hd ↑(List.foldr Max.max Bot.bot tl)) (Max.max (↑hd) tl.maximum)
    -/
    by_cases h : tl = []
      /-
        case pos
        α : Type u_1
        inst✝¹ : LinearOrder α
        inst✝ : OrderBot α
        l : List α
        hd : α
        tl : List α
        IH : Ne tl List.nil → Eq (↑(List.foldr Max.max Bot.bot tl)) tl.maximum
        h✝ : Ne (List.cons hd tl) List.nil
        h : Eq tl List.nil
        ⊢ Eq (Max.max ↑hd ↑(List.foldr Max.max Bot.bot tl)) (Max.max (↑hd) tl.maximum)
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : LinearOrder α
        inst✝ : OrderBot α
        l : List α
        hd : α
        tl : List α
        IH : Ne tl List.nil → Eq (↑(List.foldr Max.max Bot.bot tl)) tl.maximum
        h✝ : Ne (List.cons hd tl) List.nil
        h : Not (Eq tl List.nil)
        ⊢ Eq (Max.max ↑hd ↑(List.foldr Max.max Bot.bot tl)) (Max.max (↑hd) tl.maximum)
      -/
    · simp [IH h]
      /-
        🎉 no goals
      -/


theorem max_le_of_forall_le (l : List α) (a : α) (h : ∀ x ∈ l, x ≤ a) : l.foldr max ⊥ ≤ a := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    l : List α
    a : α
    h : ∀ (x : α), Membership.mem l x → LE.le x a
    ⊢ LE.le (List.foldr Max.max Bot.bot l) a
  -/
  induction' l with y l IH
    /-
      case nil
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      a : α
      h : ∀ (x : α), Membership.mem List.nil x → LE.le x a
      ⊢ LE.le (List.foldr Max.max Bot.bot List.nil) a
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      a y : α
      l : List α
      IH : (∀ (x : α), Membership.mem l x → LE.le x a) → LE.le (List.foldr Max.max B …
      h : ∀ (x : α), Membership.mem (List.cons y l) x → LE.le x a
      ⊢ LE.le (List.foldr Max.max Bot.bot (List.cons y l)) a
    -/
  · simpa [h y (mem_cons_self _ _)] using IH fun x hx => h x <| mem_cons_of_mem _ hx
    /-
      🎉 no goals
    -/


theorem le_max_of_le {l : List α} {a x : α} (hx : x ∈ l) (h : a ≤ x) : a ≤ l.foldr max ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    l : List α
    a x : α
    hx : Membership.mem l x
    h : LE.le a x
    ⊢ LE.le a (List.foldr Max.max Bot.bot l)
  -/
  induction' l with y l IH
    /-
      case nil
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      a x : α
      h : LE.le a x
      hx : Membership.mem List.nil x
      ⊢ LE.le a (List.foldr Max.max Bot.bot List.nil)
    -/
  · exact absurd hx (not_mem_nil _)
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      a x : α
      h : LE.le a x
      y : α
      l : List α
      IH : Membership.mem l x → LE.le a (List.foldr Max.max Bot.bot l)
      hx : Membership.mem (List.cons y l) x
      ⊢ LE.le a (List.foldr Max.max Bot.bot (List.cons y l))
    -/
  · obtain hl | hl := hx
      /-
        case cons.head
        α : Type u_1
        inst✝¹ : LinearOrder α
        inst✝ : OrderBot α
        a x : α
        h : LE.le a x
        l : List α
        IH : Membership.mem l x → LE.le a (List.foldr Max.max Bot.bot l)
        ⊢ LE.le a (List.foldr Max.max Bot.bot (List.cons x l))
      -/
    · simp only [foldr, foldr_cons]
      /-
        case cons.head
        α : Type u_1
        inst✝¹ : LinearOrder α
        inst✝ : OrderBot α
        a x : α
        h : LE.le a x
        l : List α
        IH : Membership.mem l x → LE.le a (List.foldr Max.max Bot.bot l)
        ⊢ LE.le a (Max.max x (List.foldr Max.max Bot.bot l))
      -/
      exact le_max_of_le_left h
      /-
        🎉 no goals
      -/
      /-
        case cons.tail
        α : Type u_1
        inst✝¹ : LinearOrder α
        inst✝ : OrderBot α
        a x : α
        h : LE.le a x
        y : α
        l : List α
        IH : Membership.mem l x → LE.le a (List.foldr Max.max Bot.bot l)
        a✝ : List.Mem x l
        ⊢ LE.le a (List.foldr Max.max Bot.bot (List.cons y l))
      -/
    · exact le_max_of_le_right (IH (by assumption))
      /-
        🎉 no goals
      -/


@[simp]
theorem foldr_min_of_ne_nil (h : l ≠ []) : ↑(l.foldr min ⊤) = l.minimum :=
  @foldr_max_of_ne_nil αᵒᵈ _ _ _ h


theorem le_min_of_forall_le (l : List α) (a : α) (h : ∀ x ∈ l, a ≤ x) : a ≤ l.foldr min ⊤ :=
  @max_le_of_forall_le αᵒᵈ _ _ _ _ h


theorem min_le_of_le (l : List α) (a : α) {x : α} (hx : x ∈ l) (h : x ≤ a) : l.foldr min ⊤ ≤ a :=
  @le_max_of_le αᵒᵈ _ _ _ _ _ hx h


