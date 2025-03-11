def parallel.aux2 : List (Computation α) → α ⊕ (List (Computation α)) :=
  List.foldr
    (fun c o =>
      match o with
      | Sum.inl a => Sum.inl a
      | Sum.inr ls => rmap (fun c' => c' :: ls) (destruct c))
    (Sum.inr [])


def parallel.aux1 :
    List (Computation α) × WSeq (Computation α) →
      α ⊕ (List (Computation α) × WSeq (Computation α))
  | (l, S) =>
    rmap
      (fun l' =>
        match Seq.destruct S with
        | none => (l', Seq.nil)
        | some (none, S') => (l', S')
        | some (some c, S') => (c :: l', S'))
      (parallel.aux2 l)


/-- Parallel computation of an infinite stream of computations,
  taking the first result -/
def parallel (S : WSeq (Computation α)) : Computation α :=
  corec parallel.aux1 ([], S)


theorem terminates_parallel.aux :
    ∀ {l : List (Computation α)} {S c},
      c ∈ l → Terminates c → Terminates (corec parallel.aux1 (l, S)) := by
  have lem1 :
    ∀ l S, (∃ a : α, parallel.aux2 l = Sum.inl a) → Terminates (corec parallel.aux1 (l, S)) := by
    intro l S e
    cases' e with a e
    have : corec parallel.aux1 (l, S) = return a := by
      apply destruct_eq_pure
      simp only [parallel.aux1, rmap, corec_eq]
      rw [e]
    rw [this]
    -- Porting note: This line is required.
    exact ret_terminates a
  /-
    α : Type u
    lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
    ⊢ ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)} {c : Computa …
  -/
  intro l S c m T
  /-
    α : Type u
    lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
    l : List (Computation α)
    S : Stream'.WSeq (Computation α)
    c : Computation α
    m : Membership.mem l c
    T : c.Terminates
    ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
  -/
  revert l S
  /-
    α : Type u
    lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
    c : Computation α
    T : c.Terminates
    ⊢ ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membership. …
  -/
  apply @terminatesRecOn _ _ c T _ _
    /-
      α : Type u
      lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
      c : Computation α
      T : c.Terminates
      ⊢ ∀ (a : α) {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Mem …
    -/
  · intro a l S m
    /-
      α : Type u
      lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
      c : Computation α
      T : c.Terminates
      a : α
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      m : Membership.mem l (Computation.pure a)
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
    apply lem1
    /-
      case a
      α : Type u
      lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
      c : Computation α
      T : c.Terminates
      a : α
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      m : Membership.mem l (Computation.pure a)
      ⊢ Exists fun a => Eq (Computation.parallel.aux2 l) (Sum.inl a)
    -/
                                 /-
                                   🎉 no goals
                                 -/
    induction' l with c l IH <;> simp at m
    /-
      case a.cons
      α : Type u
      lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
      c✝ : Computation α
      T : c✝.Terminates
      a : α
      S : Stream'.WSeq (Computation α)
      c : Computation α
      l : List (Computation α)
      IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
      m : Or (Eq (Computation.pure a) c) (Membership.mem l (Computation.pure a))
      ⊢ Exists fun a => Eq (Computation.parallel.aux2 (List.cons c l)) (Sum.inl a)
    -/
    cases' m with e m
      /-
        case a.cons.inl
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        e : Eq (Computation.pure a) c
        ⊢ Exists fun a => Eq (Computation.parallel.aux2 (List.cons c l)) (Sum.inl a)
      -/
    · rw [← e]
      /-
        case a.cons.inl
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        e : Eq (Computation.pure a) c
        ⊢ Exists fun a_1 => Eq (Computation.parallel.aux2 (List.cons (Computation.pure …
      -/
      simp only [parallel.aux2, rmap, List.foldr_cons, destruct_pure]
      /-
        case a.cons.inl
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        e : Eq (Computation.pure a) c
        ⊢ Exists fun a_1 => Eq (Computation.parallel.aux2.match_1 (fun o => Sum α (Lis …
      -/
                /-
                  🎉 no goals
                -/
      split <;> simp
                /-
                  🎉 no goals
                -/
      /-
        case a.cons.inr
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        m : Membership.mem l (Computation.pure a)
        ⊢ Exists fun a => Eq (Computation.parallel.aux2 (List.cons c l)) (Sum.inl a)
      -/
    · cases' IH m with a' e
      /-
        case a.cons.inr.intro
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        m : Membership.mem l (Computation.pure a)
        a' : α
        e : Eq (Computation.parallel.aux2 l) (Sum.inl a')
        ⊢ Exists fun a => Eq (Computation.parallel.aux2 (List.cons c l)) (Sum.inl a)
      -/
      simp only [parallel.aux2, rmap, List.foldr_cons]
      /-
        case a.cons.inr.intro
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        m : Membership.mem l (Computation.pure a)
        a' : α
        e : Eq (Computation.parallel.aux2 l) (Sum.inl a')
        ⊢ Exists fun a => Eq (Computation.parallel.aux2.match_1 (fun o => Sum α (List  …
      -/
      simp? [parallel.aux2] at e says simp only [parallel.aux2, rmap] at e
      /-
        case a.cons.inr.intro
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        m : Membership.mem l (Computation.pure a)
        a' : α
        e : Eq (List.foldr (fun c o => Computation.parallel.aux2.match_1 (fun o => Sum …
        ⊢ Exists fun a => Eq (Computation.parallel.aux2.match_1 (fun o => Sum α (List  …
      -/
      rw [e]
      /-
        case a.cons.inr.intro
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c✝ : Computation α
        T : c✝.Terminates
        a : α
        S : Stream'.WSeq (Computation α)
        c : Computation α
        l : List (Computation α)
        IH : Membership.mem l (Computation.pure a) → Exists fun a => Eq (Computation.p …
        m : Membership.mem l (Computation.pure a)
        a' : α
        e : Eq (List.foldr (fun c o => Computation.parallel.aux2.match_1 (fun o => Sum …
        ⊢ Exists fun a => Eq (Computation.parallel.aux2.match_1 (fun o => Sum α (List  …
      -/
      exact ⟨a', rfl⟩
      /-
        🎉 no goals
      -/
    /-
      α : Type u
      lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
      c : Computation α
      T : c.Terminates
      ⊢ ∀ (s : Computation α), (∀ {l : List (Computation α)} {S : Stream'.WSeq (Comp …
    -/
  · intro s IH l S m
    have H1 : ∀ l', parallel.aux2 l = Sum.inr l' → s ∈ l' := by
      induction' l with c l IH' <;> intro l' e' <;> simp at m
      cases' m with e m <;> simp [parallel.aux2] at e'
      · rw [← e] at e'
        -- Porting note: `revert e'` & `intro e'` are required.
        revert e'
        split
        · simp
        · simp only [destruct_think, Sum.inr.injEq]
          rintro rfl
          simp
      · induction' e : List.foldr (fun c o =>
            match o with
            | Sum.inl a => Sum.inl a
            | Sum.inr ls => rmap (fun c' => c' :: ls) (destruct c))
          (Sum.inr List.nil) l with a' ls <;> erw [e] at e'
        · contradiction
        have := IH' m _ e
        -- Porting note: `revert e'` & `intro e'` are required.
        revert e'
        cases destruct c <;> intro e' <;> [injection e'; injection e' with h']
        rw [← h']
        simp [this]
    /-
      α : Type u
      lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
      c : Computation α
      T : c.Terminates
      s : Computation α
      IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      m : Membership.mem l s.think
      H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
    induction' h : parallel.aux2 l with a l'
      /-
        case inl
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c : Computation α
        T : c.Terminates
        s : Computation α
        IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        m : Membership.mem l s.think
        H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
        a : α
        h : Eq (Computation.parallel.aux2 l) (Sum.inl a)
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
    · exact lem1 _ _ ⟨a, h⟩
      /-
        🎉 no goals
      -/
    · have H2 : corec parallel.aux1 (l, S) = think _ := destruct_eq_think (by
        simp only [parallel.aux1, rmap, corec_eq]
        rw [h])
      /-
        case inr
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c : Computation α
        T : c.Terminates
        s : Computation α
        IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        m : Membership.mem l s.think
        H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        H2 : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) ( …
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
      rw [H2]
      /-
        case inr
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c : Computation α
        T : c.Terminates
        s : Computation α
        IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        m : Membership.mem l s.think
        H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        H2 : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) ( …
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      refine @Computation.think_terminates _ _ ?_
      /-
        case inr
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c : Computation α
        T : c.Terminates
        s : Computation α
        IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        m : Membership.mem l s.think
        H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        H2 : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) ( …
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      have := H1 _ h
      /-
        case inr
        α : Type u
        lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
        c : Computation α
        T : c.Terminates
        s : Computation α
        IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        m : Membership.mem l s.think
        H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        H2 : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) ( …
        this : Membership.mem l' s
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      rcases Seq.destruct S with (_ | ⟨_ | c, S'⟩) <;> simp [parallel.aux1] <;> apply IH <;>
        /-
          case inr.none.m
          α : Type u
          lem1 : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), (Exist …
          c : Computation α
          T : c.Terminates
          s : Computation α
          IH : ∀ {l : List (Computation α)} {S : Stream'.WSeq (Computation α)}, Membersh …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          m : Membership.mem l s.think
          H1 : ∀ (l' : List (Computation α)), Eq (Computation.parallel.aux2 l) (Sum.inr  …
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          H2 : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) ( …
          this : Membership.mem l' s
          ⊢ Membership.mem l' s
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        simp [this]
        /-
          🎉 no goals
        -/


theorem terminates_parallel {S : WSeq (Computation α)} {c} (h : c ∈ S) [T : Terminates c] :
    Terminates (parallel S) := by
  suffices
    ∀ (n) (l : List (Computation α)) (S c),
      c ∈ l ∨ some (some c) = Seq.get? S n → Terminates c → Terminates (corec parallel.aux1 (l, S))
    from
    let ⟨n, h⟩ := h
    this n [] S c (Or.inr h) T
  /-
    α : Type u
    S : Stream'.WSeq (Computation α)
    c : Computation α
    h : Membership.mem S c
    T : c.Terminates
    ⊢ ∀ (n : Nat) (l : List (Computation α)) (S : Stream'.Seq (Option (Computation …
  -/
  intro n; induction' n with n IH <;> intro l S c o T
    /-
      case zero
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      c✝ : Computation α
      h : Membership.mem S✝ c✝
      T✝ : c✝.Terminates
      l : List (Computation α)
      S : Stream'.Seq (Option (Computation α))
      c : Computation α
      o : Or (Membership.mem l c) (Eq (Option.some (Option.some c)) (S.get? 0))
      T : c.Terminates
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
  · cases' o with a a
      /-
        case zero.inl
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Membership.mem l c
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
    · exact terminates_parallel.aux a T
      /-
        🎉 no goals
      -/
    /-
      case zero.inr
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      c✝ : Computation α
      h : Membership.mem S✝ c✝
      T✝ : c✝.Terminates
      l : List (Computation α)
      S : Stream'.Seq (Option (Computation α))
      c : Computation α
      T : c.Terminates
      a : Eq (Option.some (Option.some c)) (S.get? 0)
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
    have H : Seq.destruct S = some (some c, Seq.tail S) := by simp [Seq.destruct, (· <$> ·), ← a]
    /-
      case zero.inr
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      c✝ : Computation α
      h : Membership.mem S✝ c✝
      T✝ : c✝.Terminates
      l : List (Computation α)
      S : Stream'.Seq (Option (Computation α))
      c : Computation α
      T : c.Terminates
      a : Eq (Option.some (Option.some c)) (S.get? 0)
      H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
    induction' h : parallel.aux2 l with a l'
    · have C : corec parallel.aux1 (l, S) = pure a := by
        apply destruct_eq_pure
        rw [corec_eq, parallel.aux1]
        rw [h]
        simp only [rmap]
      /-
        case zero.inr.inl
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a✝ : Eq (Option.some (Option.some c)) (S.get? 0)
        H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
        a : α
        h : Eq (Computation.parallel.aux2 l) (Sum.inl a)
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
      rw [C]
      /-
        case zero.inr.inl
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a✝ : Eq (Option.some (Option.some c)) (S.get? 0)
        H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
        a : α
        h : Eq (Computation.parallel.aux2 l) (Sum.inl a)
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.pure a).Terminates
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    · have C : corec parallel.aux1 (l, S) = _ := destruct_eq_think (by
        simp only [corec_eq, rmap, parallel.aux1.eq_1]
        rw [h, H])
      /-
        case zero.inr.inr
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? 0)
        H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
      rw [C]
      /-
        case zero.inr.inr
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? 0)
        H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      refine @Computation.think_terminates _ _ ?_
      /-
        case zero.inr.inr
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? 0)
        H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      apply terminates_parallel.aux _ T
      /-
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? 0)
        H : Eq S.destruct (Option.some { fst := Option.some c, snd := S.tail })
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ Membership.mem (List.cons c l') c
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case succ
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      c✝ : Computation α
      h : Membership.mem S✝ c✝
      T✝ : c✝.Terminates
      n : Nat
      IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
      l : List (Computation α)
      S : Stream'.Seq (Option (Computation α))
      c : Computation α
      o : Or (Membership.mem l c) (Eq (Option.some (Option.some c)) (S.get? (HAdd.hA …
      T : c.Terminates
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
  · cases' o with a a
      /-
        case succ.inl
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        n : Nat
        IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Membership.mem l c
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
    · exact terminates_parallel.aux a T
      /-
        🎉 no goals
      -/
    /-
      case succ.inr
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      c✝ : Computation α
      h : Membership.mem S✝ c✝
      T✝ : c✝.Terminates
      n : Nat
      IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
      l : List (Computation α)
      S : Stream'.Seq (Option (Computation α))
      c : Computation α
      T : c.Terminates
      a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
      ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
    -/
    induction' h : parallel.aux2 l with a l'
    · have C : corec parallel.aux1 (l, S) = pure a := by
        apply destruct_eq_pure
        rw [corec_eq, parallel.aux1]
        rw [h]
        simp only [rmap]
      /-
        case succ.inr.inl
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        n : Nat
        IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a✝ : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
        a : α
        h : Eq (Computation.parallel.aux2 l) (Sum.inl a)
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
      rw [C]
      /-
        case succ.inr.inl
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        n : Nat
        IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a✝ : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
        a : α
        h : Eq (Computation.parallel.aux2 l) (Sum.inl a)
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.pure a).Terminates
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    · have C : corec parallel.aux1 (l, S) = _ := destruct_eq_think (by
        simp only [corec_eq, rmap, parallel.aux1.eq_1]
        rw [h])
      /-
        case succ.inr.inr
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        n : Nat
        IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }).Termina …
      -/
      rw [C]
      /-
        case succ.inr.inr
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        n : Nat
        IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      refine @Computation.think_terminates _ _ ?_
      have TT : ∀ l', Terminates (corec parallel.aux1 (l', S.tail)) := by
        intro
        apply IH _ _ _ (Or.inr _) T
        rw [a]
        cases' S with f al
        rfl
      /-
        case succ.inr.inr
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        c✝ : Computation α
        h✝ : Membership.mem S✝ c✝
        T✝ : c✝.Terminates
        n : Nat
        IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
        l : List (Computation α)
        S : Stream'.Seq (Option (Computation α))
        c : Computation α
        T : c.Terminates
        a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
        l' : List (Computation α)
        h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
        C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
        TT : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.au …
        ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
      -/
      induction' e : Seq.get? S 0 with o
      · have D : Seq.destruct S = none := by
          dsimp [Seq.destruct]
          rw [e]
          rfl
        /-
          case succ.inr.inr.none
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          c✝ : Computation α
          h✝ : Membership.mem S✝ c✝
          T✝ : c✝.Terminates
          n : Nat
          IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
          l : List (Computation α)
          S : Stream'.Seq (Option (Computation α))
          c : Computation α
          T : c.Terminates
          a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
          TT : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.au …
          e : Eq (S.get? 0) Option.none
          D : Eq S.destruct Option.none
          ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
        -/
        rw [D]
        /-
          case succ.inr.inr.none
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          c✝ : Computation α
          h✝ : Membership.mem S✝ c✝
          T✝ : c✝.Terminates
          n : Nat
          IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
          l : List (Computation α)
          S : Stream'.Seq (Option (Computation α))
          c : Computation α
          T : c.Terminates
          a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
          TT : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.au …
          e : Eq (S.get? 0) Option.none
          D : Eq S.destruct Option.none
          ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
        -/
        simp only
        /-
          case succ.inr.inr.none
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          c✝ : Computation α
          h✝ : Membership.mem S✝ c✝
          T✝ : c✝.Terminates
          n : Nat
          IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
          l : List (Computation α)
          S : Stream'.Seq (Option (Computation α))
          c : Computation α
          T : c.Terminates
          a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
          TT : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.au …
          e : Eq (S.get? 0) Option.none
          D : Eq S.destruct Option.none
          ⊢ (Computation.corec Computation.parallel.aux1 { fst := l', snd := Stream'.Seq …
        -/
        have TT := TT l'
        /-
          case succ.inr.inr.none
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          c✝ : Computation α
          h✝ : Membership.mem S✝ c✝
          T✝ : c✝.Terminates
          n : Nat
          IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
          l : List (Computation α)
          S : Stream'.Seq (Option (Computation α))
          c : Computation α
          T : c.Terminates
          a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
          TT✝ : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.a …
          e : Eq (S.get? 0) Option.none
          D : Eq S.destruct Option.none
          TT : (Computation.corec Computation.parallel.aux1 { fst := l', snd := S.tail } …
          ⊢ (Computation.corec Computation.parallel.aux1 { fst := l', snd := Stream'.Seq …
        -/
        rwa [Seq.destruct_eq_nil D, Seq.tail_nil] at TT
        /-
          🎉 no goals
        -/
      · have D : Seq.destruct S = some (o, S.tail) := by
          dsimp [Seq.destruct]
          rw [e]
          rfl
        /-
          case succ.inr.inr.some
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          c✝ : Computation α
          h✝ : Membership.mem S✝ c✝
          T✝ : c✝.Terminates
          n : Nat
          IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
          l : List (Computation α)
          S : Stream'.Seq (Option (Computation α))
          c : Computation α
          T : c.Terminates
          a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
          TT : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.au …
          o : Option (Computation α)
          e : Eq (S.get? 0) (Option.some o)
          D : Eq S.destruct (Option.some { fst := o, snd := S.tail })
          ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
        -/
        rw [D]
        /-
          case succ.inr.inr.some
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          c✝ : Computation α
          h✝ : Membership.mem S✝ c✝
          T✝ : c✝.Terminates
          n : Nat
          IH : ∀ (l : List (Computation α)) (S : Stream'.Seq (Option (Computation α))) ( …
          l : List (Computation α)
          S : Stream'.Seq (Option (Computation α))
          c : Computation α
          T : c.Terminates
          a : Eq (Option.some (Option.some c)) (S.get? (HAdd.hAdd n 1))
          l' : List (Computation α)
          h : Eq (Computation.parallel.aux2 l) (Sum.inr l')
          C : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
          TT : ∀ (l' : List (Computation α)), (Computation.corec Computation.parallel.au …
          o : Option (Computation α)
          e : Eq (S.get? 0) (Option.some o)
          D : Eq S.destruct (Option.some { fst := o, snd := S.tail })
          ⊢ (Computation.corec Computation.parallel.aux1 (Computation.parallel.aux1.matc …
        -/
                            /-
                              🎉 no goals
                            -/
        cases' o with c <;> simp [parallel.aux1, TT]
                            /-
                              🎉 no goals
                            -/


theorem exists_of_mem_parallel {S : WSeq (Computation α)} {a} (h : a ∈ parallel S) :
    ∃ c ∈ S, a ∈ c := by
  suffices
    ∀ C, a ∈ C → ∀ (l : List (Computation α)) (S),
      corec parallel.aux1 (l, S) = C → ∃ c, (c ∈ l ∨ c ∈ S) ∧ a ∈ c from
    let ⟨c, h1, h2⟩ := this _ h [] S rfl
    ⟨c, h1.resolve_left <| List.not_mem_nil _, h2⟩
  let F : List (Computation α) → α ⊕ (List (Computation α)) → Prop := by
    intro l a
    cases' a with a l'
    · exact ∃ c ∈ l, a ∈ c
    · exact ∀ a', (∃ c ∈ l', a' ∈ c) → ∃ c ∈ l, a' ∈ c
  have lem1 : ∀ l : List (Computation α), F l (parallel.aux2 l) := by
    intro l
    induction' l with c l IH <;> simp only [parallel.aux2, List.foldr]
    · intro a h
      rcases h with ⟨c, hn, _⟩
      exact False.elim <| List.not_mem_nil _ hn
    · simp only [parallel.aux2] at IH
      -- Porting note: `revert IH` & `intro IH` are required.
      revert IH
      cases' List.foldr (fun c o =>
        match o with
        | Sum.inl a => Sum.inl a
        | Sum.inr ls => rmap (fun c' => c' :: ls) (destruct c)) (Sum.inr List.nil) l with a ls <;>
        intro IH <;>
        simp only [parallel.aux2]
      · rcases IH with ⟨c', cl, ac⟩
        exact ⟨c', List.Mem.tail _ cl, ac⟩
      · induction' h : destruct c with a c' <;> simp only [rmap]
        · refine ⟨c, List.mem_cons_self _ _, ?_⟩
          rw [destruct_eq_pure h]
          apply ret_mem
        · intro a' h
          rcases h with ⟨d, dm, ad⟩
          simp? at dm says simp only [List.mem_cons] at dm
          cases' dm with e dl
          · rw [e] at ad
            refine ⟨c, List.mem_cons_self _ _, ?_⟩
            rw [destruct_eq_think h]
            exact think_mem ad
          · cases' IH a' ⟨d, dl, ad⟩ with d dm
            cases' dm with dm ad
            exact ⟨d, List.Mem.tail _ dm, ad⟩
  /-
    α : Type u
    S : Stream'.WSeq (Computation α)
    a : α
    h : Membership.mem (Computation.parallel S) a
    F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
    lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
    ⊢ ∀ (C : Computation α), Membership.mem C a → ∀ (l : List (Computation α)) (S  …
  -/
  intro C aC
  -- Porting note: `revert e'` & `intro e'` are required.
  apply memRecOn aC <;> [skip; intro C' IH] <;> intro l S e <;> have e' := congr_arg destruct e <;>
    have := lem1 l <;> simp only [parallel.aux1, corec_eq, destruct_pure, destruct_think] at e' <;>
    revert this e' <;> cases' parallel.aux2 l with a' l' <;> intro this e' <;>
    [injection e' with h'; injection e'; injection e'; injection e' with h']
    /-
      case h1.inl
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      a : α
      h : Membership.mem (Computation.parallel S✝) a
      F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
      lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
      C : Computation α
      aC : Membership.mem C a
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      e : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
      a' : α
      this : F l (Sum.inl a')
      h' : Eq a' a
      ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
    -/
  · rw [h'] at this
    /-
      case h1.inl
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      a : α
      h : Membership.mem (Computation.parallel S✝) a
      F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
      lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
      C : Computation α
      aC : Membership.mem C a
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      e : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
      a' : α
      this : F l (Sum.inl a)
      h' : Eq a' a
      ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
    -/
    rcases this with ⟨c, cl, ac⟩
    /-
      case h1.inl.intro.intro
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      a : α
      h : Membership.mem (Computation.parallel S✝) a
      F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
      lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
      C : Computation α
      aC : Membership.mem C a
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      e : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) (C …
      a' : α
      h' : Eq a' a
      c : Computation α
      cl : Membership.mem l c
      ac : Membership.mem c a
      ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
    -/
    exact ⟨c, Or.inl cl, ac⟩
    /-
      🎉 no goals
    -/
    /-
      case h2.inr
      α : Type u
      S✝ : Stream'.WSeq (Computation α)
      a : α
      h : Membership.mem (Computation.parallel S✝) a
      F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
      lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
      C : Computation α
      aC : Membership.mem C a
      C' : Computation α
      IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
      l : List (Computation α)
      S : Stream'.WSeq (Computation α)
      e : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C' …
      l' : List (Computation α)
      this : F l (Sum.inr l')
      h' : Eq (Computation.corec Computation.parallel.aux1 ((fun l' => Computation.p …
      ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
    -/
  · induction' e : Seq.destruct S with a <;> rw [e] at h'
    · exact
        let ⟨d, o, ad⟩ := IH _ _ h'
        let ⟨c, cl, ac⟩ := this a ⟨d, o.resolve_right (WSeq.not_mem_nil _), ad⟩
        ⟨c, Or.inl cl, ac⟩
      /-
        case h2.inr.some
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        a✝ : α
        h : Membership.mem (Computation.parallel S✝) a✝
        F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
        lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
        C : Computation α
        aC : Membership.mem C a✝
        C' : Computation α
        IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
        l' : List (Computation α)
        this : F l (Sum.inr l')
        a : Stream'.Seq1 (Option (Computation α))
        h' : Eq (Computation.corec Computation.parallel.aux1 ((fun l' => Computation.p …
        e : Eq (Stream'.Seq.destruct S) (Option.some a)
        ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
      -/
    · cases' a with o S'
      /-
        case h2.inr.some.mk
        α : Type u
        S✝ : Stream'.WSeq (Computation α)
        a : α
        h : Membership.mem (Computation.parallel S✝) a
        F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
        lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
        C : Computation α
        aC : Membership.mem C a
        C' : Computation α
        IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
        l : List (Computation α)
        S : Stream'.WSeq (Computation α)
        e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
        l' : List (Computation α)
        this : F l (Sum.inr l')
        o : Option (Computation α)
        S' : Stream'.Seq (Option (Computation α))
        h' : Eq (Computation.corec Computation.parallel.aux1 ((fun l' => Computation.p …
        e : Eq (Stream'.Seq.destruct S) (Option.some { fst := o, snd := S' })
        ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
      -/
      cases' o with c <;> simp [parallel.aux1] at h' <;> rcases IH _ _ h' with ⟨d, dl | dS', ad⟩
      · exact
          let ⟨c, cl, ac⟩ := this a ⟨d, dl, ad⟩
          ⟨c, Or.inl cl, ac⟩
        /-
          case h2.inr.some.mk.none.intro.intro.inr
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := l', snd := S' }) …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.none, snd := S' })
          d : Computation α
          ad : Membership.mem d a
          dS' : Membership.mem S' d
          ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
        -/
      · refine ⟨d, Or.inr ?_, ad⟩
        /-
          case h2.inr.some.mk.none.intro.intro.inr
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := l', snd := S' }) …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.none, snd := S' })
          d : Computation α
          ad : Membership.mem d a
          dS' : Membership.mem S' d
          ⊢ Membership.mem S d
        -/
        rw [Seq.destruct_eq_cons e]
        /-
          case h2.inr.some.mk.none.intro.intro.inr
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := l', snd := S' }) …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.none, snd := S' })
          d : Computation α
          ad : Membership.mem d a
          dS' : Membership.mem S' d
          ⊢ Membership.mem (Stream'.Seq.cons Option.none S') d
        -/
        exact Seq.mem_cons_of_mem _ dS'
        /-
          🎉 no goals
        -/
        /-
          case h2.inr.some.mk.some.intro.intro.inl
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          c : Computation α
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
          d : Computation α
          ad : Membership.mem d a
          dl : Membership.mem (List.cons c l') d
          ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
        -/
      · simp at dl
        /-
          case h2.inr.some.mk.some.intro.intro.inl
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          c : Computation α
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
          d : Computation α
          ad : Membership.mem d a
          dl : Or (Eq d c) (Membership.mem l' d)
          ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
        -/
        cases' dl with dc dl
          /-
            case h2.inr.some.mk.some.intro.intro.inl.inl
            α : Type u
            S✝ : Stream'.WSeq (Computation α)
            a : α
            h : Membership.mem (Computation.parallel S✝) a
            F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
            lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
            C : Computation α
            aC : Membership.mem C a
            C' : Computation α
            IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
            l : List (Computation α)
            S : Stream'.WSeq (Computation α)
            e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
            l' : List (Computation α)
            this : F l (Sum.inr l')
            S' : Stream'.Seq (Option (Computation α))
            c : Computation α
            h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
            e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
            d : Computation α
            ad : Membership.mem d a
            dc : Eq d c
            ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
          -/
        · rw [dc] at ad
          /-
            case h2.inr.some.mk.some.intro.intro.inl.inl
            α : Type u
            S✝ : Stream'.WSeq (Computation α)
            a : α
            h : Membership.mem (Computation.parallel S✝) a
            F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
            lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
            C : Computation α
            aC : Membership.mem C a
            C' : Computation α
            IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
            l : List (Computation α)
            S : Stream'.WSeq (Computation α)
            e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
            l' : List (Computation α)
            this : F l (Sum.inr l')
            S' : Stream'.Seq (Option (Computation α))
            c : Computation α
            h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
            e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
            d : Computation α
            ad : Membership.mem c a
            dc : Eq d c
            ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
          -/
          refine ⟨c, Or.inr ?_, ad⟩
          /-
            case h2.inr.some.mk.some.intro.intro.inl.inl
            α : Type u
            S✝ : Stream'.WSeq (Computation α)
            a : α
            h : Membership.mem (Computation.parallel S✝) a
            F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
            lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
            C : Computation α
            aC : Membership.mem C a
            C' : Computation α
            IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
            l : List (Computation α)
            S : Stream'.WSeq (Computation α)
            e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
            l' : List (Computation α)
            this : F l (Sum.inr l')
            S' : Stream'.Seq (Option (Computation α))
            c : Computation α
            h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
            e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
            d : Computation α
            ad : Membership.mem c a
            dc : Eq d c
            ⊢ Membership.mem S c
          -/
          rw [Seq.destruct_eq_cons e]
          /-
            case h2.inr.some.mk.some.intro.intro.inl.inl
            α : Type u
            S✝ : Stream'.WSeq (Computation α)
            a : α
            h : Membership.mem (Computation.parallel S✝) a
            F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
            lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
            C : Computation α
            aC : Membership.mem C a
            C' : Computation α
            IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
            l : List (Computation α)
            S : Stream'.WSeq (Computation α)
            e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
            l' : List (Computation α)
            this : F l (Sum.inr l')
            S' : Stream'.Seq (Option (Computation α))
            c : Computation α
            h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
            e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
            d : Computation α
            ad : Membership.mem c a
            dc : Eq d c
            ⊢ Membership.mem (Stream'.Seq.cons (Option.some c) S') c
          -/
          apply Seq.mem_cons
          /-
            🎉 no goals
          -/
        · exact
            let ⟨c, cl, ac⟩ := this a ⟨d, dl, ad⟩
            ⟨c, Or.inl cl, ac⟩
        /-
          case h2.inr.some.mk.some.intro.intro.inr
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          c : Computation α
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
          d : Computation α
          ad : Membership.mem d a
          dS' : Membership.mem S' d
          ⊢ Exists fun c => And (Or (Membership.mem l c) (Membership.mem S c)) (Membersh …
        -/
      · refine ⟨d, Or.inr ?_, ad⟩
        /-
          case h2.inr.some.mk.some.intro.intro.inr
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          c : Computation α
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
          d : Computation α
          ad : Membership.mem d a
          dS' : Membership.mem S' d
          ⊢ Membership.mem S d
        -/
        rw [Seq.destruct_eq_cons e]
        /-
          case h2.inr.some.mk.some.intro.intro.inr
          α : Type u
          S✝ : Stream'.WSeq (Computation α)
          a : α
          h : Membership.mem (Computation.parallel S✝) a
          F : List (Computation α) → Sum α (List (Computation α)) → Prop := fun l a => S …
          lem1 : ∀ (l : List (Computation α)), F l (Computation.parallel.aux2 l)
          C : Computation α
          aC : Membership.mem C a
          C' : Computation α
          IH : ∀ (l : List (Computation α)) (S : Stream'.WSeq (Computation α)), Eq (Comp …
          l : List (Computation α)
          S : Stream'.WSeq (Computation α)
          e✝ : Eq (Computation.corec Computation.parallel.aux1 { fst := l, snd := S }) C …
          l' : List (Computation α)
          this : F l (Sum.inr l')
          S' : Stream'.Seq (Option (Computation α))
          c : Computation α
          h' : Eq (Computation.corec Computation.parallel.aux1 { fst := List.cons c l',  …
          e : Eq (Stream'.Seq.destruct S) (Option.some { fst := Option.some c, snd := S' …
          d : Computation α
          ad : Membership.mem d a
          dS' : Membership.mem S' d
          ⊢ Membership.mem (Stream'.Seq.cons (Option.some c) S') d
        -/
        exact Seq.mem_cons_of_mem _ dS'
        /-
          🎉 no goals
        -/


theorem map_parallel (f : α → β) (S) : map f (parallel S) = parallel (S.map (map f)) := by
  refine
    eq_of_bisim
      (fun c1 c2 =>
        ∃ l S,
          c1 = map f (corec parallel.aux1 (l, S)) ∧
            c2 = corec parallel.aux1 (l.map (map f), S.map (map f)))
      ?_ ⟨[], S, rfl, rfl⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    S : Stream'.WSeq (Computation α)
    ⊢ Computation.IsBisimulation fun c1 c2 => Exists fun l => Exists fun S => And  …
  -/
  intro c1 c2 h
  exact
    match c1, c2, h with
    | _, _, ⟨l, S, rfl, rfl⟩ => by
      have : parallel.aux2 (l.map (map f))
          = lmap f (rmap (List.map (map f)) (parallel.aux2 l)) := by
        simp only [parallel.aux2, rmap, lmap]
        induction' l with c l IH <;> simp
        rw [IH]
        cases List.foldr _ _ _
        · simp
        · cases destruct c <;> simp
      simp only [BisimO, destruct_map, lmap, rmap, corec_eq, parallel.aux1.eq_1]
      rw [this]
      cases' parallel.aux2 l with a l' <;> simp
      induction' S using WSeq.recOn with c S S <;> simp <;>
        exact ⟨_, _, rfl, rfl⟩


theorem parallel_empty (S : WSeq (Computation α)) (h : S.head ~> none) : parallel S = empty _ :=
  eq_empty_of_not_terminates fun ⟨⟨a, m⟩⟩ => by
    /-
      α : Type u
      S : Stream'.WSeq (Computation α)
      h : S.head.Promises Option.none
      x✝ : (Computation.parallel S).Terminates
      a : α
      m : Membership.mem (Computation.parallel S) a
      ⊢ False
    -/
    let ⟨c, cs, _⟩ := exists_of_mem_parallel m
    /-
      α : Type u
      S : Stream'.WSeq (Computation α)
      h : S.head.Promises Option.none
      x✝ : (Computation.parallel S).Terminates
      a : α
      m : Membership.mem (Computation.parallel S) a
      c : Computation α
      cs : Membership.mem S c
      right✝ : Membership.mem c a
      ⊢ False
    -/
    let ⟨n, nm⟩ := WSeq.exists_get?_of_mem cs
    /-
      α : Type u
      S : Stream'.WSeq (Computation α)
      h : S.head.Promises Option.none
      x✝ : (Computation.parallel S).Terminates
      a : α
      m : Membership.mem (Computation.parallel S) a
      c : Computation α
      cs : Membership.mem S c
      right✝ : Membership.mem c a
      n : Nat
      nm : Membership.mem (S.get? n) (Option.some c)
      ⊢ False
    -/
    let ⟨c', h'⟩ := WSeq.head_some_of_get?_some nm
    /-
      α : Type u
      S : Stream'.WSeq (Computation α)
      h : S.head.Promises Option.none
      x✝ : (Computation.parallel S).Terminates
      a : α
      m : Membership.mem (Computation.parallel S) a
      c : Computation α
      cs : Membership.mem S c
      right✝ : Membership.mem c a
      n : Nat
      nm : Membership.mem (S.get? n) (Option.some c)
      c' : Computation α
      h' : Membership.mem S.head (Option.some c')
      ⊢ False
    -/
    injection h h'
    /-
      🎉 no goals
    -/

-- The reason this isn't trivial from exists_of_mem_parallel is because it eliminates to Sort

def parallelRec {S : WSeq (Computation α)} (C : α → Sort v) (H : ∀ s ∈ S, ∀ a ∈ s, C a) {a}
    (h : a ∈ parallel S) : C a := by
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    ⊢ C a
  -/
  let T : WSeq (Computation (α × Computation α)) := S.map fun c => c.map fun a => (a, c)
  have : S = T.map (map fun c => c.1) := by
    rw [← WSeq.map_comp]
    refine (WSeq.map_id _).symm.trans (congr_arg (fun f => WSeq.map f S) ?_)
    funext c
    dsimp [id, Function.comp_def]
    rw [← map_comp]
    exact (map_id _).symm
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    ⊢ C a
  -/
  have pe := congr_arg parallel this
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.parallel (Stream'.WSeq.map (Comp …
    ⊢ C a
  -/
  rw [← map_parallel] at pe
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.map (fun c => c.1) (Computation. …
    ⊢ C a
  -/
  have h' := h
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.map (fun c => c.1) (Computation. …
    h' : Membership.mem (Computation.parallel S) a
    ⊢ C a
  -/
  rw [pe] at h'
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.map (fun c => c.1) (Computation. …
    h' : Membership.mem (Computation.map (fun c => c.1) (Computation.parallel T)) a
    ⊢ C a
  -/
  haveI : Terminates (parallel T) := (terminates_map_iff _ _).1 ⟨⟨_, h'⟩⟩
  /-
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this✝ : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.map (fun c => c.1) (Computation. …
    h' : Membership.mem (Computation.map (fun c => c.1) (Computation.parallel T)) a
    this : (Computation.parallel T).Terminates
    ⊢ C a
  -/
  induction' e : get (parallel T) with a' c
  have : a ∈ c ∧ c ∈ S := by
    rcases exists_of_mem_map h' with ⟨d, dT, cd⟩
    rw [get_eq_of_mem _ dT] at e
    cases e
    dsimp at cd
    cases cd
    rcases exists_of_mem_parallel dT with ⟨d', dT', ad'⟩
    rcases WSeq.exists_of_mem_map dT' with ⟨c', cs', e'⟩
    rw [← e'] at ad'
    rcases exists_of_mem_map ad' with ⟨a', ac', e'⟩
    injection e' with i1 i2
    constructor
    · rwa [i1, i2] at ac'
    · rwa [i2] at cs'
  /-
    case mk
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this✝¹ : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.map (fun c => c.1) (Computation. …
    h' : Membership.mem (Computation.map (fun c => c.1) (Computation.parallel T)) a
    this✝ : (Computation.parallel T).Terminates
    a' : α
    c : Computation α
    e : Eq (Computation.parallel T).get { fst := a', snd := c }
    this : And (Membership.mem c a) (Membership.mem S c)
    ⊢ C a
  -/
  cases' this with ac cs
  /-
    case mk.intro
    α : Type u
    β : Type v
    S : Stream'.WSeq (Computation α)
    C : α → Sort v
    H : (s : Computation α) → Membership.mem S s → (a : α) → Membership.mem s a →  …
    a : α
    h : Membership.mem (Computation.parallel S) a
    T : Stream'.WSeq (Computation (Prod α (Computation α))) := Stream'.WSeq.map (f …
    this✝ : Eq S (Stream'.WSeq.map (Computation.map fun c => c.1) T)
    pe : Eq (Computation.parallel S) (Computation.map (fun c => c.1) (Computation. …
    h' : Membership.mem (Computation.map (fun c => c.1) (Computation.parallel T)) a
    this : (Computation.parallel T).Terminates
    a' : α
    c : Computation α
    e : Eq (Computation.parallel T).get { fst := a', snd := c }
    ac : Membership.mem c a
    cs : Membership.mem S c
    ⊢ C a
  -/
  apply H _ cs _ ac
  /-
    🎉 no goals
  -/


theorem parallel_promises {S : WSeq (Computation α)} {a} (H : ∀ s ∈ S, s ~> a) : parallel S ~> a :=
  fun _ ma' =>
  let ⟨_, cs, ac⟩ := exists_of_mem_parallel ma'
  H _ cs ac


theorem mem_parallel {S : WSeq (Computation α)} {a} (H : ∀ s ∈ S, s ~> a) {c} (cs : c ∈ S)
    (ac : a ∈ c) : a ∈ parallel S := by
  /-
    α : Type u
    S : Stream'.WSeq (Computation α)
    a : α
    H : ∀ (s : Computation α), Membership.mem S s → s.Promises a
    c : Computation α
    cs : Membership.mem S c
    ac : Membership.mem c a
    ⊢ Membership.mem (Computation.parallel S) a
  -/
  haveI := terminates_of_mem ac
  /-
    α : Type u
    S : Stream'.WSeq (Computation α)
    a : α
    H : ∀ (s : Computation α), Membership.mem S s → s.Promises a
    c : Computation α
    cs : Membership.mem S c
    ac : Membership.mem c a
    this : c.Terminates
    ⊢ Membership.mem (Computation.parallel S) a
  -/
  haveI := terminates_parallel cs
  /-
    α : Type u
    S : Stream'.WSeq (Computation α)
    a : α
    H : ∀ (s : Computation α), Membership.mem S s → s.Promises a
    c : Computation α
    cs : Membership.mem S c
    ac : Membership.mem c a
    this✝ : c.Terminates
    this : (Computation.parallel S).Terminates
    ⊢ Membership.mem (Computation.parallel S) a
  -/
  exact mem_of_promises _ (parallel_promises H)
  /-
    🎉 no goals
  -/


theorem parallel_congr_lem {S T : WSeq (Computation α)} {a} (H : S.LiftRel Equiv T) :
    (∀ s ∈ S, s ~> a) ↔ ∀ t ∈ T, t ~> a :=
  ⟨fun h1 _ tT =>
    let ⟨_, sS, se⟩ := WSeq.exists_of_liftRel_right H tT
    (promises_congr se _).1 (h1 _ sS),
    fun h2 _ sS =>
    let ⟨_, tT, se⟩ := WSeq.exists_of_liftRel_left H sS
    (promises_congr se _).2 (h2 _ tT)⟩

-- The parallel operation is only deterministic when all computation paths lead to the same value

theorem parallel_congr_left {S T : WSeq (Computation α)} {a} (h1 : ∀ s ∈ S, s ~> a)
    (H : S.LiftRel Equiv T) : parallel S ~ parallel T :=
  let h2 := (parallel_congr_lem H).1 h1
  fun a' =>
  ⟨fun h => by
    /-
      α : Type u
      S T : Stream'.WSeq (Computation α)
      a : α
      h1 : ∀ (s : Computation α), Membership.mem S s → s.Promises a
      H : Stream'.WSeq.LiftRel Computation.Equiv S T
      h2 : ∀ (t : Computation α), Membership.mem T t → t.Promises a := (Computation. …
      a' : α
      h : Membership.mem (Computation.parallel S) a'
      ⊢ Membership.mem (Computation.parallel T) a'
    -/
    have aa := parallel_promises h1 h
    /-
      α : Type u
      S T : Stream'.WSeq (Computation α)
      a : α
      h1 : ∀ (s : Computation α), Membership.mem S s → s.Promises a
      H : Stream'.WSeq.LiftRel Computation.Equiv S T
      h2 : ∀ (t : Computation α), Membership.mem T t → t.Promises a := (Computation. …
      a' : α
      h : Membership.mem (Computation.parallel S) a'
      aa : Eq a a'
      ⊢ Membership.mem (Computation.parallel T) a'
    -/
    rw [← aa]
    /-
      α : Type u
      S T : Stream'.WSeq (Computation α)
      a : α
      h1 : ∀ (s : Computation α), Membership.mem S s → s.Promises a
      H : Stream'.WSeq.LiftRel Computation.Equiv S T
      h2 : ∀ (t : Computation α), Membership.mem T t → t.Promises a := (Computation. …
      a' : α
      h : Membership.mem (Computation.parallel S) a'
      aa : Eq a a'
      ⊢ Membership.mem (Computation.parallel T) a
    -/
    rw [← aa] at h
    exact
      let ⟨s, sS, as⟩ := exists_of_mem_parallel h
      let ⟨t, tT, st⟩ := WSeq.exists_of_liftRel_left H sS
      let aT := (st _).1 as
      mem_parallel h2 tT aT,
    fun h => by
    /-
      α : Type u
      S T : Stream'.WSeq (Computation α)
      a : α
      h1 : ∀ (s : Computation α), Membership.mem S s → s.Promises a
      H : Stream'.WSeq.LiftRel Computation.Equiv S T
      h2 : ∀ (t : Computation α), Membership.mem T t → t.Promises a := (Computation. …
      a' : α
      h : Membership.mem (Computation.parallel T) a'
      ⊢ Membership.mem (Computation.parallel S) a'
    -/
    have aa := parallel_promises h2 h
    /-
      α : Type u
      S T : Stream'.WSeq (Computation α)
      a : α
      h1 : ∀ (s : Computation α), Membership.mem S s → s.Promises a
      H : Stream'.WSeq.LiftRel Computation.Equiv S T
      h2 : ∀ (t : Computation α), Membership.mem T t → t.Promises a := (Computation. …
      a' : α
      h : Membership.mem (Computation.parallel T) a'
      aa : Eq a a'
      ⊢ Membership.mem (Computation.parallel S) a'
    -/
    rw [← aa]
    /-
      α : Type u
      S T : Stream'.WSeq (Computation α)
      a : α
      h1 : ∀ (s : Computation α), Membership.mem S s → s.Promises a
      H : Stream'.WSeq.LiftRel Computation.Equiv S T
      h2 : ∀ (t : Computation α), Membership.mem T t → t.Promises a := (Computation. …
      a' : α
      h : Membership.mem (Computation.parallel T) a'
      aa : Eq a a'
      ⊢ Membership.mem (Computation.parallel S) a
    -/
    rw [← aa] at h
    exact
      let ⟨s, sS, as⟩ := exists_of_mem_parallel h
      let ⟨t, tT, st⟩ := WSeq.exists_of_liftRel_right H sS
      let aT := (st _).2 as
      mem_parallel h1 tT aT⟩


theorem parallel_congr_right {S T : WSeq (Computation α)} {a} (h2 : ∀ t ∈ T, t ~> a)
    (H : S.LiftRel Equiv T) : parallel S ~ parallel T :=
  parallel_congr_left ((parallel_congr_lem H).2 h2) H


