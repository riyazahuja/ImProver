/-- An NFA is a set of states (`σ`), a transition function from state to state labelled by the
  alphabet (`step`), a set of starting states (`start`) and a set of acceptance states (`accept`).
  Note the transition function sends a state to a `Set` of states. These are the states that it
  may be sent to. -/
structure NFA (α : Type u) (σ : Type v) where
  step : σ → α → Set σ
  start : Set σ
  accept : Set σ


instance : Inhabited (NFA α σ) :=
  ⟨NFA.mk (fun _ _ => ∅) ∅ ∅⟩


/-- `M.stepSet S a` is the union of `M.step s a` for all `s ∈ S`. -/
def stepSet (S : Set σ) (a : α) : Set σ :=
  ⋃ s ∈ S, M.step s a


theorem mem_stepSet (s : σ) (S : Set σ) (a : α) : s ∈ M.stepSet S a ↔ ∃ t ∈ S, s ∈ M.step t a := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    s : σ
    S : Set σ
    a : α
    ⊢ Iff (Membership.mem (M.stepSet S a) s) (Exists fun t => And (Membership.mem  …
  -/
  simp [stepSet]
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          α : Type u
                                                          σ : Type v
                                                          M : NFA α σ
                                                          a : α
                                                          ⊢ Eq (M.stepSet EmptyCollection.emptyCollection a) EmptyCollection.emptyCollec …
                                                        -/
theorem stepSet_empty (a : α) : M.stepSet ∅ a = ∅ := by simp [stepSet]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- `M.evalFrom S x` computes all possible paths though `M` with input `x` starting at an element
  of `S`. -/
def evalFrom (start : Set σ) : List α → Set σ :=
  List.foldl M.stepSet start


@[simp]
theorem evalFrom_nil (S : Set σ) : M.evalFrom S [] = S :=
  rfl


@[simp]
theorem evalFrom_singleton (S : Set σ) (a : α) : M.evalFrom S [a] = M.stepSet S a :=
  rfl


@[simp]
theorem evalFrom_append_singleton (S : Set σ) (x : List α) (a : α) :
    M.evalFrom S (x ++ [a]) = M.stepSet (M.evalFrom S x) a := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    S : Set σ
    x : List α
    a : α
    ⊢ Eq (M.evalFrom S (HAppend.hAppend x (List.cons a List.nil))) (M.stepSet (M.e …
  -/
  simp only [evalFrom, List.foldl_append, List.foldl_cons, List.foldl_nil]
  /-
    🎉 no goals
  -/


/-- `M.eval x` computes all possible paths though `M` with input `x` starting at an element of
  `M.start`. -/
def eval : List α → Set σ :=
  M.evalFrom M.start


@[simp]
theorem eval_nil : M.eval [] = M.start :=
  rfl


@[simp]
theorem eval_singleton (a : α) : M.eval [a] = M.stepSet M.start a :=
  rfl


@[simp]
theorem eval_append_singleton (x : List α) (a : α) : M.eval (x ++ [a]) = M.stepSet (M.eval x) a :=
  evalFrom_append_singleton _ _ _ _


/-- `M.accepts` is the language of `x` such that there is an accept state in `M.eval x`. -/
def accepts : Language α := {x | ∃ S ∈ M.accept, S ∈ M.eval x}


theorem mem_accepts {x : List α} : x ∈ M.accepts ↔ ∃ S ∈ M.accept, S ∈ M.evalFrom M.start x := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    x : List α
    ⊢ Iff (Membership.mem M.accepts x) (Exists fun S => And (Membership.mem M.acce …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `M.toDFA` is a `DFA` constructed from an `NFA` `M` using the subset construction. The
  states is the type of `Set`s of `M.state` and the step function is `M.stepSet`. -/
def toDFA : DFA α (Set σ) where
  step := M.stepSet
  start := M.start
  accept := { S | ∃ s ∈ S, s ∈ M.accept }


@[simp]
theorem toDFA_correct : M.toDFA.accepts = M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    ⊢ Eq M.toDFA.accepts M.accepts
  -/
  ext x
  /-
    case h
    α : Type u
    σ : Type v
    M : NFA α σ
    x : List α
    ⊢ Iff (Membership.mem M.toDFA.accepts x) (Membership.mem M.accepts x)
  -/
  rw [mem_accepts, DFA.mem_accepts]
  /-
    case h
    α : Type u
    σ : Type v
    M : NFA α σ
    x : List α
    ⊢ Iff (Membership.mem M.toDFA.accept (M.toDFA.eval x)) (Exists fun S => And (M …
  -/
                    /-
                      🎉 no goals
                    -/
  constructor <;> · exact fun ⟨w, h2, h3⟩ => ⟨w, h3, h2⟩
                    /-
                      🎉 no goals
                    -/


theorem pumping_lemma [Fintype σ] {x : List α} (hx : x ∈ M.accepts)
    (hlen : Fintype.card (Set σ) ≤ List.length x) :
    ∃ a b c,
      x = a ++ b ++ c ∧
        a.length + b.length ≤ Fintype.card (Set σ) ∧ b ≠ [] ∧ {a} * {b}∗ * {c} ≤ M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    inst✝ : Fintype σ
    x : List α
    hx : Membership.mem M.accepts x
    hlen : LE.le (Fintype.card (Set σ)) x.length
    ⊢ Exists fun a => Exists fun b => Exists fun c => And (Eq x (HAppend.hAppend ( …
  -/
  rw [← toDFA_correct] at hx ⊢
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    inst✝ : Fintype σ
    x : List α
    hx : Membership.mem M.toDFA.accepts x
    hlen : LE.le (Fintype.card (Set σ)) x.length
    ⊢ Exists fun a => Exists fun b => Exists fun c => And (Eq x (HAppend.hAppend ( …
  -/
  exact M.toDFA.pumping_lemma hx hlen
  /-
    🎉 no goals
  -/


/-- `M.toNFA` is an `NFA` constructed from a `DFA` `M` by using the same start and accept
  states and a transition function which sends `s` with input `a` to the singleton `M.step s a`. -/
@[simps] def toNFA (M : DFA α σ') : NFA α σ' where
  step s a := {M.step s a}
  start := {M.start}
  accept := M.accept


@[simp]
theorem toNFA_evalFrom_match (M : DFA α σ) (start : σ) (s : List α) :
    M.toNFA.evalFrom {start} s = {M.evalFrom start s} := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    start : σ
    s : List α
    ⊢ Eq (M.toNFA.evalFrom (Singleton.singleton start) s) (Singleton.singleton (M. …
  -/
  change List.foldl M.toNFA.stepSet {start} s = {List.foldl M.step start s}
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    start : σ
    s : List α
    ⊢ Eq (List.foldl M.toNFA.stepSet (Singleton.singleton start) s) (Singleton.sin …
  -/
  induction' s with a s ih generalizing start
    /-
      case nil
      α : Type u
      σ : Type v
      M : DFA α σ
      start : σ
      ⊢ Eq (List.foldl M.toNFA.stepSet (Singleton.singleton start) List.nil) (Single …
    -/
  · tauto
    /-
      🎉 no goals
    -/
  · rw [List.foldl, List.foldl,
      show M.toNFA.stepSet {start} a = {M.step start a} by simp [NFA.stepSet] ]
    /-
      case cons
      α : Type u
      σ : Type v
      M : DFA α σ
      a : α
      s : List α
      ih : ∀ (start : σ), Eq (List.foldl M.toNFA.stepSet (Singleton.singleton start) …
      start : σ
      ⊢ Eq (List.foldl M.toNFA.stepSet (Singleton.singleton (M.step start a)) s) (Si …
    -/
    tauto
    /-
      🎉 no goals
    -/


@[simp]
theorem toNFA_correct (M : DFA α σ) : M.toNFA.accepts = M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    ⊢ Eq M.toNFA.accepts M.accepts
  -/
  ext x
  /-
    case h
    α : Type u
    σ : Type v
    M : DFA α σ
    x : List α
    ⊢ Iff (Membership.mem M.toNFA.accepts x) (Membership.mem M.accepts x)
  -/
  rw [NFA.mem_accepts, toNFA_start, toNFA_evalFrom_match]
  /-
    case h
    α : Type u
    σ : Type v
    M : DFA α σ
    x : List α
    ⊢ Iff (Exists fun S => And (Membership.mem M.toNFA.accept S) (Membership.mem ( …
  -/
  constructor
    /-
      case h.mp
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      ⊢ (Exists fun S => And (Membership.mem M.toNFA.accept S) (Membership.mem (Sing …
    -/
  · rintro ⟨S, hS₁, hS₂⟩
    /-
      case h.mp.intro.intro
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      S : σ
      hS₁ : Membership.mem M.toNFA.accept S
      hS₂ : Membership.mem (Singleton.singleton (M.evalFrom M.start x)) S
      ⊢ Membership.mem M.accepts x
    -/
    rwa [Set.mem_singleton_iff.mp hS₂] at hS₁
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      ⊢ Membership.mem M.accepts x → Exists fun S => And (Membership.mem M.toNFA.acc …
    -/
  · exact fun h => ⟨M.eval x, h, rfl⟩
    /-
      🎉 no goals
    -/


