/-- An `εNFA` is a set of states (`σ`), a transition function from state to state labelled by the
  alphabet (`step`), a starting state (`start`) and a set of acceptance states (`accept`).
  Note the transition function sends a state to a `Set` of states and can make ε-transitions by
  inputting `none`.
  Since this definition allows for Automata with infinite states, a `Fintype` instance must be
  supplied for true `εNFA`'s. -/
structure εNFA (α : Type u) (σ : Type v) where
  /-- Transition function. The automaton is rendered non-deterministic by this transition function
  returning `Set σ` (rather than `σ`), and ε-transitions are made possible by taking `Option α`
  (rather than `α`). -/
  step : σ → Option α → Set σ
  /-- Starting states. -/
  start : Set σ
  /-- Set of acceptance states. -/
  accept : Set σ


/-- The `εClosure` of a set is the set of states which can be reached by taking a finite string of
ε-transitions from an element of the set. -/
inductive εClosure (S : Set σ) : Set σ
  | base : ∀ s ∈ S, εClosure S s
  | step : ∀ (s), ∀ t ∈ M.step s none, εClosure S s → εClosure S t


@[simp]
theorem subset_εClosure (S : Set σ) : S ⊆ M.εClosure S :=
  εClosure.base


@[simp]
theorem εClosure_empty : M.εClosure ∅ = ∅ :=
                                           /-
                                             α : Type u
                                             σ : Type v
                                             M : εNFA α σ
                                             s : σ
                                             hs : Membership.mem (M.εClosure EmptyCollection.emptyCollection) s
                                             ⊢ False
                                           -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  eq_empty_of_forall_not_mem fun s hs ↦ by induction hs <;> assumption
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem εClosure_univ : M.εClosure univ = univ :=
  eq_univ_of_univ_subset <| subset_εClosure _ _


/-- `M.stepSet S a` is the union of the ε-closure of `M.step s a` for all `s ∈ S`. -/
def stepSet (S : Set σ) (a : α) : Set σ :=
  ⋃ s ∈ S, M.εClosure (M.step s a)


@[simp]
theorem mem_stepSet_iff : s ∈ M.stepSet S a ↔ ∃ t ∈ S, s ∈ M.εClosure (M.step t a) := by
  /-
    α : Type u
    σ : Type v
    M : εNFA α σ
    S : Set σ
    s : σ
    a : α
    ⊢ Iff (Membership.mem (M.stepSet S a) s) (Exists fun t => And (Membership.mem  …
  -/
  simp_rw [stepSet, mem_iUnion₂, exists_prop]
  /-
    🎉 no goals
  -/


@[simp]
theorem stepSet_empty (a : α) : M.stepSet ∅ a = ∅ := by
  /-
    α : Type u
    σ : Type v
    M : εNFA α σ
    a : α
    ⊢ Eq (M.stepSet EmptyCollection.emptyCollection a) EmptyCollection.emptyCollec …
  -/
  simp_rw [stepSet, mem_empty_iff_false, iUnion_false, iUnion_empty]
  /-
    🎉 no goals
  -/


/-- `M.evalFrom S x` computes all possible paths through `M` with input `x` starting at an element
of `S`. -/
def evalFrom (start : Set σ) : List α → Set σ :=
  List.foldl M.stepSet (M.εClosure start)


@[simp]
theorem evalFrom_nil (S : Set σ) : M.evalFrom S [] = M.εClosure S :=
  rfl


@[simp]
theorem evalFrom_singleton (S : Set σ) (a : α) : M.evalFrom S [a] = M.stepSet (M.εClosure S) a :=
  rfl


@[simp]
theorem evalFrom_append_singleton (S : Set σ) (x : List α) (a : α) :
    M.evalFrom S (x ++ [a]) = M.stepSet (M.evalFrom S x) a := by
  /-
    α : Type u
    σ : Type v
    M : εNFA α σ
    S : Set σ
    x : List α
    a : α
    ⊢ Eq (M.evalFrom S (HAppend.hAppend x (List.cons a List.nil))) (M.stepSet (M.e …
  -/
  rw [evalFrom, List.foldl_append, List.foldl_cons, List.foldl_nil]
  /-
    🎉 no goals
  -/


@[simp]
theorem evalFrom_empty (x : List α) : M.evalFrom ∅ x = ∅ := by
  /-
    α : Type u
    σ : Type v
    M : εNFA α σ
    x : List α
    ⊢ Eq (M.evalFrom EmptyCollection.emptyCollection x) EmptyCollection.emptyColle …
  -/
  induction' x using List.reverseRecOn with x a ih
    /-
      case nil
      α : Type u
      σ : Type v
      M : εNFA α σ
      ⊢ Eq (M.evalFrom EmptyCollection.emptyCollection List.nil) EmptyCollection.emp …
    -/
  · rw [evalFrom_nil, εClosure_empty]
    /-
      🎉 no goals
    -/
    /-
      case append_singleton
      α : Type u
      σ : Type v
      M : εNFA α σ
      x : List α
      a : α
      ih : Eq (M.evalFrom EmptyCollection.emptyCollection x) EmptyCollection.emptyCo …
      ⊢ Eq (M.evalFrom EmptyCollection.emptyCollection (HAppend.hAppend x (List.cons …
    -/
  · rw [evalFrom_append_singleton, ih, stepSet_empty]
    /-
      🎉 no goals
    -/


/-- `M.eval x` computes all possible paths through `M` with input `x` starting at an element of
`M.start`. -/
def eval :=
  M.evalFrom M.start


@[simp]
theorem eval_nil : M.eval [] = M.εClosure M.start :=
  rfl


@[simp]
theorem eval_singleton (a : α) : M.eval [a] = M.stepSet (M.εClosure M.start) a :=
  rfl


@[simp]
theorem eval_append_singleton (x : List α) (a : α) : M.eval (x ++ [a]) = M.stepSet (M.eval x) a :=
  evalFrom_append_singleton _ _ _ _


/-- `M.accepts` is the language of `x` such that there is an accept state in `M.eval x`. -/
def accepts : Language α :=
  { x | ∃ S ∈ M.accept, S ∈ M.eval x }


/-- `M.toNFA` is an `NFA` constructed from an `εNFA` `M`. -/
def toNFA : NFA α σ where
  step S a := M.εClosure (M.step S a)
  start := M.εClosure M.start
  accept := M.accept


@[simp]
theorem toNFA_evalFrom_match (start : Set σ) :
    M.toNFA.evalFrom (M.εClosure start) = M.evalFrom start :=
  rfl


@[simp]
theorem toNFA_correct : M.toNFA.accepts = M.accepts :=
  rfl


theorem pumping_lemma [Fintype σ] {x : List α} (hx : x ∈ M.accepts)
    (hlen : Fintype.card (Set σ) ≤ List.length x) :
    ∃ a b c, x = a ++ b ++ c ∧
      a.length + b.length ≤ Fintype.card (Set σ) ∧ b ≠ [] ∧ {a} * {b}∗ * {c} ≤ M.accepts :=
  M.toNFA.pumping_lemma hx hlen


/-- `M.toεNFA` is an `εNFA` constructed from an `NFA` `M` by using the same start and accept
  states and transition functions. -/
def toεNFA (M : NFA α σ) : εNFA α σ where
  step s a := a.casesOn' ∅ fun a ↦ M.step s a
  start := M.start
  accept := M.accept


@[simp]
theorem toεNFA_εClosure (M : NFA α σ) (S : Set σ) : M.toεNFA.εClosure S = S := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    S : Set σ
    ⊢ Eq (M.toεNFA.εClosure S) S
  -/
  ext a
  /-
    case h
    α : Type u
    σ : Type v
    M : NFA α σ
    S : Set σ
    a : σ
    ⊢ Iff (Membership.mem (M.toεNFA.εClosure S) a) (Membership.mem S a)
  -/
  refine ⟨?_, εNFA.εClosure.base _⟩
  /-
    case h
    α : Type u
    σ : Type v
    M : NFA α σ
    S : Set σ
    a : σ
    ⊢ Membership.mem (M.toεNFA.εClosure S) a → Membership.mem S a
  -/
  rintro (⟨_, h⟩ | ⟨_, _, h, _⟩)
    /-
      case h.base
      α : Type u
      σ : Type v
      M : NFA α σ
      S : Set σ
      a : σ
      h : Membership.mem S a
      ⊢ Membership.mem S a
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case h.step
      α : Type u
      σ : Type v
      M : NFA α σ
      S : Set σ
      a s✝ : σ
      a✝ : M.toεNFA.εClosure S s✝
      h : Membership.mem (M.toεNFA.step s✝ Option.none) a
      ⊢ Membership.mem S a
    -/
  · cases h
    /-
      🎉 no goals
    -/


@[simp]
theorem toεNFA_evalFrom_match (M : NFA α σ) (start : Set σ) :
    M.toεNFA.evalFrom start = M.evalFrom start := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    start : Set σ
    ⊢ Eq (M.toεNFA.evalFrom start) (M.evalFrom start)
  -/
  rw [evalFrom, εNFA.evalFrom, toεNFA_εClosure]
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    start : Set σ
    ⊢ Eq (List.foldl M.toεNFA.stepSet start) (List.foldl M.stepSet start)
  -/
  suffices εNFA.stepSet (toεNFA M) = stepSet M by rw [this]
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    start : Set σ
    ⊢ Eq M.toεNFA.stepSet M.stepSet
  -/
  ext S s
  /-
    case h.h.h
    α : Type u
    σ : Type v
    M : NFA α σ
    start S : Set σ
    s : α
    x✝ : σ
    ⊢ Iff (Membership.mem (M.toεNFA.stepSet S s) x✝) (Membership.mem (M.stepSet S  …
  -/
  simp only [stepSet, εNFA.stepSet, exists_prop, Set.mem_iUnion]
  /-
    case h.h.h
    α : Type u
    σ : Type v
    M : NFA α σ
    start S : Set σ
    s : α
    x✝ : σ
    ⊢ Iff (Exists fun i => And (Membership.mem S i) (Membership.mem (M.toεNFA.εClo …
  -/
  apply exists_congr
  /-
    case h.h.h.h
    α : Type u
    σ : Type v
    M : NFA α σ
    start S : Set σ
    s : α
    x✝ : σ
    ⊢ ∀ (a : σ), Iff (And (Membership.mem S a) (Membership.mem (M.toεNFA.εClosure  …
  -/
  simp only [and_congr_right_iff]
  /-
    case h.h.h.h
    α : Type u
    σ : Type v
    M : NFA α σ
    start S : Set σ
    s : α
    x✝ : σ
    ⊢ ∀ (a : σ), Membership.mem S a → Iff (Membership.mem (M.toεNFA.εClosure (M.to …
  -/
  intro _ _
  /-
    case h.h.h.h
    α : Type u
    σ : Type v
    M : NFA α σ
    start S : Set σ
    s : α
    x✝ a✝¹ : σ
    a✝ : Membership.mem S a✝¹
    ⊢ Iff (Membership.mem (M.toεNFA.εClosure (M.toεNFA.step a✝¹ (Option.some s)))  …
  -/
  rw [M.toεNFA_εClosure]
  /-
    case h.h.h.h
    α : Type u
    σ : Type v
    M : NFA α σ
    start S : Set σ
    s : α
    x✝ a✝¹ : σ
    a✝ : Membership.mem S a✝¹
    ⊢ Iff (Membership.mem (M.toεNFA.step a✝¹ (Option.some s)) x✝) (Membership.mem  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toεNFA_correct (M : NFA α σ) : M.toεNFA.accepts = M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    ⊢ Eq M.toεNFA.accepts M.accepts
  -/
  rw [εNFA.accepts, εNFA.eval, toεNFA_evalFrom_match]
  /-
    α : Type u
    σ : Type v
    M : NFA α σ
    ⊢ Eq (setOf fun x => Exists fun S => And (Membership.mem M.toεNFA.accept S) (M …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : Zero (εNFA α σ) :=
  ⟨⟨fun _ _ ↦ ∅, ∅, ∅⟩⟩


instance : One (εNFA α σ) :=
  ⟨⟨fun _ _ ↦ ∅, univ, univ⟩⟩


instance : Inhabited (εNFA α σ) :=
  ⟨0⟩


@[simp]
theorem step_zero (s a) : (0 : εNFA α σ).step s a = ∅ :=
  rfl


@[simp]
theorem step_one (s a) : (1 : εNFA α σ).step s a = ∅ :=
  rfl


@[simp]
theorem start_zero : (0 : εNFA α σ).start = ∅ :=
  rfl


@[simp]
theorem start_one : (1 : εNFA α σ).start = univ :=
  rfl


@[simp]
theorem accept_zero : (0 : εNFA α σ).accept = ∅ :=
  rfl


@[simp]
theorem accept_one : (1 : εNFA α σ).accept = univ :=
  rfl


