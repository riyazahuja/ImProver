/-- A DFA is a set of states (`σ`), a transition function from state to state labelled by the
  alphabet (`step`), a starting state (`start`) and a set of acceptance states (`accept`). -/
structure DFA (α : Type u) (σ : Type v) where
  /-- A transition function from state to state labelled by the alphabet. -/
  step : σ → α → σ
  /-- Starting state. -/
  start : σ
  /-- Set of acceptance states. -/
  accept : Set σ


instance [Inhabited σ] : Inhabited (DFA α σ) :=
  ⟨DFA.mk (fun _ _ => default) default ∅⟩


/-- `M.evalFrom s x` evaluates `M` with input `x` starting from the state `s`. -/
def evalFrom (s : σ) : List α → σ :=
  List.foldl M.step s


@[simp]
theorem evalFrom_nil (s : σ) : M.evalFrom s [] = s :=
  rfl


@[simp]
theorem evalFrom_singleton (s : σ) (a : α) : M.evalFrom s [a] = M.step s a :=
  rfl


@[simp]
theorem evalFrom_append_singleton (s : σ) (x : List α) (a : α) :
    M.evalFrom s (x ++ [a]) = M.step (M.evalFrom s x) a := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    s : σ
    x : List α
    a : α
    ⊢ Eq (M.evalFrom s (HAppend.hAppend x (List.cons a List.nil))) (M.step (M.eval …
  -/
  simp only [evalFrom, List.foldl_append, List.foldl_cons, List.foldl_nil]
  /-
    🎉 no goals
  -/


/-- `M.eval x` evaluates `M` with input `x` starting from the state `M.start`. -/
def eval : List α → σ :=
  M.evalFrom M.start


@[simp]
theorem eval_nil : M.eval [] = M.start :=
  rfl


@[simp]
theorem eval_singleton (a : α) : M.eval [a] = M.step M.start a :=
  rfl


@[simp]
theorem eval_append_singleton (x : List α) (a : α) : M.eval (x ++ [a]) = M.step (M.eval x) a :=
  evalFrom_append_singleton _ _ _ _


theorem evalFrom_of_append (start : σ) (x y : List α) :
    M.evalFrom start (x ++ y) = M.evalFrom (M.evalFrom start x) y :=
  x.foldl_append _ _ y


/--
`M.acceptsFrom s` is the language of `x` such that `M.evalFrom s x` is an accept state.
-/
def acceptsFrom (s : σ) : Language α := {x | M.evalFrom s x ∈ M.accept}


theorem mem_acceptsFrom {s : σ} {x : List α} :
                                                          /-
                                                            α : Type u
                                                            σ : Type v
                                                            M : DFA α σ
                                                            s : σ
                                                            x : List α
                                                            ⊢ Iff (Membership.mem (M.acceptsFrom s) x) (Membership.mem M.accept (M.evalFro …
                                                          -/
    x ∈ M.acceptsFrom s ↔ M.evalFrom s x ∈ M.accept := by rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- `M.accepts` is the language of `x` such that `M.eval x` is an accept state. -/
def accepts : Language α := M.acceptsFrom M.start


                                                                             /-
                                                                               α : Type u
                                                                               σ : Type v
                                                                               M : DFA α σ
                                                                               x : List α
                                                                               ⊢ Iff (Membership.mem M.accepts x) (Membership.mem M.accept (M.eval x))
                                                                             -/
theorem mem_accepts {x : List α} : x ∈ M.accepts ↔ M.eval x ∈ M.accept := by rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem evalFrom_split [Fintype σ] {x : List α} {s t : σ} (hlen : Fintype.card σ ≤ x.length)
    (hx : M.evalFrom s x = t) :
    ∃ q a b c,
      x = a ++ b ++ c ∧
        a.length + b.length ≤ Fintype.card σ ∧
          b ≠ [] ∧ M.evalFrom s a = q ∧ M.evalFrom q b = q ∧ M.evalFrom q c = t := by
  obtain ⟨n, m, hneq, heq⟩ :=
    Fintype.exists_ne_map_eq_of_card_lt
      (fun n : Fin (Fintype.card σ + 1) => M.evalFrom s (x.take n)) (by norm_num)
  /-
    case intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    s t : σ
    hlen : LE.le (Fintype.card σ) x.length
    hx : Eq (M.evalFrom s x) t
    n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
    hneq : Ne n m
    heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
    ⊢ Exists fun q => Exists fun a => Exists fun b => Exists fun c => And (Eq x (H …
  -/
  wlog hle : (n : ℕ) ≤ m generalizing n m
    /-
      case intro.intro.intro.inr
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      this : ∀ (n m : Fin (HAdd.hAdd (Fintype.card σ) 1)), Ne n m → Eq (M.evalFrom s …
      hle : Not (LE.le ↑n ↑m)
      ⊢ Exists fun q => Exists fun a => Exists fun b => Exists fun c => And (Eq x (H …
    -/
  · exact this m n hneq.symm heq.symm (le_of_not_le hle)
    /-
      🎉 no goals
    -/
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    s t : σ
    hlen : LE.le (Fintype.card σ) x.length
    hx : Eq (M.evalFrom s x) t
    n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
    hneq : Ne n m
    heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
    hle : LE.le ↑n ↑m
    ⊢ Exists fun q => Exists fun a => Exists fun b => Exists fun c => And (Eq x (H …
  -/
  have hm : (m : ℕ) ≤ Fintype.card σ := Fin.is_le m
  refine
    ⟨M.evalFrom s ((x.take m).take n), (x.take m).take n, (x.take m).drop n,
                    x.drop m, ?_, ?_, ?_, by rfl, ?_⟩
    /-
      case refine_1
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      ⊢ Eq x (HAppend.hAppend (HAppend.hAppend (List.take (↑n) (List.take (↑m) x)) ( …
    -/
  · rw [List.take_append_drop, List.take_append_drop]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      ⊢ LE.le (HAdd.hAdd (List.take (↑n) (List.take (↑m) x)).length (List.drop (↑n)  …
    -/
  · simp only [List.length_drop, List.length_take]
    /-
      case refine_2
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      ⊢ LE.le (HAdd.hAdd (Min.min (↑n) (Min.min (↑m) x.length)) (HSub.hSub (Min.min  …
    -/
    rw [min_eq_left (hm.trans hlen), min_eq_left hle, add_tsub_cancel_of_le hle]
    /-
      case refine_2
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      ⊢ LE.le (↑m) (Fintype.card σ)
    -/
    exact hm
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      ⊢ Ne (List.drop (↑n) (List.take (↑m) x)) List.nil
    -/
  · intro h
    /-
      case refine_3
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
      ⊢ False
    -/
    have hlen' := congr_arg List.length h
    /-
      case refine_3
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
      hlen' : Eq (List.drop (↑n) (List.take (↑m) x)).length List.nil.length
      ⊢ False
    -/
    simp only [List.length_drop, List.length, List.length_take] at hlen'
    /-
      case refine_3
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
      hlen' : Eq (HSub.hSub (Min.min (↑m) x.length) ↑n) 0
      ⊢ False
    -/
    rw [min_eq_left, tsub_eq_zero_iff_le] at hlen'
      /-
        case refine_3
        α : Type u
        σ : Type v
        M : DFA α σ
        inst✝ : Fintype σ
        x : List α
        s t : σ
        hlen : LE.le (Fintype.card σ) x.length
        hx : Eq (M.evalFrom s x) t
        n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
        hneq : Ne n m
        heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
        hle : LE.le ↑n ↑m
        hm : LE.le (↑m) (Fintype.card σ)
        h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
        hlen' : LE.le ↑m ↑n
        ⊢ False
      -/
    · apply hneq
      /-
        case refine_3
        α : Type u
        σ : Type v
        M : DFA α σ
        inst✝ : Fintype σ
        x : List α
        s t : σ
        hlen : LE.le (Fintype.card σ) x.length
        hx : Eq (M.evalFrom s x) t
        n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
        hneq : Ne n m
        heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
        hle : LE.le ↑n ↑m
        hm : LE.le (↑m) (Fintype.card σ)
        h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
        hlen' : LE.le ↑m ↑n
        ⊢ Eq n m
      -/
      apply le_antisymm
      /-
        case refine_3.a
        α : Type u
        σ : Type v
        M : DFA α σ
        inst✝ : Fintype σ
        x : List α
        s t : σ
        hlen : LE.le (Fintype.card σ) x.length
        hx : Eq (M.evalFrom s x) t
        n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
        hneq : Ne n m
        heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
        hle : LE.le ↑n ↑m
        hm : LE.le (↑m) (Fintype.card σ)
        h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
        hlen' : LE.le ↑m ↑n
        ⊢ LE.le n m
      -/
      assumption'
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      α : Type u
      σ : Type v
      M : DFA α σ
      inst✝ : Fintype σ
      x : List α
      s t : σ
      hlen : LE.le (Fintype.card σ) x.length
      hx : Eq (M.evalFrom s x) t
      n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
      hneq : Ne n m
      heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
      hle : LE.le ↑n ↑m
      hm : LE.le (↑m) (Fintype.card σ)
      h : Eq (List.drop (↑n) (List.take (↑m) x)) List.nil
      hlen' : Eq (HSub.hSub (Min.min (↑m) x.length) ↑n) 0
      ⊢ LE.le (↑m) x.length
    -/
    exact hm.trans hlen
    /-
      🎉 no goals
    -/
  have hq : M.evalFrom (M.evalFrom s ((x.take m).take n)) ((x.take m).drop n) =
      M.evalFrom s ((x.take m).take n) := by
    rw [List.take_take, min_eq_left hle, ← evalFrom_of_append, heq, ← min_eq_left hle, ←
      List.take_take, min_eq_left hle, List.take_append_drop]
  /-
    case refine_4
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    s t : σ
    hlen : LE.le (Fintype.card σ) x.length
    hx : Eq (M.evalFrom s x) t
    n m : Fin (HAdd.hAdd (Fintype.card σ) 1)
    hneq : Ne n m
    heq : Eq (M.evalFrom s (List.take (↑n) x)) (M.evalFrom s (List.take (↑m) x))
    hle : LE.le ↑n ↑m
    hm : LE.le (↑m) (Fintype.card σ)
    hq : Eq (M.evalFrom (M.evalFrom s (List.take (↑n) (List.take (↑m) x))) (List.d …
    ⊢ And (Eq (M.evalFrom (M.evalFrom s (List.take (↑n) (List.take (↑m) x))) (List …
  -/
  use hq
  rwa [← hq, ← evalFrom_of_append, ← evalFrom_of_append, ← List.append_assoc,
    List.take_append_drop, List.take_append_drop]


theorem evalFrom_of_pow {x y : List α} {s : σ} (hx : M.evalFrom s x = s)
    (hy : y ∈ ({x} : Language α)∗) : M.evalFrom s y = s := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    x y : List α
    s : σ
    hx : Eq (M.evalFrom s x) s
    hy : Membership.mem (KStar.kstar (Singleton.singleton x)) y
    ⊢ Eq (M.evalFrom s y) s
  -/
  rw [Language.mem_kstar] at hy
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    x y : List α
    s : σ
    hx : Eq (M.evalFrom s x) s
    hy : Exists fun L => And (Eq y L.flatten) (∀ (y : List α), Membership.mem L y  …
    ⊢ Eq (M.evalFrom s y) s
  -/
  rcases hy with ⟨S, rfl, hS⟩
  /-
    case intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    x : List α
    s : σ
    hx : Eq (M.evalFrom s x) s
    S : List (List α)
    hS : ∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton  …
    ⊢ Eq (M.evalFrom s S.flatten) s
  -/
  induction' S with a S ih
    /-
      case intro.intro.nil
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      hS : ∀ (y : List α), Membership.mem List.nil y → Membership.mem (Singleton.sin …
      ⊢ Eq (M.evalFrom s List.nil.flatten) s
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.cons
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      a : List α
      S : List (List α)
      ih : (∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton …
      hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem (Single …
      ⊢ Eq (M.evalFrom s (List.cons a S).flatten) s
    -/
  · have ha := hS a (List.mem_cons_self _ _)
    /-
      case intro.intro.cons
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      a : List α
      S : List (List α)
      ih : (∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton …
      hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem (Single …
      ha : Membership.mem (Singleton.singleton x) a
      ⊢ Eq (M.evalFrom s (List.cons a S).flatten) s
    -/
    rw [Set.mem_singleton_iff] at ha
    /-
      case intro.intro.cons
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      a : List α
      S : List (List α)
      ih : (∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton …
      hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem (Single …
      ha : Eq a x
      ⊢ Eq (M.evalFrom s (List.cons a S).flatten) s
    -/
    rw [List.flatten, evalFrom_of_append, ha, hx]
    /-
      case intro.intro.cons
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      a : List α
      S : List (List α)
      ih : (∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton …
      hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem (Single …
      ha : Eq a x
      ⊢ Eq (M.evalFrom s S.flatten) s
    -/
    apply ih
    /-
      case intro.intro.cons
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      a : List α
      S : List (List α)
      ih : (∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton …
      hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem (Single …
      ha : Eq a x
      ⊢ ∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton x) y
    -/
    intro z hz
    /-
      case intro.intro.cons
      α : Type u
      σ : Type v
      M : DFA α σ
      x : List α
      s : σ
      hx : Eq (M.evalFrom s x) s
      a : List α
      S : List (List α)
      ih : (∀ (y : List α), Membership.mem S y → Membership.mem (Singleton.singleton …
      hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem (Single …
      ha : Eq a x
      z : List α
      hz : Membership.mem S z
      ⊢ Membership.mem (Singleton.singleton x) z
    -/
    exact hS z (List.mem_cons_of_mem a hz)
    /-
      🎉 no goals
    -/


theorem pumping_lemma [Fintype σ] {x : List α} (hx : x ∈ M.accepts)
    (hlen : Fintype.card σ ≤ List.length x) :
    ∃ a b c,
      x = a ++ b ++ c ∧
        a.length + b.length ≤ Fintype.card σ ∧ b ≠ [] ∧ {a} * {b}∗ * {c} ≤ M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx : Membership.mem M.accepts x
    hlen : LE.le (Fintype.card σ) x.length
    ⊢ Exists fun a => Exists fun b => Exists fun c => And (Eq x (HAppend.hAppend ( …
  -/
  obtain ⟨_, a, b, c, hx, hlen, hnil, rfl, hb, hc⟩ := M.evalFrom_split (s := M.start) hlen rfl
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    ⊢ Exists fun a => Exists fun b => Exists fun c => And (Eq x (HAppend.hAppend ( …
  -/
  use a, b, c, hx, hlen, hnil
  /-
    case right
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    ⊢ LE.le (HMul.hMul (HMul.hMul (Singleton.singleton a) (KStar.kstar (Singleton. …
  -/
  intro y hy
  /-
    case right
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    y : List α
    hy : Membership.mem (HMul.hMul (HMul.hMul (Singleton.singleton a) (KStar.kstar …
    ⊢ Membership.mem M.accepts y
  -/
  rw [Language.mem_mul] at hy
  /-
    case right
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    y : List α
    hy : Exists fun a_1 => And (Membership.mem (HMul.hMul (Singleton.singleton a)  …
    ⊢ Membership.mem M.accepts y
  -/
  rcases hy with ⟨ab, hab, c', hc', rfl⟩
  /-
    case right.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    ab : List α
    hab : Membership.mem (HMul.hMul (Singleton.singleton a) (KStar.kstar (Singleto …
    c' : List α
    hc' : Membership.mem (Singleton.singleton c) c'
    ⊢ Membership.mem M.accepts (HAppend.hAppend ab c')
  -/
  rw [Language.mem_mul] at hab
  /-
    case right.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    ab : List α
    hab : Exists fun a_1 => And (Membership.mem (Singleton.singleton a) a_1) (Exis …
    c' : List α
    hc' : Membership.mem (Singleton.singleton c) c'
    ⊢ Membership.mem M.accepts (HAppend.hAppend ab c')
  -/
  rcases hab with ⟨a', ha', b', hb', rfl⟩
  /-
    case right.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    c' : List α
    hc' : Membership.mem (Singleton.singleton c) c'
    a' : List α
    ha' : Membership.mem (Singleton.singleton a) a'
    b' : List α
    hb' : Membership.mem (KStar.kstar (Singleton.singleton b)) b'
    ⊢ Membership.mem M.accepts (HAppend.hAppend (HAppend.hAppend a' b') c')
  -/
  rw [Set.mem_singleton_iff] at ha' hc'
  /-
    case right.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    a b c : List α
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a b) c)
    hlen : LE.le (HAdd.hAdd a.length b.length) (Fintype.card σ)
    hnil : Ne b List.nil
    hb : Eq (M.evalFrom (M.evalFrom M.start a) b) (M.evalFrom M.start a)
    hc : Eq (M.evalFrom (M.evalFrom M.start a) c) (M.evalFrom M.start x)
    c' : List α
    hc' : Eq c' c
    a' : List α
    ha' : Eq a' a
    b' : List α
    hb' : Membership.mem (KStar.kstar (Singleton.singleton b)) b'
    ⊢ Membership.mem M.accepts (HAppend.hAppend (HAppend.hAppend a' b') c')
  -/
  substs ha' hc'
  /-
    case right.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    b : List α
    hnil : Ne b List.nil
    c' a' b' : List α
    hb' : Membership.mem (KStar.kstar (Singleton.singleton b)) b'
    hlen : LE.le (HAdd.hAdd a'.length b.length) (Fintype.card σ)
    hb : Eq (M.evalFrom (M.evalFrom M.start a') b) (M.evalFrom M.start a')
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a' b) c')
    hc : Eq (M.evalFrom (M.evalFrom M.start a') c') (M.evalFrom M.start x)
    ⊢ Membership.mem M.accepts (HAppend.hAppend (HAppend.hAppend a' b') c')
  -/
  have h := M.evalFrom_of_pow hb hb'
  /-
    case right.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    σ : Type v
    M : DFA α σ
    inst✝ : Fintype σ
    x : List α
    hx✝ : Membership.mem M.accepts x
    hlen✝ : LE.le (Fintype.card σ) x.length
    b : List α
    hnil : Ne b List.nil
    c' a' b' : List α
    hb' : Membership.mem (KStar.kstar (Singleton.singleton b)) b'
    hlen : LE.le (HAdd.hAdd a'.length b.length) (Fintype.card σ)
    hb : Eq (M.evalFrom (M.evalFrom M.start a') b) (M.evalFrom M.start a')
    hx : Eq x (HAppend.hAppend (HAppend.hAppend a' b) c')
    hc : Eq (M.evalFrom (M.evalFrom M.start a') c') (M.evalFrom M.start x)
    h : Eq (M.evalFrom (M.evalFrom M.start a') b') (M.evalFrom M.start a')
    ⊢ Membership.mem M.accepts (HAppend.hAppend (HAppend.hAppend a' b') c')
  -/
  rwa [mem_accepts, eval, evalFrom_of_append, evalFrom_of_append, h, hc]
  /-
    🎉 no goals
  -/


/--
`M.comap f` pulls back the alphabet of `M` along `f`. In other words, it applies `f` to the input
before passing it to `M`.
-/
@[simps]
def comap (f : α' → α) (M : DFA α σ) : DFA α' σ where
  step s a := M.step s (f a)
  start := M.start
  accept := M.accept


@[simp]
theorem comap_id : M.comap id = M := rfl


@[simp]
theorem evalFrom_comap (f : α' → α) (s : σ) (x : List α') :
    (M.comap f).evalFrom s x = M.evalFrom s (x.map f) := by
  induction x using List.list_reverse_induction with
  | base => simp
  | ind x a ih => simp [ih]


@[simp]
theorem eval_comap (f : α' → α) (x : List α') : (M.comap f).eval x = M.eval (x.map f) := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    α' : Type u_1
    f : α' → α
    x : List α'
    ⊢ Eq ((DFA.comap f M).eval x) (M.eval (List.map f x))
  -/
  simp [eval]
  /-
    🎉 no goals
  -/


@[simp]
theorem accepts_comap (f : α' → α) : (M.comap f).accepts = List.map f ⁻¹' M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    α' : Type u_1
    f : α' → α
    ⊢ Eq (DFA.comap f M).accepts (Set.preimage (List.map f) M.accepts)
  -/
  ext x
  conv =>
    rhs
    rw [Set.mem_preimage, mem_accepts]
  /-
    case h
    α : Type u
    σ : Type v
    M : DFA α σ
    α' : Type u_1
    f : α' → α
    x : List α'
    ⊢ Iff (Membership.mem (DFA.comap f M).accepts x) (Membership.mem M.accept (M.e …
  -/
  simp [mem_accepts]
  /-
    🎉 no goals
  -/


/-- Lifts an equivalence on states to an equivalence on DFAs. -/
@[simps apply_step apply_start apply_accept]
def reindex (g : σ ≃ σ') : DFA α σ ≃ DFA α σ' where
  toFun M := {
    step := fun s a => g (M.step (g.symm s) a)
    start := g M.start
    accept := g.symm ⁻¹' M.accept
  }
  invFun M := {
    step := fun s a => g.symm (M.step (g s) a)
    start := g.symm M.start
    accept := g ⁻¹' M.accept
  }
                   /-
                     α : Type u
                     σ : Type v
                     M✝ : DFA α σ
                     α' : Type u_1
                     σ' : Type u_2
                     g : Equiv σ σ'
                     M : DFA α σ
                     ⊢ Eq ((fun M => { step := fun s a => g.symm (M.step (g s) a), start := g.symm  …
                   -/
  left_inv M := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u
                      σ : Type v
                      M✝ : DFA α σ
                      α' : Type u_1
                      σ' : Type u_2
                      g : Equiv σ σ'
                      M : DFA α σ'
                      ⊢ Eq ((fun M => { step := fun s a => g (M.step (g.symm s) a), start := g M.sta …
                    -/
  right_inv M := by simp
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem reindex_refl : reindex (Equiv.refl σ) M = M := rfl


@[simp]
theorem symm_reindex (g : σ ≃ σ') : (reindex (α := α) g).symm = reindex g.symm := rfl


@[simp]
theorem evalFrom_reindex (g : σ ≃ σ') (s : σ') (x : List α) :
    (reindex g M).evalFrom s x = g (M.evalFrom (g.symm s) x) := by
  induction x using List.list_reverse_induction with
  | base => simp
  | ind x a ih => simp [ih]


@[simp]
theorem eval_reindex (g : σ ≃ σ') (x : List α) : (reindex g M).eval x = g (M.eval x) := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    σ' : Type u_2
    g : Equiv σ σ'
    x : List α
    ⊢ Eq (((DFA.reindex g) M).eval x) (g (M.eval x))
  -/
  simp [eval]
  /-
    🎉 no goals
  -/


@[simp]
theorem accepts_reindex (g : σ ≃ σ') : (reindex g M).accepts = M.accepts := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    σ' : Type u_2
    g : Equiv σ σ'
    ⊢ Eq ((DFA.reindex g) M).accepts M.accepts
  -/
  ext x
  /-
    case h
    α : Type u
    σ : Type v
    M : DFA α σ
    σ' : Type u_2
    g : Equiv σ σ'
    x : List α
    ⊢ Iff (Membership.mem ((DFA.reindex g) M).accepts x) (Membership.mem M.accepts …
  -/
  simp [mem_accepts]
  /-
    🎉 no goals
  -/


theorem comap_reindex (f : α' → α) (g : σ ≃ σ') :
    (reindex g M).comap f = reindex g (M.comap f) := by
  /-
    α : Type u
    σ : Type v
    M : DFA α σ
    α' : Type u_1
    σ' : Type u_2
    f : α' → α
    g : Equiv σ σ'
    ⊢ Eq (DFA.comap f ((DFA.reindex g) M)) ((DFA.reindex g) (DFA.comap f M))
  -/
  simp [comap, reindex]
  /-
    🎉 no goals
  -/


/-- A regular language is a language that is defined by a DFA with finite states. -/
def Language.IsRegular {T : Type u} (L : Language T) : Prop :=
  ∃ σ : Type, ∃ _ : Fintype σ, ∃ M : DFA T σ, M.accepts = L


