/-- The type of codes for primitive recursive functions. Unlike `Nat.Partrec.Code`, this uses a set
of operations on `List ℕ`. See `Code.eval` for a description of the behavior of the primitives. -/
inductive Code
  | zero'
  | succ
  | tail
  | cons : Code → Code → Code
  | comp : Code → Code → Code
  | case : Code → Code → Code
  | fix : Code → Code
  deriving DecidableEq, Inhabited


/-- The semantics of the `Code` primitives, as partial functions `List ℕ →. List ℕ`. By convention
we functions that return a single result return a singleton `[n]`, or in some cases `n :: v` where
`v` will be ignored by a subsequent function.

* `zero'` appends a `0` to the input. That is, `zero' v = 0 :: v`.
* `succ` returns the successor of the head of the input, defaulting to zero if there is no head:
  * `succ [] = [1]`
  * `succ (n :: v) = [n + 1]`
* `tail` returns the tail of the input
  * `tail [] = []`
  * `tail (n :: v) = v`
* `cons f fs` calls `f` and `fs` on the input and conses the results:
  * `cons f fs v = (f v).head :: fs v`
* `comp f g` calls `f` on the output of `g`:
  * `comp f g v = f (g v)`
* `case f g` cases on the head of the input, calling `f` or `g` depending on whether it is zero or
  a successor (similar to `Nat.casesOn`).
  * `case f g [] = f []`
  * `case f g (0 :: v) = f v`
  * `case f g (n+1 :: v) = g (n :: v)`
* `fix f` calls `f` repeatedly, using the head of the result of `f` to decide whether to call `f`
  again or finish:
  * `fix f v = []` if `f v = []`
  * `fix f v = w` if `f v = 0 :: w`
  * `fix f v = fix f w` if `f v = n+1 :: w` (the exact value of `n` is discarded)
-/
def Code.eval : Code → List ℕ →. List ℕ
  | Code.zero' => fun v => pure (0 :: v)
  | Code.succ => fun v => pure [v.headI.succ]
  | Code.tail => fun v => pure v.tail
  | Code.cons f fs => fun v => do
    let n ← Code.eval f v
    let ns ← Code.eval fs v
    pure (n.headI :: ns)
  | Code.comp f g => fun v => g.eval v >>= f.eval
  | Code.case f g => fun v => v.headI.rec (f.eval v.tail) fun y _ => g.eval (y::v.tail)
  | Code.fix f =>
    PFun.fix fun v => (f.eval v).map fun v => if v.headI = 0 then Sum.inl v.tail else Sum.inr v.tail


@[simp]
                                                               /-
                                                                 ⊢ Eq Turing.ToPartrec.Code.zero'.eval fun v => Pure.pure (List.cons 0 v)
                                                               -/
theorem zero'_eval : zero'.eval = fun v => pure (0 :: v) := by simp [eval]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
                                                                   /-
                                                                     ⊢ Eq Turing.ToPartrec.Code.succ.eval fun v => Pure.pure (List.cons v.headI.suc …
                                                                   -/
theorem succ_eval : succ.eval = fun v => pure [v.headI.succ] := by simp [eval]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
                                                           /-
                                                             ⊢ Eq Turing.ToPartrec.Code.tail.eval fun v => Pure.pure v.tail
                                                           -/
theorem tail_eval : tail.eval = fun v => pure v.tail := by simp [eval]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem cons_eval (f fs) : (cons f fs).eval = fun v => do {
    let n ← Code.eval f v
    let ns ← Code.eval fs v
                                 /-
                                   f fs : Turing.ToPartrec.Code
                                   ⊢ Eq (f.cons fs).eval fun v => Bind.bind (f.eval v) fun n => Bind.bind (fs.eva …
                                 -/
    pure (n.headI :: ns) } := by simp [eval]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
                                                                               /-
                                                                                 f g : Turing.ToPartrec.Code
                                                                                 ⊢ Eq (f.comp g).eval fun v => Bind.bind (g.eval v) f.eval
                                                                               -/
theorem comp_eval (f g) : (comp f g).eval = fun v => g.eval v >>= f.eval := by simp [eval]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem case_eval (f g) :
    (case f g).eval = fun v => v.headI.rec (f.eval v.tail) fun y _ => g.eval (y::v.tail) := by
  /-
    f g : Turing.ToPartrec.Code
    ⊢ Eq (f.case g).eval fun v => Nat.rec (f.eval v.tail) (fun y x => g.eval (List …
  -/
  simp [eval]
  /-
    🎉 no goals
  -/


@[simp]
theorem fix_eval (f) : (fix f).eval =
    PFun.fix fun v => (f.eval v).map fun v =>
      if v.headI = 0 then Sum.inl v.tail else Sum.inr v.tail := by
  /-
    f : Turing.ToPartrec.Code
    ⊢ Eq f.fix.eval (PFun.fix fun v => Part.map (fun v => ite (Eq v.headI 0) (Sum. …
  -/
  simp [eval]
  /-
    🎉 no goals
  -/


/-- `nil` is the constant nil function: `nil v = []`. -/
def nil : Code :=
  tail.comp succ


@[simp]
                                                  /-
                                                    v : List Nat
                                                    ⊢ Eq (Turing.ToPartrec.Code.nil.eval v) (Pure.pure List.nil)
                                                  -/
theorem nil_eval (v) : nil.eval v = pure [] := by simp [nil]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- `id` is the identity function: `id v = v`. -/
def id : Code :=
  tail.comp zero'


@[simp]
                                               /-
                                                 v : List Nat
                                                 ⊢ Eq (Turing.ToPartrec.Code.id.eval v) (Pure.pure v)
                                               -/
theorem id_eval (v) : id.eval v = pure v := by simp [id]
                                               /-
                                                 🎉 no goals
                                               -/


/-- `head` gets the head of the input list: `head [] = [0]`, `head (n :: v) = [n]`. -/
def head : Code :=
  cons id nil


@[simp]
                                                           /-
                                                             v : List Nat
                                                             ⊢ Eq (Turing.ToPartrec.Code.head.eval v) (Pure.pure (List.cons v.headI List.ni …
                                                           -/
theorem head_eval (v) : head.eval v = pure [v.headI] := by simp [head]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- `zero` is the constant zero function: `zero v = [0]`. -/
def zero : Code :=
  cons zero' nil


@[simp]
                                                     /-
                                                       v : List Nat
                                                       ⊢ Eq (Turing.ToPartrec.Code.zero.eval v) (Pure.pure (List.cons 0 List.nil))
                                                     -/
theorem zero_eval (v) : zero.eval v = pure [0] := by simp [zero]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- `pred` returns the predecessor of the head of the input:
`pred [] = [0]`, `pred (0 :: v) = [0]`, `pred (n+1 :: v) = [n]`. -/
def pred : Code :=
  case zero head


@[simp]
theorem pred_eval (v) : pred.eval v = pure [v.headI.pred] := by
  /-
    v : List Nat
    ⊢ Eq (Turing.ToPartrec.Code.pred.eval v) (Pure.pure (List.cons v.headI.pred Li …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  simp [pred]; cases v.headI <;> simp
                                 /-
                                   🎉 no goals
                                 -/


/-- `rfind f` performs the function of the `rfind` primitive of partial recursive functions.
`rfind f v` returns the smallest `n` such that `(f (n :: v)).head = 0`.

It is implemented as:

    rfind f v = pred (fix (fun (n::v) => f (n::v) :: n+1 :: v) (0 :: v))

The idea is that the initial state is `0 :: v`, and the `fix` keeps `n :: v` as its internal state;
it calls `f (n :: v)` as the exit test and `n+1 :: v` as the next state. At the end we get
`n+1 :: v` where `n` is the desired output, and `pred (n+1 :: v) = [n]` returns the result.
 -/
def rfind (f : Code) : Code :=
  comp pred <| comp (fix <| cons f <| cons succ tail) zero'


/-- `prec f g` implements the `prec` (primitive recursion) operation of partial recursive
functions. `prec f g` evaluates as:

* `prec f g [] = [f []]`
* `prec f g (0 :: v) = [f v]`
* `prec f g (n+1 :: v) = [g (n :: prec f g (n :: v) :: v)]`

It is implemented as:

    G (a :: b :: IH :: v) = (b :: a+1 :: b-1 :: g (a :: IH :: v) :: v)
    F (0 :: f_v :: v) = (f_v :: v)
    F (n+1 :: f_v :: v) = (fix G (0 :: n :: f_v :: v)).tail.tail
    prec f g (a :: v) = [(F (a :: f v :: v)).head]

Because `fix` always evaluates its body at least once, we must special case the `0` case to avoid
calling `g` more times than necessary (which could be bad if `g` diverges). If the input is
`0 :: v`, then `F (0 :: f v :: v) = (f v :: v)` so we return `[f v]`. If the input is `n+1 :: v`,
we evaluate the function from the bottom up, with initial state `0 :: n :: f v :: v`. The first
number counts up, providing arguments for the applications to `g`, while the second number counts
down, providing the exit condition (this is the initial `b` in the return value of `G`, which is
stripped by `fix`). After the `fix` is complete, the final state is `n :: 0 :: res :: v` where
`res` is the desired result, and the rest reduces this to `[res]`. -/
def prec (f g : Code) : Code :=
  let G :=
    cons tail <|
      cons succ <|
        cons (comp pred tail) <|
          cons (comp g <| cons id <| comp tail tail) <| comp tail <| comp tail tail
  let F := case id <| comp (comp (comp tail tail) (fix G)) zero'
  cons (comp F (cons head <| cons (comp f tail) tail)) nil


theorem exists_code.comp {m n} {f : List.Vector ℕ n →. ℕ} {g : Fin n → List.Vector ℕ m →. ℕ}
    (hf : ∃ c : Code, ∀ v : List.Vector ℕ n, c.eval v.1 = pure <$> f v)
    (hg : ∀ i, ∃ c : Code, ∀ v : List.Vector ℕ m, c.eval v.1 = pure <$> g i v) :
    ∃ c : Code, ∀ v : List.Vector ℕ m,
      c.eval v.1 = pure <$> ((List.Vector.mOfFn fun i => g i v) >>= f) := by
  rsuffices ⟨cg, hg⟩ :
    ∃ c : Code, ∀ v : List.Vector ℕ m,
      c.eval v.1 = Subtype.val <$> List.Vector.mOfFn fun i => g i v
    /-
      case intro
      m n : Nat
      f : PFun (List.Vector Nat n) Nat
      g : Fin n → PFun (List.Vector Nat m) Nat
      hf : Exists fun c => ∀ (v : List.Vector Nat n), Eq (c.eval ↑v) (Functor.map Pu …
      hg✝ : ∀ (i : Fin n), Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v) …
      cg : Turing.ToPartrec.Code
      hg : ∀ (v : List.Vector Nat m), Eq (cg.eval ↑v) (Functor.map Subtype.val (List …
      ⊢ Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v) (Functor.map Pure. …
    -/
  · obtain ⟨cf, hf⟩ := hf
    exact
      ⟨cf.comp cg, fun v => by
        simp [hg, hf, map_bind, seq_bind_eq, Function.comp_def]
        rfl⟩
  /-
    m n : Nat
    f : PFun (List.Vector Nat n) Nat
    g : Fin n → PFun (List.Vector Nat m) Nat
    hf : Exists fun c => ∀ (v : List.Vector Nat n), Eq (c.eval ↑v) (Functor.map Pu …
    hg : ∀ (i : Fin n), Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v)  …
    ⊢ Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v) (Functor.map Subty …
  -/
  clear hf f; induction' n with n IH
    /-
      case zero
      m : Nat
      g : Fin 0 → PFun (List.Vector Nat m) Nat
      hg : ∀ (i : Fin 0), Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v)  …
      ⊢ Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v) (Functor.map Subty …
    -/
  · exact ⟨nil, fun v => by simp [Vector.mOfFn, Bind.bind]; rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      m n : Nat
      IH : ∀ {g : Fin n → PFun (List.Vector Nat m) Nat}, (∀ (i : Fin n), Exists fun  …
      g : Fin (HAdd.hAdd n 1) → PFun (List.Vector Nat m) Nat
      hg : ∀ (i : Fin (HAdd.hAdd n 1)), Exists fun c => ∀ (v : List.Vector Nat m), E …
      ⊢ Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v) (Functor.map Subty …
    -/
  · obtain ⟨cg, hg₁⟩ := hg 0
    /-
      case succ.intro
      m n : Nat
      IH : ∀ {g : Fin n → PFun (List.Vector Nat m) Nat}, (∀ (i : Fin n), Exists fun  …
      g : Fin (HAdd.hAdd n 1) → PFun (List.Vector Nat m) Nat
      hg : ∀ (i : Fin (HAdd.hAdd n 1)), Exists fun c => ∀ (v : List.Vector Nat m), E …
      cg : Turing.ToPartrec.Code
      hg₁ : ∀ (v : List.Vector Nat m), Eq (cg.eval ↑v) (Functor.map Pure.pure (g 0 v))
      ⊢ Exists fun c => ∀ (v : List.Vector Nat m), Eq (c.eval ↑v) (Functor.map Subty …
    -/
    obtain ⟨cl, hl⟩ := IH fun i => hg i.succ
    exact
      ⟨cons cg cl, fun v => by
        simp [Vector.mOfFn, hg₁, map_bind, seq_bind_eq, bind_assoc, (· ∘ ·), hl]
        rfl⟩


theorem exists_code {n} {f : List.Vector ℕ n →. ℕ} (hf : Nat.Partrec' f) :
    ∃ c : Code, ∀ v : List.Vector ℕ n, c.eval v.1 = pure <$> f v := by
  induction hf with
  | prim hf =>
    induction hf with
    | zero => exact ⟨zero', fun ⟨[], _⟩ => rfl⟩
    | succ => exact ⟨succ, fun ⟨[v], _⟩ => rfl⟩
    | get i =>
      refine Fin.succRec (fun n => ?_) (fun n i IH => ?_) i
      · exact ⟨head, fun ⟨List.cons a as, _⟩ => by simp [Bind.bind]; rfl⟩
      · obtain ⟨c, h⟩ := IH
        exact ⟨c.comp tail, fun v => by simpa [← Vector.get_tail, Bind.bind] using h v.tail⟩
    | comp g hf hg IHf IHg =>
      simpa [Part.bind_eq_bind] using exists_code.comp IHf IHg
    | @prec n f g _ _ IHf IHg =>
      obtain ⟨cf, hf⟩ := IHf
      obtain ⟨cg, hg⟩ := IHg
      simp only [Part.map_eq_map, Part.map_some, PFun.coe_val] at hf hg
      refine ⟨prec cf cg, fun v => ?_⟩
      rw [← v.cons_head_tail]
      specialize hf v.tail
      replace hg := fun a b => hg (a ::ᵥ b ::ᵥ v.tail)
      simp only [Vector.cons_val, Vector.tail_val] at hf hg
      simp only [Part.map_eq_map, Part.map_some, Vector.cons_val, Vector.tail_cons,
        Vector.head_cons, PFun.coe_val, Vector.tail_val]
      simp only [← Part.pure_eq_some] at hf hg ⊢
      induction' v.head with n _ <;>
        simp [prec, hf, Part.bind_assoc, ← Part.bind_some_eq_map, Part.bind_some,
          show ∀ x, pure x = [x] from fun _ => rfl, Bind.bind, Functor.map]
      suffices ∀ a b, a + b = n →
        (n.succ :: 0 ::
          g (n ::ᵥ Nat.rec (f v.tail) (fun y IH => g (y ::ᵥ IH ::ᵥ v.tail)) n ::ᵥ v.tail) ::
              v.val.tail : List ℕ) ∈
          PFun.fix
            (fun v : List ℕ => Part.bind (cg.eval (v.headI :: v.tail.tail))
              (fun x => Part.some (if v.tail.headI = 0
                then Sum.inl
                  (v.headI.succ :: v.tail.headI.pred :: x.headI :: v.tail.tail.tail : List ℕ)
                else Sum.inr
                  (v.headI.succ :: v.tail.headI.pred :: x.headI :: v.tail.tail.tail))))
            (a :: b :: Nat.rec (f v.tail) (fun y IH => g (y ::ᵥ IH ::ᵥ v.tail)) a :: v.val.tail) by
        erw [Part.eq_some_iff.2 (this 0 n (zero_add n))]
        simp only [List.headI, Part.bind_some, List.tail_cons]
      intro a b e
      induction' b with b IH generalizing a
      · refine PFun.mem_fix_iff.2 (Or.inl <| Part.eq_some_iff.1 ?_)
        simp only [hg, ← e, Part.bind_some, List.tail_cons, pure]
        rfl
      · refine PFun.mem_fix_iff.2 (Or.inr ⟨_, ?_, IH (a + 1) (by rwa [add_right_comm])⟩)
        simp only [hg, eval, Part.bind_some, Nat.rec_add_one, List.tail_nil, List.tail_cons, pure]
        exact Part.mem_some_iff.2 rfl
  | comp g _ _ IHf IHg => exact exists_code.comp IHf IHg
  | @rfind n f _ IHf =>
    obtain ⟨cf, hf⟩ := IHf; refine ⟨rfind cf, fun v => ?_⟩
    replace hf := fun a => hf (a ::ᵥ v)
    simp only [Part.map_eq_map, Part.map_some, Vector.cons_val, PFun.coe_val,
      show ∀ x, pure x = [x] from fun _ => rfl] at hf ⊢
    refine Part.ext fun x => ?_
    simp only [rfind, Part.bind_eq_bind, Part.pure_eq_some, Part.map_eq_map, Part.bind_some,
      exists_prop, cons_eval, comp_eval, fix_eval, tail_eval, succ_eval, zero'_eval,
      List.headI_nil, List.headI_cons, pred_eval, Part.map_some, false_eq_decide_iff,
      Part.mem_bind_iff, List.length, Part.mem_map_iff, Nat.mem_rfind, List.tail_nil,
      List.tail_cons, true_eq_decide_iff, Part.mem_some_iff, Part.map_bind]
    constructor
    · rintro ⟨v', h1, rfl⟩
      suffices ∀ v₁ : List ℕ, v' ∈ PFun.fix
        (fun v => (cf.eval v).bind fun y => Part.some <|
          if y.headI = 0 then Sum.inl (v.headI.succ :: v.tail)
            else Sum.inr (v.headI.succ :: v.tail)) v₁ →
        ∀ n, (v₁ = n :: v.val) → (∀ m < n, ¬f (m ::ᵥ v) = 0) →
          ∃ a : ℕ,
            (f (a ::ᵥ v) = 0 ∧ ∀ {m : ℕ}, m < a → ¬f (m ::ᵥ v) = 0) ∧ [a] = [v'.headI.pred]
        by exact this _ h1 0 rfl (by rintro _ ⟨⟩)
      clear h1
      intro v₀ h1
      refine PFun.fixInduction h1 fun v₁ h2 IH => ?_
      clear h1
      rintro n rfl hm
      have := PFun.mem_fix_iff.1 h2
      simp only [hf, Part.bind_some] at this
      split_ifs at this with h
      · simp only [List.headI_nil, List.headI_cons, exists_false, or_false, Part.mem_some_iff,
          List.tail_cons, false_and, Sum.inl.injEq, reduceCtorEq] at this
        subst this
        exact ⟨_, ⟨h, @(hm)⟩, rfl⟩
      · refine IH (n.succ::v.val) (by simp_all) _ rfl fun m h' => ?_
        obtain h | rfl := Nat.lt_succ_iff_lt_or_eq.1 h'
        exacts [hm _ h, h]
    · rintro ⟨n, ⟨hn, hm⟩, rfl⟩
      refine ⟨n.succ::v.1, ?_, rfl⟩
      have : (n.succ::v.1 : List ℕ) ∈
        PFun.fix (fun v =>
          (cf.eval v).bind fun y =>
            Part.some <|
              if y.headI = 0 then Sum.inl (v.headI.succ :: v.tail)
                else Sum.inr (v.headI.succ :: v.tail))
            (n::v.val) :=
        PFun.mem_fix_iff.2 (Or.inl (by simp [hf, hn]))
      generalize (n.succ :: v.1 : List ℕ) = w at this ⊢
      clear hn
      induction n with
      | zero => exact this
      | succ n IH =>
        refine IH (fun {m} h' => hm (Nat.lt_succ_of_lt h'))
          (PFun.mem_fix_iff.2 (Or.inr ⟨_, ?_, this⟩))
        simp only [hf, hm n.lt_succ_self, Part.bind_some, List.headI, eq_self_iff_true, if_false,
          Part.mem_some_iff, and_self_iff, List.tail_cons]


/-- The type of continuations, built up during evaluation of a `Code` expression. -/
inductive Cont
  | halt
  | cons₁ : Code → List ℕ → Cont → Cont
  | cons₂ : List ℕ → Cont → Cont
  | comp : Code → Cont → Cont
  | fix : Code → Cont → Cont
  deriving Inhabited


/-- The semantics of a continuation. -/
def Cont.eval : Cont → List ℕ →. List ℕ
  | Cont.halt => pure
  | Cont.cons₁ fs as k => fun v => do
    let ns ← Code.eval fs as
    Cont.eval k (v.headI :: ns)
  | Cont.cons₂ ns k => fun v => Cont.eval k (ns.headI :: v)
  | Cont.comp f k => fun v => Code.eval f v >>= Cont.eval k
  | Cont.fix f k => fun v => if v.headI = 0 then k.eval v.tail else f.fix.eval v.tail >>= k.eval


/-- The set of configurations of the machine:

* `halt v`: The machine is about to stop and `v : List ℕ` is the result.
* `ret k v`: The machine is about to pass `v : List ℕ` to continuation `k : Cont`.

We don't have a state corresponding to normal evaluation because these are evaluated immediately
to a `ret` "in zero steps" using the `stepNormal` function. -/
inductive Cfg
  | halt : List ℕ → Cfg
  | ret : Cont → List ℕ → Cfg
  deriving Inhabited


/-- Evaluating `c : Code` in a continuation `k : Cont` and input `v : List ℕ`. This goes by
recursion on `c`, building an augmented continuation and a value to pass to it.

* `zero' v = 0 :: v` evaluates immediately, so we return it to the parent continuation
* `succ v = [v.headI.succ]` evaluates immediately, so we return it to the parent continuation
* `tail v = v.tail` evaluates immediately, so we return it to the parent continuation
* `cons f fs v = (f v).headI :: fs v` requires two sub-evaluations, so we evaluate
  `f v` in the continuation `k (_.headI :: fs v)` (called `Cont.cons₁ fs v k`)
* `comp f g v = f (g v)` requires two sub-evaluations, so we evaluate
  `g v` in the continuation `k (f _)` (called `Cont.comp f k`)
* `case f g v = v.head.casesOn (f v.tail) (fun n => g (n :: v.tail))` has the information needed
  to evaluate the case statement, so we do that and transition to either
  `f v` or `g (n :: v.tail)`.
* `fix f v = let v' := f v; if v'.headI = 0 then k v'.tail else fix f v'.tail`
  needs to first evaluate `f v`, so we do that and leave the rest for the continuation (called
  `Cont.fix f k`)
-/
def stepNormal : Code → Cont → List ℕ → Cfg
  | Code.zero' => fun k v => Cfg.ret k (0::v)
  | Code.succ => fun k v => Cfg.ret k [v.headI.succ]
  | Code.tail => fun k v => Cfg.ret k v.tail
  | Code.cons f fs => fun k v => stepNormal f (Cont.cons₁ fs v k) v
  | Code.comp f g => fun k v => stepNormal g (Cont.comp f k) v
  | Code.case f g => fun k v =>
    v.headI.rec (stepNormal f k v.tail) fun y _ => stepNormal g k (y::v.tail)
  | Code.fix f => fun k v => stepNormal f (Cont.fix f k) v


/-- Evaluating a continuation `k : Cont` on input `v : List ℕ`. This is the second part of
evaluation, when we receive results from continuations built by `stepNormal`.

* `Cont.halt v = v`, so we are done and transition to the `Cfg.halt v` state
* `Cont.cons₁ fs as k v = k (v.headI :: fs as)`, so we evaluate `fs as` now with the continuation
  `k (v.headI :: _)` (called `cons₂ v k`).
* `Cont.cons₂ ns k v = k (ns.headI :: v)`, where we now have everything we need to evaluate
  `ns.headI :: v`, so we return it to `k`.
* `Cont.comp f k v = k (f v)`, so we call `f v` with `k` as the continuation.
* `Cont.fix f k v = k (if v.headI = 0 then k v.tail else fix f v.tail)`, where `v` is a value,
  so we evaluate the if statement and either call `k` with `v.tail`, or call `fix f v` with `k` as
  the continuation (which immediately calls `f` with `Cont.fix f k` as the continuation).
-/
def stepRet : Cont → List ℕ → Cfg
  | Cont.halt, v => Cfg.halt v
  | Cont.cons₁ fs as k, v => stepNormal fs (Cont.cons₂ v k) as
  | Cont.cons₂ ns k, v => stepRet k (ns.headI :: v)
  | Cont.comp f k, v => stepNormal f k v
  | Cont.fix f k, v => if v.headI = 0 then stepRet k v.tail else stepNormal f (Cont.fix f k) v.tail


/-- If we are not done (in `Cfg.halt` state), then we must be still stuck on a continuation, so
this main loop calls `stepRet` with the new continuation. The overall `step` function transitions
from one `Cfg` to another, only halting at the `Cfg.halt` state. -/
def step : Cfg → Option Cfg
  | Cfg.halt _ => none
  | Cfg.ret k v => some (stepRet k v)


/-- In order to extract a compositional semantics from the sequential execution behavior of
configurations, we observe that continuations have a monoid structure, with `Cont.halt` as the unit
and `Cont.then` as the multiplication. `Cont.then k₁ k₂` runs `k₁` until it halts, and then takes
the result of `k₁` and passes it to `k₂`.

We will not prove it is associative (although it is), but we are instead interested in the
associativity law `k₂ (eval c k₁) = eval c (k₁.then k₂)`. This holds at both the sequential and
compositional levels, and allows us to express running a machine without the ambient continuation
and relate it to the original machine's evaluation steps. In the literature this is usually
where one uses Turing machines embedded inside other Turing machines, but this approach allows us
to avoid changing the ambient type `Cfg` in the middle of the recursion.
-/
def Cont.then : Cont → Cont → Cont
  | Cont.halt => fun k' => k'
  | Cont.cons₁ fs as k => fun k' => Cont.cons₁ fs as (k.then k')
  | Cont.cons₂ ns k => fun k' => Cont.cons₂ ns (k.then k')
  | Cont.comp f k => fun k' => Cont.comp f (k.then k')
  | Cont.fix f k => fun k' => Cont.fix f (k.then k')


theorem Cont.then_eval {k k' : Cont} {v} : (k.then k').eval v = k.eval v >>= k'.eval := by
  induction k generalizing v with
  | halt => simp only [Cont.eval, Cont.then, pure_bind]
  | cons₁ => simp only [Cont.eval, Cont.then, bind_assoc, *]
  | cons₂ => simp only [Cont.eval, Cont.then, *]
  | comp _ _ k_ih => simp only [Cont.eval, Cont.then, bind_assoc, ← k_ih]
  | fix _ _ k_ih =>
    simp only [Cont.eval, Cont.then, *]
    split_ifs <;> [rfl; simp only [← k_ih, bind_assoc]]


/-- The `then k` function is a "configuration homomorphism". Its operation on states is to append
`k` to the continuation of a `Cfg.ret` state, and to run `k` on `v` if we are in the `Cfg.halt v`
state. -/
def Cfg.then : Cfg → Cont → Cfg
  | Cfg.halt v => fun k' => stepRet k' v
  | Cfg.ret k v => fun k' => Cfg.ret (k.then k') v


/-- The `stepNormal` function respects the `then k'` homomorphism. Note that this is an exact
equality, not a simulation; the original and embedded machines move in lock-step until the
embedded machine reaches the halt state. -/
theorem stepNormal_then (c) (k k' : Cont) (v) :
    stepNormal c (k.then k') v = (stepNormal c k v).then k' := by
  induction c generalizing k v with simp only [Cont.then, stepNormal, *]
  | cons c c' ih _ => rw [← ih, Cont.then]
  | comp c c' _ ih' => rw [← ih', Cont.then]
  | case => cases v.headI <;> simp only [Nat.rec_zero]
  | fix c ih => rw [← ih, Cont.then]
  | _ => simp only [Cfg.then]


/-- The `stepRet` function respects the `then k'` homomorphism. Note that this is an exact
equality, not a simulation; the original and embedded machines move in lock-step until the
embedded machine reaches the halt state. -/
theorem stepRet_then {k k' : Cont} {v} : stepRet (k.then k') v = (stepRet k v).then k' := by
  induction k generalizing v with simp only [Cont.then, stepRet, *]
  | cons₁ =>
    rw [← stepNormal_then]
    rfl
  | comp =>
    rw [← stepNormal_then]
  | fix _ _ k_ih =>
    split_ifs
    · rw [← k_ih]
    · rw [← stepNormal_then]
      rfl
  | _ => simp only [Cfg.then]


/-- This is a temporary definition, because we will prove in `code_is_ok` that it always holds.
It asserts that `c` is semantically correct; that is, for any `k` and `v`,
`eval (stepNormal c k v) = eval (Cfg.ret k (Code.eval c v))`, as an equality of partial values
(so one diverges iff the other does).

In particular, we can let `k = Cont.halt`, and then this asserts that `stepNormal c Cont.halt v`
evaluates to `Cfg.halt (Code.eval c v)`. -/
def Code.Ok (c : Code) :=
  ∀ k v, Turing.eval step (stepNormal c k v) =
    Code.eval c v >>= fun v => Turing.eval step (Cfg.ret k v)


theorem Code.Ok.zero {c} (h : Code.Ok c) {v} :
    Turing.eval step (stepNormal c Cont.halt v) = Cfg.halt <$> Code.eval c v := by
  /-
    c : Turing.ToPartrec.Code
    h : c.Ok
    v : List Nat
    ⊢ Eq (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNormal c Turing. …
  -/
  rw [h, ← bind_pure_comp]; congr; funext v
  /-
    case e_a.h
    c : Turing.ToPartrec.Code
    h : c.Ok
    v✝ v : List Nat
    ⊢ Eq (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret Turing.ToPar …
  -/
  exact Part.eq_some_iff.2 (mem_eval.2 ⟨ReflTransGen.single rfl, rfl⟩)
  /-
    🎉 no goals
  -/


theorem stepNormal.is_ret (c k v) : ∃ k' v', stepNormal c k v = Cfg.ret k' v' := by
  induction c generalizing k v with
  | cons _f fs IHf _IHfs => apply IHf
  | comp f _g _IHf IHg => apply IHg
  | case f g IHf IHg =>
    rw [stepNormal]
    simp only []
    cases v.headI <;> [apply IHf; apply IHg]
  | fix f IHf => apply IHf
  | _ => exact ⟨_, _, rfl⟩


theorem cont_eval_fix {f k v} (fok : Code.Ok f) :
    Turing.eval step (stepNormal f (Cont.fix f k) v) =
      f.fix.eval v >>= fun v => Turing.eval step (Cfg.ret k v) := by
  /-
    f : Turing.ToPartrec.Code
    k : Turing.ToPartrec.Cont
    v : List Nat
    fok : f.Ok
    ⊢ Eq (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f (Turing …
  -/
  refine Part.ext fun x => ?_
  /-
    f : Turing.ToPartrec.Code
    k : Turing.ToPartrec.Cont
    v : List Nat
    fok : f.Ok
    x : Turing.ToPartrec.Cfg
    ⊢ Iff (Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.ste …
  -/
  simp only [Part.bind_eq_bind, Part.mem_bind_iff]
  /-
    f : Turing.ToPartrec.Code
    k : Turing.ToPartrec.Cont
    v : List Nat
    fok : f.Ok
    x : Turing.ToPartrec.Cfg
    ⊢ Iff (Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.ste …
  -/
  constructor
  · suffices ∀ c, x ∈ eval step c → ∀ v c', c = Cfg.then c' (Cont.fix f k) →
      Reaches step (stepNormal f Cont.halt v) c' →
        ∃ v₁ ∈ f.eval v, ∃ v₂ ∈ if List.headI v₁ = 0 then pure v₁.tail else f.fix.eval v₁.tail,
          x ∈ eval step (Cfg.ret k v₂) by
      intro h
      obtain ⟨v₁, hv₁, v₂, hv₂, h₃⟩ :=
        this _ h _ _ (stepNormal_then _ Cont.halt _ _) ReflTransGen.refl
      refine ⟨v₂, PFun.mem_fix_iff.2 ?_, h₃⟩
      simp only [Part.eq_some_iff.2 hv₁, Part.map_some]
      split_ifs at hv₂ ⊢
      · rw [Part.mem_some_iff.1 hv₂]
        exact Or.inl (Part.mem_some _)
      · exact Or.inr ⟨_, Part.mem_some _, hv₂⟩
    /-
      case mp
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      ⊢ ∀ (c : Turing.ToPartrec.Cfg), Membership.mem (Turing.eval Turing.ToPartrec.s …
    -/
    refine fun c he => evalInduction he fun y h IH => ?_
    /-
      case mp
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v : List Nat
      fok : f.Ok
      x c : Turing.ToPartrec.Cfg
      he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
      y : Turing.ToPartrec.Cfg
      h : Membership.mem (Turing.eval Turing.ToPartrec.step y) x
      IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step y) (Option.some  …
      ⊢ ∀ (v : List Nat) (c' : Turing.ToPartrec.Cfg), Eq y (c'.then (Turing.ToPartre …
    -/
    rintro v (⟨v'⟩ | ⟨k', v'⟩) rfl hr <;> rw [Cfg.then] at h IH <;> simp only [] at h IH
      /-
        case mp.halt
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
    · have := mem_eval.2 ⟨hr, rfl⟩
      /-
        case mp.halt
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        this : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.ste …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
      rw [fok, Part.bind_eq_bind, Part.mem_bind_iff] at this
      /-
        case mp.halt
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        this : Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turi …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
      obtain ⟨v'', h₁, h₂⟩ := this
      /-
        case mp.halt.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        v'' : List Nat
        h₁ : Membership.mem (f.eval v) v''
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.r …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
      rw [reaches_eval] at h₂
      /-
        case mp.halt.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        v'' : List Nat
        h₁ : Membership.mem (f.eval v) v''
        h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step ?m.139664) (Turing.ToPa …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
      swap
        /-
          case mp.halt.intro.intro
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          v'' : List Nat
          h₁ : Membership.mem (f.eval v) v''
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.r …
          ⊢ Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret Turing.ToPart …
        -/
      · exact ReflTransGen.single rfl
        /-
          🎉 no goals
        -/
      /-
        case mp.halt.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        v'' : List Nat
        h₁ : Membership.mem (f.eval v) v''
        h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
      cases Part.mem_unique h₂ (mem_eval.2 ⟨ReflTransGen.refl, rfl⟩)
      /-
        case mp.halt.intro.intro.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        h₁ : Membership.mem (f.eval v) v'
        h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
      refine ⟨v', h₁, ?_⟩
      /-
        case mp.halt.intro.intro.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        h₁ : Membership.mem (f.eval v) v'
        h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        ⊢ Exists fun v₂ => And (Membership.mem (ite (Eq v'.headI 0) (Pure.pure v'.tail …
      -/
      rw [stepRet] at h
      /-
        case mp.halt.intro.intro.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (ite (Eq v'.headI 0) (Tu …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        h₁ : Membership.mem (f.eval v) v'
        h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        ⊢ Exists fun v₂ => And (Membership.mem (ite (Eq v'.headI 0) (Pure.pure v'.tail …
      -/
      revert h
      /-
        case mp.halt.intro.intro.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v v' : List Nat
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        h₁ : Membership.mem (f.eval v) v'
        h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (ite (Eq v'.headI 0) (Turi …
      -/
      by_cases he : v'.headI = 0 <;> simp only [exists_prop, if_pos, if_false, he] <;> intro h
        /-
          case pos
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Eq v'.headI 0
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
          ⊢ Exists fun v₂ => And (Membership.mem (Pure.pure v'.tail) v₂) (Membership.mem …
        -/
      · refine ⟨_, Part.mem_some _, ?_⟩
        /-
          case pos
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Eq v'.headI 0
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
          ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret  …
        -/
        rw [reaches_eval]
          /-
            case pos
            f : Turing.ToPartrec.Code
            k : Turing.ToPartrec.Cont
            v✝ : List Nat
            fok : f.Ok
            x c : Turing.ToPartrec.Cfg
            he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
            v v' : List Nat
            IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
            hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
            h₁ : Membership.mem (f.eval v) v'
            h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
            h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
            he : Eq v'.headI 0
            h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
            ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step ?m.140578) x
          -/
        · exact h
          /-
            🎉 no goals
          -/
        /-
          case pos
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Eq v'.headI 0
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRe …
          ⊢ Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret k v'.tail) (T …
        -/
        exact ReflTransGen.single rfl
        /-
          🎉 no goals
        -/
        /-
          case neg
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Not (Eq v'.headI 0)
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNo …
          ⊢ Exists fun v₂ => And (Membership.mem (f.fix.eval v'.tail) v₂) (Membership.me …
        -/
      · obtain ⟨k₀, v₀, e₀⟩ := stepNormal.is_ret f Cont.halt v'.tail
        /-
          case neg.intro.intro
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Not (Eq v'.headI 0)
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNo …
          k₀ : Turing.ToPartrec.Cont
          v₀ : List Nat
          e₀ : Eq (Turing.ToPartrec.stepNormal f Turing.ToPartrec.Cont.halt v'.tail) (Tu …
          ⊢ Exists fun v₂ => And (Membership.mem (f.fix.eval v'.tail) v₂) (Membership.me …
        -/
        have e₁ := stepNormal_then f Cont.halt (Cont.fix f k) v'.tail
        /-
          case neg.intro.intro
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Not (Eq v'.headI 0)
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNo …
          k₀ : Turing.ToPartrec.Cont
          v₀ : List Nat
          e₀ : Eq (Turing.ToPartrec.stepNormal f Turing.ToPartrec.Cont.halt v'.tail) (Tu …
          e₁ : Eq (Turing.ToPartrec.stepNormal f (Turing.ToPartrec.Cont.halt.then (Turin …
          ⊢ Exists fun v₂ => And (Membership.mem (f.fix.eval v'.tail) v₂) (Membership.me …
        -/
        rw [e₀, Cont.then, Cfg.then] at e₁
        /-
          case neg.intro.intro
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Not (Eq v'.headI 0)
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNo …
          k₀ : Turing.ToPartrec.Cont
          v₀ : List Nat
          e₀ : Eq (Turing.ToPartrec.stepNormal f Turing.ToPartrec.Cont.halt v'.tail) (Tu …
          e₁ : Eq (Turing.ToPartrec.stepNormal f ((fun k' => k') (Turing.ToPartrec.Cont. …
          ⊢ Exists fun v₂ => And (Membership.mem (f.fix.eval v'.tail) v₂) (Membership.me …
        -/
        simp only [] at e₁
        obtain ⟨v₁, hv₁, v₂, hv₂, h₃⟩ :=
          IH (stepRet (k₀.then (Cont.fix f k)) v₀) (by rw [stepRet, if_neg he, e₁]; rfl)
            v'.tail _ stepRet_then (by apply ReflTransGen.single; rw [e₀]; rfl)
        /-
          case neg.intro.intro.intro.intro.intro.intro
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Not (Eq v'.headI 0)
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNo …
          k₀ : Turing.ToPartrec.Cont
          v₀ : List Nat
          e₀ : Eq (Turing.ToPartrec.stepNormal f Turing.ToPartrec.Cont.halt v'.tail) (Tu …
          e₁ : Eq (Turing.ToPartrec.stepNormal f (Turing.ToPartrec.Cont.fix f k) v'.tail …
          v₁ : List Nat
          hv₁ : Membership.mem (f.eval v'.tail) v₁
          v₂ : List Nat
          hv₂ : Membership.mem (ite (Eq v₁.headI 0) (Pure.pure v₁.tail) (f.fix.eval v₁.t …
          h₃ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.r …
          ⊢ Exists fun v₂ => And (Membership.mem (f.fix.eval v'.tail) v₂) (Membership.me …
        -/
        refine ⟨_, PFun.mem_fix_iff.2 ?_, h₃⟩
        /-
          case neg.intro.intro.intro.intro.intro.intro
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x c : Turing.ToPartrec.Cfg
          he✝ : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
          v v' : List Nat
          IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
          hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
          h₁ : Membership.mem (f.eval v) v'
          h₂✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          h₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he : Not (Eq v'.headI 0)
          h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNo …
          k₀ : Turing.ToPartrec.Cont
          v₀ : List Nat
          e₀ : Eq (Turing.ToPartrec.stepNormal f Turing.ToPartrec.Cont.halt v'.tail) (Tu …
          e₁ : Eq (Turing.ToPartrec.stepNormal f (Turing.ToPartrec.Cont.fix f k) v'.tail …
          v₁ : List Nat
          hv₁ : Membership.mem (f.eval v'.tail) v₁
          v₂ : List Nat
          hv₂ : Membership.mem (ite (Eq v₁.headI 0) (Pure.pure v₁.tail) (f.fix.eval v₁.t …
          h₃ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.r …
          ⊢ Or (Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) ( …
        -/
        simp only [Part.eq_some_iff.2 hv₁, Part.map_some, Part.mem_some_iff]
        split_ifs at hv₂ ⊢ <;> [exact Or.inl (congr_arg Sum.inl (Part.mem_some_iff.1 hv₂));
          exact Or.inr ⟨_, rfl, hv₂⟩]
      /-
        case mp.ret
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x c : Turing.ToPartrec.Cfg
        he : Membership.mem (Turing.eval Turing.ToPartrec.step c) x
        v : List Nat
        k' : Turing.ToPartrec.Cont
        v' : List Nat
        h : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.re …
        IH : ∀ (a' : Turing.ToPartrec.Cfg), Eq (Turing.ToPartrec.step (Turing.ToPartre …
        hr : Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.stepNormal f Turin …
        ⊢ Exists fun v₁ => And (Membership.mem (f.eval v) v₁) (Exists fun v₂ => And (M …
      -/
    · exact IH _ rfl _ _ stepRet_then (ReflTransGen.tail hr rfl)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      ⊢ (Exists fun a => And (Membership.mem (f.fix.eval v) a) (Membership.mem (Turi …
    -/
  · rintro ⟨v', he, hr⟩
    /-
      case mpr.intro.intro
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      v' : List Nat
      he : Membership.mem (f.fix.eval v) v'
      hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.r …
      ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNorm …
    -/
    rw [reaches_eval] at hr
    /-
      case mpr.intro.intro
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      v' : List Nat
      he : Membership.mem (f.fix.eval v) v'
      hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
      hr : Membership.mem (Turing.eval Turing.ToPartrec.step ?m.142877) x
      ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNorm …
    -/
    swap
      /-
        case mpr.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he : Membership.mem (f.fix.eval v) v'
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.r …
        ⊢ Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret k v') ?m.142877
      -/
    · exact ReflTransGen.single rfl
      /-
        🎉 no goals
      -/
    /-
      case mpr.intro.intro
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      v' : List Nat
      he : Membership.mem (f.fix.eval v) v'
      hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
      hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
      ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNorm …
    -/
    refine PFun.fixInduction he fun v (he : v' ∈ f.fix.eval v) IH => ?_
    /-
      case mpr.intro.intro
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v✝ : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      v' : List Nat
      he✝ : Membership.mem (f.fix.eval v✝) v'
      hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
      hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
      v : List Nat
      he : Membership.mem (f.fix.eval v) v'
      IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
      ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNorm …
    -/
    rw [fok, Part.bind_eq_bind, Part.mem_bind_iff]
    /-
      case mpr.intro.intro
      f : Turing.ToPartrec.Code
      k : Turing.ToPartrec.Cont
      v✝ : List Nat
      fok : f.Ok
      x : Turing.ToPartrec.Cfg
      v' : List Nat
      he✝ : Membership.mem (f.fix.eval v✝) v'
      hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
      hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
      v : List Nat
      he : Membership.mem (f.fix.eval v) v'
      IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
      ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
    -/
    obtain he | ⟨v'', he₁', _⟩ := PFun.mem_fix_iff.1 he
      /-
        case mpr.intro.intro.inl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝¹ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he✝ : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
    · obtain ⟨v', he₁, he₂⟩ := (Part.mem_map_iff _).1 he
      /-
        case mpr.intro.intro.inl.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v'✝ : List Nat
        he✝¹ : Membership.mem (f.fix.eval v✝) v'✝
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he✝ : Membership.mem (f.fix.eval v) v'✝
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
        v' : List Nat
        he₁ : Membership.mem (f.eval v) v'
        he₂ : Eq (ite (Eq v'.headI 0) (Sum.inl v'.tail) (Sum.inr v'.tail)) (Sum.inl v'✝)
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
      split_ifs at he₂ with h; cases he₂
      /-
        case pos.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v : List Nat
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v' : List Nat
        he₁ : Membership.mem (f.eval v) v'
        h : Eq v'.headI 0
        he✝¹ : Membership.mem (f.fix.eval v✝) v'.tail
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        he✝ : Membership.mem (f.fix.eval v) v'.tail
        he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
      refine ⟨_, he₁, ?_⟩
      /-
        case pos.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v : List Nat
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v' : List Nat
        he₁ : Membership.mem (f.eval v) v'
        h : Eq v'.headI 0
        he✝¹ : Membership.mem (f.fix.eval v✝) v'.tail
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        he✝ : Membership.mem (f.fix.eval v) v'.tail
        he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret  …
      -/
      rw [reaches_eval]
      /-
        case pos.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v : List Nat
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v' : List Nat
        he₁ : Membership.mem (f.eval v) v'
        h : Eq v'.headI 0
        he✝¹ : Membership.mem (f.fix.eval v✝) v'.tail
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        he✝ : Membership.mem (f.fix.eval v) v'.tail
        he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step ?m.143890) x
      -/
      swap
        /-
          case pos.refl
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x : Turing.ToPartrec.Cfg
          v : List Nat
          IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
          v' : List Nat
          he₁ : Membership.mem (f.eval v) v'
          h : Eq v'.headI 0
          he✝¹ : Membership.mem (f.fix.eval v✝) v'.tail
          hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          he✝ : Membership.mem (f.fix.eval v) v'.tail
          he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
          ⊢ Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret (Turing.ToPar …
        -/
      · exact ReflTransGen.single rfl
        /-
          🎉 no goals
        -/
      /-
        case pos.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v : List Nat
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v' : List Nat
        he₁ : Membership.mem (f.eval v) v'
        h : Eq v'.headI 0
        he✝¹ : Membership.mem (f.fix.eval v✝) v'.tail
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        he✝ : Membership.mem (f.fix.eval v) v'.tail
        he : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail) (S …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRet  …
      -/
      rwa [stepRet, if_pos h]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.inr.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v'' : List Nat
        he₁' : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail)  …
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
    · obtain ⟨v₁, he₁, he₂⟩ := (Part.mem_map_iff _).1 he₁'
      /-
        case mpr.intro.intro.inr.intro.intro.intro.intro
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v'' : List Nat
        he₁' : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail)  …
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        he₂ : Eq (ite (Eq v₁.headI 0) (Sum.inl v₁.tail) (Sum.inr v₁.tail)) (Sum.inr v'')
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
      split_ifs at he₂ with h; cases he₂
      /-
        case neg.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        h : Not (Eq v₁.headI 0)
        he₁' : Membership.mem (Part.map (fun v => ite (Eq v.headI 0) (Sum.inl v.tail)  …
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
      clear he₁'
      /-
        case neg.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        h : Not (Eq v₁.headI 0)
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Exists fun a => And (Membership.mem (f.eval v) a) (Membership.mem (Turing.ev …
      -/
      refine ⟨_, he₁, ?_⟩
      /-
        case neg.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        h : Not (Eq v₁.headI 0)
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret  …
      -/
      rw [reaches_eval]
      /-
        case neg.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        h : Not (Eq v₁.headI 0)
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step ?m.144739) x
      -/
      swap
        /-
          case neg.refl
          f : Turing.ToPartrec.Code
          k : Turing.ToPartrec.Cont
          v✝ : List Nat
          fok : f.Ok
          x : Turing.ToPartrec.Cfg
          v' : List Nat
          he✝ : Membership.mem (f.fix.eval v✝) v'
          hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
          hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
          v : List Nat
          he : Membership.mem (f.fix.eval v) v'
          IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
          v₁ : List Nat
          he₁ : Membership.mem (f.eval v) v₁
          h : Not (Eq v₁.headI 0)
          right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
          ⊢ Turing.Reaches Turing.ToPartrec.step (Turing.ToPartrec.Cfg.ret (Turing.ToPar …
        -/
      · exact ReflTransGen.single rfl
        /-
          🎉 no goals
        -/
      /-
        case neg.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        h : Not (Eq v₁.headI 0)
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepRet  …
      -/
      rw [stepRet, if_neg h]
      /-
        case neg.refl
        f : Turing.ToPartrec.Code
        k : Turing.ToPartrec.Cont
        v✝ : List Nat
        fok : f.Ok
        x : Turing.ToPartrec.Cfg
        v' : List Nat
        he✝ : Membership.mem (f.fix.eval v✝) v'
        hr✝ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.Cfg. …
        hr : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepR …
        v : List Nat
        he : Membership.mem (f.fix.eval v) v'
        IH : ∀ (a'' : List Nat), Membership.mem (Part.map (fun v => ite (Eq v.headI 0) …
        v₁ : List Nat
        he₁ : Membership.mem (f.eval v) v₁
        h : Not (Eq v₁.headI 0)
        right✝ : Membership.mem (PFun.fix (fun v => Part.map (fun v => ite (Eq v.headI …
        ⊢ Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.stepNorm …
      -/
      exact IH v₁.tail ((Part.mem_map_iff _).2 ⟨_, he₁, if_neg h⟩)
      /-
        🎉 no goals
      -/


theorem code_is_ok (c) : Code.Ok c := by
  induction c with (intro k v; rw [stepNormal])
  | cons f fs IHf IHfs =>
    rw [Code.eval, IHf]
    simp only [bind_assoc, Cont.eval, pure_bind]; congr; funext v
    rw [reaches_eval]; swap
    · exact ReflTransGen.single rfl
    rw [stepRet, IHfs]; congr; funext v'
    refine Eq.trans (b := eval step (stepRet (Cont.cons₂ v k) v')) ?_ (Eq.symm ?_) <;>
      exact reaches_eval (ReflTransGen.single rfl)
  | comp f g IHf IHg =>
    rw [Code.eval, IHg]
    simp only [bind_assoc, Cont.eval, pure_bind]; congr; funext v
    rw [reaches_eval]; swap
    · exact ReflTransGen.single rfl
    rw [stepRet, IHf]
  | case f g IHf IHg =>
    simp only [Code.eval]
    cases v.headI <;> simp only [Nat.rec_zero, Part.bind_eq_bind] <;> [apply IHf; apply IHg]
  | fix f IHf => rw [cont_eval_fix IHf]
  | _ => simp only [Code.eval, pure_bind]


theorem stepNormal_eval (c v) : eval step (stepNormal c Cont.halt v) = Cfg.halt <$> c.eval v :=
  (code_is_ok c).zero


theorem stepRet_eval {k v} : eval step (stepRet k v) = Cfg.halt <$> k.eval v := by
  induction k generalizing v with
  | halt =>
    simp only [mem_eval, Cont.eval, map_pure]
    exact Part.eq_some_iff.2 (mem_eval.2 ⟨ReflTransGen.refl, rfl⟩)
  | cons₁ fs as k IH =>
    rw [Cont.eval, stepRet, code_is_ok]
    simp only [← bind_pure_comp, bind_assoc]; congr; funext v'
    rw [reaches_eval]; swap
    · exact ReflTransGen.single rfl
    rw [stepRet, IH, bind_pure_comp]
  | cons₂ ns k IH => rw [Cont.eval, stepRet]; exact IH
  | comp f k IH =>
    rw [Cont.eval, stepRet, code_is_ok]
    simp only [← bind_pure_comp, bind_assoc]; congr; funext v'
    rw [reaches_eval]; swap
    · exact ReflTransGen.single rfl
    rw [IH, bind_pure_comp]
  | fix f k IH =>
    rw [Cont.eval, stepRet]; simp only [bind_pure_comp]
    split_ifs; · exact IH
    simp only [← bind_pure_comp, bind_assoc, cont_eval_fix (code_is_ok _)]
    congr; funext; rw [bind_pure_comp, ← IH]
    exact reaches_eval (ReflTransGen.single rfl)


/-- The alphabet for the stacks in the program. `bit0` and `bit1` are used to represent `ℕ` values
as lists of binary digits, `cons` is used to separate `List ℕ` values, and `consₗ` is used to
separate `List (List ℕ)` values. See the section documentation. -/
inductive Γ'
  | consₗ
  | cons
  | bit0
  | bit1
  deriving DecidableEq, Inhabited, Fintype


/-- The four stacks used by the program. `main` is used to store the input value in `trNormal`
mode and the output value in `Λ'.ret` mode, while `stack` is used to keep all the data for the
continuations. `rev` is used to store reversed lists when transferring values between stacks, and
`aux` is only used once in `cons₁`. See the section documentation. -/
inductive K'
  | main
  | rev
  | aux
  | stack
  deriving DecidableEq, Inhabited


/-- Continuations as in `ToPartrec.Cont` but with the data removed. This is done because we want
the set of all continuations in the program to be finite (so that it can ultimately be encoded into
the finite state machine of a Turing machine), but a continuation can handle a potentially infinite
number of data values during execution. -/
inductive Cont'
  | halt
  | cons₁ : Code → Cont' → Cont'
  | cons₂ : Cont' → Cont'
  | comp : Code → Cont' → Cont'
  | fix : Code → Cont' → Cont'
  deriving DecidableEq, Inhabited


/-- The set of program positions. We make extensive use of inductive types here to let us describe
"subroutines"; for example `clear p k q` is a program that clears stack `k`, then does `q` where
`q` is another label. In order to prevent this from resulting in an infinite number of distinct
accessible states, we are careful to be non-recursive (although loops are okay). See the section
documentation for a description of all the programs. -/
inductive Λ'
  | move (p : Γ' → Bool) (k₁ k₂ : K') (q : Λ')
  | clear (p : Γ' → Bool) (k : K') (q : Λ')
  | copy (q : Λ')
  | push (k : K') (s : Option Γ' → Option Γ') (q : Λ')
  | read (f : Option Γ' → Λ')
  | succ (q : Λ')
  | pred (q₁ q₂ : Λ')
  | ret (k : Cont')


compile_inductive% Code

compile_inductive% Cont'

compile_inductive% K'

compile_inductive% Λ'


instance Λ'.instInhabited : Inhabited Λ' :=
  ⟨Λ'.ret Cont'.halt⟩


instance Λ'.instDecidableEq : DecidableEq Λ' := fun a b => by
  /-
    a b : Turing.PartrecToTM2.Λ'
    ⊢ Decidable (Eq a b)
  -/
  induction a generalizing b <;> cases b <;> first
    | apply Decidable.isFalse; rintro ⟨⟨⟩⟩; done
    | exact decidable_of_iff' _ (by simp [funext_iff]; rfl)


/-- The type of TM2 statements used by this machine. -/
def Stmt' :=
  TM2.Stmt (fun _ : K' => Γ') Λ' (Option Γ') deriving Inhabited


/-- The type of TM2 configurations used by this machine. -/
def Cfg' :=
  TM2.Cfg (fun _ : K' => Γ') Λ' (Option Γ') deriving Inhabited


/-- A predicate that detects the end of a natural number, either `Γ'.cons` or `Γ'.consₗ` (or
implicitly the end of the list), for use in predicate-taking functions like `move` and `clear`. -/
@[simp]
def natEnd : Γ' → Bool
  | Γ'.consₗ => true
  | Γ'.cons => true
  | _ => false

/-- Pop a value from the stack and place the result in local store. -/
@[simp]
def pop' (k : K') : Stmt' → Stmt' :=
  pop k fun _ v => v


/-- Peek a value from the stack and place the result in local store. -/
@[simp]
def peek' (k : K') : Stmt' → Stmt' :=
  peek k fun _ v => v


/-- Push the value in the local store to the given stack. -/
@[simp]
def push' (k : K') : Stmt' → Stmt' :=
  push k fun x => x.iget


/-- Move everything from the `rev` stack to the `main` stack (reversed). -/
def unrev :=
  Λ'.move (fun _ => false) rev main


/-- Move elements from `k₁` to `k₂` while `p` holds, with the last element being left on `k₁`. -/
def moveExcl (p k₁ k₂ q) :=
  Λ'.move p k₁ k₂ <| Λ'.push k₁ id q


/-- Move elements from `k₁` to `k₂` without reversion, by performing a double move via the `rev`
stack. -/
def move₂ (p k₁ k₂ q) :=
  moveExcl p k₁ rev <| Λ'.move (fun _ => false) rev k₂ q


/-- Assuming `trList v` is on the front of stack `k`, remove it, and push `v.headI` onto `main`.
See the section documentation. -/
def head (k : K') (q : Λ') : Λ' :=
  Λ'.move natEnd k rev <|
    (Λ'.push rev fun _ => some Γ'.cons) <|
      Λ'.read fun s =>
        (if s = some Γ'.consₗ then id else Λ'.clear (fun x => x = Γ'.consₗ) k) <| unrev q


/-- The program that evaluates code `c` with continuation `k`. This expects an initial state where
`trList v` is on `main`, `trContStack k` is on `stack`, and `aux` and `rev` are empty.
See the section documentation for details. -/
@[simp]
def trNormal : Code → Cont' → Λ'
  | Code.zero', k => (Λ'.push main fun _ => some Γ'.cons) <| Λ'.ret k
  | Code.succ, k => head main <| Λ'.succ <| Λ'.ret k
  | Code.tail, k => Λ'.clear natEnd main <| Λ'.ret k
  | Code.cons f fs, k =>
    (Λ'.push stack fun _ => some Γ'.consₗ) <|
      Λ'.move (fun _ => false) main rev <| Λ'.copy <| trNormal f (Cont'.cons₁ fs k)
  | Code.comp f g, k => trNormal g (Cont'.comp f k)
  | Code.case f g, k => Λ'.pred (trNormal f k) (trNormal g k)
  | Code.fix f, k => trNormal f (Cont'.fix f k)


/-- The main program. See the section documentation for details. -/
def tr : Λ' → Stmt'
  | Λ'.move p k₁ k₂ q =>
    pop' k₁ <|
      branch (fun s => s.elim true p) (goto fun _ => q)
        (push' k₂ <| goto fun _ => Λ'.move p k₁ k₂ q)
  | Λ'.push k f q =>
    branch (fun s => (f s).isSome) ((push k fun s => (f s).iget) <| goto fun _ => q)
      (goto fun _ => q)
  | Λ'.read q => goto q
  | Λ'.clear p k q =>
    pop' k <| branch (fun s => s.elim true p) (goto fun _ => q) (goto fun _ => Λ'.clear p k q)
  | Λ'.copy q =>
    pop' rev <|
      branch Option.isSome (push' main <| push' stack <| goto fun _ => Λ'.copy q) (goto fun _ => q)
  | Λ'.succ q =>
    pop' main <|
      branch (fun s => s = some Γ'.bit1) ((push rev fun _ => Γ'.bit0) <| goto fun _ => Λ'.succ q) <|
        branch (fun s => s = some Γ'.cons)
          ((push main fun _ => Γ'.cons) <| (push main fun _ => Γ'.bit1) <| goto fun _ => unrev q)
          ((push main fun _ => Γ'.bit1) <| goto fun _ => unrev q)
  | Λ'.pred q₁ q₂ =>
    pop' main <|
      branch (fun s => s = some Γ'.bit0)
          ((push rev fun _ => Γ'.bit1) <| goto fun _ => Λ'.pred q₁ q₂) <|
        branch (fun s => natEnd s.iget) (goto fun _ => q₁)
          (peek' main <|
            branch (fun s => natEnd s.iget) (goto fun _ => unrev q₂)
              ((push rev fun _ => Γ'.bit0) <| goto fun _ => unrev q₂))
  | Λ'.ret (Cont'.cons₁ fs k) =>
    goto fun _ =>
      move₂ (fun _ => false) main aux <|
        move₂ (fun s => s = Γ'.consₗ) stack main <|
          move₂ (fun _ => false) aux stack <| trNormal fs (Cont'.cons₂ k)
  | Λ'.ret (Cont'.cons₂ k) => goto fun _ => head stack <| Λ'.ret k
  | Λ'.ret (Cont'.comp f k) => goto fun _ => trNormal f k
  | Λ'.ret (Cont'.fix f k) =>
    pop' main <|
      goto fun s =>
        cond (natEnd s.iget) (Λ'.ret k) <| Λ'.clear natEnd main <| trNormal f (Cont'.fix f k)
  | Λ'.ret Cont'.halt => (load fun _ => none) <| halt


@[simp]
theorem tr_move (p k₁ k₂ q) : tr (Λ'.move p k₁ k₂ q) =
    pop' k₁ (branch (fun s => s.elim true p) (goto fun _ => q)
      (push' k₂ <| goto fun _ => Λ'.move p k₁ k₂ q)) := rfl


@[simp]
theorem tr_push (k f q) : tr (Λ'.push k f q) = branch (fun s => (f s).isSome)
    ((push k fun s => (f s).iget) <| goto fun _ => q) (goto fun _ => q) := rfl


@[simp]
theorem tr_read (q) : tr (Λ'.read q) = goto q := rfl


@[simp]
theorem tr_clear (p k q) : tr (Λ'.clear p k q) = pop' k (branch
    (fun s => s.elim true p) (goto fun _ => q) (goto fun _ => Λ'.clear p k q)) := rfl


@[simp]
theorem tr_copy (q) : tr (Λ'.copy q) = pop' rev (branch Option.isSome
    (push' main <| push' stack <| goto fun _ => Λ'.copy q) (goto fun _ => q)) := rfl


@[simp]
theorem tr_succ (q) : tr (Λ'.succ q) = pop' main (branch (fun s => s = some Γ'.bit1)
    ((push rev fun _ => Γ'.bit0) <| goto fun _ => Λ'.succ q) <|
      branch (fun s => s = some Γ'.cons)
        ((push main fun _ => Γ'.cons) <| (push main fun _ => Γ'.bit1) <| goto fun _ => unrev q)
        ((push main fun _ => Γ'.bit1) <| goto fun _ => unrev q)) := rfl


@[simp]
theorem tr_pred (q₁ q₂) : tr (Λ'.pred q₁ q₂) = pop' main (branch (fun s => s = some Γ'.bit0)
    ((push rev fun _ => Γ'.bit1) <| goto fun _ => Λ'.pred q₁ q₂) <|
    branch (fun s => natEnd s.iget) (goto fun _ => q₁)
      (peek' main <|
        branch (fun s => natEnd s.iget) (goto fun _ => unrev q₂)
          ((push rev fun _ => Γ'.bit0) <| goto fun _ => unrev q₂))) := rfl


@[simp]
theorem tr_ret_cons₁ (fs k) : tr (Λ'.ret (Cont'.cons₁ fs k)) = goto fun _ =>
    move₂ (fun _ => false) main aux <|
      move₂ (fun s => s = Γ'.consₗ) stack main <|
        move₂ (fun _ => false) aux stack <| trNormal fs (Cont'.cons₂ k) := rfl


@[simp]
theorem tr_ret_cons₂ (k) : tr (Λ'.ret (Cont'.cons₂ k)) =
    goto fun _ => head stack <| Λ'.ret k := rfl


@[simp]
theorem tr_ret_comp (f k) : tr (Λ'.ret (Cont'.comp f k)) = goto fun _ => trNormal f k := rfl


@[simp]
theorem tr_ret_fix (f k) : tr (Λ'.ret (Cont'.fix f k)) = pop' main (goto fun s =>
    cond (natEnd s.iget) (Λ'.ret k) <| Λ'.clear natEnd main <| trNormal f (Cont'.fix f k)) := rfl


@[simp]
theorem tr_ret_halt : tr (Λ'.ret Cont'.halt) = (load fun _ => none) halt := rfl


/-- Translating a `Cont` continuation to a `Cont'` continuation simply entails dropping all the
data. This data is instead encoded in `trContStack` in the configuration. -/
def trCont : Cont → Cont'
  | Cont.halt => Cont'.halt
  | Cont.cons₁ c _ k => Cont'.cons₁ c (trCont k)
  | Cont.cons₂ _ k => Cont'.cons₂ (trCont k)
  | Cont.comp c k => Cont'.comp c (trCont k)
  | Cont.fix c k => Cont'.fix c (trCont k)


/-- We use `PosNum` to define the translation of binary natural numbers. A natural number is
represented as a little-endian list of `bit0` and `bit1` elements:

    1 = [bit1]
    2 = [bit0, bit1]
    3 = [bit1, bit1]
    4 = [bit0, bit0, bit1]

In particular, this representation guarantees no trailing `bit0`'s at the end of the list. -/
def trPosNum : PosNum → List Γ'
  | PosNum.one => [Γ'.bit1]
  | PosNum.bit0 n => Γ'.bit0 :: trPosNum n
  | PosNum.bit1 n => Γ'.bit1 :: trPosNum n


/-- We use `Num` to define the translation of binary natural numbers. Positive numbers are
translated using `trPosNum`, and `trNum 0 = []`. So there are never any trailing `bit0`'s in
a translated `Num`.

    0 = []
    1 = [bit1]
    2 = [bit0, bit1]
    3 = [bit1, bit1]
    4 = [bit0, bit0, bit1]
-/
def trNum : Num → List Γ'
  | Num.zero => []
  | Num.pos n => trPosNum n


/-- Because we use binary encoding, we define `trNat` in terms of `trNum`, using `Num`, which are
binary natural numbers. (We could also use `Nat.binaryRecOn`, but `Num` and `PosNum` make for
easy inductions.) -/
def trNat (n : ℕ) : List Γ' :=
  trNum n


@[simp]
                                        /-
                                          ⊢ Eq (Turing.PartrecToTM2.trNat 0) List.nil
                                        -/
theorem trNat_zero : trNat 0 = [] := by rw [trNat, Nat.cast_zero]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem trNat_default : trNat default = [] :=
  trNat_zero


/-- Lists are translated with a `cons` after each encoded number.
For example:

    [] = []
    [0] = [cons]
    [1] = [bit1, cons]
    [6, 0] = [bit0, bit1, bit1, cons, cons]
-/
@[simp]
def trList : List ℕ → List Γ'
  | [] => []
  | n::ns => trNat n ++ Γ'.cons :: trList ns


/-- Lists of lists are translated with a `consₗ` after each encoded list.
For example:

    [] = []
    [[]] = [consₗ]
    [[], []] = [consₗ, consₗ]
    [[0]] = [cons, consₗ]
    [[1, 2], [0]] = [bit1, cons, bit0, bit1, cons, consₗ, cons, consₗ]
-/
@[simp]
def trLList : List (List ℕ) → List Γ'
  | [] => []
  | l::ls => trList l ++ Γ'.consₗ :: trLList ls


/-- The data part of a continuation is a list of lists, which is encoded on the `stack` stack
using `trLList`. -/
@[simp]
def contStack : Cont → List (List ℕ)
  | Cont.halt => []
  | Cont.cons₁ _ ns k => ns :: contStack k
  | Cont.cons₂ ns k => ns :: contStack k
  | Cont.comp _ k => contStack k
  | Cont.fix _ k => contStack k


/-- The data part of a continuation is a list of lists, which is encoded on the `stack` stack
using `trLList`. -/
def trContStack (k : Cont) :=
  trLList (contStack k)


/-- This is the nondependent eliminator for `K'`, but we use it specifically here in order to
represent the stack data as four lists rather than as a function `K' → List Γ'`, because this makes
rewrites easier. The theorems `K'.elim_update_main` et. al. show how such a function is updated
after an `update` to one of the components. -/
def K'.elim (a b c d : List Γ') : K' → List Γ'
  | K'.main => a
  | K'.rev => b
  | K'.aux => c
  | K'.stack => d

-- The equation lemma of `elim` simplifies to `match` structures.


theorem K'.elim_main (a b c d) : K'.elim a b c d K'.main = a := rfl


theorem K'.elim_rev (a b c d) : K'.elim a b c d K'.rev = b := rfl


theorem K'.elim_aux (a b c d) : K'.elim a b c d K'.aux = c := rfl


theorem K'.elim_stack (a b c d) : K'.elim a b c d K'.stack = d := rfl


attribute [simp] K'.elim


@[simp]
theorem K'.elim_update_main {a b c d a'} : update (K'.elim a b c d) main a' = K'.elim a' b c d := by
  /-
    a b c d a' : List Turing.PartrecToTM2.Γ'
    ⊢ Eq (Function.update (Turing.PartrecToTM2.K'.elim a b c d) Turing.PartrecToTM …
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
  funext x; cases x <;> rfl
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem K'.elim_update_rev {a b c d b'} : update (K'.elim a b c d) rev b' = K'.elim a b' c d := by
  /-
    a b c d b' : List Turing.PartrecToTM2.Γ'
    ⊢ Eq (Function.update (Turing.PartrecToTM2.K'.elim a b c d) Turing.PartrecToTM …
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
  funext x; cases x <;> rfl
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem K'.elim_update_aux {a b c d c'} : update (K'.elim a b c d) aux c' = K'.elim a b c' d := by
  /-
    a b c d c' : List Turing.PartrecToTM2.Γ'
    ⊢ Eq (Function.update (Turing.PartrecToTM2.K'.elim a b c d) Turing.PartrecToTM …
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
  funext x; cases x <;> rfl
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem K'.elim_update_stack {a b c d d'} :
                                                               /-
                                                                 a b c d d' : List Turing.PartrecToTM2.Γ'
                                                                 ⊢ Eq (Function.update (Turing.PartrecToTM2.K'.elim a b c d) Turing.PartrecToTM …
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
    update (K'.elim a b c d) stack d' = K'.elim a b c d' := by funext x; cases x <;> rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- The halting state corresponding to a `List ℕ` output value. -/
def halt (v : List ℕ) : Cfg' :=
  ⟨none, none, K'.elim (trList v) [] [] []⟩


/-- The `Cfg` states map to `Cfg'` states almost one to one, except that in normal operation the
local store contains an arbitrary garbage value. To make the final theorem cleaner we explicitly
clear it in the halt state so that there is exactly one configuration corresponding to output `v`.
-/
def TrCfg : Cfg → Cfg' → Prop
  | Cfg.ret k v, c' =>
    ∃ s, c' = ⟨some (Λ'.ret (trCont k)), s, K'.elim (trList v) [] [] (trContStack k)⟩
  | Cfg.halt v, c' => c' = halt v


/-- This could be a general list definition, but it is also somewhat specialized to this
application. `splitAtPred p L` will search `L` for the first element satisfying `p`.
If it is found, say `L = l₁ ++ a :: l₂` where `a` satisfies `p` but `l₁` does not, then it returns
`(l₁, some a, l₂)`. Otherwise, if there is no such element, it returns `(L, none, [])`. -/
def splitAtPred {α} (p : α → Bool) : List α → List α × Option α × List α
  | [] => ([], none, [])
  | a :: as =>
    cond (p a) ([], some a, as) <|
      let ⟨l₁, o, l₂⟩ := splitAtPred p as
      ⟨a::l₁, o, l₂⟩


theorem splitAtPred_eq {α} (p : α → Bool) :
    ∀ L l₁ o l₂,
      (∀ x ∈ l₁, p x = false) →
        Option.elim' (L = l₁ ∧ l₂ = []) (fun a => p a = true ∧ L = l₁ ++ a::l₂) o →
          splitAtPred p L = (l₁, o, l₂)
  | [], _, none, _, _, ⟨rfl, rfl⟩ => rfl
                                         /-
                                           α : Type u_1
                                           p : α → Bool
                                           l₁ : List α
                                           o : α
                                           l₂ : List α
                                           x✝ : ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
                                           left✝ : Eq (p o) Bool.true
                                           h₃ : Eq List.nil (HAppend.hAppend l₁ (List.cons o l₂))
                                           ⊢ Eq (Turing.PartrecToTM2.splitAtPred p List.nil) { fst := l₁, snd := { fst := …
                                         -/
  | [], l₁, some o, l₂, _, ⟨_, h₃⟩ => by simp at h₃
                                         /-
                                           🎉 no goals
                                         -/
  | a :: L, l₁, o, l₂, h₁, h₂ => by
    /-
      α : Type u_1
      p : α → Bool
      a : α
      L l₁ : List α
      o : Option α
      l₂ : List α
      h₁ : ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
      h₂ : Option.elim' (And (Eq (List.cons a L) l₁) (Eq l₂ List.nil)) (fun a_1 => A …
      ⊢ Eq (Turing.PartrecToTM2.splitAtPred p (List.cons a L)) { fst := l₁, snd := { …
    -/
    rw [splitAtPred]
    /-
      α : Type u_1
      p : α → Bool
      a : α
      L l₁ : List α
      o : Option α
      l₂ : List α
      h₁ : ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
      h₂ : Option.elim' (And (Eq (List.cons a L) l₁) (Eq l₂ List.nil)) (fun a_1 => A …
      ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := L }  …
    -/
    have IH := splitAtPred_eq p L
    /-
      α : Type u_1
      p : α → Bool
      a : α
      L l₁ : List α
      o : Option α
      l₂ : List α
      h₁ : ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
      h₂ : Option.elim' (And (Eq (List.cons a L) l₁) (Eq l₂ List.nil)) (fun a_1 => A …
      IH : ∀ (l₁ : List α) (o : Option α) (l₂ : List α), (∀ (x : α), Membership.mem  …
      ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := L }  …
    -/
    cases' o with o
      /-
        case none
        α : Type u_1
        p : α → Bool
        a : α
        L l₁ l₂ : List α
        h₁ : ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
        IH : ∀ (l₁ : List α) (o : Option α) (l₂ : List α), (∀ (x : α), Membership.mem  …
        h₂ : Option.elim' (And (Eq (List.cons a L) l₁) (Eq l₂ List.nil)) (fun a_1 => A …
        ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := L }  …
      -/
                               /-
                                 🎉 no goals
                               -/
    · cases' l₁ with a' l₁ <;> rcases h₂ with ⟨⟨⟩, rfl⟩
      /-
        case none.cons.intro.refl
        α : Type u_1
        p : α → Bool
        a : α
        L : List α
        IH : ∀ (l₁ : List α) (o : Option α) (l₂ : List α), (∀ (x : α), Membership.mem  …
        h₁ : ∀ (x : α), Membership.mem (List.cons a L) x → Eq (p x) Bool.false
        ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := L }  …
      -/
      rw [h₁ a (List.Mem.head _), cond, IH L none [] _ ⟨rfl, rfl⟩]
      /-
        α : Type u_1
        p : α → Bool
        a : α
        L : List α
        IH : ∀ (l₁ : List α) (o : Option α) (l₂ : List α), (∀ (x : α), Membership.mem  …
        h₁ : ∀ (x : α), Membership.mem (List.cons a L) x → Eq (p x) Bool.false
        ⊢ ∀ (x : α), Membership.mem L x → Eq (p x) Bool.false
      -/
      exact fun x h => h₁ x (List.Mem.tail _ h)
      /-
        🎉 no goals
      -/
      /-
        case some
        α : Type u_1
        p : α → Bool
        a : α
        L l₁ l₂ : List α
        h₁ : ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
        IH : ∀ (l₁ : List α) (o : Option α) (l₂ : List α), (∀ (x : α), Membership.mem  …
        o : α
        h₂ : Option.elim' (And (Eq (List.cons a L) l₁) (Eq l₂ List.nil)) (fun a_1 => A …
        ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := L }  …
      -/
    · cases' l₁ with a' l₁ <;> rcases h₂ with ⟨h₂, ⟨⟩⟩
        /-
          case some.nil.intro.refl
          α : Type u_1
          p : α → Bool
          a : α
          L : List α
          IH : ∀ (l₁ : List α) (o : Option α) (l₂ : List α), (∀ (x : α), Membership.mem  …
          h₁ : ∀ (x : α), Membership.mem List.nil x → Eq (p x) Bool.false
          h₂ : Eq (p a) Bool.true
          ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := L }  …
        -/
      · rw [h₂, cond]
        /-
          🎉 no goals
        -/
      /-
        case some.cons.intro.refl
        α : Type u_1
        p : α → Bool
        a : α
        l₂ : List α
        o : α
        l₁ : List α
        h₂ : Eq (p o) Bool.true
        h₁ : ∀ (x : α), Membership.mem (List.cons a l₁) x → Eq (p x) Bool.false
        IH : ∀ (l₁_1 : List α) (o_1 : Option α) (l₂_1 : List α), (∀ (x : α), Membershi …
        ⊢ Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := l₁.a …
      -/
      rw [h₁ a (List.Mem.head _), cond, IH l₁ (some o) l₂ _ ⟨h₂, _⟩] <;> try rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
      /-
        α : Type u_1
        p : α → Bool
        a : α
        l₂ : List α
        o : α
        l₁ : List α
        h₂ : Eq (p o) Bool.true
        h₁ : ∀ (x : α), Membership.mem (List.cons a l₁) x → Eq (p x) Bool.false
        IH : ∀ (l₁_1 : List α) (o_1 : Option α) (l₂_1 : List α), (∀ (x : α), Membershi …
        ⊢ ∀ (x : α), Membership.mem l₁ x → Eq (p x) Bool.false
      -/
      exact fun x h => h₁ x (List.Mem.tail _ h)
      /-
        🎉 no goals
      -/


theorem splitAtPred_false {α} (L : List α) : splitAtPred (fun _ => false) L = (L, none, []) :=
  splitAtPred_eq _ _ _ _ _ (fun _ _ => rfl) ⟨rfl, rfl⟩


theorem move_ok {p k₁ k₂ q s L₁ o L₂} {S : K' → List Γ'} (h₁ : k₁ ≠ k₂)
    (e : splitAtPred p (S k₁) = (L₁, o, L₂)) :
    Reaches₁ (TM2.step tr) ⟨some (Λ'.move p k₁ k₂ q), s, S⟩
      ⟨some q, o, update (update S k₁ L₂) k₂ (L₁.reverseAux (S k₂))⟩ := by
  /-
    p : Turing.PartrecToTM2.Γ' → Bool
    k₁ k₂ : Turing.PartrecToTM2.K'
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L₁ : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ'
    L₂ : List Turing.PartrecToTM2.Γ'
    S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
    h₁ : Ne k₁ k₂
    e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  induction' L₁ with a L₁ IH generalizing S s
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.nil, snd := {  …
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · rw [(_ : [].reverseAux _ = _), Function.update_eq_self]
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.nil, snd := {  …
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
    swap
      /-
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        h₁ : Ne k₁ k₂
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.nil, snd := {  …
        ⊢ Eq (List.nil.reverseAux (S k₂)) (Function.update S k₁ L₂ k₂)
      -/
    · rw [Function.update_of_ne h₁.symm, List.reverseAux_nil]
      /-
        🎉 no goals
      -/
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.nil, snd := {  …
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
    refine TransGen.head' rfl ?_
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.nil, snd := {  …
      ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM2.step Turing.Par …
    -/
    simp only [TM2.step, Option.mem_def, TM2.stepAux, Option.elim, ne_eq]
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.nil, snd := {  …
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    revert e; cases' S k₁ with a Sk <;> intro e
      /-
        case nil.nil
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        h₁ : Ne k₁ k₂
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        e : Eq (Turing.PartrecToTM2.splitAtPred p List.nil) { fst := List.nil, snd :=  …
        ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
      -/
    · cases e
      /-
        case nil.nil.refl
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        h₁ : Ne k₁ k₂
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case nil.cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (List.cons a Sk)) { fst := List.nil, …
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    simp only [splitAtPred, Option.elim, List.head?, List.tail_cons, Option.iget_some] at e ⊢
    /-
      case nil.cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := Sk …
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    revert e; cases p a <;> intro e <;>
      /-
        case nil.cons.false
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        h₁ : Ne k₁ k₂
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        a : Turing.PartrecToTM2.Γ'
        Sk : List Turing.PartrecToTM2.Γ'
        e : Eq (cond Bool.false { fst := List.nil, snd := { fst := Option.some a, snd  …
        ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
      -/
      /-
        🎉 no goals
      -/
      simp only [cond_false, cond_true, Prod.mk.injEq, true_and, false_and, reduceCtorEq] at e ⊢
    /-
      case nil.cons.true
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : And (Eq (Option.some a) o) (Eq Sk L₂)
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    simp only [e]
    /-
      case nil.cons.true
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : And (Eq (Option.some a) o) (Eq Sk L₂)
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.cons a L₁, snd …
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · refine TransGen.head rfl ?_
    /-
      case cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.cons a L₁, snd …
      ⊢ Relation.TransGen (fun a b => Membership.mem (Turing.TM2.step Turing.Partrec …
    -/
    simp only [TM2.step, Option.mem_def, TM2.stepAux, Option.elim, ne_eq, List.reverseAux_cons]
    /-
      case cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := List.cons a L₁, snd …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    cases' e₁ : S k₁ with a' Sk <;> rw [e₁, splitAtPred] at e
      /-
        case cons.nil
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        h₁ : Ne k₁ k₂
        a : Turing.PartrecToTM2.Γ'
        L₁ : List Turing.PartrecToTM2.Γ'
        IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        e : Eq { fst := List.nil, snd := { fst := Option.none, snd := List.nil } } { f …
        e₁ : Eq (S k₁) List.nil
        ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
      -/
    · cases e
      /-
        🎉 no goals
      -/
    /-
      case cons.cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : Eq (cond (p a') { fst := List.nil, snd := { fst := Option.some a', snd :=  …
      e₁ : Eq (S k₁) (List.cons a' Sk)
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    cases e₂ : p a' <;> simp only [e₂, cond] at e
    /-
      case cons.cons.false
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k₁) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      e : Eq { fst := List.cons a' (Turing.PartrecToTM2.splitAtPred p Sk).1, snd :=  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    swap
      /-
        case cons.cons.true
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        h₁ : Ne k₁ k₂
        a : Turing.PartrecToTM2.Γ'
        L₁ : List Turing.PartrecToTM2.Γ'
        IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        a' : Turing.PartrecToTM2.Γ'
        Sk : List Turing.PartrecToTM2.Γ'
        e₁ : Eq (S k₁) (List.cons a' Sk)
        e₂ : Eq (p a') Bool.true
        e : Eq { fst := List.nil, snd := { fst := Option.some a', snd := Sk } } { fst  …
        ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
      -/
    · cases e
      /-
        🎉 no goals
      -/
    /-
      case cons.cons.false
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k₁) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      e : Eq { fst := List.cons a' (Turing.PartrecToTM2.splitAtPred p Sk).1, snd :=  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    rcases e₃ : splitAtPred p Sk with ⟨_, _, _⟩
    /-
      case cons.cons.false.mk.mk
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k₁) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      e : Eq { fst := List.cons a' (Turing.PartrecToTM2.splitAtPred p Sk).1, snd :=  …
      fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    rw [e₃] at e
    /-
      case cons.cons.false.mk.mk
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k₁) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e : Eq { fst := List.cons a' { fst := fst✝¹, snd := { fst := fst✝, snd := snd✝ …
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    cases e
    /-
      case cons.cons.false.mk.mk.refl
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      Sk fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      e₁ : Eq (S k₁) (List.cons a Sk)
      e₂ : Eq (p a) Bool.false
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    simp only [List.head?_cons, e₂, List.tail_cons, ne_eq, cond_false]
    /-
      case cons.cons.false.mk.mk.refl
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      Sk fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      e₁ : Eq (S k₁) (List.cons a Sk)
      e₂ : Eq (p a) Bool.false
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    convert @IH _ (update (update S k₁ Sk) k₂ (a :: S k₂)) _ using 2 <;>
      /-
        case h.e'_1.h.e'_7
        p : Turing.PartrecToTM2.Γ' → Bool
        k₁ k₂ : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        h₁ : Ne k₁ k₂
        a : Turing.PartrecToTM2.Γ'
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        Sk fst✝¹ : List Turing.PartrecToTM2.Γ'
        fst✝ : Option Turing.PartrecToTM2.Γ'
        snd✝ : List Turing.PartrecToTM2.Γ'
        e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
        e₁ : Eq (S k₁) (List.cons a Sk)
        e₂ : Eq (p a) Bool.false
        IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
        ⊢ Eq (Function.update (Function.update S k₁ Sk) k₂ (List.cons (Option.some a). …
      -/
      /-
        🎉 no goals
      -/
      simp [Function.update_of_ne, h₁, h₁.symm, e₃, List.reverseAux]
      /-
        🎉 no goals
      -/
    /-
      case h.e'_2.h.e'_7
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      h₁ : Ne k₁ k₂
      a : Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      Sk fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      e₁ : Eq (S k₁) (List.cons a Sk)
      e₂ : Eq (p a) Bool.false
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      ⊢ Eq (Function.update (Function.update S k₁ snd✝) k₂ (HAppend.hAppend fst✝¹.re …
    -/
    simp [Function.update_comm h₁.symm]
    /-
      🎉 no goals
    -/


theorem unrev_ok {q s} {S : K' → List Γ'} :
    Reaches₁ (TM2.step tr) ⟨some (unrev q), s, S⟩
      ⟨some q, none, update (update S rev []) main (List.reverseAux (S rev) (S main))⟩ :=
              /-
                q : Turing.PartrecToTM2.Λ'
                s : Option Turing.PartrecToTM2.Γ'
                S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
                ⊢ Ne Turing.PartrecToTM2.K'.rev Turing.PartrecToTM2.K'.main
              -/
  move_ok (by decide) <| splitAtPred_false _
              /-
                🎉 no goals
              -/


theorem move₂_ok {p k₁ k₂ q s L₁ o L₂} {S : K' → List Γ'} (h₁ : k₁ ≠ rev ∧ k₂ ≠ rev ∧ k₁ ≠ k₂)
    (h₂ : S rev = []) (e : splitAtPred p (S k₁) = (L₁, o, L₂)) :
    Reaches₁ (TM2.step tr) ⟨some (move₂ p k₁ k₂ q), s, S⟩
      ⟨some q, none, update (update S k₁ (o.elim id List.cons L₂)) k₂ (L₁ ++ S k₂)⟩ := by
  /-
    p : Turing.PartrecToTM2.Γ' → Bool
    k₁ k₂ : Turing.PartrecToTM2.K'
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L₁ : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ'
    L₂ : List Turing.PartrecToTM2.Γ'
    S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
    h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
    h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
    e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  refine (move_ok h₁.1 e).trans (TransGen.head rfl ?_)
  /-
    p : Turing.PartrecToTM2.Γ' → Bool
    k₁ k₂ : Turing.PartrecToTM2.K'
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L₁ : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ'
    L₂ : List Turing.PartrecToTM2.Γ'
    S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
    h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
    h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
    e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
    ⊢ Relation.TransGen (fun a b => Membership.mem (Turing.TM2.step Turing.Partrec …
  -/
  simp only [TM2.step, Option.mem_def, TM2.stepAux, id_eq, ne_eq, Option.elim]
  /-
    p : Turing.PartrecToTM2.Γ' → Bool
    k₁ k₂ : Turing.PartrecToTM2.K'
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L₁ : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ'
    L₂ : List Turing.PartrecToTM2.Γ'
    S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
    h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
    h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
    e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
    ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
  -/
  cases o <;> simp only [Option.elim, id]
    /-
      case none
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
  · simp only [TM2.stepAux, Option.isSome, cond_false]
    /-
      case none
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    convert move_ok h₁.2.1.symm (splitAtPred_false _) using 2
    /-
      case h.e'_2.h.e'_7
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Eq (Function.update (Function.update S k₁ L₂) k₂ (HAppend.hAppend L₁ (S k₂)) …
    -/
    simp only [Function.update_comm h₁.1, Function.update_idem]
    /-
      case h.e'_2.h.e'_7
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Eq (Function.update (Function.update S k₁ L₂) k₂ (HAppend.hAppend L₁ (S k₂)) …
    -/
    rw [show update S rev [] = S by rw [← h₂, Function.update_eq_self]]
    simp only [Function.update_of_ne h₁.2.2.symm, Function.update_of_ne h₁.2.1,
      Function.update_of_ne h₁.1.symm, List.reverseAux_eq, h₂, Function.update_self,
      List.append_nil, List.reverse_reverse]
    /-
      case some
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      val✝ : Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
  · simp only [TM2.stepAux, Option.isSome, cond_true]
    /-
      case some
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      val✝ : Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    convert move_ok h₁.2.1.symm (splitAtPred_false _) using 2
    simp only [h₂, Function.update_comm h₁.1, List.reverseAux_eq, Function.update_self,
      List.append_nil, Function.update_idem]
    /-
      case h.e'_2.h.e'_7
      p : Turing.PartrecToTM2.Γ' → Bool
      k₁ k₂ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ L₂ : List Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      h₁ : And (Ne k₁ Turing.PartrecToTM2.K'.rev) (And (Ne k₂ Turing.PartrecToTM2.K' …
      h₂ : Eq (S Turing.PartrecToTM2.K'.rev) List.nil
      val✝ : Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k₁)) { fst := L₁, snd := { fst := …
      ⊢ Eq (Function.update (Function.update S k₁ (List.cons val✝ L₂)) k₂ (HAppend.h …
    -/
    rw [show update S rev [] = S by rw [← h₂, Function.update_eq_self]]
    simp only [Function.update_of_ne h₁.1.symm, Function.update_of_ne h₁.2.2.symm,
      Function.update_of_ne h₁.2.1, Function.update_self, List.reverse_reverse]


theorem clear_ok {p k q s L₁ o L₂} {S : K' → List Γ'} (e : splitAtPred p (S k) = (L₁, o, L₂)) :
    Reaches₁ (TM2.step tr) ⟨some (Λ'.clear p k q), s, S⟩ ⟨some q, o, update S k L₂⟩ := by
  /-
    p : Turing.PartrecToTM2.Γ' → Bool
    k : Turing.PartrecToTM2.K'
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L₁ : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ'
    L₂ : List Turing.PartrecToTM2.Γ'
    S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
    e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := L₁, snd := { fst :=  …
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  induction' L₁ with a L₁ IH generalizing S s
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := List.nil, snd := { f …
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · refine TransGen.head' rfl ?_
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := List.nil, snd := { f …
      ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM2.step Turing.Par …
    -/
    simp only [TM2.step, Option.mem_def, TM2.stepAux, Option.elim]
    /-
      case nil
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := List.nil, snd := { f …
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    revert e; cases' S k with a Sk <;> intro e
      /-
        case nil.nil
        p : Turing.PartrecToTM2.Γ' → Bool
        k : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        e : Eq (Turing.PartrecToTM2.splitAtPred p List.nil) { fst := List.nil, snd :=  …
        ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
      -/
    · cases e
      /-
        case nil.nil.refl
        p : Turing.PartrecToTM2.Γ' → Bool
        k : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case nil.cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (List.cons a Sk)) { fst := List.nil, …
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    simp only [splitAtPred, Option.elim, List.head?, List.tail_cons] at e ⊢
    /-
      case nil.cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : Eq (cond (p a) { fst := List.nil, snd := { fst := Option.some a, snd := Sk …
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    revert e; cases p a <;> intro e <;>
      /-
        case nil.cons.false
        p : Turing.PartrecToTM2.Γ' → Bool
        k : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        a : Turing.PartrecToTM2.Γ'
        Sk : List Turing.PartrecToTM2.Γ'
        e : Eq (cond Bool.false { fst := List.nil, snd := { fst := Option.some a, snd  …
        ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
      -/
      /-
        🎉 no goals
      -/
      simp only [cond_false, cond_true, Prod.mk.injEq, true_and, false_and, reduceCtorEq] at e ⊢
    /-
      case nil.cons.true
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : And (Eq (Option.some a) o) (Eq Sk L₂)
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    rcases e with ⟨e₁, e₂⟩
    /-
      case nil.cons.true.intro
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (Option.some a) o
      e₂ : Eq Sk L₂
      ⊢ Relation.ReflTransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Opti …
    -/
    rw [e₁, e₂]
    /-
      🎉 no goals
    -/
    /-
      case cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := List.cons a L₁, snd  …
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · refine TransGen.head rfl ?_
    /-
      case cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := List.cons a L₁, snd  …
      ⊢ Relation.TransGen (fun a b => Membership.mem (Turing.TM2.step Turing.Partrec …
    -/
    simp only [TM2.step, Option.mem_def, TM2.stepAux, Option.elim]
    /-
      case cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.splitAtPred p (S k)) { fst := List.cons a L₁, snd  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    cases' e₁ : S k with a' Sk <;> rw [e₁, splitAtPred] at e
      /-
        case cons.nil
        p : Turing.PartrecToTM2.Γ' → Bool
        k : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        a : Turing.PartrecToTM2.Γ'
        L₁ : List Turing.PartrecToTM2.Γ'
        IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        e : Eq { fst := List.nil, snd := { fst := Option.none, snd := List.nil } } { f …
        e₁ : Eq (S k) List.nil
        ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
      -/
    · cases e
      /-
        🎉 no goals
      -/
    /-
      case cons.cons
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e : Eq (cond (p a') { fst := List.nil, snd := { fst := Option.some a', snd :=  …
      e₁ : Eq (S k) (List.cons a' Sk)
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    cases e₂ : p a' <;> simp only [e₂, cond] at e
    /-
      case cons.cons.false
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      e : Eq { fst := List.cons a' (Turing.PartrecToTM2.splitAtPred p Sk).1, snd :=  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    swap
      /-
        case cons.cons.true
        p : Turing.PartrecToTM2.Γ' → Bool
        k : Turing.PartrecToTM2.K'
        q : Turing.PartrecToTM2.Λ'
        o : Option Turing.PartrecToTM2.Γ'
        L₂ : List Turing.PartrecToTM2.Γ'
        a : Turing.PartrecToTM2.Γ'
        L₁ : List Turing.PartrecToTM2.Γ'
        IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
        s : Option Turing.PartrecToTM2.Γ'
        S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
        a' : Turing.PartrecToTM2.Γ'
        Sk : List Turing.PartrecToTM2.Γ'
        e₁ : Eq (S k) (List.cons a' Sk)
        e₂ : Eq (p a') Bool.true
        e : Eq { fst := List.nil, snd := { fst := Option.some a', snd := Sk } } { fst  …
        ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
      -/
    · cases e
      /-
        🎉 no goals
      -/
    /-
      case cons.cons.false
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      e : Eq { fst := List.cons a' (Turing.PartrecToTM2.splitAtPred p Sk).1, snd :=  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    rcases e₃ : splitAtPred p Sk with ⟨_, _, _⟩
    /-
      case cons.cons.false.mk.mk
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      e : Eq { fst := List.cons a' (Turing.PartrecToTM2.splitAtPred p Sk).1, snd :=  …
      fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    rw [e₃] at e
    /-
      case cons.cons.false.mk.mk
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      o : Option Turing.PartrecToTM2.Γ'
      L₂ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      L₁ : List Turing.PartrecToTM2.Γ'
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      a' : Turing.PartrecToTM2.Γ'
      Sk : List Turing.PartrecToTM2.Γ'
      e₁ : Eq (S k) (List.cons a' Sk)
      e₂ : Eq (p a') Bool.false
      fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e : Eq { fst := List.cons a' { fst := fst✝¹, snd := { fst := fst✝, snd := snd✝ …
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    cases e
    /-
      case cons.cons.false.mk.mk.refl
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      a : Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      Sk fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      e₁ : Eq (S k) (List.cons a Sk)
      e₂ : Eq (p a) Bool.false
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    simp only [List.head?_cons, e₂, List.tail_cons, cond_false]
    /-
      case cons.cons.false.mk.mk.refl
      p : Turing.PartrecToTM2.Γ' → Bool
      k : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      a : Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      S : Turing.PartrecToTM2.K' → List Turing.PartrecToTM2.Γ'
      Sk fst✝¹ : List Turing.PartrecToTM2.Γ'
      fst✝ : Option Turing.PartrecToTM2.Γ'
      snd✝ : List Turing.PartrecToTM2.Γ'
      e₃ : Eq (Turing.PartrecToTM2.splitAtPred p Sk) { fst := fst✝¹, snd := { fst := …
      e₁ : Eq (S k) (List.cons a Sk)
      e₂ : Eq (p a) Bool.false
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} {S : Turing.PartrecToTM2.K' → List  …
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
                                                /-
                                                  🎉 no goals
                                                -/
    convert @IH _ (update S k Sk) _ using 2 <;> simp [e₃]
                                                /-
                                                  🎉 no goals
                                                -/


theorem copy_ok (q s a b c d) :
    Reaches₁ (TM2.step tr) ⟨some (Λ'.copy q), s, K'.elim a b c d⟩
      ⟨some q, none, K'.elim (List.reverseAux b a) [] c (List.reverseAux b d)⟩ := by
  /-
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    a b c d : List Turing.PartrecToTM2.Γ'
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  induction' b with x b IH generalizing a d s
    /-
      case nil
      q : Turing.PartrecToTM2.Λ'
      c : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      a d : List Turing.PartrecToTM2.Γ'
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · refine TransGen.single ?_
    /-
      case nil
      q : Turing.PartrecToTM2.Λ'
      c : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      a d : List Turing.PartrecToTM2.Γ'
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr { l := Option.some q. …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    q : Turing.PartrecToTM2.Λ'
    c : List Turing.PartrecToTM2.Γ'
    x : Turing.PartrecToTM2.Γ'
    b : List Turing.PartrecToTM2.Γ'
    IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (a d : List Turing.PartrecToTM2.Γ') …
    s : Option Turing.PartrecToTM2.Γ'
    a d : List Turing.PartrecToTM2.Γ'
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  refine TransGen.head rfl ?_
  simp only [TM2.step, Option.mem_def, TM2.stepAux, elim_rev, List.head?_cons, Option.isSome_some,
    List.tail_cons, elim_update_rev, ne_eq, Function.update_of_ne, elim_main, elim_update_main,
    elim_stack, elim_update_stack, cond_true, List.reverseAux_cons]
  /-
    case cons
    q : Turing.PartrecToTM2.Λ'
    c : List Turing.PartrecToTM2.Γ'
    x : Turing.PartrecToTM2.Γ'
    b : List Turing.PartrecToTM2.Γ'
    IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (a d : List Turing.PartrecToTM2.Γ') …
    s : Option Turing.PartrecToTM2.Γ'
    a d : List Turing.PartrecToTM2.Γ'
    ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
  -/
  exact IH _ _ _
  /-
    🎉 no goals
  -/


theorem trPosNum_natEnd : ∀ (n), ∀ x ∈ trPosNum n, natEnd x = false
  | PosNum.one, _, List.Mem.head _ => rfl
  | PosNum.bit0 _, _, List.Mem.head _ => rfl
  | PosNum.bit0 n, _, List.Mem.tail _ h => trPosNum_natEnd n _ h
  | PosNum.bit1 _, _, List.Mem.head _ => rfl
  | PosNum.bit1 n, _, List.Mem.tail _ h => trPosNum_natEnd n _ h


theorem trNum_natEnd : ∀ (n), ∀ x ∈ trNum n, natEnd x = false
  | Num.pos n, x, h => trPosNum_natEnd n x h


theorem trNat_natEnd (n) : ∀ x ∈ trNat n, natEnd x = false :=
  trNum_natEnd _


theorem trList_ne_consₗ : ∀ (l), ∀ x ∈ trList l, x ≠ Γ'.consₗ
  | a :: l, x, h => by
    /-
      a : Nat
      l : List Nat
      x : Turing.PartrecToTM2.Γ'
      h : Membership.mem (Turing.PartrecToTM2.trList (List.cons a l)) x
      ⊢ Ne x Turing.PartrecToTM2.Γ'.consₗ
    -/
    simp only [trList, List.mem_append, List.mem_cons] at h
    /-
      a : Nat
      l : List Nat
      x : Turing.PartrecToTM2.Γ'
      h : Or (Membership.mem (Turing.PartrecToTM2.trNat a) x) (Or (Eq x Turing.Partr …
      ⊢ Ne x Turing.PartrecToTM2.Γ'.consₗ
    -/
    obtain h | rfl | h := h
      /-
        case inl
        a : Nat
        l : List Nat
        x : Turing.PartrecToTM2.Γ'
        h : Membership.mem (Turing.PartrecToTM2.trNat a) x
        ⊢ Ne x Turing.PartrecToTM2.Γ'.consₗ
      -/
    · rintro rfl
      /-
        case inl
        a : Nat
        l : List Nat
        h : Membership.mem (Turing.PartrecToTM2.trNat a) Turing.PartrecToTM2.Γ'.consₗ
        ⊢ False
      -/
      cases trNat_natEnd _ _ h
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        a : Nat
        l : List Nat
        ⊢ Ne Turing.PartrecToTM2.Γ'.cons Turing.PartrecToTM2.Γ'.consₗ
      -/
    · rintro ⟨⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        a : Nat
        l : List Nat
        x : Turing.PartrecToTM2.Γ'
        h : Membership.mem (Turing.PartrecToTM2.trList l) x
        ⊢ Ne x Turing.PartrecToTM2.Γ'.consₗ
      -/
    · exact trList_ne_consₗ l _ h
      /-
        🎉 no goals
      -/


theorem head_main_ok {q s L} {c d : List Γ'} :
    Reaches₁ (TM2.step tr) ⟨some (head main q), s, K'.elim (trList L) [] c d⟩
      ⟨some q, none, K'.elim (trList [L.headI]) [] c d⟩ := by
  /-
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L : List Nat
    c d : List Turing.PartrecToTM2.Γ'
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  let o : Option Γ' := List.casesOn L none fun _ _ => some Γ'.cons
  refine
    (move_ok (by decide)
          (splitAtPred_eq _ _ (trNat L.headI) o (trList L.tail) (trNat_natEnd _) ?_)).trans
      (TransGen.head rfl (TransGen.head rfl ?_))
    /-
      case refine_1
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L : List Nat
      c d : List Turing.PartrecToTM2.Γ'
      o : Option Turing.PartrecToTM2.Γ' := List.casesOn L Option.none fun x x => Opt …
      ⊢ Option.elim' (And (Eq (Turing.PartrecToTM2.K'.elim (Turing.PartrecToTM2.trLi …
    -/
                /-
                  🎉 no goals
                -/
  · cases L <;> simp [o]
                /-
                  🎉 no goals
                -/
  simp only [TM2.step, Option.mem_def, TM2.stepAux, elim_update_main, elim_rev, elim_update_rev,
    Function.update_self, trList]
  /-
    case refine_2
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L : List Nat
    c d : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ' := List.casesOn L Option.none fun x x => Opt …
    ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
  -/
  rw [if_neg (show o ≠ some Γ'.consₗ by cases L <;> simp [o])]
  /-
    case refine_2
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L : List Nat
    c d : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ' := List.casesOn L Option.none fun x x => Opt …
    ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
  -/
  refine (clear_ok (splitAtPred_eq _ _ _ none [] ?_ ⟨rfl, rfl⟩)).trans ?_
    /-
      case refine_2.refine_1
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L : List Nat
      c d : List Turing.PartrecToTM2.Γ'
      o : Option Turing.PartrecToTM2.Γ' := List.casesOn L Option.none fun x x => Opt …
      ⊢ ∀ (x : Turing.PartrecToTM2.Γ'), Membership.mem (Turing.PartrecToTM2.K'.elim  …
    -/
  · exact fun x h => Bool.decide_false (trList_ne_consₗ _ _ h)
    /-
      🎉 no goals
    -/
  /-
    case refine_2.refine_2
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L : List Nat
    c d : List Turing.PartrecToTM2.Γ'
    o : Option Turing.PartrecToTM2.Γ' := List.casesOn L Option.none fun x x => Opt …
    ⊢ Relation.TransGen (fun a b => Membership.mem (Turing.TM2.step Turing.Partrec …
  -/
  convert unrev_ok using 2; simp [List.reverseAux_eq]
                            /-
                              🎉 no goals
                            -/


theorem head_stack_ok {q s L₁ L₂ L₃} :
    Reaches₁ (TM2.step tr)
      ⟨some (head stack q), s, K'.elim (trList L₁) [] [] (trList L₂ ++ Γ'.consₗ :: L₃)⟩
      ⟨some q, none, K'.elim (trList (L₂.headI :: L₁)) [] [] L₃⟩ := by
  /-
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    L₁ L₂ : List Nat
    L₃ : List Turing.PartrecToTM2.Γ'
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  cases' L₂ with a L₂
  · refine
      TransGen.trans
        (move_ok (by decide)
          (splitAtPred_eq _ _ [] (some Γ'.consₗ) L₃ (by rintro _ ⟨⟩) ⟨rfl, rfl⟩))
        (TransGen.head rfl (TransGen.head rfl ?_))
    simp only [TM2.step, Option.mem_def, TM2.stepAux, ite_true, id_eq, trList, List.nil_append,
      elim_update_stack, elim_rev, List.reverseAux_nil, elim_update_rev, Function.update_self,
      List.headI_nil, trNat_default]
    /-
      case nil
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ : List Nat
      L₃ : List Turing.PartrecToTM2.Γ'
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    convert unrev_ok using 2
    /-
      case h.e'_2.h.e'_7
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ : List Nat
      L₃ : List Turing.PartrecToTM2.Γ'
      ⊢ Eq (Turing.PartrecToTM2.K'.elim (List.cons Turing.PartrecToTM2.Γ'.cons (Turi …
    -/
    simp
    /-
      🎉 no goals
    -/
  · refine
      TransGen.trans
        (move_ok (by decide)
          (splitAtPred_eq _ _ (trNat a) (some Γ'.cons) (trList L₂ ++ Γ'.consₗ :: L₃)
            (trNat_natEnd _) ⟨rfl, by simp⟩))
        (TransGen.head rfl (TransGen.head rfl ?_))
    simp only [TM2.step, Option.mem_def, TM2.stepAux, ite_false, trList, List.append_assoc,
      List.cons_append, elim_update_stack, elim_rev, elim_update_rev, Function.update_self,
      List.headI_cons]
    refine
      TransGen.trans
        (clear_ok
          (splitAtPred_eq _ _ (trList L₂) (some Γ'.consₗ) L₃
            (fun x h => Bool.decide_false (trList_ne_consₗ _ _ h)) ⟨rfl, by simp⟩))
        ?_
    /-
      case cons
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ : List Nat
      L₃ : List Turing.PartrecToTM2.Γ'
      a : Nat
      L₂ : List Nat
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step.match_1 (fun x => Option ( …
    -/
    convert unrev_ok using 2
    /-
      case h.e'_2.h.e'_7
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      L₁ : List Nat
      L₃ : List Turing.PartrecToTM2.Γ'
      a : Nat
      L₂ : List Nat
      ⊢ Eq (Turing.PartrecToTM2.K'.elim (HAppend.hAppend (Turing.PartrecToTM2.trNat  …
    -/
    simp [List.reverseAux_eq]
    /-
      🎉 no goals
    -/


theorem succ_ok {q s n} {c d : List Γ'} :
    Reaches₁ (TM2.step tr) ⟨some (Λ'.succ q), s, K'.elim (trList [n]) [] c d⟩
      ⟨some q, none, K'.elim (trList [n.succ]) [] c d⟩ := by
  /-
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    n : Nat
    c d : List Turing.PartrecToTM2.Γ'
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  simp only [TM2.step, trList, trNat.eq_1, Nat.cast_succ, Num.add_one]
  /-
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    n : Nat
    c d : List Turing.PartrecToTM2.Γ'
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  cases' (n : Num) with a
    /-
      case zero
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · refine TransGen.head rfl ?_
    simp only [Option.mem_def, TM2.stepAux, elim_main, decide_false, elim_update_main, ne_eq,
      Function.update_of_ne, elim_rev, elim_update_rev, decide_true, Function.update_self,
      cond_true, cond_false]
    /-
      case zero
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step Turing.PartrecToTM2.tr a)  …
    -/
    convert unrev_ok using 1
    /-
      case h.e'_2
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      ⊢ Eq { l := Option.some q, var := Option.none, stk := Turing.PartrecToTM2.K'.e …
    -/
    simp only [elim_update_rev, elim_rev, elim_main, List.reverseAux_nil, elim_update_main]
    /-
      case h.e'_2
      q : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      ⊢ Eq { l := Option.some q, var := Option.none, stk := Turing.PartrecToTM2.K'.e …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case pos
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    n : Nat
    c d : List Turing.PartrecToTM2.Γ'
    a : PosNum
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  simp only [trNum, Num.succ, Num.succ']
  suffices ∀ l₁, ∃ l₁' l₂' s',
      List.reverseAux l₁ (trPosNum a.succ) = List.reverseAux l₁' l₂' ∧
        Reaches₁ (TM2.step tr) ⟨some q.succ, s, K'.elim (trPosNum a ++ [Γ'.cons]) l₁ c d⟩
          ⟨some (unrev q), s', K'.elim (l₂' ++ [Γ'.cons]) l₁' c d⟩ by
    obtain ⟨l₁', l₂', s', e, h⟩ := this []
    simp? [List.reverseAux] at e says simp only [List.reverseAux, List.reverseAux_eq] at e
    refine h.trans ?_
    convert unrev_ok using 2
    simp [e, List.reverseAux_eq]
  /-
    case pos
    q : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    n : Nat
    c d : List Turing.PartrecToTM2.Γ'
    a : PosNum
    ⊢ ∀ (l₁ : List Turing.PartrecToTM2.Γ'), Exists fun l₁' => Exists fun l₂' => Ex …
  -/
  induction' a with m IH m _ generalizing s <;> intro l₁
    /-
      case pos.one
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
  · refine ⟨Γ'.bit0 :: l₁, [Γ'.bit1], some Γ'.cons, rfl, TransGen.head rfl (TransGen.single ?_)⟩
    /-
      case pos.one
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr (Turing.TM2.stepAux ( …
    -/
    simp [trPosNum]
    /-
      🎉 no goals
    -/
    /-
      case pos.bit1
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      m : PosNum
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
  · obtain ⟨l₁', l₂', s', e, h⟩ := IH (Γ'.bit0 :: l₁)
    /-
      case pos.bit1.intro.intro.intro.intro
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      m : PosNum
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ l₁' l₂' : List Turing.PartrecToTM2.Γ'
      s' : Option Turing.PartrecToTM2.Γ'
      e : Eq ((List.cons Turing.PartrecToTM2.Γ'.bit0 l₁).reverseAux (Turing.PartrecT …
      h : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.som …
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
    refine ⟨l₁', l₂', s', e, TransGen.head ?_ h⟩
    /-
      case pos.bit1.intro.intro.intro.intro
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      m : PosNum
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ l₁' l₂' : List Turing.PartrecToTM2.Γ'
      s' : Option Turing.PartrecToTM2.Γ'
      e : Eq ((List.cons Turing.PartrecToTM2.Γ'.bit0 l₁).reverseAux (Turing.PartrecT …
      h : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.som …
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr { l := Option.some q. …
    -/
    simp [PosNum.succ, trPosNum]
    /-
      case pos.bit1.intro.intro.intro.intro
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      m : PosNum
      IH : ∀ {s : Option Turing.PartrecToTM2.Γ'} (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ l₁' l₂' : List Turing.PartrecToTM2.Γ'
      s' : Option Turing.PartrecToTM2.Γ'
      e : Eq ((List.cons Turing.PartrecToTM2.Γ'.bit0 l₁).reverseAux (Turing.PartrecT …
      h : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.som …
      ⊢ Eq (Option.some Turing.PartrecToTM2.Γ'.bit1) ?m.256109
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case pos.bit0
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      m : PosNum
      a_ih✝ : ∀ {s : Option Turing.PartrecToTM2.Γ'} (l₁ : List Turing.PartrecToTM2.Γ …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
  · refine ⟨l₁, _, some Γ'.bit0, rfl, TransGen.single ?_⟩
    simp only [TM2.step, TM2.stepAux, elim_main, elim_update_main, ne_eq, Function.update_of_ne,
      elim_rev, elim_update_rev, Function.update_self, Option.mem_def, Option.some.injEq]
    /-
      case pos.bit0
      q : Turing.PartrecToTM2.Λ'
      n : Nat
      c d : List Turing.PartrecToTM2.Γ'
      m : PosNum
      a_ih✝ : ∀ {s : Option Turing.PartrecToTM2.Γ'} (l₁ : List Turing.PartrecToTM2.Γ …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Eq (cond (Decidable.decide (Eq (HAppend.hAppend (Turing.PartrecToTM2.trPosNu …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem pred_ok (q₁ q₂ s v) (c d : List Γ') : ∃ s',
    Reaches₁ (TM2.step tr) ⟨some (Λ'.pred q₁ q₂), s, K'.elim (trList v) [] c d⟩
      (v.headI.rec ⟨some q₁, s', K'.elim (trList v.tail) [] c d⟩ fun n _ =>
        ⟨some q₂, s', K'.elim (trList (n::v.tail)) [] c d⟩) := by
  /-
    q₁ q₂ : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    v : List Nat
    c d : List Turing.PartrecToTM2.Γ'
    ⊢ Exists fun s' => Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) {  …
  -/
  rcases v with (_ | ⟨_ | n, v⟩)
    /-
      case nil
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      ⊢ Exists fun s' => Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) {  …
    -/
  · refine ⟨none, TransGen.single ?_⟩
    /-
      case nil
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr { l := Option.some (q …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case cons.zero
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      ⊢ Exists fun s' => Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) {  …
    -/
  · refine ⟨some Γ'.cons, TransGen.single ?_⟩
    /-
      case cons.zero
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr { l := Option.some (q …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case cons.succ
    q₁ q₂ : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    c d : List Turing.PartrecToTM2.Γ'
    v : List Nat
    n : Nat
    ⊢ Exists fun s' => Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) {  …
  -/
  refine ⟨none, ?_⟩
  simp only [TM2.step, trList, trNat.eq_1, trNum, Nat.cast_succ, Num.add_one, Num.succ,
    List.tail_cons, List.headI_cons]
  /-
    case cons.succ
    q₁ q₂ : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    c d : List Turing.PartrecToTM2.Γ'
    v : List Nat
    n : Nat
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  cases' (n : Num) with a
    /-
      case cons.succ.zero
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
  · simp only [trPosNum, List.singleton_append, List.nil_append]
    /-
      case cons.succ.zero
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
    -/
    refine TransGen.head rfl ?_
    simp only [Option.mem_def, TM2.stepAux, elim_main, List.head?_cons, Option.some.injEq,
      decide_false, List.tail_cons, elim_update_main, ne_eq, Function.update_of_ne, elim_rev,
      elim_update_rev, natEnd, Function.update_self,  cond_true, cond_false]
    /-
      case cons.succ.zero
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      ⊢ Relation.TransGen (fun a b => Eq (Turing.TM2.step Turing.PartrecToTM2.tr a)  …
    -/
    convert unrev_ok using 2
    /-
      case h.e'_2.h.e'_7
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      s : Option Turing.PartrecToTM2.Γ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      ⊢ Eq (Turing.PartrecToTM2.K'.elim (List.cons Turing.PartrecToTM2.Γ'.cons (Turi …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case cons.succ.pos
    q₁ q₂ : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    c d : List Turing.PartrecToTM2.Γ'
    v : List Nat
    n : Nat
    a : PosNum
    ⊢ Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.some  …
  -/
  simp only [Num.succ']
  suffices ∀ l₁, ∃ l₁' l₂' s',
    List.reverseAux l₁ (trPosNum a) = List.reverseAux l₁' l₂' ∧
      Reaches₁ (TM2.step tr)
        ⟨some (q₁.pred q₂), s, K'.elim (trPosNum a.succ ++ Γ'.cons :: trList v) l₁ c d⟩
        ⟨some (unrev q₂), s', K'.elim (l₂' ++ Γ'.cons :: trList v) l₁' c d⟩ by
    obtain ⟨l₁', l₂', s', e, h⟩ := this []
    simp only [List.reverseAux] at e
    refine h.trans ?_
    convert unrev_ok using 2
    simp [e, List.reverseAux_eq]
  /-
    case cons.succ.pos
    q₁ q₂ : Turing.PartrecToTM2.Λ'
    s : Option Turing.PartrecToTM2.Γ'
    c d : List Turing.PartrecToTM2.Γ'
    v : List Nat
    n : Nat
    a : PosNum
    ⊢ ∀ (l₁ : List Turing.PartrecToTM2.Γ'), Exists fun l₁' => Exists fun l₂' => Ex …
  -/
  induction' a with m IH m IH generalizing s <;> intro l₁
    /-
      case cons.succ.pos.one
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
  · refine ⟨Γ'.bit1::l₁, [], some Γ'.cons, rfl, TransGen.head rfl (TransGen.single ?_)⟩
    /-
      case cons.succ.pos.one
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr (Turing.TM2.stepAux ( …
    -/
    simp [trPosNum, show PosNum.one.succ = PosNum.one.bit0 from rfl]
    /-
      🎉 no goals
    -/
    /-
      case cons.succ.pos.bit1
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      m : PosNum
      IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
  · obtain ⟨l₁', l₂', s', e, h⟩ := IH (some Γ'.bit0) (Γ'.bit1 :: l₁)
    /-
      case cons.succ.pos.bit1.intro.intro.intro.intro
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      m : PosNum
      IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ l₁' l₂' : List Turing.PartrecToTM2.Γ'
      s' : Option Turing.PartrecToTM2.Γ'
      e : Eq ((List.cons Turing.PartrecToTM2.Γ'.bit1 l₁).reverseAux (Turing.PartrecT …
      h : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.som …
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
    refine ⟨l₁', l₂', s', e, TransGen.head ?_ h⟩
    /-
      case cons.succ.pos.bit1.intro.intro.intro.intro
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      m : PosNum
      IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ l₁' l₂' : List Turing.PartrecToTM2.Γ'
      s' : Option Turing.PartrecToTM2.Γ'
      e : Eq ((List.cons Turing.PartrecToTM2.Γ'.bit1 l₁).reverseAux (Turing.PartrecT …
      h : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.som …
      ⊢ Membership.mem (Turing.TM2.step Turing.PartrecToTM2.tr { l := Option.some (q …
    -/
    simp
    /-
      case cons.succ.pos.bit1.intro.intro.intro.intro
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      m : PosNum
      IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ l₁' l₂' : List Turing.PartrecToTM2.Γ'
      s' : Option Turing.PartrecToTM2.Γ'
      e : Eq ((List.cons Turing.PartrecToTM2.Γ'.bit1 l₁).reverseAux (Turing.PartrecT …
      h : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) { l := Option.som …
      ⊢ Eq (cond (Decidable.decide (Eq ((Turing.PartrecToTM2.trPosNum m.bit1.succ).h …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · obtain ⟨a, l, e, h⟩ : ∃ a l, (trPosNum m = a::l) ∧ natEnd a = false := by
      cases m <;> refine ⟨_, _, rfl, rfl⟩
    /-
      case cons.succ.pos.bit0.intro.intro.intro
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      m : PosNum
      IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      l : List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.trPosNum m) (List.cons a l)
      h : Eq (Turing.PartrecToTM2.natEnd a) Bool.false
      ⊢ Exists fun l₁' => Exists fun l₂' => Exists fun s' => And (Eq (l₁.reverseAux  …
    -/
    refine ⟨Γ'.bit0 :: l₁, _, some a, rfl, TransGen.single ?_⟩
    simp [trPosNum, PosNum.succ, e, h, show some Γ'.bit1 ≠ some Γ'.bit0 by decide,
      Option.iget, -natEnd]
    /-
      case cons.succ.pos.bit0.intro.intro.intro
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      c d : List Turing.PartrecToTM2.Γ'
      v : List Nat
      n : Nat
      m : PosNum
      IH : ∀ (s : Option Turing.PartrecToTM2.Γ') (l₁ : List Turing.PartrecToTM2.Γ'), …
      s : Option Turing.PartrecToTM2.Γ'
      l₁ : List Turing.PartrecToTM2.Γ'
      a : Turing.PartrecToTM2.Γ'
      l : List Turing.PartrecToTM2.Γ'
      e : Eq (Turing.PartrecToTM2.trPosNum m) (List.cons a l)
      h : Eq (Turing.PartrecToTM2.natEnd a) Bool.false
      ⊢ Eq (cond (Turing.PartrecToTM2.natEnd Turing.PartrecToTM2.Γ'.bit1) { l := Opt …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem trNormal_respects (c k v s) :
    ∃ b₂,
      TrCfg (stepNormal c k v) b₂ ∧
        Reaches₁ (TM2.step tr)
          ⟨some (trNormal c (trCont k)), s, K'.elim (trList v) [] [] (trContStack k)⟩ b₂ := by
  induction c generalizing k v s with
  | zero' => refine ⟨_, ⟨s, rfl⟩, TransGen.single ?_⟩; simp
  | succ => refine ⟨_, ⟨none, rfl⟩, head_main_ok.trans succ_ok⟩
  | tail =>
    let o : Option Γ' := List.casesOn v none fun _ _ => some Γ'.cons
    refine ⟨_, ⟨o, rfl⟩, ?_⟩; convert clear_ok _ using 2
    · simp; rfl
    swap
    refine splitAtPred_eq _ _ (trNat v.headI) _ _ (trNat_natEnd _) ?_
    cases v <;> simp [o]
  | cons f fs IHf _ =>
    obtain ⟨c, h₁, h₂⟩ := IHf (Cont.cons₁ fs v k) v none
    refine ⟨c, h₁, TransGen.head rfl <| (move_ok (by decide) (splitAtPred_false _)).trans ?_⟩
    simp only [TM2.step, Option.mem_def, elim_stack, elim_update_stack, elim_update_main, ne_eq,
      Function.update_of_ne, elim_main, elim_rev, elim_update_rev]
    refine (copy_ok _ none [] (trList v).reverse _ _).trans ?_
    convert h₂ using 2
    simp [List.reverseAux_eq, trContStack]
  | comp f _ _ IHg => exact IHg (Cont.comp f k) v s
  | case f g IHf IHg =>
    rw [stepNormal]
    simp only
    obtain ⟨s', h⟩ := pred_ok _ _ s v _ _
    revert h; cases' v.headI with n <;> intro h
    · obtain ⟨c, h₁, h₂⟩ := IHf k _ s'
      exact ⟨_, h₁, h.trans h₂⟩
    · obtain ⟨c, h₁, h₂⟩ := IHg k _ s'
      exact ⟨_, h₁, h.trans h₂⟩
  | fix f IH => apply IH


theorem tr_ret_respects (k v s) : ∃ b₂,
    TrCfg (stepRet k v) b₂ ∧
      Reaches₁ (TM2.step tr)
        ⟨some (Λ'.ret (trCont k)), s, K'.elim (trList v) [] [] (trContStack k)⟩ b₂ := by
  induction k generalizing v s with
  | halt => exact ⟨_, rfl, TransGen.single rfl⟩
  | cons₁ fs as k _ =>
    obtain ⟨s', h₁, h₂⟩ := trNormal_respects fs (Cont.cons₂ v k) as none
    refine ⟨s', h₁, TransGen.head rfl ?_⟩; simp
    refine (move₂_ok (by decide) ?_ (splitAtPred_false _)).trans ?_; · rfl
    simp only [TM2.step, Option.mem_def, Option.elim, id_eq, elim_update_main, elim_main, elim_aux,
      List.append_nil, elim_update_aux]
    refine (move₂_ok (L₁ := ?_) (o := ?_) (L₂ := ?_) (by decide) rfl ?_).trans ?_
    pick_goal 4
    · exact splitAtPred_eq _ _ _ (some Γ'.consₗ) _
        (fun x h => Bool.decide_false (trList_ne_consₗ _ _ h)) ⟨rfl, rfl⟩
    refine (move₂_ok (by decide) ?_ (splitAtPred_false _)).trans ?_; · rfl
    simp only [TM2.step, Option.mem_def, Option.elim, elim_update_stack, elim_main,
      List.append_nil, elim_update_main,  id_eq, elim_update_aux, ne_eq, Function.update_of_ne,
      elim_aux, elim_stack]
    exact h₂
  | cons₂ ns k IH =>
    obtain ⟨c, h₁, h₂⟩ := IH (ns.headI :: v) none
    exact ⟨c, h₁, TransGen.head rfl <| head_stack_ok.trans h₂⟩
  | comp f k _ =>
    obtain ⟨s', h₁, h₂⟩ := trNormal_respects f k v s
    exact ⟨_, h₁, TransGen.head rfl h₂⟩
  | fix f k IH =>
    rw [stepRet]
    have :
      if v.headI = 0 then natEnd (trList v).head?.iget = true ∧ (trList v).tail = trList v.tail
      else
        natEnd (trList v).head?.iget = false ∧
          (trList v).tail = (trNat v.headI).tail ++ Γ'.cons :: trList v.tail := by
      cases' v with n
      · exact ⟨rfl, rfl⟩
      cases' n with n
      · simp
      rw [trList, List.headI, trNat, Nat.cast_succ, Num.add_one, Num.succ, List.tail]
      cases (n : Num).succ' <;> exact ⟨rfl, rfl⟩
    by_cases h : v.headI = 0 <;> simp only [h, ite_true, ite_false] at this ⊢
    · obtain ⟨c, h₁, h₂⟩ := IH v.tail (trList v).head?
      refine ⟨c, h₁, TransGen.head rfl ?_⟩
      simp only [Option.mem_def, TM2.stepAux, trContStack, contStack, elim_main, this, cond_true,
        elim_update_main]
      exact h₂
    · obtain ⟨s', h₁, h₂⟩ := trNormal_respects f (Cont.fix f k) v.tail (some Γ'.cons)
      refine ⟨_, h₁, TransGen.head rfl <| TransGen.trans ?_ h₂⟩
      simp only [Option.mem_def, TM2.stepAux, elim_main, this.1, cond_false, elim_update_main,
        trCont]
      convert clear_ok (splitAtPred_eq _ _ (trNat v.headI).tail (some Γ'.cons) _ _ _) using 2
      · simp
        convert rfl
      · exact fun x h => trNat_natEnd _ _ (List.tail_subset _ h)
      · exact ⟨rfl, this.2⟩


theorem tr_respects : Respects step (TM2.step tr) TrCfg
  | Cfg.ret _ _, _, ⟨_, rfl⟩ => tr_ret_respects _ _ _
  | Cfg.halt _, _, rfl => rfl


/-- The initial state, evaluating function `c` on input `v`. -/
def init (c : Code) (v : List ℕ) : Cfg' :=
  ⟨some (trNormal c Cont'.halt), none, K'.elim (trList v) [] [] []⟩


theorem tr_init (c v) :
    ∃ b, TrCfg (stepNormal c Cont.halt v) b ∧ Reaches₁ (TM2.step tr) (init c v) b :=
  trNormal_respects _ _ _ _


theorem tr_eval (c v) : eval (TM2.step tr) (init c v) = halt <$> Code.eval c v := by
  /-
    c : Turing.ToPartrec.Code
    v : List Nat
    ⊢ Eq (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecToTM …
  -/
  obtain ⟨i, h₁, h₂⟩ := tr_init c v
  /-
    case intro.intro
    c : Turing.ToPartrec.Code
    v : List Nat
    i : Turing.PartrecToTM2.Cfg'
    h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
    h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
    ⊢ Eq (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecToTM …
  -/
  refine Part.ext fun x => ?_
  /-
    case intro.intro
    c : Turing.ToPartrec.Code
    v : List Nat
    i : Turing.PartrecToTM2.Cfg'
    h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
    h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
    x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
    ⊢ Iff (Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) (T …
  -/
  rw [reaches_eval h₂.to_reflTransGen]; simp only [Part.map_eq_map, Part.mem_map_iff]
  /-
    case intro.intro
    c : Turing.ToPartrec.Code
    v : List Nat
    i : Turing.PartrecToTM2.Cfg'
    h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
    h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
    x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
    ⊢ Iff (Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case intro.intro.refine_1
      c : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
      h : Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) x
      ⊢ Exists fun a => And (Membership.mem (c.eval v) a) (Eq (Turing.PartrecToTM2.h …
    -/
  · obtain ⟨c, hc₁, hc₂⟩ := tr_eval_rev tr_respects h₁ h
    /-
      case intro.intro.refine_1.intro.intro
      c✝ : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c✝ Turing.ToPartre …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
      h : Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) x
      c : Turing.ToPartrec.Cfg
      hc₁ : Turing.PartrecToTM2.TrCfg c x
      hc₂ : Membership.mem (Turing.eval Turing.ToPartrec.step (Turing.ToPartrec.step …
      ⊢ Exists fun a => And (Membership.mem (c✝.eval v) a) (Eq (Turing.PartrecToTM2. …
    -/
    simp [stepNormal_eval] at hc₂
    /-
      case intro.intro.refine_1.intro.intro
      c✝ : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c✝ Turing.ToPartre …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
      h : Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) x
      c : Turing.ToPartrec.Cfg
      hc₁ : Turing.PartrecToTM2.TrCfg c x
      hc₂ : Exists fun a => And (Membership.mem (c✝.eval v) a) (Eq (Turing.ToPartrec …
      ⊢ Exists fun a => And (Membership.mem (c✝.eval v) a) (Eq (Turing.PartrecToTM2. …
    -/
    obtain ⟨v', hv, rfl⟩ := hc₂
    /-
      case intro.intro.refine_1.intro.intro.intro.intro
      c : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
      h : Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) x
      v' : List Nat
      hv : Membership.mem (c.eval v) v'
      hc₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.Cfg.halt v') x
      ⊢ Exists fun a => And (Membership.mem (c.eval v) a) (Eq (Turing.PartrecToTM2.h …
    -/
    exact ⟨_, hv, hc₁.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      c : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      x : Turing.TM2.Cfg (fun x => Turing.PartrecToTM2.Γ') Turing.PartrecToTM2.Λ' (O …
      ⊢ (Exists fun a => And (Membership.mem (c.eval v) a) (Eq (Turing.PartrecToTM2. …
    -/
  · rintro ⟨v', hv, rfl⟩
    /-
      case intro.intro.refine_2.intro.intro
      c : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      v' : List Nat
      hv : Membership.mem (c.eval v) v'
      ⊢ Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) (Tur …
    -/
    have := Turing.tr_eval (b₁ := Cfg.halt v') tr_respects h₁
    simp only [stepNormal_eval, Part.map_eq_map, Part.mem_map_iff, Cfg.halt.injEq,
      exists_eq_right] at this
    /-
      case intro.intro.refine_2.intro.intro
      c : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      v' : List Nat
      hv : Membership.mem (c.eval v) v'
      this : Membership.mem (c.eval v) v' → Exists fun b₂ => And (Turing.PartrecToTM …
      ⊢ Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) (Tur …
    -/
    obtain ⟨_, ⟨⟩, h⟩ := this hv
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.refl
      c : Turing.ToPartrec.Code
      v : List Nat
      i : Turing.PartrecToTM2.Cfg'
      h₁ : Turing.PartrecToTM2.TrCfg (Turing.ToPartrec.stepNormal c Turing.ToPartrec …
      h₂ : Turing.Reaches₁ (Turing.TM2.step Turing.PartrecToTM2.tr) (Turing.PartrecT …
      v' : List Nat
      hv : Membership.mem (c.eval v) v'
      this : Membership.mem (c.eval v) v' → Exists fun b₂ => And (Turing.PartrecToTM …
      h : Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) (T …
      ⊢ Membership.mem (Turing.eval (Turing.TM2.step Turing.PartrecToTM2.tr) i) (Tur …
    -/
    exact h
    /-
      🎉 no goals
    -/


/-- The set of machine states reachable via downward label jumps, discounting jumps via `ret`. -/
def trStmts₁ : Λ' → Finset Λ'
  | Q@(Λ'.move _ _ _ q) => insert Q <| trStmts₁ q
  | Q@(Λ'.push _ _ q) => insert Q <| trStmts₁ q
  | Q@(Λ'.read q) => insert Q <| Finset.univ.biUnion fun s => trStmts₁ (q s)
  | Q@(Λ'.clear _ _ q) => insert Q <| trStmts₁ q
  | Q@(Λ'.copy q) => insert Q <| trStmts₁ q
  | Q@(Λ'.succ q) => insert Q <| insert (unrev q) <| trStmts₁ q
  | Q@(Λ'.pred q₁ q₂) => insert Q <| trStmts₁ q₁ ∪ insert (unrev q₂) (trStmts₁ q₂)
  | Q@(Λ'.ret _) => {Q}


theorem trStmts₁_trans {q q'} : q' ∈ trStmts₁ q → trStmts₁ q' ⊆ trStmts₁ q := by
  induction q with
  | move _ _ _ q q_ih => _ | clear _ _ q q_ih => _ | copy q q_ih => _ | push _ _ q q_ih => _
  | read q q_ih => _ | succ q q_ih => _ | pred q₁ q₂ q₁_ih q₂_ih => _ | ret => _ <;>
  all_goals
    simp +contextual only [trStmts₁, Finset.mem_insert, Finset.mem_union,
      or_imp, Finset.mem_singleton, Finset.Subset.refl, imp_true_iff, true_and]
    repeat exact fun h => Finset.Subset.trans (q_ih h) (Finset.subset_insert _ _)
    /-
      case read
      q' : Turing.PartrecToTM2.Λ'
      q : Option Turing.PartrecToTM2.Γ' → Turing.PartrecToTM2.Λ'
      q_ih : ∀ (a : Option Turing.PartrecToTM2.Γ'), Membership.mem (Turing.PartrecTo …
      ⊢ Membership.mem (Finset.univ.biUnion fun s => Turing.PartrecToTM2.trStmts₁ (q …
    -/
  · simp
    /-
      case read
      q' : Turing.PartrecToTM2.Λ'
      q : Option Turing.PartrecToTM2.Γ' → Turing.PartrecToTM2.Λ'
      q_ih : ∀ (a : Option Turing.PartrecToTM2.Γ'), Membership.mem (Turing.PartrecTo …
      ⊢ ∀ (x : Option Turing.PartrecToTM2.Γ'), Membership.mem (Turing.PartrecToTM2.t …
    -/
    intro s h x h'
    /-
      case read
      q' : Turing.PartrecToTM2.Λ'
      q : Option Turing.PartrecToTM2.Γ' → Turing.PartrecToTM2.Λ'
      q_ih : ∀ (a : Option Turing.PartrecToTM2.Γ'), Membership.mem (Turing.PartrecTo …
      s : Option Turing.PartrecToTM2.Γ'
      h : Membership.mem (Turing.PartrecToTM2.trStmts₁ (q s)) q'
      x : Turing.PartrecToTM2.Λ'
      h' : Membership.mem (Turing.PartrecToTM2.trStmts₁ q') x
      ⊢ Membership.mem (Insert.insert (Turing.PartrecToTM2.Λ'.read q) (Finset.univ.b …
    -/
    simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_insert]
    /-
      case read
      q' : Turing.PartrecToTM2.Λ'
      q : Option Turing.PartrecToTM2.Γ' → Turing.PartrecToTM2.Λ'
      q_ih : ∀ (a : Option Turing.PartrecToTM2.Γ'), Membership.mem (Turing.PartrecTo …
      s : Option Turing.PartrecToTM2.Γ'
      h : Membership.mem (Turing.PartrecToTM2.trStmts₁ (q s)) q'
      x : Turing.PartrecToTM2.Λ'
      h' : Membership.mem (Turing.PartrecToTM2.trStmts₁ q') x
      ⊢ Or (Eq x (Turing.PartrecToTM2.Λ'.read q)) (Exists fun a => Membership.mem (T …
    -/
    exact Or.inr ⟨_, q_ih s h h'⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      q' q : Turing.PartrecToTM2.Λ'
      q_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q' → HasSubset.Subset ( …
      ⊢ And (Eq q' (Turing.PartrecToTM2.unrev q) → HasSubset.Subset (Insert.insert ( …
    -/
  · constructor
      /-
        case succ.left
        q' q : Turing.PartrecToTM2.Λ'
        q_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q' → HasSubset.Subset ( …
        ⊢ Eq q' (Turing.PartrecToTM2.unrev q) → HasSubset.Subset (Insert.insert (Turin …
      -/
    · rintro rfl
      /-
        case succ.left
        q : Turing.PartrecToTM2.Λ'
        q_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) (Turing.PartrecToTM2.un …
        ⊢ HasSubset.Subset (Insert.insert (Turing.PartrecToTM2.Λ'.move (fun x => Bool. …
      -/
      apply Finset.subset_insert
      /-
        🎉 no goals
      -/
      /-
        case succ.right
        q' q : Turing.PartrecToTM2.Λ'
        q_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q' → HasSubset.Subset ( …
        ⊢ Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q' → HasSubset.Subset (Turin …
      -/
    · intro h x h'
      /-
        case succ.right
        q' q : Turing.PartrecToTM2.Λ'
        q_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q' → HasSubset.Subset ( …
        h : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q'
        x : Turing.PartrecToTM2.Λ'
        h' : Membership.mem (Turing.PartrecToTM2.trStmts₁ q') x
        ⊢ Membership.mem (Insert.insert q.succ (Insert.insert (Turing.PartrecToTM2.unr …
      -/
      simp only [Finset.mem_insert]
      /-
        case succ.right
        q' q : Turing.PartrecToTM2.Λ'
        q_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q' → HasSubset.Subset ( …
        h : Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q'
        x : Turing.PartrecToTM2.Λ'
        h' : Membership.mem (Turing.PartrecToTM2.trStmts₁ q') x
        ⊢ Or (Eq x q.succ) (Or (Eq x (Turing.PartrecToTM2.unrev q)) (Membership.mem (T …
      -/
      exact Or.inr (Or.inr <| q_ih h h')
      /-
        🎉 no goals
      -/
    /-
      case pred
      q' q₁ q₂ : Turing.PartrecToTM2.Λ'
      q₁_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₁) q' → HasSubset.Subset …
      q₂_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₂) q' → HasSubset.Subset …
      ⊢ And (Membership.mem (Turing.PartrecToTM2.trStmts₁ q₁) q' → HasSubset.Subset  …
    -/
  · refine ⟨fun h x h' => ?_, fun _ x h' => ?_, fun h x h' => ?_⟩ <;> simp
      /-
        case pred.refine_1
        q' q₁ q₂ : Turing.PartrecToTM2.Λ'
        q₁_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₁) q' → HasSubset.Subset …
        q₂_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₂) q' → HasSubset.Subset …
        h : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₁) q'
        x : Turing.PartrecToTM2.Λ'
        h' : Membership.mem (Turing.PartrecToTM2.trStmts₁ q') x
        ⊢ Or (Eq x (q₁.pred q₂)) (Or (Eq x (Turing.PartrecToTM2.unrev q₂)) (Or (Member …
      -/
    · exact Or.inr (Or.inr <| Or.inl <| q₁_ih h h')
      /-
        🎉 no goals
      -/
      /-
        case pred.refine_2
        q' q₁ q₂ : Turing.PartrecToTM2.Λ'
        q₁_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₁) q' → HasSubset.Subset …
        q₂_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₂) q' → HasSubset.Subset …
        x✝ : Eq q' (Turing.PartrecToTM2.unrev q₂)
        x : Turing.PartrecToTM2.Λ'
        h' : Membership.mem (Insert.insert (Turing.PartrecToTM2.Λ'.move (fun x => Bool …
        ⊢ Or (Eq x (q₁.pred q₂)) (Or (Eq x (Turing.PartrecToTM2.unrev q₂)) (Or (Member …
      -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    · cases' Finset.mem_insert.1 h' with h' h' <;> simp [h', unrev]
                                                   /-
                                                     🎉 no goals
                                                   -/
      /-
        case pred.refine_3
        q' q₁ q₂ : Turing.PartrecToTM2.Λ'
        q₁_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₁) q' → HasSubset.Subset …
        q₂_ih : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₂) q' → HasSubset.Subset …
        h : Membership.mem (Turing.PartrecToTM2.trStmts₁ q₂) q'
        x : Turing.PartrecToTM2.Λ'
        h' : Membership.mem (Turing.PartrecToTM2.trStmts₁ q') x
        ⊢ Or (Eq x (q₁.pred q₂)) (Or (Eq x (Turing.PartrecToTM2.unrev q₂)) (Or (Member …
      -/
    · exact Or.inr (Or.inr <| Or.inr <| q₂_ih h h')
      /-
        🎉 no goals
      -/


theorem trStmts₁_self (q) : q ∈ trStmts₁ q := by
  /-
    q : Turing.PartrecToTM2.Λ'
    ⊢ Membership.mem (Turing.PartrecToTM2.trStmts₁ q) q
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
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
  induction q <;> · first |apply Finset.mem_singleton_self|apply Finset.mem_insert_self
                    /-
                      🎉 no goals
                    -/


/-- The (finite!) set of machine states visited during the course of evaluation of `c`,
including the state `ret k` but not any states after that (that is, the states visited while
evaluating `k`). -/
def codeSupp' : Code → Cont' → Finset Λ'
  | c@Code.zero', k => trStmts₁ (trNormal c k)
  | c@Code.succ, k => trStmts₁ (trNormal c k)
  | c@Code.tail, k => trStmts₁ (trNormal c k)
  | c@(Code.cons f fs), k =>
    trStmts₁ (trNormal c k) ∪
      (codeSupp' f (Cont'.cons₁ fs k) ∪
        (trStmts₁
            (move₂ (fun _ => false) main aux <|
              move₂ (fun s => s = Γ'.consₗ) stack main <|
                move₂ (fun _ => false) aux stack <| trNormal fs (Cont'.cons₂ k)) ∪
          (codeSupp' fs (Cont'.cons₂ k) ∪ trStmts₁ (head stack <| Λ'.ret k))))
  | c@(Code.comp f g), k =>
    trStmts₁ (trNormal c k) ∪
      (codeSupp' g (Cont'.comp f k) ∪ (trStmts₁ (trNormal f k) ∪ codeSupp' f k))
  | c@(Code.case f g), k => trStmts₁ (trNormal c k) ∪ (codeSupp' f k ∪ codeSupp' g k)
  | c@(Code.fix f), k =>
    trStmts₁ (trNormal c k) ∪
      (codeSupp' f (Cont'.fix f k) ∪
        (trStmts₁ (Λ'.clear natEnd main <| trNormal f (Cont'.fix f k)) ∪ {Λ'.ret k}))


@[simp]
theorem codeSupp'_self (c k) : trStmts₁ (trNormal c k) ⊆ codeSupp' c k := by
  /-
    c : Turing.ToPartrec.Code
    k : Turing.PartrecToTM2.Cont'
    ⊢ HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ (Turing.PartrecToTM2.trNormal …
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
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases c <;> first | rfl | exact Finset.union_subset_left (fun _ a ↦ a)
              /-
                🎉 no goals
              -/


/-- The (finite!) set of machine states visited during the course of evaluation of a continuation
`k`, not including the initial state `ret k`. -/
def contSupp : Cont' → Finset Λ'
  | Cont'.cons₁ fs k =>
    trStmts₁
        (move₂ (fun _ => false) main aux <|
          move₂ (fun s => s = Γ'.consₗ) stack main <|
            move₂ (fun _ => false) aux stack <| trNormal fs (Cont'.cons₂ k)) ∪
      (codeSupp' fs (Cont'.cons₂ k) ∪ (trStmts₁ (head stack <| Λ'.ret k) ∪ contSupp k))
  | Cont'.cons₂ k => trStmts₁ (head stack <| Λ'.ret k) ∪ contSupp k
  | Cont'.comp f k => codeSupp' f k ∪ contSupp k
  | Cont'.fix f k => codeSupp' (Code.fix f) k ∪ contSupp k
  | Cont'.halt => ∅


/-- The (finite!) set of machine states visited during the course of evaluation of `c` in
continuation `k`. This is actually closed under forward simulation (see `tr_supports`), and the
existence of this set means that the machine constructed in this section is in fact a proper
Turing machine, with a finite set of states. -/
def codeSupp (c : Code) (k : Cont') : Finset Λ' :=
  codeSupp' c k ∪ contSupp k


@[simp]
theorem codeSupp_self (c k) : trStmts₁ (trNormal c k) ⊆ codeSupp c k :=
  Finset.Subset.trans (codeSupp'_self _ _) (Finset.union_subset_left fun _ a ↦ a)


@[simp]
theorem codeSupp_zero (k) : codeSupp Code.zero' k = trStmts₁ (trNormal Code.zero' k) ∪ contSupp k :=
  rfl


@[simp]
theorem codeSupp_succ (k) : codeSupp Code.succ k = trStmts₁ (trNormal Code.succ k) ∪ contSupp k :=
  rfl


@[simp]
theorem codeSupp_tail (k) : codeSupp Code.tail k = trStmts₁ (trNormal Code.tail k) ∪ contSupp k :=
  rfl


@[simp]
theorem codeSupp_cons (f fs k) :
    codeSupp (Code.cons f fs) k =
      trStmts₁ (trNormal (Code.cons f fs) k) ∪ codeSupp f (Cont'.cons₁ fs k) := by
  /-
    f fs : Turing.ToPartrec.Code
    k : Turing.PartrecToTM2.Cont'
    ⊢ Eq (Turing.PartrecToTM2.codeSupp (f.cons fs) k) (Union.union (Turing.Partrec …
  -/
  simp [codeSupp, codeSupp', contSupp, Finset.union_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem codeSupp_comp (f g k) :
    codeSupp (Code.comp f g) k =
      trStmts₁ (trNormal (Code.comp f g) k) ∪ codeSupp g (Cont'.comp f k) := by
  /-
    f g : Turing.ToPartrec.Code
    k : Turing.PartrecToTM2.Cont'
    ⊢ Eq (Turing.PartrecToTM2.codeSupp (f.comp g) k) (Union.union (Turing.PartrecT …
  -/
  simp only [codeSupp, codeSupp', trNormal, Finset.union_assoc, contSupp]
  rw [← Finset.union_assoc _ _ (contSupp k),
    Finset.union_eq_right.2 (codeSupp'_self _ _)]


@[simp]
theorem codeSupp_case (f g k) :
    codeSupp (Code.case f g) k =
      trStmts₁ (trNormal (Code.case f g) k) ∪ (codeSupp f k ∪ codeSupp g k) := by
  /-
    f g : Turing.ToPartrec.Code
    k : Turing.PartrecToTM2.Cont'
    ⊢ Eq (Turing.PartrecToTM2.codeSupp (f.case g) k) (Union.union (Turing.PartrecT …
  -/
  simp [codeSupp, codeSupp', contSupp, Finset.union_assoc, Finset.union_left_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem codeSupp_fix (f k) :
    codeSupp (Code.fix f) k = trStmts₁ (trNormal (Code.fix f) k) ∪ codeSupp f (Cont'.fix f k) := by
  simp [codeSupp, codeSupp', contSupp, Finset.union_assoc, Finset.union_left_comm,
    Finset.union_left_idem]


@[simp]
theorem contSupp_cons₁ (fs k) :
    contSupp (Cont'.cons₁ fs k) =
      trStmts₁
          (move₂ (fun _ => false) main aux <|
            move₂ (fun s => s = Γ'.consₗ) stack main <|
              move₂ (fun _ => false) aux stack <| trNormal fs (Cont'.cons₂ k)) ∪
        codeSupp fs (Cont'.cons₂ k) := by
  /-
    fs : Turing.ToPartrec.Code
    k : Turing.PartrecToTM2.Cont'
    ⊢ Eq (Turing.PartrecToTM2.contSupp (Turing.PartrecToTM2.Cont'.cons₁ fs k)) (Un …
  -/
  simp [codeSupp, codeSupp', contSupp, Finset.union_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem contSupp_cons₂ (k) :
    contSupp (Cont'.cons₂ k) = trStmts₁ (head stack <| Λ'.ret k) ∪ contSupp k :=
  rfl


@[simp]
theorem contSupp_comp (f k) : contSupp (Cont'.comp f k) = codeSupp f k :=
  rfl


theorem contSupp_fix (f k) : contSupp (Cont'.fix f k) = codeSupp f (Cont'.fix f k) := by
  simp +contextual [codeSupp, codeSupp', contSupp, Finset.union_assoc,
    Finset.subset_iff]


@[simp]
theorem contSupp_halt : contSupp Cont'.halt = ∅ :=
  rfl


/-- The statement `Λ'.Supports S q` means that `contSupp k ⊆ S` for any `ret k`
reachable from `q`.
(This is a technical condition used in the proof that the machine is supported.) -/
def Λ'.Supports (S : Finset Λ') : Λ' → Prop
  | Λ'.move _ _ _ q => Λ'.Supports S q
  | Λ'.push _ _ q => Λ'.Supports S q
  | Λ'.read q => ∀ s, Λ'.Supports S (q s)
  | Λ'.clear _ _ q => Λ'.Supports S q
  | Λ'.copy q => Λ'.Supports S q
  | Λ'.succ q => Λ'.Supports S q
  | Λ'.pred q₁ q₂ => Λ'.Supports S q₁ ∧ Λ'.Supports S q₂
  | Λ'.ret k => contSupp k ⊆ S


/-- A shorthand for the predicate that we are proving in the main theorems `trStmts₁_supports`,
`codeSupp'_supports`, `contSupp_supports`, `codeSupp_supports`. The set `S` is fixed throughout
the proof, and denotes the full set of states in the machine, while `K` is a subset that we are
currently proving a property about. The predicate asserts that every state in `K` is closed in `S`
under forward simulation, i.e. stepping forward through evaluation starting from any state in `K`
stays entirely within `S`. -/
def Supports (K S : Finset Λ') :=
  ∀ q ∈ K, TM2.SupportsStmt S (tr q)


theorem supports_insert {K S q} :
                                                                             /-
                                                                               K S : Finset Turing.PartrecToTM2.Λ'
                                                                               q : Turing.PartrecToTM2.Λ'
                                                                               ⊢ Iff (Turing.PartrecToTM2.Supports (Insert.insert q K) S) (And (Turing.TM2.Su …
                                                                             -/
    Supports (insert q K) S ↔ TM2.SupportsStmt S (tr q) ∧ Supports K S := by simp [Supports]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                                                    /-
                                                                                      S : Finset Turing.PartrecToTM2.Λ'
                                                                                      q : Turing.PartrecToTM2.Λ'
                                                                                      ⊢ Iff (Turing.PartrecToTM2.Supports (Singleton.singleton q) S) (Turing.TM2.Sup …
                                                                                    -/
theorem supports_singleton {S q} : Supports {q} S ↔ TM2.SupportsStmt S (tr q) := by simp [Supports]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem supports_union {K₁ K₂ S} : Supports (K₁ ∪ K₂) S ↔ Supports K₁ S ∧ Supports K₂ S := by
  /-
    K₁ K₂ S : Finset Turing.PartrecToTM2.Λ'
    ⊢ Iff (Turing.PartrecToTM2.Supports (Union.union K₁ K₂) S) (And (Turing.Partre …
  -/
  simp [Supports, or_imp, forall_and]
  /-
    🎉 no goals
  -/


theorem supports_biUnion {K : Option Γ' → Finset Λ'} {S} :
    Supports (Finset.univ.biUnion K) S ↔ ∀ a, Supports (K a) S := by
  /-
    K : Option Turing.PartrecToTM2.Γ' → Finset Turing.PartrecToTM2.Λ'
    S : Finset Turing.PartrecToTM2.Λ'
    ⊢ Iff (Turing.PartrecToTM2.Supports (Finset.univ.biUnion K) S) (∀ (a : Option  …
  -/
  simpa [Supports] using forall_swap
  /-
    🎉 no goals
  -/


theorem head_supports {S k q} (H : (q : Λ').Supports S) : (head k q).Supports S := fun _ => by
  /-
    S : Finset Turing.PartrecToTM2.Λ'
    k : Turing.PartrecToTM2.K'
    q : Turing.PartrecToTM2.Λ'
    H : Turing.PartrecToTM2.Λ'.Supports S q
    x✝ : Option Turing.PartrecToTM2.Γ'
    ⊢ Turing.PartrecToTM2.Λ'.Supports S ((fun s => ite (Eq s (Option.some Turing.P …
  -/
                            /-
                              🎉 no goals
                            -/
  dsimp only; split_ifs <;> exact H
                            /-
                              🎉 no goals
                            -/


theorem ret_supports {S k} (H₁ : contSupp k ⊆ S) : TM2.SupportsStmt S (tr (Λ'.ret k)) := by
  /-
    S : Finset Turing.PartrecToTM2.Λ'
    k : Turing.PartrecToTM2.Cont'
    H₁ : HasSubset.Subset (Turing.PartrecToTM2.contSupp k) S
    ⊢ Turing.TM2.SupportsStmt S (Turing.PartrecToTM2.tr (Turing.PartrecToTM2.Λ'.re …
  -/
  have W := fun {q} => trStmts₁_self q
  cases k with
  | halt => trivial
  | cons₁ => rw [contSupp_cons₁, Finset.union_subset_iff] at H₁; exact fun _ => H₁.1 W
  | cons₂ => rw [contSupp_cons₂, Finset.union_subset_iff] at H₁; exact fun _ => H₁.1 W
  | comp => rw [contSupp_comp] at H₁; exact fun _ => H₁ (codeSupp_self _ _ W)
  | fix =>
    rw [contSupp_fix] at H₁
    have L := @Finset.mem_union_left; have R := @Finset.mem_union_right
    intro s; dsimp only; cases natEnd s.iget
    · refine H₁ (R _ <| L _ <| R _ <| R _ <| L _ W)
    · exact H₁ (R _ <| L _ <| R _ <| R _ <| R _ <| Finset.mem_singleton_self _)


theorem trStmts₁_supports {S q} (H₁ : (q : Λ').Supports S) (HS₁ : trStmts₁ q ⊆ S) :
    Supports (trStmts₁ q) S := by
  /-
    S : Finset Turing.PartrecToTM2.Λ'
    q : Turing.PartrecToTM2.Λ'
    H₁ : Turing.PartrecToTM2.Λ'.Supports S q
    HS₁ : HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ q) S
    ⊢ Turing.PartrecToTM2.Supports (Turing.PartrecToTM2.trStmts₁ q) S
  -/
  have W := fun {q} => trStmts₁_self q
  induction q with
  | move _ _ _ q q_ih => _ | clear _ _ q q_ih => _ | copy q q_ih => _ | push _ _ q q_ih => _
  | read q q_ih => _ | succ q q_ih => _ | pred q₁ q₂ q₁_ih q₂_ih => _ | ret => _ <;>
    /-
      case move
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      p✝ : Turing.PartrecToTM2.Γ' → Bool
      k₁✝ k₂✝ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.move p✝ k₁✝ k₂✝ …
      HS₁ : HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ (Turing.PartrecToTM2.Λ'.m …
      ⊢ Turing.PartrecToTM2.Supports (Turing.PartrecToTM2.trStmts₁ (Turing.PartrecTo …
    -/
    simp [trStmts₁, -Finset.singleton_subset_iff] at HS₁ ⊢
  any_goals
    cases' Finset.insert_subset_iff.1 HS₁ with h₁ h₂
    first | have h₃ := h₂ W | try simp [Finset.subset_iff] at h₂
    /-
      case move.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      p✝ : Turing.PartrecToTM2.Γ' → Bool
      k₁✝ k₂✝ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.move p✝ k₁✝ k₂✝ …
      HS₁ : HasSubset.Subset (Insert.insert (Turing.PartrecToTM2.Λ'.move p✝ k₁✝ k₂✝  …
      h₁ : Membership.mem S (Turing.PartrecToTM2.Λ'.move p✝ k₁✝ k₂✝ q)
      h₂ : HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ q) S
      h₃ : Membership.mem S q
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert (Turing.PartrecToTM2.Λ'.move p✝  …
    -/
  · exact supports_insert.2 ⟨⟨fun _ => h₃, fun _ => h₁⟩, q_ih H₁ h₂⟩ -- move
    /-
      🎉 no goals
    -/
    /-
      case clear.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      p✝ : Turing.PartrecToTM2.Γ' → Bool
      k✝ : Turing.PartrecToTM2.K'
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.clear p✝ k✝ q)
      HS₁ : HasSubset.Subset (Insert.insert (Turing.PartrecToTM2.Λ'.clear p✝ k✝ q) ( …
      h₁ : Membership.mem S (Turing.PartrecToTM2.Λ'.clear p✝ k✝ q)
      h₂ : HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ q) S
      h₃ : Membership.mem S q
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert (Turing.PartrecToTM2.Λ'.clear p✝ …
    -/
  · exact supports_insert.2 ⟨⟨fun _ => h₃, fun _ => h₁⟩, q_ih H₁ h₂⟩ -- clear
    /-
      🎉 no goals
    -/
    /-
      case copy.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S q.copy
      HS₁ : HasSubset.Subset (Insert.insert q.copy (Turing.PartrecToTM2.trStmts₁ q)) S
      h₁ : Membership.mem S q.copy
      h₂ : HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ q) S
      h₃ : Membership.mem S q
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert q.copy (Turing.PartrecToTM2.trSt …
    -/
  · exact supports_insert.2 ⟨⟨fun _ => h₁, fun _ => h₃⟩, q_ih H₁ h₂⟩ -- copy
    /-
      🎉 no goals
    -/
    /-
      case push.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      k✝ : Turing.PartrecToTM2.K'
      s✝ : Option Turing.PartrecToTM2.Γ' → Option Turing.PartrecToTM2.Γ'
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.push k✝ s✝ q)
      HS₁ : HasSubset.Subset (Insert.insert (Turing.PartrecToTM2.Λ'.push k✝ s✝ q) (T …
      h₁ : Membership.mem S (Turing.PartrecToTM2.Λ'.push k✝ s✝ q)
      h₂ : HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ q) S
      h₃ : Membership.mem S q
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert (Turing.PartrecToTM2.Λ'.push k✝  …
    -/
  · exact supports_insert.2 ⟨⟨fun _ => h₃, fun _ => h₃⟩, q_ih H₁ h₂⟩ -- push
    /-
      🎉 no goals
    -/
    /-
      case read.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q : Option Turing.PartrecToTM2.Γ' → Turing.PartrecToTM2.Λ'
      q_ih : ∀ (a : Option Turing.PartrecToTM2.Γ'), Turing.PartrecToTM2.Λ'.Supports  …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.read q)
      HS₁ : HasSubset.Subset (Insert.insert (Turing.PartrecToTM2.Λ'.read q) (Finset. …
      h₁ : Membership.mem S (Turing.PartrecToTM2.Λ'.read q)
      h₂ : ∀ ⦃x : Turing.PartrecToTM2.Λ'⦄ (x_1 : Option Turing.PartrecToTM2.Γ'), Mem …
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert (Turing.PartrecToTM2.Λ'.read q)  …
    -/
  · refine supports_insert.2 ⟨fun _ => h₂ _ W, ?_⟩ -- read
    /-
      case read.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q : Option Turing.PartrecToTM2.Γ' → Turing.PartrecToTM2.Λ'
      q_ih : ∀ (a : Option Turing.PartrecToTM2.Γ'), Turing.PartrecToTM2.Λ'.Supports  …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.read q)
      HS₁ : HasSubset.Subset (Insert.insert (Turing.PartrecToTM2.Λ'.read q) (Finset. …
      h₁ : Membership.mem S (Turing.PartrecToTM2.Λ'.read q)
      h₂ : ∀ ⦃x : Turing.PartrecToTM2.Λ'⦄ (x_1 : Option Turing.PartrecToTM2.Γ'), Mem …
      ⊢ Turing.PartrecToTM2.Supports (Finset.univ.biUnion fun s => Turing.PartrecToT …
    -/
    exact supports_biUnion.2 fun _ => q_ih _ (H₁ _) fun _ h => h₂ _ h
    /-
      🎉 no goals
    -/
    /-
      case succ.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S q.succ
      HS₁ : HasSubset.Subset (Insert.insert q.succ (Insert.insert (Turing.PartrecToT …
      h₁ : Membership.mem S q.succ
      h₂ : And (Membership.mem S (Turing.PartrecToTM2.unrev q)) (∀ (a : Turing.Partr …
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert q.succ (Insert.insert (Turing.Pa …
    -/
  · refine supports_insert.2 ⟨⟨fun _ => h₁, fun _ => h₂.1, fun _ => h₂.1⟩, ?_⟩ -- succ
    /-
      case succ.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q : Turing.PartrecToTM2.Λ'
      q_ih : Turing.PartrecToTM2.Λ'.Supports S q → HasSubset.Subset (Turing.PartrecT …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S q.succ
      HS₁ : HasSubset.Subset (Insert.insert q.succ (Insert.insert (Turing.PartrecToT …
      h₁ : Membership.mem S q.succ
      h₂ : And (Membership.mem S (Turing.PartrecToTM2.unrev q)) (∀ (a : Turing.Partr …
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert (Turing.PartrecToTM2.unrev q) (T …
    -/
    exact supports_insert.2 ⟨⟨fun _ => h₂.2 _ W, fun _ => h₂.1⟩, q_ih H₁ h₂.2⟩
    /-
      🎉 no goals
    -/
  · refine -- pred
      supports_insert.2 ⟨⟨fun _ => h₁, fun _ => h₂.2 _ (Or.inl W),
                          fun _ => h₂.1, fun _ => h₂.1⟩, ?_⟩
    /-
      case pred.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      q₁_ih : Turing.PartrecToTM2.Λ'.Supports S q₁ → HasSubset.Subset (Turing.Partre …
      q₂_ih : Turing.PartrecToTM2.Λ'.Supports S q₂ → HasSubset.Subset (Turing.Partre …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (q₁.pred q₂)
      HS₁ : HasSubset.Subset (Insert.insert (q₁.pred q₂) (Insert.insert (Turing.Part …
      h₁ : Membership.mem S (q₁.pred q₂)
      h₂ : And (Membership.mem S (Turing.PartrecToTM2.unrev q₂)) (∀ (a : Turing.Part …
      ⊢ Turing.PartrecToTM2.Supports (Insert.insert (Turing.PartrecToTM2.unrev q₂) ( …
    -/
    refine supports_insert.2 ⟨⟨fun _ => h₂.2 _ (Or.inr W), fun _ => h₂.1⟩, ?_⟩
    /-
      case pred.intro
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      q₁ q₂ : Turing.PartrecToTM2.Λ'
      q₁_ih : Turing.PartrecToTM2.Λ'.Supports S q₁ → HasSubset.Subset (Turing.Partre …
      q₂_ih : Turing.PartrecToTM2.Λ'.Supports S q₂ → HasSubset.Subset (Turing.Partre …
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (q₁.pred q₂)
      HS₁ : HasSubset.Subset (Insert.insert (q₁.pred q₂) (Insert.insert (Turing.Part …
      h₁ : Membership.mem S (q₁.pred q₂)
      h₂ : And (Membership.mem S (Turing.PartrecToTM2.unrev q₂)) (∀ (a : Turing.Part …
      ⊢ Turing.PartrecToTM2.Supports (Union.union (Turing.PartrecToTM2.trStmts₁ q₁)  …
    -/
    refine supports_union.2 ⟨?_, ?_⟩
      /-
        case pred.intro.refine_1
        S : Finset Turing.PartrecToTM2.Λ'
        W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
        q₁ q₂ : Turing.PartrecToTM2.Λ'
        q₁_ih : Turing.PartrecToTM2.Λ'.Supports S q₁ → HasSubset.Subset (Turing.Partre …
        q₂_ih : Turing.PartrecToTM2.Λ'.Supports S q₂ → HasSubset.Subset (Turing.Partre …
        H₁ : Turing.PartrecToTM2.Λ'.Supports S (q₁.pred q₂)
        HS₁ : HasSubset.Subset (Insert.insert (q₁.pred q₂) (Insert.insert (Turing.Part …
        h₁ : Membership.mem S (q₁.pred q₂)
        h₂ : And (Membership.mem S (Turing.PartrecToTM2.unrev q₂)) (∀ (a : Turing.Part …
        ⊢ Turing.PartrecToTM2.Supports (Turing.PartrecToTM2.trStmts₁ q₁) S
      -/
    · exact q₁_ih H₁.1 fun _ h => h₂.2 _ (Or.inl h)
      /-
        🎉 no goals
      -/
      /-
        case pred.intro.refine_2
        S : Finset Turing.PartrecToTM2.Λ'
        W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
        q₁ q₂ : Turing.PartrecToTM2.Λ'
        q₁_ih : Turing.PartrecToTM2.Λ'.Supports S q₁ → HasSubset.Subset (Turing.Partre …
        q₂_ih : Turing.PartrecToTM2.Λ'.Supports S q₂ → HasSubset.Subset (Turing.Partre …
        H₁ : Turing.PartrecToTM2.Λ'.Supports S (q₁.pred q₂)
        HS₁ : HasSubset.Subset (Insert.insert (q₁.pred q₂) (Insert.insert (Turing.Part …
        h₁ : Membership.mem S (q₁.pred q₂)
        h₂ : And (Membership.mem S (Turing.PartrecToTM2.unrev q₂)) (∀ (a : Turing.Part …
        ⊢ Turing.PartrecToTM2.Supports (Turing.PartrecToTM2.trStmts₁ q₂) S
      -/
    · exact q₂_ih H₁.2 fun _ h => h₂.2 _ (Or.inr h)
      /-
        🎉 no goals
      -/
    /-
      case ret
      S : Finset Turing.PartrecToTM2.Λ'
      W : ∀ {q : Turing.PartrecToTM2.Λ'}, Membership.mem (Turing.PartrecToTM2.trStmt …
      k✝ : Turing.PartrecToTM2.Cont'
      H₁ : Turing.PartrecToTM2.Λ'.Supports S (Turing.PartrecToTM2.Λ'.ret k✝)
      HS₁ : HasSubset.Subset (Singleton.singleton (Turing.PartrecToTM2.Λ'.ret k✝)) S
      ⊢ Turing.PartrecToTM2.Supports (Singleton.singleton (Turing.PartrecToTM2.Λ'.re …
    -/
  · exact supports_singleton.2 (ret_supports H₁)  -- ret
    /-
      🎉 no goals
    -/


theorem trStmts₁_supports' {S q K} (H₁ : (q : Λ').Supports S) (H₂ : trStmts₁ q ∪ K ⊆ S)
    (H₃ : K ⊆ S → Supports K S) : Supports (trStmts₁ q ∪ K) S := by
  /-
    S : Finset Turing.PartrecToTM2.Λ'
    q : Turing.PartrecToTM2.Λ'
    K : Finset Turing.PartrecToTM2.Λ'
    H₁ : Turing.PartrecToTM2.Λ'.Supports S q
    H₂ : HasSubset.Subset (Union.union (Turing.PartrecToTM2.trStmts₁ q) K) S
    H₃ : HasSubset.Subset K S → Turing.PartrecToTM2.Supports K S
    ⊢ Turing.PartrecToTM2.Supports (Union.union (Turing.PartrecToTM2.trStmts₁ q) K …
  -/
  simp only [Finset.union_subset_iff] at H₂
  /-
    S : Finset Turing.PartrecToTM2.Λ'
    q : Turing.PartrecToTM2.Λ'
    K : Finset Turing.PartrecToTM2.Λ'
    H₁ : Turing.PartrecToTM2.Λ'.Supports S q
    H₃ : HasSubset.Subset K S → Turing.PartrecToTM2.Supports K S
    H₂ : And (HasSubset.Subset (Turing.PartrecToTM2.trStmts₁ q) S) (HasSubset.Subs …
    ⊢ Turing.PartrecToTM2.Supports (Union.union (Turing.PartrecToTM2.trStmts₁ q) K …
  -/
  exact supports_union.2 ⟨trStmts₁_supports H₁ H₂.1, H₃ H₂.2⟩
  /-
    🎉 no goals
  -/


theorem trNormal_supports {S c k} (Hk : codeSupp c k ⊆ S) : (trNormal c k).Supports S := by
  induction c generalizing k with simp [Λ'.Supports, head]
  | zero' => exact Finset.union_subset_right Hk
  | succ => intro; split_ifs <;> exact Finset.union_subset_right Hk
  | tail => exact Finset.union_subset_right Hk
  | cons f fs IHf _ =>
    apply IHf
    rw [codeSupp_cons] at Hk
    exact Finset.union_subset_right Hk
  | comp f g _ IHg => apply IHg; rw [codeSupp_comp] at Hk; exact Finset.union_subset_right Hk
  | case f g IHf IHg =>
    simp only [codeSupp_case, Finset.union_subset_iff] at Hk
    exact ⟨IHf Hk.2.1, IHg Hk.2.2⟩
  | fix f IHf => apply IHf; rw [codeSupp_fix] at Hk; exact Finset.union_subset_right Hk


theorem codeSupp'_supports {S c k} (H : codeSupp c k ⊆ S) : Supports (codeSupp' c k) S := by
  induction c generalizing k with
  | cons f fs IHf IHfs =>
    have H' := H; simp only [codeSupp_cons, Finset.union_subset_iff] at H'
    refine trStmts₁_supports' (trNormal_supports H) (Finset.union_subset_left H) fun h => ?_
    refine supports_union.2 ⟨IHf H'.2, ?_⟩
    refine trStmts₁_supports' (trNormal_supports ?_) (Finset.union_subset_right h) fun h => ?_
    · simp only [codeSupp, Finset.union_subset_iff, contSupp] at h H ⊢
      exact ⟨h.2.2.1, h.2.2.2, H.2⟩
    refine supports_union.2 ⟨IHfs ?_, ?_⟩
    · rw [codeSupp, contSupp_cons₁] at H'
      exact Finset.union_subset_right (Finset.union_subset_right H'.2)
    exact
      trStmts₁_supports (head_supports <| Finset.union_subset_right H)
        (Finset.union_subset_right h)
  | comp f g IHf IHg =>
    have H' := H; rw [codeSupp_comp] at H'; have H' := Finset.union_subset_right H'
    refine trStmts₁_supports' (trNormal_supports H) (Finset.union_subset_left H) fun h => ?_
    refine supports_union.2 ⟨IHg H', ?_⟩
    refine trStmts₁_supports' (trNormal_supports ?_) (Finset.union_subset_right h) fun _ => ?_
    · simp only [codeSupp', codeSupp, Finset.union_subset_iff, contSupp] at h H ⊢
      exact ⟨h.2.2, H.2⟩
    exact IHf (Finset.union_subset_right H')
  | case f g IHf IHg =>
    have H' := H; simp only [codeSupp_case, Finset.union_subset_iff] at H'
    refine trStmts₁_supports' (trNormal_supports H) (Finset.union_subset_left H) fun _ => ?_
    exact supports_union.2 ⟨IHf H'.2.1, IHg H'.2.2⟩
  | fix f IHf =>
    have H' := H; simp only [codeSupp_fix, Finset.union_subset_iff] at H'
    refine trStmts₁_supports' (trNormal_supports H) (Finset.union_subset_left H) fun h => ?_
    refine supports_union.2 ⟨IHf H'.2, ?_⟩
    refine trStmts₁_supports' (trNormal_supports ?_) (Finset.union_subset_right h) fun _ => ?_
    · simp only [codeSupp', codeSupp, Finset.union_subset_iff, contSupp, trStmts₁,
        Finset.insert_subset_iff] at h H ⊢
      exact ⟨h.1, ⟨H.1.1, h⟩, H.2⟩
    exact supports_singleton.2 (ret_supports <| Finset.union_subset_right H)
  | _ => exact trStmts₁_supports (trNormal_supports H) (Finset.Subset.trans (codeSupp_self _ _) H)


theorem contSupp_supports {S k} (H : contSupp k ⊆ S) : Supports (contSupp k) S := by
  induction k with
  | halt => simp [contSupp_halt, Supports]
  | cons₁ f k IH =>
    have H₁ := H; rw [contSupp_cons₁] at H₁; have H₂ := Finset.union_subset_right H₁
    refine trStmts₁_supports' (trNormal_supports H₂) H₁ fun h => ?_
    refine supports_union.2 ⟨codeSupp'_supports H₂, ?_⟩
    simp only [codeSupp, contSupp_cons₂, Finset.union_subset_iff] at H₂
    exact trStmts₁_supports' (head_supports H₂.2.2) (Finset.union_subset_right h) IH
  | cons₂ k IH =>
    have H' := H; rw [contSupp_cons₂] at H'
    exact trStmts₁_supports' (head_supports <| Finset.union_subset_right H') H' IH
  | comp f k IH =>
    have H' := H; rw [contSupp_comp] at H'; have H₂ := Finset.union_subset_right H'
    exact supports_union.2 ⟨codeSupp'_supports H', IH H₂⟩
  | fix f k IH =>
    rw [contSupp] at H
    exact supports_union.2 ⟨codeSupp'_supports H, IH (Finset.union_subset_right H)⟩


theorem codeSupp_supports {S c k} (H : codeSupp c k ⊆ S) : Supports (codeSupp c k) S :=
  supports_union.2 ⟨codeSupp'_supports H, contSupp_supports (Finset.union_subset_right H)⟩


/-- The set `codeSupp c k` is a finite set that witnesses the effective finiteness of the `tr`
Turing machine. Starting from the initial state `trNormal c k`, forward simulation uses only
states in `codeSupp c k`, so this is a finite state machine. Even though the underlying type of
state labels `Λ'` is infinite, for a given partial recursive function `c` and continuation `k`,
only finitely many states are accessed, corresponding roughly to subterms of `c`. -/
theorem tr_supports (c k) : @TM2.Supports _ _ _ _ ⟨trNormal c k⟩ tr (codeSupp c k) :=
  ⟨codeSupp_self _ _ (trStmts₁_self _), fun _ => codeSupp_supports (Finset.Subset.refl _) _⟩


