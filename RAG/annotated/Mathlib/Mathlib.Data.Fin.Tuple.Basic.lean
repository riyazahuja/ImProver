theorem tuple0_le {α : Fin 0 → Type*} [∀ i, Preorder (α i)] (f g : ∀ i, α i) : f ≤ g :=
  finZeroElim


/-- The tail of an `n+1` tuple, i.e., its last `n` entries. -/
def tail (q : ∀ i, α i) : ∀ i : Fin n, α i.succ := fun i ↦ q i.succ


theorem tail_def {n : ℕ} {α : Fin (n + 1) → Sort*} {q : ∀ i, α i} :
    (tail fun k : Fin (n + 1) ↦ q k) = fun k : Fin n ↦ q k.succ :=
  rfl


/-- Adding an element at the beginning of an `n`-tuple, to get an `n+1`-tuple. -/
def cons (x : α 0) (p : ∀ i : Fin n, α i.succ) : ∀ i, α i := fun j ↦ Fin.cases x p j


@[simp]
theorem tail_cons : tail (cons x p) = p := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    x : α 0
    p : (i : Fin n) → α i.succ
    ⊢ Eq (Fin.tail (Fin.cons x p)) p
  -/
  simp (config := { unfoldPartialApp := true }) [tail, cons]
  /-
    🎉 no goals
  -/


@[simp]
                                                /-
                                                  n : Nat
                                                  α : Fin (HAdd.hAdd n 1) → Sort u
                                                  x : α 0
                                                  p : (i : Fin n) → α i.succ
                                                  i : Fin n
                                                  ⊢ Eq (Fin.cons x p i.succ) (p i)
                                                -/
theorem cons_succ : cons x p i.succ = p i := by simp [cons]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                         /-
                                           n : Nat
                                           α : Fin (HAdd.hAdd n 1) → Sort u
                                           x : α 0
                                           p : (i : Fin n) → α i.succ
                                           ⊢ Eq (Fin.cons x p 0) x
                                         -/
theorem cons_zero : cons x p 0 = x := by simp [cons]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem cons_one {α : Fin (n + 2) → Sort*} (x : α 0) (p : ∀ i : Fin n.succ, α i.succ) :
    cons x p 1 = p 0 := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 2) → Sort u_1
    x : α 0
    p : (i : Fin n.succ) → α i.succ
    ⊢ Eq (Fin.cons x p 1) (p 0)
  -/
  rw [← cons_succ x p]; rfl
                        /-
                          🎉 no goals
                        -/


/-- Updating a tuple and adding an element at the beginning commute. -/
@[simp]
theorem cons_update : cons x (update p i y) = update (cons x p) i.succ y := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    x : α 0
    p : (i : Fin n) → α i.succ
    i : Fin n
    y : α i.succ
    ⊢ Eq (Fin.cons x (Function.update p i y)) (Function.update (Fin.cons x p) i.su …
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    x : α 0
    p : (i : Fin n) → α i.succ
    i : Fin n
    y : α i.succ
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.cons x (Function.update p i y) j) (Function.update (Fin.cons x p) i. …
  -/
  by_cases h : j = 0
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      i : Fin n
      y : α i.succ
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Fin.cons x (Function.update p i y) j) (Function.update (Fin.cons x p) i. …
    -/
  · rw [h]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      i : Fin n
      y : α i.succ
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Fin.cons x (Function.update p i y) 0) (Function.update (Fin.cons x p) i. …
    -/
    simp [Ne.symm (succ_ne_zero i)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      i : Fin n
      y : α i.succ
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      ⊢ Eq (Fin.cons x (Function.update p i y) j) (Function.update (Fin.cons x p) i. …
    -/
  · let j' := pred j h
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      i : Fin n
      y : α i.succ
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      ⊢ Eq (Fin.cons x (Function.update p i y) j) (Function.update (Fin.cons x p) i. …
    -/
    have : j'.succ = j := succ_pred j h
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      i : Fin n
      y : α i.succ
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Fin.cons x (Function.update p i y) j) (Function.update (Fin.cons x p) i. …
    -/
    rw [← this, cons_succ]
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      i : Fin n
      y : α i.succ
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Function.update p i y j') (Function.update (Fin.cons x p) i.succ y j'.su …
    -/
    by_cases h' : j' = i
      /-
        case pos
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u
        x : α 0
        p : (i : Fin n) → α i.succ
        i : Fin n
        y : α i.succ
        j : Fin (HAdd.hAdd n 1)
        h : Not (Eq j 0)
        j' : Fin n := j.pred h
        this : Eq j'.succ j
        h' : Eq j' i
        ⊢ Eq (Function.update p i y j') (Function.update (Fin.cons x p) i.succ y j'.su …
      -/
    · rw [h']
      /-
        case pos
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u
        x : α 0
        p : (i : Fin n) → α i.succ
        i : Fin n
        y : α i.succ
        j : Fin (HAdd.hAdd n 1)
        h : Not (Eq j 0)
        j' : Fin n := j.pred h
        this : Eq j'.succ j
        h' : Eq j' i
        ⊢ Eq (Function.update p i y i) (Function.update (Fin.cons x p) i.succ y i.succ)
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u
        x : α 0
        p : (i : Fin n) → α i.succ
        i : Fin n
        y : α i.succ
        j : Fin (HAdd.hAdd n 1)
        h : Not (Eq j 0)
        j' : Fin n := j.pred h
        this : Eq j'.succ j
        h' : Not (Eq j' i)
        ⊢ Eq (Function.update p i y j') (Function.update (Fin.cons x p) i.succ y j'.su …
      -/
    · have : j'.succ ≠ i.succ := by rwa [Ne, succ_inj]
      /-
        case neg
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u
        x : α 0
        p : (i : Fin n) → α i.succ
        i : Fin n
        y : α i.succ
        j : Fin (HAdd.hAdd n 1)
        h : Not (Eq j 0)
        j' : Fin n := j.pred h
        this✝ : Eq j'.succ j
        h' : Not (Eq j' i)
        this : Ne j'.succ i.succ
        ⊢ Eq (Function.update p i y j') (Function.update (Fin.cons x p) i.succ y j'.su …
      -/
      rw [update_of_ne h', update_of_ne this, cons_succ]
      /-
        🎉 no goals
      -/


/-- As a binary function, `Fin.cons` is injective. -/
theorem cons_injective2 : Function.Injective2 (@cons n α) := fun x₀ y₀ x y h ↦
                                    /-
                                      n : Nat
                                      α : Fin (HAdd.hAdd n 1) → Sort u
                                      x₀ y₀ : α 0
                                      x y : (i : Fin n) → α i.succ
                                      h : Eq (Fin.cons x₀ x) (Fin.cons y₀ y)
                                      i : Fin n
                                      ⊢ Eq (x i) (y i)
                                    -/
  ⟨congr_fun h 0, funext fun i ↦ by simpa using congr_fun h (Fin.succ i)⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem cons_eq_cons {x₀ y₀ : α 0} {x y : ∀ i : Fin n, α i.succ} :
    cons x₀ x = cons y₀ y ↔ x₀ = y₀ ∧ x = y :=
  cons_injective2.eq_iff


theorem cons_left_injective (x : ∀ i : Fin n, α i.succ) : Function.Injective fun x₀ ↦ cons x₀ x :=
  cons_injective2.left _


theorem cons_right_injective (x₀ : α 0) : Function.Injective (cons x₀) :=
  cons_injective2.right _


/-- Adding an element at the beginning of a tuple and then updating it amounts to adding it
directly. -/
theorem update_cons_zero : update (cons x p) 0 z = cons z p := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    x : α 0
    p : (i : Fin n) → α i.succ
    z : α 0
    ⊢ Eq (Function.update (Fin.cons x p) 0 z) (Fin.cons z p)
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    x : α 0
    p : (i : Fin n) → α i.succ
    z : α 0
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.update (Fin.cons x p) 0 z j) (Fin.cons z p j)
  -/
  by_cases h : j = 0
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      z : α 0
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Function.update (Fin.cons x p) 0 z j) (Fin.cons z p j)
    -/
  · rw [h]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      z : α 0
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Function.update (Fin.cons x p) 0 z 0) (Fin.cons z p 0)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      z : α 0
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      ⊢ Eq (Function.update (Fin.cons x p) 0 z j) (Fin.cons z p j)
    -/
  · simp only [h, update_of_ne, Ne, not_false_iff]
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      z : α 0
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      ⊢ Eq (Fin.cons x p j) (Fin.cons z p j)
    -/
    let j' := pred j h
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      z : α 0
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      ⊢ Eq (Fin.cons x p j) (Fin.cons z p j)
    -/
    have : j'.succ = j := succ_pred j h
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      x : α 0
      p : (i : Fin n) → α i.succ
      z : α 0
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Fin.cons x p j) (Fin.cons z p j)
    -/
    rw [← this, cons_succ, cons_succ]
    /-
      🎉 no goals
    -/


/-- Concatenating the first element of a tuple with its tail gives back the original tuple -/
@[simp, nolint simpNF] -- Porting note: linter claims LHS doesn't simplify
theorem cons_self_tail : cons (q 0) (tail q) = q := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    ⊢ Eq (Fin.cons (q 0) (Fin.tail q)) q
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.cons (q 0) (Fin.tail q) j) (q j)
  -/
  by_cases h : j = 0
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Fin.cons (q 0) (Fin.tail q) j) (q j)
    -/
  · rw [h]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Fin.cons (q 0) (Fin.tail q) 0) (q 0)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      ⊢ Eq (Fin.cons (q 0) (Fin.tail q) j) (q j)
    -/
  · let j' := pred j h
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      ⊢ Eq (Fin.cons (q 0) (Fin.tail q) j) (q j)
    -/
    have : j'.succ = j := succ_pred j h
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Fin.cons (q 0) (Fin.tail q) j) (q j)
    -/
    rw [← this]
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Fin.cons (q 0) (Fin.tail q) j'.succ) (q j'.succ)
    -/
    unfold tail
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Fin.cons (q 0) (fun i => q i.succ) j'.succ) (q j'.succ)
    -/
    rw [cons_succ]
    /-
      🎉 no goals
    -/


/-- Equivalence between tuples of length `n + 1` and pairs of an element and a tuple of length `n`
given by separating out the first element of the tuple.

This is `Fin.cons` as an `Equiv`. -/
@[simps]
def consEquiv (α : Fin (n + 1) → Type*) : α 0 × (∀ i, α (succ i)) ≃ ∀ i, α i where
  toFun f := cons f.1 f.2
  invFun f := (f 0, tail f)
                   /-
                     m n : Nat
                     α✝ : Fin (HAdd.hAdd n 1) → Sort u
                     x : α✝ 0
                     q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                     p : (i : Fin n) → α✝ i.succ
                     i : Fin n
                     y : α✝ i.succ
                     z : α✝ 0
                     α : Fin (HAdd.hAdd n 1) → Type u_1
                     f : Prod (α 0) ((i : Fin n) → α i.succ)
                     ⊢ Eq ((fun f => { fst := f 0, snd := Fin.tail f }) ((fun f => Fin.cons f.1 f.2 …
                   -/
  left_inv f := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      m n : Nat
                      α✝ : Fin (HAdd.hAdd n 1) → Sort u
                      x : α✝ 0
                      q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                      p : (i : Fin n) → α✝ i.succ
                      i : Fin n
                      y : α✝ i.succ
                      z : α✝ 0
                      α : Fin (HAdd.hAdd n 1) → Type u_1
                      f : (i : Fin (HAdd.hAdd n 1)) → α i
                      ⊢ Eq ((fun f => Fin.cons f.1 f.2) ((fun f => { fst := f 0, snd := Fin.tail f } …
                    -/
  right_inv f := by simp
                    /-
                      🎉 no goals
                    -/


-- Porting note: Mathport removes `_root_`?

/-- Recurse on an `n+1`-tuple by splitting it into a single element and an `n`-tuple. -/
@[elab_as_elim]
def consCases {P : (∀ i : Fin n.succ, α i) → Sort v} (h : ∀ x₀ x, P (Fin.cons x₀ x))
    (x : ∀ i : Fin n.succ, α i) : P x :=
                  /-
                    m n : Nat
                    α : Fin (HAdd.hAdd n 1) → Sort u
                    x✝ : α 0
                    q : (i : Fin (HAdd.hAdd n 1)) → α i
                    p : (i : Fin n) → α i.succ
                    i : Fin n
                    y : α i.succ
                    z : α 0
                    P : ((i : Fin n.succ) → α i) → Sort v
                    h : (x₀ : α 0) → (x : (i : Fin n) → α i.succ) → P (Fin.cons x₀ x)
                    x : (i : Fin n.succ) → α i
                    ⊢ Eq (P (Fin.cons (x 0) (Fin.tail x))) (P x)
                  -/
  _root_.cast (by rw [cons_self_tail]) <| h (x 0) (tail x)
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem consCases_cons {P : (∀ i : Fin n.succ, α i) → Sort v} (h : ∀ x₀ x, P (Fin.cons x₀ x))
    (x₀ : α 0) (x : ∀ i : Fin n, α i.succ) : @consCases _ _ _ h (cons x₀ x) = h x₀ x := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    P : ((i : Fin n.succ) → α i) → Sort v
    h : (x₀ : α 0) → (x : (i : Fin n) → α i.succ) → P (Fin.cons x₀ x)
    x₀ : α 0
    x : (i : Fin n) → α i.succ
    ⊢ Eq (Fin.consCases h (Fin.cons x₀ x)) (h x₀ x)
  -/
  rw [consCases, cast_eq]
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    P : ((i : Fin n.succ) → α i) → Sort v
    h : (x₀ : α 0) → (x : (i : Fin n) → α i.succ) → P (Fin.cons x₀ x)
    x₀ : α 0
    x : (i : Fin n) → α i.succ
    ⊢ Eq (h (Fin.cons x₀ x 0) (Fin.tail (Fin.cons x₀ x))) (h x₀ x)
  -/
  congr
  /-
    🎉 no goals
  -/


/-- Recurse on a tuple by splitting into `Fin.elim0` and `Fin.cons`. -/
@[elab_as_elim]
def consInduction {α : Sort*} {P : ∀ {n : ℕ}, (Fin n → α) → Sort v} (h0 : P Fin.elim0)
    (h : ∀ {n} (x₀) (x : Fin n → α), P x → P (Fin.cons x₀ x)) : ∀ {n : ℕ} (x : Fin n → α), P x
               /-
                 m n : Nat
                 α✝ : Fin (HAdd.hAdd n 1) → Sort u
                 x✝ : α✝ 0
                 q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                 p : (i : Fin n) → α✝ i.succ
                 i : Fin n
                 y : α✝ i.succ
                 z : α✝ 0
                 α : Sort u_1
                 P : {n : Nat} → (Fin n → α) → Sort v
                 h0 : P Fin.elim0
                 h : {n : Nat} → (x₀ : α) → (x : Fin n → α) → P x → P (Fin.cons x₀ x)
                 x : Fin 0 → α
                 ⊢ P x
               -/
  | 0, x => by convert h0
               /-
                 🎉 no goals
               -/
  | _ + 1, x => consCases (fun _ _ ↦ h _ _ <| consInduction h0 h _) x


theorem cons_injective_of_injective {α} {x₀ : α} {x : Fin n → α} (hx₀ : x₀ ∉ Set.range x)
    (hx : Function.Injective x) : Function.Injective (cons x₀ x : Fin n.succ → α) := by
  /-
    n : Nat
    α : Type u_1
    x₀ : α
    x : Fin n → α
    hx₀ : Not (Membership.mem (Set.range x) x₀)
    hx : Function.Injective x
    ⊢ Function.Injective (Fin.cons x₀ x)
  -/
  refine Fin.cases ?_ ?_
    /-
      case refine_1
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      hx₀ : Not (Membership.mem (Set.range x) x₀)
      hx : Function.Injective x
      ⊢ ∀ ⦃a₂ : Fin (HAdd.hAdd n 1)⦄, Eq (Fin.cons x₀ x 0) (Fin.cons x₀ x a₂) → Eq 0 …
    -/
  · refine Fin.cases ?_ ?_
      /-
        case refine_1.refine_1
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        ⊢ Eq (Fin.cons x₀ x 0) (Fin.cons x₀ x 0) → Eq 0 0
      -/
    · intro
      /-
        case refine_1.refine_1
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        a✝ : Eq (Fin.cons x₀ x 0) (Fin.cons x₀ x 0)
        ⊢ Eq 0 0
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        ⊢ ∀ (i : Fin n), Eq (Fin.cons x₀ x 0) (Fin.cons x₀ x i.succ) → Eq 0 i.succ
      -/
    · intro j h
      /-
        case refine_1.refine_2
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        j : Fin n
        h : Eq (Fin.cons x₀ x 0) (Fin.cons x₀ x j.succ)
        ⊢ Eq 0 j.succ
      -/
      rw [cons_zero, cons_succ] at h
      /-
        case refine_1.refine_2
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        j : Fin n
        h : Eq x₀ (x j)
        ⊢ Eq 0 j.succ
      -/
      exact hx₀.elim ⟨_, h.symm⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      hx₀ : Not (Membership.mem (Set.range x) x₀)
      hx : Function.Injective x
      ⊢ ∀ (i : Fin n) ⦃a₂ : Fin (HAdd.hAdd n 1)⦄, Eq (Fin.cons x₀ x i.succ) (Fin.con …
    -/
  · intro i
    /-
      case refine_2
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      hx₀ : Not (Membership.mem (Set.range x) x₀)
      hx : Function.Injective x
      i : Fin n
      ⊢ ∀ ⦃a₂ : Fin (HAdd.hAdd n 1)⦄, Eq (Fin.cons x₀ x i.succ) (Fin.cons x₀ x a₂) → …
    -/
    refine Fin.cases ?_ ?_
      /-
        case refine_2.refine_1
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        i : Fin n
        ⊢ Eq (Fin.cons x₀ x i.succ) (Fin.cons x₀ x 0) → Eq i.succ 0
      -/
    · intro h
      /-
        case refine_2.refine_1
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        i : Fin n
        h : Eq (Fin.cons x₀ x i.succ) (Fin.cons x₀ x 0)
        ⊢ Eq i.succ 0
      -/
      rw [cons_zero, cons_succ] at h
      /-
        case refine_2.refine_1
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        i : Fin n
        h : Eq (x i) x₀
        ⊢ Eq i.succ 0
      -/
      exact hx₀.elim ⟨_, h⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        i : Fin n
        ⊢ ∀ (i_1 : Fin n), Eq (Fin.cons x₀ x i.succ) (Fin.cons x₀ x i_1.succ) → Eq i.s …
      -/
    · intro j h
      /-
        case refine_2.refine_2
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        i j : Fin n
        h : Eq (Fin.cons x₀ x i.succ) (Fin.cons x₀ x j.succ)
        ⊢ Eq i.succ j.succ
      -/
      rw [cons_succ, cons_succ] at h
      /-
        case refine_2.refine_2
        n : Nat
        α : Type u_1
        x₀ : α
        x : Fin n → α
        hx₀ : Not (Membership.mem (Set.range x) x₀)
        hx : Function.Injective x
        i j : Fin n
        h : Eq (x i) (x j)
        ⊢ Eq i.succ j.succ
      -/
      exact congr_arg _ (hx h)
      /-
        🎉 no goals
      -/


theorem cons_injective_iff {α} {x₀ : α} {x : Fin n → α} :
    Function.Injective (cons x₀ x : Fin n.succ → α) ↔ x₀ ∉ Set.range x ∧ Function.Injective x := by
  /-
    n : Nat
    α : Type u_1
    x₀ : α
    x : Fin n → α
    ⊢ Iff (Function.Injective (Fin.cons x₀ x)) (And (Not (Membership.mem (Set.rang …
  -/
  refine ⟨fun h ↦ ⟨?_, ?_⟩, fun h ↦ cons_injective_of_injective h.1 h.2⟩
    /-
      case refine_1
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      h : Function.Injective (Fin.cons x₀ x)
      ⊢ Not (Membership.mem (Set.range x) x₀)
    -/
  · rintro ⟨i, hi⟩
    /-
      case refine_1.intro
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      h : Function.Injective (Fin.cons x₀ x)
      i : Fin n
      hi : Eq (x i) x₀
      ⊢ False
    -/
    replace h := @h i.succ 0
    /-
      case refine_1.intro
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      i : Fin n
      hi : Eq (x i) x₀
      h : Eq (Fin.cons x₀ x i.succ) (Fin.cons x₀ x 0) → Eq i.succ 0
      ⊢ False
    -/
    simp [hi, succ_ne_zero] at h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      α : Type u_1
      x₀ : α
      x : Fin n → α
      h : Function.Injective (Fin.cons x₀ x)
      ⊢ Function.Injective x
    -/
  · simpa [Function.comp] using h.comp (Fin.succ_injective _)
    /-
      🎉 no goals
    -/


@[simp]
theorem forall_fin_zero_pi {α : Fin 0 → Sort*} {P : (∀ i, α i) → Prop} :
    (∀ x, P x) ↔ P finZeroElim :=
  ⟨fun h ↦ h _, fun h x ↦ Subsingleton.elim finZeroElim x ▸ h⟩


@[simp]
theorem exists_fin_zero_pi {α : Fin 0 → Sort*} {P : (∀ i, α i) → Prop} :
    (∃ x, P x) ↔ P finZeroElim :=
  ⟨fun ⟨x, h⟩ ↦ Subsingleton.elim x finZeroElim ▸ h, fun h ↦ ⟨_, h⟩⟩


theorem forall_fin_succ_pi {P : (∀ i, α i) → Prop} : (∀ x, P x) ↔ ∀ a v, P (Fin.cons a v) :=
  ⟨fun h a v ↦ h (Fin.cons a v), consCases⟩


theorem exists_fin_succ_pi {P : (∀ i, α i) → Prop} : (∃ x, P x) ↔ ∃ a v, P (Fin.cons a v) :=
  ⟨fun ⟨x, h⟩ ↦ ⟨x 0, tail x, (cons_self_tail x).symm ▸ h⟩, fun ⟨_, _, h⟩ ↦ ⟨_, h⟩⟩


/-- Updating the first element of a tuple does not change the tail. -/
@[simp]
theorem tail_update_zero : tail (update q 0 z) = tail q := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    z : α 0
    ⊢ Eq (Fin.tail (Function.update q 0 z)) (Fin.tail q)
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    z : α 0
    j : Fin n
    ⊢ Eq (Fin.tail (Function.update q 0 z) j) (Fin.tail q j)
  -/
  simp [tail, Fin.succ_ne_zero]
  /-
    🎉 no goals
  -/


/-- Updating a nonzero element and taking the tail commute. -/
@[simp]
theorem tail_update_succ : tail (update q i.succ y) = update (tail q) i y := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin n
    y : α i.succ
    ⊢ Eq (Fin.tail (Function.update q i.succ y)) (Function.update (Fin.tail q) i y)
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin n
    y : α i.succ
    j : Fin n
    ⊢ Eq (Fin.tail (Function.update q i.succ y) j) (Function.update (Fin.tail q) i …
  -/
  by_cases h : j = i
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      i : Fin n
      y : α i.succ
      j : Fin n
      h : Eq j i
      ⊢ Eq (Fin.tail (Function.update q i.succ y) j) (Function.update (Fin.tail q) i …
    -/
  · rw [h]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      i : Fin n
      y : α i.succ
      j : Fin n
      h : Eq j i
      ⊢ Eq (Fin.tail (Function.update q i.succ y) i) (Function.update (Fin.tail q) i …
    -/
    simp [tail]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      i : Fin n
      y : α i.succ
      j : Fin n
      h : Not (Eq j i)
      ⊢ Eq (Fin.tail (Function.update q i.succ y) j) (Function.update (Fin.tail q) i …
    -/
  · simp [tail, (Fin.succ_injective n).ne h, h]
    /-
      🎉 no goals
    -/


theorem comp_cons {α : Sort*} {β : Sort*} (g : α → β) (y : α) (q : Fin n → α) :
    g ∘ cons y q = cons (g y) (g ∘ q) := by
  /-
    n : Nat
    α : Sort u_1
    β : Sort u_2
    g : α → β
    y : α
    q : Fin n → α
    ⊢ Eq (Function.comp g (Fin.cons y q)) (Fin.cons (g y) (Function.comp g q))
  -/
  ext j
  /-
    case h
    n : Nat
    α : Sort u_1
    β : Sort u_2
    g : α → β
    y : α
    q : Fin n → α
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.comp g (Fin.cons y q) j) (Fin.cons (g y) (Function.comp g q) j)
  -/
  by_cases h : j = 0
    /-
      case pos
      n : Nat
      α : Sort u_1
      β : Sort u_2
      g : α → β
      y : α
      q : Fin n → α
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Function.comp g (Fin.cons y q) j) (Fin.cons (g y) (Function.comp g q) j)
    -/
  · rw [h]
    /-
      case pos
      n : Nat
      α : Sort u_1
      β : Sort u_2
      g : α → β
      y : α
      q : Fin n → α
      j : Fin (HAdd.hAdd n 1)
      h : Eq j 0
      ⊢ Eq (Function.comp g (Fin.cons y q) 0) (Fin.cons (g y) (Function.comp g q) 0)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Sort u_1
      β : Sort u_2
      g : α → β
      y : α
      q : Fin n → α
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      ⊢ Eq (Function.comp g (Fin.cons y q) j) (Fin.cons (g y) (Function.comp g q) j)
    -/
  · let j' := pred j h
    /-
      case neg
      n : Nat
      α : Sort u_1
      β : Sort u_2
      g : α → β
      y : α
      q : Fin n → α
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      ⊢ Eq (Function.comp g (Fin.cons y q) j) (Fin.cons (g y) (Function.comp g q) j)
    -/
    have : j'.succ = j := succ_pred j h
    /-
      case neg
      n : Nat
      α : Sort u_1
      β : Sort u_2
      g : α → β
      y : α
      q : Fin n → α
      j : Fin (HAdd.hAdd n 1)
      h : Not (Eq j 0)
      j' : Fin n := j.pred h
      this : Eq j'.succ j
      ⊢ Eq (Function.comp g (Fin.cons y q) j) (Fin.cons (g y) (Function.comp g q) j)
    -/
    rw [← this, cons_succ, comp_apply, comp_apply, cons_succ]
    /-
      🎉 no goals
    -/


theorem comp_tail {α : Sort*} {β : Sort*} (g : α → β) (q : Fin n.succ → α) :
    g ∘ tail q = tail (g ∘ q) := by
  /-
    n : Nat
    α : Sort u_1
    β : Sort u_2
    g : α → β
    q : Fin n.succ → α
    ⊢ Eq (Function.comp g (Fin.tail q)) (Fin.tail (Function.comp g q))
  -/
  ext j
  /-
    case h
    n : Nat
    α : Sort u_1
    β : Sort u_2
    g : α → β
    q : Fin n.succ → α
    j : Fin n
    ⊢ Eq (Function.comp g (Fin.tail q) j) (Fin.tail (Function.comp g q) j)
  -/
  simp [tail]
  /-
    🎉 no goals
  -/


theorem le_cons [∀ i, Preorder (α i)] {x : α 0} {q : ∀ i, α i} {p : ∀ i : Fin n, α i.succ} :
    q ≤ cons x p ↔ q 0 ≤ x ∧ tail q ≤ p :=
                                                                         /-
                                                                           n : Nat
                                                                           α : Fin (HAdd.hAdd n 1) → Type u_1
                                                                           inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
                                                                           x : α 0
                                                                           q : (i : Fin (HAdd.hAdd n 1)) → α i
                                                                           p : (i : Fin n) → α i.succ
                                                                           j : Fin n
                                                                           ⊢ Iff (LE.le (q j.succ) (Fin.cons x p j.succ)) (LE.le (Fin.tail q j) (p j))
                                                                         -/
  forall_fin_succ.trans <| and_congr Iff.rfl <| forall_congr' fun j ↦ by simp [tail]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem cons_le [∀ i, Preorder (α i)] {x : α 0} {q : ∀ i, α i} {p : ∀ i : Fin n, α i.succ} :
    cons x p ≤ q ↔ x ≤ q 0 ∧ p ≤ tail q :=
  @le_cons _ (fun i ↦ (α i)ᵒᵈ) _ x q p


theorem cons_le_cons [∀ i, Preorder (α i)] {x₀ y₀ : α 0} {x y : ∀ i : Fin n, α i.succ} :
    cons x₀ x ≤ cons y₀ y ↔ x₀ ≤ y₀ ∧ x ≤ y :=
                                                  /-
                                                    n : Nat
                                                    α : Fin (HAdd.hAdd n 1) → Type u_1
                                                    inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
                                                    x₀ y₀ : α 0
                                                    x y : (i : Fin n) → α i.succ
                                                    ⊢ Iff (∀ (i : Fin n), LE.le (Fin.cons x₀ x i.succ) (Fin.cons y₀ y i.succ)) (LE …
                                                  -/
  forall_fin_succ.trans <| and_congr_right' <| by simp only [cons_succ, Pi.le_def]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem range_fin_succ {α} (f : Fin (n + 1) → α) :
    Set.range f = insert (f 0) (Set.range (Fin.tail f)) :=
  Set.ext fun _ ↦ exists_fin_succ.trans <| eq_comm.or Iff.rfl


@[simp]
theorem range_cons {α} {n : ℕ} (x : α) (b : Fin n → α) :
    Set.range (Fin.cons x b : Fin n.succ → α) = insert x (Set.range b) := by
  /-
    α : Type u_1
    n : Nat
    x : α
    b : Fin n → α
    ⊢ Eq (Set.range (Fin.cons x b)) (Insert.insert x (Set.range b))
  -/
  rw [range_fin_succ, cons_zero, tail_cons]
  /-
    🎉 no goals
  -/


/-- Append a tuple of length `m` to a tuple of length `n` to get a tuple of length `m + n`.
This is a non-dependent version of `Fin.add_cases`. -/
def append (a : Fin m → α) (b : Fin n → α) : Fin (m + n) → α :=
  @Fin.addCases _ _ (fun _ => α) a b


@[simp]
theorem append_left (u : Fin m → α) (v : Fin n → α) (i : Fin m) :
    append u v (Fin.castAdd n i) = u i :=
  addCases_left _


@[simp]
theorem append_right (u : Fin m → α) (v : Fin n → α) (i : Fin n) :
    append u v (natAdd m i) = v i :=
  addCases_right _


theorem append_right_nil (u : Fin m → α) (v : Fin n → α) (hv : n = 0) :
                                  /-
                                    m n : Nat
                                    α✝ : Fin (HAdd.hAdd n 1) → Sort u
                                    x : α✝ 0
                                    q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                                    p : (i : Fin n) → α✝ i.succ
                                    i : Fin n
                                    y : α✝ i.succ
                                    z : α✝ 0
                                    α : Sort u_1
                                    u : Fin m → α
                                    v : Fin n → α
                                    hv : Eq n 0
                                    ⊢ Eq (HAdd.hAdd m n) m
                                  -/
    append u v = u ∘ Fin.cast (by rw [hv, Nat.add_zero]) := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    m n : Nat
    α : Sort u_1
    u : Fin m → α
    v : Fin n → α
    hv : Eq n 0
    ⊢ Eq (Fin.append u v) (Function.comp u (Fin.cast ⋯))
  -/
  refine funext (Fin.addCases (fun l => ?_) fun r => ?_)
    /-
      case refine_1
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hv : Eq n 0
      l : Fin m
      ⊢ Eq (Fin.append u v (Fin.castAdd n l)) (Function.comp u (Fin.cast ⋯) (Fin.cas …
    -/
  · rw [append_left, Function.comp_apply]
    /-
      case refine_1
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hv : Eq n 0
      l : Fin m
      ⊢ Eq (u l) (u (Fin.cast ⋯ (Fin.castAdd n l)))
    -/
    refine congr_arg u (Fin.ext ?_)
    /-
      case refine_1
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hv : Eq n 0
      l : Fin m
      ⊢ Eq ↑l ↑(Fin.cast ⋯ (Fin.castAdd n l))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hv : Eq n 0
      r : Fin n
      ⊢ Eq (Fin.append u v (Fin.natAdd m r)) (Function.comp u (Fin.cast ⋯) (Fin.natA …
    -/
  · exact (Fin.cast hv r).elim0
    /-
      🎉 no goals
    -/


@[simp]
theorem append_elim0 (u : Fin m → α) :
    append u Fin.elim0 = u ∘ Fin.cast (Nat.add_zero _) :=
  append_right_nil _ _ rfl


theorem append_left_nil (u : Fin m → α) (v : Fin n → α) (hu : m = 0) :
                                  /-
                                    m n : Nat
                                    α✝ : Fin (HAdd.hAdd n 1) → Sort u
                                    x : α✝ 0
                                    q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                                    p : (i : Fin n) → α✝ i.succ
                                    i : Fin n
                                    y : α✝ i.succ
                                    z : α✝ 0
                                    α : Sort u_1
                                    u : Fin m → α
                                    v : Fin n → α
                                    hu : Eq m 0
                                    ⊢ Eq (HAdd.hAdd m n) n
                                  -/
    append u v = v ∘ Fin.cast (by rw [hu, Nat.zero_add]) := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    m n : Nat
    α : Sort u_1
    u : Fin m → α
    v : Fin n → α
    hu : Eq m 0
    ⊢ Eq (Fin.append u v) (Function.comp v (Fin.cast ⋯))
  -/
  refine funext (Fin.addCases (fun l => ?_) fun r => ?_)
    /-
      case refine_1
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hu : Eq m 0
      l : Fin m
      ⊢ Eq (Fin.append u v (Fin.castAdd n l)) (Function.comp v (Fin.cast ⋯) (Fin.cas …
    -/
  · exact (Fin.cast hu l).elim0
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hu : Eq m 0
      r : Fin n
      ⊢ Eq (Fin.append u v (Fin.natAdd m r)) (Function.comp v (Fin.cast ⋯) (Fin.natA …
    -/
  · rw [append_right, Function.comp_apply]
    /-
      case refine_2
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hu : Eq m 0
      r : Fin n
      ⊢ Eq (v r) (v (Fin.cast ⋯ (Fin.natAdd m r)))
    -/
    refine congr_arg v (Fin.ext ?_)
    /-
      case refine_2
      m n : Nat
      α : Sort u_1
      u : Fin m → α
      v : Fin n → α
      hu : Eq m 0
      r : Fin n
      ⊢ Eq ↑r ↑(Fin.cast ⋯ (Fin.natAdd m r))
    -/
    simp [hu]
    /-
      🎉 no goals
    -/


@[simp]
theorem elim0_append (v : Fin n → α) :
    append Fin.elim0 v = v ∘ Fin.cast (Nat.zero_add _) :=
  append_left_nil _ _ rfl


theorem append_assoc {p : ℕ} (a : Fin m → α) (b : Fin n → α) (c : Fin p → α) :
    append (append a b) c = append a (append b c) ∘ Fin.cast (Nat.add_assoc ..) := by
  /-
    m n : Nat
    α : Sort u_1
    p : Nat
    a : Fin m → α
    b : Fin n → α
    c : Fin p → α
    ⊢ Eq (Fin.append (Fin.append a b) c) (Function.comp (Fin.append a (Fin.append  …
  -/
  ext i
  /-
    case h
    m n : Nat
    α : Sort u_1
    p : Nat
    a : Fin m → α
    b : Fin n → α
    c : Fin p → α
    i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
    ⊢ Eq (Fin.append (Fin.append a b) c i) (Function.comp (Fin.append a (Fin.appen …
  -/
  rw [Function.comp_apply]
  /-
    case h
    m n : Nat
    α : Sort u_1
    p : Nat
    a : Fin m → α
    b : Fin n → α
    c : Fin p → α
    i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
    ⊢ Eq (Fin.append (Fin.append a b) c i) (Fin.append a (Fin.append b c) (Fin.cas …
  -/
  refine Fin.addCases (fun l => ?_) (fun r => ?_) i
    /-
      case h.refine_1
      m n : Nat
      α : Sort u_1
      p : Nat
      a : Fin m → α
      b : Fin n → α
      c : Fin p → α
      i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
      l : Fin (HAdd.hAdd m n)
      ⊢ Eq (Fin.append (Fin.append a b) c (Fin.castAdd p l)) (Fin.append a (Fin.appe …
    -/
  · rw [append_left]
    /-
      case h.refine_1
      m n : Nat
      α : Sort u_1
      p : Nat
      a : Fin m → α
      b : Fin n → α
      c : Fin p → α
      i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
      l : Fin (HAdd.hAdd m n)
      ⊢ Eq (Fin.append a b l) (Fin.append a (Fin.append b c) (Fin.cast ⋯ (Fin.castAd …
    -/
    refine Fin.addCases (fun ll => ?_) (fun lr => ?_) l
      /-
        case h.refine_1.refine_1
        m n : Nat
        α : Sort u_1
        p : Nat
        a : Fin m → α
        b : Fin n → α
        c : Fin p → α
        i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
        l : Fin (HAdd.hAdd m n)
        ll : Fin m
        ⊢ Eq (Fin.append a b (Fin.castAdd n ll)) (Fin.append a (Fin.append b c) (Fin.c …
      -/
    · rw [append_left]
      /-
        case h.refine_1.refine_1
        m n : Nat
        α : Sort u_1
        p : Nat
        a : Fin m → α
        b : Fin n → α
        c : Fin p → α
        i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
        l : Fin (HAdd.hAdd m n)
        ll : Fin m
        ⊢ Eq (a ll) (Fin.append a (Fin.append b c) (Fin.cast ⋯ (Fin.castAdd p (Fin.cas …
      -/
      simp [castAdd_castAdd]
      /-
        🎉 no goals
      -/
      /-
        case h.refine_1.refine_2
        m n : Nat
        α : Sort u_1
        p : Nat
        a : Fin m → α
        b : Fin n → α
        c : Fin p → α
        i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
        l : Fin (HAdd.hAdd m n)
        lr : Fin n
        ⊢ Eq (Fin.append a b (Fin.natAdd m lr)) (Fin.append a (Fin.append b c) (Fin.ca …
      -/
    · rw [append_right]
      /-
        case h.refine_1.refine_2
        m n : Nat
        α : Sort u_1
        p : Nat
        a : Fin m → α
        b : Fin n → α
        c : Fin p → α
        i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
        l : Fin (HAdd.hAdd m n)
        lr : Fin n
        ⊢ Eq (b lr) (Fin.append a (Fin.append b c) (Fin.cast ⋯ (Fin.castAdd p (Fin.nat …
      -/
      simp [castAdd_natAdd]
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      m n : Nat
      α : Sort u_1
      p : Nat
      a : Fin m → α
      b : Fin n → α
      c : Fin p → α
      i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
      r : Fin p
      ⊢ Eq (Fin.append (Fin.append a b) c (Fin.natAdd (HAdd.hAdd m n) r)) (Fin.appen …
    -/
  · rw [append_right]
    /-
      case h.refine_2
      m n : Nat
      α : Sort u_1
      p : Nat
      a : Fin m → α
      b : Fin n → α
      c : Fin p → α
      i : Fin (HAdd.hAdd (HAdd.hAdd m n) p)
      r : Fin p
      ⊢ Eq (c r) (Fin.append a (Fin.append b c) (Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd m …
    -/
    simp [← natAdd_natAdd]
    /-
      🎉 no goals
    -/


/-- Appending a one-tuple to the left is the same as `Fin.cons`. -/
theorem append_left_eq_cons {n : ℕ} (x₀ : Fin 1 → α) (x : Fin n → α) :
    Fin.append x₀ x = Fin.cons (x₀ 0) x ∘ Fin.cast (Nat.add_comm ..) := by
  /-
    α : Sort u_1
    n : Nat
    x₀ : Fin 1 → α
    x : Fin n → α
    ⊢ Eq (Fin.append x₀ x) (Function.comp (Fin.cons (x₀ 0) x) (Fin.cast ⋯))
  -/
  ext i
  /-
    case h
    α : Sort u_1
    n : Nat
    x₀ : Fin 1 → α
    x : Fin n → α
    i : Fin (HAdd.hAdd 1 n)
    ⊢ Eq (Fin.append x₀ x i) (Function.comp (Fin.cons (x₀ 0) x) (Fin.cast ⋯) i)
  -/
  refine Fin.addCases ?_ ?_ i <;> clear i
    /-
      case h.refine_1
      α : Sort u_1
      n : Nat
      x₀ : Fin 1 → α
      x : Fin n → α
      ⊢ ∀ (i : Fin 1), Eq (Fin.append x₀ x (Fin.castAdd n i)) (Function.comp (Fin.co …
    -/
  · intro i
    /-
      case h.refine_1
      α : Sort u_1
      n : Nat
      x₀ : Fin 1 → α
      x : Fin n → α
      i : Fin 1
      ⊢ Eq (Fin.append x₀ x (Fin.castAdd n i)) (Function.comp (Fin.cons (x₀ 0) x) (F …
    -/
    rw [Subsingleton.elim i 0, Fin.append_left, Function.comp_apply, eq_comm]
    /-
      case h.refine_1
      α : Sort u_1
      n : Nat
      x₀ : Fin 1 → α
      x : Fin n → α
      i : Fin 1
      ⊢ Eq (Fin.cons (x₀ 0) x (Fin.cast ⋯ (Fin.castAdd n 0))) (x₀ 0)
    -/
    exact Fin.cons_zero _ _
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Sort u_1
      n : Nat
      x₀ : Fin 1 → α
      x : Fin n → α
      ⊢ ∀ (i : Fin n), Eq (Fin.append x₀ x (Fin.natAdd 1 i)) (Function.comp (Fin.con …
    -/
  · intro i
    /-
      case h.refine_2
      α : Sort u_1
      n : Nat
      x₀ : Fin 1 → α
      x : Fin n → α
      i : Fin n
      ⊢ Eq (Fin.append x₀ x (Fin.natAdd 1 i)) (Function.comp (Fin.cons (x₀ 0) x) (Fi …
    -/
    rw [Fin.append_right, Function.comp_apply, Fin.cast_natAdd, eq_comm, Fin.addNat_one]
    /-
      case h.refine_2
      α : Sort u_1
      n : Nat
      x₀ : Fin 1 → α
      x : Fin n → α
      i : Fin n
      ⊢ Eq (Fin.cons (x₀ 0) x i.succ) (x i)
    -/
    exact Fin.cons_succ _ _ _
    /-
      🎉 no goals
    -/


/-- `Fin.cons` is the same as appending a one-tuple to the left. -/
theorem cons_eq_append (x : α) (xs : Fin n → α) :
    cons x xs = append (cons x Fin.elim0) xs ∘ Fin.cast (Nat.add_comm ..) := by
  /-
    n : Nat
    α : Sort u_1
    x : α
    xs : Fin n → α
    ⊢ Eq (Fin.cons x xs) (Function.comp (Fin.append (Fin.cons x Fin.elim0) xs) (Fi …
  -/
  funext i; simp [append_left_eq_cons]
            /-
              🎉 no goals
            -/


@[simp] lemma append_cast_left {n m} (xs : Fin n → α) (ys : Fin m → α) (n' : ℕ)
    (h : n' = n) :
                                                                         /-
                                                                           m✝ n✝ : Nat
                                                                           α✝ : Fin (HAdd.hAdd n✝ 1) → Sort u
                                                                           x : α✝ 0
                                                                           q : (i : Fin (HAdd.hAdd n✝ 1)) → α✝ i
                                                                           p : (i : Fin n✝) → α✝ i.succ
                                                                           i : Fin n✝
                                                                           y : α✝ i.succ
                                                                           z : α✝ 0
                                                                           α : Sort u_1
                                                                           n m : Nat
                                                                           xs : Fin n → α
                                                                           ys : Fin m → α
                                                                           n' : Nat
                                                                           h : Eq n' n
                                                                           ⊢ Eq (HAdd.hAdd n' m) (HAdd.hAdd n m)
                                                                         -/
    Fin.append (xs ∘ Fin.cast h) ys = Fin.append xs ys ∘ (Fin.cast <| by rw [h]) := by
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  /-
    α : Sort u_1
    n m : Nat
    xs : Fin n → α
    ys : Fin m → α
    n' : Nat
    h : Eq n' n
    ⊢ Eq (Fin.append (Function.comp xs (Fin.cast h)) ys) (Function.comp (Fin.appen …
  -/
  subst h; simp
           /-
             🎉 no goals
           -/


@[simp] lemma append_cast_right {n m} (xs : Fin n → α) (ys : Fin m → α) (m' : ℕ)
    (h : m' = m) :
                                                                         /-
                                                                           m✝ n✝ : Nat
                                                                           α✝ : Fin (HAdd.hAdd n✝ 1) → Sort u
                                                                           x : α✝ 0
                                                                           q : (i : Fin (HAdd.hAdd n✝ 1)) → α✝ i
                                                                           p : (i : Fin n✝) → α✝ i.succ
                                                                           i : Fin n✝
                                                                           y : α✝ i.succ
                                                                           z : α✝ 0
                                                                           α : Sort u_1
                                                                           n m : Nat
                                                                           xs : Fin n → α
                                                                           ys : Fin m → α
                                                                           m' : Nat
                                                                           h : Eq m' m
                                                                           ⊢ Eq (HAdd.hAdd n m') (HAdd.hAdd n m)
                                                                         -/
    Fin.append xs (ys ∘ Fin.cast h) = Fin.append xs ys ∘ (Fin.cast <| by rw [h]) := by
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  /-
    α : Sort u_1
    n m : Nat
    xs : Fin n → α
    ys : Fin m → α
    m' : Nat
    h : Eq m' m
    ⊢ Eq (Fin.append xs (Function.comp ys (Fin.cast h))) (Function.comp (Fin.appen …
  -/
  subst h; simp
           /-
             🎉 no goals
           -/


lemma append_rev {m n} (xs : Fin m → α) (ys : Fin n → α) (i : Fin (m + n)) :
    append xs ys (rev i) = append (ys ∘ rev) (xs ∘ rev) (cast (Nat.add_comm ..) i) := by
  /-
    α : Sort u_1
    m n : Nat
    xs : Fin m → α
    ys : Fin n → α
    i : Fin (HAdd.hAdd m n)
    ⊢ Eq (Fin.append xs ys i.rev) (Fin.append (Function.comp ys Fin.rev) (Function …
  -/
  rcases rev_surjective i with ⟨i, rfl⟩
  /-
    case intro
    α : Sort u_1
    m n : Nat
    xs : Fin m → α
    ys : Fin n → α
    i : Fin (HAdd.hAdd m n)
    ⊢ Eq (Fin.append xs ys i.rev.rev) (Fin.append (Function.comp ys Fin.rev) (Func …
  -/
  rw [rev_rev]
  /-
    case intro
    α : Sort u_1
    m n : Nat
    xs : Fin m → α
    ys : Fin n → α
    i : Fin (HAdd.hAdd m n)
    ⊢ Eq (Fin.append xs ys i) (Fin.append (Function.comp ys Fin.rev) (Function.com …
  -/
  induction i using Fin.addCases
    /-
      case intro.left
      α : Sort u_1
      m n : Nat
      xs : Fin m → α
      ys : Fin n → α
      i✝ : Fin m
      ⊢ Eq (Fin.append xs ys (Fin.castAdd n i✝)) (Fin.append (Function.comp ys Fin.r …
    -/
  · simp [rev_castAdd]
    /-
      🎉 no goals
    -/
    /-
      case intro.right
      α : Sort u_1
      m n : Nat
      xs : Fin m → α
      ys : Fin n → α
      i✝ : Fin n
      ⊢ Eq (Fin.append xs ys (Fin.natAdd m i✝)) (Fin.append (Function.comp ys Fin.re …
    -/
  · simp [cast_rev, rev_addNat]
    /-
      🎉 no goals
    -/


lemma append_comp_rev {m n} (xs : Fin m → α) (ys : Fin n → α) :
    append xs ys ∘ rev = append (ys ∘ rev) (xs ∘ rev) ∘ cast (Nat.add_comm ..) :=
  funext <| append_rev xs ys


theorem append_castAdd_natAdd {f : Fin (m + n) → α} :
    append (fun i ↦ f (castAdd n i)) (fun i ↦ f (natAdd m i)) = f := by
  /-
    m n : Nat
    α : Sort u_1
    f : Fin (HAdd.hAdd m n) → α
    ⊢ Eq (Fin.append (fun i => f (Fin.castAdd n i)) fun i => f (Fin.natAdd m i)) f
  -/
  unfold append addCases
  /-
    m n : Nat
    α : Sort u_1
    f : Fin (HAdd.hAdd m n) → α
    ⊢ Eq (fun i => dite (LT.lt (↑i) m) (fun hi => Eq.rec ((fun i => f (Fin.castAdd …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Repeat `a` `m` times. For example `Fin.repeat 2 ![0, 3, 7] = ![0, 3, 7, 0, 3, 7]`. -/
-- Porting note: removed @[simp]
def «repeat» (m : ℕ) (a : Fin n → α) : Fin (m * n) → α
  | i => a i.modNat

-- Porting note: added (https://github.com/leanprover/lean4/issues/2042)

@[simp]
theorem repeat_apply (a : Fin n → α) (i : Fin (m * n)) :
    Fin.repeat m a i = a i.modNat :=
  rfl


@[simp]
theorem repeat_zero (a : Fin n → α) :
    Fin.repeat 0 a = Fin.elim0 ∘ cast (Nat.zero_mul _) :=
  funext fun x => (cast (Nat.zero_mul _) x).elim0


@[simp]
theorem repeat_one (a : Fin n → α) : Fin.repeat 1 a = a ∘ cast (Nat.one_mul _) := by
  /-
    n : Nat
    α : Sort u_1
    a : Fin n → α
    ⊢ Eq (Fin.repeat 1 a) (Function.comp a (Fin.cast ⋯))
  -/
  generalize_proofs h
  /-
    n : Nat
    α : Sort u_1
    a : Fin n → α
    h : Eq (HMul.hMul 1 n) n
    ⊢ Eq (Fin.repeat 1 a) (Function.comp a (Fin.cast h))
  -/
  apply funext
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    h : Eq (HMul.hMul 1 n) n
    ⊢ ∀ (x : Fin (HMul.hMul 1 n)), Eq (Fin.repeat 1 a x) (Function.comp a (Fin.cas …
  -/
  rw [(Fin.rightInverse_cast h.symm).surjective.forall]
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    h : Eq (HMul.hMul 1 n) n
    ⊢ ∀ (x : Fin n), Eq (Fin.repeat 1 a (Fin.cast ⋯ x)) (Function.comp a (Fin.cast …
  -/
  intro i
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    h : Eq (HMul.hMul 1 n) n
    i : Fin n
    ⊢ Eq (Fin.repeat 1 a (Fin.cast ⋯ i)) (Function.comp a (Fin.cast h) (Fin.cast ⋯ …
  -/
  simp [modNat, Nat.mod_eq_of_lt i.is_lt]
  /-
    🎉 no goals
  -/


theorem repeat_succ (a : Fin n → α) (m : ℕ) :
    Fin.repeat m.succ a =
      append a (Fin.repeat m a) ∘ cast ((Nat.succ_mul _ _).trans (Nat.add_comm ..)) := by
  /-
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m : Nat
    ⊢ Eq (Fin.repeat m.succ a) (Function.comp (Fin.append a (Fin.repeat m a)) (Fin …
  -/
  generalize_proofs h
  /-
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m : Nat
    h : Eq (HMul.hMul m.succ n) (HAdd.hAdd n (HMul.hMul m n))
    ⊢ Eq (Fin.repeat m.succ a) (Function.comp (Fin.append a (Fin.repeat m a)) (Fin …
  -/
  apply funext
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m : Nat
    h : Eq (HMul.hMul m.succ n) (HAdd.hAdd n (HMul.hMul m n))
    ⊢ ∀ (x : Fin (HMul.hMul m.succ n)), Eq (Fin.repeat m.succ a x) (Function.comp  …
  -/
  rw [(Fin.rightInverse_cast h.symm).surjective.forall]
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m : Nat
    h : Eq (HMul.hMul m.succ n) (HAdd.hAdd n (HMul.hMul m n))
    ⊢ ∀ (x : Fin (HAdd.hAdd n (HMul.hMul m n))), Eq (Fin.repeat m.succ a (Fin.cast …
  -/
  refine Fin.addCases (fun l => ?_) fun r => ?_
    /-
      case h.refine_1
      n : Nat
      α : Sort u_1
      a : Fin n → α
      m : Nat
      h : Eq (HMul.hMul m.succ n) (HAdd.hAdd n (HMul.hMul m n))
      l : Fin n
      ⊢ Eq (Fin.repeat m.succ a (Fin.cast ⋯ (Fin.castAdd (HMul.hMul m n) l))) (Funct …
    -/
  · simp [modNat, Nat.mod_eq_of_lt l.is_lt]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      n : Nat
      α : Sort u_1
      a : Fin n → α
      m : Nat
      h : Eq (HMul.hMul m.succ n) (HAdd.hAdd n (HMul.hMul m n))
      r : Fin (HMul.hMul m n)
      ⊢ Eq (Fin.repeat m.succ a (Fin.cast ⋯ (Fin.natAdd n r))) (Function.comp (Fin.a …
    -/
  · simp [modNat]
    /-
      🎉 no goals
    -/


@[simp]
theorem repeat_add (a : Fin n → α) (m₁ m₂ : ℕ) : Fin.repeat (m₁ + m₂) a =
    append (Fin.repeat m₁ a) (Fin.repeat m₂ a) ∘ cast (Nat.add_mul ..) := by
  /-
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m₁ m₂ : Nat
    ⊢ Eq (Fin.repeat (HAdd.hAdd m₁ m₂) a) (Function.comp (Fin.append (Fin.repeat m …
  -/
  generalize_proofs h
  /-
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m₁ m₂ : Nat
    h : Eq (HMul.hMul (HAdd.hAdd m₁ m₂) n) (HAdd.hAdd (HMul.hMul m₁ n) (HMul.hMul  …
    ⊢ Eq (Fin.repeat (HAdd.hAdd m₁ m₂) a) (Function.comp (Fin.append (Fin.repeat m …
  -/
  apply funext
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m₁ m₂ : Nat
    h : Eq (HMul.hMul (HAdd.hAdd m₁ m₂) n) (HAdd.hAdd (HMul.hMul m₁ n) (HMul.hMul  …
    ⊢ ∀ (x : Fin (HMul.hMul (HAdd.hAdd m₁ m₂) n)), Eq (Fin.repeat (HAdd.hAdd m₁ m₂ …
  -/
  rw [(Fin.rightInverse_cast h.symm).surjective.forall]
  /-
    case h
    n : Nat
    α : Sort u_1
    a : Fin n → α
    m₁ m₂ : Nat
    h : Eq (HMul.hMul (HAdd.hAdd m₁ m₂) n) (HAdd.hAdd (HMul.hMul m₁ n) (HMul.hMul  …
    ⊢ ∀ (x : Fin (HAdd.hAdd (HMul.hMul m₁ n) (HMul.hMul m₂ n))), Eq (Fin.repeat (H …
  -/
  refine Fin.addCases (fun l => ?_) fun r => ?_
    /-
      case h.refine_1
      n : Nat
      α : Sort u_1
      a : Fin n → α
      m₁ m₂ : Nat
      h : Eq (HMul.hMul (HAdd.hAdd m₁ m₂) n) (HAdd.hAdd (HMul.hMul m₁ n) (HMul.hMul  …
      l : Fin (HMul.hMul m₁ n)
      ⊢ Eq (Fin.repeat (HAdd.hAdd m₁ m₂) a (Fin.cast ⋯ (Fin.castAdd (HMul.hMul m₂ n) …
    -/
  · simp [modNat, Nat.mod_eq_of_lt l.is_lt]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      n : Nat
      α : Sort u_1
      a : Fin n → α
      m₁ m₂ : Nat
      h : Eq (HMul.hMul (HAdd.hAdd m₁ m₂) n) (HAdd.hAdd (HMul.hMul m₁ n) (HMul.hMul  …
      r : Fin (HMul.hMul m₂ n)
      ⊢ Eq (Fin.repeat (HAdd.hAdd m₁ m₂) a (Fin.cast ⋯ (Fin.natAdd (HMul.hMul m₁ n)  …
    -/
  · simp [modNat, Nat.add_mod]
    /-
      🎉 no goals
    -/


theorem repeat_rev (a : Fin n → α) (k : Fin (m * n)) :
    Fin.repeat m a k.rev = Fin.repeat m (a ∘ Fin.rev) k :=
  congr_arg a k.modNat_rev


theorem repeat_comp_rev (a : Fin n → α) :
    Fin.repeat m a ∘ Fin.rev = Fin.repeat m (a ∘ Fin.rev) :=
  funext <| repeat_rev a


/-- The beginning of an `n+1` tuple, i.e., its first `n` entries -/
def init (q : ∀ i, α i) (i : Fin n) : α i.castSucc :=
  q i.castSucc


theorem init_def {q : ∀ i, α i} :
    (init fun k : Fin (n + 1) ↦ q k) = fun k : Fin n ↦ q k.castSucc :=
  rfl


/-- Adding an element at the end of an `n`-tuple, to get an `n+1`-tuple. The name `snoc` comes from
`cons` (i.e., adding an element to the left of a tuple) read in reverse order. -/
def snoc (p : ∀ i : Fin n, α i.castSucc) (x : α (last n)) (i : Fin (n + 1)) : α i :=
                                        /-
                                          m n : Nat
                                          α : Fin (HAdd.hAdd n 1) → Sort u_1
                                          x✝ : α (Fin.last n)
                                          q : (i : Fin (HAdd.hAdd n 1)) → α i
                                          p✝ : (i : Fin n) → α i.castSucc
                                          i✝ : Fin n
                                          y : α i✝.castSucc
                                          z : α (Fin.last n)
                                          p : (i : Fin n) → α i.castSucc
                                          x : α (Fin.last n)
                                          i : Fin (HAdd.hAdd n 1)
                                          h : LT.lt (↑i) n
                                          ⊢ Eq (α (i.castLT h).castSucc) (α i)
                                        -/
  if h : i.val < n then _root_.cast (by rw [Fin.castSucc_castLT i h]) (p (castLT i h))
                                        /-
                                          🎉 no goals
                                        -/
                       /-
                         m n : Nat
                         α : Fin (HAdd.hAdd n 1) → Sort u_1
                         x✝ : α (Fin.last n)
                         q : (i : Fin (HAdd.hAdd n 1)) → α i
                         p✝ : (i : Fin n) → α i.castSucc
                         i✝ : Fin n
                         y : α i✝.castSucc
                         z : α (Fin.last n)
                         p : (i : Fin n) → α i.castSucc
                         x : α (Fin.last n)
                         i : Fin (HAdd.hAdd n 1)
                         h : Not (LT.lt (↑i) n)
                         ⊢ Eq (α (Fin.last n)) (α i)
                       -/
  else _root_.cast (by rw [eq_last_of_not_lt h]) x
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem init_snoc : init (snoc p x) = p := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    ⊢ Eq (Fin.init (Fin.snoc p x)) p
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    i : Fin n
    ⊢ Eq (Fin.init (Fin.snoc p x) i) (p i)
  -/
  simp only [init, snoc, coe_castSucc, is_lt, cast_eq, dite_true]
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    i : Fin n
    ⊢ Eq (p (i.castSucc.castLT ⋯)) (p i)
  -/
  convert cast_eq rfl (p i)
  /-
    🎉 no goals
  -/


@[simp]
theorem snoc_castSucc : snoc p x i.castSucc = p i := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    i : Fin n
    ⊢ Eq (Fin.snoc p x i.castSucc) (p i)
  -/
  simp only [snoc, coe_castSucc, is_lt, cast_eq, dite_true]
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    i : Fin n
    ⊢ Eq (p (i.castSucc.castLT ⋯)) (p i)
  -/
  convert cast_eq rfl (p i)
  /-
    🎉 no goals
  -/


@[simp]
theorem snoc_comp_castSucc {α : Sort*} {a : α} {f : Fin n → α} :
    (snoc f a : Fin (n + 1) → α) ∘ castSucc = f :=
                    /-
                      n : Nat
                      α : Sort u_2
                      a : α
                      f : Fin n → α
                      i : Fin n
                      ⊢ Eq (Function.comp (Fin.snoc f a) Fin.castSucc i) (f i)
                    -/
  funext fun i ↦ by rw [Function.comp_apply, snoc_castSucc]
                    /-
                      🎉 no goals
                    -/


@[simp]
                                                /-
                                                  n : Nat
                                                  α : Fin (HAdd.hAdd n 1) → Sort u_1
                                                  x : α (Fin.last n)
                                                  p : (i : Fin n) → α i.castSucc
                                                  ⊢ Eq (Fin.snoc p x (Fin.last n)) x
                                                -/
theorem snoc_last : snoc p x (last n) = x := by simp [snoc]
                                                /-
                                                  🎉 no goals
                                                -/


lemma snoc_zero {α : Sort*} (p : Fin 0 → α) (x : α) :
    Fin.snoc p x = fun _ ↦ x := by
  /-
    α : Sort u_2
    p : Fin 0 → α
    x : α
    ⊢ Eq (Fin.snoc p x) fun x_1 => x
  -/
  ext y
  /-
    case h
    α : Sort u_2
    p : Fin 0 → α
    x : α
    y : Fin (HAdd.hAdd 0 1)
    ⊢ Eq (Fin.snoc p x y) x
  -/
  have : Subsingleton (Fin (0 + 1)) := Fin.subsingleton_one
  /-
    case h
    α : Sort u_2
    p : Fin 0 → α
    x : α
    y : Fin (HAdd.hAdd 0 1)
    this : Subsingleton (Fin (HAdd.hAdd 0 1))
    ⊢ Eq (Fin.snoc p x y) x
  -/
  simp only [Subsingleton.elim y (Fin.last 0), snoc_last]
  /-
    🎉 no goals
  -/


@[simp]
theorem snoc_comp_nat_add {n m : ℕ} {α : Sort*} (f : Fin (m + n) → α) (a : α) :
    (snoc f a : Fin _ → α) ∘ (natAdd m : Fin (n + 1) → Fin (m + n + 1)) =
      snoc (f ∘ natAdd m) a := by
  /-
    n m : Nat
    α : Sort u_2
    f : Fin (HAdd.hAdd m n) → α
    a : α
    ⊢ Eq (Function.comp (Fin.snoc f a) (Fin.natAdd m)) (Fin.snoc (Function.comp f  …
  -/
  ext i
  /-
    case h
    n m : Nat
    α : Sort u_2
    f : Fin (HAdd.hAdd m n) → α
    a : α
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.comp (Fin.snoc f a) (Fin.natAdd m) i) (Fin.snoc (Function.comp  …
  -/
  refine Fin.lastCases ?_ (fun i ↦ ?_) i
    /-
      case h.refine_1
      n m : Nat
      α : Sort u_2
      f : Fin (HAdd.hAdd m n) → α
      a : α
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (Function.comp (Fin.snoc f a) (Fin.natAdd m) (Fin.last n)) (Fin.snoc (Fun …
    -/
  · simp only [Function.comp_apply]
    /-
      case h.refine_1
      n m : Nat
      α : Sort u_2
      f : Fin (HAdd.hAdd m n) → α
      a : α
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (Fin.snoc f a (Fin.natAdd m (Fin.last n))) (Fin.snoc (Function.comp f (Fi …
    -/
    rw [snoc_last, natAdd_last, snoc_last]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      n m : Nat
      α : Sort u_2
      f : Fin (HAdd.hAdd m n) → α
      a : α
      i✝ : Fin (HAdd.hAdd n 1)
      i : Fin n
      ⊢ Eq (Function.comp (Fin.snoc f a) (Fin.natAdd m) i.castSucc) (Fin.snoc (Funct …
    -/
  · simp only [comp_apply, snoc_castSucc]
    /-
      case h.refine_2
      n m : Nat
      α : Sort u_2
      f : Fin (HAdd.hAdd m n) → α
      a : α
      i✝ : Fin (HAdd.hAdd n 1)
      i : Fin n
      ⊢ Eq (Fin.snoc f a (Fin.natAdd m i.castSucc)) (f (Fin.natAdd m i))
    -/
    rw [natAdd_castSucc, snoc_castSucc]
    /-
      🎉 no goals
    -/


@[simp]
theorem snoc_cast_add {α : Fin (n + m + 1) → Sort*} (f : ∀ i : Fin (n + m), α i.castSucc)
    (a : α (last (n + m))) (i : Fin n) : (snoc f a) (castAdd (m + 1) i) = f (castAdd m i) :=
  dif_pos _

-- Porting note: Had to `unfold comp`

@[simp]
theorem snoc_comp_cast_add {n m : ℕ} {α : Sort*} (f : Fin (n + m) → α) (a : α) :
    (snoc f a : Fin _ → α) ∘ castAdd (m + 1) = f ∘ castAdd m :=
             /-
               n m : Nat
               α : Sort u_2
               f : Fin (HAdd.hAdd n m) → α
               a : α
               ⊢ ∀ (x : Fin n), Eq (Function.comp (Fin.snoc f a) (Fin.castAdd (HAdd.hAdd m 1) …
             -/
  funext (by unfold comp; exact snoc_cast_add _ _)
                          /-
                            🎉 no goals
                          -/


/-- Updating a tuple and adding an element at the end commute. -/
@[simp]
theorem snoc_update : snoc (update p i y) x = update (snoc p x) i.castSucc y := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    i : Fin n
    y : α i.castSucc
    ⊢ Eq (Fin.snoc (Function.update p i y) x) (Function.update (Fin.snoc p x) i.ca …
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    i : Fin n
    y : α i.castSucc
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.snoc (Function.update p i y) x j) (Function.update (Fin.snoc p x) i. …
  -/
  by_cases h : j.val < n
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      i : Fin n
      y : α i.castSucc
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (Fin.snoc (Function.update p i y) x j) (Function.update (Fin.snoc p x) i. …
    -/
  · rw [snoc]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      i : Fin n
      y : α i.castSucc
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (dite (LT.lt (↑j) n) (fun h => _root_.cast ⋯ (Function.update p i y (j.ca …
    -/
    simp only [h]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      i : Fin n
      y : α i.castSucc
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (dite True (fun h_1 => _root_.cast ⋯ (Function.update p i y (j.castLT ⋯)) …
    -/
    simp only [dif_pos]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      i : Fin n
      y : α i.castSucc
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (_root_.cast ⋯ (Function.update p i y (j.castLT ⋯))) (Function.update (Fi …
    -/
    by_cases h' : j = castSucc i
      /-
        case pos
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u_1
        x : α (Fin.last n)
        p : (i : Fin n) → α i.castSucc
        i : Fin n
        y : α i.castSucc
        j : Fin (HAdd.hAdd n 1)
        h : LT.lt (↑j) n
        h' : Eq j i.castSucc
        ⊢ Eq (_root_.cast ⋯ (Function.update p i y (j.castLT ⋯))) (Function.update (Fi …
      -/
    · have C1 : α i.castSucc = α j := by rw [h']
      have E1 : update (snoc p x) i.castSucc y j = _root_.cast C1 y := by
        have : update (snoc p x) j (_root_.cast C1 y) j = _root_.cast C1 y := by simp
        convert this
        · exact h'.symm
        · exact heq_of_cast_eq (congr_arg α (Eq.symm h')) rfl
      /-
        case pos
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u_1
        x : α (Fin.last n)
        p : (i : Fin n) → α i.castSucc
        i : Fin n
        y : α i.castSucc
        j : Fin (HAdd.hAdd n 1)
        h : LT.lt (↑j) n
        h' : Eq j i.castSucc
        C1 : Eq (α i.castSucc) (α j)
        E1 : Eq (Function.update (Fin.snoc p x) i.castSucc y j) (_root_.cast C1 y)
        ⊢ Eq (_root_.cast ⋯ (Function.update p i y (j.castLT ⋯))) (Function.update (Fi …
      -/
      have C2 : α i.castSucc = α (castLT j h).castSucc := by rw [castSucc_castLT, h']
      have E2 : update p i y (castLT j h) = _root_.cast C2 y := by
        have : update p (castLT j h) (_root_.cast C2 y) (castLT j h) = _root_.cast C2 y := by simp
        convert this
        · simp [h, h']
        · exact heq_of_cast_eq C2 rfl
      /-
        case pos
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u_1
        x : α (Fin.last n)
        p : (i : Fin n) → α i.castSucc
        i : Fin n
        y : α i.castSucc
        j : Fin (HAdd.hAdd n 1)
        h : LT.lt (↑j) n
        h' : Eq j i.castSucc
        C1 : Eq (α i.castSucc) (α j)
        E1 : Eq (Function.update (Fin.snoc p x) i.castSucc y j) (_root_.cast C1 y)
        C2 : Eq (α i.castSucc) (α (j.castLT h).castSucc)
        E2 : Eq (Function.update p i y (j.castLT h)) (_root_.cast C2 y)
        ⊢ Eq (_root_.cast ⋯ (Function.update p i y (j.castLT ⋯))) (Function.update (Fi …
      -/
      rw [E1, E2]
      /-
        case pos
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u_1
        x : α (Fin.last n)
        p : (i : Fin n) → α i.castSucc
        i : Fin n
        y : α i.castSucc
        j : Fin (HAdd.hAdd n 1)
        h : LT.lt (↑j) n
        h' : Eq j i.castSucc
        C1 : Eq (α i.castSucc) (α j)
        E1 : Eq (Function.update (Fin.snoc p x) i.castSucc y j) (_root_.cast C1 y)
        C2 : Eq (α i.castSucc) (α (j.castLT h).castSucc)
        E2 : Eq (Function.update p i y (j.castLT h)) (_root_.cast C2 y)
        ⊢ Eq (_root_.cast ⋯ (_root_.cast C2 y)) (_root_.cast C1 y)
      -/
      rfl
      /-
        🎉 no goals
      -/
    · have : ¬castLT j h = i := by
        intro E
        apply h'
        rw [← E, castSucc_castLT]
      /-
        case neg
        n : Nat
        α : Fin (HAdd.hAdd n 1) → Sort u_1
        x : α (Fin.last n)
        p : (i : Fin n) → α i.castSucc
        i : Fin n
        y : α i.castSucc
        j : Fin (HAdd.hAdd n 1)
        h : LT.lt (↑j) n
        h' : Not (Eq j i.castSucc)
        this : Not (Eq (j.castLT h) i)
        ⊢ Eq (_root_.cast ⋯ (Function.update p i y (j.castLT ⋯))) (Function.update (Fi …
      -/
      simp [h', this, snoc, h]
      /-
        🎉 no goals
      -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      i : Fin n
      y : α i.castSucc
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Fin.snoc (Function.update p i y) x j) (Function.update (Fin.snoc p x) i. …
    -/
  · rw [eq_last_of_not_lt h]
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      i : Fin n
      y : α i.castSucc
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Fin.snoc (Function.update p i y) x (Fin.last n)) (Function.update (Fin.s …
    -/
    simp [Fin.ne_of_gt i.castSucc_lt_last]
    /-
      🎉 no goals
    -/


/-- Adding an element at the beginning of a tuple and then updating it amounts to adding it
directly. -/
theorem update_snoc_last : update (snoc p x) (last n) z = snoc p z := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    z : α (Fin.last n)
    ⊢ Eq (Function.update (Fin.snoc p x) (Fin.last n) z) (Fin.snoc p z)
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (i : Fin n) → α i.castSucc
    z : α (Fin.last n)
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.update (Fin.snoc p x) (Fin.last n) z j) (Fin.snoc p z j)
  -/
  by_cases h : j.val < n
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      z : α (Fin.last n)
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (Function.update (Fin.snoc p x) (Fin.last n) z j) (Fin.snoc p z j)
    -/
  · have : j ≠ last n := Fin.ne_of_lt h
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      z : α (Fin.last n)
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      this : Ne j (Fin.last n)
      ⊢ Eq (Function.update (Fin.snoc p x) (Fin.last n) z j) (Fin.snoc p z j)
    -/
    simp [h, update_of_ne, this, snoc]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      z : α (Fin.last n)
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Function.update (Fin.snoc p x) (Fin.last n) z j) (Fin.snoc p z j)
    -/
  · rw [eq_last_of_not_lt h]
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (i : Fin n) → α i.castSucc
      z : α (Fin.last n)
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Function.update (Fin.snoc p x) (Fin.last n) z (Fin.last n)) (Fin.snoc p  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Concatenating the first element of a tuple with its tail gives back the original tuple -/
@[simp]
theorem snoc_init_self : snoc (init q) (q (last n)) = q := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    ⊢ Eq (Fin.snoc (Fin.init q) (q (Fin.last n))) q
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.snoc (Fin.init q) (q (Fin.last n)) j) (q j)
  -/
  by_cases h : j.val < n
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (Fin.snoc (Fin.init q) (q (Fin.last n)) j) (q j)
    -/
  · simp only [init, snoc, h, cast_eq, dite_true, castSucc_castLT]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Fin.snoc (Fin.init q) (q (Fin.last n)) j) (q j)
    -/
  · rw [eq_last_of_not_lt h]
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Fin.snoc (Fin.init q) (q (Fin.last n)) (Fin.last n)) (q (Fin.last n))
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Updating the last element of a tuple does not change the beginning. -/
@[simp]
theorem init_update_last : init (update q (last n) z) = init q := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    z : α (Fin.last n)
    ⊢ Eq (Fin.init (Function.update q (Fin.last n) z)) (Fin.init q)
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    z : α (Fin.last n)
    j : Fin n
    ⊢ Eq (Fin.init (Function.update q (Fin.last n) z) j) (Fin.init q j)
  -/
  simp [init, Fin.ne_of_lt, castSucc_lt_last]
  /-
    🎉 no goals
  -/


/-- Updating an element and taking the beginning commute. -/
@[simp]
theorem init_update_castSucc : init (update q i.castSucc y) = update (init q) i y := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin n
    y : α i.castSucc
    ⊢ Eq (Fin.init (Function.update q i.castSucc y)) (Function.update (Fin.init q) …
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    q : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin n
    y : α i.castSucc
    j : Fin n
    ⊢ Eq (Fin.init (Function.update q i.castSucc y) j) (Function.update (Fin.init  …
  -/
  by_cases h : j = i
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      i : Fin n
      y : α i.castSucc
      j : Fin n
      h : Eq j i
      ⊢ Eq (Fin.init (Function.update q i.castSucc y) j) (Function.update (Fin.init  …
    -/
  · rw [h]
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      i : Fin n
      y : α i.castSucc
      j : Fin n
      h : Eq j i
      ⊢ Eq (Fin.init (Function.update q i.castSucc y) i) (Function.update (Fin.init  …
    -/
    simp [init]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      q : (i : Fin (HAdd.hAdd n 1)) → α i
      i : Fin n
      y : α i.castSucc
      j : Fin n
      h : Not (Eq j i)
      ⊢ Eq (Fin.init (Function.update q i.castSucc y) j) (Function.update (Fin.init  …
    -/
  · simp [init, h, castSucc_inj]
    /-
      🎉 no goals
    -/


/-- `tail` and `init` commute. We state this lemma in a non-dependent setting, as otherwise it
would involve a cast to convince Lean that the two types are equal, making it harder to use. -/
theorem tail_init_eq_init_tail {β : Sort*} (q : Fin (n + 2) → β) :
    tail (init q) = init (tail q) := by
  /-
    n : Nat
    β : Sort u_2
    q : Fin (HAdd.hAdd n 2) → β
    ⊢ Eq (Fin.tail (Fin.init q)) (Fin.init (Fin.tail q))
  -/
  ext i
  /-
    case h
    n : Nat
    β : Sort u_2
    q : Fin (HAdd.hAdd n 2) → β
    i : Fin n
    ⊢ Eq (Fin.tail (Fin.init q) i) (Fin.init (Fin.tail q) i)
  -/
  simp [tail, init, castSucc_fin_succ]
  /-
    🎉 no goals
  -/


/-- `cons` and `snoc` commute. We state this lemma in a non-dependent setting, as otherwise it
would involve a cast to convince Lean that the two types are equal, making it harder to use. -/
theorem cons_snoc_eq_snoc_cons {β : Sort*} (a : β) (q : Fin n → β) (b : β) :
    @cons n.succ (fun _ ↦ β) a (snoc q b) = snoc (cons a q) b := by
  /-
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    ⊢ Eq (Fin.cons a (Fin.snoc q b)) (Fin.snoc (Fin.cons a q) b)
  -/
  ext i
  /-
    case h
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    ⊢ Eq (Fin.cons a (Fin.snoc q b) i) (Fin.snoc (Fin.cons a q) b i)
  -/
  by_cases h : i = 0
    /-
      case pos
      n : Nat
      β : Sort u_2
      a : β
      q : Fin n → β
      b : β
      i : Fin (HAdd.hAdd n.succ 1)
      h : Eq i 0
      ⊢ Eq (Fin.cons a (Fin.snoc q b) i) (Fin.snoc (Fin.cons a q) b i)
    -/
  · rw [h]
    -- Porting note: `refl` finished it here in Lean 3, but I had to add more.
    /-
      case pos
      n : Nat
      β : Sort u_2
      a : β
      q : Fin n → β
      b : β
      i : Fin (HAdd.hAdd n.succ 1)
      h : Eq i 0
      ⊢ Eq (Fin.cons a (Fin.snoc q b) 0) (Fin.snoc (Fin.cons a q) b 0)
    -/
    simp [snoc, castLT]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    h : Not (Eq i 0)
    ⊢ Eq (Fin.cons a (Fin.snoc q b) i) (Fin.snoc (Fin.cons a q) b i)
  -/
  set j := pred i h with ji
  /-
    case neg
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    h : Not (Eq i 0)
    j : Fin (HAdd.hAdd n 1) := i.pred h
    ji : Eq j (i.pred h)
    ⊢ Eq (Fin.cons a (Fin.snoc q b) i) (Fin.snoc (Fin.cons a q) b i)
  -/
  have : i = j.succ := by rw [ji, succ_pred]
  /-
    case neg
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    h : Not (Eq i 0)
    j : Fin (HAdd.hAdd n 1) := i.pred h
    ji : Eq j (i.pred h)
    this : Eq i j.succ
    ⊢ Eq (Fin.cons a (Fin.snoc q b) i) (Fin.snoc (Fin.cons a q) b i)
  -/
  rw [this, cons_succ]
  /-
    case neg
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    h : Not (Eq i 0)
    j : Fin (HAdd.hAdd n 1) := i.pred h
    ji : Eq j (i.pred h)
    this : Eq i j.succ
    ⊢ Eq (Fin.snoc q b j) (Fin.snoc (Fin.cons a q) b j.succ)
  -/
  by_cases h' : j.val < n
    /-
      case pos
      n : Nat
      β : Sort u_2
      a : β
      q : Fin n → β
      b : β
      i : Fin (HAdd.hAdd n.succ 1)
      h : Not (Eq i 0)
      j : Fin (HAdd.hAdd n 1) := i.pred h
      ji : Eq j (i.pred h)
      this : Eq i j.succ
      h' : LT.lt (↑j) n
      ⊢ Eq (Fin.snoc q b j) (Fin.snoc (Fin.cons a q) b j.succ)
    -/
  · set k := castLT j h' with jk
    /-
      case pos
      n : Nat
      β : Sort u_2
      a : β
      q : Fin n → β
      b : β
      i : Fin (HAdd.hAdd n.succ 1)
      h : Not (Eq i 0)
      j : Fin (HAdd.hAdd n 1) := i.pred h
      ji : Eq j (i.pred h)
      this : Eq i j.succ
      h' : LT.lt (↑j) n
      k : Fin n := j.castLT h'
      jk : Eq k (j.castLT h')
      ⊢ Eq (Fin.snoc q b j) (Fin.snoc (Fin.cons a q) b j.succ)
    -/
    have : j = castSucc k := by rw [jk, castSucc_castLT]
    /-
      case pos
      n : Nat
      β : Sort u_2
      a : β
      q : Fin n → β
      b : β
      i : Fin (HAdd.hAdd n.succ 1)
      h : Not (Eq i 0)
      j : Fin (HAdd.hAdd n 1) := i.pred h
      ji : Eq j (i.pred h)
      this✝ : Eq i j.succ
      h' : LT.lt (↑j) n
      k : Fin n := j.castLT h'
      jk : Eq k (j.castLT h')
      this : Eq j k.castSucc
      ⊢ Eq (Fin.snoc q b j) (Fin.snoc (Fin.cons a q) b j.succ)
    -/
    rw [this, ← castSucc_fin_succ, snoc]
    /-
      case pos
      n : Nat
      β : Sort u_2
      a : β
      q : Fin n → β
      b : β
      i : Fin (HAdd.hAdd n.succ 1)
      h : Not (Eq i 0)
      j : Fin (HAdd.hAdd n 1) := i.pred h
      ji : Eq j (i.pred h)
      this✝ : Eq i j.succ
      h' : LT.lt (↑j) n
      k : Fin n := j.castLT h'
      jk : Eq k (j.castLT h')
      this : Eq j k.castSucc
      ⊢ Eq (dite (LT.lt (↑k.castSucc) n) (fun h => _root_.cast ⋯ (q (k.castSucc.cast …
    -/
    simp [pred, snoc, cons]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    h : Not (Eq i 0)
    j : Fin (HAdd.hAdd n 1) := i.pred h
    ji : Eq j (i.pred h)
    this : Eq i j.succ
    h' : Not (LT.lt (↑j) n)
    ⊢ Eq (Fin.snoc q b j) (Fin.snoc (Fin.cons a q) b j.succ)
  -/
  rw [eq_last_of_not_lt h', succ_last]
  /-
    case neg
    n : Nat
    β : Sort u_2
    a : β
    q : Fin n → β
    b : β
    i : Fin (HAdd.hAdd n.succ 1)
    h : Not (Eq i 0)
    j : Fin (HAdd.hAdd n 1) := i.pred h
    ji : Eq j (i.pred h)
    this : Eq i j.succ
    h' : Not (LT.lt (↑j) n)
    ⊢ Eq (Fin.snoc q b (Fin.last n)) (Fin.snoc (Fin.cons a q) b (Fin.last n.succ))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem comp_snoc {α : Sort*} {β : Sort*} (g : α → β) (q : Fin n → α) (y : α) :
    g ∘ snoc q y = snoc (g ∘ q) (g y) := by
  /-
    n : Nat
    α : Sort u_2
    β : Sort u_3
    g : α → β
    q : Fin n → α
    y : α
    ⊢ Eq (Function.comp g (Fin.snoc q y)) (Fin.snoc (Function.comp g q) (g y))
  -/
  ext j
  /-
    case h
    n : Nat
    α : Sort u_2
    β : Sort u_3
    g : α → β
    q : Fin n → α
    y : α
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.comp g (Fin.snoc q y) j) (Fin.snoc (Function.comp g q) (g y) j)
  -/
  by_cases h : j.val < n
    /-
      case pos
      n : Nat
      α : Sort u_2
      β : Sort u_3
      g : α → β
      q : Fin n → α
      y : α
      j : Fin (HAdd.hAdd n 1)
      h : LT.lt (↑j) n
      ⊢ Eq (Function.comp g (Fin.snoc q y) j) (Fin.snoc (Function.comp g q) (g y) j)
    -/
  · simp [h, snoc, castSucc_castLT]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Sort u_2
      β : Sort u_3
      g : α → β
      q : Fin n → α
      y : α
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Function.comp g (Fin.snoc q y) j) (Fin.snoc (Function.comp g q) (g y) j)
    -/
  · rw [eq_last_of_not_lt h]
    /-
      case neg
      n : Nat
      α : Sort u_2
      β : Sort u_3
      g : α → β
      q : Fin n → α
      y : α
      j : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt (↑j) n)
      ⊢ Eq (Function.comp g (Fin.snoc q y) (Fin.last n)) (Fin.snoc (Function.comp g  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Appending a one-tuple to the right is the same as `Fin.snoc`. -/
theorem append_right_eq_snoc {α : Sort*} {n : ℕ} (x : Fin n → α) (x₀ : Fin 1 → α) :
    Fin.append x x₀ = Fin.snoc x (x₀ 0) := by
  /-
    α : Sort u_2
    n : Nat
    x : Fin n → α
    x₀ : Fin 1 → α
    ⊢ Eq (Fin.append x x₀) (Fin.snoc x (x₀ 0))
  -/
  ext i
  /-
    case h
    α : Sort u_2
    n : Nat
    x : Fin n → α
    x₀ : Fin 1 → α
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.append x x₀ i) (Fin.snoc x (x₀ 0) i)
  -/
  refine Fin.addCases ?_ ?_ i <;> clear i
    /-
      case h.refine_1
      α : Sort u_2
      n : Nat
      x : Fin n → α
      x₀ : Fin 1 → α
      ⊢ ∀ (i : Fin n), Eq (Fin.append x x₀ (Fin.castAdd 1 i)) (Fin.snoc x (x₀ 0) (Fi …
    -/
  · intro i
    /-
      case h.refine_1
      α : Sort u_2
      n : Nat
      x : Fin n → α
      x₀ : Fin 1 → α
      i : Fin n
      ⊢ Eq (Fin.append x x₀ (Fin.castAdd 1 i)) (Fin.snoc x (x₀ 0) (Fin.castAdd 1 i))
    -/
    rw [Fin.append_left]
    /-
      case h.refine_1
      α : Sort u_2
      n : Nat
      x : Fin n → α
      x₀ : Fin 1 → α
      i : Fin n
      ⊢ Eq (x i) (Fin.snoc x (x₀ 0) (Fin.castAdd 1 i))
    -/
    exact (@snoc_castSucc _ (fun _ => α) _ _ i).symm
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Sort u_2
      n : Nat
      x : Fin n → α
      x₀ : Fin 1 → α
      ⊢ ∀ (i : Fin 1), Eq (Fin.append x x₀ (Fin.natAdd n i)) (Fin.snoc x (x₀ 0) (Fin …
    -/
  · intro i
    /-
      case h.refine_2
      α : Sort u_2
      n : Nat
      x : Fin n → α
      x₀ : Fin 1 → α
      i : Fin 1
      ⊢ Eq (Fin.append x x₀ (Fin.natAdd n i)) (Fin.snoc x (x₀ 0) (Fin.natAdd n i))
    -/
    rw [Subsingleton.elim i 0, Fin.append_right]
    /-
      case h.refine_2
      α : Sort u_2
      n : Nat
      x : Fin n → α
      x₀ : Fin 1 → α
      i : Fin 1
      ⊢ Eq (x₀ 0) (Fin.snoc x (x₀ 0) (Fin.natAdd n 0))
    -/
    exact (@snoc_last _ (fun _ => α) _ _).symm
    /-
      🎉 no goals
    -/


/-- `Fin.snoc` is the same as appending a one-tuple -/
theorem snoc_eq_append {α : Sort*} (xs : Fin n → α) (x : α) :
    snoc xs x = append xs (cons x Fin.elim0) :=
  (append_right_eq_snoc xs (cons x Fin.elim0)).symm


theorem append_left_snoc {n m} {α : Sort*} (xs : Fin n → α) (x : α) (ys : Fin m → α) :
    Fin.append (Fin.snoc xs x) ys =
      Fin.append xs (Fin.cons x ys) ∘ Fin.cast (Nat.succ_add_eq_add_succ ..) := by
  /-
    n m : Nat
    α : Sort u_2
    xs : Fin n → α
    x : α
    ys : Fin m → α
    ⊢ Eq (Fin.append (Fin.snoc xs x) ys) (Function.comp (Fin.append xs (Fin.cons x …
  -/
  rw [snoc_eq_append, append_assoc, append_left_eq_cons, append_cast_right]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem append_right_cons {n m} {α : Sort*} (xs : Fin n → α) (y : α) (ys : Fin m → α) :
    Fin.append xs (Fin.cons y ys) =
      Fin.append (Fin.snoc xs y) ys ∘ Fin.cast (Nat.succ_add_eq_add_succ ..).symm := by
  /-
    n m : Nat
    α : Sort u_2
    xs : Fin n → α
    y : α
    ys : Fin m → α
    ⊢ Eq (Fin.append xs (Fin.cons y ys)) (Function.comp (Fin.append (Fin.snoc xs y …
  -/
  rw [append_left_snoc]; rfl
                         /-
                           🎉 no goals
                         -/


theorem append_cons {α : Sort*} (a : α) (as : Fin n → α) (bs : Fin m → α) :
    Fin.append (cons a as) bs
    = cons a (Fin.append as bs) ∘ (Fin.cast <| Nat.add_right_comm n 1 m) := by
  /-
    m n : Nat
    α : Sort u_2
    a : α
    as : Fin n → α
    bs : Fin m → α
    ⊢ Eq (Fin.append (Fin.cons a as) bs) (Function.comp (Fin.cons a (Fin.append as …
  -/
  funext i
  /-
    case h
    m n : Nat
    α : Sort u_2
    a : α
    as : Fin n → α
    bs : Fin m → α
    i : Fin (HAdd.hAdd (HAdd.hAdd n 1) m)
    ⊢ Eq (Fin.append (Fin.cons a as) bs i) (Function.comp (Fin.cons a (Fin.append  …
  -/
  rcases i with ⟨i, -⟩
  /-
    case h.mk
    m n : Nat
    α : Sort u_2
    a : α
    as : Fin n → α
    bs : Fin m → α
    i : Nat
    isLt✝ : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) m)
    ⊢ Eq (Fin.append (Fin.cons a as) bs ⟨i, isLt✝⟩) (Function.comp (Fin.cons a (Fi …
  -/
  simp only [append, addCases, cons, castLT, cast, comp_apply]
  /-
    case h.mk
    m n : Nat
    α : Sort u_2
    a : α
    as : Fin n → α
    bs : Fin m → α
    i : Nat
    isLt✝ : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) m)
    ⊢ Eq (dite (LT.lt i (HAdd.hAdd n 1)) (fun h => Fin.cases a as ⟨i, ⋯⟩) fun h => …
  -/
  cases' i with i
    /-
      case h.mk.zero
      m n : Nat
      α : Sort u_2
      a : α
      as : Fin n → α
      bs : Fin m → α
      isLt✝ : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) m)
      ⊢ Eq (dite (LT.lt 0 (HAdd.hAdd n 1)) (fun h => Fin.cases a as ⟨0, ⋯⟩) fun h => …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.mk.succ
      m n : Nat
      α : Sort u_2
      a : α
      as : Fin n → α
      bs : Fin m → α
      i : Nat
      isLt✝ : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) m)
      ⊢ Eq (dite (LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1)) (fun h => Fin.cases a as ⟨H …
    -/
  · split_ifs with h
      /-
        case pos
        m n : Nat
        α : Sort u_2
        a : α
        as : Fin n → α
        bs : Fin m → α
        i : Nat
        isLt✝ : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) m)
        h : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1)
        ⊢ Eq (Fin.cases a as ⟨HAdd.hAdd i 1, ⋯⟩) (Fin.cases a (Fin.addCases as bs) ⟨HA …
      -/
    · have : i < n := Nat.lt_of_succ_lt_succ h
      /-
        case pos
        m n : Nat
        α : Sort u_2
        a : α
        as : Fin n → α
        bs : Fin m → α
        i : Nat
        isLt✝ : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) m)
        h : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1)
        this : LT.lt i n
        ⊢ Eq (Fin.cases a as ⟨HAdd.hAdd i 1, ⋯⟩) (Fin.cases a (Fin.addCases as bs) ⟨HA …
      -/
      simp [addCases, this]
      /-
        🎉 no goals
      -/
      /-
        case neg
        m n : Nat
        α : Sort u_2
        a : α
        as : Fin n → α
        bs : Fin m → α
        i : Nat
        isLt✝ : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) m)
        h : Not (LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1))
        ⊢ Eq (Eq.rec (bs (Fin.subNat (HAdd.hAdd n 1) ⟨HAdd.hAdd i 1, ⋯⟩ ⋯)) ⋯) (Fin.ca …
      -/
    · have : ¬i < n := Nat.not_le.mpr <| Nat.lt_succ.mp <| Nat.not_le.mp h
      /-
        case neg
        m n : Nat
        α : Sort u_2
        a : α
        as : Fin n → α
        bs : Fin m → α
        i : Nat
        isLt✝ : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) m)
        h : Not (LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1))
        this : Not (LT.lt i n)
        ⊢ Eq (Eq.rec (bs (Fin.subNat (HAdd.hAdd n 1) ⟨HAdd.hAdd i 1, ⋯⟩ ⋯)) ⋯) (Fin.ca …
      -/
      simp [addCases, this]
      /-
        🎉 no goals
      -/


theorem append_snoc {α : Sort*} (as : Fin n → α) (bs : Fin m → α) (b : α) :
    Fin.append as (snoc bs b) = snoc (Fin.append as bs) b := by
  /-
    m n : Nat
    α : Sort u_2
    as : Fin n → α
    bs : Fin m → α
    b : α
    ⊢ Eq (Fin.append as (Fin.snoc bs b)) (Fin.snoc (Fin.append as bs) b)
  -/
  funext i
  /-
    case h
    m n : Nat
    α : Sort u_2
    as : Fin n → α
    bs : Fin m → α
    b : α
    i : Fin (HAdd.hAdd n (HAdd.hAdd m 1))
    ⊢ Eq (Fin.append as (Fin.snoc bs b) i) (Fin.snoc (Fin.append as bs) b i)
  -/
  rcases i with ⟨i, isLt⟩
  simp only [append, addCases, castLT, cast_mk, subNat_mk, natAdd_mk, cast, snoc.eq_1,
    cast_eq, eq_rec_constant, Nat.add_eq, Nat.add_zero, castLT_mk]
  /-
    case h.mk
    m n : Nat
    α : Sort u_2
    as : Fin n → α
    bs : Fin m → α
    b : α
    i : Nat
    isLt : LT.lt i (HAdd.hAdd n (HAdd.hAdd m 1))
    ⊢ Eq (dite (LT.lt i n) (fun h => as ⟨i, ⋯⟩) fun h => dite (LT.lt (HSub.hSub i  …
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  split_ifs with lt_n lt_add sub_lt nlt_add lt_add <;> (try rfl)
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      case neg
      m n : Nat
      α : Sort u_2
      as : Fin n → α
      bs : Fin m → α
      b : α
      i : Nat
      isLt : LT.lt i (HAdd.hAdd n (HAdd.hAdd m 1))
      lt_n : LT.lt i n
      lt_add : Not (LT.lt i (HAdd.hAdd n m))
      ⊢ Eq (as ⟨i, ⋯⟩) b
    -/
  · have := Nat.lt_add_right m lt_n
    /-
      case neg
      m n : Nat
      α : Sort u_2
      as : Fin n → α
      bs : Fin m → α
      b : α
      i : Nat
      isLt : LT.lt i (HAdd.hAdd n (HAdd.hAdd m 1))
      lt_n : LT.lt i n
      lt_add : Not (LT.lt i (HAdd.hAdd n m))
      this : LT.lt i (HAdd.hAdd n m)
      ⊢ Eq (as ⟨i, ⋯⟩) b
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      α : Sort u_2
      as : Fin n → α
      bs : Fin m → α
      b : α
      i : Nat
      isLt : LT.lt i (HAdd.hAdd n (HAdd.hAdd m 1))
      lt_n : Not (LT.lt i n)
      sub_lt : LT.lt (HSub.hSub i n) m
      nlt_add : Not (LT.lt i (HAdd.hAdd n m))
      ⊢ Eq (bs ⟨HSub.hSub i n, ⋯⟩) b
    -/
  · obtain rfl := Nat.eq_of_le_of_lt_succ (Nat.not_lt.mp nlt_add) isLt
    /-
      case neg
      m n : Nat
      α : Sort u_2
      as : Fin n → α
      bs : Fin m → α
      b : α
      isLt : LT.lt (HAdd.hAdd n m) (HAdd.hAdd n (HAdd.hAdd m 1))
      lt_n : Not (LT.lt (HAdd.hAdd n m) n)
      sub_lt : LT.lt (HSub.hSub (HAdd.hAdd n m) n) m
      nlt_add : Not (LT.lt (HAdd.hAdd n m) (HAdd.hAdd n m))
      ⊢ Eq (bs ⟨HSub.hSub (HAdd.hAdd n m) n, ⋯⟩) b
    -/
    simp [Nat.add_comm n m] at sub_lt
    /-
      🎉 no goals
    -/
    /-
      case pos
      m n : Nat
      α : Sort u_2
      as : Fin n → α
      bs : Fin m → α
      b : α
      i : Nat
      isLt : LT.lt i (HAdd.hAdd n (HAdd.hAdd m 1))
      lt_n : Not (LT.lt i n)
      sub_lt : Not (LT.lt (HSub.hSub i n) m)
      lt_add : LT.lt i (HAdd.hAdd n m)
      ⊢ Eq b (bs ⟨HSub.hSub i n, ⋯⟩)
    -/
  · have := Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp lt_n) lt_add
    /-
      case pos
      m n : Nat
      α : Sort u_2
      as : Fin n → α
      bs : Fin m → α
      b : α
      i : Nat
      isLt : LT.lt i (HAdd.hAdd n (HAdd.hAdd m 1))
      lt_n : Not (LT.lt i n)
      sub_lt : Not (LT.lt (HSub.hSub i n) m)
      lt_add : LT.lt i (HAdd.hAdd n m)
      this : LT.lt (HSub.hSub i n) m
      ⊢ Eq b (bs ⟨HSub.hSub i n, ⋯⟩)
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem comp_init {α : Sort*} {β : Sort*} (g : α → β) (q : Fin n.succ → α) :
    g ∘ init q = init (g ∘ q) := by
  /-
    n : Nat
    α : Sort u_2
    β : Sort u_3
    g : α → β
    q : Fin n.succ → α
    ⊢ Eq (Function.comp g (Fin.init q)) (Fin.init (Function.comp g q))
  -/
  ext j
  /-
    case h
    n : Nat
    α : Sort u_2
    β : Sort u_3
    g : α → β
    q : Fin n.succ → α
    j : Fin n
    ⊢ Eq (Function.comp g (Fin.init q) j) (Fin.init (Function.comp g q) j)
  -/
  simp [init]
  /-
    🎉 no goals
  -/


/-- Equivalence between tuples of length `n + 1` and pairs of an element and a tuple of length `n`
given by separating out the last element of the tuple.

This is `Fin.snoc` as an `Equiv`. -/
@[simps]
def snocEquiv (α : Fin (n + 1) → Type*) : α (last n) × (∀ i, α (castSucc i)) ≃ ∀ i, α i where
  toFun f _ := Fin.snoc f.2 f.1 _
  invFun f := ⟨f _, Fin.init f⟩
                   /-
                     m n : Nat
                     α✝ : Fin (HAdd.hAdd n 1) → Sort u_1
                     x : α✝ (Fin.last n)
                     q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                     p : (i : Fin n) → α✝ i.castSucc
                     i : Fin n
                     y : α✝ i.castSucc
                     z : α✝ (Fin.last n)
                     α : Fin (HAdd.hAdd n 1) → Type u_2
                     f : Prod (α (Fin.last n)) ((i : Fin n) → α i.castSucc)
                     ⊢ Eq ((fun f => { fst := f (Fin.last n), snd := Fin.init f }) ((fun f x => Fin …
                   -/
  left_inv f := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      m n : Nat
                      α✝ : Fin (HAdd.hAdd n 1) → Sort u_1
                      x : α✝ (Fin.last n)
                      q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                      p : (i : Fin n) → α✝ i.castSucc
                      i : Fin n
                      y : α✝ i.castSucc
                      z : α✝ (Fin.last n)
                      α : Fin (HAdd.hAdd n 1) → Type u_2
                      f : (i : Fin (HAdd.hAdd n 1)) → α i
                      ⊢ Eq ((fun f x => Fin.snoc f.2 f.1 x) ((fun f => { fst := f (Fin.last n), snd  …
                    -/
  right_inv f := by simp
                    /-
                      🎉 no goals
                    -/


/-- Recurse on an `n+1`-tuple by splitting it its initial `n`-tuple and its last element. -/
@[elab_as_elim, inline]
def snocCases {P : (∀ i : Fin n.succ, α i) → Sort*}
    (h : ∀ xs x, P (Fin.snoc xs x))
    (x : ∀ i : Fin n.succ, α i) : P x :=
                  /-
                    m n : Nat
                    α : Fin (HAdd.hAdd n 1) → Sort u_1
                    x✝ : α (Fin.last n)
                    q : (i : Fin (HAdd.hAdd n 1)) → α i
                    p : (i : Fin n) → α i.castSucc
                    i : Fin n
                    y : α i.castSucc
                    z : α (Fin.last n)
                    P : ((i : Fin n.succ) → α i) → Sort u_2
                    h : (xs : (i : Fin n) → α i.castSucc) → (x : α (Fin.last n)) → P (Fin.snoc xs x)
                    x : (i : Fin n.succ) → α i
                    ⊢ Eq (P (Fin.snoc (Fin.init x) (x (Fin.last n)))) (P x)
                  -/
  _root_.cast (by rw [Fin.snoc_init_self]) <| h (Fin.init x) (x <| Fin.last _)
                  /-
                    🎉 no goals
                  -/


@[simp] lemma snocCases_snoc
    {P : (∀ i : Fin (n+1), α i) → Sort*} (h : ∀ x x₀, P (Fin.snoc x x₀))
    (x : ∀ i : Fin n, (Fin.init α) i) (x₀ : α (Fin.last _)) :
    snocCases h (Fin.snoc x x₀) = h x x₀ := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    P : ((i : Fin (HAdd.hAdd n 1)) → α i) → Sort u_2
    h : (x : (i : Fin n) → α i.castSucc) → (x₀ : α (Fin.last n)) → P (Fin.snoc x x₀)
    x : (i : Fin n) → Fin.init α i
    x₀ : α (Fin.last n)
    ⊢ Eq (Fin.snocCases h (Fin.snoc x x₀)) (h x x₀)
  -/
  rw [snocCases, cast_eq_iff_heq, Fin.init_snoc, Fin.snoc_last]
  /-
    🎉 no goals
  -/


/-- Recurse on a tuple by splitting into `Fin.elim0` and `Fin.snoc`. -/
@[elab_as_elim]
def snocInduction {α : Sort*}
    {P : ∀ {n : ℕ}, (Fin n → α) → Sort*}
    (h0 : P Fin.elim0)
    (h : ∀ {n} (x : Fin n → α) (x₀), P x → P (Fin.snoc x x₀)) : ∀ {n : ℕ} (x : Fin n → α), P x
               /-
                 m n : Nat
                 α✝ : Fin (HAdd.hAdd n 1) → Sort u_1
                 x✝ : α✝ (Fin.last n)
                 q : (i : Fin (HAdd.hAdd n 1)) → α✝ i
                 p : (i : Fin n) → α✝ i.castSucc
                 i : Fin n
                 y : α✝ i.castSucc
                 z : α✝ (Fin.last n)
                 α : Sort u_2
                 P : {n : Nat} → (Fin n → α) → Sort u_3
                 h0 : P Fin.elim0
                 h : {n : Nat} → (x : Fin n → α) → (x₀ : α) → P x → P (Fin.snoc x x₀)
                 x : Fin 0 → α
                 ⊢ P x
               -/
  | 0, x => by convert h0
               /-
                 🎉 no goals
               -/
  | _ + 1, x => snocCases (fun _ _ ↦ h _ _ <| snocInduction h0 h _) x


/-- Define a function on `Fin (n + 1)` from a value on `i : Fin (n + 1)` and values on each
`Fin.succAbove i j`, `j : Fin n`. This version is elaborated as eliminator and works for
propositions, see also `Fin.insertNth` for a version without an `@[elab_as_elim]`
attribute. -/
@[elab_as_elim]
def succAboveCases {α : Fin (n + 1) → Sort u} (i : Fin (n + 1)) (x : α i)
    (p : ∀ j : Fin n, α (i.succAbove j)) (j : Fin (n + 1)) : α j :=
  if hj : j = i then Eq.rec x hj.symm
  else
    if hlt : j < i then @Eq.recOn _ _ (fun x _ ↦ α x) _ (succAbove_castPred_of_lt _ _ hlt) (p _)
    else @Eq.recOn _ _ (fun x _ ↦ α x) _ (succAbove_pred_of_lt _ _ <|
    (Fin.lt_or_lt_of_ne hj).resolve_left hlt) (p _)

-- This is a duplicate of `Fin.exists_fin_succ` in Core. We should upstream the name change.

alias forall_iff_succ := forall_fin_succ

-- This is a duplicate of `Fin.exists_fin_succ` in Core. We should upstream the name change.

alias exists_iff_succ := exists_fin_succ


lemma forall_iff_castSucc {P : Fin (n + 1) → Prop} :
    (∀ i, P i) ↔ P (last n) ∧ ∀ i : Fin n, P i.castSucc :=
  ⟨fun h ↦ ⟨h _, fun _ ↦ h _⟩, fun h ↦ lastCases h.1 h.2⟩


lemma exists_iff_castSucc {P : Fin (n + 1) → Prop} :
    (∃ i, P i) ↔ P (last n) ∨ ∃ i : Fin n, P i.castSucc where
  mp := by
    /-
      n : Nat
      P : Fin (HAdd.hAdd n 1) → Prop
      ⊢ (Exists fun i => P i) → Or (P (Fin.last n)) (Exists fun i => P i.castSucc)
    -/
    rintro ⟨i, hi⟩
    /-
      case intro
      n : Nat
      P : Fin (HAdd.hAdd n 1) → Prop
      i : Fin (HAdd.hAdd n 1)
      hi : P i
      ⊢ Or (P (Fin.last n)) (Exists fun i => P i.castSucc)
    -/
    induction' i using lastCases
      /-
        case intro.last
        n : Nat
        P : Fin (HAdd.hAdd n 1) → Prop
        hi : P (Fin.last n)
        ⊢ Or (P (Fin.last n)) (Exists fun i => P i.castSucc)
      -/
    · exact .inl hi
      /-
        🎉 no goals
      -/
      /-
        case intro.cast
        n : Nat
        P : Fin (HAdd.hAdd n 1) → Prop
        i✝ : Fin n
        hi : P i✝.castSucc
        ⊢ Or (P (Fin.last n)) (Exists fun i => P i.castSucc)
      -/
    · exact .inr ⟨_, hi⟩
      /-
        🎉 no goals
      -/
            /-
              n : Nat
              P : Fin (HAdd.hAdd n 1) → Prop
              ⊢ Or (P (Fin.last n)) (Exists fun i => P i.castSucc) → Exists fun i => P i
            -/
                                     /-
                                       🎉 no goals
                                     -/
  mpr := by rintro (h | ⟨i, hi⟩) <;> exact ⟨_, ‹_›⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem forall_iff_succAbove {P : Fin (n + 1) → Prop} (p : Fin (n + 1)) :
    (∀ i, P i) ↔ P p ∧ ∀ i, P (p.succAbove i) :=
  ⟨fun h ↦ ⟨h _, fun _ ↦ h _⟩, fun h ↦ succAboveCases p h.1 h.2⟩


lemma exists_iff_succAbove {P : Fin (n + 1) → Prop} (p : Fin (n + 1)) :
    (∃ i, P i) ↔ P p ∨ ∃ i, P (p.succAbove i) where
  mp := by
    /-
      n : Nat
      P : Fin (HAdd.hAdd n 1) → Prop
      p : Fin (HAdd.hAdd n 1)
      ⊢ (Exists fun i => P i) → Or (P p) (Exists fun i => P (p.succAbove i))
    -/
    rintro ⟨i, hi⟩
    /-
      case intro
      n : Nat
      P : Fin (HAdd.hAdd n 1) → Prop
      p i : Fin (HAdd.hAdd n 1)
      hi : P i
      ⊢ Or (P p) (Exists fun i => P (p.succAbove i))
    -/
    induction' i using p.succAboveCases
      /-
        case intro.x
        n : Nat
        P : Fin (HAdd.hAdd n 1) → Prop
        p : Fin (HAdd.hAdd n 1)
        hi : P p
        ⊢ Or (P p) (Exists fun i => P (p.succAbove i))
      -/
    · exact .inl hi
      /-
        🎉 no goals
      -/
      /-
        case intro.p
        n : Nat
        P : Fin (HAdd.hAdd n 1) → Prop
        p : Fin (HAdd.hAdd n 1)
        j✝ : Fin n
        hi : P (p.succAbove j✝)
        ⊢ Or (P p) (Exists fun i => P (p.succAbove i))
      -/
    · exact .inr ⟨_, hi⟩
      /-
        🎉 no goals
      -/
            /-
              n : Nat
              P : Fin (HAdd.hAdd n 1) → Prop
              p : Fin (HAdd.hAdd n 1)
              ⊢ Or (P p) (Exists fun i => P (p.succAbove i)) → Exists fun i => P i
            -/
                                     /-
                                       🎉 no goals
                                     -/
  mpr := by rintro (h | ⟨i, hi⟩) <;> exact ⟨_, ‹_›⟩
                                     /-
                                       🎉 no goals
                                     -/


/-- Remove the `p`-th entry of a tuple. -/
def removeNth (p : Fin (n + 1)) (f : ∀ i, α i) : ∀ i, α (p.succAbove i) := fun i ↦ f (p.succAbove i)


/-- Insert an element into a tuple at a given position. For `i = 0` see `Fin.cons`,
for `i = Fin.last n` see `Fin.snoc`. See also `Fin.succAboveCases` for a version elaborated
as an eliminator. -/
def insertNth (i : Fin (n + 1)) (x : α i) (p : ∀ j : Fin n, α (i.succAbove j)) (j : Fin (n + 1)) :
    α j :=
  succAboveCases i x p j


@[simp]
theorem insertNth_apply_same (i : Fin (n + 1)) (x : α i) (p : ∀ j, α (i.succAbove j)) :
                                /-
                                  n : Nat
                                  α : Fin (HAdd.hAdd n 1) → Sort u_1
                                  i : Fin (HAdd.hAdd n 1)
                                  x : α i
                                  p : (j : Fin n) → α (i.succAbove j)
                                  ⊢ Eq (i.insertNth x p i) x
                                -/
    insertNth i x p i = x := by simp [insertNth, succAboveCases]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem insertNth_apply_succAbove (i : Fin (n + 1)) (x : α i) (p : ∀ j, α (i.succAbove j))
    (j : Fin n) : insertNth i x p (i.succAbove j) = p j := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    i : Fin (HAdd.hAdd n 1)
    x : α i
    p : (j : Fin n) → α (i.succAbove j)
    j : Fin n
    ⊢ Eq (i.insertNth x p (i.succAbove j)) (p j)
  -/
  simp only [insertNth, succAboveCases, dif_neg (succAbove_ne _ _), succAbove_lt_iff_castSucc_lt]
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    i : Fin (HAdd.hAdd n 1)
    x : α i
    p : (j : Fin n) → α (i.succAbove j)
    j : Fin n
    ⊢ Eq (dite (LT.lt j.castSucc i) (fun h => Eq.rec (p ((i.succAbove j).castPred  …
  -/
  split_ifs with hlt
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : LT.lt j.castSucc i
      ⊢ Eq (Eq.rec (p ((i.succAbove j).castPred ⋯)) ⋯) (p j)
    -/
  · generalize_proofs H₁ H₂; revert H₂
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : LT.lt j.castSucc i
      H₁ : Ne (i.succAbove j) (Fin.last n)
      ⊢ ∀ (H₂ : Eq (i.succAbove ((i.succAbove j).castPred H₁)) (i.succAbove j)), Eq  …
    -/
    generalize hk : castPred ((succAbove i) j) H₁ = k
    /-
      case pos
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : LT.lt j.castSucc i
      H₁ : Ne (i.succAbove j) (Fin.last n)
      k : Fin n
      hk : Eq ((i.succAbove j).castPred H₁) k
      ⊢ ∀ (H₂ : Eq (i.succAbove k) (i.succAbove j)), Eq (Eq.rec (p k) H₂) (p j)
    -/
    rw [castPred_succAbove _ _ hlt] at hk; cases hk
    /-
      case pos.refl
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : LT.lt j.castSucc i
      H₁ : Ne (i.succAbove j) (Fin.last n)
      ⊢ ∀ (H₂ : Eq (i.succAbove j) (i.succAbove j)), Eq (Eq.rec (p j) H₂) (p j)
    -/
    intro; rfl
           /-
             🎉 no goals
           -/
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : Not (LT.lt j.castSucc i)
      ⊢ Eq (Eq.rec (p ((i.succAbove j).pred ⋯)) ⋯) (p j)
    -/
  · generalize_proofs H₀ H₁ H₂; revert H₂
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : Not (LT.lt j.castSucc i)
      H₀ : NeZero (HAdd.hAdd n 1)
      H₁ : Ne (i.succAbove j) 0
      ⊢ ∀ (H₂ : Eq (i.succAbove ((i.succAbove j).pred H₁)) (i.succAbove j)), Eq (Eq. …
    -/
    generalize hk : pred (succAbove i j) H₁ = k
    /-
      case neg
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : Not (LT.lt j.castSucc i)
      H₀ : NeZero (HAdd.hAdd n 1)
      H₁ : Ne (i.succAbove j) 0
      k : Fin n
      hk : Eq ((i.succAbove j).pred H₁) k
      ⊢ ∀ (H₂ : Eq (i.succAbove k) (i.succAbove j)), Eq (Eq.rec (p k) H₂) (p j)
    -/
    rw [pred_succAbove _ _ (Fin.not_lt.1 hlt)] at hk; cases hk
    /-
      case neg.refl
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      i : Fin (HAdd.hAdd n 1)
      x : α i
      p : (j : Fin n) → α (i.succAbove j)
      j : Fin n
      hlt : Not (LT.lt j.castSucc i)
      H₀ : NeZero (HAdd.hAdd n 1)
      H₁ : Ne (i.succAbove j) 0
      ⊢ ∀ (H₂ : Eq (i.succAbove j) (i.succAbove j)), Eq (Eq.rec (p j) H₂) (p j)
    -/
    intro; rfl
           /-
             🎉 no goals
           -/


@[simp]
theorem succAbove_cases_eq_insertNth : @succAboveCases = @insertNth :=
  rfl


@[simp] lemma removeNth_insertNth (p : Fin (n + 1)) (a : α p) (f : ∀ i, α (succAbove p i)) :
                                            /-
                                              n : Nat
                                              α : Fin (HAdd.hAdd n 1) → Sort u_1
                                              p : Fin (HAdd.hAdd n 1)
                                              a : α p
                                              f : (i : Fin n) → α (p.succAbove i)
                                              ⊢ Eq (p.removeNth (p.insertNth a f)) f
                                            -/
    removeNth p (insertNth p a f) = f := by ext; unfold removeNth; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp] lemma removeNth_zero (f : ∀ i, α i) : removeNth 0 f = tail f := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    f : (i : Fin (HAdd.hAdd n 1)) → α i
    ⊢ Eq (Fin.removeNth 0 f) (Fin.tail f)
  -/
  ext; simp [tail, removeNth]
       /-
         🎉 no goals
       -/


@[simp] lemma removeNth_last {α : Type*} (f : Fin (n + 1) → α) : removeNth (last n) f = init f := by
  /-
    n : Nat
    α : Type u_3
    f : Fin (HAdd.hAdd n 1) → α
    ⊢ Eq ((Fin.last n).removeNth f) (Fin.init f)
  -/
  ext; simp [init, removeNth]
       /-
         🎉 no goals
       -/

/- Porting note: Had to `unfold comp`. Sometimes, when I use a placeholder, if I try to insert
what Lean says it synthesized, it gives me a type error anyway. In this case, it's `x` and `p`. -/

@[simp]
theorem insertNth_comp_succAbove (i : Fin (n + 1)) (x : β) (p : Fin n → β) :
    insertNth i x p ∘ i.succAbove = p :=
             /-
               n : Nat
               β : Sort u_2
               i : Fin (HAdd.hAdd n 1)
               x : β
               p : Fin n → β
               ⊢ ∀ (x_1 : Fin n), Eq (Function.comp (i.insertNth x p) i.succAbove x_1) (p x_1)
             -/
  funext (by unfold comp; exact insertNth_apply_succAbove i _ _)
                          /-
                            🎉 no goals
                          -/


theorem insertNth_eq_iff {p : Fin (n + 1)} {a : α p} {f : ∀ i, α (p.succAbove i)} {g : ∀ j, α j} :
    insertNth p a f = g ↔ a = g p ∧ f = removeNth p g := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    p : Fin (HAdd.hAdd n 1)
    a : α p
    f : (i : Fin n) → α (p.succAbove i)
    g : (j : Fin (HAdd.hAdd n 1)) → α j
    ⊢ Iff (Eq (p.insertNth a f) g) (And (Eq a (g p)) (Eq f (p.removeNth g)))
  -/
  simp [funext_iff, forall_iff_succAbove p, removeNth]
  /-
    🎉 no goals
  -/


theorem eq_insertNth_iff {p : Fin (n + 1)} {a : α p} {f : ∀ i, α (p.succAbove i)} {g : ∀ j, α j} :
    g = insertNth p a f ↔ g p = a ∧ removeNth p g = f := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    p : Fin (HAdd.hAdd n 1)
    a : α p
    f : (i : Fin n) → α (p.succAbove i)
    g : (j : Fin (HAdd.hAdd n 1)) → α j
    ⊢ Iff (Eq g (p.insertNth a f)) (And (Eq (g p) a) (Eq (p.removeNth g) f))
  -/
  simpa [eq_comm] using insertNth_eq_iff
  /-
    🎉 no goals
  -/

/- Porting note: Once again, Lean told me `(fun x x_1 ↦ α x)` was an invalid motive, but disabling
automatic insertion and specifying that motive seems to work. -/

theorem insertNth_apply_below {i j : Fin (n + 1)} (h : j < i) (x : α i)
    (p : ∀ k, α (i.succAbove k)) :
    i.insertNth x p j = @Eq.recOn _ _ (fun x _ ↦ α x) _
    (succAbove_castPred_of_lt _ _ h) (p <| j.castPred _) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    i j : Fin (HAdd.hAdd n 1)
    h : LT.lt j i
    x : α i
    p : (k : Fin n) → α (i.succAbove k)
    ⊢ Eq (i.insertNth x p j) (Eq.recOn ⋯ (p (j.castPred ⋯)))
  -/
  rw [insertNth, succAboveCases, dif_neg (Fin.ne_of_lt h), dif_pos h]
  /-
    🎉 no goals
  -/

/- Porting note: Once again, Lean told me `(fun x x_1 ↦ α x)` was an invalid motive, but disabling
automatic insertion and specifying that motive seems to work. -/

theorem insertNth_apply_above {i j : Fin (n + 1)} (h : i < j) (x : α i)
    (p : ∀ k, α (i.succAbove k)) :
    i.insertNth x p j = @Eq.recOn _ _ (fun x _ ↦ α x) _
    (succAbove_pred_of_lt _ _ h) (p <| j.pred _) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    i j : Fin (HAdd.hAdd n 1)
    h : LT.lt i j
    x : α i
    p : (k : Fin n) → α (i.succAbove k)
    ⊢ Eq (i.insertNth x p j) (Eq.recOn ⋯ (p (j.pred ⋯)))
  -/
  rw [insertNth, succAboveCases, dif_neg (Fin.ne_of_gt h), dif_neg (Fin.lt_asymm h)]
  /-
    🎉 no goals
  -/


theorem insertNth_zero (x : α 0) (p : ∀ j : Fin n, α (succAbove 0 j)) :
    insertNth 0 x p =
      cons x fun j ↦ _root_.cast (congr_arg α (congr_fun succAbove_zero j)) (p j) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α 0
    p : (j : Fin n) → α (Fin.succAbove 0 j)
    ⊢ Eq (Fin.insertNth 0 x p) (Fin.cons x fun j => _root_.cast ⋯ (p j))
  -/
  refine insertNth_eq_iff.2 ⟨by simp, ?_⟩
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α 0
    p : (j : Fin n) → α (Fin.succAbove 0 j)
    ⊢ Eq p (Fin.removeNth 0 (Fin.cons x fun j => _root_.cast ⋯ (p j)))
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α 0
    p : (j : Fin n) → α (Fin.succAbove 0 j)
    j : Fin n
    ⊢ Eq (p j) (Fin.removeNth 0 (Fin.cons x fun j => _root_.cast ⋯ (p j)) j)
  -/
  convert (cons_succ x p j).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem insertNth_zero' (x : β) (p : Fin n → β) : @insertNth _ (fun _ ↦ β) 0 x p = cons x p := by
  /-
    n : Nat
    β : Sort u_2
    x : β
    p : Fin n → β
    ⊢ Eq (Fin.insertNth 0 x p) (Fin.cons x p)
  -/
  simp [insertNth_zero]
  /-
    🎉 no goals
  -/


theorem insertNth_last (x : α (last n)) (p : ∀ j : Fin n, α ((last n).succAbove j)) :
    insertNth (last n) x p =
      snoc (fun j ↦ _root_.cast (congr_arg α (succAbove_last_apply j)) (p j)) x := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (j : Fin n) → α ((Fin.last n).succAbove j)
    ⊢ Eq ((Fin.last n).insertNth x p) (Fin.snoc (fun j => _root_.cast ⋯ (p j)) x)
  -/
  refine insertNth_eq_iff.2 ⟨by simp, ?_⟩
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (j : Fin n) → α ((Fin.last n).succAbove j)
    ⊢ Eq p ((Fin.last n).removeNth (Fin.snoc (fun j => _root_.cast ⋯ (p j)) x))
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (j : Fin n) → α ((Fin.last n).succAbove j)
    j : Fin n
    ⊢ Eq (p j) ((Fin.last n).removeNth (Fin.snoc (fun j => _root_.cast ⋯ (p j)) x) …
  -/
  apply eq_of_heq
  /-
    case h.h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    x : α (Fin.last n)
    p : (j : Fin n) → α ((Fin.last n).succAbove j)
    j : Fin n
    ⊢ HEq (p j) ((Fin.last n).removeNth (Fin.snoc (fun j => _root_.cast ⋯ (p j)) x …
  -/
  trans snoc (fun j ↦ _root_.cast (congr_arg α (succAbove_last_apply j)) (p j)) x j.castSucc
    /-
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (j : Fin n) → α ((Fin.last n).succAbove j)
      j : Fin n
      ⊢ HEq (p j) (Fin.snoc (fun j => _root_.cast ⋯ (p j)) x j.castSucc)
    -/
  · rw [snoc_castSucc]
    /-
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (j : Fin n) → α ((Fin.last n).succAbove j)
      j : Fin n
      ⊢ HEq (p j) (_root_.cast ⋯ (p j))
    -/
    exact (cast_heq _ _).symm
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (j : Fin n) → α ((Fin.last n).succAbove j)
      j : Fin n
      ⊢ HEq (Fin.snoc (fun j => _root_.cast ⋯ (p j)) x j.castSucc) ((Fin.last n).rem …
    -/
  · apply congr_arg_heq
    /-
      case a
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Sort u_1
      x : α (Fin.last n)
      p : (j : Fin n) → α ((Fin.last n).succAbove j)
      j : Fin n
      ⊢ Eq j.castSucc ((Fin.last n).succAbove j)
    -/
    rw [succAbove_last]
    /-
      🎉 no goals
    -/


@[simp]
theorem insertNth_last' (x : β) (p : Fin n → β) :
                                                           /-
                                                             n : Nat
                                                             β : Sort u_2
                                                             x : β
                                                             p : Fin n → β
                                                             ⊢ Eq ((Fin.last n).insertNth x p) (Fin.snoc p x)
                                                           -/
    @insertNth _ (fun _ ↦ β) (last n) x p = snoc p x := by simp [insertNth_last]
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma insertNth_rev {α : Sort*} (i : Fin (n + 1)) (a : α) (f : Fin n → α) (j : Fin (n + 1)) :
    insertNth (α := fun _ ↦ α) i a f (rev j) = insertNth (α := fun _ ↦ α) i.rev a (f ∘ rev) j := by
  /-
    n : Nat
    α : Sort u_3
    i : Fin (HAdd.hAdd n 1)
    a : α
    f : Fin n → α
    j : Fin (HAdd.hAdd n 1)
    ⊢ Eq (i.insertNth a f j.rev) (i.rev.insertNth a (Function.comp f Fin.rev) j)
  -/
  induction j using Fin.succAboveCases
    /-
      case i
      n : Nat
      α : Sort u_3
      i : Fin (HAdd.hAdd n 1)
      a : α
      f : Fin n → α
      ⊢ Fin (HAdd.hAdd n 1)
    -/
  · exact rev i
    /-
      🎉 no goals
    -/
    /-
      case x
      n : Nat
      α : Sort u_3
      i : Fin (HAdd.hAdd n 1)
      a : α
      f : Fin n → α
      ⊢ Eq (i.insertNth a f i.rev.rev) (i.rev.insertNth a (Function.comp f Fin.rev)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case p
      n : Nat
      α : Sort u_3
      i : Fin (HAdd.hAdd n 1)
      a : α
      f : Fin n → α
      j✝ : Fin n
      ⊢ Eq (i.insertNth a f (i.rev.succAbove j✝).rev) (i.rev.insertNth a (Function.c …
    -/
  · simp [rev_succAbove]
    /-
      🎉 no goals
    -/


theorem insertNth_comp_rev {α} (i : Fin (n + 1)) (x : α) (p : Fin n → α) :
    (Fin.insertNth i x p) ∘ Fin.rev = Fin.insertNth (Fin.rev i) x (p ∘ Fin.rev) := by
  /-
    n : Nat
    α : Sort u_3
    i : Fin (HAdd.hAdd n 1)
    x : α
    p : Fin n → α
    ⊢ Eq (Function.comp (i.insertNth x p) Fin.rev) (i.rev.insertNth x (Function.co …
  -/
  funext x
  /-
    case h
    n : Nat
    α : Sort u_3
    i : Fin (HAdd.hAdd n 1)
    x✝ : α
    p : Fin n → α
    x : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.comp (i.insertNth x✝ p) Fin.rev x) (i.rev.insertNth x✝ (Functio …
  -/
  apply insertNth_rev
  /-
    🎉 no goals
  -/


theorem cons_rev {α n} (a : α) (f : Fin n → α) (i : Fin <| n + 1) :
    cons (α := fun _ => α) a f i.rev = snoc (α := fun _ => α) (f ∘ Fin.rev : Fin _ → α) a i := by
  /-
    α : Sort u_3
    n : Nat
    a : α
    f : Fin n → α
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.cons a f i.rev) (Fin.snoc (Function.comp f Fin.rev) a i)
  -/
  simpa using insertNth_rev 0 a f i
  /-
    🎉 no goals
  -/


theorem cons_comp_rev {α n} (a : α) (f : Fin n → α) :
    Fin.cons a f ∘ Fin.rev = Fin.snoc (f ∘ Fin.rev) a := by
  /-
    α : Sort u_3
    n : Nat
    a : α
    f : Fin n → α
    ⊢ Eq (Function.comp (Fin.cons a f) Fin.rev) (Fin.snoc (Function.comp f Fin.rev …
  -/
  funext i; exact cons_rev ..
            /-
              🎉 no goals
            -/


theorem snoc_rev {α n} (a : α) (f : Fin n → α) (i : Fin <| n + 1) :
    snoc (α := fun _ => α) f a i.rev = cons (α := fun _ => α) a (f ∘ Fin.rev : Fin _ → α) i := by
  /-
    α : Sort u_3
    n : Nat
    a : α
    f : Fin n → α
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Fin.snoc f a i.rev) (Fin.cons a (Function.comp f Fin.rev) i)
  -/
  simpa using insertNth_rev (last n) a f i
  /-
    🎉 no goals
  -/


theorem snoc_comp_rev {α n} (a : α) (f : Fin n → α) :
    Fin.snoc f a ∘ Fin.rev = Fin.cons a (f ∘ Fin.rev) :=
  funext <| snoc_rev a f


theorem insertNth_binop (op : ∀ j, α j → α j → α j) (i : Fin (n + 1)) (x y : α i)
    (p q : ∀ j, α (i.succAbove j)) :
    (i.insertNth (op i x y) fun j ↦ op _ (p j) (q j)) = fun j ↦
      op j (i.insertNth x p j) (i.insertNth y q j) :=
                           /-
                             n : Nat
                             α : Fin (HAdd.hAdd n 1) → Sort u_1
                             op : (j : Fin (HAdd.hAdd n 1)) → α j → α j → α j
                             i : Fin (HAdd.hAdd n 1)
                             x y : α i
                             p q : (j : Fin n) → α (i.succAbove j)
                             ⊢ And (Eq (op i x y) (op i (i.insertNth x p i) (i.insertNth y q i))) (Eq (fun  …
                           -/
  insertNth_eq_iff.2 <| by unfold removeNth; simp
                                             /-
                                               🎉 no goals
                                             -/


theorem insertNth_le_iff {i : Fin (n + 1)} {x : α i} {p : ∀ j, α (i.succAbove j)} {q : ∀ j, α j} :
    i.insertNth x p ≤ q ↔ x ≤ q i ∧ p ≤ fun j ↦ q (i.succAbove j) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_3
    inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
    i : Fin (HAdd.hAdd n 1)
    x : α i
    p : (j : Fin n) → α (i.succAbove j)
    q : (j : Fin (HAdd.hAdd n 1)) → α j
    ⊢ Iff (LE.le (i.insertNth x p) q) (And (LE.le x (q i)) (LE.le p fun j => q (i. …
  -/
  simp [Pi.le_def, forall_iff_succAbove i]
  /-
    🎉 no goals
  -/


theorem le_insertNth_iff {i : Fin (n + 1)} {x : α i} {p : ∀ j, α (i.succAbove j)} {q : ∀ j, α j} :
    q ≤ i.insertNth x p ↔ q i ≤ x ∧ (fun j ↦ q (i.succAbove j)) ≤ p := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_3
    inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
    i : Fin (HAdd.hAdd n 1)
    x : α i
    p : (j : Fin n) → α (i.succAbove j)
    q : (j : Fin (HAdd.hAdd n 1)) → α j
    ⊢ Iff (LE.le q (i.insertNth x p)) (And (LE.le (q i) x) (LE.le (fun j => q (i.s …
  -/
  simp [Pi.le_def, forall_iff_succAbove i]
  /-
    🎉 no goals
  -/


@[simp] lemma removeNth_update (p : Fin (n + 1)) (x) (f : ∀ j, α j) :
                                                     /-
                                                       n : Nat
                                                       α : Fin (HAdd.hAdd n 1) → Sort u_1
                                                       p : Fin (HAdd.hAdd n 1)
                                                       x : α p
                                                       f : (j : Fin (HAdd.hAdd n 1)) → α j
                                                       ⊢ Eq (p.removeNth (Function.update f p x)) (p.removeNth f)
                                                     -/
    removeNth p (update f p x) = removeNth p f := by ext i; simp [removeNth, succAbove_ne]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp] lemma insertNth_removeNth (p : Fin (n + 1)) (x) (f : ∀ j, α j) :
                                                       /-
                                                         n : Nat
                                                         α : Fin (HAdd.hAdd n 1) → Sort u_1
                                                         p : Fin (HAdd.hAdd n 1)
                                                         x : α p
                                                         f : (j : Fin (HAdd.hAdd n 1)) → α j
                                                         ⊢ Eq (p.insertNth x (p.removeNth f)) (Function.update f p x)
                                                       -/
    insertNth p x (removeNth p f) = update f p x := by simp [Fin.insertNth_eq_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma insertNth_self_removeNth (p : Fin (n + 1)) (f : ∀ j, α j) :
                                                /-
                                                  n : Nat
                                                  α : Fin (HAdd.hAdd n 1) → Sort u_1
                                                  p : Fin (HAdd.hAdd n 1)
                                                  f : (j : Fin (HAdd.hAdd n 1)) → α j
                                                  ⊢ Eq (p.insertNth (f p) (p.removeNth f)) f
                                                -/
    insertNth p (f p) (removeNth p f) = f := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem update_insertNth (p : Fin (n + 1)) (x y : α p) (f : ∀ i, α (p.succAbove i)) :
    update (p.insertNth x f) p y = p.insertNth y f := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    p : Fin (HAdd.hAdd n 1)
    x y : α p
    f : (i : Fin n) → α (p.succAbove i)
    ⊢ Eq (Function.update (p.insertNth x f) p y) (p.insertNth y f)
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_1
    p : Fin (HAdd.hAdd n 1)
    x y : α p
    f : (i : Fin n) → α (p.succAbove i)
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.update (p.insertNth x f) p y i) (p.insertNth y f i)
  -/
                                     /-
                                       🎉 no goals
                                     -/
  cases i using p.succAboveCases <;> simp [succAbove_ne]
                                     /-
                                       🎉 no goals
                                     -/


/-- Equivalence between tuples of length `n + 1` and pairs of an element and a tuple of length `n`
given by separating out the `p`-th element of the tuple.

This is `Fin.insertNth` as an `Equiv`. -/
@[simps]
def insertNthEquiv (α : Fin (n + 1) → Type u) (p : Fin (n + 1)) :
    α p × (∀ i, α (p.succAbove i)) ≃ ∀ i, α i where
  toFun f := insertNth p f.1 f.2
  invFun f := (f p, removeNth p f)
                   /-
                     m n : Nat
                     α✝ : Fin (HAdd.hAdd n 1) → Sort u_1
                     β : Sort u_2
                     α : Fin (HAdd.hAdd n 1) → Type u
                     p : Fin (HAdd.hAdd n 1)
                     f : Prod (α p) ((i : Fin n) → α (p.succAbove i))
                     ⊢ Eq ((fun f => { fst := f p, snd := p.removeNth f }) ((fun f => p.insertNth f …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv f := by ext <;> simp
                           /-
                             🎉 no goals
                           -/
                    /-
                      m n : Nat
                      α✝ : Fin (HAdd.hAdd n 1) → Sort u_1
                      β : Sort u_2
                      α : Fin (HAdd.hAdd n 1) → Type u
                      p : Fin (HAdd.hAdd n 1)
                      f : (i : Fin (HAdd.hAdd n 1)) → α i
                      ⊢ Eq ((fun f => p.insertNth f.1 f.2) ((fun f => { fst := f p, snd := p.removeN …
                    -/
  right_inv f := by simp
                    /-
                      🎉 no goals
                    -/


@[simp] lemma insertNthEquiv_zero (α : Fin (n + 1) → Type*) : insertNthEquiv α 0 = consEquiv α :=
                                       /-
                                         n : Nat
                                         α : Fin (HAdd.hAdd n 1) → Type u_3
                                         ⊢ Eq (Fin.insertNthEquiv α 0).symm (Fin.consEquiv α).symm
                                       -/
                                               /-
                                                 🎉 no goals
                                               -/
  Equiv.symm_bijective.injective <| by ext <;> rfl
                                               /-
                                                 🎉 no goals
                                               -/


/-- Note this lemma can only be written about non-dependent tuples as `insertNth (last n) = snoc` is
not a definitional equality. -/
@[simp] lemma insertNthEquiv_last (n : ℕ) (α : Type*) :
                                                                      /-
                                                                        n : Nat
                                                                        α : Type u_3
                                                                        ⊢ Eq (Fin.insertNthEquiv (fun x => α) (Fin.last n)) (Fin.snocEquiv fun x => α)
                                                                      -/
    insertNthEquiv (fun _ ↦ α) (last n) = snocEquiv (fun _ ↦ α) := by ext; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Separates an `n+1`-tuple, returning a selected index and then the rest of the tuple.
Functional form of `Equiv.piFinSuccAbove`. -/
@[deprecated removeNth (since := "2024-06-19")]
def extractNth {α : Fin (n + 1) → Type*} (i : Fin (n + 1)) (f : (∀ j, α j)) :
    α i × ∀ j, α (i.succAbove j) :=
  (f i, removeNth i f)


/-- `find p` returns the first index `n` where `p n` is satisfied, and `none` if it is never
satisfied. -/
def find : ∀ {n : ℕ} (p : Fin n → Prop) [DecidablePred p], Option (Fin n)
  | 0, _p, _ => none
  | n + 1, p, _ => by
    exact
      Option.casesOn (@find n (fun i ↦ p (i.castLT (Nat.lt_succ_of_lt i.2))) _)
        (if _ : p (Fin.last n) then some (Fin.last n) else none) fun i ↦
        some (i.castLT (Nat.lt_succ_of_lt i.2))


/-- If `find p = some i`, then `p i` holds -/
theorem find_spec :
    ∀ {n : ℕ} (p : Fin n → Prop) [DecidablePred p] {i : Fin n} (_ : i ∈ Fin.find p), p i
  | 0, _, _, _, hi => Option.noConfusion hi
  | n + 1, p, I, i, hi => by
    /-
      n : Nat
      p : Fin (HAdd.hAdd n 1) → Prop
      I : DecidablePred p
      i : Fin (HAdd.hAdd n 1)
      hi : Membership.mem (Fin.find p) i
      ⊢ p i
    -/
    rw [find] at hi
    /-
      n : Nat
      p : Fin (HAdd.hAdd n 1) → Prop
      I : DecidablePred p
      i : Fin (HAdd.hAdd n 1)
      hi : Membership.mem (Option.casesOn (Fin.find fun i => p (i.castLT ⋯)) (dite ( …
      ⊢ p i
    -/
    cases' h : find fun i : Fin n ↦ p (i.castLT (Nat.lt_succ_of_lt i.2)) with j
      /-
        case none
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Membership.mem (Option.casesOn (Fin.find fun i => p (i.castLT ⋯)) (dite ( …
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
        ⊢ p i
      -/
    · rw [h] at hi
      /-
        case none
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Membership.mem (Option.casesOn Option.none (dite (p (Fin.last n)) (fun x  …
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
        ⊢ p i
      -/
      dsimp at hi
      /-
        case none
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Membership.mem (ite (p (Fin.last n)) (Option.some (Fin.last n)) Option.no …
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
        ⊢ p i
      -/
      split_ifs at hi with hl
        /-
          case pos
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          I : DecidablePred p
          i : Fin (HAdd.hAdd n 1)
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          hl : p (Fin.last n)
          hi : Membership.mem (Option.some (Fin.last n)) i
          ⊢ p i
        -/
      · simp only [Option.mem_def, Option.some.injEq] at hi
        /-
          case pos
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          I : DecidablePred p
          i : Fin (HAdd.hAdd n 1)
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          hl : p (Fin.last n)
          hi : Eq (Fin.last n) i
          ⊢ p i
        -/
        exact hi ▸ hl
        /-
          🎉 no goals
        -/
        /-
          case neg
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          I : DecidablePred p
          i : Fin (HAdd.hAdd n 1)
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          hl : Not (p (Fin.last n))
          hi : Membership.mem Option.none i
          ⊢ p i
        -/
      · exact (Option.not_mem_none _ hi).elim
        /-
          🎉 no goals
        -/
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Membership.mem (Option.casesOn (Fin.find fun i => p (i.castLT ⋯)) (dite ( …
        j : Fin n
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some j)
        ⊢ p i
      -/
    · rw [h] at hi
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        j : Fin n
        hi : Membership.mem (Option.casesOn (Option.some j) (dite (p (Fin.last n)) (fu …
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some j)
        ⊢ p i
      -/
      dsimp at hi
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        j : Fin n
        hi : Membership.mem (Option.some (j.castLT ⋯)) i
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some j)
        ⊢ p i
      -/
      rw [← Option.some_inj.1 hi]
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        I : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        j : Fin n
        hi : Membership.mem (Option.some (j.castLT ⋯)) i
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some j)
        ⊢ p (j.castLT ⋯)
      -/
      exact @find_spec n (fun i ↦ p (i.castLT (Nat.lt_succ_of_lt i.2))) _ _ h
      /-
        🎉 no goals
      -/


/-- `find p` does not return `none` if and only if `p i` holds at some index `i`. -/
theorem isSome_find_iff :
    ∀ {n : ℕ} {p : Fin n → Prop} [DecidablePred p], (find p).isSome ↔ ∃ i, p i
  | 0, _, _ => iff_of_false (fun h ↦ Bool.noConfusion h) fun ⟨i, _⟩ ↦ Fin.elim0 i
  | n + 1, p, _ =>
    ⟨fun h ↦ by
      /-
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        h : Eq (Fin.find p).isSome Bool.true
        ⊢ Exists fun i => p i
      -/
      rw [Option.isSome_iff_exists] at h
      /-
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        h : Exists fun a => Eq (Fin.find p) (Option.some a)
        ⊢ Exists fun i => p i
      -/
      cases' h with i hi
      /-
        case intro
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Eq (Fin.find p) (Option.some i)
        ⊢ Exists fun i => p i
      -/
      exact ⟨i, find_spec _ hi⟩, fun ⟨⟨i, hin⟩, hi⟩ ↦ by
      /-
        🎉 no goals
      -/
      /-
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝¹ : DecidablePred p
        x✝ : Exists fun i => p i
        i : Nat
        hin : LT.lt i (HAdd.hAdd n 1)
        hi : p ⟨i, hin⟩
        ⊢ Eq (Fin.find p).isSome Bool.true
      -/
      dsimp [find]
      /-
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝¹ : DecidablePred p
        x✝ : Exists fun i => p i
        i : Nat
        hin : LT.lt i (HAdd.hAdd n 1)
        hi : p ⟨i, hin⟩
        ⊢ Eq (Option.rec (ite (p (Fin.last n)) (Option.some (Fin.last n)) Option.none) …
      -/
      cases' h : find fun i : Fin n ↦ p (i.castLT (Nat.lt_succ_of_lt i.2)) with j
        /-
          case none
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          x✝¹ : DecidablePred p
          x✝ : Exists fun i => p i
          i : Nat
          hin : LT.lt i (HAdd.hAdd n 1)
          hi : p ⟨i, hin⟩
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          ⊢ Eq (Option.rec (ite (p (Fin.last n)) (Option.some (Fin.last n)) Option.none) …
        -/
      · split_ifs with hl
          /-
            case pos
            n : Nat
            p : Fin (HAdd.hAdd n 1) → Prop
            x✝¹ : DecidablePred p
            x✝ : Exists fun i => p i
            i : Nat
            hin : LT.lt i (HAdd.hAdd n 1)
            hi : p ⟨i, hin⟩
            h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
            hl : p (Fin.last n)
            ⊢ Eq (Option.rec (Option.some (Fin.last n)) (fun val => Option.some (val.castL …
          -/
        · exact Option.isSome_some
          /-
            🎉 no goals
          -/
        · have := (@isSome_find_iff n (fun x ↦ p (x.castLT (Nat.lt_succ_of_lt x.2))) _).2
              ⟨⟨i, lt_of_le_of_ne (Nat.le_of_lt_succ hin) fun h ↦ by cases h; exact hl hi⟩, hi⟩
          /-
            case neg
            n : Nat
            p : Fin (HAdd.hAdd n 1) → Prop
            x✝¹ : DecidablePred p
            x✝ : Exists fun i => p i
            i : Nat
            hin : LT.lt i (HAdd.hAdd n 1)
            hi : p ⟨i, hin⟩
            h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
            hl : Not (p (Fin.last n))
            this : Eq (Fin.find fun x => p (x.castLT ⋯)).isSome Bool.true
            ⊢ Eq (Option.rec Option.none (fun val => Option.some (val.castLT ⋯)) Option.no …
          -/
          rw [h] at this
          /-
            case neg
            n : Nat
            p : Fin (HAdd.hAdd n 1) → Prop
            x✝¹ : DecidablePred p
            x✝ : Exists fun i => p i
            i : Nat
            hin : LT.lt i (HAdd.hAdd n 1)
            hi : p ⟨i, hin⟩
            h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
            hl : Not (p (Fin.last n))
            this : Eq Option.none.isSome Bool.true
            ⊢ Eq (Option.rec Option.none (fun val => Option.some (val.castLT ⋯)) Option.no …
          -/
          exact this
          /-
            🎉 no goals
          -/
        /-
          case some
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          x✝¹ : DecidablePred p
          x✝ : Exists fun i => p i
          i : Nat
          hin : LT.lt i (HAdd.hAdd n 1)
          hi : p ⟨i, hin⟩
          j : Fin n
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some j)
          ⊢ Eq (Option.rec (ite (p (Fin.last n)) (Option.some (Fin.last n)) Option.none) …
        -/
      · simp⟩
        /-
          🎉 no goals
        -/


/-- `find p` returns `none` if and only if `p i` never holds. -/
theorem find_eq_none_iff {n : ℕ} {p : Fin n → Prop} [DecidablePred p] :
                                    /-
                                      n : Nat
                                      p : Fin n → Prop
                                      inst✝ : DecidablePred p
                                      ⊢ Iff (Eq (Fin.find p) Option.none) (∀ (i : Fin n), Not (p i))
                                    -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
    find p = none ↔ ∀ i, ¬p i := by rw [← not_exists, ← isSome_find_iff]; cases find p <;> simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


/-- If `find p` returns `some i`, then `p j` does not hold for `j < i`, i.e., `i` is minimal among
the indices where `p` holds. -/
theorem find_min :
    ∀ {n : ℕ} {p : Fin n → Prop} [DecidablePred p] {i : Fin n} (_ : i ∈ Fin.find p) {j : Fin n}
      (_ : j < i), ¬p j
  | 0, _, _, _, hi, _, _, _ => Option.noConfusion hi
  | n + 1, p, _, i, hi, ⟨j, hjn⟩, hj, hpj => by
    /-
      n : Nat
      p : Fin (HAdd.hAdd n 1) → Prop
      x✝ : DecidablePred p
      i : Fin (HAdd.hAdd n 1)
      hi : Membership.mem (Fin.find p) i
      j : Nat
      hjn : LT.lt j (HAdd.hAdd n 1)
      hj : LT.lt ⟨j, hjn⟩ i
      hpj : p ⟨j, hjn⟩
      ⊢ False
    -/
    rw [find] at hi
    /-
      n : Nat
      p : Fin (HAdd.hAdd n 1) → Prop
      x✝ : DecidablePred p
      i : Fin (HAdd.hAdd n 1)
      hi : Membership.mem (Option.casesOn (Fin.find fun i => p (i.castLT ⋯)) (dite ( …
      j : Nat
      hjn : LT.lt j (HAdd.hAdd n 1)
      hj : LT.lt ⟨j, hjn⟩ i
      hpj : p ⟨j, hjn⟩
      ⊢ False
    -/
    cases' h : find fun i : Fin n ↦ p (i.castLT (Nat.lt_succ_of_lt i.2)) with k
      /-
        case none
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Membership.mem (Option.casesOn (Fin.find fun i => p (i.castLT ⋯)) (dite ( …
        j : Nat
        hjn : LT.lt j (HAdd.hAdd n 1)
        hj : LT.lt ⟨j, hjn⟩ i
        hpj : p ⟨j, hjn⟩
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
        ⊢ False
      -/
    · simp only [h] at hi
      /-
        case none
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        j : Nat
        hjn : LT.lt j (HAdd.hAdd n 1)
        hj : LT.lt ⟨j, hjn⟩ i
        hpj : p ⟨j, hjn⟩
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
        hi : Membership.mem (dite (p (Fin.last n)) (fun x => Option.some (Fin.last n)) …
        ⊢ False
      -/
      split_ifs at hi with hl
        /-
          case pos
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          x✝ : DecidablePred p
          i : Fin (HAdd.hAdd n 1)
          j : Nat
          hjn : LT.lt j (HAdd.hAdd n 1)
          hj : LT.lt ⟨j, hjn⟩ i
          hpj : p ⟨j, hjn⟩
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          hl : p (Fin.last n)
          hi : Membership.mem (Option.some (Fin.last n)) i
          ⊢ False
        -/
      · cases hi
        /-
          case pos.refl
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          x✝ : DecidablePred p
          j : Nat
          hjn : LT.lt j (HAdd.hAdd n 1)
          hpj : p ⟨j, hjn⟩
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          hl : p (Fin.last n)
          hj : LT.lt ⟨j, hjn⟩ (Fin.last n)
          ⊢ False
        -/
        rw [find_eq_none_iff] at h
        /-
          case pos.refl
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          x✝ : DecidablePred p
          j : Nat
          hjn : LT.lt j (HAdd.hAdd n 1)
          hpj : p ⟨j, hjn⟩
          h : ∀ (i : Fin n), Not (p (i.castLT ⋯))
          hl : p (Fin.last n)
          hj : LT.lt ⟨j, hjn⟩ (Fin.last n)
          ⊢ False
        -/
        exact h ⟨j, hj⟩ hpj
        /-
          🎉 no goals
        -/
        /-
          case neg
          n : Nat
          p : Fin (HAdd.hAdd n 1) → Prop
          x✝ : DecidablePred p
          i : Fin (HAdd.hAdd n 1)
          j : Nat
          hjn : LT.lt j (HAdd.hAdd n 1)
          hj : LT.lt ⟨j, hjn⟩ i
          hpj : p ⟨j, hjn⟩
          h : Eq (Fin.find fun i => p (i.castLT ⋯)) Option.none
          hl : Not (p (Fin.last n))
          hi : Membership.mem Option.none i
          ⊢ False
        -/
      · exact Option.not_mem_none _ hi
        /-
          🎉 no goals
        -/
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        hi : Membership.mem (Option.casesOn (Fin.find fun i => p (i.castLT ⋯)) (dite ( …
        j : Nat
        hjn : LT.lt j (HAdd.hAdd n 1)
        hj : LT.lt ⟨j, hjn⟩ i
        hpj : p ⟨j, hjn⟩
        k : Fin n
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some k)
        ⊢ False
      -/
    · rw [h] at hi
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        j : Nat
        hjn : LT.lt j (HAdd.hAdd n 1)
        hj : LT.lt ⟨j, hjn⟩ i
        hpj : p ⟨j, hjn⟩
        k : Fin n
        hi : Membership.mem (Option.casesOn (Option.some k) (dite (p (Fin.last n)) (fu …
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some k)
        ⊢ False
      -/
      dsimp at hi
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        i : Fin (HAdd.hAdd n 1)
        j : Nat
        hjn : LT.lt j (HAdd.hAdd n 1)
        hj : LT.lt ⟨j, hjn⟩ i
        hpj : p ⟨j, hjn⟩
        k : Fin n
        hi : Membership.mem (Option.some (k.castLT ⋯)) i
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some k)
        ⊢ False
      -/
      obtain rfl := Option.some_inj.1 hi
      /-
        case some
        n : Nat
        p : Fin (HAdd.hAdd n 1) → Prop
        x✝ : DecidablePred p
        j : Nat
        hjn : LT.lt j (HAdd.hAdd n 1)
        hpj : p ⟨j, hjn⟩
        k : Fin n
        h : Eq (Fin.find fun i => p (i.castLT ⋯)) (Option.some k)
        hj : LT.lt ⟨j, hjn⟩ (k.castLT ⋯)
        hi : Membership.mem (Option.some (k.castLT ⋯)) (k.castLT ⋯)
        ⊢ False
      -/
      exact find_min h (show (⟨j, lt_trans hj k.2⟩ : Fin n) < k from hj) hpj
      /-
        🎉 no goals
      -/


theorem find_min' {p : Fin n → Prop} [DecidablePred p] {i : Fin n} (h : i ∈ Fin.find p) {j : Fin n}
    (hj : p j) : i ≤ j := Fin.not_lt.1 fun hij ↦ find_min h hij hj


theorem nat_find_mem_find {p : Fin n → Prop} [DecidablePred p]
    (h : ∃ i, ∃ hin : i < n, p ⟨i, hin⟩) :
    (⟨Nat.find h, (Nat.find_spec h).fst⟩ : Fin n) ∈ find p := by
  /-
    n : Nat
    p : Fin n → Prop
    inst✝ : DecidablePred p
    h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
    ⊢ Membership.mem (Fin.find p) ⟨Nat.find h, ⋯⟩
  -/
  let ⟨i, hin, hi⟩ := h
  /-
    n : Nat
    p : Fin n → Prop
    inst✝ : DecidablePred p
    h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
    i : Nat
    hin : LT.lt i n
    hi : p ⟨i, hin⟩
    ⊢ Membership.mem (Fin.find p) ⟨Nat.find ⋯, ⋯⟩
  -/
  cases' hf : find p with f
    /-
      case none
      n : Nat
      p : Fin n → Prop
      inst✝ : DecidablePred p
      h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
      i : Nat
      hin : LT.lt i n
      hi : p ⟨i, hin⟩
      hf : Eq (Fin.find p) Option.none
      ⊢ Membership.mem Option.none ⟨Nat.find ⋯, ⋯⟩
    -/
  · rw [find_eq_none_iff] at hf
    /-
      case none
      n : Nat
      p : Fin n → Prop
      inst✝ : DecidablePred p
      h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
      i : Nat
      hin : LT.lt i n
      hi : p ⟨i, hin⟩
      hf : ∀ (i : Fin n), Not (p i)
      ⊢ Membership.mem Option.none ⟨Nat.find ⋯, ⋯⟩
    -/
    exact (hf ⟨i, hin⟩ hi).elim
    /-
      🎉 no goals
    -/
    /-
      case some
      n : Nat
      p : Fin n → Prop
      inst✝ : DecidablePred p
      h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
      i : Nat
      hin : LT.lt i n
      hi : p ⟨i, hin⟩
      f : Fin n
      hf : Eq (Fin.find p) (Option.some f)
      ⊢ Membership.mem (Option.some f) ⟨Nat.find ⋯, ⋯⟩
    -/
  · refine Option.some_inj.2 (Fin.le_antisymm ?_ ?_)
      /-
        case some.refine_1
        n : Nat
        p : Fin n → Prop
        inst✝ : DecidablePred p
        h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
        i : Nat
        hin : LT.lt i n
        hi : p ⟨i, hin⟩
        f : Fin n
        hf : Eq (Fin.find p) (Option.some f)
        ⊢ LE.le f ⟨Nat.find ⋯, ⋯⟩
      -/
    · exact find_min' hf (Nat.find_spec h).snd
      /-
        🎉 no goals
      -/
      /-
        case some.refine_2
        n : Nat
        p : Fin n → Prop
        inst✝ : DecidablePred p
        h : Exists fun i => Exists fun hin => p ⟨i, hin⟩
        i : Nat
        hin : LT.lt i n
        hi : p ⟨i, hin⟩
        f : Fin n
        hf : Eq (Fin.find p) (Option.some f)
        ⊢ LE.le ⟨Nat.find ⋯, ⋯⟩ f
      -/
    · exact Nat.find_min' _ ⟨f.2, by convert find_spec p hf⟩
      /-
        🎉 no goals
      -/


theorem mem_find_iff {p : Fin n → Prop} [DecidablePred p] {i : Fin n} :
    i ∈ Fin.find p ↔ p i ∧ ∀ j, p j → i ≤ j :=
  ⟨fun hi ↦ ⟨find_spec _ hi, fun _ ↦ find_min' hi⟩, by
    /-
      n : Nat
      p : Fin n → Prop
      inst✝ : DecidablePred p
      i : Fin n
      ⊢ And (p i) (∀ (j : Fin n), p j → LE.le i j) → Membership.mem (Fin.find p) i
    -/
    rintro ⟨hpi, hj⟩
    /-
      case intro
      n : Nat
      p : Fin n → Prop
      inst✝ : DecidablePred p
      i : Fin n
      hpi : p i
      hj : ∀ (j : Fin n), p j → LE.le i j
      ⊢ Membership.mem (Fin.find p) i
    -/
    cases hfp : Fin.find p
      /-
        case intro.none
        n : Nat
        p : Fin n → Prop
        inst✝ : DecidablePred p
        i : Fin n
        hpi : p i
        hj : ∀ (j : Fin n), p j → LE.le i j
        hfp : Eq (Fin.find p) Option.none
        ⊢ Membership.mem Option.none i
      -/
    · rw [find_eq_none_iff] at hfp
      /-
        case intro.none
        n : Nat
        p : Fin n → Prop
        inst✝ : DecidablePred p
        i : Fin n
        hpi : p i
        hj : ∀ (j : Fin n), p j → LE.le i j
        hfp : ∀ (i : Fin n), Not (p i)
        ⊢ Membership.mem Option.none i
      -/
      exact (hfp _ hpi).elim
      /-
        🎉 no goals
      -/
      /-
        case intro.some
        n : Nat
        p : Fin n → Prop
        inst✝ : DecidablePred p
        i : Fin n
        hpi : p i
        hj : ∀ (j : Fin n), p j → LE.le i j
        val✝ : Fin n
        hfp : Eq (Fin.find p) (Option.some val✝)
        ⊢ Membership.mem (Option.some val✝) i
      -/
    · exact Option.some_inj.2 (Fin.le_antisymm (find_min' hfp hpi) (hj _ (find_spec _ hfp)))⟩
      /-
        🎉 no goals
      -/


theorem find_eq_some_iff {p : Fin n → Prop} [DecidablePred p] {i : Fin n} :
    Fin.find p = some i ↔ p i ∧ ∀ j, p j → i ≤ j :=
  mem_find_iff


theorem mem_find_of_unique {p : Fin n → Prop} [DecidablePred p] (h : ∀ i j, p i → p j → i = j)
    {i : Fin n} (hi : p i) : i ∈ Fin.find p :=
  mem_find_iff.2 ⟨hi, fun j hj ↦ Fin.le_of_eq <| h i j hi hj⟩


/-- Sends `(g₀, ..., gₙ)` to `(g₀, ..., op gⱼ gⱼ₊₁, ..., gₙ)`. -/
def contractNth (j : Fin (n + 1)) (op : α → α → α) (g : Fin (n + 1) → α) (k : Fin n) : α :=
  if (k : ℕ) < j then g (Fin.castSucc k)
  else if (k : ℕ) = j then op (g (Fin.castSucc k)) (g k.succ) else g k.succ


theorem contractNth_apply_of_lt (j : Fin (n + 1)) (op : α → α → α) (g : Fin (n + 1) → α) (k : Fin n)
    (h : (k : ℕ) < j) : contractNth j op g k = g (Fin.castSucc k) :=
  if_pos h


theorem contractNth_apply_of_eq (j : Fin (n + 1)) (op : α → α → α) (g : Fin (n + 1) → α) (k : Fin n)
    (h : (k : ℕ) = j) : contractNth j op g k = op (g (Fin.castSucc k)) (g k.succ) := by
  /-
    n : Nat
    α : Sort u_1
    j : Fin (HAdd.hAdd n 1)
    op : α → α → α
    g : Fin (HAdd.hAdd n 1) → α
    k : Fin n
    h : Eq ↑k ↑j
    ⊢ Eq (j.contractNth op g k) (op (g k.castSucc) (g k.succ))
  -/
  have : ¬(k : ℕ) < j := not_lt.2 (le_of_eq h.symm)
  /-
    n : Nat
    α : Sort u_1
    j : Fin (HAdd.hAdd n 1)
    op : α → α → α
    g : Fin (HAdd.hAdd n 1) → α
    k : Fin n
    h : Eq ↑k ↑j
    this : Not (LT.lt ↑k ↑j)
    ⊢ Eq (j.contractNth op g k) (op (g k.castSucc) (g k.succ))
  -/
  rw [contractNth, if_neg this, if_pos h]
  /-
    🎉 no goals
  -/


theorem contractNth_apply_of_gt (j : Fin (n + 1)) (op : α → α → α) (g : Fin (n + 1) → α) (k : Fin n)
    (h : (j : ℕ) < k) : contractNth j op g k = g k.succ := by
  /-
    n : Nat
    α : Sort u_1
    j : Fin (HAdd.hAdd n 1)
    op : α → α → α
    g : Fin (HAdd.hAdd n 1) → α
    k : Fin n
    h : LT.lt ↑j ↑k
    ⊢ Eq (j.contractNth op g k) (g k.succ)
  -/
  rw [contractNth, if_neg (not_lt_of_gt h), if_neg (Ne.symm <| ne_of_lt h)]
  /-
    🎉 no goals
  -/


theorem contractNth_apply_of_ne (j : Fin (n + 1)) (op : α → α → α) (g : Fin (n + 1) → α) (k : Fin n)
    (hjk : (j : ℕ) ≠ k) : contractNth j op g k = g (j.succAbove k) := by
  /-
    n : Nat
    α : Sort u_1
    j : Fin (HAdd.hAdd n 1)
    op : α → α → α
    g : Fin (HAdd.hAdd n 1) → α
    k : Fin n
    hjk : Ne ↑j ↑k
    ⊢ Eq (j.contractNth op g k) (g (j.succAbove k))
  -/
  rcases lt_trichotomy (k : ℕ) j with (h | h | h)
    /-
      case inl
      n : Nat
      α : Sort u_1
      j : Fin (HAdd.hAdd n 1)
      op : α → α → α
      g : Fin (HAdd.hAdd n 1) → α
      k : Fin n
      hjk : Ne ↑j ↑k
      h : LT.lt ↑k ↑j
      ⊢ Eq (j.contractNth op g k) (g (j.succAbove k))
    -/
  · rwa [j.succAbove_of_castSucc_lt, contractNth_apply_of_lt]
      /-
        case inl.h
        n : Nat
        α : Sort u_1
        j : Fin (HAdd.hAdd n 1)
        op : α → α → α
        g : Fin (HAdd.hAdd n 1) → α
        k : Fin n
        hjk : Ne ↑j ↑k
        h : LT.lt ↑k ↑j
        ⊢ LT.lt k.castSucc j
      -/
    · rwa [Fin.lt_iff_val_lt_val]
      /-
        🎉 no goals
      -/
    /-
      case inr.inl
      n : Nat
      α : Sort u_1
      j : Fin (HAdd.hAdd n 1)
      op : α → α → α
      g : Fin (HAdd.hAdd n 1) → α
      k : Fin n
      hjk : Ne ↑j ↑k
      h : Eq ↑k ↑j
      ⊢ Eq (j.contractNth op g k) (g (j.succAbove k))
    -/
  · exact False.elim (hjk h.symm)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      n : Nat
      α : Sort u_1
      j : Fin (HAdd.hAdd n 1)
      op : α → α → α
      g : Fin (HAdd.hAdd n 1) → α
      k : Fin n
      hjk : Ne ↑j ↑k
      h : LT.lt ↑j ↑k
      ⊢ Eq (j.contractNth op g k) (g (j.succAbove k))
    -/
  · rwa [j.succAbove_of_le_castSucc, contractNth_apply_of_gt]
      /-
        case inr.inr.h
        n : Nat
        α : Sort u_1
        j : Fin (HAdd.hAdd n 1)
        op : α → α → α
        g : Fin (HAdd.hAdd n 1) → α
        k : Fin n
        hjk : Ne ↑j ↑k
        h : LT.lt ↑j ↑k
        ⊢ LE.le j k.castSucc
      -/
    · exact Fin.le_iff_val_le_val.2 (le_of_lt h)
      /-
        🎉 no goals
      -/


/-- To show two sigma pairs of tuples agree, it to show the second elements are related via
`Fin.cast`. -/
theorem sigma_eq_of_eq_comp_cast {α : Type*} :
    ∀ {a b : Σii, Fin ii → α} (h : a.fst = b.fst), a.snd = b.snd ∘ Fin.cast h → a = b
  | ⟨ai, a⟩, ⟨bi, b⟩, hi, h => by
    /-
      α : Type u_1
      ai : Nat
      a : Fin ai → α
      bi : Nat
      b : Fin bi → α
      hi : Eq ⟨ai, a⟩.fst ⟨bi, b⟩.fst
      h : Eq ⟨ai, a⟩.snd (Function.comp ⟨bi, b⟩.snd (Fin.cast hi))
      ⊢ Eq ⟨ai, a⟩ ⟨bi, b⟩
    -/
    dsimp only at hi
    /-
      α : Type u_1
      ai : Nat
      a : Fin ai → α
      bi : Nat
      b : Fin bi → α
      hi : Eq ai bi
      h : Eq ⟨ai, a⟩.snd (Function.comp ⟨bi, b⟩.snd (Fin.cast hi))
      ⊢ Eq ⟨ai, a⟩ ⟨bi, b⟩
    -/
    subst hi
    /-
      α : Type u_1
      ai : Nat
      a b : Fin ai → α
      h : Eq ⟨ai, a⟩.snd (Function.comp ⟨ai, b⟩.snd (Fin.cast ⋯))
      ⊢ Eq ⟨ai, a⟩ ⟨ai, b⟩
    -/
    simpa using h
    /-
      🎉 no goals
    -/


/-- `Fin.sigma_eq_of_eq_comp_cast` as an `iff`. -/
theorem sigma_eq_iff_eq_comp_cast {α : Type*} {a b : Σii, Fin ii → α} :
    a = b ↔ ∃ h : a.fst = b.fst, a.snd = b.snd ∘ Fin.cast h :=
  ⟨fun h ↦ h ▸ ⟨rfl, funext <| Fin.rec fun _ _ ↦ rfl⟩, fun ⟨_, h'⟩ ↦
    sigma_eq_of_eq_comp_cast _ h'⟩


/-- `Π i : Fin 2, α i` is equivalent to `α 0 × α 1`. See also `finTwoArrowEquiv` for a
non-dependent version and `prodEquivPiFinTwo` for a version with inputs `α β : Type u`. -/
@[simps (config := .asFn)]
def piFinTwoEquiv (α : Fin 2 → Type u) : (∀ i, α i) ≃ α 0 × α 1 where
  toFun f := (f 0, f 1)
  invFun p := Fin.cons p.1 <| Fin.cons p.2 finZeroElim
  left_inv _ := funext <| Fin.forall_fin_two.2 ⟨rfl, rfl⟩
  right_inv := fun _ => rfl

