/-- Append a single element to the end of a vector -/
def snoc : Vector α n → α → Vector α (n+1) :=
  fun xs x => append xs (x ::ᵥ Vector.nil)


@[simp]
theorem snoc_cons : (x ::ᵥ xs).snoc y = x ::ᵥ (xs.snoc y) :=
  rfl


@[simp]
theorem snoc_nil : (nil.snoc x) = x ::ᵥ nil :=
  rfl


@[simp]
theorem reverse_cons : reverse (x ::ᵥ xs) = (reverse xs).snoc x := by
  /-
    α : Type u_1
    n : Nat
    x : α
    xs : List.Vector α n
    ⊢ Eq (List.Vector.cons x xs).reverse (xs.reverse.snoc x)
  -/
  cases xs
  /-
    case mk
    α : Type u_1
    n : Nat
    x : α
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq (List.Vector.cons x ⟨val✝, property✝⟩).reverse ((List.Vector.reverse ⟨val …
  -/
  simp only [reverse, cons, toList_mk, List.reverse_cons, snoc]
  /-
    case mk
    α : Type u_1
    n : Nat
    x : α
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq ⟨HAppend.hAppend val✝.reverse (List.cons x List.nil), ⋯⟩ (List.Vector.app …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem reverse_snoc : reverse (xs.snoc x) = x ::ᵥ (reverse xs) := by
  /-
    α : Type u_1
    n : Nat
    x : α
    xs : List.Vector α n
    ⊢ Eq (xs.snoc x).reverse (List.Vector.cons x xs.reverse)
  -/
  cases xs
  /-
    case mk
    α : Type u_1
    n : Nat
    x : α
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq (List.Vector.snoc ⟨val✝, property✝⟩ x).reverse (List.Vector.cons x (List. …
  -/
  simp only [reverse, snoc, cons, toList_mk]
  /-
    case mk
    α : Type u_1
    n : Nat
    x : α
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq ⟨(List.Vector.append ⟨val✝, property✝⟩ ⟨List.cons x List.nil, ⋯⟩).toList. …
  -/
  congr
  /-
    case mk.e_val
    α : Type u_1
    n : Nat
    x : α
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq (List.Vector.append ⟨val✝, property✝⟩ ⟨List.cons x List.nil, ⋯⟩).toList.r …
  -/
  simp [toList, Vector.append, Append.append]
  /-
    🎉 no goals
  -/


theorem replicate_succ_to_snoc (val : α) :
    replicate (n+1) val = (replicate n val).snoc val := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [replicate_succ]
    conv => rhs; rw [replicate_succ]
    rw [snoc_cons, ih]


/-- Define `C v` by *reverse* induction on `v : Vector α n`.
    That is, break the vector down starting from the right-most element, using `snoc`

    This function has two arguments: `nil` handles the base case on `C nil`,
    and `snoc` defines the inductive step using `∀ x : α, C xs → C (xs.snoc x)`.

    This can be used as `induction v using Vector.revInductionOn`. -/
@[elab_as_elim]
def revInductionOn {C : ∀ {n : ℕ}, Vector α n → Sort*} {n : ℕ} (v : Vector α n)
    (nil : C nil)
    (snoc : ∀ {n : ℕ} (xs : Vector α n) (x : α), C xs → C (xs.snoc x)) :
    C v :=
           /-
             α : Type u_1
             β : Type u_2
             σ : Type u_3
             φ : Type u_4
             n✝ : Nat
             x : α
             s : σ
             xs : List.Vector α n✝
             C : {n : Nat} → List.Vector α n → Sort u_5
             n : Nat
             v : List.Vector α n
             nil : C List.Vector.nil
             snoc : {n : Nat} → (xs : List.Vector α n) → (x : α) → C xs → C (xs.snoc x)
             ⊢ Eq ((fun {n} v => C v.reverse) v.reverse) (C v)
           -/
  cast (by simp) <| inductionOn
           /-
             🎉 no goals
           -/
    (C := fun v => C v.reverse)
    v.reverse
    nil
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  σ : Type u_3
                                                  φ : Type u_4
                                                  n✝¹ : Nat
                                                  x✝ : α
                                                  s : σ
                                                  xs✝ : List.Vector α n✝¹
                                                  C : {n : Nat} → List.Vector α n → Sort u_5
                                                  n✝ : Nat
                                                  v : List.Vector α n✝
                                                  nil : C List.Vector.nil
                                                  snoc : {n : Nat} → (xs : List.Vector α n) → (x : α) → C xs → C (xs.snoc x)
                                                  n : Nat
                                                  x : α
                                                  xs : List.Vector α n
                                                  r : C xs.reverse
                                                  ⊢ Eq (C (xs.reverse.snoc x)) ((fun {n} v => C v.reverse) (List.Vector.cons x x …
                                                -/
    (@fun n x xs (r : C xs.reverse) => cast (by simp) <| snoc xs.reverse x r)
                                                /-
                                                  🎉 no goals
                                                -/


/-- Define `C v w` by *reverse* induction on a pair of vectors `v : Vector α n` and
    `w : Vector β n`. -/
@[elab_as_elim]
def revInductionOn₂ {C : ∀ {n : ℕ}, Vector α n → Vector β n → Sort*} {n : ℕ}
    (v : Vector α n) (w : Vector β n)
    (nil : C nil nil)
    (snoc : ∀ {n : ℕ} (xs : Vector α n) (ys : Vector β n) (x : α) (y : β),
      C xs ys → C (xs.snoc x) (ys.snoc y)) :
    C v w :=
           /-
             α : Type u_1
             β : Type u_2
             σ : Type u_3
             φ : Type u_4
             n✝ : Nat
             x : α
             s : σ
             xs : List.Vector α n✝
             C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_5
             n : Nat
             v : List.Vector α n
             w : List.Vector β n
             nil : C List.Vector.nil List.Vector.nil
             snoc : {n : Nat} → (xs : List.Vector α n) → (ys : List.Vector β n) → (x : α) → …
             ⊢ Eq ((fun {n} v w => C v.reverse w.reverse) v.reverse w.reverse) (C v w)
           -/
  cast (by simp) <| inductionOn₂
           /-
             🎉 no goals
           -/
    (C := fun v w => C v.reverse w.reverse)
    v.reverse
    w.reverse
    nil
    (@fun n x y xs ys (r : C xs.reverse ys.reverse) =>
               /-
                 α : Type u_1
                 β : Type u_2
                 σ : Type u_3
                 φ : Type u_4
                 n✝¹ : Nat
                 x✝ : α
                 s : σ
                 xs✝ : List.Vector α n✝¹
                 C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_5
                 n✝ : Nat
                 v : List.Vector α n✝
                 w : List.Vector β n✝
                 nil : C List.Vector.nil List.Vector.nil
                 snoc : {n : Nat} → (xs : List.Vector α n) → (ys : List.Vector β n) → (x : α) → …
                 n : Nat
                 x : α
                 y : β
                 xs : List.Vector α n
                 ys : List.Vector β n
                 r : C xs.reverse ys.reverse
                 ⊢ Eq (C (xs.reverse.snoc x) (ys.reverse.snoc y)) ((fun {n} v w => C v.reverse  …
               -/
      cast (by simp) <| snoc xs.reverse ys.reverse x y r)
               /-
                 🎉 no goals
               -/


/-- Define `C v` by *reverse* case analysis, i.e. by handling the cases `nil` and `xs.snoc x`
    separately -/
@[elab_as_elim]
def revCasesOn {C : ∀ {n : ℕ}, Vector α n → Sort*} {n : ℕ} (v : Vector α n)
    (nil : C nil)
    (snoc : ∀ {n : ℕ} (xs : Vector α n) (x : α), C (xs.snoc x)) :
    C v :=
  revInductionOn v nil fun xs x _ => snoc xs x


@[simp]
theorem map_snoc {f : α → β} : map f (xs.snoc x) = (map f xs).snoc (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    x : α
    xs : List.Vector α n
    f : α → β
    ⊢ Eq (List.Vector.map f (xs.snoc x)) ((List.Vector.map f xs).snoc (f x))
  -/
                   /-
                     🎉 no goals
                   -/
  induction xs <;> simp_all
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mapAccumr_nil {f : α → σ → σ × β} {s : σ} : mapAccumr f Vector.nil s = (s, Vector.nil) :=
  rfl


@[simp]
theorem mapAccumr_snoc {f : α → σ → σ × β} {s : σ} :
    mapAccumr f (xs.snoc x) s
    = let q := f x s
      let r := mapAccumr f xs q.1
      (r.1, r.2.snoc q.2) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    n : Nat
    x : α
    xs : List.Vector α n
    f : α → σ → Prod σ β
    s : σ
    ⊢ Eq (List.Vector.mapAccumr f (xs.snoc x) s)
        (let q := f x s;
        let r := List.Vector.mapAccumr f xs q.1;
        { fst := r.1, snd := r.2.snoc q.2 })
  -/
  induction xs
    /-
      case nil
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      n : Nat
      x : α
      xs : List.Vector α n
      f : α → σ → Prod σ β
      s : σ
      ⊢ Eq (List.Vector.mapAccumr f (List.Vector.nil.snoc x) s)
          (let q := f x s;
          let r := List.Vector.mapAccumr f List.Vector.nil q.1;
          { fst := r.1, snd := r.2.snoc q.2 })
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      n : Nat
      x : α
      xs : List.Vector α n
      f : α → σ → Prod σ β
      s : σ
      n✝ : Nat
      x✝ : α
      w✝ : List.Vector α n✝
      a✝ :
        Eq (List.Vector.mapAccumr f (w✝.snoc x) s)
          (let q := f x s;
          let r := List.Vector.mapAccumr f w✝ q.1;
          { fst := r.1, snd := r.2.snoc q.2 })
      ⊢ Eq (List.Vector.mapAccumr f ((List.Vector.cons x✝ w✝).snoc x) s)
          (let q := f x s;
          let r := List.Vector.mapAccumr f (List.Vector.cons x✝ w✝) q.1;
          { fst := r.1, snd := r.2.snoc q.2 })
    -/
  · simp [*]
    /-
      🎉 no goals
    -/


@[simp]
theorem map₂_snoc {f : α → β → σ} {y : β} :
    map₂ f (xs.snoc x) (ys.snoc y) = (map₂ f xs ys).snoc (f x y) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    n : Nat
    x : α
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ
    y : β
    ⊢ Eq (List.Vector.map₂ f (xs.snoc x) (ys.snoc y)) ((List.Vector.map₂ f xs ys). …
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  induction xs, ys using Vector.inductionOn₂ <;> simp_all
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem mapAccumr₂_nil {f : α → β → σ → σ × φ} :
    mapAccumr₂ f Vector.nil Vector.nil s = (s, Vector.nil) :=
  rfl


@[simp]
theorem mapAccumr₂_snoc (f : α → β → σ → σ × φ) (x : α) (y : β) :
    mapAccumr₂ f (xs.snoc x) (ys.snoc y) s
    = let q := f x y s
      let r := mapAccumr₂ f xs ys q.1
      (r.1, r.2.snoc q.2) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    φ : Type u_4
    n : Nat
    s : σ
    xs : List.Vector α n
    ys : List.Vector β n
    f : α → β → σ → Prod σ φ
    x : α
    y : β
    ⊢ Eq (List.Vector.mapAccumr₂ f (xs.snoc x) (ys.snoc y) s)
        (let q := f x y s;
        let r := List.Vector.mapAccumr₂ f xs ys q.1;
        { fst := r.1, snd := r.2.snoc q.2 })
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  induction xs, ys using Vector.inductionOn₂ <;> simp_all
                                                 /-
                                                   🎉 no goals
                                                 -/


